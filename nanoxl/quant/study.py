"""M0 quantization study: perplexity FP32 vs integer variants, greedy-decode agreement,
per-op error tables, per-layer drift. Writes JSON consumed by docs/quant-study.md.

  python3 -m nanoxl.quant.study --ckpt artifacts/models/tiny/ckpt.pt --out artifacts/models/tiny/study.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanoxl import golden  # noqa: E402
from nanoxl.export import calib_windows  # noqa: E402
from nanoxl.quant import ops  # noqa: E402
from nanoxl.quant.fpref import forward_f, layernorm_f, softmax_f  # noqa: E402
from nanoxl.quant.quantize import QuantConfig, build_bundle  # noqa: E402

STUDY_SEED = 7     # tuning/study seed (the tests' HELDOUT_SEED is never used here)

VARIANTS = {
    "W8A8": dict(wbits=8),
    "W4A8": dict(wbits=4),
    "W8A8 res8": dict(wbits=8, res_bits=8),
    "W8A8 smooth0.5": dict(wbits=8, smooth_alpha=0.5),
    "W4A8 smooth0.5": dict(wbits=4, smooth_alpha=0.5),
    "W8A8 pct99.99": dict(wbits=8, calib="pct"),
    "W8A8 gelu-ibert": dict(wbits=8, gelu="ibert"),
    "W8A8 exp-poly": dict(wbits=8, exp="poly"),
    "W4A8 noclip": dict(wbits=4, int4_clip_search=False),
}


def ce_rows(logits, y):
    l = logits - logits.max(-1, keepdims=True)
    return -(l[np.arange(len(y)), y] - np.log(np.exp(l).sum(-1)))


# ------------------------------------------------------------------ workers
_G = {}


def _init(A, meta):
    golden.set_gemm(golden.gemm_exact_f64)
    import torch
    torch.set_num_threads(1)
    _G["g"] = golden.Golden(A, meta)


def _win(args):
    x, = args
    g = _G["g"]
    T = len(x) - 1
    g.sat_count.clear()
    li = g.forward(x[:T]).astype(np.float64) * g.meta["scales"]["logits"]
    return float(ce_rows(li, x[1:]).sum()), li.argmax(-1), dict(g.sat_count)


def _gen(args):
    prompt, n, T = args
    g = _G["g"]
    out = list(prompt)
    if len(prompt) + n <= T:
        return g.generate(prompt, n)
    for _ in range(n):                                   # sliding window (tiny: ctx 128)
        out.append(int(np.argmax(g.forward(out[-T:])[-1])))
    return out[len(prompt):]


def fp_greedy(model, prompt, n):
    import torch
    T = model.config.block_size
    out = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            lg, _ = model(torch.tensor([out[-T:]]))
            out.append(int(lg[0, -1].argmax()))
    return out[len(prompt):]


# ------------------------------------------------------------------ per-op tables
def stats(y, ref):
    y = np.asarray(y, np.float64).ravel()
    ref = np.asarray(ref, np.float64).ravel()
    e = y - ref
    return dict(max_abs=float(np.abs(e).max()), mean=float(e.mean()), rms=float(np.sqrt((e * e).mean())),
                gain=float((y * ref).sum() / max((ref * ref).sum(), 1e-30)), n=int(e.size))


def per_op_tables(A, meta, W, mcfg, x):
    """Isolated op errors (output LSB) on FP32 activations of windows x [B,T]."""
    taps = {}
    forward_f(W, mcfg, x, taps)
    sc = meta["scales"]
    d, H = mcfg.n_embd, mcfg.n_head
    rows = {}
    rs = ops.make_rsqrt_lut()
    ex, rc = ops.make_exp2_lut(), ops.make_recip_lut()

    def ln_case(name, inp, g_, b_, s_out):
        xq = np.clip(np.round(inp.reshape(-1, d) / sc["res"]), -32768, 32767).astype(np.int64)
        c = A[name + ".c"]
        cd = dict(mu_mult=int(c[0]), mu_shift=int(c[1]), sqrt_d=int(c[2]), eps_q=int(c[3]))
        out = ops.ln_affine_i(ops.ln_norm_i(xq, cd, rs), A[name + ".g"], A[name + ".b"], int(c[4]))
        ref = np.clip(layernorm_f(xq * sc["res"], g_, b_) / s_out, -128, 127)
        return stats(out, ref)

    for l in range(mcfg.n_layer):
        p, hp = f"transformer.h.{l}.", f"h{l}."
        if meta["quant"]["smooth_alpha"] == 0:
            rows[f"LN ln_1 L{l}"] = ln_case(hp + "ln_1", taps[f"resid.{l}.in"], W[p + "ln_1.weight"],
                                            W[p + "ln_1.bias"], sc[hp + "ln_1"])
            rows[f"LN ln_2 L{l}"] = ln_case(hp + "ln_2", taps[f"resid.{l}.mid"], W[p + "ln_2.weight"],
                                            W[p + "ln_2.bias"], sc[hp + "ln_2"])
        s = taps[f"scores.{l}"]                                  # [B,H,T,T], already /sqrt(hs)
        T = s.shape[-1]
        mask = np.tril(np.ones((T, T), bool))
        xs = np.clip(np.round(s / math.log(2) * 256), -32768, 32767).astype(np.int64).reshape(-1, T)
        mk = np.broadcast_to(mask, s.shape).reshape(-1, T)
        pi = ops.softmax_i(xs, mk, ex, rc)
        ref = 127 * softmax_f(np.where(mk, xs * math.log(2) / 256, -np.inf))
        rows[f"softmax L{l}"] = stats(pi[mk], ref[mk])
        sh_, sg = sc[hp + "fc"], sc[hp + "gelu"]
        if hp + "gelu.lut" in A:
            u = taps[f"fc.{l}"]
            uq = np.clip(np.round(u / sh_), -128, 127).astype(np.int64)
            gi = ops.gelu_lut_i(uq, A[hp + "gelu.lut"])
            rows[f"GELU-LUT (op only) L{l}"] = stats(gi, np.clip(ops.gelu_f(uq * sh_) / sg, -128, 127))
            rows[f"GELU-LUT (incl. input quant) L{l}"] = stats(gi, np.clip(taps[f"gelu.{l}"] / sg, -128, 127))
    rf = taps["resid.out"]
    if meta["quant"]["smooth_alpha"] == 0:
        rows["LN ln_f"] = ln_case("ln_f", rf, W["transformer.ln_f.weight"], W["transformer.ln_f.bias"], sc["ln_f"])
    return rows


def drift(A, meta, W, mcfg, x):
    """Per-layer relative L2 error of the integer residual stream vs FP32 (one window)."""
    taps = {}
    forward_f(W, mcfg, x[None], taps)
    tr = {}
    g = golden.Golden(A, meta)
    g.forward(x, trace=tr)
    out = {}
    for k in [f"resid.{l}.{w}" for l in range(mcfg.n_layer) for w in ("in", "mid")] + ["resid.out"]:
        fi = tr[k].astype(np.float64) * meta["scales"]["res"]
        ff = taps[k][0]
        out[k] = float(np.linalg.norm(fi - ff) / np.linalg.norm(ff))
    li = tr["logits"].astype(np.float64) * meta["scales"]["logits"]
    out["logits"] = float(np.linalg.norm(li - taps["logits"][0]) / np.linalg.norm(taps["logits"][0]))
    return out


def synthetic_op_tables():
    """Exhaustive / dense-random checks of the scalar SFUs (also see tests)."""
    r = np.random.default_rng(STUDY_SEED)
    d = np.arange(1 << 16)
    e = ops.exp2_neg_i(d, ops.make_exp2_lut())
    ep = ops.exp2_neg_poly_i(d, ops.make_exp2_poly())
    ref = 32768 * np.exp2(-d / 256)
    s = r.integers(1 << 15, 1 << 23, 100000)
    y, b = ops.recip_i(s, ops.make_recip_lut())
    q = r.integers(1, 1 << 45, 100000)
    yr, k = ops.rsqrt_i(q, ops.make_rsqrt_lut())
    return {
        "exp2 LUT (Q.15 LSB, all 65536 inputs)": stats(e, ref),
        "exp2 shift+poly2 (Q.15 LSB)": stats(ep, ref),
        "recip LUT+Newton (relative)": stats(y * s / 2.0 ** (15 + b), np.ones(len(s))),
        "rsqrt LUT+Newton (relative)": stats(yr * 2.0 ** (-(22 + k)) * np.sqrt(q.astype(float)), np.ones(len(q))),
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=200, help="val windows for integer PPL")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--render", action="store_true", help="print markdown for an existing --out JSON")
    args = ap.parse_args()
    if args.render:
        print(render_md(args.out))
        return

    import torch
    torch.set_num_threads(args.procs)
    from nanoxl.model_fp import eval_loss, load_checkpoint, load_shakespeare, numpy_weights
    model = load_checkpoint(args.ckpt)
    mcfg = model.config
    T = mcfg.block_size
    train, val, chars = load_shakespeare()
    W = numpy_weights(model)
    cal = calib_windows(train, T, QuantConfig().calib_seqs)
    res = dict(ckpt=args.ckpt, config=dict(n_layer=mcfg.n_layer, n_head=mcfg.n_head, n_embd=mcfg.n_embd,
                                            block_size=T), started=time.ctime())
    t0 = time.time()
    res["fp32_val_loss_full"] = eval_loss(model, val)
    nwin_all = (len(val) - 1) // T
    widx = np.linspace(0, nwin_all - 1, min(args.windows, nwin_all)).astype(int)
    wins = [val[i * T:i * T + T + 1].astype(np.int64) for i in widx]
    with torch.no_grad():
        xs = torch.from_numpy(np.stack([w[:T] for w in wins]))
        lf = torch.cat([model(xs[i:i + 16])[0] for i in range(0, len(xs), 16)]).double().numpy()
    fp_ce = float(np.mean([ce_rows(lf[i], wins[i][1:]).mean() for i in range(len(wins))]))
    fp_arg = lf.argmax(-1)
    res["eval_windows"] = len(wins)
    res["eval_tokens"] = len(wins) * T
    res["fp32_val_loss_subset"] = fp_ce
    res["fp32_ppl_subset"] = math.exp(fp_ce)
    print(f"FP32 full val loss {res['fp32_val_loss_full']:.4f}; subset {fp_ce:.4f} ({len(wins)} windows)", flush=True)

    starts = [len(val) // 7 * (i + 1) for i in range(args.prompts)]
    prompts = [[int(t) for t in val[s:s + 16]] for s in starts]
    fp_gen = [fp_greedy(model, p, args.gen) for p in prompts]
    res["prompts"] = ["".join(chars[t] for t in p) for p in prompts]
    res["fp32_gen"] = ["".join(chars[t] for t in g) for g in fp_gen]

    res["variants"] = {}
    for name in args.variants.split(","):
        qc = QuantConfig(**VARIANTS[name])
        tq = time.time()
        A, meta = build_bundle(W, mcfg, cal, qc)
        with ProcessPoolExecutor(args.procs, initializer=_init, initargs=(A, meta)) as ex:
            outs = list(ex.map(_win, [(w,) for w in wins]))
            gens = list(ex.map(_gen, [(p, args.gen, T) for p in prompts])) if name in ("W8A8", "W4A8") else None
        ce = sum(o[0] for o in outs) / (len(wins) * T)
        top1 = float(np.mean([np.mean(o[1] == fp_arg[i]) for i, o in enumerate(outs)]))
        v = dict(val_loss=ce, ppl=math.exp(ce), ppl_delta_pct=100 * (math.exp(ce - fp_ce) - 1),
                 top1_agree_teacher_forced=top1, sat_events=int(sum(sum(o[2].values()) for o in outs)),
                 sat_by_op={k: int(sum(o[2].get(k, 0) for o in outs)) for k in outs[0][2]
                            if sum(o[2].get(k, 0) for o in outs)},
                 seconds=time.time() - tq)
        if gens is not None:
            agree = []
            for gi, gf in zip(gens, fp_gen):
                m = np.array(gi) == np.array(gf)
                first = int(np.argmin(m)) if not m.all() else len(m)
                agree.append(dict(first_divergence=first, match_frac=float(m.mean())))
            v["greedy_agreement"] = agree
            v["gen"] = ["".join(chars[t] for t in g) for g in gens]
        if name in ("W8A8", "W4A8", "W8A8 smooth0.5"):
            v["per_op"] = per_op_tables(A, meta, W, mcfg, np.stack([w[:T] for w in wins[:2]]))
            v["drift"] = drift(A, meta, W, mcfg, wins[0][:T])
        res["variants"][name] = v
        print(f"{name}: loss {ce:.4f} ppl {math.exp(ce):.3f} (+{v['ppl_delta_pct']:.2f}%) top1 {top1:.4f} "
              f"sat {v['sat_events']} {v['seconds']:.0f}s", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    res["synthetic_ops"] = synthetic_op_tables()
    res["seconds"] = time.time() - t0
    Path(args.out).write_text(json.dumps(res, indent=1))



def render_md(path):
    """Markdown tables from a study JSON (pasted into docs/quant-study.md)."""
    r = json.loads(Path(path).read_text())
    out = [f"FP32 val loss (full val split): {r['fp32_val_loss_full']:.4f} (PPL {math.exp(r['fp32_val_loss_full']):.3f}). "
           f"Integer eval subset: {r['eval_windows']} windows × {r['config']['block_size']} = {r['eval_tokens']} tokens, "
           f"FP32 on the same subset: loss {r['fp32_val_loss_subset']:.4f}, PPL {r['fp32_ppl_subset']:.3f}.", "",
           "| Variant | val loss | PPL | ΔPPL vs FP32 | top-1 agree (teacher-forced) | sat events |",
           "|---|---|---|---|---|---|"]
    for n, v in r["variants"].items():
        out.append(f"| {n} | {v['val_loss']:.4f} | {v['ppl']:.3f} | {v['ppl_delta_pct']:+.2f} % | "
                   f"{v['top1_agree_teacher_forced']:.4f} | {v['sat_events']} |")
    out += ["", "Saturation events by op (W8A8): " + json.dumps(r["variants"].get("W8A8", {}).get("sat_by_op", {})), ""]
    out += ["Greedy decode, 200 new tokens from 16-char val prompts (first divergence index / fraction of positions equal):", "",
            "| Prompt | W8A8 | W4A8 |", "|---|---|---|"]
    for i, p in enumerate(r["prompts"]):
        cells = []
        for n in ("W8A8", "W4A8"):
            a = r["variants"].get(n, {}).get("greedy_agreement")
            cells.append(f"{a[i]['first_divergence']} / {a[i]['match_frac']:.2f}" if a else "-")
        out.append(f"| `{p!r}` | {cells[0]} | {cells[1]} |")
    v = r["variants"].get("W8A8", {})
    if "per_op" in v:
        out += ["", "Per-op isolated error, W8A8 (output LSB; FP32 activations quantized at the op input):", "",
                "| Op | max abs | mean | rms | gain | n |", "|---|---|---|---|---|---|"]
        for k, s in v["per_op"].items():
            out.append(f"| {k} | {s['max_abs']:.3f} | {s['mean']:+.4f} | {s['rms']:.3f} | {s['gain']:.4f} | {s['n']} |")
    out += ["", "Synthetic SFU checks:", "", "| Op | max abs | mean | rms | gain |", "|---|---|---|---|---|"]
    for k, s in r.get("synthetic_ops", {}).items():
        out.append(f"| {k} | {s['max_abs']:.3g} | {s['mean']:+.3g} | {s['rms']:.3g} | {s['gain']:.6f} |")
    out += ["", "Residual-stream drift (relative L2 vs FP32, one window):", "", "| Tensor | W8A8 | W4A8 |", "|---|---|---|"]
    for k in r["variants"].get("W8A8", {}).get("drift", {}):
        out.append(f"| {k} | {r['variants']['W8A8']['drift'][k]:.4f} | {r['variants'].get('W4A8', {}).get('drift', {}).get(k, float('nan')):.4f} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
