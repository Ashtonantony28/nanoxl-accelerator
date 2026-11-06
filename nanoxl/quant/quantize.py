"""Calibration + quantization: FP32 nanoGPT weights -> integer model bundle (contract C5 / ADR-002).

The bundle is a flat dict {name: numpy array} plus `meta` (JSON-able dict). Schema in ADR-002.
All float math lives here (offline / compiler side); nanoxl.golden consumes only integers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np

from nanoxl import numerics as nx
from nanoxl.quant import ops
from nanoxl.quant.fpref import forward_f

LOG2E = 1.0 / math.log(2.0)


@dataclass
class QuantConfig:
    wbits: int = 8                 # 8: per-channel INT8; 4: INT4 group-wise along K (LM head stays INT8)
    group: int = 64                # INT4 group size along K
    res_bits: int = 16             # residual stream width (16 or 8)
    smooth_alpha: float = 0.0      # SmoothQuant migration strength on LN->c_attn / LN->c_fc (0 = off)
    calib: str = "max"             # activation range: "max" or "pct" (percentile of |x|)
    pct: float = 99.99
    res_margin: float = 1.25       # headroom multiplier on the residual absmax (int16 has room)
    gelu: str = "lut"              # "lut" (spec) | "ibert" (study only, float-modelled)
    exp: str = "lut"               # "lut" (spec) | "poly" (study: I-BERT-style shift+poly)
    int4_clip_search: bool = True  # MSE-optimal clipping per INT4 group
    calib_seqs: int = 16
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers
def qmul(real):
    """Float multiplier(s) in (0,1) -> (mult int32, shift int32) arrays (C2)."""
    real = np.atleast_1d(np.asarray(real, np.float64))
    assert np.all((real > 0) & (real < 1)), f"requant multiplier out of (0,1): {real.min()}..{real.max()}"
    ms = [nx.quantize_multiplier(float(r)) for r in real]
    return np.array([m for m, _ in ms], np.int32), np.array([s for _, s in ms], np.int32)


def amax(x, cfg: QuantConfig):
    a = np.abs(np.asarray(x, np.float64)).ravel()
    v = a.max() if cfg.calib == "max" else np.percentile(a, cfg.pct)
    return max(float(v), 1e-8)


def quant_w8(w):
    """w float [K,N] -> int8 per-output-channel symmetric, scale [N]."""
    s = np.maximum(np.abs(w).max(0), 1e-12) / 127.0
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s


def quant_w4(w, group, clip_search=True):
    """w float [K,N] -> (int4 values as int8 [K,N], gm uint8 [K/group, N], unit [N]).
    Effective weight = q * gm[g,n] * unit[n]; unit = max_g s_g / 255."""
    K, N = w.shape
    assert K % group == 0
    G = K // group
    wg = w.reshape(G, group, N)
    am = np.maximum(np.abs(wg).max(1), 1e-12)                  # [G,N]
    if clip_search:
        best_s = am / 7.0
        best_e = np.full(am.shape, np.inf)
        for r in np.linspace(0.6, 1.0, 17):
            s = am * r / 7.0
            q = np.clip(np.round(wg / s[:, None]), -8, 7)
            e = ((q * s[:, None] - wg) ** 2).sum(1)
            m = e < best_e
            best_e = np.where(m, e, best_e)
            best_s = np.where(m, s, best_s)
        sg = best_s
    else:
        sg = am / 7.0
    unit = sg.max(0) / 255.0                                    # [N]
    gm = np.clip(np.round(sg / unit), 1, 255).astype(np.uint8)  # [G,N]
    seff = gm.astype(np.float64) * unit                         # [G,N]
    q = np.clip(np.round(wg / seff[:, None]), -8, 7).astype(np.int8).reshape(K, N)
    return q, gm, unit


# ---------------------------------------------------------------- calibration
def calibrate(W, mcfg, calib_idx, qc: QuantConfig):
    taps = {}
    forward_f(W, mcfg, calib_idx, taps)
    return taps


def build_bundle(W: dict, mcfg, calib_idx: np.ndarray, qc: QuantConfig | None = None):
    """W: numpy state_dict (torch layout, float32). mcfg: GPTConfig. calib_idx int [B,T]
    (training-split tokens). Returns (arrays dict, meta dict)."""
    qc = qc or QuantConfig()
    W = {k: v.astype(np.float64) for k, v in W.items()}
    d, H, L, V, T = mcfg.n_embd, mcfg.n_head, mcfg.n_layer, mcfg.vocab_size, mcfg.block_size
    hs = d // H
    A: dict[str, np.ndarray] = {}

    # SmoothQuant: per-channel factors on LN outputs, folded into gamma/beta and weight rows.
    taps0 = calibrate(W, mcfg, calib_idx, qc)
    if qc.smooth_alpha > 0:
        a = qc.smooth_alpha
        for l in range(L):
            p = f"transformer.h.{l}."
            for ln, lin in (("ln_1", "attn.c_attn"), ("ln_2", "mlp.c_fc")):
                xmax = np.abs(taps0[f"{ln}.{l}"]).reshape(-1, d).max(0) + 1e-5
                wmax = np.abs(W[p + lin + ".weight"]).max(0) + 1e-5      # over out-ch: [in]
                sm = xmax ** a / wmax ** (1 - a)
                sm = sm / np.exp(np.mean(np.log(sm)))  # geometric-mean normalize (keeps LN scale)
                W[p + ln + ".weight"] = W[p + ln + ".weight"] / sm
                W[p + ln + ".bias"] = W[p + ln + ".bias"] / sm
                W[p + lin + ".weight"] = W[p + lin + ".weight"] * sm[None, :]
        taps = calibrate(W, mcfg, calib_idx, qc)
    else:
        taps = taps0

    rb = qc.res_bits
    rmax = (1 << (rb - 1)) - 1
    res_amax = max(amax(taps[f"resid.{l}.{w}"], qc) for l in range(L) for w in ("in", "mid"))
    res_amax = max(res_amax, amax(taps["resid.out"], qc))
    s_res = res_amax * (qc.res_margin if rb == 16 else 1.0) / rmax

    scales = {"res": s_res}
    A["emb.tok"] = np.clip(np.round(W["transformer.wte.weight"] / s_res), -rmax - 1, rmax).astype(
        np.int16 if rb == 16 else np.int8)
    A["emb.pos"] = np.clip(np.round(W["transformer.wpe.weight"] / s_res), -rmax - 1, rmax).astype(
        np.int16 if rb == 16 else np.int8)

    def linear(name, wf, bf, s_in, s_out, out_bits, int4):
        """wf [out,in] torch layout. Stores W [K,N] (C1 orientation) and requant params."""
        w = wf.T
        if int4:
            q, gm, unit = quant_w4(w, qc.group, qc.int4_clip_search)
            A[name + ".w4"] = nx.pack_i4(q)
            A[name + ".gm"] = gm
            sw = unit
        else:
            q, sw = quant_w8(w)
            A[name + ".w"] = q
        A[name + ".bias"] = np.round(bf / (s_in * sw)).astype(np.int32) if bf is not None \
            else np.zeros(w.shape[1], np.int32)
        A[name + ".mult"], A[name + ".shift"] = qmul(s_in * sw / s_out)
        return sw

    rmax_res = rmax
    for l in range(L):
        p = f"transformer.h.{l}."
        hp = f"h{l}."
        for ln, s_key in (("ln_1", "ln_1"), ("ln_2", "ln_2")):
            s_ln = amax(taps[f"{ln}.{l}"], qc) / 127.0
            scales[f"{hp}{ln}"] = s_ln
            c = ops.ln_consts(d, 1e-5, s_res)
            g, b, sh = ops.ln_affine_consts(W[p + ln + ".weight"], W[p + ln + ".bias"], s_ln)
            A[f"{hp}{ln}.g"], A[f"{hp}{ln}.b"] = g, b
            A[f"{hp}{ln}.c"] = np.array([c["mu_mult"], c["mu_shift"], c["sqrt_d"], c["eps_q"], sh], np.int64)
        s1 = scales[f"{hp}ln_1"]
        sq, sk, sv = (amax(taps[f"{n}.{l}"], qc) / 127.0 for n in ("q", "k", "v"))
        scales.update({f"{hp}q": sq, f"{hp}k": sk, f"{hp}v": sv})
        s_qkv = np.concatenate([np.full(d, sq), np.full(d, sk), np.full(d, sv)])
        w = W[p + "attn.c_attn.weight"].T
        int4 = qc.wbits == 4
        if int4:
            q4, gm, unit = quant_w4(w, qc.group, qc.int4_clip_search)
            A[hp + "c_attn.w4"], A[hp + "c_attn.gm"] = nx.pack_i4(q4), gm
            sw = unit
        else:
            q8, sw = quant_w8(w)
            A[hp + "c_attn.w"] = q8
        A[hp + "c_attn.bias"] = np.round(W[p + "attn.c_attn.bias"] / (s1 * sw)).astype(np.int32)
        A[hp + "c_attn.mult"], A[hp + "c_attn.shift"] = qmul(s1 * sw / s_qkv)
        # scores: int32 q.k  ->  int16 Q.8 log2 units (folds 1/sqrt(hs) and log2 e)
        real = sq * sk / math.sqrt(hs) * LOG2E * (1 << ops.SM_FRAC)
        ls = 0
        while real >= 1.0:
            real /= 2.0
            ls += 1
        A[hp + "score.mult"], A[hp + "score.shift"] = qmul(real)
        A[hp + "score.ls"] = np.array([ls], np.int32)
        # P (int8, 1/127) @ V (int8, s_v) -> y int8 (s_y <= s_v: y is a convex combination of v)
        sy = min(amax(taps[f"y.{l}"], qc) / 127.0, sv)
        scales[f"{hp}y"] = sy
        A[hp + "pv.mult"], A[hp + "pv.shift"] = qmul(sv / ops.P_ONE / sy)
        linear(hp + "attn_proj", W[p + "attn.c_proj.weight"], W[p + "attn.c_proj.bias"], sy, s_res, rb, int4)
        s2 = scales[f"{hp}ln_2"]
        if qc.gelu == "lut":
            sh_ = amax(taps[f"fc.{l}"], qc) / 127.0
            sg = amax(taps[f"gelu.{l}"], qc) / 127.0
            A[hp + "gelu.lut"] = ops.make_gelu_lut(sh_, sg)
            fc_bits = 8
        else:  # ibert study: c_fc out at int16
            sh_ = amax(taps[f"fc.{l}"], qc) / 32767.0
            sg = amax(taps[f"gelu.{l}"], qc) / 127.0
            fc_bits = 16
        scales[f"{hp}fc"], scales[f"{hp}gelu"] = sh_, sg
        linear(hp + "c_fc", W[p + "mlp.c_fc.weight"], W[p + "mlp.c_fc.bias"], s2, sh_, fc_bits, int4)
        linear(hp + "mlp_proj", W[p + "mlp.c_proj.weight"], W[p + "mlp.c_proj.bias"], sg, s_res, rb, int4)

    s_f = amax(taps["ln_f"], qc) / 127.0
    scales["ln_f"] = s_f
    c = ops.ln_consts(d, 1e-5, s_res)
    g, b, sh = ops.ln_affine_consts(W["transformer.ln_f.weight"], W["transformer.ln_f.bias"], s_f)
    A["ln_f.g"], A["ln_f.b"] = g, b
    A["ln_f.c"] = np.array([c["mu_mult"], c["mu_shift"], c["sqrt_d"], c["eps_q"], sh], np.int64)
    s_logit = amax(taps["logits"], qc) * 1.25 / 32767.0
    scales["logits"] = s_logit
    linear("lm_head", W["transformer.wte.weight"], None, s_f, s_logit, 16, False)

    A["lut.exp2"] = ops.make_exp2_lut()
    A["lut.recip"] = ops.make_recip_lut()
    A["lut.rsqrt"] = ops.make_rsqrt_lut()
    meta = dict(
        format="nanoxl-bundle", version=1,
        n_layer=L, n_head=H, n_embd=d, vocab_size=V, block_size=T,
        quant=asdict(qc), scales=scales,
        exp_poly=list(ops.make_exp2_poly()),
    )
    return A, meta
