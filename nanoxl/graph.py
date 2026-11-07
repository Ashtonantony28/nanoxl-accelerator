"""Model description for the compiler: config presets, a tensor/op graph for a GPT-2 block stack,
compiler parameters from the real SG1 bundle (``from_bundle``) or synthetic ones, and an untiled
integer reference executor.

Every nonlinear op is the C5 integer algorithm of nanoxl.quant.ops (ADR-002): LayerNorm
(``ln_norm_i`` + ``ln_affine_i``), softmax (``softmax_i`` on int16 Q.8 scores from a C2 requant),
GELU (256-entry int8 LUT), INT16 residual adds with saturation, W4A8 group scales
(``acc = sum_g gm[g, n] * acc_g``). There are no float placeholders. For a real bundle the
reference is nanoxl.golden; ``reference_run`` covers hand-built graphs with synthetic parameters.

Compiler parameter format (``params[op.param]``):
  linear  w int8 [K, N] (INT4 values unpacked when wbits = 4), wbits, out_bits, bias/mult/shift int32 [N],
          gm uint8 [ceil(K/group), N] + group (W4A8 only)
  ln      g int16 [d], b int32 [d], c int64 [5] = mu_mult, mu_shift, sqrt_d, eps_q, sh
  attn    score_mult, score_shift, score_ls, pv_mult, pv_shift (int)
  gelu    lut int8 [256]
  params["sfu"] = {exp2, recip, rsqrt: uint16 [256]} (shared SFU tables)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import numerics as nx
from .quant import ops as qops

I64 = np.int64


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layer: int
    n_head: int
    d_model: int
    ctx: int
    vocab: int
    wbits: int = 8          # 8 (W8A8) or 4 (W4A8, group-64 uint8 scales x per-channel unit)
    residual_bits: int = 16

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model

    @property
    def block_params(self) -> int:
        return 12 * self.n_layer * self.d_model ** 2

    @property
    def weight_bytes(self) -> int:
        """Bytes of GEMM weights streamed per decode token (blocks + tied LM head, INT8 in W4A8 too),
        without the W4A8 group scales (1/32 of the INT4 block weights)."""
        return self.block_params * self.wbits // 8 + self.d_model * self.vocab

    def kv_bytes(self) -> int:
        return 2 * self.n_layer * self.ctx * self.d_model

    def with_(self, **kw) -> "ModelConfig":
        from dataclasses import replace
        return replace(self, **kw)


TINY = ModelConfig("tiny", n_layer=4, n_head=4, d_model=128, ctx=128, vocab=65)
SHAKESPEARE = ModelConfig("shakespeare", n_layer=6, n_head=6, d_model=384, ctx=256, vocab=65)
# Upper end of the v1 envelope (~25M params) used to size ISA fields.
M25 = ModelConfig("m25", n_layer=8, n_head=8, d_model=512, ctx=1024, vocab=4096)
PRESETS = {m.name: m for m in (TINY, SHAKESPEARE, M25)}


@dataclass
class Op:
    kind: str                 # ln | linear | attn | add | gelu
    ins: list
    outs: list
    param: str = ""
    attrs: dict = field(default_factory=dict)


@dataclass
class Graph:
    """tensors: name -> (bits, width). Every tensor is [M, width] with M = rows of the step."""
    tensors: dict
    ops: list
    inputs: list
    outputs: list
    cfg: ModelConfig | None = None


def build_gpt_graph(cfg: ModelConfig) -> Graph:
    """GPT-2 block stack, dataflow of contract C5. Op / parameter names are the bundle's (and the
    golden model's saturation-counter names): h{l}.ln_1, h{l}.c_attn, h{l}.score / h{l}.pv (attention),
    h{l}.attn_proj, h{l}.ln_2, h{l}.c_fc, h{l}.gelu, h{l}.mlp_proj, ln_f, lm_head; residual adds are
    "resid". Input x = int16 residual rows (host embedding), output int16 logits."""
    d = cfg.d_model
    rb = cfg.residual_bits
    t = {"x": (rb, d), "h": (8, d), "qkv": (8, 3 * d), "att": (8, d), "y": (rb, d),
         "ff": (8, cfg.d_ff), "logits": (16, cfg.vocab)}
    ops = []
    for l in range(cfg.n_layer):
        p = f"h{l}."
        ops += [
            Op("ln", ["x"], ["h"], p + "ln_1"),
            Op("linear", ["h"], ["qkv"], p + "c_attn"),
            Op("attn", ["qkv"], ["att"], p + "attn", {"layer": l, "score_name": p + "score", "pv_name": p + "pv"}),
            Op("linear", ["att"], ["y"], p + "attn_proj"),
            Op("add", ["x", "y"], ["x"], attrs={"name": "resid"}),
            Op("ln", ["x"], ["h"], p + "ln_2"),
            Op("linear", ["h"], ["ff"], p + "c_fc"),
            Op("gelu", ["ff"], ["ff"], p + "gelu"),
            Op("linear", ["ff"], ["y"], p + "mlp_proj"),
            Op("add", ["x", "y"], ["x"], attrs={"name": "resid"}),
        ]
    ops += [Op("ln", ["x"], ["h"], "ln_f"), Op("linear", ["h"], ["logits"], "lm_head")]
    return Graph(t, ops, ["x"], ["logits"], cfg)


# ---------------------------------------------------------------- real model (SG1 bundle, ADR-002 schema)

def from_bundle(arrays: dict, meta: dict, name: str = "bundle"):
    """(Graph, params) for an SG1 bundle (nanoxl.export.load_bundle). Only the spec configuration of
    C5 is accepted: INT16 residual, LUT GELU, LUT exp2."""
    q = meta["quant"]
    if q["res_bits"] != 16 or q["gelu"] != "lut" or q["exp"] != "lut":
        raise ValueError(f"bundle is a study variant (res_bits={q['res_bits']}, gelu={q['gelu']}, "
                         f"exp={q['exp']}); the hardware implements the C5 spec configuration only")
    A = arrays
    wbits = 4 if "h0.c_attn.w4" in A else 8
    cfg = ModelConfig(name, n_layer=meta["n_layer"], n_head=meta["n_head"], d_model=meta["n_embd"],
                      ctx=meta["block_size"], vocab=meta["vocab_size"], wbits=wbits,
                      residual_bits=q["res_bits"])
    g = build_gpt_graph(cfg)
    ff = A["h0.c_fc.bias"].shape[0]
    if ff != cfg.d_ff:
        g.tensors["ff"] = (8, ff)
    params = {"sfu": {"exp2": A["lut.exp2"], "recip": A["lut.recip"], "rsqrt": A["lut.rsqrt"]}}
    for op in g.ops:
        n = op.param
        if not n or n in params:
            continue
        if op.kind == "linear":
            out_bits = g.tensors[op.outs[0]][0]
            p = {"out_bits": out_bits, "bias": A[n + ".bias"], "mult": A[n + ".mult"], "shift": A[n + ".shift"]}
            if n + ".w" in A:
                p.update(w=A[n + ".w"], wbits=8)
            else:
                p.update(w=nx.unpack_i4(A[n + ".w4"]), wbits=4, gm=A[n + ".gm"], group=int(q["group"]))
            params[n] = p
        elif op.kind == "ln":
            params[n] = {"g": A[n + ".g"], "b": A[n + ".b"], "c": A[n + ".c"]}
        elif op.kind == "gelu":
            params[n] = {"lut": A[n + ".lut"]}
        elif op.kind == "attn":
            hp = f"h{op.attrs['layer']}."
            params[n] = {"score_mult": int(A[hp + "score.mult"][0]), "score_shift": int(A[hp + "score.shift"][0]),
                         "score_ls": int(A[hp + "score.ls"][0]), "pv_mult": int(A[hp + "pv.mult"][0]),
                         "pv_shift": int(A[hp + "pv.shift"][0])}
    return g, params


def load_bundle_graph(path, name=None):
    """(Graph, params, arrays, meta) for a bundle file."""
    from .export import load_bundle
    A, meta = load_bundle(path)
    g, p = from_bundle(A, meta, name or str(path))
    return g, p, A, meta


# ---------------------------------------------------------------- synthetic parameters

def _requant_params(rng, K, N, a_std, w_std, out_std, per_tensor=False):
    real = out_std / (math.sqrt(K) * a_std * w_std)
    n = 1 if per_tensor else N
    reals = np.clip(real * rng.uniform(0.7, 1.3, n), 1e-9, 0.999)
    ms = [nx.quantize_multiplier(float(r)) for r in reals]
    mult = np.array([m for m, _ in ms], np.int32)
    shift = np.array([s for _, s in ms], np.int32)
    return mult, shift


def make_linear_params(rng, K, N, wbits=8, out_bits=8, a_std=40.0, bias=True, group=64):
    p = {"wbits": wbits, "out_bits": out_bits}
    if wbits == 8:
        p["w"] = np.clip(np.rint(rng.normal(0, 30, (K, N))), -128, 127).astype(np.int8)
        w_std = 30.0
    else:
        p["w"] = rng.integers(-8, 8, (K, N), dtype=np.int8)
        p["gm"] = rng.integers(1, 256, (-(-K // group), N)).astype(np.uint8)
        p["group"] = group
        w_std = 4.6 * 150.0  # effective weight magnitude including the group multiplier
    out_std = {8: 30.0, 16: 1500.0, 32: None}[out_bits]
    if out_bits != 32:
        p["mult"], p["shift"] = _requant_params(rng, K, N, a_std, w_std, out_std)
        p["bias"] = (rng.integers(-2000, 2000, N) if bias else np.zeros(N)).astype(np.int32)
    return p


def make_gelu_lut(in_scale=1 / 32, out_scale=1 / 32) -> np.ndarray:
    """lut[b] for byte b = uint8(x), x int8 (C5 / ADR-002 construction)."""
    return qops.make_gelu_lut(in_scale, out_scale)


def make_ln_params(rng, d, in_scale, out_scale=1 / 32):
    """C5 LayerNorm constants for a synthetic gamma/beta (nanoxl.quant.ops.ln_consts / ln_affine_consts)."""
    c = qops.ln_consts(d, 1e-5, in_scale)
    g, b, sh = qops.ln_affine_consts(rng.normal(1, 0.1, d), rng.normal(0, 0.1, d), out_scale)
    return {"g": g, "b": b,
            "c": np.array([c["mu_mult"], c["mu_shift"], c["sqrt_d"], c["eps_q"], sh], np.int64)}


def sfu_tables():
    return {"exp2": qops.make_exp2_lut(), "recip": qops.make_recip_lut(), "rsqrt": qops.make_rsqrt_lut()}


def make_synthetic_params(graph: Graph, seed: int = 0) -> dict:
    """Random parameters for every op of ``graph`` (gpt or hand-built), in the from_bundle format."""
    rng = np.random.default_rng(seed)
    cfg = graph.cfg
    wbits = cfg.wbits if cfg else 8
    params = {"sfu": sfu_tables()}
    for op in graph.ops:
        if op.param in params or not op.param:
            continue
        if op.kind == "linear":
            K = graph.tensors[op.ins[0]][1]
            N = graph.tensors[op.outs[0]][1]
            ob = graph.tensors[op.outs[0]][0]
            a_std = 25.0 if graph.tensors[op.ins[0]][0] == 8 else 1500.0
            wb = 8 if op.param == "lm_head" else op.attrs.get("wbits", wbits)
            params[op.param] = make_linear_params(rng, K, N, wb, ob, a_std, group=op.attrs.get("group", 64))
        elif op.kind == "ln":
            d = graph.tensors[op.ins[0]][1]
            in_bits = graph.tensors[op.ins[0]][0]
            params[op.param] = make_ln_params(rng, d, 1 / 1500 if in_bits == 16 else 1 / 32)
        elif op.kind == "gelu":
            params[op.param] = {"lut": make_gelu_lut()}
        elif op.kind == "attn":
            dh = cfg.d_head
            # int32 q.k (q, k std ~30) -> int16 Q.8 log2 logits with std ~2 (x 256)
            sm, ss = nx.quantize_multiplier(min(0.999, 512.0 / (math.sqrt(dh) * 30.0 * 30.0)))
            pv_m, pv_s = nx.quantize_multiplier(30.0 / (127 * 30.0) * 1.5)
            params[op.param] = {"score_mult": sm, "score_shift": ss, "score_ls": 0,
                                "pv_mult": pv_m, "pv_shift": pv_s}
    return params


def embed(rng_or_tokens, positions, d, seed=0):
    """Synthetic host-side embedding: int16 residual rows for (token, position) pairs."""
    tokens = np.asarray(rng_or_tokens)
    r = np.random.default_rng(seed)
    wte = r.integers(-128, 128, (4096, d)).astype(np.int32)
    wpe = r.integers(-128, 128, (4096, d)).astype(np.int32)
    return np.clip(wte[tokens % 4096] * 8 + wpe[np.asarray(positions) % 4096] * 4, -32768, 32767).astype(np.int16)


def gelu_lut_apply(x, lut):
    return qops.gelu_lut_i(x, lut)


def sat(x, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi).astype({8: np.int8, 16: np.int16, 32: np.int32}[bits])


# ---------------------------------------------------------------- reference executor (untiled, C5 integer)

def requant_ref(acc, mult, shift, bias, out_bits, counter=None, name="requant", ls=0):
    """C2 requant of wrap32((acc << ls) + bias), counting clamps like the golden model."""
    acc = np.asarray(acc, I64) << ls
    if counter is not None:
        x = acc + np.asarray(bias, I64)
        x = ((x + (1 << 31)) % (1 << 32)) - (1 << 31)
        qops.sat(nx.rdivpot(nx.srdhm(x, mult), shift), out_bits, counter=counter, name=name)
    return nx.requant(acc, mult, shift, bias, out_bits)


def group_gemm(a, w, gm, group):
    """W4A8: acc[n] = sum_g gm[g, n] * gemm_i8(a[:, g], w[g, :]) (ADR-002), int64, ragged last group."""
    a = np.ascontiguousarray(a, np.int8)
    acc = np.zeros((a.shape[0], w.shape[1]), I64)
    gm = np.asarray(gm, I64)
    for g in range(-(-w.shape[0] // group)):
        sl = slice(g * group, (g + 1) * group)
        acc += gm[g] * nx.gemm_i8(np.ascontiguousarray(a[:, sl]), np.ascontiguousarray(w[sl])).astype(I64)
    return acc


def linear_ref(a, p, counter=None, name="linear"):
    if "gm" in p:
        acc = group_gemm(a, p["w"], p["gm"], p["group"])
        acc = ((acc + (1 << 31)) % (1 << 32)) - (1 << 31)
    else:
        acc = nx.gemm_i8(np.ascontiguousarray(a, np.int8), p["w"])
    if p["out_bits"] == 32:
        return acc.astype(np.int32)
    return requant_ref(acc, p["mult"], p["shift"], p["bias"], p["out_bits"], counter, name)


def ln_ref(x, p, rsqrt_lut, counter=None, name="ln"):
    c = p["c"]
    cd = dict(mu_mult=int(c[0]), mu_shift=int(c[1]), sqrt_d=int(c[2]), eps_q=int(c[3]))
    z = qops.ln_norm_i(x, cd, rsqrt_lut, counter=counter)
    return qops.ln_affine_i(z, p["g"], p["b"], int(c[4]), counter=counter, name=name)


def new_kv_state(cfg: ModelConfig, n_seq: int):
    return [np.zeros((cfg.n_layer, 2, cfg.n_head, cfg.ctx, cfg.d_head), np.int8) for _ in range(n_seq)]


def attention_ref(qkv, p, cfg, layer, seqs, kv, sfu, counter=None, names=("score", "pv")):
    d, H, dh = cfg.d_model, cfg.n_head, cfg.d_head
    out = np.zeros((qkv.shape[0], d), np.int8)
    r = 0
    for b, (pos, n) in enumerate(seqs):
        rows = qkv[r:r + n]
        for h in range(H):
            kv[b][layer, 0, h, pos:pos + n] = rows[:, d + h * dh: d + (h + 1) * dh]
            kv[b][layer, 1, h, pos:pos + n] = rows[:, 2 * d + h * dh: 2 * d + (h + 1) * dh]
        t = pos + n
        mask = np.arange(t)[None, :] <= (pos + np.arange(n))[:, None]
        for h in range(H):
            q = np.ascontiguousarray(rows[:, h * dh:(h + 1) * dh])
            K = kv[b][layer, 0, h, :t]
            V = kv[b][layer, 1, h, :t]
            acc = nx.gemm_i8(q, np.ascontiguousarray(K.T))
            s = requant_ref(acc, p["score_mult"], p["score_shift"], 0, 16, counter, names[0], p["score_ls"])
            P = qops.softmax_i(s, mask, sfu["exp2"], sfu["recip"])
            o = nx.gemm_i8(P, np.ascontiguousarray(V))
            out[r:r + n, h * dh:(h + 1) * dh] = requant_ref(o, p["pv_mult"], p["pv_shift"], 0, 8, counter, names[1])
        r += n
    return out


def reference_run(graph: Graph, params: dict, inputs: dict, seqs=None, kv=None, counter=None) -> dict:
    """Execute ``graph`` untiled with the C5 integer ops. ``seqs`` = [(pos, n_new)] (rows concatenated);
    ``kv`` is mutated in place (list of per-sequence caches from new_kv_state). ``counter`` collects
    saturation events under the golden model's names."""
    env = dict(inputs)
    sfu = params.get("sfu") or sfu_tables()
    for op in graph.ops:
        p = params.get(op.param)
        if op.kind == "linear":
            env[op.outs[0]] = linear_ref(env[op.ins[0]], p, counter, op.param)
        elif op.kind == "ln":
            env[op.outs[0]] = ln_ref(env[op.ins[0]], p, sfu["rsqrt"], counter, op.param)
        elif op.kind == "gelu":
            env[op.outs[0]] = gelu_lut_apply(env[op.ins[0]], p["lut"])
        elif op.kind == "add":
            bits = graph.tensors[op.outs[0]][0]
            r = env[op.ins[0]].astype(I64) + env[op.ins[1]].astype(I64)
            if counter is not None:
                qops.sat(r, bits, counter=counter, name=op.attrs.get("name", "resid"))
            env[op.outs[0]] = sat(r, bits)
        elif op.kind == "attn":
            names = (op.attrs.get("score_name", op.param + ".score"), op.attrs.get("pv_name", op.param + ".pv"))
            env[op.outs[0]] = attention_ref(env[op.ins[0]], p, graph.cfg, op.attrs["layer"], seqs, kv, sfu,
                                            counter, names)
        else:
            raise ValueError(op.kind)
    return {k: env[k] for k in graph.outputs}
