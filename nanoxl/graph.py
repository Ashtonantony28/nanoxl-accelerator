"""Model description for the compiler: config presets, a tensor/op graph for a GPT-2 block stack,
synthetic quantized parameters, and an untiled integer reference executor.

SG1 owns the real quantized bundle and golden model; nothing here depends on it. The parameters
are synthetic (random int8/int4 weights, plausible requant multipliers) so the compiler, the
functional simulator and the cycle model can be developed and tested in isolation.

LayerNorm and softmax use *float placeholders* (``placeholder_layernorm``, ``placeholder_softmax``)
until ADR-002 freezes contract C5; SG5 replaces them. They are deterministic and row-wise, and the
reference executor and nanoxl.funcsim call the same functions, so compiled programs can still be
checked bit for bit end to end. GELU is already exact: an int8 -> int8 256-entry LUT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import numerics as nx

PLACEHOLDER_OPS = ("ln", "softmax")  # flagged: not the C5 integer algorithms yet


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layer: int
    n_head: int
    d_model: int
    ctx: int
    vocab: int
    wbits: int = 8          # 8 (W8A8) or 4 (W4A8, per-output-channel scales)
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
        """Bytes of GEMM weights streamed per decode token (blocks + tied LM head)."""
        return (self.block_params + self.d_model * self.vocab) * self.wbits // 8

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
    d = cfg.d_model
    t = {"x": (16, d), "h": (8, d), "qkv": (8, 3 * d), "att": (8, d), "y": (16, d),
         "ff": (8, cfg.d_ff), "logits": (32, cfg.vocab)}
    ops = []
    for l in range(cfg.n_layer):
        p = f"l{l}."
        ops += [
            Op("ln", ["x"], ["h"], p + "ln1"),
            Op("linear", ["h"], ["qkv"], p + "qkv"),
            Op("attn", ["qkv"], ["att"], p + "attn", {"layer": l}),
            Op("linear", ["att"], ["y"], p + "proj"),
            Op("add", ["x", "y"], ["x"]),
            Op("ln", ["x"], ["h"], p + "ln2"),
            Op("linear", ["h"], ["ff"], p + "fc"),
            Op("gelu", ["ff"], ["ff"], p + "gelu"),
            Op("linear", ["ff"], ["y"], p + "fc2"),
            Op("add", ["x", "y"], ["x"]),
        ]
    ops += [Op("ln", ["x"], ["h"], "lnf"), Op("linear", ["h"], ["logits"], "lm_head")]
    return Graph(t, ops, ["x"], ["logits"], cfg)


# ---------------------------------------------------------------- synthetic parameters

def _requant_params(rng, K, N, a_std, w_std, out_std, per_tensor=False):
    real = out_std / (math.sqrt(K) * a_std * w_std)
    n = 1 if per_tensor else N
    reals = np.clip(real * rng.uniform(0.7, 1.3, n), 1e-9, 0.999)
    ms = [nx.quantize_multiplier(float(r)) for r in reals]
    mult = np.array([m for m, _ in ms], np.int32)
    shift = np.array([s for _, s in ms], np.int32)
    return mult, shift


def make_linear_params(rng, K, N, wbits=8, out_bits=8, a_std=40.0, bias=True):
    if wbits == 8:
        w = np.clip(np.rint(rng.normal(0, 30, (K, N))), -128, 127).astype(np.int8)
        w_std = 30.0
    else:
        w = rng.integers(-8, 8, (K, N), dtype=np.int8)
        w_std = 4.6
    out_std = {8: 30.0, 16: 1500.0, 32: None}[out_bits]
    p = {"w": w, "wbits": wbits, "out_bits": out_bits}
    if out_bits != 32:
        p["mult"], p["shift"] = _requant_params(rng, K, N, a_std, w_std, out_std)
        p["bias"] = (rng.integers(-2000, 2000, N) if bias else np.zeros(N)).astype(np.int32)
    return p


def make_gelu_lut(in_scale=1 / 32, out_scale=1 / 32) -> np.ndarray:
    """lut[b] for byte b = uint8(x), x int8: round(gelu(x*in_scale)/out_scale) clipped to int8."""
    x = np.arange(256).astype(np.int64)
    x = np.where(x > 127, x - 256, x) * in_scale
    g = 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    return np.clip(np.rint(g / out_scale), -128, 127).astype(np.int8)


def make_ln_params(rng, d, in_scale, out_scale=1 / 32):
    return {"gamma": rng.normal(1, 0.1, d).astype(np.float32),
            "beta": rng.normal(0, 0.1, d).astype(np.float32),
            "in_scale": np.float32(in_scale), "out_scale": np.float32(out_scale)}


def make_synthetic_params(graph: Graph, seed: int = 0) -> dict:
    """Random parameters for every op of ``graph`` (gpt or hand-built)."""
    rng = np.random.default_rng(seed)
    cfg = graph.cfg
    wbits = cfg.wbits if cfg else 8
    params = {}
    for op in graph.ops:
        if op.param in params or not op.param:
            continue
        if op.kind == "linear":
            K = graph.tensors[op.ins[0]][1]
            N = graph.tensors[op.outs[0]][1]
            ob = graph.tensors[op.outs[0]][0]
            a_std = 25.0 if graph.tensors[op.ins[0]][0] == 8 else 1500.0
            wb = 8 if op.param == "lm_head" else op.attrs.get("wbits", wbits)
            params[op.param] = make_linear_params(rng, K, N, wb, ob, a_std)
        elif op.kind == "ln":
            d = graph.tensors[op.ins[0]][1]
            in_bits = graph.tensors[op.ins[0]][0]
            params[op.param] = make_ln_params(rng, d, 1 / 1500 if in_bits == 16 else 1 / 32)
        elif op.kind == "gelu":
            params[op.param] = {"lut": make_gelu_lut()}
        elif op.kind == "attn":
            dh = cfg.d_head
            pv_m, pv_s = nx.quantize_multiplier(30.0 / (127 * 30.0) * 1.5)
            params[op.param] = {"sm_scale": np.float32(2.0 / (math.sqrt(dh) * 30.0 * 30.0)),
                                "pv_mult": np.int32(pv_m), "pv_shift": np.int32(pv_s)}
    return params


def embed(rng_or_tokens, positions, d, seed=0):
    """Synthetic host-side embedding: int16 residual rows for (token, position) pairs."""
    tokens = np.asarray(rng_or_tokens)
    r = np.random.default_rng(seed)
    wte = r.integers(-128, 128, (4096, d)).astype(np.int32)
    wpe = r.integers(-128, 128, (4096, d)).astype(np.int32)
    return np.clip(wte[tokens % 4096] * 8 + wpe[np.asarray(positions) % 4096] * 4, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------- placeholders (flagged; SG5 replaces)

def placeholder_layernorm(x, gamma, beta, in_scale, out_scale, eps=1e-5):
    """FLOAT PLACEHOLDER for C5 LayerNorm. Row-wise; int in -> int8 out."""
    xf = np.ascontiguousarray(x, np.float64) * np.float64(in_scale)
    mu = xf.mean(axis=-1, keepdims=True)
    var = ((xf - mu) ** 2).mean(axis=-1, keepdims=True)
    y = (xf - mu) / np.sqrt(var + eps) * gamma.astype(np.float64) + beta.astype(np.float64)
    return np.clip(np.rint(y / np.float64(out_scale)), -128, 127).astype(np.int8)


def placeholder_softmax(scores, scale, diag, causal=True):
    """FLOAT PLACEHOLDER for C5 softmax. scores int32 [rows, cols]; row i may attend col j iff
    j <= i + diag (when causal). Output int8 probabilities round(p * 127)."""
    s = np.ascontiguousarray(scores, np.float64) * np.float64(scale)
    rows, cols = s.shape
    if causal:
        mask = np.arange(cols)[None, :] > (np.arange(rows)[:, None] + diag)
        s = np.where(mask, -np.inf, s)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(axis=-1, keepdims=True)
    return np.clip(np.rint(p * 127), 0, 127).astype(np.int8)


def gelu_lut_apply(x, lut):
    return lut[np.asarray(x, np.int8).view(np.uint8)]


def sat(x, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi).astype({8: np.int8, 16: np.int16, 32: np.int32}[bits])


# ---------------------------------------------------------------- reference executor (untiled)

def linear_ref(a, p):
    w = p["w"]
    acc = nx.gemm_i8(np.ascontiguousarray(a, np.int8), w)
    if p["out_bits"] == 32:
        return acc
    return nx.requant(acc, p["mult"], p["shift"], p["bias"], p["out_bits"])


def new_kv_state(cfg: ModelConfig, n_seq: int):
    return [np.zeros((cfg.n_layer, 2, cfg.n_head, cfg.ctx, cfg.d_head), np.int8) for _ in range(n_seq)]


def attention_ref(qkv, p, cfg, layer, seqs, kv):
    d, H, dh = cfg.d_model, cfg.n_head, cfg.d_head
    out = np.zeros((qkv.shape[0], d), np.int8)
    r = 0
    for b, (pos, n) in enumerate(seqs):
        rows = qkv[r:r + n]
        for h in range(H):
            kv[b][layer, 0, h, pos:pos + n] = rows[:, d + h * dh: d + (h + 1) * dh]
            kv[b][layer, 1, h, pos:pos + n] = rows[:, 2 * d + h * dh: 2 * d + (h + 1) * dh]
        t = pos + n
        for h in range(H):
            q = np.ascontiguousarray(rows[:, h * dh:(h + 1) * dh])
            K = kv[b][layer, 0, h, :t]
            V = kv[b][layer, 1, h, :t]
            s = nx.gemm_i8(q, np.ascontiguousarray(K.T))
            P = placeholder_softmax(s, p["sm_scale"], pos)
            o = nx.gemm_i8(P, V)
            out[r:r + n, h * dh:(h + 1) * dh] = nx.requant(o, p["pv_mult"], p["pv_shift"], 0, 8)
        r += n
    return out


def reference_run(graph: Graph, params: dict, inputs: dict, seqs=None, kv=None) -> dict:
    """Execute ``graph`` untiled with nanoxl.numerics. ``seqs`` = [(pos, n_new)] (rows concatenated);
    ``kv`` is mutated in place (list of per-sequence caches from new_kv_state)."""
    env = dict(inputs)
    for op in graph.ops:
        p = params.get(op.param)
        if op.kind == "linear":
            env[op.outs[0]] = linear_ref(env[op.ins[0]], p)
        elif op.kind == "ln":
            env[op.outs[0]] = placeholder_layernorm(env[op.ins[0]], p["gamma"], p["beta"],
                                                    p["in_scale"], p["out_scale"])
        elif op.kind == "gelu":
            env[op.outs[0]] = gelu_lut_apply(env[op.ins[0]], p["lut"])
        elif op.kind == "add":
            bits = graph.tensors[op.outs[0]][0]
            env[op.outs[0]] = sat(env[op.ins[0]].astype(np.int64) + env[op.ins[1]].astype(np.int64), bits)
        elif op.kind == "attn":
            env[op.outs[0]] = attention_ref(env[op.ins[0]], p, graph.cfg, op.attrs["layer"], seqs, kv)
        else:
            raise ValueError(op.kind)
    return {k: env[k] for k in graph.outputs}
