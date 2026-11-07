"""Integer nonlinear operators of contract C5 (docs/contracts.md, ADR-002).

Everything in the `*_i` functions is pure integer numpy (int64 internally) and is the
bit-exact specification the RTL implements. LUT *contents* are generated offline with
float math (`make_*_lut`) and shipped as data in the model bundle, so the RTL never
has to reproduce the float generation; it only loads the tables.

Shift conventions used throughout (all on two's-complement int64):
  `>>`        arithmetic shift right = floor division by 2^k
  rshr(x,k)   round-half-up shift:   (x + 2^(k-1)) >> k        (k >= 1), x for k == 0
  rdivpot     C2b (round half away from zero), from nanoxl.numerics
"""
from __future__ import annotations

import math

import numpy as np

from nanoxl import numerics as nx

I64 = np.int64

# ----------------------------------------------------------------- shared helpers

def rshr(x, k):
    """Round-half-up arithmetic right shift (x + 2^(k-1)) >> k; k may be an array, k >= 0."""
    x = np.asarray(x, I64)
    k = np.asarray(k, I64)
    half = np.where(k > 0, np.left_shift(I64(1), np.maximum(k - 1, 0)), 0)
    return (x + half) >> k


def sat(x, bits, signed=True, counter: dict | None = None, name: str = "sat"):
    """Clamp to a `bits`-wide integer range; optionally count saturation events."""
    x = np.asarray(x, I64)
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if counter is not None:
        counter[name] = counter.get(name, 0) + int(np.count_nonzero((x < lo) | (x > hi)))
    return np.clip(x, lo, hi)


def bitlen(x):
    """Number of bits of a non-negative int64 array (0 -> 0), exact (no float)."""
    x = np.asarray(x, I64).copy()
    n = np.zeros(x.shape, I64)
    for s in (32, 16, 8, 4, 2, 1):
        m = x >= (I64(1) << s)
        n = np.where(m, n + s, n)
        x = np.where(m, x >> s, x)
    return n + (x > 0)


# ----------------------------------------------------------------- exp2 (softmax)
SM_FRAC = 8          # softmax logits are int16 in log2 units, Q(.8): value = x * 2^-8 * ln 2 nats
EXP_ONE = 1 << 15    # exp LUT output scale: 2^-r/256 in Q.15 (LUT[0] = 32768, needs 16 bits unsigned)


def make_exp2_lut():
    """LUT_EXP2[r] = round(2^15 * 2^(-r/256)), r in [0,256). uint16."""
    r = np.arange(256)
    return np.floor(EXP_ONE * np.exp2(-r / 256.0) + 0.5).astype(np.uint16)


def exp2_neg_i(d, lut):
    """e = 2^(-d/256) in Q.15 for d >= 0 (int): LUT[d & 255] >> (d >> 8); 0 if d >> 8 >= 16."""
    d = np.asarray(d, I64)
    q = d >> 8
    r = d & 255
    e = np.asarray(lut, I64)[r] >> np.minimum(q, 16)
    return np.where(q >= 16, 0, e)


# shift-polynomial alternative (I-BERT style, study only): 2^(-f), f = r/256 in [0,1),
# approximated by a 2nd-order poly in r with integer coefficients, Q.15 output.
def make_exp2_poly():
    r = np.arange(256) / 256.0
    c = np.polyfit(r, np.exp2(-r), 2)              # c2 r^2 + c1 r + c0
    return tuple(int(round(v * (1 << 15))) for v in c)


def exp2_neg_poly_i(d, coef):
    c2, c1, c0 = coef
    d = np.asarray(d, I64)
    q = d >> 8
    r = d & 255
    e = ((c2 * r * r) >> 16) + ((c1 * r) >> 8) + c0
    e = np.where(r == 0, EXP_ONE, np.clip(e, 0, EXP_ONE)) >> np.minimum(q, 16)  # pin 2^0 exactly
    return np.where(q >= 16, 0, e)


# ----------------------------------------------------------------- reciprocal
RECIP_IDX_BITS = 8


def make_recip_lut():
    """LUT_RECIP[i] = round(2^31 / ((256 + i) * 128 + 64)), i in [0,256): ~2^31/m for the
    normalized m in [2^15, 2^16) whose top 9 bits are 1,i. uint16 (max 65408)."""
    i = np.arange(256)
    mid = (256 + i) * 128 + 64
    return np.floor((1 << 31) / mid + 0.5).astype(np.uint16)


def recip_i(s, lut):
    """Reciprocal of s (int, s >= 2^15) as (y, b): 1/s ~= y * 2^-(31 + b - 16).
    m = s >> (b-16) (truncate), b = bitlen(s) >= 16; y0 = LUT[(m >> 7) - 256];
    one Newton step y1 = (y0 * (2^32 - m*y0)) >> 31."""
    s = np.asarray(s, I64)
    assert np.all(s >= (1 << 15))
    b = bitlen(s)
    m = s >> (b - 16)
    y0 = np.asarray(lut, I64)[(m >> 7) - 256]
    y1 = (y0 * ((I64(1) << 32) - m * y0)) >> 31
    return y1, b


# ----------------------------------------------------------------- softmax
P_ONE = 127   # softmax probabilities are int8 in [0, 127], scale 1/127


def softmax_i(x, mask, exp_lut, recip_lut, exp_fn=None):
    """Integer softmax over the last axis.
    x: int16 logits in Q(.8) log2 units; mask: bool (True = keep), at least one True per row.
    Returns int8 p in [0,127] (p ~= 127 * softmax(x * 2^-8 * ln2)).
      m   = max_j x_j over kept j
      d_j = m - x_j                      (uint16, kept j only)
      e_j = exp2_neg_i(d_j)              (Q.15, <= 32768; masked j -> 0)
      S   = sum_j e_j                    (<= 2^23, >= 2^15)
      (y, b) = recip_i(S)
      p_j = min(127, (e_j * (127*y) + 2^(sh-1)) >> sh),  sh = 15 + b
    """
    x = np.asarray(x, I64)
    m = np.max(np.where(mask, x, -(1 << 20)), axis=-1, keepdims=True)
    d = np.where(mask, m - x, 0)
    e = exp2_neg_i(d, exp_lut) if exp_fn is None else exp_fn(d)
    e = np.where(mask, e, 0)
    s = e.sum(axis=-1, keepdims=True)
    y, b = recip_i(s, recip_lut)
    p = rshr(e * (P_ONE * y), 15 + b)
    return np.minimum(p, P_ONE).astype(np.int8)


# ----------------------------------------------------------------- rsqrt
RSQRT_FRAC = 14   # LUT / Newton output in Q2.14


def make_rsqrt_lut():
    """LUT_RSQRT[i] = round(2^14 / sqrt((i + 0.5) / 256)) for i in [64,256); entries < 64 unused (0).
    uint16 (max 32637)."""
    i = np.arange(256)
    v = np.where(i >= 64, np.floor((1 << 14) / np.sqrt((np.maximum(i, 64) + 0.5) / 256.0) + 0.5), 0)
    return v.astype(np.uint16)


def rsqrt_i(q, lut):
    """1/sqrt(q) for integer q >= 1 as (y, k): 1/sqrt(q) ~= y * 2^-(22 + k).
    e = bitlen(q) - 1; k = floor((e - 14) / 2) (may be negative);
    m = q >> 2k (k >= 0) or q << -2k (k < 0), m in [2^14, 2^16);
    y0 = LUT[m >> 8]  (Q2.14 of 1/sqrt(m / 2^16));
    Newton: t = (y0*y0) >> 14; u = (m*t) >> 16; y = (y0 * (3*2^14 - u)) >> 15."""
    q = np.asarray(q, I64)
    assert np.all(q >= 1)
    e = bitlen(q) - 1
    k = (e - 14) >> 1
    m = np.where(k >= 0, q >> np.maximum(2 * k, 0), q << np.maximum(-2 * k, 0))
    y0 = np.asarray(lut, I64)[m >> 8]
    t = (y0 * y0) >> 14
    u = (m * t) >> 16
    y = (y0 * (3 * (1 << 14) - u)) >> 15
    return y, k


# ----------------------------------------------------------------- LayerNorm
LN_Z = 10        # normalized value z = (x - mu)/sigma in Q(.10), int16
LN_SQRTD_FRAC = 10


def ln_consts(d: int, eps: float, s_in: float):
    """Per-LN integer constants derived from d, eps and the input scale s_in."""
    assert d >= 2
    mu_mult, mu_shift = nx.quantize_multiplier(1.0 / d)   # mu ~= round(S/d) via C2 primitives
    sqrt_d = int(round(math.sqrt(d) * (1 << LN_SQRTD_FRAC)))
    eps_q = int(round(d * eps / (s_in * s_in)))
    return dict(mu_mult=mu_mult, mu_shift=mu_shift, sqrt_d=sqrt_d, eps_q=eps_q)


def ln_norm_i(x, c, rsqrt_lut, trace: dict | None = None, counter=None):
    """Normalize stage. x int16 [..., d] -> z int16 = round((x-mu)/sigma * 2^10).
      S  = sum x                                  (int32)
      mu = rdivpot(srdhm(S, mu_mult), mu_shift)   (C2 primitives; mu ~= round(S/d))
      c  = x - mu                                 (int17)
      Q  = sum c^2 + eps_q                        (unsigned, <= 2^45: 48-bit accumulator)
      (y, k) = rsqrt_i(Q)
      R  = y * sqrt_d                             (sqrt_d = round(sqrt(d) * 2^10); R < 2^31)
      z  = sat16(rshr(c * R, 22 + k))             (product < 2^49)
    """
    x = np.asarray(x, I64)
    s = x.sum(-1, keepdims=True)
    mu = nx.rdivpot(nx.srdhm(s, c["mu_mult"]), c["mu_shift"]).astype(I64)
    cc = x - mu
    q = (cc * cc).sum(-1, keepdims=True) + c["eps_q"]
    q = np.maximum(q, 1)
    y, k = rsqrt_i(q, rsqrt_lut)
    r = y * c["sqrt_d"]
    z = sat(rshr(cc * r, 22 + k), 16, counter=counter, name="ln_z")
    if trace is not None:
        trace.update(mu=mu, q=q, y=y, k=k)
    return z


def ln_affine_i(z, g, b, sh, counter=None, name="ln_out"):
    """Affine stage. out = sat8(rdivpot(z * g + b, sh)); g int16, b int32 per channel."""
    acc = np.asarray(z, I64) * np.asarray(g, I64) + np.asarray(b, I64)
    return sat(nx.rdivpot(acc, sh), 8, counter=counter, name=name).astype(np.int8)


def ln_affine_consts(gamma, beta, s_out):
    """g = round(gamma/s_out * 2^(sh-10)) int16, b = round(beta/s_out * 2^sh) int32, sh as large
    as fits (max |g| <= 32767, sh <= 24)."""
    gs = np.asarray(gamma, np.float64) / s_out
    bs = np.asarray(beta, np.float64) / s_out
    gmax = max(np.max(np.abs(gs)), 1e-12)
    sh = int(min(24, math.floor(math.log2(32767.0 / gmax)) + LN_Z))
    sh = max(sh, 1)
    g = np.clip(np.round(gs * 2.0 ** (sh - LN_Z)), -32768, 32767).astype(np.int16)
    b = np.clip(np.round(bs * 2.0 ** sh), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
    return g, b, sh


# ----------------------------------------------------------------- GELU
def gelu_f(x):
    """Exact (erf) GELU, float64."""
    x = np.asarray(x, np.float64)
    if x.size <= 4096:
        return 0.5 * x * (1.0 + np.vectorize(math.erf, otypes=[np.float64])(x / math.sqrt(2.0)))
    import torch  # float reference only (vectorized erf); never used by integer ops
    return 0.5 * x * (1.0 + torch.erf(torch.from_numpy(x / math.sqrt(2.0))).numpy())


def make_gelu_lut(s_in: float, s_out: float):
    """LUT_GELU[u] for address u = x & 0xFF (x int8): sat8(round_half_away(gelu(x*s_in)/s_out)).
    Returned as int8[256] indexed by the unsigned byte address."""
    x = np.arange(256)
    x = np.where(x >= 128, x - 256, x)
    v = gelu_f(x * s_in) / s_out
    v = np.sign(v) * np.floor(np.abs(v) + 0.5)
    return np.clip(v, -128, 127).astype(np.int8)


def gelu_lut_i(x, lut):
    return np.asarray(lut)[np.asarray(x, I64) & 0xFF].astype(np.int8)


# I-BERT polynomial GELU (study alternative): gelu(x) = x/2 * (1 + L(x/sqrt2)),
# L(t) = sign(t) * (a * (min(|t|, -b) + b)^2 + 1), a = -0.2888, b = -1.769. Float-modelled
# on an int16 input (this is the accuracy ceiling of the poly; integer version would be lower).
def gelu_ibert_f(x):
    a, b = -0.2888, -1.769
    t = np.asarray(x, np.float64) / math.sqrt(2.0)
    L = np.sign(t) * (a * (np.minimum(np.abs(t), -b) + b) ** 2 + 1.0)
    return 0.5 * np.asarray(x, np.float64) * (1.0 + L)
