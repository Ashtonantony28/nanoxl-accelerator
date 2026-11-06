"""Python-level checks of the vector-unit reference (nanoxl/vec_ref.py) and of the exact
arithmetic formulation the RTL uses for requant (rtl/requant.sv, rtl/srdhm.sv), against
the frozen contract nanoxl.numerics.requant. No simulator; runs in a few seconds.
"""
from __future__ import annotations
import numpy as np
import pytest

from nanoxl import numerics as nx
from nanoxl import vec_ref as vr

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------- RTL-structured requant model
def _s32(v):
    v &= MASK32
    return v - (1 << 32) if v >> 31 else v


def _s64(v):
    v &= MASK64
    return v - (1 << 64) if v >> 63 else v


def rtl_requant_scalar(acc, bias, mult, shift, out16):
    """Bit-level transcription of the RTL datapath with Python ints and explicit widths."""
    x1 = _s32(acc + bias)                                       # S1 32-bit adder
    p = _s64(x1 * mult)                                         # S2 64-bit product
    ovf = x1 == nx.INT32_MIN and mult == nx.INT32_MIN
    s = _s64(p + ((1 << 30) if p >= 0 else -((1 << 30) - 1)))   # S4 nudge
    sadj = _s64(s + (((1 << 31) - 1) if s < 0 else 0))
    q = nx.INT32_MAX if ovf else _s32(sadj >> 31)
    mask = ((1 << shift) - 1) & MASK32                          # S5 rdivpot32
    rem = (q & MASK32) & mask
    thr = ((mask >> 1) + (1 if q < 0 else 0)) & MASK32
    y = _s32((q >> shift) + (1 if rem > thr else 0))
    lo, hi = (-32768, 32767) if out16 else (-128, 127)          # S6
    return min(max(y, lo), hi), (y < lo or y > hi)


def rtl_requant_np(acc, bias, mult, shift, out16):
    """Same formulation vectorized in int64/uint64 numpy (wrap-around arithmetic)."""
    acc = np.asarray(acc, np.int64); bias = np.asarray(bias, np.int64)
    mult = np.asarray(mult, np.int64); shift = np.asarray(shift, np.int64)
    x1 = ((acc + bias + (1 << 31)) & MASK32) - (1 << 31)
    p = x1 * mult                                               # |p| <= 2^62: exact in int64
    s = p + np.where(p >= 0, 1 << 30, -((1 << 30) - 1))
    sadj = s + np.where(s < 0, (1 << 31) - 1, 0)
    q = sadj >> 31
    q = np.where((x1 == nx.INT32_MIN) & (mult == nx.INT32_MIN), nx.INT32_MAX, q)
    mask = (np.int64(1) << shift) - 1
    rem = q & mask
    thr = (mask >> 1) + (q < 0)
    y = (q >> shift) + (rem > thr)
    lo = np.where(out16, -32768, -128); hi = np.where(out16, 32767, 127)
    return np.clip(y, lo, hi), (y < lo) | (y > hi)


def rand_requant_args(rng, n):
    acc = rng.integers(nx.INT32_MIN, nx.INT32_MAX + 1, n, dtype=np.int64)
    # Bring a third of the accumulators into a range that does not just saturate.
    small = rng.random(n) < 0.33
    acc[small] = rng.integers(-(1 << 20), 1 << 20, small.sum())
    bias = rng.integers(nx.INT32_MIN, nx.INT32_MAX + 1, n, dtype=np.int64)
    u = rng.random(n)
    bias[u < 0.4] = 0
    sm = (u >= 0.4) & (u < 0.7)
    bias[sm] = rng.integers(-(1 << 16), 1 << 16, sm.sum())
    mult = rng.integers(1 << 30, nx.INT32_MAX + 1, n, dtype=np.int64)
    shift = rng.integers(0, 32, n)
    out16 = rng.random(n) < 0.5
    return acc, bias, mult, shift, out16


def test_vec_requant_rtl_formulation_matches_contract_2M():
    rng = np.random.default_rng(1)
    for _ in range(4):
        acc, bias, mult, shift, out16 = rand_requant_args(rng, 500_000)
        got, _ = rtl_requant_np(acc, bias, mult, shift, out16)
        for bits, sel in ((8, ~out16), (16, out16)):
            ref = nx.requant(acc[sel], mult[sel], shift[sel], bias[sel], bits)
            np.testing.assert_array_equal(got[sel], ref)


def test_vec_requant_scalar_model_matches_vectorized():
    rng = np.random.default_rng(2)
    acc, bias, mult, shift, out16 = rand_requant_args(rng, 20_000)
    # include illegal mults (whole int32 domain, incl. INT32_MIN) to cover SRDHM overflow
    mult[:2000] = rng.integers(nx.INT32_MIN, nx.INT32_MAX + 1, 2000)
    mult[:10] = nx.INT32_MIN
    acc[:5] = nx.INT32_MIN; bias[:5] = 0
    v, s = rtl_requant_np(acc, bias, mult, shift, out16)
    for i in range(len(acc)):
        r = rtl_requant_scalar(int(acc[i]), int(bias[i]), int(mult[i]), int(shift[i]), bool(out16[i]))
        assert r == (int(v[i]), bool(s[i])), i


def test_vec_requant_directed_corners():
    vals = [nx.INT32_MIN, nx.INT32_MIN + 1, -1, 0, 1, nx.INT32_MAX - 1, nx.INT32_MAX, 1 << 30, -(1 << 30)]
    mults = [1 << 30, (1 << 30) + 1, nx.INT32_MAX, 1518500250]
    A, B, M, S, O = np.meshgrid(vals, vals, mults, np.arange(32), [False, True], indexing="ij")
    A, B, M, S, O = (t.ravel() for t in (A, B, M, S, O))
    got, _ = rtl_requant_np(A, B, M, S, O)
    for bits, sel in ((8, ~O), (16, O)):
        np.testing.assert_array_equal(got[sel], nx.requant(A[sel], M[sel], S[sel], B[sel], bits))


def test_vec_requant_rounding_halves_both_directions():
    # mult = 2^30 makes SRDHM an exact halving: odd x gives an exact half (rounds toward +inf).
    x = np.arange(-1001, 1002, 2, dtype=np.int64)
    np.testing.assert_array_equal(nx.srdhm(x, 1 << 30), (x + 1) // 2)
    # RoundingDivideByPOT: +-(k + 1/2) round away from zero.
    for e in range(1, 31):
        k = np.arange(0, 50, dtype=np.int64)
        pos = k * (1 << e) + (1 << (e - 1))
        np.testing.assert_array_equal(nx.rdivpot(pos, e), k + 1)
        np.testing.assert_array_equal(nx.rdivpot(-pos, e), -(k + 1))


# ---------------------------------------------------------------- ALU reference
def _alu_bigint(op, a, b, fmt, sh, lo, hi):
    def rdp(x, e):
        mask = (1 << e) - 1
        rem = x & mask
        thr = (mask >> 1) + (1 if x < 0 else 0)
        return (x >> e) + (1 if rem > thr else 0)
    w = {vr.OP_ADD: lambda: a + b, vr.OP_SUB: lambda: a - b, vr.OP_MUL: lambda: rdp(a * b, sh),
         vr.OP_MAX: lambda: max(a, b), vr.OP_MIN: lambda: min(a, b), vr.OP_SRA: lambda: a >> sh,
         vr.OP_SRAR: lambda: rdp(a, sh), vr.OP_SHL: lambda: a * (1 << sh),
         vr.OP_CLAMP: lambda: min(max(a, lo), hi)}.get(op, lambda: a)()
    rlo, rhi = vr.FMT_RANGE[fmt]
    return min(max(w, rlo), rhi), (w < rlo or w > rhi)


@pytest.mark.parametrize("op", list(range(16)))
def test_vec_alu_ref_vs_bigint(op):
    rng = np.random.default_rng(100 + op)
    n = 4000
    a = rng.integers(nx.INT32_MIN, nx.INT32_MAX + 1, n)
    b = rng.integers(nx.INT32_MIN, nx.INT32_MAX + 1, n)
    a[: n // 4] = rng.integers(-300, 300, n // 4)
    b[: n // 8] = rng.integers(-300, 300, n // 8)
    a[:4] = [nx.INT32_MIN, nx.INT32_MAX, nx.INT32_MIN, -1]
    b[:4] = [nx.INT32_MIN, nx.INT32_MAX, nx.INT32_MAX, nx.INT32_MIN]
    for fmt in range(4):
        for sh in (0, 1, 7, 15, 30, 31):
            lo, hi = sorted(int(v) for v in rng.integers(-1000, 1000, 2))
            r, s = vr.alu(op, a, b, fmt, sh, lo, hi)
            for i in range(0, n, 7):
                assert (int(r[i]), bool(s[i])) == _alu_bigint(op, int(a[i]), int(b[i]), fmt, sh, lo, hi)


def test_vec_alu_ref_mul_extremes_exact():
    r, s = vr.alu(vr.OP_MUL, [nx.INT32_MIN], [nx.INT32_MIN], vr.FMT_I32, 31)
    assert int(r[0]) == nx.INT32_MAX and s[0]           # 2^62 / 2^31 = 2^31 -> saturates
    r, s = vr.alu(vr.OP_MUL, [nx.INT32_MIN], [nx.INT32_MAX], vr.FMT_I32, 31)
    assert int(r[0]) == -(nx.INT32_MAX) and not s[0]     # -(2^31-1) exactly (no rounding)


def test_vec_reduce_ref_width_bound():
    row = np.full(384, -32768)
    r = vr.reduce_row(row)
    assert r["sumsq"] == 384 * (1 << 30) and r["sumsq"] < (1 << (vr.REDUCE_ACC_W - 1))
    assert not r["sq_err"]
    r = vr.reduce_row(np.array([70000, -1]))
    assert r["sq_err"] and r["sumsq"] == (70000 - 65536) ** 2 + 1
