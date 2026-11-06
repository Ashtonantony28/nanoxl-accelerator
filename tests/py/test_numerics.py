import numpy as np
from nanoxl import numerics as nx

rng = np.random.default_rng(0)


def test_gemm_exact():
    a = rng.integers(-128, 128, (7, 33), dtype=np.int8)
    w = rng.integers(-128, 128, (33, 5), dtype=np.int8)
    ref = a.astype(np.int64) @ w.astype(np.int64)
    assert (nx.gemm_i8(a, w) == ref).all()


def test_gemm_worst_case_no_overflow():
    a = np.full((1, 4096), -128, np.int8)
    w = np.full((4096, 1), -128, np.int8)
    assert nx.gemm_i8(a, w)[0, 0] == 128 * 128 * 4096


def test_i4_roundtrip():
    w = rng.integers(-8, 8, (9, 16), dtype=np.int8)
    p = nx.pack_i4(w)
    assert p.shape == (9, 8)
    assert (nx.unpack_i4(p) == w).all()
    assert nx.pack_i4(np.array([[-1, 1]], np.int8))[0, 0] == 0x1F


def test_srdhm_known_values():
    assert nx.srdhm(nx.INT32_MIN, nx.INT32_MIN) == nx.INT32_MAX
    assert nx.srdhm(1 << 30, 1 << 30) == 1 << 29
    assert nx.srdhm(-(1 << 30), 1 << 30) == -(1 << 29)
    # gemmlowp rounding: 1.5 -> 2, -1.5 -> -1 (half rounds toward +inf)
    assert nx.srdhm(3, 1 << 30) == 2
    assert nx.srdhm(-3, 1 << 30) == -1


def test_rdivpot_half_away_from_zero():
    assert nx.rdivpot(5, 1) == 3 and nx.rdivpot(-5, 1) == -3
    assert nx.rdivpot(4, 1) == 2 and nx.rdivpot(-4, 1) == -2
    assert nx.rdivpot(7, 0) == 7


def test_requant_matches_float_within_one_lsb():
    acc = rng.integers(-(1 << 20), 1 << 20, 10000).astype(np.int32)
    real = 0.0037
    m, s = nx.quantize_multiplier(real)
    out = nx.requant(acc, m, s)
    ref = np.clip(np.round(acc * real), -128, 127)
    assert np.abs(out.astype(np.int64) - ref).max() <= 1


def test_requant_saturates_and_int16():
    m, s = nx.quantize_multiplier(0.5)
    assert nx.requant(np.int32(10**6), m, s) == 127
    assert nx.requant(np.int32(-(10**6)), m, s) == -128
    assert nx.requant(np.int32(10**6), m, s, out_bits=16) == 32767
