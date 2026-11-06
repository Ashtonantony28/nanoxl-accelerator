"""C5 integer nonlinear ops vs float: analytic bounds, gain/mean checks, forced saturation.

HELDOUT_SEED is frozen and was never used while choosing any constant, LUT or bound in
nanoxl/quant/ops.py; tuning used seeds 0..9 only (docs/quant-study.md)."""
import math

import numpy as np
import pytest

from nanoxl import numerics as nx
from nanoxl.quant import ops
from nanoxl.quant.fpref import layernorm_f

HELDOUT_SEED = 0x5EED2026
rng = np.random.default_rng(HELDOUT_SEED)

EXP = ops.make_exp2_lut()
RECIP = ops.make_recip_lut()
RSQRT = ops.make_rsqrt_lut()


def gain(y, ref):
    """Least-squares slope of y on ref through the origin."""
    ref = np.asarray(ref, np.float64)
    return float((np.asarray(y, np.float64) * ref).sum() / (ref * ref).sum())


# ------------------------------------------------------------------ helpers
def test_bitlen_exact():
    v = np.array([0, 1, 2, 3, 4, 255, 256, (1 << 45) - 1, 1 << 45, (1 << 62)], np.int64)
    ref = [int(x).bit_length() for x in v]
    assert ops.bitlen(v).tolist() == ref


def test_rshr_round_half_up():
    assert ops.rshr(np.array([5, -5, 6, -6, 7]), 1).tolist() == [3, -2, 3, -3, 4]
    assert ops.rshr(np.array([7]), 0).tolist() == [7]


# ------------------------------------------------------------------ exp2
def test_exp2_lut_contents_and_exhaustive_bound():
    assert EXP.dtype == np.uint16 and EXP[0] == 32768 and EXP[255] == round(32768 * 2 ** (-255 / 256))
    d = np.arange(0, 1 << 16)
    e = ops.exp2_neg_i(d, EXP)
    ref = 32768 * np.exp2(-d / 256.0)
    err = e - ref
    assert np.abs(err).max() < 1.5                      # LUT round (0.5) + floor shift (<1)
    assert np.abs(err[d < 256]).max() <= 0.5
    assert np.all(e[d >= 16 * 256] == 0)                # shift >= 16 flushes to 0
    assert np.all(np.diff(e) <= 0)                      # monotone non-increasing


# ------------------------------------------------------------------ reciprocal
def test_recip_lut_newton_relative_error():
    s = np.concatenate([np.arange(1 << 15, (1 << 15) + 4096), rng.integers(1 << 15, 1 << 23, 200_000),
                        [(1 << 23) - 1, 1 << 23, (1 << 16) - 1, 1 << 16]])
    y, b = ops.recip_i(s, RECIP)
    rel = y.astype(np.float64) * s / 2.0 ** (15 + b) - 1.0
    assert np.abs(rel).max() < 2 ** -14, np.abs(rel).max()
    assert abs(rel.mean()) < 2 ** -15


# ------------------------------------------------------------------ rsqrt
def test_rsqrt_relative_error():
    q = np.concatenate([np.arange(1, 5000), rng.integers(1, 1 << 46, 200_000),
                        np.array([1 << k for k in range(46)]), np.array([(1 << k) - 1 for k in range(1, 46)])])
    y, k = ops.rsqrt_i(q, RSQRT)
    rel = y.astype(np.float64) * 2.0 ** (-(22 + k)) * np.sqrt(q.astype(np.float64)) - 1.0
    assert np.abs(rel).max() < 2 ** -12, np.abs(rel).max()
    assert abs(rel.mean()) < 2 ** -13


# ------------------------------------------------------------------ softmax
def _ref_softmax(x, mask):
    xf = np.where(mask, x * math.log(2) / 256.0, -np.inf)
    xf = xf - xf.max(-1, keepdims=True)
    e = np.exp(xf)
    return 127.0 * e / e.sum(-1, keepdims=True)


@pytest.mark.parametrize("n,spread", [(1, 0), (7, 300), (128, 2000), (256, 1500), (256, 6000)])
def test_softmax_bound_gain_mean(n, spread):
    rows = 300
    x = rng.integers(-spread - 1, spread + 1, (rows, n)) if spread else np.zeros((rows, n), np.int64)
    mask = np.ones((rows, n), bool)
    p = ops.softmax_i(x, mask, EXP, RECIP).astype(np.int64)
    ref = _ref_softmax(x, mask)
    err = p - ref
    assert p.min() >= 0 and p.max() <= 127
    assert np.abs(err).max() <= 1.0, np.abs(err).max()     # 0.5 rounding + exp/recip error
    assert abs(err.mean()) < 0.05
    assert abs(gain(p, ref) - 1) < 0.01
    if n == 1:
        assert np.all(p == 127)


def test_softmax_causal_mask_and_flush():
    x = np.array([[0, 30000, -30000, 5]], np.int64)
    mask = np.array([[True, False, True, True]])
    p = ops.softmax_i(x, mask, EXP, RECIP)
    assert p[0, 1] == 0                      # masked never contributes even if largest
    assert p[0, 2] == 0                      # d >= 4096 flushes to 0 (forced underflow)
    assert p[0, 0] + p[0, 3] in (126, 127, 128)


def test_softmax_poly_alternative_is_worse_but_bounded():
    coef = ops.make_exp2_poly()
    d = np.arange(1 << 12)
    e = ops.exp2_neg_poly_i(d, coef)
    ref = 32768 * np.exp2(-d / 256.0)
    assert np.abs(e - ref).max() < 200      # ~0.5 % of full scale for a 2nd-order poly


# ------------------------------------------------------------------ LayerNorm
@pytest.mark.parametrize("d,amp", [(128, 3000), (384, 12000), (384, 200), (1536, 25000)])
def test_ln_norm_analytic_bound(d, amp):
    x = (rng.standard_normal((400, d)) * amp * rng.uniform(0.2, 1.0, (400, 1))
         + rng.integers(-amp, amp + 1, (400, 1))).round().clip(-32768, 32767).astype(np.int64)
    c = ops.ln_consts(d, 0.0, 1.0)
    c["eps_q"] = 0
    z = ops.ln_norm_i(x, c, RSQRT)
    xf = x.astype(np.float64)
    mu = xf.mean(-1, keepdims=True)
    sd = xf.std(-1, keepdims=True)
    ref = (xf - mu) / sd * 1024
    # mean is rounded to an integer (|dmu| <= 0.5 + srdhm slop), rsqrt rel err < 2^-12,
    # sqrt(d) constant rel err < 2^-11/sqrt(d), final rounding 0.5
    bound = 1024 * 0.51 / sd + (2 ** -12 + 2 ** -11) * np.abs(ref) + 1.0
    assert np.all(np.abs(z - ref) <= bound)
    assert abs(gain(z, ref) - 1) < 2e-3
    # mean-error check with the (deliberate, unbiased) integer-mean rounding removed: the
    # remaining rsqrt/rounding path must be unbiased
    mu_i = nx.rdivpot(nx.srdhm(x.sum(-1, keepdims=True), c["mu_mult"]), c["mu_shift"])
    ref_i = (xf - mu_i) / sd * 1024
    assert abs((z - ref_i).mean()) < 0.1


def test_ln_full_vs_float_lsb_and_gain():
    d = 384
    s_in, s_out = 2e-4, 0.03
    gamma = rng.uniform(0.5, 2.0, d)
    beta = rng.uniform(-0.3, 0.3, d)
    xf = rng.standard_normal((300, d)) * 0.8 + rng.standard_normal((300, 1)) * 0.2
    x = np.round(xf / s_in).astype(np.int64)
    c = ops.ln_consts(d, 1e-5, s_in)
    g, b, sh = ops.ln_affine_consts(gamma, beta, s_out)
    out = ops.ln_affine_i(ops.ln_norm_i(x, c, RSQRT), g, b, sh).astype(np.int64)
    ref = np.clip(layernorm_f(x * s_in, gamma, beta) / s_out, -128, 127)
    err = out - ref
    assert np.abs(err).max() <= 1.0
    assert abs(err.mean()) < 0.02
    assert abs(gain(out, ref) - 1) < 2e-3


def test_ln_saturation_forced_and_counted():
    d = 2048                                        # sqrt(d)*1024 > 32767: one spike saturates z
    x = np.zeros((1, d), np.int64)
    x[0, 0] = 20000
    cnt = {}
    z = ops.ln_norm_i(x, ops.ln_consts(d, 0.0, 1.0), RSQRT, counter=cnt)
    assert z[0, 0] == 32767 and cnt["ln_z"] == 1
    g, b, sh = ops.ln_affine_consts(np.ones(4), np.full(4, 10.0), 0.03)   # beta >> range
    out = ops.ln_affine_i(np.zeros((1, 4), np.int64), g, b, sh, counter=cnt, name="o")
    assert np.all(out == 127) and cnt["o"] == 4


def test_ln_constant_row_is_beta():
    d = 128
    x = np.full((2, d), 1234, np.int64)
    c = ops.ln_consts(d, 1e-5, 1e-3)
    z = ops.ln_norm_i(x, c, RSQRT)
    assert np.all(z == 0)


@pytest.mark.parametrize("d", [128, 384, 1536])
def test_ln_mean_rounding_bound(d):
    c = ops.ln_consts(d, 0.0, 1.0)
    s = np.concatenate([rng.integers(-(d << 15), d << 15, 100000), [0, 1, -1, d // 2, -(d // 2)]])
    mu = nx.rdivpot(nx.srdhm(s, c["mu_mult"]), c["mu_shift"])
    err = mu - s / d
    # final rounding 0.5 + srdhm rounding 0.5 * 2^-shift + multiplier error |S| * 2^-31
    assert np.abs(err).max() <= 0.5 + 0.5 * 2.0 ** -c["mu_shift"] + (d << 15) / d * 2.0 ** -30
    assert abs(err.mean()) < 0.01


# ------------------------------------------------------------------ GELU
def test_gelu_lut_exhaustive():
    s_in = 0.04
    s_out = float(ops.gelu_f(np.array([127 * s_in]))[0]) / 127   # calibrated-style output scale
    lut = ops.make_gelu_lut(s_in, s_out)
    x = np.arange(-128, 128)
    y = ops.gelu_lut_i(x, lut).astype(np.int64)
    ref = ops.gelu_f(x * s_in) / s_out
    assert np.abs(y - ref).max() <= 0.5 + 1e-9
    assert abs((y - ref).mean()) < 0.05
    assert y[x == -1][0] == lut[255] and y[x == 0][0] == 0
    assert abs(gain(y, ref) - 1) < 5e-3


def test_gelu_lut_saturates():
    lut = ops.make_gelu_lut(0.1, 0.01)            # gelu(12.7)/0.01 >> 127
    assert lut[127] == 127 and lut[100] == 127


def test_gelu_ibert_poly_error():
    x = np.linspace(-6, 6, 10001)
    assert np.abs(ops.gelu_ibert_f(x) - ops.gelu_f(x)).max() < 0.02
