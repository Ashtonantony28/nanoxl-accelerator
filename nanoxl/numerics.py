"""Frozen numeric contracts (docs/contracts.md). RTL and golden model must match these bit for bit.

Every function is pure numpy integer arithmetic; no floats.
"""
from __future__ import annotations
import numpy as np

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def gemm_i8(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """C1: A int8 [M,K] @ W int8 [K,N] -> int32 [M,N], exact (K <= 4096 cannot overflow)."""
    assert a.dtype == np.int8 and w.dtype == np.int8, (a.dtype, w.dtype)
    assert a.shape[1] == w.shape[0] and a.shape[1] <= 4096
    return (a.astype(np.int64) @ w.astype(np.int64)).astype(np.int32)


def pack_i4(w: np.ndarray) -> np.ndarray:
    """C1b: int4 values (int8 array in [-8,7]) [K,N] with N even -> uint8 [K,N/2].
    Byte j of row k holds W[k,2j] in bits[3:0] and W[k,2j+1] in bits[7:4], two's complement."""
    assert w.dtype == np.int8 and w.shape[1] % 2 == 0
    assert w.min() >= -8 and w.max() <= 7
    u = (w.astype(np.int16) & 0xF).astype(np.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def unpack_i4(p: np.ndarray) -> np.ndarray:
    assert p.dtype == np.uint8
    lo = (p & 0xF).astype(np.int8)
    hi = (p >> 4).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo).astype(np.int8)
    hi = np.where(hi > 7, hi - 16, hi).astype(np.int8)
    out = np.empty((p.shape[0], p.shape[1] * 2), np.int8)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def srdhm(a, b):
    """C2a: gemmlowp SaturatingRoundingDoublingHighMul on int32 (vectorized)."""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    overflow = (a == INT32_MIN) & (b == INT32_MIN)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    s = ab + nudge
    # C-style truncating division by 2^31
    q = np.where(s >= 0, s >> 31, -((-s) >> 31))
    return np.where(overflow, INT32_MAX, q).astype(np.int32)


def rdivpot(x, exponent):
    """C2b: gemmlowp RoundingDivideByPOT (round half away from zero), exponent in [0,31]."""
    x = np.asarray(x, np.int64)
    e = np.asarray(exponent, np.int64)
    assert np.all((e >= 0) & (e <= 31))
    mask = (np.int64(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return ((x >> e) + (rem > thr)).astype(np.int32)


def requant(acc, mult, shift, bias=0, out_bits: int = 8):
    """C2: int32 acc (+int32 bias, wrapping add) -> intN.
    out = clamp(rdivpot(srdhm(acc + bias, mult), shift), -2^(N-1), 2^(N-1)-1).
    mult: int32 in [2^30, 2^31); shift in [0,31]; out_bits in {8, 16}. Per-channel args broadcast."""
    mult = np.asarray(mult, np.int64)
    assert np.all((mult >= (1 << 30)) & (mult <= INT32_MAX))
    x = (np.asarray(acc, np.int64) + np.asarray(bias, np.int64))
    x = ((x + (1 << 31)) % (1 << 32)) - (1 << 31)  # int32 wrap
    y = rdivpot(srdhm(x, mult), shift).astype(np.int64)
    lo, hi = -(1 << (out_bits - 1)), (1 << (out_bits - 1)) - 1
    return np.clip(y, lo, hi).astype(np.int8 if out_bits == 8 else np.int16)


def quantize_multiplier(real: float):
    """Float scale (0, 1) -> (mult, shift) for requant. Offline only (compiler side)."""
    assert 0.0 < real < 1.0, real
    shift = 0
    while real < 0.5:
        real *= 2.0
        shift += 1
    m = int(round(real * (1 << 31)))
    if m == (1 << 31):
        m //= 2
        shift -= 1
    assert shift <= 31
    return m, shift
