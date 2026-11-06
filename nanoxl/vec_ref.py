"""Reference model for the vector unit (rtl/vec_alu.sv, rtl/vec_reduce.sv, rtl/vec_unit.sv).

Semantics are specified in docs/vector-unit.md; this file is their executable form.
Requantization is *not* re-specified here: it is nanoxl.numerics.requant (contract C2).

All values are carried as int32 lanes (narrow operands are sign-extended on read).
All arithmetic is exact in int64 (every intermediate fits: |a*b| <= 2^62, a << 31 < 2^63).
"""
from __future__ import annotations
import numpy as np

from .numerics import INT32_MIN, INT32_MAX, requant

# ---- ALU opcodes (4 bits; vec_unit opcode space 0x00-0x0F) ----
OP_PASS, OP_ADD, OP_SUB, OP_MUL, OP_MAX, OP_MIN, OP_SRA, OP_SRAR, OP_SHL, OP_CLAMP = range(10)
ALU_OPS = {"pass": OP_PASS, "add": OP_ADD, "sub": OP_SUB, "mul": OP_MUL, "max": OP_MAX,
           "min": OP_MIN, "sra": OP_SRA, "srar": OP_SRAR, "shl": OP_SHL, "clamp": OP_CLAMP}
BINARY_OPS = (OP_ADD, OP_SUB, OP_MUL, OP_MAX, OP_MIN)

# ---- vec_unit opcodes (5 bits) ----
VOP_REQUANT = 0x10
VOP_REDUCE = 0x11
VOP_SFU_BASE = 0x18          # 0x18..0x1F -> SFU hook, sub-op = op & 7

# ---- output formats ----
FMT_I8, FMT_I16, FMT_I32 = 0, 1, 2
FMT_RANGE = {FMT_I8: (-128, 127), FMT_I16: (-32768, 32767), FMT_I32: (INT32_MIN, INT32_MAX),
             3: (INT32_MIN, INT32_MAX)}

REDUCE_ACC_W = 48            # sum / sum-of-squares accumulator width (exact, see docs)
REDUCE_SQ_W = 16             # sum-of-squares inputs must lie in int16


def rdivpot64(x, e):
    """RoundingDivideByPOT (C2b semantics) on int64 values, exponent 0..31, no narrowing."""
    x = np.asarray(x, np.int64)
    e = np.asarray(e, np.int64)
    assert np.all((e >= 0) & (e <= 31))
    mask = (np.int64(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> e) + (rem > thr)


def saturate(w, fmt):
    """Clamp int64 w to the range of fmt. Returns (value int64, saturated bool)."""
    lo, hi = FMT_RANGE[int(fmt)]
    w = np.asarray(w, np.int64)
    return np.clip(w, lo, hi), (w < lo) | (w > hi)


def alu(op, a, b=0, fmt=FMT_I32, shamt=0, lo=INT32_MIN, hi=INT32_MAX):
    """Elementwise vec_alu op on int32 lanes. Returns (result int32, sat flags bool).

    op: ALU opcode (0..15; 10..15 are reserved and behave as PASS).
    a, b: int32 arrays (b broadcast). shamt: 0..31 (scalar). lo/hi: int32 clamp bounds.
    """
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    assert np.all((a >= INT32_MIN) & (a <= INT32_MAX)) and np.all((b >= INT32_MIN) & (b <= INT32_MAX))
    assert 0 <= shamt <= 31
    a, b = np.broadcast_arrays(a, b)
    if op == OP_ADD:
        w = a + b
    elif op == OP_SUB:
        w = a - b
    elif op == OP_MUL:
        w = rdivpot64(a * b, shamt)
    elif op == OP_MAX:
        w = np.maximum(a, b)
    elif op == OP_MIN:
        w = np.minimum(a, b)
    elif op == OP_SRA:
        w = a >> shamt
    elif op == OP_SRAR:
        w = rdivpot64(a, shamt)
    elif op == OP_SHL:
        w = a << shamt                       # |a| <= 2^31, shamt <= 31: < 2^63, exact
    elif op == OP_CLAMP:
        w = np.minimum(np.maximum(a, lo), hi)  # lo > hi -> hi
    else:
        w = a.copy()                         # PASS and reserved opcodes
    r, s = saturate(w, fmt)
    return r.astype(np.int32), s


def reduce_row(x):
    """vec_reduce over one row (int32 values). Returns dict(sum, sumsq, max, min, sq_err).

    sum and sumsq are exact Python ints (the RTL keeps REDUCE_ACC_W=48 bits, which is exact
    for rows up to 2^16 int32 values and 2^17 int16 squares). sumsq squares the low
    16 bits of each value as int16; sq_err flags any value outside int16.
    """
    x = np.asarray(x, np.int64)
    assert x.size >= 1
    lo16 = ((x + 32768) % 65536) - 32768
    return {"sum": int(x.sum()), "sumsq": int((lo16 * lo16).sum()), "max": int(x.max()),
            "min": int(x.min()), "sq_err": bool(np.any(lo16 != x))}


def wrap_acc(v, w=REDUCE_ACC_W):
    """Interpret v modulo 2^w as signed (what the RTL accumulator holds)."""
    m = 1 << w
    return ((int(v) + (m >> 1)) % m) - (m >> 1)


def vec_unit_op(op, a, b=None, fmt=FMT_I32, shamt=0, lo=INT32_MIN, hi=INT32_MAX,
                bias=None, mult=None, shift=None):
    """One vec_unit command over a row. Returns int32 row (elementwise ops) or reduce dict.

    REQUANT: a=acc, b/bias = bias, mult/shift per element (or scalar); fmt 0 -> int8, else int16.
    Output is sign-extended into int32 lanes.
    """
    a = np.asarray(a, np.int64)
    if op == VOP_REQUANT:
        bias = b if bias is None else bias
        out_bits = 8 if fmt == FMT_I8 else 16
        return requant(a, mult, shift, 0 if bias is None else bias, out_bits).astype(np.int32)
    if op == VOP_REDUCE:
        return reduce_row(a)
    if op >= VOP_SFU_BASE:
        raise ValueError("SFU ops are modelled by SG5")
    alu_op = op & 0xF if op < 0x10 else OP_PASS   # 0x12..0x17 reserved -> PASS
    return alu(alu_op, a, 0 if b is None else b, fmt, shamt, lo, hi)[0]
