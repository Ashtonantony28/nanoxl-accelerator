"""Shared helpers for the SFU benches (rtl/sfu_*.sv, rtl/softmax.sv, rtl/layernorm.sv).

The bit-exact references are nanoxl.quant.ops (contract C5); this file only packs and
unpacks lanes and loads the bundle tables (ADR-002 schema, nanoxl.export.load_bundle).
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl.export import load_bundle
from nanoxl.quant import ops
from vec_common_cocotb import Sink, Source, env_int, pack, run_bench, start, u, unpack  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / os.environ.get("SFU_BUNDLE", "artifacts/models/tiny/w8a8.npz")
I64 = np.int64

# sub-op codes on the vec_unit SFU hook (cmd_op = 0x18 + sub-op), docs/sfu.md
OP_EXP2, OP_RECIP, OP_RSQRT, OP_GELU, OP_SOFTMAX, OP_LN = range(6)

# sfu.sv configuration address map: addr = region << 24 | slot << 16 | index
R_EXP2, R_RECIP, R_RSQRT, R_GELU, R_LN_GB, R_LN_C = range(6)


def cfg_addr(region, slot, index):
    return (region << 24) | (slot << 16) | index


_BUNDLE_CACHE = {}


def bundle():
    if BUNDLE not in _BUNDLE_CACHE:
        _BUNDLE_CACHE[BUNDLE] = load_bundle(BUNDLE)
    return _BUNDLE_CACHE[BUNDLE]


def bundle_luts():
    """(exp2, recip, rsqrt, [gelu per layer]) from the model bundle."""
    A, meta = bundle()
    gel = [A[f"h{l}.gelu.lut"] for l in range(meta["n_layer"])]
    return A["lut.exp2"], A["lut.recip"], A["lut.rsqrt"], gel


def ln_slots():
    """[(name, g, b, c)] for every LayerNorm of the bundle, in slot order:
    h0.ln_1, h0.ln_2, h1.ln_1, ..., ln_f (the order the SFU slot index uses in the tests)."""
    A, meta = bundle()
    names = [f"h{l}.{n}" for l in range(meta["n_layer"]) for n in ("ln_1", "ln_2")] + ["ln_f"]
    return [(n, A[n + ".g"], A[n + ".b"], [int(v) for v in A[n + ".c"]]) for n in names]


def s32(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v >> 31 else v


# ---------------------------------------------------------------- lane references
def ref_elem(op, x, exp_lut, recip_lut, rsqrt_lut, gelu_lut):
    """sfu_elem lane function (docs/sfu.md §3) for int32 lane values x (array)."""
    x = np.asarray(x, I64)
    xu = x & 0xFFFFFFFF
    if op == OP_EXP2:
        return ops.exp2_neg_i(xu & 0xFFFF, exp_lut)
    if op == OP_RECIP:
        y, b = ops.recip_i(xu, recip_lut)
        return np.array([s32((int(bb) << 24) | int(yy)) for yy, bb in zip(y, b)], I64)
    if op == OP_RSQRT:
        y, k = ops.rsqrt_i(np.maximum(xu, 1), rsqrt_lut)
        return np.array([s32(((int(kk) & 0xFF) << 24) | (int(yy) & 0xFFFFFF)) for yy, kk in zip(y, k)], I64)
    if op == OP_GELU:
        return ops.gelu_lut_i(x, gelu_lut).astype(I64)
    return x


def softmax_ref(s, mask):
    """C5 softmax of one row (int16 s, bool mask) -> int8 p (masked -> 0); all-masked -> 0."""
    exp_lut, recip_lut, _, _ = bundle_luts()
    s = np.asarray(s, I64)[None]
    m = np.asarray(mask, bool)[None]
    if not m.any():
        return np.zeros(s.shape[1], I64)
    p = ops.softmax_i(s, m, exp_lut, recip_lut).astype(I64)[0]
    return np.where(m[0], p, 0)


def ln_ref(x, g, b, c, rsqrt_lut, keep=None):
    """C5 LayerNorm of one row over the kept elements. Returns (out int8 row, n_z_sat,
    n_out_sat); dropped (keep = 0) positions read 0 and are not counted."""
    x = np.asarray(x, I64)
    keep = np.ones(len(x), bool) if keep is None else np.asarray(keep, bool)
    cd = dict(mu_mult=c[0], mu_shift=c[1], sqrt_d=c[2], eps_q=c[3])
    cnt = {}
    xs = x[keep][None]
    z = ops.ln_norm_i(xs, cd, rsqrt_lut, counter=cnt)
    o = ops.ln_affine_i(z, np.asarray(g, I64)[keep], np.asarray(b, I64)[keep], c[4], counter=cnt)
    out = np.zeros(len(x), I64)
    out[keep] = o[0]
    return out, cnt.get("ln_z", 0), cnt.get("ln_out", 0)


def beats_of(row, keep, L, rng=None, junk_bits=32):
    """Split a row into L-lane beats: list of (lanes, keep_bits, last). Lanes past the end
    of the row carry random junk (keep = 0) when rng is given, else 0."""
    n = len(row)
    out = []
    for j in range(0, n, L):
        seg = [int(v) for v in row[j:j + L]]
        kp = [int(bool(k)) for k in keep[j:j + L]]
        pad = L - len(seg)
        seg += [(rng.getrandbits(junk_bits) - (1 << (junk_bits - 1))) if rng else 0 for _ in range(pad)]
        kp += [0] * pad
        out.append((seg, kp, int(j + L >= n)))
    return out


async def write_port(dut, writes, we, addr, data):
    """Drive (we_value, addr, data) tuples on a write port, one per cycle, then idle."""
    for w, a, dv in writes:
        await FallingEdge(dut.clk)
        getattr(dut, we).value = w
        getattr(dut, addr).value = a
        getattr(dut, data).value = dv
    await FallingEdge(dut.clk)
    getattr(dut, we).value = 0


async def cycles_until(dut, cond, limit=100000):
    n = 0
    while True:
        await ReadOnly()
        if cond():
            return n
        await FallingEdge(dut.clk)
        n += 1
        assert n < limit
