"""cocotb bench for rtl/sfu_elem.sv (EXP2, RECIP, RSQRT, GELU lanes) vs nanoxl.quant.ops.

Tables are loaded at run time through the write port: the bundle's tables, and in one
test random tables (the unit must follow whatever was loaded).
"""
from __future__ import annotations
import random

import cocotb
import numpy as np
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from sfu_common_cocotb import (OP_EXP2, OP_GELU, OP_RECIP, OP_RSQRT, Sink, Source, bundle_luts, table_lines,
                               env_int, pack, ref_elem, run_bench, start, u, unpack, write_port)

L = env_int("L", 4)
NG = env_int("NG", 4)
EL = env_int("EL", L)          # EXP2 / RECIP / RSQRT primitive lanes
R = L // EL                     # cycles per EXP2 / RECIP / RSQRT beat
EXHAUSTIVE = env_int("EXHAUSTIVE", 1)
N_RAND = env_int("N_RAND", 2000)
INPUTS = ["in_valid", "in_op", "in_slot", "in_data", "in_keep", "in_last", "out_ready",
          "lut_we", "lut_waddr", "lut_wdata"]


async def load_tables(dut, exp_lut, recip_lut, rsqrt_lut, gelu):
    w = []
    for bit, t in ((0, exp_lut), (1, recip_lut), (2, rsqrt_lut)):
        w += [(1 << bit, i, ln) for i, ln in enumerate(table_lines(np.asarray(t, np.uint16)))]
    for s, t in enumerate(gelu):
        w += [(8, (s << 4) | i, ln) for i, ln in enumerate(table_lines(np.asarray(t, np.int8)))]
    await write_port(dut, w, "lut_we", "lut_waddr", "lut_wdata")


def recip_inputs(rng, n):
    edges = [1 << 15, (1 << 15) + 1, (1 << 16) - 1, 1 << 16, (1 << 24), (1 << 24) - 1, (1 << 32) - 1]
    for b in range(15, 32):
        edges += [1 << b, (1 << (b + 1)) - 1, (1 << b) | (127 << max(b - 9, 0))]
    rnd = [rng.randrange(1 << rng.randrange(15, 32), 1 << 32) if rng.random() < 0.5 else
           rng.randrange(1 << 15, 1 << 25) for _ in range(n)]
    return [v for v in edges + rnd if v >= 1 << 15]


def rsqrt_inputs(rng, n):
    edges = [0, 1, 2, 3, 4, 255, (1 << 14) - 1, 1 << 14, (1 << 14) + 1, (1 << 16) - 1, 1 << 16,
             (1 << 32) - 1, 20171, 20172]
    for b in range(32):
        edges += [1 << b, (1 << (b + 1)) - 1]
    rnd = [rng.randrange(0, 1 << rng.randrange(1, 33)) for _ in range(n)]
    return edges + rnd


def rows_for(op, vals, slot, rng):
    """Group lane values into rows of random length (op/slot constant per row); random
    interior keep holes; junk in the upper bits where the op ignores them."""
    items = []
    i = 0
    while i < len(vals):
        n = rng.randrange(1, 5 * L + 3)
        row = vals[i:i + n]
        i += n
        for j in range(0, len(row), L):
            seg = list(row[j:j + L])
            keep = [1] * len(seg) + [0] * (L - len(seg))
            seg += [rng.getrandbits(32) for _ in range(L - len(seg))]
            if rng.random() < 0.1:
                keep = [k & (rng.random() < 0.8) for k in keep]
            if op == OP_EXP2:
                seg = [(v & 0xFFFF) | (rng.getrandbits(16) << 16) for v in seg]
            if op == OP_GELU:
                seg = [(v & 0xFF) | (rng.getrandbits(24) << 8) for v in seg]
            items.append((op, slot, [v & 0xFFFFFFFF for v in seg], keep, int(j + L >= len(row))))
    return items


def expect(items, tabs):
    exp_lut, recip_lut, rsqrt_lut, gelu = tabs
    out = []
    for op, slot, seg, keep, last in items:
        x = np.array([v - (1 << 32) if v >> 31 else v for v in seg], np.int64)
        if op in (OP_RECIP,):  # out-of-contract S < 2^15 only appears in masked lanes
            x = np.where(np.array(keep, bool), x, 1 << 15)
        r = ref_elem(op, x, exp_lut, recip_lut, rsqrt_lut, gelu[slot] if op == OP_GELU else None)
        out.append(([int(v) if k else 0 for v, k in zip(r, keep)], keep, last))
    return out


async def run_items(dut, items, tabs, p_src, p_snk, seed):
    rng = random.Random(seed)
    exp = expect(items, tabs)

    def drive(it):
        op, slot, seg, keep, last = it
        dut.in_op.value = op
        dut.in_slot.value = slot
        dut.in_data.value = pack(seg, 32)
        dut.in_keep.value = pack(keep, 1)
        dut.in_last.value = last

    def junk():
        dut.in_data.value = rng.getrandbits(32 * L)
        dut.in_op.value = rng.getrandbits(3)

    def cap():
        return (unpack(u(dut.out_data), L, 32), unpack(u(dut.out_keep), L, 1, False), u(dut.out_last))

    src = Source(dut.in_valid, dut.in_ready, items, drive, junk, p_src, random.Random(seed + 1))
    snk = Sink(dut.out_valid, dut.out_ready, cap, p_snk, random.Random(seed + 2))
    cyc = await run_bench(dut, [src], [snk], [len(exp)])
    for i, (g, e) in enumerate(zip(snk.got, exp)):
        assert g == e, f"beat {i} op {items[i][0]}: in {items[i][2]} got {g} expected {e}"
    return cyc


def all_items(rng, tabs):
    items = []
    exp_vals = list(range(65536)) if EXHAUSTIVE else [rng.randrange(65536) for _ in range(N_RAND)]
    items += rows_for(OP_EXP2, exp_vals, 0, rng)
    items += rows_for(OP_RECIP, recip_inputs(rng, N_RAND), 0, rng)
    items += rows_for(OP_RSQRT, rsqrt_inputs(rng, N_RAND), 0, rng)
    for s in range(len(tabs[3])):
        items += rows_for(OP_GELU, list(range(256)), s, rng)
    items += rows_for(5, [rng.getrandbits(32) for _ in range(4 * L)], 0, rng)   # pass-through
    return items


@cocotb.test()
async def elem_bundle_tables(dut):
    """Every op on the bundle tables; EXP2 and GELU exhaustive over their input domains;
    shuffled row order (ops back to back), three handshake regimes."""
    await start(dut, INPUTS)
    tabs = bundle_luts()
    tabs = (tabs[0], tabs[1], tabs[2], tabs[3][:NG])
    await load_tables(dut, *tabs)
    rng = random.Random(5)
    items = all_items(rng, tabs)
    n = len(items)
    # full rate on one third, then two backpressure regimes, each on a shuffled subset
    a, b = n // 3, 2 * n // 3
    cyc = await run_items(dut, items[:a], tabs, 1.0, 1.0, 11)
    need = sum(R if it[0] <= OP_RSQRT else 1 for it in items[:a])   # shared ops: R cycles per beat
    assert cyc <= need + 8 + R, f"not full throughput: {a} beats ({need} cycles of work) took {cyc} cycles"
    sub = items[a:]
    rows = []
    cur = []
    for it in sub:
        cur.append(it)
        if it[4]:
            rows.append(cur)
            cur = []
    rng.shuffle(rows)
    sub = [it for r in rows for it in r] + cur
    await run_items(dut, sub[:b - a], tabs, 0.6, 0.4, 12)
    await run_items(dut, sub[b - a:], tabs, 0.9, 0.7, 13)
    dut._log.info(f"L={L}: {n} beats checked, {n * L} lane results")


@cocotb.test()
async def elem_random_tables(dut):
    """Random table contents: the unit computes from whatever was loaded (RTL never
    regenerates tables). Then the bundle tables are reloaded and re-checked."""
    await start(dut, INPUTS)
    rng = random.Random(7)
    rt = lambda: np.array([rng.getrandbits(16) for _ in range(256)], np.uint16)
    tabs = (rt(), rt(), rt(), [np.array([rng.randrange(-128, 128) for _ in range(256)], np.int8)
                               for _ in range(NG)])
    await load_tables(dut, *tabs)
    items = []
    items += rows_for(OP_EXP2, [rng.randrange(65536) for _ in range(1000)], 0, rng)
    items += rows_for(OP_RECIP, recip_inputs(rng, 500), 0, rng)
    items += rows_for(OP_RSQRT, rsqrt_inputs(rng, 500), 0, rng)
    for s in range(NG):
        items += rows_for(OP_GELU, list(range(256)), s, rng)
    await run_items(dut, items, tabs, 0.8, 0.8, 21)
    tabs = bundle_luts()
    tabs = (tabs[0], tabs[1], tabs[2], tabs[3][:NG])
    await load_tables(dut, *tabs)
    items = rows_for(OP_GELU, list(range(256)), NG - 1, rng) + \
        rows_for(OP_EXP2, [rng.randrange(65536) for _ in range(300)], 0, rng)
    await run_items(dut, items, tabs, 0.8, 0.8, 22)


@cocotb.test()
async def elem_gelu_exhaustive(dut):
    """GELU: all 256 INT8 inputs for every per-layer table of the bundle, one table per slot."""
    await start(dut, INPUTS)
    tabs = bundle_luts()
    tabs = (tabs[0], tabs[1], tabs[2], tabs[3][:NG])
    await load_tables(dut, *tabs)
    rng = random.Random(3)
    items = []
    for s in range(NG):
        vals = list(range(-128, 128))
        rng.shuffle(vals)
        items += rows_for(OP_GELU, [v & 0xFF for v in vals], s, rng)
    await run_items(dut, items, tabs, 0.7, 0.6, 31)


@cocotb.test()
async def elem_pipeline_depth(dut):
    """A single beat accepted at edge k appears on out_* after edge k+5 (6 stages); an EXP2
    beat with shared lanes (EL < L) after edge k+5+R-1, and in_ready is low for R-1 cycles."""
    await start(dut, INPUTS)
    tabs = bundle_luts()
    await load_tables(dut, tabs[0], tabs[1], tabs[2], tabs[3][:NG])
    dut.out_ready.value = 1
    await FallingEdge(dut.clk)
    dut.in_valid.value = 1
    dut.in_op.value = OP_EXP2
    dut.in_data.value = 0
    dut.in_keep.value = (1 << L) - 1
    dut.in_last.value = 1
    await ReadOnly()
    assert dut.in_ready.value == 1
    await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    for _ in range(R - 1):
        await ReadOnly()
        assert dut.in_ready.value == 0
        await FallingEdge(dut.clk)
    n = R
    while True:
        await ReadOnly()
        if dut.out_valid.value == 1:
            break
        await RisingEdge(dut.clk)
        n += 1
        assert n < 20
    assert n == 6 + R - 1, n
    assert unpack(u(dut.out_data), L, 32) == [32768] * L
