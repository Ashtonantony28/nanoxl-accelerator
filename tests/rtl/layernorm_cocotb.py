"""cocotb bench for rtl/layernorm.sv vs nanoxl.quant.ops.ln_norm_i + ln_affine_i (C5).

Slots 0..2n_layer hold the tiny bundle's LayerNorms (h0.ln_1, h0.ln_2, ..., ln_f); the
remaining slots hold random constants and affine parameters for rows up to D_MAX that
also drive both saturation points (z and out). Rows: real LN inputs captured from
nanoxl.golden, random rows, corners; random valid gaps and backpressure; exact cycle
budget at full rate; saturation counters against the reference counts.
"""
from __future__ import annotations
import math
import random

import cocotb
import numpy as np
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl import golden as gm
from nanoxl import numerics as nx
from nanoxl.funcsim import ln_block
from sfu_common_cocotb import (Sink, Source, beats_of, bundle, bundle_luts, env_int, ln_ref,
                               ln_slots, pack, run_bench, start, table_lines, to_lines, u,
                               unpack, write_port)

L = env_int("L", 4)
D_MAX = env_int("D_MAX", 384)
NSLOT = env_int("NSLOT", 16)
N_ROWS = env_int("N_ROWS", 200)
N_GOLDEN = env_int("N_GOLDEN", 300)      # 0 = every captured LN row (then sat counts compared)
INPUTS = ["in_valid", "in_slot", "in_data", "in_keep", "in_last", "out_ready", "lut_we",
          "lut_waddr", "lut_wdata", "blk_we", "blk_slot", "blk_line", "blk_cols", "blk_data",
          "sat_clr"]


def make_slots(rng):
    """[(name, g[D_MAX], b[D_MAX], c[5], kind)] for every slot."""
    slots = [(n, g, b, c, "bundle") for n, g, b, c in ln_slots()]
    assert len(slots) < NSLOT
    while len(slots) < NSLOT:
        i = len(slots)
        d_c = rng.choice([D_MAX, D_MAX, rng.randrange(D_MAX, 4097)])
        mu_mult, mu_shift = nx.quantize_multiplier(1.0 / d_c)
        # sqrt_d of a wider LN on some slots: |z| can then exceed int16 (sat16 events)
        d_s = 4096 if i % 4 == 2 else rng.choice([D_MAX, rng.randrange(2, 4097)])
        sqrt_d = min(65535, int(round(math.sqrt(d_s) * 1024)))
        eps_q = rng.choice([0, 1, rng.randrange(0, 1 << 16), rng.randrange(0, 1 << 30)])
        if i % 4 == 3:     # small g, small sh: many exact halves in rdivpot, few clamps
            sh = rng.randrange(1, 7)
            g = np.array([rng.randrange(-2, 3) for _ in range(D_MAX)], np.int64)
            b = np.array([rng.randrange(-(20 << sh), 20 << sh) for _ in range(D_MAX)], np.int64)
        elif i % 4 == 0:   # model-like affine (from float gamma/beta, as the bundle builds it)
            from nanoxl.quant.ops import ln_affine_consts
            g, b, sh = ln_affine_consts(np.random.default_rng(i).normal(1, 0.3, D_MAX),
                                        np.random.default_rng(i + 99).normal(0, 0.2, D_MAX),
                                        rng.choice([0.01, 0.03, 0.1]))
        else:              # extreme affine: many sat8 events, every sh
            sh = rng.randrange(0, 32) if i % 4 == 1 else rng.randrange(8, 20)
            g = np.array([rng.randrange(-32768, 32768) for _ in range(D_MAX)], np.int64)
            b = np.array([rng.randrange(-(1 << 31), 1 << 31) >> rng.randrange(0, 20) for _ in range(D_MAX)],
                         np.int64)
        slots.append((f"rand{i}", np.asarray(g, np.int64), np.asarray(b, np.int64),
                      [mu_mult, mu_shift, sqrt_d, eps_q, int(sh)], "rand"))
    return slots


async def load(dut, slots):
    """rsqrt table (LUT_LOAD layout) and every slot's LN parameter block exactly as the
    compiler lays it out in PBUF (nanoxl.funcsim.ln_block), one 16-byte line per cycle."""
    rs = bundle_luts()[2]
    await write_port(dut, [(1, i, ln) for i, ln in enumerate(table_lines(rs))],
                     "lut_we", "lut_waddr", "lut_wdata")
    for s, (_, g, b, c, _) in enumerate(slots):
        for i, ln in enumerate(to_lines(ln_block(c, g, b))):
            await FallingEdge(dut.clk)
            dut.blk_we.value = 1
            dut.blk_slot.value = s
            dut.blk_line.value = i
            dut.blk_cols.value = len(g)
            dut.blk_data.value = ln
    await FallingEdge(dut.clk)
    dut.blk_we.value = 0


def rand_x(rng, n):
    kind = rng.randrange(6)
    if kind == 0:
        return [rng.randrange(-32768, 32768) for _ in range(n)]
    sig = (5, 200, 2000, 9000, 20000)[kind - 1]
    mu = rng.choice([0, rng.randrange(-8000, 8000)])
    return [int(max(-32768, min(32767, round(rng.gauss(mu, sig))))) for _ in range(n)]


def rand_row(rng, slots):
    s = rng.randrange(len(slots))
    if slots[s][4] == "bundle":
        n = len(slots[s][1])
        return s, rand_x(rng, n), [1] * n
    r = rng.random()
    n = 1 if r < 0.04 else D_MAX if r < 0.15 else rng.randrange(1, L + 2) if r < 0.25 else \
        rng.randrange(1, D_MAX + 1)
    keep = [1] * n
    if rng.random() < 0.15 and n > 1:
        keep = [int(rng.random() < 0.8) for _ in range(n)]
        keep[0] = 1
    return s, rand_x(rng, n), keep


def corner_rows(slots):
    rows = []
    rs = [i for i, s in enumerate(slots) if s[4] == "rand"]
    for s in (0, len(ln_slots()) - 1, *rs[:4]):
        n_full = len(slots[s][1]) if slots[s][4] == "bundle" else D_MAX
        lens = (n_full,) if slots[s][4] == "bundle" else (1, 2, L - 1 or 1, L + 1, n_full - 1, n_full)
        for n in lens:
            for v in (-32768, -1, 0, 7, 32767):
                rows.append((s, [v] * n, [1] * n))                     # all equal (Q = eps_q)
            rows.append((s, [32767 if i % 2 else -32768 for i in range(n)], [1] * n))
            rows.append((s, [-32768] * (n - 1) + [32767], [1] * n))    # one outlier
            rows.append((s, [32767] + [-32768] * (n - 1), [1] * n))
    return rows


_GOLD = None


def golden_rows():
    """[(slot, x int16 row, expected int8 row)] for every LN row of one tiny forward pass
    (128 tokens), plus the golden saturation counts of that pass."""
    global _GOLD
    if _GOLD is None:
        A, meta = bundle()
        g = gm.Golden(A, meta)
        toks = np.random.default_rng(20261107).integers(0, meta["vocab_size"], meta["block_size"])
        tr = {}
        g.forward(toks, trace=tr)
        rows = []
        for l in range(meta["n_layer"]):
            for t in range(len(toks)):
                rows.append((2 * l, tr[f"resid.{l}.in"][t], tr[f"ln_1.{l}"][t]))
                rows.append((2 * l + 1, tr[f"resid.{l}.mid"][t], tr[f"ln_2.{l}"][t]))
        for t in range(len(toks)):
            rows.append((2 * meta["n_layer"], tr["resid.out"][t], tr["ln_f"][t]))
        names = [n for n, *_ in ln_slots()]
        sat = (g.sat_count.get("ln_z", 0), sum(g.sat_count.get(n, 0) for n in names))
        _GOLD = (rows, sat)
    return _GOLD


class Bench:
    def __init__(self, dut, slots):
        self.dut, self.slots = dut, slots
        self.zsat = self.osat = 0

    def build(self, rows, rng):
        items, exp = [], []
        for s, x, keep in rows:
            _, g, b, c, _ = self.slots[s]
            n = len(x)
            o, nz, no = ln_ref(np.array(x), g[:n], b[:n], c, bundle_luts()[2], keep)
            self.zsat += nz
            self.osat += no
            items += [(s, *bt) for bt in beats_of([int(v) for v in x], keep, L, rng)]
            exp += [([int(v) * k for v, k in zip(seg, kp)], kp, last)
                    for seg, kp, last in beats_of([int(v) for v in o], keep, L)]
        return items, exp

    async def run(self, rows, p_src, p_snk, seed):
        dut = self.dut
        rng = random.Random(seed)
        items, exp = self.build(rows, rng)

        def drive(it):
            s, seg, kp, last = it
            dut.in_slot.value = s
            dut.in_data.value = pack(seg, 32)
            dut.in_keep.value = pack(kp, 1)
            dut.in_last.value = last

        def junk():
            dut.in_data.value = rng.getrandbits(32 * L)
            dut.in_slot.value = rng.getrandbits(4)

        def cap():
            return (unpack(u(dut.out_data), L, 32), unpack(u(dut.out_keep), L, 1, False),
                    u(dut.out_last))

        src = Source(dut.in_valid, dut.in_ready, items, drive, junk, p_src, random.Random(seed + 1))
        snk = Sink(dut.out_valid, dut.out_ready, cap, p_snk, random.Random(seed + 2))
        cyc = await run_bench(dut, [src], [snk], [len(exp)])
        for i, (g, e) in enumerate(zip(snk.got, exp)):
            assert g == e, f"beat {i} (slot {items[i][0]}): got {g} expected {e}"
        await ReadOnly()
        assert dut.busy.value == 0 and dut.err_len.value == 0
        assert u(dut.z_sat_count) == self.zsat, (u(dut.z_sat_count), self.zsat)
        assert u(dut.out_sat_count) == self.osat, (u(dut.out_sat_count), self.osat)
        await FallingEdge(dut.clk)
        return cyc


async def setup(dut, seed=1):
    await start(dut, INPUTS)
    slots = make_slots(random.Random(seed))
    await load(dut, slots)
    return Bench(dut, slots)


@cocotb.test()
async def ln_random(dut):
    bench = await setup(dut)
    rng = random.Random(200 + L)
    rows = [rand_row(rng, bench.slots) for _ in range(N_ROWS)]
    # short rows on the slots with a wide sqrt_d always reach the z clamp
    wide = [i for i, s in enumerate(bench.slots) if s[4] == "rand" and i % 4 == 2]
    rows += [(s, rand_x(rng, rng.randrange(1, 9)), None) for s in wide for _ in range(3)]
    rows = [(s, x, [1] * len(x) if k is None else k) for s, x, k in rows]
    rng.shuffle(rows)
    k = len(rows) // 3
    await bench.run(rows[:k], 0.6, 0.5, 1)
    await bench.run(rows[k:2 * k], 0.95, 0.3, 2)
    await bench.run(rows[2 * k:], 0.85, 0.85, 3)
    dut._log.info(f"L={L}: {len(rows)} random rows, z sat {bench.zsat}, out sat {bench.osat}")
    assert bench.zsat > 0 and bench.osat > 0, "saturation paths not exercised"


@cocotb.test()
async def ln_corners(dut):
    bench = await setup(dut)
    await bench.run(corner_rows(bench.slots), 0.7, 0.6, 4)


@cocotb.test()
async def ln_full_rate(dut):
    """in_valid = out_ready = 1: exact cost model 3*nb + 19 cycles per row (docs/sfu.md)."""
    bench = await setup(dut)
    rng = random.Random(9)
    rows = [rand_row(rng, bench.slots) for _ in range(12)]
    rows += [(NSLOT - 1, [3] * D_MAX, [1] * D_MAX), (NSLOT - 1, [-9], [1])]
    cyc = await bench.run(rows, 1.0, 1.0, 5)
    budget = sum(3 * (-(-len(x) // L)) + 19 for _, x, _ in rows)
    dut._log.info(f"L={L}: {len(rows)} rows in {cyc} cycles (model {budget})")
    assert budget - 2 <= cyc <= budget + 2, (cyc, budget)


@cocotb.test()
async def ln_golden_rows(dut):
    """LN input rows captured from nanoxl.golden on the tiny bundle: bit-exact outputs;
    with every row replayed, the saturation counters equal the golden per-op counts."""
    bench = await setup(dut)
    rows, sat = golden_rows()
    full = not (0 < N_GOLDEN < len(rows))
    if not full:
        rows = random.Random(55).sample(rows, N_GOLDEN)
    for s, x, o in rows:
        _, g, b, c, _ = bench.slots[s]
        assert np.array_equal(ln_ref(x, g, b, c, bundle_luts()[2])[0], o)
    await bench.run([(s, [int(v) for v in x], [1] * len(x)) for s, x, _ in rows], 0.8, 0.7, 6)
    if full:
        assert (u(dut.z_sat_count), u(dut.out_sat_count)) == sat
    dut._log.info(f"L={L}: {len(rows)} golden LN rows bit-exact (full={full}, golden sat {sat})")


@cocotb.test()
async def ln_sat_counter_clear(dut):
    """sat_clr clears both counters and the sticky flag; events keep counting after."""
    bench = await setup(dut)
    rs = [i for i, s in enumerate(bench.slots) if s[4] == "rand" and i % 4 == 1][0]
    rows = [(rs, [32767 if i % 2 else -32768 for i in range(D_MAX)], [1] * D_MAX)]
    await bench.run(rows, 1.0, 1.0, 7)
    await ReadOnly()
    assert dut.sat_sticky.value == (1 if bench.zsat + bench.osat else 0)
    await FallingEdge(dut.clk)
    dut.sat_clr.value = 1
    await FallingEdge(dut.clk)
    dut.sat_clr.value = 0
    await ReadOnly()
    assert u(dut.z_sat_count) == 0 and u(dut.out_sat_count) == 0 and dut.sat_sticky.value == 0
    await FallingEdge(dut.clk)
    bench.zsat = bench.osat = 0
    await bench.run(rows, 0.7, 0.7, 8)


@cocotb.test()
async def ln_too_long(dut):
    """A row longer than D_MAX sets err_len, keeps the beat count, and the unit recovers."""
    bench = await setup(dut)
    rng = random.Random(3)
    n = D_MAX + L + 1
    items = [(NSLOT - 1, *bt) for bt in beats_of([0] * n, [1] * n, L, rng)]

    def drive(it):
        dut.in_slot.value = it[0]
        dut.in_data.value = pack(it[1], 32)
        dut.in_keep.value = pack(it[2], 1)
        dut.in_last.value = it[3]

    src = Source(dut.in_valid, dut.in_ready, items, drive, None, 1.0)
    snk = Sink(dut.out_valid, dut.out_ready, lambda: u(dut.out_last), 1.0)
    await run_bench(dut, [src], [snk], [len(items)])
    assert snk.got[-1] == 1 and sum(snk.got) == 1
    await ReadOnly()
    assert dut.err_len.value == 1
    await FallingEdge(dut.clk)
    dut.sat_clr.value = 1
    await FallingEdge(dut.clk)
    dut.sat_clr.value = 0
    bench.zsat = bench.osat = 0
    rows = [rand_row(rng, bench.slots) for _ in range(4)]
    items, exp = bench.build(rows, rng)
    src = Source(dut.in_valid, dut.in_ready, items, drive, None, 0.8)
    snk = Sink(dut.out_valid, dut.out_ready,
               lambda: (unpack(u(dut.out_data), L, 32), unpack(u(dut.out_keep), L, 1, False),
                        u(dut.out_last)), 0.8)
    await run_bench(dut, [src], [snk], [len(exp)])
    assert snk.got == exp
