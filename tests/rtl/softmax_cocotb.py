"""cocotb bench for rtl/softmax.sv vs nanoxl.quant.ops.softmax_i (C5, exact 3-pass).

Rows: random (several logit distributions, interior masks), real attention-score rows
captured from nanoxl.golden on the tiny bundle, and directed corners. Random valid gaps
and output backpressure; exact cycle budget at full rate.
"""
from __future__ import annotations
import random

import cocotb
import numpy as np
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl import golden as gm
from nanoxl.graph import requant_ref
from sfu_common_cocotb import (Sink, Source, beats_of, bundle, bundle_luts, env_int, pack,
                               run_bench, softmax_ref, start, table_lines, to_lines, u, unpack,
                               write_port)

L = env_int("L", 4)
CTX = env_int("CTX_MAX", 256)
N_ROWS = env_int("N_ROWS", 300)
N_GOLDEN = env_int("N_GOLDEN", 400)      # golden score rows to replay (0 = all)
NCTX = env_int("NCTX", 8)
RQL = env_int("RQL", L)                 # score requant lanes
R = L // RQL                            # cycles per int32 input beat
INPUTS = ["in_valid", "in_data", "in_keep", "in_last", "in_rq", "in_slot", "in_nkeep",
          "out_ready", "lut_we", "lut_waddr", "lut_wdata", "ctx_we", "ctx_slot", "ctx_data",
          "sat_clr"]


def ctx_records():
    """Score records {bias, mult, shift, ls} per slot: slots 0..n_layer-1 = the bundle's
    h{l}.score (bias 0, the PBUF softmax ctx), the rest random (bias, ls, every shift)."""
    A, meta = bundle()
    recs = [(0, int(A[f"h{l}.score.mult"][0]), int(A[f"h{l}.score.shift"][0]), int(A[f"h{l}.score.ls"][0]))
            for l in range(meta["n_layer"])]
    rng = random.Random(42)
    while len(recs) < NCTX:
        recs.append((rng.randrange(-(1 << 20), 1 << 20), rng.randrange(1 << 30, 1 << 31),
                     rng.randrange(0, 32), rng.randrange(0, 4)))
    return recs


async def load(dut):
    exp_lut, recip_lut, _, _ = bundle_luts()
    w = [(1, i, ln) for i, ln in enumerate(table_lines(exp_lut))] + \
        [(2, i, ln) for i, ln in enumerate(table_lines(recip_lut))]
    await write_port(dut, w, "lut_we", "lut_waddr", "lut_wdata")
    w = [(1, s, to_lines(np.array(r, np.int32).tobytes())[0]) for s, r in enumerate(ctx_records())]
    await write_port(dut, w, "ctx_we", "ctx_slot", "ctx_data")


def rand_row(rng):
    r = rng.random()
    n = (1 if r < 0.05 else CTX if r < 0.1 else rng.randrange(1, L + 1) if r < 0.25
         else rng.randrange(1, CTX + 1))
    kind = rng.randrange(6)
    if kind == 0:
        x = [rng.randrange(-32768, 32768) for _ in range(n)]
    elif kind == 5:
        c = rng.randrange(-30000, 30000)
        x = [max(-32768, min(32767, c + rng.randrange(-300, 300))) for _ in range(n)]
    else:
        sig = (30, 300, 1500, 6000)[kind - 1]
        mu = rng.randrange(-20000, 20000)
        x = [int(max(-32768, min(32767, round(rng.gauss(mu, sig))))) for _ in range(n)]
    if rng.random() < 0.25:
        keep = [int(rng.random() < 0.7) for _ in range(n)]
        if not any(keep):
            keep[rng.randrange(n)] = 1
    else:
        keep = [1] * n
    if rng.random() < 0.2:      # causal limit (kept prefix), at least one visible column
        nk = rng.randrange(1, n + 1)
        keep[rng.randrange(nk)] = 1
        return dict(x=x, keep=keep, nkeep=nk)
    return x, keep


def rand_rq_row(rng):
    """int32 score row for the requant front end, with a random record slot."""
    n = rng.choice([1, L, L + 1, CTX, rng.randrange(1, CTX + 1), rng.randrange(1, CTX + 1)])
    slot = rng.randrange(NCTX)
    big = rng.random() < 0.3
    c = rng.randrange(-(1 << 24), 1 << 24)
    x = [rng.randrange(-(1 << 31), 1 << 31) if big else c + rng.randrange(-(1 << 18), 1 << 18)
         for _ in range(n)]
    keep = [1] * n
    nk = rng.randrange(1, n + 1) if rng.random() < 0.5 else 0
    return dict(x=x, keep=keep, rq=1, slot=slot, nkeep=nk)


def corner_rows():
    rows = []
    for v in (-32768, -1, 0, 1, 12345, 32767):
        for n in (1, 2, L, L + 1, 3 * L - 1, CTX):
            rows.append(([v] * n, [1] * n))                       # all equal
    for n in (1, 2, L + 1, CTX - 1, CTX):
        rows.append(([32767 if i % 2 else -32768 for i in range(n)], [1] * n))   # max magnitude
        rows.append(([-32768] * (n - 1) + [32767], [1] * n))
        rows.append(([32767] + [-32768] * (n - 1), [1] * n))
    # d around the EXP2 cut-off (q = d >> 8 = 15 / 16) and every residue of r
    rows.append(([0] + [-(15 * 256 + r) for r in range(256)][:CTX - 1], [1] * min(CTX, 257)))
    rows.append(([0] + [-(16 * 256 - 1 - r) for r in range(CTX - 1)], [1] * CTX))
    # masked lanes carrying the row's largest values (must be ignored) and a single kept lane
    n = 3 * L + 1
    rows.append(([32767 if i % 3 else -5 for i in range(n)], [int(i % 3 == 0) for i in range(n)]))
    rows.append(([32767] * n, [int(i == n - 1) for i in range(n)]))
    return [(x[:CTX], k[:CTX]) for x, k in rows]


_GOLDEN_ROWS = None


def golden_rows():
    """Attention-score rows (int16, Q.8 log2) of every layer and head, captured from the
    integer golden model on the tiny bundle, with the golden softmax outputs."""
    global _GOLDEN_ROWS
    if _GOLDEN_ROWS is None:
        A, meta = bundle()
        g = gm.Golden(A, meta)
        rng = np.random.default_rng(20261107)
        T = min(meta["block_size"], CTX)
        toks = rng.integers(0, meta["vocab_size"], T)
        tr = {}
        g.forward(toks, trace=tr)
        rows, rq_rows = [], []
        hs = meta["n_embd"] // meta["n_head"]
        for l in range(meta["n_layer"]):
            sc, att = tr[f"scores.{l}"], tr[f"att.{l}"]
            q, k = tr[f"q.{l}"].astype(np.int64), tr[f"k.{l}"].astype(np.int64)
            for h in range(meta["n_head"]):
                acc = q[:, h * hs:(h + 1) * hs] @ k[:, h * hs:(h + 1) * hs].T     # int32 q.K^T
                for i in range(T):
                    rows.append((sc[h, i, :i + 1].astype(np.int64), att[h, i, :i + 1].astype(np.int64)))
                    rq_rows.append((dict(x=acc[i], keep=[1] * T, rq=1, slot=l, nkeep=i + 1),
                                    att[h, i].astype(np.int64)))
        sat = sum(g.sat_count.get(f"h{l}.score", 0) for l in range(meta["n_layer"]))
        _GOLDEN_ROWS = (rows, rq_rows, sat)
    return _GOLDEN_ROWS


def norm(r):
    """Row as dict: x (int16 logits, or int32 scores when rq), keep, rq, slot, nkeep."""
    if isinstance(r, dict):
        return dict(dict(rq=0, slot=0, nkeep=0), **r)
    return dict(x=r[0], keep=r[1], rq=0, slot=0, nkeep=0)


def row_beats(rows, rng, sat=None):
    items, exp = [], []
    recs = ctx_records()
    for r in map(norm, rows):
        x, keep = [int(v) for v in r["x"]], [int(k) for k in r["keep"]]
        n = len(x)
        if r["rq"]:
            bias, mult, shift, ls = recs[r["slot"]]
            cnt = {}
            kk = np.array(keep, bool)
            s = requant_ref(np.array(x, np.int64)[kk], mult, shift, bias, 16, cnt, "score", ls)
            if sat is not None:
                sat[0] += cnt["score"]
            s16 = np.zeros(n, np.int64)
            s16[kk] = s
        else:
            s16 = np.array(x, np.int64)
        mask = np.array(keep, bool) & ((np.arange(n) < r["nkeep"]) if r["nkeep"] else True)
        p = softmax_ref(s16, mask)
        items += [(seg, kp, last, r) for seg, kp, last in beats_of(x, keep, L, rng)]
        exp += [([int(v) * k for v, k in zip(seg, kp)], kp, last)
                for seg, kp, last in beats_of([int(v) for v in p], keep, L)]
    return items, exp


async def run_rows(dut, rows, p_src, p_snk, seed):
    rng = random.Random(seed)
    sat = [u(dut.sat_count)]
    items, exp = row_beats(rows, rng, sat)

    def drive(it):
        seg, kp, last, r = it
        dut.in_data.value = pack(seg, 32)
        dut.in_keep.value = pack(kp, 1)
        dut.in_last.value = last
        dut.in_rq.value = r["rq"]
        dut.in_slot.value = r["slot"]
        dut.in_nkeep.value = r["nkeep"]

    def junk():
        dut.in_data.value = rng.getrandbits(32 * L)
        dut.in_keep.value = rng.getrandbits(L)
        dut.in_rq.value = rng.getrandbits(1)
        dut.in_nkeep.value = rng.getrandbits(16)

    def cap():
        return (unpack(u(dut.out_data), L, 32), unpack(u(dut.out_keep), L, 1, False), u(dut.out_last))

    src = Source(dut.in_valid, dut.in_ready, items, drive, junk, p_src, random.Random(seed + 1))
    snk = Sink(dut.out_valid, dut.out_ready, cap, p_snk, random.Random(seed + 2))
    cyc = await run_bench(dut, [src], [snk], [len(exp)])
    for i, (g, e) in enumerate(zip(snk.got, exp)):
        assert g == e, f"beat {i}: got {g} expected {e}"
    await ReadOnly()
    assert dut.busy.value == 0 and dut.err_len.value == 0
    assert u(dut.sat_count) == sat[0], (u(dut.sat_count), sat[0])
    await FallingEdge(dut.clk)
    return cyc


def nb(n):
    return -(-n // L)


@cocotb.test()
async def softmax_random(dut):
    await start(dut, INPUTS)
    await load(dut)
    rng = random.Random(100 + L)
    rows = [rand_row(rng) for _ in range(N_ROWS)]
    k = len(rows) // 3
    await run_rows(dut, rows[:k], 0.6, 0.5, 1)
    await run_rows(dut, rows[k:2 * k], 0.95, 0.3, 2)
    await run_rows(dut, rows[2 * k:], 0.85, 0.85, 3)
    dut._log.info(f"L={L}: {len(rows)} random rows, {sum(len(norm(r)["x"]) for r in rows)} elements")


@cocotb.test()
async def softmax_corners(dut):
    await start(dut, INPUTS)
    await load(dut)
    await run_rows(dut, corner_rows(), 0.7, 0.6, 4)


@cocotb.test()
async def softmax_full_rate(dut):
    """in_valid = out_ready = 1: exact cost model 3*nb + 13 cycles per row (docs/sfu.md)."""
    await start(dut, INPUTS)
    await load(dut)
    rng = random.Random(9)
    rows = [rand_row(rng) for _ in range(20)] + [([5] * CTX, [1] * CTX), ([7], [1])]
    cyc = await run_rows(dut, rows, 1.0, 1.0, 5)
    budget = sum(3 * nb(len(norm(r)["x"])) + 13 for r in rows)
    dut._log.info(f"L={L}: {len(rows)} rows in {cyc} cycles (model {budget})")
    assert budget - 2 <= cyc <= budget + 2, (cyc, budget)


@cocotb.test()
async def softmax_golden_rows(dut):
    """Attention-score rows from nanoxl.golden (tiny bundle): bit-exact vs the golden P."""
    await start(dut, INPUTS)
    await load(dut)
    rows = golden_rows()[0]
    rng = random.Random(77)
    if 0 < N_GOLDEN < len(rows):
        # always include the longest rows, then a random sample
        long = [r for r in rows if len(r[0]) >= CTX // 2 and len(r[0]) % 37 == 0]
        rows = long + rng.sample(rows, N_GOLDEN - len(long))
    for x, p in rows:
        assert np.array_equal(softmax_ref(x, np.ones(len(x), bool)), p)
    await run_rows(dut, [(x, [1] * len(x)) for x, _ in rows], 0.8, 0.7, 6)
    dut._log.info(f"L={L}: {len(rows)} golden score rows bit-exact")


@cocotb.test()
async def softmax_too_long(dut):
    """A row longer than CTX_MAX sets err_len, returns the right number of beats, and the
    unit recovers for the next row."""
    await start(dut, INPUTS)
    await load(dut)
    rng = random.Random(3)
    n = CTX + L + 1
    items = beats_of([0] * n, [1] * n, L, rng)
    got = []

    def drive(it):
        dut.in_data.value = pack(it[0], 32)
        dut.in_keep.value = pack(it[1], 1)
        dut.in_last.value = it[2]
        r = norm(it[3]) if len(it) > 3 else norm(([], []))
        dut.in_rq.value = r["rq"]
        dut.in_nkeep.value = r["nkeep"]
        dut.in_slot.value = r["slot"]

    src = Source(dut.in_valid, dut.in_ready, items, drive, None, 1.0)
    snk = Sink(dut.out_valid, dut.out_ready, lambda: u(dut.out_last), 1.0)
    await run_bench(dut, [src], [snk], [len(items)])
    assert snk.got[-1] == 1 and sum(snk.got) == 1
    await ReadOnly()
    assert dut.err_len.value == 1
    await FallingEdge(dut.clk)
    rows = [rand_row(rng) for _ in range(5)]
    items, exp = row_beats(rows, rng)
    src = Source(dut.in_valid, dut.in_ready, items, drive, None, 0.8)
    snk = Sink(dut.out_valid, dut.out_ready,
               lambda: (unpack(u(dut.out_data), L, 32), unpack(u(dut.out_keep), L, 1, False),
                        u(dut.out_last)), 0.8)
    await run_bench(dut, [src], [snk], [len(exp)])
    assert snk.got == exp


@cocotb.test()
async def softmax_rq_random(dut):
    """int32 score rows through the requant front end (SOFTMAX_ONLINE): every record slot,
    causal limits, int16 clamps counted."""
    await start(dut, INPUTS)
    await load(dut)
    rng = random.Random(300 + L)
    rows = [rand_rq_row(rng) for _ in range(max(N_ROWS // 3, 20))]
    k = len(rows) // 2
    await run_rows(dut, rows[:k] + [rand_row(rng) for _ in range(5)], 0.7, 0.6, 11)
    await run_rows(dut, rows[k:], 0.95, 0.9, 12)
    assert u(dut.sat_count) > 0, "score saturation not exercised"
    dut.sat_clr.value = 1
    await FallingEdge(dut.clk)
    dut.sat_clr.value = 0
    await ReadOnly()
    assert u(dut.sat_count) == 0 and dut.sat_sticky.value == 0
    await FallingEdge(dut.clk)


@cocotb.test()
async def softmax_rq_golden(dut):
    """The golden model's int32 q.K^T rows (all keys of the step, causal limit i+1, the
    layer's score record): P bit-exact; with every row replayed the score saturation count
    equals Golden.sat_count["h{l}.score"] summed over layers."""
    await start(dut, INPUTS)
    await load(dut)
    _, rows, sat = golden_rows()
    full = not (0 < N_GOLDEN < len(rows))
    if not full:
        rows = random.Random(78).sample(rows, N_GOLDEN // 2)
    for r, p in rows:   # the reference path itself reproduces the golden P
        _, exp = row_beats([r], random.Random(0))
        got = [v for seg, kp, _ in exp for v in seg][:len(p)]
        assert got == [int(v) for v in p]
    await run_rows(dut, [r for r, _ in rows], 0.8, 0.8, 13)
    if full:
        assert u(dut.sat_count) == sat
    dut._log.info(f"L={L}: {len(rows)} golden int32 score rows bit-exact (full={full})")


@cocotb.test()
async def softmax_rq_full_rate(dut):
    """Requant rows at full rate: (R+2)*nb + 19 cycles per row (R = L/RQL, 3*nb + 19 unshared)."""
    await start(dut, INPUTS)
    await load(dut)
    rng = random.Random(19)
    rows = [rand_rq_row(rng) for _ in range(10)]
    cyc = await run_rows(dut, rows, 1.0, 1.0, 14)
    budget = sum((R + 2) * nb(len(r["x"])) + 19 for r in rows)
    dut._log.info(f"L={L}: {len(rows)} requant rows in {cyc} cycles (model {budget})")
    assert budget - 2 <= cyc <= budget + 2, (cyc, budget)
