"""cocotb bench for rtl/vec_reduce.sv against nanoxl.vec_ref.reduce_row."""
from __future__ import annotations
import random

import cocotb
import numpy as np

from nanoxl import vec_ref as vr
from vec_common_cocotb import Sink, Source, env_int, pack, run_bench, start, u, unpack

L = env_int("L", 4)
N_ROWS = env_int("N_ROWS", 300)
ACC_W = vr.REDUCE_ACC_W
I32_MIN, I32_MAX = vr.INT32_MIN, vr.INT32_MAX

INPUTS = ["in_valid", "in_data", "in_keep", "in_last", "out_ready"]


def row_to_beats(vals, rng, ragged_keep):
    """Split a row into beats of L lanes; the final beat is padded (keep=0) with junk.
    With ragged_keep, interior lanes are also randomly masked (and replaced by junk);
    the returned list of kept values is what the reference must see."""
    beats, kept = [], []
    for j in range(0, len(vals), L):
        grp = list(vals[j:j + L])
        keep = [1] * len(grp) + [0] * (L - len(grp))
        if ragged_keep:
            keep = [k if rng.random() < 0.8 else 0 for k in keep]
        data = [g if k else rng.randrange(I32_MIN, I32_MAX + 1)
                for g, k in zip(grp + [0] * (L - len(grp)), keep)]
        kept += [d for d, k in zip(data, keep) if k]
        beats.append({"data": data, "keep": keep, "last": 0})
    beats[-1]["last"] = 1
    return beats, kept


def ref(kept):
    if not kept:
        return {"sum": 0, "sumsq": 0, "max": I32_MIN, "min": I32_MAX, "sq_err": False}
    return vr.reduce_row(np.array(kept, np.int64))


def rand_row(rng):
    R = rng.choice([1, 2, L - 1 or 1, L, L + 1, 3 * L, 384, rng.randrange(1, 8 * L + 2),
                    rng.randrange(1, 400)])
    kind = rng.random()
    if kind < 0.6:
        vals = [rng.randrange(-32768, 32768) for _ in range(R)]
    elif kind < 0.75:
        vals = [rng.randrange(-200, 200) for _ in range(R)]
    elif kind < 0.9:
        vals = [rng.randrange(I32_MIN, I32_MAX + 1) for _ in range(R)]   # sq_err rows
    else:
        vals = [rng.choice([-32768, 32767, I32_MIN, I32_MAX, 0, -1]) for _ in range(R)]
    return vals


def directed_rows():
    return [[-32768] * 384, [32767] * 384, [I32_MIN] * 384, [I32_MAX] * 384, [0] * 5,
            [-1], [I32_MIN], [I32_MAX], list(range(-192, 192)),
            [-32768] * 2048,                         # sumsq = 2^41: far above 32 bits
            [I32_MIN + 1, I32_MAX] * 7 + [I32_MIN]]


async def run_rows(dut, rows, rng, ragged, p_in, p_out):
    beats, refs = [], []
    for vals in rows:
        b, kept = row_to_beats(vals, rng, ragged)
        beats += b
        refs.append(ref(kept))

    def drive(b):
        dut.in_data.value = pack(b["data"], 32)
        dut.in_keep.value = pack(b["keep"], 1)
        dut.in_last.value = b["last"]

    def junk():
        dut.in_data.value = rng.getrandbits(32 * L)
        dut.in_last.value = rng.getrandbits(1)

    def capture():
        return {"sum": unpack(u(dut.out_sum), 1, ACC_W)[0], "sumsq": unpack(u(dut.out_sumsq), 1, ACC_W)[0],
                "max": unpack(u(dut.out_max), 1, 32)[0], "min": unpack(u(dut.out_min), 1, 32)[0],
                "sq_err": bool(u(dut.out_sq_err))}

    src = Source(dut.in_valid, dut.in_ready, beats, drive, junk, p_in, random.Random(rng.random()))
    snk = Sink(dut.out_valid, dut.out_ready, capture, p_out, random.Random(rng.random()))
    await run_bench(dut, [src], [snk], [len(rows)])
    for n, (g, e) in enumerate(zip(snk.got, refs)):
        e = dict(e, sum=vr.wrap_acc(e["sum"]), sumsq=vr.wrap_acc(e["sumsq"]))
        assert g == e, f"row {n} (len {len(rows[n])}): got {g} exp {e}"
    return len(beats)


@cocotb.test()
async def directed(dut):
    rng = random.Random(8)
    await start(dut, INPUTS)
    nb = await run_rows(dut, directed_rows(), rng, False, 0.9, 0.9)
    nb += await run_rows(dut, directed_rows(), rng, True, 0.7, 0.5)
    dut._log.info(f"L={L}: directed rows, {nb} beats")


@cocotb.test()
async def random_rows_backpressure(dut):
    rng = random.Random(9)
    await start(dut, INPUTS)
    rows = [rand_row(rng) for _ in range(N_ROWS)]
    h = len(rows) // 3
    nb = await run_rows(dut, rows[:h], rng, False, 1.0, 1.0)
    nb += await run_rows(dut, rows[h:2 * h], rng, True, 0.8, 0.3)
    nb += await run_rows(dut, rows[2 * h:], rng, False, 0.4, 0.9)
    # single-beat rows back to back at full rate (result register turnaround)
    nb += await run_rows(dut, [[rng.randrange(-99, 99) for _ in range(rng.randrange(1, L + 1))]
                               for _ in range(200)], rng, False, 1.0, 1.0)
    dut._log.info(f"L={L}: {len(rows) + 200} random rows, {nb} beats")


def documented_latency(lanes):
    """Result valid 2 + NTS cycles after the last beat; NTS = tree register stages of 2 levels."""
    lv = (max(lanes, 1) - 1).bit_length()
    return 2 + max(1, -(-lv // 2))


@cocotb.test()
async def result_latency(dut):
    """A one-beat row accepted at edge k gives out_valid after edge k + latency - 1
    (latency counted as in the vec_alu / requant depth tests)."""
    from cocotb.triggers import FallingEdge, ReadOnly
    await start(dut, INPUTS)
    dut.out_ready.value = 1
    dut.in_data.value = pack(list(range(1, L + 1)), 32)
    dut.in_keep.value = (1 << L) - 1
    dut.in_last.value = 1
    dut.in_valid.value = 1
    await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    for c in range(1, 12):
        await ReadOnly()
        if dut.out_valid.value == 1:
            break
        await FallingEdge(dut.clk)
    assert c == documented_latency(L), f"latency {c}, documented {documented_latency(L)}"
    assert u(dut.out_sum) == L * (L + 1) // 2
