"""cocotb bench for rtl/vec_alu.sv against nanoxl.vec_ref.alu."""
from __future__ import annotations
import random

import cocotb
import numpy as np
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl import vec_ref as vr
from vec_common_cocotb import Sink, Source, env_int, pack, run_bench, start, u, unpack

L = env_int("L", 16)
N_PER_OP = env_int("N_PER_OP", 20000)      # random lane-results per op
I32 = (vr.INT32_MIN, vr.INT32_MAX + 1)
EXT = [vr.INT32_MIN, vr.INT32_MIN + 1, -32769, -32768, -129, -128, -1, 0, 1, 127, 128,
       32767, 32768, vr.INT32_MAX - 1, vr.INT32_MAX]

INPUTS = ["in_valid", "in_a", "in_b", "in_op", "in_fmt", "in_shamt", "in_lo", "in_hi",
          "in_keep", "in_last", "out_ready", "sat_clr"]


def rand_val(rng):
    r = rng.random()
    if r < 0.3:
        return rng.randrange(*I32)
    if r < 0.55:
        return rng.randrange(-40000, 40000)
    if r < 0.8:
        return rng.randrange(-300, 300)
    if r < 0.9:
        return rng.randrange(-(1 << 20), 1 << 20)
    return rng.choice(EXT)


def beat(rng, op, a, b, fmt=None, sh=None):
    lo, hi = sorted(rng.choice([rand_val(rng), rng.randrange(-200, 200)]) for _ in range(2))
    if rng.random() < 0.05:
        lo, hi = hi, lo                       # lo > hi: result is hi (documented)
    return {"op": op, "a": a, "b": b, "fmt": rng.randrange(4) if fmt is None else fmt,
            "sh": rng.randrange(32) if sh is None else sh, "lo": lo, "hi": hi,
            "keep": [int(rng.random() < 0.9) for _ in range(L)], "last": int(rng.random() < 0.25)}


def expected(b):
    r, s = vr.alu(b["op"], b["a"], b["b"], b["fmt"], b["sh"], b["lo"], b["hi"])
    s = s & np.array(b["keep"], bool)
    return [int(x) for x in r], [int(x) for x in s]


def make_src_snk(dut, beats, rng, p_in, p_out):
    def drive(b):
        dut.in_a.value = pack(b["a"], 32)
        dut.in_b.value = pack(b["b"], 32)
        dut.in_op.value = b["op"]
        dut.in_fmt.value = b["fmt"]
        dut.in_shamt.value = b["sh"]
        dut.in_lo.value = b["lo"] & 0xFFFFFFFF
        dut.in_hi.value = b["hi"] & 0xFFFFFFFF
        dut.in_keep.value = pack(b["keep"], 1)
        dut.in_last.value = b["last"]

    def junk():
        dut.in_a.value = rng.getrandbits(32 * L)
        dut.in_op.value = rng.getrandbits(4)
        dut.in_fmt.value = rng.getrandbits(2)

    def capture():
        return (unpack(u(dut.out_data), L, 32), unpack(u(dut.out_sat), L, 1, False),
                unpack(u(dut.out_keep), L, 1, False), u(dut.out_last))

    return (Source(dut.in_valid, dut.in_ready, beats, drive, junk, p_in, random.Random(rng.random())),
            Sink(dut.out_valid, dut.out_ready, capture, p_out, random.Random(rng.random())))


def check(beats, got):
    nsat = 0
    for n, (b, g) in enumerate(zip(beats, got)):
        data, sat, keep, last = g
        er, es = expected(b)
        assert keep == b["keep"] and last == b["last"], n
        for i in range(L):
            assert data[i] == er[i], (f"beat {n} lane {i} op={b['op']} fmt={b['fmt']} sh={b['sh']} "
                                      f"a={b['a'][i]} b={b['b'][i]} lo={b['lo']} hi={b['hi']}: "
                                      f"got {data[i]} exp {er[i]}")
        assert sat == es, n
        nsat += sum(es)
    return nsat


async def run_and_check(dut, beats, rng, p_in=0.8, p_out=0.7):
    src, snk = make_src_snk(dut, beats, rng, p_in, p_out)
    cyc = await run_bench(dut, [src], [snk], [len(beats)])
    return check(beats, snk.got), cyc


@cocotb.test()
async def directed_extremes(dut):
    """Every op x fmt x a selection of shamt on extreme operand pairs."""
    rng = random.Random(5)
    await start(dut, INPUTS)
    pairs = [(a, b) for a in EXT for b in EXT]
    beats = []
    for op in range(16):
        for fmt in range(4):
            for sh in ((0, 1, 15, 31) if op in (vr.OP_MUL, vr.OP_SRA, vr.OP_SRAR, vr.OP_SHL) else (0,)):
                for j in range(0, len(pairs), L):
                    grp = pairs[j:j + L]
                    grp += [(0, 0)] * (L - len(grp))
                    beats.append(beat(rng, op, [p[0] for p in grp], [p[1] for p in grp], fmt, sh))
    # rounding halves for MUL / SRAR: a*b = +-(k*2^e + 2^(e-1))
    for e in range(1, 32):
        vals = []
        for k in range(L):
            v = (k * (1 << e) + (1 << (e - 1))) * (1 if k % 2 else -1)
            vals.append(max(min(v, vr.INT32_MAX), vr.INT32_MIN))
        beats.append(beat(rng, vr.OP_SRAR, vals, [0] * L, 2, e))
        beats.append(beat(rng, vr.OP_MUL, vals, [1] * L, 2, e))
        beats.append(beat(rng, vr.OP_MUL, [v >> 1 if abs(v) > 1 else v for v in vals], [2] * L, 2, e))
    nsat, _ = await run_and_check(dut, beats, rng)
    assert u(dut.sat_count) == nsat
    dut._log.info(f"L={L}: {len(beats)} directed beats, {nsat} saturations")


@cocotb.test()
async def random_ops_backpressure(dut):
    rng = random.Random(6)
    await start(dut, INPUTS)
    nb = max(1, N_PER_OP // L)
    beats = []
    for op in list(range(10)) + [10, 15]:
        n = nb if op < 10 else max(1, nb // 10)
        for _ in range(n):
            beats.append(beat(rng, op, [rand_val(rng) for _ in range(L)], [rand_val(rng) for _ in range(L)]))
    rng.shuffle(beats)                           # mixed ops back-to-back in the pipeline
    total = 0
    third = len(beats) // 3
    for k, (pi, po) in enumerate(((1.0, 1.0), (0.85, 0.35), (0.35, 0.9))):
        part = beats[k * third:(k + 1) * third] if k < 2 else beats[2 * third:]
        ns, cyc = await run_and_check(dut, part, rng, pi, po)
        if k == 0:
            assert cyc <= len(part) + 5, cyc   # full throughput
        total += ns
    assert u(dut.sat_count) == total and int(dut.sat_sticky.value) == int(total > 0)
    dut.sat_clr.value = 1
    await FallingEdge(dut.clk)
    dut.sat_clr.value = 0
    await ReadOnly()
    assert u(dut.sat_count) == 0
    dut._log.info(f"L={L}: {len(beats)} random beats ({len(beats) * L} lane-results), {total} saturations")


@cocotb.test()
async def pipeline_depth(dut):
    rng = random.Random(7)
    await start(dut, INPUTS)
    dut.out_ready.value = 1
    b = beat(rng, vr.OP_ADD, [1] * L, [2] * L, 2, 0)
    dut.in_a.value = pack(b["a"], 32); dut.in_b.value = pack(b["b"], 32)
    dut.in_op.value = vr.OP_ADD; dut.in_fmt.value = 2; dut.in_valid.value = 1
    await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    for c in range(1, 10):
        await ReadOnly()
        if dut.out_valid.value == 1:
            break
        await FallingEdge(dut.clk)
    assert c == 3, f"pipeline depth {c}, documented 3"
    assert unpack(u(dut.out_data), L, 32) == [3] * L
