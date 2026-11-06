"""cocotb bench for rtl/requant.sv against nanoxl.numerics (contract C2)."""
from __future__ import annotations
import random

import cocotb
import numpy as np
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl import numerics as nx
from vec_common_cocotb import Sink, Source, env_int, pack, run_bench, start, u, unpack

L = env_int("L", 4)
N_RANDOM = env_int("N_RANDOM", 20000)      # random lane-results per test
I32 = (nx.INT32_MIN, nx.INT32_MAX)


def ref_lane(acc, bias, mult, shift, out16):
    """C2 composed from the contract primitives; valid for any int32 mult (the contract
    function nx.requant additionally asserts mult >= 2^30). Returns (out, saturated)."""
    x = ((acc + bias + (1 << 31)) % (1 << 32)) - (1 << 31)
    y = int(nx.rdivpot(nx.srdhm(x, mult), shift))
    lo, hi = (-32768, 32767) if out16 else (-128, 127)
    return min(max(y, lo), hi), (y < lo or y > hi)


def check_against_contract(beats):
    """Cross-check ref_lane against nx.requant itself on the legal-mult lanes."""
    acc, bias, mult, shift, o16 = [], [], [], [], []
    for b in beats:
        for i in range(L):
            if b["mult"][i] >= (1 << 30):
                acc.append(b["acc"][i]); bias.append(b["bias"][i]); mult.append(b["mult"][i])
                shift.append(b["shift"][i]); o16.append(b["o16"])
    o16 = np.array(o16, bool)
    for bits, sel in ((8, ~o16), (16, o16)):
        if sel.any():
            ref = nx.requant(np.array(acc)[sel], np.array(mult)[sel], np.array(shift)[sel],
                             np.array(bias)[sel], bits)
            mine = [ref_lane(a, b_, m, s, bits == 16)[0] for a, b_, m, s in
                    zip(np.array(acc)[sel], np.array(bias)[sel], np.array(mult)[sel], np.array(shift)[sel])]
            assert list(ref) == mine


def make_beat(rng, lanes, o16, last=False, keep=None):
    acc, bias, mult, shift = zip(*lanes)
    return {"acc": list(acc), "bias": list(bias), "mult": list(mult), "shift": list(shift),
            "o16": o16, "last": last, "keep": keep if keep is not None else [1] * L}


def rand_lane(rng):
    shift = rng.randrange(32)
    r = rng.random()
    if r < 0.4:      # in-range-ish: |acc| ~ 2^(shift+8..16)
        mag = 1 << min(31, shift + rng.randrange(4, 18))
        acc = rng.randrange(-mag, mag)
    elif r < 0.7:
        acc = rng.randrange(*I32)
    else:
        acc = rng.choice([nx.INT32_MIN, nx.INT32_MAX, 0, -1, 1]) + rng.randrange(-3, 4)
        acc = max(min(acc, nx.INT32_MAX), nx.INT32_MIN)
    rb = rng.random()
    bias = 0 if rb < 0.3 else rng.randrange(-(1 << 16), 1 << 16) if rb < 0.6 else rng.randrange(*I32)
    rm = rng.random()
    mult = (1 << 30) if rm < 0.1 else nx.INT32_MAX if rm < 0.2 else rng.randrange(1 << 30, 1 << 31)
    return acc, bias, mult, shift


def directed_lanes(rng):
    """Corner cases: extremes, mult=2^30 / 2^31-1, shift 0 / 31, bias wraparound,
    exact halves in SRDHM (mult=2^30, odd x) and in RoundingDivideByPOT (both signs)."""
    out = []
    ext = [nx.INT32_MIN, nx.INT32_MIN + 1, -1, 0, 1, nx.INT32_MAX - 1, nx.INT32_MAX]
    for acc in ext:
        for bias in [0, nx.INT32_MAX, nx.INT32_MIN, -1, 1]:
            for mult in [1 << 30, nx.INT32_MAX]:
                for shift in [0, 1, 30, 31]:
                    out.append((acc, bias, mult, shift))
    # SRDHM half: mult = 2^30, odd x -> exact half, rounds toward +inf.
    for x in list(range(-41, 42, 2)) + [nx.INT32_MAX, nx.INT32_MIN + 1]:
        out.append((x, 0, 1 << 30, 0))
    # RoundingDivideByPOT halves: with mult = 2^31-1, srdhm(x) = x for |x| < 2^30
    # (error < 1/2), so x = +-(k*2^e + 2^(e-1)) hits exact halves.
    for e in range(1, 31):
        for k in (0, 1, 2, 5):
            v = k * (1 << e) + (1 << (e - 1))
            if v < (1 << 30):
                for x in (v, -v):
                    assert int(nx.srdhm(x, nx.INT32_MAX)) == x
                    out.append((x, 0, nx.INT32_MAX, e))
    # bias wraparound in both directions
    out += [(nx.INT32_MAX, 1, nx.INT32_MAX, 24), (nx.INT32_MIN, -1, nx.INT32_MAX, 24),
            (nx.INT32_MAX, nx.INT32_MAX, 1 << 30, 31), (nx.INT32_MIN, nx.INT32_MIN, 1 << 30, 0)]
    # illegal (out-of-contract) mults incl. INT32_MIN*INT32_MIN SRDHM overflow
    out += [(nx.INT32_MIN, 0, nx.INT32_MIN, s) for s in (0, 1, 31)]
    out += [(rng.randrange(*I32), 0, rng.randrange(*I32), rng.randrange(32)) for _ in range(200)]
    return out


def lanes_to_beats(rng, lanes, both_modes):
    beats = []
    while len(lanes) % L:
        lanes.append(rand_lane(rng))
    for j in range(0, len(lanes), L):
        grp = lanes[j:j + L]
        modes = (False, True) if both_modes else (rng.random() < 0.5,)
        for o16 in modes:
            beats.append(make_beat(rng, grp, o16))
    return beats


def setup_bench(dut, beats, rng, p_in, p_out):
    def drive(b):
        dut.in_acc.value = pack(b["acc"], 32)
        dut.in_bias.value = pack(b["bias"], 32)
        dut.in_mult.value = pack(b["mult"], 32)
        dut.in_shift.value = pack(b["shift"], 5)
        dut.in_keep.value = pack(b["keep"], 1)
        dut.in_last.value = int(b["last"])
        dut.in_out16.value = int(b["o16"])

    def junk():
        dut.in_acc.value = rng.getrandbits(32 * L)
        dut.in_mult.value = rng.getrandbits(32 * L)
        dut.in_out16.value = rng.getrandbits(1)

    def capture():
        return (unpack(u(dut.out_data), L, 16), unpack(u(dut.out_sat), L, 1, False),
                unpack(u(dut.out_keep), L, 1, False), int(dut.out_last.value))

    src = Source(dut.in_valid, dut.in_ready, beats, drive, junk, p_in, random.Random(rng.random()))
    snk = Sink(dut.out_valid, dut.out_ready, capture, p_out, random.Random(rng.random()))
    return src, snk


def check(beats, got):
    nsat = 0
    for n, (b, g) in enumerate(zip(beats, got)):
        data, sat, keep, last = g
        assert keep == b["keep"] and last == int(b["last"]), n
        for i in range(L):
            exp, s = ref_lane(b["acc"][i], b["bias"][i], b["mult"][i], b["shift"][i], b["o16"])
            s = s and bool(b["keep"][i])
            assert data[i] == exp, (f"beat {n} lane {i}: acc={b['acc'][i]} bias={b['bias'][i]} "
                                    f"mult={b['mult'][i]} shift={b['shift'][i]} o16={b['o16']}: "
                                    f"got {data[i]} exp {exp}")
            assert sat[i] == int(s), (n, i)
            nsat += s
    return nsat


INPUTS = ["in_valid", "in_acc", "in_bias", "in_mult", "in_shift", "in_keep", "in_last",
          "in_out16", "out_ready", "sat_clr"]


@cocotb.test()
async def directed_corners(dut):
    rng = random.Random(11)
    await start(dut, INPUTS)
    beats = lanes_to_beats(rng, directed_lanes(rng), both_modes=True)
    for b in beats:                      # random keep / last sideband
        b["keep"] = [rng.random() < 0.9 for _ in range(L)]
        b["last"] = rng.random() < 0.2
    check_against_contract(beats)
    src, snk = setup_bench(dut, beats, rng, 0.8, 0.7)
    await run_bench(dut, [src], [snk], [len(beats)])
    nsat = check(beats, snk.got)
    assert u(dut.sat_count) == nsat and int(dut.sat_sticky.value) == int(nsat > 0)
    dut._log.info(f"L={L}: {len(beats)} directed beats ({len(beats) * L} lanes), {nsat} saturations")


@cocotb.test()
async def random_backpressure(dut):
    rng = random.Random(22)
    await start(dut, INPUTS)
    lanes = [rand_lane(rng) for _ in range(N_RANDOM)]
    beats = lanes_to_beats(rng, lanes, both_modes=False)
    check_against_contract(beats[:500])
    # three phases: full throughput, heavy backpressure, sparse input
    third = len(beats) // 3
    total = 0
    for k, (pi, po) in enumerate(((1.0, 1.0), (0.9, 0.3), (0.3, 0.9))):
        part = beats[k * third:(k + 1) * third] if k < 2 else beats[2 * third:]
        src, snk = setup_bench(dut, part, rng, pi, po)
        cyc = await run_bench(dut, [src], [snk], [len(part)])
        if k == 0:   # full throughput: one beat per cycle after the 6-cycle fill
            assert cyc <= len(part) + 8, cyc
        total += check(part, snk.got)
    assert u(dut.sat_count) == total
    # sat_clr clears the counter and the sticky flag
    dut.sat_clr.value = 1
    await FallingEdge(dut.clk)
    dut.sat_clr.value = 0
    await FallingEdge(dut.clk)
    assert u(dut.sat_count) == 0 and int(dut.sat_sticky.value) == 0
    dut._log.info(f"L={L}: {len(beats)} random beats ({len(beats) * L} lanes), {total} saturations")


@cocotb.test()
async def pipeline_depth(dut):
    """Exactly 6 cycles from input handshake to output valid, with out_ready high."""
    rng = random.Random(3)
    await start(dut, INPUTS)
    dut.out_ready.value = 1
    b = make_beat(rng, [rand_lane(rng) for _ in range(L)], True)
    dut.in_acc.value = pack(b["acc"], 32); dut.in_bias.value = pack(b["bias"], 32)
    dut.in_mult.value = pack(b["mult"], 32); dut.in_shift.value = pack(b["shift"], 5)
    dut.in_keep.value = (1 << L) - 1; dut.in_out16.value = 1
    dut.in_valid.value = 1
    await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    for c in range(1, 10):
        await ReadOnly()
        if dut.out_valid.value == 1:
            break
        await FallingEdge(dut.clk)
    assert c == 6, f"pipeline depth {c}, documented 6"
