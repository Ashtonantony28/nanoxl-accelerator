"""cocotb bench for rtl/srdhm.sv alone: a new (a, b) pair every cycle, en toggling randomly; after
an enabled edge, y = nanoxl.numerics.srdhm of the pair captured 3 enabled edges earlier (y is the
third register). Operands: full-range random, small, near powers of two (the tile boundaries of
both DSP decompositions), and all pairs of the extremes, including INT32_MIN x INT32_MIN."""
from __future__ import annotations
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl import numerics as nx

N = int(os.environ.get("N_SAMPLES", "60000"))
MIN, MAX = -(1 << 31), (1 << 31) - 1
EXT = [MIN, MIN + 1, -(1 << 30) - 1, -(1 << 30), -(1 << 30) + 1, -(1 << 26), -(1 << 17), -65537, -65536,
       -32768, -2, -1, 0, 1, 2, 32767, 65535, 65536, (1 << 17) - 1, 1 << 17, (1 << 26) - 1, 1 << 26,
       (1 << 30) - 1, 1 << 30, (1 << 30) + 1, MAX - 1, MAX]


def s32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >> 31 else x


def rv(rng):
    r = rng.random()
    if r < 0.4:
        return rng.randrange(MIN, MAX + 1)
    if r < 0.6:
        return rng.randrange(-70000, 70000)
    if r < 0.8:
        k = rng.randrange(31)
        return max(MIN, min(MAX, rng.choice([1, -1]) * (1 << k) + rng.randrange(-3, 4)))
    return rng.choice(EXT)


@cocotb.test()
async def srdhm_random(dut):
    rng = random.Random(11)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pairs = [(a, b) for a in EXT for b in EXT] + [(rv(rng), rv(rng)) for _ in range(N)]
    exp = [int(nx.srdhm(a, b)) for a, b in pairs]
    cap, i, checked, last_en = [], 0, 0, False
    while checked < len(pairs):
        await FallingEdge(dut.clk)
        await ReadOnly()
        if last_en and len(cap) >= 3:            # the edge that just passed was enabled
            k = cap[-3]
            y = s32(int(dut.y.value))
            assert y == exp[k], f"a={pairs[k][0]} b={pairs[k][1]}: got {y} exp {exp[k]}"
            checked += 1
        # drive the next cycle (still after the falling edge, outside ReadOnly)
        await cocotb.triggers.Timer(1, unit="ns")
        en = rng.random() < 0.8
        dut.en.value = int(en)
        a, b = pairs[min(i, len(pairs) - 1)]
        dut.a.value = a & 0xFFFFFFFF
        dut.b.value = b & 0xFFFFFFFF
        if en:
            cap.append(min(i, len(pairs) - 1))
            i += 1
        last_en = en
    dut._log.info(f"srdhm NDSP: {checked} samples")
