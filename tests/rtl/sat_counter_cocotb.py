"""cocotb bench for rtl/sat_counter.sv with a narrow W so saturation is reachable."""
from __future__ import annotations
import random

import cocotb
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from vec_common_cocotb import env_int, start, u

N = env_int("N", 4)
W = env_int("W", 6)


@cocotb.test()
async def random_events(dut):
    rng = random.Random(1)
    await start(dut, ["clr", "en", "inc"])
    count, sticky, hit_max = 0, 0, 0
    for cyc in range(4000):
        clr = int(rng.random() < 0.01)
        en = int(rng.random() < 0.7)
        inc = rng.getrandbits(N) if rng.random() < 0.6 else 0
        dut.clr.value, dut.en.value, dut.inc.value = clr, en, inc
        await RisingEdge(dut.clk)
        pc = bin(inc).count("1") if en else 0
        count = min((0 if clr else count) + pc, (1 << W) - 1)
        sticky = int((0 if clr else sticky) or pc > 0)
        hit_max += count == (1 << W) - 1
        await ReadOnly()
        assert u(dut.count) == count and u(dut.sticky) == sticky, (cyc, u(dut.count), count)
        await FallingEdge(dut.clk)
    assert hit_max > 100, "saturation was never exercised"
