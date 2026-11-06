"""cocotb tests for rtl/scratchpad.sv: scratchpad (1R1W, fixed latency) and spad_fifo.

Env: SP_WIDTH, SP_DEPTH, SP_LAT (scratchpad) / FIFO_W, FIFO_DEPTH (spad_fifo); SP_TOP selects.
"""
import os
import random
from collections import deque

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

TOP = os.environ.get("SP_TOP", "scratchpad")


async def setup(dut, sigs):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for s in sigs:
        getattr(dut, s).value = 0
    for _ in range(3):
        await FallingEdge(dut.clk)
    dut.rst_n.value = 1


@cocotb.test(skip=(TOP != "scratchpad"))
async def scratchpad_random(dut):
    """Random reads and writes (never the same address in one cycle) vs a dict model;
    rd_valid/rd_data appear exactly RD_LAT cycles after rd_en."""
    width, depth, lat = (int(os.environ[k]) for k in ("SP_WIDTH", "SP_DEPTH", "SP_LAT"))
    rng = random.Random(depth * 7 + lat)
    await setup(dut, ["wr_en", "wr_addr", "wr_data", "rd_en", "rd_addr"])
    model = {}
    pend = deque([None] * (lat - 1))  # expected rd_data per cycle (None = no read)
    nread = 0
    # fill every word first so all later reads are defined
    for a in range(depth):
        v = rng.getrandbits(width)
        dut.wr_en.value, dut.wr_addr.value, dut.wr_data.value = 1, a, v
        model[a] = v
        await FallingEdge(dut.clk)
    dut.wr_en.value = 0
    for _ in range(3000):
        we, re = rng.random() < 0.5, rng.random() < 0.6
        wa, ra = rng.randrange(depth), rng.randrange(depth)
        if we and re and wa == ra:
            we = False
        wv = rng.getrandbits(width)
        dut.wr_en.value, dut.wr_addr.value, dut.wr_data.value = int(we), wa, wv
        dut.rd_en.value, dut.rd_addr.value = int(re), ra
        pend.append(model[ra] if re else None)
        if we:
            model[wa] = wv
        await FallingEdge(dut.clk)
        exp = pend.popleft()
        # after this edge the read issued `lat` cycles ago is visible
        assert int(dut.rd_valid.value) == int(exp is not None)
        if exp is not None:
            assert int(dut.rd_data.value) == exp
            nread += 1
    assert nread > 1000


@cocotb.test(skip=(TOP != "spad_fifo"))
async def spad_fifo_random(dut):
    """Random push (only when !full) and random pop backpressure; order and data preserved,
    no overflow, count consistent. Then sustained 1 push + 1 pop per cycle."""
    w, depth = int(os.environ["FIFO_W"]), int(os.environ["FIFO_DEPTH"])
    rng = random.Random(depth)
    await setup(dut, ["push", "din", "out_ready"])
    model = deque()
    npop = 0
    for phase in range(3):
        p_push, p_pop = [(0.7, 0.3), (0.3, 0.8), (1.0, 1.0)][phase]
        for _ in range(1500):
            full = int(dut.full.value)
            push = (not full) and rng.random() < p_push
            v = rng.getrandbits(w)
            dut.push.value, dut.din.value = int(push), v
            ready = rng.random() < p_pop
            dut.out_ready.value = int(ready)
            ov = int(dut.out_valid.value)
            od = int(dut.out_data.value) if ov else None
            await RisingEdge(dut.clk)
            if ov and ready:
                assert od == model.popleft()
                npop += 1
            if push:
                model.append(v)
            await FallingEdge(dut.clk)
            assert int(dut.count.value) == len(model)
            assert int(dut.overflow.value) == 0
            assert len(model) <= depth
    # sustained throughput: with push and pop every cycle the FIFO keeps flowing
    assert npop > 2000
    dut.push.value = 0
    dut.out_ready.value = 1
    for _ in range(depth + 5):
        if int(dut.out_valid.value):
            assert int(dut.out_data.value) == model.popleft()
        await FallingEdge(dut.clk)
    assert not model


@cocotb.test(skip=(TOP != "spad_fifo"))
async def spad_fifo_throughput(dut):
    """Push every cycle with out_ready=1: after a 3-cycle latency one pop per cycle."""
    await setup(dut, ["push", "din", "out_ready"])
    dut.out_ready.value = 1
    got = []
    for i in range(100):
        dut.push.value, dut.din.value = 1, i
        if int(dut.out_valid.value):
            got.append(int(dut.out_data.value))
        await FallingEdge(dut.clk)
    dut.push.value = 0
    assert got == list(range(len(got))) and len(got) >= 96, got[:5]
    assert int(dut.count.value) <= 4
