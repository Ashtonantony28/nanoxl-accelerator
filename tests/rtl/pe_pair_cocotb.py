"""cocotb wrapper for tests/rtl/tb_pe_pair.sv: clock, reset, wait for the self-checking bench."""
from __future__ import annotations
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

STEP = int(os.environ.get("PP_STEP", "1"))


@cocotb.test()
async def pe_pair_exhaustive(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    n = ((1 << 24) + STEP - 1) // STEP
    await RisingEdge(dut.done)
    await FallingEdge(dut.clk)
    checked, errors = int(dut.checked.value), int(dut.errors.value)
    dut._log.info(f"pe_pair: {checked} cases (STEP {STEP}), {errors} errors")
    assert checked == n, (checked, n)
    assert errors == 0
