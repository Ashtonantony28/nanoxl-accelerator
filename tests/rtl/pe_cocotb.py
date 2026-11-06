"""cocotb tests for rtl/pe.sv: exhaustive INT8 x INT8 over both weight banks."""
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

M32 = (1 << 32) - 1


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


async def reset(dut):
    dut.rst_n.value = 0
    for s in ("a_in", "a_v_in", "a_sel_in", "psum_in", "w_wr_en", "w_wr_bank", "w_wr_data"):
        getattr(dut, s).value = 0
    for _ in range(2):
        await FallingEdge(dut.clk)
    dut.rst_n.value = 1


@cocotb.test()
async def pe_exhaustive(dut):
    """Every (a, w) pair in [-128,127]^2 through a randomly selected bank, random psum_in
    including extremes, and random concurrent writes to either bank (a write lands on the
    next cycle, so it must not affect the product of the cycle it happens in).
    Checks psum_out = psum_in + a*w[sel] exactly and the a/v/sel pass-through."""
    rng = random.Random(1)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    bank_w = [0, 0]
    extremes = [-(1 << 31) + (1 << 14), (1 << 31) - 1 - (1 << 14), 0, -1, 1 << 26, -(1 << 26)]
    seen = set()
    for w in range(-128, 128):
        # load w into both banks (two cycles), then sweep all activations
        for bank in (0, 1):
            await FallingEdge(dut.clk)
            dut.a_v_in.value = 0
            dut.w_wr_en.value = 1
            dut.w_wr_bank.value = bank
            dut.w_wr_data.value = w & 0xFF
            bank_w[bank] = w
        await FallingEdge(dut.clk)
        dut.w_wr_en.value = 0
        for a in range(-128, 128):
            sel = rng.randint(0, 1)
            p = rng.choice(extremes) if rng.random() < 0.1 else rng.randint(-(1 << 30), 1 << 30)
            dut.a_in.value = a & 0xFF
            dut.a_v_in.value = 1
            dut.a_sel_in.value = sel
            dut.psum_in.value = p & M32
            # occasionally overwrite the *selected* bank this cycle and restore it next
            # cycle; the product must use the old value of the selected bank.
            if rng.random() < 0.1:
                dut.w_wr_en.value = 1
                dut.w_wr_bank.value = sel
                dut.w_wr_data.value = rng.randint(-128, 127) & 0xFF
                clobber = True
            else:
                dut.w_wr_en.value = 0
                clobber = False
            cur = p + a * bank_w[sel]
            await FallingEdge(dut.clk)
            assert dut.psum_out.value.to_signed() == s32(cur), (a, w, sel, p)
            assert dut.a_out.value.to_signed() == a
            assert int(dut.a_sel_out.value) == sel and int(dut.a_v_out.value) == 1
            seen.add((a, bank_w[sel]))
            if clobber:  # restore (costs one idle cycle)
                dut.a_v_in.value = 0
                dut.w_wr_en.value = 1
                dut.w_wr_bank.value = sel
                dut.w_wr_data.value = w & 0xFF
                await FallingEdge(dut.clk)
                dut.w_wr_en.value = 0
    assert len(seen) == 65536, len(seen)


@cocotb.test()
async def pe_bank_select(dut):
    """Bank 0 and bank 1 hold different weights; sel picks the right one every cycle."""
    rng = random.Random(2)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    for _ in range(200):
        wv = [rng.randint(-128, 127), rng.randint(-128, 127)]
        for bank in (0, 1):
            await FallingEdge(dut.clk)
            dut.w_wr_en.value = 1
            dut.w_wr_bank.value = bank
            dut.w_wr_data.value = wv[bank] & 0xFF
        await FallingEdge(dut.clk)
        dut.w_wr_en.value = 0
        for _ in range(4):
            a, sel = rng.randint(-128, 127), rng.randint(0, 1)
            dut.a_in.value = a & 0xFF
            dut.a_sel_in.value = sel
            dut.psum_in.value = 0
            await FallingEdge(dut.clk)
            assert dut.psum_out.value.to_signed() == a * wv[sel]


@cocotb.test()
async def pe_reset_clears(dut):
    """Reset mid-operation clears psum, pass-through regs and both weight banks."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    await FallingEdge(dut.clk)
    dut.w_wr_en.value = 1
    dut.w_wr_data.value = 0x7F
    await FallingEdge(dut.clk)
    dut.w_wr_en.value = 0
    dut.a_in.value = 0x55
    dut.a_v_in.value = 1
    dut.psum_in.value = 1234
    await FallingEdge(dut.clk)
    assert dut.psum_out.value.to_signed() == 1234 + 0x55 * 0x7F
    dut.rst_n.value = 0
    await FallingEdge(dut.clk)
    assert int(dut.psum_out.value) == 0 and int(dut.a_v_out.value) == 0
    dut.rst_n.value = 1
    dut.psum_in.value = 0
    await FallingEdge(dut.clk)
    assert int(dut.psum_out.value) == 0  # weight bank 0 was cleared
