"""cocotb tests for rtl/systolic_array.sv (core level, no controller).

Reference: nanoxl.numerics.gemm_i8 / unpack_i4 (contracts C1, C1b).
Env: ARR_N = array size.
"""
import os
import random

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

from nanoxl.numerics import gemm_i8, pack_i4, unpack_i4

N = int(os.environ.get("ARR_N", "4"))
LAT = 2 * N - 1


def pack_bytes(vals, width=8):
    v = 0
    for i, x in enumerate(vals):
        v |= (int(x) & ((1 << width) - 1)) << (i * width)
    return v


def unpack_s32(v):
    out = []
    for c in range(N):
        x = (v >> (32 * c)) & 0xFFFFFFFF
        out.append(x - (1 << 32) if x >> 31 else x)
    return out


class Harness:
    def __init__(self, dut):
        self.dut = dut
        self.cycle = 0
        self.outs = []  # (cycle, tag, row)
        self.ins = {}   # tag -> cycle in which in_valid was sampled

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        await self.reset()
        cocotb.start_soon(self.monitor())

    async def reset(self):
        d = self.dut
        d.rst_n.value = 0
        for s in ("in_valid", "in_a", "in_sel", "in_tag", "w_wr_en", "w_wr_bank", "w_wr_row",
                  "w_wr_int4", "w_wr_data"):
            getattr(d, s).value = 0
        for _ in range(3):
            await FallingEdge(d.clk)
        d.rst_n.value = 1

    async def monitor(self):
        # sample at the rising edge: inputs are what the DUT samples, outputs are the
        # registered values of the cycle that ends at this edge
        d = self.dut
        while True:
            await RisingEdge(d.clk)
            self.cycle += 1
            if d.rst_n.value == 1 and int(d.in_valid.value):
                self.ins[int(d.in_tag.value)] = self.cycle
            if d.rst_n.value == 1 and int(d.out_valid.value):
                self.outs.append((self.cycle, int(d.out_tag.value), unpack_s32(int(d.out_psum.value))))

    async def tick(self):
        await FallingEdge(self.dut.clk)

    async def load(self, bank, w, int4=False):
        """w: int8 [K<=N, N]. int4: values in [-8,7] sent packed (C1b) in the low N*4 bits."""
        d = self.dut
        k = w.shape[0]
        packed = pack_i4(w) if int4 else None
        for r in range(N):
            if int4:
                data = pack_bytes(packed[r], 8) if r < k else 0
            else:
                data = pack_bytes(w[r]) if r < k else 0
            d.w_wr_en.value = 1
            d.w_wr_bank.value = bank
            d.w_wr_row.value = r
            d.w_wr_int4.value = int(int4)
            d.w_wr_data.value = data
            await self.tick()
        d.w_wr_en.value = 0

    async def stream(self, a, sel, tag0=0, bubble=0.0, rng=None):
        """a: int8 [M, N]. Returns the list of tags."""
        d = self.dut
        issued = []
        for m in range(a.shape[0]):
            while rng is not None and rng.random() < bubble:
                d.in_valid.value = 0
                d.in_a.value = rng.getrandbits(8 * N)  # must be ignored
                await self.tick()
            d.in_valid.value = 1
            d.in_a.value = pack_bytes(a[m])
            d.in_sel.value = sel
            d.in_tag.value = (tag0 + m) & 0xFF
            issued.append((tag0 + m) & 0xFF)
            await self.tick()
        d.in_valid.value = 0
        return issued

    async def drain(self):
        for _ in range(LAT + 2):
            await self.tick()


def check(h, issued, expect):
    got = {t: (c, row) for c, t, row in h.outs}
    assert len(issued) == len(expect)
    for tag, row in zip(issued, expect):
        assert tag in got, f"missing output tag {tag}"
        c, r = got[tag]
        ci = h.ins[tag]
        assert c - ci == LAT, f"latency {c - ci} != {LAT}"
        assert r == [int(x) for x in row], (tag, r, list(row))


@cocotb.test()
async def array_random_int8(dut):
    """Random tiles, both banks, ragged K (zero rows), random input bubbles; exact latency."""
    rng = random.Random(10 + N)
    nrng = np.random.default_rng(10 + N)
    h = Harness(dut)
    await h.start()
    tag = 0
    for t in range(6):
        bank = t & 1
        k = rng.randint(1, N)
        m = rng.randint(1, 3 * N)
        w = nrng.integers(-128, 128, (k, N), dtype=np.int8)
        a = nrng.integers(-128, 128, (m, k), dtype=np.int8)
        await h.load(bank, w)
        a_full = np.zeros((m, N), np.int8)
        a_full[:, :k] = a
        h.outs.clear()
        issued = await h.stream(a_full, bank, tag, bubble=0.3, rng=rng)
        await h.drain()
        check(h, issued, gemm_i8(a, w))
        tag += m
    assert int(dut.hazard_seen.value) == 0


@cocotb.test()
async def array_extremes(dut):
    """-128 x -128 over the full K (largest positive sum) and -128 x 127 (most negative)."""
    h = Harness(dut)
    await h.start()
    w = np.full((N, N), -128, np.int8)
    w[:, 1::2] = 127
    await h.load(0, w)
    a = np.array([[-128] * N, [127] * N, [-128, 127] * (N // 2)], np.int8)
    issued = await h.stream(a, 0)
    await h.drain()
    exp = gemm_i8(a, w)
    assert exp[0, 0] == N * 16384
    check(h, issued, exp)


@cocotb.test()
async def array_int4(dut):
    """W4A8: packed INT4 weights (C1b) sign-extended by the loader; equals gemm(unpack_i4)."""
    nrng = np.random.default_rng(20 + N)
    h = Harness(dut)
    await h.start()
    for bank in (1, 0):
        w4 = nrng.integers(-8, 8, (N, N), dtype=np.int8)
        w4[0, :] = -8
        w4[1, :] = 7
        await h.load(bank, w4, int4=True)
        a = nrng.integers(-128, 128, (2 * N, N), dtype=np.int8)
        a[0, :] = -128
        h.outs.clear()
        issued = await h.stream(a, bank, 0)
        await h.drain()
        check(h, issued, gemm_i8(a, unpack_i4(pack_i4(w4))))


@cocotb.test()
async def array_overlap_load(dut):
    """Load bank 1 while bank 0 rows stream (legal by the bank-reuse rule); the in-flight
    rows still see the old bank and the hazard checker stays quiet."""
    nrng = np.random.default_rng(30 + N)
    h = Harness(dut)
    await h.start()
    w0 = nrng.integers(-128, 128, (N, N), dtype=np.int8)
    w1 = nrng.integers(-128, 128, (N, N), dtype=np.int8)
    await h.load(0, w0)
    a0 = nrng.integers(-128, 128, (N, N), dtype=np.int8)
    a1 = nrng.integers(-128, 128, (N, N), dtype=np.int8)
    stream0 = cocotb.start_soon(h.stream(a0, 0, 0))
    await h.load(1, w1)  # concurrently with stream0
    issued0 = await stream0
    issued1 = await h.stream(a1, 1, 100)
    await h.drain()
    check(h, issued0 + issued1, np.concatenate([gemm_i8(a0, w0), gemm_i8(a1, w1)]))
    assert int(dut.hazard_seen.value) == 0


@cocotb.test()
async def array_hazard_checker_fires(dut):
    """Sanity check of the simulation hazard checker: rewriting bank 0 while bank-0 rows
    are still inside the array must set hazard_seen."""
    nrng = np.random.default_rng(40)
    h = Harness(dut)
    await h.start()
    w0 = nrng.integers(-128, 128, (N, N), dtype=np.int8)
    await h.load(0, w0)
    a0 = nrng.integers(-128, 128, (N, N), dtype=np.int8)
    stream0 = cocotb.start_soon(h.stream(a0, 0, 0))
    await h.tick()
    await h.load(0, w0)
    await stream0
    await h.drain()
    assert int(dut.hazard_seen.value) == 1


@cocotb.test()
async def array_reset_mid_stream(dut):
    """Reset while rows are in flight: no stale outputs afterwards, next GEMM exact."""
    nrng = np.random.default_rng(50 + N)
    h = Harness(dut)
    await h.start()
    w = nrng.integers(-128, 128, (N, N), dtype=np.int8)
    await h.load(0, w)
    a = nrng.integers(-128, 128, (2 * N, N), dtype=np.int8)
    stream = cocotb.start_soon(h.stream(a, 0, 0))
    for _ in range(N + 1):
        await h.tick()
    stream.cancel()
    await h.reset()
    h.outs.clear()
    for _ in range(LAT + 2):
        await h.tick()
    assert not h.outs, "outputs after reset"
    # weights were cleared by reset; reload and run
    await h.load(1, w)
    issued = await h.stream(a, 1, 7)
    await h.drain()
    check(h, issued, gemm_i8(a, w))
