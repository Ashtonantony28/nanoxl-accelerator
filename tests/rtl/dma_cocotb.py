"""cocotb tests for rtl/dma_engine.sv: DMA_LOAD / DMA_STORE against cocotbext-axi AxiRam and a
Python scratchpad model, with random AXI and scratchpad backpressure / latency.

Env: DMA_MAXB (max burst beats the DUT was built with), DMA_NCMD (random commands).
Reference: ctrl_common_cocotb.dma_apply (byte-granular geometry of docs/control.md).
"""
import os
import random
from collections import deque

import cocotb
from cocotbext.axi import AxiBus, AxiRam

from ctrl_common_cocotb import AxiMonitor, Bench, SpadModel, dma_apply, set_axi_pauses, word_int
from nanoxl import isa

MAXB = int(os.environ.get("DMA_MAXB", "256"))
NCMD = int(os.environ.get("DMA_NCMD", "60"))
WIN_BASE = 0x4000_0000
WIN = 1 << 20


class CmdDriver:
    def __init__(self, dut, rng, p=0.7):
        self.d, self.rng, self.p = dut, rng, p
        self.q = deque()
        self.sent = self.done = 0
        self.cur = None

    def reset(self):
        self.q.clear()
        self.cur = None

    def drive(self):
        if self.cur is None and self.q and self.rng.random() < self.p:
            self.cur = self.q.popleft()
        self.d.cmd_valid.value = int(self.cur is not None)
        self.d.cmd_word.value = word_int(self.cur) if self.cur is not None else 0

    def sample(self):
        if self.cur is not None and int(self.d.cmd_ready.value):
            self.cur = None
            self.sent += 1
        if int(self.d.done.value):
            self.done += 1


class TB:
    def __init__(self, dut, seed, p_axi=0.3, p_sp=0.7):
        self.dut = dut
        self.rng = random.Random(seed)
        self.bench = Bench(dut)
        self.ram = AxiRam(AxiBus.from_prefix(dut, "m_axi"), dut.clk, dut.rst_n,
                          reset_active_level=False, size=1 << 32)
        if p_axi:
            set_axi_pauses(self.ram, self.rng, p_axi)
        self.cmd = CmdDriver(dut, self.rng)
        self.sp = SpadModel(dut, self.rng, p_ready=p_sp)
        self.mon = AxiMonitor(dut, "m_axi_", max_beats=MAXB)
        self.bench.agents = [self.cmd, self.sp, self.mon]
        # shadow state
        self.ddr = bytearray(self.rng.randbytes(WIN))
        self.ram.write(WIN_BASE, bytes(self.ddr))
        self.spad = {b: bytearray(1 << 21) for b in range(8)}
        for b in range(5):
            init = self.rng.randbytes(1 << 21)
            self.spad[b][:] = init
            self.sp.mem[b][:] = init
        self.bytes = 0

    async def start(self):
        self.dut.rst_n.value = 0
        self.dut.quiesce.value = 0
        self.dut.cmd_valid.value = 0
        self.dut.cmd_word.value = 0
        await self.bench.start_clock()
        await self.bench.reset()

    def ddr_read(self, a, n):
        if WIN_BASE <= a and a + n <= WIN_BASE + WIN:
            return bytes(self.ddr[a - WIN_BASE:a - WIN_BASE + n])
        return self.ram.read(a, n)   # outside the window: whatever the RAM holds (zeros)

    def ddr_write(self, a, data):
        if WIN_BASE <= a and a + len(data) <= WIN_BASE + WIN:
            self.ddr[a - WIN_BASE:a - WIN_BASE + len(data)] = data
        else:   # split where the 32-bit address space wraps
            n0 = min(len(data), (1 << 32) - a)
            self.outside.append((a, data[:n0]))
            if n0 < len(data):
                self.outside.append((0, data[n0:]))

    def expect(self, c):
        self.outside = []
        dma_apply(c, self.ddr_read, self.ddr_write, self.spad)
        self.bytes += c.f["rows"] * c.f["row_bytes"]
        return self.outside

    async def run_cmds(self, cmds, limit=400_000):
        outside = []
        for c in cmds:
            self.cmd.q.append(c)
            outside += self.expect(c)
        target = self.cmd.done + len(cmds)
        await self.bench.run_until(lambda: self.cmd.done >= target and not self.cmd.q and self.cmd.cur is None,
                                   limit, "DMA done")
        for _ in range(3):
            await self.bench.step()
        assert self.cmd.done == target, (self.cmd.done, target)
        assert int(self.dut.axi_idle.value) == 1 and int(self.dut.cmd_ready.value) == 1
        return outside

    def check(self, outside=()):
        got = self.ram.read(WIN_BASE, WIN)
        if got != bytes(self.ddr):
            i = next(i for i in range(WIN) if got[i] != self.ddr[i])
            raise AssertionError(f"DDR mismatch at {WIN_BASE + i:#x}: got {got[i]:#x} exp {self.ddr[i]:#x}")
        for a, data in outside:
            assert self.ram.read(a, len(data)) == data, hex(a)
        for b in range(5):
            if self.sp.mem[b] != self.spad[b]:
                i = next(i for i in range(1 << 21) if self.sp.mem[b][i] != self.spad[b][i])
                raise AssertionError(f"spad {b} mismatch at {i:#x}: got {self.sp.mem[b][i]:#x} "
                                     f"exp {self.spad[b][i]:#x}")


def rand_dma(rng, op=None):
    op = op or rng.choice(["DMA_LOAD", "DMA_STORE"])
    u = rng.random()
    rb = rng.randint(1, 48) if u < 0.4 else rng.randint(49, 700) if u < 0.85 else rng.randint(3000, 9000)
    rows = rng.choice([1, 1, 2, 3, 4, rng.randint(5, 12)])
    if rb > 3000:
        rows = rng.randint(1, 3)
    ds = 0 if rng.random() < 0.4 else rb + rng.randint(0, 100)
    if rng.random() < 0.05 and rows > 1:
        ds = rng.randint(1, rb)              # overlapping DDR rows (legal; later rows win)
    span = (rows - 1) * (ds or rb) + rb
    if rng.random() < 0.3:                   # start just below a 4 KiB boundary
        k = rng.randint(1, (WIN - span) // 4096)
        off = k * 4096 - rng.randint(1, 40)
    else:
        off = rng.randint(0, WIN - span)
    off = max(0, min(off, WIN - span))
    lines = -(-rb // 16)
    ss = 0 if rng.random() < 0.4 else min(4095, lines + rng.randint(0, 3))
    if ss and ss * 16 < rb:
        ss = 0
    return isa.C(op, chan=rng.randint(0, 2), sp_buf=rng.randint(0, 4),
                 sp_addr=rng.randrange(1 << 21), ddr_addr=WIN_BASE + off, rows=rows, row_bytes=rb,
                 ddr_stride=ds, sp_stride=ss)


@cocotb.test()
async def dma_random(dut):
    """Random LOAD/STORE geometry (byte-granular addresses, strides and lengths), random AXI
    and scratchpad backpressure; DDR window and every scratchpad byte compared."""
    tb = TB(dut, 1)
    await tb.start()
    rng = random.Random(2)
    left = NCMD
    while left:
        n = min(left, rng.randint(1, 4))
        cmds = [rand_dma(rng) for _ in range(n)]
        out = await tb.run_cmds(cmds)
        tb.check(out)
        left -= n
    dut._log.info("peak outstanding reads %d writes %d, %d AR %d AW bursts",
                  tb.mon.peak_rd, tb.mon.peak_wr, len(tb.mon.ars), len(tb.mon.aws))
    assert tb.mon.peak_rd >= 2, "expected multiple outstanding read bursts"


@cocotb.test()
async def dma_4k_and_bursts(dut):
    """Directed: 4 KiB splits, maximum-length bursts, single bytes at every offset."""
    tb = TB(dut, 3, p_axi=0.0, p_sp=1.0)
    await tb.start()
    B = WIN_BASE
    # one row of 64 bytes starting 8 bytes below a 4 KiB boundary: beats 0xFF0, then 0x000..0x030
    c = isa.C("DMA_LOAD", sp_buf=0, sp_addr=0x100, ddr_addr=B + 0x1FF8, rows=1, row_bytes=64)
    tb.mon.ars.clear()
    await tb.run_cmds([c])
    tb.check()
    assert tb.mon.ars == [(B + 0x1FF0, 1), (B + 0x2000, 4)], tb.mon.ars
    # 3 x 4 KiB-aligned rows of 8 KiB: bursts of min(MAXB, 256) beats, never crossing 4 KiB
    tb.mon.ars.clear()
    c = isa.C("DMA_LOAD", sp_buf=1, sp_addr=0x20000, ddr_addr=B + 0x10000, rows=3, row_bytes=8192,
              ddr_stride=12288)
    await tb.run_cmds([c])
    tb.check()
    per = min(MAXB, 256)
    assert all(n == per for _, n in tb.mon.ars), tb.mon.ars
    assert sum(n for _, n in tb.mon.ars) == 3 * 512
    # same for stores, unaligned start: first burst stops at the boundary
    tb.mon.aws.clear()
    c = isa.C("DMA_STORE", sp_buf=1, sp_addr=0x20003, ddr_addr=B + 0x30FF7, rows=1, row_bytes=5000)
    await tb.run_cmds([c])
    tb.check()
    assert tb.mon.aws[0] == (B + 0x30FF0, 1)
    assert sum(n for _, n in tb.mon.aws) == -(-(7 + 5000) // 16)
    # 1-byte rows at every DDR offset x scratchpad offset
    cmds = []
    for do in range(16):
        for so in range(0, 16, 5):
            op = "DMA_LOAD" if (do + so) % 2 else "DMA_STORE"
            cmds.append(isa.C(op, sp_buf=2, sp_addr=0x4000 + 64 * do + so, ddr_addr=B + 0x50000 + 64 * so + do,
                              rows=2, row_bytes=1 + (do % 3), ddr_stride=17, sp_stride=2))
    await tb.run_cmds(cmds)
    tb.check()
    # DMA_STORE at the very top of the 32-bit space wraps to address 0 (rows past 2^32)
    c = isa.C("DMA_STORE", sp_buf=3, sp_addr=0x10, ddr_addr=0xFFFF_FFF8, rows=2, row_bytes=24, ddr_stride=32)
    out = await tb.run_cmds([c])
    tb.check(out)
    assert tb.ram.read(0x18, 24) == bytes(tb.spad[3][0x10 + 24:0x10 + 48])


@cocotb.test()
async def dma_store_then_load(dut):
    """A load after a store to the same DDR bytes observes the store (DMA queue order)."""
    tb = TB(dut, 5)
    await tb.start()
    B = WIN_BASE + 0x8000
    cmds = [isa.C("DMA_STORE", sp_buf=0, sp_addr=0x1235, ddr_addr=B + 3, rows=4, row_bytes=100, ddr_stride=111),
            isa.C("DMA_LOAD", sp_buf=4, sp_addr=0x7771, ddr_addr=B + 3, rows=1, row_bytes=444)]
    await tb.run_cmds(cmds)
    tb.check()


@cocotb.test()
async def dma_axi_errors(dut):
    """RRESP / BRESP errors set the sticky err output; the command still completes."""
    tb = TB(dut, 7, p_axi=0.2)
    await tb.start()
    bad = (WIN_BASE + 0x9000, WIN_BASE + 0xA000)
    orig_r, orig_w = tb.ram.read_if._read, tb.ram.write_if._write

    async def bad_read(addr, length):
        if bad[0] <= addr < bad[1]:
            raise IOError("injected")
        return await orig_r(addr, length)

    async def bad_write(addr, data):
        if bad[0] <= addr < bad[1]:
            raise IOError("injected")
        return await orig_w(addr, data)

    tb.ram.read_if._read = bad_read
    tb.ram.write_if._write = bad_write
    ok = isa.C("DMA_LOAD", sp_buf=0, sp_addr=0, ddr_addr=WIN_BASE, rows=1, row_bytes=256)
    await tb.run_cmds([ok])
    assert int(dut.err.value) == 0
    c = isa.C("DMA_LOAD", sp_buf=0, sp_addr=0, ddr_addr=bad[0] + 32, rows=2, row_bytes=40)
    tb.cmd.q.append(c)
    await tb.bench.run_until(lambda: tb.cmd.done == 2, 5000, "erroring load done")
    assert int(dut.err.value) == 1
    await tb.bench.reset()
    assert int(dut.err.value) == 0
    tb.sp.reset()
    c = isa.C("DMA_STORE", sp_buf=0, sp_addr=0, ddr_addr=bad[0] + 32, rows=1, row_bytes=40)
    tb.cmd.q.append(c)
    await tb.bench.run_until(lambda: tb.cmd.done == 3, 5000, "erroring store done")
    assert int(dut.err.value) == 1


@cocotb.test()
async def dma_reset_mid_stream(dut):
    """Reset while a long transfer is in flight (AXI model reset too); the DUT returns to idle
    and the next commands execute correctly."""
    tb = TB(dut, 9)
    await tb.start()
    rng = random.Random(10)
    for trial in range(4):
        op = "DMA_LOAD" if trial % 2 == 0 else "DMA_STORE"
        c = isa.C(op, sp_buf=1, sp_addr=0x40000, ddr_addr=WIN_BASE + 0x20000 + 5, rows=4, row_bytes=3000)
        tb.cmd.q.append(c)
        for _ in range(rng.randint(20, 400)):
            await tb.bench.step()
        tb.cmd.reset()
        tb.sp.reset()
        await tb.bench.reset()
        assert int(dut.cmd_ready.value) == 1 and int(dut.axi_idle.value) == 1 and int(dut.busy.value) == 0
        # the interrupted command left partial data: resynchronise the shadow state
        tb.ddr[:] = tb.ram.read(WIN_BASE, WIN)
        for b in range(5):
            tb.spad[b][:] = tb.sp.mem[b]
        tb.cmd.done = 0
        cmds = [rand_dma(rng) for _ in range(3)]
        out = await tb.run_cmds(cmds)
        tb.check(out)
