"""Shared cocotb helpers for the SG6 control-plane tests (cmd / dma / csr / ctrl).

Bench style (as in array_ctrl_cocotb.py): one loop per clock cycle drives every input on the
falling edge, then samples the settled values in ReadOnly, i.e. exactly what the DUT sees at
the next rising edge. Agents expose drive() and sample(); Bench.step() calls them in order.
"""
from __future__ import annotations

import random
from collections import deque

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl import isa


def word_int(cmd) -> int:
    return int.from_bytes(isa.encode(cmd), "little")


def rand_pause(rng: random.Random, p_stall: float):
    """Pause generator for cocotbext-axi channels: True = stall this cycle."""
    while True:
        yield rng.random() < p_stall


def set_axi_pauses(ram, rng, p: float, read_only=False):
    """Random backpressure/latency on every channel of an AxiRam / AxiRamRead."""
    rif = ram if read_only else ram.read_if
    rif.ar_channel.set_pause_generator(rand_pause(random.Random(rng.random()), p))
    rif.r_channel.set_pause_generator(rand_pause(random.Random(rng.random()), p))
    if not read_only:
        w = ram.write_if
        w.aw_channel.set_pause_generator(rand_pause(random.Random(rng.random()), p))
        w.w_channel.set_pause_generator(rand_pause(random.Random(rng.random()), p))
        w.b_channel.set_pause_generator(rand_pause(random.Random(rng.random()), p))


def sig(dut, name):
    return int(getattr(dut, name).value)


class Bench:
    def __init__(self, dut, clk="clk"):
        self.dut = dut
        self.clk = getattr(dut, clk)
        self.agents = []
        self.cycle = 0
        self.pending = {}

    def set(self, name, value):
        """Write a DUT input at the next falling edge (writes are illegal in ReadOnly)."""
        self.pending[name] = value

    async def start_clock(self):
        cocotb.start_soon(Clock(self.clk, 10, unit="ns").start())

    async def step(self):
        await FallingEdge(self.clk)
        self.cycle += 1
        for k, v in self.pending.items():
            getattr(self.dut, k).value = v
        self.pending = {}
        for a in self.agents:
            a.drive()
        await ReadOnly()
        for a in self.agents:
            a.sample()

    async def run_until(self, cond, limit, what="condition"):
        for _ in range(limit):
            await self.step()
            if cond():
                return
        raise AssertionError(f"timeout after {limit} cycles waiting for {what}")

    async def reset(self, n=4):
        self.set("rst_n", 0)
        for _ in range(n):
            await self.step()
        self.set("rst_n", 1)
        await self.step()


class SpadModel:
    """Scratchpad line port of dma_engine / ctrl_top: byte-strobed writes, in-order reads with
    random latency, random ready on both request channels. Memory is 5 buffers x 2 MiB."""

    def __init__(self, dut, rng, p_ready=0.7, max_lat=6, prefix="sp_"):
        self.d, self.rng, self.p, self.max_lat = dut, rng, p_ready, max_lat
        self.pre = prefix
        self.mem = {b: bytearray(1 << 21) for b in range(8)}
        self.rq = deque()          # (ready_cycle, data)
        self.cycle = 0
        self.writes = self.reads = 0
        self.w_ready = self.ar_ready = 0
        self.r_head = None

    def g(self, n):
        return int(getattr(self.d, self.pre + n).value)

    def s(self, n, v):
        getattr(self.d, self.pre + n).value = v

    def reset(self):
        self.rq.clear()

    def drive(self):
        self.cycle += 1
        self.w_ready = int(self.rng.random() < self.p)
        self.ar_ready = int(self.rng.random() < self.p)
        self.s("wready", self.w_ready)
        self.s("arready", self.ar_ready)
        if self.rq and self.rq[0][0] <= self.cycle:
            self.r_head = self.rq.popleft()[1]
            self.s("rvalid", 1)
            self.s("rdata", self.r_head)
        else:
            self.r_head = None
            self.s("rvalid", 0)

    def sample(self):
        if self.w_ready and self.g("wvalid"):
            b, a, data, strb = self.g("wbuf"), self.g("waddr"), self.g("wdata"), self.g("wstrb")
            m = self.mem[b]
            for i in range(16):
                if strb >> i & 1:
                    m[a * 16 + i] = (data >> (8 * i)) & 0xFF
            self.writes += 1
        if self.ar_ready and self.g("arvalid"):
            b, a = self.g("arbuf"), self.g("araddr")
            data = int.from_bytes(self.mem[b][a * 16:a * 16 + 16], "little")
            t = self.cycle + 1 + self.rng.randrange(self.max_lat)
            if self.rq:
                t = max(t, self.rq[-1][0])
            self.rq.append((t, data))
            self.reads += 1


class AxiMonitor:
    """Protocol checks on an AXI4 master: bursts INCR, 16-byte beats, aligned, <= max_beats,
    never crossing 4 KiB; tracks the peak number of outstanding read/write bursts."""

    def __init__(self, dut, prefix, max_beats=256, write=True):
        self.d, self.pre, self.max_beats, self.write = dut, prefix, max_beats, write
        self.rd_out = self.wr_out = 0
        self.peak_rd = self.peak_wr = 0
        self.ars, self.aws = [], []

    def g(self, n):
        return int(getattr(self.d, self.pre + n).value)

    def drive(self):
        pass

    def _check(self, addr, ln, size, burst):
        beats = ln + 1
        assert size == 4 and burst == 1, (size, burst)
        assert addr % 16 == 0, hex(addr)
        assert beats <= self.max_beats, beats
        assert (addr & 0xFFF) + 16 * beats <= 0x1000, f"4 KiB crossing at {addr:#x} x{beats}"

    def sample(self):
        if self.g("arvalid") and self.g("arready"):
            a, ln = self.g("araddr"), self.g("arlen")
            self._check(a, ln, self.g("arsize"), self.g("arburst"))
            self.ars.append((a, ln + 1))
            self.rd_out += 1
        if self.g("rvalid") and self.g("rready") and self.g("rlast"):
            self.rd_out -= 1
        self.peak_rd = max(self.peak_rd, self.rd_out)
        if self.write:
            if self.g("awvalid") and self.g("awready"):
                a, ln = self.g("awaddr"), self.g("awlen")
                self._check(a, ln, self.g("awsize"), self.g("awburst"))
                self.aws.append((a, ln + 1))
                self.wr_out += 1
            if self.g("bvalid") and self.g("bready"):
                self.wr_out -= 1
            self.peak_wr = max(self.peak_wr, self.wr_out)


def dma_geometry(f):
    """(ddr_row_addrs, sp_row_addrs, row_bytes) of a DMA command's fields, per docs/control.md."""
    rb = f["row_bytes"]
    ds = f["ddr_stride"] or rb
    ss = (f["sp_stride"] * 16) or rb
    rows = f["rows"]
    return ([(f["ddr_addr"] + r * ds) % (1 << 32) for r in range(rows)],
            [(f["sp_addr"] + r * ss) % (1 << 21) for r in range(rows)], rb)


def dma_apply(cmd, ddr_read, ddr_write, spad_mem):
    """Reference effect of one DMA command on (ddr, spad) through accessor callables."""
    dd, sp, rb = dma_geometry(cmd.f)
    m = spad_mem[cmd.f["sp_buf"]]
    for da, sa in zip(dd, sp):
        if cmd.op == "DMA_LOAD":
            data = ddr_read(da, rb)
            for i in range(rb):
                m[(sa + i) % (1 << 21)] = data[i]
        else:
            data = bytes(m[(sa + i) % (1 << 21)] for i in range(rb))
            ddr_write(da, data)



def rand_field(rng, w, k):
    lo, hi = isa.field_range(w, k)
    u = rng.random()
    if u < 0.15:
        return lo
    if u < 0.3:
        return hi
    return rng.randint(lo, hi)


def rand_legal(rng, op, **fixed):
    """A random command of `op` that nanoxl.isa.validate accepts (fields in `fixed` pinned)."""
    for _ in range(1000):
        f = {n: rand_field(rng, w, k) for n, w, k in isa.LAYOUTS[op]}
        f.update(fixed)
        c = isa.Cmd(op, f)
        try:
            isa.validate(c)
            return c
        except isa.IsaError:
            continue
    raise RuntimeError(op)
