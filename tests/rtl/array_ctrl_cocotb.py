"""cocotb tests for rtl/array_ctrl.sv (tile controller + systolic array + scratchpads).

Reference: nanoxl.numerics.gemm_i8 / pack_i4 / unpack_i4 (contracts C1, C1b).
Env: ARR_N, ARR_MDEPTH. Protocol under test: contract C3 (docs/contracts.md).

Every driver writes on the falling edge and samples the handshake in the ReadOnly phase of
the same half-cycle, i.e. the settled values the DUT sees at the next rising edge.
"""
import os
import random
from collections import deque

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly
from cocotb.utils import get_sim_time

from nanoxl.numerics import gemm_i8, pack_i4, unpack_i4

N = int(os.environ.get("ARR_N", "4"))
MDEPTH = int(os.environ.get("ARR_MDEPTH", "64"))
LAT = 2 * N - 1


def pack(vals, width, total_bits, rng):
    """Pack vals into the low positions; fill every remaining bit with random garbage."""
    v = rng.getrandbits(total_bits)
    mask = (1 << width) - 1
    for i, x in enumerate(vals):
        v &= ~(mask << (i * width))
        v |= (int(x) & mask) << (i * width)
    return v


def unpack_row(v):
    out = []
    for c in range(N):
        x = (v >> (32 * c)) & 0xFFFFFFFF
        out.append(x - (1 << 32) if x >> 31 else x)
    return out


class TB:
    def __init__(self, dut, seed, p_cmd=1.0, p_w=1.0, p_a=1.0, p_out=1.0):
        self.dut = dut
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        self.p = dict(cmd=p_cmd, w=p_w, a=p_a, out=p_out)
        self.cmds, self.ws, self.as_ = deque(), deque(), deque()
        self.expect = deque()   # (row list, last)
        self.got = []
        self.first_a = None
        self.out_cycles = []
        self.tasks = []
        self.stall_until = 0    # out_ready forced low until this cycle

    # ------------------------------------------------------------------ plumbing
    @property
    def cycle(self):
        return int(get_sim_time("ns")) // 10

    async def start(self, clock=True):
        if clock:
            cocotb.start_soon(Clock(self.dut.clk, 10, unit="ns").start())
        await self.reset()

    def stop(self):
        for t in self.tasks:
            t.cancel()
        self.tasks = []

    async def reset(self):
        d = self.dut
        self.stop()
        d.rst_n.value = 0
        for s in ("cmd_valid", "cmd_m", "cmd_k", "cmd_n", "cmd_abase", "cmd_acc", "cmd_emit",
                  "cmd_int4", "w_valid", "w_data", "a_valid", "a_data", "out_ready"):
            getattr(d, s).value = 0
        for _ in range(3):
            await FallingEdge(d.clk)
        d.rst_n.value = 1
        self.cmds.clear(); self.ws.clear(); self.as_.clear(); self.expect.clear()
        self.got = []
        self.tasks = [cocotb.start_soon(c) for c in
                      (self.drive_cmd(), self.drive_stream("w"), self.drive_stream("a"),
                       self.monitor_out())]

    async def drive_cmd(self):
        d = self.dut
        while True:
            await FallingEdge(d.clk)
            if self.cmds and self.rng.random() < self.p["cmd"]:
                c = self.cmds[0]
                d.cmd_valid.value = 1
                d.cmd_m.value, d.cmd_k.value, d.cmd_n.value = c["m"], c["k"], c["n"]
                d.cmd_abase.value = c["abase"]
                d.cmd_acc.value, d.cmd_emit.value, d.cmd_int4.value = c["acc"], c["emit"], c["int4"]
                await ReadOnly()
                if int(d.cmd_ready.value):
                    self.cmds.popleft()
            else:
                d.cmd_valid.value = 0
                d.cmd_m.value = self.rng.getrandbits(4)  # junk while invalid

    async def drive_stream(self, name):
        d = self.dut
        q = self.ws if name == "w" else self.as_
        valid, data, ready = (getattr(d, f"{name}_{s}") for s in ("valid", "data", "ready"))
        while True:
            await FallingEdge(d.clk)
            if q and self.rng.random() < self.p[name]:
                valid.value = 1
                data.value = q[0]
                await ReadOnly()
                if int(ready.value):
                    q.popleft()
                    if name == "a" and self.first_a is None:
                        self.first_a = self.cycle
            else:
                valid.value = 0
                data.value = self.rng.getrandbits(8 * N)

    async def monitor_out(self):
        d = self.dut
        while True:
            await FallingEdge(d.clk)
            rdy = self.cycle >= self.stall_until and self.rng.random() < self.p["out"]
            d.out_ready.value = int(rdy)
            await ReadOnly()
            if rdy and int(d.out_valid.value):
                row = unpack_row(int(d.out_data.value))
                last = int(d.out_last.value)
                assert self.expect, f"unexpected output row {row}"
                erow, elast = self.expect.popleft()
                assert row == erow, (len(self.got), row, erow)
                assert last == elast, (len(self.got), last, elast)
                self.got.append(row)
                self.out_cycles.append(self.cycle)

    # ------------------------------------------------------------------ jobs
    def gemm(self, a, w, int4=False, abase=None, emit=True):
        """Queue a full GEMM a[M,K] @ w[K,n] (n <= N, any K) as ceil(K/N) tile commands
        accumulating into the accumulator rows abase..abase+M-1."""
        m, ktot = a.shape
        n = w.shape[1]
        assert n <= N and m <= MDEPTH
        if abase is None:
            abase = self.rng.randrange(MDEPTH)
        tiles = [(k0, min(N, ktot - k0)) for k0 in range(0, ktot, N)]
        for i, (k0, k) in enumerate(tiles):
            last = i == len(tiles) - 1
            self.cmds.append(dict(m=m, k=k, n=n, abase=abase, acc=int(i > 0),
                                  emit=int(last and emit), int4=int(int4)))
            for r in range(k):
                wr = w[k0 + r]
                if int4:
                    self.ws.append(pack(wr, 4, 8 * N, self.rng))
                else:
                    self.ws.append(pack(wr, 8, 8 * N, self.rng))
            for mm in range(m):
                self.as_.append(pack(a[mm, k0:k0 + k], 8, 8 * N, self.rng))
        if emit:
            exp = np.zeros((m, N), np.int64)
            exp[:, :n] = gemm_i8(a, w)
            for mm in range(m):
                self.expect.append(([int(x) for x in exp[mm]], int(mm == m - 1)))

    def rand_gemm(self, int4=None):
        if int4 is None:
            int4 = self.rng.random() < 0.3
        m = self.rng.randint(1, MDEPTH if self.rng.random() < 0.15 else min(MDEPTH, 3 * N))
        ktot = self.rng.randint(1, 3 * N)
        n = self.rng.randint(1, N)
        if int4:
            w = self.nrng.integers(-8, 8, (ktot, n), dtype=np.int8)
        else:
            w = self.nrng.integers(-128, 128, (ktot, n), dtype=np.int8)
        a = self.nrng.integers(-128, 128, (m, ktot), dtype=np.int8)
        self.gemm(a, w, int4=int4)

    async def wait_done(self, limit=200000):
        d = self.dut
        start = self.cycle
        while self.cmds or self.ws or self.as_ or self.expect or not int(d.idle.value):
            await FallingEdge(d.clk)
            assert self.cycle - start < limit, "timeout"
        assert int(d.hazard_seen.value) == 0, "weight bank overwritten while in use"
        assert int(d.overflow_seen.value) == 0, "output FIFO overflow"


@cocotb.test()
async def ctrl_random_gemms(dut):
    """Random M, K (multi-tile), n (ragged), INT8 and INT4, random accumulator base, random
    bubbles on cmd/w/a and random output backpressure."""
    tb = TB(dut, 100 + N, p_cmd=0.6, p_w=0.7, p_a=0.7, p_out=0.5)
    await tb.start()
    for _ in range(int(os.environ.get("ARR_NJOBS", "40"))):
        tb.rand_gemm()
    await tb.wait_done()
    dut._log.info("rows checked: %d", len(tb.got))


@cocotb.test()
async def ctrl_int4_full(dut):
    """W4A8 with every INT4 code present, full and ragged tiles, K accumulation."""
    tb = TB(dut, 200 + N, p_w=0.8, p_a=0.9, p_out=0.7)
    await tb.start()
    for n in (N, N - 1, 1):
        ktot = 2 * N + 3
        w = tb.nrng.integers(-8, 8, (ktot, n), dtype=np.int8)
        w[0, :] = -8
        w[1, :] = 7
        w[2, : min(n, 16)] = np.arange(-8, 8)[: min(n, 16)]
        a = tb.nrng.integers(-128, 128, (N + 2, ktot), dtype=np.int8)
        a[0, :] = -128
        tb.gemm(a, unpack_i4(pack_i4(np.pad(w, ((0, 0), (0, n % 2)))))[:, :n], int4=True)
    await tb.wait_done()


@cocotb.test()
async def ctrl_extremes(dut):
    """-128 x -128 over K = 4N (4 accumulated tiles), and -128 x 127; exact int32."""
    tb = TB(dut, 300 + N)
    await tb.start()
    ktot = 4 * N
    w = np.full((ktot, N), -128, np.int8)
    w[:, 1::2] = 127
    a = np.array([[-128] * ktot, [127] * ktot, [-128, 127] * (ktot // 2)], np.int8)
    tb.gemm(a, w)
    await tb.wait_done()
    assert tb.got[0][0] == ktot * 16384


@cocotb.test()
async def ctrl_backpressure_long(dut):
    """out_ready held low for long stretches while many rows are in flight: the controller
    must stop issuing (credits) and lose nothing."""
    tb = TB(dut, 400 + N, p_out=0.3)
    await tb.start()
    for _ in range(4):
        tb.rand_gemm(int4=False)
    tb.stall_until = tb.cycle + 6 * N + 40
    await tb.wait_done()


@cocotb.test()
async def ctrl_accumulate_tiny_tiles(dut):
    """M=1, K=1 tiles back to back, accumulating into one row (exercises the accumulator
    read-during-write bypass), interleaved with overwrite commands to the same row."""
    tb = TB(dut, 500 + N)
    await tb.start()
    for _ in range(6):
        ktot = tb.rng.randint(1, 12)
        # one-row GEMM split into K=1 tiles by building the command list by hand
        a = tb.nrng.integers(-128, 128, (1, ktot), dtype=np.int8)
        w = tb.nrng.integers(-128, 128, (ktot, N), dtype=np.int8)
        for i in range(ktot):
            last = i == ktot - 1
            tb.cmds.append(dict(m=1, k=1, n=N, abase=5, acc=int(i > 0), emit=int(last), int4=0))
            tb.ws.append(pack(w[i], 8, 8 * N, tb.rng))
            tb.as_.append(pack(a[0, i:i + 1], 8, 8 * N, tb.rng))
        tb.expect.append(([int(x) for x in gemm_i8(a, w)[0]], 1))
    await tb.wait_done()


@cocotb.test()
async def ctrl_reset_mid_stream(dut):
    """Reset while tiles are loading, computing and draining; afterwards a fresh GEMM is
    exact and nothing stale comes out."""
    tb = TB(dut, 600 + N, p_out=0.5)
    await tb.start()
    for _ in range(3):
        tb.rand_gemm()
    for _ in range(3 * N + 7):
        await FallingEdge(dut.clk)
    await tb.reset()
    for _ in range(4 * N):
        await FallingEdge(dut.clk)
    assert not tb.got and int(dut.out_valid.value) == 0 and int(dut.idle.value) == 1
    for _ in range(3):
        tb.rand_gemm()
    await tb.wait_done()


async def run_perf(dut, label, m, k, n, tiles, seed, clock):
    tb = TB(dut, seed)
    await tb.start(clock)
    for _ in range(tiles):
        a = tb.nrng.integers(-128, 128, (m, k), dtype=np.int8)
        w = tb.nrng.integers(-128, 128, (k, n), dtype=np.int8)
        tb.gemm(a, w)
    tb.first_a = None
    tb.out_cycles = []
    await tb.wait_done()
    period = (tb.out_cycles[-1] - tb.out_cycles[m - 1]) / (tiles - 1)
    first = tb.out_cycles[0] - tb.first_a
    tb.stop()
    return period, first


@cocotb.test()
async def ctrl_perf_streaming(dut):
    """Back-to-back tiles, no input bubbles, out_ready=1. Steady-state cycles per tile must
    be exactly the ideal max(M, K): weight-load bound (K) for GEMV/decode, issue bound (M)
    for prefill, no bubbles between tiles for any shape. Fill latency (first a beat
    accepted to first result beat accepted) is LAT + 5."""
    cases = [("gemv", 1, N), ("gemv-kN-1", 1, N - 1), ("gemv-k/2", 1, max(1, N // 2)),
             ("gemv-k1", 1, 1), ("square", N, N), ("prefill", 2 * N, N), ("prefill-k/2", 2 * N, N // 2),
             ("m-eq-k-small", 2, 2), ("k-gt-m", 3, N)]
    for i, (label, m, k) in enumerate(cases):
        period, first = await run_perf(dut, label, m, k, N, 16, 700 + m + k, i == 0)
        dut._log.info("PERF %-12s N=%d M=%-3d K=%-3d %.2f cycles/tile (ideal max(M,K)=%d), "
                      "fill latency %d", label, N, m, k, period, max(m, k), first)
        assert period == max(m, k), (label, period)  # no bubbles
        assert first == LAT + 5, first  # a-beat reg + LAT + RMW stage + FIFO write/read/queue
