"""cocotb tests for rtl/ctrl_top.sv: the whole control plane (CSRs over AXI4-Lite, ring fetch over
AXI4, dispatch into three in-order queues, WAIT/SIGNAL semaphores, END/IRQ barriers, the real
DMA engine against cocotbext-axi AxiRam, ARRAY/VEC engine stubs with random latency).

Random command streams obey the compiler invariants of docs/isa.md §2 and are checked against a
Python scoreboard: per-queue order and words, every SIGNAL->WAIT dependency (no engine command
starts before the commands its WAITs depend on are done), barriers, final semaphore values,
DDR/scratchpad contents after the DMA commands, and the perf counters.
Env: CTRL_NCMD (stream length of the long random test).
"""
import mmap
import logging
import os
import random
from collections import deque

import cocotb
from cocotb.triggers import RisingEdge
from cocotbext.axi import AxiBus, AxiLiteBus, AxiLiteMaster, AxiRam, AxiReadBus
from cocotbext.axi.axi_ram import AxiRamRead

from ctrl_common_cocotb import (AxiMonitor, Bench, SpadModel, dma_apply, rand_legal, rand_pause,
                                set_axi_pauses, word_int)
from nanoxl import isa

NCMD = int(os.environ.get("CTRL_NCMD", "300"))

REG = dict(ID=0x000, VERSION=0x004, CTRL=0x008, STATUS=0x00C, ERR_CODE=0x010, ERR_INDEX=0x014,
           ERR_W0=0x018, RING_BASE=0x030, RING_SIZE=0x034, RING_HEAD=0x038, DOORBELL=0x03C,
           IRQ_EN=0x040, IRQ_STATUS=0x044, IRQ_CODE=0x048, END_COUNT=0x04C, DBG_STEP=0x050,
           DBG_LAST=0x054, SEM_SEL=0x058, SEM_VAL=0x05C, PERF_CTRL=0x060, PERF_CYCLES=0x064,
           PERF_BUSY_DMA=0x068, PERF_BUSY_ARR=0x06C, PERF_BUSY_VEC=0x070, PERF_RETIRED=0x074,
           PERF_BYTES_LO=0x078, PERF_BYTES_HI=0x07C, SAT0=0x080)
ST_HALTED, ST_IDLE, ST_EMPTY, ST_BUSY = 1, 2, 4, 8
ERR_OPCODE, ERR_RESV, ERR_FIELD, ERR_FETCH, ERR_DMA, ERR_RING, ERR_ENGINE = 1, 2, 3, 4, 5, 6, 7

RING_BASE = 0x1000_0F80          # the ring straddles a 4 KiB boundary after 8 words
WIN_BASE, WIN = 0x4000_0000, 1 << 18
Q_NAMES = ("dma", "arr", "vec")


# ----------------------------------------------------------------------------- agents
class EngineStub:
    """ARRAY / VEC engine stub: random ready, accepts up to `cap` commands, completes them in
    order after a random latency in [lat_lo, lat_hi] cycles."""

    def __init__(self, dut, pre, rng, p_ready=0.8, lat=(1, 8), cap=4):
        self.d, self.pre, self.rng, self.p, self.lat, self.cap = dut, pre, rng, p_ready, lat, cap
        self.pend = deque()          # done cycles
        self.cycle = 0
        self.ready = self.done_now = 0
        self.err = 0
        self.acc, self.dones = [], []  # (cycle, word), cycle

    def reset(self):
        self.pend.clear()

    def s(self, n, v):
        getattr(self.d, f"{self.pre}_{n}").value = v

    def drive(self):
        self.cycle += 1
        self.ready = int(len(self.pend) < self.cap and self.rng.random() < self.p)
        self.done_now = int(bool(self.pend) and self.pend[0] <= self.cycle)
        if self.done_now:
            self.pend.popleft()
        self.s("cmd_ready", self.ready)
        self.s("done", self.done_now)
        self.s("err", self.err)

    def sample(self):
        if self.done_now:
            self.dones.append(self.cycle)
        if self.ready and int(getattr(self.d, f"{self.pre}_cmd_valid").value):
            w = int(getattr(self.d, f"{self.pre}_cmd_word").value)
            self.acc.append((self.cycle, w))
            t = self.cycle + self.rng.randint(*self.lat)
            if self.pend:
                t = max(t, self.pend[-1] + 1)
            self.pend.append(t)


class ProcMonitor:
    """Watches the DMA engine's command port and the barrier pulses inside ctrl_top."""

    def __init__(self, dut):
        self.d = dut
        self.cycle = 0
        self.acc, self.dones, self.irqs, self.ends = [], [], [], []

    def drive(self):
        self.cycle += 1

    def sample(self):
        d = self.d.u_dma
        if int(d.cmd_valid.value) and int(d.cmd_ready.value):
            self.acc.append((self.cycle, int(d.cmd_word.value)))
        if int(d.done.value):
            self.dones.append(self.cycle)
        p = self.d.u_proc
        if int(p.irq_pulse.value):
            self.irqs.append(self.cycle)
        if int(p.end_pulse.value):
            self.ends.append(self.cycle)


# ----------------------------------------------------------------------------- bench
class TB:
    def __init__(self, dut, seed, p_axi=0.25, p_sp=0.7, eng_p=0.8, eng_lat=(1, 8), eng_cap=4,
                 ring_size=48):
        self.dut = dut
        self.rng = random.Random(seed)
        self.bench = Bench(dut)
        for n in ("md_axi", "mf_axi", "s_axil"):
            logging.getLogger(f"cocotb.{dut._name}.{n}").setLevel(logging.ERROR)
        self.mem = mmap.mmap(-1, 1 << 32)
        self.ram = AxiRam(AxiBus.from_prefix(dut, "md_axi"), dut.clk, dut.rst_n,
                          reset_active_level=False, size=1 << 32, mem=self.mem)
        self.fram = AxiRamRead(AxiReadBus.from_prefix(dut, "mf_axi"), dut.clk, dut.rst_n,
                               reset_active_level=False, size=1 << 32, mem=self.mem)
        if p_axi:
            set_axi_pauses(self.ram, self.rng, p_axi)
            set_axi_pauses(self.fram, self.rng, p_axi, read_only=True)
        self.axil = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axil"), dut.clk, dut.rst_n,
                                  reset_active_level=False)
        for ch in (self.axil.write_if.aw_channel, self.axil.write_if.w_channel,
                   self.axil.read_if.ar_channel):
            ch.set_pause_generator(rand_pause(random.Random(self.rng.random()), 0.3))
        for lg in (self.ram.write_if.log, self.ram.read_if.log, self.fram.log,
                   self.axil.write_if.log, self.axil.read_if.log):
            lg.setLevel(logging.WARNING)
        self.sp = SpadModel(dut, self.rng, p_ready=p_sp)
        self.arr = EngineStub(dut, "arr", self.rng, eng_p, eng_lat, eng_cap)
        self.vec = EngineStub(dut, "vec", self.rng, eng_p, eng_lat, eng_cap)
        self.mon = ProcMonitor(dut)
        self.fmon = AxiMonitor(dut, "mf_axi_", max_beats=8, write=False)
        self.dmon = AxiMonitor(dut, "md_axi_", max_beats=256)
        self.bench.agents = [self.sp, self.arr, self.vec, self.mon, self.fmon, self.dmon]
        self.ring_size = ring_size
        self.tail = 0
        self.ddr = bytearray(self.rng.randbytes(WIN))
        self.ram.write(WIN_BASE, bytes(self.ddr))
        self.spad = {b: bytearray(1 << 21) for b in range(8)}
        self.stream_all = []      # every word submitted since the last reset, in ring order
        self.dma_bytes = 0

    async def start(self):
        d = self.dut
        d.rst_n.value = 0
        d.sat_cnt.value = 0
        await self.bench.start_clock()
        cocotb.start_soon(self._run())
        await self.reset()

    async def _run(self):
        while True:
            await self.bench.step()

    async def cycles(self, n):
        for _ in range(n):
            await RisingEdge(self.dut.clk)

    async def reset(self):
        self.bench.set("rst_n", 0)
        await self.cycles(5)
        self.sp.reset()
        self.arr.reset()
        self.vec.reset()
        self.bench.set("rst_n", 1)
        await self.cycles(3)
        self.tail = 0
        await self.configure()

    async def configure(self, size=None):
        await self.wr("RING_BASE", RING_BASE)
        await self.wr("RING_SIZE", size or self.ring_size)
        await self.wr("IRQ_EN", 7)

    async def wr(self, reg, val):
        await self.axil.write_dword(REG[reg] if isinstance(reg, str) else reg, val)

    async def rd(self, reg):
        return await self.axil.read_dword(REG[reg] if isinstance(reg, str) else reg)

    # ------------------------------------------------------------------ software ring driver
    async def submit(self, words, chunk_max=12, limit=200_000):
        """Write words into the ring as space frees up and ring the doorbell after each chunk."""
        i = 0
        t0 = self.bench.cycle
        while i < len(words):
            head = await self.rd("RING_HEAD")
            free = self.ring_size - 1 - (self.tail - head) % self.ring_size
            if free == 0:
                await self.cycles(self.rng.randint(5, 40))
                assert self.bench.cycle - t0 < limit, "ring never drained"
                st = await self.rd("STATUS")
                if st & ST_HALTED:
                    return i
                continue
            n = min(free, self.rng.randint(1, chunk_max), len(words) - i)
            for w in words[i:i + n]:
                b = w if isinstance(w, bytes) else isa.encode(w)
                self.mem[RING_BASE + 16 * self.tail:RING_BASE + 16 * self.tail + 16] = b
                self.tail = (self.tail + 1) % self.ring_size
            self.stream_all += words[i:i + n]
            i += n
            await self.wr("DOORBELL", self.tail)
        return i

    async def wait_idle(self, limit=400_000, ends=None):
        t0 = self.bench.cycle
        while True:
            await self.cycles(20)
            st = await self.rd("STATUS")
            if st & ST_HALTED:
                return st
            if (st & ST_IDLE) and (st & ST_EMPTY) and (ends is None or len(self.mon.ends) >= ends):
                return st
            assert self.bench.cycle - t0 < limit, f"timeout, STATUS={st:#x}"

    # ------------------------------------------------------------------ reference model
    def dma_ref(self, c):
        def rd(a, n):
            return bytes(self.ddr[a - WIN_BASE:a - WIN_BASE + n])

        def wr(a, data):
            self.ddr[a - WIN_BASE:a - WIN_BASE + len(data)] = data
        dma_apply(c, rd, wr, self.spad)
        self.dma_bytes += c.f["rows"] * c.f["row_bytes"]

    def mark(self):
        """Log positions, so check_stream can look at one stream only."""
        return ((len(self.mon.acc), len(self.arr.acc), len(self.vec.acc)),
                (len(self.mon.irqs), len(self.mon.ends)))

    def set_engines(self, p, lat, cap):
        for e in (self.arr, self.vec):
            e.p, e.lat, e.cap = p, lat, cap

    def check_memory(self):
        assert self.ram.read(WIN_BASE, WIN) == bytes(self.ddr), "DDR window mismatch"
        for b in range(5):
            assert self.sp.mem[b] == self.spad[b], f"scratchpad {b} mismatch"


# ----------------------------------------------------------------------------- streams
def small_dma(rng):
    op = rng.choice(["DMA_LOAD", "DMA_STORE"])
    rb = rng.randint(1, 80)
    rows = rng.randint(1, 3)
    ds = rng.choice([0, rb + rng.randint(0, 40)])
    span = (rows - 1) * (ds or rb) + rb
    return isa.C(op, chan=rng.randint(0, 2), sp_buf=rng.randint(0, 4), sp_addr=rng.randrange(1 << 21),
                 ddr_addr=WIN_BASE + rng.randint(0, WIN - span), rows=rows, row_bytes=rb,
                 ddr_stride=ds, sp_stride=rng.choice([0, -(-rb // 16) + rng.randint(0, 2)]))


def engine_cmd(rng, q):
    if q == isa.Q_DMA:
        return small_dma(rng)
    ops = [o for o, qq in isa.OP_QUEUE.items() if qq == q]
    return rand_legal(rng, rng.choice(ops))


def gen_stream(rng, n, n_sems=10, p_irq=0.02, end=True):
    """Random stream obeying docs/isa.md §2: each semaphore has one producer and one consumer
    queue, and every WAIT consumes only SIGNALs that appear earlier in the stream."""
    sems = rng.sample(range(32), n_sems)
    pc = {s: tuple(rng.sample(range(3), 2)) for s in sems}
    pending = {s: 0 for s in sems}
    out = []
    while len(out) < n:
        u = rng.random()
        if u < 0.18:
            s = rng.choice(sems)
            out.append(isa.C("SIGNAL", queue=pc[s][0], sem=s))
            pending[s] += 1
        elif u < 0.36:
            ready = [s for s in sems if pending[s]]
            if not ready:
                continue
            s = rng.choice(ready)
            k = rng.randint(1, min(3, pending[s]))
            out.append(isa.C("WAIT", queue=pc[s][1], sem=s, count=k))
            pending[s] -= k
        elif u < 0.36 + p_irq:
            out.append(isa.C("IRQ", code=rng.randrange(1 << 16)))
        else:
            out.append(engine_cmd(rng, rng.randrange(3)))
    for s in sems:            # drain every semaphore so all counters end at zero
        if pending[s]:
            out.append(isa.C("WAIT", queue=pc[s][1], sem=s, count=pending[s]))
    if end:
        out.append(isa.C("END"))
    return out


# ----------------------------------------------------------------------------- scoreboard
def check_stream(tb, stream, mark=((0, 0, 0), (0, 0))):
    """Order, words, dependencies and barriers of `stream` against the monitors' logs, from the
    log positions in `mark` (TB.mark() taken before the stream was submitted)."""
    base_counts, base_bar = mark
    acc = [tb.mon.acc[base_counts[0]:], tb.arr.acc[base_counts[1]:], tb.vec.acc[base_counts[2]:]]
    dones = [tb.mon.dones[base_counts[0]:], tb.arr.dones[base_counts[1]:], tb.vec.dones[base_counts[2]:]]
    irqs, ends = tb.mon.irqs[base_bar[0]:], tb.mon.ends[base_bar[1]:]
    eng = [[], [], []]                 # per queue: stream indices of engine commands
    sigs = {}                          # sem -> list of (producer q, #engine cmds of q before it)
    used = {}                          # sem -> signals consumed so far
    req = [[], [], []]                 # per queue: (engine position from which it applies, (p, k))
    bars = []                          # (kind, #engine cmds per queue before it)
    for i, c in enumerate(stream):
        if c.op == "SIGNAL":
            sigs.setdefault(c.f["sem"], []).append((c.f["queue"], len(eng[c.f["queue"]])))
        elif c.op == "WAIT":
            s, q = c.f["sem"], c.f["queue"]
            used[s] = used.get(s, 0) + c.f["count"]
            p, k = sigs[s][used[s] - 1]
            if k:
                req[q].append((len(eng[q]), (p, k)))
        elif c.op in ("IRQ", "END"):
            bars.append((c.op, tuple(len(e) for e in eng)))
        else:
            eng[c.queue].append(i)
    for q in range(3):
        assert len(acc[q]) == len(eng[q]), f"{Q_NAMES[q]}: {len(acc[q])} accepted, {len(eng[q])} expected"
        assert len(dones[q]) == len(eng[q]), f"{Q_NAMES[q]}: {len(dones[q])} done"
        for j, si in enumerate(eng[q]):
            assert acc[q][j][1] == word_int(stream[si]), f"{Q_NAMES[q]} command {j} out of order"
        for pos, (p, k) in req[q]:
            bound = dones[p][k - 1]
            for j in range(pos, len(eng[q])):
                assert acc[q][j][0] > bound, (f"{Q_NAMES[q]} cmd {j} started at {acc[q][j][0]} before "
                                              f"{Q_NAMES[p]} cmd {k - 1} done at {bound}")
    ni = ne = 0
    for kind, before in bars:
        t = irqs[ni] if kind == "IRQ" else ends[ne]
        if kind == "IRQ":
            ni += 1
        else:
            ne += 1
        for q in range(3):
            if before[q]:
                assert dones[q][before[q] - 1] < t, f"{kind} before {Q_NAMES[q]} drained"
            if before[q] < len(acc[q]):
                assert acc[q][before[q]][0] > t, f"{Q_NAMES[q]} started before {kind} executed"
    assert ni == len(irqs) and ne == len(ends)
    return eng


def busy_edges(acc, dones):
    s = set()
    for (a, _), d in zip(acc, dones):
        s.update(range(a + 1, d + 1))
    return len(s)


# ----------------------------------------------------------------------------- tests
@cocotb.test()
async def ctrl_random_stream(dut):
    """Long random stream through a small ring (wraps many times, straddles 4 KiB), random AXI /
    AXI-Lite / scratchpad / engine backpressure and latency; full scoreboard."""
    tb = TB(dut, 21)
    await tb.start()
    rng = random.Random(22)
    stream = gen_stream(rng, NCMD)
    for c in stream:
        if c.op.startswith("DMA"):
            tb.dma_ref(c)
    n = await tb.submit(stream)
    assert n == len(stream)
    st = await tb.wait_idle(ends=1)
    assert not st & ST_HALTED, f"halted, ERR_CODE={await tb.rd('ERR_CODE')}"
    eng = check_stream(tb, stream)
    tb.check_memory()
    # semaphores all back to zero
    for s in range(32):
        await tb.wr("SEM_SEL", s)
        assert await tb.rd("SEM_VAL") == 0, s
    # perf counters
    assert await tb.rd("PERF_RETIRED") == len(stream)
    lo, hi = await tb.rd("PERF_BYTES_LO"), await tb.rd("PERF_BYTES_HI")
    assert lo | hi << 32 == tb.dma_bytes
    logs = [(tb.mon.acc, tb.mon.dones), (tb.arr.acc, tb.arr.dones), (tb.vec.acc, tb.vec.dones)]
    for q, reg in enumerate(("PERF_BUSY_DMA", "PERF_BUSY_ARR", "PERF_BUSY_VEC")):
        assert await tb.rd(reg) == busy_edges(*logs[q]), reg
    assert await tb.rd("END_COUNT") == 1
    assert (await tb.rd("IRQ_STATUS")) & 1 and int(dut.irq.value) == 1
    # overlap: the three engines really ran concurrently at some point
    busy_any = [set().union(*(range(a + 1, d + 1) for (a, _), d in zip(*logs[q]))) for q in range(3)]
    assert busy_any[0] & busy_any[1] and busy_any[1] & busy_any[2], "no cross-queue overlap"
    dut._log.info("%d commands (%s engine), %d cycles busy, peak DMA reads %d, %d fetch bursts",
                  len(stream), [len(e) for e in eng], await tb.rd("PERF_CYCLES"), tb.dmon.peak_rd,
                  len(tb.fmon.ars))


@cocotb.test()
async def ctrl_latency_mixes(dut):
    """Several programs back to back (multiple ENDs, doorbells covering partial programs) under
    different engine latency / backpressure mixes and ring sizes, including single-slot engines."""
    tb = TB(dut, 30)
    await tb.start()
    rng = random.Random(40)
    for i, (p, lat, cap) in enumerate([(1.0, (1, 1), 1), (0.3, (5, 40), 2), (1.0, (1, 3), 4)]):
        tb.set_engines(p, lat, cap)
        tb.ring_size = 16 + 7 * i
        await tb.reset()
        m0 = tb.mark()
        for prog in range(3):
            stream = gen_stream(rng, 60)
            for c in stream:
                if c.op.startswith("DMA"):
                    tb.dma_ref(c)
            m = tb.mark()
            await tb.submit(stream, chunk_max=40)
            st = await tb.wait_idle(ends=m[1][1] + 1)
            assert not st & ST_HALTED
            check_stream(tb, stream, m)
        tb.check_memory()
        assert await tb.rd("END_COUNT") == 3
        assert len(tb.mon.ends) - m0[1][1] == 3


@cocotb.test()
async def ctrl_semaphore_saturation(dut):
    """260 SIGNALs to one semaphore before its WAITs: the 8-bit counter stalls the producer at
    255 instead of wrapping (the last 5 wait in the VEC queue), the WAITs then drain it; nothing
    lost or duplicated. (More than 255 + QDEPTH outstanding SIGNALs on one semaphore ahead of
    their WAITs would block the in-order dispatcher: a documented stream limit.)"""
    tb = TB(dut, 50, eng_p=1.0, eng_lat=(1, 2), ring_size=64)
    await tb.start()
    # VEC signals sem 7 260 times; ARRAY's WAITs come after all signals in the stream, so the VEC
    # queue must block at 255 until the first WAIT runs.
    peak = []

    async def watch():
        while True:
            await RisingEdge(dut.clk)
            peak.append(int(dut.u_proc.sem_dbg_val.value))
    await tb.wr("SEM_SEL", 7)
    w = cocotb.start_soon(watch())
    stream = [isa.C("SIGNAL", queue=2, sem=7) for _ in range(260)]
    stream += [isa.C("WAIT", queue=1, sem=7, count=c) for c in (100, 100, 60)]
    stream += [rand_legal(random.Random(1), "GEMM"), isa.C("END")]
    await tb.submit(stream, chunk_max=63)
    st = await tb.wait_idle(ends=1)
    assert not st & ST_HALTED
    await tb.wr("SEM_SEL", 7)
    assert await tb.rd("SEM_VAL") == 0
    assert len(tb.arr.acc) == 1
    assert await tb.rd("PERF_RETIRED") == len(stream)
    w.cancel()
    assert max(peak) == 255, max(peak)


@cocotb.test()
async def ctrl_irq_and_csr(dut):
    """IRQ command: IRQ_CODE, IRQ_STATUS bit 1 (W1C), irq output gated by IRQ_EN; ID/VERSION,
    read/write with byte strobes, saturation counter inputs, PERF/sat clears, unmapped reads."""
    tb = TB(dut, 60)
    await tb.start()
    assert await tb.rd("ID") == 0x4E584C43
    assert await tb.rd("VERSION") == 0x00010000
    assert await tb.rd("RING_BASE") == RING_BASE
    await tb.wr("RING_BASE", RING_BASE | 0xF)            # low 4 bits read as 0
    assert await tb.rd("RING_BASE") == RING_BASE
    await tb.axil.write(REG["RING_SIZE"], bytes([0x22]))   # byte-strobed write: byte 0 only
    assert await tb.rd("RING_SIZE") == 0x22
    await tb.wr("RING_SIZE", tb.ring_size)
    assert await tb.rd(0x7F0) == 0 and await tb.rd(0x0F0) == 0
    tb.bench.set("sat_cnt", sum(v << (32 * i) for i, v in enumerate([5, 0xFFFFFFFF, 123456, 7, 0xA5A5])))
    await tb.cycles(3)
    assert [await tb.rd(REG["SAT0"] + 4 * i) for i in range(5)] == [5, 0xFFFFFFFF, 123456, 7, 0xA5A5]
    await tb.wr("IRQ_EN", 0)
    await tb.submit([isa.C("IRQ", code=0xBEEF), isa.C("END")])
    await tb.wait_idle(ends=1)
    assert await tb.rd("IRQ_CODE") == 0xBEEF
    assert await tb.rd("IRQ_STATUS") == 3
    assert int(dut.irq.value) == 0                        # masked
    await tb.wr("IRQ_EN", 2)
    await tb.cycles(2)
    assert int(dut.irq.value) == 1
    await tb.wr("IRQ_STATUS", 2)                          # write 1 to clear bit 1 only
    assert await tb.rd("IRQ_STATUS") == 1
    await tb.cycles(2)
    assert int(dut.irq.value) == 0
    await tb.wr("IRQ_STATUS", 0)
    assert await tb.rd("IRQ_STATUS") == 1
    assert await tb.rd("PERF_RETIRED") == 2 and await tb.rd("PERF_CYCLES") > 0
    await tb.wr("PERF_CTRL", 1)
    assert await tb.rd("PERF_RETIRED") == 0 and await tb.rd("PERF_CYCLES") == 0
    # sat_clr is a one-cycle pulse on PERF_CTRL bit 1
    seen = []

    async def watch():
        for _ in range(60):
            await RisingEdge(dut.clk)
            seen.append(int(dut.sat_clr.value))
    w = cocotb.start_soon(watch())
    await tb.wr("PERF_CTRL", 2)
    await w
    assert sum(seen) == 1


@cocotb.test()
async def ctrl_single_step(dut):
    """Single-step debug mode: one word dispatched per DBG_STEP write."""
    tb = TB(dut, 70, eng_p=1.0, eng_lat=(1, 2))
    await tb.start()
    await tb.wr("CTRL", 1)
    rng = random.Random(71)
    stream = [engine_cmd(rng, 1), engine_cmd(rng, 2), isa.C("SIGNAL", queue=1, sem=3),
              isa.C("WAIT", queue=2, sem=3, count=1), engine_cmd(rng, 2), isa.C("END")]
    await tb.submit(stream, chunk_max=10)
    await tb.cycles(100)
    assert await tb.rd("RING_HEAD") == 0 and not tb.arr.acc
    for i in range(4):
        await tb.wr("DBG_STEP", 1)
        await tb.cycles(30)
        assert await tb.rd("RING_HEAD") == i + 1
        assert await tb.rd("DBG_LAST") == i
    assert len(tb.arr.acc) == 1 and len(tb.vec.acc) == 1
    await tb.wr("CTRL", 0)
    await tb.wait_idle(ends=1)
    check_stream(tb, stream)


async def expect_halt(tb, code, index=None, word=None):
    st = await tb.wait_idle()
    assert st & ST_HALTED, f"not halted, STATUS={st:#x}"
    for _ in range(2000):                 # the queues drain what was dispatched before the error
        if st & ST_IDLE:
            break
        await tb.cycles(10)
        st = await tb.rd("STATUS")
    assert st & ST_IDLE, f"queues did not drain after the halt, STATUS={st:#x}"
    assert await tb.rd("ERR_CODE") == code
    if index is not None:
        assert await tb.rd("ERR_INDEX") == index
    if word is not None:
        got = 0
        for i in range(4):
            got |= (await tb.rd(REG["ERR_W0"] + 4 * i)) << (32 * i)
        assert got == word, f"{got:032x} != {word:032x}"
    assert (await tb.rd("IRQ_STATUS")) & 4 and int(tb.dut.irq.value) == 1


async def soft_reset_and_rerun(tb, seed):
    """CTRL.soft_reset (drains AXI, clears the control plane), then a fresh program runs."""
    await tb.wr("CTRL", 2)
    for _ in range(100):
        if not (await tb.rd("CTRL")) & 2:
            break
    st = await tb.rd("STATUS")
    assert not st & ST_HALTED and await tb.rd("RING_HEAD") == 0 and await tb.rd("DOORBELL") == 0
    assert await tb.rd("ERR_CODE") == 0
    await tb.wr("IRQ_STATUS", 7)
    tb.tail = 0
    m = tb.mark()
    stream = gen_stream(random.Random(seed), 40, p_irq=0)
    for c in stream:
        if c.op.startswith("DMA"):
            tb.dma_ref(c)
    await tb.submit(stream)
    st = await tb.wait_idle(ends=m[1][1] + 1)
    assert not st & ST_HALTED
    check_stream(tb, stream, m)
    tb.check_memory()


@cocotb.test()
async def ctrl_illegal_commands(dut):
    """Opcode 0 / undefined opcode / reserved bits / invalid field mid-stream: error status with
    index and word, halt (nothing after it is dispatched), error IRQ; soft reset recovers."""
    rng = random.Random(80)
    cases = [(bytes(16), ERR_OPCODE),
             ((0x3F).to_bytes(16, "little"), ERR_OPCODE),
             ((int.from_bytes(isa.encode(isa.C("SIGNAL", queue=1, sem=2)), "little") | 1 << 100)
              .to_bytes(16, "little"), ERR_RESV),
             ((int.from_bytes(isa.encode(isa.C("WAIT", queue=1, sem=2, count=1)), "little") | 3 << 6)
              .to_bytes(16, "little"), ERR_FIELD)]
    tb = TB(dut, 81)
    await tb.start()
    for k, (bad, code) in enumerate(cases):
        pre = [c for c in gen_stream(rng, 25, p_irq=0, end=False) if c.op not in ("WAIT", "SIGNAL")]
        post = [engine_cmd(rng, 1), engine_cmd(rng, 2), isa.C("END")]
        for c in pre:
            if c.op.startswith("DMA"):
                tb.dma_ref(c)
        m = tb.mark()
        start = tb.tail
        await tb.submit(pre + [bad] + post, chunk_max=40)
        await expect_halt(tb, code, index=(start + len(pre)) % tb.ring_size,
                          word=int.from_bytes(bad, "little"))
        check_stream(tb, pre, m)     # everything before ran, nothing after was dispatched
        tb.check_memory()
        await soft_reset_and_rerun(tb, 90 + k)


@cocotb.test()
async def ctrl_error_sources(dut):
    """Fetch RRESP error, DMA AXI error, bad ring configuration, engine error input; each one
    halts with its code and is recovered by a soft reset."""
    tb = TB(dut, 100)
    await tb.start()
    rng = random.Random(101)
    # fetch error on ring word 3
    orig = tb.fram._read

    async def bad_ring(addr, length):
        if addr == RING_BASE + 16 * 3:
            raise IOError("injected")
        return await orig(addr, length)
    tb.fram._read = bad_ring
    await tb.submit([engine_cmd(rng, 1) for _ in range(6)] + [isa.C("END")])
    await expect_halt(tb, ERR_FETCH, index=3)
    tb.fram._read = orig
    await soft_reset_and_rerun(tb, 102)
    # DMA AXI error
    origd = tb.ram.read_if._read

    async def bad_ddr(addr, length):
        if WIN_BASE + 0x100 <= addr < WIN_BASE + 0x200:
            raise IOError("injected")
        return await origd(addr, length)
    tb.ram.read_if._read = bad_ddr
    c = isa.C("DMA_LOAD", sp_buf=0, sp_addr=0, ddr_addr=WIN_BASE + 0x120, rows=1, row_bytes=32)
    tb.dma_ref(c)
    await tb.submit([c, isa.C("END")])
    await expect_halt(tb, ERR_DMA)
    tb.ram.read_if._read = origd
    tb.spad[0][:] = tb.sp.mem[0]       # the failed load wrote (zero) data
    await soft_reset_and_rerun(tb, 103)
    # ring configuration: tail beyond the ring
    await tb.wr("DOORBELL", tb.ring_size + 3)
    await expect_halt(tb, ERR_RING)
    await soft_reset_and_rerun(tb, 104)
    # engine error input
    tb.vec.err = 1
    await tb.cycles(3)
    await expect_halt(tb, ERR_ENGINE)
    tb.vec.err = 0
    await soft_reset_and_rerun(tb, 105)


@cocotb.test()
async def ctrl_reset_mid_stream(dut):
    """Hardware reset in the middle of a random program (AXI models reset too), then a complete
    program runs correctly; repeated at random points."""
    tb = TB(dut, 110)
    await tb.start()
    rng = random.Random(111)
    for trial in range(3):
        stream = gen_stream(rng, 80)
        task = cocotb.start_soon(tb.submit(stream))
        await tb.cycles(rng.randint(50, 1500))
        task.cancel()
        await tb.reset()
        m = tb.mark()
        assert await tb.rd("RING_HEAD") == 0 and (await tb.rd("STATUS")) & ST_IDLE
        # resynchronise the memory shadow (the interrupted program left partial DMA effects)
        tb.ddr[:] = tb.ram.read(WIN_BASE, WIN)
        for b in range(5):
            tb.spad[b][:] = tb.sp.mem[b]
        stream = gen_stream(rng, 60)
        for c in stream:
            if c.op.startswith("DMA"):
                tb.dma_ref(c)
        await tb.submit(stream)
        st = await tb.wait_idle(ends=m[1][1] + 1)
        assert not st & ST_HALTED
        check_stream(tb, stream, m)
        tb.check_memory()
