"""Shared harness for the VEC engine benches (rtl/vec_seq.sv).

The bench plays the command processor (engine port, random valid gaps) and the scratchpads:
five byte-addressed buffers (KV260 sizes, the same arrays nanoxl.funcsim uses) behind three
16-byte line read ports (A, B, P) with random ready, random in-order latency 1..LAT and the
16-line credit checked, and one line write port with random ready and byte strobes. PBUF
writes made by the bench (the DMA's role) are reported on the snoop port.

Reference: nanoxl.funcsim.FuncSim, stepped over the same commands. Checks, per command at
its `done` pulse: every byte the command writes equals funcsim's, every byte the RTL wrote
since the previous done lies inside the command's write set, and the five saturation
counters equal funcsim's per-op counts accumulated so far. At the end of a batch, all five
buffers are compared byte for byte.
"""
from __future__ import annotations

import os
import random
from collections import deque

import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from nanoxl import isa
from nanoxl.funcsim import FuncSim, accesses
from nanoxl.hwconfig import KV260

L = int(os.environ.get("L", "4"))
LINE = 16
BUFS = (isa.ABUF, isa.WBUF, isa.ACC, isa.PBUF, isa.KVBUF)
# saturation counter index (vec_seq sat_cnt): 0 requant, 1 ALU, 2 LN out, 3 LN z, 4 softmax score
CAT = {"REQUANT": 0, "VEC": 1, "LN": 2, "SOFTMAX_ONLINE": 4}
INPUTS = ["clr", "cmd_valid", "cmd_word", "ra_arready", "ra_rvalid", "ra_rdata", "rb_arready",
          "rb_rvalid", "rb_rdata", "rp_arready", "rp_rvalid", "rp_rdata", "wd_ready",
          "pb_snoop_valid", "pb_snoop_line", "sat_clr"]


def u(sig):
    v = sig.value
    return v.to_unsigned() if hasattr(v, "to_unsigned") else int(v)


def word(c: isa.Cmd) -> int:
    return int.from_bytes(isa.encode(c, aligned=True), "little")


def sat_delta(before: dict, after: dict, op: str):
    """Per-category saturation events of one funcsim step."""
    d = [0] * 5
    for k, v in after.items():
        n = v - before.get(k, 0)
        if not n:
            continue
        if k == "ln_z":
            d[3] += n
        else:
            d[CAT[op]] += n
    return d


class ReadPort:
    def __init__(self, tb, pfx, fixed_buf=None):
        self.tb, self.fixed = tb, fixed_buf
        d = tb.dut
        self.arvalid = getattr(d, pfx + "_arvalid")
        self.arready = getattr(d, pfx + "_arready")
        self.arbuf = getattr(d, pfx + "_arbuf") if fixed_buf is None else None
        self.araddr = getattr(d, pfx + "_araddr")
        self.rvalid = getattr(d, pfx + "_rvalid")
        self.rdata = getattr(d, pfx + "_rdata")
        self.q = deque()       # (due cycle, data)
        self.last_due = -1
        self.rdy = 0
        self.resp = False
        self.reads = 0

    def drive(self, cyc):
        tb = self.tb
        self.rdy = 1 if tb.rng.random() < tb.p_ar else 0
        self.arready.value = self.rdy
        self.resp = bool(self.q) and self.q[0][0] <= cyc
        self.rvalid.value = 1 if self.resp else 0
        self.rdata.value = self.q[0][1] if self.resp else tb.rng.getrandbits(128)

    def sample(self, cyc):
        tb = self.tb
        if self.resp:
            self.q.popleft()
        if self.rdy and self.arvalid.value == 1:
            b = self.fixed if self.fixed is not None else u(self.arbuf)
            a = u(self.araddr)
            m = tb.mem[b]
            assert (a + 1) * LINE <= len(m), f"read past {isa.BUF_NAMES[b]}: line {a}"
            data = int.from_bytes(m[a * LINE:(a + 1) * LINE].tobytes(), "little")
            due = max(cyc + tb.rng.randint(1, tb.lat), self.last_due + 1)
            self.last_due = due
            self.q.append((due, data))
            self.reads += 1
            assert len(self.q) <= 16, "more than 16 lines in flight on a read port"
            tb.port_log.append((b, a))


class TB:
    def __init__(self, dut, seed=1, p_ar=0.8, p_w=0.8, p_cmd=0.7, lat=6, hw=KV260):
        self.dut = dut
        self.rng = random.Random(seed)
        self.p_ar, self.p_w, self.p_cmd, self.lat = p_ar, p_w, p_cmd, lat
        self.fs = FuncSim(hw)
        self.mem = {b: self.fs.mem[b].copy() for b in BUFS}
        self.ports = [ReadPort(self, "ra"), ReadPort(self, "rb"), ReadPort(self, "rp", isa.PBUF)]
        self.cmdq = deque()       # words to send
        self.holding = None
        self.snoopq = deque()
        self.cyc = 0
        self.accepted = 0
        self.dones = 0
        self.writes = []          # (buf, line, strb) since the last done
        self.port_log = []
        self.expect = deque()     # per accepted-or-queued command: expectation dict
        self.sat = [0] * 5        # funcsim cumulative, by category
        self.sat_base = [0] * 5   # RTL counter value when the bench's count started
        self.wrdy = 0
        self.checking = True
        self.done_cycles = []

    async def start(self):
        from cocotb import start_soon
        start_soon(Clock(self.dut.clk, 10, unit="ns").start())
        for n in INPUTS:
            getattr(self.dut, n).value = 0
        await self.hard_reset()

    async def hard_reset(self):
        self.dut.rst_n.value = 0
        for p in self.ports:
            p.q.clear()
            p.last_due = -1
        for _ in range(3):
            await RisingEdge(self.dut.clk)
        await FallingEdge(self.dut.clk)
        self.dut.rst_n.value = 1
        self.sat_base = [0] * 5
        self.sat = [0] * 5

    # ---------------------------------------------------------------- memory helpers
    def write_mem(self, buf, addr, data, snoop=True):
        """Bench-side write (the DMA's role) into both the RTL's memory and funcsim's."""
        b = np.frombuffer(bytes(data), np.uint8)
        self.mem[buf][addr:addr + len(b)] = b
        self.fs.mem[buf][addr:addr + len(b)] = b
        if buf == isa.PBUF and snoop:
            for ln in range(addr // LINE, (addr + len(b) - 1) // LINE + 1):
                self.snoopq.append(ln)

    def sync_from_rtl(self):
        for b in BUFS:
            self.fs.mem[b][:] = self.mem[b]

    def sync_to_rtl(self, snoop="changed"):
        """Copy funcsim's buffers into the RTL's memory; PBUF lines that changed (or all of
        ``snoop`` if it is a list of lines) are reported on the snoop port."""
        if snoop == "changed":
            a = self.mem[isa.PBUF].reshape(-1, LINE)
            b = self.fs.mem[isa.PBUF].reshape(-1, LINE)
            self.snoopq.extend(int(i) for i in np.nonzero((a != b).any(axis=1))[0])
        else:
            self.snoopq.extend(snoop)
        for b in BUFS:
            self.mem[b][:] = self.fs.mem[b]

    # ---------------------------------------------------------------- expectations
    def add(self, c: isa.Cmd, note=""):
        """Step funcsim over ``c`` and queue its word with what the RTL must produce."""
        before = dict(self.fs.sat_count)
        self.fs.step(c)
        d = sat_delta(before, self.fs.sat_count, c.op)
        self.sat = [a + b for a, b in zip(self.sat, d)]
        rects, allowed = [], {}
        for (buf, wr, addr, rb, rows, stride) in accesses(c):
            if not wr:
                continue
            rects.append((buf, addr, rb, rows, stride, self.fs.read(buf, addr, rb, rows, stride)))
            for r in range(rows):
                a0 = addr + r * stride
                for ln in range(a0 // LINE, (a0 + rb - 1) // LINE + 1):
                    lo, hi = max(a0, ln * LINE), min(a0 + rb, (ln + 1) * LINE)
                    m = ((1 << (hi - lo)) - 1) << (lo - ln * LINE)
                    allowed[(buf, ln)] = allowed.get((buf, ln), 0) | m
        self.expect.append(dict(cmd=c, rects=rects, allowed=allowed, sat=list(self.sat), note=note))
        self.cmdq.append(word(c))

    def add_raw(self, w, unchecked=False):
        """Queue an illegal word: the RTL must skip it (no writes, counters unchanged) and still
        pulse done. ``unchecked``: anything may be written (an error found mid-command)."""
        self.expect.append(dict(cmd=None, rects=[], allowed=None if unchecked else {}, sat=list(self.sat),
                                note="illegal"))
        self.cmdq.append(w)

    def check_done(self):
        assert self.expect, "done pulse without an outstanding command"
        e = self.expect.popleft()
        if e["allowed"] is None:
            self.writes = []
            return
        for (b, ln, strb) in self.writes:
            ok = e["allowed"].get((b, ln), 0)
            assert strb & ~ok == 0, (f"cmd {e['cmd']}: write outside its destination: "
                                     f"{isa.BUF_NAMES[b]} line {ln} strb {strb:04x} allowed {ok:04x}")
        self.writes = []
        for (buf, addr, rb, rows, stride, exp) in e["rects"]:
            m = self.mem[buf]
            idx = addr + np.arange(rows)[:, None] * stride + np.arange(rb)[None, :]
            got = m[idx]
            if not np.array_equal(got, exp):
                bad = np.argwhere(got != exp)
                r, c = bad[0]
                raise AssertionError(
                    f"cmd #{self.dones} {e['cmd']} ({e['note']}): {len(bad)} bytes differ, first at row {r}"
                    f" byte {c}: got {got[r, c]} expected {exp[r, c]}")
        sc = u(self.dut.sat_cnt)
        got = [((sc >> (32 * i)) & 0xFFFFFFFF) - self.sat_base[i] for i in range(5)]
        assert got == e["sat"], f"cmd #{self.dones} {e['cmd']}: sat counters {got} expected {e['sat']}"

    # ---------------------------------------------------------------- cycle loop
    async def cycle(self):
        d = self.dut
        await FallingEdge(d.clk)
        self.cyc += 1
        # engine command port
        if self.holding is None and self.cmdq and not self.snoopq and self.rng.random() < self.p_cmd:
            self.holding = self.cmdq.popleft()
        d.cmd_valid.value = 1 if self.holding is not None else 0
        d.cmd_word.value = self.holding if self.holding is not None else self.rng.getrandbits(128)
        for p in self.ports:
            p.drive(self.cyc)
        self.wrdy = 1 if self.rng.random() < self.p_w else 0
        d.wd_ready.value = self.wrdy
        sn = self.snoopq.popleft() if self.snoopq else None
        d.pb_snoop_valid.value = 1 if sn is not None else 0
        d.pb_snoop_line.value = sn if sn is not None else 0
        await ReadOnly()
        if self.holding is not None and d.cmd_ready.value == 1:
            self.holding = None
            self.accepted += 1
        for p in self.ports:
            p.sample(self.cyc)
        if self.wrdy and d.wd_valid.value == 1:
            b, a, strb = u(d.wd_buf), u(d.wd_addr), u(d.wd_strb)
            data = u(d.wd_data).to_bytes(16, "little")
            m = self.mem[b]
            assert (a + 1) * LINE <= len(m), f"write past {isa.BUF_NAMES[b]}: line {a}"
            for i in range(16):
                if strb >> i & 1:
                    m[a * LINE + i] = data[i]
            self.writes.append((b, a, strb))
        if d.done.value == 1:
            self.dones += 1
            self.done_cycles.append(self.cyc)
            if self.checking:
                self.check_done()

    async def drain(self, max_cycles=3_000_000):
        """Run until every queued command is done."""
        n = 0
        while self.expect or self.cmdq or self.holding is not None or self.snoopq:
            await self.cycle()
            n += 1
            assert n < max_cycles, f"timeout: {len(self.expect)} commands not done (deadlock?)"
        for _ in range(8):
            await self.cycle()
        assert not self.writes, "writes after the last done"

    async def idle(self, n):
        for _ in range(n):
            await self.cycle()

    def compare_all(self):
        for b in BUFS:
            if not np.array_equal(self.mem[b], self.fs.mem[b]):
                i = int(np.nonzero(self.mem[b] != self.fs.mem[b])[0][0])
                raise AssertionError(f"{isa.BUF_NAMES[b]} differs at byte {i}: got {self.mem[b][i]}"
                                     f" expected {self.fs.mem[b][i]}")

    def err(self):
        return int(self.dut.err.value), u(self.dut.err_code)
