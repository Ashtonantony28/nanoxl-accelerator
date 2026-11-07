"""Driver backends: the accelerator seen as (a) 32-bit CSRs over AXI-Lite and (b) one physically
contiguous DDR buffer shared with the PL.

* ``PynqBackend``: the real board (pynq.Overlay MMIO + pynq.allocate CMA buffer). Thin by design and
  untested until hardware exists; pynq is imported lazily.
* ``SimBackend``: executes the command ring with nanoxl.funcsim and emulates the CSR semantics of
  ctrl_top (docs/control.md §1, §5): ring head/tail and wrap, three in-order queues of depth 8 with
  counting semaphores (SIGNAL stalls at 255), END / IRQ barriers that raise IRQ_STATUS bits and keep
  dispatching when tail has moved past them, halt on an illegal word with ERR_CODE / ERR_INDEX /
  ERR_W0..3, W1C IRQ_STATUS, single step, soft reset, perf counters (cycle counts from
  nanoxl.cyclemodel over the dispatched words) and the four saturation counters.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .. import isa
from ..funcsim import DDR, FuncSim, SimError
from ..hwconfig import KV260, HwConfig
from . import csr as R


class Backend:
    """Interface the host driver uses."""

    poll_interval = 1e-4

    def read32(self, off: int) -> int:
        raise NotImplementedError

    def write32(self, off: int, val: int) -> None:
        raise NotImplementedError

    def alloc(self, nbytes: int):
        """Allocate the shared DDR buffer; returns (uint8 numpy view, physical address)."""
        raise NotImplementedError

    def flush(self, off: int = 0, n: int | None = None) -> None:
        """CPU writes -> device (cache clean)."""

    def invalidate(self, off: int = 0, n: int | None = None) -> None:
        """device writes -> CPU (cache invalidate)."""

    def wait_irq(self, timeout: float) -> bool:
        """Block until the irq line is high or ``timeout`` s pass; default: poll IRQ_STATUS & IRQ_EN."""
        t0 = time.monotonic()
        while True:
            if self.read32(R.IRQ_STATUS) & self.read32(R.IRQ_EN):
                return True
            if time.monotonic() - t0 > timeout:
                return False
            if self.poll_interval:
                time.sleep(self.poll_interval)


class PynqBackend(Backend):
    """PYNQ / Kria: ``bitstream`` is the .bit with its .hwh next to it; ``ip`` the name of the NanoXL
    AXI-Lite slave in ``Overlay.ip_dict``. UNTESTED until hardware exists (docs/driver.md)."""

    def __init__(self, bitstream: str, ip: str = "nanoxl_0", irq: str | None = None):
        import pynq  # lazy: not installed off-board
        self._pynq = pynq
        self.ol = pynq.Overlay(bitstream)
        desc = self.ol.ip_dict[ip]
        self.mmio = pynq.MMIO(desc["phys_addr"], desc.get("addr_range", 0x1000))
        self.irq = pynq.Interrupt(irq) if irq else None
        self.buf = None

    def read32(self, off):
        return int(self.mmio.read(off)) & 0xFFFFFFFF

    def write32(self, off, val):
        self.mmio.write(off, int(val) & 0xFFFFFFFF)

    def alloc(self, nbytes):
        self.buf = self._pynq.allocate(shape=(nbytes,), dtype=np.uint8, cacheable=False)
        return np.asarray(self.buf), int(self.buf.physical_address)

    def flush(self, off=0, n=None):
        self.buf.flush()

    def invalidate(self, off=0, n=None):
        self.buf.invalidate()

    def wait_irq(self, timeout):
        if self.irq is None:
            return super().wait_irq(timeout)
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(asyncio.wait_for(self.irq.wait(), timeout))
            return True
        except asyncio.TimeoutError:
            return False


def _sat32(x):
    return min(int(x), 0xFFFFFFFF)


class SimBackend(Backend):
    """ctrl_top + engines emulated around nanoxl.funcsim (see module docstring).

    Execution is functional: an engine command completes when it leaves its queue. Dispatch puts
    each word into its queue (depth ``QDEPTH``) and then drains every queue as far as semaphores allow,
    so a race-free stream executes exactly as the decoupled hardware would; a WAIT without its SIGNAL
    or a semaphore past 255 + a full producer queue leaves the machine stuck (STATUS busy), as in RTL."""

    poll_interval = 0

    def __init__(self, hw: HwConfig = KV260, phys_base: int = 0x6000_0000, timing: bool = True):
        self.hw = hw
        self.phys_base = phys_base
        self.timing = timing
        self.mem = None
        self.fs = None
        self.ring_base = 0
        self.ring_size = 0
        self.irq_en = 0
        self.irq_status = 0
        self.irq_code = 0
        self.end_count = 0
        self.step_mode = 0
        self.sem_sel = 0
        self.dbg_last = 0
        self.perf = dict(cycles=0, busy_dma=0, busy_arr=0, busy_vec=0, retired=0, bytes=0)
        self.sat = [0] * R.NSAT
        self.soft_resets = 0
        self._clear()

    # -- state cleared by soft reset / power-on
    def _clear(self):
        self.head = 0
        self.tail = 0
        self.err_code = 0
        self.err_index = 0
        self.err_words = [0, 0, 0, 0]
        self.halted = False
        self.q = [deque() for _ in range(3)]
        self.sem = [0] * R.SEM_COUNT
        self.credit = 0
        self.batch = []           # words dispatched since the last perf update
        if self.fs is not None:
            self.fs.sfu.clear()   # eng_clr: engines reset (SFU tables must be reloaded; programs do)

    # -- memory
    def alloc(self, nbytes):
        if self.mem is not None:
            raise RuntimeError("SimBackend holds one DDR buffer")
        self.mem = np.zeros(nbytes, np.uint8)
        self.fs = FuncSim(self.hw, ddr_base=self.phys_base, ddr_size=nbytes)
        self.fs.mem[DDR] = self.mem          # funcsim works on the driver's buffer in place
        return self.mem, self.phys_base

    # -- CSR read
    def read32(self, off):
        st = self._status
        v = {
            R.ID: R.ID_VALUE, R.VERSION: R.VERSION_VALUE, R.CTRL: self.step_mode,
            R.STATUS: st(), R.ERR_CODE: self.err_code, R.ERR_INDEX: self.err_index,
            R.RING_BASE: self.ring_base, R.RING_SIZE: self.ring_size, R.RING_HEAD: self.head,
            R.DOORBELL: self.tail, R.IRQ_EN: self.irq_en, R.IRQ_STATUS: self.irq_status,
            R.IRQ_CODE: self.irq_code, R.END_COUNT: self.end_count, R.DBG_LAST: self.dbg_last,
            R.SEM_SEL: self.sem_sel, R.SEM_VAL: self.sem[self.sem_sel],
            R.PERF_CYCLES: self.perf["cycles"], R.PERF_BUSY_DMA: self.perf["busy_dma"],
            R.PERF_BUSY_ARR: self.perf["busy_arr"], R.PERF_BUSY_VEC: self.perf["busy_vec"],
            R.PERF_RETIRED: self.perf["retired"], R.PERF_BYTES_LO: self.perf["bytes"] & 0xFFFFFFFF,
            R.PERF_BYTES_HI: self.perf["bytes"] >> 32,
        }
        for i in range(4):
            v[R.ERR_W0 + 4 * i] = self.err_words[i]
        for i in range(R.NSAT):
            v[R.SAT_BASE + 4 * i] = self.sat[i]
        return v.get(off, 0) & 0xFFFFFFFF

    def _cfg_ok(self):
        return self.ring_size != 0 and self.tail < self.ring_size

    def _status(self):
        idle_q = [not q for q in self.q]
        qidle = all(idle_q)
        s = (R.ST_HALTED if self.halted else 0) | (R.ST_QUEUES_IDLE if qidle else 0)
        s |= R.ST_RING_EMPTY if self.head == self.tail else 0
        busy = not qidle or (self.head != self.tail and not self.halted)
        s |= R.ST_BUSY if busy else 0
        s |= sum(int(b) << (R.ST_QIDLE_SHIFT + i) for i, b in enumerate(idle_q))
        s |= R.ST_RING_CFG_OK if self._cfg_ok() else 0
        return s

    # -- CSR write
    def write32(self, off, val):
        val = int(val) & 0xFFFFFFFF
        if off == R.CTRL:
            self.step_mode = val & R.CTRL_STEP
            if val & R.CTRL_SOFT_RESET:
                self.soft_resets += 1
                self._clear()
            self._run()
        elif off == R.RING_BASE:
            self.ring_base = val & ~0xF
        elif off == R.RING_SIZE:
            self.ring_size = val & 0xFFFFFF
        elif off == R.DOORBELL:
            self.tail = val
            if not self._cfg_ok():
                self._error(R.ERR_RING, self.head)
            else:
                self._run()
        elif off == R.IRQ_EN:
            self.irq_en = val & 7
        elif off == R.IRQ_STATUS:
            self.irq_status &= ~val & 7
        elif off == R.DBG_STEP:
            if self.step_mode:
                self.credit += 1
                self._run()
        elif off == R.SEM_SEL:
            self.sem_sel = val % R.SEM_COUNT
        elif off == R.PERF_CTRL:
            if val & 1:
                self.perf = dict.fromkeys(self.perf, 0)
            if val & 2:
                self.sat = [0] * R.NSAT
        # RO / unmapped: ignored

    @property
    def irq(self) -> bool:
        return bool(self.irq_status & self.irq_en)

    # -- execution
    def _error(self, code, index, word=None):
        if self.err_code == 0:  # first error latched
            self.err_code, self.err_index = code, index
            if word is not None:
                w = int.from_bytes(word, "little")
                self.err_words = [(w >> (32 * i)) & 0xFFFFFFFF for i in range(4)]
        self.halted = True
        self.irq_status |= R.IRQ_ERR

    def _word(self, idx):
        o = self.ring_base + 16 * idx - self.phys_base
        if self.mem is None or o < 0 or o + 16 > len(self.mem):
            return None
        return bytes(self.mem[o:o + 16])

    def _exec(self, c: isa.Cmd):
        """Run one engine command in funcsim; update perf and saturation counters."""
        sc = self.fs.sat_count
        tot0, z0 = sum(sc.values()), sc.get("ln_z", 0)
        self.fs.step(c)
        d, dz = sum(sc.values()) - tot0, sc.get("ln_z", 0) - z0
        unit = {"REQUANT": 0, "SOFTMAX_ONLINE": 0, "VEC": 1, "LN": 3}.get(c.op)
        if c.op == "LN":
            self.sat[2] = _sat32(self.sat[2] + dz)
            d -= dz
        if unit is not None and d:
            self.sat[unit] = _sat32(self.sat[unit] + d)
        if c.op in ("DMA_LOAD", "DMA_STORE"):
            self.perf["bytes"] = (self.perf["bytes"] + c["rows"] * c["row_bytes"]) % (1 << 64)

    def _drain(self):
        """Advance every queue head while it can (semaphores); returns True on progress."""
        progress = False
        moved = True
        while moved and not self.halted:
            moved = False
            for qi, q in enumerate(self.q):
                while q and not self.halted:
                    idx, c = q[0]
                    if c.op == "WAIT":
                        if self.sem[c["sem"]] < c["count"]:
                            break
                        self.sem[c["sem"]] -= c["count"]
                    elif c.op == "SIGNAL":
                        if self.sem[c["sem"]] >= R.SEM_MAX:
                            break
                        self.sem[c["sem"]] += 1
                    else:
                        try:
                            self._exec(c)
                        except (SimError, isa.IsaError) as e:   # engine-side fault
                            self.engine_error = str(e)
                            q.popleft()
                            self._error(R.ERR_ENGINE, idx)
                            return True
                    q.popleft()
                    self.perf["retired"] += 1
                    moved = progress = True
        return progress

    def _run(self):
        """Dispatch words from head toward tail (in-order), draining queues as they go."""
        while not self.halted:
            self._drain()
            if self.halted or self.head == self.tail:
                break
            if self.step_mode and self.credit == 0:
                break
            word = self._word(self.head)
            if word is None:
                self._error(R.ERR_FETCH, self.head)
                break
            try:
                c = isa.decode(word)
            except isa.IsaError as e:
                m = str(e)
                code = R.ERR_OPCODE if "opcode" in m else R.ERR_RESV if "reserved" in m else R.ERR_FIELD
                self._error(code, self.head, word)
                break
            if c.queue == isa.Q_CTRL:
                if any(self.q):
                    break                      # barrier: wait for the queues (stuck if they are)
                if c.op == "END":
                    self.irq_status |= R.IRQ_END
                    self.end_count = (self.end_count + 1) & 0xFFFFFFFF
                else:
                    self.irq_status |= R.IRQ_CMD
                    self.irq_code = c["code"] if "code" in c.f else 0
                self.perf["retired"] += 1
            else:
                if len(self.q[c.queue]) >= R.QDEPTH:
                    break                      # full queue blocks the in-order dispatcher
                self.q[c.queue].append((self.head, c))
            self.batch.append(c)
            self.dbg_last = self.head
            self.head = (self.head + 1) % self.ring_size
            if self.step_mode:
                self.credit -= 1
        self._account()

    def _account(self):
        """Perf cycle counters from the cycle model over the words dispatched so far (whole programs)."""
        if not self.timing or not self.batch or any(self.q):
            return
        from ..cyclemodel import ModelError, simulate
        try:
            t = simulate(self.batch, self.hw)
        except ModelError:
            t = None
        self.batch = []
        if t is not None:
            p = self.perf
            p["cycles"] = _sat32(p["cycles"] + t.cycles) & 0xFFFFFFFF
            p["busy_dma"] = (p["busy_dma"] + t.busy["dma"]) & 0xFFFFFFFF
            p["busy_arr"] = (p["busy_arr"] + t.busy["array"]) & 0xFFFFFFFF
            p["busy_vec"] = (p["busy_vec"] + t.busy["vec"]) & 0xFFFFFFFF
