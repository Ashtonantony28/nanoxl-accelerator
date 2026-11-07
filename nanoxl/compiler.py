"""NanoXL compiler: lowers a nanoxl.graph.Graph (from the SG1 bundle via graph.from_bundle, or
synthetic) to the command ISA (nanoxl.isa). Every op lowers to exact integer commands whose funcsim
semantics are the C5 reference ops (LN, SOFTMAX_ONLINE, GELU, INT16 residual VEC ADD, GEMM_GS for
W4A8 group scales, ADR-004); SFU tables are loaded with LUT_LOAD in each program's prologue.

Pipeline
  1. DDR image (once per model + hardware config): command ring, parameter blocks, pre-tiled
     weight panels, DDR KV region, I/O regions (``DdrImage``).
  2. Lowering (once per step = a set of sequences with (pos, n_new)): ops become work commands
     with K/N tiling for the N x N array, K accumulation, M tiling to fit the accumulator, buffer
     allocation in the named scratchpads (first-fit arena for ABUF, rings for WBUF / PBUF / ACC /
     KV staging).
  3. Sync insertion: an exact dependency pass over 16-byte blocks inserts WAIT/SIGNAL pairs only
     where a command depends on an earlier command of another queue. Weight-ring reuse makes the
     DMA wait for the GEMM two panels back, which is exactly double buffering. Every WAIT's
     SIGNAL precedes it in the stream (dispatch is deadlock-free for any FIFO depth >= 1), and
     ``funcsim.check_hazards`` re-proves race freedom independently.

Alignment rule (docs/isa.md §3): every scratchpad and DDR base address the compiler emits is a
multiple of 16 bytes (allocators round to lines, panels split at lcm(N_array, 16) offsets);
``SyncBuilder.emit`` checks it on every command.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import isa
from . import numerics as nx
from .funcsim import DDR, REQ_REC, SFU_LUT_BYTES, SM_CTX, _Shadow, accesses, ln_block, rect_blocks
from .graph import Graph
from .hwconfig import HwConfig, MiB

ALIGN_DDR = 4096
CH_WEIGHT, CH_ACT, CH_KV = 0, 1, 2

# semaphore id for (producer queue, consumer queue)
SEM = {(isa.Q_DMA, isa.Q_ARR): 0, (isa.Q_DMA, isa.Q_VEC): 1, (isa.Q_ARR, isa.Q_DMA): 2,
       (isa.Q_ARR, isa.Q_VEC): 3, (isa.Q_VEC, isa.Q_DMA): 4, (isa.Q_VEC, isa.Q_ARR): 5}


class CompileError(RuntimeError):
    pass


def rup(x, a):
    return (x + a - 1) // a * a


def lines(nbytes):
    assert nbytes % isa.LINE == 0, nbytes
    return nbytes // isa.LINE


# ---------------------------------------------------------------- allocators

class Arena:
    """First-fit allocator with free (ABUF activations)."""

    def __init__(self, cap, name, align=isa.LINE):
        self.cap, self.name, self.align = cap, name, align
        self.live = {}  # addr -> size
        self.peak = 0

    def alloc(self, n):
        n = rup(max(n, 1), self.align)
        a = 0
        for s, sz in sorted(self.live.items()):
            if a + n <= s:
                break
            a = max(a, s + sz)
        if a + n > self.cap:
            raise CompileError(f"{self.name}: cannot allocate {n} B (capacity {self.cap}, live "
                               f"{sum(self.live.values())})")
        self.live[a] = n
        self.peak = max(self.peak, a + n)
        return a

    def free(self, a):
        del self.live[a]


class Ring:
    """Circular allocator. Unpinned allocations may be overwritten by any later allocation (their
    consumers precede the overwrite in stream order); pinned ones are skipped until released."""

    def __init__(self, cap, name, align=isa.LINE):
        self.cap, self.name, self.align = cap, name, align
        self.head = 0
        self.pinned = {}
        self.peak = 0

    def alloc(self, n, pin=False):
        n = rup(max(n, 1), self.align)
        if n > self.cap:
            raise CompileError(f"{self.name}: {n} B exceeds ring capacity {self.cap}")
        a, wraps = self.head, 0
        while True:
            if a + n > self.cap:
                a, wraps = 0, wraps + 1
                if wraps > 1:
                    raise CompileError(f"{self.name}: no room for {n} B beside pinned {self.pinned}")
            clash = [s + sz for s, sz in self.pinned.items() if s < a + n and a < s + sz]
            if not clash:
                break
            a = max(clash)
        self.head = a + n
        self.peak = max(self.peak, a + n)
        if pin:
            self.pinned[a] = n
        return a

    def release(self, a):
        del self.pinned[a]


# ---------------------------------------------------------------- sync insertion

@dataclass
class _Entry:
    cmd: isa.Cmd
    waits: list = field(default_factory=list)
    signals: list = field(default_factory=list)


class SyncBuilder:
    def __init__(self, ddr_base):
        self.ddr_base = ddr_base
        self.entries = []
        self.qof = []      # queue of each work cmd
        self.seq = []      # own sequence number in its queue
        self.vc = []       # vector clock at completion
        self.vcq = [np.zeros(3, np.int64) for _ in range(3)]
        self.count = [0, 0, 0]
        self.last_sig = {}
        self.shadow = _Shadow(3)
        self.n_waits = 0

    def emit(self, c: isa.Cmd):
        isa.validate(c)
        isa.check_aligned(c)
        q = c.queue
        assert q != isa.Q_CTRL
        vc = self.vcq[q].copy()
        deps = {}
        acc = []
        qof = np.asarray(self.qof, np.int64) if self.qof else np.zeros(0, np.int64)
        for (buf, wr, addr, rb, rows, stride) in accesses(c):
            blk = rect_blocks(addr, rb, rows, stride, self.ddr_base if buf == DDR else 0)
            sh = self.shadow.get(buf, int(blk.max()) + 1)
            arrs = [sh[0][blk]] + ([sh[1 + p][blk] for p in range(3) if p != q] if wr else [])
            for arr in arrs:
                ids = arr[arr >= 0]
                if ids.size:
                    qs = qof[ids]
                    for p in range(3):
                        if p != q:
                            m = ids[qs == p]
                            if m.size:
                                deps[p] = max(deps.get(p, -1), int(m.max()))
            acc.append((buf, blk, wr))
        e = _Entry(c)
        for p, x in sorted(deps.items()):
            if self.seq[x] <= vc[p]:
                continue
            sem = SEM[(p, q)]
            assert self.last_sig.get(sem, -1) < x, "signal order invariant"
            self.last_sig[sem] = x
            self.entries[x].signals.append(isa.C("SIGNAL", queue=p, sem=sem))
            e.waits.append(isa.C("WAIT", queue=q, sem=sem, count=1))
            self.n_waits += 1
            vc = np.maximum(vc, self.vc[x])
        cid = len(self.entries)
        self.count[q] += 1
        vc[q] = self.count[q]
        self.entries.append(e)
        self.qof.append(q)
        self.seq.append(self.count[q])
        self.vc.append(vc.copy())
        self.vcq[q] = vc
        for buf, blk, wr in acc:
            sh = self.shadow.get(buf, int(blk.max()) + 1)  # re-fetch: a later rect may have grown it
            if wr:
                sh[0][blk] = cid
                for p in range(3):
                    sh[1 + p][blk] = -1
            else:
                sh[1 + q][blk] = cid

    def stream(self, end=True):
        out = []
        for e in self.entries:
            out += e.waits
            out.append(e.cmd)
            out += e.signals
        if end:
            out.append(isa.C("END"))
        return out


# ---------------------------------------------------------------- DDR image

@dataclass
class DdrImage:
    base: int
    size: int = 0
    regions: dict = field(default_factory=dict)   # name -> (addr, size)
    chunks: list = field(default_factory=list)    # (offset from base, bytes)
    _cursor: int = 0

    def region(self, name, size):
        addr = self.base + self._cursor
        self.regions[name] = (addr, size)
        self._cursor = rup(self._cursor + size, ALIGN_DDR)
        self.size = self._cursor
        return addr

    def put(self, addr, data: bytes):
        self.chunks.append((addr - self.base, bytes(data)))


@dataclass
class Panel:
    n0: int
    ntc: int        # real output columns
    ntp: int        # columns in the array (even for int4)
    k0: int
    kc: int
    ddr: int
    nbytes: int


@dataclass
class LinearPlan:
    K: int
    N: int
    wbits: int
    out_bits: int
    panels: list
    nt: int
    kc: int
    param_ddr: int = 0
    group: int | None = None   # W4A8 group size (GEMM_GS)
    glog2: int = 0


@dataclass
class Program:
    cmds: list
    io: dict              # tensor -> (ddr_addr, rows, width, bits)
    seqs: list
    stats: dict
    image: DdrImage

    def ring_bytes(self) -> bytes:
        return isa.encode_stream(self.cmds, aligned=True)


def _gs_glog2(group: int) -> int:
    g = int(group).bit_length() - 1
    if (1 << g) != group or not 0 <= g - isa.GS_LOG2_BASE <= 7:
        raise CompileError(f"W4A8 group {group} is not a power of two in [8, 1024] (GEMM_GS.glog2)")
    return g - isa.GS_LOG2_BASE


def _lcm(*xs):
    out = 1
    for x in xs:
        out = out * x // math.gcd(out, x)
    return out


def _rec(bias, mult, shift, ls=0):
    """Requant records [n, 4] int32 {bias, mult, shift, ls} (docs/isa.md §3)."""
    mult = np.atleast_1d(np.asarray(mult, np.int64))
    n = len(mult)
    rec = np.zeros((n, 4), np.int32)
    rec[:, 0] = np.broadcast_to(np.asarray(bias, np.int64), n)
    rec[:, 1] = mult
    rec[:, 2] = np.broadcast_to(np.asarray(shift, np.int64), n)
    rec[:, 3] = ls
    return rec.tobytes()


class Compiler:
    def __init__(self, graph: Graph, params: dict, hw: HwConfig, *, max_rows: int | None = None,
                 n_slots: int = 1, ddr_base: int = 0x4000_0000, cmd_ring_bytes: int = 4 * MiB):
        self.g, self.p, self.hw = graph, params, hw
        self.cfg = graph.cfg
        self.n_slots = n_slots
        self.max_rows = max_rows or (self.cfg.ctx * n_slots if self.cfg else 4096)
        if ddr_base % ALIGN_DDR:
            raise CompileError("DDR base must be 4 KiB aligned")
        if self.cfg and (self.cfg.d_model % isa.LINE or self.cfg.d_head % isa.LINE):
            raise CompileError("alignment rule: d_model and d_head must be multiples of 16 (per-head "
                               "column offsets and KV rows are scratchpad / DDR base addresses)")
        self.img = DdrImage(ddr_base)
        self.img.region("cmd_ring", cmd_ring_bytes)
        self._build_image()

    # -- image
    def plan_linear(self, K, N, wbits, out_bits, group=None) -> LinearPlan:
        """Split a [K, N] weight into DDR panels [kc, nt] (each <= half the WBUF ring, so two are in
        flight). Alignment rule: panel K offsets and column offsets are multiples of lcm(N_array, 16) so
        every A-operand column offset, REQUANT destination and ACC tile is a 16-byte line address;
        W4A8 panels also start on a group boundary (their scale rows travel with the panel)."""
        hw, na = self.hw, self.hw.array_n
        half = min(hw.wbuf // 2, max(hw.panel_bytes, na * na))
        if K > 4096:
            raise CompileError("K > 4096 violates C1")
        nq = _lcm(na, isa.LINE)
        kq = nq if wbits == 8 else _lcm(nq, group)
        glog2 = _gs_glog2(group) if wbits == 4 else 0

        def pbytes(kc, nt):
            if wbits == 8:
                return kc * nt
            return isa.gs_layout(kc, nt + nt % 2, glog2)[2]

        n0 = min(nq, rup(N, 2))
        if pbytes(K, n0) <= half:
            kc = K
        else:
            kc = kq
            while kc + kq <= K and pbytes(kc + kq, n0) <= half:
                kc += kq
            if pbytes(kc, n0) > half:
                raise CompileError(f"WBUF {hw.wbuf} too small for a {kq}-row x {nq}-column panel")
        nt = nq
        while pbytes(kc, nt + nq) <= half:
            nt += nq
        nt = min(nt, rup(N, nq), max(nq, (hw.acc // 4) // 4 // nq * nq), 4092 // nq * nq)
        panels = []
        for n0 in range(0, N, nt):
            ntc = min(nt, N - n0)
            ntp = ntc + (ntc % 2 if wbits == 4 else 0)
            for k0 in range(0, K, kc):
                kcc = min(kc, K - k0)
                panels.append(Panel(n0, ntc, ntp, k0, kcc, 0, pbytes(kcc, ntp)))
        return LinearPlan(K, N, wbits, out_bits, panels, nt, kc, group=group, glog2=glog2)

    def _panel_bytes(self, pr, plan, pn) -> bytes:
        w = pr["w"][pn.k0:pn.k0 + pn.kc, pn.n0:pn.n0 + pn.ntc]
        pad = pn.ntp - pn.ntc
        if pad:
            w = np.concatenate([w, np.zeros((pn.kc, pad), np.int8)], 1)
        if plan.wbits == 8:
            return np.ascontiguousarray(w).tobytes()
        G = plan.group
        packed = nx.pack_i4(np.ascontiguousarray(w)).tobytes()
        gm = pr["gm"][pn.k0 // G: -(-(pn.k0 + pn.kc) // G), pn.n0:pn.n0 + pn.ntc]
        if pad:
            gm = np.concatenate([gm, np.ones((gm.shape[0], pad), np.uint8)], 1)
        _, off, total = isa.gs_layout(pn.kc, pn.ntp, plan.glog2)
        out = bytearray(total)
        out[:len(packed)] = packed
        out[off:] = np.ascontiguousarray(gm, np.uint8).tobytes()
        return bytes(out)

    def _build_image(self):
        g, cfg, img = self.g, self.cfg, self.img
        self.lin, self.blob = {}, {}
        pbytes, wbytes = [], []
        sfu = self.p.get("sfu")
        if any(op.kind in ("ln", "attn") for op in g.ops):
            if sfu is None:
                raise CompileError("LN / softmax need the SFU tables (params['sfu'])")
            pbytes.append(("sfu", b"".join(np.asarray(sfu[k], np.uint16).tobytes()
                                           for k in ("exp2", "recip", "rsqrt"))))
        for op in g.ops:
            if not op.param or op.param in self.lin or op.param in self.blob:
                continue
            pr = self.p[op.param]
            if op.kind == "linear":
                K, N = pr["w"].shape
                if pr["wbits"] == 4 and "gm" not in pr:
                    raise CompileError(f"{op.param}: W4A8 weights need group scales (gm)")
                plan = self.plan_linear(K, N, pr["wbits"], pr["out_bits"], pr.get("group"))
                self.lin[op.param] = plan
                for pn in plan.panels:
                    wbytes.append((op.param, pn, self._panel_bytes(pr, plan, pn)))
                if pr["out_bits"] != 32:
                    pbytes.append((op.param, _rec(pr["bias"], pr["mult"], pr["shift"])))
            elif op.kind == "ln":
                pbytes.append((op.param, ln_block(pr["c"], pr["g"], pr["b"])))
            elif op.kind == "gelu":
                pbytes.append((op.param, np.asarray(pr["lut"], np.int8).tobytes()))
            elif op.kind == "attn":
                if not 0 <= pr["score_ls"] <= 31:
                    raise CompileError("score.ls out of [0, 31]")
                pbytes.append((op.param, _rec(0, pr["score_mult"], pr["score_shift"], pr["score_ls"])
                               + _rec(0, pr["pv_mult"], pr["pv_shift"])))
        p_addr = img.region("params", sum(rup(len(b), 64) for _, b in pbytes))
        for name, b in pbytes:
            img.put(p_addr, b)
            if name in self.lin:
                self.lin[name].param_ddr = p_addr
            else:
                self.blob[name] = (p_addr, len(b))
            p_addr += rup(len(b), 64)
        w_addr = img.region("weights", sum(rup(len(b), 64) for _, _, b in wbytes))
        for _, pn, b in wbytes:
            pn.ddr = w_addr
            img.put(w_addr, b)
            w_addr += rup(len(b), 64)
        if cfg is not None:
            self.kv_slot = 2 * cfg.n_layer * cfg.ctx * cfg.d_model
            if self.hw.kv_onchip:
                if self.kv_slot * self.n_slots > self.hw.kvbuf:
                    raise CompileError(f"on-chip KV: {self.n_slots} x {self.kv_slot} B > KVBUF {self.hw.kvbuf}")
                if self.kv_slot * self.n_slots > 1 << 21:
                    raise CompileError("KVBUF address field is 21 bits")
            else:
                self.kv_ddr = img.region("kv", self.kv_slot * self.n_slots)
        self.io_addr = {}
        for t in list(g.inputs) + list(g.outputs):
            bits, width = g.tensors[t]
            self.io_addr[t] = img.region(f"io_{t}", self.max_rows * width * bits // 8)

    def kv_onchip(self, b, layer, kind, h):
        cfg = self.cfg
        return b * self.kv_slot + ((layer * 2 + kind) * cfg.n_head + h) * cfg.ctx * cfg.d_head

    def kv_ddr_addr(self, b, layer, kind, h):
        return self.kv_ddr + self.kv_onchip(b, layer, kind, h)

    # -- lowering
    def compile_step(self, seqs) -> Program:
        """seqs: list of (pos, n_new) per KV slot (rows concatenated in that order)."""
        g, hw = self.g, self.hw
        M = sum(n for _, n in seqs)
        if M < 1 or M > min(self.max_rows, 4096):
            raise CompileError(f"rows {M} out of range")
        if len(seqs) > self.n_slots:
            raise CompileError("more sequences than KV slots")
        if any(n < 1 for _, n in seqs):
            raise CompileError("every sequence of a step needs at least one new row")
        if self.cfg:
            for pos, n in seqs:
                if pos + n > self.cfg.ctx:
                    raise CompileError("sequence exceeds ctx")
        self.M, self.seqs = M, seqs
        self.sb = SyncBuilder(self.img.base)
        self.abuf = Arena(hw.abuf, "ABUF")
        self.wring = Ring(hw.wbuf, "WBUF")
        self.pring = Ring(hw.pbuf, "PBUF")
        self.aring = Ring(hw.acc, "ACC")
        self.kring = Ring(hw.kvbuf, "KVBUF-staging") if self.cfg and not hw.kv_onchip else None
        self.env = {}
        # liveness: a value dies after its last read; an op whose output name is also an input
        # (add, gelu) updates in place and extends the value's life.
        frees = {i: [] for i in range(len(g.ops))}
        live_until = {}
        for i, op in enumerate(g.ops):
            for t in op.ins:
                live_until[t] = i
            for t in op.outs:
                if t in op.ins:
                    if op.kind not in ("add", "gelu"):
                        raise CompileError(f"{op.kind} cannot run in place")
                elif t in live_until:
                    frees[live_until.pop(t)].append(t)
        for t, i in live_until.items():
            if t not in g.outputs:
                frees[i].append(t)
        if "sfu" in self.blob:
            # prologue: the three C5 SFU tables (exp2, recip, rsqrt) into the SFU's LUT RAMs
            pa = self._load_param("sfu")
            for tbl in (isa.LUT_EXP2, isa.LUT_RECIP, isa.LUT_RSQRT):
                self.emit("LUT_LOAD", table=tbl, src_addr=pa + tbl * SFU_LUT_BYTES)
            self.pring.release(pa)
        for t in g.inputs:
            addr, stride = self._new_tensor(t)
            bits, width = g.tensors[t]
            rb = width * bits // 8
            self._dma("DMA_LOAD", CH_ACT, isa.ABUF, addr, self.io_addr[t], M, rb, rb, lines(stride))
        for i, op in enumerate(g.ops):
            getattr(self, "_lower_" + op.kind)(op)
            for t in frees[i]:
                if t in self.env and not (t in op.outs and t in op.ins):
                    self.abuf.free(self.env.pop(t)[0])
        for t in g.outputs:
            if t in self.env:
                addr, stride = self.env[t]
                bits, width = g.tensors[t]
                rb = width * bits // 8
                self._dma("DMA_STORE", CH_ACT, isa.ABUF, addr, self.io_addr[t], M, rb, rb, lines(stride))
        cmds = self.sb.stream()
        if len(cmds) * isa.WORD_BYTES > self.img.regions["cmd_ring"][1]:
            raise CompileError("command ring overflow")
        io = {t: (self.io_addr[t], M, g.tensors[t][1], g.tensors[t][0]) for t in list(g.inputs) + list(g.outputs)}
        stats = {"cmds": len(cmds), "waits": self.sb.n_waits, "abuf_peak": self.abuf.peak,
                 "acc_peak": self.aring.peak, "wbuf_peak": self.wring.peak, "pbuf_peak": self.pring.peak}
        return Program(cmds, io, list(seqs), stats, self.img)

    def compile_prefill(self, T, chunk=None):
        """Prefill a T-token prompt in slot 0 as a list of Programs (one per chunk). With ``chunk``
        None the largest power-of-two-halving chunk that fits the scratchpads is chosen (chunked
        prefill when a single step does not fit)."""
        chunk = chunk or T
        while chunk >= 1:
            try:
                return [self.compile_step([(p0, min(chunk, T - p0))]) for p0 in range(0, T, chunk)]
            except CompileError as e:
                if "ABUF" not in str(e) and "ACC" not in str(e):
                    raise
                chunk //= 2
        raise CompileError("prefill does not fit even one token per chunk")

    def emit(self, op, note="", **f):
        self.sb.emit(isa.C(op, note, **f))

    def _new_tensor(self, t):
        bits, width = self.g.tensors[t]
        stride = rup(width * bits // 8, isa.LINE)
        addr = self.abuf.alloc(stride * self.M)
        self.env[t] = (addr, stride)
        return addr, stride

    def _out_tensor(self, op, t):
        if t in op.ins:  # in-place
            return self.env[t]
        if t in self.env:
            self.abuf.free(self.env.pop(t)[0])
        return self._new_tensor(t)

    def _dma(self, op, chan, sp_buf, sp_addr, ddr_addr, rows, row_bytes, ddr_stride, sp_stride):
        """Emit one DMA. When both sides are contiguous (ddr_stride and the scratchpad stride equal
        row_bytes) the rows are coalesced into single-row commands of up to 64 KiB (the row_bytes
        field limit): the DMA engine issues at most one burst per row and pays a realigner bubble per
        row, so many short rows are latency bound (docs/control.md §3, ADR-003)."""
        sp_contig = sp_stride == 0 or sp_stride * isa.LINE == row_bytes
        if rows == 1 or (ddr_stride in (0, row_bytes) and sp_contig):
            n = rows * row_bytes
            for off in range(0, n, 65536):
                rb = min(65536, n - off)
                self.emit(op, chan=chan, sp_buf=sp_buf, sp_addr=sp_addr + off, ddr_addr=ddr_addr + off,
                          rows=1, row_bytes=rb, ddr_stride=rb)
            return
        self.emit(op, chan=chan, sp_buf=sp_buf, sp_addr=sp_addr, ddr_addr=ddr_addr, rows=rows,
                  row_bytes=row_bytes, ddr_stride=ddr_stride, sp_stride=sp_stride)

    def _dma_dense(self, sp_buf, sp_addr, ddr_addr, n, chan=CH_WEIGHT):
        self._dma("DMA_LOAD", chan, sp_buf, sp_addr, ddr_addr, 1, n, n, 0)

    def _load_param(self, name, nbytes=None, pin=True):
        addr, size = self.blob[name] if name in self.blob else (self.lin[name].param_ddr, None)
        n = nbytes or size
        a = self.pring.alloc(n, pin=pin)
        self._dma_dense(isa.PBUF, a, addr, n)
        return a

    def _lower_ln(self, op):
        src, ss = self.env[op.ins[0]]
        dst, ds = self._out_tensor(op, op.outs[0])
        bits, d = self.g.tensors[op.ins[0]]
        if bits not in (8, 16):
            raise CompileError("LN input must be int8 or int16")
        pa = self._load_param(op.param)
        self.emit("LN", note=op.param, in16=int(bits == 16), src_addr=src, dst_addr=dst, param_addr=pa,
                  rows=self.M, cols=d, src_stride=lines(ss), dst_stride=lines(ds))
        self.pring.release(pa)

    def _lower_gelu(self, op):
        src, ss = self.env[op.ins[0]]
        dst, ds = self._out_tensor(op, op.outs[0])
        pa = self._load_param(op.param)
        self.emit("GELU", src_addr=src, dst_addr=dst, lut_addr=pa, len=self.M * ss)
        self.pring.release(pa)

    def _lower_add(self, op):
        a, sa = self.env[op.ins[0]]
        b, sb = self.env[op.ins[1]]
        dst, ds = self._out_tensor(op, op.outs[0])
        bits, width = self.g.tensors[op.outs[0]]
        assert sa == sb == ds
        dt = {8: isa.DT_I8, 16: isa.DT_I16, 32: isa.DT_I32}[bits]
        eb = bits // 8
        name = op.attrs.get("name", "resid")
        # padded rows: one command per row so the saturation counter never sees the pad elements
        spans = [(0, self.M * sa // eb)] if width * eb == sa else [(r * sa, width) for r in range(self.M)]
        for bo0, n in spans:
            for off in range(0, n, 1 << 20):
                ln = min(1 << 20, n - off)
                bo = bo0 + off * eb
                self.emit("VEC", note=name, vop=isa.VOPS["ADD"], dt_in=dt, dt_out=dt, src0_buf=isa.ABUF,
                          src0_addr=a + bo, src1_buf=isa.ABUF, src1_addr=b + bo, dst_buf=isa.ABUF,
                          dst_addr=dst + bo, len=ln)

    def _lower_linear(self, op):
        plan = self.lin[op.param]
        hw = self.hw
        a, sa = self.env[op.ins[0]]
        out_name = op.outs[0]
        direct = plan.out_bits == 32
        if direct:
            if out_name not in self.g.outputs:
                raise CompileError("int32 linear outputs must be graph outputs (stored from ACC)")
        else:
            dst, ds = self._out_tensor(op, out_name)
            pa = self._load_param(op.param, REQ_REC * plan.N)
        ob = plan.out_bits // 8
        ntp_max = max(pn.ntp for pn in plan.panels)
        cs = rup(4 * ntp_max, isa.LINE)
        mt = max(1, min(self.M, (hw.acc // 4) // cs))
        if direct and mt < self.M and mt >= 4:
            mt -= mt % 4      # DDR row offsets m0 * 4N stay 16-byte aligned
        gemv = int(self.M <= max(1, hw.array_n // 4))
        for m0 in range(0, self.M, mt):
            mtc = min(mt, self.M - m0)
            tile = None
            for pn in plan.panels:
                if pn.k0 == 0:
                    tile = self.aring.alloc(mtc * cs, pin=True)
                w = self.wring.alloc(pn.nbytes)
                if plan.wbits == 4:
                    self._dma_dense(isa.WBUF, w, pn.ddr, pn.nbytes)
                    self.emit("GEMM_GS", gemv=gemv, accumulate=int(pn.k0 > 0), a_addr=a + m0 * sa + pn.k0,
                              w_addr=w, c_addr=tile, M=mtc, N=pn.ntp, K=pn.kc, a_stride=lines(sa),
                              c_stride=lines(cs), glog2=plan.glog2)
                else:
                    rb = pn.nbytes // pn.kc
                    self._dma("DMA_LOAD", CH_WEIGHT, isa.WBUF, w, pn.ddr, pn.kc, rb, rb, 0)
                    self.emit("GEMM", gemv=gemv, accumulate=int(pn.k0 > 0), a_addr=a + m0 * sa + pn.k0,
                              w_buf=0, w_addr=w, c_addr=tile, M=mtc, N=pn.ntp, K=pn.kc, a_stride=lines(sa),
                              c_stride=lines(cs))
                if pn.k0 + pn.kc == plan.K:
                    if direct:
                        rbo = 4 * plan.N
                        self._dma("DMA_STORE", CH_ACT, isa.ACC, tile, self.io_addr[out_name] + m0 * rbo + 4 * pn.n0,
                                  mtc, 4 * pn.ntc, rbo, lines(cs))
                    else:
                        self.emit("REQUANT", note=op.param, out16=int(ob == 2), src_addr=tile,
                                  dst_addr=dst + m0 * ds + pn.n0 * ob, param_addr=pa + REQ_REC * pn.n0,
                                  M=mtc, N=pn.ntc, src_stride=lines(cs), dst_stride=lines(ds))
                    self.aring.release(tile)
        if not direct:
            self.pring.release(pa)

    def _lower_attn(self, op):
        """Per sequence and head: S = q K^T over all keys 0..pos+n-1 of the step (int32, ACC);
        SOFTMAX_ONLINE requantizes S to int16 Q.8 (score record) and runs the C5 full-row 3-pass
        softmax with the causal mask (P int8 0..127, masked = 0, in ABUF); O = P V (ACC); REQUANT to
        int8 with the PV record. Scores are computed for the full key range of the step, like the
        golden model, so the score saturation counter sees the same elements."""
        cfg, hw = self.cfg, self.hw
        d, H, dh, L = cfg.d_model, cfg.n_head, cfg.d_head, op.attrs["layer"]
        sname = op.attrs.get("score_name", op.param + ".score")
        pvname = op.attrs.get("pv_name", op.param + ".pv")
        qkv, sq = self.env[op.ins[0]]
        att, sa = self._out_tensor(op, op.outs[0])
        pa = self._load_param(op.param)  # [score requant record 16 B][pv requant record 16 B]
        row = 0
        for b, (pos, n) in enumerate(self.seqs):
            t_tot = pos + n
            src = qkv + row * sq
            if hw.kv_onchip:
                for kind in (0, 1):
                    self.emit("KV_APPEND", src_addr=src + (1 + kind) * d, dst_addr=self.kv_onchip(b, L, kind, 0),
                              pos=pos, heads=H, dh=dh, ctx=cfg.ctx, rows=n, src_stride=lines(sq))
            else:
                for kind in (0, 1):
                    for h in range(H):
                        self._dma("DMA_STORE", CH_KV, isa.ABUF, src + (1 + kind) * d + h * dh,
                                  self.kv_ddr_addr(b, L, kind, h) + pos * dh, n, dh, dh, lines(sq))
            cs = rup(4 * t_tot, isa.LINE)
            ps = rup(t_tot, isa.LINE)
            tq = max(1, min(n, (hw.acc // 4) // cs))
            for h in range(H):
                if hw.kv_onchip:
                    ka, va = self.kv_onchip(b, L, 0, h), self.kv_onchip(b, L, 1, h)
                else:
                    ka = self.kring.alloc(t_tot * dh, pin=True)
                    va = self.kring.alloc(t_tot * dh, pin=True)
                    for kind, sp in ((0, ka), (1, va)):
                        self._dma_dense(isa.KVBUF, sp, self.kv_ddr_addr(b, L, kind, h), t_tot * dh, CH_KV)
                for r0 in range(0, n, tq):
                    tqc = min(tq, n - r0)
                    sc = self.aring.alloc(tqc * cs, pin=True)
                    gemv = int(tqc <= max(1, hw.array_n // 4))
                    self.emit("GEMM", gemv=gemv, w_trans=1, a_addr=qkv + (row + r0) * sq + h * dh, w_buf=1,
                              w_addr=ka, c_addr=sc, M=tqc, N=t_tot, K=dh, a_stride=lines(sq), c_stride=lines(cs))
                    P = self.abuf.alloc(tqc * ps)
                    self.emit("SOFTMAX_ONLINE", note=sname, first=1, last=1, causal=1, src_addr=sc, dst_addr=P,
                              ctx_addr=pa, rows=tqc, cols=t_tot, diag=pos + r0, src_stride=lines(cs),
                              dst_stride=lines(ps))
                    self.aring.release(sc)
                    ocs = rup(4 * dh, isa.LINE)
                    o = self.aring.alloc(tqc * ocs, pin=True)
                    self.emit("GEMM", gemv=gemv, a_addr=P, w_buf=1, w_addr=va, c_addr=o, M=tqc, N=dh, K=t_tot,
                              a_stride=lines(ps), c_stride=lines(ocs))
                    self.emit("REQUANT", note=pvname, per_tensor=1, src_addr=o,
                              dst_addr=att + (row + r0) * sa + h * dh, param_addr=pa + SM_CTX, M=tqc, N=dh,
                              src_stride=lines(ocs), dst_stride=lines(sa))
                    self.aring.release(o)
                    self.abuf.free(P)
                if not hw.kv_onchip:
                    self.kring.release(ka)
                    self.kring.release(va)
            row += n
        self.pring.release(pa)


def relocate(cmds, delta):
    """Shift every DDR address in a stream by ``delta`` (driver relocation to the CMA buffer).
    ``delta`` must keep the 16-byte alignment of DDR bases (any 4 KiB-aligned CMA buffer does)."""
    if delta % isa.LINE:
        raise CompileError("relocation must preserve 16-byte DDR alignment")
    out = []
    for c in cmds:
        if c.op in ("DMA_LOAD", "DMA_STORE"):
            c = isa.Cmd(c.op, dict(c.f, ddr_addr=c.f["ddr_addr"] + delta), c.note)
        out.append(c)
    return out
