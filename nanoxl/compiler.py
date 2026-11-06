"""NanoXL compiler skeleton: lowers a nanoxl.graph.Graph to the command ISA (nanoxl.isa).

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
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import isa
from . import numerics as nx
from .funcsim import DDR, LN_HDR, REQ_REC, SM_CTX, _Shadow, accesses, rect_blocks
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


@dataclass
class Program:
    cmds: list
    io: dict              # tensor -> (ddr_addr, rows, width, bits)
    seqs: list
    stats: dict
    image: DdrImage

    def ring_bytes(self) -> bytes:
        return isa.encode_stream(self.cmds)


def _f32(*xs):
    return np.array(xs, np.float32).tobytes()


class Compiler:
    def __init__(self, graph: Graph, params: dict, hw: HwConfig, *, max_rows: int | None = None,
                 n_slots: int = 1, ddr_base: int = 0x4000_0000, cmd_ring_bytes: int = 4 * MiB):
        self.g, self.p, self.hw = graph, params, hw
        self.cfg = graph.cfg
        self.n_slots = n_slots
        self.max_rows = max_rows or (self.cfg.ctx * n_slots if self.cfg else 4096)
        self.img = DdrImage(ddr_base)
        self.img.region("cmd_ring", cmd_ring_bytes)
        self._build_image()

    # -- image
    def plan_linear(self, K, N, wbits, out_bits) -> LinearPlan:
        hw, na = self.hw, self.hw.array_n
        half = min(hw.wbuf // 2, max(hw.panel_bytes, na * na))
        wb = wbits / 8
        if K > 4096:
            raise CompileError("K > 4096 violates C1")
        kc = K if K * na * wb <= half else max(na, int(half / (na * wb)) // na * na)
        if kc * na * wb > half:
            raise CompileError(f"WBUF {hw.wbuf} too small for a {na}x{na} tile")
        nt = na * max(1, int(half // (kc * na * wb)))
        nt = min(nt, rup(N, na), (hw.acc // 4) // 4 // na * na or na, 4092 // na * na)
        panels = []
        for n0 in range(0, N, nt):
            ntc = min(nt, N - n0)
            ntp = ntc + (ntc % 2 if wbits == 4 else 0)
            for k0 in range(0, K, kc):
                kcc = min(kc, K - k0)
                panels.append(Panel(n0, ntc, ntp, k0, kcc, 0, int(kcc * ntp * wb)))
        return LinearPlan(K, N, wbits, out_bits, panels, nt, kc)

    def _build_image(self):
        g, cfg, img = self.g, self.cfg, self.img
        self.lin, self.blob = {}, {}
        pbytes, wbytes = [], []
        for op in g.ops:
            if not op.param or op.param in self.lin or op.param in self.blob:
                continue
            pr = self.p[op.param]
            if op.kind == "linear":
                K, N = pr["w"].shape
                plan = self.plan_linear(K, N, pr["wbits"], pr["out_bits"])
                self.lin[op.param] = plan
                for pn in plan.panels:
                    w = pr["w"][pn.k0:pn.k0 + pn.kc, pn.n0:pn.n0 + pn.ntc]
                    if pn.ntp != pn.ntc:
                        w = np.concatenate([w, np.zeros((pn.kc, pn.ntp - pn.ntc), np.int8)], 1)
                    data = nx.pack_i4(np.ascontiguousarray(w)) if pr["wbits"] == 4 else np.ascontiguousarray(w)
                    wbytes.append((op.param, pn, data.tobytes()))
                if pr["out_bits"] != 32:
                    rec = np.zeros((N, 4), np.int32)
                    rec[:, 0], rec[:, 1], rec[:, 2] = pr["bias"], pr["mult"], pr["shift"]
                    pbytes.append((op.param, rec.tobytes()))
            elif op.kind == "ln":
                pbytes.append((op.param, _f32(pr["in_scale"], pr["out_scale"], 1e-5, 0)
                               + pr["gamma"].tobytes() + pr["beta"].tobytes()))
            elif op.kind == "gelu":
                pbytes.append((op.param, pr["lut"].tobytes()))
            elif op.kind == "attn":
                rec = np.array([0, pr["pv_mult"], pr["pv_shift"], 0], np.int32).tobytes()
                pbytes.append((op.param, _f32(pr["sm_scale"], 0, 0, 0) + rec))
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
        for t in g.inputs:
            a = self._new_tensor(t)
            addr, stride = a
            bits, width = g.tensors[t]
            rb = width * bits // 8
            self.emit("DMA_LOAD", chan=CH_ACT, sp_buf=isa.ABUF, sp_addr=addr, ddr_addr=self.io_addr[t],
                      rows=M, row_bytes=rb, ddr_stride=rb, sp_stride=lines(stride))
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
                self.emit("DMA_STORE", chan=CH_ACT, sp_buf=isa.ABUF, sp_addr=addr, ddr_addr=self.io_addr[t],
                          rows=M, row_bytes=rb, ddr_stride=rb, sp_stride=lines(stride))
        cmds = self.sb.stream()
        if len(cmds) * isa.WORD_BYTES > self.img.regions["cmd_ring"][1]:
            raise CompileError("command ring overflow")
        io = {t: (self.io_addr[t], M, g.tensors[t][1], g.tensors[t][0]) for t in list(g.inputs) + list(g.outputs)}
        stats = {"cmds": len(cmds), "waits": self.sb.n_waits, "abuf_peak": self.abuf.peak,
                 "acc_peak": self.aring.peak, "wbuf_peak": self.wring.peak, "pbuf_peak": self.pring.peak}
        return Program(cmds, io, list(seqs), stats, self.img)

    def compile_prefill(self, T, slot_seqs=None):
        """Prefill a T-token prompt in slot 0, chunking the prompt (chunked prefill) if a single step
        does not fit the scratchpads. Returns a list of Programs (one per chunk)."""
        chunk = T
        while chunk >= 1:
            try:
                progs = []
                for p0 in range(0, T, chunk):
                    progs.append(self.compile_step([(p0, min(chunk, T - p0))]))
                return progs
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

    def _load_param(self, name, nbytes=None, pin=True):
        addr, size = self.blob[name] if name in self.blob else (self.lin[name].param_ddr, None)
        n = nbytes or size
        a = self.pring.alloc(n, pin=pin)
        for off in range(0, n, 65536):
            rb = min(65536, n - off)
            self.emit("DMA_LOAD", chan=CH_WEIGHT, sp_buf=isa.PBUF, sp_addr=a + off, ddr_addr=addr + off,
                      rows=1, row_bytes=rb, ddr_stride=rb)
        return a

    def _lower_ln(self, op):
        src, ss = self.env[op.ins[0]]
        dst, ds = self._out_tensor(op, op.outs[0])
        bits, d = self.g.tensors[op.ins[0]]
        pa = self._load_param(op.param)
        self.emit("LN", in16=int(bits == 16), src_addr=src, dst_addr=dst, param_addr=pa, rows=self.M,
                  cols=d, src_stride=lines(ss), dst_stride=lines(ds))
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
        bits = self.g.tensors[op.outs[0]][0]
        assert sa == sb == ds
        dt = {8: isa.DT_I8, 16: isa.DT_I16, 32: isa.DT_I32}[bits]
        n = self.M * sa // (bits // 8)
        for off in range(0, n, 1 << 20):
            ln = min(1 << 20, n - off)
            bo = off * bits // 8
            self.emit("VEC", vop=isa.VOPS["ADD"], dt_in=dt, dt_out=dt, src0_buf=isa.ABUF, src0_addr=a + bo,
                      src1_buf=isa.ABUF, src1_addr=b + bo, dst_buf=isa.ABUF, dst_addr=dst + bo, len=ln)

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
        gemv = int(self.M <= max(1, hw.array_n // 4))
        for m0 in range(0, self.M, mt):
            mtc = min(mt, self.M - m0)
            tile = None
            for pn in plan.panels:
                if pn.k0 == 0:
                    tile = self.aring.alloc(mtc * cs, pin=True)
                rb = pn.nbytes // pn.kc
                w = self.wring.alloc(pn.nbytes)
                self.emit("DMA_LOAD", chan=CH_WEIGHT, sp_buf=isa.WBUF, sp_addr=w, ddr_addr=pn.ddr,
                          rows=pn.kc, row_bytes=rb, ddr_stride=rb, sp_stride=0)
                self.emit("GEMM", int4=int(plan.wbits == 4), gemv=gemv, accumulate=int(pn.k0 > 0),
                          a_addr=a + m0 * sa + pn.k0, w_buf=0, w_addr=w, c_addr=tile, M=mtc, N=pn.ntp,
                          K=pn.kc, a_stride=lines(sa), c_stride=lines(cs))
                if pn.k0 + pn.kc == plan.K:
                    if direct:
                        rbo = 4 * plan.N
                        self.emit("DMA_STORE", chan=CH_ACT, sp_buf=isa.ACC, sp_addr=tile,
                                  ddr_addr=self.io_addr[out_name] + m0 * rbo + 4 * pn.n0, rows=mtc,
                                  row_bytes=4 * pn.ntc, ddr_stride=rbo, sp_stride=lines(cs))
                    else:
                        self.emit("REQUANT", out16=int(ob == 2), src_addr=tile, dst_addr=dst + m0 * ds + pn.n0 * ob,
                                  param_addr=pa + REQ_REC * pn.n0, M=mtc, N=pn.ntc, src_stride=lines(cs),
                                  dst_stride=lines(ds))
                    self.aring.release(tile)
        if not direct:
            self.pring.release(pa)

    def _lower_attn(self, op):
        cfg, hw = self.cfg, self.hw
        d, H, dh, L = cfg.d_model, cfg.n_head, cfg.d_head, op.attrs["layer"]
        qkv, sq = self.env[op.ins[0]]
        att, sa = self._out_tensor(op, op.outs[0])
        pa = self._load_param(op.param)  # [softmax ctx 16 B][pv requant record 16 B]
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
                        self.emit("DMA_STORE", chan=CH_KV, sp_buf=isa.ABUF, sp_addr=src + (1 + kind) * d + h * dh,
                                  ddr_addr=self.kv_ddr_addr(b, L, kind, h) + pos * dh, rows=n, row_bytes=dh,
                                  ddr_stride=dh, sp_stride=lines(sq))
            cs_full = rup(4 * t_tot, isa.LINE)
            tq = max(1, min(n, (hw.acc // 4) // cs_full))
            for h in range(H):
                if hw.kv_onchip:
                    ka, va = self.kv_onchip(b, L, 0, h), self.kv_onchip(b, L, 1, h)
                else:
                    ka = self.kring.alloc(t_tot * dh, pin=True)
                    va = self.kring.alloc(t_tot * dh, pin=True)
                    for kind, sp in ((0, ka), (1, va)):
                        self.emit("DMA_LOAD", chan=CH_KV, sp_buf=isa.KVBUF, sp_addr=sp,
                                  ddr_addr=self.kv_ddr_addr(b, L, kind, h), rows=1, row_bytes=t_tot * dh)
                for r0 in range(0, n, tq):
                    tqc = min(tq, n - r0)
                    tc = pos + r0 + tqc  # causal: keys 0..tc-1
                    cs = rup(4 * tc, isa.LINE)
                    ps = rup(tc, isa.LINE)
                    sc = self.aring.alloc(tqc * cs, pin=True)
                    gemv = int(tqc <= max(1, hw.array_n // 4))
                    self.emit("GEMM", gemv=gemv, w_trans=1, a_addr=qkv + (row + r0) * sq + h * dh, w_buf=1,
                              w_addr=ka, c_addr=sc, M=tqc, N=tc, K=dh, a_stride=lines(sq), c_stride=lines(cs))
                    P = self.abuf.alloc(tqc * ps)
                    self.emit("SOFTMAX_ONLINE", first=1, last=1, causal=1, src_addr=sc, dst_addr=P, ctx_addr=pa,
                              rows=tqc, cols=tc, diag=pos + r0, src_stride=lines(cs), dst_stride=lines(ps))
                    self.aring.release(sc)
                    ocs = rup(4 * dh, isa.LINE)
                    o = self.aring.alloc(tqc * ocs, pin=True)
                    self.emit("GEMM", gemv=gemv, a_addr=P, w_buf=1, w_addr=va, c_addr=o, M=tqc, N=dh, K=tc,
                              a_stride=lines(ps), c_stride=lines(ocs))
                    self.emit("REQUANT", per_tensor=1, src_addr=o, dst_addr=att + (row + r0) * sa + h * dh,
                              param_addr=pa + SM_CTX, M=tqc, N=dh, src_stride=lines(ocs), dst_stride=lines(sa))
                    self.aring.release(o)
                    self.abuf.free(P)
                if not hw.kv_onchip:
                    self.kring.release(ka)
                    self.kring.release(va)
            row += n
        self.pring.release(pa)


def relocate(cmds, delta):
    """Shift every DDR address in a stream by ``delta`` (driver relocation to the CMA buffer)."""
    out = []
    for c in cmds:
        if c.op in ("DMA_LOAD", "DMA_STORE"):
            c = isa.Cmd(c.op, dict(c.f, ddr_addr=c.f["ddr_addr"] + delta), c.note)
        out.append(c)
    return out
