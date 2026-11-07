"""Functional simulator for the NanoXL command ISA, plus per-command access sets, an independent
happens-before race checker and a host-driver model (``HostRunner``).

Execution is sequential in stream order (the architectural semantics); the race checker proves
that the WAIT/SIGNAL edges in a stream make every conflicting pair of accesses ordered, so the
sequential result is the only possible result on the decoupled DMA / ARRAY / VEC queues.

Every command is exact integer arithmetic. GEMM / GEMM_GS / REQUANT / VEC use nanoxl.numerics
(C1/C2); LN, SOFTMAX_ONLINE and GELU call the C5 reference ops in nanoxl.quant.ops with the
parameter blocks and SFU tables (LUT_LOAD) the program put in PBUF. There are no float paths.

Saturation events are counted per command into ``FuncSim.sat_count`` under the command's
``note`` (the compiler annotates each saturating command with the golden model's counter name:
the linear name, h{l}.score, h{l}.pv, resid, the LN name) and, for LN's normalize clamp,
"ln_z". This mirrors nanoxl.golden.Golden.sat_count exactly.
"""
from __future__ import annotations

import numpy as np

from . import isa
from . import numerics as nx
from .graph import group_gemm, requant_ref, sat
from .hwconfig import HwConfig
from .quant import ops as qops

DDR = 5  # access-set key for DDR (scratchpads use their buffer ids 0..4)
REQ_REC = 16  # bytes per requant record: int32 bias, mult, shift, ls (ls must be 0 for REQUANT)
LN_HDR = 32   # int32 mu_mult, mu_shift, sqrt_d, sh, int64 eps_q, 8 B zero; then g int16[cols] (line-padded), b int32[cols]
SM_CTX = 16   # softmax ctx = the score requant record {0, mult, shift, ls}
LUT_BYTES = 256              # GELU table: int8[256]
SFU_LUT_BYTES = 2 * isa.LUT_ENTRIES   # LUT_LOAD table: uint16[256]


class SimError(RuntimeError):
    pass


def _rup(x, a):
    return (x + a - 1) // a * a


def ln_block_layout(cols: int):
    """(g offset, b offset, total bytes) of an LN parameter block (docs/isa.md §3)."""
    g_off = LN_HDR
    b_off = g_off + _rup(2 * cols, isa.LINE)
    return g_off, b_off, b_off + 4 * cols


def ln_block(c, g, b) -> bytes:
    """Pack bundle LN constants (.c int64[5] = mu_mult, mu_shift, sqrt_d, eps_q, sh; g int16; b int32)."""
    c = [int(v) for v in c]
    cols = len(g)
    g_off, b_off, n = ln_block_layout(cols)
    if not (0 <= c[3] < 1 << 47):
        raise ValueError("eps_q out of the 48-bit LN accumulator range")
    out = np.zeros(n, np.uint8)
    out[0:16] = np.frombuffer(np.array([c[0], c[1], c[2], c[4]], np.int32).tobytes(), np.uint8)
    out[16:24] = np.frombuffer(np.array([c[3]], np.int64).tobytes(), np.uint8)
    out[g_off:g_off + 2 * cols] = np.frombuffer(np.asarray(g, np.int16).tobytes(), np.uint8)
    out[b_off:b_off + 4 * cols] = np.frombuffer(np.asarray(b, np.int32).tobytes(), np.uint8)
    return out.tobytes()


def buf_sizes(hw: HwConfig) -> dict:
    return {isa.ABUF: hw.abuf, isa.WBUF: hw.wbuf, isa.ACC: hw.acc, isa.PBUF: hw.pbuf, isa.KVBUF: hw.kvbuf}


def _l(x):  # line-unit stride -> bytes
    return x * isa.LINE


def accesses(c: isa.Cmd):
    """List of (buf, is_write, addr, row_bytes, rows, stride_bytes) touched by command ``c``.
    (The SFU tables written by LUT_LOAD are private to the VEC queue, so they are not listed.)"""
    f, op = c.f, c.op
    A = []
    if op in ("DMA_LOAD", "DMA_STORE"):
        spr = (f["sp_buf"], f["sp_addr"], f["row_bytes"], f["rows"], _l(f["sp_stride"]) or f["row_bytes"])
        ddr = (DDR, f["ddr_addr"], f["row_bytes"], f["rows"], f["ddr_stride"] or f["row_bytes"])
        if op == "DMA_LOAD":
            A += [(ddr[0], False) + ddr[1:], (spr[0], True) + spr[1:]]
        else:
            A += [(spr[0], False) + spr[1:], (ddr[0], True) + ddr[1:]]
    elif op in ("GEMM", "GEMM_GS"):
        M, N, K = f["M"], f["N"], f["K"]
        A.append((isa.ABUF, False, f["a_addr"], K, M, _l(f["a_stride"])))
        if op == "GEMM_GS":
            wb = isa.gs_layout(K, N, f["glog2"])[2]
            A.append((isa.WBUF, False, f["w_addr"], wb, 1, wb))
        else:
            wb = K * N // 2 if f["int4"] else K * N
            A.append((isa.KVBUF if f["w_buf"] else isa.WBUF, False, f["w_addr"], wb, 1, wb))
        cr = (isa.ACC, f["c_addr"], 4 * N, M, _l(f["c_stride"]))
        if f["accumulate"]:
            A.append((cr[0], False) + cr[1:])
        A.append((cr[0], True) + cr[1:])
    elif op == "VEC":
        n = f["len"]
        A.append((f["src0_buf"], False, f["src0_addr"], n * isa.DT_BYTES[f["dt_in"]], 1, 0))
        if isa.VOP_NAMES[f["vop"]] in ("ADD", "SUB", "MUL", "MAX", "MIN"):
            A.append((f["src1_buf"], False, f["src1_addr"], n * isa.DT_BYTES[f["dt_in"]], 1, 0))
        A.append((f["dst_buf"], True, f["dst_addr"], n * isa.DT_BYTES[f["dt_out"]], 1, 0))
    elif op == "REQUANT":
        M, N = f["M"], f["N"]
        A.append((isa.ACC, False, f["src_addr"], 4 * N, M, _l(f["src_stride"])))
        A.append((isa.PBUF, False, f["param_addr"], REQ_REC * (1 if f["per_tensor"] else N), 1, 0))
        A.append((isa.ABUF, True, f["dst_addr"], N * (2 if f["out16"] else 1), M, _l(f["dst_stride"])))
    elif op == "LN":
        eb = 2 if f["in16"] else 1
        A.append((isa.ABUF, False, f["src_addr"], f["cols"] * eb, f["rows"], _l(f["src_stride"])))
        A.append((isa.PBUF, False, f["param_addr"], ln_block_layout(f["cols"])[2], 1, 0))
        A.append((isa.ABUF, True, f["dst_addr"], f["cols"], f["rows"], _l(f["dst_stride"])))
    elif op == "SOFTMAX_ONLINE":
        A.append((isa.ACC, False, f["src_addr"], 4 * f["cols"], f["rows"], _l(f["src_stride"])))
        A.append((isa.PBUF, False, f["ctx_addr"], SM_CTX, 1, 0))
        A.append((isa.ABUF, True, f["dst_addr"], f["cols"], f["rows"], _l(f["dst_stride"])))
    elif op == "GELU":
        A.append((isa.ABUF, False, f["src_addr"], f["len"], 1, 0))
        A.append((isa.PBUF, False, f["lut_addr"], LUT_BYTES, 1, 0))
        A.append((isa.ABUF, True, f["dst_addr"], f["len"], 1, 0))
    elif op == "KV_APPEND":
        H, dh, rows = f["heads"], f["dh"], f["rows"]
        A.append((isa.ABUF, False, f["src_addr"], H * dh, rows, _l(f["src_stride"])))
        for h in range(H):
            A.append((isa.KVBUF, True, f["dst_addr"] + (h * f["ctx"] + f["pos"]) * dh, rows * dh, 1, 0))
    elif op == "LUT_LOAD":
        A.append((isa.PBUF, False, f["src_addr"], SFU_LUT_BYTES, 1, 0))
    elif op == "ATTN_TILE":
        A.append((isa.ABUF, False, f["q_addr"], 64, f["rows"], 64))
        A.append((isa.KVBUF, False, f["k_addr"], 64 * f["len"], 1, 0))
        A.append((isa.KVBUF, False, f["v_addr"], 64 * f["len"], 1, 0))
        A.append((isa.PBUF, False, f["ctx_addr"], SM_CTX + 16 * f["rows"], 1, 0))
        A.append((isa.ACC, True, f["o_addr"], 4 * 64, f["rows"], 256))
    return A


def rect_bytes(addr, row_bytes, rows, stride):
    """Byte range [lo, hi) spanned by a rectangle."""
    return addr, addr + (rows - 1) * stride + row_bytes


def _wrap32(x):
    return ((np.asarray(x, np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int32)


class FuncSim:
    def __init__(self, hw: HwConfig, image=None, ddr_base: int = 0, ddr_size: int = 0, strict_align: bool = True):
        self.hw = hw
        self.mem = {b: np.zeros(n, np.uint8) for b, n in buf_sizes(hw).items()}
        if image is not None:
            ddr_base, ddr_size = image.base, image.size
        self.ddr_base = ddr_base
        self.mem[DDR] = np.zeros(ddr_size, np.uint8)
        if image is not None:
            for off, data in image.chunks:
                self.mem[DDR][off:off + len(data)] = np.frombuffer(data, np.uint8)
        self.strict_align = strict_align
        self.sfu = {}            # SFU tables loaded by LUT_LOAD: name -> uint16[256]
        self.sat_count = {}      # saturation events, golden-model names
        self.stats = {"cmds": 0}

    # -- raw access
    def _off(self, buf, addr):
        return addr - self.ddr_base if buf == DDR else addr

    def read(self, buf, addr, row_bytes, rows, stride) -> np.ndarray:
        m = self.mem[buf]
        o = self._off(buf, addr)
        lo, hi = rect_bytes(o, row_bytes, rows, stride)
        if lo < 0 or hi > len(m):
            raise SimError(f"read out of bounds: {_bn(buf)}[{lo}:{hi}] (size {len(m)})")
        if rows == 1:
            return m[o:o + row_bytes].reshape(1, row_bytes).copy()
        idx = o + np.arange(rows)[:, None] * stride + np.arange(row_bytes)[None, :]
        return m[idx]

    def write(self, buf, addr, data: np.ndarray, stride) -> None:
        data = np.ascontiguousarray(data)
        rows = data.shape[0]
        data = data.view(np.uint8).reshape(rows, -1)
        row_bytes = data.shape[1]
        m = self.mem[buf]
        o = self._off(buf, addr)
        lo, hi = rect_bytes(o, row_bytes, rows, stride)
        if lo < 0 or hi > len(m):
            raise SimError(f"write out of bounds: {_bn(buf)}[{lo}:{hi}] (size {len(m)})")
        if rows == 1 or stride == row_bytes:
            m[o:o + rows * row_bytes] = data.reshape(-1)
        else:
            idx = o + np.arange(rows)[:, None] * stride + np.arange(row_bytes)[None, :]
            m[idx] = data

    def read_t(self, buf, addr, cols, rows, stride, dtype):
        eb = np.dtype(dtype).itemsize
        return self.read(buf, addr, cols * eb, rows, stride).view(dtype).reshape(rows, cols)

    # -- host-side I/O through the program's DDR regions
    def write_input(self, prog, name, arr):
        addr, rows, width, bits = prog.io[name]
        dt = {8: np.int8, 16: np.int16, 32: np.int32}[bits]
        arr = np.asarray(arr)
        assert arr.shape == (rows, width), (arr.shape, rows, width)
        self.write(DDR, addr, arr.astype(dt), width * bits // 8)

    def read_output(self, prog, name):
        addr, rows, width, bits = prog.io[name]
        dt = {8: np.int8, 16: np.int16, 32: np.int32}[bits]
        return self.read_t(DDR, addr, width, rows, width * bits // 8, dt)

    # -- execution
    def run(self, cmds) -> None:
        for c in cmds:
            self.step(c)

    def _sat(self, key, y, bits):
        qops.sat(y, bits, counter=self.sat_count, name=key)

    def _lut(self, name):
        if name not in self.sfu:
            raise SimError(f"SFU table {name} used before LUT_LOAD")
        return self.sfu[name]

    def step(self, c: isa.Cmd) -> None:
        f, op = c.f, c.op
        self.stats["cmds"] += 1
        key = c.note or op.lower()
        if self.strict_align:
            try:
                isa.check_aligned(c)
            except isa.IsaError as e:
                raise SimError(str(e)) from None
        if op in ("WAIT", "SIGNAL", "IRQ", "END"):
            return
        if op in ("DMA_LOAD", "DMA_STORE"):
            spb, sps = f["sp_buf"], _l(f["sp_stride"]) or f["row_bytes"]
            ds = f["ddr_stride"] or f["row_bytes"]
            if op == "DMA_LOAD":
                d = self.read(DDR, f["ddr_addr"], f["row_bytes"], f["rows"], ds)
                self.write(spb, f["sp_addr"], d, sps)
            else:
                d = self.read(spb, f["sp_addr"], f["row_bytes"], f["rows"], sps)
                self.write(DDR, f["ddr_addr"], d, ds)
        elif op in ("GEMM", "GEMM_GS"):
            M, N, K = f["M"], f["N"], f["K"]
            a = np.ascontiguousarray(self.read_t(isa.ABUF, f["a_addr"], K, M, _l(f["a_stride"]), np.int8))
            if op == "GEMM_GS":
                _, off, _ = isa.gs_layout(K, N, f["glog2"])
                G = isa.gs_group(f["glog2"])
                w = nx.unpack_i4(self.read(isa.WBUF, f["w_addr"], N // 2, K, N // 2))
                gm = self.read(isa.WBUF, f["w_addr"] + off, N, -(-K // G), N)
                acc = group_gemm(a, w, gm, G)
            else:
                wbuf = isa.KVBUF if f["w_buf"] else isa.WBUF
                if f["int4"]:
                    w = nx.unpack_i4(self.read(wbuf, f["w_addr"], N // 2, K, N // 2))
                elif f["w_trans"]:
                    w = self.read_t(wbuf, f["w_addr"], K, N, K, np.int8).T
                else:
                    w = self.read_t(wbuf, f["w_addr"], N, K, N, np.int8)
                acc = nx.gemm_i8(a, np.ascontiguousarray(w)).astype(np.int64)
            cs = _l(f["c_stride"])
            if f["accumulate"]:
                acc = acc + self.read_t(isa.ACC, f["c_addr"], N, M, cs, np.int32)
            self.write(isa.ACC, f["c_addr"], _wrap32(acc), cs)
        elif op == "REQUANT":
            M, N = f["M"], f["N"]
            acc = self.read_t(isa.ACC, f["src_addr"], N, M, _l(f["src_stride"]), np.int32)
            nrec = 1 if f["per_tensor"] else N
            rec = self.read(isa.PBUF, f["param_addr"], REQ_REC * nrec, 1, 0).view(np.int32).reshape(nrec, 4)
            if np.any(rec[:, 3]):
                raise SimError("REQUANT record word 3 (ls) must be 0")
            ob = 16 if f["out16"] else 8
            out = requant_ref(acc, rec[:, 1], rec[:, 2], rec[:, 0], ob, self.sat_count, key)
            self.write(isa.ABUF, f["dst_addr"], out, _l(f["dst_stride"]))
        elif op == "VEC":
            self._vec(f, key)
        elif op == "LUT_LOAD":
            t = self.read(isa.PBUF, f["src_addr"], SFU_LUT_BYTES, 1, 0).view(np.uint16).reshape(-1)
            self.sfu[isa.LUT_NAMES[f["table"]]] = t.copy()
        elif op == "LN":
            rows, cols = f["rows"], f["cols"]
            x = self.read_t(isa.ABUF, f["src_addr"], cols, rows, _l(f["src_stride"]),
                            np.int16 if f["in16"] else np.int8)
            g_off, b_off, n = ln_block_layout(cols)
            blk = self.read(isa.PBUF, f["param_addr"], n, 1, 0).reshape(-1)
            h = blk[:16].view(np.int32)
            cd = dict(mu_mult=int(h[0]), mu_shift=int(h[1]), sqrt_d=int(h[2]),
                      eps_q=int(blk[16:24].view(np.int64)[0]))
            g = blk[g_off:g_off + 2 * cols].view(np.int16)
            b = blk[b_off:b_off + 4 * cols].view(np.int32)
            z = qops.ln_norm_i(x, cd, self._lut("rsqrt"), counter=self.sat_count)
            y = qops.ln_affine_i(z, g, b, int(h[3]), counter=self.sat_count, name=key)
            self.write(isa.ABUF, f["dst_addr"], y, _l(f["dst_stride"]))
        elif op == "SOFTMAX_ONLINE":
            if not (f["first"] and f["last"]):
                raise SimError("C5 softmax is a full-row 3-pass: SOFTMAX_ONLINE needs first=last=1 (ADR-002)")
            rows, cols = f["rows"], f["cols"]
            acc = self.read_t(isa.ACC, f["src_addr"], cols, rows, _l(f["src_stride"]), np.int32)
            rec = self.read(isa.PBUF, f["ctx_addr"], SM_CTX, 1, 0).view(np.int32).reshape(-1)
            if not 0 <= rec[3] <= 31:
                raise SimError("softmax score pre-shift ls out of [0, 31]")
            s = requant_ref(acc, int(rec[1]), int(rec[2]), int(rec[0]), 16, self.sat_count, key, int(rec[3]))
            if f["causal"]:
                mask = np.arange(cols)[None, :] <= (np.arange(rows)[:, None] + f["diag"])
            else:
                mask = np.ones((rows, cols), bool)
            if not mask.any(axis=1).all():
                raise SimError("softmax row with no visible column")
            p = qops.softmax_i(s, mask, self._lut("exp2"), self._lut("recip"))
            self.write(isa.ABUF, f["dst_addr"], p, _l(f["dst_stride"]))
        elif op == "GELU":
            n = f["len"]
            x = self.read(isa.ABUF, f["src_addr"], n, 1, n)
            lut = self.read(isa.PBUF, f["lut_addr"], LUT_BYTES, 1, 0).reshape(-1)
            self.write(isa.ABUF, f["dst_addr"], lut[x], n)
        elif op == "KV_APPEND":
            H, dh, rows, ctx = f["heads"], f["dh"], f["rows"], f["ctx"]
            src = self.read(isa.ABUF, f["src_addr"], H * dh, rows, _l(f["src_stride"]))
            if f["pos"] + rows > ctx:
                raise SimError("KV_APPEND past ctx")
            for h in range(H):
                self.write(isa.KVBUF, f["dst_addr"] + (h * ctx + f["pos"]) * dh,
                           np.ascontiguousarray(src[:, h * dh:(h + 1) * dh]), dh)
        elif op == "ATTN_TILE":
            raise SimError("ATTN_TILE (fused flash-decoding tile) is reserved for SG8")
        else:
            raise SimError(op)

    def _vec(self, f, key):
        n = f["len"]
        din = {0: np.int8, 1: np.int16, 2: np.int32}[f["dt_in"]]
        obits = 8 * isa.DT_BYTES[f["dt_out"]]
        vop = isa.VOP_NAMES[f["vop"]]
        x = self.read_t(f["src0_buf"], f["src0_addr"], n, 1, 0, din).astype(np.int64)
        if vop in ("ADD", "SUB", "MUL", "MAX", "MIN"):
            y = self.read_t(f["src1_buf"], f["src1_addr"], n, 1, 0, din).astype(np.int64)
        imm = f["imm"]
        if vop == "COPY":
            r = x
        elif vop == "ADD":
            r = x + y
        elif vop == "SUB":
            r = x - y
        elif vop == "MUL":
            r = x * y
        elif vop == "MAX":
            r = np.maximum(x, y)
        elif vop == "MIN":
            r = np.minimum(x, y)
        elif vop == "ADDI":
            r = x + imm
        elif vop == "SHRR":
            if not 0 <= imm <= 31:
                raise SimError("SHRR imm must be in [0,31]")
            r = nx.rdivpot(np.clip(x, nx.INT32_MIN, nx.INT32_MAX), imm).astype(np.int64)
        elif vop == "SHL":
            if not 0 <= imm <= 31:
                raise SimError("SHL imm must be in [0,31]")
            r = x << imm
        self._sat(key, r, obits)
        self.write(f["dst_buf"], f["dst_addr"], sat(r, obits), 0)


def _bn(b):
    return "DDR" if b == DDR else isa.BUF_NAMES[b]


# ---------------------------------------------------------------- happens-before race checker

BLOCK = 16


def rect_blocks(addr, row_bytes, rows, stride, base=0):
    a = addr - base
    if rows == 1 or stride == row_bytes:
        return np.arange(a // BLOCK, (a + rows * row_bytes - 1) // BLOCK + 1)
    starts = a + np.arange(rows) * stride
    lo, hi = starts // BLOCK, (starts + row_bytes - 1) // BLOCK
    if (hi - lo).max() == 0:
        return np.unique(lo)
    return np.unique(np.concatenate([np.arange(l, h + 1) for l, h in zip(lo, hi)]))


class _Shadow:
    """Per 16-byte block: last writer command id and last reader id per queue (-1 = none)."""

    def __init__(self, n_queues=3):
        self.nq = n_queues
        self.m = {}

    def get(self, buf, nblocks):
        if buf not in self.m or len(self.m[buf][0]) < nblocks:
            old = self.m.get(buf)
            n = max(nblocks, 1024) if old is None else max(nblocks, 2 * len(old[0]))
            new = [np.full(n, -1, np.int64) for _ in range(1 + self.nq)]
            if old is not None:
                for a, b in zip(new, old):
                    a[:len(b)] = b
            self.m[buf] = new
        return self.m[buf]


class RaceError(AssertionError):
    pass


SEM_MAX = 255       # RTL semaphore counter saturates (a SIGNAL at 255 stalls, docs/control.md §1)
QUEUE_DEPTH = 8     # RTL per-queue command FIFO (QDEPTH)
SEM_STREAM_LIMIT = SEM_MAX + QUEUE_DEPTH


def check_hazards(cmds, ddr_base=0, sem_limit=SEM_STREAM_LIMIT):
    """Prove that every pair of conflicting accesses (RAW/WAR/WAW on any 16-byte block of any buffer
    or DDR) in ``cmds`` is ordered by happens-before: same-queue program order plus SIGNAL -> WAIT
    edges (counting semaphores) plus END/IRQ barriers. Also checks that each WAIT's signals appear
    earlier in the stream (the dispatch-deadlock-freedom invariant), and that no semaphore ever has more
    than ``sem_limit`` = 255 + 8 SIGNALs outstanding ahead of their WAITs in stream order: past that the
    counter holds 255, the producer queue fills with stalled SIGNALs, the in-order dispatcher blocks and
    the consuming WAIT is never dispatched (deadlock, docs/control.md §1 "Stream limit").
    Returns summary stats (incl. ``max_outstanding_sem``)."""
    nq = 3
    vc_q = [np.zeros(nq, np.int64) for _ in range(nq)]  # current vector clock of each queue
    signals = {}   # sem -> list of vector clocks at signal time (stream order)
    consumed = {}
    own = []       # per work command: (queue, own seq number)
    shadow = _Shadow(nq)
    checked = 0
    max_out = 0
    for c in cmds:
        q = c.queue
        if c.op == "SIGNAL":
            lst = signals.setdefault(c["sem"], [])
            lst.append(vc_q[q].copy())
            out = len(lst) - consumed.get(c["sem"], 0)
            if out > sem_limit:
                raise RaceError(f"semaphore {c['sem']}: {out} SIGNALs outstanding ahead of their WAITs "
                                f"(> {sem_limit} = 255 + queue depth 8): the RTL in-order dispatcher deadlocks")
            max_out = max(max_out, out)
            continue
        if c.op == "WAIT":
            s, k = c["sem"], c["count"]
            got = signals.get(s, [])
            n0 = consumed.get(s, 0)
            if len(got) < n0 + k:
                raise RaceError(f"WAIT on sem {s} needs signal #{n0 + k} but only {len(got)} precede it")
            for v in got[n0:n0 + k]:
                vc_q[q] = np.maximum(vc_q[q], v)
            consumed[s] = n0 + k
            continue
        if c.op in ("END", "IRQ"):
            j = np.max(np.stack(vc_q), axis=0)
            for i in range(nq):
                vc_q[i] = j.copy()
            continue
        vc_q[q][q] += 1
        cid = len(own)
        own.append((q, vc_q[q][q]))
        vc = vc_q[q]
        for (buf, wr, addr, rb, rows, stride) in accesses(c):
            base = ddr_base if buf == DDR else 0
            blk = rect_blocks(addr, rb, rows, stride, base)
            sh = shadow.get(buf, int(blk.max()) + 1)
            prev = [sh[0][blk]] + ([sh[1 + p][blk] for p in range(nq)] if wr else [])
            for arr in prev:
                ids = np.unique(arr[arr >= 0])
                for pid in ids:
                    pq, pseq = own[pid]
                    if pq != q and pseq > vc[pq]:
                        raise RaceError(f"race: cmd #{cid} {c} vs earlier #{pid} on {_bn(buf)}"
                                        f" ({'write' if wr else 'read'})")
                    checked += 1
            if wr:
                sh[0][blk] = cid
                for p in range(nq):
                    sh[1 + p][blk] = -1
            else:
                sh[1 + q][blk] = cid
    return {"work_cmds": len(own), "pairs_checked": checked, "max_outstanding_sem": max_out}


# ---------------------------------------------------------------- host driver model (v1)

class HostRunner:
    """The v1 host driver around a compiled model and one FuncSim (docs/isa.md §6): the host does the
    embedding lookup ``x0 = sat16(emb.tok[t] + emb.pos[p])`` (counted as "resid", like the golden
    model), writes x0 to the program's input region, runs one compiled program per step and does the
    greedy argmax (ties -> lowest index). KV caches persist in the simulator between steps.

    ``arrays, meta`` is an SG1 bundle (nanoxl.export.load_bundle). Slot b of a batched step is KV slot b."""

    def __init__(self, arrays, meta, hw: HwConfig, n_slots: int = 1, name: str = "bundle", **compiler_kw):
        from .compiler import Compiler
        from .graph import from_bundle
        self.A = arrays
        g, params = from_bundle(arrays, meta, name)
        self.cfg = g.cfg
        self.comp = Compiler(g, params, hw, n_slots=n_slots, **compiler_kw)
        self.sim = FuncSim(hw, self.comp.img)
        self.host_sat = {}
        self.pos = [0] * n_slots
        self.programs = 0

    @property
    def sat_count(self) -> dict:
        out = dict(self.host_sat)
        for k, v in self.sim.sat_count.items():
            out[k] = out.get(k, 0) + v
        return out

    def embed(self, tokens, p0):
        t = np.asarray(tokens, np.int64)
        x = self.A["emb.tok"][t].astype(np.int64) + self.A["emb.pos"][p0:p0 + len(t)].astype(np.int64)
        return qops.sat(x, 16, counter=self.host_sat, name="resid").astype(np.int16)

    def step(self, tokens_per_slot):
        """Append tokens_per_slot[b] (non-empty list) to slot b for every b; returns int16 logits per slot."""
        seqs = [(self.pos[b], len(t)) for b, t in enumerate(tokens_per_slot)]
        prog = self.comp.compile_step(seqs)
        x0 = np.concatenate([self.embed(t, self.pos[b]) for b, t in enumerate(tokens_per_slot)])
        self.sim.write_input(prog, "x", x0)
        self.sim.run(prog.cmds)
        self.programs += 1
        logits = self.sim.read_output(prog, "logits")
        out, r = [], 0
        for b, (_, n) in enumerate(seqs):
            out.append(logits[r:r + n].copy())
            r += n
            self.pos[b] += n
        return out

    def prefill(self, prompt, chunk=None):
        """Slot-0 prefill (single-slot runner), optionally in chunks of ``chunk`` tokens (chunked
        prefill); returns the logits of every prompt row."""
        assert len(self.pos) == 1, "prefill() drives slot 0 of a single-slot runner; use step() for batches"
        chunk = chunk or len(prompt)
        rows = []
        for i in range(0, len(prompt), chunk):
            rows.append(self.step([list(prompt[i:i + chunk])])[0])
        return np.concatenate(rows)

    def generate(self, prompt, n_new, chunk=None, trace=None):
        """Greedy decode in slot 0, the same loop as nanoxl.golden.Golden.generate (use_cache=True).
        ``trace`` (list) receives the logits of every step."""
        logits = self.prefill(prompt, chunk)
        if trace is not None:
            trace.append(logits)
        out = []
        for _ in range(n_new):
            nxt = int(np.argmax(logits[-1]))
            out.append(nxt)
            if len(prompt) + len(out) >= self.cfg.ctx or len(out) >= n_new:
                break
            logits = self.step([[nxt]])[0]
            if trace is not None:
                trace.append(logits)
        return out

    def kv_cache(self, slot=0):
        """KV contents of a slot as [layer, K|V, head, ctx, d_head] int8 (on-chip tier or DDR)."""
        cfg, comp = self.cfg, self.comp
        out = np.zeros((cfg.n_layer, 2, cfg.n_head, cfg.ctx, cfg.d_head), np.int8)
        for l in range(cfg.n_layer):
            for kind in (0, 1):
                for h in range(cfg.n_head):
                    if self.sim.hw.kv_onchip:
                        buf, a = isa.KVBUF, comp.kv_onchip(slot, l, kind, h)
                    else:
                        buf, a = DDR, comp.kv_ddr_addr(slot, l, kind, h)
                    n = cfg.ctx * cfg.d_head
                    out[l, kind, h] = self.read_kv(buf, a, n).reshape(cfg.ctx, cfg.d_head)
        return out

    def read_kv(self, buf, addr, n):
        return self.sim.read(buf, addr, n, 1, n).view(np.int8).reshape(-1)
