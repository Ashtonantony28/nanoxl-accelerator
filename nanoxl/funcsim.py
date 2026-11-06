"""Functional simulator for the NanoXL command ISA, plus per-command access sets and an
independent happens-before race checker.

Execution is sequential in stream order (the architectural semantics); the race checker proves
that the WAIT/SIGNAL edges in a stream make every conflicting pair of accesses ordered, so the
sequential result is the only possible result on the decoupled DMA / ARRAY / VEC queues.

GEMM, REQUANT, VEC, GELU (LUT) and KV_APPEND are exact (nanoxl.numerics). LN and SOFTMAX_ONLINE
call the FLOAT PLACEHOLDERS in nanoxl.graph until SG5 lands the C5 integer units.
"""
from __future__ import annotations

import numpy as np

from . import isa
from . import numerics as nx
from .graph import placeholder_layernorm, placeholder_softmax, sat
from .hwconfig import HwConfig

DDR = 5  # access-set key for DDR (scratchpads use their buffer ids 0..4)
REQ_REC = 16  # bytes per requant record: int32 bias, mult, shift, pad
LN_HDR = 16   # float32 in_scale, out_scale, eps, pad; then gamma[cols], beta[cols] float32
SM_CTX = 16   # float32 scale, pad x3 (placeholder; online state appended by SG5/SG8)
LUT_BYTES = 256


class SimError(RuntimeError):
    pass


def buf_sizes(hw: HwConfig) -> dict:
    return {isa.ABUF: hw.abuf, isa.WBUF: hw.wbuf, isa.ACC: hw.acc, isa.PBUF: hw.pbuf, isa.KVBUF: hw.kvbuf}


def _l(x):  # line-unit stride -> bytes
    return x * isa.LINE


def accesses(c: isa.Cmd):
    """List of (buf, is_write, addr, row_bytes, rows, stride_bytes) touched by command ``c``."""
    f, op = c.f, c.op
    A = []
    if op in ("DMA_LOAD", "DMA_STORE"):
        spr = (f["sp_buf"], f["sp_addr"], f["row_bytes"], f["rows"], _l(f["sp_stride"]) or f["row_bytes"])
        ddr = (DDR, f["ddr_addr"], f["row_bytes"], f["rows"], f["ddr_stride"] or f["row_bytes"])
        if op == "DMA_LOAD":
            A += [(ddr[0], False) + ddr[1:], (spr[0], True) + spr[1:]]
        else:
            A += [(spr[0], False) + spr[1:], (ddr[0], True) + ddr[1:]]
    elif op == "GEMM":
        M, N, K = f["M"], f["N"], f["K"]
        A.append((isa.ABUF, False, f["a_addr"], K, M, _l(f["a_stride"])))
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
        A.append((isa.PBUF, False, f["param_addr"], LN_HDR + 8 * f["cols"], 1, 0))
        A.append((isa.ABUF, True, f["dst_addr"], f["cols"], f["rows"], _l(f["dst_stride"])))
    elif op == "SOFTMAX_ONLINE":
        A.append((isa.ACC, False, f["src_addr"], 4 * f["cols"], f["rows"], _l(f["src_stride"])))
        A.append((isa.PBUF, False, f["ctx_addr"], SM_CTX + 16 * f["rows"], 1, 0))
        if not f["last"]:
            A.append((isa.PBUF, True, f["ctx_addr"] + SM_CTX, 16 * f["rows"], 1, 0))
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


class FuncSim:
    def __init__(self, hw: HwConfig, image=None, ddr_base: int = 0, ddr_size: int = 0):
        self.hw = hw
        self.mem = {b: np.zeros(n, np.uint8) for b, n in buf_sizes(hw).items()}
        if image is not None:
            ddr_base, ddr_size = image.base, image.size
        self.ddr_base = ddr_base
        self.mem[DDR] = np.zeros(ddr_size, np.uint8)
        if image is not None:
            for off, data in image.chunks:
                self.mem[DDR][off:off + len(data)] = np.frombuffer(data, np.uint8)
        self.stats = {"placeholder_ops": 0, "cmds": 0}

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

    def step(self, c: isa.Cmd) -> None:
        f, op = c.f, c.op
        self.stats["cmds"] += 1
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
        elif op == "GEMM":
            M, N, K = f["M"], f["N"], f["K"]
            a = self.read_t(isa.ABUF, f["a_addr"], K, M, _l(f["a_stride"]), np.int8)
            wbuf = isa.KVBUF if f["w_buf"] else isa.WBUF
            if f["int4"]:
                w = nx.unpack_i4(self.read(wbuf, f["w_addr"], N // 2, K, N // 2))
            elif f["w_trans"]:
                w = self.read_t(wbuf, f["w_addr"], K, N, K, np.int8).T
            else:
                w = self.read_t(wbuf, f["w_addr"], N, K, N, np.int8)
            acc = nx.gemm_i8(np.ascontiguousarray(a), np.ascontiguousarray(w)).astype(np.int64)
            cs = _l(f["c_stride"])
            if f["accumulate"]:
                acc += self.read_t(isa.ACC, f["c_addr"], N, M, cs, np.int32)
            acc = ((acc + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int32)
            self.write(isa.ACC, f["c_addr"], acc, cs)
        elif op == "REQUANT":
            M, N = f["M"], f["N"]
            acc = self.read_t(isa.ACC, f["src_addr"], N, M, _l(f["src_stride"]), np.int32)
            nrec = 1 if f["per_tensor"] else N
            rec = self.read(isa.PBUF, f["param_addr"], REQ_REC * nrec, 1, 0).view(np.int32).reshape(nrec, 4)
            ob = 16 if f["out16"] else 8
            out = nx.requant(acc, rec[:, 1], rec[:, 2], rec[:, 0], ob)
            self.write(isa.ABUF, f["dst_addr"], out, _l(f["dst_stride"]))
        elif op == "VEC":
            self._vec(f)
        elif op == "LN":
            rows, cols = f["rows"], f["cols"]
            x = self.read_t(isa.ABUF, f["src_addr"], cols, rows, _l(f["src_stride"]),
                            np.int16 if f["in16"] else np.int8)
            blk = self.read(isa.PBUF, f["param_addr"], LN_HDR + 8 * cols, 1, 0).view(np.float32).reshape(-1)
            y = placeholder_layernorm(x, blk[4:4 + cols], blk[4 + cols:4 + 2 * cols], blk[0], blk[1])
            self.write(isa.ABUF, f["dst_addr"], y, _l(f["dst_stride"]))
            self.stats["placeholder_ops"] += 1
        elif op == "SOFTMAX_ONLINE":
            if not (f["first"] and f["last"]):
                raise SimError("multi-tile online softmax is SG5/SG8 work; placeholder needs first=last=1")
            rows, cols = f["rows"], f["cols"]
            s = self.read_t(isa.ACC, f["src_addr"], cols, rows, _l(f["src_stride"]), np.int32)
            scale = self.read(isa.PBUF, f["ctx_addr"], 4, 1, 0).view(np.float32)[0, 0]
            p = placeholder_softmax(s, scale, f["diag"], bool(f["causal"]))
            self.write(isa.ABUF, f["dst_addr"], p, _l(f["dst_stride"]))
            self.stats["placeholder_ops"] += 1
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

    def _vec(self, f):
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


def check_hazards(cmds, ddr_base=0):
    """Prove that every pair of conflicting accesses (RAW/WAR/WAW on any 16-byte block of any buffer
    or DDR) in ``cmds`` is ordered by happens-before: same-queue program order plus SIGNAL -> WAIT
    edges (counting semaphores) plus END/IRQ barriers. Also checks that each WAIT's signals appear
    earlier in the stream (the dispatch-deadlock-freedom invariant). Returns summary stats."""
    nq = 3
    vc_q = [np.zeros(nq, np.int64) for _ in range(nq)]  # current vector clock of each queue
    signals = {}   # sem -> list of vector clocks at signal time (stream order)
    consumed = {}
    own = []       # per work command: (queue, own seq number)
    shadow = _Shadow(nq)
    checked = 0
    for c in cmds:
        q = c.queue
        if c.op == "SIGNAL":
            signals.setdefault(c["sem"], []).append(vc_q[q].copy())
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
    return {"work_cmds": len(own), "pairs_checked": checked}
