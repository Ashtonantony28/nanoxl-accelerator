"""Cycle-approximate performance model of a NanoXL command stream, calibrated to the measured RTL.

Every cost below is a measured number from the RTL docs (wave 2), not an estimate:
docs/control.md (command fetch, dispatch, queues, DMA engine), docs/array.md (C3 tile timing) and
docs/sfu.md (LN / softmax / LUT timing). The only free parameters are the board-level DDR
bandwidth and latency in ``HwConfig`` (unknown until M7) and the "what-if" switches in
``RtlTiming`` used to price proposed fixes. The older estimate knobs of ``HwConfig``
(``wload_rows_per_cycle``, ``issue_cycles``, ``fifo_depth``, ``vec_latency``, ``row_latency_*``,
``sfu_row_bubble``, ``fill_drain``) are no longer read.

Machine model (docs/control.md §1)
  * **Fetch**: the command ring is read over AXI in INCR bursts of up to 8 words, one burst in
    flight, into a 16-word prefetch FIFO whose space is reserved before AR; bursts never cross
    the ring end or a 4 KiB page. A word arrives ``ddr latency + k + 1`` cycles after its burst issues.
  * **Dispatch**: one word per cycle in stream order into three in-order queues (DMA, ARRAY, VEC)
    of depth 8. A full queue blocks dispatch. An entry leaves the queue when it issues.
  * **Queues**: one command at a time per engine in this model. WAIT(s, k) blocks until k more
    SIGNALs fired; SIGNAL fires once every earlier command of its queue has completed (results
    visible). IRQ / END are barriers.
  * Because every WAIT's SIGNALs precede it in the stream (compiler invariant, checked), one pass in
    stream order computes exact event times.

Per-command costs (cycles; N = array size, L = vector lanes; occupancy = engine busy, then latency
until results are visible):
  DMA       not pipelined against each other: occ = AR latency + data beats + 1 realigner bubble per
            row; beats = sum over rows of the 16-byte beats the row touches (address-aligned), paced
            by min(port bytes/cycle, DDR bytes/cycle); at most 4 bursts (<= 256 beats, no 4 KiB
            crossing) in flight bound strided small rows to 4 bursts per latency. STORE: done at the
            last B response (same formula).
  GEMM      C3 tiles: for each N-tile and K-tile, max(M, k_tile) cycles (one weight row and one
            activation row per cycle, zero inter-tile bubbles; INT4 still moves one K-row per beat);
            latency 2N + 4 (first a beat -> first out beat).
  GEMM_GS   as GEMM (K-tiles split at group boundaries), plus one scale beat on the first K-tile
            of each group: max(M, k_tile + 1) (ADR-004).
  VEC unit  each vec_unit row command costs 1 accept cycle + its beats (docs/vector-unit.md §5.4):
            VEC 1 + ceil(len/L); REQUANT M (1 + ceil(N/L)); KV_APPEND rows x heads x (1 + ceil(dh/L));
            GELU 1 + ceil(len/L) (+16 table load when not resident); latency 6 (pipeline).
  LN        rows x (1 + 3 ceil(cols/L) + 19), rows non-overlapping (one row buffer)
            + LN block load 2 + ceil(cols/8) + ceil(cols/4) when the block is not resident.
  SOFTMAX   rows x (1 + 3 ceil(cols/L) + 19) (SOFTMAX_RQ, int32 scores) + 1 record load if not resident.
  LUT_LOAD  32 cycles per table (one 16-byte PBUF line per cycle).
Residency: the VEC engine keeps an LN block / GELU table / score record resident per PBUF address
until a DMA writes that PBUF range (the current compiler reloads parameters every program, so in
practice every LN pays its block load).

``simulate(..., trace=True)`` also returns a critical-path attribution (``Timeline.critical``):
the longest chain of events (fetch, dispatch, command occupancy/latency) ending at the last
completion, summed by op class (``op_class``). ``Timeline.busy_by_class`` is engine occupancy.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

from . import isa
from .funcsim import ln_block_layout
from .hwconfig import HwConfig


class ModelError(RuntimeError):
    pass


def _cd(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class RtlTiming:
    """Measured RTL timing (defaults) plus what-if switches for proposed fixes."""
    # control plane (docs/control.md §1)
    fetch_over_axi: bool = True    # False: what-if, words available at dispatch rate (on-chip ring)
    fetch_burst: int = 8           # words per fetch burst
    fetch_fifo: int = 16           # prefetch FIFO words (space reserved before AR)
    qdepth: int = 8                # per-queue command FIFO
    # DMA engine (docs/control.md §3)
    dma_beat_bytes: int = 16       # 128-bit AXI4 master
    axi_port_bytes: int = 16       # PS HP port width (KV260 128-bit; Zynq-7000 HP is 64-bit)
    dma_masters: int = 1           # what-if: >1 masters on separate HP ports, beats striped
    dma_pipelined: bool = False    # what-if: next command's AR overlaps this command's data
    signal_blocks: bool = True     # SIGNAL at a queue head blocks the queue until it fires; what-if
                                   # False: the signal rides on the command's done (needed for DMA pipelining)
    dma_row_bubble: int = 1        # realigner bubble per row
    dma_coalesce: bool = False     # what-if: contiguous rows (ds = ss = row_bytes) merged into one row
    dma_max_bursts: int = 4        # outstanding bursts
    dma_max_beats: int = 256       # beats per burst
    # array (docs/array.md §2, C3)
    array_rows_per_cycle: int = 1  # what-if: weight rows per cycle into the array (ADR-003 proposed 2)
    array_lat_extra: int = 4       # first output = 2N + 4
    # vector unit / SFU (docs/vector-unit.md §5.4, docs/sfu.md)
    vec_accept: int = 1            # command-accept cycle per vec_unit row command
    vec_latency: int = 6
    ln_row_fixed: int = 19         # LN: 4 mu + 3 pass-2 tail + 6 rsqrt + 6 pass-3 tail
    sm_row_fixed: int = 19         # SOFTMAX_RQ: 6 requant + 3 + 5 recip + 5
    sfu_row_buffers: int = 1       # what-if 2: ping-pong row buffer overlaps pass 1 of row n+1
    lut_load_cycles: int = 32      # per LUT_LOAD table
    gelu_table_cycles: int = 16
    params_resident: bool = False  # what-if: LN blocks / GELU tables / score records stay in SFU slots

    def with_(self, **kw) -> "RtlTiming":
        return replace(self, **kw)


MEASURED = RtlTiming()
BOARD_TIMING = {"kv260": MEASURED, "pynq-z2": MEASURED.with_(axi_port_bytes=8)}


def timing_for(hw: HwConfig) -> RtlTiming:
    return BOARD_TIMING.get(hw.name, MEASURED)


def ddr_latency(hw: HwConfig) -> int:
    return math.ceil(hw.ddr_latency_ns * hw.clock_mhz / 1e3)


# ---------------------------------------------------------------- op classes

CLASSES = ("dma weights", "dma params", "dma act/kv", "gemm linear", "gemm attn", "requant", "ln",
           "softmax", "gelu", "vec add", "kv append", "lut load", "sync", "cmd fetch", "dispatch")


def op_class(c: isa.Cmd) -> str:
    op = c.op
    if op in ("DMA_LOAD", "DMA_STORE"):
        b = c["sp_buf"]
        if op == "DMA_LOAD" and b == isa.WBUF:
            return "dma weights"
        if op == "DMA_LOAD" and b == isa.PBUF:
            return "dma params"
        return "dma act/kv"
    if op == "GEMM":
        return "gemm attn" if c["w_buf"] or c["w_trans"] else "gemm linear"
    return {"GEMM_GS": "gemm linear", "ATTN_TILE": "gemm attn", "REQUANT": "requant", "LN": "ln",
            "SOFTMAX_ONLINE": "softmax", "GELU": "gelu", "VEC": "vec add", "KV_APPEND": "kv append",
            "LUT_LOAD": "lut load"}.get(op, "sync")


# ---------------------------------------------------------------- per-command costs

def _dma_cost(f, hw, tm):
    rows, rb = f["rows"], f["row_bytes"]
    ds = f["ddr_stride"] or rb
    ss = (f["sp_stride"] * isa.LINE) or rb
    if tm.dma_coalesce and rows > 1 and ds == rb and ss == rb:
        rows, rb = 1, rows * rb
    total = rows * rb
    BB = tm.dma_beat_bytes
    a0 = f["ddr_addr"]
    if ds % BB == 0 or rows == 1:
        beats = rows * _cd(a0 % BB + rb, BB)
    else:
        beats = sum(_cd((a0 + r * ds) % BB + rb, BB) for r in range(rows))
    # one burst per row at least (bursts never cross a row end), split at dma_max_beats and 4 KiB pages
    bursts = rows * (_cd(_cd(rb, BB), tm.dma_max_beats) + (a0 % 4096 + rb - 1) // 4096)
    lat = ddr_latency(hw)
    port = min(BB, tm.axi_port_bytes) * tm.dma_masters
    bpc = min(port, hw.ddr_bytes_per_cycle)
    data = max(math.ceil(total / bpc), math.ceil(beats / tm.dma_masters),
               (_cd(bursts, tm.dma_max_bursts) - 1) * lat)
    data += rows * tm.dma_row_bubble
    if tm.dma_pipelined:
        return data, lat, {"ddr_bytes": total}
    return lat + data, 0, {"ddr_bytes": total}


def _gemm_tiles(M, N, K, na, kt_max, rpc, scale_every=None):
    """Sum over C3 tiles of max(M, k_tile / rpc) (+1 scale beat on group-first tiles)."""
    cyc = 0
    k = 0
    while k < K:
        kt = min(kt_max, K - k)
        w = _cd(kt, rpc)
        if scale_every is not None and k % scale_every == 0:
            w += 1
        cyc += max(M, w)
        k += kt
    return cyc * _cd(N, na)


def cost(c: isa.Cmd, hw: HwConfig, tm: RtlTiming | None = None, resident: bool = False):
    """(occupancy, latency, info) for one command; info carries ddr_bytes / macs. ``resident`` says the
    command's SFU parameter block (LN block, GELU table, score record) is already loaded."""
    tm = tm or timing_for(hw)
    f, op, na, L = c.f, c.op, hw.array_n, hw.lanes
    alat = 2 * na + tm.array_lat_extra
    if op in ("DMA_LOAD", "DMA_STORE"):
        return _dma_cost(f, hw, tm)
    if op == "GEMM":
        M, N, K = f["M"], f["N"], f["K"]
        return _gemm_tiles(M, N, K, na, na, tm.array_rows_per_cycle), alat, {"macs": M * N * K}
    if op == "GEMM_GS":
        M, N, K = f["M"], f["N"], f["K"]
        G = isa.gs_group(f["glog2"])
        # tiles split at group boundaries: tile height min(N, G) when G divides into tiles
        kt = min(na, G)
        occ = _gemm_tiles(M, N, K, na, kt, tm.array_rows_per_cycle, scale_every=G)
        return occ, alat, {"macs": M * N * K}
    if op == "ATTN_TILE":
        rows, ln = f["rows"], f["len"]
        occ = 2 * _cd(ln, na) * 2 * max(rows, na)
        return occ, alat, {"macs": 2 * rows * ln * 64}
    acc, vl = tm.vec_accept, tm.vec_latency
    if op == "LUT_LOAD":
        return tm.lut_load_cycles, 0, {}
    if op == "VEC":
        return acc + _cd(f["len"], L), vl, {}
    if op == "GELU":
        return acc + _cd(f["len"], L) + (0 if resident else tm.gelu_table_cycles), vl, {}
    if op == "REQUANT":
        return f["M"] * (acc + _cd(f["N"], L)), vl, {}
    if op == "KV_APPEND":
        return f["rows"] * f["heads"] * (acc + _cd(f["dh"], L)), vl, {}
    if op in ("LN", "SOFTMAX_ONLINE"):
        rows, cols = f["rows"], f["cols"]
        nb = _cd(cols, L)
        fixed = tm.ln_row_fixed if op == "LN" else tm.sm_row_fixed
        if tm.sfu_row_buffers >= 2:
            # pass 1 of row n+1 overlaps passes 2-3 of row n (modelled, not measured)
            occ = rows * (acc + 2 * nb + fixed) + nb
        else:
            occ = rows * (acc + 3 * nb + fixed)
        if not resident:
            occ += (2 + _cd(cols, 8) + _cd(cols, 4)) if op == "LN" else 1
        return occ, 0, {}
    if op in ("WAIT", "SIGNAL", "IRQ", "END"):
        return 0, 0, {}
    raise ModelError(op)


def _param_key(c):
    f = c.f
    if c.op == "LN":
        return f["param_addr"], f["param_addr"] + ln_block_layout(f["cols"])[2]
    if c.op == "GELU":
        return f["lut_addr"], f["lut_addr"] + 256
    if c.op == "SOFTMAX_ONLINE":
        return f["ctx_addr"], f["ctx_addr"] + 16
    return None


# ---------------------------------------------------------------- timeline

@dataclass
class Timeline:
    cycles: int
    hw: HwConfig
    busy: dict                 # queue name -> busy cycles
    ddr_bytes: int
    macs: int
    n_cmds: int
    max_outstanding_sem: int
    trace: list = field(default_factory=list)          # (cmd index, queue, start, end) if requested
    busy_by_class: dict = field(default_factory=dict)  # op class -> engine occupancy cycles
    critical: dict = field(default_factory=dict)       # op class -> cycles on the critical path (trace=True)

    @property
    def seconds(self):
        return self.cycles / self.hw.clock_hz

    @property
    def array_util(self):
        return self.macs / (self.hw.array_n ** 2 * max(self.cycles, 1))

    @property
    def ddr_util(self):
        return self.ddr_bytes / (self.hw.ddr_bytes_per_cycle * max(self.cycles, 1))

    def summary(self):
        return {"cycles": self.cycles, "us": round(self.seconds * 1e6, 2),
                "array_util": round(self.array_util, 4), "ddr_util": round(self.ddr_util, 4),
                "ddr_MB": round(self.ddr_bytes / 1e6, 3), "busy": dict(self.busy), "cmds": self.n_cmds}


class _Events:
    """Event DAG for critical-path attribution: each event has a time, the event that determined it
    and the class of the segment between them."""

    def __init__(self):
        self.t, self.pred, self.lab = [0], [-1], [None]   # event 0 = time zero

    def add(self, cands, track):
        """cands: iterable of (time, pred_event, label). Returns (time, event id)."""
        t, p, *lab = max(cands, key=lambda x: x[0])
        lab = lab[0] if lab else None
        if not track:
            return t, 0
        self.t.append(t)
        self.pred.append(p)
        self.lab.append(lab)
        return t, len(self.t) - 1

    def attribute(self, e):
        out = {}
        while e > 0:
            p = self.pred[e]
            dt = self.t[e] - self.t[p]
            if dt:
                out[self.lab[e]] = out.get(self.lab[e], 0) + dt
            e = p
        return out


def simulate(cmds, hw: HwConfig, trace=False, tm: RtlTiming | None = None) -> Timeline:
    tm = tm or timing_for(hw)
    nq = 3
    lat = ddr_latency(hw)
    ev = _Events()
    E0 = (0, 0, None)
    busy_until = [E0] * nq          # (time, event) engine free for the next command
    done_until = [E0] * nq          # (time, event) all issued commands of the queue have completed
    qissue = [deque() for _ in range(nq)]   # (issue time, event) of the last qdepth entries per queue
    busy = [0] * nq
    by_class = {}
    sig = {}                        # sem -> list of (fire time, event)
    consumed = {}
    disp = E0                       # last dispatch (time, event)
    disp_t = []                     # dispatch time of every word (fetch FIFO space)
    burst = None                    # (issue time, first word index, event of issue)
    last_burst_done = E0
    resident = {}                   # (lo, hi) PBUF ranges of resident SFU parameter blocks
    ddr = macs = 0
    max_out = 0
    tr = []
    for i, c in enumerate(cmds):
        q = c.queue
        # -- fetch: bursts of fetch_burst words, one in flight, FIFO space reserved before AR
        if tm.fetch_over_axi:
            if i % tm.fetch_burst == 0:
                n = min(tm.fetch_burst, len(cmds) - i)
                cand = [last_burst_done]
                j = i + n - tm.fetch_fifo - 1   # this word's dispatch frees the FIFO space
                if j >= 0:
                    cand.append(disp_t[j])
                bt, be = ev.add([(t, e, "cmd fetch") for t, e, *_ in cand], trace)
                burst = (bt, i, be)
                last_burst_done = ev.add([(bt + lat + n, be, "cmd fetch")], trace)
            arr = (burst[0] + lat + i - burst[1] + 1, burst[2], "cmd fetch")
        else:
            arr = E0
        cand = [arr, (disp[0] + 1, disp[1], "dispatch")]
        if q != isa.Q_CTRL and len(qissue[q]) >= tm.qdepth:
            it, ie = qissue[q][0]
            cand.append((it, ie, "dispatch"))
        dt, de = ev.add(cand, trace)
        disp = (dt, de)
        disp_t.append(disp)
        if q == isa.Q_CTRL:
            st, se = ev.add([(dt, de, "sync")] + [(x[0], x[1], "sync") for x in done_until + busy_until], trace)
            for j in range(nq):
                busy_until[j] = done_until[j] = (st, se)
            disp = (st, se)
            if trace:
                tr.append((i, q, st, st))
            continue
        res = False
        key = _param_key(c)
        if key is not None:
            res = tm.params_resident or key in resident
            resident[key] = True
        occ, l, info = cost(c, hw, tm, resident=res)
        if c.op == "DMA_LOAD" and c["sp_buf"] == isa.PBUF and resident:
            lo, hi = c["sp_addr"], c["sp_addr"] + c["rows"] * max(c["sp_stride"] * isa.LINE, c["row_bytes"])
            for k2 in [k2 for k2 in resident if k2[0] < hi and lo < k2[1]]:
                del resident[k2]
        cand = [(dt, de, "dispatch"), busy_until[q]]
        if c.op == "WAIT":
            s, k = c["sem"], c["count"]
            need = consumed.get(s, 0) + k
            got = sig.get(s, [])
            if len(got) < need:
                raise ModelError(f"cmd {i}: WAIT sem {s} needs signal #{need}, only {len(got)} precede it")
            cand.append(got[need - 1])
            consumed[s] = need
        elif c.op == "SIGNAL" and tm.signal_blocks:
            # head of its queue until every earlier command of the queue has completed
            cand.append(done_until[q])
        cls = op_class(c)
        st, se = ev.add([(t, e, cls) for t, e, *_ in cand], trace)
        qissue[q].append((st, se))
        if len(qissue[q]) > tm.qdepth:
            qissue[q].popleft()
        et, ee = ev.add([(st + occ, se, cls)], trace)
        if c.op == "SIGNAL":
            ft, fe = (et, ee) if tm.signal_blocks else ev.add([(et, ee, "sync"), done_until[q]], trace)
            lst = sig.setdefault(c["sem"], [])
            if lst and ft < lst[-1][0]:
                raise ModelError("semaphore with more than one producer queue")
            lst.append((ft, fe))
            max_out = max(max_out, len(lst) - consumed.get(c["sem"], 0))
        busy_until[q] = (et, ee)
        ct, ce = ev.add([(et + l, ee, cls)], trace)
        if ct >= done_until[q][0]:
            done_until[q] = (ct, ce)
        busy[q] += occ
        by_class[cls] = by_class.get(cls, 0) + occ
        ddr += info.get("ddr_bytes", 0)
        macs += info.get("macs", 0)
        if trace:
            tr.append((i, q, st, ct))
    final = max(done_until + busy_until + [disp], key=lambda x: x[0])
    names = {isa.Q_DMA: "dma", isa.Q_ARR: "array", isa.Q_VEC: "vec"}
    crit = ev.attribute(final[1]) if trace else {}
    return Timeline(final[0], hw, {names[q]: busy[q] for q in range(nq)}, ddr, macs, len(cmds), max_out, tr,
                    by_class, crit)


# ---------------------------------------------------------------- model-level estimates

def decode_positions(ctx, n=5):
    return sorted({min(ctx - 1, int(round(x))) for x in [i * (ctx - 1) / (n - 1) for i in range(n)]})


def _merge(dst, src, w=1.0):
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + w * v


def estimate(cfg, hw: HwConfig, batch=1, prefill_len=None, params=None, n_pos=5, tm: RtlTiming | None = None,
             breakdown=False):
    """Prefill latency (slot 0, T = prefill_len or ctx) and decode tokens/s at ``batch`` sequences,
    averaged over decode positions spread across the context. Returns a dict; on-chip KV that does
    not fit returns {'fits': False, ...}. ``breakdown`` adds critical-path cycles per op class
    (``prefill_critical`` summed over chunks, ``decode_critical`` averaged over positions)."""
    from .compiler import Compiler, CompileError
    from .graph import build_gpt_graph, make_synthetic_params
    tm = tm or timing_for(hw)
    g = build_gpt_graph(cfg)
    params = params or make_synthetic_params(g, 0)
    res = {"model": cfg.name, "wbits": cfg.wbits, "board": hw.name, "N": hw.array_n, "lanes": hw.lanes,
           "mhz": hw.clock_mhz, "ddr_gbps": hw.ddr_gbps, "kv": "onchip" if hw.kv_onchip else "ddr",
           "batch": batch}
    try:
        comp = Compiler(g, params, hw, n_slots=batch)
    except CompileError as e:
        res.update(fits=False, why=str(e))
        return res
    res["fits"] = True
    if prefill_len != 0:
        T = prefill_len or cfg.ctx
        progs = comp.compile_prefill(T)
        tls = [simulate(p.cmds, hw, trace=breakdown, tm=tm) for p in progs]
        cyc = sum(t.cycles for t in tls)
        res["prefill_T"] = T
        res["prefill_chunk"] = progs[0].seqs[0][1]
        res["prefill_ms"] = cyc / hw.clock_hz * 1e3
        res["prefill_array_util"] = sum(t.macs for t in tls) / (hw.array_n ** 2 * cyc)
        res["prefill_gops"] = 2 * sum(t.macs for t in tls) / (cyc / hw.clock_hz) / 1e9
        if breakdown:
            crit, bb = {}, {}
            for t in tls:
                _merge(crit, t.critical)
                _merge(bb, t.busy_by_class)
            res["prefill_cycles"] = cyc
            res["prefill_critical"] = crit
            res["prefill_busy"] = bb
    times, ddr, util, macs = [], [], [], []
    crit, bb = {}, {}
    pos = decode_positions(cfg.ctx, n_pos)
    for p in pos:
        prog = comp.compile_step([(p, 1)] * batch)
        t = simulate(prog.cmds, hw, trace=breakdown, tm=tm)
        times.append(t.seconds)
        ddr.append(t.ddr_bytes)
        util.append(t.ddr_util)
        macs.append(t.macs)
        _merge(crit, t.critical, 1 / len(pos))
        _merge(bb, t.busy_by_class, 1 / len(pos))
    step = sum(times) / len(times)
    res["decode_step_us"] = step * 1e6
    res["decode_tok_s"] = batch / step
    res["decode_ddr_MB_per_step"] = sum(ddr) / len(ddr) / 1e6
    res["decode_ddr_util"] = sum(util) / len(util)
    res["decode_ops_per_byte"] = 2 * sum(macs) / max(sum(ddr), 1)
    res["decode_gops"] = 2 * sum(macs) / len(macs) / step / 1e9
    if breakdown:
        res["decode_cycles"] = step * hw.clock_hz
        res["decode_critical"] = crit
        res["decode_busy"] = bb
    # simple bound: stream all GEMM weights once per step at the configured bandwidth
    res["weight_bound_tok_s"] = batch * hw.ddr_gbps * 1e9 / cfg.weight_bytes
    return res


# what-if fixes priced by bench/roofline.py (ADR-003 "bottlenecks")
FIXES = {
    "coalesce contiguous DMA rows": dict(dma_coalesce=True),
    "DMA command pipelining": dict(dma_pipelined=True, signal_blocks=False),
    "second DMA master (2 HP ports)": dict(dma_masters=2),
    "on-chip / prefetched command ring": dict(fetch_over_axi=False),
    "SFU second row buffer": dict(sfu_row_buffers=2),
    "resident SFU parameter slots": dict(params_resident=True),
    "2-row array weight port": dict(array_rows_per_cycle=2),
}
