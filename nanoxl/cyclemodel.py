"""Cycle-approximate performance model of a NanoXL command stream, calibrated to the measured RTL.

Every cost below is calibrated to measured RTL, not estimated: docs/control.md (command fetch,
dispatch, queues, DMA engine), docs/array.md §7 (array_seq tile timing, measured at N = 32) and
docs/vector-unit.md §9.3 / bench/results/vec_seq_perf.json (the VEC engine, rtl/vec_seq.sv,
measured per command at L = 4, 16, 64). The variable parts of the VEC costs follow from the port
and lane widths; only per-command / per-row fixed costs are fitted (``VEC_CAL``, within 5 % of every
measured shape). The only free parameters are the board-level DDR bandwidth and latency in
``HwConfig`` (unknown until M7) and the "what-if" switches in ``RtlTiming`` used to price proposed
fixes. The older estimate knobs of ``HwConfig`` (``wload_rows_per_cycle``, ``issue_cycles``,
``fifo_depth``, ``vec_latency``, ``row_latency_*``, ``sfu_row_bubble``, ``fill_drain``) are not read.

Machine model (docs/control.md §1)
  * **Fetch**: the command ring is read over AXI in INCR bursts of up to 8 words, one burst in
    flight, into a 16-word prefetch FIFO whose space is reserved before AR; bursts never cross
    the ring end or a 4 KiB page. A word arrives ``ddr latency + k + 1`` cycles after its burst issues.
  * **Dispatch**: one word per cycle in stream order into three in-order queues (DMA, ARRAY, VEC)
    of depth 8. A full queue blocks dispatch. An entry leaves the queue when it issues.
  * **Queues**: one command at a time per engine in this model; the DMA queue has two engines
    (channel ``chan & 1``, ADR-005), issued in queue order. WAIT(s, k) blocks until k more SIGNALs
    fired; SIGNAL fires once every earlier command of its queue (both DMA channels) has completed
    (results visible) and does not block its queue (carried on done). A DMA_STORE is a fence: it
    starts when both DMA channels are done and nothing starts until it is done. IRQ / END are
    barriers.
  * Because every WAIT's SIGNALs precede it in the stream (compiler invariant, checked), one pass in
    stream order computes exact event times.

Per-command costs (cycles; N = array size, L = vector lanes; occupancy = engine busy, then latency
until results are visible):
  DMA       pipelined (SG10a): occ = data beats + 1 realigner bubble per row + ``dma_cmd_gap``
            (descriptor hand-off), latency = AR latency, so the next command's ARs overlap this one's
            data; beats = sum over rows of the 16-byte beats the row touches (address-aligned), paced
            by min(port bytes/cycle, DDR bytes/cycle / channels busy); at most 4 bursts (<= 256
            beats, no 4 KiB crossing) in flight bound strided small rows to 4 bursts per latency.
            STORE: done at the last B response (same formula). ``dma_pipelined=False``,
            ``dma_masters=1``, ``signal_blocks=True`` give the pre-SG10a single engine.
  GEMM      C3 tiles (array_seq): for each N-tile and K-tile, max(M, W-port beats of the tile)
            cycles: one weight row per beat (INT4 too); a weight row segment that straddles the
            ceil(N/16)-line W window (ragged rows such as the 65-byte LM head) costs a second beat.
            Row chunks of M_DEPTH = 256. Zero inter-tile bubbles. Latency (last input beat ->
            done) 2N + 4 + 7 + min(M, 256), + N/2 for w_trans (transposer).
  GEMM_GS   as GEMM, K-tiles split at group boundaries, plus one scale beat on every tile:
            max(M, k_tile + 1) (array_seq; ADR-004's "first tile of each group" is superseded).
  VEC engine one command at a time (no overlap; the drain is in the fixed cost). Each of the four
            16-byte line ports (A, B, P, D) moves one line per cycle, so a stream of n elements of
            eb bytes takes max(ceil(n/L), ceil(n eb/16)) cycles ("stream"):
            VEC       F + sum over 4096-element chunks of (1 + stream(len, max eb))
            GELU      F + 1 + stream(len, 1) (+17 when the table is not resident)
            KV_APPEND F + rows x heads x (1 + stream(dh, 1))
            REQUANT   per column chunk of <= 256: N records (one P line each; 1 if per-tensor)
                      + F_chunk + M x (1 + stream(N, 4))   [int32 ACC rows: 4 elements / cycle]
            LN, SOFTMAX_ONLINE  F + rows x (1 + p1 + nb + p3 + f) with nb = ceil(cols/L),
                      p1 = max(nb, input lines - hidden), p3 = max(nb, output lines - hidden)
                      (int16/int8 LN rows, int32 scores), + block lines + 1 when not resident
            LUT_LOAD  40 cycles (32 PBUF lines + loader), 6 when the table is resident
            Latency (occupancy -> done) 2.
Residency: the VEC engine keeps an LN block / GELU table / score record / LUT resident per PBUF address
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


# VEC engine fixed costs fitted to bench/results/vec_seq_perf.json (rtl/vec_seq.sv measured at
# L = 4, 16 on Icarus and 64 on Verilator, ports always ready, read latency 2; docs/vector-unit.md
# §9.3). Variable parts are not fitted: they are the line-port and lane counts. Keys:
#   vec / gelu / kv   per-command fixed cycles (decode, row setup, pipeline drain)
#   rq_chunk, rq_row  REQUANT fixed cycles per column chunk (<= RQ_MAX) and per row of a chunk
#   ln, sm            (H, Hw, f, F): input lines hidden behind the previous row beyond one beat,
#                     output lines hidden, fixed cycles per row, fixed cycles per command
VEC_CAL = {
    4: dict(vec=13, gelu=21, kv=12, rq_chunk=25, rq_row=1, ln=(0, 0, 18, 17), sm=(0, 0, 18, 17)),
    16: dict(vec=14, gelu=23, kv=13, rq_chunk=26, rq_row=1, ln=(0, 0, 18, 21), sm=(2, 0, 18, 23)),
    64: dict(vec=17, gelu=26, kv=16, rq_chunk=29, rq_row=1, ln=(6, 0, 15, 36), sm=(4, 10, 17, 46)),
}
RQ_MAX = 256       # vec_seq REQUANT column chunk (record buffer)
VEC_CH = 4096      # vec_seq VEC / GELU row chunk (elements)


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
    dma_masters: int = 2           # DMA engines / AXI masters (SG10a): a command runs on engine chan & 1
    dma_pipelined: bool = True     # next command's AR overlaps this command's data (SG10a; False = pre-SG10a)
    signal_blocks: bool = False    # True (pre-SG10a): SIGNAL at a queue head blocks the queue until it
                                   # fires; False: the signal is carried on the commands' done
    dma_row_bubble: int = 1        # realigner bubble per row
    dma_cmd_gap: int = 4           # pipelined: cycles between two commands' data on one engine (measured)
    dma_coalesce: bool = False     # what-if: contiguous rows (ds = ss = row_bytes) merged into one row
    dma_max_bursts: int = 4        # outstanding bursts
    dma_max_beats: int = 256       # beats per burst
    # array (docs/array.md §2 and §7, C3, array_seq)
    array_rows_per_cycle: int = 1  # what-if: weight rows per cycle into the array (ADR-003 proposed 2)
    array_lat_extra: int = 4       # first output = 2N + 4
    array_wb_extra: int = 7        # array_seq accept->done = occ + 2N+4 + 7 + M (+ N/2 for w_trans)
    array_w_lines: int = 0         # W-port window in 16-byte lines; 0 = ceil(N/16) (array_seq WPL default).
                                   # A ragged weight row that straddles the window costs a second beat
    gs_beat_every_tile: bool = True  # array_seq: every GEMM_GS tile carries its group's scale beat
    m_depth: int = 256             # array_ctrl accumulator rows (row chunk)
    # VEC engine (rtl/vec_seq.sv, docs/vector-unit.md §9): line ports + per-L fitted fixed costs
    vec_port_lines: int = 1        # 16-byte lines per cycle per VEC port (A, B, P, D); what-if > 1 = wider/banked
    vec_latency: int = 2           # accept->done minus occupancy (commands run one at a time)
    vec_cal: dict = field(default_factory=lambda: dict(VEC_CAL))
    sfu_row_buffers: int = 1       # what-if 2: ping-pong row buffer overlaps pass 1 of row n+1
    lut_load_cycles: int = 40      # LUT_LOAD miss (32 PBUF lines + loader); measured
    lut_hit_cycles: int = 6        # LUT_LOAD of a table already resident at that address
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

def _dma_cost(f, hw, tm, share=1):
    """``share``: DMA engines moving data at the same time (they split the DDR bandwidth)."""
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
    port = min(BB, tm.axi_port_bytes)              # one engine's master
    bpc = min(port, hw.ddr_bytes_per_cycle / share)
    data = max(math.ceil(total / bpc), beats, (_cd(bursts, tm.dma_max_bursts) - 1) * lat)
    data += rows * tm.dma_row_bubble
    if tm.dma_pipelined:
        return data + tm.dma_cmd_gap, lat, {"ddr_bytes": total}
    return lat + data, 0, {"ddr_bytes": total}


def _seg_beats(start, nbytes, window):
    """array_seq_rd: a byte segment takes one beat when it fits the window of ``window`` bytes
    from its start line, else two."""
    return 1 if start % 16 + nbytes <= window else 2


def _w_row_beats(f, op, k0, kt, n0, nt, window):
    """W-port beats of one tile (array_seq addressing, docs/array.md §7)."""
    w, N, K = f["w_addr"], f["N"], f["K"]
    if op == "GEMM" and f["w_trans"]:
        rb, step, base = kt, K, w + n0 * K + k0             # nt columns of kt bytes
        cnt = nt
    elif op == "GEMM_GS" or f["int4"]:
        rb, step, base = _cd(nt, 2), N // 2, w + k0 * (N // 2) + n0 // 2
        cnt = kt
    else:
        rb, step, base = nt, N, w + k0 * N + n0
        cnt = kt
    if step % 16 == 0:
        return cnt * _seg_beats(base, rb, window)
    return sum(_seg_beats(base + r * step, rb, window) for r in range(cnt))


def _gemm_occ(c, hw, tm, kt_max, gs):
    """Sum over C3 tiles of max(mc, W-port beats of the tile / rows per cycle) (+1 scale beat per
    GEMM_GS tile), row chunks of m_depth."""
    f, op, na = c.f, c.op, hw.array_n
    M, N, K = f["M"], f["N"], f["K"]
    window = 16 * (tm.array_w_lines or _cd(na, 16))
    G = isa.gs_group(f["glog2"]) if gs else None
    cyc = 0
    for n0 in range(0, N, na):
        nt = min(na, N - n0)
        k0 = 0
        while k0 < K:
            kt = min(kt_max, K - k0)
            w = _w_row_beats(f, op, k0, kt, n0, nt, window)
            if op == "GEMM" and f["w_trans"]:
                w = max(w, kt)                               # transposer emits kt K-rows
            w = _cd(w, tm.array_rows_per_cycle)
            if gs and (tm.gs_beat_every_tile or k0 % G == 0):
                w += 1
            for m0 in range(0, M, tm.m_depth):
                cyc += max(min(tm.m_depth, M - m0), w)
            k0 += kt
    return cyc


def vec_cal(hw: HwConfig, tm: RtlTiming) -> dict:
    """Fitted VEC-engine constants for hw.lanes (nearest measured lane count in log2)."""
    L = hw.lanes
    k = min(tm.vec_cal, key=lambda x: (abs(math.log2(x) - math.log2(L)), x))
    return tm.vec_cal[k]


def _vec_cost(c, hw, tm, resident):
    """VEC-engine occupancy (rtl/vec_seq.sv, docs/vector-unit.md §9.3): commands run one at a time;
    each port moves vec_port_lines 16-byte lines per cycle, so a stream of n elements of eb bytes
    takes max(ceil(n/L), ceil(n eb / (16 w))) cycles; plus fitted fixed costs (VEC_CAL)."""
    f, op, L = c.f, c.op, hw.lanes
    cal = vec_cal(hw, tm)
    pb = 16 * tm.vec_port_lines

    def stream(n, eb):
        return max(_cd(n, L), _cd(n * eb, pb))

    if op == "LUT_LOAD":
        return (tm.lut_hit_cycles if resident else tm.lut_load_cycles), tm.vec_latency
    if op == "VEC":
        eb = max(isa.DT_BYTES[f["dt_in"]], isa.DT_BYTES[f["dt_out"]])
        n, occ = f["len"], cal["vec"]
        for c0 in range(0, max(n, 1), VEC_CH):
            occ += 1 + stream(min(VEC_CH, n - c0), eb)
        return occ, tm.vec_latency
    if op == "GELU":
        n, occ = f["len"], cal["gelu"]
        for c0 in range(0, max(n, 1), VEC_CH):
            occ += 1 + stream(min(VEC_CH, n - c0), 1)
        if not resident:
            occ += 256 // pb + 1                              # GELU table: 16 PBUF lines
        return occ, tm.vec_latency
    if op == "KV_APPEND":
        return cal["kv"] + f["rows"] * f["heads"] * (1 + stream(f["dh"], 1)), tm.vec_latency
    if op == "REQUANT":
        M, N = f["M"], f["N"]
        occ = 0
        for n0 in range(0, N, RQ_MAX):
            nc = min(RQ_MAX, N - n0)
            rec = 1 if f["per_tensor"] else _cd(nc * 16, pb)   # one 16-byte record per channel (P port)
            occ += rec + cal["rq_chunk"] + M * (cal["rq_row"] + stream(nc, 4))
        return occ, tm.vec_latency
    if op in ("LN", "SOFTMAX_ONLINE"):
        rows, cols = f["rows"], f["cols"]
        eb = (2 if f["in16"] else 1) if op == "LN" else 4
        H, Hw, fx, F = cal["ln" if op == "LN" else "sm"]
        nb = _cd(cols, L)
        il, ol = _cd(cols * eb, pb), _cd(cols, pb)
        hide = (_cd(L * eb, 16) + H) / tm.vec_port_lines
        p1 = max(nb, il - hide)                                 # pass 1: stream in (A port)
        p3 = max(nb, ol - Hw / tm.vec_port_lines)               # pass 3: out (D port)
        if tm.sfu_row_buffers >= 2:
            # what-if (not measured): pass 1 of row n+1 overlaps passes 2-3 of row n
            occ = F + p1 + rows * (1 + max(p1, nb + p3 + fx))
        else:
            occ = F + rows * (1 + p1 + nb + p3 + fx)
        if not resident:
            occ += (ln_block_layout(cols)[2] // 16 if op == "LN" else 1) + 1
        return math.ceil(occ), tm.vec_latency
    raise ModelError(op)


def cost(c: isa.Cmd, hw: HwConfig, tm: RtlTiming | None = None, resident: bool = False, share: int = 1):
    """(occupancy, latency, info) for one command; info carries ddr_bytes / macs. ``resident`` says the
    command's SFU parameter block (LN block, GELU table, score record, LUT) is already loaded;
    ``share`` how many DMA engines move data at once (DDR bandwidth split)."""
    tm = tm or timing_for(hw)
    f, op, na = c.f, c.op, hw.array_n
    if op in ("DMA_LOAD", "DMA_STORE"):
        return _dma_cost(f, hw, tm, share)
    if op in ("GEMM", "GEMM_GS"):
        M, N, K = f["M"], f["N"], f["K"]
        gs = op == "GEMM_GS"
        kt = min(na, isa.gs_group(f["glog2"])) if gs else na    # GEMM_GS tiles never straddle a group
        occ = _gemm_occ(c, hw, tm, kt, gs)
        lat = 2 * na + tm.array_lat_extra + tm.array_wb_extra + min(M, tm.m_depth)
        if op == "GEMM" and f["w_trans"]:
            lat += na // 2
        return occ, lat, {"macs": M * N * K}
    if op == "ATTN_TILE":
        rows, ln = f["rows"], f["len"]
        occ = 2 * _cd(ln, na) * 2 * max(rows, na)
        return occ, 2 * na + tm.array_lat_extra, {"macs": 2 * rows * ln * 64}
    if op in ("WAIT", "SIGNAL", "IRQ", "END"):
        return 0, 0, {}
    occ, lat = _vec_cost(c, hw, tm, resident)
    return occ, lat, {}


def _param_key(c):
    f = c.f
    if c.op == "LN":
        return f["param_addr"], f["param_addr"] + ln_block_layout(f["cols"])[2]
    if c.op == "GELU":
        return f["lut_addr"], f["lut_addr"] + 256
    if c.op == "SOFTMAX_ONLINE":
        return f["ctx_addr"], f["ctx_addr"] + 16
    if c.op == "LUT_LOAD":
        return f["src_addr"], f["src_addr"] + 512
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
    busy_until = [E0] * nq          # (time, event) engine free for the next command (DMA: queue issue frontier)
    nch = 2 if tm.dma_masters > 1 else 1
    chan_busy = [E0] * nch          # DMA engines: (time, event) data stage free
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
        dma = c.op in ("DMA_LOAD", "DMA_STORE")
        ch = (c["chan"] & 1) if dma and nch == 2 else 0
        if not dma:
            occ, l, info = cost(c, hw, tm, resident=res)
        if c.op == "DMA_LOAD" and c["sp_buf"] == isa.PBUF and resident:
            lo, hi = c["sp_addr"], c["sp_addr"] + c["rows"] * max(c["sp_stride"] * isa.LINE, c["row_bytes"])
            for k2 in [k2 for k2 in resident if k2[0] < hi and lo < k2[1]]:
                del resident[k2]
        cand = [(dt, de, "dispatch"), busy_until[q]]
        if dma:
            cand.append(chan_busy[ch])
            if c.op == "DMA_STORE":         # fence: both channels done
                cand += [done_until[q]]
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
        if dma:
            share = 2 if nch == 2 and chan_busy[1 - ch][0] > st else 1
            occ, l, info = cost(c, hw, tm, share=share)
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
        ct, ce = ev.add([(et + l, ee, cls)], trace)
        if dma:
            chan_busy[ch] = (et, ee)
            busy_until[q] = (st, se)        # the queue issues in order; the engines overlap
            if c.op == "DMA_STORE":         # ...and no DMA command starts until the fence is done
                chan_busy = [(ct, ce)] * nch
            elif not tm.dma_pipelined:      # pre-SG10a: one command at a time per engine
                chan_busy[ch] = (ct, ce) if nch == 2 else (et, ee)
                if nch == 1:
                    busy_until[q] = (et, ee)
        else:
            busy_until[q] = (et, ee)
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


# the control plane before SG10a (one DMA engine, not pipelined, blocking SIGNAL): ADR-005 before/after
PRE_SG10A = dict(dma_masters=1, dma_pipelined=False, signal_blocks=True)

# what-if fixes priced by bench/roofline.py (ADR-003 "bottlenecks"). "DMA command pipelining" and
# "second DMA master (2 HP ports)" are built (SG10a, ADR-005) and are part of every row now.
FIXES = {
    "coalesce contiguous DMA rows": dict(dma_coalesce=True),
    "on-chip / prefetched command ring": dict(fetch_over_axi=False),
    "4-line (64 B) VEC engine ports": dict(vec_port_lines=4),
    "SFU second row buffer": dict(sfu_row_buffers=2),
    "resident SFU parameter slots": dict(params_resident=True),
    "2-row array weight port": dict(array_rows_per_cycle=2),
}
