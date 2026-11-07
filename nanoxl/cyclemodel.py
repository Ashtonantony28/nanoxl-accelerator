"""Cycle-approximate performance model of a NanoXL command stream.

Model
  * A command processor fetches/dispatches one command per ``issue_cycles`` in stream order into
    per-queue FIFOs of depth ``fifo_depth``; dispatch stalls while the target FIFO is full.
  * Three in-order queues (DMA, ARRAY, VEC) each run one command at a time. A command has an
    *occupancy* (cycles the unit is busy, after which the next command of that queue may start)
    and a *latency* (extra cycles until its results are visible: DDR round trip, array drain,
    vector pipeline). SIGNAL fires once every earlier command of its queue has completed.
  * WAIT(sem, k) blocks its queue until k more SIGNALs than already consumed have fired.
  * END/IRQ wait for all queues to drain.
Because every WAIT's SIGNAL precedes it in the stream (compiler invariant, checked here), a single
pass in stream order computes exact start/end times for this model.

Per-command costs (cycles; N = array size, L = vector lanes, B = DDR bytes/cycle):
  DMA      occ = max(bytes / B, bursts) + 4     lat = DDR latency
  GEMM     occ = wl + ceil(N_out/N) ceil(K/N) max(M, wl), wl = N / (wload_rows_per_cycle x (2 if int4));
           the leading wl is hidden when the GEMM starts back to back with the previous GEMM
           (double-buffered PE weight registers preload across commands)
           lat = 2N (N in GEMV mode, which bypasses skew/deskew)
  GEMM_GS  as an int4 GEMM, plus one scale beat (N uint8 = two int4 weight rows of port bandwidth) on
           the first K-tile of every group: that tile costs max(M, wl_s), wl_s = ceil((N + 2) / (2 rpc))
           (ADR-004; the per-column uint8 x int32 multiply sits in the accumulator RMW stage, 0 cycles)
  VEC/GELU occ = ceil(len / L);  REQUANT occ = M ceil(N_out / L)
  LN       C5 3-pass: occ = rows x (3 ceil(cols / L) + sfu_row_bubble) + row_latency_ln
           (pass 1 sum, pass 2 sum of squares around mu, rsqrt, pass 3 normalize + affine)
  SOFTMAX  C5 3-pass: occ = rows x (3 ceil(cols / L) + sfu_row_bubble) + row_latency_softmax
           (pass 1 score requant + max, pass 2 exp2 + sum, recip, pass 3 normalize)
  LUT_LOAD occ = 256 / 16 (16 entries per cycle into the SFU table RAM)
  KV_APPEND occ = rows x heads x ceil(dh / L)
LN / SOFTMAX / LUT_LOAD are estimates (SG5 had not documented SFU timing when this was written):
the per-row bubble covers the reduce-tree latency between passes that row interleaving cannot hide.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from . import isa
from .hwconfig import HwConfig


class ModelError(RuntimeError):
    pass


def _cd(a, b):
    return -(-a // b)


def cost(c: isa.Cmd, hw: HwConfig):
    """(occupancy, latency, info) for one command; info carries ddr_bytes / macs / array_cycles."""
    f, op, na, L = c.f, c.op, hw.array_n, hw.lanes
    if op in ("DMA_LOAD", "DMA_STORE"):
        rows, rb = f["rows"], f["row_bytes"]
        total = rows * rb
        contig = rows == 1 or f["ddr_stride"] in (0, rb)
        bursts = _cd(total, hw.burst_bytes) if contig else rows * _cd(rb, hw.burst_bytes)
        occ = math.ceil(max(total / hw.ddr_bytes_per_cycle, bursts)) + 4
        lat = math.ceil(hw.ddr_latency_ns * hw.clock_mhz / 1e3)
        return occ, lat, {"ddr_bytes": total}
    if op == "GEMM":
        M, N, K = f["M"], f["N"], f["K"]
        wl = _cd(na, hw.wload_rows_per_cycle * (2 if f["int4"] else 1))
        tiles = _cd(N, na) * _cd(K, na)
        occ = wl + tiles * max(M, wl)
        lat = hw.fill_drain or (na if f["gemv"] else 2 * na)
        return occ, lat, {"macs": M * N * K, "preload": wl}
    if op == "GEMM_GS":
        M, N, K = f["M"], f["N"], f["K"]
        rpc = hw.wload_rows_per_cycle
        wl, wl_s = _cd(na, 2 * rpc), _cd(na + 2, 2 * rpc)
        G = isa.gs_group(f["glog2"])
        kt = min(na, G)                       # the controller splits K-tiles at group boundaries
        nct = _cd(N, na)
        ktiles = _cd(K, kt)
        gtiles = _cd(K, G)                    # tiles that carry a scale beat (first of each group)
        occ = wl + nct * (gtiles * max(M, wl_s) + (ktiles - gtiles) * max(M, wl))
        lat = hw.fill_drain or (na if f["gemv"] else 2 * na)
        return occ, lat, {"macs": M * N * K, "preload": wl}
    if op == "LUT_LOAD":
        return isa.LUT_ENTRIES // 16, hw.vec_latency, {}
    if op == "ATTN_TILE":
        rows, ln = f["rows"], f["len"]
        occ = 2 * (_cd(ln, na) * 2 * max(rows, na))
        return occ, 2 * na, {"macs": 2 * rows * ln * 64}
    if op == "VEC":
        return _cd(f["len"], L), hw.vec_latency, {}
    if op == "GELU":
        return _cd(f["len"], L), hw.vec_latency, {}
    if op == "REQUANT":
        return f["M"] * _cd(f["N"], L), hw.vec_latency, {}
    if op == "LN":
        return f["rows"] * (3 * _cd(f["cols"], L) + hw.sfu_row_bubble) + hw.row_latency_ln, hw.vec_latency, {}
    if op == "SOFTMAX_ONLINE":
        return (f["rows"] * (3 * _cd(f["cols"], L) + hw.sfu_row_bubble) + hw.row_latency_softmax,
                hw.vec_latency, {})
    if op == "KV_APPEND":
        return f["rows"] * f["heads"] * _cd(f["dh"], L), hw.vec_latency, {}
    if op in ("WAIT", "SIGNAL", "IRQ", "END"):
        return 1, 0, {}
    raise ModelError(op)


@dataclass
class Timeline:
    cycles: int
    hw: HwConfig
    busy: dict                 # queue name -> busy cycles
    ddr_bytes: int
    macs: int
    n_cmds: int
    max_outstanding_sem: int
    trace: list = field(default_factory=list)  # (cmd index, queue, start, end) if requested

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


def simulate(cmds, hw: HwConfig, trace=False) -> Timeline:
    nq = 3
    busy_until = [0] * nq      # unit free for the next command
    done_until = [0] * nq      # all issued commands of the queue have completed
    fifo = [deque() for _ in range(nq)]   # start times of the last fifo_depth commands per queue
    busy = [0] * nq
    sig_times, consumed = {}, {}
    disp = 0
    ddr = macs = 0
    max_out = 0
    tr = []
    last_gemm_end = -10
    for i, c in enumerate(cmds):
        q = c.queue
        occ, lat, info = cost(c, hw)
        disp += hw.issue_cycles
        if q == isa.Q_CTRL:
            start = max([disp] + done_until + busy_until)
            disp = start + occ
            for j in range(nq):
                busy_until[j] = done_until[j] = max(busy_until[j], disp)
            if trace:
                tr.append((i, q, start, disp))
            continue
        if len(fifo[q]) >= hw.fifo_depth:
            disp = max(disp, fifo[q][0])  # wait until the oldest queued command has started
        start = max(disp, busy_until[q])
        if c.op == "WAIT":
            s, k = c["sem"], c["count"]
            need = consumed.get(s, 0) + k
            got = sig_times.get(s, [])
            if len(got) < need:
                raise ModelError(f"cmd {i}: WAIT sem {s} needs signal #{need}, only {len(got)} precede it")
            start = max(start, got[need - 1])
            consumed[s] = need
        if "preload" in info and start <= last_gemm_end + 2:
            occ -= info["preload"]  # first weight tile was preloaded under the previous GEMM
        end = start + occ
        if c.op == "GEMM":
            last_gemm_end = end
        if c.op == "SIGNAL":
            # fires when every earlier command of the queue has completed; the queue itself only
            # spends one issue slot on it and moves on
            fire = max(end, done_until[q])
            st = sig_times.setdefault(c["sem"], [])
            if st and fire < st[-1]:
                raise ModelError("semaphore with more than one producer queue")
            st.append(fire)
            max_out = max(max_out, len(st) - consumed.get(c["sem"], 0))
        busy_until[q] = end
        done_until[q] = max(done_until[q], end + lat)
        busy[q] += occ
        fifo[q].append(start)
        if len(fifo[q]) > hw.fifo_depth:
            fifo[q].popleft()
        ddr += info.get("ddr_bytes", 0)
        macs += info.get("macs", 0)
        if trace:
            tr.append((i, q, start, end + lat))
    total = max(done_until + busy_until + [disp])
    names = {isa.Q_DMA: "dma", isa.Q_ARR: "array", isa.Q_VEC: "vec"}
    return Timeline(total, hw, {names[q]: busy[q] for q in range(nq)}, ddr, macs, len(cmds), max_out, tr)


# ---------------------------------------------------------------- model-level estimates

def decode_positions(ctx, n=5):
    return sorted({min(ctx - 1, int(round(x))) for x in [i * (ctx - 1) / (n - 1) for i in range(n)]})


def estimate(cfg, hw: HwConfig, batch=1, prefill_len=None, params=None, n_pos=5):
    """Prefill latency (slot 0, T = prefill_len or ctx) and decode tokens/s at ``batch`` sequences,
    averaged over decode positions spread across the context. Returns a dict; on-chip KV that does
    not fit returns {'fits': False, ...}."""
    from .compiler import Compiler, CompileError
    from .graph import build_gpt_graph, make_synthetic_params
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
        tls = [simulate(p.cmds, hw) for p in progs]
        cyc = sum(t.cycles for t in tls)
        res["prefill_T"] = T
        res["prefill_chunk"] = progs[0].seqs[0][1]
        res["prefill_ms"] = cyc / hw.clock_hz * 1e3
        res["prefill_array_util"] = sum(t.macs for t in tls) / (hw.array_n ** 2 * cyc)
        res["prefill_gops"] = 2 * sum(t.macs for t in tls) / (cyc / hw.clock_hz) / 1e9
    times, ddr, util, macs = [], [], [], []
    for p in decode_positions(cfg.ctx, n_pos):
        prog = comp.compile_step([(p, 1)] * batch)
        t = simulate(prog.cmds, hw)
        times.append(t.seconds)
        ddr.append(t.ddr_bytes)
        util.append(t.ddr_util)
        macs.append(t.macs)
    step = sum(times) / len(times)
    res["decode_step_us"] = step * 1e6
    res["decode_tok_s"] = batch / step
    res["decode_ddr_MB_per_step"] = sum(ddr) / len(ddr) / 1e6
    res["decode_ddr_util"] = sum(util) / len(util)
    res["decode_ops_per_byte"] = 2 * sum(macs) / max(sum(ddr), 1)
    res["decode_gops"] = 2 * sum(macs) / len(macs) / step / 1e9
    # simple bound: stream all GEMM weights once per step at the configured bandwidth
    res["weight_bound_tok_s"] = batch * hw.ddr_gbps * 1e9 / cfg.weight_bytes
    return res
