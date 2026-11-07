"""Cycle model, measured-RTL costs (docs/control.md, docs/array.md, docs/sfu.md): per-command costs,
queue / semaphore / fetch semantics, critical-path attribution and model-level sanity bounds."""
import pytest

from nanoxl import isa
from nanoxl.cyclemodel import (FIXES, MEASURED, VEC_CAL, ModelError, RtlTiming, cost, estimate, simulate,
                               timing_for)
from nanoxl.graph import SHAKESPEARE, TINY
from nanoxl.hwconfig import KV260, PYNQ_Z2

C = isa.C
LAT = 75  # KV260: 300 ns at 250 MHz


def test_cyclemodel_dma_cost_measured():
    hw = KV260  # 8 GB/s = 32 B/cycle DDR, but one 128-bit master = 16 B/cycle
    occ, lat, info = cost(C("DMA_LOAD", rows=1, row_bytes=32768), hw)
    assert (occ, lat) == (LAT + 32768 // 16 + 1, 0) and info["ddr_bytes"] == 32768   # not pipelined
    # strided small rows: one burst per row, 4 outstanding -> 4 rows per DDR latency
    occ2, _, _ = cost(C("DMA_LOAD", rows=1024, row_bytes=8, ddr_stride=64), hw)
    assert occ2 == LAT + (1024 // 4 - 1) * LAT + 1024
    # contiguous 64-byte rows (the compiler's int8 weight panels) are burst bound too...
    panel = C("DMA_LOAD", rows=384, row_bytes=64, ddr_stride=64)
    assert cost(panel, hw)[0] == LAT + (384 // 4 - 1) * LAT + 384
    # ...unless coalesced into one row (what-if): 1536 beats + one bubble
    assert cost(panel, hw, MEASURED.with_(dma_coalesce=True))[0] == LAT + 1536 + 1
    # unaligned DDR start touches one more beat per row
    assert cost(C("DMA_LOAD", rows=1, row_bytes=32, ddr_addr=8), hw)[0] == LAT + 3 + 1
    # pipelined (what-if): latency leaves the occupancy
    occ3, lat3, _ = cost(C("DMA_STORE", rows=1, row_bytes=32768), hw, MEASURED.with_(dma_pipelined=True))
    assert (occ3, lat3) == (2048 + 1, LAT)
    # two masters: DDR (32 B/cycle) becomes the limit
    assert cost(C("DMA_LOAD", rows=1, row_bytes=32768), hw, MEASURED.with_(dma_masters=2))[0] == LAT + 1024 + 1
    # PYNQ-Z2: 64-bit HP port, 150 MHz, 400 ns: 8 B/cycle, 60 cycles latency
    occ4, _, _ = cost(C("DMA_LOAD", rows=1, row_bytes=4096), PYNQ_Z2)
    assert occ4 == 60 + 4096 // 8 + 1


def test_cyclemodel_gemm_cost_measured():
    hw = KV260  # N = 32: tile max(M, k); last input -> done 2N + 4 + 7 + M (array.md §7)
    occ, lat, info = cost(C("GEMM", M=256, N=64, K=96), hw)
    assert occ == 2 * 3 * 256 and lat == 68 + 7 + 256 and info["macs"] == 256 * 64 * 96
    assert cost(C("GEMM", M=300, N=32, K=32), hw)[0] == 256 + 44                 # M_DEPTH row chunks
    assert cost(C("GEMM", M=1, N=32, K=32, w_trans=1), hw)[1] == 68 + 7 + 1 + 16  # transposer
    assert cost(C("GEMM", M=1, N=64, K=96, gemv=1), hw)[0] == 2 * 96          # GEMV: K rows per tile
    assert cost(C("GEMM", M=1, N=64, K=96, gemv=1, int4=1), hw)[0] == 2 * 96  # INT4: still 1 K-row / beat
    assert cost(C("GEMM", M=1, N=32, K=40, gemv=1), hw)[0] == 32 + 8          # ragged last K tile
    assert cost(C("GEMM", M=10, N=32, K=40), hw)[0] == 32 + 10
    assert cost(C("GEMM", M=1, N=32, K=64), hw, MEASURED.with_(array_rows_per_cycle=2))[0] == 32


def test_cyclemodel_gemm_gs_cost():
    """array_seq: every GEMM_GS tile carries its group's scale beat (AUDIT-1 A1-05; ADR-004 charged
    only the first tile of a group); hidden when M >= k + 1."""
    hw = KV260
    occ_gs, lat, info = cost(C("GEMM_GS", M=1, N=64, K=128, glog2=3, gemv=1), hw)  # G = 64
    assert occ_gs == 2 * 4 * 33 and lat == 68 + 7 + 1 and info["macs"] == 64 * 128
    first_only = cost(C("GEMM_GS", M=1, N=64, K=128, glog2=3), hw, MEASURED.with_(gs_beat_every_tile=False))[0]
    assert first_only == 2 * (2 * 33 + 2 * 32)
    big_gs = cost(C("GEMM_GS", M=256, N=64, K=128, glog2=3), hw)[0]
    assert big_gs == cost(C("GEMM", M=256, N=64, K=128, int4=1), hw)[0]
    assert cost(C("GEMM_GS", M=1, N=32, K=128, glog2=1, gemv=1), hw)[0] == 8 * 17   # G = 16 < N


def test_cyclemodel_ragged_weight_rows():
    """A weight row that straddles the ceil(N/16)-line W window costs a second beat (LM head N=65:
    624 measured at N=32, docs/array.md §7); a WPL+1 window (what-if) removes it."""
    hw = KV260
    lm = C("GEMM", M=16, N=65, K=128)
    assert cost(lm, hw)[0] == 624
    assert cost(lm, hw, MEASURED.with_(array_w_lines=3))[0] == 4 * 3 * 32       # weight-row bound: kt = 32 rows per tile


def test_cyclemodel_sfu_costs_measured():
    """VEC-engine costs = port/lane stream + fitted fixed costs (VEC_CAL); the full comparison with
    every measured shape is tests/audit (A1-01) against bench/results/vec_seq_perf.json."""
    hw = KV260  # 64 lanes, 16-byte ports
    cal = VEC_CAL[64]
    H, Hw, f, F = cal["ln"]
    nb, il = 6, 48                                   # LN 384 int16: 6 beats, 48 input lines
    p1 = il - (8 + H)                              # one beat (8 lines) + H hidden behind the previous row
    p3 = max(nb, 24 - Hw)
    blk = (2 + 384 // 8 + 384 // 4) + 1
    assert cost(C("LN", rows=4, cols=384, in16=1), hw, resident=True)[0] == F + 4 * (1 + p1 + nb + p3 + f)
    assert cost(C("LN", rows=4, cols=384, in16=1), hw)[0] == F + 4 * (1 + p1 + nb + p3 + f) + blk
    # REQUANT at L = 64 is port bound: 4 int32 per cycle and one record line per channel
    rq = C("REQUANT", M=16, N=384)
    assert cost(rq, hw)[0] == 384 + 2 * cal["rq_chunk"] + 16 * (1 + 64) + 16 * (1 + 32)
    assert cost(C("REQUANT", M=16, N=384, per_tensor=1), hw)[0] == 2 + 2 * cal["rq_chunk"] + 16 * (65 + 33)
    assert cost(C("LUT_LOAD", table=1), hw)[0] == 40
    assert cost(C("LUT_LOAD", table=1), hw, resident=True)[0] == 6
    assert cost(C("GELU", len=640), hw)[0] == cal["gelu"] + 1 + 40 + 17
    assert cost(C("VEC", len=1000, dt_in=isa.DT_I32, dt_out=isa.DT_I32), hw)[0] == cal["vec"] + 1 + 250
    # L = 4: lane bound everywhere; LN d=384 is 3 nb + 19 cycles per row (+ accept), as sfu.md §5
    hw4 = KV260.with_(lanes=4)
    H4, _, f4, F4 = VEC_CAL[4]["ln"]
    assert cost(C("LN", rows=2, cols=384, in16=1), hw4, resident=True)[0] == F4 + 2 * (1 + 3 * 96 + f4)
    # wider ports (what-if) remove the port bound at L = 64
    w4 = MEASURED.with_(vec_port_lines=4)
    assert cost(rq, hw, w4)[0] < 0.4 * cost(rq, hw)[0]
    # ping-pong row buffer (what-if) overlaps pass 1 of the next row
    pp = cost(C("LN", rows=4, cols=384, in16=1), hw, MEASURED.with_(sfu_row_buffers=2), resident=True)[0]
    assert pp < cost(C("LN", rows=4, cols=384, in16=1), hw, resident=True)[0]


def test_cyclemodel_ln_block_residency():
    """An LN block stays resident per PBUF address until a DMA writes that range."""
    hw = KV260
    ln = C("LN", rows=1, cols=384, param_addr=0)
    blk = 2 + 48 + 96 + 1
    one = simulate([ln, C("END")], hw).busy["vec"]
    two = simulate([ln, ln, C("END")], hw).busy["vec"]
    assert two == 2 * one - blk
    reload = [ln, C("DMA_LOAD", sp_buf=isa.PBUF, sp_addr=0, rows=1, row_bytes=64),
              C("SIGNAL", queue=0, sem=1), C("WAIT", queue=2, sem=1, count=1), ln, C("END")]
    assert simulate(reload, hw).busy["vec"] == 2 * one


def test_cyclemodel_queues_overlap_and_semaphores():
    hw = KV260
    d = C("DMA_LOAD", rows=1, row_bytes=16 * 1000)          # 75 + 1000 + 1
    g = C("GEMM", M=500, N=32, K=32)                        # 500 occ, 68 lat
    t = simulate([d, g, C("END")], hw)
    assert t.cycles < 1076 + 568
    dep = [d, C("SIGNAL", queue=0, sem=0), C("WAIT", queue=1, sem=0, count=1), g, C("END")]
    t2 = simulate(dep, hw, trace=True)
    start_g = [s for i, q, s, e in t2.trace if i == 3][0]
    assert start_g >= 1076
    assert t2.cycles >= 1076 + 568


def test_cyclemodel_signal_blocks_queue_until_completion():
    """SIGNAL at a queue head blocks that queue until every earlier command completed (incl. the
    GEMM's 2N+4 latency), so a GEMM after it starts late; the what-if non-blocking SIGNAL does not."""
    hw = KV260
    g = C("GEMM", M=100, N=32, K=32)
    cmds = [g, C("SIGNAL", queue=1, sem=3), g, C("WAIT", queue=2, sem=3, count=1), C("END")]
    t = simulate(cmds, hw, trace=True)
    s = {i: st for i, q, st, e in t.trace}
    assert s[2] - s[0] >= 100 + 68
    t2 = simulate(cmds, hw, trace=True, tm=MEASURED.with_(signal_blocks=False))
    s2 = {i: st for i, q, st, e in t2.trace}
    assert s2[2] - s2[0] == 100


def test_cyclemodel_rejects_wait_before_signal():
    with pytest.raises(ModelError):
        simulate([C("WAIT", queue=1, sem=0, count=1), C("SIGNAL", queue=0, sem=0), C("END")], KV260)


def test_cyclemodel_command_fetch_over_axi():
    """Fetch: 8-word bursts, one in flight: n cheap words cost ~ n/8 x (latency + 8) cycles."""
    hw = KV260
    cmds = [C("GEMM", M=1, N=1, K=1)] * 64 + [C("END")]
    t = simulate(cmds, hw, trace=True)
    bursts = -(-65 // 8)
    assert t.cycles >= (bursts - 1) * (LAT + 8) + LAT
    assert t.critical.get("cmd fetch", 0) > 0.8 * t.cycles
    fast = simulate(cmds, hw, tm=MEASURED.with_(fetch_over_axi=False))
    assert fast.cycles < 200


def test_cyclemodel_queue_depth_stalls_dispatch():
    hw = KV260
    longd = C("DMA_LOAD", rows=1, row_bytes=16 * 5000)
    cmds = [longd] * 10 + [C("GEMM", M=1, N=32, K=32), C("END")]
    t = simulate(cmds, hw, trace=True, tm=MEASURED.with_(fetch_over_axi=False))
    gs = [s for i, q, s, e in t.trace if i == 10][0]
    assert gs > 5000            # 1 issued + 8 queued: the 10th DMA blocks dispatch
    deep = simulate(cmds, hw, trace=True, tm=MEASURED.with_(fetch_over_axi=False, qdepth=16))
    assert [s for i, q, s, e in deep.trace if i == 10][0] < 20


def test_cyclemodel_critical_path_sums_to_total():
    from nanoxl.compiler import Compiler
    from nanoxl.graph import build_gpt_graph, make_synthetic_params
    g = build_gpt_graph(TINY)
    comp = Compiler(g, make_synthetic_params(g, 0), KV260)
    for seqs in ([(40, 1)], [(0, 64)]):
        t = simulate(comp.compile_step(seqs).cmds, KV260, trace=True)
        assert sum(t.critical.values()) == t.cycles
        assert sum(t.busy_by_class.values()) == sum(t.busy.values())


@pytest.fixture(scope="module")
def shk():
    return estimate(SHAKESPEARE, KV260, n_pos=3, breakdown=True)


def test_cyclemodel_shakespeare_measured(shk):
    """As built (wave 2 RTL + coalesced compiler DMAs), KV260 INT8 decode is bound by the single
    128-bit DMA master (16 B/cycle = 4 GB/s at 250 MHz), half the 8 GB/s board assumption: the
    critical path is almost all weight DMA. Prefill is mostly array bound; with the VEC engine priced at
    its 16-byte line ports (AUDIT-1 A1-01) the array is ~65 % busy, still above G3's 50 %."""
    assert shk["fits"]
    port_bound = 16 * KV260.clock_hz / SHAKESPEARE.weight_bytes
    assert 0.85 * port_bound < shk["decode_tok_s"] <= port_bound
    assert shk["decode_tok_s"] < 0.55 * shk["weight_bound_tok_s"]
    assert shk["decode_critical"]["dma weights"] > 0.85 * shk["decode_cycles"]
    assert shk["prefill_critical"]["gemm linear"] > 0.6 * shk["prefill_cycles"]
    assert 0.5 < shk["prefill_array_util"] < 0.75
    assert shk["prefill_critical"]["softmax"] > 0.08 * shk["prefill_cycles"]   # int32 scores, 4 per cycle


def test_cyclemodel_vec_port_widening_restores_prefill(shk):
    """What-if (ADR-003 AUDIT-1 update): 4-line (64-byte) VEC engine ports bring KV260 prefill back
    near the lane-rate model (~0.8 array utilization); decode does not move."""
    tm = timing_for(KV260).with_(**FIXES["4-line (64 B) VEC engine ports"])
    w = estimate(SHAKESPEARE, KV260, n_pos=3, tm=tm)
    assert w["prefill_array_util"] > 0.75 and w["prefill_ms"] < 0.9 * shk["prefill_ms"]
    assert abs(w["decode_tok_s"] / shk["decode_tok_s"] - 1) < 0.01


def test_cyclemodel_fixes_order(shk):
    """The ADR-003 bottleneck ranking with the coalesced compiler: a second DMA master first (~1.9x
    decode), DMA pipelining (+ SIGNAL on done) a few %, the SFU second row buffer only in prefill, and
    the 2-row array weight port pays off for INT4 once the DMA is doubled."""
    tm = timing_for(KV260)
    est = lambda cfg, **kw: estimate(cfg, KV260, prefill_len=0, n_pos=3, tm=tm.with_(**kw))["decode_tok_s"]
    two = FIXES["second DMA master (2 HP ports)"]
    base = shk["decode_tok_s"]
    assert est(SHAKESPEARE, **two) > 1.6 * base
    assert 1.0 < est(SHAKESPEARE, **FIXES["DMA command pipelining"]) / base < 1.15
    assert abs(est(SHAKESPEARE, **FIXES["SFU second row buffer"]) / base - 1) < 0.01
    # coalescing is already done by the compiler: the what-if changes nothing
    assert est(SHAKESPEARE, **FIXES["coalesce contiguous DMA rows"]) == base
    w4 = SHAKESPEARE.with_(wbits=4)
    both = dict(two, **FIXES["2-row array weight port"])
    assert est(w4, **both) > 1.5 * est(w4, **two)
    sfu = estimate(SHAKESPEARE, KV260, n_pos=3, tm=tm.with_(**FIXES["SFU second row buffer"]))
    assert sfu["prefill_ms"] < shk["prefill_ms"]


def test_cyclemodel_int4_and_pynq():
    int4 = estimate(SHAKESPEARE.with_(wbits=4), KV260, prefill_len=0, n_pos=3)
    # W4A8 panels are dense single-row DMAs; the array weight port (1 K-row per beat) and the
    # 16 B/cycle DMA bound it near 4 GB/s-equivalent
    assert 400 < int4["decode_tok_s"] < int4["weight_bound_tok_s"]
    pz = estimate(SHAKESPEARE, PYNQ_Z2, prefill_len=32, n_pos=3)
    assert pz["decode_tok_s"] < pz["weight_bound_tok_s"]
    t = estimate(TINY, KV260, n_pos=3)
    assert t["decode_tok_s"] > 5 * estimate(SHAKESPEARE, KV260, prefill_len=0, n_pos=3)["decode_tok_s"]


def test_cyclemodel_runs_real_bundle_programs():
    """The cycle model times compiled programs of the real tiny bundles (every opcode they use is costed)."""
    from pathlib import Path
    from nanoxl.compiler import Compiler
    from nanoxl.export import load_bundle
    from nanoxl.graph import from_bundle
    root = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "tiny"
    for name in ("w8a8", "w4a8"):
        g, p = from_bundle(*load_bundle(root / f"{name}.npz"))
        comp = Compiler(g, p, KV260)
        dec = simulate(comp.compile_step([(64, 1)]).cmds, KV260)
        pre = simulate(comp.compile_step([(0, 64)]).cmds, KV260)
        assert 0 < dec.cycles < pre.cycles
        assert dec.ddr_bytes >= g.cfg.weight_bytes


def test_cyclemodel_timing_is_frozen_dataclass():
    assert isinstance(MEASURED, RtlTiming)
    with pytest.raises(Exception):
        MEASURED.qdepth = 4


def _top_measured():
    """docs/top.md §6: measured PERF_CYCLES of nanoxl_top (N=16, L=16, tiny, Verilator) per bundle/KV:
    (prefill 16 rows, decode min, decode max)."""
    import re
    from pathlib import Path
    txt = (Path(__file__).resolve().parents[2] / "docs" / "top.md").read_text()
    sec = txt[txt.index("## 6. Performance"):]
    out = {}
    for m in re.finditer(r"\| (W[48]A8), (on-chip|DDR KV) \| ([\d,]+) / [\d,]+ [^|]*\| ([\d,]+)–([\d,]+) / ", sec):
        n = lambda s: int(s.replace(",", ""))
        out[(m.group(1).lower(), m.group(2) == "on-chip")] = (n(m.group(3)), n(m.group(4)), n(m.group(5)))
    return out


@pytest.mark.parametrize("bundle", ["w8a8", "w4a8"])
@pytest.mark.parametrize("kv_onchip", [True, False], ids=["onchip", "ddrkv"])
def test_cyclemodel_matches_measured_top_level(bundle, kv_onchip):
    """Whole-program check against the RTL top level (docs/top.md §6, SG7c): the same compiled tiny
    programs at N=16, L=16 with the test instance's buffers; the model is within 10 % of the measured
    prefill (16 rows) and decode (pos 16 and 23) cycles. Before A1-01 prefill was 13-20 % low."""
    from pathlib import Path
    from nanoxl.compiler import Compiler
    from nanoxl.export import load_bundle
    from nanoxl.graph import from_bundle
    from nanoxl.hwconfig import KiB
    meas = _top_measured()
    assert len(meas) == 4, meas
    pre_m, dec_lo, dec_hi = meas[(bundle, kv_onchip)]
    hw = KV260.with_(array_n=16, lanes=16, kv_onchip=kv_onchip, abuf=64 * KiB, wbuf=32 * KiB, acc=64 * KiB,
                     pbuf=32 * KiB, kvbuf=128 * KiB)            # tests/rtl/top_cocotb.top_hw
    g, p = from_bundle(*load_bundle(Path(__file__).resolve().parents[2] / "artifacts" / "models" / "tiny"
                                    / f"{bundle}.npz"))
    comp = Compiler(g, p, hw)
    pre = simulate(comp.compile_step([(0, 16)]).cmds, hw).cycles
    assert abs(pre / pre_m - 1) < 0.10, (pre, pre_m)
    for pos in (16, 23):
        dec = simulate(comp.compile_step([(pos, 1)]).cmds, hw).cycles
        assert 0.9 * dec_lo < dec < 1.1 * dec_hi, (pos, dec, dec_lo, dec_hi)
