"""Cycle model, measured-RTL costs (docs/control.md, docs/array.md, docs/sfu.md): per-command costs,
queue / semaphore / fetch semantics, critical-path attribution and model-level sanity bounds."""
import pytest

from nanoxl import isa
from nanoxl.cyclemodel import (FIXES, MEASURED, ModelError, RtlTiming, cost, estimate, simulate,
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
    hw = KV260  # N = 32: tile max(M, k), first output 2N + 4
    occ, lat, info = cost(C("GEMM", M=256, N=64, K=96), hw)
    assert occ == 2 * 3 * 256 and lat == 68 and info["macs"] == 256 * 64 * 96
    assert cost(C("GEMM", M=1, N=64, K=96, gemv=1), hw)[0] == 2 * 96          # GEMV: K rows per tile
    assert cost(C("GEMM", M=1, N=64, K=96, gemv=1, int4=1), hw)[0] == 2 * 96  # INT4: still 1 K-row / beat
    assert cost(C("GEMM", M=1, N=32, K=40, gemv=1), hw)[0] == 32 + 8          # ragged last K tile
    assert cost(C("GEMM", M=10, N=32, K=40), hw)[0] == 32 + 10
    assert cost(C("GEMM", M=1, N=32, K=64), hw, MEASURED.with_(array_rows_per_cycle=2))[0] == 32


def test_cyclemodel_gemm_gs_cost():
    """ADR-004: one extra scale beat on the first K-tile of each group; hidden when M >= k + 1."""
    hw = KV260
    occ_gs, lat, info = cost(C("GEMM_GS", M=1, N=64, K=128, glog2=3, gemv=1), hw)  # G = 64
    assert occ_gs == 2 * (2 * 33 + 2 * 32) and lat == 68 and info["macs"] == 64 * 128
    big_gs = cost(C("GEMM_GS", M=256, N=64, K=128, glog2=3), hw)[0]
    assert big_gs == cost(C("GEMM", M=256, N=64, K=128, int4=1), hw)[0]
    assert cost(C("GEMM_GS", M=1, N=32, K=128, glog2=1, gemv=1), hw)[0] == 8 * 17   # G = 16 < N


def test_cyclemodel_sfu_costs_measured():
    hw = KV260  # 64 lanes
    nb = 6      # 384 / 64
    blk = 2 + 384 // 8 + 384 // 4
    assert cost(C("LN", rows=4, cols=384), hw)[0] == 4 * (1 + 3 * nb + 19) + blk
    assert cost(C("LN", rows=4, cols=384), hw, resident=True)[0] == 4 * (1 + 3 * nb + 19)
    assert cost(C("SOFTMAX_ONLINE", rows=2, cols=100, first=1, last=1), hw)[0] == 2 * (1 + 3 * 2 + 19) + 1
    assert cost(C("LUT_LOAD", table=1), hw)[0] == 32
    assert cost(C("GELU", len=640), hw)[0] == 1 + 10 + 16
    assert cost(C("REQUANT", M=3, N=100), hw)[0] == 3 * (1 + 2)
    # sfu.md examples at L = 4: LN d=384 307 cycles/row, SOFTMAX_RQ ctx 256 211 cycles/row (+accept)
    hw4 = KV260.with_(lanes=4)
    assert cost(C("LN", rows=1, cols=384), hw4, resident=True)[0] == 307 + 1
    assert cost(C("SOFTMAX_ONLINE", rows=1, cols=256, first=1, last=1), hw4, resident=True)[0] == 211 + 1
    # ping-pong row buffer (what-if) overlaps pass 1 of the next row
    pp = cost(C("LN", rows=4, cols=384), hw, MEASURED.with_(sfu_row_buffers=2), resident=True)[0]
    assert pp == 4 * (1 + 2 * nb + 19) + nb


def test_cyclemodel_ln_block_residency():
    """An LN block stays resident per PBUF address until a DMA writes that range."""
    hw = KV260
    ln = C("LN", rows=1, cols=384, param_addr=0)
    blk = 2 + 48 + 96
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
    cmds = [C("VEC", len=1)] * 64 + [C("END")]
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
    critical path is almost all weight DMA. Prefill is array bound."""
    assert shk["fits"]
    port_bound = 16 * KV260.clock_hz / SHAKESPEARE.weight_bytes
    assert 0.85 * port_bound < shk["decode_tok_s"] <= port_bound
    assert shk["decode_tok_s"] < 0.55 * shk["weight_bound_tok_s"]
    assert shk["decode_critical"]["dma weights"] > 0.85 * shk["decode_cycles"]
    assert shk["prefill_critical"]["gemm linear"] > 0.6 * shk["prefill_cycles"]
    assert shk["prefill_array_util"] > 0.7


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
