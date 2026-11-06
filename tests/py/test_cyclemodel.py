"""Cycle model: per-command costs, queue/semaphore timing semantics, and model-level sanity bounds."""
import math

import pytest

from nanoxl import isa
from nanoxl.cyclemodel import ModelError, cost, estimate, simulate
from nanoxl.graph import SHAKESPEARE, TINY
from nanoxl.hwconfig import KV260, PYNQ_Z2

C = isa.C


def test_cyclemodel_dma_cost():
    hw = KV260  # 8 GB/s at 250 MHz = 32 B/cycle, 300 ns = 75 cycles
    occ, lat, info = cost(C("DMA_LOAD", rows=1, row_bytes=32768), hw)
    assert occ == 32768 // 32 + 4 and lat == 75 and info["ddr_bytes"] == 32768
    # strided small rows are burst-limited (one burst per row per cycle at best)
    occ2, _, _ = cost(C("DMA_LOAD", rows=1024, row_bytes=8, ddr_stride=64), hw)
    assert occ2 == 1024 + 4


def test_cyclemodel_gemm_cost():
    hw = KV260.with_(wload_rows_per_cycle=1)  # N = 32
    occ, lat, info = cost(C("GEMM", M=256, N=64, K=96), hw)
    assert occ == 32 + 2 * 3 * 256 and lat == 64 and info["macs"] == 256 * 64 * 96
    occ, lat, _ = cost(C("GEMM", M=1, N=64, K=96, gemv=1), hw)
    assert occ == 32 + 6 * 32 and lat == 32          # GEMV is weight-load bound
    occ4, _, _ = cost(C("GEMM", M=1, N=64, K=96, gemv=1, int4=1), hw)
    assert occ4 == 16 + 6 * 16                       # int4: two weight rows per byte row


def test_cyclemodel_queues_overlap_and_semaphores():
    hw = KV260.with_(wload_rows_per_cycle=1)
    d = C("DMA_LOAD", rows=1, row_bytes=32 * 1000)          # 1004 occ + 75 lat
    g = C("GEMM", M=500, N=32, K=32)                        # 32 + 500 occ, 64 lat
    # independent: run in parallel
    t = simulate([d, g, C("END")], hw)
    assert t.cycles < 1004 + 75 + 532 + 64
    # dependent: GEMM waits for the DMA to complete (occupancy + latency)
    dep = [d, C("SIGNAL", queue=0, sem=0), C("WAIT", queue=1, sem=0, count=1), g, C("END")]
    t2 = simulate(dep, hw, trace=True)
    start_g = [s for i, q, s, e in t2.trace if i == 3][0]
    assert start_g >= 1004 + 75
    assert t2.cycles >= 1004 + 75 + 532 + 64


def test_cyclemodel_rejects_wait_before_signal():
    with pytest.raises(ModelError):
        simulate([C("WAIT", queue=1, sem=0, count=1), C("SIGNAL", queue=0, sem=0), C("END")], KV260)


def test_cyclemodel_fifo_depth_stalls_dispatch():
    hw = KV260.with_(fifo_depth=2, issue_cycles=1)
    longd = C("DMA_LOAD", rows=1, row_bytes=32 * 5000)
    # one DMA running + two waiting fill a depth-2 FIFO; the 4th DMA blocks dispatch of the GEMM
    cmds = [longd, longd, longd, longd, C("GEMM", M=1, N=32, K=32), C("END")]
    deep = simulate(cmds, KV260.with_(issue_cycles=1), trace=True)
    shallow = simulate(cmds, hw, trace=True)
    gs = lambda t: [s for i, q, s, e in t.trace if i == 4][0]
    assert gs(deep) < 10 and gs(shallow) > 5000


@pytest.fixture(scope="module")
def shk():
    return estimate(SHAKESPEARE, KV260)


def test_cyclemodel_shakespeare_decode_vs_roofline(shk):
    """Design-plan back-of-envelope: ~750 tok/s INT8 at 8 GB/s. The model must not beat the pure
    weight-streaming bound and should reach >= 60% of it (goal G3)."""
    assert shk["fits"]
    assert shk["decode_tok_s"] <= shk["weight_bound_tok_s"]
    assert shk["decode_tok_s"] >= 0.6 * shk["weight_bound_tok_s"]
    assert 700 < shk["weight_bound_tok_s"] < 800
    assert shk["prefill_array_util"] > 0.5         # goal G3 prefill target
    assert shk["prefill_ms"] < 30


def test_cyclemodel_trends(shk):
    int4 = estimate(SHAKESPEARE.with_(wbits=4), KV260, prefill_len=0)
    assert int4["decode_tok_s"] > 1.5 * shk["decode_tok_s"]
    fast = estimate(SHAKESPEARE, KV260.with_(ddr_gbps=12), prefill_len=0)
    assert fast["decode_tok_s"] > 1.2 * shk["decode_tok_s"]
    # a 1-row/cycle weight port caps GEMV at N x f = 8 GB/s, so extra DDR bandwidth is wasted
    narrow = estimate(SHAKESPEARE, KV260.with_(ddr_gbps=12, wload_rows_per_cycle=1), prefill_len=0)
    assert narrow["decode_tok_s"] < 0.75 * fast["decode_tok_s"]
    b4 = estimate(SHAKESPEARE, KV260.with_(kv_onchip=False), batch=4, prefill_len=0)
    assert b4["decode_tok_s"] > 2.5 * shk["decode_tok_s"]
    assert not estimate(SHAKESPEARE, KV260, batch=4, prefill_len=0)["fits"]  # KV tier holds 1 sequence


def test_cyclemodel_pynq_and_tiny():
    pz = estimate(SHAKESPEARE, PYNQ_Z2, prefill_len=32)
    assert 0.6 * pz["weight_bound_tok_s"] <= pz["decode_tok_s"] <= pz["weight_bound_tok_s"]
    t = estimate(TINY, KV260)
    assert t["decode_tok_s"] > 5 * estimate(SHAKESPEARE, KV260, prefill_len=0)["decode_tok_s"]
    assert t["prefill_ms"] < 2
