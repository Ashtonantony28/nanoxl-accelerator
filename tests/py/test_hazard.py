"""funcsim.check_hazards: the semaphore stream limit (docs/control.md §1). More than 255 + 8 SIGNALs
outstanding on one semaphore ahead of their WAITs deadlocks the RTL in-order dispatcher; the checker
rejects such streams, and the compiler never emits one for tiny or shakespeare."""
from pathlib import Path

import pytest

from nanoxl import isa
from nanoxl.compiler import Compiler
from nanoxl.export import load_bundle
from nanoxl.funcsim import SEM_STREAM_LIMIT, RaceError, check_hazards
from nanoxl.graph import SHAKESPEARE, build_gpt_graph, from_bundle, make_synthetic_params
from nanoxl.hwconfig import KV260, PYNQ_Z2

C = isa.C
TINY = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "tiny"


def burst_stream(n, consume_every=None):
    """n DMA->ARRAY SIGNALs on sem 0 (each after a tiny DMA), then the WAITs."""
    out = []
    for i in range(n):
        out += [C("DMA_LOAD", sp_buf=isa.ABUF, sp_addr=16 * (i % 64), ddr_addr=16 * i, rows=1, row_bytes=16),
                C("SIGNAL", queue=isa.Q_DMA, sem=0)]
        if consume_every and (i + 1) % consume_every == 0:
            out.append(C("WAIT", queue=isa.Q_ARR, sem=0, count=consume_every))
    if not consume_every:
        out += [C("WAIT", queue=isa.Q_ARR, sem=0, count=1)] * n
    return out + [C("END")]


def test_hazard_sem_limit_value():
    assert SEM_STREAM_LIMIT == 255 + 8


def test_hazard_sem_limit_boundary():
    st = check_hazards(burst_stream(SEM_STREAM_LIMIT))
    assert st["max_outstanding_sem"] == 263
    with pytest.raises(RaceError, match="deadlock"):
        check_hazards(burst_stream(SEM_STREAM_LIMIT + 1))


def test_hazard_sem_limit_counts_consumed_waits():
    """WAITs (with count k) retire signals, so a long stream that consumes as it goes is legal."""
    st = check_hazards(burst_stream(1000, consume_every=200))
    assert st["max_outstanding_sem"] == 200
    with pytest.raises(RaceError):
        check_hazards(burst_stream(1000, consume_every=264))
    # per semaphore: 200 on each of two semaphores is fine
    s = []
    for i in range(200):
        s += [C("SIGNAL", queue=isa.Q_DMA, sem=0), C("SIGNAL", queue=isa.Q_DMA, sem=1)]
    s += [C("WAIT", queue=isa.Q_ARR, sem=0, count=200), C("WAIT", queue=isa.Q_VEC, sem=1, count=200), C("END")]
    assert check_hazards(s)["max_outstanding_sem"] == 200


def test_hazard_sem_limit_custom():
    with pytest.raises(RaceError):
        check_hazards(burst_stream(10), sem_limit=9)


@pytest.mark.parametrize("name", ["w8a8", "w4a8"])
@pytest.mark.parametrize("hw", [KV260, KV260.with_(kv_onchip=False), PYNQ_Z2], ids=["kv260", "kv260-ddrKV", "pynq"])
def test_hazard_compiler_tiny_within_limit(name, hw):
    g, p = from_bundle(*load_bundle(TINY / f"{name}.npz"))
    comp = Compiler(g, p, hw, n_slots=1 if hw.kv_onchip else 3)
    steps = [[(0, 16)], [(16, 1)], [(127, 1)], [(0, 128)]]
    if not hw.kv_onchip:
        steps.append([(5, 1), (60, 1), (100, 2)])
    for seqs in steps:
        st = check_hazards(comp.compile_step(seqs).cmds, comp.img.base)
        assert st["max_outstanding_sem"] <= 8, (seqs, st)


@pytest.mark.parametrize("wbits", [8, 4])
def test_hazard_compiler_shakespeare_within_limit(wbits):
    cfg = SHAKESPEARE.with_(wbits=wbits)
    g = build_gpt_graph(cfg)
    comp = Compiler(g, make_synthetic_params(g, 0), KV260)
    for seqs in ([(0, 128)], [(128, 128)], [(255, 1)], [(10, 1)]):
        st = check_hazards(comp.compile_step(seqs).cmds, comp.img.base)
        assert st["max_outstanding_sem"] <= 8, (seqs, st)
