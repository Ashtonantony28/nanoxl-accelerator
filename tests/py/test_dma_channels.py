"""SG10a / ADR-005: two DMA channels (engine = chan & 1). funcsim.check_hazards treats the channels as
separate agents (only same-channel program order, STORE fences, DMA SIGNAL/WAIT and barriers order
them); the compiler's SyncBuilder moves or orders conflicting LOADs; compiled programs use both
channels and stay bit-exact (test_funcsim_golden covers the full runs)."""
import pytest

from nanoxl import isa
from nanoxl.compiler import SEM, Compiler, SyncBuilder
from nanoxl.funcsim import RaceError, check_hazards
from nanoxl.graph import SHAKESPEARE, TINY, build_gpt_graph, make_synthetic_params
from nanoxl.hwconfig import KV260

C = isa.C
BASE = 0x4000_0000


def ld(ch, sp, ddr=BASE, n=256, buf=isa.WBUF):
    return C("DMA_LOAD", chan=ch, sp_buf=buf, sp_addr=sp, ddr_addr=ddr, rows=1, row_bytes=n)


def test_cross_channel_waw_is_a_race():
    """Two LOADs on different channels writing the same scratchpad bytes, nothing between: rejected."""
    with pytest.raises(RaceError):
        check_hazards([ld(0, 0x100), ld(1, 0x100, BASE + 0x1000), C("END")], BASE)
    # same channel: ordered by program order
    check_hazards([ld(0, 0x100), ld(2, 0x100, BASE + 0x1000), C("END")], BASE)
    # different channels, disjoint bytes: fine
    check_hazards([ld(0, 0x100), ld(1, 0x200, BASE + 0x1000), C("END")], BASE)


def test_cross_channel_ordering_points():
    """A STORE fence, a DMA SIGNAL -> DMA WAIT pair, or a barrier orders the two channels; a DMA
    SIGNAL alone does not (it is carried on done and does not block the queue)."""
    a, b = ld(0, 0x100), ld(1, 0x100, BASE + 0x1000)
    st = C("DMA_STORE", chan=1, sp_buf=isa.ABUF, sp_addr=0x4000, ddr_addr=BASE + 0x8000, rows=1, row_bytes=64)
    check_hazards([a, st, b, C("END")], BASE)
    s6 = SEM[(isa.Q_DMA, isa.Q_DMA)]
    check_hazards([a, C("SIGNAL", queue=0, sem=s6), C("WAIT", queue=0, sem=s6, count=1), b, C("END")], BASE)
    check_hazards([a, C("IRQ", code=1), b, C("END")], BASE)
    with pytest.raises(RaceError):
        check_hazards([a, C("SIGNAL", queue=0, sem=s6), b, C("WAIT", queue=0, sem=s6, count=1), C("END")], BASE)
    # a STORE reading what the other channel loaded is ordered by the fence (RAW across channels)
    rd = C("DMA_STORE", chan=1, sp_buf=isa.WBUF, sp_addr=0x100, ddr_addr=BASE + 0x9000, rows=1, row_bytes=64)
    check_hazards([a, rd, C("END")], BASE)


def test_cross_channel_signal_covers_both_channels():
    """A DMA SIGNAL covers every earlier DMA command on both channels: an ARRAY GEMM behind its WAIT
    may read what either channel loaded."""
    g = C("GEMM", M=1, N=16, K=16, w_addr=0x100, a_addr=0x0, c_addr=0x0)
    s0 = SEM[(isa.Q_DMA, isa.Q_ARR)]
    check_hazards([ld(1, 0x100), ld(0, 0x4000, BASE + 0x2000, buf=isa.ABUF), C("SIGNAL", queue=0, sem=s0),
                   C("WAIT", queue=1, sem=s0, count=1), g, C("END")], BASE)


def test_syncbuilder_moves_or_orders_conflicting_load():
    sb = SyncBuilder(BASE)
    sb.emit(ld(0, 0x100))
    sb.emit(ld(1, 0x100, BASE + 0x1000))        # conflicts with ch0 only: moved to ch0
    assert sb.moved == 1 and sb.entries[1].cmd["chan"] == 0 and not sb.entries[1].waits
    sb.emit(ld(1, 0x800, BASE + 0x2000))
    sb.emit(ld(0, 0x800, BASE + 0x3000))        # conflicts with ch1 only: moved to ch1
    assert sb.entries[3].cmd["chan"] == 1
    # conflicts with both channels: ordered with the DMA -> DMA semaphore
    sb.emit(C("DMA_LOAD", chan=0, sp_buf=isa.WBUF, sp_addr=0x100, ddr_addr=BASE + 0x4000, rows=1, row_bytes=0x800))
    assert [w["sem"] for w in sb.entries[4].waits] == [SEM[(isa.Q_DMA, isa.Q_DMA)]]
    check_hazards(sb.stream(), BASE)


@pytest.mark.parametrize("cfg", [TINY, SHAKESPEARE], ids=lambda c: c.name)
def test_compiled_programs_use_both_channels(cfg):
    g = build_gpt_graph(cfg)
    comp = Compiler(g, make_synthetic_params(g, 0), KV260)
    for seqs in ([(5, 1)], [(0, 16)]):
        cmds = comp.compile_step(seqs).cmds
        check_hazards(cmds, comp.img.base)
        by = [0, 0]
        for c in cmds:
            if c.op == "DMA_LOAD" and c["sp_buf"] == isa.WBUF:
                by[c["chan"] & 1] += c["rows"] * c["row_bytes"]
        assert min(by) > 0.4 * sum(by), by          # the weight stream is split over both masters
