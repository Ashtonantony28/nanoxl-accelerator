"""Compiler skeleton: allocators, DDR map, tiling plan, sync invariants, field coverage on real models."""
import numpy as np
import pytest

from nanoxl import isa
from nanoxl.compiler import Arena, CompileError, Compiler, Ring, relocate
from nanoxl.cyclemodel import RtlTiming, simulate
from nanoxl.funcsim import check_hazards
from nanoxl.graph import M25, SHAKESPEARE, TINY, build_gpt_graph, make_synthetic_params
from nanoxl.hwconfig import KV260, PYNQ_Z2, KiB


def test_compiler_arena_first_fit():
    a = Arena(1024, "T")
    x, y, z = a.alloc(100), a.alloc(200), a.alloc(16)
    assert (x, y, z) == (0, 112, 320)
    a.free(y)
    assert a.alloc(64) == 112          # reuses the hole
    assert a.alloc(300) == 336
    with pytest.raises(CompileError):
        a.alloc(1024)


def test_compiler_ring_pins():
    r = Ring(100 * 16, "R")
    p = r.alloc(400, pin=True)
    u = [r.alloc(400) for _ in range(5)]
    assert all(not (s < p + 400 and p < s + 400) for s in u)  # never overlaps the pinned block
    r.release(p)
    with pytest.raises(CompileError):
        r.alloc(2000)
    r2 = Ring(1024, "R2")
    r2.alloc(600, pin=True)
    with pytest.raises(CompileError):
        r2.alloc(600, pin=True)


@pytest.fixture(scope="module")
def tiny():
    g = build_gpt_graph(TINY)
    return g, make_synthetic_params(g, 0)


def test_compiler_ddr_map(tiny):
    g, p = tiny
    comp = Compiler(g, p, PYNQ_Z2)
    regs = sorted(comp.img.regions.values())
    for (a0, s0), (a1, _) in zip(regs, regs[1:]):
        assert a0 % 4096 == 0 and a0 + s0 <= a1
    assert set(comp.img.regions) >= {"cmd_ring", "params", "weights", "kv", "io_x", "io_logits"}
    w0, ws = comp.img.regions["weights"]
    assert ws >= TINY.weight_bytes
    for plan in comp.lin.values():
        for pn in plan.panels:
            assert w0 <= pn.ddr and pn.ddr + pn.nbytes <= w0 + ws
            assert pn.kc * pn.ntp * plan.wbits // 8 == pn.nbytes
    assert "kv" not in Compiler(g, p, KV260).img.regions  # on-chip KV tier


def test_compiler_tiling_plan_fits_wbuf():
    comp = Compiler(build_gpt_graph(SHAKESPEARE), make_synthetic_params(build_gpt_graph(SHAKESPEARE), 0),
                    KV260, max_rows=4)
    for name, plan in comp.lin.items():
        assert plan.nt % KV260.array_n == 0 or plan.nt >= plan.N
        for pn in plan.panels:
            assert pn.nbytes <= KV260.wbuf // 2      # two panels in flight => double buffering
        ks = sorted({(pn.k0, pn.kc) for pn in plan.panels})
        assert sum(kc for _, kc in ks) == plan.K
        cols = sorted({(pn.n0, pn.ntc) for pn in plan.panels})
        assert sum(n for _, n in cols) == plan.N


def test_compiler_sync_invariants_and_encoding(tiny):
    g, p = tiny
    comp = Compiler(g, p, KV260)
    prog = comp.compile_step([(0, 16)])
    assert prog.cmds[-1].op == "END" and sum(c.op == "END" for c in prog.cmds) == 1
    stats = check_hazards(prog.cmds, comp.img.base)
    assert stats["pairs_checked"] > 100
    # one producer and one consumer queue per semaphore
    prod, cons = {}, {}
    for c in prog.cmds:
        if c.op == "SIGNAL":
            prod.setdefault(c["sem"], set()).add(c["queue"])
        if c.op == "WAIT":
            cons.setdefault(c["sem"], set()).add(c["queue"])
    assert all(len(v) == 1 for v in prod.values()) and all(len(v) == 1 for v in cons.values())
    assert set(prod) == set(cons)
    # binary ring round trip
    ring = prog.ring_bytes()
    assert len(ring) == 16 * len(prog.cmds)
    assert isa.decode_stream(ring) == prog.cmds
    # weight DMAs overlap compute: some DMA_LOAD into WBUF is not preceded by a DMA wait at all
    tl = simulate(prog.cmds, KV260)
    assert tl.cycles < tl.busy["dma"] + tl.busy["array"] + tl.busy["vec"]
    assert tl.max_outstanding_sem < 256


def test_compiler_double_buffering_waits_two_panels_back():
    """With a WBUF ring that holds exactly two panels, the DMA of panel i waits for the GEMM of panel
    i-2 (WAR on the ring slot) and runs while the GEMM of panel i-1 computes (prefetch overlap)."""
    from nanoxl.graph import Graph, Op
    g = Graph({"a": (8, 64), "y": (8, 256)}, [Op("linear", ["a"], ["y"], "fc")], ["a"], ["y"])
    p = make_synthetic_params(g, 0)
    hw = KV260.with_(array_n=8, wbuf=1024)
    comp = Compiler(g, p, hw, max_rows=64)
    assert all(pn.nbytes == 512 for pn in comp.lin["fc"].panels)
    prog = comp.compile_step([(0, 64)])
    # an ordering test: idealized DMA timing (pipelined, non-blocking SIGNAL) so the 2-panel overlap
    # shows; the measured-timing utilization is checked in test_compiler_decode_utilization_measured
    ideal = RtlTiming(dma_coalesce=True, dma_pipelined=True, signal_blocks=False)
    tl = simulate(prog.cmds, hw, trace=True, tm=ideal)
    t = {i: (s, e) for i, _, s, e in tl.trace}
    dma = [i for i, c in enumerate(prog.cmds) if c.op == "DMA_LOAD" and c["sp_buf"] == isa.WBUF]
    gemm = [i for i, c in enumerate(prog.cmds) if c.op == "GEMM"]
    assert len(dma) == len(gemm) == 32
    for k in range(2, 32):
        prev = prog.cmds[dma[k] - 1]
        assert prev.op == "WAIT" and prev["queue"] == isa.Q_DMA
        assert t[dma[k]][0] >= t[gemm[k - 2]][1]      # slot reuse waits for the GEMM two back
        assert t[dma[k]][0] < t[gemm[k - 1]][1]       # ...and overlaps the GEMM one back
    assert prog.cmds[dma[1] - 1].op != "WAIT"
    check_hazards(prog.cmds, comp.img.base)
    assert tl.array_util > 0.8


def test_compiler_relocate(tiny):
    g, p = tiny
    comp = Compiler(g, p, KV260)
    prog = comp.compile_step([(3, 1)])
    moved = relocate(prog.cmds, 0x1000_0000)
    for a, b in zip(prog.cmds, moved):
        if a.op.startswith("DMA"):
            assert b["ddr_addr"] == a["ddr_addr"] + 0x1000_0000
        else:
            assert a == b


def test_compiler_chunked_prefill_and_limits(tiny):
    g, p = tiny
    small = KV260.with_(abuf=96 * KiB)
    comp = Compiler(g, p, small)
    progs = comp.compile_prefill(128)
    assert len(progs) > 1
    assert sum(pr.seqs[0][1] for pr in progs) == 128
    assert [pr.seqs[0][0] for pr in progs] == list(range(0, 128, progs[0].seqs[0][1]))
    with pytest.raises(CompileError):
        comp.compile_step([(120, 9)])           # beyond ctx
    with pytest.raises(CompileError):
        comp.compile_step([(0, 1), (0, 1)])     # one KV slot only
    with pytest.raises(CompileError):
        Compiler(g, p, KV260.with_(kvbuf=64 * KiB))   # on-chip KV does not fit


def test_compiler_batch_kv_capacity():
    g = build_gpt_graph(SHAKESPEARE)
    p = make_synthetic_params(g, 0)
    Compiler(g, p, KV260, n_slots=1)
    with pytest.raises(CompileError):
        Compiler(g, p, KV260, n_slots=4)          # 4 x 1.18 MB does not fit the URAM tier
    Compiler(g, p, KV260.with_(kv_onchip=False), n_slots=8)


@pytest.mark.parametrize("cfg", [SHAKESPEARE, SHAKESPEARE.with_(wbits=4), M25], ids=lambda c: f"{c.name}-w{c.wbits}")
def test_compiler_real_models_fit_isa_fields(cfg):
    """Every command of a full prefill and a late decode step validates (field widths suffice)."""
    g = build_gpt_graph(cfg)
    p = make_synthetic_params(g, 0)
    hw = KV260 if cfg.kv_bytes() <= KV260.kvbuf else KV260.with_(kv_onchip=False)
    comp = Compiler(g, p, hw)
    progs = comp.compile_prefill(cfg.ctx) + [comp.compile_step([(cfg.ctx - 1, 1)])]
    for pr in progs:
        for c in pr.cmds:
            isa.validate(c)
    dec = progs[-1]
    gemm_w = sum(c["K"] * c["N"] * (0.5 if c["int4"] else 1) for c in dec.cmds if c.op == "GEMM" and not c["w_buf"])
    gemm_w += sum(c["K"] * c["N"] // 2 for c in dec.cmds if c.op == "GEMM_GS")
    assert gemm_w == pytest.approx(cfg.weight_bytes, rel=0.01)  # each weight streamed once per token
    if cfg.wbits == 4:   # W4A8 blocks use GEMM_GS; group scales add K/64 x N bytes (1/32 of the INT4 weights)
        gs = [c for c in dec.cmds if c.op == "GEMM_GS"]
        assert gs and all(isa.gs_group(c["glog2"]) == 64 for c in gs)
        scale_b = sum(isa.gs_layout(c["K"], c["N"], c["glog2"])[2] - c["K"] * c["N"] // 2 for c in gs)
        assert scale_b == pytest.approx(cfg.block_params / 64, rel=0.02)
    for pr in progs:
        isa.encode_stream(pr.cmds, aligned=True)     # alignment rule holds for every emitted command


def test_compiler_sync_regression_shadow_growth():
    """Regression: a command whose later access rect grows the shadow arrays (KV_APPEND head 1 lies
    far beyond head 0) must still record its earlier rects; a cross-queue reader of head 0 needs a WAIT."""
    from nanoxl.compiler import SyncBuilder
    sb = SyncBuilder(0)
    sb.emit(isa.C("KV_APPEND", src_addr=0, dst_addr=0, pos=0, heads=2, dh=16, ctx=4096, rows=4, src_stride=2))
    sb.emit(isa.C("GEMM", w_trans=1, w_buf=1, w_addr=0, a_addr=4096, M=1, N=4, K=16, a_stride=1, c_stride=1))
    s = sb.stream()
    assert [c.op for c in s] == ["KV_APPEND", "SIGNAL", "WAIT", "GEMM", "END"]
    check_hazards(s)


def test_compiler_alignment_rule_and_w4_group_panels():
    """Alignment rule (docs/isa.md §3): panels split at lcm(N_array, 16) along K and N, W4A8 panels start
    on a group boundary and carry their scale rows; the sync builder refuses an unaligned command."""
    from nanoxl.compiler import SyncBuilder
    from nanoxl.graph import Graph, Op
    g = Graph({"a": (8, 200), "y": (8, 150)}, [Op("linear", ["a"], ["y"], "fc", {"wbits": 4})], ["a"], ["y"])
    p = make_synthetic_params(g, 3)
    for na in (4, 8, 32):
        hw = KV260.with_(array_n=na, wbuf=2 * KiB * (1 + (na == 32)), panel_bytes=KiB * (1 + (na == 32)))
        comp = Compiler(g, p, hw, max_rows=8)
        plan = comp.lin["fc"]
        assert plan.wbits == 4 and plan.group == 64 and isa.gs_group(plan.glog2) == 64
        for pn in plan.panels:
            assert pn.k0 % 64 == 0 and pn.n0 % 16 == 0 and pn.ntp % 2 == 0
            assert pn.nbytes == isa.gs_layout(pn.kc, pn.ntp, plan.glog2)[2] <= hw.wbuf // 2
        assert len({pn.k0 for pn in plan.panels}) > 1            # K really split (at group boundaries)
        prog = comp.compile_step([(0, 8)])
        isa.encode_stream(prog.cmds, aligned=True)
        assert any(c.op == "GEMM_GS" and c["accumulate"] for c in prog.cmds)
    sb = SyncBuilder(0)
    with pytest.raises(isa.IsaError):
        sb.emit(isa.C("GEMM", a_addr=8, M=1, N=16, K=16))
    with pytest.raises(CompileError):
        relocate(prog.cmds, 8)


def test_compiler_from_bundle_graph():
    """graph.from_bundle maps the ADR-002 bundle onto the compiler graph: golden names, INT4 unpacked
    with group scales, int16 logits, real LN parameter blocks in the DDR image."""
    from pathlib import Path
    from nanoxl import numerics as nx
    from nanoxl.export import load_bundle
    from nanoxl.funcsim import ln_block_layout
    from nanoxl.graph import from_bundle
    root = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "tiny"
    A, meta = load_bundle(root / "w4a8.npz")
    g, p = from_bundle(A, meta, "tiny-w4a8")
    assert g.cfg.wbits == 4 and g.tensors["logits"] == (16, 65) and g.tensors["x"] == (16, 128)
    assert p["h0.c_fc"]["wbits"] == 4 and p["h0.c_fc"]["gm"].shape == (2, 512) and p["lm_head"]["wbits"] == 8
    assert np.array_equal(nx.pack_i4(p["h1.mlp_proj"]["w"]), A["h1.mlp_proj.w4"])
    comp = Compiler(g, p, KV260)
    addr, n = comp.blob["h0.ln_1"]
    assert n == ln_block_layout(128)[2]
    blob = dict(comp.img.chunks)[addr - comp.img.base]
    assert np.frombuffer(blob[:16], np.int32).tolist() == [int(v) for v in A["h0.ln_1.c"][[0, 1, 2, 4]]]
    assert np.frombuffer(blob[16:24], np.int64)[0] == A["h0.ln_1.c"][3]
    assert {op.param for op in g.ops if op.kind == "attn"} == {f"h{l}.attn" for l in range(4)}
    with pytest.raises(ValueError):
        from_bundle(A, dict(meta, quant=dict(meta["quant"], res_bits=8)))


def test_compiler_decode_utilization_measured():
    """Regression for the coalesced DMA emission: under the MEASURED RTL timing (one 128-bit DMA
    master, 4 outstanding bursts, a realigner bubble per row) shakespeare decode on the KV260 keeps the
    DMA port >= 85 % busy moving bytes, prefill keeps the array >= 70 % busy, and no compiled DMA moves
    a contiguous transfer as several rows."""
    from nanoxl.cyclemodel import estimate, timing_for
    from nanoxl.graph import SHAKESPEARE
    g = build_gpt_graph(SHAKESPEARE)
    comp = Compiler(g, make_synthetic_params(g, 0), KV260)
    for seqs in ([(100, 1)], [(0, 128)]):
        for c in comp.compile_step(seqs).cmds:
            if c.op in ("DMA_LOAD", "DMA_STORE") and c["rows"] > 1:
                contig_sp = c["sp_stride"] in (0, c["row_bytes"] // 16) and c["row_bytes"] % 16 == 0
                assert not (c["ddr_stride"] in (0, c["row_bytes"]) and contig_sp), c
    r = estimate(SHAKESPEARE, KV260, n_pos=3)
    port = timing_for(KV260).dma_beat_bytes
    cycles = KV260.clock_hz / r["decode_tok_s"]
    assert r["decode_ddr_MB_per_step"] * 1e6 / (port * cycles) > 0.85
    assert r["prefill_array_util"] > 0.7
