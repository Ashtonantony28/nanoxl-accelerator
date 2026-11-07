"""Functional simulator: compiled (tiled, K-accumulated, double-buffered) programs reproduce the untiled
C5 integer reference (nanoxl.graph.reference_run) bit for bit, including saturation counts, and the race
checker proves the sync is sufficient. The real-model end-to-end check is test_funcsim_golden.py."""
import numpy as np
import pytest

from nanoxl import isa
from nanoxl import numerics as nx
from nanoxl.compiler import Compiler
from nanoxl.funcsim import DDR, FuncSim, RaceError, SimError, check_hazards
from nanoxl.graph import (TINY, Graph, Op, build_gpt_graph, embed, make_synthetic_params, new_kv_state,
                          reference_run)
from nanoxl.hwconfig import KV260, PYNQ_Z2, KiB


def mlp_graph(K0, H, N, out_bits=16, residual=False):
    t = {"a": (8, K0), "h": (8, H), "y": (out_bits, N)}
    ops = [Op("linear", ["a"], ["h"], "fc"), Op("gelu", ["h"], ["h"], "gelu"), Op("linear", ["h"], ["y"], "fc2")]
    if residual:
        t["r"] = (16, N)
        ops.append(Op("add", ["r", "y"], ["r"]))
        return Graph(t, ops, ["a", "r"], ["r"])
    return Graph(t, ops, ["a"], ["y"])


def run_graph(g, params, hw, inputs, M):
    comp = Compiler(g, params, hw, max_rows=M)
    prog = comp.compile_step([(0, M)])
    sim = FuncSim(hw, comp.img)
    for k, v in inputs.items():
        sim.write_input(prog, k, v)
    sim.run(prog.cmds)
    check_hazards(prog.cmds, comp.img.base)
    return {k: sim.read_output(prog, k) for k in g.outputs}, prog


SMALL = KV260.with_(array_n=8, wbuf=2 * KiB, acc=4 * KiB, pbuf=8 * KiB, abuf=64 * KiB)


@pytest.mark.parametrize("seed", range(8))
def test_funcsim_tiled_mlp_matches_untiled(seed):
    """Random ragged shapes; tiny WBUF/ACC force K chunking (accumulate=1) and M tiling."""
    rng = np.random.default_rng(100 + seed)
    K0, H, N = (int(x) for x in rng.integers(1, 200, 3))
    M = int(rng.integers(1, 70))
    wbits = 4 if seed % 2 else 8
    na = [4, 8, 16][seed % 3]
    hw = SMALL.with_(array_n=na)
    g = mlp_graph(K0, H, N, out_bits=[8, 16][seed % 2], residual=seed % 4 == 3)
    for op in g.ops:
        op.attrs["wbits"] = wbits
    params = make_synthetic_params(g, seed)
    inputs = {"a": rng.integers(-128, 128, (M, K0)).astype(np.int8)}
    if "r" in g.tensors:
        inputs["r"] = rng.integers(-30000, 30000, (M, N)).astype(np.int16)
    out, prog = run_graph(g, params, hw, inputs, M)
    ref = reference_run(g, params, inputs)
    for k in g.outputs:
        assert out[k].dtype == ref[k].dtype
        assert np.array_equal(out[k], ref[k]), (k, K0, H, N, M, na, wbits)
    # independent check of the reference itself against the raw numerics
    p = params["fc"]
    if wbits == 4:   # W4A8: acc = sum_g gm[g, n] * acc_g, written out longhand
        acc = sum(p["gm"][g].astype(np.int64) * nx.gemm_i8(np.ascontiguousarray(inputs["a"][:, 64 * g:64 * g + 64]),
                                                           p["w"][64 * g:64 * g + 64]) for g in range(len(p["gm"])))
    else:
        acc = nx.gemm_i8(inputs["a"], p["w"])
    assert np.array_equal(reference_run(Graph(g.tensors, g.ops[:1], ["a"], ["h"]), params, inputs)["h"],
                          nx.requant(acc, p["mult"], p["shift"], p["bias"], 8))


def test_funcsim_tiling_really_happens():
    g = mlp_graph(150, 190, 70)
    params = make_synthetic_params(g, 5)
    rng = np.random.default_rng(5)
    a = rng.integers(-128, 128, (61, 150)).astype(np.int8)
    out, prog = run_graph(g, params, SMALL, {"a": a}, 61)
    gemms = [c for c in prog.cmds if c.op == "GEMM"]
    assert any(c["accumulate"] for c in gemms), "expected K chunking"
    assert len({c["a_addr"] for c in gemms}) > 4
    assert any(c["M"] < 61 for c in gemms), "expected M tiling"
    assert max(c["N"] for c in gemms) <= 190 and all(c["K"] <= 150 or c["K"] <= 190 for c in gemms)
    assert np.array_equal(out["y"], reference_run(g, params, {"a": a})["y"])
    # results are non-trivial: some saturation-free spread
    assert out["y"].std() > 10


def test_funcsim_extreme_values_saturate_like_reference():
    g = mlp_graph(64, 32, 16, out_bits=8)
    params = make_synthetic_params(g, 2)
    for k in ("fc", "fc2"):
        params[k]["w"][:] = -128
        params[k]["mult"][:] = (1 << 31) - 1
        params[k]["shift"][:] = 0
    a = np.full((5, 64), -128, np.int8)
    a[1] = 127
    out, _ = run_graph(g, params, SMALL, {"a": a}, 5)
    ref = reference_run(g, params, {"a": a})["y"]
    assert np.array_equal(out["y"], ref)
    assert (np.abs(ref.astype(int)) >= 127).mean() > 0.5


@pytest.mark.parametrize("hw", [KV260, PYNQ_Z2, KV260.with_(kv_onchip=False, array_n=16)],
                         ids=["kv260-onchipKV", "pynq-ddrKV", "kv260-ddrKV-n16"])
def test_funcsim_tiny_gpt_prefill_then_decode(hw):
    """Full tiny model (synthetic parameters): prefill 9 tokens then 3 decode steps; logits, KV cache
    and saturation counts bit-exact vs the untiled C5 reference."""
    cfg = TINY
    g = build_gpt_graph(cfg)
    params = make_synthetic_params(g, 0)
    comp = Compiler(g, params, hw)
    sim = FuncSim(hw, comp.img)
    kv = new_kv_state(cfg, 1)
    counter = {}
    steps = [(0, 9), (9, 1), (10, 1), (11, 1)]
    for i, (pos, n) in enumerate(steps):
        prog = comp.compile_step([(pos, n)])
        x0 = embed(np.arange(pos, pos + n) * 7 + 3, np.arange(pos, pos + n), cfg.d_model)
        sim.write_input(prog, "x", x0)
        sim.run(prog.cmds)
        out = sim.read_output(prog, "logits")
        ref = reference_run(g, params, {"x": x0}, [(pos, n)], kv, counter)["logits"]
        assert np.array_equal(out, ref), f"step {i}"
        assert ref.std() > 100
        if i == 0:
            check_hazards(prog.cmds, comp.img.base)
    # KV cache contents
    for l in range(cfg.n_layer):
        for kind in (0, 1):
            for h in range(cfg.n_head):
                if hw.kv_onchip:
                    got = sim.read(isa.KVBUF, comp.kv_onchip(0, l, kind, h), cfg.d_head, 12, cfg.d_head)
                else:
                    got = sim.read(DDR, comp.kv_ddr_addr(0, l, kind, h), cfg.d_head, 12, cfg.d_head)
                assert np.array_equal(got.view(np.int8), kv[0][l, kind, h, :12])
    nz = lambda d: {k: v for k, v in d.items() if v}
    assert nz(sim.sat_count) == nz(counter)
    ops = {c.op for c in prog.cmds}
    assert {"LN", "SOFTMAX_ONLINE", "GELU", "LUT_LOAD"} <= ops


def test_funcsim_batched_decode_different_positions():
    cfg = TINY
    g = build_gpt_graph(cfg)
    params = make_synthetic_params(g, 3)
    hw = KV260
    comp = Compiler(g, params, hw, n_slots=3)
    sim = FuncSim(hw, comp.img)
    kv = new_kv_state(cfg, 3)
    plan = [[(0, 5), (0, 2), (0, 7)], [(5, 1), (2, 1), (7, 1)], [(6, 1), (3, 1), (8, 1)]]
    for seqs in plan:
        prog = comp.compile_step(seqs)
        M = sum(n for _, n in seqs)
        x0 = embed(np.arange(M) + 11, np.concatenate([np.arange(p, p + n) for p, n in seqs]), cfg.d_model)
        sim.write_input(prog, "x", x0)
        sim.run(prog.cmds)
        ref = reference_run(g, params, {"x": x0}, seqs, kv)["logits"]
        assert np.array_equal(sim.read_output(prog, "logits"), ref)
    check_hazards(prog.cmds, comp.img.base)


def test_funcsim_vec_ops_saturate():
    hw = SMALL
    sim = FuncSim(hw, ddr_size=64)
    rng = np.random.default_rng(0)
    x = rng.integers(-32768, 32768, 64).astype(np.int16)
    y = rng.integers(-32768, 32768, 64).astype(np.int16)
    x[:4] = [32767, -32768, 32767, -32768]
    y[:4] = [1, -1, 32767, -32768]
    sim.write(isa.ABUF, 0, x.reshape(1, -1), 0)
    sim.write(isa.ABUF, 256, y.reshape(1, -1), 0)
    X, Y = x.astype(np.int64), y.astype(np.int64)
    cases = {"ADD": X + Y, "SUB": X - Y, "MUL": X * Y, "MAX": np.maximum(X, Y), "MIN": np.minimum(X, Y),
             "ADDI": X + 1000, "SHRR": nx.rdivpot(X, 3), "SHL": X << 3, "COPY": X}
    for name, exp in cases.items():
        for dt_out, bits in ((isa.DT_I8, 8), (isa.DT_I16, 16), (isa.DT_I32, 32)):
            imm = 3 if name in ("SHRR", "SHL") else 1000
            sim.step(isa.C("VEC", vop=isa.VOPS[name], dt_in=isa.DT_I16, dt_out=dt_out, src0_buf=0, src0_addr=0,
                           src1_buf=0, src1_addr=256, dst_buf=0, dst_addr=1024, len=64, imm=imm))
            dt = {8: np.int8, 16: np.int16, 32: np.int32}[bits]
            got = sim.read_t(isa.ABUF, 1024, 64, 1, 0, dt)[0]
            want = np.clip(exp, -(1 << (bits - 1)), (1 << (bits - 1)) - 1)
            assert np.array_equal(got, want), (name, bits)


def test_funcsim_bounds_and_reserved_ops():
    sim = FuncSim(SMALL, ddr_size=1024)
    with pytest.raises(SimError):
        sim.step(isa.C("GEMM", a_addr=SMALL.abuf - 4, M=2, N=2, K=8, a_stride=1))
    with pytest.raises(SimError):
        sim.step(isa.C("ATTN_TILE"))
    with pytest.raises(SimError):
        sim.step(isa.C("SOFTMAX_ONLINE", first=1, last=0))


@pytest.mark.parametrize("dh", [8, 24, 1])
def test_funcsim_rejects_kv_append_dh_not_multiple_of_16(dh):
    """vec_seq rejects KV_APPEND with dh % 16 != 0 (error 7); funcsim must too (AUDIT-1 A1-07)."""
    sim = FuncSim(KV260)
    with pytest.raises(SimError, match="multiple of 16"):
        sim.step(isa.C("KV_APPEND", src_addr=0, dst_addr=0, pos=0, heads=2, dh=dh, ctx=8, rows=1, src_stride=4))
    sim.step(isa.C("KV_APPEND", src_addr=0, dst_addr=0, pos=0, heads=2, dh=16, ctx=8, rows=1, src_stride=4))


def test_funcsim_race_checker_catches_missing_wait():
    g = mlp_graph(100, 64, 40)
    params = make_synthetic_params(g, 1)
    comp = Compiler(g, params, SMALL, max_rows=16)
    prog = comp.compile_step([(0, 16)])
    check_hazards(prog.cmds, comp.img.base)
    waits = [i for i, c in enumerate(prog.cmds) if c.op == "WAIT"]
    assert waits
    caught = 0
    for i in waits[:: max(1, len(waits) // 12)]:
        broken = prog.cmds[:i] + prog.cmds[i + 1:]
        try:
            check_hazards(broken, comp.img.base)
        except RaceError:
            caught += 1
    # every WAIT the compiler inserts is necessary: dropping any one creates a race
    assert caught == len(waits[:: max(1, len(waits) // 12)])
    # a WAIT placed before its SIGNAL in the stream violates the dispatch invariant
    j = waits[0]
    with pytest.raises(RaceError):
        check_hazards([prog.cmds[j]] + prog.cmds[:j] + prog.cmds[j + 1:], comp.img.base)
