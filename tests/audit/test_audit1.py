"""AUDIT-1 repro tests (docs/audits/AUDIT-1.md).

Failing findings are xfail(strict=True) with the finding id in the reason, so a fix turns them into
XPASS (= a failure that says "remove the xfail"). Passing tests close coverage holes the audit found.
Not collected by the default `pytest` (testpaths = tests/py, tests/rtl); run with

    python3 -m pytest -q tests/audit                 # fast part (< 1 min)
    NANOXL_DATA=<dir with input.txt> python3 -m pytest -q tests/audit -m slow   # numerics re-derivation (~10 s)
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

from nanoxl import cyclemodel as cm
from nanoxl import golden, isa
from nanoxl.driver import csr as R
from nanoxl.export import load_bundle
from nanoxl.hwconfig import KV260

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "artifacts" / "models" / "tiny"


def _gemm(op="GEMM", **kw):
    f = dict(accumulate=0, a_addr=0, w_addr=0, c_addr=0, a_stride=8, c_stride=64, gemv=0)
    if op == "GEMM":
        f.update(int4=0, w_trans=0, w_buf=0)
    f.update(kw)
    return isa.C(op, **f)


# ----------------------------------------------------------------------------- A1-02
_RTL2NAME = {"rq_cnt": "requant", "alu_cnt": "alu", "lno_cnt": "ln_out", "lnz_cnt": "ln_z", "sm_cnt": "score"}


def _vec_seq_sat_order():
    src = (ROOT / "rtl" / "vec_seq.sv").read_text()
    m = re.search(r"assign sat_cnt = \{([^}]*)\};", src)
    return [_RTL2NAME[s.strip()] for s in m.group(1).split(",")][::-1]   # concatenation is MSB first


def test_a1_02_sat_csr_map_matches_rtl():
    """A1-02 fixed (SG7c): one canonical SAT map, vec_seq.sat_cnt order, NSAT = 5 in RTL and driver."""
    rtl = _vec_seq_sat_order()
    assert rtl == ["requant", "alu", "ln_out", "ln_z", "score"]          # what the RTL drives (sanity)
    nsat_top = int(re.search(r"parameter int NSAT\s*=\s*(\d+)", (ROOT / "rtl/ctrl_top.sv").read_text()).group(1))
    assert R.NSAT == len(rtl) and nsat_top == len(rtl)
    assert list(R.SAT_NAMES) == rtl


def test_a1_02_simbackend_slots_follow_rtl():
    """A1-02 fixed (SG7c): the SimBackend charges each clamp to the slot vec_seq.sat_cnt uses (LN normalize
    clamp -> SAT_3 ln_z, LN output clamp -> SAT_2 ln_out, softmax score -> SAT_4), and nanoxl_top wires
    vec_seq.sat_cnt to ctrl_top unchanged."""
    src = (ROOT / "nanoxl/driver/backends.py").read_text()
    assert '"SOFTMAX_ONLINE": 4' in src and "self.sat[3] = _sat32(self.sat[3] + dz)" in src
    assert R.SAT_NAMES[2] == "ln_out" and R.SAT_NAMES[3] == "ln_z" and R.SAT_NAMES[4] == "score"
    top = (ROOT / "rtl/nanoxl_top.sv").read_text()
    assert re.search(r"assign sat_cnt\s*=\s*vec_sat;", top) and ".NSAT(NSAT)" in top


# ----------------------------------------------------------------------------- A1-01
def test_a1_01_vec_cost_respects_line_ports():
    """A1-01 (fixed): the cycle model prices VEC-engine streams at the 16-byte line ports
    (4 int32 / 8 int16 per cycle) and REQUANT's N record lines."""
    L = KV260.lanes
    assert L == 64
    M, N = 16, 384
    rq = isa.C("REQUANT", out16=0, per_tensor=0, src_addr=0, dst_addr=0, param_addr=0, M=M, N=N,
               src_stride=96, dst_stride=24)
    port_bound = M * (1 + math.ceil(4 * N / 16)) + N          # int32 lines per row + record lines
    assert cm.cost(rq, KV260)[0] >= port_bound
    sm = isa.C("SOFTMAX_ONLINE", first=1, last=1, causal=1, src_addr=0, dst_addr=0, ctx_addr=0, rows=M,
               cols=256, diag=0, src_stride=64, dst_stride=16)
    assert cm.cost(sm, KV260)[0] >= M * math.ceil(4 * 256 / 16)


PERF = ROOT / "bench" / "results" / "vec_seq_perf.json"


def _doc_perf_table():
    """docs/vector-unit.md §9.3 measured table: {shape name: (occ L=4, L=16, L=64)}."""
    txt = (ROOT / "docs" / "vector-unit.md").read_text()
    sec = txt[txt.index("### 9.3"):txt.index("### 9.4")]
    out = {}
    for line in sec.splitlines():
        m = re.match(r"\| (.+?) \| (\d+) \| (\d+) \| (\d+) \|", line)
        if m:
            out[m.group(1)] = tuple(int(m.group(i)) for i in (2, 3, 4))
    return out


def test_a1_01_cyclemodel_pinned_to_measured_vec_seq():
    """A1-01 (fixed): every VEC-engine shape measured on rtl/vec_seq.sv (bench/vec_seq_perf.py, L = 4 and
    16 on Icarus, 64 on Verilator; table in docs/vector-unit.md §9.3) is within 5 % of cyclemodel.cost,
    and the doc table equals the measurement file."""
    data = json.loads(PERF.read_text())
    assert sorted(data) == ["16", "4", "64"]
    doc = _doc_perf_table()
    assert len(doc) == len(data["4"]["rows"]) >= 30
    for L, v in data.items():
        hw = KV260.with_(lanes=int(L))
        for r in v["rows"]:
            assert doc[r["name"]][("4", "16", "64").index(L)] == r["occ"], (L, r["name"])
            c = isa.C(r["op"], **r["f"])
            resident = r["op"] in ("LN", "GELU", "SOFTMAX_ONLINE", "LUT_LOAD") and "miss" not in r["name"]
            got = cm.cost(c, hw, resident=resident)[0]
            assert abs(got / r["occ"] - 1) <= 0.05, (L, r["name"], got, r["occ"])


# ----------------------------------------------------------------------------- A1-05 / A1-06
MEASURED_N32 = {   # docs/array.md §7 table, measured occupancy at N=32 (Verilator)
    "w8a8 c_attn M16": (_gemm(M=16, N=256, K=128), 1024),
    "w4a8 GEMM_GS c_attn M16": (_gemm("GEMM_GS", M=16, N=384, K=128, glog2=3), 1584),
    "w4a8 GEMM_GS c_attn M1": (_gemm("GEMM_GS", M=1, N=384, K=128, glog2=3), 1584),
    "LM head ragged N=65": (_gemm(M=16, N=65, K=128), 624),
}


def test_a1_06_cyclemodel_now_matches_weight_bound_int8_tiles():
    """Evidence for A1-06 (stale docs): the measured-RTL cycle model already prices INT8 weight-bound
    tiles at one weight row per cycle, so array.md §7 'model occ' (528) and PLAN.md SG7a ('cycle model
    2-4x optimistic') describe the pre-SG4c model."""
    c, meas = MEASURED_N32["w8a8 c_attn M16"]
    assert cm.cost(c, KV260)[0] == meas


@pytest.mark.parametrize("key", ["w4a8 GEMM_GS c_attn M16", "w4a8 GEMM_GS c_attn M1"])
def test_a1_05_gemm_gs_scale_beat_per_tile(key):
    c, meas = MEASURED_N32[key]
    assert cm.cost(c, KV260)[0] == meas


def test_a1_05_lm_head_ragged_rows():
    c, meas = MEASURED_N32["LM head ragged N=65"]
    assert cm.cost(c, KV260)[0] == meas


ACCEPT_DONE_N32 = {   # docs/array.md §7 table, accept->done at N=32 (M16 / M1)
    "c_attn": (dict(N=256, K=128), 1115, 1100),
    "mlp proj": (dict(N=64, K=512), 1115, 1100),
    "LM head": (dict(N=65, K=128), 715, 700),
    "scores w_trans KVBUF": (dict(N=16, K=32, w_trans=1, w_buf=1), 139, None),
    "P.V KVBUF": (dict(N=32, K=16, w_buf=1), 107, None),
    "GEMM_GS c_attn": (dict(op="GEMM_GS", N=384, K=128, glog2=3), 1675, 1660),
    "GEMM_GS fc": (dict(op="GEMM_GS", N=480, K=128, glog2=3), 2071, 2056),
}


@pytest.mark.parametrize("key", sorted(ACCEPT_DONE_N32))
def test_a1_05_array_seq_accept_to_done(key):
    """A1-05 (fixed): occupancy + latency = measured array_seq accept->done (2N+4 fill + 7 + M write-back,
    + N/2 through the w_trans transposer), within 1 cycle."""
    kw, m16, m1 = ACCEPT_DONE_N32[key]
    kw = dict(kw)
    op = kw.pop("op", "GEMM")
    for M, meas in ((16, m16), (1, m1)):
        if meas is None:
            continue
        occ, lat, _ = cm.cost(_gemm(op, M=M, **kw), KV260)
        assert abs(occ + lat - meas) <= 1, (M, occ + lat, meas)


def test_a1_05_ln_and_lut_load_fixed_costs():
    """A1-05 (fixed): LN 1x128 at L=4 accept->done 134 (hit) / 185 (miss), LUT_LOAD 42 / 8 (docs/sfu.md §10)."""
    hw4 = KV260.with_(lanes=4)
    ln = isa.C("LN", in16=1, src_addr=0, dst_addr=0, param_addr=0, rows=1, cols=128, src_stride=16, dst_stride=8)
    hit, miss = (sum(cm.cost(ln, hw4, resident=r)[:2]) for r in (True, False))
    assert (hit, miss) == (134, 185)
    lut = isa.C("LUT_LOAD", table=1, src_addr=512)
    assert tuple(sum(cm.cost(lut, hw4, resident=r)[:2]) for r in (False, True)) == (42, 8)


# ----------------------------------------------------------------------------- A1-09 coverage hole
@pytest.mark.parametrize("ls", [1, 3])
def test_a1_09_score_ls_path_golden_equals_funcsim(ls):
    """Both committed bundles have score.ls = 0, so no existing test runs the golden model's
    `acc << ls` path against the compiled program. Bundle variant with ls>0 (shift raised by ls to keep
    the scale): compiled prefill + 2 decode steps in funcsim equal nanoxl.golden bit for bit."""
    from nanoxl.funcsim import HostRunner
    A, meta = load_bundle(TINY / "w8a8.npz")
    A = dict(A)
    for l in range(meta["n_layer"]):
        assert int(A[f"h{l}.score.ls"][0]) == 0
        A[f"h{l}.score.ls"] = np.array([ls], np.int32)
        A[f"h{l}.score.shift"] = (A[f"h{l}.score.shift"] + ls).astype(np.int32)
        assert int(A[f"h{l}.score.shift"][0]) <= 31
    ref = json.loads((TINY / "golden_ref.json").read_text())
    prompt = ref["prompt"]
    r = HostRunner(A, meta, KV260, name="w8a8-ls")
    trace = []
    toks = r.generate(prompt, 3, trace=trace)
    g = golden.Golden(A, meta)
    cache = g.new_cache()
    want = [g.forward(prompt, cache)] + [g.forward([t], cache) for t in toks[:-1]]
    for i, (got, w) in enumerate(zip(trace, want)):
        assert np.array_equal(got, w), f"step {i}"
    assert {k: v for k, v in r.sat_count.items() if v} == {k: v for k, v in g.sat_count.items() if v}


# ----------------------------------------------------------------------------- numerics re-derivation
DATA = Path(os.environ.get("NANOXL_DATA", ROOT / "data" / "raw"))
_need_data = pytest.mark.skipif(not (DATA / "input.txt").exists(),
                                reason=f"Tiny Shakespeare not at {DATA} (set NANOXL_DATA)")


@pytest.mark.slow
@_need_data
def test_a1_numerics_committed_bundles_rebuild_identically():
    """The committed bundles are exactly build_bundle(ckpt, 16 train-split calibration windows)."""
    from nanoxl.export import calib_windows
    from nanoxl.model_fp import load_checkpoint, load_shakespeare, numpy_weights
    from nanoxl.quant.quantize import QuantConfig, build_bundle
    train, _, _ = load_shakespeare(DATA)
    model = load_checkpoint(TINY / "ckpt.pt")
    cal = calib_windows(train, model.config.block_size, QuantConfig().calib_seqs)
    W = numpy_weights(model)
    for name, wb in (("w8a8", 8), ("w4a8", 4)):
        A, _ = build_bundle(W, model.config, cal, QuantConfig(wbits=wb))
        C, _ = load_bundle(TINY / f"{name}.npz")
        assert sorted(A) == sorted(C)
        assert all(np.array_equal(A[k], C[k]) for k in C), name


@pytest.mark.slow
@_need_data
def test_a1_numerics_quant_study_headline_rederived():
    """Every 5th of the study's 200 val windows (40 windows, 5,120 tokens): ΔPPL vs FP32 on the same
    windows. Audit run: W8A8 +0.55 % (study +0.96 %), W4A8 +3.28 % (study +3.29 %)."""
    import torch
    from nanoxl.model_fp import load_checkpoint, load_shakespeare
    _, val, _ = load_shakespeare(DATA)
    model = load_checkpoint(TINY / "ckpt.pt")
    T = model.config.block_size
    nwin = (len(val) - 1) // T
    widx = np.linspace(0, nwin - 1, min(200, nwin)).astype(int)[::5]
    wins = [val[i * T:i * T + T + 1].astype(np.int64) for i in widx]

    def ce_rows(l, y):
        l = l - l.max(-1, keepdims=True)
        return -(l[np.arange(len(y)), y] - np.log(np.exp(l).sum(-1)))
    with torch.no_grad():
        lf = model(torch.from_numpy(np.stack([w[:T] for w in wins])))[0].double().numpy()
    fp = float(np.mean([ce_rows(lf[i], w[1:]).mean() for i, w in enumerate(wins)]))
    golden.set_gemm(golden.gemm_exact_f64)
    try:
        study = {"w8a8": 0.96, "w4a8": 3.29}
        for name in ("w8a8", "w4a8"):
            g = golden.Golden(*load_bundle(TINY / f"{name}.npz"))
            s = sum(float(ce_rows(g.forward(w[:T]).astype(np.float64) * g.meta["scales"]["logits"],
                                  w[1:]).sum()) for w in wins)
            dppl = 100 * (math.exp(s / (len(wins) * T) - fp) - 1)
            assert abs(dppl - study[name]) < 1.0, (name, dppl)
            assert dppl < (3.0 if name == "w8a8" else 8.0)
    finally:
        golden.set_gemm(golden.nx.gemm_i8)


# ----------------------------------------------------------------------------- A1-03 coverage hole
def test_a1_03_c5_rounding_vectors_from_contract_text():
    """Hand-derived from docs/contracts.md C5 (not from ops.py): which rounding each step uses.
    tests/py checks these ops only against float bounds, so swapping rdivpot for rshr in the LN affine
    step survived every `tests/py -k golden` test (AUDIT-1 mutation gold_lnaff_rshr)."""
    from nanoxl.quant import ops
    # LN step 7: out = sat8(rdivpot(z*g + b, sh)): halves round away from zero
    assert ops.ln_affine_i(np.array([-1, 1, -3, 3]), np.ones(4), np.zeros(4), 1).tolist() == [-1, 1, -2, 2]
    # LN step 6 / softmax step 6: rshr = round half up
    assert ops.rshr(np.array([-1, 1, -3, 3]), 1).tolist() == [0, 1, -1, 2]
    # EXP2: q = d >> 8, r = d & 255, e = lut[r] >> q, 0 for q >= 16
    lut = ops.make_exp2_lut()
    assert ops.exp2_neg_i(np.array([0, 256, 257, 15 * 256 + 255, 16 * 256]), lut).tolist() == \
        [32768, 16384, int(lut[1]) >> 1, int(lut[255]) >> 15, 0]
