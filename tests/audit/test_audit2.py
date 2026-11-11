"""AUDIT-2 repro tests (docs/audits/AUDIT-2.md).

Failing findings were xfail(strict=True) with the finding id in the reason; the AUDIT-2 fix branch
closed all of them, and those tests now pin the fixes (test_a2_0x_*). Other tests pin claims the audit re-derived
(arithmetic identities used by the SG11 area changes, the CSR map, ADR-005 compiler claims).
Not collected by the default `pytest` (testpaths = tests/py, tests/rtl); run with

    python3 -m pytest -q tests/audit/test_audit2.py -rxX          # < 30 s

The AUDIT-2 mutation list is MUTS below. Run it on a scratch COPY of the repo, never on the working
tree (each mutant rewrites a source in place and restores it afterwards):

    cp -r rtl tests nanoxl formal artifacts pyproject.toml /tmp/mut2/
    python3 tests/audit/test_audit2.py /tmp/mut2                 # all mutants, one JSON line each
    python3 tests/audit/test_audit2.py /tmp/mut2 pp_nocorr       # one mutant
    SIM=verilator SEL="tests/rtl/test_vec_seq.py::test_vec_seq[16-1]" python3 tests/audit/test_audit2.py /tmp/mut2 rd_full_k1
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _src(p):
    return (ROOT / p).read_text()


# ----------------------------------------------------------------------------- arithmetic identities
def test_a2_wp486_pair_identity_exhaustive():
    """rtl/pe.sv pe_pair, modelled bit for bit in numpy over all 2^24 (a, wA, wB): P = (wA*2^18 + wB)*a
    as a 36-bit two's-complement word, lo = P[17:0] signed, hi = P[35:18] + P[17]. Includes the
    extremes the brief asks about (a = wA = wB = -128: lo = +16384 so no borrow; wB*a < 0: lo's sign
    borrows one from hi, which the + P[17] term returns). The RTL bench (tb_pe_pair) checks the same
    exhaustively only on Verilator; Icarus (CI) runs every 13th case."""
    v = np.arange(-128, 128, dtype=np.int64)
    a = v[:, None, None]
    wa = v[None, :, None]
    wb = v[None, None, :]
    p = ((wa << 18) + wb) * a
    p36 = p & ((1 << 36) - 1)
    lo = p36 & ((1 << 18) - 1)
    lo = np.where(lo >= 1 << 17, lo - (1 << 18), lo)
    hi = (p36 >> 18) + ((p36 >> 17) & 1)
    hi &= (1 << 18) - 1
    hi = np.where(hi >= 1 << 17, hi - (1 << 18), hi)
    assert np.array_equal(lo, np.broadcast_to(a * wb, lo.shape))
    assert np.array_equal(hi, np.broadcast_to(a * wa, hi.shape))
    # without the correction term exactly the negative low products break (the mutant pp_nocorr)
    bad = ((p36 >> 18) & ((1 << 18) - 1)) != (hi & ((1 << 18) - 1))
    assert np.array_equal(bad, np.broadcast_to(a * wb < 0, bad.shape))


def test_a2_srdhm_stage3_identity():
    """rtl/srdhm.sv (SG11c C6): trunc_div(p + nudge, 2^31) == ((p >>> 30) + 1) >>> 1, taken as the low
    32 bits of a 34-bit increment, equals numerics.srdhm for random and extreme operands."""
    from nanoxl import numerics as nx
    rng = np.random.default_rng(2)
    ext = np.array([-(1 << 31), -(1 << 31) + 1, -1, 0, 1, (1 << 30), (1 << 31) - 1], np.int64)
    a = np.concatenate([rng.integers(-(1 << 31), 1 << 31, 400_000), np.repeat(ext, len(ext))])
    b = np.concatenate([rng.integers(-(1 << 31), 1 << 31, 400_000), np.tile(ext, len(ext))])
    p = a * b                                                     # fits int64 except INT32_MIN^2
    q = ((p >> 30) + 1) & ((1 << 34) - 1)
    y = (q >> 1) & 0xFFFF_FFFF
    y = np.where(y >= 1 << 31, y - (1 << 32), y)
    y = np.where((a == -(1 << 31)) & (b == -(1 << 31)), (1 << 31) - 1, y)
    assert np.array_equal(y, nx.srdhm(a, b).astype(np.int64))


def test_a2_vec_alu_mul_rounding_model():
    """rtl/vec_alu.sv multi-cycle MUL (SG11c C4): 64-bit product shifted right shamt times keeping the
    last-out bit (rb) and the OR of the earlier ones (st), clamped to 34 bits, + (rb && (P >= 0 || st)),
    then saturated, equals vec_ref.alu(MUL) for every shamt (incl. exact halves of both signs)."""
    from nanoxl import vec_ref as vr
    rng = np.random.default_rng(3)
    a = rng.integers(-(1 << 31), 1 << 31, 60_000)
    b = rng.integers(-(1 << 31), 1 << 31, 60_000)
    a[:64] = -(1 << 31)
    b[:32] = -(1 << 31)
    a[64:2000] = rng.integers(-300, 300, 1936)                  # small products: many exact halves
    b[64:2000] = rng.integers(-300, 300, 1936)
    for sh in range(32):
        P = a * b
        floor = P >> sh
        rb = (P >> (sh - 1)) & 1 if sh else np.zeros_like(P)
        st = (P & ((1 << max(sh - 1, 0)) - 1)) != 0 if sh > 1 else np.zeros_like(P, bool)
        lim = 1 << 33
        mcl = np.clip(floor, -lim, lim - 1)
        res = mcl + ((rb == 1) & ((P >= 0) | st))
        for fmt in (0, 1, 2):
            got, gs = vr.saturate(res, fmt)
            ref, rs = vr.alu(vr.OP_MUL, a, b, fmt=fmt, shamt=sh)
            assert np.array_equal(got, ref.astype(np.int64)) and np.array_equal(gs, rs), (sh, fmt)


def test_a2_vec_reduce_stage_alignment_all_L():
    """rtl/vec_reduce.sv (SG11c C7): for every L in 1..64 the number of registered tree levels
    (depths 0 < d < LV with (LV - d) % 2 == 0) plus the root register equals NTS, the length of the
    valid/last/err delay line, so data and valid stay aligned. L = 32 / 64 (NTS = 3) have no module
    test of their own (only test_vec_seq_l64, slow, Verilator)."""
    for L in range(1, 65):
        P = 1 if L <= 1 else 1 << (L - 1).bit_length()
        LV = P.bit_length() - 1
        nts = 1 if LV <= 2 else (LV + 1) // 2
        cuts = sum(1 for d in range(1, LV) if (LV - d) % 2 == 0)
        assert cuts + 1 == nts, L
        assert L not in (16, 64) or nts == {16: 2, 64: 3}[L]


def test_a2_barrel_unpack_slice_index():
    """rtl/vec_seq_rd.sv (SG11b C9): slice `sub` of a narrow beat starts at byte sub * L * 2^ebl, the
    shift is su = sub << ebl units of L bytes, and su < NU = 16P / L whenever sub can be non-zero, so
    the log2(NU)-stage barrel never truncates the unit index (all L, P, ebl the engine is built for)."""
    for L in (1, 2, 4, 8, 16, 32, 64):
        for P in (1, 2, 4):
            NU = (16 * P) // L if 16 * P > L else 1
            UW = max(1, (NU - 1).bit_length()) if NU > 1 else 1
            for ebl in (0, 1, 2):
                LL = (L - 1).bit_length()
                PL = (P - 1).bit_length()
                bpl_l2 = (4 + PL - ebl - LL) if ebl + LL < 4 + PL else 0
                for sub in range(1 << bpl_l2):
                    su = sub << ebl
                    assert su * L == sub * L * (1 << ebl)
                    assert su < NU or (sub == 0 and su == 0)
                    assert su < (1 << UW) or su == 0


# ----------------------------------------------------------------------------- CSR map
def test_a2_csr_offsets_match_rtl():
    """nanoxl/driver/csr.py offsets == rtl/csr_axil.sv localparams (every A_* register), NSAT == 5 in
    the driver, nanoxl_top and the SAT window; the error codes follow cmd_proc."""
    from nanoxl.driver import csr as R
    rtl = {k[2:]: int(v, 16) for k, v in re.findall(r"(A_\w+)\s*=\s*12'h([0-9A-Fa-f]+)", _src("rtl/csr_axil.sv"))}
    alias = {"PERF_BUSY0": "PERF_BUSY_DMA", "SAT0": "SAT_BASE"}
    assert len(rtl) == 26
    for k, v in rtl.items():
        assert getattr(R, alias.get(k, k)) == v, k
    assert R.PERF_BUSY_ARR == rtl["PERF_BUSY0"] + 4 and R.PERF_BUSY_VEC == rtl["PERF_BUSY0"] + 8
    assert "localparam int NSAT = 5;" in _src("rtl/nanoxl_top.sv") and R.NSAT == 5
    proc = _src("rtl/cmd_proc.sv")
    for name, code in (("ERR_FETCH", 4), ("ERR_DMA", 5), ("ERR_RING", 6), ("ERR_ENGINE", 7)):
        assert re.search(rf"{name} = 4'd{code}", proc) and getattr(R, name) == code


# ----------------------------------------------------------------------------- ADR-005 claims
def test_a2_adr005_compiled_programs_need_no_moved_loads():
    """ADR-005 §5: in the compiled tiny and shakespeare programs no LOAD moves channel and no DMA->DMA
    semaphore is needed; both channels carry loads; check_hazards accepts every program."""
    from nanoxl import isa
    from nanoxl.compiler import SEM, Compiler
    from nanoxl.funcsim import check_hazards
    from nanoxl.graph import SHAKESPEARE, TINY, build_gpt_graph, make_synthetic_params
    from nanoxl.hwconfig import KV260
    for cfg in (TINY, SHAKESPEARE):
        g = build_gpt_graph(cfg)
        comp = Compiler(g, make_synthetic_params(g, 0), KV260)
        for seqs in ([(5, 1)], [(0, 16)]):
            cmds = comp.compile_step(seqs).cmds
            check_hazards(cmds, comp.img.base)
            assert comp.sb.moved == 0
            assert not any(c.op == "WAIT" and c["sem"] == SEM[(isa.Q_DMA, isa.Q_DMA)] for c in cmds)
            ch = [0, 0]
            for c in cmds:
                if c.op == "DMA_LOAD":
                    ch[c["chan"] & 1] += 1
            assert min(ch) > 0


# ----------------------------------------------------------------------------- test tiers (A2-01 .. A2-03)
def _collect(paths, env_sim="icarus", marks="not slow"):
    env = dict(os.environ, SIM=env_sim)
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                        "-m", marks] + paths, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    return [ln.strip() for ln in r.stdout.splitlines() if "::" in ln]


def _make_cmd(target):
    """The pytest selection a Makefile target runs: (SIM value, pytest args after `-m pytest`)."""
    r = subprocess.run(["make", "-n", target], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if "pytest" in ln)
    sim = re.search(r"SIM=(\w+)", line)
    import shlex
    args = shlex.split(line.split("-m pytest", 1)[1])
    return (sim.group(1) if sim else "icarus"), args


def _ci():
    import yaml
    return yaml.safe_load(_src(".github/workflows/ci.yml"))


def _verilator_fast_ids():
    sim, args = _make_cmd("test-verilator-fast")
    assert sim == "verilator"
    paths = [a for a in args if a.startswith("tests/")]
    marks = args[args.index("-m") + 1] if "-m" in args else ""
    return _collect(paths, env_sim="verilator", marks=marks)


# Fixed by the AUDIT-2 fix branch: these were xfail(strict) repros; they now pin the fix.
def test_a2_01_ci_tier_runs_a_wide_vec_engine():
    """A2-01 (fixed): the per-push CI job verilator-fast (make test-verilator-fast) collects the
    L >= 16 VEC engine benches (multi-chunk beat assembly, shared SFU lanes, 2-stage reduce tree)."""
    ci = _ci()
    on = ci.get("on", ci.get(True))
    assert "push" in on and "pull_request" in on
    steps = ci["jobs"]["verilator-fast"]["steps"]
    assert any(st.get("run") == "make test-verilator-fast" for st in steps)
    assert "if" not in ci["jobs"]["verilator-fast"]
    ids = _verilator_fast_ids()
    assert ids, "collection failed"
    assert [i for i in ids if re.search(r"test_vec_seq\[16-1\]", i)], ids
    assert [i for i in ids if re.search(r"test_vec_seq\[16-4\]", i)], ids


def test_a2_01_ci_runs_top_saturation_and_oreg():
    """A2-01 (fixed): test_top_smoke_saturation / _oreg / _pbuf_ch1 and test_top_smoke (with its decode
    step) are in the Verilator per-push selection and are not skipped there (setup only, no sim)."""
    ids = _verilator_fast_ids()
    want = ["test_top_smoke", "test_top_smoke_saturation", "test_top_smoke_oreg", "test_top_smoke_pbuf_ch1"]
    for w in want:
        assert any(i.endswith("::" + w) for i in ids), (w, ids)
    env = dict(os.environ, SIM="verilator")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "--setup-only", "-p", "no:cacheprovider",
                        "-m", "not slow", "tests/rtl/test_top.py", "-k", "smoke"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "skipped" not in r.stdout, r.stdout[-400:]
    # the decode step runs on Verilator (TOP_DECODE=1 in every smoke test there)
    top = _src("tests/rtl/test_top.py")
    assert 'TOP_DECODE=int(SIM == "verilator")' in top
    # docstrings no longer call a Verilator-only test plain "Fast tier"
    for m in re.finditer(r"@VERILATOR_ONLY\ndef (\w+)\(\w*\):\n    \"\"\"(.{0,40})", top):
        assert not m.group(2).startswith("Fast tier"), m.group(1)


def test_a2_02_ci_runs_verilator_or_slow_tier():
    """A2-02 (fixed): CI builds Verilator v5.040 once (actions/cache keyed on the version) and runs the
    full slow tier on Verilator nightly and on workflow_dispatch; the Icarus job stays."""
    ci = _ci()
    on = ci.get("on", ci.get(True))
    assert "schedule" in on and "workflow_dispatch" in on
    assert ci["env"]["VERILATOR_VERSION"] == "v5.040"
    jobs = ci["jobs"]
    assert any(st.get("run") == "make test-fast SIM=icarus" for st in jobs["fast"]["steps"])
    build = jobs["verilator-build"]["steps"]
    cache = next(st for st in build if str(st.get("uses", "")).startswith("actions/cache@"))
    assert "VERILATOR_VERSION" in cache["with"]["key"]
    assert any("cache-hit" in str(st.get("if", "")) for st in build)
    slow = jobs["slow"]
    assert "schedule" in slow["if"] and "workflow_dispatch" in slow["if"]
    assert any(st.get("run") == "make test-slow SIM=verilator" for st in slow["steps"])
    sim, args = _make_cmd("test-slow")
    assert "slow" in args
    ids = _collect(["tests/rtl/test_top.py"], env_sim="verilator", marks="slow")
    assert sum("test_top_tiny[" in i for i in ids) == 4, ids


def test_a2_03_dma_ch1_pbuf_test_in_ci():
    """A2-03 (fixed): test_top_smoke_pbuf_ch1 puts every parameter-block load on DMA channel 1, racing a
    channel-0 PBUF load (kills top_snoop_ch1 / top_ch1_gate); it is in the per-push Verilator job."""
    assert any(i.endswith("::test_top_smoke_pbuf_ch1") for i in _verilator_fast_ids())
    assert 'TOP_PBUF_CHAN="ch1"' in _src("tests/rtl/test_top.py")


def test_a2_04_vec_reduce_tested_at_three_tree_stages():
    ids = _collect(["tests/rtl/test_vec_reduce.py"], marks="not slow")
    assert ids, "collection failed"
    assert any(re.search(r"test_vec_reduce\[32\]", i) for i in ids), ids
    assert any(re.search(r"test_vec_reduce\[64\]", i) for i in ids), ids


def test_a2_05_vec_seq_wr_wide_check():
    """A2-05 (addressed by a check): vec_seq_wr carries a simulation-only check that a wide beat's
    emitted chunks agree with the walker's row end ($fatal), and test_vec_seq_wr has a directed case
    in which it fires (plus legal, interior-masked and ragged cases in which it must not)."""
    src = _src("rtl/vec_seq_wr.sv")
    assert "`ifndef SYNTHESIS" in src and "$fatal" in src and "w_rll != (h_last && hm == hn)" in src
    ids = _collect(["tests/rtl/test_vec_seq_wr.py"], marks="not slow")
    assert any("violate" in i for i in ids) and any("interior" in i for i in ids), ids


# ----------------------------------------------------------------------------- docs (A2-0x)
def test_a2_06_plan_status_current():
    plan = _src("PLAN.md")
    row = next(ln for ln in plan.splitlines() if ln.startswith("| SG11c "))
    assert row.rstrip(" |").split("|")[-1].strip().startswith("done")
    assert "Unmerged, pushed as WIP branches" not in plan


def test_a2_06_top_md_headline_table_is_current():
    sec = _src("docs/top.md")
    sec = sec[sec.index("## 6."):sec.index("### 6.1")]
    assert "80,098 / 81,226" not in sec


# ----------------------------------------------------------------------------- mutation list
PP = "tests/rtl/test_pe.py::test_pe_pair_packing tests/rtl/test_array.py::test_array_core[1-4]"
SP = "tests/rtl/test_top_spad.py"
CQ = "tests/rtl/test_ctrl_formal.py -k queue"      # cocotb-only re-run: SEL='tests/rtl/test_ctrl.py::test_ctrl_top'
CQS = "tests/rtl/test_ctrl.py::test_ctrl_top"
VS = "tests/rtl/test_vec_seq.py::test_vec_seq[4-1] tests/rtl/test_vec_seq.py::test_vec_seq[4-4]"
SM = "tests/rtl/test_softmax.py::test_softmax_rq_shared"
VR = "tests/rtl/test_vec_reduce.py"
VA = "tests/rtl/test_vec_alu.py"
LR = "tests/rtl/test_sfu_elem.py tests/rtl/test_softmax.py::test_softmax[4] tests/rtl/test_layernorm.py::test_layernorm[4]"
HZ = "tests/py/test_dma_channels.py"
TOP = "tests/rtl/test_top.py::test_top_smoke"
# (file, exact old text, replacement, default pytest selector); KILLED = the selector failed
MUTS = {
 # pe_pair (WP486)
 "pp_nocorr":  ("rtl/pe.sv", "assign hi  = p[35:18] + {17'd0, p[17]};", "assign hi  = p[35:18];", PP),
 "pp_lo_zext": ("rtl/pe.sv", "psum_out_b <= psum_in_b + {{(ACCW-18){lo_r[17]}}, lo_r};", "psum_out_b <= psum_in_b + {{(ACCW-18){1'b0}}, lo_r};", PP),
 "pp_wb_zext": ("rtl/pe.sv", "{{19{wb[7]}}, wb};", "{19'd0, wb};", PP),
 "pp_hi_narrow": ("rtl/pe.sv", "psum_out_a <= psum_in_a + {{(ACCW-18){hi[17]}}, hi};", "psum_out_a <= psum_in_a + {{(ACCW-17){hi[16]}}, hi[16:0]};", PP),
 # cmd_queue / cmd_proc DMA channel ordering
 "cq_fence_noall": ("rtl/cmd_queue.sv", "(h_fence ? all_zero : !fence_out);", "(h_fence ? 1'b1 : !fence_out);", CQ),
 "cq_behind_fence": ("rtl/cmd_queue.sv", "(h_fence ? all_zero : !fence_out);", "(h_fence ? all_zero : 1'b1);", CQ),
 "cq_sig_ch0": ("rtl/cmd_queue.sv", "for (int c = 0; c < NCH; c++) p_head_ok = p_head_ok && p_left[32'(p_rd) * NCH + c] == '0;", "p_head_ok = p_left[32'(p_rd) * NCH] == '0;", CQ),
 "cq_byp_noidle": ("rtl/cmd_queue.sv", "assign byp     = h_sig && p_cnt == '0 && all_zero;", "assign byp     = h_sig && p_cnt == '0;", CQ),
 "cq_left_nodn": ("rtl/cmd_queue.sv", "p_left[32'(p_wr) * NCH + c] <= outst[c] - OB'(dn[c]);", "p_left[32'(p_wr) * NCH + c] <= outst[c];", CQ),
 "proc_fence_op": ("rtl/cmd_proc.sv", "(in_word[5:0] == 6'h09);", "(in_word[5:0] == 6'h0A);", CQS),
 # buf_spad crossbar
 "bs_norot":   ("rtl/buf_spad.sv", "rr <= (32'(rr) == NR - 1) ? '0 : rr + 1'b1;", "rr <= '0;", SP),
 "bs_wnorot":  ("rtl/buf_spad.sv", "wr <= (32'(wr) == NW - 1) ? '0 : wr + 1'b1;", "wr <= '0;", SP),
 "bs_row_le":  ("rtl/buf_spad.sv", "ad = ad | ({DW{s}} & ((LB > 0 && 32'(b) < 32'(rs[p])) ? rr1[p] : rr0[p]));", "ad = ad | ({DW{s}} & ((LB > 0 && 32'(b) <= 32'(rs[p])) ? rr1[p] : rr0[p]));", SP),
 "bs_oreg_sh": ("rtl/buf_spad.sv", "assign sh = (OREG != 0) ? ds1[p] : ds0[p];", "assign sh = ds0[p];", SP),
 "bs_wconf":   ("rtl/buf_spad.sv", "wc[p][q] = (p != q) && ((wm[p] & wm[q]) != '0);", "wc[p][q] = (p != q) && ((wm[p] & wm[q] & 1) != '0);", SP),
 # vec_seq_rd barrel unpack
 "rd_su_noebl": ("rtl/vec_seq_rd.sv", "assign su    = UW'({9'd0, sub} << ebl);", "assign su    = UW'({9'd0, sub});", VS),
 "rd_full_k1": ("rtl/vec_seq_rd.sv", "full       <= (k_eff + 8'd1 == 8'(1 << lpb_l2)) || (lines_left == 14'd1);", "full       <= 1'b1;", VS),
 "rd_lastslice": ("rtl/vec_seq_rd.sv", "assign last_slice = (sub == 7'((1 << bpl_l2) - 1)) || (beats_left == 16'd1);", "assign last_slice = (sub == 7'((1 << bpl_l2) - 1));", VS),
 "rd_shr_half": ("rtl/vec_seq_rd.sv", "if (u[i]) x = x >> ((1 << i) * L * 8);", "if (u[i]) x = x >> ((1 << i) * L * 4);", VS),
 # lut_rom
 "lr_noen":   ("rtl/lut_rom.sv", "if (en) rdata <= ent[raddr[SW-1:0]];", "rdata <= ent[raddr[SW-1:0]];", LR),
 "lr_endian": ("rtl/lut_rom.sv", "assign ent[k] = line_a[k*DW +: DW];", "assign ent[k] = line_a[(WN-1-k)*DW +: DW];", LR),
 # softmax RQL reassembly
 "rq_oc0":    ("rtl/softmax.sv", "assign rqw_fire = rq_out_valid && (oc == RW'(R - 1));", "assign rqw_fire = rq_out_valid && (oc == RW'(0));", SM),
 "rq_h_nols": ("rtl/softmax.sv", "h_acc   <= acc_sh[L*32-1:RQL*32];", "h_acc   <= in_data[L*32-1:RQL*32];", SM),
 "rq_h_keep": ("rtl/softmax.sv", "h_keep  <= in_keep[L-1:RQL];", "h_keep  <= '1;", SM),
 "rq_hbias_live": ("rtl/softmax.sv", "rq_bias[j*32 +: 32] = h_bias;", "rq_bias[j*32 +: 32] = r_bias[sl];", SM),
 # vec_reduce pipeline alignment
 "vr_v_early": ("rtl/vec_reduce.sv", "assign v2    = vt[NTS-1];", "assign v2    = vt[0];", VR),
 "vr_last_early": ("rtl/vec_reduce.sv", "assign last2 = lt[NTS-1];", "assign last2 = lt[0];", VR),
 "vr_nts32": ("rtl/vec_reduce.sv", "localparam int NTS = (LV <= TLV) ? 1 : (LV + TLV - 1) / TLV;", "localparam int NTS = (LV <= TLV) ? 1 : (LV == 5) ? 2 : (LV + TLV - 1) / TLV;", VR),
 # vec_alu multi-cycle MUL
 "va_rnd_nosticky": ("rtl/vec_alu.sv", "assign mrnd = mrb && (!mP[63] || mst);", "assign mrnd = mrb;", VA),
 "va_push_noadv": ("rtl/vec_alu.sv", "assign push     = (ms == 3'd6) && (mcnt == 5'd0) && adv;", "assign push     = (ms == 3'd6) && (mcnt == 5'd0);", VA),
 "va_mcl33": ("rtl/vec_alu.sv", "assign mcl  = (mP[63:33] == {31{mP[63]}}) ? mP[33:0]", "assign mcl  = (mP[63:32] == {32{mP[63]}}) ? mP[33:0]", VA),
 # check_hazards (python)
 "hz_store_after": ("nanoxl/funcsim.py", "            vc_q[3 - a] = vc_q[a].copy()\n", "            pass\n", "tests/py"),
 "hz_store_before": ("nanoxl/funcsim.py", "                vc_q[a] = np.maximum(vc_q[0], vc_q[3])\n", "                pass\n", HZ),
 "hz_sig_own": ("nanoxl/funcsim.py", "lst.append(np.max(np.stack([vc_q[a] for a in agents(q)]), axis=0))", "lst.append(vc_q[agents(q)[0]].copy())", HZ),
 "hz_wait_ch0": ("nanoxl/funcsim.py", "                for a in agents(q):\n                    vc_q[a] = np.maximum(vc_q[a], v)", "                for a in agents(q)[:1]:\n                    vc_q[a] = np.maximum(vc_q[a], v)", HZ),
 "top_sat_swap": ("rtl/nanoxl_top.sv", "assign sat_cnt = vec_sat;", "assign sat_cnt = {vec_sat[159:128], vec_sat[95:64], vec_sat[127:96], vec_sat[63:0]};", TOP),
 "top_snoop_ch1": ("rtl/nanoxl_top.sv", "assign pb_snoop_valid = (dma0_pb && sp_wready) || (sp1_wvalid && sp1_wready && sp1_wbuf == 3'd3);", "assign pb_snoop_valid = (dma0_pb && sp_wready);", TOP),
 "top_ch1_gate": ("rtl/nanoxl_top.sv", "assign sp1_wvalid     = sp1_wv_raw && !(dma0_pb && sp1_wbuf == 3'd3);", "assign sp1_wvalid     = sp1_wv_raw;", TOP),
 "dma_st_behind": ("rtl/dma_engine.sv", "(rq_count == '0 || (!f_is_store && !st));", "(rq_count == '0 || (!f_is_store));", "tests/rtl/test_dma.py tests/rtl/test_ctrl.py::test_ctrl_top"),
 "proc_ch0": ("rtl/cmd_proc.sv", "assign q_ch    = (d_kind == K_ENGINE) && in_word[6];", "assign q_ch    = 1'b0;", CQS),
 "hz_onechan": ("nanoxl/funcsim.py", "            a = 3 if c[\"chan\"] & 1 else 0\n", "            a = 0\n", HZ),
}


def _run_mutant(root: Path, mid: str) -> dict:
    import json, time
    f, old, new, test = MUTS[mid]
    test = os.environ.get("SEL", test)
    p = root / f
    src = p.read_text()
    assert src.count(old) == 1, (mid, src.count(old))
    p.write_text(src.replace(old, new))
    t0 = time.time()
    try:
        cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-m", "not slow"] + test.split()
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=3000)
        res = "KILLED" if r.returncode != 0 else "SURVIVED"
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:]
    except subprocess.TimeoutExpired:
        res, tail = "TIMEOUT", []
    finally:
        p.write_text(src)
    return dict(id=mid, result=res, secs=round(time.time() - t0), sim=os.environ.get("SIM", "icarus"), sel=test, tail=tail)


if __name__ == "__main__":
    import json
    target = Path(sys.argv[1]).resolve()
    assert target != ROOT, "run the mutants on a scratch copy, not the working tree"
    for mid in sys.argv[2:] or list(MUTS):
        print(json.dumps(_run_mutant(target, mid)), flush=True)
