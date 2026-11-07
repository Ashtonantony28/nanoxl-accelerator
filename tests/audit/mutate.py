"""AUDIT-1 mutation runner (docs/audits/AUDIT-1.md §3). Not a test (not collected).

Run it on a scratch COPY of the repo, never on the working tree (it rewrites sources in place and
restores them after each mutant; a kill -9 mid-run leaves the mutant behind):

    cp -r rtl tests nanoxl formal artifacts pyproject.toml /tmp/mut/
    python3 tests/audit/mutate.py /tmp/mut            # all mutants, one JSON line each
    python3 tests/audit/mutate.py /tmp/mut sm_round   # one mutant

Each entry: (file, exact old text, replacement, pytest selector). KILLED = the selector failed.
"""
import subprocess, sys, time, json
from pathlib import Path
ROOT = Path(sys.argv[1])
RQ = "tests/rtl/test_requant.py::test_requant[4]"
MUTS = {
 # requant / srdhm
 "rq_thr_sign":  ("rtl/requant.sv", "thr  = (mask >> 1) + {31'd0, x[31]};", "thr  = (mask >> 1);", RQ),
 "rq_rem_ge":    ("rtl/requant.sv", "((rem > thr) ? 32'sd1", "((rem >= thr) ? 32'sd1", RQ),
 "srdhm_nudge":  ("rtl/srdhm.sv", "-64'sh0000_0000_3FFF_FFFF", "-64'sh0000_0000_4000_0000", RQ),
 "srdhm_ovf":    ("rtl/srdhm.sv", "y    <= ovf2 ? 32'sh7FFF_FFFF : q[31:0];", "y    <= q[31:0];", RQ),
 "rq_satkeep":   ("rtl/requant.sv", "assign out_sat[i] = s6 & keep_p[D-1][i];", "assign out_sat[i] = s6;", RQ),
 "rq_o16_stage": ("rtl/requant.sv", "assign lo = o16_p[D-2]", "assign lo = o16_p[D-1]", RQ),
 # array_ctrl group scales
 "ac_gm_signed": ("rtl/array_ctrl.sv", "$signed({1'b0, gm_row[c*8 +: 8]})", "$signed(gm_row[c*8 +: 8])", "tests/rtl/test_array.py::test_array_ctrl[4]"),
 "ac_gsw_m1":    ("rtl/array_ctrl.sv", "localparam int GSW   = $clog2(N) + 12;", "localparam int GSW   = $clog2(N) + 11;", "tests/rtl/test_array.py::test_array_ctrl[8]"),
 "ac_gm_pop":    ("rtl/array_ctrl.sv", ".out_ready(s1_v && s1_gs && s1_last)", ".out_ready(s1_v && s1_gs)", "tests/rtl/test_array.py::test_array_ctrl[4]"),
 "ac_nofwd":     ("rtl/array_ctrl.sv", "s1_fwd      <= s1_v && (s1_addr == out_addr);", "s1_fwd      <= 1'b0;", "tests/rtl/test_array.py::test_array_ctrl[4]"),
 # array_seq
 "as_skp_every": ("rtl/array_seq.sv", "if (((k0 + kt_full) & gmask) == '0) s_kp <= s_kp + 21'(r_n);", "s_kp <= s_kp + 21'(r_n);", "tests/rtl/test_array_seq.py::test_array_seq[4]"),
 "as_raw_gate":  ("rtl/array_seq.sv", "pf_v && (pq_cseq == done_cnt);", "pf_v;", "tests/rtl/test_array_seq.py::test_array_seq[4]"),
 # softmax
 "sm_causal_le": ("rtl/softmax.sv", "32'(wcnt) * 32'(L) + 32'(j) < 32'(nk)", "32'(wcnt) * 32'(L) + 32'(j) <= 32'(nk)", "tests/rtl/test_softmax.py::test_softmax[4]"),
 "sm_round":     ("rtl/softmax.sv", "(48'd1 << (shv - 6'd1))", "(48'd1 << (shv - 6'd2))", "tests/rtl/test_softmax.py::test_softmax[4]"),
 "sm_sat_all":   ("rtl/softmax.sv", ".in_keep(in_keep), .in_last(in_last), .in_out16(1'b1),", ".in_keep(wr_mask), .in_last(in_last), .in_out16(1'b1),", "tests/rtl/test_softmax.py::test_softmax[4]"),
 # layernorm
 "ln_qmax1":     ("rtl/layernorm.sv", "if (qfin == '0) qfin = 48'd1;", "", "tests/rtl/test_layernorm.py::test_layernorm[4]"),
 "ln_zround":    ("rtl/layernorm.sv", "rndD = prC + (50'sd1 <<< (shz - 6'd1));", "rndD = prC;", "tests/rtl/test_layernorm.py::test_layernorm[4]"),
 # sem_unit
 "sem_usedw":    ("rtl/sem_unit.sv", "(32'(cnt[ws[q]]) >= used_w + 32'(wc[q]));", "(32'(cnt[ws[q]]) >= 32'(wc[q]));", "tests/rtl/test_ctrl.py::test_ctrl_top"),
 "sem_useds":    ("rtl/sem_unit.sv", "(32'(cnt[ss[q]]) + used_s + 1 <= MAXC);", "(32'(cnt[ss[q]]) + 1 <= MAXC);", "tests/rtl/test_ctrl.py::test_ctrl_top"),
 # dma
 "dma_4k_cross": ("rtl/dma_engine.sv", "if (13'(a_to4k) < m) m = 13'(a_to4k);", "if (13'(a_to4k) + 13'd1 < m) m = 13'(a_to4k) + 13'd1;", "tests/rtl/test_dma.py::test_dma_engine[16-32]"),
 # golden ops
 "gold_lnaff_rshr": ("nanoxl/quant/ops.py", "return sat(nx.rdivpot(acc, sh), 8, counter=counter, name=name).astype(np.int8)", "return sat(rshr(acc, sh), 8, counter=counter, name=name).astype(np.int8)", "tests/py -k golden"),
 "gold_ls_drop":  ("nanoxl/golden.py", "s = self._requant(acc << ls,", "s = self._requant(acc,", "tests/py -k golden"),
 "sem_usedw_formal": ("rtl/sem_unit.sv", "(32'(cnt[ws[q]]) >= used_w + 32'(wc[q]));", "(32'(cnt[ws[q]]) >= 32'(wc[q]));", "tests/rtl/test_ctrl_formal.py -k sem"),
 "sem_useds_formal": ("rtl/sem_unit.sv", "(32'(cnt[ss[q]]) + used_s + 1 <= MAXC);", "(32'(cnt[ss[q]]) + 1 <= MAXC);", "tests/rtl/test_ctrl_formal.py -k sem"),
 "ac_wmask_le":  ("rtl/array_ctrl.sv", "if (c < ld_n) begin", "if (c <= ld_n) begin", "tests/rtl/test_array.py::test_array_ctrl[4]"),
 "ln_noeps":     ("rtl/layernorm.sv", "qfin = qacc + r_eps;", "qfin = qacc;", "tests/rtl/test_layernorm.py::test_layernorm[4]"),
 "vs_diag":      ("rtl/vec_seq.sv", "t_diag1    <= 14'(signed'(sm_diag)) + 14'sd1;", "t_diag1    <= 14'(signed'(sm_diag)) + 14'sd2;", "tests/rtl/test_vec_seq.py::test_vec_seq[4]"),
 "gold_recip_nonewton": ("nanoxl/quant/ops.py", "y1 = (y0 * ((I64(1) << 32) - m * y0)) >> 31", "y1 = y0 * 2", "tests/py -k golden"),
}


def run(mid):
    f, old, new, test = MUTS[mid]
    p = ROOT / f
    src = p.read_text()
    assert src.count(old) == 1, (mid, src.count(old))
    p.write_text(src.replace(old, new))
    t0 = time.time()
    try:
        cmd = ["python3", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-m", "not slow"] + test.split()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=2400)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:]
        res = "KILLED" if r.returncode != 0 else "SURVIVED"
    except subprocess.TimeoutExpired:
        res, tail = "TIMEOUT", []
    finally:
        p.write_text(src)
    return dict(id=mid, result=res, secs=round(time.time() - t0), tail=tail)


if __name__ == "__main__":
    ids = sys.argv[2:] or list(MUTS)
    for i in ids:
        print(json.dumps(run(i)), flush=True)
