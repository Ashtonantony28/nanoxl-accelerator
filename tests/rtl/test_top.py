"""nanoxl_top (SG7c, docs/top.md): the whole accelerator driven by nanoxl.driver over AXI, compiled tiny
model bit-exact against nanoxl.golden. cocotb module: top_cocotb.py."""
import os

import pytest

from simrun import run

CTRL = ["rtl/fifo_sync.sv", "rtl/cmd_fields.sv", "rtl/cmd_decode.sv", "rtl/cmd_ring_ptr.sv", "rtl/cmd_fetch.sv",
        "rtl/sem_unit.sv", "rtl/cmd_queue.sv", "rtl/cmd_proc.sv", "rtl/dma_realign.sv", "rtl/dma_engine.sv",
        "rtl/csr_axil.sv", "rtl/ctrl_top.sv"]
ARRAY = ["rtl/pe.sv", "rtl/skew.sv", "rtl/systolic_array.sv", "rtl/scratchpad.sv", "rtl/array_ctrl.sv",
         "rtl/array_seq_rd.sv", "rtl/array_seq.sv"]
VEC = ["rtl/lut_rom.sv", "rtl/sfu_exp2.sv", "rtl/sfu_recip.sv", "rtl/sfu_rsqrt.sv", "rtl/sfu_elem.sv", "rtl/srdhm.sv",
       "rtl/sat_counter.sv", "rtl/softmax.sv", "rtl/layernorm.sv", "rtl/sfu.sv", "rtl/requant.sv", "rtl/vec_alu.sv",
       "rtl/vec_reduce.sv", "rtl/vec_unit.sv", "rtl/sfu_vec.sv", "rtl/vec_seq_walk.sv", "rtl/vec_seq_rd.sv",
       "rtl/vec_seq_wr.sv", "rtl/pbuf_loader.sv", "rtl/vec_seq.sv"]
TOP_SRC = CTRL + ARRAY + VEC + ["rtl/spad_bank.sv", "rtl/buf_spad.sv", "rtl/nanoxl_top.sv"]
SIM = os.environ.get("SIM", "icarus")
VERILATOR_ONLY = pytest.mark.skipif(SIM != "verilator", reason="full-model run; Verilator only")


def _top(n, l, tag, vp=4, rtl=None, **env):
    """vp: VEC engine port lines (nanoxl_top VP; 4 = the built default, 1 = the 16-byte ports).
    rtl: further nanoxl_top parameters (e.g. OREG, per-buffer NB)."""
    e = {"TOP_N": str(n), "TOP_L": str(l), "TOP_VP": str(vp)}
    e.update({k: str(v) for k, v in env.items()})
    run("nanoxl_top", TOP_SRC, "top_cocotb", parameters={"N": n, "L": l, "VP": vp, **(rtl or {})}, extra_env=e,
        tag=tag)


def test_top_smoke():
    """Fast tier: prefill of a 2-token prompt (+ one decode step on Verilator, i.e. in the CI job
    verilator-fast; Icarus runs the prefill only, about 90 s) on a 1-layer cut of tiny W8A8 (N=8,
    L=4, 64-byte VEC ports); logits, KV cache and per-op saturation counters bit-exact vs golden. The compiler gets a 10 KiB PBUF ring, so different parameter blocks
    reuse PBUF lines (the residency snoop must invalidate)."""
    _top(8, 4, "_smoke", TOP_BUNDLE="w8a8", TOP_LAYERS=1, TOP_PROMPT=2, TOP_DECODE=int(SIM == "verilator"),
         TOP_PBUF_KB=10)


@VERILATOR_ONLY
def test_top_smoke_pbuf_ch1():
    """Verilator fast tier (CI job verilator-fast; skipped on Icarus). AUDIT-2 A2-03: as test_top_smoke
    (prefill + one decode step) with every parameter-block load on DMA channel 1, each racing a 4 KiB
    channel-0 PBUF load (top_cocotb.pbuf_channels). The 10 KiB PBUF ring makes ch1 loads overwrite
    resident blocks, so the snoop's ch1 input and the ch0-priority PBUF gate in nanoxl_top are both
    needed for a bit-exact result (mutants top_snoop_ch1 / top_ch1_gate fail it)."""
    _top(8, 4, "_smoke", TOP_BUNDLE="w8a8", TOP_LAYERS=1, TOP_PROMPT=2, TOP_DECODE=1,
         TOP_PBUF_KB=10, TOP_PBUF_CHAN="ch1")


@VERILATOR_ONLY
def test_top_smoke_saturation():
    """Verilator fast tier (CI job verilator-fast; skipped on Icarus): as test_top_smoke with lowered
    requant / LN shifts, so the requant, score, ALU (residual) and LN clamps all fire; per-op counts and the SAT_0..4 CSRs must equal golden's."""
    _top(8, 4, "_smoke", TOP_BUNDLE="w8a8", TOP_LAYERS=1, TOP_PROMPT=2, TOP_DECODE=1, TOP_STRESS=1, TOP_PBUF_KB=10)


@VERILATOR_ONLY
def test_top_smoke_oreg():
    """Verilator fast tier (CI job verilator-fast; skipped on Icarus): as test_top_smoke with the
    scratchpad bank output register (OREG = 1, read latency 2 on every port, null buffer included) and per-buffer bank counts (ACC 16 banks, WBUF 4, the rest 8):
    every requester's credit counter and the ACC read-modify-write must tolerate it, bit-exact."""
    _top(8, 4, "_smoke", rtl={"OREG": 1, "ACC_NB": 16, "WBUF_NB": 4, "RAM_STYLE": 2}, TOP_BUNDLE="w8a8",
         TOP_LAYERS=1, TOP_PROMPT=2, TOP_DECODE=1, TOP_PBUF_KB=10)


@pytest.mark.slow
@VERILATOR_ONLY
@pytest.mark.parametrize("bundle,kv", [("w8a8", "onchip"), ("w4a8", "onchip"), ("w8a8", "ddr"), ("w4a8", "ddr")])
def test_top_tiny(bundle, kv, tmp_path):
    """Headline: the 16-token reference prompt prefilled and 8 tokens greedily decoded by the real host
    driver on N=16, L=16; logits every step, KV cache, per-op saturation counters == golden; tokens ==
    golden_ref.json. W8A8 and W4A8 (GEMM_GS), each with the on-chip KV tier and the DDR KV mode; 64-byte
    VEC engine ports (VP = 4, the built default)."""
    _top(16, 16, "_tiny", TOP_BUNDLE=bundle, TOP_KV=kv, TOP_PROMPT=16, TOP_DECODE=8,
         TOP_PERF_JSON=os.environ.get("TOP_PERF_DIR", str(tmp_path)) + f"/perf_{bundle}_{kv}.json")


@pytest.mark.slow
@VERILATOR_ONLY
@pytest.mark.parametrize("bundle", ["w8a8", "w4a8"])
def test_top_tiny_vp1(bundle, tmp_path):
    """As test_top_tiny (on-chip KV) with 16-byte VEC engine ports (VP = 1): still bit-exact; the
    measured prefill/decode cycles are the VP = 1 column of docs/top.md §6."""
    _top(16, 16, "_tiny", vp=1, TOP_BUNDLE=bundle, TOP_KV="onchip", TOP_PROMPT=16, TOP_DECODE=8,
         TOP_PERF_JSON=os.environ.get("TOP_PERF_DIR", str(tmp_path)) + f"/perf_{bundle}_onchip_vp1.json")


@pytest.mark.slow
@VERILATOR_ONLY
def test_top_tiny_oreg(tmp_path):
    """As test_top_tiny (W8A8, on-chip KV) with OREG = 1 (read latency 2): bit-exact; its cycles are the
    OREG column of docs/top.md §3."""
    _top(16, 16, "_tiny", rtl={"OREG": 1}, TOP_BUNDLE="w8a8", TOP_KV="onchip", TOP_PROMPT=16, TOP_DECODE=8,
         TOP_PERF_JSON=os.environ.get("TOP_PERF_DIR", str(tmp_path)) + "/perf_w8a8_onchip_oreg.json")


@pytest.mark.slow
@VERILATOR_ONLY
def test_top_tiny_saturation():
    """Full 4-layer W4A8 with lowered shifts (all clamps fire in every layer), 4-token prompt + 2 decode
    steps, random AXI backpressure on every channel of both masters."""
    _top(16, 16, "_tiny", TOP_BUNDLE="w4a8", TOP_PROMPT=4, TOP_DECODE=2, TOP_STRESS=1, TOP_P_AXI=0.3)
