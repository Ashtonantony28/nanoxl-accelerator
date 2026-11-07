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


def _top(n, l, tag, **env):
    e = {"TOP_N": str(n), "TOP_L": str(l)}
    e.update({k: str(v) for k, v in env.items()})
    run("nanoxl_top", TOP_SRC, "top_cocotb", parameters={"N": n, "L": l}, extra_env=e, tag=tag)


@VERILATOR_ONLY
def test_top_smoke():
    """Fast tier: one prefill step of a 2-token prompt on a 1-layer cut of tiny W8A8 (N=8, L=4), logits,
    KV cache and per-op saturation counters bit-exact vs golden."""
    _top(8, 4, "_smoke", TOP_BUNDLE="w8a8", TOP_LAYERS=1, TOP_PROMPT=2, TOP_DECODE=0)


@pytest.mark.slow
@VERILATOR_ONLY
@pytest.mark.parametrize("bundle,kv", [("w8a8", "onchip"), ("w4a8", "onchip"), ("w8a8", "ddr")])
def test_top_tiny(bundle, kv, tmp_path):
    """Headline: the 16-token reference prompt prefilled and 8 tokens greedily decoded by the real host
    driver on N=16, L=16; logits every step, KV cache, per-op saturation counters == golden; tokens ==
    golden_ref.json. W8A8 and W4A8 (GEMM_GS) with the on-chip KV tier, W8A8 with the DDR KV mode."""
    _top(16, 16, f"_tiny", TOP_BUNDLE=bundle, TOP_KV=kv, TOP_PROMPT=16, TOP_DECODE=8,
         TOP_PERF_JSON=os.environ.get("TOP_PERF_DIR", str(tmp_path)) + f"/perf_{bundle}_{kv}.json")
