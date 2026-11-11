"""rtl/nanoxl_axi_wrap.v behaves exactly like nanoxl_top: the top-level smoke (tests/rtl/test_top.py
test_top_smoke: 1-layer tiny W8A8, 2-token prompt, N=8, L=4, real driver over AXI, bit-exact vs golden)
runs through the wrapper's Vivado port names. cocotb module: wrap_cocotb.py. Also (slow, Verilator): the
whole tiny model on nanoxl_top built with exactly the KV260 bitstream's parameters (ADR-006)."""
import os

import pytest
from simrun import run
from test_top import TOP_SRC

from fpga import boards

SIM = os.environ.get("SIM", "icarus")


def test_wrap_smoke():
    """Same configuration and checks as test_top_smoke (Icarus: prefill only, ~90 s; Verilator: + 1 decode)."""
    env = {"TOP_N": "8", "TOP_L": "4", "TOP_VP": "4", "TOP_BUNDLE": "w8a8", "TOP_LAYERS": "1", "TOP_PROMPT": "2",
           "TOP_DECODE": str(int(SIM == "verilator")), "TOP_PBUF_KB": "10"}
    run("nanoxl_axi_wrap", TOP_SRC + ["rtl/nanoxl_axi_wrap.v"], "wrap_cocotb", parameters={"N": 8, "L": 4, "VP": 4},
        extra_env=env, tag="_smoke")


@pytest.mark.slow
@pytest.mark.skipif(SIM != "verilator", reason="full-model run; Verilator only")
@pytest.mark.parametrize("bundle,kv", [("w8a8", "onchip"), ("w4a8", "onchip"), ("w8a8", "ddr"), ("w4a8", "ddr")])
def test_board_rtl_tiny(bundle, kv, tmp_path):
    """ADR-006: nanoxl_top with exactly the KV260 bitstream's parameters (fpga/boards.py RTL_PARAMS: N=8, L=4,
    VP=1, NB=2, the board's buffer sizes, URAM RAM_STYLE) runs test_top_tiny's headline check: the 16-token
    reference prompt + 8 greedy decode steps by the real driver; logits, KV cache, per-op saturation ==
    golden; tokens == golden_ref.json. The compiler uses top_cocotb's (smaller) buffer capacities, which the
    RTL exceeds, as running on the bitstream allows."""
    p = boards.RTL_PARAMS["kv260"]
    env = {"TOP_N": str(p["N"]), "TOP_L": str(p["L"]), "TOP_VP": str(p["VP"]), "TOP_BUNDLE": bundle, "TOP_KV": kv,
           "TOP_PROMPT": "16", "TOP_DECODE": "8", "TOP_PERF_JSON": str(tmp_path / f"perf_{bundle}_{kv}.json")}
    run("nanoxl_top", TOP_SRC, "top_cocotb", parameters=dict(p), extra_env=env, tag="_board_kv260")
