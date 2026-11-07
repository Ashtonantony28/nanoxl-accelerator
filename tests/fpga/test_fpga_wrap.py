"""rtl/nanoxl_axi_wrap.v behaves exactly like nanoxl_top: the top-level smoke (tests/rtl/test_top.py
test_top_smoke: 1-layer tiny W8A8, 2-token prompt, N=8, L=4, real driver over AXI, bit-exact vs golden)
runs through the wrapper's Vivado port names. cocotb module: wrap_cocotb.py."""
import os

from simrun import run
from test_top import TOP_SRC

SIM = os.environ.get("SIM", "icarus")


def test_wrap_smoke():
    """Same configuration and checks as test_top_smoke (Icarus: prefill only, ~90 s; Verilator: + 1 decode)."""
    env = {"TOP_N": "8", "TOP_L": "4", "TOP_VP": "4", "TOP_BUNDLE": "w8a8", "TOP_LAYERS": "1", "TOP_PROMPT": "2",
           "TOP_DECODE": str(int(SIM == "verilator")), "TOP_PBUF_KB": "10"}
    run("nanoxl_axi_wrap", TOP_SRC + ["rtl/nanoxl_axi_wrap.v"], "wrap_cocotb", parameters={"N": 8, "L": 4, "VP": 4},
        extra_env=env, tag="_smoke")
