"""pytest wrappers: rtl/softmax.sv vs nanoxl.quant.ops.softmax_i (C5)."""
import os

import pytest
from simrun import run

SRC = ["rtl/lut_rom.sv", "rtl/sfu_exp2.sv", "rtl/sfu_recip.sv", "rtl/softmax.sv"]
VERILATOR = os.environ.get("SIM", "icarus") == "verilator"


@pytest.mark.parametrize("L", [1, 4, 16])
def test_softmax(L):
    if VERILATOR:
        rows, gold = {1: 600, 4: 2000, 16: 3000}[L], 0      # 0 = every golden score row
    else:
        rows, gold = {1: 80, 4: 400, 16: 1200}[L], {1: 60, 4: 300, 16: 600}[L]
    run("softmax", SRC, "softmax_cocotb", parameters={"L": L, "CTX_MAX": 256},
        extra_env={"L": str(L), "CTX_MAX": "256", "N_ROWS": str(rows), "N_GOLDEN": str(gold)})
