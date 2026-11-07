"""pytest wrappers: rtl/layernorm.sv vs nanoxl.quant.ops.ln_norm_i / ln_affine_i (C5)."""
import os

import pytest
from simrun import run

SRC = ["rtl/lut_rom.sv", "rtl/sfu_rsqrt.sv", "rtl/srdhm.sv", "rtl/sat_counter.sv",
       "rtl/layernorm.sv"]
VERILATOR = os.environ.get("SIM", "icarus") == "verilator"


@pytest.mark.parametrize("L", [1, 4, 16])
def test_layernorm(L):
    if VERILATOR:
        rows, gold = {1: 300, 4: 1500, 16: 2500}[L], 0       # 0 = every golden LN row
    else:
        rows, gold = {1: 40, 4: 250, 16: 900}[L], {1: 30, 4: 200, 16: 0}[L]
    run("layernorm", SRC, "layernorm_cocotb",
        parameters={"L": L, "D_MAX": 384, "NSLOT": 16},
        extra_env={"L": str(L), "D_MAX": "384", "NSLOT": "16", "N_ROWS": str(rows),
                   "N_GOLDEN": str(gold)})
