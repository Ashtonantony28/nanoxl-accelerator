"""pytest wrappers: rtl/sfu_elem.sv (EXP2 / RECIP / RSQRT / GELU lanes) vs nanoxl.quant.ops."""
import os

import pytest
from simrun import run

SRC = ["rtl/lut_rom.sv", "rtl/sfu_exp2.sv", "rtl/sfu_recip.sv", "rtl/sfu_rsqrt.sv",
       "rtl/sfu_elem.sv"]
VERILATOR = os.environ.get("SIM", "icarus") == "verilator"


def _run(L, filt=None, exhaustive=1, n_rand=2000):
    env = {"L": str(L), "NG": "4", "EXHAUSTIVE": str(exhaustive), "N_RAND": str(n_rand)}
    if filt:
        env["COCOTB_TEST_FILTER"] = filt
    run("sfu_elem", SRC, "sfu_elem_cocotb", parameters={"L": L, "NG": 4}, extra_env=env)


@pytest.mark.parametrize("L", [1, 4, 16])
def test_sfu_elem(L):
    # EXP2 exhaustive (all 65,536 uint16 inputs) and GELU exhaustive at every L.
    _run(L, exhaustive=1, n_rand={1: 2000, 4: 4000, 16: 8000}[L])
