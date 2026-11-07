import os

import pytest

from simrun import run

CORE = ["rtl/pe.sv", "rtl/skew.sv", "rtl/systolic_array.sv"]
CTRL = CORE + ["rtl/scratchpad.sv", "rtl/array_ctrl.sv"]


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("pack", [0, 1])
def test_array_core(n, pack):
    """pack=1: WP486 2-MAC pe_pair grid (the default); pack=0: one pe per DSP."""
    run("systolic_array", CORE, "array_cocotb", parameters={"N": n, "TAG_W": 8, "PACK": pack},
        extra_env={"ARR_N": str(n)})


_ICARUS = os.environ.get("SIM", "icarus") != "verilator"
# N=16 takes ~110 s on Icarus: fast tier on Verilator, slow tier on Icarus (AUDIT-1 A1-10)
_SLOW_ON_ICARUS = (pytest.mark.slow,) if _ICARUS else ()


@pytest.mark.parametrize("n", [4, 8, pytest.param(16, marks=_SLOW_ON_ICARUS)])
def test_array_ctrl(n):
    run("array_ctrl", CTRL, "array_ctrl_cocotb", parameters={"N": n, "M_DEPTH": 64},
        extra_env={"ARR_N": str(n), "ARR_MDEPTH": "64"})


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("SIM", "icarus") != "verilator",
                    reason="N=32 (1024 PEs) is impractically slow on Icarus; Verilator only")
def test_array_ctrl_kv260_size():
    """Default KV260 configuration N=32 with a deeper accumulator; longer random run
    (all controller tests, ~2 min on Verilator)."""
    run("array_ctrl", CTRL, "array_ctrl_cocotb", parameters={"N": 32, "M_DEPTH": 256},
        extra_env={"ARR_N": "32", "ARR_MDEPTH": "256", "ARR_NJOBS": "60"})
