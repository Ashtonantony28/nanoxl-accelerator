import pytest

from simrun import run

CORE = ["rtl/pe.sv", "rtl/skew.sv", "rtl/systolic_array.sv"]
CTRL = CORE + ["rtl/scratchpad.sv", "rtl/array_ctrl.sv"]


@pytest.mark.parametrize("n", [4, 8])
def test_array_core(n):
    run("systolic_array", CORE, "array_cocotb", parameters={"N": n, "TAG_W": 8},
        extra_env={"ARR_N": str(n)})


@pytest.mark.parametrize("n", [4, 8, 16])
def test_array_ctrl(n):
    run("array_ctrl", CTRL, "array_ctrl_cocotb", parameters={"N": n, "M_DEPTH": 64},
        extra_env={"ARR_N": str(n), "ARR_MDEPTH": "64"})


@pytest.mark.slow
def test_array_ctrl_kv260_size():
    """Default KV260 configuration N=32 with a deeper accumulator; longer random run."""
    run("array_ctrl", CTRL, "array_ctrl_cocotb", parameters={"N": 32, "M_DEPTH": 256},
        extra_env={"ARR_N": "32", "ARR_MDEPTH": "256", "ARR_NJOBS": "60"})
