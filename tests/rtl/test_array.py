import pytest

from simrun import run

CORE = ["rtl/pe.sv", "rtl/skew.sv", "rtl/systolic_array.sv"]


@pytest.mark.parametrize("n", [4, 8])
def test_array_core(n):
    run("systolic_array", CORE, "array_cocotb", parameters={"N": n, "TAG_W": 8},
        extra_env={"ARR_N": str(n)})
