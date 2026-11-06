"""pytest wrappers: rtl/requant.sv vs contract C2 at L = 1, 4, 16."""
import pytest
from simrun import run

SRC = ["rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/requant.sv"]


@pytest.mark.parametrize("L", [1, 4, 16])
def test_requant(L):
    run("requant", SRC, "requant_cocotb", parameters={"L": L}, extra_env={"L": str(L)})
