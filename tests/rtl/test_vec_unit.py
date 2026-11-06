"""pytest wrappers: rtl/vec_unit.sv (sequencer + requant + vec_alu + vec_reduce + SFU hook)."""
import pytest
from simrun import run

SRC = ["rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/requant.sv", "rtl/vec_alu.sv",
       "rtl/vec_reduce.sv", "rtl/vec_unit.sv"]


@pytest.mark.parametrize("L,has_sfu", [(1, 1), (4, 0), (4, 1), (16, 1)])
def test_vec_unit(L, has_sfu):
    n = {1: 200, 4: 400, 16: 400}[L]
    run("vec_unit", SRC, "vec_unit_cocotb", parameters={"L": L, "HAS_SFU": has_sfu},
        extra_env={"L": str(L), "HAS_SFU": str(has_sfu), "N_CMDS": str(n)})
