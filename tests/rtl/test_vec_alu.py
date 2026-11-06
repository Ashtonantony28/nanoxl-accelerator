"""pytest wrappers: rtl/vec_alu.sv vs nanoxl.vec_ref.alu."""
import pytest
from simrun import run

SRC = ["rtl/sat_counter.sv", "rtl/vec_alu.sv"]


@pytest.mark.parametrize("L", [1, 4, 16])
def test_vec_alu(L):
    # ~20k random lane-results per op at L=16; fewer beats-per-op at small L keeps runtime flat.
    n = {1: 3000, 4: 8000, 16: 20000}[L]
    run("vec_alu", SRC, "vec_alu_cocotb", parameters={"L": L},
        extra_env={"L": str(L), "N_PER_OP": str(n)})
