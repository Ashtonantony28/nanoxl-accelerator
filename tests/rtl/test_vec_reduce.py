"""pytest wrappers: rtl/vec_reduce.sv vs nanoxl.vec_ref.reduce_row."""
import pytest
from simrun import run

SRC = ["rtl/vec_reduce.sv"]


@pytest.mark.parametrize("L", [1, 3, 4, 8, 16])
def test_vec_reduce(L):
    rows = {1: 60, 3: 150, 4: 200, 8: 200, 16: 400}[L]
    run("vec_reduce", SRC, "vec_reduce_cocotb", parameters={"L": L},
        extra_env={"L": str(L), "N_ROWS": str(rows)})
