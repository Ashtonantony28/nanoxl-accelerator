"""pytest wrappers: rtl/vec_reduce.sv vs nanoxl.vec_ref.reduce_row."""
import pytest
from simrun import run

SRC = ["rtl/vec_reduce.sv"]


# L = 32 / 64: three tree stages (NTS = 3, AUDIT-2 A2-04; the mutant vr_nts32 fails L = 32)
@pytest.mark.parametrize("L", [1, 3, 4, 8, 16, 32, 64])
def test_vec_reduce(L):
    rows = {1: 60, 3: 150, 4: 200, 8: 200, 16: 400, 32: 300, 64: 200}[L]
    run("vec_reduce", SRC, "vec_reduce_cocotb", parameters={"L": L},
        extra_env={"L": str(L), "N_ROWS": str(rows)})
