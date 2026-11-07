"""GELU LUT (C5): exhaustive over the INT8 input domain for every per-layer table of the
tiny bundle, through rtl/sfu_elem.sv (the GELU lanes live there)."""
import pytest
from simrun import run

from test_sfu_elem import SRC


@pytest.mark.parametrize("L", [1, 4])
def test_gelu_exhaustive(L):
    run("sfu_elem", SRC, "sfu_elem_cocotb", parameters={"L": L, "NG": 4},
        extra_env={"L": str(L), "NG": "4", "COCOTB_TEST_FILTER": "elem_gelu_exhaustive"})
