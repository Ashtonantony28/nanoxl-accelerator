"""pytest wrappers: rtl/requant.sv vs contract C2 at L = 1, 4, 16 (both srdhm DSP decompositions);
rtl/srdhm.sv alone vs nanoxl.numerics.srdhm."""
import pytest
from simrun import run

SRC = ["rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/requant.sv"]


@pytest.mark.parametrize("L,ndsp", [(1, 4), (4, 4), (16, 4), (4, 3), (16, 3)])
def test_requant(L, ndsp):
    run("requant", SRC, "requant_cocotb", parameters={"L": L, "NDSP": ndsp}, extra_env={"L": str(L)})


@pytest.mark.parametrize("ndsp", [4, 3])
def test_srdhm(ndsp):
    run("srdhm", ["rtl/srdhm.sv"], "srdhm_cocotb", parameters={"NDSP": ndsp},
        extra_env={"N_SAMPLES": "60000"})
