"""pytest wrappers: rtl/sfu_vec.sv (vec_unit + rtl/sfu.sv on the SFU hook)."""
import pytest
from simrun import run

SRC = ["rtl/lut_rom.sv", "rtl/sfu_exp2.sv", "rtl/sfu_recip.sv", "rtl/sfu_rsqrt.sv",
       "rtl/sfu_elem.sv", "rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/softmax.sv",
       "rtl/layernorm.sv", "rtl/sfu.sv", "rtl/requant.sv", "rtl/vec_alu.sv",
       "rtl/vec_reduce.sv", "rtl/vec_unit.sv", "rtl/sfu_vec.sv"]


@pytest.mark.parametrize("L", [1, 4, 16])
def test_sfu_vec(L):
    n = {1: 90, 4: 240, 16: 300}[L]
    run("sfu_vec", SRC, "sfu_vec_cocotb", parameters={"L": L, "NSLOT": 16},
        extra_env={"L": str(L), "N_CMDS": str(n)})
