"""pytest wrappers: rtl/vec_seq.sv, the VEC engine (docs/vector-unit.md §9)."""
import os

import pytest

from simrun import run

SRC = ["rtl/fifo_sync.sv", "rtl/cmd_fields.sv", "rtl/lut_rom.sv", "rtl/sfu_exp2.sv", "rtl/sfu_recip.sv",
       "rtl/sfu_rsqrt.sv", "rtl/sfu_elem.sv", "rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/softmax.sv",
       "rtl/layernorm.sv", "rtl/sfu.sv", "rtl/requant.sv", "rtl/vec_alu.sv", "rtl/vec_reduce.sv",
       "rtl/vec_unit.sv", "rtl/sfu_vec.sv", "rtl/vec_seq_walk.sv", "rtl/vec_seq_rd.sv", "rtl/vec_seq_wr.sv",
       "rtl/pbuf_loader.sv", "rtl/vec_seq.sv"]
VERILATOR = os.environ.get("SIM", "icarus") == "verilator"


@pytest.mark.parametrize("L", [4, 16])
def test_vec_seq(L):
    n = (120 if L == 4 else 80) if VERILATOR else (30 if L == 4 else 20)
    run("vec_seq", SRC, "vec_seq_cocotb", parameters={"L": L},
        extra_env={"L": str(L), "N_CMDS": str(n)})
