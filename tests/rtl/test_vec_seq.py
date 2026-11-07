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


# L=16 takes ~85 s on Icarus: fast tier on Verilator, slow tier on Icarus (AUDIT-1 A1-10)
_SLOW_ON_ICARUS = () if VERILATOR else (pytest.mark.slow,)


@pytest.mark.parametrize("L", [4, pytest.param(16, marks=_SLOW_ON_ICARUS)])
def test_vec_seq(L):
    """All benches of vec_seq_cocotb: smoke, random (two port regimes), tiny W8A8 prefill + decode,
    errors, clear/reset mid-command, residency."""
    n = (150 if L == 4 else 100) if VERILATOR else (40 if L == 4 else 25)
    run("vec_seq", SRC, "vec_seq_cocotb", parameters={"L": L},
        extra_env={"L": str(L), "N_CMDS": str(n), "TINY_T": "6" if L == 4 else "8"})


@pytest.mark.slow
@pytest.mark.skipif(not VERILATOR, reason="KV260 lane count; Verilator only")
def test_vec_seq_l64():
    """L = 64 (KV260): random commands and a 16-token prefill + decode of the tiny bundle, with the
    'changed' snoop (parameters stay resident across the two programs)."""
    run("vec_seq", SRC, "vec_seq_cocotb", parameters={"L": 64},
        extra_env={"L": "64", "N_CMDS": "60", "TINY_T": "16", "TINY_SNOOP": "changed"}, tag="_l64")
