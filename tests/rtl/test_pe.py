import os

from simrun import run


def test_pe_exhaustive():
    run("pe", ["rtl/pe.sv"], "pe_cocotb")


def test_pe_pair_packing():
    """WP486 2-MAC packing (pe_pair) against plain products: all 2^24 int8 a x wA x wB on
    Verilator (~1 min); every 13th case (1.3 M) on Icarus."""
    step = 1 if os.environ.get("SIM", "icarus") == "verilator" else 13
    run("tb_pe_pair", ["rtl/pe.sv", "tests/rtl/tb_pe_pair.sv"], "pe_pair_cocotb",
        parameters={"STEP": step}, extra_env={"PP_STEP": str(step)})
