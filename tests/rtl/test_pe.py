from simrun import run


def test_pe_exhaustive():
    run("pe", ["rtl/pe.sv"], "pe_cocotb")
