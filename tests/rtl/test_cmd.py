"""Command decoder RTL (rtl/cmd_fields.sv generated from nanoxl/isa.py, rtl/cmd_decode.sv)."""
import pytest

from simrun import run


@pytest.mark.parametrize("top", ["cmd_fields", "cmd_decode"])
def test_cmd_decoder(top):
    src = ["rtl/cmd_fields.sv"] + (["rtl/cmd_decode.sv"] if top == "cmd_decode" else [])
    run(top, src, "cmd_cocotb", extra_env={"CMD_TOP": top})
