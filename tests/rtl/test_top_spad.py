"""Physical scratchpad of nanoxl_top (rtl/buf_spad.sv + rtl/spad_bank.sv): banking by line, rotating
port arbitration, multi-line beats, write-visible-to-later-read. cocotb module: top_spad_cocotb.py."""
import pytest

from simrun import run

SRC = ["rtl/spad_bank.sv", "rtl/buf_spad.sv"]


def _pack(pl):
    return sum(v << (8 * i) for i, v in enumerate(pl))


@pytest.mark.parametrize("lines,nb,rpl,wpl", [
    (64, 4, [1, 2, 4, 1], [1, 4, 2]),                  # widest beats occupy every bank
    (128, 8, [1, 1, 2, 4, 1, 1, 1], [1, 4, 1]),        # nanoxl_top N=16 port set (NB = 2 x widest)
    (32, 2, [1, 1, 1], [1, 1]),                        # N <= 4: all ports one line
])
def test_top_spad(lines, nb, rpl, wpl):
    env = {"SP_LINES": str(lines), "SP_NB": str(nb), "SP_RPL": ",".join(map(str, rpl)),
           "SP_WPL": ",".join(map(str, wpl)), "SP_CYCLES": "3000"}
    run("buf_spad", SRC, "top_spad_cocotb",
        parameters={"LINES": lines, "NB": nb, "NR": len(rpl), "NW": len(wpl), "PMAX": max(rpl + wpl),
                    "RPL": _pack(rpl), "WPL": _pack(wpl)},
        extra_env=env, tag=f"_{len(rpl)}r{len(wpl)}w")
