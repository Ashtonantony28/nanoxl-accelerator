"""Physical scratchpad of nanoxl_top (rtl/buf_spad.sv + rtl/spad_bank.sv): banking by line, rotating
port arbitration, multi-line beats, write-visible-to-later-read. cocotb module: top_spad_cocotb.py."""
import pytest

from simrun import run

SRC = ["rtl/spad_bank.sv", "rtl/buf_spad.sv"]


def _pack(pl):
    """Sized Verilog literal: a plain integer -G/-P value is truncated to 32 bits by Verilator."""
    return f"{8 * len(pl)}'h{sum(v << (8 * i) for i, v in enumerate(pl)):x}"


@pytest.mark.parametrize("lines,nb,rpl,wpl,oreg", [
    (64, 4, [1, 2, 4, 1], [1, 4, 2], 0),               # widest beats occupy every bank
    (128, 8, [1, 1, 2, 4, 1, 1, 1], [1, 4, 1], 0),     # nanoxl_top N=16, VP=1 port set (NB = 2 x widest)
    (128, 8, [1, 1, 1, 4, 4, 4, 4], [1, 4, 4], 0),     # nanoxl_top N=16, VP=4: 64-byte VEC A/B/P/D ports
    (64, 8, [1, 1, 1, 2, 4, 4, 4], [1, 2, 4], 0),      # nanoxl_top N=8, VP=4 (smoke config)
    (32, 2, [1, 1, 1], [1, 1], 0),                     # N <= 4: all ports one line
    (128, 8, [1, 2, 2, 8, 4, 4, 4, 1], [1, 8, 4, 1], 0),    # KV260 port set (N=32, VP=4, NB=8 as built)
    (256, 16, [1, 2, 2, 8, 4, 4, 4, 1], [1, 8, 4, 1], 0),   # N=32 with the automatic NB = 16
    (64, 4, [1, 2, 4, 1], [1, 4, 2], 1),               # bank output register: read latency 2
    (128, 8, [1, 2, 2, 8, 4, 4, 4, 1], [1, 8, 4, 1], 1),
    (16, 1, [1, 1], [1], 1),                           # one bank
])
def test_top_spad(lines, nb, rpl, wpl, oreg):
    env = {"SP_LINES": str(lines), "SP_NB": str(nb), "SP_RPL": ",".join(map(str, rpl)),
           "SP_WPL": ",".join(map(str, wpl)), "SP_CYCLES": "3000", "SP_OREG": str(oreg)}
    run("buf_spad", SRC, "top_spad_cocotb",
        parameters={"LINES": lines, "NB": nb, "NR": len(rpl), "NW": len(wpl), "PMAX": max(rpl + wpl),
                    "RPL": _pack(rpl), "WPL": _pack(wpl), "OREG": oreg},
        extra_env=env, tag=f"_{len(rpl)}r{len(wpl)}w")
