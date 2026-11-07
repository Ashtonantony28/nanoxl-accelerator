"""ARRAY engine sequencer (rtl/array_seq.sv): ISA GEMM / GEMM_GS words -> array_ctrl tiles,
scratchpad line ports, ACC write-back. cocotb module: array_seq_cocotb.py."""
import os

import pytest

from simrun import run

SRCS = ["rtl/pe.sv", "rtl/skew.sv", "rtl/systolic_array.sv", "rtl/scratchpad.sv", "rtl/array_ctrl.sv",
        "rtl/fifo_sync.sv", "rtl/cmd_fields.sv", "rtl/array_seq_rd.sv", "rtl/array_seq.sv"]
VERILATOR_ONLY = pytest.mark.skipif(os.environ.get("SIM", "icarus") != "verilator",
                                    reason="long run; Verilator only")


def _run(n, mdepth, mode, **env):
    e = {"SEQ_N": str(n), "SEQ_MDEPTH": str(mdepth), "SEQ_MODE": mode}
    e.update({k: str(v) for k, v in env.items()})
    run("array_seq", SRCS, "array_seq_cocotb", parameters={"N": n, "M_DEPTH": mdepth},
        extra_env=e, tag=f"_{mode}")


@pytest.mark.parametrize("n", [4, 8])
def test_array_seq(n):
    """Random commands under backpressure, GEMM_GS groups, illegal opcode, rst_n / clr mid-command;
    N=8 also runs the cheap compiler-emitted commands of tiny W8A8 + W4A8."""
    _run(n, 16, "fast")


@pytest.mark.slow
@VERILATOR_ONLY
def test_array_seq_compiler_all():
    """Every GEMM / GEMM_GS the compiler emits for tiny W8A8 and W4A8 (prefill + decode), N=16."""
    _run(16, 64, "compiler_all")


@pytest.mark.slow
@VERILATOR_ONLY
def test_array_seq_kv260_size():
    """N=32 (KV260): random commands, every distinct compiler shape exact, measured cycles per GEMM
    vs nanoxl.cyclemodel (logged as PERF lines)."""
    _run(32, 64, "kv260", SEQ_NCMDS=40)
