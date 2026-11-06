"""Shared cocotb runner. SIM env var selects icarus (default) or verilator.

Usage from a pytest file:
    from simrun import run
    def test_foo(): run("foo", ["rtl/foo.sv"], "test_foo_cocotb", parameters={"N": 4})
"""
from __future__ import annotations
import os
from pathlib import Path
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
VERILATOR_BIN = "/opt/verilator/bin"


def run(toplevel: str, sources: list[str], test_module: str, parameters: dict | None = None,
        sim: str | None = None, extra_env: dict | None = None, tag: str = ""):
    sim = sim or os.environ.get("SIM", "icarus")
    if sim == "verilator" and Path(VERILATOR_BIN).exists() and VERILATOR_BIN not in os.environ["PATH"]:
        os.environ["PATH"] = VERILATOR_BIN + ":" + os.environ["PATH"]
    params = parameters or {}
    ptag = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    build_dir = ROOT / "sim_build" / sim / f"{toplevel}_{ptag}{tag}"
    runner = get_runner(sim)
    build_args = []
    if sim == "verilator":
        build_args = ["-Wno-fatal", "--assert"]
    runner.build(sources=[str(ROOT / s) for s in sources], hdl_toplevel=toplevel,
                 parameters=params, build_dir=build_dir, timescale=("1ns", "1ps"),
                 build_args=build_args, always=False)
    runner.test(hdl_toplevel=toplevel, test_module=test_module, build_dir=build_dir,
                test_dir=Path(__file__).parent, extra_env=extra_env or {},
                timescale=("1ns", "1ps"))
