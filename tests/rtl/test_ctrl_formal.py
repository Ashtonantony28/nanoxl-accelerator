"""Formal proofs (BMC + k-induction with yosys-smtbmc) for the control-plane blocks: the generic
FIFO (no overflow/underflow, order and data integrity), the semaphore unit (counts never negative
or wrapping, no lost signals, progress) and the command-ring pointers (never past tail, no burst
across the ring end or a 4 KiB boundary). Skips cleanly when yosys / yosys-smtbmc / z3 are missing.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "formal"))
import run_formal  # noqa: E402

_missing = run_formal.tools_available()


@pytest.mark.skipif(_missing is not None, reason=f"formal tool missing: {_missing}")
@pytest.mark.parametrize("job", list(run_formal.JOBS))
def test_ctrl_formal(job):
    ok, log = run_formal.run_job(job)
    assert ok, log
