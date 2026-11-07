"""Formal proofs (BMC + k-induction with yosys-smtbmc) for the control-plane blocks: the generic
FIFO (no overflow/underflow, order and data integrity), the semaphore unit (counts never negative
or wrapping, no lost signals, progress) and the command-ring pointers (never past tail, no burst
across the ring end or a 4 KiB boundary). Skips cleanly when yosys / yosys-smtbmc / z3 are missing,
unless NANOXL_REQUIRE_FORMAL=1 (set in CI): then a missing tool is a failure, because these proofs are
the only tests that kill some sem_unit mutants (AUDIT-1 A1-10).
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "formal"))
import run_formal  # noqa: E402

_missing = run_formal.tools_available()
_required = os.environ.get("NANOXL_REQUIRE_FORMAL", "") not in ("", "0")


@pytest.mark.skipif(_missing is not None and not _required, reason=f"formal tool missing: {_missing}")
@pytest.mark.parametrize("job", list(run_formal.JOBS))
def test_ctrl_formal(job):
    if _missing is not None:
        pytest.fail(f"NANOXL_REQUIRE_FORMAL is set but the formal tool {_missing!r} is not installed")
    ok, log = run_formal.run_job(job)
    assert ok, log
