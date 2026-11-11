"""pytest wrappers: rtl/vec_seq_wr.sv wide mode (vec_unit beats wider than a port beat) and its
simulation-only walker check (AUDIT-2 A2-05). cocotb module: vec_seq_wr_cocotb.py."""
import pytest

from simrun import run

SRC = ["rtl/fifo_sync.sv", "rtl/vec_seq_walk.sv", "rtl/vec_seq_wr.sv"]


@pytest.mark.parametrize("case", ["legal", "interior", "ragged", "violate"])
@pytest.mark.parametrize("L,P", [(16, 1), (16, 2)])
def test_vec_seq_wr_wide(L, P, case):
    """legal / interior / ragged: the written lines are exactly the row's kept bytes and the check
    stays quiet; violate: a non-final beat with empty trailing chunks makes the check fire."""
    run("vec_seq_wr", SRC, "vec_seq_wr_cocotb", parameters={"L": L, "P": P},
        extra_env={"L": str(L), "P": str(P), "CASE": case}, tag=f"_{case}")
