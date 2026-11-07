import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# root (nanoxl, fpga), tests/rtl (simrun, top_cocotb), tests/fpga (wrap_cocotb); the cocotb runner passes
# sys.path to the simulator's Python as PYTHONPATH.
for p in (ROOT, ROOT / "tests" / "rtl", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
