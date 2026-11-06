"""Yosys generic synthesis of the vector-unit RTL at L=4: must synthesize with no latches.
Prints cell counts (docs/vector-unit.md §7.1). Skipped when yosys is not installed."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEPS = {
    "requant": ["rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/requant.sv"],
    "vec_alu": ["rtl/sat_counter.sv", "rtl/vec_alu.sv"],
    "vec_reduce": ["rtl/vec_reduce.sv"],
    "vec_unit": ["rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/requant.sv", "rtl/vec_alu.sv",
                 "rtl/vec_reduce.sv", "rtl/vec_unit.sv"],
}


def synth(top, L=4):
    srcs = " ".join(str(ROOT / s) for s in DEPS[top])
    script = f"read_verilog -sv {srcs}; chparam -set L {L} {top}; synth -flatten -top {top}; stat"
    r = subprocess.run(["yosys", "-p", script], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    log = r.stdout
    assert "Latch inferred" not in log and "$_DLATCH" not in log, "latch inferred"
    stat = log[log.rindex("Printing statistics"):]
    cells = int(re.search(r"Number of cells:\s+(\d+)", stat).group(1))
    ffs = sum(int(n) for k, n in re.findall(r"\$_(S?DFF\w*)\s+(\d+)", stat))
    return cells, ffs


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not installed")
@pytest.mark.parametrize("top", ["requant", "vec_alu",
                                 pytest.param("vec_reduce", marks=pytest.mark.slow),
                                 pytest.param("vec_unit", marks=pytest.mark.slow)])
def test_vec_synth(top):
    cells, ffs = synth(top)
    print(f"\n[yosys synth L=4] {top}: {cells} cells, {ffs} flip-flops")
    assert cells > 0 and ffs > 0
