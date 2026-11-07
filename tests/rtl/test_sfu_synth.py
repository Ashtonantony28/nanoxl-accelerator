"""Yosys generic synthesis of the SFU RTL at L = 4: must synthesize with no latches.
Prints cell and flip-flop counts (docs/sfu.md §7). Skipped when yosys is not installed.
Row buffers and parameter memories are mapped to flip-flops by generic synthesis (no RAM
library), so the counts are dominated by storage; NSLOT = 1 keeps LayerNorm small."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PRIMS = ["rtl/lut_rom.sv", "rtl/sfu_exp2.sv", "rtl/sfu_recip.sv", "rtl/sfu_rsqrt.sv"]
DEPS = {
    "sfu_elem": (PRIMS + ["rtl/sfu_elem.sv"], {"L": 4, "NG": 4}),
    "softmax": (PRIMS + ["rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/requant.sv", "rtl/softmax.sv"],
                {"L": 4, "CTX_MAX": 256, "NCTX": 4}),
    "layernorm": (PRIMS + ["rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/layernorm.sv"],
                  {"L": 4, "D_MAX": 384, "NSLOT": 1}),
    "sfu": (PRIMS + ["rtl/sfu_elem.sv", "rtl/srdhm.sv", "rtl/sat_counter.sv", "rtl/requant.sv",
                     "rtl/softmax.sv", "rtl/layernorm.sv", "rtl/sfu.sv"],
            {"L": 4, "CTX_MAX": 256, "D_MAX": 384, "NG": 4, "NSLOT": 1, "NCTX": 4}),
}


def synth(top):
    srcs, params = DEPS[top]
    files = " ".join(str(ROOT / s) for s in srcs)
    chp = " ".join(f"-set {k} {v}" for k, v in params.items())
    script = f"read_verilog -sv {files}; chparam {chp} {top}; synth -flatten -top {top}; stat"
    r = subprocess.run(["yosys", "-q", "-p", script, "-l", "/dev/stdout"], capture_output=True,
                       text=True, timeout=3000)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    log = r.stdout
    assert "Latch inferred" not in log and "$_DLATCH" not in log, "latch inferred"
    stat = log[log.rindex("Number of cells"):]
    cells = int(re.search(r"Number of cells:\s+(\d+)", stat).group(1))
    ffs = sum(int(n) for k, n in re.findall(r"\$_(S?DFF\w*)\s+(\d+)", stat))
    return cells, ffs


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not installed")
@pytest.mark.parametrize("top", [pytest.param("sfu_elem", marks=pytest.mark.slow),
                                 pytest.param("softmax", marks=pytest.mark.slow),
                                 pytest.param("layernorm", marks=pytest.mark.slow),
                                 pytest.param("sfu", marks=pytest.mark.slow)])
def test_sfu_synth(top):
    cells, ffs = synth(top)
    print(f"\n[yosys synth L=4] {top}: {cells} cells, {ffs} flip-flops")
    assert cells > 0 and ffs > 0
