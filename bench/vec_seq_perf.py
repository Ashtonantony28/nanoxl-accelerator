"""Measure per-command occupancy of the VEC engine RTL (rtl/vec_seq.sv) at L = 4, 16 (Icarus or
Verilator) and 64 (Verilator), with PL = 1 and 4 lines per port beat, and write
bench/results/vec_seq_perf.json ({"pl<PL>": {"<L>": {...}}}), the table that nanoxl/cyclemodel.py's
VEC costs are pinned to (tests/audit/test_audit1.py, AUDIT-1 A1-01).

    python3 bench/vec_seq_perf.py                 # L = 4, 16, 64 x PL = 1, 4 (needs Verilator for 64)
    python3 bench/vec_seq_perf.py --lanes 4 16 --ports 4 --sim icarus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "rtl"))
sys.path.insert(0, str(ROOT))

from simrun import run  # noqa: E402
from test_vec_seq import SRC  # noqa: E402

OUT = ROOT / "bench" / "results" / "vec_seq_perf.json"


def measure(L, sim, pl=1, lat=2):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "perf.jsonl"
        pp = os.pathsep.join([str(ROOT / "bench"), str(ROOT / "tests" / "rtl"), str(ROOT),
                              os.environ.get("PYTHONPATH", "")])
        run("vec_seq", SRC, "vec_seq_perf_cocotb", parameters={"L": L, "PL": pl}, sim=sim, tag="_perf",
            extra_env={"L": str(L), "PL": str(pl), "LAT": str(lat), "PERF_OUT": str(out), "PYTHONPATH": pp})
        return [json.loads(x) for x in out.read_text().splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--ports", type=int, nargs="+", default=[1, 4], help="PL: lines per port beat")
    ap.add_argument("--sim", default=None, help="default: icarus for L <= 16, verilator for 64")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    old = json.loads(out.read_text()) if out.exists() else {}
    for pl in a.ports:
        for L in a.lanes:
            sim = a.sim or ("verilator" if L > 16 else "icarus")
            rows = measure(L, sim, pl)
            old.setdefault(f"pl{pl}", {})[str(L)] = {"sim": sim, "rows": rows}
            for r in rows:
                print(f"PL={pl} L={L:3d} {r['name']:34s} first {r['first']:6d} occ {r['occ']:6d}")
    out.write_text(json.dumps(old, indent=1) + "\n")


if __name__ == "__main__":
    main()
