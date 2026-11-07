#!/usr/bin/env python3
"""Run the SG6 formal proofs with yosys + yosys-smtbmc (SymbiYosys is not required).

Each job elaborates one module's `ifdef FORMAL` properties (formal/<job>.ys -> SMT2) and runs
  * BMC:              yosys-smtbmc -s <solver> -t <depth>      (no counterexample from reset)
  * k-induction:      yosys-smtbmc -s <solver> -i -t <depth>   (the properties are inductive)
Both passing = an unbounded proof of the asserted properties.

Usage: python3 formal/run_formal.py [job ...]      (default: every job)
Env:   FORMAL_SOLVER (default z3), FORMAL_BUILD (default sim_build/formal)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# job: (BMC depth, induction depth)
JOBS = {
    "fifo": (12, 12),     # fifo_sync W=4 DEPTH=4
    "fifo8": (20, 20),    # fifo_sync W=3 DEPTH=8
    "sem": (8, 8),        # sem_unit NQ=3 NSEM=2 CW=3
    "ring": (10, 10),     # cmd_ring_ptr PW=4 FDEPTH=4 FBURST=3
    "queue": (8, 8),      # cmd_queue QDEPTH=2 MAXOUT=2 WW=13 NCH=2 PSIG=2 (SG10a: carried SIGNALs, STORE fence)
}


def tools_available(solver: str = None) -> str | None:
    """None if everything is installed, else the name of the missing tool."""
    solver = solver or os.environ.get("FORMAL_SOLVER", "z3")
    for t in ("yosys", "yosys-smtbmc", solver):
        if shutil.which(t) is None:
            return t
    return None


def run_job(job: str, solver: str = None, build: Path = None) -> tuple[bool, str]:
    solver = solver or os.environ.get("FORMAL_SOLVER", "z3")
    build = Path(build or os.environ.get("FORMAL_BUILD", ROOT / "sim_build" / "formal")) / job
    build.mkdir(parents=True, exist_ok=True)
    script = (HERE / f"{job}.ys").read_text().replace("../rtl/", str(ROOT / "rtl") + "/")
    (build / f"{job}.ys").write_text(script)
    log = []
    r = subprocess.run(["yosys", "-q", "-s", f"{job}.ys"], cwd=build, capture_output=True, text=True)
    log.append(r.stdout + r.stderr)
    if r.returncode:
        return False, "\n".join(log)
    bmc, ind = JOBS[job]
    for mode, args in (("bmc", ["-t", str(bmc)]), ("induction", ["-i", "-t", str(ind)])):
        r = subprocess.run(["yosys-smtbmc", "-s", solver, *args, f"{job}.smt2"], cwd=build,
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        log.append(f"--- {job} {mode} ---\n" + out[-3000:])
        if r.returncode or "Status: PASSED" not in out:
            return False, "\n".join(log)
    return True, "\n".join(log)


def main(argv):
    missing = tools_available()
    if missing:
        print(f"formal: {missing} not found; install yosys (with yosys-smtbmc) and z3 "
              f"(pip install z3-solver)")
        return 2
    jobs = argv or list(JOBS)
    ok = True
    for j in jobs:
        good, log = run_job(j)
        print(f"{j:6s} {'PASS' if good else 'FAIL'}")
        if not good:
            print(log)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
