"""Yosys resource estimates of nanoxl_top for the FPGA boards (SG9a, docs/fpga.md §2).

    python3 fpga/yosys/estimate.py kv260            # synth_xilinx -family xcup -uram, KV260 parameters
    python3 fpga/yosys/estimate.py pynq-z2          # synth_xilinx -family xc7, PYNQ-Z2 parameters
    python3 fpga/yosys/estimate.py kv260 --params N=8,L=4   # override parameters (quick runs, not written)
    python3 fpga/yosys/estimate.py kv260 --only array_seq   # one block

Per-block synthesis, summed: ctrl_top, array_seq, vec_seq, the scratchpad router (buf_spad, banks as
black boxes, x5), one spad_bank per distinct depth, and the nanoxl_top glue (blocks as black boxes)
are separate Yosys runs (2 in parallel by default). Each run is hierarchical (no -flatten): every module is mapped once per parameter set, and
`stat -top` multiplies the per-module counts up the hierarchy (1024 identical PEs cost one PE
synthesis). The RTL is first rewritten by fpga/yosys/prep.py (labels for unnamed blocks, cast
spelling) because Yosys 0.33's SystemVerilog frontend rejects those constructs. Writes
bench/results/resources_<board>.json and the Yosys logs under sim_build/fpga/<board>/.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fpga.boards import DEVICE, RTL_PARAMS  # noqa: E402
from fpga.yosys.prep import main as prep_main  # noqa: E402

FAMILY = {"kv260": ["-family", "xcup", "-uram"], "pynq-z2": ["-family", "xc7"]}

# Xilinx primitive -> resource class
CLASSES = {
    "LUT": re.compile(r"^LUT[1-6]$"),
    "LUTRAM": re.compile(r"^(RAM\d+[SDXM].*|RAM\d+M\d*|SRL.*)$"),
    "FF": re.compile(r"^FD[CPRSE]+$"),
    "CARRY": re.compile(r"^CARRY(4|8)$"),
    "MUXF": re.compile(r"^MUXF[789]$"),
    "DSP": re.compile(r"^DSP48E[12]$"),
    "BRAM36": re.compile(r"^RAMB36E[12]$"),
    "BRAM18": re.compile(r"^RAMB18E[12]$"),
    "URAM": re.compile(r"^URAM288$"),
}
# LUT6 sites per distributed-RAM / shift-register primitive (UG574 / UG474): LUT-as-memory occupies SLICEM
# LUTs, so the device LUT budget is LUT (logic) + these
LUTRAM_LUT6 = {"RAM32M16": 8, "RAM64M8": 8, "RAM32M": 4, "RAM64M": 4, "RAM32X1D": 2, "RAM64X1D": 2,
               "RAM128X1D": 4, "RAM32X1S": 1, "RAM64X1S": 1, "RAM128X1S": 2, "RAM256X1S": 4,
               "SRL16E": 1, "SRLC16E": 1, "SRLC32E": 1}
BLOCK_MODULES = ("ctrl_top", "array_seq", "vec_seq", "buf_spad")
# Pre-synthesis steps. Memories with many read ports -- array_seq's K-transposer `trc` (64 x 256 b, 32 read
# ports at N=32), vec_seq's per-lane tables (t_cols, t_end, rec_bias, ... with L+1 read ports) -- send
# memory_libmap past 13 GB and are killed. They can only be flip-flops anyway (Vivado does the same: no RAM
# primitive has more than 2 read ports, LUTRAM up to 7), so every memory with more than 4 read ports is
# mapped to registers before synth_xilinx.
PRE_ALL = ["proc", "opt_clean", "memory -nomap", "memory_map t:$mem_v2 r:RD_PORTS>4 %i"]
PRE = {"array_seq": PRE_ALL, "vec_seq": PRE_ALL}


def rtl_files():
    """nanoxl_top's source set (the list the simulations use, tests/rtl/test_top.py)."""
    sys.path.insert(0, str(ROOT / "tests" / "rtl"))
    import test_top  # noqa: E402
    return list(test_top.TOP_SRC)


def classify(cells: dict) -> dict:
    out = {k: 0 for k in CLASSES}
    other = {}
    for name, n in cells.items():
        for k, rx in CLASSES.items():
            if rx.match(name):
                out[k] += n
                break
        else:
            other[name] = n
    out["other"] = other
    out["LUTRAM_LUT6"] = sum(n * LUTRAM_LUT6.get(name, 8) for name, n in cells.items() if CLASSES["LUTRAM"].match(name))
    return out


def parse_stat(text: str) -> tuple[dict, dict]:
    """({module: {cell: n}}, design-hierarchy totals {cell: n}) from `stat -top` output."""
    mods, cur, totals, in_total, cells = {}, None, {}, False, False
    for line in text.splitlines():
        if "=== design hierarchy ===" in line:
            cur, in_total, cells = None, True, False
            continue
        m = re.match(r"^=== (.+) ===$", line.strip())
        if m:
            cur, in_total, cells = m.group(1), False, False
            mods[cur] = {}
            continue
        if "Number of cells" in line:
            cells = True              # the cell-type list follows (skips the submodule instance list)
            continue
        m = re.match(r"^\s+(\S+)\s+(\d+)$", line)
        if m and cells and (cur or in_total):
            (totals if in_total else mods[cur])[m.group(1)] = int(m.group(2))
    return mods, totals


def blocks(params: dict) -> list[tuple]:
    """(job, module, parameters, multiplicity, black boxes), mirroring nanoxl_top's instantiations.

    The five buf_spad instances differ only in LINES (bank depth / row-address width), so the router
    (buf_spad with its banks as black boxes) is synthesized once at the ACC depth and counted 5 times,
    and each distinct bank depth is one spad_bank job counted NB x (buffers of that depth)."""
    N = params["N"]
    apl = wpl = (N + 15) // 16
    cpl = N // 4 if N >= 4 else 1
    vp = params.get("VP", 1)
    pmax = max(apl, cpl, wpl, vp)
    nb = params["NB"] or (2 if pmax <= 1 else 4 if pmax <= 2 else 8 if pmax <= 4 else 16)
    # read ports: DMA0, array A, W, ACC, VEC A, B, P, DMA1; write: DMA0, array ACC, VEC D, DMA1
    rpl = [1, apl, wpl, cpl, vp, vp, vp, 1]    # read port p in byte p (rtl/nanoxl_top.sv RPL)
    wplv = [1, cpl, vp, 1]

    def hexv(v):
        return f"{8 * len(v)}'h" + "".join(f"{x:02x}" for x in reversed(v))

    out = [("ctrl_top", "ctrl_top", dict(AW=params.get("AW", 32), IDW=4, MAXB=params["MAXB"], LDF=params["LDF"],
                                         NSAT=5), 1, ()),
           # array_seq at N=32 runs Yosys out of memory (13 GB) in memory_libmap on its wide RAMs, so its
           # three array_ctrl memories are separate jobs: accumulator (M_DEPTH x 32N), output FIFO, group-scale FIFO
           ("array_seq", "array_seq", dict(N=N, M_DEPTH=params["M_DEPTH"]), 1, ("scratchpad", "spad_fifo")),
           ("arr_acc_ram", "scratchpad", dict(WIDTH=32 * N, DEPTH=params["M_DEPTH"], RD_LAT=1), 1, ()),
           ("arr_out_fifo", "spad_fifo", dict(W=32 * N + 1, DEPTH=2 * N + 8), 1, ()),
           ("arr_gs_fifo", "spad_fifo", dict(W=8 * N, DEPTH=N + 4), 1, ()),
           ("vec_seq", "vec_seq", dict(L=params["L"], AW=17, CTX_MAX=params["CTX_MAX"], D_MAX=params["D_MAX"],
                                       NG=params["NG"], NSLOT=params["NSLOT"], NCTX=params["NCTX"], PL=vp), 1, ()),
           ("spad_router", "buf_spad", dict(LINES=params["ACC_LINES"], NB=nb, NR=len(rpl), NW=len(wplv), PMAX=pmax, RPL=hexv(rpl),
                                            WPL=hexv(wplv)), 5, ("spad_bank",))]
    depths = {}
    for b in ("ABUF", "WBUF", "ACC", "PBUF", "KV"):
        d = params[f"{b}_LINES"] // nb
        depths[d] = depths.get(d, 0) + nb
    for d, n in sorted(depths.items()):
        out.append((f"spad_bank_{d}", "spad_bank", dict(DEPTH=d), n, ()))
    out.append(("glue", "nanoxl_top", dict(params), 1, BLOCK_MODULES))
    return out


def _synth_job(job):
    name, module, prm, bbox, work, files, family = job
    stat = work / f"{name}.stat"
    chp = " ".join(f"-set {k} {v}" for k, v in prm.items())
    # hierarchy derives the parameterized top under a $paramod name; rename it back
    lines = [f"read_verilog -sv -DSYNTHESIS {files}", f"chparam {chp} {module}", f"hierarchy -top {module}",
             f"rename -top {module}"]
    if bbox:
        lines.append("blackbox " + " ".join(f"*{m}*" for m in bbox))
    lines += PRE.get(module, [])
    lines += [f"synth_xilinx {' '.join(family)} -noiopad -top {module}", f"tee -q -o {stat} stat -top {module}"]
    script = "\n".join(lines) + "\n"
    ys, meta = work / f"{name}.ys", work / f"{name}.time"
    if stat.exists() and ys.exists() and ys.read_text() == script and meta.exists():   # resume after a restart
        return name, float(meta.read_text()), stat.read_text()
    ys.write_text(script)
    t0 = time.time()
    with open(work / f"{name}.log", "w") as log:
        rc = subprocess.run(["yosys", "-s", str(ys)], stdout=log, stderr=subprocess.STDOUT).returncode
    secs = time.time() - t0
    if rc != 0:                       # e.g. killed for memory: report the block as missing, keep going
        return name, secs, None
    meta.write_text(f"{secs:.1f}")
    return name, secs, stat.read_text()


def run(board: str, params: dict, write: bool = True, tag: str = "", jobs: int = 2, only=None):
    work = ROOT / "sim_build" / "fpga" / f"{board}{tag}"
    work.mkdir(parents=True, exist_ok=True)
    srcs = rtl_files()
    prep_main(["prep", str(work / "rtl"), *[str(ROOT / s) for s in srcs]])
    files = " ".join(str(work / "rtl" / Path(s).name) for s in srcs)
    bl = [b for b in blocks(params) if not only or b[0] in only]
    t0 = time.time()
    with ThreadPoolExecutor(jobs) as ex:
        done = list(ex.map(_synth_job, [(n, m, p, bb, work, files, FAMILY[board]) for n, m, p, _, bb in bl]))
    wall = time.time() - t0
    per_block, totals = {}, {}
    failed = []
    for (name, secs, text), (_, module, prm, mult, bbox) in zip(done, bl):
        if text is None:
            failed.append(name)
            per_block[name] = {"module": module, "count": mult, "runtime_s": round(secs, 1), "failed": True}
            continue
        mods, hier = parse_stat(text)
        if not hier:                               # a leaf module: its own counts
            hier = next(iter(mods.values()))
        hier = {k: v for k, v in hier.items() if not any(m in k for m in bbox)}
        c = classify(hier)
        per_block[name] = {"module": module, "count": mult, "runtime_s": round(secs, 1),
                           **{k: v for k, v in c.items() if v}}
        if module in ("array_seq", "vec_seq", "ctrl_top"):
            from fpga.yosys.breakdown import breakdown
            per_block[name]["submodules"] = [{"module": m, "instances": n, **cc}
                                             for m, n, cc in breakdown(text)[:12]]
        for k, v in hier.items():
            totals[k] = totals.get(k, 0) + v * mult
    tot = classify(totals)
    dev = DEVICE[board]
    util = {}
    for k in ("LUT", "FF", "DSP", "BRAM36", "URAM"):
        have = dev.get(k, 0)
        used = tot[k] + (tot["BRAM18"] / 2 if k == "BRAM36" else 0) + (tot["LUTRAM_LUT6"] if k == "LUT" else 0)
        util[k] = {"used": used, "avail": have, "pct": round(100 * used / have, 1) if have else None,
                   "over": bool(have and used > have) or (not have and used > 0)}
    res = {"board": board, "device": dev["name"],
           "yosys": subprocess.run(["yosys", "-V"], capture_output=True, text=True).stdout.strip(),
           "lut_note": "utilization LUT = LUT (logic) + LUTRAM_LUT6 (distributed RAM / SRL in LUT6 sites); INV excluded",
           "command": f"synth_xilinx {' '.join(FAMILY[board])}, per block (hierarchical), summed",
           "params": params, "runtime_s": round(wall, 1), "cpu_s": round(sum(b["runtime_s"] for b in per_block.values()), 1),
           "jobs": jobs, "totals": tot, "utilization": util,
           "blocks": per_block, "failed": failed}
    if write:
        out = ROOT / "bench" / "results" / f"resources_{board}.json"
        out.write_text(json.dumps(res, indent=1) + "\n")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("board", choices=sorted(RTL_PARAMS))
    ap.add_argument("--params", default="", help="override, e.g. N=8,L=4 (result not written)")
    ap.add_argument("--only", default="", help="comma-separated job names (result not written)")
    ap.add_argument("--jobs", type=int, default=2, help="parallel Yosys runs (each can take several GB)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    params = dict(RTL_PARAMS[a.board])
    over = {k: int(v) for k, v in (kv.split("=") for kv in a.params.split(",") if kv)}
    params.update(over)
    only = [x for x in a.only.split(",") if x]
    tag = ("_" + a.params.replace(",", "_").replace("=", "")) if over else ""
    res = run(a.board, params, write=not (a.no_write or over or only), tag=tag, jobs=a.jobs, only=only)
    print(f"{a.board}: {res['runtime_s']} s wall")
    for n, b in res["blocks"].items():
        if b.get("failed"):
            print(f"  {n:16s} FAILED after {b['runtime_s']} s (see sim_build/fpga/{a.board}{tag}/{n}.log)")
            continue
        print(f"  {n:16s} x{b['count']:<3} {b['runtime_s']:7.1f} s  " +
              " ".join(f"{k}={b[k]}" for k in ("LUT", "FF", "DSP", "BRAM36", "BRAM18", "URAM", "LUTRAM", "LUTRAM_LUT6") if k in b))
    for k, u in res["utilization"].items():
        print(f"  {k:7s} {u['used']:>9} / {u['avail']:<7} {'' if u['pct'] is None else str(u['pct']) + ' %'}"
              f"{'  OVER' if u['over'] else ''}")
    print("  other:", res["totals"]["other"])


if __name__ == "__main__":
    main()
