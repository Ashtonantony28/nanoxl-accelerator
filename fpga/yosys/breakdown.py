"""Per-module breakdown of a Yosys `stat -top` report (fpga/yosys/estimate.py writes them under
sim_build/fpga/<board>/<job>.stat): own cells of each module x its instance count in the hierarchy.

    python3 fpga/yosys/breakdown.py sim_build/fpga/kv260/vec_seq.stat [--top 15]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.yosys.estimate import classify, parse_stat  # noqa: E402


def short(name: str) -> str:
    return re.sub(r"^\$paramod(\$[0-9a-f]+)?\\", "", name).split("\\")[0]


def breakdown(text: str) -> list[tuple[str, int, dict]]:
    """[(module, instances, classified totals)] sorted by LUT + FF."""
    mods, _ = parse_stat(text)
    # instance counts from the hierarchy listing (indentation = depth)
    lines = text[text.index("=== design hierarchy ==="):].splitlines()
    inst, stack = {}, []
    for ln in lines[1:]:
        m = re.match(r"^(\s+)(\S+)\s+(\d+)$", ln)
        if not m:
            if "Number of wires" in ln:
                break
            continue
        depth, name, n = len(m.group(1)), m.group(2), int(m.group(3))
        while stack and stack[-1][0] >= depth:
            stack.pop()
        mult = n * (stack[-1][1] if stack else 1)
        inst[name] = inst.get(name, 0) + mult
        stack.append((depth, mult))
    out = []
    for name, cells in mods.items():
        n = inst.get(name, 0)
        if not n:
            continue
        c = classify(cells)
        out.append((short(name), n, {k: v * n for k, v in c.items() if k != "other" and v}))
    agg = {}
    for name, n, c in out:
        a = agg.setdefault(name, [0, {}])
        a[0] += n
        for k, v in c.items():
            a[1][k] = a[1].get(k, 0) + v
    res = [(k, v[0], v[1]) for k, v in agg.items()]
    return sorted(res, key=lambda r: -(r[2].get("LUT", 0) + r[2].get("FF", 0)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stat")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    for name, n, c in breakdown(Path(a.stat).read_text())[:a.top]:
        print(f"{name:22s} x{n:<5d} " + " ".join(f"{k}={v}" for k, v in c.items()))


if __name__ == "__main__":
    main()
