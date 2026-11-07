"""Rewrite rtl/*.sv into a form Yosys 0.33's built-in SystemVerilog frontend reads (synthesis-only copies).

Yosys 0.33 rejects variable declarations inside *unnamed* begin/end blocks (legal SV, used throughout
the RTL in always_comb loops) and type casts (`int'(x)`, `signed'(x)`, `~W'(x)`). This gives every unnamed `begin` a unique label and rewrites the casts as $signed / size casts; nothing else changes,
so the netlist is the same design. Usage: python3 fpga/yosys/prep.py OUT_DIR rtl/a.sv rtl/b.sv ...
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BEGIN = re.compile(r"\bbegin\b(?!\s*:)")
_COMMENT = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)


def _casts(src: str) -> str:
    """int'(x) -> $signed(32'(x)), signed'(x) -> $signed(x) (Yosys 0.33 has size casts only);
    ~W'(x) -> ~(W'(x)) (Yosys 0.33 misparses a size cast right after a unary operator)."""
    out, i = [], 0
    pat = re.compile(r"(?:\b(int|signed)|~\s*(\w+))'\(")
    while True:
        m = pat.search(src, i)
        if not m:
            out.append(src[i:])
            return "".join(out)
        out.append(src[i:m.start()])
        j, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(src[j], 0)
            j += 1
        inner = _casts(src[m.end():j - 1])
        if m.group(2):
            out.append(f"~({m.group(2)}'({inner}))")
        else:
            out.append(f"$signed(32'({inner}))" if m.group(1) == "int" else f"$signed({inner})")
        i = j


def prep_text(src: str, tag: str) -> str:
    n = [0]

    def lab(m):
        n[0] += 1
        return f"begin : yb_{tag}_{n[0]}"

    # strip comments first so that 'begin' in a comment is not touched (and labels stay unique)
    src = _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src)
    return _casts(_BEGIN.sub(lab, src))


def main(argv):
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(argv[2:]):
        p = Path(f)
        (out / p.name).write_text(prep_text(p.read_text(), f"{i}"))


if __name__ == "__main__":
    main(sys.argv)
