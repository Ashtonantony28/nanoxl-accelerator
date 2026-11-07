"""rtl/cmd_fields.sv is generated from nanoxl.isa (contract C4). This test keeps them in sync.

Regenerate after an ISA change:  python3 tests/rtl/test_cmd_isagen.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanoxl import isa  # noqa: E402

OUT = ROOT / "rtl" / "cmd_fields.sv"


def port_name(op: str, field: str) -> str:
    return f"{op.lower()}_{field.lower()}"


def out_width(w: int, kind: str) -> int:
    return w + 1 if kind == "m1" else w


def generate() -> str:
    L = []
    L.append("// cmd_fields.sv -- GENERATED from nanoxl/isa.py by tests/rtl/test_cmd_isagen.py; do not edit.")
    L.append("// Regenerate: python3 tests/rtl/test_cmd_isagen.py  (test_cmd_isagen fails while stale)")
    L.append("//")
    L.append("// Pure combinational field extraction for a 128-bit command word (docs/isa.md):")
    L.append("//   known      opcode is defined in the ISA (opcode 0 and every unlisted value are illegal)")
    L.append("//   resv_ok    all bits above the opcode's last field are zero")
    L.append("//   is_<op>    one-hot opcode match")
    L.append("//   fixed_q    queue for opcodes with a fixed queue (0 DMA, 1 ARRAY, 2 VEC, 3 CTRL);")
    L.append("//              WAIT/SIGNAL report 3 here and carry their queue in the queue field")
    L.append("//   <op>_<f>   every field, decoded: 'm1' counts have one extra bit and the +1 applied,")
    L.append("//              's' fields are raw two's complement")
    L.append("module cmd_fields (")
    L.append("    input  logic [127:0] w,")
    L.append("    output logic [5:0]   opcode,")
    L.append("    output logic         known,")
    L.append("    output logic         resv_ok,")
    L.append("    output logic [1:0]   fixed_q,")
    ports = []
    for op in isa.OPCODES:
        ports.append(f"    output logic {''.ljust(7)} is_{op.lower()}")
    for op in isa.OPCODES:
        for n, lsb, msb, k in isa.bit_map(op)[1:]:
            ow = out_width(msb - lsb + 1, k)
            rng = f"[{ow - 1}:0]" if ow > 1 else ""
            ports.append(f"    output logic {rng.ljust(7)} {port_name(op, n)}")
    L.append(",\n".join(ports))
    L.append(");")
    L.append("    assign opcode = w[5:0];")
    for op, code in isa.OPCODES.items():
        L.append(f"    assign is_{op.lower()} = (w[5:0] == 6'h{code:02x});")
    L.append("    assign known = " + " | ".join(f"is_{op.lower()}" for op in isa.OPCODES) + ";")
    rv, fq = [], []
    for op, code in isa.OPCODES.items():
        top = isa.bit_map(op)[-1][2] + 1
        chk = "1'b1" if top >= 128 else f"(w[127:{top}] == '0)"
        rv.append(f"(is_{op.lower()} & {chk})")
        q = 3 if op in ("WAIT", "SIGNAL") else isa.OP_QUEUE[op]
        fq.append(f"is_{op.lower()} ? 2'd{q}")
    L.append("    assign resv_ok = " + "\n                   | ".join(rv) + ";")
    L.append("    assign fixed_q = " + "\n                   : ".join(fq) + " : 2'd3;")
    for op in isa.OPCODES:
        for n, lsb, msb, k in isa.bit_map(op)[1:]:
            w = msb - lsb + 1
            src = f"w[{msb}:{lsb}]" if w > 1 else f"w[{lsb}]"
            if k == "m1":
                L.append(f"    assign {port_name(op, n)} = {{1'b0, {src}}} + {w + 1}'d1;")
            else:
                L.append(f"    assign {port_name(op, n)} = {src};")
    L.append("endmodule")
    return "\n".join(L) + "\n"


def test_cmd_fields_generated_in_sync():
    assert OUT.read_text() == generate(), (
        "rtl/cmd_fields.sv is stale relative to nanoxl/isa.py; run python3 tests/rtl/test_cmd_isagen.py")


if __name__ == "__main__":
    OUT.write_text(generate())
    print(f"wrote {OUT}")
