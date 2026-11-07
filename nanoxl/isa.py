"""NanoXL command ISA (contract C4). Normative description: docs/isa.md.

Every command is one 128-bit little-endian word. Bits [5:0] hold the opcode; the
remaining fields are packed LSB-first in the order listed in ``LAYOUTS``. Unused
high bits are zero and must decode as zero (the decoder rejects non-zero padding).

Field kinds:
  u   unsigned, stored as is, range [0, 2^w)
  m1  count, stored minus one, range [1, 2^w]  (so 12 bits encode 1..4096)
  s   signed two's complement, range [-2^(w-1), 2^(w-1))
"""
from __future__ import annotations

from dataclasses import dataclass, field

WORD_BITS = 128
WORD_BYTES = 16
OPCODE_BITS = 6
LINE = 16  # scratchpad "line": every scratchpad row stride is in units of 16 bytes

# Scratchpad buffer ids (3-bit field where a command can name any buffer).
ABUF, WBUF, ACC, PBUF, KVBUF = 0, 1, 2, 3, 4
BUF_NAMES = {ABUF: "ABUF", WBUF: "WBUF", ACC: "ACC", PBUF: "PBUF", KVBUF: "KVBUF"}

# Queues.
Q_DMA, Q_ARR, Q_VEC, Q_CTRL = 0, 1, 2, 3
QUEUE_NAMES = {Q_DMA: "DMA", Q_ARR: "ARRAY", Q_VEC: "VEC", Q_CTRL: "CTRL"}

# Vector micro-ops (VEC.vop) and element types (VEC.dt_in / dt_out).
VOPS = {"COPY": 0, "ADD": 1, "SUB": 2, "MUL": 3, "MAX": 4, "MIN": 5, "ADDI": 6, "SHRR": 7, "SHL": 8}
VOP_NAMES = {v: k for k, v in VOPS.items()}
DT_I8, DT_I16, DT_I32 = 0, 1, 2
DT_BYTES = {DT_I8: 1, DT_I16: 2, DT_I32: 4}

OPCODES = {
    "END": 0x01, "IRQ": 0x02, "WAIT": 0x03, "SIGNAL": 0x04,
    "DMA_LOAD": 0x08, "DMA_STORE": 0x09,
    "GEMM": 0x10, "ATTN_TILE": 0x11,
    "VEC": 0x18, "REQUANT": 0x19, "LN": 0x1A, "SOFTMAX_ONLINE": 0x1B, "GELU": 0x1C, "KV_APPEND": 0x1D,
    # added in ADR-004 (additive: new opcodes only, existing encodings unchanged)
    "GEMM_GS": 0x12, "LUT_LOAD": 0x1E,
}
OP_NAMES = {v: k for k, v in OPCODES.items()}

# Fixed queue per opcode; WAIT/SIGNAL carry an explicit queue field.
OP_QUEUE = {
    "END": Q_CTRL, "IRQ": Q_CTRL,
    "DMA_LOAD": Q_DMA, "DMA_STORE": Q_DMA,
    "GEMM": Q_ARR, "ATTN_TILE": Q_ARR, "GEMM_GS": Q_ARR,
    "LUT_LOAD": Q_VEC,
    "VEC": Q_VEC, "REQUANT": Q_VEC, "LN": Q_VEC, "SOFTMAX_ONLINE": Q_VEC, "GELU": Q_VEC, "KV_APPEND": Q_VEC,
}

# (name, width, kind). Scratchpad addresses: 20 bits (1 MiB) for ABUF/ACC/PBUF,
# 21 bits (2 MiB) where WBUF/KVBUF can be addressed. See docs/isa.md for justification.
LAYOUTS: dict[str, list[tuple[str, int, str]]] = {
    "END": [],
    "IRQ": [("code", 16, "u")],
    "WAIT": [("queue", 2, "u"), ("sem", 5, "u"), ("count", 8, "u")],
    "SIGNAL": [("queue", 2, "u"), ("sem", 5, "u")],
    "DMA_LOAD": [("chan", 2, "u"), ("sp_buf", 3, "u"), ("sp_addr", 21, "u"), ("ddr_addr", 32, "u"),
                 ("rows", 12, "m1"), ("row_bytes", 16, "m1"), ("ddr_stride", 24, "u"), ("sp_stride", 12, "u")],
    "GEMM": [("int4", 1, "u"), ("gemv", 1, "u"), ("w_trans", 1, "u"), ("accumulate", 1, "u"),
             ("a_addr", 20, "u"), ("w_buf", 1, "u"), ("w_addr", 21, "u"), ("c_addr", 20, "u"),
             ("M", 12, "m1"), ("N", 12, "m1"), ("K", 12, "m1"), ("a_stride", 10, "u"), ("c_stride", 10, "u")],
    "ATTN_TILE": [("first", 1, "u"), ("last", 1, "u"), ("q_addr", 20, "u"), ("k_addr", 21, "u"),
                  ("v_addr", 21, "u"), ("o_addr", 20, "u"), ("ctx_addr", 20, "u"), ("rows", 6, "m1"),
                  ("len", 12, "m1")],
    "VEC": [("vop", 5, "u"), ("dt_in", 2, "u"), ("dt_out", 2, "u"),
            ("src0_buf", 3, "u"), ("src0_addr", 21, "u"), ("src1_buf", 3, "u"), ("src1_addr", 21, "u"),
            ("dst_buf", 3, "u"), ("dst_addr", 21, "u"), ("len", 20, "m1"), ("imm", 16, "s")],
    "REQUANT": [("out16", 1, "u"), ("per_tensor", 1, "u"), ("src_addr", 20, "u"), ("dst_addr", 20, "u"),
                ("param_addr", 20, "u"), ("M", 12, "m1"), ("N", 12, "m1"),
                ("src_stride", 12, "u"), ("dst_stride", 12, "u")],
    "LN": [("in16", 1, "u"), ("src_addr", 20, "u"), ("dst_addr", 20, "u"), ("param_addr", 20, "u"),
           ("rows", 12, "m1"), ("cols", 12, "m1"), ("src_stride", 12, "u"), ("dst_stride", 12, "u")],
    "SOFTMAX_ONLINE": [("first", 1, "u"), ("last", 1, "u"), ("causal", 1, "u"),
                       ("src_addr", 20, "u"), ("dst_addr", 20, "u"), ("ctx_addr", 20, "u"),
                       ("rows", 12, "m1"), ("cols", 12, "m1"), ("diag", 13, "s"),
                       ("src_stride", 10, "u"), ("dst_stride", 10, "u")],
    "GELU": [("src_addr", 20, "u"), ("dst_addr", 20, "u"), ("lut_addr", 20, "u"), ("len", 20, "m1")],
    "KV_APPEND": [("src_addr", 20, "u"), ("dst_addr", 21, "u"), ("pos", 12, "u"), ("heads", 6, "m1"),
                  ("dh", 8, "m1"), ("ctx", 12, "m1"), ("rows", 12, "m1"), ("src_stride", 12, "u")],
    # ADR-004: INT4 GEMM with per-(group, column) uint8 scales applied to each K-group partial sum.
    "GEMM_GS": [("gemv", 1, "u"), ("accumulate", 1, "u"), ("a_addr", 20, "u"), ("w_addr", 21, "u"),
                ("c_addr", 20, "u"), ("M", 12, "m1"), ("N", 12, "m1"), ("K", 12, "m1"),
                ("a_stride", 10, "u"), ("c_stride", 10, "u"), ("glog2", 3, "u")],
    # ADR-004: load one 256-entry uint16 SFU table (C5 lut.exp2 / lut.recip / lut.rsqrt) from PBUF.
    "LUT_LOAD": [("table", 2, "u"), ("src_addr", 20, "u")],
}
LAYOUTS["DMA_STORE"] = LAYOUTS["DMA_LOAD"]

# LUT_LOAD.table ids, GEMM_GS group size = 2^(glog2 + GS_LOG2_BASE)
LUT_EXP2, LUT_RECIP, LUT_RSQRT = 0, 1, 2
LUT_NAMES = {LUT_EXP2: "exp2", LUT_RECIP: "recip", LUT_RSQRT: "rsqrt"}
LUT_ENTRIES = 256
GS_LOG2_BASE = 3


def gs_group(glog2: int) -> int:
    return 1 << (glog2 + GS_LOG2_BASE)


def gs_layout(K: int, N: int, glog2: int) -> tuple[int, int, int]:
    """GEMM_GS operand block in WBUF: (packed-weight bytes, scale offset, total bytes).
    Packed int4 W [K, N/2] (C1b) at w_addr, then gm uint8 [ceil(K/G), N] at w_addr + scale offset
    (the weight part rounded up to a 16-byte line)."""
    wb = K * N // 2
    off = -(-wb // LINE) * LINE
    G = gs_group(glog2)
    return wb, off, off + -(-K // G) * N

for _op, _fs in LAYOUTS.items():
    _bits = OPCODE_BITS + sum(w for _, w, _ in _fs)
    assert _bits <= WORD_BITS, (_op, _bits)


def field_range(width: int, kind: str) -> tuple[int, int]:
    if kind == "u":
        return 0, (1 << width) - 1
    if kind == "m1":
        return 1, 1 << width
    if kind == "s":
        return -(1 << (width - 1)), (1 << (width - 1)) - 1
    raise ValueError(kind)


class IsaError(ValueError):
    pass


@dataclass
class Cmd:
    """One decoded command. ``f`` holds every field of the opcode's layout (missing ones default to 0,
    or 1 for m1 counts). ``note`` is a free-form comment that is not encoded."""
    op: str
    f: dict = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        if self.op not in LAYOUTS:
            raise IsaError(f"unknown op {self.op}")
        names = {n for n, _, _ in LAYOUTS[self.op]}
        extra = set(self.f) - names
        if extra:
            raise IsaError(f"{self.op}: unknown fields {sorted(extra)}")
        for n, w, k in LAYOUTS[self.op]:
            self.f.setdefault(n, 1 if k == "m1" else 0)

    def __getitem__(self, k):
        return self.f[k]

    @property
    def queue(self) -> int:
        if self.op in ("WAIT", "SIGNAL"):
            return self.f["queue"]
        return OP_QUEUE[self.op]

    def __eq__(self, other):
        return isinstance(other, Cmd) and self.op == other.op and self.f == other.f

    def __repr__(self):
        body = ", ".join(f"{k}={v}" for k, v in self.f.items())
        return f"{self.op}({body})"


def C(op: str, note: str = "", **fields) -> Cmd:
    return Cmd(op, dict(fields), note)


def validate(cmd: Cmd) -> None:
    """Field-range and semantic validation (raises IsaError)."""
    for n, w, k in LAYOUTS[cmd.op]:
        v = cmd.f[n]
        if isinstance(v, bool):
            raise IsaError(f"{cmd.op}.{n}: bool is not a field value")
        if not isinstance(v, int):
            if hasattr(v, "__index__"):
                v = int(v)
                cmd.f[n] = v
            else:
                raise IsaError(f"{cmd.op}.{n}: not an int ({v!r})")
        lo, hi = field_range(w, k)
        if not lo <= v <= hi:
            raise IsaError(f"{cmd.op}.{n}={v} outside [{lo}, {hi}]")
    op, f = cmd.op, cmd.f
    if op in ("WAIT", "SIGNAL") and f["queue"] == Q_CTRL:
        raise IsaError(f"{op}: queue must be DMA, ARRAY or VEC")
    if op == "WAIT" and f["count"] == 0:
        raise IsaError("WAIT.count must be >= 1")
    if op in ("DMA_LOAD", "DMA_STORE") and f["sp_buf"] not in BUF_NAMES:
        raise IsaError(f"{op}.sp_buf={f['sp_buf']} is not a buffer")
    if op == "VEC":
        if f["vop"] not in VOP_NAMES:
            raise IsaError(f"VEC.vop={f['vop']} undefined")
        for b in ("src0_buf", "src1_buf", "dst_buf"):
            if f[b] not in BUF_NAMES:
                raise IsaError(f"VEC.{b}={f[b]} is not a buffer")
        if f["dt_in"] not in DT_BYTES or f["dt_out"] not in DT_BYTES:
            raise IsaError("VEC dtype must be 0 (i8), 1 (i16) or 2 (i32)")
    if op == "GEMM_GS" and f["N"] % 2:
        raise IsaError("GEMM_GS: int4 needs even N (C1b packs two per byte along N)")
    if op == "LUT_LOAD" and f["table"] not in LUT_NAMES:
        raise IsaError(f"LUT_LOAD.table={f['table']} undefined")
    if op == "GEMM":
        if f["int4"] and f["w_trans"]:
            raise IsaError("GEMM: int4 weights cannot be transposed")
        if f["int4"] and f["N"] % 2:
            raise IsaError("GEMM: int4 needs even N (C1b packs two per byte along N)")


# ---------------------------------------------------------------- alignment rule (docs/isa.md §3)
# Scratchpad addresses that the RTL may treat as 16-byte line addresses. Every command the compiler
# emits satisfies the rule; ``encode(..., aligned=True)`` and ``check_aligned`` enforce it. The bit
# format itself still carries the low 4 bits (the decoder vectors exercise full field ranges), but a
# command that violates the rule has undefined behaviour on hardware.
ALIGNED_FIELDS = {
    "DMA_LOAD": ("sp_addr", "ddr_addr"), "DMA_STORE": ("sp_addr", "ddr_addr"),
    "GEMM": ("a_addr", "w_addr", "c_addr"), "GEMM_GS": ("a_addr", "w_addr", "c_addr"),
    "ATTN_TILE": ("q_addr", "k_addr", "v_addr", "o_addr", "ctx_addr"),
    "VEC": ("src0_addr", "src1_addr", "dst_addr"),
    "REQUANT": ("src_addr", "dst_addr", "param_addr"),
    "LN": ("src_addr", "dst_addr", "param_addr"),
    "SOFTMAX_ONLINE": ("src_addr", "dst_addr", "ctx_addr"),
    "GELU": ("src_addr", "dst_addr", "lut_addr"),
    "KV_APPEND": ("src_addr", "dst_addr"),
    "LUT_LOAD": ("src_addr",),
}


def check_aligned(cmd: Cmd) -> None:
    """Raise IsaError unless every base address of ``cmd`` is a multiple of LINE (16 bytes)."""
    for n in ALIGNED_FIELDS.get(cmd.op, ()):
        if cmd.f[n] % LINE:
            raise IsaError(f"{cmd.op}.{n}=0x{cmd.f[n]:x} is not {LINE}-byte aligned")


def encode(cmd: Cmd, aligned: bool = False) -> bytes:
    validate(cmd)
    if aligned:
        check_aligned(cmd)
    word = OPCODES[cmd.op]
    pos = OPCODE_BITS
    for n, w, k in LAYOUTS[cmd.op]:
        v = cmd.f[n]
        if k == "m1":
            v -= 1
        elif k == "s":
            v &= (1 << w) - 1
        word |= v << pos
        pos += w
    return word.to_bytes(WORD_BYTES, "little")


def decode(b: bytes) -> Cmd:
    if len(b) != WORD_BYTES:
        raise IsaError(f"command word must be {WORD_BYTES} bytes")
    word = int.from_bytes(b, "little")
    opc = word & ((1 << OPCODE_BITS) - 1)
    if opc not in OP_NAMES:
        raise IsaError(f"illegal opcode 0x{opc:02x}")
    op = OP_NAMES[opc]
    pos = OPCODE_BITS
    f = {}
    for n, w, k in LAYOUTS[op]:
        v = (word >> pos) & ((1 << w) - 1)
        if k == "m1":
            v += 1
        elif k == "s" and v >> (w - 1):
            v -= 1 << w
        f[n] = v
        pos += w
    if word >> pos:
        raise IsaError(f"{op}: non-zero reserved bits above bit {pos}")
    cmd = Cmd(op, f)
    validate(cmd)
    return cmd


def encode_stream(cmds, aligned: bool = False) -> bytes:
    return b"".join(encode(c, aligned) for c in cmds)


def decode_stream(b: bytes) -> list[Cmd]:
    if len(b) % WORD_BYTES:
        raise IsaError("stream length not a multiple of 16")
    return [decode(b[i:i + WORD_BYTES]) for i in range(0, len(b), WORD_BYTES)]


def bit_map(op: str) -> list[tuple[str, int, int, str]]:
    """[(field, lsb, msb, kind)] including the opcode, for docs and RTL decoders."""
    out = [("opcode", 0, OPCODE_BITS - 1, "u")]
    pos = OPCODE_BITS
    for n, w, k in LAYOUTS[op]:
        out.append((n, pos, pos + w - 1, k))
        pos += w
    return out


def markdown_tables() -> str:
    """Field tables in docs/isa.md are generated from LAYOUTS (a test keeps them in sync)."""
    lines = []
    for op in OPCODES:
        q = "explicit" if op in ("WAIT", "SIGNAL") else QUEUE_NAMES[OP_QUEUE[op]]
        lines.append(f"#### {op} (opcode 0x{OPCODES[op]:02x}, queue {q})\n")
        lines.append("| field | bits | kind |")
        lines.append("|---|---|---|")
        for n, lsb, msb, k in bit_map(op):
            lines.append(f"| {n} | [{msb}:{lsb}] | {k} |")
        lines.append("")
    return "\n".join(lines)
