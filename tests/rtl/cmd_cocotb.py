"""cocotb tests for the command decoder: rtl/cmd_fields.sv (generated) and rtl/cmd_decode.sv.

Everything is driven from nanoxl.isa at test time (LAYOUTS, OPCODES, encode/decode/validate) plus
the golden byte vectors committed in tests/py/test_isa.py, so an additive ISA change shows up
here as soon as rtl/cmd_fields.sv is regenerated (test_cmd_isagen enforces that).
Env: CMD_TOP = cmd_fields | cmd_decode.
"""
import importlib.util
import os
import random
from pathlib import Path

import cocotb
from cocotb.triggers import Timer

from ctrl_common_cocotb import rand_legal
from nanoxl import isa

TOP = os.environ.get("CMD_TOP", "cmd_fields")
ROOT = Path(__file__).resolve().parents[2]


def golden():
    spec = importlib.util.spec_from_file_location("_t_isa", ROOT / "tests" / "py" / "test_isa.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return [(c, bytes.fromhex(h)) for c, h in m.GOLDEN]


def to_int(b: bytes) -> int:
    return int.from_bytes(b, "little")


async def apply(dut, word):
    dut.w.value = word
    await Timer(1, unit="ns")


def check_fields(dut, cmd, word):
    ops = list(isa.OPCODES)
    assert int(dut.opcode.value) == isa.OPCODES[cmd.op]
    assert int(dut.known.value) == 1 and int(dut.resv_ok.value) == 1, cmd
    for op in ops:
        assert int(getattr(dut, f"is_{op.lower()}").value) == int(op == cmd.op), (op, cmd)
    q = 3 if cmd.op in ("WAIT", "SIGNAL") else isa.OP_QUEUE[cmd.op]
    assert int(dut.fixed_q.value) == q
    for n, w, k in isa.LAYOUTS[cmd.op]:
        got = int(getattr(dut, f"{cmd.op.lower()}_{n.lower()}").value)
        exp = cmd.f[n]
        if k == "s":
            exp &= (1 << w) - 1
        assert got == exp, f"{cmd.op}.{n}: rtl {got} expected {exp} (word {word:032x})"


@cocotb.test(skip=TOP != "cmd_fields")
async def fields_golden_vectors(dut):
    """Every golden byte vector of tests/py/test_isa.py decodes to the committed fields."""
    for cmd, b in golden():
        assert isa.decode(b) == cmd
        await apply(dut, to_int(b))
        check_fields(dut, cmd, to_int(b))


@cocotb.test(skip=TOP != "cmd_fields")
async def fields_random_legal(dut):
    """Random legal commands of every opcode (encoded by nanoxl.isa): every field decodes."""
    rng = random.Random(11)
    for op in isa.OPCODES:
        for _ in range(150):
            c = rand_legal(rng, op)
            w = to_int(isa.encode(c))
            await apply(dut, w)
            check_fields(dut, c, w)


def classify(word: int):
    """Expected (err, route, kind) of cmd_decode from nanoxl.isa: err 0 legal, 1 opcode,
    2 reserved bits, 3 field validation."""
    opc = word & 0x3F
    if opc not in isa.OP_NAMES:
        return 1, None, None
    op = isa.OP_NAMES[opc]
    top = isa.bit_map(op)[-1][2] + 1
    if word >> top:
        err = 2
    else:
        try:
            isa.decode(word.to_bytes(16, "little"))
            err = 0
        except isa.IsaError:
            err = 3
    if err == 0:
        c = isa.decode(word.to_bytes(16, "little"))
        kind = {"WAIT": 1, "SIGNAL": 2, "IRQ": 3, "END": 4}.get(op, 0)
        return 0, c.queue, kind
    return err, None, None


@cocotb.test(skip=TOP != "cmd_decode")
async def decode_legality_equivalence(dut):
    """cmd_decode accepts a word iff nanoxl.isa.decode does (random words biased towards
    near-legal ones), with the right error class, queue, kind, semaphore and IRQ code."""
    rng = random.Random(12)
    counts = [0, 0, 0, 0]
    ops = list(isa.OPCODES)
    for i in range(4000):
        u = rng.random()
        if u < 0.1:
            w = rng.getrandbits(128)                       # anything, mostly illegal opcodes
        elif u < 0.15:
            w = rng.getrandbits(128) & ~0x3F                # opcode 0 (zeroed memory)
        else:
            op = rng.choice(ops)
            c = rand_legal(rng, op)
            w = to_int(isa.encode(c))
            top = isa.bit_map(op)[-1][2] + 1
            v = rng.random()
            if v < 0.2 and top < 128:
                w |= 1 << rng.randrange(top, 128)          # one reserved bit
            elif v < 0.6 and len(isa.bit_map(op)) > 1:
                fs = isa.bit_map(op)[1:]
                narrow = [f for f in fs if f[2] - f[1] < 5]   # small enum-like fields fail validation most
                n, lsb, msb, k = rng.choice(narrow if narrow and rng.random() < 0.8 else fs)
                w &= ~(((1 << (msb - lsb + 1)) - 1) << lsb)
                w |= rng.getrandbits(msb - lsb + 1) << lsb   # random raw field (may be invalid)
        err, route, kind = classify(w)
        counts[err] += 1
        await apply(dut, w)
        got = int(dut.err.value)
        assert got == err, f"word {w:032x}: rtl err {got}, isa says {err}"
        if err == 0:
            assert int(dut.route.value) == route, f"{w:032x}"
            assert int(dut.kind.value) == kind, f"{w:032x}"
            c = isa.decode(w.to_bytes(16, "little"))
            if c.op in ("WAIT", "SIGNAL"):
                assert int(dut.sem.value) == c.f["sem"]
            if c.op == "WAIT":
                assert int(dut.wait_count.value) == c.f["count"]
            if c.op == "IRQ":
                assert int(dut.irq_code.value) == c.f["code"]
    dut._log.info("legal %d, bad opcode %d, reserved %d, field %d", *counts)
    assert min(counts) > 100


@cocotb.test(skip=TOP != "cmd_decode")
async def decode_golden_and_zero(dut):
    """Golden vectors are legal; the all-zero word and every undefined opcode are illegal."""
    for cmd, b in golden():
        await apply(dut, to_int(b))
        assert int(dut.err.value) == 0, cmd
        assert int(dut.route.value) == cmd.queue
    await apply(dut, 0)
    assert int(dut.err.value) == 1
    for opc in range(64):
        if opc not in isa.OP_NAMES:
            await apply(dut, opc)
            assert int(dut.err.value) == 1, opc
