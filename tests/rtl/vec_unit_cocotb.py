"""cocotb bench for rtl/vec_unit.sv: random command sequences over rows of random length,
all streams with random valid gaps / backpressure, and a behavioural SFU on the hook port."""
from __future__ import annotations
import random

import cocotb
import numpy as np
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl import numerics as nx
from nanoxl import vec_ref as vr
from vec_common_cocotb import Sink, Source, env_int, pack, run_bench, start, u, unpack

L = env_int("L", 4)
HAS_SFU = env_int("HAS_SFU", 1)
N_CMDS = env_int("N_CMDS", 150)
ACC_W = vr.REDUCE_ACC_W
M32 = 0xFFFFFFFF

INPUTS = ["cmd_valid", "cmd_op", "cmd_len", "cmd_fmt", "cmd_shamt", "cmd_lo", "cmd_hi",
          "cmd_b_imm", "cmd_imm_b", "cmd_p_imm", "cmd_mult", "a_valid", "a_data", "b_valid",
          "b_data", "p_valid", "p_mult", "p_shift", "d_ready", "r_ready", "sfu_in_ready",
          "sfu_out_valid", "sfu_out_data", "sfu_out_keep", "sfu_out_last", "stat_clr"]


def wrap32(v):
    return ((int(v) + (1 << 31)) & M32) - (1 << 31)


def sfu_model(x, op, imm, shamt, fmt):
    """What the behavioural SFU stub computes (lets the bench check every hook field)."""
    return wrap32(3 * x + 1000 * op + imm + 7 * shamt + 100000 * fmt)


def unit_of(op):
    if op < 0x10:
        return "alu"
    if op == vr.VOP_REQUANT:
        return "rq"
    if op == vr.VOP_REDUCE:
        return "red"
    if op >= 0x18 and HAS_SFU:
        return "sfu"
    return "alu"            # reserved / SFU without HAS_SFU -> PASS, illegal_op


def is_illegal(op):
    return op >= 0x10 and unit_of(op) == "alu"


def rand_val(rng):
    r = rng.random()
    if r < 0.3:
        return rng.randrange(nx.INT32_MIN, nx.INT32_MAX + 1)
    if r < 0.7:
        return rng.randrange(-32768, 32768)
    return rng.randrange(-300, 300)


def rand_cmd(rng):
    ops = list(range(10)) + [0x0B, vr.VOP_REQUANT, vr.VOP_REQUANT, vr.VOP_REDUCE, vr.VOP_REDUCE,
                             0x13, 0x18, 0x19, 0x1F, 0x1C]
    op = rng.choice(ops)
    r = rng.random()
    n = (0 if r < 0.05 else rng.randrange(1, L + 1) if r < 0.35 else
         rng.randrange(L + 1, 5 * L + 4) if r < 0.95 else 384)
    lo, hi = sorted(rng.randrange(-500, 500) for _ in range(2))
    return {"op": op, "len": n, "fmt": rng.randrange(4), "shamt": rng.randrange(32), "lo": lo,
            "hi": hi, "b_imm": int(rng.random() < 0.3), "imm_b": rand_val(rng),
            "p_imm": int(rng.random() < 0.3), "mult": rng.randrange(1 << 30, 1 << 31)}


def build(cmds, rng):
    """Source beats for every stream and the expected outputs, from the reference model."""
    a, b, p, d, r = [], [], [], [], []
    alu_sat = rq_sat = 0
    for c in cmds:
        n, op, unit = c["len"], c["op"], unit_of(c["op"])
        if n == 0:
            continue
        nb = -(-n // L)
        row_a = [rand_val(rng) for _ in range(n)]
        aop = op if op < 0x10 else vr.OP_PASS
        need_b = not c["b_imm"] and (unit == "rq" or (unit == "alu" and aop in vr.BINARY_OPS))
        need_p = not c["p_imm"] and unit == "rq"
        row_b = [rand_val(rng) for _ in range(n)] if need_b else [c["imm_b"]] * n
        if unit == "rq":
            row_a = [rng.randrange(-(1 << 22), 1 << 22) if rng.random() < 0.7 else v for v in row_a]
            mult = [rng.randrange(1 << 30, 1 << 31) for _ in range(n)] if need_p else [c["mult"]] * n
            shift = [rng.randrange(32) for _ in range(n)] if need_p else [c["shamt"]] * n
        pad = lambda row, w: [row[j:j + L] + [rng.getrandbits(w) - (1 << (w - 1))] * (L - len(row[j:j + L]))
                              for j in range(0, n, L)]
        a += pad(row_a, 32)
        if need_b:
            b += pad(row_b, 32)
        if need_p:
            p += list(zip(pad(mult, 31), pad(shift, 5)))
        if unit == "red":
            res = vr.reduce_row(np.array(row_a, np.int64))
            r.append(dict(res, sum=vr.wrap_acc(res["sum"]), sumsq=vr.wrap_acc(res["sumsq"])))
            continue
        if unit == "alu":
            out, sat = vr.alu(aop, row_a, row_b, c["fmt"], c["shamt"], c["lo"], c["hi"])
            alu_sat += int(sat.sum())
        elif unit == "rq":
            bits = 8 if c["fmt"] == 0 else 16
            out = vr.vec_unit_op(vr.VOP_REQUANT, row_a, fmt=c["fmt"], bias=row_b, mult=mult, shift=shift)
            x = (np.array(row_a, np.int64) + np.array(row_b, np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)
            y = nx.rdivpot(nx.srdhm(x, mult), shift).astype(np.int64)
            rq_sat += int(((y < -(1 << (bits - 1))) | (y > (1 << (bits - 1)) - 1)).sum())
        else:
            out = [sfu_model(v, op & 7, c["imm_b"], c["shamt"], c["fmt"]) for v in row_a]
        out = [int(v) for v in out]
        for j in range(nb):
            seg = out[j * L:(j + 1) * L]
            k = len(seg)
            d.append((seg + [0] * (L - k), [1] * k + [0] * (L - k), int(j == nb - 1)))
    return a, b, p, d, r, alu_sat, rq_sat


class SfuStub:
    """In-order SFU with a small queue, random input backpressure and output gaps."""

    def __init__(self, dut, rng):
        self.dut, self.rng = dut, rng
        self.q, self.holding, self.rdy, self.seen = [], False, 0, 0

    def step_drive(self):
        dut = self.dut
        self.rdy = int(len(self.q) < 3 and self.rng.random() < 0.7)
        dut.sfu_in_ready.value = self.rdy
        if not self.holding and self.q and self.rng.random() < 0.7:
            self.holding = True
            data, keep, last = self.q[0]
            dut.sfu_out_data.value = pack(data, 32)
            dut.sfu_out_keep.value = pack(keep, 1)
            dut.sfu_out_last.value = last
        dut.sfu_out_valid.value = int(self.holding)

    def step_sample(self):
        dut = self.dut
        if self.holding and dut.sfu_out_ready.value == 1:
            self.q.pop(0)
            self.holding = False
        if self.rdy and dut.sfu_in_valid.value == 1:
            op, imm = u(dut.sfu_in_op), unpack(u(dut.sfu_in_imm), 1, 32)[0]
            sh, fmt = u(dut.sfu_in_shamt), u(dut.sfu_in_fmt)
            data = unpack(u(dut.sfu_in_data), L, 32)
            keep = unpack(u(dut.sfu_in_keep), L, 1, False)
            # the SFU sees zeros on lanes >= len; emulate a real unit that maps 0 -> f(0),
            # vec_unit must still zero those lanes on d.
            out = [sfu_model(x, op, imm, sh, fmt) for x in data]
            self.q.append((out, keep, u(dut.sfu_in_last)))
            self.seen += 1


def make_bench(dut, cmds, rng, p_src, p_snk):
    a, b, p, d, r, alu_sat, rq_sat = build(cmds, rng)
    R = lambda: random.Random(rng.random())

    def drive_cmd(c):
        for k in ("op", "len", "fmt", "shamt", "b_imm", "p_imm"):
            getattr(dut, "cmd_" + k).value = c[k]
        for k in ("lo", "hi", "imm_b", "mult"):
            getattr(dut, "cmd_" + k).value = c[k] & M32

    def junk_cmd():
        dut.cmd_op.value = rng.getrandbits(5)
        dut.cmd_len.value = rng.getrandbits(4)

    def drive_a(v):
        dut.a_data.value = pack(v, 32)

    def drive_b(v):
        dut.b_data.value = pack(v, 32)

    def drive_p(v):
        dut.p_mult.value = pack(v[0], 32)
        dut.p_shift.value = pack(v[1], 5)

    def cap_d():
        return (unpack(u(dut.d_data), L, 32), unpack(u(dut.d_keep), L, 1, False), u(dut.d_last))

    def cap_r():
        return {"sum": unpack(u(dut.r_sum), 1, ACC_W)[0], "sumsq": unpack(u(dut.r_sumsq), 1, ACC_W)[0],
                "max": unpack(u(dut.r_max), 1, 32)[0], "min": unpack(u(dut.r_min), 1, 32)[0],
                "sq_err": bool(u(dut.r_sq_err))}

    srcs = [Source(dut.cmd_valid, dut.cmd_ready, cmds, drive_cmd, junk_cmd, p_src, R()),
            Source(dut.a_valid, dut.a_ready, a, drive_a, lambda: drive_a([rng.getrandbits(32)] * L), p_src, R()),
            Source(dut.b_valid, dut.b_ready, b, drive_b, None, p_src, R()),
            Source(dut.p_valid, dut.p_ready, p, drive_p, None, p_src, R())]
    snks = [Sink(dut.d_valid, dut.d_ready, cap_d, p_snk, R()), Sink(dut.r_valid, dut.r_ready, cap_r, p_snk, R())]
    return srcs, snks, d, r, alu_sat, rq_sat


async def run_cmds(dut, cmds, rng, p_src=0.75, p_snk=0.75):
    srcs, snks, d, r, alu_sat, rq_sat = make_bench(dut, cmds, rng, p_src, p_snk)
    hooks = [SfuStub(dut, random.Random(rng.random()))] if HAS_SFU else []
    if not HAS_SFU:
        dut.sfu_in_ready.value = 0
        dut.sfu_out_valid.value = 0
    cyc = await run_bench(dut, srcs, snks, [len(d), len(r)], hooks=hooks)
    for s in srcs:
        assert s.done(), "a source stream was not fully consumed"
    for n, (g, e) in enumerate(zip(snks[0].got, d)):
        assert g == e, f"d beat {n}: got {g} exp {e}"
    for n, (g, e) in enumerate(zip(snks[1].got, r)):
        assert g == e, f"r result {n}: got {g} exp {e}"
    return cyc, alu_sat, rq_sat, sum(-(-c["len"] // L) for c in cmds)


@cocotb.test()
async def random_commands(dut):
    rng = random.Random(40 + L + 7 * HAS_SFU)
    await start(dut, INPUTS)
    cmds = [rand_cmd(rng) for _ in range(N_CMDS)]
    tot_alu = tot_rq = 0
    for k, (ps, pk) in enumerate(((0.75, 0.75), (1.0, 0.3), (0.4, 1.0), (1.0, 1.0))):
        part = cmds[k::4]
        _, sa, sr, _ = await run_cmds(dut, part, rng, ps, pk)
        tot_alu += sa
        tot_rq += sr
    await ReadOnly()
    assert u(dut.alu_sat_count) == tot_alu, (u(dut.alu_sat_count), tot_alu)
    assert u(dut.rq_sat_count) == tot_rq, (u(dut.rq_sat_count), tot_rq)
    assert u(dut.illegal_op) == int(any(is_illegal(c["op"]) for c in cmds))
    assert u(dut.busy) == 0
    dut._log.info(f"L={L} HAS_SFU={HAS_SFU}: {len(cmds)} commands, "
                  f"{sum(c['len'] for c in cmds)} elements, sat alu={tot_alu} rq={tot_rq}")


@cocotb.test()
async def back_to_back_throughput(dut):
    """Rows pipeline without draining. Documented cost model (docs/vector-unit.md):
    1 command-accept cycle per command, plus 3 cycles each time an ALU command follows a
    REQUANT command (ALU depth 3 < requant depth 6, outputs are in order)."""
    rng = random.Random(77)
    await start(dut, INPUTS)
    for seq in ([vr.OP_ADD, vr.OP_MUL, vr.OP_MAX, vr.OP_SHL],
                [vr.OP_ADD, vr.VOP_REQUANT, vr.OP_MUL, vr.OP_MAX],
                [vr.VOP_REQUANT]):
        cmds = []
        for i in range(40):
            c = rand_cmd(rng)
            c.update(op=seq[i % len(seq)], len=3 * L + (i % L) + 1)
            cmds.append(c)
        cyc, _, _, nb = await run_cmds(dut, cmds, rng, 1.0, 1.0)
        rq_to_alu = sum(1 for x, y in zip(cmds, cmds[1:]) if x["op"] == vr.VOP_REQUANT and y["op"] < 0x10)
        bound = nb + len(cmds) + 3 * rq_to_alu + 12      # +12: pipeline fill and bench tail
        assert cyc <= bound, (cyc, bound, nb)
        dut._log.info(f"L={L}: {nb} beats / {len(cmds)} commands in {cyc} cycles (bound {bound})")


@cocotb.test()
async def illegal_and_clear(dut):
    rng = random.Random(5)
    await start(dut, INPUTS)
    c = rand_cmd(rng)
    c.update(op=0x15, len=L + 1)
    await run_cmds(dut, [c], rng)
    await ReadOnly()
    assert u(dut.illegal_op) == 1
    await FallingEdge(dut.clk)
    dut.stat_clr.value = 1
    await FallingEdge(dut.clk)
    dut.stat_clr.value = 0
    await ReadOnly()
    assert u(dut.illegal_op) == 0 and u(dut.alu_sat_count) == 0 and u(dut.rq_sat_count) == 0
