"""cocotb bench for rtl/sfu_vec.sv: vec_unit with the real SFU block (rtl/sfu.sv) on its
hook. Random command mixes of ALU, REQUANT, REDUCE and every SFU sub-op (EXP2, RECIP,
RSQRT, GELU, SOFTMAX, LN, reserved) with random valid gaps and backpressure on all streams.
Tables and LN parameters are written through the cfg port from the tiny bundle.
Reference: nanoxl.vec_ref (vector ops) and nanoxl.quant.ops (C5 ops)."""
from __future__ import annotations
import random

import cocotb
import numpy as np
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl import numerics as nx
from nanoxl import vec_ref as vr
from sfu_common_cocotb import (OP_EXP2, OP_GELU, OP_LN, OP_RECIP, OP_RSQRT, OP_SOFTMAX, R_EXP2,
                               R_GELU, R_LN_C, R_LN_GB, R_RECIP, R_RSQRT, Sink, Source,
                               bundle_luts, cfg_addr, env_int, ln_ref, ln_slots, pack, ref_elem,
                               run_bench, softmax_ref, start, u, unpack)

L = env_int("L", 4)
N_CMDS = env_int("N_CMDS", 150)
ACC_W = vr.REDUCE_ACC_W
M32 = 0xFFFFFFFF
INPUTS = ["cmd_valid", "cmd_op", "cmd_len", "cmd_fmt", "cmd_shamt", "cmd_lo", "cmd_hi",
          "cmd_b_imm", "cmd_imm_b", "cmd_p_imm", "cmd_mult", "a_valid", "a_data", "b_valid",
          "b_data", "p_valid", "p_mult", "p_shift", "d_ready", "r_ready", "cfg_we", "cfg_addr",
          "cfg_wdata", "stat_clr"]


async def load_cfg(dut):
    exp_lut, recip_lut, rsqrt_lut, gel = bundle_luts()
    w = []
    for reg, t in ((R_EXP2, exp_lut), (R_RECIP, recip_lut), (R_RSQRT, rsqrt_lut)):
        w += [(cfg_addr(reg, 0, i), int(t[i])) for i in range(256)]
    for s, t in enumerate(gel):
        w += [(cfg_addr(R_GELU, s, i), int(t[i]) & 0xFF) for i in range(256)]
    for s, (_, g, b, c) in enumerate(ln_slots()):
        w += [(cfg_addr(R_LN_GB, s, i), (int(g[i]) & 0xFFFF) | ((int(b[i]) & M32) << 16))
              for i in range(len(g))]
        w += [(cfg_addr(R_LN_C, s, k), int(c[k])) for k in range(5)]
    for a, d in w:
        await FallingEdge(dut.clk)
        dut.cfg_we.value = 1
        dut.cfg_addr.value = a
        dut.cfg_wdata.value = d
    await FallingEdge(dut.clk)
    dut.cfg_we.value = 0
    for _ in range(3):
        await FallingEdge(dut.clk)


def rand_val(rng):
    r = rng.random()
    if r < 0.3:
        return rng.randrange(nx.INT32_MIN, nx.INT32_MAX + 1)
    if r < 0.7:
        return rng.randrange(-32768, 32768)
    return rng.randrange(-300, 300)


def rand_cmd(rng):
    n_ln = len(ln_slots())
    d = len(ln_slots()[0][1])
    ops = [0, 1, 3, 7, 9, vr.VOP_REQUANT, vr.VOP_REDUCE] + [0x18 + k for k in range(7)] + \
        [0x18 + OP_SOFTMAX, 0x18 + OP_LN, 0x18 + OP_GELU]
    op = rng.choice(ops)
    sub = op - 0x18 if op >= 0x18 else None
    r = rng.random()
    n = (0 if r < 0.03 else rng.randrange(1, L + 1) if r < 0.3 else rng.randrange(L + 1, 5 * L + 4)
         if r < 0.9 else rng.randrange(1, 257))
    imm = rand_val(rng)
    if sub == OP_LN:
        n = d if rng.random() < 0.8 else rng.randrange(1, d + 1)
        imm = rng.randrange(n_ln)
    if sub == OP_GELU:
        imm = rng.randrange(4)
    lo, hi = sorted(rng.randrange(-500, 500) for _ in range(2))
    return {"op": op, "len": n, "fmt": rng.randrange(4), "shamt": rng.randrange(32), "lo": lo,
            "hi": hi, "b_imm": int(rng.random() < 0.3), "imm_b": imm,
            "p_imm": int(rng.random() < 0.3), "mult": rng.randrange(1 << 30, 1 << 31)}


def sfu_row(sub, n, rng):
    if sub == OP_EXP2:
        return [rng.randrange(65536) for _ in range(n)]
    if sub == OP_RECIP:
        return [rng.randrange(1 << 15, 1 << rng.randrange(16, 32)) for _ in range(n)]
    if sub == OP_RSQRT:
        return [rng.randrange(0, 1 << rng.randrange(1, 32)) for _ in range(n)]
    if sub == OP_GELU:
        return [rng.randrange(-128, 128) for _ in range(n)]
    if sub == OP_SOFTMAX:
        sig = rng.choice([50, 500, 3000])
        return [int(max(-32768, min(32767, round(rng.gauss(0, sig))))) for _ in range(n)]
    if sub == OP_LN:
        sig = rng.choice([30, 1000, 8000])
        return [int(max(-32768, min(32767, round(rng.gauss(0, sig))))) for _ in range(n)]
    return [rand_val(rng) for _ in range(n)]


def build(cmds, rng):
    exp_lut, recip_lut, rsqrt_lut, gel = bundle_luts()
    slots = ln_slots()
    a, b, p, d, r = [], [], [], [], []
    cnt = dict(alu=0, rq=0, z=0, o=0, illegal=0)
    for c in cmds:
        n, op = c["len"], c["op"]
        if n == 0:
            continue
        nb = -(-n // L)
        sub = op - 0x18 if op >= 0x18 else None
        unit = "sfu" if sub is not None else "rq" if op == vr.VOP_REQUANT else \
            "red" if op == vr.VOP_REDUCE else "alu"
        row_a = sfu_row(sub, n, rng) if unit == "sfu" else [rand_val(rng) for _ in range(n)]
        need_b = not c["b_imm"] and (unit == "rq" or (unit == "alu" and op in vr.BINARY_OPS))
        need_p = not c["p_imm"] and unit == "rq"
        row_b = [rand_val(rng) for _ in range(n)] if need_b else [c["imm_b"]] * n
        if unit == "rq":
            row_a = [rng.randrange(-(1 << 22), 1 << 22) for _ in row_a]
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
            out, sat = vr.alu(op, row_a, row_b, c["fmt"], c["shamt"], c["lo"], c["hi"])
            cnt["alu"] += int(sat.sum())
        elif unit == "rq":
            bits = 8 if c["fmt"] == 0 else 16
            out = vr.vec_unit_op(vr.VOP_REQUANT, row_a, fmt=c["fmt"], bias=row_b, mult=mult, shift=shift)
            x = (np.array(row_a, np.int64) + np.array(row_b, np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)
            y = nx.rdivpot(nx.srdhm(x, mult), shift).astype(np.int64)
            cnt["rq"] += int(((y < -(1 << (bits - 1))) | (y > (1 << (bits - 1)) - 1)).sum())
        elif sub == OP_SOFTMAX:
            out = softmax_ref(row_a, np.ones(n, bool))
        elif sub == OP_LN:
            _, g, bb, cc = slots[c["imm_b"]]
            out, nz, no = ln_ref(np.array(row_a), g[:n], bb[:n], cc, rsqrt_lut)
            cnt["z"] += nz
            cnt["o"] += no
        elif sub in (OP_EXP2, OP_RECIP, OP_RSQRT, OP_GELU):
            out = ref_elem(sub, row_a, exp_lut, recip_lut, rsqrt_lut,
                           gel[c["imm_b"]] if sub == OP_GELU else None)
        else:
            out = row_a
            cnt["illegal"] = 1
        out = [int(v) for v in out]
        for j in range(nb):
            seg = out[j * L:(j + 1) * L]
            k = len(seg)
            d.append((seg + [0] * (L - k), [1] * k + [0] * (L - k), int(j == nb - 1)))
    return a, b, p, d, r, cnt


async def run_cmds(dut, cmds, rng, p_src, p_snk):
    a, b, p, d, r, cnt = build(cmds, rng)
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
    await run_bench(dut, srcs, snks, [len(d), len(r)])
    for i, (g, e) in enumerate(zip(snks[0].got, d)):
        assert g == e, f"D beat {i}: got {g} expected {e}"
    assert snks[1].got == r
    for s in srcs:
        assert s.done()
    await ReadOnly()
    assert dut.busy.value == 0
    assert dut.sfu_err_len.value == 0
    assert u(dut.alu_sat_count) == cnt["alu"] and u(dut.rq_sat_count) == cnt["rq"]
    assert (u(dut.ln_z_sat_count), u(dut.ln_out_sat_count)) == (cnt["z"], cnt["o"])
    assert dut.illegal_op.value == cnt["illegal"]
    await FallingEdge(dut.clk)
    return cnt


@cocotb.test()
async def sfu_vec_random(dut):
    await start(dut, INPUTS)
    await load_cfg(dut)
    rng = random.Random(1000 + L)
    for p_src, p_snk in ((0.8, 0.8), (0.5, 0.95), (0.95, 0.4)):
        dut.stat_clr.value = 1
        await FallingEdge(dut.clk)
        dut.stat_clr.value = 0
        cmds = [rand_cmd(rng) for _ in range(N_CMDS // 3)]
        cnt = await run_cmds(dut, cmds, rng, p_src, p_snk)
        dut._log.info(f"L={L}: {len(cmds)} commands ok {cnt}")


@cocotb.test()
async def sfu_vec_layer_rows(dut):
    """A decoder-layer-like sequence: LN (every bundle slot), GELU (every table), softmax
    rows of every length 1..CTX in a stride, back to back with no gaps."""
    await start(dut, INPUTS)
    await load_cfg(dut)
    rng = random.Random(77)
    d = len(ln_slots()[0][1])
    base = dict(fmt=1, shamt=0, lo=0, hi=0, b_imm=1, p_imm=1, mult=1 << 30)
    cmds = []
    for s in range(len(ln_slots())):
        cmds.append(dict(base, op=0x18 + OP_LN, len=d, imm_b=s))
        cmds.append(dict(base, op=0x18 + OP_GELU, len=4 * d, imm_b=s % 4))
        cmds.append(dict(base, op=0x18 + OP_SOFTMAX, len=1 + 29 * s, imm_b=0))
    cmds.append(dict(base, op=0x18 + OP_SOFTMAX, len=256, imm_b=0))
    await run_cmds(dut, cmds, rng, 1.0, 1.0)
