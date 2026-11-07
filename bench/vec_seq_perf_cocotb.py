"""cocotb module for bench/vec_seq_perf.py: per-command occupancy of rtl/vec_seq.sv (the VEC engine).

Ports always ready, read latency LAT (default 2), commands back to back. Each shape runs as a pair
of identical commands (the second one is queued while the first runs), so

    occ   = done(2nd) - done(1st)     (engine busy per command, the cycle model's occupancy)
    first = done(1st) - issue          (accept -> done of a lone command, incl. the pipeline drain)

LN / GELU / LUT_LOAD / softmax pairs reuse the same parameter block, so the 2nd command is a
residency hit; the `*_miss` shapes alternate two distinct blocks so both commands miss. Every
command is still checked against funcsim by the shared harness. Results: JSON lines in $PERF_OUT.
"""
from __future__ import annotations

import json
import os
import random

import cocotb

from nanoxl import isa
from vec_seq_cocotb import CH, LINE, P_CTX, P_GELU, P_LUT, P_LUT2, P_REC, Gen, rec_bytes, rand_record, rup
from vec_seq_common_cocotb import TB, L

OUT = os.environ.get("PERF_OUT", "vec_seq_perf.jsonl")
LAT = int(os.environ.get("LAT", "2"))


def shapes(g: Gen):
    """(name, [cmd, cmd]) pairs; sources are filled once by the generators."""
    tb, rng = g.tb, g.rng
    out = []

    def requant(M, N, out16=0, pt=0):
        ob = 2 if out16 else 1
        ss, ds = rup(4 * N) // LINE, rup(N * ob) // LINE
        src = g.src(isa.ACC, M * ss * LINE)
        g.fill(isa.ACC, src, M * ss * LINE, "i32")
        dst = g.dst(isa.ABUF, M * ds * LINE)
        nrec = 1 if pt else N
        tb.write_mem(isa.PBUF, P_REC, rec_bytes([rand_record(rng) for _ in range(nrec)]))
        return isa.C("REQUANT", out16=out16, per_tensor=pt, src_addr=src, dst_addr=dst, param_addr=P_REC,
                     M=M, N=N, src_stride=ss, dst_stride=ds)

    for M, N, o16, pt in ((16, 384, 0, 0), (16, 128, 0, 0), (1, 384, 0, 0), (16, 256, 1, 0), (16, 256, 0, 1),
                          (4, 64, 0, 0), (1, 65, 0, 0)):
        c = requant(M, N, o16, pt)
        out.append((f"REQUANT M{M} N{N}{' out16' if o16 else ''}{' pt' if pt else ''}", [c, c], dict(c.f)))

    def softmax(rows, cols, causal=0, diag=0, ctx=0):
        ss, ds = rup(4 * cols) // LINE, rup(cols) // LINE
        src = g.src(isa.ACC, rows * ss * LINE)
        g.fill(isa.ACC, src, rows * ss * LINE, "i32")
        dst = g.dst(isa.ABUF, rows * ds * LINE)
        return isa.C("SOFTMAX_ONLINE", first=1, last=1, causal=causal, src_addr=src, dst_addr=dst,
                     ctx_addr=P_CTX + 16 * ctx, rows=rows, cols=cols, diag=diag, src_stride=ss, dst_stride=ds)

    for rows, cols, causal in ((16, 256, 0), (16, 64, 0), (1, 256, 0), (16, 16, 0), (4, 128, 1)):
        c = softmax(rows, cols, causal)
        out.append((f"SOFTMAX rows{rows} cols{cols}{' causal' if causal else ''}", [c, c], dict(c.f)))

    def ln(rows, i, in16=1):
        addr, cols = g.lnb[i]
        eb = 2 if in16 else 1
        ss, ds = rup(cols * eb) // LINE, rup(cols) // LINE
        src = g.src(isa.ABUF, rows * ss * LINE)
        g.fill(isa.ABUF, src, rows * ss * LINE, "i16" if in16 else "rand")
        dst = g.dst(isa.ABUF, rows * ds * LINE)
        return isa.C("LN", in16=in16, src_addr=src, dst_addr=dst, param_addr=addr, rows=rows, cols=cols,
                     src_stride=ss, dst_stride=ds)

    # lnb: 0..2 cols 128 (distinct blocks), 5 = D_MAX (384)
    for rows, i, in16 in ((1, 0, 1), (16, 0, 1), (16, 0, 0), (1, 5, 1), (16, 5, 1)):
        c = ln(rows, i, in16)
        out.append((f"LN rows{rows} cols{g.lnb[i][1]}{'' if in16 else ' in8'}", [c, c], dict(c.f)))
    for rows in (1, 16):
        a, b = ln(rows, 1), ln(rows, 2)
        f = dict(a.f)
        f["_resnoop"] = [g.lnb[1][0], g.lnb[2][0]]     # snooped rewrite (same bytes) -> both miss
        out.append((f"LN rows{rows} cols128 miss", [a, b], f))

    def gelu(n, t=0):
        src = g.src(isa.ABUF, n)
        g.fill(isa.ABUF, src, n)
        return isa.C("GELU", src_addr=src, dst_addr=g.dst(isa.ABUF, n), lut_addr=P_GELU + 256 * t, len=n)

    for n in (512, 128, 2048):
        c = gelu(n)
        out.append((f"GELU len{n}", [c, c], dict(c.f)))
    out.append(("GELU len512 miss", [gelu(512, 1), gelu(512, 2)], dict(len=512)))

    def vec(vop, dt, n):
        eb = isa.DT_BYTES[dt]
        a0, a1 = g.src(isa.ABUF, n * eb), g.src(isa.ACC, n * eb)
        g.fill(isa.ABUF, a0, n * eb)
        g.fill(isa.ACC, a1, n * eb)
        return isa.C("VEC", vop=vop, dt_in=dt, dt_out=dt, src0_buf=isa.ABUF, src0_addr=a0, src1_buf=isa.ACC,
                     src1_addr=a1, dst_buf=isa.ABUF, dst_addr=g.dst(isa.ABUF, n * eb), len=n, imm=0)

    for vop, dt, n in ((1, 0, 2048), (1, 1, 2048), (1, 2, 1024), (0, 0, 2048), (1, 0, 128), (1, 0, 64)):
        c = vec(vop, dt, n)
        out.append((f"VEC {'ADD' if vop else 'COPY'} {('int8', 'int16', 'int32')[dt]} len{n}", [c, c], dict(c.f)))

    def kv(rows, H, dh):
        ss = rup(H * dh) // LINE
        src = g.src(isa.ABUF, rows * ss * LINE)
        g.fill(isa.ABUF, src, rows * ss * LINE)
        ctx = 64
        return isa.C("KV_APPEND", src_addr=src, dst_addr=g.dst(isa.KVBUF, H * ctx * dh), pos=0, heads=H, dh=dh,
                     ctx=ctx, rows=rows, src_stride=ss)

    for rows, H, dh in ((16, 4, 32), (1, 4, 32), (16, 6, 64), (1, 6, 64)):
        c = kv(rows, H, dh)
        out.append((f"KV_APPEND rows{rows} heads{H} dh{dh}", [c, c], dict(c.f)))
    lut_a = isa.C("LUT_LOAD", table=1, src_addr=P_LUT + 512)
    out.append(("LUT_LOAD hit", [lut_a, lut_a], {}))
    out.append(("LUT_LOAD miss", [isa.C("LUT_LOAD", table=1, src_addr=P_LUT2 + 512), lut_a], {}))
    return out


@cocotb.test()
async def vec_seq_perf(dut):
    tb = TB(dut, seed=7, p_ar=1.0, p_w=1.0, p_cmd=1.0, lat=LAT)
    await tb.start()
    g = Gen(tb, random.Random(8))
    g.setup()
    await tb.drain()
    res = []
    for name, cmds, f in shapes(g):
        for addr in f.pop("_resnoop", []):
            nb = 2400
            tb.write_mem(isa.PBUF, addr, tb.mem[isa.PBUF][addr:addr + nb].tobytes())
        await tb.drain()
        c0 = tb.cyc
        n0 = len(tb.done_cycles)
        for c in cmds:
            tb.add(c)
        await tb.drain()
        tb.compare_all()
        d = tb.done_cycles[n0:]
        r = dict(name=name, op=cmds[0].op, L=L, lat=LAT, first=d[0] - c0, occ=d[1] - d[0], f=f)
        dut._log.info(f"{name}: first {r['first']} occ {r['occ']}")
        res.append(r)
    assert tb.err() == (0, 0)
    with open(OUT, "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
