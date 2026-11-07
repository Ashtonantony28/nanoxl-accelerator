"""cocotb benches for rtl/vec_seq.sv, the VEC engine (ISA VEC / REQUANT / LN / SOFTMAX_ONLINE /
GELU / KV_APPEND / LUT_LOAD on vec_unit + SFU). Reference: nanoxl.funcsim (see
vec_seq_common_cocotb for the harness and the checks)."""
from __future__ import annotations

import math
import os
import random

import cocotb
import numpy as np

from nanoxl import isa
from nanoxl import numerics as nx
from nanoxl.export import load_bundle
from nanoxl.funcsim import ln_block, ln_block_layout
from nanoxl.quant.ops import ln_affine_consts
from vec_seq_common_cocotb import LINE, TB, L, u, word

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLE = os.path.join(ROOT, "artifacts/models/tiny/w8a8.npz")
N_CMDS = int(os.environ.get("N_CMDS", "40"))
D_MAX = int(os.environ.get("D_MAX", "384"))
CTX_MAX = int(os.environ.get("CTX_MAX", "256"))
RQ_MAX = int(os.environ.get("RQ_MAX", "256"))
CH = 4096
KiB = 1024

_B = {}


def bundle():
    if "a" not in _B:
        _B["a"] = load_bundle(BUNDLE)
    return _B["a"]


def rup(x, a=LINE):
    return (x + a - 1) // a * a


# ------------------------------------------------------------------ PBUF layout (random tests)
P_LUT = 0                 # 3 x 512: exp2, recip, rsqrt
P_LUT2 = 1536             # a second copy of the three tables (same bytes, other address)
P_GELU = 4 * KiB          # 6 GELU tables (more than NG = 4 slots)
P_CTX = 6 * KiB           # 10 score records (more than NCTX = 8 slots)
P_LN = 8 * KiB            # LN blocks, 2400 B apart
P_LN_SPARE = 24 * KiB     # spare variants of the LN blocks (sources of VEC rewrites)
P_REC = 32 * KiB          # requant records (fresh per command)
P_DST = 48 * KiB          # VEC destinations in PBUF
LN_STEP = 2400
N_GELU, N_CTX = 6, 10


def rand_ln_params(rng, cols):
    """C5 LN constants in the bit-exact domain (as tests/rtl/layernorm_cocotb.make_slots)."""
    kind = rng.randrange(4)
    d_c = max(cols, rng.choice([cols, D_MAX, rng.randrange(cols, 4097)]))
    mu_mult, mu_shift = nx.quantize_multiplier(1.0 / d_c)
    d_s = 4096 if kind == 2 else rng.choice([cols, rng.randrange(2, 4097)])
    sqrt_d = min(65535, int(round(math.sqrt(d_s) * 1024)))
    eps_q = rng.choice([0, 1, rng.randrange(0, 1 << 16), rng.randrange(0, 1 << 30)])
    if kind == 3:
        sh = rng.randrange(1, 7)
        g = [rng.randrange(-2, 3) for _ in range(cols)]
        b = [rng.randrange(-(20 << sh), 20 << sh) for _ in range(cols)]
    elif kind == 0:
        g, b, sh = ln_affine_consts(np.random.default_rng(rng.randrange(1 << 30)).normal(1, 0.3, cols),
                                    np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.2, cols),
                                    rng.choice([0.01, 0.03, 0.1]))
    else:
        sh = rng.randrange(0, 32) if kind == 1 else rng.randrange(8, 20)
        g = [rng.randrange(-32768, 32768) for _ in range(cols)]
        b = [rng.randrange(-(1 << 31), 1 << 31) >> rng.randrange(0, 20) for _ in range(cols)]
    return [mu_mult, mu_shift, sqrt_d, eps_q, int(sh)], np.asarray(g, np.int64), np.asarray(b, np.int64)


def ln_variant(rng, cols, bundle_i=None):
    A, meta = bundle()
    if bundle_i is not None:
        names = [f"h{l}.{n}" for l in range(meta["n_layer"]) for n in ("ln_1", "ln_2")] + ["ln_f"]
        n = names[bundle_i % len(names)]
        return ln_block([int(v) for v in A[n + ".c"]], A[n + ".g"], A[n + ".b"])
    c, g, b = rand_ln_params(rng, cols)
    return ln_block(c, g, b)


def rand_record(rng, score=False):
    if score:
        A, meta = bundle()
        if rng.random() < 0.5:
            l = rng.randrange(meta["n_layer"])
            return [0, int(A[f"h{l}.score.mult"][0]), int(A[f"h{l}.score.shift"][0]), int(A[f"h{l}.score.ls"][0])]
        return [rng.randrange(-(1 << 20), 1 << 20), rng.randrange(1 << 30, 1 << 31), rng.randrange(0, 32),
                rng.randrange(0, 4)]
    bias = rng.randrange(-(1 << 31), 1 << 31) if rng.random() < 0.1 else rng.randrange(-(1 << 20), 1 << 20)
    return [bias, rng.randrange(1 << 30, 1 << 31), rng.randrange(0, 32), 0]


def rec_bytes(recs):
    return np.array(recs, np.int64).astype(np.int32).tobytes()


class Gen:
    """Random legal VEC-queue commands over the PBUF layout above; sources in the low half
    of each buffer, destinations in the high half (so no command reads what it writes,
    except the deliberate in-place forms)."""

    def __init__(self, tb, rng):
        self.tb, self.rng = tb, rng
        A, meta = bundle()
        self.lnb = []        # (addr, cols)
        cols_list = [128, 128, 128, rng.randrange(1, 40), rng.randrange(40, D_MAX + 1), D_MAX, L + 1]
        for i, cols in enumerate(cols_list):
            addr = P_LN + i * LN_STEP
            assert ln_block_layout(cols)[2] <= LN_STEP
            self.lnb.append((addr, cols))

    def setup(self):
        """Tables and parameter pools into PBUF; LUT_LOADs."""
        tb, rng = self.tb, self.rng
        A, meta = bundle()
        luts = b"".join(np.asarray(A[k], np.uint16).tobytes() for k in ("lut.exp2", "lut.recip", "lut.rsqrt"))
        tb.write_mem(isa.PBUF, P_LUT, luts)
        tb.write_mem(isa.PBUF, P_LUT2, luts)
        for i in range(N_GELU):
            t = A[f"h{i}.gelu.lut"] if i < meta["n_layer"] else np.array(
                [rng.randrange(-128, 128) for _ in range(256)], np.int8)
            tb.write_mem(isa.PBUF, P_GELU + 256 * i, np.asarray(t, np.int8).tobytes())
        for i in range(N_CTX):
            tb.write_mem(isa.PBUF, P_CTX + 16 * i, rec_bytes([rand_record(rng, score=True)]))
        for i, (addr, cols) in enumerate(self.lnb):
            tb.write_mem(isa.PBUF, addr, ln_variant(rng, cols, i if cols == 128 else None))
            tb.write_mem(isa.PBUF, P_LN_SPARE + i * LN_STEP, ln_variant(rng, cols, i + 3 if cols == 128 else None))
        for t in range(3):
            tb.add(isa.C("LUT_LOAD", table=t, src_addr=P_LUT + 512 * t))

    # -- region helpers
    def src(self, buf, nbytes):
        half = len(self.tb.mem[buf]) // 2
        if buf == isa.PBUF:          # parameters / tables area (read only here)
            half = 32 * KiB
        assert nbytes <= half
        return self.rng.randrange(0, (half - nbytes) // LINE + 1) * LINE

    def dst(self, buf, nbytes):
        if buf == isa.PBUF:
            lo, size = P_DST, 16 * KiB
        else:
            size = len(self.tb.mem[buf]) // 2
            lo = size
        assert nbytes <= size
        return lo + self.rng.randrange(0, (size - nbytes) // LINE + 1) * LINE

    def fill(self, buf, addr, nbytes, kind="rand"):
        rng = self.rng
        if kind == "i32":
            big = rng.random() < 0.3
            c = rng.randrange(-(1 << 24), 1 << 24)
            v = np.array([rng.randrange(-(1 << 31), 1 << 31) if big else c + rng.randrange(-(1 << 18), 1 << 18)
                          for _ in range(nbytes // 4)], np.int64).astype(np.int32)
            self.tb.write_mem(buf, addr, v.tobytes())
        elif kind == "i16":
            sig = rng.choice([5, 200, 2000, 9000, 20000])
            mu = rng.choice([0, rng.randrange(-8000, 8000)])
            v = np.clip(np.random.default_rng(rng.randrange(1 << 30)).normal(mu, sig, nbytes // 2), -32768, 32767)
            if rng.random() < 0.2:
                v = np.array([rng.randrange(-32768, 32768) for _ in range(nbytes // 2)])
            self.tb.write_mem(buf, addr, np.round(v).astype(np.int16).tobytes())
        else:
            self.tb.write_mem(buf, addr, bytes(rng.getrandbits(8) for _ in range(nbytes)))

    # -- command generators
    def vec(self):
        rng = self.rng
        vop = rng.randrange(9)
        dt_in, dt_out = rng.randrange(3), rng.randrange(3)
        r = rng.random()
        n = rng.randrange(1, 5) if r < 0.15 else rng.randrange(CH - 3, CH + 2 * L + 40) if r < 0.2 else \
            rng.randrange(1, 300)
        if vop in (7, 8):
            imm = rng.randrange(32)
        else:
            imm = rng.randrange(-32768, 32768) if rng.random() < 0.5 else rng.randrange(-300, 300)
        bufs = [rng.choice([0, 0, 1, 2, 3, 4]) for _ in range(3)]
        eb_i, eb_o = isa.DT_BYTES[dt_in], isa.DT_BYTES[dt_out]
        if n * eb_o > 16 * KiB and bufs[2] == isa.PBUF:
            bufs[2] = isa.ABUF
        a0 = self.src(bufs[0], n * eb_i)
        a1 = self.src(bufs[1], n * eb_i)
        for b_, a_ in ((bufs[0], a0), (bufs[1], a1)):
            if b_ != isa.PBUF:
                self.fill(b_, a_, n * eb_i)
        d = self.dst(bufs[2], n * eb_o)
        if rng.random() < 0.15 and bufs[0] != isa.PBUF:     # in place
            dt_out, bufs[2], d = dt_in, bufs[0], a0
        return isa.C("VEC", vop=vop, dt_in=dt_in, dt_out=dt_out, src0_buf=bufs[0], src0_addr=a0,
                     src1_buf=bufs[1], src1_addr=a1, dst_buf=bufs[2], dst_addr=d, len=n, imm=imm)

    def requant(self):
        rng = self.rng
        M = rng.randrange(1, 5)
        N = rng.choice([rng.randrange(1, 40), rng.randrange(1, 300), rng.randrange(RQ_MAX - 3, RQ_MAX + 40), 128])
        out16 = rng.randrange(2)
        pt = int(rng.random() < 0.3)
        ob = 2 if out16 else 1
        ss = rup(4 * N) // LINE + rng.randrange(3)
        ds = rup(N * ob) // LINE + rng.randrange(3)
        src = self.src(isa.ACC, M * ss * LINE)
        self.fill(isa.ACC, src, M * ss * LINE, "i32")
        dst = self.dst(isa.ABUF, M * ds * LINE)
        nrec = 1 if pt else N
        pa = P_REC + rup(rng.randrange(0, 16 * KiB - 16 * nrec + 1))
        self.tb.write_mem(isa.PBUF, pa, rec_bytes([rand_record(rng) for _ in range(nrec)]))
        return isa.C("REQUANT", out16=out16, per_tensor=pt, src_addr=src, dst_addr=dst, param_addr=pa,
                     M=M, N=N, src_stride=ss, dst_stride=ds)

    def ln(self):
        rng = self.rng
        addr, cols = rng.choice(self.lnb)
        rows = rng.randrange(1, 4)
        in16 = int(rng.random() < 0.8)
        eb = 2 if in16 else 1
        ss = rup(cols * eb) // LINE + rng.randrange(2)
        ds = rup(cols) // LINE + rng.randrange(2)
        src = self.src(isa.ABUF, rows * ss * LINE)
        self.fill(isa.ABUF, src, rows * ss * LINE, "i16" if in16 else "rand")
        dst = self.dst(isa.ABUF, rows * ds * LINE)
        return isa.C("LN", in16=in16, src_addr=src, dst_addr=dst, param_addr=addr, rows=rows, cols=cols,
                     src_stride=ss, dst_stride=ds)

    def softmax(self):
        rng = self.rng
        rows = rng.randrange(1, 4)
        cols = rng.choice([1, L, L + 1, rng.randrange(1, 140), rng.randrange(1, CTX_MAX + 1)])
        causal = int(rng.random() < 0.6)
        diag = rng.randrange(0, cols + 2)
        ss = rup(4 * cols) // LINE + rng.randrange(2)
        ds = rup(cols) // LINE + rng.randrange(2)
        src = self.src(isa.ACC, rows * ss * LINE)
        self.fill(isa.ACC, src, rows * ss * LINE, "i32")
        dst = self.dst(isa.ABUF, rows * ds * LINE)
        ctx = P_CTX + 16 * rng.randrange(N_CTX)
        return isa.C("SOFTMAX_ONLINE", first=1, last=1, causal=causal, src_addr=src, dst_addr=dst, ctx_addr=ctx,
                     rows=rows, cols=cols, diag=diag, src_stride=ss, dst_stride=ds)

    def gelu(self):
        rng = self.rng
        n = rng.randrange(CH - 2, CH + 70) if rng.random() < 0.08 else rng.randrange(1, 600)
        src = self.src(isa.ABUF, n)
        self.fill(isa.ABUF, src, n)
        dst = src if rng.random() < 0.3 else self.dst(isa.ABUF, n)
        return isa.C("GELU", src_addr=src, dst_addr=dst, lut_addr=P_GELU + 256 * rng.randrange(N_GELU), len=n)

    def kv(self):
        rng = self.rng
        H = rng.randrange(1, 5)
        dh = rng.choice([16, 32, 48, 64])
        rows = rng.randrange(1, 5)
        ctx = rng.randrange(rows, 100)
        pos = rng.randrange(0, ctx - rows + 1)
        ss = rup(H * dh) // LINE + rng.randrange(3)
        src = self.src(isa.ABUF, rows * ss * LINE)
        self.fill(isa.ABUF, src, rows * ss * LINE)
        dst = self.dst(isa.KVBUF, H * ctx * dh)
        return isa.C("KV_APPEND", src_addr=src, dst_addr=dst, pos=pos, heads=H, dh=dh, ctx=ctx, rows=rows,
                     src_stride=ss)

    def lut(self):
        t = self.rng.randrange(3)
        return isa.C("LUT_LOAD", table=t, src_addr=self.rng.choice([P_LUT, P_LUT2]) + 512 * t)

    def ln_rewrite_by_vec(self):
        """VEC COPY a spare LN block over a pool block (PBUF write by the engine itself)."""
        i = self.rng.randrange(len(self.lnb))
        addr, cols = self.lnb[i]
        n = ln_block_layout(cols)[2]
        return isa.C("VEC", vop=0, dt_in=0, dt_out=0, src0_buf=isa.PBUF, src0_addr=P_LN_SPARE + i * LN_STEP,
                     src1_buf=0, src1_addr=0, dst_buf=isa.PBUF, dst_addr=addr, len=n, imm=0)

    def rewrite_pool(self):
        """Between batches: the 'DMA' rewrites a parameter block in place (snooped)."""
        rng, tb = self.rng, self.tb
        k = rng.randrange(3)
        if k == 0:
            i = rng.randrange(len(self.lnb))
            addr, cols = self.lnb[i]
            tb.write_mem(isa.PBUF, addr, ln_variant(rng, cols, rng.randrange(9) if cols == 128 else None))
        elif k == 1:
            t = np.array([rng.randrange(-128, 128) for _ in range(256)], np.int8)
            tb.write_mem(isa.PBUF, P_GELU + 256 * rng.randrange(N_GELU), t.tobytes())
        else:
            tb.write_mem(isa.PBUF, P_CTX + 16 * rng.randrange(N_CTX), rec_bytes([rand_record(rng, score=True)]))

    def any(self):
        only = os.environ.get("VS_ONLY")
        if only:
            return getattr(self, only)()
        r = self.rng.random()
        if r < 0.25:
            return self.vec()
        if r < 0.42:
            return self.requant()
        if r < 0.58:
            return self.ln()
        if r < 0.74:
            return self.softmax()
        if r < 0.86:
            return self.gelu()
        if r < 0.94:
            return self.kv()
        if r < 0.97:
            return self.lut()
        return self.ln_rewrite_by_vec()


async def run_random(dut, seed, n_cmds, **tbkw):
    tb = TB(dut, seed=seed, **tbkw)
    await tb.start()
    rng = random.Random(seed + 100)
    g = Gen(tb, rng)
    g.setup()
    await tb.drain()
    left = n_cmds
    while left > 0:
        k = min(left, rng.randrange(1, 5))
        cmds = [g.any() for _ in range(k)]     # all source fills happen before any execution
        for c in cmds:
            tb.add(c)
        left -= k
        await tb.drain()
        tb.compare_all()
        if rng.random() < 0.4:
            g.rewrite_pool()
    dut._log.info(f"{n_cmds} random commands in {tb.cyc} cycles; loader hits {u(dut.st_hits)}"
                  f" misses {u(dut.st_misses)} lines {u(dut.st_lines)}")
    assert u(dut.err) == 0
    return tb


@cocotb.test()
async def vec_seq_smoke(dut):
    """One of each command, full-rate ports."""
    tb = TB(dut, seed=3, p_ar=1.0, p_w=1.0, p_cmd=1.0, lat=1)
    await tb.start()
    g = Gen(tb, random.Random(5))
    g.setup()
    for c in [f() for f in (g.vec, g.requant, g.ln, g.softmax, g.gelu, g.kv, g.lut)]:
        tb.add(c)
    await tb.drain()
    tb.compare_all()
    assert tb.err() == (0, 0)


@cocotb.test()
async def vec_seq_random(dut):
    """Random legal commands with random port latency / ready, command gaps, pool rewrites."""
    await run_random(dut, seed=11, n_cmds=N_CMDS)


@cocotb.test()
async def vec_seq_random_fast_ports(dut):
    """Same with always-ready ports and back-to-back commands (pipelined issue)."""
    await run_random(dut, seed=12, n_cmds=max(10, N_CMDS // 2), p_ar=1.0, p_w=1.0, p_cmd=1.0, lat=2)
