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


# ------------------------------------------------------------------ compiled tiny model
async def run_tiny(dut, steps, snoop="dma", seed=21, **tbkw):
    """Every VEC-queue command the compiler emits for the tiny W8A8 bundle, for the given steps
    [(pos, n)] (prefill + decode), executed on the RTL with all other commands (DMA, GEMM) done
    by funcsim in stream order. Consecutive VEC commands run as one pipelined batch; before a
    batch the scratchpads are synced from funcsim. ``snoop``: "dma" reports every PBUF line a
    DMA_LOAD wrote (what the hardware sees), "changed" only lines whose bytes changed (a
    compiler that keeps parameter blocks resident)."""
    from nanoxl.funcsim import HostRunner
    from nanoxl.hwconfig import KV260
    A, meta = bundle()
    runner = HostRunner(A, meta, KV260, name="w8a8")
    tb = TB(dut, seed=seed, **tbkw)
    tb.fs = runner.sim
    tb.mem = {b: runner.sim.mem[b].copy() for b in tb.mem}
    await tb.start()
    rng = random.Random(seed)
    counts = {}
    for pos, n in steps:
        prog = runner.comp.compile_step([(pos, n)])
        toks = [rng.randrange(runner.cfg.vocab) for _ in range(n)]
        runner.sim.write_input(prog, "x", runner.embed(toks, pos))
        batch, dma_lines = False, []
        for c in prog.cmds:
            if c.queue == isa.Q_VEC and c.op not in ("WAIT", "SIGNAL"):
                if not batch:
                    tb.sync_to_rtl("changed" if snoop == "changed" else dma_lines)
                    dma_lines, batch = [], True
                tb.add(c, note=c.note)
                counts[c.op] = counts.get(c.op, 0) + 1
                continue
            if batch and any(a[0] in BUFS_SP and a[1] for a in accesses(c)):
                await tb.drain()
                tb.compare_all()
                batch = False
            if c.op == "DMA_LOAD" and c["sp_buf"] == isa.PBUF:
                a0, nb = c["sp_addr"], c["rows"] * c["row_bytes"]
                dma_lines += list(range(a0 // LINE, (a0 + nb - 1) // LINE + 1))
            runner.sim.step(c)
        if batch:
            await tb.drain()
            tb.compare_all()
        runner.pos[0] += n
    golden_sat = runner.sim.sat_count
    dut._log.info(f"tiny {steps}: VEC-queue commands {counts}; {tb.cyc} cycles; loader hits "
                  f"{u(dut.st_hits)} misses {u(dut.st_misses)} lines {u(dut.st_lines)}; sat {tb.sat}; "
                  f"funcsim per-op sat {dict((k, v) for k, v in golden_sat.items() if v)}")
    assert u(dut.err) == 0
    return tb


BUFS_SP = (isa.ABUF, isa.WBUF, isa.ACC, isa.PBUF, isa.KVBUF)
from nanoxl.funcsim import accesses  # noqa: E402


@cocotb.test()
async def vec_seq_tiny_forward(dut):
    """Tiny W8A8: prefill of TINY_T tokens + one decode step, every VEC-queue command."""
    t = int(os.environ.get("TINY_T", "6"))
    await run_tiny(dut, [(0, t), (t, 1)], snoop=os.environ.get("TINY_SNOOP", "dma"))


# ------------------------------------------------------------------ errors, clear, residency
@cocotb.test()
async def vec_seq_errors(dut):
    """Illegal commands are skipped (no writes), still pulse done, raise err with the first cause;
    clr clears it."""
    tb = TB(dut, seed=7)
    await tb.start()
    g = Gen(tb, random.Random(8))

    async def expect_err(w, code, unchecked=False):
        tb.add_raw(w, unchecked)
        await tb.drain()
        assert tb.err() == (1, code), (tb.err(), code)
        await tb.clear()
        assert tb.err() == (0, 0)

    tb.clear = lambda: clear(tb)
    # LN / SOFTMAX before the tables are loaded
    g.setup()
    tb.cmdq.clear()
    tb.expect.clear()
    tb.fs.sfu.clear()
    c = g.ln()
    await expect_err(word(c), 6)
    await expect_err(word(isa.C("GEMM", M=1, N=16, K=16)), 1)
    await expect_err(word(isa.C("VEC", vop=7, len=4, imm=40)), 3)
    await expect_err(word(isa.C("VEC", vop=8, len=4, imm=-1)), 3)
    await expect_err(word(isa.C("KV_APPEND", heads=1, dh=24, ctx=8, rows=1)), 7)
    await expect_err(word(isa.C("KV_APPEND", heads=1, dh=16, ctx=8, rows=4, pos=5)), 7)
    await expect_err(int.from_bytes(isa.encode(isa.C("VEC", vop=1, len=4)), "little") | (9 << 6), 2)
    await expect_err(word(isa.C("LUT_LOAD", table=0)) | (3 << 6), 2)
    for t in range(3):
        tb.add(isa.C("LUT_LOAD", table=t, src_addr=P_LUT + 512 * t))
    await tb.drain()
    sm = g.softmax()
    w0 = word(sm) & ~(1 << 6)                     # first = 0
    await expect_err(w0, 4)
    sm.f.update(causal=1, diag=-1)
    await expect_err(int.from_bytes(isa.encode(sm), "little"), 4)
    big = isa.C("LN", in16=1, rows=1, cols=D_MAX + 16, src_stride=60, dst_stride=30)
    await expect_err(word(big), 5)
    # a REQUANT record with ls != 0: detected while loading, the command still completes
    rq = g.requant()
    tb.write_mem(isa.PBUF, rq["param_addr"], rec_bytes([[1, 1 << 30, 3, 2]]))
    await expect_err(word(rq), 8, unchecked=True)
    tb.sync_from_rtl()
    # the first error is the one reported
    tb.add_raw(word(isa.C("VEC", vop=7, len=4, imm=40)))
    tb.add_raw(word(isa.C("GEMM", M=1, N=16, K=16)))
    await tb.drain()
    assert tb.err() == (1, 3)
    # a legal command after an error still executes
    for t in range(3):
        tb.add(isa.C("LUT_LOAD", table=t, src_addr=P_LUT + 512 * t))
    tb.add(g.vec())
    await tb.drain()
    tb.compare_all()


async def clear(tb, hard=False):
    """Pulse the engine clear (or rst_n) for one cycle; the bench keeps answering reads that
    are in flight (unless hard). Afterwards the RTL must be idle, write nothing, pulse no done;
    tables, slots and saturation counters start from zero."""
    d = tb.dut
    tb.checking = False
    tb.cmdq.clear()
    tb.holding = None
    tb.busy_clears = getattr(tb, "busy_clears", 0) + int(int(d.busy.value) == 1 and any(p.q for p in tb.ports))
    if hard:
        await tb.hard_reset()
    else:
        tb.clr_now = True
        await tb.cycle()
    tb.writes = []
    n0 = tb.dones
    for _ in range(60):
        await tb.cycle()
    assert tb.dones == n0, "done after clear"
    assert not tb.writes, "writes after clear"
    assert int(d.busy.value) == 0
    assert all(not p.q for p in tb.ports), "read responses outstanding after clear"
    tb.expect.clear()
    tb.checking = True
    tb.sync_from_rtl()
    tb.fs.sfu.clear()
    tb.sat = [0] * 5
    tb.sat_base = [0] * 5


async def FallingEdge_(d):
    from cocotb.triggers import FallingEdge
    await FallingEdge(d.clk)


@cocotb.test()
async def vec_seq_clear_mid_command(dut):
    """clr (soft reset) and rst_n at random points inside long commands, with reads in flight;
    then the engine runs a fresh batch correctly."""
    tb = TB(dut, seed=31, lat=8)
    await tb.start()
    rng = random.Random(32)
    g = Gen(tb, rng)
    g.setup()
    await tb.drain()
    for trial in range(9):
        long_cmds = [g.ln(), g.vec(), g.requant(), g.softmax(), g.gelu()]
        c = long_cmds[trial % len(long_cmds)]
        tb.checking = False
        tb.cmdq.append(word(c))
        tb.cmdq.append(word(g.vec()))
        wait = rng.randrange(5, 200)
        for _ in range(5000):           # a random point at which reads are in flight
            await tb.cycle()
            wait -= 1
            if wait <= 0 and int(dut.busy.value) == 1 and any(p.q for p in tb.ports):
                break
        await clear(tb, hard=(trial % 3 == 2))
        dut._log.info(f"trial {trial}: cleared during {c.op} at cycle {tb.cyc}")
        for t in range(3):
            tb.add(isa.C("LUT_LOAD", table=t, src_addr=P_LUT + 512 * t))
        cmds = [g.any() for _ in range(4)]
        for c2 in cmds:
            tb.add(c2)
        await tb.drain()
        tb.compare_all()
        assert tb.err() == (0, 0)
    assert tb.busy_clears >= 5, f"only {tb.busy_clears} clears hit a command with reads in flight"


@cocotb.test()
async def vec_seq_residency(dut):
    """Parameter slots: a repeated block is a hit (no PBUF reads), a snooped rewrite of it is a miss,
    eviction beyond the slot count reloads; cycle costs are logged for docs/vector-unit.md §9."""
    tb = TB(dut, seed=41, p_ar=1.0, p_w=1.0, p_cmd=1.0, lat=2)
    await tb.start()
    rng = random.Random(42)
    g = Gen(tb, rng)
    g.setup()
    await tb.drain()

    async def one(c):
        h0, m0, l0 = u(dut.st_hits), u(dut.st_misses), u(dut.st_lines)
        rp0 = tb.ports[2].reads
        c0 = tb.cyc
        tb.add(c)
        await tb.drain()
        tb.compare_all()
        cyc = tb.done_cycles[-1] - c0
        return u(dut.st_hits) - h0, u(dut.st_misses) - m0, u(dut.st_lines) - l0, tb.ports[2].reads - rp0, cyc

    addr, cols = g.lnb[0]
    ln = g.ln()
    ln.f.update(param_addr=addr, cols=cols, rows=1, src_stride=rup(2 * cols) // LINE + 1, in16=1,
                dst_stride=rup(cols) // LINE)
    first = await one(ln)
    again = await one(ln)
    assert again[:4] == (1, 0, 0, 0), again
    tb.write_mem(isa.PBUF, addr, ln_variant(rng, cols, 5))      # snooped rewrite
    third = await one(ln)
    assert third[:3] == (0, 1, ln_block_layout(cols)[2] // LINE), third
    dut._log.info(f"LN 1x{cols}: miss {first[4]} cycles, hit {again[4]} cycles, "
                  f"reload {third[2]} lines")
    # LUT_LOAD from the same address again: hit, 0 lines
    lut = await one(isa.C("LUT_LOAD", table=1, src_addr=P_LUT + 512))
    assert lut[:4] == (1, 0, 0, 0), lut
    lut2 = await one(isa.C("LUT_LOAD", table=1, src_addr=P_LUT2 + 512))
    assert lut2[:3] == (0, 1, 32), lut2
    dut._log.info(f"LUT_LOAD: miss {lut2[4]} cycles, hit {lut[4]} cycles")
    # GELU tables: 6 distinct blocks through NG = 4 slots
    hits = 0
    for rep in range(2):
        for i in range(N_GELU):
            c = g.gelu()
            c.f.update(lut_addr=P_GELU + 256 * i)
            hits += (await one(c))[0]
    assert hits == 0, "6 tables round-robin through 4 slots: every access misses"
    for rep in range(2):
        for i in range(3):
            c = g.gelu()
            c.f.update(lut_addr=P_GELU + 256 * i)
            r = await one(c)
            if rep == 1:
                assert r[0] == 1, r
    # softmax ctx records: 4 used repeatedly are resident
    for rep in range(3):
        for i in range(4):
            c = g.softmax()
            c.f.update(ctx_addr=P_CTX + 16 * i)
            r = await one(c)
    assert r[0] == 1
