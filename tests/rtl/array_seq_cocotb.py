"""cocotb tests for rtl/array_seq.sv (ARRAY engine: ISA GEMM / GEMM_GS -> array_ctrl tiles).

Commands are built with nanoxl.isa.encode; scratchpad images are numpy buffers; the reference
is nanoxl.funcsim (GEMM / GEMM_GS semantics, compiler operand layouts, the ACC buffer).

Env: SEQ_N (array size), SEQ_MDEPTH (array_ctrl accumulator rows = row chunk), SEQ_MODE:
  fast          random commands, reset / clr mid-command, illegal opcode, compiler subset
  compiler_all  every GEMM / GEMM_GS the compiler emits for tiny W8A8 + W4A8 (prefill + decode)
  kv260         N=32: random commands, every distinct compiler shape, cycles vs the cycle model

The scratchpad model serves four ports (A, W, ACC read, ACC write) with random ready and random
in-order response latency. A read accepted at a clock edge sees every write accepted at an earlier
edge; a read and a write of the same ACC line at the same edge is flagged (the engine must never
do that: its ordering guarantee would then depend on the memory).
"""
import json
import os
import random
from collections import deque
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly

from nanoxl import cyclemodel, isa
from nanoxl.funcsim import FuncSim, accesses, buf_sizes, rect_bytes
from nanoxl.hwconfig import KV260, KiB

NA = int(os.environ.get("SEQ_N", "8"))
MD = int(os.environ.get("SEQ_MDEPTH", "16"))
MODE = os.environ.get("SEQ_MODE", "fast")
APL = WPL = max(1, (NA + 15) // 16)
CPL = NA // 4
SMALL = KV260.with_(array_n=NA, abuf=64 * KiB, wbuf=64 * KiB, acc=64 * KiB, pbuf=4 * KiB,
                    kvbuf=64 * KiB)
ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "artifacts" / "models" / "tiny"

PORTS = (("a", APL), ("w", WPL), ("cr", CPL))
BUF_OK = {"a": {isa.ABUF}, "w": {isa.WBUF, isa.KVBUF}, "cr": {isa.ACC}}


def word(c):
    return int.from_bytes(isa.encode(c, aligned=True), "little")


class SeqTB:
    def __init__(self, dut, seed, p_cmd=1.0, p_ar=1.0, p_cw=1.0, lat=(1, 1), hw=SMALL):
        self.dut = dut
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        self.p_cmd, self.p_ar, self.p_cw, self.lat = p_cmd, p_ar, p_cw, lat
        self.hw = hw
        self.mem = {b: np.zeros(n, np.uint8) for b, n in buf_sizes(hw).items()}
        self.cmdq = deque()
        self.accepted, self.dones = [], []
        self.resp = {p: deque() for p, _ in PORTS}
        self.last_due = {p: 0 for p, _ in PORTS}
        self.cycle = 0
        self.in_reset = False
        self.sig = {}
        for name in ("cmd_valid", "cmd_ready", "cmd_word", "done", "err", "idle", "cw_valid",
                     "cw_ready", "cw_buf", "cw_addr", "cw_data", "cw_strb", "hazard_seen",
                     "overflow_seen", "rst_n", "clr"):
            self.sig[name] = getattr(dut, name)
        for p, _ in PORTS:
            for s in ("arvalid", "arready", "arbuf", "araddr", "rvalid", "rdata"):
                self.sig[f"{p}_{s}"] = getattr(dut, f"{p}_{s}")
        self.reads_per_port = {p: 0 for p, _ in PORTS}
        self.writes = 0
        self.task = None

    # ---------------------------------------------------------------- plumbing
    async def start(self, clock=True):
        if clock:
            cocotb.start_soon(Clock(self.dut.clk, 10, unit="ns").start())
        s = self.sig
        s["clr"].value = 0
        s["cmd_valid"].value = 0
        s["cmd_word"].value = 0
        s["cw_ready"].value = 0
        for p, _ in PORTS:
            s[f"{p}_arready"].value = 0
            s[f"{p}_rvalid"].value = 0
            s[f"{p}_rdata"].value = 0
        await self.hard_reset()
        self.task = cocotb.start_soon(self.loop())

    def stop(self):
        if self.task:
            self.task.cancel()
            self.task = None

    async def hard_reset(self, cycles=3):
        """rst_n low: engine and memory both reset (pending responses are lost)."""
        self.in_reset = True
        self.sig["rst_n"].value = 0
        for _ in range(cycles):
            await FallingEdge(self.dut.clk)
        for q in self.resp.values():
            q.clear()
        self.cmdq.clear()
        self.accepted, self.dones = [], []
        self.sig["rst_n"].value = 1
        self.in_reset = False

    async def soft_clear(self):
        """clr pulse (ctrl_top eng_clr): the memory keeps answering requests issued before."""
        await FallingEdge(self.dut.clk)
        self.in_reset = True
        self.cmdq.clear()
        self.sig["clr"].value = 1
        self.sig["cmd_valid"].value = 0
        await FallingEdge(self.dut.clk)
        self.sig["clr"].value = 0
        self.accepted, self.dones = [], []
        self.in_reset = False

    def _read(self, buf, line, pl):
        m = self.mem[buf]
        idx = np.arange(line * 16, line * 16 + 16 * pl)
        return int.from_bytes(np.take(m, idx, mode="wrap").tobytes(), "little")

    async def loop(self):
        s, rng = self.sig, self.rng
        clk = self.dut.clk
        while True:
            await FallingEdge(clk)
            self.cycle += 1
            cyc = self.cycle
            drive_cmd = False
            if not self.in_reset:
                if self.cmdq and rng.random() < self.p_cmd:
                    s["cmd_valid"].value = 1
                    s["cmd_word"].value = self.cmdq[0][0]
                    drive_cmd = True
                else:
                    s["cmd_valid"].value = 0
            else:
                s["cmd_valid"].value = 0
            ar_rdy = {}
            for p, _ in PORTS:
                r = rng.random() < self.p_ar
                ar_rdy[p] = r
                s[f"{p}_arready"].value = int(r)
                q = self.resp[p]
                if q and q[0][0] <= cyc:
                    s[f"{p}_rvalid"].value = 1
                    s[f"{p}_rdata"].value = q.popleft()[1]
                else:
                    s[f"{p}_rvalid"].value = 0
            cw_rdy = rng.random() < self.p_cw
            s["cw_ready"].value = int(cw_rdy)
            await ReadOnly()
            if not int(s["rst_n"].value):
                continue
            if drive_cmd and int(s["cmd_ready"].value):
                self.accepted.append(cyc)
                self.cmdq.popleft()
            if int(s["done"].value):
                self.dones.append(cyc)
            acc_lines = set()
            for p, pl in PORTS:
                if ar_rdy[p] and int(s[f"{p}_arvalid"].value):
                    buf = int(s[f"{p}_arbuf"].value)
                    line = int(s[f"{p}_araddr"].value)
                    assert buf in BUF_OK[p], (p, buf)
                    if p == "cr":
                        acc_lines.update(range(line, line + pl))
                    due = max(self.last_due[p] + 1, cyc + rng.randint(*self.lat))
                    self.last_due[p] = due
                    self.resp[p].append((due, self._read(buf, line, pl)))
                    self.reads_per_port[p] += 1
            if cw_rdy and int(s["cw_valid"].value):
                assert int(s["cw_buf"].value) == isa.ACC
                line = int(s["cw_addr"].value)
                strb = int(s["cw_strb"].value)
                data = int(s["cw_data"].value).to_bytes(4 * NA, "little")
                assert not acc_lines & set(range(line, line + CPL)), "same-edge ACC read/write"
                m = self.mem[isa.ACC]
                base = line * 16
                for j in range(4 * NA):
                    if strb >> j & 1:
                        assert base + j < len(m), "ACC write out of range"
                        m[base + j] = data[j]
                self.writes += 1

    # ---------------------------------------------------------------- commands
    def push(self, c):
        self.cmdq.append((word(c), c))

    async def wait_all(self, limit=2_000_000):
        s = self.sig
        start = self.cycle
        while True:
            await FallingEdge(self.dut.clk)
            if not self.cmdq and len(self.dones) == len(self.accepted) and int(s["idle"].value):
                break
            assert self.cycle - start < limit, "timeout"
        self.check_done_order()
        assert int(s["hazard_seen"].value) == 0, "array weight hazard"
        assert int(s["overflow_seen"].value) == 0, "array_ctrl FIFO overflow / missing scale row"

    def check_done_order(self):
        assert len(self.dones) == len(self.accepted), (len(self.dones), len(self.accepted))
        for i, (a, dn) in enumerate(zip(self.accepted, self.dones)):
            assert dn > a, f"done {i} at {dn} not after accept at {a}"


# -------------------------------------------------------------------- random commands
def rand_cmd(rng, hw, prev=None, nmax=None, kmax=None, mmax=None):
    """A random legal, aligned GEMM / GEMM_GS within the SMALL buffers. With ``prev`` (and some
    probability) it accumulates into prev's C tile (K-split panel chain, RAW through ACC)."""
    nmax = nmax or 3 * NA + 5
    kmax = kmax or 3 * NA + 5
    mmax = mmax or 2 * MD + 3
    kind = rng.choice(["i8", "i8", "i4", "trans", "kv", "gs", "gs"])
    chain = prev is not None and rng.random() < 0.35
    if chain:
        M, N = prev.f["M"], prev.f["N"]
        if kind in ("i4", "gs") and N % 2:
            kind = "i8"
    else:
        M = rng.randint(1, mmax) if rng.random() < 0.7 else rng.choice([1, 1, MD, MD + 1])
        N = rng.randint(1, nmax)
        if kind in ("i4", "gs"):
            N += N % 2
    glog2 = rng.choice([0, 0, 1, 3])
    G = isa.gs_group(glog2)
    if kind == "gs":
        K = rng.randint(1, max(kmax, 2 * G + 3))
    else:
        K = rng.randint(1, kmax)
    if K * N > 4000:
        K = max(1, 4000 // N)
    sa = -(-K // 16) + rng.randint(0, 2)
    if chain:
        c_addr, cs = prev.f["c_addr"], prev.f["c_stride"]
    else:
        cs = -(-4 * N // 16) + rng.randint(0, 2)
        c_addr = 16 * rng.randrange((hw.acc - M * cs * 16) // 16)
    a_addr = 16 * rng.randrange((hw.abuf - M * sa * 16) // 16)
    acc = int(chain or rng.random() < 0.3)
    if kind == "gs":
        wb = isa.gs_layout(K, N, glog2)[2]
    elif kind == "i4":
        wb = K * N // 2
    else:
        wb = K * N
    wcap = hw.kvbuf if kind == "kv" else hw.wbuf
    w_addr = 16 * rng.randrange((wcap - wb) // 16)
    common = dict(accumulate=acc, a_addr=a_addr, w_addr=w_addr, c_addr=c_addr, M=M, N=N, K=K,
                  a_stride=sa, c_stride=cs, gemv=int(M == 1))
    if kind == "gs":
        return isa.C("GEMM_GS", glog2=glog2, **common)
    return isa.C("GEMM", int4=int(kind == "i4"), w_trans=int(kind in ("trans", "kv") and rng.random() < 0.7),
                 w_buf=int(kind == "kv"), **common)


def fresh_images(tb, seed):
    """Random bytes in every buffer of the TB memory and of a FuncSim with the same contents."""
    nrng = np.random.default_rng(seed)
    fs = FuncSim(tb.hw)
    for b in tb.mem:
        tb.mem[b][:] = nrng.integers(0, 256, len(tb.mem[b]), dtype=np.uint8)
        fs.mem[b][:] = tb.mem[b]
    return fs


def check_images(tb, fs):
    for b in tb.mem:
        if not np.array_equal(tb.mem[b], fs.mem[b]):
            bad = np.nonzero(tb.mem[b] != fs.mem[b])[0]
            raise AssertionError(f"{isa.BUF_NAMES[b]} differs at {len(bad)} bytes, first {bad[:8]}")


async def run_random(dut, tb, n_cmds, seed, **kw):
    fs = fresh_images(tb, seed)
    prev = None
    cmds = []
    for _ in range(n_cmds):
        c = rand_cmd(tb.rng, tb.hw, prev, **kw)
        cmds.append(c)
        prev = c
    for c in cmds:
        fs.step(c)
        tb.push(c)
    await tb.wait_all()
    check_images(tb, fs)
    assert int(tb.sig["err"].value) == 0
    return cmds


# -------------------------------------------------------------------- compiler commands
class CaptureSim(FuncSim):
    """FuncSim that records every GEMM / GEMM_GS with the bytes it reads (and the C tile before)
    and the C tile after execution."""
    def step(self, c):
        if c.op not in ("GEMM", "GEMM_GS"):
            return super().step(c)
        regs = []
        for buf, _wr, addr, rb, rows, stride in accesses(c):
            lo, hi = rect_bytes(addr, rb, rows, stride)
            regs.append((buf, lo, self.mem[buf][lo:hi].copy()))
        super().step(c)
        f = c.f
        lo, hi = rect_bytes(f["c_addr"], 4 * f["N"], f["M"], 16 * f["c_stride"])
        self.cases.append((c, regs, lo, self.mem[isa.ACC][lo:hi].copy()))


def compiler_cases(name, hw, decode_steps=1):
    from nanoxl.export import load_bundle
    from nanoxl.funcsim import HostRunner
    ref = json.loads((TINY / "golden_ref.json").read_text())
    A, meta = load_bundle(TINY / f"{name}.npz")
    r = HostRunner(A, meta, hw, name=name)
    r.sim.__class__ = CaptureSim
    r.sim.cases = []
    logits = r.prefill(ref["prompt"])
    for _ in range(decode_steps):
        logits = r.step([[int(np.argmax(logits[-1]))]])[0]
    return r.sim.cases


def shape_key(c):
    f = c.f
    return (c.op, f["M"], f["N"], f["K"], f.get("w_trans", 0), f.get("int4", 0))


async def run_cases(tb, cases, label):
    """Each case on its own: load its operand bytes, run the command, compare the C tile bytes."""
    for i, (c, regs, lo, after) in enumerate(cases):
        for buf, rlo, data in regs:
            tb.mem[buf][rlo:rlo + len(data)] = data
        tb.push(c)
        await tb.wait_all()
        got = tb.mem[isa.ACC][lo:lo + len(after)]
        if not np.array_equal(got, after):
            bad = np.nonzero(got != after)[0]
            raise AssertionError(f"{label} case {i} {c}: ACC differs at {len(bad)} bytes, first {bad[:8]}")
    assert int(tb.sig["err"].value) == 0


def subset(cases, keys=None):
    seen, out = set(), []
    for case in cases:
        k = shape_key(case[0])
        if k in seen or (keys is not None and not keys(case[0])):
            continue
        seen.add(k)
        out.append(case)
    return out


async def measure(tb, c, regs):
    """(latency accept->done, steady occupancy) of one command, run twice back to back."""
    for buf, rlo, data in regs:
        tb.mem[buf][rlo:rlo + len(data)] = data
    tb.accepted, tb.dones = [], []
    tb.push(c)
    tb.push(c)
    await tb.wait_all()
    return tb.dones[0] - tb.accepted[0], tb.dones[1] - tb.dones[0]


# -------------------------------------------------------------------- tests
FAST = MODE == "fast"


@cocotb.test(skip=MODE not in ("fast", "kv260"))
async def seq_random(dut):
    """Random GEMM (INT8, INT4, w_trans, KVBUF) and GEMM_GS (group 8..64) commands back to back:
    ragged M (beyond the row chunk), N and K, K accumulation chains into the previous command's
    C tile (RAW through ACC), random strides and aligned bases, garbage in every unused byte.
    Random ready on every port, random response latency 1..6, random cmd_valid bubbles."""
    tb = SeqTB(dut, 1000 + NA, p_cmd=0.6, p_ar=0.7, p_cw=0.6, lat=(1, 6))
    await tb.start()
    n = int(os.environ.get("SEQ_NCMDS", "30"))
    cmds = await run_random(dut, tb, n, 77 + NA)
    dut._log.info("random: %d commands, %d ACC writes, reads %s", len(cmds), tb.writes, tb.reads_per_port)
    tb.stop()


@cocotb.test(skip=not FAST)
async def seq_random_nobp(dut):
    """Same generator, no backpressure and latency 1 (the streaming corner: one segment per cycle)."""
    tb = SeqTB(dut, 2000 + NA)
    await tb.start()
    await run_random(dut, tb, 20, 99 + NA)
    tb.stop()


@cocotb.test(skip=not FAST)
async def seq_gs_groups(dut):
    """GEMM_GS directed: every group size 8..128 with K spanning several groups and a ragged last
    group, extreme operands (a=-128, w=-8, gm=255), accumulate=1 on top of the previous result."""
    tb = SeqTB(dut, 3000 + NA, p_ar=0.8, p_cw=0.8, lat=(1, 3))
    await tb.start()
    fs = fresh_images(tb, 5)
    cmds = []
    c_addr = 0
    for glog2 in range(5):
        G = isa.gs_group(glog2)
        K = 2 * G + 5 if G <= 32 else G + 9
        N = 2 * NA + 2
        M = 3
        wb = isa.gs_layout(K, N, glog2)
        w_addr = 4096
        # extreme operands in the first rows / columns
        fs.mem[isa.ABUF][0:K] = 0x80
        fs.mem[isa.WBUF][w_addr:w_addr + N // 2 * 4] = 0x88
        fs.mem[isa.WBUF][w_addr + wb[1]:w_addr + wb[1] + N] = 255
        for b in (isa.ABUF, isa.WBUF):
            tb.mem[b][:] = fs.mem[b]
        for acc in (0, 1):
            c = isa.C("GEMM_GS", gemv=0, accumulate=acc, a_addr=0, w_addr=w_addr, c_addr=c_addr, M=M, N=N,
                      K=K, a_stride=-(-K // 16), c_stride=-(-4 * N // 16), glog2=glog2)
            fs.step(c)
            tb.push(c)
        await tb.wait_all()
        check_images(tb, fs)
        cmds.append(c)
        c_addr += 4096
    tb.stop()


@cocotb.test(skip=not FAST)
async def seq_illegal_opcode(dut):
    """ATTN_TILE (reserved) between two GEMMs: err rises, every command still gets its done in
    order, both GEMMs are exact."""
    tb = SeqTB(dut, 4000 + NA, lat=(1, 3))
    await tb.start()
    fs = fresh_images(tb, 6)
    g1 = rand_cmd(tb.rng, tb.hw)
    g2 = rand_cmd(tb.rng, tb.hw)
    bad = isa.C("ATTN_TILE", first=1, last=1, q_addr=0, k_addr=0, v_addr=0, o_addr=0, ctx_addr=0,
                rows=1, len=1)
    for c in (g1, bad, g2):
        if c.op != "ATTN_TILE":
            fs.step(c)
        tb.push(c)
    await tb.wait_all()
    check_images(tb, fs)
    assert int(tb.sig["err"].value) == 1
    tb.stop()


@cocotb.test(skip=not FAST)
async def seq_reset_mid_command(dut):
    """rst_n and clr in the middle of commands (rows in the array, reads in flight, writes
    pending). After rst_n the memory is reset too; after clr it keeps returning the responses of
    requests issued before the clear, which the engine must drop. Fresh commands afterwards are
    exact, and no done appears for a command accepted before the reset."""
    tb = SeqTB(dut, 5000 + NA, p_ar=0.9, p_cw=0.5, lat=(2, 8))
    await tb.start()
    for how in ("rst", "clr", "clr"):
        fresh_images(tb, 7)
        tb.accepted, tb.dones = [], []
        for _ in range(3):
            tb.push(rand_cmd(tb.rng, tb.hw, mmax=MD, nmax=2 * NA, kmax=2 * NA))
        for _ in range(tb.rng.randint(4 * NA, 10 * NA)):
            await FallingEdge(dut.clk)
        assert tb.accepted and len(tb.dones) < 3, "reset should hit mid-stream"
        if how == "rst":
            await tb.hard_reset()
        else:
            for _ in range(500):   # clear while responses are in flight
                if any(tb.resp[p] for p, _ in PORTS):
                    break
                await FallingEdge(dut.clk)
            assert any(tb.resp[p] for p, _ in PORTS), "want responses in flight at clr"
            await tb.soft_clear()
        for _ in range(12 * NA):
            await FallingEdge(dut.clk)
        assert not tb.dones, "done after reset for a command accepted before it"
        assert int(tb.sig["idle"].value) == 1 and int(tb.sig["err"].value) == 0
        await run_random(dut, tb, 4, 8 + len(how))
    tb.stop()


@cocotb.test(skip=MODE == "kv260" or (FAST and NA != 8))
async def seq_compiler(dut):
    """GEMM / GEMM_GS commands exactly as the compiler emits them for the tiny W8A8 and W4A8
    bundles (prefill of the 16-token reference prompt + one decode step; KV260 buffers, array N),
    operands from funcsim's scratchpad state right before each command, result compared with
    funcsim's ACC tile. fast: one command per distinct shape among the cheap ones (attention
    scores w_trans/KVBUF, PV, ragged LM head N=65, the smallest linears); compiler_all: every one."""
    hw = KV260.with_(array_n=NA)
    tb = SeqTB(dut, 6000 + NA, p_ar=0.85, p_cw=0.85, lat=(1, 4), hw=hw)
    await tb.start()
    total = 0
    for name in ("w8a8", "w4a8"):
        cases = compiler_cases(name, hw)
        if FAST:
            cases = subset(cases, lambda c: c.f["N"] * c.f["K"] * max(c.f["M"], NA) <= 16 * 128 * 16
                           or c.f["N"] == 65 and c.f["M"] == 1)
        dut._log.info("%s: %d GEMM commands: %s", name, len(cases),
                      sorted({shape_key(c[0]) for c in cases}))
        await run_cases(tb, cases, name)
        total += len(cases)
    dut._log.info("compiler commands checked: %d", total)
    tb.stop()


@cocotb.test(skip=MODE != "kv260")
async def seq_compiler_perf(dut):
    """KV260 size: every distinct compiler GEMM shape of tiny W8A8/W4A8 is checked exactly, then run
    twice back to back without backpressure (latency 1) to measure accept->done latency and the
    steady-state occupancy (done to done), reported next to nanoxl.cyclemodel's estimate."""
    hw = KV260.with_(array_n=NA)
    tb = SeqTB(dut, 7000 + NA, hw=hw)
    await tb.start()
    rows = []
    for name in ("w8a8", "w4a8"):
        cases = subset(compiler_cases(name, hw))
        await run_cases(tb, cases, name)
        for c, regs, _lo, _after in cases:
            lat, occ = await measure(tb, c, regs)
            model = cyclemodel.cost(c, hw)
            f = c.f
            tiles = -(-f["N"] // NA) * -(-f["K"] // (min(NA, isa.gs_group(f["glog2"])) if c.op == "GEMM_GS" else NA))
            rows.append((name, shape_key(c), tiles, occ, lat, model[0], model[1]))
    dut._log.info("PERF bundle | op M N K trans int4 | tiles | measured occ | accept->done | model occ | model lat")
    for r in rows:
        dut._log.info("PERF %s | %s | %d | %d | %d | %d | %d", *r)
    tb.stop()


@cocotb.test(skip=not FAST)
async def seq_perf_streaming(dut):
    """Steady-state cycles per command (done to done, the same command twice back to back, no
    backpressure, latency 1) equal sum over tiles of max(m, k) (k+1 for GEMM_GS tiles) for aligned
    operands: the engine adds no bubble between tiles, k-chains, n-tiles, row chunks or commands."""
    tb = SeqTB(dut, 8000 + NA)
    await tb.start()
    fresh_images(tb, 9)
    D = 2 * NA

    def g(op="GEMM", M=1, N=D, K=D, **kw):
        sa, cs = -(-K // 16), -(-4 * N // 16)
        return isa.C(op, gemv=int(M == 1), accumulate=0, a_addr=0, w_addr=0, c_addr=0, M=M, N=N, K=K,
                     a_stride=sa, c_stride=cs, **kw)

    def ideal(c):
        f = c.f
        gs = c.op == "GEMM_GS"
        kt = min(NA, isa.gs_group(f["glog2"])) if gs else NA
        tot = 0
        for m0 in range(0, f["M"], MD):
            mc = min(MD, f["M"] - m0)
            for _n0 in range(0, f["N"], NA):
                for k0 in range(0, f["K"], kt):
                    tot += max(mc, min(kt, f["K"] - k0) + gs)
        return tot

    cases = [("gemv int8", g(N=4 * NA, K=4 * NA, int4=0)),
             ("gemv int4", g(int4=1)),
             ("gemv gs G=64", g("GEMM_GS", N=D, K=64, glog2=3)),
             ("gemv gs G=8", g("GEMM_GS", N=D, K=32, glog2=0)),
             ("prefill int8", g(M=D, int4=0)),
             ("prefill gs", g("GEMM_GS", M=D, K=64, glog2=3)),
             ("rows > chunk", g(M=MD + 3, N=NA, K=NA, int4=0)),
             ("w_trans", g(N=D, K=D, int4=0, w_trans=1)),
             ("w_trans KVBUF M=2N", g(M=D, N=D, K=D, int4=0, w_trans=1, w_buf=1))]
    for label, c in cases:
        lat, occ = await measure(tb, c, [])
        dut._log.info("PERF N=%d %-20s M=%-3d N=%-3d K=%-3d occ %4d (ideal %4d)  accept->done %d",
                      NA, label, c.f["M"], c.f["N"], c.f["K"], occ, ideal(c), lat)
        assert occ == ideal(c), (label, occ, ideal(c))
    # unaligned weight rows (ragged N = 17, like the attention PV GEMM): straddling segments cost a beat
    c = g(M=1, N=17, K=D, int4=0)
    lat, occ = await measure(tb, c, [])
    dut._log.info("PERF N=%d %-20s M=1 N=17 K=%d occ %d (aligned ideal %d)", NA, "ragged rows", D, occ, ideal(c))
    tb.stop()
