"""Host driver (nanoxl.driver) through the SimBackend: end-to-end generate() equals the committed tiny
reference token sequences (W8A8, W4A8), plus the CSR semantics of docs/control.md emulated by the
SimBackend (ring wrap, END not blocking a moved tail, illegal words, W1C, soft reset, single step,
semaphore limit, perf / saturation counters) and the CLI."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nanoxl import isa
from nanoxl.cyclemodel import simulate
from nanoxl.driver import (SHAKESPEARE_CHARS, CharCodec, DriverError, DriverTimeout, NanoXL, SimBackend,
                           csr as R, open_bundle, sample)
from nanoxl.export import load_bundle
from nanoxl.hwconfig import KV260

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "artifacts" / "models" / "tiny"
REF = json.loads((TINY / "golden_ref.json").read_text())
C = isa.C


@pytest.fixture(scope="module", params=["w8a8", "w4a8"])
def bundle(request):
    A, meta = load_bundle(TINY / f"{request.param}.npz")
    return request.param, A, meta


# ---------------------------------------------------------------- end to end

def test_driver_generate_matches_reference(bundle):
    """100 greedy tokens through the ring, CSRs and DDR buffer (relocated to a different physical base)
    equal golden_ref.json; the SAT counters + host embedding count add up to the golden counts."""
    name, A, meta = bundle
    be = SimBackend(KV260, phys_base=0x7123_4000)
    dev = NanoXL(be, A, meta, KV260, name=name)
    assert dev.delta != 0
    toks = dev.generate(REF["prompt"], REF["n_new"])
    ref = REF["bundles"][name]
    assert toks == ref["tokens"]
    assert CharCodec().decode(toks) == ref["text"]
    assert dev.programs == REF["n_new"]
    total = sum(dev.saturation().values()) + sum(dev.host_sat.values())
    assert total == sum(ref["sat_count"].values())
    p = dev.perf()
    assert p["cycles"] > 0 and p["bytes"] > 0 and p["retired"] > 0
    assert be.read32(R.ERR_CODE) == 0 and be.read32(R.STATUS) & R.ST_QUEUES_IDLE


def test_driver_codec():
    c = CharCodec()
    assert len(SHAKESPEARE_CHARS) == 65
    assert c.encode(REF["prompt_text"]) == REF["prompt"]
    assert c.decode(REF["prompt"]) == REF["prompt_text"]
    raw = ROOT / "data" / "raw" / "input.txt"
    if raw.exists():
        assert "".join(sorted(set(raw.read_text()))) == SHAKESPEARE_CHARS
    with pytest.raises(ValueError):
        c.encode("é")


def test_driver_sampling():
    l = np.array([3, 9, 9, 1, -5], np.int16)
    assert sample(l) == 1                                    # ties -> lowest index
    assert sample(l, top_k=1, temperature=0.7, rng=np.random.default_rng(0)) == 1
    picks = {sample(l, top_k=2, temperature=1.0, rng=np.random.default_rng(s)) for s in range(50)}
    assert picks == {1, 2}
    a = [sample(l, top_k=3, temperature=5, rng=np.random.default_rng(7), scale=1.0) for _ in range(3)]
    b = [sample(l, top_k=3, temperature=5, rng=np.random.default_rng(7), scale=1.0) for _ in range(3)]
    assert a == b and set(a) <= {0, 1, 2}


def test_driver_topk_generate_runs(bundle):
    name, A, meta = bundle
    dev = NanoXL(SimBackend(KV260), A, meta, KV260, name=name)
    t1 = dev.generate(REF["prompt"], 6, top_k=5, temperature=0.8, seed=3)
    t2 = dev.generate(REF["prompt"], 6, top_k=5, temperature=0.8, seed=3)   # resets KV, same seed
    assert t1 == t2 and len(t1) == 6 and all(0 <= t < 65 for t in t1)


def test_driver_ring_wrap(bundle):
    """A ring a little larger than the prefill program: decode programs wrap around the ring end
    (written in two pieces) and tokens still match the reference."""
    name, A, meta = bundle
    probe = NanoXL(SimBackend(KV260), A, meta, KV260, name=name)
    pre = len(probe.comp.compile_step([(0, len(REF["prompt"]))]).cmds)
    ring_bytes = 16 * (pre + 37)
    be = SimBackend(KV260)
    dev = NanoXL(be, A, meta, KV260, name=name, cmd_ring_bytes=ring_bytes)
    assert dev.ring_words == pre + 37
    tails, wraps = [], 0
    orig = dev._write_ring

    def spy(words):
        nonlocal wraps
        wraps += dev.tail + len(words) // 16 > dev.ring_words
        orig(words)
        tails.append(dev.tail)
    dev._write_ring = spy
    toks = dev.generate(REF["prompt"], 12)
    assert toks == REF["bundles"][name]["tokens"][:12]
    assert wraps >= 2
    assert be.read32(R.RING_HEAD) == dev.tail


# ---------------------------------------------------------------- CSR emulation (raw streams)

class Raw:
    """SimBackend with a small ring and a scratch DDR area, driven word by word."""

    def __init__(self, ring_words=64, nbytes=1 << 16):
        self.be = SimBackend(KV260, phys_base=0x6000_0000, timing=True)
        self.mem, self.phys = self.be.alloc(nbytes)
        self.rw = ring_words
        self.be.write32(R.RING_BASE, self.phys)
        self.be.write32(R.RING_SIZE, ring_words)
        self.be.write32(R.IRQ_EN, 7)
        self.tail = 0
        self.data = self.phys + 16 * ring_words      # DDR scratch after the ring

    def put(self, words, ring=True):
        for w in words:
            if isinstance(w, isa.Cmd):
                w = isa.encode(w)
            o = 16 * self.tail
            self.mem[o:o + 16] = np.frombuffer(w, np.uint8)
            self.tail = (self.tail + 1) % self.rw
        if ring:
            self.be.write32(R.DOORBELL, self.tail)

    def r(self, off):
        return self.be.read32(off)


def load(raw, sp_addr, k):
    """DMA_LOAD of 16 bytes from DDR scratch line k to ABUF sp_addr."""
    return C("DMA_LOAD", sp_buf=isa.ABUF, sp_addr=sp_addr, ddr_addr=raw.data + 16 * k, rows=1, row_bytes=16)


def abuf(raw, a):
    return bytes(raw.be.fs.mem[isa.ABUF][a:a + 16])


def test_driver_csr_id_and_unmapped():
    raw = Raw()
    assert raw.r(R.ID) == 0x4E584C43 and raw.r(R.VERSION) == 0x00010000
    assert raw.r(0x0F0) == 0
    raw.be.write32(R.ID, 0)          # RO
    assert raw.r(R.ID) == 0x4E584C43
    raw.be.write32(R.RING_BASE, 0x1234_5678)
    assert raw.r(R.RING_BASE) == 0x1234_5670
    assert raw.r(R.STATUS) & R.ST_RING_CFG_OK


def test_driver_csr_end_continues_past_moved_tail_and_w1c():
    raw = Raw()
    raw.mem[16 * raw.rw:16 * raw.rw + 32] = np.arange(32, dtype=np.uint8)
    # two programs queued before one doorbell: dispatch carries on past the first END
    raw.put([load(raw, 0, 0), C("END"), load(raw, 16, 1), C("END")])
    assert raw.r(R.END_COUNT) == 2 and raw.r(R.RING_HEAD) == 4
    assert abuf(raw, 16) == bytes(range(16, 32))
    st = raw.r(R.STATUS)
    assert st & R.ST_RING_EMPTY and st & R.ST_QUEUES_IDLE and not st & R.ST_BUSY
    assert raw.r(R.IRQ_STATUS) == R.IRQ_END and raw.be.irq
    # an IRQ command: bit 1 and IRQ_CODE
    raw.put([C("IRQ", code=0xBEEF)])
    assert raw.r(R.IRQ_CODE) == 0xBEEF and raw.r(R.IRQ_STATUS) == R.IRQ_END | R.IRQ_CMD
    # W1C: writing bit 0 clears only bit 0
    raw.be.write32(R.IRQ_STATUS, R.IRQ_END)
    assert raw.r(R.IRQ_STATUS) == R.IRQ_CMD
    raw.be.write32(R.IRQ_STATUS, 0)
    assert raw.r(R.IRQ_STATUS) == R.IRQ_CMD
    # irq line = |(IRQ_STATUS & IRQ_EN)
    raw.be.write32(R.IRQ_EN, R.IRQ_END)
    assert not raw.be.irq
    raw.be.write32(R.IRQ_EN, R.IRQ_CMD)
    assert raw.be.irq
    raw.be.write32(R.IRQ_STATUS, 0xFF)
    assert raw.r(R.IRQ_STATUS) == 0 and not raw.be.irq


def test_driver_csr_ring_wrap_raw():
    raw = Raw(ring_words=8)
    raw.mem[16 * 8:16 * 8 + 16 * 20] = (np.arange(16 * 20) % 251).astype(np.uint8)
    n = 0
    for k in range(20):  # 20 one-load programs of 2 words through an 8-word ring
        raw.put([load(raw, 16 * k, k), C("END")])
        n += 1
        assert raw.r(R.END_COUNT) == n and raw.r(R.RING_HEAD) == raw.tail
    for k in range(20):
        assert abuf(raw, 16 * k) == bytes((np.arange(16 * k, 16 * k + 16) % 251).astype(np.uint8))


@pytest.mark.parametrize("bad,code", [
    (b"\x00" * 16, R.ERR_OPCODE),                                            # zeroed memory
    ((0x3F).to_bytes(16, "little"), R.ERR_OPCODE),                           # undefined opcode
    ((isa.OPCODES["END"] | (1 << 100)).to_bytes(16, "little"), R.ERR_RESV),  # reserved bit
    (int.from_bytes(isa.encode(C("LUT_LOAD", table=0, src_addr=0)), "little")
     .__or__(3 << 6).to_bytes(16, "little"), R.ERR_FIELD),                   # LUT_LOAD table 3
], ids=["zero", "opcode3f", "resv", "field"])
def test_driver_csr_illegal_word_halts(bad, code):
    raw = Raw()
    raw.mem[16 * raw.rw:16 * raw.rw + 32] = 0xAB
    raw.put([load(raw, 0, 0), bad, load(raw, 16, 1), C("END")])
    assert raw.r(R.ERR_CODE) == code and raw.r(R.ERR_INDEX) == 1
    w = b"".join(raw.r(R.ERR_W0 + 4 * i).to_bytes(4, "little") for i in range(4))
    assert w == bad
    st = raw.r(R.STATUS)
    assert st & R.ST_HALTED
    assert raw.r(R.IRQ_STATUS) & R.IRQ_ERR and raw.r(R.END_COUNT) == 0
    assert abuf(raw, 0) == b"\xab" * 16 and abuf(raw, 16) == b"\x00" * 16   # before ran, after did not
    assert raw.r(R.RING_HEAD) == 1
    # further doorbells do nothing while halted
    raw.put([C("END")])
    assert raw.r(R.END_COUNT) == 0
    # soft reset recovers: ERR cleared, head = tail = 0, config and IRQ_EN kept
    retired = raw.r(R.PERF_RETIRED)
    raw.be.write32(R.CTRL, R.CTRL_SOFT_RESET)
    assert raw.r(R.CTRL) & R.CTRL_SOFT_RESET == 0
    assert raw.r(R.ERR_CODE) == 0 and not raw.r(R.STATUS) & R.ST_HALTED
    assert raw.r(R.RING_HEAD) == 0 and raw.r(R.DOORBELL) == 0
    assert raw.r(R.RING_SIZE) == raw.rw and raw.r(R.IRQ_EN) == 7 and raw.r(R.PERF_RETIRED) == retired
    raw.tail = 0
    raw.put([load(raw, 16, 1), C("END")])
    assert raw.r(R.END_COUNT) == 1 and abuf(raw, 16) == b"\xab" * 16


def test_driver_csr_ring_config_error():
    raw = Raw()
    raw.be.write32(R.DOORBELL, raw.rw)          # tail >= RING_SIZE
    assert raw.r(R.ERR_CODE) == R.ERR_RING and raw.r(R.STATUS) & R.ST_HALTED
    raw.be.write32(R.CTRL, R.CTRL_SOFT_RESET)
    raw.be.write32(R.RING_SIZE, 0)
    raw.be.write32(R.DOORBELL, 1)
    assert raw.r(R.ERR_CODE) == R.ERR_RING


def test_driver_csr_engine_error():
    """A command the engines cannot execute (out-of-range DDR read) halts with ERR 7."""
    raw = Raw()
    raw.put([C("DMA_LOAD", sp_buf=isa.ABUF, sp_addr=0, ddr_addr=0x1000_0000, rows=1, row_bytes=16), C("END")])
    assert raw.r(R.ERR_CODE) == R.ERR_ENGINE and raw.r(R.ERR_INDEX) == 0 and raw.r(R.END_COUNT) == 0


def test_driver_csr_deadlock_and_soft_reset():
    """A WAIT whose SIGNAL never comes leaves the machine busy (no error); the driver times out and a
    soft reset clears queues and semaphores."""
    raw = Raw()
    raw.put([C("WAIT", queue=isa.Q_ARR, sem=0, count=1), C("GEMM", M=1, N=32, K=32), C("END")])
    st = raw.r(R.STATUS)
    assert st & R.ST_BUSY and not st & R.ST_QUEUES_IDLE and not st & R.ST_HALTED
    assert not (st >> (R.ST_QIDLE_SHIFT + 1)) & 1          # ARRAY queue not idle
    assert raw.r(R.END_COUNT) == 0 and raw.r(R.ERR_CODE) == 0
    raw.be.write32(R.CTRL, R.CTRL_SOFT_RESET)
    st = raw.r(R.STATUS)
    assert st & R.ST_QUEUES_IDLE and not st & R.ST_BUSY


def test_driver_timeout_and_recovery(bundle):
    name, A, meta = bundle
    dev = NanoXL(SimBackend(KV260), A, meta, KV260, name=name, timeout=0.05)
    with pytest.raises(DriverTimeout):
        dev.wait(dev.submit([C("WAIT", queue=isa.Q_VEC, sem=3, count=1), C("END")]))
    dev.reset()
    assert dev.generate(REF["prompt"], 5) == REF["bundles"][name]["tokens"][:5]


def test_driver_error_raises_and_recovers(bundle):
    name, A, meta = bundle
    dev = NanoXL(SimBackend(KV260), A, meta, KV260, name=name)
    dev.submit_raw(b"\x00" * 16)
    with pytest.raises(DriverError) as e:
        dev.wait(dev.ends + 1)
    assert e.value.code == R.ERR_OPCODE and e.value.words == [0, 0, 0, 0]
    dev.reset()
    assert dev.generate(REF["prompt"], 5) == REF["bundles"][name]["tokens"][:5]


def sig_stream(n):
    return ([C("SIGNAL", queue=isa.Q_DMA, sem=7)] * n + [C("WAIT", queue=isa.Q_ARR, sem=7, count=1)] * n
            + [C("END")])


def test_driver_csr_semaphore_limit():
    """255 in the counter + 8 queued SIGNALs dispatch; the 264th blocks the in-order dispatcher
    before the WAITs (deadlock), matching funcsim.check_hazards' stream limit."""
    from nanoxl.funcsim import RaceError, check_hazards
    raw = Raw(ring_words=1024)
    raw.put(sig_stream(263))
    assert raw.r(R.END_COUNT) == 1
    raw.be.write32(R.SEM_SEL, 7)
    assert raw.r(R.SEM_VAL) == 0
    raw2 = Raw(ring_words=1024)
    raw2.put(sig_stream(264))
    assert raw2.r(R.END_COUNT) == 0 and raw2.r(R.STATUS) & R.ST_BUSY
    raw2.be.write32(R.SEM_SEL, 7)
    assert raw2.r(R.SEM_VAL) == 255 and raw2.r(R.RING_HEAD) == 263
    check_hazards(sig_stream(263))
    with pytest.raises(RaceError):
        check_hazards(sig_stream(264))


def test_driver_csr_single_step():
    raw = Raw()
    raw.mem[16 * raw.rw:16 * raw.rw + 32] = 0x5A
    raw.be.write32(R.CTRL, R.CTRL_STEP)
    raw.put([load(raw, 0, 0), C("SIGNAL", queue=0, sem=1), C("WAIT", queue=2, sem=1, count=1), C("END")])
    assert raw.r(R.RING_HEAD) == 0 and abuf(raw, 0) == b"\x00" * 16
    for k in range(3):
        raw.be.write32(R.DBG_STEP, 1)
        assert raw.r(R.RING_HEAD) == k + 1 and raw.r(R.DBG_LAST) == k
    assert abuf(raw, 0) == b"\x5a" * 16 and raw.r(R.END_COUNT) == 0
    raw.be.write32(R.DBG_STEP, 1)
    assert raw.r(R.END_COUNT) == 1
    raw.be.write32(R.CTRL, 0)


def test_driver_perf_and_saturation_counters(bundle):
    name, A, meta = bundle
    be = SimBackend(KV260)
    dev = NanoXL(be, A, meta, KV260, name=name)
    dev.clear_counters()
    seqs = [(0, len(REF["prompt"]))]
    prog = dev.comp.compile_step(seqs)
    dev.step([REF["prompt"]])
    p = dev.perf()
    assert p["retired"] == len(prog.cmds)
    assert p["bytes"] == sum(c["rows"] * c["row_bytes"] for c in prog.cmds if c.op.startswith("DMA"))
    t = simulate(prog.cmds, KV260)
    assert p["cycles"] == t.cycles and p["busy_arr"] == t.busy["array"] and p["busy_dma"] == t.busy["dma"]
    assert dev.last_perf == p
    sat = dev.saturation()
    assert set(sat) == {"requant", "alu", "ln_z", "ln_out"}
    dev.clear_counters()
    assert all(v == 0 for v in dev.perf().values()) and all(v == 0 for v in dev.saturation().values())


def test_driver_pynq_backend_is_lazy():
    import nanoxl.driver.backends as b
    try:
        import pynq  # noqa: F401
        pytest.skip("pynq installed: board backend not exercised here")
    except ImportError:
        with pytest.raises(ImportError):
            b.PynqBackend("missing.bit")


def test_driver_open_bundle_and_cli(tmp_path):
    dev = open_bundle(TINY / "w8a8.npz")
    assert isinstance(dev.be, SimBackend)
    toks = ",".join(map(str, REF["prompt"]))
    out = subprocess.run([sys.executable, "-m", "nanoxl.driver", "--bundle", str(TINY / "w8a8.npz"),
                          "--tokens", toks, "--n", "50", "--json"], capture_output=True, text=True,
                         cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    rec = json.loads(out.stdout)
    assert rec["tokens"] == REF["bundles"]["w8a8"]["tokens"][:50] and rec["programs"] == 50
    out = subprocess.run([sys.executable, "-m", "nanoxl.driver", "--bundle", str(TINY / "w4a8.npz"),
                          "--prompt", REF["prompt_text"], "--n", "50"], capture_output=True, text=True,
                         cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout == REF["prompt_text"] + REF["bundles"]["w4a8"]["text"][:50] + "\n"
    assert "[sim] 50 tokens" in out.stderr
