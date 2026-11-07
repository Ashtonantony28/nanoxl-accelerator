"""First-light checklist for a NanoXL overlay (SG9a, docs/fpga.md §6). Runs on the board with
`--backend pynq`, and here against the CSR emulator (`--backend sim`, what tests/fpga does).

    python3 fpga/first_light.py --board kv260 --backend pynq --bitstream nanoxl.bit [--n 32]
    python3 fpga/first_light.py --board kv260 --backend sim

Steps (each prints PASS/FAIL and stops at the first failure):
  1 csr        ID / VERSION, RING_BASE / RING_SIZE / IRQ_EN write-read-back, STATUS sane
  2 end        a program of one END: END_COUNT advances, IRQ_STATUS bit 0 set, W1C clears it
  3 dma        loopback through every scratchpad (ABUF, WBUF, ACC, PBUF, KVBUF) on both DMA channels: random
               bytes DMA_LOAD'ed from DDR and DMA_STORE'd back to another DDR area are unchanged
  4 bandwidth  1 MiB DMA_LOADs on channel 0, channel 1 and both at once, a 1 MiB DMA_STORE; PL DDR
               bandwidth from PERF_CYCLES at the board clock
  5 layer      one step of a 1-layer cut of tiny W8A8 (GEMM, REQUANT, LN, softmax, GELU): logits == nanoxl.golden
  6 generate   the full tiny W8A8 model: reference prompt + `--n` greedy tokens == golden_ref.json
The scratch DDR of steps 3-4 is the top 2 MiB of the image's 4 MiB command ring (the steps' own programs
use its first words only). Nothing here writes outside the driver's CMA buffer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpga.boards import CLOCK_MHZ, hwconfig  # noqa: E402
from nanoxl import isa  # noqa: E402
from nanoxl.driver import NanoXL  # noqa: E402
from nanoxl.driver import csr as R  # noqa: E402
from nanoxl.export import load_bundle  # noqa: E402
from nanoxl.golden import Golden  # noqa: E402

TINY = ROOT / "artifacts" / "models" / "tiny"
MiB = 1 << 20


class Fail(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def make_backend(kind, board, bitstream=None, ip="nanoxl_0"):
    if kind == "pynq":
        from nanoxl.driver import PynqBackend
        return PynqBackend(bitstream, ip)
    from nanoxl.driver import SimBackend
    return SimBackend(hwconfig(board))


def run_prog(dev, cmds):
    dev.wait(dev.submit(list(cmds) + [isa.C("END")]))


def step_csr(be, dev):
    check(be.read32(R.ID) == R.ID_VALUE, f"ID 0x{be.read32(R.ID):08x} != 0x{R.ID_VALUE:08x}")
    check(be.read32(R.VERSION) == R.VERSION_VALUE, f"VERSION 0x{be.read32(R.VERSION):08x}")
    base, size = be.read32(R.RING_BASE), be.read32(R.RING_SIZE)
    check(base == (dev.phys + dev.ring_off) & 0xFFFFFFF0, f"RING_BASE 0x{base:08x} (driver wrote it)")
    check(size == dev.ring_words, f"RING_SIZE {size} != {dev.ring_words}")
    be.write32(R.IRQ_EN, 0x5)
    check(be.read32(R.IRQ_EN) == 0x5, "IRQ_EN read-back")
    be.write32(R.IRQ_EN, R.IRQ_END | R.IRQ_ERR)
    st = be.read32(R.STATUS)
    check(st & R.ST_RING_CFG_OK and not st & R.ST_HALTED, f"STATUS 0x{st:04x}")
    check(be.read32(R.ERR_CODE) == 0, f"ERR_CODE {be.read32(R.ERR_CODE)}")
    return f"ID NXLC, VERSION 0x{R.VERSION_VALUE:08x}, ring {size} words at 0x{base:08x}"


def step_end(be, dev):
    n0 = be.read32(R.END_COUNT)
    end = dev.submit([isa.C("END")])
    t0 = time.monotonic()
    while be.read32(R.END_COUNT) == n0:
        check(time.monotonic() - t0 < 5, f"END_COUNT stuck at {n0} (STATUS 0x{be.read32(R.STATUS):04x})")
    check(be.read32(R.IRQ_STATUS) & R.IRQ_END, "IRQ_STATUS bit 0 not set after END")
    dev.wait(end)
    check(not be.read32(R.IRQ_STATUS) & R.IRQ_END, "IRQ_STATUS bit 0 not cleared by W1C")
    return f"END_COUNT {n0} -> {be.read32(R.END_COUNT)}"


def _scratch(dev):
    ring_addr, ring_bytes = dev.comp.img.regions["cmd_ring"]
    check(ring_bytes >= 4 * MiB, "needs the default 4 MiB command ring")
    off = ring_addr - dev.comp.img.base + ring_bytes - 2 * MiB
    return off, dev.comp.img.base + off             # buffer offset, unrelocated DDR address


def step_dma(be, dev, nbytes=16384):
    off, ddr = _scratch(dev)
    rng = np.random.default_rng(7)
    caps = {isa.ABUF: dev.comp.hw.abuf, isa.WBUF: dev.comp.hw.wbuf, isa.ACC: dev.comp.hw.acc,
            isa.PBUF: dev.comp.hw.pbuf, isa.KVBUF: dev.comp.hw.kvbuf}
    for (buf, cap), (lch, sch) in zip(caps.items(), ((0, 1), (1, 0), (0, 0), (1, 1), (0, 1))):
        # both DMA channels (md_axi = chan 0, md1_axi = chan 1, ADR-005) load and store
        n = min(nbytes, cap)
        src = rng.integers(0, 256, n, dtype=np.uint8)
        dev.mem[off:off + n] = src
        dev.mem[off + MiB:off + MiB + n] = 0
        be.flush()
        sp = cap - n                                   # top of the buffer: exercises the high line addresses
        run_prog(dev, [isa.C("DMA_LOAD", chan=lch, sp_buf=buf, sp_addr=sp, ddr_addr=ddr, rows=1, row_bytes=n,
                             ddr_stride=n, sp_stride=0),
                       isa.C("DMA_STORE", chan=sch, sp_buf=buf, sp_addr=sp, ddr_addr=ddr + MiB, rows=1, row_bytes=n,
                             ddr_stride=n, sp_stride=0)])
        be.invalidate()
        got = np.asarray(dev.mem[off + MiB:off + MiB + n])
        bad = np.flatnonzero(got != src)
        check(not len(bad), f"{isa.BUF_NAMES[buf]} (load ch{lch}, store ch{sch}): {len(bad)} of {n} bytes "
                            f"differ, first at {bad[:1].tolist()}")
    return f"{nbytes} B through each of ABUF WBUF ACC PBUF KVBUF, both DMA channels"


def step_bandwidth(be, dev, mhz):
    off, ddr = _scratch(dev)
    out = []

    def mib(op, ch, base):
        # 16 rows of 64 KiB from consecutive DDR; on chip the rows overlap (sp_stride = 1 line), so the
        # transfer fits every ABUF size
        return isa.C(op, chan=ch, sp_buf=isa.ABUF, sp_addr=0, ddr_addr=base, rows=16, row_bytes=65536,
                     ddr_stride=65536, sp_stride=1)

    cases = [("load ch0", [mib("DMA_LOAD", 0, ddr)]), ("load ch1", [mib("DMA_LOAD", 1, ddr)]),
             ("load ch0+ch1", [mib("DMA_LOAD", 0, ddr), mib("DMA_LOAD", 1, ddr + MiB)]),
             ("store ch0", [mib("DMA_STORE", 0, ddr)])]
    for name, cmds in cases:
        dev.clear_counters()
        run_prog(dev, cmds)
        p = dev.perf()
        nb = MiB * len(cmds)
        check(p["bytes"] == nb, f"{name}: PERF_BYTES {p['bytes']} != {nb}")
        gbs = nb / max(p["cycles"], 1) * mhz * 1e6 / 1e9
        out.append(f"{name} {gbs:.2f} GB/s")
    return ", ".join(out)


def step_layer(be, dev_layer):
    dev, g, cache = dev_layer
    ref = json.loads((TINY / "golden_ref.json").read_text())
    prompt = ref["prompt"][:4]
    logits = dev.step([prompt])[0]
    want = g.forward(prompt, cache)
    bad = np.argwhere(logits != want)
    check(not len(bad), f"1-layer logits differ at {len(bad)} entries, first {bad[:1].tolist()}")
    return f"4-token prefill of a 1-layer cut, {logits.size} logits bit-exact; saturation {dev.saturation()}"


def step_generate(dev, n):
    ref = json.loads((TINY / "golden_ref.json").read_text())
    want = ref["bundles"]["w8a8"]["tokens"][:n]
    t0 = time.monotonic()
    toks = dev.generate(ref["prompt"], len(want))
    dt = time.monotonic() - t0
    check(toks == want, f"tokens {toks} != golden_ref {want}")
    return f"{len(toks)} tokens == golden_ref.json in {dt:.2f} s host time; perf {dev.perf()}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=("kv260", "pynq-z2"), default="kv260")
    ap.add_argument("--backend", choices=("sim", "pynq"), default="sim")
    ap.add_argument("--bitstream", default="nanoxl.bit")
    ap.add_argument("--ip", default="nanoxl_0")
    ap.add_argument("--n", type=int, default=32, help="greedy tokens in step 6 (<= len(golden_ref))")
    ap.add_argument("--skip", default="", help="comma-separated step names to skip")
    a = ap.parse_args(argv)
    hw = hwconfig(a.board, kv_onchip=True)
    be = make_backend(a.backend, a.board, a.bitstream, a.ip)
    A, meta = load_bundle(TINY / "w8a8.npz")
    dev = NanoXL(be, A, meta, hw, name="w8a8")
    skip = set(a.skip.split(","))
    results = []

    def fresh_be():
        # the CSR emulator holds one DDR buffer per instance; on the board a second pynq.allocate is fine
        return make_backend("sim", a.board) if a.backend == "sim" else be

    def layer_dev():
        m = dict(meta, n_layer=1)
        d = NanoXL(fresh_be(), A, m, hw, name="w8a8-1layer")
        g = Golden(A, m)
        return d, g, g.new_cache()

    steps = [("csr", lambda: step_csr(be, dev)), ("end", lambda: step_end(be, dev)),
             ("dma", lambda: step_dma(be, dev)), ("bandwidth", lambda: step_bandwidth(be, dev, CLOCK_MHZ[a.board])),
             ("layer", lambda: step_layer(be, layer_dev())),
             ("generate", lambda: step_generate(NanoXL(fresh_be(), A, meta, hw, name="w8a8"), a.n))]
    ok = True
    for i, (name, fn) in enumerate(steps, 1):
        if name in skip:
            print(f"[{i}] {name:9s} SKIP")
            continue
        try:
            msg = fn()
            print(f"[{i}] {name:9s} PASS  {msg}")
            results.append((name, True, msg))
        except Exception as e:  # noqa: BLE001 -- report and stop
            print(f"[{i}] {name:9s} FAIL  {type(e).__name__}: {e}")
            results.append((name, False, str(e)))
            ok = False
            break
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
