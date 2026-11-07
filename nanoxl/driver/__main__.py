"""CLI: python3 -m nanoxl.driver --bundle artifacts/models/tiny/w8a8.npz --prompt "ROMEO:" --n 50

--backend sim (default) runs the command ring in the funcsim-backed CSR emulator; --backend pynq
drives the board (needs --bitstream; untested until hardware exists, docs/driver.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from ..hwconfig import BOARDS
from . import CharCodec, PynqBackend, SimBackend, open_bundle


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m nanoxl.driver")
    ap.add_argument("--bundle", required=True, help="SG1 bundle .npz")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="prompt text (tiny-shakespeare characters)")
    g.add_argument("--tokens", help="prompt as comma-separated token ids")
    ap.add_argument("--n", type=int, default=50, help="new tokens")
    ap.add_argument("--backend", choices=("sim", "pynq"), default="sim")
    ap.add_argument("--board", choices=sorted(BOARDS), default="kv260")
    ap.add_argument("--bitstream", help="overlay .bit (pynq backend)")
    ap.add_argument("--ip", default="nanoxl_0", help="AXI-Lite IP name in the overlay (pynq backend)")
    ap.add_argument("--top-k", type=int, default=None, help="top-k sampling (default greedy)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None, help="prefill chunk (tokens per program)")
    ap.add_argument("--json", action="store_true", help="print a JSON record instead of text")
    a = ap.parse_args(argv)
    hw = BOARDS[a.board]
    if a.backend == "pynq":
        if not a.bitstream:
            ap.error("--backend pynq needs --bitstream")
        be = PynqBackend(a.bitstream, a.ip)
    else:
        be = SimBackend(hw)
    dev = open_bundle(a.bundle, be, hw)
    codec = CharCodec()
    prompt = [int(t) for t in a.tokens.split(",")] if a.tokens else codec.encode(a.prompt)
    t0 = time.monotonic()
    toks = dev.generate(prompt, a.n, top_k=a.top_k, temperature=a.temperature, seed=a.seed, chunk=a.chunk)
    dt = time.monotonic() - t0
    p = dev.perf()
    if a.json:
        print(json.dumps({"prompt": prompt, "tokens": toks, "text": codec.decode(toks), "programs": dev.programs,
                          "host_seconds": round(dt, 3), "perf": p, "saturation": dev.saturation()}))
    else:
        sys.stdout.write(codec.decode(prompt) + codec.decode(toks) + "\n")
        mhz = hw.clock_mhz
        sys.stderr.write(f"[{a.backend}] {len(toks)} tokens, {dev.programs} programs, {dt:.2f} s host time; "
                         f"modelled device time {p['cycles'] / mhz / 1e3:.2f} ms at {mhz:.0f} MHz, "
                         f"{p['bytes'] / 1e6:.1f} MB DMA\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
