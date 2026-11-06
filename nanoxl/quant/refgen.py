"""Regenerate the committed tiny-model integer bundles and the golden reference token sequences.

  python3 -m nanoxl.quant.refgen      (writes artifacts/models/tiny/{w8a8,w4a8}.npz, golden_ref.json)

The bundles are the frozen inputs of every downstream test (compiler, RTL). Regenerating them
is a deliberate act: the reference sequences then change and must be re-committed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanoxl import golden  # noqa: E402
from nanoxl.export import quantize_checkpoint, save_bundle  # noqa: E402
from nanoxl.model_fp import REPO, load_shakespeare  # noqa: E402
from nanoxl.quant.quantize import QuantConfig  # noqa: E402

TINY = REPO / "artifacts" / "models" / "tiny"
PROMPT_LEN = 16
N_NEW = 100


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    _, val, chars = load_shakespeare()
    prompt = [int(t) for t in val[:PROMPT_LEN]]
    ref = {"prompt": prompt, "prompt_text": "".join(chars[t] for t in prompt), "n_new": N_NEW,
           "ckpt_sha256": sha(TINY / "ckpt.pt"), "bundles": {}}
    for name, wb in (("w8a8", 8), ("w4a8", 4)):
        A, meta, _ = quantize_checkpoint(TINY / "ckpt.pt", QuantConfig(wbits=wb))
        meta["source_ckpt"] = "artifacts/models/tiny/ckpt.pt"
        path = TINY / f"{name}.npz"
        save_bundle(path, A, meta)
        g = golden.Golden(A, meta)          # contract GEMM (nx.gemm_i8)
        toks = g.generate(prompt, N_NEW)
        ref["bundles"][name] = {"file": f"{name}.npz", "sha256": sha(path), "tokens": toks,
                                "text": "".join(chars[t] for t in toks)}
        print(name, repr(ref["bundles"][name]["text"]))
    (TINY / "golden_ref.json").write_text(json.dumps(ref, indent=1))


if __name__ == "__main__":
    main()
