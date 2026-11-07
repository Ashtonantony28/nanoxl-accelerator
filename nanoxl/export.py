"""Quantized model bundle export / load (schema: docs/adr/ADR-002-quantization.md §Bundle).

  python3 -m nanoxl.export --ckpt artifacts/models/tiny/ckpt.pt --out artifacts/models/tiny/w8a8.npz
  python3 -m nanoxl.export --ckpt ... --wbits 4 --out .../w4a8.npz

The npz holds every integer array of the bundle under its name plus `__meta__`, a uint8
array containing UTF-8 JSON (config, quant options, float scales for debugging/dequant).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CALIB_SEED = 20261106     # calibration windows come from the TRAIN split with this seed


def save_bundle(path, arrays: dict, meta: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8)
    np.savez_compressed(path, __meta__=blob, **arrays)


def load_bundle(path):
    z = np.load(path)
    meta = json.loads(bytes(z["__meta__"]).decode())
    arrays = {k: z[k] for k in z.files if k != "__meta__"}
    return arrays, meta


def calib_windows(train: np.ndarray, block: int, n: int, seed: int = CALIB_SEED):
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(train) - block - 1, n)
    return np.stack([train[i:i + block] for i in ix]).astype(np.int64)


def quantize_checkpoint(ckpt, qc=None):
    from nanoxl.model_fp import load_checkpoint, load_shakespeare, numpy_weights
    from nanoxl.quant.quantize import QuantConfig, build_bundle
    qc = qc or QuantConfig()
    model = load_checkpoint(ckpt)
    train, _, _ = load_shakespeare()
    cal = calib_windows(train, model.config.block_size, qc.calib_seqs)
    A, meta = build_bundle(numpy_weights(model), model.config, cal, qc)
    meta["source_ckpt"] = str(ckpt)
    return A, meta, model


def main(argv=None):
    from nanoxl.quant.quantize import QuantConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wbits", type=int, default=8)
    ap.add_argument("--res-bits", type=int, default=16)
    ap.add_argument("--smooth", type=float, default=None)
    a = ap.parse_args(argv)
    qc = QuantConfig(wbits=a.wbits, res_bits=a.res_bits)
    if a.smooth is not None:
        qc.smooth_alpha = a.smooth
    A, meta, _ = quantize_checkpoint(a.ckpt, qc)
    save_bundle(a.out, A, meta)
    print(f"wrote {a.out}: {len(A)} arrays, {sum(v.nbytes for v in A.values())} bytes")


if __name__ == "__main__":
    main()
