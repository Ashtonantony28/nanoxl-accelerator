# M0 quantization study (SG1)

Decision record: `docs/adr/ADR-002-quantization.md`. Contract: C5 in `docs/contracts.md`.
Raw data: `artifacts/models/tiny/study.json`, `artifacts/models/shakespeare/study.json`
(produced by `python3 -m nanoxl.quant.study`; tables below by `--render`).

## Method
- **Data:** Tiny Shakespeare (char-rnn `input.txt`, sha256 `86c4e6aa…565ed`), vocab 65, 90/10 split as nanoGPT `prepare.py`.
- **FP32 metric:** mean cross-entropy over the *whole* val split in non-overlapping windows of `block_size` (every window starts with empty context; FP32 and integer use identical windows).
- **Integer metric:** the golden model (`nanoxl/golden.py`) on an evenly spaced subset of those windows (the integer model is numpy, so the full split for every variant would take hours); FP32 is reported on the *same* subset, and ΔPPL is relative to that.
- **Calibration:** 16 windows from the **train** split (seed 20261106); evaluation never sees calibration data.
- **Seeds:** study/tuning seed 7 (`STUDY_SEED`) and seeds 0–9; the tests' `HELDOUT_SEED = 0x5EED2026` was never used to choose any constant or threshold (test floors were set from study numbers, then margin).
- **Per-op tables:** each integer op is fed the FP32 activation quantized at that op's input scale and compared against the float op on the *same dequantized input*, in output LSB (max |err|, mean err, rms, gain = LS slope through the origin). "incl. input quant" rows compare against the float op on the unquantized FP32 input.
- **Greedy agreement:** 200 new tokens, 16-char prompts at val offsets len/7·{1,2,3}. FP32 greedy uses torch; integer uses the golden model (KV cache when prompt+200 ≤ ctx, otherwise sliding-window recompute, as for `tiny` with ctx 128).

RESULTS_PLACEHOLDER
