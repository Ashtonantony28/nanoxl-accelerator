# ADR-002: Quantization scheme, integer nonlinear ops (C5), model bundle

Status: accepted (SG1 / M0, 2026-11-06). Freezes contract C5 in `docs/contracts.md`.
Headline (docs/quant-study.md): W8A8 ΔPPL +0.96 % (tiny) / +0.81 % (shakespeare); W4A8 +3.29 % / +0.74 %. Both G2 targets (+3 % / +8 %) are met.
INT8 residual: +17.5 % / +890 %. SmoothQuant α=0.5: within ±0.1 % (off).
Evidence: `docs/quant-study.md` (numbers), `nanoxl/golden.py` + `nanoxl/quant/ops.py` (executable spec),
`tests/py/test_golden_*.py` (bounds, determinism, KV-cache equivalence).

## Decision

| Item | Decision | Alternatives measured (quant-study) |
|---|---|---|
| FP model | nanoGPT GPT-2: LayerNorm(bias, eps 1e-5), **exact erf GELU**, learned pos-emb, MHA, tied LM head (`nanoxl/model_fp.py`, weight-compatible with nanoGPT state_dict; nanoGPT-extended's `ff.` key accepted) | — |
| Weights W8 | symmetric per-output-channel INT8 in [-127,127], `s_w[n] = max|W[:,n]|/127` | — |
| Weights W4 (W4A8) | symmetric INT4 [-8,7], **group 64 along K**, MSE clip search per group; group scale = `gm[g,n] * unit[n]`, `gm` uint8 in [1,255], `unit[n] = max_g s_g[n]/255`. LM head and embeddings stay INT8/INT16 | no clip search |
| Activations | INT8 symmetric, **static per-tensor scales** from absmax over 16 calibration windows of the *train* split (seed 20261106) | 99.99th percentile; SmoothQuant α=0.5 |
| Residual stream | **INT16**, one global scale `s_res` = 1.25 × calib absmax / 32767 | INT8 residual |
| Softmax | log2-domain Q.8 int16 logits, **256-entry exp2 LUT + shift**, reciprocal LUT + 1 Newton step, **P int8 in [0,127] (scale 1/127)**; exact 3-pass (max, exp-sum, normalize) over the full row | I-BERT shift+poly exp |
| LayerNorm | integer mean via C2 multiply (1/d), 48-bit Σc², **rsqrt LUT (256) + 1 Newton step**, z in Q.10 int16, per-channel affine `g` int16 / `b` int32 + rounding shift | — |
| GELU | **256-entry LUT per layer**, indexed by the INT8 c_fc output (exact on the INT8 grid) | I-BERT 2nd-order poly on an int16 input |
| LM head | INT8 GEMM vs tied `wte^T` (per-vocab-row scale) → **int16 logits** at one scale; greedy = argmax over int16, ties → lowest index | — |
| SmoothQuant | **off** (no measurable gain on these models; see study). Bundle format supports it (folded into γ/β and W rows). | α = 0.5 |

Requantization everywhere is C2 (gemmlowp), with `mult/shift` per output channel where the weight is per-channel.
All requant multipliers are asserted to be in (0,1); the one place that can exceed 1 (attention scores)
gets an exact left pre-shift `ls` of the int32 accumulator (`score.ls`, 0 for both trained models).

## Why
- INT16 residual is free in area (adds only) and removes residual saturation; INT8 residual costs measurable PPL (study).
- LUT GELU is exact on its INT8 domain (max 0.5 LSB, gain 1.000) and costs 256 B per layer; the poly is worse and needs an int16 c_fc output.
- exp2 LUT + shift is ≤1.5 Q.15 LSB on all 65,536 inputs and needs no multiplier; the shift+poly is two multipliers and has a 64× larger max error (64 vs 1 Q.15 LSB). PPL is the same within noise.
- P = int8 [0,127] keeps the P@V GEMM inside C1 (signed INT8 array); scale 1/127 (not 1/128) makes p=1.0 exact, so there is no gain bias for peaked rows.
- Static per-tensor scales keep the vector unit free of a max-reduction before each GEMM (dynamic per-token is an M8 option).

## Consequences (for other sub-goals)
- **SG5 (SFU RTL)** implements exactly C5. LUT *contents* come from the bundle (`lut.*`, `h{l}.gelu.lut`); RTL never regenerates them.
- **SG2/SG4 (W4A8)**: group scales are applied to **per-group partial sums**: `acc[n] = Σ_g gm[g,n] · acc_g[n]` (int32: |acc_g| ≤ 64·128·8 = 2^16, × gm ≤ 255, × K/64 ≤ 64 groups < 2^30 for K ≤ 4096). With group 64 and an N=32 array, each K-tile lies inside one group, so the accumulator path needs a `uint8 × int32` multiply-accumulate per K-tile flush (or K-tiles of 64). Bias and requant then use `unit[n]`.
- **SG8 (online softmax)**: C5 softmax is defined as a full-row 3-pass. An online/flash variant is *not* bit-identical (per-element floor shifts do not commute with rescaling). ctx ≤ 256 rows fit on chip, so v1 hardware should buffer the int16 score row; an online variant requires a new ADR that changes the golden model.
- **Saturation**: requant outputs, residual adds, LN z and LN outputs are clamped; the golden model counts each (`Golden.sat_count`) and RTL must expose the same events per unit. Calibration is absmax over 16 windows, so a small number of activation saturations on unseen text is expected and measured (study).

## Bundle schema (`nanoxl/export.py`, npz)
`__meta__`: uint8 UTF-8 JSON: `format="nanoxl-bundle"`, `version=1`, `n_layer, n_head, n_embd, vocab_size, block_size`,
`quant` (the QuantConfig), `scales` (float scales per tensor, debug/dequant only), `exp_poly`, `source_ckpt`.
Arrays (L = layer index, d = n_embd, V = vocab, T = block_size; W in C1 orientation [K, N]):

| Key | dtype, shape | Meaning |
|---|---|---|
| `emb.tok`, `emb.pos` | int16 [V,d], [T,d] | embeddings already in the residual scale; `x0 = sat16(tok[t] + pos[p])` |
| `h{L}.ln_1.g`, `.b` | int16 [d], int32 [d] | LN affine; `.c` int64[5] = `mu_mult, mu_shift, sqrt_d, eps_q, sh` |
| `h{L}.ln_2.*`, `ln_f.*` | same | |
| `h{L}.c_attn.w` | int8 [d,3d] | QKV weight (W8) — or `.w4` uint8 [d,3d/2] (C1b packing) + `.gm` uint8 [d/64,3d] (W4) |
| `h{L}.c_attn.bias/.mult/.shift` | int32 [3d] | requant to q,k,v int8 (per-channel; q/k/v blocks have their own output scale) |
| `h{L}.score.mult/.shift/.ls` | int32 [1] | int32 q·k → int16 Q.8 log2 logits (`ls` = exact left pre-shift) |
| `h{L}.pv.mult/.shift` | int32 [1] | int32 P@V → int8 attention output |
| `h{L}.attn_proj.{w|w4,gm},bias,mult,shift` | | → int16 residual delta |
| `h{L}.c_fc.*` | | → int8 GELU input |
| `h{L}.gelu.lut` | int8 [256] | addressed by the INT8 input byte (`x & 0xFF`) |
| `h{L}.mlp_proj.*` | | → int16 residual delta |
| `lm_head.w/bias/mult/shift` | int8 [d,V], int32 [V] ×3 | → int16 logits (bias = 0) |
| `lut.exp2`, `lut.recip`, `lut.rsqrt` | uint16 [256] | shared SFU tables |

Committed bundles: `artifacts/models/tiny/{w8a8,w4a8}.npz` (+ `golden_ref.json`, the reference token sequences).
Regenerate with `python3 -m nanoxl.quant.refgen`; a regeneration is a contract event (re-commit the reference).
