# Frozen interface contracts

The executable form of C1–C2 is `nanoxl/numerics.py`; its tests are `tests/py/test_numerics.py`.
A contract changes only through an ADR in `docs/adr/` that updates this file, the Python, and every RTL test in one commit.

## C1 GEMM (INT8)
`A int8 [M,K] @ W int8 [K,N] -> int32 [M,N]`, exact. K ≤ 4096, so no accumulator overflow is possible (|sum| ≤ 2^26).
Weight-stationary orientation: W is the stationary operand; A rows stream through; one output row per A row.

## C1b INT4 weights (W4A8)
INT4 values in [-8, 7], packed two per byte along N: byte j of row k = `W[k,2j]` in bits [3:0], `W[k,2j+1]` in bits [7:4], two's complement. Product with INT8 activation is exact, same int32 accumulation as C1. Group scales are applied at requant time (per output channel / group mult+shift), never inside the array.

## C2 Requantization (int32 -> int8 / int16)
```
x   = wrap32(acc + bias)                          # bias int32, per output channel
y   = RoundingDivideByPOT(SRDHM(x, mult), shift)  # gemmlowp semantics
out = clamp(y, -2^(B-1), 2^(B-1)-1),  B ∈ {8,16}
```
- `mult` int32 in [2^30, 2^31-1], `shift` in [0, 31]; both may be per output channel.
- SRDHM: `INT32_MIN*INT32_MIN -> INT32_MAX`; otherwise `(a*b + nudge) / 2^31` with C truncating division, nudge = 2^30 if a*b ≥ 0 else 1-2^30 (so exact halves round toward +inf).
- RoundingDivideByPOT: `(x >> e) + (rem > thr)`, `rem = x & (2^e-1)`, `thr = ((2^e-1)>>1) + (x<0)` (halves round away from zero).
- Saturation events must be countable in RTL (sticky counter per unit).

## C3 Systolic array port protocol
Owned by SG2; written into this section when SG2 lands.

## C4 Command ISA and DDR image format
Owned by SG4; lives in `docs/isa.md` and `nanoxl/isa.py`.

## C5 Nonlinear ops, residual stream, model dataflow (frozen by ADR-002)
Executable form: `nanoxl/quant/ops.py` (ops) and `nanoxl/golden.py` (dataflow); tests `tests/py/test_golden_*.py`.
All arithmetic is two's-complement integer. `>>` = arithmetic shift (floor). `rshr(x,k) = (x + 2^(k-1)) >> k` (round half up; `x` when k = 0).
`rdivpot`, `srdhm`, `requant` are C2. `satN` clamps to signed N bits and raises the unit's saturation event. LUT contents are data from the bundle (ADR-002 schema); RTL loads them, it does not compute them.

**Dataflow per token row** (row-wise except attention; widths in brackets):
```
x      = sat16(emb.tok[t] + emb.pos[p])                                   [int16 residual]
per layer:
  h    = LN(x; ln_1)                                                      [int8]
  qkv  = requant8(gemm(h, c_attn.w) + bias; per-channel mult/shift)       [int8 q|k|v]; K,V rows -> KV cache
  per head: s = requant16((q·K^T) << ls; score.mult/shift)                [int16, Q.8 log2 units]
            P = SOFTMAX(s over keys 0..p)                                 [int8 0..127]
            y = requant8(P @ V; pv.mult/shift)                            [int8]
  x    = sat16(x + requant16(gemm(y, attn_proj.w) + bias))                [int16]
  h    = LN(x; ln_2);  u = requant8(gemm(h, c_fc.w) + bias)               [int8]
  g    = gelu.lut[u & 0xFF]                                               [int8]
  x    = sat16(x + requant16(gemm(g, mlp_proj.w) + bias))                 [int16]
h = LN(x; ln_f);  logits = requant16(gemm(h, lm_head.w))                  [int16]; next = argmax (ties -> lowest index)
```
W4A8 linear: `acc[n] = Σ_g gm[g,n] · gemm(h[:, 64g:64g+64], W4[64g:64g+64, n])` (int32), then the same requant.

**EXP2** (input `d` uint16 = row max − score, output Q.15 uint16 ≤ 32768):
`q = d >> 8; r = d & 255; e = (q >= 16) ? 0 : lut.exp2[r] >> q`, with `lut.exp2[r] = round(2^15 · 2^(−r/256))`.

**RECIP** (input `S` ≥ 2^15, ≤ 2^24): `b = bitlen(S)` (≥ 16); `m = S >> (b − 16)` ∈ [2^15, 2^16);
`y0 = lut.recip[(m >> 7) − 256]` (`= round(2^31 / ((256+i)·128 + 64))`); `y = (y0 · (2^32 − m·y0)) >> 31` (17-bit result; `m·y0` 33 bits, product 48 bits).
Meaning: `1/S ≈ y · 2^−(15+b)`.

**SOFTMAX** over kept (causal) positions j of one row, input int16 `s_j`:
1. `M = max_j s_j`; 2. `d_j = M − s_j` (uint16); 3. `e_j = EXP2(d_j)`; 4. `S = Σ e_j` (24-bit; S ≥ 32768);
5. `(y, b) = RECIP(S)`; 6. `p_j = min(127, rshr(e_j · (127·y), 15 + b))` (product ≤ 2^39). Masked positions: p = 0.
Defined as this exact 3-pass over the whole row (an online variant is not bit-identical; needs a new ADR).

**RSQRT** (input `Q` ≥ 1, ≤ 2^46): `e = bitlen(Q) − 1; k = (e − 14) >> 1` (floor, may be negative);
`m = k ≥ 0 ? Q >> 2k : Q << −2k` ∈ [2^14, 2^16); `y0 = lut.rsqrt[m >> 8]` (`= round(2^14 / sqrt((i+0.5)/256))`, i ≥ 64);
`t = (y0·y0) >> 14; u = (m·t) >> 16; y = (y0 · (3·2^14 − u)) >> 15`. Meaning: `1/sqrt(Q) ≈ y · 2^−(22+k)`.

**LN** (input int16 `x[0..d)`; constants `.c = [mu_mult, mu_shift, sqrt_d, eps_q, sh]`, per-channel `g` int16, `b` int32):
1. `S = Σ x` (int32); 2. `mu = rdivpot(srdhm(S, mu_mult), mu_shift)` (≈ round(S/d); mu_mult/shift = C2 encoding of 1/d);
3. `c_i = x_i − mu` (int17); 4. `Q = Σ c_i² + eps_q` (48-bit unsigned; `Q = max(Q, 1)`);
5. `(y, k) = RSQRT(Q)`; `R = y · sqrt_d` (`sqrt_d = round(sqrt(d)·2^10)`, R < 2^31);
6. `z_i = sat16(rshr(c_i · R, 22 + k))` (product < 2^49; z = normalized value in Q.10);
7. `out_i = sat8(rdivpot(z_i · g_i + b_i, sh))` (40-bit accumulator).

**GELU**: `out = lut[x & 0xFF]` (int8 in, int8 out), one 256-entry table per layer.

**Saturation events** (each countable): every requant clamp, both residual adds, LN `z` (step 6) and LN out (step 7). Softmax and the SFUs cannot overflow by construction (bounds above).
