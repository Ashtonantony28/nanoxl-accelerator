# Frozen interface contracts

The executable form of C1–C2 is `nanoxl/numerics.py`; its tests are `tests/py/test_numerics.py`.
A contract changes only through an ADR in `docs/adr/` that updates this file, the Python, and every RTL test in one commit.

## C1 GEMM (INT8)
`A int8 [M,K] @ W int8 [K,N] -> int32 [M,N]`, exact. K ≤ 4096, so no accumulator overflow is possible (|sum| ≤ 2^26).
Weight-stationary orientation: W is the stationary operand; A rows stream through; one output row per A row.

## C1b INT4 weights (W4A8)
INT4 values in [-8, 7], packed two per byte along N: byte j of row k = `W[k,2j]` in bits [3:0], `W[k,2j+1]` in bits [7:4], two's complement. Product with INT8 activation is exact, same int32 accumulation as C1. Per-output-channel scales are applied at requant time. Per-K-group uint8 scales (W4A8, GEMM_GS) are applied inside the array controller at each group flush per ADR-004: ACC = (acc ? ACC : 0) + gm[g,c]·psum_g[c].

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
Owned by SG2 (`rtl/array_ctrl.sv`, details and measurements in `docs/array.md`); `gs` added by
SG7a per ADR-004 (additive: `cmd_gs = 0` behaves exactly as before).
Clocking: single `clk`, synchronous active-low `rst_n` (reset clears all control state,
weights and in-flight rows; accumulator contents are undefined after reset).
Parameters: `N` (array N×N), `M_DEPTH` (accumulator rows, power of 2), `OF_DEPTH` (output
FIFO, ≥ 2N+5 for full throughput; default 2N+8), `GM_DEPTH` (group-scale rows in flight,
default N+4; any value ≥ 1 is correct, ≥ N+3 gives full throughput).

**Handshake.** Every stream is valid/ready; a beat transfers on a rising edge with
`valid && ready`. Producers must hold valid and data until the transfer. `ready` may depend
combinationally on other inputs (`cmd_ready` on `w_valid`).

**Command** (`cmd_*`, one per weight tile): `m` ∈ [1, M_DEPTH] activation rows, `k` ∈ [1, N]
weight rows (K extent), `n` ∈ [1, N] output columns, `abase` accumulator row base,
`acc` (0: overwrite, 1: add into rows), `emit` (stream the resulting rows out), `int4`,
`gs` (W4A8 group scales, ADR-004; only with `int4=1`).
Semantics: for m' in 0..m-1, column c,
`ACC[(abase+m') mod M_DEPTH][c] = (acc ? ACC[...][c] : 0) + s[c] · (A[m', 0..k-1] @ W[0..k-1, :])[c]`
with W columns ≥ n treated as 0, exact INT32 (C1/C1b), where `s[c]` = 1 when `gs=0` and the
uint8 multiplier of the scale beat when `gs=1` (the product is exact: |psum| ≤ N·2^10 for an INT4
tile, so s·psum < 2^25).

**Weight stream** (`w_data`, N×8 bits): exactly `k` beats per command (`k+1` when `gs=1`),
commands in order; beat r = W[r, :]. INT8: byte c = W[r,c]. INT4: nibble c (bits [4c+3:4c]) =
W[r,c] (C1b order), upper N×4 bits ignored. Lanes c ≥ n are ignored. With `gs=1`, beat 0 is the
scale row (byte c = `s[c]`, uint8) and beats 1..k are the weight rows; the scale beat may be
accepted while the previous tile still occupies the bank.

**Activation stream** (`a_data`, N×8 bits): exactly `m` beats per command, in order;
beat m' byte r = A[m', r]; bytes r ≥ k are ignored.

**Output stream** (`out_data`, N×32 bits, `out_last`): for each `emit=1` command, exactly
`m` beats in row order m' = 0..m-1, column c = INT32 result (two's complement) of column c,
`out_last=1` on the beat of row m-1. Columns ≥ n are 0 when every command in the
accumulation chain used the same n. Results of different commands appear in command order.
`out_ready` may be deasserted at any time for any length; nothing is lost or reordered.

**Ordering and overlap.** Commands execute in order. The weight tile of command i+1 is
loaded while command i streams (double-buffered PE banks); a later command's `acc=1`
always sees the result of every earlier command on the same rows.

**Timing** (no bubbles, `out_ready=1`): first `a` beat accepted → first `out` beat accepted
= 2N+4 cycles (also with `gs=1`: the scale multiply sits in the existing RMW stage); steady
state max(m, k) cycles per command, max(m, k+1) with `gs=1` (one `w` and one `a` beat per
cycle, zero inter-tile bubbles). The core `systolic_array` alone has fixed latency
2N-1 from `in_valid` to `out_valid` and never stalls.

**Status.** `idle` = no command loading/issuing, no row in flight and no un-popped output.
`hazard_seen` / `overflow_seen` are simulation-only sticky checkers (tied 0 in synthesis);
`overflow_seen` also covers the group-scale FIFO (overflow, or a `gs` row reaching the
accumulator without its scale row).

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
