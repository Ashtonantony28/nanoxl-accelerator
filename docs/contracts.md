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

## C5 Nonlinear ops (softmax, LayerNorm, GELU, residual width)
Owned by SG1 (M0 quantization study); frozen by ADR-002 when SG1 lands. RTL for these waits for that ADR.
