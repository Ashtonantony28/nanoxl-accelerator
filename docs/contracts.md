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
Owned by SG2 (`rtl/array_ctrl.sv`, details and measurements in `docs/array.md`).
Clocking: single `clk`, synchronous active-low `rst_n` (reset clears all control state,
weights and in-flight rows; accumulator contents are undefined after reset).
Parameters: `N` (array N×N), `M_DEPTH` (accumulator rows, power of 2), `OF_DEPTH` (output
FIFO, ≥ 2N+5 for full throughput; default 2N+8).

**Handshake.** Every stream is valid/ready; a beat transfers on a rising edge with
`valid && ready`. Producers must hold valid and data until the transfer. `ready` may depend
combinationally on other inputs (`cmd_ready` on `w_valid`).

**Command** (`cmd_*`, one per weight tile): `m` ∈ [1, M_DEPTH] activation rows, `k` ∈ [1, N]
weight rows (K extent), `n` ∈ [1, N] output columns, `abase` accumulator row base,
`acc` (0: overwrite, 1: add into rows), `emit` (stream the resulting rows out), `int4`.
Semantics: for m' in 0..m-1,
`ACC[(abase+m') mod M_DEPTH][0..N-1] = (acc ? ACC[...] : 0) + A[m', 0..k-1] @ W[0..k-1, 0..N-1]`
with W columns ≥ n treated as 0, exact INT32 (C1/C1b).

**Weight stream** (`w_data`, N×8 bits): exactly `k` beats per command, commands in order;
beat r = W[r, :]. INT8: byte c = W[r,c]. INT4: nibble c (bits [4c+3:4c]) = W[r,c]
(C1b order), upper N×4 bits ignored. Lanes c ≥ n are ignored.

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
= 2N+4 cycles; steady state max(m, k) cycles per command (one `w` and one `a` beat per
cycle, zero inter-tile bubbles). The core `systolic_array` alone has fixed latency
2N-1 from `in_valid` to `out_valid` and never stalls.

**Status.** `idle` = no command loading/issuing, no row in flight and no un-popped output.
`hazard_seen` / `overflow_seen` are simulation-only sticky checkers (tied 0 in synthesis).

## C4 Command ISA and DDR image format
Owned by SG4; lives in `docs/isa.md` and `nanoxl/isa.py`.

## C5 Nonlinear ops (softmax, LayerNorm, GELU, residual width)
Owned by SG1 (M0 quantization study); frozen by ADR-002 when SG1 lands. RTL for these waits for that ADR.
