# ADR-004: W4A8 group scales in the array, SFU table loading, alignment rule (ISA additions)

Status: **accepted and implemented** (proposed by SG4b / M1 close-out, 2026-11-07; implemented in
C1b/C3 by `rtl/array_ctrl.sv` and `rtl/array_seq.sv`, SG7a, wave 3; status updated after AUDIT-1
A1-06). Extends contract C4 (`docs/isa.md`) additively. When this ADR was written no RTL was changed;
the C3 additions below are now built.
Evidence: `tests/py/test_funcsim_golden.py` (compiled tiny W8A8 and W4A8 bit-identical to
`nanoxl.golden`: logits, KV cache, saturation counts, 100-token sequences), `tests/py/test_isa.py`
(new golden vectors), `tests/py/test_cyclemodel.py::test_cyclemodel_gemm_gs_cost`.

## Context

ADR-002 fixed W4A8 as INT4 weights with **group-64 uint8 multipliers** along K:
`acc[n] = Σ_g gm[g,n] · acc_g[n]`, `acc_g = A[:, 64g:64g+64] @ W4[64g:64g+64, n]`, then bias and the
per-channel requant with `unit[n]`. The ISA from SG4 had only per-output-channel scales (GEMM `int4=1`
plus REQUANT), and GEMM has no spare bits (its 13 fields fill all 128 bits). The SG4 float
placeholders for LN and softmax also had no way to give the SFUs their C5 LUTs (`lut.exp2`,
`lut.recip`, `lut.rsqrt`), and the ISA let base addresses be byte-granular while strides are in
16-byte lines.

## Decision

### 1. `GEMM_GS` (new opcode 0x12, ARRAY queue): group scales applied in the accumulator RMW
`C[M,N] (+)= Σ_g gm[g,n] · (A[:, gG:(g+1)G] @ W4[gG:(g+1)G, :])`, exact int32. Fields are GEMM's minus
`int4` (implicit), `w_trans` and `w_buf` (always WBUF), plus `glog2` (3 bits, G = 2^(glog2+3), 8..1024;
3 for ADR-002's 64). The operand block in WBUF is the packed INT4 panel `[K, N/2]` (C1b) followed, at
the next 16-byte line, by `gm uint8 [ceil(K/G), N]` (`isa.gs_layout`). DDR panels use the same layout,
so one dense DMA moves weights and their scales; scales are 1/32 of the INT4 bytes (+3.1 % DDR).

Hardware: the array controller already issues one command per weight tile of K_t ≤ N rows. For
GEMM_GS it walks K-tiles of `min(N, G)` rows, so each tile lies inside one group (with N = 32 and
G = 64, two tiles per group), and the accumulator read-modify-write becomes
`ACC[m][c] = (acc ? ACC[m][c] : 0) + gm_t[c] · psum[m][c]`
instead of `... + psum[m][c]`. Because the group multiplier distributes over the tiles of a group,
scaling each tile partial is exactly `gm[g] · acc_g`. Bounds: |psum| ≤ 32·128·8 = 2^15 for an INT4
tile of 32 rows, so the product is 8 × 17 bits → 25 bits, and the running sum stays < 2^30 for
K ≤ 4096 (ADR-002), so there is no overflow anywhere.

Alternatives considered:
* **VEC op on the accumulator buffer** (GEMM each group into a temporary ACC tile, then a vector
  `ACC += gm ⊙ tmp`): exact, but costs M·N/L vector cycles and two ACC reads + one write per element
  per group. Prefill c_attn of shakespeare: 6 groups × 256·1152/64 = 27.6 k VEC cycles against
  110 k array cycles, +25 % VEC load and ACC port contention, plus a GEMM/VEC semaphore pair per group.
* **K-tiles of 64 with a single scale at flush** (ADR-002's other suggestion): needs N = 64 or a
  2-tile accumulation register in the controller; strictly more state than scaling every tile.
* **GEMM `int4=1, w_trans=1`** (currently rejected) as the flag: no bits for the group size, and SG6's
  decoder already treats the combination as illegal. A new opcode is cleaner and purely additive.

### 2. `LUT_LOAD` (new opcode 0x1E, VEC queue)
Copies 256 uint16 from PBUF into SFU table `table` (0 exp2, 1 recip, 2 rsqrt; 3 reserved). The
compiler emits a 1.5 KiB DMA and three LUT_LOADs in every program's prologue (decode-step overhead
≈ 50 DMA cycles + 48 VEC cycles, hidden by the first weight panel). LN and SOFTMAX_ONLINE read the
tables; they are in the same in-order queue, so no semaphore is needed. The GELU table stays a
per-layer PBUF operand of the existing GELU command (it is indexed per element, one table per layer).

### 3. C5 semantics for existing opcodes (encodings unchanged)
* **LN**: the C5 integer LayerNorm with a 32-byte header block (`mu_mult, mu_shift, sqrt_d, sh`,
  int64 `eps_q`) + `g int16` + `b int32` (`docs/isa.md` §3).
* **SOFTMAX_ONLINE**: requantizes the int32 scores with the 16-byte score record in `ctx_addr`
  (`{0, mult, shift, ls}`; the old pad word now carries `score.ls`) to int16 Q.8, then the C5
  full-row 3-pass softmax; `first = last = 1` is required (C5 is not an online algorithm). Scores are
  computed over all keys of the step, so the score saturation counter matches the golden model.
* **REQUANT** record word 3 must be 0. **VEC ADD** int16 is the saturating residual add.

### 4. Alignment rule
All base-address fields (scratchpad and DDR) are 16-byte aligned (`isa.ALIGNED_FIELDS`). Enforced by
`isa.encode(cmd, aligned=True)` / `isa.check_aligned`, by the compiler on every emitted command, and
by funcsim (strict by default). The bit format still carries the low 4 bits, so the existing golden
decoder vectors (some deliberately unaligned) stay valid bit patterns.

## Cost (cycle model, `nanoxl/cyclemodel.py`)
* **As built (AUDIT-1 A1-05/A1-06; supersedes the estimate in this bullet):** array_seq sends the
  scale beat on **every** GEMM_GS tile (tiles are split at min(N, G) rows), so a weight-bound tile
  costs k_tile + 1 cycles at one weight row per cycle (c_attn N=384 K=128 G=64 at N=32: 1,584 cycles,
  measured = model), and the scale multiply sits in the existing RMW stage with **no extra latency**
  (C3: 2N+4 with `gs` too). The original estimate follows.
* GEMM_GS (original estimate): the scale row is one extra beat of N bytes (= two INT4 weight rows of
  port bandwidth) on the first K-tile of every group. Weight-load bound tiles (GEMV): `ceil((N+2)/(2·rpc))` instead of
  `ceil(N/(2·rpc))` cycles, i.e. 9 vs 8 at N = 32, 2 rows/cycle (+6 % on the array time of W4 decode,
  which is DDR-bound anyway: 512 B per tile = 16 DDR cycles at 32 B/cycle). Array-bound tiles
  (prefill, M ≥ 9): 0 cycles. The multiply sits in the existing RMW stage (+1 pipeline stage of
  latency, no throughput cost). KV260 shakespeare W4A8 decode: 1320 tok/s (was 1360 with
  per-channel-only scales, the difference is mostly the 3 % scale bytes).
* LN / SOFTMAX: 3 passes over the row + a per-row bubble (estimate at the time: SG5 had not published
  SFU timing). Superseded by the measured VEC-engine costs, docs/vector-unit.md §9.3 (AUDIT-1 A1-01).

## What `rtl/array_ctrl.sv` (SG2) needs — proposed C3 addition
1. Command field `cmd_gs` (1 bit). When set, the weight stream carries **k + 1** beats: beat 0 is
   the scale row (byte c = `gm[c]`, uint8, lanes ≥ n ignored), then the k weight rows as today
   (INT4 only). The scale row is latched into the tile's bank alongside its weights (N × 8 bits per bank).
2. RMW stage: `sum[c] = (acc ? acc_old[c] : 0) + (gs ? gm[c] * psum[c] : psum[c])`, with `gm` traveling
   in the row tag (or read from the bank register by the tile generation bit). N unsigned 8 × signed
   17-bit multipliers (one DSP48E2 each, or LUT multipliers; 32 at N = 32), one extra register stage.
3. The command processor (SG6) expands GEMM_GS into per-tile array commands, splitting K at
   `min(N, G)`; tiles that start a group send the scale beat. A later optimization can keep the scale
   register across the tiles of one group and send it once per group.
Proposed C3 text: *"`gs` (W4A8 group scales): only with `int4=1`; the weight stream carries k+1 beats,
beat 0 = per-column uint8 multipliers `s[c]`; semantics `ACC[...] = (acc ? ACC[...] : 0) + s[c] ·
(A @ W)[c]` per column, exact INT32."*

## Consequences
* SG6 decoder: two new opcodes (0x12, 0x1E) from `nanoxl/isa.py`; no existing field moves.
* SG5: LUT RAMs are written by LUT_LOAD (256 × 16 bits each); LN/softmax read their parameter
  blocks as laid out in `docs/isa.md` §3.
* SG8: ATTN_TILE stays reserved; an online softmax needs a new ADR that changes the golden model.
