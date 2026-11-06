# Systolic array, tile controller and scratchpads (SG2 / M2)

RTL: `rtl/pe.sv`, `rtl/skew.sv` (skew + `delay_pipe`), `rtl/systolic_array.sv`,
`rtl/array_ctrl.sv`, `rtl/scratchpad.sv` (`scratchpad` + `spad_fifo`).
Tests: `tests/rtl/test_pe.py`, `test_array.py`, `test_scratchpad.py` (cocotb modules
`*_cocotb.py`). Port protocol: contract **C3** in `docs/contracts.md`.
Synthesis check: `fpga/synth_array.ys`.

All blocks use `clk` and a synchronous active-low `rst_n`. Defaults: N = 8 in the RTL, 32
intended for the KV260 (ADR-003 decides). INT8 activations and weights, INT32 accumulation,
exact (contract C1; K ≤ 4096 cannot overflow).

## 1. Structure

```
 a row (N x int8) ──► [mask k] ─► reg ─► skew (row r delayed r) ─► PE grid ─► deskew (col c delayed N-1-c)
                                                                    ▲              │
 w row (N x int8 / N x int4) ─► [mask n] ─► reg ─► INT4 sext ─► write skew        ▼
                                                   (col c delayed c)        RMW: acc[m] (+)= row
                                                                                   │      (scratchpad, 1R1W)
                                                                                   ▼
                                                                credit-checked output FIFO ─► out (valid/ready)
```

### PE (`pe.sv`)
`psum_out <= psum_in + a_in * w[a_sel_in]`, `a_out <= a_in`. Two weight registers (banks).
The bank is chosen **per activation** by `a_sel`, which travels with the activation, so there
is no global "swap" pulse: the bank switch moves through the array as a diagonal wavefront
together with the first row of the new tile. The multiply is a single `always_comb` (see §6).

### Array core (`systolic_array.sv`)
- PE(r,c) holds W[r,c] (r = K index, c = output column). A-row element r enters row r after r
  cycles of skew; psums flow down; column c is deskewed by N-1-c, so a full INT32 row leaves
  in one cycle.
- PE(r,c) multiplies the row presented in cycle T during cycle T+r+c; the output row is
  presented in cycle **T + LAT, LAT = 2N-1** (measured in `array_cocotb.py::check`).
- **Skewed weight writes.** A row write issued in cycle W reaches PE(r,c) at the end of cycle
  W+c: the write wavefront travels along the row at the same speed as the activations. This
  is what makes reloading cheap:
  - *reuse rule*: rewriting row r of bank b is legal from W ≥ T_last(b) + r, i.e. right
    behind the last activation that reads that bank (it can never overtake it);
  - *use rule*: a new-tile activation may enter row r when T + r ≥ W_r + 1 (it can chase the
    write wavefront one cycle behind).
  With row-broadcast (unskewed) writes the reuse rule would instead be W ≥ T_last + r + N-1,
  which costs ~N extra cycles per bank reuse; the first implementation did this and measured
  10.8 cycles/tile for M=K=N=8 (ideal 8). Cost of the skew: N(N-1)/2 × (8+log2N+3) FFs
  (~7.4 k FFs at N=32), about half the output deskew triangle.
- **INT4 (W4A8)**: `w_wr_int4=1` reads N packed nibbles (C1b order: element c in bits
  [4c+3:4c]) from the low N*4 bits and sign-extends each to 8 bits. Products and
  accumulation are then identical to INT8 (`gemm_i8(a, unpack_i4(p))`). A weight beat is one
  K-row either way; in INT4 mode the upstream weight buffer/DMA only moves N/2 bytes per
  row, which halves the DDR traffic that bounds decode.
- **Simulation checker** (`hazard_seen`, compiled out under `SYNTHESIS`): each tile carries a
  generation bit (`in_gen`, `w_wr_gen`); a shadow `{written, gen}` per PE per bank is
  compared against every valid, non-zero activation. Any multiply with a weight of another
  tile, or never written, sets the sticky flag. Zero activations (ragged-K padding) are
  exempt since their product cannot depend on the weight. Two tests prove the checker fires
  (overwrite under in-flight rows; activation one cycle ahead of the write) and one test
  runs both rules at their exact limits (`array_tight_reuse_and_chase`; tightening it by one
  cycle makes it fail).

### Tile controller (`array_ctrl.sv`)
One command = one weight tile (K ≤ N rows, n ≤ N columns) applied to M activation rows.
- **Loader**: accepts the next command as soon as the previous load finished; writes K rows
  (one per `w` beat) into bank `tile_count[0]`, but only while that bank is not holding an
  un-issued or issuing tile (`w_ready = ld_active && !loaded[bank]`). Columns ≥ n are masked
  to zero on the write (bytes, or nibbles in INT4 mode).
- **Issuer**: streams the M `a` beats of the loaded tile; bytes ≥ k are masked to zero, so
  stale weight rows ≥ k never contribute. Each row carries a tag {last, emit, acc, addr}.
- **Accumulate**: rows leaving the array read `acc[abase+m]` (registered-read scratchpad),
  add it when `acc=1` (else overwrite), and write back. A one-entry bypass covers a read of
  the row being written in the same cycle (back-to-back M=1, K=1 tiles hit it; tested).
- **Output**: rows of `emit` commands go to an `spad_fifo`. The array never stalls; instead
  the issuer only issues an emit row when a FIFO slot is reserved (credit counter ≤
  OF_DEPTH, decremented on output pop). So `out_ready` can drop at any time with no loss
  and no global clock-enable fan-out. OF_DEPTH ≥ 2N+5 is needed for full throughput under
  `out_ready=1` (default 2N+8).
- **K accumulation** across tiles: commands with the same `abase`/M, first with `acc=0`, the
  rest `acc=1`, the last with `emit=1`. ragged K and n need no padding upstream.

### Scratchpad (`scratchpad.sv`)
`scratchpad`: single-clock 1R1W, `logic mem[DEPTH]`, synchronous write, registered read,
`RD_LAT` = 1 (BRAM) or 2 (adds the BRAM `DOB_REG` / URAM `OREG` output register). Read of
the address written in the same cycle is unspecified (callers bypass). Used for the INT32
accumulator buffer and the output FIFO here; weight ping-pong and activation buffers
(SG6/SG7) instantiate the same module. `spad_fifo`: FIFO on a scratchpad with a 2-entry
fall-through output queue, 1 push + 1 pop per cycle, push→out latency 3.

## 2. Latency and throughput (measured, `ctrl_perf_streaming`)

Fill latency, first `a` beat accepted → first `out` beat accepted: **LAT + 5 = 2N + 4**
cycles (input register 1, array 2N-1, RMW stage 1, FIFO write/read/queue 3). That is 12 at
N=4, 20 at N=8, 68 at N=32. A GEMV (decode) result therefore appears ~2N+4 cycles after its
activation row; this is a pure latency, not a throughput cost, because tiles overlap.

Steady-state cycles per tile, back-to-back tiles, no bubbles, `out_ready=1` (16 tiles each,
Icarus; the test asserts the measured period equals the ideal exactly):

| case | M | K | N=4 | N=8 | ideal max(M,K) |
|---|---|---|---|---|---|
| GEMV / decode | 1 | N | 4.00 | 8.00 | N (weight-load bound) |
| GEMV, K=N-1 | 1 | N-1 | 3.00 | 7.00 | K |
| GEMV, K=N/2 | 1 | N/2 | 2.00 | 4.00 | K |
| GEMV, K=1 | 1 | 1 | 1.00 | 1.00 | 1 |
| square | N | N | 4.00 | 8.00 | N |
| prefill | 2N | N | 8.00 | 16.00 | M (issue bound) |
| prefill, K=N/2 | 2N | N/2 | 8.00 | 16.00 | M |
| small | 2 | 2 | 2.00 | 2.00 | 2 |
| K > M | 3 | N | 4.00 | 8.00 | K |

So tiles stream with zero bubbles for every shape. Why the bound is max(M, K): one weight row
and one activation row are accepted per cycle, and with skewed writes bank b of tile t+2 can
be written from the cycle after tile t's last row, which gives 2P ≥ M + K ⇒ P ≥ max(M,K)
always holds. Decode (M=1) is weight-bandwidth bound at one K-row per cycle: N=32 at 200 MHz
needs 6.4 GB/s of INT8 weights from the weight buffer (3.2 GB/s in INT4), which is why the
on-chip weight ping-pong buffer, not the array, sets decode speed. A design-plan §6.1
"GEMV bypass of skew/deskew" would only shave the 2N fill latency of a single token; it is
not needed for throughput and is left out.

## 3. Synthesis (Yosys 0.33 generic, N = 8, `fpga/synth_array.ys`)

| design | level | result |
|---|---|---|
| `systolic_array` N=8 | word | 64 `$macc` (one 8×8 MAC per PE), 56 `$alu`, 397 FF cells (`$sdff`/`$sdffe`), no latches, no memories |
| `systolic_array` N=8 | gates (`synth -flatten`) | 46 818 cells: 4 751 FF bits (1 024 `$_SDFFE_` = weight banks, 3 727 `$_SDFF_`), 10 540 XOR, 2 486 XNOR, 14 407 ANDNOT, 5 263 OR, … |
| `array_ctrl` N=8, M_DEPTH=64 | word | 67 `$macc`, 5 `$mem_v2`: accumulator `u_acc.mem` 64×256 b and FIFO `u_ofifo.u_mem.mem` 24×257 b (BRAM candidates) plus three 2-entry descriptor arrays (become FFs), 435 FF cells |

The gate count is dominated by the 64 multipliers built from generic gates; on the KV260
each `$macc` maps to one DSP48E2 (or half of one with packing, §6). Synthesis-only
constructs: none; the `hazard_seen`/`overflow` checkers sit under `ifndef SYNTHESIS`.

## 4. Verification summary

| test | what |
|---|---|
| `pe_exhaustive` | all 65 536 (a, w) pairs, random bank select, random psum incl. ±2^31 neighbourhood, same-cycle writes don't affect the current product |
| `pe_bank_select`, `pe_reset_clears` | bank independence; reset clears regs and banks |
| `array_*` (N=4, 8) | random tiles with ragged K and input bubbles, exact latency 2N-1; -128×-128 over full K; INT4 with every code; overlap load; exact-limit reuse/chase; checker sanity ×2; reset mid-stream |
| `ctrl_*` (N=4, 8, 16) | 40 random GEMMs (M up to 64, K up to 3N ⇒ multi-tile accumulation, ragged n, 30 % INT4, random accumulator base with wrap, garbage in masked lanes, random bubbles on cmd/w/a, 50 % random backpressure); INT4 with full/ragged/odd n; -128×-128 over K = 4N; long `out_ready=0` stall; M=1/K=1 accumulate chains (bypass); reset mid-stream; streaming performance |
| `scratchpad_*` | random R/W at RD_LAT 1/2; FIFO random push/pop, count, throughput |
| slow tier | `test_array_ctrl_kv260_size`: N=32, M_DEPTH=256, all controller tests with 60 random GEMMs (Verilator only; Icarus is impractically slow at 1024 PEs) |

Every controller test ends by asserting `hazard_seen == 0`, `overflow_seen == 0`, `idle == 1`
and that every expected row arrived in order with the right `last` flag.

## 5. Open items for integration
- `cmd_m`, `cmd_k`, `cmd_n` must be ≥ 1 (0 is not checked in RTL).
- The accumulator holds M_DEPTH rows of N INT32; the compiler (SG4) allocates `abase`
  ranges and can double-buffer by alternating two ranges.
- `cmd_ready` is combinational from `w_valid` (the last weight beat of the current load).
- INT4 group scales are applied at requant (C1b), never in the array; for groups smaller
  than the K extent of an accumulation chain the compiler must emit a partial result per
  group (`emit=1` with a new `acc=0` chain), as anticipated in design-plan §6.1.

## 6. Future option: DSP48E2 2-MAC packing (AMD WP486)
Two weights that share one activation can share one DSP48E2: compute
`a × (w0·2^18 + w1)` in the 27×18 multiplier and recover `a·w0` and `a·w1` from the two
fields of the product (with the sign-correction term from WP486). In this array the two
PEs (r, 2j) and (r, 2j+1) see the same activation one cycle apart. The change is local:
a `pe2` cell replaces two horizontally adjacent PEs, registers the activation once per
pair (activations then advance two columns per register, so the input skew per row is
unchanged and the deskew triangle halves in width), holds two bank pairs of weights, and
splits the packed product into the two psum columns. Nothing outside `systolic_array.sv`'s
generate loop changes, because `pe_mul` is the only arithmetic in `pe.sv` and the
port-level timing (LAT, write wavefront) can be preserved by an extra column delay. Kept as
a behavioural multiply for now; enable once the Vivado DSP budget (ADR-003) needs it.
