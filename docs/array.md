# Systolic array, tile controller and scratchpads (SG2 / M2)

RTL: `rtl/pe.sv`, `rtl/skew.sv` (skew + `delay_pipe`), `rtl/systolic_array.sv`,
`rtl/array_ctrl.sv`, `rtl/scratchpad.sv` (`scratchpad` + `spad_fifo`); ARRAY engine
sequencer (SG7a, §7): `rtl/array_seq.sv`, `rtl/array_seq_rd.sv`.
Tests: `tests/rtl/test_pe.py`, `test_array.py`, `test_scratchpad.py`, `test_array_seq.py`
(cocotb modules `*_cocotb.py`). Port protocol: contract **C3** in `docs/contracts.md`.
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
- **Group scales** (`cmd_gs`, ADR-004, SG7a): the tile's weight stream starts with one scale
  beat (byte c = uint8 `s[c]`) that goes into a group-scale FIFO (`spad_fifo`, `GM_DEPTH` =
  N+4 rows) instead of a bank, so it can arrive while the bank is still busy. Rows carry a `gs`
  bit in their tag; the RMW stage computes `(acc ? old : 0) + s[c]·psum[c]` with the FIFO head
  and pops it on the tile's last row (rows leave the array in tile order). The multiply is
  N unsigned-8 × signed-(log2 N + 12) bits (exact for INT4 tiles: |psum| ≤ N·2^10), in the
  same stage as the add, so latency is unchanged (2N+4) and a `gs` tile costs max(m, k+1).
  (If timing needs it, the product can be registered one stage earlier, ADR-004's "+1 stage".)
  The scale beat waits while the FIFO is full; a `gs` row that finds no scale row sets
  `overflow_seen` (sim). With `cmd_gs=0` the datapath is the SG2 one.

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
| `ctrl_gs_*` (N=4, 8, 16) | `cmd_gs` tiles mixed with plain INT8/INT4 GEMMs (groups 8..64, ragged last group, random uint8 scales, bubbles, backpressure); extremes (a=-128, w=-8, s=255/0/1); M=1 K=1 gs tiles back to back (scale FIFO occupancy, bypass); streaming max(m, k+1) |
| slow tier | `test_array_ctrl_kv260_size`: N=32, M_DEPTH=256, all controller tests with 60 random GEMMs (Verilator only; Icarus is impractically slow at 1024 PEs) |

Every controller test ends by asserting `hazard_seen == 0`, `overflow_seen == 0`, `idle == 1`
and that every expected row arrived in order with the right `last` flag.

## 5. Open items for integration
- `cmd_m`, `cmd_k`, `cmd_n` must be ≥ 1 (0 is not checked in RTL).
- The accumulator holds M_DEPTH rows of N INT32; the compiler (SG4) allocates `abase`
  ranges and can double-buffer by alternating two ranges.
- `cmd_ready` is combinational from `w_valid` (the last weight beat of the current load).
- INT4 group scales: superseded by ADR-004 (`cmd_gs` in array_ctrl, and the sequencer in §7).

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

## 7. ARRAY engine sequencer (`array_seq.sv`, SG7a / M5)

Turns ISA `GEMM` / `GEMM_GS` words (docs/isa.md) into array_ctrl tile commands, streams the
operands from the scratchpads and writes the INT32 results into the ACC buffer. It sits on
ctrl_top's `arr_*` engine port (docs/control.md §2).

```
 cmd_word ─► decode (cmd_fields) ─► tile generator ─┬─► array_ctrl cmd FIFO ───────────────────────────► array_ctrl
                                                    ├─► W tile FIFO ─► W row gen ─► array_seq_rd ─► [transposer] ─► w
                                                    ├─► A tile FIFO ─► A row gen ─► array_seq_rd ───────────────► a
                                                    ├─► ACC prefetch jobs ─► gate (earlier cmds done) ─► array_seq_rd (ACC)
                                                    └─► write-back jobs ─► out row (+ old row) ─► ACC write port ─► done
```

**Loop order** per command: row chunks of `M_DEPTH` rows (the array_ctrl accumulator, `abase` 0),
then output column tiles of N, then K-tiles of N rows (`GEMM_GS`: min(N, G) rows, so a tile
never straddles a scale group; every `GEMM_GS` tile carries its group's scale beat). The
K-chain of one column tile runs inside array_ctrl (`acc` = k-tile > 0, `emit` on the last), so
the ACC buffer sees each output row once; with `accumulate=1` the write-back adds the old ACC
row. Operand addressing follows the compiler (docs/isa.md §3), all incremental (no multiplier
except the one-time `rup(K·N/2, 16)` of the `GEMM_GS` scale offset):

| operand | segment of a tile | bytes |
|---|---|---|
| A row m | `a_addr + m·a_stride·16 + k0` | kt |
| W INT8 row r | `w_addr + (k0+r)·N + n0` | nt |
| W INT4 / GEMM_GS row r | `w_addr + (k0+r)·N/2 + n0/2` | nt/2 |
| GEMM_GS scale row | `w_addr + rup(K·N/2,16) + (k0/G)·N + n0` | nt |
| W `w_trans` (stored [N,K]) column c | `w_addr + (n0+c)·K + k0` | kt |
| ACC row m | `c_addr + m·c_stride·16 + 4·n0` | 4·nt (strobes) |

`w_buf=1` reads W from KVBUF (the attention GEMMs). `w_trans` tiles pass through a
double-buffered N×N byte transposer (collect nt columns, emit kt K-rows; 2·N²·8 bits of
registers, 16 kbit at N=32); other tiles bypass it once it is empty, so order is kept.
`gemv` is a timing hint and is ignored (results are identical; the latency is already the
2N+4 fill).

**Scratchpad ports.** Four ports with the DMA's line protocol (docs/control.md §3: buffer id,
17-bit line address, in-order responses with any latency, no response backpressure): A read
(ABUF, `APL` lines per beat), W read (WBUF/KVBUF, `WPL` lines), ACC read and ACC write (one
INT32 row = N/4 lines per beat, byte strobes). Defaults `APL = WPL = ceil(N/16)`. A beat is
`PL` consecutive lines starting at any line address, so the integrator banks the physical
buffers by line (ACC rows start on any line: `c_stride` is in lines). `array_seq_rd` turns
byte-granular segments into beats: one beat when the segment fits the PL-line window from its
start line, else two. Under the alignment rule A rows, ACC rows and weight rows of panels whose
row length is a multiple of 16 always take one beat; ragged panels (the LM head, N=65) straddle
and cost a second beat per row; `WPL = ceil(N/16)+1` would remove that. Each reader keeps at
most `RCRED` = 8 beats requested and not consumed, with an RCRED-entry response FIFO, so
responses never need backpressure; full throughput needs a memory latency below about RCRED-2.

**Ordering, done.** Commands are accepted while the tile generator is idle and fewer than
`OUTS` = 4 are outstanding (cmd_proc's MAXOUT). `done` pulses (registered, one per command, in
order, ≥ 2 cycles after accept) when the command's last ACC write is accepted by the port; the
integrator's ACC must make an accepted write visible to every read accepted at a later edge.
The old rows of an `accumulate=1` command are read only after every earlier command is done,
so a GEMM accumulating onto the previous GEMM's tile (K-split panels) is RAW-safe; within one
command every ACC tile row is read once, before it is written. Opcodes other than GEMM/GEMM_GS
(ATTN_TILE is reserved) set the sticky `err` (→ ctrl ERR 7) and still get their done. `clr`
(ctrl_top `eng_clr`) resets the engine like `rst_n`, and the readers drop the responses of
requests issued before it (counted in flight). `idle` = no command outstanding.

### Measured throughput (`seq_perf_streaming`, `seq_compiler_perf`)

Steady state, the same command twice back to back, no backpressure, memory latency 1:
occupancy (done to done) equals Σ over tiles of max(mc, kt) (+1 per `GEMM_GS` tile) exactly
for aligned operands, i.e. the engine adds no bubble between tiles, K-chains, column tiles, row
chunks or commands (asserted at N=4 and N=8 for GEMV INT8/INT4, GEMM_GS G=8/64, prefill, M >
M_DEPTH, w_trans from WBUF/KVBUF). Accept→done latency = occupancy + 2N+4 (array fill) + 8..23
(readers, write-back, done register).

Tiny model GEMM shapes at the KV260 size (N=32, Verilator, compiled for `array_n=32`), next to
`nanoxl.cyclemodel.cost` (occupancy; the model column is the SG7a-era model, see the note below):

| bundle | op | M | N | K | tiles | measured occ | accept→done | model occ (SG7a era, superseded) |
|---|---|---|---|---|---|---|---|---|
| w8a8 | GEMM c_attn | 16 / 1 | 256 | 128 | 32 | 1024 | 1115 / 1100 | 528 |
| w8a8 | GEMM attn_proj | 16 / 1 | 128 | 128 | 16 | 512 | 603 / 588 | 272 |
| w8a8 | GEMM mlp proj | 16 / 1 | 64 | 512 | 32 | 1024 | 1115 / 1100 | 528 |
| both | GEMM LM head (ragged 65-byte rows) | 16 / 1 | 65 | 128 | 12 | 624 | 715 / 700 | 208 |
| both | GEMM scores (w_trans, KVBUF) | 16 / 1 | 16 / 17 | 32 | 1 | 32 | 139 / 125 | 32 |
| both | GEMM P·V (KVBUF) | 16 / 1 | 32 | 16 / 17 | 1 | 16 / 17 | 107 / 93 | 32 |
| w4a8 | GEMM_GS c_attn | 16 / 1 | 384 | 128 | 48 | 1584 | 1675 / 1660 | 776 / 416 |
| w4a8 | GEMM_GS fc | 16 / 1 | 480 | 128 | 60 | 1980 | 2071 / 2056 | 968 / 518 |
| w4a8 | GEMM_GS attn_proj | 16 / 1 | 128 | 128 | 16 | 528 | 619 / 604 | 264 / 144 |
| w4a8 | GEMM_GS mlp proj panels | 16 / 1 | 96, 32 | 512 | 48, 16 | 1584, 528 | 1675, 619 / 1660, 604 | 776, 264 / 416, 144 |

The tiny model has M ≤ 16 < N = 32, so every tile is weight-load bound (kt = 32 rows): the RTL
loads **one weight row per cycle** (array_ctrl, C3). The "model occ" column was measured against the
M1-era cycle model (`wload_rows_per_cycle = 2`, INT4 two rows per beat). **Superseded:** the merged
SG4c model already charges one row per cycle without the INT4 doubling, and since AUDIT-1 (A1-05)
it also charges the scale beat on every `GEMM_GS` tile, the second beat of ragged weight rows (LM
head N=65: 624) and the accept→done latency (2N+4 + 7 + M, + N/2 through the w_trans transposer).
`nanoxl.cyclemodel.cost` now equals the "measured occ" column for every row of the table, and
accept→done to within 1 cycle (`tests/audit/test_audit1.py::test_a1_05_*`, `test_a1_06_*`). A
2-row weight port (two bank rows per cycle, a 2N-byte W port and a WBUF read port of that width)
remains a priced what-if (`RtlTiming.array_rows_per_cycle`, bench/results/fixes.md); `WPL =
ceil(N/16)+1` would remove the ragged-row beat (`RtlTiming.array_w_lines`).

### Verification (`tests/rtl/test_array_seq.py`, cocotb `array_seq_cocotb.py`)

Reference: `nanoxl.funcsim` on the same buffer images (all bytes random); every buffer is
compared byte for byte after the run. Commands are `isa.encode(..., aligned=True)` words.

| test | what |
|---|---|
| `seq_random` (N=4, 8; 40 cmds at N=32) | 30 random GEMM (INT8, INT4, w_trans, KVBUF) / GEMM_GS (G = 8..64) back to back: M up to 2·M_DEPTH+3, ragged N and K, random strides and bases, 35 % K-chains accumulating into the previous command's tile; random ready on all four ports, response latency 1..6, cmd bubbles |
| `seq_random_nobp` | same generator, no backpressure, latency 1 |
| `seq_gs_groups` | GEMM_GS with G = 8, 16, 32, 64, 128, K over several groups plus a ragged last group, a=-128, w=-8, gm=255, accumulate on top |
| `seq_illegal_opcode` | ATTN_TILE between two GEMMs: `err`, three dones in order, GEMMs exact |
| `seq_reset_mid_command` | `rst_n` and twice `clr` with rows in the array and responses in flight (the memory keeps answering after `clr`); no stale done; fresh commands exact |
| `seq_compiler` (N=8) | the cheap compiler GEMM shapes of tiny W8A8 + W4A8 (attention w_trans/KVBUF, P·V, LM head N=65, smallest GEMM_GS), operands = funcsim state right before the command |
| `seq_perf_streaming` (N=4, 8) | asserts occupancy = ideal for 9 shapes (above), logs the ragged-row cost |
| slow: `test_array_seq_compiler_all` | N=16, all 236 GEMM / GEMM_GS commands the compiler emits for both bundles (prefill of the reference prompt + one decode step) |
| slow: `test_array_seq_kv260_size` | N=32: `seq_random` (40) and `seq_compiler_perf` (every distinct shape exact + the table above) |

The scratchpad model also flags a read and a write of the same ACC line at the same edge (never
seen). Mutation checks (each killed): scale multiply bypassed in the RMW (`ctrl_gs_*`,
`seq_random`, `seq_gs_groups`); scale FIFO popped on every row instead of every tile
(`ctrl_gs_*`); K-chain `acc` bit forced to 0 (every seq test); ACC prefetch not gated on the
earlier commands' done (`seq_random`, `seq_random_nobp`, `seq_reset_mid_command`).
