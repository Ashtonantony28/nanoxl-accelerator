# Vector unit (SG3 / M3a)

The vector unit is an L-lane SIMD datapath that sits behind the systolic array. It does
requantization (contract C2), elementwise integer ops and horizontal reductions. The
special-function units (exp, recip, rsqrt, GELU) attach to a reserved hook and are SG5's
work, which waits for ADR-002.

| File | Module | What it is |
|---|---|---|
| `rtl/srdhm.sv` | `srdhm` | C2a SaturatingRoundingDoublingHighMul, 3 stages |
| `rtl/requant.sv` | `requant` | C2 requantizer, L lanes, 6 stages |
| `rtl/sat_counter.sv` | `sat_counter` | sticky saturating event counter |
| `rtl/vec_alu.sv` | `vec_alu` | elementwise ALU, L lanes, 3 stages |
| `rtl/vec_reduce.sv` | `vec_reduce` | sum, sum of squares, max and min across lanes and beats |
| `rtl/vec_unit.sv` | `vec_unit` | command sequencer, stream plumbing, SFU hook |
| `nanoxl/vec_ref.py` | | executable reference for everything below (requant comes from `nanoxl.numerics`) |

Default parameters: `L` = 4 in the RTL. The KV260 configuration is `L = 64` (ADR-001 / design plan §5); ADR-003 sets the final value.

## 1. Conventions

- **Lanes are int32.** Every stream carries `L × 32` bits; lane `i` is bits `[32i+31:32i]`.
  int8 and int16 operands arrive sign-extended, and results narrower than 32 bits are returned sign-extended. Packing to bytes is the buffer's job, not the vector unit's.
- **Format codes** (`fmt`, 2 bits): `0` = int8, `1` = int16, `2` and `3` = int32.
- **Rows and beats.** A command works on a row of `len` elements, sent as `ceil(len/L)` beats. Element `e` is lane `e mod L` of beat `e div L`.
- **Handshake.** Every port is valid/ready. A beat moves on a rising edge when valid and ready are both high. A source holds valid and data until the beat is accepted.
- **Stall style.** Each pipeline stalls as a whole: `in_ready = !out_valid || out_ready`. A pipeline holding `D` stages runs at full throughput while the consumer is ready. `in_ready` is combinational from `out_ready`. If that path limits timing at integration, add a skid buffer at the vec_unit boundary.
- **RoundingDivideByPOT** (`RDivPOT(x, e)`, C2b) is `(x >> e) + (rem > thr)`, where `rem = x & (2^e-1)` and `thr = ((2^e-1) >> 1) + (x < 0)`. Exact halves round away from zero. The vector unit uses it on 32-bit values (in requant) and on 64-bit values (in the ALU), with `e` in [0, 31] in both cases.

## 2. Requant (`requant`, contract C2)

```
x   = wrap32(acc + bias)
y   = RDivPOT(SRDHM(x, mult), shift)
out = clamp(y, int8)  if out16 = 0
      clamp(y, int16) if out16 = 1
```

The result is bit-identical to `nanoxl.numerics.requant`. Each lane has its own `acc`, `bias`, `mult` and `shift` (one lane per output channel). `out16` applies to the whole beat.
SRDHM is implemented for the whole int32 domain, including `INT32_MIN × INT32_MIN → INT32_MAX`. Contract-legal multipliers (`mult` ≥ 2^30) can never reach the overflow case, but the tests still exercise it with out-of-contract multipliers.

**Pipeline: 6 stages, 1 beat per cycle.**

| Stage | Work |
|---|---|
| S1 | 32-bit wrapping `acc + bias` |
| S2 | 64-bit signed product `x · mult` (DSP) |
| S3 | product register (DSP P register; the 32×32 product cascades over 4 DSP48E2) |
| S4 | add the nudge (`2^30`, or `1 − 2^30` when the product is negative); C truncating `/2^31`, computed as add `2^31−1` when negative, then `>>> 31`; overflow select |
| S5 | `RDivPOT(q, shift)` (32-bit barrel shifter) |
| S6 | clamp to int8 or int16; saturation flag |

A beat accepted at edge `k` appears on `out_*` after edge `k+5`, i.e. 6 cycles later. The test `pipeline_depth` checks this.

**Saturation counting.** `out_sat[i] = saturated(i) & out_keep[i]`. The `sat_counter` adds `popcount(out_sat)` on every output transfer. It is 32 bits wide, saturates instead of wrapping, and has a sticky flag; `sat_clr` clears both. A clear in the same cycle as events keeps those events.

## 3. Elementwise ALU (`vec_alu`)

Each op computes an **exact** signed 64-bit intermediate `w` from the int32 lane values `a` and `b`. Nothing in the table can overflow 64 bits: `|a·b| ≤ 2^62`, and `|a·2^31| ≤ 2^62`. The unit then saturates `w` to the beat's `fmt` and returns the result sign-extended to 32 bits. The lane's saturation flag is set when `w` lies outside the `fmt` range.

| op | name | `w` | operands |
|---|---|---|---|
| 0 | PASS | `a` (with `fmt` saturation, this is the *saturate/narrow* op) | a |
| 1 | ADD | `a + b` (33-bit exact) | a, b |
| 2 | SUB | `a − b` (33-bit exact) | a, b |
| 3 | MUL | `RDivPOT(a · b, shamt)` (64-bit product, rounded right shift) | a, b |
| 4 | MAX | `max(a, b)` | a, b |
| 5 | MIN | `min(a, b)` | a, b |
| 6 | SRA | `a >> shamt` (arithmetic, floor) | a |
| 7 | SRAR | `RDivPOT(a, shamt)` (rounding arithmetic shift) | a |
| 8 | SHL | `a · 2^shamt` (exact; saturates rather than wrapping) | a |
| 9 | CLAMP | `min(max(a, lo), hi)`; when `lo > hi` the result is `hi` | a |
| 10–15 | reserved | executed as PASS | a |

`shamt` is 0..31, and `lo` and `hi` are int32. All of them, together with `op` and `fmt`, are **per-beat** inputs that travel down the pipeline with the data. Beats from different commands can therefore occupy the pipeline back to back.

**Pipeline: 3 stages.**

| Stage | Work |
|---|---|
| S1 | add, subtract, 32×32 multiply or compare |
| S2 | shift, rounding or clamp |
| S3 | `fmt` saturation and saturation flag |

The ALU has its own saturation counter, which counts kept lanes on transfers.

Widening rules, summarised: inputs widen to 64 bits, the op is computed exactly, and the only narrowing is the final saturate to `fmt`. Nothing wraps, with one exception: the bias add in requant wraps, because C2 requires it.

## 4. Reductions (`vec_reduce`)

`vec_reduce` computes four results for each row (a run of beats ending in `last`) in a single pass. It only looks at lanes whose `keep` bit is set.

| Output | Width | Value |
|---|---|---|
| `sum` | `ACC_W` = 48, signed | Σ x |
| `sumsq` | 48, signed | Σ x², with each x taken as **int16** (its low 16 bits) |
| `max`, `min` | 32 | max x, min x. A row with no kept lane gives INT32_MIN for max and INT32_MAX for min |
| `sq_err` | 1 | some kept x was outside int16, so `sumsq` is not Σx² for that row |

**Width argument.** An int16 square is at most 2^30. A row of 2^17 int16 values therefore sums to at most 2^47 in `sumsq`, and a row of 2^16 int32 values to at most 2^47 in `sum`, so 48 signed bits are exact for those row lengths. A GPT-2 LayerNorm row (d = 384, or 768 for `shakespeare`-sized models) needs at most 39 or 40 bits. The tests include 384 × (−32768)², 2048 × (−32768)² = 2^41, and 384 × INT32_MIN. Longer rows wrap modulo 2^48; `vec_ref.wrap_acc` models this.

LayerNorm uses `sum` and `sumsq` from the same pass. Softmax uses `max`. Squaring only int16 values keeps the unit at one 16×16 multiplier (one DSP) per lane. The design plan keeps the residual stream in int16, and any wider input raises `sq_err`.

**Pipeline: 3 stages.**

| Stage | Work |
|---|---|
| S1 | keep mask, 16×16 squares, int16 range check |
| S2 | balanced adder and compare trees across L lanes (heap layout, L padded to a power of two) |
| S3 | row accumulator and result register |

The result is valid 3 cycles after the beat carrying `last`. A pending result that has not been taken stalls the input.

## 5. Sequencer (`vec_unit`)

### 5.1 Command

The command is one valid/ready transfer with these fields:

| Field | Bits | Meaning |
|---|---|---|
| `cmd_op` | 5 | opcode (§5.2) |
| `cmd_len` | RW (16) | row length in elements; `0` is a no-op that produces nothing, including no REDUCE result |
| `cmd_fmt` | 2 | output format. For REQUANT: `0` = int8, otherwise int16 |
| `cmd_shamt` | 5 | ALU shift amount. For REQUANT with `p_imm`, the broadcast shift |
| `cmd_lo`, `cmd_hi` | 32 | CLAMP bounds |
| `cmd_b_imm`, `cmd_imm_b` | 1, 32 | when set, B is `imm_b` broadcast to every lane and the B stream is not read. For REQUANT, B is the bias |
| `cmd_p_imm`, `cmd_mult` | 1, 32 | REQUANT only: when set, every lane uses `mult = cmd_mult` and `shift = cmd_shamt` (per-tensor scale) and the P stream is not read |

### 5.2 Opcode space

| `cmd_op` | Unit | Streams read | Output |
|---|---|---|---|
| `0x00–0x0F` | vec_alu, op = `cmd_op[3:0]` | A; B for ADD/SUB/MUL/MAX/MIN unless `b_imm` | D |
| `0x10` | REQUANT | A = acc; B = bias unless `b_imm`; P = mult/shift unless `p_imm` | D (int8/int16 sign-extended) |
| `0x11` | REDUCE | A | R (one result per row) |
| `0x12–0x17` | reserved: run as ALU PASS, set sticky `illegal_op` | A | D |
| `0x18–0x1F` | **SFU hook**; sub-op = `cmd_op[2:0]` | A | D |

When the unit is built with `HAS_SFU = 0`, opcodes `0x18–0x1F` run as ALU PASS and set `illegal_op`, so a missing SFU cannot hang the unit.
Suggested sub-op allocation for SG5, to be fixed by ADR-002: 0 exp, 1 recip, 2 rsqrt, 3 GELU LUT, 4–7 free.

### 5.3 Streams

| Port | Payload |
|---|---|
| `a_*` | `L×32` |
| `b_*` | `L×32` |
| `p_*` | `L×32` mult plus `L×5` shift |
| `d_*` | output: `L×32` data, `L` keep bits, last |
| `r_*` | REDUCE result: sum, sumsq, max, min, sq_err |

A command reads exactly `ceil(len/L)` beats from each stream it uses, and writes the same number of beats to D (or one result to R). In the final beat, the lanes at positions ≥ `len`:

- are zeroed before they reach any unit, so junk on those lanes cannot leak through;
- are never counted as saturation events;
- have `d_keep = 0` and read as 0 on D.

`d_last` marks a row's final beat.

### 5.4 Ordering and throughput

The input side runs one command at a time. The command-accept cycle costs one idle input cycle per command. After that, beats issue at one per cycle.

The output side keeps a queue of up to `RECQ = 4` elementwise commands still in flight. A row can start before the previous row has drained, and D beats always leave in command order. Each unit is in order, and the output mux only takes beats from the unit at the head of the queue. Because the ALU pipeline (3 stages) is shorter than requant (6 stages), an ALU command that directly follows a REQUANT command waits up to 3 cycles for the requant beats ahead of it. The `back_to_back_throughput` test asserts the cost model exactly:

`cycles ≤ beats + commands + 3 × (REQUANT→ALU transitions) + fill`

If that bubble matters for the schedule, the fix is to give the ALU path 3 more stages to match requant.

### 5.5 Status

| Signal | Meaning |
|---|---|
| `rq_sat_count`, `alu_sat_count` | 32-bit saturating counters |
| `sat_sticky` | OR of the two sticky flags |
| `illegal_op` | sticky |
| `busy` | the input side is issuing, an elementwise row is draining, or a REDUCE result is pending |

`stat_clr` clears the counters, the sticky flags and `illegal_op`.

### 5.6 SFU hook (for SG5)

`vec_unit` drives:
```
sfu_in_valid, sfu_in_op[2:0], sfu_in_fmt[1:0], sfu_in_shamt[4:0], sfu_in_imm[31:0],
sfu_in_data[L*32], sfu_in_keep[L], sfu_in_last      ← sfu_in_ready
```
The SFU drives:
```
sfu_out_valid, sfu_out_data[L*32], sfu_out_keep[L], sfu_out_last ← sfu_out_ready
```

The SFU must meet four requirements:

1. Emit exactly one output beat per input beat.
2. Emit beats in order.
3. Return `keep` and `last` unchanged.
4. Never make `sfu_in_ready` or `sfu_out_valid` depend combinationally on the other side's valid or ready in a way that creates a loop. Registered outputs are the safe choice.

Its latency is free, and it may stall. `sfu_in_op`, `sfu_in_fmt`, `sfu_in_shamt` and `sfu_in_imm` are constant for the length of a row; `imm` is `cmd_imm_b`, for example a softmax or LayerNorm scale. Lanes with `keep = 0` arrive as 0, and `vec_unit` zeroes them on D whatever the SFU returns.

The cocotb bench drives this port with a behavioural SFU that has random latency and backpressure. Its function `3x + 1000·op + imm + 7·shamt + 100000·fmt` lets the bench check every forwarded field.
To attach the real SFUs, SG5 instantiates its SFU block next to `vec_unit` with `HAS_SFU = 1` and connects the two port groups. Nothing else in `vec_unit` changes. If an SFU needs a second operand stream, extend the B-stream rule in `need_b`.

## 6. Verification

The verify command is `SIM=icarus python3 -m pytest -q tests/rtl -k "requant or vec"`. It also passes with `SIM=verilator`.

| Test | What it checks |
|---|---|
| `test_vec_ref.py` (no simulator) | The RTL's exact arithmetic (64-bit product, nudge, truncation by bias-then-shift, 32-bit RDivPOT) against `numerics.requant`: 2M random samples, the full 9×9×4×32×2 extremes cross product, and a bit-level Python-int transcription. `vec_ref.alu` against a pure big-int model for all 16 opcodes. The reduce width bound. |
| `test_requant.py` (L = 1, 4, 16) | 20k random lane-results per L, under three handshake regimes: full rate (asserts 1 beat per cycle), heavy backpressure and sparse input. Directed corners: INT32_MIN and MAX, mult = 2^30 and 2^31−1, shift = 0, 1, 30 and 31, bias wraparound both ways, exact halves in SRDHM (mult = 2^30, odd x) and in RDivPOT (both signs, every e), out-of-contract mults including INT32_MIN². Random keep and last. Exact saturation-counter value, sticky flag, `sat_clr`, pipeline depth = 6. |
| `test_vec_alu.py` (L = 1, 4, 16) | Every op × fmt × shamt on 15 extreme values (all pairs). Rounding halves for MUL and SRAR. 20k random lane-results per op at L = 16 with ops shuffled back to back. Saturation counter, depth = 3. |
| `test_vec_reduce.py` (L = 1, 3, 4, 16) | Random rows of 1..400 elements, including ragged final beats and randomly masked interior lanes with junk under the mask. Width-limit rows. Single-beat rows back to back. Three backpressure regimes. |
| `test_vec_unit.py` (L = 1, 4, 16 with an SFU stub; L = 4 without) | 200–400 random commands covering every unit, reserved opcodes, `b_imm` and `p_imm`, `len` = 0, 1..5L+3 and 384, with random valid and ready on all six streams and the SFU. Every D and R beat, both saturation counters, `illegal_op` and `busy` are checked. Back-to-back throughput bound. |
| `test_vec_sat_counter.py` | W = 6 so that saturation, clear and the sticky flag are all exercised. |

Mutation sanity results are recorded in the SG3 report.

## 7. Synthesis (Yosys 0.33 generic `synth`, L = 4)

`tests/rtl/test_vec_synth.py` runs this synthesis. It asserts that the design synthesizes with no latches and prints the counts. The requant and vec_alu runs are in the fast tier; vec_reduce and vec_unit are marked `slow`. To rerun: `python3 -m pytest -q -s tests/rtl/test_vec_synth.py`.

### 7.1 Cell counts

Cell counts are generic gates after `synth -flatten`. Yosys generic synthesis maps the multipliers to gates, so these numbers measure synthesizability, not FPGA area.

| Top (L = 4) | Cells | Flip-flops |
|---|---|---|
| `requant` | 36,403 | 1,254 |
| `vec_alu` | 45,953 | 772 |
| `vec_reduce` | 10,758 | 844 |
| `vec_unit` (all of the above + sequencer) | 94,532 | 3,058 |

In both `requant` and `vec_alu`, most of the cells are the four 32×32 multipliers. In `vec_alu` the 64-bit rounding barrel shifter adds to that. Vivado maps the multipliers to DSP48E2 slices (see §8).

## 8. Integration notes and known limits

- **DSP budget (decide in ADR-003).** At L = 64, requant needs one 32×32 multiplier per lane, which is about 4 DSP48E2s, so about 256 DSPs. The ALU MUL is also 32×32 per lane, another 256. The reduce squares are 16×16, one DSP each, so 64. That totals about 576 of the KV260's 1,248 DSPs before the array.
  - Options: restrict ALU MUL to 16×16 operands, which the design plan's int16 residual and LayerNorm scale probably allow, or share the multiplier between requant and the ALU, since the two never run in the same cycle for the same command.
- **Reduce tree timing.** The tree adds L lanes in a single cycle: at L = 64 that is 6 levels of 48-bit adders. At 250 MHz it may need one more pipeline register. The tree is written as a heap, so a register can be inserted at any level.
- **Combinational `in_ready` path.** `in_ready` depends combinationally on `out_ready` (§1).

## 9. VEC engine (`vec_seq`, SG7b)

`rtl/vec_seq.sv` executes the VEC-queue ISA commands (docs/isa.md §4) on `sfu_vec`
(vec_unit + SFU). Helpers: `vec_seq_walk.sv` (line-address walker), `vec_seq_rd.sv`
(source stream: port beats → L-lane beats), `vec_seq_wr.sv` (D beats → port beats with byte
strobes), `pbuf_loader.sv` (parameter loads and slot residency, docs/sfu.md §10). Parameter `PL`
(SG10b): lines per port beat on the A, B, P and D ports; `PL = 1` is the original 16-byte line port
(cycle-exact with the SG7b RTL), `nanoxl_top` builds `PL = VP = 4` (64 bytes).
Bench: `tests/rtl/test_vec_seq.py` (`vec_seq_cocotb.py`, harness `vec_seq_common_cocotb.py`).

### 9.1 Ports

| group | signals | notes |
|---|---|---|
| engine | `cmd_valid/cmd_ready/cmd_word[127:0]`, `done`, `err`, `err_code[3:0]`, `busy` | docs/control.md §2. 2-entry command queue; one `done` per command, in order, ≥ 1 cycle after accept, after the last write of the command was accepted |
| read A, B | `r{a,b}_arvalid/arready/arbuf[2:0]/araddr[16:0]`, `r{a,b}_rvalid/rdata[128PL-1:0]` | DMA `sp_*` read shape widened to PL lines: `araddr` = first line of the beat (any line), line j of the response in bits [128j +: 128]; in-order responses, any latency, no backpressure, ≤ 16 beats in flight per port (credit) |
| read P | `rp_arvalid/arready/araddr`, `rp_rvalid/rdata[128PL-1:0]` | always PBUF |
| write D | `wd_valid/wd_ready/wd_buf/wd_addr/wd_data[128PL-1:0]/wd_strb[16PL-1:0]` | PL lines from `wd_addr`, byte strobes per line (a line past the row end has strobe 0, and may lie past the buffer end); a write is complete (visible to every later read) when accepted |
| snoop | `pb_snoop_valid`, `pb_snoop_line[16:0]` | every PBUF line written by another unit (DMA); the engine's own PBUF writes are snooped internally |
| clear | `clr` | `eng_clr` of the command processor; same effect as reset, except that read responses still in flight are absorbed (no new reads until they are back) |
| status | `sat_clr`, `sat_cnt[159:0]`, `st_hits/st_misses/st_lines` | `sat_cnt` = {score, ln_z, ln_out, ALU, requant}; wire to `ctrl_top` with `NSAT = 5` (SAT_0..4) |

### 9.2 Command mapping

Rows are vec_unit commands (`cmd_len` ≤ 16 bits); every row starts on a line (alignment rule).

| ISA | rows | vec_unit op | A (elem) | B / P | D |
|---|---|---|---|---|---|
| VEC | ceil(len/4096) chunks | COPY→PASS, ADD, SUB, MUL (shamt 0), MAX, MIN, ADDI→ADD `b_imm`, SHRR→SRAR, SHL; `fmt` = dt_out | src0 (dt_in) | B = src1 for ADD..MIN | dst (dt_out) |
| REQUANT | M per column chunk of ≤ RQ_MAX (256) | REQUANT, `fmt` = out16 | ACC int32 | per-channel: bias/mult/shift from the record buffer; per-tensor: `b_imm`/`p_imm` from record 0 | ABUF int8/16 |
| LN | rows | SFU LN, imm = LN slot | ABUF int16/int8 | – | ABUF int8 |
| SOFTMAX_ONLINE | rows | SFU SOFTMAX_RQ, imm = {visible cols, ctx slot}; visible = min(cols, i + diag + 1) if causal | ACC int32 | – | ABUF int8 |
| GELU | ceil(len/4096) chunks | SFU GELU, imm = table slot | ABUF int8 | – | ABUF int8 |
| KV_APPEND | rows × heads | PASS int8 | ABUF, (t, h) at src + t·stride + h·dh | – | KVBUF + (h·ctx + pos + t)·dh |
| LUT_LOAD | – | 32 cfg writes (pbuf_loader) | – | – | – |

Errors (command skipped, `done` still pulses, first cause in `err_code`): 1 not a VEC-queue
opcode / reserved bits, 2 bad vop/dt/buffer/table, 3 SHRR/SHL imm ∉ [0, 31], 4 SOFTMAX
first/last ≠ 1 or causal with diag < 0 (a row with no visible column), 5 LN cols > D_MAX or
softmax cols > CTX_MAX, 6 LN/SOFTMAX before the LUT_LOAD of its tables, 7 KV_APPEND dh not a
multiple of 16 or pos + rows > ctx, 8 REQUANT record with ls ≠ 0 (found while loading; the
command completes), 9 SFU `err_len` / vec_unit `illegal_op`.

Limits: commands run one at a time (no overlap between commands; the pipeline drains, about
10–20 cycles). A destination may overlap its source only element for element (in place, same
element size: the compiler's residual ADD and GELU); funcsim's read-all-then-write semantics
are not reproduced for other overlaps. KV_APPEND needs dh % 16 = 0 (the compiler's rule).

### 9.3 Throughput

**Port beats (SG10b).** Each port moves one beat of `PL` consecutive 16-byte lines per cycle, from
any start line. A row (which starts on a line, alignment rule) is read as ceil(lines/PL) beats from
its first line, PL lines apart; the last beat is ragged: its lines past the row end are read but
never used (`vec_seq_rd` drops them with the row's last vec_unit beat, whose lanes past the row end
vec_unit masks), and on the D port they carry strobe 0, as do the bytes past the row end inside its
last line (`vec_seq_wr`: strobes = the lanes' keep bits). `vec_seq_walk` steps PL lines per beat;
the read credit counts beats. REQUANT records: `pbuf_loader` reads PL records per P-port beat and
writes them into the record buffer in one cycle (record `rec_idx + j` from line j). SFU blocks (LN,
GELU table, ctx record, LUT_LOAD) still enter the SFU through its 16-byte `cfg` port, one line per
cycle (docs/sfu.md §10), so their load cost does not change with PL. The engine's own PBUF writes are
snooped per line that carries a strobe.

A stream therefore delivers min(L, 16·PL/eb) elements per cycle:

| | int8 | int16 | int32 |
|---|---|---|---|
| PL = 1 (16 B) | min(L, 16) | min(L, 8) | 4 |
| PL = 4 (64 B, as built) | min(L, 64) | min(L, 32) | min(L, 16) |

so with PL = 4 every stream runs at vec_unit rate at L ≤ 16, and at L = 64 (KV260) int8 rows do
too while int16 rows run at 32 and int32 (REQUANT and softmax reading ACC) at 16 elements per cycle:
**the next VEC bottleneck at L = 64 is still the int32 port rate** (a 16-line, 256-byte A port would
be needed for 64 int32 per cycle), followed by the SFU's per-row fixed cost (LN / softmax: ~18–23
cycles per row on top of 3 passes of ceil(cols/L) beats; LN rows16 cols128 at L = 64 is 420 cycles,
26 per row, of which only 6 are data beats). vec_unit's L lanes consume the wider beats without change
(a 64-byte int8 response is 4 beats at L = 16, one at L = 64; int32 at L = 64 needs 4 responses per
beat, assembled in `vec_seq_rd`). On the modelled KV260 shakespeare 256-token prefill the critical path
is then 77 % GEMM, 8 % softmax, 7 % attention GEMMs, 4 % LN (`nanoxl.cyclemodel`, docs/top.md §6).

**Bank requirement.** A PL-line beat needs PL distinct banks from any start line, so `buf_spad` needs
NB ≥ PL (checked at elaboration); `nanoxl_top` builds NB = 2 × its widest port, i.e. 8 banks at N ≤ 16
with VP = 4 and 16 at N = 32 (ACC rows of 8 lines). See docs/top.md §3 for concurrent streams.

**Measured per-command occupancy** (AUDIT-1 A1-01, re-measured for SG10b; `python3 bench/vec_seq_perf.py`,
module `bench/vec_seq_perf_cocotb.py`, raw data and command fields in `bench/results/vec_seq_perf.json`
as `{"pl1"|"pl4": {L: ...}}`). Ports always ready, read latency 2, each shape run as two identical
back-to-back commands; occupancy = done to done (the engine runs one command at a time, so this
includes the drain), accept→done of a lone command is 1–3 cycles more. L = 4 and 16 on Icarus, L = 64
on Verilator. The PL = 1 columns are cycle-identical to the SG7b RTL (16-byte ports). The second
command of a pair is a residency hit (LN block, GELU table, score record, LUT) except in the `miss`
rows; every command is checked against funcsim by the shared harness. Columns: PL = 1 (L = 4, 16, 64),
then PL = 4 (L = 4, 16, 64).

| shape | L = 4 | L = 16 | L = 64 | L = 4 | L = 16 | L = 64 |
|---|---|---|---|---|---|---|
| REQUANT M16 N384 | 1999 | 2001 | 2007 | 1710 | 559 | 561 |
| REQUANT M16 N128 | 681 | 682 | 685 | 585 | 201 | 202 |
| REQUANT M1 N384 | 529 | 531 | 537 | 239 | 169 | 171 |
| REQUANT M16 N256 out16 | 1321 | 1323 | 1329 | 1129 | 361 | 363 |
| REQUANT M16 N256 pt | 1065 | 1067 | 1070 | 1065 | 298 | 299 |
| REQUANT M4 N64 | 157 | 158 | 161 | 109 | 61 | 62 |
| REQUANT M1 N65 | 108 | 109 | 112 | 60 | 48 | 49 |
| SOFTMAX rows16 cols256 | 3393 | 1768 | 1219 | 3393 | 1089 | 709 |
| SOFTMAX rows16 cols64 | 1089 | 616 | 436 | 1089 | 513 | 421 |
| SOFTMAX rows1 cols256 | 228 | 133 | 124 | 228 | 84 | 64 |
| SOFTMAX rows16 cols16 | 513 | 373 | 421 | 513 | 368 | 418 |
| SOFTMAX rows4 cols128 causal | 477 | 268 | 184 | 477 | 189 | 145 |
| LN rows1 cols128 | 132 | 69 | 63 | 131 | 60 | 45 |
| LN rows16 cols128 | 1857 | 804 | 483 | 1856 | 704 | 420 |
| LN rows16 cols128 in8 | 1857 | 706 | 475 | 1857 | 705 | 417 |
| LN rows1 cols384 | 324 | 133 | 115 | 323 | 107 | 61 |
| LN rows16 cols384 | 4929 | 1828 | 1315 | 4928 | 1472 | 676 |
| LN rows1 cols128 miss | 183 | 121 | 115 | 184 | 111 | 97 |
| LN rows16 cols128 miss | 1909 | 856 | 535 | 1909 | 756 | 472 |
| GELU len512 | 151 | 56 | 59 | 151 | 54 | 32 |
| GELU len128 | 54 | 32 | 35 | 54 | 30 | 25 |
| GELU len2048 | 535 | 152 | 155 | 534 | 151 | 56 |
| GELU len512 miss | 168 | 74 | 77 | 168 | 73 | 50 |
| VEC ADD int8 len2048 | 526 | 143 | 146 | 525 | 142 | 47 |
| VEC ADD int16 len2048 | 525 | 272 | 278 | 526 | 142 | 80 |
| VEC ADD int32 len1024 | 271 | 274 | 286 | 270 | 79 | 82 |
| VEC COPY int8 len2048 | 525 | 143 | 146 | 525 | 142 | 47 |
| VEC ADD int8 len128 | 45 | 23 | 26 | 46 | 22 | 17 |
| VEC ADD int8 len64 | 30 | 19 | 22 | 30 | 17 | 16 |
| KV_APPEND rows16 heads4 dh32 | 589 | 206 | 207 | 588 | 204 | 142 |
| KV_APPEND rows1 heads4 dh32 | 48 | 25 | 26 | 49 | 24 | 22 |
| KV_APPEND rows16 heads6 dh64 | 1645 | 494 | 497 | 1644 | 493 | 205 |
| KV_APPEND rows1 heads6 dh64 | 115 | 44 | 47 | 115 | 43 | 26 |
| LUT_LOAD hit | 6 | 6 | 6 | 6 | 6 | 6 |
| LUT_LOAD miss | 40 | 40 | 40 | 40 | 40 | 40 |

Reading the table: with 16-byte ports REQUANT costs the same at every L (int32 ACC rows at 4 elements
per cycle, plus N record lines: M16 N384 = 384 + 16·(64 + 32) lines + 81 fixed); with 64-byte ports
it drops 3.6× at L = 16 and 64 (M16 N384: 2001 → 559 ≈ 96 record beats + 16·(1 + 16) + 16·(1 + 8) row beats + 2 × 25 fixed;
L = 64 equals L = 16 because int32 is port bound at 16 per cycle). Softmax and int16 LN rows at L = 64
halve (1171 → 616, 1315 → 676); int8 streams at L = 64 reach 64 per cycle (GELU len2048 155 → 56,
KV_APPEND 497 → 205). At L = 4 only REQUANT changes (records 4 per cycle). `nanoxl/cyclemodel.py`
prices every row of this table within 5 % at its port width (`VEC_CAL[PL][L]`: the stream terms come
from the port and lane widths, only the fixed per-command / per-row costs are fitted; GELU / LN block
loads are priced at the 16-byte cfg port), pinned by `tests/audit/test_audit1.py::test_a1_01_*`.

### 9.4 Verification

`SIM=icarus|verilator python3 -m pytest -q tests/rtl/test_vec_seq.py` (L = 4, 16 × PL = 1, 4; slow
Verilator-only L = 64 × PL = 1, 4). The bench's ports move PL lines per beat from any line (reads wrap
at the buffer end like `buf_spad`); a write line with a strobe must lie inside its buffer. Every bench runs funcsim as the reference and checks, at each
`done`: the command's whole destination byte for byte, that no write fell outside it (byte
strobes), and all five saturation counters equal funcsim's per-op counts so far; every
batch ends with a compare of all five buffers.

| bench | what |
|---|---|
| `vec_seq_smoke` | one command of each kind, full-rate ports |
| `vec_seq_random` / `_fast_ports` | 40 / 25 (Icarus) or 150 / 100 (Verilator) random legal commands: every vop × dt_in × dt_out, any buffers, lengths 1..9000 (chunking), in-place forms, REQUANT per-channel / per-tensor / N > RQ_MAX, LN with bundle and random in-domain blocks (cols 1..384), softmax with causal limits and bundle / random score records, GELU, KV_APPEND, repeated LUT_LOADs, engine-written PBUF (VEC COPY over an LN block), DMA-rewritten parameter blocks between batches; random port ready/latency and command gaps, or full rate with back-to-back commands |
| `vec_seq_tiny_forward` | every VEC-queue command the compiler emits for the tiny W8A8 bundle for a prefill (6 or 8 tokens) + one decode step (LUT_LOAD, LN, REQUANT, KV_APPEND, SOFTMAX_ONLINE, VEC, GELU), DMA/GEMM done by funcsim, PBUF lines of every DMA_LOAD snooped |
| `vec_seq_errors` | every error code, skip + done, first cause kept, clr clears, legal commands still run after an error |
| `vec_seq_clear_mid_command` | 9 clr / rst_n pulses inside long commands with reads in flight: no done or write afterwards, idle, then a fresh batch is exact |
| `vec_seq_residency` | hit / miss / snooped-rewrite / eviction behaviour and cycle costs (docs/sfu.md §10) |

Mutation sanity (each killed, then reverted): pbuf_loader without snoop invalidation;
REQUANT per-channel mult taken from the wrong record (SG7b). SG10b, ragged-beat byte strobes at
PL = 4, L = 4: int8 lanes past the row end strobed (`ps[j] = 1`), and the narrow-beat merge
strobing every byte of its slice (`cs_n[p] = 1`); both fail `vec_seq_random` (write outside the
destination) and were reverted.
