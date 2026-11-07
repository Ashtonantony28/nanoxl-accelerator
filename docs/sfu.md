# Special-function units (SG5 / M3b)

The SFU block implements the integer nonlinear operators of contract C5 (frozen by
ADR-002): EXP2, RECIP, RSQRT, GELU, the full-row 3-pass SOFTMAX and LayerNorm. It attaches
to the vector unit's SFU hook (`docs/vector-unit.md` §5.6) and is bit-exact against the
executable spec `nanoxl/quant/ops.py`. LUT contents and LayerNorm/softmax parameters are
model-bundle data loaded at run time, in the PBUF layouts of `docs/isa.md` §3 (ADR-004:
`LUT_LOAD` tables, LN parameter blocks, the softmax ctx = score requant record), one
16-byte PBUF line per cycle; the RTL never computes them.

| File | Module | What it is |
|---|---|---|
| `rtl/lut_rom.sv` | `lut_rom` | runtime-written table: 2^AW/WN lines of WN entries (one PBUF line per write), one synchronous read port with enable; maps to LUTRAM |
| `rtl/sfu_exp2.sv` | `sfu_exp2` | C5 EXP2, one lane, 2 stages |
| `rtl/sfu_recip.sv` | `sfu_recip` | C5 RECIP (LUT + 1 Newton step), one lane, 4 stages |
| `rtl/sfu_rsqrt.sv` | `sfu_rsqrt` | C5 RSQRT (LUT + 1 Newton step), one lane, 5 stages |
| `rtl/sfu_elem.sv` | `sfu_elem` | L-lane elementwise pipeline: GELU LUT on L lanes, EXP2 / RECIP / RSQRT on EL shared lanes; 6 stages |
| `rtl/softmax.sv` | `softmax` | C5 SOFTMAX, exact full-row 3-pass with a row buffer (CTX_MAX); optional SOFTMAX_ONLINE score requant front end (SG3 `requant`) with per-layer score records and causal limit |
| `rtl/layernorm.sv` | `layernorm` | C5 LN with per-slot constants and g/b (D_MAX, NSLOT), saturation counters |
| `rtl/sfu.sv` | `sfu` | the hook-side block: routing, in-order output, configuration port, status |
| `rtl/sfu_vec.sv` | `sfu_vec` | `vec_unit` (HAS_SFU = 1) + `sfu`, wired port group to port group |

`vec_unit.sv` is **unchanged**: the hook already carries everything the SFUs need.

## 1. Operations on the hook

The vector unit forwards `cmd_op` 0x18–0x1F with sub-op `cmd_op[2:0]` and `imm = cmd_imm_b`.
Lanes are int32 (vec-unit convention); narrow values arrive sign-extended.

| Sub-op | cmd_op | Name | Lane input | Lane output | `imm` |
|---|---|---|---|---|---|
| 0 | 0x18 | EXP2 | `d` = lane[15:0] (uint16) | `e` (Q.15, ≤ 32768), zero-extended | – |
| 1 | 0x19 | RECIP | `S` = lane (uint32) | `{2'b0, b[5:0], 7'b0, y[16:0]}` | – |
| 2 | 0x1A | RSQRT | `Q` = lane (uint32) | `{k[7:0], y[23:0]}`, both two's complement | – |
| 3 | 0x1B | GELU | lane[7:0] | `gelu.lut[slot][x & 0xFF]`, sign-extended int8 | table slot (layer) |
| 4 | 0x1C | SOFTMAX | int16 logit (Q.8 log2), row ≤ CTX_MAX | `p` in [0,127] (int8, scale 1/127) | `[31:16]` visible columns (0 = all) |
| 5 | 0x1D | LN | int16 residual, row ≤ D_MAX | int8, sign-extended | `[7:0]` LN slot |
| 6 | 0x1E | SOFTMAX_RQ | int32 score `q·K^T`, row ≤ CTX_MAX | as SOFTMAX | `[7:0]` score-record slot, `[31:16]` visible columns |
| 7 | 0x1F | reserved | – | lane passed through; sticky `illegal_op` | – |

**ISA mapping.** `SOFTMAX_ONLINE` (first = last = 1) is one SOFTMAX_RQ row per score row:
`len = cols`, record slot = the layer's ctx record, visible columns = `min(cols, i + diag + 1)`
when `causal` (row i), else 0. SOFTMAX_RQ computes `s = requant16(wrap32((acc << ls) + bias);
mult, shift)` (C2, `nanoxl.graph.requant_ref`) on every lane, counts its int16 clamps on all
in-row lanes (the golden model counts `h{l}.score` over all keys of the step, masked or
not), then masks columns ≥ the visible count (p = 0, `keep` still 1). `LN` is one LN row per
input row with `imm` = the slot its block was loaded into; `GELU` is a GELU row with the
layer's table slot. `LUT_LOAD table t` is 32 cfg writes to region t.

A *row* is one vec-unit command (`ceil(len/L)` beats ending with `last`). EXP2, RECIP,
RSQRT and GELU are elementwise; SOFTMAX and LN reduce over the whole row and stream the
row back. In every case the hook rules hold: one output beat per input beat, in order,
`keep` and `last` unchanged. Inside `softmax` and `layernorm`, a lane with `keep = 0` is
a masked position: it is left out of max/sum/Σc² and returns 0. `vec_unit` only clears
`keep` on the tail lanes of the last beat; the standalone benches also test interior masks.

The packed RECIP/RSQRT formats exist so the primitives can be tested and used on their own;
SOFTMAX and LN use the same primitive modules internally and never unpack lanes.

### Domains and out-of-contract inputs

| Op | Bit-exact domain (vs `nanoxl.quant.ops`) | Outside it |
|---|---|---|
| EXP2 | all 65,536 inputs, any 16-bit table | – |
| RECIP | S in [2^15, 2^32), any 16-bit table (C5 needs [2^15, 2^24]) | S < 2^15: unspecified |
| RSQRT | Q in [1, 2^32) elementwise, [1, 2^48) inside LN (C5: ≤ 2^46), any 16-bit table (y is signed for that reason) | Q = 0 is treated as 1 |
| GELU | all 256 inputs, any table | – |
| SOFTMAX | rows of 1..CTX_MAX elements with ≥ 1 kept lane | no kept lane: all-zero output; longer row: `err_len`, data unspecified, beat count still correct |
| LN | rows of 1..D_MAX elements; `|mu| < 2^16` (true whenever `mu_mult/mu_shift` encode 1/d' with d' ≥ the number of kept elements, as the bundle does); `R = y·sqrt_d < 2^31` (C5) | longer row: `err_len`, data unspecified |

## 2. Configuration port (`sfu.sv`)

`cfg_we`, `cfg_region[3:0]`, `cfg_slot[7:0]`, `cfg_line[11:0]`, `cfg_cols[12:0]`,
`cfg_data[127:0]`: **one 16-byte PBUF line per cycle**, little-endian exactly as the bytes
sit in PBUF, registered decode. A loader (the VEC engine, SG6) copies a block line by line
from its PBUF address. Write only while the SFU is idle (`busy = 0`).

| Region | Contents (ISA §3 block, bundle key) | Lines | Cycles |
|---|---|---|---|
| 0, 1, 2 | `LUT_LOAD` table 0 `lut.exp2`, 1 `lut.recip`, 2 `lut.rsqrt` (uint16[256]) → `sfu_elem` and `softmax` / `layernorm` copies | 0..31 (8 entries each) | **32 per LUT_LOAD** |
| 3 | GELU LUT `h{l}.gelu.lut` (int8[256]) of slot `cfg_slot` | 0..15 (16 entries each) | 16 |
| 4 | LN parameter block (`funcsim.ln_block`) of slot `cfg_slot`; `cfg_cols` = the block's cols | line 0 `{mu_mult, mu_shift, sqrt_d, sh}`, line 1 `eps_q`, then `ceil(cols/8)` g lines, `ceil(cols/4)` b lines | 2 + ceil(cols/8) + ceil(cols/4): 50 at d = 128, 146 at d = 384 |
| 5 | softmax ctx = score requant record `{bias, mult, shift, ls}` of slot `cfg_slot` | 0 | 1 |

The slots are the loader's choice; the tests use LN slots `h0.ln_1, h0.ln_2, h1.ln_1, …,
ln_f` = 0 .. 2·n_layer, GELU slot = layer, score-record slot = layer. LN, GELU and score
parameters stay resident across tokens, so a loader that keeps a slot per (layer, op)
loads them once per program (the full tiny bundle is 614 line writes); a loader that
reloads a block before every LN command pays the line count above per command.
Tables are replicated per lane read port (`lut_rom` per consumer lane), as an FPGA mapping
would do with LUTRAM anyway; one write updates every copy. Each copy is stored as lines of one
PBUF line (WN = 8 uint16 or 16 int8 entries): one write port writing a whole line, one read
port reading a whole line asynchronously, the entry selected by the low address bits in front
of the read register. That is a plain 1W/1R memory, so it maps to LUTRAM (a 256 × uint16 table:
10 RAM32M16 + 48 LUT + 16 FF; a GELU table of 8 slots: 38 RAM64M8 + 90 LUT + 8 FF) instead of
flip-flops and a 256:1 mux (docs/fpga.md §2.3 C1). Read timing and contents are unchanged.

## 3. Elementwise pipeline (`sfu_elem`)

6 stages, whole-pipeline stall (`in_ready = (!out_valid || out_ready)` and no sub-beat
pending). Each primitive is fed from the tap of an input delay line that makes all of them
finish in stage 5, and stage 6 selects by the per-beat op and packs. Op and slot travel with
the beat, so rows of different ops follow each other with no bubble.

**Lane sharing** (parameter `EL`, docs/fpga.md §2.3 C3). GELU and the pass-through sub-ops
have L lanes and run at one beat per cycle. EXP2, RECIP and RSQRT have `EL` primitive lanes
(`EL` divides L; `sfu` sets `EL = min(L, 8)`): a beat of those ops is split into R = L/EL
sub-beats that enter the pipeline on consecutive cycles (sub-beat 0 straight from `in_data`,
the rest from a shift register) and are shifted into the output register from the top;
`out_valid` rises with the last one. So they cost R cycles per beat and R − 1 cycles of extra
latency; results are unchanged. The compiler emits no elementwise EXP2 / RECIP / RSQRT (softmax
and LN use their own primitives), so no compiled program is slower.

| L (as built) | EL | EXP2 / RECIP / RSQRT | GELU, pass-through |
|---|---|---|---|
| 4 | 4 | 1 cycle / beat (4 elements) | 1 cycle / beat |
| 16 | 8 | 2 cycles / beat (8 elements per cycle) | 1 cycle / beat |
| 64 | 8 | 8 cycles / beat (8 elements per cycle) | 1 cycle / beat |

| Primitive | Stages | Work |
|---|---|---|
| RSQRT | E1 bitlen, normalise `m` and `k` · E2 LUT · E3 `t = y0²>>14` · E4 `u = m·t>>16`, `3·2^14 − u` · E5 `y0·(…)` | 3 multipliers (16×16, 16×18, 16×20) |
| RECIP | E1 bitlen, normalise · E2 LUT · E3 `2^32 − m·y0` · E4 `y0·(…)`, `y = p >> 31` | 2 multipliers (16×16, 16×33) |
| EXP2 | E1 LUT[`d & 255`], `q = d >> 8` · E2 `q ≥ 16 ? 0 : lut >> q` | none |
| GELU | E1 LUT[`{slot, x & 0xFF}`] | none |

Latency: a GELU / pass-through beat accepted at edge k is on `out_*` after edge k+5, an EXP2 /
RECIP / RSQRT beat after edge k+5+R−1 (test `elem_pipeline_depth`).

## 4. SOFTMAX (`softmax`)

C5 exactly, as the full-row 3-pass (an online variant is not bit-identical; ADR-002):

0. **Requant front end** (SOFTMAX_RQ only): per lane `(acc << ls)` (32-bit wrap) into the SG3
   `requant` pipeline (6 stages) with the slot's `bias, mult, shift`, int16 out; clamps are
   counted (`sm_sat_count`). `NCTX` record slots. The requantizer has `RQL` lanes (parameter,
   divides L; `sfu` sets `SM_RQL = min(L, 16)`, docs/fpga.md §2.3 C6): with RQL < L each int32
   beat goes through as R = L/RQL sub-beats on consecutive cycles (sub-beat 0 straight from the
   input, the rest from a shift register) and is reassembled before the row buffer, so the row
   streams in at one beat per R cycles. 16 lanes is the int32 rate of the 64-byte VEC A port,
   so at L = 64 the requant costs 48 × 4 = 192 DSP48E2 less for a few cycles per row (below).
1. **Pass 1** (while the row streams in, `in_ready = 1`): `M = max s_j` over visible lanes
   (`keep` and column < visible count); the int16 row, its keep bits and its mask go into
   the row buffer (`ceil(CTX_MAX/L)` beats).
2. **Pass 2**: buffer read → `d = M − s` (uint16) → EXP2 → lane sum → `S` (`16 + log2(CTX_MAX) + 1` bits).
3. **RECIP** of `S` (scalar `sfu_recip`), then `127·y` and `sh = 15 + b`.
4. **Pass 3**: buffer re-read → EXP2 again → `e·(127·y)` (16×24 multiplier per lane) →
   `rshr(·, sh)` → `min(127, ·)` → output.

| Pipeline (passes 2/3) | A buffer read · B d, EXP2 E1 · C EXP2 E2, sum (pass 2) · D product (pass 3) · E round, clamp, out register |
|---|---|
| Cycles per row, full rate (measured) | **SOFTMAX: 3·nb + 13**, nb = ceil(len/L): nb in, nb + 3 pass 2, 5 RECIP, nb + 5 pass 3. **SOFTMAX_RQ: (R + 2)·nb + 19** (R·nb in, +6 requant pipeline; R = L/RQL, 1 at L ≤ 16) |
| Examples (SOFTMAX_RQ) | ctx 256: 211 cycles at L = 4, 67 at L = 16, 787 at L = 1. ctx 128 (tiny) at L = 4: 115. KV260 L = 64 (RQL = 16), ctx 256: 43 (31 with RQL = 64) |
| Backpressure | `out_ready` stalls pass 3 only; the next row is accepted once pass 3 has drained |

The softmax itself cannot saturate (C5 bounds); the only clamp is the score requant of
SOFTMAX_RQ (`sm_sat_count`, 32-bit saturating, sticky; matches `Golden.sat_count["h{l}.score"]`).

## 5. LayerNorm (`layernorm`)

C5 exactly. Constants and affine parameters are per slot (`NSLOT`, `in_slot = imm[7:0]`),
written from the ISA LN parameter block (§2) and latched at the first beat of the row. The
g/b memories are banked so that one PBUF line (8 g or 4 b) is one write and one beat's L
values are one read (`L` must be a power of two).

1. **Pass 1** (streaming in): `S = Σ x` (int32) and the row buffer.
2. **mu** = `rdivpot(srdhm(S, mu_mult), mu_shift)`: the SG3 `srdhm` (3 stages) + a 32-bit RDivPOT.
3. **Pass 2**: `c = x − mu` (int18) → `c²` (17×17-class multiplier per lane) → lane sum → `Q` (48 bits);
   `Q = max(Q + eps_q, 1)`.
4. **RSQRT** of `Q` (scalar `sfu_rsqrt`, QW = 48), then `R = y·sqrt_d` (kept as int32; C5: R < 2^31) and `22 + k`.
5. **Pass 3**: `c·R` (18×32 per lane) → `z = sat16(rshr(·, 22+k))` [**ln_z** event] →
   `z·g + b` (16×16 + 32) → `rdivpot(·, sh)` → int32 wrap (the reference's `astype(int32)`) →
   `sat8` [**ln_out** event].

| Pipeline (passes 2/3) | A buffer + g/b read · B c · C c² or c·R · D Q accumulate or z · E z·g+b · F rdivpot, sat8, out register |
|---|---|
| Cycles per row, full rate (measured) | **3·nb + 19**: nb in, 4 mu, nb + 3 pass 2, 6 RSQRT, nb + 6 pass 3 |
| Examples | d = 384: 307 cycles at L = 4, 91 at L = 16, 37 at L = 64 (KV260). d = 128 (tiny) at L = 4: 115 |
| Storage | row buffer `ceil(D_MAX/L)` × L × 16 bits; g `NSLOT · D_MAX` × 16 bits, b `NSLOT · D_MAX` × 32 bits |

**Saturation counters**: `z_sat_count` and `out_sat_count` (exposed by `sfu` as
`ln_z_sat_count` / `ln_out_sat_count`) are 32-bit saturating `sat_counter`s with a sticky
flag, counting kept lanes that clamp, on output transfers; `stat_clr` clears them. They
are the RTL side of `Golden.sat_count["ln_z"]` and of the sum of the per-LN `ln_out`
counts (`Golden.sat_count["h{l}.ln_1"]`, …, `["ln_f"]`). These are the only saturation
points C5 lists for the SFUs; EXP2, RECIP, RSQRT, softmax and GELU cannot overflow.

## 6. SFU top (`sfu`) and integration

`sfu` routes each row to `sfu_elem` (sub-ops 0–3, 7), `softmax` (4, 6) or `layernorm` (5)
and keeps a queue (depth 4) of the unit of each row in flight; the output mux follows the
queue head, so rows leave in arrival order even when a softmax row is still computing
while a GELU row queues behind it. `in_ready` depends combinationally on `out_ready`
(through the elementwise pipeline's stall) and never on `in_valid`, so the hook has no
combinational loop. Status: `sm_sat_count`, `ln_*_sat_count`, `sat_sticky`, `illegal_op`
(sub-op 7), `err_len`, `busy`.

`sfu_vec` instantiates `vec_unit #(.HAS_SFU(1))` and `sfu` and connects the hook; its
status ORs the sticky flags and `busy`. The VEC engine (SG6) executes `LUT_LOAD` as 32 cfg
line writes and loads the LN blocks / GELU tables / score records from PBUF into slots.

### Measured timing for the cycle model (`nanoxl/cyclemodel.py`, ADR-004 estimates)

| Command | Measured SFU occupancy | Cycle-model knobs |
|---|---|---|
| `LN` (rows × cols) | rows × (3·ceil(cols/L) + 19); rows do not overlap; + block load 2 + ceil(cols/8) + ceil(cols/4) cycles if the slot is not resident | `sfu_row_bubble = 19`, `row_latency_ln = 0` |
| `SOFTMAX_ONLINE` (rows × cols) | rows × ((R + 2)·ceil(cols/L) + 19) (int32 scores via SOFTMAX_RQ; R = L/SM_RQL = 4 at L = 64, else 1); + 1 cycle per score record load. In vec_seq the int32 A-port rate hides most of R (`VEC_CAL[·][64]["sm"]` refit for SG11a) | `sfu_row_bubble = 19`, `row_latency_softmax = 0` |
| `LUT_LOAD` | 32 cycles per table (one 16-byte line per cycle; the ADR-004 model assumes 16) | `LUT_ENTRIES // 8` |
| `GELU` (len) | ceil(len/L) cycles + 6 latency, back to back with other elementwise rows; + 16 cycles per table load | — |

All numbers are exact for `in_valid = out_ready = 1` (tests `softmax_full_rate`,
`softmax_rq_full_rate`, `ln_full_rate`, `elem_bundle_tables`), and exclude the vec_unit
command-accept cycle (`docs/vector-unit.md` §5.4).

## 7. Verification

`SIM=verilator python3 -m pytest -q tests/rtl -k "sfu or softmax or layernorm or gelu or vec"`;
the same on Icarus (default `SIM`). Sample sizes are larger under Verilator.

| Test | What it checks |
|---|---|
| `test_sfu_elem.py` (L = 1, 4, 16; shared lanes L/EL = 4/1, 16/4, 16/8) | EXP2 **exhaustive** over all 65,536 inputs (junk in the upper lane bits), GELU **exhaustive** (256 × 4 bundle tables), RECIP over [2^15, 2^32) and RSQRT over [0, 2^32): every binade edge + 2–8k random each. Ops shuffled back to back, interior keep holes, three handshake regimes; full-rate throughput; depth = 6. A second test loads **random tables** (all four) and checks the unit follows them, then reloads the bundle tables. |
| `test_gelu.py` (L = 1, 4) | GELU exhaustive for every per-layer table of the tiny bundle, shuffled inputs. |
| `test_softmax.py` (L = 1, 4, 16; shared score requant L/RQL = 4/1, 16/4) | **SOFTMAX_RQ**: the golden model's int32 `q·K^T` rows (all keys of the step, causal limit i + 1, the layer's score record), P bit-exact and, on Verilator with every row replayed, the score clamp count equal to `Golden.sat_count["h{l}.score"]`; random int32 rows on random records (bias, ls 0..3, every shift) with clamps counted exactly, `sat_clr`; full-rate count 3·nb + 19. Records written as PBUF lines. Random int16 rows (lengths 1..256, six logit distributions, 25 % with interior masks, 20 % with a causal limit); **real attention-score rows** captured from `nanoxl.golden` on the tiny bundle (all 4·4·128 = 2,048 rows on Verilator, a sample incl. the long rows on Icarus), bit-exact against the golden P; corners: all-equal rows, ±32768 alternating and single-outlier rows, length 1 and 256, the EXP2 cut-off d = 15·256..16·256, masked lanes holding the row maximum, a single kept lane; full-rate cycle count = 3·nb + 13 exactly; `err_len` on a CTX_MAX + L + 1 element row and recovery. |
| `test_layernorm.py` (L = 1, 4, 16) | Every slot is loaded from its PBUF LN block (`nanoxl.funcsim.ln_block`), line by line. Slots 0..8 = the tiny bundle's LayerNorms, slots 9..15 random constants (eps_q 0..2^30, sqrt_d of a wider LN, every `sh`, extreme g/b, small g with many exact rounding halves) for rows up to 384. Random rows (lengths 1..384, interior masks) with both saturation points exercised and the counters equal to the reference counts; **every LN input row** of one 128-token golden forward pass (9·128 = 1,152 rows; sampled at L = 1, 4 on Icarus), bit-exact, with the counters equal to `Golden.sat_count`; corners: all-equal rows (Q = eps_q, Q = 0 → 1), ±32768 rows, one-outlier rows, length 1, L ± 1, 383, 384; full-rate cycle count = 3·nb + 19; `sat_clr`/sticky; `err_len` and recovery. |
| `test_sfu_vec.py` (L = 1, 4, 16) | `vec_unit` + `sfu` end to end: random command mixes of ALU, REQUANT, REDUCE and every SFU sub-op (incl. SOFTMAX_RQ with causal limits, reserved → `illegal_op`) with random valid gaps / backpressure on all six streams; LUT_LOAD tables, GELU tables, LN blocks and score records written through the cfg port as PBUF lines; every D and R beat, all five saturation counters, `illegal_op`, `busy`. A layer-like back-to-back sequence (LN per slot, GELU per table, softmax and SOFTMAX_RQ rows, lengths 1..256). |
| `test_sfu_synth.py` (slow) | Yosys generic synthesis at L = 4, no latches (§8). |
| `test_vec_seq.py` (SG7b) | the VEC engine drives these units from ISA commands; its L = 64 run found the lane-position overflow fixed in `layernorm.sv` (gp/bp were 5 bits, wrapping for lanes ≥ 32) |
| existing `test_vec_*` | unchanged and passing (`vec_unit.sv` is not modified). |

Mutation sanity results are recorded in the SG5 report.

## 8. Synthesis (Yosys 0.33 generic `synth -flatten`, L = 4)

Generic synthesis maps memories and multipliers to gates and flip-flops, so these counts
measure synthesizability, not FPGA area (on the KV260 the tables go to LUTRAM, the row
buffers and g/b memories to BRAM, the multipliers to DSP48E2).

`tests/rtl/test_sfu_synth.py` (all `slow`; about 8 minutes in total) asserts no latches and
prints the counts. To rerun: `python3 -m pytest -q -s tests/rtl/test_sfu_synth.py`.

| Top (L = 4) | Parameters | Cells | Flip-flops |
|---|---|---|---|
| `sfu_elem` | NG = 4 | 149,634 | 22,779 |
| `softmax` | CTX_MAX = 256, NCTX = 4 | 97,969 | 14,930 |
| `layernorm` | D_MAX = 384, NSLOT = 1 | 119,836 | 31,562 |
| `sfu` (all of the above + routing, cfg decode) | same | 371,448 | 69,456 |

Most flip-flops are storage (per-lane table copies, row buffers, g/b memories); most cells
are the multipliers and the memory read muxes.

## 9. Known limits and integration notes

- **Throughput.** Softmax and LN hold one row at a time (≈ 3 passes over the row). A second
  row buffer (ping-pong) would overlap pass 1 of row n+1 with passes 2–3 of row n and bring
  the cost to ≈ 2·nb per row; worth it only if ADR-003's schedule shows the SFU on the
  critical path (decode rows are short: H softmax rows of p+1 elements and 2 LN rows per layer).
- **Multipliers per lane**: elementwise 5 per shared lane (RECIP 2, RSQRT 3; `EL` lanes, not L),
  softmax 1 (16×24) per lane + the score requant's 32×32 per requant lane (`SM_RQL`), LN 3
  (17×17, 18×32, 16×16), plus scalar srdhm (32×32), RECIP and RSQRT.
- **Tables per lane**: each consumer lane has its own copy (one read port each); the cfg
  port writes all copies at once.
- `in_fmt` and `in_shamt` from the hook are not used by any C5 op.

## 10. PBUF parameter loader and slot residency (`pbuf_loader`, SG7b)

`rtl/pbuf_loader.sv` is the loader §2 and §6 ask for: the VEC engine (`vec_seq`,
docs/vector-unit.md §9) hands it one block per command, and it copies the block from PBUF
over a 16-byte line read port (up to 16 lines in flight, one line per cycle) into the cfg port.

| ISA command | block | region / slot | lines |
|---|---|---|---|
| LUT_LOAD t | 256 × uint16 | region t (0..2) | 32 |
| GELU | GELU LUT at `lut_addr` | 3, one of NG slots | 16 |
| LN | LN block at `param_addr` (`cols`) | 4, one of NSLOT slots | 2 + ceil(cols/8) + ceil(cols/4) |
| SOFTMAX_ONLINE | score record at `ctx_addr` | 5, one of NCTX slots | 1 |
| REQUANT | N records (or record 0 if per-tensor) | engine record buffer, never cached | N per column chunk |

**Residency policy.** Each slot carries a tag {valid, first PBUF line, end line, cols}. A
block whose first line (and, for LN, cols) matches a valid tag is a hit: no PBUF read, 1
cycle. A miss loads into the first free slot of its kind, else the round-robin victim. A tag
is dropped when any PBUF line in [first, end) is written: the DMA's writes come in on
`pb_snoop_*`, the engine's own (a VEC with `dst_buf` = PBUF) are snooped internally; reset and
`clr` drop all tags (and the LUT-loaded flags). A hit therefore always uses exactly the bytes
PBUF holds at that moment, for any program, with no compiler cooperation. This is the same
rule `nanoxl/cyclemodel.py` (SG4c) models.

**The current compiler** DMA-loads every LN block, GELU table, score+PV record pair and the
SFU tables in every program (they live in the PBUF ring), so the snoop invalidates them and
every LN / GELU / LUT_LOAD of a program misses; the softmax ctx record is shared by the H
heads of a layer and hits H − 1 times. Measured on the tiny W8A8 bundle (2 programs:
prefill + decode): 40 misses, 24 hits (all ctx). Per program the reloads are
9 LN × 50 + 4 GELU × 16 + 4 ctx + 3 × 32 LUT = 614 lines, about 614 + 20 × (latency + 4)
cycles, i.e. ≈ 0.7k cycles of a 1-token decode program's VEC queue at L = 4. Measured
per command (L = 4, read latency 2): LN 1 × 128 miss 185 cycles vs hit 134 (51 cycles =
50 lines + 1); LUT_LOAD miss 42 vs hit 8. With a compiler that pins parameter blocks at fixed
PBUF addresses and skips their re-DMA (`TINY_SNOOP=changed` in the bench models this), every
program after the first is all hits; the hardware needs no change for that.

Slot counts must cover the working set for full residency: tiny needs NSLOT ≥ 9 LN, NG ≥ 4
GELU, NCTX ≥ 4; `shakespeare` (6 layers) needs NSLOT ≥ 13, NG ≥ 6, NCTX ≥ 6 (NG = 6 exceeds the `sfu_vec`
default NG = 4). Fewer slots are still correct (round-robin
evictions reload).
