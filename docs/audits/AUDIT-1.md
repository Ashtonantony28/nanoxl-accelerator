# AUDIT-1: merged NanoXL code on main (waves 1-3)

Scope: `main` at `ae3f1c4` (SG0-SG7b merged): `nanoxl/` (numerics, quant, golden, export, compiler,
funcsim, isa, cyclemodel, driver), `rtl/` (all except `nanoxl_top.sv`), `tests/`, and the docs listed
in the brief. Independent audit; nothing was fixed. Repro tests: `tests/audit/` (failing findings are
`xfail(strict=True)` and name their finding id). The mutation harness is `tests/audit/mutate.py`.

**Result: 0 critical, 1 high, 1 medium, 8 low.** No RTL-vs-contract correctness bug was found. The RTL
engines match their references bit for bit on every test the audit ran or wrote, and 23 of 28 targeted
mutants were killed by the existing suite (§3). The high and medium findings are a performance-model
gap and a CSR mapping mismatch between the RTL and the host driver.

Baseline runs on this checkout (Icarus 12):

| command | result |
|---|---|
| `python3 -m pytest -q tests/py -m "not slow"` | 193 passed, 2 deselected, 101 s |
| `SIM=icarus python3 -m pytest -q tests/rtl -m "not slow"` | 79 passed, 11 deselected, 577 s |
| `python3 -m pytest -q tests/rtl/test_ctrl_formal.py` | 4 passed, 23 s (yosys-smtbmc + z3 present) |
| `python3 -m pytest -q tests/audit -rxX` | 7 passed, 5 xfailed, 8 s |
| `NANOXL_DATA=/home/claude/nanoxl/data/raw python3 -m pytest -q tests/audit -m slow` | 2 passed, 8 s |
| `make lint` (Verilator 5.020 `--lint-only -Wall`) | warnings only (PINCONNECTEMPTY in vec_seq, WIDTHTRUNC `layernorm.sv:238-239`) |

## 1. Findings (ranked)

### A1-01 (high): the cycle model prices VEC-queue work at 64 lanes, but the built VEC engine moves 16 bytes per cycle per port
* **Where:** `nanoxl/cyclemodel.py:206-225` (`cost()` for VEC/REQUANT/LN/SOFTMAX/GELU/KV_APPEND), its
  module docstring ("Every cost below is a measured number"), against `rtl/vec_seq.sv` and
  `docs/vector-unit.md` §9.3 ("Each port moves one 16-byte line per cycle ... at L = 16/64 int32
  (REQUANT and softmax scores from ACC) and wide int16 rows are port-bound").
* **Scenario:** KV260 (`lanes=64`). The model charges `REQUANT M=16, N=384` as 16·(1+6) = 112 cycles.
  The engine reads int32 ACC rows at 4 elements per cycle and also reads one 16-byte record per output
  channel per chunk (N lines), so the same command takes at least 16·(1+96) + 384 = 1,936 cycles,
  17× more. Softmax scores (int32) and LN rows (int16) are 16× and 8× port-bound in the same way. The
  model also leaves out the per-command pipeline drain (vector-unit.md §9.2: 10-20 cycles; the engine
  runs commands one at a time).
* **Impact:** ADR-003's update says "Batch-1 G3 now holds for prefill: the array is 83 % busy" on the
  KV260. That number assumes the 64-lane rate. Re-running `cyclemodel.estimate` for shakespeare W8 with
  `lanes` = 64 / 16 / 4 (int8 / int16 / int32 port rates bracket the real mix) gives prefill
  13.9 / 16.5 / 27.4 ms and array utilisation 0.83 / 0.70 / 0.42. Decode stays DMA-bound (333 tok/s
  either way). The G3 prefill claim is therefore not supported by the as-built RTL. SG7c may widen the
  ports, as §9.3 asks. Until then the model is up to about 2× optimistic for prefill.
* **Repro:** `python3 -m pytest -q tests/audit -k a1_01 -rxX` (xfail: `cost(REQUANT) >= port bound`).
  What-if numbers: `estimate(SHAKESPEARE.with_(wbits=8), KV260.with_(lanes=L), prefill_len=None, n_pos=3)`.

### A1-02 (medium): SAT CSR map disagrees between RTL, driver and SimBackend; the softmax score counter is not reachable
* **Where:** `rtl/vec_seq.sv:320` `assign sat_cnt = {sm_cnt, lnz_cnt, lno_cnt, alu_cnt, rq_cnt}`
  (LSB first: 0 requant, 1 ALU, **2 LN out, 3 LN z**, 4 softmax score; vector-unit.md §9.1 says
  "wire to ctrl_top with NSAT = 5") vs `rtl/ctrl_top.sv:20` `NSAT = 4`; `nanoxl/driver/csr.py:34-35`
  `NSAT = 4`, `SAT_NAMES = ("requant", "alu", "ln_z", "ln_out")`; `nanoxl/driver/backends.py:254-259`
  (SimBackend charges LN z to slot 2, LN out to slot 3, and folds `SOFTMAX_ONLINE` score clamps into
  slot 0); docs/control.md §5 / docs/driver.md describe SAT_0..3 only.
* **Scenario:** on hardware (or an RTL top wired as vec_seq documents), an LN whose normalize step
  clamps raises SAT_3, which `NanoXL.saturation()` reports as `ln_out`. An LN output clamp is reported
  as `ln_z`. With `NSAT = 4` the softmax score counter (the golden `h{l}.score` events, which are C5
  countable) is never readable. With `NSAT = 5` it is readable but the driver does not read it. The M6
  plan in driver.md ("SAT_* with the golden saturation counts") would show SimBackend and hardware
  disagreeing on SAT_0/2/3 for a correct design.
* **Repro:** `python3 -m pytest -q tests/audit -k a1_02 -rxX` (xfail `test_a1_02_sat_csr_map_matches_rtl`).
  SG7c owns the `nanoxl_top` wiring; the driver/SimBackend need the same order.

### A1-03 (low): the C5 integer ops are pinned only by accuracy bounds and committed tokens in tests/py
* **Where:** `tests/py/test_golden_ops.py` (float error bounds), `tests/py/test_golden_model.py::test_reproduces_committed_reference_tokens`.
* **Scenario:** mutant `gold_lnaff_rshr` (`nanoxl/quant/ops.py` `ln_affine_i`: `rdivpot(acc, sh)` →
  `rshr(acc, sh)`, a change of C5 LN step 7) passes all 71 `tests/py -k golden` tests: the 100
  reference tokens and saturation counts do not change, and funcsim imports the same op, so
  funcsim-vs-golden stays equal. Only `tests/rtl/test_layernorm` would notice, because the unchanged RTL
  would then disagree with its (mutated) reference. The executable spec has no bit-exact vectors of its
  own derived from the contract text.
* **Repro:** `tests/audit/mutate.py <copy> gold_lnaff_rshr` (SURVIVED). Closed by the new
  `test_a1_03_c5_rounding_vectors_from_contract_text` (kills the mutant).

### A1-04 (low): the held-out accuracy test is too weak to catch a large quantization regression
* **Where:** `tests/py/test_golden_model.py:80-108`.
* **Scenario:** 4 windows (512 tokens), and the budgets are the G2 targets themselves (ΔPPL < +3 % /
  +8 %), while the measured values are +0.96 % / +3.29 %. A change that doubles the W4A8 loss (to
  about +6.5 %) still passes. The top-1 floors (0.92 / 0.84) are 2.6 / 4 points under the study. The
  "held-out" windows come from the same val split that the study evaluated and that picked the
  checkpoint ("best" = last eval at iter 5,000, `ckpt.json`). Only the seed is held out; no tuning
  data leaks into the model. A tighter guard: the study's own subset (`tests/audit` slow test) with
  ±1 pp around the study values.
* **Repro:** inferred from reading; the guard is `test_a1_numerics_quant_study_headline_rederived` (passes).

### A1-05 (low): smaller cycle-model gaps against the measured RTL
* **Where:** `nanoxl/cyclemodel.py:167-198` and docstring; ADR-004 "Cost".
* `GEMM_GS`: array_seq sends the scale beat on **every** tile ("every GEMM_GS tile carries its group's
  scale beat", array.md §7). The model, its docstring and ADR-004 charge it on the first tile of each
  group only: 1,560 vs measured 1,584 cycles (-1.5 %) for c_attn at N=32, G=64.
* Ragged weight rows (LM head, N = 65 bytes) cost a second W-port beat per row: model 384 vs measured
  624 (-38 %).
* Per-row/command fixed costs: LN 1×128 at L=4 is modelled at 116 (hit) / 166 (miss) cycles, measured
  134 / 185 (sfu.md §10, ~13 % low); LUT_LOAD 32 vs 42 measured; array_seq accept→done adds 8-23 cycles
  beyond 2N+4 (array.md §7), not modelled.
* **Repro:** `python3 -m pytest -q tests/audit -k a1_05 -rxX` (3 xfails, using the measured numbers in docs/array.md §7).

### A1-06 (low): stale or contradictory documentation
* `docs/array.md` §7 table column "model occ" (528, 272, 416 ...) and the paragraph after it ("the
  cycle model assumes `wload_rows_per_cycle = 2` ... predicts 2x to 3.8x fewer cycles"), plus PLAN.md
  SG7a ("cycle model 2-4x optimistic for weight-bound tiles"): the merged SG4c model already uses one
  row per cycle and matches 1,024 exactly (`test_a1_06_cyclemodel_now_matches_weight_bound_int8_tiles`, passes).
* `docs/adr/ADR-004-*.md`: status still "proposed", and it says "no RTL is changed here", but C1b/C3
  and `array_ctrl`/`array_seq` implement it. Its cost section says the scale multiply adds "+1 pipeline
  stage of latency", while C3 (and the RTL, which multiplies in the existing RMW stage) says 2N+4 also
  with `gs`.
* `docs/isa.md` §4 GEMM: "`gemv=1` selects the low-latency path that skips skew/deskew". No such path
  exists. array_seq ignores the bit (array.md §7), and the model charges 2N+4.
* `docs/design-plan.md` §4 and §6 still describe online softmax with UINT8 probabilities, SmoothQuant
  on, and requant rounding "round-half-to-even". ADR-002/C2/C5 chose a full-row 3-pass softmax, int8
  P in [0,127], SmoothQuant off, and gemmlowp rounding (half toward +inf in SRDHM, half away from zero
  in RDivPOT). Nothing in the plan points to the superseding ADR.
* `docs/control.md` §5 SAT_i row ("i < NSAT = 4: requant / ALU / LN"): see A1-02.

### A1-07 (low): funcsim accepts a command the RTL rejects (KV_APPEND dh % 16)
* **Where:** `rtl/vec_seq.sv:385` (`kv_dh[3:0] != 0` → error 7) vs `nanoxl/funcsim.py` KV_APPEND (no
  check) and `docs/isa.md` §4 KV_APPEND (no restriction stated; §3 only says the compiler keeps
  `d_head % 16 == 0`).
* **Scenario:** a hand-written or future-compiler KV_APPEND with dh = 8 runs in funcsim/SimBackend and
  halts the hardware with ERR 7. The HW size limits (LN cols > D_MAX, softmax cols > CTX_MAX) are the
  same kind of mismatch, but they are parameters and are documented in vector-unit.md §9.2.
* **Repro:** inferred from reading.

### A1-08 (low): quantizer casts bias to int32 without a range check
* **Where:** `nanoxl/quant/quantize.py:142, 171` (`np.round(bf / (s_in*sw)).astype(np.int32)`).
* **Scenario:** a model with a large bias relative to `s_in·unit` (W4A8 `unit = max_g s_g/255` is
  small) wraps silently instead of failing. Today's bundles have max |bias| = 2^14.8 (W8A8) and 2^18.8
  (W4A8), so this does not trigger. The multipliers have an assert (`qmul`); the bias does not.
* **Repro:** inferred from reading. The numbers came from a one-line check over the committed bundles.

### A1-09 (low): the score pre-shift `ls` path is never exercised end to end by the existing tests
* **Where:** `nanoxl/golden.py:134` (`acc << ls`). Both committed bundles have `score.ls = 0`.
* **Scenario:** mutant `gold_ls_drop` (drop `<< ls` in golden) survives `tests/py -k golden`. RTL
  softmax tests cover `ls` against `ops`/funcsim, but nothing compares golden with the compiled program
  for `ls > 0`.
* **Repro:** closed by the new `test_a1_09_score_ls_path_golden_equals_funcsim[1,3]` (bundle variant
  with ls = 1, 3; passes, and kills the mutant).

### A1-10 (low): fast-tier coverage and runtime
* The shipped configuration (array N=32, L=64, M_DEPTH 256) is exercised only by slow,
  Verilator-only tests. `make test-fast` runs N ≤ 16 and L ≤ 16.
* `sem_unit` same-cycle arbitration (`used_w`, `used_s`) is killed only by formal (`test_ctrl_formal`).
  The cocotb ctrl scoreboard passes both mutants (§3), and the formal test skips silently when z3 or
  yosys-smtbmc is missing.
* The fast tier takes about 11.3 min on Icarus (101 s Python + 577 s RTL). The G5 target is 10 min on
  Verilator, which this audit did not measure.

## 2. Numerics claims re-derived

| check | result |
|---|---|
| Committed `tiny/{w8a8,w4a8}.npz` == `build_bundle(ckpt, calib_windows(train, 128, 16, seed 20261106))` | identical, every array (`test_a1_numerics_committed_bundles_rebuild_identically`) |
| ΔPPL on 40 of the study's 200 val windows (every 5th, 5,120 tokens), FP32 on the same windows | W8A8 **+0.55 %** (study +0.96 %), W4A8 **+3.28 %** (study +3.29 %). Within the study's ±0.3 % subset noise scaled to 1/5 of the tokens; both under G2 |
| Calibration only from the train split | yes (`export.calib_windows` on `train`; eval uses `val`) |
| Golden semantics vs FP32 model | GELU exact erf (LUT built from `gelu_f` = erf; `model_fp` uses `F.gelu` default) ✓; LN eps 1e-5 folded as `eps_q = d·eps/s_res²` (biased variance, as torch) ✓; tied LM head (`lm_head` quantized from `wte`) ✓; scores scaled by `1/sqrt(hs)·log2 e·2^8` matching SDPA's default scale ✓; causal mask `k <= q` ✓ |
| Width bounds in C5 | recip y ≤ 65535 (17-bit field ok), 127·y < 2^23 (fits softmax's 24-bit `y127`), rsqrt y ≤ 32767, max bias 2^18.8 |

## 3. Mutation results (each mutant applied to a scratch copy, its selector run, then reverted)

`python3 tests/audit/mutate.py <copy-of-repo> [ids]`. 28 mutants: 23 killed, 5 survived.

| id | change | selector | result |
|---|---|---|---|
| rq_thr_sign | requant RDivPOT threshold without the sign term | test_requant[4] | killed |
| rq_rem_ge | `rem > thr` → `>=` | test_requant[4] | killed |
| srdhm_nudge | negative nudge 1-2^30 → -2^30 | test_requant[4] | killed |
| srdhm_ovf | drop INT32_MIN² → INT32_MAX | test_requant[4] | killed |
| rq_satkeep | sat flag ignores keep | test_requant[4] | killed |
| rq_o16_stage | int8/int16 mode bit from the wrong stage | test_requant[4] | killed |
| ac_gm_signed | group scale treated as signed | test_array_ctrl[4] | killed |
| ac_gsw_m1 | group-scale psum operand 1 bit narrower | test_array_ctrl[8] | killed |
| ac_gm_pop | scale FIFO popped every row | test_array_ctrl[4] | killed |
| ac_nofwd | RMW read-during-write bypass removed | test_array_ctrl[4] | killed |
| ac_wmask_le | ragged-n weight mask off by one | test_array_ctrl[4] | killed |
| as_skp_every | GEMM_GS scale row pointer advanced every tile | test_array_seq[4] | killed |
| as_raw_gate | ACC prefetch not gated on earlier done | test_array_seq[4] | killed |
| sm_causal_le | causal limit `<` → `<=` | test_softmax[4] | killed |
| sm_round | softmax final rshr half → quarter | test_softmax[4] | killed |
| sm_sat_all | score sat counted on masked lanes only | test_softmax[4] | killed |
| ln_zround | LN z rounding removed | test_layernorm[4] | killed |
| ln_noeps | eps_q dropped | test_layernorm[4] | killed |
| ln_qmax1 | `Q = max(Q,1)` removed | test_layernorm[4] | survived: **equivalent** (Q = 0 implies every kept c = 0, so z = 0 whatever RSQRT returns) |
| vs_diag | vec_seq causal limit diag+1 → diag+2 | test_vec_seq[4] | killed |
| sem_usedw | WAIT grant ignores same-cycle earlier WAITs | test_ctrl_top | **survived**; killed by test_ctrl_formal -k sem (A1-10) |
| sem_useds | SIGNAL saturation ignores same-cycle SIGNALs | test_ctrl_top | **survived**; killed by test_ctrl_formal -k sem (A1-10) |
| dma_4k_cross | burst may cross 4 KiB by one beat | test_dma_engine[16-32] | killed |
| gold_lnaff_rshr | C5 LN step 7 rdivpot → rshr | tests/py -k golden | **survived** (A1-03); killed by tests/audit |
| gold_ls_drop | golden ignores score.ls | tests/py -k golden | **survived** (A1-09); killed by tests/audit |
| gold_recip_nonewton | RECIP without the Newton step | tests/py -k golden | killed |

## 4. Checked and found sound (do not repeat)

* **C2 requant RTL** (`requant.sv`, `srdhm.sv`, `sat_counter.sv`): wrap32 bias add, SRDHM nudge and
  truncation, the INT32_MIN² case, RDivPOT thresholds, int8/int16 clamp per beat, sat counting with keep.
  Test references are `nanoxl.numerics` plus directed halves and extremes. Six mutants killed.
* **C1/C1b/C3 array** (`pe`, `systolic_array`, `array_ctrl`): exact products, INT4 sign extension,
  ragged k/n masking, bank reuse rule with sim checker, RMW bypass, credits. The GS path is
  `sc = psum[GSW-1:0] · {0,gm}` with GSW = log2N+12, exact for |psum| ≤ N·2^10. The scale FIFO pops
  per tile. A gs=1 command with int4=0 would truncate psum, but C3 forbids that combination and
  array_seq never issues it.
* **array_seq**: K-tiles split at min(N, G), scale-row pointer `w + rup(K·N/2,16) + (k0/G)·N + n0`,
  w_trans transposer, ACC RAW gate on earlier `done`, in-order dones incl. NOP for illegal opcodes,
  `clr` handling. The reference is funcsim on random buffer images, with byte-for-byte comparison.
* **softmax.sv**: masked max (an init of -32768 is equivalent), 26-bit sum, 24-bit 127·y, 40-bit
  product, causal mask `beat·L + lane < nkeep` with nkeep = 0 meaning "keep all". vec_seq never emits
  0 for a causal row because negative diag is rejected; funcsim rejects such rows too.
* **layernorm.sv** widths vs C5 (c int18, Q 48 bit, R int32, c·R 50 bit, z·g+b 34 bit), slot/lane
  g/b selection incl. the L ≥ 32 fix.
* **sem_unit** (formal: counts never negative/wrap, no lost signals, progress) and **dma_engine** 4 KiB
  split (`256 - addr[11:4]`), row-beat count width (≤ 4097 beats), DDR wrap.
* **Engine port contract** (control.md §2) vs array_seq / vec_seq: one done per command, in order,
  ≥ 1 cycle after accept, done = last write accepted, at most one per cycle (NOP and write dones are
  mutually exclusive in array_seq). Line addressing is the same on every port (17-bit line =
  byte >> 4); the alignment rule and the DMA's byte-granular superset agree.
* **ISA/encoding**: `cmd_fields.sv` generated and checked against `isa.py`; golden vectors decoded by
  RTL (test_cmd).
* **Golden model**: KV-cache decode == full recompute (test), W4A8 group sum bound < 2^31 asserted,
  saturation names match the funcsim notes. Compiled tiny W8A8/W4A8 in funcsim is bit-identical to
  golden (test_funcsim_golden), now also with `ls > 0` (tests/audit).
* **Quantization**: calibration from train only; committed bundles reproducible; headline ΔPPL
  re-derived (§2).
* **Cycle model GEMM** INT8 tiles: matches measured occupancy exactly for weight-bound tiles at N=32.
* **Driver ring logic**: free space = `(head - tail - 1) mod size` uses RING_HEAD (next to dispatch),
  which is safe against the 16-word prefetch because the fetcher never reads past tail.
* **CSR offsets**: `nanoxl/driver/csr.py` matches `csr_axil.sv` localparams for every register
  except the SAT map (A1-02).

## 5. Notes for SG7c / the next audit

* Wire `vec_seq.sat_cnt` with NSAT = 5 and align `driver/csr.py` and `SimBackend` to {requant, alu,
  ln_out, ln_z, score} (A1-02). Wire `pb_snoop_*` to every DMA write into PBUF; without it a resident
  LN/GELU/ctx block can go stale silently (sfu.md §10).
* ACC is read by array_seq in N/4-line beats and by vec_seq in 1-line beats. A write accepted on
  either port must be visible to every later-accepted read on the other port (array.md §7).
* Re-measure VEC throughput at L = 64 once SG7c fixes the port width, then update the cycle model
  (A1-01).
