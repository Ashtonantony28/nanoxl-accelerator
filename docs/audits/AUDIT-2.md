# AUDIT-2: NanoXL work merged on main since AUDIT-1 (waves 4-6)

Scope: `main` at `eb1269f`. Covered: SG7c top level (`nanoxl_top.sv`, `buf_spad.sv`, `spad_bank.sv`);
SG10a dual DMA and pipelining (`dma_engine`, `cmd_queue`, `cmd_proc`, `ctrl_top`, ADR-005, contract C6,
the channel rules in `funcsim.check_hazards`, compiler channel assignment); SG10b wide VEC ports;
SG9a `nanoxl_axi_wrap.v` and the driver against the CSR map; SG11a-c area changes (`lut_rom`, `sfu_elem`
EL, softmax RQL, the `buf_spad` static-priority crossbar and OREG, the `vec_seq_rd` barrel unpack, the
multi-cycle `vec_alu` MUL, `pe_pair` WP486 packing, `srdhm`, pipelined `vec_reduce`); and the cycle-model
calibration in docs/top.md and docs/vector-unit.md. Not covered: `fpga/`, `bench/results/resources_*.json`,
`docs/fpga.md` and `nanoxl/hwconfig.py`, because SG12 is changing them in parallel. Items AUDIT-1 §4 found
sound are not repeated here.

This is an independent audit, and nothing was fixed. The repro tests are in `tests/audit/test_audit2.py`.
Failing findings are `xfail(strict=True)` tests whose reason names the finding id. The same file holds
the mutation list (`MUTS`) and a runner (`python3 tests/audit/test_audit2.py <scratch copy> [ids]`).

**Result: 0 critical, 0 high, 2 medium, 5 low.** The audit found no RTL correctness bug. Every
arithmetic identity behind the SG11 changes holds exhaustively or on random plus extreme inputs (§2). The
DMA channel ordering in RTL agrees with `check_hazards` (§4). Of 41 targeted mutants, 34 are killed by
the CI tier. Of the other 7, 2 are equivalent, 2 are killed only by tests that CI never runs, and 3 are
killed by no test at all (§3). Both medium findings are about verification: CI does not run what the
docs say it runs.

Baseline runs on this checkout (container, 4 cores):

| command | result |
|---|---|
| `python3 -m pytest -q tests/py -m "not slow"` | 218 passed, 3 deselected, 200 s |
| targeted RTL subset, Icarus (pe, top_spad, dma, ctrl, vec_seq[4-*], vec_reduce, vec_alu, softmax_rq_shared) | 27 passed, 372 s |
| `python3 -m pytest -q tests/rtl/test_ctrl_formal.py -k queue` | 1 passed, 8 s |
| `TOP_PERF_DIR=... SIM=verilator pytest tests/rtl/test_top.py -k "tiny and onchip and w8a8 and not vp1 and not oreg"` | 1 passed, 195 s; bit-exact; cycles in A2-07 |
| `SIM=verilator pytest "tests/rtl/test_vec_seq.py::test_vec_seq[16-1]"` | 1 passed |
| `python3 -m pytest -q tests/audit -rxX` | 28 passed, 6 xfailed, 45 s (AUDIT-1 + AUDIT-2) |

## 1. Findings (ranked)

### A2-01 (medium): the CI tier never runs an L ≥ 16 VEC engine, the top-level SAT/OREG tests or a top-level decode step
* **Where:** `.github/workflows/ci.yml` (`make test-fast SIM=icarus`); `tests/rtl/test_vec_seq.py:17-25`
  (L = 16 is `slow` on Icarus); `tests/rtl/test_top.py:39-55` (`test_top_smoke_saturation` and
  `test_top_smoke_oreg` say "Fast tier" in their docstrings but are `VERILATOR_ONLY`); `test_top.py:30`
  (on Icarus `test_top_smoke` runs `TOP_DECODE=0`); docs/top.md §5 lists `test_top_smoke_saturation`
  as "fast".
* **What CI therefore never exercises:**
  - `vec_seq_rd`'s multi-chunk beat assembly (4L > 16P: `lpb_l2 > 0`, k > 0, clearing of the other
    chunks) and `vec_seq_wr`'s multi-chunk wide mode (`hm`/`hn` > 0). At L = 4 the assembly buffer
    always holds one chunk.
  - The integrated shared SFU lanes (`sfu.sv` uses EL = min(L, 8)) and the shared softmax requant
    (SM_RQL = min(L, 16)). At L = 4 both equal L. Only the module tests cover the shared modes.
  - The SAT_0..4 CSR wiring (the A1-02 fix) checked against non-zero counts. Without stress the
    smoke-test counters stay at 0.
  - OREG = 1, RAM_STYLE and per-buffer NB at top level.
  - Any decode step through the RTL, and the DDR-KV mode.
* **Concrete failing scenarios (mutation repro):**
  - `rd_full_k1` makes `vec_seq_rd` treat every chunk as a complete beat, so every L ≥ 16 int32 row is
    garbled. It passes `test_vec_seq[4-1]` and `[4-4]` on Icarus (152 s) and is killed by
    `SIM=verilator test_vec_seq[16-1]`.
  - `top_sat_swap` swaps SAT_2 and SAT_3 in `nanoxl_top` (the A1-02 bug again). It passes
    `test_top_smoke` on Icarus and is killed by `SIM=verilator test_top_smoke_saturation` (38 s).
    The docs/top.md §5 mutation note says this test kills it, but CI never runs that test.
* **Repro:** `python3 -m pytest -q tests/audit/test_audit2.py -k a2_01 -rxX` (2 xfails). Mutants:
  `python3 tests/audit/test_audit2.py <copy> rd_full_k1 top_sat_swap`. The Verilator kills come from
  the `SIM=verilator SEL=...` variants shown in the file header.
* **Note:** `test_vec_seq[16-*]` takes about 20 s per config on Verilator. A Verilator CI job running
  today's Verilator fast tier would close most of this finding cheaply.

### A2-02 (medium): no CI job runs Verilator or the slow tier, so the headline bit-exact test is never automated
* **Where:** `.github/workflows/ci.yml` has one job: Icarus fast tier plus `make test-audit`. That
  contradicts docs/design-plan.md:35 and :186 ("full cocotb suite in CI on Verilator ... integration
  tiers nightly").
* **Scenario:** `test_top_tiny[*]` is the only full-model RTL check (tiny W8A8/W4A8 × on-chip/DDR KV,
  16-token prefill + 8 decode steps, bit-exact against golden), and it is `slow` + Verilator-only. The
  same holds for `test_top_tiny_saturation` (AXI backpressure on both masters), `test_vec_seq_l64`
  (the only L = 64 run, i.e. KV260 lanes: NTS = 3 reduce tree, EL = 8, SM_RQL = 16, AL > 1 at P = 4),
  `test_array_ctrl_kv260_size` (N = 32) and `test_ctrl_top_long`. The wave-6 rule ("the top-level tiny
  bit-exact tests must pass at the end") is enforced only by hand. A merge that breaks the full model
  but keeps the 1-layer prefill smoke test green goes unnoticed.
* **Repro:** `python3 -m pytest -q tests/audit/test_audit2.py -k a2_02 -rxX` (xfail).

### A2-03 (low): DMA channel 1 writes into PBUF (residency snoop and ch0 priority gate) are never exercised
* **Where:** `rtl/nanoxl_top.sv:293-295`:
  `sp1_wvalid = sp1_wv_raw && !(dma0_pb && sp1_wbuf == 3'd3)` and the second term of `pb_snoop_valid`.
* **Scenario:** the compiler loads every parameter block on chan 0. In compiled tiny and shakespeare
  (KV260, prefill and decode) there are 35 and 51 PBUF loads, none on ch1. Two mutants survive every
  smoke test on Icarus and Verilator (`test_top_smoke`, `_saturation`, `_oreg`):
  - `top_snoop_ch1` drops the ch1 snoop.
  - `top_ch1_gate` lets ch1 write PBUF in the same cycle as ch0, so one write goes unsnooped.

  The ISA allows `chan = 1` for a PBUF load, and `SyncBuilder` may move a LOAD to the other channel
  (compiler.py:41). Such a program would then depend on logic that has never been simulated. By
  reading, the logic is correct: the gate makes the single snoop port see one PBUF write per cycle, and
  the snoop line follows the granted channel.
* **Repro:** `python3 tests/audit/test_audit2.py <copy> top_snoop_ch1 top_ch1_gate` (SURVIVED).
  Closing it takes a top-level directed test with a PBUF load on ch1 that overwrites a resident LN
  block.

### A2-04 (low): `vec_reduce` at L = 32/64 (three tree stages) has no module test
* **Where:** `tests/rtl/test_vec_reduce.py` (L = 1, 3, 4, 8, 16). The NTS formula at
  `rtl/vec_reduce.sv:44`.
* **Scenario:** mutant `vr_nts32` sets NTS = 2 at L = 32, which misaligns valid against data. It
  survives every test. L = 64 is reached only through `test_vec_seq_l64` (slow, Verilator), and
  `vec_seq` leaves the reduce port unconnected, so even that test does not exercise it. Impact today is
  nil, because REDUCE is not in the ISA. `test_a2_vec_reduce_stage_alignment_all_L` re-derives the
  alignment for every L ≤ 64 from the RTL formulas, and it passes.
* **Repro:** `python3 -m pytest -q tests/audit/test_audit2.py -k a2_04 -rxX` (xfail);
  `python3 tests/audit/test_audit2.py <copy> vr_nts32`.

### A2-05 (low, latent): `vec_seq_wr` wide mode assumes keep is a prefix of the beat
* **Where:** `rtl/vec_seq_wr.sv:116-120,141-162`. For a wide beat (≥ 16P bytes) the writer emits
  chunks 0..`hn` only, where `hn` is the last chunk that holds a kept lane. The walker steps once per
  emitted chunk.
* **Scenario:** a non-final beat whose last chunk(s) have no kept lane emits fewer chunks than the
  walker's line count for that row. Every later chunk of the row set is then written one beat too early,
  to the wrong addresses. Today every producer's keep is "the first `len` lanes", and only the final beat
  is ragged, so nothing triggers it. A future op with interior masking (for example a keep-masked
  softmax output written in place) would.
* **Repro:** inferred from reading. No test covers it.

### A2-06 (low): stale or contradictory documentation
* `PLAN.md:25`: the SG11c status is still "wave 6", and the "Status at pause" section (lines 29-33)
  calls SG11b/SG11c unmerged WIP. Both are merged (`26aa46a`, `3792b5d`). ADR-006 is listed as
  pending, yet `docs/array.md:201` and `docs/vector-unit.md:57` already cite ADR-006 as the reason for
  PACK/NDSP choices. No ADR-006 exists.
* `docs/top.md` §6: the headline table ("As built") gives W8A8 prefill 80,098 and decode 66,104. Those
  are the pre-SG10a numbers. The built design measures 76,394 / 61,923-62,035 (§6.1's "after" row, and
  this audit's re-run).
* `docs/top.md` §5: "`test_top_spad` (3 configs)" (there are 10, including OREG). `test_top_smoke_oreg`
  is missing from the table. `test_top_smoke_saturation` is listed as "fast", but CI never runs it
  (A2-01).
* `rtl/systolic_array.sv` header: "PE(r,c) is written at the end of cycle W + c". With PACK = 1 (the
  default), column 2p+1 is written at W + 2p. The inline comment at the write skew states this
  correctly, and the bank rules are unchanged (re-derived, §4).
* **Repro:** `python3 -m pytest -q tests/audit/test_audit2.py -k a2_06 -rxX` (2 xfails, PLAN and
  top.md §6). The rest was found by reading.

### A2-07 (low): cycle-model calibration is current at N = 16/L = 16, but the L = 64 VEC constants predate SG11b
* **Measured now** (Verilator, W8A8 on-chip, VP = 4, AxiRam without stalls):
  - Prefill 76,394 cycles. The model gives 79,088 (+3.5 %).
  - Decode 61,923-62,035. The model gives 63,269-63,381 (+2.2 %).

  These cycle counts are identical to docs/top.md §6.1's SG10a "after" row. SG11a-c therefore did not
  move the end-to-end timing at this size. The model's pessimism comes from its fixed 300 ns DDR
  latency. At 75 cycles of AXI latency, docs/top.md reports the model within 0.1 % on decode, which is
  consistent.
* **Gap:** `VEC_CAL[·][64]` (`nanoxl/cyclemodel.py:97-109`) and `bench/results/vec_seq_perf.json`
  were last measured at SG11a (`1c3ec73`), before the SG11b barrel unpack and static crossbar. By
  reading, neither changes the per-beat handshake. SG11c's reduce and MUL changes touch ops the compiler
  never emits (vector-unit.md §4 says so too), so the L = 64 constants are probably still right. This
  audit did not re-measure L = 64.
* **Repro:** the `TOP_PERF_DIR` command in the baseline table. Compare `steps[].cycles` with `model`.

## 2. Arithmetic re-derived (passing tests in `test_audit2.py`)

| claim | check | result |
|---|---|---|
| WP486 `pe_pair`: lo = P[17:0] = a·wB, hi = P[35:18] + P[17] = a·wA | bit-exact numpy model of the 36-bit word over **all 2^24** (a, wA, wB), incl. −128·−128 pairs; without the +P[17] term exactly the cases a·wB < 0 break | holds |
| `srdhm` stage 3: trunc((p+nudge)/2^31) = ((p>>>30)+1)>>>1, low 32 bits | 400 k random + 49 extreme pairs vs `numerics.srdhm` | holds |
| `vec_alu` multi-cycle MUL: shift with round/sticky bits, clamp 34 bits, + rnd | vs `vec_ref.alu(MUL)` for every shamt 0..31 × fmt, 60 k pairs incl. small products (exact halves of both signs) and INT32_MIN | holds |
| `vec_reduce` NTS = registered cut levels + 1 | every L in 1..64 | holds |
| `vec_seq_rd` slice index su = sub << ebl < NU | every L ∈ {1..64}, P ∈ {1, 2, 4}, ebl, sub | holds |
| CSR map: `driver/csr.py` == `csr_axil.sv` (26 registers), NSAT = 5, STATUS bits, error codes | parsed from RTL | holds |
| ADR-005 §5: compiled tiny/shakespeare need no moved LOAD and no DMA→DMA semaphore; both channels carry loads | KV260, prefill + decode | holds |

## 3. Mutation results

`python3 tests/audit/test_audit2.py <scratch copy> [ids]` (each mutant is applied, its selector is run
with `-m "not slow"`, and the source is restored). Default simulator: Icarus, i.e. what CI runs. For
the four `cq_*` mutants, both the formal job (`queue`, in CI with `NANOXL_REQUIRE_FORMAL=1`) and
`test_ctrl_top` alone (`SEL=tests/rtl/test_ctrl.py::test_ctrl_top`) kill them. Unmutated baselines pass.

| id | change | result (CI tier) |
|---|---|---|
| pp_nocorr | pe_pair hi without the + P[17] borrow correction | killed (test_pe_pair_packing, Icarus 1/13 stride) |
| pp_lo_zext | low product zero-extended into psum_b | killed |
| pp_wb_zext | wB zero-extended in the pre-adder | killed |
| pp_hi_narrow | hi sign-extended from bit 16 | survived: **equivalent** (\|a·wA\| ≤ 2^14 fits 17 bits) |
| cq_fence_noall | DMA_STORE issues with other commands outstanding | killed (formal queue; test_ctrl_top) |
| cq_behind_fence | commands issue while a STORE is outstanding | killed (formal; test_ctrl_top) |
| cq_sig_ch0 | carried SIGNAL waits for channel 0 only | killed (formal; test_ctrl_top) |
| cq_byp_noidle | SIGNAL bypass ignores outstanding commands | killed (formal; test_ctrl_top) |
| cq_left_nodn | pending count ignores a same-cycle done | killed (formal; test_ctrl_top) |
| proc_fence_op | fence keyed on the wrong opcode | killed (test_ctrl_top) |
| proc_ch0 | every DMA command on channel 0 | killed (test_ctrl_top) |
| dma_st_behind | engine accepts a LOAD behind a STORE | killed (test_dma / test_ctrl_top) |
| bs_norot / bs_wnorot | read / write rotation frozen | killed (starvation bound) |
| bs_row_le | wrapped-beat row select `<` → `<=` | killed |
| bs_oreg_sh | OREG response rotation from the grant-cycle register | killed (OREG configs) |
| bs_wconf | write conflict check on bank 0 only | killed |
| rd_su_noebl | barrel shift ignores element size | killed |
| rd_full_k1 | every chunk completes a beat | **survived** in CI; killed by Verilator test_vec_seq[16-1] (A2-01) |
| rd_lastslice | ragged final beat not closed | killed (hang → timeout) |
| rd_shr_half | barrel unit = L/2 bytes | killed |
| lr_noen | lut_rom ignores the read enable | killed |
| lr_endian | entries of a line reversed | killed |
| rq_oc0 | RQL reassembly fires on the first sub-beat | killed |
| rq_h_nols | held sub-beats skip the `<< ls` pre-shift | killed |
| rq_h_keep | held sub-beats keep all lanes | killed |
| rq_hbias_live | held sub-beats use the live slot's bias | killed |
| vr_v_early / vr_last_early | valid / last taken from the first tree stage | killed (L = 8, 16) |
| vr_nts32 | NTS = 2 at L = 32 | **survived** everywhere (A2-04) |
| va_rnd_nosticky | MUL rounds negative halves up | killed |
| va_push_noadv | MUL result pushed under backpressure | killed |
| va_mcl33 | MUL clamp test at 33 instead of 34 bits | survived: **equivalent** (anything ≥ 2^31 saturates either way) |
| hz_store_after / _before | STORE not a fence after / before | killed (after: only by the full tests/py run, not test_dma_channels) |
| hz_sig_own / hz_wait_ch0 / hz_onechan | DMA SIGNAL covers one channel / WAIT holds one channel / channels merged | killed (test_dma_channels) |
| top_sat_swap | SAT_2 ↔ SAT_3 in nanoxl_top | **survived** in CI (Icarus test_top_smoke); killed by Verilator test_top_smoke_saturation (A2-01) |
| top_snoop_ch1 / top_ch1_gate | ch1 PBUF snoop dropped / ch1 not gated | **survived** every smoke test incl. Verilator (A2-03) |

Totals: 41 mutants. CI kills 34. Of the 7 survivors, 2 are equivalent, 2 are killed only outside CI,
and 3 are killed by no test.

## 4. Checked and found sound

* **Cross-channel DMA ordering vs `check_hazards`.** The question was whether a program could pass
  `check_hazards` and still be ordered differently by the RTL. It cannot.
  - The checker gives each DMA channel its own vector clock and assigns channel `chan & 1`, which is
    also what `cmd_proc` uses (word bit 6).
  - A DMA SIGNAL takes the max over both channel clocks. RTL: the pending entry records `outst[c] - dn[c]`
    for both channels, and the bypass fires only when all are zero. The `queue` formal job and four
    killed mutants confirm this.
  - A DMA WAIT updates both channel clocks. RTL: a WAIT at the queue head blocks issue on every channel.
  - `DMA_STORE` gets the clock join before and a copy after. RTL: `in_fence` on opcode 0x09 issues only
    when `all_zero`, and nothing issues while `fence_out` is set. `dma_engine` independently refuses to
    take a command behind a STORE.
  - END/IRQ: the RTL barrier waits for `&q_idle`, which includes pending SIGNALs.
  - Done = the last scratchpad write was accepted, and `buf_spad` makes accepted writes visible to any
    read granted at a later edge, so a SIGNAL never overtakes its data.
  - LOAD↔LOAD on different channels are the only unordered pairs: STOREs fence, and loads do not read
    the scratchpad. The checker rejects them (byte granularity is conservative at 16 B).
  - AXI uses one ID per master, so R data arrives in order.
  - Engines see a command only after every WAIT ahead of it has been granted, so array/VEC prefetch and
    PBUF residency loads cannot run ahead of a semaphore.
* **PBUF residency:** both snoop inputs exist (the DMA line and the VEC engine's own P-line D writes).
  The array never writes PBUF. Tags drop on clear. A snoop that coincides with a tag fill can only
  happen in a program the checker rejects.
* **`buf_spad` / `spad_bank`.** The static priority chain per rotation grants a port at the latest when
  the rotation reaches it, which bounds starvation at NR/NW cycles; the test asserts this bound. The
  one-hot bank ownership, wrapped-beat rows (`b < s → r0+1`) and the barrel rotations are correct.
  With OREG, `rd2 <= rd1` is unconditional and both `dv1` and `ds1` are delayed in step. A second read
  of the same bank on the next cycle cannot corrupt the first response. Read-first semantics hold with
  OREG. The null buffer matches the 1 + OREG latency.
* **`nanoxl_top` router and snoop:** the ch1 PBUF gate depends only on ch0's valid, never on ready, so
  there is no combinational loop and no deadlock. The snoop line follows the accepted channel. NSAT = 5
  is wired straight through.
* **`pe_pair` timing and bank rules:** column c+1's product uses the weight selected at T and is added
  at T+1, where its psum input arrives. Its write is advanced by one cycle together with its read, so
  the reuse rule W ≥ T_last + r and the use rule T + r ≥ W_r + 1 are those of the unpacked array. The
  simulation checker models this through `act_gen`. INT4 weights are sign-extended to 8 bits before the
  pair.
* **`srdhm` NDSP = 3:** the operand split (aL 26u, aH 6s, bL 17u, bH 15s), the partial-product widths,
  the fabric `aH·bH` (fits 21 bits) and the stage alignment of `ovf1/ovf2` against `p1/p2` are correct.
* **`vec_alu` MUL:** the sequencer (ms 1-5 then shift), `mcnt` and `mP` use the same old count, so
  exactly shamt shifts happen. Results enter S1 only with `adv`, `in_ready` stays low while `ms != 0`,
  and engine clear resets `ms` through `vec_seq`'s `rst_i`.
* **`vec_seq_rd` barrel unpack:** rows start at chunk byte 0 (line-aligned rows, beat from the row's
  first line), so the bank start offset is handled entirely by `buf_spad`. Every start line × element
  size × ragged end reduces to (sub, ebl, beats_left). The index identity holds (§2), and the
  ragged-end mutants are killed.
* **`lut_rom`:** asynchronous line read plus registered entry select. A same-edge write returns the old
  entry, as documented.
* **softmax RQL:** sub-beats leave requant on consecutive cycles (out_ready = 1), `oc` counts in groups
  of R, held lanes use the ls-shifted accumulator, the latched record and keep, and clamps are counted
  per sub-beat with keep.
* **`dma_engine` pipelining:** `aq`/`rq` descriptor FIFOs stay in order, the load FIFO is reserved at
  AR issue, `axi_idle` covers every outstanding item (AR/AW/W/B and scratchpad reads) for the
  soft-reset drain, and the 4 KiB split is unchanged.
* **Driver/CSR:** offsets, STATUS bits, error codes and the SAT order match the RTL (§2). PERF_BYTES sums
  both channels.
* **`nanoxl_axi_wrap.v`:** a pure pass-through with the same parameter set and defaults as `nanoxl_top`.
  Vivado parameterization was not audited (SG12).
* **ADR-005 / C6 text vs RTL and compiler:** consistent, including "no LOAD needs to move" (§2) and K on
  ch0 / V on ch1.

## 5. Notes for the next audit / owners

* Add a Verilator CI job running `-m "not slow"` (vec_seq L = 16, top smoke saturation/OREG, array
  N = 16, pe_pair exhaustive), plus a nightly slow job (`test_top_tiny*`, `test_vec_seq_l64`).
* Add a directed top test with a ch1 PBUF load over a resident block (A2-03). Add `test_vec_reduce[32]`
  and `[64]` (A2-04).
* Re-measure `vec_seq_perf.json` at L = 64 after SG11b to retire A2-07.
