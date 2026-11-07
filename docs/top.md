# NanoXL top level (`nanoxl_top`, SG7c / M5–M6)

RTL: `rtl/nanoxl_top.sv`, `rtl/buf_spad.sv`, `rtl/spad_bank.sv`. Tests: `tests/rtl/test_top.py`
(cocotb `top_cocotb.py`), `tests/rtl/test_top_spad.py` (cocotb `top_spad_cocotb.py`).
Status: the compiled `tiny` model (W8A8 and W4A8, on-chip KV and DDR KV) runs end to end through the
real host driver and is bit-exact with `nanoxl.golden` (logits every step, KV cache, per-op saturation
counts, greedy tokens = `golden_ref.json`).

## 1. Block diagram

```
 host ── AXI4-Lite s_axil ──► ctrl_top ── mf_axi (AXI4 read: command ring) ──► DDR
                               │  csr_axil, cmd_fetch, cmd_proc (3 queues, semaphores), dma_engine ×2 ── md_axi, md1_axi (AXI4) ──► DDR
                               │ arr_cmd/done/err      │ vec_cmd/done/err     │ sp_*, sp1_* (16-byte line ports, DMA ch0/ch1)
                               ▼                       ▼                      │
                          array_seq                 vec_seq                    │
                    (array_ctrl + systolic)   (vec_unit + SFU, pbuf_loader)    │
                     A  W  CR  CW               RA  RB  RP  WD   ◄─ pb_snoop ──┤ (DMA writes into PBUF)
                     └──┴──┴───┴──────────┬──────┴───┴───┴───┴─────────────────┘
                              scratchpad router (by buffer id, 8 read + 4 write ports)
                          ABUF   WBUF   ACC   PBUF   KVBUF      (buf_spad: NB line-interleaved banks each)
 irq ◄── ctrl_top (IRQ_STATUS & IRQ_EN)          eng_clr ──► array_seq.clr, vec_seq.clr     sat_clr ──► vec_seq
```

External ports: AXI4-Lite CSR slave (`s_axil_*`, docs/control.md §5), AXI4 read master `mf_axi_*`
(fetch), AXI4 masters `md_axi_*` (DMA ch0) and `md1_axi_*` (DMA ch1, SG10a / ADR-005), `irq`. All
masters are 128-bit, 32-bit address, ID 0. Board ports (ADR-005 §4): KV260 `mf` → HPC0, `md` → HP0,
`md1` → HP1; PYNQ-Z2 `md` → HP0, `md1` → HP1, `mf` shares HP0 through a 2:1 SmartConnect.

| parameter | default | meaning |
|---|---|---|
| `N`, `L` | 8, 4 | array size, vector lanes (tests: N=8/L=4 fast tier, N=16/L=16 full model) |
| `M_DEPTH` | 64 | array_ctrl accumulator rows |
| `ABUF_LINES` … `KV_LINES` | 64 / 32 / 64 / 32 / 128 KiB | buffer capacities in 16-byte lines (powers of two); line addresses wrap |
| `NB` | 0 (auto) | banks per buffer; auto = 2 × widest port, power of two (N ≤ 16: 8, N = 32: 16) |
| `VP` | 4 | VEC engine port beat in 16-byte lines (A, B, P, D; `vec_seq.PL`, SG10b); 1 = the earlier 16-byte ports, cycle-identical |
| `CTX_MAX`, `D_MAX`, `NG`, `NSLOT`, `NCTX` | 256, 384, 4, 16, 8 | vec_seq limits / parameter slots (shakespeare needs NG ≥ 6) |
| `MAXB`, `LDF` | 256, 512 | DMA burst / load FIFO (each DMA engine; `ctrl_top.NCMD` = 4 commands in flight per engine) |

The test configuration of the compiler is `top_cocotb.top_hw(N, L, kv)`: KV260 timing knobs with the
buffer capacities above.

## 2. Wiring contract

* **Engines.** `arr_*` ↔ `array_seq`, `vec_*` ↔ `vec_seq` (docs/control.md §2). `arr_err`/`vec_err`
  (sticky) → ERR 7. `eng_clr` (soft reset) drives both engines' `clr`.
* **PBUF snoop.** Every DMA line accepted into PBUF, on either channel, invalidates resident LN / GELU
  / ctx / LUT blocks that cover it (docs/sfu.md §10). The snoop port takes one line per cycle, so in a
  cycle where ch0 offers a PBUF write, ch1's PBUF write is held back (`sp1_wvalid` masked):
  `pb_snoop_valid = (dma0_pb && sp_wready) || (sp1_wvalid && sp1_wready && sp1_wbuf == PBUF)`,
  `pb_snoop_line = dma0_pb ? sp_waddr : sp1_waddr`, with `dma0_pb = sp_wvalid && sp_wbuf == PBUF`. The VEC engine snoops its own PBUF writes internally; the array never
  writes PBUF.
* **Saturation counters** (AUDIT-1 A1-02, canonical map). `vec_seq.sat_cnt` goes to `ctrl_top.sat_cnt`
  unchanged with `NSAT = 5`: SAT_0 requant, SAT_1 ALU, SAT_2 LN out, SAT_3 LN z, SAT_4 softmax score.
  `nanoxl/driver/csr.py` (`SAT_NAMES`), the SimBackend and docs/control.md §5 use the same slots.
  `sat_clr` (PERF_CTRL bit 1) clears them.

## 3. Physical scratchpads (`buf_spad`, `spad_bank`)

Each buffer is `NB` banks of 128-bit lines, **line-interleaved**: line *l* is bank *l* mod NB, row
(*l* mod LINES)/NB. A bank is a simple-dual-port RAM (one read, one byte-strobed write per cycle,
registered read), so BRAM/URAM inference is direct.

Ports (all with the DMA's line protocol: buffer id, 17-bit line address, in-order responses, no
response backpressure):

| # | read port | lines/beat | | # | write port | lines/beat |
|---|---|---|---|---|---|---|
| 0 | DMA (`sp_ar*`) | 1 | | 0 | DMA (`sp_w*`) | 1 |
| 1 | array A (ABUF) | ceil(N/16) | | 1 | array ACC (`cw_*`) | N/4 |
| 2 | array W (WBUF/KVBUF) | ceil(N/16) | | 2 | VEC D (`wd_*`) | VP (4) |
| 3 | array ACC (`cr_*`) | N/4 | | | | |
| 4, 5 | VEC A, B (any buffer) | VP (4) | | | | |
| 6 | VEC P (PBUF) | VP (4) | | 3 | DMA ch1 (`sp1_w*`) | 1 |
| 7 | DMA ch1 (`sp1_ar*`) | 1 | | | | |

* **Routing.** The router sends each request to the buffer named by its id (ids 5–7: a null buffer
  that accepts, returns zeros, drops writes; the decoders reject such ids before they get here).
* **Banking.** A beat of P lines from any start line touches P distinct banks when NB ≥ P (checked
  at elaboration), so array beats that start mid-window (ACC rows on any line, ragged weight rows)
  and VEC beats that start on any row line cost no extra cycle.
* **Bank count per VEC port width (SG10b).** Required: NB ≥ max(VP, N/4, ceil(N/16)). Built: NB = 2 ×
  that (VP = 4: 8 banks at N = 8/16, 16 at N = 32; VP = 1: 4 at N = 8, 8 at N = 16). Two P-line
  streams that both want a beat every cycle in the same buffer are conflict-free only when their start
  banks differ by at least P both ways round (mod NB); otherwise the rotating priority alternates them
  and each runs at half rate. With NB = 2P that is one offset in P; NB = 4P would make it half of all
  offsets. The common case does not need it: the full-rate VEC streams (REQUANT and softmax reading
  ACC, LN) are single-source, the D port arbitrates against the write ports only, and int8 streams at
  L ≤ 16 use a quarter of the A/B beats, which the 16-beat read FIFO absorbs. The two-source case at
  full rate is a VEC ADD of int8 at L = 64 with both sources in one buffer (the residual add, a few
  hundred beats per layer); docs/vector-unit.md §9.3.
* **Arbitration.** Per buffer and cycle, reads and writes separately: ports are visited in a rotating
  order (start advances every cycle); a request is granted when none of its banks was taken by an
  earlier port in that order. `ready` = grant (combinational on `valid`; no requester makes `valid`
  depend on `ready`). A request therefore waits fewer than 7 (read) / 3 (write) cycles behind
  conflicts; the bench asserts this bound.
* **Latency / ordering.** Reads return exactly one cycle after the grant, so responses are in request
  order per port and the requesters' credit counters (≤ 8 / 16 in flight) never overflow.
* **Visibility.** A write granted at edge *t* is visible to every read granted at an edge > *t*, on
  any port and any port width (array ACC N/4-line beats vs VEC 1-line reads hit the same banks); a
  read granted at the same edge returns the old line. This is the rule array_seq's ACC RMW
  (array.md §7), vec_seq's in-place ops and the semaphore protocol rely on.

## 4. Integration checklist (`integration-notes-for-sg7c.md`, AUDIT-1 §5)

| source | item | handled |
|---|---|---|
| SG3 | vec unit ~576 DSPs at L=64; narrow ALU MUL | deferred to SG9 synthesis / ADR-003 (no functional effect) |
| SG3 | `in_ready` comb on `out_ready` | internal to vec_seq; no top-level loop (Verilator UNOPTFLAT-free) |
| SG3 | reduce tree may need a pipe register | timing, SG9 |
| SG3/SG5 | SFU sub-op numbering, SOFTMAX_RQ = 6 | vec_seq (SG7b); exercised by every softmax of the run |
| SG4 | LN/softmax param blocks, softmax out 0..127, INT4 group scales | resolved by SG4b/SG5 (ADR-002/004); bit-exact end to end |
| SG4 | byte vs 16-byte addressing | alignment rule (ADR-004); ports are line addressed |
| SG4/SG8 | ATTN_TILE / online softmax | reserved; array_seq flags it (ERR 7). Not used by the compiler |
| SG4/SG4c/SG7a | KV260 2-row weight port, `wload_rows_per_cycle`, LM head `WPL = ceil(N/16)+1` | deferred (perf); the RTL loads one weight row per cycle, cycle model already does too (§6) |
| SG2 | `cmd_m/k/n ≥ 1` unchecked | the ISA's m1 encoding cannot express 0 |
| SG2 | `cmd_ready` comb on `w_valid`, Icarus genvar-cast miscompile | internal to array_ctrl; no new instance of the pattern in the top |
| SG2 | N=32 Verilator-only | top tests use N ≤ 16; N=32 lints clean |
| SG2/SG7a | compiler `abase`, tile time | array_seq uses abase 0 / owns tiling |
| SG4b | `cmd_gs` + scale beat, logits int16 | GEMM_GS path runs in the W4A8 tests; logits compared as int16 |
| SG6 | DMA 16-byte line port → physical buffers | §3 |
| SG6 | two AXI masters | three since SG10a: `mf_axi` (fetch) + `md_axi` / `md1_axi` (DMA ch0/ch1), all on one memory in the bench |
| SG6/SG4c | `chan` ignored, one DMA master (4 GB/s cap at KV260); DMA not pipelined | **fixed (SG10a, ADR-005)**: two DMA engines by `chan[0]` on `md_axi` / `md1_axi`, up to 4 commands in flight each, SIGNALs carried on done; §6 |
| SG6 | > 263 outstanding SIGNALs deadlock | compiler emits 1–2 (funcsim hazard check, SG4c) |
| SG5/SG7b | SFU param slots from PBUF, LUT_LOAD | pbuf_loader; every program's LUT_LOAD / LN / GELU runs |
| SG5 | LN needs L power of two | `L` is a power of two in every configuration |
| SG7a | banking by line, ACC write visible to later reads | §3 (bench: `test_top_spad`, and the end-to-end runs) |
| SG7b / A1-02 | NSAT = 5, counter order | §2 (canonical map, driver/SimBackend aligned, audit test passing) |
| SG7b / AUDIT-1 | PBUF snoop on every DMA write into PBUF | §2; mutation-checked (§5) |
| SG7b | `eng_clr` → `clr` | §2; pulsed by the driver's soft reset at every `NanoXL()` |
| SG7b / A1-01 | 16-byte VEC ports limit int32 streams at L = 16/64 | **fixed (SG10b)**: VEC A/B/P/D ports are VP = 4 lines (64 B); int32 at 16 per cycle (lane rate at L = 16, still port bound at L = 64). Prefill −10 % (W8A8) / −12 % (W4A8), §6 |
| SG7b | one VEC command at a time; KV_APPEND dh % 16 | unchanged; compiler satisfies both |
| SG7b/SG4c | shakespeare NG ≥ 6, LN params reloaded per program | `NG` is a top parameter; reload costs measured in §6 |

## 5. Verification

| test | tier | what |
|---|---|---|
| `test_top_spad` (3 configs) | fast, Icarus/Verilator | randomized multi-port traffic on `buf_spad` (beats of 1/2/4 lines at any start, wrap at the end, strobes, valid held until ready) vs a line model with write-visible-to-later-read; no two grants share a bank; starvation bound |
| `test_top_smoke` | fast (Icarus: prefill only, ~90 s; Verilator: + 1 decode step, ~10 s) | 1-layer cut of tiny W8A8, 2-token prompt, N=8, L=4, VP=4, real driver over AXI; logits, KV cache, per-op saturation counts, SAT CSRs == golden. The compiler's PBUF ring is 10 KiB so different LN blocks reuse PBUF lines |
| `test_top_smoke_saturation` | fast, Verilator | same with requant/LN shifts lowered so the requant, score, PV, residual (ALU) and LN-out clamps fire (hundreds of events), counts per op == golden |
| `test_top_tiny[w8a8/w4a8 × onchip/ddr]` | slow, Verilator (~70 s each) | full tiny, N=16, L=16: 16-token reference prompt + 8 greedy decode steps through `NanoXL.step` (ring, doorbell, IRQ wait, logits read, argmax); logits every step, KV cache (on-chip: DMA'd out by a host program; DDR: the image's KV region), per-op saturation counts == golden; tokens == `golden_ref.json` |
| `test_top_tiny_vp1[w8a8/w4a8]` | slow, Verilator | as `test_top_tiny` on-chip with VP = 1 (16-byte VEC ports): bit-exact, and its cycles equal the SG7c measurement exactly |
| `test_top_tiny_saturation` | slow, Verilator | full 4-layer W4A8 with lowered shifts, 4-token prompt + 2 steps, 30 % random stalls on all AXI channels |

**Bench.** `top_cocotb.py` runs the unmodified `nanoxl.driver.NanoXL` in a cocotb `bridge` thread; its
`RtlBackend` turns `read32` / `write32` into AXI4-Lite transactions (`resume`), `wait_irq` into a wait
on the `irq` pin (`use_irq=True`), and hands the driver a numpy view of the AxiRam memory (one 4 GiB
sparse mmap shared by `md_axi` and `mf_axi`; buffer at 0x5000_0000, image built at 0x4000_0000, so
relocation is exercised). Per-op saturation counts: a monitor snapshots `u_vec.sat_cnt` at every
VEC `done`; deltas are charged to the command's compiler note (the golden counter name) and compared
with `Golden.sat_count` (+ the driver's host-embedding `resid` count).

**Mutation checks** (each killed, then reverted):
* PBUF snoop disconnected (`pb_snoop_valid = 0`): `test_top_smoke` and `test_top_smoke_saturation` fail
  (stale LN block used after the PBUF ring wrapped). With the default 32 KiB ring the compiler never
  reuses a line for a different block in tiny, which is why the smoke config shrinks it.
* SAT_2 / SAT_3 swapped in the `sat_cnt` wiring: `test_top_smoke_saturation` fails (CSR vs per-op totals).
* `buf_spad` rotation frozen: `test_top_spad` fails (starvation bound).

## 6. Performance: measured vs `nanoxl.cyclemodel`

PERF_CYCLES per step (cycles the device was not idle), N=16, L=16, test buffers, AxiRam with no
stalls (≈ 2-cycle DDR latency), against `cyclemodel.simulate(prog.cmds, hw, tm)` for the same program,
`HwConfig` and VEC port width (`vec_port_lines` = VP). As built (VP = 4, 64-byte VEC ports, SG10b):

| bundle, KV | prefill 16 rows: measured / model | decode (pos 16..23): measured / model |
|---|---|---|
| W8A8, on-chip | 80,098 / 81,226 (−1 %) | 66,104–66,216 / 70,073–70,185 (−6 %) |
| W4A8, on-chip | 84,342 / 85,377 (−1 %) | 71,958–72,070 / 73,935–74,047 (−3 %) |
| W8A8, DDR KV | 80,271 / 86,150 (−7 %) | 66,360–66,528 / 72,189–72,637 (−8 %) |
| W4A8, DDR KV | 84,749 / 90,909 (−7 %) | 72,254–72,422 / 76,363–76,811 (−6 %) |

With 16-byte VEC ports (VP = 1, `test_top_tiny_vp1`; identical to the SG7c measurement):

| bundle, KV | prefill 16 rows: measured / model | decode (pos 16..23): measured / model |
|---|---|---|
| W8A8, on-chip, VP = 1 | 88,976 / 90,227 (−1 %) | 67,241–67,365 / 70,745–70,857 (−5 %) |
| W4A8, on-chip, VP = 1 | 96,202 / 96,900 (−1 %) | 72,815–72,939 / 74,843–74,955 (−3 %) |

**VP = 1 → 4:** prefill −10.0 % (W8A8) and −12.3 % (W4A8); decode −1.7 % / −1.2 % (M = 1 VEC work is
small next to the weight stream). VEC engine busy in the W8A8 on-chip prefill 47.9k → 26.8k cycles
(model 48.0k → 26.9k, within 0.3 %); W4A8 47.4k → 26.3k. The other engines do not move: ARRAY 56.3k
(model 51.3k: GEMM_GS / tile bubbles), DMA 56.0k (model 62.0k: the model's 300 ns DDR latency is not
exercised by the AxiRam). Prefill is now ARRAY + DMA bound at this size; the remaining VEC time
is the int32 softmax / REQUANT and the SFU per-row fixed costs (docs/vector-unit.md §9.3).

Modelled KV260 shakespeare (N = 32, L = 64, 256-token prefill, `estimate(SHAKESPEARE, KV260)`): prefill
17.6 → 14.7 ms, array utilization 0.66 → 0.78 (W8A8; W4A8 the same), decode unchanged by SG10b (332 / 594
tok/s W8A8 / W4A8, single-DMA-master bound; see below for SG10a). The prefill critical path is 77 % GEMM, 8 % softmax (int32
scores at 16 per cycle), 7 % attention GEMMs, 4 % LN.

**SG11a (shared softmax score requant, `SM_RQL = 16` at L = 64, docs/fpga.md §2.3 C6):** the SOFTMAX_ONLINE
rows at L = 64 cost 3 cycles more each at PL = 1 and 3-6 more at PL = 4 (re-measured,
`bench/results/vec_seq_perf.json`, `VEC_CAL[·][64]` refit). Modelled KV260 (current model with SG10a + SG10b):
shakespeare W8A8 prefill 14.01 → 14.17 ms (+1.2 %), decode 619.4 → 619.3 tok/s; tiny prefill 0.969 →
1.005 ms, decode 5321 → 5316 tok/s. L ≤ 16 builds are unchanged (SM_RQL = L), and so is GELU / LN timing
at every L.

### 6.1 Two DMA masters, pipelined DMA, carried SIGNALs (SG10a, ADR-005)

Same programs and tables as above, on-chip KV, VP = 4. "Before" is main with SG10b and without SG10a
(one master, one DMA command at a time, blocking SIGNALs). `TOP_AXI_LAT` = pipelined read latency in
cycles on every read master (AR → first R beat; 75 cycles = the model's 300 ns DDR latency at 250 MHz):

| bundle, AXI latency | prefill: before → after (model) | decode (pos 16..23): before → after (model) |
|---|---|---|
| W8A8, 75 cycles | 81,573 → 77,598 (−4.9 %; 79,088) | 70,783–70,895 → 63,319–63,431 (−10.5 %; 63,269–63,381) |
| W4A8, 75 cycles | 85,628 → 83,266 (−2.8 %; 83,287) | 74,400–74,512 → 69,755–69,869 (−6.2 %; 69,760–69,872) |
| W8A8, 0 (no stalls) | 80,098 → 76,394 (−4.6 %) | 66,104–66,216 → 61,923–62,035 (−6.3 %) |
| W4A8, 0 (no stalls) | 84,342 → 82,598 (−2.1 %) | 71,958–72,070 → 68,730–68,842 (−4.5 %) |

At 75 cycles the model matches the decode steps within 0.1 %. DMA busy in W8A8 decode falls
61.9k → 50.9k cycles. Tiny at N = 16 is now array bound (ARRAY busy 54–57k of 63k cycles), so the
measured gain is small. The large gain is at shakespeare scale, where decode streams 10.6 MB of INT8 weights
per token. Modelled (`estimate(SHAKESPEARE, board)`, built timing vs `PRE_SG10A`): KV260 W8A8 decode
332 → 619 tok/s (DDR weight bound 752), W4A8 594 → 651 (array bound); PYNQ-Z2 99 → 128 / 175 → 186.
KV260 prefill 14.7 → 14.0 ms. The measurement runs are `TOP_AXI_LAT=75 TOP_PERF_DIR=<dir>
SIM=verilator pytest tests/rtl/test_top.py -k "tiny and onchip and not vp1"`.

## 7. Lint

`verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDPARAM` on `nanoxl_top` (N/L = 4/4, 8/4,
16/16, 32/64): no warnings in `nanoxl_top.sv`, `buf_spad.sv`, `spad_bank.sv`. 42 warnings remain in
earlier modules (width extensions/truncations and unused bits in vec_seq, layernorm, sfu_*, vec_unit,
softmax, srdhm, vec_seq_rd/wr; `PINCONNECTEMPTY` in vec_seq). Those inspected (vec_seq record index
13→8 bits with RQ_MAX = 256 chunks, kv/requant address extensions, unused high product bits) are
intentional; none showed up as a functional difference in the end-to-end runs. They are left to their
owners (not SG7c paths).
