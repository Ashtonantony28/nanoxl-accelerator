# NanoXL top level (`nanoxl_top`, SG7c / M5–M6)

RTL: `rtl/nanoxl_top.sv`, `rtl/buf_spad.sv`, `rtl/spad_bank.sv`. Tests: `tests/rtl/test_top.py`
(cocotb `top_cocotb.py`), `tests/rtl/test_top_spad.py` (cocotb `top_spad_cocotb.py`).
Status: the compiled `tiny` model (W8A8 and W4A8, on-chip KV and DDR KV) runs end to end through the
real host driver and is bit-exact with `nanoxl.golden` (logits every step, KV cache, per-op saturation
counts, greedy tokens = `golden_ref.json`).

## 1. Block diagram

```
 host ── AXI4-Lite s_axil ──► ctrl_top ── mf_axi (AXI4 read: command ring) ──► DDR
                               │  csr_axil, cmd_fetch, cmd_proc (3 queues, semaphores), dma_engine ── md_axi (AXI4) ──► DDR
                               │ arr_cmd/done/err      │ vec_cmd/done/err     │ sp_* (16-byte line port)
                               ▼                       ▼                      │
                          array_seq                 vec_seq                    │
                    (array_ctrl + systolic)   (vec_unit + SFU, pbuf_loader)    │
                     A  W  CR  CW               RA  RB  RP  WD   ◄─ pb_snoop ──┤ (DMA writes into PBUF)
                     └──┴──┴───┴──────────┬──────┴───┴───┴───┴─────────────────┘
                              scratchpad router (by buffer id, 7 read + 3 write ports)
                          ABUF   WBUF   ACC   PBUF   KVBUF      (buf_spad: NB line-interleaved banks each)
 irq ◄── ctrl_top (IRQ_STATUS & IRQ_EN)          eng_clr ──► array_seq.clr, vec_seq.clr     sat_clr ──► vec_seq
```

External ports: AXI4-Lite CSR slave (`s_axil_*`, docs/control.md §5), AXI4 read master `mf_axi_*`
(fetch), AXI4 master `md_axi_*` (DMA), `irq`. Both masters are 128-bit, 32-bit address, ID 0.

| parameter | default | meaning |
|---|---|---|
| `N`, `L` | 8, 4 | array size, vector lanes (tests: N=8/L=4 fast tier, N=16/L=16 full model) |
| `M_DEPTH` | 64 | array_ctrl accumulator rows |
| `ABUF_LINES` … `KV_LINES` | 64 / 32 / 64 / 32 / 128 KiB | buffer capacities in 16-byte lines (powers of two); line addresses wrap |
| `NB` | 0 (auto) | banks per buffer; auto = 2 × widest port, power of two (N=16: 8) |
| `CTX_MAX`, `D_MAX`, `NG`, `NSLOT`, `NCTX` | 256, 384, 4, 16, 8 | vec_seq limits / parameter slots (shakespeare needs NG ≥ 6) |
| `MAXB`, `LDF` | 256, 512 | DMA burst / load FIFO |

The test configuration of the compiler is `top_cocotb.top_hw(N, L, kv)`: KV260 timing knobs with the
buffer capacities above.

## 2. Wiring contract

* **Engines.** `arr_*` ↔ `array_seq`, `vec_*` ↔ `vec_seq` (docs/control.md §2). `arr_err`/`vec_err`
  (sticky) → ERR 7. `eng_clr` (soft reset) drives both engines' `clr`.
* **PBUF snoop.** `pb_snoop_valid = sp_wvalid && sp_wready && sp_wbuf == PBUF`, `pb_snoop_line =
  sp_waddr`: every DMA line accepted into PBUF invalidates resident LN / GELU / ctx / LUT blocks that
  cover it (docs/sfu.md §10). The VEC engine snoops its own PBUF writes internally; the array never
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
| 2 | array W (WBUF/KVBUF) | ceil(N/16) | | 2 | VEC D (`wd_*`) | 1 |
| 3 | array ACC (`cr_*`) | N/4 | | | | |
| 4, 5 | VEC A, B (any buffer) | 1 | | | | |
| 6 | VEC P (PBUF) | 1 | | | | |

* **Routing.** The router sends each request to the buffer named by its id (ids 5–7: a null buffer
  that accepts, returns zeros, drops writes; the decoders reject such ids before they get here).
* **Banking.** A beat of P lines from any start line touches P distinct banks when NB ≥ P (checked
  at elaboration), so array beats that start mid-window (ACC rows on any line, ragged weight rows)
  cost no extra cycle.
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
| SG6 | two AXI masters | `mf_axi` (fetch) + `md_axi` (DMA), both on the AxiRam in the bench |
| SG6/SG4c | `chan` ignored, one DMA master (4 GB/s cap at KV260); DMA not pipelined | deferred (perf, SG10); measured costs in §6 |
| SG6 | > 263 outstanding SIGNALs deadlock | compiler emits 1–2 (funcsim hazard check, SG4c) |
| SG5/SG7b | SFU param slots from PBUF, LUT_LOAD | pbuf_loader; every program's LUT_LOAD / LN / GELU runs |
| SG5 | LN needs L power of two | `L` is a power of two in every configuration |
| SG7a | banking by line, ACC write visible to later reads | §3 (bench: `test_top_spad`, and the end-to-end runs) |
| SG7b / A1-02 | NSAT = 5, counter order | §2 (canonical map, driver/SimBackend aligned, audit test passing) |
| SG7b / AUDIT-1 | PBUF snoop on every DMA write into PBUF | §2; mutation-checked (§5) |
| SG7b | `eng_clr` → `clr` | §2; pulsed by the driver's soft reset at every `NanoXL()` |
| SG7b / A1-01 | 16-byte VEC ports limit int32 streams at L = 16/64 | **deferred**: ports stay 16 bytes. Measured cost at L=16 in §6 (VEC busy 2× the model in prefill). Widening `vec_seq_rd/wr` to a parameterized line count (and giving the VEC ports `L·4/16` lines) is the follow-up; `buf_spad` already supports multi-line beats per port |
| SG7b | one VEC command at a time; KV_APPEND dh % 16 | unchanged; compiler satisfies both |
| SG7b/SG4c | shakespeare NG ≥ 6, LN params reloaded per program | `NG` is a top parameter; reload costs measured in §6 |

## 5. Verification

| test | tier | what |
|---|---|---|
| `test_top_spad` (3 configs) | fast, Icarus/Verilator | randomized multi-port traffic on `buf_spad` (beats of 1/2/4 lines at any start, wrap at the end, strobes, valid held until ready) vs a line model with write-visible-to-later-read; no two grants share a bank; starvation bound |
| `test_top_smoke` | fast (Icarus: prefill only, ~90 s; Verilator: + 1 decode step, ~10 s) | 1-layer cut of tiny W8A8, 2-token prompt, N=8, L=4, real driver over AXI; logits, KV cache, per-op saturation counts, SAT CSRs == golden. The compiler's PBUF ring is 10 KiB so different LN blocks reuse PBUF lines |
| `test_top_smoke_saturation` | fast, Verilator | same with requant/LN shifts lowered so the requant, score, PV, residual (ALU) and LN-out clamps fire (hundreds of events), counts per op == golden |
| `test_top_tiny[w8a8/w4a8 × onchip/ddr]` | slow, Verilator (~70 s each) | full tiny, N=16, L=16: 16-token reference prompt + 8 greedy decode steps through `NanoXL.step` (ring, doorbell, IRQ wait, logits read, argmax); logits every step, KV cache (on-chip: DMA'd out by a host program; DDR: the image's KV region), per-op saturation counts == golden; tokens == `golden_ref.json` |
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
stalls (≈ 2-cycle DDR latency), against `cyclemodel.simulate(prog.cmds, hw)` for the same program and
`HwConfig`:

| bundle, KV | prefill 16 rows: measured / model | decode (pos 16..23): measured / model |
|---|---|---|
| W8A8, on-chip | 88,976 / 78,492 (+13 %) | 67,241–67,365 / 68,707–68,819 (−2 %) |
| W4A8, on-chip | 96,202 / 80,319 (**+20 %**) | 72,815–72,939 / 70,021–70,133 (+4 %) |
| W8A8, DDR KV | 89,137 / 83,388 (+7 %) | 67,493–67,673 / 71,347–71,795 (−5 %) |
| W4A8, DDR KV | 96,570 / 85,823 (+13 %) | 73,107–73,287 / 72,973–73,421 (0 %) |

Engine busy cycles (W8A8 on-chip prefill, measured / model): DMA 56.0k / 62.0k, ARRAY 56.3k / 50.8k,
VEC 47.9k / 23.7k. Decode is DMA-bound (all weights, ~800 KB W8A8 / ~410 KB W4A8, stream through one
128-bit master) and the array overlaps it, so the model is within 5 %.

**The gap over 15 % (W4A8 prefill, +20 %)** is the VEC engine. Its scratchpad ports move one 16-byte
line per cycle, so at L = 16 the int32 streams (REQUANT and softmax reading ACC) deliver 4 elements
per cycle, and REQUANT also reads one 16-byte record per output channel; the model prices VEC work at
L elements per cycle (AUDIT-1 A1-01, being fixed in the cycle model by another worker). In prefill
(M = 16) the REQUANT → next GEMM chain puts this on the critical path; W4A8 suffers more because its
DMA time is half of W8A8's, so less of the VEC time is hidden. The second part is GEMM_GS tiles
(one scale beat per K-tile, ARRAY busy 58.7k vs 51.6k). In decode (M = 1) both effects are small
next to the weight stream. The model's DDR latency (300 ns) is not exercised by the AxiRam, which is
why measured DMA busy is ~10 % below the model.

## 7. Lint

`verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDPARAM` on `nanoxl_top` (N/L = 4/4, 8/4,
16/16, 32/64): no warnings in `nanoxl_top.sv`, `buf_spad.sv`, `spad_bank.sv`. 42 warnings remain in
earlier modules (width extensions/truncations and unused bits in vec_seq, layernorm, sfu_*, vec_unit,
softmax, srdhm, vec_seq_rd/wr; `PINCONNECTEMPTY` in vec_seq). Those inspected (vec_seq record index
13→8 bits with RQ_MAX = 256 chunks, kv/requant address extensions, unused high product bits) are
intentional; none showed up as a functional difference in the end-to-end runs. They are left to their
owners (not SG7c paths).
