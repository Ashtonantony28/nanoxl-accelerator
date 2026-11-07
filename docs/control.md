# Control plane: command processor, DMA, AXI, CSRs (SG6 / M4)

RTL top: `rtl/ctrl_top.sv`. ISA: `docs/isa.md` (contract C4). Tests: `tests/rtl/test_{cmd,cmd_isagen,dma,ctrl,ctrl_formal}.py`
(cocotb modules `*_cocotb.py`). Formal: `formal/` (yosys + yosys-smtbmc + z3).

```
             AXI4-Lite                         AXI4 (read)                     AXI4 (r/w)
 host ──► csr_axil ──ring base/size/tail──► cmd_fetch ──words──► cmd_proc ─┬─► dma_engine u_dma0 ◄──► DDR (md_axi)
             ▲  ▲                         (cmd_ring_ptr,       (cmd_decode,│      └─► scratchpad line port sp_*
             │  └── status / perf ──────── prefetch FIFO)       3 x cmd_queue,└─► dma_engine u_dma1 ◄──► DDR (md1_axi)
             │                                                  sem_unit)            └─► scratchpad line port sp1_*
             └── irq                                              │ arr_cmd_* / vec_cmd_*
                                                                  ▼
                                                  ARRAY / VEC engines (SG7 sequencers; stubs in tests)
```

SG10a (ADR-005): the DMA queue dispatches to **two DMA engines**, channel = the command's `chan[0]`,
each with its own AXI4 master; DMA LOADs are **pipelined** inside an engine; SIGNALs of every queue
are **carried on done** instead of blocking their queue; DMA_STORE is a **fence** across both channels.

| file | module | what |
|---|---|---|
| `fifo_sync.sv` | `fifo_sync` | generic FWFT FIFO (power-of-two depth), formal-proved |
| `cmd_fields.sv` | `cmd_fields` | **generated** from `nanoxl/isa.py`: every field of every opcode, `known`, `resv_ok`, `fixed_q` |
| `cmd_decode.sv` | `cmd_decode` | legality (== `isa.decode`), route, kind, semaphore fields |
| `cmd_ring_ptr.sv` | `cmd_ring_ptr` | head / fetch pointer / occupancy, burst choice; formal-proved |
| `cmd_fetch.sv` | `cmd_fetch` | AXI4 read master over the ring, 16-word prefetch FIFO |
| `cmd_queue.sv` | `cmd_queue` | one in-order queue: engine issue on 1 or 2 channels, STORE fence, WAIT, carried SIGNALs; formal-proved |
| `sem_unit.sv` | `sem_unit` | 32 x 8-bit counting semaphores; formal-proved |
| `cmd_proc.sv` | `cmd_proc` | dispatch, barriers, error halt, single step, perf events |
| `dma_realign.sv`, `dma_engine.sv` | | DMA_LOAD / DMA_STORE, AXI4 master, byte realigner, LOAD pipelining (two instances) |
| `csr_axil.sv` | `csr_axil` | registers over AXI4-Lite |

`cmd_fields.sv` is generated: `python3 tests/rtl/test_cmd_isagen.py`. `test_cmd_isagen` fails while it is
stale, and the decoder tests derive every expectation from `nanoxl.isa` at test time, so an ISA change
shows up as a failing test until the RTL is regenerated (GEMM_GS / LUT_LOAD from ADR-004 went in this way).

## 1. Execution model (as implemented)

* **Fetch.** Words in `[head, tail)` of the ring are fetched in INCR bursts of up to 8 beats. A burst
  never wraps the ring end and never crosses 4 KiB. One burst is in flight, and prefetch space is
  reserved before AR is issued, so RREADY stays high.
* **Dispatch** takes one word per cycle, in stream order. Engine commands, WAITs and SIGNALs go to their
  queue (depth `QDEPTH` = 8). A full queue blocks the dispatcher (in-order dispatch).
* **Queues.** The head entry is one of:
  * an engine command: issued on the engine port; up to `MAXOUT` = 4 can be outstanding at the engine
    (per DMA channel). DMA commands go to DMA engine `chan & 1` (channel 0: `md_axi`/`sp_*`, channel 1:
    `md1_axi`/`sp1_*`). Each channel completes in order; the two channels complete independently.
    A `DMA_STORE` is a fence: it issues only when nothing is outstanding on either channel, and nothing
    issues while it is outstanding;
  * `WAIT(s, k)`: blocks until the semaphore holds at least k, then subtracts k. A DMA WAIT holds both
    channels (the queue issues in order);
  * `SIGNAL(s)`: increments s once every earlier command of the queue, on every channel, has
    completed. It is **carried on done**: it leaves the queue head at once into a 4-entry pending
    buffer that counts, per channel, the earlier commands still outstanding (decremented by each
    `done`), so later commands of the queue keep issuing. Pending SIGNALs fire in stream order. With
    nothing pending and nothing outstanding the head SIGNAL fires directly (same cycle as before
    SG10a). A full pending buffer stalls the queue head; a counter at 255 stalls the SIGNAL. It never
    wraps and never loses a signal.

  Same-cycle requests to one semaphore are granted in fixed queue order (DMA, ARRAY, VEC).
* **IRQ / END** are barriers. They execute once all three queues are empty, nothing is
  outstanding and no SIGNAL is pending. `IRQ(code)` latches IRQ_CODE and sets IRQ_STATUS bit 1. `END` sets IRQ_STATUS bit 0 and
  increments END_COUNT. **END does not block later words.** If software has already moved tail past
  END (the next program is queued), dispatch carries on. Otherwise fetching stops because the ring
  is empty. This avoids a lost-doorbell race when the host queues program n+1 before program n ends.
* **Stream limit.** At most 255 SIGNALs may be outstanding on one semaphore ahead of their WAITs, plus
  whatever fits in the producer queue (8) and its pending-SIGNAL buffer (4); `funcsim.check_hazards`
  keeps the conservative 255 + 8. More than that deadlocks the in-order dispatcher. Compiler
  output has 1-2 (docs/isa.md §2). `ctrl_semaphore_saturation` runs at 260.
* **Errors halt dispatch**, and the first error is latched. The queues still drain what was dispatched
  before the error, so every command before a bad word executes and none after it does. The halt also
  sets IRQ_STATUS bit 2. Recovery is a soft reset (CTRL bit 1) or `rst_n`.

| ERR_CODE | cause |
|---|---|
| 1 OPCODE | opcode not in `isa.OPCODES`: 0x00 (zeroed memory) and every undefined value. There are no "reserved but ignored" opcodes. A future opcode is illegal until `cmd_fields.sv` is regenerated |
| 2 RESV | a reserved bit above the opcode's last field is set |
| 3 FIELD | an `isa.validate` rule: WAIT/SIGNAL queue = 3, WAIT count = 0, DMA sp_buf > 4, VEC vop > 8 / buffer id > 4 / dtype 3, GEMM int4 with w_trans or odd N, GEMM_GS odd N, LUT_LOAD table 3 |
| 4 FETCH | RRESP error on a fetched word |
| 5 DMA | RRESP/BRESP error in either DMA engine (the command still completes) |
| 6 RING | doorbell with RING_SIZE = 0 or tail >= RING_SIZE |
| 7 ENGINE | `arr_err` or `vec_err` input |

ERR_INDEX holds the ring index of the bad word. ERR_W0..3 hold the word itself (decode errors and
fetch errors only).

* **Soft reset** (CTRL bit 1). It blocks new AR/AW and dispatch, waits until neither AXI master and no
  scratchpad read is outstanding, then pulses an internal clear for one cycle. The clear empties the
  fetcher, queues (and pending SIGNALs), semaphores and both DMA engines, and sets head = tail = 0 and
  ERR = 0. `eng_clr` carries the same
  pulse to the ARRAY/VEC engines, which must reset with it. Ring base/size, IRQ_EN and the perf
  counters are kept. CTRL bit 1 reads 1 while the reset is in progress.
* **Single step** (CTRL bit 0). Each write to DBG_STEP lets exactly one word dispatch, or one barrier
  execute. DBG_LAST holds the ring index of the last dispatched word.

## 2. Engine command port (ARRAY = `arr_*`, VEC = `vec_*`; DMA uses the same port internally)

| signal | dir | meaning |
|---|---|---|
| `*_cmd_valid` / `*_cmd_ready` | out / in | valid/ready; valid and word are held until accepted |
| `*_cmd_word[127:0]` | out | the ISA word, unchanged. Decode it with `cmd_fields` (generated) |
| `*_done` | in | one pulse per accepted command, **in acceptance order** (per DMA channel), at least 1 cycle after acceptance; one per cycle max |
| `*_err` | in | level; halts the processor (ERR 7) |

"Done" means the command's results are visible to every other unit (written to the scratchpad),
because a SIGNAL is released by it. The engine may accept new commands before earlier ones are done
(up to MAXOUT), and it must keep its own same-queue interlocks (docs/isa.md §2).

**Adapters SG7 needs.** Both are command sequencers. The port shape itself needs no change.
* **ARRAY:** GEMM / GEMM_GS / ATTN_TILE word → C3 tile commands for `array_ctrl`
  (`cmd_m/k/n/abase/acc/emit/int4`, one per weight tile over N/n × K/n), WBUF → `w_data` and
  ABUF → `a_data` streams, and `out_*` rows → ACC writes. `*_done` pulses after the last ACC write of
  the last tile. `array_ctrl.idle` alone is not enough, because it does not see the ACC write-back
  stream of the adapter.
* **VEC:** VEC / REQUANT / LN / SOFTMAX_ONLINE / GELU / KV_APPEND / LUT_LOAD → a sequence of
  `vec_unit` row commands (`cmd_op/len/fmt/...`, docs/vector-unit.md §5) with scratchpad readers for
  the A/B/P streams and a writer for D. `vec_done` pulses after the writer drains the last row.
  `vec_unit.illegal_op` → `vec_err`.

## 3. DMA engine

`DMA_LOAD` / `DMA_STORE` per docs/isa.md. Row r moves `row_bytes` bytes between DDR
`ddr_addr + r*ds` and scratchpad `sp_addr + r*ss` of `sp_buf`. `ds = ddr_stride` (0 means
`row_bytes`). `ss = 16*sp_stride` (0 means `row_bytes`).

* **Alignment.** The engine is **byte-granular** in every address, stride and length. A per-row
  realigner shifts bytes between the DDR beat grid and the scratchpad line grid, with one bubble
  cycle per row. It is therefore a strict superset of the ADR-004 rule (16-byte-aligned bases,
  byte-granular `row_bytes` / `ddr_stride`). Commands that break that rule still behave as defined
  here. DDR addresses wrap at 2^32 and scratchpad addresses at 2^21.
* **AXI4 master.** 128-bit data and one ID, so responses arrive in order. Bursts are INCR with 16-byte
  beats, beat-aligned, and up to `MAXB` = 256 beats; a burst never crosses 4 KiB or a row end. Partial
  beats use WSTRB. Up to 4 read and 4 write bursts can be outstanding. Read data lands in an
  `LDF`-beat FIFO (512 beats = 8 KiB by default, sized for 2 full bursts) whose space is reserved at AR
  time, so RREADY is always 1.
* **Pipelining (SG10a).** Up to `NCMD` = 4 commands are in an engine. Accepted commands become
  descriptors in two in-order FIFOs: the address stage issues a command's ARs/AWs and moves to the
  next command as soon as its last burst is issued; the data stage (realigner + scratchpad writes or W
  beats) works on the oldest command and switches to the next without draining the AXI side. So a
  LOAD's ARs go out while earlier LOADs' data is still landing (one DDR latency per stream of LOADs,
  not per command); measured: 8 back-to-back 1 KiB loads at 40 cycles latency take 588 cycles instead
  of ~850 (`dma_pipelined_loads`), ~4 cycles of hand-off per command. A STORE is exclusive: it enters
  only an empty engine and nothing enters behind it.
* **Completion.** `done` pulses, in acceptance order, when a LOAD's last scratchpad write is accepted,
  or when a STORE's last B response arrives. A later load in the DMA queue sees an earlier store (the
  STORE fence covers both channels).
* **`chan`** selects the engine: `chan & 1` (ADR-005). The compiler sends WBUF weight panels to the
  channel with fewer bytes so far, activations to channel 1, parameters to channel 0, K/V loads to
  channels 2/3 (engines 0/1).
* **Cross-channel hazards.** Commands of different channels are ordered only by a STORE fence, a
  semaphore path (a DMA SIGNAL covers both channels; a DMA WAIT holds both), or a barrier. Two
  unordered LOADs on different channels that write the same scratchpad bytes are a **race, rejected by
  `funcsim.check_hazards`**; the compiler's SyncBuilder either moves such a LOAD to the other channel
  or orders it with a DMA→DMA semaphore (sem 6). The hardware does not compare addresses.
* **Scratchpad line port** (`sp_*`; line address = byte address >> 4, 17 bits):
  * Write: `sp_wvalid/sp_wready`, `sp_wbuf[2:0]`, `sp_waddr`, `sp_wdata[127:0]`, `sp_wstrb[15:0]` (byte enables).
  * Read request: `sp_arvalid/sp_arready`, `sp_arbuf`, `sp_araddr`.
  * Read response: `sp_rvalid`, `sp_rdata`. Responses come in request order with any latency and no
    backpressure; at most 16 lines are in flight (credit).

  SG7 maps lines onto the physical buffers (ABUF/WBUF/ACC/PBUF/KVBUF widths and banking) and arbitrates
  them against the engines.
* **Performance notes.** Each row still costs one realigner bubble plus ~2 cycles of row start; the
  compiler coalesces contiguous transfers (`ds = ss = row_bytes`) into single rows of up to 64 KiB, so
  the weight stream pays it once per 64 KiB (removing it would not move decode).

## 4. AXI ports of `ctrl_top`

`mf_axi_*`: fetch, read channels only. `md_axi_*`: DMA channel 0, full AXI4. `md1_axi_*`: DMA channel 1,
full AXI4. `s_axil_*`: CSRs. All have 32-bit addresses (parameter `AW`), ID width `IDW` = 4 (always 0),
AxCACHE = 0011 and AxPROT = 0. Board mapping (ADR-005 §4): KV260 `md` → S_AXI_HP0_FPD, `md1` →
S_AXI_HP1_FPD, `mf` → S_AXI_HPC0_FPD; PYNQ-Z2 `md` → S_AXI_HP0, `md1` → S_AXI_HP1, `mf` shares HP0 with
`md` through a 2:1 AXI SmartConnect (its traffic is ~1 % of the DMA's).

## 5. CSR map (AXI4-Lite, 32-bit; byte strobes honoured; unmapped reads 0, writes ignored)

| off | name | access | content |
|---|---|---|---|
| 0x000 | ID | RO | 0x4E584C43 ("NXLC") |
| 0x004 | VERSION | RO | 0x00010000 |
| 0x008 | CTRL | RW | [0] step_mode, [1] soft_reset (write 1; reads 1 while in progress) |
| 0x00C | STATUS | RO | [0] halted, [1] queues idle (nothing queued or outstanding), [2] ring empty (head = tail), [3] busy, [6:4] queue idle (DMA, ARR, VEC), [10:8] engine busy, [13] ring cfg ok, [14] soft reset busy, [15] queue overflow (done without a command; sim aid) |
| 0x010 | ERR_CODE | RO | §1 table |
| 0x014 | ERR_INDEX | RO | ring index of the bad word |
| 0x018-0x024 | ERR_W0..3 | RO | the bad word, little-endian 32-bit pieces |
| 0x030 | RING_BASE | RW | byte address, bits [3:0] read 0 |
| 0x034 | RING_SIZE | RW | ring size in 16-byte words (24 bits) |
| 0x038 | RING_HEAD | RO | next word to dispatch |
| 0x03C | DOORBELL | RW | tail (the next word software will write); a write rings the doorbell. Keep at least one slot free |
| 0x040 | IRQ_EN | RW | [0] END, [1] IRQ command, [2] error |
| 0x044 | IRQ_STATUS | W1C | same bits; `irq = |(IRQ_STATUS & IRQ_EN)` |
| 0x048 | IRQ_CODE | RO | code of the last IRQ command |
| 0x04C | END_COUNT | RO | ENDs executed |
| 0x050 | DBG_STEP | WO | single step |
| 0x054 | DBG_LAST | RO | ring index of the last dispatched word |
| 0x058 / 0x05C | SEM_SEL / SEM_VAL | RW / RO | semaphore debug read |
| 0x060 | PERF_CTRL | WO | [0] clear perf counters, [1] `sat_clr` pulse |
| 0x064 | PERF_CYCLES | RO | cycles not idle (and not halted) |
| 0x068/6C/70 | PERF_BUSY_DMA/ARR/VEC | RO | cycles with at least one command outstanding at that engine (DMA: either channel) |
| 0x074 | PERF_RETIRED | RO | commands completed (engine commands at done, WAIT/SIGNAL at grant, IRQ/END at execution) |
| 0x078/7C | PERF_BYTES_LO/HI | RO | DMA bytes moved, both channels (64-bit) |
| 0x080+4i | SAT_i | RO | `sat_cnt[32i+31:32i]` (i < NSAT = 5), 32-bit saturating event counters of the VEC engine in the canonical order of `vec_seq.sat_cnt`: SAT_0 requant (REQUANT), SAT_1 ALU (VEC, incl. the residual add), SAT_2 LN out (LN affine clamp), SAT_3 LN z (LN normalize clamp), SAT_4 softmax score (SOFTMAX_ONLINE score requant). `nanoxl_top` wires `vec_seq.sat_cnt` straight through; `nanoxl/driver/csr.py` `SAT_NAMES` and the SimBackend use the same slots (AUDIT-1 A1-02). PERF_CTRL bit 1 clears them |

Host sequence: set RING_BASE, RING_SIZE and IRQ_EN; write words at index tail and onward; write
DOORBELL = new tail; on END (IRQ bit 0, or END_COUNT) clear the status with W1C.

## 6. Verification

| test | what |
|---|---|
| `test_cmd_isagen` | `cmd_fields.sv` equals the generator output for the current `nanoxl/isa.py` |
| `test_cmd` (`cmd_cocotb`) | every golden vector of `tests/py/test_isa.py`; 150 random legal commands per opcode, each field compared; 4000 random/near-legal words: RTL legality and error class equal `isa.decode`, plus route/kind/sem/count/code |
| `test_dma` (MAXB 16/LDF 32 and 256/512) | `dma_pipelined_loads`: 8 back-to-back loads at 40-cycle read latency overlap (next AR before previous done; < 8·64 + 2·lat cycles), then mixed LOAD/STORE batches on overlapping bytes under latency and backpressure; 40 random LOAD/STORE with byte-granular geometry (1 B to 9 KB rows, strides, overlapping rows, starts just below 4 KiB), AxiRam random backpressure on all 5 channels, random scratchpad ready/latency; full DDR window and all scratchpad bytes compared; AXI monitor (aligned, ≤ MAXB, no 4 KiB crossing, ≥ 2 reads outstanding); directed 4 KiB splits, max bursts, every byte offset, top-of-address-space wrap; store→load ordering; RRESP/BRESP errors; reset mid-transfer |
| `test_ctrl` (`ctrl_cocotb`) | `ctrl_dual_dma` (60-cycle latency on both DMA masters): a short LOAD on one channel finishes before a long one on the other, and the DMA SIGNAL behind both releases its WAIT only after the later done (both orders); STORE fence across channels (store sees the other channel's load, a later load sees the store, issue waits for both); a LOAD behind a pending SIGNAL issues before the SIGNAL fires; PERF_BYTES / PERF_RETIRED. ctrl_top with shared AxiRam (x2) / AxiRamRead memory, a scratchpad model per DMA channel, AxiLiteMaster with random AW/W/AR stalls, engine stubs. A 1500-command random stream obeying the compiler invariants runs through a 48-word ring that wraps and straddles 4 KiB. The scoreboard checks per-queue order and words, every SIGNAL→WAIT dependency, barriers, final semaphores = 0, DDR/scratchpad contents, PERF_RETIRED, PERF_BYTES, exact PERF_BUSY per engine, and cross-queue overlap. DMA commands use random channels (cross-channel scoreboard: a SIGNAL's bound is the latest done of all earlier DMA commands; streams keep the ADR-005 hazard rule) Also: multi-program runs under three latency mixes, semaphore saturation at 255, IRQ/CSR/W1C/strobes/sat inputs, single step, four illegal-word cases and four error sources each followed by soft-reset recovery, and hardware reset mid-stream |
| `test_ctrl_formal` | BMC + k-induction (yosys-smtbmc, z3) for `fifo_sync` (depth 4 and 8: no overflow/underflow, count consistency, tagged-element order and data integrity), `cmd_queue` with 2 channels (`queue`: a tagged SIGNAL fires only after every earlier engine command of both channels completed; a STORE fence issues only with nothing outstanding and nothing issues while it runs; pending-buffer invariants), `sem_unit` (counts never negative or wrapping, granted signals = consumed + held, grants only when enough, progress), `cmd_ring_ptr` (pointers in range, occupancy = fetched-not-dispatched ≤ FDEPTH, never past tail, no burst across the ring end or 4 KiB). Skips if a tool is missing (`pip install z3-solver`) |

Mutation sanity:
* Semaphore grant ignoring same-cycle earlier WAITs: killed by formal `sem`.
* Carried SIGNAL counting only channel 0's outstanding commands (SG10a): killed by the ctrl scoreboard
  (`ctrl_random_stream`, `ctrl_latency_mixes`, `ctrl_dual_dma`, ...: "arr cmd started before dma cmd
  done") and by formal `queue`.
* SIGNAL not waiting for outstanding commands: killed by the ctrl scoreboard.
* DMA without 4 KiB splitting: killed by `test_dma`.
* Ring pointer without the 4 KiB cap: killed by formal `ring`.
* FIFO `in_ready = 1`: killed by formal `fifo`.
