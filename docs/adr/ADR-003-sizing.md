# ADR-003: Array size, vector lanes and scratchpad sizes per board (M1)

Status: proposed by SG4 (2026-11-06), from the M1 cycle model. Numbers are modelled, not measured;
M6 checks the model against RTL perf counters (target: within 10 %) and M7 against the board.

Data: `python3 bench/roofline.py` writes `bench/results/roofline.{csv,json,png}` and
`bench/results/boards.md`. Model: `nanoxl/cyclemodel.py` running the real command streams the
compiler emits (`nanoxl/compiler.py`), with synthetic weights of the right shapes. Decode numbers
average positions 0, ctx/2 and ctx-1.

## Decision

| | KV260 | PYNQ-Z2 |
|---|---|---|
| Array | **32 x 32** (1,024 MACs, 512 DSP48E2 with 2-MAC packing) | **16 x 16** (256 MACs, 128 DSP48E1) |
| Clock target | 250 MHz | 150 MHz |
| Weight port into the array | **2 rows/cycle (64 B/cycle, two WBUF banks)** | 1 row/cycle (16 B/cycle) |
| Vector lanes | **64** | **16** |
| ABUF (activations) | 512 KiB (URAM) | 192 KiB |
| WBUF (weight ring) | 128 KiB, 32 KiB panels (4 in flight) | 64 KiB, 16 KiB panels |
| ACC (int32) | 256 KiB | 96 KiB |
| PBUF (params, LUTs) | 64 KiB | 32 KiB |
| KV tier | 1.5 MiB URAM, on-chip for one `shakespeare` sequence; DDR-KV mode for batch > 1 | DDR only, 96 KiB staging ring |
| Total on chip | ~2.5 MiB (URAM 2.25 MiB + BRAM) | ~480 KiB of 630 KiB BRAM |

These are the defaults in `nanoxl/hwconfig.py` (`KV260`, `PYNQ_Z2`).

## Modelled results with this decision

| board | model | W | KV | batch | prefill (T = ctx) | prefill array util | decode tok/s | weight-stream bound |
|---|---|---|---|---|---|---|---|---|
| KV260 | shakespeare | 8 | on-chip | 1 | 12.4 ms (256, chunks of 128) | 0.93 | **712** | 752 |
| KV260 | shakespeare | 4 | on-chip | 1 | 12.3 ms | 0.93 | **1,360** | 1,504 |
| KV260 | shakespeare | 8 | DDR | 1 / 4 / 8 | 12.4 ms | 0.93 | 675 / 2,310 / 3,873 | 752 / 3,007 / 6,014 |
| KV260 | shakespeare | 4 | DDR | 8 | | | 5,730 | 12,028 |
| KV260 | tiny | 8 | on-chip | 1 | 0.61 ms (128) | 0.76 | 8,765 | 10,066 |
| PYNQ-Z2 | shakespeare | 8 | DDR | 1 / 8 | 82 ms (chunks of 32) | 0.92 | 128 / 744 | 141 / 1,128 |
| PYNQ-Z2 | shakespeare | 4 | DDR | 1 | 82 ms | 0.92 | 234 | 282 |
| PYNQ-Z2 | tiny | 8 | DDR | 1 | 3.7 ms | 0.80 | 1,488 | 1,887 |

Goal G3 (>= 60 % of DDR bandwidth in decode, >= 50 % array utilization in prefill) is met in the
model everywhere at batch 1: KV260 decode runs at 95 % of the weight-streaming bound with 98 % DDR
utilization; prefill is compute-bound at 92-93 % utilization.

## Sanity check against the design plan

* Plan §5: ~750 tok/s INT8 decode at 8 GB/s. Model: **712 tok/s**. The plan's number is the pure
  weight-streaming bound (10.7 MB per token). The model is 5 % below because each token also moves
  0.25 MB of requant/LN parameters, the per-layer serial chain (LN -> QKV -> attention -> proj ->
  LN -> FC -> FC2) leaves short DMA bubbles at op boundaries, and attention reads K/V through the
  array's weight port (6 heads x 6 layers x 2 GEMMs per token).
* Plan §5: ~1,400 tok/s INT4. Model: **1,360**.
* Plan §5: prefill of 256 tokens ~21 ms at an assumed 50 % utilization. Model: **12.4 ms** at
  93 %, because weight-stationary tiles are reused across a 128-row chunk and weight DMA overlaps
  compute through the WBUF ring. Treat 93 % as optimistic: the model has no SRAM bank conflicts
  and no AXI stalls.
* Plan §5: PYNQ-Z2 ~140 tok/s. Model: **128** (91 % of the bound).

## Why these numbers (findings from the sweep)

1. **The array's weight port is a second roofline.** In GEMV (batch-1 decode) every weight byte
   enters the array once, at `N x rows_per_cycle` bytes per cycle. At N = 32, 250 MHz and one row
   per cycle that is 8 GB/s, the same as the DDR estimate. The two limits then add up instead of
   overlapping: decode reached 668 tok/s at 8 GB/s and stayed at **670 tok/s at 12 GB/s**. A
   2-row weight port (two WBUF banks, double-buffered PE weight registers loading two rows per
   cycle) makes 12 GB/s usable (1,043 tok/s) and gives +7 % at 8 GB/s. It costs a 512-bit WBUF
   read path and doubled preload wiring, not DSPs. Chosen for KV260. At PYNQ-Z2 bandwidth (1.5
   GB/s against a 2.4 GB/s port) a 1-row port is enough.
2. **N = 64 does not fit KV260** (4,096 MACs = 2,048 DSPs with packing, against 1,248). It would
   cut prefill to 4 ms but only buys decode what the 2-row port at N = 32 already buys.
   N = 16 caps decode at ~350 tok/s through the weight port (16 B x 250 MHz = 4 GB/s), and N = 8
   caps it at ~176 tok/s.
3. **PYNQ-Z2:** N = 16 reaches 91 % of the 1.5 GB/s bound. N = 32 (512 MACs = 256 DSPs) does not
   fit 220 DSPs, and the sweep shows it gains nothing for decode at 1.5 GB/s.
4. **Lanes:** prefill is 14.7 / 13.2 / 12.4 / 12.1 ms at 16 / 32 / 64 / 128 lanes on KV260, and
   decode is flat (712 tok/s). 64 lanes keeps the vector work (requant, LN, softmax) under ~7 % of
   prefill. PYNQ-Z2: 16 lanes (82 ms) against 8 (86 ms).
5. **Panel size:** the WBUF ring should hold about 4 panels. 4-8 KiB panels add per-command
   overhead; panels of half the ring (2 in flight) cannot prefetch across op boundaries. With
   32 KiB panels in a 128 KiB ring, the DMA queue runs up to 3 panels ahead.
6. **KV placement:** on-chip KV is worth +5 % at batch 1 (712 against 675 tok/s), because DDR
   KV traffic shares the DMA queue with weight streaming. The 1.5 MiB tier holds one
   `shakespeare` sequence (1.18 MB); batch 4/8 must use DDR-KV mode and still reach 3.2x / 5.4x
   the batch-1 throughput. A future option is per-sequence KV tiles in URAM for small batches.
7. **ABUF:** 384 KiB suffices with 64-row prefill chunks, at the same latency (12.4 ms) because
   prefill is compute-bound. 512 KiB is chosen for headroom (128-row chunks, the m25 model).
8. **INT4 group scales:** the ISA and compiler keep INT4 scales per output channel (applied in
   REQUANT). K-group scales would need a GEMM per group into a scratch accumulator plus a scaled
   vector accumulate. This is left to ADR-002 (SG1's accuracy data). The model's INT4 numbers
   assume per-channel scales.

## Model limits (what M6/M7 must check)

No SRAM bank conflicts; no AXI write-response or outstanding-transaction limits beyond one
latency per DMA; LN/softmax cost is a guess until SG5 (2-3 passes plus a fixed per-row latency);
command fetch is charged as issue cycles, not DDR bandwidth (~16 B per command, under 1 % of
traffic); no clock-domain or refresh effects. Achievable PL DDR bandwidth (8 GB/s on KV260 needs
two 128-bit HP ports at 250 MHz) is the largest unknown and is measured first in M7.

## Update (SG4c, wave 3): cycle model on measured RTL costs

Status: the board numbers above came from the M1 estimates. They are **superseded** by the numbers in
this section. `nanoxl/cyclemodel.py` now charges the costs the RTL docs measured. Command fetch is
over AXI (8-word bursts, one in flight, 16-word FIFO), dispatch is one word per cycle into 8-deep
queues, and a SIGNAL at a queue head blocks that queue until the earlier commands complete
(docs/control.md). The DMA engine moves one 128-bit beat per cycle on a single master. It does not
pipeline commands against each other, it allows at most 4 bursts in flight (each at most 256
beats and within one row), and it adds a 1-cycle realigner bubble per row. C3 tiles cost max(M, k)
with a first output after 2N + 4 cycles, and INT4 still moves one K-row per beat (docs/array.md).
LN and softmax cost 3·nb + 19 per row plus 1 accept cycle, with no overlap between rows. An LN
block costs 2 + cols/8 + cols/4 cycles to load when it is not resident, and LUT_LOAD costs 32
cycles (docs/sfu.md). On the PYNQ-Z2 the 64-bit HP port caps the DMA at 8 B/cycle. The DDR
bandwidth and latency per board are unchanged assumptions (8 GB/s / 300 ns, 1.5 GB/s / 400 ns).
`python3 bench/roofline.py` regenerates `bench/results/{boards,fixes,breakdown}.md`.

**Compiler fix applied in this update.** The compiler emitted every int8 weight panel as `kc` rows
of `nt` bytes (384 × 64 B for shakespeare), even though both sides are contiguous. Each 64-byte
row becomes its own burst, and with 4 bursts in flight that is about 4 rows per DDR latency. The
compiler now coalesces every DMA whose source and destination are contiguous into single-row
commands of up to 64 KiB (`Compiler._dma`). The GEMM commands and the buffer layouts are
unchanged. The funcsim-vs-golden suite, including the slow shakespeare cases, is still
bit-exact. `test_compiler_decode_utilization_measured` guards against a regression.

| board | model | W | decode tok/s: M1 estimate → measured, old compiler → **measured, coalesced** | prefill ms (T = 256): estimate → old → **now** | weight bound |
|---|---|---|---|---|---|
| KV260 | shakespeare | 8 | 712 → 55 → **333** | 12.4 → 38.0 → **13.9** | 752 |
| KV260 | shakespeare | 4 | 1,320 → 579 → **597** | 12.7 → 14.2 → **14.2** | 1,500 |
| KV260 | tiny | 8 | 8,765 → – → **3,780** | 0.67 → – → **0.91** (T = 128) | 10,066 |
| PYNQ-Z2 | shakespeare | 8 | 128 → 21 → **99** | 82.7 → 395 → **86.5** | 141 |
| PYNQ-Z2 | shakespeare | 4 | 229 → – → **176** | 82.3 → – → **84.5** | 281 |

Batch-1 G3 now holds for prefill: the array is 83 % busy on the KV260 and 87 % on the PYNQ-Z2.
It does not hold for KV260 decode, which uses 46 % of the assumed 8 GB/s because the single
128-bit DMA master tops out at 4 GB/s. The PYNQ-Z2 reaches 70 % of its weight bound and 88 %
of what its 64-bit port can deliver.

### Where the time goes (KV260, shakespeare W8, `bench/results/breakdown.md`)

Two numbers per op class: its share of the critical path, then the engine busy time as a share
of the total.

* **Decode** (751k cycles per token): DMA weights 93.0 % / 93.0 %, DMA params (requant records,
  LN blocks) 3.5 % / 3.6 %, attention GEMMs (K and V read through the weight port) 2.5 %, linear
  GEMMs 0.8 % on the path while the array is 44 % busy. Fetch, dispatch, LN, softmax and requant
  are each under 0.5 %. Decode is a pure DMA stream.
* **Prefill** (3,469k cycles, two 128-row chunks): linear GEMMs 78.5 % (the array is 77 % busy),
  softmax 7.7 %, attention GEMMs 6.4 %, LN 3.8 %, GELU and KV_APPEND 1.1 % each. Weight DMA is
  40 % busy and fully hidden behind the GEMMs.
* **W4 decode** (419k cycles): DMA weights 85 % on the path, and the array is 81 % busy. INT4
  halves the DMA bytes, but the array still takes one K-row per cycle, so the 1-row weight port
  and the DMA are now about equal limits.

### Top 3 bottlenecks and modelled gains (`bench/results/fixes.md`)

1. **One 128-bit DMA master = 4 GB/s at 250 MHz** (half the assumed 8 GB/s). Fix: a second DMA
   master on a second HP port, striping or splitting by the `chan` hint, which the ISA already
   carries; the scratchpad line port must widen to match. KV260 W8 decode 333 → **614 tok/s
   (1.84x)**. W4 gains only 1.09x (597 → 649) until fix 3. PYNQ-Z2 W8 99 → 122 (1.24x).
2. **The DMA runs one command at a time, and SIGNAL blocks its queue until the command
   completes.** Every 24 KiB panel pays a full 75-cycle DDR round trip. Fix: overlap the next
   command's AR with the current data phase, and carry the SIGNAL on the command's `done`
   instead of the queue head. KV260 W8 decode **+5 %** (350), prefill +2 %, PYNQ-Z2 +4 %. The
   gain is small and costs control-plane complexity, so it is second priority.
3. **Array weight port: 1 K-row per cycle, also for INT4** (C3). This binds W4A8 decode once the
   DMA is doubled. Fix: the 2-row port that ADR-003 proposed above (two WBUF banks, two rows per
   cycle; for INT4, two nibble rows per beat). KV260 W4 decode with fixes 1-2: 674 → **1,238
   tok/s (1.84x)**, and W8 gains another 3 %. With all fixes, KV260 W8 decode reaches 696 tok/s
   (93 % of the 8 GB/s weight bound).

Not worth doing now: the **SFU second row buffer** (ping-pong) leaves decode unchanged and gives
prefill 1.4 % (13.88 → 13.69 ms), although softmax is 7.7 % of the prefill critical path. Resident
SFU parameter slots and an on-chip command ring each give under 1 %. Command fetch over AXI hides
behind the DMA queue (critical-path share < 0.1 %).

**Decision impact.** The array size, lanes and buffer sizes in the table at the top stay as they
are. The weight-port row in that table ("2 rows/cycle" for the KV260) is not what was built: the RTL
has 1 row per cycle. It becomes fix 3, behind the second DMA master (fix 1), which matters most
for M7. The 8 GB/s KV260 assumption now requires two DMA masters on two HP ports.
