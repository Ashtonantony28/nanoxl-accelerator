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
