# ADR-006: First-bitstream board configuration

Status: **accepted** (SG12, 2026-11-11). Nothing here has been through Vivado. The choice rests on
Yosys estimates and the cycle model, and it is meant to be revisited once Vivado has counted (see
"Step-up" below).
Evidence: `bench/results/resources_kv260.json` and `resources_pynq-z2.json` (the chosen configs),
the candidate runs below (`python3 fpga/yosys/estimate.py kv260 --params ...`, per block, `--jobs 1`),
`nanoxl.cyclemodel.estimate` with the as-built `RtlTiming` (`BOARD_TIMING[board]`, `vec_port_lines = VP`),
`tests/fpga` (`test_hwconfig_boards_mirror_fpga`, `test_resource_estimates_current`, `test_build_tcl_*`) and
`tests/fpga/test_fpga_wrap.py::test_board_rtl_tiny` (slow, Verilator: the bitstream's exact `nanoxl_top`
parameters, tiny W8A8/W4A8 with on-chip and DDR KV, bit-exact).

## Context

SG9a found that the as-designed NanoXL (N = 32, L = 64) needed 15× the XCK26's LUTs. All area changes
C1–C9 (docs/fpga.md §2.3) are now merged. They change no numeric result and no ISA. What remains to
decide is the structure of the first bitstream: array size `N`, vector lanes `L`, VEC port width `VP`,
banks `NB`, buffer sizes and clock. A first bitstream exists to bring the board up: CSR, DMA loopback,
measured DDR bandwidth (ADR-003's largest unknown), and tiny bit-exact. So the risk that it fails to place
or route counts for more than its speed.

## Margin: what "fits" means here

**Target: Yosys LUTs (logic + LUT-as-memory) ≤ 70 % of the device; FF, DSP, BRAM, URAM ≤ 80 %.**

* The LUT figure **counts LUTRAM / SRL in LUT6 sites** (`RAM32M16` = 8, `RAM64M8` = 8, `SRL16E` = 1, ...;
  `fpga/yosys/estimate.py LUTRAM_LUT6`). The SG9a totals left them out. After C1 the SFU tables are
  LUTRAM, so this adds 4–12 k LUT, 3–10 % of the XCK26. `INV` cells are still excluded, because Vivado
  folds them into the driving LUT.
* **Direction of the Yosys error.** Yosys's generic `synth_xilinx` mapping usually *over-counts* LUTs
  against Vivado, by 10–40 % on wide muxes and arithmetic (docs/fpga.md §2.2). The per-block sum also
  misses Vivado's cross-boundary constant propagation. Both errors make a Yosys "fit" pessimistic. The
  over-count has not been measured on this design, so the ADR does not spend it. It is the safety margin.
* **What the 30 % headroom covers.** The PS8 block design adds 4 SmartConnects, an `axi_intc` and a
  `proc_sys_reset`: about 5–10 k LUT, 4–8 % of the XCK26. Placement and routing at 200–250 MHz get
  uncomfortable above about 80–85 % LUT on UltraScale+. 70 % of Yosys LUTs + 8 % platform stays under
  80 % even if Yosys over-counts nothing. If Yosys over-counts by the usual 10–40 %, Vivado lands
  near 45–65 %.

## Candidates (XCK26: 117,120 LUT, 234,240 FF, 1,248 DSP, 144 BRAM36, 64 URAM)

Yosys 0.33 per block, KV260 buffer sizes (ABUF 512 KiB, WBUF 128 KiB, ACC 256 KiB, PBUF 64 KiB, KVBUF
1 MiB unless noted), `RAM_STYLE = 2`. Shakespeare (6 layers, d = 384, ctx 256) in the cycle model,
batch 1. Decode is the mean over context positions. Prefill is the 256-token prompt. KV is in DDR,
because shakespeare's 1.18 MB KV cache never fits on chip: KV_LINES must be a power of two, and 2 MiB
would take all 64 URAMs. At 250 MHz unless marked.

| config (N, L, VP, NB) | LUT % | FF % | DSP % | BRAM36 % | URAM % | W8 tok/s | W4 tok/s | prefill ms | Yosys CPU s |
|---|---|---|---|---|---|---|---|---|---|
| 32, 64, 4, 8 (SG9a, before C1–C9) | 1509 | 1006 | 213 | 80 | 100 | 607 | 644 | 15.3 | 4,470 |
| 32, 16, 4, 8 | > 169 ¹ | | | | | 607 | 642 | 17.2 | router 59 |
| 16, 16, 4, 8 | **195** (228.5 k) | 36.9 | 35.9 | 44.4 | 100 | 344 | 325 | 51.0 | 1,236 |
| 16, 16, 1, 4 | ~116 ² | | | | | 343 | 324 | 52.9 | vec_seq 735 |
| 16, 8, 1, 4 | **99.9** (116.9 k) | 31.7 | 26.3 | 22.2 | 100 | 342 | 323 | 55.2 | 699 |
| 16, 4, 1, 4 | **82.8** (97.0 k) | 28.6 | 19.6 | 21.5 | 100 | 340 | 321 | 60.7 | 467 |
| 8, 8, 1, 2 | **79.8** (93.5 k) | 21.5 | 17.9 | 11.1 | 96.9 | 175 | 156 | 193.3 | 595 |
| **8, 4, 1, 2, KVBUF 512 KiB (chosen)** | **62.2** (72.8 k) | 18.4 | 11.2 | 10.4 | 75.0 | 174 | 156 | 198.8 | 433 |
| … the chosen config at **200 MHz** | | | | | | **139** | **125** | **248** | |

¹ The five scratchpad routers (5 × 20.4 k) and `vec_seq` (84 k + 11.5 k LUTRAM) alone are 198 k.
² `vec_seq` at VP = 1 is 78.4 k LUT against 84.3 k at VP = 4, and the routers at NB = 4 are 5 × 4.9 k.
tiny (on-chip KV, chosen config, 200 MHz): prefill 12.1 ms (128 tokens), 1,650 tok/s W8 / 1,496 W4.

What the table says:

* **N decides the speed.** Decode runs at one weight row per cycle (`wload_rows_per_cycle = 1`), so it is
  weight-port bound: N = 16 → 8 → about 340 → 175 tok/s. `L` and `VP` barely move either number:
  L = 16 → 4 costs 19 % of prefill and < 1 % of decode, because LN, softmax and requant are a small
  share at d = 384.
* **L and the routers decide the area.** `vec_seq` scales with L: 84 k (L = 16), 49 k (L = 8), 32 k (L = 4)
  LUT. It also has a fixed part of about 20 k: its own sequencing (7.3 k LUT, 18 k FF), `pbuf_loader`
  3 k, LN 3.8 k at L = 4. Each of the five `buf_spad` routers carries all 8 read and 4 write ports.
  Its cost is set by the widest port and NB: 20.4 k (N = 32, NB 8), 18.7 k (N = 16, VP 4, NB 8),
  4.9 k (NB 4, VP 1), 3.0 k (N = 8, NB 2). NB ≥ max(VP, N/4) is required (docs/top.md §3), so N = 32
  forces NB ≥ 8 and 100 k of routers.
* **W4 is 7–10 % slower than W8 on decode** at small N in this model, because every GEMM_GS tile carries
  a scale beat (`max(M, k_tile + 1)`). Its benefit is half the DDR bytes, and that does not help while the
  array's weight port, not DDR, is the bound.

## Decision

**KV260 first bitstream** (`fpga/boards.py RTL_PARAMS["kv260"]`, `fpga/kv260/build.tcl`,
`nanoxl.hwconfig.BOARDS["kv260-fpga"]`):

| parameter | value | why |
|---|---|---|
| `N` | **8** | the largest array that meets the 70 % target (N = 16 is 83 % even at L = 4) |
| `L` | **4** | L = 8 costs +21 k LUT (+18 %) for 3 % of prefill; L = 4 is also the fast-tier smoke config |
| `VP` | **1** (16-byte VEC ports) | at L = 4 one line already carries a full int32 beat (4 × 4 B), so 64-byte ports buy nothing; VP = 1 also allows NB = 2 |
| `NB` | **2** | = widest port (array ACC, N/4 = 2 lines); routers 5 × 3.0 k instead of 5 × 4.9 k at NB = 4 |
| EL, SM_RQL | `sfu` defaults: EL = min(L, 8) = 4, SM_RQL = min(L, 16) = 4 | equal to L, so no sub-beat serialization |
| `PACK` | 1 (`systolic_array` default; not a `nanoxl_top` parameter) | see below |
| `requant NDSP` | 4 (default) | see below |
| buffers | ABUF 512 KiB, WBUF 128 KiB, ACC 256 KiB, PBUF 64 KiB, **KVBUF 512 KiB** (was 1 MiB), all URAM | shakespeare runs DDR-KV at any power-of-two KVBUF, and tiny's KV fits in 128 KiB. Halving KVBUF frees 16 URAMs (48/64 used) and keeps every bank to a cascade of ≤ 4 URAMs deep (16384 × 128 b) |
| `RAM_STYLE` | 2 | every bank is ≥ 2048 lines deep at NB = 2, so all five buffers sit in URAM288 and BRAM is left to the SFU / FIFOs (15 BRAM36) |
| `OREG` | 0 | +0.2 % cycles when on; it is the first timing fallback (docs/fpga.md §5 T6), `-params "OREG 1"`, with no compiler change |
| clock | **200 MHz** (`-clk 250` once timed) | no timing report exists yet. T4 (`srdhm` 32 × 32 over 4 DSP48E2 with one internal register) and T5 (ready chains) are open, and a first bitstream that misses timing costs a rebuild. The cycle-model numbers scale with the clock: 174 / 156 tok/s at 250 MHz |
| KV mode | tiny on chip; **shakespeare DDR-KV** (`fpga/run_board.py --kv ddr`) | |

Yosys, chosen config (`bench/results/resources_kv260.json`): LUT 72.8 k incl. LUTRAM (62 %), FF 43 k
(18 %), DSP 140 (11 %), BRAM36 15 (10 %), URAM 48 (75 %).

**PACK and NDSP.** On this board and config, LUTs are the limit and DSPs are not (11 %). So no
DSP-saving option that costs LUTs is worth turning on. `requant #(.NDSP(3))` saves one DSP per lane
for about 175 LUT each, so it stays at the default 4. WP486 packing (`PACK = 1`, C5) halves the array's
DSPs for about 6 % more `systolic_array` LUTs (N = 16: +768). It stays the default only because `PACK`
is not plumbed through `array_ctrl` / `array_seq` / `nanoxl_top`, and at N = 8 the difference is a few
hundred LUTs. Plumbing it and building with `PACK = 0` is a small LUT win on both boards, and on the 7
series it is the right setting (docs/array.md §6).

**PYNQ-Z2: no useful configuration fits.** The smallest useful build (N = 8, L = 4, VP = 1, NB = 2:
shakespeare 103 tok/s W8 / 93 W4, prefill 324 ms at 150 MHz) needs **65.2 k LUT, 123 % of the
XC7Z020** (53.2 k) by Yosys. DSP is 78 % and BRAM36 85 %. The floor is structural: `vec_seq` at L = 4 is
31 k LUT incl. LUTRAM, `ctrl_top` 8.8 k, five routers ≥ 10 k, glue 2.5 k. Even N = 4 (35 tok/s, 1.2 s
prefill: not useful) would sit near 100 %, and the 70 % target is 37 k LUT. `RTL_PARAMS["pynq-z2"]` is
set to that N = 8 / L = 4 build as a reference, and the script and estimate are kept working. Do not
expect it to place until the RTL work below is done.

## Step-up (second bitstream)

1. Run Vivado synthesis only, `vivado ... build.tcl -tclargs -params "N 16 NB 4" -proj_only`, then
   `launch_runs synth_1`, and read `utilization`. If Vivado's CLB LUT count for the N = 16 / L = 4 core
   is ≤ about 75 %, build it: decode doubles to about 340 tok/s (250 MHz). Update `RTL_PARAMS`, `build.tcl`
   and `BOARDS["kv260-fpga"]` together, and re-run `estimate.py` (tests/fpga checks all four agree).
2. After the first timing report, raise the clock to 250 MHz (`-clk 250`, `CLOCK_MHZ`).
3. RTL changes that would bring N = 16 / L = 16 or N = 32 within reach (not in SG12's paths, owners'
   call; none changes results):
   * **C10 router port pruning.** Each buffer instantiates all 8 read and 4 write ports, but only some
     sources can address it. VEC P → PBUF only, array ACC write → ACC only; if the compiler constrains
     the array A / W / ACC read ports to the buffers it actually uses, each router keeps 4–5 of its 8
     read ports. That is about −40–50 % of 5 × 3–20 k LUT.
   * **`vec_seq` fixed cost.** 18 k FF / 7.3 k LUT of sequencing, `pbuf_loader` 3 k LUT, and
     `ctrl_top/sem_unit` 3.1 k LUT with 1.5 k MUXF (a run-time indexed counter array).
   * **Per-lane cost at L ≥ 8.** `layernorm` (13.5 k at L = 16), the two `requant` instances (7.7 k),
     `srdhm` ×(2L+1).

## Consequences

* The compiler and driver must use `fpga/boards.py:hwconfig("kv260")`, which equals
  `nanoxl.hwconfig.BOARDS["kv260-fpga"]`, on this bitstream (`fpga/run_board.py` does). The ADR-003
  planning configs `BOARDS["kv260"]` / `["pynq-z2"]` are unchanged and remain the sizing study
  (`bench/roofline.py` sweeps only those).
* `nanoxl.cyclemodel.timing_for(hw)` keys on `hw.name`, so `"pynq-z2-fpga"` gets the default timing
  (128-bit HP port) instead of the PYNQ-Z2's `axi_port_bytes = 8`. Pass `tm=BOARD_TIMING["pynq-z2"]` until
  `timing_for` strips the `-fpga` suffix. KV260 timing is the default either way.
* The modelled speed of the first bitstream is a quarter of the planned design's: 139 vs 607 tok/s
  shakespeare W8, 248 vs 15 ms prefill. The decision buys a first bitstream that is likely to place,
  route and run. It is not the performance target.
