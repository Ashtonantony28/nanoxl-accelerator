# NanoXL on the FPGA (SG9a / M7 prep): build, bring-up, resources, timing

**Status: nothing in this document has run on Vivado or on a board.** It was written in a container
without Vivado or hardware. What *was* run here: Yosys resource estimates (§2), a tclsh dry run of both
build scripts against logging stubs of every Vivado command, the RTL file-set and wrapper-port checks,
the `nanoxl_top` smoke test through the Vivado wrapper (bit-exact against golden), and the first-light
checklist against the CSR emulator (`python3 -m pytest -q tests/fpga`). Treat every Vivado step as a
first attempt, and expect to fix a property name or two.

Pinned versions this was written for:

| tool | version | notes |
|---|---|---|
| Vivado ML **Standard** (free) | **2024.2** | covers XCK26 (KV260) and XC7Z020 (PYNQ-Z2); every Tcl command used exists since 2020.2 |
| Kria KV260 | Ubuntu 22.04 for Kria + Kria-PYNQ (**PYNQ 3.0.1**) | `sudo bash install.sh -b KV260` |
| PYNQ-Z2 | **PYNQ v3.0.1** SD image | |
| Host Python | this repo (`nanoxl`), numpy | copy the repo to the board |

| file | what |
|---|---|
| `rtl/nanoxl_axi_wrap.v` | Verilog-2001 shell around `nanoxl_top` (IP integrator module references need a Verilog/VHDL top). Pure rename: `aclk`, `aresetn`, `irq`, `s_axi_ctrl_*`, `m_axi_mf_*`, `m_axi_md_*`, `m_axi_md1_*`, with `X_INTERFACE_INFO` attributes. **Generated** by `python3 fpga/gen_wrap.py` from `nanoxl_top`'s port list; re-run it after any port or parameter change (a test fails otherwise) |
| `fpga/boards.py` | **single source of truth**: `nanoxl_top` parameters per board, clock, part, CSR base, device capacities, and `hwconfig(board)`, the `HwConfig` the compiler must use with that bitstream |
| `fpga/kv260/build.tcl`, `fpga/pynqz2/build.tcl` | Vivado project + block design + synth/impl/bitstream + reports + overlay export |
| `fpga/common/nanoxl_build.tcl`, `fpga/common/rtl_files.tcl` | shared procs (argument parsing, project, core cell, address mapping, run/report/export) and the RTL list |
| `fpga/kv260/nanoxl_kv260.xdc`, `fpga/pynqz2/nanoxl_pynqz2.xdc` | board constraints (none needed today: the scratchpad bank primitive is the `RAM_STYLE` parameter of `nanoxl_top`, SG11b); clocks come from the PS IP |
| `fpga/yosys/estimate.py`, `fpga/yosys/prep.py`, `fpga/yosys/breakdown.py` | Yosys resource estimates → `bench/results/resources_<board>.json`; `breakdown.py` prints a per-module table (own cells × instances) from a job's `.stat` |
| `fpga/first_light.py` | first-light checklist (§6), board or emulator |
| `fpga/run_board.py` | `python3 -m nanoxl.driver` with the as-built `HwConfig` (use this on the board, §4) |
| `tests/fpga/` | static checks + wrapper smoke (§7) |

## 1. Board configurations (as built)

**ADR-006** chose these configurations: the first KV260 bitstream is N = 8, L = 4. `nanoxl_top` needs
power-of-two buffer capacities (line addresses wrap), and the banks must land on whole BRAM/URAM
primitives, so the bitstreams differ from the ADR-003 planning numbers. The compiler must never use
more than the RTL has. `fpga/boards.py:hwconfig()` gives the matching `HwConfig`, also available as
`nanoxl.hwconfig.BOARDS["kv260-fpga"]` / `["pynq-z2-fpga"]`: structure from the RTL, timing assumptions
from ADR-003, 1-row weight port as built.

| | KV260 (XCK26, `xck26-sfvc784-2LV-c`) | PYNQ-Z2 (`xc7z020clg400-1`): **does not fit** (ADR-006) |
|---|---|---|
| clock (pl_clk0 / FCLK_CLK0) | **200 MHz** (`-clk 250` once timing is known) | 150 MHz requested (the PS7 PLLs give the nearest value) |
| array `N`, lanes `L`, VEC ports `VP` | **8, 4, 1** (16-byte ports) | 8, 4, 1 (reference build) |
| `NB` banks / buffer | **2** (= widest port, array ACC N/4) | 2 |
| ABUF | 512 KiB, URAM (2 × 16384 lines = 8 URAM288 per bank) | 128 KiB, BRAM |
| WBUF | 128 KiB, URAM (2 × 4096 lines) | 64 KiB, BRAM |
| ACC | 256 KiB, URAM (2 × 8192 lines) | 64 KiB, BRAM |
| PBUF | 64 KiB, URAM (2 × 2048 lines) | 32 KiB, BRAM |
| KVBUF | **512 KiB** URAM (tiny on chip; shakespeare DDR-KV) | 128 KiB (tiny on chip; shakespeare: DDR-KV mode) |
| URAM / BRAM36 for scratchpads | 48 / 0 of 64 / 144 | 0 / 104 of 0 / 140 |
| `NG` (GELU slots) | 6 (shakespeare) | 6 |
| step-up (ADR-006) | `-params "N 16 NB 4"` after a Vivado synthesis count (~340 tok/s instead of ~175) | none until the vector unit shrinks |

KV260 `shakespeare` with on-chip KV needs 1.18 MB. A power-of-two KVBUF that holds it is 2 MiB, all 64
URAMs, so shakespeare always runs with `--kv ddr` (ADR-003 batch > 1 mode). `tiny` fits on chip on both boards. With every bank ≥ 2048 lines deep at NB = 2,
`RAM_STYLE = 2` puts all five buffers into URAM288. The deepest banks (ABUF, KVBUF: 16384 lines) are
cascades of 4 URAMs, half of the 1 MiB KVBUF's 8.

## 2. Resource estimates (Yosys)

`python3 fpga/yosys/estimate.py kv260` / `pynq-z2` (Yosys 0.33, `synth_xilinx -family xcup -uram` /
`-family xc7`). A whole-design flat run did not finish within 10 minutes even at N = 8, so the script
synthesizes per block and sums: `ctrl_top`, `array_seq`, `vec_seq`, the scratchpad router (`buf_spad` with
its banks as black boxes, counted 5 times, because the five instances differ only in row-address width),
one `spad_bank` per distinct depth (counted `NB` × buffers), and the `nanoxl_top` glue (blocks as black
boxes). Each block is synthesized hierarchically, so 1024 PEs cost one PE synthesis. `--params N=16,L=8,...`
estimates any other configuration (not written). Results with per-block detail:
`bench/results/resources_kv260.json`, `resources_pynq-z2.json`.

**LUT = LUT (logic) + LUT-as-memory** in LUT6 sites (`RAM32M16` / `RAM64M8` = 8, `RAM32M` / `RAM64M` = 4,
`SRL16E` = 1; `LUTRAM_LUT6` in the JSON). Since C1 the SFU tables are LUTRAM, which is 4–12 k LUT; the
SG9a numbers did not count it. `INV` cells, which Vivado folds into LUTs, are left out. The PS8 / PS7, the
SmartConnects and the interrupt controller are not included either; they add roughly 5–10 k LUT.
**ADR-006 fit target: LUT ≤ 70 % of the device, other resources ≤ 80 %.**

Totals for the configurations of §1:

| resource | KV260 (XCK26) used / avail | % | PYNQ-Z2 (XC7Z020) used / avail | % |
|---|---|---|---|---|
| LUT (logic + LUTRAM) | 72,792 / 117,120 | **62.2 %** | 65,205 / 53,200 | **122.6 %** |
| of which LUTRAM (LUT6 sites) | 5,541 | | 5,801 | |
| FF | 43,048 / 234,240 | 18.4 % | 41,586 / 106,400 | 39.1 % |
| DSP | 140 / 1,248 | 11.2 % | 172 / 220 | 78.2 % |
| BRAM36 (RAMB18 = ½) | 15 / 144 | 10.4 % | 119 / 140 | 85.0 % |
| URAM288 | 48 / 64 | 75.0 % | – | – |
| Yosys CPU time, sum of jobs (`--jobs 1`) | 433 s (vec_seq 292 s) | | 422 s (vec_seq 288 s) | |

Per block (LUT incl. LUTRAM / FF / DSP / memory; count = instances in the sum), N = 8, L = 4, VP = 1, NB = 2:

| block | count | KV260 | PYNQ-Z2 |
|---|---|---|---|
| `ctrl_top` (cmd fetch, CSRs, DMA ×2) | 1 | 9.4 k / 2.6 k / 0 / 4 BRAM36 | 8.9 k / 2.6 k / 0 / 4 BRAM36 |
| `array_seq` (array, array_ctrl, reader/writer) | 1 | 7.9 k / 9.2 k / 41 | 7.9 k / 9.2 k / 73 (no DSP48E1 packing) |
| array_ctrl memories (acc ram, out / gs FIFOs) | 3 | 1.2 k / 1.3 k / 0 | 1.3 k / 1.3 k / 0 |
| `vec_seq` (vector unit, SFUs, LN, softmax, PBUF loader) | 1 | **35.5 k** / 29.8 k / 99 / 11 BRAM36 | 33.9 k / 28.4 k / 99 / 11 BRAM36 |
| scratchpad router (`buf_spad` minus banks) | 5 | 3.0 k / 0 / 0 each | 2.1 k / 0 / 0 each |
| scratchpad banks | 10 | 48 URAM | 104 BRAM36 |
| `nanoxl_top` glue | 1 | 2.5 k / 0 / 0 | 2.5 k / 0 / 0 |

Biggest leaf modules in `vec_seq` at L = 4 (`python3 fpga/yosys/breakdown.py <work dir>/vec_seq.stat`), KV260:
`vec_seq` own sequencing 7.4 k LUT / 18.3 k FF; `vec_alu` 3.9 k; `layernorm` 3.8 k + 11 BRAM36; `pbuf_loader`
3.0 k; `lut_rom` ×22 1.2 k + 2.7 k LUTRAM; `softmax` 2.1 k; `requant` ×2 1.9 k; `srdhm` ×9 1.1 k / 36 DSP.
In `ctrl_top`: `sem_unit` 3.1 k LUT / 1.5 k MUXF.

### 2.1 What the numbers say (configuration sweep, ADR-006)

**With C1–C9 merged, the KV260 holds an N = 8 / L = 4 NanoXL with margin. N = 16 is borderline by Yosys,
and N = 32 or L ≥ 16 do not fit. No useful configuration fits the PYNQ-Z2.** KV260 sweep (Yosys LUT incl.
LUTRAM; shakespeare decode W8 / W4 and 256-token prefill in the cycle model at 250 MHz, DDR KV; the full
table with FF / DSP / BRAM / URAM is in ADR-006):

| N, L, VP, NB | LUT | % of 117 k | DSP | shakespeare tok/s W8 / W4 | prefill ms |
|---|---|---|---|---|---|
| 32, 64, 4, 8 (SG9a, before C1–C9) | 1,767 k | 1509 % | 2,658 | 607 / 644 | 15.3 |
| 32, 16, 4, 8 | > 198 k (routers + vec_seq alone) | > 169 % | | 607 / 642 | 17.2 |
| 16, 16, 4, 8 | 228.5 k | 195 % | 448 | 344 / 325 | 51.0 |
| 16, 8, 1, 4 | 116.9 k | 99.9 % | 328 | 342 / 323 | 55.2 |
| 16, 4, 1, 4 (step-up) | 97.0 k | 82.8 % | 244 | 340 / 321 | 60.7 |
| 8, 8, 1, 2 | 93.5 k | 79.8 % | 224 | 175 / 156 | 193.3 |
| **8, 4, 1, 2 (built; 200 MHz: 139 / 125 tok/s, 248 ms)** | **72.8 k** | **62.2 %** | 140 | 174 / 156 | 198.8 |

1. **N sets decode speed; L and VP barely matter.** With one weight row per cycle into the array, decode is
   weight-port bound: about 340 tok/s at N = 16, 175 at N = 8. Going from L = 16 to 4 costs 19 % of
   prefill and < 1 % of decode.
2. **Area is `vec_seq` (scales with L) plus the five routers (scale with NB × the widest port).** `vec_seq`
   is 96 k / 55 k / 35 k LUT at L = 16 / 8 / 4. One router is 20.4 k at N = 32 (NB ≥ 8 forced by the 8-line
   ACC port), 18.7 k at N = 16 / VP = 4 / NB = 8, 4.9 k at NB = 4 / VP = 1 and 3.0 k at N = 8 / NB = 2.
3. **DSP, BRAM and URAM are no longer constraints** on the KV260 (≤ 36 % DSP in every row). LUTs are.
4. **PYNQ-Z2**: the N = 8 / L = 4 build is 123 % of the XC7Z020's LUTs. Its floor (vec_seq ~31 k + ctrl_top
   ~9 k + routers ~10 k + glue) is already near 100 %. It needs the RTL work listed in ADR-006 ("Step-up").

### 2.2 Yosys caveats

* **Yosys is not Vivado.** LUT counts from `synth_xilinx` are typically 10-40 % above Vivado's on
  arithmetic, and Yosys does not do Vivado's cross-boundary optimization (blocks are summed, the glue is
  synthesized against black boxes). DSP inference differs: Yosys maps each `$mul` to DSP48E2/E1 tiles on its
  own rules (no 2-MAC INT8 packing, no SIMD); Vivado's mapping of the same RTL will differ in both directions.
* **URAM:** Yosys infers URAM288 only with `-uram` and only for depth ≥ 2048 lines here (1024-deep banks map
  to RAMB36). Vivado does **not** infer URAM without `ram_style = "ultra"`; `nanoxl_top RAM_STYLE = 2` (KV260)
  sets it in `spad_bank` on the ABUF / ACC / KVBUF banks (SG11b; the former XDC patterns are gone).
* `fpga/yosys/prep.py` relabels unnamed `begin` blocks and rewrites `int'(x)` / `signed'(x)` / `~W'(x)`
  casts in synthesis-only copies because Yosys 0.33's SystemVerilog frontend rejects them; the logic is
  unchanged. Vivado reads the original files.

### 2.3 Change list for the RTL owners (C1–C9 all done; next candidates: ADR-006 "Step-up")

Ordered by resources saved. None changes what the compiler emits or any numeric result; each is an
internal re-implementation the owner's existing tests cover. Estimates are rough (from the Yosys counts).

| # | owner / file | change | saves (KV260) |
|---|---|---|---|
| C1 | SG5 `rtl/lut_rom.sv` | **Done (SG11a).** The table is stored as `2^AW / WN` lines of `WN·DW` bits (one write port writing one line per LUT_LOAD beat, one asynchronous line read), the entry selected by the low address bits in front of the `DW`-bit read register (selecting after a registered line read costs `WN·DW` FFs per table and lets Yosys put the 2048-entry GELU tables into 2 BRAM36 each). Per table: 256 × uint16 1,418 LUT / 4,112 FF → 48 LUT + 10 RAM32M16 / 16 FF; GELU (8 slots) 6,490 / 16,392 → 90 LUT + 38 RAM64M8 / 8 FF | measured with C3/C6 (table below): `sfu` L = 64 **−820 k LUT, −2.14 M FF**; L = 16 −198 k LUT, −535 k FF |
| C2 | SG7c `rtl/buf_spad.sv` | **done (SG11b).** Arbitration is now NR (NW) static priority chains over a pairwise bank-conflict matrix, selected by the rotation counter; per-bank one-hot AND-OR muxes; one barrel rotate per wide port (writes onto the banks, reads back into address order); no run-time index anywhere. Same grants cycle for cycle (tiny top-level cycle counts unchanged). Per-buffer `ABUF_NB` … `KV_NB` and `RAM_STYLE` (`spad_bank`, set per board in `fpga/boards.py`) replace the XDC RAM patterns. Measured router (banks black-boxed): KV260 95.1 k LUT / 33.1 k MUXF / 10 DSP → 20.1-20.7 k LUT / 3.1-4.3 k MUXF / 0 DSP per buffer; PYNQ-Z2 39.3 k / 10 DSP → 18.0-18.2 k / 0 | **−374 k LUT, −50 DSP** (KV260, ×5); −106 k LUT, −50 DSP (PYNQ-Z2) |
| C3 | SG5 `rtl/sfu_elem.sv` | **Done (SG11a).** EXP2 / RECIP / RSQRT run on `EL` shared primitive lanes (`sfu`: `EL = min(L, 8)`); a beat is split into L/EL sub-beats through a shift register and reassembled in the output register. GELU and pass-through stay at L lanes, 1 beat/cycle. Elementwise EXP2 / RECIP / RSQRT (never emitted by the compiler) now take L/EL cycles per beat (8 at L = 64, 2 at L = 16) and L/EL − 1 cycles more latency; results unchanged (docs/sfu.md §3) | `sfu_elem` L = 64: 384 → 48 DSP (−336), 46.2 k → 14.7 k LUT after C1; L = 16: 96 → 48 DSP |
| C4 | SG3 `rtl/vec_alu.sv` | **Done (SG11c).** VEC MUL is a multi-cycle 17×17 multiplier per lane (4 partial products, registered operands and product, serial rounding shift): 1 beat per 7 + shamt cycles, latency 9 + shamt; every other op unchanged (1 beat/cycle, depth 3). Datapath narrowed from 64 to 35 bits (clamp35 saturates like the exact value), one shared 33-bit adder for ADD/SUB/MAX/MIN. Bit-identical (docs/vector-unit.md §3). Yosys KV260 L = 64: 104,030 → 62,928 LUT, 256 → 64 DSP; L = 16: 26,022 → 15,529 LUT, 64 → 16 DSP | done: −41 k LUT, −192 DSP (L = 64); −10.5 k LUT, −48 DSP (L = 16) |
| C5 | SG2 `rtl/pe.sv` / `systolic_array.sv` | **Done (SG11c).** WP486 packing: `pe_pair` (one DSP48E2 per two PEs; the column 2p+1 weight is written with column 2p and its product registered one cycle), `systolic_array #(.PACK(1))` default; exhaustive 2^24-case pair test. Yosys KV260 `systolic_array` N = 32: 44,540 / 88,782 / 1,024 → 47,612 / 97,742 / **512** (LUT / FF / DSP); N = 16: 11,260 / 21,870 / 256 → 12,028 / 24,054 / 128. No saving on the 7 series (DSP48E1 is 25×18): PYNQ-Z2 needs PACK = 0, which is not yet plumbed through array_ctrl / array_seq / nanoxl_top (ADR-006) | done: −512 DSP at N = 32 for +3 k LUT |
| C6 | SG3 `rtl/requant.sv` / SG5 `softmax.sv` | **Softmax side done (SG11a):** the score requant has `RQL` lanes (`sfu`: `SM_RQL = min(L, 16)`, the int32 rate of the 64-byte VEC A port); an int32 beat goes through as L/RQL sub-beats and is reassembled before the row buffer, results and clamp counts unchanged. At L = 64 SOFTMAX_ONLINE rows cost +3 cycles (PL = 1) / +3-6 (PL = 4) each (re-measured, `VEC_CAL[·][64]` refit; shakespeare KV260 prefill +1.2 %, docs/top.md §6). L ≤ 16 unchanged. VEC REQUANT (SG3) is not touched **Vec side done (SG11c):** `srdhm` stage 3 is `((p >>> 30) + 1) >>> 1` (exact identity, docs/vector-unit.md §2) and the requant clamp an equality test: `srdhm` 209 → 118 LUT, requant L = 64 30,349 → 23,185 LUT, L = 16 7,650 → 5,837 (also shrinks the softmax score requant and LN's srdhm). DSPs unchanged (4 per lane): a 32×32 product needs 4 tiles (each 27×18 tile can hold one corner of the square); the 3-DSP decomposition with a fabric corner is implemented and tested (`requant #(.NDSP(3))`) but costs ~175 LUT per DSP saved on the KV260 and saves nothing on the 7 series, so it is off.  | softmax: `softmax` L = 64: 324 → 132 DSP (−192), 67.7 k → 45.3 k LUT after C1; VEC REQUANT: −7 k LUT (L = 64), −1.8 k (L = 16); 0 DSP |
| C7 | SG3 `rtl/vec_reduce.sv` | **Done (SG11c).** Tree registered every 2 levels: 3 tree stages at L = 64, 2 at L = 16 (result latency 2 + NTS: 5 / 4). REDUCE is not in the ISA and vec_seq leaves the R port open, so no cycle-model change. Yosys KV260 L = 64: 23,121 → 12,028 LUT (the single-cycle compare tree became MUXF-heavy), FF +3.2 k | timing; −11 k LUT (L = 64) |
| C8 | SG7c `rtl/buf_spad.sv` / `spad_bank.sv` | **done (SG11b), off by default.** `nanoxl_top` / `buf_spad` / `spad_bank` parameter `OREG` = 1: unconditional second read register (URAM OREG / DOB_REG), read latency 2 on every port and on the null buffer. No requester needed a change; the visibility rule (docs/top.md §3) still holds and the buf_spad bench checks it at latency 2. Cost: +0.2 % cycles on tiny W8A8 (prefill 76,394 → 76,545, 8 decode steps 495,832 → 496,824), so the default stays 0 until timing needs it | timing (§5) |
| C9 | SG7b `rtl/vec_seq_rd.sv` (and `vec_seq.sv`) | **done (SG11b).** `vec_seq_rd`: one barrel shift of the assembly buffer by `sub << ebl` units of L bytes (log2(16P/L) fixed stages, none at L = 64), then fixed per-lane wiring per element size; chunk writes as a fixed decode. `vec_seq`: the REQUANT record buffer write lines are rotated into slots so every entry has one fixed write source (it was a PL-way mux per bit). Measured: `vec_seq_rd` 11.1 k → 1.06 k LUT (L = 16), 25.3 k → 3.97 k (L = 64), MUXF 6.4 k / 13.8 k → ~0.1 k; `vec_seq` own logic 43.8 k → 7.9 k LUT (L = 16), 44.3 k → 8.4 k (L = 64) | −56 k LUT at L = 16, −78 k at L = 64 (×2 readers + vec_seq) |

Measured C1 + C3 + C6 (SG11a; per module, `synth_xilinx -family xcup -uram`, hierarchical, same
pre-steps as the `vec_seq` job; `sfu` = `sfu_elem` + `softmax` + `layernorm` + routing with NG = 6, CTX_MAX =
256, D_MAX = 384, NSLOT = 16, NCTX = 8; `sfu_elem` / `softmax` rows at their module defaults EL = L / RQL = L
unless marked). LUT / FF / DSP (LUTRAM in LUT6 units listed separately):

| module | L = 64 before | L = 64 after | L = 16 before | L = 16 after |
|---|---|---|---|---|
| `lut_rom` 256 × uint16 (one table) | 1,418 / 4,112 / 0 | 48 + 10 LUTRAM / 16 / 0 | same | same |
| `lut_rom` GELU, 8 slots (one table) | 6,490 / 16,392 / 0 | 90 + 38 LUTRAM / 8 / 0 | same | same |
| `sfu_elem` (EL = L) | 722,305 / 1,861,958 / 384 | 46,210 / 26,956 / 384 | 180,577 / 465,494 / 96 | 11,906 / 6,748 / 96 |
| `sfu_elem` EL = 8 | – | 14,724 / 7,854 / **48** | – | 7,281 / 3,868 / **48** |
| `softmax` (RQL = L) | 156,749 / 290,800 / 324 | 67,729 / 24,560 / 324 | 40,540 / 76,000 / 84 | 17,235 / 6,368 / 84 |
| `softmax` RQL = 16 | – | 45,349 / 13,834 / **132** | (= RQL = L) | (= RQL = L) |
| `layernorm` | 56,328 / 17,882 / 264 | 55,019 / 13,786 / 264 | 15,650 / 8,042 / 72 | 14,331 / 3,946 / 72 |
| **`sfu`** (as built: EL = 8, SM_RQL = 16) | **937,572 / 2,170,823 / 972** | **117,136 / 35,657 / 444** | **237,346 / 549,719 / 252** | **39,392 / 14,365 / 204** |
| `sfu` LUTRAM / BRAM36 | 6,424 / 48 | 9,259 / 48 | 1,648 / 12 | 2,764 / 12 |

The `sfu` row is what moved inside `vec_seq`. `bench/results/resources_*.json` have been re-run since
with all of C1–C9 merged, for the ADR-006 configurations (§2, §2.1).

## 3. Block design and address map

```
 KV260 (zynq_ultra_ps_e_0)                                 PYNQ-Z2 (processing_system7_0)
   M_AXI_HPM0_FPD ─ smc_ctrl ─┬─ nanoxl_0/s_axi_ctrl          M_AXI_GP0 ─ smc_ctrl ─┬─ nanoxl_0/s_axi_ctrl
                              └─ axi_intc_0/s_axi                                  └─ axi_intc_0/s_axi
   S_AXI_HPC0_FPD (128 b) ─ smc_mf  ─ nanoxl_0/m_axi_mf       S_AXI_HP0 (64 b) ─ smc_hp0 (2:1) ─┬─ m_axi_mf
   S_AXI_HP0_FPD  (128 b) ─ smc_md  ─ nanoxl_0/m_axi_md                                        └─ m_axi_md
   S_AXI_HP1_FPD  (128 b) ─ smc_md1 ─ nanoxl_0/m_axi_md1      S_AXI_HP1 (64 b) ─ smc_md1 ─ m_axi_md1
   pl_ps_irq0 ─ axi_intc_0 ─ nanoxl_0/irq                     IRQ_F2P ─ axi_intc_0 ─ nanoxl_0/irq
   pl_clk0 200 MHz, proc_sys_reset_0 (pl_resetn0)             FCLK_CLK0 150 MHz, proc_sys_reset_0
```

| segment | KV260 | PYNQ-Z2 | size |
|---|---|---|---|
| `nanoxl_0/s_axi_ctrl` (CSRs, docs/control.md §5) | `0xA000_0000` | `0x4000_0000` | 4 KiB |
| `axi_intc_0/s_axi` | `0xA001_0000` | `0x4180_0000` | 4 KiB |
| `m_axi_mf` → DDR | HPC0 `HPC0_DDR_LOW` (0 – 2 GiB) | HP0 `HP0_DDR_LOWOCM` (0 – 512 MiB) | 32-bit master |
| `m_axi_md` → DDR | HP0 `HP0_DDR_LOW` | HP0 (shared with `mf`) | 32-bit master |
| `m_axi_md1` → DDR | HP1 `HP1_DDR_LOW` | HP1 `HP1_DDR_LOWOCM` | 32-bit master |

The port assignment is ADR-005 §4 (docs/control.md §4): fetch (`mf`), DMA channel 0 (`md`, ISA
`chan & 1 = 0`) and DMA channel 1 (`md1`, `chan & 1 = 1`) are separate masters. On the KV260 each DMA
channel has its own 128-bit HP port (2 × 4 GB/s at 250 MHz, the 8 GB/s ADR-003 assumes; 2 × 3.2 GB/s at the first bitstream's 200 MHz) and fetch uses
HPC0 (no coherency is requested: AxCACHE = 0011, the CMA buffer is uncached). On the PYNQ-Z2 fetch
(~1 % of the traffic) shares HP0 with DMA channel 0 through a 2:1 SmartConnect, and the SmartConnects
convert 128 → 64 bits. Everything the driver needs is in the `.hwh`: `PynqBackend` takes
`ip_dict["nanoxl_0"]["phys_addr"]` (any base works), allocates the image with
`pynq.allocate(..., cacheable=False)` and relocates every program to its physical address
(docs/driver.md).

## 4. Step by step

### 4.1 Install Vivado (x86-64 Linux or Windows, 16 GB RAM minimum, ~100 GB disk)

1. Download the AMD Unified Installer for **Vivado 2024.2** (account required, free) from
   https://www.xilinx.com/support/download.html.
2. Run it, choose **Vivado ML Standard**, and tick only the devices you need: *Zynq UltraScale+ MPSoC*
   (KV260) and/or *Zynq-7000* (PYNQ-Z2). No license file is needed for these devices.
3. `source /tools/Xilinx/Vivado/2024.2/settings64.sh` (adjust the path), check `vivado -version`.
4. Optional: the KV260 board files (Vivado → Tools → *Vivado Store* → Boards → *Kria KV260*). The script
   uses them when present and builds from the bare part otherwise (PYNQ loads the overlay on a booted
   system, so the PS presets do not matter for the overlay).

### 4.2 Build the bitstream

```
git clone <this repo> nanoxl && cd nanoxl
vivado -mode batch -source fpga/kv260/build.tcl  -tclargs -jobs 8      # ~1-3 h, 16-32 GB RAM
vivado -mode batch -source fpga/pynqz2/build.tcl -tclargs -jobs 8
```

Outputs (`fpga/<board>/build/`, git-ignored): `overlay/nanoxl.bit` + `overlay/nanoxl.hwh` (same base name,
what PYNQ needs) + `nanoxl.xsa`; `reports/timing_summary.rpt`, `utilization.rpt`, `utilization_hier.rpt`,
`ram_utilization.rpt`, `logic_levels.rpt`, `summary.txt` (WNS / WHS). The script prints
`NANOXL: WNS = ...` and a warning if timing is not met; it still writes the bitstream.

Options (after `-tclargs`): `-jobs N`, `-out DIR`, `-clk MHZ` (e.g. `-clk 200`), `-params "N 16 L 16"`
(quoted: one argument; any `nanoxl_top` parameter), `-strategy NAME` (default
`Performance_ExplorePostRoutePhysOpt`), `-proj_only` (stop after the block design: open
`build/nanoxl_<board>.xpr` in the GUI to inspect it), `-no_bit` (implementation and reports only).

**If you change `-params` or `-clk`, the host must know:** buffer capacities and N/L go into the
compiler via `fpga/boards.py`. Edit `RTL_PARAMS` / `CLOCK_MHZ` there and in `build.tcl` together
(`tests/fpga` fails while they disagree), rather than overriding on the command line, for anything you keep.

First things to check in the reports:
1. `ram_utilization.rpt` (KV260): 64 URAM288 (ABUF 16, ACC 16, KVBUF 32). If the URAM count is 0, the
   `RAM_STYLE` patterns in `nanoxl_kv260.xdc` did not match: look up the real cell names with
   `get_cells -hier -filter {NAME =~ *u_bank/mem_reg*}` in the synthesized design and fix the patterns.
2. `utilization.rpt` against §2, `timing_summary.rpt` against §5.

### 4.3 Prepare the board

KV260: flash *Ubuntu 22.04 for Kria* to a ≥ 32 GB microSD, boot, `sudo xmutil bootfw_update` if the Kria wiki
says so for that image, then
```
git clone --recursive https://github.com/Xilinx/Kria-PYNQ && cd Kria-PYNQ && sudo bash install.sh -b KV260
sudo xmutil unloadapp                 # free the PL from the default k26 app (each boot)
```
PYNQ-Z2: flash the PYNQ v3.0.1 image, boot, log in (xilinx/xilinx).

### 4.4 Copy and run

```
scp -r nanoxl  <board>:~/                                   # the repo: nanoxl/, fpga/, artifacts/models/tiny
scp fpga/kv260/build/overlay/nanoxl.{bit,hwh} <board>:~/nanoxl/
ssh <board>
sudo -i                                                     # PYNQ needs root for the PL and CMA
source /etc/profile.d/pynq_venv.sh && cd ~xilinx/nanoxl     # (KV260; on PYNQ-Z2 the venv is already active)
python3 fpga/first_light.py --board kv260 --backend pynq --bitstream nanoxl.bit
python3 fpga/run_board.py --board kv260 --backend pynq --bitstream nanoxl.bit \
    --bundle artifacts/models/tiny/w8a8.npz --tokens 12,0,0,19,30,17,25,21,27,10,0,19,53,53,42,1 --n 100 --json
python3 fpga/run_board.py --board kv260 --backend pynq --bitstream nanoxl.bit \
    --bundle artifacts/models/tiny/w8a8.npz --prompt "ROMEO:" --n 200
```

`fpga/run_board.py` is `python3 -m nanoxl.driver` (same options, docs/driver.md) with the as-built
configuration; plain `python3 -m nanoxl.driver --board kv260` uses the ADR-003 planning config, whose
PYNQ-Z2 ABUF (192 KiB) is larger than the bitstream's and would corrupt data. `--kv ddr` selects the
DDR-KV mode. If `pynq.allocate` fails, raise the CMA pool (`cma=512M` boot argument).

## 5. Timing: risks and mitigations

Nothing here has been timed. The candidates, in the order to look at them in `timing_summary.rpt`:

| # | path | why it is at risk | mitigation |
|---|---|---|---|
| T1 | `vec_alu` S1: operand → 64 × 64 `MUL` / add / compare mux → `w1` | the multiply has no input register and is 64 bits wide: four or more cascaded DSP48E2 plus the op mux in one cycle; cannot meet 4 ns | C4 (narrow, register operands). Vivado retiming (enabled in the script) cannot move registers into the multiply without an input register |
| T2 | `vec_reduce` S2: 64-leaf (KV260) / 16-leaf balanced tree of 48-bit adders and comparators in one cycle | 6 levels of 48-bit carry chains plus routing across 64 lanes ≈ 5-7 ns | C7 (2-3 stages). The PYNQ-Z2 (16 leaves, 6.7 ns) is likely fine |
| T3 | `buf_spad` arbitration → `ready` (combinational grant) → requesters' `valid`/state, and → bank address/data mux | 8 read + 4 write ports, rotating priority, dynamic indexing; the grant feeds back into `array_seq_rd`, `vec_seq_rd/wr` and the DMA in the same cycle (docs/top.md §3: `ready` = grant) | C2; then register the bank-side write path (one more cycle of write latency, visibility rule re-check) |
| T4 | `srdhm` 32 × 32 → 64 (requant, softmax, LN; 129 instances on the KV260) | the product is split over 4 DSP48E2 (27 × 18) with one internal register (stage 2 = P register); at 250 MHz a 4-DSP cascade usually needs MREG + PREG and a cascade register | Vivado retiming may pull stage 2 into the DSPs; if not, add one pipeline stage in `srdhm` (requant depth 6 → 7; requant's in/out handshake absorbs it) |
| T5 | whole-pipeline stall enables: `in_ready = !out_valid \|\| out_ready` chained through `vec_unit` / `requant` / `vec_reduce` / `sfu`, and the `adv` enable fanning out to every pipeline flop of 64 lanes | a combinational ready chain across several modules plus a fan-out of thousands | skid buffers at module boundaries; let `phys_opt_design` replicate (the default strategy here does); `max_fanout` on `adv` |
| T6 | URAM288 read (`spad_bank`, ABUF/ACC/KVBUF on the KV260) → router read-data mux → requester | 1-cycle registered read without the URAM output register; KVBUF banks are 8192 deep = 2 URAMs cascaded in depth; then the 8-port return mux | C8 (OREG, latency 2); or keep KVBUF/ACC banks ≤ 4096 deep (more banks) |
| T7 | `array_ctrl` / `array_seq` ready chains (`cmd_ready` combinational on `w_valid`, docs/top.md §4) and the 1024-bit accumulator RMW mux | known combinational handshakes from SG2; wide (32 × 32-bit) muxes | usually fine at 250 MHz after placement; check `logic_levels.rpt` |
| T8 | AXI SmartConnects (clock = core clock), HPC0/HP width | low risk: SmartConnect registers its slices | none needed; at `-clk` above 250 MHz use an `axi_register_slice` |

Also check, whatever the slack: **URAM read/write collisions.** `spad_bank` returns the *old* line when a
read and a write of the same line are granted at the same edge (docs/top.md §3). Vivado maps a simple
dual-port URAM so that port A's operation precedes port B's; it must put the read on port A to keep that
behaviour. Look for collision warnings in the synthesis log, and the DMA loopback (§6 step 3) plus the
single-layer step catch a violation. The PYNQ-Z2 at 150 MHz on a -1 Zynq-7020 has the same list
(T1 and T4 are the likely failures; DSP48E1 is 25 × 18, so 32 × 32 needs 4 of them too).

The KV260 first bitstream already runs at 200 MHz (ADR-006); `-clk 250` is the step-up once this list is clear. Fallbacks without RTL changes: `-clk 150` (KV260) / `-clk 125` (PYNQ-Z2) and keep `CLOCK_MHZ` in
`fpga/boards.py` in step (the cycle-model numbers scale with the clock); `-strategy
Performance_ExtraTimingOpt`; for resources, the first bitstream is already the N = 8 / L = 4 build (ADR-006).

## 6. First-light checklist

`python3 fpga/first_light.py --board <b> --backend pynq --bitstream nanoxl.bit` runs these in order and
stops at the first failure (the same script runs against the emulator in `tests/fpga`):

| # | step | pass criterion | if it fails |
|---|---|---|---|
| 1 | **CSR** | ID = `0x4E584C43` ("NXLC"), VERSION = `0x00010000`, RING_BASE / RING_SIZE / IRQ_EN read back, STATUS ring-cfg-ok, ERR 0 | ID = 0 or `0xFFFFFFFF`: wrong `--ip` / base address (check `Overlay.ip_dict`), PL not loaded, or the core held in reset (`aresetn`) |
| 2 | **END** | one-word END program: END_COUNT +1, IRQ_STATUS bit 0, W1C clears it | RING_HEAD stuck at 0: fetch master cannot reach DDR (HPC0 / HP0 address map, AxCACHE, CMA physical address) |
| 3 | **DMA loopback** | 16 KiB random bytes DMA_LOAD → each of ABUF / WBUF / ACC / PBUF / KVBUF (top of the buffer) → DMA_STORE: unchanged, with loads and stores on both DMA channels | ERR 5: BRESP/RRESP error (HP0 / HP1 map of `md` / `md1`); data mismatch in one buffer: its bank mapping / `RAM_STYLE` / URAM read-collision behaviour |
| 4 | **DDR bandwidth** | 1 MiB loads on channel 0, channel 1 and both at once, one 1 MiB store; GB/s from PERF_CYCLES at the board clock. Ceiling per channel on the KV260: 128 b × the clock, i.e. 3.2 GB/s at 200 MHz (4 GB/s at 250), so up to 6.4 (8) GB/s with both; PYNQ-Z2 1.2 GB/s per 64-bit HP port at 150 MHz, both together bounded by its ~1.5 GB/s DDR. Record it: ADR-003's largest unknown | far below: SmartConnect width/clock conversion, HP port width not 128 b; both-channel number ≈ one-channel number: the two masters land on the same port |
| 5 | **single layer** | 4-token prefill of a 1-layer cut of tiny W8A8: every logit == `nanoxl.golden` (exercises GEMM, REQUANT, LN, softmax, GELU, KV_APPEND) | reproduce in the SimBackend and Verilator (docs/driver.md §3.6), single-step with CTRL bit 0 |
| 6 | **full tiny generate** | reference prompt + 32 greedy tokens == `artifacts/models/tiny/golden_ref.json` | compare per-step logits with `Golden.forward`; SAT_* with golden saturation counts |

Afterwards: `PERF_*` per step vs `nanoxl.cyclemodel` at the board's DDR latency (M7), tokens/s
(`--json` output, `host_seconds`), then W4A8.

## 7. Verification here (`python3 -m pytest -q tests/fpga`)

| test | what |
|---|---|
| `test_build_tcl_dry_run[kv260, pynq-z2]` | each `build.tcl` runs to the end in tclsh with every Vivado command stubbed (unknown commands are errors); checks the `nanoxl_0` CONFIG == `fpga/boards.py`, part, clock, CSR base 4 KiB, all three masters → SmartConnect → the HP port of ADR-005 → DDR segment, irq/clock/reset pins, bitstream + `.bit`/`.hwh` export, RTL list |
| `test_build_tcl_options` | `-clk`, `-params`, `-proj_only`; a bad option fails |
| `test_rtl_fileset_matches_hierarchy[...]` | `fpga/common/rtl_files.tcl` exists file by file and equals the files of every module in Yosys's elaborated hierarchy of `nanoxl_axi_wrap` (with each board's parameters), and equals the simulations' `TOP_SRC` |
| `test_wrapper_ports_follow_vivado_axi_naming` | every `nanoxl_top` port is in the wrapper, renamed, same direction and width, and the file equals `fpga/gen_wrap.py`'s output; `<intf>_<signal>` names with the full AXI4-Lite / AXI4 signal sets, master/slave directions, `X_INTERFACE_INFO` / `X_INTERFACE_PARAMETER` (clock ASSOCIATED_BUSIF, active-low reset, AXI4LITE / READ_ONLY); no SystemVerilog in the shell; same parameters |
| `test_estimate_mirrors_top_ports` | the scratchpad port lists the estimate script synthesizes (`RPL`, `WPLV`) equal the ones `nanoxl_top` instantiates, so a port added in the RTL cannot silently drop out of the estimate |
| `test_wrapper_lint` | Verilator `-Wall` lint of the wrapper top: no message in the wrapper |
| `test_wrap_smoke` | **behaviour**: the unmodified `nanoxl_top` smoke bench (`top_cocotb.top_tiny`, real driver over AXI, 1-layer tiny W8A8, logits / KV cache / per-op saturation == golden) through the wrapper's port names (Icarus ~90 s) |
| `test_hwconfig_fits_rtl_and_runs_tiny[...]` | as-built `HwConfig` within the RTL capacities (powers of two, 17-bit line addresses), tiny generates == golden on it (emulator) |
| `test_first_light_sim[...]`, `test_first_light_reports_failure`, `test_run_board_cli[...]` | the board tools against the emulator, and a corrupted loopback is reported as FAIL |
| `test_resource_estimates_current[...]` | `bench/results/resources_*.json` exist for today's `RTL_PARAMS`, contain every block, and record no failed job |
| `test_hwconfig_boards_mirror_fpga[...]` | `nanoxl.hwconfig.BOARDS["<board>-fpga"]` == `fpga/boards.py:hwconfig(board)` (ADR-006) |
| `test_board_rtl_tiny[...]` (slow, Verilator) | `nanoxl_top` with exactly `RTL_PARAMS["kv260"]` runs the tiny headline test (16-token prompt + 8 decode steps, W8A8 / W4A8, on-chip / DDR KV) bit-exact vs golden |

Mutation check: swapping two bytes of `m_axi_md_rdata` inside the wrapper makes `test_wrap_smoke` fail
(`step 0: logits differ at 130 entries`); tying `md_axi_rlast` to `rvalid` hangs it.
