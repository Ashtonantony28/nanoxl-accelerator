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
| `fpga/kv260/nanoxl_kv260.xdc`, `fpga/pynqz2/nanoxl_pynqz2.xdc` | `RAM_STYLE` for the scratchpad banks (URAM on the KV260); clocks come from the PS IP |
| `fpga/yosys/estimate.py`, `fpga/yosys/prep.py`, `fpga/yosys/breakdown.py` | Yosys resource estimates → `bench/results/resources_<board>.json`; `breakdown.py` prints a per-module table (own cells × instances) from a job's `.stat` |
| `fpga/first_light.py` | first-light checklist (§6), board or emulator |
| `fpga/run_board.py` | `python3 -m nanoxl.driver` with the as-built `HwConfig` (use this on the board, §4) |
| `tests/fpga/` | static checks + wrapper smoke (§7) |

## 1. Board configurations (as built)

`nanoxl_top` needs power-of-two buffer capacities (line addresses wrap), and the banks must land on
whole BRAM/URAM primitives, so the bitstreams differ from the ADR-003 planning numbers. The compiler
must never use more than the RTL has; `fpga/boards.py:hwconfig()` gives the matching `HwConfig`
(structure from the RTL, timing assumptions from ADR-003, 1-row weight port as built).

| | KV260 (XCK26, `xck26-sfvc784-2LV-c`) | PYNQ-Z2 (`xc7z020clg400-1`) |
|---|---|---|
| clock (pl_clk0 / FCLK_CLK0) | 250 MHz | 150 MHz requested (the PS7 PLLs give the nearest value) |
| array `N`, lanes `L` | 32, 64 | 16, 16 |
| `NB` banks / buffer | **8** (= widest port; auto would be 16) | 8 (auto) |
| ABUF | 512 KiB, URAM (8 × 4096 lines = 2 URAM288 per bank) | 128 KiB, BRAM |
| WBUF | 128 KiB, BRAM | 64 KiB, BRAM |
| ACC | 256 KiB, URAM (8 × 2048 lines) | 64 KiB, BRAM |
| PBUF | 64 KiB, BRAM | 32 KiB, BRAM |
| KVBUF | **1 MiB** URAM (ADR-003: 1.5 MiB, not a power of two) | 128 KiB (tiny on chip; shakespeare: DDR-KV mode) |
| URAM / BRAM36 for scratchpads | 64 / 48 of 64 / 144 | 0 / 104 of 0 / 140 |
| `NG` (GELU slots) | 6 (shakespeare) | 6 |

Why `NB = 8` on the KV260: with the automatic 16 banks every URAM bank would be 2048 lines deep,
half of a 4096-deep URAM288, and ABUF + KVBUF alone would need all 64 URAMs. With 8 banks ABUF, ACC and
KVBUF fit exactly into 64. Cost: the array ACC port (8 lines/beat) then covers every bank, so a VEC read
of ACC and an array ACC beat in the same cycle always conflict (arbitration still bounds the wait;
functionally identical, docs/top.md §3). If ACC goes to BRAM instead (64 RAMB36), `NB = 16` is possible
for ACC only after an RTL change (per-buffer `NB`, change C2 in §2.3).

KV260 `shakespeare` with on-chip KV needs 1.18 MB > 1 MiB: run it with `--kv ddr` (ADR-003 batch > 1
mode; ~5 % slower decode in the model). `tiny` fits on chip on both boards.

## 2. Resource estimates (Yosys)

`python3 fpga/yosys/estimate.py kv260` / `pynq-z2` (Yosys 0.33, `synth_xilinx -family xcup -uram` /
`-family xc7`). A whole-design flat run did not finish within 10 minutes even at N = 8, so the script
synthesizes per block and sums (each block hierarchically, so 1024 PEs cost one PE synthesis): `ctrl_top`,
`array_seq`, `vec_seq`, the scratchpad router (`buf_spad` with its banks as black boxes, counted 5 times:
the five instances differ only in row-address width), one `spad_bank` per distinct depth (counted
`NB` × buffers), and the `nanoxl_top` glue (blocks as black boxes). Results with per-block detail:
`bench/results/resources_kv260.json`, `resources_pynq-z2.json`.

Totals against the device (Yosys `INV` cells, which Vivado folds into LUTs, are left out; the PS7 /
PS8 and the SmartConnect / interrupt IPs are not included; they add roughly 5-10 k LUTs):

| resource | KV260 (XCK26) used / avail | % | PYNQ-Z2 (XC7Z020) used / avail | % |
|---|---|---|---|---|
| LUT | 1,767,319 / 117,120 | **1509 %** | 558,168 / 53,200 | **1049 %** |
| FF | 2,356,120 / 234,240 | **1006 %** | 605,464 / 106,400 | **569 %** |
| DSP | 2,658 / 1,248 | **213 %** | 722 / 220 | **328 %** |
| BRAM36 (RAMB18 = ½) | 114.5 / 144 | 80 % | 128 / 140 | 91 % |
| URAM288 | 64 / 64 | 100 % | – | – |
| Yosys CPU time, sum of jobs (`--jobs 1`) | 4,470 s (vec_seq alone 3,731 s) | | 1,505 s (vec_seq 1,210 s) | |

Per block (LUT / FF / DSP / memory; count = instances in the sum):

| block | count | KV260 (N = 32, L = 64) | PYNQ-Z2 (N = 16, L = 16) |
|---|---|---|---|
| `ctrl_top` (cmd fetch, CSRs, DMA ×2) | 1 | 8.5 k / 2.6 k / 0 / 4 BRAM36 | 8.0 k / 2.6 k / 0 / 4 BRAM36 |
| `array_seq` (array, array_ctrl, reader/writer) | 1 | 61 k / 112 k / 1,057 | 10.4 k / 21 k / 273 |
| array_ctrl memories (acc ram, out / gs FIFOs) | 3 | 2.6 k / 3.9 k / 0 / 29 BRAM18 | 1.3 k / 2.5 k / 0 / LUTRAM |
| `vec_seq` (vector unit, SFUs, LN, softmax, PBUF loader) | 1 | **1,211 k / 2,238 k / 1,551** / 48 BRAM36 | **336 k / 579 k / 399** / 12 BRAM36 |
| of which `lut_rom` (exp2 / recip / rsqrt / GELU tables) | 322 / 82 | **781 k / 2,110 k** / 0 | **193 k / 534 k** / 0 |
| scratchpad router (`buf_spad` minus banks) | 5 | 95 k / 0 / 10 each | 39 k / 0 / 10 each |
| scratchpad banks | 40 / 40 | 48 BRAM36 + 64 URAM | 112 BRAM36 |
| `nanoxl_top` glue | 1 | 6.8 k / 0 / 0 | 5.3 k / 0 / 0 |

Biggest leaf modules in `vec_seq` (`python3 fpga/yosys/breakdown.py sim_build/fpga/<board>/vec_seq.stat`),
KV260: `lut_rom` ×322 781 k LUT / 2.11 M FF; `vec_alu` 104 k LUT / 256 DSP; `layernorm` 54 k / 257 DSP;
`vec_seq_rd` ×2 50 k; `requant` ×2 34 k; `srdhm` ×129 516 DSP; `softmax` 32 k / 65 DSP; `vec_reduce` 23 k /
64 DSP; `sfu_rsqrt` ×65 and `sfu_recip` ×65 195 DSP each. In `array_seq`: 1,024 `pe` (1 DSP, 58 FF each)
and 30 k FF of skew `delay_pipe`.

### 2.1 What the numbers say

**Neither configuration fits, by a wide margin, and memory is not the problem.** BRAM / URAM are within
budget on both boards (the buffer sizes of §1 were chosen for that). LUT, FF and DSP are 2-15× over.
The causes are concentrated:

1. **`lut_rom` is 44 % of the KV260 LUTs and 90 % of its FFs** (C1). The SFU tables are written several
   entries per cycle into an entry-wide array, which no RAM primitive implements, so every table becomes
   registers plus a 256:1 mux, and there are 322 of them (one per lane per function). Re-organized as lines
   they are LUTRAM: −740 k LUT, −2.1 M FF on the KV260, −183 k LUT, −534 k FF on the PYNQ-Z2.
2. **The scratchpad routers** (C2): 5 × 95 k LUT and 50 DSPs that a static crossbar does not need.
   −400 k LUT (KV260), −170 k (PYNQ-Z2).
3. **DSPs**: elementwise RECIP / RSQRT at full width (390 DSP, never emitted by the compiler: C3), VEC MUL
   64 × 64 (C4), 1 DSP per PE without INT8 packing (C5), two 64-lane requantizers (C6).
4. **The vector width.** Everything in `vec_seq` scales with `L`: at L = 16 (PYNQ-Z2 build) `vec_seq` minus
   its tables is 143 k LUT; at L = 64 it is 430 k.

Arithmetic, KV260 (LUT; FF and BRAM then fit, DSP noted):

| configuration | LUT | % of 117 k | DSP |
|---|---|---|---|
| as built (N = 32, L = 64) | 1,767 k | 1509 % | 2,658 |
| + C1 + C2 | ~630 k | ~540 % | ~2,600 |
| + C3 + C4 (C1-C6) | ~540 k | ~460 % | ~1,450 |
| N = 32, **L = 16**, C1 + C2 (vec_seq from the PYNQ run, routers ~12 k each after C2) | ~280 k | ~240 % | ~1,450 |
| + C9 (`vec_seq_rd` / `vec_seq` selects, ~50 k at L = 16) + C3/C4 at L = 16 | ~220 k | ~190 % | ~1,200 |
| … and N = 16 (array 10 k instead of 61 k) | ~170 k | ~145 % | ~420 |

PYNQ-Z2 with C1 + C2: ~205 k LUT (385 % of 53 k), FF ~71 k (67 %), DSP ~670 (C5 and C6 bring it to ~480,
still 2×).

So **the as-designed NanoXL does not fit either board even after the cheap RTL fixes**, by Yosys's count.
Two qualifications before acting on it: Yosys over-counts LUTs on wide muxes and arithmetic by 10-40 %
against Vivado (§2.2), and it does not share logic across the summed blocks; and the remaining ~140 k LUT of
`vec_seq` at L = 16 has not been broken down to the RTL line yet (C9 is the first, visible part). The
realistic plan is:

* **Do C1 and C2 first** (they are the bulk, they are internal re-implementations, and they change no
  result), then re-run `estimate.py` (cached jobs are skipped; only changed blocks re-synthesize).
* **Then run Vivado synthesis (not implementation) on the KV260 with `-params "L 16"`** and
  `-proj_only`/`launch_runs synth_1` to get the real LUT count; Vivado's number decides between "N = 32,
  L = 16" and "N = 16, L = 16" for the first bitstream. `L` and `N` are parameters; the compiler takes the
  matching `HwConfig` from `fpga/boards.py`.
* **The PYNQ-Z2 is not a viable first target** for the full datapath: even at ~150 k LUT after all of C1-C9
  it is 3× the XC7Z020. It stays in the flow (scripts, tests) for a reduced build (e.g. N = 8, L = 8) once
  the KV260 numbers are known.
* DSP: on the KV260 at L = 16 the DSP budget is met with C3-C6; at L = 64 even C3-C6 together (~1,450) leave it over.

### 2.2 Yosys caveats

* **Yosys is not Vivado.** LUT counts from `synth_xilinx` are typically 10-40 % above Vivado's on
  arithmetic, and Yosys does not do Vivado's cross-boundary optimization (blocks are summed, the glue is
  synthesized against black boxes). DSP inference differs: Yosys maps each `$mul` to DSP48E2/E1 tiles on its
  own rules (no 2-MAC INT8 packing, no SIMD); Vivado's mapping of the same RTL will differ in both directions.
* **URAM:** Yosys infers URAM288 only with `-uram` and only for depth ≥ 2048 lines here (1024-deep banks map
  to RAMB36). Vivado does **not** infer URAM without `RAM_STYLE = "ultra"`; the KV260 XDC sets it on the
  ABUF / ACC / KVBUF banks by hierarchical name, which needs global synthesis (the build script sets
  `synth_checkpoint_mode None`).
* `fpga/yosys/prep.py` relabels unnamed `begin` blocks and rewrites `int'(x)` / `signed'(x)` / `~W'(x)`
  casts in synthesis-only copies because Yosys 0.33's SystemVerilog frontend rejects them; the logic is
  unchanged. Vivado reads the original files.

### 2.3 Change list for the RTL owners (not applied: outside SG9a's paths)

Ordered by resources saved. None changes what the compiler emits or any numeric result; each is an
internal re-implementation the owner's existing tests cover. Estimates are rough (from the Yosys counts).

| # | owner / file | change | saves (KV260) |
|---|---|---|---|
| C1 | SG5 `rtl/lut_rom.sv` | store the table as `2^AW / WN` lines of `WN·DW` bits (one write port, one line per LUT_LOAD beat) and select the entry after the registered line read. Today `WN` = 8 / 16 entries are written per cycle into a `DW`-wide array, a multi-write-port memory that no RAM primitive implements, so all 322 tables (exp2 ×128, recip ×65, rsqrt ×65, GELU ×64) become flip-flops plus 256:1 read muxes. As lines they map to LUTRAM (~10 RAM32M16 + a 16-bit 8:1 select, ~120 LUTs per table) | **~2.1 M FF, ~740 k LUT** |
| C2 | SG7c `rtl/buf_spad.sv` | rewrite the arbitration as `NR` precomputed static priority orders selected by the rotation counter, with per-bank one-hot AND-OR muxes and a barrel rotate per wide port, instead of `(rr + k) % NR` dynamic indices and dynamic part selects (`w_data[p*PMAX*128 +: 128]`, `b_wdata[bank_of(...)]`), which Yosys turns into 32-bit multipliers and shifters (10 DSPs and ~95 k LUTs per buffer, ×5). A crossbar of 8 read / 4 write ports onto 8 banks is ~10-15 k LUTs per buffer. Also: a per-buffer `NB` and a `RAM_STYLE` parameter on `spad_bank` (then the XDC patterns of §1 are not needed) | ~400 k LUT, 50 DSP |
| C3 | SG5 `rtl/sfu_elem.sv` (+ `sfu_recip`, `sfu_rsqrt`, `sfu_exp2`) | time-multiplex the elementwise RECIP / RSQRT / EXP2 lanes (e.g. 8 per cycle instead of L = 64). The compiler emits **no** elementwise RECIP or RSQRT at all (LN uses one rsqrt per row, softmax one recip per row), so their throughput is irrelevant to every compiled model (ADR-003 lists them as the first reduction) | ~340 DSP, ~25 k LUT |
| C4 | SG3 `rtl/vec_alu.sv` | **Done (SG11c).** VEC MUL is a multi-cycle 17×17 multiplier per lane (4 partial products, registered operands and product, serial rounding shift): 1 beat per 7 + shamt cycles, latency 9 + shamt; every other op unchanged (1 beat/cycle, depth 3). Datapath narrowed from 64 to 35 bits (clamp35 saturates like the exact value), one shared 33-bit adder for ADD/SUB/MAX/MIN. Bit-identical (docs/vector-unit.md §3). Yosys KV260 L = 64: 104,030 → 62,928 LUT, 256 → 64 DSP; L = 16: 26,022 → 15,529 LUT, 64 → 16 DSP | done: −41 k LUT, −192 DSP (L = 64); −10.5 k LUT, −48 DSP (L = 16) |
| C5 | SG2 `rtl/pe.sv` / `systolic_array.sv` | **Done (SG11c).** WP486 packing: `pe_pair` (one DSP48E2 per two PEs; the column 2p+1 weight is written with column 2p and its product registered one cycle), `systolic_array #(.PACK(1))` default; exhaustive 2^24-case pair test. Yosys KV260 `systolic_array` N = 32: 44,540 / 88,782 / 1,024 → 47,612 / 97,742 / **512** (LUT / FF / DSP); N = 16: 11,260 / 21,870 / 256 → 12,028 / 24,054 / 128. No saving on the 7 series (DSP48E1 is 25×18): PYNQ-Z2 needs PACK = 0, which is not yet plumbed through array_ctrl / array_seq / nanoxl_top (ADR-006) | done: −512 DSP at N = 32 for +3 k LUT |
| C6 | SG3 `rtl/requant.sv` / SG5 `softmax.sv` | **Vec side done (SG11c):** `srdhm` stage 3 is `((p >>> 30) + 1) >>> 1` (exact identity, docs/vector-unit.md §2) and the requant clamp an equality test: `srdhm` 209 → 118 LUT, requant L = 64 30,349 → 23,185 LUT, L = 16 7,650 → 5,837 (also shrinks the softmax score requant and LN's srdhm). DSPs unchanged (4 per lane): a 32×32 product needs 4 tiles (each 27×18 tile can hold one corner of the square); the 3-DSP decomposition with a fabric corner is implemented and tested (`requant #(.NDSP(3))`) but costs ~175 LUT per DSP saved on the KV260 and saves nothing on the 7 series, so it is off. Score-requant side: SG11a | −7 k LUT (L = 64), −1.8 k (L = 16); 0 DSP |
| C7 | SG3 `rtl/vec_reduce.sv` | **Done (SG11c).** Tree registered every 2 levels: 3 tree stages at L = 64, 2 at L = 16 (result latency 2 + NTS: 5 / 4). REDUCE is not in the ISA and vec_seq leaves the R port open, so no cycle-model change. Yosys KV260 L = 64: 23,121 → 12,028 LUT (the single-cycle compare tree became MUXF-heavy), FF +3.2 k | timing; −11 k LUT (L = 64) |
| C8 | SG7c `rtl/buf_spad.sv` / `spad_bank.sv` | optional second read register (URAM OREG) behind a parameter, with the read latency of the ports raised to 2 (the requesters' credit counters already allow it; the visibility rule of docs/top.md §3 must be re-checked) | timing (§5) |
| C9 | SG7b `rtl/vec_seq_rd.sv` (and `vec_seq.sv`) | the lane unpack selects each lane's element with `asm_b[(e % AE)*8 +: 8]` per lane and element size: an `AE`-byte mux per lane (×L, ×2 instances). The element offsets of consecutive lanes are consecutive, so a single barrel rotate of the assembly buffer by the start offset followed by fixed wiring per lane gives the same result. Same pattern for `vec_seq`'s own row assembly (44 k LUT at L = 16) | ~35 k LUT at L = 16, ~100 k at L = 64 (estimated) |

With C1-C6 the KV260 estimate drops to roughly 0.54 M LUT, still 4.6× the XCK26; with L = 16 and C1, C2,
C9 to ~220 k (§2.1 has the table). **The first KV260 bitstream therefore needs L = 16 and probably N = 16,
to be decided by a Vivado synthesis run after C1 and C2.**

## 3. Block design and address map

```
 KV260 (zynq_ultra_ps_e_0)                                 PYNQ-Z2 (processing_system7_0)
   M_AXI_HPM0_FPD ─ smc_ctrl ─┬─ nanoxl_0/s_axi_ctrl          M_AXI_GP0 ─ smc_ctrl ─┬─ nanoxl_0/s_axi_ctrl
                              └─ axi_intc_0/s_axi                                  └─ axi_intc_0/s_axi
   S_AXI_HPC0_FPD (128 b) ─ smc_mf  ─ nanoxl_0/m_axi_mf       S_AXI_HP0 (64 b) ─ smc_hp0 (2:1) ─┬─ m_axi_mf
   S_AXI_HP0_FPD  (128 b) ─ smc_md  ─ nanoxl_0/m_axi_md                                        └─ m_axi_md
   S_AXI_HP1_FPD  (128 b) ─ smc_md1 ─ nanoxl_0/m_axi_md1      S_AXI_HP1 (64 b) ─ smc_md1 ─ m_axi_md1
   pl_ps_irq0 ─ axi_intc_0 ─ nanoxl_0/irq                     IRQ_F2P ─ axi_intc_0 ─ nanoxl_0/irq
   pl_clk0 250 MHz, proc_sys_reset_0 (pl_resetn0)             FCLK_CLK0 150 MHz, proc_sys_reset_0
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
channel has its own 128-bit HP port (2 × 4 GB/s at 250 MHz, the 8 GB/s ADR-003 assumes) and fetch uses
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

Fallbacks without RTL changes: `-clk 200` (KV260) / `-clk 125` (PYNQ-Z2) and keep `CLOCK_MHZ` in
`fpga/boards.py` in step (the cycle-model numbers scale with the clock); `-strategy
Performance_ExtraTimingOpt`; for resources, `-params "N 16 L 16"` on the KV260 as a first bitstream.

## 6. First-light checklist

`python3 fpga/first_light.py --board <b> --backend pynq --bitstream nanoxl.bit` runs these in order and
stops at the first failure (the same script runs against the emulator in `tests/fpga`):

| # | step | pass criterion | if it fails |
|---|---|---|---|
| 1 | **CSR** | ID = `0x4E584C43` ("NXLC"), VERSION = `0x00010000`, RING_BASE / RING_SIZE / IRQ_EN read back, STATUS ring-cfg-ok, ERR 0 | ID = 0 or `0xFFFFFFFF`: wrong `--ip` / base address (check `Overlay.ip_dict`), PL not loaded, or the core held in reset (`aresetn`) |
| 2 | **END** | one-word END program: END_COUNT +1, IRQ_STATUS bit 0, W1C clears it | RING_HEAD stuck at 0: fetch master cannot reach DDR (HPC0 / HP0 address map, AxCACHE, CMA physical address) |
| 3 | **DMA loopback** | 16 KiB random bytes DMA_LOAD → each of ABUF / WBUF / ACC / PBUF / KVBUF (top of the buffer) → DMA_STORE: unchanged, with loads and stores on both DMA channels | ERR 5: BRESP/RRESP error (HP0 / HP1 map of `md` / `md1`); data mismatch in one buffer: its bank mapping / `RAM_STYLE` / URAM read-collision behaviour |
| 4 | **DDR bandwidth** | 1 MiB loads on channel 0, channel 1 and both at once, one 1 MiB store; GB/s from PERF_CYCLES at the board clock. Ceiling per channel: 4 GB/s on the KV260 (128 b × 250 MHz), so up to 8 GB/s with both; PYNQ-Z2 1.2 GB/s per 64-bit HP port at 150 MHz, both together bounded by its ~1.5 GB/s DDR. Record it: ADR-003's largest unknown | far below: SmartConnect width/clock conversion, HP port width not 128 b; both-channel number ≈ one-channel number: the two masters land on the same port |
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

Mutation check: swapping two bytes of `m_axi_md_rdata` inside the wrapper makes `test_wrap_smoke` fail
(`step 0: logits differ at 130 entries`); tying `md_axi_rlast` to `rvalid` hangs it.
