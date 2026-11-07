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
| `rtl/nanoxl_axi_wrap.v` | Verilog-2001 shell around `nanoxl_top` (IP integrator module references need a Verilog/VHDL top). Pure rename: `aclk`, `aresetn`, `irq`, `s_axi_ctrl_*`, `m_axi_mf_*`, `m_axi_md_*`, with `X_INTERFACE_INFO` attributes |
| `fpga/boards.py` | **single source of truth**: `nanoxl_top` parameters per board, clock, part, CSR base, device capacities, and `hwconfig(board)`, the `HwConfig` the compiler must use with that bitstream |
| `fpga/kv260/build.tcl`, `fpga/pynqz2/build.tcl` | Vivado project + block design + synth/impl/bitstream + reports + overlay export |
| `fpga/common/nanoxl_build.tcl`, `fpga/common/rtl_files.tcl` | shared procs (argument parsing, project, core cell, address mapping, run/report/export) and the RTL list |
| `fpga/kv260/nanoxl_kv260.xdc`, `fpga/pynqz2/nanoxl_pynqz2.xdc` | `RAM_STYLE` for the scratchpad banks (URAM on the KV260); clocks come from the PS IP |
| `fpga/yosys/estimate.py`, `fpga/yosys/prep.py` | Yosys resource estimates → `bench/results/resources_<board>.json` |
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
for ACC only after an RTL change (per-buffer `NB`, see §2.4).

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

RESULTS_TABLE

### 2.1 What the numbers say

FINDINGS

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

CHANGELIST

## 3. Block design and address map

```
 KV260                                                   PYNQ-Z2
 zynq_ultra_ps_e_0                                       processing_system7_0
   M_AXI_HPM0_FPD ─ smc_ctrl ─┬─ nanoxl_0/s_axi_ctrl       M_AXI_GP0 ─ smc_ctrl ─┬─ nanoxl_0/s_axi_ctrl
                              └─ axi_intc_0/s_axi                               └─ axi_intc_0/s_axi
   S_AXI_HP0_FPD (128 b) ─ smc_mf ─ nanoxl_0/m_axi_mf      S_AXI_HP0 (64 b) ─ smc_mf ─ nanoxl_0/m_axi_mf
   S_AXI_HP1_FPD (128 b) ─ smc_md ─ nanoxl_0/m_axi_md      S_AXI_HP1 (64 b) ─ smc_md ─ nanoxl_0/m_axi_md
   pl_ps_irq0 ─ axi_intc_0 ─ nanoxl_0/irq                  IRQ_F2P ─ axi_intc_0 ─ nanoxl_0/irq
   pl_clk0 250 MHz, proc_sys_reset_0 (pl_resetn0)          FCLK_CLK0 150 MHz, proc_sys_reset_0
```

| segment | KV260 | PYNQ-Z2 | size |
|---|---|---|---|
| `nanoxl_0/s_axi_ctrl` (CSRs, docs/control.md §5) | `0xA000_0000` | `0x4000_0000` | 4 KiB |
| `axi_intc_0/s_axi` | `0xA001_0000` | `0x4180_0000` | 4 KiB |
| `m_axi_mf`, `m_axi_md` → DDR | HPx `DDR_LOW` (0 – 2 GiB) | HPx `DDR_LOWOCM` (0 – 512 MiB) | 32-bit masters |

Everything the driver needs is in the `.hwh`: `PynqBackend` takes `ip_dict["nanoxl_0"]["phys_addr"]`
(any base works), allocates the image with `pynq.allocate(..., cacheable=False)` and relocates every
program to its physical address (docs/driver.md). The masters use AxCACHE = 0011 (non-coherent HP
ports, uncached CMA buffer). The two masters go to two HP ports: `mf` (command fetch, small) on HP0 and
`md` (DMA, all the traffic) on HP1, so the 4 GB/s single-master DMA ceiling (ADR-003) is not shared with
fetch. A second DMA master (SG10a) would take HP2 the same way.

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

TIMING

Fallbacks without RTL changes: `-clk 200` (KV260) / `-clk 125` (PYNQ-Z2) and keep `CLOCK_MHZ` in
`fpga/boards.py` in step (the cycle-model numbers scale with the clock); `-strategy
Performance_ExtraTimingOpt`; for resources, `-params "N 16 L 16"` on the KV260 as a first bitstream.

## 6. First-light checklist

`python3 fpga/first_light.py --board <b> --backend pynq --bitstream nanoxl.bit` runs these in order and
stops at the first failure (the same script runs against the emulator in `tests/fpga`):

| # | step | pass criterion | if it fails |
|---|---|---|---|
| 1 | **CSR** | ID = `0x4E584C43` ("NXLC"), VERSION = `0x00010000`, RING_BASE / RING_SIZE / IRQ_EN read back, STATUS ring-cfg-ok, ERR 0 | ID = 0 or `0xFFFFFFFF`: wrong `--ip` / base address (check `Overlay.ip_dict`), PL not loaded, or the core held in reset (`aresetn`) |
| 2 | **END** | one-word END program: END_COUNT +1, IRQ_STATUS bit 0, W1C clears it | RING_HEAD stuck at 0: fetch master cannot reach DDR (HP0 address map, AxCACHE, CMA physical address) |
| 3 | **DMA loopback** | 16 KiB random bytes DMA_LOAD → each of ABUF / WBUF / ACC / PBUF / KVBUF (top of the buffer) → DMA_STORE: unchanged | ERR 5: BRESP/RRESP error (HP1 map); data mismatch in one buffer: its bank mapping / `RAM_STYLE` / URAM read-collision behaviour |
| 4 | **DDR bandwidth** | 1 MiB load and store, GB/s from PERF_CYCLES at the board clock. Expect ≤ 4 GB/s on the KV260 (one 128-bit master at 250 MHz), ≤ 1.2 GB/s on the PYNQ-Z2 (64-bit HP at 150 MHz). Record it: ADR-003's largest unknown | far below: SmartConnect width/clock conversion, HP port width not 128 b |
| 5 | **single layer** | 4-token prefill of a 1-layer cut of tiny W8A8: every logit == `nanoxl.golden` (exercises GEMM, REQUANT, LN, softmax, GELU, KV_APPEND) | reproduce in the SimBackend and Verilator (docs/driver.md §3.6), single-step with CTRL bit 0 |
| 6 | **full tiny generate** | reference prompt + 32 greedy tokens == `artifacts/models/tiny/golden_ref.json` | compare per-step logits with `Golden.forward`; SAT_* with golden saturation counts |

Afterwards: `PERF_*` per step vs `nanoxl.cyclemodel` at the board's DDR latency (M7), tokens/s
(`--json` output, `host_seconds`), then W4A8.

## 7. Verification here (`python3 -m pytest -q tests/fpga`)

| test | what |
|---|---|
| `test_build_tcl_dry_run[kv260, pynq-z2]` | each `build.tcl` runs to the end in tclsh with every Vivado command stubbed (unknown commands are errors); checks the `nanoxl_0` CONFIG == `fpga/boards.py`, part, clock, CSR base 4 KiB, both masters → SmartConnect → the right HP port → DDR segment, irq/clock/reset pins, bitstream + `.bit`/`.hwh` export, RTL list |
| `test_build_tcl_options` | `-clk`, `-params`, `-proj_only`; a bad option fails |
| `test_rtl_fileset_matches_hierarchy[...]` | `fpga/common/rtl_files.tcl` exists file by file and equals the files of every module in Yosys's elaborated hierarchy of `nanoxl_axi_wrap` (with each board's parameters), and equals the simulations' `TOP_SRC` |
| `test_wrapper_ports_follow_vivado_axi_naming` | every `nanoxl_top` port is in the wrapper, renamed, same direction and width; `<intf>_<signal>` names with the full AXI4-Lite / AXI4 signal sets, master/slave directions, `X_INTERFACE_INFO` / `X_INTERFACE_PARAMETER` (clock ASSOCIATED_BUSIF, active-low reset, AXI4LITE / READ_ONLY); no SystemVerilog in the shell; same parameters |
| `test_wrapper_lint` | Verilator `-Wall` lint of the wrapper top: no message in the wrapper |
| `test_wrap_smoke` | **behaviour**: the unmodified `nanoxl_top` smoke bench (`top_cocotb.top_tiny`, real driver over AXI, 1-layer tiny W8A8, logits / KV cache / per-op saturation == golden) through the wrapper's port names (Icarus ~90 s) |
| `test_hwconfig_fits_rtl_and_runs_tiny[...]` | as-built `HwConfig` within the RTL capacities (powers of two, 17-bit line addresses), tiny generates == golden on it (emulator) |
| `test_first_light_sim[...]`, `test_first_light_reports_failure`, `test_run_board_cli[...]` | the board tools against the emulator, and a corrupted loopback is reported as FAIL |
| `test_resource_estimates_current[...]` | `bench/results/resources_*.json` exist for today's `RTL_PARAMS` |

Mutation check: swapping two bytes of `m_axi_md_rdata` inside the wrapper makes `test_wrap_smoke` fail
(`step 0: logits differ at 130 entries`); tying `md_axi_rlast` to `rvalid` hangs it.
