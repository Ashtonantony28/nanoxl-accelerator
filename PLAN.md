# NanoXL sub-goal tracker

Design plan: `docs/design-plan.md`. Contracts: `docs/contracts.md`. Each sub-goal has a Verify command; it is done when that command passes on a clean checkout.

| SG | Milestone | Scope | Verify | Status |
|---|---|---|---|---|
| SG0 | — | Scaffold, numeric contracts C1/C2, runner, CI | `make test-py` | done |
| SG1 | M0 | Train `tiny` + `shakespeare`, FP32 PPL; integer golden model of the full model; quantization study; ADR-002 freezing C5 | `pytest tests/py -k golden` | done (ADR-002, C5 frozen) |
| SG2 | M2 | PE, weight-stationary systolic array (INT8 + INT4 mode), scratchpad SRAM; cocotb vs `gemm_i8` | `make test-rtl` (array tests) | done |
| SG3 | M3a | Requant unit (C2), vector ALU (add/mul/max/shift/sat), reduction tree; cocotb vs numerics | `make test-rtl` (vec tests) | done (wave 1; spec docs/vector-unit.md) |
| SG4 | M1 | Command ISA + encoder, compiler skeleton (tiling, buffer alloc), cycle model, roofline report; array/lane sizing ADR-003 | `pytest tests/py -k "isa or compiler or cyclemodel or funcsim"` | done (ADR-003 proposed) |
| SG5 | M3b | SFUs + softmax/LayerNorm/GELU RTL per ADR-002 | | done (spec + timing docs/sfu.md) |
| SG4b | M1 | Compiler/funcsim consume the SG1 bundle and C5 integer ops (no float placeholders), W4A8 group-scale lowering; compiled `tiny` in funcsim bit-identical to golden.py | `pytest tests/py -k "funcsim_golden or compiler"` | done (ADR-004 proposed: GEMM_GS, LUT_LOAD, alignment rule) |
| SG6 | M4 | Command processor, DMA, AXI (cocotbext-axi), formal on FIFOs/arbiters | | done (docs/control.md; verify `SIM=verilator pytest tests/rtl -k "cmd or dma or csr or sem or ctrl"`) |
| SG7a | M5 | ARRAY engine sequencer: ISA GEMM/GEMM_GS → C3 tiles incl. ACC write-back; `cmd_gs` group-scale MAC in array_ctrl per ADR-004 | `pytest tests/rtl -k array_seq` | done (docs/array.md §7; verify `SIM=verilator pytest tests/rtl -k array`; cycle model now matches the measured tile occupancy, see §7 and AUDIT-1 A1-05/A1-06) |
| SG7b | M5 | VEC engine sequencer: ISA VEC/REQUANT/LN/SOFTMAX/GELU/LUT_LOAD → vec_unit+sfu row commands, PBUF slot loader | `pytest tests/rtl -k vec_seq` | done (rtl/vec_seq*.sv, pbuf_loader.sv; docs/vector-unit.md §9, sfu.md §10) |
| SG4c | M1 | Cycle model with measured SG5/SG6 costs, hazard check for semaphore depth, PYNQ host driver (mocked against funcsim) | `pytest tests/py -k "cyclemodel or driver or hazard"` | done (measured-RTL cycle model, ADR-003 update; compiler DMA coalescing; docs/driver.md, board steps untested) |
| SG7c | M5/M6 | `nanoxl_top` (ctrl_top + DMA + scratchpads + KV tier + engines); compiled `tiny` prefill+decode bit-exact in Verilator vs golden; AUDIT-1 | `pytest tests/rtl -k top` | done (docs/top.md; verify `SIM=verilator pytest tests/rtl -k top`; tiny W8A8/W4A8 x on-chip/DDR KV bit-exact, 16-token prefill + 8 decode; A1-02 SAT map fixed; VEC port widening deferred) |
| SG8 | M6 | Folded into SG7c (C5 chose full-row softmax; KV modes already in compiler). Online softmax would need a new ADR. | — | merged into SG7c |
| SG9a | M7 | FPGA flow without hardware: Vivado TCL (KV260 block design, PYNQ-Z2 variant), constraints, overlay packaging; Yosys synth_xilinx resource estimates per board config | `pytest tests/fpga` | done, untested on Vivado/hardware (docs/fpga.md; Yosys estimates bench/results/resources_*.json: neither board fits, LUT 15x/10x over; RTL change list C1-C9 for the owners) |
| SG10a | M8 | Second DMA master on a second HP port (`chan`), DMA command pipelining; measured decode gain | `pytest tests/rtl -k "dma or ctrl or top"` | done (ADR-005: one DMA queue → two engines by `chan`, md1_axi on HP1; 4 commands in flight per engine; SIGNALs carried on done; cross-channel WAW rejected by check_hazards; formal `queue`; modelled KV260 shakespeare W8 decode 332 → 619 tok/s; verify `SIM=verilator pytest tests/rtl -k "dma or ctrl or top or cmd or sem"`) |
| SG10b | M8 | 64-byte (4-line) VEC engine ports + banked buffer port widening; measured prefill gain | `pytest tests/rtl -k "vec_seq or top"` | done (vec_seq PL / nanoxl_top VP = 4; PL=1 cycle-exact; tiny prefill −10 % W8A8 / −12 % W4A8 measured; KV260 shakespeare prefill 17.6 → 14.7 ms modelled; docs/vector-unit.md §9.3, docs/top.md §6) |
| SG9 | M7 | Vivado project for KV260, timing, PYNQ driver (sim-checked; board run needs hardware) | | wave 4 |
| SG10 | M8 | INT4 path, batching, perf; AUDIT-2 | | wave 4 |
