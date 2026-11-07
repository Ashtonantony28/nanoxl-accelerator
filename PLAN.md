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
| SG7a | M5 | ARRAY engine sequencer: ISA GEMM/GEMM_GS → C3 tiles incl. ACC write-back; `cmd_gs` group-scale MAC in array_ctrl per ADR-004 | `pytest tests/rtl -k array_seq` | done (docs/array.md §7; verify `SIM=verilator pytest tests/rtl -k array`; cycle model 2-4x optimistic for weight-bound tiles, see §7) |
| SG7b | M5 | VEC engine sequencer: ISA VEC/REQUANT/LN/SOFTMAX/GELU/LUT_LOAD → vec_unit+sfu row commands, PBUF slot loader | `pytest tests/rtl -k vec_seq` | done (rtl/vec_seq*.sv, pbuf_loader.sv; docs/vector-unit.md §9, sfu.md §10) |
| SG4c | M1 | Cycle model with measured SG5/SG6 costs, hazard check for semaphore depth, PYNQ host driver (mocked against funcsim) | `pytest tests/py -k "cyclemodel or driver or hazard"` | done (measured-RTL cycle model, ADR-003 update; compiler DMA coalescing; docs/driver.md, board steps untested) |
| SG7c | M5/M6 | `nanoxl_top` (ctrl_top + DMA + scratchpads + KV tier + engines); compiled `tiny` prefill+decode bit-exact in Verilator vs golden; AUDIT-1 | `pytest tests/rtl -k top` | wave 4 |
| SG8 | M6 | Folded into SG7c (C5 chose full-row softmax; KV modes already in compiler). Online softmax would need a new ADR. | — | merged into SG7c |
| SG9 | M7 | Vivado project for KV260, timing, PYNQ driver (sim-checked; board run needs hardware) | | wave 4 |
| SG10 | M8 | INT4 path, batching, perf; AUDIT-2 | | wave 4 |
