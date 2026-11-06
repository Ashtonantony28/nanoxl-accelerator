# NanoXL sub-goal tracker

Design plan: `docs/design-plan.md`. Contracts: `docs/contracts.md`. Each sub-goal has a Verify command; it is done when that command passes on a clean checkout.

| SG | Milestone | Scope | Verify | Status |
|---|---|---|---|---|
| SG0 | — | Scaffold, numeric contracts C1/C2, runner, CI | `make test-py` | done |
| SG1 | M0 | Train `tiny` + `shakespeare`, FP32 PPL; integer golden model of the full model; quantization study; ADR-002 freezing C5 | `pytest tests/py -k golden` | wave 1 |
| SG2 | M2 | PE, weight-stationary systolic array (INT8 + INT4 mode), scratchpad SRAM; cocotb vs `gemm_i8` | `make test-rtl` (array tests) | done |
| SG3 | M3a | Requant unit (C2), vector ALU (add/mul/max/shift/sat), reduction tree; cocotb vs numerics | `make test-rtl` (vec tests) | done (wave 1; spec docs/vector-unit.md) |
| SG4 | M1 | Command ISA + encoder, compiler skeleton (tiling, buffer alloc), cycle model, roofline report; array/lane sizing ADR-003 | `pytest tests/py -k "isa or compiler or cyclemodel or funcsim"` | done (ADR-003 proposed) |
| SG5 | M3b | SFUs + softmax/LayerNorm/GELU RTL per ADR-002 | | wave 2 |
| SG6 | M4 | Command processor, DMA, AXI (cocotbext-axi), formal on FIFOs/arbiters | | wave 2 |
| SG7 | M5 | Top-level integration; `tiny` decode bit-exact in sim; AUDIT-1 | | wave 3 |
| SG8 | M6 | KV cache tier, online-softmax attention, prefill/decode | | wave 3 |
| SG9 | M7 | Vivado project for KV260, timing, PYNQ driver (sim-checked; board run needs hardware) | | wave 4 |
| SG10 | M8 | INT4 path, batching, perf; AUDIT-2 | | wave 4 |
