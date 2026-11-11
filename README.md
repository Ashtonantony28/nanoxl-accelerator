# NanoXL

A systolic-array INT8/INT4 transformer inference accelerator in SystemVerilog. It runs a trained nanoGPT end to end, and its output matches a numpy integer golden model bit for bit.

## Status (2026-11-11)

| Area | State |
|---|---|
| Models | `tiny` (4L, d=128) and `shakespeare` (6L, d=384, time-boxed training, val loss 1.61) trained from nanoGPT-extended. W8A8 PPL +0.8 %, W4A8 +0.7 % on shakespeare (`docs/quant-study.md`). |
| Golden model | `nanoxl/golden.py`: an integer-only model of the whole network (C1–C5 in `docs/contracts.md`, ADR-002). |
| Compiler / funcsim | `nanoxl/compiler.py` lowers the model to the 128-bit command ISA (`docs/isa.md`). The compiled tiny model runs bit-identical to golden. |
| RTL | `rtl/nanoxl_top.sv` has the command processor, dual DMA, banked scratchpads, a weight-stationary systolic array with WP486 DSP packing and W4A8 group scales, a vector unit, and SFUs (softmax, LayerNorm, GELU). The tiny model prefills and greedily decodes **bit-exact vs golden in Verilator** (W8A8 and W4A8, KV cache on-chip or in DDR). |
| Host driver | `nanoxl/driver`: `SimBackend` (funcsim plus CSR emulation) and `PynqBackend` (untested until hardware exists). |
| Performance (modelled, measured-cost cycle model) | KV260 first bitstream (N=8, L=4, 200 MHz): shakespeare 139 tok/s W8, tiny 1,650 tok/s. The planned N=32 design is modelled at 607 tok/s but does not fit (ADR-006). |
| FPGA | Vivado 2024.2 build scripts for KV260 (primary) and PYNQ-Z2 in `fpga/`; Yosys resource estimates in `bench/results/`. **Nothing has been through Vivado or on a board yet** (`docs/fpga.md` has the bring-up guide). |
| Verification | About 250 Python tests and about 140 cocotb tests (Icarus and Verilator), formal proofs (yosys-smtbmc), and two independent audits (`docs/audits/`) with every finding closed. CI runs Icarus and Verilator on each push, plus a nightly slow tier. |

## Quick start

```bash
pip install numpy torch pytest cocotb cocotbext-axi z3-solver
make test-fast                     # Icarus fast tier
make test-verilator-fast           # needs Verilator >= 5.036
python3 -m nanoxl.driver --bundle artifacts/models/tiny/w8a8.npz --prompt "ROMEO:" --n 50
```

## Where to read

`PLAN.md` (sub-goal tracker), `docs/design-plan.md` (original design), `docs/contracts.md`, `docs/adr/`, `docs/top.md` (integration and measured cycles), `docs/fpga.md` (board bring-up).

## Next steps

1. Run Vivado synthesis and implementation for the KV260 N=8 build. Then try the `N 16 NB 4` step-up (about 340 tok/s), which ADR-006 lists.
2. Board first light (`fpga/first_light.py`), then tiny generate on hardware against golden.
3. The area work in ADR-006 (pruning router ports per buffer, called C10) to get N=16/32 onto the KV260.
