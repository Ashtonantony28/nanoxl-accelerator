# Plan: "NanoXL", a systolic INT8/INT4 transformer inference accelerator on FPGA

Status: **PLANNING, awaiting review.** No implementation code until Ashton says proceed.
Working name: `nanoxl` (placeholder; pick any repo name). Drafted 2026-11-06.

---

## 0. One-paragraph pitch

Design, verify and run on a real FPGA a small inference accelerator that executes a trained nanoGPT model end to end: tokens in over a Python driver on the Zynq's ARM cores, tokens out, **bit-identical** to a numpy integer golden model. The datapath is a weight-stationary INT8 systolic array (with a W4A8 INT4-weight mode), a SIMD vector unit with integer softmax / LayerNorm / GELU special-function units, a scratchpad + DDR memory hierarchy with an on-chip KV-cache tier, and a small command processor fed by a compiler that lowers a nanoGPT checkpoint to an accelerator instruction stream. It is the natural successor to two things you already own: **tiny-gpu-extended** (cocotb, SystemVerilog, the xf int8 transformer track) and **nanoGPT-extended** (the model and its variants).

## 1. Why this project, and what it carries forward from your earlier work

The tiny-gpu xf track proved a toy SIMT GPU can run one int8 MLP block and one attention head. Its audits also left a clear list of things that a "real" design should do differently. This plan treats those as requirements:

| Lesson from earlier work (source) | What this project does instead |
|---|---|
| MLP half and attention half were separate images; no full block, no residual chaining (tiny-gpu analysis §7 #18, AUDIT-1 M1) | The unit of correctness is the **whole model**: N blocks, residuals, final LN, LM head, sampling loop. |
| Output-only tolerance hid gain bugs (a LUT off by one index passed; AUDIT-7 H2) | **Bit-exact** RTL vs integer golden model at every layer boundary, plus a separate float-vs-int accuracy budget (perplexity), plus per-op gain/mean checks. |
| Attention TOL was not a real bound (AUDIT-6 H1) | Error budget derived analytically per op and checked statistically on a truly held-out set with a frozen seed that is never used for tuning. |
| Weights used PyTorch default init, not a trained model (AUDIT-1 L2) | Use a **trained** nanoGPT checkpoint (Shakespeare-char), so a deleted branch visibly hurts perplexity. |
| Overflow guards untested; mutants survived (AUDIT-3 M1, AUDIT-7 M1) | Saturation counters in RTL, tests that force saturation, and a mutation-testing pass on core modules. |
| Full matrix took ~53 min on Icarus (tiny-gpu SG12) | **Verilator** for cocotb (Icarus kept only as a second simulator for unit tests). Tiered fast/slow test gates. |
| Telemetry counters under-counted (tiny-gpu §7 #8) | Perf counters are themselves verified against the cycle model. |
| nanoGPT-extended: GQA has no KV cache, so no inference benefit (nanoGPT analysis §1) | This project builds the KV cache in hardware, which is exactly where GQA pays off (stretch milestone M9). |
| nanoGPT-extended: `train.py` crashes with non-default LR schedules (nanoGPT §7) | Train with the default schedule, or fix that bug first in nanoGPT-extended as a tiny side PR. |

## 2. Goals and non-goals

### Goals (measurable)
1. **G1 Correctness:** greedy decoding of 1,000 tokens on the FPGA produces exactly the same token sequence as the integer golden model, for 3 prompts. Every layer's output tensor matches bit for bit in simulation.
2. **G2 Accuracy:** INT8 (W8A8) validation perplexity within **+3 %** of FP32; W4A8 within **+8 %** (to be confirmed in M0; these are starting targets, not promises).
3. **G3 Performance:** decode is memory-bound, so the target is a fraction of the DDR roofline: **≥ 60 % of achievable DDR bandwidth** during decode, and **≥ 50 % array utilization** during prefill. Concrete numbers in §5.
4. **G4 Speed-up:** beat a well-optimized INT8 CPU baseline on the same board's ARM cores (numpy / ONNX Runtime on the Cortex-A53 or A9) by **≥ 10×** tokens/s.
5. **G5 Engineering:** parameterized RTL (array size, lanes, buffer sizes), full cocotb suite in CI on Verilator under 10 minutes for the fast tier, timing closure at the target clock, a written architecture doc and ADRs (same style as tiny-gpu).

### Non-goals (for v1)
- Training on the accelerator, backprop, FP16/BF16 datapaths.
- Models larger than ~25 M parameters, or HuggingFace GPT-2 124M (possible later from DDR, but not a v1 target).
- An ASIC flow (an OpenLane run of the array alone is a fun stretch, see M9).
- A general compiler (TVM/MLIR). The compiler is purpose-built for nanoGPT graphs.

## 3. Models

| Name | Config | Params | INT8 size | INT4 size | KV cache (INT8) | Role |
|---|---|---|---|---|---|---|
| `tiny` | 4 layers, 4 heads, d=128, ctx=128, vocab 65 | ≈0.81 M | ≈0.8 MB | ≈0.4 MB | 2·4·128·128 = 128 KB | Bring-up. Fits fully on-chip on KV260 (URAM). INT4 fits on-chip even on PYNQ-Z2. |
| `shakespeare` | 6 layers, 6 heads, d=384, ctx=256 (nanoGPT's `train_shakespeare_char`) | ≈10.7 M | ≈10.7 MB | ≈5.4 MB | 2·6·256·384 ≈ 1.18 MB | Headline demo. Weights stream from DDR; KV cache on-chip on KV260, in DDR on PYNQ-Z2. |
| `llama-ish` (stretch) | same sizes with RMSNorm, SwiGLU, RoPE, GQA (n_kv_heads=2) from nanoGPT-extended | ≈10–12 M | | | GQA cuts KV by 3× | Shows the hardware cost/benefit of modern variants. |

Both models train in minutes on a laptop GPU or Colab, and in well under an hour on CPU for `tiny`.

Architecture for v1 is classic GPT-2 (LayerNorm, GELU, learned positional embedding, MHA), because that is what nanoGPT trains by default and it exercises the hardest integer ops (LayerNorm variance, GELU, softmax).

## 4. Quantization scheme

Decided in M0 using the golden model; the starting design:

- **Weights:** symmetric, per-output-channel INT8. INT4 mode: symmetric, group-wise (group size 64 along K) with an INT8/FP16-free fixed-point scale per group, packed two per byte. Optional GPTQ/AWQ-style rounding offline (pure software, cheap win).
- **Activations:** INT8, symmetric, **static per-tensor scales** from calibration, with SmoothQuant-style scale migration into weights for the LN→QKV and LN→FC inputs (GPT activations have outlier channels). Dynamic per-token quant is an M8 option because it needs a max-reduction before every GEMM.
- **Accumulation:** INT32 in the array. **Requantization** INT32→INT8 with a fixed-point multiplier and right shift with round-half-to-even and saturation (the gemmlowp / TFLite scheme), done in the vector unit.
- **Residual stream:** kept at **INT16** (or INT8 with a wider scale) to avoid the classic residual-saturation failure. Decided by M0 measurements.
- **Softmax:** integer-only. Subtract running max, `exp` via a shift-and-polynomial (I-BERT style) or a 256-entry LUT on a fixed-point exponent, INT32 sum, reciprocal via LUT + one Newton step, output UINT8 probabilities. Implemented as **online softmax** (running max and rescaled sum), so attention can stream K/V tiles.
- **LayerNorm:** INT32 mean and variance accumulation over d (two passes or Welford), `rsqrt` via LUT + Newton, per-channel γ/β folded into fixed-point.
- **GELU:** I-BERT second-order polynomial on the erf, or a 256-entry LUT indexed by the INT8 input (the LUT is exact for INT8 inputs, which makes it the likely winner).
- **Embedding + LM head:** INT8 table lookup; LM head is a GEMM against the (tied) embedding; argmax / top-k on the host for v1, in hardware as a stretch.

Every one of these has a float reference, an integer golden implementation, and an error measurement (max LSB error, mean error, gain) in M0.

## 5. Hardware target, roofline, and sizing

### Board options

| | **AMD Kria KV260 (recommended)** | PYNQ-Z2 (budget) |
|---|---|---|
| Device | Zynq UltraScale+ K26 SOM (XCK26) | Zynq-7020 |
| CPU | 4× Cortex-A53 @ 1.3 GHz | 2× Cortex-A9 @ 650 MHz |
| LUTs / DSPs | ≈117 K / 1,248 DSP48E2 | 53.2 K / 220 DSP48E1 |
| On-chip RAM | 144 BRAM36 (≈5.1 Mb) **+ 64 URAM (18 Mb ≈ 2.25 MB)** | 140 BRAM36 (≈4.9 Mb ≈ 630 KB) |
| DDR | 4 GB DDR4, 64-bit (≈19 GB/s theoretical, est. 8–12 GB/s from PL) | 512 MB DDR3, 16-bit (≈2.1 GB/s theoretical, est. ~1.5 GB/s) |
| Python on board | PYNQ for Kria (Ubuntu) | PYNQ image |
| Proposed config | 32×32 array (1,024 MACs, ≈512 DSPs with 2-MAC packing), 64-lane vector unit, 250 MHz target | 16×16 array (256 MACs, ≈128 DSPs), 16-lane vector unit, 150 MHz target |
| Vivado license | Free (ML Standard) | Free (ML Standard) |

All board numbers are from memory of the datasheets and should be confirmed when ordering.

### Back-of-envelope performance (KV260, `shakespeare` model)
- **Peak compute:** 1,024 MAC × 2 ops × 250 MHz ≈ **512 GOPS** INT8.
- **Decode (batch 1):** each token reads all weights once: 10.7 MB INT8. At 8 GB/s that is ≈1.3 ms → **≈750 tokens/s** (INT8), **≈1,400 tokens/s** (INT4). Compute per token is ~21 MOP, i.e. ~40 µs at peak, so decode is **~30× memory-bound**. This is the central architectural fact: batch-1 decode is GEMV and leaves a 32×32 array mostly idle.
- **Prefill (256-token prompt):** ≈5.5 GOP of GEMM. At 50 % utilization ≈ **21 ms**. Compute-bound, this is where the array earns its area.
- **Consequences for the design:** (a) the DMA/weight-streaming path matters more than the array for decode; (b) INT4 weights are the biggest decode lever; (c) **batched decode** (B sequences sharing one weight read) turns GEMV back into GEMM and is a cheap, impressive M8 result; (d) `tiny` fits on-chip, so it shows the compute-bound ceiling.
- **PYNQ-Z2:** ≈1.5 GB/s → **≈140 tokens/s** INT8 decode for `shakespeare`; `tiny` INT4 fully on-chip runs much faster.

The M1 cycle model will replace these estimates with modelled numbers before any RTL is written, and the array size is chosen there.

## 6. Architecture

```
            ┌──────────────────────── Zynq PS (ARM, Linux, PYNQ) ────────────────────────┐
            │  Python driver: load weights → DDR, write command buffer, ring doorbell,     │
            │  poll/IRQ, read logits/tokens, sampling (v1)                                 │
            └──────┬──────────────── AXI-Lite (CSRs) ───────────── AXI HP/HPC (DDR) ──────┘
                   │                                                │
   ┌───────────────▼─────────────── PL: nanoxl core ────────────────▼────────────────────┐
   │  Command processor ──► dispatch queues ──► [DMA-W] [DMA-A] [DMA-KV] (AXI4 masters)   │
   │        │ scoreboard / semaphores                                                     │
   │        ▼                                                                             │
   │  ┌──────────────┐   ┌───────────────────────────┐   ┌──────────────────────────┐     │
   │  │ Weight buf   │──►│ Systolic array N×N        │──►│ Accumulator buf (INT32)   │    │
   │  │ (ping-pong,  │   │ weight-stationary, INT8   │   │ double-buffered           │    │
   │  │  INT4 unpack)│   │ + W4A8 mode, 2-MAC/DSP    │   └────────────┬─────────────┘     │
   │  └──────────────┘   └─────────▲─────────────────┘                │                   │
   │                     ┌─────────┴───────┐              ┌───────────▼───────────────┐   │
   │                     │ Activation buf  │◄─────────────│ Vector unit (L lanes)      │   │
   │                     │ (INT8/INT16)    │              │ requant, add, mul, max,    │   │
   │                     └─────────────────┘              │ reduce; SFUs: exp, recip,  │   │
   │  ┌───────────────────────────────┐                   │ rsqrt, GELU LUT            │   │
   │  │ KV-cache SRAM tier (URAM)     │◄─────────────────►│ online-softmax datapath    │   │
   │  │ [layer][head][pos][d_head]    │                   └───────────────────────────┘   │
   │  └───────────────────────────────┘          Perf counters, trace FIFO, ILA hooks     │
   └──────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.1 Systolic array
- Weight-stationary N×N PEs, INT8×INT8→INT32, activations flow left to right, partial sums top to bottom. Weight-stationary suits transformers because a weight tile is reused across all tokens of a prefill, and the same activation is shared along a row, which is exactly the shared-operand pattern that lets one DSP48E2 compute two INT8 products (the AMD/Xilinx INT8 packing technique, WP486).
- W4A8 mode: INT4 weights unpacked and dequant-free (scales applied at requant time, per group, which forces group size to be a multiple of N or a per-group partial-sum flush; decided in M1).
- Preload of the next weight tile overlaps compute (double-buffered weight registers in each PE).
- Decode path: a **GEMV mode** that bypasses the skew/deskew latency and processes one activation vector per weight tile, so latency is not dominated by array fill/drain.

### 6.2 Vector unit and special-function units
- L-lane SIMD (64 on KV260) over INT8/INT16/INT32 with a small micro-op set: requant, add, mul, max, shift, saturate, horizontal reduce (sum, max).
- SFUs shared across lanes or replicated (area vs throughput decided in M1): `exp` (fixed-point), `recip`, `rsqrt` (LUT + Newton), `gelu` (LUT).
- **Softmax** = max-reduce → exp → sum-reduce → recip → scale, implemented online so it can consume score tiles as they come out of the array.
- **LayerNorm** = sum and sum-of-squares reduce → mean/var → rsqrt → scale/shift.

### 6.3 Memory hierarchy and KV cache
| Level | Contents | KV260 | PYNQ-Z2 |
|---|---|---|---|
| L0 | PE weight registers, accumulators | in fabric | in fabric |
| L1 scratchpads | weight ping-pong, activation buffer, INT32 accumulator buffer | BRAM, ~1–1.5 MB total | BRAM, ~300–400 KB |
| KV tier | K and V for all layers/heads/positions, INT8 with per-head scales | URAM, up to ~1.5 MB (`shakespeare` ctx 256 fits) | DDR with an on-chip tile buffer |
| DDR | weights (streamed), embeddings, overflow KV, command buffers, logits | 4 GB | 512 MB |

- KV layout `[layer][head][pos][d_head]` so one head's K for all positions is a contiguous burst; V the same.
- Decode step appends one K/V row per head per layer; attention reads K/V rows `0..t` as tiles and feeds online softmax. This is flash-decoding style: the score row is never materialized beyond a tile.
- Context overflow policy for v1: stop at ctx (nanoGPT crops the window). Stretch: ring buffer + sliding window.
- A **DDR-only KV mode** exists on both boards so the URAM tier is an optimization you can measure, not a dependency.

### 6.4 Control: command ISA and command processor
Coarse-grained, VLIW-free "macro" instructions written by the compiler into a DDR ring buffer:

`DMA_LOAD(dst_buf, ddr_addr, shape, stride)`, `DMA_STORE`, `GEMM(a_buf, w_buf, acc_buf, M,N,K, mode=int8|int4|gemv)`, `VEC(op, src, dst, len, imm)`, `REQUANT(acc, dst, mult, shift)`, `LN(src, dst, gamma, beta)`, `SOFTMAX_ONLINE(...)`, `KV_APPEND(layer, head, pos)`, `ATTN_TILE(...)`, `WAIT(sem)` / `SIGNAL(sem)`, `IRQ`, `END`.

- Separate in-order queues for DMA, array and vector unit, synchronized with semaphores (the classic decoupled access/execute pattern used by VTA and Gemmini), so weight streaming overlaps compute.
- CSRs over AXI-Lite: doorbell, ring pointers, status, perf counters, saturation counters, a debug single-step mode.

### 6.5 Software stack
- `nanoxl/export.py`: load nanoGPT checkpoint (pin a nanoGPT-extended commit, as the tiny-gpu xf track did), calibrate, quantize, pack weights.
- `nanoxl/golden.py`: numpy **integer** reference of every op and the full model. This is the spec.
- `nanoxl/compiler.py`: lowers the model to the command ISA with tiling, buffer allocation and double-buffering; emits a DDR image.
- `nanoxl/cyclemodel.py`: cycle-approximate simulator of the same ISA (used for sizing and to cross-check perf counters).
- `nanoxl/driver/`: PYNQ overlay class (`Overlay`, `allocate` for CMA buffers), command submission, IRQ or polling, a `generate(prompt, n)` API and a tiny CLI / Jupyter demo.
- Baseline: the same integer model run with numpy and ONNX Runtime on the ARM cores for G4.

## 7. Verification plan (cocotb-first)

Tiers, mirroring the tiny-gpu habit of a Verify command per sub-goal:

1. **Golden model tests (pytest):** each integer op vs float within an analytic bound; gain and mean-error checks (not just max error); held-out seed set frozen in M0 and never used for tuning.
2. **Unit RTL tests (cocotb + Verilator):** PE, array (random M/N/K including ragged edges, INT4 mode, saturation), each SFU exhaustively over its INT8 input domain (cheap: 256 or 65,536 points), vector unit micro-ops, scratchpad arbitration, command decoder.
3. **Protocol tests:** AXI4/AXI-Lite with `cocotbext-axi` (random backpressure, out-of-order-free but with randomized latency), DMA descriptor edge cases (4 KB boundary crossings, unaligned lengths).
4. **Formal (SymbiYosys, small blocks):** FIFOs, arbiters, semaphore unit, ring-buffer pointer logic. Proves no deadlock / no overflow for the control plane.
5. **SVA assertions** compiled into Verilator runs: no read of an un-filled buffer, no accumulator overflow without the saturation flag, queue invariants.
6. **Integration:** one transformer block bit-exact vs golden; then the full `tiny` model; then `shakespeare` (slow tier, nightly).
7. **Coverage and mutation:** functional coverage bins (tile shapes, modes, backpressure), and a mutation pass on PE / requant / softmax to make sure the suite kills obvious mutants (directly answers the surviving-mutant findings from tiny-gpu AUDIT-7).
8. **Hardware-in-the-loop:** the same Python test vectors run through the PYNQ driver on the board; layer-by-layer dump mode for bisecting mismatches; ILA probes on the command processor.
9. **Independent audits:** at the end of M5 and M8, an audit pass in the same AUDIT-N style you used before.

CI: GitHub Actions with Verilator for tiers 1–5 on every PR (target < 10 min), integration tiers nightly. Vivado synthesis is not in CI (licence/size); a local `make synth` reports utilization and timing.

## 8. Milestones

Each milestone ends with a concrete exit check. Durations assume part-time work (~10–15 h/week); with heavy Claude assistance the software milestones go faster, the FPGA bring-up does not.

| # | Milestone | Exit criterion | Est. |
|---|---|---|---|
| **M0** | Model + quantization study: train `tiny` and `shakespeare`, write `golden.py`, pick scheme | INT8 PPL ≤ +3 % (or a documented revised target); every op has a measured error table; scheme frozen in an ADR | 1.5–2 wk |
| **M1** | ISA + cycle model + sizing | Compiler emits command streams for both models; cycle model reports decode tok/s and prefill latency; array size, lanes, buffer sizes chosen per board in an ADR | 2 wk |
| **M2** | Systolic array + scratchpads RTL | Random GEMMs bit-exact in cocotb/Verilator, INT8 and INT4; synth of array alone meets timing at target clock on the chosen part | 2–3 wk |
| **M3** | Vector unit + SFUs (requant, softmax, LN, GELU) | Exhaustive SFU tests pass; softmax/LN rows bit-exact vs golden | 2–3 wk |
| **M4** | Command processor + DMA + AXI | Command streams from M1 execute in sim against an AXI memory model with random latency; formal proofs on control blocks | 2–3 wk |
| **M5** | Full model in simulation | `tiny` greedy decode of 64 tokens bit-exact in Verilator; one `shakespeare` block bit-exact; AUDIT-1 | 2 wk |
| **M6** | KV cache tier + decode loop + online-softmax attention | Prefill + decode with KV cache bit-exact; URAM and DDR KV modes both pass; perf counters within 10 % of cycle model | 2 wk |
| **M7** | FPGA bring-up | Bitstream closes timing; PYNQ driver runs `tiny` then `shakespeare`; **G1 met on hardware**; first real tok/s numbers | 3–4 wk |
| **M8** | Performance: INT4 path, overlap tuning, batched decode, CPU baseline | G3 and G4 met (or the gap explained); benchmark report with roofline plot; AUDIT-2 | 2–3 wk |
| **M9** | Stretch menu (pick any) | LLaMA-ish variants (RMSNorm, SwiGLU, RoPE unit, **GQA KV savings measured**); hardware top-k sampling; speculative decoding with `tiny` drafting for `shakespeare`; OpenLane/ASIC run of the array; power measurement (tokens per joule) | open |

Total for M0–M8: roughly **5–6 months part-time**. M0–M6 need no hardware at all, so the board purchase can wait until about week 12.

## 9. Repository layout (proposed)

```
rtl/        pe.sv, systolic_array.sv, vec_unit.sv, sfu_{exp,recip,rsqrt,gelu}.sv, softmax.sv,
            layernorm.sv, scratchpad.sv, kv_cache.sv, dma.sv, cmd_proc.sv, csr.sv, nanoxl_top.sv
fpga/       kv260/ and pynqz2/ Vivado TCL (block design, constraints), make targets
nanoxl/       export, golden, compiler, cyclemodel, isa, driver/
tests/      cocotb (unit, protocol, integration), pytest (golden, compiler), formal/ (sby)
bench/      CPU baseline, board benchmarks, frozen JSON results
docs/       architecture.md, isa.md, ADRs, milestone reports, audits
PLAN.md     sub-goal tracker with Verify commands (same convention as tiny-gpu-extended)
```

## 10. Hardware, software and budget

| Item | Needed when | Approx. cost (USD, verify at purchase) |
|---|---|---|
| Kria KV260 Vision AI Starter Kit (includes power supply) | M7 (≈week 12) | ≈$250–$350 |
| *or* PYNQ-Z2 | M7 | ≈$150–$230 |
| microSD 32 GB (A1/A2), USB-UART cable if not included, Ethernet cable | M7 | ≈$15–$25 |
| Small fan / heatsink for KV260 under sustained load | M8 | ≈$10 |
| Optional USB power meter (tokens/joule stretch) | M9 | ≈$20–$40 |
| Vivado ML Standard + Vitis (free tier covers both boards) | M2 (synth checks) | $0 |
| Verilator, cocotb, cocotbext-axi, SymbiYosys, PYNQ | M2 | $0 |
| Model training (Colab free tier or any laptop GPU; `tiny` trains on CPU) | M0 | $0 |

**Total: ≈$275–$425 for the KV260 route, ≈$175–$300 for PYNQ-Z2.**

**Machine requirement:** Vivado runs on x86-64 **Linux or Windows only** (not macOS natively, and not on Apple Silicon without an x86 VM, which is painful). Plan on 16 GB RAM minimum, 32 GB comfortable, ~100 GB free disk for the install. If your main machine is a Mac, the options are a cheap x86 Linux box, a dual-boot PC, or a cloud x86 VM for synthesis only. Simulation (Verilator, cocotb) runs fine anywhere, including this cloud environment.

## 11. Risks and mitigations

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| Timing closure at 250 MHz on a 32×32 array with DSP packing | Med / High | Synthesize the array alone in M2, before anything else depends on it. Fall back to 200 MHz or a 16×32 array; the cycle model shows the cost. |
| DDR bandwidth from PL far below theoretical | High / Med | Measure early in M7 with a standalone DMA bandwidth test bitstream; wide (128-bit) bursts, multiple HP ports, outstanding transactions. Decode targets are stated as a fraction of *measured* bandwidth. |
| Integer softmax/LN error breaks perplexity target | Med / Med | M0 owns this before RTL: try LUT vs polynomial, INT16 residuals, SmoothQuant. Targets can be revised in the M0 ADR with data. |
| Bit-exactness drift between golden model and RTL (rounding modes, saturation order) | High / High | The golden model is the spec and specifies rounding/saturation per op; RTL tests compare every op; a layer-dump bisect tool exists from M5. |
| On-board debug is slow and opaque | High / Med | Layer-by-layer dump mode, saturation and perf counters over CSR, ILA on the command processor, single-step mode. Never debug on the board what can be reproduced in Verilator. |
| Vivado / PYNQ toolchain friction (versions, Linux image, device tree) | High / Med | Pin one Vivado version and one PYNQ image in an ADR; script everything in TCL; budget the full 3–4 weeks for M7. |
| GEMV decode leaves the array idle, making the "accelerator" look weak | Certain / Low | Report prefill and decode separately against the roofline; batched decode and INT4 in M8 are the answers; this is a real-world lesson, worth writing up. |
| Scope creep (stretch features before end-to-end works) | Med / High | Nothing from M9 starts before G1 is met on hardware. |
| No x86 machine for Vivado | Unknown / High | Decide in the open questions below before M2. |

## 12. Decisions needed from you before "proceed"

1. **Board:** KV260 (recommended: URAM fits the KV cache, 5× the DSPs, much more DDR bandwidth) or PYNQ-Z2 (cheaper, more tutorials, but tight). The RTL stays parameterized either way.
2. **Model family for v1:** classic GPT-2 nanoGPT (recommended), or start directly with the LLaMA-style variants from nanoGPT-extended.
3. **Repo:** create an empty GitHub repo (any name), since threads cannot create repos; work would land as small sub-goal PRs like the tiny-gpu xf track.
4. **Vivado machine:** do you have an x86 Linux/Windows machine with ≥16 GB RAM?
5. **Ambition dial:** stop at M8, or treat M9 (speculative decoding, GQA measurement, ASIC run) as part of the project.

## 13. References to read during M0–M1
- Gemmini (Berkeley) and VTA (TVM) for decoupled access/execute and systolic generator design.
- I-BERT (Kim et al., 2021) for integer-only GELU, softmax, LayerNorm.
- SmoothQuant (Xiao et al., 2023), GPTQ (Frantar et al., 2023), AWQ (Lin et al., 2024) for weight/activation quantization.
- Jacob et al., 2018 (gemmlowp / TFLite integer-only inference) for requantization.
- FlashAttention / Flash-Decoding for online softmax and tiled attention.
- AMD WP486 "Deep Learning with INT8 Optimization on Xilinx Devices" for 2-MAC DSP packing.
- PYNQ and Kria KV260 getting-started docs for overlays and CMA buffers.
