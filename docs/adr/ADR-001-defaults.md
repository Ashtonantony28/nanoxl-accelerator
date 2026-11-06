# ADR-001: Defaults chosen at kickoff (2026-11-06)

Status: accepted (user approved the plan and delegated open decisions to the recommendations).

- Primary board: AMD Kria KV260. RTL is parameterized (array N, vector lanes, buffer depths) so PYNQ-Z2 remains a build target.
- Model family v1: classic GPT-2 nanoGPT (LayerNorm, GELU, learned positions, MHA). LLaMA-style variants are M9.
- Source of the model code: nanoGPT-extended (`Ashtonantony28/nanoGPT_LLMArchitecturalExtension`), pinned commit `24e8407`.
- Simulators: Icarus 12 (default, everywhere) and Verilator ≥ 5.036 (cocotb 2.x needs it). RTL must compile on both: plain modules, `logic`, `always_ff/always_comb`, generate, packed arrays; no interfaces, classes or SV structs in ports.
- Requantization semantics: gemmlowp/TFLite (contract C2).
