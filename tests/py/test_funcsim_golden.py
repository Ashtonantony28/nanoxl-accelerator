"""Headline M1 check (SG4b): the compiler + functional simulator run the REAL quantized model exactly
as the integer golden model (nanoxl.golden, contract C5) does.

For the committed tiny W8A8 and W4A8 bundles, compiled programs executed in nanoxl.funcsim must give
bit-identical int16 logits at every step, the same KV cache contents, the same saturation counts per
op, and the same greedy token sequences as nanoxl.golden and artifacts/models/tiny/golden_ref.json.
Covered: single-shot and chunked prefill, on-chip and DDR KV, batched decode at different positions.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from nanoxl import golden, isa
from nanoxl.export import load_bundle
from nanoxl.funcsim import HostRunner, check_hazards
from nanoxl.hwconfig import KV260, PYNQ_Z2

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "artifacts" / "models" / "tiny"
REF = json.loads((TINY / "golden_ref.json").read_text())
SHAKESPEARE_CKPT = Path(os.environ.get("NANOXL_SHAKESPEARE_CKPT",
                                       "/mnt/project-files/nanoxl-artifacts/shakespeare_ckpt.pt"))


def nz(d):
    return {k: v for k, v in d.items() if v}


@pytest.fixture(scope="module", params=["w8a8", "w4a8"])
def bundle(request):
    A, meta = load_bundle(TINY / f"{request.param}.npz")
    return request.param, A, meta


def golden_steps(g, prompt, tokens, chunk=None):
    """Logits of every step of the golden model for prefill (in chunks) + decode of ``tokens``;
    returns (list of logits per step, cache)."""
    cache = g.new_cache()
    chunk = chunk or len(prompt)
    pre = [g.forward(prompt[i:i + chunk], cache) for i in range(0, len(prompt), chunk)]
    steps = [np.concatenate(pre)]
    for t in tokens[:-1]:
        steps.append(g.forward([t], cache))
    return steps, cache


def assert_kv_equal(runner, slot, cache, n):
    cfg = runner.cfg
    kv = runner.kv_cache(slot)
    dh = cfg.d_head
    for l in range(cfg.n_layer):
        for kind, key in ((0, "k"), (1, "v")):
            for h in range(cfg.n_head):
                assert np.array_equal(kv[l, kind, h, :n], cache[key][l, :n, h * dh:(h + 1) * dh]), (l, key, h)
    assert cache["n"] == n


def test_funcsim_golden_generate_matches_reference(bundle):
    """Prefill the 16-token reference prompt in one program, greedy-decode 100 tokens (one program per
    token, on-chip KV): every step's logits, the KV cache, the saturation counts and the token sequence
    equal nanoxl.golden and the committed golden_ref.json."""
    name, A, meta = bundle
    ref = REF["bundles"][name]
    prompt = REF["prompt"]
    r = HostRunner(A, meta, KV260, name=name)
    trace = []
    toks = r.generate(prompt, REF["n_new"], trace=trace)
    assert toks == ref["tokens"]
    g = golden.Golden(A, meta)
    steps, cache = golden_steps(g, prompt, toks)
    assert len(steps) == len(trace) == len(toks)
    for i, (got, want) in enumerate(zip(trace, steps)):
        assert got.dtype == want.dtype == np.int16
        assert np.array_equal(got, want), f"logits differ at step {i}"
    assert nz(r.sat_count) == nz(g.sat_count) == ref["sat_count"]
    assert_kv_equal(r, 0, cache, len(prompt) + len(toks) - 1)
    # the program really uses the C5 / ADR-004 commands (no placeholder path exists any more)
    prog = r.comp.compile_step([(5, 1)])
    ops = {c.op for c in prog.cmds}
    assert {"LUT_LOAD", "LN", "SOFTMAX_ONLINE", "GELU", "REQUANT", "VEC"} <= ops
    if name == "w4a8":
        assert "GEMM_GS" in ops and not any(c.op == "GEMM" and c["int4"] for c in prog.cmds)
    else:
        assert "GEMM_GS" not in ops
    check_hazards(prog.cmds, r.comp.img.base)
    isa.decode_stream(prog.ring_bytes())


@pytest.mark.parametrize("hw,chunk", [(PYNQ_Z2, 5), (KV260.with_(kv_onchip=False), 7), (KV260, 3)],
                         ids=["pynq-ddrKV-chunk5", "kv260-ddrKV-chunk7", "kv260-onchip-chunk3"])
def test_funcsim_golden_chunked_prefill(bundle, hw, chunk):
    """Chunked prefill (several programs per prompt) on both KV tiers, then 12 decode steps; the golden
    model runs the same chunks. Logits also equal the single-shot prefill rows (causality)."""
    name, A, meta = bundle
    prompt = REF["prompt"]
    r = HostRunner(A, meta, hw, name=name)
    trace = []
    toks = r.generate(prompt, 12, chunk=chunk, trace=trace)
    assert r.programs == -(-len(prompt) // chunk) + len(toks) - 1
    assert toks == REF["bundles"][name]["tokens"][:12]
    g = golden.Golden(A, meta)
    steps, cache = golden_steps(g, prompt, toks, chunk)
    for i, (got, want) in enumerate(zip(trace, steps)):
        assert np.array_equal(got, want), f"logits differ at step {i}"
    assert nz(r.sat_count) == nz(g.sat_count)
    assert_kv_equal(r, 0, cache, len(prompt) + len(toks) - 1)
    one_shot = golden.Golden(A, meta).forward(prompt)
    assert np.array_equal(trace[0], one_shot)


@pytest.mark.parametrize("hw", [KV260, KV260.with_(kv_onchip=False)], ids=["onchipKV", "ddrKV"])
def test_funcsim_golden_batched_decode(bundle, hw):
    """Three sequences with different prompts and lengths share every program: one batched prefill step
    (rows of all three slots concatenated) then 8 batched greedy decode steps at different positions.
    Each slot matches its own golden model; saturation counts add up."""
    name, A, meta = bundle
    base = REF["prompt"] + REF["bundles"][name]["tokens"]
    prompts = [base[:16], base[3:12], base[20:33]]
    r = HostRunner(A, meta, hw, n_slots=3, name=name)
    gs = [golden.Golden(A, meta) for _ in prompts]
    caches = [g.new_cache() for g in gs]
    got = r.step([list(p) for p in prompts])
    want = [g.forward(p, c) for g, p, c in zip(gs, prompts, caches)]
    for _ in range(8):
        for a, b in zip(got, want):
            assert np.array_equal(a, b)
        nxt = [int(np.argmax(a[-1])) for a in got]
        got = r.step([[t] for t in nxt])
        want = [g.forward([t], c) for g, t, c in zip(gs, nxt, caches)]
    for a, b in zip(got, want):
        assert np.array_equal(a, b)
    total = {}
    for g in gs:
        for k, v in g.sat_count.items():
            total[k] = total.get(k, 0) + v
    assert nz(r.sat_count) == nz(total)
    for b, (p, c) in enumerate(zip(prompts, caches)):
        assert_kv_equal(r, b, c, len(p) + 8)


@pytest.mark.slow
@pytest.mark.skipif(not SHAKESPEARE_CKPT.exists(), reason="shakespeare checkpoint not present")
@pytest.mark.parametrize("wbits", [8, 4])
def test_funcsim_golden_shakespeare(tmp_path, wbits):
    """The shakespeare model (d=384, 6 layers, ctx 256): bundle exported from the checkpoint (kept out of
    git), 64-token prompt prefilled in chunks of 32, 24 decode steps, bit-exact vs golden."""
    from nanoxl.export import quantize_checkpoint
    from nanoxl.model_fp import load_shakespeare
    from nanoxl.quant.quantize import QuantConfig
    A, meta, _ = quantize_checkpoint(SHAKESPEARE_CKPT, QuantConfig(wbits=wbits))
    _, val, _ = load_shakespeare()
    prompt = [int(t) for t in val[1000:1064]]
    r = HostRunner(A, meta, KV260, name="shakespeare")
    trace = []
    toks = r.generate(prompt, 24, chunk=32, trace=trace)
    g = golden.Golden(A, meta)
    golden.set_gemm(golden.gemm_exact_f64)
    try:
        steps, cache = golden_steps(g, prompt, toks, 32)
        assert golden.Golden(A, meta).generate(prompt, 24) == toks
    finally:
        golden.set_gemm(golden.nx.gemm_i8)
    for i, (a, b) in enumerate(zip(trace, steps)):
        assert np.array_equal(a, b), f"step {i}"
    assert_kv_equal(r, 0, cache, len(prompt) + len(toks) - 1)
    assert nz(r.sat_count) == nz(g.sat_count)
