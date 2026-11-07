"""Whole-model integer golden: determinism vs committed reference, KV-cache == full recompute
(bit-exact), held-out accuracy budget, bundle schema, forced saturation."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from nanoxl import golden
from nanoxl import numerics as nx
from nanoxl.export import load_bundle, save_bundle

HELDOUT_SEED = 0x5EED2026      # frozen; never used for tuning (see test_golden_ops.py)
TINY = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "tiny"
REF = json.loads((TINY / "golden_ref.json").read_text())


@pytest.fixture(scope="module", params=["w8a8", "w4a8"])
def bundle(request):
    A, meta = load_bundle(TINY / f"{request.param}.npz")
    return request.param, A, meta


@pytest.fixture(autouse=True)
def _contract_gemm():
    golden.set_gemm(nx.gemm_i8)
    yield
    golden.set_gemm(nx.gemm_i8)


def test_fast_gemm_equals_contract_gemm():
    rng = np.random.default_rng(HELDOUT_SEED)
    for m, k, n in [(1, 128, 65), (37, 384, 129), (5, 4096, 3)]:
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        w = rng.integers(-128, 128, (k, n), dtype=np.int8)
        assert np.array_equal(golden.gemm_exact_f64(a, w), nx.gemm_i8(a, w))
    a = np.full((2, 4096), -128, np.int8)
    assert np.array_equal(golden.gemm_exact_f64(a, a.T.copy()), nx.gemm_i8(a, a.T.copy()))


def test_reproduces_committed_reference_tokens(bundle):
    name, A, meta = bundle
    g = golden.Golden(A, meta)
    toks = g.generate(REF["prompt"], REF["n_new"])
    assert toks == REF["bundles"][name]["tokens"]
    # saturation events are part of the deterministic behaviour (RTL counters must match)
    assert {k: v for k, v in g.sat_count.items() if v} == REF["bundles"][name]["sat_count"]


def test_kv_cache_decode_equals_full_recompute(bundle):
    _, A, meta = bundle
    g = golden.Golden(A, meta)
    rng = np.random.default_rng(HELDOUT_SEED + 1)
    toks = rng.integers(0, meta["vocab_size"], 48)
    full_tr = {}
    full = g.forward(toks, trace=full_tr)
    cache = g.new_cache()
    rows = [g.forward(toks[:20], cache)]                 # prefill
    rows.append(g.forward(toks[20:31], cache))           # chunked prefill
    step_tr = {}
    for t in toks[31:]:                                  # single-token decode
        rows.append(g.forward([t], cache, trace=step_tr))
    inc = np.concatenate(rows)
    assert inc.dtype == np.int16 and np.array_equal(inc, full)
    fc = g.new_cache()
    g.forward(toks, fc)
    assert np.array_equal(cache["k"], fc["k"]) and np.array_equal(cache["v"], fc["v"])
    for key in ("resid.1.mid", "gelu.2", "ln_f", "y.3"):  # last row of each layer tensor
        assert np.array_equal(step_tr[key][-1], full_tr[key][-1]), key


def test_generate_cache_vs_nocache(bundle):
    _, A, meta = bundle
    g = golden.Golden(A, meta)
    p = REF["prompt"]
    assert g.generate(p, 24, use_cache=True) == g.generate(p, 24, use_cache=False)


def test_heldout_accuracy_budget(bundle):
    """Held-out val windows (frozen seed): integer CE within budget of FP32 CE."""
    name, A, meta = bundle
    from nanoxl.model_fp import load_shakespeare, load_checkpoint, numpy_weights
    from nanoxl.quant.fpref import forward_f
    _, val, _ = load_shakespeare()
    model = load_checkpoint(TINY / "ckpt.pt")
    W = numpy_weights(model)
    T = meta["block_size"]
    rng = np.random.default_rng(HELDOUT_SEED)
    starts = rng.integers(0, len(val) - T - 1, 4)
    golden.set_gemm(golden.gemm_exact_f64)
    g = golden.Golden(A, meta)

    def ce(l, y):
        l = l - l.max(-1, keepdims=True)
        return float(-(l[np.arange(len(y)), y] - np.log(np.exp(l).sum(-1))).mean())
    d_ce, agree = [], []
    for s in starts:
        x = val[s:s + T + 1].astype(np.int64)
        lf = forward_f(W, model.config, x[None, :T])[0]
        li = g.forward(x[:T]).astype(np.float64) * meta["scales"]["logits"]
        d_ce.append(ce(li, x[1:]) - ce(lf, x[1:]))
        agree.append(np.mean(lf.argmax(-1) == li.argmax(-1)))
    # Budgets: the G2 targets (+3 % / +8 % PPL). Top-1 floors set from the study windows
    # (STUDY_SEED, docs/quant-study.md: 0.946 W8A8, 0.881 W4A8) minus a margin for 4-window noise.
    budget = math.log(1.03) if name == "w8a8" else math.log(1.08)
    assert np.mean(d_ce) < budget, d_ce
    assert np.mean(agree) > (0.92 if name == "w8a8" else 0.84), agree


def test_bundle_schema(bundle):
    _, A, meta = bundle
    assert meta["format"] == "nanoxl-bundle" and meta["version"] == 1
    L, d, V, T = meta["n_layer"], meta["n_embd"], meta["vocab_size"], meta["block_size"]
    assert A["emb.tok"].shape == (V, d) and A["emb.tok"].dtype == np.int16
    assert A["emb.pos"].shape == (T, d)
    for k in ("lut.exp2", "lut.recip", "lut.rsqrt"):
        assert A[k].shape == (256,) and A[k].dtype == np.uint16
    for l in range(L):
        assert A[f"h{l}.gelu.lut"].shape == (256,) and A[f"h{l}.gelu.lut"].dtype == np.int8
        for ln in ("ln_1", "ln_2"):
            assert A[f"h{l}.{ln}.g"].dtype == np.int16 and A[f"h{l}.{ln}.b"].dtype == np.int32
    for k, v in A.items():
        if k.endswith(".mult"):
            assert v.dtype == np.int32 and v.min() >= 1 << 30
            sh = A[k[:-5] + ".shift"]
            assert sh.min() >= 0 and sh.max() <= 31
        if k.endswith(".w"):
            assert v.dtype == np.int8 and v.min() >= -127
        if k.endswith(".w4"):
            assert v.dtype == np.uint8 and A[k[:-3] + ".gm"].dtype == np.uint8
            assert A[k[:-3] + ".gm"].shape[0] * meta["quant"]["group"] == v.shape[0]
    assert "lm_head.w" in A and A["lm_head.w"].shape == (d, V)


def test_bundle_roundtrip(tmp_path, bundle):
    _, A, meta = bundle
    save_bundle(tmp_path / "b.npz", A, meta)
    A2, meta2 = load_bundle(tmp_path / "b.npz")
    assert meta2 == json.loads(json.dumps(meta, sort_keys=True))
    assert set(A2) == set(A) and all(np.array_equal(A[k], A2[k]) for k in A)


def test_forced_residual_saturation_is_counted():
    A, meta = load_bundle(TINY / "w8a8.npz")
    A = dict(A)
    A["emb.tok"] = np.full_like(A["emb.tok"], 30000)
    A["emb.pos"] = np.full_like(A["emb.pos"], 30000)
    g = golden.Golden(A, meta)
    out = g.forward([1, 2, 3])
    assert g.sat_count["resid"] >= 3 * meta["n_embd"]
    assert out.dtype == np.int16


def test_context_overflow_rejected():
    A, meta = load_bundle(TINY / "w8a8.npz")
    g = golden.Golden(A, meta)
    with pytest.raises(AssertionError):
        g.forward(np.zeros(meta["block_size"] + 1, np.int64))
