"""Quantizer range checks (AUDIT-1 A1-08): an int32 bias that does not fit raises instead of wrapping."""
import numpy as np
import pytest

from nanoxl.quant.quantize import qbias


def test_qbias_rounds_like_before():
    bf = np.array([0.5, -1.25, 3.0, 1e3], np.float32)
    s = 0.01
    assert np.array_equal(qbias(bf, s), np.round(bf / s).astype(np.int32))


@pytest.mark.parametrize("v", [2.0 ** 31, -(2.0 ** 31) - 2, 1e12, np.inf, np.nan])
def test_qbias_out_of_int32_range_raises(v):
    with pytest.raises(ValueError, match="int32"):
        qbias(np.array([0.0, v]), 1.0)


def test_qbias_int32_extremes_fit():
    q = qbias(np.array([2.0 ** 31 - 1, -(2.0 ** 31)]), 1.0)
    assert q.tolist() == [2 ** 31 - 1, -(2 ** 31)]


def test_committed_bundles_bias_fits_with_margin():
    """The committed bundles are far from the limit (max |bias| about 2^19), so the check never fires."""
    from pathlib import Path
    from nanoxl.export import load_bundle
    root = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "tiny"
    for name in ("w8a8", "w4a8"):
        A, _ = load_bundle(root / f"{name}.npz")
        mx = max(int(np.abs(v.astype(np.int64)).max()) for k, v in A.items() if k.endswith(".bias"))
        assert mx < 2 ** 24, (name, mx)
