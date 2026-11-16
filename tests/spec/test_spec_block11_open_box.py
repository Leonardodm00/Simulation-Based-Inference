"""Spec-derived tests of block 11's amendment of 2026-11-14 as the SPEC now
words it (`clip_open_box`, BOX_DELTA; the -0.0 sentence added 2026-11-16),
written by the integration tester from specs/SPEC.md alone.

Run: python3 -m pytest -q tests/spec/test_spec_block11_open_box.py
Pure ASCII, LF only.
"""

import os
import sys

import numpy as np
import numpy.testing as npt
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "hpc", "joint", "stage1"))
import latent_sbi_simulator as LS  # noqa: E402

D = 2.0 ** -24          # "BOX_DELTA = 2.0 ** -24, the spacing of float32 in [1/2, 1)"


def test_box_delta_value():
    assert LS.BOX_DELTA == D
    assert np.float32(1 - D) < 1 and np.float32(D) > 0


def test_known_answers_and_float32_strictly_inside():
    xs = np.array([0.0, 1e-300, 2 ** -25, 2 ** -24, 0.5, 1 - 2 ** -24, 1 - 2 ** -25, 1 - 1e-9,
                   1 - np.finfo(np.float64).eps, 1.0])
    out = LS.clip_open_box(xs)
    npt.assert_array_equal(out, np.clip(xs, D, 1 - D))       # exact: clip moves no other value
    f32 = out.astype(np.float32)
    assert np.all(f32 > 0) and np.all(f32 < 1)
    inside = (xs >= D) & (xs <= 1 - D)
    npt.assert_array_equal(out[inside], xs[inside])


def test_shape_dtype_and_input_untouched():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, size=(4, 3, 2))
    x[0, 0, 0], x[1, 1, 1] = 0.0, 1.0
    keep = x.copy()
    out = LS.clip_open_box(x)
    assert out.shape == x.shape and out.dtype == np.float64
    assert out is not x
    npt.assert_array_equal(x, keep)
    x32 = x.astype(np.float32)
    assert LS.clip_open_box(x32).dtype == np.float64


def test_negative_zero_is_clipped_not_refused():
    # SPEC block 11 (2026-11-16): "-0.0 is not below 0 for clip_open_box: it is clipped to BOX_DELTA"
    out = LS.clip_open_box(np.array([-0.0, 0.25]))
    npt.assert_array_equal(out, [D, 0.25])


@pytest.mark.parametrize("bad", [-0.2, -1e-300, 1.5, 1.0 + 1e-15, np.nan, np.inf, -np.inf])
def test_out_of_range_or_non_finite_raises(bad):
    x = np.array([0.5, bad, 0.25])
    keep = x.copy()
    with pytest.raises(FloatingPointError):
        LS.clip_open_box(x)
    npt.assert_array_equal(x, keep)


def test_sample_box_prior_contract():
    spec = LS.LatentSBISpec() if hasattr(LS, "LatentSBISpec") else None
    if spec is None:
        pytest.skip("no LatentSBISpec default in this module")
    rng = np.random.default_rng(11)
    c, phi = LS.sample_box_prior(spec, 2000, rng)
    assert LS.BOX_CLASS == -1
    assert c.shape == (2000,) and c.dtype == np.int64 and np.all(c == -1)
    assert phi.dtype == np.float64 and phi.shape[0] == 2000
    assert phi.min() >= D and phi.max() <= 1 - D
    f32 = phi.astype(np.float32)
    assert f32.min() > 0 and f32.max() < 1
    from scipy import stats
    # KS against U(0, 1) on every axis; p > 1e-3 (block 11's own threshold) at a fixed seed
    for k in range(phi.shape[1]):
        assert stats.kstest(phi[:, k], stats.uniform(0, 1).cdf).pvalue > 1e-3
