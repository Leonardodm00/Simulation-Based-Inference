"""Spec-derived tests of block 13 (the shifted duty clipped at 1; SPEC.md sec.
2.4 gap (a), block 13). Oracles O13.1-O13.3 of the t1-B report.

The expected parameters are computed here from sec. 2.4's map,
a_k = L_k (1 - theta_k) + U_k theta_k, the free axes' ranges displaced by
delta (U_k - L_k), then a_7 = min(a_7, 1). The ranges [L_k, U_k] are the
table the spec names (`BENCH_AXES`); the by-hand numbers are block 13's.
The code before blocks 11-14 (`git archive 59465e1^`) is the reference for
"the clip acts only where the old code refused".
"""
import os

import numpy as np
import pytest

import t1b_helpers as H

H.joint_paths()
import bench_burst_provider as BP      # noqa: E402  (code under test)

FIELDS = ("lambda_b", "sigma_d", "ibi_cv", "n_frag_mean", "lambda_burst", "p_mean", "d_med",
          "duty", "kappa", "lambda_bg")          # BenchBurstParams, in the axis order of BENCH_AXES


def _params(phi, shift, free_idx=None):
    kw = {} if free_idx is None else {"free_idx": free_idx}
    p = BP.phi_to_bench_params(np.asarray(phi, dtype=np.float64), n_neurons=10, duration_s=5.0,
                               free_axis_shift=shift, **kw)
    return np.array([getattr(p, f) for f in FIELDS], dtype=np.float64)


def _expected(phi, shift, free=(7, 8, 9)):
    """sec. 2.4 (a), written out here."""
    out = np.empty(10)
    for k, (_n, lo, hi) in enumerate(BP.BENCH_AXES):
        if k in free:
            w = hi - lo
            lo, hi = lo + shift * w, hi + shift * w
        out[k] = lo * (1.0 - phi[k]) + hi * phi[k]
    out[7] = min(out[7], 1.0)
    return out


def test_o13_1_by_hand():
    assert BP.BENCH_AXES[7][1:] == (0.35, 1.00)              # L_7, U_7 behind the spec's numbers
    delta = 0.35 * 0.3
    phi = np.full(10, 0.5)
    phi[7] = 0.95
    raw = 0.41825 * 0.05 + 1.06825 * 0.95                      # the spec's 1.0357
    assert abs(raw - 1.0357) < 5e-5
    assert _params(phi, delta)[7] == 1.0                       # clipped
    phi[7] = 0.5
    assert abs(_params(phi, delta)[7] - 0.74325) <= 1e-15      # unclipped (the spec's 1e-15)
    phi[7] = 1.0
    assert _params(phi, 0.0)[7] == 1.0                          # duty 1 at theta_7 = 1, shift 0
    # the threshold: theta_7 = 1 - delta = 0.895
    phi[7] = 0.895 - 1e-9
    assert _params(phi, delta)[7] < 1.0
    phi[7] = 0.895 + 1e-9
    assert _params(phi, delta)[7] == 1.0


def test_o13_2_map_on_random_theta():
    """2,000 random theta: every parameter equals sec. 2.4's map, shifted
    (delta 0.105) and unshifted, to 1e-12 (float64 roundoff of a convex
    combination of O(100) numbers is ~1e-14); the label axes never move."""
    rng = np.random.default_rng(13)
    for phi in rng.uniform(0, 1, size=(2000, 10)):
        np.testing.assert_allclose(_params(phi, 0.105), _expected(phi, 0.105), rtol=0, atol=1e-12)
        np.testing.assert_allclose(_params(phi, 0.0), _expected(phi, 0.0), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(_params(phi, 0.105)[:7], _params(phi, 0.0)[:7])


def test_o13_2_clip_identity_at_shift_0():
    for t in np.linspace(0, 1, 101):
        phi = np.full(10, 0.3)
        phi[7] = t
        assert _params(phi, 0.0)[7] == 0.35 * (1 - t) + 1.0 * t


R3_ARGS = ["--arm", "R", "--provider", "bench", "--donors-per-class", "9", "--n-traces", "54",
           "--wells-per-donor", "2", "--n-windows", "1", "--pi", "0.3",
           "--gap-modes", "range_shift,drift,contamination", "--seed", "31",
           "--fs", "10.0", "--T-win", "12.8"]


@pytest.mark.parametrize("centres", [[], ["--class-centres", "regular", "--centre-scale", "0.6"]],
                         ids=["simplex", "regular0.6"])
def test_o13_3_pi03_cohort_builds_4_clipped(centres):
    """block 13: a pi = 0.3 cohort of 3 x 9 donors x 2 wells with the three
    modes builds; 4 of its 27 donors clipped at seed 31 (theta_7 > 0.895)."""
    d = H.tmpdir("t1b_b13_")
    try:
        rc, out = H.run_builder(d, R3_ARGS + centres)
        assert rc == 0, out
        a, side = H.load_shard(os.path.join(d, "shard_0000.npz"))
        th7 = np.array([a["theta"][a["donor"] == k][0, 7] for k in np.unique(a["donor"])],
                       dtype=np.float64)
        assert th7.size == 27
        assert int(np.sum(th7 > 1 - 0.105)) == 4, np.sort(th7)[-6:]
    finally:
        H.rmtree(d)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_o13_3_clip_acts_only_where_old_code_refused(seed):
    """A pi = 0.3 range-shift bank of 6 donors: when no donor has theta_7 >
    0.895 the code before blocks 11-14 builds the same arrays; when one does,
    the old code refuses the shard and the new one builds it."""
    args = ["--arm", "R", "--provider", "bench", "--n-traces", "12", "--wells-per-donor", "2",
            "--n-windows", "1", "--pi", "0.3", "--gap-modes", "range_shift", "--seed", str(seed),
            "--fs", "10.0", "--T-win", "12.8"]
    dn, do = H.tmpdir("t1b_b13n_"), H.tmpdir("t1b_b13o_")
    try:
        rc, out = H.run_builder(dn, args)
        assert rc == 0, out
        a_new, _ = H.load_shard(os.path.join(dn, "shard_0000.npz"))
        clipped = bool(np.any(a_new["theta"][:, 7].astype(np.float64) > 0.895))
        rc_old, out_old = H.run_builder(do, args, tree=H.old_tree(H.PRE_BLOCK11))
        if clipped:
            assert rc_old != 0, "the old code built a shard the clip was needed for"
        else:
            assert rc_old == 0, out_old
            a_old, _ = H.load_shard(os.path.join(do, "shard_0000.npz"))
            for k in a_old:
                np.testing.assert_array_equal(a_new[k], a_old[k], err_msg=k)
    finally:
        H.rmtree(dn)
        H.rmtree(do)
