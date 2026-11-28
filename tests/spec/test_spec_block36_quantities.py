"""Spec tests of block 36's pure functions (edge split (R1)-(R2), the well's
summary (R4), the flagged shares, k_gate and the size band (R9)-(R10), the
check share, the example well, the window summary (R3)) and of its additive
change to block 20's ev_metrics.py. Integration tester, 2026-11-23,
incremental run on dc096e5.

Expected values are computed here from specs/SPEC.md sec. 2.6 with numpy and
scipy, or (for the change to ev_metrics.py) are the base commit 6c6c297's
outputs, computed in a subprocess on `git archive 6c6c297 hpc`.
"""

import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL = os.path.join(REPO, "hpc", "pipeline", "evaluation")
if EVAL not in sys.path:
    sys.path.insert(0, EVAL)
sys.dont_write_bytecode = True

import ev_rec_metrics as RM  # noqa: E402

BASE = "6c6c297f9fb8b5031d147d7ca01151875d557f39"
BETAS = np.round(np.linspace(0.0, 1.0, 101), 10)     # sec. 2.6: beta = 0, 0.01, ..., 1


# ---------------------------------------------------------------- (R1)

def test_edge_rows_by_hand_strict():
    eps = 0.006693
    th = np.array([[0.5, 0.5, 0.5],
                   [eps, 0.5, 0.5],             # exactly eps: interior ((R1) is strict)
                   [0.5, 1 - 0.5 * eps, 0.5],   # within eps of 1
                   [0.5, 0.5, 0.5 * eps],
                   [0.2, 0.8, 0.99],
                   [1 - 2 * eps, 2 * eps, 0.3]])
    exp = np.array([np.min(np.minimum(r, 1 - r)) < eps for r in th])
    assert_array_equal(RM.edge_rows(th, eps), exp)
    assert list(exp) == [False, False, True, True, False, False]


# ---------------------------------------------------------------- (R2)

def _cov(g, betas=BETAS):
    return np.array([np.mean(g <= b) for b in betas])


def test_edge_split_identity_and_counts():
    rng = np.random.default_rng(1)
    g = rng.uniform(size=500)
    edge = rng.uniform(size=500) < 0.15
    s, a = RM.edge_split(g, edge, 0.006693, 10)
    assert s["n_cal"] == 500 and s["n_edge"] == edge.sum() and s["n_int"] == 500 - edge.sum()
    assert abs(s["pi_edge"] - edge.mean()) < 1e-15
    # pi_edge's value under the box prior, 1 - (1 - 2 eps)^p = 0.1261 at p = 10
    assert abs(s["pi_edge_expected"] - (1 - (1 - 2 * 0.006693) ** 10)) < 1e-15
    assert abs(s["pi_edge_expected"] - 0.1261) < 5e-5
    # the two curves by definition, (R2) within float64 roundoff
    assert_allclose(a["coverage_edge"], _cov(g[edge]), atol=0)
    assert_allclose(a["coverage_int"], _cov(g[~edge]), atol=0)
    whole = _cov(g)
    rhs = s["pi_edge"] * a["coverage_edge"] + (1 - s["pi_edge"]) * a["coverage_int"]
    assert np.max(np.abs(whole - rhs)) < 1e-12
    assert s["r2_max_abs_dev"] < 1e-12


def test_edge_split_empty_sides():
    g = np.linspace(0, 1, 20)
    s, a = RM.edge_split(g, np.zeros(20, bool), 0.01, 3)
    assert s["edge"] is None and s["n_edge"] == 0 and np.all(np.isnan(a["coverage_edge"]))
    assert s["interior"] is not None and s["r2_max_abs_dev"] is None
    s, a = RM.edge_split(g, np.ones(20, bool), 0.01, 3)
    assert s["interior"] is None and np.all(np.isnan(a["coverage_int"]))
    s, a = RM.edge_split(np.zeros(0), np.zeros(0, bool), 0.01, 3)
    assert s["n_cal"] == 0 and s["pi_edge"] is None


def test_edge_split_cut_support_toy():
    """R36a reduced: theta ~ U(0,1)^10, x = theta + 0.15 xi; draws from the
    exact posterior with its support cut to [eps, 1-eps]^10 (a product of
    truncated normals). The edge rows' C^edge(beta) = 0 for every beta < 1
    (theta lies where the cut posterior has density 0, so every draw has
    higher density); pi_edge within 3 SE of 0.1261."""
    eps, p, n, n_post = 0.006693, 10, 1500, 300
    rng = np.random.default_rng(2)
    th = rng.uniform(size=(n, p))
    x = th + 0.15 * rng.standard_normal((n, p))
    lo, hi = (eps - x) / 0.15, (1 - eps - x) / 0.15
    draws = stats.truncnorm.rvs(lo[:, None, :], hi[:, None, :], loc=x[:, None, :], scale=0.15,
                                size=(n, n_post, p), random_state=rng)
    logq = lambda t: np.sum(stats.truncnorm.logpdf(t, lo, hi, loc=x, scale=0.15), axis=-1)  # noqa
    lq_ref = logq(th)
    lq_dr = np.stack([logq(draws[:, s, :]) for s in range(n_post)], axis=1)
    gamma = np.mean(lq_dr > lq_ref[:, None], axis=1)   # sec. 2.5's HPD rank by definition
    edge = RM.edge_rows(th, eps)
    s, a = RM.edge_split(gamma, edge, eps, p)
    pe = 0.12607786820515499
    assert abs(s["pi_edge"] - pe) < 3 * math.sqrt(pe * (1 - pe) / n)
    assert np.all(a["coverage_edge"][BETAS < 1] == 0)
    # C^int near the diagonal: 0.05 for n_int ~ 1300 rows (KS 0.999 bound 1.95/sqrt(1300) = 0.054)
    assert np.max(np.abs(a["coverage_int"] - BETAS)) < 0.054
    assert s["r2_max_abs_dev"] < 1e-12


# ---------------------------------------------------------------- (R4)

def test_well_summary_pooled_by_definition():
    rng = np.random.default_rng(3)
    p, n_post = 3, 400
    well = np.array([7, 7, 7, 2, 2, 9, 7, 2])
    draws = rng.normal(rng.uniform(0.2, 0.8, (8, 1, p)), rng.uniform(0.02, 0.1, (8, 1, p)),
                       size=(8, n_post, p)).astype(np.float32)
    out = RM.well_summary(draws, well)
    assert_array_equal(out["well_ids"], [2, 7, 9])
    assert_array_equal(out["well_n"], [3, 4, 1])
    for gi, w in enumerate([2, 7, 9]):
        pooled = draws[well == w].reshape(-1, p).astype(np.float64)
        # numpy's own statistics of the pooled draws; 1e-12 relative: same sums
        assert_allclose(out["well_mean"][gi], pooled.mean(0), rtol=1e-12)
        assert_allclose(out["well_sd"][gi], pooled.std(0, ddof=1), rtol=1e-12)
        assert_allclose(out["well_q05"][gi], np.quantile(pooled, 0.05, axis=0), rtol=1e-12)
        assert_allclose(out["well_q50"][gi], np.quantile(pooled, 0.5, axis=0), rtol=1e-12)
        assert_allclose(out["well_q95"][gi], np.quantile(pooled, 0.95, axis=0), rtol=1e-12)
        # law of total variance with equal weights: var_pooled (ddof 0) = mean var + var of means
        v0 = pooled.var(0)
        lt = draws[well == w].astype(np.float64).var(1).mean(0) + \
            draws[well == w].astype(np.float64).mean(1).var(0)
        assert_allclose(v0, lt, rtol=1e-10)
    # a well of one window equals that window
    one = draws[well == 9][0].astype(np.float64)
    assert_allclose(out["well_sd"][2], one.std(0, ddof=1), rtol=1e-12)


def test_well_sd_does_not_shrink():
    rng = np.random.default_rng(4)
    s = 0.05
    d8 = rng.normal(0.5, s, (8, 1000, 2))
    d2 = rng.normal(0.5, s, (2, 1000, 2))
    out = RM.well_summary(np.concatenate([d8, d2]), np.r_[np.zeros(8, int), np.ones(2, int)])
    # within 5 %: 8,000 and 2,000 draws, the SD's relative SE is ~1/sqrt(2N) <= 1.6 %
    assert np.all(np.abs(out["well_sd"] / s - 1) < 0.05)


# ---------------------------------------------------------------- shares, size, check

def test_flag_shares_by_hand():
    ids, n, k = RM.flag_shares([1, 0, 1, 1, 0, 0], [5, 5, 3, 9, 3, 3])
    assert_array_equal(ids, [3, 5, 9])
    assert_array_equal(n, [3, 2, 1])
    assert_array_equal(k, [1, 1, 1])


@pytest.mark.parametrize("alpha,N,k", [(0.05, 500, 25), (0.05, 99, 4), (0.05, 19, 0),
                                       (0.5, 2, 1), (0.05, 1000, 50), (0.05, 50, 2),
                                       (0.1, 99, 9), (0.3, 9, 2)])
def test_gate_size_kgate_and_band(alpha, N, k):
    s = RM.gate_size(alpha, N, 0.95)
    assert s["k_gate"] == k
    if k == 0:
        assert s["size_mean"] is None and s["size_lo"] is None
        return
    b = stats.beta(k, N + 1 - k)
    assert abs(s["size_mean"] - k / (N + 1)) < 1e-15 and s["size_mean"] <= alpha + 1e-15
    assert abs(s["size_lo"] - b.ppf(0.025)) < 1e-12 and abs(s["size_hi"] - b.ppf(0.975)) < 1e-12


def test_kgate_matches_flag_rule_exhaustively():
    """(R9): p < alpha iff c < k_gate, c the number of null values >= T, for
    every count c and the configuration's alphas with float roundoff."""
    for N in range(1, 300):
        for alpha in (0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.3, 0.07):
            k = RM.gate_size(alpha, N, 0.95)["k_gate"]
            flagged = [c for c in range(N + 1) if (1.0 + c) / (1.0 + N) < alpha]
            assert flagged == list(range(k)), (alpha, N, k, flagged[-3:])


def test_check_share_clopper_pearson():
    f = np.r_[np.ones(3, bool), np.zeros(37, bool)]
    s = RM.check_share(f, 0.95)
    ci = stats.binomtest(3, 40).proportion_ci(0.95, method="exact")
    assert s["n"] == 40 and s["n_flagged"] == 3 and abs(s["share"] - 0.075) < 1e-15
    assert_allclose(s["ci"], [ci.low, ci.high], rtol=1e-14)


def test_example_wells_median_mean_T():
    # class 0: 5 wells with mean T 0.1..0.5 -> the median 0.3 (well 12);
    # class 1: 4 wells -> the lower middle (2nd smallest)
    T, well, cls = [], [], []
    for w, (c, m) in {10: (0, .1), 11: (0, .5), 12: (0, .3), 13: (0, .2), 14: (0, .4),
                      20: (1, .9), 21: (1, .6), 22: (1, .8), 23: (1, .7)}.items():
        for _ in range(3):
            T.append(m); well.append(w); cls.append(c)
    out = RM.example_wells(np.array(T), np.array(well), np.array(cls))
    got = dict((x["cls"], (x["well"], x["first_window"])) for x in out)
    assert got[0][0] == 12 and got[1][0] == 23
    assert got[0][1] == well.index(12) and got[1][1] == well.index(23)


def test_window_summary_by_definition():
    """(R3) per window: the draws' mean, SD, quantiles, chi = 1 - sigma^2/(1/12)."""
    rng = np.random.default_rng(5)
    d = rng.normal(rng.uniform(0.3, 0.7, (5, 1, 4)), 0.05, (5, 4000, 4)).astype(np.float32)
    w = RM.window_summary(d)
    dd = d.astype(np.float64)
    assert_allclose(w["mhat"], dd.mean(1), rtol=1e-10)
    assert_allclose(w["q50"], np.quantile(dd, 0.5, axis=1), rtol=1e-10)
    assert_allclose(w["q05"], np.quantile(dd, 0.05, axis=1), rtol=1e-6)
    assert_allclose(w["q95"], np.quantile(dd, 0.95, axis=1), rtol=1e-6)
    # sigma: the draws' SD (ddof 0 or 1 differ by 1/8000 relative; 2e-4 covers either)
    assert_allclose(w["sigma"], dd.std(1, ddof=1), rtol=2e-4)
    assert_allclose(w["contraction"], 1 - 12 * w["sigma"] ** 2, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- ev_metrics change (block 20)

_OLD = r"""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1]); sys.dont_write_bytecode = True
import ev_metrics as EM
rng = np.random.default_rng(11)
th = rng.uniform(size=(60, 10)); dr = rng.uniform(size=(60, 50, 10))
s, a = EM.sbc_block(th, dr, 0.95, 3)
rec = EM.recovery_arrays(th, dr)
rows = np.arange(60) % 2 == 0
r = EM.recovery_summary(th, rec, rows)
np.savez(sys.argv[2], **dict(("a_" + k, v) for k, v in a.items()))
json.dump({"s": EM._clean(s), "r": EM._clean(r)}, open(sys.argv[3], "w"))
"""


def test_ev_metrics_axes_none_unchanged_and_named():
    import ev_metrics as EM
    with tempfile.TemporaryDirectory() as td:
        base = os.environ.get("REC_SPEC_BASE", "/tmp/tst/base")
        if not os.path.isdir(os.path.join(base, "hpc")):
            os.makedirs(base, exist_ok=True)
            subprocess.check_call("git -C %s archive %s hpc | tar -x -C %s" % (REPO, BASE, base),
                                  shell=True)
        sc = os.path.join(td, "old.py")
        open(sc, "w").write(_OLD)
        subprocess.check_call([sys.executable, sc, os.path.join(base, "hpc/pipeline/evaluation"),
                               os.path.join(td, "a.npz"), os.path.join(td, "s.json")])
        old = json.load(open(os.path.join(td, "s.json")))
        oa = np.load(os.path.join(td, "a.npz"))
        rng = np.random.default_rng(11)
        th = rng.uniform(size=(60, 10)); dr = rng.uniform(size=(60, 50, 10))
        s, a = EM.sbc_block(th, dr, 0.95, 3)
        rec = EM.recovery_arrays(th, dr)
        rows = np.arange(60) % 2 == 0
        r = EM.recovery_summary(th, rec, rows)
        assert json.loads(json.dumps(EM._clean(s))) == old["s"]
        assert json.loads(json.dumps(EM._clean(r))) == old["r"]
        for k, v in a.items():
            assert_array_equal(v, oa["a_" + k])
        # with 14 named axes
        names = ["ax%d" % i for i in range(14)]
        th14 = rng.uniform(size=(40, 14)); dr14 = rng.uniform(size=(40, 30, 14))
        s14, a14 = EM.sbc_block(th14, dr14, 0.95, 1, axes=names)
        assert a14["sbc_ranks"].shape == (40, 14) and len(s14["ks_p"]) == 14
        r14 = EM.recovery_summary(th14, EM.recovery_arrays(th14, dr14), np.ones(40, bool),
                                  axes=names)
        assert r14["axes"] == names and len(r14["contraction"]) == 14
