#!/usr/bin/env python3
"""Smoke test of block 36: the quantities of a recorded project and the validation tool (specs/SPEC.md block 36, "Test
oracles", checks R36*).

CONTRACT SKELETON (wave B): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 36's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_r36a_edge           the edge split: (R1) by hand, the toy's two posteriors, (R2)
    test_r36b_recording      a recording bank's quantities and the well's mixture, by definition
    test_r36c_gate           the gate's wiring, the shares, k_gate and the size band, the example well
    test_r36d_as_recorded    D-162's second target: the metrics on D_sim equal block 20's
    test_r36e_validate       ev_rec_validate: identical trees pass, four perturbations fail
    test_r36f_axes           ev_metrics' optional axes: unchanged without, named with
    test_r36g_end_to_end     the recorded-kind project through the sampler and the metrics

Run
    python3 hpc/pipeline/tests/test_smoke_rec_metrics.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<dir> reuses a fixture tree built
        before (rec_fixture.py); REC_KEEP=1 keeps the one this run builds.

Pure ASCII, LF only.
"""

import os
import sys
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_EVAL = os.path.join(_PIPE, "evaluation")
for _p in (_HERE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rec_fixture  # noqa: E402

import atexit  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

SEED = 20261123  # every random number of this file derives from it

_TMP = []


def _cleanup_tmp():
    if os.environ.get("REC_KEEP", "").strip() == "1":
        for d in _TMP:
            print("[test_smoke_rec_metrics] kept: %s" % d, flush=True)
        return
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_tmp)


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="r36%s_" % tag)
    _TMP.append(d)
    return d


def _np():
    import numpy as np
    return np


def _close(a, b, tol, what):
    """a and b of one shape, NaN at the same places, |a - b| <= tol elsewhere."""
    np = _np()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, "%s: shapes %s and %s" % (what, a.shape, b.shape)
    na, nb = np.isnan(a), np.isnan(b)
    assert np.array_equal(na, nb), "%s: NaN at different places" % what
    if a.size and not na.all():
        d = float(np.max(np.abs(a[~na] - b[~na])))
        assert d <= tol, "%s: largest |diff| %.3g > %.3g" % (what, d, tol)


def _num(v):
    return float("nan") if v is None else v


def _close_json(a, b, tol, what):
    """Two JSON values: numbers within tol (null as NaN), the rest equal."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a) == sorted(b), "%s: keys %s and %s" % (
            what, sorted(a), sorted(b) if isinstance(b, dict) else b)
        for k in a:
            _close_json(a[k], b[k], tol, "%s.%s" % (what, k))
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), "%s: %r and %r" % (what, a, b)
        for i, (x, y) in enumerate(zip(a, b)):
            _close_json(x, y, tol, "%s[%d]" % (what, i))
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str):
        assert a == b, "%s: %r and %r" % (what, a, b)
    elif a is None or b is None or isinstance(a, (int, float)):
        x, y = _num(a), _num(b)
        assert (math.isnan(x) and math.isnan(y)) or abs(x - y) <= tol, "%s: %r and %r" % (
            what, a, b)
    else:
        assert a == b, "%s: %r and %r" % (what, a, b)


def _write_ecfg(src, out_root, name, **over):
    """A copy of the evaluation configuration `src` with another out_root (and
    `over`), written into out_root's parent; returns its path."""
    with open(src) as fh:
        d = json.load(fh)
    d["out_root"] = out_root
    d.update(over)
    path = os.path.join(os.path.dirname(out_root), name)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    return path


def _toy_ecfg(alpha=0.5):
    """An evaluation configuration of the fixture's values (no fixture
    needed), out_root in a temporary directory."""
    import ev_rec_config
    root = _tmpdir("cfg")
    d = rec_fixture.rec_eval_dict(os.path.join(root, "out"), gate={"alpha": alpha})
    path = os.path.join(root, "toy_rec_eval.json")
    with open(path, "w") as fh:
        json.dump(d, fh)
    return ev_rec_config.load(path)


# ------------------------------------------------------------------ R36a ---

def _toy_gammas(theta, x, s, lo, hi, n_draws, rng):
    """HPD ranks of theta under the exact posterior of the toy: a product of
    N(x_k, s^2) truncated to [lo, hi]. The density at a draw is compared with
    the density at theta through -sum_k (t_k - x_k)^2 (the normaliser depends
    on x alone and cancels); theta outside [lo, hi]^p has density 0."""
    np = _np()
    from scipy.stats import truncnorm
    import ev_metrics as EM
    n, p = theta.shape
    g = np.empty(n)
    inside = np.all((theta >= lo) & (theta <= hi), axis=1)
    lq_ref = np.where(inside, -np.sum((theta - x) ** 2, axis=1), -np.inf)
    for s0 in range(0, n, 400):
        xc = x[s0:s0 + 400][:, None, :]
        d = truncnorm.rvs((lo - xc) / s, (hi - xc) / s, loc=xc, scale=s,
                          size=(xc.shape[0], n_draws, p), random_state=rng)
        g[s0:s0 + 400] = EM.hpd_ranks(-np.sum((d - xc) ** 2, axis=2), lq_ref[s0:s0 + 400])
    return g


def test_r36a_edge():
    """the edge split: (R1) by hand, the toy's two posteriors, (R2) (specs/SPEC.md block 36)."""
    np = _np()
    from scipy.stats import kstwo
    import ev_metrics as EM
    import ev_rec_metrics as RM
    # (R1) by hand on six rows of p = 3; eps = 0.125 is exact in binary, so
    # 1 - 0.875 = eps exactly: a coordinate at eps is interior ((R1) is strict)
    th = np.array([[0.5, 0.5, 0.5], [0.125, 0.5, 0.5], [0.5, 0.875, 0.5],
                   [0.1, 0.5, 0.5], [0.5, 0.5, 0.95], [0.3, 0.7, 0.13]])
    got = RM.edge_rows(th, 0.125)
    assert got.tolist() == [False, False, False, True, True, False], got
    eps = 0.006693
    th2 = np.array([[eps, 0.5, 0.5], [0.5, 0.0066929, 0.5], [0.5, 0.5, 0.9933075],
                    [0.2, 0.2, 0.2], [0.99999, 0.5, 0.5], [0.5, 0.5, 0.5]])
    got = RM.edge_rows(th2, eps)
    assert got.tolist() == [False, True, True, False, True, False], got

    # the toy of the tool's document: theta ~ U(0, 1)^10, x = theta + 0.15 xi
    rng = np.random.default_rng(SEED)
    n, p, nd, s = 4000, 10, 1000, 0.15
    theta = rng.uniform(size=(n, p))
    x = theta + s * rng.standard_normal((n, p))
    edge = RM.edge_rows(theta, eps)

    def r2_by_hand(g, summ, arr):
        whole = EM.coverage_curve(g)
        pi = summ["pi_edge"]
        return float(np.max(np.abs(whole - (pi * arr["coverage_edge"]
                                            + (1.0 - pi) * arr["coverage_int"]))))

    # (1) the exact posterior: C_E within the Kolmogorov bound at 0.999 of the
    # diagonal (n = 4,000 independent rows: kstwo.ppf(0.999, 4000) = 0.031;
    # the 1,000 draws add a grid of 1e-3), no claim on the split curves
    g1 = _toy_gammas(theta, x, s, 0.0, 1.0, nd, rng)
    c1 = EM.coverage_curve(g1)
    dev1 = float(np.max(np.abs(c1 - EM.BETAS)))
    bound = float(kstwo.ppf(0.999, n)) + 1.0 / nd
    assert dev1 <= bound, "exact posterior: max |C - beta| %.4f > %.4f" % (dev1, bound)
    s1, a1 = RM.edge_split(g1, edge, eps, p)
    # (R2) is an identity of the two partitions: float64 roundoff of sums of
    # at most 4,000 terms of 1/4000, far below 1e-12
    assert s1["r2_max_abs_dev"] is not None and s1["r2_max_abs_dev"] <= 1e-12, s1
    assert r2_by_hand(g1, s1, a1) <= 1e-12

    # (2) the support cut to [eps, 1 - eps]^10: the truth has density 0 on the
    # edge rows
    g2 = _toy_gammas(theta, x, s, eps, 1.0 - eps, nd, rng)
    s2, a2 = RM.edge_split(g2, edge, eps, p)
    pi_exp = 1.0 - (1.0 - 2.0 * eps) ** p
    assert abs(s2["pi_edge_expected"] - pi_exp) <= 1e-15 and abs(pi_exp - 0.1261) < 5e-5, s2
    se = math.sqrt(pi_exp * (1.0 - pi_exp) / n)               # binomial SE, 0.0052
    assert abs(s2["pi_edge"] - pi_exp) <= 3.0 * se, (s2["pi_edge"], pi_exp, se)
    assert s2["n_cal"] == n and s2["n_edge"] + s2["n_int"] == n
    # C^edge(beta) = 0 for every beta < 1 of the grid: every edge row has gamma = 1
    assert np.all(a2["coverage_edge"][:-1] == 0.0) and a2["coverage_edge"][-1] == 1.0, a2
    # C^int within 0.03 of the diagonal (the tool's document measured 0.0069
    # on this toy; a property of the toy, not a theorem)
    dint = float(np.max(np.abs(a2["coverage_int"] - EM.BETAS)))
    assert dint <= 0.03, "C^int max |dev| %.4f > 0.03" % dint
    # C(0.95) within 0.02 of 0.95 (1 - pi_edge)
    c95 = float(EM.coverage_curve(g2)[95])
    want = 0.95 * (1.0 - s2["pi_edge"])
    assert abs(c95 - want) <= 0.02, (c95, want)
    assert s2["r2_max_abs_dev"] <= 1e-12 and r2_by_hand(g2, s2, a2) <= 1e-12, s2
    # an empty edge set: null and no error
    s3, a3 = RM.edge_split(g1[~edge][:50], np.zeros(50, dtype=bool), eps, p)
    assert s3["edge"] is None and s3["r2_max_abs_dev"] is None and s3["pi_edge"] == 0.0
    assert np.all(np.isnan(a3["coverage_edge"])) and s3["interior"] is not None
    s4, a4 = RM.edge_split(np.zeros(0), np.zeros(0, dtype=bool), eps, p)
    assert s4["pi_edge"] is None and s4["edge"] is None and s4["interior"] is None
    print("    R36a: exact posterior max|C - beta| %.4f (bound %.4f); cut support pi_edge %.4f "
          "(expected %.4f, SE %.4f), C^int max|dev| %.4f, C(0.95) %.4f against %.4f; R2 %.2g, %.2g"
          % (dev1, bound, s2["pi_edge"], pi_exp, se, dint, c95, want, s1["r2_max_abs_dev"],
             s2["r2_max_abs_dev"]), flush=True)


# ------------------------------------------------------------------ R36b ---

def test_r36b_recording():
    """a recording bank's quantities and the well's mixture, by definition (specs/SPEC.md block 36)."""
    np = _np()
    from scipy.stats import norm
    import ev_rec_metrics as RM
    rng = np.random.default_rng(SEED + 1)
    nw, N, p = 6, 4000, 3
    mu = rng.uniform(0.3, 0.7, size=(nw, p))
    sd = rng.uniform(0.02, 0.1, size=(nw, p))
    draws = (mu[:, None, :] + sd[:, None, :] * rng.standard_normal((nw, N, p))).astype(np.float32)
    w = RM.window_summary(draws)
    # each within 4 Monte Carlo standard errors of the law's value: the mean
    # s / sqrt(N); the SD s / sqrt(2 (N - 1)); a quantile sqrt(q (1 - q) / N)
    # / f(x_q), f the normal density at the quantile
    worst = 0.0
    for key, q in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95)):
        zq = norm.ppf(q)
        se = math.sqrt(q * (1 - q) / N) * sd / norm.pdf(zq)
        r = np.abs(w[key] - (mu + sd * zq)) / se
        worst = max(worst, float(r.max()))
        assert np.all(r <= 4.0), (key, r)
    r = np.abs(w["mhat"] - mu) / (sd / math.sqrt(N))
    assert np.all(r <= 4.0), r
    r2 = np.abs(w["sigma"] - sd) / (sd / math.sqrt(2.0 * (N - 1)))
    assert np.all(r2 <= 4.0), r2
    # chi = 1 - 12 sigma^2 from the returned sigma: float64 roundoff only
    assert np.max(np.abs(w["contraction"] - (1.0 - 12.0 * w["sigma"] ** 2))) <= 1e-12
    # the well's mixture (R4): wells 7 (3 windows), 3 (2), 9 (1)
    well = np.array([7, 7, 7, 3, 3, 9])
    ws = RM.well_summary(draws, well)
    assert ws["well_ids"].tolist() == [3, 7, 9] and ws["well_n"].tolist() == [2, 3, 1], ws
    d64 = draws.astype(np.float64)
    for gi, g in enumerate([3, 7, 9]):
        sel = well == g
        m_j = d64[sel].mean(axis=1)                       # (n_g, p) the windows' means
        v_j = d64[sel].var(axis=1, ddof=1)                # their variances
        ng = int(sel.sum())
        # the pooled mean is the mean of the windows' means (every window has
        # N draws): equal up to float64 roundoff; 1e-6 as the entry states
        assert np.max(np.abs(ws["well_mean"][gi] - m_j.mean(axis=0))) <= 1e-6
        # the law of total variance with its finite-sample factors: the pooled
        # sum of squares is sum_j [(N - 1) v_j + N (m_j - m)^2], over n_g N - 1
        tot = ((N - 1) * v_j.sum(axis=0) + N * ((m_j - m_j.mean(axis=0)) ** 2).sum(axis=0)) / (
            ng * N - 1)
        rel = np.abs(ws["well_sd"][gi] ** 2 - tot) / tot
        assert np.all(rel <= 1e-3), rel
    # a well of one window equals that window (the same numbers from the same draws)
    for key, wk in (("mhat", "well_mean"), ("sigma", "well_sd"), ("q05", "well_q05"),
                    ("q50", "well_q50"), ("q95", "well_q95")):
        assert np.max(np.abs(ws[wk][2] - w[key][5])) <= 1e-12, key
    # windows of one law N(0.5, 0.05^2): a well of 8 windows and one of 2 have
    # the pooled SD of the law within 2 % (relative SE of an SD over 32,000
    # and 8,000 draws: 0.4 % and 0.8 %); it does not shrink like s / sqrt(n_g)
    s0 = 0.05
    one = (0.5 + s0 * rng.standard_normal((10, N, 1))).astype(np.float32)
    ws1 = RM.well_summary(one, np.array([1] * 8 + [2] * 2))
    rel1 = np.abs(ws1["well_sd"][:, 0] - s0) / s0
    assert np.all(rel1 <= 0.02), rel1
    print("    R36b: worst quantile error %.2f SE; pooled SD of 8 and 2 windows %.5f, %.5f "
          "(s = 0.05)" % (worst, ws1["well_sd"][0, 0], ws1["well_sd"][1, 0]), flush=True)


# ------------------------------------------------------------------ R36c ---

def _toy_sim(rng, M=2, n_post=20, p=3, d_z=4, alpha=0.5):
    """A <stem>_sim.npz of block 35's contract, by hand: 30 networks of 3
    rows, k_draw 2 (so the draw rows are neither every row nor the
    calibration rows), the gate's roles of ev_gate.assign_roles."""
    np = _np()
    from scipy.special import logsumexp
    import ev_gate
    import ev_rec_data
    donor = np.tile(np.arange(30) * 3 + 5, 3)                 # global order: round robin
    n = donor.size
    theta = rng.uniform(size=(n, p))
    theta[0, 0], theta[31, 1], theta[2, 2] = 0.001, 0.9995, 0.003   # edge rows
    calib = ev_rec_data.calibration_rows(donor)
    draw = ev_rec_data.draw_rows(donor, 2)
    role = ev_gate.assign_roles(donor, (0.4, 0.25, 0.25, 0.1), 800, 1000, 1000, 1, 17)
    didx = np.flatnonzero(draw)
    zidx = np.flatnonzero(draw | (role != 0))
    lqm = rng.normal(size=(n, M))
    z = {"rows": np.arange(n, dtype=np.int64) * 2, "donor": donor.astype(np.int64),
         "well": donor.astype(np.int64), "cls": (donor % 2).astype(np.int64),
         "theta_ref": theta, "in_box": np.ones(n, dtype=bool), "logq_ref_members": lqm,
         "logq_ref": logsumexp(lqm, axis=1) - math.log(M), "calib": calib, "draw": draw,
         "gate_role": role, "draw_idx": didx,
         "draws": rng.uniform(size=(didx.size, n_post, p)).astype(np.float32),
         "draw_member": np.repeat(np.arange(M), n_post // M).astype(np.int64),
         "logq_draws": rng.normal(size=(didx.size, n_post)).astype(np.float32),
         "logq_draws_own": rng.normal(size=(didx.size, n_post)).astype(np.float32),
         "stats": rng.normal(size=(didx.size, 8)).astype(np.float32), "z_idx": zidx,
         "z": rng.normal(size=(M, zidx.size, d_z)).astype(np.float32)}
    return z


def _toy_bank(rng, M=2, n_post=20, p=3, d_z=4):
    """A <stem>_<r>.npz of block 35's contract: 6 wells of 4 windows, wells
    104-106 displaced in the embedding so that some windows are flagged."""
    np = _np()
    well = np.repeat(np.array([101, 102, 103, 104, 105, 106]), 4)
    n = well.size
    z = rng.normal(size=(M, n, d_z))
    z[:, well >= 104, 0] += 3.0
    return {"rows": np.arange(n, dtype=np.int64), "cls": (well >= 104).astype(np.int64),
            "donor": (well // 2).astype(np.int64), "well": well.astype(np.int64),
            "batch": (well % 2).astype(np.int64), "subregion": np.zeros(n, dtype=np.int64),
            "window_idx": np.tile(np.arange(4), 6).astype(np.int64),
            "rep_group": np.full(n, -1, dtype=np.int64), "contaminated": np.zeros(n, dtype=bool),
            "draws": rng.uniform(size=(n, n_post, p)).astype(np.float32),
            "draw_member": np.repeat(np.arange(M), n_post // M).astype(np.int64),
            "logq_draws": rng.normal(size=(n, n_post)).astype(np.float32),
            "logq_draws_own": rng.normal(size=(n, n_post)).astype(np.float32),
            "z": z.astype(np.float32), "stats": rng.normal(size=(n, 8)).astype(np.float32)}


def test_r36c_gate():
    """the gate's wiring, the shares, k_gate and the size band, the example well (specs/SPEC.md block 36)."""
    np = _np()
    from scipy.stats import beta, binomtest
    import ev_gate
    import ev_metrics as EM
    import ev_plan
    import ev_rec_metrics as RM
    import ev_rec_plan
    rng = np.random.default_rng(SEED + 2)
    ecfg = _toy_ecfg(alpha=0.5)
    e = ev_rec_plan.RecEnsemble(name="toy:A1", stem="toy_A1", search="toy", campaign="S-A1",
                                arm="A1", folder="/nonexistent", seeds=(1, 2), records=(),
                                banks=("real",))
    S = lambda *q: ev_plan.seed_of(ecfg.seed, "metrics", e.stem, *q)   # noqa: E731
    z = _toy_sim(rng)
    prov = {"axes": {"names": ["a", "b", "c"], "bounds": None}}
    js, arr, gref = RM._sim_metrics(ecfg, e, prov, z, S)
    # the gate on the embeddings of the roles, by hand: every row's embedding
    # put back at its row, the roles' rows in row order
    n = z["theta_ref"].shape[0]
    full = np.full((2, n, 4), np.nan, dtype=np.float32)
    full[:, z["z_idx"]] = z["z"]
    role = z["gate_role"]
    R = dict((k, np.flatnonzero(role == k)) for k in (1, 2, 3, 4))
    want = ev_gate.gate(full[:, R[1]], full[:, R[2]], full[:, R[3]], full[:, R[4]],
                        list(ecfg.gate["factors"]))
    # the same function on the same arrays: equal (T and p are formed from
    # integer counts, so exactly; the continuous ones to float64 equality too)
    assert np.array_equal(arr["gate_T_check"], want["T"]), "T on the check rows"
    assert np.array_equal(arr["gate_p_check"], want["p"]), "p on the check rows"
    assert np.array_equal(arr["gate_T_null"], want["T_null"]), "T on the null rows"
    assert np.array_equal(arr["gate_check_rows"], R[4])
    g = js["gate"]
    assert (g["n_ref"], g["n_rank"], g["n_null"], g["n_check"]) == tuple(R[k].size for k in
                                                                         (1, 2, 3, 4)), g
    _close(g["bandwidths"], want["bandwidths"], 0.0, "bandwidths")
    flag = want["p"] < 0.5
    ci = binomtest(int(flag.sum()), int(flag.size)).proportion_ci(confidence_level=0.95,
                                                                  method="exact")
    assert g["check"]["n"] == flag.size and g["check"]["n_flagged"] == int(flag.sum())
    assert g["check"]["ci"] == [float(ci.low), float(ci.high)], (g["check"], ci)
    k = math.ceil(0.5 * (R[3].size + 1) - 1e-9) - 1
    assert g["k_gate"] == k >= 1, (g["k_gate"], k)
    # the edge split is over S_cal, not over the draw rows (30 calibration
    # rows, 60 draw rows here)
    s_cal = z["calib"] & z["in_box"]
    assert js["edge"]["n_cal"] == int(s_cal.sum()) == 30, js["edge"]
    es, ea = RM.edge_split(arr["gamma"][s_cal], RM.edge_rows(z["theta_ref"], 0.006693)[s_cal],
                           0.006693, 3)
    _close(arr["coverage_edge"], ea["coverage_edge"], 0.0, "coverage_edge")
    _close(arr["coverage_int"], ea["coverage_int"], 0.0, "coverage_int")
    assert js["edge"]["n_edge"] == es["n_edge"] >= 1, js["edge"]

    # a recording bank: its gate is ev_gate.gate on its windows against the same sets
    zr = _toy_bank(rng)
    jr, ar = RM._bank_metrics(ecfg, e, prov, "real", zr, gref, S)
    wb = ev_gate.gate(full[:, R[1]], full[:, R[2]], full[:, R[3]], zr["z"],
                      list(ecfg.gate["factors"]))
    for key, wk in (("gate_t", "t"), ("gate_r", "r"), ("gate_p_member", "p_member"),
                    ("gate_T", "T"), ("gate_p", "p")):
        assert np.array_equal(ar[key], wb[wk]), key
    assert np.array_equal(ar["gate_flag"], wb["p"] < 0.5)
    # flag_shares by hand, per well and per class
    fl = wb["p"] < 0.5
    for entry in jr["gate"]["per_well"]:
        sel = zr["well"] == entry["well"]
        assert entry["n"] == int(sel.sum()) and entry["n_flagged"] == int(fl[sel].sum()), entry
    for entry in jr["gate"]["per_class"]:
        sel = zr["cls"] == entry["cls"]
        assert entry["n"] == int(sel.sum()) and entry["n_flagged"] == int(fl[sel].sum()), entry
    assert jr["gate"]["n_flagged"] == int(fl.sum()) and jr["gate"]["n"] == fl.size
    ids, nn, kk = RM.flag_shares(np.array([1, 0, 1, 1, 0, 0, 1], dtype=bool),
                                 np.array([5, 5, 2, 9, 2, 9, 9]))
    assert ids.tolist() == [2, 5, 9] and nn.tolist() == [2, 2, 3] and kk.tolist() == [1, 1, 2]
    assert ids.dtype == nn.dtype == kk.dtype == np.int64
    assert 0 < int(fl.sum()) < fl.size, "the toy bank should flag some windows, not all"

    # k_gate (R9) by hand: 25, 4 and 0 (a refusal)
    for (a, nnull), kw in (((0.05, 500), 25), ((0.05, 99), 4), ((0.05, 19), 0)):
        gs = RM.gate_size(a, nnull, 0.95)
        assert gs["k_gate"] == kw, (a, nnull, gs)
    assert RM.gate_size(0.05, 19, 0.95)["size_lo"] is None
    ecfg05 = _toy_ecfg(alpha=0.05)                    # N_null = 7: 0.05 x 8 <= 1
    try:
        RM._sim_metrics(ecfg05, e, prov, z, S)
        raise AssertionError("k_gate = 0 was not refused")
    except ev_plan.RefuseError as exc:
        assert "gate.alpha 0.05" in str(exc) and "N_null = 7" in str(exc), exc
    # on random T values (multiples of 1/500, so with ties): p < alpha_gate
    # exactly when T exceeds the k_gate-th largest null value
    for a in (0.05, 0.1, 0.5):
        nnull = 99
        Tn = rng.integers(0, 501, nnull) / 500.0
        T = rng.integers(0, 501, 5000) / 500.0
        pv = (1.0 + (Tn[None, :] >= T[:, None]).sum(axis=1)) / (nnull + 1.0)
        kg = RM.gate_size(a, nnull, 0.95)["k_gate"]
        kth = np.sort(Tn)[::-1][kg - 1]
        assert np.array_equal(pv < a, T > kth), a
    # the size band: the central 95 % interval of Beta(k, N + 1 - k) (scipy's
    # own interval; 1e-12 covers (1 - level) / 2 = 0.025 + 2e-17 in float64)
    for nnull, kw in ((99, 4), (500, 25), (1000, 50)):
        gs = RM.gate_size(0.05, nnull, 0.95)
        lo, hi = beta(kw, nnull + 1 - kw).interval(0.95)
        assert abs(gs["size_lo"] - lo) <= 1e-12 and abs(gs["size_hi"] - hi) <= 1e-12, (gs, lo, hi)
        assert gs["size_mean"] == kw / (nnull + 1.0) and gs["k_gate"] == kw, gs
    # the example well: 5 wells (the median of the mean T), 4 (the lower middle one)
    T5 = np.array([0.4, 0.6, 0.1, 0.1, 0.8, 1.0, 0.2, 0.4, 0.6, 0.8])  # means .5 .1 .9 .3 .7
    w5 = np.repeat([1, 2, 3, 4, 5], 2)
    ex = RM.example_wells(T5, w5, np.zeros(10, dtype=np.int64))
    assert [(x["well"], x["first_window"]) for x in ex] == [(1, 0)], ex
    T4 = np.array([0.3, 0.5, 0.1, 0.3, 0.7, 0.9, 0.6, 0.6])         # means .4 .2 .8 .6
    w4 = np.repeat([1, 2, 3, 4], 2)
    ex = RM.example_wells(T4, w4, np.zeros(8, dtype=np.int64))
    assert [(x["well"], abs(x["mean_T"] - 0.4) < 1e-12) for x in ex] == [(1, True)], ex
    ex2 = jr["example_wells"]
    assert [x["cls"] for x in ex2] == [0, 1], ex2
    # the bank's files' words (D-160)
    assert jr["wells"][0]["what"] == "the mixture over the well's windows"
    assert "well's posterior" not in json.dumps(jr)
    # the command line: exit 2 on the arguments
    assert RM.main([]) == RM.EXIT_ARGS


# ------------------------------------------------------------------ R36d ---

def _recast(F, ecfg, view):
    """Block 19's samples of the fixture recast in block 35's contract (every
    row a draw row): <stem>_sim.npz, <stem>_<cohort>.npz and
    <stem>_sample.json under ecfg's samples/."""
    np = _np()
    import ev_gate
    import ev_plan
    import ev_rec_data
    import ev_rec_plan
    import ev_rec_sample
    sdir = ecfg.sub("samples")
    os.makedirs(sdir, exist_ok=True)
    old = os.path.join(F["old_out"], "samples")
    index, _paths = ev_rec_data.sim_index(view)
    gk = ecfg.gate
    for e in view.ensembles:
        with np.load(os.path.join(old, "%s_sim.npz" % e.stem)) as z:
            zs = dict((k, z[k]) for k in z.files)
        n, p = zs["theta_ref"].shape
        assert np.array_equal(zs["calib"], ev_rec_data.calibration_rows(zs["donor"]))
        role = ev_gate.assign_roles(zs["donor"], gk["donor_fracs"], gk["n_ref"], gk["n_rank"],
                                    gk["n_null"], gk["rows_per_donor"],
                                    ev_plan.seed_of(ecfg.seed, "gate"))
        out = dict((k, zs[k]) for k in ("rows", "donor", "well", "cls", "theta_ref", "in_box",
                                         "logq_ref_members", "logq_ref", "calib", "draws",
                                         "draw_member", "logq_draws", "logq_draws_own",
                                         "stats", "z"))
        out.update(draw=np.ones(n, dtype=bool), gate_role=role.astype(np.int8),
                   draw_idx=np.arange(n, dtype=np.int64), z_idx=np.arange(n, dtype=np.int64))
        assert sorted(out) == sorted(ev_rec_sample.SIM_ARRAYS), sorted(out)
        np.savez(os.path.join(sdir, "%s_sim.npz" % e.stem), **out)
        files = [os.path.join(sdir, "%s_sim.npz" % e.stem)]
        for r in e.banks:
            rec = ev_rec_data.recording(view, r)
            with np.load(os.path.join(old, "%s_%s.npz" % (e.stem, r))) as z:
                zc = dict((k, z[k]) for k in z.files)
            for k in ("donor", "well", "cls"):
                assert np.array_equal(zc[k], rec[k]), (e.stem, r, k)
            ob = dict((k, rec[k]) for k in ev_rec_data.RECORDING_IDS)
            ob.update(rows=zc["rows"], contaminated=rec["contaminated"])
            ob.update(dict((k, zc[k]) for k in ("draws", "draw_member", "logq_draws",
                                                 "logq_draws_own", "z", "stats")))
            assert sorted(ob) == sorted(ev_rec_sample.REC_ARRAYS), sorted(ob)
            np.savez(os.path.join(sdir, "%s_%s.npz" % (e.stem, r)), **ob)
            files.append(os.path.join(sdir, "%s_%s.npz" % (e.stem, r)))
        names, bounds = ev_rec_data.axes_of(index.sidecar, p)
        with open(os.path.join(old, "%s_sample.json" % e.stem)) as fh:
            prov_old = json.load(fh)
        prov = dict((k, None) for k in ev_rec_sample.PROVENANCE_KEYS)
        prov.update({"schema": ev_rec_sample.SCHEMA, "ensemble": e.name, "stem": e.stem,
                     "search": e.search, "campaign": e.campaign, "arm": e.arm,
                     "folder": e.folder, "view_kind": view.kind,
                     "members": prov_old.get("members"), "banks": list(e.banks),
                     "bank_globs": dict((r, view.banks[r]) for r in e.banks),
                     "sim_bank": view.sim_glob, "sample_params": ecfg.sample_params(),
                     "seeds": prov_old.get("seeds"), "gate": {
                         "n_ref": int((role == 1).sum()), "n_rank": int((role == 2).sum()),
                         "n_null": int((role == 3).sum()), "n_check": int((role == 4).sum()),
                         "networks": None, "seed": ev_plan.seed_of(ecfg.seed, "gate")},
                     "n": int(n), "n_draw": int(n), "n_z": int(n),
                     "M": int(zs["logq_ref_members"].shape[1]), "n_post": int(zs["draws"].shape[1]),
                     "p": int(p), "axes": {"names": names, "bounds": (None if bounds is None
                                                                      else bounds.tolist())},
                     "files": files, "eval_config": ecfg.path,
                     "note": "test_smoke_rec_metrics: block 19's samples recast (every row a "
                             "draw row) until block 35 is merged"})
        with open(ev_rec_plan.sample_json(ecfg, e.stem), "w") as fh:
            json.dump(prov, fh, indent=1, sort_keys=True)


def _as_recorded_samples(F, ecfg_path, ecfg, view):
    """Block 35's samples when its sampler is merged; block 19's recast
    otherwise. Returns which."""
    import ev_rec_sample
    try:
        for e in view.ensembles:
            ev_rec_sample.run(F["cfg"], ecfg_path, e.name, log=lambda t: None)
        return "block 35's sampler"
    except NotImplementedError:
        _recast(F, ecfg, view)
        return "block 19's samples recast"


def test_r36d_as_recorded():
    """D-162's second target: the metrics on D_sim equal block 20's (specs/SPEC.md block 36)."""
    np = _np()
    import ev_rec_config
    import ev_rec_data
    import ev_rec_metrics as RM
    import ev_rec_plan
    import ev_rec_validate as RV
    F = rec_fixture.fixture()
    root = _tmpdir("d")
    out = os.path.join(root, "rec_eval_as_out")
    ecfg_path = _write_ecfg(F["ecfg_as"], out, "rec_eval_as.json")
    ecfg = ev_rec_config.load(ecfg_path)
    view = ev_rec_plan.load_view(F["cfg"], ecfg)
    how = _as_recorded_samples(F, ecfg_path, ecfg, view)
    lines = []
    idx = RM.compute(F["cfg"], ecfg_path, ncpus=1, log=lines.append)
    assert lines and lines[-1].startswith("REC METRICS OK: "), lines
    mold = os.path.join(F["old_out"], "metrics")
    mnew = ecfg.sub("metrics")
    nll_dev, other_dev = 0.0, 0.0
    for e in view.ensembles:
        assert e.name in idx["ensembles"], (e.name, idx["left_out"])
        with open(os.path.join(mold, "%s_sim.json" % e.stem)) as fh:
            jo = json.load(fh)
        with open(os.path.join(mnew, "%s_sim.json" % e.stem)) as fh:
            jn = json.load(fh)
        zo, zn = np.load(os.path.join(mold, "%s_sim.npz" % e.stem)), np.load(
            os.path.join(mnew, "%s_sim.npz" % e.stem))
        assert np.array_equal(zo["rows"], zn["rows"]) and np.array_equal(zo["calib"], zn["calib"])
        assert zn["draw"].all(), "every row a draw row (sim_draw_rows_per_donor null)"
        # the NLL per row within 1e-9 nats (sec. 2.6, "The validation": the
        # same float64 log densities; the tolerance covers a sampler's
        # recomputation in another batch order)
        _close(zn["nll_rows"], zo["nll_rows"], 1e-9, "%s nll_rows" % e.stem)
        nll_dev = max(nll_dev, float(np.nanmax(np.abs(zn["nll_rows"] - zo["nll_rows"]))))
        _close_json(jn["nll"], jo["nll"], 1e-9, "%s nll" % e.stem)
        # the rest within 1e-12: the same functions on the same arrays (the
        # tolerance covers the order of floating-point sums only)
        for k in ("gamma", "coverage", "coverage_lo", "coverage_hi", "coverage_members",
                  "sbc_ranks", "sbc_ecdf_diff", "sbc_hist_counts", "tarp_x_f", "tarp_random_f",
                  "mhat", "sigma", "q05", "q95", "zeta", "contraction", "donor_ids",
                  "donor_nll"):
            _close(zn[k], zo[k], 1e-12, "%s %s" % (e.stem, k))
            other_dev = max(other_dev, float(np.nanmax(np.abs(
                zn[k].astype(np.float64) - zo[k].astype(np.float64)))) if zn[k].size else 0.0)
        _close_json(jn["coverage"], jo["coverage"], 1e-12, "%s coverage" % e.stem)
        _close_json(jn["sbc"], jo["sbc"], 1e-12, "%s sbc" % e.stem)
        _close_json(jn["tarp"], jo["tarp"], 1e-12, "%s tarp" % e.stem)
        ro, rn = dict(jo["recovery"]), dict(jn["recovery"])
        assert rn.pop("axes") == ev_rec_data.axes_of(ev_rec_data.sim_index(view)[0].sidecar,
                                                     len(ro["contraction"]))[0]
        ro.pop("axes")
        _close_json(rn, ro, 1e-12, "%s recovery" % e.stem)
        # the edge split is over S_cal (8 calibration rows; 16 draw rows)
        s_cal = zn["calib"] & zn["in_box"]
        es, ea = RM.edge_split(zn["gamma"][s_cal], zn["edge"][s_cal], ecfg.edge_eps,
                               zn["theta_ref"].shape[1])
        assert jn["edge"]["n_cal"] == int(s_cal.sum()) < int(zn["draw"].sum()), jn["edge"]
        _close(zn["coverage_edge"], ea["coverage_edge"], 0.0, "coverage_edge")
        _close(zn["coverage_int"], ea["coverage_int"], 0.0, "coverage_int")
        assert np.array_equal(zn["edge"], RM.edge_rows(zn["theta_ref"], ecfg.edge_eps))
        # the gate: k_gate 1 (alpha 0.5, two null rows), the size band
        assert jn["gate"]["n_null"] == 2 and jn["gate"]["k_gate"] == 1, jn["gate"]
        for r in e.banks:
            assert os.path.isfile(os.path.join(mnew, "%s_%s.json" % (e.stem, r))), r
    # the paired files of every comparison equal block 27's on D_sim within 1e-12
    made = ev_rec_plan.comparisons(view)[0]
    assert [c["name"] for c in idx["comparisons"]] == [c.name for c in made], idx["comparisons"]
    assert made, "the as-recorded view has a comparison"
    for c in made:
        with open(os.path.join(mold, "paired_%s_sim.json" % c.name)) as fh:
            po = json.load(fh)
        with open(os.path.join(mnew, "paired_%s_sim.json" % c.name)) as fh:
            pn = json.load(fh)
        for k in po:
            if k in ("schema",):
                continue
            _close_json(pn[k], po[k], 1e-12, "paired %s %s" % (c.name, k))
        zo = np.load(os.path.join(mold, "paired_%s_sim.npz" % c.name))
        zn = np.load(os.path.join(mnew, "paired_%s_sim.npz" % c.name))
        for k in zo.files:
            _close(zn[k], zo[k], 1e-12, "paired %s %s" % (c.name, k))
    # the validation tool passes on the two trees
    n_ens, fails = RV.validate(F["old_out"], ecfg.out_root)
    assert n_ens == len(view.ensembles) and not fails, fails
    print("    R36d (%s): %d ensemble(s); largest |nll diff| %.3g nats, largest other |diff| "
          "%.3g; %d comparison(s) equal to block 27's" % (how, len(view.ensembles), nll_dev,
                                                         other_dev, len(made)), flush=True)


# ------------------------------------------------------------------ R36e ---

def _vtree(root, rng, names=("A:A1", "B:A2")):
    np = _np()
    import ev_metrics as EM
    md = os.path.join(root, "metrics")
    os.makedirs(md)
    ens = {}
    for name in names:
        stem = name.replace(":", "_")
        ens[name] = {"stem": stem}
        nll = rng.normal(size=12)
        nll[5] = np.nan
        np.savez(os.path.join(md, "%s_sim.npz" % stem), rows=np.arange(12) * 3, nll_rows=nll,
                 coverage=np.sort(rng.uniform(size=101)), betas=EM.BETAS)
        with open(os.path.join(md, "%s_sim.json" % stem), "w") as fh:
            json.dump({"level": 0.95, "sbc": {"verdict": ["ok", "ok", "TOO NARROW"],
                                              "holm_rejected": [False, False, True]},
                       "tarp": {"x": {"ks_p": 0.3}, "random": {"ks_p": 0.01}}}, fh)
    with open(os.path.join(md, "METRICS.json"), "w") as fh:
        json.dump({"ensembles": ens}, fh)


def _edit_npz(path, key, fn):
    np = _np()
    with np.load(path) as z:
        d = dict((k, z[k]) for k in z.files)
    d[key] = fn(d[key].copy())
    np.savez(path, **d)


def _edit_json(path, fn):
    with open(path) as fh:
        d = json.load(fh)
    fn(d)
    with open(path, "w") as fh:
        json.dump(d, fh)


def test_r36e_validate():
    """ev_rec_validate: identical trees pass, four perturbations fail (specs/SPEC.md block 36)."""
    np = _np()
    import ev_rec_validate as RV
    rng = np.random.default_rng(SEED + 4)
    root = _tmpdir("e")
    old = os.path.join(root, "old")
    _vtree(old, rng)

    def fresh(tag):
        new = os.path.join(root, tag)
        shutil.copytree(old, new)
        return new

    new = fresh("same")
    n, f = RV.validate(old, new)
    assert n == 2 and f == [], f
    assert RV.main(["--old", old, "--new", new]) == 0
    # one NLL moved by 0.003 (> 0.002): fails naming the ensemble and the row
    new = fresh("nll")
    _edit_npz(os.path.join(new, "metrics", "B_A2_sim.npz"), "nll_rows",
              lambda a: a + np.where(np.arange(a.size) == 7, 0.003, 0.0))
    n, f = RV.validate(old, new)
    assert len(f) == 1 and f[0].startswith("B:A2: nll row 7 "), f
    assert RV.main(["--old", old, "--new", new]) == RV.EXIT_FAIL
    # moved by 0.001: within the tolerance
    new = fresh("nll_ok")
    _edit_npz(os.path.join(new, "metrics", "B_A2_sim.npz"), "nll_rows", lambda a: a + 0.001)
    assert RV.validate(old, new)[1] == []
    # the coverage of one ensemble moved by 0.02 at one beta
    new = fresh("cov")
    _edit_npz(os.path.join(new, "metrics", "A_A1_sim.npz"), "coverage",
              lambda a: a + np.where(np.arange(a.size) == 40, 0.02, 0.0))
    n, f = RV.validate(old, new)
    assert len(f) == 1 and f[0].startswith("A:A1: coverage at beta 0.40"), f
    # an SBC verdict flipped; a Holm rejection flipped; a TARP verdict flipped
    new = fresh("sbc")
    _edit_json(os.path.join(new, "metrics", "A_A1_sim.json"),
               lambda d: d["sbc"]["verdict"].__setitem__(0, "TOO BROAD"))
    n, f = RV.validate(old, new)
    assert len(f) == 1 and "sbc verdict axis 0" in f[0] and f[0].startswith("A:A1"), f
    new = fresh("tarp")
    _edit_json(os.path.join(new, "metrics", "B_A2_sim.json"),
               lambda d: d["tarp"]["random"].__setitem__("ks_p", 0.2))
    n, f = RV.validate(old, new)
    assert len(f) == 1 and "tarp random verdict" in f[0], f
    # an ensemble missing from one tree
    new = fresh("missing")
    _edit_json(os.path.join(new, "metrics", "METRICS.json"),
               lambda d: d["ensembles"].pop("B:A2"))
    n, f = RV.validate(old, new)
    assert f == ["B:A2: in the old tree only"], f
    assert RV.main(["--old", old, "--new", new]) == RV.EXIT_FAIL
    # rows that differ
    new = fresh("rows")
    _edit_npz(os.path.join(new, "metrics", "A_A1_sim.npz"), "rows", lambda a: a[::-1])
    n, f = RV.validate(old, new)
    assert len(f) == 1 and "rows of D_sim differ" in f[0], f
    # --record (the contract of wave C): the verdict's file, pass or fail, with the
    # digest of the new tree's METRICS.json computed here with hashlib; the exit
    # codes are those without it
    import hashlib
    for tag, want_rc, want_ok, n_fail in (("same", 0, True, 0), ("nll", RV.EXIT_FAIL, False, 1)):
        new = os.path.join(root, tag)
        rec_path = os.path.join(root, "records", "VALIDATION_%s.json" % tag)
        assert RV.main(["--old", old, "--new", new, "--record", rec_path]) == want_rc
        with open(rec_path) as fh:
            rec = json.load(fh)
        assert sorted(rec) == sorted(RV.RECORD_KEYS), sorted(rec)
        assert rec["schema"] == "rec_eval_validation/1" and rec["ok"] is want_ok, rec
        assert rec["n_ensembles"] == 2 and len(rec["failures"]) == n_fail, rec
        assert rec["old"] == os.path.abspath(old) and rec["new"] == os.path.abspath(new)
        assert rec["nll_tol"] == RV.NLL_TOL and rec["coverage_tol"] == RV.COVERAGE_TOL
        with open(os.path.join(new, "metrics", "METRICS.json"), "rb") as fh:
            assert rec["metrics_index_sha256"] == hashlib.sha256(fh.read()).hexdigest()
        assert not os.path.exists(rec_path + ".tmp")
    if n_fail:
        assert rec["failures"][0].startswith("B:A2: nll row 7 "), rec["failures"]
    # a new tree without its METRICS.json: a failure, the digest null
    new = fresh("noindex")
    os.remove(os.path.join(new, "metrics", "METRICS.json"))
    rec_path = os.path.join(root, "records", "VALIDATION_noindex.json")
    assert RV.main(["--old", old, "--new", new, "--record", rec_path]) == RV.EXIT_FAIL
    with open(rec_path) as fh:
        rec = json.load(fh)
    assert rec["ok"] is False and rec["metrics_index_sha256"] is None, rec
    # the arguments
    assert RV.main(["--old", old]) == RV.EXIT_ARGS
    assert RV.main(["--old", old, "--new", os.path.join(root, "nope")]) == RV.EXIT_ARGS


# ------------------------------------------------------------------ R36f ---

# the unchanged code's values on the input of _axes_input (ev_metrics at the
# contract commit 711b66e, before the change): sbc_block(th, dr, 0.95, 7) and
# recovery_summary(th, rec, rows)
_SBC_KS = [0.09013621042096098, 0.045292021495489054, 0.08390471007931555, 0.02712674454908881,
           0.004709554611804778, 0.07477677561498985, 0.014090815976682824,
           0.08032114983480865, 0.02333890681402595, 0.09321377566799516]
_SBC_HOLM = [0.37388387807494927, 0.2717521289729343, 0.37388387807494927, 0.18988721184362167,
             0.04709554611804778, 0.37388387807494927, 0.12681734379014542,
             0.37388387807494927, 0.1867112545122076, 0.37388387807494927]
_SBC_RANK_SUMS = [112, 119, 122, 124, 137, 117, 101, 109, 115, 117]
_SBC_ECDF_SUM = 16.24999996414408
_REC = {
    "contraction": [0.9177110579155661, 0.8868395919024632, 0.9040749119348266,
                    0.8804753889866745, 0.8479006683366588, 0.8413846877223552,
                    0.9140607250811392, 0.9046372207833149, 0.8608349761941452,
                    0.9326803537518724],
    "zeta_mean": [0.10307331670344555, 0.09135581524500311, 0.2604551033194139,
                  -0.14801152411498547, -0.07664417734480244, -0.008863365158382005,
                  0.29884774712775014, 0.09975265154752447, 0.11962478819418204,
                  0.00044782939791054854],
    "zeta_sd": [0.1665468119219768, 0.14881738204813466, 0.29475949050255196,
                0.31018957587006457, 0.1492985399308504, 0.15236293813609428,
                0.34069998062418944, 0.16243590485920237, 0.14707216700574613,
                0.7068546618475863],
    "rmse": [0.014132056802219201, 0.016323042902983913, 0.03036645290207021,
             0.029343346831908663, 0.01350791065758108, 0.015740514111816806,
             0.03044503982959673, 0.01343546110077178, 0.018136480833101258,
             0.03975848784612733],
    "sigma_mean": [0.08163165520043486, 0.09691319150277859, 0.08388420354240665,
                   0.0996259278071854, 0.1105261677953446, 0.1148243035909566,
                   0.08370339227817684, 0.08799984178071465, 0.10681430740581979,
                   0.0739982104774668],
    "interval90_coverage": [1.0] * 10}


def _axes_input(p):
    np = _np()
    rng = np.random.default_rng(SEED)
    th = rng.uniform(size=(12, p))
    dr = np.clip(th[:, None, :] + 0.1 * rng.standard_normal((12, 20, p)), 0, 1)
    return th, dr


def test_r36f_axes():
    """ev_metrics' optional axes: unchanged without, named with (specs/SPEC.md block 36)."""
    np = _np()
    import ev_metrics as EM
    th, dr = _axes_input(10)
    s, a = EM.sbc_block(th, dr, 0.95, 7)
    # the same code path on the same input: equal to the last bit (0.0)
    assert s["ks_p"] == _SBC_KS and s["holm_adjusted_p"] == _SBC_HOLM, s
    assert a["sbc_ranks"].sum(axis=0).tolist() == _SBC_RANK_SUMS and s["n_rejected"] == 1
    assert float(a["sbc_ecdf_diff"].astype(np.float64).sum()) == _SBC_ECDF_SUM
    rec = EM.recovery_arrays(th, dr)
    rows = np.arange(12) % 3 == 0
    r = EM.recovery_summary(th, rec, rows)
    assert r["axes"] == list(EM.AXES) and r["n_rows"] == 4
    for k, v in _REC.items():
        assert r[k] == v, (k, r[k], v)
    # 14 names: 14 named axes, the numbers those of the default path
    names = ["axis_%02d" % k for k in range(14)]
    th14, dr14 = _axes_input(14)
    s14, a14 = EM.sbc_block(th14, dr14, 0.95, 7, axes=names)
    assert len(s14["ks_p"]) == 14 and a14["sbc_ranks"].shape == (12, 14), s14
    r14 = EM.recovery_summary(th14, EM.recovery_arrays(th14, dr14), rows, axes=names)
    assert r14["axes"] == names and len(r14["contraction"]) == 14, r14["axes"]
    s10n, _a = EM.sbc_block(th, dr, 0.95, 7, axes=["n%d" % k for k in range(10)])
    assert s10n["ks_p"] == _SBC_KS
    # the synthetic suite's E20a-c still pass (block 20's own checks, in a
    # process of their own; EVAL_FIXTURE, when set, reuses their fixture)
    r = subprocess.run([sys.executable, os.path.join(_HERE, "test_smoke_evaluation.py"),
                        "test_e20a_quantities", "test_e20b_lc2st", "test_e20c_metrics"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=3600)
    table = [l for l in r.stdout.splitlines() if l.startswith("| test_e20")]
    print("    R36f: E20a-c: %s" % "; ".join(table), flush=True)
    assert r.returncode == 0 and len(table) == 3 and all("| PASS |" in l for l in table), (
        r.stdout[-4000:])


# ------------------------------------------------------------------ R36g ---

def _snapshot(mdir):
    """Every metrics file's content: JSON without its timings, NPZ arrays."""
    np = _np()
    out = {}
    for p in sorted(glob.glob(os.path.join(mdir, "*"))):
        b = os.path.basename(p)
        if p.endswith(".json"):
            with open(p) as fh:
                d = json.load(fh)
            d.pop("seconds", None)
            out[b] = json.dumps(d, sort_keys=True)
        elif p.endswith(".npz"):
            with np.load(p) as z:
                out[b] = dict((k, z[k]) for k in z.files)
    return out


def _same_snapshot(a, b):
    np = _np()
    assert sorted(a) == sorted(b), (sorted(a), sorted(b))
    for k in a:
        if isinstance(a[k], str):
            assert a[k] == b[k], k
        else:
            assert sorted(a[k]) == sorted(b[k]), k
            for x in a[k]:
                assert np.array_equal(a[k][x], b[k][x], equal_nan=a[k][x].dtype.kind == "f"), (
                    k, x)


def test_r36g_end_to_end():
    """the recorded-kind project through the sampler and the metrics (specs/SPEC.md block 36)."""
    import ev_plan
    import ev_rec_config
    import ev_rec_metrics as RM
    import ev_rec_plan
    import ev_rec_sample
    F = rec_fixture.fixture()
    root = _tmpdir("g")
    out = os.path.join(root, "rec_eval_out")
    ecfg_path = _write_ecfg(F["ecfg_rec"], out, "rec_eval.json")
    ecfg = ev_rec_config.load(ecfg_path)
    view = ev_rec_plan.load_view(F["cfg_rec"], ecfg)
    try:
        for e in view.ensembles:
            ev_rec_sample.run(F["cfg_rec"], ecfg_path, e.name, log=lambda t: None)
    except NotImplementedError:
        raise unittest.SkipTest("block 35's sampler is not merged yet")
    lines = []
    idx = RM.compute(F["cfg_rec"], ecfg_path, ncpus=1, log=lines.append)
    assert lines[-1].startswith("REC METRICS OK: "), lines
    mdir = ecfg.sub("metrics")
    assert os.path.isfile(os.path.join(mdir, "METRICS.json"))
    for f in idx["files"]:
        assert os.path.isfile(f), f
    made = ev_rec_plan.comparisons(view)[0]
    assert [c["name"] for c in idx["comparisons"]] == [c.name for c in made], idx["comparisons"]
    assert sorted(idx["ensembles"]) == sorted(e.name for e in view.ensembles), idx["left_out"]
    # one worker and three give the same files
    one = _snapshot(mdir)
    RM.compute(F["cfg_rec"], ecfg_path, ncpus=3, log=lambda t: None)
    _same_snapshot(one, _snapshot(mdir))
    # a sample of other values is left out and named
    e0 = view.ensembles[0]
    pj = ev_rec_plan.sample_json(ecfg, e0.stem)
    keep = open(pj).read()
    try:
        _edit_json(pj, lambda d: d["sample_params"].__setitem__("seed", ecfg.seed + 1))
        idx2 = RM.compute(F["cfg_rec"], ecfg_path, ncpus=1, log=lambda t: None)
        lo = [x for x in idx2["left_out"] if x["ensemble"] == e0.name]
        assert lo and lo[0]["why"].startswith("sampled with other values"), idx2["left_out"]
        assert e0.name not in idx2["ensembles"]
    finally:
        with open(pj, "w") as fh:
            fh.write(keep)
    # [2026-11-23, after the final check, F5] a sample made from another best fit than the
    # folder's current one is left out and named (the launcher calls it stale, block 38)
    keep = open(pj).read()
    try:
        _edit_json(pj, lambda d: d.__setitem__("best_fit_complete_sha256", "0" * 64))
        idx3 = RM.compute(F["cfg_rec"], ecfg_path, ncpus=1, log=lambda t: None)
        lo = [x for x in idx3["left_out"] if x["ensemble"] == e0.name]
        assert lo and lo[0]["why"].startswith("sampled from another best fit"), idx3["left_out"]
        assert e0.name not in idx3["ensembles"]
    finally:
        with open(pj, "w") as fh:
            fh.write(keep)
    # none current: a refusal, exit 3
    other = _write_ecfg(ecfg_path, out, "rec_eval_other.json", seed=ecfg.seed + 1)
    try:
        RM.compute(F["cfg_rec"], other, ncpus=1, log=lambda t: None)
        raise AssertionError("no current sample was not refused")
    except ev_plan.RefuseError as exc:
        assert "no ensemble has a current sample" in str(exc), exc
    assert RM.main(["--config", F["cfg_rec"], "--eval-config", other]) == RM.EXIT_REFUSE


# ------------------------------------------------------------------ R36h ---

def test_r36h_out_of_box():
    """rows of D_sim outside the open box: counted, out of the NLL and the coverage (specs/SPEC.md block 36)."""
    np = _np()
    import ev_rec_config
    import ev_rec_metrics as RM
    import ev_rec_plan
    import ev_rec_report
    import ev_rec_sample
    F = rec_fixture.fixture()
    root = _tmpdir("h")
    name, stem = "S-A1:A1", "S-A1_A1"
    # the unmodified project, then a copy with a draw row and another row on the boundary
    out0 = os.path.join(root, "plain", "out")
    os.makedirs(os.path.dirname(out0))
    ec0 = _write_ecfg(F["ecfg_rec"], out0, "rec_eval.json")
    ev_rec_sample.run(F["cfg_rec"], ec0, name, log=lambda t: None)
    RM.compute(F["cfg_rec"], ec0, names=[name], ncpus=1, log=lambda t: None)
    m0 = os.path.join(out0, "metrics")
    z0 = np.load(os.path.join(m0, "%s_sim.npz" % stem))
    n = z0["rows"].shape[0]
    i_d = int(np.flatnonzero(z0["calib"])[1])
    i_o = int(np.flatnonzero(~z0["draw"])[0])
    box = os.path.join(root, "box")
    os.makedirs(box)
    cfg = rec_fixture.boundary_copy(box, [(z0["rows"][i_d], 3, 1.0), (z0["rows"][i_o], 8, 0.0)])
    out = os.path.join(box, "out")
    ec = _write_ecfg(F["ecfg_rec"], out, "rec_eval.json")
    prov = ev_rec_sample.run(cfg, ec, name, log=lambda t: None)
    assert prov["n_out_of_box"] == 2
    lines = []
    RM.compute(cfg, ec, names=[name], ncpus=1, log=lines.append)
    assert lines[-1].startswith("REC METRICS OK: "), lines
    m = os.path.join(out, "metrics")
    with open(os.path.join(m, "%s_sim.json" % stem)) as fh:
        js = json.load(fh)
    with open(os.path.join(m0, "%s_sim.json" % stem)) as fh:
        js0 = json.load(fh)
    z = np.load(os.path.join(m, "%s_sim.npz" % stem))
    out_rows = np.zeros(n, dtype=bool)
    out_rows[[i_d, i_o]] = True
    assert np.array_equal(~z["in_box"], out_rows)
    # the NLL: the two rows counted and left out, the others the unmodified run's (1e-4
    # nats: float32 roundoff of batches of another composition, as in R35b)
    assert js["nll"]["n_out_of_box"] == 2 and js["nll"]["n_rows"] == n - 2, js["nll"]
    assert js0["nll"]["n_out_of_box"] == 0 and js0["nll"]["n_rows"] == n, js0["nll"]
    assert js["n_in_box"] == n - 2
    assert np.all(np.isnan(z["nll_rows"][out_rows])) or not np.any(np.isfinite(
        z["nll_rows"][out_rows])), z["nll_rows"][out_rows]
    _close(z["nll_rows"][~out_rows], z0["nll_rows"][~out_rows], 1e-4, "nll of the other rows")
    want_mean = float(np.mean(z["nll_rows"][~out_rows]))
    assert abs(js["nll"]["mean"] - want_mean) <= 1e-12, (js["nll"]["mean"], want_mean)
    # S_cal loses the calibration row outside the box; gamma is NaN there
    assert js["n_calib"] == js0["n_calib"] and js["n_cal"] == js0["n_cal"] - 1, (js["n_cal"],
                                                                                 js0["n_cal"])
    assert js["coverage"]["n_rows"] == js["n_cal"] == js["edge"]["n_cal"]
    assert np.isnan(z["gamma"][i_d]) and np.isnan(z["gamma"][i_o])
    in_cal = z["calib"] & z["in_box"]
    assert np.all(np.isfinite(z["gamma"][in_cal])) and int(in_cal.sum()) == js["n_cal"]
    # gamma of the other rows of S_cal: the unmodified run's (the same draws; the
    # densities at theta_i within the roundoff above cannot move a rank here)
    assert np.array_equal(z["gamma"][in_cal], z0["gamma"][in_cal])
    want_curve = np.mean(z["gamma"][in_cal][:, None] <= z["betas"][None, :], axis=0)
    _close(z["coverage"], want_curve, 1e-12, "C_E over S_cal without the row outside the box")
    # [2026-11-24, D-168] SBC and TARP are over S_cal: the row at theta_k = 1 would have
    # rank n_post on that axis whatever the calibration; the ranks kept are the
    # unmodified run's on the same rows
    assert js["sbc"]["n_rows"] == js["n_cal"] and z["sbc_ranks"].shape[0] == js["n_cal"]
    assert z["tarp_x_f"].shape[0] == js["n_cal"] and z["tarp_random_f"].shape[0] == js["n_cal"]
    k_cal = int(np.flatnonzero(np.flatnonzero(z0["calib"]) == i_d)[0])
    assert np.array_equal(z["sbc_ranks"], np.delete(z0["sbc_ranks"], k_cal, axis=0))
    # a sample one of whose files is missing counts as absent: left out and named
    gone = os.path.join(out, "samples", "%s_real.npz" % stem)
    os.rename(gone, gone + ".away")
    try:
        RM.compute(cfg, ec, names=[name], ncpus=1, log=lambda t: None)
        raise AssertionError("a sample with a missing file was reported")
    except RM.RefuseError as exc:
        assert "no ensemble has a current sample" in str(exc), exc
    finally:
        os.rename(gone + ".away", gone)
    # the gate does not use theta: its statistics are the unmodified run's
    for k in ("gate_role", "gate_T_null", "gate_T_check", "gate_p_check"):
        assert np.array_equal(z[k], z0[k]), k
    # the report job on it: no error, the count in the report's text
    rec = ev_rec_report.run(cfg, ec, ncpus=1, log=lambda t: None)
    assert rec["ensembles"] == [name] and rec["n_figures"] > 0, rec
    with open(rec["report"]) as fh:
        text = fh.read()
    assert "%d rows, %d in the box" % (n, n - 2) in text, text[:3000]
    print("    R36h: 2 of %d rows outside the open box: n_out_of_box 2, the NLL over %d rows, "
          "n_cal %d of %d calibration rows, gamma NaN there, SBC and TARP on %d rows; %d figure(s) and "
          "the report written" % (n, n - 2, js["n_cal"], js["n_calib"], js["sbc"]["n_rows"],
                                  rec["n_figures"]))


CHECKS = [("test_r36a_edge", test_r36a_edge),
          ("test_r36b_recording", test_r36b_recording),
          ("test_r36c_gate", test_r36c_gate),
          ("test_r36d_as_recorded", test_r36d_as_recorded),
          ("test_r36e_validate", test_r36e_validate),
          ("test_r36f_axes", test_r36f_axes),
          ("test_r36g_end_to_end", test_r36g_end_to_end),
          ("test_r36h_out_of_box", test_r36h_out_of_box)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    try:
        for name, fn in CHECKS:
            if only and name not in only:
                continue
            t1 = time.time()
            try:
                fn()
                rows.append((name, "PASS", "", time.time() - t1))
            except NotImplementedError as exc:
                bad += 1
                rows.append((name, "TODO", str(exc), time.time() - t1))
            except unittest.SkipTest as exc:
                rows.append((name, "SKIP", str(exc), time.time() - t1))
            except AssertionError as exc:
                bad += 1
                rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
            except Exception:
                bad += 1
                rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    finally:
        rec_fixture.cleanup()
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
