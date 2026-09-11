"""Smoke test for Stage 3c (plan v0.6).

Run:  python3 smoke_test_stage3c.py

The floors, the aliasing test and the stratification are each checked against
a case whose answer is known by construction, not against a trained model:
these are estimators, and an estimator is tested by feeding it data with a
planted truth.

Pure ASCII, LF only.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "stage1"), os.path.join(_HERE, "..", "stage2"),
           os.path.join(_HERE, "..", "stage3"), _HERE):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aliasing import aliasing_report, principal_angles          # noqa: E402
from floor_core import concentration_on, floor_summary          # noqa: E402
from nuisance_floor import validate_against_injected            # noqa: E402
from realisation_floor import kernel_axis_concentration         # noqa: E402
from stratify import (diagnosis_covariance, patient_covariance,  # noqa: E402
                      replicate_covariance, stratify,
                      validate_on_bench)
from window_aggregation import curve_slope                      # noqa: E402

RESULTS = []


def report(name, status, detail):
    RESULTS.append(status)
    print("[%s] %-50s %s" % (status, name, detail))


def ok(name, cond, detail):
    report(name, "PASS" if cond else "FAIL", detail)


# ---------------------------------------------------------------------------
# C1 -- floor_summary recovers a planted covariance structure
# ---------------------------------------------------------------------------

def test_c1():
    rng = np.random.default_rng(0)
    d = 5
    v = np.zeros(d)
    v[2] = 1.0
    means = np.outer(rng.standard_normal(400), v) + 0.01 * rng.standard_normal((400, d))
    f = floor_summary(means, param_names=["a%d" % k for k in range(d)])
    ok("C1a the leading floor direction is the planted one",
       f["directions"][0]["dominant_axes"][0] == "a2"
       and f["directions"][0]["variance_share"] > 0.95,
       "share %.3f on %s" % (f["directions"][0]["variance_share"],
                             f["directions"][0]["dominant_axes"][0]))

    share = concentration_on([2], f)
    ok("C1b concentration_on measures variance attribution",
       share > 0.95, "%.3f of the trace on axis 2 (uniform would be %.2f)"
       % (share, 1.0 / d))

    ok("C1c one draw is refused rather than returning a zero covariance",
       _raises(lambda: floor_summary(means[:1])), "needs >= 2 draws")


def _raises(fn, exc=ValueError):
    try:
        fn()
        return False
    except exc:
        return True


# ---------------------------------------------------------------------------
# C2 -- the D17 concentration test
# ---------------------------------------------------------------------------

def test_c2():
    rng = np.random.default_rng(1)
    d = 6
    kernel = [3, 4, 5]
    m = rng.standard_normal((300, d)) * 0.01
    m[:, kernel] += rng.standard_normal((300, 3))
    f = floor_summary(m)
    k = kernel_axis_concentration(f, kernel)
    ok("C2a a realisation floor on the kernel axes is called concentrated",
       k["concentrated"] and k["concentration"] > 0.9,
       "%.3f vs uniform %.2f" % (k["concentration"], k["uniform_reference"]))

    m2 = rng.standard_normal((300, d))
    k2 = kernel_axis_concentration(floor_summary(m2), kernel)
    ok("C2b an evenly spread floor is NOT",
       not k2["concentrated"],
       "%.3f, close to the uniform reference %.2f -- masking the kernel axes "
       "would not address the confound"
       % (k2["concentration"], k2["uniform_reference"]))


# ---------------------------------------------------------------------------
# C3 -- aliasing on a constructed linear map
# ---------------------------------------------------------------------------

def test_c3():
    E, d = 8, 4
    rng = np.random.default_rng(2)
    J_theta = rng.standard_normal((E, d))

    # A nuisance direction lying exactly INSIDE the parameter span: a_m = 1
    # and delta_theta must recover the coefficients that built it.
    coef = np.array([0.5, -1.5, 0.0, 2.0])
    g_in = J_theta @ coef
    # A nuisance direction ORTHOGONAL to the span: a_m = 0.
    u, _, _ = np.linalg.svd(J_theta, full_matrices=True)
    g_out = u[:, d:][:, 0]
    J_nu = np.stack([g_in, g_out], axis=1)

    al = aliasing_report(J_theta, J_nu, nu_names=["inside", "outside"])
    a_in = al["directions"][0]["a_m"]
    a_out = al["directions"][1]["a_m"]
    ok("C3a a nuisance inside the parameter span has a_m = 1",
       abs(a_in - 1.0) < 1e-8, "a_m = %.10f" % a_in)
    ok("C3b one orthogonal to it has a_m = 0",
       abs(a_out) < 1e-8, "a_m = %.2e" % a_out)

    rec = np.asarray(al["directions"][0]["delta_theta"])
    ok("C3c delta_theta recovers the coefficients that built it",
       np.allclose(rec, coef, atol=1e-8),
       "max |diff| = %.2e" % np.max(np.abs(rec - coef)))

    ok("C3d the 'absorbed' reading fires only for large a_m",
       "absorbed" in al["directions"][0]["reading"]
       and "transverse" in al["directions"][1]["reading"],
       "inside -> absorbed, outside -> transverse")

    # A rank-deficient J_theta must be truncated, not inverted.
    Jd = J_theta.copy()
    Jd[:, 3] = Jd[:, 0]
    ald = aliasing_report(Jd, J_nu)
    ok("C3e a rank-deficient J_theta is truncated and the rank reported",
       ald["rank_J_theta"] == 3,
       "rank %d of %d (P11: aliasing is worst where the spectrum is smallest)"
       % (ald["rank_J_theta"], ald["n_theta"]))

    ang = principal_angles(J_theta, g_out.reshape(-1, 1))
    ok("C3f principal_angles gives 90 degrees for an orthogonal direction",
       abs(np.degrees(ang[0]) - 90.0) < 1e-6,
       "%.4f degrees" % np.degrees(ang[0]))

    alq = aliasing_report(J_theta, np.zeros((E, 1)), nu_names=["quant"],
                          quantised=[True])
    ok("C3g a quantised component reports a_m = None, not 0",
       alq["directions"][0]["a_m"] is None
       and "QUANTISED" in alq["directions"][0]["note"],
       "a_m = 0 would read as 'perfectly separable', the opposite of the truth")


# ---------------------------------------------------------------------------
# C4 -- stratification: null value, separation, and the gate
# ---------------------------------------------------------------------------

def _make_cohort(d=4, n_donor=30, k=2, spread_axis=1, spread=0.6,
                 noise=0.10, seed=0):
    rng = np.random.default_rng(seed)
    donor_true = np.zeros((n_donor, d))
    if spread_axis is not None:
        donor_true[:, spread_axis] = rng.standard_normal(n_donor) * spread
    means, donors, labels = [], [], []
    for i in range(n_donor):
        for _ in range(k):
            means.append(donor_true[i] + rng.standard_normal(d) * noise)
            donors.append(i)
            labels.append(i % 2)
    return np.asarray(means), np.asarray(donors), np.asarray(labels)


def test_c4():
    m, don, lab = _make_cohort()
    r = stratify(m, don, lab)
    ok("C4a the null value is 1/(2k), not 1",
       abs(r["null_mu"] - 0.25) < 1e-12,
       "null_mu = %.4f at k = 2; comparing against 1 is conservative by 2k"
       % r["null_mu"])

    v = validate_on_bench(r, [1])
    ok("C4b the planted axis is identified with a clear separation",
       v["passed"] and v["separation_ratio"] > 3.0,
       "mu[0]/mu[1] = %.2f, leading %s" % (v["separation_ratio"],
                                           v["leading_direction_axes"]))

    m0, don0, lab0 = _make_cohort(spread_axis=None, seed=5)
    v0 = validate_on_bench(stratify(m0, don0, lab0), [1])
    ok("C4c a cohort with NO planted spread fails the gate",
       not v0["passed"],
       "separation ratio %.2f -- the gate is not a formality"
       % v0["separation_ratio"])

    # Sigma_rep is taken about zero, not about the sample mean.
    S, info = replicate_covariance(m, don)
    ok("C4d Sigma_rep uses all within-donor pairs and counts singletons",
       info["n_pairs"] == 30 and info["n_singleton_donors"] == 0,
       "%d pairs from 30 donors of 2 wells" % info["n_pairs"])

    single = np.arange(len(don))
    ok("C4e a cohort with no same-donor pairs is refused (D12)",
       _raises(lambda: replicate_covariance(m, single)),
       "this is exactly the D12 risk: 35 cultures, one well each")

    S_dx = diagnosis_covariance(m, don, lab)
    ok("C4f Sigma_dx is computed from the diagnosis means",
       S_dx.shape == (m.shape[1], m.shape[1]), "shape %s" % (S_dx.shape,))


# ---------------------------------------------------------------------------
# C5 -- the P10 slope reading
# ---------------------------------------------------------------------------

def test_c5():
    flat = [{"n": n, "gain": 1.0, "ess": 100.0, "trusted": True}
            for n in (1, 2, 4, 8)]
    s = curve_slope(flat)
    ok("C5a a flat curve is called collapse", "FLAT" in s["reading"],
       "slope %.4f" % s["slope"])

    rising = [{"n": n, "gain": 1.0 + 0.5 * np.log(n), "ess": 100.0,
               "trusted": True} for n in (1, 2, 4, 8)]
    s2 = curve_slope(rising)
    ok("C5b a rising curve is called informative",
       "RISING" in s2["reading"] and abs(s2["slope"] - 0.5) < 1e-9,
       "slope %.4f per e-fold" % s2["slope"])

    falling = [{"n": n, "gain": 1.0 - 0.4 * np.log(n), "ess": 100.0,
                "trusted": True} for n in (1, 2, 4, 8)]
    ok("C5c a falling curve names the independence violation",
       "independence" in curve_slope(falling)["reading"],
       "windows of one subregion share a realisation (S2.7)")

    degen = [{"n": 1, "gain": 1.0, "ess": 2.0, "trusted": False},
             {"n": 2, "gain": 9.0, "ess": 1.0, "trusted": False}]
    s4 = curve_slope(degen)
    ok("C5d points whose importance weights degenerated are not fitted",
       np.isnan(s4["slope"]) and "ESS" in s4["reading"],
       "a curve that bends because the estimator broke looks like one that "
       "bends because information saturated")


# ---------------------------------------------------------------------------
# C6 -- the real-data guard
# ---------------------------------------------------------------------------

def test_c6():
    tmp = tempfile.mkdtemp()
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(_HERE, "run_stage3c.py"),
             "--ckpt", "/nonexistent.pt", "--sim-shards", "/nonexistent/*.npz",
             "--out-dir", tmp, "--allow-real"],
            capture_output=True, text=True, cwd=_HERE)
        ok("C6a --allow-real without a validation record is refused",
           r.returncode != 0 and "validation" in (r.stdout + r.stderr),
           "plan Stage 3c: not applied to real data until the bench passes")

        bad = os.path.join(tmp, "v.json")
        with open(bad, "w") as fh:
            json.dump({"stratification": {"passed": False}}, fh)
        r2 = subprocess.run(
            [sys.executable, os.path.join(_HERE, "run_stage3c.py"),
             "--ckpt", "/nonexistent.pt", "--sim-shards", "/nonexistent/*.npz",
             "--out-dir", tmp, "--allow-real", "--validation", bad],
            capture_output=True, text=True, cwd=_HERE)
        ok("C6b and a FAILING validation record is refused too",
           r2.returncode != 0
           and "failed" in (r2.stdout + r2.stderr).lower(),
           "the gate reads the record rather than its existence")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# C7 -- d17_realisation_audit records and never gates (D17, option (c))
# ---------------------------------------------------------------------------

def test_c7():
    from run_stage3c import d17_realisation_audit
    kernel_idx = [0, 1, 2]
    theta = np.tile(np.array([[0.2, 0.5, 0.8, 0.3],
                              [0.4, 0.5, 0.1, 0.9]]), (3, 1))

    # no realisation_id field: the historical-bank case reports, not raises
    r0 = d17_realisation_audit({"theta": theta}, kernel_idx)
    ok("C7a missing realisation_id is reported, not raised",
       r0["available"] is False and "note" in r0 and bool(r0["note"]),
       "available=False with a note")
    ok("C7a and carries no passed key",
       "passed" not in r0,
       "structurally cannot fail the run")

    # 1 realisation per kernel value: the exact scenario D17 is about
    sim1 = {"theta": theta,
            "realisation_id": np.array([7, 9, 7, 9, 7, 9], dtype=np.uint64)}
    r1 = d17_realisation_audit(sim1, kernel_idx)
    ok("C7b worst case 1 realisation per kernel value is counted",
       r1["available"] is True and r1["worst_case_realisations"] == 1
       and r1["n_kernel_values"] == 2,
       "worst=1 over 2 kernel values")
    ok("C7b the D17 scenario itself does not fail",
       "passed" not in r1,
       "count of 1 is recorded, not flagged (option (c))")

    # >= 2 realisations: recording is symmetric, a good count is no pass
    sim2 = {"theta": theta,
            "realisation_id": np.array([1, 1, 2, 2, 3, 3], dtype=np.uint64)}
    r2 = d17_realisation_audit(sim2, kernel_idx)
    ok("C7c worst case >= 2 is counted the same way",
       r2["available"] is True and r2["worst_case_realisations"] == 3,
       "worst=3, same code path")
    ok("C7c and a good count is not a pass either",
       "passed" not in r2,
       "recording is symmetric; passing was never the point")


def main():
    print("=" * 80)
    print("Smoke test: Stage 3c -- floors, aliasing, stratification, P10")
    print("=" * 80)
    test_c1()
    test_c2()
    test_c3()
    test_c4()
    test_c5()
    test_c6()
    test_c7()
    print("-" * 80)
    n_fail = RESULTS.count("FAIL")
    print("%d passed, %d failed, %d skipped"
          % (RESULTS.count("PASS"), n_fail, RESULTS.count("SKIP")))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
