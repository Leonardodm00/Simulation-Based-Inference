#!/usr/bin/env python3
"""
smoke_test_bootstrap_paired.py -- does the paired bootstrap measure what it
claims to? (JOINT_DSN_NPE_PLAN v0.6, S2.4a)

Run:
    python smoke_test_bootstrap_paired.py            # B1-B12, ~10 s
    python smoke_test_bootstrap_paired.py --full     # adds B13 coverage, ~1-2 min
    python smoke_test_bootstrap_paired.py -k B3

Every test validates against a truth known ANALYTICALLY from the generative
model, never against a second implementation of the same estimator. The
model is one-way random effects,

    d_gi = mu + a_g + e_gi,   a_g ~ N(0, rho sigma^2),  e_gi ~ N(0, (1-rho) sigma^2),

for which, with G balanced groups of size n (N = G n):

    Var[row mean]            = sigma^2 (1 + (n-1) rho) / N      (= DEFF sigma^2 / N)
    Var[mean of G single rows] = sigma^2 / G
    naive row-bootstrap SE   ~ sigma / sqrt(N)
    DEFF                     = 1 + (n - 1) rho

  B1  ICC recovery: rho_hat within 4 Fisher SDs of the planted rho.
  B2  Bootstrap SE matches the analytic SD, per scheme, averaged over
      replicate datasets so the check is on the estimator not one draw.
  B3  The error law: se(none)/se(all) = 1/sqrt(DEFF_raw) across rho, with
      the UNCLIPPED rho -- at rho = 0 the sample's rho_raw is often < 0 and
      the group bootstrap is then correctly narrower than the row bootstrap.
  B4  `mean` dominates `one`; at rho = 0 the variance ratio is ~n.
  B5  `all` == `mean` exactly under balance (same estimand, same draws);
      they DIVERGE in estimate when group size correlates with d.
  B6  Interval collapse on constant d, with the degenerate ICC flagged.
  B7  Guards: non-finite d, length mismatch, unknown scheme, missing groups,
      a single group, n_boot 0, level outside (0,1) -- every one raises.
  B8  Determinism: same seed, identical output; different seed, same
      estimate, different interval.
  B9  Tail diagnostic: quiet on Gaussian d, loud when one row dominates.
  B10 The access patterns the callers use: res.ci, res["ci"], res.get("ci"),
      getattr(icc, "rho"); and format_comparison names all four schemes.
  B11 Unbalanced groups: n0 < n_bar when sizes vary, equals n when they do
      not, and the ratio estimator of `all` reproduces the row mean.
  B12 The CLI reads two per-row .npz files (Stage 3 layout) and refuses
      unpaired rows.
  B13 (--full) COVERAGE. rho = 0.15, n = 40, G = 30: the nominal 95% `none`
      interval covers the truth far below nominal (~57% expected) while
      `all` and `mean` stay near nominal (~93%). Nothing in a single
      failing run says anything is wrong; that is the point of S2.4a.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from typing import Callable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bootstrap_paired as bp  # noqa: E402
from bootstrap_paired import (  # noqa: E402
    SCHEMES, compare_schemes, format_comparison, intraclass_correlation,
    paired_bootstrap, tail_share,
)

RESULTS: List[Tuple[str, bool, str]] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run(test_id: str, fn: Callable[[], str], selector: str = "") -> None:
    if selector and selector not in test_id:
        return
    try:
        detail = fn() or ""
        RESULTS.append((test_id, True, detail))
        print("  PASS  %-26s %s" % (test_id, detail), flush=True)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        RESULTS.append((test_id, False, "%s | %s" % (exc, tb)))
        print("  FAIL  %-26s %s" % (test_id, exc), flush=True)


# ---------------------------------------------------------------------------
# The generative model
# ---------------------------------------------------------------------------

def random_effects(G, n, rho, mu=0.3, sigma=1.0, seed=0, sizes=None):
    """Balanced (size n) or unbalanced (`sizes`) one-way random-effects data.

    Returns (d, groups, a_g). rho is the intraclass correlation; sigma^2 the
    marginal variance of one row.
    """
    rng = np.random.default_rng(seed)
    if sizes is None:
        sizes = np.full(G, int(n), dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    a = rng.normal(0.0, np.sqrt(rho) * sigma, G)
    parts, groups = [], []
    for g in range(G):
        e = rng.normal(0.0, np.sqrt(1.0 - rho) * sigma, int(sizes[g]))
        parts.append(mu + a[g] + e)
        groups.append(np.full(int(sizes[g]), g, dtype=np.int64))
    return np.concatenate(parts), np.concatenate(groups), a


def fisher_icc_sd(rho, n, G):
    """Large-sample SD of the ANOVA ICC estimator (Fisher), balanced design."""
    return np.sqrt(2.0 * (1 - rho) ** 2 * (1 + (n - 1) * rho) ** 2
                   / (n * (n - 1) * (G - 1)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def b1_icc_recovery():
    G, n = 200, 40
    out = []
    for k, rho in enumerate((0.0, 0.05, 0.3, 0.8)):
        d, g, _ = random_effects(G, n, rho, seed=10 + k)
        icc = intraclass_correlation(d, g)
        tol = 4.0 * fisher_icc_sd(max(rho, 1e-6), n, G) + 0.005
        check(abs(icc["rho"] - rho) < tol,
              "rho %.2f recovered as %.4f (tol %.4f)" % (rho, icc["rho"], tol))
        check(abs(icc["n0"] - n) < 1e-9, "n0 must equal n under balance")
        check(abs(icc["deff"] - (1 + (n - 1) * icc["rho"])) < 1e-9,
              "DEFF must be 1 + (n0-1) rho")
        out.append("%.2f->%.3f" % (rho, icc["rho"]))
    # rho = 1 exactly: no within-group variance at all
    d, g, _ = random_effects(50, 10, 1.0, seed=99)
    icc = intraclass_correlation(d, g)
    check(abs(icc["rho"] - 1.0) < 1e-9 and abs(icc["msw"]) < 1e-20,
          "rho = 1 must be recovered exactly when e = 0")
    return "; ".join(out) + "; 1.00->%.3f" % icc["rho"]


def b2_se_matches_analytic_sd():
    G, n, rho, sigma = 100, 20, 0.2, 1.0
    N = G * n
    analytic = {"all": sigma * np.sqrt((1 + (n - 1) * rho) / N),
                "mean": sigma * np.sqrt((1 + (n - 1) * rho) / N),
                "one": sigma / np.sqrt(G),
                "none": sigma / np.sqrt(N)}      # the naive (wrong) SE
    R = 20
    acc = {s: [] for s in SCHEMES}
    for r in range(R):
        d, g, _ = random_effects(G, n, rho, sigma=sigma, seed=200 + r)
        for s in SCHEMES:
            acc[s].append(paired_bootstrap(d, g, within=s, n_boot=1500,
                                           seed=r)["se"])
    out = []
    for s in SCHEMES:
        m = float(np.mean(acc[s]))
        rel = abs(m / analytic[s] - 1.0)
        check(rel < 0.06, "%s: mean bootstrap SE %.4f vs analytic %.4f "
                          "(rel err %.3f)" % (s, m, analytic[s], rel))
        out.append("%s %.4f/%.4f" % (s, m, analytic[s]))
    return "; ".join(out)


def b3_error_law():
    G, n = 100, 20
    out = []
    for k, rho in enumerate((0.0, 0.05, 0.3, 1.0)):
        d, g, _ = random_effects(G, n, rho, seed=300 + k)
        res = compare_schemes(d, g, n_boot=8000, seed=k)
        meas, pred = res["se_ratio_none_all"], res["predicted_ratio"]
        check(abs(meas - pred) < 0.03,
              "rho %.2f: se(none)/se(all) = %.3f but 1/sqrt(DEFF_raw) = %.3f"
              % (rho, meas, pred))
        # ... and the clipped DEFF must NOT be used: at rho = 0 it would
        # predict exactly 1.0 whenever rho_raw < 0.
        if res["icc"]["rho_raw"] < 0:
            check(pred > 1.0, "rho_raw < 0 must give a predicted ratio > 1")
        out.append("%.2f: %.3f/%.3f" % (rho, meas, pred))
    return "; ".join(out)


def b4_mean_dominates_one():
    G, n = 100, 20
    out = []
    for k, rho in enumerate((0.0, 0.3)):
        d, g, _ = random_effects(G, n, rho, seed=400 + k)
        se_mean = paired_bootstrap(d, g, within="mean", n_boot=4000,
                                   seed=1)["se"]
        se_one = paired_bootstrap(d, g, within="one", n_boot=4000,
                                  seed=2)["se"]
        check(se_one > se_mean, "rho %.2f: se(one) %.4f must exceed "
                                "se(mean) %.4f" % (rho, se_one, se_mean))
        ratio = (se_one / se_mean) ** 2
        expect = n / (1 + (n - 1) * rho)
        check(abs(ratio / expect - 1.0) < 0.25,
              "rho %.2f: Var(one)/Var(mean) = %.2f, expected ~%.2f"
              % (rho, ratio, expect))
        out.append("rho %.1f: var ratio %.1f (expect %.1f)" % (rho, ratio,
                                                                expect))
    return "; ".join(out)


def b5_all_vs_mean():
    # Balanced: identical estimand AND identical draws -> identical numbers.
    d, g, _ = random_effects(60, 15, 0.2, seed=500)
    ra = paired_bootstrap(d, g, within="all", n_boot=1000, seed=7,
                          return_samples=True)
    rm = paired_bootstrap(d, g, within="mean", n_boot=1000, seed=7,
                          return_samples=True)
    check(abs(ra["estimate"] - rm["estimate"]) < 1e-12,
          "balanced: estimates must coincide")
    check(np.max(np.abs(ra["samples"] - rm["samples"])) < 1e-10,
          "balanced, same seed: every resample statistic must coincide")
    # Size-effect confounding: big groups carry a positive shift. Row
    # weighting (all) then sits above group weighting (mean) by construction,
    # and the gap must be large against either interval's width.
    rng = np.random.default_rng(501)
    G = 80
    a = rng.normal(0.0, 0.5, G)
    sizes = 5 + np.round(60.0 / (1.0 + np.exp(-3.0 * a))).astype(np.int64)
    parts, groups = [], []
    for k in range(G):
        parts.append(a[k] + rng.normal(0.0, 0.3, int(sizes[k])))
        groups.append(np.full(int(sizes[k]), k))
    d, g = np.concatenate(parts), np.concatenate(groups)
    ra = paired_bootstrap(d, g, within="all", n_boot=2000, seed=1)
    rm = paired_bootstrap(d, g, within="mean", n_boot=2000, seed=2)
    gap = ra["estimate"] - rm["estimate"]
    check(gap > 4.0 * max(ra["se"], rm["se"]),
          "confounded sizes: all - mean = %.4f, not large against se %.4f"
          % (gap, max(ra["se"], rm["se"])))
    # And the estimates are what the definitions say, exactly.
    check(abs(ra["estimate"] - d.mean()) < 1e-12, "all must estimate the row mean")
    gm = np.array([d[g == k].mean() for k in range(G)])
    check(abs(rm["estimate"] - gm.mean()) < 1e-12,
          "mean must estimate the mean of group means")
    return "balanced identical; confounded gap %.3f (%.1f se)" % (
        gap, gap / max(ra["se"], rm["se"]))


def b6_constant_d():
    d = np.full(300, 0.37)
    g = np.repeat(np.arange(30), 10)
    for s in SCHEMES:
        r = paired_bootstrap(d, g, within=s, n_boot=500, seed=3)
        check(abs(r["estimate"] - 0.37) < 1e-12, "%s estimate" % s)
        check(r["se"] < 1e-12, "%s se must be 0, got %g" % (s, r["se"]))
        check(abs(r["ci"][0] - 0.37) < 1e-12 and abs(r["ci"][1] - 0.37) < 1e-12,
              "%s interval must collapse to the constant" % s)
    icc = intraclass_correlation(d, g)
    check(icc["degenerate"] and icc["rho"] == 0.0 and icc["deff"] == 1.0,
          "constant d: ICC must be flagged degenerate with rho 0, DEFF 1")
    # Every group a single row: MSW undefined, must NOT raise, must flag.
    icc1 = intraclass_correlation(np.arange(10.0), np.arange(10))
    check(icc1["degenerate"] and np.isnan(icc1["rho"]),
          "N == G: rho must be nan and flagged, not an exception")
    return "collapse OK; degenerate ICC flagged both ways"


def b7_guards():
    d, g, _ = random_effects(10, 5, 0.1, seed=700)
    cases = []

    def expect_raise(fn, what):
        try:
            fn()
        except (ValueError, TypeError):
            cases.append(what)
            return
        raise AssertionError("%s did not raise" % what)

    bad = d.copy(); bad[3] = np.nan
    expect_raise(lambda: paired_bootstrap(bad, g), "NaN in d")
    bad = d.copy(); bad[3] = np.inf
    expect_raise(lambda: paired_bootstrap(bad, g), "inf in d")
    expect_raise(lambda: paired_bootstrap(d, g[:-1]), "length mismatch")
    expect_raise(lambda: paired_bootstrap(d, g, within="rows"), "unknown scheme")
    expect_raise(lambda: paired_bootstrap(d, None, within="all"), "missing groups")
    expect_raise(lambda: paired_bootstrap(d, np.zeros_like(g), within="all"),
                 "single group")
    expect_raise(lambda: paired_bootstrap(d, g, n_boot=0), "n_boot 0")
    expect_raise(lambda: paired_bootstrap(d, g, level=1.0), "level 1.0")
    expect_raise(lambda: paired_bootstrap(d, g, level=0.0), "level 0.0")
    expect_raise(lambda: intraclass_correlation(d, np.zeros_like(g)),
                 "ICC single group")
    expect_raise(lambda: tail_share(d, top_frac=0.0), "top_frac 0")
    expect_raise(lambda: paired_bootstrap(np.array([1.0]), None, within="none"),
                 "one row")
    # `none` without groups is legal.
    paired_bootstrap(d, None, within="none", n_boot=50)
    return "%d guards raise; `none` without groups is legal" % len(cases)


def b8_determinism():
    d, g, _ = random_effects(40, 12, 0.2, seed=800)
    for s in SCHEMES:
        r1 = paired_bootstrap(d, g, within=s, n_boot=800, seed=11,
                              return_samples=True)
        r2 = paired_bootstrap(d, g, within=s, n_boot=800, seed=11,
                              return_samples=True)
        check(np.array_equal(r1["samples"], r2["samples"]) and r1["ci"] == r2["ci"],
              "%s: same seed must give identical output" % s)
        r3 = paired_bootstrap(d, g, within=s, n_boot=800, seed=12,
                              return_samples=True)
        check(r3["estimate"] == r1["estimate"],
              "%s: the estimate does not depend on the seed" % s)
        check(not np.array_equal(r1["samples"], r3["samples"]),
              "%s: a different seed must give different resamples" % s)
    return "identical under one seed, estimate seed-free, resamples differ"


def b9_tail():
    rng = np.random.default_rng(900)
    d = rng.normal(0.0, 1.0, 5000)
    t = tail_share(d, top_frac=0.01)
    check(t["n_top"] == 50, "top 1%% of 5000 rows is 50, got %d" % t["n_top"])
    check(t["share_top"] < 0.06, "Gaussian: top 1%% carries %.3f, expected < 0.06"
          % t["share_top"])
    check(t["share_max"] < 0.002, "Gaussian: largest row %.4f" % t["share_max"])
    d2 = d.copy(); d2[17] = 5000.0
    t2 = tail_share(d2, top_frac=0.01)
    check(t2["share_max"] > 0.5, "planted row must dominate: %.3f" % t2["share_max"])
    check(t2["ratio"] > 20.0, "planted row: ratio %.1f" % t2["ratio"])
    t0 = tail_share(np.zeros(100))
    check(t0["share_top"] == t0["uniform_share"] and t0["ratio"] == 1.0,
          "all-zero d: uniform shares, ratio 1")
    return "gaussian top1%% %.3f; planted max %.3f" % (t["share_top"],
                                                       t2["share_max"])


def b10_caller_access_patterns():
    d, g, _ = random_effects(30, 10, 0.2, seed=1000)
    res = paired_bootstrap(d, groups=g, within="all")          # report_joint_arms
    ci = res.get("ci") if isinstance(res, dict) else getattr(res, "ci", None)
    check(ci is not None and len(ci) == 2 and ci[0] <= res["estimate"] <= ci[1],
          "dict-style access to ci failed")
    check(res.ci == res["ci"], "attribute access must equal item access")
    icc = intraclass_correlation(d, g)
    rho = getattr(icc, "rho", icc)                                # report_joint_arms
    check(isinstance(rho, float) and 0.0 <= rho <= 1.0, "getattr(icc, 'rho')")
    try:
        _ = res.no_such_key
        raise AssertionError("missing attribute must raise AttributeError")
    except AttributeError:
        pass
    txt = format_comparison(compare_schemes(d, g, label="B10 label",
                                            n_boot=300))
    for s in SCHEMES:
        check(("| %s" % s) in txt, "format_comparison must list %s" % s)
    check("B10 label" in txt and "PRIMARY" in txt and "DEFF" in txt,
          "format_comparison must carry label, primary marker and DEFF")
    check(all(ord(c) < 128 for c in txt), "format_comparison must be ASCII")
    # groups as strings (the real cohort keys on the specs `culture` field)
    gs = np.array(["c%02d" % k for k in g])
    r_str = paired_bootstrap(d, groups=gs, within="all", n_boot=200, seed=1)
    r_int = paired_bootstrap(d, groups=g, within="all", n_boot=200, seed=1)
    check(r_str["ci"] == r_int["ci"], "string and int labels must give the "
                                      "same resampling")
    return "ci/get/getattr OK; table ASCII; string labels OK"


def b11_unbalanced():
    rng = np.random.default_rng(1100)
    G = 40
    sizes = rng.integers(3, 60, G)
    d, g, _ = random_effects(G, None, 0.2, seed=1101, sizes=sizes)
    icc = intraclass_correlation(d, g)
    check(icc["n0"] < icc["n_bar"], "unequal sizes: n0 %.2f must be < n_bar %.2f"
          % (icc["n0"], icc["n_bar"]))
    N = d.size
    n0_direct = (N - np.sum(sizes ** 2) / N) / (G - 1)
    check(abs(icc["n0"] - n0_direct) < 1e-9, "n0 formula")
    r = paired_bootstrap(d, g, within="all", n_boot=300, seed=1)
    check(abs(r["estimate"] - d.mean()) < 1e-12,
          "`all` point estimate is the row mean = sum d / sum n_g")
    # `one` must draw rows only from the drawn group: plant a group with a
    # huge constant and check every resample statistic is a multiple of the
    # admissible step.
    d2 = d.copy()
    big = np.flatnonzero(g == 0)
    d2[big] = 1e6
    others = np.setdiff1d(np.arange(d2.size), big)
    d2[others] = 0.0
    r1 = paired_bootstrap(d2, g, within="one", n_boot=400, seed=2,
                          return_samples=True)
    k = np.round(r1["samples"] * G / 1e6)
    check(np.max(np.abs(r1["samples"] - k * 1e6 / G)) < 1e-6,
          "`one` must average exactly one row per drawn group")
    check(abs(k.mean() - 1.0) < 0.25, "group 0 is drawn ~once per resample "
                                      "on average, got %.2f" % k.mean())
    return "n0 %.1f < n_bar %.1f; `one` draws one row per group" % (
        icc["n0"], icc["n_bar"])


def b12_cli():
    d_a = np.random.default_rng(1200).normal(1.0, 0.5, 400)
    d_b = d_a - 0.2 + np.random.default_rng(1201).normal(0.0, 0.1, 400)
    g = np.repeat(np.arange(20), 20)
    tmp = tempfile.mkdtemp()
    try:
        pa = os.path.join(tmp, "A_seed0_perrow.npz")
        pb = os.path.join(tmp, "B_seed0_perrow.npz")
        np.savez(pa, nll=d_a, group=g)
        np.savez(pb, nll=d_b, group=g)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = bp.main(["--a", pa, "--b", pb, "--n-boot", "200"])
        txt = buf.getvalue()
        check(code == 0 and "PRIMARY" in txt and "DEFF" in txt,
              "CLI must print the comparison")
        check("0.2" in txt, "the planted D = 0.2 should appear in the table")
        # Unpaired rows must be refused, not averaged.
        pc = os.path.join(tmp, "C_seed0_perrow.npz")
        np.savez(pc, nll=d_b[:-1], group=g[:-1])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bp.main(["--a", pa, "--b", pc])
            raise AssertionError("shape mismatch must be refused")
        except SystemExit:
            pass
        np.savez(pc, nll=d_b, group=g[::-1])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bp.main(["--a", pa, "--b", pc])
            raise AssertionError("differing group labels must be refused")
        except SystemExit:
            pass
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return "CLI runs on Stage 3 layout; refuses unpaired and mislabelled rows"


def b13_coverage():
    G, n, rho, mu = 30, 40, 0.15, 0.3
    R, n_boot = 300, 500
    hits = {s: 0 for s in ("none", "all", "mean")}
    for r in range(R):
        d, g, _ = random_effects(G, n, rho, mu=mu, seed=2000 + r)
        for s in hits:
            lo, hi = paired_bootstrap(d, g, within=s, n_boot=n_boot,
                                      seed=r)["ci"]
            hits[s] += int(lo <= mu <= hi)
    cov = {s: hits[s] / R for s in hits}
    deff = 1 + (n - 1) * rho
    from math import erf, sqrt
    # A nominal 95% interval whose SE is short by 1/sqrt(DEFF) has effective
    # half-width 1.96/sqrt(DEFF) SDs: coverage 2 Phi(1.96/sqrt(DEFF)) - 1.
    z = 1.959964 / sqrt(deff)
    expect_none = erf(z / sqrt(2.0))
    check(cov["none"] < 0.72, "`none` coverage %.3f should be far below 0.95 "
                              "(analytic ~%.3f)" % (cov["none"], expect_none))
    check(abs(cov["none"] - expect_none) < 0.10,
          "`none` coverage %.3f vs analytic %.3f" % (cov["none"], expect_none))
    for s in ("all", "mean"):
        check(cov[s] > 0.88, "%s coverage %.3f should be near nominal"
              % (s, cov[s]))
    return "none %.3f (analytic %.3f), all %.3f, mean %.3f" % (
        cov["none"], expect_none, cov["all"], cov["mean"])


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", dest="selector", default="")
    ap.add_argument("--full", action="store_true",
                    help="also run the coverage test B13 (~1-2 min)")
    args = ap.parse_args()

    print("=" * 74)
    print("bootstrap_paired validated against a random-effects model")
    print("=" * 74)

    print("\n[estimators against analytic truth]")
    run("B1_icc_recovery", b1_icc_recovery, args.selector)
    run("B2_se_matches_analytic", b2_se_matches_analytic_sd, args.selector)
    run("B3_error_law_1_sqrt_deff", b3_error_law, args.selector)
    run("B4_mean_dominates_one", b4_mean_dominates_one, args.selector)
    run("B5_all_vs_mean", b5_all_vs_mean, args.selector)

    print("\n[edge cases and guards]")
    run("B6_constant_d", b6_constant_d, args.selector)
    run("B7_guards", b7_guards, args.selector)
    run("B8_determinism", b8_determinism, args.selector)
    run("B9_tail_diagnostic", b9_tail, args.selector)

    print("\n[interfaces]")
    run("B10_caller_access", b10_caller_access_patterns, args.selector)
    run("B11_unbalanced", b11_unbalanced, args.selector)
    run("B12_cli", b12_cli, args.selector)

    if args.full:
        print("\n[coverage, --full]")
        run("B13_coverage", b13_coverage, args.selector)
    else:
        print("\n  SKIP  B13_coverage             (run with --full)")

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print("\n" + "=" * 74)
    print("%d passed, %d failed, %d total" % (n_pass, n_fail, len(RESULTS)))
    print("=" * 74)
    if n_fail:
        print("\nFailures:")
        for tid, ok, detail in RESULTS:
            if not ok:
                print("  %s: %s" % (tid, detail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
