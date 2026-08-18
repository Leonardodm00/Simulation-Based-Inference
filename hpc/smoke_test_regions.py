#!/usr/bin/env python3
"""
smoke_test_regions.py -- does the calibrated box mean what it claims to?

Run:
    python smoke_test_regions.py
    python smoke_test_regions.py -k R4

Every test scores npe_regions.py against exact analytic posteriors from
gmm_benchmark.py, so any failure is attributable to the calibration or
union logic rather than to a trained estimator's approximation error.

  R0  Plumbing. retained_mass is monotone in a box's size; calibrate_box
      hits its target mass to within Monte Carlo tolerance; union_intervals
      merges and separates correctly on hand-built cases.
  R1  calibrate_box on a single recording's pooled draws achieves the
      requested mass, across several targets and several recordings.
  R2  MONOTONICITY. A higher target mass gives a box that contains the
      lower-mass box, on every axis, for every recording.
  R3  CONSERVATIVE UNION. Pooling W windows gives a box at least as wide as
      any single window's own calibrated box, at the same target mass --
      reported as a rate over seeds, since a single instance could go
      either way by chance.
  R4  THE TEST THAT JUSTIFIES THE CALIBRATION RULE. Across many synthetic
      recordings, the fraction whose TRUE generating theta falls inside its
      own calibrated box is close to the target mass. This is the guarantee
      the module claims; R4 measures it directly rather than asserting it.
  R5  Union across recordings: hull matches manual min/max; a deliberately
      disjoint pair of recordings produces 2 segments on the corrupted axis
      and 1 elsewhere; clipping to the prior box is applied and flagged.
  R6  Shrinkage is in [0, 1], is 1.0 for an axis truncated not at all
      (recovers the full prior box by construction), and near 0 for an axis
      truncated hard.
  R7  require_preconditions blocks by default and passes through when told
      to.
  R8  build_and_write_all end to end: files exist, JSON round-trips through
      npe_contract.Contract.from_dict without modification, CSV has one row
      per (axis, target_mass).

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from typing import Callable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_benchmark import GMMBenchmark  # noqa: E402
from npe_contract import Contract  # noqa: E402
from npe_regions import (  # noqa: E402
    AxisUnion, build_and_write_all, build_region_sets, calibrate_box,
    extract_recording_region, require_preconditions, retained_mass,
    union_across_recordings, union_intervals,
)

RESULTS: List[Tuple[str, str, str]] = []
N_DIM, N_OBS, N_COMP = 6, 3, 3


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run(test_id: str, fn: Callable[[], str], selector: str = "") -> None:
    if selector and selector not in test_id:
        return
    try:
        detail = fn() or ""
        RESULTS.append((test_id, "PASS", detail))
        print("  PASS  %-26s %s" % (test_id, detail), flush=True)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        RESULTS.append((test_id, "FAIL", "%s | %s" % (exc, tb)))
        print("  FAIL  %-26s %s" % (test_id, exc), flush=True)


# ---------------------------------------------------------------------------
# Synthetic recordings with a known answer
# ---------------------------------------------------------------------------

def prior_mean_sd(bench: GMMBenchmark) -> Tuple[np.ndarray, np.ndarray]:
    """Mixture mean and per-axis marginal std of the PRIOR itself.

    GMMBenchmark stores its prior as a mixture (weights, means, covs) but
    exposes no ready-made mean()/cov(); GMMPosterior has that pair of
    methods but the prior is not a GMMPosterior. Law of total variance,
    replicated here rather than imported, since it is three lines.
    """
    mu = np.sum(bench.weights[:, None] * bench.means, axis=0)
    within = np.sum(bench.weights[:, None, None] * bench.covs, axis=0)
    dm = bench.means - mu[None, :]
    between = np.einsum("k,ki,kj->ij", bench.weights, dm, dm)
    cov = within + between
    return mu, np.sqrt(np.diag(cov))


def make_bench() -> GMMBenchmark:
    return GMMBenchmark(n_dim=N_DIM, n_obs=N_OBS, n_components=N_COMP,
                        separation=6.0, prior_scale=1.0, obs_noise=0.4,
                        seed=0)


def make_recording(bench: GMMBenchmark, theta_r: np.ndarray, n_windows: int,
                   n_draws: int, rng) -> List[np.ndarray]:
    """W independent noisy windows of ONE recording's true theta.

    simulate() draws independent noise per row (confirmed from source), so
    tiling theta_r n_windows times gives n_windows genuinely independent
    observations of the same underlying parameter -- exactly a recording.
    """
    Z = bench.simulate(np.tile(theta_r[None, :], (n_windows, 1)), rng)
    return [bench.posterior(Z[w]).sample(n_draws, rng) for w in range(n_windows)]


def make_cohort(bench: GMMBenchmark, n_recordings: int, n_windows: int,
                n_draws: int, seed: int):
    """n_recordings recordings, each with its own true theta and W windows.

    Returns (per_window_draws dict, theta_by_recording dict).
    """
    rng = np.random.default_rng(seed)
    thetas = bench.prior_sample(n_recordings, rng)
    per_window, truth = {}, {}
    for r in range(n_recordings):
        name = "rec_%03d" % r
        per_window[name] = make_recording(bench, thetas[r], n_windows,
                                          n_draws, rng)
        truth[name] = thetas[r]
    return per_window, truth


def make_contract(bench: GMMBenchmark) -> Contract:
    mu, sd = prior_mean_sd(bench)
    lo = mu - 6.0 * sd
    hi = mu + 6.0 * sd
    # Alternate ln/linear so R8 actually exercises both branches of the
    # natural-unit conversion, not just the identity one. The GMM draws
    # themselves are unrestricted Gaussians and know nothing about this
    # label -- coord is external metadata being tested here, not a
    # constraint on the synthetic model.
    coord = ["ln" if j % 2 else "linear" for j in range(N_DIM)]
    return Contract(param_names=["p%d" % j for j in range(N_DIM)],
                    coord=coord,
                    bounds_theta=np.stack([lo, hi], axis=1),
                    embedding_dim=N_OBS)


# ---------------------------------------------------------------------------
# R0 -- plumbing
# ---------------------------------------------------------------------------

def r0_plumbing() -> str:
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(5000, 4))

    lo_wide, hi_wide = np.quantile(draws, [0.001, 0.999], axis=0)
    lo_tight, hi_tight = np.quantile(draws, [0.3, 0.7], axis=0)
    check(retained_mass(draws, lo_wide, hi_wide)
          >= retained_mass(draws, lo_tight, hi_tight),
          "retained_mass is not monotone in box size")

    for target in [0.5, 0.9, 0.99]:
        lo, hi, alpha, achieved = calibrate_box(draws, target)
        check(abs(achieved - target) < 0.01,
              "calibrate_box target=%.2f achieved=%.4f, off by more than "
              "Monte Carlo tolerance" % (target, achieved))
        check(0.0 < alpha < 1.0, "alpha out of range: %.4f" % alpha)

    merged = union_intervals([(0.0, 1.0), (0.5, 1.5), (3.0, 4.0)])
    check(merged == [(0.0, 1.5), (3.0, 4.0)],
          "union_intervals merged/separated incorrectly: %s" % merged)
    touch = union_intervals([(0.0, 1.0), (1.0, 2.0)])
    check(touch == [(0.0, 2.0)],
          "touching intervals were not merged: %s" % touch)

    return "monotone box mass OK; calibration hit target within 0.01; merge OK"


# ---------------------------------------------------------------------------
# R1, R2 -- calibration and monotonicity
# ---------------------------------------------------------------------------

def r1_calibration_hits_target() -> str:
    bench = make_bench()
    per_window, _ = make_cohort(bench, n_recordings=8, n_windows=6,
                                n_draws=3000, seed=0)
    targets = [0.95, 0.99, 0.999]
    worst = 0.0
    for windows in per_window.values():
        regions = extract_recording_region(windows, targets)
        for t, box in regions.items():
            worst = max(worst, abs(box.achieved_mass - t))
    check(worst < 0.01,
          "worst |achieved - target| across 8 recordings x 3 targets is "
          "%.4f, expected < 0.01" % worst)
    return "worst |achieved-target| = %.4f over 8 recordings x 3 targets" % worst


def r2_monotone_in_target_mass() -> str:
    bench = make_bench()
    per_window, _ = make_cohort(bench, n_recordings=6, n_windows=6,
                                n_draws=3000, seed=1)
    targets = [0.95, 0.99, 0.999]
    violations = 0
    for windows in per_window.values():
        regions = extract_recording_region(windows, targets)
        for j in range(N_DIM):
            lo95, hi95 = regions[0.95].lo[j], regions[0.95].hi[j]
            lo99, hi99 = regions[0.99].lo[j], regions[0.99].hi[j]
            lo999, hi999 = regions[0.999].lo[j], regions[0.999].hi[j]
            if not (lo999 <= lo99 <= lo95 and hi95 <= hi99 <= hi999):
                violations += 1
    check(violations == 0,
          "%d (recording, axis) pairs were not monotone in target mass"
          % violations)
    return "0 monotonicity violations over 6 recordings x %d axes" % N_DIM


# ---------------------------------------------------------------------------
# R3 -- union across windows is conservative
# ---------------------------------------------------------------------------

def r3_union_is_conservative() -> str:
    """At n_draws=2000 this comparison is dominated by finite-sample
    quantile-estimation noise -- a single window's 99% box, estimated from
    2000 draws, is noisy enough to occasionally beat even a correctly wider
    pooled estimate by chance. n_draws=5000 was the smallest budget at which
    the comparison stopped being noise-dominated in a preliminary sweep (10/10
    seeds at 5000 vs a mix of wins/losses at 2000); kept here rather than
    raised further because the point is the qualitative direction, not a
    tight bound.
    """
    bench = make_bench()
    n_seeds = 10
    wins = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        theta_r = bench.prior_sample(1, rng)[0]
        windows = make_recording(bench, theta_r, n_windows=6, n_draws=5000,
                                 rng=rng)
        pooled_box = extract_recording_region(windows, [0.99])[0.99]
        widest_single = None
        for w in windows:
            lo, hi, _, _ = calibrate_box(w, 0.99)
            width = np.sum(hi - lo)
            if widest_single is None or width > widest_single:
                widest_single = width
        pooled_width = np.sum(pooled_box.hi - pooled_box.lo)
        if pooled_width >= widest_single - 1e-9:
            wins += 1
    check(wins >= n_seeds - 1,
          "the pooled (union) box was narrower than the widest single "
          "window's own box on %d/%d seeds -- pooling should widen, not "
          "narrow" % (n_seeds - wins, n_seeds))
    return "pooled box >= widest single-window box on %d/%d seeds" % (
        wins, n_seeds)


# ---------------------------------------------------------------------------
# R4 -- the test that justifies the calibration rule
# ---------------------------------------------------------------------------

def r4_coverage_matches_target() -> str:
    """Does the emitted box actually contain the true theta at the stated
    rate, across many independent recordings?

    This is the module's entire reason for existing: "this box contains
    target% of posterior mass" should translate into "the true parameter
    falls inside it about target% of the time" when averaged over many
    recordings drawn from the prior. Reported as a rate with a binomial
    tolerance band, not a single-instance assertion.
    """
    bench = make_bench()
    target = 0.95
    n_recordings = 200
    per_window, truth = make_cohort(bench, n_recordings=n_recordings,
                                    n_windows=6, n_draws=2000, seed=2)
    hits = 0
    for name, windows in per_window.items():
        box = extract_recording_region(windows, [target])[target]
        theta_r = truth[name]
        if np.all((theta_r >= box.lo) & (theta_r <= box.hi)):
            hits += 1
    rate = hits / float(n_recordings)
    # Binomial standard error at n=200, p=0.95 is ~1.5 percentage points;
    # allow a wide band since this checks a real statistical property, not
    # arithmetic.
    check(0.85 <= rate <= 1.0,
          "coverage rate %.3f over %d recordings at target 0.95 is outside "
          "[0.85, 1.0]" % (rate, n_recordings))
    return "coverage rate %.3f over %d recordings at target 0.95" % (
        rate, n_recordings)


# ---------------------------------------------------------------------------
# R5 -- union across recordings: hull, segments, clipping
# ---------------------------------------------------------------------------

def r5_union_across_recordings() -> str:
    bench = make_bench()
    per_window, _ = make_cohort(bench, n_recordings=10, n_windows=6,
                                n_draws=2000, seed=3)
    target = 0.99
    per_rec = {name: extract_recording_region(w, [target])[target]
              for name, w in per_window.items()}
    names = ["p%d" % j for j in range(N_DIM)]
    mu, sd = prior_mean_sd(bench)
    prior_lo = mu - 6.0 * sd
    prior_hi = mu + 6.0 * sd
    axes = union_across_recordings(per_rec, names, prior_lo, prior_hi)

    # hull matches manual min/max
    los = np.stack([b.lo for b in per_rec.values()])
    his = np.stack([b.hi for b in per_rec.values()])
    for j, a in enumerate(axes):
        manual_lo = max(float(np.min(los[:, j])), float(prior_lo[j]))
        manual_hi = min(float(np.max(his[:, j])), float(prior_hi[j]))
        check(abs(a.hull_lo - manual_lo) < 1e-9
              and abs(a.hull_hi - manual_hi) < 1e-9,
              "axis %d hull (%.4f, %.4f) does not match manual min/max "
              "(%.4f, %.4f)" % (j, a.hull_lo, a.hull_hi, manual_lo, manual_hi))

    # deliberately disjoint case on axis 0
    fake = {
        "rec_A": type(per_rec["rec_000"])(
            lo=np.array([0.0] + [per_rec["rec_000"].lo[j] for j in range(1, N_DIM)]),
            hi=np.array([1.0] + [per_rec["rec_000"].hi[j] for j in range(1, N_DIM)]),
            alpha=0.01, achieved_mass=0.99, target_mass=0.99, n_draws=100,
            n_windows=6),
        "rec_B": type(per_rec["rec_000"])(
            lo=np.array([10.0] + [per_rec["rec_000"].lo[j] for j in range(1, N_DIM)]),
            hi=np.array([11.0] + [per_rec["rec_000"].hi[j] for j in range(1, N_DIM)]),
            alpha=0.01, achieved_mass=0.99, target_mass=0.99, n_draws=100,
            n_windows=6),
    }
    wide_lo = np.minimum(prior_lo, np.array([-1.0] + [prior_lo[j] for j in range(1, N_DIM)]))
    wide_hi = np.maximum(prior_hi, np.array([12.0] + [prior_hi[j] for j in range(1, N_DIM)]))
    axes_fake = union_across_recordings(fake, names, wide_lo, wide_hi)
    check(axes_fake[0].n_segments == 2,
          "deliberately disjoint recordings produced %d segments on axis 0, "
          "expected 2" % axes_fake[0].n_segments)
    for j in range(1, N_DIM):
        check(axes_fake[j].n_segments == 1,
              "axis %d should be a single segment (identical boxes), got %d"
              % (j, axes_fake[j].n_segments))

    # clipping
    narrow_prior_lo = prior_lo.copy()
    narrow_prior_lo[0] = 5.0    # cuts into rec_000's own box
    axes_clip = union_across_recordings(per_rec, names, narrow_prior_lo,
                                        prior_hi)
    check(axes_clip[0].hull_lo >= 5.0 - 1e-9,
          "clipping to a narrower prior box did not take effect")

    return ("hull matches manual min/max on %d axes; disjoint case gives 2 "
            "segments; clipping enforced" % N_DIM)


# ---------------------------------------------------------------------------
# R6 -- shrinkage
# ---------------------------------------------------------------------------

def r6_shrinkage() -> str:
    prior_lo, prior_hi = 0.0, 10.0
    a_full = AxisUnion(name="x", hull_lo=prior_lo, hull_hi=prior_hi,
                       segments=[(prior_lo, prior_hi)], prior_lo=prior_lo,
                       prior_hi=prior_hi, clipped=False)
    check(abs(a_full.shrinkage - 1.0) < 1e-9,
          "an axis spanning the full prior range has shrinkage %.4f, "
          "expected 1.0" % a_full.shrinkage)

    a_tight = AxisUnion(name="x", hull_lo=4.9, hull_hi=5.1,
                        segments=[(4.9, 5.1)], prior_lo=prior_lo,
                        prior_hi=prior_hi, clipped=False)
    check(a_tight.shrinkage < 0.05,
          "a tightly truncated axis has shrinkage %.4f, expected < 0.05"
          % a_tight.shrinkage)
    check(0.0 <= a_tight.shrinkage <= 1.0 and 0.0 <= a_full.shrinkage <= 1.0,
          "shrinkage outside [0, 1]")
    return "full range -> shrinkage 1.0; tight box -> shrinkage %.3f" % (
        a_tight.shrinkage)


# ---------------------------------------------------------------------------
# R7 -- preconditions
# ---------------------------------------------------------------------------

def r7_preconditions() -> str:
    try:
        require_preconditions({"gate": True, "coverage": False})
        raised = False
    except RuntimeError:
        raised = True
    check(raised, "require_preconditions did not raise when a check failed "
                  "and strict=True (the default)")

    failed = require_preconditions({"gate": True, "coverage": False},
                                   strict=False)
    check(failed == ["coverage"],
          "require_preconditions(strict=False) returned %s, expected "
          "['coverage']" % failed)

    require_preconditions({"gate": True, "coverage": True})   # must not raise
    return "raises by default on a failed check; strict=False returns the list"


# ---------------------------------------------------------------------------
# R8 -- end to end IO
# ---------------------------------------------------------------------------

def r8_end_to_end_io() -> str:
    bench = make_bench()
    per_window, _ = make_cohort(bench, n_recordings=6, n_windows=6,
                                n_draws=1500, seed=4)
    contract = make_contract(bench)
    targets = [0.95, 0.99, 0.999]

    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, "regions")
        out = build_and_write_all(
            per_window, targets, contract.param_names, contract.coord,
            contract.low, contract.high, stem, contract=contract,
            provenance={"note": "smoke test"})

        for t in targets:
            json_path, csv_path = out[t]
            check(os.path.isfile(json_path), "missing %s" % json_path)
            with open(json_path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            reloaded = Contract.from_dict(d)
            check(reloaded.param_names == contract.param_names,
                  "round-tripped Contract has different param_names")
            check(reloaded.embedding_dim == contract.embedding_dim,
                  "round-tripped embedding_dim=%d, expected %d -- provenance "
                  "was not correctly threaded through" % (
                      reloaded.embedding_dim, contract.embedding_dim))
            check(reloaded.bounds_theta.shape == (N_DIM, 2),
                  "round-tripped bounds_theta has wrong shape")
            check(np.all(reloaded.low <= reloaded.high),
                  "round-tripped bounds_theta has an inverted axis")
            check(isinstance(d["region_shrinkage"][contract.param_names[0]],
                             float),
                  "shrinkage did not round-trip as a plain float")
            check(isinstance(d["region_clipped_to_prior"][
                             contract.param_names[0]], bool),
                  "clipped flag did not round-trip as a plain bool (numpy "
                  "bool_ is not JSON-native)")

        csv_path = out[targets[0]][1]
        check(os.path.isfile(csv_path), "missing %s" % csv_path)
        with open(csv_path, "r", encoding="utf-8") as fh:
            rows = fh.read().strip().splitlines()
        # header + one row per (axis, target_mass)
        check(len(rows) - 1 == N_DIM * len(targets),
              "csv has %d data rows, expected %d (%d axes x %d targets)"
              % (len(rows) - 1, N_DIM * len(targets), N_DIM, len(targets)))
        import csv as _csv
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = list(_csv.DictReader(fh))
        check(all("lo_natural" in r and "hi_natural" in r for r in reader),
              "natural-unit columns missing from the csv despite a contract "
              "being supplied")
        for r in reader[:N_DIM]:
            j = contract.param_names.index(r["axis_name"])
            lo_s, hi_s = float(r["lo_stored"]), float(r["hi_stored"])
            expect_lo = np.exp(lo_s) if contract.coord[j] == "ln" else lo_s
            expect_hi = np.exp(hi_s) if contract.coord[j] == "ln" else hi_s
            check(abs(float(r["lo_natural"]) - expect_lo) < 1e-6
                  and abs(float(r["hi_natural"]) - expect_hi) < 1e-6,
                  "natural-unit bounds for axis %s do not match the "
                  "expected per-axis coordinate transform" % r["axis_name"])

    # preconditions block file creation before any of the above runs
    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, "blocked")
        blocked = False
        try:
            build_and_write_all(
                per_window, targets, contract.param_names, contract.coord,
                contract.low, contract.high, stem,
                preconditions={"gate": False})
        except RuntimeError:
            blocked = True
        check(blocked, "build_and_write_all did not refuse on a failed "
                       "precondition")
        check(not os.path.isfile(stem + ".csv"),
              "a file was written despite a failed precondition")

    return ("%d JSON files round-trip through Contract.from_dict; CSV has "
            "%d rows; precondition failure blocks all output"
            % (len(targets), N_DIM * len(targets)))


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", dest="selector", default="")
    args = ap.parse_args()

    print("=" * 74)
    print("Region extraction validated against exact analytic posteriors")
    print("=" * 74)

    print("\n[plumbing]")
    run("R0_plumbing", r0_plumbing, args.selector)

    print("\n[calibration]")
    run("R1_calibration_hits_target", r1_calibration_hits_target,
        args.selector)
    run("R2_monotone_in_target_mass", r2_monotone_in_target_mass,
        args.selector)

    print("\n[windows and coverage]")
    run("R3_union_is_conservative", r3_union_is_conservative, args.selector)
    run("R4_coverage_matches_target", r4_coverage_matches_target,
        args.selector)

    print("\n[across recordings]")
    run("R5_union_across_recordings", r5_union_across_recordings,
        args.selector)
    run("R6_shrinkage", r6_shrinkage, args.selector)

    print("\n[preconditions and io]")
    run("R7_preconditions", r7_preconditions, args.selector)
    run("R8_end_to_end_io", r8_end_to_end_io, args.selector)

    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print("\n" + "=" * 74)
    print("%d passed, %d failed, %d total" % (n_pass, n_fail, len(RESULTS)))
    print("=" * 74)
    if n_fail:
        print("\nFailures:")
        for tid, status, detail in RESULTS:
            if status == "FAIL":
                print("  %s: %s" % (tid, detail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
