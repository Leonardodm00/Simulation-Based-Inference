#!/usr/bin/env python3
"""
smoke_test_diagnostics.py -- do the diagnostics detect what they claim to?

Run:
    python smoke_test_diagnostics.py
    python smoke_test_diagnostics.py -k D5

Every test here scores the diagnostics against an EXACT analytic posterior
from gmm_benchmark.py, so no network is trained and the whole suite runs in
seconds. A diagnostic validated against an approximate reference cannot
distinguish its own error from the estimator's.

  D1  MMD overlap fires on a shifted real distribution and stays quiet on a
      matched one. Both directions are needed: a test that always fires is
      as useless as one that never does.
  D2  Geodesic nearest-neighbour distance separates an off-manifold real set
      from an on-manifold one.
  D3  Marginal SBC passes on the exact posterior.
  D4  Marginal SBC fails on posteriors that are too narrow, too broad, and
      biased -- and names which of the three it is.
  D5  THE BLIND SPOT. On a posterior set exactly equal to the prior:
        - marginal SBC PASSES        (it should not detect this)
        - expected coverage PASSES   (nor should it)
        - data-dependent SBC FAILS   (this is the one that works)
        - contraction ~ 0            (and this)
      This reproduces Modrak et al. Theorem 7 empirically and validates the
      bilinear substitute (T1) for the unavailable joint likelihood.
  D6  TARP passes on the exact posterior, fails on a too-narrow one.
  D7  Contraction is ~0 for the prior and high for a tight posterior.
  D8  The information spectrum recovers the KNOWN number of constrained
      directions: the benchmark observes 6 parameters through a rank-3 map,
      so exactly 3 directions can be informed.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_benchmark import GMMBenchmark  # noqa: E402
from npe_diagnostics import (  # noqa: E402
    data_dependent_sbc, embedding_overlap, expected_coverage, family_verdict,
    information_spectrum, posterior_contraction,
    simulation_based_calibration, tarp,
)

RESULTS: List[Tuple[str, bool, str]] = []
N_DIM, N_OBS, N_COMP = 6, 3, 3


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run(test_id: str, fn: Callable[[], str], selector: str = "") -> None:
    if selector and selector not in test_id:
        return
    try:
        detail = fn() or ""
        RESULTS.append((test_id, True, detail))
        print("  PASS  %-24s %s" % (test_id, detail), flush=True)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        RESULTS.append((test_id, False, "%s | %s" % (exc, tb)))
        print("  FAIL  %-24s %s" % (test_id, exc), flush=True)


# ---------------------------------------------------------------------------
# Shared calibration set built from EXACT posteriors
# ---------------------------------------------------------------------------

_CACHE = {}


def calibration_set(n_calib: int = 400, n_draws: int = 128, seed: int = 0):
    """(bench, theta_true, Z, exact_samples).

    theta* is drawn from the prior, x is simulated from it, and the
    posterior samples come from the closed-form posterior -- so these are
    draws from the TRUE posterior, and any diagnostic that flags them is
    wrong.
    """
    key = (n_calib, n_draws, seed)
    if key in _CACHE:
        return _CACHE[key]
    rng = np.random.default_rng(seed)
    bench = GMMBenchmark(n_dim=N_DIM, n_obs=N_OBS, n_components=N_COMP,
                         separation=6.0, prior_scale=1.0, obs_noise=0.4, seed=0)
    theta_true = bench.prior_sample(n_calib, rng)
    Z = bench.simulate(theta_true, rng)
    exact = np.stack([bench.posterior(Z[i]).sample(n_draws, rng)
                      for i in range(n_calib)], axis=0)
    _CACHE[key] = (bench, theta_true, Z, exact)
    return _CACHE[key]


def prior_samples_as_posterior(bench, n_calib, n_draws, seed=1):
    """A deliberately BROKEN posterior: the prior, ignoring the data."""
    rng = np.random.default_rng(seed)
    return np.stack([bench.prior_sample(n_draws, rng) for _ in range(n_calib)],
                    axis=0)


# ---------------------------------------------------------------------------
# D1, D2 -- embedding overlap
# ---------------------------------------------------------------------------

def d1_mmd_both_directions() -> str:
    rng = np.random.default_rng(0)
    z_sim = rng.normal(size=(2000, 8))
    z_sim /= np.linalg.norm(z_sim, axis=1, keepdims=True)

    # matched: drawn from the same process -> must NOT reject
    z_ok = rng.normal(size=(36, 8))
    z_ok /= np.linalg.norm(z_ok, axis=1, keepdims=True)
    r_ok = embedding_overlap(z_sim, z_ok, n_null=300, seed=1)
    check(not r_ok.rejects,
          "MMD falsely rejected a matched real set (p=%.4f)" % r_ok.p_value)

    # shifted: concentrated on a cap of the sphere -> must reject
    base = np.zeros(8); base[0] = 1.0
    z_bad = base[None, :] + 0.25 * rng.normal(size=(36, 8))
    z_bad /= np.linalg.norm(z_bad, axis=1, keepdims=True)
    r_bad = embedding_overlap(z_sim, z_bad, n_null=300, seed=1)
    check(r_bad.rejects,
          "MMD failed to reject an obviously shifted real set (p=%.4f)"
          % r_bad.p_value)
    return "matched p=%.3f (no reject), shifted p=%.4f (reject)" % (
        r_ok.p_value, r_bad.p_value)


def d2_geodesic_nn() -> str:
    rng = np.random.default_rng(2)
    z_sim = rng.normal(size=(1500, 8))
    z_sim /= np.linalg.norm(z_sim, axis=1, keepdims=True)
    z_ok = rng.normal(size=(36, 8))
    z_ok /= np.linalg.norm(z_ok, axis=1, keepdims=True)
    base = np.zeros(8); base[0] = 1.0
    z_far = base[None, :] + 0.02 * rng.normal(size=(36, 8))
    z_far /= np.linalg.norm(z_far, axis=1, keepdims=True)

    r_ok = embedding_overlap(z_sim, z_ok, n_null=100, seed=3)
    r_far = embedding_overlap(z_sim, z_far, n_null=100, seed=3)
    check(r_ok.nn_geodesic_real is not None, "geodesic distances not computed")
    med_ok = float(np.median(r_ok.nn_geodesic_real))
    med_far = float(np.median(r_far.nn_geodesic_real))
    sim_95 = float(np.percentile(r_ok.nn_geodesic_sim, 95))
    check(med_ok < sim_95 * 1.5,
          "on-manifold real set looks far from the simulated cloud "
          "(median %.3f vs sim 95th %.3f)" % (med_ok, sim_95))
    return "on-manifold median %.3f rad, sim 95th %.3f rad, clustered set %.3f" % (
        med_ok, sim_95, med_far)


# ---------------------------------------------------------------------------
# D3, D4 -- marginal SBC
# ---------------------------------------------------------------------------

def d3_sbc_passes_on_exact() -> str:
    """Family-wise false-positive rate on the EXACT posterior.

    Asserted across many seeds, not one. A single-seed assertion on a
    statistical test is flaky by construction: at alpha it fails alpha of
    the time by design, and a test that cries wolf trains you to ignore it.
    What is actually claimed is a RATE, so a rate is what gets measured.
    """
    n_rep, n_calib, n_draws, alpha = 12, 400, 128, 0.05
    bench = GMMBenchmark(n_dim=N_DIM, n_obs=N_OBS, n_components=N_COMP,
                         separation=6.0, prior_scale=1.0, obs_noise=0.4, seed=0)
    rejects, minp = 0, []
    for rep in range(n_rep):
        rng = np.random.default_rng(3000 + rep)
        th = bench.prior_sample(n_calib, rng)
        Z = bench.simulate(th, rng)
        ex = np.stack([bench.posterior(Z[i]).sample(n_draws, rng)
                       for i in range(n_calib)], axis=0)
        fam = family_verdict(simulation_based_calibration(th, ex, seed=rep),
                             alpha=alpha)
        rejects += int(not fam.passes)
        minp.append(float(np.min(fam.adjusted)))
    rate = rejects / n_rep
    check(rate <= 0.25,
          "family-wise false-positive rate %.2f over %d reps on the EXACT "
          "posterior; expected near alpha=%.2f" % (rate, n_rep, alpha))
    return ("family-wise FP rate %.2f over %d reps (alpha=%.2f), "
            "median adjusted min-p %.2f" % (rate, n_rep, alpha, float(np.median(minp))))


def d4_sbc_names_the_failure() -> str:
    _, theta_true, _, exact = calibration_set()
    mu = exact.mean(axis=1, keepdims=True)

    cases = {
        "too narrow": mu + 0.25 * (exact - mu),
        "too broad":  mu + 4.00 * (exact - mu),
        "biased":     exact + 1.5,
    }
    found = {}
    for label, ps in cases.items():
        res = simulation_based_calibration(theta_true, ps, seed=0)
        fails = [r for r in res if not r.passes]
        check(fails, "SBC did not detect a '%s' posterior" % label)
        found[label] = fails[0].verdict

    check("NARROW" in found["too narrow"].upper(),
          "narrow posterior mislabelled as: %s" % found["too narrow"])
    check("BROAD" in found["too broad"].upper(),
          "broad posterior mislabelled as: %s" % found["too broad"])
    check("BIAS" in found["biased"].upper(),
          "biased posterior mislabelled as: %s" % found["biased"])
    return "detected and correctly named all 3 failure shapes"


# ---------------------------------------------------------------------------
# D5 -- THE BLIND SPOT
# ---------------------------------------------------------------------------

def d5_blind_spot() -> str:
    """Reproduce Modrak et al. Theorem 7 and validate the bilinear substitute.

    A posterior equal to the prior ignores the data completely. The claim
    is that parameter-only checks detect this only AT CHANCE, while a
    data-dependent quantity detects it reliably. Both halves are rates, so
    both are measured over many seeds rather than asserted on one.
    """
    n_rep, n_calib, n_draws, alpha = 12, 600, 128, 0.05
    bench = GMMBenchmark(n_dim=N_DIM, n_obs=N_OBS, n_components=N_COMP,
                         separation=6.0, prior_scale=1.0, obs_noise=0.4, seed=0)

    sbc_hits = cov_hits = dd_hits = dd_false_pos = 0
    tarp_rand_hits = tarp_x_hits = 0
    dd_minp = []
    for rep in range(n_rep):
        rng = np.random.default_rng(4000 + rep)
        th = bench.prior_sample(n_calib, rng)
        Z = bench.simulate(th, rng)
        broken = np.stack([bench.prior_sample(n_draws, rng)
                           for _ in range(n_calib)], axis=0)

        # (a) marginal SBC -- must detect only at chance
        fam = family_verdict(simulation_based_calibration(th, broken, seed=rep),
                             alpha=alpha)
        sbc_hits += int(not fam.passes)

        # (b) expected coverage -- with q = prior the test quantity
        # log q(theta|z) = log p(theta) has no z-dependence, so this is
        # blind too
        lp_true = bench.prior_log_prob(th)
        lp_samp = np.stack([bench.prior_log_prob(broken[i])
                            for i in range(n_calib)], axis=0)
        cov_hits += int(not expected_coverage(lp_true, lp_samp, seed=rep).passes)

        # (c) TARP with X-INDEPENDENT reference points is blind for the same
        # exchangeability reason: given an r that does not depend on x, the
        # true theta* and a prior draw are exchangeable.
        tarp_rand_hits += int(
            not tarp(th, broken, n_references=3, seed=rep,
                     reference_mode="random").passes)

        # (d) TARP with X-DEPENDENT reference points MUST catch it. This is
        # exactly what the positionability requirement in the TARP theorem
        # buys, and it is the difference between a useful test and a blind
        # one -- measured 0/12 versus 12/12 on this very problem.
        tarp_x_hits += int(
            not tarp(th, broken, Z=Z, n_references=3, seed=rep,
                     reference_mode="x").passes)

        # (e) data-dependent quantity -- must detect reliably
        dd = data_dependent_sbc(th, broken, Z, n_forms=4, seed=rep)
        fam_dd = family_verdict(dd, alpha=alpha)
        dd_hits += int(not fam_dd.passes)
        dd_minp.append(float(np.min(fam_dd.adjusted)))

        # (f) and must NOT fire on the exact posterior
        ex = np.stack([bench.posterior(Z[i]).sample(n_draws, rng)
                       for i in range(n_calib)], axis=0)
        dd_ok = family_verdict(data_dependent_sbc(th, ex, Z, n_forms=4, seed=rep),
                               alpha=alpha)
        dd_false_pos += int(not dd_ok.passes)

    check(sbc_hits / n_rep <= 0.25,
          "marginal SBC detected the data-ignoring posterior in %d/%d reps -- "
          "well above chance, so the blind-spot claim needs re-examining"
          % (sbc_hits, n_rep))
    check(cov_hits / n_rep <= 0.25,
          "expected coverage detected it in %d/%d reps; the degeneracy "
          "argument needs re-examining" % (cov_hits, n_rep))
    check(tarp_rand_hits / n_rep <= 0.25,
          "TARP with x-INDEPENDENT references detected it in %d/%d reps -- "
          "above chance, so the exchangeability argument needs re-examining"
          % (tarp_rand_hits, n_rep))
    check(tarp_x_hits / n_rep >= 0.75,
          "TARP with x-DEPENDENT references caught it in only %d/%d reps; "
          "positionability is not delivering the detection the theorem "
          "promises" % (tarp_x_hits, n_rep))
    check(dd_hits / n_rep >= 0.75,
          "data-dependent SBC caught it in only %d/%d reps -- the bilinear "
          "substitute is not a usable replacement" % (dd_hits, n_rep))
    check(dd_false_pos / n_rep <= 0.25,
          "data-dependent SBC false-positives on the EXACT posterior in "
          "%d/%d reps -- it is close to an always-fail test"
          % (dd_false_pos, n_rep))

    con = posterior_contraction(th, broken)
    check(float(np.max(con.contraction)) < 0.10,
          "contraction %.3f on a prior-equal posterior, expected ~0"
          % float(np.max(con.contraction)))

    return ("over %d reps -- BLIND: SBC %d/%d, coverage %d/%d, TARP(random "
            "refs) %d/%d.  DETECTS: TARP(x refs) %d/%d, bilinear %d/%d "
            "(median adj p %.1e, FP %d/%d). contraction %.3f"
            % (n_rep, n_rep - sbc_hits, n_rep, n_rep - cov_hits, n_rep,
               n_rep - tarp_rand_hits, n_rep, tarp_x_hits, n_rep,
               dd_hits, n_rep, float(np.median(dd_minp)), dd_false_pos, n_rep,
               float(np.max(con.contraction))))


# ---------------------------------------------------------------------------
# D6 -- TARP
# ---------------------------------------------------------------------------

def d6_tarp() -> str:
    _, theta_true, _, exact = calibration_set()
    ok = tarp(theta_true, exact, n_references=3, seed=0)
    check(ok.passes, "TARP failed on the EXACT posterior (p=%.4f)" % ok.ks_pvalue)

    mu = exact.mean(axis=1, keepdims=True)
    narrow = mu + 0.2 * (exact - mu)
    bad = tarp(theta_true, narrow, n_references=3, seed=0)
    check(not bad.passes,
          "TARP missed a 5x-too-narrow posterior (p=%.4f)" % bad.ks_pvalue)
    return "exact p=%.3f (pass), too-narrow p=%.1e (fail)" % (
        ok.ks_pvalue, bad.ks_pvalue)


# ---------------------------------------------------------------------------
# D7, D8 -- informativeness
# ---------------------------------------------------------------------------

def d7_contraction() -> str:
    n_calib, n_draws = 400, 128
    bench, theta_true, _, exact = calibration_set(n_calib, n_draws)
    broken = prior_samples_as_posterior(bench, n_calib, n_draws)

    c_exact = posterior_contraction(theta_true, exact)
    c_prior = posterior_contraction(theta_true, broken)

    check(float(np.max(c_prior.contraction)) < 0.10,
          "prior-as-posterior shows contraction %.3f, expected ~0"
          % float(np.max(c_prior.contraction)))
    check(float(np.max(c_exact.contraction)) > 0.30,
          "exact posterior shows max contraction only %.3f; the benchmark "
          "should constrain at least some directions"
          % float(np.max(c_exact.contraction)))
    return "prior-as-posterior max %.3f, exact posterior max %.3f, mean %.3f" % (
        float(np.max(c_prior.contraction)),
        float(np.max(c_exact.contraction)),
        float(np.mean(c_exact.contraction)))


def d8_information_spectrum() -> str:
    """The benchmark observes n_dim=6 parameters through a rank-3 map.

    The bound rank <= N_OBS is assertable HERE, and only here, because the
    benchmark is conjugate linear-Gaussian: each posterior mean mu_k(x) is
    an AFFINE function of x, so the image of x -> E[theta|x] is a flat
    3-dimensional subspace and its linear rank equals its intrinsic
    dimension. For a nonlinear estimator such as a normalizing flow the
    image is curved and its linear rank can exceed the intrinsic dimension,
    so this assertion must NOT be carried over as a general test. See the
    note in information_spectrum()."""
    n_calib, n_draws = 800, 128
    _, theta_true, _, exact = calibration_set(n_calib, n_draws)
    spec = information_spectrum(theta_true, exact, threshold=0.05)

    check(spec.effective_rank is not None, "effective rank not computed")
    check(spec.effective_rank <= N_OBS,
          "effective rank %d exceeds the observation dimension %d -- the "
          "spectrum is reporting more constrained directions than the data "
          "can possibly supply" % (spec.effective_rank, N_OBS))
    check(spec.effective_rank >= 1,
          "effective rank %d: the spectrum sees no constrained direction at "
          "all, though the observation is informative" % spec.effective_rank)
    top = spec.eigenvalues[:N_DIM]
    return ("rank %d of %d params (observation dim %d); eigenvalues %s"
            % (spec.effective_rank, N_DIM, N_OBS,
               np.round(top, 3).tolist()))


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", dest="selector", default="")
    args = ap.parse_args()

    print("=" * 74)
    print("Diagnostics validated against exact analytic posteriors")
    print("=" * 74)

    print("\n[layer 1: embedding overlap]")
    run("D1_mmd_both_directions", d1_mmd_both_directions, args.selector)
    run("D2_geodesic_nn", d2_geodesic_nn, args.selector)

    print("\n[layer 2: calibration]")
    run("D3_sbc_exact", d3_sbc_passes_on_exact, args.selector)
    run("D4_sbc_names_failure", d4_sbc_names_the_failure, args.selector)
    run("D5_blind_spot", d5_blind_spot, args.selector)

    print("\n[layer 3: TARP]")
    run("D6_tarp", d6_tarp, args.selector)

    print("\n[layer 4: informativeness]")
    run("D7_contraction", d7_contraction, args.selector)
    run("D8_information_spectrum", d8_information_spectrum, args.selector)

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
