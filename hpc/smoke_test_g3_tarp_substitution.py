#!/usr/bin/env python3
"""
smoke_test_g3_tarp_substitution.py -- tests for the change that removes
expected coverage from G3's gating path and fixes TARP's reference-point
allocation.

Two independent changes, tested separately:

  A. npe_diagnostics.tarp: the x-dependent branch allocated reference points
     with np.empty_like and filled them fold by fold. A fold skipped by the
     `continue` left its rows UNINITIALISED, and the post-loop guard tested
     only the last `fit` left in the loop variable -- so a first half too
     small with an adequate second half passed the guard while half the
     reference points were garbage memory. Now NaN-filled with an all-rows
     guard, falling back to random references (which G3 treats as a FAILURE,
     deliberately -- the theorem's hypothesis is unmet, so the result is
     evidence neither way).

  B. npe_tune_gates.gate_g3_joint_calibration: expected coverage is computed
     and reported but no longer gates. gate_on_coverage=True restores the old
     behaviour.

The tests that matter here are the NEGATIVE ones: a gate that stops failing
is only an improvement if it still fails on everything it should. T3-T6 exist
to prove nothing else was loosened.

Run:
    python3 smoke_test_g3_tarp_substitution.py

Pure ASCII, LF only.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import npe_diagnostics as D          # noqa: E402
import npe_tune_gates as G           # noqa: E402

OK = []


def check(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          ("   [%s]" % extra) if extra and not cond else ""))


SIGMA = 0.5


def calibrated(n, p, m, seed=0):
    """A correctly calibrated Gaussian posterior, plus its conditioner.

    Calibration means the truth is exchangeable with the draws GIVEN the
    conditioner: both theta* and the draws are iid N(mu_i, SIGMA^2) around a
    per-observation centre mu_i that z carries. Centring the draws ON the
    truth instead would be underconfident, not calibrated -- which is what
    the first version of this fixture got wrong.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(n, p))                      # posterior centre
    theta = mu + SIGMA * rng.normal(size=(n, p))      # truth ~ q(.|z)
    draws = mu[:, None, :] + SIGMA * rng.normal(size=(n, m, p))
    Z = mu + rng.normal(0, 0.05, size=(n, p))         # z carries mu
    return theta, draws, Z, mu


def prior_posterior(n, p, m, seed=0):
    """The failure G3 exists to catch: draws that ignore z entirely."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(n, p))
    theta = mu + SIGMA * rng.normal(size=(n, p))
    draws = rng.normal(0, np.sqrt(1.0 + SIGMA ** 2),
                       size=(n, m, p))                # the prior, ignores z
    Z = mu + rng.normal(0, 0.05, size=(n, p))
    return theta, draws, Z, mu


def main():
    print("A. TARP reference-point allocation")

    # T1: the exact trigger -- first half too small to fit, second adequate.
    # p = 4 needs p+1 = 5 rows per half; n = 9 gives halves of 4 and 5.
    p, n, m = 4, 9, 64
    theta, draws, Z, _ = calibrated(n, p, m, seed=1)
    tr = D.tarp(theta, draws, Z=Z, reference_mode="auto", seed=3)
    check("T1 undersized fold does not silently produce garbage references",
          tr.reference_mode == "random",
          "reference_mode=%s" % tr.reference_mode)
    check("T1 the fallback is explained in notes",
          any("fell back to random references" in s for s in tr.notes),
          "notes=%r" % (tr.notes,))
    check("T1 the note names how many rows could not be fit",
          any("of %d observations" % n in s for s in tr.notes),
          "notes=%r" % (tr.notes,))
    check("T1 outputs are finite (the old code could emit NaN/garbage)",
          np.isfinite(tr.ks_pvalue) and np.all(np.isfinite(tr.ecdf_x)))

    # T2: with enough rows the x-dependent branch is still taken -- the fix
    # must not have disabled the branch it was protecting.
    n2 = 200
    theta2, draws2, Z2, _ = calibrated(n2, p, m, seed=2)
    tr2 = D.tarp(theta2, draws2, Z=Z2, reference_mode="auto", seed=3)
    check("T2 adequate n still uses x-dependent references",
          tr2.reference_mode == "x-dependent",
          "reference_mode=%s" % tr2.reference_mode)
    check("T2 no spurious fallback note", not tr2.notes, "notes=%r" % (tr2.notes,))

    # T3: pass_alpha is now a visible field, default unchanged at 0.005.
    check("T3 TARPResult.pass_alpha defaults to 0.005 (callers unchanged)",
          abs(getattr(tr2, "pass_alpha", -1) - 0.005) < 1e-12)

    print("B. G3: coverage demoted, everything else unchanged")

    n3, p3, m3 = 300, 4, 128
    theta_c, draws_c, Z_c, mu_c = calibrated(n3, p3, m3, seed=5)
    theta_b, draws_b, Z_b, _ = prior_posterior(n3, p3, m3, seed=6)

    # Log-densities from a deliberately OVER-confident model (sigma too
    # small), so the coverage diagnostic has something to complain about
    # while (ii) and (iii) still see a calibrated posterior. This is exactly
    # the case the substitution changes: G3 must now PASS here.
    s_narrow = SIGMA / 3.0
    lp_true = -0.5 * np.sum(((theta_c - mu_c) / s_narrow) ** 2, axis=1)
    lp_samp = -0.5 * np.sum(((draws_c - mu_c[:, None, :]) / s_narrow) ** 2,
                            axis=2)

    r_cal = G.gate_g3_joint_calibration(
        theta_c, draws_c, Z_c, log_prob_true=lp_true,
        log_prob_samples=lp_samp, seed=11)
    check("T4 calibrated posterior passes G3", r_cal.passed,
          "detail=%s" % r_cal.detail)
    check("T4 coverage is still REPORTED",
          any(k.startswith("coverage") for k in r_cal.stats),
          "stats keys=%r" % sorted(r_cal.stats))
    check("T4 coverage is marked non-gating",
          r_cal.stats.get("coverage_gating") is False)
    check("T4 the TARP threshold disagreement is recorded",
          "tarp_threshold_disagreement" in r_cal.stats)

    # T5: the failure G3 exists to catch must STILL fail.
    r_bad = G.gate_g3_joint_calibration(theta_b, draws_b, Z_b, seed=11)
    check("T5 data-ignoring posterior still FAILS G3", not r_bad.passed,
          "detail=%s" % r_bad.detail)
    check("T5 and it fails on (ii) or (iii), not on coverage",
          ("SBC" in r_bad.detail) or ("TARP" in r_bad.detail),
          "detail=%s" % r_bad.detail)

    # T6: the random-reference fallback must remain a FAILURE (handoff 4c) --
    # it is now the last line of defence.
    r_norefs = G.gate_g3_joint_calibration(
        theta_b[:9], draws_b[:9], Z_b[:9], seed=11)
    check("T6 random-reference fallback is still a G3 FAILURE",
          (not r_norefs.passed)
          and "x-independent reference points" in r_norefs.detail,
          "detail=%s" % r_norefs.detail)

    # T7: with no log-densities G3 still runs, and says so explicitly rather
    # than silently skipping a gate.
    r_nolp = G.gate_g3_joint_calibration(theta_c, draws_c, Z_c, seed=11)
    check("T7 G3 runs without log-densities", r_nolp.passed,
          "detail=%s" % r_nolp.detail)
    check("T7 their absence is recorded, not silent",
          "not computed" in str(r_nolp.stats.get("coverage", "")),
          "coverage=%r" % r_nolp.stats.get("coverage"))

    # T8: the escape hatch restores the old behaviour.
    r_on = G.gate_g3_joint_calibration(
        theta_c, draws_c, Z_c, log_prob_true=lp_true,
        log_prob_samples=lp_samp, gate_on_coverage=True, seed=11)
    check("T8 gate_on_coverage=True flips coverage back to gating",
          r_on.stats.get("coverage_gating") is True)

    print("")
    print("SMOKE TEST: %s  (%d/%d)"
          % ("PASS" if all(OK) else "FAIL", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    sys.exit(main())
