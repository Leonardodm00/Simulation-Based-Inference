#!/usr/bin/env python3
"""
smoke_test_local.py -- do the LOCAL diagnostics detect what they claim to?

Run:
    python smoke_test_local.py
    python smoke_test_local.py -k L3
    python smoke_test_local.py --fast        # skip the many-seed rate tests
    python smoke_test_local.py --seeds 24    # more seeds for L4/L5

Every test scores npe_local.py against EXACT analytic posteriors from
gmm_benchmark.py, so no network is trained and any failure is attributable
to the diagnostic rather than to an approximate reference. Same discipline
as smoke_test_diagnostics.py, and for the same reason.

  L0  Plumbing. hpd_values agrees with the ranks expected_coverage already
      computes (HPD = 1 - r/M); benjamini_hochberg reproduces a
      hand-checkable example, never exceeds Holm, and BY never undercuts BH.
  L1  LCT does not fire on an EXACT posterior, at the family level.
  L2  LCT fires on a GLOBALLY too-narrow posterior at essentially every
      z_o, with the exact posterior as the negative control under the
      IDENTICAL criterion.
  L3  THE TEST THAT JUSTIFIES THE MODULE. A posterior correct everywhere
      except inside a ball in embedding space:
        - marginal SBC PASSES        (global, averaged, defect diluted)
        - expected coverage PASSES   (same)
        - LCT is QUIET far from the ball
        - LCT FIRES inside the ball
      Reported as a RATE over several seeds, not a single-seed pass: the
      global tests have their own false-positive rate, so a single-seed
      assertion here would be flaky by construction. This is the lesson
      already recorded for D3/D5.
  L4  False-positive RATE at nominal alpha over many seeds. A local test
      that fires everywhere is worse than none, because it will fire on the
      real data and be believed.
  L5  The Monte Carlo null is calibrated: p-values under H0 are
      approximately uniform (jittered KS -- they are discrete on multiples
      of 1/(B+1), and an unjittered KS manufactures rejections).
  L6  L-C2ST fires on a gross failure and is quiet on the exact posterior.
                                                       [needs sbi + torch]
  L7  Reuse of the CACHED NULL CLASSIFIERS across observations is valid,
      and the null genuinely varies with z_o.          [needs sbi + torch]
  L8  L-C2ST detects the L3 local failure that the global battery misses.
                                                       [needs sbi + torch]

Tests marked [needs sbi + torch] SKIP with a visible message if sbi is not
importable. A skip is reported as a skip, NEVER as a pass.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_benchmark import GMMBenchmark, GMMPosterior  # noqa: E402
from npe_diagnostics import (  # noqa: E402
    expected_coverage, family_verdict, holm_bonferroni,
    simulation_based_calibration,
)
from npe_local import (  # noqa: E402
    benjamini_hochberg, hpd_values, local_c2st, local_coverage_test,
    local_family_verdict, single_draw_per_observation,
)

RESULTS: List[Tuple[str, str, str]] = []   # (id, PASS|FAIL|SKIP, detail)
N_DIM, N_OBS, N_COMP = 6, 3, 3

# -- L3 configuration -------------------------------------------------------
# Tuned so the global battery genuinely passes: if the defect is large
# enough for expected coverage to catch it, L3 proves nothing. These values
# were selected by sweeping (n_calib, corrupt_frac, scale, k) and keeping
# the setting where, over 6 seeds, the global tests stayed blind 6/6 while
# the local test fired 6/6. The window is narrow -- see the module's own
# report in the docstring of l3_local_failure_global_blind.
L3_N_CALIB = 1200
L3_FRAC = 0.10        # fraction of calibration observations inside the ball
L3_SCALE = 0.86       # covariance scale factor inside the ball
L3_K = 45             # k-NN neighbourhood for the LCT regressor
L3_SEEDS = 6
L3_MIN_RATE = 5.0 / 6.0

GLOBAL_NARROW_SCALE = 0.5   # for L2 and L6: a uniformly too-narrow posterior


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class Skip(Exception):
    """Raised to report a test as skipped rather than passed or failed."""


def run(test_id: str, fn: Callable[[], str], selector: str = "") -> None:
    if selector and selector not in test_id:
        return
    try:
        detail = fn() or ""
        RESULTS.append((test_id, "PASS", detail))
        print("  PASS  %-26s %s" % (test_id, detail), flush=True)
    except Skip as exc:
        RESULTS.append((test_id, "SKIP", str(exc)))
        print("  SKIP  %-26s %s" % (test_id, exc), flush=True)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        RESULTS.append((test_id, "FAIL", "%s | %s" % (exc, tb)))
        print("  FAIL  %-26s %s" % (test_id, exc), flush=True)


def have_sbi() -> bool:
    try:
        import torch  # noqa: F401
        from sbi.diagnostics.lc2st import LC2ST  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Calibration sets with a KNOWN answer
# ---------------------------------------------------------------------------

def narrow_posterior(post: GMMPosterior, scale: float) -> GMMPosterior:
    """The same mixture with every component covariance scaled by scale^2.

    scale < 1 gives a posterior that is too NARROW (overconfident), scale >
    1 one that is too BROAD. Crucially this keeps sample() and log_prob()
    mutually consistent, so the HPD values computed from it are the HPD
    values OF IT -- which they would not be if we shrank draws from the
    exact posterior while keeping the exact density.
    """
    return GMMPosterior(weights=post.weights.copy(),
                        means=post.means.copy(),
                        covs=post.covs * (float(scale) ** 2))


@dataclass
class Calib:
    """One calibration set, plus the geometry of any corrupted region."""
    bench: GMMBenchmark
    theta_true: np.ndarray      # (N, p)
    Z: np.ndarray               # (N, E)
    samples: np.ndarray         # (N, M, p)
    log_true: np.ndarray        # (N,)
    log_samples: np.ndarray     # (N, M)
    dist: np.ndarray            # (N,) distance to the corruption anchor
    radius: float               # ball radius; 0.0 if nothing is corrupted
    corrupted: np.ndarray       # (N,) bool

    @property
    def hpd(self) -> np.ndarray:
        return hpd_values(self.log_true, self.log_samples)

    def core(self, factor: float = 0.65) -> np.ndarray:
        """Indices well INSIDE the corrupted ball.

        Points near the boundary have mixed neighbourhoods, so a local test
        evaluated there is diluted by construction. Evaluating in the core
        is not cheating -- it is the honest statement of what a local test
        can resolve, and the boundary behaviour is what the healthy-region
        false-positive check bounds from the other side.
        """
        return np.flatnonzero(self.dist <= factor * self.radius)

    def far(self, factor: float = 2.0) -> np.ndarray:
        """Indices well OUTSIDE the corrupted ball."""
        return np.flatnonzero(self.dist >= factor * self.radius)


_CACHE: Dict[tuple, Calib] = {}


def calibration_set(n_calib: int = 800, n_draws: int = 128, seed: int = 0,
                    scale: float = 1.0, corrupt_frac: float = 0.0) -> Calib:
    """Build a calibration set whose correct verdict is known by construction.

    theta*_n is drawn from the prior and z_n simulated from it, so
    (theta*_n, z_n) ~ p(theta, z) exactly. The posterior used at z_n is the
    closed-form one, narrowed by `scale` either everywhere (corrupt_frac =
    0) or only inside a ball containing a `corrupt_frac` fraction of the
    calibration observations.

    scale = 1.0 and corrupt_frac = 0.0 gives the EXACT posterior, which no
    diagnostic is allowed to flag.

    The corrupted region is a BALL rather than a half-space: a half-space
    slab gives every interior point a neighbourhood half of which lies
    outside, which dilutes the local statistic for a reason that has
    nothing to do with the diagnostic's power. The anchor is the
    calibration observation closest to the centroid of Z, so the ball sits
    where the embedding cloud is dense.
    """
    key = (int(n_calib), int(n_draws), int(seed), float(scale),
           float(corrupt_frac))
    if key in _CACHE:
        return _CACHE[key]

    rng = np.random.default_rng(seed)
    bench = GMMBenchmark(n_dim=N_DIM, n_obs=N_OBS, n_components=N_COMP,
                         separation=6.0, prior_scale=1.0, obs_noise=0.4,
                         seed=0)
    theta_true = bench.prior_sample(n_calib, rng)
    Z = bench.simulate(theta_true, rng)

    if corrupt_frac > 0.0:
        anchor = Z[int(np.argmin(np.linalg.norm(Z - Z.mean(axis=0), axis=1)))]
        dist = np.linalg.norm(Z - anchor, axis=1)
        radius = float(np.quantile(dist, corrupt_frac))
        corrupted = dist <= radius
    else:
        dist = np.zeros(n_calib, dtype=np.float64)
        radius = 0.0
        corrupted = np.full(n_calib, scale != 1.0, dtype=bool)

    samples = np.empty((n_calib, n_draws, N_DIM), dtype=np.float64)
    log_true = np.empty(n_calib, dtype=np.float64)
    log_samples = np.empty((n_calib, n_draws), dtype=np.float64)
    for i in range(n_calib):
        q = bench.posterior(Z[i])
        if corrupted[i] and scale != 1.0:
            q = narrow_posterior(q, scale)
        samples[i] = q.sample(n_draws, rng)
        log_true[i] = float(q.log_prob(theta_true[i][None, :])[0])
        log_samples[i] = q.log_prob(samples[i])

    out = Calib(bench=bench, theta_true=theta_true, Z=Z, samples=samples,
                log_true=log_true, log_samples=log_samples, dist=dist,
                radius=radius, corrupted=corrupted)
    _CACHE[key] = out
    return out


def pick(idx_pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(min(n, idx_pool.size))
    return np.sort(rng.choice(idx_pool, size=n, replace=False))


# ---------------------------------------------------------------------------
# L0 -- plumbing
# ---------------------------------------------------------------------------

def l0_plumbing() -> str:
    cal = calibration_set(n_calib=200, n_draws=64, seed=0)

    # (a) HPD == 1 - r/M, with r the rank expected_coverage computes.
    hpd = cal.hpd
    ranks = np.sum(cal.log_samples < cal.log_true[:, None], axis=1)
    implied = 1.0 - ranks / float(cal.log_samples.shape[1])
    check(np.allclose(hpd, implied, atol=1e-12),
          "hpd_values disagrees with 1 - r/M; max abs diff %.3e"
          % np.max(np.abs(hpd - implied)))
    check(np.all((hpd >= 0.0) & (hpd <= 1.0)), "HPD values outside [0, 1]")

    # (b) BH on a hand-checkable example: p = 0.01..0.05, n = 5, so
    #     (n/i) * p_(i) = 0.05 for every i, and every adjusted value is 0.05.
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    bh = benjamini_hochberg(p, alpha=0.05)
    check(np.allclose(bh.adjusted, 0.05, atol=1e-12),
          "BH adjusted p-values wrong on the textbook example: %s"
          % np.round(bh.adjusted, 4).tolist())

    # (c) BH is never MORE conservative than Holm, on random families.
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.random(40) ** 2
        check(np.all(benjamini_hochberg(q, alpha=0.05).adjusted
                     <= holm_bonferroni(q, alpha=0.05).adjusted + 1e-12),
              "BH adjusted p exceeded Holm adjusted p, which cannot happen")

    # (d) Benjamini-Yekutieli is strictly more conservative than BH.
    by = benjamini_hochberg(p, alpha=0.05, dependent=True)
    check(np.all(by.adjusted >= bh.adjusted - 1e-12),
          "BY was less conservative than BH")

    # (e) one draw per observation, correct shape and provenance
    one = single_draw_per_observation(cal.samples, seed=0)
    check(one.shape == (cal.samples.shape[0], N_DIM),
          "single_draw_per_observation returned shape %s" % (one.shape,))
    member = [bool(np.any(np.all(np.isclose(cal.samples[i], one[i]), axis=1)))
              for i in range(20)]
    check(all(member),
          "single_draw_per_observation returned a row that is not one of the "
          "posterior draws at that observation")
    return ("HPD==1-r/M exact; BH textbook case OK; BH<=Holm on 50 families; "
            "BY>=BH; single-draw provenance OK")


# ---------------------------------------------------------------------------
# L1, L2 -- LCT on globally correct and globally broken posteriors
# ---------------------------------------------------------------------------

def _lct_global(scale: float, seed: int, n_eval: int = 40, n_null: int = 200,
                n_calib: int = 800, k: int = 60):
    cal = calibration_set(n_calib=n_calib, n_draws=128, seed=seed, scale=scale)
    idx = pick(np.arange(cal.Z.shape[0]), n_eval, seed=seed + 1000)
    res = local_coverage_test(cal.hpd, cal.Z, cal.Z[idx], n_null=n_null,
                              n_neighbors=k, seed=seed, alpha=0.05,
                              z_names=["z_%03d" % i for i in idx])
    return res, idx, cal


def l1_lct_exact() -> str:
    res, _, _ = _lct_global(1.0, seed=0)
    fam = res.family()
    check(fam.passes,
          "LCT rejected on an EXACT posterior after multiplicity correction: "
          "FWER %d, FDR %d rejections; min p=%.4f"
          % (fam.fwer.n_rejected, fam.fdr.n_rejected, res.pvalues.min()))
    return ("exact posterior: 0 family-level rejections, min p=%.3f, "
            "median T=%.5f"
            % (res.pvalues.min(), float(np.median(res.statistics))))


def l2_lct_global_failure() -> str:
    res_bad, _, _ = _lct_global(GLOBAL_NARROW_SCALE, seed=0)
    frac = float(np.mean(res_bad.rejected))
    check(frac >= 0.9,
          "LCT caught a uniformly too-narrow posterior at only %.0f%% of "
          "evaluation points" % (100 * frac))
    check(not res_bad.family().passes,
          "family verdict did not reject a global failure")

    # NEGATIVE CONTROL under the IDENTICAL criterion (the smoke_test_gmm G6
    # pattern): a threshold loose enough to accommodate noise may also
    # accommodate a diagnostic that fires on everything.
    res_ok, _, _ = _lct_global(1.0, seed=0)
    frac_ok = float(np.mean(res_ok.rejected))
    check(frac_ok < 0.2,
          "negative control failed: LCT also fired at %.0f%% of points on the "
          "EXACT posterior, so L2 proves nothing" % (100 * frac_ok))

    worst = int(np.argmin(res_bad.pvalues))
    shape = res_bad.shape_at(worst)
    check("OVERCONFIDENT" in shape,
          "LCT named the failure mode '%s' for a posterior that is too "
          "narrow by construction" % shape)
    return ("narrow(%.2f): %.0f%% rejected; exact: %.0f%% rejected; "
            "shape said '%s'"
            % (GLOBAL_NARROW_SCALE, 100 * frac, 100 * frac_ok, shape))


# ---------------------------------------------------------------------------
# L3 -- the test that justifies the module
# ---------------------------------------------------------------------------

def _l3_one_seed(seed: int, n_eval: int = 30):
    """Return (global_blind, local_fires, local_quiet, detail) for one seed."""
    cal = calibration_set(n_calib=L3_N_CALIB, n_draws=128, seed=seed,
                          scale=L3_SCALE, corrupt_frac=L3_FRAC)

    sbc = simulation_based_calibration(cal.theta_true, cal.samples, seed=seed)
    fam_sbc = family_verdict(sbc, alpha=0.05)
    cov = expected_coverage(cal.log_true, cal.log_samples, seed=seed)
    global_blind = bool(fam_sbc.passes and cov.passes)

    core = pick(cal.core(0.65), n_eval, seed=seed + 7)
    far = pick(cal.far(2.0), n_eval, seed=seed + 8)
    idx = np.concatenate([far, core])
    res = local_coverage_test(
        cal.hpd, cal.Z, cal.Z[idx], n_null=200, n_neighbors=L3_K, seed=seed,
        alpha=0.05,
        z_names=["healthy_%04d" % i for i in far]
                + ["corrupt_%04d" % i for i in core])

    p_far = res.pvalues[:far.size]
    p_core = res.pvalues[far.size:]
    local_fires = not local_family_verdict(p_core, alpha=0.05).passes
    local_quiet = local_family_verdict(p_far, alpha=0.05).passes
    detail = ("sbc_rej=%d cov_p=%.3f | corrupt hit=%.2f healthy hit=%.2f"
              % (fam_sbc.n_rejected, cov.ks_pvalue,
                 float(np.mean(p_core < 0.05)), float(np.mean(p_far < 0.05))))
    return global_blind, local_fires, local_quiet, detail


def l3_local_failure_global_blind() -> str:
    """A defect confined to a ball in embedding space.

    Reported as a rate over L3_SEEDS seeds. Three things must hold jointly,
    and each is required at a rate of at least L3_MIN_RATE:

      (i)   the global battery stays blind,
      (ii)  the local test fires inside the ball,
      (iii) the local test stays quiet far from it.

    (i) is the delicate one. Make the defect stronger and expected coverage
    catches it, at which point L3 proves nothing; make it weaker and the
    LCT loses power. The window exists because, for a defect on a fraction
    f of observations, the global test's signal-to-noise scales as
    f * D * sqrt(N) while the local test's scales as D * sqrt(k) with
    k <~ f * N -- so the local test is ahead by a factor ~ 1 / sqrt(f),
    about 3 at f = 0.10. That factor is the entire justification for this
    module, and it is why the settings above are narrow rather than
    arbitrary.
    """
    gb = lf = lq = 0
    details = []
    for seed in range(L3_SEEDS):
        a, b, c, d = _l3_one_seed(seed)
        gb += int(a)
        lf += int(b)
        lq += int(c)
        details.append("s%d[%s%s%s]" % (seed, "G" if a else "-",
                                        "F" if b else "-", "Q" if c else "-"))
    n = float(L3_SEEDS)
    check(gb / n >= L3_MIN_RATE,
          "the global battery was NOT blind on %d/%d seeds -- the defect is "
          "too large for L3 to prove anything. Move L3_SCALE towards 1.0 or "
          "reduce L3_FRAC. Details: %s"
          % (L3_SEEDS - gb, L3_SEEDS, "; ".join(details)))
    check(lf / n >= L3_MIN_RATE,
          "the LCT failed to fire inside the corrupted ball on %d/%d seeds. "
          "If this cannot be fixed by raising L3_K or L3_N_CALIB, the module "
          "has no reason to exist and that is the finding to report. "
          "Details: %s" % (L3_SEEDS - lf, L3_SEEDS, "; ".join(details)))
    check(lq / n >= L3_MIN_RATE,
          "the LCT fired FAR from the corrupted ball on %d/%d seeds; the "
          "localisation is not real. Details: %s"
          % (L3_SEEDS - lq, L3_SEEDS, "; ".join(details)))
    return ("over %d seeds: global blind %d, local fires %d, local quiet %d "
            "(frac=%.2f scale=%.2f k=%d N=%d)"
            % (L3_SEEDS, gb, lf, lq, L3_FRAC, L3_SCALE, L3_K, L3_N_CALIB))


# ---------------------------------------------------------------------------
# L4, L5 -- size and null calibration over many seeds
# ---------------------------------------------------------------------------

_PVAL_CACHE: Dict[int, np.ndarray] = {}


def _pvalues_over_seeds(n_seeds: int) -> np.ndarray:
    if n_seeds in _PVAL_CACHE:
        return _PVAL_CACHE[n_seeds]
    out = []
    for s in range(n_seeds):
        res, _, _ = _lct_global(1.0, seed=s, n_eval=20, n_null=200,
                                n_calib=400, k=50)
        out.append(res.pvalues)
    _PVAL_CACHE[n_seeds] = np.concatenate(out)
    return _PVAL_CACHE[n_seeds]


def make_l4(n_seeds: int) -> Callable[[], str]:
    def l4_false_positive_rate() -> str:
        p = _pvalues_over_seeds(n_seeds)
        rate = float(np.mean(p < 0.05))
        # The 20 tests within one seed share calibration points and are NOT
        # independent, so this is a generous sanity bound, not an exact
        # size test. L5 is the sharper statement.
        check(rate <= 0.15,
              "uncorrected rejection rate %.3f on EXACT posteriors, far above "
              "the nominal 0.05 over %d seeds (%d tests)"
              % (rate, n_seeds, p.size))
        return ("rejection rate %.3f at nominal 0.05 over %d seeds "
                "(%d local tests)" % (rate, n_seeds, p.size))
    return l4_false_positive_rate


def make_l5(n_seeds: int) -> Callable[[], str]:
    def l5_null_calibration() -> str:
        from scipy import stats
        p = _pvalues_over_seeds(n_seeds)
        # Monte Carlo p-values live on multiples of 1/(B+1); jitter before a
        # continuous KS test, for the same reason _rank_uniformity jitters
        # ranks.
        rng = np.random.default_rng(0)
        b = 200
        pj = np.clip(p - rng.random(p.size) / (b + 1.0), 0.0, 1.0)
        ks = float(stats.kstest(pj, "uniform").pvalue)
        med = float(np.median(p))
        check(0.30 <= med <= 0.70,
              "median p-value under H0 is %.3f, not near 0.5 -- the null is "
              "mis-centred" % med)
        check(ks > 0.001,
              "p-values under H0 are not uniform (jittered KS p=%.2e); the "
              "Monte Carlo null is not calibrated" % ks)
        return ("median p=%.3f, jittered KS p=%.3f over %d seeds (%d tests)"
                % (med, ks, n_seeds, p.size))
    return l5_null_calibration


# ---------------------------------------------------------------------------
# L6, L7, L8 -- L-C2ST (needs sbi + torch)
# ---------------------------------------------------------------------------

def _lc2st_args(cal: Calib, eval_idx: np.ndarray, seed: int) -> dict:
    return dict(theta_cal=cal.theta_true,
                Z_cal=cal.Z,
                theta_q=single_draw_per_observation(cal.samples, seed=seed),
                Z_eval=cal.Z[eval_idx],
                theta_o=cal.samples[eval_idx])


def l6_lc2st_gross_failure() -> str:
    if not have_sbi():
        raise Skip("sbi/torch not importable here; run where sbi==0.27.0 lives")
    seed = 0
    cal_ok = calibration_set(n_calib=800, n_draws=128, seed=seed, scale=1.0)
    cal_bad = calibration_set(n_calib=800, n_draws=128, seed=seed,
                              scale=GLOBAL_NARROW_SCALE)
    idx = pick(np.arange(cal_ok.Z.shape[0]), 12, seed=seed + 1000)

    bad = local_c2st(n_null=50, seed=1, alpha=0.05,
                     **_lc2st_args(cal_bad, idx, seed))
    ok = local_c2st(n_null=50, seed=1, alpha=0.05,
                    **_lc2st_args(cal_ok, idx, seed))

    f_bad = float(np.mean(bad.rejected))
    f_ok = float(np.mean(ok.rejected))
    check(f_bad >= 0.8,
          "L-C2ST caught the too-narrow posterior at only %.0f%% of points"
          % (100 * f_bad))
    check(f_ok <= 0.2,
          "NEGATIVE CONTROL: L-C2ST also fired at %.0f%% of points on the "
          "EXACT posterior" % (100 * f_ok))
    check(bad.probabilities is not None
          and bad.probabilities.shape[0] == idx.size,
          "per-draw classifier probabilities were not returned; that map is "
          "the reason to prefer L-C2ST over a P-P plot")
    return ("narrow(%.2f): %.0f%% rejected (median t=%.4f); exact: %.0f%% "
            "rejected (median t=%.4f)"
            % (GLOBAL_NARROW_SCALE, 100 * f_bad,
               float(np.median(bad.statistics)), 100 * f_ok,
               float(np.median(ok.statistics))))


def l7_cached_null_is_valid() -> str:
    """The CORRECTED version of the handoff's L7.

    The handoff asserted that one null distribution serves every
    observation. It does not: sbi's get_statistics_under_null_hypothesis
    takes x_o, and the null law of t_hat(z_o) genuinely depends on z_o --
    sparse regions of embedding space give noisier classifiers and a wider
    null. What is reusable is the B TRAINED CLASSIFIERS. So the claims to
    test are:

      (a) building the cache once and evaluating it at several z_o is
          deterministic and reproducible, and
      (b) the resulting null is NOT constant across z_o -- if it were, the
          per-observation evaluation would be unnecessary and the handoff's
          original claim would have been right after all.
    """
    if not have_sbi():
        raise Skip("sbi/torch not importable here; run where sbi==0.27.0 lives")
    seed = 0
    cal = calibration_set(n_calib=800, n_draws=128, seed=seed, scale=1.0)
    idx = pick(np.arange(cal.Z.shape[0]), 6, seed=seed + 1000)
    args = _lc2st_args(cal, idx, seed)

    a = local_c2st(n_null=50, seed=1, **args)
    b = local_c2st(n_null=50, seed=1, **args)

    check(np.array_equal(a.rejected, b.rejected),
          "two runs with the same seed disagreed on the verdict; the cache "
          "or the seeding is not deterministic")
    check(np.allclose(a.pvalues, b.pvalues),
          "two runs with the same seed gave different p-values")

    mean_null = a.null_statistics.mean(axis=1)
    spread = float(mean_null.std() / max(mean_null.mean(), 1e-12))
    check(spread > 0.0,
          "the null statistic is IDENTICAL at every z_o, which would mean "
          "the cached null needs no re-evaluation; check that "
          "get_statistics_under_null_hypothesis is really receiving x_o")
    return ("6 observations, identical decisions across two builds; null "
            "mean varies by %.1f%% across z_o" % (100 * spread))


def l8_lc2st_local_failure() -> str:
    if not have_sbi():
        raise Skip("sbi/torch not importable here; run where sbi==0.27.0 lives")
    seed = 0
    cal = calibration_set(n_calib=L3_N_CALIB, n_draws=128, seed=seed,
                          scale=L3_SCALE, corrupt_frac=L3_FRAC)
    core = pick(cal.core(0.65), 10, seed=seed + 7)
    far = pick(cal.far(2.0), 10, seed=seed + 8)
    idx = np.concatenate([far, core])

    res = local_c2st(n_null=50, seed=1, alpha=0.05,
                     z_names=["healthy_%04d" % i for i in far]
                             + ["corrupt_%04d" % i for i in core],
                     **_lc2st_args(cal, idx, seed))
    hit = float(np.mean(res.rejected[far.size:]))
    false_hit = float(np.mean(res.rejected[:far.size]))
    check(hit >= 0.6,
          "L-C2ST fired at only %.0f%% of corrupted observations on a defect "
          "the LCT does catch (see L3)" % (100 * hit))
    check(false_hit <= 0.3,
          "L-C2ST fired at %.0f%% of healthy observations" % (100 * false_hit))
    return "corrupted %.0f%% fired, healthy %.0f%% fired" % (
        100 * hit, 100 * false_hit)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", dest="selector", default="")
    ap.add_argument("--fast", action="store_true",
                    help="skip the many-seed rate tests L4 and L5")
    ap.add_argument("--seeds", type=int, default=12,
                    help="number of seeds for L4 and L5 (default 12)")
    args = ap.parse_args()

    print("=" * 74)
    print("Local calibration diagnostics validated against exact posteriors")
    print("=" * 74)
    print("sbi available: %s"
          % ("yes" if have_sbi() else "NO -- L6/L7/L8 will SKIP, not pass"))

    print("\n[plumbing]")
    run("L0_plumbing", l0_plumbing, args.selector)

    print("\n[LCT: does it fire when it should, and only then]")
    run("L1_lct_exact", l1_lct_exact, args.selector)
    run("L2_lct_global_failure", l2_lct_global_failure, args.selector)
    run("L3_local_failure", l3_local_failure_global_blind, args.selector)

    if not args.fast:
        print("\n[LCT: size and null calibration over %d seeds]" % args.seeds)
        run("L4_false_positive_rate", make_l4(args.seeds), args.selector)
        run("L5_null_calibration", make_l5(args.seeds), args.selector)
    else:
        print("\n[LCT: rate tests skipped (--fast)]")

    print("\n[L-C2ST]")
    run("L6_lc2st_gross_failure", l6_lc2st_gross_failure, args.selector)
    run("L7_cached_null", l7_cached_null_is_valid, args.selector)
    run("L8_lc2st_local_failure", l8_lc2st_local_failure, args.selector)

    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print("\n" + "=" * 74)
    print("%d passed, %d failed, %d skipped, %d total"
          % (n_pass, n_fail, n_skip, len(RESULTS)))
    print("=" * 74)
    if n_fail:
        print("\nFailures:")
        for tid, status, detail in RESULTS:
            if status == "FAIL":
                print("  %s: %s" % (tid, detail))
    if n_skip:
        print("\nSkipped (NOT passes):")
        for tid, status, detail in RESULTS:
            if status == "SKIP":
                print("  %s: %s" % (tid, detail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
