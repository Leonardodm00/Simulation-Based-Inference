#!/usr/bin/env python3
"""
npe_tune_gates.py -- the three eligibility gates.

Scope boundary: gates only. This module decides pass/fail; it never ranks.
Ranking is by held-out NLL and lives in the driver. The distinction is the
whole point:

  * Optimising a CALIBRATION statistic selects for prior-like posteriors. A
    posterior exactly equal to the prior is perfectly calibrated when theta
    is drawn from the prior, so a search that maximised SBC uniformity would
    be actively rewarded for learning nothing.

  * Optimising NLL alone cannot see overconfidence. A sharp, badly centred
    posterior can score well on average while its credible regions
    under-cover.

Neither direction sees the other, so both are needed and neither may be the
objective. Hence: gates gate, NLL ranks.

The gates, in increasing cost:

  G1 informativeness -- Delta_hat above a MEASURED threshold, with the
     bootstrap interval and the across-seed spread both clear of it. Costs
     nothing beyond the score that was computed anyway, so it runs at every
     evaluation. This is the gate the calibration battery is provably blind
     to (repository README, measured on a prior-equal posterior: marginal
     SBC 0/12, expected coverage 0/12).

  G2 marginal calibration -- per-axis SBC rank uniformity, aggregated with
     Holm-Bonferroni over the p axes. Without the correction, a perfectly
     calibrated posterior trips something roughly 13% of the time at 27
     axes and alpha=0.005 per axis.

  G3 joint calibration and data-dependence -- expected coverage,
     data-dependent SBC, and TARP with the Z argument supplied. The Z
     argument is not cosmetic: TARP's theorem needs reference points that
     are a function of the observation, and the repository measures 12/12
     detections with x-dependent references against 0/12 with random ones.

G2 and G3 need posterior sampling for every calibration observation, for
every member, which is why they run on the finalists only. Diagnostics are
imported lazily so this module can be imported (and its logic tested)
without the deep-learning stack present.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "GateResult",
    "GateBattery",
    "gate_g1_informativeness",
    "gate_g2_marginal_calibration",
    "gate_g3_joint_calibration",
    "run_all_gates",
    "delta_min_from_control",
]


@dataclass
class GateResult:
    """One gate's verdict, with the numbers that produced it."""

    name: str
    passed: bool
    detail: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return "  %-32s %s  %s" % (self.name,
                                   "PASS" if self.passed else "FAIL",
                                   self.detail)


@dataclass
class GateBattery:
    """The full verdict over the gates that were actually run."""

    results: List[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results) and bool(self.results)

    def get(self, name: str) -> Optional[GateResult]:
        for r in self.results:
            if r.name == name:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed,
                "results": [r.to_dict() for r in self.results]}

    def summary(self) -> str:
        lines = ["  gate battery: %s" % ("PASS" if self.passed else "FAIL")]
        lines += [r.summary() for r in self.results]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# G1 -- informativeness
# ---------------------------------------------------------------------------

def delta_min_from_control(control_deltas: Sequence[float],
                           k_sigma: float = 3.0,
                           floor: float = 0.0) -> float:
    """Set the G1 threshold from the shuffled-pairs control, not by taste.

    The control destroys the association between theta and z while keeping
    both marginals, so its achievable optimum IS the prior and its measured
    information gain is the "learned nothing" floor for this bank, including
    whatever optimism the finite evaluation split contributes. The threshold
    is that floor plus k_sigma standard deviations.

    With fewer than two control runs the spread is unknown, so the threshold
    falls back to `floor` and the caller is expected to say so in the report
    rather than present a measured-looking number.
    """
    d = np.asarray([x for x in control_deltas if np.isfinite(x)],
                   dtype=np.float64)
    if d.size == 0:
        return float(floor)
    if d.size == 1:
        return float(max(floor, d[0]))
    return float(max(floor, np.mean(d) + k_sigma * np.std(d, ddof=1)))


def gate_g1_informativeness(delta: float,
                            delta_ci_lo: float,
                            delta_min: float,
                            seed_deltas: Optional[Sequence[float]] = None,
                            k_sigma: float = 1.0) -> GateResult:
    """G1: the estimator must beat the prior floor by a measured margin.

    Requires all of:
      (a) the point estimate Delta_hat > delta_min;
      (b) the LOWER end of the bootstrap interval over evaluation rows also
          above delta_min -- so the margin is not an artefact of which rows
          happened to land in the split;
      (c) when several seeds are available, mean - k_sigma * sd above
          delta_min as well -- so it is not an artefact of one lucky
          initialisation.

    Failing G1 is not a defect of the tuner. It is the informative branch of
    the identity Delta = I(theta; z) - E[KL]: either the embedding carries
    nothing about theta, or the estimator did not extract it. Which of the
    two is decided by the capacity sweep and the learning curve, not here.
    """
    stats: Dict[str, Any] = {"delta": float(delta),
                             "delta_ci_lo": float(delta_ci_lo),
                             "delta_min": float(delta_min)}
    ok_point = bool(np.isfinite(delta) and delta > delta_min)
    ok_ci = bool((not np.isfinite(delta_ci_lo)) or delta_ci_lo > delta_min)
    ok_seed = True
    if seed_deltas is not None and len(seed_deltas) >= 2:
        d = np.asarray(seed_deltas, dtype=np.float64)
        lo = float(np.mean(d) - k_sigma * np.std(d, ddof=1))
        stats.update({"seed_mean": float(np.mean(d)),
                      "seed_sd": float(np.std(d, ddof=1)),
                      "seed_lower": lo, "n_seeds": int(d.size)})
        ok_seed = bool(lo > delta_min)

    passed = bool(ok_point and ok_ci and ok_seed)
    bits = []
    if not ok_point:
        bits.append("point estimate %.4f <= threshold %.4f" % (delta, delta_min))
    if not ok_ci:
        bits.append("bootstrap lower bound %.4f <= threshold" % delta_ci_lo)
    if not ok_seed:
        bits.append("across-seed lower bound %.4f <= threshold"
                    % stats.get("seed_lower", float("nan")))
    detail = ("gain %.4f nats/row vs threshold %.4f" % (delta, delta_min)
              if passed else "; ".join(bits))
    return GateResult(name="G1_informativeness", passed=passed,
                      detail=detail, stats=stats)


# ---------------------------------------------------------------------------
# G2 -- marginal calibration
# ---------------------------------------------------------------------------

def gate_g2_marginal_calibration(theta_true: np.ndarray,
                                 posterior_samples: np.ndarray,
                                 param_names: Optional[Sequence[str]] = None,
                                 alpha: float = 0.05,
                                 seed: int = 0) -> GateResult:
    """G2: marginal SBC rank uniformity, Holm-Bonferroni over the p axes.

    theta_true : (N, p); posterior_samples : (N, n_draws, p).
    """
    import npe_diagnostics as D

    ranks = D.simulation_based_calibration(theta_true, posterior_samples,
                                           param_names=param_names, seed=seed)
    fam = D.family_verdict(ranks, alpha=alpha)
    n_rej = int(np.sum(np.asarray(fam.rejected, dtype=bool)))
    worst = [n for n, r in zip(fam.names, fam.rejected) if r][:5]
    passed = bool(n_rej == 0)
    return GateResult(
        name="G2_marginal_calibration",
        passed=passed,
        detail=("no axis rejected at alpha=%.3f (Holm-Bonferroni over %d)"
                % (alpha, len(fam.names)) if passed else
                "%d/%d axes rejected; worst: %s"
                % (n_rej, len(fam.names), ", ".join(worst))),
        stats={"n_axes": len(fam.names), "n_rejected": n_rej,
               "alpha": float(alpha),
               "min_pvalue": float(np.min(fam.pvalues)) if len(fam.pvalues) else float("nan"),
               "rejected_names": list(worst)},
    )


# ---------------------------------------------------------------------------
# G3 -- joint calibration and data-dependence
# ---------------------------------------------------------------------------

def gate_g3_joint_calibration(theta_true: np.ndarray,
                              posterior_samples: np.ndarray,
                              Z: np.ndarray,
                              log_prob_true: Optional[np.ndarray] = None,
                              log_prob_samples: Optional[np.ndarray] = None,
                              alpha: float = 0.05,
                              n_forms: int = 4,
                              coverage_tol: float = 0.05,
                              seed: int = 0) -> GateResult:
    """G3: expected coverage, data-dependent SBC, and TARP with Z.

    Sub-checks, all of which must pass:

      (i)   expected coverage does not fall below nominal by more than
            coverage_tol anywhere -- an UNDER-covering posterior is
            overconfident, which is the failure that matters
            scientifically; over-coverage is conservative and is allowed;
      (ii)  data-dependent SBC (f = theta^T W z), Holm-Bonferroni over the
            random forms -- the check that actually detects a posterior
            ignoring its conditioner;
      (iii) TARP with Z supplied, so the reference points are a function of
            the observation as the theorem requires.

    Coverage is skipped, with that fact recorded, when the caller could not
    supply log-densities; the other two still run.
    """
    import npe_diagnostics as D

    stats: Dict[str, Any] = {}
    fails: List[str] = []

    # (i) expected coverage
    if log_prob_true is not None and log_prob_samples is not None:
        cov = D.expected_coverage(np.asarray(log_prob_true),
                                  np.asarray(log_prob_samples), seed=seed)
        levels = np.asarray(getattr(cov, "levels", getattr(cov, "ecdf_x", [])),
                            dtype=np.float64)
        emp = np.asarray(getattr(cov, "coverage", getattr(cov, "ecdf_y", [])),
                         dtype=np.float64)
        if levels.size and emp.size == levels.size:
            deficit = float(np.max(levels - emp))
            stats["coverage_max_deficit"] = deficit
            if deficit > coverage_tol:
                fails.append("under-coverage by %.3f (tol %.3f)"
                             % (deficit, coverage_tol))
        else:
            ks = float(getattr(cov, "ks_pvalue", float("nan")))
            stats["coverage_ks_pvalue"] = ks
            if np.isfinite(ks) and ks < alpha:
                fails.append("coverage KS p=%.4f < alpha" % ks)
    else:
        stats["coverage"] = "skipped: log-densities not supplied"

    # (ii) data-dependent SBC
    dd = D.data_dependent_sbc(theta_true, posterior_samples, np.asarray(Z),
                              n_forms=int(n_forms), seed=seed)
    fam = D.family_verdict(dd, alpha=alpha)
    n_rej = int(np.sum(np.asarray(fam.rejected, dtype=bool)))
    stats["dd_sbc_n_forms"] = len(fam.names)
    stats["dd_sbc_n_rejected"] = n_rej
    stats["dd_sbc_min_pvalue"] = (float(np.min(fam.pvalues))
                                  if len(fam.pvalues) else float("nan"))
    if n_rej:
        fails.append("data-dependent SBC rejected %d/%d forms"
                     % (n_rej, len(fam.names)))

    # (iii) TARP, with x-dependent reference points
    tr = D.tarp(theta_true, posterior_samples, Z=np.asarray(Z),
                reference_mode="auto", seed=seed)
    stats["tarp_ks_pvalue"] = float(tr.ks_pvalue)
    stats["tarp_reference_mode"] = str(tr.reference_mode)
    stats["tarp_max_deviation"] = float(tr.max_deviation)
    if str(tr.reference_mode).lower().startswith("random"):
        fails.append("TARP fell back to x-independent reference points; the "
                     "theorem's hypothesis is then unmet and the result is "
                     "not evidence either way")
    elif np.isfinite(tr.ks_pvalue) and tr.ks_pvalue < alpha:
        fails.append("TARP KS p=%.4f < alpha" % tr.ks_pvalue)

    passed = not fails
    return GateResult(
        name="G3_joint_calibration",
        passed=passed,
        detail=("coverage, data-dependent SBC and TARP all clear"
                if passed else "; ".join(fails)),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------

def run_all_gates(delta: float,
                  delta_ci_lo: float,
                  delta_min: float,
                  theta_true: Optional[np.ndarray] = None,
                  posterior_samples: Optional[np.ndarray] = None,
                  Z: Optional[np.ndarray] = None,
                  log_prob_true: Optional[np.ndarray] = None,
                  log_prob_samples: Optional[np.ndarray] = None,
                  param_names: Optional[Sequence[str]] = None,
                  seed_deltas: Optional[Sequence[float]] = None,
                  alpha: float = 0.05,
                  n_forms: int = 4,
                  seed: int = 0,
                  run_calibration: bool = True) -> GateBattery:
    """Run G1 always; G2 and G3 when calibration draws were supplied.

    run_calibration=False runs G1 alone, which is the mode used at every GP
    evaluation. The finalist stage passes the draws and gets all three.
    """
    battery = GateBattery()
    battery.results.append(
        gate_g1_informativeness(delta, delta_ci_lo, delta_min,
                                seed_deltas=seed_deltas))
    if run_calibration and theta_true is not None and posterior_samples is not None:
        battery.results.append(
            gate_g2_marginal_calibration(theta_true, posterior_samples,
                                         param_names=param_names,
                                         alpha=alpha, seed=seed))
        if Z is not None:
            battery.results.append(
                gate_g3_joint_calibration(theta_true, posterior_samples, Z,
                                          log_prob_true=log_prob_true,
                                          log_prob_samples=log_prob_samples,
                                          alpha=alpha, n_forms=n_forms,
                                          seed=seed))
    return battery
