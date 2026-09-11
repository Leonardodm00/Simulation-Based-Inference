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

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """PROVISIONAL G1 screen from a bank-level shuffled-pairs control.

    The control destroys the association between theta and z while keeping
    both marginals, so its achievable optimum IS the prior and its measured
    information gain is the "learned nothing" floor for this bank, including
    whatever optimism the finite evaluation split contributes. The threshold
    is that floor plus k_sigma standard deviations.

    With fewer than two control runs the spread is unknown, so the threshold
    falls back to `floor` and the caller is expected to say so in the report
    rather than present a measured-looking number.

    STATUS (HANDOFF_DELTA_MIN_PER_CONFIG_v1 S3, Tier 1). This is a cheap
    SCREEN evaluated at every GP evaluation, not the authoritative verdict.
    Two reasons, both measured rather than assumed:

      1. The null gain distribution is capacity- and recipe-dependent. How
         much of a finite shuffled sample a network can fit, and how much of
         that survives to the held-out split, depends on the architecture,
         the optimiser and the early-stopping rule. A floor measured under
         ONE reference configuration is mis-calibrated for every candidate
         far from it, in a direction nobody can sign without measuring.
      2. `mean + k_sigma * sd` is not a tail probability at small n_ctrl.
         With n_ctrl = 5, s is a poor estimate of sigma and "mean + 3s" is
         nowhere near a 1e-3 tail; its realised false-pass rate is measured
         in test S21 and is far above nominal.

    The authoritative object is `control_pvalue` below, computed per
    finalist against that finalist's OWN controls, then Holm-corrected across
    finalists. Keep this function's output as the screen and as a
    cross-check: a finalist whose own floor differs from the bank-level floor
    by a large factor says how much the configuration's capacity is driving
    the floor, and should be reported rather than silently overridden.
    """
    d = np.asarray([x for x in control_deltas if np.isfinite(x)],
                   dtype=np.float64)
    if d.size == 0:
        return float(floor)
    if d.size == 1:
        return float(max(floor, d[0]))
    return float(max(floor, np.mean(d) + k_sigma * np.std(d, ddof=1)))


@dataclass
class ControlVerdict:
    """One finalist's G1 evidence against its OWN shuffled control.

    Fields
    ------
    name          : finalist identifier (trial id, arm name, ...).
    delta         : Delta_hat_j, the candidate's gain on the GATE split,
                    averaged over its n_s training seeds.
    n_seeds       : n_s, the number of seeds behind `delta`. 1 when `delta`
                    is a single realisation.
    control_mean  : mean of that finalist's own control gains.
    control_sd    : sample SD (ddof=1) of the same.
    n_control     : n_ctrl, the number of control runs.
    t             : the statistic of eq. (S4.1).
    pvalue        : one-sided P[T_nu > t], nu = n_ctrl - 1.
    delta_min_provisional : the bank-level screen, carried for comparison.
    floor         : the hard minimum in nats/row, applied regardless of p.
    above_floor   : delta > floor.
    """

    name: str
    delta: float
    n_seeds: int
    control_mean: float
    control_sd: float
    n_control: int
    t: float
    pvalue: float
    delta_min_provisional: float = float("nan")
    floor: float = 0.0
    above_floor: bool = True
    adjusted_pvalue: float = float("nan")
    rejected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def control_pvalue(delta: float,
                   control_deltas: Sequence[float],
                   n_seeds: int = 1) -> Tuple[float, float]:
    """One-sided t for "this configuration beats its own noise floor".

    For each fixed finalist j, with Delta_hat_j the candidate's gain on the
    gate split averaged over its n_s training seeds, and
    Delta^ctrl_{j,1..n_ctrl} that finalist's own control gains,

        t_j = (Delta_hat_j - mean(Delta^ctrl_j))
              / (s_j * sqrt(1/n_s + 1/n_ctrl)),
        p_j = P[T_nu > t_j],   nu = n_ctrl - 1,

    where s_j is the sample SD (ddof=1) of that finalist's controls.

    The sqrt(1/n_s + 1/n_ctrl) factor is what makes this a PREDICTION
    interval rather than a confidence interval on the control mean, which is
    the correct object because Delta_hat_j is one realisation and not a
    population parameter. Using sqrt(1/n_ctrl) alone treats the candidate as
    if it had no sampling variability of its own and is anticonservative.

    Returns (t, p). Degenerate inputs return (nan, nan) rather than raising,
    so a caller looping over finalists keeps the other verdicts:
      - fewer than 2 controls: nu = 0, no scale estimate;
      - zero control SD: t is +-inf, and p is 0.0 or 1.0 by the sign of the
        difference, which is the correct limit but rests on an SD of exactly
        zero, so it is returned as nan to force the caller to look.
    """
    from scipy import stats

    d = np.asarray([x for x in control_deltas if np.isfinite(x)],
                   dtype=np.float64)
    n_ctrl = int(d.size)
    n_s = max(1, int(n_seeds))
    if n_ctrl < 2 or not np.isfinite(delta):
        return float("nan"), float("nan")
    s = float(np.std(d, ddof=1))
    if not np.isfinite(s) or s <= 0.0:
        return float("nan"), float("nan")
    se = s * math.sqrt(1.0 / n_s + 1.0 / n_ctrl)
    t = (float(delta) - float(np.mean(d))) / se
    p = float(stats.t.sf(t, df=n_ctrl - 1))
    return float(t), p


def per_finalist_control_verdicts(
        candidates: Sequence[Dict[str, Any]],
        alpha: float = 0.05,
        floor: float = 0.0,
        delta_min_provisional: float = float("nan")) -> List[ControlVerdict]:
    """Test ALL K finalists against their own controls, Holm across them.

    `candidates` is a sequence of dicts with keys `name`, `delta`,
    `control_deltas`, and optionally `n_seeds` (default 1).

    Walking down a ranked list and stopping at the first configuration that
    passes is a sequential multiple-comparison procedure: testing up to K
    candidates each at a nominal per-test level inflates the family-wise
    error roughly K-fold. A valid early-stopping version needs an
    alpha-spending rule, which is more machinery than K = 3 is worth. So all
    K are tested, the K p-values are Holm-corrected, and the caller ships the
    best-RANKED configuration among those rejected -- not the one with the
    smallest p.

    A candidate whose p-value is undefined (fewer than 2 controls, or zero
    control spread) is carried with `rejected = False` and an adjusted p of
    nan; it is excluded from the Holm family, since a missing test is not a
    non-significant one. `above_floor` is reported separately and is a hard
    minimum: a configuration can be statistically distinguishable from its
    own noise floor while still being useless in nats/row.
    """
    import npe_diagnostics as D

    out: List[ControlVerdict] = []
    for c in candidates:
        ctrl = np.asarray([x for x in c.get("control_deltas", [])
                           if np.isfinite(x)], dtype=np.float64)
        delta = float(c["delta"])
        n_s = int(c.get("n_seeds", 1) or 1)
        t, p = control_pvalue(delta, ctrl, n_seeds=n_s)
        out.append(ControlVerdict(
            name=str(c.get("name", "")),
            delta=delta, n_seeds=n_s,
            control_mean=(float(np.mean(ctrl)) if ctrl.size else float("nan")),
            control_sd=(float(np.std(ctrl, ddof=1)) if ctrl.size > 1
                        else float("nan")),
            n_control=int(ctrl.size), t=t, pvalue=p,
            delta_min_provisional=float(delta_min_provisional),
            floor=float(floor), above_floor=bool(delta > floor)))

    testable = [k for k, v in enumerate(out) if np.isfinite(v.pvalue)]
    if testable:
        fam = D.holm_bonferroni([out[k].pvalue for k in testable],
                                alpha=float(alpha),
                                names=[out[k].name for k in testable])
        for slot, k in enumerate(testable):
            out[k].adjusted_pvalue = float(fam.adjusted[slot])
            out[k].rejected = bool(fam.rejected[slot]
                                   and out[k].above_floor)
    return out


def format_control_verdicts(verdicts: Sequence[ControlVerdict],
                            alpha: float = 0.05) -> str:
    """The full K-row table. Reporting all K is not optional.

    This procedure has a garden-of-forking-paths failure mode if reported
    selectively, so passes AND failures are printed, with raw p, Holm p, the
    candidate's own gain, and its own control mean and SD.
    """
    lines = ["  per-finalist control test (alpha=%.3f, Holm over %d testable "
             "of %d)" % (alpha,
                         sum(1 for v in verdicts if np.isfinite(v.pvalue)),
                         len(verdicts)),
             "  | finalist | Delta_hat | n_s | ctrl mean | ctrl sd | n_ctrl "
             "| t | raw p | Holm p | > floor | verdict |",
             "  |---|---|---|---|---|---|---|---|---|---|---|"]
    for v in verdicts:
        lines.append("  | %s | %.4f | %d | %.4f | %.4f | %d | %.3f | %.4g "
                     "| %.4g | %s | %s |"
                     % (v.name, v.delta, v.n_seeds, v.control_mean,
                        v.control_sd, v.n_control, v.t, v.pvalue,
                        v.adjusted_pvalue, "yes" if v.above_floor else "NO",
                        "REJECT null" if v.rejected else "not rejected"))
    return "\n".join(lines)


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
                              gate_on_coverage: bool = False,
                              seed: int = 0) -> GateResult:
    """G3: data-dependent SBC and TARP with Z. Expected coverage is a
    reported diagnostic, no longer a gating sub-check.

    Gating sub-checks, both of which must pass:

      (ii)  data-dependent SBC (f = theta^T W z), Holm-Bonferroni over the
            random forms -- the check that actually detects a posterior
            ignoring its conditioner;
      (iii) TARP with Z supplied, so the reference points are a function of
            the observation as the theorem requires.

    Reported but NOT gating:

      (i)   expected coverage. Demoted because it is provably blind exactly
            where G3 matters: for q(theta|z) = p(theta) the HPD generator
            loses its z-dependence, so ECP = 1 - alpha at every alpha by
            construction (Lemos et al. 2023, eq. 20; their Thm. 3 attributes
            this to the HPD generator not being positionable). Sub-check
            (iii) is credited with every failure mode (i) catches, so no
            detection capability is given up. Two further wins: the
            under-coverage branch below was unreachable in practice (the
            getattr chains never matched expected_coverage's RankResult, so
            control always fell to the KS branch and coverage_tol had no
            effect), and G3 no longer depends on log-densities at all --
            neither (ii) nor (iii) needs them -- so a battery can no longer
            report a G3 pass having silently skipped a sub-check.

    Set gate_on_coverage=True to restore the pre-substitution behaviour.

    G2 (marginal SBC) is untouched and remains the only gate that localises
    a failure to a named parameter axis.
    """
    import npe_diagnostics as D

    stats: Dict[str, Any] = {}
    fails: List[str] = []
    notes: List[str] = []   # recorded, never gating

    # (i) expected coverage -- DIAGNOSTIC ONLY unless gate_on_coverage.
    stats["coverage_gating"] = bool(gate_on_coverage)
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
                msg = ("under-coverage by %.3f (tol %.3f)"
                       % (deficit, coverage_tol))
                (fails if gate_on_coverage else notes).append(msg)
        else:
            ks = float(getattr(cov, "ks_pvalue", float("nan")))
            stats["coverage_ks_pvalue"] = ks
            if np.isfinite(ks) and ks < alpha:
                msg = "coverage KS p=%.4f < alpha" % ks
                (fails if gate_on_coverage else notes).append(msg)
    else:
        # Recorded, not silent: with coverage demoted this is no longer a
        # skipped GATE, but the absence of the diagnostic should still be
        # visible in the ledger rather than passed over.
        stats["coverage"] = "not computed: log-densities not supplied"

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
    # Two thresholds coexist: TARPResult.passes uses its own pass_alpha
    # (0.005) while this gate uses `alpha` (0.05 by default). They can
    # disagree on the same data in either direction. This gate is
    # authoritative; the other is recorded so a disagreement is visible in
    # the ledger instead of latent.
    stats["tarp_gate_alpha"] = float(alpha)
    stats["tarp_passes_at_pass_alpha"] = bool(tr.passes)
    stats["tarp_threshold_disagreement"] = bool(
        bool(tr.passes) != bool(not (np.isfinite(tr.ks_pvalue)
                                     and tr.ks_pvalue < alpha)))
    if str(tr.reference_mode).lower().startswith("random"):
        fails.append("TARP fell back to x-independent reference points; the "
                     "theorem's hypothesis is then unmet and the result is "
                     "not evidence either way")
    elif np.isfinite(tr.ks_pvalue) and tr.ks_pvalue < alpha:
        fails.append("TARP KS p=%.4f < alpha" % tr.ks_pvalue)

    if notes:
        stats["non_gating_notes"] = list(notes)

    passed = not fails
    return GateResult(
        name="G3_joint_calibration",
        passed=passed,
        detail=("data-dependent SBC and TARP clear"
                + ("; coverage diagnostic flagged: " + "; ".join(notes)
                   if notes else "")
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
