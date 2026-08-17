#!/usr/bin/env python3
"""
npe_local.py -- layer 5: LOCAL calibration diagnostics.

Every diagnostic in npe_diagnostics.py is GLOBAL: its statistic averages
over the whole calibration set, so it answers "is the estimator calibrated
on average?". This module answers "is the estimator calibrated HERE?", at
one fixed observation z_o.

Two diagnostics, deliberately different in cost and in what they can see.

  1. LCT -- Local Coverage Test (Zhao, Dalmasso, Izbicki, Lee, UAI 2021).
     Compress the posterior at each calibration point to a scalar HPD value,
     then REGRESS that value on the observation. Cheap: it reuses arrays the
     pipeline already persists and trains no classifier.

  2. L-C2ST -- Local Classifier Two-Sample Test (Linhart, Gramfort,
     Rodrigues, NeurIPS 2023; arXiv:2306.03580). Train a binary classifier
     to separate the true joint from the estimated joint, then read its
     output at a fixed z_o. Sharper: it compares full joint distributions
     rather than a scalar summary. Implemented here as a THIN WRAPPER over
     sbi.diagnostics.lc2st.LC2ST, which is the authors' own implementation
     and ships inside the pinned sbi==0.27.0.

DEFINITIONS (notation follows this repo: z is the conditioning variable,
the papers call it x).

HPD value, for each fixed z and each fixed theta:

    HPD(theta ; z) = INT_{theta' : q(theta'|z) >= q(theta|z)} q(theta'|z) dtheta'

estimated from M posterior draws theta^q_{n,m} ~ q(theta | z_n) by

    HPD_hat_n = (1/M) SUM_m 1{ log q(theta^q_{n,m} | z_n)
                               >= log q(theta*_n | z_n) }.            (H)

Theorem 3 of Zhao et al.: for each fixed z, if the LOCAL null
H0(z) : q(. | z) = p(. | z) holds, then HPD(theta ; z) | z ~ Unif(0,1).
This is the multivariate-valid route -- it does NOT require a coordinate
ordering, so it is the correct statistic at p = 27, unlike per-coordinate
PIT or per-coordinate SBC ranks.

LCT statistic, for each fixed z_o and a grid G of levels alpha:

    r_alpha(z)   = E[ 1{ HPD(theta ; z) < alpha } | z ]              (R)
    T(z_o)       = (1/|G|) SUM_{alpha in G} ( r_hat_alpha(z_o) - alpha )^2  (T)

with r_hat_alpha any regression estimator of (R). Under the local null,
r_alpha(z_o) = alpha for all alpha in (0,1), so T(z_o) = 0.

L-C2ST statistic, for each fixed z_o:

    t_hat(z_o) = (1/M) SUM_m ( d_omega(theta^q_{o,m}, z_o) - 1/2 )^2  (C)

with theta^q_{o,m} ~ q(theta | z_o) and d_omega the trained classifier.
Under the local null d_omega(. , z_o) == 1/2 and t_hat(z_o) = 0.

WHY THIS LAYER EXISTS (and why the global layer cannot replace it).

Theorem 4 of Zhao et al.: if there exists any map g with
q(theta | z) = p(theta | g(z)), then HPD(theta ; z) ~ Unif(0,1)
MARGINALLY. Take g constant: that is the posterior that ignores the data
entirely, and every test of marginal uniformity passes on it. This is the
peer-reviewed statement of the blind spot this project measured in
smoke_test_diagnostics.py D5. Conditioning on z is what breaks the
degeneracy: HPD(theta ; z) | z is NOT uniform for such a q, so the LCT
sees what expected_coverage provably cannot.

BOTH NULL DISTRIBUTIONS ARE z_o-DEPENDENT. What amortizes is the
EXPENSIVE FITTING, not the null quantiles:

  - LCT: the |G| x B null regressions are fitted once on labels that do not
    depend on the estimator at all, then EVALUATED at every z_o.
  - L-C2ST: the B permutation classifiers are trained once, then EVALUATED
    at every z_o.

With the default k-nearest-neighbour regressor the LCT null happens to be
z_o-independent as well (see local_coverage_test), but that is a property
of that particular smoother, not of the method, and it does NOT carry over
to a random forest, an MLP, or to L-C2ST.

MULTIPLICITY. 216 real embeddings means 216 local tests. Zhao et al.
recommend controlling the false discovery rate (Benjamini-Hochberg); the
repo previously offered only family-wise control (Holm-Bonferroni, in
npe_diagnostics.holm_bonferroni). local_family_verdict reports BOTH, since
they answer different questions and disagreement between them is itself
informative.

DEVIATIONS FROM THE SOURCES, recorded deliberately:

  D-1 p-value convention. Zhao et al. Algorithm 1 and sbi's
      LC2ST.p_value() both use  p = (1/B) SUM_b 1{ T < T_b }.  This module
      uses  p = (#{b : T_b >= T} + 1) / (B + 1),  matching
      npe_diagnostics.embedding_overlap and never returning exactly zero.
      It is the Phipson-Smyth convention and is conservative relative to
      the sources by at most 1/(B+1).
  D-2 z-scoring. sbi's LC2ST defaults to z_score=False. This module
      defaults to z_score=True, because theta here has mixed physical
      scales while z lies on the unit sphere, and an unscaled MLP would be
      dominated by whichever block has the larger variance.
  D-3 LCT regressor. Zhao et al. leave the regressor open ("kernel
      smoothers to random forests to neural networks") and recommend
      choosing by validation MSE. The default here is a k-NN kernel
      smoother, chosen because it makes the null computation exact and
      essentially free (see local_coverage_test). Pass regressor_factory
      to use anything else.

NOT IMPLEMENTED HERE, on purpose:
  - L-C2ST-NF (the normalizing-flow variant). sbi ships it as
    sbi.diagnostics.lc2st.LC2ST_NF; it requires the estimator to expose an
    invertible conditional transform, so it belongs behind a capability
    check rather than in this module's unconditional path.
  - Posterior predictive checks: they test the MODEL, not the inference,
    and need the simulator. See npe_ppc.py.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from npe_diagnostics import FamilyResult, holm_bonferroni

__all__ = [
    "hpd_values",
    "single_draw_per_observation",
    "benjamini_hochberg",
    "LocalFamilyResult",
    "local_family_verdict",
    "LocalCoverageResult",
    "local_coverage_test",
    "LocalC2STResult",
    "local_c2st",
]


# ---------------------------------------------------------------------------
# Inputs shared by both diagnostics
# ---------------------------------------------------------------------------

def hpd_values(log_prob_true: np.ndarray,
               log_prob_samples: np.ndarray) -> np.ndarray:
    """HPD values, equation (H), one per calibration observation.

    Parameters
    ----------
    log_prob_true : ndarray, shape (N,)
        log q(theta*_n | z_n), the estimator's log density at the ground
        truth, for each fixed n.
    log_prob_samples : ndarray, shape (N, M)
        log q(theta^q_{n,m} | z_n) for the M posterior draws at z_n.

    Returns
    -------
    ndarray, shape (N,), values in [0, 1]
        HPD_hat_n. SMALL means the truth sits in the high-density core;
        LARGE means it sits in the tail. Uniform on (0,1) under the local
        null H0(z_n), for each fixed n (Zhao et al., Theorem 3).

    Notes
    -----
    These are exactly the complement of the ranks that
    npe_diagnostics.expected_coverage already computes:

        HPD_hat_n = 1 - r_n / M,    r_n = #{ m : log q(theta^q_{n,m} | z_n)
                                                 <  log q(theta*_n | z_n) }

    up to the treatment of exact ties, which are counted into the HPD value
    here (>=) and excluded from the rank there (<). Ties have probability
    zero for a continuous estimator; the identity is asserted in the
    validation suite (L0).
    """
    lt = np.asarray(log_prob_true, dtype=np.float64)
    ls = np.asarray(log_prob_samples, dtype=np.float64)
    if lt.ndim != 1:
        raise ValueError("log_prob_true must be (N,), got %s" % (lt.shape,))
    if ls.ndim != 2 or ls.shape[0] != lt.shape[0]:
        raise ValueError("shape mismatch: log_prob_true %s, samples %s"
                         % (lt.shape, ls.shape))
    return np.mean(ls >= lt[:, None], axis=1)


def single_draw_per_observation(posterior_samples: np.ndarray,
                                seed: int = 0) -> np.ndarray:
    """Take ONE posterior draw per calibration observation.

    L-C2ST's class-0 sample is one draw theta^q_n ~ q(theta | z_n) at each
    calibration observation z_n, not M draws: using several draws at the
    same z_n would break the assumption that the two classes share the
    marginal over z with equal counts.

    Parameters
    ----------
    posterior_samples : ndarray, shape (N, M, p)
    seed : int

    Returns
    -------
    ndarray, shape (N, p)
    """
    ps = np.asarray(posterior_samples, dtype=np.float64)
    if ps.ndim != 3:
        raise ValueError("posterior_samples must be (N, M, p), got %s"
                         % (ps.shape,))
    rng = np.random.default_rng(seed)
    n, m, _ = ps.shape
    pick = rng.integers(0, m, size=n)
    return ps[np.arange(n), pick, :]


# ---------------------------------------------------------------------------
# Multiplicity: FDR alongside the repo's existing FWER
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05,
                       names: Optional[Sequence[str]] = None,
                       dependent: bool = False) -> FamilyResult:
    """Benjamini-Hochberg step-up FDR control over a family of p-values.

    Returns the same FamilyResult dataclass as
    npe_diagnostics.holm_bonferroni, so the two are interchangeable at the
    call site.

    Parameters
    ----------
    pvalues : sequence of float
    alpha : float
        Target false discovery rate.
    names : sequence of str or None
    dependent : bool
        If True, apply the Benjamini-Yekutieli correction, multiplying the
        adjusted p-values by c(n) = SUM_{i=1..n} 1/i. This is valid under
        ARBITRARY dependence between the tests. Plain BH requires
        independence or positive regression dependence (PRDS). LOCAL TESTS
        AT NEARBY z_o ARE NOT INDEPENDENT -- they share calibration points,
        and with a k-NN regressor they may share most of them -- so
        dependent=True is the defensible choice when the evaluation points
        are dense in embedding space. Default False to match Zhao et al.,
        who recommend plain BH.

    Notes
    -----
    Interpretation differs from Holm-Bonferroni and the difference matters
    at 216 tests: Holm controls the probability of AT LEAST ONE false
    rejection (FWER), BH controls the EXPECTED FRACTION of rejections that
    are false (FDR). BH will reject strictly more often. Neither is
    "correct" in the abstract; report both.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        raise ValueError("benjamini_hochberg needs at least one p-value")
    names = list(names) if names is not None else \
        ["test_%d" % i for i in range(n)]
    if len(names) != n:
        raise ValueError("names has length %d but there are %d p-values"
                         % (len(names), n))

    c = float(np.sum(1.0 / np.arange(1, n + 1))) if dependent else 1.0
    order = np.argsort(p)                     # ascending
    ranked = p[order] * c * n / np.arange(1, n + 1)
    # step-up: enforce monotonicity from the largest p downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n, dtype=np.float64)
    adj[order] = np.minimum(ranked, 1.0)
    return FamilyResult(names=names, pvalues=p, adjusted=adj,
                        rejected=adj < alpha, alpha=alpha)


@dataclass
class LocalFamilyResult:
    """Both multiplicity corrections over the same family of local tests."""
    fwer: FamilyResult          # Holm-Bonferroni
    fdr: FamilyResult           # Benjamini-Hochberg (or -Yekutieli)
    dependent: bool

    @property
    def passes(self) -> bool:
        """True only if NEITHER correction rejects anything.

        The strict reading on purpose: this is a gate, and the cheaper
        conclusion should not be the default one.
        """
        return self.fwer.passes and self.fdr.passes

    @property
    def disagree(self) -> bool:
        """True if FDR rejects something FWER does not.

        Worth surfacing: it means the evidence is 'several weak signals'
        rather than 'one strong signal', which changes what to do next.
        """
        return int(self.fdr.n_rejected) > int(self.fwer.n_rejected)

    def summary(self) -> str:
        lines = [
            "  multiplicity over %d local tests at alpha=%.3f"
            % (len(self.fwer.names), self.fwer.alpha),
            "    FWER (Holm-Bonferroni)        : %d rejected"
            % self.fwer.n_rejected,
            "    FDR  (Benjamini-%s) : %d rejected"
            % ("Yekutieli" if self.dependent else "Hochberg ",
               self.fdr.n_rejected),
        ]
        if self.disagree:
            lines.append("    NOTE: FDR rejects more than FWER -- several weak "
                         "signals rather than one strong one")
        return "\n".join(lines)


def local_family_verdict(pvalues: Sequence[float],
                         names: Optional[Sequence[str]] = None,
                         alpha: float = 0.05,
                         dependent: bool = False) -> LocalFamilyResult:
    """Holm-Bonferroni AND Benjamini-Hochberg over the same p-values."""
    return LocalFamilyResult(
        fwer=holm_bonferroni(pvalues, alpha=alpha, names=names),
        fdr=benjamini_hochberg(pvalues, alpha=alpha, names=names,
                               dependent=dependent),
        dependent=dependent,
    )


# ---------------------------------------------------------------------------
# LCT -- Local Coverage Test
# ---------------------------------------------------------------------------

@dataclass
class LocalCoverageResult:
    """Result of the Local Coverage Test at a set of evaluation points."""
    name: str
    z_names: List[str]
    statistics: np.ndarray          # (n_eval,)  T(z_o), equation (T)
    pvalues: np.ndarray             # (n_eval,)
    alpha_grid: np.ndarray          # (G,)
    coverage_curves: np.ndarray     # (n_eval, G)  r_hat_alpha(z_o) -- ALP data
    null_statistics: np.ndarray     # (n_eval, B)
    n_null: int
    alpha: float
    regressor: str

    @property
    def rejected(self) -> np.ndarray:
        """Per-observation rejection, UNCORRECTED for multiplicity."""
        return self.pvalues < self.alpha

    @property
    def passes(self) -> bool:
        """True if no evaluation point is rejected after BOTH corrections."""
        return local_family_verdict(self.pvalues, self.z_names,
                                    alpha=self.alpha).passes

    def family(self, dependent: bool = False) -> LocalFamilyResult:
        return local_family_verdict(self.pvalues, self.z_names,
                                    alpha=self.alpha, dependent=dependent)

    def shape_at(self, index: int) -> str:
        """Name the failure mode at one evaluation point from its P-P curve.

        Reading follows Figure 1 of Zhao et al.: r_hat_alpha above the
        diagonal everywhere means too much mass at low HPD, i.e. the truth
        lands in the high-density core too often, i.e. credible regions too
        LARGE (conservative). Below the diagonal is the reverse. An S-shape
        is a dispersion error rather than a location error.
        """
        d = self.coverage_curves[index] - self.alpha_grid
        if np.all(d > 0.02):
            return "CONSERVATIVE here (credible regions too large)"
        if np.all(d < -0.02):
            return "OVERCONFIDENT here (credible regions too small)"
        if np.max(np.abs(d)) <= 0.02:
            return "no visible deviation"
        lo = float(np.mean(d[self.alpha_grid < 0.5]))
        hi = float(np.mean(d[self.alpha_grid > 0.5]))
        if lo * hi < 0:
            return "S-shaped: dispersion error (over/under-dispersed)"
        return "mixed deviation; inspect the local P-P curve"

    def summary(self, top: int = 5) -> str:
        order = np.argsort(self.pvalues)[:top]
        lines = ["  %-26s %d/%d rejected uncorrected at alpha=%.3f"
                 % (self.name, int(self.rejected.sum()),
                    self.pvalues.size, self.alpha)]
        for i in order:
            lines.append("    %-18s T=%.5f  p=%.4f  %s"
                         % (self.z_names[i], self.statistics[i],
                            self.pvalues[i], self.shape_at(int(i))))
        lines.append(self.family().summary())
        return "\n".join(lines)


def local_coverage_test(hpd: np.ndarray,
                        Z_cal: np.ndarray,
                        Z_eval: np.ndarray,
                        alpha_grid: Optional[np.ndarray] = None,
                        n_null: int = 200,
                        n_neighbors: int = 50,
                        regressor_factory: Optional[Callable[[], object]] = None,
                        alpha: float = 0.05,
                        seed: int = 0,
                        z_names: Optional[Sequence[str]] = None,
                        name: str = "LCT (HPD)") -> LocalCoverageResult:
    """Local Coverage Test of Zhao et al. (2021), equations (R) and (T).

    Parameters
    ----------
    hpd : ndarray, shape (N,)
        HPD values from hpd_values(), one per calibration observation.
    Z_cal : ndarray, shape (N, E)
        The calibration observations these HPD values belong to.
    Z_eval : ndarray, shape (n_eval, E)
        The observations to diagnose. For the real data these are the 216
        real embeddings; for validation they are held-out synthetic ones.
    alpha_grid : ndarray or None
        The grid G of levels. Default: 11 points evenly spaced on
        (0, 1) excluding the endpoints, where the indicator is degenerate.
    n_null : int
        B, the number of Monte Carlo null replicates.
    n_neighbors : int
        k for the default k-NN kernel smoother. Ignored if
        regressor_factory is given.
    regressor_factory : callable or None
        Zero-argument callable returning a fresh, unfitted sklearn-style
        regressor with .fit(X, y) and .predict(X). If None, the fast k-NN
        path below is used.
    alpha : float
        Significance level for the per-observation verdict.
    seed : int
    z_names : sequence of str or None
        Labels for the evaluation points, used in summaries.
    name : str

    Returns
    -------
    LocalCoverageResult

    Notes
    -----
    THE FAST PATH. With a k-NN smoother, r_hat_alpha(z_o) is the mean of the
    indicator over the k nearest calibration points to z_o, and that
    neighbour SET does not depend on alpha, on the replicate b, or on the
    estimator. So the neighbour search runs once and every one of the
    (B + 1) x |G| regressions collapses to an average over a fixed index
    set -- vectorised, no refitting. This turns the null from B x |G| model
    fits into one k-NN query plus array arithmetic.

    A CONSEQUENCE, which must not be generalised: with fixed k the null law
    of T(z_o) is IDENTICAL at every z_o, because it depends on the
    neighbourhood only through its size. That is a property of this
    smoother. With a random forest, an MLP, or with L-C2ST, the null is
    genuinely z_o-dependent -- sparse regions of embedding space give
    noisier fits and hence a wider null -- and the statistic must be
    evaluated against a null computed AT that z_o. The generic path below
    does exactly that.

    THE NULL IS ESTIMATOR-FREE. Under the global null the HPD values are
    i.i.d. Unif(0,1) independent of z, so a replicate is generated by
    drawing U_1, ..., U_N ~ Unif(0,1) i.i.d. and recomputing the indicators
    from them (Zhao et al., Algorithm 1, lines 9-16). No posterior is
    sampled and no model is refitted. This is what makes the LCT cheap.
    """
    hpd = np.asarray(hpd, dtype=np.float64).ravel()
    Zc = np.atleast_2d(np.asarray(Z_cal, dtype=np.float64))
    Ze = np.atleast_2d(np.asarray(Z_eval, dtype=np.float64))
    n_cal = hpd.size
    if Zc.shape[0] != n_cal:
        raise ValueError("hpd has %d entries but Z_cal has %d rows"
                         % (n_cal, Zc.shape[0]))
    if Ze.shape[1] != Zc.shape[1]:
        raise ValueError("Z_eval has %d columns, Z_cal has %d"
                         % (Ze.shape[1], Zc.shape[1]))
    if np.any(hpd < 0.0) or np.any(hpd > 1.0):
        raise ValueError("hpd values must lie in [0, 1]")
    if n_null < 1:
        raise ValueError("n_null must be >= 1")

    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 13)[1:-1]
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64).ravel()
    if np.any(alpha_grid <= 0.0) or np.any(alpha_grid >= 1.0):
        raise ValueError("alpha_grid must lie strictly inside (0, 1)")

    n_eval = Ze.shape[0]
    names = list(z_names) if z_names is not None else \
        ["z_%03d" % i for i in range(n_eval)]
    if len(names) != n_eval:
        raise ValueError("z_names has length %d but Z_eval has %d rows"
                         % (len(names), n_eval))

    rng = np.random.default_rng(seed)
    g = alpha_grid.size

    def statistic(curves: np.ndarray) -> np.ndarray:
        """curves (..., n_eval, G) -> T (..., n_eval), equation (T)."""
        return np.mean((curves - alpha_grid) ** 2, axis=-1)

    if regressor_factory is None:
        # -- fast path: k-NN kernel smoother, fixed neighbour sets --------
        from sklearn.neighbors import NearestNeighbors
        k = int(min(n_neighbors, n_cal))
        if k < 5:
            raise ValueError("n_neighbors resolves to %d; too few calibration "
                             "points for a local statement" % k)
        nn = NearestNeighbors(n_neighbors=k).fit(Zc)
        idx = nn.kneighbors(Ze, return_distance=False)      # (n_eval, k)

        # observed: W (G, N) -> curves (n_eval, G)
        W = (hpd[None, :] < alpha_grid[:, None]).astype(np.float64)
        curves = W[:, idx].mean(axis=2).T                   # (n_eval, G)
        stat = statistic(curves)

        # null: U (N,) per replicate, same neighbour sets
        null = np.empty((n_eval, n_null), dtype=np.float64)
        for b in range(n_null):
            u = rng.random(n_cal)
            Wb = (u[None, :] < alpha_grid[:, None]).astype(np.float64)
            null[:, b] = statistic(Wb[:, idx].mean(axis=2).T)
        reg_name = "knn(k=%d)" % k
    else:
        # -- generic path: refit the regressor per level and per replicate -
        curves = np.empty((n_eval, g), dtype=np.float64)
        for j, a in enumerate(alpha_grid):
            r = regressor_factory()
            r.fit(Zc, (hpd < a).astype(np.float64))
            curves[:, j] = np.clip(np.asarray(r.predict(Ze)).ravel(), 0.0, 1.0)
        stat = statistic(curves)

        null = np.empty((n_eval, n_null), dtype=np.float64)
        for b in range(n_null):
            u = rng.random(n_cal)
            cb = np.empty((n_eval, g), dtype=np.float64)
            for j, a in enumerate(alpha_grid):
                r = regressor_factory()
                r.fit(Zc, (u < a).astype(np.float64))
                cb[:, j] = np.clip(np.asarray(r.predict(Ze)).ravel(), 0.0, 1.0)
            null[:, b] = statistic(cb)
        reg_name = type(regressor_factory()).__name__

    # Deviation D-1: (#{b : T_b >= T} + 1) / (B + 1), not (1/B) SUM 1{T < T_b}
    pvals = (np.sum(null >= stat[:, None], axis=1) + 1.0) / (n_null + 1.0)

    return LocalCoverageResult(
        name=name, z_names=names, statistics=stat, pvalues=pvals,
        alpha_grid=alpha_grid, coverage_curves=curves, null_statistics=null,
        n_null=int(n_null), alpha=float(alpha), regressor=reg_name,
    )


# ---------------------------------------------------------------------------
# L-C2ST -- wrapper over the authors' implementation in sbi
# ---------------------------------------------------------------------------

@dataclass
class LocalC2STResult:
    """Result of L-C2ST at a set of evaluation points."""
    name: str
    z_names: List[str]
    statistics: np.ndarray              # (n_eval,)  t_hat(z_o), equation (C)
    pvalues: np.ndarray                 # (n_eval,)
    null_statistics: np.ndarray         # (n_eval, B)
    probabilities: Optional[np.ndarray]  # (n_eval, M) per-draw class-0 probs
    n_null: int
    alpha: float
    classifier: str

    @property
    def rejected(self) -> np.ndarray:
        return self.pvalues < self.alpha

    @property
    def passes(self) -> bool:
        return local_family_verdict(self.pvalues, self.z_names,
                                    alpha=self.alpha).passes

    def family(self, dependent: bool = False) -> LocalFamilyResult:
        return local_family_verdict(self.pvalues, self.z_names,
                                    alpha=self.alpha, dependent=dependent)

    def summary(self, top: int = 5) -> str:
        order = np.argsort(self.pvalues)[:top]
        lines = ["  %-26s %d/%d rejected uncorrected at alpha=%.3f  [%s]"
                 % (self.name, int(self.rejected.sum()), self.pvalues.size,
                    self.alpha, self.classifier)]
        for i in order:
            lines.append("    %-18s t=%.5f  p=%.4f"
                         % (self.z_names[i], self.statistics[i],
                            self.pvalues[i]))
        lines.append(self.family().summary())
        return "\n".join(lines)


def _require_lc2st():
    """Import sbi's LC2ST, with an actionable message if it is absent.

    Kept at the boundary on purpose: everything above this line is numpy +
    scikit-learn and is unit-testable without torch installed, which is the
    same pattern npe_model.py uses for the estimator.
    """
    try:
        import torch  # noqa: F401
        from sbi.diagnostics.lc2st import LC2ST
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "L-C2ST needs sbi (pinned: sbi==0.27.0) and torch. The LCT in "
            "this module does not. Original error: %s" % (exc,)) from exc
    return LC2ST


def local_c2st(theta_cal: np.ndarray,
               Z_cal: np.ndarray,
               theta_q: np.ndarray,
               Z_eval: np.ndarray,
               theta_o: np.ndarray,
               n_null: int = 100,
               classifier: str = "mlp",
               classifier_kwargs: Optional[dict] = None,
               num_ensemble: int = 1,
               z_score: bool = True,
               alpha: float = 0.05,
               seed: int = 1,
               z_names: Optional[Sequence[str]] = None,
               return_probabilities: bool = True,
               verbosity: int = 0,
               name: str = "L-C2ST") -> LocalC2STResult:
    """L-C2ST at each row of Z_eval, wrapping sbi.diagnostics.lc2st.LC2ST.

    Parameters
    ----------
    theta_cal : ndarray, shape (N, p)
        Ground truths theta*_n. Together with Z_cal these are draws from
        the TRUE joint p(theta, z), which is the whole trick: the true
        posterior cannot be sampled but the true joint can.
        NOTE ON sbi's NAMING: this is passed as `prior_samples`, which is
        misleading -- it must be the theta that GENERATED the matching row
        of Z_cal, not an independent prior draw.
    Z_cal : ndarray, shape (N, E)
        Calibration observations (sbi's `xs`).
    theta_q : ndarray, shape (N, p)
        ONE draw theta^q_n ~ q(theta | z_n) per calibration observation
        (sbi's `posterior_samples`). Use single_draw_per_observation().
    Z_eval : ndarray, shape (n_eval, E)
        Observations to diagnose.
    theta_o : ndarray, shape (n_eval, M, p)
        Draws from q(theta | z_o) at each evaluation point, for each fixed
        z_o. These are what equation (C) averages over.
    n_null : int
        B, the number of permutation replicates.
    classifier : {"mlp", "random_forest"} or an sklearn classifier CLASS
    classifier_kwargs : dict or None
    num_ensemble : int
        Averaging several classifiers reduces variance from the classifier
        itself, which is the free parameter with no principled setting.
    z_score : bool
        See deviation D-2 in the module docstring. Default True here.
    alpha, seed, z_names, verbosity, name : as for local_coverage_test.
    return_probabilities : bool
        If True, also return the per-draw class-0 probabilities
        d_omega(theta^q_{o,m}, z_o). THIS IS THE POINT OF L-C2ST over a
        P-P plot: it turns a verdict into a map of WHERE in parameter space
        the estimated posterior is wrong.

    Returns
    -------
    LocalC2STResult

    Notes
    -----
    CLASS LABELS. sbi assigns class 0 to `posterior_samples` (the
    ESTIMATOR) and class 1 to `prior_samples` (the true joint) -- the
    opposite of equation (1) in the project handoff. The statistic (C) is
    symmetric so this does not change any p-value, but it inverts the
    reading of `probabilities`: a value ABOVE 1/2 at (theta, z_o) means the
    classifier thinks a point there is more likely to have come from the
    estimator than from the true joint, i.e. THE ESTIMATOR PUTS TOO MUCH
    MASS THERE. Below 1/2 means too little.

    WHAT IS AMORTIZED. train_under_null_hypothesis() trains B permutation
    classifiers ONCE; they are then evaluated at every z_o. The null law of
    t_hat(z_o) is itself z_o-dependent, so the B classifiers must be
    re-evaluated at each new observation -- what is reused is the training,
    not the quantiles. Validated in the suite as L7.

    PERMUTATION SCHEME. sbi's permute_data concatenates the two classes'
    JOINT rows [(theta*, z) ; (theta^q, z)] and permutes the rows, i.e. it
    permutes the labels while keeping each (theta, z) pair intact. Each z_n
    therefore appears twice in the pool and is randomly assigned to either
    class, so the two classes' z-marginals differ slightly within a
    replicate. That is the reference behaviour and is kept.
    """
    LC2ST = _require_lc2st()
    import torch

    tc = np.atleast_2d(np.asarray(theta_cal, dtype=np.float64))
    zc = np.atleast_2d(np.asarray(Z_cal, dtype=np.float64))
    tq = np.atleast_2d(np.asarray(theta_q, dtype=np.float64))
    ze = np.atleast_2d(np.asarray(Z_eval, dtype=np.float64))
    to = np.asarray(theta_o, dtype=np.float64)
    if to.ndim != 3:
        raise ValueError("theta_o must be (n_eval, M, p), got %s" % (to.shape,))
    if not (tc.shape == tq.shape):
        raise ValueError("theta_cal %s and theta_q %s must have the same shape"
                         % (tc.shape, tq.shape))
    if zc.shape[0] != tc.shape[0]:
        raise ValueError("Z_cal has %d rows, theta_cal has %d"
                         % (zc.shape[0], tc.shape[0]))
    if ze.shape[0] != to.shape[0]:
        raise ValueError("Z_eval has %d rows, theta_o has %d"
                         % (ze.shape[0], to.shape[0]))
    if to.shape[2] != tc.shape[1]:
        raise ValueError("theta_o has parameter dimension %d, theta_cal has %d"
                         % (to.shape[2], tc.shape[1]))

    n_eval = ze.shape[0]
    names = list(z_names) if z_names is not None else \
        ["z_%03d" % i for i in range(n_eval)]
    if len(names) != n_eval:
        raise ValueError("z_names has length %d but Z_eval has %d rows"
                         % (len(names), n_eval))

    def t32(a: np.ndarray):
        return torch.as_tensor(a, dtype=torch.float32)

    test = LC2ST(
        prior_samples=t32(tc),          # class 1: the TRUE joint
        xs=t32(zc),
        posterior_samples=t32(tq),      # class 0: the ESTIMATOR
        seed=seed,
        num_ensemble=num_ensemble,
        classifier=classifier,
        z_score=z_score,
        classifier_kwargs=classifier_kwargs,
        num_trials_null=int(n_null),
        permutation=True,
    )
    test.train_on_observed_data(verbosity=verbosity)
    test.train_under_null_hypothesis(verbosity=verbosity)

    stat = np.empty(n_eval, dtype=np.float64)
    null = np.empty((n_eval, int(n_null)), dtype=np.float64)
    probs = np.empty((n_eval, to.shape[1]), dtype=np.float64) \
        if return_probabilities else None

    for i in range(n_eval):
        th_o = t32(to[i])
        z_o = t32(ze[i]).reshape(1, -1)
        if return_probabilities:
            res = test.get_scores(theta_o=th_o, x_o=z_o,
                                  trained_clfs=test.trained_clfs)
            stat[i] = float(np.asarray(res.scores).mean())
            probs[i] = np.asarray(res.probabilities).reshape(-1, to.shape[1])\
                         .mean(axis=0)
        else:
            stat[i] = float(test.get_statistic_on_observed_data(
                theta_o=th_o, x_o=z_o))
        res_null = test.get_statistics_under_null_hypothesis(
            theta_o=th_o, x_o=z_o, verbosity=0)
        s_null = np.asarray(res_null.scores, dtype=np.float64).ravel()
        # In sbi 0.27.0 this is exactly one number per null trial (the mean
        # over folds/ensemble is taken inside). Assert rather than silently
        # truncating: a future version that returns one entry per fold would
        # otherwise corrupt every p-value without any visible symptom.
        if s_null.size != n_null:
            raise RuntimeError(
                "expected %d null statistics, got %d -- sbi's null score "
                "aggregation has changed; check "
                "get_statistics_under_null_hypothesis before trusting any "
                "p-value from this module" % (n_null, s_null.size))
        null[i] = s_null

    # Deviation D-1, as for the LCT.
    pvals = (np.sum(null >= stat[:, None], axis=1) + 1.0) / (n_null + 1.0)

    clf_name = classifier if isinstance(classifier, str) \
        else getattr(classifier, "__name__", str(classifier))
    return LocalC2STResult(
        name=name, z_names=names, statistics=stat, pvalues=pvals,
        null_statistics=null, probabilities=probs, n_null=int(n_null),
        alpha=float(alpha),
        classifier="%s%s" % (clf_name,
                             "" if num_ensemble == 1
                             else " x%d" % num_ensemble),
    )
