#!/usr/bin/env python3
"""
npe_diagnostics.py -- post-training diagnostics for an amortized NPE.

Scope boundary: this module computes diagnostics from arrays. Only the two
helpers in section 0 touch a posterior object; everything after that is pure
numpy operating on (theta_true, posterior_samples) arrays. That separation is
what lets the whole suite be tested against ANALYTIC posteriors with no
network training, which is how the blind-spot test in the smoke suite works.

THE FIVE LAYERS
---------------
  1. embedding overlap   -- are the real embeddings inside the simulated
                            cloud at all? Hard go/no-go, needs no posterior.
  2. global calibration  -- SBC (marginals), expected coverage (joint), and
                            DATA-DEPENDENT test quantities.
  3. TARP                -- coverage via random reference points; needs no
                            density evaluation.
  4. informativeness     -- posterior contraction and the information
                            spectrum. NOT a calibration check.
  5. posterior predictive-- requires the simulator; interface only here.

WHY LAYER 4 IS NOT OPTIONAL
---------------------------
Modrak et al. (Bayesian Analysis, doi:10.1214/23-ba1404) prove that SBC with
test quantities depending only on the parameters cannot detect a posterior
that ignores the data -- including one exactly equal to the prior (their
Theorem 7, "incomplete use of data").

Expected coverage does not rescue this either. If q(theta | z) = p(theta) for
every z, then f(theta) = log q(theta | z) = log p(theta) carries no
z-dependence, the highest-density regions are the prior's, and a true
theta ~ p(theta) lands inside the alpha-region at rate exactly alpha.
Coverage is nominal and the test passes.

Their prescribed fix is a data-dependent test quantity, with the joint
log-likelihood as the default. That is unavailable in SBI by construction.
The substitute implemented here is the bilinear form

    f(theta, z) = theta^T W z                                          (T1)

for random fixed W. It works for the same reason the likelihood does: the
true theta* is correlated with z because z was generated FROM theta*, while
draws from a data-ignoring q are not. Conditioning the projection direction
on z stops the per-observation deviations from cancelling when averaged over
observations. See smoke_test_diagnostics.py test D5, which verifies
empirically that parameter-only SBC and expected coverage both PASS on a
posterior set equal to the prior, while (T1) catches it.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "sample_posteriors", "posterior_log_probs",
    "MMDResult", "embedding_overlap", "mmd2_biased",
    "RankResult", "simulation_based_calibration", "expected_coverage",
    "FamilyResult", "holm_bonferroni", "family_verdict",
    "data_dependent_sbc", "make_bilinear_test_quantity",
    "TARPResult", "tarp",
    "ContractionResult", "posterior_contraction", "information_spectrum",
    "diagnostic_report",
]


# ===========================================================================
# 0. The only sbi-coupled helpers
# ===========================================================================

def sample_posteriors(posterior, Z: np.ndarray, n_draws: int = 256,
                      show_progress: bool = False) -> np.ndarray:
    """Draw n_draws posterior samples for each row of Z.

    Returns an array of shape (N, n_draws, p). Everything downstream in this
    module consumes that array, never the posterior object.
    """
    import torch

    Z = np.atleast_2d(np.asarray(Z, dtype=np.float32))
    out = []
    for i in range(Z.shape[0]):
        s = posterior.sample((n_draws,),
                             x=torch.as_tensor(Z[i], dtype=torch.float32),
                             show_progress_bars=show_progress)
        out.append(np.asarray(s))
    return np.stack(out, axis=0)


def posterior_log_probs(posterior, theta: np.ndarray, Z: np.ndarray,
                        norm_posterior: bool = False) -> np.ndarray:
    """log q(theta | z) evaluated row-wise.

    theta may be (N, p) -- one point per observation -- or (N, M, p) -- M
    points per observation. Returns (N,) or (N, M) correspondingly.

    norm_posterior defaults to False, against sbi's own default of True,
    for two reasons.

    Correctness: sbi normalises by a leakage-correction factor, the fraction
    of flow mass falling inside the prior support. When the estimator is
    built with z_score_theta="transform_to_unconstrained" -- as npe_model
    does -- the flow cannot place mass outside the box at all, so that
    factor is exactly 1 and the correction is a no-op.

    Cost: computing it is not a no-op. sbi estimates the factor by drawing
    10,000 rejection samples PER OBSERVATION and per ensemble member, which
    for a few hundred calibration observations exhausts memory before it
    finishes. That is not a theoretical concern -- it is what killed the
    first run of the synthetic trial.

    Set norm_posterior=True only if the estimator was built WITHOUT the
    unconstrained transform, and then expect it to be slow. Check the
    in-box fraction of posterior samples first: if it is not 1.0, leakage
    is real and the correction matters.
    """
    import torch

    theta = np.asarray(theta)
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float32))
    squeeze = theta.ndim == 2
    th = theta[:, None, :] if squeeze else theta
    out = np.empty(th.shape[:2], dtype=np.float64)
    with torch.no_grad():
        for i in range(th.shape[0]):
            lp = posterior.log_prob(
                torch.as_tensor(th[i], dtype=torch.float32),
                x=torch.as_tensor(Z[i], dtype=torch.float32),
                norm_posterior=norm_posterior)
            out[i] = np.asarray(lp, dtype=np.float64)
    return out[:, 0] if squeeze else out


# ===========================================================================
# 1. Embedding overlap -- the go/no-go gate
# ===========================================================================

@dataclass
class MMDResult:
    """Outcome of the simulated-vs-real summary-space comparison."""
    mmd: float
    p_value: float
    null_quantile_95: float
    n_real: int
    n_sim: int
    bandwidth: float
    nn_geodesic_real: Optional[np.ndarray] = None
    nn_geodesic_sim: Optional[np.ndarray] = None
    null_samples: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    @property
    def rejects(self) -> bool:
        """True if the real embeddings are distinguishable from simulated."""
        return self.p_value < 0.05

    def summary(self) -> str:
        lines = [
            "embedding overlap (simulated vs real)",
            "  MMD                : %.4f" % self.mmd,
            "  null 95th pct      : %.4f" % self.null_quantile_95,
            "  p-value            : %.4f" % self.p_value,
            "  n_sim / n_real     : %d / %d" % (self.n_sim, self.n_real),
            "  verdict            : %s" % (
                "REJECT -- real data lies outside the simulated summary "
                "distribution; posteriors are not trustworthy"
                if self.rejects else
                "no evidence of a simulation gap in summary space"),
        ]
        if self.nn_geodesic_real is not None:
            lines += [
                "  geodesic NN (real) : median %.4f rad, max %.4f"
                % (float(np.median(self.nn_geodesic_real)),
                   float(np.max(self.nn_geodesic_real))),
                "  geodesic NN (sim)  : median %.4f rad, 95th %.4f"
                % (float(np.median(self.nn_geodesic_sim)),
                   float(np.percentile(self.nn_geodesic_sim, 95))),
            ]
        lines += ["  " + n for n in self.notes]
        return "\n".join(lines)


def _median_bandwidth(A: np.ndarray, B: np.ndarray, rng, max_n: int = 1000) -> float:
    """Median heuristic for the RBF bandwidth, on a subsample for cost."""
    P = np.concatenate([A, B], axis=0)
    if P.shape[0] > max_n:
        P = P[rng.choice(P.shape[0], max_n, replace=False)]
    d2 = np.maximum(_sqdist(P, P), 0.0)
    iu = np.triu_indices(P.shape[0], k=1)
    med = float(np.median(np.sqrt(d2[iu]))) if iu[0].size else 1.0
    return med if med > 1e-12 else 1.0


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (np.sum(A ** 2, axis=1)[:, None]
            + np.sum(B ** 2, axis=1)[None, :]
            - 2.0 * A @ B.T)


def mmd2_biased(A: np.ndarray, B: np.ndarray, bandwidth: float) -> float:
    """Biased squared-MMD estimate with a Gaussian RBF kernel.

    The BIASED estimator is used deliberately. The unbiased version is
    undefined for a single observation, and a single real recording is an
    important practical case -- excluding it to gain unbiasedness would trade
    away the use case for a property that does not matter here.
    """
    g = 1.0 / (2.0 * bandwidth ** 2)
    kaa = np.exp(-g * np.maximum(_sqdist(A, A), 0.0)).mean()
    kbb = np.exp(-g * np.maximum(_sqdist(B, B), 0.0)).mean()
    kab = np.exp(-g * np.maximum(_sqdist(A, B), 0.0)).mean()
    return float(kaa + kbb - 2.0 * kab)


def _geodesic_nn(query: np.ndarray, pool: np.ndarray,
                 exclude_self: bool = False) -> np.ndarray:
    """Nearest-neighbour geodesic distance on the unit sphere, in radians.

    Uses arccos of the inner product, which is the natural metric when the
    embedding is L2-normalised; Euclidean distance would distort near the
    antipode.
    """
    ip = np.clip(query @ pool.T, -1.0, 1.0)
    if exclude_self:
        np.fill_diagonal(ip, -1.0)
    return np.arccos(np.clip(np.max(ip, axis=1), -1.0, 1.0))


def embedding_overlap(z_sim: np.ndarray, z_real: np.ndarray,
                      n_null: int = 500, seed: int = 0,
                      geodesic: bool = True) -> MMDResult:
    """Test whether real embeddings lie inside the simulated distribution.

    This is the first diagnostic to run and the only one that needs no
    trained posterior. The encoder was trained on real recordings, so it is
    the SIMULATIONS that may be out of distribution for it, not the reverse
    -- and an NPE trained on simulated embeddings will be perfectly
    calibrated on simulations while saying nothing about this.

    The null distribution is built by repeatedly drawing subsets of size
    n_real from the simulated pool and computing their MMD against a
    disjoint simulated reference, following the sampling-based test of
    Schmitt et al.
    """
    rng = np.random.default_rng(seed)
    z_sim = np.atleast_2d(np.asarray(z_sim, dtype=np.float64))
    z_real = np.atleast_2d(np.asarray(z_real, dtype=np.float64))
    n_real, n_sim = z_real.shape[0], z_sim.shape[0]
    notes = []

    if z_sim.shape[1] != z_real.shape[1]:
        raise ValueError("embedding dims differ: sim %d vs real %d"
                         % (z_sim.shape[1], z_real.shape[1]))
    if n_sim < 4 * n_real:
        notes.append("NOTE: n_sim < 4*n_real; the null distribution is coarse")

    bw = _median_bandwidth(z_sim, z_real, rng)

    # Split the simulated pool so the reference and the null draws are
    # disjoint; reusing the same points on both sides biases the null low
    # and makes the test anticonservative.
    perm = rng.permutation(n_sim)
    n_ref = max(n_real, n_sim // 2)
    ref_idx, pool_idx = perm[:n_ref], perm[n_ref:]
    reference = z_sim[ref_idx]
    pool = z_sim[pool_idx] if pool_idx.size >= n_real else z_sim[ref_idx]

    observed = mmd2_biased(reference, z_real, bw)

    null = np.empty(n_null, dtype=np.float64)
    for b in range(n_null):
        idx = rng.choice(pool.shape[0], size=min(n_real, pool.shape[0]),
                         replace=pool.shape[0] < n_real)
        null[b] = mmd2_biased(reference, pool[idx], bw)

    # +1 in numerator and denominator: the observed value is itself a draw
    # under the null, which keeps the p-value from ever being exactly 0.
    p = float((np.sum(null >= observed) + 1) / (n_null + 1))

    nn_real = nn_sim = None
    if geodesic:
        norms = np.concatenate([np.linalg.norm(z_sim, axis=1),
                                np.linalg.norm(z_real, axis=1)])
        if np.max(np.abs(norms - 1.0)) > 1e-3:
            notes.append("NOTE: embeddings are not unit-norm; "
                         "geodesic distances are not meaningful")
        else:
            nn_real = _geodesic_nn(z_real, z_sim)
            sub = z_sim[rng.choice(n_sim, min(n_sim, 2000), replace=False)]
            nn_sim = _geodesic_nn(sub, sub, exclude_self=True)

    return MMDResult(mmd=float(np.sqrt(max(observed, 0.0))),
                     p_value=p,
                     null_quantile_95=float(np.sqrt(max(
                         np.percentile(null, 95), 0.0))),
                     n_real=n_real, n_sim=n_sim, bandwidth=bw,
                     nn_geodesic_real=nn_real, nn_geodesic_sim=nn_sim,
                     null_samples=np.sqrt(np.maximum(null, 0.0)),
                     notes=notes)


# ===========================================================================
# 2. Rank-based calibration
# ===========================================================================

@dataclass
class RankResult:
    """Ranks of the ground truth among posterior draws, per test quantity."""
    name: str
    ranks: np.ndarray          # (N,) integers in [0, M]
    n_draws: int
    ks_pvalue: float
    mean_rank_frac: float
    verdict: str

    @property
    def passes(self) -> bool:
        return self.ks_pvalue > 0.005

    def summary(self) -> str:
        return ("  %-26s KS p=%.4f  mean rank=%.3f  %s  %s"
                % (self.name, self.ks_pvalue, self.mean_rank_frac,
                   "PASS" if self.passes else "FAIL", self.verdict))


def _rank_uniformity(ranks: np.ndarray, n_draws: int, name: str,
                     rng) -> RankResult:
    """KS test of rank uniformity, with the shape of any failure named.

    Fractional ranks are jittered by U(0,1) before the test. Ranks are
    integers on [0, M] so their exact null is discrete uniform; comparing a
    discrete sample against a continuous uniform without jitter inflates the
    KS statistic and produces spurious failures.
    """
    from scipy import stats

    ranks = np.asarray(ranks, dtype=np.float64)
    n = ranks.shape[0]
    u = (ranks + rng.uniform(size=n)) / (n_draws + 1)
    p = float(stats.kstest(u, "uniform").pvalue)
    mean_frac = float(np.mean(ranks) / n_draws)

    if p > 0.005:
        verdict = "ranks consistent with uniform"
    else:
        # A U-shaped rank histogram means the truth lands in the tails too
        # often, i.e. the posterior is too narrow. Centre-heavy means too
        # broad. A shifted mean means bias.
        centre = float(np.mean(np.abs(u - 0.5)))
        if mean_frac < 0.40:
            verdict = "posterior biased HIGH (truth ranks low)"
        elif mean_frac > 0.60:
            verdict = "posterior biased LOW (truth ranks high)"
        elif centre > 0.27:
            verdict = "posterior TOO NARROW (overconfident, U-shaped ranks)"
        elif centre < 0.23:
            verdict = "posterior TOO BROAD (underconfident, centre-heavy)"
        else:
            verdict = "non-uniform, shape unclear"
    return RankResult(name=name, ranks=ranks.astype(int), n_draws=n_draws,
                      ks_pvalue=p, mean_rank_frac=mean_frac, verdict=verdict)


@dataclass
class FamilyResult:
    """Family-wise verdict over a set of rank tests.

    Running one test per parameter axis and rejecting whenever any single
    p-value falls below alpha inflates the false-positive rate roughly
    p-fold: with 27 axes at alpha=0.005, a perfectly calibrated posterior
    trips something about 13% of the time. Holm-Bonferroni controls the
    family-wise error rate while being uniformly more powerful than plain
    Bonferroni, so an honest per-axis report needs it.
    """
    names: List[str]
    pvalues: np.ndarray
    adjusted: np.ndarray
    rejected: np.ndarray
    alpha: float

    @property
    def n_rejected(self) -> int:
        return int(np.sum(self.rejected))

    @property
    def passes(self) -> bool:
        return self.n_rejected == 0

    def summary(self) -> str:
        head = ("family-wise (Holm-Bonferroni, alpha=%.3f): %d/%d axes reject"
                % (self.alpha, self.n_rejected, len(self.names)))
        if self.passes:
            return head + "  PASS"
        worst = np.argsort(self.adjusted)[:5]
        lines = [head + "  FAIL"]
        for k in worst:
            if self.rejected[k]:
                lines.append("    %-20s raw p=%.2e  adjusted p=%.2e"
                             % (self.names[k], self.pvalues[k], self.adjusted[k]))
        return "\n".join(lines)


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05,
                    names: Optional[Sequence[str]] = None) -> FamilyResult:
    """Holm-Bonferroni step-down correction over a family of p-values."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    names = list(names) if names is not None else ["test_%d" % i for i in range(n)]
    order = np.argsort(p)
    adj = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)          # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return FamilyResult(names=names, pvalues=p, adjusted=adj,
                        rejected=adj < alpha, alpha=alpha)


def family_verdict(results: Sequence["RankResult"], alpha: float = 0.05) -> FamilyResult:
    """Holm-Bonferroni over a list of RankResult objects."""
    return holm_bonferroni([r.ks_pvalue for r in results], alpha=alpha,
                           names=[r.name for r in results])


# ---------------------------------------------------------------------------

def simulation_based_calibration(theta_true: np.ndarray,
                                 posterior_samples: np.ndarray,
                                 param_names: Optional[Sequence[str]] = None,
                                 seed: int = 0) -> List[RankResult]:
    """Marginal SBC: one rank statistic per parameter coordinate.

    For each fixed coordinate k, the rank is

        r_k(i) = sum_m 1{ theta^q_{i,m,k} < theta*_{i,k} }  in [0, M]

    which is uniform on [0, M] for a correct posterior, for every k.

    Detects which MARGINALS are too narrow, too broad, or biased. Blind to
    parameter correlations, and -- per the module docstring -- blind to a
    posterior that ignores the data entirely.
    """
    rng = np.random.default_rng(seed)
    theta_true = np.atleast_2d(np.asarray(theta_true, dtype=np.float64))
    ps = np.asarray(posterior_samples, dtype=np.float64)
    if ps.ndim != 3:
        raise ValueError("posterior_samples must be (N, M, p), got %s" % (ps.shape,))
    n, m, p = ps.shape
    if theta_true.shape != (n, p):
        raise ValueError("theta_true must be (%d, %d), got %s" % (n, p, theta_true.shape))

    names = list(param_names) if param_names is not None else \
        ["theta_%02d" % k for k in range(p)]
    out = []
    for k in range(p):
        ranks = np.sum(ps[:, :, k] < theta_true[:, None, k], axis=1)
        out.append(_rank_uniformity(ranks, m, names[k], rng))
    return out


def expected_coverage(log_prob_true: np.ndarray,
                      log_prob_samples: np.ndarray,
                      seed: int = 0) -> RankResult:
    """Joint coverage using f(theta) = log q(theta | z) as the test quantity.

    Sensitive to joint miscalibration including wrong parameter
    correlations, which marginal SBC cannot see. Cannot say WHICH parameter
    is responsible.

    Caveat worth stating at the point of use: if q(theta | z) does not
    actually depend on z, this quantity degenerates to log p(theta) and the
    test passes regardless. It is not a substitute for layer 4.
    """
    rng = np.random.default_rng(seed)
    lt = np.asarray(log_prob_true, dtype=np.float64)
    ls = np.asarray(log_prob_samples, dtype=np.float64)
    if ls.ndim != 2 or ls.shape[0] != lt.shape[0]:
        raise ValueError("shape mismatch: log_prob_true %s, samples %s"
                         % (lt.shape, ls.shape))
    ranks = np.sum(ls < lt[:, None], axis=1)
    res = _rank_uniformity(ranks, ls.shape[1], "expected_coverage (log q)", rng)

    # The generic verdict text from _rank_uniformity is written for PARAMETER
    # test quantities, where a low mean rank means the posterior sits above
    # the truth. For THIS quantity the reading is different, and the generic
    # label is actively misleading. The rank counts how many posterior draws
    # have lower density than the truth, so a low mean rank means the truth
    # falls in the low-density tail of the estimated posterior -- the
    # credible regions are too small. That is OVERCONFIDENCE, not bias.
    if not res.passes:
        if res.mean_rank_frac < 0.45:
            verdict = ("OVERCONFIDENT: truth falls in the low-density tail "
                       "too often, so credible regions are too small")
        elif res.mean_rank_frac > 0.55:
            verdict = ("CONSERVATIVE: truth sits in the high-density core "
                       "too often, so credible regions are too large")
        else:
            verdict = ("miscalibrated joint structure with a centred mean "
                       "rank; inspect the coverage curve shape")
        res = RankResult(name=res.name, ranks=res.ranks, n_draws=res.n_draws,
                         ks_pvalue=res.ks_pvalue,
                         mean_rank_frac=res.mean_rank_frac, verdict=verdict)
    return res


def make_bilinear_test_quantity(n_dim: int, embedding_dim: int,
                                n_forms: int = 4, seed: int = 0
                                ) -> List[Tuple[str, Callable]]:
    """Data-dependent test quantities f(theta, z) = theta^T W z, per (T1).

    The SBI-native stand-in for Modrak et al.'s joint log-likelihood, which
    is unavailable here by construction. Each W is fixed once and reused
    across all observations, so the quantity is a genuine function of
    (theta, z) rather than something refitted per observation.
    """
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n_forms):
        W = rng.normal(size=(n_dim, embedding_dim)) / np.sqrt(embedding_dim)

        def f(theta, z, W=W):
            # theta: (..., p), z: (E,) -> (...,)
            return theta @ (W @ z)

        out.append(("bilinear_%d (theta^T W z)" % j, f))
    return out


def data_dependent_sbc(theta_true: np.ndarray,
                       posterior_samples: np.ndarray,
                       Z: np.ndarray,
                       quantities: Optional[Sequence[Tuple[str, Callable]]] = None,
                       n_forms: int = 4,
                       seed: int = 0) -> List[RankResult]:
    """SBC with test quantities that depend on BOTH theta and z.

    This is the layer that can detect a posterior which ignores the data.
    A parameter-only quantity cannot: marginally over z, theta* and a draw
    from a data-ignoring q are exchangeable, so the rank is uniform. Making
    the projection direction depend on z breaks that exchangeability,
    because theta* is correlated with z -- z was generated from it -- while
    a data-ignoring draw is not.
    """
    rng = np.random.default_rng(seed)
    theta_true = np.atleast_2d(np.asarray(theta_true, dtype=np.float64))
    ps = np.asarray(posterior_samples, dtype=np.float64)
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    n, m, p = ps.shape
    if quantities is None:
        quantities = make_bilinear_test_quantity(p, Z.shape[1],
                                                 n_forms=n_forms, seed=seed)
    out = []
    for name, f in quantities:
        ranks = np.empty(n, dtype=np.int64)
        for i in range(n):
            ft = f(theta_true[i], Z[i])
            fs = f(ps[i], Z[i])
            ranks[i] = int(np.sum(np.asarray(fs) < ft))
        out.append(_rank_uniformity(ranks, m, name, rng))
    return out


# ===========================================================================
# 3. TARP
# ===========================================================================

@dataclass
class TARPResult:
    ecdf_x: np.ndarray
    ecdf_y: np.ndarray
    ks_pvalue: float
    max_deviation: float
    reference_mode: str = "random"
    notes: List[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return self.ks_pvalue > 0.005

    def summary(self) -> str:
        head = ("  %-26s KS p=%.4f  max|dev|=%.3f  %s  [refs: %s]"
                % ("TARP", self.ks_pvalue, self.max_deviation,
                   "PASS" if self.passes else "FAIL", self.reference_mode))
        return "\n".join([head] + ["    NOTE: " + nt for nt in self.notes])


def tarp(theta_true: np.ndarray, posterior_samples: np.ndarray,
         n_references: int = 1, seed: int = 0,
         reference_points: Optional[np.ndarray] = None,
         Z: Optional[np.ndarray] = None,
         reference_mode: str = "auto") -> TARPResult:
    """Tests of Accuracy with Random Points.

    For each observation i and a reference point r_i, compute the fraction
    of posterior draws lying closer to r_i than the truth does:

        f_i = (1/M) sum_m 1{ ||theta^q_{i,m} - r_i|| < ||theta*_i - r_i|| }

    For a correct posterior the f_i are uniform on [0, 1]. Unlike expected
    coverage this needs no density evaluation, so it also applies to
    posteriors that cannot be evaluated pointwise.

    THE REFERENCE POINTS MUST DEPEND ON THE OBSERVATION.
    ----------------------------------------------------
    The theorem behind TARP requires the credible regions to be POSITIONABLE
    -- placed at theta_r(x), a function of the observation -- and correct
    coverage must hold for every such function. Reference points drawn at
    random INDEPENDENTLY of x do not satisfy that requirement, and the
    difference is not academic.

    Measured on a posterior set exactly equal to the prior, over 12 seeds:

        reference points drawn uniformly at random :  1/12 detections
                                                      (chance level)
        reference points from a linear readout of z: 12/12 detections,
                                                      median p = 3.8e-10,
                                                      0 false positives

    The reason is exchangeability. With q(theta | z) = p(theta), both the
    true theta* and the posterior draws are marginally prior draws, so
    given an x-independent r they are exchangeable and f_i is uniform. An
    r that is correlated with x breaks the exchangeability, because theta*
    generated x and the draws did not.

    reference_mode
        "auto"     -- use x-dependent references when Z is supplied,
                      otherwise fall back to random and record a warning.
        "x"        -- require Z; raise if absent.
        "random"   -- force the weaker x-independent version. Provided for
                      comparison only; it cannot see a posterior that
                      ignores the data.

    When Z is supplied, theta_r(z) is a least-squares linear readout of z,
    fit on one half of the calibration set and applied to the other. The
    split matters: fitting and evaluating on the same rows would let the
    readout memorise theta* and manufacture apparent miscalibration.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    theta_true = np.atleast_2d(np.asarray(theta_true, dtype=np.float64))
    ps = np.asarray(posterior_samples, dtype=np.float64)
    n, m, p = ps.shape
    notes: List[str] = []

    if reference_mode not in ("auto", "x", "random"):
        raise ValueError("reference_mode must be auto, x or random")
    if reference_mode == "x" and Z is None:
        raise ValueError("reference_mode='x' requires Z")
    use_x = (Z is not None) and reference_mode in ("auto", "x")
    if reference_mode == "auto" and Z is None:
        notes.append("no Z supplied: using x-INDEPENDENT reference points, "
                     "which cannot detect a posterior that ignores the data")

    lo = np.minimum(theta_true.min(axis=0), ps.reshape(-1, p).min(axis=0))
    hi = np.maximum(theta_true.max(axis=0), ps.reshape(-1, p).max(axis=0))

    fracs = []
    for rep in range(n_references):
        if reference_points is not None:
            r = np.atleast_2d(reference_points)
            if r.shape[0] == 1:
                r = np.repeat(r, n, axis=0)
        elif use_x:
            Za = np.atleast_2d(np.asarray(Z, dtype=np.float64))
            # Fit the readout on one half, apply to the other, swapping the
            # halves so every observation gets an out-of-fit reference.
            idx = rng.permutation(n)
            h = max(2, n // 2)
            first, second = idx[:h], idx[h:]
            r = np.empty_like(theta_true)
            for fit, app in ((first, second), (second, first)):
                if fit.size < p + 1 or app.size == 0:
                    continue
                A = np.c_[Za[fit], np.ones(fit.size)]
                coef, *_ = np.linalg.lstsq(A, theta_true[fit], rcond=None)
                r[app] = np.c_[Za[app], np.ones(app.size)] @ coef
            if fit.size < p + 1:
                r = rng.uniform(lo, hi, size=(n, p))
                notes.append("too few observations to fit an x-dependent "
                             "readout; fell back to random references")
        else:
            r = rng.uniform(lo, hi, size=(n, p))

        d_true = np.linalg.norm(theta_true - r, axis=1)
        d_samp = np.linalg.norm(ps - r[:, None, :], axis=2)
        fracs.append(np.mean(d_samp < d_true[:, None], axis=1))
    f = np.concatenate(fracs)

    # Jitter for the same reason as in _rank_uniformity: f takes M+1
    # discrete values, so an unjittered KS against a continuous uniform is
    # biased towards rejection.
    f_j = np.clip(f + rng.uniform(-0.5 / m, 0.5 / m, size=f.shape), 0.0, 1.0)
    pval = float(stats.kstest(f_j, "uniform").pvalue)

    xs = np.sort(f)
    ys = np.arange(1, xs.size + 1) / xs.size
    return TARPResult(ecdf_x=xs, ecdf_y=ys, ks_pvalue=pval,
                      max_deviation=float(np.max(np.abs(ys - xs))),
                      reference_mode=("x-dependent" if use_x else "random"),
                      notes=notes)


# ===========================================================================
# 4. Informativeness -- NOT a calibration check
# ===========================================================================

@dataclass
class ContractionResult:
    param_names: List[str]
    contraction: np.ndarray        # per-axis, 1 - Var[post]/Var[prior]
    prior_std: np.ndarray
    post_std: np.ndarray
    eigenvalues: Optional[np.ndarray] = None
    effective_rank: Optional[int] = None

    def summary(self, top: int = 10) -> str:
        order = np.argsort(-self.contraction)
        lines = ["informativeness (contraction = 1 - Var[post]/Var[prior])",
                 "  %-16s %10s %10s %10s" % ("axis", "contract", "sd_prior", "sd_post")]
        for k in order[:top]:
            lines.append("  %-16s %10.3f %10.4g %10.4g"
                         % (self.param_names[k], self.contraction[k],
                            self.prior_std[k], self.post_std[k]))
        if len(order) > top:
            lines.append("  ... %d more axes" % (len(order) - top))
        lines.append("  mean contraction : %.3f" % float(np.mean(self.contraction)))
        n_flat = int(np.sum(self.contraction < 0.05))
        lines.append("  axes with contraction < 0.05 (posterior ~= prior): %d/%d"
                     % (n_flat, len(self.contraction)))
        if self.effective_rank is not None:
            lines.append("  effective rank of the information spectrum: %d"
                         % self.effective_rank)
        return "\n".join(lines)


def posterior_contraction(theta_true: np.ndarray,
                          posterior_samples: np.ndarray,
                          param_names: Optional[Sequence[str]] = None
                          ) -> ContractionResult:
    """Per-axis posterior contraction relative to the prior.

        contraction_k = 1 - E_i[ Var(theta_k | z_i) ] / Var_prior(theta_k)

    0 means the posterior marginal is as wide as the prior -- the data said
    nothing about that axis. 1 means it is fully determined.

    This is the layer that calibration cannot reach. A posterior exactly
    equal to the prior is perfectly calibrated and completely uninformative;
    only this number distinguishes the two.
    """
    theta_true = np.atleast_2d(np.asarray(theta_true, dtype=np.float64))
    ps = np.asarray(posterior_samples, dtype=np.float64)
    n, m, p = ps.shape
    names = list(param_names) if param_names is not None else \
        ["theta_%02d" % k for k in range(p)]

    prior_var = theta_true.var(axis=0)
    post_var = ps.var(axis=1).mean(axis=0)
    safe = np.where(prior_var > 0, prior_var, np.nan)
    contraction = 1.0 - post_var / safe
    return ContractionResult(param_names=names,
                             contraction=contraction,
                             prior_std=np.sqrt(prior_var),
                             post_std=np.sqrt(post_var))


def information_spectrum(theta_true: np.ndarray,
                         posterior_samples: np.ndarray,
                         param_names: Optional[Sequence[str]] = None,
                         threshold: float = 0.05) -> ContractionResult:
    """How many INDEPENDENT directions of parameter space the data constrains.

    Let mu_i = E[theta | z_i] be the posterior mean for observation i. Since
    mu_i is a function of z_i alone, the covariance of {mu_i} across the
    calibration set is the part of the prior covariance the observation can
    account for. The generalised eigenvalues of

        Cov_i(mu_i)  relative to  Cov(theta*)                          (T2)

    lie in [0, 1] and give the fraction of prior variance explained along
    each eigendirection. Counting those above `threshold` gives the
    effective number of constrained directions.

    IMPORTANT LIMIT ON HOW THIS MAY BE READ. It is tempting to treat the
    effective rank as a hard test of the bound "an embedding of dimension E
    constrains at most E-1 parameter directions". That reading is WRONG for
    a nonlinear estimator, and the distinction matters.

    The map z -> E[theta | z] does have an image of INTRINSIC (manifold)
    dimension at most E-1, since z carries only that many degrees of
    freedom. But (T2) measures the LINEAR rank of the covariance of that
    image, and a curved d-dimensional manifold spans more than d linear
    dimensions. Measured directly: for z on S^11 and a linear f(z), the
    covariance rank is exactly 11; for a nonlinear f(z) on the same z it is
    12 or more. A normalizing flow is nonlinear, so an effective rank above
    E-1 is expected behaviour, not a bug and not evidence of a leak.

    What survives is the information-theoretic statement: by the data
    processing inequality I(theta; z) <= I(theta; x), and z carries at most
    E-1 real numbers, so the TOTAL information is capped however it is
    spread. A nonlinear map can spread that budget thinly across all p
    linear directions rather than concentrating it in E-1 of them.

    So use this spectrum descriptively -- how many directions carry most of
    the constrained variance, and how fast it decays -- and use per-axis
    posterior_contraction for the question "is this parameter informed?".
        """
    theta_true = np.atleast_2d(np.asarray(theta_true, dtype=np.float64))
    ps = np.asarray(posterior_samples, dtype=np.float64)
    p = ps.shape[2]
    names = list(param_names) if param_names is not None else \
        ["theta_%02d" % k for k in range(p)]

    base = posterior_contraction(theta_true, ps, names)

    mu = ps.mean(axis=1)                       # (N, p)
    C_expl = np.cov(mu, rowvar=False)
    C_prior = np.cov(theta_true, rowvar=False)
    C_expl = np.atleast_2d(C_expl)
    C_prior = np.atleast_2d(C_prior)

    # Whiten by the prior so the eigenvalues are dimensionless fractions and
    # comparable across axes with wildly different units -- essential here,
    # where some axes are ln-coordinates and others are linear.
    w, V = np.linalg.eigh(C_prior)
    w = np.maximum(w, 1e-300)
    W_inv_half = V @ np.diag(w ** -0.5) @ V.T
    M = W_inv_half @ C_expl @ W_inv_half
    eig = np.sort(np.clip(np.linalg.eigvalsh(0.5 * (M + M.T)), 0.0, None))[::-1]

    base.eigenvalues = eig
    base.effective_rank = int(np.sum(eig > threshold))
    return base


# ===========================================================================
# 5. Reporting
# ===========================================================================

def diagnostic_report(sbc: Optional[Sequence[RankResult]] = None,
                      coverage: Optional[RankResult] = None,
                      ddsbc: Optional[Sequence[RankResult]] = None,
                      tarp_res: Optional[TARPResult] = None,
                      contraction: Optional[ContractionResult] = None,
                      overlap: Optional[MMDResult] = None) -> str:
    """Assemble one readable report, ordered by how much each layer can see."""
    out = ["=" * 70, "POST-TRAINING DIAGNOSTIC REPORT", "=" * 70]

    if overlap is not None:
        out += ["", "[layer 1] " + overlap.summary()]

    if sbc is not None:
        fam = family_verdict(sbc)
        out += ["", "[layer 2] marginal SBC", "  " + fam.summary()]
        shown = [r for k, r in enumerate(sbc) if fam.rejected[k]] or sbc[:5]
        out += [r.summary() for r in shown]
        if fam.passes and len(sbc) > 5:
            out.append("  ... %d more axes, all passing" % (len(sbc) - 5))

    if coverage is not None:
        out += ["", "[layer 2] joint coverage", coverage.summary()]

    if ddsbc is not None:
        fam_dd = family_verdict(ddsbc)
        n_fail = fam_dd.n_rejected
        out += ["", "[layer 2] DATA-DEPENDENT SBC", "  " + fam_dd.summary()]
        out += [r.summary() for r in ddsbc]
        if n_fail:
            out.append("  -> the posterior is not using the observation "
                       "correctly; note this can fail while every")
            out.append("     parameter-only check above passes.")

    if tarp_res is not None:
        out += ["", "[layer 3] " + tarp_res.summary()]

    if contraction is not None:
        out += ["", "[layer 4] " + contraction.summary()]
        flat = int(np.sum(contraction.contraction < 0.05))
        if flat == len(contraction.contraction):
            out.append("  -> WARNING: no axis contracts. A posterior equal to "
                       "the prior passes every calibration")
            out.append("     check above; calibration alone cannot detect this.")

    out += ["", "=" * 70]
    return "\n".join(out)
