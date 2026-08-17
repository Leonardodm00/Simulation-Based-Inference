#!/usr/bin/env python3
"""
gmm_benchmark.py -- a K-component, n-dimensional Gaussian benchmark with an
EXACT analytic posterior, used to test whether the NPE recovers a known
multimodal answer.

THE MODEL
---------
Prior over parameters, a mixture of K Gaussians in n dimensions:

    p(theta) = sum_k w_k * N(theta ; m_k, S_k),      theta in R^n         (G1)

Likelihood, linear-Gaussian with a d-dimensional observation:

    p(x | theta) = N(x ; A theta + b, Sigma),        x in R^d             (G2)

with A of shape (d, n). The posterior is then, exactly and in closed form,
again a K-component Gaussian mixture:

    p(theta | x) = sum_k wt_k(x) * N(theta ; mu_k(x), C_k)                (G3)

    C_k      = ( S_k^-1 + A^T Sigma^-1 A )^-1                             (G4)
    mu_k(x)  = C_k ( S_k^-1 m_k + A^T Sigma^-1 (x - b) )                  (G5)
    wt_k(x) proportional to  w_k * N(x ; A m_k + b, A S_k A^T + Sigma)    (G6)

(G4)-(G6) are the standard conjugate linear-Gaussian update applied
component-wise, using p(theta) p(x|theta) = p(x) p(theta|x) on each term.
They are DERIVED here, so smoke test G1 re-verifies them numerically by
self-normalised importance sampling rather than trusting the algebra.

WHY THIS PARTICULAR CONSTRUCTION
--------------------------------
Set d < n and place the component means so that they differ ONLY within
null(A), the (n - d)-dimensional subspace the observation cannot see. Then

  * A m_k + b is identical for every k, so by (G6) with equal S_k the
    posterior weights equal the PRIOR weights exactly -- an exact, testable
    statement rather than an approximate one;
  * with S_k = s^2 I, any v in null(A) satisfies C_k^-1 v = v / s^2, so
    C_k v = s^2 v and the mode separation survives the update undiminished.

So the posterior is genuinely and permanently K-modal: no amount of data in
these d directions collapses it. That is the controlled analogue of having
fewer informative embedding directions than parameters -- prior structure
survives in the unresolved subspace, and an estimator that quietly reports a
single mode is wrong in a way that calibration checks alone will not reveal.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["GMMBenchmark", "GMMPosterior"]


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

def _logdet_spd(M: np.ndarray) -> float:
    """Log-determinant of a symmetric positive-definite matrix via Cholesky."""
    L = np.linalg.cholesky(M)
    return 2.0 * float(np.sum(np.log(np.diag(L))))


def _mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Multivariate normal log-density. x may be (d,) or (n, d)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    d = mean.shape[0]
    L = np.linalg.cholesky(cov)
    diff = (x - mean[None, :]).T
    sol = np.linalg.solve(L, diff)
    quad = np.sum(sol ** 2, axis=0)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (quad + logdet + d * np.log(2.0 * np.pi))


def _log_normalise(log_w: np.ndarray) -> np.ndarray:
    m = np.max(log_w)
    w = np.exp(log_w - m)
    return w / np.sum(w)


# ---------------------------------------------------------------------------
# Posterior container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GMMPosterior:
    """An exact K-component Gaussian mixture posterior for one observation.

    Attributes
    ----------
    weights : ndarray, shape (K,)   -- wt_k(x) from (G6), summing to 1.
    means   : ndarray, shape (K, n) -- mu_k(x) from (G5).
    covs    : ndarray, shape (K, n, n) -- C_k from (G4).
    """

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_dim(self) -> int:
        return int(self.means.shape[1])

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        """Exact log p(theta | x) for each row of theta, for the fixed x that
        produced this object."""
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        parts = np.stack([
            np.log(self.weights[k]) + _mvn_logpdf(theta, self.means[k], self.covs[k])
            for k in range(self.n_components)
        ])
        m = np.max(parts, axis=0)
        return m + np.log(np.sum(np.exp(parts - m[None, :]), axis=0))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact draws: pick a component by weight, then sample it."""
        which = rng.choice(self.n_components, size=n, p=self.weights)
        out = np.empty((n, self.n_dim), dtype=np.float64)
        for k in range(self.n_components):
            idx = np.flatnonzero(which == k)
            if idx.size:
                out[idx] = rng.multivariate_normal(self.means[k], self.covs[k], size=idx.size)
        return out

    def mean(self) -> np.ndarray:
        return np.sum(self.weights[:, None] * self.means, axis=0)

    def cov(self) -> np.ndarray:
        """Law of total covariance: E[Cov] + Cov[E]."""
        mu = self.mean()
        within = np.sum(self.weights[:, None, None] * self.covs, axis=0)
        dm = self.means - mu[None, :]
        between = np.einsum("k,ki,kj->ij", self.weights, dm, dm)
        return within + between

    def assign(self, theta: np.ndarray) -> np.ndarray:
        """Hard-assign each row of theta to its most probable component."""
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        parts = np.stack([
            np.log(self.weights[k]) + _mvn_logpdf(theta, self.means[k], self.covs[k])
            for k in range(self.n_components)
        ])
        return np.argmax(parts, axis=0)

    def min_pairwise_separation(self) -> float:
        """Smallest centre-to-centre distance in units of the within-component
        standard deviation along the connecting direction.

        A value well above ~2 means the modes are genuinely resolved, so a
        recovery test on this problem is actually discriminating rather than
        trivially passable by a unimodal fit.
        """
        best = np.inf
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                dm = self.means[i] - self.means[j]
                dist = float(np.linalg.norm(dm))
                if dist == 0.0:
                    return 0.0
                u = dm / dist
                sd_i = float(np.sqrt(u @ self.covs[i] @ u))
                sd_j = float(np.sqrt(u @ self.covs[j] @ u))
                best = min(best, dist / (0.5 * (sd_i + sd_j)))
        return float(best)


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

class GMMBenchmark:
    """K n-dimensional Gaussians to be recovered from d-dimensional data.

    Parameters
    ----------
    n_dim : int
        n, the parameter dimension.
    n_obs : int
        d, the observation dimension. Choose d < n for the null-space
        construction described in the module docstring.
    n_components : int
        K, the number of Gaussians. Default 3.
    separation : float
        Distance between adjacent component means, in units of the prior
        component standard deviation.
    prior_scale : float
        s, the isotropic prior component standard deviation. Isotropic by
        default so the exact weight- and separation-preservation properties
        hold; pass component_covs to override.
    obs_noise : float
        Observation noise standard deviation (isotropic).
    weights : ndarray or None
        Mixture weights. Default: unequal (so the test can detect an
        estimator that recovers the modes but not their relative mass).
    separate_in_nullspace : bool
        If True, offset the component means inside null(A) so the modes are
        unresolvable by the data and survive exactly. If False, offset them
        in a generic direction, in which case the data partially resolves
        them and the posterior weights differ from the prior weights.
    """

    def __init__(
        self,
        n_dim: int = 6,
        n_obs: int = 3,
        n_components: int = 3,
        separation: float = 6.0,
        prior_scale: float = 1.0,
        obs_noise: float = 0.4,
        weights: Optional[np.ndarray] = None,
        separate_in_nullspace: bool = True,
        seed: int = 0,
    ) -> None:
        if separate_in_nullspace and n_obs >= n_dim:
            raise ValueError(
                "separate_in_nullspace requires n_obs < n_dim so that null(A) "
                "is non-trivial; got n_obs=%d, n_dim=%d" % (n_obs, n_dim))
        if n_components < 2:
            raise ValueError("n_components must be at least 2")

        rng = np.random.default_rng(seed)
        self.n_dim = int(n_dim)
        self.n_obs = int(n_obs)
        self.n_components = int(n_components)
        self.prior_scale = float(prior_scale)
        self.separate_in_nullspace = bool(separate_in_nullspace)

        # -- observation map ------------------------------------------------
        self.A = rng.normal(size=(self.n_obs, self.n_dim)) / np.sqrt(self.n_dim)
        self.b = rng.normal(size=(self.n_obs,))
        self.Sigma = (obs_noise ** 2) * np.eye(self.n_obs)

        # -- an orthonormal basis of null(A) via the SVD --------------------
        _, sv, Vt = np.linalg.svd(self.A)
        rank = int(np.sum(sv > 1e-10 * sv.max()))
        self.null_basis = Vt[rank:]          # (n - rank, n)
        self.row_basis = Vt[:rank]           # (rank, n)
        if self.null_basis.shape[0] == 0 and separate_in_nullspace:
            raise ValueError("A has trivial null space; cannot separate in null(A)")

        # -- component means -------------------------------------------------
        # Offsets are placed on a simplex-like set of directions so no two
        # components coincide and none sits at the origin of the offset set.
        offs = rng.normal(size=(self.n_components, max(1, self.null_basis.shape[0])))
        offs -= offs.mean(axis=0, keepdims=True)
        offs /= np.maximum(np.linalg.norm(offs, axis=1, keepdims=True), 1e-12)
        if separate_in_nullspace:
            directions = offs @ self.null_basis          # (K, n), inside null(A)
        else:
            gen = rng.normal(size=(self.n_components, self.n_dim))
            gen -= gen.mean(axis=0, keepdims=True)
            gen /= np.maximum(np.linalg.norm(gen, axis=1, keepdims=True), 1e-12)
            directions = gen
        centre = rng.normal(size=(self.n_dim,)) * 0.5
        self.means = centre[None, :] + separation * prior_scale * directions

        # -- component covariances (isotropic by construction) --------------
        self.covs = np.stack([
            (prior_scale ** 2) * np.eye(self.n_dim) for _ in range(self.n_components)
        ])

        # -- weights ---------------------------------------------------------
        if weights is None:
            base = np.linspace(1.0, 2.0, self.n_components)
            weights = base / base.sum()
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.n_components,):
            raise ValueError("weights must have shape (%d,)" % self.n_components)
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("weights must sum to 1")
        self.weights = weights

        # -- cached precisions ------------------------------------------------
        self._Sigma_inv = np.linalg.inv(self.Sigma)
        self._S_inv = np.stack([np.linalg.inv(c) for c in self.covs])
        AtSiA = self.A.T @ self._Sigma_inv @ self.A
        self._post_covs = np.stack([
            np.linalg.inv(self._S_inv[k] + AtSiA) for k in range(self.n_components)
        ])
        # marginal covariance of x under each component, for (G6)
        self._marg_covs = np.stack([
            self.A @ self.covs[k] @ self.A.T + self.Sigma for k in range(self.n_components)
        ])
        self._marg_means = np.stack([
            self.A @ self.means[k] + self.b for k in range(self.n_components)
        ])

    # -- prior ---------------------------------------------------------------

    def prior_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        which = rng.choice(self.n_components, size=n, p=self.weights)
        out = np.empty((n, self.n_dim), dtype=np.float64)
        for k in range(self.n_components):
            idx = np.flatnonzero(which == k)
            if idx.size:
                out[idx] = rng.multivariate_normal(self.means[k], self.covs[k], size=idx.size)
        return out

    def prior_log_prob(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        parts = np.stack([
            np.log(self.weights[k]) + _mvn_logpdf(theta, self.means[k], self.covs[k])
            for k in range(self.n_components)
        ])
        m = np.max(parts, axis=0)
        return m + np.log(np.sum(np.exp(parts - m[None, :]), axis=0))

    def torch_prior(self, device: str = "cpu"):
        """The same prior as a torch Distribution, for use as the sbi prior.

        A plain MixtureSameFamily carries a MixtureSameFamilyConstraint, which
        torch cannot build a bijection for; sbi needs one in order to map the
        prior support to an unconstrained space when sampling. A mixture of
        full-support Gaussians is supported on all of R^n, so the constraint
        is declared explicitly here rather than left to be inferred.
        """
        import torch
        from torch.distributions import (Categorical, Distribution,
                                         MixtureSameFamily, MultivariateNormal,
                                         constraints)

        mix = Categorical(torch.as_tensor(self.weights, dtype=torch.float32, device=device))
        comp = MultivariateNormal(
            torch.as_tensor(self.means, dtype=torch.float32, device=device),
            covariance_matrix=torch.as_tensor(self.covs, dtype=torch.float32, device=device),
        )
        base = MixtureSameFamily(mix, comp)

        class RealVectorMixture(Distribution):
            """MixtureSameFamily with support declared as R^n."""

            arg_constraints: dict = {}
            has_rsample = False

            def __init__(self, inner):
                self._inner = inner
                super().__init__(inner.batch_shape, inner.event_shape,
                                 validate_args=False)

            @constraints.dependent_property
            def support(self):
                return constraints.independent(constraints.real, 1)

            def sample(self, sample_shape=torch.Size()):
                return self._inner.sample(sample_shape)

            def log_prob(self, value):
                return self._inner.log_prob(value)

            @property
            def mean(self):
                return self._inner.mean

            @property
            def variance(self):
                return self._inner.variance

        return RealVectorMixture(base)

    # -- likelihood ----------------------------------------------------------

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw x ~ N(A theta + b, Sigma), one row per row of theta."""
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        mu = theta @ self.A.T + self.b[None, :]
        L = np.linalg.cholesky(self.Sigma)
        eps = rng.normal(size=(theta.shape[0], self.n_obs)) @ L.T
        return mu + eps

    def log_likelihood(self, theta: np.ndarray, x_o: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        mu = theta @ self.A.T + self.b[None, :]
        L = np.linalg.cholesky(self.Sigma)
        sol = np.linalg.solve(L, (np.asarray(x_o)[None, :] - mu).T)
        quad = np.sum(sol ** 2, axis=0)
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        return -0.5 * (quad + logdet + self.n_obs * np.log(2.0 * np.pi))

    # -- exact posterior ------------------------------------------------------

    def posterior(self, x_o: np.ndarray) -> GMMPosterior:
        """Exact p(theta | x_o) via (G4)-(G6), for each fixed x_o."""
        x_o = np.asarray(x_o, dtype=np.float64).ravel()
        if x_o.shape != (self.n_obs,):
            raise ValueError("x_o must have shape (%d,), got %s" % (self.n_obs, x_o.shape))

        rhs_data = self.A.T @ self._Sigma_inv @ (x_o - self.b)
        means = np.stack([
            self._post_covs[k] @ (self._S_inv[k] @ self.means[k] + rhs_data)
            for k in range(self.n_components)
        ])
        log_w = np.array([
            np.log(self.weights[k])
            + float(_mvn_logpdf(x_o, self._marg_means[k], self._marg_covs[k])[0])
            for k in range(self.n_components)
        ])
        return GMMPosterior(weights=_log_normalise(log_w),
                            means=means,
                            covs=self._post_covs.copy())

    # -- reporting ------------------------------------------------------------

    def describe(self, x_o: Optional[np.ndarray] = None) -> str:
        lines = [
            "GMMBenchmark: K=%d Gaussians in n=%d dims, observed through d=%d dims"
            % (self.n_components, self.n_dim, self.n_obs),
            "  null(A) dimension        : %d" % self.null_basis.shape[0],
            "  means separated in       : %s"
            % ("null(A) -- unresolvable by data" if self.separate_in_nullspace
               else "a generic direction -- partially resolvable"),
            "  prior weights            : %s" % np.round(self.weights, 4).tolist(),
        ]
        if x_o is not None:
            post = self.posterior(x_o)
            lines += [
                "  posterior weights        : %s" % np.round(post.weights, 4).tolist(),
                "  mode separation (sd units): %.2f" % post.min_pairwise_separation(),
            ]
        return "\n".join(lines)


def reference_posterior_by_importance(
    bench: GMMBenchmark,
    x_o: np.ndarray,
    n_samples: int = 400000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Brute-force posterior moments by self-normalised importance sampling.

    Proposal is the prior, weights are the likelihood. This is deliberately
    a different route to the same answer than (G4)-(G6), so agreement
    between the two validates the conjugate algebra rather than merely
    re-executing it.

    Returns (mean, cov, diagnostics).
    """
    rng = rng or np.random.default_rng(0)
    theta = bench.prior_sample(n_samples, rng)
    log_w = bench.log_likelihood(theta, x_o)
    w = _log_normalise(log_w)
    mean = w @ theta
    dm = theta - mean[None, :]
    cov = np.einsum("i,ij,ik->jk", w, dm, dm)
    ess = 1.0 / float(np.sum(w ** 2))
    return mean, cov, {"ess": ess, "ess_frac": ess / n_samples}
