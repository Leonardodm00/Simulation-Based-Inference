"""Replicate-consistency statistic T_gg' and its null target p_eff.

Implements, for two wells g and g' of one donor:

    Delta      = m_g - m_g'                                   (difference of
                                                               posterior means)
    C_bar      = (C_g + C_g') / 2                             (symmetrised
                                                               posterior cov)
    T          = Delta^T (2 C_bar)^{-1} Delta                  eq. (3)
    p_eff      = d - tr(Sigma0^{-1} C_bar)                     eq. (9)
               = sum_j lambda_j / (1 + lambda_j)               eq. (10)
    L_rep      = ( log T - log p_eff )^2                       eq. (12)

Nothing here evaluates V = Var(Delta | theta_star): V appears in the
DERIVATION of the p_eff target only. The quantities actually evaluated are
Sigma0 (known prior) and C (posterior covariance from the flow).

Conventions
-----------
d          : dimension of theta_glob (23 in the joint DSN/NPE plan)
Sigma0     : prior covariance, (d, d), SPD
C_g, C_gp  : posterior covariances at x_g and x_g', (d, d), SPD
All matrices are numpy arrays, float64. No Greek letters, pure ASCII source.

Separation of concerns: this module contains estimator logic only. It does no
I/O, no plotting, and no flow evaluation. Feed it posterior SAMPLES or
posterior MOMENTS produced elsewhere.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh


# ---------------------------------------------------------------------------
# 1. Moments from posterior samples
# ---------------------------------------------------------------------------

def posterior_moments(samples):
    """Sample mean and covariance of posterior draws.

    Parameters
    ----------
    samples : (S, d) array of draws from q_phi(theta | x) for ONE well.

    Returns
    -------
    m : (d,) posterior mean estimate
    C : (d, d) posterior covariance estimate (unbiased, ddof=1)
    S : int, number of draws (needed for the finite-sample correction)
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("samples must be (S, d)")
    S = samples.shape[0]
    if S < 2:
        raise ValueError("need at least 2 draws")
    m = samples.mean(axis=0)
    C = np.cov(samples, rowvar=False, ddof=1)
    return m, np.atleast_2d(C), S


# ---------------------------------------------------------------------------
# 2. The statistic
# ---------------------------------------------------------------------------

def replicate_statistic(m_g, m_gp, C_g, C_gp, n_draws=None, correct_mc=False):
    """T = Delta^T (2 C_bar)^{-1} Delta, with C_bar = (C_g + C_gp)/2.

    The symmetrised C_bar is used so that T is exchangeable in (g, g'); the
    plan documents write a single C without specifying which well it is
    evaluated at.

    Parameters
    ----------
    m_g, m_gp : (d,) posterior means for the two wells.
    C_g, C_gp : (d, d) posterior covariances for the two wells.
    n_draws   : int or None. Number of posterior draws used to form each mean.
                Required if correct_mc is True.
    correct_mc: bool. If True, subtract the finite-sample inflation d/n_draws
                (see notes) from T.

    Returns
    -------
    T : float, non-negative.

    Notes
    -----
    Each mean is estimated from S draws, so m_hat = m + e with
    Var(e) = C / S. Then Delta_hat = Delta + e_g - e_gp carries an extra
    covariance 2 C / S, and

        E[T_hat] = E[T] + tr( (2 C)^{-1} 2 C / S ) = E[T] + d / S.

    With d = 23 and S = 1000 this is 0.023, small next to p_eff ~ 3; with
    S = 100 it is 0.23, which is not negligible.
    """
    m_g = np.asarray(m_g, dtype=np.float64).ravel()
    m_gp = np.asarray(m_gp, dtype=np.float64).ravel()
    d = m_g.size
    if m_gp.size != d:
        raise ValueError("m_g and m_gp must have the same length")

    C_bar = 0.5 * (np.asarray(C_g, dtype=np.float64)
                   + np.asarray(C_gp, dtype=np.float64))
    C_bar = 0.5 * (C_bar + C_bar.T)          # enforce exact symmetry

    delta = m_g - m_gp
    # Solve (2 C_bar) z = delta by Cholesky rather than forming an inverse.
    cho = cho_factor(2.0 * C_bar, lower=True)
    z = cho_solve(cho, delta)
    T = float(delta @ z)

    if correct_mc:
        if n_draws is None:
            raise ValueError("correct_mc=True requires n_draws")
        T = T - d / float(n_draws)
    return T


def mahalanobis_statistic(delta, M):
    """T = delta^T M delta for an explicit metric M (used for M = V^{-1})."""
    delta = np.asarray(delta, dtype=np.float64).ravel()
    M = np.asarray(M, dtype=np.float64)
    return float(delta @ M @ delta)


# ---------------------------------------------------------------------------
# 3. The null target
# ---------------------------------------------------------------------------

def p_eff_from_trace(Sigma0, C_bar):
    """p_eff = d - tr(Sigma0^{-1} C_bar), eq. (9). The evaluable form."""
    Sigma0 = np.asarray(Sigma0, dtype=np.float64)
    C_bar = np.asarray(C_bar, dtype=np.float64)
    d = Sigma0.shape[0]
    cho = cho_factor(Sigma0, lower=True)
    X = cho_solve(cho, C_bar)                # X = Sigma0^{-1} C_bar
    return float(d - np.trace(X))


def p_eff_from_spectrum(lambdas):
    """p_eff = sum_j lambda_j / (1 + lambda_j), eq. (10).

    lambdas are the generalised eigenvalues of (F, Sigma0^{-1}); this is what
    npe_diagnostics.information_spectrum reports.
    """
    lam = np.asarray(lambdas, dtype=np.float64)
    return float(np.sum(lam / (1.0 + lam)))


def generalised_spectrum(F, Sigma0):
    """Generalised eigenvalues lambda_j of F v = lambda Sigma0^{-1} v.

    Bench-only: requires F, which is unavailable on real data.
    """
    F = np.asarray(F, dtype=np.float64)
    Sigma0_inv = np.linalg.inv(np.asarray(Sigma0, dtype=np.float64))
    lam = eigh(F, Sigma0_inv, eigvals_only=True)
    return np.sort(lam)[::-1]


def box_prior_covariance(lower, upper):
    """Moment covariance of a uniform box prior: diag((U-L)^2 / 12).

    NOTE this is an approximation in context: the derivation of eq. (9)
    treats the prior as Gaussian with covariance Sigma0. Using the box's
    moment matrix there is a separate approximation from the conjugacy one.
    """
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    return np.diag((upper - lower) ** 2 / 12.0)


# ---------------------------------------------------------------------------
# 4. The loss
# ---------------------------------------------------------------------------

def replicate_loss(T, p_eff, eps=1e-12):
    """L_rep = (log T - log p_eff)^2, eq. (12). Two-sided about p_eff."""
    return float((np.log(max(T, eps)) - np.log(max(p_eff, eps))) ** 2)


# ---------------------------------------------------------------------------
# 5. Conjugate Gaussian reference model (bench validation only)
# ---------------------------------------------------------------------------

def conjugate_posterior_covariance(Sigma0, F):
    """C = (Sigma0^{-1} + F)^{-1}. Bench-only: needs F."""
    Sigma0_inv = np.linalg.inv(np.asarray(Sigma0, dtype=np.float64))
    return np.linalg.inv(Sigma0_inv + np.asarray(F, dtype=np.float64))


def simulate_replicate_pairs(theta_star, Sigma0, F, n_pairs, rng):
    """Simulate n_pairs of posterior means from a shared theta_star.

    Conjugate Gaussian model: theta_hat | theta_star ~ N(theta_star, F^{-1})
    and m = C F theta_hat, eq. (4). Two wells are conditionally independent
    given theta_star, hence two independent theta_hat draws per pair.

    Returns
    -------
    deltas : (n_pairs, d) array of Delta = m_g - m_gp
    C      : (d, d) posterior covariance (data-independent in this model)
    """
    theta_star = np.asarray(theta_star, dtype=np.float64).ravel()
    d = theta_star.size
    C = conjugate_posterior_covariance(Sigma0, F)
    F_inv = np.linalg.inv(np.asarray(F, dtype=np.float64))
    A = C @ np.asarray(F, dtype=np.float64)          # m = A theta_hat

    L = np.linalg.cholesky(F_inv)
    zg = rng.standard_normal((n_pairs, d)) @ L.T
    zgp = rng.standard_normal((n_pairs, d)) @ L.T
    theta_hat_g = theta_star + zg
    theta_hat_gp = theta_star + zgp
    deltas = (theta_hat_g - theta_hat_gp) @ A.T
    return deltas, C


def sampling_covariance_V(Sigma0, F):
    """V = 2 C F C, eq. (6). Bench-only. Never evaluated on real data."""
    C = conjugate_posterior_covariance(Sigma0, F)
    return 2.0 * C @ np.asarray(F, dtype=np.float64) @ C
