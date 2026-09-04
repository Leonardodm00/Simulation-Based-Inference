"""Smoke test for replicate_statistic.py.

Run:  python3 smoke_test_replicate_statistic.py

Each test prints PASS or FAIL and the numbers behind the verdict. The script
exits non-zero if any test fails, so it can be dropped into CI or into the
cluster-side verification block.

What is being validated, and why each test exists:

  T1  p_eff computed two ways (trace formula vs generalised eigenvalues)
      agree, on a NON-DIAGONAL correlated model. Guards eq. (9) = eq. (10).
  T2  E[T] under M = (2C)^{-1} equals p_eff. This is THE bench check the
      plan requires before the p_eff target may be trusted.
  T3  E[T] under M = V^{-1} equals d, and T is chi^2_d in distribution
      (Kolmogorov-Smirnov). Turns the target into a test.
  T4  T is invariant under theta -> A theta for invertible A. Admissibility.
  T5  Finite posterior-sample inflation: E[T_hat] = p_eff + d/S, and the
      correction removes it.
  T6  Limits: F -> 0 gives p_eff -> 0 and Delta -> 0 (sloppy limit,
      replicates agree exactly); F >> Sigma0^{-1} gives p_eff -> d.
  T7  The loss is minimised at T = p_eff and is two-sided.

All source is pure ASCII (HPC-safe).
"""

import sys

import numpy as np
from scipy import stats

from replicate_statistic import (
    box_prior_covariance,
    conjugate_posterior_covariance,
    generalised_spectrum,
    mahalanobis_statistic,
    p_eff_from_spectrum,
    p_eff_from_trace,
    posterior_moments,
    replicate_loss,
    replicate_statistic,
    sampling_covariance_V,
    simulate_replicate_pairs,
)

RESULTS = []


def report(name, ok, detail):
    RESULTS.append(bool(ok))
    print("[%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))


def random_spd(d, rng, scale=1.0, cond_spread=2.0):
    """Random SPD matrix with a spread of eigenvalues (non-diagonal)."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigvals = scale * np.exp(rng.uniform(-cond_spread, cond_spread, size=d))
    return Q @ np.diag(eigvals) @ Q.T


def make_model(d, rng, f_scale=1.0):
    """A correlated, non-diagonal (Sigma0, F) pair with a mix of stiff and
    sloppy directions."""
    Sigma0 = random_spd(d, rng, scale=1.0, cond_spread=0.5)
    F = random_spd(d, rng, scale=f_scale, cond_spread=3.0)
    return Sigma0, F


# ---------------------------------------------------------------------------
# T1 -- p_eff two ways
# ---------------------------------------------------------------------------

def test_peff_consistency():
    rng = np.random.default_rng(0)
    d = 23
    Sigma0, F = make_model(d, rng)
    C = conjugate_posterior_covariance(Sigma0, F)

    p_trace = p_eff_from_trace(Sigma0, C)
    lam = generalised_spectrum(F, Sigma0)
    p_spec = p_eff_from_spectrum(lam)

    ok = np.isclose(p_trace, p_spec, rtol=1e-8, atol=1e-8)
    report("T1 p_eff: trace form == eigenvalue form", ok,
           "trace=%.6f  spectrum=%.6f  (d=%d)" % (p_trace, p_spec, d))
    return p_trace


# ---------------------------------------------------------------------------
# T2 -- E[T] == p_eff under M = (2C)^{-1}
# ---------------------------------------------------------------------------

def test_null_expectation_2C():
    rng = np.random.default_rng(1)
    d = 23
    n_pairs = 400000
    Sigma0, F = make_model(d, rng)
    C = conjugate_posterior_covariance(Sigma0, F)
    p_eff = p_eff_from_trace(Sigma0, C)

    deltas, C_sim = simulate_replicate_pairs(np.zeros(d), Sigma0, F,
                                             n_pairs, rng)
    M = np.linalg.inv(2.0 * C_sim)
    T = np.einsum("ni,ij,nj->n", deltas, M, deltas)

    mean_T = T.mean()
    se = T.std(ddof=1) / np.sqrt(n_pairs)
    ok = abs(mean_T - p_eff) < 4.0 * se
    report("T2 E[T] == p_eff for M = (2C)^{-1}", ok,
           "mean(T)=%.4f +/- %.4f (4 se)   p_eff=%.4f" %
           (mean_T, 4.0 * se, p_eff))


# ---------------------------------------------------------------------------
# T3 -- exact chi^2_d null under M = V^{-1}
# ---------------------------------------------------------------------------

def test_chi2_null_Vinv():
    rng = np.random.default_rng(2)
    d = 23
    n_pairs = 200000
    Sigma0, F = make_model(d, rng)

    deltas, _ = simulate_replicate_pairs(np.zeros(d), Sigma0, F, n_pairs, rng)
    V = sampling_covariance_V(Sigma0, F)
    M = np.linalg.inv(V)
    T = np.einsum("ni,ij,nj->n", deltas, M, deltas)

    mean_T = T.mean()
    var_T = T.var(ddof=1)
    se = T.std(ddof=1) / np.sqrt(n_pairs)
    ok_mean = abs(mean_T - d) < 4.0 * se
    ok_var = abs(var_T - 2 * d) < 0.05 * 2 * d

    ks = stats.kstest(T[:20000], "chi2", args=(d,))
    ok_ks = ks.pvalue > 0.001

    report("T3a E[T] == d for M = V^{-1}", ok_mean,
           "mean(T)=%.4f +/- %.4f   d=%d" % (mean_T, 4.0 * se, d))
    report("T3b Var[T] == 2d for M = V^{-1}", ok_var,
           "var(T)=%.3f   2d=%d" % (var_T, 2 * d))
    report("T3c T ~ chi2_d (KS on 20k draws)", ok_ks,
           "KS stat=%.5f  p=%.4f" % (ks.statistic, ks.pvalue))


# ---------------------------------------------------------------------------
# T4 -- invariance under linear reparameterisation
# ---------------------------------------------------------------------------

def test_invariance():
    rng = np.random.default_rng(3)
    d = 23
    Sigma0, F = make_model(d, rng)
    C = conjugate_posterior_covariance(Sigma0, F)

    m_g = rng.standard_normal(d)
    m_gp = rng.standard_normal(d)
    T0 = replicate_statistic(m_g, m_gp, C, C)

    A = rng.standard_normal((d, d))
    while abs(np.linalg.det(A)) < 1e-6:
        A = rng.standard_normal((d, d))
    T1 = replicate_statistic(A @ m_g, A @ m_gp, A @ C @ A.T, A @ C @ A.T)

    ok = np.isclose(T0, T1, rtol=1e-6)
    report("T4 T invariant under theta -> A theta", ok,
           "T=%.8f   T'=%.8f   rel diff=%.2e" %
           (T0, T1, abs(T1 - T0) / max(abs(T0), 1e-12)))

    # And the Euclidean metric is NOT invariant, which is why it is
    # inadmissible. This must FAIL to be equal.
    TE0 = mahalanobis_statistic(m_g - m_gp, np.eye(d))
    TE1 = mahalanobis_statistic(A @ m_g - A @ m_gp, np.eye(d))
    ok2 = not np.isclose(TE0, TE1, rtol=1e-3)
    report("T4b Euclidean metric is NOT invariant", ok2,
           "T_I=%.4f   T_I'=%.4f" % (TE0, TE1))


# ---------------------------------------------------------------------------
# T5 -- finite posterior-sample inflation d/S
# ---------------------------------------------------------------------------

def test_finite_sample_bias():
    rng = np.random.default_rng(4)
    d = 8                      # smaller d so the bias is visible with fewer reps
    n_pairs = 40000
    S = 200                    # posterior draws per well
    Sigma0, F = make_model(d, rng)
    C = conjugate_posterior_covariance(Sigma0, F)
    p_eff = p_eff_from_trace(Sigma0, C)

    deltas, _ = simulate_replicate_pairs(np.zeros(d), Sigma0, F, n_pairs, rng)
    L = np.linalg.cholesky(C)
    # Monte Carlo error on each posterior mean: e ~ N(0, C/S)
    e_g = (rng.standard_normal((n_pairs, d)) @ L.T) / np.sqrt(S)
    e_gp = (rng.standard_normal((n_pairs, d)) @ L.T) / np.sqrt(S)
    deltas_hat = deltas + e_g - e_gp

    M = np.linalg.inv(2.0 * C)
    T_hat = np.einsum("ni,ij,nj->n", deltas_hat, M, deltas_hat)
    predicted = p_eff + d / S
    se = T_hat.std(ddof=1) / np.sqrt(n_pairs)

    ok = abs(T_hat.mean() - predicted) < 4.0 * se
    report("T5a finite-S inflation E[T_hat]=p_eff+d/S", ok,
           "mean=%.4f +/- %.4f   predicted=%.4f (p_eff=%.4f, d/S=%.4f)" %
           (T_hat.mean(), 4.0 * se, predicted, p_eff, d / S))

    corrected = T_hat.mean() - d / S
    ok2 = abs(corrected - p_eff) < 4.0 * se
    report("T5b correction recovers p_eff", ok2,
           "corrected=%.4f   p_eff=%.4f" % (corrected, p_eff))


# ---------------------------------------------------------------------------
# T6 -- limiting behaviour
# ---------------------------------------------------------------------------

def test_limits():
    rng = np.random.default_rng(5)
    d = 10
    Sigma0 = random_spd(d, rng, scale=1.0, cond_spread=0.3)

    # Sloppy limit: F -> 0. Posterior = prior, posterior mean = prior mean,
    # replicates agree EXACTLY, and p_eff -> 0.
    F_small = 1e-8 * np.eye(d)
    C_small = conjugate_posterior_covariance(Sigma0, F_small)
    p_small = p_eff_from_trace(Sigma0, C_small)
    deltas_small, _ = simulate_replicate_pairs(np.zeros(d), Sigma0, F_small,
                                               2000, rng)
    ok1 = p_small < 1e-6 and np.abs(deltas_small).max() < 1e-3
    report("T6a sloppy limit: p_eff -> 0, Delta -> 0", ok1,
           "p_eff=%.3e   max|Delta|=%.3e" %
           (p_small, np.abs(deltas_small).max()))

    # Stiff limit: F >> Sigma0^{-1}. p_eff -> d.
    F_big = 1e8 * np.eye(d)
    C_big = conjugate_posterior_covariance(Sigma0, F_big)
    p_big = p_eff_from_trace(Sigma0, C_big)
    ok2 = abs(p_big - d) < 1e-4
    report("T6b stiff limit: p_eff -> d", ok2,
           "p_eff=%.6f   d=%d" % (p_big, d))

    # Box prior helper sanity: variance of U(L,U) is (U-L)^2/12.
    lower = np.zeros(4)
    upper = np.array([1.0, 2.0, 3.0, 4.0])
    S0 = box_prior_covariance(lower, upper)
    ok3 = np.allclose(np.diag(S0), (upper - lower) ** 2 / 12.0)
    report("T6c box prior covariance = (U-L)^2/12", ok3,
           "diag=%s" % np.array2string(np.diag(S0), precision=4))


# ---------------------------------------------------------------------------
# T7 -- the loss is two-sided with minimum at p_eff
# ---------------------------------------------------------------------------

def test_loss_shape():
    p_eff = 3.0
    at_target = replicate_loss(p_eff, p_eff)
    below = replicate_loss(0.3, p_eff)
    above = replicate_loss(30.0, p_eff)
    ok = (at_target < 1e-12) and (below > 0) and (above > 0)
    report("T7 loss minimal at p_eff, penalises both sides", ok,
           "L(p_eff)=%.2e  L(0.1*p_eff)=%.4f  L(10*p_eff)=%.4f" %
           (at_target, below, above))


# ---------------------------------------------------------------------------
# T8 -- end-to-end from posterior SAMPLES (the real-data code path)
# ---------------------------------------------------------------------------

def test_end_to_end_from_samples():
    rng = np.random.default_rng(6)
    d = 12
    S = 4000
    Sigma0, F = make_model(d, rng)
    C = conjugate_posterior_covariance(Sigma0, F)
    p_eff = p_eff_from_trace(Sigma0, C)

    deltas, _ = simulate_replicate_pairs(np.zeros(d), Sigma0, F, 1, rng)
    m_g_true = deltas[0] / 2.0
    m_gp_true = -deltas[0] / 2.0

    L = np.linalg.cholesky(C)
    samples_g = m_g_true + rng.standard_normal((S, d)) @ L.T
    samples_gp = m_gp_true + rng.standard_normal((S, d)) @ L.T

    m_g, C_g, S_g = posterior_moments(samples_g)
    m_gp, C_gp, S_gp = posterior_moments(samples_gp)
    T = replicate_statistic(m_g, m_gp, C_g, C_gp,
                            n_draws=min(S_g, S_gp), correct_mc=True)
    p_hat = p_eff_from_trace(Sigma0, 0.5 * (C_g + C_gp))
    loss = replicate_loss(T, p_hat)

    ok = np.isfinite(T) and T > 0 and abs(p_hat - p_eff) < 0.5
    report("T8 end-to-end from posterior samples runs", ok,
           "T=%.4f   p_eff_hat=%.4f (true %.4f)   L_rep=%.4f" %
           (T, p_hat, p_eff, loss))


def main():
    print("=" * 74)
    print("Smoke test: replicate statistic T_gg' and null target p_eff")
    print("=" * 74)
    test_peff_consistency()
    test_null_expectation_2C()
    test_chi2_null_Vinv()
    test_invariance()
    test_finite_sample_bias()
    test_limits()
    test_loss_shape()
    test_end_to_end_from_samples()
    print("-" * 74)
    n_fail = RESULTS.count(False)
    print("%d/%d passed" % (RESULTS.count(True), len(RESULTS)))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
