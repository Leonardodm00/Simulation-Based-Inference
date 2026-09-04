"""Per-arm diagnostics for Stage 3 (plan v0.6, S3 and Stage 3).

Everything here operates on a trained `JointDSNNPE` plus a held-out split. It
computes, and nothing else -- no training, no I/O, no plotting.

What is and is not reimplemented
--------------------------------
* The information spectrum is the REPO's `npe_diagnostics.information_spectrum`
  whenever the SBI repo is importable (`SBI_HPC_DIR`), because `p_eff` must be
  the same number in the diagnostic and in the replicate target or S2.5b's
  identification is vacuous. `p_eff_from_spectrum_result` only sums what that
  function returns. A local fallback exists for the bench, and it is LABELLED
  as a fallback in the output rather than silently substituted.
* `replicate_statistic.py` supplies T and p_eff from the posterior side.
  Cross-checking the two routes against each other is the point of J13 and is
  reported here per arm as `p_eff_gap`.

The per-axis quantity is CONTRACTION, not per-axis information gain. Gain per
axis needs the flow's marginals, which a normalizing flow does not give in
closed form; estimating each one by KDE across 26 axes and every held-out row
would dominate the cost of Stage 3 and carry an unquantified bias. Contraction
is defined from posterior samples alone, is exact up to Monte Carlo error, and
answers the same question ("did the data narrow this axis") without pretending
to a decomposition it cannot support. Flagged here rather than in a footnote
because the plan's P1/P2 are stated in terms of per-axis GAIN and this is a
deviation from them.

Pure ASCII, LF only.
"""

import os
import sys

import numpy as np
import torch

__all__ = ["per_row_nll", "mc_prior_floor", "information_gain",
           "per_axis_contraction", "effective_rank", "cluster_scores",
           "replicate_report", "load_repo_diagnostics",
           "p_eff_from_spectrum_result", "local_information_spectrum"]


# ---------------------------------------------------------------------------
# NPE score
# ---------------------------------------------------------------------------

def per_row_nll(model, theta, x, batch_size=256):
    """(N,) array of -log q_omega(theta_i | h_psi(x_i)), in nats.

    Persisted per run: without the PER-ROW array the paired bootstrap of S2.4a
    cannot be built after the fact, only re-run.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, theta.shape[0], batch_size):
            out.append(model.npe_loss(theta[i:i + batch_size],
                                      x[i:i + batch_size]).cpu().numpy())
    model.train()
    return np.concatenate(out).astype(np.float64)


def mc_prior_floor(prior_log_prob_fn, theta):
    """L_0 = -E[log p(theta)], estimated on the held-out thetas themselves.

    The bench prior is a MIXTURE (eq. 7), so the analytic box floor does not
    apply and Monte Carlo is required. Using the held-out thetas rather than
    fresh prior draws makes L_0 and L averages over the same points, so
    `information_gain` is a paired difference and its Monte Carlo error partly
    cancels.
    """
    lp = np.asarray(prior_log_prob_fn(theta), dtype=np.float64)
    finite = np.isfinite(lp)
    if not finite.all():
        raise ValueError("%d of %d held-out thetas have zero prior density; "
                         "the split or the gap shift has put rows outside the "
                         "prior box" % (int((~finite).sum()), lp.size))
    return float(-lp.mean())


def information_gain(nll, prior_floor):
    """Delta_hat = L_0 - L, in nats/row. Positive means informative."""
    return float(prior_floor - np.mean(nll))


# ---------------------------------------------------------------------------
# Posterior geometry
# ---------------------------------------------------------------------------

def per_axis_contraction(samples, prior_std):
    """kappa_k = 1 - Var(theta_k | x) / Var_prior(theta_k), per axis.

    Parameters
    ----------
    samples : (N, S, d) posterior draws
    prior_std : (d,) prior standard deviations

    Returns (d,) array in (-inf, 1]. Negative means the posterior is WIDER
    than the prior on that axis, which is a finding rather than a bug.
    """
    s = np.asarray(samples, dtype=np.float64)
    post_var = s.var(axis=1, ddof=1).mean(axis=0)
    prior_var = np.asarray(prior_std, dtype=np.float64) ** 2
    return 1.0 - post_var / prior_var


def effective_rank(z):
    """Participation-ratio effective rank of an embedding cloud.

    r_eff = (sum_j lambda_j)^2 / sum_j lambda_j^2 for the eigenvalues of the
    covariance. Equals 1 exactly for a rank-one cloud -- the collapse value
    C - 1 at C = 2, which is what the r2 encoder measures at [KB].
    """
    z = np.asarray(z, dtype=np.float64)
    c = np.cov(z - z.mean(0, keepdims=True), rowvar=False)
    ev = np.linalg.eigvalsh(np.atleast_2d(c))
    ev = np.clip(ev, 0.0, None)
    denom = float((ev ** 2).sum())
    if denom <= 0.0:
        return 1.0
    return float((ev.sum() ** 2) / denom)


def cluster_scores(z, labels, seed=0):
    """ARI and silhouette of the embedding against the class labels.

    Logged, never optimised -- a clustering statistic as objective rewards
    label-only codes, which is the whole thing S2.3 warns about.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, silhouette_score
    except ImportError:
        return {"ari": None, "silhouette": None, "note": "sklearn absent"}
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    n_classes = len(np.unique(labels))
    if n_classes < 2 or z.shape[0] <= n_classes:
        return {"ari": None, "silhouette": None, "note": "too few rows"}
    km = KMeans(n_clusters=n_classes, n_init=10, random_state=seed).fit(z)
    out = {"ari": float(adjusted_rand_score(labels, km.labels_))}
    try:
        out["silhouette"] = float(silhouette_score(z, labels, metric="cosine"))
    except ValueError:
        out["silhouette"] = None
    return out


# ---------------------------------------------------------------------------
# The information spectrum, and p_eff
# ---------------------------------------------------------------------------

def load_repo_diagnostics(sbi_hpc_dir=None):
    """Import the repo's npe_diagnostics, or return None."""
    d = sbi_hpc_dir or os.environ.get("SBI_HPC_DIR")
    if not d or not os.path.isfile(os.path.join(d, "npe_diagnostics.py")):
        return None
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import npe_diagnostics
        return npe_diagnostics
    except ImportError:
        return None


def local_information_spectrum(theta_true, posterior_samples):
    """Fallback: generalised eigenvalues of Cov_i(m_i) against Cov(theta*).

    Same object as the repo's, computed locally so the bench can run without
    the SBI repo checked out. Any output built from this is tagged
    `spectrum_source = "local-fallback"`, because two numbers that are supposed
    to be the same number must not become the same number by coincidence of
    two implementations.
    """
    theta_true = np.asarray(theta_true, dtype=np.float64)
    m = np.asarray(posterior_samples, dtype=np.float64).mean(axis=1)
    cov_m = np.cov(m, rowvar=False)
    cov_t = np.cov(theta_true, rowvar=False)
    d = theta_true.shape[1]
    jitter = 1e-10 * float(np.trace(np.atleast_2d(cov_t))) / max(d, 1)
    from scipy.linalg import eigh
    ev = eigh(np.atleast_2d(cov_m),
              np.atleast_2d(cov_t) + jitter * np.eye(d), eigvals_only=True)
    return np.clip(np.sort(ev)[::-1], 0.0, None)


def spectrum_is_estimable(n_rows, d, factor=4):
    """Whether Cov(theta*) can support a generalised eigenproblem at all.

    The spectrum is the generalised eigenvalues of Cov_i(m_i) against
    Cov(theta*). Both are d x d and estimated from n_rows points, so with
    n_rows close to d the DENOMINATOR is near-singular and the eigenvalues
    diverge -- measured on a 14-row report split at d = 6 they came back at
    1e300, which is not a large p_eff but a rank deficiency wearing one as a
    disguise. Refuse rather than report it.
    """
    return int(n_rows) >= max(int(factor) * int(d), int(d) + 2)


def p_eff_from_spectrum_result(result, n_rows=None, d=None):
    """Sum the generalised eigenvalues: p_eff, eqs. (10)-(11').

    Accepts either the repo's ContractionResult or a bare array. The repo's
    function COUNTS eigenvalues above a threshold; p_eff is their SUM, which
    is a different number, and conflating the two is exactly the mistake
    S3.9 warns against.

    Eigenvalues are clipped to [0, 1] because they are FRACTIONS of prior
    variance explained along each direction -- the repo's own docstring says
    so. A value above 1 is not information, it is estimation noise in a
    near-singular denominator, and summing it inflates p_eff without limit.

    Returns (p_eff, note). p_eff is None when the spectrum is not estimable.
    """
    ev = getattr(result, "eigenvalues", None)
    if ev is None:
        ev = getattr(result, "generalised_eigenvalues", None)
    if ev is None:
        ev = result
    ev = np.asarray(ev, dtype=np.float64)

    if n_rows is not None and d is not None and \
            not spectrum_is_estimable(n_rows, d):
        return None, ("not estimable: %d rows for d = %d; need >= %d"
                      % (n_rows, d, max(4 * d, d + 2)))
    n_over = int((ev > 1.0 + 1e-6).sum())
    note = "ok" if n_over == 0 else ("%d of %d eigenvalues exceeded 1 and were "
                                     "clipped" % (n_over, ev.size))
    return float(np.clip(ev, 0.0, 1.0).sum()), note


# ---------------------------------------------------------------------------
# Replicate consistency, reported for EVERY arm whether trained on or not
# ---------------------------------------------------------------------------

def replicate_report(model, x_g, x_gp, Sigma0, n_draws=256, correct_mc=True):
    """T, p_eff and the per-direction decomposition on same-donor pairs."""
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "stage2")))
    from joint_losses import ReplicateConsistencyLoss

    crit = ReplicateConsistencyLoss(Sigma0, n_draws=n_draws,
                                    correct_mc=correct_mc)
    model.eval()
    with torch.no_grad():
        sg = model.sample_posterior(n_draws, x_g)
        sgp = model.sample_posterior(n_draws, x_gp)
        loss, info = crit(sg, sgp)
    model.train()
    T = info["T"].cpu().numpy()
    p_eff = info["p_eff"].cpu().numpy()
    # T after the d/S correction can come out NEGATIVE when the raw statistic
    # is already at the Monte Carlo floor -- i.e. when the two posterior means
    # are identical because the flow is ignoring its conditioner. That is the
    # collapse signature, not a numerical accident, so it is COUNTED rather
    # than clamped away: a non-zero count here with a converged flow is a G1
    # failure and should be read alongside T_mean, which the clamp inside
    # replicate_loss would otherwise hide.
    n_nonpositive = int((T <= 0.0).sum())
    return {
        "n_T_nonpositive": n_nonpositive,
        "T_mean": float(T.mean()),
        "T": T.tolist(),
        "p_eff_mean": float(p_eff.mean()),
        "p_eff": p_eff.tolist(),
        "log_ratio_mean": float(np.mean(np.log(np.maximum(T, 1e-12))
                                        - np.log(np.maximum(p_eff, 1e-12)))),
        "n_p_eff_invalid": int(info["n_p_eff_invalid"]),
        "loss": float(loss),
        "per_direction_mean": info["per_direction"].mean(0).cpu().numpy().tolist(),
    }
