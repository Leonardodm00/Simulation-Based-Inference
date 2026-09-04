"""Shared driver for the nuisance and realisation floors (plan v0.6, S2.6).

Both floors ask the same question with one factor varied and everything else
held: how much apparent mechanistic heterogeneity does a factor that is NOT
theta manufacture, measured in parameter coordinates?

    fix theta*, vary nu           -> nuisance floor      (Lever 3)
    fix theta* and nu, vary G     -> realisation floor   (new in v0.6, D17)

The answer is a covariance of posterior means, so the two differ only in which
argument is resampled. Keeping the driver in one place means the two floors
cannot drift apart in how they simulate, encode or summarise -- which matters
because they are compared against each other and against Sigma_rep in S2.7.

Common random numbers, and why they are not optional
----------------------------------------------------
Every draw in a floor holds theta* fixed and varies ONE factor. Anything else
that is stochastic -- the spike times, the realisation when it is not the
factor under study -- must be held fixed across the draws too, or its variance
is added to the floor and attributed to the factor. `_fixed_seeds` does that
by deriving one seed per held factor and reusing it for every draw.

Pure ASCII, LF only.
"""

import numpy as np
import torch

__all__ = ["posterior_means_over", "floor_summary"]


def posterior_means_over(model, spec, provider, theta_star, draws,
                         realisation_spec, base_seed, nuisance_spec=None,
                         nu_rows=None, realisation_keys=None, gap_spec=None,
                         n_draws_post=128, window_reduce="mean"):
    """Posterior means for a list of controlled draws at one fixed theta*.

    Parameters
    ----------
    draws : int
        Number of repeats.
    nu_rows : (draws, N_COMPONENTS) or None
        The nuisance vector for each draw. None means nu is held at ZERO for
        every draw (which is the identity map, so the observable is untouched).
    realisation_keys : sequence of (donor, well, subregion) or None
        The realisation identity for each draw. None means one FIXED key
        reused for every draw, so the graph is held.
    window_reduce : "mean" or "first"
        A trace is J windows; the posterior is per window. "mean" averages the
        per-window posterior means into one culture-level summary, which is
        what m_g means in S2.7.

    Returns
    -------
    (draws, d_theta) array of posterior means.
    """
    from latent_sbi_simulator import simulate_windows

    theta_star = np.asarray(theta_star, dtype=np.float64).ravel()
    means = []
    for i in range(int(draws)):
        nu_row = None if nu_rows is None else np.asarray(nu_rows[i])
        if realisation_keys is None:
            key = ("FLOOR", "W0", 0)          # held fixed across draws
        else:
            key = realisation_keys[i]
        x, _ = simulate_windows(
            spec, theta_star, provider, donor=key[0], well=key[1],
            subregion=key[2], realisation_spec=realisation_spec,
            base_seed=base_seed, nuisance_spec=nuisance_spec, nu_row=nu_row,
            gap_spec=gap_spec, rng=np.random.default_rng(base_seed))
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        with torch.no_grad():
            s = model.sample_posterior(n_draws_post, xt)   # (J, S, d)
            m = s.mean(dim=1).cpu().numpy()                # (J, d)
        means.append(m.mean(axis=0) if window_reduce == "mean" else m[0])
    return np.asarray(means, dtype=np.float64)


def floor_summary(means, param_names=None, top_k=5):
    """Covariance of the posterior means, plus its leading directions.

    The eigenvectors are the floor's subspace IN PARAMETER COORDINATES: the
    nuisance directions are discovered, not named (S2.7, "closing the loop").
    """
    means = np.atleast_2d(np.asarray(means, dtype=np.float64))
    n, d = means.shape
    if n < 2:
        raise ValueError("need at least 2 draws to form a floor covariance")
    cov = np.cov(means, rowvar=False)
    cov = np.atleast_2d(cov)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(-ev)
    ev = np.clip(ev[order], 0.0, None)
    evec = evec[:, order]
    names = list(param_names) if param_names else ["axis%d" % k
                                                   for k in range(d)]
    total = float(ev.sum())
    out = {
        "n_draws": int(n),
        "cov": cov.tolist(),
        "per_axis_sd": np.sqrt(np.clip(np.diag(cov), 0.0, None)).tolist(),
        "axes": names,
        "eigenvalues": ev.tolist(),
        "variance_share": (ev / total).tolist() if total > 0 else None,
        "directions": [],
    }
    for k in range(min(int(top_k), d)):
        v = evec[:, k]
        heavy = np.argsort(-np.abs(v))[:3]
        out["directions"].append({
            "index": k,
            "eigenvalue": float(ev[k]),
            "variance_share": float(ev[k] / total) if total > 0 else None,
            "vector": v.tolist(),
            "dominant_axes": [names[j] for j in heavy],
            "dominant_loadings": [float(v[j]) for j in heavy],
        })
    return out


def concentration_on(subset_idx, summary):
    """Fraction of the floor's total variance carried by a subset of axes.

    Used for the D17 validation: the realisation floor must CONCENTRATE on the
    connectivity-kernel axes. A trace ratio rather than an eigenvector overlap,
    because the question is about variance attribution and not about
    directions.
    """
    cov = np.asarray(summary["cov"], dtype=np.float64)
    tr = float(np.trace(cov))
    if tr <= 0:
        return 0.0
    idx = np.asarray(list(subset_idx), dtype=int)
    return float(np.trace(cov[np.ix_(idx, idx)]) / tr)
