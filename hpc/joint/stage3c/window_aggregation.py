"""The window-aggregation curve (P10) and the factorised-NPE composition.

Plan v0.6, S2.7. Windows of one subregion share theta, so a culture-level
posterior should be built from all of them rather than one. For conditionally
independent windows given theta,

    p(theta | x_1..N)  proportional to  p(theta)^{1-N} prod_i p(theta | x_i)  (C1)

which is the factorised-NPE composition.

P10, and why it is free
-----------------------
Under a COLLAPSED encoder every window of a culture maps to the same z, so
every factor in (C1) is the same density and aggregating adds exactly nothing:
the information gain is flat in N. Under a theta-informative encoder the
factors differ and the gain compounds. So the SLOPE of gain against N is a
collapse detector that needs no ground truth on real data and no labels --
though the version here uses the bench's known theta*, because that is where
the plan says to validate it before use.

How the normalising constant is handled
---------------------------------------
(C1) is only proportional, so the gain needs log Z. It is estimated by
self-normalised importance sampling with the FIRST window's posterior as the
proposal:

    Z = E_{theta ~ q(.|z_1)} [ prod_{i>=2} q(theta | z_i) / p(theta) ]     (C2)

The estimator is consistent but its variance grows with N, because the
proposal is one window's posterior and the target is N windows' worth and
therefore much narrower. `effective_sample_size` is returned with every point
on the curve, and a point whose ESS has collapsed is reported rather than
silently plotted -- a curve that bends because the estimator broke looks
exactly like a curve that bends because information saturated.

Pure ASCII, LF only.
"""

import numpy as np
import torch

__all__ = ["compose_log_posterior", "aggregation_curve", "curve_slope"]


def _log_q(model, theta, x_row):
    """log q_omega(theta | h_psi(x_row)) for a batch of thetas, one window."""
    with torch.no_grad():
        cond = x_row.unsqueeze(0).expand(theta.shape[0], -1)
        return -model.npe_loss(theta, cond)


def compose_log_posterior(model, x_windows, theta_eval, prior_log_prob_fn,
                          n_is=512, seed=0):
    """log p(theta_eval | x_1..N) under (C1), with log Z by (C2).

    Parameters
    ----------
    x_windows : (N, W) tensor, the windows of ONE culture
    theta_eval : (P, d) tensor, where to evaluate (typically theta* alone)
    prior_log_prob_fn : callable (numpy (P, d)) -> (P,) log prior density

    Returns
    -------
    dict with `log_post` (P,), `log_Z`, `ess` and `n_windows`.
    """
    N = int(x_windows.shape[0])
    theta_eval = torch.as_tensor(theta_eval, dtype=torch.float32)
    if theta_eval.dim() == 1:
        theta_eval = theta_eval.unsqueeze(0)

    # Numerator of (C1) at the evaluation points.
    lp_prior_eval = np.asarray(prior_log_prob_fn(theta_eval.numpy()),
                               dtype=np.float64)
    num = np.zeros(theta_eval.shape[0], dtype=np.float64)
    for i in range(N):
        num += _log_q(model, theta_eval, x_windows[i]).cpu().numpy()
    num -= (N - 1) * lp_prior_eval

    if N == 1:
        return {"log_post": num, "log_Z": 0.0, "ess": float(n_is),
                "n_windows": 1}

    # (C2): proposal is window 0's posterior.
    torch.manual_seed(int(seed))
    with torch.no_grad():
        prop = model.sample_posterior(int(n_is),
                                      x_windows[0].unsqueeze(0))[0]  # (S, d)
    lp_prior_prop = np.asarray(prior_log_prob_fn(prop.numpy()),
                               dtype=np.float64)
    log_w = np.zeros(prop.shape[0], dtype=np.float64)
    for i in range(1, N):
        log_w += _log_q(model, prop, x_windows[i]).cpu().numpy()
    log_w -= (N - 1) * lp_prior_prop

    finite = np.isfinite(log_w)
    if finite.sum() < 2:
        return {"log_post": np.full(theta_eval.shape[0], np.nan),
                "log_Z": float("nan"), "ess": 0.0, "n_windows": N,
                "note": "importance weights degenerate"}
    lw = log_w[finite]
    mx = lw.max()
    w = np.exp(lw - mx)
    log_Z = float(mx + np.log(w.mean()))
    ess = float((w.sum() ** 2) / (w ** 2).sum())
    return {"log_post": num - log_Z, "log_Z": log_Z, "ess": ess,
            "n_windows": N}


def aggregation_curve(model, x_windows, theta_star, prior_log_prob_fn,
                      n_list=None, n_is=512, seed=0, ess_floor=20.0):
    """Information gain at theta* against the number of windows aggregated.

    Returns a list of points, each with `n`, `gain` (nats), `ess` and a
    `trusted` flag. `gain` is log p(theta* | x_1..n) - log p(theta*), i.e. how
    many nats about this culture's own parameters the aggregate carries.
    """
    N = int(x_windows.shape[0])
    n_list = n_list or sorted({1, 2, 4, max(1, N // 2), N})
    n_list = [n for n in sorted(set(int(n) for n in n_list)) if 1 <= n <= N]
    th = torch.as_tensor(np.asarray(theta_star, dtype=np.float32)).reshape(1, -1)
    lp0 = float(np.asarray(prior_log_prob_fn(th.numpy()))[0])

    out = []
    for n in n_list:
        r = compose_log_posterior(model, x_windows[:n], th, prior_log_prob_fn,
                                  n_is=n_is, seed=seed)
        gain = float(r["log_post"][0] - lp0)
        out.append({"n": n, "gain": gain, "ess": r["ess"],
                    "log_Z": r["log_Z"],
                    "trusted": bool(np.isfinite(gain) and r["ess"] >= ess_floor)})
    return out


def curve_slope(points, trusted_only=True):
    """Least-squares slope of gain against log(n), plus the P10 reading.

    log(n) rather than n: for independent windows the gain grows roughly
    linearly in n only while the posterior is far from the prior, and the
    log axis is the one on which "flat means collapse" is a clean statement
    across scales.
    """
    pts = [p for p in points if (p["trusted"] or not trusted_only)]
    if len(pts) < 2:
        return {"slope": float("nan"), "n_points": len(pts),
                "reading": "too few trusted points to fit a slope; the "
                           "importance-sampling ESS collapsed"}
    x = np.log(np.asarray([p["n"] for p in pts], dtype=np.float64))
    y = np.asarray([p["gain"] for p in pts], dtype=np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    span = float(y.max() - y.min())
    if abs(slope) < 0.05 and span < 0.1:
        reading = ("FLAT: aggregating windows adds nothing. Under a collapsed "
                   "encoder every window of a culture gives the same z, so "
                   "this is the collapse signature (P10).")
    elif slope > 0:
        reading = ("RISING at %.3f nats per e-fold of windows: the windows "
                   "carry independent information and the encoder is not "
                   "collapsed." % slope)
    else:
        reading = ("FALLING at %.3f nats per e-fold: aggregating makes the "
                   "posterior WORSE at theta*, which means the composition's "
                   "independence assumption is violated -- windows of one "
                   "subregion share a realisation and are exchangeable but "
                   "not independent (S2.7)." % slope)
    return {"slope": float(slope), "n_points": len(pts), "span": span,
            "reading": reading}
