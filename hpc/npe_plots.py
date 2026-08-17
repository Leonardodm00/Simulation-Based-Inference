#!/usr/bin/env python3
"""
npe_plots.py -- figures for every diagnostic layer.

Scope boundary: this module only draws. It computes nothing that
npe_diagnostics does not already return, so a figure can never disagree
with the number it is supposed to depict.

WHAT GETS PLOTTED AND WHY
-------------------------
Files are numbered so an `ls` puts them in reading order, which is also the
order of decreasing consequence: if figure 01 is bad, nothing after it
matters.

  01_embedding_overlap   the go/no-go gate. Three panels: the MMD null with
                         the observed value marked, geodesic
                         nearest-neighbour distances, and a PCA projection
                         of both clouds. The PCA panel is the one to look at
                         first -- if the real points sit off to one side,
                         stop and fix the simulator.
  02_sbc_ecdf            all axes overlaid on one rank-ECDF plot with a
                         simultaneous KS band. Deviations outside the band
                         mean miscalibration; the SHAPE says which kind.
  03_sbc_histograms      per-axis rank histograms. U-shaped = overconfident,
                         dome = underconfident, sloped = biased.
  04_coverage_pp         nominal vs empirical coverage. Below the diagonal
                         is overconfidence, which is the dangerous side.
  05_data_dependent_sbc  the quantities that can see a data-ignoring
                         posterior when 02 and 04 cannot.
  06_tarp                coverage from random reference points.
  07_contraction         per-axis contraction, sorted. THE scientific
                         figure: it says which parameters the data actually
                         informs.
  08_information_spectrum eigenvalue decay on a log axis. Reads as a
                         sloppiness spectrum -- a long tail means most
                         parameter combinations are unconstrained.
  09_recovery            true vs inferred, per axis, with 68% intervals.
                         Points on the diagonal with small bars = recovered;
                         a flat cloud = that axis is at the prior.
  10_posterior_pairs     2D marginals for the best-constrained axes, with
                         the truth marked. Reveals ridges and multimodality
                         that 1D marginals hide.
  11_ensemble_spread     disagreement between ensemble members. Wide
                         disagreement is the honest uncertainty that a
                         single network would have hidden.

ASCII-only by policy (HPC transfer safety), including axis labels: "theta"
rather than the Greek letter, so a figure caption cannot reintroduce a
non-ASCII byte into a source file.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["save_all_plots", "plots_available"]

_DPI = 130


def plots_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _mpl():
    """Import matplotlib with a headless backend.

    Agg must be selected BEFORE pyplot is imported: on a compute node there
    is no display, and the default backend raises at import time rather than
    at draw time, which makes the traceback point at the wrong place.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_dir: str, name: str, saved: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    saved.append(path)


def _ks_band(n: int, alpha: float = 0.05) -> float:
    """Half-width of the simultaneous KS band for an ECDF of n points."""
    c = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}.get(alpha, 1.358)
    return c / np.sqrt(max(n, 1))


# ---------------------------------------------------------------------------
# 01 -- embedding overlap
# ---------------------------------------------------------------------------

def plot_embedding_overlap(overlap, z_sim, z_real, out_dir, saved,
                           tag: str = "", null_samples=None):
    null_samples = null_samples if null_samples is not None else overlap.null_samples
    plt = _mpl()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # panel 1 -- MMD against its null
    if null_samples is not None and len(null_samples):
        ax[0].hist(null_samples, bins=40, color="0.75", edgecolor="none",
                   label="null (simulated vs simulated)")
    ax[0].axvline(overlap.null_quantile_95, color="0.35", ls="--",
                  label="null 95th pct")
    ax[0].axvline(overlap.mmd, color="crimson", lw=2.2,
                  label="observed (real)")
    ax[0].set_xlabel("MMD"); ax[0].set_ylabel("count")
    ax[0].set_title("MMD vs null   p = %.4f" % overlap.p_value)
    ax[0].legend(fontsize=7)

    # panel 2 -- geodesic nearest-neighbour distances, drawn as ECDFs.
    #
    # Histograms are the obvious choice and the wrong one: if any query
    # point coincides with a simulated point the distance is exactly zero,
    # and a density histogram turns that delta into a spike tens of
    # thousands high that flattens everything else into the axis. An ECDF
    # shows the same comparison and degrades gracefully.
    if overlap.nn_geodesic_real is not None:
        for arr, col, lab in ((overlap.nn_geodesic_sim, "steelblue", "sim -> sim"),
                              (overlap.nn_geodesic_real, "crimson", "real -> sim")):
            x = np.sort(np.asarray(arr))
            y = np.arange(1, x.size + 1) / x.size
            ax[1].step(x, y, where="post", color=col, lw=1.8, label=lab)
        q95 = float(np.percentile(overlap.nn_geodesic_sim, 95))
        ax[1].axvline(q95, color="0.4", ls="--", lw=1,
                      label="sim 95th pct = %.3f" % q95)
        ax[1].set_xlabel("geodesic NN distance (rad)")
        ax[1].set_ylabel("ECDF")
        ax[1].set_title("distance to the simulated cloud")
        ax[1].legend(fontsize=7, loc="lower right")
        ax[1].set_ylim(0, 1.02)
    else:
        ax[1].text(0.5, 0.5, "embeddings not unit-norm;\ngeodesic distance "
                             "not meaningful", ha="center", va="center",
                   transform=ax[1].transAxes, fontsize=9)
        ax[1].set_axis_off()

    # panel 3 -- DISCRIMINATIVE projection, not plain PCA.
    #
    # Plain PCA on the simulated cloud picks the directions of greatest
    # SIMULATED variance, which need not be the direction in which the real
    # data differs. In a first version of this figure a query set the gate
    # correctly rejected looked perfectly well mixed in the top two PCs,
    # because the shift lay in a low-variance direction. That is worse than
    # no picture at all.
    #
    # Horizontal axis is therefore the whitened difference of means -- the
    # single most separating direction. Vertical is the leading simulated PC
    # orthogonal to it. If the two clouds overlap HERE, they overlap
    # everywhere; caveat below on why that is optimistic in the other
    # direction.
    zs = np.asarray(z_sim, dtype=np.float64)
    zr = np.atleast_2d(np.asarray(z_real, dtype=np.float64))
    mu = zs.mean(axis=0)
    Zc = zs - mu
    C = np.cov(Zc, rowvar=False) + 1e-8 * np.eye(zs.shape[1])
    w, V = np.linalg.eigh(C)
    Winv = V @ np.diag(np.maximum(w, 1e-12) ** -0.5) @ V.T
    d = zr.mean(axis=0) - mu
    a1 = Winv @ d
    n1 = np.linalg.norm(a1)
    a1 = a1 / n1 if n1 > 1e-12 else np.eye(zs.shape[1])[0]
    resid = Zc @ Winv
    resid = resid - np.outer(resid @ a1, a1)
    _, _, Vt2 = np.linalg.svd(resid, full_matrices=False)
    a2 = Vt2[0]

    rng2 = np.random.default_rng(0)
    sub = zs[rng2.choice(zs.shape[0], min(3000, zs.shape[0]), replace=False)]
    A = (sub - mu) @ Winv
    B = (zr - mu) @ Winv
    ax[2].scatter(A @ a1, A @ a2, s=4, alpha=0.25, color="steelblue",
                  label="simulated", rasterized=True)
    ax[2].scatter(B @ a1, B @ a2, s=34, color="crimson", edgecolor="k",
                  linewidth=0.4, label="real / query", zorder=3)
    ax[2].set_xlabel("most separating direction (whitened)")
    ax[2].set_ylabel("leading orthogonal direction")
    ax[2].set_title("summary space, discriminative view")
    ax[2].legend(fontsize=7)
    ax[2].text(0.02, 0.02,
               "axis chosen to separate: overlap here is strong evidence,\n"
               "separation here is weak evidence (it can overfit at small n)",
               fontsize=5.5, color="0.35", transform=ax[2].transAxes)

    verdict = "REJECT: simulation gap" if overlap.rejects else "no gap detected"
    fig.suptitle("Embedding overlap gate%s   --   %s"
                 % ((" [%s]" % tag) if tag else "", verdict),
                 fontsize=11,
                 color="crimson" if overlap.rejects else "darkgreen")
    fig.tight_layout()
    _save(fig, out_dir, "01_embedding_overlap%s" % (("_" + tag) if tag else ""), saved)


# ---------------------------------------------------------------------------
# 02, 03, 05 -- rank-based calibration
# ---------------------------------------------------------------------------

def _ecdf_panel(ax, results, title, max_lines: int = 40):
    n = len(results[0].ranks)
    band = _ks_band(n)
    grid = np.linspace(0, 1, 200)
    ax.fill_between(grid, np.clip(grid - band, 0, 1), np.clip(grid + band, 0, 1),
                    color="0.85", label="95% simultaneous band")
    ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1)
    for r in results[:max_lines]:
        u = np.sort((r.ranks + 0.5) / (r.n_draws + 1))
        y = np.arange(1, u.size + 1) / u.size
        col = "crimson" if not r.passes else "steelblue"
        ax.plot(u, y, lw=1.6 if not r.passes else 0.7,
                alpha=0.95 if not r.passes else 0.45, color=col)
    ax.set_xlabel("fractional rank"); ax.set_ylabel("ECDF")
    ax.set_title(title); ax.legend(fontsize=7, loc="upper left")


def plot_sbc_ecdf(sbc, out_dir, saved, family=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    n_fail = sum(1 for r in sbc if not r.passes)
    sub = "%d/%d axes outside the band (red)" % (n_fail, len(sbc))
    if family is not None:
        sub += "  |  Holm-Bonferroni: %d reject" % family.n_rejected
    _ecdf_panel(ax, sbc, "Marginal SBC\n" + sub)
    fig.tight_layout()
    _save(fig, out_dir, "02_sbc_ecdf", saved)


def plot_sbc_histograms(sbc, out_dir, saved, max_axes: int = 30):
    plt = _mpl()
    res = sbc[:max_axes]
    ncol = 6
    nrow = int(np.ceil(len(res) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 1.9 * nrow),
                             squeeze=False)
    for i, r in enumerate(res):
        ax = axes[i // ncol][i % ncol]
        u = (r.ranks + 0.5) / (r.n_draws + 1)
        ax.hist(u, bins=14, range=(0, 1),
                color="crimson" if not r.passes else "steelblue", alpha=0.85)
        ax.axhline(len(u) / 14.0, color="0.35", ls="--", lw=0.9)
        ax.set_title("%s\np=%.3f" % (r.name[:14], r.ks_pvalue), fontsize=7)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([])
        ax.tick_params(labelsize=6)
    for j in range(len(res), nrow * ncol):
        axes[j // ncol][j % ncol].set_axis_off()
    fig.suptitle("SBC rank histograms -- flat is correct; U-shape means "
                 "overconfident, dome means underconfident", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "03_sbc_histograms", saved)


def plot_coverage_pp(coverage, out_dir, saved):
    """Nominal vs empirical coverage.

    CAREFUL WITH THE PARAMETRISATION -- an earlier version of this figure
    plotted the rank ECDF and labelled the y-axis "empirical coverage",
    which is 1 MINUS coverage. The curve then sat above the diagonal while
    the verdict said overconfident, and the annotation contradicted both.

    The relationship: theta* lies inside the (1-alpha) highest-density
    region exactly when the fraction u of posterior draws with lower
    density than theta* exceeds alpha. So, writing c = 1 - alpha for the
    nominal credibility level,

        empirical_coverage(c) = mean_i( u_i > 1 - c ),  for each fixed c.

    With this parametrisation, and only with it, a curve BELOW the diagonal
    means empirical coverage falls short of nominal -- overconfidence, the
    dangerous direction.
    """
    plt = _mpl()
    u = (coverage.ranks + 0.5) / (coverage.n_draws + 1)
    c = np.linspace(0.0, 1.0, 201)
    emp = np.array([np.mean(u > (1.0 - ci)) for ci in c])

    # Binomial standard error at each level, for a visual sense of scale.
    n = u.size
    se = np.sqrt(np.maximum(emp * (1.0 - emp), 1e-12) / n)

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.fill_between(c, np.clip(c - 1.96 * np.sqrt(c * (1 - c) / n), 0, 1),
                    np.clip(c + 1.96 * np.sqrt(c * (1 - c) / n), 0, 1),
                    color="0.85", label="95% band under the null")
    ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1, label="ideal")
    col = "crimson" if not coverage.passes else "steelblue"
    ax.plot(c, emp, color=col, lw=2.2, label="observed")
    ax.fill_between(c, np.clip(emp - 1.96 * se, 0, 1),
                    np.clip(emp + 1.96 * se, 0, 1), color=col, alpha=0.18)

    ax.set_xlabel("nominal credibility level  $1-\\alpha$")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Expected coverage (joint)\nKS p=%.3g -- %s"
                 % (coverage.ks_pvalue, "PASS" if coverage.passes else "FAIL"),
                 fontsize=10)
    ax.text(0.50, 0.13, "below diagonal = OVERCONFIDENT\n"
                        "(credible regions too small)",
            fontsize=7.5, color="crimson", ha="center", transform=ax.transAxes)
    ax.text(0.97, 0.93, "above diagonal = CONSERVATIVE",
            fontsize=7.5, color="0.35", ha="right", transform=ax.transAxes)

    # Quote the coverage actually delivered at the levels people report.
    lines = []
    for lvl in (0.68, 0.90, 0.95):
        got = float(np.mean(u > (1.0 - lvl)))
        lines.append("nominal %.0f%% -> %.0f%%" % (100 * lvl, 100 * got))
    ax.text(0.03, 0.55, "\n".join(lines), fontsize=8, transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir, "04_coverage_pp", saved)


def plot_data_dependent(ddsbc, out_dir, saved, family=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    n_fail = sum(1 for r in ddsbc if not r.passes)
    _ecdf_panel(ax, ddsbc,
                "Data-dependent SBC,  f(theta, z) = theta^T W z\n"
                "%d/%d quantities reject -- the only calibration layer that\n"
                "can see a posterior which ignores the observation"
                % (n_fail, len(ddsbc)))
    fig.tight_layout()
    _save(fig, out_dir, "05_data_dependent_sbc", saved)


def plot_tarp(tarp_res, out_dir, saved):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    band = _ks_band(tarp_res.ecdf_x.size)
    ax.fill_between(tarp_res.ecdf_x,
                    np.clip(tarp_res.ecdf_x - band, 0, 1),
                    np.clip(tarp_res.ecdf_x + band, 0, 1),
                    color="0.85", label="95% band")
    ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1)
    ax.plot(tarp_res.ecdf_x, tarp_res.ecdf_y,
            color="crimson" if not tarp_res.passes else "steelblue", lw=2)
    ax.set_xlabel("expected coverage"); ax.set_ylabel("observed coverage")
    ax.set_title("TARP\nKS p=%.3g -- %s"
                 % (tarp_res.ks_pvalue, "PASS" if tarp_res.passes else "FAIL"))
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir, "06_tarp", saved)


# ---------------------------------------------------------------------------
# 07, 08 -- informativeness
# ---------------------------------------------------------------------------

def plot_contraction(con, out_dir, saved):
    plt = _mpl()
    order = np.argsort(-con.contraction)
    vals = con.contraction[order]
    names = [con.param_names[k] for k in order]
    colors = ["darkgreen" if v > 0.5 else ("goldenrod" if v > 0.05 else "crimson")
              for v in vals]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.32 * len(vals)), 4.6))
    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.axhline(0.05, color="0.35", ls="--", lw=1)
    ax.text(len(vals) * 0.72, 0.075, "below: posterior ~= prior",
            fontsize=7, color="crimson")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("contraction  1 - Var[post]/Var[prior]")
    ax.set_ylim(min(0.0, float(vals.min()) - 0.05), 1.0)
    n_flat = int(np.sum(vals < 0.05))
    ax.set_title("Posterior contraction per axis -- %d/%d axes uninformed "
                 "(mean %.3f)" % (n_flat, len(vals), float(np.mean(vals))),
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "07_contraction", saved)


def plot_information_spectrum(con, out_dir, saved, embedding_dim=None):
    if con.eigenvalues is None:
        return
    plt = _mpl()
    ev = np.maximum(con.eigenvalues, 1e-12)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.semilogy(np.arange(1, ev.size + 1), ev, "o-", color="steelblue", ms=4)
    ax.axhline(0.05, color="0.4", ls="--", lw=1, label="threshold 0.05")
    if embedding_dim is not None:
        ax.axvline(embedding_dim - 1, color="crimson", ls=":", lw=1.4,
                   label="E - 1 = %d (dof in z)" % (embedding_dim - 1))
    ax.set_xlabel("direction index"); ax.set_ylabel("fraction of prior variance explained")
    ax.set_title("Information spectrum -- effective rank %s\n"
                 "descriptive only: a nonlinear estimator may exceed E-1"
                 % con.effective_rank, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out_dir, "08_information_spectrum", saved)


# ---------------------------------------------------------------------------
# 09, 10, 11 -- posterior inspection
# ---------------------------------------------------------------------------

def plot_recovery(theta_true, ps, con, out_dir, saved, max_axes: int = 12):
    plt = _mpl()
    order = np.argsort(-con.contraction)[:max_axes]
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow),
                             squeeze=False)
    for i, k in enumerate(order):
        ax = axes[i // ncol][i % ncol]
        t = theta_true[:, k]
        # Centre on the MEDIAN, not the mean. The bars are the 16th-84th
        # percentile interval, and for a skewed marginal the mean can fall
        # outside it -- which makes yerr negative and kills the whole figure
        # with a ValueError. The median is inside the interval by
        # construction. R2 below is still computed on the posterior mean,
        # since that is the Bayes estimator under squared error.
        m = np.median(ps[:, :, k], axis=1)
        lo = np.percentile(ps[:, :, k], 16, axis=1)
        hi = np.percentile(ps[:, :, k], 84, axis=1)
        lo = np.minimum(lo, m)
        hi = np.maximum(hi, m)
        ax.errorbar(t, m, yerr=[m - lo, hi - m], fmt="o", ms=2.4, lw=0.5,
                    alpha=0.55, color="steelblue", ecolor="0.7")
        lim = [min(t.min(), m.min()), max(t.max(), m.max())]
        ax.plot(lim, lim, color="crimson", ls="--", lw=1)
        mean_k = ps[:, :, k].mean(axis=1)
        ss_res = float(np.sum((mean_k - t) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.set_title("%s\ncontraction %.2f, R2 %.2f"
                     % (con.param_names[k][:16], con.contraction[k], r2), fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(len(order), nrow * ncol):
        axes[j // ncol][j % ncol].set_axis_off()
    fig.suptitle("Recovery: true (x) vs posterior mean (y) with 68% interval\n"
                 "best-constrained axes first; a flat cloud means that axis "
                 "is at the prior", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "09_recovery", saved)


def plot_posterior_pairs(theta_true, ps, con, out_dir, saved,
                         obs_index: int = 0, n_axes: int = 5):
    plt = _mpl()
    order = np.argsort(-con.contraction)[:n_axes]
    d = len(order)
    fig, axes = plt.subplots(d, d, figsize=(2.0 * d, 2.0 * d), squeeze=False)
    s = ps[obs_index]
    t = theta_true[obs_index]
    for i in range(d):
        for j in range(d):
            ax = axes[i][j]
            ki, kj = order[i], order[j]
            if i == j:
                ax.hist(s[:, ki], bins=26, color="steelblue", alpha=0.85)
                ax.axvline(t[ki], color="crimson", lw=1.6)
            elif i > j:
                ax.scatter(s[:, kj], s[:, ki], s=3, alpha=0.3,
                           color="steelblue", rasterized=True)
                ax.plot(t[kj], t[ki], "*", color="crimson", ms=11,
                        markeredgecolor="k", markeredgewidth=0.4)
            else:
                ax.set_axis_off(); continue
            ax.tick_params(labelsize=5)
            if i == d - 1:
                ax.set_xlabel(con.param_names[order[j]][:12], fontsize=6)
            if j == 0 and i > 0:
                ax.set_ylabel(con.param_names[order[i]][:12], fontsize=6)
    fig.suptitle("Posterior pair plot, observation %d (red = truth)\n"
                 "ridges and multiple modes are visible here and nowhere else"
                 % obs_index, fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "10_posterior_pairs", saved)


def plot_ensemble_spread(member_samples, con, out_dir, saved, n_axes: int = 8):
    """member_samples: list of (M, p) arrays, one per ensemble member."""
    if member_samples is None or len(member_samples) < 2:
        return
    plt = _mpl()
    order = np.argsort(-con.contraction)[:n_axes]
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow),
                             squeeze=False)
    for i, k in enumerate(order):
        ax = axes[i // ncol][i % ncol]
        for j, ms in enumerate(member_samples):
            ax.hist(ms[:, k], bins=30, histtype="step", lw=1.1,
                    density=True, label="member %d" % j if i == 0 else None)
        ax.set_title(con.param_names[k][:16], fontsize=7)
        ax.tick_params(labelsize=6); ax.set_yticks([])
    for j in range(len(order), nrow * ncol):
        axes[j // ncol][j % ncol].set_axis_off()
    if len(member_samples) <= 10:
        axes[0][0].legend(fontsize=5)
    fig.suptitle("Ensemble member disagreement, one observation\n"
                 "spread between members IS the extra uncertainty a single "
                 "network would have hidden", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "11_ensemble_spread", saved)


# ---------------------------------------------------------------------------

def save_all_plots(out_dir: str,
                   sbc=None, coverage=None, ddsbc=None, tarp_res=None,
                   contraction=None, overlap=None, overlap_shifted=None,
                   z_sim=None, z_real=None, z_real_shifted=None,
                   theta_true=None, posterior_samples=None,
                   member_samples=None, embedding_dim=None,
                   sbc_family=None, ddsbc_family=None) -> List[str]:
    """Draw every figure for which the inputs were supplied.

    Missing inputs are skipped silently rather than raising: a diagnostics
    run that produced fewer quantities should still get the figures it can
    support, and a plotting failure must never destroy a completed run.
    """
    saved: List[str] = []
    if not plots_available():
        print("  matplotlib not available; skipping figures")
        return saved
    os.makedirs(out_dir, exist_ok=True)

    jobs = [
        (overlap is not None and z_sim is not None and z_real is not None,
         lambda: plot_embedding_overlap(overlap, z_sim, z_real, out_dir, saved)),
        (overlap_shifted is not None and z_sim is not None and z_real_shifted is not None,
         lambda: plot_embedding_overlap(overlap_shifted, z_sim, z_real_shifted,
                                        out_dir, saved, tag="shifted")),
        (sbc is not None, lambda: plot_sbc_ecdf(sbc, out_dir, saved, sbc_family)),
        (sbc is not None, lambda: plot_sbc_histograms(sbc, out_dir, saved)),
        (coverage is not None, lambda: plot_coverage_pp(coverage, out_dir, saved)),
        (ddsbc is not None, lambda: plot_data_dependent(ddsbc, out_dir, saved,
                                                        ddsbc_family)),
        (tarp_res is not None, lambda: plot_tarp(tarp_res, out_dir, saved)),
        (contraction is not None, lambda: plot_contraction(contraction, out_dir, saved)),
        (contraction is not None,
         lambda: plot_information_spectrum(contraction, out_dir, saved, embedding_dim)),
        (contraction is not None and theta_true is not None and posterior_samples is not None,
         lambda: plot_recovery(theta_true, posterior_samples, contraction, out_dir, saved)),
        (contraction is not None and theta_true is not None and posterior_samples is not None,
         lambda: plot_posterior_pairs(theta_true, posterior_samples, contraction,
                                      out_dir, saved)),
        (contraction is not None and member_samples is not None,
         lambda: plot_ensemble_spread(member_samples, contraction, out_dir, saved)),
    ]
    for cond, fn in jobs:
        if not cond:
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print("  figure failed (%s: %s)" % (type(exc).__name__, exc))
    return saved
