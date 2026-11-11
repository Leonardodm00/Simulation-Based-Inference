#!/usr/bin/env python3
"""Block 21: the evaluation's figures, from block 20's metrics and block 19's
samples (specs/SPEC.md block 21; D-126, D-127, Q-E6).

    python3 hpc/pipeline/evaluation/ev_figures.py --config hpc/pipeline/projects/synth/synth.json \\
        --eval-config hpc/pipeline/projects/synth/synth_eval.json

Writes <out_root>/figures/ (formats and dpi from the evaluation's
configuration) and FIGURES.md, one line of caption per figure:
    ensembles/<stem>/cal_*    calibration: the HPD expected coverage (members thin,
                              the ensemble thick, its band), SBC's ECDF differences
                              with the KS band and its rank histograms, TARP
    ensembles/<stem>/acc_*    accuracy and sharpness: the recovery per axis, the
                              z-scores against the contraction, the posterior SDs
                              against the prior's, the NLL per window and donor
    ensembles/<stem>/post_*   posterior views and checks: corner plots of the
                              example donors, the posterior-predictive ranks,
                              L-C2ST per window
    ensembles/<stem>/emb_*    the members' embeddings: the first two principal
                              components of D_sim and of the cohort, with MMD^2
    searches/<q>/arms_*       a search's two arms: coverage, NLL per donor, PPC,
                              L-C2ST, and the donor-paired forest plot (E3)
    searches/<q>/search_*     the search: L in proposal order with its running
                              minimum per arm, L against L_pseudo_real, the
                              members' learning curves
    across/*                  every search: Dbar with its interval and 2
                              sigma_seed; every ensemble's NLL on R0 and R3
Pass line: EVAL FIGURES OK: <n> figure(s); exit 3 when there are no metrics.

It draws and computes nothing scientific: every quantity is block 20's, read
from its files; the corner plots read block 19's draws. Colours follow the
entity in every figure: an arm keeps its hue (A1 blue, A2 orange, A5 aqua),
a condition its hue and marker; text stays in the ink colours; one y-axis
per panel.

Pure ASCII, LF only.
"""

import argparse
import json
import os
import sys
import time

import _evalpath  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import ev_config  # noqa: E402
import ev_plan  # noqa: E402

EXIT_REFUSE = 3

# ---------------------------------------------------------------- the style
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
ARM_COLOR = {"A1": "#2a78d6", "A2": "#eb6834", "A5": "#1baf7a"}
OTHER_ARM = "#6250d6"
COND_COLOR = ("#e87ba4", "#6250d6", "#008300")        # conditions 0, 1, 2 (validated all-pairs)
COND_MARKER = ("o", "s", "^")
SIM_GRAY = "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf",
                                                     "#184f95", "#0d366b"])
LW, LW_THIN, LW_REF = 1.6, 0.8, 0.8
WASH = 0.15
DATASET_TITLE = {"sim": "D_sim (arm S, report split)", "R0": "R0 (pi = 0)", "R3": "R3 (pi = 0.3)",
                 "R3dag": "R3 against theta-dagger"}


def arm_color(arm):
    return ARM_COLOR.get(arm, OTHER_ARM)


def ds_title(D):
    return DATASET_TITLE.get(D, D)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "text.color": INK, "axes.axisbelow": True})


def panels(n, ncols, w=2.6, h=2.2, sharex=False, sharey=False):
    nrows = int(np.ceil(n / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False,
                             sharex=sharex, sharey=sharey)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def diag(ax):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=LW_REF, zorder=1)


def note(ax, text, loc="upper left"):
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=INK2, fontsize=6.5)


def finish(fig, title=None, legend_from=None, ncol=4):
    """The figure's title on top, its legend (the handles of the axes
    `legend_from`) in a row below the panels, where it covers no data;
    margins in inches, whatever the figure's height."""
    hs, ls = (legend_from.get_legend_handles_labels() if legend_from is not None else ([], []))
    t_in = 0.24 if title else 0.0
    b_in = 0.28 if hs else 0.0
    h = fig.get_figheight() + t_in + b_in       # the panels keep their size
    fig.set_figheight(h)
    if title:
        fig.suptitle(title, color=INK, fontsize=9, y=1.0 - 0.04 / h, va="top")
    if hs:
        fig.legend(hs, ls, loc="lower center", ncol=min(ncol, len(hs)), frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0.0, b_in / h, 1.0, 1.0 - t_in / h))


def fmt(x, nd=3):
    return "-" if x is None else ("%." + str(nd) + "f") % x


# ---------------------------------------------------------------- the files

class Store(object):
    """block 20's files under metrics/ and block 19's under samples/."""

    def __init__(self, ecfg):
        self.mdir = ecfg.sub("metrics")
        self.sdir = ecfg.sub("samples")
        with open(os.path.join(self.mdir, "METRICS.json")) as fh:
            self.index = json.load(fh)
        self._cache = {}

    def js(self, name):
        p = os.path.join(self.mdir, name + ".json")
        return json.load(open(p)) if os.path.isfile(p) else None

    def arr(self, name):
        p = os.path.join(self.mdir, name + ".npz")
        if p not in self._cache:           # one file at a time: the figures are drawn in order
            self._cache = {p: (np.load(p) if os.path.isfile(p) else None)}
        return self._cache[p]

    def sample(self, stem, D):
        return np.load(os.path.join(self.sdir, "%s_%s.npz" % (stem, D)))


# ---------------------------------------------------------------- per ensemble

def fig_coverage(st, e, dss):
    fig, axs = panels(len(dss), min(4, len(dss)), w=2.7, h=2.6)
    col = arm_color(e["arm"])
    for ax, D in zip(axs, dss):
        a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
        b = a["betas"]
        ax.fill_between(b, a["coverage_lo"], a["coverage_hi"], color=col if D != "sim" else MUTED,
                        alpha=WASH, lw=0, label="band")
        for k, c in enumerate(a["coverage_members"]):
            ax.plot(b, c, color=col, lw=LW_THIN, alpha=0.55, label="members" if k == 0 else None)
        ax.plot(b, a["coverage"], color=col, lw=LW, label="ensemble")
        diag(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(ds_title(D))
        ax.set_xlabel("credibility level beta")
        cv = j["coverage"]
        note(ax, "n = %d rows\narea %+.3f" % (cv["n_rows"], cv["signed_area"]))
    axs[0].set_ylabel("expected coverage C(beta)")
    finish(fig, "%s: HPD expected coverage" % e["name"], axs[0])
    return fig, ("HPD expected coverage C(beta) of %s on each data set: the ensemble thick, its "
                 "members thin; band: where a calibrated estimator lies (D_sim, binomial) or the "
                 "curve's donor bootstrap (cohorts); below the diagonal is overconfident"
                 % e["name"])


def fig_sbc_ecdf(st, e, D):
    a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
    sb = j["sbc"]
    names = j["recovery"]["axes"]
    fig, axs = panels(len(names), 5, w=2.2, h=1.9, sharey=True)
    eps = sb["ks_band_halfwidth"] or 0.0
    b = a["betas"]
    col = arm_color(e["arm"])
    for k, ax in enumerate(axs):
        ax.fill_between(b, -eps, eps, color=MUTED, alpha=WASH, lw=0)
        ax.plot(b, a["sbc_ecdf_diff"][k], color=col, lw=LW)
        ax.axhline(0.0, color=MUTED, lw=LW_REF)
        ax.set_title("%s, p %.2g%s" % (names[k], sb["ks_p"][k],
                                         " (rejected)" if sb["holm_rejected"][k] else ""))
        ax.set_xlim(0, 1)
    for ax in axs[::5]:
        ax.set_ylabel("ECDF - uniform")
    finish(fig, "%s on %s: SBC rank ECDF differences (%d rows; band: KS at %.2f)"
           % (e["name"], ds_title(D), sb["n_rows"], j["level"]))
    return fig, ("SBC of %s on %s: per axis the ECDF of the normalised ranks of theta^ref among "
                 "the draws minus the uniform CDF, inside the gray KS band for a calibrated "
                 "marginal; 'rejected' marks the Holm family's rejections" % (e["name"], D))


def fig_sbc_hist(st, e, D):
    a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
    names = j["recovery"]["axes"]
    fig, axs = panels(len(names), 5, w=2.2, h=1.8, sharey=True)
    edges = a["sbc_hist_edges"]
    mid = (edges[1:] + edges[:-1]) / 2.0
    wd = np.diff(edges) * 0.8
    col = arm_color(e["arm"])
    for k, ax in enumerate(axs):
        ax.fill_between(mid, a["sbc_hist_lo"], a["sbc_hist_hi"], step="mid", color=MUTED,
                        alpha=WASH, lw=0)
        ax.bar(mid, a["sbc_hist_counts"][k], width=wd, color=col, lw=0)
        ax.plot(mid, a["sbc_hist_expected"], color=MUTED, lw=LW_REF, drawstyle="steps-mid")
        ax.set_title(names[k])
    for ax in axs[5:]:
        ax.set_xlabel("rank of theta^ref")
    finish(fig, "%s on %s: SBC rank histograms (band: per-bin binomial %.2f)"
           % (e["name"], ds_title(D), j["sbc"]["hist_level"]))
    return fig, ("SBC rank histograms of %s on %s, %d bins; a U shape is overconfidence, a hump "
                 "underconfidence, a slope bias" % (e["name"], D, len(mid)))


def fig_tarp(st, e, dss):
    fig, axs = panels(len(dss), min(4, len(dss)), w=2.7, h=2.6)
    col = arm_color(e["arm"])
    for ax, D in zip(axs, dss):
        a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
        for mode, c, lab in (("x", col, "references from x"), ("random", MUTED, "random references")):
            f = a["tarp_%s_f" % mode]
            ax.plot(np.r_[0.0, f, 1.0], np.r_[0.0, np.arange(1, f.size + 1) / f.size, 1.0], color=c,
                    lw=LW if mode == "x" else LW_THIN, label=lab)
        diag(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(ds_title(D))
        ax.set_xlabel("credibility level")
        t = j["tarp"]
        note(ax, "KS p: x %.3g, random %.3g" % (t["x"]["ks_p"], t["random"]["ks_p"]))
    axs[0].set_ylabel("expected coverage (TARP)")
    finish(fig, "%s: TARP" % e["name"], axs[0])
    return fig, ("TARP expected coverage of %s (Lemos et al. 2023): reference points read out "
                 "linearly from the windows' statistics (primary) and drawn uniformly" % e["name"])


def fig_recovery(st, e, D):
    a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
    names = j["recovery"]["axes"]
    fig, axs = panels(len(names), 5, w=2.2, h=2.2)
    col = arm_color(e["arm"])
    t = a["theta_ref"]
    for k, ax in enumerate(axs):
        ax.errorbar(t[:, k], a["mhat"][:, k],
                    yerr=[a["mhat"][:, k] - a["q05"][:, k], a["q95"][:, k] - a["mhat"][:, k]],
                    fmt="o", ms=2.5, color=col, ecolor=col, elinewidth=0.4, alpha=0.6,
                    markeredgewidth=0)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=LW_REF)
        ax.set_title(names[k])
        lo = min(0.0, float(np.nanmin(t[:, k])))
        hi = max(1.0, float(np.nanmax(t[:, k])))
        ax.set_xlim(lo - 0.02, hi + 0.02)
        ax.set_ylim(-0.02, 1.02)
        note(ax, "cov90 %.2f\ncontr %.2f" % (j["recovery"]["interval90_coverage"][k],
                                             j["recovery"]["contraction"][k]))
    for ax in axs[5:]:
        ax.set_xlabel("theta^ref")
    for ax in axs[::5]:
        ax.set_ylabel("posterior mean, 90 % interval")
    finish(fig, "%s on %s: recovery" % (e["name"], ds_title(D)))
    return fig, ("Recovery of %s on %s: per axis the posterior mean and its central 90 %% interval "
                 "against theta^ref; cov90 is the interval's coverage, contr the contraction"
                 % (e["name"], D))


def fig_zscore(st, e, D):
    a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
    names = j["recovery"]["axes"]
    fig, axs = panels(len(names), 5, w=2.2, h=2.0, sharex=True, sharey=True)
    col = arm_color(e["arm"])
    for k, ax in enumerate(axs):
        ax.scatter(a["contraction"][:, k], a["zeta"][:, k], s=4, color=col, alpha=0.5, lw=0)
        for y in (-2.0, 2.0):
            ax.axhline(y, color=MUTED, lw=LW_REF)
        ax.set_title(names[k])
        ax.set_xlim(-0.05, 1.0)
    for ax in axs[5:]:
        ax.set_xlabel("contraction 1 - sigma^2 / sigma_0^2")
    for ax in axs[::5]:
        ax.set_ylabel("z-score")
    finish(fig, "%s on %s: z-score against contraction" % (e["name"], ds_title(D)))
    return fig, ("Posterior z-score (mhat - theta^ref) / sigma against the contraction per row and "
                 "axis for %s on %s: good recovery sits at high contraction inside |z| < 2"
                 % (e["name"], D))


def fig_sigma(st, e, dss):
    fig, axs = panels(len(dss), min(4, len(dss)), w=3.2, h=2.4, sharey=True)
    col = arm_color(e["arm"])
    for ax, D in zip(axs, dss):
        a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
        names = j["recovery"]["axes"]
        bp = ax.boxplot([a["sigma"][:, k] for k in range(len(names))], widths=0.5,
                        patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=col, alpha=0.35, lw=0.6, edgecolor=col)
        for key in ("whiskers", "caps", "medians"):
            for ln in bp[key]:
                ln.set(color=col, lw=0.8)
        ax.axhline(j["recovery"]["sigma0"], color=MUTED, lw=LW_REF)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=60, ha="right")
        ax.set_title(ds_title(D))
    axs[0].set_ylabel("posterior SD sigma (line: the prior's)")
    finish(fig, "%s: posterior SDs per axis" % e["name"])
    return fig, ("Posterior SDs per axis of %s on each data set, against the box prior's SD "
                 "sqrt(1/12) (the line)" % e["name"])


def fig_nll(st, e, dss):
    fig, axs = panels(len(dss), min(4, len(dss)), w=3.0, h=2.4)
    col = arm_color(e["arm"])
    legend_ax = None
    for ax, D in zip(axs, dss):
        a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
        l = a["nll_rows"]
        if D == "sim":
            ok = np.isfinite(l)
            ax.hist(l[ok], bins=40, color=col, lw=0)
            ax.set_xlabel("NLL per window (nats)")
            ax.set_ylabel("windows")
        else:
            ids, lbar = a["donor_ids"], a["donor_nll"]
            order = np.argsort(np.where(np.isfinite(lbar), lbar, np.inf))
            pos = dict((int(ids[i]), r) for r, i in enumerate(order))
            seen = {}
            xw = []
            for d in a["donor"]:                       # a donor's windows side by side
                k = seen.get(int(d), 0)
                seen[int(d)] = k + 1
                xw.append(k)
            n_per = max(seen.values())
            xw = [pos[int(d)] - 0.3 + 0.6 * (k / max(1.0, n_per - 1.0))
                  for d, k in zip(a["donor"], xw)]
            ax.scatter(xw, l, s=4, color=MUTED, alpha=0.6, lw=0, label="windows")
            legend_ax = ax
            cls_of = dict((int(d), int(c)) for d, c in zip(a["donor"], a["cls"]))
            for c in sorted(set(cls_of.values())):
                sel = [i for i in order if cls_of[int(ids[i])] == c]
                ax.scatter([pos[int(ids[i])] for i in sel], lbar[sel], s=22,
                           color=COND_COLOR[c % 3], marker=COND_MARKER[c % 3],
                           edgecolors=SURFACE, linewidths=0.8, label="condition %d" % c, zorder=3)
            ax.set_xlabel("donor (ordered by its mean NLL)")
            ax.set_ylabel("NLL (nats)")
        ax.set_title("%s: mean %s" % (ds_title(D), fmt(j["nll"]["mean"])))
    finish(fig, "%s: NLL of theta^ref (nats)" % e["name"], legend_ax)
    return fig, ("NLL -log q_E(theta^ref | x) of %s: per window on D_sim; per window (gray) and "
                 "per donor (by condition) on the cohorts; rows outside the open box are left "
                 "out" % e["name"])


def fig_corner(st, e, D, ex, dag=None):
    """The posterior of the example donor's first window, its draws as 2-D
    histograms and marginal histograms, theta* (and theta-dagger) marked."""
    z = st.sample(e["stem"], D)
    r = int(ex["row"])
    d = z["draws"][r]
    t = z["theta_ref"][r]
    td = dag["theta_ref"][r] if dag is not None else None
    names = [n for n in st.js("%s_%s" % (e["stem"], D))["recovery"]["axes"]]
    p = len(names)
    fig, axes = plt.subplots(p, p, figsize=(1.25 * p, 1.25 * p))
    col = arm_color(e["arm"])
    for i in range(p):
        for k in range(p):
            ax = axes[i, k]
            ax.grid(False)
            if k > i:
                ax.set_visible(False)
                continue
            if i == k:
                ax.hist(d[:, i], bins=30, range=(0, 1), color=col, lw=0)
                ax.axvline(t[i], color=INK, lw=0.8)
                if td is not None and td[i] != t[i]:
                    ax.axvline(min(td[i], 1.0), color=INK2, lw=0.8)
                ax.set_yticks([])
            else:
                ax.hist2d(d[:, k], d[:, i], bins=24, range=[[0, 1], [0, 1]], cmap=SEQ, cmin=1)
                ax.plot([t[k]], [t[i]], marker="+", color=INK, ms=7, mew=1.0)
                if td is not None and (td[k] != t[k] or td[i] != t[i]):
                    ax.plot([min(td[k], 1.0)], [min(td[i], 1.0)], marker="x", color=INK2, ms=5,
                            mew=0.9)
            ax.set_xlim(0, 1)
            if i < p - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[k], fontsize=6)
            if k > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(names[i], fontsize=6, labelpad=2)
            ax.tick_params(labelsize=5, length=2)
    mark = "+: theta*" + ("; x: theta-dagger (capped at the box)" if td is not None else "")
    finish(fig, "%s on %s: donor %d (condition %d), window %d; %s"
           % (e["name"], ds_title(D), ex["donor"], ex["cls"], r, mark))
    return fig, ("Corner plot of %s's posterior at the first window of donor %d (condition %d, "
                 "the median NLL of its condition under %s) on %s: %d draws; %s"
                 % (e["name"], ex["donor"], ex["cls"], ex.get("chosen_by", "A1"), D,
                    d.shape[0], mark))


def fig_ppc(st, e, D):
    a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
    pp = j["ppc"]
    J = pp["n_sim_per_row"]
    fig, axs = panels(len(pp["stats"]), 4, w=2.3, h=1.9, sharey=True)
    col = arm_color(e["arm"])
    ok = a["ppc_ok"]
    for k, ax in enumerate(axs):
        cnt = np.bincount(a["ppc_rank"][ok, k], minlength=J + 1)
        ax.bar(np.arange(J + 1), cnt, width=0.8, color=col, lw=0)
        ax.axhline(ok.sum() / float(J + 1), color=MUTED, lw=LW_REF)
        ax.set_title(pp["stats"][k])
        note(ax, "outside 90 %%: %.2f" % pp["frac_outside_90"][k], "upper right")
    for ax in axs[4:]:
        ax.set_xlabel("rank among %d simulated" % J)
    finish(fig, "%s on %s: posterior-predictive ranks (%d windows)"
           % (e["name"], ds_title(D), pp["n_rows"]))
    return fig, ("Posterior-predictive check of %s on %s: per statistic the rank of the observed "
                 "window's value among %d windows simulated at posterior draws (the bench "
                 "generator, no gap); flat is consistent, the line its expectation" % (e["name"], D, J))


def fig_lc2st(st, e, dss):
    dss = [D for D in dss if st.js("%s_%s" % (e["stem"], D)).get("lc2st")]
    fig, axs = panels(len(dss), min(4, len(dss)), w=2.8, h=2.4, sharey=True)
    col = arm_color(e["arm"])
    q95 = ((st.js("%s_sim" % e["stem"]) or {}).get("lc2st") or {}).get("stat_q95")
    for ax, D in zip(axs, dss):
        a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
        ax.scatter(np.arange(a["lc2st_stat"].size), a["lc2st_stat"], s=5,
                   color=(SIM_GRAY if D == "sim" else col), lw=0)
        if q95 is not None:
            ax.axhline(q95, color=MUTED, lw=LW_REF)
        ax.set_title(ds_title(D))
        ax.set_xlabel("window")
        lc = j["lc2st"]
        note(ax, "p < %.2f: %.2f of %d" % (lc["alpha"], lc["frac_p_below_alpha"], lc["n_rows"]),
             "upper right")
    axs[0].set_ylabel("L-C2ST statistic")
    finish(fig, "%s: L-C2ST (line: D_sim's 95th percentile)" % e["name"])
    return fig, ("Local C2ST of %s (Linhart et al. 2023; sbi's LC2ST on the calibration bank, "
                 "features theta and the eight statistics): the statistic per window, D_sim's PPC "
                 "rows as the reference" % e["name"])


def fig_embedding(st, e, D):
    a, j = st.arr("%s_%s" % (e["stem"], D)), st.js("%s_%s" % (e["stem"], D))
    M = a["pca_coords"].shape[0]
    fig, axs = panels(M, min(5, M), w=2.4, h=2.4)
    src, cls = a["pca_source"], a["pca_cls"]
    for m, ax in enumerate(axs):
        xy = a["pca_coords"][m]
        ax.scatter(xy[src == 0, 0], xy[src == 0, 1], s=3, color=SIM_GRAY, lw=0, label="D_sim")
        for c in sorted(set(cls[src == 1].tolist())):
            sel = (src == 1) & (cls == c)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=12, color=COND_COLOR[int(c) % 3],
                       marker=COND_MARKER[int(c) % 3], edgecolors=SURFACE, linewidths=0.5,
                       label="%s condition %d" % (D, c))
        mm = j["embedding"]["members"][m]
        ax.set_title("member %d: MMD^2 %.3g (p %.3g)" % (m + 1, mm["mmd2"], mm["p"]))
        ax.set_xlabel("PC 1 (%.0f %%)" % (100 * a["pca_evr"][m][0]))
        ax.set_ylabel("PC 2 (%.0f %%)" % (100 * a["pca_evr"][m][1]))
    finish(fig, "%s: embeddings of D_sim and %s" % (e["name"], ds_title(D)), axs[0])
    return fig, ("Each member's embeddings h_psi(x) of D_sim (gray) and of %s by condition, on "
                 "their first two principal components, with the MMD^2 of the cohort against "
                 "D_sim and its permutation p-value" % D)


# ---------------------------------------------------------------- per search

def fig_arms_coverage(st, q, eT, eA, dss):
    fig, axs = panels(len(dss), min(4, len(dss)), w=2.7, h=2.6)
    for ax, D in zip(axs, dss):
        for e in (eA, eT):
            a = st.arr("%s_%s" % (e["stem"], D))
            col = arm_color(e["arm"])
            ax.fill_between(a["betas"], a["coverage_lo"], a["coverage_hi"], color=col,
                            alpha=WASH * 0.8, lw=0)
            ax.plot(a["betas"], a["coverage"], color=col, lw=LW, label=e["arm"])
        diag(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(ds_title(D))
        ax.set_xlabel("credibility level beta")
    axs[0].set_ylabel("expected coverage C(beta)")
    finish(fig, "%s: the two arms' HPD expected coverage" % q, axs[0])
    return fig, ("HPD expected coverage of the two arms of %s on each data set they share, with "
                 "their bands" % q)


def fig_arms_nll(st, q, eT, eA, D):
    aT, aA = st.arr("%s_%s" % (eT["stem"], D)), st.arr("%s_%s" % (eA["stem"], D))
    ids = aA["donor_ids"]
    lA, lT = aA["donor_nll"], aT["donor_nll"]
    order = np.argsort(np.where(np.isfinite(lA), lA, np.inf))
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    xs = np.arange(ids.size)
    for x, i in zip(xs, order):
        ax.plot([x, x], [lA[i], lT[i]], color=AXIS, lw=LW_THIN, zorder=1)
    for e, l in ((eA, lA), (eT, lT)):
        ax.scatter(xs, l[order], s=20, color=arm_color(e["arm"]), edgecolors=SURFACE,
                   linewidths=0.8, label=e["arm"], zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(d)) for d in ids[order]], rotation=90, fontsize=5)
    ax.set_xlabel("donor (ordered by A1's mean NLL)")
    ax.set_ylabel("donor mean NLL (nats)")
    finish(fig, "%s on %s: donor mean NLL of the two arms" % (q, ds_title(D)), ax)
    return fig, ("Donor mean NLL lbar_d of the two arms of %s on %s, each donor's pair joined; "
                 "below the A1 dot the term arm does better on that donor" % (q, D))


def fig_arms_ppc(st, q, eT, eA, D):
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    jT, jA = st.js("%s_%s" % (eT["stem"], D)), st.js("%s_%s" % (eA["stem"], D))
    stats = jA["ppc"]["stats"]
    x = np.arange(len(stats))
    for off, e, j in ((-0.15, eA, jA), (0.15, eT, jT)):
        ref = st.js("%s_sim" % e["stem"])["ppc"]["frac_outside_90"]
        col = arm_color(e["arm"])
        for xi, a_, r_ in zip(x + off, j["ppc"]["frac_outside_90"], ref):
            ax.plot([xi, xi], [r_, a_], color=col, lw=LW_THIN, alpha=0.6, zorder=1)
        ax.scatter(x + off, j["ppc"]["frac_outside_90"], s=26, color=col, edgecolors=SURFACE,
                   linewidths=0.8, zorder=3, label="%s on %s" % (e["arm"], D))
        ax.scatter(x + off, ref, s=22, facecolors=SURFACE, edgecolors=col, linewidths=1.0,
                   zorder=3, label="%s on D_sim" % e["arm"])
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("fraction outside the 90 % range")
    finish(fig, "%s on %s: posterior-predictive misses per statistic" % (q, ds_title(D)), ax)
    return fig, ("Fraction of %s's windows whose statistic falls outside the 5-95 %% range of "
                 "the windows simulated at the posterior draws, both arms of %s (filled), and the "
                 "same on D_sim (rings), the reference; about 0.1 when the model fits" % (D, q))


def fig_arms_lc2st(st, q, eT, eA, D):
    aT, aA = st.arr("%s_%s" % (eT["stem"], D)), st.arr("%s_%s" % (eA["stem"], D))
    ids = aA["lc2st_donor_ids"]
    order = np.argsort(aA["lc2st_donor_stat"])
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    xs = np.arange(ids.size)
    for e, a in ((eA, aA), (eT, aT)):
        m = dict(zip(a["lc2st_donor_ids"].tolist(), a["lc2st_donor_stat"].tolist()))
        ax.scatter(xs, [m.get(int(ids[i]), np.nan) for i in order], s=20,
                   color=arm_color(e["arm"]), edgecolors=SURFACE, linewidths=0.8, label=e["arm"])
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(ids[i])) for i in order], rotation=90, fontsize=5)
    ax.set_xlabel("donor (ordered by A1's statistic)")
    ax.set_ylabel("donor mean L-C2ST statistic")
    finish(fig, "%s on %s: L-C2ST per donor" % (q, ds_title(D)), ax)
    return fig, "Donor mean of the local C2ST statistic for the two arms of %s on %s" % (q, D)


def fig_forest(st, q, D):
    j = st.js("paired_%s_%s" % (q, D))
    a = st.arr("paired_%s_%s" % (q, D))
    order = np.argsort(a["delta_d"])
    dd, se = a["delta_d"][order], a["se_d"][order]
    n = dd.size
    fig, ax = plt.subplots(figsize=(4.8, max(2.4, 0.11 * n + 1.2)))
    ys = np.arange(n)
    col = arm_color(j["term_arm"])
    lo, hi = j["ci"]
    ax.axvspan(lo, hi, color=col, alpha=WASH, lw=0, label="Dbar, %d %% interval" % round(100 * j["level"]))
    ax.axvline(j["Dbar"], color=col, lw=LW)
    for s in (-1, 1):
        ax.axvline(s * j["two_sigma_seed"], color=MUTED, lw=LW_REF)
    ax.axvline(0.0, color=INK2, lw=LW_REF)
    if n <= 80:
        ax.errorbar(dd, ys, xerr=1.96 * np.nan_to_num(se), fmt="o", ms=3, color=col, ecolor=col,
                    elinewidth=0.6, markeredgewidth=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([str(int(d)) for d in a["donor_ids"][order]], fontsize=5)
        ax.set_ylabel("donor")
    else:
        ax.scatter(dd, ys, s=3, color=col, lw=0)
        ax.set_yticks([])
        ax.set_ylabel("%d donors, ordered" % n)
    ax.set_xlabel("Delta_d = lbar_d(%s) - lbar_d(A1) (nats)" % j["term_arm"])
    verdict = "resolved, %s lower" % j["lower_nll"] if j["resolved"] else "not resolved"
    finish(fig, "%s on %s: %s (gray lines: +-2 sigma_seed)" % (q, ds_title(D), verdict), ax)
    return fig, ("Donor-paired difference (E3) of %s's two arms on %s: per donor Delta_d with "
                 "1.96 SE over its windows, Dbar %+.4f with its donor-cluster bootstrap interval "
                 "[%+.4f, %+.4f], +-2 sigma_seed = %.4f; G.2: %s"
                 % (q, D, j["Dbar"], lo, hi, j["two_sigma_seed"], verdict))


def fig_trials(st, q, s):
    tr = s["trials"]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    xs = np.arange(len(tr))
    arms = sorted(set(t["arm"] for t in tr if t["arm"]))
    for arm in arms:
        sel = [i for i, t in enumerate(tr) if t["arm"] == arm and t["status"] == "ok"
               and t["nll"] is not None]
        col = arm_color(arm)
        ax.scatter(xs[sel], [tr[i]["nll"] for i in sel], s=14, color=col, edgecolors=SURFACE,
                   linewidths=0.6, label=arm, zorder=3)
        rm = [t.get("running_min_nll", {}).get(arm) for t in tr]
        ax.step(xs, [np.nan if v is None else v for v in rm], where="post", color=col, lw=LW_THIN)
    fin = [i for i, t in enumerate(tr) if t["trial_id"] == s.get("finalist")]
    if fin:
        ax.scatter(xs[fin], [tr[fin[0]]["nll"]], s=70, facecolors="none", edgecolors=INK,
                   linewidths=1.0, zorder=4, label="finalist")
    ax.set_xlabel("trial, in proposal order")
    ax.set_ylabel("L, report-split NLL (nats)")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    finish(fig, "%s: the search (%d trials)" % (q, len(tr)), ax)
    return fig, ("The search %s: each trial's simulated report-split NLL L in proposal order by "
                 "arm, the running minimum per arm, the finalist ringed" % q)


def fig_nll_vs_lpr(st, q, s):
    tr = [t for t in s["trials"] if t["status"] == "ok" and t["nll"] is not None
          and t["L_pseudo_real"] is not None]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for arm in sorted(set(t["arm"] for t in tr)):
        sel = [t for t in tr if t["arm"] == arm]
        ax.scatter([t["nll"] for t in sel], [t["L_pseudo_real"] for t in sel], s=14,
                   color=arm_color(arm), edgecolors=SURFACE, linewidths=0.6, label=arm)
    ax.set_xlabel("L, simulated report split (nats)")
    ax.set_ylabel("L_pseudo_real, %s (nats)" % s["real"])
    finish(fig, "%s: L against L_pseudo_real" % q, ax)
    return fig, ("Every completed trial of %s: the simulated report-split NLL it was ranked on "
                 "against the NLL of %s's withheld theta (D-121)" % (q, s["real"]))


def fig_learning(st, q, s):
    names = sorted(s["ensembles"])
    fig, axs = panels(len(names), min(3, len(names)), w=3.0, h=2.4)
    for ax, name in zip(axs, names):
        ens = s["ensembles"][name]
        col = arm_color(ens["arm"])
        for k, m in enumerate(ens["members"]):
            ep = [x for x in m["epoch"]]
            ax.plot(ep, [np.nan if v is None else v for v in m["npe"]], color=col, lw=LW_THIN,
                    alpha=0.35, label="training NPE loss" if k == 0 else None)
            ax.plot(ep, [np.nan if v is None else v for v in m["val_npe"]], color=col, lw=LW_THIN,
                    label="selection NLL" if k == 0 else None)
        ax.set_title("%s (%d members)" % (name, len(ens["members"])))
        ax.set_xlabel("epoch")
    axs[0].set_ylabel("nats")
    finish(fig, "%s: the members' learning curves" % q, axs[0])
    return fig, "Learning curves of every member of %s's ensembles: the training loss (light) and the selection-split NLL" % q


# ---------------------------------------------------------------- across searches

def fig_dbar_all(st, rows):
    fig, ax = plt.subplots(figsize=(5.4, max(2.0, 0.32 * len(rows) + 1.0)))
    ys = np.arange(len(rows))[::-1]
    for y, (label, j) in zip(ys, rows):
        col = arm_color(j["term_arm"])
        ax.plot([-j["two_sigma_seed"], j["two_sigma_seed"]], [y, y], color=AXIS, lw=4.0,
                solid_capstyle="butt", zorder=1)
        ax.errorbar([j["Dbar"]], [y], xerr=[[j["Dbar"] - j["ci"][0]], [j["ci"][1] - j["Dbar"]]],
                    fmt="o", ms=4, color=col, ecolor=col, elinewidth=1.0, zorder=3)
    for arm in sorted(set(j["term_arm"] for _l, j in rows)):
        ax.scatter([], [], s=16, color=arm_color(arm), label="term arm %s" % arm)
    ax.axvline(0.0, color=INK2, lw=LW_REF)
    ax.set_yticks(ys)
    ax.set_yticklabels([l for l, _j in rows], fontsize=6.5)
    ax.set_xlabel("Dbar = term arm - A1 (nats per window); gray: +-2 sigma_seed")
    finish(fig, "Every search: the donor-paired difference of its two arms", ax)
    return fig, ("Dbar (E3) of every search with both arms, on each data set, with its "
                 "donor-cluster bootstrap interval; the gray bar is +-2 sigma_seed (G.2): a "
                 "resolved difference lies outside it with an interval that excludes 0")


def fig_lpr_all(st, ens_rows, cohorts):
    fig, axs = panels(len(cohorts), len(cohorts), w=3.0, h=max(2.0, 0.3 * len(ens_rows) + 1.0),
                      sharey=True)
    ys = np.arange(len(ens_rows))[::-1]
    for ax, c in zip(axs, cohorts):
        for y, (name, arm, vals) in zip(ys, ens_rows):
            if vals.get(c) is not None:
                ax.scatter([vals[c]], [y], s=22, color=arm_color(arm), edgecolors=SURFACE,
                           linewidths=0.8, zorder=3)
        ax.set_title(ds_title(c))
        ax.set_xlabel("mean NLL of theta* (nats)")
        ax.set_yticks(ys)
        ax.set_yticklabels([n for n, _a, _v in ens_rows], fontsize=6.5)
    for arm in sorted(set(a for _n, a, _v in ens_rows)):
        axs[0].scatter([], [], s=16, color=arm_color(arm), label=arm)
    finish(fig, "Every ensemble: L_pseudo_real of the mixture", axs[0])
    return fig, ("Every ensemble's mean NLL of the withheld theta* (its mixture's L_pseudo_real) "
                 "on each cohort it is scored on")


# ---------------------------------------------------------------- the run

def make_all(cfg_path, ecfg_path, log=print):
    t0 = time.time()
    cfg = ev_plan.load_project(cfg_path)
    ecfg = ev_config.load(ecfg_path)
    if not os.path.isfile(os.path.join(ecfg.sub("metrics"), "METRICS.json")):
        raise ev_plan.RefuseError("no metrics at %s (block 20's ev_metrics.py first)"
                                  % ecfg.sub("metrics"))
    st = Store(ecfg)
    style()
    fdir = ecfg.sub("figures")
    index = []

    def save(fig, rel, caption, family):
        base = os.path.join(fdir, rel)
        os.makedirs(os.path.dirname(base), exist_ok=True)
        for f in ecfg.formats:
            fig.savefig("%s.%s" % (base, f), dpi=ecfg.dpi)
        plt.close(fig)
        index.append((family, rel, caption))

    ens = []
    for name, info in st.index["ensembles"].items():
        e = dict(info, name=name)
        ens.append(e)
        stem, dss = e["stem"], list(info["datasets"])
        rel = os.path.join("ensembles", stem)
        f, c = fig_coverage(st, e, dss)
        save(f, os.path.join(rel, "cal_coverage"), c, "calibration")
        f, c = fig_tarp(st, e, dss)
        save(f, os.path.join(rel, "cal_tarp"), c, "calibration")
        for D in dss:
            f, c = fig_sbc_ecdf(st, e, D)
            save(f, os.path.join(rel, "cal_sbc_ecdf_%s" % D), c, "calibration")
            f, c = fig_sbc_hist(st, e, D)
            save(f, os.path.join(rel, "cal_sbc_hist_%s" % D), c, "calibration")
            f, c = fig_recovery(st, e, D)
            save(f, os.path.join(rel, "acc_recovery_%s" % D), c, "accuracy")
            f, c = fig_zscore(st, e, D)
            save(f, os.path.join(rel, "acc_zscore_%s" % D), c, "accuracy")
            j = st.js("%s_%s" % (stem, D))
            if j.get("ppc"):
                f, c = fig_ppc(st, e, D)
                save(f, os.path.join(rel, "post_ppc_%s" % D), c, "posterior")
            if j.get("embedding"):
                f, c = fig_embedding(st, e, D)
                save(f, os.path.join(rel, "emb_pca_%s" % D), c, "embeddings")
        f, c = fig_sigma(st, e, dss)
        save(f, os.path.join(rel, "acc_sigma"), c, "accuracy")
        f, c = fig_nll(st, e, dss)
        save(f, os.path.join(rel, "acc_nll"), c, "accuracy")
        if any((st.js("%s_%s" % (stem, D)) or {}).get("lc2st") for D in dss):
            f, c = fig_lc2st(st, e, dss)
            save(f, os.path.join(rel, "post_lc2st"), c, "posterior")
        s = st.js("search_%s" % e["search"]) or {}
        for coh, ex in sorted((s.get("example_donors") or {}).items()):
            if coh not in dss:
                continue
            dag = st.sample(stem, coh + "dag") if (coh + "dag") in dss else None
            for x in ex["donors"]:
                x = dict(x, chosen_by=ex["chosen_by"])
                f, c = fig_corner(st, e, coh, x, dag)
                save(f, os.path.join(rel, "post_corner_%s_c%d" % (coh, x["cls"])), c, "posterior")
    # the searches: both arms, then the search itself
    rows = []
    for q in cfg.data["synthetic"]["searches"]:
        mine = [e for e in ens if e["search"] == q["name"]]
        s = st.js("search_%s" % q["name"])
        rel = os.path.join("searches", q["name"])
        arms = dict((e["arm"], e) for e in mine)
        if len(arms) >= 2 and "A1" in arms:
            eA = arms["A1"]
            eT = [e for a, e in arms.items() if a != "A1"][0]
            dss = [D for D in eA["datasets"] if D in eT["datasets"]]
            f, c = fig_arms_coverage(st, q["name"], eT, eA, dss)
            save(f, os.path.join(rel, "arms_coverage"), c, "searches")
            for D in dss:
                if D != "sim":
                    f, c = fig_arms_nll(st, q["name"], eT, eA, D)
                    save(f, os.path.join(rel, "arms_nll_%s" % D), c, "searches")
                if st.js("%s_%s" % (eA["stem"], D)).get("ppc") and D != "sim":
                    f, c = fig_arms_ppc(st, q["name"], eT, eA, D)
                    save(f, os.path.join(rel, "arms_ppc_%s" % D), c, "searches")
                if st.js("%s_%s" % (eA["stem"], D)).get("lc2st") and D != "sim":
                    f, c = fig_arms_lc2st(st, q["name"], eT, eA, D)
                    save(f, os.path.join(rel, "arms_lc2st_%s" % D), c, "searches")
                if st.js("paired_%s_%s" % (q["name"], D)):
                    f, c = fig_forest(st, q["name"], D)
                    save(f, os.path.join(rel, "arms_forest_%s" % D), c, "searches")
                    rows.append(("%s, %s" % (q["name"], D), st.js("paired_%s_%s" % (q["name"], D))))
        if s:
            f, c = fig_trials(st, q["name"], s)
            save(f, os.path.join(rel, "search_trials"), c, "searches")
            f, c = fig_nll_vs_lpr(st, q["name"], s)
            save(f, os.path.join(rel, "search_nll_vs_lpr"), c, "searches")
            if s.get("ensembles"):
                f, c = fig_learning(st, q["name"], s)
                save(f, os.path.join(rel, "search_learning"), c, "searches")
    if rows:
        f, c = fig_dbar_all(st, rows)
        save(f, os.path.join("across", "dbar"), c, "across")
    cohorts = [r["name"] for r in cfg.data["synthetic"]["real"]]
    ens_rows = []
    for e in ens:
        vals = dict((c, st.js("%s_%s" % (e["stem"], c))["nll"]["mean"])
                    for c in cohorts if c in e["datasets"])
        ens_rows.append((e["name"], e["arm"], vals))
    if ens_rows:
        f, c = fig_lpr_all(st, ens_rows, cohorts)
        save(f, os.path.join("across", "lpr"), c, "across")
    # the index
    lines = ["# The evaluation's figures", "",
             "Written by `hpc/pipeline/evaluation/ev_figures.py` (block 21) from `%s`; formats %s at "
             "%d dpi. Left out of the metrics: %s." % (os.path.join(ecfg.out_root, "metrics"),
                                                       ", ".join(ecfg.formats), ecfg.dpi,
                                                       "; ".join("%s (%s)" % (x["ensemble"], x["why"])
                                                                 for x in st.index["left_out"])
                                                       or "none"), ""]
    for fam in ("calibration", "accuracy", "posterior", "embeddings", "searches", "across"):
        sel = [(r, c) for f_, r, c in index if f_ == fam]
        if not sel:
            continue
        lines += ["## %s" % fam, ""]
        lines += ["- `%s.{%s}`: %s." % (r, ",".join(ecfg.formats), c) for r, c in sel]
        lines.append("")
    with open(os.path.join(fdir, "FIGURES.md.tmp"), "w") as fh:
        fh.write("\n".join(lines))
    os.replace(os.path.join(fdir, "FIGURES.md.tmp"), os.path.join(fdir, "FIGURES.md"))
    log("EVAL FIGURES OK: %d figure(s) in %s (%.0f s)" % (len(index), fdir, time.time() - t0))
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the pipeline's project file (<tag>.json)")
    ap.add_argument("--eval-config", required=True, help="the evaluation's file (<tag>_eval.json)")
    args = ap.parse_args(argv)
    try:
        make_all(args.config, args.eval_config, log=lambda t: print(t, flush=True))
    except (ev_plan.RefuseError, ev_config.EvalConfigError) as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
