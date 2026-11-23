#!/usr/bin/env python3
"""Block 37: the figures of a recorded project's evaluation (specs/SPEC.md
sec. 2.6 and block 37; decisions D-159, D-160, D-165).

    python3 hpc/pipeline/evaluation/ev_rec_figures.py --config <tag>.json \\
        --eval-config <tag>_rec_eval.json

Reads <out_root>/metrics/ (block 36: METRICS.json and the files it names)
and, for the corner figures alone, the example wells' draws in
<out_root>/samples/ (block 35); writes <out_root>/figures/<family>/<name>.
<fmt> in the configuration's formats and FIGURES.md, one line per figure
with a one-sentence caption. It draws what the metrics hold and computes
nothing scientific: no mean, SD, quantile or median is taken here, and no
statistics library is imported (R37b).

    ensembles/<stem>/cal_*      C_E with its band and the members' curves,
                                C_E^edge and C_E^int beside it; SBC; TARP
    ensembles/<stem>/acc_*      recovery, z-scores, the NLL (every row, the
                                interior and the edge rows), the contraction
    ensembles/<stem>/gate_*     the gate: T on the null rows, the check rows
                                and each bank's windows; per bank the wells'
                                flagged shares against alpha_gate and the size
                                band, p per window, r_m per member and window
    ensembles/<stem>/emb_pca_*  the members' embeddings of the reference rows
                                and of a bank's windows
    ensembles/<stem>/rec_*      per bank the mixture over each well's windows
                                per axis, and the pooled draws of each
                                class's example well
    comparisons/<name>/cmp_*    block 28's NLL and forest plot, on D_sim
    across/dbar                 every comparison's Dbar (block 28's)

The style, the colours and the figures that block 21 and block 28 already
draw from the same keys (SBC histograms, recovery, z-scores, the comparisons)
are ev_figures', imported, never copied: an arm its hue, a class its hue and
marker, gray for references and for D_sim as context, one y-axis per panel,
the legend below the panels, no title inside the image.

Pass line: REC FIGURES OK: <n> figure(s); exit 3 when there are no metrics.

CONTRACT (wave C): the constants and the signatures below are fixed by
specs/SPEC.md block 37; the bodies are the implementer's.

Pure ASCII, LF only.
"""

import argparse
import os
import sys
import time

import _evalpath  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import ev_figures as F21  # noqa: E402  (block 21's style, colours and figures)
import ev_rec_config  # noqa: E402
from ev_plan import RefuseError  # noqa: E402

EXIT_REFUSE = 3
FIGURES_INDEX = "FIGURES.md"

# The names of the figures (block 37, "The figures"); <stem> an ensemble's
# stem, <r> one of its recording banks, <k> a class of that bank, <name> a
# comparison's name.
#   ensembles/<stem>/<f>            f in ENSEMBLE_FIGURES
#   ensembles/<stem>/<f % r>        f in BANK_FIGURES, for each bank r
#   ensembles/<stem>/<CORNER % (r, k)>  for each class k of bank r
#   comparisons/<name>/<f>          f in COMPARISON_FIGURES
#   across/<f>                      f in ACROSS_FIGURES, when there is a comparison
ENSEMBLE_FIGURES = ("cal_coverage", "cal_sbc_ecdf", "cal_sbc_hist", "cal_tarp", "acc_recovery",
                    "acc_zscore", "acc_nll", "acc_contraction", "gate_null")
BANK_FIGURES = ("gate_wells_%s", "gate_windows_%s", "gate_members_%s", "emb_pca_%s",
                "rec_wells_%s")
CORNER_FIGURE = "rec_corner_%s_c%d"
COMPARISON_FIGURES = ("cmp_nll", "cmp_forest")
ACROSS_FIGURES = ("dbar",)

# a recording bank's markers when several share a panel (acc_contraction):
# the bank is a data set, not an arm or a class, so it keeps the ensemble's
# hue and takes a shape of its own
BANK_MARKERS = ("o", "D", "v", "P", "X", "h")
SIM_MARKER = "s"
FLAG_RING = 2.2        # a flagged window's ring, times its dot's size
LEGEND_IN_PER_ENTRY = 1.7   # inches a legend entry takes in a row (7 pt labels)


# ---------------------------------------------------------------- helpers

def _cls_style(c):
    """A class's hue and marker: block 21's condition palette, by class id
    (block 21's rule, c mod 3)."""
    return F21.COND_COLOR[int(c) % 3], F21.COND_MARKER[int(c) % 3]


def _f(x, spec="%.3f"):
    """A JSON number for a figure's text; null as '-'."""
    return "-" if x is None else spec % x


def _sentence(text):
    """A caption without its final full stop (FIGURES.md adds it; block 21's
    convention)."""
    return text.rstrip().rstrip(".")


def _well_order(wells):
    """The positions of a bank's wells along an axis: by class, then batch,
    then well id (the JSON's `wells` entries), with a gap of one position
    between classes. Returns (order, x, groups): `order` indices into
    `wells`, `x` their positions, `groups` [(cls, batch, x_first, x_last)]."""
    order = sorted(range(len(wells)), key=lambda i: (wells[i]["cls"], wells[i]["batch"],
                                                     wells[i]["well"]))
    x, groups, pos, last = [], [], 0, None
    for i in order:
        key = (wells[i]["cls"], wells[i]["batch"])
        if last is not None and key[0] != last[0]:
            pos += 1                                   # a gap between classes
        if key != last:
            groups.append([key[0], key[1], pos, pos])
        groups[-1][3] = pos
        x.append(pos)
        pos += 1
        last = key
    return order, np.asarray(x, dtype=np.float64), [tuple(g) for g in groups]


def _window_positions(well, wells):
    """Each window's position along an axis that groups the windows by well,
    the wells in _well_order's order (a well's windows side by side, in bank
    order). Returns (x (n_r,), well ticks [(x_centre, well id)], groups as
    _well_order's, in window positions)."""
    order, _x, _g = _well_order(wells)
    x = np.empty(well.size, dtype=np.float64)
    ticks, groups, pos, last = [], [], 0, None
    for i in order:
        w = wells[i]
        key = (w["cls"], w["batch"])
        if last is not None and key[0] != last[0]:
            pos += 1
        if key != last:
            groups.append([key[0], key[1], pos, pos])
        rows = np.flatnonzero(well == w["well"])
        x[rows] = pos + np.arange(rows.size)
        ticks.append((pos + (rows.size - 1) / 2.0, w["well"]))
        pos += rows.size
        groups[-1][3] = pos - 1
        last = key
    return x, ticks, [tuple(g) for g in groups]


def _class_bands(ax, groups):
    """Thin separators between batches and a class label above each class's
    run (groups from _well_order or _window_positions)."""
    seen = {}
    for c, _b, x0, x1 in groups:
        lo, hi = seen.get(c, (x0, x1))
        seen[c] = (min(lo, x0), max(hi, x1))
    for k, (_c, _b, x0, _x1) in enumerate(groups):
        if k:
            ax.axvline(x0 - 0.5, color=F21.GRID, lw=F21.LW_THIN, zorder=0)
    for c, (x0, x1) in seen.items():
        ax.text((x0 + x1) / 2.0, 1.01, "class %d" % c, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5, color=_cls_style(c)[0])


def _class_legend(ax, classes):
    for c in classes:
        col, mk = _cls_style(c)
        ax.scatter([], [], s=18, color=col, marker=mk, edgecolors=F21.SURFACE, linewidths=0.6,
                   label="class %d" % c)


def _axis_label(names, bounds, k):
    """An axis's name with its declared range in its declared coordinate
    (the metrics' `axes`), the values being in the unit box."""
    if bounds is None:
        return names[k]
    return "%s (unit box of [%.3g, %.3g])" % (names[k], bounds[k][0], bounds[k][1])


def _finish(fig, legend_from=None, ncol=4):
    """Block 21's layout without a title (none inside the image, block 37):
    the legend of the axes `legend_from` below the panels, in as many rows
    as its entries need at the figure's width, with room for each row."""
    hs, ls = (legend_from.get_legend_handles_labels() if legend_from is not None else ([], []))
    ncol = max(1, min(ncol, len(hs) or 1, int(fig.get_figwidth() / LEGEND_IN_PER_ENTRY)))
    rows = (len(hs) + ncol - 1) // ncol
    b_in = (0.12 + 0.17 * rows) if hs else 0.0
    h = fig.get_figheight() + b_in             # the panels keep their size
    fig.set_figheight(h)
    if hs:
        fig.legend(hs, ls, loc="lower center", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0.0, b_in / h, 1.0, 1.0))


def _u(pair):
    """(figure, caption) of block 21's or block 28's function, untitled."""
    return _untitled(pair[0]), pair[1]


def _untitled(fig):
    """A figure drawn by block 21's or block 28's function, its title taken
    out (none inside the image, block 37: the caption is FIGURES.md's) and
    the room block 21's finish gave it returned."""
    t = getattr(fig, "_suptitle", None)
    if t is None:
        return fig
    t_in = 0.24 + 0.15 * t.get_text().count("\n")      # block 21's finish
    t.remove()
    fig._suptitle = None
    h = fig.get_figheight() - t_in
    fig.set_figheight(h)
    b_in = 0.28 if fig.legends else 0.0                  # block 21's finish
    fig.tight_layout(rect=(0.0, b_in / h, 1.0, 1.0))
    return fig


# ---------------------------------------------------------------- D_sim

def fig_coverage(st, e):
    """cal_coverage: C_E with its binomial band and the members' curves;
    beside it C_E, C_E^edge and C_E^int (sec. 2.6, (R2))."""
    a, j = st.arr("%s_sim" % e["stem"]), st.js("%s_sim" % e["stem"])
    b = a["betas"]
    col = F21.arm_color(e["arm"])
    fig, (ax0, ax1) = F21.panels(2, 2, w=2.9, h=2.7)
    ax0.fill_between(b, a["coverage_lo"], a["coverage_hi"], color=F21.MUTED, alpha=F21.WASH,
                     lw=0, label="band of a calibrated estimator")
    for k, c in enumerate(a["coverage_members"]):
        ax0.plot(b, c, color=col, lw=F21.LW_THIN, alpha=0.55, label="members" if k == 0 else None)
    ax0.plot(b, a["coverage"], color=col, lw=F21.LW, label="C_E, the ensemble")
    cv = j["coverage"]
    F21.note(ax0, "S_cal: %d rows\narea %s" % (cv["n_rows"], _f(cv["signed_area"], "%+.3f")))
    ax0.set_title("C_E over S_cal")
    ed = j["edge"]
    ax1.plot(b, a["coverage"], color=col, lw=F21.LW, label="C_E")
    ax1.plot(b, a["coverage_int"], color=col, lw=F21.LW_THIN, ls="--", label="C_E^int")
    ax1.plot(b, a["coverage_edge"], color=F21.INK2, lw=F21.LW_THIN, ls=":", label="C_E^edge")
    F21.note(ax1, "edge rows %d, interior %d\npi_edge %s (box prior %s)"
             % (ed["n_edge"], ed["n_int"], _f(ed["pi_edge"]), _f(ed["pi_edge_expected"])))
    ax1.set_title("the edge split (eps %.4g)" % ed["eps"])
    for ax in (ax0, ax1):
        F21.diag(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("credibility level beta")
    ax0.set_ylabel("expected coverage")
    # one legend for both panels, from proxies on the right one
    ax1.fill_between([], [], [], color=F21.MUTED, alpha=F21.WASH, lw=0,
                     label="band of a calibrated estimator")
    ax1.plot([], [], color=col, lw=F21.LW_THIN, alpha=0.55, label="members")
    _finish(fig, ax1, ncol=5)
    return fig, ("HPD expected coverage of %s on D_sim: C_E over the calibration rows in the box "
                 "(thick) with the band where a calibrated estimator's curve lies and the members' "
                 "curves (thin); beside it C_E, C_E^int (dashed) and C_E^edge (dotted), which need "
                 "not follow the diagonal even for a calibrated estimator, read through (R2)"
                 % e["name"])


def fig_sbc_ecdf(st, e):
    """cal_sbc_ecdf: per axis the SBC rank ECDF minus the uniform CDF, with
    the KS band (block 21's figure, its p-values null-safe)."""
    a, j = st.arr("%s_sim" % e["stem"]), st.js("%s_sim" % e["stem"])
    sb = j["sbc"]
    names = j["recovery"]["axes"]
    ncol = min(5, len(names))
    fig, axs = F21.panels(len(names), ncol, w=2.2, h=1.9, sharey=True)
    hw = sb["ks_band_halfwidth"] or 0.0
    b = a["betas"]
    col = F21.arm_color(e["arm"])
    for k, ax in enumerate(axs):
        ax.fill_between(b, -hw, hw, color=F21.MUTED, alpha=F21.WASH, lw=0)
        ax.plot(b, a["sbc_ecdf_diff"][k], color=col, lw=F21.LW)
        ax.axhline(0.0, color=F21.MUTED, lw=F21.LW_REF)
        ax.set_title("%s, p %s%s" % (names[k], _f(sb["ks_p"][k], "%.2g"),
                                       " (rejected)" if sb["holm_rejected"][k] else ""))
        ax.set_xlim(0, 1)
    for ax in axs[::ncol]:
        ax.set_ylabel("ECDF - uniform")
    for ax in axs[-ncol:]:
        ax.set_xlabel("normalised rank")
    _finish(fig)
    return fig, ("SBC of %s over D_sim's %d calibration rows: per axis the ECDF of the normalised "
                 "ranks of theta among the draws minus the uniform CDF, inside the gray KS band "
                 "at %.2f for a calibrated marginal; 'rejected' marks the Holm family's rejections"
                 % (e["name"], sb["n_rows"], j["level"]))


def fig_tarp(st, e):
    """cal_tarp: TARP's ECDF for both reference modes (block 21's figure on
    D_sim alone, its p-values null-safe)."""
    a, j = st.arr("%s_sim" % e["stem"]), st.js("%s_sim" % e["stem"])
    col = F21.arm_color(e["arm"])
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    for mode, c, lab in (("x", col, "references from x"), ("random", F21.MUTED,
                                                            "random references")):
        f = a["tarp_%s_f" % mode]
        ax.plot(np.r_[0.0, f, 1.0], np.r_[0.0, np.arange(1, f.size + 1) / float(f.size), 1.0],
                color=c, lw=F21.LW if mode == "x" else F21.LW_THIN, label=lab)
    F21.diag(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("credibility level")
    ax.set_ylabel("expected coverage (TARP)")
    t = j["tarp"]
    F21.note(ax, "KS p: x %s, random %s" % (_f(t["x"]["ks_p"], "%.3g"),
                                             _f(t["random"]["ks_p"], "%.3g")))
    _finish(fig, ax, ncol=2)
    return fig, ("TARP expected coverage of %s over D_sim's calibration rows: reference points read "
                 "out linearly from the windows' statistics (the mode actually used is in the "
                 "metrics) and drawn uniformly" % e["name"])


def fig_nll(st, e):
    """acc_nll: l_i(E) over every row of D_sim in the box, the interior rows
    and the edge rows, on one scale."""
    a, j = st.arr("%s_sim" % e["stem"]), st.js("%s_sim" % e["stem"])
    l, edge = a["nll_rows"], a["edge"]
    ok = np.isfinite(l)
    col = F21.arm_color(e["arm"])
    fig, axs = F21.panels(3, 3, w=2.5, h=2.2, sharex=True)
    lo, hi = (float(np.nanmin(l[ok])), float(np.nanmax(l[ok]))) if ok.any() else (0.0, 1.0)
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    ed = j["edge"]
    sets = (("every row", ok, j["nll"]["mean"], j["nll"]["n_rows"], col),
            ("interior rows", ok & ~edge, ed["nll_int_mean"], ed["nll_int_n"], col),
            ("edge rows", ok & edge, ed["nll_edge_mean"], ed["nll_edge_n"], F21.INK2))
    for ax, (title, sel, m, n, c) in zip(axs, sets):
        if sel.any():
            ax.hist(l[sel], bins=30, range=(lo, hi), color=c, lw=0)
        else:
            F21.note(ax, "no row", "upper right")
        ax.set_title("%s: %s rows, mean %s" % (title, n, _f(m)))
        ax.set_xlabel("NLL per window (nats)")
    axs[0].set_ylabel("windows")
    _finish(fig)
    return fig, ("NLL -log q_E(theta_i | x_i) of %s on every row of D_sim whose theta is in the "
                 "open box, then on its interior rows and on its edge rows (R1), with the means the "
                 "metrics report" % e["name"])


def fig_contraction(st, e, banks):
    """acc_contraction: per axis D_sim's contraction (block 20's, over the
    calibration rows) and each bank's mean chi over its windows with the 5 %
    to 95 % range of chi over the windows."""
    j = st.js("%s_sim" % e["stem"])
    names = j["recovery"]["axes"]
    p = len(names)
    col = F21.arm_color(e["arm"])
    fig, ax = plt.subplots(figsize=(max(3.6, 0.42 * p + 1.4), 2.8))
    x = np.arange(p, dtype=np.float64)
    nb = len(banks)
    width = 0.6
    off = [(-width / 2.0 + width * (k + 1) / (nb + 1.0)) for k in range(nb + 1)]
    ax.scatter(x + off[0], F21.num(j["recovery"]["contraction"]), s=22, color=F21.SIM_GRAY,
               marker=SIM_MARKER, edgecolors=F21.INK2, linewidths=0.5, zorder=3,
               label="D_sim (calibration rows)")
    for k, r in enumerate(banks):
        jr = st.js("%s_%s" % (e["stem"], r))
        po = jr["posterior"]
        xs = x + off[k + 1]
        ax.vlines(xs, F21.num(po["contraction_q05"]), F21.num(po["contraction_q95"]), color=col,
                  lw=F21.LW_THIN, zorder=2)
        ax.scatter(xs, F21.num(po["contraction_mean"]), s=22, color=col,
                   marker=BANK_MARKERS[k % len(BANK_MARKERS)], edgecolors=F21.SURFACE,
                   linewidths=0.6, zorder=3, label="%s (mean, 5-95 %% over windows)" % r)
    ax.axhline(0.0, color=F21.MUTED, lw=F21.LW_REF)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylabel("contraction 1 - sigma^2 / sigma_0^2")
    _finish(fig, ax, ncol=3)
    return fig, ("Contraction per axis of %s: on D_sim over the calibration rows (gray), and on "
                 "each recording bank the mean of chi (R3) over its windows with the 5 %% to 95 %% "
                 "range over the windows; 0 is as wide as the prior" % e["name"])


def fig_gate_null(st, e, banks):
    """gate_null: T on the null rows, the check rows and each bank's
    windows, one panel per bank (null and check rows in each as context)."""
    a, j = st.arr("%s_sim" % e["stem"]), st.js("%s_sim" % e["stem"])
    Tn, Tc = a["gate_T_null"], a["gate_T_check"]
    col = F21.arm_color(e["arm"])
    pans = list(banks) or [None]
    fig, axs = F21.panels(len(pans), min(4, len(pans)), w=3.4, h=2.3, sharex=True)
    bins = np.linspace(0.0, 1.0, 21)
    for ax, r in zip(axs, pans):
        ax.hist(Tn, bins=bins, density=True, color=F21.SIM_GRAY, lw=0,
                label="null rows (%d)" % Tn.size)
        if Tc.size:
            ax.hist(Tc, bins=bins, density=True, histtype="step", color=F21.INK2, lw=F21.LW_THIN,
                    label="check rows (%d)" % Tc.size)
        if r is not None:
            Tr = st.arr("%s_%s" % (e["stem"], r))["gate_T"]
            ax.hist(Tr, bins=bins, density=True, histtype="step", color=col, lw=F21.LW,
                    label="a bank's windows")
            ax.set_title("bank %s: %d windows" % (r, Tr.size))
        ax.set_xlim(0, 1)
        ax.set_xlabel("T(x), the members' mean rank")
    axs[0].set_ylabel("density")
    _finish(fig, axs[0], ncol=3)
    return fig, ("The gate's statistic T (R7) of %s on its %d null rows (gray), its %d check rows "
                 "and each recording bank's windows; a window is flagged by the gate when fewer "
                 "than k_gate null values are at least its T" % (e["name"], Tn.size, Tc.size))


# ---------------------------------------------------------------- a recording bank

def fig_gate_wells(st, e, r):
    """gate_wells_<r>: each well's share of windows flagged by the gate, by
    class, with alpha_gate and the size band (D-165)."""
    jr = st.js("%s_%s" % (e["stem"], r))
    wells, g = jr["wells"], jr["gate"]
    order, x, groups = _well_order(wells)
    fig, ax = plt.subplots(figsize=(min(12.0, max(5.2, 0.09 * x.size + 2.0)), 2.8))
    if g["size_lo"] is not None:
        ax.axhspan(g["size_lo"], g["size_hi"], color=F21.MUTED, alpha=F21.WASH, lw=0,
                   label="size band at %.2f" % g["level"])
    if g["size_mean"] is not None:
        ax.axhline(g["size_mean"], color=F21.MUTED, lw=F21.LW_REF,
                   label="k_gate / (N_null + 1)")
    ax.axhline(g["alpha"], color=F21.INK2, lw=F21.LW_REF, ls="--", label="alpha_gate")
    for c in jr["classes"]:
        sel = [k for k, i in enumerate(order) if wells[i]["cls"] == c]
        colr, mk = _cls_style(c)
        ax.scatter(x[sel], [wells[order[k]]["share_flagged"] for k in sel], s=20, color=colr,
                   marker=mk, edgecolors=F21.SURFACE, linewidths=0.6, zorder=3,
                   label="class %d" % c)
    _class_bands(ax, groups)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([])
    ax.set_xlabel("wells, by class and batch (%d wells)" % len(wells))
    ax.set_ylabel("share of windows flagged")
    _finish(fig, ax, ncol=4)
    return fig, ("Per well of bank %s, the share of its windows flagged by the gate under %s, by "
                 "class; a share is read against the size band (N_null = %d) and not against "
                 "alpha_gate alone, the windows of one well sharing a culture" % (r, e["name"],
                                                                                 g["n_null"]))


def fig_gate_windows(st, e, r):
    """gate_windows_<r>: p(x) per window, grouped by well, the flagged ones
    ringed, with alpha_gate."""
    jr = st.js("%s_%s" % (e["stem"], r))
    ar = st.arr("%s_%s" % (e["stem"], r))
    x, ticks, groups = _window_positions(ar["well"], jr["wells"])
    pv, flag, cls = ar["gate_p"], ar["gate_flag"], ar["cls"]
    fig, ax = plt.subplots(figsize=(min(12.0, max(5.2, 0.05 * x.size + 2.0)), 2.8))
    for c in jr["classes"]:
        sel = cls == c
        colr, mk = _cls_style(c)
        ax.scatter(x[sel], pv[sel], s=10, color=colr, marker=mk, lw=0, zorder=3,
                   label="class %d" % c)
    ax.scatter(x[flag], pv[flag], s=10 * FLAG_RING ** 2, facecolors="none", edgecolors=F21.INK,
               linewidths=0.8, zorder=4, label="flagged by the gate")
    ax.axhline(jr["gate"]["alpha"], color=F21.INK2, lw=F21.LW_REF, ls="--", label="alpha_gate")
    _class_bands(ax, groups)
    ax.set_yscale("log")
    ax.set_ylim(top=1.3)
    ax.set_xticks([])
    ax.set_xlabel("windows, grouped by well (%d windows, %d wells)" % (x.size, len(ticks)))
    ax.set_ylabel("p(x)")
    _finish(fig, ax, ncol=5)
    return fig, ("The gate's p-value (R8) of every window of bank %s under %s, the windows grouped "
                 "by well and the wells by class and batch, the windows flagged by the gate ringed"
                 % (r, e["name"]))


def fig_gate_members(st, e, r):
    """gate_members_<r>: r_m per member and window, one panel per member."""
    jr = st.js("%s_%s" % (e["stem"], r))
    ar = st.arr("%s_%s" % (e["stem"], r))
    x, _ticks, groups = _window_positions(ar["well"], jr["wells"])
    R, flag, cls = ar["gate_r"], ar["gate_flag"], ar["cls"]
    M = R.shape[0]
    ncol = min(4, M)
    fig, axs = F21.panels(M, ncol, w=min(6.0, max(2.8, 0.03 * x.size + 1.8)), h=2.1,
                          sharey=True)
    for m, ax in enumerate(axs):
        for c in jr["classes"]:
            sel = cls == c
            colr, mk = _cls_style(c)
            ax.scatter(x[sel], R[m][sel], s=8, color=colr, marker=mk, lw=0, zorder=3,
                       label="class %d" % c)
        ax.scatter(x[flag], R[m][flag], s=8 * FLAG_RING ** 2, facecolors="none",
                   edgecolors=F21.INK, linewidths=0.7, zorder=4, label="flagged by the gate")
        _class_bands(ax, groups)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([])
        ax.set_title("member %d" % (m + 1), pad=12)
    for ax in axs[::ncol]:
        ax.set_ylabel("r_m(x), rank among rank rows")
    for ax in axs[-ncol:]:
        ax.set_xlabel("windows, grouped by well")
    _finish(fig, axs[0], ncol=5)
    return fig, ("Each member's rank r_m of every window of bank %s among the rank rows, under %s "
                 "(the ensemble statistic T is their mean, R7), the windows grouped by well, those "
                 "flagged by the gate ringed" % (r, e["name"]))


def fig_emb_pca(st, e, r):
    """emb_pca_<r>: each member's first two principal components of the
    reference rows' and the bank's embeddings (block 36's coordinates)."""
    jr = st.js("%s_%s" % (e["stem"], r))
    ar = st.arr("%s_%s" % (e["stem"], r))
    xy_all, evr, src = ar["pca_coords"], ar["pca_evr"], ar["pca_source"]
    cls = ar["cls"]
    M = xy_all.shape[0]
    fig, axs = F21.panels(M, min(4, M), w=2.5, h=2.4)
    for m, ax in enumerate(axs):
        xy = xy_all[m]
        ref, win = xy[src == 0], xy[src == 1]
        ax.scatter(ref[:, 0], ref[:, 1], s=6, color=F21.SIM_GRAY, lw=0,
                   label="D_sim, the gate's reference rows")
        for c in jr["classes"]:
            sel = cls == c
            colr, mk = _cls_style(c)
            ax.scatter(win[sel, 0], win[sel, 1], s=12, color=colr, marker=mk,
                       edgecolors=F21.SURFACE, linewidths=0.4, label="%s class %d" % (r, c))
        ax.set_title("member %d" % (m + 1))
        ax.set_xlabel("PC 1 (%.0f %%)" % (100 * evr[m][0]))
        ax.set_ylabel("PC 2 (%.0f %%)" % (100 * evr[m][1]))
    _finish(fig, axs[0], ncol=4)
    return fig, ("Each member's embeddings h_psi(x) of the gate's %d reference rows of D_sim (gray) "
                 "and of bank %s's windows by class, on the first two principal components of the "
                 "two pooled, under %s" % (jr["pca"]["n_ref"], r, e["name"]))


def fig_rec_wells(st, e, r):
    """rec_wells_<r>: per axis, the mixture over each well's windows (R4):
    its median and its 5 % to 95 % range, by class and batch, on the prior's
    range."""
    jr = st.js("%s_%s" % (e["stem"], r))
    wells = jr["wells"]
    names, bounds = jr["axes"]["names"], jr["axes"]["bounds"]
    order, x, groups = _well_order(wells)
    p = len(names)
    ncol = min(4, p)
    fig, axs = F21.panels(p, ncol, w=min(6.0, max(2.6, 0.06 * x.size + 1.6)), h=2.0,
                          sharex=True)
    batch_rank = {}
    for c, b, _x0, _x1 in groups:                       # alternate fills within a class
        batch_rank[(c, b)] = sum(1 for k in batch_rank if k[0] == c)
    for k, ax in enumerate(axs):
        for c in jr["classes"]:
            colr, mk = _cls_style(c)
            sel = [q for q, i in enumerate(order) if wells[i]["cls"] == c]
            if not sel:
                continue
            ii = [order[q] for q in sel]
            ax.vlines(x[sel], [wells[i]["q05"][k] for i in ii], [wells[i]["q95"][k] for i in ii],
                      color=colr, lw=F21.LW_THIN, alpha=0.8, zorder=2)
            filled = [q for q, i in zip(sel, ii)
                      if batch_rank[(c, wells[i]["batch"])] % 2 == 0]
            hollow = [q for q, i in zip(sel, ii)
                      if batch_rank[(c, wells[i]["batch"])] % 2 == 1]
            if filled:
                ax.scatter(x[filled], [wells[order[q]]["q50"][k] for q in filled], s=14,
                           color=colr, marker=mk, edgecolors=F21.SURFACE, linewidths=0.4,
                           zorder=3, label="class %d" % c)
            if hollow:
                ax.scatter(x[hollow], [wells[order[q]]["q50"][k] for q in hollow], s=14,
                           facecolors=F21.SURFACE, edgecolors=colr, marker=mk, linewidths=0.8,
                           zorder=3)
        _class_bands(ax, groups)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_title(_axis_label(names, bounds, k), pad=11)
    for ax in axs[::ncol]:
        ax.set_ylabel("unit box")
    for ax in axs[-ncol:]:
        ax.set_xlabel("wells, by class and batch")
    _finish(fig, axs[0], ncol=4)
    return fig, ("Per axis, the mixture over the well's windows (R4) of every well of bank %s under "
                 "%s: its median (dot) and its 5 %% to 95 %% range, the wells by class and then by "
                 "batch (thin lines between batches, filled and open dots alternating), on the "
                 "prior's range; it is not a posterior of the well's parameter and does not narrow "
                 "with more windows (D-160)" % (r, e["name"]))


def fig_rec_corner(e, r, ex, draws, well_entry, names):
    """rec_corner_<r>_c<k>: the pooled draws of class k's example well (the
    mixture over the well's windows), its median per axis (the metrics')
    marked."""
    d = draws.reshape(-1, draws.shape[-1])
    p = len(names)
    q50 = well_entry["q50"]
    col = F21.arm_color(e["arm"])
    fig, axes = plt.subplots(p, p, figsize=(1.15 * p, 1.15 * p), squeeze=False)
    for i in range(p):
        for k in range(p):
            ax = axes[i, k]
            ax.grid(False)
            if k > i:
                ax.set_visible(False)
                continue
            if i == k:
                ax.hist(d[:, i], bins=30, range=(0, 1), color=col, lw=0)
                ax.axvline(q50[i], color=F21.INK, lw=0.8)
                ax.set_yticks([])
            else:
                ax.hist2d(d[:, k], d[:, i], bins=24, range=[[0, 1], [0, 1]], cmap=F21.SEQ,
                          cmin=1)
                ax.plot([q50[k]], [q50[i]], marker="+", color=F21.INK, ms=6, mew=1.0)
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
    _finish(fig)
    return fig, ("Corner plot of the mixture over the well's windows (R4) for class %d's example "
                 "well %d of bank %s under %s (the well whose mean T is its class's median): the "
                 "%d pooled draws of its %d windows in the unit box, its median per axis marked"
                 % (ex["cls"], ex["well"], r, e["name"], d.shape[0], well_entry["n"]))


# ---------------------------------------------------------------- the run

def _expected_index(st):
    """Refusals on a breach of block 36's contract, before anything is
    drawn."""
    ens = st.index.get("ensembles") or {}
    if not ens:
        raise RefuseError("%s lists no ensemble" % os.path.join(st.mdir, "METRICS.json"))
    for name, info in sorted(ens.items()):
        for D in info["datasets"]:
            for ext in ("json", "npz"):
                p = os.path.join(st.mdir, "%s_%s.%s" % (info["stem"], D, ext))
                if not os.path.isfile(p):
                    raise RefuseError("%s: %s is absent (block 36's metrics)" % (name, p))


def make_all(cfg_path, ecfg_path, log=print):
    """Draws every figure of the evaluation at `ecfg_path` for the project
    at `cfg_path` from the metrics present; writes them and FIGURES.md.
    Returns the list of (name, caption): `name` the figure's path relative
    to <out_root>/figures/ without its extension, with "/" as separator
    (ensembles/<stem>/cal_coverage), in the order of FIGURES.md; `caption`
    one sentence. ev_plan.RefuseError when <out_root>/metrics/METRICS.json
    is absent or lists no ensemble."""
    t0 = time.time()
    ecfg = ev_rec_config.load(ecfg_path)
    mdir = ecfg.sub("metrics")
    if not os.path.isfile(os.path.join(mdir, "METRICS.json")):
        raise RefuseError("no metrics at %s (block 36's ev_rec_metrics.py first)" % mdir)
    st = F21.Store(ecfg)
    _expected_index(st)
    F21.style()
    fdir = ecfg.sub("figures")
    index, sections = [], []

    def save(fig, name, caption):
        base = os.path.join(fdir, *name.split("/"))
        os.makedirs(os.path.dirname(base), exist_ok=True)
        for f in ecfg.formats:
            fig.savefig("%s.%s" % (base, f), dpi=ecfg.dpi)
        plt.close(fig)
        index.append((name, _sentence(caption)))
        sections[-1][1].append(len(index) - 1)

    for name in sorted(st.index["ensembles"]):
        info = st.index["ensembles"][name]
        e = dict(info, name=name)
        stem, banks = e["stem"], [D for D in info["datasets"] if D != "sim"]
        rel = "ensembles/%s/" % stem
        sections.append(("%s (`ensembles/%s/`)" % (name, stem), []))
        log("figures of %s: D_sim and %s" % (name, ", ".join(banks) or "no recording bank"))
        draw = {"cal_coverage": lambda: fig_coverage(st, e),
                "cal_sbc_ecdf": lambda: fig_sbc_ecdf(st, e),
                "cal_sbc_hist": lambda: _u(F21.fig_sbc_hist(st, e, "sim")),
                "cal_tarp": lambda: fig_tarp(st, e),
                "acc_recovery": lambda: _u(F21.fig_recovery(st, e, "sim")),
                "acc_zscore": lambda: _u(F21.fig_zscore(st, e, "sim")),
                "acc_nll": lambda: fig_nll(st, e),
                "acc_contraction": lambda: fig_contraction(st, e, banks),
                "gate_null": lambda: fig_gate_null(st, e, banks)}
        for fname in ENSEMBLE_FIGURES:
            f, c = draw[fname]()
            save(f, rel + fname, c)
        per_bank = (fig_gate_wells, fig_gate_windows, fig_gate_members, fig_emb_pca,
                    fig_rec_wells)
        for r in banks:
            for pattern, fn in zip(BANK_FIGURES, per_bank):
                f, c = fn(st, e, r)
                save(f, rel + pattern % r, c)
            jr = st.js("%s_%s" % (stem, r))
            exs = jr["example_wells"]
            if exs:
                z = st.sample(stem, r)                 # block 35's draws: the corners alone
                well, draws = z["well"], z["draws"]
                by_well = dict((w["well"], w) for w in jr["wells"])
                for ex in exs:
                    rows = np.flatnonzero(well == ex["well"])
                    f, c = fig_rec_corner(e, r, ex, draws[rows], by_well[ex["well"]],
                                          jr["axes"]["names"])
                    save(f, rel + CORNER_FIGURE % (r, ex["cls"]), c)
                del z, draws
    # the comparisons (block 28's figures, on D_sim)
    rows = []
    comps = st.index.get("comparisons") or []
    if comps:
        sections.append(("comparisons (`comparisons/<name>/`)", []))
    for cm in comps:
        eE, eR = F21.comparison_ensembles(st.index, cm)
        F21.paired_datasets(cm, eE, eR)
        rel = "comparisons/%s/" % cm["name"]
        f, _c = _u(F21.fig_cmp_nll(st, cm, eE, eR, "sim"))
        f.axes[0].set_xlabel("simulated network (ordered by its mean NLL under %s)"
                             % F21.ens_label(eR))
        save(f, rel + "cmp_nll", "Mean NLL per simulated network of D_sim of %s against %s, "
                                 "each network's pair joined and the networks ordered by %s's"
             % (eE["name"], eR["name"], eR["name"]))
        j, a = F21.paired_of(st, cm, "sim")
        f, c = _u(F21.fig_cmp_forest(cm, eE, eR, "sim", j, a))
        f.axes[0].set_ylabel("simulated network")
        save(f, rel + "cmp_forest", c.replace("per donor", "per simulated network")
             .replace("donor-cluster", "network-cluster"))
        rows.append(("%s, sim" % cm["name"], j, cm))
    if rows:
        sections.append(("across (`across/`)", []))
        f, c = _u(F21.fig_dbar_all(st, rows))
        save(f, "across/dbar", c.replace("donor-cluster", "network-cluster"))
    _write_index(ecfg, st, fdir, index, sections)
    log("REC FIGURES OK: %d figure(s) in %s (%.0f s)" % (len(index), fdir, time.time() - t0))
    return index


def _write_index(ecfg, st, fdir, index, sections):
    def named(entries, key):
        return "; ".join("%s (%s)" % (x[key], x["why"]) for x in entries) or "none"

    lines = ["# The figures of a recorded project's evaluation", "",
             "Written by `hpc/pipeline/evaluation/ev_rec_figures.py` (block 37) from the metrics "
             "in `%s`; formats %s at %d dpi. Left out of the metrics: %s. Comparisons left out: "
             "%s." % (st.mdir, ", ".join(ecfg.formats), ecfg.dpi,
                      named(st.index.get("left_out") or [], "ensemble"),
                      named(st.index.get("comparisons_left_out") or [], "comparison")), ""]
    for title, ids in sections:
        lines += ["## %s" % title, ""]
        lines += ["- `%s.{%s}`: %s." % (index[i][0], ",".join(ecfg.formats), index[i][1])
                  for i in ids]
        lines.append("")
    os.makedirs(fdir, exist_ok=True)
    tmp = os.path.join(fdir, FIGURES_INDEX + ".tmp")
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines))
    os.replace(tmp, os.path.join(fdir, FIGURES_INDEX))


def main(argv=None):
    """The command line: 0 with the pass line, 2 on the arguments,
    EXIT_REFUSE on a refusal."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the project's configuration (<tag>.json)")
    ap.add_argument("--eval-config", required=True,
                    help="the evaluation's configuration (<tag>_rec_eval.json)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    try:
        make_all(args.config, args.eval_config, log=lambda t: print(t, flush=True))
    except (RefuseError, ev_rec_config.RecEvalConfigError) as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
