#!/usr/bin/env python3
"""Block 23: the data overview's figures, from ev_dataview.py's
data_view.npz (specs/SPEC.md block 23; D-130).

    ev_data_figures.make_all(<out_root>/data, ecfg)     # called by ev_dataview.run

Writes <out_root>/data/figures/<name>.<fmt> (formats and dpi from the
evaluation's configuration) and <out_root>/data/FIGURES.md, one caption line
per figure:
    windows_cohorts        D1: example windows of each cohort by condition, full length
    windows_cohorts_zoom   D1: the same windows around their maximum
    windows_sim            D1: windows of S_sub, full and zoomed
    windows_contaminated   D1: contaminated windows of a cohort, full and zoomed (when any)
    stats_hist             D2: the eight statistics, S_sub's histogram and the cohorts'
    stats_strip            D2: every cohort window against S_sub's 1-99 % band
    theta_cohorts          D3: the cohorts' theta* on the unit box, by condition
    theta_dagger           D3: theta-dagger against theta* on the shifted axes (when any)
    stats_pca              D4: S_sub's PCA plane with the cohorts' windows; the loadings

It draws and computes nothing scientific: every quantity, frame, histogram
height, jitter and zoom is ev_dataview's. Block 21's style and colours: a
condition keeps its hue and marker, S_sub is gray, the cohorts are told
apart by line style or panel, text stays in the ink colours, one y-axis per
panel, the legend in a row below the panels.

Pure ASCII, LF only.
"""

import math
import os
import textwrap
import time

import _evalpath  # noqa: F401

import ev_figures as F
from ev_figures import (COND_COLOR, COND_MARKER, GRID, INK, INK2, LW, LW_REF, LW_THIN, MUTED,
                        SIM_GRAY, SURFACE, note, panels, plt)
from matplotlib.ticker import MaxNLocator
import numpy as np

NPZ = "data_view.npz"
COHORT_LS = ("-", "--", ":")         # the cohorts' line styles, in the configuration's order
TRACE_LW = 0.5
SIM_TRACE = INK2                     # S_sub's windows: a neutral, as D_sim's gray in block 21


def handles(axs):
    """The legend entries of every axes, each label once, in order."""
    hs, ls = [], []
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h)
                ls.append(l)
    return hs, ls


def finish(fig, title, axs=(), ncol=4):
    """Block 21's finish with the title wrapped to the figure's width and the
    legend -- every axes' entries, once each -- in rows below the panels;
    margins in inches, the panels keep their size."""
    hs, ls = handles(axs)
    lines = textwrap.wrap(title, width=max(40, int(fig.get_figwidth() * 15)))
    t_in = 0.08 + 0.16 * len(lines)
    rows = int(math.ceil(len(hs) / float(ncol))) if hs else 0
    b_in = 0.06 + 0.2 * rows if hs else 0.0
    h = fig.get_figheight() + t_in + b_in
    fig.set_figheight(h)
    sup = fig.suptitle("\n".join(lines), color=INK, fontsize=9, y=1.0 - 0.04 / h, va="top")
    sup.set_in_layout(False)            # its room is t_in: tight_layout would count it twice
    if hs:
        fig.legend(hs, ls, loc="lower center", ncol=min(ncol, len(hs)), frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0.0, b_in / h, 1.0, 1.0 - t_in / h))


def cond_color(c):
    return COND_COLOR[int(c) % len(COND_COLOR)]


def cond_marker(c):
    return COND_MARKER[int(c) % len(COND_MARKER)]


class Data(object):
    """data_view.npz, read once."""

    def __init__(self, ddir):
        with np.load(os.path.join(ddir, NPZ), allow_pickle=False) as z:
            self.a = dict((k, z[k]) for k in z.files)
        self.names = [str(n) for n in self.a["cohort_names"]]
        self.stats = [str(n) for n in self.a["stat_names"]]
        self.fs = float(self.a["fs"])
        self.W = int(self.a["W"])
        self.L = int(self.a["zoom_len"])
        self.t = np.arange(self.W) / self.fs
        self.xlab = str(self.a["x_label"])

    def __getitem__(self, k):
        return self.a[k]


# ---------------------------------------------------------------- D1: windows

def _trace(ax, d, x, z0, color, zoom):
    if zoom:
        sl = slice(int(z0), int(z0) + d.L)
        ax.plot(d.t[sl], x[sl], color=color, lw=TRACE_LW + 0.3, rasterized=True)
        ax.set_xlim(d.t[sl][0], d.t[sl][-1])
    else:
        ax.axvspan(z0 / d.fs, (z0 + d.L) / d.fs, color=GRID, lw=0, zorder=0)
        ax.plot(d.t, x, color=color, lw=TRACE_LW, rasterized=True)
        ax.set_xlim(0, d.W / d.fs)


def _example_label(d, n, kind, i):
    r = int(d[n + "_%s_rows" % kind][i])
    return ("%s, condition %d: donor %d, well %d, window %d"
            % (n, int(d[n + "_cls"][r]), int(d[n + "_donor"][r]), int(d[n + "_well"][r]),
               int(d[n + "_widx"][r])))


def fig_windows_cohorts(d, zoom):
    """Rows: (condition, example); columns: the cohorts."""
    keys = []
    slot = {}
    for ci, n in enumerate(d.names):
        rows = d[n + "_ex_rows"]
        seen = {}
        for i, r in enumerate(rows):
            c = int(d[n + "_cls"][r])
            j = seen.get(c, 0)
            seen[c] = j + 1
            slot[(c, j, ci)] = i
            if (c, j) not in keys:
                keys.append((c, j))
    keys.sort()
    nr, nc = len(keys), len(d.names)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 1.0 * nr + 0.3), squeeze=False,
                             sharey="row", sharex=not zoom)
    for ri, (c, j) in enumerate(keys):
        for ci, n in enumerate(d.names):
            ax = axes[ri, ci]
            i = slot.get((c, j, ci))
            if i is None:
                ax.set_visible(False)
                continue
            _trace(ax, d, d[n + "_ex_x"][i], d[n + "_ex_z0"][i], cond_color(c), zoom)
            ax.set_title(_example_label(d, n, "ex", i), fontsize=6.5, loc="left", pad=2)
            if ci == 0:
                ax.set_ylabel("x", labelpad=2)
            if ri == nr - 1:
                ax.set_xlabel("t (s)")
    what = ("%g s around each window's maximum" % (d.L / d.fs)) if zoom else \
        "the full %g s, the zoom shaded" % (d.W / d.fs)
    finish(fig, "Example windows of %s by condition: %s; %s" % (", ".join(d.names), what, d.xlab))
    k = max(j for _c, j in keys) + 1
    return fig, ("%d window(s) per condition of each cohort (%s), each the first window of the "
                 "first well of a donor drawn with the evaluation's seed: %s; y is %s"
                 % (k, ", ".join(d.names), what, d.xlab))


def fig_windows_pairs(d, which):
    """Rows: examples; columns: full, zoom. which = "sim" or "contaminated"."""
    items = []
    if which == "sim":
        for i in range(d["sim_ex_rows"].size):
            items.append((d["sim_ex_x"][i], int(d["sim_ex_z0"][i]), SIM_TRACE,
                          "the simulated bank's row %d: donor %d"
                          % (int(d["sim_ex_bank_rows"][i]), int(d["sim_ex_donor"][i]))))
    else:
        for n in d.names:
            for i in range(d[n + "_cx_rows"].size):
                r = int(d[n + "_cx_rows"][i])
                items.append((d[n + "_cx_x"][i], int(d[n + "_cx_z0"][i]),
                              cond_color(d[n + "_cls"][r]),
                              _example_label(d, n, "cx", i) + " (contaminated)"))
    nr = len(items)
    fig, axes = plt.subplots(nr, 2, figsize=(7.2, 1.0 * nr + 0.3), squeeze=False,
                             sharey="row", gridspec_kw={"width_ratios": (2.2, 1.0)})
    for ri, (x, z0, color, label) in enumerate(items):
        for zi, zoom in enumerate((False, True)):
            ax = axes[ri, zi]
            _trace(ax, d, x, z0, color, zoom)
            if zi == 0:
                ax.set_title(label, fontsize=6.5, loc="left", pad=2)
                ax.set_ylabel("x", labelpad=2)
            if ri == nr - 1:
                ax.set_xlabel("t (s)")
    if which == "sim":
        title = ("Windows of S_sub (theta from the box prior), full and %g s around the "
                 "maximum; %s" % (d.L / d.fs, d.xlab))
        cap = ("%d windows of S_sub drawn with the evaluation's seed, each a draw of the prior "
               "predictive: full length (the zoom shaded) and %g s around its maximum; y is %s"
               % (nr, d.L / d.fs, d.xlab))
    else:
        title = ("Contaminated windows (the gap's mode d), full and %g s around the maximum; %s"
                 % (d.L / d.fs, d.xlab))
        cap = ("%d contaminated window(s) of the cohorts (gap mode d, the bank's contaminated "
               "flag) drawn with the evaluation's seed: full length and %g s around the maximum"
               % (nr, d.L / d.fs))
    finish(fig, title)
    return fig, cap


# ---------------------------------------------------------------- D2: statistics

def fig_stats_hist(d):
    fig, axs = panels(len(d.stats), 4, w=2.3, h=1.9)
    for j, ax in enumerate(axs):
        e = d["edges"][j]
        ax.stairs(d["hist_sim"][j], e, fill=True, color=SIM_GRAY, lw=0,
                  label="S_sub (prior predictive)")
        notes = []
        for ci, n in enumerate(d.names):
            ax.stairs(d[n + "_hist"][j], e, color=INK, lw=LW_THIN + 0.2,
                      ls=COHORT_LS[ci % len(COHORT_LS)], label=n)
            out = int(d[n + "_below_frame"][j]) + int(d[n + "_above_frame"][j])
            if out:
                notes.append("%s %d" % (n, out))
        ax.set_xlim(e[0], e[-1])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel(d.stats[j])
        if j % 4 == 0:
            ax.set_ylabel("density")
        if notes:
            ax.set_title("beyond the frame: " + ", ".join(notes), fontsize=6.5, color=INK2)
    finish(fig, "The eight window statistics: S_sub against the cohorts", axs, ncol=4)
    return fig, ("Each window statistic's density over S_sub (gray, the prior predictive of "
                 "the simulated arm) and over each cohort (%s; ink, line style per cohort), on "
                 "a frame holding S_sub and every uncontaminated cohort window; a cohort's "
                 "windows beyond the frame, all contaminated, are counted in the panel"
                 % ", ".join(d.names))


def _rows(d):
    """[(cohort index, cohort, condition, y)], top to bottom."""
    out, y = [], 0.0
    for ci, n in enumerate(d.names):
        for c in sorted(set(int(v) for v in d[n + "_cls"])):
            out.append((ci, n, c, y))
            y -= 1.0
        y -= 0.6
    return out


def _edge_points(ax, xv, yv, lo, hi):
    """Points beyond [lo, hi] drawn at the edge, pointing out."""
    for sel, mk, xe in ((xv < lo, "<", lo), (xv > hi, ">", hi)):
        if np.any(sel):
            ax.scatter(np.full(int(sel.sum()), xe), yv[sel], marker=mk, s=16, color=INK, lw=0,
                       zorder=4, clip_on=False, label="beyond the frame")


def fig_stats_strip(d):
    rows = _rows(d)
    fig, axs = panels(len(d.stats), 4, w=2.3, h=0.32 * len(rows) + 0.9)
    q = d["q_sim"]
    for j, ax in enumerate(axs):
        lo, hi = d["frame_T"][j]
        ax.axvspan(q[0][j], q[-1][j], color=GRID, lw=0, zorder=0,
                   label="S_sub 1-99 %")
        ax.axvline(q[1][j], color=MUTED, lw=LW_REF, zorder=1, label="S_sub median")
        notes = []
        for ci, n, c, y in rows:
            T = d[n + "_T"][:, j].astype(np.float64)
            sel = d[n + "_cls"] == c
            xv, yv = T[sel], y + d[n + "_jit"][sel]
            inside = (xv >= lo) & (xv <= hi)
            ax.scatter(xv[inside], yv[inside], s=6, color=cond_color(c), marker=cond_marker(c),
                       lw=0, alpha=0.75, zorder=2, label="condition %d" % c if ci == 0 else None)
            cont = d[n + "_contam"][sel] & inside
            if np.any(cont):
                ax.scatter(xv[cont], yv[cont], s=26, facecolors="none", edgecolors=INK, lw=0.6,
                           zorder=3, label="contaminated")
            _edge_points(ax, xv, yv, lo, hi)
        for ci, n in enumerate(d.names):
            notes.append("%s %d | %d" % (n, int(d[n + "_below_q"][j]), int(d[n + "_above_q"][j])))
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylim(rows[-1][3] - 0.6, 0.6)
        ax.set_yticks([y for _ci, _n, _c, y in rows])
        ax.set_yticklabels(["%s c%d" % (n, c) for _ci, n, c, _y in rows] if j % 4 == 0 else [])
        ax.grid(False, axis="y")
        ax.set_xlabel(d.stats[j])
        ax.set_title("below | above 1-99 %%: %s" % ", ".join(notes), fontsize=6.5, color=INK2)
    finish(fig, "Every cohort window's statistics against S_sub's 1-99 % band", axs, ncol=7)
    return fig, ("Each cohort window (rows: cohort and condition, jittered) by statistic, "
                 "against S_sub's 1-99 % band (shaded) and median (line); contaminated windows "
                 "ringed, those beyond the frame drawn at its edge as arrowheads; each panel's "
                 "title counts each cohort's windows below and above the band")


# ---------------------------------------------------------------- D3: parameters

def fig_theta_cohorts(d):
    rows = []
    y = 0.0
    for ci, n in enumerate(d.names):
        for c in sorted(set(int(v) for v in d[n + "_cls_d"])):
            rows.append((ci, n, c, y))
            y -= 1.0
        y -= 0.6
    names = [str(v) for v in d["axis_names"]]
    label_idx = [int(v) for v in d["label_idx"]]
    fig, axs = panels(len(names), 5, w=2.0, h=0.3 * len(rows) + 0.9)
    for k, ax in enumerate(axs):
        for ci, n, c, y0 in rows:
            sel = d[n + "_cls_d"] == c
            ax.scatter(d[n + "_theta_d"][sel, k], y0 + d[n + "_jit_d"][sel], s=12,
                       color=cond_color(c), marker=cond_marker(c), lw=0, alpha=0.85, zorder=3,
                       label="condition %d" % c if ci == 0 else None)
            if k in label_idx:
                mu = d["centres"][c, label_idx.index(k)]
                ax.plot([mu, mu], [y0 - 0.42, y0 + 0.42], color=cond_color(c), lw=LW + 0.4,
                        zorder=2)
        if k in label_idx:
            ax.plot([], [], color=INK2, lw=LW + 0.4, label="a condition's centre (in its hue)")
        ax.set_xlim(0, 1)
        ax.set_ylim(rows[-1][3] - 0.6, 0.6)
        ax.set_yticks([y0 for _ci, _n, _c, y0 in rows])
        ax.set_yticklabels(["%s c%d" % (n, c) for _ci, n, c, _y in rows] if k % 5 == 0 else [])
        ax.grid(False, axis="y")
        ax.set_title("%d %s (%s)" % (k, names[k], "label" if k in label_idx else "free"))
        ax.set_xlabel("theta_%d" % k)
    finish(fig, "The cohorts' donors' theta* on the unit box, where the simulated arm's prior "
                "is uniform", axs, ncol=4)
    return fig, ("Each cohort donor's withheld theta*_d (%s; rows: cohort and condition, "
                 "jittered) on every axis of the unit box, on which the simulated arm's box "
                 "prior is uniform; bars: the conditions' centres on the label axes, about which "
                 "the cohorts' theta* are truncated normals (tau_ov); the free axes are uniform "
                 "in every arm" % ", ".join(d.names))


def fig_theta_dagger(d):
    shifted = [n for n in d.names if float(d[n + "_delta"]) > 0]
    free = [int(v) for v in d["free_idx"]]
    names = [str(v) for v in d["axis_names"]]
    fig, axs = panels(len(free) * len(shifted), len(free), w=2.5, h=2.4)
    i = 0
    for n in shifted:
        delta = float(d[n + "_delta"])
        th, dag = d[n + "_theta_d"], d[n + "_theta_dag_d"]
        for k in free:
            ax = axs[i]
            i += 1
            ax.plot([0, 1], [0, 1], color=MUTED, lw=LW_REF, label="theta-dagger = theta*")
            ax.plot([0, 1], [delta, 1 + delta], color=INK2, lw=LW_REF, ls="--",
                    label="theta* + delta")
            ax.axhline(1.0, color=INK2, lw=LW_REF, ls=":", label="the box's edge")
            for c in sorted(set(int(v) for v in d[n + "_cls_d"])):
                sel = d[n + "_cls_d"] == c
                ax.scatter(th[sel, k], dag[sel, k], s=16, color=cond_color(c),
                           marker=cond_marker(c), edgecolors=SURFACE, linewidths=0.4, zorder=3,
                           label="condition %d" % c)
            out = int((dag[:, k] >= 1.0).sum())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(1.0 + delta, float(dag[:, k].max())) + 0.03)
            ax.set_xlabel("theta*_%d" % k)
            ax.set_ylabel("theta-dagger_%d" % k)
            ax.set_title("%s: %d %s" % (n, k, names[k]))
            note(ax, "delta %.3g\n%d of %d at or beyond 1" % (delta, out, th.shape[0]),
                 "lower right")
    ins = "; ".join("%s: %d of %d donors in the open box" % (n, int(d[n + "_in_box_d"].sum()),
                                                            d[n + "_in_box_d"].size)
                    for n in shifted)
    finish(fig, "theta-dagger against theta* on the free axes (%s)" % ins, axs, ncol=6)
    cap = (", the duty capped where the generator clips it" if 7 in free else
           " (the duty is class-bearing here and not shifted, block 25)")
    return fig, ("theta-dagger_d against theta*_d on the free axes for each donor of the "
                 "range-shifted cohort(s) (%s): theta* + delta%s; the donors whose theta-dagger "
                 "reaches the box's edge are those the evaluation scores outside the box (%s)"
                 % (", ".join(shifted), cap, ins))


# ---------------------------------------------------------------- D4: PCA

def _pc_cloud(ax, d, n, clip):
    lo, hi = d["frame_pc"][:, 0], d["frame_pc"][:, 1]
    ax.scatter(d["sim_pc"][:, 0], d["sim_pc"][:, 1], s=2.5, color=SIM_GRAY, lw=0,
               rasterized=True, zorder=1, label="S_sub")
    pc, cls, cont = d[n + "_pc"].astype(np.float64), d[n + "_cls"], d[n + "_contam"]
    out = d[n + "_pc_out"] if clip else np.zeros(cls.size, dtype=bool)
    for c in sorted(set(int(v) for v in cls)):
        sel = (cls == c) & ~out
        ax.scatter(pc[sel, 0], pc[sel, 1], s=12, color=cond_color(c), marker=cond_marker(c),
                   edgecolors=SURFACE, linewidths=0.4, zorder=3, label="condition %d" % c)
    sel = cont & ~out
    if np.any(sel):
        ax.scatter(pc[sel, 0], pc[sel, 1], s=34, facecolors="none", edgecolors=INK, lw=0.6,
                   zorder=4, label="contaminated")
    if clip and np.any(out):
        e = np.clip(pc[out], lo, hi)
        ax.scatter(e[:, 0], e[:, 1], marker="D", s=12, color=INK, lw=0, zorder=5, clip_on=False,
                   label="beyond the frame")
        note(ax, "%d window(s) beyond the frame,\nat its edge; full extent at right"
             % int(out.sum()), "lower right")
    if clip:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
    evr = d["pca_evr"]
    ax.set_xlabel("PC 1 (%.0f %% of S_sub's variance)" % (100 * evr[0]))
    ax.set_ylabel("PC 2 (%.0f %%)" % (100 * evr[1]))
    ax.set_title("%s on S_sub%s" % (n, "" if clip else ", full extent"))


def fig_stats_pca(d):
    spec = []
    for n in d.names:
        spec.append(("cloud", n, True))
        if np.any(d[n + "_pc_out"]):
            spec.append(("cloud", n, False))
    spec.append(("loadings", None, None))
    fig, axs = panels(len(spec), min(4, len(spec)), w=2.7, h=2.6)
    for ax, (kind, n, clip) in zip(axs, spec):
        if kind == "cloud":
            _pc_cloud(ax, d, n, clip)
            continue
        comp = d["pca_components"]
        y = np.arange(len(d.stats))[::-1]
        ax.barh(y + 0.2, comp[0], height=0.38, color=INK2, label="PC 1 loading")
        ax.barh(y - 0.2, comp[1], height=0.38, color=MUTED, label="PC 2 loading")
        ax.axvline(0, color=INK2, lw=LW_REF)
        ax.set_yticks(y)
        ax.set_yticklabels(d.stats)
        ax.grid(False, axis="y")
        ax.set_xlabel("loading (on z = (T - m) / s)")
        ax.set_title("the components")
    finish(fig, "The eight statistics standardised on S_sub, on S_sub's first two principal "
                "components", axs, ncol=8)
    return fig, ("The windows' eight statistics, standardised by S_sub's mean and SD, on the "
                 "first two principal components of S_sub (gray): each cohort's windows by "
                 "condition, contaminated ones ringed, those beyond the frame at its edge and in "
                 "a full-extent panel; the loadings of the two components")


# ---------------------------------------------------------------- all

def make_all(ddir, ecfg, log=print):
    """Every figure of the data overview from <ddir>/data_view.npz; returns
    [(name, caption)]."""
    t0 = time.time()
    d = Data(ddir)
    F.style()
    fdir = os.path.join(ddir, "figures")
    os.makedirs(fdir, exist_ok=True)
    index = []

    def save(fig, name, caption, family):
        for f in ecfg.formats:
            fig.savefig(os.path.join(fdir, "%s.%s" % (name, f)), dpi=ecfg.dpi)
        plt.close(fig)
        index.append((family, name, caption))

    f, c = fig_windows_cohorts(d, zoom=False)
    save(f, "windows_cohorts", c, "D1: windows")
    f, c = fig_windows_cohorts(d, zoom=True)
    save(f, "windows_cohorts_zoom", c, "D1: windows")
    f, c = fig_windows_pairs(d, "sim")
    save(f, "windows_sim", c, "D1: windows")
    if any(d[n + "_cx_rows"].size for n in d.names):
        f, c = fig_windows_pairs(d, "contaminated")
        save(f, "windows_contaminated", c, "D1: windows")
    f, c = fig_stats_hist(d)
    save(f, "stats_hist", c, "D2: statistics")
    f, c = fig_stats_strip(d)
    save(f, "stats_strip", c, "D2: statistics")
    f, c = fig_theta_cohorts(d)
    save(f, "theta_cohorts", c, "D3: parameters")
    if any(float(d[n + "_delta"]) > 0 for n in d.names):
        f, c = fig_theta_dagger(d)
        save(f, "theta_dagger", c, "D3: parameters")
    f, c = fig_stats_pca(d)
    save(f, "stats_pca", c, "D4: principal components")
    lines = ["# The data overview's figures", "",
             "Written by `hpc/pipeline/evaluation/ev_dataview.py` (block 23) from `%s`; formats "
             "%s at %d dpi." % (os.path.join(ddir, NPZ), ", ".join(ecfg.formats), ecfg.dpi), ""]
    for fam in ("D1: windows", "D2: statistics", "D3: parameters", "D4: principal components"):
        sel = [(r, c) for f_, r, c in index if f_ == fam]
        if not sel:
            continue
        lines += ["## %s" % fam, ""]
        lines += ["- `figures/%s.{%s}`: %s." % (r, ",".join(ecfg.formats), c) for r, c in sel]
        lines.append("")
    with open(os.path.join(ddir, "FIGURES.md.tmp"), "w") as fh:
        fh.write("\n".join(lines))
    os.replace(os.path.join(ddir, "FIGURES.md.tmp"), os.path.join(ddir, "FIGURES.md"))
    log("EVAL DATA FIGURES: %d figure(s) in %s (%.0f s)" % (len(index), fdir, time.time() - t0))
    return [(r, c) for _f, r, c in index]
