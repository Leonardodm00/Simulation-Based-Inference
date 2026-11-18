"""Set-up figures of the synthetic-runs deck: F03 parameters, F04 the
conditions' traces, F05 the two data sets, F06 R3's gap.

Inputs: `data/traces.npz` (gen_traces.py, the pipeline's bench generator), the
synth v4 metrics folder (`S-A1_A1_sim.npz`, `S-A1_A1_R0.npz`: theta_ref, cls,
donor) and `centres.py` (the condition centres in the cluster's basis).
Usage: python3 figs_setup.py <deck folder>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deckstyle as S                                 # noqa: E402
from centres import SIGNS_CLUSTER, TAU, simplex_centres   # noqa: E402
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.transforms import blended_transform_factory   # noqa: E402

# ---- configuration ----------------------------------------------------------
METRICS = "data/runs/synth/evaluation/metrics"
TRACES = "data/traces.npz"
OUT = "figures"
FS = 100.0                # Hz (SPEC 2.4)
# SBI hpc/joint/stage1/bench_burst_provider.py, BENCH_AXES, with the units of
# its comments
BENCH_AXES = (
    ("burst rate", 0.10, 0.40, "bursts/s"),
    ("irregularity", 0.30, 0.90, ""),
    ("IBI CV", 0.25, 1.00, ""),
    ("n fragments", 1.0, 4.0, "count"),
    ("intraburst rate", 60.0, 140.0, "spikes/s"),
    ("participation mean", 0.45, 0.80, "probability"),
    ("burst duration", 0.15, 0.35, "s"),
    ("fragment duty", 0.35, 1.00, ""),
    ("participation kappa", 3.0, 15.0, ""),
    ("background", 0.01, 0.06, "spikes/s/neuron"),
)
DELTA = 0.105             # R3's range shift = pi * shift_max = 0.3 * 0.35 (latent_gap.py)
YLAB = "x  (counts per 10-ms bin, per unit)"
N_SIM_SHOWN = 0
FOOT = "synth v4 deck | %s | SBI sci/pipeline 54edb32 | 2026-11-18"


def cohort_donors(stem="S-A1_A1_R0.npz"):
    r = np.load(os.path.join(METRICS, stem))
    th, cls, donor = r["theta_ref"], r["cls"], r["donor"]
    ids = np.unique(donor)
    first = np.array([np.flatnonzero(donor == d)[0] for d in ids])
    return th[first], cls[first]


# ---- F04 (traces) ---------------------------------------------------------------------
def fig04_traces(z):
    x = z["cond_x"]
    t = np.arange(x.shape[1]) / FS
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(3, 2, width_ratios=[3.1, 1.0], hspace=0.22, wspace=0.07,
                          left=0.075, right=0.985, top=0.93, bottom=0.135)
    zoom = (60.0, 80.0)
    ymax = float(np.max(x)) * 1.05
    for c in range(3):
        col = S.COND[c]
        ax = fig.add_subplot(gs[c, 0])
        ax.plot(t, x[c], color=col, lw=0.9)
        ax.axvspan(*zoom, color=S.BOX, lw=0, zorder=0)
        ax.set_xlim(0, 180)
        ax.set_ylim(-0.02, ymax)
        ax.grid(axis="x", visible=False)
        ax.set_yticks([0, 0.5, 1.0])
        ax.text(2.0, ymax * 0.98, "condition %d" % (c + 1), va="top", ha="left",
                fontsize=14, fontweight="semibold", color=S.INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        if c == 0:
            ax.set_title("one 180-s window at the condition's centre", loc="left",
                         fontsize=14, fontweight="normal", color=S.INK2)
        if c < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time in the window (s)")
        axz = fig.add_subplot(gs[c, 1])
        m = (t >= zoom[0]) & (t <= zoom[1])
        axz.fill_between(t[m], 0, x[c][m], color=col, alpha=0.15, lw=0)
        axz.plot(t[m], x[c][m], color=col, lw=1.4)
        axz.set_xlim(*zoom)
        axz.set_ylim(-0.02, ymax)
        axz.set_yticks([0, 0.5, 1.0])
        axz.set_yticklabels([])
        axz.grid(axis="x", visible=False)
        axz.set_facecolor(S.BOX)
        if c == 0:
            axz.set_title("zoom, 60-80 s", loc="left", fontsize=14,
                          fontweight="normal", color=S.INK2)
        if c < 2:
            axz.set_xticklabels([])
        else:
            axz.set_xlabel("time (s)")
    fig.text(0.012, 0.52, YLAB, rotation=90, va="center", fontsize=13, color=S.INK2)
    S.footer(fig, FOOT % "simulated here with the pipeline's bench generator: scripts/gen_traces.py")
    S.save(fig, OUT, "F04_condition_traces")


# ---- F03 ---------------------------------------------------------------------
def fig03_parameters(mu):
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(left=0.30, right=0.745, top=0.86, bottom=0.04)
    n = len(BENCH_AXES)
    for k, (name, lo, hi, unit) in enumerate(BENCH_AXES):
        y = n - 1 - k
        free = k >= 8
        ax.plot([0, 1], [y, y], color=S.GRID, lw=9, solid_capstyle="round", zorder=1)
        if free:
            ax.plot([0, 1], [y, y], color=S.OCHRE, lw=9, alpha=0.4,
                    solid_capstyle="round", zorder=2)
            ax.plot([DELTA, 1 + DELTA], [y - 0.3, y - 0.3], color=S.RED, lw=3.0,
                    solid_capstyle="round", zorder=2, clip_on=False)
        else:
            for c in range(3):
                ax.scatter([mu[c, k]], [y], s=150, color=S.COND[c], zorder=4,
                           edgecolor="white", linewidth=2)
        ax.text(-0.03, y, name, ha="right", va="center", fontsize=14, color=S.INK)
        unit_s = (", %s" % unit) if unit else ""
        ax.text(1.17, y, "%g to %g%s" % (lo, hi, unit_s), ha="left", va="center",
                fontsize=12.5, color=S.INK2)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.75, n - 0.35)
    ax.axis("off")
    ax.text(1.17, n - 0.35, "physical range", fontsize=12.5, color=S.INK2,
            fontweight="semibold", va="bottom")
    ax.text(0.0, n - 0.35, "0", fontsize=12, color=S.MUTED, ha="center", va="bottom")
    ax.text(1.0, n - 0.35, "1", fontsize=12, color=S.MUTED, ha="center", va="bottom")
    ax.text(0.5, n - 0.35, "unit-box coordinate", fontsize=12, color=S.MUTED,
            ha="center", va="bottom")
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    ax.plot([0.02, 0.98], [1.5, 1.5], color=S.GRID, lw=1.2, transform=tr, clip_on=False)
    ax.text(0.085, 5.5, "8 label axes\nthe condition\nsets their\ncentre", ha="center",
            va="center", fontsize=13, color=S.INK2, fontweight="semibold", transform=tr,
            linespacing=1.3)
    ax.text(0.085, 0.5, "2 free axes\nuniform in\nevery data set", ha="center",
            va="center", fontsize=13, color=S.INK2, fontweight="semibold", transform=tr,
            linespacing=1.3)
    handles = [Line2D([], [], ls="none", marker="o", ms=11, mfc=S.COND[c], mec="white",
                      mew=2, label="condition %d centre" % (c + 1)) for c in range(3)]
    handles.append(Line2D([], [], color=S.OCHRE, lw=9, alpha=0.4, label="uniform"))
    handles.append(Line2D([], [], color=S.RED, lw=3, label="R3: range moved up by 10.5 %"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.995), ncol=5,
               fontsize=12.5, handlelength=1.6, columnspacing=1.6)
    S.footer(fig, FOOT % "ranges: BENCH_AXES; centres: D-132, scripts/centres.py")
    S.save(fig, OUT, "F03_parameters")


# ---- F04 ---------------------------------------------------------------------
def plane_basis(mu):
    """Orthonormal basis (8, 2) of the plane through the three centres, and its origin."""
    origin = mu.mean(0)
    q, _r = np.linalg.qr((mu[1:] - mu[0]).T)
    e = q[:, :2]
    p = (mu - origin) @ e
    # rotate so that condition 1 points straight up
    a = np.arctan2(p[0, 0], p[0, 1])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return origin, e @ rot.T


def fig04_design(mu):
    global N_SIM_SHOWN
    sim = np.load(os.path.join(METRICS, "S-A1_A1_sim.npz"))["theta_ref"]
    N_SIM_SHOWN = len(np.unique(sim, axis=0))
    th, cls = cohort_donors()
    origin, e = plane_basis(mu)
    ps = (sim[:, :8] - origin) @ e
    pd = (th[:, :8] - origin) @ e
    pc = (mu - origin) @ e
    lim = 0.72
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.12], wspace=0.16, left=0.03,
                          right=0.99, top=0.80, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    ax.scatter(ps[:, 0], ps[:, 1], s=9, color=S.A1, alpha=0.45, lw=0)
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(ps[:, 0], ps[:, 1], s=9, color=S.GRID, alpha=0.9, lw=0, zorder=1)
    th_c = np.linspace(0, 2 * np.pi, 200)
    for c in range(3):
        ax2.fill(pc[c, 0] + TAU * np.cos(th_c), pc[c, 1] + TAU * np.sin(th_c),
                 color=S.COND[c], alpha=0.12, lw=0, zorder=2)
        m = cls == c
        ax2.scatter(pd[m, 0], pd[m, 1], s=95, color=S.COND[c], edgecolor="white",
                    linewidth=1.6, zorder=3, label="condition %d" % (c + 1))
        ax2.scatter([pc[c, 0]], [pc[c, 1]], s=260, facecolor="none", edgecolor=S.INK,
                    linewidth=1.6, zorder=4)
    for a in (ax, ax2):
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_visible(False)
        a.plot([-0.62, -0.42], [-0.66, -0.66], color=S.INK2, lw=2)
        a.text(-0.52, -0.645, "0.2", ha="center", va="bottom", fontsize=11, color=S.INK2)
    ax.text(0.0, 1.16, "Bank (arm S): trains the posterior", transform=ax.transAxes,
            fontsize=15, fontweight="semibold", va="bottom")
    ax.text(0.0, 1.02, "20,000 parameter sets, uniform in the box;\nshown: the {:,} of the "
            "report split".format(N_SIM_SHOWN), transform=ax.transAxes, fontsize=11.5,
            color=S.INK2, va="bottom", linespacing=1.3)
    ax2.text(0.0, 1.16, "Cohort R0 (arm R): scores the posterior", transform=ax2.transAxes,
             fontsize=15, fontweight="semibold", va="bottom")
    ax2.text(0.0, 1.02, "27 donors' true parameters, hidden from training;\nrings: the "
             "condition centres, discs: 1 SD (0.10)", transform=ax2.transAxes,
             fontsize=11.5, color=S.INK2, va="bottom", linespacing=1.3)
    ax2.legend(loc="lower right", fontsize=12, handletextpad=0.1, borderaxespad=0.1,
               markerscale=0.8)
    fig.text(0.03, 0.035, "Both panels: the 8 label axes seen in the plane through the "
             "three condition centres (unit-box units; scale bar 0.2).", fontsize=11.5,
             color=S.INK2)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.axis("off")
    levels = [("3 conditions", "a centre on the 8 label axes", 86),
              ("9 donors per condition", "true parameters = centre + noise (SD 0.10)", 64),
              ("2 wells per donor", "same parameters, own randomness", 42),
              ("6 windows per well", "180 s each", 20)]
    for i, (big, small, y) in enumerate(levels):
        S.card(ax3, 4, y - 8.5, 92, 16, fill=S.WASH[S.A2], r=2.5)
        ax3.text(9, y + 2.0, big, fontsize=15, fontweight="semibold", va="center")
        ax3.text(9, y - 3.8, small, fontsize=12.5, color=S.INK2, va="center")
        if i < 3:
            S.arrow(ax3, (50, y - 9.0), (50, y - 13.0), color=S.A2, lw=2, ms=14)
    ax3.text(50, 4.5, "27 donors, 54 wells, 324 windows", ha="center", fontsize=14.5,
             fontweight="semibold")
    ax3.text(50, -1.5, "per cohort; R0 and R3 are built alike, with other seeds",
             ha="center", fontsize=11.5, color=S.INK2)
    ax3.text(4, 101, "How a cohort is built", fontsize=15, fontweight="semibold",
             va="bottom")
    S.footer(fig, FOOT % "theta_ref of metrics/S-A1_A1_sim.npz and S-A1_A1_R0.npz")
    S.save(fig, OUT, "F05_data_sets")


# ---- F05 ---------------------------------------------------------------------
def fig05_gap(z):
    clean, gap, cont = z["well_clean"], z["well_gap"], z["contaminated"]
    J, W = clean.shape
    t = np.arange(J * W) / FS
    jc = int(np.flatnonzero(cont)[0])
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.62, wspace=0.3,
                          left=0.08, right=0.985, top=0.93, bottom=0.135)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, gap.ravel(), color=S.RED, lw=0.6, label="R3 (with the gap)")
    ax.plot(t, clean.ravel(), color=S.INK2, lw=0.6, alpha=0.85, label="same well, no gap")
    for j in range(1, J):
        ax.axvline(j * W / FS, color=S.MUTED, lw=0.8, ls=(0, (2, 3)))
    ax.annotate("(d) contamination: this window x2-5, plus noise",
                xy=((jc + 0.62) * W / FS, float(np.max(gap[jc])) * 0.85),
                xytext=((jc + 1.25) * W / FS, float(np.max(gap[jc])) * 0.82),
                fontsize=13, color=S.INK, va="center",
                arrowprops=dict(arrowstyle="-|>", color=S.INK2, lw=1.4))
    ax.set_xlim(0, J * W / FS)
    ax.set_xlabel("time in one well (s): its six 180-s windows")
    ax.set_ylabel("x")
    ax.legend(loc="upper right", ncol=2, fontsize=12.5, handlelength=1.4)
    ax.grid(axis="x", visible=False)
    ax.set_title("One well of a condition-1 donor, the same spikes with and without the gap",
                 loc="left", fontsize=14.5)
    # (c) drift
    ax2 = fig.add_subplot(gs[1, 0])
    k = int(10 * FS)
    d = (gap - clean).reshape(J, W // k, k).mean(2)
    d[cont] = np.nan
    tt = (np.arange(J)[:, None] * (W / FS) + (np.arange(W // k)[None, :] + 0.5) * 10.0)
    ax2.axhline(0, color=S.MUTED, lw=1)
    for j in range(J):
        ax2.plot(tt[j], d[j], color=S.RED, lw=2, marker="o", ms=3.5)
    ax2.axvspan(jc * W / FS, (jc + 1) * W / FS, color=S.BOX, lw=0)
    ax2.text((jc + 0.5) * W / FS, 0.0, "window\n(d)\nleft\nout", ha="center",
             va="center", fontsize=10.5, color=S.INK2, linespacing=1.15)
    ax2.set_ylim(-0.01, 0.01)
    ax2.set_xlim(0, J * W / FS)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("R3 - clean")
    ax2.set_title("(c) drift: 10-s means, period 120 s", loc="left", fontsize=13.5)
    ax2.grid(axis="x", visible=False)
    # (d) zoom
    ax3 = fig.add_subplot(gs[1, 1])
    tw = np.arange(W) / FS
    m = (tw >= 60) & (tw <= 90)
    ax3.plot(tw[m], gap[jc][m], color=S.RED, lw=0.9)
    ax3.plot(tw[m], clean[jc][m], color=S.INK2, lw=1.1)
    ax3.set_xlabel("time in window (d) (s)")
    ax3.set_ylabel("x")
    ax3.set_title("(d) contamination, 30-s zoom", loc="left", fontsize=13.5)
    ax3.grid(axis="x", visible=False)
    # (a) range shift
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_xlim(-0.02, 1.3)
    ax4.set_ylim(-1.05, 1.95)
    ax4.axis("off")
    for i, (name, lo, hi, unit) in enumerate([BENCH_AXES[8], BENCH_AXES[9]]):
        y = 1.15 - i * 1.55
        w = hi - lo
        ax4.text(0, y + 0.42, name, fontsize=12.5, fontweight="semibold", va="center")
        ax4.plot([0, 1], [y, y], color=S.OCHRE, lw=8, alpha=0.45, solid_capstyle="round")
        ax4.plot([DELTA, 1 + DELTA], [y - 0.28, y - 0.28], color=S.RED, lw=4,
                 solid_capstyle="round")
        ax4.text(0, y - 0.68, "%g-%g  becomes  %.4g-%.4g" % (lo, hi, lo + DELTA * w,
                                                         hi + DELTA * w),
                 fontsize=11.5, color=S.INK2, va="center")
    ax4.set_title("(a) range shift, the 2 free axes", loc="left", fontsize=13.5)
    S.footer(fig, FOOT % "pi = 0.3 through the pipeline's latent_gap.py; scripts/gen_traces.py")
    S.save(fig, OUT, "F06_gap")


def main(deck):
    os.chdir(deck)
    S.setup()
    z = np.load(TRACES)
    mu = simplex_centres(SIGNS_CLUSTER)
    assert np.allclose(z["theta_cond"][:, :8], mu), "traces.npz was made with other centres"
    fig04_traces(z)
    fig03_parameters(mu)
    fig04_design(mu)
    fig05_gap(z)


if __name__ == "__main__":
    main(sys.argv[1])
