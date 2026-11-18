"""Result figures of the synthetic-runs deck, from the synth v4 evaluation files:
F11 searches, F12 members and mixture, F13 what is learned per axis,
F14 calibration and the box edges, F15 R0 against R3 window by window,
F16 the fifteen paired comparisons, F17 the trade under R3's tie.

Inputs: the metrics folder of synth_v4_results.tar.gz (search_<q>.json,
<stem>_<D>.json / .npz, paired_<name>_<D>.json). Every number drawn is read
from those files or computed from them here; the printed lines are the
checks against the verified tables of the results document.
Usage: python3 figs_results.py <deck folder>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deckstyle as S                                  # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402

# ---- configuration ----------------------------------------------------------
METRICS = "data/runs/synth/evaluation/metrics"
OUT = "figures"
SEARCHES = ["S-A1", "S-A2-R0", "S-A5-R0", "S-A2-R3", "S-A5-R3"]
STEM = {"S-A1": "S-A1_A1", "S-A2-R0": "S-A2-R0_A2", "S-A5-R0": "S-A5-R0_A5",
        "S-A2-R3": "S-A2-R3_A2", "S-A5-R3": "S-A5-R3_A5"}
ARM = {"S-A1": "A1", "S-A2-R0": "A2", "S-A5-R0": "A5", "S-A2-R3": "A2", "S-A5-R3": "A5"}
LS = {"S-A1": "-", "S-A2-R0": "-", "S-A5-R0": "-", "S-A2-R3": (0, (5, 2.5)),
      "S-A5-R3": (0, (5, 2.5))}
AXES = ["burst rate", "irregularity", "IBI CV", "n fragments", "intraburst rate",
        "participation mean", "burst duration", "fragment duty", "participation kappa",
        "background"]
EDGE_EPS = 1.0 / (1.0 + np.exp(5.0))     # sigmoid(-5): the flow's spline bound in logit space
SAT = 0.2499                              # L-C2ST statistic at its maximum 0.25
N_BOOT, SEED = 10000, 4242
FOOT = "synth v4 deck | %s | evaluation of 2026-11-16 | 2026-11-18"


def jload(name):
    with open(os.path.join(METRICS, name + ".json")) as fh:
        return json.load(fh)


def nload(name):
    return np.load(os.path.join(METRICS, name + ".npz"))


def col(q):
    return S.ARM[ARM[q]]


def repel(ys, gap):
    """Spread label positions so that neighbours are at least `gap` apart."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for _ in range(200):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            if out[b] - out[a] < gap:
                mid = 0.5 * (out[a] + out[b])
                out[a], out[b] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    return out


def donor_boot(d, donors, rng):
    ids = np.unique(donors)
    sums = np.array([d[donors == i].sum() for i in ids])
    cnts = np.array([(donors == i).sum() for i in ids])
    idx = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    est = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(d.mean()), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


# ---- F10 ---------------------------------------------------------------------
def fig10_searches():
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(left=0.075, right=0.80, top=0.90, bottom=0.12)
    bands = [(0.5, 20.5, "20 random starts"), (20.5, 30.5, "GP round 1"),
             (30.5, 40.5, "round 2"), (40.5, 50.5, "round 3"), (50.5, 60.5, "round 4")]
    for i, (a, b, lab) in enumerate(bands):
        if i % 2 == 0:
            ax.axvspan(a, b, color=S.BOX, lw=0, zorder=0)
        ax.text(0.5 * (a + b), 4.25, lab, ha="center", va="bottom", fontsize=12.5,
                color=S.INK2)
    ax.axhline(0, color=S.INK2, lw=1.2, zorder=1)
    ax.text(60.2, 0.1, "uniform prior: L = 0", fontsize=12, color=S.INK2, va="bottom",
            ha="right")
    ends, labels = [], []
    for q in SEARCHES:
        j = jload("search_" + q)
        t = j["trials"]
        nll = np.array([x["nll"] for x in t], float)
        rmin = np.array([list(x["running_min_nll"].values())[0] for x in t], float)
        k = np.arange(1, len(t) + 1)
        fin = [i for i, x in enumerate(t) if x["trial_id"] == j["finalist"]][0]
        ax.scatter(k, nll, s=16, color=col(q), alpha=0.28, lw=0, zorder=2)
        ax.step(k, rmin, where="post", color=col(q), lw=2.4, ls=LS[q], zorder=3)
        ax.scatter([k[fin]], [nll[fin]], s=90, facecolor="white", edgecolor=col(q),
                   linewidth=2.2, zorder=4)
        ends.append(rmin[-1])
        labels.append("%s  %.2f" % (q, rmin[-1]))
        print("F10 %s: finalist trial %d, L %.3f, median round 0 %+.2f" %
              (q, k[fin], nll[fin], np.median(nll[:20])))
    ly = repel(ends, 0.34)
    for q, y0, y1, lab in zip(SEARCHES, ends, ly, labels):
        ax.plot([60.6, 61.4, 63.4], [y0, y1, y1], color=col(q), lw=2.2, ls=LS[q],
                clip_on=False)
        ax.text(63.9, y1, lab, va="center", fontsize=13, color=S.INK, clip_on=False)
    ax.text(61.0, max(ly) + 0.6, "best L found", fontsize=12.5, color=S.INK2,
            fontweight="semibold", va="center")
    ax.set_xlim(0.5, 60.5)
    ax.set_ylim(-4.7, 4.2)
    ax.set_xlabel("trial (in the order run)")
    ax.set_ylabel("L: NLL at the truth, simulated\nreport split (nats per window)")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], color=S.INK2, lw=2.4, label="best so far"),
               Line2D([], [], ls="none", marker="o", ms=5, color=S.INK2, alpha=0.4,
                      label="one trial"),
               Line2D([], [], ls="none", marker="o", ms=9, mfc="white", mec=S.INK2, mew=2,
                      label="the finalist"),
               Line2D([], [], color=S.INK2, lw=2.4, ls=(0, (5, 2.5)),
                      label="dashed: the term reads R3")]
    ax.legend(handles=handles, loc="lower left", ncol=4, fontsize=12.5,
              bbox_to_anchor=(0.0, 0.0))
    S.footer(fig, FOOT % "metrics/search_<q>.json; scripts/figs_results.py")
    S.save(fig, OUT, "F11_searches")


# ---- F11 ---------------------------------------------------------------------
def fig11_members():
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.42, left=0.075,
                          right=0.975, top=0.88, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    j = jload("search_S-A1")
    mem = list(j["ensembles"].values())[0]["members"]
    for m in mem:
        e = np.arange(len(m["npe"]))
        ax.plot(e, m["npe"], color=S.A1, lw=1.1, alpha=0.45)
        ax.plot(e, m["val_npe"], color=S.A1, lw=2.4)
        b = int(np.argmin(m["val_npe"]))
        ax.scatter([b], [m["val_npe"][b]], s=55, color="white", edgecolor=S.A1,
                   linewidth=2, zorder=4)
    ax.set_xlabel("epoch (one pass over the training split)")
    ax.set_ylabel("nats per window")
    ax.set_ylim(-11.5, 2)
    ax.set_xlim(-0.5, 36)
    ax.text(14.5, -9.3, "training loss\n(thin)", fontsize=13, color=S.INK, ha="left",
            va="top")
    ax.text(9.0, -1.7, "held-out (selection-split) NLL (thick)", fontsize=13,
            color=S.INK, ha="left")
    ax.annotate("", xy=(30, -8.6), xytext=(30, -4.6),
                arrowprops=dict(arrowstyle="<->", color=S.INK2, lw=1.4))
    ax.text(30.7, -6.6, "overfit", fontsize=12.5, color=S.INK2, va="center")
    ax.set_title("S-A1's five members; circle: the epoch kept\n(best held-out NLL)",
                 loc="left", fontsize=14.5)
    ax.grid(axis="x", visible=False)
    # right: members against the mixture
    ax2 = fig.add_subplot(gs[1])
    rows = SEARCHES[::-1]
    for r, q in enumerate(rows):
        s = jload(STEM[q] + "_sim")
        mm = np.array(s["nll"]["members"], float)
        mix = float(s["nll"]["mean"])
        ax2.plot([mm.mean(), mix], [r, r], color=S.GRID, lw=6, solid_capstyle="round",
                 zorder=1)
        ax2.scatter(mm, np.full(5, r), s=60, color=col(q), alpha=0.45, lw=0, zorder=2)
        ax2.plot([mm.mean()] * 2, [r - 0.22, r + 0.22], color=S.INK2, lw=2, zorder=3)
        ax2.scatter([mix], [r], s=170, marker="D", color=col(q), edgecolor=S.INK,
                    linewidth=1.2, zorder=4)
        ax2.text(mix - 0.08, r + 0.3, "gain %.2f" % (mm.mean() - mix), fontsize=12,
                 color=S.INK2, ha="right", va="center")
        print("F11 %s: members' mean %.3f, mixture %.3f, gain %.3f" %
              (q, mm.mean(), mix, mm.mean() - mix))
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels(["%s : %s" % (q, ARM[q]) for q in rows], fontsize=13.5)
    ax2.set_ylim(-0.6, len(rows) - 0.3)
    ax2.set_xlim(-7.15, -4.4)
    ax2.set_xlabel("NLL at the truth, simulated test windows (nats)")
    ax2.grid(axis="y", visible=False)
    handles = [Line2D([], [], ls="none", marker="o", ms=8, color=S.INK2, alpha=0.45,
                      label="5 members"),
               Line2D([], [], ls="none", marker="|", ms=14, mew=2, color=S.INK2,
                      label="their mean"),
               Line2D([], [], ls="none", marker="D", ms=10, mfc=S.INK2, mec=S.INK,
                      label="ensemble (mixture)")]
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=3,
               fontsize=12.5, columnspacing=1.2, handletextpad=0.3)
    ax2.text(-7.1, -0.55, "\u2190 better", fontsize=12.5, color=S.INK2, va="bottom")
    S.footer(fig, FOOT % "search_<q>.json (npe, val_npe); <stem>_sim.json (nll)")
    S.save(fig, OUT, "F12_members_mixture")


# ---- F12 ---------------------------------------------------------------------
def fig12_contraction():
    M = np.array([jload(STEM[q] + "_sim")["recovery"]["contraction"] for q in SEARCHES])
    order = [0, 1, 2, 3, 6, 7, 4, 5, 8, 9]       # learned well | weakly | free
    Mo = M[:, order]
    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#bcd6f3", S.A1, "#163f73"])
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.84, bottom=0.21)
    ax.imshow(Mo, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(Mo.shape[0]):
        for k in range(Mo.shape[1]):
            ax.text(k, i, "%.2f" % Mo[i, k], ha="center", va="center", fontsize=13.5,
                    color="white" if Mo[i, k] > 0.55 else S.INK)
    ax.set_yticks(range(len(SEARCHES)))
    ax.set_yticklabels(["%s : %s" % (q, ARM[q]) for q in SEARCHES], fontsize=13.5)
    ax.set_xticks(range(10))
    ax.set_xticklabels([AXES[k] for k in order], rotation=32, ha="right", fontsize=13)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for x in (5.5, 7.5):
        ax.axvline(x, color="white", lw=8)
    groups = [(-0.5, 5.5, "learned well: %.2f to %.2f" % (Mo[:, :6].min(), Mo[:, :6].max())),
              (5.5, 7.5, "weakly: %.2f to %.2f" % (Mo[:, 6:8].min(), Mo[:, 6:8].max())),
              (7.5, 9.5, "free axes: %.2f to %.2f" % (Mo[:, 8:].min(), Mo[:, 8:].max()))]
    for a, b, lab in groups:
        ax.plot([a + 0.08, b - 0.08], [-0.62, -0.62], color=S.INK2, lw=1.5, clip_on=False)
        ax.text(0.5 * (a + b), -0.72, lab, ha="center", va="bottom", fontsize=13.5,
                color=S.INK, fontweight="semibold", clip_on=False)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(4.5, -0.5)
    S.footer(fig, FOOT % "<stem>_sim.json, recovery.contraction = 1 - (mean posterior variance) / (1/12)")
    S.save(fig, OUT, "F13_contraction")
    return M


# ---- F13 ---------------------------------------------------------------------
LAB = {}


def fig13_calibration():
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28, left=0.07,
                          right=0.985, top=0.88, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    b = np.linspace(0, 1, 201)
    n_int = None
    for q in SEARCHES:
        z = nload(STEM[q] + "_sim")
        g = z["gamma"][z["calib"]]
        th = z["theta_ref"][z["calib"]]
        edge = (np.minimum(th, 1 - th) < EDGE_EPS).any(1)
        c_all = np.array([(g <= x).mean() for x in b])
        c_int = np.array([(g[~edge] <= x).mean() for x in b])
        n_int = int((~edge).sum())
        main = q == "S-A1"
        if main:
            LAB["int"] = float(np.interp(0.42, b, c_int))
            LAB["all"] = float(np.interp(0.95, b, c_all))
            ax.plot(b, c_all, color=S.A1, lw=2.4, ls=(0, (4, 2)), zorder=4)
            ax.plot(b, c_int, color=S.A1, lw=2.8, zorder=5)
        else:
            ax.plot(b, c_int, color=S.MUTED, lw=1.2, alpha=0.8, zorder=3)
        print("F13 %s: calibration rows %d, edge %d (%.3f), C(0.95) all %.3f interior %.3f" %
              (q, len(g), edge.sum(), edge.mean(), (g <= 0.95).mean(), (g[~edge] <= 0.95).mean()))
    se = np.sqrt(b * (1 - b) / n_int)
    ax.fill_between(b, b - 1.96 * se, b + 1.96 * se, color=S.GRID, lw=0, zorder=1)
    ax.plot([0, 1], [0, 1], color=S.INK2, lw=1.2, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("credibility level of the region")
    ax.set_ylabel("share of truths inside it")
    ax.annotate("interior rows (solid)", xy=(0.42, LAB["int"]), xytext=(0.04, 0.80),
                fontsize=13, color=S.INK, arrowprops=dict(arrowstyle="-", color=S.INK2,
                                                         lw=1.2))
    ax.annotate("all rows (dashed):\n0.85 at the 0.95 region", xy=(0.95, LAB["all"]),
                xytext=(0.50, 0.28), fontsize=13, color=S.INK,
                arrowprops=dict(arrowstyle="-", color=S.INK2, lw=1.2))
    ax.text(0.98, 0.04, "grey: the other four\nensembles, interior rows", fontsize=11.5,
            color=S.INK2, ha="right", va="bottom")
    ax.set_title("Coverage, S-A1 : A1 (1,000 test windows)", loc="left", fontsize=14.5)
    # right: NLL against the distance to the nearest edge
    ax2 = fig.add_subplot(gs[1])
    z = nload("S-A1_A1_sim")
    th, nl = z["theta_ref"], z["nll_rows"]
    dist = np.minimum(th, 1 - th).min(1)
    dist = np.maximum(dist, 1e-5)
    edge = dist < EDGE_EPS
    ax2.axvspan(1e-5, EDGE_EPS, color=S.WASH[S.RED], lw=0, zorder=0)
    ax2.scatter(dist[~edge], nl[~edge], s=10, color=S.A1, alpha=0.35, lw=0, zorder=2)
    ax2.scatter(dist[edge], nl[edge], s=14, color=S.RED, alpha=0.6, lw=0, zorder=2)
    ax2.axhline(0, color=S.INK2, lw=1.2)
    ax2.axvline(EDGE_EPS, color=S.INK2, lw=1.2, ls=(0, (3, 2)))
    ax2.set_xscale("log")
    ax2.set_xlim(1e-5, 0.5)
    ax2.set_ylim(-14, 32)
    ax2.set_xlabel("distance of the truth's nearest coordinate to a box edge")
    ax2.set_ylabel("NLL at the truth (nats)")
    ax2.text(EDGE_EPS * 1.15, 29, "0.0067: where the flow's\nsupport stops", fontsize=12.5,
             color=S.INK, va="top")
    ax2.text(1.3e-5, 29, "edge rows:\n%.1f %% of windows,\nworse than the prior" %
             (100 * edge.mean()), fontsize=12.5, color=S.INK, va="top")
    ax2.text(0.45, 5.5, "interior rows: mean %.2f" % nl[~edge].mean(), fontsize=12.5,
             color=S.INK, ha="right", va="bottom")
    ax2.text(0.48, 0.6, "uniform prior = 0", fontsize=11.5, color=S.INK2, ha="right",
             va="bottom")
    ax2.set_title("Where the deficit sits: truths at the box edges (S-A1 : A1)", loc="left",
                  fontsize=14.5)
    ax2.grid(axis="x", visible=False)
    print("F13 S-A1 rows %d, edge share %.3f, NLL edge %.2f interior %.2f" %
          (len(nl), edge.mean(), nl[edge].mean(), nl[~edge].mean()))
    S.footer(fig, FOOT % "<stem>_sim.npz (gamma, calib, theta_ref, nll_rows)")
    S.save(fig, OUT, "F14_calibration_edges")


# ---- F14 ---------------------------------------------------------------------
def r3_masks():
    zs = [nload(STEM[q] + "_R3") for q in ("S-A1", "S-A2-R3", "S-A5-R3")]
    flag = np.logical_and.reduce([z["lc2st_stat"] >= SAT for z in zs])
    th = zs[0]["theta_ref"]
    edge = (np.minimum(th, 1 - th) < EDGE_EPS).any(1)
    return flag, edge, zs[0]["donor"]


def fig14_windows():
    flag, edge, donor = r3_masks()
    rest = ~flag & ~edge
    cols = [("R0", "S-A1"), ("R0", "S-A2-R0"), ("R0", "S-A5-R0"),
            ("R3", "S-A1"), ("R3", "S-A2-R3"), ("R3", "S-A5-R3")]
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.18)
    ax.axvspan(2.5, 5.6, color=S.BOX, lw=0, zorder=0)
    ax.axhline(0, color=S.INK2, lw=1.2, zorder=1)
    for i, (D, q) in enumerate(cols):
        nl = nload(STEM[q] + "_" + D)["nll_rows"]
        jit = i + rng.uniform(-0.3, 0.3, size=len(nl))
        if D == "R0":
            ax.scatter(jit, nl, s=11, color=col(q), alpha=0.55, lw=0, zorder=2)
            txt = "mean %.2f" % nl.mean()
            print("F14 R0 %s: mean %.3f, max %.2f" % (q, nl.mean(), nl.max()))
        else:
            ax.scatter(jit[rest], nl[rest], s=11, color=col(q), alpha=0.55, lw=0, zorder=2)
            ax.scatter(jit[edge & ~flag], nl[edge & ~flag], s=30, marker="s",
                       facecolor="none", edgecolor=S.INK, lw=1.1, zorder=3)
            ax.scatter(jit[flag], nl[flag], s=34, marker="D", color=S.INK,
                       edgecolor="white", lw=0.8, zorder=4)
            txt = "mean %.2f\nothers %.2f" % (nl.mean(), nl[rest].mean())
            print("F14 R3 %s: mean %.3f, flagged %d mean %.2f, donor0 others %d mean %.2f, rest %d mean %.3f" %
                  (q, nl.mean(), flag.sum(), nl[flag].mean(), (edge & ~flag).sum(),
                   nl[edge & ~flag].mean(), rest.sum(), nl[rest].mean()))
        ax.plot([i - 0.36, i + 0.36], [nl.mean()] * 2, color=S.INK, lw=2.4, zorder=5)
        ax.text(i, 53.0, txt, ha="center", va="top", fontsize=12, color=S.INK,
                linespacing=1.2)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["%s\n%s" % (ARM[q], q) for _D, q in cols], fontsize=12.5)
    from matplotlib.transforms import blended_transform_factory
    trb = blended_transform_factory(ax.transData, fig.transFigure)
    ax.text(1.0, 0.035, "R0: no gap", ha="center", fontsize=14.5, fontweight="semibold",
            transform=trb)
    ax.text(4.0, 0.035, "R3: with the gap", ha="center", fontsize=14.5,
            fontweight="semibold", transform=trb)
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(-14, 54)
    ax.set_ylabel("NLL at the truth, one dot per window (nats)")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], ls="none", marker="D", ms=8, color=S.INK, mec="white",
                      label="18 flagged windows (classifier certain: L-C2ST = 0.25)"),
               Line2D([], [], ls="none", marker="s", ms=8, mfc="none", mec=S.INK,
                      label="donor 0's other windows (a truth at a box edge)"),
               Line2D([], [], color=S.INK, lw=2.4, label="mean of the 324")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3,
              fontsize=12.5, columnspacing=1.5)
    ax.text(-0.55, 0.6, "uniform prior = 0", fontsize=11.5, color=S.INK2, ha="left",
            va="bottom")
    S.footer(fig, FOOT % "<stem>_R0.npz, <stem>_R3.npz (nll_rows, lc2st_stat, theta_ref)")
    S.save(fig, OUT, "F15_r0_r3_windows")
    return flag, edge, donor


# ---- F15 ---------------------------------------------------------------------
def fig15_forest():
    rows = [("S-A2-R0", "A2 (S-A2-R0) \u2212 A1", ["sim", "R0"]),
            ("S-A5-R0", "A5 (S-A5-R0) \u2212 A1", ["sim", "R0"]),
            ("S-A2-R3", "A2 (S-A2-R3) \u2212 A1", ["sim", "R3", "R3dag"]),
            ("S-A5-R3", "A5 (S-A5-R3) \u2212 A1", ["sim", "R3", "R3dag"]),
            ("S-A2-R0_vs_S-A5-R0", "A2 \u2212 A5 (R0 searches)", ["sim", "R0"]),
            ("S-A2-R3_vs_S-A5-R3", "A2 \u2212 A5 (R3 searches)", ["sim", "R3", "R3dag"])]
    dname = {"sim": "simulated", "R0": "R0", "R3": "R3", "R3dag": "R3, shifted truth"}
    marker = {"sim": "o", "R0": "s", "R3": "s", "R3dag": "D"}
    fig, ax = plt.subplots(figsize=(13.33, 7.2))
    fig.subplots_adjust(left=0.30, right=0.985, top=0.86, bottom=0.115)
    y = 0
    yt, yl = [], []
    n_res = 0
    for name, lab, ds in rows:
        c = S.ARM["A2"] if name.startswith("S-A2-R") and "vs" not in name else \
            S.ARM["A5"] if name.startswith("S-A5") else S.INK
        for k, D in enumerate(ds):
            p = jload("paired_%s_%s" % (name, D))
            tw = p["two_sigma_seed"]
            ax.plot([-tw, tw], [y, y], color=S.GRID, lw=9, solid_capstyle="butt", zorder=1)
            ax.plot(p["ci"], [y, y], color=S.INK2, lw=2, zorder=2)
            ax.scatter([p["Dbar"]], [y], s=80, marker=marker[D], color=c, edgecolor=S.INK,
                       linewidth=0.8, zorder=3)
            yt.append(y)
            yl.append(("%s   " % lab if k == 0 else "") + dname[D])
            n_res += int(bool(p["resolved"]))
            y -= 1
        y -= 0.6
    ax.axvline(0, color=S.INK2, lw=1.2, zorder=0)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=12.5)
    ax.set_ylim(y + 0.3, 0.8)
    ax.set_xlim(-1.45, 1.45)
    ax.set_xlabel("paired difference in NLL at the truth, D\u0304 (nats per window)")
    ax.grid(axis="y", visible=False)
    ax.text(-1.42, 0.95, "\u2190 below 0: the first-named arm puts more density on the truth",
            fontsize=12, color=S.INK2, va="bottom")
    ax.text(1.42, 0.95, "resolved: %d of %d" % (n_res, len(yt)), fontsize=13,
            fontweight="semibold", color=S.INK, va="bottom", ha="right")
    handles = [Line2D([], [], color=S.GRID, lw=9, label="\u00b12\u03c3 seed: training noise"),
               Line2D([], [], color=S.INK2, lw=2, label="95 % donor interval"),
               Line2D([], [], ls="none", marker="o", ms=8, color=S.INK2,
                      label="D\u0304, simulated"),
               Line2D([], [], ls="none", marker="s", ms=8, color=S.INK2, label="cohort"),
               Line2D([], [], ls="none", marker="D", ms=8, color=S.INK2,
                      label="R3, shifted truth")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=5,
               fontsize=12, columnspacing=1.2, handletextpad=0.4)
    print("F15: %d pairs, %d resolved" % (len(yt), n_res))
    S.footer(fig, FOOT % "paired_<name>_<D>.json (Dbar, ci, two_sigma_seed)")
    S.save(fig, OUT, "F16_forest")


# ---- F16 ---------------------------------------------------------------------
def fig16_trade(flag, donor):
    rng = np.random.default_rng(SEED)
    nl = {q: nload(STEM[q] + "_R3")["nll_rows"] for q in ("S-A1", "S-A2-R3", "S-A5-R3")}
    pairs = [("S-A5-R3", "A5 \u2212 A1  (S-A5-R3 against S-A1)"),
             ("S-A2-R3", "A2 \u2212 A1  (S-A2-R3 against S-A1)")]
    n, nf = len(flag), int(flag.sum())
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.17)
    x0 = 0
    for q, lab in pairs:
        d = nl[q] - nl["S-A1"]
        mu_u, lo_u, hi_u = donor_boot(d[~flag], donor[~flag], rng)
        mu_f, lo_f, hi_f = donor_boot(d[flag], donor[flag], rng)
        mu_a, lo_a, hi_a = donor_boot(d, donor, rng)
        cu, cf = (n - nf) / n * mu_u, nf / n * mu_f
        print("F16 %s: unflagged %+.3f [%+.3f, %+.3f], flagged %+.2f [%+.2f, %+.2f], all %+.3f [%+.3f, %+.3f]; contributions %+.3f + %+.3f = %+.3f" %
              (lab, mu_u, lo_u, hi_u, mu_f, lo_f, hi_f, mu_a, lo_a, hi_a, cu, cf, cu + cf))
        c = S.ARM[q.split("-")[1]]
        bars = [(x0, cu, c, 1.0, "%d ordinary windows\n%+.2f per window" % (n - nf, mu_u)),
                (x0 + 1, cf, c, 0.55, "%d flagged windows\n%+.2f per window" % (nf, mu_f)),
                (x0 + 2, cu + cf, S.INK2, 1.0, "all %d: the tie\nD\u0304 = %+.3f" % (n, mu_a))]
        for x, h, cc, al, txt in bars:
            ax.bar([x], [h], width=0.62, color=cc, alpha=al, edgecolor="white", lw=0)
            ax.text(x, -0.52, txt, ha="center", va="top", fontsize=12, color=S.INK,
                    linespacing=1.25)
            ax.text(x, h + (0.02 if h >= 0 else -0.02), "%+.2f" % h, ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=13, fontweight="semibold")
        ax.plot([x0 + 0.31, x0 + 0.69], [cu, cu], color=S.INK2, lw=1, ls=(0, (2, 2)))
        ax.plot([x0 + 1.31, x0 + 1.69], [cu + cf, cu + cf], color=S.INK2, lw=1,
                ls=(0, (2, 2)))
        ax.text(x0 + 1, 0.47, lab, ha="center", fontsize=15, fontweight="semibold")
        x0 += 3.6
    ax.axhline(0, color=S.INK2, lw=1.2)
    ax.set_xlim(-0.6, x0 - 0.9)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("contribution to the mean NLL\ndifference on R3 (nats per window)")
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.text(-0.55, -0.47, "below 0: the term arm is better", fontsize=12, color=S.INK2,
            va="bottom")
    S.footer(fig, FOOT % "<stem>_R3.npz (nll_rows, lc2st_stat); donor-cluster bootstrap")
    S.save(fig, OUT, "F17_r3_trade")


def main(deck):
    os.chdir(deck)
    S.setup()
    fig10_searches()
    fig11_members()
    M = fig12_contraction()
    print("F12 contraction:\n", np.round(M, 2))
    fig13_calibration()
    flag, _edge, donor = fig14_windows()
    fig15_forest()
    fig16_trade(flag, donor)


if __name__ == "__main__":
    main(sys.argv[1])
