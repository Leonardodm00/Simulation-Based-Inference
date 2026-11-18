"""Explanatory schematics of the synthetic-runs deck: F01 the idea, F02 the
simulator, F07 the joint model, F08 the three arms, F09 the procedure,
F10 how a posterior is scored, F18 the take-home.

Schematics, not data plots: every number written on them is a configured value
of the run (synth.json, SPEC 2.4-2.5) or a result taken from the verified
results document, named in the deck design document next to the slide.
The raster of F02 is a fresh simulation (`gen_raster.py`).
Usage: python3 figs_schematics.py <deck folder>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deckstyle as S                                  # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import Ellipse, Polygon, Circle, FancyBboxPatch  # noqa: E402

OUT = "figures"
FOOT = "synth v4 deck | %s | 2026-11-18"
H = 100 * S.FULL[1] / S.FULL[0]         # canvas height in canvas units (49.5)


# ---- helpers ------------------------------------------------------------------
def inset(fig, x, y, w, h):
    """An axes at canvas coordinates (x, y, w, h), x in [0, 100], y in [0, H]."""
    ax = fig.add_axes([x / 100.0, y / H, w / 100.0, h / H])
    ax.axis("off")
    return ax


def pill(ax, x, y, w, h, text, fill, ink=S.INK, fs=13, weight="semibold", r=1.2, lw=0,
         edge=None, z=2):
    S.card(ax, x, y, w, h, fill=fill, edge=edge, lw=lw, r=r, z=z)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=ink, zorder=z + 1, linespacing=1.25)


def mini_trace(ax, color, seed=0, n=400, lw=1.4, bursts=6):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    x = 0.04 * rng.random(n)
    for c in rng.uniform(0.05, 0.95, bursts):
        x += rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((t - c) / 0.006) ** 2)
    ax.plot(t, x, color=color, lw=lw)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)


def mini_cloud(ax, color, n=140, seed=1, clusters=None):
    rng = np.random.default_rng(seed)
    if clusters is None:
        p = rng.uniform(0, 1, (n, 2))
        ax.scatter(p[:, 0], p[:, 1], s=6, color=color, alpha=0.6, lw=0)
    else:
        for c, (cx, cy) in zip(S.COND, clusters):
            p = rng.normal([cx, cy], 0.07, (9, 2))
            ax.scatter(p[:, 0], p[:, 1], s=26, color=c, edgecolor="white", lw=0.8)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=S.GRID, lw=1.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")


# ---- F01 ------------------------------------------------------------------------
def fig01_idea():
    fig, ax = S.canvas()
    # arm S card
    S.card(ax, 1.5, 26.5, 44, 21.5, fill=S.WASH[S.A1], r=1.5)
    ax.text(3.5, 45.6, "Simulated world  (arm S)", fontsize=16, fontweight="semibold",
            va="center", color=S.INK)
    mini_cloud(inset(fig, 3.5, 30.5, 9.5, 12), S.A1)
    ax.text(8.25, 28.6, "parameters θ", ha="center", fontsize=12, color=S.INK2)
    S.arrow(ax, (14.0, 36.5), (18.0, 36.5), color=S.INK2, lw=2, ms=14)
    pill(ax, 18.3, 33.2, 9.5, 6.6, "simulator", S.SURFACE, fs=13, edge=S.GRID, lw=1.2)
    S.arrow(ax, (28.1, 36.5), (32.1, 36.5), color=S.INK2, lw=2, ms=14)
    mini_trace(inset(fig, 32.6, 32.8, 11.5, 7.5), S.A1, seed=3)
    ax.text(38.3, 28.6, "windows x", ha="center", fontsize=12, color=S.INK2)
    ax.text(23.0, 43.0, "20,000 θ drawn uniformly; each θ is known",
            ha="center", fontsize=12.5, color=S.INK2)
    # arm R card
    S.card(ax, 1.5, 1.5, 44, 23.0, fill=S.WASH[S.A2], r=1.5)
    ax.text(3.5, 22.1, "Synthetic cohorts  (arm R: R0 and R3)", fontsize=16,
            fontweight="semibold", va="center")
    mini_cloud(inset(fig, 3.5, 5.0, 9.5, 12), None,
               clusters=[(0.5, 0.78), (0.24, 0.3), (0.76, 0.3)], seed=4)
    ax.text(8.25, 3.0, "donors' θ*", ha="center", fontsize=12, color=S.INK2)
    S.arrow(ax, (14.0, 11.0), (18.0, 11.0), color=S.INK2, lw=2, ms=14)
    pill(ax, 18.3, 7.7, 9.5, 6.6, "simulator\n+ gap (R3)", S.SURFACE, fs=12.5,
         edge=S.GRID, lw=1.2)
    S.arrow(ax, (28.1, 11.0), (32.1, 11.0), color=S.INK2, lw=2, ms=14)
    for k, c in enumerate(S.COND):
        mini_trace(inset(fig, 32.6, 12.6 - 3.6 * k, 11.5, 3.6), c, seed=10 + k, lw=1.1,
                   bursts=4 + 2 * k)
    ax.text(38.3, 3.0, "windows x", ha="center", fontsize=12, color=S.INK2)
    ax.text(23.0, 18.4, "27 donors, 3 conditions; θ* recorded and hidden",
            ha="center", fontsize=12.5, color=S.INK2)
    # the joint network
    S.card(ax, 53.0, 15.5, 19.0, 18.5, fill=S.BOX, r=1.5)
    ax.text(62.5, 29.6, "joint network", ha="center", fontsize=16, fontweight="semibold")
    ax.text(62.5, 24.6, "summary network\n+ normalizing flow", ha="center", fontsize=12.5,
            color=S.INK2, linespacing=1.3)
    ax.text(62.5, 18.6, "trained on both", ha="center", fontsize=12.5, color=S.INK2)
    S.arrow(ax, (45.8, 37.0), (52.8, 30.0), color=S.A1, lw=2.6, ms=18)
    ax.text(48.6, 38.6, "(x, θ) pairs", fontsize=12.5, color=S.INK, ha="left")
    S.arrow(ax, (45.8, 12.0), (52.8, 19.5), color=S.A2, lw=2.6, ms=18)
    ax.text(51.5, 13.6, "windows x, condition\nlabels, well pairs;\nno θ*", fontsize=12,
            color=S.INK, ha="left", va="top", linespacing=1.25)
    # posterior and score
    S.arrow(ax, (72.3, 24.75), (78.3, 24.75), color=S.INK2, lw=2.4, ms=18)
    S.card(ax, 78.5, 17.5, 20.0, 14.5, fill=S.SURFACE, edge=S.INK2, lw=1.4, r=1.5)
    ax.text(88.5, 29.2, "posterior q(θ | x)", ha="center", fontsize=14,
            fontweight="semibold")
    axp = inset(fig, 80.5, 18.5, 16.0, 8.5)
    g = np.linspace(0, 1, 300)
    dens = np.exp(-0.5 * ((g - 0.58) / 0.07) ** 2)
    axp.fill_between(g, 0, dens, color=S.A5, alpha=0.25, lw=0)
    axp.plot(g, dens, color=S.A5, lw=2)
    axp.axvline(0.62, color=S.INK, lw=1.6, ls=(0, (3, 2)))
    axp.text(0.66, 1.08, "θ*", fontsize=13, color=S.INK, va="top")
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1.15)
    S.card(ax, 78.5, 2.0, 20.0, 12.5, fill=S.WASH[S.VIOLET], r=1.5)
    ax.text(88.5, 12.2, "the score", ha="center", fontsize=14, fontweight="semibold",
            va="center")
    ax.text(88.5, 10.0, "how much density the\nposterior puts on the\nhidden θ*",
            ha="center", fontsize=12, color=S.INK2, linespacing=1.25, va="top")
    S.arrow(ax, (88.5, 17.3), (88.5, 14.7), color=S.INK2, lw=2, ms=14)
    # the hidden truth, dashed route along the bottom
    S.arrow(ax, (46.0, 4.0), (78.2, 4.0), color=S.INK2, lw=1.6, ms=14, ls=(0, (4, 3)))
    ax.text(62.0, 4.9, "θ* is used only here", ha="center", fontsize=12,
            color=S.INK2, va="bottom")
    ax.text(76.0, 45.6, "Real recordings have no θ*: this world has one.",
            ha="center", fontsize=15, fontweight="semibold", color=S.INK)
    ax.text(76.0, 41.2, "So we can ask whether reading the cohorts during\ntraining "
            "gives a better posterior on them.", ha="center", fontsize=13, color=S.INK2,
            linespacing=1.3)
    S.footer(fig, FOOT % "schematic; sizes from synth.json (D-121, D-132)")
    S.save(fig, OUT, "F01_idea")


# ---- F02 ------------------------------------------------------------------------
def fig02_simulator():
    z = np.load("data/raster.npz")
    t, i, x, fs = z["t_spk"], z["i_spk"], z["x"], float(z["fs"])
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.25, 1.0], height_ratios=[2.2, 1.0],
                          hspace=0.08, wspace=0.06, left=0.06, right=0.985, top=0.92,
                          bottom=0.135)
    axr = fig.add_subplot(gs[0, 0])
    axr.scatter(t, i, s=0.6, color=S.INK, lw=0, rasterized=True)
    axr.set_xlim(0, 40)
    axr.set_ylim(-1, 100)
    axr.set_ylabel("neuron")
    axr.tick_params(labelbottom=False)
    axr.grid(False)
    axr.set_title("100 neurons, 40 s, at condition 3's centre", loc="left", fontsize=14,
                  fontweight="normal", color=S.INK2)
    axx = fig.add_subplot(gs[1, 0], sharex=axr)
    tt = np.arange(len(x)) / fs
    axx.fill_between(tt, 0, x, color=S.OCHRE, alpha=0.2, lw=0)
    axx.plot(tt, x, color=S.OCHRE, lw=1.2)
    axx.set_xlabel("time (s)")
    axx.set_ylabel("x")
    axx.grid(axis="x", visible=False)
    # zoom on one burst
    pk = np.argmax(x)
    t0 = tt[pk] - 0.6
    t1 = t0 + 1.6
    axr.axvspan(t0, t1, color=S.WASH[S.OCHRE], lw=0, zorder=0)
    axx.axvspan(t0, t1, color=S.WASH[S.OCHRE], lw=0, zorder=0)
    az = fig.add_subplot(gs[0, 1])
    m = (t >= t0) & (t <= t1)
    az.scatter(t[m], i[m], s=5, color=S.INK, lw=0)
    az.set_xlim(t0, t1)
    az.set_ylim(-1, 100)
    az.tick_params(labelleft=False, labelbottom=False)
    az.grid(False)
    az.set_facecolor(S.WASH[S.OCHRE])
    az.set_title("zoom: one burst in two fragments", loc="left", fontsize=14,
                 fontweight="normal", color=S.INK2)
    azx = fig.add_subplot(gs[1, 1], sharex=az)
    mm = (tt >= t0) & (tt <= t1)
    azx.fill_between(tt[mm], 0, x[mm], color=S.OCHRE, alpha=0.2, lw=0)
    azx.plot(tt[mm], x[mm], color=S.OCHRE, lw=1.8)
    azx.tick_params(labelleft=False)
    azx.set_xlabel("time (s)")
    azx.grid(axis="x", visible=False)
    azx.set_facecolor(S.WASH[S.OCHRE])
    S.footer(fig, FOOT % "simulated here with the pipeline's bench generator: scripts/gen_raster.py")
    S.save(fig, OUT, "F02_simulator")


# ---- F07 ------------------------------------------------------------------------
def fig07_model():
    fig, ax = S.canvas()
    yc = 30.0
    # window
    mini_trace(inset(fig, 1.0, yc - 3.5, 11.0, 7.0), S.INK2, seed=21)
    ax.text(6.5, yc + 5.6, "window x", ha="center", fontsize=13.5, fontweight="semibold")
    ax.text(6.5, yc - 6.3, "18,000 samples", ha="center", fontsize=11.5, color=S.INK2)
    S.arrow(ax, (12.4, yc), (16.0, yc), color=S.INK2, lw=2.2, ms=16)
    # encoder trapezoid
    enc = Polygon([[16.3, yc - 8.0], [16.3, yc + 8.0], [31.5, yc + 4.0], [31.5, yc - 4.0]],
                  closed=True, facecolor=S.WASH[S.A1], edgecolor=S.A1, lw=1.6, zorder=2)
    ax.add_patch(enc)
    ax.text(23.6, yc + 1.3, "summary\nnetwork hψ", ha="center", va="center",
            fontsize=13.5, fontweight="semibold", linespacing=1.25, zorder=3)
    S.arrow(ax, (31.8, yc), (35.2, yc), color=S.INK2, lw=2.2, ms=16)
    # summary vector
    vals = [0.8, 0.35, 0.6, 0.2, 0.95, 0.5, 0.7, 0.3]
    for k, v in enumerate(vals):
        S.card(ax, 35.6 + k * 0.95, yc - 3.5, 0.75, 7.0 * v, fill=S.A1, r=0.2, z=3,
               alpha=0.75)
    ax.text(39.4, yc + 5.6, "summary z", ha="center", fontsize=13.5, fontweight="semibold")
    ax.text(39.4, yc - 6.3, "z = hψ(x)", ha="center", fontsize=12, color=S.INK2)
    S.arrow(ax, (43.6, yc), (47.0, yc), color=S.INK2, lw=2.2, ms=16)
    # flow
    S.card(ax, 47.3, yc - 7.0, 15.5, 14.0, fill=S.WASH[S.A1], edge=S.A1, lw=1.6, r=1.5)
    ax.text(55.05, yc + 1.6, "normalizing\nflow qω(θ | z)", ha="center",
            va="center", fontsize=13.5, fontweight="semibold", linespacing=1.25, zorder=3)
    S.arrow(ax, (63.1, yc), (66.5, yc), color=S.INK2, lw=2.2, ms=16)
    # posterior
    axp = inset(fig, 66.8, yc - 5.0, 14.0, 10.5)
    g = np.linspace(0, 1, 300)
    for mu, sd, a in [(0.3, 0.08, 1.0), (0.62, 0.05, 0.8)]:
        pass
    dens = 0.6 * np.exp(-0.5 * ((g - 0.33) / 0.08) ** 2) + np.exp(-0.5 * ((g - 0.64) / 0.05) ** 2)
    axp.fill_between(g, 0, dens, color=S.A1, alpha=0.2, lw=0)
    axp.plot(g, dens, color=S.A1, lw=2)
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1.15)
    ax.text(73.8, yc + 7.4, "posterior over\nthe 10 parameters", ha="center", fontsize=13,
            fontweight="semibold", linespacing=1.2)
    # ---- the three losses
    # NPE (blue), top
    S.card(ax, 40.0, 41.0, 41.0, 7.2, fill=S.WASH[S.A1], r=1.2)
    ax.text(41.3, 46.0, "NPE loss, on simulated pairs (x, θ)", fontsize=13.5,
            fontweight="semibold", va="center")
    ax.text(41.3, 42.7, "high density at the true θ; trains ψ and ω",
            fontsize=12.5, color=S.INK2, va="center")
    S.arrow(ax, (60.0, 40.8), (55.05, yc + 7.3), color=S.A1, lw=2.0, ms=14, ls="--")
    S.arrow(ax, (47.0, 40.8), (25.0, yc + 6.2), color=S.A1, lw=2.0, ms=14, ls="--",
            rad=0.15)
    # DSN (orange), bottom-left
    S.card(ax, 1.0, 1.5, 33.0, 13.5, fill=S.WASH[S.A2], r=1.2)
    ax.text(2.3, 12.6, "DSN loss  (λ_dsn),  arm A2", fontsize=13.5,
            fontweight="semibold", va="center")
    ax.text(2.3, 8.8, "on cohort windows' summaries with\ntheir condition labels: same "
            "condition\nclose, conditions apart", fontsize=12, color=S.INK2, va="center",
            linespacing=1.25)
    axd = inset(fig, 25.0, 2.3, 8.5, 9.0)
    th = np.linspace(0, 2 * np.pi, 200)
    axd.plot(np.cos(th), np.sin(th), color=S.GRID, lw=1.2)
    rng = np.random.default_rng(5)
    for k, c in enumerate(S.COND):
        a = np.pi / 2 + k * 2 * np.pi / 3
        pts = a + rng.normal(0, 0.18, 7)
        axd.scatter(np.cos(pts), np.sin(pts), s=18, color=c, edgecolor="white", lw=0.6,
                    zorder=3)
        axd.plot([0, 0.62 * np.cos(a)], [0, 0.62 * np.sin(a)], color=c, lw=1.6)
    axd.set_xlim(-1.25, 1.25)
    axd.set_ylim(-1.25, 1.25)
    axd.set_aspect("equal")
    S.arrow(ax, (21.0, 15.2), (22.6, yc - 8.6), color=S.A2, lw=2.0, ms=14, ls="--")
    ax.text(12.5, 18.4, "trains ψ only", fontsize=12.5, color=S.INK, ha="center")
    # replicate (green), bottom-right
    S.card(ax, 51.0, 1.5, 48.0, 13.5, fill=S.WASH[S.A5], r=1.2)
    ax.text(52.3, 12.6, "replicate loss  (λ_rep),  arm A5", fontsize=13.5,
            fontweight="semibold", va="center")
    ax.text(52.3, 8.8, "two windows from the two wells of one\ndonor: their posteriors "
            "should agree\nas well as their widths allow", fontsize=12, color=S.INK2,
            va="center", linespacing=1.25)
    axr = inset(fig, 83.0, 2.3, 15.0, 11.0)
    for (cx, cy), cc in [((0.38, 0.5), S.A5), ((0.62, 0.45), "#0f7a55")]:
        e = Ellipse((cx, cy), 0.42, 0.7, angle=25, facecolor=cc, alpha=0.18,
                    edgecolor=cc, lw=1.6)
        axr.add_patch(e)
        axr.scatter([cx], [cy], s=16, color=cc, zorder=3)
    axr.annotate("", xy=(0.62, 0.45), xytext=(0.38, 0.5),
                 arrowprops=dict(arrowstyle="<->", color=S.INK, lw=1.2))
    axr.text(0.5, 0.55, "Δ", ha="center", fontsize=12)
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.set_aspect("equal")
    S.arrow(ax, (53.0, 15.2), (28.5, yc - 6.6), color=S.A5, lw=2.0, ms=14, ls="--",
            rad=-0.12)
    ax.text(38.5, 16.1, "trains ψ only", fontsize=12.5, color=S.INK, ha="center")
    # blocked route into the flow's weights: the stop-gradient
    ax.plot([60.0, 57.0], [15.2, yc - 7.4], color=S.A5, lw=2.0, ls="--")
    ax.plot([57.3, 59.7], [18.6, 21.0], color=S.RED, lw=3)
    ax.plot([57.3, 59.7], [21.0, 18.6], color=S.RED, lw=3)
    ax.text(61.2, 19.8, "stop-gradient: the flow's ω\nis not changed by this term",
            fontsize=12, color=S.INK, va="center", linespacing=1.2)
    ax.text(1.0, 47.6, "dashed arrows: which weights each loss trains", fontsize=12,
            color=S.INK2, va="center")
    S.footer(fig, FOOT % "schematic of eq. (1); E4, P2, P3 design notes")
    S.save(fig, OUT, "F07_model")


# ---- F08 ------------------------------------------------------------------------
def fig08_arms():
    fig, ax = S.canvas()
    cols = [(1.0, S.A1, "A1", "NPE only"),
            (34.5, S.A2, "A2", "NPE + DSN loss"),
            (68.0, S.A5, "A5", "NPE + replicate loss")]
    w = 31.0
    for x0, c, name, sub in cols:
        S.card(ax, x0, 1.5, w, 46.5, fill=S.WASH[c], r=1.5)
        S.card(ax, x0, 44.0, w, 4.0, fill=c, r=1.5)
        ax.text(x0 + 1.5, 46.0, "%s   %s" % (name, sub), fontsize=15.5,
                fontweight="semibold", color="white", va="center")
    # A1
    a1 = inset(fig, 4.0, 21.0, 25.0, 19.5)
    g = np.linspace(0, 1, 300)
    a1.plot(g, np.ones_like(g), color=S.MUTED, lw=1.6, ls=(0, (4, 2)))
    a1.fill_between(g, 0, 4.5 * np.exp(-0.5 * ((g - 0.6) / 0.09) ** 2), color=S.A1,
                    alpha=0.2, lw=0)
    a1.plot(g, 4.5 * np.exp(-0.5 * ((g - 0.6) / 0.09) ** 2), color=S.A1, lw=2.2)
    a1.axvline(0.63, color=S.INK, lw=1.5, ls=(0, (3, 2)))
    a1.text(0.645, 4.6, "true θ", fontsize=12.5)
    a1.text(0.02, 1.25, "prior", fontsize=12, color=S.INK2)
    a1.text(0.02, 3.4, "posterior", fontsize=12, color=S.INK)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 5.2)
    ax.text(2.5, 16.5, "Learns from simulated pairs (x, θ)\nalone: the reference.",
            fontsize=13, va="top", linespacing=1.3)
    ax.text(2.5, 6.2, "reads: arm S only\nchanges: summary network and flow",
            fontsize=12.5, color=S.INK2, va="top", linespacing=1.3)
    # A2
    a2 = inset(fig, 38.0, 20.0, 24.0, 21.0)
    th = np.linspace(0, 2 * np.pi, 200)
    a2.plot(np.cos(th), np.sin(th), color=S.GRID, lw=1.4)
    rng = np.random.default_rng(3)
    for k, c in enumerate(S.COND):
        a = np.pi / 2 + k * 2 * np.pi / 3
        pts = a + rng.normal(0, 0.16, 9)
        a2.scatter(np.cos(pts), np.sin(pts), s=34, color=c, edgecolor="white", lw=0.8,
                   zorder=3)
        a2.annotate("", xy=(0.66 * np.cos(a), 0.66 * np.sin(a)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
    a2.text(0.0, -0.22, "120°", ha="center", va="top", fontsize=12, color=S.INK2)
    a2.set_xlim(-1.3, 1.3)
    a2.set_ylim(-1.3, 1.3)
    a2.set_aspect("equal")
    ax.text(36.0, 16.5, "Summaries of one condition's windows\nclose together; the "
            "condition means\nspread apart.", fontsize=13, va="top", linespacing=1.3)
    ax.text(36.0, 6.2, "reads: the cohort's condition labels\nchanges: the summary "
            "network only", fontsize=12.5, color=S.INK2, va="top", linespacing=1.3)
    # A5
    a5 = inset(fig, 71.0, 20.0, 25.0, 21.0)
    for (cx, cy), cc, lab in [((0.36, 0.52), S.A5, "well 1"), ((0.64, 0.46), "#0f7a55",
                                                               "well 2")]:
        a5.add_patch(Ellipse((cx, cy), 0.36, 0.62, angle=25, facecolor=cc, alpha=0.18,
                             edgecolor=cc, lw=1.8))
        a5.scatter([cx], [cy], s=26, color=cc, zorder=3)
        a5.text(cx, cy - 0.42 if cx < 0.5 else cy + 0.36, lab, ha="center", fontsize=12,
                color=S.INK)
    a5.annotate("", xy=(0.64, 0.46), xytext=(0.36, 0.52),
                arrowprops=dict(arrowstyle="<->", color=S.INK, lw=1.4))
    a5.text(0.5, 0.56, "Δ", ha="center", fontsize=14)
    a5.set_xlim(0, 1)
    a5.set_ylim(0, 1)
    a5.set_aspect("equal")
    ax.text(69.5, 16.5, "Same donor, two wells: the posteriors'\ngap Δ, measured in "
            "their widths (T),\nshould match what chance allows (p_eff).", fontsize=13,
            va="top", linespacing=1.3)
    ax.text(69.5, 6.2, "reads: pairs of windows, two wells\nchanges: the summary network "
            "only", fontsize=12.5, color=S.INK2, va="top", linespacing=1.3)
    S.footer(fig, FOOT % "schematic; D-118, D-121, D-122; P2 (eq. P2.8), P3 (eqs. P3.4-P3.7)")
    S.save(fig, OUT, "F08_arms")


# ---- F09 ------------------------------------------------------------------------
def fig09_procedure():
    fig, ax = S.canvas()
    # stage 1: the search
    S.card(ax, 1.0, 14.0, 36.0, 34.0, fill=S.BOX, r=1.5)
    ax.text(2.5, 45.6, "1  Search  (60 trials)", fontsize=15.5, fontweight="semibold",
            va="center")
    ax.text(2.5, 42.6, "each trial: one network, 6 passes, scored by L\n"
            "(NLL at the truth on simulated windows);\n20 random, then 4 rounds of 10 "
            "picked by a\nGaussian-process model of L (darker = later)",
            fontsize=12, color=S.INK2, va="top", linespacing=1.3)
    shades = ["#bcd6f3", "#8fb9ea", "#5f99de", S.A1, "#163f73"]
    k = 0
    fin = (7, 5)
    for r, n in enumerate([20, 10, 10, 10, 10]):
        for _j in range(n):
            col_, row_ = k % 10, k // 10
            xx, yy = 4.2 + col_ * 3.15, 32.4 - row_ * 2.55
            if (col_, row_) == fin:
                ax.add_patch(Circle((xx, yy), 1.45, facecolor="white", edgecolor=S.INK,
                                    lw=2.2, zorder=4))
            else:
                ax.add_patch(Circle((xx, yy), 0.95, color=shades[r], zorder=3))
            k += 1
    ax.add_patch(Circle((4.2, 15.9), 1.2, facecolor="white", edgecolor=S.INK, lw=2.0,
                        zorder=4))
    ax.text(6.2, 15.9, "the finalist: the trial with the lowest L", fontsize=12,
            va="center", color=S.INK)
    S.arrow(ax, (37.3, 31.0), (41.3, 31.0), color=S.INK2, lw=2.4, ms=18)
    # stage 2: the ensemble
    S.card(ax, 41.6, 14.0, 26.0, 34.0, fill=S.BOX, r=1.5)
    ax.text(43.1, 45.6, "2  Retrain it 5 times", fontsize=15.5, fontweight="semibold",
            va="center")
    ax.text(43.1, 42.6, "seeds 1-5; up to 40 epochs; stop\nafter 10 without improvement",
            fontsize=12, color=S.INK2, va="top", linespacing=1.3)
    for m in range(5):
        yy = 33.0 - m * 3.5
        ax.add_patch(Circle((46.6, yy), 1.25, color=S.A1, alpha=0.55, zorder=3))
        ax.text(48.6, yy, "member %d" % (m + 1), fontsize=12, va="center")
        S.arrow(ax, (55.2, yy), (59.4, 26.0), color=S.MUTED, lw=1.2, ms=10)
    d = Polygon([[60.0, 26.0], [62.4, 29.0], [64.8, 26.0], [62.4, 23.0]], closed=True,
                facecolor=S.A1, edgecolor=S.INK, lw=1.2, zorder=4)
    ax.add_patch(d)
    ax.text(62.4, 21.6, "ensemble =\naverage of\nthe 5\ndensities", ha="center",
            fontsize=12, va="top", linespacing=1.2)
    S.arrow(ax, (67.9, 31.0), (71.9, 31.0), color=S.INK2, lw=2.4, ms=18)
    # stage 3: the evaluation
    S.card(ax, 72.2, 14.0, 26.8, 34.0, fill=S.BOX, r=1.5)
    ax.text(73.7, 45.6, "3  Score the ensemble", fontsize=15.5, fontweight="semibold",
            va="center")
    ax.text(73.7, 42.6, "NLL at the truth, coverage,\npaired comparisons", fontsize=12,
            color=S.INK2, va="top", linespacing=1.3)
    targets = [("simulated test windows", "1,000 donors x 2 windows", S.A1),
               ("cohort R0", "324 windows; \u03b8* used only here", S.A2),
               ("cohort R3", "324 windows, with the gap", S.A2)]
    for j, (a, b, c) in enumerate(targets):
        yy = 33.2 - j * 7.4
        S.card(ax, 73.7, yy - 3.0, 23.8, 6.4, fill=S.SURFACE, edge=c, lw=1.4, r=1.0)
        ax.text(75.0, yy + 1.15, a, fontsize=13, fontweight="semibold", va="center")
        ax.text(75.0, yy - 1.45, b, fontsize=11.5, color=S.INK2, va="center")
    # bottom band: five searches and the split
    names = ["S-A1", "S-A2-R0", "S-A5-R0", "S-A2-R3", "S-A5-R3"]
    arms = ["A1", "A2", "A5", "A2", "A5"]
    ax.text(1.0, 10.2, "Done 5 times, one search per arm and cohort:", fontsize=13,
            fontweight="semibold", va="center")
    for j, (nm, a) in enumerate(zip(names, arms)):
        x0 = 33.5 + j * 13.2
        pill(ax, x0, 7.9, 12.2, 4.6, nm, S.ARM[a], ink="white", fs=13)
    ax.text(1.0, 3.6, "Bank donors split 85 / 10 / 5 %: training / selection (early "
            "stopping) / report (L and the simulated test)", fontsize=12.5,
            color=S.INK2, va="center")
    S.footer(fig, FOOT % "schematic; synth.json search, best_fit, split; SPEC 2.5")
    S.save(fig, OUT, "F09_procedure")


# ---- F10 ------------------------------------------------------------------------
def fig10_scoring():
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.2, left=0.06,
                          right=0.985, top=0.86, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    g = np.linspace(0, 1, 600)
    ts = 0.62

    def gauss(mu, sd):
        return np.exp(-0.5 * ((g - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    good, bad = gauss(0.6, 0.05), gauss(0.40, 0.06)
    ax.plot(g, np.ones_like(g), color=S.MUTED, lw=2.2, ls=(0, (5, 2.5)))
    ax.fill_between(g, 0, good, color=S.A5, alpha=0.18, lw=0)
    ax.plot(g, good, color=S.A5, lw=2.4)
    ax.fill_between(g, 0, bad, color=S.RED, alpha=0.12, lw=0)
    ax.plot(g, bad, color=S.RED, lw=2.4)
    ax.axvline(ts, color=S.INK, lw=1.6, ls=(0, (3, 2)))
    qg = float(np.interp(ts, g, good))
    qb = float(np.interp(ts, g, bad))
    ax.scatter([ts, ts], [qg, 1.0], s=60, color=S.INK, zorder=5)
    ax.text(ts + 0.015, 9.0, "true \u03b8*", fontsize=13, va="top")
    ax.text(0.70, 6.3, "sharp and right:\nNLL = \u2212log %.1f = %.1f" % (qg, -np.log(qg)),
            fontsize=12.5, va="center", linespacing=1.3)
    ax.text(0.70, 1.35, "prior: NLL = 0", fontsize=12.5, color=S.INK2)
    ax.text(0.02, 8.3, "sharp and wrong:\nNLL = \u2212log %.3f = +%.1f" % (qb, -np.log(qb)),
            fontsize=12.5, va="center", linespacing=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 9.4)
    ax.set_xlabel("one parameter (unit-box coordinate)")
    ax.set_ylabel("posterior density")
    ax.grid(axis="x", visible=False)
    ax.set_title("Score: NLL at the truth (lower is better)", loc="left", fontsize=14.5)
    # the decision rule
    ax2 = fig.add_subplot(gs[1])
    cases = [(1.15, (0.8, 1.5), 0.85, "resolved: outside the band and the interval excludes 0"),
             (0.55, (0.35, 0.75), 1.0, "not resolved: the interval excludes 0, but inside the band"),
             (0.15, (-0.15, 0.45), 1.0, "not resolved: the interval contains 0")]
    for r, (dbar, ci, tw, lab) in enumerate(cases):
        y = 2 - r
        ax2.plot([-tw, tw], [y, y], color=S.GRID, lw=12, solid_capstyle="butt")
        ax2.plot(ci, [y, y], color=S.INK2, lw=2.2)
        ax2.scatter([dbar], [y], s=90, color=S.INK, zorder=3)
        ax2.text(-1.75, y + 0.3, lab, fontsize=12.5, va="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax2.axvline(0, color=S.INK2, lw=1.2)
    ax2.set_xlim(-1.8, 1.7)
    ax2.set_ylim(-0.5, 2.95)
    ax2.set_yticks([])
    ax2.set_xticks([-1, 0, 1])
    ax2.set_xlabel("D\u0304: paired difference in NLL between two arms (nats)")
    ax2.grid(axis="y", visible=False)
    ax2.text(-1.75, 2.66, "grey band: \u00b12\u03c3 seed, the spread of single members\n"
             "dot: D\u0304;  line: its 95 % interval over donors", fontsize=11.5,
             color=S.INK2, va="center", linespacing=1.3)
    ax2.set_title("Rule G.2: when is a difference real?", loc="left", fontsize=14.5)
    S.footer(fig, FOOT % "illustration of eqs. (1)-(2) and (10)-(11) of the results document; rule G.2, deck pack 08")
    S.save(fig, OUT, "F10_scoring")


# ---- F18 ------------------------------------------------------------------------
def fig18_take_home():
    fig, ax = S.canvas()
    tiles = [
        (S.A1, "Optimisation", "worked, not finished",
         "every search went from worse\nthan the prior to 3-4 nats\nbetter; 4 of 5 finalists came\n"
         "from the last round"),
        (S.A1, "Training", "members overfit;\nthe ensemble repairs it",
         "averaging 5 members gains\n0.7-1.1 nats; the ensemble is\ncalibrated to conservative\n"
         "inside the box"),
        (S.RED, "Calibration", "good inside,\nfails at the box edges",
         "10.8 % of simulated truths lie\nwithin 0.0067 of an edge,\nwhere the flow puts almost\n"
         "no mass"),
        (S.INK2, "Comparison", "no arm wins or loses",
         "0 of 15 differences resolved;\nthe searches chose small\nterm weights; on R3 the tie\n"
         "hides a trade"),
    ]
    w, h = 23.5, 29.0
    for k, (c, head, verdict, body) in enumerate(tiles):
        x0 = 1.0 + k * 24.8
        S.card(ax, x0, 18.0, w, h, fill=S.BOX, r=1.5)
        S.card(ax, x0, 18.0 + h - 1.3, w, 1.3, fill=c, r=0.6)
        ax.text(x0 + 1.4, 43.0, head, fontsize=15.5, fontweight="semibold", va="center")
        ax.text(x0 + 1.4, 40.0, verdict, fontsize=13.5, va="top", color=S.INK,
                linespacing=1.25, fontweight="semibold")
        ax.text(x0 + 1.4, 32.6, body, fontsize=12.3, va="top", color=S.INK2,
                linespacing=1.35)
    S.card(ax, 1.0, 1.5, 98.0, 14.5, fill=S.WASH[S.VIOLET], r=1.5)
    ax.text(2.5, 13.3, "Open before any verdict on the arms", fontsize=14.5,
            fontweight="semibold", va="center")
    ax.text(2.5, 9.6, "\u2022 run the searches longer?   \u2022 the flow's edge support: "
            "bound 5 or 10?   \u2022 confirm that the 18 flagged windows are the "
            "contaminated ones", fontsize=12.5, color=S.INK, va="center")
    # [corrected 2026-11-18] was: "To discuss: a ranking criterion that does not push
    # the terms towards off; ..." -- in the 6-pass trials a larger weight also worsened
    # the cohort score (study doc of 2026-11-18, check 5), so the criterion is not the
    # whole reason the weights are small.
    ax.text(2.5, 5.9, "To discuss: full ensembles at a fixed larger term weight (in 6-pass "
            "trials larger weights also", fontsize=12.5, color=S.INK2, va="center")
    ax.text(2.5, 3.4, "worsened the cohort score); a second ensemble per arm, to measure "
            "an ensemble's seed noise", fontsize=12.5, color=S.INK2, va="center")
    S.footer(fig, FOOT % "summary of the results document (synth v4 evaluation, 2026-11-16)")
    S.save(fig, OUT, "F18_take_home")


def main(deck):
    os.chdir(deck)
    S.setup()
    fig01_idea()
    fig02_simulator()
    fig07_model()
    fig08_arms()
    fig09_procedure()
    fig10_scoring()
    fig18_take_home()


if __name__ == "__main__":
    main(sys.argv[1])
