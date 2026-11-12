#!/usr/bin/env python3
"""Figure 01 -- schematic: how a window of each data set is made, and which
of the data overview's figure families (D1-D4) reads which object.

A structure figure, not to scale and computed from nothing; its content is
read from the code (hpc/joint/stage1: build_latent_bank.py,
latent_sbi_simulator.py, bench_burst_provider.py, latent_gap.py,
latent_nuisance.py; hpc/pipeline/evaluation/ev_dataview.py).

    python3 make_fig01_schematic.py        # writes ../figures/fig01_window_pipeline.png

Pure ASCII, LF only (the figure's text uses matplotlib's mathtext).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = {"dpi": 110, "out": os.path.join(HERE, "..", "figures", "fig01_window_pipeline.png")}

INK, MUTED, EDGE = "#1f1f1f", "#5a5a5a", "#8a8a8a"
FILL = {"param": "#eef2f8", "gen": "#f3f3f3", "gap": "#fbeeee", "obs": "#eef6ef", "view": "#fff8e6"}


def box(ax, x, y, w, h, title, body, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=FILL[kind], ec=EDGE, lw=0.8))
    ax.text(x + 0.012, y + h - 0.018, title, ha="left", va="top", fontsize=8.2,
            fontweight="bold", color=INK)
    ax.text(x + 0.012, y + h - 0.050, body, ha="left", va="top", fontsize=7.2, color=INK,
            linespacing=1.35)


def arrow(ax, p, q, text=None, dx=0.006):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=EDGE))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2, text, fontsize=6.8, color=MUTED,
                ha="left", va="center")


def tag(ax, x, y, text):
    ax.text(x, y, text, fontsize=7.4, color="#8a5a00", fontweight="bold", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff1cc", ec="#e0b24d", lw=0.6))


def main():
    fig = plt.figure(figsize=(9.6, 8.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, "How a window of each data set is made (schematic, not to scale; read "
            "from the code)", ha="center", va="top", fontsize=9.5, color=INK)

    # parameters: two boxes side by side
    box(ax, 0.03, 0.835, 0.44, 0.120, "Simulated arm S  (S_sub: shards 0-1)",
        "one $\\theta_d$ per donor $d$, i.i.d. uniform on the box $(0,1)^{10}$\n"
        "2 wells per donor, 1 window per well ($J=1$)\n"
        "2,000 donors, 4,000 windows", "param")
    box(ax, 0.53, 0.835, 0.44, 0.120, "Cohorts R0 ($\\pi=0$) and R3 ($\\pi=0.3$)",
        "one $\\theta^*_d$ per donor, condition $c\\in\\{0,1,2\\}$, 9 donors each:\n"
        "label axes 0-6: normal($\\mu_{c,k}$, $\\tau=0.10$) truncated to $(0,1)$\n"
        "free axes 7-9: uniform on $(0,1)$; 2 wells $\\times$ 6 windows ($J=6$)", "param")
    tag(ax, 0.80, 0.818, "D3: theta_cohorts, theta_dagger")
    box(ax, 0.03, 0.680, 0.94, 0.120, "Axis map  $\\theta \\mapsto a$  (physical parameters)",
        "$a_k = L_k\\,(1-\\theta_k) + U_k\\,\\theta_k$ on each axis $k$ (ranges in Table 2)\n"
        "R3 only, gap mode (a), range shift: each free axis' range $[L_k, U_k]$ moved up by "
        "$\\delta\\,(U_k-L_k)$, $\\delta = \\pi \\times 0.35 = 0.105$;\n"
        "the duty $a_7$ capped at 1.  The recorded $\\theta^*$ is never shifted: "
        "$\\theta^\\dagger = \\theta^* + \\delta$ is the box point with R3's physics", "param")
    box(ax, 0.03, 0.545, 0.94, 0.105, "Bench burst generator  +  IFR",
        "spike trains of $N = 100$ units for $J \\times 180$ s (one realisation per well) "
        "$\\rightarrow$ population count per 10-ms bin\n"
        "$\\rightarrow$ Gaussian smoothing, SD $\\sigma_{sm} = 40$ ms $\\rightarrow$ divided by $N$ "
        "$\\rightarrow$ cut into $J$ windows $x^{gen}$ of $W = 18{,}000$ samples ($f_s = 100$ Hz)",
        "gen")
    box(ax, 0.03, 0.400, 0.94, 0.115, "Gap, trace layer  (R3 only; identity at $\\pi = 0$)",
        "$x^{gen} \\rightarrow x^{gap}$.  (c) drift: $+\\;\\pi \\cdot 0.5 \\cdot \\bar{x}_{tr}\\,"
        "\\sin(2\\pi t/120\\,\\mathrm{s} + \\varphi)$, one phase $\\varphi$ per trace, "
        "$\\bar{x}_{tr}$ the trace mean\n"
        "(d) contamination: 0 or 1 window per trace (probability 0.36), "
        "$x \\rightarrow \\max(g\\,x + \\epsilon,\\ 0)$, $g \\sim U(2,5)$, "
        "$\\epsilon$ white with SD $\\mathrm{sd}(x)$", "gap")
    box(ax, 0.03, 0.255, 0.94, 0.115, "Nuisance  (every arm; observation level, applied last)",
        "$x = A(\\nu)\\,x^{gap} + \\nu_b + \\nu_{dr}\\,\\sin(2\\pi t/600\\,\\mathrm{s})$, "
        "$t$ the trace's absolute time\n"
        "$\\nu = \\nu_{batch} + \\nu_{donor} + \\nu_{well}$ (Gaussian, nested); "
        "$A(\\nu) > 0$: gain, threshold, dropout", "obs")
    tag(ax, 0.73, 0.238, "D1: windows_*  (the window $x$)")
    box(ax, 0.03, 0.040, 0.44, 0.165, "Window statistics  $T(x) \\in \\mathbb{R}^8$",
        "mean, sd, max, q90, burst_frac,\nacf1, skew, roughness  (eqs. 12-19)\n"
        "S_sub's 1, 50, 99 % quantiles $\\hat q_\\alpha$:\nthe band of the strips", "view")
    tag(ax, 0.06, 0.022, "D2: stats_hist, stats_strip")
    box(ax, 0.53, 0.040, 0.44, 0.165, "Standardise, project",
        "$z = (T(x) - m) \\oslash s$, with $m$, $s$ S_sub's mean, SD\n"
        "PCA fitted on S_sub's $z$ alone:\n"
        "$y_k = u_k^\\top z$, $k = 1, 2$ (the plane of D4)", "view")
    tag(ax, 0.56, 0.022, "D4: stats_pca")

    arrow(ax, (0.25, 0.835), (0.25, 0.800), r"$\theta_d$")
    arrow(ax, (0.75, 0.835), (0.75, 0.800), r"$\theta^*_d$")
    arrow(ax, (0.50, 0.680), (0.50, 0.650), r"$a$: rates, durations, participation")
    arrow(ax, (0.50, 0.545), (0.50, 0.515), "$x^{gen}$: the $J$ windows of the well")
    arrow(ax, (0.50, 0.400), (0.50, 0.370), "$x^{gap}$ ($= x^{gen}$ in S_sub and R0)")
    arrow(ax, (0.25, 0.255), (0.25, 0.205), "the stored window $x$ (every data set)")
    arrow(ax, (0.47, 0.120), (0.53, 0.120))
    fig.savefig(os.path.normpath(CONFIG["out"]), dpi=CONFIG["dpi"])
    print("wrote %s" % os.path.normpath(CONFIG["out"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
