"""House style for the synthetic-runs deck figures (shared by every figure script).

Palette: the project's deck-pack style guide (claude/deck_pack/11_STYLE_GUIDE.md,
2026-09-22), validated with the dataviz skill's palette checker on 2026-11-18:
- arms A1 / A2 / A5: #2a78d6 / #eb6834 / #1baf7a (PASS; green needs a visible label);
- conditions 1 / 2 / 3: #4a3aa7 / #e87ba4 / #eda100 (PASS; magenta, ochre need labels);
- red #e34948 marks problems and is never used next to the A2 orange.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch   # noqa: E402

# ---- tokens -----------------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BOX = "#f0efec"
SURFACE = "#ffffff"

A1 = "#2a78d6"
A2 = "#eb6834"
A5 = "#1baf7a"
ARM = {"A1": A1, "A2": A2, "A5": A5}
VIOLET = "#4a3aa7"
MAGENTA = "#e87ba4"
OCHRE = "#eda100"
RED = "#e34948"
COND = (VIOLET, MAGENTA, OCHRE)

# light washes (about 12 % of the hue on white) for card fills
WASH = {A1: "#e3eefa", A2: "#fce9e1", A5: "#ddf4ec", VIOLET: "#e7e4f4",
        MAGENTA: "#fbeaf1", OCHRE: "#fcf1d6", RED: "#fbe3e3"}

FULL = (13.33, 6.6)       # inches, the content area of a 16:9 slide under its title
HALF = (6.5, 6.0)


def setup():
    matplotlib.rcParams.update({
        "font.family": "Inter",
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.titleweight": "semibold",
        "axes.labelsize": 15,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.frameon": False,
        "legend.fontsize": 13,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.fonttype": "none",
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def card(ax, x, y, w, h, fill=BOX, edge=None, lw=0.0, r=0.04, z=1, alpha=1.0):
    """A rounded box in data coordinates (axes with xlim/ylim set)."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r,
                       facecolor=fill, edgecolor=edge or fill, linewidth=lw,
                       zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=INK2, lw=2.0, style="-|>", ms=18, z=3, ls="-",
          rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
                        linewidth=lw, zorder=z, linestyle=ls,
                        connectionstyle="arc3,rad=%g" % rad,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def canvas(size=FULL):
    """A blank drawing canvas in unit coordinates, x in [0, 100], y in [0, 50*h/w*2]."""
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * size[1] / size[0])
    ax.axis("off")
    return fig, ax


def footer(fig, text):
    fig.text(0.995, 0.008, text, ha="right", va="bottom", fontsize=9, color=MUTED)


def save(fig, out_dir, name):
    import os
    for ext, dpi in (("png", 200), ("svg", None)):
        path = os.path.join(out_dir, "%s.%s" % (name, ext))
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("wrote", name)
