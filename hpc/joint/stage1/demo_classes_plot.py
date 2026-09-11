#!/usr/bin/env python3
"""Figures for the 3-class demonstration set. PLOTTING ONLY.

Reads the .npz written by `demo_classes_generate.py` and writes three
figures. It never simulates: every number plotted came from the file, so
a figure can be redrawn without re-running the generator and cannot
silently disagree with the data it claims to show.

    demo_traces_full.png   one panel per class, `--n-show` example traces
                           each, all J windows concatenated
    demo_traces_zoom.png   the same exemplars over `--zoom-s` seconds,
                           where individual bursts are resolved
    demo_tsne.png          t-SNE of the GENERATING coordinates (default)
                           or of the traces, class draws over the
                           uniform-box background

On the t-SNE, twice
-------------------
t-SNE preserves neighbourhoods, not distances: cluster separation on the
map is evidence that the classes have distinct neighbourhoods, while the
GAP WIDTHS and the positions of clusters relative to each other are not
interpretable, and change with perplexity and seed. The figure is
therefore annotated with both, and `--perplexity` is exposed so the
stability of the picture can be checked rather than assumed. For the
phi embedding the honest reading is narrow: the class structure of
eq. (7) lives on 7 of 10 axes by construction, so finding it there
confirms the sampler, not the generator. The x embedding is the one that
carries scientific content -- it asks whether that structure SURVIVES
the map phi -> spikes -> IFR, which is the question the bench exists to
answer.

Standardisation before the embedding is deliberate: phi axes are already
commensurate (all in [0, 1]), but trace features are not, and an
unscaled t-SNE on x would be dominated by whichever axis happens to
carry the largest counts. PCA to `--pca-dim` precedes t-SNE on x for the
usual reason (denoising and speed at high input dimension); it is skipped
for phi, where p = 10 is already below it.

Matplotlib + scikit-learn. Pure ASCII, LF only.
"""

import argparse
import ast
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

from sklearn.decomposition import PCA                             # noqa: E402
from sklearn.manifold import TSNE                                 # noqa: E402
from sklearn.preprocessing import StandardScaler                  # noqa: E402

CLASS_COLOURS = ("#1b6ca8", "#c0392b", "#1e8449", "#8e44ad", "#d68910")
BG_COLOUR = "#b9b9b9"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_demo(path):
    """Read the demo file. Returns (arrays dict, meta dict)."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in ("x", "phi", "cls", "rid")}
        meta = ast.literal_eval(str(z["meta"]))
    return arrays, meta


def concat_windows(x_trace):
    """(J, W) -> (J * W,). Windows are disjoint and consecutive."""
    return np.asarray(x_trace, dtype=np.float64).ravel()


def exemplars(cls, n_show, rng):
    """Indices of `n_show` traces per class, background excluded."""
    out = {}
    for k in sorted(set(int(c) for c in cls if c >= 0)):
        idx = np.flatnonzero(cls == k)
        out[k] = rng.permutation(idx)[:int(n_show)]
    return out


# ---------------------------------------------------------------------------
# Figure 1 and 2: traces
# ---------------------------------------------------------------------------

def plot_traces(arrays, meta, out_path, n_show=3, t_lo=None, t_hi=None,
                title=None, seed=0):
    x, cls = arrays["x"], arrays["cls"]
    fs = float(meta["fs"])
    J, W = x.shape[1], x.shape[2]
    t = np.arange(J * W) / fs
    sel = exemplars(cls, n_show, np.random.default_rng(seed))
    lo = 0.0 if t_lo is None else float(t_lo)
    hi = t[-1] if t_hi is None else float(t_hi)
    m = (t >= lo) & (t <= hi)

    fig, axes = plt.subplots(len(sel), 1, figsize=(11, 2.1 * len(sel) + 1.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (k, idx) in zip(axes, sorted(sel.items())):
        colour = CLASS_COLOURS[k % len(CLASS_COLOURS)]
        for j, i in enumerate(idx):
            ax.plot(t[m], concat_windows(x[i])[m], lw=0.8, color=colour,
                    alpha=1.0 if j == 0 else 0.45)
        ax.set_ylabel("class %d\nIFR [counts/bin/unit]" % k, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if t_hi is None and J > 1:
            for b in range(1, J):
                ax.axvline(b * W / fs, color="0.75", lw=0.6, ls=":")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title or "Bench generator: example traces by class",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: t-SNE
# ---------------------------------------------------------------------------

def trace_features(x):
    """(n, J, W) -> (n, J * W). The whole trace, no summary statistic.

    Deliberately not a feature engineering step. Expect this embedding to
    show NO class structure, and note that this is a fact about the
    REPRESENTATION, not about the generator: bursts occur at random
    times, so two traces from one class are misaligned and the Euclidean
    distance between them is dominated by burst PHASE. Sample-by-sample
    comparison of a stochastic point process is uninformative by
    construction -- which is the reason the joint stack learns an
    encoder instead of feeding raw x to anything.
    """
    return np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)


def psd_features(x, fs, smooth_bins=5):
    """(n, J, W) -> (n, F) log power spectral density, window-averaged.

    Shift-invariant by construction: |FFT|^2 discards phase, so a burst
    at t = 2 s and the same burst at t = 9 s give IDENTICAL features
    (exactly, to float precision -- asserted by D5b). This is the minimum
    change that makes "did the class structure survive the map
    phi -> spikes -> IFR?" a question about the dynamics rather than
    about burst timing.

    [CORRECTION] An earlier version used `scipy.signal.welch` with its
    defaults. That is the better estimator in general and the wrong one
    here: Welch segments the window and applies a Hann taper, so a burst
    landing near a segment edge is attenuated, and the same burst at two
    different times gives spectra differing by up to ~1.4 log10 units
    (6.1 with a boxcar window and segmentation) -- measured, not
    estimated. The taper destroys exactly the invariance the feature
    exists to provide. A single full-length untapered periodogram gives
    0.000 spread across shifts, so that is what is used.

    Variance, which segmentation would otherwise have bought, is
    recovered two ways that both PRESERVE the invariance because both
    are linear operations on |FFT|^2: averaging over the J windows of a
    trace, and Daniell smoothing across `smooth_bins` adjacent
    frequencies (`scipy.ndimage.uniform_filter1d`).

    log10(PSD + eps) rather than PSD: burst spectra span orders of
    magnitude across the box, and an unlogged spectrum would let the
    single loudest frequency set every distance.
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import periodogram
    x = np.asarray(x, dtype=np.float64)
    n, J, W = x.shape
    _f, p = periodogram(x.reshape(n * J, W), fs=float(fs), window="boxcar",
                        scaling="density", axis=-1)
    p = p.reshape(n, J, -1).mean(axis=1)          # average over windows
    m = int(smooth_bins)
    if m > 1:
        p = uniform_filter1d(p, size=m, axis=-1, mode="nearest")
    return np.log10(p + 1e-12)


def embed(features, perplexity=25.0, pca_dim=0, seed=0):
    """Standardise -> optional PCA -> t-SNE. Returns (n, 2)."""
    z = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
    n_comp = int(pca_dim)
    if n_comp and n_comp < z.shape[1]:
        z = PCA(n_components=min(n_comp, z.shape[0] - 1),
                random_state=seed).fit_transform(z)
    # sklearn's HARD constraint is perplexity < n_samples; the (n - 1) / 3
    # cap below it is the usual rule of thumb, applied so a small demo set
    # does not get a perplexity that smears every neighbourhood into one.
    # [CORRECTION] the previous clamp floored at 5.0, which for n < 16
    # exceeded the rule-of-thumb cap it claimed to enforce.
    per = float(min(float(perplexity), max(2.0, (z.shape[0] - 1) / 3.0)))
    return TSNE(n_components=2, perplexity=per, init="pca",
                random_state=seed).fit_transform(z), per


def plot_tsne(arrays, meta, out_path, source="phi", perplexity=25.0,
              pca_dim=50, seed=0):
    cls = arrays["cls"]
    if source == "phi":
        feats, pca_used = arrays["phi"], 0
        what = "generating coordinates phi in [0, 1]^%d" % arrays["phi"].shape[1]
    elif source == "x":
        feats, pca_used = trace_features(arrays["x"]), pca_dim
        what = "raw traces x (%d samples each) -- phase-dominated" % feats.shape[1]
    elif source == "psd":
        feats = psd_features(arrays["x"], meta["fs"])
        pca_used = 0
        what = "log PSD of x (%d bands) -- shift-invariant" % feats.shape[1]
    else:
        raise ValueError("source must be 'phi', 'x' or 'psd', got %r"
                         % (source,))

    y, per_used = embed(feats, perplexity=perplexity, pca_dim=pca_used,
                        seed=seed)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    bg = cls < 0
    if np.any(bg):
        ax.scatter(y[bg, 0], y[bg, 1], s=16, c=BG_COLOUR, alpha=0.65,
                   linewidths=0, label="full box (uniform)")
    for k in sorted(set(int(c) for c in cls if c >= 0)):
        m = cls == k
        ax.scatter(y[m, 0], y[m, 1], s=26,
                   c=CLASS_COLOURS[k % len(CLASS_COLOURS)],
                   edgecolors="white", linewidths=0.4, label="class %d" % k)
    ax.set_xlabel("t-SNE 1 [arbitrary units]")
    ax.set_ylabel("t-SNE 2 [arbitrary units]")
    ax.set_title("Where the class-generating parameters fall\n"
                 "t-SNE of %s" % what, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.01, -0.13,
            "perplexity %.0f, seed %d%s. Neighbourhoods are meaningful; "
            "gap widths and inter-cluster distances are NOT." %
            (per_used, seed, ", PCA to %d dims" % pca_used if pca_used else ""),
            transform=ax.transAxes, fontsize=7.5, color="0.35")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", default="demo_classes.npz")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--n-show", type=int, default=3,
                   help="example traces per class in the trace figures")
    p.add_argument("--zoom-s", type=float, default=6.0,
                   help="close-up span in seconds")
    p.add_argument("--zoom-start", type=float, default=0.0)
    p.add_argument("--embed", choices=("phi", "x", "psd", "all"),
                   default="all")
    p.add_argument("--perplexity", type=float, default=25.0)
    p.add_argument("--pca-dim", type=int, default=50,
                   help="PCA dims before t-SNE on x; 0 disables")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    arrays, meta = load_demo(args.demo)
    os.makedirs(args.out_dir, exist_ok=True)
    written = []

    written.append(plot_traces(
        arrays, meta, os.path.join(args.out_dir, "demo_traces_full.png"),
        n_show=args.n_show, seed=args.seed,
        title="Bench generator: full traces by class (%s nuisance)"
              % ("with" if meta.get("nuisance") else "no")))
    written.append(plot_traces(
        arrays, meta, os.path.join(args.out_dir, "demo_traces_zoom.png"),
        n_show=args.n_show, t_lo=args.zoom_start,
        t_hi=args.zoom_start + args.zoom_s, seed=args.seed,
        title="Close-up: %.0f s, individual bursts resolved" % args.zoom_s))

    sources = (("phi", "x", "psd") if args.embed == "all"
               else (args.embed,))
    for src in sources:
        written.append(plot_tsne(
            arrays, meta,
            os.path.join(args.out_dir, "demo_tsne_%s.png" % src),
            source=src, perplexity=args.perplexity, pca_dim=args.pca_dim,
            seed=args.seed))

    for w in written:
        print("wrote %s" % w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
