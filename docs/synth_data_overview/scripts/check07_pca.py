#!/usr/bin/env python3
"""Check 07 -- the PCA of D4: why the statistics are standardised, and why
the components are fitted on S_sub alone.

Claims checked (document sec. 3.6):
  (a) data_view.npz's components and explained-variance ratios are
      scikit-learn's PCA(2) fitted on z = (T - m) / s over S_sub, m and s
      S_sub's mean and SD (ddof 1) -- recomputed and compared, up to sign;
  (b) without the standardisation the first component is the statistic of
      largest variance (skew) almost alone: the raw PCA is a PCA of the
      units, not of the windows;
  (c) fitted on S_sub together with R3's windows, the plane turns toward
      R3's 18 contaminated windows (the principal angles between the two
      planes), so the pooled picture would describe the contamination and
      not the reference; fitted on S_sub with R3's uncontaminated windows
      only, it barely moves.
Writes the twin figure ../figures/check07_pca_fit_twin.png: R3 on the plane
fitted on S_sub (block 23's choice) and on the plane fitted on S_sub and R3
pooled.

Run (numpy, scipy, scikit-learn, matplotlib):

    python3 check07_pca.py \
        --data-view /davinci-1/home/ldellamea/runs/synth/evaluation/data/data_view.npz

Expected: (a) differences at float roundoff; (b) the raw PC 1's skew
loading near 1; (c) the angles of the pooled fit far larger than those
of the uncontaminated pooled fit; then "CHECK 07 DONE".

Pure ASCII, LF only.
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "seed": 4242,     # the evaluation's seed (synth_eval.json), PCA's random_state
    "dpi": 110,
}


def fit(Z, seed):
    return PCA(n_components=2, random_state=seed).fit(Z)


def sign_align(C, ref):
    """Flip each row of C to agree in sign with the same row of ref."""
    sgn = np.sign(np.sum(C * ref, axis=1))
    return C * sgn[:, None]


def angles_deg(A, B):
    return np.degrees(subspace_angles(A.T, B.T))


def plot_twin(v, Z_s, Z_r, con, pca_s, pca_p, path, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#888888", "axes.grid": True,
                         "grid.color": "#e6e6e6", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, pca, title in ((axs[0], pca_s, "fitted on S_sub alone (block 23's choice)"),
                           (axs[1], pca_p, "fitted on S_sub and R3 pooled")):
        Ys, Yr = pca.transform(Z_s), pca.transform(Z_r)
        evr = pca.explained_variance_ratio_
        ax.scatter(Ys[:, 0], Ys[:, 1], s=2, color="#bbbbb0", label="S_sub", rasterized=True)
        ax.scatter(Yr[~con, 0], Yr[~con, 1], s=6, color="#3b5b92", label="R3, uncontaminated")
        ax.scatter(Yr[con, 0], Yr[con, 1], s=22, facecolor="none", edgecolor="#b4473f",
                   linewidth=1.0, label="R3, contaminated")
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel("PC 1 (%.0f %% of the fit set's variance)" % (100 * evr[0]))
        ax.set_ylabel("PC 2 (%.0f %%)" % (100 * evr[1]))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, markerscale=1.5)
    fig.suptitle("The same windows on two PCA planes of the standardised statistics "
                 "z = (T - m) / s (m, s from S_sub)", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-view", required=True)
    p.add_argument("--fig", default=os.path.join(HERE, "..", "figures", "check07_pca_fit_twin.png"))
    a = p.parse_args(argv)
    cfg = CONFIG
    v = np.load(a.data_view, allow_pickle=False)
    names = [str(n) for n in v["stat_names"]]
    Ts = v["sim_T"].astype(np.float64)
    m, s = Ts.mean(axis=0), Ts.std(axis=0, ddof=1)
    print("CHECK 07 (a): m, s recomputed: max |diff| %.1e, %.1e"
          % (np.abs(m - v["m"]).max(), np.abs(s - v["s"]).max()))
    Z_s = (Ts - m) / s
    pca_s = fit(Z_s, cfg["seed"])
    C = sign_align(pca_s.components_, v["pca_components"])
    print("  components: max |diff| up to sign %.1e; EVR %s (data_view %s)"
          % (np.abs(C - v["pca_components"]).max(), np.round(pca_s.explained_variance_ratio_, 4),
             np.round(v["pca_evr"], 4)))
    print("  loadings (PC 1 | PC 2): " + "  ".join("%s %+.2f|%+.2f" % (n, C[0, j], C[1, j])
                                                 for j, n in enumerate(names)))
    print("CHECK 07 (b): S_sub's SD per statistic: " + "  ".join("%s %.3g" % (n, x)
                                                                for n, x in zip(names, s)))
    raw = fit(Ts - m, cfg["seed"])
    j1 = int(np.argmax(np.abs(raw.components_[0])))
    print("  raw PCA: EVR %s; PC 1's largest loading %s %+.3f"
          % (np.round(raw.explained_variance_ratio_, 4), names[j1], raw.components_[0, j1]))
    Tr = v["R3_T"].astype(np.float64)
    con = v["R3_contam"].astype(bool)
    Z_r = (Tr - m) / s
    pca_p = fit(np.vstack([Z_s, Z_r]), cfg["seed"])
    pca_u = fit(np.vstack([Z_s, Z_r[~con]]), cfg["seed"])
    print("CHECK 07 (c): principal angles to the S_sub plane (degrees): pooled with all of R3 %s;"
          " pooled with R3's uncontaminated %s"
          % (np.round(angles_deg(pca_p.components_, pca_s.components_), 1),
             np.round(angles_deg(pca_u.components_, pca_s.components_), 2)))
    print("  pooled EVR %s; S_sub's own spread on the pooled plane: SD %s (on its own plane %s)"
          % (np.round(pca_p.explained_variance_ratio_, 3),
             np.round(pca_p.transform(Z_s).std(axis=0, ddof=1), 2),
             np.round(pca_s.transform(Z_s).std(axis=0, ddof=1), 2)))
    Yc = pca_s.transform(Z_r[con])
    print("  R3's contaminated on the S_sub plane: PC 1 %.1f to %.1f, PC 2 %.1f to %.1f "
          "(S_sub: PC 1 %.1f to %.1f, PC 2 %.1f to %.1f)"
          % (Yc[:, 0].min(), Yc[:, 0].max(), Yc[:, 1].min(), Yc[:, 1].max(),
             *np.ravel([[pca_s.transform(Z_s)[:, k].min(), pca_s.transform(Z_s)[:, k].max()]
                        for k in (0, 1)])))
    plot_twin(v, Z_s, Z_r, con, pca_s, pca_p, os.path.normpath(a.fig), cfg["dpi"])
    print("  figure: %s" % os.path.normpath(a.fig))
    print("CHECK 07 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
