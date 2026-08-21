#!/usr/bin/env python3
"""Figures for the misspecification gate.

Reads only what gate_run.py wrote (<stem>_results.json and <stem>_arrays.npz),
so plots can be restyled or re-cut without re-running the gate.

WHICH FIGURES EXIST, AND WHY THESE ONES
---------------------------------------
GateResult exposes summary scalars (mmd_group, p_group, reject_fraction,
mmd_iid, p_iid) and per-group p-values. It does NOT expose the raw
permutation replicates or the per-window-choice p-value array, so a
null-distribution histogram and a p-across-choices strip cannot be drawn
faithfully. They are deliberately NOT reconstructed here: a reimplementation
of the null could disagree with the gate's own and the figure would look
authoritative while being a different calculation. reject_fraction is the
honest summary of the same information and is plotted instead.

  1 overlap      real vs simulated in the simulated cloud's own PCA plane.
                 Answers "does real sit inside the cloud", which a p-value
                 alone cannot distinguish from "the cloud happens to cover a
                 thin shell real occupies".
  2 marginals    per-coordinate densities, both arms. Localises WHERE in
                 embedding space any mismatch lives -- the actionable part.
  3 verdicts     p_group vs p_iid per space and class, with alpha marked.
                 Shows on THIS data the over-rejection that the i.i.d. null
                 produces, which the suite's G3 demonstrated synthetically.
  4 per_group    per-recording p-values, sorted, with BH and Holm outcomes.
                 Separates "a few outlier cultures" from "systemic".
  5 mde          detection rate vs shift size. Converts a PASS from "we did
                 not reject" into "we could have detected a shift >= X".
  6 spectrum     covariance eigenvalues and effective rank for both arms.
                 The collapse caveat, shown rather than asserted.

USAGE
    python3 gate_plots.py --stem results/gate --outdir results/figs
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

C_SIM = "#4C72B0"
C_REAL = "#C44E52"
C_ALT = "#55A868"
DPI = 150


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", p)
    return p


def fig_overlap(A, res, outdir: str, space: str = "z") -> Optional[str]:
    zs = A.get("sim_%s" % space)
    zr = A.get("real_%s" % space)
    if zs is None or zr is None:
        return None
    zs = np.asarray(zs); zr = np.asarray(zr)
    mu = zs.mean(axis=0)
    # PCA basis from the SIMULATED cloud: the question is whether real lies
    # inside the simulated distribution, so the simulated geometry defines
    # the frame. Using a joint basis would rotate the frame toward whatever
    # real does and hide exactly the discrepancy being looked for.
    U, S, Vt = np.linalg.svd(zs - mu, full_matrices=False)
    P = Vt[:2].T
    ps = (zs - mu) @ P
    pr = (zr - mu) @ P
    cls = A.get("real_classes")
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    n = min(6000, ps.shape[0])
    idx = np.random.default_rng(0).choice(ps.shape[0], n, replace=False)
    ax.scatter(ps[idx, 0], ps[idx, 1], s=6, c=C_SIM, alpha=0.18,
               linewidths=0, label="simulated (n=%d shown of %d)"
               % (n, ps.shape[0]))
    if cls is not None:
        cls = np.asarray([str(c) for c in cls])
        for c, col, mk in zip(sorted(set(cls.tolist())), (C_REAL, C_ALT),
                              ("o", "^")):
            m = cls == c
            ax.scatter(pr[m, 0], pr[m, 1], s=22, c=col, marker=mk,
                       edgecolors="k", linewidths=0.3, alpha=0.85,
                       label="real, condition %s (n=%d)" % (c, int(m.sum())))
    else:
        ax.scatter(pr[:, 0], pr[:, 1], s=22, c=C_REAL, label="real")
    var = (S ** 2) / np.sum(S ** 2)
    ax.set_xlabel("PC1 of simulated cloud (%.1f%% var)" % (100 * var[0]))
    ax.set_ylabel("PC2 of simulated cloud (%.1f%% var)" % (100 * var[1]))
    ax.set_title("Embedding overlap, space = %s" % space)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    return _save(fig, outdir, "fig1_overlap_%s.png" % space)


def fig_marginals(A, outdir: str, space: str = "z") -> Optional[str]:
    zs = A.get("sim_%s" % space)
    zr = A.get("real_%s" % space)
    if zs is None or zr is None:
        return None
    zs = np.asarray(zs); zr = np.asarray(zr)
    E = zs.shape[1]
    ncol = 4
    nrow = int(np.ceil(E / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for j in range(E):
        ax = axes[j]
        lo = min(zs[:, j].min(), zr[:, j].min())
        hi = max(zs[:, j].max(), zr[:, j].max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(zs[:, j], bins=bins, density=True, color=C_SIM, alpha=0.55,
                label="sim" if j == 0 else None)
        ax.hist(zr[:, j], bins=bins, density=True, color=C_REAL, alpha=0.55,
                label="real" if j == 0 else None)
        ax.set_title("dim %d" % j, fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(E, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-coordinate marginals, space = %s" % space, y=1.005)
    fig.tight_layout()
    return _save(fig, outdir, "fig2_marginals_%s.png" % space)


def fig_verdicts(res: Dict, outdir: str) -> Optional[str]:
    gate = res.get("gate", {})
    if not gate:
        return None
    keys = sorted(gate.keys())
    labels = [k.replace("::", "\n") for k in keys]
    pg = [gate[k]["p_group"] for k in keys]
    pi = [gate[k]["p_iid"] for k in keys]
    rf = [gate[k]["reject_fraction"] for k in keys]
    alpha = gate[keys[0]].get("alpha", 0.05)
    x = np.arange(len(keys))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(1.6 * len(keys) + 3, 7.2),
                                   sharex=True)
    ax1.bar(x - 0.2, pg, 0.4, color=C_SIM, label="p_group (the verdict)")
    ax1.bar(x + 0.2, pi, 0.4, color="#999999", label="p_iid (contrast only)")
    ax1.axhline(alpha, color="k", ls="--", lw=1,
                label="alpha = %.2g" % alpha)
    ax1.set_yscale("log")
    ax1.set_ylabel("p-value (log)")
    ax1.set_title("Gate verdicts. Below the dashed line = reject "
                  "(misspecification detected)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2, axis="y")
    ax2.bar(x, rf, 0.5, color=C_ALT)
    ax2.axhline(0.5, color="k", ls=":", lw=1)
    ax2.set_ylabel("reject fraction\n(over window choices)")
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.grid(alpha=0.2, axis="y")
    ax2.set_title("Stability: a value near 0 or 1 is a robust verdict; "
                  "near 0.5 the verdict is a coin flip")
    fig.tight_layout()
    return _save(fig, outdir, "fig3_verdicts.png")


def fig_per_group(A, res: Dict, outdir: str) -> List[str]:
    out = []
    gate = res.get("gate", {})
    for key, rec in sorted(gate.items()):
        pk = "%s::per_group_p" % key
        if pk not in A:
            continue
        p = np.asarray(A[pk], dtype=float)
        names = rec.get("per_group_names") or ["g%d" % i for i in range(p.size)]
        order = np.argsort(p)
        p = p[order]
        names = [str(names[i]) for i in order]
        alpha = rec.get("alpha", 0.05)
        m = p.size
        bh = alpha * (np.arange(1, m + 1) / m)
        holm = alpha / (m - np.arange(m))
        fig, ax = plt.subplots(figsize=(max(6.0, 0.22 * m + 3), 4.6))
        ax.scatter(np.arange(m), p, s=26, c=C_REAL, zorder=3,
                   label="per-recording p")
        ax.plot(np.arange(m), bh, color=C_SIM, lw=1.2,
                label="BH threshold (FDR)")
        ax.plot(np.arange(m), holm, color="#888888", lw=1.2, ls="--",
                label="Holm threshold (FWER)")
        ax.axhline(alpha, color="k", lw=0.8, ls=":", label="alpha")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(m))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_ylabel("p-value (log)")
        ax.set_title("Per-recording p-values, sorted -- %s" % key)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        out.append(_save(fig, outdir,
                         "fig4_per_group_%s.png" % key.replace("::", "_")))
    return out


def fig_mde(A, res: Dict, outdir: str) -> Optional[str]:
    mde = res.get("mde", {})
    if not mde:
        return None
    deltas = (0.02, 0.05, 0.1, 0.2, 0.4)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for space, col in zip(sorted(mde.keys()), (C_SIM, C_ALT)):
        rates = np.asarray(mde[space].get("rates", []), dtype=float).ravel()
        if rates.size == 0:
            continue
        d = np.asarray(deltas[:rates.size], dtype=float)
        ax.plot(d, rates, "o-", color=col, label="space = %s" % space)
        v = mde[space].get("mde")
        if v is not None:
            ax.axvline(float(v), color=col, ls="--", lw=1,
                       label="MDE(%s) = %.3g" % (space, float(v)))
    ax.axhline(0.8, color="k", ls=":", lw=1, label="80% power")
    ax.set_xscale("log")
    ax.set_xlabel("rigid shift magnitude")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Gate power. A PASS only rules out shifts the gate could see")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save(fig, outdir, "fig5_mde.png")


def fig_spectrum(A, res: Dict, outdir: str) -> Optional[str]:
    ss = A.get("sim_spectrum")
    sr = A.get("real_spectrum")
    if ss is None or sr is None:
        return None
    ss = np.asarray(ss, float); sr = np.asarray(sr, float)
    emb = res.get("embedding", {})
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    k = np.arange(1, ss.size + 1)
    ax.semilogy(k, np.clip(ss, 1e-18, None), "o-", color=C_SIM,
                label="simulated (r_eff = %.2f)" % emb.get("sim_r_eff", float("nan")))
    ax.semilogy(k, np.clip(sr, 1e-18, None), "s-", color=C_REAL,
                label="real (r_eff = %.2f)" % emb.get("real_r_eff", float("nan")))
    ax.set_xlabel("component")
    ax.set_ylabel("covariance eigenvalue (log)")
    ax.set_title("Embedding spectrum. A collapsed space weakens a PASS,\n"
                 "but leaves a REJECTION decisive")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    return _save(fig, outdir, "fig6_spectrum.png")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True,
                    help="the --out stem given to gate_run.py")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with open(args.stem + "_results.json") as fh:
        res = json.load(fh)
    A = np.load(args.stem + "_arrays.npz", allow_pickle=False)

    print("figures ->", args.outdir)
    for space in ("z", "zraw"):
        fig_overlap(A, res, args.outdir, space)
        fig_marginals(A, args.outdir, space)
    fig_verdicts(res, args.outdir)
    fig_per_group(A, res, args.outdir)
    fig_mde(A, res, args.outdir)
    fig_spectrum(A, res, args.outdir)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
