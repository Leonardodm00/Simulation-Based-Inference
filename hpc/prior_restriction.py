#!/usr/bin/env python3
"""Local coverage of the real data by the simulated cloud, and the theta
region that produces the covering simulations.

WHY THIS AND NOT THE POOLED MMD
-------------------------------
A two-sample test between a PRIOR PREDICTIVE and one world's data is
essentially guaranteed to reject at high power: the prior predictive is
deliberately broader than any single dataset, so the two distributions are
not supposed to be equal. The pooled verdict therefore conflates two very
different situations:

  (a) real data lies OUTSIDE the simulator's support        -> model gap
  (b) real data lies INSIDE the support, in a LOW-DENSITY
      region, while most prior mass sits elsewhere          -> restrict the
                                                               prior

What NPE actually needs is (b)-style coverage: training examples near the
observation. This script measures it directly, per real point:

  d_k(x) = distance from real embedding x to its k-th nearest simulated
           neighbour,

compared against the same quantity computed sim-to-sim (leave-one-out). A
real point whose d_k sits inside the sim-to-sim distribution has simulated
neighbours at typical spacing: covered. A real point beyond the sim-to-sim
p99 has no simulated data at normal density around it: locally uncovered.

It then takes the union of the k_seed nearest simulations to every real
point, collects their theta, and reports the per-axis range: a PROPOSED
TRUNCATED PRIOR BOX in the contract's own coordinates (ln where the axis is
ln). This is the constructive output -- the box is what "restrict the prior"
means operationally, and is the natural initialisation for TSNPE-style
truncation.

CAVEATS PRINTED WITH THE RESULTS
--------------------------------
- Distances live in the frozen encoder's embedding, which is collapsed
  (low effective rank relative to E). Coverage claims are claims about THIS encoder's view.
- A box is the coarsest possible restriction: it cannot represent a curved
  or disconnected seed region, and it over-covers by construction. Fractions
  are reported so the over-coverage is visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


def load_theta(pattern: str):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("no shards match %r" % (pattern,))
    names = None
    blocks = []
    for p in paths:
        sc = os.path.splitext(p)[0] + ".json"
        with open(sc) as fh:
            doc = json.load(fh)
        pn = doc.get("param_names") or doc.get("label_axes", {}).get(
            "param_names")
        cols = ["th_%s" % n for n in pn] if pn else None
        t = pq.read_table(p)
        if cols is None:
            cols = [c for c in t.column_names if c.startswith("th_")]
            pn = [c[3:] for c in cols]
        if names is None:
            names = pn
        elif names != pn:
            raise RuntimeError("shards disagree on param names")
        blocks.append(np.column_stack(
            [np.asarray(t[c].to_numpy(zero_copy_only=False),
                        dtype=np.float64) for c in cols]))
    return np.vstack(blocks), list(names), paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", required=True,
                    help="<stem>_arrays.npz written by gate_run.py")
    ap.add_argument("--sim", required=True,
                    help="the SAME shard glob that gate_run.py used (row "
                         "order must match the arrays)")
    ap.add_argument("--out", required=True, help="output stem")
    ap.add_argument("--space", default="z", choices=["z", "zraw"])
    ap.add_argument("--k_cover", type=int, default=16,
                    help="which nearest neighbour defines local coverage")
    ap.add_argument("--k_seed", type=int, default=25,
                    help="how many nearest sims per real point seed the box")
    ap.add_argument("--n_baseline", type=int, default=4000,
                    help="sim points used for the sim-to-sim baseline")
    ap.add_argument("--box_quantiles", default="0.5,99.5",
                    help="robust per-axis quantiles for the proposed box")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A = np.load(args.arrays, allow_pickle=False)
    zs = np.asarray(A["sim_%s" % args.space], dtype=np.float64)
    zr = np.asarray(A["real_%s" % args.space], dtype=np.float64)
    groups = np.asarray([str(g) for g in A["real_groups"]])
    classes = np.asarray([str(c) for c in A["real_classes"]])

    theta, names, paths = load_theta(args.sim)
    if theta.shape[0] != zs.shape[0]:
        raise RuntimeError(
            "theta rows (%d) != sim embedding rows (%d): the --sim glob "
            "does not match the one gate_run.py used for these arrays"
            % (theta.shape[0], zs.shape[0]))

    print("sim %d x %d   real %d x %d   space=%s"
          % (*zs.shape, *zr.shape, args.space))

    # ---- 1. local coverage ------------------------------------------------
    tree = cKDTree(zs)
    k = int(args.k_cover)
    d_real = tree.query(zr, k=k)[0][:, -1]

    rng = np.random.default_rng(args.seed)
    base_idx = rng.choice(zs.shape[0], size=min(args.n_baseline, zs.shape[0]),
                          replace=False)
    # k+1 then drop the self-match: leave-one-out
    d_sim = tree.query(zs[base_idx], k=k + 1)[0][:, -1]

    pct = np.searchsorted(np.sort(d_sim), d_real) / float(d_sim.size) * 100.0
    p99 = np.percentile(d_sim, 99.0)
    outside = d_real > p99

    print("")
    print("LOCAL COVERAGE (d_%d to nearest sims, vs sim-to-sim baseline)" % k)
    print("  sim-to-sim d_%d: median=%.4g  p90=%.4g  p99=%.4g"
          % (k, np.median(d_sim), np.percentile(d_sim, 90), p99))
    print("  real       d_%d: median=%.4g  p90=%.4g  max=%.4g"
          % (k, np.median(d_real), np.percentile(d_real, 90), d_real.max()))
    print("  real rows beyond sim p99: %d / %d  (%.1f%%)"
          % (int(outside.sum()), outside.size, 100.0 * outside.mean()))
    per_class: Dict[str, Dict] = {}
    for c in sorted(set(classes.tolist())):
        m = classes == c
        per_class[c] = {
            "median_percentile": float(np.median(pct[m])),
            "frac_beyond_p99": float(outside[m].mean()),
        }
        print("  class %s: median percentile %.1f, beyond-p99 %.1f%%"
              % (c, per_class[c]["median_percentile"],
                 100 * per_class[c]["frac_beyond_p99"]))
    worst = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        worst[g] = float(outside[m].mean())
    worst_list = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
    print("  worst cultures by uncovered fraction:",
          ["%s:%.0f%%" % (g, 100 * f) for g, f in worst_list])

    # ---- 2. the seed region in theta space --------------------------------
    # Only COVERED real points contribute. An uncovered point's "nearest"
    # sims are not near -- they are merely the least far -- and including
    # them drags the box toward theta regions that do not actually produce
    # data resembling those observations. The smoke test plants exactly this
    # case: one gross outlier whose neighbours would otherwise triple the
    # box width on the driving axis.
    ks = int(args.k_seed)
    covered = ~outside
    if not covered.any():
        raise RuntimeError("no real point is covered; a seed region would "
                           "be meaningless")
    nn = tree.query(zr[covered], k=ks)[1]
    seed_idx = np.unique(nn.ravel())
    th_seed = theta[seed_idx]
    qlo, qhi = (float(x) for x in args.box_quantiles.split(","))
    lo = np.percentile(th_seed, qlo, axis=0)
    hi = np.percentile(th_seed, qhi, axis=0)

    # prior box measured from the data itself: the drawn range per axis
    plo = theta.min(axis=0)
    phi = theta.max(axis=0)
    width_ratio = np.clip((hi - lo) / np.where(phi > plo, phi - plo, 1.0),
                          0.0, 1.0)
    inside = np.all((theta >= lo) & (theta <= hi), axis=1)

    print("")
    print("SEED REGION: union of %d nearest sims per COVERED real row -> %d distinct "
          "sims (%.1f%% of the cloud)"
          % (ks, seed_idx.size, 100.0 * seed_idx.size / theta.shape[0]))
    print("PROPOSED TRUNCATED PRIOR BOX ([p%g, p%g] per axis, contract "
          "coordinates):" % (qlo, qhi))
    print("  %-14s %10s %10s %10s %10s  %s"
          % ("axis", "prior_lo", "prior_hi", "box_lo", "box_hi", "width"))
    order = np.argsort(width_ratio)
    for j in order:
        print("  %-14s %10.4g %10.4g %10.4g %10.4g  %5.1f%%"
              % (names[j], plo[j], phi[j], lo[j], hi[j],
                 100 * width_ratio[j]))
    print("  box volume fraction of the drawn prior: %.3g"
          % float(np.prod(width_ratio)))
    print("  sims inside the box: %d / %d (%.1f%%)   "
          "(over-coverage of a box relative to the true seed set: %.1fx)"
          % (int(inside.sum()), theta.shape[0], 100.0 * inside.mean(),
             inside.sum() / max(1, seed_idx.size)))

    doc = {
        "space": args.space,
        "k_cover": k, "k_seed": ks,
        "coverage": {
            "sim_d_median": float(np.median(d_sim)),
            "sim_d_p99": float(p99),
            "real_d_median": float(np.median(d_real)),
            "real_frac_beyond_p99": float(outside.mean()),
            "per_class": per_class,
            "per_culture_uncovered_frac": worst,
        },
        "seed_region": {
            "n_seed_sims": int(seed_idx.size),
            "box_quantiles": [qlo, qhi],
            "param_names": names,
            "box_lo": lo.tolist(), "box_hi": hi.tolist(),
            "drawn_prior_lo": plo.tolist(), "drawn_prior_hi": phi.tolist(),
            "width_ratio": width_ratio.tolist(),
            "volume_fraction": float(np.prod(width_ratio)),
            "sims_inside_box_frac": float(inside.mean()),
        },
        "caveats": [
            "Distances are measured in the frozen encoder's embedding, which "
            "is collapsed (low effective rank relative to E); coverage is coverage as THIS "
            "encoder sees it.",
            "A box cannot represent a curved or disconnected seed region "
            "and over-covers by construction; the over-coverage factor is "
            "reported.",
            "Axes are in the contract's coordinates: ln for ln axes.",
        ],
    }
    with open(args.out + "_coverage.json", "w") as fh:
        json.dump(doc, fh, indent=2)
    np.savez_compressed(args.out + "_coverage.npz",
                        d_real=d_real, d_sim_baseline=d_sim,
                        real_percentile=pct, outside=outside,
                        seed_idx=seed_idx,
                        box_lo=lo, box_hi=hi)
    print("")
    print("wrote %s_coverage.json / _coverage.npz" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
