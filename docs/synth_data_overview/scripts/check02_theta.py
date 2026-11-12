#!/usr/bin/env python3
"""Check 02 -- how theta is drawn in each data set.

Claims checked (document sec. 3.5):
  (a) S_sub's theta is uniform on the unit box (0, 1)^10, one draw per donor,
      the two wells of a donor sharing it exactly;
  (b) a cohort donor's theta* on a label axis k is a normal of SD tau = 0.10
      about its condition's centre mu_{c,k}, truncated to (0, 1) -- so the
      deviations theta*_{d,k} - mu_{c(d),k} have mean ~0 and SD ~tau;
  (c) a cohort donor's theta* on a free axis is uniform on (0, 1);
  (d) the conditions' centres in physical units, and their distances.
Kolmogorov-Smirnov tests (scipy.stats.kstest) for the uniform claims; a
chi-square interval for the SD in (b).

Run:

    python3 check02_theta.py \
        --data-view /davinci-1/home/ldellamea/runs/synth/evaluation/data/data_view.npz \
        --sim-dir /davinci-1/home/ldellamea/banks/synth_sim

Expected: KS p-values not small for (a) and (c) (a p-value below 0.01 on
one axis of ten is still expected about once in ten runs of independent
data), the SD of (b) inside its 95 % interval and close to 0.10, and the
line "CHECK 02 DONE".

Pure ASCII, LF only.
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy import stats

CONFIG = {
    "sim_shards": 2,     # S_sub = the simulated bank's first 2 shards
    "tau_ov": 0.10,      # the cohorts' truncated-normal SD (D-120)
    "level": 0.95,       # the SD's chi-square interval
    "T_win": 180.0,      # s, a window's length (D-120)
}


def load_sim_theta(sim_dir, k):
    """(theta (n_rows, 10) float64, donor (n_rows,)) of the first k shards."""
    shards = sorted(glob.glob(os.path.join(sim_dir, "shard_*.npz")))[:k]
    th, don = [], []
    for s in shards:
        d = np.load(s)
        th.append(d["theta"].astype(np.float64))
        don.append(d["donor"])
    return np.concatenate(th), np.concatenate(don)


def per_donor(theta, donor):
    """(ids, theta_d, shared): one theta per donor; shared is True when every
    row of every donor carries the same theta."""
    ids, first, inv = np.unique(donor, return_index=True, return_inverse=True)
    theta_d = theta[first]
    shared = bool(np.array_equal(theta, theta_d[inv]))
    return ids, theta_d, shared


def ks_uniform(cols):
    """KS statistic and p-value of each column against U(0, 1)."""
    return [stats.kstest(c, "uniform") for c in cols.T]


def sd_interval(dev, level):
    """Sample SD of dev (assumed mean 0, known) and its chi-square interval."""
    n = dev.size
    s = float(np.sqrt(np.mean(dev ** 2)))
    lo = s * np.sqrt(n / stats.chi2.ppf(0.5 + level / 2, n))
    hi = s * np.sqrt(n / stats.chi2.ppf(0.5 - level / 2, n))
    return s, lo, hi, n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-view", required=True)
    p.add_argument("--sim-dir", required=True)
    a = p.parse_args(argv)
    cfg = CONFIG
    v = np.load(a.data_view, allow_pickle=False)
    names = [str(n) for n in v["axis_names"]]
    label_idx, free_idx = list(v["label_idx"]), list(v["free_idx"])
    centres = v["centres"]

    theta, donor = load_sim_theta(a.sim_dir, cfg["sim_shards"])
    ids, theta_d, shared = per_donor(theta, donor)
    print("CHECK 02 (a): S_sub, %d windows, %d donors; every donor's rows share theta: %s"
          % (theta.shape[0], ids.size, shared))
    for k, r in enumerate(ks_uniform(theta_d)):
        print("    axis %d %-19s KS D %.4f  p %.3f" % (k, names[k], r.statistic, r.pvalue))

    for c in [str(n) for n in v["cohort_names"]]:
        th, cl = v[c + "_theta_d"], v[c + "_cls_d"]
        dev = np.concatenate([th[cl == cc][:, label_idx] - centres[cc][None, :]
                              for cc in np.unique(cl)]).ravel()
        s, lo, hi, n = sd_interval(dev, cfg["level"])
        print("CHECK 02 (b): %s, %d donors, label axes: %d deviations from the centres, "
              "mean %+.4f, SD %.4f (%d %% interval %.4f-%.4f; tau %.2f)"
              % (c, th.shape[0], n, dev.mean(), s, int(100 * cfg["level"]), lo, hi,
                 cfg["tau_ov"]))
        print("CHECK 02 (c): %s, free axes:" % c)
        for k, r in zip(free_idx, ks_uniform(th[:, free_idx])):
            print("    axis %d %-19s KS D %.4f  p %.3f" % (k, names[k], r.statistic, r.pvalue))
    lo, hi = v["axis_lo"], v["axis_hi"]
    print("CHECK 02 (d): the conditions' centres in physical units, a_k = L_k + mu_ck (U_k - L_k)"
          " (label axes; free axes uniform over their whole range)")
    for cc in range(centres.shape[0]):
        a_c = lo[label_idx] + centres[cc] * (hi[label_idx] - lo[label_idx])
        print("  condition %d: " % cc + "  ".join("%s %.3g" % (names[k], x)
                                                for k, x in zip(label_idx, a_c)))
        print("    burst onsets per %g-s window at the centre: %.1f" % (cfg["T_win"], a_c[0] * cfg["T_win"]))
    d_c = [float(np.linalg.norm(centres[i] - centres[j]))
           for i in range(len(centres)) for j in range(i + 1, len(centres))]
    print("  pairwise distances between the centres (unit box, label axes): %s"
          % ", ".join("%.4f" % x for x in d_c))
    print("CHECK 02 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
