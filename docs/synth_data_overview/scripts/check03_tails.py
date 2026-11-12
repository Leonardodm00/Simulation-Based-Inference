#!/usr/bin/env python3
"""Check 03 -- the cohorts against S_sub's 1-99 % band, statistic by statistic.

Claims checked (document sec. 3.4):
  (a) data_view.npz's band q_sim is S_sub's 1, 50 and 99 % quantiles
      (numpy's default linear interpolation);
  (b) the counts below and above the band, split into contaminated and
      uncontaminated windows: the uncontaminated windows of both cohorts
      fall outside the band far less often than the 1 % per tail that a
      cohort exchangeable with S_sub would show;
  (c) the cohorts' uncontaminated windows are more concentrated than
      S_sub's: the ratio of interquartile ranges IQR(cohort) / IQR(S_sub)
      is below 1 on most statistics;
  (d) a histogram's area over its frame is the fraction of its windows
      inside the frame (heights = count / (n x bin width)).

Run:

    python3 check03_tails.py \
        --data-view /davinci-1/home/ldellamea/runs/synth/evaluation/data/data_view.npz

Expected: (a) differences at float64 roundoff; the tables of (b) and (c);
(d) S_sub's areas 1 or just below, R3's 1 - 18/324 = 0.944 where every
contaminated window is beyond the frame; then "CHECK 03 DONE".

Pure ASCII, LF only.
"""

import argparse
import sys

import numpy as np

CONFIG = {
    "quantiles": (0.01, 0.5, 0.99),   # block 23's band and median
}


def tail_counts(T, lo, hi):
    """(below, above) per statistic: strict inequalities, as block 23 counts."""
    return (T < lo[None, :]).sum(axis=0), (T > hi[None, :]).sum(axis=0)


def iqr(T):
    q1, q3 = np.quantile(T, [0.25, 0.75], axis=0)
    return q3 - q1


def areas(hist, edges):
    """Sum of height x width per statistic."""
    return (hist * np.diff(edges, axis=1)).sum(axis=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-view", required=True)
    a = p.parse_args(argv)
    v = np.load(a.data_view, allow_pickle=False)
    names = [str(n) for n in v["stat_names"]]
    Ts = v["sim_T"].astype(np.float64)
    q = np.quantile(Ts, CONFIG["quantiles"], axis=0)
    print("CHECK 03 (a): band recomputed, max |diff| %.1e" % np.abs(q - v["q_sim"]).max())
    lo, hi = v["q_sim"][0], v["q_sim"][-1]
    print("  S_sub 1 % | 50 % | 99 %:")
    for j, n in enumerate(names):
        print("    %-10s %+.5f | %+.5f | %+.5f" % (n, lo[j], v["q_sim"][1][j], hi[j]))
    print("CHECK 03 (b): windows below | above the band (exchangeable with S_sub: "
          "about 1 %% per tail, %.2f of 324)" % (0.01 * 324))
    for c in [str(n) for n in v["cohort_names"]]:
        T = v[c + "_T"].astype(np.float64)
        con = v[c + "_contam"].astype(bool)
        bu, au = tail_counts(T[~con], lo, hi)
        bc, ac = tail_counts(T[con], lo, hi)
        print("  %s: %d uncontaminated, %d contaminated windows" % (c, (~con).sum(), con.sum()))
        for j, n in enumerate(names):
            print("    %-10s uncontaminated %2d | %2d    contaminated %2d | %2d"
                  % (n, bu[j], au[j], bc[j], ac[j]))
        print("    total over the 16 tails: uncontaminated %d (exchangeable: %.1f)"
              % (bu.sum() + au.sum(), 16 * 0.01 * (~con).sum()))
    print("CHECK 03 (c): IQR(cohort, uncontaminated) / IQR(S_sub)")
    iq_s = iqr(Ts)
    for c in [str(n) for n in v["cohort_names"]]:
        T = v[c + "_T"].astype(np.float64)
        con = v[c + "_contam"].astype(bool)
        r = iqr(T[~con]) / iq_s
        print("  %s: " % c + "  ".join("%s %.2f" % (n, x) for n, x in zip(names, r)))
    print("CHECK 03 (d): histogram areas over the frames")
    print("  S_sub: " + "  ".join("%.3f" % x for x in areas(v["hist_sim"], v["edges"])))
    for c in [str(n) for n in v["cohort_names"]]:
        print("  %s:    " % c + "  ".join("%.3f" % x for x in areas(v[c + "_hist"], v["edges"])))
    print("CHECK 03 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
