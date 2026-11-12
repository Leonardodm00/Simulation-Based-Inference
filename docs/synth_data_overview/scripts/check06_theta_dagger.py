#!/usr/bin/env python3
"""Check 06 -- theta-dagger, and how many of R3's donors stay inside the box.

Claims checked (document sec. 3.5):
  (a) theta-dagger_d = theta*_d + delta on the free axes 7, 8, 9 (delta =
      pi x shift_max = 0.3 x 0.35 = 0.105), with the duty axis capped at
      (1 - L_7) / (U_7 - L_7) = 1; the label axes unchanged -- recomputed from
      data_view.npz's theta*_d and compared with its theta-dagger_d;
  (b) a donor is inside the open box exactly when theta*_{d,k} < 1 - delta on
      all three free axes, so with uniform free axes P(in box) = (1 - delta)^3
      = 0.7169 and the count over 27 donors is Binomial(27, 0.7169): mean
      19.36, SD 2.34; per axis the donors at or beyond 1 are Binomial(27,
      delta): mean 2.84. Exact (scipy) and Monte Carlo; R3's counts from
      data_view.npz.

Run:

    python3 check06_theta_dagger.py \
        --data-view /davinci-1/home/ldellamea/runs/synth/evaluation/data/data_view.npz

Expected: (a) max |difference| 0; (b) R3's count of donors in the box
inside the binomial's 95 % interval; then "CHECK 06 DONE".

Pure ASCII, LF only.
"""

import argparse
import sys

import numpy as np
from scipy import stats

CONFIG = {
    "seed": 20261114,
    "pi": 0.3,              # R3's severity
    "shift_max": 0.35,      # GapSpec's default, R3's sidecar
    "n_donors": 27,         # 3 conditions x 9 donors (D-120)
    "n_mc": 200000,         # Monte Carlo cohorts
    "duty_axis": 7,
}


def dagger(theta, delta, free_idx, duty_axis, lo, hi):
    t = np.array(theta, dtype=np.float64, copy=True)
    t[:, free_idx] += delta
    t[:, duty_axis] = np.minimum(t[:, duty_axis], (1.0 - lo[duty_axis]) / (hi[duty_axis] - lo[duty_axis]))
    return t, np.all((t > 0.0) & (t < 1.0), axis=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-view", required=True)
    a = p.parse_args(argv)
    cfg = CONFIG
    v = np.load(a.data_view, allow_pickle=False)
    free_idx = list(v["free_idx"])
    lo, hi = v["axis_lo"], v["axis_hi"]
    delta = cfg["pi"] * cfg["shift_max"]
    th = v["R3_theta_d"]
    td, inb = dagger(th, delta, free_idx, cfg["duty_axis"], lo, hi)
    print("CHECK 06 (a): delta %.3f (data_view: %.3f); duty cap %.3f; max |theta-dagger "
          "recomputed - data_view| %.1e; in-box flags equal: %s"
          % (delta, float(v["R3_delta"]), (1 - lo[7]) / (hi[7] - lo[7]),
             np.abs(td - v["R3_theta_dag_d"]).max(), bool(np.array_equal(inb, v["R3_in_box_d"]))))
    p_in = (1 - delta) ** 3
    b = stats.binom(cfg["n_donors"], p_in)
    rng = np.random.default_rng(cfg["seed"])
    u = rng.uniform(size=(cfg["n_mc"], cfg["n_donors"], 3))
    mc = np.all(u + delta < 1.0, axis=2).sum(axis=1)
    print("CHECK 06 (b): P(in box) = (1 - delta)^3 = %.4f; Binomial(%d, %.4f): mean %.2f, "
          "SD %.2f, 95 %% interval [%d, %d]; Monte Carlo (seed %d): mean %.2f, SD %.2f"
          % (p_in, cfg["n_donors"], p_in, b.mean(), b.std(), b.ppf(0.025), b.ppf(0.975),
             cfg["seed"], mc.mean(), mc.std(ddof=1)))
    print("  R3 observed: %d of %d donors in the open box; P(X <= %d) = %.2f"
          % (inb.sum(), inb.size, inb.sum(), b.cdf(inb.sum())))
    for k in free_idx:
        n_out = int(np.sum(td[:, k] >= 1.0))
        print("  axis %d %-19s donors at or beyond 1: %d (Binomial(27, %.3f): mean %.2f)"
              % (k, str(v["axis_names"][k]), n_out, delta, cfg["n_donors"] * delta))
    print("  donors out on two or more axes: %d" % int(np.sum((td[:, free_idx] >= 1.0).sum(axis=1) >= 2)))
    print("CHECK 06 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
