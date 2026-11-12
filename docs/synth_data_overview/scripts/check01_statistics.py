#!/usr/bin/env python3
"""Check 01 -- the eight window statistics are what the document says they are.

Claim checked (document sec. 3.4, eqs. 12-19): the data overview's eight
statistics T_1(x), ..., T_8(x) of a window x are the mean, the SD (ddof 1,
plus 1e-8), the maximum, the 0.9 quantile, the fraction of samples above
mean + SD, the lag-1 product mean of the standardised window, the third
moment of the standardised window, and the mean absolute first difference.
The check recomputes them in numpy (float64) from the bank's windows and
compares them with the pipeline's own values (block 19's fixed_stats, torch
float32), read from data_view.npz. It also prints the unit conversion of
the mean: x is spikes per unit per bin, so mean(x) * fs is spikes per unit
per second.

Run (sandbox or davinci; numpy only):

    python3 check01_statistics.py \
        --data-view /davinci-1/home/ldellamea/runs/synth/evaluation/data/data_view.npz \
        --bank-dirs /davinci-1/home/ldellamea/banks/synth_real_R0 \
                    /davinci-1/home/ldellamea/banks/synth_real_R3 \
        --sim-dir /davinci-1/home/ldellamea/banks/synth_sim

Expected: every statistic's largest relative difference below the
tolerance (CONFIG), and the line "CHECK 01 PASS".

Pure ASCII, LF only.
"""

import argparse
import glob
import os
import sys

import numpy as np

CONFIG = {
    "seed": 20261112,          # draws the S_sub windows recomputed
    "n_sim_rows": 200,         # S_sub windows recomputed (of 4,000)
    "sim_shards": 2,           # S_sub = the simulated bank's first 2 shards
    # Tolerance: the pipeline computes in float32 (torch); float32 sums over
    # W = 18,000 samples carry relative errors of order sqrt(W) * 6e-8 ~ 1e-5,
    # and burst_frac can flip a few samples sitting at the threshold
    # (k / W = 5.6e-5 per sample). 1e-3 relative (or 1e-3 absolute near 0)
    # is well above both and well below any definitional difference.
    "rtol": 1e-3,
    "atol": 1e-3,
}

STAT_NAMES = ("mean", "sd", "max", "q90", "burst_frac", "acf1", "skew", "roughness")


def window_stats(x):
    """(n, 8) float64: the eight statistics of each row of x (n, W).

    T1 mean; T2 SD (ddof 1) + 1e-8; T3 max; T4 0.9 quantile (linear
    interpolation); T5 fraction of samples above T1 + T2; with
    xc = (x - T1) / T2: T6 mean of xc[s] * xc[s+1] over the W - 1 pairs,
    T7 mean of xc**3; T8 mean of |x[s+1] - x[s]| over the W - 1 pairs.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=1, keepdims=True) + 1e-8
    xc = (x - mu) / sd
    out = np.column_stack([
        mu[:, 0],
        sd[:, 0],
        x.max(axis=1),
        np.quantile(x, 0.9, axis=1),
        (x > mu + sd).mean(axis=1),
        (xc[:, 1:] * xc[:, :-1]).mean(axis=1),
        (xc ** 3).mean(axis=1),
        np.abs(np.diff(x, axis=1)).mean(axis=1),
    ])
    return out


def load_view(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_shard(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in ("x", "donor", "well", "window_idx")}


def compare(mine, theirs, rtol, atol):
    """Per statistic: (max abs diff, max rel diff, pass)."""
    rows = []
    for j, name in enumerate(STAT_NAMES):
        a, b = mine[:, j], theirs[:, j].astype(np.float64)
        ad = np.abs(a - b)
        rd = ad / np.maximum(np.abs(b), 1e-12)
        ok = bool(np.all(ad <= atol + rtol * np.abs(b)))
        rows.append((name, float(ad.max()), float(np.median(rd)), ok))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-view", required=True)
    p.add_argument("--bank-dirs", nargs="+", required=True,
                   help="the cohort banks, in data_view's cohort order (R0, R3)")
    p.add_argument("--sim-dir", required=True)
    a = p.parse_args(argv)
    cfg = CONFIG
    v = load_view(a.data_view)
    names = [str(n) for n in v["cohort_names"]]
    fs = float(v["fs"])
    all_ok = True
    print("CHECK 01: the eight statistics, numpy float64 against the pipeline's float32")
    for name, bdir in zip(names, a.bank_dirs):
        sh = load_shard(sorted(glob.glob(os.path.join(bdir, "shard_*.npz")))[0])
        # data_view's cohort rows are the bank's rows in order (block 23)
        mine = window_stats(sh["x"])
        rows = compare(mine, v[name + "_T"], cfg["rtol"], cfg["atol"])
        print("  %s (%d windows):" % (name, mine.shape[0]))
        for r in rows:
            print("    %-10s max |diff| %.2e  median rel diff %.1e  %s"
                  % (r[0], r[1], r[2], "ok" if r[3] else "FAIL"))
            all_ok &= r[3]
    shards = sorted(glob.glob(os.path.join(a.sim_dir, "shard_*.npz")))[:cfg["sim_shards"]]
    x = np.concatenate([np.load(s)["x"] for s in shards], axis=0)
    rng = np.random.default_rng(cfg["seed"])
    rows_ = np.sort(rng.choice(x.shape[0], size=cfg["n_sim_rows"], replace=False))
    mine = window_stats(x[rows_])
    rows = compare(mine, v["sim_T"][rows_], cfg["rtol"], cfg["atol"])
    print("  S_sub (%d of %d windows, seed %d):" % (rows_.size, x.shape[0], cfg["seed"]))
    for r in rows:
        print("    %-10s max |diff| %.2e  median rel diff %.1e  %s"
              % (r[0], r[1], r[2], "ok" if r[3] else "FAIL"))
        all_ok &= r[3]
    med = np.median(v["sim_T"][:, 0].astype(np.float64))
    print("  units: S_sub's median window mean %.4f spikes/unit/bin = %.2f spikes/unit/s "
          "(x fs = %g Hz)" % (med, med * fs, fs))
    print("CHECK 01 %s" % ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
