#!/usr/bin/env python3
"""Check 05 -- the observation layer: the nuisance's size, the gap's drift
(mode c), and the nuisance drift's phase across a trace's windows.

Claims checked (document sec. 3.2, 3.3 and 3.7):
  (a) the nuisance nu of a well, read from S_sub's banks: the gain A has a
      log-SD near sqrt(0.1^2 + 2 x 0.05^2 + (0.5)^2 (0.1^2 + 2 x 0.05^2)) =
      0.137; the offset b and the drift amplitude have SD near
      sqrt(0.02^2 + 2 x 0.01^2) = 0.0245; the electrode dropout leaves all 9
      electrodes in almost every well;
  (b) the gap's drift in R3 has amplitude pi x 0.5 x mean(x) of its trace,
      about 0.15 x 0.04 -- several times smaller than the nuisance's drift
      amplitude -- read by inverting the nuisance on R3's windows with the
      bank's nu (the trace mean is unchanged by the drift: 1080 s are 9 of
      its 120-s periods);
  (c) applied to R0's windows (nuisance inverted, the drift added with a
      uniform phase, nuisance re-applied), mode c moves no statistic by more
      than a few hundredths of S_sub's SD;
  (d) the nuisance drift sin(2 pi t / 600 s) runs on the trace's absolute
      time t, and every S_sub window is a trace's window 0 (t in [0, 180) s),
      while a cohort trace's windows j = 0..5 cover [180 j, 180 (j + 1)) s:
      the window-average of the drift's sine differs by j (closed form and
      numerically).

Run (numpy; imports latent_nuisance from the repository's hpc/joint/stage1
and window_stats from check01):

    python3 check05_drifts.py \
        --data-view /davinci-1/home/ldellamea/runs/synth/evaluation/data/data_view.npz \
        --sim-dir /davinci-1/home/ldellamea/banks/synth_sim \
        --r0-dir /davinci-1/home/ldellamea/banks/synth_real_R0 \
        --r3-dir /davinci-1/home/ldellamea/banks/synth_real_R3

Expected: the tables below and "CHECK 05 DONE".

Pure ASCII, LF only.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "hpc", "joint", "stage1")))
from latent_nuisance import NuisanceSpec, apply_nuisance, invert_nuisance, nuisance_affine  # noqa: E402
from check01_statistics import STAT_NAMES, window_stats  # noqa: E402

CONFIG = {
    "seed": 20261113,
    "sim_shards": 2,
    "pi": 0.3,               # R3's severity
    "drift_amp_max": 0.5,    # GapSpec's default, R3's sidecar
    "gap_period_s": 120.0,   # GapSpec's default, R3's sidecar
}


def load_shards(d, k=None):
    shards = sorted(glob.glob(os.path.join(d, "shard_*.npz")))[:k]
    arr = [np.load(s) for s in shards]
    with open(shards[0][:-4] + ".json") as fh:
        side = json.load(fh)
    keys = ("x", "nu", "window_idx", "well", "contaminated")
    return {key: np.concatenate([a[key] for a in arr]) for key in keys}, side


def nspec_of(side):
    ns = side["nuisance_spec"]
    return NuisanceSpec(scales=ns["scales"], kappa=ns["kappa"], n_electrodes=ns["n_electrodes"],
                        drift_period_s=ns["drift_period_s"], dropout_logit0=ns["dropout_logit0"])


def grid(j, side):
    return j * float(side["T_win"]) + np.arange(int(side["W"]), dtype=np.float64) / float(side["fs"])


def avg_sin_closed(j, T_win, P):
    """(1 / T_win) x integral of sin(2 pi t / P) over [j T_win, (j + 1) T_win)."""
    a, b = 2 * np.pi * j * T_win / P, 2 * np.pi * (j + 1) * T_win / P
    return P / (2 * np.pi * T_win) * (np.cos(a) - np.cos(b))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-view", required=True)
    p.add_argument("--sim-dir", required=True)
    p.add_argument("--r0-dir", required=True)
    p.add_argument("--r3-dir", required=True)
    a = p.parse_args(argv)
    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])
    v = np.load(a.data_view, allow_pickle=False)
    s_j = v["s"]

    sim, side = load_shards(a.sim_dir, cfg["sim_shards"])
    spec = nspec_of(side)
    sc = {k: np.asarray(x) for k, x in spec.scales.items()}
    sd_comp = np.sqrt(sc["batch"] ** 2 + sc["donor"] ** 2 + sc["well"] ** 2)
    A = np.array([nuisance_affine(spec, nu, np.zeros(1))[0] for nu in sim["nu"].astype(np.float64)])
    n_act = np.array([max(1, int(round(spec.n_electrodes * (1 - 1 / (1 + np.exp(
        -(spec.dropout_logit0 + nu[3]))))))) for nu in sim["nu"].astype(np.float64)])
    print("CHECK 05 (a): the nuisance over S_sub's %d wells" % sim["nu"].shape[0])
    print("  configured SD per component (batch + donor + well): " +
          ", ".join("%s %.4f" % (n, s) for n, s in zip(
              ("log_gain", "baseline", "dthr", "dropout_logit", "drift_amp"), sd_comp)))
    print("  log A: SD %.4f (analytic %.4f); A in [%.3f, %.3f], 2.5-97.5 %%: %.3f-%.3f"
          % (np.log(A).std(ddof=1), np.sqrt(sd_comp[0] ** 2 + (spec.kappa * sd_comp[2]) ** 2),
             A.min(), A.max(), *np.quantile(A, [0.025, 0.975])))
    print("  baseline b: SD %.4f; drift_amp: SD %.4f, median |.| %.4f"
          % (sim["nu"][:, 1].std(ddof=1), sim["nu"][:, 4].std(ddof=1), np.median(np.abs(sim["nu"][:, 4]))))
    print("  electrodes left by the dropout: %s" % dict(zip(*np.unique(n_act, return_counts=True))))
    print("  S_sub's window_idx: %s (every window is its trace's window 0)"
          % np.unique(sim["window_idx"]).tolist())

    r3, side3 = load_shards(a.r3_dir)
    x_gap = np.array([invert_nuisance(spec, r3["x"][i].astype(np.float64),
                                      r3["nu"][i].astype(np.float64), grid(r3["window_idx"][i], side3))
                      for i in range(r3["x"].shape[0])])
    amps, clean = [], []
    for w in np.unique(r3["well"]):
        rows = r3["well"] == w
        amps.append(cfg["pi"] * cfg["drift_amp_max"] * x_gap[rows].mean())
        clean.append(not r3["contaminated"][rows].any())
    amps, clean = np.array(amps), np.array(clean)
    dr = np.abs(np.array([r3["nu"][r3["well"] == w][0, 4] for w in np.unique(r3["well"])]))
    print("CHECK 05 (b): R3's gap drift amplitude 0.15 x mean(x) per trace (%d uncontaminated "
          "traces): median %.4f, range %.4f-%.4f; the same wells' nuisance |drift_amp|: median %.4f"
          % (clean.sum(), np.median(amps[clean]), amps[clean].min(), amps[clean].max(), np.median(dr)))

    r0, side0 = load_shards(a.r0_dir)
    before = r0["x"].astype(np.float64)
    after = np.empty_like(before)
    for w in np.unique(r0["well"]):
        rows = np.flatnonzero(r0["well"] == w)
        xs = [invert_nuisance(spec, before[i], r0["nu"][i].astype(np.float64),
                              grid(r0["window_idx"][i], side0)) for i in rows]
        amp = cfg["pi"] * cfg["drift_amp_max"] * float(np.mean(xs))
        ph = rng.uniform(0, 2 * np.pi)
        for i, xg in zip(rows, xs):
            t = grid(r0["window_idx"][i], side0)
            xd = xg + amp * np.sin(2 * np.pi * t / cfg["gap_period_s"] + ph)
            after[i] = apply_nuisance(spec, xd, r0["nu"][i].astype(np.float64), t)
    Tb, Ta = window_stats(before), window_stats(after)
    lo, hi = v["q_sim"][0], v["q_sim"][-1]
    print("CHECK 05 (c): mode c added to R0's 324 windows: |change| / s_j, median and max;"
          " windows moved across a band edge")
    for j, n in enumerate(STAT_NAMES):
        ch = np.abs(Ta[:, j] - Tb[:, j]) / s_j[j]
        crossed = int(np.sum(((Tb[:, j] < lo[j]) != (Ta[:, j] < lo[j])) |
                             ((Tb[:, j] > hi[j]) != (Ta[:, j] > hi[j]))))
        print("  %-10s median %.3f  max %.3f  crossed %d" % (n, np.median(ch), ch.max(), crossed))

    T_win, P = float(side["T_win"]), float(spec.drift_period_s)
    print("CHECK 05 (d): the nuisance drift's window-average sin(2 pi t / %g s) by window index j"
          % P)
    sd_b, sd_d = sd_comp[1], sd_comp[4]
    for j in range(6):
        num = float(np.mean(np.sin(2 * np.pi * grid(j, side) / P)))
        cl = avg_sin_closed(j, T_win, P)
        print("  j %d  t in [%4.0f, %4.0f) s  closed form %+.4f  numeric %+.4f  "
              "SD of the window-mean offset b + d x avg: %.4f"
              % (j, j * T_win, (j + 1) * T_win, cl, num, np.sqrt(sd_b ** 2 + (sd_d * cl) ** 2)))
    print("  S_sub's SD of the mean statistic, s_mean = %.4f" % s_j[0])
    print("CHECK 05 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
