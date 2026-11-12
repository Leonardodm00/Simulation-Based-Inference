#!/usr/bin/env python3
"""Check 04 -- gap mode (d), contamination: how many windows, and what it
does to a window's statistics.

Claims checked (document sec. 3.3 and 3.4):
  (a) at pi = 0.3 and J = 6 windows per trace the builder contaminates
      0 or 1 window per trace, 1 with probability frac * J = 0.3 * 0.2 * 6
      = 0.36, so a cohort of 54 traces has Binomial(54, 0.36) contaminated
      windows: mean 19.44, SD 3.53. Checked exactly (scipy) and by Monte
      Carlo through the repository's own latent_gap.apply_trace_gap;
      R3's observed count and per-trace counts read from data_view.npz.
  (b) the contamination x -> max(s x + n, 0), s ~ U(2, 5), n ~ N(0, sd(x)^2)
      white, applied before the nuisance, lowers acf1 and raises roughness
      so far that a contaminated window leaves S_sub's 1-99 % band on both:
      applied here to each of R0's 324 windows (the nuisance inverted with
      the bank's nu, the formula applied, the nuisance re-applied) and
      compared with R3's 18 contaminated windows.
Writes the twin figure ../figures/check04_contamination_twin.png: one R0
window as built and the same window contaminated by the formula.

Run (numpy, scipy, matplotlib; imports latent_gap and latent_nuisance
from the repository's hpc/joint/stage1 by a path relative to this file):

    python3 check04_contamination.py \
        --data-view /davinci-1/home/ldellamea/runs/synth/evaluation/data/data_view.npz \
        --r0-dir /davinci-1/home/ldellamea/banks/synth_real_R0

Expected: (a) the Monte Carlo mean and SD within a few hundredths of
19.44 and 3.53, R3's count inside the binomial's 95 % interval, every R3
trace with 0 or 1 contaminated window; (b) about 100 % of the contaminated
R0 windows below the band on acf1 and above it on roughness; then
"CHECK 04 DONE".

Pure ASCII, LF only.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "hpc", "joint", "stage1")))
from latent_gap import GapSpec, apply_trace_gap          # noqa: E402
from latent_nuisance import NuisanceSpec, apply_nuisance, invert_nuisance  # noqa: E402
from check01_statistics import STAT_NAMES, window_stats  # noqa: E402

CONFIG = {
    "seed": 20261112,
    "pi": 0.3,                 # R3's severity (D-119)
    "contam_frac_max": 0.2,    # GapSpec's default, R3's sidecar
    "J": 6,                    # windows per trace (D-120)
    "n_traces": 54,            # traces (wells) per cohort (D-120)
    "n_mc": 4000,              # Monte Carlo cohorts
    "fig_row": None,           # R0 row shown in the figure; None: the seeded draw
    "zoom_s": 20.0,            # seconds shown around the maximum
    "dpi": 110,
}


def count_exact(n, p):
    b = stats.binom(n, p)
    return b.mean(), b.std(), b.ppf(0.025), b.ppf(0.975)


def count_mc(cfg, rng):
    """Contaminated windows per cohort of n_traces traces, by apply_trace_gap."""
    spec = GapSpec(modes=("contamination",), pi=cfg["pi"],
                   contam_frac_max=cfg["contam_frac_max"])
    x = np.ones((cfg["J"], 4))
    t = np.zeros((cfg["J"], 4))
    out = np.empty(cfg["n_mc"], dtype=np.int64)
    per_trace_max = 0
    for m in range(cfg["n_mc"]):
        k = 0
        for _ in range(cfg["n_traces"]):
            _x, con = apply_trace_gap(spec, x, t, rng)
            k += int(con.sum())
            per_trace_max = max(per_trace_max, int(con.sum()))
        out[m] = k
    return out, per_trace_max


def contaminate(x, rng):
    """The builder's formula (latent_gap.apply_trace_gap, mode d) on one window."""
    s = rng.uniform(2.0, 5.0)
    n = rng.standard_normal(x.shape[-1]) * np.std(x)
    return np.maximum(x * s + n, 0.0)


def load_r0(r0_dir):
    shard = sorted(glob.glob(os.path.join(r0_dir, "shard_*.npz")))[0]
    d = np.load(shard)
    with open(shard[:-4] + ".json") as fh:
        side = json.load(fh)
    return {k: d[k] for k in ("x", "nu", "window_idx", "donor", "well")}, side


def time_grid(j, T_win, W, fs):
    return j * T_win + np.arange(W, dtype=np.float64) / fs


def contaminate_all(r0, side, rng):
    """(before, after) (n, W): R0's windows and the same windows contaminated
    under the builder's order (nuisance inverted, formula, nuisance re-applied)."""
    ns = side["nuisance_spec"]
    spec = NuisanceSpec(scales=ns["scales"], kappa=ns["kappa"], n_electrodes=ns["n_electrodes"],
                        drift_period_s=ns["drift_period_s"], dropout_logit0=ns["dropout_logit0"])
    T_win, fs, W = float(side["T_win"]), float(side["fs"]), int(side["W"])
    before = r0["x"].astype(np.float64)
    after = np.empty_like(before)
    for i in range(before.shape[0]):
        t = time_grid(int(r0["window_idx"][i]), T_win, W, fs)
        nu = r0["nu"][i].astype(np.float64)
        x_gen = invert_nuisance(spec, before[i], nu, t)
        after[i] = apply_nuisance(spec, contaminate(x_gen, rng), nu, t)
    return before, after


def zoom(x, L):
    W = x.shape[-1]
    return int(min(max(int(np.argmax(x)) - L // 2, 0), W - L))


def plot_twin(before, after, Tb, Ta, fs, cfg, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#888888", "axes.grid": True,
                         "grid.color": "#e6e6e6", "axes.spines.top": False,
                         "axes.spines.right": False})
    L = int(round(cfg["zoom_s"] * fs))
    z0 = zoom(after, L)
    t = np.arange(before.size) / fs
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 4.4), gridspec_kw={"width_ratios": [2.2, 1]},
                            sharey="row")
    ia, ir = STAT_NAMES.index("acf1"), STAT_NAMES.index("roughness")
    for r, (x, T, what) in enumerate(((before, Tb, "as built (R0, pi = 0)"),
                                     (after, Ta, "contaminated (gap mode d)"))):
        ax = axs[r, 0]
        ax.plot(t, x, lw=0.5, color="#3b5b92" if r == 0 else "#b4473f")
        ax.axvspan(t[z0], t[z0 + L - 1], color="#dddddd", zorder=0)
        ax.set_title("%s: acf1 %.4f, roughness %.4f" % (what, T[ia], T[ir]),
                     loc="left", fontsize=8)
        ax.set_ylabel("x (spikes per unit per 10-ms bin)")
        axz = axs[r, 1]
        axz.plot(t[z0:z0 + L], x[z0:z0 + L], lw=0.7, color="#3b5b92" if r == 0 else "#b4473f")
        axz.set_title("20 s around the contaminated maximum", loc="left", fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("t (s, from the window's start)")
    fig.suptitle("%s, twice: as built, and contaminated by x -> max(s x + n, 0), s ~ U(2, 5), "
                 "n white of SD sd(x), before the nuisance" % label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg["dpi"])
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-view", required=True)
    p.add_argument("--r0-dir", required=True)
    p.add_argument("--fig", default=os.path.join(HERE, "..", "figures",
                                                 "check04_contamination_twin.png"))
    a = p.parse_args(argv)
    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])
    v = np.load(a.data_view, allow_pickle=False)

    frac = cfg["pi"] * cfg["contam_frac_max"]
    p1 = frac * cfg["J"] - np.floor(frac * cfg["J"])
    mean, sd, lo, hi = count_exact(cfg["n_traces"], p1)
    print("CHECK 04 (a): frac = %.3f, frac * J = %.2f -> 0 or 1 window per trace, P(1) = %.2f"
          % (frac, frac * cfg["J"], p1))
    print("  exact Binomial(%d, %.2f): mean %.2f, SD %.2f, 95 %% interval [%d, %d]"
          % (cfg["n_traces"], p1, mean, sd, lo, hi))
    mc, ptm = count_mc(cfg, rng)
    print("  Monte Carlo (apply_trace_gap, %d cohorts, seed %d): mean %.2f, SD %.2f, "
          "most per trace %d" % (cfg["n_mc"], cfg["seed"], mc.mean(), mc.std(ddof=1), ptm))
    con, well, widx = v["R3_contam"].astype(bool), v["R3_well"], v["R3_widx"]
    per_trace = np.array([con[well == w].sum() for w in np.unique(well)])
    print("  R3 observed: %d contaminated windows; traces with 0 / 1 / >1: %d / %d / %d; "
          "P(X <= %d) = %.2f; window index of the contaminated: %s"
          % (con.sum(), (per_trace == 0).sum(), (per_trace == 1).sum(), (per_trace > 1).sum(),
             con.sum(), stats.binom(cfg["n_traces"], p1).cdf(con.sum()),
             np.bincount(widx[con], minlength=cfg["J"]).tolist()))

    r0, side = load_r0(a.r0_dir)
    before, after = contaminate_all(r0, side, rng)
    Tb, Ta = window_stats(before), window_stats(after)
    lo_q, hi_q, s = v["q_sim"][0], v["q_sim"][-1], v["s"]
    T3 = v["R3_T"][con].astype(np.float64)
    print("CHECK 04 (b): R0's 324 windows contaminated by the formula, against S_sub's band;"
          " R3's %d contaminated windows beside them" % con.sum())
    print("  %-10s %22s %18s %18s" % ("statistic", "median change (in s_j)", "R0 cont. below|above",
                                       "R3 cont. below|above"))
    for j, n in enumerate(STAT_NAMES):
        ch = np.median((Ta[:, j] - Tb[:, j]) / s[j])
        print("  %-10s %22.1f %10.0f%% |%4.0f%% %10.0f%% |%4.0f%%"
              % (n, ch, 100 * np.mean(Ta[:, j] < lo_q[j]), 100 * np.mean(Ta[:, j] > hi_q[j]),
                 100 * np.mean(T3[:, j] < lo_q[j]), 100 * np.mean(T3[:, j] > hi_q[j])))
    row = cfg["fig_row"]
    if row is None:
        row = int(np.random.default_rng(cfg["seed"] + 1).integers(before.shape[0]))
    label = "R0 donor %d, well %d, window %d" % (r0["donor"][row], r0["well"][row],
                                                 r0["window_idx"][row])
    plot_twin(before[row], after[row], Tb[row], Ta[row], float(side["fs"]), cfg,
              os.path.normpath(a.fig), label)
    print("  figure: %s (%s)" % (os.path.normpath(a.fig), label))
    print("CHECK 04 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
