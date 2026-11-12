#!/usr/bin/env python3
"""Check 08 -- R3's windows rebuilt from their recorded ingredients, and the
range shift's counterfactual.

Claims checked (document sec. 3.2, 3.5 and 3.7):
  (a) the order of operations of fig. 1 is the bank's: each of R3's 54 traces
      is rebuilt from its recorded theta*, realisation seed and nu, through
      the repository's bench provider (with the range shift), its trace-gap
      layer (drift and contamination, on the builder's per-trace generator)
      and its nuisance, and compared with the bank's windows;
  (b) theta-dagger is the point of the box coordinates with R3's physics:
      the unshifted axis map at theta-dagger gives the same physical
      parameters as the shifted map at theta* -- inside the open box, and
      outside it where the map is extended affinely (the simulated arm never
      evaluates it there);
  (c) the range shift's effect on a window's statistics: every trace rebuilt
      a second time WITHOUT the shift, with the same realisation seed, the
      same trace generator (so the same drift phase and the same
      contamination draws) and the same nu; the change of each statistic in
      units of S_sub's SD s_j, for the donors inside and outside the box.

Run (numpy, scipy; imports the repository's hpc/joint/stage1 and the DSN
tree it loads; about 2 minutes):

    python3 check08_rebuild.py \
        --data-view /davinci-1/home/ldellamea/runs/synth/evaluation/data/data_view.npz \
        --r3-dir /davinci-1/home/ldellamea/banks/synth_real_R3

Expected: (a) max |rebuilt - bank| at float32 storage precision (~1e-7
relative), (b) max |difference| 0 or roundoff, (c) the table; then
"CHECK 08 DONE".

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
from bench_burst_provider import BENCH_FREE_IDX, load_bench_provider, phi_to_bench_params  # noqa: E402
from latent_gap import GapSpec, apply_trace_gap, param_overrides  # noqa: E402
from latent_nuisance import NuisanceSpec, apply_nuisance  # noqa: E402
from latent_realisation import realisation_key, realisation_seed, RealisationSpec  # noqa: E402
from latent_sbi_simulator import LatentSBISpec, sample_prior_given_classes  # noqa: E402
from check01_statistics import STAT_NAMES, window_stats  # noqa: E402

CONFIG = {
    "trace_seed_stride": 7919,   # build_latent_bank.py: rng = default_rng(base + 7919 * (i + 1))
    "duration_s": 1080.0,        # used only by the axis-map comparison of (b)
}

PARAM_FIELDS = ("lambda_b", "sigma_d", "ibi_cv", "n_frag_mean", "lambda_burst",
                "p_mean", "d_med", "duty", "kappa", "lambda_bg")


def load_bank(d):
    shard = sorted(glob.glob(os.path.join(d, "shard_*.npz")))[0]
    z = np.load(shard)
    with open(shard[:-4] + ".json") as fh:
        side = json.load(fh)
    return {k: z[k] for k in z.files}, side


def build_trace(provider, theta, seed, J, W, fs, N, gspec, nspec, nu, t_all, rng, shift_on):
    ov = param_overrides(gspec) if shift_on else None
    x = np.asarray(provider(theta, J, W, fs, N, seed, param_overrides=ov or None), dtype=np.float64)
    x, con = apply_trace_gap(gspec, x, t_all, rng)
    for j in range(J):
        x[j] = apply_nuisance(nspec, x[j], nu, t_all[j])
    return x, con


def redraw_theta(side, n_classes, per_class):
    """theta*_d in float64, re-drawn as build_latent_bank.py draws it (its
    generator seeded with the shard's base seed: the class permutation, then
    sample_prior_given_classes); the bank stores only its float32 copy."""
    lat = side["latent_spec"]
    spec = LatentSBISpec(n_latent=int(lat["n_latent"]), label_idx=tuple(lat["label_idx"]),
                         class_centres=np.asarray(lat["class_centres"], dtype=np.float64),
                         tau_ov=float(lat["tau_ov"]), n_windows_per_trace=int(lat["n_windows_per_trace"]),
                         T_win=float(lat["T_win"]), fs=float(lat["fs"]),
                         n_neurons=int(lat["n_neurons"]), seed=int(lat["seed"]))
    rng = np.random.default_rng(int(side["provenance"]["base_seed"]))
    cls = rng.permutation(np.repeat(np.arange(n_classes), per_class))
    return cls, sample_prior_given_classes(spec, cls, rng)


def params_vec(p):
    return np.array([getattr(p, f) for f in PARAM_FIELDS], dtype=np.float64)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-view", required=True)
    p.add_argument("--r3-dir", required=True)
    a = p.parse_args(argv)
    cfg = CONFIG
    v = np.load(a.data_view, allow_pickle=False)
    s_j, lo, hi = v["s"], v["q_sim"][0], v["q_sim"][-1]
    bank, side = load_bank(a.r3_dir)
    lat = side["latent_spec"]
    J, W, fs, N = int(lat["n_windows_per_trace"]), int(side["W"]), float(side["fs"]), int(lat["n_neurons"])
    T_win = float(side["T_win"])
    base = int(side["provenance"]["base_seed"])
    gs = side["gap_spec"]
    gspec = GapSpec(modes=tuple(gs["modes"]), pi=gs["pi"], shift_max=gs["shift_max"],
                    drift_amp_max=gs["drift_amp_max"], drift_period_s=gs["drift_period_s"],
                    contam_frac_max=gs["contam_frac_max"])
    ns = side["nuisance_spec"]
    nspec = NuisanceSpec(scales=ns["scales"], kappa=ns["kappa"], n_electrodes=ns["n_electrodes"],
                         drift_period_s=ns["drift_period_s"], dropout_logit0=ns["dropout_logit0"])
    rspec = RealisationSpec(n_per_theta=side["realisation_spec"]["n_per_theta"])
    provider = load_bench_provider()
    t_all = np.stack([j * T_win + np.arange(W, dtype=np.float64) / fs for j in range(J)])

    n_cls = len(lat["class_centres"])
    cls64, th64 = redraw_theta(side, n_cls, int(lat["donors_per_class"]))
    ids_b = np.unique(bank["donor"])
    first = np.array([np.flatnonzero(bank["donor"] == d)[0] for d in ids_b])
    redrawn = bool(np.array_equal(th64.astype(np.float32), bank["theta"][first]) and
                   np.array_equal(cls64, bank["cls"][first]))
    print("CHECK 08 (a0): theta* re-drawn in float64 from the shard's base seed; its float32 copy "
          "equals the bank's theta and classes: %s" % redrawn)
    wells = np.unique(bank["well"])
    worst, worst_rel, seeds_ok = 0.0, 0.0, True
    T_on, T_off, con_all, don_all = [], [], [], []
    for w in wells:
        rows = np.flatnonzero(bank["well"] == w)
        rows = rows[np.argsort(bank["window_idx"][rows])]
        i = int(w)                                   # shard 0: the well id is the trace index
        d = int(bank["donor"][rows[0]])
        theta = th64[d] if redrawn else bank["theta"][rows[0]].astype(np.float64)
        seed = int(bank["realisation_id"][rows[0]])
        seeds_ok &= seed == realisation_seed(base, realisation_key(rspec, d, i, 0))
        nu = bank["nu"][rows[0]].astype(np.float64)
        x_on, con = build_trace(provider, theta, seed, J, W, fs, N, gspec, nspec, nu, t_all,
                                np.random.default_rng(base + cfg["trace_seed_stride"] * (i + 1)), True)
        x_off, _ = build_trace(provider, theta, seed, J, W, fs, N, gspec, nspec, nu, t_all,
                               np.random.default_rng(base + cfg["trace_seed_stride"] * (i + 1)), False)
        ref = bank["x"][rows].astype(np.float64)
        diff = np.abs(x_on - ref)
        worst = max(worst, float(diff.max()))
        worst_rel = max(worst_rel, float((diff / np.maximum(np.abs(ref), 1e-3)).max()))
        if not np.array_equal(con, bank["contaminated"][rows]):
            seeds_ok = False
        T_on.append(window_stats(x_on))
        T_off.append(window_stats(x_off))
        con_all.append(con)
        don_all.append(np.full(J, d))
    T_on, T_off = np.vstack(T_on), np.vstack(T_off)
    con_all, don_all = np.concatenate(con_all), np.concatenate(don_all)
    print("CHECK 08 (a): %d traces rebuilt; max |rebuilt - bank| %.2e (largest relative %.1e); "
          "realisation seeds and contamination flags reproduced: %s"
          % (wells.size, worst, worst_rel, seeds_ok))

    delta = float(v["R3_delta"])
    th_d, dag_d, inb_d, ids = v["R3_theta_d"], v["R3_theta_dag_d"], v["R3_in_box_d"], v["R3_donors"]
    md = {True: 0.0, False: 0.0}
    for k in range(th_d.shape[0]):
        p_shift = params_vec(phi_to_bench_params(th_d[k], N, cfg["duration_s"], free_axis_shift=delta))
        p_box = params_vec(phi_to_bench_params(dag_d[k], N, cfg["duration_s"], free_axis_shift=0.0))
        md[bool(inb_d[k])] = max(md[bool(inb_d[k])], float(np.max(
            np.abs(p_shift - p_box) / np.maximum(np.abs(p_box), 1e-12))))
    print("CHECK 08 (b): physical parameters a(theta-dagger, unshifted map) vs a(theta*, shifted "
          "map), max relative difference: %d donors in the box %.1e; %d outside (the unshifted "
          "map extended affinely beyond the box) %.1e"
          % (inb_d.sum(), md[True], (~inb_d).sum(), md[False]))

    inbox_rows = np.isin(don_all, ids[inb_d])
    keep = ~con_all
    print("CHECK 08 (c): the range shift's effect, |T(shifted) - T(unshifted)| / s_j, "
          "uncontaminated windows (median / 95th pct / max); windows crossing a band edge")
    for j, n in enumerate(STAT_NAMES):
        ch = np.abs(T_on[:, j] - T_off[:, j]) / s_j[j]
        cross = int(np.sum(keep & (((T_on[:, j] < lo[j]) != (T_off[:, j] < lo[j])) |
                                   ((T_on[:, j] > hi[j]) != (T_off[:, j] > hi[j])))))
        a_in, a_out = ch[keep & inbox_rows], ch[keep & ~inbox_rows]
        print("  %-10s in box (%3d) %.3f / %.3f / %.3f   out of box (%3d) %.3f / %.3f / %.3f   "
              "crossed %d" % (n, a_in.size, np.median(a_in), np.quantile(a_in, 0.95), a_in.max(),
                              a_out.size, np.median(a_out), np.quantile(a_out, 0.95), a_out.max(),
                              cross))
    print("CHECK 08 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
