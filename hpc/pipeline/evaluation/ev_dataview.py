#!/usr/bin/env python3
"""Block 23: the data overview -- what the synthetic data set's windows,
statistics and parameters look like, before any network (specs/SPEC.md
block 23; D-130).

    python3 hpc/pipeline/evaluation/ev_dataview.py --config hpc/pipeline/projects/synth/synth.json \\
        --eval-config hpc/pipeline/projects/synth/synth_eval.json [--ncpus 2]

Submitted by the launcher as the job ev_data (submit_eval.sh --what data); it
can also be run by hand. It reads S_sub, the first data_view.sim_shards
shards of the simulated bank -- windows of the prior predictive, each donor's
theta drawn from the box prior -- and every window of the cohorts of
synthetic.real, and computes (`compute`):

    T(x)          the eight fixed window statistics (block 19's fixed_stats)
    m, s, z       S_sub's mean and SD (ddof 1) per statistic; z = (T - m) / s
    PCA           two components fitted on z of S_sub alone; every window's
                  coordinates, the explained variance ratios, the loadings
    quantiles     S_sub's 1, 50 and 99 % per statistic; the cohorts' counts outside
    frames        the display range of each statistic and of the PCA plane:
                  S_sub and the cohorts' uncontaminated windows, widened by PAD
    histograms    per data set, count / (n x bin width) over each frame
    theta         per cohort donor theta* and its condition, theta-dagger,
                  in_box and delta (ev_data.pseudo_true); the conditions' centres
    examples      the windows the figures show, drawn with ev_plan.seed_of(seed,
                  "data_view", ...), and each one's zoom

and writes <out_root>/data/data_view.npz, the figures (ev_data_figures.py)
with FIGURES.md, and last DATA_VIEW.json, which records ev_plan.data_inputs
as read before anything else -- the launcher's test of done.

Pass line: EVAL DATA OK: ...; exit 2 on the arguments, 3 on a refusal.

Pure ASCII, LF only.
"""

import argparse
import datetime
import json
import os
import sys
import time

import _evalpath  # noqa: F401

import numpy as np

import ev_config
import ev_data
import ev_plan
import ev_simulate
from bench_burst_provider import BENCH_AXES

EXIT_ARGS, EXIT_REFUSE = 2, 3
QUANTILES = (0.01, 0.5, 0.99)    # S_sub's band and median (D2)
PAD = 0.02                       # the frames' margin, a fraction of their width [Claude's]
JITTER = 0.28                    # the strips' half-height; rows are 1 apart [Claude's]
NPZ = "data_view.npz"


# ---------------------------------------------------------------- the data

def sim_subset(cfg, k):
    """S_sub: dict(x (n, W) float32, theta (n, 10) float64, donor (n,) int64,
    rows (n,) int64 -- the bank's global row ids --, paths, sidecar) of the
    first k shards of the simulated bank, sorted; RefuseError when the bank
    has fewer."""
    index, paths = ev_data.sim_index(cfg)
    if index.n_shards < int(k):
        raise ev_plan.RefuseError("data_view.sim_shards is %d, the simulated bank has %d shard(s)"
                                  % (int(k), index.n_shards))
    rows = np.concatenate([index.rows_of(j) for j in range(int(k))])
    x = np.concatenate([index.load_x(j) for j in range(int(k))], axis=0)
    return {"x": x, "theta": np.asarray(index.theta[rows], dtype=np.float64),
            "donor": np.asarray(index.donor[rows], dtype=np.int64), "rows": rows,
            "paths": list(paths[:int(k)]), "sidecar": index.sidecar}


def x_label(sidecar):
    """The windows' unit, from a bank's sidecar."""
    if sidecar.get("scale_convention") == "per_unit_mean":
        return "x (spikes per unit per %g-ms bin)" % (1000.0 * float(sidecar["w_size"]))
    return "x (%s, bins of %g s)" % (sidecar.get("scale_convention", "?"), float(sidecar["w_size"]))


# ---------------------------------------------------------------- the quantities

def standardise(t_ref):
    """(m, s): the mean and the SD (ddof 1) of each column of t_ref, float64;
    RefuseError when a column is constant."""
    t = np.asarray(t_ref, dtype=np.float64)
    m, s = t.mean(axis=0), t.std(axis=0, ddof=1)
    if not np.all(s > 0):
        raise ev_plan.RefuseError("statistic(s) constant over S_sub: %s"
                                  % ", ".join(ev_simulate.STAT_NAMES[j]
                                              for j in np.flatnonzero(~(s > 0))))
    return m, s


def pca_fit(z_ref, seed):
    """scikit-learn's PCA with two components fitted on z_ref (n, 8)."""
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=int(seed)).fit(np.asarray(z_ref, dtype=np.float64))


def frame(values, pad=PAD):
    """(lo, hi): the range of `values` widened by `pad` of its width on each
    side (a zero width by pad of |lo|, or of 1 at 0)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = float(v.min()), float(v.max())
    w = hi - lo
    if not w > 0:
        w = abs(lo) if lo != 0 else 1.0
    return lo - pad * w, hi + pad * w


def heights(values, edges, n):
    """count / (n x bin width) per bin: the area over a frame is the fraction
    of the n windows inside it (numpy's histogram; values outside are not
    counted)."""
    counts, _e = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
    return counts / (float(n) * np.diff(edges))


def zoom_start(x, L):
    """The first sample of the L-sample stretch centred on the window's
    maximum, kept inside the window."""
    W = int(np.asarray(x).shape[-1])
    L = min(int(L), W)
    return int(min(max(int(np.argmax(x)) - L // 2, 0), W - L))


def donors_of(coh):
    """(ids, theta_d, cls_d): the cohort's donors, sorted, with the theta and
    the condition every row of the donor carries (asserted)."""
    ids, first, inv = np.unique(coh["donor"], return_index=True, return_inverse=True)
    theta_d, cls_d = coh["theta"][first], coh["cls"][first]
    if not (np.array_equal(coh["theta"], theta_d[inv]) and np.array_equal(coh["cls"], cls_d[inv])):
        raise ev_plan.RefuseError("the rows of a cohort donor carry different theta or conditions")
    return ids, theta_d, cls_d


def cohort_examples(coh, k, seed, name):
    """Row ids: per condition (ascending), k of its donors drawn without
    replacement, sorted, each shown by the first window of its first well."""
    out = []
    for c in sorted(set(coh["cls"].tolist())):
        donors_c = np.unique(coh["donor"][coh["cls"] == c])
        rng = np.random.default_rng(ev_plan.seed_of(seed, "data_view", name, int(c)))
        for d in np.sort(rng.choice(donors_c, size=min(int(k), donors_c.size), replace=False)):
            rows = np.flatnonzero(coh["donor"] == d)
            rows = rows[coh["well"][rows] == coh["well"][rows].min()]
            out.append(int(rows[np.argmin(coh["window_idx"][rows])]))
    return np.asarray(out, dtype=np.int64)


def drawn_rows(candidates, k, seed, *key):
    """k of the candidate row ids drawn without replacement, sorted."""
    candidates = np.asarray(candidates, dtype=np.int64)
    rng = np.random.default_rng(ev_plan.seed_of(seed, "data_view", *key))
    return np.sort(rng.choice(candidates, size=min(int(k), candidates.size), replace=False))


def compute(cfg, ecfg, ncpus=1, log=print):
    """(arrays, info): the data overview's arrays (data_view.npz) and its
    summary (DATA_VIEW.json's body)."""
    dv = ecfg.data_view
    seed = int(ecfg.seed)
    sim = sim_subset(cfg, dv["sim_shards"])
    want = ev_plan.data_bank_paths(cfg, ecfg)["sim"]
    if [os.path.abspath(p) for p in sim["paths"]] != [os.path.abspath(p) for p in want]:
        raise ev_plan.RefuseError("the simulated bank's index reads %s, its glob %s"
                                  % (sim["paths"], want))
    side = sim["sidecar"]
    fs, W = float(side["fs"]), int(side["W"])
    L = int(round(float(dv["zoom_s"]) * fs))
    if L > W:
        raise ev_plan.RefuseError("data_view.zoom_s %g s is longer than a window (%g s)"
                                  % (dv["zoom_s"], W / fs))
    names = [r["name"] for r in cfg.data["synthetic"]["real"]]
    t0 = time.time()
    T_sim = ev_simulate.fixed_stats(sim["x"])
    cohorts = dict((n, ev_data.cohort(cfg, n)) for n in names)
    T_coh = dict((n, ev_simulate.fixed_stats(c["x"])) for n, c in cohorts.items())
    log("EVAL DATA: statistics of %d simulated and %s cohort windows (%.0f s)"
        % (T_sim.shape[0], " + ".join(str(c["x"].shape[0]) for c in cohorts.values()),
           time.time() - t0))
    for n, c in cohorts.items():
        if int(c["sidecar"]["W"]) != W or float(c["sidecar"]["fs"]) != fs:
            raise ev_plan.RefuseError("cohort %s has W %s at %s Hz, the simulated bank %d at %g Hz"
                                      % (n, c["sidecar"]["W"], c["sidecar"]["fs"], W, fs))
    for n, T in [("S_sub", T_sim)] + sorted(T_coh.items()):
        if not np.all(np.isfinite(T)):
            raise ev_plan.RefuseError("%s has windows with non-finite statistics" % n)
    m, s = standardise(T_sim)
    z_sim = (T_sim.astype(np.float64) - m) / s
    pca = pca_fit(z_sim, seed)
    pc_sim = pca.transform(z_sim)
    a = {"stat_names": np.array(ev_simulate.STAT_NAMES), "sim_T": T_sim,
         "sim_pc": pc_sim.astype(np.float32), "sim_rows": sim["rows"], "sim_donor": sim["donor"],
         "m": m, "s": s, "pca_components": pca.components_, "pca_mean": pca.mean_,
         "pca_evr": pca.explained_variance_ratio_,
         "q_sim": np.quantile(T_sim.astype(np.float64), QUANTILES, axis=0),
         "quantiles": np.array(QUANTILES), "fs": np.float64(fs), "W": np.int64(W),
         "w_size": np.float64(side["w_size"]), "zoom_len": np.int64(L),
         "axis_names": np.array([ax[0] for ax in BENCH_AXES]),
         "axis_lo": np.array([ax[1] for ax in BENCH_AXES], dtype=np.float64),
         "axis_hi": np.array([ax[2] for ax in BENCH_AXES], dtype=np.float64),
         "x_label": np.array(x_label(side)), "cohort_names": np.array(names)}
    pc_coh = dict((n, pca.transform((T_coh[n].astype(np.float64) - m) / s)) for n in names)
    # the frames: S_sub and the cohorts' uncontaminated windows
    clean = dict((n, ~cohorts[n]["contaminated"]) for n in names)
    in_T = np.concatenate([T_sim.astype(np.float64)] + [T_coh[n][clean[n]] for n in names])
    in_pc = np.concatenate([pc_sim] + [pc_coh[n][clean[n]] for n in names])
    a["frame_T"] = np.array([frame(in_T[:, j]) for j in range(in_T.shape[1])])
    a["frame_pc"] = np.array([frame(in_pc[:, j]) for j in range(2)])
    edges = np.array([np.linspace(lo, hi, int(dv["bins"]) + 1) for lo, hi in a["frame_T"]])
    a["edges"] = edges
    a["hist_sim"] = np.array([heights(T_sim[:, j], edges[j], T_sim.shape[0])
                              for j in range(edges.shape[0])])
    lab = cohorts[names[0]]["sidecar"]["latent_spec"]
    a["centres"] = np.asarray(lab["class_centres"], dtype=np.float64)
    a["label_idx"] = np.asarray(lab["label_idx"], dtype=np.int64)
    a["free_idx"] = np.asarray(lab["free_idx"], dtype=np.int64)
    info = {"sim": {"n": int(T_sim.shape[0]), "donors": int(np.unique(sim["donor"]).size),
                    "shards": [os.path.basename(p) for p in sim["paths"]]},
            "cohorts": {}, "stat_names": list(ev_simulate.STAT_NAMES), "x_label": x_label(side),
            "pca": {"evr": [float(v) for v in pca.explained_variance_ratio_],
                    "components": pca.components_.tolist()},
            "quantiles": list(QUANTILES), "fs": fs, "W": W, "zoom_len": L}
    for n in names:
        c = cohorts[n]
        if not np.array_equal(np.asarray(c["sidecar"]["latent_spec"]["class_centres"],
                                         dtype=np.float64), a["centres"]):
            raise ev_plan.RefuseError("cohorts %s and %s have different condition centres"
                                      % (names[0], n))
        T, pc, nn = T_coh[n].astype(np.float64), pc_coh[n], c["x"].shape[0]
        lo_f, hi_f = a["frame_T"][:, 0], a["frame_T"][:, 1]
        off_T = (T < lo_f) | (T > hi_f)
        off_pc = ~np.all((pc >= a["frame_pc"][:, 0]) & (pc <= a["frame_pc"][:, 1]), axis=1)
        if np.any(np.any(off_T, axis=1) & ~c["contaminated"]) or np.any(off_pc & ~c["contaminated"]):
            raise AssertionError("an uncontaminated window of %s lies outside its frame" % n)
        ids, theta_d, cls_d = donors_of(c)
        dag, in_box, delta = ev_data.pseudo_true(theta_d, c["sidecar"].get("gap_spec"),
                                                  ev_data.free_axes_of(c["sidecar"]))
        ex = cohort_examples(c, dv["examples_per_condition"], seed, n)
        cx = drawn_rows(np.flatnonzero(c["contaminated"]), dv["examples_per_condition"], seed, n,
                        "contaminated")
        jit = np.random.default_rng(ev_plan.seed_of(seed, "data_view", n, "jitter"))
        jit_d = np.random.default_rng(ev_plan.seed_of(seed, "data_view", n, "jitter_d"))
        a.update({
            n + "_T": T_coh[n], n + "_pc": pc.astype(np.float32), n + "_cls": c["cls"],
            n + "_donor": c["donor"], n + "_well": c["well"], n + "_widx": c["window_idx"],
            n + "_contam": c["contaminated"], n + "_jit": jit.uniform(-JITTER, JITTER, nn),
            n + "_hist": np.array([heights(T[:, j], edges[j], nn) for j in range(edges.shape[0])]),
            n + "_below_q": (T < a["q_sim"][0]).sum(axis=0),
            n + "_above_q": (T > a["q_sim"][-1]).sum(axis=0),
            n + "_below_frame": (T < lo_f).sum(axis=0), n + "_above_frame": (T > hi_f).sum(axis=0),
            n + "_pc_out": off_pc,
            n + "_donors": ids, n + "_theta_d": theta_d, n + "_cls_d": cls_d,
            n + "_theta_dag_d": dag, n + "_in_box_d": in_box, n + "_delta": np.float64(delta),
            n + "_jit_d": jit_d.uniform(-JITTER, JITTER, ids.size),
            n + "_ex_rows": ex, n + "_ex_x": c["x"][ex],
            n + "_ex_z0": np.array([zoom_start(c["x"][r], L) for r in ex], dtype=np.int64),
            n + "_cx_rows": cx, n + "_cx_x": c["x"][cx],
            n + "_cx_z0": np.array([zoom_start(c["x"][r], L) for r in cx], dtype=np.int64)})
        gs = c["sidecar"].get("gap_spec") or {}
        info["cohorts"][n] = {"n": int(nn), "donors": int(ids.size),
                              "contaminated": int(c["contaminated"].sum()),
                              "pi": float(gs.get("pi", 0.0)), "delta": float(delta),
                              "donors_in_box": int(in_box.sum()),
                              "below_q": [int(v) for v in a[n + "_below_q"]],
                              "above_q": [int(v) for v in a[n + "_above_q"]],
                              "beyond_frame_T": int(np.any(off_T, axis=1).sum()),
                              "beyond_frame_pc": int(off_pc.sum())}
    ex = drawn_rows(np.arange(T_sim.shape[0]), dv["sim_examples"], seed, "sim")
    a.update({"sim_ex_rows": ex, "sim_ex_x": sim["x"][ex], "sim_ex_theta": sim["theta"][ex],
              "sim_ex_donor": sim["donor"][ex], "sim_ex_bank_rows": sim["rows"][ex],
              "sim_ex_z0": np.array([zoom_start(sim["x"][r], L) for r in ex], dtype=np.int64)})
    info["cohort_names"] = names
    return a, info


# ---------------------------------------------------------------- the files

def write_npz(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(path + ".tmp", path)


def run(cfg_path, ecfg_path, ncpus=None, log=print):
    """The data overview: the quantities, the figures, then DATA_VIEW.json."""
    import torch
    import ev_data_figures
    t0 = time.time()
    cfg = ev_plan.load_project(cfg_path)
    ecfg = ev_config.load(ecfg_path)
    inputs = ev_plan.data_inputs(cfg, ecfg)          # before anything is read
    ncpus = int(ncpus or ecfg.ncpus)
    torch.set_num_threads(max(1, ncpus))
    arrays, info = compute(cfg, ecfg, ncpus=ncpus, log=log)
    ddir = ev_plan.data_dir(ecfg)
    write_npz(os.path.join(ddir, NPZ), arrays)
    t1 = time.time()
    figs = ev_data_figures.make_all(ddir, ecfg, log=log)
    rec = dict(info, schema="eval_data_view/1", inputs=inputs, config=cfg.path,
               eval_config=ecfg.path, npz=os.path.join(ddir, NPZ),
               figures_index=os.path.join(ddir, "FIGURES.md"), n_figures=len(figs),
               seconds={"quantities": round(t1 - t0, 1), "figures": round(time.time() - t1, 1)},
               finished=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    path = ev_plan.data_view_json(ecfg)
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)
    log("EVAL DATA OK: %d simulated windows, %s cohort windows, %d figure(s) (%.0f s); %s"
        % (info["sim"]["n"], " + ".join(str(info["cohorts"][n]["n"]) for n in info["cohort_names"]),
           len(figs), time.time() - t0, rec["figures_index"]))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the pipeline's project file (<tag>.json)")
    ap.add_argument("--eval-config", required=True, help="the evaluation's file (<tag>_eval.json)")
    ap.add_argument("--ncpus", type=int, default=None, help="default: the eval config's ncpus")
    args = ap.parse_args(argv)
    try:
        run(args.config, args.eval_config, ncpus=args.ncpus, log=lambda t: print(t, flush=True))
    except (ev_plan.RefuseError, ev_config.EvalConfigError) as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
