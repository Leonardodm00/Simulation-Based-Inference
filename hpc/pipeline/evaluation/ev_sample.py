#!/usr/bin/env python3
"""Block 19: one ensemble's posterior draws, densities, embeddings and
posterior-predictive windows, per data set (specs/SPEC.md sec. 2.5 and block
19; D-126, D-127).

    python3 hpc/pipeline/evaluation/ev_sample.py --config hpc/pipeline/projects/synth/synth.json \
        --eval-config hpc/pipeline/projects/synth/synth_eval.json --ensemble S-A2-R0:A1 [--ncpus 8]

Writes under <out_root>/samples/ (or --out):
    <stem>_<D>.npz     per data set D: sim (arm S's report split), each of
                       the ensemble's cohorts (R0, R3), and <cohort>dag (sec.
                       2.5's D_R3+: R3 against theta-dagger) for a cohort with
                       a range shift
    <stem>_cal.npz     L-C2ST's calibration: the calibration bank's theta and
                       statistics, one mixture draw per window
    <stem>_sample.json the provenance: members, checkpoints' sha256, seeds,
                       the rebuild oracle, the timings
Pass line: EVAL SAMPLE OK: <stem> ...; exit 2 on the arguments, 3 on a
refusal (a member that does not reproduce its own per-row NLL, a bank of
another split, a calibration bank that is not independent).

The mixture (E1): q_E(theta | x) = (1/M) sum_m q_m(theta | x); its log density
at every draw needs every member at every draw. A theta^ref outside the open
box gets log q = -inf by definition: the flow is never read there (it returns
finite values, its transform clamping).

Seeds are common to the ensembles (common random numbers): the draws of
member k on data set D start from seed_of(seed, D) + k, the posterior-
predictive windows of D from seed_of(seed, D, "ppc") and the calibration
draws from seed_of(seed, "cal"), whatever the ensemble, so two arms' paired
differences are not widened by independent Monte Carlo noise.

Pure ASCII, LF only.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import _evalpath  # noqa: F401

import numpy as np
import torch
from scipy.special import logsumexp

import ev_config
import ev_data
import ev_models
import ev_plan
import ev_simulate

import run_joint_arms as R
from latent_bank import ShardIndex

EXIT_ARGS, EXIT_REFUSE = 2, 3
ORACLE_TOL = 1e-4          # nats: float32 batches of another size (the runner's are 256 rows)


def seed_of(base, *parts):
    """A seed for (base, parts...), stable across machines and runs."""
    h = hashlib.sha256(("%d|" % int(base) + "|".join(str(p) for p in parts)).encode("ascii"))
    return int(h.hexdigest()[:8], 16)


def _batches(n, size):
    for i in range(0, n, int(size)):
        yield slice(i, min(n, i + int(size)))


# ---------------------------------------------------------------- per member

def logq_at(model, theta, x, in_box, batch):
    """(n,) float64: log q_m(theta_i | x_i) for in-box rows, -inf elsewhere."""
    out = np.full(theta.shape[0], -np.inf, dtype=np.float64)
    idx = np.flatnonzero(in_box)
    with torch.no_grad():
        for sl in _batches(idx.size, batch):
            r = idx[sl]
            th = torch.as_tensor(theta[r], dtype=torch.float32)
            out[r] = -model.npe_loss(th, torch.as_tensor(x[r])).numpy().astype(np.float64)
    return out


def draws_of(model, x, n_draws, seed, batch):
    """(n, n_draws, d) float32 posterior draws of one member."""
    torch.manual_seed(int(seed))
    out = []
    with torch.no_grad():
        for sl in _batches(x.shape[0], batch):
            out.append(model.sample_posterior(int(n_draws), torch.as_tensor(x[sl])).numpy())
    return np.concatenate(out).astype(np.float32)


def logq_draws_of(model, draws, x, batch):
    """(n, S) float64: log q_m at every draw of every row (draws (n, S, d)).

    Each window is embedded once and the flow's distribution, conditioned
    on the B embeddings, is evaluated at the (S, B, d) draws by broadcasting.
    sbi 0.27's `ZukoFlow.log_prob` would give the same numbers but expands
    the condition to (S, B) before its embedding net, embedding every window
    S times. The same reach into `_embedding_net` and `net` as
    `JointDSNNPE.rsample_posterior`; smoke test E19c checks the equality
    against `estimator.log_prob`."""
    est = model.estimator
    out = []
    with torch.no_grad():
        for sl in _batches(x.shape[0], batch):
            th = torch.as_tensor(np.ascontiguousarray(draws[sl].transpose(1, 0, 2)))
            dist = est.net(est._embedding_net(torch.as_tensor(x[sl])))             # batch (B,)
            out.append(dist.log_prob(th).numpy().T.astype(np.float64))              # (B, S)
    return np.concatenate(out)


def embed(model, x, batch):
    out = []
    with torch.no_grad():
        for sl in _batches(x.shape[0], batch):
            out.append(model.encode(torch.as_tensor(x[sl])).numpy())
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- per data set

def sample_set(models, x, theta_ref, in_box, n_post, seed, batch, log):
    """The arrays of one data set (block 19's npz contract, without the ids,
    the statistics and the PPC)."""
    M = len(models)
    per = n_post // M
    n = x.shape[0]
    lq_ref = np.stack([logq_at(m, theta_ref, x, in_box, batch) for m in models], axis=1)
    draws = np.concatenate([draws_of(m, x, per, seed + k, batch) for k, m in enumerate(models)],
                           axis=1)
    draw_member = np.repeat(np.arange(M), per)
    lq_all = np.empty((n, n_post, M), dtype=np.float64)
    for k, m in enumerate(models):
        lq_all[:, :, k] = logq_draws_of(m, draws, x, batch)
    log("    densities of %d x %d draws under %d member(s)" % (n, n_post, M))
    lq_mix = logsumexp(lq_all, axis=2) - math.log(M)
    lq_own = lq_all[:, np.arange(n_post), draw_member]
    z = np.stack([embed(m, x, batch) for m in models])
    with np.errstate(divide="ignore"):
        lq_ref_mix = np.where(in_box, logsumexp(lq_ref, axis=1) - math.log(M), -np.inf)
    return {"logq_ref_members": lq_ref, "logq_ref": lq_ref_mix, "draws": draws,
            "draw_member": draw_member.astype(np.int64),
            "logq_draws": lq_mix.astype(np.float32), "logq_draws_own": lq_own.astype(np.float32),
            "z": z}


def ppc_set(draws, ppc_rows, ppc_draws, latent, seed, ncpus):
    """(ppc_draw_idx, ppc_stats (k, ppc_draws, 8)): the statistics of one
    simulated window per chosen draw of each PPC row (Q-E4)."""
    n_post = draws.shape[1]
    idx = (np.arange(ppc_draws) * (n_post // ppc_draws)).astype(np.int64)
    if ppc_rows.size == 0:
        return idx, np.zeros((0, ppc_draws, len(ev_simulate.STAT_NAMES)), dtype=np.float32)
    th = draws[ppc_rows][:, idx].reshape(-1, draws.shape[2]).astype(np.float64)
    xs = ev_simulate.windows_at(th, latent, seed, ncpus=ncpus)
    st = ev_simulate.fixed_stats(xs).reshape(ppc_rows.size, ppc_draws, -1)
    return idx, st


def write_npz(path, arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------- calibration

def calibration_bank(ecfg, sim_side, coh_sides):
    """The calibration bank's ShardIndex (Q-E5), checked before any sampling:
    arm S's box prior, a builder seed none of the banks has, `cal_n` windows
    of as many donors (one window per theta). RefuseError otherwise."""
    paths = sorted(glob.glob(ecfg.cal_glob))
    if not paths:
        raise ev_plan.RefuseError("no calibration bank at %s (block 22's eval_calbank job)"
                                  % ecfg.cal_glob)
    idx = ShardIndex(paths)
    lat = idx.sidecar.get("latent_spec") or {}
    if lat.get("kind") != "box":
        raise ev_plan.RefuseError("the calibration bank is not on the box prior (%s)" % paths[0])
    used = [int((sim_side.get("latent_spec") or {}).get("seed", -1))]
    used += [int((s.get("latent_spec") or {}).get("seed", -2)) for s in coh_sides]
    if int(lat.get("seed", -3)) in used:
        raise ev_plan.RefuseError("the calibration bank's seed %s is a bank's seed %s: it "
                                  "would repeat their nuisance draws (F-aw)"
                                  % (lat.get("seed"), used))
    if idx.n_rows != ecfg.cal_n or np.unique(idx.donor).size != idx.n_rows:
        raise ev_plan.RefuseError("the calibration bank has %d rows of %d donors; cal_n is %d, "
                                  "one window per theta" % (idx.n_rows, np.unique(idx.donor).size,
                                                            ecfg.cal_n))
    return idx


def calibration(models, idx, batch_rows, seed, log):
    """L-C2ST's calibration set (Q-E5): the calibration bank's windows, one
    mixture draw each (a member uniformly at random, then its draw), and the
    windows' statistics."""
    M = len(models)
    rng = np.random.default_rng(seed)
    member = rng.integers(0, M, size=idx.n_rows)
    draws = np.empty((idx.n_rows, idx.theta.shape[1]), dtype=np.float32)
    stats = np.empty((idx.n_rows, len(ev_simulate.STAT_NAMES)), dtype=np.float32)
    mask = np.ones(idx.n_rows, dtype=bool)
    torch.manual_seed(seed)
    with torch.no_grad():
        for rows, x in idx.iter_batches(mask, batch_rows):
            xx = torch.as_tensor(np.asarray(x, dtype=np.float32))
            stats[rows] = ev_simulate.fixed_stats(x)
            for k in range(M):
                sel = np.flatnonzero(member[rows] == k)
                if sel.size:
                    d = models[k].sample_posterior(1, xx[sel]).numpy()[:, 0, :]
                    draws[rows[sel]] = d
    log("    calibration: %d windows, one draw each" % idx.n_rows)
    return {"theta": np.asarray(idx.theta, dtype=np.float64), "stats": stats, "draws": draws,
            "member": member.astype(np.int64)}


# ---------------------------------------------------------------- the job

def run(cfg_path, ecfg_path, name, out=None, ncpus=None, log=print):
    t0 = time.time()
    cfg = ev_plan.load_project(cfg_path)
    ecfg = ev_config.load(ecfg_path)
    e = ev_plan.find(cfg, name)
    out = out or ecfg.sub("samples")
    os.makedirs(out, exist_ok=True)
    ncpus = int(ncpus or ecfg.ncpus)
    log("ensemble %s: %d member(s) in %s; cohorts %s" % (e.name, len(e.records), e.folder,
                                                       ", ".join(e.cohorts)))
    index, sim_paths = ev_data.sim_index(cfg)
    coh_sides = [ShardIndex(ev_data.cohort_paths(cfg, c)).sidecar for c in e.cohorts]
    cal_idx = calibration_bank(ecfg, index.sidecar, coh_sides)
    models, infos = [], []
    for r in e.records:
        m, info = ev_models.rebuild(r, index)
        models.append(m)
        infos.append(info)
        log("  member seed %d rebuilt (%s backbone, E %d)" % (info["seed"], info["backbone"],
                                                             info["embedding_size"]))
    M = len(models)
    if ecfg.n_post % M:
        raise ev_plan.RefuseError("n_post %d is not a multiple of the %d members"
                                  % (ecfg.n_post, M))
    hashes = set(i["split_hash"] for i in infos)
    if len(hashes) != 1:
        raise ev_plan.RefuseError("the members were trained on different splits")
    with open(e.records[0]) as fh:
        rec0 = json.load(fh)
    torch.set_num_threads(max(1, ncpus))

    oracle = {}
    seeds = {}
    written = []
    sim_side = index.sidecar
    latent = sim_side["latent_spec"]
    batch = ecfg.batch_rows

    def finish(ds, base, extra):
        arrays = dict(base)
        arrays.update(extra)
        p = os.path.join(out, "%s_%s.npz" % (e.stem, ds))
        write_npz(p, arrays)
        written.append(p)
        log("  %s: %d row(s) -> %s" % (ds, arrays["theta_ref"].shape[0], p))

    # ---- D_sim
    rows = ev_data.report_rows(index, rec0["split_fracs"], rec0["split_seed"], rec0["split_hash"])
    x = R._rows_x(index, rows).numpy()
    th = np.asarray(index.theta, dtype=np.float64)[rows]
    in_box = np.all((th > 0.0) & (th < 1.0), axis=1)
    donor = np.asarray(index.donor, dtype=np.int64)[rows]
    calib = ev_data.calibration_rows(donor)
    ppc_rows = np.flatnonzero(calib)[:ecfg.ppc_rows]
    seeds["sim"] = seed_of(ecfg.seed, "sim")
    log("  sim: %d report row(s) of %d donor(s)" % (rows.size, int(calib.sum())))
    s = sample_set(models, x, th, in_box, ecfg.n_post, seeds["sim"], batch, log)
    for k, info in enumerate(infos):
        pr = np.load(info["record"][:-len(".json")] + "_perrow.npz")
        diff = float(np.max(np.abs(pr["nll"] + s["logq_ref_members"][:, k])))
        oracle["sim/seed%d" % info["seed"]] = diff
        if not diff <= ORACLE_TOL:
            raise ev_plan.RefuseError("member seed %d does not reproduce its report-split NLL: "
                                      "largest difference %.3g nats (%s)"
                                      % (info["seed"], diff, info["record"]))
    pidx, pst = ppc_set(s["draws"], ppc_rows, ecfg.ppc_draws, latent,
                        seed_of(ecfg.seed, "sim", "ppc"), ncpus)
    finish("sim", s, {"theta_ref": th, "in_box": in_box, "rows": rows.astype(np.int64),
                      "donor": donor, "well": np.asarray(index.well, dtype=np.int64)[rows],
                      "cls": np.asarray(index.cls, dtype=np.int64)[rows],
                      "stats": ev_simulate.fixed_stats(x), "calib": calib,
                      "ppc_rows": ppc_rows.astype(np.int64), "ppc_draw_idx": pidx,
                      "ppc_stats": pst})
    del x

    # ---- the cohorts, and theta-dagger
    for cname in e.cohorts:
        coh = ev_data.cohort(cfg, cname)
        n = coh["x"].shape[0]
        th = coh["theta"]
        in_box = np.all((th > 0.0) & (th < 1.0), axis=1)
        seeds[cname] = seed_of(ecfg.seed, cname)
        s = sample_set(models, coh["x"], th, in_box, ecfg.n_post, seeds[cname], batch, log)
        if cname == e.search["real"]:
            for k, info in enumerate(infos):
                pr = np.load(info["record"][:-len(".json")] + "_perrow.npz")
                if "nll_pseudo_real" not in pr.files:
                    continue
                diff = float(np.max(np.abs(pr["nll_pseudo_real"] + s["logq_ref_members"][:, k])))
                oracle["%s/seed%d" % (cname, info["seed"])] = diff
                if not diff <= ORACLE_TOL:
                    raise ev_plan.RefuseError(
                        "member seed %d does not reproduce its NLL on %s: largest difference "
                        "%.3g nats" % (info["seed"], cname, diff))
        stats = ev_simulate.fixed_stats(coh["x"])
        allrows = np.arange(n, dtype=np.int64)
        pidx, pst = ppc_set(s["draws"], allrows, ecfg.ppc_draws, latent,
                            seed_of(ecfg.seed, cname, "ppc"), ncpus)
        ids = {"rows": allrows, "donor": coh["donor"], "well": coh["well"], "cls": coh["cls"],
               "stats": stats, "calib": np.ones(n, dtype=bool)}
        finish(cname, s, dict(ids, theta_ref=th, in_box=in_box, ppc_rows=allrows,
                              ppc_draw_idx=pidx, ppc_stats=pst))
        dag, dag_in, delta = ev_data.pseudo_true(th, coh["sidecar"].get("gap_spec"))
        if delta > 0.0:
            lq = np.stack([logq_at(m, dag, coh["x"], dag_in, batch) for m in models], axis=1)
            with np.errstate(divide="ignore"):
                mix = np.where(dag_in, logsumexp(lq, axis=1) - math.log(M), -np.inf)
            s2 = dict(s, logq_ref_members=lq, logq_ref=mix)
            finish(cname + "dag", s2, dict(ids, theta_ref=dag, in_box=dag_in,
                                         ppc_rows=np.zeros(0, dtype=np.int64), ppc_draw_idx=pidx,
                                         ppc_stats=np.zeros((0,) + pst.shape[1:], np.float32)))
            log("  %sdag: delta %.4f, %d of %d donor(s) with theta-dagger in the open box"
                % (cname, delta, np.unique(coh["donor"][dag_in]).size,
                   np.unique(coh["donor"]).size))

    # ---- L-C2ST's calibration
    seeds["cal"] = seed_of(ecfg.seed, "cal")
    cal = calibration(models, cal_idx, batch, seeds["cal"], log)
    p = os.path.join(out, "%s_cal.npz" % e.stem)
    write_npz(p, cal)
    written.append(p)

    prov = {"schema": "eval_sample/1", "ensemble": e.name, "stem": e.stem, "kind": e.kind,
            "folder": e.folder, "members": infos, "cohorts": list(e.cohorts),
            "n_post": ecfg.n_post, "ppc_draws": ecfg.ppc_draws, "ppc_rows": ecfg.ppc_rows,
            "seeds": seeds, "oracle_max_abs_nats": oracle, "oracle_tol": ORACLE_TOL,
            "sim_bank": sim_paths, "cal_bank": ecfg.cal_glob, "files": written,
            "eval_config": ecfg.path, "config": cfg.path, "config_sha256": cfg.sha256,
            "seconds": round(time.time() - t0, 1), "torch_threads": torch.get_num_threads(),
            "ncpus": ncpus}
    pj = os.path.join(out, "%s_sample.json" % e.stem)
    with open(pj + ".tmp", "w") as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)
    os.replace(pj + ".tmp", pj)
    log("EVAL SAMPLE OK: %s (%d file(s), M=%d, n_post=%d, oracle max %.2g nats, %.0f s)"
        % (e.stem, len(written), M, ecfg.n_post, max(oracle.values()) if oracle else float("nan"),
           time.time() - t0))
    return prov


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the pipeline's project file (<tag>.json)")
    ap.add_argument("--eval-config", required=True, help="the evaluation's file (<tag>_eval.json)")
    ap.add_argument("--ensemble", required=True, help="<search>:<arm>, e.g. S-A2-R0:A1")
    ap.add_argument("--out", default=None, help="default <out_root>/samples")
    ap.add_argument("--ncpus", type=int, default=None, help="default: the eval config's ncpus")
    args = ap.parse_args(argv)
    try:
        run(args.config, args.eval_config, args.ensemble, out=args.out, ncpus=args.ncpus,
            log=lambda t: print(t, flush=True))
    except (ev_plan.RefuseError, ev_config.EvalConfigError) as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
