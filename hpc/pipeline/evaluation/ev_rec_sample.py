#!/usr/bin/env python3
"""Block 35: one ensemble of a recorded project -- its densities at the
held-out simulations' own parameters, its posterior draws on the draw rows
and on every recorded window, their densities, each member's embeddings
and the gate's roles (specs/SPEC.md sec. 2.6 and block 35; decisions D-159,
D-160, D-162, D-163).

    python3 hpc/pipeline/evaluation/ev_rec_sample.py --config <tag>.json \\
        --eval-config <tag>_rec_eval.json --ensemble <name> [--out DIR] [--ncpus N]

Writes under <out_root>/samples/ (or --out), each file to a temporary name
first:
    <stem>_sim.npz      D_sim, the report rows of the members' split
    <stem>_<r>.npz      each recording bank r of the ensemble
    <stem>_sample.json  the provenance, written last
Pass line: REC SAMPLE OK: <stem> ...; exit 2 on the arguments, 3 on a
refusal, which writes nothing.

CONTRACT (wave B): the constants and the signatures below are fixed by
specs/SPEC.md block 35.

The rule (block 35): the per-member functions are block 19's
(ev_sample.logq_at, draws_of, logq_draws_of, embed, sample_set), imported;
member k's draws on data set D start from seed_of(seed, D) + k, D being
"sim" or the recording bank's name, so that a synthetic project read as
recorded with every row a draw row gives block 19's arrays (D-162). On
D_sim the draw rows go through one sample_set call; the other rows are read
shard by shard for their densities at theta_i (and, for the gate's rows,
their embeddings). A recording bank's rows go through sample_set with no row
in the box: no density at a parameter is computed, none is written.
Every refusal comes before the first file is written.

Pure ASCII, LF only.
"""

import argparse
import json
import math
import os
import sys
import time

import _evalpath  # noqa: F401  (sys.path; the pipeline's config first)

import numpy as np
import torch
from scipy.special import logsumexp

import config as C
import ev_gate
import ev_models
import ev_plan
import ev_rec_config
import ev_rec_data
import ev_rec_plan
import ev_sample
import ev_simulate

import run_joint_arms as R

seed_of = ev_plan.seed_of      # a seed for (base, parts...), stable across machines and runs

EXIT_ARGS, EXIT_REFUSE = 2, 3
SCHEMA = "rec_eval_sample/1"

# <stem>_sim.npz: key -> (shape, dtype), with n the rows of D_sim, n_d the
# draw rows, n_z the rows with an embedding (the draw rows and the gate
# rows), M the members, p the axes, d_z the embedding's dimension
SIM_ARRAYS = {
    "rows": ("(n,)", "int64"),                    # bank rows, ascending
    "donor": ("(n,)", "int64"),
    "well": ("(n,)", "int64"),
    "cls": ("(n,)", "int64"),
    "theta_ref": ("(n, p)", "float64"),
    "in_box": ("(n,)", "bool"),
    "logq_ref_members": ("(n, M)", "float64"),    # -inf where not in the box
    "logq_ref": ("(n,)", "float64"),
    "calib": ("(n,)", "bool"),
    "draw": ("(n,)", "bool"),
    "gate_role": ("(n,)", "int8"),                # ev_gate.ROLE_*
    "draw_idx": ("(n_d,)", "int64"),              # indices into the n rows, ascending
    "draws": ("(n_d, n_post, p)", "float32"),
    "draw_member": ("(n_post,)", "int64"),
    "logq_draws": ("(n_d, n_post)", "float32"),
    "logq_draws_own": ("(n_d, n_post)", "float32"),
    "stats": ("(n_d, 8)", "float32"),
    "z_idx": ("(n_z,)", "int64"),                 # indices into the n rows, ascending
    "z": ("(M, n_z, d_z)", "float32"),
}

# <stem>_<r>.npz: n the rows of the recording bank, in bank order. No
# theta_ref, in_box or logq_ref* (D-162).
REC_ARRAYS = {
    "rows": ("(n,)", "int64"),
    "cls": ("(n,)", "int64"),
    "donor": ("(n,)", "int64"),
    "well": ("(n,)", "int64"),
    "batch": ("(n,)", "int64"),
    "subregion": ("(n,)", "int64"),
    "window_idx": ("(n,)", "int64"),
    "rep_group": ("(n,)", "int64"),
    "contaminated": ("(n,)", "bool"),
    "draws": ("(n, n_post, p)", "float32"),
    "draw_member": ("(n_post,)", "int64"),
    "logq_draws": ("(n, n_post)", "float32"),
    "logq_draws_own": ("(n, n_post)", "float32"),
    "z": ("(M, n, d_z)", "float32"),
    "stats": ("(n, 8)", "float32"),
}

# the keys of <stem>_sample.json (block 35, "the keys")
PROVENANCE_KEYS = (
    "schema", "ensemble", "stem", "search", "campaign", "arm", "folder", "view_kind",
    "members", "banks", "bank_globs", "sim_bank", "best_fit_complete_sha256", "sample_params",
    "seeds", "oracle_max_abs_nats", "oracle_mean_abs_nats", "oracle_tol", "gate", "n",
    "n_draw", "n_z", "n_out_of_box", "M", "n_post", "p", "axes", "files", "eval_config",
    "config",
    "config_sha256", "seconds", "ncpus", "torch_threads")


# ---------------------------------------------------------------- helpers

def _log_mixture(lq_members, in_box):
    """(n,) float64: the mixture's log density (E1) at theta_i from the
    members' (n, M), -inf where theta_i is not in the open box: the
    expression of ev_sample.sample_set, so that a row gets the same float
    whichever path computed its members' densities."""
    M = lq_members.shape[1]
    with np.errstate(divide="ignore"):
        return np.where(in_box, logsumexp(lq_members, axis=1) - math.log(M), -np.inf)


def _shard_blocks(index, rows):
    """Yields (positions, bank rows) of `rows` (ascending bank rows), one
    shard at a time, so that a block's windows are one shard's read."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return
    shard = np.searchsorted(np.asarray(index.offsets), rows, side="right") - 1
    for k in np.unique(shard):
        pos = np.flatnonzero(shard == k)
        yield pos, rows[pos]


def _gate_roles(donor, gate, seed):
    """(role, networks): ev_gate.assign_roles with the configuration's gate
    values; `networks` the numbers of networks dealt to the four groups
    (block 34's rule: floor(f G) for the first three, the rest to the
    fourth). A group without a network is a refusal."""
    try:
        role = ev_gate.assign_roles(donor, gate["donor_fracs"], gate["n_ref"], gate["n_rank"],
                                    gate["n_null"], gate["rows_per_donor"], seed)
    except ValueError as exc:
        raise ev_plan.RefuseError("the gate's roles cannot be dealt on D_sim: %s" % exc)
    G = int(np.unique(donor).size)
    sizes = [int(np.floor(float(f) * G + ev_gate._FLOOR_EPS)) for f in gate["donor_fracs"][:3]]
    sizes.append(G - sum(sizes))
    return role, sizes


# ---------------------------------------------------------------- the job

def run(cfg_path, ecfg_path, name, out=None, ncpus=None, log=print):
    """Samples ensemble `name` of the project at `cfg_path` under the
    evaluation's configuration at `ecfg_path`; writes the files under
    <out_root>/samples/ (or `out`); returns the provenance dict
    (PROVENANCE_KEYS). ev_plan.RefuseError on a refusal, nothing written."""
    t0 = time.time()
    ecfg = ev_rec_config.load(ecfg_path)
    view = ev_rec_plan.load_view(cfg_path, ecfg)
    e = ev_rec_plan.find(view, name)
    out = out or ecfg.sub("samples")
    ncpus = int(ncpus or ecfg.ncpus)
    torch.set_num_threads(max(1, ncpus))
    log("ensemble %s (%s view): %d member(s) in %s; recording bank(s) %s"
        % (e.name, view.kind, len(e.records), e.folder, ", ".join(e.banks)))

    # ---- the members
    index, sim_paths = ev_rec_data.sim_index(view)
    models, infos = [], []
    for r in e.records:
        m, info = ev_models.rebuild(r, index)
        models.append(m)
        infos.append(info)
        log("  member seed %d rebuilt (%s backbone, E %d)" % (info["seed"], info["backbone"],
                                                             info["embedding_size"]))
    M = len(models)
    n_post = int(ecfg.n_post)
    if n_post % M:
        raise ev_plan.RefuseError("n_post %d is not a multiple of the %d members" % (n_post, M))
    with open(e.records[0]) as fh:
        rec0 = json.load(fh)
    batch = int(ecfg.batch_rows)
    seed = int(ecfg.seed)
    seeds = {"sim": seed_of(seed, "sim"), "gate": seed_of(seed, "gate")}

    # ---- D_sim: its rows and their subsets
    rows = np.asarray(ev_rec_data.report_rows(index, rec0["split_fracs"], rec0["split_seed"],
                                              rec0["split_hash"]), dtype=np.int64)
    n = int(rows.size)
    th = np.asarray(index.theta, dtype=np.float64)[rows]
    p = int(th.shape[1])
    in_box = np.all((th > 0.0) & (th < 1.0), axis=1)
    n_out = int(n - in_box.sum())
    if n_out == n:
        raise ev_plan.RefuseError("no row of D_sim has its theta in the open box (%d row(s)): "
                                  "nothing can be evaluated at theta_i" % n)
    donor = np.asarray(index.donor, dtype=np.int64)[rows]
    calib = np.asarray(ev_rec_data.calibration_rows(donor), dtype=bool)
    draw = np.asarray(ev_rec_data.draw_rows(donor, ecfg.sim_draw_rows_per_donor), dtype=bool)
    role, networks = _gate_roles(donor, ecfg.gate, seeds["gate"])
    sizes = dict((k, int(np.sum(role == c))) for k, c in (
        ("n_ref", ev_gate.ROLE_REF), ("n_rank", ev_gate.ROLE_RANK),
        ("n_null", ev_gate.ROLE_NULL), ("n_check", ev_gate.ROLE_CHECK)))
    if sizes["n_null"] < int(ecfg.gate["n_null_min"]):
        raise ev_plan.RefuseError(
            "the gate's null set has %d row(s), below gate.n_null_min %d (%d network(s) in "
            "D_sim, gate.donor_fracs %s)" % (sizes["n_null"], int(ecfg.gate["n_null_min"]),
                                            int(np.unique(donor).size),
                                            list(ecfg.gate["donor_fracs"])))
    draw_idx = np.flatnonzero(draw).astype(np.int64)
    z_idx = np.flatnonzero(draw | (role != ev_gate.ROLE_NONE)).astype(np.int64)
    log("  sim: %d report row(s) of %d network(s); %d draw row(s); gate %d/%d/%d/%d rows"
        % (n, int(calib.sum()), draw_idx.size, sizes["n_ref"], sizes["n_rank"],
           sizes["n_null"], sizes["n_check"]))
    if n_out:
        # sec. 2.6, "The rows outside the open box": q_E(theta_i | x_i) = 0 there
        log("  sim: %d row(s) with theta outside the open box (a coordinate <= 0 or >= 1): "
            "log q = -inf there; left out of the oracle and of every quantity at theta_i"
            % n_out)

    # ---- the recording banks, read before any sampling (a refusal writes nothing)
    recs = [(r, ev_rec_data.recording(view, r)) for r in e.banks]
    for r, rec in recs:
        seeds[r] = seed_of(seed, r)
        log("  bank %s: %d window(s) of %d well(s)" % (r, rec["x"].shape[0],
                                                     np.unique(rec["well"]).size))

    # ---- D_sim: the draw rows, in one sample_set call
    x_d = R._rows_x(index, rows[draw_idx]).numpy()
    s = ev_sample.sample_set(models, x_d, th[draw_idx], in_box[draw_idx], n_post, seeds["sim"],
                             batch, log)
    stats = ev_simulate.fixed_stats(x_d)
    d_z = int(s["z"].shape[2])
    lq_members = np.full((n, M), -np.inf, dtype=np.float64)
    lq_members[draw_idx] = s["logq_ref_members"]
    z = np.empty((M, z_idx.size, d_z), dtype=np.float32)
    z[:, np.searchsorted(z_idx, draw_idx)] = s["z"]
    del x_d

    # ---- D_sim: the other rows, shard by shard (densities; embeddings of the gate's rows)
    rest = np.flatnonzero(~draw)
    for pos, brows in _shard_blocks(index, rows[rest]):
        idx = rest[pos]
        x_b = R._rows_x(index, brows).numpy()
        for k, m in enumerate(models):
            lq_members[idx, k] = ev_sample.logq_at(m, th[idx], x_b, in_box[idx], batch)
        g = np.flatnonzero(role[idx] != ev_gate.ROLE_NONE)
        if g.size:
            zg = np.stack([ev_sample.embed(m, x_b[g], batch) for m in models])
            z[:, np.searchsorted(z_idx, idx[g])] = zg
    lq_mix = _log_mixture(lq_members, in_box)

    # ---- the oracle (block 19's): each member's report-split NLL, row for row, on the
    # rows in the open box (outside it the record holds the flow's finite density and
    # the evaluation's is 0 under the box prior: nothing to compare, sec. 2.6)
    o_max, o_mean = [], []
    for k, info in enumerate(infos):
        pr = np.load(info["record"][:-len(".json")] + "_perrow.npz")
        nll = np.asarray(pr["nll"], dtype=np.float64)
        if nll.shape != (n,):
            raise ev_plan.RefuseError("member seed %d: its _perrow.npz has %d report rows, D_sim "
                                      "has %d (%s)" % (info["seed"], nll.size, n, info["record"]))
        dif = np.abs(nll[in_box] + lq_members[in_box, k])
        o_max.append(float(np.max(dif)))
        o_mean.append(float(np.mean(dif)))
        if not o_max[-1] <= ecfg.oracle_tol:
            raise ev_plan.RefuseError(
                "member seed %d does not reproduce its report-split NLL: largest difference "
                "%.3g nats, mean %.3g, above oracle_tol %.3g (%s)"
                % (info["seed"], o_max[-1], o_mean[-1], ecfg.oracle_tol, info["record"]))

    # ---- the files: D_sim first, then each recording bank, the provenance last
    os.makedirs(out, exist_ok=True)
    written = []
    p_sim = os.path.join(out, "%s_sim.npz" % e.stem)
    ev_sample.write_npz(p_sim, {
        "rows": rows, "donor": donor,
        "well": np.asarray(index.well, dtype=np.int64)[rows],
        "cls": np.asarray(index.cls, dtype=np.int64)[rows],
        "theta_ref": th, "in_box": in_box, "logq_ref_members": lq_members, "logq_ref": lq_mix,
        "calib": calib, "draw": draw, "gate_role": role.astype(np.int8), "draw_idx": draw_idx,
        "draws": s["draws"], "draw_member": s["draw_member"], "logq_draws": s["logq_draws"],
        "logq_draws_own": s["logq_draws_own"], "stats": stats, "z_idx": z_idx, "z": z})
    written.append(p_sim)
    log("  sim: %d row(s) -> %s" % (n, p_sim))
    del s, z

    for r, rec in recs:
        nr = int(rec["x"].shape[0])
        s = ev_sample.sample_set(models, rec["x"], np.zeros((nr, p), dtype=np.float64),
                                 np.zeros(nr, dtype=bool), n_post, seeds[r], batch, log)
        arrays = {"rows": np.arange(nr, dtype=np.int64)}
        for k in ev_rec_data.RECORDING_IDS:
            arrays[k] = np.asarray(rec[k], dtype=np.int64)
        arrays["contaminated"] = np.asarray(rec["contaminated"], dtype=bool)
        for k in ("draws", "draw_member", "logq_draws", "logq_draws_own", "z"):
            arrays[k] = s[k]
        arrays["stats"] = ev_simulate.fixed_stats(rec["x"])
        pr_ = os.path.join(out, "%s_%s.npz" % (e.stem, r))
        ev_sample.write_npz(pr_, arrays)
        written.append(pr_)
        log("  %s: %d window(s) -> %s" % (r, nr, pr_))
        del s, arrays

    names, bounds = ev_rec_data.axes_of(index.sidecar, p)
    cfg = view.cfg
    gate = dict(sizes)
    gate.update(networks=networks, seed=seeds["gate"])
    prov = {"schema": SCHEMA, "ensemble": e.name, "stem": e.stem, "search": e.search,
            "campaign": e.campaign, "arm": e.arm, "folder": e.folder, "view_kind": view.kind,
            "members": infos, "banks": list(e.banks),
            "bank_globs": dict((r, view.banks[r]) for r in e.banks), "sim_bank": sim_paths,
            "best_fit_complete_sha256": ev_plan.sha256_file(
                os.path.join(e.folder, "BEST_FIT_COMPLETE.json")),
            "sample_params": ecfg.sample_params(), "seeds": seeds,
            "oracle_max_abs_nats": o_max, "oracle_mean_abs_nats": o_mean,
            "oracle_tol": ecfg.oracle_tol, "gate": gate, "n": n, "n_draw": int(draw_idx.size),
            "n_z": int(z_idx.size), "n_out_of_box": n_out, "M": M, "n_post": n_post, "p": p,
            "axes": {"names": names, "bounds": None if bounds is None else bounds.tolist()},
            "files": written, "eval_config": ecfg.path, "config": cfg.path,
            "config_sha256": cfg.sha256, "seconds": round(time.time() - t0, 1), "ncpus": ncpus,
            "torch_threads": torch.get_num_threads()}
    pj = os.path.join(out, "%s_sample.json" % e.stem)
    with open(pj + ".tmp", "w") as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)
    os.replace(pj + ".tmp", pj)
    log("REC SAMPLE OK: %s (%d file(s), M=%d, n_post=%d, %d draw row(s) of %d, oracle max %.2g "
        "nats, %.0f s)" % (e.stem, len(written) + 1, M, n_post, draw_idx.size, n, max(o_max),
                           time.time() - t0))
    return prov


def main(argv=None):
    """The command line: 0 with the pass line, EXIT_ARGS, EXIT_REFUSE."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the pipeline's project file (<tag>.json)")
    ap.add_argument("--eval-config", required=True,
                    help="the evaluation's file (<tag>_rec_eval.json)")
    ap.add_argument("--ensemble", required=True, help="<q>:<arm> or best:<arm>")
    ap.add_argument("--out", default=None, help="default <out_root>/samples")
    ap.add_argument("--ncpus", type=int, default=None, help="default: the eval config's ncpus")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return EXIT_ARGS if exc.code else 0
    try:
        run(args.config, args.eval_config, args.ensemble, out=args.out, ncpus=args.ncpus,
            log=lambda t: print(t, flush=True))
    except (ev_plan.RefuseError, ev_rec_config.RecEvalConfigError, C.ConfigError) as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    except ValueError as exc:
        # a bank that breaks its contract (latent_bank.ContractError, a ValueError: a NaN
        # in theta, ...) is a refusal that names it, not a traceback; nothing was written
        if type(exc).__name__ != "ContractError":
            raise
        print("REFUSE: the bank breaks its contract: %s" % exc, flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
