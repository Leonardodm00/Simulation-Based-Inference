#!/usr/bin/env python3
"""
npe_tune.py -- the hyperparameter tuning driver.

Amortized NPE over the simulator's parameters, conditioned on frozen DSN
embeddings. This driver tunes the ensemble's hyperparameters against the
held-out log-probability of a frozen, group-disjoint selection split, using
a Gaussian-process surrogate search, and gates the finalists on
informativeness and calibration before anything is shipped.

Everything is stateless across invocations. The results directory IS the
campaign state: the search warm-starts by replaying it, a crashed
coordinator loses nothing, and raising the budget never repeats work.

Subcommands, in the order a campaign uses them:

    freeze-split     load the bank, cut the 3-way grouped split, freeze it
    baseline         the default configuration, plus the shuffled control
    propose          ask the GP for the next batch of configurations
    evaluate         run ONE configuration (this is the job body)
    status           ledger, convergence trace, escalation verdict
    finalists        promote the top-K to full M, run the gate battery
    learning-curve   train at reducing training-set size at a fixed config
    report           the final report, including the report-split score

Typical use on the cluster:

    python npe_tune.py freeze-split --sim '...*.parquet' --out-dir run1
    python npe_tune.py baseline     --out-dir run1
    python npe_tune.py propose      --out-dir run1 --n 8
    qsub -v OUT_DIR=run1 jobs/npe_tune.pbs          # one job per pending spec
    python npe_tune.py status       --out-dir run1
    python npe_tune.py finalists    --out-dir run1 --top-k 3
    python npe_tune.py report       --out-dir run1

Every command prints a shape report first. That block is what a cluster run
should be checked against: if p, E, n or the split hash is not what was
expected, nothing after it matters.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import npe_tune_data as TD
import npe_tune_gates as TG
import npe_tune_ledger as TL
import npe_tune_score as TS
import npe_tune_search as TSR

DEFAULT_SEEDS = tuple(range(10))          # M = 10, the deliverable size
CAMPAIGN_FILE = "campaign.json"
SPLIT_FILE = "split_manifest.json"
RESULTS_DIR = "results"
PENDING_DIR = "pending"
MODELS_DIR = "models"


# ---------------------------------------------------------------------------
# Campaign state on disk
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# The repository's NPEConfig defaults, mirrored so that the coordinator
# commands (baseline, propose, status) run on a login node with no training
# stack installed. npe_model is preferred whenever it imports, so the two can
# never drift silently: the smoke test asserts they agree.
FALLBACK_DEFAULTS = {
    "hidden_features": 128,
    "num_transforms": 8,
    "num_bins": 10,
    "learning_rate": 5e-4,
    "training_batch_size": 512,
}


def default_config(verbose: bool = True) -> Dict[str, Any]:
    """The searched knobs at their repository defaults."""
    try:
        import npe_model
        d = npe_model.NPEConfig()
        return {"hidden_features": int(d.hidden_features),
                "num_transforms": int(d.num_transforms),
                "num_bins": int(d.num_bins),
                "learning_rate": float(d.learning_rate),
                "training_batch_size": int(d.training_batch_size)}
    except Exception as exc:
        if verbose:
            print("[note] npe_model did not import (%s); using the mirrored "
                  "defaults %s. Fine for queueing work, but confirm the "
                  "training environment before evaluating anything."
                  % (type(exc).__name__, FALLBACK_DEFAULTS), flush=True)
        return dict(FALLBACK_DEFAULTS)


def campaign_path(out_dir: str) -> str:
    return os.path.join(out_dir, CAMPAIGN_FILE)


def save_campaign(out_dir: str, state: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = campaign_path(out_dir)
    with open(p, "w", encoding="ascii") as fh:
        json.dump(TL.json_safe(state), fh, indent=2, sort_keys=True)
    return p


def load_campaign(out_dir: str) -> Dict[str, Any]:
    p = campaign_path(out_dir)
    if not os.path.exists(p):
        raise SystemExit(
            "no campaign found at %r. Run `freeze-split` first: it loads the "
            "bank, cuts the split and records the loader arguments that every "
            "later command must reuse." % p)
    with open(p, "r", encoding="ascii") as fh:
        return json.load(fh)


def open_bank_and_split(out_dir: str,
                        verbose: bool = True) -> Tuple[TD.Bank, TD.SplitManifest, Dict[str, Any]]:
    """Reload exactly the bank the campaign was frozen on, and its split."""
    state = load_campaign(out_dir)
    ld = state["loader_args"]
    bank = TD.load_bank(
        sim_glob=ld["sim_glob"],
        activity_path=ld.get("activity_path"),
        min_rate=ld.get("min_rate"),
        max_rate=ld.get("max_rate"),
        dedup_theta=bool(ld.get("dedup_theta", True)),
        max_rows=ld.get("max_rows"),
        topology_axes=tuple(ld.get("topology_axes", TD.DEFAULT_TOPOLOGY_AXES)),
        allow_iid_split=bool(ld.get("allow_iid_split", False)),
        seed=int(ld.get("loader_seed", 0)),
    )
    manifest = TD.load_split(os.path.join(out_dir, SPLIT_FILE))
    TD.check_split(bank, manifest)
    if verbose:
        print(TD.shape_report(bank, manifest), flush=True)
    return bank, manifest, state


# ---------------------------------------------------------------------------
# freeze-split
# ---------------------------------------------------------------------------

def cmd_freeze_split(args: argparse.Namespace) -> int:
    bank = TD.load_bank(
        sim_glob=args.sim,
        activity_path=args.activity,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        dedup_theta=not args.no_dedup,
        max_rows=args.max_rows,
        topology_axes=tuple(args.topology_axes),
        allow_iid_split=args.allow_iid_split,
        seed=args.loader_seed,
    )
    print(TD.shape_report(bank), flush=True)

    # A-T1: after deduplication every row should carry a distinct theta.
    uniq = np.unique(bank.theta, axis=0).shape[0]
    if uniq != bank.n:
        print("[warn] A-T1: %d rows but only %d distinct theta -- "
              "deduplication did not leave one row per parameter draw"
              % (bank.n, uniq), flush=True)
    else:
        print("[ok] A-T1: one row per distinct theta (%d)" % uniq, flush=True)

    manifest = TD.make_split(bank, fractions=tuple(args.fractions),
                             seed=args.split_seed,
                             min_groups_per_side=args.min_groups_per_side,
                             min_rows_per_side=args.min_rows_per_side)
    os.makedirs(args.out_dir, exist_ok=True)
    TD.save_split(manifest, os.path.join(args.out_dir, SPLIT_FILE))
    if manifest.has_gate_split:
        print("[split] four-way: `sel` ranks, `gate` gates, `rep` reports "
              "(touched once).", flush=True)
    else:
        print("[split] three-way: `sel` both ranks AND gates. The finalist "
              "gate is then anti-conservative -- the candidate's score on "
              "`sel` is a maximum over the whole search, and a null "
              "calibrated for a single configuration understates the "
              "false-pass rate. Re-freeze with four fractions to separate "
              "them (S4.2).", flush=True)

    floor = TS.prior_floor(bank.contract.bounds_theta)
    space = TSR.default_space(bank.p, bank.embedding_dim,
                              n_train=len(manifest.train))
    state = {
        "created_utc": _utc(),
        "git_sha": _git_sha(),
        "loader_args": {
            "sim_glob": args.sim, "activity_path": args.activity,
            "min_rate": args.min_rate, "max_rate": args.max_rate,
            "dedup_theta": not args.no_dedup, "max_rows": args.max_rows,
            "topology_axes": list(args.topology_axes),
            "allow_iid_split": bool(args.allow_iid_split),
            "loader_seed": int(args.loader_seed),
        },
        "split_hash": manifest.hash(),
        "contract_digest": manifest.contract_digest,
        "prior_floor": floor,
        "p": bank.p, "embedding_dim": bank.embedding_dim, "n_rows": bank.n,
        "n_groups": bank.n_groups,
        "seeds": [int(s) for s in args.seeds],
        "m_probe": int(args.m_probe),
        "space": space.to_dict(),
        "library_versions": _safe_versions(),
    }
    save_campaign(args.out_dir, state)

    print(TD.shape_report(bank, manifest), flush=True)
    print("[floor] L_0 = %.6f nats/row (differential entropy of the prior "
          "box, exact from the contract)" % floor, flush=True)
    print("[split] frozen at %s (hash %s)"
          % (os.path.join(args.out_dir, SPLIT_FILE), manifest.hash()),
          flush=True)
    print("[space] %s" % json.dumps(space.to_dict(), sort_keys=True),
          flush=True)
    return 0


def _safe_versions() -> Dict[str, str]:
    try:
        import npe_tune_train as TT
        return TT.library_versions()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# evaluate -- the job body
# ---------------------------------------------------------------------------

def _resolve_spec(args: argparse.Namespace,
                  state: Dict[str, Any]) -> Dict[str, Any]:
    """Everything this evaluation should do, from a pending spec or the CLI.

    A spec file wins over the corresponding CLI flag, because the spec is
    what the coordinator committed to and what the trial id was computed
    from. Silently letting a flag override it would produce a result filed
    under an id that does not describe it.
    """
    if args.spec:
        with open(args.spec, "r", encoding="ascii") as fh:
            spec = json.load(fh)
        if "config" not in spec:
            raise SystemExit("spec %r has no 'config' block" % args.spec)
        spec = dict(spec)
        spec.setdefault("tag", "")
        return spec

    d = default_config()
    cfg = {
        "hidden_features": args.hidden_features or d["hidden_features"],
        "num_transforms": args.num_transforms or d["num_transforms"],
        "num_bins": args.num_bins or d["num_bins"],
        "learning_rate": args.learning_rate or d["learning_rate"],
        "training_batch_size": args.batch_size or d["training_batch_size"],
    }
    spec = {"config": cfg, "tag": str(args.tag or "")}
    if args.n_members:
        spec["n_members"] = int(args.n_members)
    if args.train_fraction is not None:
        spec["train_fraction"] = float(args.train_fraction)
    if args.shuffle_control:
        spec["shuffle_control"] = True
        spec["shuffle_seed"] = int(args.shuffle_seed)
    return spec


def cmd_evaluate(args: argparse.Namespace) -> int:
    import npe_tune_train as TT

    bank, manifest, state = open_bank_and_split(args.out_dir)
    spec = _resolve_spec(args, state)
    config = dict(spec["config"])
    tag = str(spec.get("tag", ""))

    seeds_all = [int(s) for s in state.get("seeds", DEFAULT_SEEDS)]
    if spec.get("seeds"):
        seeds = [int(s) for s in spec["seeds"]]
        n_members = len(seeds)
    else:
        n_members = int(spec.get("n_members")
                        or args.n_members or state.get("m_probe", 3))
        seeds = seeds_all[:n_members]
        if len(seeds) < n_members:
            raise SystemExit(
                "campaign has %d seeds but %d members were requested; "
                "re-freeze with a longer --seeds list"
                % (len(seeds_all), n_members))

    train_fraction = spec.get("train_fraction", args.train_fraction)
    shuffle_control = bool(spec.get("shuffle_control", args.shuffle_control))
    shuffle_seed = int(spec.get("shuffle_seed", args.shuffle_seed))

    train_idx = np.asarray(manifest.train, dtype=np.int64)
    sel_idx = np.asarray(manifest.sel, dtype=np.int64)

    # Learning-curve points shrink the TRAINING set by whole topology groups,
    # never the evaluation split, so every point is scored on the same rows.
    if train_fraction is not None and float(train_fraction) < 1.0:
        train_idx = TD.subsample_groups(bank, train_idx, float(train_fraction),
                                        seed=int(args.subsample_seed))
        tag = tag or ("frac%0.3f" % float(train_fraction))

    z_tr, th_tr = bank.z[train_idx], bank.theta[train_idx]
    z_se, th_se = bank.z[sel_idx], bank.theta[sel_idx]

    # The shuffled-pairs control: permute theta against z within the training
    # split only. Both marginals survive, the association does not, so the
    # achievable optimum IS the prior and the resulting gain measures the
    # "learned nothing" floor for this bank empirically.
    if shuffle_control:
        rng = np.random.default_rng(shuffle_seed)
        th_tr = th_tr[rng.permutation(th_tr.shape[0])]
        tag = tag or "shuffled_control"

    tid = spec.get("trial_id") or TL.trial_id(
        config, manifest.hash(), manifest.contract_digest, n_members, seeds,
        tag=tag)
    results_dir = os.path.join(args.out_dir, RESULTS_DIR)
    if not args.force and os.path.exists(
            os.path.join(results_dir, "trial_%s.json" % tid)):
        print("[skip] trial %s already evaluated under this split; pass "
              "--force to redo it" % tid, flush=True)
        TL.clear_pending(os.path.join(args.out_dir, PENDING_DIR), tid)
        return 0

    print("[trial] id=%s tag=%r M=%d seeds=%s" % (tid, tag, n_members, seeds),
          flush=True)
    print("[trial] config=%s" % json.dumps(config, sort_keys=True), flush=True)
    print("[trial] train rows=%d  sel rows=%d" % (z_tr.shape[0], z_se.shape[0]),
          flush=True)

    started = _utc()
    rec = TL.TrialRecord(
        trial_id=tid, tag=tag, config=config, seeds=seeds,
        n_members=n_members, split_hash=manifest.hash(),
        contract_digest=manifest.contract_digest, p=bank.p,
        embedding_dim=bank.embedding_dim, n_train_rows=int(z_tr.shape[0]),
        git_sha=state.get("git_sha", ""), env_name=os.environ.get("CONDA_DEFAULT_ENV", ""),
        library_versions=TT.library_versions(), hostname=socket.gethostname(),
        started_utc=started,
    )

    try:
        cfg_obj = TT.make_config(config, device=args.device)
        prior = bank.contract.prior(device=args.device)
        model_dir = os.path.join(args.out_dir, MODELS_DIR, tid)
        posteriors, records = TT.train_members(z_tr, th_tr, prior, cfg_obj,
                                               seeds, verbose=True)
        if args.save_model:
            TT.save_members(posteriors, model_dir, seeds, cfg_obj, records,
                            contract=bank.contract,
                            extra={"trial_id": tid, "tag": tag,
                                   "split_hash": manifest.hash()})
            rec.model_dir = model_dir

        ens = TT.build_ensemble(posteriors) if len(posteriors) > 1 else posteriors[0]
        member_lp, mix_lp = TT.evaluate_log_probs(posteriors, th_se, z_se,
                                                  ensemble=ens if len(posteriors) > 1 else None)
        score = TS.score_from_log_probs(mix_lp, bank.contract.bounds_theta,
                                        member_log_probs=member_lp,
                                        n_members=n_members,
                                        n_boot=int(args.n_boot),
                                        seed=int(args.boot_seed))
        rec.nll = score.nll
        rec.floor = score.floor
        rec.delta = score.delta
        rec.delta_ci_lo = score.delta_ci_lo
        rec.delta_ci_hi = score.delta_ci_hi
        rec.member_nll = list(score.member_nll or [])
        rec.member_nll_mean = score.member_nll_mean
        rec.jensen_ok = score.jensen_ok
        rec.n_eval_rows = score.n_rows
        rec.epochs = [r.epochs for r in records]
        rec.best_val_log_prob = [r.best_val_log_prob for r in records]
        rec.hit_max_epochs = [r.hit_max_epochs for r in records]
        rec.train_seconds = [r.train_seconds for r in records]
        rec.extra["member_notes"] = [r.notes for r in records]
        if train_fraction is not None:
            rec.extra["train_fraction"] = float(train_fraction)
        if shuffle_control:
            rec.extra["shuffle_control"] = True
        rec.status = "ok"
        print("[score] %s" % score.summary(), flush=True)
        if score.jensen_ok is False:
            print("[warn] the mixture scored WORSE than the mean of its "
                  "members. Eq. (4) forbids this for an arithmetic mixture; "
                  "check how the ensemble is being combined.", flush=True)
    except Exception as exc:                     # record, do not hide
        rec.status = "error"
        rec.error = "%s: %s" % (type(exc).__name__, exc)
        print("[error] %s" % rec.error, flush=True)
    finally:
        rec.finished_utc = _utc()
        path = TL.write_trial(rec, results_dir)
        TL.clear_pending(os.path.join(args.out_dir, PENDING_DIR), tid)
        print("[ledger] wrote %s" % path, flush=True)

    return 0 if rec.status == "ok" else 1


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

def cmd_baseline(args: argparse.Namespace) -> int:
    """Queue the default configuration at several seeds plus the control.

    This measures three things the rest of the campaign depends on and none
    of which may be guessed: the baseline score, the across-seed spread
    (which sets both the one-SE rule and the escalation tolerance), and the
    shuffled-pairs floor (which sets the G1 threshold).
    """
    state = load_campaign(args.out_dir)
    cfg = default_config()
    pending_dir = os.path.join(args.out_dir, PENDING_DIR)
    m_probe = int(state.get("m_probe", 3))
    seeds = [int(s) for s in state.get("seeds", DEFAULT_SEEDS)][:m_probe]

    specs = []
    for rep in range(int(args.n_seed_reps)):
        # Distinct seed windows give the ACROSS-seed spread; the identical
        # window repeated (rep 0 vs --repeat-identical) gives the
        # run-to-run spread. Both are needed and they mean different things.
        s = [int(x) for x in state["seeds"]][rep * m_probe:(rep + 1) * m_probe]
        if len(s) < m_probe:
            print("[baseline] only %d seed window(s) available for M_probe=%d"
                  % (rep, m_probe), flush=True)
            break
        specs.append({"config": cfg, "tag": "baseline_seedwin%d" % rep,
                      "seeds": s})
    for rep in range(int(args.n_control)):
        specs.append({"config": cfg, "tag": "shuffled_control%d" % rep,
                      "seeds": seeds, "shuffle_control": True,
                      "shuffle_seed": 1000 + rep})

    for spec in specs:
        spec["trial_id"] = TL.trial_id(spec["config"], state["split_hash"],
                                       state["contract_digest"],
                                       len(spec["seeds"]), spec["seeds"],
                                       tag=spec["tag"])
        TL.write_pending(spec, pending_dir)
        print("[baseline] queued %s (%s)" % (spec["trial_id"], spec["tag"]),
              flush=True)
    print("[baseline] %d spec(s) written to %s. Run them with `evaluate "
          "--spec <file>` (one job each), then re-run `status`."
          % (len(specs), pending_dir), flush=True)
    return 0


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------

def _observations(out_dir: str, state: Dict[str, Any],
                  tag: Optional[str] = None,
                  verbose: bool = True) -> List[Tuple[Dict[str, Any], float]]:
    """(config, score) pairs from the ledger -- the warm start, and the only
    state the search has."""
    recs = TL.load_ledger(os.path.join(out_dir, RESULTS_DIR),
                          split_hash=state["split_hash"],
                          contract_digest=state["contract_digest"],
                          require_ok=True, verbose=verbose)
    obs = []
    for r in recs:
        if r.tag and r.tag.startswith(("shuffled_control", "frac")):
            continue                      # controls are not search points
        if tag is not None and r.tag != tag:
            continue
        if not np.isfinite(r.nll):
            continue
        try:
            TSR.point_from_config(r.config)
        except KeyError:
            continue                      # not a searched configuration
        obs.append((dict(r.config), float(r.nll)))
    return obs


def cmd_propose(args: argparse.Namespace) -> int:
    state = load_campaign(args.out_dir)
    spec = TSR.SpaceSpec(**{k: (tuple(v) if isinstance(v, list) else v)
                            for k, v in state["space"].items()
                            if k != "anchored_to"})
    spec.anchored_to = state["space"].get("anchored_to", {})

    obs = _observations(args.out_dir, state)
    pending_dir = os.path.join(args.out_dir, PENDING_DIR)
    pend_ids = set(TL.pending_ids(pending_dir))
    pend_cfgs = []
    for tid in pend_ids:
        p = os.path.join(pending_dir, "trial_%s.json" % tid)
        try:
            with open(p, "r", encoding="ascii") as fh:
                pend_cfgs.append(json.load(fh)["config"])
        except Exception:
            pass

    noise = None
    if args.noise_sd is not None:
        noise = float(args.noise_sd) ** 2

    configs = TSR.propose(spec, obs, n_points=int(args.n),
                          n_initial_points=int(args.n_initial),
                          seed=int(args.seed), noise=noise,
                          exclude=pend_cfgs)

    m_probe = int(state.get("m_probe", 3))
    seeds = [int(s) for s in state.get("seeds", DEFAULT_SEEDS)][:m_probe]
    written = 0
    for cfg in configs:
        tid = TL.trial_id(cfg, state["split_hash"], state["contract_digest"],
                          m_probe, seeds, tag="")
        if tid in pend_ids:
            continue
        TL.write_pending({"trial_id": tid, "config": cfg, "tag": "",
                          "seeds": seeds, "n_members": m_probe}, pending_dir)
        written += 1
        print("[propose] %s  %s" % (tid, json.dumps(cfg, sort_keys=True)),
              flush=True)
    print("[propose] %d observation(s) replayed, %d new spec(s) in %s"
          % (len(obs), written, pending_dir), flush=True)
    if len(obs) < int(args.n_initial):
        print("[propose] note: still inside the random initial design "
              "(%d of %d); proposals are random, not surrogate-guided."
              % (len(obs), int(args.n_initial)), flush=True)
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _seed_spread(records: Sequence[TL.TrialRecord]) -> Dict[str, float]:
    """Across-seed and run-to-run spreads, from the baseline replicates."""
    base = [r for r in records if r.tag.startswith("baseline_seedwin")]
    out = {"sigma_seed": float("nan"), "n_seed_windows": len(base)}
    if len(base) >= 2:
        vals = np.asarray([r.nll for r in base], dtype=np.float64)
        out["sigma_seed"] = float(np.std(vals, ddof=1))
        out["baseline_nll_mean"] = float(np.mean(vals))
    elif len(base) == 1:
        out["baseline_nll_mean"] = float(base[0].nll)
    return out


def cmd_status(args: argparse.Namespace) -> int:
    state = load_campaign(args.out_dir)
    recs = TL.load_ledger(os.path.join(args.out_dir, RESULTS_DIR),
                          split_hash=state["split_hash"],
                          contract_digest=state["contract_digest"],
                          require_ok=False, verbose=True)
    ok = [r for r in recs if r.status == "ok"]
    bad = [r for r in recs if r.status != "ok"]
    print("\n" + TL.ledger_table(ok), flush=True)
    if bad:
        print("\n[failed trials] %d:" % len(bad), flush=True)
        for r in bad[:10]:
            print("   %s %s" % (r.trial_id[:8], r.error[:120]), flush=True)

    floor = float(state.get("prior_floor", float("nan")))
    print("\n[floor] L_0 = %.6f nats/row" % floor, flush=True)

    spread = _seed_spread(ok)
    ctrl = [r for r in ok if r.tag.startswith("shuffled_control")]
    if ctrl:
        dmin = TG.delta_min_from_control([r.delta for r in ctrl],
                                         k_sigma=float(args.k_sigma))
        print("[control] %d shuffled-pairs run(s), gain %s -> G1 threshold "
              "delta_min = %.4f nats/row"
              % (len(ctrl), ", ".join("%.4f" % r.delta for r in ctrl), dmin),
              flush=True)
        for r in ctrl:
            if r.delta > 0.05 * max(1.0, abs(floor)):
                print("[warn] a shuffled control scored a gain of %.4f. The "
                      "control destroys the theta-z association, so its "
                      "optimum is the prior: a clearly positive gain means "
                      "the split leaks and nothing downstream is "
                      "interpretable." % r.delta, flush=True)
    else:
        print("[control] none run yet: the G1 threshold is unmeasured. Run "
              "`baseline` before trusting any gate verdict.", flush=True)

    if np.isfinite(spread.get("sigma_seed", float("nan"))):
        print("[noise] sigma_seed = %.4f nats/row over %d baseline seed "
              "window(s); this is tau_stop and the one-SE scale"
              % (spread["sigma_seed"], spread["n_seed_windows"]), flush=True)
    else:
        print("[noise] sigma_seed unmeasured (need >= 2 baseline seed "
              "windows); escalation falls back to --tau-stop", flush=True)

    obs = _observations(args.out_dir, state, verbose=False)
    if obs:
        trace = TSR.convergence_trace([y for _, y in obs])
        print("\n[trace] best-so-far NLL after each search evaluation:",
              flush=True)
        print("   " + " ".join("%.4f" % t for t in trace), flush=True)

    spec = TSR.SpaceSpec(**{k: (tuple(v) if isinstance(v, list) else v)
                            for k, v in state["space"].items()
                            if k != "anchored_to"})
    tau = spread.get("sigma_seed")
    if tau is None or not np.isfinite(tau):
        tau = float(args.tau_stop)
    verdict = TSR.escalation_verdict(obs, spec, tau_stop=float(tau),
                                     window=args.window)
    print("\n" + verdict.summary(), flush=True)
    if verdict.best_config:
        print("[best] NLL %.4f  gain %.4f  %s"
              % (verdict.best_score, floor - verdict.best_score,
                 json.dumps(verdict.best_config, sort_keys=True)), flush=True)

    pend = TL.pending_ids(os.path.join(args.out_dir, PENDING_DIR))
    print("[pending] %d spec(s) awaiting evaluation" % len(pend), flush=True)
    return 0


# ---------------------------------------------------------------------------
# finalists
# ---------------------------------------------------------------------------

def cmd_finalists(args: argparse.Namespace) -> int:
    """Promote the top-K configurations to the full ensemble, then gate them.

    Promotion trains only the members that do not exist yet, because members
    are independent of one another and of M. The gate battery runs here and
    nowhere earlier: G2 and G3 need posterior draws for every calibration
    observation for every member, a cost that scales as n_calib * M and does
    not benefit from a bigger machine.
    """
    import npe_tune_train as TT

    bank, manifest, state = open_bank_and_split(args.out_dir)
    recs = TL.load_ledger(os.path.join(args.out_dir, RESULTS_DIR),
                          split_hash=state["split_hash"],
                          contract_digest=state["contract_digest"],
                          require_ok=True)
    search = [r for r in recs
              if not r.tag or not r.tag.startswith(("shuffled_control", "frac"))]
    search = [r for r in search if np.isfinite(r.nll)]
    if not search:
        raise SystemExit("no completed search trials under this split")

    ctrl = [r for r in recs if r.tag.startswith("shuffled_control")]
    delta_min = TG.delta_min_from_control([r.delta for r in ctrl],
                                          k_sigma=float(args.k_sigma))
    if not ctrl:
        print("[warn] no shuffled control has been run, so the G1 threshold "
              "is 0.0 rather than a measured floor. Say so wherever these "
              "verdicts are reported.", flush=True)

    search.sort(key=lambda r: r.nll)
    top = search[:int(args.top_k)]
    seeds = [int(s) for s in state.get("seeds", DEFAULT_SEEDS)]
    M = int(args.m_full or len(seeds))
    seeds = seeds[:M]

    train_idx = np.asarray(manifest.train, dtype=np.int64)
    z_tr, th_tr = bank.z[train_idx], bank.theta[train_idx]

    # Rank on `sel` (the ledger NLL that produced `top`), gate on `gate`.
    # Scoring the finalists again on `sel` would compare a maximum over the
    # whole search against a null calibrated for a single configuration; the
    # fourth split makes the gate statistic independent of the selection.
    # HANDOFF_DELTA_MIN_PER_CONFIG_v1 S4.2. A three-way manifest still runs,
    # with the bias stated rather than hidden.
    rank_split = "sel"
    gate_split = "gate" if manifest.has_gate_split else "sel"
    gate_idx = np.asarray(manifest.side(gate_split), dtype=np.int64)
    z_ga, th_ga = bank.z[gate_idx], bank.theta[gate_idx]
    if not manifest.has_gate_split:
        print("[warn] this manifest is three-way, so the finalists are "
              "ranked AND gated on `sel`. The gate verdict is then "
              "anti-conservative by an unmeasured amount (winner's curse "
              "over %d search trials): report it as such, or re-freeze the "
              "split with four fractions." % len(search), flush=True)

    rng = np.random.default_rng(int(args.calib_seed))
    n_calib = min(int(args.n_calib), z_ga.shape[0])
    calib = np.sort(rng.choice(z_ga.shape[0], size=n_calib, replace=False))
    z_cal, th_cal = z_ga[calib], th_ga[calib]

    print("[finalists] %d candidate(s); M=%d; delta_min=%.4f; n_calib=%d"
          % (len(top), M, delta_min, n_calib), flush=True)
    print("[finalists] ranked on %r (%d rows), gated on %r (%d rows)"
          % (rank_split, len(manifest.sel), gate_split, gate_idx.size),
          flush=True)

    out: List[Dict[str, Any]] = []
    for r in top:
        print("\n[finalist] %s  probe NLL %.4f  %s"
              % (r.trial_id[:8], r.nll, json.dumps(r.config, sort_keys=True)),
              flush=True)
        cfg_obj = TT.make_config(r.config, device=args.device)
        prior = bank.contract.prior(device=args.device)
        model_dir = os.path.join(args.out_dir, MODELS_DIR, r.trial_id)
        posteriors, mrecs = TT.extend_ensemble(model_dir, z_tr, th_tr, prior,
                                               cfg_obj, seeds,
                                               contract=bank.contract)
        ens = TT.build_ensemble(posteriors)

        member_lp, mix_lp = TT.evaluate_log_probs(posteriors, th_ga, z_ga,
                                                  ensemble=ens)
        score = TS.score_from_log_probs(mix_lp, bank.contract.bounds_theta,
                                        member_log_probs=member_lp,
                                        n_members=M, n_boot=int(args.n_boot),
                                        seed=int(args.boot_seed))
        print("[finalist] full-M score on the %r split: %s"
              % (gate_split, score.summary()), flush=True)

        import npe_diagnostics as D
        samples = D.sample_posteriors(ens, z_cal, n_draws=int(args.n_draws))
        lp_true = np.asarray(D.posterior_log_probs(ens, th_cal, z_cal),
                             dtype=np.float64)
        lp_samp = np.stack(
            [np.asarray(D.posterior_log_probs(ens, samples[:, k, :], z_cal),
                        dtype=np.float64)
             for k in range(samples.shape[1])], axis=1)

        seed_deltas = [score.floor + float(np.mean(row))
                       for row in member_lp]     # per-member gain
        battery = TG.run_all_gates(
            delta=score.delta, delta_ci_lo=score.delta_ci_lo,
            delta_min=delta_min, theta_true=th_cal,
            posterior_samples=samples, Z=z_cal, log_prob_true=lp_true,
            log_prob_samples=lp_samp,
            param_names=list(bank.contract.param_names),
            seed_deltas=seed_deltas, alpha=float(args.alpha),
            n_forms=int(args.n_forms), seed=int(args.calib_seed),
            run_calibration=True)
        print(battery.summary(), flush=True)

        contraction = D.posterior_contraction(
            th_cal, samples, param_names=list(bank.contract.param_names))
        spectrum = D.information_spectrum(
            th_cal, samples, param_names=list(bank.contract.param_names))
        c = np.asarray(contraction.contraction, dtype=np.float64)
        print("[contraction] median %.4f  max %.4f  axes>0.1: %d/%d"
              % (float(np.median(c)), float(np.max(c)),
                 int(np.sum(c > 0.1)), c.size), flush=True)

        out.append({
            "trial_id": r.trial_id, "config": r.config,
            "probe_nll": r.nll, "probe_n_members": r.n_members,
            "full_nll": score.nll, "full_delta": score.delta,
            "full_delta_ci_lo": score.delta_ci_lo,
            "rank_split": rank_split, "gate_split": gate_split,
            "gate_split_rows": int(gate_idx.size),
            "gate_independent_of_selection": bool(manifest.has_gate_split),
            "n_members": M, "seeds": seeds,
            "gates": battery.to_dict(),
            "contraction": {"param_names": list(contraction.param_names),
                            "values": c.tolist(),
                            "median": float(np.median(c))},
            "information_spectrum_effective_rank":
                float(getattr(spectrum, "effective_rank", float("nan"))),
            "model_dir": model_dir,
        })

    eligible = [o for o in out if o["gates"]["passed"]]
    print("\n[finalists] %d of %d passed every gate" % (len(eligible), len(out)),
          flush=True)

    chosen = None
    if eligible:
        best = min(o["full_nll"] for o in eligible)
        tau = float(args.one_se or 0.0)
        if tau <= 0:
            spread = _seed_spread([r for r in recs
                                   if r.tag.startswith("baseline_seedwin")])
            tau = spread.get("sigma_seed", 0.0)
            tau = 0.0 if not np.isfinite(tau) else tau
        tied = [o for o in eligible if o["full_nll"] <= best + tau]
        # One-SE rule: among statistical ties, take the least complex.
        tied.sort(key=lambda o: (o["config"].get("num_transforms", 0),
                                 o["config"].get("hidden_features", 0),
                                 o["config"].get("num_bins", 0)))
        chosen = tied[0]
        print("[select] %d configuration(s) within one SE (%.4f) of the best; "
              "taking the least complex" % (len(tied), tau), flush=True)
        print("[select] lambda* = %s  (NLL %.4f, gain %.4f)"
              % (json.dumps(chosen["config"], sort_keys=True),
                 chosen["full_nll"], chosen["full_delta"]), flush=True)
    else:
        print("[select] NO configuration passed. Do not relax a gate. Route "
              "the failure:\n"
              "   G1 failed everywhere -> the information-limited branch of "
              "the identity: the embedding, not the estimator, is binding.\n"
              "   G2/G3 failed while G1 passed -> overconfidence; raise M or "
              "investigate the estimator, do not ship.\n"
              "   both failed -> suspect a pipeline defect and re-run the "
              "synthetic benchmark, which exists to distinguish that case.",
              flush=True)

    payload = {"created_utc": _utc(), "delta_min": delta_min, "m_full": M,
               "seeds": seeds, "finalists": out,
               "chosen": chosen, "n_calib": n_calib,
               "n_draws": int(args.n_draws)}
    p = os.path.join(args.out_dir, "finalists.json")
    with open(p, "w", encoding="ascii") as fh:
        json.dump(TL.json_safe(payload), fh, indent=2, sort_keys=True)
    print("[finalists] wrote %s" % p, flush=True)
    return 0 if chosen else 2


# ---------------------------------------------------------------------------
# learning-curve
# ---------------------------------------------------------------------------

def cmd_learning_curve(args: argparse.Namespace) -> int:
    """Queue training-set-size points at a fixed configuration.

    Read against the identity Delta = I(theta; z) - E[KL]: a curve still
    descending at full n says the KL term dominates and more simulations are
    the lever; a flat curve with a gain near zero says the mutual information
    itself is the binding constraint and no amount of tuning will help.
    """
    state = load_campaign(args.out_dir)
    cfg = None
    fp = os.path.join(args.out_dir, "finalists.json")
    if os.path.exists(fp) and not args.config_json:
        with open(fp, "r", encoding="ascii") as fh:
            fin = json.load(fh)
        if fin.get("chosen"):
            cfg = fin["chosen"]["config"]
    if args.config_json:
        cfg = json.loads(args.config_json)
    if cfg is None:
        raise SystemExit("no configuration: run `finalists` first, or pass "
                         "--config-json")

    m_probe = int(state.get("m_probe", 3))
    seeds = [int(s) for s in state.get("seeds", DEFAULT_SEEDS)][:m_probe]
    pending_dir = os.path.join(args.out_dir, PENDING_DIR)
    for frac in args.fractions:
        tag = "frac%0.3f" % float(frac)
        tid = TL.trial_id(cfg, state["split_hash"], state["contract_digest"],
                          m_probe, seeds, tag=tag)
        TL.write_pending({"trial_id": tid, "config": cfg, "tag": tag,
                          "seeds": seeds, "n_members": m_probe,
                          "train_fraction": float(frac)}, pending_dir)
        print("[curve] queued %s at fraction %.3f" % (tid, float(frac)),
              flush=True)
    print("[curve] evaluate each with `evaluate --spec <file>`; every point "
          "is scored on the SAME selection split, so the points are "
          "comparable.", flush=True)
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    """Score the chosen ensemble ONCE on the report split, and write up.

    The report split has never been touched by any decision, which is the
    whole reason its number is the one quoted. Running this command twice
    does not make it two independent numbers.
    """
    import npe_tune_train as TT

    bank, manifest, state = open_bank_and_split(args.out_dir)
    fp = os.path.join(args.out_dir, "finalists.json")
    if not os.path.exists(fp):
        raise SystemExit("run `finalists` first: there is nothing to report")
    with open(fp, "r", encoding="ascii") as fh:
        fin = json.load(fh)
    chosen = fin.get("chosen")
    if not chosen:
        raise SystemExit("no configuration passed the gates; there is nothing "
                         "to freeze. The failure routing printed by "
                         "`finalists` is the result.")

    posteriors, meta = TT.load_members(chosen["model_dir"])
    ens = TT.build_ensemble(posteriors)

    rep_idx = np.asarray(manifest.rep, dtype=np.int64)
    z_rp, th_rp = bank.z[rep_idx], bank.theta[rep_idx]
    member_lp, mix_lp = TT.evaluate_log_probs(posteriors, th_rp, z_rp,
                                              ensemble=ens)
    score = TS.score_from_log_probs(mix_lp, bank.contract.bounds_theta,
                                    member_log_probs=member_lp,
                                    n_members=len(posteriors),
                                    n_boot=int(args.n_boot),
                                    seed=int(args.boot_seed))
    print("\n[report] REPORT SPLIT (touched once): %s" % score.summary(),
          flush=True)
    print("[report] %r split, for comparison: NLL %.4f gain %.4f"
          % (chosen.get("gate_split", "sel"), chosen["full_nll"],
             chosen["full_delta"]), flush=True)

    overlap = None
    if args.real:
        import gate_data
        import npe_diagnostics as D
        arm = gate_data.load_real(args.real)
        res = D.embedding_overlap(bank.z, np.asarray(arm.z),
                                  n_null=int(args.n_null), seed=0)
        overlap = {"mmd": float(res.mmd), "p_value": float(res.p_value),
                   "n_real": int(res.n_real), "n_sim": int(res.n_sim)}
        print("[overlap] real-vs-simulated embedding MMD %.5f, p=%.4f"
              % (overlap["mmd"], overlap["p_value"]), flush=True)
        if overlap["p_value"] < 0.05:
            print("[overlap] REJECTED: the real embeddings do not sit inside "
                  "the simulated cloud. The flow extrapolates at real "
                  "windows and the posterior there is not licensed by any "
                  "diagnostic above, all of which were computed where the "
                  "training distribution lives.", flush=True)

    payload = {"created_utc": _utc(), "git_sha": _git_sha(),
               "chosen": chosen, "report_split": score.to_dict(),
               "split_hash": manifest.hash(),
               "contract_digest": manifest.contract_digest,
               "prior_floor": float(state.get("prior_floor", float("nan"))),
               "embedding_overlap": overlap,
               "conditional_on": {
                   "loader_args": state["loader_args"],
                   "note": ("every score is conditional on the activity "
                            "filter, the theta deduplication and the frozen "
                            "encoder; the estimator targets the FILTERED "
                            "prior predictive and must be reported as such")}}
    p = os.path.join(args.out_dir, "tuning_report.json")
    with open(p, "w", encoding="ascii") as fh:
        json.dump(TL.json_safe(payload), fh, indent=2, sort_keys=True)
    print("[report] wrote %s" % p, flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="npe_tune.py",
        description="Hyperparameter tuning for the NPE ensemble.")
    sub = ap.add_subparsers(dest="command", required=True)

    # -- freeze-split ----------------------------------------------------
    q = sub.add_parser("freeze-split", help="load the bank and freeze the split")
    q.add_argument("--sim", required=True, help="glob for simulated shards")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--activity", default=None, help="activity_*.npz table")
    q.add_argument("--min-rate", type=float, default=0.1,
                   help="MFR floor, Hz/electrode (default 0.1, as the gate)")
    q.add_argument("--max-rate", type=float, default=None)
    q.add_argument("--no-dedup", action="store_true",
                   help="do NOT deduplicate on theta (not recommended)")
    q.add_argument("--max-rows", type=int, default=None)
    q.add_argument("--fractions", type=float, nargs="+",
                   default=[0.8, 0.1, 0.1], metavar="F",
                   help="three fractions TRAIN SEL REP, or four "
                        "TRAIN SEL GATE REP. With four, the search ranks on "
                        "`sel` and the gates run on `gate`, so the gate "
                        "statistic is independent of the quantity that "
                        "selected the finalist "
                        "(HANDOFF_DELTA_MIN_PER_CONFIG_v1 S4.2). Recommended "
                        "four-way value: 0.8 0.07 0.07 0.06")
    q.add_argument("--split-seed", type=int, default=0)
    q.add_argument("--loader-seed", type=int, default=0)
    q.add_argument("--min-groups-per-side", type=int, default=5)
    q.add_argument("--min-rows-per-side", type=int, default=200)
    q.add_argument("--topology-axes", nargs="*",
                   default=list(TD.DEFAULT_TOPOLOGY_AXES))
    q.add_argument("--allow-iid-split", action="store_true",
                   help="accept an ungrouped split when no topology axis is "
                        "present; inflates every held-out score")
    q.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS),
                   help="the ordered seed list; its length is M")
    q.add_argument("--m-probe", type=int, default=3,
                   help="members trained per search evaluation")

    # -- baseline ---------------------------------------------------------
    q = sub.add_parser("baseline", help="queue the default config and controls")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--n-seed-reps", type=int, default=2,
                   help="baseline replicates on disjoint seed windows")
    q.add_argument("--n-control", type=int, default=2,
                   help="shuffled-pairs control runs")

    # -- propose ----------------------------------------------------------
    q = sub.add_parser("propose", help="ask the GP for the next configurations")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--n", type=int, default=8, help="proposals in this batch")
    q.add_argument("--n-initial", type=int, default=20,
                   help="size of the random initial design")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--noise-sd", type=float, default=None,
                   help="measured run-to-run spread, nats/row; its square is "
                        "given to the GP as the observation noise")

    # -- evaluate ---------------------------------------------------------
    q = sub.add_parser("evaluate", help="evaluate ONE configuration")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--spec", default=None, help="a pending trial JSON")
    q.add_argument("--hidden-features", type=int, default=None)
    q.add_argument("--num-transforms", type=int, default=None)
    q.add_argument("--num-bins", type=int, default=None)
    q.add_argument("--learning-rate", type=float, default=None)
    q.add_argument("--batch-size", type=int, default=None)
    q.add_argument("--n-members", type=int, default=None)
    q.add_argument("--tag", default="")
    q.add_argument("--train-fraction", type=float, default=None)
    q.add_argument("--subsample-seed", type=int, default=0)
    q.add_argument("--shuffle-control", action="store_true")
    q.add_argument("--shuffle-seed", type=int, default=1000)
    q.add_argument("--device", default="cpu")
    q.add_argument("--n-boot", type=int, default=2000)
    q.add_argument("--boot-seed", type=int, default=0)
    q.add_argument("--save-model", action="store_true", default=True)
    q.add_argument("--no-save-model", dest="save_model", action="store_false",
                   help="do not persist members; blocks cheap promotion later")
    q.add_argument("--force", action="store_true",
                   help="re-evaluate even if this trial id already exists")

    # -- status -----------------------------------------------------------
    q = sub.add_parser("status", help="ledger, trace and escalation verdict")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--tau-stop", type=float, default=0.01,
                   help="fallback tolerance when sigma_seed is unmeasured")
    q.add_argument("--window", type=int, default=None)
    q.add_argument("--k-sigma", type=float, default=3.0)

    # -- finalists --------------------------------------------------------
    q = sub.add_parser("finalists", help="promote top-K to full M and gate")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--top-k", type=int, default=3)
    q.add_argument("--m-full", type=int, default=None)
    q.add_argument("--n-calib", type=int, default=500)
    q.add_argument("--n-draws", type=int, default=128)
    q.add_argument("--calib-seed", type=int, default=0)
    q.add_argument("--alpha", type=float, default=0.05)
    q.add_argument("--n-forms", type=int, default=4)
    q.add_argument("--k-sigma", type=float, default=3.0)
    q.add_argument("--one-se", type=float, default=None)
    q.add_argument("--device", default="cpu")
    q.add_argument("--n-boot", type=int, default=2000)
    q.add_argument("--boot-seed", type=int, default=0)

    # -- learning-curve ---------------------------------------------------
    q = sub.add_parser("learning-curve", help="queue training-size points")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--fractions", type=float, nargs="*",
                   default=[0.125, 0.25, 0.5, 1.0])
    q.add_argument("--config-json", default=None)

    # -- report -----------------------------------------------------------
    q = sub.add_parser("report", help="score the report split, once")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--real", default=None,
                   help="real-cohort parquet, for the embedding-overlap gate")
    q.add_argument("--n-null", type=int, default=500)
    q.add_argument("--n-boot", type=int, default=2000)
    q.add_argument("--boot-seed", type=int, default=0)
    return ap


COMMANDS = {
    "freeze-split": cmd_freeze_split,
    "baseline": cmd_baseline,
    "propose": cmd_propose,
    "evaluate": cmd_evaluate,
    "status": cmd_status,
    "finalists": cmd_finalists,
    "learning-curve": cmd_learning_curve,
    "report": cmd_report,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
