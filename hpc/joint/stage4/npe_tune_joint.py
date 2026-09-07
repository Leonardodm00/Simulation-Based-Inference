#!/usr/bin/env python3
"""
npe_tune_joint.py -- the Stage 4 search driver over the joint space
(plan S5.1/S5.2), with the per-finalist control test of
HANDOFF_DELTA_MIN_PER_CONFIG_v1.

Relation to the rest of the stack
---------------------------------
Nothing here re-implements anything that already exists:

  search mechanics   npe_tune_search, through joint_space.adapter(campaign)
  ledger             npe_tune_ledger.TrialRecord / trial_id, UNCHANGED
  gates              npe_tune_gates.per_finalist_control_verdicts
  training           joint/stage3/run_joint_arms.py, invoked as a subprocess

The last one is deliberate. `run_joint_arms.py` is the tested Stage 3 runner:
it assembles the backbone, the three-stream batcher, the criteria and the
training loop, and it writes a JSON record per run. Re-implementing that
assembly inside the tuner would double the surface where the bench and the
search could disagree about what an arm IS. So the tuner's job is to turn a
CONFIGURATION into an ARGV, hand it to that runner, and read the record back.
That translation is pure and is the part this module's smoke test covers end
to end; the training itself is covered by the Stage 3 suite.

Subcommands
-----------
  space       print the space, the campaign, and where each range came from
  propose     GP proposal over the campaign's free axes -> pending specs
  argv        show the run_joint_arms argv a configuration resolves to
  evaluate    run one configuration (or --dry-run) and write a TrialRecord
  status      ledger table + escalation verdict against measured sigma_seed
  finalists   the top-k by ranking split, with the gate split named
  controls    per-finalist shuffled controls: plan them, or score them
  report      the shipped configuration and everything required to trust it

Two things this module refuses to do
------------------------------------
1. Choose the campaign or the budget. Both are arguments (decision D4).
2. Ship a configuration that was never gated. `report` fails loudly when the
   control verdicts are missing rather than falling back to the ranking.

Pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", "stage3")),
           os.environ.get("SBI_HPC_DIR", ""),
           os.path.abspath(os.path.join(_HERE, "..", ".."))):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import joint_space as JS  # noqa: E402

PENDING_FILE = "pending"
LEDGER_FILE = "trials"
CONTROLS_FILE = "controls.json"
FINALISTS_FILE = "finalists.json"

# The runner script the tuner drives. Located relative to this file so a
# checkout anywhere works, with an override for an unusual layout.
DEFAULT_RUNNER = os.path.abspath(
    os.path.join(_HERE, "..", "stage3", "run_joint_arms.py"))


# ---------------------------------------------------------------------------
# Configuration -> argv. This is the whole integration surface, so it is
# pure, total, and tested.
# ---------------------------------------------------------------------------

# Which run_joint_arms flag each searched axis drives. An axis absent from
# this table is NOT passed to the runner, and that must be a deliberate,
# stated decision rather than an oversight -- `unmapped_axes()` reports them
# and `build_argv` refuses to run when any unmapped axis is FREE.
AXIS_TO_FLAG: Dict[str, str] = {
    "depth_exponent": "--depth-exponent",
    "width_multiplier": "--width-multiplier",
    "block_family": "--block-family",
    "embedding_size": "--embedding-size",
    "head_fusion": "--head-fusion",
    "dropout": "--dropout",
    "warmup_frac_rep": "--warmup-frac-rep",
    "n_posterior_draws": "--n-posterior-draws",
    "hidden_features": "--hidden-features",
    "num_transforms": "--num-transforms",
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "batch_size_npe": "--b-sim",
    "one_minus_beta1": "--one-minus-beta1",
    "loss_type": "--loss-type",
    "mining_strategy": "--mining-strategy",
    "margin": "--margin",
    "angular_alpha_deg": "--angular-alpha-deg",
    "lambda_sep": "--lambda-sep",
}

# Axes the runner cannot take as a flag today, with the reason. Keeping the
# reason next to the name is the point: a silent gap here is a search over an
# axis that never reaches the trainer, which looks like "the axis does not
# matter" in the partial dependence.
DERIVED: Dict[str, str] = {
    "dsn_on": "gates --lambda-dsn: 0 when off, 10 ** log10_lambda_dsn when on",
    "rep_on": "gates --lambda-rep: 0 when off, 10 ** log10_lambda_rep when on",
    "log10_lambda_dsn": "--lambda-dsn = 10 ** x, when dsn_on = 1",
    "log10_lambda_rep": "--lambda-rep = 10 ** x, when rep_on = 1",
}

# Axes that genuinely do NOT reach the trainer today, with the reason. A
# silent gap here is a search over an axis that never reaches training, which
# shows up in the partial dependence as "this axis does not matter" -- a null
# result that is confidently wrong rather than merely uninformative.
UNREACHABLE: Dict[str, str] = {}


def unmapped_free_axes(campaign_name: str) -> List[str]:
    """FREE axes of a campaign that do not reach the runner.

    A pinned axis being unmapped is harmless -- it is not being searched. A
    FREE one is a hole in the experiment.
    """
    return [a for a in JS.free_axes(campaign_name) if a in UNREACHABLE]


def build_argv(config: Dict[str, Any],
               arm: str,
               sim_shards: str,
               out_dir: str,
               seed: int,
               real_shards: Optional[str] = None,
               epochs: int = 10,
               steps_per_epoch: int = 25,
               runner: str = DEFAULT_RUNNER,
               python: Optional[str] = None,
               dsn_main_dir: Optional[str] = None,
               sbi_hpc_dir: Optional[str] = None,
               dry_run: bool = False,
               spec_fixed: Optional[Dict[str, Any]] = None,
               extra: Optional[Sequence[str]] = None) -> List[str]:
    """Turn one joint configuration into a run_joint_arms command line.

    The two loss weights are the only non-trivial conversion: the space
    searches log10(lambda) with an on/off switch, while the runner takes the
    weight itself. `dsn_on = 0` therefore becomes `--lambda-dsn 0` and NOT
    `10 ** log10_lambda_dsn`, which is the whole point of the switch -- and
    is why the canonicalisation pins log10_lambda_dsn to a harmless value
    rather than leaving it free to imply a weight.

    Every value is formatted with repr-grade precision: passing a rounded
    learning rate would make the trial_id disagree with what actually ran.
    """
    missing = [k for k in JS.JOINT_KNOB_ORDER if k not in config]
    if missing:
        raise KeyError("config is missing axes: %s" % ", ".join(missing))

    # Refuse to build a command line that silently drops a searched axis.
    # Only axes at a NON-canonical value matter: under dsn_on = 0 the loss
    # axes are pinned by canonicalisation and dropping them changes nothing,
    # which is exactly why S-A1 and S-A5 can run today and S-A2 cannot.
    dropped = []
    for axis in UNREACHABLE:
        canon = JS.INACTIVE_CANONICAL.get(axis)
        if canon is None:
            continue
        v = config[axis]
        same = (str(v) == str(canon) if axis in JS._STR_AXES
                else abs(float(v) - float(canon)) <= 1e-12)
        if not same:
            dropped.append("%s=%r (canonical %r)" % (axis, v, canon))
    if dropped:
        raise ValueError(
            "this configuration sets %d axis/axes that run_joint_arms cannot "
            "receive, so training would silently use different values: %s. "
            "Add the flags to run_joint_arms.py before evaluating this "
            "campaign." % (len(dropped), "; ".join(dropped)))

    argv = [python or sys.executable, runner,
            "--arm", str(arm),
            "--sim-shards", str(sim_shards),
            "--out-dir", str(out_dir),
            "--seed", str(int(seed)),
            "--epochs", str(int(epochs)),
            "--steps-per-epoch", str(int(steps_per_epoch))]
    if real_shards:
        argv += ["--real-shards", str(real_shards)]

    for axis, flag in AXIS_TO_FLAG.items():
        v = config[axis]
        if axis in JS._STR_AXES:
            argv += [flag, str(v)]
        elif axis in JS._INT_AXES:
            argv += [flag, str(int(v))]
        else:
            argv += [flag, repr(float(v))]

    # strict_semihard is FIXED by S5.1, not searched, but the runner must be
    # told which value the space declares -- leaving it to the runner's own
    # default would let the two disagree silently, and the legality
    # projection's result depends on it.
    if spec_fixed is not None and "strict_semihard" in spec_fixed:
        argv += ["--strict-semihard", str(int(spec_fixed["strict_semihard"]))]

    lam_dsn = (10.0 ** float(config["log10_lambda_dsn"])
               if int(config["dsn_on"]) else 0.0)
    lam_rep = (10.0 ** float(config["log10_lambda_rep"])
               if int(config["rep_on"]) else 0.0)
    argv += ["--lambda-dsn", repr(lam_dsn), "--lambda-rep", repr(lam_rep)]

    if dsn_main_dir:
        argv += ["--dsn-main-dir", str(dsn_main_dir)]
    if sbi_hpc_dir:
        argv += ["--sbi-hpc-dir", str(sbi_hpc_dir)]
    if dry_run:
        argv += ["--dry-run"]
    if extra:
        argv += [str(x) for x in extra]
    return argv


def arm_for_config(config: Dict[str, Any]) -> str:
    """Which Stage 3 arm a configuration corresponds to.

    The arms are named experiments and the campaigns search inside them, so
    the mapping is on the two switches alone. A configuration with both
    terms on has no Stage 3 arm; that is a real gap, reported rather than
    silently mapped onto A2.
    """
    dsn, rep = int(config["dsn_on"]), int(config["rep_on"])
    if config.get("shuffle_pairs", 0):
        return "shuffled"
    if dsn and rep:
        raise ValueError(
            "no Stage 3 arm trains both the DSN and the replicate term "
            "(dsn_on = rep_on = 1). Campaign S-A25 needs a new arm in "
            "run_joint_arms before it can be evaluated; S-A1, S-A2 and S-A5 "
            "map to A1, A2 and A5.")
    if dsn:
        return "A2"
    if rep:
        return "A5"
    return "A1"


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def _ledger_dir(root: str, campaign_name: str) -> str:
    return os.path.join(root, campaign_name, LEDGER_FILE)


def _pending_dir(root: str, campaign_name: str) -> str:
    return os.path.join(root, campaign_name, PENDING_FILE)


def load_observations(root: str, campaign_name: str,
                      spec: JS.JointSpaceSpec,
                      objective: str = "nll") -> List[Tuple[Dict[str, Any], float]]:
    """Every completed trial of this campaign, as (config, score) pairs.

    Records whose score is absent or non-finite are DROPPED, not imputed: a
    failed trial carries no information about the objective, and telling the
    GP a made-up value at that point is worse than telling it nothing.
    Records carrying a `tag` are dropped too -- a learning-curve point or a
    control ran a different question and must not enter the surrogate.
    """
    out: List[Tuple[Dict[str, Any], float]] = []
    d = _ledger_dir(root, campaign_name)
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(path, "r", encoding="ascii") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if rec.get("tag"):
            continue
        if str(rec.get("status", "ok")) != "ok":
            continue
        y = rec.get(objective)
        cfg = rec.get("config") or {}
        if y is None or not np.isfinite(float(y)):
            continue
        if any(k not in cfg for k in JS.JOINT_KNOB_ORDER):
            continue
        out.append((cfg, float(y)))
    return out


def _deltas_by_key(root: str, campaign_name: str) -> Dict[str, float]:
    """Delta = L0 - L for every completed, untagged trial, keyed by config."""
    out: Dict[str, float] = {}
    for path in sorted(glob.glob(os.path.join(_ledger_dir(root, campaign_name),
                                              "*.json"))):
        try:
            with open(path, "r", encoding="ascii") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if rec.get("tag") or rec.get("status") != "ok":
            continue
        cfg, d = rec.get("config") or {}, rec.get("delta")
        if d is None or not np.isfinite(float(d)):
            continue
        if all(k in cfg for k in JS.JOINT_KNOB_ORDER):
            out[JS.config_key(cfg)] = float(d)
    return out


def write_record(root: str, campaign_name: str, record: Dict[str, Any]) -> str:
    d = _ledger_dir(root, campaign_name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "%s.json" % record["trial_id"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)          # atomic: a killed job leaves no half file
    return path


def _trial_id(config: Dict[str, Any], split_hash: str, contract_digest: str,
              seeds: Sequence[int], tag: str = "") -> str:
    import npe_tune_ledger as TL
    return TL.trial_id(config=config, split_hash=split_hash,
                       contract_digest=contract_digest, n_members=1,
                       seeds=list(seeds), tag=tag)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _spec_from_args(args) -> JS.JointSpaceSpec:
    return JS.default_joint_space(p=int(args.p),
                                  embedding_dim=int(args.embedding_dim),
                                  d_theta=int(args.d_theta),
                                  n_train=(int(args.n_train)
                                           if args.n_train else None))


def cmd_space(args) -> int:
    spec = _spec_from_args(args)
    print(JS.provenance_report(spec, args.campaign))
    holes = unmapped_free_axes(args.campaign)
    print("")
    print("axes reaching the runner by conversion rather than by a flag:")
    for a, why in sorted(DERIVED.items()):
        print("  %-20s %s" % (a, why))
    print("")
    if holes:
        print("BLOCKING: %d FREE axis/axes do not reach the trainer at all, "
              "so the search would move them with no effect on training and "
              "the partial dependence would report them as irrelevant:"
              % len(holes))
        for a in holes:
            print("  %-20s %s" % (a, UNREACHABLE[a]))
        print("")
        print("Do not spend budget on campaign %s until either the flags "
              "exist in run_joint_arms.py or these axes are pinned."
              % args.campaign)
    else:
        print("every free axis of campaign %s reaches the trainer."
              % args.campaign)
    return 1 if holes else 0


def cmd_propose(args) -> int:
    import npe_tune_search as TSR

    spec = _spec_from_args(args)
    ad = JS.adapter(args.campaign, spec)
    obs = load_observations(args.results_dir, args.campaign, spec)
    pend = _pending_dir(args.results_dir, args.campaign)
    os.makedirs(pend, exist_ok=True)
    exclude = []
    for path in sorted(glob.glob(os.path.join(pend, "*.json"))):
        try:
            with open(path, "r", encoding="ascii") as fh:
                exclude.append(json.load(fh).get("config", {}))
        except (OSError, ValueError):
            continue

    props = TSR.propose(spec, observations=obs, n_points=int(args.n_points),
                        n_initial_points=int(args.n_initial_points),
                        seed=int(args.seed),
                        noise=(float(args.sigma_seed) ** 2
                               if args.sigma_seed else None),
                        exclude=exclude, adapter=ad)

    # The shared propose() ACCEPTS duplicates once it has exhausted its
    # resampling budget, on the grounds that "an accepted duplicate is merely
    # a wasted job, which the trial-id check upstream will detect". This IS
    # that upstream check. Without it a duplicate would be written to the
    # same trial_id filename, overwrite the pending spec already there, and
    # the round would silently produce nothing while reporting success.
    known = set()
    for path in sorted(glob.glob(os.path.join(pend, "*.json"))):
        known.add(os.path.basename(path)[:-5])
    known |= {os.path.basename(p)[:-5]
              for p in glob.glob(os.path.join(
                  _ledger_dir(args.results_dir, args.campaign), "*.json"))}

    written, dup = [], 0
    for cfg in props:
        tid = _trial_id(cfg, args.split_hash, args.contract_digest,
                        [int(args.seed)])
        if tid in known:
            dup += 1
            continue
        known.add(tid)
        path = os.path.join(pend, "%s.json" % tid)
        with open(path, "w", encoding="ascii") as fh:
            json.dump({"trial_id": tid, "campaign": args.campaign,
                       "config": cfg}, fh, indent=2, sort_keys=True)
        written.append((tid, cfg))

    print("[propose] campaign %s: %d observation(s), %d already pending, "
          "%d proposed, %d NEW"
          % (args.campaign, len(obs), len(exclude), len(props), len(written)),
          flush=True)
    for tid, cfg in written:
        print("  %s  %s" % (tid, JS.config_key(cfg)[:110]))
    if dup:
        print("[propose] %d proposal(s) duplicated a trial that is already "
              "pending or already evaluated, and were DROPPED rather than "
              "overwriting it." % dup, flush=True)
    if len(written) < int(args.n_points):
        print("[propose] asked for %d, wrote %d. Either the surrogate has "
              "converged on a region it keeps re-proposing, or this seed has "
              "been used before: change --seed for the next round."
              % (int(args.n_points), len(written)), flush=True)
    return 0


def cmd_argv(args) -> int:
    spec = _spec_from_args(args)
    cfg = _load_config(args, spec)
    argv = build_argv(cfg, arm=arm_for_config(cfg), sim_shards=args.sim_shards,
                      out_dir=args.out_dir, seed=int(args.seed),
                      real_shards=args.real_shards, epochs=int(args.epochs),
                      steps_per_epoch=int(args.steps_per_epoch),
                      runner=args.runner, dsn_main_dir=args.dsn_main_dir,
                      sbi_hpc_dir=args.sbi_hpc_dir, dry_run=args.dry_run,
                      spec_fixed=spec.fixed)
    print(" ".join(argv))
    return 0


def _load_config(args, spec: JS.JointSpaceSpec) -> Dict[str, Any]:
    """A configuration from --config-json, --pending-id, or a random draw."""
    if args.config_json:
        with open(args.config_json, "r", encoding="ascii") as fh:
            d = json.load(fh)
        cfg = d.get("config", d)
        return JS.canonicalise_config(cfg, spec=spec)
    if getattr(args, "pending_id", None):
        path = os.path.join(_pending_dir(args.results_dir, args.campaign),
                            "%s.json" % args.pending_id)
        with open(path, "r", encoding="ascii") as fh:
            return JS.canonicalise_config(json.load(fh)["config"], spec=spec)
    raise SystemExit("give --config-json or --pending-id")


def cmd_evaluate(args) -> int:
    spec = _spec_from_args(args)
    cfg = _load_config(args, spec)
    arm = arm_for_config(cfg)
    tid = _trial_id(cfg, args.split_hash, args.contract_digest,
                    [int(args.seed)], tag=args.tag)
    out_dir = os.path.join(args.out_dir, tid)
    argv = build_argv(cfg, arm=arm, sim_shards=args.sim_shards,
                      out_dir=out_dir, seed=int(args.seed),
                      real_shards=args.real_shards, epochs=int(args.epochs),
                      steps_per_epoch=int(args.steps_per_epoch),
                      runner=args.runner, dsn_main_dir=args.dsn_main_dir,
                      sbi_hpc_dir=args.sbi_hpc_dir, dry_run=args.dry_run,
                      spec_fixed=spec.fixed)
    print("[evaluate] %s  arm=%s  tag=%r" % (tid, arm, args.tag), flush=True)
    print("[evaluate] %s" % " ".join(argv), flush=True)
    if args.dry_run:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    proc = subprocess.run(argv)
    rec: Dict[str, Any] = {"trial_id": tid, "tag": args.tag, "config": cfg,
                           "campaign": args.campaign, "arm": arm,
                           "seeds": [int(args.seed)],
                           "split_hash": args.split_hash,
                           "contract_digest": args.contract_digest,
                           "argv": argv, "status": "ok", "error": ""}
    if proc.returncode != 0:
        rec.update(status="failed",
                   error="runner exited %d" % proc.returncode)
        write_record(args.results_dir, args.campaign, rec)
        print("[evaluate] FAILED, recorded as status=failed", flush=True)
        return proc.returncode

    # Read back whatever the runner wrote, without assuming its filename.
    found = sorted(glob.glob(os.path.join(out_dir, "*_seed%d.json"
                                          % int(args.seed))))
    if not found:
        rec.update(status="failed", error="runner wrote no record in %s"
                   % out_dir)
        write_record(args.results_dir, args.campaign, rec)
        return 2
    with open(found[0], "r", encoding="ascii") as fh:
        run = json.load(fh)
    for k in ("L", "L0", "delta_hat", "delta_hat_pseudo_real",
              "L_pseudo_real", "r_eff", "split_hash", "cluster_ari",
              "p_eff_spectrum"):
        if k in run:
            rec[k] = run[k]
    # The objective is the held-out NLL, and it is named `nll` in the ledger
    # because that is what the shared search machinery reads.
    if "L" in run:
        rec["nll"] = run["L"]
    if "delta_hat" in run:
        rec["delta"] = run["delta_hat"]
    rec["runner_record"] = found[0]
    path = write_record(args.results_dir, args.campaign, rec)
    pend = os.path.join(_pending_dir(args.results_dir, args.campaign),
                        "%s.json" % tid)
    if os.path.exists(pend):
        os.remove(pend)
    print("[evaluate] nll=%s  wrote %s" % (rec.get("nll"), path), flush=True)
    return 0


def cmd_status(args) -> int:
    import npe_tune_search as TSR

    spec = _spec_from_args(args)
    ad = JS.adapter(args.campaign, spec)
    obs = load_observations(args.results_dir, args.campaign, spec)
    print("[status] campaign %s: %d completed trial(s)"
          % (args.campaign, len(obs)))
    if not obs:
        return 0
    order = np.argsort([y for _, y in obs])
    print("")
    print("| rank | nll | dsn_on | rep_on | key |")
    print("|---|---|---|---|---|")
    for r, i in enumerate(order[:int(args.top)]):
        cfg, y = obs[i]
        print("| %d | %.4f | %d | %d | %s |"
              % (r + 1, y, int(cfg["dsn_on"]), int(cfg["rep_on"]),
                 JS.config_key(cfg)[:90]))
    v = TSR.escalation_verdict(obs, spec,
                               tau_stop=float(args.sigma_seed or 0.0),
                               adapter=ad)
    print("")
    print(v.summary())
    return 0


def cmd_finalists(args) -> int:
    spec = _spec_from_args(args)
    obs = load_observations(args.results_dir, args.campaign, spec)
    if not obs:
        raise SystemExit("no completed trials to rank")
    order = np.argsort([y for _, y in obs])[:int(args.top_k)]
    deltas = _deltas_by_key(args.results_dir, args.campaign)
    fin = []
    for r, i in enumerate(order):
        cfg = obs[i][0]
        entry = {"rank": r + 1, "nll_rank_split": float(obs[i][1]),
                 "config": cfg,
                 "trial_id": _trial_id(cfg, args.split_hash,
                                       args.contract_digest,
                                       [int(args.seed)])}
        d = deltas.get(JS.config_key(cfg))
        if d is not None:
            entry["delta"] = float(d)
        fin.append(entry)
    payload = {"campaign": args.campaign, "K": len(fin),
               "rank_split": args.rank_split, "gate_split": args.gate_split,
               "finalists": fin}
    os.makedirs(os.path.join(args.results_dir, args.campaign), exist_ok=True)
    path = os.path.join(args.results_dir, args.campaign, FINALISTS_FILE)
    with open(path, "w", encoding="ascii") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("[finalists] K=%d ranked on %r; the gate will run on %r"
          % (len(fin), args.rank_split, args.gate_split))
    if args.rank_split == args.gate_split:
        print("[warn] ranking and gating on the SAME split. The finalists "
              "are the argmax of that split over the whole search, so a null "
              "calibrated for a single configuration understates the "
              "false-pass rate and Holm does not repair it. Re-freeze a "
              "four-way split (S4.2).")
    for f in fin:
        print("  %d  nll=%.4f  %s" % (f["rank"], f["nll_rank_split"],
                                      f["trial_id"]))
    print("wrote %s" % path)
    return 0


def cmd_controls(args) -> int:
    """Plan the per-finalist controls, or score them once they have run."""
    path = os.path.join(args.results_dir, args.campaign, FINALISTS_FILE)
    if not os.path.exists(path):
        raise SystemExit("run `finalists` first: %s not found" % path)
    with open(path, "r", encoding="ascii") as fh:
        fin = json.load(fh)

    if args.plan:
        n = 0
        for f in fin["finalists"]:
            cand = dict(f["config"])
            cand["shuffle_pairs"] = 0
            for s in range(int(args.n_control)):
                ctrl = JS.control_config_from(cand, permutation_seed=1000 + s)
                # The assertion is the point of the plan step: if the control
                # is not byte-identical to its candidate but for the shuffle,
                # the floor it measures is not that candidate's floor.
                JS.assert_control_identity(cand, ctrl)
                tid = _trial_id(ctrl, args.split_hash, args.contract_digest,
                                [int(args.seed)], tag="control")
                out = os.path.join(args.results_dir, args.campaign,
                                   "control_specs")
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "%s.json" % tid), "w",
                          encoding="ascii") as fh:
                    json.dump({"trial_id": tid, "campaign": args.campaign,
                               "finalist_rank": f["rank"],
                               "finalist_trial_id": f["trial_id"],
                               "control_seed": 1000 + s, "tag": "control",
                               "config": ctrl}, fh, indent=2, sort_keys=True)
                n += 1
        print("[controls] planned %d control run(s): %d finalist(s) x %d seeds"
              % (n, len(fin["finalists"]), int(args.n_control)))
        print("[controls] recipe identity asserted for every one of them")
        print("[controls] evaluate each with `evaluate --pending-id <id> "
              "--tag control`, then re-run `controls --score`")
        return 0

    # --score: collect each finalist's own controls and test.
    import npe_tune_gates as TG

    by_finalist: Dict[str, List[float]] = {}
    d = _ledger_dir(args.results_dir, args.campaign)
    specs_dir = os.path.join(args.results_dir, args.campaign, "control_specs")
    owner = {}
    for p in sorted(glob.glob(os.path.join(specs_dir, "*.json"))):
        with open(p, "r", encoding="ascii") as fh:
            s = json.load(fh)
        owner[s["trial_id"]] = s["finalist_trial_id"]
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(p, "r", encoding="ascii") as fh:
            rec = json.load(fh)
        if rec.get("tag") != "control" or rec.get("status") != "ok":
            continue
        fid = owner.get(rec["trial_id"])
        if fid is None or "delta" not in rec:
            continue
        by_finalist.setdefault(fid, []).append(float(rec["delta"]))

    cands = []
    missing_delta = []
    for f in fin["finalists"]:
        ctrls = by_finalist.get(f["trial_id"], [])
        if "delta_gate_split" in f:
            d = float(f["delta_gate_split"])
        elif "delta" in f:
            d = float(f["delta"])
        else:
            missing_delta.append(f["trial_id"])
            d = float("nan")
        cands.append({"name": "rank%d:%s" % (f["rank"], f["trial_id"][:8]),
                      "delta": d, "control_deltas": ctrls,
                      "n_seeds": int(args.n_seeds)})
    if missing_delta:
        # Falling back to the ranking NLL here would silently compare an NLL
        # against a distribution of Delta = L0 - L. They are different
        # quantities and the resulting p-value would be meaningless.
        raise SystemExit(
            "%d finalist(s) carry no gate-split Delta: %s. The control test "
            "compares Delta against a distribution of Delta; there is no "
            "substitute for it, and the ranking NLL is NOT one. Re-run "
            "`finalists` after scoring the finalists on the gate split."
            % (len(missing_delta), ", ".join(missing_delta)))
    verdicts = TG.per_finalist_control_verdicts(
        cands, alpha=float(args.alpha), floor=float(args.floor),
        delta_min_provisional=float(args.delta_min_provisional))
    print(TG.format_control_verdicts(verdicts, alpha=float(args.alpha)))
    out = {"campaign": args.campaign, "alpha": float(args.alpha),
           "floor": float(args.floor), "n_seeds": int(args.n_seeds),
           "rank_split": fin.get("rank_split"),
           "gate_split": fin.get("gate_split"),
           "verdicts": [v.to_dict() for v in verdicts]}
    path = os.path.join(args.results_dir, args.campaign, CONTROLS_FILE)
    with open(path, "w", encoding="ascii") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    survivors = [v for v in verdicts if v.rejected]
    if not survivors:
        print("\nNO finalist rejected its own noise floor. That is the "
              "result. Do NOT extend K after seeing it: extending K post hoc "
              "reintroduces exactly the multiplicity Holm was applied to "
              "remove.")
    else:
        print("\nshipped configuration: the BEST-RANKED survivor, %s (not the "
              "smallest p)" % survivors[0].name)
    print("wrote %s" % path)
    return 0


def cmd_report(args) -> int:
    path = os.path.join(args.results_dir, args.campaign, CONTROLS_FILE)
    if not os.path.exists(path):
        raise SystemExit(
            "no control verdicts at %s. A configuration that was never "
            "gated does not get shipped, so this command will not fall back "
            "to the ranking." % path)
    with open(path, "r", encoding="ascii") as fh:
        c = json.load(fh)
    print("campaign %s: ranked on %r, gated on %r, alpha=%.3f"
          % (c["campaign"], c.get("rank_split"), c.get("gate_split"),
             c["alpha"]))
    print("ALL %d finalist(s) are reported below, passes and failures."
          % len(c["verdicts"]))
    print("")
    print("| finalist | Delta_hat | ctrl mean | ctrl sd | n_ctrl | raw p | "
          "Holm p | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for v in c["verdicts"]:
        print("| %s | %.4f | %.4f | %.4f | %d | %.4g | %.4g | %s |"
              % (v["name"], v["delta"], v["control_mean"], v["control_sd"],
                 v["n_control"], v["pvalue"], v["adjusted_pvalue"],
                 "REJECT null" if v["rejected"] else "not rejected"))
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(q, need_shards=False):
        q.add_argument("--campaign", default="S-A1",
                       help="one of: %s" % ", ".join(sorted(JS.CAMPAIGNS)))
        q.add_argument("--results-dir", default="tune_joint")
        q.add_argument("--p", type=int, default=26)
        q.add_argument("--embedding-dim", type=int, default=12)
        q.add_argument("--d-theta", type=int, default=26)
        q.add_argument("--n-train", type=int, default=None)
        q.add_argument("--split-hash", default="")
        q.add_argument("--contract-digest", default="")
        q.add_argument("--seed", type=int, default=0)
        if need_shards:
            q.add_argument("--sim-shards", required=True)
            q.add_argument("--real-shards", default=None)
            q.add_argument("--out-dir", default="runs_joint")
            q.add_argument("--epochs", type=int, default=10)
            q.add_argument("--steps-per-epoch", type=int, default=25)
            q.add_argument("--runner", default=DEFAULT_RUNNER)
            q.add_argument("--dsn-main-dir", default=None)
            q.add_argument("--sbi-hpc-dir", default=None)
            q.add_argument("--dry-run", action="store_true")
        return q

    common(sub.add_parser("space"))

    q = common(sub.add_parser("propose"))
    q.add_argument("--n-points", type=int, default=8)
    q.add_argument("--n-initial-points", type=int, default=12)
    q.add_argument("--sigma-seed", type=float, default=None,
                   help="measured across-seed spread; its SQUARE is given to "
                        "the GP as the observation noise variance")

    q = common(sub.add_parser("argv"), need_shards=True)
    q.add_argument("--config-json", default=None)
    q.add_argument("--pending-id", default=None)

    q = common(sub.add_parser("evaluate"), need_shards=True)
    q.add_argument("--config-json", default=None)
    q.add_argument("--pending-id", default=None)
    q.add_argument("--tag", default="")

    q = common(sub.add_parser("status"))
    q.add_argument("--top", type=int, default=10)
    q.add_argument("--sigma-seed", type=float, default=None)

    q = common(sub.add_parser("finalists"))
    q.add_argument("--top-k", type=int, default=3)
    q.add_argument("--rank-split", default="sel")
    q.add_argument("--gate-split", default="gate")

    q = common(sub.add_parser("controls"))
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--score", action="store_true")
    q.add_argument("--n-control", type=int, default=5)
    q.add_argument("--n-seeds", type=int, default=1)
    q.add_argument("--alpha", type=float, default=0.05)
    q.add_argument("--floor", type=float, default=0.0)
    q.add_argument("--delta-min-provisional", type=float, default=float("nan"))

    common(sub.add_parser("report"))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "campaign", None) is not None \
            and args.campaign not in JS.CAMPAIGNS:
        # A typo in --campaign must read as a typo, not as a traceback that
        # a shell guard will then misreport as something else entirely.
        raise SystemExit("unknown campaign %r; known: %s"
                         % (args.campaign, ", ".join(sorted(JS.CAMPAIGNS))))
    return {"space": cmd_space, "propose": cmd_propose, "argv": cmd_argv,
            "evaluate": cmd_evaluate, "status": cmd_status,
            "finalists": cmd_finalists, "controls": cmd_controls,
            "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
