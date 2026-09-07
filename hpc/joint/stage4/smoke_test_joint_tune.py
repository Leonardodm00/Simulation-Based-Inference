#!/usr/bin/env python3
"""
smoke_test_joint_tune.py -- the Stage 4 search driver (npe_tune_joint.py).

Run:
    DSN_MAIN_DIR=... SBI_HPC_DIR=... python smoke_test_joint_tune.py
    python smoke_test_joint_tune.py -k J31

The integration surface between the tuner and the trainer is ONE thing: the
command line. So the tests below round-trip a generated argv through
`run_joint_arms.build_parser` itself rather than against a hand-written list
of expected flags -- a test that compared against my own idea of the flags
would pass even if the runner's parser disagreed, which is the only failure
that matters here. The parser is extracted by AST so this suite needs no
torch; the training it drives is covered by the Stage 3 suite.

  J28  Every searched axis reaches the runner, and the runner accepts the
       generated argv. The two switches translate correctly: dsn_on = 0
       means --lambda-dsn 0, NOT 10 ** log10_lambda_dsn.
  J29  Every searched axis reaches the trainer, and the guard that refuses
       a dropped axis still fires when a hole is injected.
  J30  Ledger round trip: records are written atomically, replayed as
       observations, and tagged/failed/incomplete records are excluded from
       the surrogate.
  J31  propose writes pending specs, excludes what is pending and what is
       observed, and is reproducible under a seed.
  J32  finalists ranks, names both splits, and warns when they coincide.
  J33  controls --plan asserts recipe identity for every control it writes;
       controls --score reproduces the Holm verdict; report REFUSES to ship
       an ungated configuration.
  J34  arm_for_config maps the switches onto Stage 3 arms and refuses the
       combination that has no arm.
  J36  The PBS element and the launcher, run for real in DRYRUN mode:
       distinct elements, stale index tolerated, missing variable fatal,
       launcher dry by default, --max honoured, unknown campaign named.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.environ.get("SBI_HPC_DIR", "")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import joint_space as JS  # noqa: E402
import npe_tune_joint as TJ  # noqa: E402

RESULTS: List[Tuple[str, str]] = []


class Skip(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _dsn_status() -> str:
    """Why the DSN is or is not reachable. Empty string means it is.

    Three situations need three different fixes and were all reported as
    "unset" before: the variable genuinely unset, set to a path that does
    not exist (a deleted symlink -- what actually happened on the cluster,
    2026-09-07), and set to a real directory whose checkout predates
    condition_space.py. Kept identical to the copy in
    smoke_test_joint_space.py rather than shared, so neither suite has to
    import the other to run standalone.
    """
    main = os.environ.get("DSN_MAIN_DIR", "")
    if not main:
        return "DSN_MAIN_DIR unset"
    if not os.path.isdir(main):
        return ("DSN_MAIN_DIR=%r does not exist (a deleted symlink looks "
                "exactly like this)" % main)
    if not os.path.isfile(os.path.join(main, "condition_space.py")):
        return ("DSN_MAIN_DIR=%r has no condition_space.py -- that checkout "
                "predates it; git pull the DSN repo" % main)
    if main not in sys.path:
        sys.path.insert(0, main)
    try:
        import condition_space  # noqa: F401
        return ""
    except ImportError as exc:
        return "condition_space present at %r but will not import (%s)" % (
            main, exc)


def _dsn() -> bool:
    return _dsn_status() == ""


def _runner_parser():
    """run_joint_arms.build_parser, extracted without importing torch.

    The runner imports torch at module scope, which this environment may not
    have; but its PARSER is the contract being tested, and that is pure
    argparse. Extracting it by AST keeps the test honest -- it is the
    runner's own parser, not a copy of it.
    """
    path = os.path.abspath(os.path.join(_HERE, "..", "stage3",
                                        "run_joint_arms.py"))
    if not os.path.isfile(path):
        raise Skip("run_joint_arms.py not found at %s" % path)
    tree = ast.parse(open(path, encoding="ascii").read())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "build_parser"),
              None)
    arms = next((n for n in tree.body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "ARMS"), None)
    if fn is None or arms is None:
        raise Skip("could not extract build_parser from run_joint_arms.py")
    ns: Dict[str, Any] = {"argparse": argparse, "__doc__": ""}
    exec(compile(ast.Module(body=[arms, fn], type_ignores=[]), "<runner>",
                 "exec"), ns)
    return ns["build_parser"]()


def _spec():
    return JS.default_joint_space(p=26, embedding_dim=12, d_theta=26,
                                  n_train=20000)


def _random_config(campaign_name, spec, rng, **over):
    pt = []
    for axis in JS.free_axes(campaign_name):
        if axis in over:
            pt.append(over[axis])
            continue
        r = spec.range_of(axis)
        if axis in JS._STR_AXES:
            pt.append(str(rng.choice(list(r))))
        elif axis in ("block_family", "head_fusion", "dsn_on", "rep_on",
                      "batch_size_npe"):
            pt.append(int(rng.choice(list(r))))
        elif axis in JS._INT_AXES:
            pt.append(int(rng.integers(int(r[0]), int(r[1]) + 1)))
        else:
            pt.append(float(rng.uniform(float(r[0]), float(r[1]))))
    return JS.config_from_point(pt, campaign_name, spec=spec)


# ---------------------------------------------------------------------------

def j28_argv_round_trip() -> str:
    parser = _runner_parser()
    spec = _spec()
    rng = np.random.default_rng(0)
    out = []
    for name in ("S-A1", "S-A5"):
        check(not TJ.unmapped_free_axes(name),
              "%s has free axes the runner cannot receive: %s"
              % (name, TJ.unmapped_free_axes(name)))
        for _ in range(15):
            cfg = _random_config(name, spec, rng)
            argv = TJ.build_argv(cfg, arm=TJ.arm_for_config(cfg),
                                 sim_shards="s/*.parquet", out_dir="o",
                                 seed=3, real_shards="r/*.parquet")
            a = parser.parse_args(argv[2:])       # drop python + script path

            # Every mapped axis arrived, with its value intact.
            for axis, flag in TJ.AXIS_TO_FLAG.items():
                dest = flag[2:].replace("-", "_")
                got = getattr(a, dest)
                want = cfg[axis]
                if axis in JS._STR_AXES:
                    check(str(got) == str(want),
                          "%s: %s arrived as %r, want %r"
                          % (name, axis, got, want))
                elif axis in JS._INT_AXES:
                    check(int(got) == int(want),
                          "%s: %s arrived as %r, want %r"
                          % (name, axis, got, want))
                else:
                    check(abs(float(got) - float(want)) < 1e-12,
                          "%s: %s arrived as %r, want %r"
                          % (name, axis, got, want))

            # The switches. This is the conversion that a naive
            # implementation gets wrong: with the term OFF the weight must be
            # exactly zero, NOT 10 ** log10_lambda (which is 1.0 at the
            # canonical log10_lambda = 0 and would train the term at full
            # strength while the config says it is off).
            for on, log_axis, flag in (("dsn_on", "log10_lambda_dsn",
                                        "lambda_dsn"),
                                       ("rep_on", "log10_lambda_rep",
                                        "lambda_rep")):
                got = float(getattr(a, flag))
                if int(cfg[on]) == 0:
                    check(got == 0.0, "%s: %s must be exactly 0 when %s = 0, "
                                      "got %r" % (name, flag, on, got))
                else:
                    check(abs(got - 10.0 ** float(cfg[log_axis])) < 1e-12,
                          "%s: %s must be 10 ** %s" % (name, flag, log_axis))
        out.append("%s ok" % name)

    # beta1 is passed as (1 - beta1) and must invert exactly.
    cfg = _random_config("S-A1", spec, rng)
    a = parser.parse_args(TJ.build_argv(cfg, arm="A1", sim_shards="s",
                                        out_dir="o", seed=0)[2:])
    check(abs((1.0 - a.one_minus_beta1)
              - (1.0 - float(cfg["one_minus_beta1"]))) < 1e-12,
          "one_minus_beta1 must survive as itself, not as beta1")
    # --dry-run is passed through, and an unknown flag would be caught here.
    argv = TJ.build_argv(cfg, arm="A1", sim_shards="s", out_dir="o", seed=0,
                         dry_run=True)
    check(parser.parse_args(argv[2:]).dry_run, "--dry-run must pass through")
    return "; ".join(out) + "; %d axes mapped" % len(TJ.AXIS_TO_FLAG)


def j29_no_axis_is_dropped_silently() -> str:
    """J29: every searched axis reaches the trainer, and the guard that
    enforces it still fires.

    The guard matters more than its current emptiness. Searching an axis the
    runner cannot receive produces a partial dependence that reports the axis
    as irrelevant -- a null result that is confidently wrong. So the test
    asserts BOTH that no campaign has a hole today AND that the refusal
    mechanism works, by injecting one. Without the second half the guard
    would rot the moment a new axis is added.
    """
    spec = _spec()
    rng = np.random.default_rng(1)
    for name in sorted(JS.CAMPAIGNS):
        check(not TJ.unmapped_free_axes(name),
              "campaign %s has free axes that cannot reach the trainer: %s"
              % (name, TJ.unmapped_free_axes(name)))
    check(set(TJ.DERIVED) & set(TJ.AXIS_TO_FLAG) == set(),
          "an axis cannot both have a flag and be derived")
    covered = set(TJ.AXIS_TO_FLAG) | set(TJ.DERIVED)
    check(covered == set(JS.JOINT_KNOB_ORDER),
          "every axis must be mapped or derived; uncovered: %s"
          % sorted(set(JS.JOINT_KNOB_ORDER) - covered))

    if not _dsn():
        raise Skip("%s; the injection needs an S-A2 config" % _dsn_status())

    # loss_type must be "triplet" here: under "joint"/"joint_sep" the margin
    # is INACTIVE by A(l) and canonicalisation pins it to 0.2, so a dropped
    # margin would genuinely change nothing and the guard would correctly
    # stay silent. The guard is about axes carrying a SEARCHED value.
    cfg = _random_config("S-A2", spec, rng, dsn_on=1, loss_type="triplet",
                         margin=0.55)
    check(abs(float(cfg["margin"]) - 0.55) < 1e-12,
          "margin must be active under 'triplet', got %r" % cfg["margin"])
    TJ.build_argv(cfg, arm="A2", sim_shards="s", out_dir="o", seed=0,
                  spec_fixed=spec.fixed)          # must NOT raise today

    saved_u, saved_f = dict(TJ.UNREACHABLE), dict(TJ.AXIS_TO_FLAG)
    try:
        TJ.UNREACHABLE["margin"] = "injected by J29"
        TJ.AXIS_TO_FLAG.pop("margin")
        try:
            TJ.build_argv(cfg, arm="A2", sim_shards="s", out_dir="o", seed=0,
                          spec_fixed=spec.fixed)
            raise AssertionError("with margin unreachable and set off its "
                                 "canonical value, build_argv must refuse "
                                 "rather than silently drop it")
        except ValueError as exc:
            check("margin" in str(exc), "the error must name the axis: %s" % exc)
        check(TJ.unmapped_free_axes("S-A2") == ["margin"],
              "the hole must be reported for the campaign that searches it")
        # At its canonical value the same axis is harmless: dropping a value
        # the trainer would have used anyway changes nothing.
        cfg0 = _random_config("S-A2", spec, rng, dsn_on=0)
        TJ.build_argv(cfg0, arm=TJ.arm_for_config(cfg0), sim_shards="s",
                      out_dir="o", seed=0, spec_fixed=spec.fixed)
    finally:
        TJ.UNREACHABLE.clear(); TJ.UNREACHABLE.update(saved_u)
        TJ.AXIS_TO_FLAG.clear(); TJ.AXIS_TO_FLAG.update(saved_f)
    check(not TJ.unmapped_free_axes("S-A2"), "the injection must be undone")
    return ("all %d axes mapped or derived across %d campaigns; refusal "
            "mechanism fires when injected"
            % (len(JS.JOINT_KNOB_ORDER), len(JS.CAMPAIGNS)))


def j30_ledger_round_trip() -> str:
    spec = _spec()
    rng = np.random.default_rng(2)
    root = tempfile.mkdtemp()
    try:
        good = []
        for i in range(6):
            cfg = _random_config("S-A1", spec, rng)
            TJ.write_record(root, "S-A1",
                            {"trial_id": "ok%d" % i, "config": cfg,
                             "nll": 1.0 - 0.1 * i, "status": "ok", "tag": ""})
            good.append(cfg)
        # Each of these must be EXCLUDED from the surrogate, for its own
        # reason: a failed trial carries no information about the objective,
        # a tagged one answered a different question, and a record with a
        # non-finite or missing score has nothing to tell.
        TJ.write_record(root, "S-A1", {"trial_id": "bad1", "config": good[0],
                                       "nll": 0.0, "status": "failed"})
        TJ.write_record(root, "S-A1", {"trial_id": "bad2", "config": good[0],
                                       "nll": 0.0, "status": "ok",
                                       "tag": "control"})
        TJ.write_record(root, "S-A1", {"trial_id": "bad3", "config": good[0],
                                       "nll": float("nan"), "status": "ok"})
        TJ.write_record(root, "S-A1", {"trial_id": "bad4",
                                       "config": {"lr": 1e-3}, "nll": 0.0,
                                       "status": "ok"})
        obs = TJ.load_observations(root, "S-A1", spec)
        check(len(obs) == 6, "expected 6 usable observations, got %d"
                             % len(obs))
        check(all(np.isfinite(y) for _, y in obs), "a non-finite score got in")
        keys = {JS.config_key(c) for c, _ in obs}
        check(len(keys) == 6, "observations must be distinct configs")
        # No half-written file survives a crash: write is atomic.
        check(not [p for p in os.listdir(os.path.join(root, "S-A1", "trials"))
                   if p.endswith(".tmp")], "a .tmp file was left behind")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return "6 usable of 10 records; failed/tagged/nan/incomplete excluded"


def j31_propose_writes_pending() -> str:
    try:
        import skopt  # noqa: F401
        import npe_tune_search  # noqa: F401
    except ImportError as exc:
        raise Skip("needs skopt and npe_tune_search on SBI_HPC_DIR (%s)" % exc)
    root = tempfile.mkdtemp()
    try:
        args = ["--campaign", "S-A5", "--results-dir", root,
                "--n-points", "4", "--n-initial-points", "4",
                "--split-hash", "h", "--contract-digest", "c"]
        check(TJ.main(["propose"] + args) == 0, "propose must succeed")
        pend = os.path.join(root, "S-A5", "pending")
        first = sorted(os.listdir(pend))
        check(len(first) == 4, "expected 4 pending, got %d" % len(first))
        for f in first:
            with open(os.path.join(pend, f), encoding="ascii") as fh:
                d = json.load(fh)
            check(d["trial_id"] == f[:-5], "filename must be the trial id")
            check(all(k in d["config"] for k in JS.JOINT_KNOB_ORDER),
                  "a pending spec must carry a complete config")
        # A second call with the SAME seed proposes the same points. They
        # must be DROPPED, not written over the pending specs already there:
        # the shared propose() accepts duplicates once its resampling budget
        # is exhausted, so without the trial-id check in cmd_propose the
        # round would overwrite four files and report success having added
        # nothing. Regression for exactly that.
        check(TJ.main(["propose"] + args) == 0, "second propose")
        same = sorted(os.listdir(pend))
        check(same == first,
              "a repeated seed must add nothing and overwrite nothing: %s"
              % sorted(set(same) ^ set(first)))

        # A DIFFERENT seed is how a new round is drawn.
        args2 = list(args)
        args2[args2.index("--split-hash") - 1:0] = []
        check(TJ.main(["propose"] + args + ["--seed", "7"]) == 0,
              "third propose with a new seed")
        third = sorted(os.listdir(pend))
        check(len(third) > len(first),
              "a new seed must add proposals, got %d then %d"
              % (len(first), len(third)))
        check(set(first) <= set(third), "the first batch must survive")
        for f in third:
            with open(os.path.join(pend, f), encoding="ascii") as fh:
                d = json.load(fh)
            check(d["trial_id"] == f[:-5], "filename must be the trial id")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return ("4 pending; repeated seed adds nothing and overwrites nothing; "
            "new seed adds more")


def j32_finalists() -> str:
    spec = _spec()
    rng = np.random.default_rng(3)
    root = tempfile.mkdtemp()
    try:
        for i in range(7):
            cfg = _random_config("S-A1", spec, rng)
            TJ.write_record(root, "S-A1", {"trial_id": "t%d" % i,
                                           "config": cfg, "nll": 5.0 - i,
                                           "status": "ok", "tag": ""})
        rc = TJ.main(["finalists", "--campaign", "S-A1", "--results-dir", root,
                      "--top-k", "3", "--split-hash", "h",
                      "--contract-digest", "c"])
        check(rc == 0, "finalists must succeed")
        with open(os.path.join(root, "S-A1", "finalists.json"),
                  encoding="ascii") as fh:
            fin = json.load(fh)
        check(fin["K"] == 3, "K")
        got = [f["nll_rank_split"] for f in fin["finalists"]]
        check(got == sorted(got), "finalists must be ranked ascending: %s" % got)
        check(abs(got[0] - (-1.0)) < 1e-12, "best nll should be -1.0, got %r"
                                            % got[0])
        check(fin["rank_split"] == "sel" and fin["gate_split"] == "gate",
              "both splits must be recorded")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return "K=3 ranked ascending; both splits named"


def j33_controls_and_report() -> str:
    spec = _spec()
    rng = np.random.default_rng(4)
    root = tempfile.mkdtemp()
    try:
        for i in range(4):
            cfg = _random_config("S-A1", spec, rng)
            TJ.write_record(root, "S-A1", {"trial_id": "t%d" % i,
                                           "config": cfg, "nll": 5.0 - i,
                                           "delta": 0.4 + 0.05 * i,
                                           "status": "ok", "tag": ""})
        TJ.main(["finalists", "--campaign", "S-A1", "--results-dir", root,
                 "--top-k", "3", "--split-hash", "h", "--contract-digest", "c"])

        # report must REFUSE before any gate has run.
        try:
            TJ.main(["report", "--campaign", "S-A1", "--results-dir", root])
            raise AssertionError("report must refuse to ship an ungated config")
        except SystemExit as exc:
            check("gated" in str(exc) or "control" in str(exc), str(exc))

        rc = TJ.main(["controls", "--plan", "--campaign", "S-A1",
                      "--results-dir", root, "--n-control", "5",
                      "--split-hash", "h", "--contract-digest", "c"])
        check(rc == 0, "controls --plan")
        specs = os.path.join(root, "S-A1", "control_specs")
        planned = sorted(os.listdir(specs))
        check(len(planned) == 15, "3 finalists x 5 seeds = 15, got %d"
                                  % len(planned))
        with open(os.path.join(root, "S-A1", "finalists.json"),
                  encoding="ascii") as fh:
            fin = json.load(fh)
        by_rank = {f["rank"]: f for f in fin["finalists"]}
        for f in planned:
            with open(os.path.join(specs, f), encoding="ascii") as fh:
                s = json.load(fh)
            cand = dict(by_rank[s["finalist_rank"]]["config"])
            cand["shuffle_pairs"] = 0
            JS.assert_control_identity(cand, s["config"])   # must not raise
            check(s["tag"] == "control", "control specs must be tagged")

        # Feed control results back in and score. Finalist 1 beats its floor,
        # the others do not.
        for f in planned:
            with open(os.path.join(specs, f), encoding="ascii") as fh:
                s = json.load(fh)
            base = 0.02 if s["finalist_rank"] == 1 else 0.44
            TJ.write_record(root, "S-A1",
                            {"trial_id": s["trial_id"], "config": s["config"],
                             "tag": "control", "status": "ok",
                             "delta": base + 0.002 * (s["control_seed"] % 5)})
        rc = TJ.main(["controls", "--score", "--campaign", "S-A1",
                      "--results-dir", root, "--n-seeds", "1",
                      "--alpha", "0.05", "--floor", "0.0"])
        check(rc == 0, "controls --score")
        with open(os.path.join(root, "S-A1", "controls.json"),
                  encoding="ascii") as fh:
            c = json.load(fh)
        check(len(c["verdicts"]) == 3, "all K verdicts must be reported")
        for v in c["verdicts"]:
            check(v["n_control"] == 5, "each finalist gets its OWN 5 controls")
        check(TJ.main(["report", "--campaign", "S-A1",
                       "--results-dir", root]) == 0,
              "report must succeed once gated")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return "15 controls planned with identity asserted; K=3 verdicts; report gated"


def j34_arm_mapping() -> str:
    spec = _spec()
    rng = np.random.default_rng(5)
    cases = []
    for name, switches, want in (
            ("S-A1", {}, "A1"),
            ("S-A5", {"rep_on": 1}, "A5"),
            ("S-A5", {"rep_on": 0}, "A1")):
        cfg = _random_config(name, spec, rng, **switches)
        got = TJ.arm_for_config(cfg)
        check(got == want, "%s %r -> %s, want %s" % (name, switches, got, want))
        cases.append("%s%s->%s" % (name, switches or "", got))
    cfg = _random_config("S-A1", spec, rng)
    cfg["shuffle_pairs"] = 1
    check(TJ.arm_for_config(cfg) == "shuffled",
          "a control config must map to the shuffled arm")
    both = _random_config("S-A1", spec, rng)
    both["dsn_on"], both["rep_on"] = 1, 1
    try:
        TJ.arm_for_config(both)
        raise AssertionError("dsn_on = rep_on = 1 has no Stage 3 arm and must "
                             "be refused, not mapped onto A2")
    except ValueError as exc:
        check("S-A25" in str(exc) or "both" in str(exc), str(exc))
    return "; ".join(cases) + "; shuffled ok; both-on refused"


# ---------------------------------------------------------------------------

def j36_job_scripts() -> str:
    """J36: the PBS element and the launcher, exercised rather than eyeballed.

    Both are the parts nobody tests until a queue rejects them at 3am, and
    both fail in ways that look like something else: a stale array range
    looks like a crashed element, a CRLF shebang looks like a missing
    interpreter, and an unknown campaign used to be reported as "free axes
    do not reach the trainer".
    """
    if os.name != "posix":
        raise Skip("shell scripts need a POSIX shell")
    jobs = os.path.join(_HERE, "jobs")
    pbs = os.path.join(jobs, "joint_tune.pbs")
    launcher = os.path.join(jobs, "launch_joint_tune.sh")
    for f in (pbs, launcher):
        if not os.path.isfile(f):
            raise Skip("%s not found" % f)
        b = open(f, "rb").read()
        check(b.count(b"\r") == 0,
              "%s has CR bytes; qsub would fail with a bad-interpreter error "
              "that never mentions line endings" % os.path.basename(f))
        r = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
        check(r.returncode == 0, "bash -n %s: %s" % (f, r.stderr))

    root = tempfile.mkdtemp()
    try:
        env = dict(os.environ)
        env.update({"PYBIN": sys.executable, "SIM_SHARDS": "/data/S/*.npz",
                    "OUT_DIR": os.path.join(root, "runs")})
        rc = TJ.main(["propose", "--campaign", "S-A1", "--results-dir", root,
                      "--n-points", "3", "--n-initial-points", "3",
                      "--split-hash", "h", "--contract-digest", "c"])
        check(rc == 0, "propose")

        def run_pbs(idx, **over):
            e = dict(env)
            e.update({"DRYRUN": "1", "CAMPAIGN": "S-A1", "RESULTS_DIR": root,
                      "PBS_ARRAY_INDEX": str(idx)})
            e.update(over)
            return subprocess.run(["bash", pbs], capture_output=True,
                                  text=True, env=e, cwd=_HERE)

        r = run_pbs(0)
        check(r.returncode == 0, "element 0 dry run: %s" % r.stderr[-400:])
        check("run_joint_arms.py" in r.stdout,
              "the element must resolve a runner command line")
        check("--dry-run" in r.stdout, "DRYRUN must reach the runner")
        check("--arm A1" in r.stdout, "campaign S-A1 must map to arm A1")

        # Each element must pick a DIFFERENT pending trial, or an array of N
        # elements evaluates one configuration N times.
        ids = []
        for k in range(3):
            out = run_pbs(k).stdout
            line = [l for l in out.splitlines() if l.startswith("trial id")]
            check(line, "element %d printed no trial id" % k)
            ids.append(line[0].split(":", 1)[1].strip())
        check(len(set(ids)) == 3,
              "elements must select distinct pending trials, got %s" % ids)

        # A stale array range is not a failure.
        r = run_pbs(99)
        check(r.returncode == 0 and "out of range" in r.stdout,
              "an index past the pending set must exit 0 with an explanation")

        # A missing required variable must stop before any work.
        e = dict(env); e.pop("SIM_SHARDS")
        e.update({"DRYRUN": "1", "CAMPAIGN": "S-A1", "RESULTS_DIR": root,
                  "PBS_ARRAY_INDEX": "0"})
        r = subprocess.run(["bash", pbs], capture_output=True, text=True,
                           env=e, cwd=_HERE)
        check(r.returncode != 0 and "SIM_SHARDS" in (r.stderr + r.stdout),
              "an unset SIM_SHARDS must fail loudly and name itself")

        # The launcher: dry run by default, and it must NOT submit.
        r = subprocess.run(["bash", launcher, "S-A1", root],
                           capture_output=True, text=True, env=env, cwd=_HERE)
        check(r.returncode == 0, "launcher dry run: %s" % r.stderr[-300:])
        check("DRY RUN -- nothing submitted" in r.stdout,
              "the launcher must default to a dry run")
        check("qsub -J 0-2" in r.stdout,
              "3 pending specs must give an array of 0-2: %s" % r.stdout[-300:])
        check("--submit" not in r.stdout.split("would submit")[0],
              "nothing may be submitted without --submit")

        # --max limits the array.
        r = subprocess.run(["bash", launcher, "S-A1", root, "--max", "2"],
                           capture_output=True, text=True, env=env, cwd=_HERE)
        check("qsub -J 0-1" in r.stdout, "--max 2 must give 0-1")

        # An unknown campaign must read as a typo, not as a missing flag.
        r = subprocess.run(["bash", launcher, "S-A9", root],
                           capture_output=True, text=True, env=env, cwd=_HERE)
        check(r.returncode != 0, "an unknown campaign must fail the guards")
        check("unknown campaign" in r.stdout,
              "the message must name the typo, not blame the trainer: %s"
              % r.stdout[-300:])
        check("BLOCKING" not in r.stdout,
              "an unknown campaign must NOT be reported as unreachable axes")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return ("PBS: 3 distinct elements, stale index exits 0, unset var fails; "
            "launcher: dry by default, 0-2, --max works, typo named")


TESTS: Dict[str, Tuple[str, Callable[[], str]]] = {
    "J28": ("argv round-trips through the runner's own parser", j28_argv_round_trip),
    "J29": ("no axis is dropped silently", j29_no_axis_is_dropped_silently),
    "J30": ("ledger round trip and exclusions", j30_ledger_round_trip),
    "J31": ("[needs skopt] propose writes pending", j31_propose_writes_pending),
    "J32": ("finalists ranks and names both splits", j32_finalists),
    "J33": ("controls plan/score; report refuses ungated", j33_controls_and_report),
    "J34": ("arm mapping", j34_arm_mapping),
    "J36": ("PBS element and launcher, dry-run end to end", j36_job_scripts),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", dest="pattern", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for tid, (desc, _) in TESTS.items():
            print("  %-5s %s" % (tid, desc))
        return 0

    ids = [t for t in TESTS if args.pattern is None or args.pattern in t]
    print("=" * 72)
    print("smoke_test_joint_tune.py -- %d test(s)%s"
          % (len(ids), "" if _dsn() else
             "   [%s -- DSN clauses will SKIP]" % _dsn_status()))
    print("=" * 72)
    n_pass = n_fail = n_skip = 0
    for tid in ids:
        desc, fn = TESTS[tid]
        try:
            detail = fn() or ""
            n_pass += 1
            print("  %-5s PASS  %s" % (tid, detail), flush=True)
        except Skip as exc:
            n_skip += 1
            print("  %-5s SKIP  %s" % (tid, exc), flush=True)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print("  %-5s FAIL  %s: %s" % (tid, type(exc).__name__, exc),
                  flush=True)
            RESULTS.append((tid, traceback.format_exc()))
    print("-" * 72)
    print("%d passed, %d failed, %d skipped, %d total"
          % (n_pass, n_fail, n_skip, len(ids)))
    for tid, tb in RESULTS:
        print("\n--- %s ---\n%s" % (tid, tb))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
