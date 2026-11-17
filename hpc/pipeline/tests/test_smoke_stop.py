#!/usr/bin/env python3
"""Smoke test for block 30 of specs/SPEC.md (D-148 (2), D-141, D-150): a
search stopped by hand -- `pipeline.py stop-search` (stages/search_stop.py)
and the search stage's poll while a request stands (stages/search.py).

Checks (specs/SPEC.md block 30, "Test oracles" T30a-T30e)
    test_known_answer   T30a, the command, by hand (no scheduler): each
                        refusal -- --search given on a configuration with
                        searches null, missing or unknown on a list one; a
                        search with no completed trial (the ledger read as
                        SearchStage reads it: failed, killed, unreadable and
                        set-aside records are not ok, a record without a
                        status is); an ended search; --cancel without a
                        request, or with --kill-running -- each with nothing
                        written (every file's bytes and mtime); the request's
                        file and content (keys, schema, ISO 8601 UTC time,
                        user, host, ASCII, sorted keys, no temporary file
                        left); the last line; the replacement (the line
                        names the previous time and kill_running); --cancel;
                        status_line (with `; the search ended (user_stop)`
                        only once SEARCH_COMPLETE.json says so); the CLI
                        through pipeline.py (cmd_stop_search); a request
                        file that cannot be read (a stop without kill)
    test_reference      T30b and T30d end to end (the fake PBS, the real
                        tuner, the stand-in runner below; the throttle at
                        2): trial A ok, B and C running, D waiting when
                        `stop-search` (no kill) is run; then B finishes ok
                        and C fails of a node fault (SIGBUS): nothing new
                        is submitted (three trial jobs in all, none after
                        the request), C is not resubmitted (its record set
                        aside, the trial cancelled), D is cancelled, never
                        submitted; SEARCH_COMPLETE.json says user_stop with
                        the request's content, its finalist the best ok
                        trial; one `search: stop requested` line; the best
                        fit starts on it; PIPELINE DONE
    test_contract       T30c end to end: the same with --kill-running and B,
                        C held: their jobs deleted (the fake qdel records
                        it), their records killed with the reason `stopped
                        by the user (stop-search --kill-running)`, none
                        resubmitted though resubmit_killed is true; D
                        cancelled; the best fit; PIPELINE DONE
    test_invariants     T30e, at the level of two search views of one list
                        configuration (a copy of linddup.json; the
                        sequencer's multi-search graph is block 29's): a
                        request for A2 stops A2 only -- A2's poll submits
                        nothing, cancels its waiting trial and finishes
                        (user_stop) once its running trial ends, while A1's
                        poll submits its waiting trial and goes on, with no
                        stop line, no deletion and no artefact; the status
                        lines; the request is never removed by the stage
    test_edge_cases     the stage's rules by hand (stdlib, a recorder and a
                        tuner in this process): under a request no
                        resubmission of any kind -- a walltime kill, a hold
                        on an avoided node (exit 94), an environment fault
                        (exit 96) are recorded killed with the reason
                        `ended without a record after a stop request`, a
                        node fault is healed and then cancelled, never
                        submitted; --kill-running then --cancel: the deleted
                        trial stays killed and the search goes on; a request
                        at the gate's end proposes no round 0 and the finish
                        fails with no completed trial; the start under a
                        request proposes no round 0, the withdrawal does; a
                        user_stop finish that did not complete is re-opened
                        on a fresh state (the request still there: it
                        finishes again; withdrawn: it goes on); a request
                        file that cannot be read stops without killing
    test_convergence    not a discretisation: skipped
    test_determinism    no randomness in the pipeline's part: skipped

The stand-in runner (written into a copy of hpc/ as `joint/stage3/
run_joint_arms.py`, as tests/test_smoke_block7b.py installs its fault
wrapper): as a module, the real runner (the split tool imports it); as a
script, the action of the trial in the order of first calls, from
`stop_plan.json` beside it -- `ok` the real runner, `hold_ok` and
`hold_sigbus` wait for the file `release` beside it and then run the real
runner, or kill themselves with SIGBUS (a node fault, D-096); probes and
members always the real runner.

Run
    python3 hpc/pipeline/tests/test_smoke_stop.py [check ...]
        A few minutes at one thread (two reduced searches with their best
        fits); names of checks run only those. test_known_answer,
        test_invariants and test_edge_cases need the standard library only;
        the end-to-end checks need the training environment (torch, sbi,
        zuko, scikit-optimize). PIPELINE_SKIP_CONDA=1 is set by the
        helpers. Nothing touches davinci.

Pure ASCII, LF only.
"""

import contextlib
import getpass
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C                      # noqa: E402
import pbs as PBS                       # noqa: E402
import stages.common as CM              # noqa: E402
import stages.search as SS              # noqa: E402
import stages.search_stop as SST        # noqa: E402
from stages import runs as RU           # noqa: E402

PYTHON = sys.executable
GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
LINDDUP = os.path.join(_PIPE, "projects", "linddup", "linddup.json")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
TAIL = "the sequencer acts at its next tick (every driver.poll_seconds)"


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_stop_")


def _write(path, obj):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    if isinstance(obj, bytes):
        with open(path, "wb") as fh:
            fh.write(obj)
        return path
    with open(path, "w") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, indent=1, sort_keys=True)
    return path


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _tree(root):
    """{relative path: (sha256, mtime_ns)} of every file under root."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = (_sha(p), os.stat(p).st_mtime_ns)
    return out


def _cfg_copy(src, root, name):
    """A committed project file with its work, runs, tune and banks roots
    under `root`, loaded as the sequencer loads it."""
    with open(src, "rb") as fh:
        d = C._strip_docs(json.loads(fh.read().decode("ascii")))
    for k in ("work_root", "runs_root", "tune_root", "banks_root"):
        d["paths"][k] = os.path.join(root, k)
    return C.load_config(_write(os.path.join(root, name), d), synthetic_ok=True)


def _cli(cfg, search=None, kill=False, cancel=False):
    lines = []
    rc = SST.cli(cfg, search, kill, cancel, out=lines.append)
    return rc, lines


def _ledger(view):
    return SS.SearchStage()._dirs(view)["ledger"]


def _ok_record(view, tid, nll):
    return _write(os.path.join(_ledger(view), tid + ".json"),
                  {"trial_id": tid, "campaign": view.data["search"]["campaign"], "status": "ok",
                   "nll": float(nll), "config": {"lr": 0.001, "batch_size_npe": 16},
                   "split_hash": "a" * 64})


@contextlib.contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield value
    finally:
        setattr(obj, name, old)


class _Rec(object):
    """The sequencer's recorder as the stage sees it (pipeline.Submitter),
    for the checks that drive SearchStage in this process: the scheduler's
    answers from `states` (job id -> 'R', or ('F', exit status)); every
    submission (into the stage record's jobs, as the Submitter does) and
    every deletion recorded; `fits` always."""

    def __init__(self, st):
        self.st = st
        self.states = {}
        self.lines = []
        self.submitted = []
        self.deleted = []

    def log(self, text):
        self.lines.append(str(text))

    def fits(self, n):
        return True

    def status(self, job_id):
        v = self.states.get(job_id, "R")
        if isinstance(v, tuple):
            return PBS.JobStatus(job_id, v[0], exit_status=v[1], walltime_s=1.0)
        return PBS.JobStatus(job_id, v)

    def submit(self, argv, role, log_path=None, cwd=None, env=None, ncpus=None, after=None):
        jid = "%d.rec" % (900 + len(self.submitted))
        self.submitted.append((role, jid))
        self.st["jobs"].append({"id": jid, "role": role, "cmd": list(argv), "log": log_path,
                                "finished": False, "exit_status": None, "walltime_s": None,
                                "submitted_utc": CM.utc_now(), "ncpus": 1})
        self.states[jid] = "R"
        return jid

    def delete(self, job_id):
        self.deleted.append(job_id)
        self.states[job_id] = ("F", 271)
        return True

    def joined(self):
        return "\n".join(self.lines)


class _Tuner(object):
    """common.run_tuner for the checks that drive the stage in this process:
    `finalists` ranks the ok records of the ledger on their nll (the tuner's
    rule, on hand-made records) and writes finalists.json; `propose` writes
    the pending specs of `propose_ids`; any other call fails the check."""

    def __init__(self, propose_ids=()):
        self.calls = []
        self.propose_ids = list(propose_ids)

    def __call__(self, cfg, args, log_path=None, timeout=None):
        self.calls.append(args[0])
        dirs = SS.SearchStage()._dirs(cfg)
        if args[0] == "finalists":
            oks = []
            for p in sorted(glob.glob(os.path.join(dirs["ledger"], "*.json"))):
                rec = json.load(open(p))
                if str(rec.get("status", "ok")) == "ok":
                    oks.append(rec)
            if not oks:
                raise CM.ToolError("no completed trials to rank")
            best = min(oks, key=lambda r: r["nll"])
            _write(dirs["finalists"], {"K": 1, "finalists": [
                {"rank": 1, "trial_id": best["trial_id"], "nll_rank_split": best["nll"],
                 "config": best.get("config")}]})
            return "[finalists] K=1 (the check's stand-in)"
        if args[0] == "propose":
            for tid in self.propose_ids:
                _write(os.path.join(dirs["pending"], tid + ".json"),
                       {"trial_id": tid, "campaign": cfg.data["search"]["campaign"],
                        "config": {"lr": 0.002, "batch_size_npe": 32}})
            return "[propose] %d (the check's stand-in)" % len(self.propose_ids)
        raise AssertionError("the stage called the tuner's %r" % (args[0],))


def _identity(view):
    ident = dict(SS.config_identity(view))
    ident.update({"split_hash": "a" * 64, "contract_digest": "b" * 64, "n_train": 100, "n_sel": 10,
                  "n_report": 5, "n_rows": 115, "n_shards": 1, "W": 128, "latent_kind": None,
                  "d_theta": int(view.data["search"]["shape_anchors"]["d_theta"])})
    return ident


def _hand_search(view, spec):
    """A search of `view` in its phase rounds, by hand: round 0 holds the
    trials of `spec`, [(trial id, kind, nll)], kind `ok` (an ok ledger
    record), `running` (its job J_<id> running; a pending spec) or `waiting`
    (never submitted; a pending spec); the identity is the configuration's
    with a hand-made bank part. Returns (stage, st, rec)."""
    stage = SS.SearchStage()
    dirs = stage._dirs(view)
    camp = view.data["search"]["campaign"]
    sub = {"identity": _identity(view), "phase": "rounds", "round": 0,
           "rounds": [{"round": 0, "trial_ids": [t for t, _, _ in spec], "proposed_utc": CM.utc_now(),
                       "n_requested": len(spec), "orphan_pending_before": 0}],
           "trials": {}, "gate": {"verdict": {"ok": True}}, "clock_start_utc": CM.utc_now(),
           "ended_by": None}
    st = {"status": "running", "jobs": [], "sub": sub, "resubmits": 0}
    rec = _Rec(st)
    for tid, kind, nll in spec:
        tr = {"round": 0, "jobs": [], "resubmits": 0, "status": kind}
        if kind == "ok":
            _ok_record(view, tid, nll)
            tr["nll"] = float(nll)
        else:
            _write(os.path.join(dirs["pending"], tid + ".json"),
                   {"trial_id": tid, "campaign": camp, "config": {"lr": 0.001, "batch_size_npe": 16}})
        if kind == "running":
            jid = "J_" + tid
            tr["jobs"] = [jid]
            st["jobs"].append({"id": jid, "role": "trial " + tid[:12], "cmd": [], "finished": False,
                               "log": os.path.join(dirs["jobs"], "%s.0.log" % tid), "exit_status": None,
                               "walltime_s": None, "submitted_utc": CM.utc_now(), "ncpus": 1})
            rec.states[jid] = "R"
        sub["trials"][tid] = tr
    return stage, st, rec


def _statuses(st):
    return dict((t, tr["status"]) for t, tr in st["sub"]["trials"].items())


def _lines_with(rec, needle):
    return [ln for ln in rec.lines if needle in ln]


# ------------------------------------------------- the stand-in runner ---

_STOP_RUNNER = r'''#!/usr/bin/env python3
"""The stand-in for `hpc/joint/stage3/run_joint_arms.py` of test_smoke_stop.py
(block 30; not pipeline code), written into a temporary copy of hpc/. As a
module it is the real runner (the split tool imports grouped_split). As a
script it acts by `stop_plan.json` beside it, {"trials": [a0, a1, ...]}: the
i-th trial run folder seen (the order of first calls) takes a_i -- ok (the
real runner), hold_ok / hold_sigbus (wait for the file `release` beside this
one, then the real runner, or SIGBUS: a node fault, D-096); a trial beyond
the plan, a probe (--probe-steps) and a member: the real runner.

Pure ASCII, LF only; stdlib only.
"""

import importlib.util
import json
import os
import runpy
import signal
import sys
import time

REAL_RUNNER = "@@REAL_RUNNER@@"

if __name__ != "__main__":
    _spec = importlib.util.spec_from_file_location("_real_run_joint_arms", REAL_RUNNER)
    _real = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_real)
    globals().update(dict((k, v) for k, v in _real.__dict__.items() if not k.startswith("__")))


def _here():
    return os.path.dirname(os.path.abspath(__file__))


def _run_real():
    sys.argv = [REAL_RUNNER] + sys.argv[1:]
    try:
        runpy.run_path(REAL_RUNNER, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    return 0


def _order_of(name):
    """The run folder's place in the order of first calls (two runners may
    start at once: a lock file)."""
    st = os.path.join(_here(), "stop_state")
    if not os.path.isdir(st):
        try:
            os.makedirs(st)
        except OSError:
            pass
    lock = os.path.join(st, "lock")
    end = time.time() + 60
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except OSError:
            if time.time() > end:
                break
            time.sleep(0.05)
    try:
        p = os.path.join(st, "order.json")
        order = json.load(open(p)) if os.path.isfile(p) else []
        if name not in order:
            order.append(name)
            with open(p + ".tmp", "w") as fh:
                json.dump(order, fh)
            os.replace(p + ".tmp", p)
        return order.index(name)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def main():
    argv = sys.argv[1:]
    if "--probe-steps" in argv:
        return _run_real()
    out_dir = argv[argv.index("--out-dir") + 1] if "--out-dir" in argv else ""
    name = os.path.basename(os.path.normpath(out_dir))
    if name.startswith("member_seed"):
        act = "ok"
    else:
        plan = json.load(open(os.path.join(_here(), "stop_plan.json")))["trials"]
        i = _order_of(name)
        act = plan[i] if i < len(plan) else "ok"
    print("stop_runner: %s -> %s" % (name, act), flush=True)
    if act.startswith("hold_"):
        rel = os.path.join(_here(), "release")
        end = time.time() + 900
        while not os.path.exists(rel):
            if time.time() > end:
                sys.stderr.write("stop_runner: %s was never released\n" % name)
                return 98
            time.sleep(0.2)
        act = act[len("hold_"):]
        print("stop_runner: %s released -> %s" % (name, act), flush=True)
    if act == "ok":
        return _run_real()
    if act == "sigbus":
        os.kill(os.getpid(), signal.SIGBUS)
        time.sleep(30)
        return 99
    sys.stderr.write("stop_runner: unknown action %r\n" % (act,))
    return 97


if __name__ == "__main__":
    sys.exit(main())
'''


# ------------------------------------------------ the end-to-end runs ---

_E2E = {}


def _wait_until(pred, timeout, proc=None):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        if proc is not None and proc.poll() is not None:
            return pred()
        time.sleep(0.3)
    return False


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _spool_records(spool):
    out = []
    for p in glob.glob(os.path.join(spool, "*.json")):
        if os.path.basename(p)[:-5].isdigit():
            with open(p) as fh:
                out.append(json.load(fh))
    return sorted(out, key=lambda r: r["num"])


def _e2e(kind):
    """The end-to-end run `nokill` (T30b, T30d) or `kill` (T30c), once per
    process; its facts kept for the checks."""
    hit = _E2E.get(kind)
    if hit is not None:
        if hit.get("skip"):
            raise unittest.SkipTest(hit["skip"])
        if hit.get("error"):
            raise AssertionError("the end-to-end run %s failed: %s" % (kind, hit["error"]))
        return hit
    R = {}
    _E2E[kind] = R
    try:
        _e2e_run(kind == "kill", R)
    except unittest.SkipTest as exc:
        R["skip"] = str(exc)
        raise
    except Exception:
        R["error"] = traceback.format_exc()
        raise
    return R


def _e2e_run(kill, R):
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("the training environment is needed (%s)" % exc)
    import test_smoke_block7b as B7B
    root = _tmp()
    R["root"] = root
    plan = ["ok", "hold_ok", "hold_ok"] if kill else ["ok", "hold_ok", "hold_sigbus"]
    path, c = B7B._fault_project(root, {"trials": []}, search={
        "n_initial_points": 4, "gp_rounds": 1, "trials_per_round": 2, "max_concurrent_trials": 2,
        "resubmit_killed": True, "resubmit_node_faults": True})
    stage3 = os.path.join(c["paths"]["sbi_dir"], "hpc", "joint", "stage3")
    _write(os.path.join(stage3, "run_joint_arms.py"),
           _STOP_RUNNER.replace("@@REAL_RUNNER@@", os.path.join(_REPO, "hpc", "joint", "stage3",
                                                                "run_joint_arms.py")))
    _write(os.path.join(stage3, "stop_plan.json"), {"trials": plan})
    camp = B7B._seed_probes(c)
    spool = os.path.join(root, "spool")
    env = B7B._env(spool)
    plog = os.path.join(c["paths"]["work_root"], "pipeline.log")
    order_path = os.path.join(stage3, "stop_state", "order.json")
    R.update(path=path, c=c, camp=camp, spool=spool, env=env, stage3=stage3, plan=plan)

    def order():
        try:
            return json.load(open(order_path))
        except (OSError, ValueError):
            return []

    proc, out_path = B7B._background(root, path, [], env)
    try:
        # A ended ok, B and C submitted and holding in their runners, D waiting
        ok = _wait_until(lambda: _read(plog).count(": submitted trial ") >= 3 and len(order()) >= 3,
                         1500, proc)
        assert ok, ("three trial jobs never ran", _read(plog)[-4000:], _read(out_path)[-3000:])
        R["order_at_request"] = order()
        R["trial_jobs_at_request"] = sum(1 for r in _spool_records(spool) if r["name"].startswith("t_"))
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "stop-search", "--config", path]
                           + (["--kill-running"] if kill else []), env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        R["cmd"] = (r.returncode, r.stdout)
        req_path = os.path.join(camp, "STOP_REQUEST.json")
        R["req_bytes"] = open(req_path, "rb").read() if os.path.isfile(req_path) else None
        R["req_mtime"] = os.stat(req_path).st_mtime if os.path.isfile(req_path) else None
        assert r.returncode == 0, r.stdout
        assert _wait_until(lambda: "search: stop requested" in _read(plog), 300, proc), _read(plog)[-3000:]
        if not kill:
            _write(os.path.join(stage3, "release"), "go\n")
        proc.wait(timeout=1500)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        # no job of the run is left holding
        for rec in _spool_records(spool):
            if rec["state"] != "F":
                subprocess.run(["qdel", rec["id"]], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    R["rc"] = proc.returncode
    R["out"] = _read(out_path)
    R["log"] = _read(plog)
    R["spool"] = _spool_records(spool)
    R["order"] = order()
    R["state"] = json.load(open(os.path.join(c["paths"]["work_root"], "state.json")))
    R["records"] = dict((os.path.basename(p)[:-5], json.load(open(p)))
                        for p in glob.glob(os.path.join(camp, "trials", "*.json")))
    scp = os.path.join(camp, "SEARCH_COMPLETE.json")
    R["sc"] = json.load(open(scp)) if os.path.isfile(scp) else None
    rfp = os.path.join(camp, "pipeline_rounds.json")
    R["rounds_file"] = json.load(open(rfp)) if os.path.isfile(rfp) else None
    bfp = os.path.join(c["paths"]["runs_root"], "synth_best", "BEST_FIT_COMPLETE.json")
    R["bf"] = json.load(open(bfp)) if os.path.isfile(bfp) else None
    R["req_end"] = open(req_path, "rb").read() if os.path.isfile(req_path) else None
    R["status_line"] = SST.status_line(C.load_config(path))


def _jobs_of(R, tid):
    name = CM.job_name("t_" + tid[:12])
    return [r for r in R["spool"] if r["name"] == name]


def _common_end(R, kill):
    """What both end-to-end runs share: the command's line, the one stop
    line, nothing submitted after the request, the waiting trial cancelled
    and never submitted, SEARCH_COMPLETE.json, the best fit, PIPELINE DONE.
    Returns (A, B, C, D, sub)."""
    rc, out = R["cmd"]
    assert rc == 0 and out.strip().splitlines()[-1] == (
        "STOP REQUESTED: search S-A1 (S-A1); kill running: %s; %s" % ("yes" if kill else "no", TAIL)), out
    assert R["rc"] == 0 and "PIPELINE DONE: every stage done or skipped" in R["out"], R["out"][-5000:]
    stops = [ln for ln in R["log"].splitlines() if "search: stop requested " in ln]
    assert len(stops) == 1, stops
    req = json.loads(R["req_bytes"].decode("ascii"))
    assert stops[0].endswith("search: stop requested %s (kill running: %s): no new trial; 2 running"
                             % (req["requested_utc"], "yes" if kill else "no")), stops
    assert req["kill_running"] is bool(kill)
    A, B, C = R["order"][:3]
    assert R["order"] == R["order_at_request"], (R["order"], R["order_at_request"])
    sub = R["state"]["stages"]["search"]["sub"]
    ids = sub["rounds"][0]["trial_ids"]
    assert len(sub["rounds"]) == 1 and len(ids) == 4 and set(R["order"]) < set(ids), sub["rounds"]
    D = [t for t in ids if t not in (A, B, C)][0]
    # nothing new submitted: the three trial jobs of the request's time, none after it
    tjobs = [r for r in R["spool"] if r["name"].startswith("t_")]
    assert len(tjobs) == 3 == R["trial_jobs_at_request"], [(r["name"], r["submitted"]) for r in tjobs]
    assert all(r["submitted"] < R["req_mtime"] for r in tjobs)
    assert not _jobs_of(R, D) and not sub["trials"][D]["jobs"] and sub["trials"][D]["status"] == "cancelled"
    assert os.path.isfile(os.path.join(R["camp"], "pending", D + ".json")), "the cancelled trial's spec kept"
    assert D not in R["records"]
    # SEARCH_COMPLETE.json: user_stop, the request's content, the best ok trial
    sc = R["sc"]
    assert sc and sc["ended_by"] == "user_stop", sc
    assert sc["stop_request"] == req, (sc["stop_request"], req)
    oks = sorted((r for r in R["records"].values() if r["status"] == "ok"), key=lambda r: r["nll"])
    assert sc["finalist"]["trial_id"] == oks[0]["trial_id"], (sc["finalist"], [r["trial_id"] for r in oks])
    assert R["rounds_file"]["ended_by"] == "user_stop"
    assert "search: SEARCH COMPLETE (user_stop)" in R["log"]
    # the best fit started after the search, on its finalist, and completed
    assert R["log"].index("search: done:") < R["log"].index("best_fit: running"), R["log"][-3000:]
    assert R["bf"] and R["bf"]["finalist"]["trial_id"] == sc["finalist"]["trial_id"], R["bf"]
    assert len([r for r in R["spool"] if r["name"].startswith("bf_")]) == 2
    # the request stays, byte for byte (the stage never removes it), and the status line says so
    assert R["req_end"] == R["req_bytes"]
    assert R["status_line"] == "stop requested %s (kill running: %s); the search ended (user_stop)" % (
        req["requested_utc"], "yes" if kill else "no"), R["status_line"]
    return A, B, C, D, sub


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    """T30a: the command, by hand."""
    root = _tmp()
    try:
        cfg = _cfg_copy(GIULIA, root, "rec.json")         # searches null: the single search
        camp = cfg.data["search"]["campaign"]
        cdir = os.path.join(cfg.data["paths"]["tune_root"], camp)
        req_path = os.path.join(cdir, "STOP_REQUEST.json")
        scp = os.path.join(cdir, "SEARCH_COMPLETE.json")
        led = _ledger(cfg)
        assert SST.request_path(cfg) == req_path
        assert dict(RU.search_views(cfg)) == {None: cfg}
        label = "search %s (%s)" % (camp, camp)

        def refused(c, args, needles):
            before = _tree(root)
            rc, lines = _cli(c, *args)
            assert rc == 2 and len(lines) == 1 and lines[0].startswith("REFUSED: "), (args, rc, lines)
            for n in needles:
                assert n in lines[0], (n, lines)
            assert _tree(root) == before, ("a refusal wrote or touched a file", args)
            return lines[0]

        shown = []
        # (a) --search on a configuration with searches null
        shown.append(refused(cfg, ("S-A1",), ["--search S-A1", "single search", "searches null"]))
        # (b) no completed trial: no ledger at all; then only records the stage does not read
        # as ok (failed, killed, empty, NUL bytes) and ok records set aside out of the ledger
        shown.append(refused(cfg, (), [label, "no completed trial yet",
                                       "nothing to fit yet; stop the whole run with `stop` instead"]))
        assert not os.path.exists(cdir), "a refusal created the campaign folder"
        _write(os.path.join(led, "t_failed.json"), {"trial_id": "t_failed", "status": "failed"})
        _write(os.path.join(led, "t_killed.json"), {"trial_id": "t_killed", "status": "killed"})
        _write(os.path.join(led, "t_empty.json"), b"")
        _write(os.path.join(led, "t_nul.json"), b"\0" * 64)
        _write(os.path.join(led, "t_aside.json.killed1"), {"status": "ok", "nll": 1.0})
        _write(os.path.join(led, "t_aside.json.nodefault1"), {"status": "ok", "nll": 1.0})
        assert SS.completed_trials(cfg) == []
        refused(cfg, (), ["nothing to fit yet"])
        # a record without a status is ok, as the stage reads one: str(rec.get("status", "ok"))
        nost = _write(os.path.join(led, "t_nostatus.json"), {"trial_id": "t_nostatus", "nll": 2.0})
        assert SS.completed_trials(cfg) == ["t_nostatus"]
        os.remove(nost)
        _ok_record(cfg, "t_ok", 1.5)
        assert SS.completed_trials(cfg) == ["t_ok"]
        # (c) the request: the last line, the file, its content
        t0 = time.time()
        rc, lines = _cli(cfg)
        assert rc == 0 and lines == ["STOP REQUESTED: %s; kill running: no; %s" % (label, TAIL)], lines
        raw = open(req_path, "rb").read()
        text = raw.decode("ascii")                       # ASCII
        req = json.loads(text)
        assert sorted(req) == ["host", "kill_running", "requested_utc", "schema", "user"], req
        assert req["schema"] == "search_stop/1" and req["kill_running"] is False, req
        assert UTC_RE.match(req["requested_utc"]), req
        # the command's own clock, to the second: within 120 s of this process's (a slow machine;
        # a time zone or a local time instead of UTC would be off by 3600 s or more)
        assert abs(CM.utc_to_epoch(req["requested_utc"]) - t0) < 120, (req, t0)
        assert req["user"] == getpass.getuser() and req["host"] == socket.gethostname(), req
        keys = re.findall(r'^\s*"([a-z_]+)":', text, re.M)
        assert keys == sorted(keys), keys                # sorted keys
        assert text.endswith("\n") and "\r" not in text
        assert sorted(os.listdir(cdir)) == ["STOP_REQUEST.json", "trials"], os.listdir(cdir)
        assert SST.read_request(cfg) == req
        assert SST.status_line(cfg) == "stop requested %s (kill running: no)" % req["requested_utc"]
        # (d) the replacement: the line names the previous request's time and kill_running
        time.sleep(1.1)                                  # a later second
        rc, lines = _cli(cfg, None, True)
        req2 = json.load(open(req_path))
        assert rc == 0 and lines == [
            "STOP REQUESTED: %s; kill running: yes; %s; it replaces the request of %s (kill running: no)"
            % (label, TAIL, req["requested_utc"])], lines
        assert req2["kill_running"] is True and req2["requested_utc"] != req["requested_utc"], req2
        shown.append(lines[0])
        # (e) the status line: the end named only when SEARCH_COMPLETE.json says user_stop
        assert SST.status_line(cfg) == "stop requested %s (kill running: yes)" % req2["requested_utc"]
        _write(scp, {"ended_by": "user_stop"})
        assert SST.status_line(cfg) == ("stop requested %s (kill running: yes); the search ended "
                                        "(user_stop)" % req2["requested_utc"])
        _write(scp, {"ended_by": "rounds"})
        assert SST.status_line(cfg) == "stop requested %s (kill running: yes)" % req2["requested_utc"]
        # (f) an ended search: a request and a cancel refused, naming its ended_by; the file kept
        shown.append(refused(cfg, (), [label, "has ended", "ended_by rounds"]))
        refused(cfg, (None, False, True), [label, "has ended", "ended_by rounds"])
        _write(scp, b"\0\0")
        refused(cfg, (), ["has ended", "ended_by unknown", "cannot be read"])
        os.remove(scp)
        # (g) --cancel with --kill-running; then --cancel; then --cancel without a request
        refused(cfg, (None, True, True), ["--cancel", "--kill-running"])
        rc, lines = _cli(cfg, None, False, True)
        assert rc == 0 and len(lines) == 1 and lines[0].startswith(
            "STOP CANCELLED: %s: the request of %s (kill running: yes) is withdrawn"
            % (label, req2["requested_utc"])), lines
        assert "a trial already deleted by it stays killed" in lines[0], lines
        shown.append(lines[0])
        assert not os.path.exists(req_path) and SST.read_request(cfg) is None
        assert SST.status_line(cfg) is None
        shown.append(refused(cfg, (None, False, True), [label, "no stop request", "nothing to cancel"]))
        # (h) the file helpers by hand: write (atomic: no temporary file), read, clear
        got = SST.write_request(cfg, True, "someone", "login9", "2026-11-17T08:00:00Z")
        assert got == {"schema": "search_stop/1", "requested_utc": "2026-11-17T08:00:00Z",
                       "kill_running": True, "user": "someone", "host": "login9"}, got
        assert json.load(open(req_path)) == got and not glob.glob(req_path + ".tmp*")
        assert SST.clear_request(cfg) is True and SST.clear_request(cfg) is False
        # (i) a request file that cannot be read stands as a request without kill_running
        _write(req_path, "stop it\n")
        bad = SST.read_request(cfg)
        assert bad["kill_running"] is False and bad["requested_utc"] is None and bad.get("problem"), bad
        assert SST.kills_running(bad) is False
        line = SST.status_line(cfg)
        assert line.startswith("stop requested ? (kill running: no)") and "cannot be read" in line, line
        for v, want in ((True, True), (False, False), ("yes", False), (1, False), (None, False)):
            assert SST.kills_running({"kill_running": v}) is want, (v, want)
        assert SST.kills_running(None) is False
        os.remove(req_path)
        # (j) a list configuration (v11 searches): --search names one of them
        lroot = os.path.join(root, "list")
        os.makedirs(lroot)
        lcfg = _cfg_copy(LINDDUP, lroot, "list.json")
        views = dict(RU.search_views(lcfg))
        assert sorted(views) == ["A1", "A2", "A5"], sorted(views)
        shown.append(refused(lcfg, (), ["this configuration has the searches A1, A2, A5: give "
                                        "--search <name>"]))
        refused(lcfg, ("A9",), ["no search is named 'A9'", "A1, A2, A5"])
        refused(lcfg, ("A2",), ["search A2 (S-A2-only)", "nothing to fit yet"])
        _ok_record(views["A2"], "t_ok", 1.0)
        rc, lines = _cli(lcfg, "A2")
        assert rc == 0 and lines == ["STOP REQUESTED: search A2 (S-A2-only); kill running: no; %s"
                                     % TAIL], lines
        ltune = lcfg.data["paths"]["tune_root"]
        assert os.path.isfile(os.path.join(ltune, "A2", "S-A2-only", "STOP_REQUEST.json"))
        assert SST.read_request(views["A1"]) is None and SST.read_request(views["A5"]) is None
        refused(lcfg, ("A1",), ["search A1 (S-A1)", "nothing to fit yet"])
        # (k) through pipeline.py stop-search (cmd_stop_search): the same, by exit code
        env = dict(os.environ)
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "stop-search", "--config", lcfg.path,
                            "--search", "A2", "--cancel"], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        assert r.returncode == 0 and r.stdout.startswith("STOP CANCELLED: search A2 (S-A2-only)"), r.stdout
        assert SST.read_request(views["A2"]) is None
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "stop-search", "--config", cfg.path,
                            "--search", "A2"], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        assert r.returncode == 2 and r.stdout.startswith("REFUSED: --search A2"), r.stdout
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "stop-search", "--config", cfg.path,
                            "--kill-running"], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        assert r.returncode == 0 and r.stdout.strip() == "STOP REQUESTED: %s; kill running: yes; %s" % (
            label, TAIL), r.stdout
    finally:
        shutil.rmtree(root, ignore_errors=True)
    for ln in shown:
        print("    " + ln[:220])


def test_reference():
    """T30b (and T30d) end to end, without --kill-running."""
    R = _e2e("nokill")
    A, B, C, D, sub = _common_end(R, kill=False)
    recs = R["records"]
    # T30b: the running trial B ran to its end, ok, on its one job
    assert recs[A]["status"] == "ok" and recs[B]["status"] == "ok", (recs.get(A), recs.get(B))
    assert len(_jobs_of(R, A)) == 1 and len(_jobs_of(R, B)) == 1
    assert sub["trials"][B]["status"] == "ok" and len(sub["trials"][B]["jobs"]) == 1
    # T30d: C failed of a node fault after the request: healed (its record set aside) but never
    # resubmitted -- one job; cancelled with the waiting trials
    assert len(_jobs_of(R, C)) == 1 and len(sub["trials"][C]["jobs"]) == 1, _jobs_of(R, C)
    assert C not in recs, recs.get(C)
    aside = os.path.join(R["camp"], "trials", C + ".json.nodefault1")
    assert os.path.isfile(aside), os.listdir(os.path.join(R["camp"], "trials"))
    rec_c = json.load(open(aside))
    assert rec_c["status"] == "failed" and rec_c["error"] == "runner exited -7", rec_c
    assert "search: trial %s NODE FAULT on " % C[:12] in R["log"], R["log"][-4000:]
    assert sub["trials"][C]["status"] == "cancelled", sub["trials"][C]
    assert R["log"].count("search: submitted trial %s" % C[:12]) == 1
    sc = R["sc"]
    n = sc["n_trials"]
    assert (n["ok"], n["failed"], n["killed"], n["cancelled"]) == (2, 0, 0, 2), n
    assert sc["rounds_evaluated"] == 1
    assert "search: no trial running after the stop request of %s: 2 waiting trial(s) cancelled" % (
        sc["stop_request"]["requested_utc"]) in R["log"], R["log"][-3000:]
    for ln in R["log"].splitlines():
        if "search: stop requested" in ln or "NODE FAULT" in ln or "SEARCH COMPLETE" in ln:
            print("    " + ln.strip()[:220])
    print("    n_trials %s; finalist %s" % (json.dumps(n, sort_keys=True), sc["finalist"]["trial_id"][:12]))


def test_contract():
    """T30c end to end, with --kill-running."""
    R = _e2e("kill")
    A, B, C, D, sub = _common_end(R, kill=True)
    recs = R["records"]
    assert recs[A]["status"] == "ok", recs.get(A)
    assert json.load(open(R["path"]))["search"]["resubmit_killed"] is True
    for t in (B, C):
        jobs = _jobs_of(R, t)
        # the job deleted by the stage (the fake qdel's record of a running job), once
        assert len(jobs) == 1 and jobs[0]["comment"].startswith("Job deleted by user"), jobs
        assert jobs[0]["exit_status"] == 271, jobs
        tr = sub["trials"][t]
        assert tr["status"] == "killed" and len(tr["jobs"]) == 1, tr
        assert tr["stop_deleted"]["job"] == tr["jobs"][0] == jobs[0]["id"], tr
        assert tr["stop_deleted"]["accepted"] is True, tr
        r = recs[t]
        assert r["status"] == "killed" and r["recorded_by"] == "pipeline", r
        assert "stopped by the user (stop-search --kill-running)" in r["error"], r
        assert R["log"].count("search: trial %s: its job %s deleted (stop-search --kill-running)"
                              % (t[:12], jobs[0]["id"])) == 1, R["log"][-4000:]
        assert "search: trial %s stopped by the user (stop-search --kill-running)" % t[:12] in R["log"]
        assert R["log"].count("search: submitted trial %s" % t[:12]) == 1, "resubmitted"
    n = R["sc"]["n_trials"]
    assert (n["ok"], n["failed"], n["killed"], n["cancelled"]) == (1, 0, 2, 1), n
    assert R["sc"]["finalist"]["trial_id"] == A
    for ln in R["log"].splitlines():
        if "search: stop requested" in ln or "stop-search --kill-running" in ln or "SEARCH COMPLETE" in ln:
            print("    " + ln.strip()[:220])
    print("    n_trials %s" % json.dumps(n, sort_keys=True))


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the block is control flow over files and jobs")


def test_invariants():
    """T30e: two search views of one list configuration (block 29's graph is
    not built here: the stage driven on each view in this process)."""
    root = _tmp()
    try:
        lcfg = _cfg_copy(LINDDUP, root, "list.json")
        views = dict(RU.search_views(lcfg))
        assert [v.data["search"]["campaign"] for _, v in RU.search_views(lcfg)] == \
            ["S-A1", "S-A2-only", "S-A5-only"]
        hands = {}
        for q, nll in (("A1", 1.25), ("A2", 2.5)):
            hands[q] = _hand_search(views[q], [("t_ok", "ok", nll), ("t_run", "running", None),
                                               ("t_wait", "waiting", None)])
        rc, lines = _cli(lcfg, "A2")
        assert rc == 0, lines
        req = SST.read_request(views["A2"])
        req_bytes = open(SST.request_path(views["A2"]), "rb").read()
        tuner = _Tuner()
        with _patched(CM, "run_tuner", tuner):
            # poll 1: A2 submits nothing; A1 fills its free slot
            got = {}
            for q in ("A1", "A2"):
                stage, st, rec = hands[q]
                got[q] = stage.poll(views[q], st, rec)
            s1, st1, r1 = hands["A1"]
            s2, st2, r2 = hands["A2"]
            assert got == {"A1": "running", "A2": "running"}, got
            assert r2.submitted == [] and r2.deleted == [], (r2.submitted, r2.deleted)
            assert _statuses(st2) == {"t_ok": "ok", "t_run": "running", "t_wait": "waiting"}
            assert _lines_with(r2, "search: stop requested %s (kill running: no): no new trial; 1 running"
                               % req["requested_utc"]) and len(_lines_with(r2, "stop requested")) == 1
            assert [role for role, _ in r1.submitted] == ["trial t_wait"], r1.submitted
            assert _statuses(st1) == {"t_ok": "ok", "t_run": "running", "t_wait": "running"}
            assert not _lines_with(r1, "stop") and r1.deleted == [], r1.lines
            # poll 2: the running trials end ok -- A2 cancels its waiting trial and finishes
            # (user_stop); A1 goes on with t_wait running
            for q in ("A1", "A2"):
                stage, st, rec = hands[q]
                _ok_record(views[q], "t_run", 0.5 if q == "A1" else 1.75)
                rec.states["J_t_run"] = ("F", 0)
                got[q] = stage.poll(views[q], st, rec)
            assert got == {"A1": "running", "A2": "finished"}, got
            assert _statuses(st2) == {"t_ok": "ok", "t_run": "ok", "t_wait": "cancelled"}, _statuses(st2)
            assert r2.submitted == [], r2.submitted
            assert _statuses(st1) == {"t_ok": "ok", "t_run": "ok", "t_wait": "running"}, _statuses(st1)
            assert len(r1.submitted) == 1 and not _lines_with(r1, "stop")
        assert tuner.calls == ["finalists"], tuner.calls
        d2 = SS.SearchStage()._dirs(views["A2"])
        d1 = SS.SearchStage()._dirs(views["A1"])
        sc = json.load(open(d2["complete"]))
        assert sc["ended_by"] == "user_stop" and sc["stop_request"] == req, sc
        assert sc["n_trials"]["ok"] == 2 and sc["n_trials"]["cancelled"] == 1, sc["n_trials"]
        assert sc["finalist"]["trial_id"] == "t_run" and sc["campaign"] == "S-A2-only", sc["finalist"]
        assert st2["sub"]["phase"] == "complete" and s2.verify(views["A2"], st2)[0] is True
        assert not os.path.exists(d1["complete"]) and SST.read_request(views["A1"]) is None
        assert st1["sub"]["phase"] == "rounds" and s1.verify(views["A1"], st1)[0] is False
        # the status lines; the request kept byte for byte by the stage
        assert SST.status_line(views["A2"]) == ("stop requested %s (kill running: no); the search ended "
                                                "(user_stop)" % req["requested_utc"])
        assert SST.status_line(views["A1"]) is None and SST.status_line(views["A5"]) is None
        assert open(SST.request_path(views["A2"]), "rb").read() == req_bytes
        # the search ended: the command now refuses A2 (naming user_stop), and still takes A1
        rc, lines = _cli(lcfg, "A2", False, True)
        assert rc == 2 and "has ended (ended_by user_stop" in lines[0], lines
        rc, lines = _cli(lcfg, "A1", True)
        assert rc == 0 and lines[0].startswith("STOP REQUESTED: search A1 (S-A1); kill running: yes"), lines
        print("    A2: " + _lines_with(r2, "stop requested")[0][:200])
        print("    A2: " + _lines_with(r2, "SEARCH COMPLETE")[0][:200])
        print("    A1: submitted %s, no stop line" % ", ".join(role for role, _ in r1.submitted))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_determinism():
    raise unittest.SkipTest("no randomness in the pipeline's part (sec. 4)")


def _edge_every_kind(root):
    """Under a request (no kill) no resubmission of any kind: a walltime kill
    (271), a hold on an avoided node (94), an environment fault (96) --
    killed with the reason; a node fault (a failed record, SIGBUS) -- healed,
    never submitted, cancelled; resubmit_killed and resubmit_node_faults on."""
    cfg = _cfg_copy(GIULIA, os.path.join(root, "kinds"), "rec.json")
    assert cfg.data["search"]["resubmit_killed"] is True and cfg.data["search"]["resubmit_node_faults"]
    stage, st, rec = _hand_search(cfg, [("t_ok", "ok", 1.0), ("t_wall", "running", None),
                                        ("t_hold", "running", None), ("t_env", "running", None),
                                        ("t_nf", "running", None), ("t_wait", "waiting", None)])
    rec.states.update({"J_t_wall": ("F", 271), "J_t_hold": ("F", 94), "J_t_env": ("F", 96),
                       "J_t_nf": ("F", 0)})
    _write(os.path.join(_ledger(cfg), "t_nf.json"),
           {"trial_id": "t_nf", "status": "failed", "error": "runner exited -7",
            "config": {"lr": 0.001, "batch_size_npe": 16}})
    rc, lines = _cli(cfg)
    assert rc == 0, lines
    tuner = _Tuner()
    with _patched(CM, "run_tuner", tuner):
        r = stage.poll(cfg, st, rec)
    assert r == "finished" and rec.submitted == [] and rec.deleted == [], (r, rec.submitted, rec.joined())
    assert _statuses(st) == {"t_ok": "ok", "t_wall": "killed", "t_hold": "killed", "t_env": "killed",
                             "t_nf": "cancelled", "t_wait": "cancelled"}, _statuses(st)
    for t in ("t_wall", "t_hold", "t_env"):
        kr = json.load(open(os.path.join(_ledger(cfg), t + ".json")))
        assert kr["status"] == "killed" and "ended without a record after a stop request" in kr["error"], kr
        assert len(st["sub"]["trials"][t]["jobs"]) == 1
        assert _lines_with(rec, "search: trial %s ended without a record after a stop request" % t)
    assert os.path.isfile(os.path.join(_ledger(cfg), "t_nf.json.nodefault1"))
    assert not os.path.exists(os.path.join(_ledger(cfg), "t_nf.json"))
    assert _lines_with(rec, "search: trial t_nf NODE FAULT on ?")
    sc = json.load(open(SS.SearchStage()._dirs(cfg)["complete"]))
    assert sc["ended_by"] == "user_stop" and sc["n_trials"]["killed"] == 3 and \
        sc["n_trials"]["cancelled"] == 2 and sc["finalist"]["trial_id"] == "t_ok", sc
    assert tuner.calls == ["finalists"], tuner.calls
    return rec


def _edge_cancel_after_kill(root):
    """--kill-running, then --cancel before the deleted job is seen ended:
    the deleted trial stays killed; the search goes on (the waiting trial
    submitted)."""
    cfg = _cfg_copy(GIULIA, os.path.join(root, "cancel"), "rec.json")
    stage, st, rec = _hand_search(cfg, [("t_ok", "ok", 1.0), ("t_run", "running", None),
                                        ("t_wait", "waiting", None)])
    assert _cli(cfg, None, True)[0] == 0
    with _patched(CM, "run_tuner", _Tuner()):
        assert stage.poll(cfg, st, rec) == "running"
        assert rec.deleted == ["J_t_run"] and rec.submitted == [], (rec.deleted, rec.submitted)
        sd = st["sub"]["trials"]["t_run"]["stop_deleted"]
        assert sd["job"] == "J_t_run" and sd["accepted"] is True and UTC_RE.match(sd["utc"]), sd
        rec.states["J_t_run"] = "R"          # the scheduler still shows it: a job takes a moment to end
        assert stage.poll(cfg, st, rec) == "running"
        assert rec.deleted == ["J_t_run"], "deleted twice"
        rc, lines = _cli(cfg, None, False, True)
        assert rc == 0 and lines[0].startswith("STOP CANCELLED: "), lines
        rec.states["J_t_run"] = ("F", 271)   # now it has ended, without a record
        assert stage.poll(cfg, st, rec) == "running"
    tr = st["sub"]["trials"]["t_run"]
    assert tr["status"] == "killed" and len(tr["jobs"]) == 1, tr
    kr = json.load(open(os.path.join(_ledger(cfg), "t_run.json")))
    assert "stopped by the user (stop-search --kill-running)" in kr["error"], kr
    assert [role for role, _ in rec.submitted] == ["trial t_wait"], rec.submitted
    assert len(_lines_with(rec, "was withdrawn (stop-search --cancel)")) == 1, rec.joined()
    assert "stop_seen" not in st["sub"]
    return rec


def _edge_gate_end(root):
    """A request at the gate's end (a hand-made file: the command refuses one
    with no completed trial): no round 0; _finish fails as today."""
    cfg = _cfg_copy(GIULIA, os.path.join(root, "gate"), "rec.json")
    stage = SS.SearchStage()
    dirs = stage._dirs(cfg)
    _write(os.path.join(dirs["gate"], "b16", "A1_seed0_probe.json"),
           {"step_seconds": 0.01, "val_row_seconds": 0.001, "maxrss_gb": 0.5, "steps_per_epoch": 3,
            "projected_trial_seconds": 1.0})
    sub = {"identity": _identity(cfg), "phase": "gate", "rounds": [], "trials": {},
           "clock_start_utc": None, "ended_by": None,
           "gate": {"probes": {"16": {"job": None, "reused": True,
                                      "probe_json": os.path.join(dirs["gate"], "b16", "A1_seed0_probe.json")}},
                    "verdict": None, "started_utc": CM.utc_now()}}
    st = {"status": "running", "jobs": [], "sub": sub, "resubmits": 0}
    rec = _Rec(st)
    SST.write_request(cfg, False, "someone", "login9", CM.utc_now())
    tuner = _Tuner()
    with _patched(CM, "run_tuner", tuner):
        assert stage.poll(cfg, st, rec) == "finished"
    assert tuner.calls == [] and sub["rounds"] == [] and rec.submitted == [], (tuner.calls, sub["rounds"])
    assert sub["phase"] == "failed" and sub["fail"].startswith(
        "the search ended (user_stop) with no completed trial"), sub.get("fail")
    assert _lines_with(rec, "search: budget gate passed") and _lines_with(rec, "search: stop requested")
    ok, why = stage.verify(cfg, st)
    assert not ok and "no completed trial" in why, why
    assert json.load(open(dirs["rounds_file"]))["ended_by"] == "user_stop"
    return rec, why


def _edge_start(root):
    """The start under a request (the gate passed, no round yet): no round 0;
    the request withdrawn: the next poll proposes it, the next submits."""
    cfg = _cfg_copy(GIULIA, os.path.join(root, "start"), "rec.json")
    stage = SS.SearchStage()
    sub = {"identity": _identity(cfg), "gate": {"verdict": {"ok": True}}, "rounds": []}
    st = {"status": "running", "jobs": [], "sub": sub, "resubmits": 0}
    rec = _Rec(st)
    SST.write_request(cfg, False, "someone", "login9", CM.utc_now())
    tuner = _Tuner(propose_ids=["t_a", "t_b"])
    with _patched(CM, "run_tuner", tuner):
        stage.start(cfg, st, rec)
        assert tuner.calls == [] and sub["phase"] == "rounds" and sub["rounds"] == [], (tuner.calls, sub)
        assert SST.clear_request(cfg) is True
        assert stage.poll(cfg, st, rec) == "running"
        assert tuner.calls == ["propose"] and [len(r["trial_ids"]) for r in sub["rounds"]] == [2], sub["rounds"]
        assert stage.poll(cfg, st, rec) == "running"
    assert sorted(role for role, _ in rec.submitted) == ["trial t_a", "trial t_b"], rec.submitted
    return rec


def _edge_reopen(root):
    """A user_stop finish that did not complete, then a fresh state (a
    --retry): with the request still there the search finishes again; with
    it withdrawn, the search goes on."""
    out = {}
    for case in ("standing", "withdrawn"):
        cfg = _cfg_copy(GIULIA, os.path.join(root, "reopen_" + case), "rec.json")
        stage = SS.SearchStage()
        dirs = stage._dirs(cfg)
        ident = _identity(cfg)
        _write(dirs["rounds_file"], {"schema": "pipeline_rounds/1", "identity": ident,
                                     "config_sha256": cfg.sha256, "clock_start_utc": CM.utc_now(),
                                     "gate": {"verdict": {"ok": True}}, "ended_by": "user_stop",
                                     "rounds": [{"round": 0, "trial_ids": ["t_ok", "t_wait"]}]})
        _ok_record(cfg, "t_ok", 1.0)
        _write(os.path.join(dirs["pending"], "t_wait.json"),
               {"trial_id": "t_wait", "campaign": "S-A1", "config": {"lr": 0.001}})
        if case == "standing":
            SST.write_request(cfg, False, "someone", "login9", CM.utc_now())
        st = {"status": "running", "jobs": [], "sub": {"identity": ident}, "resubmits": 0}
        rec = _Rec(st)
        with _patched(CM, "run_tuner", _Tuner()):
            stage.start(cfg, st, rec)
            assert st["sub"]["phase"] == "rounds" and st["sub"]["ended_by"] is None, st["sub"]
            assert json.load(open(dirs["rounds_file"]))["ended_by"] is None
            r = stage.poll(cfg, st, rec)
        if case == "standing":
            assert r == "finished" and rec.submitted == [], (r, rec.submitted)
            sc = json.load(open(dirs["complete"]))
            assert sc["ended_by"] == "user_stop" and sc["n_trials"]["cancelled"] == 1, sc
        else:
            assert r == "running" and [role for role, _ in rec.submitted] == ["trial t_wait"], rec.submitted
            assert not os.path.exists(dirs["complete"])
        out[case] = rec
    return out


def _edge_unreadable(root):
    """A request file that cannot be read: a stop without kill, its problem
    logged; nothing deleted, nothing submitted."""
    cfg = _cfg_copy(GIULIA, os.path.join(root, "unreadable"), "rec.json")
    stage, st, rec = _hand_search(cfg, [("t_ok", "ok", 1.0), ("t_run", "running", None),
                                        ("t_wait", "waiting", None)])
    _write(SST.request_path(cfg), b"\0" * 16)
    with _patched(CM, "run_tuner", _Tuner()):
        assert stage.poll(cfg, st, rec) == "running"
        assert stage.poll(cfg, st, rec) == "running"
    assert rec.deleted == [] and rec.submitted == [], (rec.deleted, rec.submitted)
    assert len(_lines_with(rec, "cannot be read")) == 1 and \
        len(_lines_with(rec, "search: stop requested ? (kill running: no): no new trial; 1 running")) == 1, \
        rec.joined()
    return rec


def test_edge_cases():
    root = _tmp()
    try:
        rec = _edge_every_kind(root)
        print("    every kind: " + _lines_with(rec, "ended without a record after a stop request")[0][:160])
        rec = _edge_cancel_after_kill(root)
        print("    cancel after kill: " + _lines_with(rec, "withdrawn")[0][:160])
        rec, why = _edge_gate_end(root)
        print("    gate end: " + why[:160])
        _edge_start(root)
        print("    start: no round 0 under a request; proposed once withdrawn")
        _edge_reopen(root)
        print("    re-open: the request standing finishes again; withdrawn, the search goes on")
        rec = _edge_unreadable(root)
        print("    unreadable: " + _lines_with(rec, "cannot be read")[0][:160])
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- runner ---

def main(argv=None):
    names = list(argv if argv is not None else sys.argv[1:])
    checks = [(n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)]
    if names:
        checks = [(n, o) for n, o in checks if n in names]
    print("== %s" % os.path.basename(__file__))
    print("python %s | %s" % (platform.python_version(), platform.platform()))
    results = []
    try:
        for name, func in checks:
            t0 = time.time()
            try:
                func()
                status, detail = "PASS", ""
            except unittest.SkipTest as exc:
                status, detail = "SKIP", str(exc)
            except NotImplementedError as exc:
                status, detail = "TODO", str(exc)
            except AssertionError as exc:
                status, detail = "FAIL", str(exc) or traceback.format_exc()
            except Exception:
                status, detail = "ERROR", traceback.format_exc()
            results.append((name, status, time.time() - t0, detail))
    finally:
        for R in _E2E.values():
            if R.get("root"):
                shutil.rmtree(R["root"], ignore_errors=True)
    width = max([len(r[0]) for r in results] + [4])
    for name, status, sec, detail in results:
        head = (detail.strip().splitlines() or [""])[0]
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, sec, head[:200])).rstrip())
    for name, status, _, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()[-6000:]))
    counts = dict((s, sum(1 for r in results if r[1] == s)) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (counts[s], s.lower()) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")))
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
