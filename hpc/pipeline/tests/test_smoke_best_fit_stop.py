#!/usr/bin/env python3
"""Smoke test for block 30's amendment 6 of specs/SPEC.md (D-185): a best fit stopped by
hand -- `pipeline.py stop-best-fit` (stages/best_fit_stop.py) and the best
fit's start and poll while a request stands (stages/best_fit.py).

Checks (specs/SPEC.md block 30, amendment 6, "Test oracles" T30f-T30l)
    test_known_answer   T30f, the command, by hand (no scheduler): each
                        refusal -- a synthetic configuration, --search
                        missing or unknown on a list configuration, given on
                        a null one, the search not ended, SEARCH_COMPLETE.json
                        unreadable or naming no finalist, the fit complete,
                        the fit stopped -- each with nothing written (every
                        file's bytes and mtime); a complete fit of another
                        trial does not refuse; the request's file and content
                        (keys, schema, finalist, ISO 8601 UTC, ASCII, sorted
                        keys, no temporary file left), the root created when
                        absent; the repeat (exit 0, the file unchanged); a
                        request naming another trial, and one that cannot be
                        read, replaced; status_line; the command through
                        pipeline.py (exit 0 and 2)
    test_reference      T30g, the poll in this process (the best fit of A2 on a
                        copy of linddup.json; a recorder for the scheduler):
                        member 1 ok, 2-4 running, 5-10 waiting; after the
                        request the three running jobs are deleted once,
                        nothing is submitted, the poll runs while they have
                        not ended; then 2 and 4 are killed with the reason,
                        3 (its record written before the deletion took
                        effect) stays ok, 5-10 are cancelled; the poll
                        finishes; BEST_FIT_STOPPED.json's content; no
                        ensemble, no tuner, no BEST_FIT_COMPLETE.json; verify
                        ok with the reason; status and search_lines
    test_invariants     T30j and T30k: a request naming another trial is not
                        acted on (one line in two polls); one that cannot be
                        read is; the file removed after the stop was seen:
                        the stop goes on; done(s) with the stopped record of
                        the search's finalist, of another trial (not done;
                        the root set aside at the start, D-095), unreadable
                        (the same); G_q's done(s) unchanged
    test_contract       T30l at the sequencer's level under the fake PBS: the
                        committed linddup.json (A1's cap 440, D-185) with its
                        roots here; a stand-in for A1's search (48-core jobs
                        as long as fits() allows) and the real best fits of
                        A2 and A5 on a state holding three running members
                        each: A1 runs 3 jobs; after stop-best-fit for A2 and
                        A5 the six member jobs are deleted (Exit_status 271),
                        both best fits end done, nothing is submitted for
                        them, and A1 runs 9 jobs of 48 (432 cores, inside the
                        stages' 440)
    test_edge_cases     T30h and T30i: under a request a hold on an avoided
                        node (exit 94), an environment fault (exit 96), a node
                        fault (SIGBUS) and a walltime kill (271) are each
                        killed, never resubmitted -- where the same members
                        without a request are all submitted again; the start
                        under a request submits nothing (the root holding only
                        the request, or a member record of the finalist): the
                        members without a record cancelled, the stopped
                        record, the poll finished
    test_convergence    not a discretisation: skipped
    test_determinism    no randomness in the pipeline's part: skipped

Mutations (BFS_MUTATION=<name>; each must make at least one check FAIL):
    no_delete, ignore_request, done_ignores_stopped, cli_no_state_check,
    start_submits, stopped_any_trial.

Run
    python3 hpc/pipeline/tests/test_smoke_best_fit_stop.py [check ...]
        Under a minute; the standard library only (test_contract uses the
        fake PBS of tests/fake_pbs/). Nothing trains, nothing touches
        davinci.

Pure ASCII, LF only.
"""

import contextlib
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_FAKE = os.path.join(_HERE, "fake_pbs")
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C                      # noqa: E402
import pbs as PBS                       # noqa: E402
import pipeline as P                    # noqa: E402
import stages as S                      # noqa: E402
import stages.best_fit as BF            # noqa: E402
import stages.best_fit_stop as BFS      # noqa: E402
import stages.common as CM              # noqa: E402
import state as ST                      # noqa: E402
from stages import runs as RU           # noqa: E402

PYTHON = sys.executable
LINDDUP = os.path.join(_PIPE, "projects", "linddup", "linddup.json")
SYNTH = os.path.join(_PIPE, "projects", "synth", "synth.json")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
TID = "f" * 12 + "0123456789abcdef"        # the search's finalist
OTHER = "e" * 12 + "fedcba9876543210"      # another trial
TAIL = ("at its next tick (every driver.poll_seconds) the sequencer deletes the running member "
        "jobs, submits none and ends the stage without an ensemble")

# --------------------------------------------------------------- mutations ---

MUTATION = os.environ.pop("BFS_MUTATION", "")
_PATCHES = {
    # the stage never deletes a member's job
    "no_delete": "BFS.delete_job = lambda pbs, job_id: False\n",
    # the stage ignores the request (as before block 30, amendment 6)
    "ignore_request": "BF.BestFitStage._stop_request = lambda self, cfg, st, log: None\n",
    # done(s) ignores the stopped record
    "done_ignores_stopped": ("BF.BestFitStage._stopped_done = lambda self, cfg: "
                             "(False, 'ignored')\n"),
    # the command writes whatever the fit's state (complete, stopped)
    "cli_no_state_check": ("_orig = BFS.read_stopped\n"
                           "BFS.read_stopped = lambda view: (None, None)\n"
                           "BFS.COMPLETE_FILE = 'NO_SUCH_FILE.json'\n"),
    # the start submits the waiting members although a request stands
    "start_submits": ("_orig_start = BF.BestFitStage.start\n"
                      "def _start(self, cfg, st, pbs):\n"
                      "    keep = BF.BestFitStage._stop_request\n"
                      "    BF.BestFitStage._stop_request = lambda s, c, t, l: None\n"
                      "    try:\n"
                      "        return _orig_start(self, cfg, st, pbs)\n"
                      "    finally:\n"
                      "        BF.BestFitStage._stop_request = keep\n"
                      "BF.BestFitStage.start = _start\n"),
    # done(s) takes a stopped record of any trial
    "stopped_any_trial": ("_orig_sd = BF.BestFitStage._stopped_done\n"
                          "def _sd(self, cfg):\n"
                          "    p = self._dirs(cfg)['stopped']\n"
                          "    return (True, 'stopped by hand (any trial): ' + p)\n"
                          "BF.BestFitStage._stopped_done = _sd\n"),
}
if MUTATION:
    if MUTATION not in _PATCHES:
        raise SystemExit("unknown BFS_MUTATION %r; known: %s" % (MUTATION, ", ".join(sorted(_PATCHES))))
    exec(_PATCHES[MUTATION], {"BF": BF, "BFS": BFS})


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_bfstop_")


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


def _load_json(path):
    with open(path, "rb") as fh:
        return C._strip_docs(json.loads(fh.read().decode("ascii")))


def _cfg_copy(src, root, name, edit=None):
    """A committed project file with its work, runs, tune and banks roots
    under `root` (`edit(d)` applied), loaded as the sequencer loads it."""
    d = _load_json(src)
    for k in ("work_root", "runs_root", "tune_root", "banks_root"):
        d["paths"][k] = os.path.join(root, k)
    if edit is not None:
        edit(d)
    return C.load_config(_write(os.path.join(root, name), d), synthetic_ok=True)


def _null_edit(d):
    d["searches"] = None
    d["search"]["campaign"] = "S-A1"
    d["search"]["arm"] = "A1"


def _view(cfg, q):
    return dict(RU.search_views(cfg))[q]


def _stage(q, cfg=None):
    """The real best-fit stage of search q (RunBestFitStage): on the run of
    `cfg`'s searches list (its max_cores: the stage's core group), else on a
    run without a cap."""
    if cfg is not None:
        return RU.stage_classes()[1](dict((r["name"], r) for r in cfg.data["searches"])[q])
    return RU.stage_classes()[1]({"name": q, "campaign": {"A1": "S-A1", "A2": "S-A2-only",
                                                          "A5": "S-A5-only"}[q]})


class _EndedSearch(S.Stage):
    """A search that has ended (its stage done in the state): a node of the
    graph for the best fit's dependency."""

    max_resubmits = 0

    def __init__(self, name, core_group):
        self.name = name
        self.deps = ()
        self.core_group = core_group

    def is_done(self, cfg):
        return True, "ended (the test's stand-in)"


def _identity():
    return {"n_train": 100, "split_hash": "a" * 64, "contract_digest": "b" * 64, "n_sel": 10,
            "n_report": 5}


def _search_complete(view, finalist=TID):
    camp = view.data["search"]["campaign"]
    p = os.path.join(view.data["paths"]["tune_root"], camp, "SEARCH_COMPLETE.json")
    return _write(p, {"campaign": camp, "ended_by": "user_stop", "identity": _identity(),
                      "finalist": ({"trial_id": finalist, "nll": 1.25,
                                    "config": {"lr": 0.001, "batch_size_npe": 16}}
                                   if finalist else None)})


def _member_record(view, seed, L=2.5):
    d = BF.BestFitStage()._member_dir(view, seed)
    return _write(os.path.join(d, "linddup_A2_seed%d.json" % int(seed)), {"L": float(L)})


class _Rec(object):
    """The sequencer's recorder as the stage sees it (pipeline.Submitter):
    the scheduler's answers from `states` (job id -> 'R', or ('F', exit
    status)); every submission (into the stage record's jobs, as the
    Submitter does) and every deletion recorded; `fits` always."""

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
                                "submitted_utc": CM.utc_now(), "ncpus": 48})
        self.states[jid] = "R"
        return jid

    def delete(self, job_id):
        self.deleted.append(job_id)
        self.states[job_id] = ("F", 271)
        return True

    def joined(self):
        return "\n".join(self.lines)


class _NoTool(object):
    """CM.run_tuner and CM.E.run_in_env replaced: a call is recorded and,
    unless `argv_ok`, fails the check (no tuner and no ensemble under a
    stop). With `argv_ok` the tuner's `argv` writes its --argv-file."""

    def __init__(self, argv_ok=False):
        self.calls = []
        self.argv_ok = argv_ok

    def tuner(self, cfg, args, log_path=None, timeout=None):
        self.calls.append(("tuner", args[0]))
        if self.argv_ok and args[0] == "argv":
            _write(args[args.index("--argv-file") + 1], "python3 runner.py\n")
            return ""
        raise AssertionError("the stage called the tuner's %r under a stop" % (args[0],))

    def env(self, *a, **k):
        self.calls.append(("run_in_env",) + tuple(str(x) for x in a[:1]))
        raise AssertionError("the stage ran a tool in the environment (the ensemble?) under a stop")


@contextlib.contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield value
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def _no_tools(argv_ok=False):
    nt = _NoTool(argv_ok)
    with _patched(CM, "run_tuner", nt.tuner), _patched(CM.E, "run_in_env", nt.env):
        yield nt


def _running_fit(view, spec):
    """A best fit of `view` in its phase members, by hand: `spec` {seed:
    kind}, kind `ok` (a record), `running` (its job J<seed> running) or
    `waiting`; finalist_config.json naming TID. Returns (st, rec)."""
    stage = BF.BestFitStage()
    dirs = stage._dirs(view)
    _search_complete(view)
    _write(dirs["finalist_config"], {"trial_id": TID, "campaign": view.data["search"]["campaign"],
                                     "nll_search": 1.25, "config": {"lr": 0.001},
                                     "search_complete": dirs["search_complete"]})
    sub = {"finalist": {"trial_id": TID, "nll_search": 1.25, "identity": _identity(),
                        "config_file": dirs["finalist_config"]},
           "members": {}, "phase": "members"}
    st = {"status": "running", "jobs": [], "sub": sub, "resubmits": 0}
    rec = _Rec(st)
    for seed in view.data["best_fit"]["member_seeds"]:
        kind = spec.get(seed, "waiting")
        m = {"jobs": [], "resubmits": 0, "status": kind}
        if kind == "ok":
            m["record"] = _member_record(view, seed)
            m["L"] = 2.5
        if kind == "running":
            jid = "J%d" % int(seed)
            m["jobs"] = [jid]
            st["jobs"].append({"id": jid, "role": "member seed %d" % int(seed), "cmd": [],
                               "finished": False, "ncpus": 48,
                               "log": os.path.join(dirs["jobs"], "member_seed%d.0.log" % int(seed)),
                               "exit_status": None, "walltime_s": None,
                               "submitted_utc": CM.utc_now()})
            rec.states[jid] = "R"
        sub["members"][str(seed)] = m
    return st, rec


def _cli(cfg, search=None):
    lines = []
    rc = BFS.cli(cfg, search, out=lines.append)
    return rc, lines


def _statuses(st):
    return dict((int(k), m["status"]) for k, m in st["sub"]["members"].items())


# ---------------------------------------------------------------- T30f ---

def _refused(cfg, root, search, needle):
    before = _tree(root)
    rc, lines = _cli(cfg, search)
    assert rc == 2 and len(lines) == 1 and lines[0].startswith("REFUSED: ") and \
        lines[0].endswith("nothing was written"), (rc, lines)
    assert needle in lines[0], (needle, lines)
    assert _tree(root) == before, "a refusal wrote something"
    return lines[0]


def test_known_answer():
    root = _tmp()
    shown = []
    try:
        lst = _cfg_copy(LINDDUP, root, "linddup_here.json")
        v2 = _view(lst, "A2")
        req_p = BFS.request_path(v2)
        stopped_p = BFS.stopped_path(v2)
        # the refusals, each with nothing written
        syn = _cfg_copy(SYNTH, root, "synth_here.json")
        shown.append(_refused(syn, root, "A1", "synthetic configuration"))
        _refused(lst, root, None, "give --search <name>")
        _refused(lst, root, "A9", "no search is named 'A9'")
        null = _cfg_copy(LINDDUP, root, "null_here.json", _null_edit)
        _refused(null, root, "A2", "a single search")
        _refused(null, root, None, "has not ended")
        shown.append(_refused(lst, root, "A2", "the search has not ended"))
        assert not os.path.lexists(os.path.dirname(req_p)), "a refusal created the root"
        sc = _search_complete(v2)
        _write(sc, "{not json")
        _refused(lst, root, "A2", "cannot be read")
        _search_complete(v2, finalist=None)
        _refused(lst, root, "A2", "names no finalist")
        _search_complete(v2)
        _write(os.path.join(os.path.dirname(req_p), "BEST_FIT_COMPLETE.json"),
               {"finalist": {"trial_id": TID}})
        shown.append(_refused(lst, root, "A2", "is complete"))
        os.remove(os.path.join(os.path.dirname(req_p), "BEST_FIT_COMPLETE.json"))
        _write(stopped_p, {"finalist": {"trial_id": TID}})
        shown.append(_refused(lst, root, "A2", "was stopped already"))
        os.remove(stopped_p)
        shutil.rmtree(os.path.dirname(req_p))
        # the request: the root created, its content
        assert BFS.status_line(v2) is None
        rc, lines = _cli(lst, "A2")
        assert rc == 0 and lines == ["STOP REQUESTED: best fit of search A2 (finalist %s); %s"
                                     % (TID, TAIL)], (rc, lines)
        shown.append(lines[0][:120] + " ...")
        raw = open(req_p, "rb").read()
        assert all(b < 128 for b in raw), "not ASCII"
        obj = json.loads(raw.decode("ascii"))
        assert sorted(obj) == ["finalist", "host", "requested_utc", "schema", "user"], obj
        assert obj["schema"] == "best_fit_stop/1" and obj["finalist"] == TID, obj
        assert UTC_RE.match(obj["requested_utc"]), obj
        keys = re.findall(r'"([a-z_]+)":', raw.decode("ascii"))
        assert keys == sorted(keys), keys
        assert not [f for f in os.listdir(os.path.dirname(req_p)) if f.endswith(".tmp")]
        assert BFS.read_request(v2) == obj
        assert BFS.status_line(v2) == "stop requested %s (finalist %s)" % (obj["requested_utc"], TID)
        # the repeat: unchanged
        before = _tree(root)
        rc, lines = _cli(lst, "A2")
        assert rc == 0 and lines[0].endswith("(already requested %s; unchanged)"
                                             % obj["requested_utc"]), lines
        assert _tree(root) == before, "the repeat rewrote the file"
        # a request naming another trial, and one that cannot be read: replaced
        _write(req_p, {"schema": "best_fit_stop/1", "requested_utc": "2026-11-01T00:00:00Z",
                       "finalist": OTHER, "user": "u", "host": "h"})
        rc, lines = _cli(lst, "A2")
        assert rc == 0 and lines[0].endswith("; it replaces the request of 2026-11-01T00:00:00Z "
                                             "for trial %s" % OTHER), lines
        assert json.load(open(req_p))["finalist"] == TID
        _write(req_p, "{broken")
        rc, lines = _cli(lst, "A2")
        assert rc == 0 and "it replaces a request file that could not be read" in lines[0], lines
        assert json.load(open(req_p))["finalist"] == TID
        # a complete fit of another trial does not refuse (the start sets its root aside)
        _write(os.path.join(os.path.dirname(req_p), "BEST_FIT_COMPLETE.json"),
               {"finalist": {"trial_id": OTHER}})
        os.remove(req_p)
        rc, lines = _cli(lst, "A2")
        assert rc == 0 and lines[0].startswith("STOP REQUESTED"), lines
        os.remove(os.path.join(os.path.dirname(req_p), "BEST_FIT_COMPLETE.json"))
        # status_line once stopped
        req = json.load(open(req_p))
        _write(stopped_p, {"finalist": {"trial_id": TID}})
        assert BFS.status_line(v2) == ("stop requested %s (finalist %s); stopped "
                                       "(BEST_FIT_STOPPED.json)" % (req["requested_utc"], TID))
        os.remove(stopped_p)
        # through pipeline.py
        v5 = _view(lst, "A5")
        _search_complete(v5)
        env = dict(os.environ)
        env["PIPELINE_SKIP_CONDA"] = "1"
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "stop-best-fit",
                            "--config", lst.path, "--search", "A5"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=120)
        assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == (
            "STOP REQUESTED: best fit of search A5 (finalist %s); %s" % (TID, TAIL)), r.stdout
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "stop-best-fit",
                            "--config", lst.path], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        assert r.returncode == 2 and "REFUSED: " in r.stdout, r.stdout
    finally:
        shutil.rmtree(root, ignore_errors=True)
    for ln in shown:
        print("    " + ln[:170])


# ---------------------------------------------------------------- T30g ---

def test_reference():
    root = _tmp()
    try:
        cfg = _cfg_copy(LINDDUP, root, "linddup_here.json")
        view = _view(cfg, "A2")
        stage = _stage("A2")
        st, rec = _running_fit(view, {1: "ok", 2: "running", 3: "running", 4: "running"})
        with _no_tools() as nt:
            # without a request the poll runs (the waiting members do not fit here)
            rec.fits = lambda n: False
            assert stage.poll(cfg, st, rec) == "running"
            assert rec.submitted == [] and rec.deleted == []
            rec.fits = lambda n: True
            rc, lines = _cli(cfg, "A2")
            assert rc == 0, lines
            req = json.load(open(BFS.request_path(view)))
            # the first poll under the request: the three jobs deleted, nothing submitted
            assert stage.poll(cfg, st, rec) == "running"
            assert rec.deleted == ["J2", "J3", "J4"] and rec.submitted == [], (rec.deleted, rec.submitted)
            for s in (2, 3, 4):
                sd = st["sub"]["members"][str(s)]["stop_deleted"]
                assert sd["job"] == "J%d" % s and sd["accepted"] is True and UTC_RE.match(sd["utc"])
            assert st["sub"]["stop_request"] == req
            first = [ln for ln in rec.lines if "stop requested" in ln]
            assert first == ["best_fit_A2: stop requested %s (stop-best-fit): the best fit of trial "
                             "%s is abandoned: 3 running member job(s) to delete, nothing "
                             "submitted, no ensemble" % (req["requested_utc"], TID)], rec.lines
            # member 3 wrote its record before the deletion took effect
            _member_record(view, 3, L=3.75)
            assert stage.poll(cfg, st, rec) == "finished"
            assert rec.deleted == ["J2", "J3", "J4"] and rec.submitted == [], "deleted once, none submitted"
            want = dict((s, "cancelled") for s in range(5, 11))
            want.update({1: "ok", 2: "killed", 3: "ok", 4: "killed"})
            assert _statuses(st) == want, _statuses(st)
            for s in (2, 4):
                assert BFS.STOP_KILLED in st["sub"]["members"][str(s)]["why"]
            assert st["sub"]["members"]["3"]["L"] == 3.75 and st["sub"]["phase"] == "stopped"
            # the stopped record
            dirs = stage._dirs(view)
            obj = json.load(open(dirs["stopped"]))
            assert obj["schema"] == "best_fit_stopped/1" and obj["dataset"] == "linddup"
            assert obj["finalist"]["trial_id"] == TID and obj["stop_request"] == req
            assert obj["config_sha256"] == cfg.sha256 and UTC_RE.match(obj["stopped_utc"])
            assert [m["seed"] for m in obj["members"]] == list(range(1, 11))
            assert dict((m["seed"], m["status"]) for m in obj["members"]) == want
            assert [m["seed"] for m in obj["members"] if m["stop_deleted"]] == [2, 3, 4]
            assert not os.path.exists(dirs["complete"]) and not os.path.exists(dirs["ensemble"])
            assert nt.calls == [], nt.calls
            end = [ln for ln in rec.lines if "BEST FIT STOPPED" in ln]
            assert end == ["best_fit_A2: BEST FIT STOPPED (stop-best-fit): members ok 2, killed 2, "
                           "cancelled 6; no ensemble; %s" % dirs["stopped"]], rec.lines
            # verify, and the poll again (phase stopped)
            ok, why = stage.verify(cfg, st)
            assert ok and why == ("stopped by hand (stop-best-fit, requested %s): no ensemble; %s"
                                  % (req["requested_utc"], dirs["stopped"])), (ok, why)
            assert stage.poll(cfg, st, rec) == "finished" and rec.submitted == []
            assert stage.is_done(cfg) == (True, why)
        # status and search_lines
        line = BFS.status_line(view)
        assert line == "stop requested %s (finalist %s); stopped (BEST_FIT_STOPPED.json)" % (
            req["requested_utc"], TID), line
        state = ST.State.new(cfg.data["paths"]["work_root"], "linddup", cfg.path, cfg.sha256)
        lines = P.search_lines(cfg, state)
        assert "best fit A2: " + line in lines, lines
        assert not [ln for ln in lines if ln.startswith("best fit A1") or ln.startswith("best fit A5")]
        print("    " + end[0][:170])
        print("    verify: " + why[:150])
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------- T30j, T30k ---

def test_invariants():
    root = _tmp()
    try:
        cfg = _cfg_copy(LINDDUP, root, "linddup_here.json")
        view = _view(cfg, "A2")
        stage = _stage("A2")
        dirs = stage._dirs(view)
        # T30j (1): a request naming another trial is not acted on
        st, rec = _running_fit(view, dict((s, "running" if s <= 3 else "ok") for s in range(1, 11)))
        _write(dirs["stop_request"], {"schema": "best_fit_stop/1", "requested_utc":
                                      "2026-11-01T00:00:00Z", "finalist": OTHER, "user": "u",
                                      "host": "h"})
        with _no_tools():
            assert stage.poll(cfg, st, rec) == "running" and stage.poll(cfg, st, rec) == "running"
        assert rec.deleted == [] and rec.submitted == [] and "stop_request" not in st["sub"]
        ign = [ln for ln in rec.lines if "not acted on" in ln]
        assert ign == ["best_fit_A2: the stop request %s names trial %s, not the finalist %s this "
                       "best fit fits: not acted on" % (dirs["stop_request"], OTHER, TID)], rec.lines
        assert any("not acted on" in ln for ln in stage.dry_run(cfg, st)), stage.dry_run(cfg, st)
        # T30j (2): a request that cannot be read is acted on
        _write(dirs["stop_request"], "{broken")
        with _no_tools():
            assert stage.poll(cfg, st, rec) == "running"
        assert rec.deleted == ["J1", "J2", "J3"], rec.deleted
        assert any("cannot be read" in ln for ln in rec.lines), rec.lines
        # T30j (3): the file removed after the stop was seen: the stop goes on
        os.remove(dirs["stop_request"])
        with _no_tools():
            assert stage.poll(cfg, st, rec) == "finished"
        assert rec.submitted == [] and _statuses(st) == dict(
            (s, "killed" if s <= 3 else "ok") for s in range(1, 11)), _statuses(st)
        obj = json.load(open(dirs["stopped"]))
        assert obj["stop_request"].get("problem") and obj["stop_request"]["finalist"] is None, obj
        ok, why = stage.verify(cfg, st)
        assert ok and why.startswith("stopped by hand (stop-best-fit, requested ?)"), why
        # T30k: done(s) and the root
        assert stage.is_done(cfg)[0] is True
        assert stage._root_owner(view, TID) is None, "the stopped root of the finalist is kept"
        obj["finalist"]["trial_id"] = OTHER
        _write(dirs["stopped"], obj)
        done, why = stage.is_done(cfg)
        assert not done and why == ("%s names the finalist %s, but the search's finalist is %s (%s)"
                                    % (dirs["stopped"], OTHER, TID, dirs["search_complete"])), why
        os.remove(dirs["finalist_config"])     # only the stopped record names the trial now
        assert stage._root_owner(view, TID) == "its BEST_FIT_STOPPED.json names trial %s" % OTHER
        st2 = {"status": "pending", "jobs": [], "sub": {}, "resubmits": 0}
        lines = []
        stage._follow_finalist(view, st2, lines.append)
        assert not os.path.lexists(dirs["root"]), "the root of another trial was not set aside"
        aside = glob.glob(dirs["root"] + ".set_aside*")
        assert len(aside) == 1 and os.path.isfile(os.path.join(aside[0], "BEST_FIT_STOPPED.json"))
        assert lines and "moved aside" in lines[0], lines
        _write(dirs["stopped"], "{broken")
        done, why = stage.is_done(cfg)
        assert not done and why.startswith("%s unreadable" % dirs["stopped"]), why
        assert stage._root_owner(view, TID).startswith("its BEST_FIT_STOPPED.json is unreadable")
        # G_q (block 18's other arm): its done(s) does not read a stopped record
        alt = RU.stage_classes()[2]({"name": "A2", "campaign": "S-A2-only"})
        adirs = alt._dirs(alt._view(cfg))
        _write(adirs["stopped"], {"finalist": {"trial_id": TID, "search_finalist": {"trial_id": TID}}})
        done, why = alt.is_done(cfg)
        assert not done and why == "%s absent" % adirs["complete"], (done, why)
        assert alt._stop_request(alt._view(cfg), {"sub": {"finalist": {"trial_id": TID}}},
                                 lambda t: None) is None
        print("    another trial: " + ign[0][:150])
        print("    G_q: " + why[:150])
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- T30l ---

class ThrottleStage(S.Stage):
    """A stand-in for A1's search (test_smoke_runs.py's): `n_jobs` jobs of
    `ncpus` cores, as many at once as its Submitter's fits() allows, each a
    sleep under the fake PBS, in the core group given."""

    max_resubmits = 0

    def __init__(self, name, deps, n_jobs, ncpus, core_group, sleep):
        self.name = name
        self.deps = tuple(deps)
        self.n_jobs = int(n_jobs)
        self.ncpus = int(ncpus)
        self.core_group = core_group
        self.sleep = float(sleep)

    def _dir(self, cfg):
        return os.path.join(cfg.data["paths"]["work_root"], "throttle", self.name)

    def is_done(self, cfg):
        p = os.path.join(self._dir(cfg), "DONE")
        return os.path.isfile(p), p

    def _qsub(self, cfg, k):
        d = self._dir(cfg)
        script = _write(os.path.join(d, "job_%d.sh" % k), "#!/bin/bash\nsleep %g\necho done\n"
                        % self.sleep)
        return ["qsub", "-N", "%s_%d" % (self.name, k), "-q", "cpu",
                "-l", "select=1:ncpus=%d:mem=1gb" % self.ncpus, "-l", "walltime=00:10:00",
                "-j", "oe", "-o", os.path.join(d, "job_%d.log" % k), script]

    def _submit_waiting(self, cfg, st, pbs):
        sub = st["sub"]
        while sub["next"] < self.n_jobs:
            if not pbs.fits(self.ncpus):
                return
            k = sub["next"]
            pbs.submit(self._qsub(cfg, k), role="job %d" % k,
                       log_path=os.path.join(self._dir(cfg), "job_%d.log" % k), ncpus=self.ncpus)
            sub["next"] = k + 1

    def start(self, cfg, st, pbs):
        st["sub"]["next"] = 0
        if not pbs.fits(self.ncpus):
            raise S.Deferred("a job of %d cores does not fit now" % self.ncpus)
        self._submit_waiting(cfg, st, pbs)

    def poll(self, cfg, st, pbs):
        for job in st["jobs"]:
            if not job.get("finished") and pbs.status(job["id"]).finished:
                job["finished"] = True
        self._submit_waiting(cfg, st, pbs)
        if st["sub"]["next"] < self.n_jobs or any(not j.get("finished") for j in st["jobs"]):
            return "running"
        _write(os.path.join(self._dir(cfg), "DONE"), "done\n")
        return "finished"

    def verify(self, cfg, st):
        ok, why = self.is_done(cfg)
        return (True, why) if ok else (False, "killed: " + why)


@contextlib.contextmanager
def _fake_pbs(spool):
    saved = dict(os.environ)
    os.environ["PATH"] = _FAKE + os.pathsep + saved.get("PATH", "")
    os.environ["FAKE_PBS_SPOOL"] = spool
    os.environ["FAKE_PBS_DELAY"] = "0.2"
    os.environ["FAKE_PBS_ENFORCE_WALLTIME"] = "0"
    os.environ["PIPELINE_SKIP_CONDA"] = "1"
    os.environ.pop("PBS_JOBID", None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _records(spool):
    out = []
    for p in glob.glob(os.path.join(spool, "*.json")):
        if os.path.basename(p)[:-5].isdigit():
            with open(p) as fh:
                out.append(json.load(fh))
    return sorted(out, key=lambda r: r["num"])


def _member_jobs(cfg, q, seeds, sleep):
    """Three running members of search q's best fit, as fake PBS jobs."""
    view = _view(cfg, q)
    dirs = BF.BestFitStage()._dirs(view)
    out = []
    for seed in seeds:
        log = os.path.join(dirs["jobs"], "member_seed%d.0.log" % seed)
        script = _write(os.path.join(dirs["jobs"], "member_seed%d.sh" % seed),
                        "#!/bin/bash\nsleep %g\n" % sleep)
        argv = ["qsub", "-N", "bf_linddu_s%d" % seed, "-q", "cpu", "-l",
                "select=1:ncpus=48:mem=1gb", "-l", "walltime=10:00:00", "-j", "oe", "-o", log,
                script]
        out.append((seed, PBS.submit(argv), argv, log))
    return out


def test_contract():
    root = _tmp()
    spool = os.path.join(root, "spool")
    os.makedirs(spool)
    try:
        cfg = _cfg_copy(LINDDUP, root, "linddup_here.json")
        groups = dict((n, RU.run_core_group(v.run)) for n, v in RU.search_views(cfg))
        assert groups["A1"] == (440, ("search_A1", "best_fit_A1")), groups
        assert groups["A2"][0] == groups["A5"][0] == 148 and P.stage_budget(cfg) == 440
        wr = cfg.data["paths"]["work_root"]
        throttle = ThrottleStage("search_A1", (), 12, 48, groups["A1"], 25.0)
        graph = [throttle, _EndedSearch("search_A2", groups["A2"]), _stage("A2", cfg),
                 _EndedSearch("search_A5", groups["A5"]), _stage("A5", cfg)]
        assert [g.core_group for g in graph] == [groups["A1"], groups["A2"], groups["A2"],
                                                 groups["A5"], groups["A5"]]
        ticks, lines = [], []

        def probe():
            stt = ST.State.load(wr)
            ticks.append({"*": P.cores_in_flight(stt)[0],
                          "A1": P.cores_in_flight(stt, names=groups["A1"][1])[0],
                          "A2": P.cores_in_flight(stt, names=groups["A2"][1])[0],
                          "A5": P.cores_in_flight(stt, names=groups["A5"][1])[0],
                          "status": dict((n, stt.stage(n)["status"]) for n in
                                         ("search_A1", "best_fit_A2", "best_fit_A5"))})

        with _fake_pbs(spool), _no_tools():
            state = ST.State.new(wr, "linddup", cfg.path, cfg.sha256)
            fixture_ids = []
            for q in ("A2", "A5"):
                view = _view(cfg, q)
                st, _rec = _running_fit(view, {})       # every member waiting, then three running
                for seed, jid, argv, log in _member_jobs(cfg, q, (1, 2, 3), 600):
                    m = st["sub"]["members"][str(seed)]
                    m["status"], m["jobs"] = "running", [jid]
                    st["jobs"].append({"id": jid, "role": "member seed %d" % seed, "cmd": argv,
                                       "log": log, "finished": False, "exit_status": None,
                                       "walltime_s": None, "submitted_utc": CM.utc_now(),
                                       "ncpus": 48})
                    fixture_ids.append(jid)
                rec = state.stage("best_fit_" + q)
                rec.update(st)
                rec["started_utc"] = CM.utc_now()
                state.set_status("search_" + q, "done", "ended (the test's fixture)")
            state.save()
            # phase 1: the best fits hold 288 cores; A1 gets the rest, 3 jobs
            rc1 = P.run(cfg, graph=graph, poll_seconds=0.3, log=lines.append, on_tick=probe,
                        max_ticks=4)
            before = list(ticks)
            # the user's two commands
            for q in ("A2", "A5"):
                rc, out = _cli(cfg, q)
                assert rc == 0, out
            # phase 2: the stop; then A1 takes the stages' share
            rc2 = P.run(cfg, graph=graph, poll_seconds=0.3, log=lines.append, on_tick=probe,
                        max_ticks=40)
            after = ticks[len(before):]
            recs = _records(spool)
            for r in recs:                              # leave no fake job behind
                if r.get("state") != "F":
                    subprocess.run([os.path.join(_FAKE, "qdel"), str(r["num"])],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert rc1 == P.EXIT_OK and rc2 == P.EXIT_OK, (rc1, rc2, lines[-6:])
        assert before and max(t["A1"] for t in before) == 144, before
        assert all(t["A2"] == 144 and t["A5"] == 144 for t in before), before
        done_at = [i for i, t in enumerate(after)
                   if t["status"]["best_fit_A2"] == "done" and t["status"]["best_fit_A5"] == "done"]
        assert done_at, (after[-1], lines[-12:])
        assert max(t["A1"] for t in after[done_at[0]:]) == 432, [t["A1"] for t in after]
        assert max(t["*"] for t in ticks) <= 440, [t["*"] for t in ticks]
        assert all(t["A2"] == t["A5"] == 0 for t in after[done_at[0]:]), after[done_at[0]:]
        joined = "\n".join(lines)
        for q in ("A2", "A5"):
            assert re.search(r"best_fit_%s: done: stopped by hand \(stop-best-fit, requested "
                             r"[0-9TZ:-]+\): no ensemble; " % q, joined), joined[-3000:]
            assert "best_fit_%s: BEST FIT STOPPED (stop-best-fit): members ok 0, killed 3, " \
                   "cancelled 7; no ensemble" % q in joined, joined[-3000:]
        by_id = dict((r["id"], r) for r in recs)
        for jid in fixture_ids:
            r = by_id.get(jid)
            assert r is not None and r.get("state") == "F" and r.get("exit_status") == 271, (jid, r)
        names = [r.get("name") for r in recs]
        assert sum(1 for n in names if n and n.startswith("bf_")) == 6, names   # the fixture's only
        print("    before the stop: A1 %d cores, A2 %d, A5 %d; after: A1 up to %d, the project up "
              "to %d (stages' share 440)" % (max(t["A1"] for t in before), before[-1]["A2"],
                                             before[-1]["A5"],
                                             max(t["A1"] for t in after),
                                             max(t["*"] for t in ticks)))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------- T30h, T30i ---

def _ended_fit(root, q, with_request):
    """Members 1-4 running whose jobs ended -- a hold (94), an environment
    fault (96), a node fault (SIGBUS) and a walltime kill (271) -- and 5-10
    ok; the poll with or without a request. (st, rec, statuses)."""
    cfg = _cfg_copy(LINDDUP, root, "linddup_%s.json" % ("req" if with_request else "none"))
    view = _view(cfg, q)
    stage = _stage(q)
    st, rec = _running_fit(view, dict((s, "running" if s <= 4 else "ok") for s in range(1, 11)))
    logs = {1: "RUN_ARGV host: n1\nRUN_ARGV AVOID: n1 is on the avoid list\n",
            2: "RUN_ARGV ENV FAULT: no conda on n2\n",
            3: "RUN_ARGV host: n3\nRUN_ARGV EXIT: 135\n",
            4: "RUN_ARGV host: n4\n=>> PBS: job killed: walltime exceeded limit\n"}
    for s, code in ((1, 94), (2, 96), (3, 135), (4, 271)):
        _write(st["jobs"][s - 1]["log"], logs[s])
        rec.states["J%d" % s] = ("F", code)
    if with_request:
        assert _cli(cfg, q)[0] == 0
    with _no_tools(argv_ok=True) as nt:
        r = stage.poll(cfg, st, rec)
    return st, rec, r, nt


def test_edge_cases():
    root = _tmp()
    try:
        # T30h: without a request the four are submitted again; under one, none is
        st0, rec0, r0, nt0 = _ended_fit(os.path.join(root, "a"), "A2", False)
        assert r0 == "running" and len(rec0.submitted) == 4, (rec0.submitted, rec0.lines[-6:])
        st1, rec1, r1, nt1 = _ended_fit(os.path.join(root, "b"), "A2", True)
        assert r1 == "finished" and rec1.submitted == [] and rec1.deleted == [], rec1.lines
        assert nt1.calls == [], nt1.calls
        assert _statuses(st1) == dict((s, "killed" if s <= 4 else "ok") for s in range(1, 11))
        for s in range(1, 5):
            assert BFS.STOP_ENDED in st1["sub"]["members"][str(s)]["why"], st1["sub"]["members"][str(s)]
        print("    without a request: %d member(s) submitted again; under one: 0, the four "
              "killed (%s)" % (len(rec0.submitted), BFS.STOP_ENDED))
        # T30i: the start under a request -- the root holding only the request
        cfg = _cfg_copy(LINDDUP, os.path.join(root, "c"), "linddup_here.json")
        view = _view(cfg, "A5")
        stage = _stage("A5")
        _search_complete(view)
        assert _cli(cfg, "A5")[0] == 0
        st = {"status": "pending", "jobs": [], "sub": {}, "resubmits": 0}
        rec = _Rec(st)
        with _no_tools() as nt:
            stage.start(cfg, st, rec)
            assert stage.poll(cfg, st, rec) == "finished"
        assert rec.submitted == [] and nt.calls == [] and st["sub"]["phase"] == "stopped"
        assert _statuses(st) == dict((s, "cancelled") for s in range(1, 11)), _statuses(st)
        assert stage.verify(cfg, st)[0] is True
        assert os.path.isfile(stage._dirs(view)["finalist_config"])
        # T30i: a member record of the finalist already there
        cfg = _cfg_copy(LINDDUP, os.path.join(root, "d"), "linddup_here.json")
        view = _view(cfg, "A5")
        _search_complete(view)
        dirs = stage._dirs(view)
        _write(dirs["finalist_config"], {"trial_id": TID})
        _member_record(view, 4, L=4.5)
        assert _cli(cfg, "A5")[0] == 0
        st = {"status": "pending", "jobs": [], "sub": {}, "resubmits": 0}
        rec = _Rec(st)
        with _no_tools() as nt:
            stage.start(cfg, st, rec)
            assert stage.poll(cfg, st, rec) == "finished"
        assert rec.submitted == [] and nt.calls == []
        want = dict((s, "cancelled") for s in range(1, 11))
        want[4] = "ok"
        assert _statuses(st) == want, _statuses(st)
        assert os.path.isdir(dirs["root"]) and not glob.glob(dirs["root"] + ".set_aside*")
        obj = json.load(open(dirs["stopped"]))
        assert [m["L"] for m in obj["members"] if m["seed"] == 4] == [4.5]
        print("    start under a request: nothing submitted; " + [ln for ln in rec.lines
                                                                   if "STOPPED" in ln][0][:120])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_determinism():
    raise unittest.SkipTest("no randomness in the pipeline's part")


# ---------------------------------------------------------------- runner ---

def main(argv=None):
    names = list(argv if argv is not None else sys.argv[1:])
    checks = [(n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)]
    if names:
        checks = [(n, o) for n, o in checks if n in names]
    print("== %s%s" % (os.path.basename(__file__), (" [BFS_MUTATION=%s]" % MUTATION) if MUTATION
                       else ""))
    print("python %s | %s" % (platform.python_version(), platform.platform()))
    results = []
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
