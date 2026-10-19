#!/usr/bin/env python3
"""Smoke test for block 1's amendment (d) of 2026-10-19 (D-099): the whole
pipeline under resources.max_cores, the sequencer's own job included.

Checks (specs/SPEC.md block 1, amended 2026-10-19 (d))
    test_known_answer   the committed Giulia configuration: v9, max_cores
                        200, the stages' share 196, the room of each sized
                        job (4 trials, 4 members, 4 re-detection tasks, 20
                        bank tasks), the dry run's cores lines (on a copy
                        whose paths lie in a temporary folder: on davinci
                        the committed work root holds the run's state)
    test_reference      cores_of_select, cores_of_argv, job_cores and
                        cores_in_flight against hand-counted values; the
                        Submitter's budget, free and fits, its guard, its
                        refusal of a job that can never start
    test_launchers      the cores Sbi-extractor's launchers and job scripts
                        ask for, read from the clone in SBIX_CLONE (SKIP
                        without one)
    test_convergence    not a discretisation: skipped
    test_invariants     the driver chain, two independent stages whose jobs
                        do not fit together: the second waits, the fake
                        PBS's records never hold more than the stages'
                        share at once, every job once
    test_contract       the dry run's and --status's cores lines; the
                        state records each job's cores; the waiting note
                        logged once and cleared when the stage starts
    test_determinism    the count from a state read back equals the count
                        in memory, and a job recorded before D-099 (no
                        ncpus) counts from its qsub line; a driver killed
                        while a stage waits: its successor keeps it waiting
                        and runs it once
    test_edge_cases     a stage that submits without asking (the guard
                        stops it, nothing submitted); a job larger than the
                        whole share, at a start and in a poll (stopped with
                        the reason, no driver crash); the switch from 13
                        trials in flight to the cap; the configuration's
                        refusals of the cap

Run
    python3 hpc/pipeline/tests/test_smoke_block1c.py
        Prints a PASS / FAIL / ERROR / TODO / SKIP table and the message of
        every failure. Exit code 1 if any check failed, errored or is TODO.
        About 2 minutes. SBIX_CLONE=<Sbi-extractor checkout> for
        test_launchers.
    MUTATION=<name> python3 hpc/pipeline/tests/test_smoke_block1c.py
        Re-runs the checks with one piece of the code disabled; the check
        named in _PATCHES must then FAIL: fits_always, footprint_sum,
        done_counted, never_fits, guard.

Everything runs under the fake PBS (tests/fake_pbs/, first on PATH) in a
temporary folder, on the synthetic graph of synthetic_graph.py (A and B
independent with SYNTH_PARALLEL=1, C after both). Nothing touches davinci.

Pure ASCII, LF only; stdlib only.
"""

import copy
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
_FAKE = os.path.join(_HERE, "fake_pbs")
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C              # noqa: E402
import pbs as PBS               # noqa: E402
import pipeline as P            # noqa: E402
import stages as S              # noqa: E402
import stages.common as CM      # noqa: E402
import state as ST              # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
PYTHON = sys.executable
SG = os.path.join(_HERE, "synthetic_graph.py")

# A mutation disables one piece of the code; the check named after it must FAIL.
MUTATION = os.environ.get("MUTATION", "")
_PATCHES = {
    # test_invariants: B would be submitted beside A, over the share
    "fits_always": "P.Submitter.fits = lambda self, n: True\n",
    # test_reference: a gate after its array counted beside it (the sum, not the larger)
    "footprint_sum": ("def _cif(state):\n"
                      "    t, n, u = 0, 0, []\n"
                      "    for st in state.data.get('stages', {}).values():\n"
                      "        if st.get('status') in ('done', 'skipped'):\n"
                      "            continue\n"
                      "        for j in st.get('jobs', []):\n"
                      "            if j.get('finished'):\n"
                      "                continue\n"
                      "            c = P.job_cores(j)\n"
                      "            if c is None:\n"
                      "                u.append(j['id'])\n"
                      "            else:\n"
                      "                t, n = t + c, n + 1\n"
                      "    return t, n, u\n"
                      "P.cores_in_flight = _cif\n"),
    # test_reference: a done stage's stale record would hold its cores for ever
    "done_counted": ("_orig_cif = P.cores_in_flight\n"
                     "def _cif(state):\n"
                     "    import copy as _c\n"
                     "    s2 = _c.deepcopy(state)\n"
                     "    for st in s2.data.get('stages', {}).values():\n"
                     "        st['status'] = 'running'\n"
                     "    return _orig_cif(s2)\n"
                     "P.cores_in_flight = _cif\n"),
    # test_edge_cases: a job larger than the share would wait for ever
    "never_fits": ("def _fits(self, n):\n"
                   "    b = self.budget()\n"
                   "    return b is None or n is None or int(n) <= b - self.in_flight()\n"
                   "P.Submitter.fits = _fits\n"),
    # test_edge_cases: a stage that submits without asking would get its job
    "guard": ("def _sub(self, argv, role, log_path=None, cwd=None, env=None, ncpus=None,\n"
              "         after=None):\n"
              "    n = ncpus if ncpus is not None else PBS.cores_of_argv(argv)\n"
              "    job_id = self._pbs.submit(argv, cwd=cwd, env=env)\n"
              "    self._state.add_job(self._name, job_id, role, argv, log=log_path, ncpus=n,\n"
              "                        after=after)\n"
              "    self._state.save()\n"
              "    self._log('%s: submitted %s -> %s' % (self._name, role, job_id))\n"
              "    return job_id\n"
              "P.Submitter.submit = _sub\n"),
}
if MUTATION:
    if MUTATION not in _PATCHES:
        raise SystemExit("unknown MUTATION %r; known: %s" % (MUTATION, ", ".join(sorted(_PATCHES))))
    exec(_PATCHES[MUTATION], {"P": P, "PBS": PBS, "ST": ST, "S": S, "C": C})


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_b1c_")


def _project(root, max_cores=4, walltime="00:01:00", poll=1):
    """A synthetic configuration under `root`: the sequencer's job 1 core,
    every sized job 1 core, resources.max_cores `max_cores` (so the stages
    share max_cores - 1); sbi_dir this checkout (the driver jobs run its
    jobs/run_argv.pbs)."""
    with open(GIULIA, "rb") as fh:
        raw = json.loads(fh.read().decode("ascii"))
    cfg = C._strip_docs(raw)
    cfg["dataset"] = "synth"
    for k in list(cfg["paths"]):
        cfg["paths"][k] = os.path.join(root, k)
    cfg["paths"]["sbi_dir"] = _REPO
    cfg["real"]["cohort_config"] = os.path.join(root, "cohort.json")
    cfg["real"]["extract_root"] = os.path.join(root, "extracted")
    for k in ("sim_main", "out_root", "tools"):
        cfg["sim"][k] = os.path.join(root, k)
    cfg["driver"] = {"queue": "test", "ncpus": 1, "mem": "1gb", "walltime": walltime,
                     "poll_seconds": int(poll)}
    cfg["search"]["trial"]["ncpus"] = 1
    cfg["best_fit"]["ncpus"] = 1
    cfg["sim"]["ncpus"] = 1
    cfg["bank"]["verify"]["ncpus"] = 1
    cfg["resources"] = {"max_cores": int(max_cores)}
    path = os.path.join(root, "synth.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    return path, cfg


def _giulia_here(root):
    """The committed Giulia configuration with every path under `root`: its
    dry run then reads no state of the cluster's (on davinci the committed
    file's work root holds the run's state, which a dry run would read and,
    after a configuration change, refuse) and prints the same lines in the
    sandbox and on davinci."""
    with open(GIULIA, "rb") as fh:
        raw = json.loads(fh.read().decode("ascii"))
    cfg = C._strip_docs(raw)
    for k in list(cfg["paths"]):
        cfg["paths"][k] = os.path.join(root, k)
    cfg["real"]["cohort_config"] = os.path.join(root, "cohort.json")
    cfg["real"]["extract_root"] = os.path.join(root, "extracted")
    for k in ("sim_main", "out_root", "tools"):
        cfg["sim"][k] = os.path.join(root, k)
    path = os.path.join(root, "giulia.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    return path


def _env(root, sleep="3", **extra):
    env = dict(os.environ)
    env["PATH"] = _FAKE + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = os.path.join(root, "spool")
    env["FAKE_PBS_DELAY"] = "0.5"
    env["FAKE_PBS_ENFORCE_WALLTIME"] = "1"
    env["PIPELINE_SKIP_CONDA"] = "1"
    env["PIPELINE_PYTHON"] = PYTHON
    env["PYTHONPATH"] = _HERE + os.pathsep + env.get("PYTHONPATH", "")
    env["SYNTH_JOB_SLEEP"] = sleep
    for k in ("SYNTH_REFUSE", "SYNTH_SILENT_KILL", "SYNTH_CRASH_POLL", "SYNTH_PARALLEL",
              "SYNTH_STAGE_NCPUS", "SYNTH_NO_CHECK", "SYNTH_POLL_ASK", "MUTATION_PATCH",
              "PBS_JOBID", "NCPUS"):
        env.pop(k, None)
    if MUTATION:
        env["MUTATION_PATCH"] = _PATCHES[MUTATION]
    env.update(dict((k, str(v)) for k, v in extra.items()))
    return env


def _pl(root, env, *args, **kw):
    """synthetic_graph.py <verb> --config <root>/synth.json [args]."""
    verb = args[0]
    cmd = [PYTHON, SG, verb, "--config", os.path.join(root, "synth.json")] + list(args[1:])
    proc = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True,
                          timeout=kw.get("timeout", 120))
    return proc.returncode, proc.stdout


def _foreground(root, env, timeout=120):
    """synthetic_graph.py --config ... : P.run in this process's child, no
    driver chain (the edge cases: quicker, the same run loop)."""
    cmd = [PYTHON, SG, "--config", os.path.join(root, "synth.json"), "--poll-seconds", "0.2"]
    proc = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return proc.returncode, proc.stdout


def _plog(cfg):
    return os.path.join(cfg["paths"]["work_root"], "pipeline.log")


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _wait_for(cfg, text, timeout=90.0, count=1):
    end = time.time() + timeout
    while time.time() < end:
        log = _read(_plog(cfg))
        if log.count(text) >= count:
            return log
        time.sleep(0.25)
    raise AssertionError("after %.0f s pipeline.log does not hold %r x%d; its tail:\n%s"
                         % (timeout, text, count, "\n".join(_read(_plog(cfg)).splitlines()[-25:])))


def _jobs(root):
    """The fake PBS's job records, oldest first."""
    out = []
    for p in glob.glob(os.path.join(root, "spool", "*.json")):
        b = os.path.basename(p)[:-5]
        if b.isdigit():
            try:
                with open(p) as fh:
                    out.append(json.load(fh))
            except (OSError, ValueError):
                pass
    return sorted(out, key=lambda j: j["num"])


def _named(root, name):
    return [j for j in _jobs(root) if j["name"] == name]


def _wait_quiet(root, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if all(j["state"] == "F" for j in _jobs(root)):
            return
        time.sleep(0.25)
    raise AssertionError("jobs still live after %.0f s: %r"
                         % (timeout, [(j["id"], j["name"], j["state"]) for j in _jobs(root)
                                      if j["state"] != "F"]))


def _sweep(root, env):
    for _ in range(4):
        live = [j["id"] for j in _jobs(root) if j["state"] != "F"]
        if not live:
            return
        subprocess.run(["qdel"] + live, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1.0)


def _record_cores(rec):
    """The cores a fake PBS record can hold at once, from its -l and -J."""
    per = 0
    for r in rec.get("resources", []):
        c = PBS.cores_of_select(r)
        if c is not None:
            per += c
    argv = ["qsub", "-l", "select=1:ncpus=%d" % per]
    if rec.get("array_spec"):
        argv += ["-J", rec["array_spec"]]
    return PBS.cores_of_argv(argv + ["x"])


def _peak(records):
    """The most cores the records held at once, queued, held or running:
    from each one's submission to its end (a finish before a submission at
    one instant: the sequencer saw it first)."""
    ev = []
    for r in records:
        c = _record_cores(r)
        ev.append((r["submitted"], 1, c))
        ev.append((r["finished"] if r["finished"] else float("inf"), 0, -c))
    peak, now = 0, 0
    for _t, _k, d in sorted(ev):
        now += d
        peak = max(peak, now)
    return peak


class _Lines(object):
    def __init__(self):
        self.lines = []

    def __call__(self, text):
        self.lines.append(text)

    def text(self):
        return "\n".join(self.lines)


class _NoPBS(object):
    """A pbs_module that must not be reached (no select: no adoption)."""

    def __init__(self):
        self.calls = []

    def submit(self, argv, cwd=None, env=None):
        self.calls.append(list(argv))
        return "%d.fake" % (100 + len(self.calls))


def _trial_line(n=48):
    return ["qsub", "-N", "t_x", "-q", "cpu", "-l", "select=1:ncpus=%d:mem=160gb" % n,
            "-l", "walltime=120:00:00", "-j", "oe", "-o", "/x/t.log", "run_argv.pbs"]


def _state(root, stages):
    """A state of hand-written stage records {name: (status, [jobs])}."""
    st = ST.State.new(os.path.join(root, "wr"), "synth", "/x.json", "a" * 64)
    for name, (status, jobs) in stages.items():
        rec = st.stage(name)
        rec["status"] = status
        rec["jobs"] = [dict(j) for j in jobs]
    return st


def _job(jid, ncpus=None, cmd=None, after=None, finished=False):
    j = {"id": jid, "role": "x", "cmd": list(cmd or ["launcher.sh"]), "finished": finished}
    if ncpus is not None:
        j["ncpus"] = ncpus
    if after:
        j["after"] = after
    return j


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = C.load_config(GIULIA)
    d = cfg.data
    assert d["config_version"] == 9 and d["resources"] == {"max_cores": 200}, d.get("resources")
    assert d["driver"]["ncpus"] == 4
    assert P.stage_budget(cfg) == 196, "200 - 4, by hand"
    # the room of each sized job: floor(196 / 48) = 4 (4 x 48 = 192 <= 196 < 240)
    assert d["search"]["trial"]["ncpus"] == 48 and d["search"]["max_concurrent_trials"] == 25
    assert CM.cap_room(cfg, 48, 25) == 4, "trials: min(25, 4)"
    assert d["best_fit"]["ncpus"] == 48 and CM.cap_room(cfg, 48) == 4, "members"
    assert d["sim"]["ncpus"] == 48 and d["sim"]["concurrency"] == 20
    assert CM.cap_room(cfg, 48, 20) == 4, "re-detection tasks: min(20, 4)"
    assert d["bank"]["concurrency"] == 20 and CM.cap_room(cfg, 1, 20) == 20, "bank tasks of 1 core"
    assert d["bank"]["verify"]["ncpus"] == 8 and CM.cap_room(cfg, 8) == 24
    root = _tmp()
    try:
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config",
                            _giulia_here(root), "--dry-run"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=300)
    finally:
        shutil.rmtree(root)
    for want in ("cores: at most 200 at once (resources.max_cores, D-099): the sequencer's job 4, "
                 "the stages' jobs 196; in flight per the state file: 0 in 0 job(s)",
                 "cores (D-099): at most 4 trial(s) at once -- max_concurrent_trials 25, and the cap "
                 "leaves room for 4 of 48 cores each; the gate's probes likewise",
                 "cores (D-099): at most 4 member(s) at once -- the cap leaves room for 4 of 48 "
                 "cores each",
                 "cores (D-099): the test task 48; the array at CONCURRENCY 4 = min(sim.concurrency "
                 "20, the cap's room 4) of 48 cores each"):
        assert want in r.stdout, "%r not in the dry run:\n%s" % (want, r.stdout[-4000:])


def test_reference():
    # ---- one -l value: the sum of K x N over its chunks, by hand ----------
    for text, want in (("select=1:ncpus=48:mem=180gb", 48), ("select=2:ncpus=4:mem=8gb", 8),
                       ("select=1:ncpus=2+3:ncpus=1", 5), ("select=ncpus=4", 4),
                       ("ncpus=6", 6), ("select=1:ncpus=2:mem=4gb,walltime=01:00:00", 2),
                       ("walltime=01:00:00", None), ("select=1:mem=4gb", None)):
        assert PBS.cores_of_select(text) == want, (text, PBS.cores_of_select(text), want)
    # ---- a qsub line: the -l cores x min(%c, the array's elements) --------
    base = ["qsub", "-N", "x", "-l", "select=1:ncpus=2:mem=4gb"]
    for extra, want in ((["-J", "0-9%3"], 6), (["-J", "0-9"], 20), (["-J", "0-9:2"], 10),
                        (["-J", "0-9:2%10"], 10), (["-J", "1-4%2"], 4), ([], 2),
                        (["-l", "walltime=00:10:00"], 2)):
        argv = base + extra + ["s.pbs"]
        assert PBS.cores_of_argv(argv) == want, (extra, PBS.cores_of_argv(argv), want)
    assert PBS.cores_of_argv(["qsub", "-N", "x", "s.pbs"]) is None
    assert PBS.cores_of_argv(["launch_bank.sh", "array"]) is None
    # ---- a recorded job: its ncpus, else its qsub line ----------------------
    assert P.job_cores({"ncpus": 12, "cmd": _trial_line(48)}) == 12
    assert P.job_cores({"cmd": _trial_line(48)}) == 48, "a job recorded before D-099"
    assert P.job_cores({"cmd": ["launch_bank.sh", "array"]}) is None and P.job_cores({}) is None
    # ---- the count over a hand-built state ----------------------------------
    root = _tmp()
    try:
        st = _state(root, {
            # a done stage's record its polls never marked finished: no cores
            "S_done": ("done", [_job("1.x", ncpus=100)]),
            "S_skip": ("skipped", [_job("2.x", ncpus=100)]),
            # two trials in flight (one recorded before D-099), one finished
            "search": ("running", [_job("3.x", ncpus=48), _job("4.x", cmd=_trial_line(48)),
                                   _job("5.x", ncpus=48, finished=True)]),
            # an array of 3 and its gate of 8 after it: the larger, 8; an array of
            # 20 and its gate of 8: 20; a gate after a finished array: 8
            "bank": ("running", [_job("6[].x", ncpus=3), _job("7.x", ncpus=8, after="6[].x"),
                                 _job("8[].x", ncpus=20), _job("9.x", ncpus=8, after="8[].x"),
                                 _job("10[].x", ncpus=3, finished=True),
                                 _job("11.x", ncpus=8, after="10[].x")]),
            # a chain 4 -> 6 -> 10: 10; two jobs of 3 and 4 after one of 2: 7
            "chain": ("running", [_job("12.x", ncpus=4), _job("13.x", ncpus=6, after="12.x"),
                                  _job("14.x", ncpus=10, after="13.x"),
                                  _job("15.x", ncpus=2), _job("16.x", ncpus=3, after="15.x"),
                                  _job("17.x", ncpus=4, after="15.x")]),
            # a failed stage's job still counts (it may run); one of unknown cores
            "S_failed": ("failed", [_job("18.x", ncpus=5)]),
            "S2": ("running", [_job("19[].x", cmd=["launch_sim_reextract.sh", "array"])]),
            "pending": ("pending", []),
        })
        # by hand: 48 + 48 + max(3, 8) + max(20, 8) + 8 + max(4, max(6, 10)) +
        # max(2, 3 + 4) + 5 = 96 + 8 + 20 + 8 + 10 + 7 + 5 = 154; counted: 3.x,
        # 4.x, 6-9, 11, 12-17, 18 = 14 jobs; unknown: 19[].x
        got = P.cores_in_flight(st)
        assert got == (154, 14, ["19[].x"]), got
        # ---- the Submitter on it under a cap of 200 (driver 4): 196 - 154 = 42
        path, raw = _project(root, max_cores=200)
        d = json.loads(json.dumps(raw))
        d["driver"]["ncpus"] = 4
        with open(path, "w") as fh:
            json.dump(d, fh)
        cfg = C.load_config(path)
        lines = _Lines()
        nop = _NoPBS()
        sub = P.Submitter(nop, st, "search", lines, cfg)
        assert (sub.budget(), sub.in_flight(), sub.free()) == (196, 154, 42)
        assert sub.fits(42) and not sub.fits(43) and sub.fits(None)
        try:
            sub.submit(_trial_line(48), role="trial y")
            raise AssertionError("48 cores with 42 free must raise CapExceeded")
        except P.CapExceeded as exc:
            assert "search: trial y asks 48 cores; 42 of 196 free under resources.max_cores" \
                in str(exc), exc
        assert not nop.calls, "nothing reached the scheduler"
        jid = sub.submit(_trial_line(40), role="trial z", log_path="/x/z.log")
        assert nop.calls and st.stage("search")["jobs"][-1]["ncpus"] == 40 and jid == "101.fake"
        assert sub.free() == 2
        sub.record("20[].x", "an array", ["launch.sh", "array"], ncpus=30)   # no check: a launcher's
        sub.record("21.x", "its gate", ["launch.sh", "array"], ncpus=8, after="20[].x")
        assert st.stage("search")["jobs"][-1]["after"] == "20[].x"
        assert sub.free() == 2 - 30, sub.free()
        try:
            sub.fits(197)
            raise AssertionError("197 cores can never fit in 196: must raise")
        except P.CapExceeded as exc:
            assert ("search: a job of 197 cores can never start: the stages' jobs share 196 cores "
                    "(resources.max_cores 200 less the sequencer's job, 4)") in str(exc), exc
        # ---- without a configuration (the unit tests' Submitter): everything fits
        free = P.Submitter(nop, st, "search", lines)
        assert free.budget() is None and free.free() is None and free.fits(10 ** 6)
        # ---- the status line ------------------------------------------------
        assert P.cores_line(cfg, st) == (
            "cores: at most 200 at once (resources.max_cores, D-099): the sequencer's job 4, the "
            "stages' jobs 196; in flight per the state file: 224 in 17 job(s); 1 job(s) of "
            "unknown cores, not counted: 19[].x"), P.cores_line(cfg, st)
    finally:
        shutil.rmtree(root)


def test_launchers():
    clone = os.environ.get("SBIX_CLONE")
    if not clone or not os.path.isdir(clone):
        raise unittest.SkipTest("no Sbi-extractor checkout: set SBIX_CLONE (the launchers' cores "
                                "are read from its files)")
    # the facts of Sbi-extractor@e9aeb55, read by hand: a well 2 cores, the
    # aggregation 1, the bank's task 1 and gate 8, the re-detection's gate 8
    ex = os.path.join(clone, "extractor")
    assert CM.pbs_header_ncpus(os.path.join(ex, "run_extractor_array_mea.pbs")) == 2
    assert CM.pbs_header_ncpus(os.path.join(ex, "run_cohort_manifest.pbs")) == 1
    lb = os.path.join(clone, "export_windows", "launch_bank.sh")
    assert CM.shell_default(lb, "NCPUS") == "1" and CM.shell_default(lb, "GATE_NCPUS") == "8"
    assert CM.shell_default(lb, "CONCURRENCY") == "20"
    gate = os.path.join(clone, "sim_reextract", "run_sim_reextract_gate.pbs")
    assert CM.pbs_header_ncpus(gate) == 8
    ls = os.path.join(clone, "sim_reextract", "launch_sim_reextract.sh")
    assert CM.shell_default(ls, "NCPUS") == "48" and CM.shell_default(ls, "CONCURRENCY") == "20"
    # the extraction launcher prints `qsub -J 0-<n-1>`, which the stage reads
    with open(os.path.join(ex, "launch_stage_d.sh")) as fh:
        assert 'ARR_CMD=(qsub -J "0-$((n - 1))")' in fh.read()
    # the readers' refusals
    try:
        CM.shell_default(lb, "NOT_A_VARIABLE")
        raise AssertionError("must raise")
    except RuntimeError as exc:
        assert "no NOT_A_VARIABLE=" in str(exc), exc
    try:
        CM.pbs_header_ncpus(lb)
        raise AssertionError("must raise")
    except RuntimeError as exc:
        assert "no #PBS -l select=...:ncpus=N line" in str(exc), exc


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the cap has no convergence order")


def test_invariants():
    root = _tmp()
    path, cfg = _project(root, max_cores=4)          # the stages share 3
    env = _env(root, sleep="3", SYNTH_PARALLEL="1", SYNTH_STAGE_NCPUS="2")
    try:
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        log = _wait_for(cfg, "the chain ends", timeout=120)
        assert "PIPELINE DONE" in log, log
        assert log.count("B: waits for cores: job asks 2 cores; 1 free") == 1, log
        _wait_quiet(root)
        stage_jobs = [j for j in _jobs(root) if j["name"].startswith("synth_")]
        for name in ("synth_A", "synth_B", "synth_C"):
            assert len(_named(root, name)) == 1, (name, _named(root, name))
        a, b, c = [_named(root, n)[0] for n in ("synth_A", "synth_B", "synth_C")]
        assert b["submitted"] >= a["finished"], "B was submitted while A held its cores"
        assert c["submitted"] >= max(a["finished"], b["finished"])
        peak = _peak(stage_jobs)
        assert peak <= 3, "the stages' jobs held %d cores at once, over their share of 3" % peak
        drivers = _named(root, "pl_synth")
        assert drivers and all(_record_cores(j) == 1 for j in drivers)
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def test_contract():
    root = _tmp()
    path, cfg = _project(root, max_cores=4)
    env = _env(root, sleep="8", SYNTH_PARALLEL="1", SYNTH_STAGE_NCPUS="2")
    wr = cfg["paths"]["work_root"]
    try:
        r = subprocess.run([PYTHON, SG, "--config", path, "--dry-run"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=60)
        assert r.returncode == 0 and ("cores: at most 4 at once (resources.max_cores, D-099): the "
                                      "sequencer's job 1, the stages' jobs 3; in flight per the state "
                                      "file: 0 in 0 job(s)") in r.stdout, r.stdout
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        _wait_for(cfg, "B: waits for cores")
        rc, out = _pl(root, env, "status")
        assert rc == 0 and "in flight per the state file: 2 in 1 job(s)" in out, out
        st = ST.State.load(wr)
        b = st.stage("B")
        assert b["status"] == "pending" and b["waiting_note"] == "job asks 2 cores; 1 free", b
        a = st.stage("A")
        assert a["jobs"][0]["ncpus"] == 2 and "after" not in a["jobs"][0], a["jobs"]
        log = _wait_for(cfg, "the chain ends", timeout=120)
        assert "PIPELINE DONE" in log and log.count("B: waits for cores") == 1, log
        st = ST.State.load(wr)
        assert "waiting_note" not in st.stage("B"), st.stage("B")
        assert [j["ncpus"] for n in ("A", "B", "C") for j in st.stage(n)["jobs"]] == [2, 2, 2]
        rc, out = _pl(root, env, "status")
        assert "in flight per the state file: 0 in 0 job(s)" in out, out
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def test_determinism():
    # (a) the count is a function of the state file alone
    root = _tmp()
    try:
        jobs_new = [_job("%d.x" % i, ncpus=48, cmd=_trial_line(48)) for i in range(5)]
        jobs_old = [_job("%d.x" % i, cmd=_trial_line(48)) for i in range(5)]
        st = _state(root, {"search": ("running", jobs_new + [_job("9.x", ncpus=8, after="0.x")])})
        st.save()
        back = ST.State.load(st.work_root)
        assert P.cores_in_flight(back) == P.cores_in_flight(st) == (240, 6, [])
        old = _state(root, {"search": ("running", jobs_old)})
        assert P.cores_in_flight(old)[0] == P.cores_in_flight(st)[0], "before D-099: from the qsub line"
    finally:
        shutil.rmtree(root)
    # (b) a driver killed while B waits: the successor keeps it waiting, then runs it once
    root = _tmp()
    path, cfg = _project(root, max_cores=4, walltime="00:01:00")
    env = _env(root, sleep="8", SYNTH_PARALLEL="1", SYNTH_STAGE_NCPUS="2")
    try:
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        _wait_for(cfg, "B: waits for cores")
        first = _named(root, "pl_synth")[0]
        subprocess.run(["qdel", first["id"]], env=env)
        log = _wait_for(cfg, "the chain ends", timeout=150)
        assert ("the previous driver %s ended without releasing the lock" % first["id"]) in log, log
        assert "PIPELINE DONE" in log, log
        assert log.count("B: waits for cores") == 1, "the note is in the state: logged once\n" + log
        _wait_quiet(root)
        for name in ("synth_A", "synth_B", "synth_C"):
            assert len(_named(root, name)) == 1, (name, _named(root, name))
        a, b = _named(root, "synth_A")[0], _named(root, "synth_B")[0]
        assert a["num"] < b["num"] and b["submitted"] >= a["finished"]
        assert _peak([j for j in _jobs(root) if j["name"].startswith("synth_")]) <= 3
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def _edge_guard():
    root = _tmp()
    path, cfg = _project(root, max_cores=4)
    env = _env(root, sleep="5", SYNTH_PARALLEL="1", SYNTH_STAGE_NCPUS="2", SYNTH_NO_CHECK="1")
    try:
        rc, out = _foreground(root, env)
        assert rc == P.EXIT_FAILED, (rc, out)
        assert "STOP: stage B refused: B: job asks 2 cores; 1 of 3 free under resources.max_cores" \
            in out, out
        assert len(_named(root, "synth_A")) == 1 and not _named(root, "synth_B"), \
            [(j["name"], j["state"]) for j in _jobs(root)]
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def _edge_never_fits_at_start():
    root = _tmp()
    path, cfg = _project(root, max_cores=4)
    env = _env(root, SYNTH_STAGE_NCPUS="5")
    try:
        rc, out = _foreground(root, env, timeout=60)
        assert rc == P.EXIT_FAILED, (rc, out)
        assert ("STOP: stage A refused: A: a job of 5 cores can never start: the stages' jobs share "
                "3 cores (resources.max_cores 4 less the sequencer's job, 1); raise "
                "resources.max_cores (--stop, then a submission with --accept-config-change) or ask "
                "fewer cores (D-099)") in out, out
        assert not _jobs(root), "nothing submitted"
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def _edge_never_fits_in_poll():
    root = _tmp()
    path, cfg = _project(root, max_cores=4)
    env = _env(root, sleep="5", SYNTH_POLL_ASK="5")
    try:
        rc, out = _foreground(root, env, timeout=60)
        assert rc == P.EXIT_FAILED, (rc, out)
        assert "STOP: stage A failed: A: a job of 5 cores can never start" in out, out
        assert "Traceback" not in out, out
        st = ST.State.load(cfg["paths"]["work_root"])
        assert st.stage("A")["status"] == "failed" and "can never start" in st.stage("A")["note"]
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def _edge_switch_to_cap():
    # the davinci state of 2026-10-19: 13 trials of 48 cores in flight, recorded
    # before D-099 (no ncpus), under Giulia's cap: no new job until <= 3 remain
    root = _tmp()
    try:
        cfg = C.load_config(GIULIA)
        trials = [_job("%d.pbsserver01" % (1851700 + i), cmd=_trial_line(48)) for i in range(13)]
        st = _state(root, {"search": ("running", trials)})
        sub = P.Submitter(_NoPBS(), st, "search", _Lines(), cfg)
        assert (sub.in_flight(), sub.free()) == (624, 196 - 624)
        assert not sub.fits(48) and not sub.fits(1)
        for j in st.stage("search")["jobs"][:10]:
            j["finished"] = True
        assert (sub.in_flight(), sub.free()) == (144, 52)
        assert sub.fits(48) and not sub.fits(53)
    finally:
        shutil.rmtree(root)


def _edge_config():
    with open(GIULIA, "rb") as fh:
        base = C._strip_docs(json.loads(fh.read().decode("ascii")))
    cases = (("v8", lambda x: x.update(config_version=8),
              "config_version: config_version must be 9, got 8"),
             ("missing", lambda x: x.pop("resources"),
              "resources.max_cores: missing (every key is required)"),
             ("51", lambda x: x["resources"].update(max_cores=51),
              "resources.max_cores: 51 cannot hold the sequencer's job (driver.ncpus 4) and the "
              "largest job (search.trial.ncpus 48)"),
             ("zero", lambda x: x["resources"].update(max_cores=0),
              "resources.max_cores: must be an integer >= 1, got 0"),
             ("string", lambda x: x["resources"].update(max_cores="200"),
              "resources.max_cores: must be an integer >= 1, got '200'"))
    for label, mut, want in cases:
        x = copy.deepcopy(base)
        mut(x)
        got = C.validate(x)
        assert want in got, (label, got)
    x = copy.deepcopy(base)
    x["resources"]["max_cores"] = 52           # 4 + 48: the boundary is accepted
    assert C.validate(x) == [], C.validate(x)
    x["bank"]["verify"]["ncpus"] = 60          # the largest job is now the verifier
    assert ("resources.max_cores: 52 cannot hold the sequencer's job (driver.ncpus 4) and the "
            "largest job (bank.verify.ncpus 60)") in C.validate(x), C.validate(x)


def test_edge_cases():
    failures = []
    for part in (_edge_guard, _edge_never_fits_at_start, _edge_never_fits_in_poll,
                 _edge_switch_to_cap, _edge_config):
        if os.environ.get("ONLY") and os.environ["ONLY"] not in part.__name__:
            continue
        try:
            part()
        except AssertionError as exc:
            failures.append("%s: %s" % (part.__name__, exc))
        except subprocess.TimeoutExpired as exc:
            failures.append("%s: timed out: %s" % (part.__name__, exc))
    assert not failures, "\n\n".join(failures)


# ---------------------------------------------------------------- runner ---

def main():
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print("python %s | fake PBS %s" % (sys.version.split()[0], _FAKE))
    if MUTATION:
        print("MUTATION=%s: the named piece is disabled; the check it names must FAIL" % MUTATION)
    results = []
    for name, func in checks:
        if os.environ.get("CHECK") and os.environ["CHECK"] != name:
            continue
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(r[0]) for r in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline[:200]))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()[:6000]))
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
