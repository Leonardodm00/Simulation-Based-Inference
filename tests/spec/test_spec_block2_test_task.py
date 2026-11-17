"""Block 2, contract amendment 4: the bank's test task (specs/SPEC.md block 2).

Oracles (written from the spec before the code was read; tester run
2026-11-17_1730 on 6c6c297, base a915950):
  O2.1 TEST_INDEX = min{ i : plan["tasks"][i]["n_iters"] > 0 } (the position in
       the plan's `tasks`, as the launcher's own WALLTIME=auto rule reads it).
  O2.2 the same TEST_INDEX goes to `test` and to every `array` call (round 0 and
       the rerun rounds, which also keep round 0's WALLTIME).
  O2.3 the skip rule reads tasks/task_<iiii>.json of the test task (this plan's
       sha256 and a `finished` time); another task's marker does not skip it.
  O2.4 the test log is logs/bank_test_<i>.log, the record names `bank test
       task <i>`; the environment-fault rule (amendment 2026-10-29 (b)) reads
       that log, so a fault line in it is seen and one in another task's log
       is not.
  O2.5 a plan that cannot be read, or holds no task with simulations, stops
       the stage before any launcher call; never an exception.
  O2.6 every task holding simulations: every launcher call, record and log
       line as at the base commit a915950 (run on the base tree in a
       subprocess, the same scenario).
Expected values come from the spec: the index rule is recomputed here by its
definition; the base commit's behaviour is the reference for O2.6.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

import _spec_util as U

import config as C                     # noqa: E402
import pbs as PBS                      # noqa: E402
import stages as S                     # noqa: E402
import stages.bank as B                # noqa: E402

BASE_PIPE = "/tmp/base_a915950/hpc/pipeline"   # git archive a915950 hpc | tar -x -C /tmp/base_a915950


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def spec_first_task(tasks):
    """The spec's rule, written here: the smallest index whose n_iters is an
    integer above 0; None when there is none."""
    for i, t in enumerate(tasks):
        if t["n_iters"] > 0:
            return i
    return None


class Sub(object):
    """The sequencer's Submitter, in process: job states set by the test, the
    launcher's jobs kept as (id, role, log), lines kept."""

    def __init__(self):
        self.lines, self.records, self.states = [], [], {}

    def status(self, job_id):
        st, ex = self.states.get(job_id, ("R", None))
        return PBS.JobStatus(job_id, st, exit_status=ex, walltime_s=1.0, exec_host="nodeQ")

    def log(self, text):
        self.lines.append(text)

    def record(self, job_id, role, cmd, log_path=None, **kw):
        self.records.append((job_id, role, log_path))

    def budget(self):
        return None

    def free(self):
        return None

    def check(self, n):
        return True

    def fits(self, n):
        return True


@pytest.fixture
def env():
    """(cfg, stage, plan path, log dir, calls list) on a Giulia copy whose
    paths are under a temporary root, with S1-S3's artefacts as dummy files
    and a launcher file that carries the NCPUS / GATE_NCPUS defaults."""
    root = U.tmpdir("spec_b2_")
    path, d = U.giulia_copy(root)
    cfg = C.load_config(path)
    for name in ("real_extraction", "sim_redetection", "label_freeze"):
        p = S.artefact_of(name, cfg)[1]
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write('{"dummy": "%s"}\n' % name)
    ew = os.path.join(d["paths"]["sbix_dir"], "export_windows")
    os.makedirs(ew, exist_ok=True)
    with open(os.path.join(ew, "launch_bank.sh"), "w") as fh:
        fh.write('NCPUS="${NCPUS:-1}"\nGATE_NCPUS="${GATE_NCPUS:-8}"\n')
    plan = S.artefact_of("bank_plan", cfg)[1]
    os.makedirs(os.path.dirname(plan), exist_ok=True)
    logdir = os.path.join(os.path.dirname(plan), "logs")
    os.makedirs(logdir, exist_ok=True)
    stage = B.SimBankStage()
    calls = []
    outputs = {"test": "  job   : 71.fake\n",
               "array": "# tasks       : 3  (8 simulations); largest: x\n  job   : 72[].fake\n"
                        "  gate  : 73.fake\n",
               "gate": "  gate  : 74.fake\n"}

    def launcher(c, mode, log_path, **extra):
        calls.append((mode, dict(extra)))
        return 0, outputs[mode]
    stage._run_launcher = launcher
    saved = os.environ.get("PIPELINE_LOG_GRACE_S")
    os.environ["PIPELINE_LOG_GRACE_S"] = "3600"
    try:
        yield cfg, stage, plan, logdir, calls, d
    finally:
        if saved is None:
            os.environ.pop("PIPELINE_LOG_GRACE_S", None)
        else:
            os.environ["PIPELINE_LOG_GRACE_S"] = saved
        shutil.rmtree(root, ignore_errors=True)


def write_plan(path, n_iters):
    tasks = [{"index": i, "name": "c/sweep_task%04d" % i, "n_iters": n,
              "shards": [{"shard": i, "start": 0, "stop": 1}]} for i, n in enumerate(n_iters)]
    with open(path, "w") as fh:
        json.dump({"tasks": tasks, "n_rows": 0}, fh, indent=1, sort_keys=True)
    return _sha(path)


def write_marker(bank, i, plan_sha, finished="2026-11-17T15:00:00+01:00"):
    p = os.path.join(bank, "tasks", "task_%04d.json" % i)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    m = {"plan_sha256": plan_sha}
    if finished is not None:
        m["finished"] = finished
    with open(p, "w") as fh:
        json.dump(m, fh)
    return p


def bank_of(d):
    return os.path.join(d["paths"]["banks_root"], d["dataset"] + "_sim")


# ------------------------------------------------------------------ O2.1, O2.5 ---

@pytest.mark.parametrize("n_iters", [[0, 5, 3], [3, 5, 2], [0, 0, 7, 0], [4], [-1, 4], [0, -2, 0, 1]])
def test_first_task_known_answers(env, n_iters):
    cfg, stage, plan, logdir, calls, d = env
    write_plan(plan, n_iters)
    want = spec_first_task([{"n_iters": n} for n in n_iters])
    assert B.first_task_with_simulations(plan) == (want, None)


@pytest.mark.parametrize("body, why", [
    ({"tasks": [{"n_iters": 0}, {"n_iters": 0}]}, "has no task with simulations"),
    ({"tasks": []}, "has no task with simulations"),
    ({"tasks": [{"n_iters": 5.0}]}, "cannot be read"),        # a float is not an integer
    ({"tasks": [{"n_iters": "5"}]}, "cannot be read"),
    ({"tasks": [{"n_iters": True}]}, "cannot be read"),       # JSON true is not a number
    ({"tasks": [{"n_iters": None}]}, "cannot be read"),
    ({"tasks": [{"name": "no n_iters"}]}, "cannot be read"),
    ({"tasks": ["not an object"]}, "cannot be read"),
    ({"tasks": {"0": {"n_iters": 3}}}, "cannot be read"),
    ({"plan": "no tasks"}, "cannot be read"),
    ([{"n_iters": 3}], "cannot be read"),                     # not a JSON object
    ("{not json", "cannot be read"),
    ("", "cannot be read"),
    (None, "cannot be read"),                                  # absent
    ("DIR", "cannot be read"),                                 # a directory in its place
])
def test_unreadable_or_empty_plans_stop_before_any_launcher_call(env, body, why):
    cfg, stage, plan, logdir, calls, d = env
    if body is None:
        pass
    elif body == "DIR":
        os.makedirs(plan)
    elif isinstance(body, str):
        with open(plan, "w") as fh:
            fh.write(body)
    else:
        with open(plan, "w") as fh:
            json.dump(body, fh)
    i, reason = B.first_task_with_simulations(plan)          # never raises
    assert i is None and reason.startswith("the bank plan %s %s" % (plan, why)), reason
    sub = Sub()
    st = {"jobs": [], "sub": {}}
    stage.start(cfg, st, sub)
    assert calls == [] and sub.records == [], (calls, sub.records)
    assert st["sub"]["phase"] == "failed" and st["sub"]["fail"] == reason, st["sub"]
    ok, text = stage.verify(cfg, st)
    assert not ok and text == reason


# ------------------------------------------------------------- O2.2, O2.3, O2.4 ---

def test_test_task_one_through_test_array_and_rerun_round(env):
    cfg, stage, plan, logdir, calls, d = env
    sha = write_plan(plan, [0, 5, 3])
    sub = Sub()
    st = {"jobs": [], "sub": {}}
    stage.start(cfg, st, sub)
    assert calls == [("test", {"TEST_INDEX": "1"})], calls
    assert sub.records == [("71.fake", "bank test task 1", os.path.join(logdir, "bank_test_1.log"))]
    # task 1's marker (this plan's, finished) passes the test; the array gets TEST_INDEX 1
    st["jobs"] = [{"id": "71.fake", "role": "bank test task 1", "finished": False}]
    sub.states["71.fake"] = ("F", 0)
    write_marker(bank_of(d), 1, sha)
    del calls[:]
    assert stage.poll(cfg, st, sub) == "running"
    assert len(calls) == 1 and calls[0][0] == "array" and calls[0][1]["TEST_INDEX"] == "1", calls
    # a rerun round (round 1 after a quarantine): round 0's walltime and the same index
    st["sub"].update(round=1, walltime_used="01:15:00")
    del calls[:]
    stage._start_array(cfg, st, sub)
    assert calls == [("array", {"WALLTIME": "01:15:00", "TEST_INDEX": "1",
                                "CONCURRENCY": str(d["bank"]["concurrency"])})], calls
    assert sub.records[-2][1] == "bank array (rerun round 1)", sub.records


def test_skip_rule_reads_the_test_tasks_marker(env):
    cfg, stage, plan, logdir, calls, d = env
    sha = write_plan(plan, [0, 0, 6])
    # task 0's marker of this plan (an empty task's): not the test task's -> the test runs
    write_marker(bank_of(d), 0, sha)
    sub = Sub()
    st = {"jobs": [], "sub": {}}
    stage.start(cfg, st, sub)
    assert calls == [("test", {"TEST_INDEX": "2"})], calls
    # task 2's marker of ANOTHER plan: the test runs
    write_marker(bank_of(d), 2, "0" * 64)
    del calls[:]
    stage.start(cfg, {"jobs": [], "sub": {}}, Sub())
    assert calls == [("test", {"TEST_INDEX": "2"})], calls
    # task 2's marker of this plan without `finished`: the test runs
    write_marker(bank_of(d), 2, sha, finished=None)
    del calls[:]
    stage.start(cfg, {"jobs": [], "sub": {}}, Sub())
    assert calls == [("test", {"TEST_INDEX": "2"})], calls
    # task 2's marker of this plan, finished: skipped, the array with TEST_INDEX 2
    write_marker(bank_of(d), 2, sha)
    del calls[:]
    sub = Sub()
    st = {"jobs": [], "sub": {}}
    stage.start(cfg, st, sub)
    assert [c[0] for c in calls] == ["array"] and calls[0][1]["TEST_INDEX"] == "2", calls
    assert st["sub"].get("test_skipped") is True


@pytest.mark.parametrize("exit_status, log_body, resubmitted", [
    (96, "RUN_ARGV ENV FAULT: host broken: the environment sbi_export cannot start (x)\n", True),
    (None, "RUN_ARGV ENV FAULT: host broken: the environment sbi_export cannot start (x)\n", True),
    (94, "RUN_ARGV AVOID: host nodeX is on the avoid list\nRUN_ARGV EXIT: 94\n", True),
    (-2, "", True),                     # PBS could not start it (JOB_EXEC_FAULTS)
    (1, "Traceback (most recent call last):\nValueError: x\n[job] bank_task_exit=1\n", False),
])
def test_test_tasks_faults_read_its_own_log(env, exit_status, log_body, resubmitted):
    """D-113 for the test task at TEST_INDEX 1: the fault is read from
    bank_test_1.log and the task is submitted again with TEST_INDEX 1."""
    cfg, stage, plan, logdir, calls, d = env
    write_plan(plan, [0, 5, 3])
    sub = Sub()
    st = {"jobs": [], "sub": {}}
    stage.start(cfg, st, sub)
    with open(os.path.join(logdir, "bank_test_1.log"), "w") as fh:
        fh.write(log_body)
    st["jobs"] = [{"id": "71.fake", "role": "bank test task 1", "finished": False}]
    sub.states["71.fake"] = ("F", exit_status)
    del calls[:]
    out = stage.poll(cfg, st, sub)
    if resubmitted:
        assert out == "running" and calls == [("test", {"TEST_INDEX": "1"})], (calls, sub.lines)
        assert st["sub"]["phase"] == "test"
    else:
        assert calls == [] and st["sub"]["phase"] == "failed", (calls, st["sub"])
        assert "bank_test_1.log" in st["sub"]["fail"], st["sub"]["fail"]


def test_a_fault_line_in_another_tasks_log_is_not_this_tests(env):
    """The contrast of the case above: no exit status yet, the fault line only
    in bank_test_0.log (another test's log): no resubmission (the stage waits
    for its own log within the grace period)."""
    cfg, stage, plan, logdir, calls, d = env
    write_plan(plan, [0, 5, 3])
    sub = Sub()
    st = {"jobs": [], "sub": {}}
    stage.start(cfg, st, sub)
    with open(os.path.join(logdir, "bank_test_0.log"), "w") as fh:
        fh.write("RUN_ARGV ENV FAULT: host broken: the environment cannot start (x)\n")
    st["jobs"] = [{"id": "71.fake", "role": "bank test task 1", "finished": False}]
    sub.states["71.fake"] = ("F", None)
    del calls[:]
    assert stage.poll(cfg, st, sub) == "running" and calls == [], (calls, sub.lines)


def test_dry_run_names_the_rule(env):
    cfg, stage, plan, logdir, calls, d = env
    lines = stage.dry_run(cfg, {})
    test = [l for l in lines if l.startswith("test (")]
    assert len(test) == 1 and "TEST_INDEX=<the plan's first task with simulations>" in test[0], lines
    assert calls == []


# ------------------------------------------------------------------------ O2.6 ---

_SCENARIO = textwrap.dedent(r'''
    import json, os, sys, hashlib
    pipe, cfg_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, pipe)
    sys.dont_write_bytecode = True
    os.environ["PIPELINE_LOG_GRACE_S"] = "3600"
    import config as C, pbs as PBS, stages as S, stages.bank as B
    cfg = C.load_config(cfg_path)
    plan = S.artefact_of("bank_plan", cfg)[1]
    bank = os.path.join(cfg.data["paths"]["banks_root"], cfg.data["dataset"] + "_sim")
    sha = hashlib.sha256(open(plan, "rb").read()).hexdigest()
    class Sub(object):
        def __init__(self): self.lines, self.records, self.states = [], [], {}
        def status(self, j):
            s, e = self.states.get(j, ("R", None))
            return PBS.JobStatus(j, s, exit_status=e, walltime_s=1.0, exec_host="nodeQ")
        def log(self, t): self.lines.append(t)
        def record(self, j, role, cmd, log_path=None, **kw): self.records.append([j, role, log_path, cmd, kw])
        def budget(self): return None
        def free(self): return None
        def check(self, n): return True
        def fits(self, n): return True
    calls = []
    outputs = {"test": "  job   : 71.fake\n",
               "array": "# tasks       : 3  (8 simulations); largest: x\n[launch] WALLTIME=auto: "
                        "test task 0 read 5 simulations in 9 s; ... -> 00:45:00\n  job   : 72[].fake\n"
                        "  gate  : 73.fake\n"}
    stage = B.SimBankStage()
    def launcher(c, mode, log_path, **extra):
        calls.append([mode, extra])
        return 0, outputs[mode]
    stage._run_launcher = launcher
    res = {}
    sub = Sub(); st = {"jobs": [], "sub": {}}
    stage.start(cfg, st, sub)
    st["jobs"] = [{"id": "71.fake", "role": "t", "finished": False}]
    sub.states["71.fake"] = ("F", 0)
    logdir = os.path.join(os.path.dirname(plan), "logs")
    open(os.path.join(logdir, "bank_test_0.log"), "w").write("TASK 0 OK\n[job] bank_task_exit=0\n")
    os.makedirs(os.path.join(bank, "tasks"), exist_ok=True)
    json.dump({"plan_sha256": sha, "finished": "2026-11-17T15:00:00+01:00"},
              open(os.path.join(bank, "tasks", "task_0000.json"), "w"))
    res["poll"] = stage.poll(cfg, st, sub)
    res["calls"], res["records"], res["lines"] = calls, sub.records, sub.lines
    res["sub"] = dict((k, v) for k, v in st["sub"].items()
                      if k not in ("array_submitted", "test_index"))
    sub2 = Sub(); st2 = {"jobs": [], "sub": {}}; calls2 = []
    stage._run_launcher = lambda c, m, l, **e: (calls2.append([m, e]) or (0, outputs[m]))
    stage.start(cfg, st2, sub2)
    res["skip_calls"], res["skip_lines"], res["skip_records"] = calls2, sub2.lines, sub2.records
    res["dry"] = [l for l in stage.dry_run(cfg, {}) if not l.startswith("test (")]
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
''')


def test_every_task_with_simulations_is_the_base_commits_run(env, tmp_path):
    """O2.6: on a plan [3, 5, 2] every launcher call, record and log line of
    the start, the test's pass, the array, the skip and the dry run (but its
    test line, an accepted change) equals the base commit's."""
    if not os.path.isdir(BASE_PIPE):
        pytest.skip("no base tree (git archive a915950 hpc | tar -x -C /tmp/base_a915950)")
    cfg, stage, plan, logdir, calls, d = env
    write_plan(plan, [3, 5, 2])
    script = tmp_path / "scenario.py"
    script.write_text(_SCENARIO)
    outs = {}
    for name, pipe in (("head", U.PIPE), ("base", BASE_PIPE)):
        bank = bank_of(d)
        if os.path.isdir(bank):
            shutil.rmtree(bank)
        # each run starts from the same files: no profile, no test log
        prof = os.path.join(d["paths"]["work_root"], "bank_profile_%s.json" % d["dataset"])
        for f in (prof, os.path.join(logdir, "bank_test_0.log")):
            if os.path.exists(f):
                os.remove(f)
        o = str(tmp_path / ("%s.json" % name))
        r = subprocess.run([sys.executable, str(script), pipe, cfg.path, o],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=120,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        assert r.returncode == 0, r.stdout
        outs[name] = json.load(open(o))
    h, b = outs["head"], outs["base"]
    for k in ("poll", "calls", "records", "lines", "sub", "skip_calls", "skip_lines",
              "skip_records", "dry"):
        assert h[k] == b[k], (k, h[k], b[k])
    assert h["calls"][0] == ["test", {"TEST_INDEX": "0"}]
