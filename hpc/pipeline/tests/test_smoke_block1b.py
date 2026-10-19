#!/usr/bin/env python3
"""Smoke test for block 1's amendment (c) of 2026-10-18 (D-098): the
sequencer as a chain of PBS jobs on compute nodes, never a process on a
login node.

Checks (specs/SPEC.md block 1, amended 2026-10-18 (c))
    test_known_answer   the committed Giulia configuration: v8, the driver
                        block, the job name pl_giulia, the 85,500 s budget,
                        the dry run's sequencer line
    test_reference      job ids (sequence part, number), the budget formula
                        and the driver's qsub line against hand-written
                        values; the three lock formats
    test_convergence    not a discretisation: skipped
    test_invariants     the state's own temporary name and its .prev
                        fallback; the lock (O_EXCL, a live job refused, a
                        finished one replaced, a pid on another host by
                        its state writes); no nohup in the pipeline's code
    test_contract       the chain under the fake PBS: hand-over by budget,
                        one driver at a time, every stage job once, the
                        last successor deleted; a second submission refused
    test_determinism    a driver killed mid-run: its successor takes the
                        stale lock and finishes, no job submitted twice; a
                        job of the same name and output file already
                        queued is adopted, one of another output file is not
    test_edge_cases     drive without PBS_JOBID; stop from any node, then
                        submit again; --retry and a configuration change at
                        submit; a crash loop ends after 3 drivers; two
                        drivers at once (the older stays); a signal keeps
                        the chain; retire (the transition) against an
                        old-style sequencer

Run
    python3 hpc/pipeline/tests/test_smoke_block1b.py
        Prints a PASS / FAIL / ERROR / TODO / SKIP table and the message of
        every failure. Exit code 1 if any check failed, errored or is TODO.
    MUTATION=<name> python3 hpc/pipeline/tests/test_smoke_block1b.py
        Re-runs the checks with one piece of the code disabled; the check
        named in _PATCHES must then FAIL: successor, unclean_limit,
        lock_check, prev_fallback, heartbeat, tiebreak, retry_at_submit,
        adopt, adopt_output, finished_filter.

Everything runs under the fake PBS (tests/fake_pbs/, first on PATH) in a
temporary folder, on the synthetic graph A -> B -> C of synthetic_graph.py;
the sequencer's jobs run that file's `drive`. Nothing touches davinci.
The fake PBS hands its environment to its jobs (davinci passes only -v):
that is how the stages' switches and a mutation reach the driver jobs.

Pure ASCII, LF only; stdlib only.
"""

import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
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

import config as C      # noqa: E402
import pbs as PBS       # noqa: E402
import pipeline as P    # noqa: E402
import stages as S      # noqa: E402
import state as ST      # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
PYTHON = sys.executable
SG = os.path.join(_HERE, "synthetic_graph.py")

# A mutation disables one piece of the code; the check named after it must FAIL.
MUTATION = os.environ.get("MUTATION", "")
_PATCHES = {
    # test_contract: no successor is ever submitted, so the chain cannot hand over
    "successor": ("_orig = P.submit_driver\n"
                  "def _no_succ(cfg, depend=None, **k):\n"
                  "    if depend:\n"
                  "        raise PBS.SubmitError('mutation: no successor')\n"
                  "    return _orig(cfg, depend=depend, **k)\n"
                  "P.submit_driver = _no_succ\n"),
    # test_edge_cases (crash loop): the chain would loop for ever
    "unclean_limit": "P.DRIVER_MAX_UNCLEAN = 10 ** 6\n",
    # test_invariants: a live driver's lock would be taken
    "lock_check": "P._lock_is_stale = lambda *a, **k: True\n",
    # test_invariants: a state file that does not parse would not fall back
    "prev_fallback": ("import json as _j, os as _o\n"
                      "def _load(cls, wr, log=None):\n"
                      "    p = cls.path_of(wr)\n"
                      "    if not _o.path.isfile(p):\n"
                      "        return None\n"
                      "    return cls(wr, _j.loads(open(p, 'rb').read().decode('ascii')))\n"
                      "ST.State.load = classmethod(_load)\n"),
    # test_invariants: a pid lock on another host taken while its state is written
    "heartbeat": "P._state_heartbeat = lambda *a, **k: False\n",
    # test_edge_cases (two drivers at once): both would exit
    "tiebreak": ("P._older_running = lambda others, me: "
                 "[j for j, s in others if s in PBS.RUNNING_STATES]\n"),
    # test_edge_cases (--retry at submit): the live job would not be seen
    "retry_at_submit": "P._unfinished_jobs = lambda st, pbs_module: []\n",
    # test_determinism (adopt): the queued job of the same name would be ignored
    "adopt": "P._qsub_name = lambda argv: None\n",
    # test_determinism (adopt): a job of another output file would be adopted
    "adopt_output": "P._same_output = lambda a, b: True\n",
    # test_edge_cases (stop, then submit): finished drivers listed by qselect
    # would count as queued or running and refuse the submission
    "finished_filter": ("def _d(cfg, pbs_module=PBS):\n"
                        "    return [(j, pbs_module.status(j).state)\n"
                        "            for j in pbs_module.select(P.driver_name(cfg))]\n"
                        "P._drivers = _d\n"),
}
if MUTATION:
    if MUTATION not in _PATCHES:
        raise SystemExit("unknown MUTATION %r; known: %s" % (MUTATION, ", ".join(sorted(_PATCHES))))
    exec(_PATCHES[MUTATION], {"P": P, "PBS": PBS, "ST": ST, "S": S, "C": C})


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_b1b_")


def _project(root, walltime="00:00:20", poll=1):
    """A synthetic configuration under `root` whose sbi_dir is this checkout
    (the driver jobs run its jobs/run_argv.pbs) and whose driver block is
    small: walltime `walltime`, a tick every `poll` s."""
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
    path = os.path.join(root, "synth.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    return path, cfg


def _env(root, sleep="0.3", **extra):
    env = dict(os.environ)
    env["PATH"] = _FAKE + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = os.path.join(root, "spool")
    env["FAKE_PBS_DELAY"] = "0.5"
    env["FAKE_PBS_ENFORCE_WALLTIME"] = "1"
    env["FAKE_PBS_QSELECT_SUFFIX"] = ".cluster"   # qselect spells the ids otherwise
    env["PIPELINE_SKIP_CONDA"] = "1"
    env["PIPELINE_PYTHON"] = PYTHON
    env["PYTHONPATH"] = _HERE + os.pathsep + env.get("PYTHONPATH", "")
    env["SYNTH_JOB_SLEEP"] = sleep
    for k in ("SYNTH_REFUSE", "SYNTH_SILENT_KILL", "SYNTH_CRASH_POLL", "MUTATION_PATCH",
              "PBS_JOBID"):
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


def _plog(cfg):
    return os.path.join(cfg["paths"]["work_root"], "pipeline.log")


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _wait_for(cfg, text, timeout=90.0, count=1):
    """Wait until pipeline.log holds `text` at least `count` times; return the log."""
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
    """Wait until no fake job is queued, held or running."""
    end = time.time() + timeout
    while time.time() < end:
        if all(j["state"] == "F" for j in _jobs(root)):
            return
        time.sleep(0.25)
    raise AssertionError("jobs still live after %.0f s: %r"
                         % (timeout, [(j["id"], j["name"], j["state"]) for j in _jobs(root)
                                      if j["state"] != "F"]))


def _sweep(root, env):
    """Delete every live fake job (cleanup after a failure, or a mutation's loop)."""
    for _ in range(4):
        live = [j["id"] for j in _jobs(root) if j["state"] != "F"]
        if not live:
            return
        subprocess.run(["qdel"] + live, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        time.sleep(1.0)


def _qsub(env, argv):
    proc = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split()[0]


def _no_overlap(drivers):
    """No two sequencer jobs ran at the same time (started / finished times)."""
    ran = sorted([(j["started"], j["finished"] or time.time(), j["id"]) for j in drivers
                  if j["started"]])
    for (s0, f0, a), (s1, f1, b) in zip(ran, ran[1:]):
        assert s1 >= f0 - 1e-6, "drivers %s and %s overlap: %r" % (a, b, ran)


class _Stub(object):
    """A pbs_module for acquire_lock: status() of a dict of finished flags."""

    def __init__(self, finished):
        self.finished = finished

    def status(self, job_id):
        return PBS.JobStatus(job_id, "F" if self.finished.get(job_id) else "R",
                             exit_status=0 if self.finished.get(job_id) else None)


class _Lines(object):
    def __init__(self):
        self.lines = []

    def __call__(self, text):
        self.lines.append(text)

    def text(self):
        return "\n".join(self.lines)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = C.load_config(GIULIA)
    assert cfg.data["config_version"] == 9
    assert cfg.data["driver"] == {"queue": "cpu", "ncpus": 4, "mem": "16gb",
                                  "walltime": "24:00:00", "poll_seconds": 60}, cfg.data["driver"]
    assert P.driver_name(cfg) == "pl_giulia"
    # 24 h less min(max(900, 5 x 60), 24 h / 4) = 86400 - 900, by hand
    assert P._driver_budget_s(cfg) == 85500.0
    r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", GIULIA,
                        "--dry-run"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=120)
    want = ("the sequencer: PBS jobs named pl_giulia on queue cpu (4 cores, 16gb, walltime "
            "24:00:00, a tick every 60 s), one at a time, each submitting its successor "
            "(afterany); nothing runs on a login node (D-098)")
    assert want in r.stdout, r.stdout[-3000:]


def test_reference():
    # the Output_Path of the recorded davinci record (wrapped), without its host
    import test_smoke_pipeline as B1
    js = PBS.parse_qstat("1829506.davinci", B1._QSTAT_SAMPLE)
    assert js.output_path == ("/davinci-1/home/ldellamea/SBI/hpc/joint/stage3/jobs/"
                              "joint_arms.o1829506"), js.output_path
    assert PBS.parse_qstat("7.x", "Job Id: 7.x\n    job_state = R\n").output_path is None
    assert P._same_output("/a/b/../c.log", "/a/c.log") and not P._same_output(None, "/a")
    # job ids: the sequence part before the first dot; hand-read
    assert PBS.job_seq("1829506.davinci-mgt01") == "1829506"
    assert PBS.job_seq("1829506[].davinci-mgt01") == "1829506[]"
    assert PBS.same_job("1829506.davinci-mgt01", "1829506.davinci-mgt01.hpc.local")
    assert not PBS.same_job("1829506.davinci-mgt01", "1829507.davinci-mgt01")
    assert not PBS.same_job("", "1829506.x") and not PBS.same_job(None, "1.fake")
    assert PBS.job_number("1829506.davinci-mgt01") == 1829506
    assert PBS.job_number("12[].fake") == 12 and PBS.job_number("x.y") == -1
    # the budget: walltime - min(max(900, 5 x poll), walltime / 4), by hand
    root = _tmp()
    try:
        for wall, poll, want in (("24:00:00", 60, 85500.0), ("00:00:40", 1, 30.0),
                                 ("01:00:00", 300, 2700.0), ("00:00:20", 1, 15.0)):
            path, _ = _project(root, walltime=wall, poll=poll)
            got = P._driver_budget_s(C.load_config(path))
            assert got == want, (wall, poll, got, want)
        # the driver's qsub line, captured
        path, raw = _project(root)
        cfg = C.load_config(path)
        seen = []

        class Cap(object):
            @staticmethod
            def submit(argv, cwd=None, env=None):
                seen.append(list(argv))
                return "9.fake"

        old = os.environ.get("PIPELINE_SKIP_CONDA")
        os.environ["PIPELINE_SKIP_CONDA"] = "1"
        try:
            job, argv_file, log_path = P.submit_driver(cfg, depend="8.fake", pbs_module=Cap,
                                                       entry=["py", "/x/pipeline.py"])
            job2, argv2, _ = P.submit_driver(cfg, extra=["--accept-config-change"],
                                             pbs_module=Cap, entry=["py", "/x/pipeline.py"])
        finally:
            if old is None:
                os.environ.pop("PIPELINE_SKIP_CONDA", None)
            else:
                os.environ["PIPELINE_SKIP_CONDA"] = old
        wr = raw["paths"]["work_root"]
        want = ["qsub", "-N", "pl_synth", "-q", "test", "-l", "select=1:ncpus=1:mem=1gb",
                "-l", "walltime=00:00:20", "-j", "oe", "-o",
                os.path.join(wr, "driver", "driver.1.log"), "-v",
                "ARGV_FILE=%s,ENV_NAME=%s,WORKDIR=%s,SKIP_CONDA=1"
                % (os.path.join(wr, "driver", "driver.1.argv"), raw["envs"]["extract"], wr),
                "-W", "depend=afterany:8.fake",
                os.path.join(_REPO, "hpc", "pipeline", "jobs", "run_argv.pbs")]
        assert seen[0] == want, "\n%r\n%r" % (seen[0], want)
        assert job == "9.fake" and argv_file == os.path.join(wr, "driver", "driver.1.argv")
        assert log_path == os.path.join(wr, "driver", "driver.1.log")
        with open(argv_file) as fh:
            assert fh.read().split("\n")[:-1] == ["py", "/x/pipeline.py", "drive", "--config",
                                                  os.path.abspath(path)]
        assert "-W" not in seen[1] and argv2.endswith("driver.2.argv")
        with open(argv2) as fh:
            assert fh.read().split("\n")[-2] == "--accept-config-change"
        # the three lock formats
        wr2 = os.path.join(root, "locks")
        os.makedirs(wr2)
        for text, want in (("job:12.fake dvnode006 2026-10-18T17:00:00Z\n", ("job:12.fake",
                                                                             "dvnode006")),
                           ("pid:4242 vm 2026-10-18T17:00:00Z\n", ("pid:4242", "vm")),
                           ("98765 dvlogin04 2026-10-18T12:55:01Z\n", ("98765", "dvlogin04"))):
            with open(os.path.join(wr2, "pipeline.lock"), "w") as fh:
                fh.write(text)
            assert P.read_lock(wr2) == want, (text, P.read_lock(wr2))
        os.remove(os.path.join(wr2, "pipeline.lock"))
        assert P.read_lock(wr2) is None
    finally:
        shutil.rmtree(root)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the sequencer has no convergence order")


def test_invariants():
    root = _tmp()
    try:
        # ---- the state: its own temporary name, .prev, the fallback -------
        wr = os.path.join(root, "wr")
        os.makedirs(os.path.join(wr, "state.json.tmp"))      # the transition's blocker
        st = ST.State.new(wr, "synth", "/x.json", "a" * 64)
        st.save()
        first = json.load(open(ST.State.path_of(wr)))
        st.event("second")
        st.save()
        prev = json.load(open(ST.State.path_of(wr) + ".prev"))
        assert prev == first, "the .prev file holds the state before the last save"
        assert not glob.glob(os.path.join(wr, "state.json.*.tmp")), "no temporary file left"
        assert os.path.isdir(os.path.join(wr, "state.json.tmp")), "the blocker is untouched"
        second = json.load(open(ST.State.path_of(wr)))
        with open(ST.State.path_of(wr), "wb") as fh:
            fh.write(b"{\"schema\": 1, \"stag")                # a node that wrote half a file
        lines = _Lines()
        got = ST.State.load(wr, lines)
        assert got is not None and got.data == prev, "the previous state is used"
        assert "does not parse" in lines.text() and ".prev" in lines.text(), lines.text()
        with open(ST.State.path_of(wr) + ".prev", "wb") as fh:
            fh.write(b"garbage")
        try:
            ST.State.load(wr)
            raise AssertionError("both files bad: must raise")
        except ValueError as exc:
            assert "no state file parses" in str(exc), exc
        assert second["events"][-1][1] == "second"
        os.remove(ST.State.path_of(wr))                      # removed: a fresh start
        with open(ST.State.path_of(wr) + ".prev", "w") as fh:
            json.dump(prev, fh)
        assert ST.State.load(wr) is None, "a missing state.json is no state (.prev not read)"

        # ---- the lock ------------------------------------------------------
        lk = os.path.join(root, "lk")
        os.makedirs(lk)
        p = os.path.join(lk, "pipeline.lock")
        me = P._holder_token()
        assert P._create_lock(p, me) and not P._create_lock(p, "job:x"), "O_EXCL"
        os.remove(p)
        # (i) a live driver job holds it: refused
        with open(p, "w") as fh:
            fh.write("job:12.fake dvnode006 2026-10-18T17:00:00Z\n")
        lines = _Lines()
        assert not P.acquire_lock(lk, lines, pbs_module=_Stub({"12.fake": False})), \
            "a lock of a driver job that runs must be refused; log:\n" + lines.text()
        assert "another sequencer (driver job 12.fake, on dvnode006)" in lines.text(), lines.text()
        assert P.read_lock(lk)[0] == "job:12.fake", "the holder's lock is untouched"
        P.release_lock(lk)
        assert P.read_lock(lk)[0] == "job:12.fake", "release_lock never removes another's"
        # (ii) the job has finished: replaced, then this sequencer's
        lines = _Lines()
        assert P.acquire_lock(lk, lines, pbs_module=_Stub({"12.fake": True}))
        assert "stale lock of driver job 12.fake (finished) replaced" in lines.text()
        assert P.read_lock(lk)[0] == me
        P.release_lock(lk)
        assert P.read_lock(lk) is None
        # (iii) a pid on another host whose state is being written: refused
        with open(p, "w") as fh:
            fh.write("4242 dvlogin04 2026-10-18T12:55:01Z\n")
        stop = threading.Event()

        def writer():
            while not stop.is_set():
                with open(ST.State.path_of(lk), "w") as fh:
                    fh.write("{}\n")
                os.utime(ST.State.path_of(lk), None)
                time.sleep(0.3)

        th = threading.Thread(target=writer)
        th.start()
        try:
            time.sleep(0.4)
            lines = _Lines()
            assert not P.acquire_lock(lk, lines, quiet_s=4.0), \
                "a pid lock on another host whose state is being written must be refused; " \
                "log:\n" + lines.text()
            assert "another sequencer (pid 4242 on dvlogin04) is still writing the state" \
                in lines.text(), lines.text()
        finally:
            stop.set()
            th.join()
        # (iv) silent for quiet_s: replaced
        lines = _Lines()
        t0 = time.time()
        assert P.acquire_lock(lk, lines, quiet_s=2.0)
        assert time.time() - t0 >= 2.0
        assert "stale lock from pid 4242 on dvlogin04 replaced: no state write for 2 s" \
            in lines.text(), lines.text()
        P.release_lock(lk)
        # (v) a pid of this host that is not alive (nor 0, nor -1)
        assert not P._pid_alive(0) and not P._pid_alive(-1)
        with open(p, "w") as fh:
            fh.write("pid:999999 %s 2026-10-18T17:00:00Z\n" % socket.gethostname())
        lines = _Lines()
        assert P.acquire_lock(lk, lines) and "stale lock from pid 999999" in lines.text()
        P.release_lock(lk)

        # ---- D-098 in the code: nothing left running on a login node --------
        bad = []
        for dirpath, dirs, files in os.walk(_PIPE):
            dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
            for f in files:
                if f.endswith((".py", ".sh", ".pbs")):
                    with open(os.path.join(dirpath, f)) as fh:
                        if "nohup" in fh.read():
                            bad.append(os.path.join(dirpath, f))
        assert not bad, "nohup in %r" % bad
        sh = _read(os.path.join(_PIPE, "run_pipeline.sh"))
        assert 'exec python3 "$HERE/pipeline.py" submit --config' in sh
    finally:
        shutil.rmtree(root)


def test_contract():
    root = _tmp()
    path, cfg = _project(root)
    env = _env(root, sleep="6")
    try:
        # the dry run names the sequencer's jobs and submits nothing
        r = subprocess.run([PYTHON, SG, "--config", path, "--dry-run"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=60)
        assert r.returncode == 0 and ("the sequencer: PBS jobs named pl_synth on queue test (1 "
                                      "cores, 1gb, walltime 00:00:20, a tick every 1 s)") \
            in r.stdout, r.stdout
        assert not _jobs(root), "a dry run submits nothing"
        rc, out = _pl(root, env, "submit")
        assert rc == 0 and "SUBMITTED the sequencer: job 1.fake (pl_synth)" in out, out
        _wait_for(cfg, "A: running")
        rc2, out2 = _pl(root, env, "submit")
        assert rc2 == 4 and "REFUSE: the sequencer is already queued or running" in out2, out2
        rc3, out3 = _pl(root, env, "status")
        assert rc3 == 0 and "sequencer jobs (pl_synth): 1.fake.cluster R" in out3, out3
        log = _wait_for(cfg, "the chain ends", timeout=120)
        assert "PIPELINE DONE: every stage done or skipped" in log, log
        assert log.count("time budget reached; the successor") >= 1, log
        assert "the chain ends (done, exit 0); its successor" in log, log
        _wait_quiet(root)
        for name in ("synth_A", "synth_B", "synth_C"):
            assert len(_named(root, name)) == 1, (name, _named(root, name))
        drivers = _named(root, "pl_synth")
        assert len(drivers) >= 3, drivers
        _no_overlap(drivers)
        last = drivers[-1]
        assert last["started"] is None and last["comment"] == "Job deleted by user", last
        assert P.read_lock(cfg["paths"]["work_root"]) is None, "no lock left"
        st = ST.State.load(cfg["paths"]["work_root"])
        assert st.statuses() == {"A": "done", "B": "done", "C": "done"}, st.statuses()
        assert st.data["job"] in [d["id"] for d in drivers], st.data["job"]
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def test_determinism():
    # (a) a driver killed mid-run (the fake's qdel: SIGKILL of its process group)
    root = _tmp()
    path, cfg = _project(root, walltime="00:01:00")
    env = _env(root, sleep="5")
    wr = cfg["paths"]["work_root"]
    try:
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        _wait_for(cfg, "A: running")
        first = _named(root, "pl_synth")[0]
        assert first["state"] == "R", first
        subprocess.run(["qdel", first["id"]], env=env)
        log = _wait_for(cfg, "the chain ends", timeout=120)
        assert ("the previous driver %s ended without releasing the lock" % first["id"]) in log, log
        assert ("stale lock of driver job %s (finished) replaced" % first["id"]) in log, log
        assert "PIPELINE DONE" in log, log
        _wait_quiet(root)
        for name in ("synth_A", "synth_B", "synth_C"):
            assert len(_named(root, name)) == 1, (name, _named(root, name))
        assert P._unclean(wr)["count"] == 0, P._unclean(wr)
    finally:
        _sweep(root, env)
        shutil.rmtree(root)
    # (b) a job of the same name already queued or running is adopted
    root = _tmp()
    path, cfg = _project(root)
    env = _env(root, sleep="4")
    try:
        import synthetic_graph as SGM
        old = dict(os.environ)
        os.environ.update(env)
        try:
            argv = SGM.SyntheticStage("A", (), "ok")._qsub(C.load_config(path))
        finally:
            os.environ.clear()
            os.environ.update(old)
        pre = _qsub(env, argv)
        # a job of the same name writing ANOTHER output file (another project's):
        # never adopted
        i = argv.index("-o")
        other = list(argv)
        other[i + 1] = os.path.join(root, "another_project", "A.log")
        stranger = _qsub(env, other)
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        log = _wait_for(cfg, "the chain ends", timeout=120)
        assert ("A: adopted job -> %s.cluster (a job of that name and output file was queued or "
                "running but not recorded)" % pre) in log, log
        assert ("adopted job -> %s.cluster" % stranger) not in log, log
        assert "PIPELINE DONE" in log, log
        _wait_quiet(root)
        assert len(_named(root, "synth_A")) == 2, "ours (adopted) and the stranger's, no third"
        for name in ("synth_B", "synth_C"):
            assert len(_named(root, name)) == 1, (name, _named(root, name))
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def _edge_drive_without_jobid():
    root = _tmp()
    try:
        path, cfg = _project(root)
        env = _env(root)
        rc, out = _pl(root, env, "drive")
        assert rc == 2 and "REFUSE: `drive` is the sequencer's PBS job (PBS_JOBID is unset)" in out, out
    finally:
        shutil.rmtree(root)


def _edge_stop_then_submit():
    root = _tmp()
    path, cfg = _project(root, walltime="00:01:00")
    # qselect lists the finished jobs too here: they must not count as live
    env = _env(root, sleep="4", FAKE_PBS_QSELECT_HISTORY="1")
    try:
        rc, out = _pl(root, env, "stop")
        assert rc == 0 and "stop: no sequencer is running" in out, out
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        _wait_for(cfg, "A: running")
        rc, out = _pl(root, env, "stop", "--wait", "60")
        assert rc == 0 and "stop: requested" in out and "stop: the sequencer has stopped" in out, out
        log = _wait_for(cfg, "the chain ends (stopped, exit 0)")
        assert "stopped on request (STOP_REQUEST: asked" in log, log
        _wait_quiet(root)
        rc, out = _pl(root, env, "status")
        assert "sequencer jobs (pl_synth): none queued or running" in out, out
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        _wait_for(cfg, "PIPELINE DONE", timeout=120)
        _wait_for(cfg, "the chain ends (done, exit 0)")
        _wait_quiet(root)
        for name in ("synth_A", "synth_B", "synth_C"):
            assert len(_named(root, name)) == 1, (name, _named(root, name))
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def _edge_retry_and_config_change():
    root = _tmp()
    path, cfg = _project(root, walltime="00:01:00")
    env = _env(root, sleep="5")
    try:
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        _wait_for(cfg, "A: running")
        rc, out = _pl(root, env, "submit", "--retry", "A")
        assert rc == 2 and "REFUSE: --retry A: the stage has 1 job(s) still queued or running" \
            in out, out
        rc, out = _pl(root, env, "submit", "--retry", "Z")
        assert rc == 2 and "REFUSE: --retry names no stage: Z" in out, out
        _wait_for(cfg, "the chain ends (done", timeout=120)
        _wait_quiet(root)
        # a configuration change: refused at submit, accepted with the flag,
        # and the flag reaches the first driver only
        d = json.loads(json.dumps(cfg))
        d["search"]["gp_rounds"] = 21
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
        rc, out = _pl(root, env, "submit")
        assert rc == 2 and "REFUSE: the configuration changed since this run started" in out, out
        n0 = len(_named(root, "pl_synth"))
        rc, out = _pl(root, env, "submit", "--accept-config-change")
        assert rc == 0, out
        _wait_for(cfg, "the chain ends", count=2, timeout=60)
        _wait_quiet(root)
        drv = os.path.join(cfg["paths"]["work_root"], "driver")
        firsts = sorted(glob.glob(os.path.join(drv, "driver.*.argv")),
                        key=lambda q: int(q.split(".")[-2]))
        assert "--accept-config-change" in _read(firsts[n0]), _read(firsts[n0])
        assert "--accept-config-change" not in _read(firsts[n0 + 1]), _read(firsts[n0 + 1])
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def _edge_crash_loop():
    root = _tmp()
    path, cfg = _project(root, walltime="00:01:00")
    env = _env(root, sleep="30", SYNTH_CRASH_POLL="A")
    wr = cfg["paths"]["work_root"]
    try:
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        log = _wait_for(cfg, "the chain ends", timeout=60)
        time.sleep(3.0)
        log = _read(_plog(cfg))
        n = log.count("the sequencer raised RuntimeError")
        assert n == 3, "%d drivers raised (want 3):\n%s" % (n, log)
        assert "unclean end 3 in a row; the chain ends; its successor" in log, log
        assert P._unclean(wr)["count"] == 3
        drivers = _named(root, "pl_synth")
        assert len(drivers) == 4 and drivers[-1]["comment"] == "Job deleted by user", \
            [(j["id"], j["state"], j["comment"]) for j in drivers]
        assert len(_named(root, "synth_A")) == 1
        # a submission by the user resets the count
        _sweep(root, env)
        _wait_quiet(root)
        env2 = _env(root, sleep="0.3")
        rc, out = _pl(root, env2, "submit")
        assert rc == 0 and "ended uncleanly in a row (3) was reset by this submission" in out, out
        _wait_for(cfg, "PIPELINE DONE", timeout=90)
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def _edge_two_drivers():
    root = _tmp()
    path, cfg = _project(root)
    env = _env(root, sleep="2")
    try:
        gate = os.path.join(root, "gate.sh")
        with open(gate, "w") as fh:
            fh.write("#!/bin/bash\nsleep 2\n")
        g = _qsub(env, ["qsub", "-N", "gate", "-l", "walltime=00:01:00", gate])
        old = dict(os.environ)
        os.environ.update(env)
        try:
            c = C.load_config(path)
            ids = [P.submit_driver(c, depend=g, entry=[PYTHON, SG])[0] for _ in range(2)]
        finally:
            os.environ.clear()
            os.environ.update(old)
        log = _wait_for(cfg, "PIPELINE DONE", timeout=90)
        _wait_for(cfg, "the chain ends (done")
        _wait_quiet(root)
        log = _read(_plog(cfg))
        started = log.count("sequencer started: job")
        assert started == 1, "%d sequencers ticked (want 1):\n%s" % (started, log)
        assert "sequencer started: job %s on" % ids[0] in log, "the older one ticks:\n" + log
        assert (("driver %s on %s: another driver %s.cluster is running; this one exits"
                 % (ids[1], socket.gethostname(), ids[0])) in log
                or ("driver %s: deleted the waiting driver %s.cluster" % (ids[0], ids[1])) in log), log
        for name in ("synth_A", "synth_B", "synth_C"):
            assert len(_named(root, name)) == 1, (name, _named(root, name))
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def _edge_signal():
    root = _tmp()
    path, cfg = _project(root, walltime="00:01:00")
    env = _env(root, sleep="5")
    wr = cfg["paths"]["work_root"]
    try:
        rc, out = _pl(root, env, "submit")
        assert rc == 0, out
        _wait_for(cfg, "A: running")
        st = ST.State.load(wr)
        assert st.data["job"] == "1.fake", st.data["job"]
        os.kill(int(st.data["pid"]), signal.SIGTERM)     # PBS's first word to a job it ends
        log = _wait_for(cfg, "the chain ends", timeout=120)
        assert "driver 1.fake: ended by signal 15 (PBS ending the job: its walltime, a qdel); " \
            "unclean end 1 in a row; the successor 2.fake continues" in log, log
        assert "PIPELINE DONE" in log, log
        _wait_quiet(root)
        for name in ("synth_A", "synth_B", "synth_C"):
            assert len(_named(root, name)) == 1, (name, _named(root, name))
        assert P._unclean(wr)["count"] == 0
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


_OLD_STYLE = '''import json, os, sys, time
# the sequencer of the code before 2026-10-18 as far as the transition sees it:
# a bare-pid lock, the state written through state.json.tmp after every tick,
# the lock removed in its finally (f1a2b7a's release_lock: unconditional)
wr, host = sys.argv[1], sys.argv[2]
lock, p = os.path.join(wr, "pipeline.lock"), os.path.join(wr, "state.json")
with open(lock, "w") as fh:
    fh.write("%d %s 2026-10-18T12:55:01Z\\n" % (os.getpid(), host))
try:
    n = 0
    while True:
        n += 1
        with open(p, "rb") as fh:
            data = json.loads(fh.read().decode("ascii"))
        data["events"].append(["2026-10-18T13:00:00Z", "old tick %d" % n])
        tmp = p + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(json.dumps(data, sort_keys=True).encode("ascii"))
        os.replace(tmp, p)
        time.sleep(0.5)
finally:
    os.remove(lock)
'''


def _edge_retire():
    root = _tmp()
    path, cfg = _project(root)
    env = _env(root)
    wr = cfg["paths"]["work_root"]
    os.makedirs(wr)
    try:
        rc, out = _pl(root, env, "retire")
        assert rc == 0 and "RETIRE: no lock in" in out, out
        ST.State.new(wr, "synth", path, C.load_config(path).sha256).save()
        # (a) an old-style sequencer alive on another host
        old = os.path.join(root, "old_style_sequencer.py")
        with open(old, "w") as fh:
            fh.write(_OLD_STYLE)
        proc = subprocess.Popen([PYTHON, old, wr, "dvlogin04"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        for _ in range(100):
            if "old tick 3" in _read(ST.State.path_of(wr)):
                break
            time.sleep(0.1)
        rc, out = _pl(root, env, "retire", "--wait", "20")
        assert rc == 0 and "RETIRE: OLD SEQUENCER STOPPED: its lock is gone" in out, out
        o, _ = proc.communicate(timeout=10)
        assert proc.returncode != 0 and "IsADirectoryError" in o, (proc.returncode, o)
        assert P.read_lock(wr) is None
        assert os.path.isdir(ST.State.path_of(wr) + ".tmp"), "the blocker stays"
        last = ST.State.load(wr)
        assert last.data["events"][-1][1].startswith("old tick"), "the old state is intact"
        # (b) a lock with no writer behind it, and a temporary FILE a crash left
        os.rmdir(ST.State.path_of(wr) + ".tmp")
        with open(os.path.join(wr, "pipeline.lock"), "w") as fh:
            fh.write("4242 dvlogin04 2026-10-18T12:55:01Z\n")
        with open(ST.State.path_of(wr) + ".tmp", "w") as fh:
            fh.write("{}")
        rc, out = _pl(root, env, "retire", "--wait", "3", timeout=60)
        assert rc == 0 and "temporary state file left by an earlier crash was moved aside" in out \
            and "RETIRE: the lock is still there after 3 s" in out, out
        assert glob.glob(ST.State.path_of(wr) + ".tmp.stale-*"), "kept, not deleted"
        # (c) the first driver takes that stale lock after the quiet time, and finishes
        env3 = _env(root, PIPELINE_HEARTBEAT_QUIET_S="3")
        rc, out = _pl(root, env3, "submit")
        assert rc == 0 and "a sequencer started on a login node" in out, out
        log = _wait_for(cfg, "PIPELINE DONE", timeout=90)
        assert "stale lock from pid 4242 on dvlogin04 replaced: no state write for 3 s" in log, log
        _wait_for(cfg, "the chain ends (done")
        # (d) a job's lock: nothing to retire
        with open(os.path.join(wr, "pipeline.lock"), "w") as fh:
            fh.write("job:7.fake dvnode006 2026-10-18T17:00:00Z\n")
        rc, out = _pl(root, env, "retire")
        assert rc == 0 and "nothing to retire; --stop stops it" in out, out
    finally:
        _sweep(root, env)
        shutil.rmtree(root)


def test_edge_cases():
    failures = []
    for part in (_edge_drive_without_jobid, _edge_stop_then_submit, _edge_retry_and_config_change,
                 _edge_crash_loop, _edge_two_drivers, _edge_signal, _edge_retire):
        if os.environ.get("ONLY") and os.environ["ONLY"] not in part.__name__:
            continue
        try:
            part()
        except AssertionError as exc:
            failures.append("%s: %s" % (part.__name__, exc))
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
