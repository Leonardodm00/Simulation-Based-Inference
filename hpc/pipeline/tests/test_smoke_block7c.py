#!/usr/bin/env python3
"""Smoke test for D-111 (specs/SPEC.md block 7, "Environment faults and
reruns", and block 1, amended 2026-10-29 (g)): a node without conda, or one
on which the environment cannot start at all, never ends the sequencer's
chain and never uses up a trial's or a member's resubmission; every trial
and member runs with --resume.

Found on davinci 2026-10-29: PBS ran the sequencer's job and two trial jobs
again on nodes where the conda installation was not mounted; all three
exited 98 before any command, then the successor too, and the chain ended
with no line in the log.

Checks (SPEC block 7, "Environment faults and reruns", Test oracles)
    test_known_answer   activate_units: jobs/activate_env.sh by hand -- a fake
                        conda that works, is absent, fails its activation;
                        the recorded prefix, $HOME/.conda/envs,
                        CONDA_ENVS_PATH; nothing usable (96); SKIP_CONDA=1 --;
                        envs.run_in_env through the fallback and with
                        nothing usable (EnvError); common.env_fault's codes;
                        the qsub line's variables and the chain template;
                        the guard's streak
    test_reference      job_units: jobs/run_argv.pbs and chain.sh by hand
                        under the fake PBS -- a stage job's environment
                        fault (held, 96, then the watchdog revives a missing
                        chain); a driver job's (a replacement with -a and its
                        backoff; nothing when another job of the chain lives
                        or a stop was asked); a job of a submitter before
                        D-111 (exit 98, as before); a refusal's deliberate end
    test_convergence    not a discretisation: skipped
    test_invariants     chain_end: the synthetic graph, drivers as jobs: a
                        stage refusal ends the chain on purpose (CHAIN_END,
                        --status prints it), a submission removes it and the
                        pipeline finishes
    test_contract       env_faults: end to end on the fixture, the jobs under
                        the fake conda: a trial through the fallback (host
                        noconda), a trial's and a member's first jobs on host
                        broken (96): submitted again, not counted;
                        PIPELINE DONE; every trial argv with --resume
    test_determinism    chain_replace: the synthetic graph, drivers as jobs
                        under the fake conda: the second driver lands on host
                        broken, exits 96, and its replacement (-a) finishes
                        the pipeline; no two drivers at once
    test_edge_cases     watchdog: the second driver fails to start (PBS -2)
                        while stage A's job runs; that job's end revives the
                        chain; the same with a stop asked before that end:
                        nothing revived, and a submission finishes the run
    MUTATION=<name>     (environment) re-runs the checks against one disabled
                        piece, which must then FAIL: fallback (the folder
                        route of activate_env.sh: activate_units,
                        job_units, env_faults), env_free (an environment
                        fault taken as a kill: env_faults), always_resume
                        (--resume only on a resubmission: env_faults),
                        streak (the guard never stops: activate_units),
                        replace (the driver's replacement: job_units,
                        chain_replace), watchdog (job_units, watchdog),
                        chain_end (the deliberate end ignored by the
                        watchdog: watchdog's stop run)

Run
    python3 hpc/pipeline/tests/test_smoke_block7c.py
        About 3 minutes at one thread. test_contract (env_faults) needs torch,
        sbi, zuko, scikit-optimize (the training environment) and is skipped
        without them; the rest is bash and stdlib, so it runs in any
        environment, on a login node too. Nothing touches the cluster's
        scheduler: the fake PBS of tests/fake_pbs/ is first on PATH.
    python3 hpc/pipeline/tests/test_smoke_block7c.py activate_units job_units
        Some checks only, by name (the parts: activate_units, job_units,
        chain_end, env_faults, chain_replace, watchdog).

Pure ASCII, LF only.
"""

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
_FAKE_CONDA = os.path.join(_HERE, "fake_conda")
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C                      # noqa: E402
import envs as E                        # noqa: E402
import pipeline as P                    # noqa: E402
from stages import common as CM         # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
PYTHON = sys.executable
SG = os.path.join(_HERE, "synthetic_graph.py")
ENVS = ("sbi_export", "sbi_env")          # the configuration's envs.extract and envs.train

# ------------------------------------------------------------- mutations ---

MUTATION = os.environ.get("MUTATION", "")
# applied to the copy of hpc/pipeline/jobs a check runs from: (file, [(old, new)])
_BASH = {
    "fallback": ("activate_env.sh", [(
        "    # 2. the environment's own folder\n",
        "    # 2. the environment's own folder\n"
        "    echo \"RUN_ARGV ENV FAULT: host $(_pa_host): MUTATION fallback\"\n"
        "    [ \"$had_u\" = \"1\" ] && set -u\n"
        "    return 96\n")]),
    "replace": ("chain.sh", [(
        "pipeline_chain_replace() {   # <why>: a driver job that could not start its command\n",
        "pipeline_chain_replace() {   # <why>: a driver job that could not start its command\n"
        "    echo 'RUN_ARGV CHAIN: MUTATION replace'; return 0\n")]),
    "watchdog": ("chain.sh", [(
        "pipeline_watchdog() {    # a stage job's end: revive a chain that is gone without "
        "ending on purpose\n",
        "pipeline_watchdog() {    # a stage job's end: revive a chain that is gone without "
        "ending on purpose\n    return 0\n")]),
    "chain_end": ("chain.sh", [
        ("    [ -f \"$d/CHAIN_END\" ] && return 0\n", ""),
        (" && [ ! -f \"$d/CHAIN_END\" ]", "")]),
}
# exec'd in the sequencer's process (the end-to-end check's bootstrap)
_PY = {
    "env_free": "CM.env_fault = lambda *a, **k: None\n",
    "always_resume": ("_o = SS.SearchStage._trial_argv\n"
                      "def _t(self, cfg, st, tid, resume):\n"
                      "    a = _o(self, cfg, st, tid, resume)\n"
                      "    return a if resume else [x for x in a if x != '--resume']\n"
                      "SS.SearchStage._trial_argv = _t\n"),
}
if MUTATION and MUTATION not in _BASH and MUTATION not in _PY and MUTATION != "streak":
    raise SystemExit("unknown MUTATION %r; known: %s"
                     % (MUTATION, ", ".join(sorted(list(_BASH) + list(_PY) + ["streak"]))))


def _jobs_copy(root):
    """A copy of hpc/pipeline/jobs under `root` (the mutation applied to it),
    as `<root>/sbi_copy/hpc/pipeline/jobs`; returns the copy's sbi_dir."""
    sbi = os.path.join(root, "sbi_copy")
    dst = os.path.join(sbi, "hpc", "pipeline", "jobs")
    shutil.copytree(os.path.join(_PIPE, "jobs"), dst)
    _mutate(dst)
    return sbi


def _mutate(jobs_dir):
    if MUTATION not in _BASH:
        return
    name, pairs = _BASH[MUTATION]
    p = os.path.join(jobs_dir, name)
    text = open(p).read()
    for old, new in pairs:
        assert text.count(old) == 1, ("mutation %s: anchor not found once" % MUTATION, old)
        text = text.replace(old, new)
    with open(p, "w") as fh:
        fh.write(text)


def _seq_boot(root):
    """The sequencer's entry: pipeline.py, or with a Python MUTATION a
    bootstrap that applies it first."""
    if MUTATION not in _PY:
        return os.path.join(_PIPE, "pipeline.py")
    boot = os.path.join(root, "pipeline_mutated.py")
    with open(boot, "w") as fh:
        fh.write("import sys\nsys.path.insert(0, %r)\nimport pipeline as P\n"
                 "import stages.common as CM\nimport stages.search as SS\n%s\n"
                 "sys.stderr.write('MUTATION %s applied\\n')\nsys.exit(P.main(sys.argv[1:]))\n"
                 % (_PIPE, _PY[MUTATION], MUTATION))
    return boot


# --------------------------------------------------------------- fixtures ---

def _host_bin(root, host):
    """A folder whose `hostname` prints `host`."""
    d = os.path.join(root, "hostbin_" + host)
    if not os.path.isdir(d):
        os.makedirs(d)
        with open(os.path.join(d, "hostname"), "w") as fh:
            fh.write("#!/bin/sh\necho '%s'\n" % host)
        os.chmod(os.path.join(d, "hostname"), 0o755)
    return d


def _fake_env(folder, name):
    """A conda-like environment folder: conda-meta/, bin/python (this
    interpreter, unless `hostname` is in FAKE_ENV_BROKEN_HOSTS: then it
    exits 1, an environment that cannot start on that node) and an
    activate.d script that exports FAKE_ENV_ACTIVATED=<name>."""
    p = os.path.join(folder, name)
    for d in ("conda-meta", "bin", os.path.join("etc", "conda", "activate.d")):
        os.makedirs(os.path.join(p, d), exist_ok=True)
    # bin/python only: the fake PBS's own commands run `python3` from PATH, which a
    # job's inherited PATH (the fake hands its environment on) must not turn into this
    for exe in ("python",):
        with open(os.path.join(p, "bin", exe), "w") as fh:
            fh.write("#!/bin/bash\nh=$(hostname 2>/dev/null)\n"
                     "case \",${FAKE_ENV_BROKEN_HOSTS:-},\" in *\",$h,\"*) "
                     "echo \"python: cannot start on $h\" >&2; exit 1 ;; esac\n"
                     "exec %s \"$@\"\n" % PYTHON)
        os.chmod(os.path.join(p, "bin", exe), 0o755)
    with open(os.path.join(p, "etc", "conda", "activate.d", "zz_marker.sh"), "w") as fh:
        fh.write("export FAKE_ENV_ACTIVATED=%s\n" % name)
    return p


def _envs_root(root):
    d = os.path.join(root, "conda_envs")
    for n in ENVS + ("myenv",):
        _fake_env(d, n)
    return d


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _spool_jobs(spool):
    out = []
    for p in glob.glob(os.path.join(spool, "*.json")):
        b = os.path.basename(p)[:-5]
        if b.isdigit():
            try:
                with open(p) as fh:
                    out.append(json.load(fh))
            except (OSError, ValueError):
                pass
    return sorted(out, key=lambda j: j["num"])


def _wait(cond, timeout, what):
    end = time.time() + timeout
    while time.time() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.25)
    raise AssertionError("after %.0f s: %s" % (timeout, what))


# ------------------------------------------------------- activate_units ---

def _activate(root, jobs, host, env_extra, body="pipeline_activate myenv; rc=$?"):
    """bash: source activate_env.sh with `set -u` on, run `body`, print the
    outcome. Returns (exit, stdout+stderr)."""
    env = {"HOME": os.path.join(root, "home"), "PATH": os.pathsep.join(
        [_host_bin(root, host), _FAKE_CONDA, "/usr/bin", "/bin"]),
        "FAKE_CONDA_ENVS": os.path.join(root, "conda_envs"),
        "WORK_ROOT": os.path.join(root, "wr")}
    env.update(env_extra)
    script = ("set -u\n. '%s/activate_env.sh'\n%s\n"
              "echo \"rc=$rc flags=$- python=$(command -v python) act=${FAKE_ENV_ACTIVATED:-none} "
              "prefix=${CONDA_PREFIX:-none}\"\n" % (jobs, body))
    p = subprocess.run(["bash", "-c", script], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
    return p.returncode, p.stdout


def _test_activate_units():
    root = tempfile.mkdtemp(prefix="b7c_act_")
    try:
        os.makedirs(os.path.join(root, "home", ".conda", "envs"))
        os.makedirs(os.path.join(root, "wr"))
        envs_root = _envs_root(root)
        jobs = os.path.join(_jobs_copy(root), "hpc", "pipeline", "jobs")
        mine = os.path.join(envs_root, "myenv")
        cache = os.path.join(root, "wr", "envs", "myenv.prefix")
        # (a) conda works: activated by conda, the prefix recorded, nothing printed
        rc, out = _activate(root, jobs, "goodnode", {})
        assert rc == 0 and out.strip() == ("rc=0 flags=huBc python=%s/bin/python act=none "
                                           "prefix=%s" % (mine, mine)), out
        assert _read(cache) == mine + "\n", _read(cache)
        # (b) conda absent on the node: the recorded prefix, its activate.d sourced, -u restored
        rc, out = _activate(root, jobs, "noconda", {"FAKE_CONDA_ABSENT_HOSTS": "noconda"})
        assert ("RUN_ARGV ENV: host noconda: conda unavailable (bash: conda: command not found); "
                "myenv activated from its folder %s\n" % mine) in out, (out, "mutation fallback")
        assert ("rc=0 flags=huBc python=%s/bin/python act=myenv prefix=%s" % (mine, mine)) in out
        # (c) conda there, its activation failing: the same fallback, its reason
        rc, out = _activate(root, jobs, "goodnode", {"FAKE_CONDA_ACTIVATE_FAILS": "1"})
        assert ("conda unavailable (EnvironmentNameNotFound: Could not find conda environment: "
                "myenv); myenv activated from its folder %s" % mine) in out and "rc=0" in out, out
        # (d) no record: $HOME/.conda/envs/<env>
        os.remove(cache)
        home_env = os.path.join(root, "home", ".conda", "envs", "myenv")
        os.symlink(mine, home_env)
        rc, out = _activate(root, jobs, "noconda", {"FAKE_CONDA_ABSENT_HOSTS": "noconda"})
        assert "activated from its folder %s\n" % home_env in out and \
            "prefix=%s" % home_env in out, out
        # (e) only CONDA_ENVS_PATH (an absent folder first)
        os.remove(home_env)
        other = os.path.join(root, "other")
        os.makedirs(other)
        os.symlink(mine, os.path.join(other, "myenv"))
        rc, out = _activate(root, jobs, "noconda", {"FAKE_CONDA_ABSENT_HOSTS": "noconda",
                                                    "CONDA_ENVS_PATH": "/nonexistent:" + other})
        assert "activated from its folder %s/myenv\n" % other in out and "rc=0" in out, out
        # (f) nothing usable: 96, every folder named with why
        os.makedirs(home_env)                      # a folder that is no environment
        rc, out = _activate(root, jobs, "broken", {"FAKE_CONDA_ABSENT_HOSTS": "broken",
                                                   "FAKE_ENV_BROKEN_HOSTS": "broken",
                                                   "CONDA_ENVS_PATH": other})
        want = ("RUN_ARGV ENV FAULT: host broken: the environment myenv cannot start (conda: bash: "
                "conda: command not found; folders: %s (no conda-meta), %s/myenv (its python does "
                "not start on this node))" % (home_env, other))
        assert want in out and "rc=96 flags=huBc" in out, out
        # (g) SKIP_CONDA=1: nothing
        rc, out = _activate(root, jobs, "broken", {"SKIP_CONDA": "1",
                                                   "FAKE_CONDA_ABSENT_HOSTS": "broken"})
        assert out.strip().startswith("rc=0 flags=huBc") and "RUN_ARGV" not in out and \
            "prefix=none" in out, out
        # run_in_env: through the fallback, then with nothing usable (EnvError)
        os.rmdir(home_env)
        saved = dict(os.environ)
        try:
            os.environ.update({"PIPELINE_SKIP_CONDA": "0", "FAKE_CONDA_ABSENT_HOSTS": "noconda",
                               "FAKE_CONDA_ENVS": envs_root, "WORK_ROOT": os.path.join(root, "wr"),
                               "HOME": os.path.join(root, "home"),
                               "PATH": os.pathsep.join([_host_bin(root, "noconda"), _FAKE_CONDA,
                                                        os.environ.get("PATH", "")])})
            os.environ.pop("CONDA_ENVS_PATH", None)
            with open(cache, "w") as fh:
                fh.write(mine + "\n")
            act = E.ACTIVATE
            if MUTATION in _BASH:
                E.ACTIVATE = os.path.join(jobs, "activate_env.sh")
            try:
                rc, out = E.run_in_env("myenv", ["python", "-c", "print('IN ENV')"])
                assert rc == 0 and "IN ENV" in out and \
                    "RUN_ARGV ENV: host noconda: conda unavailable" in out, (rc, out)
                os.environ["FAKE_ENV_BROKEN_HOSTS"] = "noconda"
                try:
                    E.run_in_env("myenv", ["python", "-c", "print('IN ENV')"])
                    raise AssertionError("run_in_env did not raise EnvError")
                except E.EnvError as exc:
                    assert "the environment myenv cannot start on this node (exit 96)" in str(exc), exc
            finally:
                E.ACTIVATE = act
        finally:
            os.environ.clear()
            os.environ.update(saved)
        # env_fault's codes
        ok = {96: "exit 96: the environment could not start (no log)",
              97: "exit 97: the conda hook failed (a job script before D-111)",
              98: "exit 98: the conda activation failed (a job script before D-111)",
              -2: "PBS could not start or keep the job on its node (Exit_status -2)",
              -11: "PBS could not start or keep the job on its node (Exit_status -11)"}
        for code, why in ok.items():
            assert CM.env_fault({"exit_status": code}) == why, (code, CM.env_fault({"exit_status": code}))
        line = "RUN_ARGV ENV FAULT: host nX: the environment e cannot start (conda: c; folders: f)"
        assert CM.env_fault({"exit_status": 96}, "x\n" + line + "\n") == line[len("RUN_ARGV ENV FAULT: "):]
        assert CM.env_fault({"exit_status": None}, line) == line[len("RUN_ARGV ENV FAULT: "):]
        assert CM.env_fault({"exit_status": "96"}) == ok[96]
        for code in (0, 1, 2, 3, 94, 135, 271, -29, None):
            assert CM.env_fault({"exit_status": code}, "Traceback\nboom\n") is None, code
        assert CM.host_of_log(line) == "nX" and \
            CM.host_of_log("RUN_ARGV ENV: host nY: conda unavailable (x); e activated") == "nY"
        # the qsub line's variables, the chain template
        path, cfg = _synthetic_project(root, os.path.join(root, "proj"))
        c = C.load_config(path)
        wr = c.data["paths"]["work_root"]
        line = CM.qsub_line(c, "t_x", "q", 1, "1gb", "00:01:00", "/l.log", "/a.argv", "/w")
        sk = "1" if E.skip_conda() else "0"
        assert line[line.index("-v") + 1] == (
            "ARGV_FILE=/a.argv,ENV_NAME=sbi_env,WORKDIR=/w,SKIP_CONDA=%s,JOBS_DIR=%s,WORK_ROOT=%s,"
            "CHAIN_NAME=pl_synth,ROLE=stage" % (sk, os.path.join(cfg["paths"]["sbi_dir"], "hpc",
                                                                 "pipeline", "jobs"), wr)), line
        line = CM.qsub_line(c, "t_x", "q", 1, "1gb", "00:01:00", "/l.log", "/a.argv", "/w",
                            role="driver")
        assert line[line.index("-v") + 1].endswith(",ROLE=driver"), line
        try:
            CM.qsub_line(c, "t_x", "q", 1, "1gb", "00:01:00", "/l.log", "/a.argv", "/w", role="x")
            raise AssertionError("role x accepted")
        except ValueError:
            pass
        P.write_chain_template(c, entry=["py", "/x/pipeline.py"])
        tpl = _read(os.path.join(wr, "driver", "chain.qsub")).split("\n")[:-1]
        d = c.data["driver"]
        assert tpl == CM.qsub_line(c, "pl_synth", d["queue"], d["ncpus"], d["mem"], d["walltime"],
                                   "@LOG@", "@ARGV@", wr, env_name="sbi_export", role="driver"), tpl
        assert "-W" not in tpl and "-o" in tpl and tpl[tpl.index("-o") + 1] == "@LOG@"
        assert _read(os.path.join(wr, "driver", "revive.argv")).split("\n")[:-1] == \
            ["py", "/x/pipeline.py", "drive", "--config", os.path.abspath(path)]
        # the guard: 12 environment faults in a row on 3 hosts stop the stage; on 2 they do not
        if MUTATION == "streak":
            CM.ENV_FAULT_STREAK = 10 ** 6
        apath = os.path.join(root, "avoid.txt")
        sub, lines = {}, []
        for k in range(11):
            n, stop = CM.note_env_fault(sub, "h%d" % (k % 3 + 1), "why%d" % k, "search",
                                        lines.append, apath)
            assert stop is None, (k, stop)
        n, stop = CM.note_env_fault(sub, "h3", "the last", "search", lines.append, apath)
        assert stop is not None and stop.startswith(
            "12 job(s) in a row could not start (environment faults on 3 node(s): h1, h2, h3; "
            "the last: the last): the environment or the cluster, not a node -- fix it, delete "
            "from %s the hosts this run listed (h1, h2, h3), then --retry search" % apath), \
            (stop, "mutation streak: the guard never stops")
        assert sub["node_fault_hosts"] == {"h1": 4, "h2": 4, "h3": 4}, sub
        CM.clear_env_fault_streak(sub)
        assert sub["env_fault_streak"] == {"n": 0, "hosts": []}
        for k in range(12):
            n, stop = CM.note_env_fault(sub, "h%d" % (k % 2 + 1), "w", "best_fit", lines.append,
                                        apath)
            assert stop is None, (k, stop)
        assert sub["env_fault_streak"]["n"] == 12
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------- job_units ---

def _job(root, jobs, host, env_extra, argv=("echo", "HELLO"), timeout=120):
    """jobs/run_argv.pbs by hand (bash), the fake PBS and the fake conda on
    PATH, `hostname` printing `host`. Returns (exit, output, seconds)."""
    wr = os.path.join(root, "wr")
    argv_file = os.path.join(root, "cmd.argv")
    with open(argv_file, "w") as fh:
        fh.write("".join(a + "\n" for a in argv))
    env = {"HOME": os.path.join(root, "home"),
           "PATH": os.pathsep.join([_host_bin(root, host), _FAKE, _FAKE_CONDA, "/usr/bin",
                                    "/bin"]),
           "FAKE_PBS_SPOOL": os.path.join(root, "spool"), "FAKE_PBS_DELAY": "0.1",
           "FAKE_CONDA_ENVS": os.path.join(root, "conda_envs"), "USER": "tester",
           "ARGV_FILE": argv_file, "ENV_NAME": "myenv", "WORKDIR": root, "SKIP_CONDA": "0",
           "JOBS_DIR": jobs, "WORK_ROOT": wr, "CHAIN_NAME": "pl_t", "ROLE": "stage",
           "AVOID_HOLD_S": "1", "PBS_JOBID": "900.fake", "PIPELINE_REPLACE_BASE_S": "1",
           "PIPELINE_CHAIN_QSUB_WAIT_S": "0", "PYTHONPATH": _PIPE}
    env.update(env_extra)
    t0 = time.time()
    p = subprocess.run(["bash", os.path.join(jobs, "run_argv.pbs")], env=env, cwd=root,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=timeout)
    return p.returncode, p.stdout, time.time() - t0


def _template(wr, root):
    """A chain template whose job is a dummy that prints its argv file."""
    dummy = os.path.join(root, "dummy_driver.sh")
    with open(dummy, "w") as fh:
        fh.write("#!/bin/bash\necho \"DUMMY DRIVER $ARGV_FILE\"\n")
    d = os.path.join(wr, "driver")
    os.makedirs(d, exist_ok=True)
    lines = ["qsub", "-N", "pl_t", "-q", "test", "-l", "select=1:ncpus=1:mem=1gb", "-l",
             "walltime=00:01:00", "-j", "oe", "-o", "@LOG@", "-v", "ARGV_FILE=@ARGV@", dummy]
    with open(os.path.join(d, "chain.qsub"), "w") as fh:
        fh.write("".join(a + "\n" for a in lines))
    with open(os.path.join(d, "revive.argv"), "w") as fh:
        fh.write("py\n/x/pipeline.py\ndrive\n")


def _test_job_units():
    root = tempfile.mkdtemp(prefix="b7c_job_")
    try:
        os.makedirs(os.path.join(root, "home", ".conda", "envs"))
        wr = os.path.join(root, "wr")
        os.makedirs(wr)
        envs_root = _envs_root(root)
        jobs = os.path.join(_jobs_copy(root), "hpc", "pipeline", "jobs")
        spool = os.path.join(root, "spool")
        bad = {"FAKE_CONDA_ABSENT_HOSTS": "broken,noconda", "FAKE_ENV_BROKEN_HOSTS": "broken"}
        # a good node: conda, the command, its exit
        rc, out, _ = _job(root, jobs, "good", {})
        assert rc == 0 and "HELLO\n" in out and out.endswith("RUN_ARGV EXIT: 0\n"), out
        assert _read(os.path.join(wr, "envs", "myenv.prefix")) == \
            os.path.join(envs_root, "myenv") + "\n"
        # no conda on the node: the folder, the command
        rc, out, _ = _job(root, jobs, "noconda", bad)
        assert rc == 0 and "RUN_ARGV ENV: host noconda: conda unavailable" in out and \
            "HELLO\n" in out, (out, "mutation fallback")
        # a stage job on a node where nothing starts: held, 96, the command not run; no
        # template yet, so the watchdog does nothing
        rc, out, dt = _job(root, jobs, "broken", bad)
        assert rc == 96 and "RUN_ARGV ENV FAULT: host broken: the environment myenv cannot start" \
            in out and "RUN_ARGV: holding this slot for 1 s (D-111), then exiting 96\n" in out and \
            out.endswith("RUN_ARGV EXIT: 96\n") and "HELLO" not in out and dt >= 1.0, out
        assert not _spool_jobs(spool), "nothing submitted without a template"
        # with the template and no job of the chain: the watchdog revives it (the revived job
        # then waits 30 s in the fake queue)
        _template(wr, root)
        rc, out, _ = _job(root, jobs, "broken", dict(bad, FAKE_PBS_DELAY="30"))
        assert rc == 96 and "RUN_ARGV CHAIN: no sequencer job was queued, held or running and the " \
            "chain had not ended on purpose; submitted 1.fake to revive it" in out, \
            (out, "mutation watchdog")
        jl = _spool_jobs(spool)
        assert [j["name"] for j in jl] == ["pl_t"] and jl[0]["vars"]["ARGV_FILE"] == \
            os.path.join(wr, "driver", "revive.argv") and jl[0]["exec_time"] is None, jl
        assert "watchdog: job 900.fake on broken: no sequencer job was queued, held or running " \
            "and the chain had not ended on purpose; submitted 1.fake to revive it (D-111)" in \
            _read(os.path.join(wr, "pipeline.log"))
        # while the revived job waits, another stage job's end submits nothing more
        rc, out, _ = _job(root, jobs, "good", {})
        assert rc == 0 and "RUN_ARGV CHAIN" not in out and len(_spool_jobs(spool)) == 1, \
            (out, _spool_jobs(spool))
        subprocess.run(["qdel", "1.fake"], env={"PATH": _FAKE + os.pathsep + "/usr/bin:/bin",
                                               "FAKE_PBS_SPOOL": spool})
        _wait(lambda: all(j["state"] == "F" for j in _spool_jobs(spool)), 30, "the dummy ended")
        # a stop asked: no revival
        with open(os.path.join(wr, "STOP_REQUEST"), "w") as fh:
            fh.write("asked\n")
        rc, out, _ = _job(root, jobs, "good", {})
        assert "RUN_ARGV CHAIN" not in out and len(_spool_jobs(spool)) == 1, out
        os.remove(os.path.join(wr, "STOP_REQUEST"))
        # the chain ended on purpose: no revival
        with open(os.path.join(wr, "driver", "CHAIN_END"), "w") as fh:
            fh.write("2026-10-29T00:00:00Z 1.fake done (exit 0)\n")
        rc, out, _ = _job(root, jobs, "good", {})
        assert "RUN_ARGV CHAIN" not in out and len(_spool_jobs(spool)) == 1, \
            (out, "mutation chain_end")
        os.remove(os.path.join(wr, "driver", "CHAIN_END"))
        # a driver job on a node where nothing starts: a replacement with -a, its own argv file
        rc, out, dt = _job(root, jobs, "broken", dict(bad, ROLE="driver"))
        jl = _spool_jobs(spool)
        assert rc == 96 and dt < 5.0 and "RUN_ARGV CHAIN: submitted the replacement sequencer job " \
            "2.fake, to start in 1 s (replacement 1 since the last full tick)" in out, \
            (out, "mutation replace")
        assert len(jl) == 2 and jl[1]["name"] == "pl_t" and jl[1]["exec_time"] is not None and \
            jl[1]["vars"]["ARGV_FILE"] == os.path.join(root, "cmd.argv") and \
            abs(jl[1]["exec_time"] - jl[1]["submitted"] - 1.0) < 1.5, jl[1]
        assert _read(os.path.join(wr, "driver", "env_replacements")) == "1\n"
        assert ("driver 900.fake: could not start on broken (the environment myenv cannot start); "
                "its replacement 2.fake starts in 1 s (replacement 1 since the last full tick; "
                "D-111)") in _read(os.path.join(wr, "pipeline.log"))
        _wait(lambda: all(j["state"] == "F" for j in _spool_jobs(spool)), 30, "replacement ran")
        # the backoff: the second replacement waits twice as long
        rc, out, _ = _job(root, jobs, "broken", dict(bad, ROLE="driver"))
        assert "to start in 2 s (replacement 2 since the last full tick)" in out, out
        _wait(lambda: all(j["state"] == "F" for j in _spool_jobs(spool)), 30, "replacement 2 ran")
        # another job of the chain lives (a held successor): nothing submitted
        held = subprocess.run(["qsub", "-h", "-N", "pl_t", os.path.join(root, "dummy_driver.sh")],
                              env={"PATH": _FAKE + os.pathsep + "/usr/bin:/bin",
                                   "FAKE_PBS_SPOOL": spool}, stdout=subprocess.PIPE,
                              universal_newlines=True).stdout.strip()
        n0 = len(_spool_jobs(spool))
        rc, out, _ = _job(root, jobs, "broken", dict(bad, ROLE="driver"))
        assert rc == 96 and "RUN_ARGV CHAIN: the sequencer's chain goes on with %s" % held in out \
            and len(_spool_jobs(spool)) == n0, out
        subprocess.run(["qdel", held], env={"PATH": _FAKE + os.pathsep + "/usr/bin:/bin",
                                            "FAKE_PBS_SPOOL": spool})
        # a stop asked: no replacement
        with open(os.path.join(wr, "STOP_REQUEST"), "w") as fh:
            fh.write("asked\n")
        rc, out, _ = _job(root, jobs, "broken", dict(bad, ROLE="driver"))
        assert "a stop was requested or the chain ended on purpose: no replacement" in out and \
            len(_spool_jobs(spool)) == n0, out
        os.remove(os.path.join(wr, "STOP_REQUEST"))
        # a driver's command that refuses (exit 2): the chain's deliberate end
        rc, out, _ = _job(root, jobs, "good", {"ROLE": "driver"}, argv=("bash", "-c", "exit 2"))
        assert rc == 2 and "RUN_ARGV CHAIN: the chain ended on purpose: the sequencer job 900.fake " \
            "refused to start (exit 2; its log says why)" in out, out
        assert " 900.fake the sequencer job 900.fake refused to start" in \
            _read(os.path.join(wr, "driver", "CHAIN_END"))
        os.remove(os.path.join(wr, "driver", "CHAIN_END"))
        # a job submitted by code before D-111 (no JOBS_DIR): the activation of before
        rc, out, _ = _job(root, jobs, "noconda", dict(bad, JOBS_DIR=""))
        assert rc == 98 and "RUN_ARGV" not in out, out
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------- the synthetic graph (1b) ---

def _synthetic_project(root, proj, walltime="00:00:20", poll=1, sbi_dir=None):
    """The synthetic graph's configuration under `proj`, its driver block
    small; sbi_dir: where the job scripts come from (default this checkout)."""
    with open(GIULIA, "rb") as fh:
        raw = json.loads(fh.read().decode("ascii"))
    cfg = C._strip_docs(raw)
    cfg["dataset"] = "synth"
    for k in list(cfg["paths"]):
        cfg["paths"][k] = os.path.join(proj, k)
    cfg["paths"]["sbi_dir"] = sbi_dir or _REPO
    cfg["real"]["cohort_config"] = os.path.join(proj, "cohort.json")
    cfg["real"]["extract_root"] = os.path.join(proj, "extracted")
    for k in ("sim_main", "out_root", "tools"):
        cfg["sim"][k] = os.path.join(proj, k)
    cfg["driver"] = {"queue": "test", "ncpus": 1, "mem": "1gb", "walltime": walltime,
                     "poll_seconds": int(poll)}
    os.makedirs(proj, exist_ok=True)
    path = os.path.join(proj, "synth.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    return path, cfg


def _sg_env(root, **extra):
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([_FAKE, _FAKE_CONDA, env.get("PATH", "")])
    env["FAKE_PBS_SPOOL"] = os.path.join(root, "spool")
    env["FAKE_PBS_DELAY"] = "0.5"
    env["FAKE_PBS_ENFORCE_WALLTIME"] = "1"
    env["FAKE_PBS_QSELECT_SUFFIX"] = ".cluster"
    env["PIPELINE_SKIP_CONDA"] = "1"
    env["PIPELINE_PYTHON"] = PYTHON
    env["PYTHONPATH"] = _HERE + os.pathsep + _PIPE + os.pathsep + env.get("PYTHONPATH", "")
    env["SYNTH_JOB_SLEEP"] = "0.3"
    env["PIPELINE_REPLACE_BASE_S"] = "1"
    env["PIPELINE_CHAIN_QSUB_WAIT_S"] = "1"
    for k in ("SYNTH_REFUSE", "SYNTH_SILENT_KILL", "SYNTH_CRASH_POLL", "SYNTH_CRASH_ONCE",
              "MUTATION_PATCH", "PBS_JOBID", "MUTATION", "FAKE_PBS_HOSTS",
              "FAKE_PBS_HOST_BY_NAME", "FAKE_PBS_FAIL_START", "SYNTH_RUN_ARGV"):
        env.pop(k, None)
    env.update(dict((k, str(v)) for k, v in extra.items()))
    return env


def _sg(proj, env, *args, **kw):
    cmd = [PYTHON, SG, args[0], "--config", os.path.join(proj, "synth.json")] + list(args[1:])
    p = subprocess.run(cmd, cwd=proj, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=kw.get("timeout", 180))
    return p.returncode, p.stdout


def _plog(cfg):
    return os.path.join(cfg["paths"]["work_root"], "pipeline.log")


def _driver_logs(cfg):
    """The tails of the driver folder's logs (a failing check's evidence)."""
    out = []
    for p in sorted(glob.glob(os.path.join(cfg["paths"]["work_root"], "driver", "*.log"))):
        out.append("-- %s\n%s" % (os.path.basename(p), "\n".join(_read(p).splitlines()[-12:])))
    return "\n".join(out)


def _wait_log(cfg, text, timeout=150.0, count=1):
    try:
        return _wait(lambda: _read(_plog(cfg)) if _read(_plog(cfg)).count(text) >= count else None,
                     timeout, "pipeline.log does not hold %r x%d" % (text, count))
    except AssertionError as exc:
        raise AssertionError("%s; its tail:\n%s\nthe driver logs:\n%s"
                             % (exc, "\n".join(_read(_plog(cfg)).splitlines()[-30:]),
                                _driver_logs(cfg)))


def _named(root, name):
    return [j for j in _spool_jobs(os.path.join(root, "spool")) if j["name"] == name]


def _sweep(root, env):
    for _ in range(4):
        live = [j["id"] for j in _spool_jobs(os.path.join(root, "spool")) if j["state"] != "F"]
        if not live:
            return
        subprocess.run(["qdel"] + live, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1.0)


def _quiet(root, timeout=60.0):
    _wait(lambda: all(j["state"] == "F" for j in _spool_jobs(os.path.join(root, "spool"))),
          timeout, "jobs still live: %r" % [(j["id"], j["name"], j["state"]) for j in
                                            _spool_jobs(os.path.join(root, "spool"))
                                            if j["state"] != "F"])


def _no_overlap(drivers):
    ran = sorted([(j["started"], j["finished"] or time.time(), j["id"]) for j in drivers
                  if j["started"] and j.get("exit_status") not in (-2, 96)])
    for (s0, f0, a), (s1, f1, b) in zip(ran, ran[1:]):
        assert s1 >= f0 - 1e-6, "drivers %s and %s overlap: %r" % (a, b, ran)


def _test_chain_end():
    """A stage refusal ends the chain on purpose; --status prints it; a
    submission removes it and the pipeline finishes."""
    root = tempfile.mkdtemp(prefix="b7c_end_")
    env = _sg_env(root, SYNTH_REFUSE="B")
    try:
        proj = os.path.join(root, "p")
        path, cfg = _synthetic_project(root, proj)
        rc, out = _sg(proj, env, "submit")
        assert rc == 0 and "SUBMITTED the sequencer" in out, out
        _wait_log(cfg, "the chain ends (failed, exit 3)")
        ce = _read(os.path.join(cfg["paths"]["work_root"], "driver", "CHAIN_END"))
        assert " failed (exit 3)" in ce, ce
        _quiet(root)
        rc, out = _sg(proj, env, "status")
        assert "chain: ended on purpose at " in out and "failed (exit 3)" in out, out
        env2 = _sg_env(root)
        rc, out = _sg(proj, env2, "submit", "--retry", "B")
        assert rc == 0 and "the chain's deliberate end of " in out and \
            "was removed by this submission (D-111)" in out, out
        _wait_log(cfg, "PIPELINE DONE")
        _quiet(root)
        ce = _read(os.path.join(cfg["paths"]["work_root"], "driver", "CHAIN_END"))
        assert " done (exit 0)" in ce, ce
    finally:
        _sweep(root, env)
        shutil.rmtree(root, ignore_errors=True)


def _test_chain_replace():
    """The second driver job lands on a node where its environment cannot
    start: it exits 96 and its replacement, delayed with -a, finishes."""
    root = tempfile.mkdtemp(prefix="b7c_rep_")
    os.makedirs(os.path.join(root, "home"))
    envs_root = _envs_root(root)
    env = _sg_env(root, PIPELINE_SKIP_CONDA="0", FAKE_CONDA_ENVS=envs_root,
                  FAKE_CONDA_ABSENT_HOSTS="broken", FAKE_ENV_BROKEN_HOSTS="broken",
                  FAKE_PBS_HOST_BY_NAME="pl_synth:2=broken", HOME=os.path.join(root, "home"),
                  SYNTH_SLEEP_A="8", SYNTH_SLEEP_B="8")
    try:
        proj = os.path.join(root, "p")
        path, cfg = _synthetic_project(root, proj, sbi_dir=_jobs_copy(root))
        rc, out = _sg(proj, env, "submit")
        assert rc == 0 and "SUBMITTED the sequencer" in out, out
        assert "environment sbi_export: " in out and \
            "(recorded for nodes without conda, D-111)" in out, out
        wr = cfg["paths"]["work_root"]
        assert _read(os.path.join(wr, "envs", "sbi_export.prefix")) == \
            os.path.join(envs_root, "sbi_export") + "\n"
        log = _wait_log(cfg, "PIPELINE DONE", timeout=120)
        drivers = _named(root, "pl_synth")
        d2 = drivers[1]
        assert d2.get("host") == "broken" and d2["exit_status"] == 96, d2
        assert ("driver %s: could not start on broken (the environment sbi_export cannot start); "
                "its replacement " % d2["id"]) in log and \
            "starts in 1 s (replacement 1 since the last full tick; D-111)" in log, \
            (log[-3000:], "mutation replace")
        rep = drivers[2]
        assert rep["exec_time"] is not None and rep["vars"]["ARGV_FILE"] == d2["vars"]["ARGV_FILE"] \
            and rep["out"].startswith(os.path.join(wr, "driver", "replace.")), rep
        d2log = _read(d2["out"])
        assert "RUN_ARGV ENV FAULT: host broken: the environment sbi_export cannot start" in d2log \
            and "RUN_ARGV CHAIN: submitted the replacement sequencer job %s" % rep["id"] in d2log \
            and d2log.endswith("RUN_ARGV EXIT: 96\n"), d2log
        assert "driver %s on " % rep["id"] in log, log[-3000:]
        _quiet(root)
        assert not os.path.exists(os.path.join(wr, "driver", "env_replacements"))
        assert " done (exit 0)" in _read(os.path.join(wr, "driver", "CHAIN_END"))
        _no_overlap(_named(root, "pl_synth"))
        for s in ("A", "B", "C"):
            assert len(_named(root, "synth_" + s)) == 1, s
    finally:
        _sweep(root, env)
        shutil.rmtree(root, ignore_errors=True)


def _test_watchdog():
    """The second driver job fails to start (PBS -2) while stage A's job runs:
    that job's end revives the chain; with a stop asked first, nothing."""
    for stop_first in (False, True):
        root = tempfile.mkdtemp(prefix="b7c_dog_")
        env = _sg_env(root, SYNTH_RUN_ARGV="1", FAKE_PBS_FAIL_START="pl_synth:2",
                      SYNTH_SLEEP_A="30")
        try:
            proj = os.path.join(root, "p")
            path, cfg = _synthetic_project(root, proj, sbi_dir=_jobs_copy(root))
            wr = cfg["paths"]["work_root"]
            rc, out = _sg(proj, env, "submit")
            assert rc == 0 and "SUBMITTED the sequencer" in out, out
            _wait(lambda: len(_named(root, "pl_synth")) >= 2 and
                  _named(root, "pl_synth")[1]["state"] == "F", 60, "the second driver ended")
            d2 = _named(root, "pl_synth")[1]
            assert d2["exit_status"] == -2 and not os.path.exists(d2["out"]), d2
            a = _named(root, "synth_A")
            assert len(a) == 1 and a[0]["state"] != "F", ("A outlives the first driver", a)
            if not stop_first:
                log = _wait_log(cfg, "PIPELINE DONE", timeout=120)
                assert ("watchdog: job %s on " % a[0]["id"]) in log and \
                    "no sequencer job was queued, held or running and the chain had not ended " \
                    "on purpose; submitted " in log and log.count("watchdog: ") == 1, \
                    (log[-3000:], "mutation watchdog")
                assert "RUN_ARGV CHAIN: no sequencer job was queued" in _read(a[0]["out"])
                _quiet(root)
                assert len(_named(root, "synth_A")) == 1 and len(_named(root, "synth_B")) == 1
            else:
                rc, out = _sg(proj, env, "stop")
                assert rc == 0 and "stop: no sequencer is running" in out, out
                assert "stopped on request" in _read(os.path.join(wr, "driver", "CHAIN_END"))
                _quiet(root, timeout=60)
                time.sleep(2.0)
                log = _read(_plog(cfg))
                assert "watchdog: " not in log and len(_named(root, "pl_synth")) == 2, \
                    (log[-3000:], "mutation chain_end: a stopped chain was revived")
                rc, out = _sg(proj, _sg_env(root, SYNTH_RUN_ARGV="1"), "submit")
                assert rc == 0 and "was removed by this submission (D-111)" in out, out
                _wait_log(cfg, "PIPELINE DONE", timeout=120)
                _quiet(root)
        finally:
            _sweep(root, env)
            shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------- end to end, the fixture ---

def _seed_probes(cfg):
    """The gate's probe files, so the stage reuses them (no probe jobs)."""
    camp = os.path.join(cfg["paths"]["tune_root"], "S-A1")
    for b_ in (16, 32):
        d = os.path.join(camp, "gate", "b%d" % b_)
        os.makedirs(d)
        with open(os.path.join(d, "A1_seed0_probe.json"), "w") as fh:
            json.dump({"step_seconds": 0.01, "val_row_seconds": 0.001, "maxrss_gb": 0.5,
                       "steps_per_epoch": 3, "projected_trial_seconds": 1.0}, fh)
    return camp


def _test_env_faults():
    import test_smoke_block7 as B7
    root = tempfile.mkdtemp(prefix="b7c_e2e_")
    try:
        path, cfg = B7._synthetic_project(root)
        copy = os.path.join(root, "sbi_copy")
        shutil.copytree(os.path.join(_REPO, "hpc"), os.path.join(copy, "hpc"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        _mutate(os.path.join(copy, "hpc", "pipeline", "jobs"))
        cfg["paths"]["sbi_dir"] = copy
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=1, sort_keys=True)
        camp = _seed_probes(cfg)
        os.makedirs(os.path.join(root, "home"))
        envs_root = _envs_root(root)
        spool = os.path.join(root, "spool")
        env = B7._env(spool)
        env["PATH"] = os.pathsep.join([_FAKE, _FAKE_CONDA, env["PATH"]])
        env["PIPELINE_SKIP_CONDA"] = "0"
        env.pop("PIPELINE_PYTHON", None)
        env.update({"FAKE_CONDA_ENVS": envs_root, "FAKE_CONDA_ABSENT_HOSTS": "noconda,broken",
                    "FAKE_ENV_BROKEN_HOSTS": "broken", "HOME": os.path.join(root, "home"),
                    "FAKE_PBS_HOSTS": ",".join(["noconda", "broken"] +
                                               ["n%02d" % k for k in range(3, 61)]),
                    "FAKE_PBS_HOST_BY_NAME": "bf_synth_s1:1=broken",
                    "PIPELINE_AVOID_HOLD_S": "2"})
        for k in ("MUTATION", "CONDA_ENVS_PATH"):
            env.pop(k, None)
        cmd = [PYTHON, _seq_boot(root), "run", "--config", path, "--poll-seconds", "0.5"]
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, timeout=1500)
        out = proc.stdout
        assert proc.returncode == 0 and "PIPELINE DONE" in out, out[-6000:]
        st = json.load(open(os.path.join(cfg["paths"]["work_root"], "state.json")))
        ss = st["stages"]["search"]["sub"]
        trials = ss["trials"]
        jobs = dict((j["id"], j) for j in st["stages"]["search"]["jobs"])
        first = sorted(trials, key=lambda t: int(trials[t]["jobs"][0].split(".")[0]))
        t_noconda, t_broken = first[0], first[1]
        # the first trial ran on noconda through the environment's folder
        jl = _read(jobs[trials[t_noconda]["jobs"][0]]["log"])
        assert "RUN_ARGV ENV: host noconda: conda unavailable (bash: conda: command not found); " \
            "sbi_env activated from its folder %s" % os.path.join(envs_root, "sbi_env") in jl, \
            (jl[-3000:], "mutation fallback")
        assert trials[t_noconda]["status"] == "ok" and trials[t_noconda]["resubmits"] == 0 and \
            len(trials[t_noconda]["jobs"]) == 1, trials[t_noconda]
        # the second trial's first job on broken: 96, submitted again, not counted
        tb = trials[t_broken]
        assert tb["status"] == "ok" and tb["resubmits"] == 0 and len(tb["jobs"]) == 2, \
            (tb, "mutation env_free: the fault counted")
        assert [(e["host"], e["job"]) for e in tb["env_faults"]] == [("broken", tb["jobs"][0])], tb
        bl = _read(jobs[tb["jobs"][0]]["log"])
        assert "RUN_ARGV ENV FAULT: host broken: the environment sbi_env cannot start" in bl and \
            bl.endswith("RUN_ARGV EXIT: 96\n"), bl
        assert jobs[tb["jobs"][0]]["host"] == "broken" and \
            jobs[tb["jobs"][0]]["exit_status"] == 96, jobs[tb["jobs"][0]]
        assert ("search: trial %s could not start on broken (host broken: the environment sbi_env "
                "cannot start" % t_broken[:12]) in out and \
            "an environment fault, not the trial's (D-111); submitted again at once (not a " \
            "resubmission); 1 node fault(s) on broken so far" in out, out[-6000:]
        assert ss["node_fault_hosts"] == {"broken": 1} and ss["env_fault_streak"]["n"] == 0, ss
        # every trial's argv carries --resume (D-111)
        argvs = glob.glob(os.path.join(camp, "jobs", "*.argv"))
        assert len(argvs) >= 5 and all("--resume\n" in _read(a) for a in argvs), \
            ([a for a in argvs if "--resume\n" not in _read(a)], "mutation always_resume")
        # the member's first job on broken: likewise
        bs = st["stages"]["best_fit"]["sub"]
        m1 = bs["members"]["1"]
        assert m1["status"] == "ok" and m1["resubmits"] == 0 and len(m1["jobs"]) == 2 and \
            m1["env_faults"][0]["host"] == "broken", (m1, "mutation env_free")
        assert "best_fit: member seed 1 could not start on broken" in out and \
            "an environment fault, not the member's (D-111)" in out
        assert bs["node_fault_hosts"] == {"broken": 1}, bs
        margv = glob.glob(os.path.join(cfg["paths"]["runs_root"], "synth_best", "jobs", "*.argv"))
        assert margv and all("--resume" in _read(a) for a in margv), margv
        # the sequencer's own commands recorded the training environment's prefix
        assert _read(os.path.join(cfg["paths"]["work_root"], "envs", "sbi_env.prefix")) == \
            os.path.join(envs_root, "sbi_env") + "\n"
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    _test_activate_units()


def test_reference():
    _test_job_units()


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    _test_chain_end()


def test_contract():
    _test_env_faults()


def test_determinism():
    _test_chain_replace()


def test_edge_cases():
    _test_watchdog()


CHECKS = [("test_known_answer", test_known_answer), ("test_reference", test_reference),
          ("test_convergence", test_convergence), ("test_invariants", test_invariants),
          ("test_contract", test_contract), ("test_determinism", test_determinism),
          ("test_edge_cases", test_edge_cases),
          ("activate_units", _test_activate_units), ("job_units", _test_job_units),
          ("chain_end", _test_chain_end), ("env_faults", _test_env_faults),
          ("chain_replace", _test_chain_replace), ("watchdog", _test_watchdog)]
_PARTS = ("activate_units", "job_units", "chain_end", "env_faults", "chain_replace", "watchdog")


_NEEDS_TRAINING = ("test_contract", "env_faults")      # the real tuner and runner in the jobs


def main():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        missing = None
    except ImportError as exc:
        missing = str(exc)
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    checks = [(n, f) for n, f in CHECKS if (n in wanted if wanted else n not in _PARTS)]
    if missing:
        # the other checks are bash and stdlib: they run in any environment
        def _skip(why=missing):
            raise unittest.SkipTest("needs the training environment (%s)" % why)
        checks = [(n, _skip if n in _NEEDS_TRAINING else f) for n, f in checks]
    if MUTATION:
        print("MUTATION=%s: the named piece is disabled; the checks it names must FAIL" % MUTATION)
    rows, bad = [], 0
    t0 = time.time()
    for name, fn in checks:
        t1 = time.time()
        try:
            fn()
            rows.append((name, "PASS", "", time.time() - t1))
        except unittest.SkipTest as exc:
            rows.append((name, "SKIP", str(exc), time.time() - t1))
        except AssertionError as exc:
            bad += 1
            rows.append((name, "FAIL", str(exc)[:2500], time.time() - t1))
        except Exception:
            bad += 1
            rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
