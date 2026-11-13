"""Block 22 (the evaluation's jobs and run sheet), derived from specs/SPEC.md
block 22 and block 23's `--what data` (oracles O22.1-O22.3 of the t1-C
report).

Pure ASCII, LF only.
"""

import json
import os
import re
import shutil
import stat
import subprocess

import pytest

from conftest import EVAL, PIPE, PY, REPO, files_under, load_json, run_py, write_eval_config

JOB = os.path.join(EVAL, "jobs", "eval_run.pbs")
SUBMIT = os.path.join(EVAL, "ev_submit.py")
WRAPPER = os.path.join(EVAL, "submit_eval.sh")
FAKE = os.path.join(PIPE, "tests", "fake_pbs")
SYNTH = os.path.join(PIPE, "projects", "synth", "synth.json")
SYNTH_EVAL = os.path.join(PIPE, "projects", "synth", "synth_eval.json")


# --------------------------------------------------------- O22.1 the job

def test_job_script_static():
    raw = open(JOB, "rb").read()
    assert all(b < 128 for b in raw), "non-ASCII byte"
    assert b"\r" not in raw, "CR byte"
    assert os.stat(JOB).st_mode & stat.S_IXUSR, "not executable"
    r = subprocess.run(["bash", "-n", JOB], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0, r.stdout
    text = raw.decode("ascii")
    code = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
    body = "\n".join(code)
    assert body.index("pipeline_activate") < body.index("roadmap_job_start")
    assert not [l for l in code if re.search(r"(^|\s)conda activate", l)]


def _job(tmp_path, argv_lines, **env):
    argv = tmp_path / "cmd.argv"
    argv.write_text("\n".join(argv_lines) + "\n")
    e = dict(os.environ, ARGV_FILE=str(argv), WORKDIR=str(tmp_path), SKIP_CONDA="1",
             ROADMAP_STAGE="", ENV_NAME="none")
    e.pop("ROADMAP_DISABLE", None)
    e.update(env)
    r = subprocess.run(["bash", JOB], env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=300)
    return r.returncode, r.stdout


def _last(out):
    lines = [l for l in out.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def test_job_exit_codes(tmp_path):
    rc, out = _job(tmp_path, ["bash", "-c", "exit 7"], WORKDIR=str(tmp_path / "absent"))
    assert rc == 90 and _last(out) == "EVAL JOB EXIT: 90", out
    rc, out = _job(tmp_path, ["true"], ARGV_FILE=str(tmp_path / "absent.argv"))
    assert rc == 91 and _last(out) == "EVAL JOB EXIT: 91", out
    rc, out = _job(tmp_path, ["true"], SKIP_CONDA="0", JOBS_DIR="")
    assert rc == 95 and _last(out) == "EVAL JOB EXIT: 95", out
    stub = tmp_path / "jobs"
    stub.mkdir()
    (stub / "activate_env.sh").write_text("pipeline_activate() { return 1; }\n")
    rc, out = _job(tmp_path, ["true"], SKIP_CONDA="0", JOBS_DIR=str(stub))
    assert rc == 96 and _last(out) == "EVAL JOB EXIT: 96", out
    rc, out = _job(tmp_path, ["bash", "-c", "exit 7"])
    assert rc == 7 and _last(out) == "EVAL JOB EXIT: 7", out
    rc, out = _job(tmp_path, [PY, "-c", "import sys; print('ARG=%r' % sys.argv[1])", "a b  c"])
    assert rc == 0 and "ARG='a b  c'" in out and _last(out) == "EVAL JOB EXIT: 0", out


def _standin_checkout(tmp_path):
    """A clone with specs/roadmap/hpc_status.sh and a stand-in origin."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    (work / "specs" / "roadmap").mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "specs", "roadmap", "hpc_status.sh"),
                str(work / "specs" / "roadmap" / "hpc_status.sh"))
    g = ["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "x"], check=True)
    subprocess.run(g + ["remote", "add", "origin", str(origin)], check=True)
    subprocess.run(g + ["push", "-q", "origin", "HEAD:refs/heads/main"], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return work, origin


@pytest.mark.parametrize("cmd,rc_want,state", [("exit 0", 0, "done"), ("exit 7", 7, "failed")])
def test_job_markers_and_last_line(tmp_path, cmd, rc_want, state):
    """The roadmap's markers pushed to the origin (done / failed), and the
    job's last line still `EVAL JOB EXIT: <n>` (block 22; the log is the
    job's stdout and stderr, -j oe)."""
    if not shutil.which("flock"):
        pytest.skip("flock is not installed")
    work, origin = _standin_checkout(tmp_path)
    rc, out = _job(tmp_path, ["bash", "-c", cmd], WORKDIR=str(work), ROADMAP_STAGE="I19",
                   PBS_JOBID="4242.fake", PBS_JOBNAME="ev_cal0")
    assert rc == rc_want, out
    r = subprocess.run(["git", "--git-dir", str(origin), "show", "hpc-status:runs/I19/4242.fake.json"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert r.returncode == 0 and ('"state": "%s"' % state) in r.stdout, r.stdout
    assert _last(out) == "EVAL JOB EXIT: %d" % rc_want, out[-800:]


# ----------------------------------------------- O22.2 the launcher's plan

def _committed_cfg(tmp_path):
    d = json.load(open(SYNTH))
    d["paths"]["sbi_dir"] = REPO
    p = tmp_path / "synth.json"
    p.write_text(json.dumps(d, indent=1))
    return str(p), d


def _launch(args, env_extra=None, skip_conda="0"):
    env = dict(os.environ)
    env.pop("PIPELINE_SKIP_CONDA", None)
    if skip_conda == "1":
        env["PIPELINE_SKIP_CONDA"] = "1"
    env.update(env_extra or {})
    return run_py([SUBMIT] + list(args), env=env, timeout=600)


def _qsub_lines(out):
    return [l.strip() for l in out.splitlines() if l.strip().startswith("qsub ")]


def test_committed_dry_run(tmp_path):
    cfgp, d = _committed_cfg(tmp_path)
    e = json.load(open(SYNTH_EVAL))
    rc, out = _launch(["--config", cfgp, "--eval-config", SYNTH_EVAL, "--dry-run"])
    assert rc == 0 and "EVAL DRY RUN:" in out and "nothing written" in out, out[-3000:]
    q = _qsub_lines(out)
    assert len(q) == e["cal_shards"], q
    j = e["jobs"]["calbank"]
    for k, line in enumerate(q):
        name = re.search(r"-N (\S+)", line).group(1)
        assert name == "ev_cal%d" % k and len(name) <= 15
        assert "select=1:ncpus=%d:mem=%s" % (j["ncpus"], j["mem"]) in line
        assert "walltime=%s" % j["walltime"] in line
        assert "ROADMAP_STAGE=I19" in line and "SKIP_CONDA=0" in line
    cmds = [l for l in out.splitlines() if l.strip().startswith("command:")]
    per = e["cal_n"] // e["cal_shards"]
    for k, c in enumerate(cmds):
        t = c.split()
        assert t[t.index("--arm") + 1] == "S" and t[t.index("--prior") + 1] == "box"
        assert t[t.index("--wells-per-donor") + 1] == "1" and t[t.index("--n-per-theta") + 1] == "1"
        assert t[t.index("--n-windows") + 1] == "1" and t[t.index("--seed") + 1] == str(e["cal_seed"])
        assert t[t.index("--n-traces") + 1] == str(per)
        assert t[t.index("--shard-index") + 1] == str(k)
        b = d["synthetic"]["builder"]
        assert float(t[t.index("--fs") + 1]) == b["fs"] and float(t[t.index("--T-win") + 1]) == b["T_win"]
        assert t[t.index("--label-axes") + 1] == ",".join(str(x) for x in b["label_axes"])
    n_folders = sum(1 + bool(q["alt_best_fit"]) for q in d["synthetic"]["searches"])
    assert out.count("EVAL SUBMIT: not complete:") == n_folders
    assert "report: nothing to report yet" in out
    assert not [l for l in q if "-N ev_data" in l] and "ev_dataview.py" not in out


def test_dry_run_writes_nothing(tmp_path):
    cfgp, _d = _committed_cfg(tmp_path)
    out_root = str(tmp_path / "eval_out")
    ec = write_eval_config(str(tmp_path / "e.json"), out_root)
    for what in ("calbank,sample,report", "data"):
        rc, out = _launch(["--config", cfgp, "--eval-config", ec, "--dry-run", "--what", what])
        assert rc == 0, out[-2000:]
    assert not os.path.exists(out_root)


def test_what_data_plan(tmp_path):
    cfgp, _d = _committed_cfg(tmp_path)
    e = json.load(open(SYNTH_EVAL))
    rc, out = _launch(["--config", cfgp, "--eval-config", SYNTH_EVAL, "--dry-run", "--what", "data"])
    assert rc == 0, out[-2000:]
    q = _qsub_lines(out)
    assert len(q) == 1, q
    j = e["jobs"]["data"]
    assert "-N ev_data " in q[0] and "ROADMAP_STAGE=I23" in q[0] and "-W depend" not in q[0]
    assert "select=1:ncpus=%d:mem=%s" % (j["ncpus"], j["mem"]) in q[0]
    assert "walltime=%s" % j["walltime"] in q[0]
    assert "ev_dataview.py" in out


@pytest.mark.parametrize("what", ["calbank,bogus", ""])
def test_bad_what_exit_2(tmp_path, what):
    cfgp, _d = _committed_cfg(tmp_path)
    rc, out = _launch(["--config", cfgp, "--eval-config", SYNTH_EVAL, "--dry-run", "--what", what])
    assert rc == 2, (rc, out[-800:])


def test_wrapper_plan_equals_launcher(tmp_path):
    cfgp, _d = _committed_cfg(tmp_path)
    rc1, a = _launch(["--config", cfgp, "--eval-config", SYNTH_EVAL, "--dry-run"], skip_conda="1")
    env = dict(os.environ, SKIP_CONDA="1")
    r = subprocess.run(["bash", WRAPPER, "--config", cfgp, "--eval-config", SYNTH_EVAL, "--dry-run"],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=600)
    assert rc1 == 0 and r.returncode == 0, r.stdout[-1500:]
    assert _qsub_lines(a) == _qsub_lines(r.stdout)


# ------------------------------------------------- O22.3 the calibration bank

def test_foreign_calibration_bank_refused(tmp_path):
    cfgp, _d = _committed_cfg(tmp_path)
    out_root = str(tmp_path / "eval_out")
    ec = write_eval_config(str(tmp_path / "e.json"), out_root)
    cal = os.path.join(out_root, "calbank")
    os.makedirs(cal)
    side = os.path.join(cal, "shard_0000.json")
    with open(side, "w") as fh:
        json.dump({"provenance": {"argv": ["--arm", "S", "--seed", "999"]}}, fh)
    before = open(side).read()
    rc, out = _launch(["--config", cfgp, "--eval-config", ec, "--what", "calbank"])
    assert rc == 3 and "REFUSE" in out, out[-1500:]
    assert open(side).read() == before and files_under(cal) == ["shard_0000.json"]
    os.remove(side)
    open(os.path.join(cal, "shard_0099.npz"), "wb").close()
    rc, out = _launch(["--config", cfgp, "--eval-config", ec, "--what", "calbank"])
    assert rc == 3 and "REFUSE" in out, out[-1500:]


# --------------------------------------- the report after live samplings

def _fake_env(spool, delay):
    return {"PATH": FAKE + os.pathsep + os.environ.get("PATH", ""), "FAKE_PBS_SPOOL": spool,
            "FAKE_PBS_DELAY": str(delay), "FAKE_PBS_ENFORCE_WALLTIME": "0",
            "PIPELINE_SKIP_CONDA": "1", "ROADMAP_DISABLE": "1"}


def test_report_waits_for_live_samplings(evaluation, tmp_path):
    """SPEC block 22: the report runs after the sampling jobs. With a sampling
    queued (live) and the report planned on its own (--what report), the
    report's qsub line must still wait for it."""
    out_root = str(tmp_path / "eval_out")
    ec = write_eval_config(str(tmp_path / "e.json"), out_root)
    shutil.copytree(os.path.join(evaluation["out"], "samples"), os.path.join(out_root, "samples"))
    shutil.copytree(os.path.join(evaluation["out"], "calbank"), os.path.join(out_root, "calbank"))
    spool = str(tmp_path / "spool")
    os.makedirs(spool)
    env = _fake_env(spool, 3000)
    ids = []
    try:
        rc, out = _launch(["--config", evaluation["cfg"], "--eval-config", ec, "--what", "sample",
                           "--resample", "all"], env_extra=env, skip_conda="1")
        assert rc == 0 and "EVAL SUBMIT OK: 3 job(s) submitted" in out, out[-2000:]
        ids = re.findall(r"submitted (\S+\.fake)", out)
        rc, out = _launch(["--config", evaluation["cfg"], "--eval-config", ec, "--what", "report",
                           "--dry-run"], env_extra=env, skip_conda="1")
        assert rc == 0, out[-2000:]
        q = _qsub_lines(out)
        assert len(q) == 1 and "-N ev_report" in q[0], q
        assert "-W depend=afterok:" in q[0], q[0]
    finally:
        for j in ids:
            subprocess.run(["qdel", j], env=dict(os.environ, **env), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)


def test_sample_staleness(evaluation, tmp_path):
    """A sampling done under the current values is done; another n_post makes
    every one stale (named); a changed BEST_FIT_COMPLETE.json makes its
    ensemble stale."""
    env = {"PIPELINE_SKIP_CONDA": "1"}
    rc, out = _launch(["--config", evaluation["cfg"], "--eval-config", evaluation["ecfg"],
                       "--what", "sample", "--dry-run"], env_extra=env, skip_conda="1")
    assert rc == 0 and "0 job(s) would be submitted, 3 done" in out, out[-1500:]
    e = load_json(evaluation["ecfg"])
    ec = write_eval_config(str(tmp_path / "e.json"), e["out_root"], n_post=80,
                           cal_glob=e["cal_glob"])
    rc, out = _launch(["--config", evaluation["cfg"], "--eval-config", ec, "--what", "sample",
                       "--dry-run"], env_extra=env, skip_conda="1")
    assert rc == 0 and out.count("n_post was 40, now 80") == 3, out[-2000:]
    bf = os.path.join(evaluation["root"], "runs_root", "S-A1", "synth_best", "BEST_FIT_COMPLETE.json")
    keep = open(bf, "rb").read()
    try:
        with open(bf, "wb") as fh:
            fh.write(keep + b"\n")
        rc, out = _launch(["--config", evaluation["cfg"], "--eval-config", evaluation["ecfg"],
                           "--what", "sample", "--dry-run"], env_extra=env, skip_conda="1")
        assert rc == 0 and "1 job(s) would be submitted, 2 done" in out, out[-2000:]
        assert "sample:S-A1:A1 would be submitted (stale: its best fit's" in out
    finally:
        with open(bf, "wb") as fh:
            fh.write(keep)
