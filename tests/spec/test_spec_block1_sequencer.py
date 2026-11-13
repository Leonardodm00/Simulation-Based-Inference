"""Spec-derived tests of the sequencer's loop and resume rule (SPEC secs.
2.1-2.2 and block 1, its amendments of 2026-10-16, 2026-10-18 (a), (c)
and (e)); t1-A run of 2026-11-13 (D-131).

The scheduler is an in-process stand-in (submit / status / select) and the
graph a toy chain A -> B -> C whose artefacts of record are plain files a
"job" writes when it ends. Expected values are the spec's: the exit codes
0 / 2 / 3 / 4, the statuses, the number of submissions.
"""
import json
import os
import socket
import time

import pytest

from conftest import GIULIA

import config as C
import pbs as PBS
import pipeline as P
import stages as S
from state import State


# ------------------------------------------------------------- fixtures ----

class FakeSched(object):
    """Jobs finish (F, exit 0) at their first status unless their name is in
    `hold` (state R); a finishing job runs its callback (writes an artefact)."""

    def __init__(self):
        self.jobs = {}
        self.n = 0
        self.hold = set()
        self.on_finish = {}
        self.submitted = []

    def submit(self, argv, cwd=None, env=None):
        self.n += 1
        jid = "%d.fake" % self.n
        name = argv[argv.index("-N") + 1]
        self.jobs[jid] = {"name": name, "state": "Q"}
        self.submitted.append(name)
        return jid

    def status(self, jid):
        j = self.jobs.get(jid)
        if j is None:
            return PBS.JobStatus(jid, "unknown")
        if j["name"] in self.hold:
            return PBS.JobStatus(jid, "R")
        if j["state"] != "F":
            j["state"] = "F"
            cb = self.on_finish.get(j["name"])
            if cb:
                cb()
        return PBS.JobStatus(jid, "F", exit_status=0)

    def select(self, name):
        return []


class Toy(S.Stage):
    def __init__(self, name, deps, root, behaviour="ok", max_resubmits=0):
        self.name = name
        self.deps = tuple(deps)
        self.root = root
        self.behaviour = behaviour
        self.max_resubmits = max_resubmits
        self.starts = 0

    def art(self):
        return os.path.join(self.root, self.name + ".done")

    def is_done(self, cfg):
        p = self.art()
        return os.path.isfile(p), p

    def start(self, cfg, st, sub):
        self.starts += 1
        if self.behaviour == "start_refuses":
            raise RuntimeError("REFUSED: toy start refusal")
        if self.behaviour == "deferred":
            raise S.Deferred("toy waits for 1 core")
        sub.submit(["qsub", "-N", "t_" + self.name, "-l", "select=1:ncpus=1",
                    "-o", os.path.join(self.root, self.name + ".log"), "job.pbs"], "main")

    def verify(self, cfg, st):
        if self.behaviour == "refuse":
            return False, "REFUSED: toy verifier refusal"
        return S.Stage.verify(self, cfg, st)

    def dry_run(self, cfg, st):
        return ["qsub toy %s" % self.name]


def _cfg(tmp_path, **changes):
    raw = json.load(open(GIULIA))
    raw["paths"]["work_root"] = str(tmp_path / "work")
    for k, v in changes.items():
        raw[k] = v
    p = tmp_path / "toy.json"
    p.write_text(json.dumps(raw, indent=1))
    return C.load_config(str(p))


def _graph(tmp_path, sched, behaviours=None, max_resubmits=0):
    behaviours = behaviours or {}
    root = str(tmp_path / "arts")
    os.makedirs(root, exist_ok=True)
    g = [Toy("A", (), root, behaviours.get("A", "ok"), max_resubmits),
         Toy("B", ("A",), root, behaviours.get("B", "ok"), max_resubmits),
         Toy("C", ("B",), root, behaviours.get("C", "ok"), max_resubmits)]
    for s in g:
        if s.behaviour == "ok":           # a refusing job writes no artefact
            sched.on_finish["t_" + s.name] = (lambda p=s.art(): open(p, "w").write("x"))
    return g


def _run(cfg, g, sched, **kw):
    lines = []
    kw.setdefault("poll_seconds", 0.0)
    kw.setdefault("max_ticks", 20)
    rc = P.run(cfg, graph=g, pbs_module=sched, log=lines.append, **kw)
    return rc, lines


def _statuses(cfg):
    st = State.load(cfg.data["paths"]["work_root"])
    return dict((k, v["status"]) for k, v in st.data["stages"].items())


# ---------------------------------------------------------------- tests ----

def test_full_run_exit0_every_stage_done(tmp_path):
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    rc, lines = _run(cfg, g, sched)
    assert rc == 0                                                  # sec. 2.2: exit 0
    assert _statuses(cfg) == {"A": "done", "B": "done", "C": "done"}
    assert sched.submitted == ["t_A", "t_B", "t_C"]
    assert any("PIPELINE DONE" in l for l in lines)
    assert not os.path.exists(os.path.join(cfg.data["paths"]["work_root"], "pipeline.lock"))


def test_pending_with_artefact_becomes_skipped(tmp_path):
    """sec. 2.2 (2): pending and done(s) holds -> skipped, never started."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    open(g[2].art(), "w").write("pre-existing")
    rc, _ = _run(cfg, g, sched)
    assert rc == 0
    assert _statuses(cfg) == {"A": "done", "B": "done", "C": "skipped"}
    assert g[2].starts == 0 and sched.submitted == ["t_A", "t_B"]


def test_done_is_not_rechecked(tmp_path):
    """sec. 2.2 (1): done / skipped are left alone; a later deletion of an
    artefact does not silently re-run the stage."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    assert _run(cfg, g, sched)[0] == 0
    os.remove(g[0].art())
    rc, _ = _run(cfg, g, sched)
    assert rc == 0 and sched.submitted == ["t_A", "t_B", "t_C"]     # nothing submitted again
    assert _statuses(cfg)["A"] == "done"


def test_refusal_stops_exit3_and_stays_failed(tmp_path):
    """sec. 2.2 (4): a verifier's refusal stops the run (exit 3, a STOP:
    line); a re-run without --retry stops again with the same report and
    submits nothing; --retry B starts B again from pending."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched, {"B": "refuse"})
    rc, lines = _run(cfg, g, sched)
    assert rc == 3
    assert any(l.startswith("STOP:") and "B" in l for l in lines)
    assert _statuses(cfg) == {"A": "done", "B": "failed", "C": "pending"}
    n = len(sched.submitted)
    rc2, lines2 = _run(cfg, g, sched)
    assert rc2 == 3 and len(sched.submitted) == n
    assert any(l.startswith("STOP:") for l in lines2)
    g[1].behaviour = "ok"
    sched.on_finish["t_B"] = (lambda p=g[1].art(): open(p, "w").write("x"))
    rc3, _ = _run(cfg, g, sched, retry=("B",))
    assert rc3 == 0 and _statuses(cfg) == {"A": "done", "B": "done", "C": "done"}


def test_start_refusal_is_a_failed_stage(tmp_path):
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched, {"A": "start_refuses"})
    rc, lines = _run(cfg, g, sched)
    assert rc == 3 and _statuses(cfg)["A"] == "failed"
    assert sched.submitted == []


def test_retry_unknown_stage_exit2(tmp_path):
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    rc, lines = _run(cfg, g, sched, retry=("nope",))
    assert rc == 2 and sched.submitted == []


def test_retry_refused_while_a_job_runs(tmp_path):
    """Block 1, amended 2026-10-18 (a): --retry of a stage whose recorded
    job is not finished is refused, exit 2, nothing submitted and the state
    file not rewritten."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    sched.hold.add("t_A")
    rc, _ = _run(cfg, g, sched, max_ticks=1)
    assert rc == 0 and _statuses(cfg)["A"] == "running"
    sp = State.path_of(cfg.data["paths"]["work_root"])
    before = open(sp, "rb").read()
    rc2, lines = _run(cfg, g, sched, retry=("A",))
    assert rc2 == 2
    assert any("REFUSE: --retry A" in l for l in lines)
    assert open(sp, "rb").read() == before
    assert sched.submitted == ["t_A"]


def test_running_stage_reattaches_without_resubmission(tmp_path):
    """sec. 2.2 (3): a running stage re-attaches to its recorded job; once it
    has finished the verifier reads the artefact: one submission of A."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    sched.hold.add("t_A")
    assert _run(cfg, g, sched, max_ticks=2)[0] == 0
    sched.hold.discard("t_A")
    rc, _ = _run(cfg, g, sched)
    assert rc == 0
    assert sched.submitted.count("t_A") == 1
    assert _statuses(cfg) == {"A": "done", "B": "done", "C": "done"}


def test_killed_job_resubmitted_then_failed(tmp_path):
    """sec. 2.2 (3): a job that ended with neither artefact nor refusal is
    killed; the policy resubmits it at most max_resubmits times, then fails."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched, {"A": "no_artefact"}, max_resubmits=1)
    rc, lines = _run(cfg, g, sched)
    assert rc == 3
    assert sched.submitted == ["t_A", "t_A"]
    assert _statuses(cfg)["A"] == "failed"


def test_lock_held_by_live_pid_exit4(tmp_path):
    """sec. 2.2: exit 4 when another sequencer holds the project's lock (a
    pid lock on this host whose process is alive)."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    wr = cfg.data["paths"]["work_root"]
    os.makedirs(wr)
    lock = os.path.join(wr, "pipeline.lock")
    line = "pid:%d %s 2026-11-13T00:00:00Z\n" % (os.getppid(), socket.gethostname())
    open(lock, "w").write(line)
    rc, _ = _run(cfg, g, sched)
    assert rc == 4 and sched.submitted == []
    assert open(lock).read() == line                       # another's lock is never removed


@pytest.mark.parametrize("pid", [0, -1])
def test_lock_of_pid_0_or_minus1_is_stale(tmp_path, pid):
    """Block 1 (c): pids 0 and -1 are never alive (never os.kill(0 | -1))."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    wr = cfg.data["paths"]["work_root"]
    os.makedirs(wr)
    open(os.path.join(wr, "pipeline.lock"), "w").write(
        "pid:%d %s 2026-11-13T00:00:00Z\n" % (pid, socket.gethostname()))
    rc, _ = _run(cfg, g, sched)
    assert rc == 0
    assert not os.path.exists(os.path.join(wr, "pipeline.lock"))


def test_config_change_refused_without_flag(tmp_path):
    """sec. 3.2: a run under a configuration of another digest is refused
    (exit 2) unless --accept-config-change; a refused run writes nothing;
    the accepted change reaches the state file (block 1 (e))."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    sched.hold.add("t_A")
    assert _run(cfg, g, sched, max_ticks=1)[0] == 0
    sp = State.path_of(cfg.data["paths"]["work_root"])
    before = open(sp, "rb").read()
    raw = json.load(open(cfg.path))
    raw["driver"]["poll_seconds"] = 61
    open(cfg.path, "w").write(json.dumps(raw))
    cfg2 = C.load_config(cfg.path)
    rc, lines = _run(cfg2, g, sched)
    assert rc == 2 and any(l.startswith("REFUSE: the configuration changed") for l in lines)
    assert open(sp, "rb").read() == before
    rc2, _ = _run(cfg2, g, sched, accept_config_change=True, max_ticks=1)
    assert rc2 == 0
    assert State.load(cfg.data["paths"]["work_root"]).data["config_sha256"] == cfg2.sha256


def test_dry_run_submits_and_writes_nothing(tmp_path):
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched)
    rc, lines = _run(cfg, g, sched, dry_run=True)
    assert rc == 0 and sched.submitted == []
    assert not os.path.exists(State.path_of(cfg.data["paths"]["work_root"]))
    assert any("qsub toy A" in l for l in lines) and any("qsub toy B" in l for l in lines)


def test_dry_run_on_a_failed_stage(tmp_path):
    """Block 1, amended 2026-10-16: the dry run reports a stage the state
    records as failed, ends with a STOP: line and exits 3; with --retry it
    exits 0 and the reset is not saved."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched, {"B": "refuse"})
    assert _run(cfg, g, sched)[0] == 3
    sp = State.path_of(cfg.data["paths"]["work_root"])
    before = open(sp, "rb").read()
    rc, lines = _run(cfg, g, sched, dry_run=True)
    assert rc == 3 and lines[-1].startswith("STOP:")
    assert any("B: FAILED in the state file" in l for l in lines)
    rc2, _ = _run(cfg, g, sched, dry_run=True, retry=("B",))
    assert rc2 == 0
    assert open(sp, "rb").read() == before


def test_deferred_start_stays_pending(tmp_path):
    """Block 1 (d): `Deferred` from a start leaves the stage pending, logged
    once per reason (`<stage>: waits for cores: <reason>`)."""
    sched = FakeSched()
    cfg = _cfg(tmp_path)
    g = _graph(tmp_path, sched, {"A": "deferred"})
    rc, lines = _run(cfg, g, sched, max_ticks=3)
    assert rc == 0 and _statuses(cfg)["A"] == "pending"
    assert g[0].starts == 3                                    # start asked at every tick
    assert sum("A: waits for cores" in l for l in lines) == 1


@pytest.mark.parametrize("wall,poll,expected", [
    ("24:00:00", 60, 86400 - 900),       # Giulia v8: 85,500 s (block 1 (c))
    ("01:00:00", 60, 3600 - 900),
    ("00:20:00", 60, 1200 - 300),        # walltime / 4 is the smaller margin
    ("24:00:00", 600, 86400 - 3000),     # 5 x poll is the larger floor
])
def test_driver_budget(tmp_path, wall, poll, expected):
    """Block 1 (c) (5): budget = walltime - min(max(900 s, 5 x poll),
    walltime / 4)."""
    raw = json.load(open(GIULIA))
    raw["driver"]["walltime"] = wall
    raw["driver"]["poll_seconds"] = poll
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw))
    assert P._driver_budget_s(C.load_config(str(p))) == float(expected)


def test_constants_of_the_chain():
    """Block 1 (c), (f): DRIVER_MAX_UNCLEAN 3, SCHEDULER_FAIL_TICKS 10 (the
    environment overrides are the tests' and unset here)."""
    if os.environ.get("PIPELINE_DRIVER_MAX_UNCLEAN") or os.environ.get("PIPELINE_SCHEDULER_FAIL_TICKS"):
        pytest.skip("overridden in this environment")
    assert P.DRIVER_MAX_UNCLEAN == 3 and P.SCHEDULER_FAIL_TICKS == 10


# --------------------------------------------- the commands (block 1 (c)) ---

import subprocess    # noqa: E402
import sys           # noqa: E402

from conftest import FAKE_PBS, PIPE   # noqa: E402

PIPELINE_PY = os.path.join(PIPE, "pipeline.py")


def _cli_env(tmp_path, delay="600"):
    env = dict(os.environ)
    env["PATH"] = FAKE_PBS + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = str(tmp_path / "spool")
    env["FAKE_PBS_DELAY"] = delay           # jobs stay queued: nothing runs
    env["PIPELINE_SKIP_CONDA"] = "1"
    env["SKIP_CONDA"] = "1"
    env.pop("PBS_JOBID", None)
    return env


def _cli(tmp_path, *args, env=None):
    r = subprocess.run([sys.executable, PIPELINE_PY] + list(args), env=env or _cli_env(tmp_path),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=300)
    return r.returncode, r.stdout


def _toy_config(tmp_path):
    raw = json.load(open(GIULIA))
    for k in ("work_root", "banks_root", "runs_root", "tune_root"):
        raw["paths"][k] = str(tmp_path / k)
    raw["paths"]["sbi_dir"] = os.path.dirname(os.path.dirname(PIPE))   # this checkout's jobs/
    p = tmp_path / "toy.json"
    p.write_text(json.dumps(raw, indent=1))
    return str(p), raw


def test_drive_without_pbs_jobid_refused(tmp_path):
    """Block 1 (c): `pipeline.py drive` (the sequencer's job) is refused,
    exit 2, without PBS_JOBID."""
    cfgp, _ = _toy_config(tmp_path)
    rc, out = _cli(tmp_path, "drive", "--config", cfgp)
    assert rc == 2, out


def test_stop_with_no_sequencer(tmp_path):
    """Block 1 (c): `pipeline.py stop` with no pl_<tag> job and no lock:
    `stop: no sequencer is running`, and a stop request left behind is
    removed."""
    cfgp, raw = _toy_config(tmp_path)
    wr = raw["paths"]["work_root"]
    os.makedirs(wr)
    open(os.path.join(wr, "STOP_REQUEST"), "w").write("old\n")
    rc, out = _cli(tmp_path, "stop", "--config", cfgp)
    assert "stop: no sequencer is running" in out, out
    assert not os.path.exists(os.path.join(wr, "STOP_REQUEST"))


def test_submit_then_second_submit_refused(tmp_path):
    """Block 1 (c): `submit` submits one pl_<tag> job running `drive --config
    <file>` from the argv file <work_root>/driver/driver.<k>.argv (exit 0,
    `SUBMITTED the sequencer: job <id> (pl_<tag>)`); a second submission
    while that job is queued is refused (`REFUSE: the sequencer is already
    queued or running`, exit 4). D-111 (1): the chain's template
    driver/chain.qsub (with @ARGV@ and @LOG@, no -W) and driver/revive.argv
    are written at the submission."""
    cfgp, raw = _toy_config(tmp_path)
    wr = raw["paths"]["work_root"]
    env = _cli_env(tmp_path)
    rc, out = _cli(tmp_path, "submit", "--config", cfgp, env=env)
    try:
        assert rc == 0, out
        sub = [l for l in out.split("\n") if "SUBMITTED the sequencer: job " in l]
        assert len(sub) == 1 and "(pl_giulia)" in sub[0]
        argvs = sorted(f for f in os.listdir(os.path.join(wr, "driver")) if f.endswith(".argv")
                       and f.startswith("driver."))
        assert argvs == ["driver.1.argv"]
        toks = open(os.path.join(wr, "driver", "driver.1.argv")).read().split("\n")
        assert "drive" in toks and toks[toks.index("--config") + 1] == os.path.abspath(cfgp)
        tpl = open(os.path.join(wr, "driver", "chain.qsub")).read()
        assert "@ARGV@" in tpl and "@LOG@" in tpl and "-W" not in tpl.split("\n")
        assert os.path.isfile(os.path.join(wr, "driver", "revive.argv"))
        rc2, out2 = _cli(tmp_path, "submit", "--config", cfgp, env=env)
        assert rc2 == 4 and "REFUSE: the sequencer is already queued or running" in out2, out2
    finally:
        spool = env["FAKE_PBS_SPOOL"]
        if os.path.isdir(spool):
            ids = subprocess.run(["qselect"], env=env, stdout=subprocess.PIPE,
                                 universal_newlines=True).stdout.split()
            for j in ids:
                subprocess.run(["qdel", j], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_submit_refuses_a_config_change(tmp_path):
    """Block 1 (c): `submit` checks a run's refusals with the run's own code:
    a configuration change without --accept-config-change -> exit 2 and
    nothing submitted."""
    cfgp, raw = _toy_config(tmp_path)
    wr = raw["paths"]["work_root"]
    st = State.new(wr, "giulia", cfgp, "0" * 64)          # a run under another digest
    st.save()
    env = _cli_env(tmp_path)
    rc, out = _cli(tmp_path, "submit", "--config", cfgp, env=env)
    assert rc == 2 and "REFUSE: the configuration changed" in out
    assert not os.path.isdir(env["FAKE_PBS_SPOOL"]) or not os.listdir(env["FAKE_PBS_SPOOL"])


def test_swappability_another_project(tmp_path):
    """Directive audit (swappability) and D-091: another data set is a new
    configuration and the same command -- a project `toyset` with other
    paths, a 2-shard buffer, 8-core trials, two member seeds, a 2-level
    space override and a 30 s tick: the dry run exits 0, names no `giulia`,
    and carries the new values."""
    raw = json.load(open(GIULIA))
    raw["dataset"] = "toyset"
    for k in raw["paths"]:
        raw["paths"][k] = str(tmp_path / ("p_" + k))
    raw["loader"]["buffer_shards"] = 2
    raw["search"]["trial"]["ncpus"] = 8
    raw["best_fit"]["member_seeds"] = [3, 7]
    raw["search"]["space_overrides"]["embedding_size"] = [16, 32]
    raw["driver"]["poll_seconds"] = 30
    for k in ("cohort_config", "extract_root"):
        raw["real"][k] = str(tmp_path / ("r_" + k))
    for k in ("sim_main", "out_root", "tools"):
        raw["sim"][k] = str(tmp_path / ("s_" + k))
    raw["sim"]["campaign_glob"] = "campaign_toy_v*"
    raw["freeze"]["campaigns"] = "campaign_toy_v1"
    raw["real"]["cohort_tag"] = "toyset"
    raw["freeze"]["out"] = "artifacts/label_axes_toyset.json"
    p = tmp_path / "toyset.json"
    p.write_text(json.dumps(raw, indent=1))
    rc, out = _cli(tmp_path, "run", "--config", str(p), "--dry-run")
    assert rc == 0, out[-3000:]
    assert "giulia" not in out.lower()
    assert "REFUSE" not in out
    assert "bf_toyset_s<k>" in out and "pl_toyset" in out
    assert "ncpus=8" in out
    assert "a tick every 30 s" in out
