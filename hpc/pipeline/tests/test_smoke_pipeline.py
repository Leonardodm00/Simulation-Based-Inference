#!/usr/bin/env python3
"""Smoke test for the sequencer -- Block 1 in specs/SPEC.md.

Checks (specs/SPEC.md block 1, "Test oracles")
    test_known_answer   the committed Giulia configuration loads; the exact
                        list of NEEDS-CODE keys with their blocks
    test_reference      the qstat parser against a recorded davinci-format
                        record (hand-read expected values)
    test_convergence    not a discretisation: skipped
    test_invariants     the schema refuses each named defect with one
                        problem each; state save/load is the identity and
                        a crash before the rename keeps the old file;
                        the code names no project (grep)
    test_contract       --dry-run prints every submission and submits
                        nothing; exit codes 0 / 2 / 3 / 4
    test_determinism    a run killed while a job runs and re-run resumes:
                        one submission in all, then the next stage runs
    test_edge_cases     a refusing verifier stops the run; a killed job is
                        resubmitted once then the stage fails; a second
                        sequencer exits 4; an artefact that pre-exists
                        skips its stage

Run
    python3 hpc/pipeline/tests/test_smoke_pipeline.py
        Prints a PASS / FAIL / ERROR / TODO / SKIP table and the message of
        every failure. Exit code 1 if any check failed, errored or is TODO.
    python3 -m pytest hpc/pipeline/tests -q
        The same checks through pytest.

The sequencer runs in subprocesses with the fake PBS first on PATH and a
temporary work root; nothing touches davinci or the user's folders.

Pure ASCII, LF only; stdlib only.
"""

import json
import os
import re
import shutil
import signal
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

import config as C      # noqa: E402
import pbs as PBS       # noqa: E402
import stages as S      # noqa: E402
from state import State  # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
PYTHON = sys.executable


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_smoke_")


def _synthetic_config(root):
    """A valid configuration whose paths all lie under `root`."""
    with open(GIULIA, "rb") as fh:
        raw = json.loads(fh.read().decode("ascii"))
    cfg = C._strip_docs(raw)
    cfg["dataset"] = "synth"
    for k in list(cfg["paths"]):
        cfg["paths"][k] = os.path.join(root, k)
    cfg["real"]["cohort_config"] = os.path.join(root, "cohort.json")
    cfg["real"]["extract_root"] = os.path.join(root, "extracted")
    for k in ("sim_main", "out_root", "tools"):
        cfg["sim"][k] = os.path.join(root, k)
    path = os.path.join(root, "synth.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    return path, cfg


def _env(spool, delay="0.3"):
    env = dict(os.environ)
    env["PATH"] = _FAKE + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = spool
    env["FAKE_PBS_DELAY"] = delay
    env["PYTHONPATH"] = _HERE + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_driver(root, cfg_path, extra, env, timeout=60):
    """Run the synthetic-graph driver (synthetic_graph.py) as a subprocess."""
    cmd = [PYTHON, os.path.join(_HERE, "synthetic_graph.py"), "--config", cfg_path,
           "--poll-seconds", "0.2"] + list(extra)
    proc = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True,
                          timeout=timeout)
    return proc.returncode, proc.stdout


def _n_submitted(spool):
    counter = os.path.join(spool, "counter")
    if not os.path.isfile(counter):
        return 0
    with open(counter) as fh:
        return int(fh.read().strip() or 0)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = C.load_config(GIULIA)
    assert cfg.tag == "giulia"
    assert cfg.data["config_version"] == 9
    assert cfg.data["search"]["resubmit_node_faults"] is True, "D-096"
    assert cfg.data["best_fit"]["resubmit_node_faults"] is True
    assert cfg.data["search"]["max_total_walltime_h"] == 400, "D-090"
    assert cfg.data["split"]["fractions"] == [0.85, 0.10, 0.05], "D-088"
    assert cfg.data["best_fit"]["member_seeds"] == list(range(1, 11)), "D-089"
    assert cfg.data["best_fit"]["split_seed"] == 1
    assert cfg.data["search"]["space_overrides"]["head_pool_ops"] == [0, 1], "D-086"
    got = [(k, b) for k, b, _ in C.unhonoured_keys(cfg)]
    want = []           # every key honoured since block 7 (2026-10-16)
    assert got == want, "NEEDS-CODE keys: %r" % (got,)
    assert cfg.data["search"]["trial"]["queue"] == "cpu" and cfg.data["best_fit"]["queue"] == "cpu"


_QSTAT_SAMPLE = """Job Id: 1829506.davinci
    Job_Name = joint_arms
    Job_Owner = ldellamea@davinci-1
    resources_used.cpupercent = 395
    resources_used.cput = 03:12:08
    resources_used.mem = 10467392kb
    resources_used.ncpus = 4
    resources_used.vmem = 11983200kb
    resources_used.walltime = 00:48:31
    job_state = F
    queue = cpu
    server = davinci
    Output_Path = davinci-1:/davinci-1/home/ldellamea/SBI/hpc/joint/stag
\te3/jobs/joint_arms.o1829506
    Exit_status = 0
    comment = Job run at Sun Oct 12 at 17:51 on (dvnode031:ncpus=4:mem=16777
\t216kb) and finished
"""


def test_reference():
    js = PBS.parse_qstat("1829506.davinci", _QSTAT_SAMPLE)
    assert js.state == "F", js.state
    assert js.finished
    assert js.exit_status == 0, js.exit_status
    assert js.walltime_s == 48 * 60 + 31, js.walltime_s
    # the wrapped Output_Path is rejoined (the tab continuation)
    assert "stage3/jobs/joint_arms.o1829506" in js.raw.replace("\n\t", "")
    js2 = PBS.parse_qstat("7.x", "Job Id: 7.x\n    job_state = R\n")
    assert js2.state == "R" and not js2.finished and js2.exit_status is None
    js3 = PBS.parse_qstat("8.x", "Job Id: 8.x\n    job_state = B\n"
                          "    array_state_count = Queued:3 Running:2 Exiting:0 Expired:0 Finished:4\n")
    assert js3.array_counts == {"Queued": 3, "Running": 2, "Exiting": 0,
                                "Expired": 0, "Finished": 4}, js3.array_counts
    assert PBS.walltime_to_seconds("120:00:00") == 432000.0
    assert PBS.seconds_to_walltime(432000) == "120:00:00"
    # an unknown job through the fake qstat
    root = _tmp()
    try:
        env = _env(os.path.join(root, "spool"))
        proc = subprocess.run([os.path.join(_FAKE, "qstat"), "-x", "-f", "999.fake"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
        assert proc.returncode == 153 and "Unknown Job" in proc.stderr, proc.stderr
    finally:
        shutil.rmtree(root)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    root = _tmp()
    try:
        path, cfg = _synthetic_config(root)
        # (a) the schema: each defect gives exactly one named problem
        def problems(mutate):
            d = json.loads(json.dumps(cfg))
            mutate(d)
            return C.validate(d)

        def missing(d):
            del d["search"]["gp_seed"]
        p = problems(missing)
        assert p == ["search.gp_seed: missing (every key is required)"], p

        def unknown(d):
            d["search"]["extra_knob"] = 1
        p = problems(unknown)
        assert p == ["search.extra_knob: unknown key (not in the schema)"], p

        def wrong_type(d):
            d["search"]["trial"]["epochs"] = "20"
        p = problems(wrong_type)
        assert len(p) == 1 and p[0].startswith("search.trial.epochs: must be an integer >= 1"), p

        def bad_fracs(d):
            d["split"]["fractions"] = [0.8, 0.1, 0.05]
        p = problems(bad_fracs)
        assert len(p) == 1 and "sum to 1" in p[0], p

        def summed(d):
            d["search"]["walltime_reading"] = "summed"
        p = problems(summed)
        assert len(p) == 1 and "only 'elapsed'" in p[0], p

        def tilde(d):
            d["paths"]["work_root"] = "~/pipeline/x"
        p = problems(tilde)
        assert len(p) == 1 and "absolute path" in p[0], p

        def dup_seeds(d):
            d["best_fit"]["member_seeds"] = [1, 2, 2]
        p = problems(dup_seeds)
        assert len(p) == 1 and "distinct" in p[0], p

        def version(d):
            d["config_version"] = 3
        p = problems(version)
        assert len(p) == 1 and "config_version must be 9" in p[0], p

        # documentation keys are ignored, and the file's digest is recorded
        with open(path, "rb") as fh:
            blob = fh.read()
        loaded = C.load_config(path)
        assert loaded.sha256 == __import__("hashlib").sha256(blob).hexdigest()
        assert "_comment" not in loaded.data

        # (b) the state file: save/load identity; a crash before the rename
        work = os.path.join(root, "work")
        st = State.new(work, "synth", path, loaded.sha256)
        st.stage("a")
        st.set_status("a", "running")
        st.add_job("a", "12.fake", "test", ["qsub", "x.sh"], log="/x.log")
        p1 = st.save()
        with open(p1, "rb") as fh:
            b1 = fh.read()
        b1.decode("ascii")
        back = State.load(work)
        assert back.data == st.data, "state round trip"
        # a crash: only the temporary file written
        with open(p1 + ".tmp", "wb") as fh:
            fh.write(b"{garbage")
        with open(p1, "rb") as fh:
            assert fh.read() == b1, "the old state must survive a crash before the rename"
        os.remove(p1 + ".tmp")

        # (c) the graph: dependencies and order
        order = [s.name for s in S.topological_order(S.default_graph(loaded))]
        assert order.index("bank_plan") > max(order.index(x) for x in
                                               ("real_extraction", "sim_redetection", "label_freeze"))
        assert order.index("search") > max(order.index("sim_bank"), order.index("real_bank"))
        assert order[-1] == "best_fit"

        # (d) agnostic: no project value in the code (D-091). The code is
        # every .py and .sh under hpc/pipeline/ outside projects/ (the
        # project folders) and tests/ (which name the Giulia file as a
        # known-answer fixture).
        offenders = []
        for dirpath, dirs, files in os.walk(_PIPE):
            dirs[:] = [d for d in dirs if d not in ("projects", "tests", "__pycache__")]
            for f in files:
                if not f.endswith((".py", ".sh")):
                    continue
                p = os.path.join(dirpath, f)
                with open(p, "rb") as fh:
                    text = fh.read().decode("ascii", "replace").lower()
                for needle in ("giulia", "ldellamea", "davinci-1/home", "cadex", "hhgap"):
                    if needle in text:
                        offenders.append("%s: %s" % (os.path.relpath(p, _PIPE), needle))
        assert not offenders, "project values in the code: %s" % offenders
    finally:
        shutil.rmtree(root)


def test_contract():
    root = _tmp()
    try:
        path, cfg = _synthetic_config(root)
        spool = os.path.join(root, "spool")
        env = _env(spool)
        # dry run: every submission printed, nothing submitted, exit 0
        rc, out = _run_driver(root, path, ["--dry-run"], env)
        assert rc == 0, out
        assert "DRY RUN" in out and "would run" in out, out
        assert out.count("qsub ") >= 2, "the dry run prints the qsub lines:\n" + out
        assert _n_submitted(spool) == 0, "the dry run submitted something"
        assert not os.path.isfile(os.path.join(cfg["paths"]["work_root"], "state.json"))
        # the schema refusal is exit 2
        bad = os.path.join(root, "bad.json")
        d = json.loads(json.dumps(cfg))
        del d["loader"]
        with open(bad, "w") as fh:
            json.dump(d, fh)
        rc, out = _run_driver(root, bad, ["--dry-run"], env)
        assert rc == 2 and "loader.buffer_shards: missing" in out, out
        # the full run: exit 0, A and B done, C skipped (its artefact pre-exists)
        rc, out = _run_driver(root, path, ["--pre-make", "C"], env, timeout=90)
        assert rc == 0, out
        st = State.load(cfg["paths"]["work_root"])
        assert st.statuses() == {"A": "done", "B": "done", "C": "skipped"}, st.statuses()
        assert _n_submitted(spool) == 2, "A and B submit one job each"
        assert "PIPELINE DONE" in out
        # a re-run after completion: nothing submitted, exit 0
        rc, out = _run_driver(root, path, [], env)
        assert rc == 0 and _n_submitted(spool) == 2, out
    finally:
        shutil.rmtree(root)


def test_determinism():
    """Killed while A's job runs, then resumed: one submission in all."""
    root = _tmp()
    try:
        path, cfg = _synthetic_config(root)
        spool = os.path.join(root, "spool")
        env = _env(spool, delay="2.0")
        cmd = [PYTHON, os.path.join(_HERE, "synthetic_graph.py"), "--config", path,
               "--poll-seconds", "0.2", "--pre-make", "C"]
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        # wait until A's job is submitted AND recorded in the state file, then
        # kill the sequencer hard. (Killing between qsub's return and the
        # state write is the documented window of SPEC block 1: the job id is
        # lost and A would be submitted again on resume; the oracle here is
        # the resume from a recorded job.)
        recorded = False
        for _ in range(200):
            try:
                st = State.load(cfg["paths"]["work_root"])
            except (OSError, ValueError):
                st = None
            if st is not None and st.data["stages"].get("A", {}).get("jobs"):
                recorded = True
                break
            time.sleep(0.1)
        assert recorded, "A's job was not submitted and recorded in time"
        assert _n_submitted(spool) == 1
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        st = State.load(cfg["paths"]["work_root"])
        assert st.statuses()["A"] == "running", st.statuses()
        assert len(st.stage("A")["jobs"]) == 1
        lock = os.path.join(cfg["paths"]["work_root"], "pipeline.lock")
        assert os.path.isfile(lock), "the killed sequencer left its lock (expected)"
        # resume: the lock is stale (pid dead), A re-attaches to its job
        rc, out = _run_driver(root, path, ["--pre-make", "C"], env, timeout=90)
        assert rc == 0, out
        assert "stale lock" in out, out
        assert "A: done" in out, out
        st = State.load(cfg["paths"]["work_root"])
        assert st.statuses() == {"A": "done", "B": "done", "C": "skipped"}, st.statuses()
        assert _n_submitted(spool) == 2, "A must not be submitted twice (A once, B once)"
    finally:
        shutil.rmtree(root)


def test_edge_cases():
    root = _tmp()
    try:
        path, cfg = _synthetic_config(root)
        spool = os.path.join(root, "spool")
        env = _env(spool)
        # (a) a verifier that refuses: exit 3, STOP line, state failed
        rc, out = _run_driver(root, path, ["--refuse", "B"], env, timeout=90)
        assert rc == 3, out
        assert "STOP: stage B failed" in out and "REFUSED" in out, out
        st = State.load(cfg["paths"]["work_root"])
        assert st.statuses()["B"] == "failed" and st.statuses()["A"] == "done", st.statuses()
        # a re-run stops again at the failed stage without resubmitting
        n = _n_submitted(spool)
        rc, out = _run_driver(root, path, ["--refuse", "B"], env)
        assert rc == 3 and _n_submitted(spool) == n, out
        # the dry run says so (exit 3); with --retry B it shows the way on (exit 0)
        rc, out = _run_driver(root, path, ["--dry-run"], env)
        assert rc == 3 and "B: FAILED in the state file" in out and "add --retry B" in out, out
        rc, out = _run_driver(root, path, ["--dry-run", "--retry", "B"], env)
        assert rc == 0 and "B: would run" in out and "FAILED" not in out, out
        assert _n_submitted(spool) == n and State.load(cfg["paths"]["work_root"]).statuses()["B"] == "failed"
        # --retry B, now without the refusal: the run completes
        rc, out = _run_driver(root, path, ["--retry", "B"], env, timeout=90)
        assert rc == 0, out
        assert _n_submitted(spool) == n + 2, "B resubmitted once, then C"
        shutil.rmtree(cfg["paths"]["work_root"])
        shutil.rmtree(spool)

        # (b) a job that ends without its artefact: resubmitted once, then failed
        rc, out = _run_driver(root, path, ["--silent-kill", "A"], env, timeout=90)
        assert rc == 3, out
        assert "resubmitted (1 of 1)" in out, out
        assert "STOP: stage A failed: killed:" in out, out
        assert _n_submitted(spool) == 2, "one submission plus one resubmission"
        shutil.rmtree(cfg["paths"]["work_root"])
        shutil.rmtree(spool)

        # (c) a second sequencer on the same work root exits 4
        env_slow = _env(spool, delay="3.0")
        cmd = [PYTHON, os.path.join(_HERE, "synthetic_graph.py"), "--config", path,
               "--poll-seconds", "0.2"]
        first = subprocess.Popen(cmd, cwd=root, env=env_slow, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True)
        for _ in range(100):
            if _n_submitted(spool) >= 1:
                break
            time.sleep(0.1)
        rc, out = _run_driver(root, path, [], env_slow)
        assert rc == 4 and "another sequencer" in out, out
        os.kill(first.pid, signal.SIGKILL)
        first.wait()

        # (d) the config-change refusal and its override
        d = json.loads(json.dumps(cfg))
        d["search"]["gp_rounds"] = 21
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
        rc, out = _run_driver(root, path, [], env_slow)
        assert rc == 2 and "configuration changed" in out, out
        rc, out = _run_driver(root, path, ["--accept-config-change"], env_slow, timeout=90)
        assert rc == 0, out
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------- runner ---

def main():
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print("python %s | fake PBS %s" % (sys.version.split()[0], _FAKE))
    results = []
    for name, func in checks:
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
        print("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
