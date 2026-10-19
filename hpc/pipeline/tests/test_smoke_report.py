#!/usr/bin/env python3
"""Smoke test for `tools/search_report.py` (specs/SPEC.md block 7, an
unreadable record, amended 2026-10-19): the read-only report of the
search's trials that is run on davinci before a submission.

A hand-made campaign folder and state file (a copy of the Giulia
configuration with every path under a temporary folder) holds one trial of
each kind the report must tell apart:

    ok        its job ended, an ok record (the pending spec removed)
    sigbus    recorded failed, `runner exited -7`, never examined
    value1/2  recorded failed of a ValueError (two tensor shapes), never
              examined, their logs complete
    oom       recorded failed, `runner exited -9`, already examined
    empty     its job ended, a record of 0 bytes, the pending spec there
    nulnopend its job ended by its log only, a record of NUL bytes, no
              pending spec
    run       its job running (no RUN_ARGV EXIT line yet)
    wait      waiting, --resume next

Checks
    test_known_answer   every trial's `next:` and record text by hand; the
                        counts; the failures grouped (value1 and value2 one
                        group); the pass line
    test_reference      failure_cause and next_action on hand-built facts
    test_convergence    not a discretisation: skipped
    test_invariants     the CLI writes nothing: every file's bytes and
                        mtime unchanged, no file added
    test_contract       --trial prints one trial with its log's last lines;
                        --qstat asks the scheduler once for every last job
                        (a fake qstat): a job it reports running is not taken
                        as ended, a failed job's node and memory are shown;
                        why the failed records failed (a traceback tail per
                        cause) and the knobs, failed against ok and open
    test_edge_cases     no state file: the ledger's trials, status `?`; a
                        configuration that does not load: exit 2
    MUTATION=classify   (environment) failure_cause disabled: the known
                        answer must FAIL

Run
    python3 hpc/pipeline/tests/test_smoke_report.py
        A few seconds; stdlib only (runs on davinci as in the sandbox).

Pure ASCII, LF only; stdlib only.
"""

import contextlib
import hashlib
import io
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
_TOOLS = os.path.join(_PIPE, "tools")
for _p in (_PIPE, _TOOLS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C              # noqa: E402
import state as ST              # noqa: E402
import stages.common as CM      # noqa: E402
import search_report as SR      # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
TOOL = os.path.join(_TOOLS, "search_report.py")

MUTATION = os.environ.get("MUTATION", "")
if MUTATION == "classify":
    SR.failure_cause = lambda *a, **k: "no error text"
elif MUTATION:
    raise SystemExit("unknown MUTATION %r; known: classify" % MUTATION)

VALUE_LOG = ("Traceback (most recent call last):\n"
             "  File \"run_joint_arms.py\", line 812, in <module>\n"
             "ValueError: Expected parameter loc (Tensor of shape (%d, 23)) to satisfy the "
             "constraint Real(), but found invalid values:\n"
             "tensor([[nan, nan]])\n"
             "[evaluate] FAILED, recorded as status=failed\n"
             "RUN_ARGV EXIT: 1\n")


# each trial's configuration: the failed ones at a high learning rate, the
# others low; block_family the same in all
CONF = dict((n, {"lr": lr, "block_family": 1}) for n, lr in (
    ("ok", 0.0005), ("sigbus", 0.003), ("value1", 0.004), ("value2", 0.005), ("oom", 0.006),
    ("empty", 0.0007), ("nulnopend", 0.002), ("run", 0.0008), ("wait", 0.0009)))

FAKE_QSTAT = """#!/bin/sh
# qstat -x -f <id> ...: the run trial's job runs; the oom trial's ended near its memory
shift 2
for id in "$@"; do
  echo "Job Id: $id"
  case "$id" in
    1851307.pbsserver01)
      echo "    job_state = R"
      echo "    exec_host = dvnode007/0*48";;
    1851304.pbsserver01)
      echo "    job_state = F"
      echo "    Exit_status = 1"
      echo "    exec_host = dvnode004/0*48"
      echo "    resources_used.mem = 167000000kb"
      echo "    resources_used.walltime = 05:10:00"
      echo "    Resource_List.mem = 160gb";;
    *)
      echo "    job_state = F"
      echo "    Exit_status = 0"
      echo "    exec_host = dvnode001/0*48"
      echo "    resources_used.mem = 78400000kb"
      echo "    resources_used.walltime = 02:00:00"
      echo "    Resource_List.mem = 160gb";;
  esac
done
"""


def _tid(name):
    return hashlib.sha256(name.encode("ascii")).hexdigest()


def _project(root):
    """The committed Giulia file with every path under `root`."""
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


def _campaign(root, with_state=True):
    """The folder and the state of the module docstring; returns
    (config path, {name: trial id}, campaign folder)."""
    path = _project(root)
    cfg = C.load_config(path)
    camp = os.path.join(cfg.data["paths"]["tune_root"], cfg.data["search"]["campaign"])
    d = dict((k, os.path.join(camp, k)) for k in ("trials", "pending", "runs", "jobs"))
    for p in d.values():
        os.makedirs(p)
    ids = dict((n, _tid(n)) for n in ("ok", "sigbus", "value1", "value2", "oom", "empty",
                                      "nulnopend", "run", "wait"))

    def rec(name, raw):
        with open(os.path.join(d["trials"], ids[name] + ".json"), "wb") as fh:
            fh.write(raw)

    def obj(name, **kw):
        r = {"trial_id": ids[name], "config": CONF[name], "status": "ok", "error": ""}
        r.update(kw)
        rec(name, json.dumps(r).encode("ascii"))

    def pend(name):
        with open(os.path.join(d["pending"], ids[name] + ".json"), "w") as fh:
            json.dump({"trial_id": ids[name], "config": CONF[name]}, fh)

    def log(name, host, body, n=0):
        with open(os.path.join(d["jobs"], "%s.%d.log" % (ids[name], n)), "w") as fh:
            fh.write("RUN_ARGV: host %s, job %s.pbs, ncpus 48, env sbi_export\n%s"
                     % (host, name, body))

    obj("ok", nll=0.25)
    log("ok", "dvnode001", "[evaluate] nll=0.25\nRUN_ARGV EXIT: 0\n")
    obj("sigbus", status="failed", error="runner exited -7")
    pend("sigbus")
    log("sigbus", "dvnode012", "[evaluate] FAILED, recorded as status=failed\n")
    for n, shape in (("value1", 16), ("value2", 32)):
        obj(n, status="failed", error="runner exited 1")
        pend(n)
        log(n, "dvnode00%d" % (2 if n == "value1" else 3), VALUE_LOG % shape)
    obj("oom", status="failed", error="runner exited -9")
    log("oom", "dvnode004", "Killed\n[evaluate] FAILED, recorded as status=failed\nRUN_ARGV EXIT: 1\n")
    rec("empty", b"")
    pend("empty")
    log("empty", "dvnode005", "fault: the node went down\nRUN_ARGV EXIT: 137\n")
    os.makedirs(os.path.join(d["runs"], ids["empty"]))
    with open(os.path.join(d["runs"], ids["empty"], "A1_seed0_train_ckpt.pt"), "wb") as fh:
        fh.write(b"PK")
    rec("nulnopend", b"\0" * 7)
    log("nulnopend", "dvnode006", "RUN_ARGV EXIT: 0\n")
    pend("run")
    log("run", "dvnode007", "[epoch 3] ...\n")
    pend("wait")
    with open(os.path.join(d["trials"], ids["oom"] + ".json.nodefault1"), "w") as fh:
        fh.write("{}")
    if with_state:
        st = ST.State.new(cfg.data["paths"]["work_root"], "giulia", cfg.path, cfg.sha256)
        trials = {}
        jobs = []
        for i, (n, status, extra) in enumerate((
                ("ok", "running", {}), ("sigbus", "failed", {}), ("value1", "failed", {}),
                ("value2", "failed", {}), ("oom", "failed", {"fault_checked": True}),
                ("empty", "running", {}), ("nulnopend", "running", {}), ("run", "running", {}),
                ("wait", "waiting", {"resume_next": True, "resubmits": 1}))):
            jid = "%d.pbsserver01" % (1851300 + i)
            tr = {"round": 0, "jobs": [jid] if n != "wait" else [], "resubmits": 0,
                  "status": status}
            tr.update(extra)
            trials[ids[n]] = tr
            if n != "wait":
                jobs.append({"id": jid, "role": "trial %s" % ids[n][:12],
                             "finished": n in ("ok", "sigbus", "value1", "value2", "oom", "empty"),
                             "exit_status": ({"ok": 0, "empty": 137}.get(n, 1)
                                             if n not in ("nulnopend", "run") else None),
                             "walltime_s": 7200.0})
        st.data["stages"]["search"] = {"status": "running", "jobs": jobs, "resubmits": 0,
                                       "note": "", "started_utc": None, "finished_utc": None,
                                       "sub": {"phase": "rounds", "round": 0, "trials": trials,
                                               "rounds": [{"round": 0,
                                                           "trial_ids": sorted(trials)}]}}
        st.save()
    return path, ids, camp


def _report(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = SR.main(args)
    return rc, buf.getvalue()


def _line(out, tid):
    found = [ln for ln in out.splitlines() if ln.startswith("trial %s " % tid[:12])]
    assert len(found) == 1, (tid[:12], out)
    return found[0]


def _snapshot(root):
    out = {}
    for d, _subs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            stt = os.stat(p)
            with open(p, "rb") as fh:
                out[p] = (stt.st_size, stt.st_mtime_ns, hashlib.sha256(fh.read()).hexdigest())
    return out


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    root = tempfile.mkdtemp(prefix="report_ka_")
    try:
        path, ids, camp = _campaign(root)
        rc, out = _report(["--config", path])
        assert rc == 0, out
        want = {
            "ok": ("record ok nll 0.250000", "next: marked ok nll 0.250000"),
            "sigbus": ("record failed nll", None),
            "value1": ("record failed error 'runner exited 1'",
                       "next: final failure: not a node fault"),
            "oom": ("cause SIGKILL: memory (the OOM killer) or an operator", "next: final (failed)"),
            "empty": ("record UNREADABLE (JSONDecodeError: Expecting value: line 1 column 1 "
                      "(char 0) (0 bytes))",
                      "next: its record cannot be read: read again for 300 s, then set aside as "
                      ".unreadable1; then resubmitted with --resume (1 of 2) when the cores allow"),
            "nulnopend": ("record UNREADABLE (JSONDecodeError: Expecting value: line 1 column 1 "
                          "(char 0) (7 bytes, all NUL))",
                          "next: its record cannot be read: read again for 300 s, then set aside "
                          "as .unreadable1; then recorded killed (no pending spec to resubmit "
                          "from)"),
            "run": ("record none", "next: nothing until its job 1851307.pbsserver01 ends (not "
                                   "ended by the state file, no RUN_ARGV EXIT in its log"),
            "wait": ("record none", "next: waits for a slot and the cores (then --resume)"),
        }
        for name, (a, b) in want.items():
            ln = _line(out, ids[name])
            if a == "record failed nll":
                assert "record failed error 'runner exited -7'" in ln and \
                    "cause node fault: SIGBUS: runner exited -7" in ln and \
                    "next: NODE FAULT: record set aside, queued again with --resume (1 of 2)" \
                    in ln, ln
                continue
            assert a in ln, (name, a, ln)
            if b:
                assert b in ln, (name, b, ln)
        assert "cause ValueError: Expected parameter loc (Tensor of shape (16, 23)) to satisfy" \
            in _line(out, ids["value1"]), _line(out, ids["value1"])
        assert "log dvnode002 exit 1: ValueError: Expected parameter loc" in _line(out, ids["value1"])
        assert "aside nodefault1" in _line(out, ids["oom"])
        assert "ckpt yes, runner record no" in _line(out, ids["empty"])
        assert "pending no" in _line(out, ids["nulnopend"]) and \
            "pending yes" in _line(out, ids["empty"])
        assert "trials: 9 in 1 round(s); by the state (state updated " in out
        assert "running 4, waiting 1, failed 4" in out, out
        assert "ledger records: 5 readable (failed 4, ok 1), 2 UNREADABLE: " in out, out
        assert "set aside earlier: 1" in out
        assert "failed records by cause (a heuristic label; the stage heals only node faults): 4" in out
        grp = [ln for ln in out.splitlines() if ln.startswith("   2 x ValueError: Expected "
                                                              "parameter loc (Tensor of shape (")]
        assert len(grp) == 1 and "-- hosts dvnode002,dvnode003 -- " in grp[0] and \
            ids["value1"][:12] in grp[0] and ids["value2"][:12] in grp[0], out
        assert "   1 x node fault: SIGBUS: runner exited -7 -- hosts dvnode012" in out, out
        assert out.rstrip().endswith("SEARCH REPORT OK: 9 trial(s) read; nothing was written"), out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reference():
    assert SR.failure_cause("runner exited -7", None, "SIGBUS: runner exited -7") == \
        "node fault: SIGBUS: runner exited -7"
    assert SR.failure_cause("runner exited -9", "Killed\n", None) == \
        "SIGKILL: memory (the OOM killer) or an operator"
    assert SR.failure_cause("runner exited -11", "", None) == "signal 11"
    assert SR.failure_cause("runner exited 1", "=>> PBS: job killed: walltime 432010 exceeded "
                                               "limit 432000\n", None) == "walltime exceeded"
    assert SR.failure_cause("runner exited 1", "x\nRuntimeError: CUDA? no: a bad shape\n", None) \
        == "RuntimeError: CUDA? no: a bad shape"
    assert SR.failure_cause("runner wrote no record in /x", "", None) == "runner wrote no record in /x"
    assert SR.failure_cause(None, None, None) == "no error text"
    assert SR.cause_key("ValueError: shape (16, 23)") == SR.cause_key("ValueError: shape (32, 23)")
    assert SR.last_error_line("a\nOSError: [Errno 5] Input/output error\n[evaluate] FAILED\n"
                              "RUN_ARGV EXIT: 1\n") == "OSError: [Errno 5] Input/output error"
    root = tempfile.mkdtemp(prefix="report_ref_")
    try:
        cfg = C.load_config(_project(root))
        base = {"status": "running", "fault_checked": False, "resume_next": False, "resubmits": 0,
                "qstat": None, "job_finished": True, "log_exit": 0, "last_job": "1.x",
                "record_exists": False, "record": None, "pending": True, "node_fault": None,
                "unreadable_before": 0}

        def act(**kw):
            f = dict(base)
            f.update(kw)
            return SR.next_action(cfg, f)
        assert act() == "ended without a record: resubmitted with --resume (1 of 2) when the " \
                        "cores allow"
        assert act(resubmits=2) == "ended without a record: recorded killed (no resubmission " \
                                   "left after 2)"
        assert act(pending=False) == "ended without a record: recorded killed (no pending spec " \
                                     "to resubmit from)"
        assert act(job_finished=False, log_exit=None, qstat=("R", None)) == \
            "nothing until its job 1.x ends (the scheduler: R)"
        assert act(job_finished=False, log_exit=None, qstat=("unknown", None)).startswith(
            "ended without a record"), "a job the scheduler forgot has ended, as the stage reads it"
        assert act(status="examining", record_exists=True, unreadable_before=2) == \
            "its record cannot be read: read again for 300 s, then set aside as .unreadable3; " \
            "then queued again with --resume (1 of 2) when the cores allow"
        assert act(status="failed", record_exists=True, record={"status": "failed"},
                   log_exit=None) == "examined: waits up to 300 s for its log's RUN_ARGV EXIT " \
                                     "line, then final unless that log shows a node fault"
        assert act(status="failed", record_exists=True, record={"status": "failed"},
                   node_fault="x", pending=False) == \
            "NODE FAULT, but no configuration and no pending spec: final"
        assert act(status="failed", record_exists=True, record={"status": "failed", "config": {}},
                   node_fault="x", resubmits=2, pending=False) == \
            "NODE FAULT, but no configuration and no pending spec: final"
        assert act(status="failed", record_exists=True,
                   record={"status": "failed", "config": {"a": 1}}, node_fault="x",
                   resubmits=2) == "NODE FAULT: record set aside, no resubmission left: recorded " \
                                   "killed"
        assert act(status="ok") == "final (ok)" and act(status="killed") == "final (killed)"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("a report has no discretisation parameter")


def test_invariants():
    """The CLI, as on davinci: nothing written, nothing added."""
    root = tempfile.mkdtemp(prefix="report_inv_")
    try:
        path, ids, camp = _campaign(root)
        before = _snapshot(root)
        time.sleep(0.05)
        env = dict(os.environ)
        env["PIPELINE_QSTAT"] = os.path.join(root, "no-such-qstat")
        for extra in ([], ["--trial", ids["empty"][:8]], ["--qstat"]):
            proc = subprocess.run([sys.executable, TOOL, "--config", path] + extra, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  universal_newlines=True, timeout=120)
            assert proc.returncode == 0 and "SEARCH REPORT OK: " in proc.stdout, proc.stdout
        after = _snapshot(root)
        assert sorted(after) == sorted(before), set(after) ^ set(before)
        changed = [p for p in before if before[p] != after[p]]
        assert not changed, changed
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_contract():
    root = tempfile.mkdtemp(prefix="report_con_")
    try:
        path, ids, camp = _campaign(root)
        rc, out = _report(["--config", path, "--trial", ids["value2"][:10]])
        assert rc == 0 and "== trial %s" % ids["value2"] in out, out
        assert "   | ValueError: Expected parameter loc (Tensor of shape (32, 23))" in out, out
        assert "job logs: %s.0.log" % ids["value2"] in out, out
        assert "SEARCH REPORT OK: 1 trial(s) read; nothing was written" in out
        # --qstat: a fake qstat says the running trial's job runs, the NUL trial's has ended
        fake = os.path.join(root, "qstat")
        with open(fake, "w") as fh:
            fh.write(FAKE_QSTAT)
        os.chmod(fake, 0o755)
        old = SR.PBS.QSTAT
        SR.PBS.QSTAT = fake
        try:
            rc, out = _report(["--config", path, "--qstat"])
        finally:
            SR.PBS.QSTAT = old
        assert rc == 0, out
        assert "next: nothing until its job 1851307.pbsserver01 ends (the scheduler: R)" in \
            _line(out, ids["run"]), _line(out, ids["run"])
        assert "qstat F" in _line(out, ids["nulnopend"]) and \
            "then recorded killed (no pending spec" in _line(out, ids["nulnopend"])
        assert "qstat: 8 of 8 last job(s) known to the scheduler" in out, out
        assert "qstat F on dvnode004 exit 1, 05:10:00, mem 159.3GB of 160gb" in \
            _line(out, ids["oom"]), _line(out, ids["oom"])
        assert "qstat F on dvnode001 exit 0, 02:00:00, mem 74.8GB of 160gb" in \
            _line(out, ids["value1"]), _line(out, ids["value1"])
        # why: one traceback tail per cause, the largest group first
        lines = out.splitlines()
        i = lines.index("why the failed records failed -- per cause, one trial's log (--trial "
                        "<id> shows any one in full):")
        assert lines[i + 1].startswith("  -- 2 x ValueError: Expected parameter loc (Tensor of "
                                       "shape ("), lines[i + 1]
        assert "(record 'runner exited 1'; job exit 0, 02:00:00, mem 74.8GB)" in lines[i + 1]
        assert lines[i + 2] == "  | Traceback (most recent call last):", lines[i + 2]
        assert "  | tensor([[nan, nan]])" in lines[i + 2:i + 7], lines[i:i + 8]
        oom = [ln for ln in lines if ln.startswith("  -- 1 x SIGKILL: memory")]
        assert len(oom) == 1 and "on dvnode004 (record 'runner exited -9'; job exit 1, 05:10:00, " \
            "mem 159.3GB)" in oom[0], oom
        assert lines[lines.index(oom[0]) + 1] == "  | Killed", lines[lines.index(oom[0]):][:3]
        # the knobs: the failed records against the ok ones and the open trials
        assert "the knobs: 4 failed record(s) against 1 ok and 3 open (pending specs)" in out, out
        lr = [ln for ln in lines if ln.startswith("  lr ")]
        assert lr == ["  lr                 failed 0.003, 0.004, 0.005, 0.006 | ok min 0.0005 "
                      "median 0.0005 max 0.0005 (1) | open min 0.0007 median 0.0008 max 0.0009 "
                      "(3)"], lr
        assert "  the same in all: block_family=1" in lines, out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_edge_cases():
    root = tempfile.mkdtemp(prefix="report_edge_")
    try:
        path, ids, camp = _campaign(root, with_state=False)
        rc, out = _report(["--config", path])
        assert rc == 0 and "state: none under " in out, out
        n = len([ln for ln in out.splitlines() if ln.startswith("trial ")])
        assert n == 9, out           # 7 records and 2 pending specs of their own
        assert "next: unknown to the state file" in _line(out, ids["run"])
        assert "SEARCH REPORT OK: 9 trial(s) read" in out
        bad = os.path.join(root, "bad.json")
        with open(bad, "w") as fh:
            fh.write("{}")
        rc, out = _report(["--config", bad])
        assert rc == 2 and out.startswith("SEARCH REPORT: the configuration "), out
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- runner ---

def main():
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print("python %s" % sys.version.split()[0])
    if MUTATION:
        print("MUTATION=%s: the named piece is disabled; test_known_answer must FAIL" % MUTATION)
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max(len(r[0]) for r in results)
    for name, status, seconds, detail in results:
        head = (detail.strip().splitlines() or [""])[0]
        print("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, head[:200]))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()[:6000]))
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "SKIP"))
    print("\n-- %d pass, %d fail, %d error, %d skip"
          % (counts["PASS"], counts["FAIL"], counts["ERROR"], counts["SKIP"]))
    return 1 if counts["FAIL"] or counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
