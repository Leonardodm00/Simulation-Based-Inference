#!/usr/bin/env python3
"""Smoke test for `tools/search_reset.py` (specs/SPEC.md block 7, amended
2026-10-19, the restart of a search from nothing): the list-then-confirm
tool that sets a search's rounds aside and deletes its live trial jobs.

A campaign folder in the shape of a round under way (the fixture of
`test_smoke_report.py`: 9 trials -- ok, failed, unreadable, running,
waiting -- with their records, pending specs, run folders and logs), a
budget gate folder to keep, and under the fake PBS two trial jobs still
running (the state's search stage records them) and one finished.

Checks
    test_known_answer   the listing: the seven items (five present), the
                        gate kept, the ledger's counts, the two live jobs,
                        the CODE line; nothing moved, nothing deleted
    test_reference      the CODE is a function of the listing: the same
                        twice, another once a file or a job changes
    test_convergence    not a discretisation: skipped
    test_invariants     bytes: every file of the round is in the set-aside
                        folder with the same sha256; the gate's files
                        untouched; nothing written into the campaign folder
                        but the empty place where the round was
    test_contract       --confirm with the right CODE: the two jobs
                        deleted (F in the fake PBS), the round set aside,
                        RESET_RECORD.json, the pass line; then the
                        pipeline's dry run with --retry search
                        --accept-config-change is accepted (exit 0, no
                        REFUSE); a second listing finds nothing to set aside
    test_edge_cases     a wrong CODE: refused, nothing done; a sequencer
                        job queued: refused; a configuration that does not
                        load: exit 2
    MUTATION=code       (environment) the CODE made constant, equal to the
                        wrong CODE the edge case offers: test_edge_cases
                        (a wrong CODE acts) and test_reference must FAIL

Run
    python3 hpc/pipeline/tests/test_smoke_reset.py
        About 20 s; stdlib and the fake PBS.

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
_FAKE = os.path.join(_HERE, "fake_pbs")
for _p in (_PIPE, _TOOLS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MUTATION = os.environ.pop("MUTATION", "")      # the report test reads its own MUTATION

import config as C                  # noqa: E402
import pbs as PBS                   # noqa: E402
import state as ST                  # noqa: E402
import search_reset as SR           # noqa: E402
import test_smoke_report as TR      # noqa: E402

PYTHON = sys.executable
TOOL = os.path.join(_TOOLS, "search_reset.py")

if MUTATION == "code":
    SR.code_of = lambda lst: "deadbeef"
elif MUTATION:
    raise SystemExit("unknown MUTATION %r; known: code" % MUTATION)


# --------------------------------------------------------------- helpers ---

def _env(root):
    env = dict(os.environ)
    env["PATH"] = _FAKE + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = os.path.join(root, "spool")
    env["FAKE_PBS_DELAY"] = "0.2"
    env["FAKE_PBS_QSELECT_SUFFIX"] = ".cluster"
    env["PIPELINE_SKIP_CONDA"] = "1"
    env["PIPELINE_PYTHON"] = PYTHON
    for k in ("PBS_JOBID", "MUTATION"):
        env.pop(k, None)
    return env


def _qsub(env, name, seconds, root):
    script = os.path.join(root, "%s.sh" % name)
    with open(script, "w") as fh:
        fh.write("#!/bin/bash\nsleep %s\necho done\n" % seconds)
    proc = subprocess.run(["qsub", "-N", name, "-q", "cpu", "-l", "select=1:ncpus=1:mem=1gb",
                           "-l", "walltime=00:10:00", "-o", os.path.join(root, name + ".log"),
                           script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split()[0]


def _fixture(root):
    """The report test's campaign and state, plus a gate folder and three
    fake trial jobs recorded in the state: two long, one already done."""
    path, ids, camp = TR._campaign(root)
    gate = os.path.join(camp, "gate", "b16")
    os.makedirs(gate)
    with open(os.path.join(gate, "A1_seed0_probe.json"), "w") as fh:
        json.dump({"projected_trial_seconds": 1.0}, fh)
    with open(os.path.join(camp, "bank_split.json"), "w") as fh:
        json.dump({"split_hash": "s"}, fh)
    env = _env(root)
    cfg = C.load_config(path)
    wr = cfg.data["paths"]["work_root"]
    long1 = _qsub(env, "t_" + ids["run"][:12], 60, root)
    long2 = _qsub(env, "t_" + ids["empty"][:12], 60, root)
    short = _qsub(env, "t_" + ids["ok"][:12], 0, root)
    st = ST.State.load(wr)
    jobs = st.data["stages"]["search"]["jobs"]
    for j, fake in ((long1, ids["run"]), (long2, ids["empty"]), (short, ids["ok"])):
        jobs.append({"id": j, "role": "trial %s" % fake[:12], "finished": False})
    st.save()
    time.sleep(1.5)                      # the short one ends, the long ones run
    return path, ids, camp, env, (long1, long2, short)


def _run(args, env):
    """The tool in this process (so a MUTATION reaches it), with the fake
    PBS on PATH; (exit code, output)."""
    buf = io.StringIO()

    def go():
        with contextlib.redirect_stdout(buf):
            return SR.main(args)
    rc = _with_env(env, go)
    return rc, buf.getvalue()


def _run_cli(args, env):
    """The tool as the command line runs it."""
    proc = subprocess.run([PYTHON, TOOL] + args, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=300)
    return proc.returncode, proc.stdout


def _code(out):
    for ln in out.splitlines():
        if ln.startswith("CODE "):
            return ln.split()[1]
    raise AssertionError("no CODE line in:\n" + out)


def _digests(top):
    out = {}
    for d, _subs, files in os.walk(top):
        for f in files:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, top)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _fake_state(env, job_id):
    proc = subprocess.run(["qstat", "-x", "-f", job_id], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    return PBS.parse_qstat(job_id, proc.stdout).state


def _with_env(env, fn):
    """Run fn() with the fake PBS on PATH for this process (the tool's
    functions called in-process use pbs.py's qstat/qdel/qselect)."""
    old = dict(os.environ)
    os.environ.update(env)
    try:
        return fn()
    finally:
        os.environ.clear()
        os.environ.update(old)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    root = tempfile.mkdtemp(prefix="reset_ka_")
    try:
        path, ids, camp, env, (l1, l2, short) = _fixture(root)
        before = _digests(camp)
        rc, out = _run_cli(["--config", path], env)
        assert rc == 0, out
        assert "SEARCH RESET (nothing done yet): campaign folder %s" % camp in out, out
        for name in ("trials", "pending", "runs", "jobs"):
            assert "  would set aside %-24s" % name in out, (name, out)
        assert "pipeline_rounds.json" not in out.split("kept in place")[0], out   # absent
        assert "  kept in place: gate, bank_split.json" in out, out
        assert "  the ledger: failed 4, ok 1, unreadable 2" in out, out
        assert "  would DELETE job %s (trial %s, state R)" % (l1, ids["run"][:12]) in out, out
        assert "  would DELETE job %s (trial %s, state R)" % (l2, ids["empty"][:12]) in out, out
        assert short not in out, "the finished job is not a job to delete"
        assert "the state file is not touched" in out
        code = _code(out)
        assert len(code) == 8 and int(code, 16) >= 0
        assert _digests(camp) == before, "the listing moved something"
        assert _fake_state(env, l1) == "R" and _fake_state(env, l2) == "R"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reference():
    root = tempfile.mkdtemp(prefix="reset_ref_")
    try:
        path, ids, camp, env, (l1, l2, short) = _fixture(root)
        cfg = C.load_config(path)
        lst1 = _with_env(env, lambda: SR.listing(cfg))
        lst2 = _with_env(env, lambda: SR.listing(cfg))
        assert SR.code_of(lst1) == SR.code_of(lst2)
        c0 = SR.code_of(lst1)
        with open(os.path.join(camp, "trials", "extra.json"), "w") as fh:
            fh.write("{}")
        c1 = SR.code_of(_with_env(env, lambda: SR.listing(cfg)))
        assert c1 != c0, "a new record must change the CODE"
        subprocess.run(["qdel", l2], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end = time.time() + 20
        while time.time() < end and _fake_state(env, l2) != "F":
            time.sleep(0.5)
        assert _fake_state(env, l2) == "F", "the fake PBS did not end the deleted job"
        c2 = SR.code_of(_with_env(env, lambda: SR.listing(cfg)))
        assert c2 != c1, "a job that ended must change the CODE"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("a reset has no discretisation parameter")


def test_invariants():
    """Bytes are preserved: the round's files reappear unchanged in the
    set-aside folder; the gate stays; nothing else is written in place."""
    root = tempfile.mkdtemp(prefix="reset_inv_")
    try:
        path, ids, camp, env, (l1, l2, short) = _fixture(root)
        before = _digests(camp)
        rc, out = _run(["--config", path], env)
        code = _code(out)
        rc, out = _run(["--config", path, "--confirm", code], env)
        assert rc == 0, out
        aside = [d for d in os.listdir(os.path.dirname(camp)) if d.startswith("S-A1_set_aside_")]
        assert len(aside) == 1, aside
        aside = os.path.join(os.path.dirname(camp), aside[0])
        after_aside = _digests(aside)
        after_aside.pop("RESET_RECORD.json")
        moved = dict((k, v) for k, v in before.items()
                     if k.split(os.sep)[0] in ("trials", "pending", "runs", "jobs"))
        assert after_aside == moved, (sorted(after_aside), sorted(moved))
        left = _digests(camp)
        kept = dict((k, v) for k, v in before.items()
                    if k.split(os.sep)[0] in ("gate", "bank_split.json"))
        assert left == kept, (sorted(left), sorted(kept))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_contract():
    root = tempfile.mkdtemp(prefix="reset_con_")
    try:
        path, ids, camp, env, (l1, l2, short) = _fixture(root)
        rc, out = _run(["--config", path], env)
        code = _code(out)
        rc, out = _run(["--config", path, "--confirm", code], env)
        assert rc == 0, out
        assert "deleted job %s (trial %s)" % (l1, ids["run"][:12]) in out and \
            "deleted job %s (trial %s)" % (l2, ids["empty"][:12]) in out, out
        assert _fake_state(env, l1) == "F" and _fake_state(env, l2) == "F"
        assert "SEARCH RESET DONE: 2 job(s) deleted, round(s) set aside as " in out and \
            "; next: submit with --retry search --accept-config-change" in out, out
        aside = [d for d in os.listdir(os.path.dirname(camp)) if d.startswith("S-A1_set_aside_")]
        rec = json.load(open(os.path.join(os.path.dirname(camp), aside[0], "RESET_RECORD.json")))
        assert sorted(rec["jobs_deleted"]) == sorted([l1, l2]) and rec["code"] == code and \
            sorted(rec["moved"]) == ["jobs", "pending", "runs", "trials"] and \
            rec["ledger_before"]["ok"] == 1, rec
        for name in ("trials", "pending", "runs", "jobs"):
            assert not os.path.exists(os.path.join(camp, name)), name
        assert os.path.isfile(os.path.join(camp, "gate", "b16", "A1_seed0_probe.json"))
        # the pipeline's own path to the restart: --retry search is accepted now
        proc = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", path,
                               "--dry-run", "--retry", "search", "--accept-config-change"], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=300)
        assert "REFUSE" not in proc.stdout and "search reset to pending (--retry)" in proc.stdout, \
            proc.stdout[-3000:]
        # a second listing: nothing to set aside, no job to delete
        rc, out = _run(["--config", path], env)
        assert rc == 0 and "nothing to set aside" in out and \
            "no trial job queued or running" in out, out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_edge_cases():
    root = tempfile.mkdtemp(prefix="reset_edge_")
    try:
        path, ids, camp, env, (l1, l2, short) = _fixture(root)
        before = _digests(camp)
        # (a) a wrong CODE: refused, nothing moved, the jobs still run
        rc, out = _run(["--config", path, "--confirm", "deadbeef"], env)
        assert rc == 4 and "SEARCH RESET REFUSED: the CODE deadbeef does not match" in out, out
        assert _digests(camp) == before and _fake_state(env, l1) == "R", "a wrong CODE acted"
        # (b) a sequencer job queued: refused before anything
        drv = _qsub(env, "pl_giulia", 30, root)
        rc, out = _run(["--config", path], env)
        assert rc == 4 and \
            "SEARCH RESET REFUSED: the sequencer is queued or running (%s" % drv in out, out
        subprocess.run(["qdel", drv], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # (c) a configuration that does not load
        bad = os.path.join(root, "bad.json")
        with open(bad, "w") as fh:
            fh.write("{}")
        rc, out = _run(["--config", bad], env)
        assert rc == 2 and out.startswith("SEARCH RESET: the configuration "), out
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- runner ---

def main():
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print("python %s | fake PBS %s" % (sys.version.split()[0], _FAKE))
    if MUTATION:
        print("MUTATION=%s: the named piece is disabled; test_edge_cases and test_reference must "
              "FAIL" % MUTATION)
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
