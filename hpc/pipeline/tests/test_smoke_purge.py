#!/usr/bin/env python3
"""Smoke test for `tools/purge_outputs.py` (specs/SPEC.md block 10, added
2026-11-02): the list-then-delete tool that deletes a project's search,
best fit and simulated bank.

A synthetic project (the report test's campaign and state, under the fake
PBS) with: a budget gate folder and `bank_split.json` in the campaign; a
set-aside folder `S-A1_set_aside_<utc>` and a decoy `S-A1_notes` beside
it; a best fit folder; a simulated bank (a shard, its sidecar, a task
marker, the record and the certificate with their `.sha256`); a
quarantine; and, to keep, the real bank, the bank plan's folder, another
data set's bank and the work root.

Checks
    test_known_answer   the listing: the five targets with their counts
                        (against `find`), the ledger's counts, the record's
                        sha256, the kept paths, the totals, the CODE line;
                        nothing deleted (every file's sha256 unchanged)
    test_reference      the CODE equals the SPEC's formula computed here
                        from `find`'s sizes and times; the same twice;
                        another once a file in any target changes
    test_convergence    not a discretisation: skipped
    test_invariants     after --confirm: exactly the targets gone; every
                        other file's sha256 unchanged (the state file, the
                        log, the kept banks, the plan, the decoys)
    test_contract       --confirm through the command line: the record
                        (done, every target deleted, the CODE), the DELETED
                        lines, the pass line; a second listing: every
                        target absent, nothing to delete
    test_determinism    the targets' order is the documented one whatever
                        the order the folders were made in
    test_edge_cases     refused, each with nothing deleted: a wrong CODE, a
                        lock, a sequencer job, a job the state records still
                        running, a job naming the chain (its CHAIN_NAME
                        split by the qstat wrap) -- a job of another chain
                        and a look-alike name do not refuse --, a target
                        that is a symbolic link; exit 2 for an empty or
                        unknown --what and a configuration that does not load
    MUTATION=code       (environment) the CODE made constant, equal to the
                        wrong CODE the edge case offers: test_edge_cases (a
                        wrong CODE deletes) and test_reference must FAIL
    MUTATION=chain      the chain scan off: test_edge_cases (a job naming
                        the chain does not refuse) must FAIL

Block 24 (added 2026-11-13; D-132): a reduced synthetic project
(test_smoke_synthetic._small, searches S-A1 and S-A2-R0 with the other
arm's fit) and its evaluation configuration, with its three banks, two
searches (one with a set-aside folder), three best fits, an evaluation
(samples, calbank, metrics, figures, data, jobs) and a work root (the
state file and its .prev, the log, a reports folder, an older purge
record); decoys: another data set's bank, a file beside the banks.
    test_t24a_listing   the targets in the documented order with `find`'s
                        counts, the archive's members (none under samples/
                        or calbank/) and the skipped subtrees, kept purge/;
                        the CODE is the SPEC's formula computed here, the
                        same twice, another once a result file changes;
                        nothing deleted
    test_t24b_confirm   --confirm through the command line: the archive
                        holds the listing's members with the files' bytes,
                        its .sha256 matches, every target gone, purge/ keeps
                        the older record and a new one done, the decoys and
                        the configurations unchanged, the pass line; a
                        second listing: every target absent, no archive
    test_t24c_arguments exit 2, nothing written: evaluation without
                        --archive; an archive that exists, under out_root,
                        under a target, in a missing folder; --archive
                        without evaluation, with --no-archive; block 10's
                        kinds here and these kinds on block 10's project
    test_t24d_refusals  exit 4, nothing deleted: a live job of the
                        evaluation's ledger, a wrong CODE, a symbolic link
                        among the archive's members
    test_t24e_archive_fails the tar writer raising after one member: exit 5,
                        PURGE ARCHIVE FAILED, nothing deleted, no archive or
                        .tmp left, the record "archive failed"
    MUTATION=evaljobs   the evaluation's ledger check off: test_t24d (a live
                        job does not refuse) must FAIL; MUTATION=code fails
                        test_t24a and test_t24d as well

Run
    python3 hpc/pipeline/tests/test_smoke_purge.py
    MUTATION=code python3 hpc/pipeline/tests/test_smoke_purge.py
    MUTATION=chain python3 hpc/pipeline/tests/test_smoke_purge.py
    MUTATION=evaljobs python3 hpc/pipeline/tests/test_smoke_purge.py
        About 20 s each; stdlib, `find` and the fake PBS.

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
import purge_outputs as PO          # noqa: E402
import test_smoke_report as TR      # noqa: E402
import test_smoke_synthetic as TS   # noqa: E402  (block 24: the reduced synthetic project)

PYTHON = sys.executable
TOOL = os.path.join(_TOOLS, "purge_outputs.py")
WHAT = "search,best_fit,sim_bank"
ASIDE = "S-A1_set_aside_2026-10-19T161735Z"
SWHAT = "evaluation,best_fits,searches,banks,state"      # block 24

if MUTATION == "code":
    PO.code_of = lambda items, extra=None: "deadbeef"
elif MUTATION == "chain":
    PO.names_chain = lambda raw, chain: False
elif MUTATION == "evaljobs":
    PO.eval_jobs_check = lambda ecfg, ev_submit, pbs_module=None: ([], "the ledger check off")
elif MUTATION:
    raise SystemExit("unknown MUTATION %r; known: code, chain, evaljobs" % MUTATION)


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


def _write(path, data):
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "wb") as fh:
        fh.write(data if isinstance(data, bytes) else data.encode("ascii"))


def _sha_line(path):
    with open(path, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    _write(path + ".sha256", "%s  %s\n" % (h, os.path.basename(path)))
    return h


def _fixture(root):
    """The project of the module docstring; returns (config path, cfg, the
    five target paths in the documented order, the record's sha256)."""
    path, _ids, camp = TR._campaign(root)
    cfg = C.load_config(path)
    p = cfg.data["paths"]
    _write(os.path.join(camp, "gate", "b16", "A1_seed0_probe.json"), '{"projected": 1.0}')
    _write(os.path.join(camp, "bank_split.json"), '{"split_hash": "s"}')
    aside = os.path.join(p["tune_root"], ASIDE)
    _write(os.path.join(aside, "trials", "x.json"), '{"status": "ok"}')
    _write(os.path.join(aside, "RESET_RECORD.json"), '{"schema": "search_reset/1"}')
    _write(os.path.join(p["tune_root"], "S-A1_notes", "keep.txt"), "kept: not a set-aside name")
    best = os.path.join(p["runs_root"], "giulia_best")
    _write(os.path.join(best, "member_seed1", "A1_split1_seed1.json"), '{"nll": 1.0}')
    bank = os.path.join(p["banks_root"], "giulia_sim")
    _write(os.path.join(bank, "shard_0000.npz"), os.urandom(3000))
    _write(os.path.join(bank, "shard_0000.json"), '{"rows": 10}')
    _write(os.path.join(bank, "tasks", "task_0000.json"), '{"task": 0}')
    _write(os.path.join(bank, "BANK_RECORD.json"), '{"tasks": 1}')
    rec_sha = _sha_line(os.path.join(bank, "BANK_RECORD.json"))
    _write(os.path.join(bank, "BANK_VERIFIED.json"), '{"verified": true}')
    _sha_line(os.path.join(bank, "BANK_VERIFIED.json"))
    quar = os.path.join(p["banks_root"], "giulia_sim_quarantine")
    _write(os.path.join(quar, "20261018_120000_round0", "shard_0074.npz"), os.urandom(500))
    _write(os.path.join(quar, "20261018_120000_round0", "QUARANTINE.json"), '{"moved": 1}')
    # kept
    _write(os.path.join(p["banks_root"], "giulia_real", "shard_0000.npz"), os.urandom(200))
    _write(os.path.join(p["banks_root"], "giulia_real", "shard_0000.json"), '{"rows": 288}')
    _write(os.path.join(p["banks_root"], "other_sim", "shard_0000.npz"), os.urandom(100))
    _write(os.path.join(p["sbix_dir"], "export_windows", "giulia", "bank_plan.json"), '{"plan": 1}')
    _write(os.path.join(p["sbix_dir"], "export_windows", "giulia", "logs", "bank_task_1.log"), "ok\n")
    _write(os.path.join(p["work_root"], "pipeline.log"), "2026-11-02 10:00:00 stop: ...\n")
    targets = [camp, aside, best, bank, quar]
    return path, cfg, targets, rec_sha


def _with_env(env, fn):
    old = dict(os.environ)
    os.environ.update(env)
    try:
        return fn()
    finally:
        os.environ.clear()
        os.environ.update(old)


def _run(args, env):
    """The tool in this process (so a MUTATION reaches it); (exit, output)."""
    buf = io.StringIO()

    def go():
        with contextlib.redirect_stdout(buf):
            return PO.main(args)
    rc = _with_env(env, go)
    return rc, buf.getvalue()


def _run_cli(args, env):
    proc = subprocess.run([PYTHON, TOOL] + args, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=300)
    return proc.returncode, proc.stdout


def _code(out):
    for ln in out.splitlines():
        if ln.startswith("CODE "):
            return ln.split()[1]
    raise AssertionError("no CODE line in:\n" + out)


def _digests(top, skip=()):
    """{relative path: sha256} of every file under top, but under `skip`."""
    out = {}
    skip = tuple(os.path.abspath(s) + os.sep for s in skip)
    for d, _subs, files in os.walk(top):
        for f in files:
            q = os.path.join(d, f)
            if os.path.abspath(q).startswith(skip):
                continue
            with open(q, "rb") as fh:
                out[os.path.relpath(q, top)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _find(path):
    """(files, bytes, newest) by `find` (an independent count): lstat sizes,
    links not followed; newest over the folder itself and its files."""
    proc = subprocess.run(["find", path, "-type", "f", "-printf", "%s %T@\n"],
                          stdout=subprocess.PIPE, universal_newlines=True, check=True)
    n, b, newest = 0, 0, int(os.lstat(path).st_mtime)
    for ln in proc.stdout.splitlines():
        s, t = ln.split()
        n += 1
        b += int(s)
        newest = max(newest, int(float(t)))
    return n, b, newest


def _qsub(env, root, name, seconds, variables=None):
    sdir = os.path.join(root, "spool", "scripts")    # under the spool: outside every digest
    if not os.path.isdir(sdir):
        os.makedirs(sdir)
    script = os.path.join(sdir, "%s_%d.sh" % (name, int(time.time() * 1000)))
    with open(script, "w") as fh:
        fh.write("#!/bin/bash\nsleep %s\necho done\n" % seconds)
    argv = ["qsub", "-N", name, "-q", "cpu", "-l", "select=1:ncpus=1:mem=1gb",
            "-l", "walltime=00:10:00", "-o", script + ".log"]
    if variables:
        argv += ["-v", variables]
    proc = subprocess.run(argv + [script], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split()[0]


def _says(out, head, job, tail):
    """A line `<head><job, as qselect or qsub spells it><tail>`."""
    import re
    pat = re.escape(head) + re.escape(str(PBS.job_seq(job))) + r"(\.[^ ]+)?" + re.escape(tail)
    return re.search(pat, out) is not None


def _qdel(env, job, timeout=60.0):
    """qdel, then wait until the fake PBS no longer reports the job live (a
    loaded machine takes longer than the fake's delay)."""
    subprocess.run(["qdel", job], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end = time.time() + timeout
    while _live(env, job):
        assert time.time() < end, ("the fake PBS still reports the job live", job)
        time.sleep(0.2)


def _live(env, job):
    proc = subprocess.run(["qstat", "-x", "-f", job], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    for ln in proc.stdout.splitlines():
        if ln.strip().startswith("job_state = "):
            return ln.split("=", 1)[1].strip() in ("Q", "R", "H", "W")
    return False


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    root = tempfile.mkdtemp(prefix="purge_ka_")
    try:
        path, cfg, targets, rec_sha = _fixture(root)
        env = _env(root)
        before = _digests(root, skip=[os.path.join(root, "spool")])
        rc, out = _run_cli(["--config", path, "--what", WHAT], env)
        assert rc == 0, out
        wr = cfg.data["paths"]["work_root"]
        assert "PURGE (nothing deleted yet): project giulia, work root %s" % wr in out, out
        assert ("  jobs: none of the chain pl_giulia queued, held or running (lock: none; sequencer "
                "jobs (pl_giulia): 0; the state's jobs: 8 recorded; the user's jobs: 0 listed)"
                in out), out
        notes = ["the search's campaign folder", "rounds set aside by search_reset.py",
                 "the best fit's folder", "the simulated bank", "the simulated bank's quarantine"]
        kinds = ["search", "search", "best_fit", "sim_bank", "sim_bank"]
        tot_n = tot_b = 0
        for t, kind, note in zip(targets, kinds, notes):
            n, b, newest = _find(t)
            tot_n += n
            tot_b += b
            head = "  would DELETE [%s] %s: %d file(s), %.2f GB, newest %s (%s" % (
                kind, t, n, b / 1e9, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest)), note)
            assert head in out, (head, out)
        # the campaign's ledger (the report fixture: ok 1, failed 4, unreadable 2; one run folder)
        assert "the ledger: failed 4, ok 1, unreadable 2; 1 run folder(s))" in out, out
        assert "record sha256 %s (its .sha256))" % rec_sha[:16] in out, out
        p = cfg.data["paths"]
        for kept in (os.path.join(p["banks_root"], "giulia_real"),
                     os.path.join(p["sbix_dir"], "export_windows", "giulia"), wr):
            assert "  kept: %s\n" % kept in out, (kept, out)
        assert "S-A1_notes" not in out and "other_sim" not in out, out
        assert "  in all: 5 folder(s), %d file(s), %.2f GB" % (tot_n, tot_b / 1e9) in out, out
        code = _code(out)
        assert len(code) == 8 and int(code, 16) >= 0, code
        assert "CODE %s -- run again with --confirm %s to delete the folders above" % (code, code) \
            in out, out
        assert _digests(root, skip=[os.path.join(root, "spool")]) == before, "the listing changed files"
        assert all(os.path.isdir(t) for t in targets)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reference():
    root = tempfile.mkdtemp(prefix="purge_ref_")
    try:
        path, cfg, targets, _sha = _fixture(root)
        env = _env(root)
        kinds = ["search", "search", "best_fit", "sim_bank", "sim_bank"]
        items = []
        for t, kind in zip(targets, kinds):
            n, b, newest = _find(t)
            items.append({"kind": kind, "path": t, "present": True, "files": n, "bytes": b,
                          "newest": newest})
        # the SPEC's formula, computed here from find's numbers
        want = hashlib.sha256(json.dumps(items, sort_keys=True).encode("ascii")).hexdigest()[:8]
        rc, out = _run(["--config", path, "--what", WHAT], env)
        assert rc == 0 and _code(out) == want, (want, out)
        rc, out2 = _run(["--config", path, "--what", WHAT], env)
        assert _code(out2) == want, "the CODE changed with nothing changed"
        seen = {want}
        # a file in any target changed (bytes, a new file, a newer time) -> another CODE
        changes = [lambda: _write(os.path.join(targets[3], "shard_0000.npz"), os.urandom(3001)),
                   lambda: _write(os.path.join(targets[4], "new.npz"), b"x"),
                   lambda: os.utime(os.path.join(targets[0], "bank_split.json"),
                                    (time.time() + 100, time.time() + 100)),
                   lambda: _write(os.path.join(targets[1], "trials", "y.json"), "{}"),
                   lambda: _write(os.path.join(targets[2], "member_seed2", "z.json"), "{}")]
        for ch in changes:
            ch()
            rc, out = _run(["--config", path, "--what", WHAT], env)
            c = _code(out)
            assert c not in seen, ("a changed target kept a CODE seen before", c, seen)
            seen.add(c)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    root = tempfile.mkdtemp(prefix="purge_inv_")
    try:
        path, cfg, targets, _sha = _fixture(root)
        env = _env(root)
        wr = cfg.data["paths"]["work_root"]
        skip = targets + [os.path.join(root, "spool"), os.path.join(wr, "purge")]
        before = _digests(root, skip=skip)
        assert any(k.endswith("state.json") for k in before), sorted(before)[:5]
        rc, out = _run(["--config", path, "--what", WHAT], env)
        rc, out = _run(["--config", path, "--what", WHAT, "--confirm", _code(out)], env)
        assert rc == 0, out
        for t in targets:
            assert not os.path.lexists(t), ("a target is still there", t)
        assert _digests(root, skip=skip) == before, "a file outside the targets changed"
        p = cfg.data["paths"]
        for kept in (os.path.join(p["banks_root"], "giulia_real", "shard_0000.npz"),
                     os.path.join(p["banks_root"], "other_sim", "shard_0000.npz"),
                     os.path.join(p["tune_root"], "S-A1_notes", "keep.txt"),
                     os.path.join(p["sbix_dir"], "export_windows", "giulia", "bank_plan.json")):
            assert os.path.isfile(kept), kept
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_contract():
    root = tempfile.mkdtemp(prefix="purge_con_")
    try:
        path, cfg, targets, _sha = _fixture(root)
        env = _env(root)
        wr = cfg.data["paths"]["work_root"]
        counts = [_find(t) for t in targets]
        rc, out = _run_cli(["--config", path, "--what", WHAT], env)
        code = _code(out)
        rc, out = _run_cli(["--config", path, "--what", WHAT, "--confirm", code], env)
        assert rc == 0, out
        kinds = ["search", "search", "best_fit", "sim_bank", "sim_bank"]
        for t, kind, (n, b, _t) in zip(targets, kinds, counts):
            assert "DELETED [%s] %s: %d file(s), %.2f GB (" % (kind, t, n, b / 1e9) in out, (t, out)
        n = sum(c[0] for c in counts)
        b = sum(c[1] for c in counts)
        assert ("PURGE DONE: 5 folder(s) deleted, %d file(s), %.2f GB; the state file is not touched"
                % (n, b / 1e9)) in out, out
        recs = sorted(os.listdir(os.path.join(wr, "purge")))
        assert len(recs) == 1 and recs[0].startswith("PURGE_") and recs[0].endswith(".json"), recs
        assert ("PURGE: record %s written" % os.path.join(wr, "purge", recs[0])) in out, out
        with open(os.path.join(wr, "purge", recs[0])) as fh:
            rec = json.load(fh)
        assert rec["schema"] == "purge_outputs/1" and rec["status"] == "done" and \
            rec["code"] == code and rec["config_sha256"] == cfg.sha256, rec
        assert [r["path"] for r in rec["targets"]] == targets, rec["targets"]
        assert all(r["result"] == "deleted" for r in rec["targets"]), rec["targets"]
        # a second listing: every target absent, nothing to delete
        rc, out = _run_cli(["--config", path, "--what", WHAT], env)
        assert rc == 0, out
        for t in (targets[0], targets[2], targets[3], targets[4]):
            assert "  absent       [" in out and t in out, (t, out)
        assert ASIDE not in out and "  in all: 0 folder(s), 0 file(s), 0.00 GB" in out, out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_determinism():
    root = tempfile.mkdtemp(prefix="purge_det_")
    try:
        path, cfg, targets, _sha = _fixture(root)
        env = _env(root)
        tr = cfg.data["paths"]["tune_root"]
        # two more set-aside folders, made in reverse order of their names
        later = [os.path.join(tr, "S-A1_set_aside_2026-11-01T000000Z"),
                 os.path.join(tr, "S-A1_set_aside_2026-10-20T000000Z")]
        for d in later:
            _write(os.path.join(d, "RESET_RECORD.json"), "{}")
        got = [it["path"] for it in _with_env(env, lambda: PO.listing(cfg, PO.KINDS))]
        want = [targets[0]] + sorted([targets[1]] + later) + targets[2:]
        assert got == want, (got, want)
        got2 = [it["path"] for it in _with_env(env, lambda: PO.listing(cfg, ["sim_bank", "search"]))]
        assert got2 == [targets[0]] + sorted([targets[1]] + later) + targets[3:], got2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_edge_cases():
    root = tempfile.mkdtemp(prefix="purge_edge_")
    try:
        path, cfg, targets, _sha = _fixture(root)
        env = _env(root)
        wr = cfg.data["paths"]["work_root"]
        before = _digests(root, skip=[os.path.join(root, "spool")])

        def untouched(why):
            assert _digests(root, skip=[os.path.join(root, "spool")]) == before, \
                "%s: files changed" % why
            assert all(os.path.lexists(t) for t in targets), "%s: a target is gone" % why

        rc, out = _run(["--config", path, "--what", WHAT], env)
        code = _code(out)
        # (a) a wrong CODE
        rc, out = _run(["--config", path, "--what", WHAT, "--confirm", "deadbeef"], env)
        assert rc == 4 and "PURGE REFUSED: the CODE deadbeef does not match %s" % code in out, out
        assert "CODE %s -- run again" % code in out, out
        untouched("a wrong CODE")
        # (b) a lock in the work root
        lock = os.path.join(wr, "pipeline.lock")
        _write(lock, "job:123.pbsserver01 dvnode001\n")
        rc, out = _run(["--config", path, "--what", WHAT, "--confirm", code], env)
        assert rc == 4 and "names job:123.pbsserver01 on dvnode001: a sequencer holds the work " \
            "root" in out and "PURGE REFUSED: the pipeline's jobs must have ended first" in out, out
        os.remove(lock)
        before = _digests(root, skip=[os.path.join(root, "spool")])
        untouched("a lock")
        # (c) a sequencer job queued or running
        drv = _qsub(env, root, "pl_giulia", 60)
        rc, out = _run(["--config", path, "--what", WHAT, "--confirm", code], env)
        assert rc == 4 and _says(out, "the sequencer job ", drv, " is "), out
        untouched("a sequencer job")
        _qdel(env, drv)
        # (d) a job the state records, still running
        tj = _qsub(env, root, "t_live", 60)
        st = ST.State.load(wr)
        st.data["stages"]["search"]["jobs"].append({"id": tj, "role": "trial live",
                                                    "finished": False})
        st.save()
        before = _digests(root, skip=[os.path.join(root, "spool")])
        rc, out = _run(["--config", path, "--what", WHAT, "--confirm", code], env)
        assert rc == 4 and ("the job %s (search, trial live) is " % tj) in out, out
        untouched("a recorded job")
        _qdel(env, tj)
        # (e) a job naming the chain, not in the state, its CHAIN_NAME split by the wrap
        # (the fake PBS keeps the variables sorted by name and wraps Variable_List every 54
        # characters: AAPAD=x*42 puts the split after "CHAIN"); jobs of another chain and of
        # a look-alike name do not refuse
        other = _qsub(env, root, "t_other", 60, "WORK_ROOT=/elsewhere,CHAIN_NAME=pl_other,ROLE=stage")
        alike = _qsub(env, root, "t_alike", 60, "CHAIN_NAME=pl_giulia2,ROLE=stage")
        mine = _qsub(env, root, "t_mine", 60, "AAPAD=" + "x" * 42 + ",CHAIN_NAME=pl_giulia,ROLE=stage")
        raw = subprocess.run(["qstat", "-x", "-f", mine], env=env, stdout=subprocess.PIPE,
                             universal_newlines=True).stdout
        assert "CHAIN\n\t_NAME=pl_giulia" in raw, ("the wrap does not split the name", raw)
        rc, out = _run(["--config", path, "--what", WHAT, "--confirm", code], env)
        assert rc == 4 and _says(out, "the job ", mine, " names the chain pl_giulia and is "), out
        assert PBS.job_seq(other) not in out and PBS.job_seq(alike) not in out, out
        untouched("a job naming the chain")
        _qdel(env, mine)
        rc, out = _run(["--config", path, "--what", WHAT], env)
        assert rc == 0 and _live(env, other) and _live(env, alike), \
            ("another chain's or a look-alike's job refused the listing", out)
        assert "the user's jobs: 2 listed" in out, out
        _qdel(env, other)
        _qdel(env, alike)
        # (f) a target that is a symbolic link
        quar = targets[4]
        moved = quar + "_real_folder"
        os.rename(quar, moved)
        os.symlink(moved, quar)
        before = _digests(root, skip=[os.path.join(root, "spool")])
        rc, out = _run(["--config", path, "--what", WHAT], env)
        assert rc == 4 and "is a symbolic link" in out and "PURGE REFUSED: a target is not a plain " \
            "folder under its root" in out, out
        assert os.path.islink(quar) and os.path.isdir(moved), "the link or its folder changed"
        untouched("a symbolic link")
        # (g) --what empty or unknown; (h) a configuration that does not load
        for what in ("", ",", "search,bogus"):
            rc, out = _run(["--config", path, "--what", what], env)
            assert rc == 2 and out.startswith("PURGE: --what must name one or more of"), (what, out)
        bad = os.path.join(root, "spool", "bad.json")      # outside the digests
        _write(bad, "{}")
        rc, out = _run(["--config", bad, "--what", WHAT], env)
        assert rc == 2 and out.startswith("PURGE: the configuration "), out
        untouched("the argument errors")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------- block 24: synthetic ---

def _eval_json(root, out_root):
    """A reduced evaluation configuration at the default path
    <root>/synth_eval.json (test_smoke_evaluation._eval_config's values)."""
    d = {"out_root": out_root, "seed": 4242, "n_post": 40, "batch_rows": 16,
         "oracle_tol": 1e-4, "ppc_rows": 3, "ppc_draws": 4,
         "cal_glob": os.path.join(out_root, "calbank", "shard_*.npz"), "cal_n": 20,
         "cal_shards": 2, "cal_seed": 55, "lc2st_null": 5, "lc2st_eval_draws": 10,
         "bootstrap_n": 200, "level": 0.95, "dpi": 60, "formats": ["png"], "ncpus": 2,
         "jobs": {"queue": "test", "calbank": {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00"},
                  "sample": {"ncpus": 2, "mem": "2gb", "walltime": "00:20:00"},
                  "report": {"ncpus": 2, "mem": "2gb", "walltime": "00:30:00"},
                  "data": {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00"}},
         "data_view": {"sim_shards": 2, "examples_per_condition": 1, "sim_examples": 3,
                       "zoom_s": 4.0, "bins": 12}}
    path = os.path.join(root, "synth_eval.json")
    _write(path, json.dumps(d, indent=1, sort_keys=True))
    return path


def _synth_fixture(root):
    """Block 24's project (module docstring); returns (config path, eval
    path, cfg, the targets in the documented order, out_root)."""
    path, _d = TS._small(root, searches=[
        {"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False},
        {"name": "S-A2-R0", "campaign": "S-A2", "real": "R0", "alt_best_fit": True}])
    cfg = C.load_config(path, synthetic_ok=True)
    p = cfg.data["paths"]
    ev = os.path.join(p["runs_root"], "evaluation")
    epath = _eval_json(root, ev)
    banks = []
    for b in ("sim", "R0", "R3"):
        bd = os.path.join(p["banks_root"], "synth_sim" if b == "sim" else "synth_real_" + b)
        _write(os.path.join(bd, "shard_0000.npz"), os.urandom(700))
        _write(os.path.join(bd, "shard_0000.json"), '{"rows": 8}')
        _write(os.path.join(bd, "BANK_RECORD.json"), '{"bank": "%s"}' % b)
        _sha_line(os.path.join(bd, "BANK_RECORD.json"))
        banks.append(bd)
    _write(os.path.join(p["banks_root"], "giulia_sim", "shard_0000.npz"), os.urandom(100))
    _write(os.path.join(p["banks_root"], "notes.txt"), "kept: beside the banks\n")
    for q, camp in (("S-A1", "S-A1"), ("S-A2-R0", "S-A2")):
        c = os.path.join(p["tune_root"], q, camp)
        _write(os.path.join(c, "trials", "t0.json"), '{"status": "ok"}')
        _write(os.path.join(c, "trials", "t1.json"), '{"status": "failed"}')
        _write(os.path.join(c, "runs", "r0", "ckpt.pt"), os.urandom(400))
    _write(os.path.join(p["tune_root"], "S-A1", "S-A1_set_aside_2026-11-01T000000Z", "trials",
                        "x.json"), '{"status": "ok"}')
    _write(os.path.join(p["runs_root"], "S-A1", "synth_best", "member_seed1", "m.json"), '{"nll": 1}')
    _write(os.path.join(p["runs_root"], "S-A2-R0", "synth_best", "member_seed1", "m.json"),
           '{"nll": 2}')
    _write(os.path.join(p["runs_root"], "S-A2-R0", "synth_best_alt", "member_seed1", "m.json"),
           '{"nll": 3}')
    _write(os.path.join(ev, "samples", "S-A1_A1_sim.npz"), os.urandom(900))
    _write(os.path.join(ev, "samples", "S-A1_A1_sample.json"), '{"s": 1}')
    _write(os.path.join(ev, "calbank", "shard_0000.npz"), os.urandom(800))
    _write(os.path.join(ev, "metrics", "METRICS.json"), '{"ensembles": {}}')
    _write(os.path.join(ev, "metrics", "paired_S-A2-R0_R0.json"), '{"Dbar": -0.1}')
    _write(os.path.join(ev, "metrics", "paired_S-A2-R0_R0.npz"), os.urandom(300))
    _write(os.path.join(ev, "figures", "FIGURES.md"), "# figures\n")
    _write(os.path.join(ev, "figures", "across", "dbar.png"), os.urandom(250))
    _write(os.path.join(ev, "data", "figures", "stats_pca.png"), os.urandom(150))
    _write(os.path.join(ev, "data", "DATA_VIEW.json"), '{"n": 1}')
    _write(os.path.join(ev, "REPORT_COMPLETE.json"), '{"done": true}')
    _write(os.path.join(ev, "jobs", "logs", "report.log"), "EVAL REPORT OK\n")
    _write(os.path.join(ev, "jobs", "ledger.json"),
           json.dumps({"report": {"job_id": "999990.cluster"},
                       "sample:S-A1:A1": {"job_id": "999991.cluster"}}))
    wr = p["work_root"]
    st = ST.State.new(wr, "synth", path, cfg.sha256)
    st.save()
    st.save()                                           # state.json and its .prev
    _write(os.path.join(wr, "pipeline.log"), "2026-11-12 18:00:00 tick\n")
    _write(os.path.join(wr, "reports", "bank_sim.txt"), "ok\n")
    _write(os.path.join(wr, "purge", "PURGE_2026-11-02T100000Z.json"), '{"schema": "purge_outputs/1"}')
    targets = ([ev]
               + [os.path.join(p["runs_root"], q) for q in ("S-A1", "S-A2-R0")]
               + [os.path.join(p["tune_root"], q) for q in ("S-A1", "S-A2-R0")]
               + banks
               + [os.path.join(wr, n) for n in ("pipeline.log", "reports", "state.json",
                                                 "state.json.prev")])
    return path, epath, cfg, targets, ev


def _members_by_find(ev):
    """[[arcname, size, mtime]] of the archive, by `find`: every regular
    file under ev but its top-level samples/ and calbank/, sorted."""
    proc = subprocess.run(["find", ev, "-type", "f", "-printf", "%P %s %T@\n"],
                          stdout=subprocess.PIPE, universal_newlines=True, check=True)
    out = []
    base = os.path.basename(os.path.normpath(ev))
    for ln in proc.stdout.splitlines():
        rel, s, t = ln.rsplit(" ", 2)
        if rel.split("/")[0] in ("samples", "calbank"):
            continue
        out.append([base + "/" + rel, int(s), int(float(t))])
    return sorted(out)


def _deleted_lines(out):
    return [ln.split("] ", 1)[1].split(":")[0] for ln in out.splitlines()
            if ln.startswith("  would DELETE [")]


def test_t24a_listing():
    root = tempfile.mkdtemp(prefix="purge_t24a_")
    try:
        path, _epath, cfg, targets, ev = _synth_fixture(root)
        env = _env(root)
        arch = os.path.join(root, "archive", "synth_eval.tar.gz")
        os.makedirs(os.path.dirname(arch))
        before = _digests(root, skip=[os.path.join(root, "spool")])
        rc, out = _run(["--config", path, "--what", SWHAT, "--archive", arch], env)
        assert rc == 0, out
        assert _deleted_lines(out) == targets, (_deleted_lines(out), targets)
        for t in targets:
            n, b = (_find(t)[:2] if os.path.isdir(t) else (1, os.lstat(t).st_size))
            assert ("would DELETE [" in out and "] %s: %d file(s), %.2f GB" % (t, n, b / 1e9) in out), (t, out)
        members = _members_by_find(ev)
        assert len(members) == 10, members
        assert ("  would ARCHIVE %d file(s), %.2f GB uncompressed, into %s"
                % (len(members), sum(m[1] for m in members) / 1e9, arch)) in out, out
        assert "    not archived: samples/ (2 file(s)" in out and \
            "    not archived: calbank/ (1 file(s)" in out, out
        assert "  kept: %s" % os.path.join(cfg.data["paths"]["work_root"], "purge") in out, out
        assert "the next submission starts a new run" in out, out
        assert "  in all: 12 target(s)" in out, out
        # the CODE by the SPEC's formula, here
        items = []
        for t in targets:
            n, b, newest = (_find(t) if os.path.isdir(t)
                            else (1, os.lstat(t).st_size, int(os.lstat(t).st_mtime)))
            kind = ("evaluation" if t == ev else "best_fits" if os.sep + "runs_root" + os.sep in t
                    else "searches" if os.sep + "tune_root" + os.sep in t
                    else "banks" if os.sep + "banks_root" + os.sep in t else "state")
            items.append({"kind": kind, "path": t, "present": True, "files": n, "bytes": b,
                          "newest": newest})
        canon = json.dumps({"targets": items, "extra": {"archive": arch, "members": members}},
                           sort_keys=True)
        want = hashlib.sha256(canon.encode("ascii")).hexdigest()[:8]
        code = _code(out)
        assert code == want, (code, want)
        rc2, out2 = _run(["--config", path, "--what", SWHAT, "--archive", arch], env)
        assert rc2 == 0 and _code(out2) == code, out2
        assert _digests(root, skip=[os.path.join(root, "spool")]) == before, "the listing wrote"
        time.sleep(1.1)
        _write(os.path.join(ev, "metrics", "METRICS.json"), '{"ensembles": {"x": 1}}')
        rc3, out3 = _run(["--config", path, "--what", SWHAT, "--archive", arch], env)
        assert rc3 == 0 and _code(out3) != code, "a changed result file left the CODE as it was"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_t24b_confirm():
    root = tempfile.mkdtemp(prefix="purge_t24b_")
    try:
        path, _epath, cfg, targets, ev = _synth_fixture(root)
        env = _env(root)
        wr = cfg.data["paths"]["work_root"]
        arch = os.path.join(root, "archive", "synth_eval.tar.gz")
        os.makedirs(os.path.dirname(arch))
        members = _members_by_find(ev)
        base = os.path.dirname(ev)
        content = {}
        for m in members:
            with open(os.path.join(base, m[0]), "rb") as fh:
                content[m[0]] = fh.read()
        counts = [(_find(t)[:2] if os.path.isdir(t) else (1, os.lstat(t).st_size)) for t in targets]
        tot_n, tot_b = sum(c[0] for c in counts), sum(c[1] for c in counts)
        before = _digests(root, skip=[os.path.join(root, "spool")])
        rc, out = _run_cli(["--config", path, "--what", SWHAT, "--archive", arch], env)
        code = _code(out)
        rc, out = _run_cli(["--config", path, "--what", SWHAT, "--archive", arch,
                            "--confirm", code], env)
        assert rc == 0, out
        import tarfile
        with tarfile.open(arch, "r:gz") as tf:
            got = [(ti.name, ti.size) for ti in tf.getmembers()]
            data = dict((ti.name, tf.extractfile(ti).read()) for ti in tf.getmembers())
        assert got == [(m[0], m[1]) for m in members], (got, members)
        assert data == content, "the archive's bytes are not the files'"
        with open(arch, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        with open(arch + ".sha256") as fh:
            assert fh.read() == "%s  %s\n" % (sha, os.path.basename(arch))
        for t in targets:
            assert not os.path.lexists(t), "still there: %s" % t
        assert sorted(os.listdir(wr)) == ["purge"], os.listdir(wr)
        recs = sorted(os.listdir(os.path.join(wr, "purge")))
        assert len(recs) == 2 and recs[0] == "PURGE_2026-11-02T100000Z.json", recs
        after = _digests(root, skip=[os.path.join(root, "spool")])
        old = os.path.join("work_root", "purge", "PURGE_2026-11-02T100000Z.json")
        assert after[old] == before[old], "the older record changed"
        for k, v in before.items():
            if k.startswith(("banks_root" + os.sep + "giulia_sim", "banks_root" + os.sep + "notes")) \
                    or k in ("synth.json", "synth_eval.json"):
                assert after.get(k) == v, "a decoy or a configuration changed: %s" % k
        with open(os.path.join(wr, "purge", recs[1])) as fh:
            rec = json.load(fh)
        assert rec["status"] == "done" and rec["code"] == code, rec
        assert rec["archive"]["result"] == "written" and rec["archive"]["sha256"] == sha, rec
        assert rec["archive"]["members"] == len(members), rec
        assert all(t["result"] == "deleted" for t in rec["targets"]), rec["targets"]
        assert ("PURGE DONE: 12 target(s) deleted, %d file(s), %.2f GB; archive %s (%d file(s), "
                "sha256 %s); the work root keeps purge/ only: the next submission starts a new run"
                % (tot_n, tot_b / 1e9, arch, len(members), sha[:16])) in out, (tot_n, out)
        assert "ARCHIVED %d file(s)" % len(members) in out, out
        rc2, out2 = _run(["--config", path, "--what", SWHAT], env)
        assert rc2 == 0 and "would DELETE" not in out2 and "would ARCHIVE" not in out2, out2
        assert "  in all: 0 target(s), 0 file(s), 0.00 GB" in out2, out2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_t24c_arguments():
    root = tempfile.mkdtemp(prefix="purge_t24c_")
    try:
        path, _epath, cfg, _targets, ev = _synth_fixture(root)
        env = _env(root)
        p = cfg.data["paths"]
        adir = os.path.join(root, "archive")
        os.makedirs(adir)
        exists = os.path.join(adir, "old.tar.gz")
        _write(exists, "x")
        before = _digests(root, skip=[os.path.join(root, "spool")])
        cases = [
            ("no --archive", ["--what", SWHAT], "needs --archive"),
            ("exists", ["--what", SWHAT, "--archive", exists], "exists"),
            ("under out_root", ["--what", SWHAT, "--archive", os.path.join(ev, "a.tar.gz")],
             "lies under"),
            ("under a target", ["--what", SWHAT, "--archive",
                                os.path.join(p["runs_root"], "S-A1", "a.tar.gz")], "lies under"),
            ("missing folder", ["--what", SWHAT, "--archive", os.path.join(adir, "no", "a.tar.gz")],
             "does not exist"),
            ("without evaluation", ["--what", "banks", "--archive",
                                    os.path.join(adir, "a.tar.gz")], "needs --what evaluation"),
            ("both", ["--what", SWHAT, "--archive", os.path.join(adir, "a.tar.gz"), "--no-archive"],
             "exclude each other"),
            ("block 10's kinds", ["--what", "search,best_fit"], "for a synthetic project"),
        ]
        for name, args, text in cases:
            rc, out = _run(["--config", path] + args, env)
            assert rc == 2 and text in out, (name, rc, out)
        # these kinds, and --archive, on block 10's project
        r10 = os.path.join(root, "p10")
        os.makedirs(r10)
        path10, _cfg10, _t10, _sha10 = _fixture(r10)
        rc, out = _run(["--config", path10, "--what", "banks,state"], env)
        assert rc == 2 and "for this project" in out, out
        rc, out = _run(["--config", path10, "--what", WHAT, "--archive",
                        os.path.join(adir, "a.tar.gz")], env)
        assert rc == 2 and "are for a synthetic project" in out, out
        after = _digests(root, skip=[os.path.join(root, "spool"), r10])
        assert after == before, "an argument error wrote or deleted"
        assert sorted(os.listdir(adir)) == ["old.tar.gz"], os.listdir(adir)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_t24d_refusals():
    root = tempfile.mkdtemp(prefix="purge_t24d_")
    try:
        path, _epath, cfg, _targets, ev = _synth_fixture(root)
        env = _env(root)
        arch = os.path.join(root, "archive", "a.tar.gz")
        os.makedirs(os.path.dirname(arch))
        before = _digests(root, skip=[os.path.join(root, "spool")])

        def untouched(what):
            assert _digests(root, skip=[os.path.join(root, "spool")]) == before, \
                "%s: something changed" % what
            assert not os.path.lexists(arch), "%s: an archive was written" % what
        # (a) a live job of the evaluation's ledger
        led = os.path.join(ev, "jobs", "ledger.json")
        with open(led) as fh:
            ledger_text = fh.read()
        jid = _qsub(env, root, "ev_s0a", 30)
        _write(led, json.dumps({"report": {"job_id": "999990.cluster"},
                                "sample:S-A1:A1": {"job_id": jid}}))
        try:
            rc, out = _run(["--config", path, "--what", SWHAT, "--archive", arch], env)
            assert rc == 4 and ("the evaluation's job %s (sample:S-A1:A1)" % jid) in out and \
                "PURGE REFUSED: the pipeline's jobs must have ended first" in out, out
        finally:
            subprocess.run(["qdel", jid], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _write(led, ledger_text)
        untouched("a live job of the ledger")
        # (b) a wrong CODE
        rc, out = _run(["--config", path, "--what", SWHAT, "--archive", arch,
                        "--confirm", "deadbeef"], env)
        assert rc == 4 and "PURGE REFUSED: the CODE deadbeef does not match" in out, out
        untouched("a wrong CODE")
        # (c) a symbolic link among the archive's members
        link = os.path.join(ev, "figures", "link.png")
        os.symlink(os.path.join(ev, "figures", "FIGURES.md"), link)
        try:
            rc, out = _run(["--config", path, "--what", SWHAT, "--archive", arch], env)
            assert rc == 4 and "is a symbolic link" in out and "[archive]" in out, out
        finally:
            os.remove(link)
        untouched("a link among the members")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_t24e_archive_fails():
    import tarfile
    root = tempfile.mkdtemp(prefix="purge_t24e_")
    real_open = PO._TAR_OPEN
    try:
        path, _epath, cfg, targets, _ev = _synth_fixture(root)
        env = _env(root)
        wr = cfg.data["paths"]["work_root"]
        arch = os.path.join(root, "archive", "a.tar.gz")
        os.makedirs(os.path.dirname(arch))
        rc, out = _run(["--config", path, "--what", SWHAT, "--archive", arch], env)
        code = _code(out)
        before = _digests(root, skip=[os.path.join(root, "spool"), os.path.join(wr, "purge")])

        def failing_open(name, mode, **kw):
            tf = tarfile.open(name, mode, **kw)
            real_add, calls = tf.add, [0]

            def add(*a, **k):
                calls[0] += 1
                if calls[0] > 1:
                    raise OSError("No space left on device (the test)")
                return real_add(*a, **k)
            tf.add = add
            return tf
        PO._TAR_OPEN = failing_open
        rc, out = _run(["--config", path, "--what", SWHAT, "--archive", arch,
                        "--confirm", code], env)
        assert rc == 5 and "PURGE ARCHIVE FAILED: No space left on device (the test); nothing " \
            "deleted" in out, out
        assert not os.path.lexists(arch) and not os.path.lexists(arch + ".tmp"), \
            os.listdir(os.path.dirname(arch))
        after = _digests(root, skip=[os.path.join(root, "spool"), os.path.join(wr, "purge")])
        assert after == before, "a failed archive deleted something"
        for t in targets:
            assert os.path.lexists(t), "gone after a failed archive: %s" % t
        recs = sorted(os.listdir(os.path.join(wr, "purge")))
        with open(os.path.join(wr, "purge", recs[-1])) as fh:
            rec = json.load(fh)
        assert rec["status"] == "archive failed" and rec["archive"]["result"].startswith("failed"), rec
    finally:
        PO._TAR_OPEN = real_open
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- runner ---

def main():
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print("python %s | fake PBS %s" % (sys.version.split()[0], _FAKE))
    if MUTATION:
        print("MUTATION=%s: the named piece is disabled; a check must FAIL" % MUTATION)
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
