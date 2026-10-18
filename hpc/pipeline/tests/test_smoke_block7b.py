#!/usr/bin/env python3
"""Smoke test for block 7's amendment of 2026-10-18 (D-096, D-097) and block
1's (the --retry guard, the requested stop) in specs/SPEC.md: trials and
members that fail of a node fault are healed; checkpoints whose zip CRCs
fail are moved aside; a running sequencer stops on request; --retry is
refused while the stage's jobs run.

The trial jobs run the real tuner and, behind `fault_runner.py`, the real
runner, in a copy of hpc/ (the copy's `stage3/run_joint_arms.py` is the
wrapper), under the fake PBS on the blocks 4-6 bank fixture.

Checks (SPEC block 7, "Node faults and corrupt checkpoints", Test oracles)
    test_known_answer   node_fault on the davinci lines of 2026-10-18 and on
                        lines that are not node faults; checkpoint_problem on
                        a good, a flipped, a truncated and an unreadable file
    test_reference      the committed Giulia configuration (v7) has both
                        keys on, and its dry run names the rule
    test_convergence    not a discretisation: skipped
    test_invariants     node_fault is a pure function; a heal never deletes:
                        every ledger record of a healed trial is kept aside
    test_contract       the healing: SIGBUS then ok, EIO then ok, a
                        ValueError (final), ESHUTDOWN three times (killed);
                        a member's EIO then ok; PIPELINE DONE
    test_determinism    the legacy sweep: a trial recorded failed with the
                        healing off is healed after a requested stop and a
                        restart with it on; no job twice
    test_edge_cases     the checkpoint guard (a flipped byte, a truncated
                        file) and the stop / --retry guard
    MUTATION=<name>     (environment) re-runs one check against a disabled
                        piece of the code, which must then FAIL: signature,
                        sweep, guard, retry_guard

Run
    python3 hpc/pipeline/tests/test_smoke_block7b.py
        About 5 minutes at one thread. Needs torch, sbi, zuko, scikit-
        optimize (the training environment). Nothing touches davinci.
    python3 hpc/pipeline/tests/test_smoke_block7b.py test_contract
        One check (any check names on the line).

Pure ASCII, LF only.
"""

import glob
import json
import os
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
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C                      # noqa: E402
import test_smoke_block7 as B7          # noqa: E402

GIULIA = B7.GIULIA
PYTHON = B7.PYTHON


# --------------------------------------------------------------- helpers ---

def _fault_project(root, plan, search=None, best_fit=None, pre_make_best=False):
    """B7's synthetic project, its code from a copy of hpc/ whose runner is
    the fault wrapper with `plan`."""
    path, cfg = B7._synthetic_project(root, search=search, best_fit=best_fit,
                                      pre_make_best=pre_make_best)
    copy = os.path.join(root, "sbi_copy")
    shutil.copytree(os.path.join(_REPO, "hpc"), os.path.join(copy, "hpc"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    stage3 = os.path.join(copy, "hpc", "joint", "stage3")
    with open(os.path.join(_HERE, "fault_runner.py")) as fh:
        text = fh.read().replace("@@REAL_RUNNER@@", os.path.join(_REPO, "hpc", "joint", "stage3",
                                                                 "run_joint_arms.py"))
    with open(os.path.join(stage3, "run_joint_arms.py"), "w") as fh:
        fh.write(text)
    with open(os.path.join(stage3, "fault_plan.json"), "w") as fh:
        json.dump(plan, fh)
    cfg["paths"]["sbi_dir"] = copy
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    return path, cfg


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


def _order(cfg):
    p = os.path.join(cfg["paths"]["sbi_dir"], "hpc", "joint", "stage3", "fault_state", "order.json")
    return json.load(open(p)) if os.path.isfile(p) else []


def _seq_cmd(cfg_path, extra):
    """The sequencer's command line: pipeline.py itself, or under MUTATION a
    bootstrap that disables one piece of the code before calling main()."""
    entry = _MUT_BOOT if _MUT_BOOT else os.path.join(_PIPE, "pipeline.py")
    return [PYTHON, entry, "run", "--config", cfg_path, "--poll-seconds", "0.5"] + list(extra)


def _run(root, cfg_path, extra, env, timeout=900):
    proc = subprocess.run(_seq_cmd(cfg_path, extra), cwd=root, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return proc.returncode, proc.stdout


def _background(root, cfg_path, extra, env):
    out = open(os.path.join(root, "seq_%d.out" % int(time.time() * 1000)), "w")
    proc = subprocess.Popen(_seq_cmd(cfg_path, extra), cwd=root, env=env, stdout=out,
                            stderr=subprocess.STDOUT)
    return proc, out.name


def _wait_for(path, needle, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        if os.path.isfile(path) and needle in open(path).read():
            return True
        time.sleep(0.3)
    return False


def _stop(cfg_path, env, wait=60):
    r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "stop", "--config", cfg_path,
                        "--wait", str(wait)], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=wait + 30)
    return r.returncode, r.stdout


def _records(camp):
    return dict((os.path.basename(p)[:-5], json.load(open(p)))
                for p in glob.glob(os.path.join(camp, "trials", "*.json")))


# ------------------------------------------------------------- mutations ---

MUTATION = os.environ.get("MUTATION", "")
_PATCHES = {
    "signature": "CM.node_fault = lambda *a, **k: None",
    "sweep": "SS.SearchStage._sweep_failed = lambda self, *a, **k: None",
    "guard": "CM.set_aside_bad_checkpoints = lambda *a, **k: []",
    "retry_guard": "P._unfinished_jobs = lambda st, pbs_module: []",
}


def _mutation_boot():
    """A sequencer entry point with one piece of the code disabled (the
    checks run under it must FAIL); None without MUTATION."""
    if not MUTATION:
        return None, None
    d = tempfile.mkdtemp(prefix="b7b_mut_")
    boot = os.path.join(d, "pipeline_mutated.py")
    with open(boot, "w") as fh:
        fh.write("import sys\nsys.path.insert(0, %r)\nimport pipeline as P\n"
                 "import stages.common as CM\nimport stages.search as SS\n%s\n"
                 "sys.stderr.write('MUTATION %s applied\\n')\nsys.exit(P.main(sys.argv[1:]))\n"
                 % (_PIPE, _PATCHES[MUTATION], MUTATION))
    return d, boot


_MUT_DIR, _MUT_BOOT = _mutation_boot()


def _env(spool, **kw):
    return B7._env(spool, **kw)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    import stages.common as CM
    # the davinci lines of 2026-10-18 (trial logs 16641a4d79ce, 4006d8ff495f), verbatim
    fault_lines = [
        "BrokenPipeError: [Errno 108] Cannot send after transport endpoint shutdown: "
        "'/davinci-1/home/ldellamea/SBI/hpc/dsn'",
        "OSError: [Errno 5] Input/output error: '/davinci-1/home/ldellamea/banks/giulia_sim/"
        "shard_0064.npz'",
        "Bus error (core dumped)",
        "OSError: [Errno 116] Stale file handle",
        "Could not chdir to home directory /davinci-1/home/ldellamea: Input/output error",
    ]
    for line in fault_lines:
        why = CM.node_fault("runner exited 1", "x\n" + line + "\n[evaluate] FAILED\nRUN_ARGV EXIT: 1\n")
        assert why is not None and why in line, (line, why)
    assert CM.node_fault("runner exited -7", None) == "SIGBUS: runner exited -7"
    assert CM.node_fault(None, "RUN_ARGV EXIT: 135\n") == "SIGBUS: RUN_ARGV EXIT 135"
    assert CM.node_fault(None, "RUN_ARGV EXIT: 249\n") == "SIGBUS: RUN_ARGV EXIT 249"
    not_faults = [
        ("runner exited 1", "Traceback (most recent call last):\nValueError: invalid values\n"
                            "RUN_ARGV EXIT: 1\n"),
        ("runner exited 1", "FileNotFoundError: [Errno 2] No such file or directory: 'x'\n"),
        ("runner exited 1", "OSError: [Errno 28] No space left on device\n"),
        ("runner exited 1", "OSError: [Errno 122] Disk quota exceeded\n"),
        ("runner exited 1", "OSError: [Errno 52] Invalid exchange\n"),
        ("runner exited -9", "RUN_ARGV EXIT: 247\n"),
        ("runner exited 1", None),
        (None, ""),
    ]
    for err, text in not_faults:
        assert CM.node_fault(err, text) is None, (err, text)
    # checkpoints: torch.save's zip archive, good / one flipped byte / cut in half / unreadable
    import torch
    root = tempfile.mkdtemp(prefix="b7b_ka_")
    try:
        good = os.path.join(root, "A1_seed0_train_ckpt.pt")
        torch.save({"model_state": {"w": torch.randn(64, 64)}, "epoch": 1}, good)
        assert CM.checkpoint_problem(good) == (None, "")
        raw = bytearray(open(good, "rb").read())
        import zipfile
        with zipfile.ZipFile(good) as z:
            info = max(z.infolist(), key=lambda i: i.file_size)
        off = info.header_offset
        at = (off + 30 + int.from_bytes(raw[off + 26:off + 28], "little")
              + int.from_bytes(raw[off + 28:off + 30], "little") + 100)
        flip = bytearray(raw)
        flip[at] ^= 0xFF
        fl = os.path.join(root, "flip_train_ckpt.pt")
        open(fl, "wb").write(bytes(flip))
        kind, why = CM.checkpoint_problem(fl)
        assert kind == "content" and info.filename in why and "CRC" in why, (kind, why)
        # torch reads the flipped file WITHOUT an error and returns altered weights: why the guard exists
        ck = torch.load(fl, map_location="cpu", weights_only=False)
        assert not torch.equal(ck["model_state"]["w"],
                               torch.load(good, map_location="cpu", weights_only=False)["model_state"]["w"])
        tr = os.path.join(root, "trunc_train_ckpt.pt")
        open(tr, "wb").write(bytes(raw[:len(raw) // 2]))
        kind, why = CM.checkpoint_problem(tr)
        assert kind == "content" and "not a readable zip archive" in why, (kind, why)
        d = os.path.join(root, "dir_train_ckpt.pt")
        os.makedirs(d)
        kind, why = CM.checkpoint_problem(d)
        assert kind == "io", (kind, why)
        # set aside: the two damaged files move to .corrupt1, the good and the unreadable stay
        lines = []
        moved = CM.set_aside_bad_checkpoints(root, lines.append, "search", "trial x")
        assert sorted(os.path.basename(m) for m in moved) == ["flip_train_ckpt.pt.corrupt1",
                                                               "trunc_train_ckpt.pt.corrupt1"], moved
        assert os.path.isfile(good) and os.path.isdir(d)
        assert any("left in place" in ln for ln in lines) and \
            sum("failed its CRC check" in ln for ln in lines) == 2, lines
        # a second damaged file of the same name goes to .corrupt2 (nothing overwritten)
        open(fl, "wb").write(bytes(flip))
        moved = CM.set_aside_bad_checkpoints(root, lines.append, "search", "trial x")
        assert [os.path.basename(m) for m in moved] == ["flip_train_ckpt.pt.corrupt2"], moved
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reference():
    cfg = C.load_config(GIULIA)
    assert cfg.data["config_version"] == 7
    assert cfg.data["search"]["resubmit_node_faults"] is True, "D-096"
    assert cfg.data["best_fit"]["resubmit_node_faults"] is True
    root = tempfile.mkdtemp(prefix="b7b_ref_")
    try:
        rc, out = _run(root, GIULIA, ["--dry-run"], _env(os.path.join(root, "spool")), timeout=120)
        assert "node faults (D-096): a trial failed of SIGBUS or a file-system error is queued " \
               "again with --resume" in out, out
        assert "node faults (D-096): a member failed of SIGBUS or a file-system error is " \
               "resubmitted with --resume" in out, out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("the pipeline has no discretisation parameter")


def test_invariants():
    import stages.common as CM
    a = ("runner exited 1", "OSError: [Errno 5] Input/output error: 'x'\n")
    assert CM.node_fault(*a) == CM.node_fault(*a), "a pure function"
    # the module names no project (D-091), as the rest of the pipeline code
    for p in ("stages/common.py", "stages/search.py", "stages/best_fit.py", "pipeline.py"):
        text = open(os.path.join(_PIPE, p)).read().lower()
        for needle in ("giulia", "ldellamea", "davinci-1/home", "dvnode"):
            assert needle not in text, (p, needle)


def test_contract():
    """The healing, trials and members, end to end."""
    root = B7._tmp()
    try:
        plan = {"trials": [["sigbus", "ok"], ["eio", "ok"], ["value"],
                           ["eshutdown", "eshutdown", "eshutdown"]],
                "members": {"member_seed1": ["eio", "ok"], "member_seed2": ["ok"]}}
        path, c = _fault_project(root, plan, search={"max_concurrent_trials": 1})
        camp = _seed_probes(c)
        spool = os.path.join(root, "spool")
        rc, out = _run(root, path, [], _env(spool), timeout=1500)
        assert rc == 0 and "PIPELINE DONE" in out, out[-6000:]
        order = _order(c)
        assert len(order) == 4, order
        t0, t1, t2, t3 = order
        recs = _records(camp)
        assert recs[t0]["status"] == "ok" and recs[t1]["status"] == "ok", (recs[t0], recs[t1])
        assert recs[t2]["status"] == "failed" and "runner exited 1" in recs[t2]["error"], recs[t2]
        assert recs[t3]["status"] == "killed" and "node fault" in recs[t3]["error"] \
            and "after 2 resubmission(s)" in recs[t3]["error"], recs[t3]
        tr = os.path.join(camp, "trials")
        for t, n in ((t0, 1), (t1, 1), (t2, 0), (t3, 3)):
            got = sorted(os.path.basename(p) for p in glob.glob(os.path.join(tr, t + ".json.nodefault*")))
            assert got == ["%s.json.nodefault%d" % (t, k) for k in range(1, n + 1)], (t, got)
        aside = json.load(open(os.path.join(tr, t0 + ".json.nodefault1")))
        assert aside["status"] == "failed" and aside["error"] == "runner exited -7", aside
        # the lines the user watches
        assert "search: trial %s failed (runner exited -7); examining it for a node fault" % t0[:12] in out
        assert "search: trial %s NODE FAULT on " % t0[:12] in out and \
            "(SIGBUS: runner exited -7); record set aside as %s.json.nodefault1; queued again with " \
            "--resume (1 of 2)" % t0 in out, out
        assert "(OSError: [Errno 5] Input/output error: '/x/banks/synth_sim/shard_0064.npz'); record set " \
               "aside as %s.json.nodefault1" % t1 in out, out
        assert "search: trial %s failed: not a node fault (runner exited 1); final" % t2[:12] in out, out
        assert "BrokenPipeError: [Errno 108] Cannot send after transport endpoint shutdown" in out
        assert "no resubmission left: recorded killed" in out, out
        # the second argv of a healed trial carries --resume; the first does not
        for t in (t0, t1):
            a0 = open(os.path.join(camp, "jobs", t + ".0.argv")).read().split("\n")
            a1 = open(os.path.join(camp, "jobs", t + ".1.argv")).read().split("\n")
            assert "--resume" not in a0 and "--resume" in a1, t
        sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
        assert sc["n_trials"]["ok"] == 2 and sc["n_trials"]["failed"] == 1 and \
            sc["n_trials"]["killed"] == 1, sc["n_trials"]
        # the member's node fault: resubmitted, then done
        assert "best_fit: member seed 1 NODE FAULT on " in out and \
            "resubmitted with --resume (1 of 1)" in out, out
        best = os.path.join(c["paths"]["runs_root"], "synth_best")
        bf = json.load(open(os.path.join(best, "BEST_FIT_COMPLETE.json")))
        assert sorted(m["seed"] for m in bf["members"]) == [1, 2]
        # submissions: trials 2 + 2 + 1 + 3, members 2 + 1
        assert B7._n_submitted(spool) == 11, B7._n_submitted(spool)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_determinism():
    """The legacy sweep: a failed record from before the healing, healed after a restart."""
    root = B7._tmp()
    try:
        plan = {"trials": [["sigbus", "ok"], ["sleep8_ok"], ["ok"], ["ok"]]}
        path, c = _fault_project(root, plan, search={"max_concurrent_trials": 1,
                                                     "resubmit_node_faults": False},
                                 pre_make_best=True)
        camp = _seed_probes(c)
        spool = os.path.join(root, "spool")
        env = _env(spool)
        plog = os.path.join(c["paths"]["work_root"], "pipeline.log")
        proc, _ = _background(root, path, [], env)
        assert _wait_for(plog, "search: submitted trial", 300), "no trial submitted"
        ok_failed = _wait_for(plog, " failed\n", 300)
        assert ok_failed, open(plog).read()[-3000:]
        # the healing is off: the failure is final, as before D-096
        assert "examining it for a node fault" not in open(plog).read()
        rc, out = _stop(path, env, wait=60)
        assert rc == 0 and "stop: requested" in out and "the sequencer has stopped" in out, out
        proc.wait(timeout=60)
        assert proc.returncode == 0, proc.returncode
        text = open(plog).read()
        assert "stopped on request (STOP_REQUEST: asked " in text, text[-2000:]
        n_mid = B7._n_submitted(spool)
        t0 = _order(c)[0]
        assert _records(camp)[t0]["status"] == "failed"
        # the healing on, and a restart: the sweep finds the failed trial, never examined
        cfg = json.load(open(path))
        cfg["search"]["resubmit_node_faults"] = True
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=1, sort_keys=True)
        rc, out = _run(root, path, ["--accept-config-change"], env, timeout=1200)
        assert rc == 0 and "PIPELINE DONE" in out, out[-5000:]
        assert "search: trial %s is recorded failed and was never examined for a node fault; " \
               "examining it" % t0[:12] in out, out
        assert "record set aside as %s.json.nodefault1; queued again with --resume (1 of 2)" % t0 in out
        recs = _records(camp)
        assert len(recs) == 4 and all(r["status"] == "ok" for r in recs.values()), recs
        assert os.path.isfile(os.path.join(camp, "trials", t0 + ".json.nodefault1"))
        # no job twice: trial 0 twice by design, the other three once (the running one re-attached)
        assert B7._n_submitted(spool) == 5, (n_mid, B7._n_submitted(spool))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _test_checkpoint_guard():
    root = B7._tmp()
    try:
        plan = {"trials": [["stop1_flip", "ok"], ["stop1_trunc", "ok"], ["ok"], ["ok"]]}
        path, c = _fault_project(root, plan, search={"max_concurrent_trials": 1}, pre_make_best=True)
        camp = _seed_probes(c)
        spool = os.path.join(root, "spool")
        rc, out = _run(root, path, [], _env(spool), timeout=1500)
        assert rc == 0 and "PIPELINE DONE" in out, out[-5000:]
        t0, t1 = _order(c)[:2]
        for t, why in ((t0, "fails its CRC-32"), (t1, "not a readable zip archive")):
            ck = glob.glob(os.path.join(camp, "runs", t, "*_train_ckpt.pt.corrupt1"))
            assert len(ck) == 1, (t, os.listdir(os.path.join(camp, "runs", t)))
            assert "search: trial %s checkpoint %s failed its CRC check (" % (
                t[:12], os.path.basename(ck[0])[:-len(".corrupt1")]) in out and why in out, out
            assert "moved aside as %s; it restarts from epoch 0" % os.path.basename(ck[0]) in out
        recs = _records(camp)
        assert len(recs) == 4 and all(r["status"] == "ok" for r in recs.values()), recs
        # the stopped jobs ended without a record: the kill path resubmitted them with --resume
        assert "search: trial %s ended without a record" % t0[:12] in out
        assert B7._n_submitted(spool) == 6, B7._n_submitted(spool)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _test_stop_and_retry():
    root = B7._tmp()
    try:
        plan = {"trials": [["sleep25_ok"], ["ok"], ["ok"], ["ok"]]}
        path, c = _fault_project(root, plan, search={"max_concurrent_trials": 1}, pre_make_best=True)
        camp = _seed_probes(c)
        spool = os.path.join(root, "spool")
        env = _env(spool)
        plog = os.path.join(c["paths"]["work_root"], "pipeline.log")
        # (a) SIGTERM: a stop at the tick boundary, exit 0
        proc, _ = _background(root, path, [], env)
        assert _wait_for(plog, "search: submitted trial", 300)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
        assert proc.returncode == 0, proc.returncode
        assert "stopped on request (signal %d); resumable: 1 job(s) still queued or running keep " \
               "running" % signal.SIGTERM in open(plog).read(), open(plog).read()[-2000:]
        n1 = B7._n_submitted(spool)
        assert n1 == 1, n1
        # (b) --retry search while that trial job runs: refused, in the dry run and the run
        for extra in (["--dry-run"], []):
            rc, out = _run(root, path, ["--retry", "search"] + extra, env, timeout=120)
            assert rc == 2 and "REFUSE: --retry search: the stage has 1 job(s) still queued or " \
                               "running" in out, out
        assert B7._n_submitted(spool) == n1, "nothing submitted by a refused retry"
        # (c) the stop command on a running sequencer
        proc, _ = _background(root, path, [], env)
        assert _wait_for(plog, "sequencer started: pid %d" % proc.pid, 120)
        rc, out = _stop(path, env, wait=60)
        assert rc == 0 and "the sequencer has stopped" in out, out
        proc.wait(timeout=60)
        assert proc.returncode == 0
        # (d) once the job has ended, --retry search is accepted and the search completes
        end = time.time() + 300
        while time.time() < end:
            recs = [json.load(open(p)) for p in glob.glob(os.path.join(spool, "*.json"))]
            if recs and all(r.get("finished") for r in recs):
                break
            time.sleep(0.5)
        rc, out = _run(root, path, ["--retry", "search"], env, timeout=1200)
        assert rc == 0 and "search reset to pending (--retry)" in out and "PIPELINE DONE" in out, \
            out[-4000:]
        assert B7._n_submitted(spool) == 4, B7._n_submitted(spool)
        # (e) stop with no sequencer running
        rc, out = _stop(path, env, wait=0)
        assert rc == 0 and "no sequencer is running" in out, out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_edge_cases():
    _test_checkpoint_guard()
    _test_stop_and_retry()


# ------------------------------------------------------------------ main ---

CHECKS = [("test_known_answer", test_known_answer), ("test_reference", test_reference),
          ("test_convergence", test_convergence), ("test_invariants", test_invariants),
          ("test_contract", test_contract), ("test_determinism", test_determinism),
          ("test_edge_cases", test_edge_cases),
          ("checkpoint_guard", _test_checkpoint_guard), ("stop_and_retry", _test_stop_and_retry)]


def main():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
    except ImportError as exc:
        print("SKIP: the training environment is needed (%s)" % exc)
        return 0
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    checks = [(n, f) for n, f in CHECKS if (n in wanted if wanted else not n.startswith(("checkpoint",
                                                                                          "stop_")))]
    if MUTATION:
        print("MUTATION=%s: the named piece is disabled; the checks run must FAIL" % MUTATION)
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
            rows.append((name, "FAIL", str(exc)[:1500], time.time() - t1))
        except Exception:
            bad += 1
            rows.append((name, "ERROR", traceback.format_exc()[-2500:], time.time() - t1))
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    for d in list(B7._SHARED.values()):
        if isinstance(d, str) and os.path.isdir(d) and "banks_" in d:
            shutil.rmtree(d, ignore_errors=True)
    if _MUT_DIR:
        shutil.rmtree(_MUT_DIR, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
