#!/usr/bin/env python3
"""Smoke test for block 2 of specs/SPEC.md: the bank stages S4-S6 through
Sbi-extractor's `launch_bank.sh` on the stand-in davinci (tests/standin.py:
a copy of the Sbi-extractor tools, a fake conda, the bank fixture's
synthetic data set) under the fake PBS, and the self-healing simulated bank
(D-094, D-095): the verification job and the quarantine.

Checks (specs/SPEC.md block 2, "Test oracles")
    test_known_answer   the profile the stage writes loads through
                        bank_plan_config with the fixture's digests and the
                        configuration's values; the committed Giulia
                        configuration gives the committed profile's values
    test_reference      the Giulia dry run prints the three stages' lines,
                        the verification job's among them
    test_convergence    skipped
    test_invariants     the launcher's job lines are parsed; a refusal is
                        read from ABORT/REFUSED lines; the launcher's auto
                        walltime is read from its line
    test_contract       S4 -> S5 (the test task, the array of 3, its gate,
                        the verification job, the certificate) -> S6 on the
                        fixture; a re-run skips every stage; the davinci
                        situation (the plan of record, no banks) builds both
    test_determinism    the sequencer stopped while the test task is queued
                        and re-run re-attaches: no job submitted twice
    test_verify_tool    the verifier on the fixture's bank: intact; truncated;
                        a byte flipped in x with the marker rewritten (the
                        gate passes it, the verifier does not); no marker; a
                        leftover; a shard of no chunk; an unreadable shard;
                        no plan (exit 2); no bank folder (exit 75)
    test_edge_cases     a task that fails every time: quarantine, one rerun,
                        STOP with its problem and its FAILED line; the plan's
                        own refusal stops S4 before any job
    test_dropped_gate   the scheduler ends the gate unrun: with a task that
                        fails, verify names it, STOP (no rerun left); with
                        every task good, verify OK, the gate alone, the
                        certificate; a stale gate log is not read
    test_healing        (h1) the gate passes a CRC-bad shard: verify ->
                        quarantine of the task and the record -> rerun with
                        round 0's walltime -> done; (h2) the davinci case: an
                        element that never starts and a truncated shard with
                        its marker -> healed in one rerun; (h4) a shard of no
                        chunk -> STOP, nothing moved
    test_log_timing     in process: the test task passes by its marker; the
                        gate's outcome (passed / refused / never ran / wait)
                        from its own files, the grace period, a stale log
    test_decisions      in process: what the stage does with each kind of
                        verification report, and with a job that leaves none
    test_cores          D-099: under a cap that leaves 9 cores, with NCPUS=4
                        and GATE_NCPUS=16 in the sequencer's environment (a
                        PBS job's): the tasks ask the launcher's 1 core, the
                        gate its 8, the array's CONCURRENCY is 9, the gate
                        after it, nothing over 9 in the fake PBS's records;
                        in process, the array waits while its gate does not
                        fit, and a cap that can never hold the gate stops
                        the stage at its start

Run
    python3 hpc/pipeline/tests/test_smoke_block2.py
        About 3 minutes. Needs the Sbi-extractor checkout (SBIX_CLONE, or a
        known place) and numpy; nothing touches davinci.

Pure ASCII, LF only.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
_FAKE = os.path.join(_HERE, "fake_pbs")
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C          # noqa: E402
import standin as ST        # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
PYTHON = sys.executable
VERIFY_TOOL = os.path.join(_PIPE, "tools", "bank_verify.py")


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_b2_")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class Standin(object):
    """A stand-in with the data set, the configuration and S7/S8 pre-made."""

    def __init__(self, bank_walltime="00:30:00", pre_make_search=True, max_rerun_rounds=None):
        self.root = _tmp()
        clone = ST.sbix_clone()
        if clone is None:
            raise unittest.SkipTest("no Sbi-extractor checkout (set SBIX_CLONE)")
        self.sbix = ST.build_sbix(self.root, clone)
        self.fakebin = ST.fake_conda(self.root)
        self.ds = ST.write_dataset(self.root, self.sbix)
        self.cfg = ST.synthetic_config(self.root, self.sbix, _REPO, self.ds,
                                       bank_walltime=bank_walltime)
        if max_rerun_rounds is not None:
            self.cfg["bank"]["max_rerun_rounds"] = int(max_rerun_rounds)
        self.path = os.path.join(self.root, "synth.json")
        with open(self.path, "w") as fh:
            json.dump(self.cfg, fh, indent=1, sort_keys=True)
        if pre_make_search:
            for p in (os.path.join(self.cfg["paths"]["tune_root"], "S-A1", "SEARCH_COMPLETE.json"),
                      os.path.join(self.cfg["paths"]["runs_root"], "synth_best",
                                   "BEST_FIT_COMPLETE.json")):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as fh:
                    fh.write('{"pre_made": true}\n')
        self.spool = os.path.join(self.root, "spool")
        self.bank = os.path.join(self.cfg["paths"]["banks_root"], "synth_sim")
        self.plan = os.path.join(self.sbix, "export_windows", "synth", "bank_plan.json")
        self.qroot = os.path.join(self.cfg["paths"]["banks_root"], "synth_sim_quarantine")

    def env(self, delay="0.2", extra=None):
        env = dict(os.environ)
        env["PATH"] = _FAKE + os.pathsep + self.fakebin + os.pathsep + env.get("PATH", "")
        env["FAKE_PBS_SPOOL"] = self.spool
        env["FAKE_PBS_DELAY"] = delay
        env["PIPELINE_SKIP_CONDA"] = "1"
        env["PYTHONPATH"] = _PIPE + os.pathsep + env.get("PYTHONPATH", "")
        for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "PROFILE", "QUEUE", "CONCURRENCY",
                  "FAKE_PBS_ARRAY_DEPEND", "FAKE_PBS_DROP_DEPENDENTS", "FAKE_PBS_FAIL_ONCE",
                  "PIPELINE_LOG_GRACE_S"):
            env.pop(k, None)
        env["PIPELINE_LOG_GRACE_S"] = "1"
        env.update(extra or {})
        return env

    def run(self, extra, timeout=600, delay="0.2", env_extra=None):
        cmd = [PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", self.path,
               "--poll-seconds", "0.5"] + list(extra)
        proc = subprocess.run(cmd, cwd=self.root, env=self.env(delay, env_extra),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=timeout)
        return proc.returncode, proc.stdout

    def n_submitted(self):
        counter = os.path.join(self.spool, "counter")
        if not os.path.isfile(counter):
            return 0
        with open(counter) as fh:
            return int(fh.read().strip() or 0)

    def jobs(self):
        """The fake PBS's records, in submission order."""
        recs = [json.load(open(p)) for p in glob.glob(os.path.join(self.spool, "*.json"))]
        return sorted(recs, key=lambda r: r["num"])

    def verify(self, bank=None, plan=None, out=None):
        out = out or os.path.join(self.root, "report_%d.json" % int(time.time() * 1e6))
        r = subprocess.run([PYTHON, VERIFY_TOOL, "--plan", plan or self.plan, "--bank-root",
                            bank or self.bank, "--out", out, "--workers", "2"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                           timeout=300)
        rep = json.load(open(out)) if os.path.isfile(out) else None
        return r.returncode, r.stdout, rep

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def _marker(bank, index):
    return os.path.join(bank, "tasks", "task_%04d.json" % index)


def _rewrite_marker_sha(bank, index):
    """Make task `index`'s marker record the sha256 of its shards as they are
    now on disk -- bytes a failing node wrote and then hashed."""
    p = _marker(bank, index)
    m = json.load(open(p))
    for s in m["shards"]:
        if s.get("file"):
            s["sha256"] = _sha(os.path.join(bank, s["file"]))
            s["json_sha256"] = _sha(os.path.join(bank, s["json"]))
    with open(p, "w") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)


def _truncate(path):
    blob = open(path, "rb").read()
    with open(path, "wb") as fh:
        fh.write(blob[:len(blob) // 2])


def _flip_in_member(path, member="x.npy"):
    """Flip one byte in the middle of `member`'s compressed data; the zip's
    directory and the other members stay intact."""
    with zipfile.ZipFile(path) as z:
        zi = z.getinfo(member)
    blob = bytearray(open(path, "rb").read())
    off = zi.header_offset
    n_name = int.from_bytes(blob[off + 26:off + 28], "little")
    n_extra = int.from_bytes(blob[off + 28:off + 30], "little")
    start = off + 30 + n_name + n_extra
    at = start + zi.compress_size // 2
    blob[at] ^= 0xFF
    with open(path, "wb") as fh:
        fh.write(bytes(blob))


def _remove_certificate(bank):
    for f in ("BANK_VERIFIED.json", "BANK_VERIFIED.json.sha256"):
        p = os.path.join(bank, f)
        if os.path.exists(p):
            os.remove(p)


def _built(max_rerun_rounds=None, bank_walltime="00:30:00"):
    """A stand-in whose banks are built and verified (S4-S6 done)."""
    s = Standin(max_rerun_rounds=max_rerun_rounds, bank_walltime=bank_walltime)
    rc, out = s.run([], timeout=900)
    assert rc == 0 and "PIPELINE DONE" in out and "sim_bank: VERIFIED:" in out, out
    return s


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    s = Standin()
    try:
        import stages.bank as B
        cfg = C.load_config(s.path)
        stage = B.BankPlanStage()
        stage.write_profile(cfg)
        sys.path.insert(0, os.path.join(s.sbix, "export_windows"))
        import bank_plan_config as BPC
        c = BPC.load_profile(stage._profile_path(cfg), repo_root=s.sbix)
        fx = s.ds["fixture"]
        assert c.record_sha256_prefix == fx["digests"]["record"][:16]
        assert c.freeze_sha256_prefix == fx["digests"]["freeze"][:16]
        assert c.cohort_manifest_sha256_prefix == fx["digests"]["manifest"][:16]
        assert c.cohort_manifest_path.endswith("artifacts/cohort_manifest_synth/cohort_manifest.json")
        assert c.plan_out == os.path.join(s.sbix, "export_windows", "synth", "bank_plan.json")
        assert c.bank_root == os.path.join(s.root, "banks", "synth_sim")
        assert c.expected_p == s.ds["expected_p"] and c.rows_per_shard == 100 and c.sample_seed == 11
        assert c.on_axis == "O_N" and tuple(c.on_declared_box) == (0.01, 3.0)
        # the committed Giulia configuration reproduces the committed profile's decided values
        g = C.load_config(GIULIA)
        clone = ST.sbix_clone()
        with open(os.path.join(clone, "export_windows", "profiles", "giulia.json")) as fh:
            committed = dict((k, v) for k, v in json.load(fh).items() if not k.startswith("_"))
        b = g.data["bank"]
        for k in ("on_axis", "on_declared_box", "topology_axes", "expected_p", "t_win_s", "n_e",
                  "floor_hz", "floor_rtol", "sample_seed", "bootstrap_resamples", "bootstrap_seed",
                  "bytes_sample_rows", "rows_per_shard"):
            assert committed[k] == b[k], (k, committed[k], b[k])
        assert committed["bank_root"] == os.path.join(g.data["paths"]["banks_root"], "giulia_sim")
        assert committed["real_bank_root"] == os.path.join(g.data["paths"]["banks_root"], "giulia_real")
        assert committed["record_path"] == os.path.join(g.data["sim"]["out_root"], "REEXTRACTION_RECORD.json")
        assert committed["sim_dir"] == g.data["sim"]["sim_main"]
        assert committed["freeze_path"] == g.data["freeze"]["out"]
        assert committed["plan_out"] == "export_windows/giulia/bank_plan.json"
        # config v6: the verification job and the rerun budget (D-094); v7 since D-096
        assert g.data["config_version"] == 9
        assert b["verify"] == {"ncpus": 8, "mem": "16gb", "walltime": "02:00:00"}, b["verify"]
        assert b["max_rerun_rounds"] == 2
    finally:
        s.close()


def test_reference():
    root = _tmp()
    try:
        env = dict(os.environ, PATH=_FAKE + os.pathsep + os.environ.get("PATH", ""),
                   FAKE_PBS_SPOOL=os.path.join(root, "spool"), PIPELINE_SKIP_CONDA="1")
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", GIULIA,
                            "--dry-run"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=120)
        out = r.stdout
        assert "REFUSE: stage bank_plan" not in out and "REFUSE: stage sim_bank" not in out \
            and "REFUSE: stage real_bank" not in out, out
        assert "bash /davinci-1/home/ldellamea/repos/Sbi-extractor/export_windows/launch_bank.sh plan" in out
        assert "WALLTIME=auto CONCURRENCY=20 QUEUE=cpu bash" in out and "launch_bank.sh array" in out
        assert "pass line: REAL BANK OK; artefact /davinci-1/home/ldellamea/banks/giulia_real/shard_0000.npz + .json" in out
        assert "bank_plan: would wait for" in out
        assert ("verify, after the gate: qsub -N bank_verify -q cpu -l select=1:ncpus=8:mem=16gb -l "
                "walltime=02:00:00") in out, out
        assert "/davinci-1/home/ldellamea/banks/giulia_sim_quarantine/<time>_round<r>/" in out, out
        assert "at most 2 rerun round(s)" in out, out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    import stages.bank as B
    st = B.SimBankStage()
    jobs = st._jobs_of("[launch] x\n  (cd /x && qsub ...)\n  job   : 1234[].pbsserver01\n"
                       "  gate  : 1235.pbsserver01\n\nwatch: ...")
    assert jobs == {"job": "1234[].pbsserver01", "gate": "1235.pbsserver01"}, jobs
    assert st._refusal("[launch] a\nABORT: WALLTIME=hh:mm:ss is required\n") == \
        "ABORT: WALLTIME=hh:mm:ss is required"
    assert st._refusal("x\nREFUSED: the plan lists another record\nmore\n") == \
        "REFUSED: the plan lists another record"
    assert st._refusal("just text\nlast line\n") == "last line"
    # davinci's line of 2026-10-16
    line = ("[launch] WALLTIME=auto: test task 0 read 66240 simulations in 323 s; the largest task "
            "holds 73287: 2 x 357 s + 10 min, in quarter hours, at least 30 min -> 00:30:00")
    assert B._WALLTIME_USED.search(line).group(1) == "00:30:00"


def test_contract():
    s = Standin()
    try:
        rc, out = s.run(["--dry-run"])
        assert rc == 0 and "REFUSE" not in out, out
        assert "bank_plan: would run" in out and "sim_bank: would wait for bank_plan" in out, out
        assert s.n_submitted() == 0
        rc, out = s.run([], timeout=900)
        assert rc == 0 and "PIPELINE DONE" in out, out
        cfg = s.cfg
        assert os.path.isfile(s.plan) and os.path.isfile(s.plan + ".sha256")
        assert "bank_plan: BANK PLAN OK" in out, out
        assert "sim_bank: test task passed: TASK 0 OK" in out, out
        assert "sim_bank: wrote %s/BANK_RECORD.json" % s.bank in out, out
        assert "sim_bank: VERIFY OK on " in out and "sim_bank: VERIFIED: 3 shard files" in out, out
        assert "real_bank: REAL BANK OK" in out, out
        for f in ("BANK_RECORD.json", "BANK_RECORD.json.sha256", "BANK_VERIFIED.json",
                  "BANK_VERIFIED.json.sha256"):
            assert os.path.isfile(os.path.join(s.bank, f)), f
        cert = json.load(open(os.path.join(s.bank, "BANK_VERIFIED.json")))
        assert cert["record_sha256"] == _sha(os.path.join(s.bank, "BANK_RECORD.json"))
        assert cert["n_npz_checked"] == 3 and cert["rerun_rounds"] == 0 and cert["quarantines"] == []
        assert len(glob.glob(os.path.join(s.bank, "shard_*.npz"))) >= 1
        real = os.path.join(cfg["paths"]["banks_root"], "synth_real")
        assert os.path.isfile(os.path.join(real, "shard_0000.npz")) and os.path.isfile(os.path.join(real, "shard_0000.json"))
        assert "search: skipped, artefact of record exists" in out and "best_fit: skipped" in out
        # one test job, one array, one gate, one verification job
        assert s.n_submitted() == 4, s.n_submitted()
        names = sorted(r["name"] for r in s.jobs())
        assert names == ["bank_gate", "bank_task", "bank_test", "bank_verify"], names
        v = [r for r in s.jobs() if r["name"] == "bank_verify"][0]
        assert "select=1:ncpus=8:mem=16gb" in v["resources"] and "walltime=02:00:00" in v["resources"], v
        assert v["vars"]["ENV_NAME"] == cfg["envs"]["extract"], v["vars"]
        assert not os.path.exists(s.qroot)
        # the profile the stage wrote, the pass lines in the login-node log
        prof = json.load(open(os.path.join(cfg["paths"]["work_root"], "bank_profile_synth.json")))
        assert prof["rows_per_shard"] == 100 and prof["sim_dir"] == s.ds["sim_main"]
        llog = open(os.path.join(cfg["paths"]["work_root"], "logs", "bank_login_node.log")).read()
        assert "BANK PLAN OK" in llog and "REAL BANK OK" in llog and "launch_bank.sh array" in llog
        # a re-run on the state: nothing to do; on a fresh state: every stage skipped
        rc, out2 = s.run([])
        assert rc == 0 and "PIPELINE DONE" in out2 and s.n_submitted() == 4, out2
        os.remove(os.path.join(cfg["paths"]["work_root"], "state.json"))
        rc, out3 = s.run([])
        assert rc == 0 and "PIPELINE DONE" in out3 and s.n_submitted() == 4, out3
        assert out3.count("skipped, artefact of record exists") == 8, out3
        assert "verified (%s/BANK_VERIFIED.json)" % s.bank in out3, out3
        # the real bank's rows: the fixture's real arm
        import numpy as np
        with np.load(os.path.join(real, "shard_0000.npz")) as z:
            n_real = z["x"].shape[0]
        assert n_real >= 1
        # the davinci situation of 2026-10-16: the plan is of record (S4 skipped), the
        # banks are not, the work root is fresh -- no earlier stage wrote the profile:
        # S5 and S6 write it themselves
        shutil.rmtree(cfg["paths"]["work_root"])
        shutil.rmtree(s.bank)
        shutil.rmtree(real)
        rc, out4 = s.run([], timeout=900)
        assert rc == 0 and "PIPELINE DONE" in out4, out4
        assert "bank_plan: skipped, artefact of record exists" in out4, out4
        assert "sim_bank: profile %s written" % os.path.join(cfg["paths"]["work_root"], "bank_profile_synth.json") in out4, out4
        assert "sim_bank: wrote" in out4 and "sim_bank: VERIFIED:" in out4 and "real_bank: REAL BANK OK" in out4, out4
        assert s.n_submitted() == 8, s.n_submitted()
    finally:
        s.close()


def test_determinism():
    """The sequencer stopped while the test task is queued; the re-run re-attaches."""
    s = Standin()
    try:
        # the fake PBS starts a job 4 s after its submission: 3 polls of 0.5 s stop
        # the sequencer while the test task is still queued
        rc1, out1 = s.run(["--max-ticks", "3"], timeout=600, delay="4.0")
        assert rc1 == 0 and "tick limit 3 reached" in out1 and "recorded bank test task 0" in out1, out1
        n_mid = s.n_submitted()
        assert n_mid == 1
        rc2, out2 = s.run([], timeout=900)
        assert rc2 == 0 and "PIPELINE DONE" in out2, out1 + out2
        assert s.n_submitted() == 4, (n_mid, s.n_submitted())
        assert "sim_bank: test task passed: TASK 0 OK" in out2 and "sim_bank: VERIFIED:" in out2, out2
        assert os.path.isfile(os.path.join(s.bank, "BANK_VERIFIED.json"))
    finally:
        s.close()


def test_verify_tool():
    s = _built()
    try:
        def fresh_copy(name):
            d = os.path.join(s.root, name)
            shutil.copytree(s.bank, d)
            return d

        # (1) the intact bank
        rc, out, rep = s.verify()
        assert rc == 0 and rep["verdict"] == "OK" and "VERIFY OK: 3 tasks, 3 shard files" in out, out
        assert rep["n_npz_checked"] == 3 and rep["n_members_checked"] == 33, rep
        assert rep["record"]["present"] and rep["record"]["sidecar_ok"] and rep["record"]["plan_matches"]
        # (2) a shard truncated: not a zip file
        b = fresh_copy("b_trunc")
        _truncate(os.path.join(b, "shard_0001.npz"))
        rc, out, rep = s.verify(bank=b)
        assert rc == 0 and rep["verdict"] == "BAD" and sorted(rep["bad_tasks"]) == ["1"], rep
        probs = " ".join(rep["bad_tasks"]["1"]["problems"])
        assert "changed since its marker" in probs and "not a zip file" in probs, probs
        assert rep["bad_tasks"]["1"]["files"] == ["shard_0001.json", "shard_0001.npz",
                                                  "tasks/task_0001.json"], rep["bad_tasks"]["1"]
        assert "VERIFY BAD: 1 task(s) [1], 0 global problem(s)" in out, out
        # (3) a byte flipped inside x's data, the marker rewritten to match: the
        #     gate (the real Sbi-extractor tool) passes this bank, the verifier does not
        b = fresh_copy("b_flip")
        _flip_in_member(os.path.join(b, "shard_0000.npz"))
        _rewrite_marker_sha(b, 0)
        rc, out, rep = s.verify(bank=b)
        assert rc == 0 and sorted(rep["bad_tasks"]) == ["0"], rep
        probs = " ".join(rep["bad_tasks"]["0"]["problems"])
        assert "member x.npy" in probs and "changed since" not in probs, probs
        env = dict(os.environ, SBI_HPC_DIR=os.path.join(_REPO, "hpc"))
        g = subprocess.run([PYTHON, os.path.join(s.sbix, "export_windows", "bank_gate.py"), "--plan", s.plan,
                            "--bank-root", b, "--workers", "2"], cwd=os.path.join(s.sbix, "export_windows"),
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=300)
        assert g.returncode == 0 and "wrote %s/BANK_RECORD.json" % b in g.stdout, g.stdout
        # (4) a marker removed: no marker; its shard is its file
        b = fresh_copy("b_nomarker")
        os.remove(_marker(b, 2))
        rc, out, rep = s.verify(bank=b)
        assert sorted(rep["bad_tasks"]) == ["2"], rep
        assert rep["bad_tasks"]["2"]["problems"] == ["no marker (the task did not finish)"], rep
        assert rep["bad_tasks"]["2"]["files"] == ["shard_0002.json", "shard_0002.npz"], rep
        # (5) a killed write's leftover belongs to its chunk's task
        b = fresh_copy("b_leftover")
        open(os.path.join(b, "shard_0002.npz.tmp.npz"), "w").write("x")
        open(os.path.join(b, "tasks", "task_0001.json.tmp"), "w").write("x")
        rc, out, rep = s.verify(bank=b)
        assert sorted(rep["bad_tasks"]) == ["1", "2"], rep
        assert "shard_0002.npz.tmp.npz" in rep["bad_tasks"]["2"]["files"], rep
        assert "tasks/task_0001.json.tmp" in rep["bad_tasks"]["1"]["files"], rep
        # (6) a shard of no chunk of the plan: a global problem; a README: reported only
        b = fresh_copy("b_global")
        open(os.path.join(b, "shard_0999.npz"), "w").write("x")
        open(os.path.join(b, "README.txt"), "w").write("notes")
        rc, out, rep = s.verify(bank=b)
        assert rep["verdict"] == "BAD" and rep["bad_tasks"] == {}, rep
        assert len(rep["global_problems"]) == 1 and "shard_0999.npz" in rep["global_problems"][0], rep
        assert rep["other_files"] == ["README.txt"], rep
        # (7) an unreadable shard (a directory in its place): an I/O problem
        b = fresh_copy("b_unread")
        os.remove(os.path.join(b, "shard_0001.npz"))
        os.makedirs(os.path.join(b, "shard_0001.npz"))
        rc, out, rep = s.verify(bank=b)
        assert sorted(rep["bad_tasks"]) == ["1"] and rep["bad_tasks"]["1"]["io_only"], rep
        assert "unreadable" in " ".join(rep["bad_tasks"]["1"]["problems"]), rep
        # (8) no plan: exit 2, no report; (9) no bank folder: exit 75 (the node is suspect)
        rc, out, rep = s.verify(plan=os.path.join(s.root, "absent.json"))
        assert rc == 2 and rep is None, out
        rc, out, rep = s.verify(bank=os.path.join(s.root, "no_such_bank"))
        assert rc == 75 and rep is None and "SUSPECT NODE" in out, out
    finally:
        s.close()


def test_edge_cases():
    # (a) a task that fails every time (its detections vanish after the plan):
    #     round 0's gate refuses, verify names it, quarantine, one rerun, the same
    #     -> STOP with its problem and its task log's FAILED line; 7 jobs
    s = Standin(max_rerun_rounds=1)
    try:
        rc, out = s.run(["--max-ticks", "2"], timeout=600, delay="4.0")
        assert rc == 0 and "tick limit 2 reached" in out and "recorded bank test task 0" in out, out
        task1 = s.ds["fixture"]["tasks"][1]
        shutil.rmtree(task1["out_dir"])
        rc, out = s.run([], timeout=900)
        assert rc == 3, out
        assert "sim_bank: test task passed: TASK 0 OK" in out, out
        assert "sim_bank: the gate 3.fake refused (exit 1): [gate] FAIL" in out, out
        assert "sim_bank: VERIFY BAD task 1 (" in out and "rerun round 1 of at most 1" in out, out
        stop = [l for l in out.split("\n") if "STOP: stage sim_bank failed:" in l][-1]
        assert "after 1 rerun round(s), 1 task(s) still bad" in stop, stop
        assert "task 1 (" in stop and "no marker (the task did not finish)" in stop, stop
        assert "[log: FAILED:" in stop, stop
        assert not os.path.isfile(os.path.join(s.bank, "BANK_RECORD.json"))
        assert s.n_submitted() == 7, s.n_submitted()
        qs = glob.glob(os.path.join(s.qroot, "*_round0", "QUARANTINE.json"))
        assert len(qs) == 1, qs
        q = json.load(open(qs[0]))
        assert q["round"] == 0 and q["moves"] == [] and "D-095" in q["rule"], q
    finally:
        s.close()
    # (b) the plan's own refusal (a wrong label width) stops the run before any job
    s = Standin()
    try:
        s.cfg["bank"]["expected_p"] = 99
        with open(s.path, "w") as fh:
            json.dump(s.cfg, fh, indent=1, sort_keys=True)
        rc, out = s.run([], timeout=300)
        assert rc == 3 and "STOP: stage bank_plan failed: the bank plan was refused" in out, out
        assert "REFUSED" in out, out
        assert s.n_submitted() == 0
        assert not os.path.isfile(s.plan)
    finally:
        s.close()


def test_dropped_gate():
    """The scheduler ends the gate unrun (davinci, 2026-10-16 12:29)."""
    # (i) an element fails and the fake PBS gates the array's dependent: the gate
    #     never runs; verify names the task; no rerun left -> STOP; 4 jobs
    s = Standin(max_rerun_rounds=0)
    try:
        env = {"FAKE_PBS_ARRAY_DEPEND": "gating"}
        rc, out = s.run(["--max-ticks", "2"], timeout=600, delay="4.0", env_extra=env)
        assert rc == 0 and "tick limit 2 reached" in out and "recorded bank test task 0" in out, out
        shutil.rmtree(s.ds["fixture"]["tasks"][1]["out_dir"])
        rc, out = s.run([], timeout=900, env_extra=env)
        assert rc == 3, out
        assert "never ran (finished with exit None, no record, no log at" in out, out
        assert "verifying the bank, round 0" in out and "VERIFY BAD task 1" in out, out
        assert "STOP: stage sim_bank failed: after 0 rerun round(s), 1 task(s) still bad" in out, out
        assert not os.path.isfile(os.path.join(s.bank, "BANK_RECORD.json"))
        assert s.n_submitted() == 4, s.n_submitted()
        gate = [r for r in s.jobs() if r["name"] == "bank_gate"][0]
        assert gate["state"] == "F" and gate["exit_status"] is None and gate["started"] is None, gate
        assert "dependency" in gate["comment"], gate
    finally:
        s.close()
    # (ii) every element fine, the gate dropped anyway, a stale gate log with a
    #      REFUSED from an earlier run: not read; verify OK; the gate alone writes
    #      the record; the certificate; 5 jobs
    s = Standin()
    try:
        env = {"FAKE_PBS_DROP_DEPENDENTS": "1"}
        logdir = os.path.join(s.sbix, "export_windows", "synth", "logs")
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "bank_gate.log"), "w") as fh:
            fh.write("[gate] FAIL stale/task: an earlier run\nREFUSED: 1 problem(s), nothing "
                     "written\n[job] bank_gate_exit=1 finished=x\n")
        rc, out = s.run([], timeout=900, env_extra=env)
        assert rc == 0 and "PIPELINE DONE" in out, out
        assert "never ran" in out and "VERIFY OK on" in out, out
        assert "every task verified and no record (the gate never ran): the gate alone" in out, out
        assert "sim_bank: recorded bank gate alone -> " in out and "sim_bank: VERIFIED:" in out, out
        assert "stale/task" not in out, out
        assert "sim_bank: wrote %s/BANK_RECORD.json" % s.bank in out, out
        assert os.path.isfile(os.path.join(s.bank, "BANK_VERIFIED.json.sha256"))
        assert s.n_submitted() == 5, s.n_submitted()
    finally:
        s.close()


def test_healing():
    # (h1) the gate passes a bank whose test task's shard is CRC-bad (its marker
    #      rewritten to the bad bytes): verify -> quarantine of task 0 AND the
    #      record -> rerun with round 0's walltime (task 0's marker, which the
    #      launcher's auto rule reads, is in the quarantine) -> done
    s = _built(bank_walltime="auto")
    try:
        n0 = s.n_submitted()
        _flip_in_member(os.path.join(s.bank, "shard_0000.npz"))
        _rewrite_marker_sha(s.bank, 0)
        _remove_certificate(s.bank)
        rc, out = s.run(["--retry", "sim_bank"], timeout=900)
        assert rc == 0 and "PIPELINE DONE" in out, out
        assert "sim_bank: the test task's marker belongs to this plan: test skipped" in out, out
        assert "sim_bank: wrote %s/BANK_RECORD.json" % s.bank in out, out
        assert "VERIFY BAD task 0 (" in out and "member x.npy" in out, out
        assert "sim_bank: quarantine task 0 (" in out and "rerun round 1 of at most 2" in out, out
        assert out.count("sim_bank: VERIFY OK on") == 1 and "sim_bank: VERIFIED:" in out, out
        assert s.n_submitted() - n0 == 6, (n0, s.n_submitted())
        arrays = [r for r in s.jobs() if r["name"] == "bank_task"]
        walls = [[x for x in r["resources"] if x.startswith("walltime=")][0] for r in arrays]
        assert len(arrays) == 3 and walls[1] == walls[2] == walls[0], walls
        qs = glob.glob(os.path.join(s.qroot, "*_round0", "QUARANTINE.json"))
        assert len(qs) == 1, qs
        moved = sorted(os.path.relpath(m["to"], os.path.dirname(qs[0])) for m in json.load(open(qs[0]))["moves"])
        assert moved == ["BANK_RECORD.json", "BANK_RECORD.json.sha256", "shard_0000.json",
                         "shard_0000.npz", "tasks/task_0000.json"], moved
        for rel in moved:
            assert os.path.exists(os.path.join(os.path.dirname(qs[0]), rel)), rel
        cert = json.load(open(os.path.join(s.bank, "BANK_VERIFIED.json")))
        assert cert["rerun_rounds"] == 1 and cert["quarantines"] == [os.path.dirname(qs[0])], cert
        rc2, out2, rep = s.verify()
        assert rep["verdict"] == "OK", out2
    finally:
        s.close()
    # (h2) the davinci case: task 1's element never starts in round 0 (exit -2,
    #      the gating switch drops the gate) and task 2's shard is truncated with
    #      its marker rewritten (a node that wrote bad bytes and hashed them) ->
    #      verify names both -> quarantine -> one rerun -> done
    s = _built()
    try:
        n0 = s.n_submitted()
        os.remove(_marker(s.bank, 1))
        _truncate(os.path.join(s.bank, "shard_0002.npz"))
        _rewrite_marker_sha(s.bank, 2)
        for f in ("BANK_RECORD.json", "BANK_RECORD.json.sha256"):
            os.remove(os.path.join(s.bank, f))
        _remove_certificate(s.bank)
        env = {"FAKE_PBS_ARRAY_DEPEND": "gating", "FAKE_PBS_FAIL_ONCE": "1"}
        rc, out = s.run(["--retry", "sim_bank"], timeout=900, env_extra=env)
        assert rc == 0 and "PIPELINE DONE" in out, out
        assert "never ran (finished with exit None" in out, out
        assert "VERIFY BAD task 1 (" in out and "VERIFY BAD task 2 (" in out, out
        assert "not a zip file" in out and "no marker" in out, out
        assert "rerun round 1 of at most 2" in out and "sim_bank: VERIFIED:" in out, out
        arrays = [r for r in s.jobs() if r["name"] == "bank_task"]
        assert arrays[-2]["elements"]["1"] == -2 and arrays[-1]["elements"]["1"] == 0, arrays[-2:]
        assert s.n_submitted() - n0 == 6, (n0, s.n_submitted())
        q = json.load(open(glob.glob(os.path.join(s.qroot, "*_round0", "QUARANTINE.json"))[0]))
        assert sorted(m["task"] for m in q["moves"]) == [1, 1, 2, 2, 2], q["moves"]
        rc2, out2, rep = s.verify()
        assert rep["verdict"] == "OK", out2
    finally:
        s.close()
    # (h4) a shard file of no chunk of the plan: STOP with the global problem,
    #      nothing moved
    s = _built()
    try:
        open(os.path.join(s.bank, "shard_0999.npz"), "w").write("x")
        _remove_certificate(s.bank)
        rc, out = s.run(["--retry", "sim_bank"], timeout=900)
        assert rc == 3, out
        stop = [l for l in out.split("\n") if "STOP: stage sim_bank failed:" in l][-1]
        assert "the bank folder has problems no rerun fixes" in stop and "shard_0999.npz" in stop, stop
        assert "nothing was moved" in stop and not os.path.exists(s.qroot), stop
        assert os.path.isfile(_marker(s.bank, 0)) and os.path.isfile(os.path.join(s.bank, "shard_0000.npz"))
    finally:
        s.close()


class _FakeSub(object):
    """The sequencer's Submitter, in process: every job finished (or the ones
    `running` names still running), submissions counted, lines kept."""

    def __init__(self, running=()):
        self.lines = []
        self.running = set(running)
        self.submitted = []

    def status(self, job_id):
        import pbs as PBS
        if job_id in self.running:
            return PBS.JobStatus(job_id, "R")
        return PBS.JobStatus(job_id, "F", exit_status=0, walltime_s=1.0)

    def log(self, text):
        self.lines.append(text)

    def record(self, *a, **k):
        pass

    # the core cap (D-099): the sequencer's Submitter without a configuration,
    # where everything fits (block 2's in-process checks are not about the cap)
    def budget(self):
        return None

    def free(self):
        return None

    def check(self, n):
        return True

    def fits(self, n):
        return True

    def submit(self, argv, role, log_path=None, cwd=None, env=None, ncpus=None, after=None):
        job_id = "%d.fake" % (900 + len(self.submitted))
        self.submitted.append((role, argv))
        return job_id


def _in_process():
    """A stand-in, its configuration loaded, its log folder, a stand-in plan."""
    s = Standin()
    cfg = C.load_config(s.path)
    logdir = os.path.join(s.sbix, "export_windows", "synth", "logs")
    os.makedirs(logdir, exist_ok=True)
    with open(s.plan, "w") as fh:
        fh.write('{"plan": "stand-in"}\n')
    return s, cfg, logdir


def test_log_timing():
    import stages.bank as B
    s, cfg, logdir = _in_process()
    saved = os.environ.get("PIPELINE_LOG_GRACE_S")
    try:
        plan_sha = _sha(s.plan)

        def state(phase, job, extra_jobs=()):
            jobs = [{"id": job, "role": "x", "finished": False}]
            jobs += [{"id": j, "role": "y", "finished": False} for j in extra_jobs]
            st = {"jobs": jobs, "sub": {"phase": phase}}
            if phase == "test":
                st["sub"].update(test_job=job, test_log=os.path.join(logdir, "bank_test_0.log"))
            else:
                st["sub"].update(gate_job=job, gate_log=os.path.join(logdir, "bank_gate.log"))
            return st

        stage = B.SimBankStage()
        started, verifies = [], []
        stage._start_array = lambda c, st, p: (started.append(1), st["sub"].update(phase="gate"))

        def fake_verify(c, st, p):
            verifies.append(dict(st["sub"]))
            st["sub"]["phase"] = "verify"
            return "running"
        stage._submit_verify = fake_verify
        os.environ["PIPELINE_LOG_GRACE_S"] = "3600"
        sub = _FakeSub()
        # (e) no marker yet but the log says OK: wait for the marker
        with open(os.path.join(logdir, "bank_test_0.log"), "w") as fh:
            fh.write("TASK 0 OK\n[job] bank_task_exit=0 finished=x\n")
        st = state("test", "1.fake")
        assert stage.poll(cfg, st, sub) == "running" and not started
        # (a) the marker arrives; the log carries epilogue lines after the pass line
        os.makedirs(os.path.join(s.bank, "tasks"))
        with open(_marker(s.bank, 0), "w") as fh:
            json.dump({"plan_sha256": plan_sha, "finished": "2026-10-16T11:00:00+02:00"}, fh)
        with open(os.path.join(logdir, "bank_test_0.log"), "a") as fh:
            fh.write("".join("epilogue line %d: resources used\n" % i for i in range(15)))
        assert stage.poll(cfg, st, sub) == "running" and started == [1], sub.lines
        assert any("test task passed: TASK 0 OK (its marker is this plan's)" in l for l in sub.lines)
        # (b) the gate finished, no record, no log: wait; its log arrives with its
        #     end line: refused -> the verification job
        st = state("gate", "2.fake")
        assert stage.poll(cfg, st, sub) == "running" and stage.poll(cfg, st, sub) == "running"
        assert not verifies
        with open(os.path.join(logdir, "bank_gate.log"), "w") as fh:
            fh.write("[gate] FAIL campaign_x/sweep_task0001: shard_0003.npz missing\n"
                     "REFUSED: 1 task(s)\n[job] bank_gate_exit=1 finished=x\n")
        assert stage.poll(cfg, st, sub) == "running" and len(verifies) == 1
        assert st["sub"]["gate_outcome"] == "refused" and "[gate] FAIL campaign_x" in st["sub"]["gate_said"]
        # (c) the record with its sidecar, no log at all: passed -> the verification job
        os.remove(os.path.join(logdir, "bank_gate.log"))
        ST.premake_bank_record(s.bank)
        os.remove(os.path.join(s.bank, "BANK_VERIFIED.json"))
        st = state("gate", "3.fake")
        assert stage.poll(cfg, st, sub) == "running" and st["sub"]["gate_outcome"] == "passed", st["sub"]
        # (d) the grace run out with neither record nor log: never ran -> verify
        for f in ("BANK_RECORD.json", "BANK_RECORD.json.sha256", "BANK_VERIFIED.json.sha256"):
            os.remove(os.path.join(s.bank, f))
        os.environ["PIPELINE_LOG_GRACE_S"] = "0"
        st = state("gate", "4.fake")
        assert stage.poll(cfg, st, sub) == "running" and st["sub"]["gate_outcome"] == "never_ran", st["sub"]
        # (f) the gate gone while its array still runs: wait for the array
        st = state("gate", "6.fake", extra_jobs=["7[].fake"])
        st["sub"]["array_job"] = "7[].fake"
        n = len(verifies)
        assert stage.poll(cfg, st, _FakeSub(running=["7[].fake"])) == "running" and len(verifies) == n
        assert stage.poll(cfg, st, sub) == "running" and len(verifies) == n + 1
        # (g) a log left by an earlier gate (its stat unchanged since this gate's
        #     submission) is not this gate's: never ran, not refused
        with open(os.path.join(logdir, "bank_gate.log"), "w") as fh:
            fh.write("[gate] FAIL stale/task: an earlier run\nREFUSED: 1 problem(s)\n"
                     "[job] bank_gate_exit=1 finished=x\n")
        st = state("gate", "10.fake")
        st["sub"]["gate_log_stat"] = B.file_stat(os.path.join(logdir, "bank_gate.log"))
        assert stage.poll(cfg, st, sub) == "running" and st["sub"]["gate_outcome"] == "never_ran"
        # ... while a state written before the stat was kept reads any log (as before)
        st = state("gate", "11.fake")
        assert stage.poll(cfg, st, sub) == "running" and st["sub"]["gate_outcome"] == "refused"
        assert "stale/task" in st["sub"]["gate_said"]
        # a record written before this gate's submission is not this gate's
        ST.premake_bank_record(s.bank)
        st = state("gate", "12.fake")
        st["sub"]["record_stat"] = B.file_stat(os.path.join(s.bank, "BANK_RECORD.json"))
        st["sub"]["gate_log_stat"] = B.file_stat(os.path.join(logdir, "bank_gate.log"))
        assert stage.poll(cfg, st, sub) == "running" and st["sub"]["gate_outcome"] == "never_ran"
        # the freshness rule itself
        p = os.path.join(logdir, "fresh_probe.log")
        assert B.file_stat(p) is None and not B.log_is_fresh(p, None)
        with open(p, "w") as fh:
            fh.write("a\n")
        assert B.log_is_fresh(p, None) and B.log_is_fresh(p, "unknown")
        before = B.file_stat(p)
        assert not B.log_is_fresh(p, before)
        with open(p, "a") as fh:
            fh.write("b\n")
        assert B.log_is_fresh(p, before)
        # the end-line reader
        assert B.job_exit_line("x\n[job] bank_gate_exit=3 finished=y\n", "bank_gate") == 3
        assert B.job_exit_line("cohort_manifest_exit=0\n", "cohort_manifest") == 0
        assert B.job_exit_line("[job] bank_task_exit=0\n", "bank_gate") is None
    finally:
        if saved is None:
            os.environ.pop("PIPELINE_LOG_GRACE_S", None)
        else:
            os.environ["PIPELINE_LOG_GRACE_S"] = saved
        s.close()


def test_decisions():
    import stages.bank as B
    s, cfg, logdir = _in_process()
    saved = os.environ.get("PIPELINE_LOG_GRACE_S")
    os.environ["PIPELINE_LOG_GRACE_S"] = "0"
    try:
        stage = B.SimBankStage()
        calls = []
        stage._start_array = lambda c, st, p: (calls.append("array"), st["sub"].update(phase="gate"))
        stage._start_gate_alone = lambda c, st, p: (calls.append("gate_alone"),
                                                   st["sub"].update(phase="gate_alone"))
        real_submit = stage._submit_verify

        def report(verdict="OK", bad=None, glob_p=None, n_tasks=3):
            path = os.path.join(s.root, "rep_%d.json" % len(calls + [0]) + str(time.time()))
            rep = {"verdict": verdict, "host": "dvnode999", "n_tasks": n_tasks, "n_npz_checked": 3,
                   "n_members_checked": 33, "seconds": 1.0, "bad_tasks": bad or {},
                   "global_problems": glob_p or []}
            json.dump(rep, open(path, "w"))
            return rep, path

        def state(**kw):
            sub = {"phase": "verify", "round": 0, "verify_try": 1}
            sub.update(kw)
            return {"jobs": [], "sub": sub}

        sub = _FakeSub()
        # OK and this round's record: the certificate; done
        ST.premake_bank_record(s.bank)
        os.remove(os.path.join(s.bank, "BANK_VERIFIED.json"))
        os.remove(os.path.join(s.bank, "BANK_VERIFIED.json.sha256"))
        rep, path = report()
        st = state()
        assert stage._act_on_report(cfg, st, sub, rep, path) == "finished" and st["sub"]["phase"] == "complete"
        assert stage.is_done(cfg)[0], stage.is_done(cfg)
        cert = json.load(open(os.path.join(s.bank, "BANK_VERIFIED.json")))
        assert cert["record_sha256"] == _sha(os.path.join(s.bank, "BANK_RECORD.json"))
        assert cert["verified_by"] == "dvnode999" and cert["report"] == path
        # the record rewritten after the certificate: not done
        with open(os.path.join(s.bank, "BANK_RECORD.json"), "a") as fh:
            fh.write("\n")
        with open(os.path.join(s.bank, "BANK_RECORD.json.sha256"), "w") as fh:
            fh.write("%s  BANK_RECORD.json\n" % _sha(os.path.join(s.bank, "BANK_RECORD.json")))
        ok, why = stage.is_done(cfg)
        assert not ok and "certifies another record" in why, why
        # OK, a record older than this round's gate, the gate never ran: the gate alone
        st = state(gate_outcome="never_ran",
                   record_stat=B.file_stat(os.path.join(s.bank, "BANK_RECORD.json")))
        assert stage._act_on_report(cfg, st, sub, *report()) == "running" and calls[-1] == "gate_alone"
        # ... and when the gate alone has not run either: STOP
        st = state(gate_outcome="never_ran", gate_alone_unrun_of="5.fake", gate_job="6.fake",
                   record_stat=B.file_stat(os.path.join(s.bank, "BANK_RECORD.json")))
        assert stage._act_on_report(cfg, st, sub, *report()) == "finished"
        assert "the gate did not run, alone either (jobs 5.fake and 6.fake)" in st["sub"]["fail"], st["sub"]
        # OK, no record, the gate refused: STOP with its lines
        st = state(gate_outcome="refused", gate_said="[gate] FAIL t: contract differs",
                   record_stat=B.file_stat(os.path.join(s.bank, "BANK_RECORD.json")))
        assert stage._act_on_report(cfg, st, sub, *report()) == "finished"
        assert "[gate] FAIL t: contract differs" in st["sub"]["fail"], st["sub"]
        assert "the verifier does not test" in st["sub"]["fail"], st["sub"]
        # a global problem: STOP, nothing moved
        st = state()
        assert stage._act_on_report(cfg, st, sub, *report("BAD", glob_p=["shard_0999.npz: ..."])) == "finished"
        assert "no rerun fixes" in st["sub"]["fail"] and not os.path.exists(s.qroot), st["sub"]
        # most tasks unreadable by I/O errors alone: the verifier is suspect -> verify again
        submitted = []
        stage._submit_verify = lambda c, st, p: (submitted.append(1), "running")[1]
        io_bad = dict((str(i), {"name": "t%d" % i, "problems": ["shard_%04d.npz unreadable (EIO)" % i],
                                "io_only": True, "files": []}) for i in range(3))
        st = state()
        assert stage._act_on_report(cfg, st, sub, *report("BAD", bad=io_bad, n_tasks=4)) == "running"
        assert submitted == [1] and not os.path.exists(s.qroot)
        # ... unless that has happened MAX_VERIFY_TRIES times
        st = state(verify_try=B.MAX_VERIFY_TRIES)
        assert stage._act_on_report(cfg, st, sub, *report("BAD", bad=io_bad, n_tasks=4)) == "finished"
        assert "could not read 3 of 4 tasks" in st["sub"]["fail"]
        # bad tasks, a rerun left: the quarantine moves their files and the record; rerun
        for rel in ("tasks/task_0001.json", "shard_0001.npz", "shard_0001.json"):
            p = os.path.join(s.bank, rel)
            if not os.path.isdir(os.path.dirname(p)):
                os.makedirs(os.path.dirname(p))
            with open(p, "w") as fh:
                fh.write(rel)
        bad = {"1": {"name": "t1", "problems": ["shard_0001.npz: not a zip file"], "io_only": False,
                     "files": ["shard_0001.json", "shard_0001.npz", "tasks/task_0001.json"]}}
        st = state()
        n_calls = len(calls)
        assert stage._act_on_report(cfg, st, sub, *report("BAD", bad=bad)) == "running"
        assert calls[n_calls:] == ["array"] and st["sub"]["round"] == 1, (calls, st["sub"])
        qdir = st["sub"]["quarantines"][0]
        q = json.load(open(os.path.join(qdir, "QUARANTINE.json")))
        assert sorted(os.path.relpath(m["to"], qdir) for m in q["moves"]) == [
            "BANK_RECORD.json", "BANK_RECORD.json.sha256", "BANK_VERIFIED.json",
            "BANK_VERIFIED.json.sha256", "shard_0001.json", "shard_0001.npz",
            "tasks/task_0001.json"], q["moves"]
        assert open(os.path.join(qdir, "tasks", "task_0001.json")).read() == "tasks/task_0001.json"
        assert not os.path.exists(os.path.join(s.bank, "shard_0001.npz"))
        assert not os.path.exists(os.path.join(s.bank, "BANK_RECORD.json"))
        # bad tasks, no rerun left: STOP with the tasks
        with open(os.path.join(logdir, "bank_task_1.log"), "w") as fh:
            fh.write("[task] x\nFAILED: [Errno 5] Input/output error: '/x/mea_iter_00112.npz'\n"
                     "[job] bank_task_exit=1 finished=x\n")
        st = state(round=2)
        assert stage._act_on_report(cfg, st, sub, *report("BAD", bad=bad)) == "finished"
        fail = st["sub"]["fail"]
        assert "after 2 rerun round(s), 1 task(s) still bad" in fail and "task 1 (t1)" in fail, fail
        assert "[log: FAILED: [Errno 5] Input/output error" in fail, fail
        # a verification job that ends without its report: again, then STOP
        stage._submit_verify = real_submit
        st = {"jobs": [{"id": "50.fake", "role": "v", "finished": False}],
              "sub": {"phase": "verify", "round": 0, "verify_try": 1, "verify_job": "50.fake",
                      "verify_report": os.path.join(s.root, "never.json"),
                      "verify_log": os.path.join(s.root, "never.log")}}
        assert stage.poll(cfg, st, sub) == "running" and st["sub"]["verify_try"] == 2, st["sub"]
        assert sub.submitted[-1][0] == "bank verify (round 0, try 2)", sub.submitted
        argv = sub.submitted[-1][1]
        assert argv[0] == "qsub" and "bank_verify" in argv and argv[-1].endswith("run_argv.pbs"), argv
        st["sub"]["verify_try"] = B.MAX_VERIFY_TRIES
        st["jobs"].append({"id": st["sub"]["verify_job"], "role": "v", "finished": False})
        assert stage.poll(cfg, st, sub) == "finished"
        assert "ended without its report 3 times in round 0" in st["sub"]["fail"], st["sub"]
    finally:
        if saved is None:
            os.environ.pop("PIPELINE_LOG_GRACE_S", None)
        else:
            os.environ["PIPELINE_LOG_GRACE_S"] = saved
        s.close()


def _record_cores(rec):
    """The cores a fake PBS record can hold at once: its -l values, times
    min(%c, the elements) for an array."""
    import pbs as PBS
    argv = ["qsub"]
    for r in rec.get("resources", []):
        argv += ["-l", r]
    if rec.get("array_spec"):
        argv += ["-J", rec["array_spec"]]
    return PBS.cores_of_argv(argv + ["x"]) or 0


def _peak(records):
    """The most cores the records held at once: from each one's submission
    -- or the end of the job it depends on -- to its end."""
    by_id = dict((r["id"], r) for r in records)
    ev = []
    for r in records:
        begin = r["submitted"]
        dep = by_id.get(r.get("depend") or "")
        if dep is not None:
            begin = max(begin, dep["finished"] or float("inf"))
        end = r["finished"] if r["finished"] else float("inf")
        if begin < end:
            ev += [(begin, 1, _record_cores(r)), (end, 0, -_record_cores(r))]
    peak, now = 0, 0
    for _t, _k, d in sorted(ev):
        now += d
        peak = max(peak, now)
    return peak


def _capped(s, stages_share):
    """resources.max_cores = the sequencer's cores + `stages_share`; the
    sized jobs small enough for the load's cross-check (the verifier 2)."""
    c = s.cfg
    c["search"]["trial"]["ncpus"] = 1
    c["best_fit"]["ncpus"] = 1
    c["sim"]["ncpus"] = 1
    c["bank"]["verify"]["ncpus"] = 2
    c["resources"]["max_cores"] = int(c["driver"]["ncpus"]) + int(stages_share)
    with open(s.path, "w") as fh:
        json.dump(c, fh, indent=1, sort_keys=True)
    return C.load_config(s.path)


def test_cores():
    import pipeline as P
    import state as SF
    import stages.bank as B
    s = Standin()
    try:
        cfg = _capped(s, 9)
        rc, out = s.run(["--dry-run"])
        assert rc == 0 and ("cores (D-099): a task 1 core(s) and the gate 8 (the launcher's NCPUS / "
                            "GATE_NCPUS, passed to it); the array at CONCURRENCY 9 = min(bank."
                            "concurrency 20, the cap's room 9), its gate after it; the verification "
                            "job 2") in out, out
        # in process first: another stage's job holds 2 of the 9: the array (its
        # gate after it: 8) waits, logged once; the stage's own state untouched
        held = SF.State.new(os.path.join(s.root, "wr_inproc"), "synth", s.path, "a" * 64)
        held.stage("other")["status"] = "running"
        held.add_job("other", "77.fake", "x", ["qsub", "-l", "select=1:ncpus=2", "x.pbs"])
        lines = []
        sub = P.Submitter(None, held, "sim_bank", lines.append, cfg)
        st = {"jobs": [], "sub": {"phase": "test", "round": 0}}
        B.SimBankStage()._start_array(cfg, st, sub)
        B.SimBankStage()._start_array(cfg, st, sub)
        assert st["sub"]["phase"] == "array_waiting" and not st["jobs"], st
        assert lines == ["sim_bank: the array (its gate after it) waits for cores (8 asked, 7 free "
                         "under resources.max_cores, D-099)"], lines
        # a cap whose share (7) can never hold the gate (8): the stage stops at its start
        c7 = json.loads(json.dumps(s.cfg))
        c7["resources"]["max_cores"] = int(c7["driver"]["ncpus"]) + 7
        p7 = os.path.join(s.root, "cap7.json")
        with open(p7, "w") as fh:
            json.dump(c7, fh, indent=1, sort_keys=True)
        cfg7 = C.load_config(p7)
        sub7 = P.Submitter(None, SF.State.new(os.path.join(s.root, "wr7"), "synth", p7, "b" * 64),
                           "sim_bank", lines.append, cfg7)
        try:
            B.SimBankStage().start(cfg7, {"jobs": [], "sub": {}}, sub7)
            raise AssertionError("a gate of 8 cores in a share of 7 must stop the stage")
        except P.CapExceeded as exc:
            assert ("sim_bank: a job of 8 cores can never start: the stages' jobs share 7 cores "
                    "(resources.max_cores %d less the sequencer's job, %d)"
                    % (int(c7["driver"]["ncpus"]) + 7, int(c7["driver"]["ncpus"]))) in str(exc), exc
        # the run, with the cores PBS gives a job in the sequencer's environment
        rc, out = s.run([], timeout=900, env_extra={"NCPUS": "4", "GATE_NCPUS": "16"})
        assert rc == 0 and "PIPELINE DONE" in out and "sim_bank: VERIFIED:" in out, out
        assert ("sim_bank: the array at CONCURRENCY 9 (bank.concurrency 20; the cap leaves room for 9 "
                "task(s) of 1 core(s), D-099)") in out, out
        recs = s.jobs()
        names = [r["name"] for r in recs]
        assert names == ["bank_test", "bank_task", "bank_gate", "bank_verify"], names
        test, task, gate, verify = recs
        assert _record_cores(test) == 1 and _record_cores(gate) == 8 and _record_cores(verify) == 2, \
            [(r["name"], r["resources"]) for r in recs]
        assert task["array_spec"] == "0-2%9" and _record_cores(task) == 3, task["array_spec"]
        assert gate["depend"] == task["id"]
        assert _peak(recs) <= 9, "the bank's jobs held %d cores at once, over 9" % _peak(recs)
        st = json.load(open(os.path.join(s.cfg["paths"]["work_root"], "state.json")))
        jobs = st["stages"]["sim_bank"]["jobs"]
        assert [j.get("ncpus") for j in jobs] == [1, 3, 8, 2], jobs
        assert jobs[2]["after"] == jobs[1]["id"]
    finally:
        s.close()


# ------------------------------------------------------------------ main ---

CHECKS = [("test_known_answer", test_known_answer), ("test_reference", test_reference),
          ("test_convergence", test_convergence), ("test_invariants", test_invariants),
          ("test_contract", test_contract), ("test_determinism", test_determinism),
          ("test_verify_tool", test_verify_tool), ("test_edge_cases", test_edge_cases),
          ("test_dropped_gate", test_dropped_gate), ("test_healing", test_healing),
          ("test_log_timing", test_log_timing), ("test_decisions", test_decisions),
          ("test_cores", test_cores)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        t1 = time.time()
        try:
            fn()
            rows.append((name, "PASS", "", time.time() - t1))
        except unittest.SkipTest as exc:
            rows.append((name, "SKIP", str(exc), time.time() - t1))
        except AssertionError as exc:
            bad += 1
            rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
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
