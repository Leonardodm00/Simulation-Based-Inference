#!/usr/bin/env python3
"""Smoke test for block 2 of specs/SPEC.md: the bank stages S4-S6 through
Sbi-extractor's `launch_bank.sh` on the stand-in davinci (tests/standin.py:
a copy of the Sbi-extractor tools, a fake conda, the bank fixture's
synthetic data set) under the fake PBS.

Checks (specs/SPEC.md block 2, "Test oracles")
    test_known_answer   the profile the stage writes loads through
                        bank_plan_config with the fixture's digests and the
                        configuration's values; the committed Giulia
                        configuration gives the committed profile's values
    test_reference      the Giulia dry run prints the three stages' lines
    test_convergence    skipped
    test_invariants     the launcher's job lines are parsed; a refusal is
                        read from ABORT/REFUSED lines
    test_contract       S4 -> S5 (the test task, the array of 3 and its
                        gate) -> S6 on the fixture: BANK PLAN OK, TASK 0 OK,
                        the gate's `wrote .../BANK_RECORD.json`, REAL BANK
                        OK, the artefacts with their sidecars, S7/S8 skipped
                        (pre-made); a re-run skips every stage
    test_determinism    the sequencer stopped while the array runs and re-run
                        re-attaches: no job submitted twice
    test_edge_cases     a refused gate (a shard removed after the test task)
                        stops the run with the [gate] FAIL line; a missing
                        input stops S4 before anything runs

Run
    python3 hpc/pipeline/tests/test_smoke_block2.py
        About 2 minutes. Needs the Sbi-extractor checkout (SBIX_CLONE, or a
        known place) and numpy; nothing touches davinci.

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
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C          # noqa: E402
import standin as ST        # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
PYTHON = sys.executable


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_b2_")


class Standin(object):
    """A stand-in with the data set, the configuration and S7/S8 pre-made."""

    def __init__(self, bank_walltime="00:30:00", pre_make_search=True):
        self.root = _tmp()
        clone = ST.sbix_clone()
        if clone is None:
            raise unittest.SkipTest("no Sbi-extractor checkout (set SBIX_CLONE)")
        self.sbix = ST.build_sbix(self.root, clone)
        self.fakebin = ST.fake_conda(self.root)
        self.ds = ST.write_dataset(self.root, self.sbix)
        self.cfg = ST.synthetic_config(self.root, self.sbix, _REPO, self.ds,
                                       bank_walltime=bank_walltime)
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

    def env(self, delay="0.2"):
        env = dict(os.environ)
        env["PATH"] = _FAKE + os.pathsep + self.fakebin + os.pathsep + env.get("PATH", "")
        env["FAKE_PBS_SPOOL"] = self.spool
        env["FAKE_PBS_DELAY"] = delay
        env["PIPELINE_SKIP_CONDA"] = "1"
        env["PYTHONPATH"] = _PIPE + os.pathsep + env.get("PYTHONPATH", "")
        for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "PROFILE", "QUEUE", "CONCURRENCY"):
            env.pop(k, None)
        return env

    def run(self, extra, timeout=600, delay="0.2"):
        cmd = [PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", self.path,
               "--poll-seconds", "0.5"] + list(extra)
        proc = subprocess.run(cmd, cwd=self.root, env=self.env(delay), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
        return proc.returncode, proc.stdout

    def n_submitted(self):
        counter = os.path.join(self.spool, "counter")
        if not os.path.isfile(counter):
            return 0
        with open(counter) as fh:
            return int(fh.read().strip() or 0)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    s = Standin()
    try:
        import stages.bank as B
        cfg = C.load_config(s.path)
        stage = B.BankPlanStage()
        prof = stage.write_profile(cfg)
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
        plan = os.path.join(s.sbix, "export_windows", "synth", "bank_plan.json")
        assert os.path.isfile(plan) and os.path.isfile(plan + ".sha256")
        assert "bank_plan: BANK PLAN OK" in out, out
        assert "sim_bank: test task passed: TASK 0 OK" in out, out
        assert "sim_bank: wrote %s/BANK_RECORD.json" % os.path.join(cfg["paths"]["banks_root"], "synth_sim") in out, out
        assert "real_bank: REAL BANK OK" in out, out
        bank = os.path.join(cfg["paths"]["banks_root"], "synth_sim")
        rec = json.load(open(os.path.join(bank, "BANK_RECORD.json")))
        assert os.path.isfile(os.path.join(bank, "BANK_RECORD.json.sha256"))
        assert len(glob.glob(os.path.join(bank, "shard_*.npz"))) >= 1
        real = os.path.join(cfg["paths"]["banks_root"], "synth_real")
        assert os.path.isfile(os.path.join(real, "shard_0000.npz")) and os.path.isfile(os.path.join(real, "shard_0000.json"))
        assert "search: skipped, artefact of record exists" in out and "best_fit: skipped" in out
        # one test job, one array, one gate
        assert s.n_submitted() == 3, s.n_submitted()
        names = sorted(json.load(open(p))["name"] for p in glob.glob(os.path.join(s.spool, "*.json")))
        assert names == ["bank_gate", "bank_task", "bank_test"], names
        # the profile the stage wrote, the pass lines in the login-node log
        prof = json.load(open(os.path.join(cfg["paths"]["work_root"], "bank_profile_synth.json")))
        assert prof["rows_per_shard"] == 100 and prof["sim_dir"] == s.ds["sim_main"]
        llog = open(os.path.join(cfg["paths"]["work_root"], "logs", "bank_login_node.log")).read()
        assert "BANK PLAN OK" in llog and "REAL BANK OK" in llog and "launch_bank.sh array" in llog
        # a re-run on the state: nothing to do; on a fresh state: every stage skipped
        rc, out2 = s.run([])
        assert rc == 0 and "PIPELINE DONE" in out2 and s.n_submitted() == 3, out2
        os.remove(os.path.join(cfg["paths"]["work_root"], "state.json"))
        rc, out3 = s.run([])
        assert rc == 0 and "PIPELINE DONE" in out3 and s.n_submitted() == 3, out3
        assert out3.count("skipped, artefact of record exists") == 8, out3
        # the real bank's rows: the fixture's real arm
        import numpy as np
        with np.load(os.path.join(real, "shard_0000.npz")) as z:
            n_real = z["x"].shape[0]
        assert n_real >= 1
        assert rec.get("plan_sha256") or rec.get("n_rows") is not None
        # the davinci situation of 2026-10-16: the plan is of record (S4 skipped), the
        # banks are not, the work root is fresh -- no earlier stage wrote the profile:
        # S5 and S6 write it themselves
        shutil.rmtree(cfg["paths"]["work_root"])
        shutil.rmtree(bank)
        shutil.rmtree(real)
        rc, out4 = s.run([], timeout=900)
        assert rc == 0 and "PIPELINE DONE" in out4, out4
        assert "bank_plan: skipped, artefact of record exists" in out4, out4
        assert "sim_bank: profile %s written" % os.path.join(cfg["paths"]["work_root"], "bank_profile_synth.json") in out4, out4
        assert "sim_bank: wrote" in out4 and "real_bank: REAL BANK OK" in out4, out4
        assert s.n_submitted() == 6, s.n_submitted()
    finally:
        s.close()


def test_determinism():
    """The sequencer stopped while the array runs; the re-run re-attaches."""
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
        assert s.n_submitted() == 3, (n_mid, s.n_submitted())
        assert "sim_bank: test task passed: TASK 0 OK" in out2 and "sim_bank: wrote" in out2, out2
        assert os.path.isfile(os.path.join(s.cfg["paths"]["banks_root"], "synth_sim", "BANK_RECORD.json"))
    finally:
        s.close()


def test_edge_cases():
    # (a) a refused gate: the plan and the test task run, the sequencer stops
    #     while the test task is queued; a task's detections vanish; the re-run's
    #     array fails that task, the gate (ordering only after an array, D-006)
    #     refuses it, the run stops with the [gate] FAIL line, no record written
    s = Standin()
    try:
        rc, out = s.run(["--max-ticks", "2"], timeout=600, delay="4.0")
        assert rc == 0 and "tick limit 2 reached" in out and "recorded bank test task 0" in out, out
        bank = os.path.join(s.cfg["paths"]["banks_root"], "synth_sim")
        task1 = s.ds["fixture"]["tasks"][1]
        shutil.rmtree(task1["out_dir"])
        rc, out = s.run([], timeout=900)
        assert rc == 3 and "STOP: stage sim_bank failed: the gate refused" in out, out
        assert "[gate] FAIL" in out, out
        assert "sim_bank: test task passed: TASK 0 OK" in out, out
        assert not os.path.isfile(os.path.join(bank, "BANK_RECORD.json"))
        assert s.n_submitted() == 3
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
        assert not os.path.isfile(os.path.join(s.sbix, "export_windows", "synth", "bank_plan.json"))
    finally:
        s.close()


# ------------------------------------------------------------------ main ---

CHECKS = [("test_known_answer", test_known_answer), ("test_reference", test_reference),
          ("test_convergence", test_convergence), ("test_invariants", test_invariants),
          ("test_contract", test_contract), ("test_determinism", test_determinism),
          ("test_edge_cases", test_edge_cases)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    for name, fn in CHECKS:
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
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
