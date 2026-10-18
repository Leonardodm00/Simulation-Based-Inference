#!/usr/bin/env python3
"""Smoke test for block 8 of specs/SPEC.md: the one command end to end on
the stand-in davinci, in the situation of the Giulia data set -- S1-S3's
artefacts exist, S4-S8 run with the REAL tools (Sbi-extractor's bank tools
on the bank fixture's synthetic simulations and recordings; SBI's tuner
and runner for the search and the best fit) under the fake PBS, with the
sequencer stopped and re-run at several points.

Checks
    test_contract      from an empty work root to BEST_FIT_COMPLETE.json:
                       the plan, the test task, the array, its gate and the
                       verification job (the certificate), the
                       real bank, the budget gate's probes, 2 + 2 trials,
                       the finalists, 2 members, the ensemble NLL; the
                       sequencer stopped after 3, 8 and 15 polls and re-run
                       each time (a kill and a resume at every stage); no
                       job submitted twice; `--status` after the end; the
                       README's pass lines appear
    test_convergence   skipped
    (the other oracles are blocks 2, 3 and 7's)

Run
    python3 hpc/pipeline/tests/test_smoke_block8.py
        About 3-4 minutes. Needs the Sbi-extractor checkout (SBIX_CLONE or
        a known place), torch, sbi, scikit-optimize.

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

import standin as ST        # noqa: E402

PYTHON = sys.executable


class Standin(object):
    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="pipeline_b8_")
        clone = ST.sbix_clone()
        if clone is None:
            raise unittest.SkipTest("no Sbi-extractor checkout (set SBIX_CLONE)")
        self.sbix = ST.build_sbix(self.root, clone)
        self.fakebin = ST.fake_conda(self.root)
        # a larger synthetic data set: 12 more topologies of 2 simulations per task
        # (topology k's conn_prob is 0.05 + 0.01 (k + 1) + 0.1 task, inside the
        # declared box [0.05, 0.4] up to k = 14 for the third task)
        extra = dict((i, dict((k, [0, 1]) for k in range(3, 15))) for i in range(3))
        self.ds = ST.write_dataset(self.root, self.sbix, extra_topos=extra)
        cfg = ST.synthetic_config(self.root, self.sbix, _REPO, self.ds)
        cfg["split"] = {"fractions": [0.6, 0.2, 0.2]}
        cfg["search"].update({
            "shape_anchors": {"p": 23, "d_theta": 23, "embedding_dim": 8},
            "space_overrides": {"depth_exponent": [3, 3], "width_multiplier": [1.5, 1.5],
                                "embedding_size": [8, 8], "hidden_features": [24, 24],
                                "num_transforms": [3, 3], "batch_size_npe": [16, 32],
                                "head_pool_ops": [0, 1]},
            "n_initial_points": 2, "gp_rounds": 1, "trials_per_round": 2,
            "max_concurrent_trials": 2, "gp_seed": 3, "train_seed": 0,
            "budget_gate": {"probe_steps": 2, "probe_val_rows": 10, "safety_factor": 1.5},
            "max_total_walltime_h": 400})
        cfg["search"]["trial"].update({"passes": 1.0, "epochs": 2, "ncpus": 1, "mem": "8gb",
                                       "walltime": "00:20:00", "queue": "test", "max_resubmits": 1})
        cfg["best_fit"].update({"split_seed": 1, "member_seeds": [1, 2], "passes": 1.0, "epochs": 2,
                                "patience_epochs": 10, "n_posterior_draws": 16, "ncpus": 1,
                                "mem": "8gb", "walltime": "00:20:00", "queue": "test",
                                "max_resubmits": 1})
        cfg["loader"] = {"buffer_shards": 2}
        self.cfg = cfg
        self.path = os.path.join(self.root, "synth.json")
        with open(self.path, "w") as fh:
            json.dump(cfg, fh, indent=1, sort_keys=True)
        self.spool = os.path.join(self.root, "spool")

    def env(self, delay="0.3"):
        env = dict(os.environ)
        env["PATH"] = _FAKE + os.pathsep + self.fakebin + os.pathsep + env.get("PATH", "")
        env["FAKE_PBS_SPOOL"] = self.spool
        env["FAKE_PBS_DELAY"] = delay
        env["PIPELINE_SKIP_CONDA"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["PYTHONPATH"] = _PIPE + os.pathsep + env.get("PYTHONPATH", "")
        for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "PROFILE", "QUEUE", "CONCURRENCY"):
            env.pop(k, None)
        return env

    def run(self, extra, timeout=900):
        cmd = [PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", self.path,
               "--poll-seconds", "0.5"] + list(extra)
        proc = subprocess.run(cmd, cwd=self.root, env=self.env(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
        return proc.returncode, proc.stdout

    def status(self):
        proc = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "status", "--config",
                               self.path], cwd=self.root, env=self.env(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        return proc.returncode, proc.stdout

    def n_submitted(self):
        counter = os.path.join(self.spool, "counter")
        if not os.path.isfile(counter):
            return 0
        with open(counter) as fh:
            return int(fh.read().strip() or 0)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def test_contract():
    s = Standin()
    try:
        rc, out = s.run(["--dry-run"], timeout=120)
        assert rc == 0 and "REFUSE" not in out, out
        assert s.n_submitted() == 0
        outs = []
        for ticks in (3, 8, 15):
            rc, out = s.run(["--max-ticks", str(ticks)], timeout=900)
            assert rc == 0 and ("tick limit %d reached" % ticks in out or "PIPELINE DONE" in out), out
            outs.append(out)
            if "PIPELINE DONE" in out:
                break
        else:
            rc, out = s.run([], timeout=1200)
            assert rc == 0 and "PIPELINE DONE" in out, out
            outs.append(out)
        log = "\n".join(outs)
        cfg = s.cfg
        bank = os.path.join(cfg["paths"]["banks_root"], "synth_sim")
        # the bank stages' pass lines and artefacts
        assert "bank_plan: BANK PLAN OK" in log, log
        assert "sim_bank: test task passed: TASK 0 OK" in log, log
        assert "sim_bank: wrote %s/BANK_RECORD.json" % bank in log, log
        assert "sim_bank: VERIFY OK on " in log and "sim_bank: VERIFIED:" in log, log
        assert os.path.isfile(os.path.join(bank, "BANK_VERIFIED.json.sha256"))
        assert "real_bank: REAL BANK OK" in log, log
        n_shards = len(glob.glob(os.path.join(bank, "shard_*.npz")))
        assert n_shards >= 3, n_shards
        # the search and the best fit on that bank
        camp = os.path.join(cfg["paths"]["tune_root"], "S-A1")
        assert "search: budget gate passed" in log, log
        sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
        assert sc["ended_by"] == "rounds" and sc["n_trials"]["ok"] == 4, sc
        assert sc["identity"]["n_shards"] == n_shards and sc["identity"]["d_theta"] == 23
        assert sc["identity"]["W"] == 18000, sc["identity"]
        best = os.path.join(cfg["paths"]["runs_root"], "synth_best")
        bf = json.load(open(os.path.join(best, "BEST_FIT_COMPLETE.json")))
        assert bf["ensemble"]["M"] == 2 and bf["ensemble"]["L_bar"] <= sum(bf["ensemble"]["L_members"]) / 2.0 + 1e-12
        assert "best_fit: BEST FIT COMPLETE" in log and "PIPELINE DONE" in log
        # the jobs: 4 bank jobs (the test task, the array, its gate, the verification
        # job) + 2 probes + 4 trials + 2 members; none twice
        assert s.n_submitted() == 12, s.n_submitted()
        # every stage done in the state; --status prints it
        state = json.load(open(os.path.join(cfg["paths"]["work_root"], "state.json")))
        statuses = dict((k, v["status"]) for k, v in state["stages"].items())
        assert statuses == {"real_extraction": "skipped", "sim_redetection": "skipped",
                            "label_freeze": "skipped", "bank_plan": "done", "sim_bank": "done",
                            "real_bank": "done", "search": "done", "best_fit": "done"}, statuses
        rc, st = s.status()
        assert rc == 0 and st.count("| done     | yes") == 5 and st.count("| skipped  | yes") == 3, st
        # a last run on the state: nothing to do
        rc, out = s.run([])
        assert rc == 0 and "PIPELINE DONE" in out and s.n_submitted() == 12
    finally:
        s.close()


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


CHECKS = [("test_contract", test_contract), ("test_convergence", test_convergence)]


def main():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
    except ImportError as exc:
        print("SKIP: the training environment is needed (%s)" % exc)
        return 0
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
