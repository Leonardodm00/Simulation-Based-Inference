#!/usr/bin/env python3
"""Smoke test for block 7 of specs/SPEC.md: the search stage S7 and the
best-fit stage S8 under the fake PBS, on the blocks 4-6 bank fixture (8
simulated shards of unequal sizes, W 128, p 23, 44 donors).

Checks (specs/SPEC.md block 7, "Test oracles")
    test_known_answer   the tools: the split hash equals the runner's on
                        the fixture; the ensemble NLL of M copies of one
                        member equals that member's L; L_bar <= the
                        members' mean (Jensen); the costliest configuration
                        of the Giulia space is the one the configuration
                        names; the committed configuration has no
                        NEEDS-CODE key left
    test_reference      the dry run of the committed Giulia configuration
                        prints the gate's, the trials' and the members'
                        qsub lines and no REFUSE line
    test_convergence    not a discretisation: skipped
    test_invariants     every tuner call of a search carries the same
                        identity arguments; a results folder of another
                        identity is refused; the pipeline code still names
                        no project
    test_contract       the full chain on the fixture: gate (2 probes),
                        round 0 (2 random) and round 1 (2 by the GP) with
                        the throttle at 1, SEARCH_COMPLETE ended_by rounds,
                        the best fit's 2 members on split seed 1, the
                        ensemble NLL, BEST_FIT_COMPLETE, PIPELINE DONE; the
                        sequencer stopped mid-search and re-run submits no
                        job twice; every trial record carries the
                        identity's split hash; --num-bins 10 and
                        --head-pool-ops reach the runner
    test_determinism    a trial killed at its walltime is resubmitted with
                        --resume max_resubmits times, then recorded killed;
                        a search with no completed trial stops the run;
                        --retry search after raising the walltime revives
                        the killed trials and the search completes
    test_edge_cases     the elapsed clock at a tiny cap ends the search
                        after round 0 (ended_by budget), the GP round never
                        proposed; a pre-made BEST_FIT_COMPLETE skips S8

Run
    python3 hpc/pipeline/tests/test_smoke_block7.py
        About 5 minutes at one thread. Needs torch, sbi, zuko, scikit-
        optimize (the training environment); run with PIPELINE_SKIP_CONDA=1
        set by the test itself. Nothing touches davinci.

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

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
_FAKE = os.path.join(_HERE, "fake_pbs")
_STAGE3 = os.path.join(_REPO, "hpc", "joint", "stage3")
for _p in (_PIPE, _HERE, _STAGE3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C      # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
PYTHON = sys.executable
TOOLS = os.path.join(_PIPE, "tools")


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_b7_")


def _env(spool, delay="0.2", enforce_walltime=False):
    env = dict(os.environ)
    env["PATH"] = _FAKE + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = spool
    env["FAKE_PBS_DELAY"] = delay
    env["FAKE_PBS_ENFORCE_WALLTIME"] = "1" if enforce_walltime else "0"
    env["PIPELINE_SKIP_CONDA"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = _PIPE + os.pathsep + env.get("PYTHONPATH", "")
    return env


_SHARED = {}


def banks():
    """The blocks 4-6 fixture banks, built once per process."""
    if "root" not in _SHARED:
        import smoke_test_blocks_4_6 as B46
        root = tempfile.mkdtemp(prefix="pipeline_b7_banks_")
        B46.build_banks(root)
        _SHARED["root"] = root
        _SHARED["sim_dir"] = os.path.join(root, "S")
        _SHARED["real_dir"] = os.path.join(root, "R")
    return _SHARED


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    with open(path + ".sha256", "w") as fh:
        fh.write(h.hexdigest() + "  " + os.path.basename(path) + "\n")


def _synthetic_project(root, search=None, best_fit=None, pre_make_best=False):
    """A configuration on the fixture banks with S1-S6 already done."""
    with open(GIULIA, "rb") as fh:
        raw = json.loads(fh.read().decode("ascii"))
    cfg = C._strip_docs(raw)
    cfg["dataset"] = "synth"
    for k in list(cfg["paths"]):
        cfg["paths"][k] = os.path.join(root, k)
    cfg["paths"]["sbi_dir"] = _REPO
    cfg["real"]["cohort_config"] = os.path.join(root, "cohort.json")
    cfg["real"]["extract_root"] = os.path.join(root, "extracted")
    for k in ("sim_main", "out_root", "tools"):
        cfg["sim"][k] = os.path.join(root, k)
    cfg["search"].update({
        "shape_anchors": {"p": 23, "d_theta": 23, "embedding_dim": 8},
        "space_overrides": {"depth_exponent": [3, 3], "width_multiplier": [1.5, 1.5],
                            "embedding_size": [8, 8], "hidden_features": [24, 24],
                            "num_transforms": [3, 3], "batch_size_npe": [16, 32],
                            "head_pool_ops": [0, 1]},
        "n_initial_points": 2, "gp_rounds": 1, "trials_per_round": 2,
        "max_concurrent_trials": 1, "gp_seed": 3, "train_seed": 0,
        "budget_gate": {"probe_steps": 2, "probe_val_rows": 10, "safety_factor": 1.5},
        "max_total_walltime_h": 400,
    })
    cfg["search"]["trial"].update({"passes": 1.0, "epochs": 2, "ncpus": 1, "mem": "4gb",
                                   "walltime": "00:10:00", "queue": "test", "max_resubmits": 2})
    cfg["best_fit"].update({"split_seed": 1, "member_seeds": [1, 2], "passes": 1.0, "epochs": 2,
                            "patience_epochs": 10, "n_posterior_draws": 16, "ncpus": 1,
                            "mem": "4gb", "walltime": "00:10:00", "queue": "test",
                            "max_resubmits": 1})
    cfg["loader"] = {"buffer_shards": 2}
    if search:
        for k, v in search.items():
            if isinstance(v, dict):
                cfg["search"][k].update(v)
            else:
                cfg["search"][k] = v
    if best_fit:
        cfg["best_fit"].update(best_fit)
    path = os.path.join(root, "synth.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)

    # S1-S6 done: their artefacts of record
    b = banks()
    sim_dir = os.path.join(cfg["paths"]["banks_root"], "synth_sim")
    real_dir = os.path.join(cfg["paths"]["banks_root"], "synth_real")
    shutil.copytree(b["sim_dir"], sim_dir)
    shutil.copytree(b["real_dir"], real_dir)
    for p in (os.path.join(cfg["real"]["extract_root"], "cohort_manifest.json"),
              os.path.join(cfg["sim"]["out_root"], "REEXTRACTION_RECORD.json"),
              os.path.join(cfg["paths"]["sbix_dir"], cfg["freeze"]["out"]),
              os.path.join(cfg["paths"]["sbix_dir"], "export_windows", "synth", "bank_plan.json"),
              os.path.join(sim_dir, "BANK_RECORD.json")):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write('{"synthetic": true}\n')
        _sha256_file(p)
    if pre_make_best:
        p = os.path.join(cfg["paths"]["runs_root"], "synth_best", "BEST_FIT_COMPLETE.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write('{"pre_made": true}\n')
    return path, cfg


def _run(root, cfg_path, extra, env, timeout=900):
    cmd = [PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", cfg_path,
           "--poll-seconds", "0.5"] + list(extra)
    proc = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return proc.returncode, proc.stdout


def _n_submitted(spool):
    counter = os.path.join(spool, "counter")
    if not os.path.isfile(counter):
        return 0
    with open(counter) as fh:
        return int(fh.read().strip() or 0)


def _tool(name, args, timeout=600):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    r = subprocess.run([PYTHON, os.path.join(TOOLS, name)] + args, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return r.returncode, r.stdout


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    b = banks()
    root = _tmp()
    try:
        # (a) the split tool against the runner's own function on the fixture
        out = os.path.join(root, "split.json")
        rc, text = _tool("bank_split_hash.py", ["--sim-shards", os.path.join(b["sim_dir"], "*.npz"),
                                                "--split-fracs", "0.85", "0.1", "0.05",
                                                "--split-seed", "0", "--out", out])
        assert rc == 0 and "SPLIT OK:" in text, text
        rec = json.load(open(out))
        import latent_bank as LB
        from run_joint_arms import grouped_split
        idx = LB.ShardIndex(sorted(glob.glob(os.path.join(b["sim_dir"], "*.npz"))))
        split, h = grouped_split(idx.donor, fracs=(0.85, 0.1, 0.05), seed=0)
        assert rec["split_hash"] == h
        assert rec["n_train"] == int((split == 0).sum()) and rec["n_report"] == int((split == 2).sum())
        assert rec["contract_digest"] == idx.sidecar["contract_digest"]
        assert rec["n_rows"] == idx.n_rows == 194 and rec["W"] == 128 and rec["d_theta"] == 23
        # (b) the ensemble NLL: M copies of one member; Jensen on two members
        rng = np.random.default_rng(0)
        l1 = rng.gamma(2.0, 1.0, size=50)
        l2 = l1 + rng.normal(0, 0.5, size=50)
        g = np.arange(50) // 5
        th = rng.uniform(size=(50, 3)).astype(np.float32)
        np.savez(os.path.join(root, "m1.npz"), nll=l1, group=g, theta=th)
        np.savez(os.path.join(root, "m2.npz"), nll=l2, group=g, theta=th)
        np.savez(os.path.join(root, "m3.npz"), nll=l1[:40], group=g[:40], theta=th[:40])
        o = os.path.join(root, "ens1.json")
        rc, text = _tool("ensemble_nll.py", ["--out", o, "--members", os.path.join(root, "m1.npz"),
                                             os.path.join(root, "m1.npz"), os.path.join(root, "m1.npz")])
        assert rc == 0 and "ENSEMBLE OK:" in text, text
        e = json.load(open(o))
        assert e["M"] == 3 and e["n_report"] == 50
        assert abs(e["L_bar"] - float(np.mean(l1))) < 1e-12, (e["L_bar"], np.mean(l1))
        o2 = os.path.join(root, "ens2.json")
        rc, text = _tool("ensemble_nll.py", ["--out", o2, "--members", os.path.join(root, "m1.npz"),
                                             os.path.join(root, "m2.npz")])
        assert rc == 0, text
        e2 = json.load(open(o2))
        assert e2["L_bar"] <= 0.5 * (np.mean(l1) + np.mean(l2)) + 1e-12, "Jensen"
        assert e2["L_bar"] <= min(np.mean(l1), np.mean(l2)) + 1.0
        rc, text = _tool("ensemble_nll.py", ["--out", os.path.join(root, "ens3.json"), "--members",
                                             os.path.join(root, "m1.npz"), os.path.join(root, "m3.npz")])
        assert rc != 0 and "not the same split" in text, text
        # (c) the costliest configuration of the committed Giulia space
        cfg = C.load_config(GIULIA)
        s = cfg.data["search"]
        env = dict(os.environ)
        r = subprocess.run([PYTHON, os.path.join(_REPO, "hpc", "joint", "stage4", "npe_tune_joint.py"),
                            "costliest", "--campaign", s["campaign"], "--p", str(s["shape_anchors"]["p"]),
                            "--embedding-dim", str(s["shape_anchors"]["embedding_dim"]),
                            "--d-theta", str(s["shape_anchors"]["d_theta"]), "--n-train", "536000",
                            "--space-overrides", json.dumps(s["space_overrides"]),
                            "--out-dir", os.path.join(root, "cost")], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode == 0 and "COSTLIEST OK: 3 batch level(s): 256, 512, 1024" in r.stdout, r.stdout
        c = json.load(open(os.path.join(root, "cost", "costliest_b1024.json")))["config"]
        want = {"depth_exponent": 5, "width_multiplier": 3.0, "block_family": 0, "embedding_size": 64,
                "head_fusion": 1, "head_pool_ops": 1, "hidden_features": 512, "num_transforms": 12,
                "batch_size_npe": 1024, "dsn_on": 0, "rep_on": 0}
        got = dict((k, c[k]) for k in want)
        assert got == want, got
        # (d) no NEEDS-CODE key left
        assert C.unhonoured_keys(cfg) == [], C.unhonoured_keys(cfg)
        assert cfg.data["config_version"] == 5
        assert cfg.data["search"]["trial"]["queue"] == "cpu" and cfg.data["best_fit"]["queue"] == "cpu"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reference():
    """The committed Giulia configuration's dry run (davinci paths absent)."""
    root = _tmp()
    try:
        env = _env(os.path.join(root, "spool"))
        rc, out = _run(root, GIULIA, ["--dry-run"], env, timeout=120)
        # blocks 2 and 3 are not built: their six REFUSE lines are expected; none for S7/S8
        assert "REFUSE: this code cannot honour" not in out, out
        assert "REFUSE: stage search" not in out and "REFUSE: stage best_fit" not in out, out
        assert "search: would wait for sim_bank, real_bank" in out, out
        assert "qsub -N gate_b<B> -q cpu -l select=1:ncpus=48:mem=160gb -l walltime=120:00:00" in out
        assert "qsub -N t_<trial> -q cpu -l select=1:ncpus=48:mem=160gb -l walltime=120:00:00" in out
        assert "qsub -N bf_giulia_s<k> -q cpu -l select=1:ncpus=48:mem=200gb -l walltime=300:00:00" in out
        assert "round 0 = 50 random configurations, then 20 GP round(s) of 10 (250 trials in all), at most 25" in out
        assert "cap 400 h (D-090)" in out
        assert "search: would wait for" in out and "best_fit: would wait for" in out
        assert "tools/bank_split_hash.py --sim-shards /davinci-1/home/ldellamea/banks/giulia_sim/shard_*.npz --split-fracs 0.85 0.1 0.05 --split-seed 0" in out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("the pipeline has no discretisation parameter")


def test_invariants():
    import re
    import stages.common as CM
    cfg = C.load_config(GIULIA)
    ident = {"n_train": 10, "split_hash": "a" * 64, "contract_digest": "b" * 64}
    a = CM.tuner_common_args(cfg, ident)
    for flag in ("--campaign", "--results-dir", "--p", "--embedding-dim", "--d-theta", "--n-train",
                 "--split-hash", "--contract-digest", "--seed", "--space-overrides"):
        assert flag in a, flag
    assert a == CM.tuner_common_args(cfg, dict(ident)), "the identity arguments are a pure function"
    assert CM.job_name("t_" + "f" * 40) == "t_" + "f" * 13 and CM.job_name("<x>") == "j_x_"
    assert abs(CM.mem_to_gb("160gb") - 160.0) < 1e-12 and abs(CM.mem_to_gb("512mb") - 0.5) < 1e-12
    assert CM.utc_to_epoch("1970-01-01T01:00:00Z") == 3600.0
    # (b) a results folder of another identity is refused before anything is submitted
    root = _tmp()
    try:
        path, c = _synthetic_project(root)
        rounds = os.path.join(c["paths"]["tune_root"], c["search"]["campaign"], "pipeline_rounds.json")
        os.makedirs(os.path.dirname(rounds))
        with open(rounds, "w") as fh:
            json.dump({"schema": "pipeline_rounds/1", "identity": {"split_hash": "0" * 64,
                       "contract_digest": "0" * 64, "n_train": 1, "campaign": "S-A1"},
                       "rounds": [], "gate": None, "clock_start_utc": None, "ended_by": None}, fh)
        env = _env(os.path.join(root, "spool"))
        rc, out = _run(root, path, [], env, timeout=300)
        assert rc == 3 and "STOP: stage search refused: REFUSE:" in out and \
            "holds a search of another identity" in out, out
        assert _n_submitted(os.path.join(root, "spool")) == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)
    # (c) agnostic: no project value in the pipeline code (D-091)
    offenders = []
    for dirpath, _, files in os.walk(_PIPE):
        if "projects" in dirpath.split(os.sep) or "__pycache__" in dirpath:
            continue
        for f in files:
            if not f.endswith((".py", ".sh", ".pbs")):
                continue
            p = os.path.join(dirpath, f)
            with open(p, "rb") as fh:
                text = fh.read().decode("ascii")
            if "tests" in dirpath.split(os.sep):
                continue
            for needle in ("giulia", "ldellamea", "davinci-1/home", "cadex", "hhgap"):
                if re.search(needle, text, re.IGNORECASE):
                    offenders.append("%s: %s" % (os.path.relpath(p, _PIPE), needle))
    assert not offenders, offenders


def test_contract():
    """The full chain on the fixture, with a sequencer stop mid-search."""
    root = _tmp()
    try:
        path, c = _synthetic_project(root)
        spool = os.path.join(root, "spool")
        env = _env(spool)
        rc, out = _run(root, path, ["--dry-run"], env, timeout=120)
        assert rc == 0 and "REFUSE" not in out, out
        assert _n_submitted(spool) == 0
        # a first sequencer stops after 12 polls (the gate and the first trials are in flight)
        rc1, out1 = _run(root, path, ["--max-ticks", "12"], env, timeout=600)
        assert rc1 == 0 and "tick limit 12 reached" in out1, out1
        n_mid = _n_submitted(spool)
        rc2, out2 = _run(root, path, [], env, timeout=900)
        out = out1 + out2
        assert rc2 == 0 and "PIPELINE DONE: every stage done or skipped" in out2, out
        camp = os.path.join(c["paths"]["tune_root"], "S-A1")
        # the gate: 2 probes (batch 16 and 32), passed
        assert "search: budget gate passed" in out, out
        for b_ in (16, 32):
            assert glob.glob(os.path.join(camp, "gate", "b%d" % b_, "*_probe.json")), b_
        # the rounds: 2 + 2 trials, all ok, SEARCH_COMPLETE ended_by rounds
        sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
        assert sc["ended_by"] == "rounds" and sc["rounds_evaluated"] == 2, sc
        assert sc["n_trials"]["ok"] == 4 and sc["n_trials"]["failed"] == 0 and sc["n_trials"]["killed"] == 0, sc
        rf = json.load(open(os.path.join(camp, "pipeline_rounds.json")))
        assert [len(r["trial_ids"]) for r in rf["rounds"]] == [2, 2], rf["rounds"]
        assert rf["clock_start_utc"] and rf["gate"]["verdict"]["ok"]
        recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(camp, "trials", "*.json")))]
        assert len(recs) == 4 and all(r["status"] == "ok" for r in recs)
        assert all(r["split_hash"] == sc["identity"]["split_hash"] for r in recs), \
            "the runner's split hash is the tool's"
        assert all(np.isfinite(r["nll"]) for r in recs)
        assert not glob.glob(os.path.join(camp, "pending", "*.json")), "pending files consumed"
        fin = json.load(open(os.path.join(camp, "finalists.json")))
        assert fin["K"] == 1 and fin["finalists"][0]["nll_rank_split"] == min(r["nll"] for r in recs)
        assert sc["finalist"]["trial_id"] == fin["finalists"][0]["trial_id"]
        # the throttle at 1: never two trial jobs running at once (the fake PBS's records)
        trial_jobs = []
        for p in glob.glob(os.path.join(spool, "*.json")):
            j = json.load(open(p))
            if j["name"].startswith("t_"):
                trial_jobs.append((j["started"], j["finished"]))
        trial_jobs.sort()
        assert len(trial_jobs) == 4
        for (s0, f0), (s1, f1) in zip(trial_jobs, trial_jobs[1:]):
            assert s1 >= f0 - 1e-6, "two trial jobs overlapped under max_concurrent_trials 1"
        # the runner received the space's num_bins and head_pool_ops, and the pipeline's flags
        logs = sorted(glob.glob(os.path.join(camp, "jobs", "*.0.log")))
        tlog = [p for p in logs if "/gate_" not in p]
        text = open(tlog[0]).read()
        for needle in ("--num-bins 10", "--head-pool-ops", "--passes 1.0", "--split-fracs 0.85 0.1 0.05",
                       "--buffer-shards 2", "--threads 1", "--flow-hidden-layers 2", "--patience 99",
                       "[evaluate] nll=", "RUN_ARGV EXIT: 0"):
            assert needle in text, (needle, text[-3000:])
        # the best fit: 2 members on split seed 1, the ensemble, the artefact
        best = os.path.join(c["paths"]["runs_root"], "synth_best")
        bf = json.load(open(os.path.join(best, "BEST_FIT_COMPLETE.json")))
        assert bf["ensemble"]["M"] == 2 and bf["split_seed"] == 1
        assert sorted(m["seed"] for m in bf["members"]) == [1, 2]
        m1 = json.load(open(os.path.join(best, "member_seed1", "A1_seed1.json")))
        m2 = json.load(open(os.path.join(best, "member_seed2", "A1_split1_seed2.json")))
        assert m1["split_seed"] == 1 and m2["split_seed"] == 1 and m1["split_hash"] == m2["split_hash"]
        assert m1["split_hash"] != sc["identity"]["split_hash"], "the best fit's split is fresh (seed 1)"
        assert m1["config"]["n_posterior_draws"] == 16, "best_fit.n_posterior_draws reached the runner"
        assert m1["config"]["patience"] == 10 and m1["passes"] == 1.0
        assert abs(bf["ensemble"]["L_members"][0] - m1["L"]) < 1e-9
        assert bf["ensemble"]["L_bar"] <= float(np.mean(bf["ensemble"]["L_members"])) + 1e-12, "Jensen"
        # the finalist's configuration reached the members
        fc = sc["finalist"]["config"]
        for k in ("lr", "weight_decay", "head_pool_ops", "head_fusion", "batch_size_npe"):
            v = m1["config"][{"lr": "lr", "weight_decay": "weight_decay", "head_pool_ops": "head_pool_ops",
                               "head_fusion": "head_fusion", "batch_size_npe": "b_sim"}[k]]
            assert abs(float(v) - float(fc[k])) < 1e-12, (k, v, fc[k])
        # no job submitted twice across the stop and the re-run: 2 probes + 4 trials + 2 members
        assert _n_submitted(spool) == 8, _n_submitted(spool)
        assert n_mid >= 1
        # --status after the end
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "status", "--config", path],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode == 0 and "| search" in r.stdout and "done" in r.stdout, r.stdout
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_determinism():
    """A trial killed at its walltime: resubmitted with --resume, then killed."""
    root = _tmp()
    try:
        path, c = _synthetic_project(root, search={"trial": {"walltime": "00:00:03", "max_resubmits": 2},
                                                   "max_concurrent_trials": 2, "gp_rounds": 1})
        spool = os.path.join(root, "spool")
        env = _env(spool, enforce_walltime=True)
        camp = os.path.join(c["paths"]["tune_root"], "S-A1")
        # the probes would be killed at 3 s too: pre-seed the gate with probe files
        # (the stage reuses an existing probe file instead of re-probing)
        for b_ in (16, 32):
            d = os.path.join(camp, "gate", "b%d" % b_)
            os.makedirs(d)
            with open(os.path.join(d, "A1_seed0_probe.json"), "w") as fh:
                json.dump({"step_seconds": 0.01, "val_row_seconds": 0.001, "maxrss_gb": 0.5,
                           "steps_per_epoch": 3, "projected_trial_seconds": 1.0}, fh)
        rc, out = _run(root, path, [], env, timeout=900)
        assert rc == 3, out
        # round 1 still runs (gp_rounds >= 1 by the schema): with no observation the
        # proposals are the initial design's, and they are killed in turn: 4 killed in all
        assert "STOP: stage search failed: the search ended (rounds) with no completed trial: 0 failed, 4 killed" in out, out
        assert "search: budget gate passed" in out and "gate_b16" not in out, "the probes were reused"
        recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(camp, "trials", "*.json")))]
        assert len(recs) == 4 and all(r["status"] == "killed" for r in recs), recs
        assert all(len(r["jobs"]) == 3 for r in recs), "1 submission + 2 resubmissions"
        assert out.count("resubmitted with --resume (1 of 2)") == 4 and \
            out.count("resubmitted with --resume (2 of 2)") == 4, out
        assert out.count("recorded killed after 2 resubmission(s)") == 4, out
        assert "round 0 complete (0 ok / 0 failed / 2 killed" in out and \
            "round 1 complete (0 ok / 0 failed / 4 killed" in out, out
        for r in recs:
            argv1 = open(os.path.join(camp, "jobs", r["trial_id"] + ".1.argv")).read().split("\n")
            argv0 = open(os.path.join(camp, "jobs", r["trial_id"] + ".0.argv")).read().split("\n")
            assert "--resume" in argv1 and "--resume" not in argv0
        assert not glob.glob(os.path.join(camp, "pending", "*.json"))
        assert not os.path.exists(os.path.join(camp, "SEARCH_COMPLETE.json"))
        assert _n_submitted(spool) == 4 * 3, _n_submitted(spool)
        # the fake PBS reported the kills as PBS does
        for p in glob.glob(os.path.join(spool, "*.json")):
            j = json.load(open(p))
            assert j["exit_status"] == 271 and "walltime" in j["comment"], j
        # the recovery: a longer walltime, --retry search: the killed trials are revived
        # (their records set aside, their pending specs rewritten), run, and the search ends
        cfg = json.load(open(path))
        cfg["search"]["trial"]["walltime"] = "00:10:00"
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=1, sort_keys=True)
        env2 = _env(spool, enforce_walltime=False)
        rc, out = _run(root, path, ["--retry", "search", "--accept-config-change"], env2, timeout=900)
        assert rc == 0 and "PIPELINE DONE" in out, out
        sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
        assert sc["ended_by"] == "rounds" and sc["n_trials"]["ok"] == 4 and sc["n_trials"]["killed"] == 0, sc
        assert len(glob.glob(os.path.join(camp, "trials", "*.json.killed1"))) == 4, "the killed records kept aside"
        assert len(glob.glob(os.path.join(camp, "trials", "*.json"))) == 4
        rf = json.load(open(os.path.join(camp, "pipeline_rounds.json")))
        assert rf["ended_by"] == "rounds" and [len(r["trial_ids"]) for r in rf["rounds"]] == [2, 2]
        assert _n_submitted(spool) == 4 * 3 + 4 + 2, _n_submitted(spool)   # + 4 revived trials + 2 members
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_edge_cases():
    """The elapsed clock at a tiny cap: round 0 only; S8 pre-made is skipped."""
    root = _tmp()
    try:
        path, c = _synthetic_project(root, search={"max_total_walltime_h": 1e-6,
                                                   "max_concurrent_trials": 2},
                                     pre_make_best=True)
        spool = os.path.join(root, "spool")
        env = _env(spool)
        rc, out = _run(root, path, [], env, timeout=900)
        assert rc == 0 and "PIPELINE DONE" in out, out
        assert "reached max_total_walltime_h 1e-06: no new round (D-090)" in out, out
        camp = os.path.join(c["paths"]["tune_root"], "S-A1")
        sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
        assert sc["ended_by"] == "budget" and sc["rounds_evaluated"] == 1 and sc["n_trials"]["ok"] == 2, sc
        assert sc["elapsed_h"] > 1e-6
        assert "best_fit: skipped, artefact of record exists" in out, out
        assert _n_submitted(spool) == 2 + 2, _n_submitted(spool)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ main ---

CHECKS = [("test_known_answer", test_known_answer), ("test_reference", test_reference),
          ("test_convergence", test_convergence), ("test_invariants", test_invariants),
          ("test_contract", test_contract), ("test_determinism", test_determinism),
          ("test_edge_cases", test_edge_cases)]


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
    for d in list(_SHARED.values()):
        if isinstance(d, str) and os.path.isdir(d) and "banks_" in d:
            shutil.rmtree(d, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
