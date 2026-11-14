"""Chain C3 (the evaluation) of the split audit D-131, run t2-C3: the fixture
shared by tests/spec/test_spec_chain_C3_*.py. Not a test file.

What it builds (once per process, or reused from C3_FIXTURE=<dir>, which a
first build prints), all under one temporary folder:

  1. the producer side (chain C2's outputs, the inputs of C3): a reduced
     synthetic project shaped like the committed synth.json v4 (label axes
     0-7, the centre scale 0.562743, the term-only campaigns) at W 128 (f_s
     10 Hz, T_win 12.8 s), with four searches --
         S-A1      campaign S-A1      on R0  (alt_best_fit false)
         S-A2-R0   campaign S-A2-only on R0  (alt_best_fit false; block 26)
         S-A5-R3   campaign S-A5-only on R3  (alt_best_fit false; block 26)
         S-A2x-R0  campaign S-A2      on R0  (alt_best_fit true;  block 18)
     -- run by the pipeline's own sequencer (`pipeline.py run`) to PIPELINE
     DONE under the fake PBS (sec. 3.3), so every artefact the evaluation
     reads was written by its real producer (blocks 7, 16, 18);
  2. chain run A: `submit_eval.sh --config ... --eval-config ... --what
     calbank,sample,report,data` once (the real entry point, block 22), every
     job through `eval_run.pbs` under the fake PBS, then the same command
     again (resume: nothing to do);
  3. chain run B, the same values in another out_root, interrupted at the
     stage boundaries: the calibration bank alone; the samplings with one
     queued sampling deleted (qdel); the rest with the report; a resampling
     killed while it runs; the data overview; then the full command again;
  4. scenarios on run A's tree, each undone afterwards: a best fit made
     incomplete (W1), a member's checkpoint replaced by another member's
     (W2), the samplings asked for while the calibration bank's jobs are still
     queued (W3, a fresh out_root), a cohort bank rebuilt after the sampling
     (W4), a calibration bank of another build (n_neurons 50) in a fresh
     out_root (W5).

Every submission's output, every fake-PBS job record and the digests the
tests need are written to <root>/chain_c3.json; the tests read that file and
the trees. Values derived from SPEC.md only; the code under test is run, never
asked for an expected value.

Run (from the repository root):
    python3 -m pytest tests/spec -k chain_C3 -q          # builds the fixture (about 20 min)
    C3_FIXTURE=<dir> python3 -m pytest tests/spec -k chain_C3 -q   # reuses a built one
    python3 tests/spec/chain_c3_support.py               # builds it and prints its folder

Needs the training environment (torch, sbi 0.27, zuko 1.6, scikit-optimize,
pytorch-metric-learning < 3, numpy, scipy, scikit-learn, matplotlib). Nothing
touches davinci; the roadmap's markers are off (ROADMAP_DISABLE=1) and a git
push to github.com is made to fail inside the jobs.

Pure ASCII, LF only.
"""

import copy
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
EVAL = os.path.join(PIPE, "evaluation")
FAKE = os.path.join(PIPE, "tests", "fake_pbs")
SYNTH = os.path.join(PIPE, "projects", "synth", "synth.json")
SYNTH_EVAL = os.path.join(PIPE, "projects", "synth", "synth_eval.json")
SUBMIT = os.path.join(EVAL, "submit_eval.sh")
BUILDER = os.path.join(REPO, "hpc", "joint", "stage1", "build_latent_bank.py")
PYTHON = sys.executable
RECORD = "chain_c3.json"

SEARCHES = [
    {"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False},
    {"name": "S-A2-R0", "campaign": "S-A2-only", "real": "R0", "alt_best_fit": False},
    {"name": "S-A5-R3", "campaign": "S-A5-only", "real": "R3", "alt_best_fit": False},
    {"name": "S-A2x-R0", "campaign": "S-A2", "real": "R0", "alt_best_fit": True},
]
REAL = [
    {"name": "R0", "seed": 22, "donors_per_class": 2, "n_windows": 2, "pi": 0.0,
     "gap_modes": ["range_shift", "drift", "contamination"]},
    {"name": "R3", "seed": 34, "donors_per_class": 2, "n_windows": 2, "pi": 0.3,
     "gap_modes": ["range_shift", "drift", "contamination"]},
]
SIM = {"seed": 11, "n_theta": 60, "n_windows": 1, "traces_per_shard": 40}

_CACHE = {}


# ------------------------------------------------------------------ helpers

def strip_docs(obj):
    if isinstance(obj, dict):
        return dict((k, strip_docs(v)) for k, v in obj.items() if not str(k).startswith("_"))
    if isinstance(obj, list):
        return [strip_docs(v) for v in obj]
    return obj


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(d):
    """{relative path: sha256} of every regular file under d."""
    out = {}
    for a, _b, fs in os.walk(d):
        for f in fs:
            p = os.path.join(a, f)
            if os.path.isfile(p) and not os.path.islink(p):
                out[os.path.relpath(p, d)] = sha256_file(p)
    return out


def project_config(root):
    """The reduced project (dict): the committed synth.json v4 with its
    paths under `root`, the checkout as sbi_dir, and the sizes cut down. The
    builder's label axes and centre scale, the split, the flow settings and
    every value not named below are the committed file's."""
    with open(SYNTH, "rb") as fh:
        d = strip_docs(json.loads(fh.read().decode("ascii")))
    for k in list(d["paths"]):
        d["paths"][k] = os.path.join(root, k)
    d["paths"]["sbi_dir"] = REPO
    d["synthetic"]["builder"].update({"fs": 10.0, "T_win": 12.8})
    d["synthetic"]["sim"] = dict(SIM)
    d["synthetic"]["real"] = copy.deepcopy(REAL)
    d["synthetic"]["searches"] = copy.deepcopy(SEARCHES)
    d["synthetic"]["bank_job"] = {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00",
                                  "queue": "test", "max_resubmits": 1}
    d["search"].update({
        "shape_anchors": {"p": 10, "d_theta": 10, "embedding_dim": 8},
        "space_overrides": {"depth_exponent": [3, 3], "width_multiplier": [1.5, 1.5],
                            "embedding_size": [8, 8], "hidden_features": [24, 24],
                            "num_transforms": [3, 3], "batch_size_npe": [16, 32],
                            "head_pool_ops": [0, 1]},
        "n_initial_points": 2, "gp_rounds": 1, "trials_per_round": 1,
        "max_concurrent_trials": 4, "gp_seed": 3, "train_seed": 0,
        "budget_gate": {"probe_steps": 2, "probe_val_rows": 10, "safety_factor": 1.5}})
    d["search"]["trial"].update({"passes": 1.0, "epochs": 2, "ncpus": 1, "mem": "4gb",
                                 "walltime": "00:10:00", "queue": "test", "max_resubmits": 1})
    d["best_fit"].update({"member_seeds": [1, 2], "passes": 1.0, "epochs": 2,
                          "n_posterior_draws": 16, "ncpus": 1, "mem": "4gb",
                          "walltime": "00:10:00", "queue": "test", "max_resubmits": 1})
    d["loader"] = {"buffer_shards": 3}
    d["driver"] = {"queue": "test", "ncpus": 1, "mem": "1gb", "walltime": "01:00:00",
                   "poll_seconds": 1}
    d["resources"] = {"max_cores": 5}
    return d


def eval_config(out_root, **over):
    """The evaluation's values: the committed synth_eval.json's keys and its
    oracle_tol (2e-3), sizes cut down to the reduced project."""
    with open(SYNTH_EVAL, "rb") as fh:
        committed = strip_docs(json.loads(fh.read().decode("ascii")))
    d = {"out_root": out_root, "seed": 4242, "n_post": 40, "batch_rows": 16,
         "oracle_tol": committed["oracle_tol"], "ppc_rows": 3, "ppc_draws": 4,
         "cal_glob": os.path.join(out_root, "calbank", "shard_*.npz"), "cal_n": 20,
         "cal_shards": 2, "cal_seed": 55, "lc2st_null": 5, "lc2st_eval_draws": 10,
         "bootstrap_n": 200, "level": 0.95, "dpi": 40, "formats": ["png"], "ncpus": 2,
         "jobs": {"queue": "test", "calbank": {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00"},
                  "sample": {"ncpus": 2, "mem": "2gb", "walltime": "00:20:00"},
                  "report": {"ncpus": 2, "mem": "2gb", "walltime": "00:30:00"},
                  "data": {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00"}},
         "data_view": {"sim_shards": 2, "examples_per_condition": 1, "sim_examples": 3,
                       "zoom_s": 4.0, "bins": 12}}
    assert sorted(d) == sorted(committed), "the committed keys changed: %s" % sorted(committed)
    d.update(over)
    return d


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    return path


def fake_env(spool, delay="0.2", extra=None):
    """The fake PBS on PATH for the launcher and (the fake hands its own
    environment to its jobs) for every job; conda skipped, the roadmap's
    markers off, a push to github.com made to fail."""
    os.makedirs(spool, exist_ok=True)
    env = dict(os.environ)
    env.update({"PATH": FAKE + os.pathsep + env.get("PATH", "").replace(FAKE + os.pathsep, ""),
                "FAKE_PBS_SPOOL": spool, "FAKE_PBS_DELAY": str(delay),
                "FAKE_PBS_ENFORCE_WALLTIME": "0", "SKIP_CONDA": "1", "PIPELINE_SKIP_CONDA": "1",
                "ROADMAP_DISABLE": "1", "OMP_NUM_THREADS": "1",
                "PIPELINE_QSTAT_RETRY_WAIT_S": "0.2",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "url./nonexistent-push-guard/.pushInsteadOf",
                "GIT_CONFIG_VALUE_0": "https://github.com/"})
    env.pop("FAKE_PBS_FAIL_START", None)
    env.update(extra or {})
    return env


def records(spool):
    out = []
    for p in glob.glob(os.path.join(spool, "*.json")):
        if os.path.basename(p)[:-5].isdigit():
            with open(p) as fh:
                out.append(json.load(fh))
    return sorted(out, key=lambda r: r["num"])


def wait_all(spool, timeout=3600, n_min=0):
    """Wait until every job of `spool` has finished; returns the records."""
    t0 = time.time()
    while True:
        recs = records(spool)
        if len(recs) >= n_min and all(r.get("finished") for r in recs):
            return recs
        if time.time() - t0 > timeout:
            raise RuntimeError("jobs still running after %d s: %s"
                               % (timeout, [(r["name"], r["state"]) for r in recs
                                            if not r.get("finished")]))
        time.sleep(2)


def submit(cfg_path, ecfg_path, env, *args, timeout=900):
    """The wrapper (the chain's entry point) with --config/--eval-config;
    returns dict(argv, rc, out)."""
    argv = ["bash", SUBMIT, "--config", cfg_path, "--eval-config", ecfg_path] + list(args)
    r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, env=env, timeout=timeout, cwd=REPO)
    return {"argv": argv[2:], "rc": r.returncode, "out": r.stdout}


def qdel(env, job_id):
    r = subprocess.run(["qdel", job_id], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, env=env)
    return {"job": job_id, "rc": r.returncode, "out": r.stdout}


def ledger(ecfg_dict):
    p = os.path.join(ecfg_dict["out_root"], "jobs", "ledger.json")
    with open(p) as fh:
        return json.load(fh)


def job_log_tails(recs, n=4000):
    out = {}
    for r in recs:
        p = r.get("out")
        txt = ""
        if p and os.path.isfile(p):
            with open(p, errors="replace") as fh:
                txt = fh.read()
        out[str(r["num"])] = txt[-n:]
    return out


# ------------------------------------------------------------------ phases

def build_producer(root, rec, log):
    """Phase P: the reduced project through the pipeline's sequencer."""
    cfg = project_config(root)
    cfg_path = write_json(os.path.join(root, "c3.json"), cfg)
    spool = os.path.join(root, "spool_pipe")
    env = fake_env(spool, extra={"PYTHONPATH": PIPE + os.pathsep + os.environ.get("PYTHONPATH", "")})
    t0 = time.time()
    r = subprocess.run([PYTHON, os.path.join(PIPE, "pipeline.py"), "run", "--config", cfg_path,
                        "--poll-seconds", "0.5"], cwd=root, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=5400)
    rec["producer"] = {"rc": r.returncode, "tail": r.stdout[-8000:], "seconds": time.time() - t0,
                       "n_jobs": len(records(spool))}
    log("[c3] producer: exit %d in %.0f s, %d jobs" % (r.returncode, time.time() - t0,
                                                       len(records(spool))))
    if r.returncode != 0 or "PIPELINE DONE" not in r.stdout:
        raise RuntimeError("the reduced project did not reach PIPELINE DONE:\n" + r.stdout[-6000:])
    rec["cfg"] = cfg_path
    rec["work_root_before"] = tree_digest(cfg["paths"]["work_root"])
    return cfg_path


def run_A(root, cfg_path, rec, log):
    """Phase A: the whole evaluation in one submission, then again."""
    out = os.path.join(root, "evalA")
    e = eval_config(out)
    ecfg = write_json(os.path.join(root, "evalA.json"), e)
    spool = os.path.join(root, "spool_A")
    env = fake_env(spool)
    A = {"ecfg": ecfg, "out_root": out, "spool": spool, "submissions": []}
    t0 = time.time()
    s = submit(cfg_path, ecfg, env, "--what", "calbank,sample,report,data")
    A["submissions"].append(s)
    log("[c3] A1: exit %d; %s" % (s["rc"], s["out"].strip().splitlines()[-1:]))
    recs = wait_all(spool, timeout=5400)
    A["jobs"] = recs
    A["logs"] = job_log_tails(recs)
    A["seconds"] = time.time() - t0
    log("[c3] A: %d jobs finished in %.0f s: %s" % (len(recs), time.time() - t0,
                                                    [(r["name"], r["exit_status"]) for r in recs]))
    s2 = submit(cfg_path, ecfg, env, "--what", "calbank,sample,report,data")
    A["submissions"].append(s2)
    A["n_jobs_after_rerun"] = len(records(spool))
    A["ledger"] = ledger(e)
    A["digest"] = tree_digest(out)
    rec["A"] = A
    rec["work_root_after_A"] = tree_digest(json.load(open(cfg_path))["paths"]["work_root"])
    return A


def _wait_state(spool, num, states, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for r in records(spool):
            if r["num"] == num and r["state"] in states:
                return r
        time.sleep(0.2)
    raise RuntimeError("job %d never reached %s" % (num, states))


def run_B(root, cfg_path, rec, log):
    """Phase B: the same values in another out_root, interrupted at every
    stage boundary."""
    out = os.path.join(root, "evalB")
    e = eval_config(out)
    ecfg = write_json(os.path.join(root, "evalB.json"), e)
    spool = os.path.join(root, "spool_B")
    B = {"ecfg": ecfg, "out_root": out, "spool": spool, "steps": []}

    def step(name, args, delay="0.2", wait=True):
        env = fake_env(spool, delay=delay)
        n0 = len(records(spool))
        s = submit(cfg_path, ecfg, env, *args)
        s["name"] = name
        s["new_jobs"] = [r for r in records(spool)][n0:]
        B["steps"].append(s)
        log("[c3] B %s: exit %d; %s" % (name, s["rc"], s["out"].strip().splitlines()[-1:]))
        if wait:
            wait_all(spool, timeout=5400)
        return s, env

    # B1: the calibration bank alone
    step("B1_calbank", ["--what", "calbank"])
    # B2: the samplings, kept queued 25 s; the first one deleted while queued
    s, env = step("B2_samples_queued", ["--what", "calbank,sample"], delay="25", wait=False)
    led = ledger(e)
    first = sorted(k for k in led if k.startswith("sample:"))[0]
    s["qdel"] = qdel(env, led[first]["job_id"])
    s["qdel_unit"] = first
    # while they wait: the report asked for with the samplings live
    s3, _env = step("B3_report_while_live", ["--what", "calbank,sample,report", "--dry-run"],
                    wait=False)
    wait_all(spool, timeout=5400)
    # B4: the rest, with the report
    step("B4_rest_and_report", ["--what", "calbank,sample,report"])
    # B5: a resampling killed while it runs, then the plan
    led = ledger(e)
    samples = os.path.join(out, "samples")
    before = dict((os.path.basename(p), (os.path.getmtime(p), sha256_file(p)))
                  for p in glob.glob(os.path.join(samples, "S-A1_A1_*")))
    env = fake_env(spool, delay="0.2")
    n0 = len(records(spool))
    s = submit(cfg_path, ecfg, env, "--what", "sample", "--resample", "S-A1:A1")
    s["name"] = "B5_resample_killed"
    new = records(spool)[n0:]
    s["new_jobs"] = new
    job = [r for r in new if r["name"].startswith("ev_s")][0]
    # wait until it has rewritten its first file, then kill it (qdel of a running job)
    t0 = time.time()
    first_new = None
    while time.time() - t0 < 900:
        for p in glob.glob(os.path.join(samples, "S-A1_A1_*.npz")):
            b = before.get(os.path.basename(p))
            if b is None or os.path.getmtime(p) > b[0]:
                first_new = os.path.basename(p)
        if first_new:
            break
        time.sleep(0.05)
    s["first_rewritten"] = first_new
    s["qdel"] = qdel(env, job["id"])
    wait_all(spool, timeout=900)
    after = dict((os.path.basename(p), (os.path.getmtime(p), sha256_file(p)))
                 for p in glob.glob(os.path.join(samples, "S-A1_A1_*")))
    s["files_before"] = before
    s["files_after"] = after
    s["job_after"] = [r for r in records(spool) if r["num"] == job["num"]][0]
    B["steps"].append(s)
    log("[c3] B5: killed %s after %s was rewritten" % (job["id"], first_new))
    dry = submit(cfg_path, ecfg, env, "--what", "calbank,sample,report", "--dry-run")
    dry["name"] = "B6_plan_after_kill"
    B["steps"].append(dry)
    # B7: the data overview; B8: everything again
    step("B7_data", ["--what", "data"])
    step("B8_all_again", ["--what", "calbank,sample,report,data"])
    B["jobs"] = records(spool)
    B["logs"] = job_log_tails(B["jobs"])
    B["ledger"] = ledger(e)
    B["digest"] = tree_digest(out)
    rec["B"] = B
    return B


def scenario_W1(root, cfg_path, rec, log):
    """A best fit made incomplete (its BEST_FIT_COMPLETE.json moved aside):
    the plan, then restored."""
    A = rec["A"]
    cfg = json.load(open(cfg_path))
    folder = os.path.join(cfg["paths"]["runs_root"], "S-A5-R3", cfg["dataset"] + "_best")
    bfc = os.path.join(folder, "BEST_FIT_COMPLETE.json")
    keep = bfc + ".c3aside"
    os.rename(bfc, keep)
    try:
        env = fake_env(A["spool"])
        s = submit(cfg_path, A["ecfg"], env, "--what", "calbank,sample,report", "--dry-run")
    finally:
        os.rename(keep, bfc)
    rec["W1"] = s
    log("[c3] W1: exit %d" % s["rc"])


def scenario_W2(root, cfg_path, rec, log):
    """A member's checkpoint replaced by the other member's (a wrong
    upstream artefact): the resampling must refuse and the report after it
    must not run; then the checkpoint restored."""
    A = rec["A"]
    cfg = json.load(open(cfg_path))
    folder = os.path.join(cfg["paths"]["runs_root"], "S-A2-R0", cfg["dataset"] + "_best")
    c1 = glob.glob(os.path.join(folder, "member_seed1", "*_ckpt.pt"))[0]
    c2 = glob.glob(os.path.join(folder, "member_seed2", "*_ckpt.pt"))[0]
    keep = c2 + ".c3aside"
    out = A["out_root"]
    rc_path = os.path.join(out, "REPORT_COMPLETE.json")
    W = {"report_sha_before": sha256_file(rc_path), "ckpt": c2,
         "samples_before": dict((os.path.basename(p), sha256_file(p))
                                for p in glob.glob(os.path.join(out, "samples", "*")))}
    shutil.copy2(c2, keep)
    shutil.copy2(c1, c2)
    spool = A["spool"]
    env = fake_env(spool)
    n0 = len(records(spool))
    try:
        s = submit(cfg_path, A["ecfg"], env, "--resample", "S-A2-R0:A2")
        recs = wait_all(spool, timeout=1800)
        W["submission"] = s
        W["jobs"] = recs[n0:]
        W["logs"] = job_log_tails(recs[n0:])
    finally:
        os.replace(keep, c2)
    W["report_sha_after"] = sha256_file(rc_path)
    W["samples_after"] = dict((os.path.basename(p), sha256_file(p))
                              for p in glob.glob(os.path.join(out, "samples", "*")))
    W["plan_after_restore"] = submit(cfg_path, A["ecfg"], env, "--dry-run")
    rec["W2"] = W
    log("[c3] W2: %s" % [(r["name"], r["exit_status"]) for r in W["jobs"]])


def scenario_W3(root, cfg_path, rec, log):
    """The samplings asked for (--what sample) while the calibration bank's
    jobs are still queued, against the same with --what calbank,sample."""
    out = os.path.join(root, "evalC")
    e = eval_config(out)
    ecfg = write_json(os.path.join(root, "evalC.json"), e)
    spool = os.path.join(root, "spool_C")
    env = fake_env(spool, delay="3000")
    W = {"calbank": submit(cfg_path, ecfg, env, "--what", "calbank")}
    W["sample_only"] = submit(cfg_path, ecfg, env, "--what", "sample", "--dry-run")
    W["calbank_sample"] = submit(cfg_path, ecfg, env, "--what", "calbank,sample", "--dry-run")
    W["calbank_jobs"] = records(spool)
    # the ledger's bytes lost (a write a node fault left empty, cf. t1-D's F701): the plan
    lp = os.path.join(out, "jobs", "ledger.json")
    with open(lp, "rb") as fh:
        keep = fh.read()
    with open(lp, "wb") as fh:
        fh.write(b"")
    try:
        W["plan_empty_ledger"] = submit(cfg_path, ecfg, env, "--what", "calbank", "--dry-run")
    finally:
        with open(lp, "wb") as fh:
            fh.write(keep)
    for r in records(spool):
        qdel(env, r["id"])
    wait_all(spool, timeout=120)
    rec["W3"] = W
    log("[c3] W3: done")


def scenario_W4(root, cfg_path, rec, log):
    """A cohort bank (R3) rebuilt at another seed after the sampling, the
    original folder kept aside and put back afterwards: what each done
    predicate says."""
    A = rec["A"]
    cfg = json.load(open(cfg_path))
    bdir = os.path.join(cfg["paths"]["banks_root"], "%s_real_R3" % cfg["dataset"])
    aside = bdir + ".c3aside"
    W = {}
    side = json.load(open(os.path.join(bdir, "shard_0000.json")))
    argv = [str(a) for a in side["provenance"]["argv"]]
    if "--out-dir" in argv:
        i = argv.index("--out-dir")
        del argv[i:i + 2]
    argv[argv.index("--seed") + 1] = "35"
    os.rename(bdir, aside)
    try:
        os.makedirs(bdir)
        r = subprocess.run([PYTHON, BUILDER, "--out-dir", bdir] + argv, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=600,
                           cwd=os.path.dirname(BUILDER))
        W["rebuild"] = {"rc": r.returncode, "tail": r.stdout[-2000:], "argv": argv}
        env = fake_env(A["spool"])
        W["plan"] = submit(cfg_path, A["ecfg"], env, "--what", "calbank,sample,report,data",
                           "--dry-run")
    finally:
        shutil.rmtree(bdir, ignore_errors=True)
        os.rename(aside, bdir)
    W["plan_restored"] = submit(cfg_path, A["ecfg"], fake_env(A["spool"]),
                                "--what", "calbank,sample,report,data", "--dry-run")
    rec["W4"] = W
    log("[c3] W4: rebuild exit %d" % W["rebuild"]["rc"])


def scenario_W5(root, cfg_path, rec, log):
    """A calibration bank of another build (n_neurons 50 where the project's
    builder says 100, every other argument the build's) in a fresh out_root:
    the plan with and without `calbank` in --what, then the sampler itself."""
    out = os.path.join(root, "evalW5")
    e = eval_config(out)
    ecfg = write_json(os.path.join(root, "evalW5.json"), e)
    cal_dir = os.path.join(out, "calbank")
    os.makedirs(cal_dir)
    W = {"builds": []}
    # the build's arguments (ev_plan.calbank_build's, the launcher's argv files are
    # not there yet): read from a dry run's command lines
    env = fake_env(os.path.join(root, "spool_W5"))
    dry = submit(cfg_path, ecfg, env, "--what", "calbank", "--dry-run")
    W["dry_calbank"] = dry
    cmds = [l.split("command: ", 1)[1].split() for l in dry["out"].splitlines()
            if "command: " in l and "build_latent_bank.py" in l]
    for c in cmds:
        argv = c[c.index("--out-dir") + 2:]
        i = argv.index("--n-neurons")
        argv[i + 1] = "50"
        r = subprocess.run([PYTHON, BUILDER, "--out-dir", cal_dir] + argv, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=600,
                           cwd=os.path.dirname(BUILDER))
        W["builds"].append({"argv": argv, "rc": r.returncode, "tail": r.stdout[-1500:]})
    W["plan_sample_only"] = submit(cfg_path, ecfg, env, "--what", "sample", "--dry-run")
    W["plan_calbank_sample"] = submit(cfg_path, ecfg, env, "--what", "calbank,sample", "--dry-run")
    r = subprocess.run([PYTHON, os.path.join(EVAL, "ev_sample.py"), "--config", cfg_path,
                        "--eval-config", ecfg, "--ensemble", "S-A2-R0:A2", "--ncpus", "1"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       env=fake_env(os.path.join(root, "spool_W5")), timeout=1800, cwd=REPO)
    W["sampler"] = {"rc": r.returncode, "tail": r.stdout[-3000:]}
    rec["W5"] = W
    log("[c3] W5: sampler exit %d" % r.returncode)


def build(root=None, log=print):
    root = os.path.abspath(root) if root else tempfile.mkdtemp(prefix="c3_chain_")
    os.makedirs(root, exist_ok=True)
    rec = {"root": root, "started": time.time()}
    cfg_path = build_producer(root, rec, log)
    write_json(os.path.join(root, RECORD), rec)
    run_A(root, cfg_path, rec, log)
    write_json(os.path.join(root, RECORD), rec)
    for name, fn in (("B", run_B), ("W1", scenario_W1), ("W3", scenario_W3),
                     ("W4", scenario_W4), ("W5", scenario_W5),
                     ("W2", scenario_W2)):           # W2 last: it writes into run A's tree
        try:
            fn(root, cfg_path, rec, log)
        except Exception:                            # recorded; that phase's tests then fail
            import traceback
            rec[name + "_error"] = traceback.format_exc()
            log("[c3] %s raised:\n%s" % (name, rec[name + "_error"]))
        write_json(os.path.join(root, RECORD), rec)
    rec["finished"] = time.time()
    rec["work_root_after_all"] = tree_digest(json.load(open(cfg_path))["paths"]["work_root"])
    write_json(os.path.join(root, RECORD), rec)
    log("[c3] fixture built in %.0f s: %s" % (rec["finished"] - rec["started"], root))
    return root


def fixture():
    """The built fixture: dict(root, rec, cfg, ...). Skips (pytest) when the
    training environment is missing."""
    if "root" in _CACHE:
        return _CACHE
    import importlib
    try:
        for mod in ("torch", "sbi", "skopt", "zuko"):
            importlib.import_module(mod)
    except ImportError as exc:
        import pytest
        pytest.skip("the training environment is needed (%s)" % exc)
    given = os.environ.get("C3_FIXTURE", "").strip()
    root = os.path.abspath(given) if given else build()
    with open(os.path.join(root, RECORD)) as fh:
        rec = json.load(fh)
    if "finished" not in rec:
        raise RuntimeError("%s holds an unfinished fixture" % root)
    _CACHE.update(root=root, rec=rec, cfg=rec["cfg"])
    return _CACHE


if __name__ == "__main__":
    r = build(sys.argv[1] if len(sys.argv) > 1 else None)
    print(r)
