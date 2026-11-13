"""Shared helpers of the spec-derived tests of blocks 11-17 (independent
tester, run t1-B of the 2026-11-13 split audit, D-131).

Nothing here computes an expected value: the helpers locate the code, run the
builder and the pipeline as their users do (a subprocess, the fake PBS of
hpc/pipeline/tests/fake_pbs, which SPEC.md sec. 3.3 describes as the tests'
scheduler), and write reduced configurations. The expected values live in the
test files, computed from SPEC.md, from NumPy/SciPy, or by hand.

Older code trees (the reference for the spec's "unchanged draw for draw"
oracles) are extracted with `git archive` from this repository's own history
into a temporary folder; the commits are named in the tests.

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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HPC = os.path.join(REPO, "hpc")
PIPE = os.path.join(HPC, "pipeline")
JOINT = os.path.join(HPC, "joint")
STAGE1 = os.path.join(JOINT, "stage1")
STAGE2 = os.path.join(JOINT, "stage2")
STAGE3 = os.path.join(JOINT, "stage3")
STAGE4 = os.path.join(JOINT, "stage4")
DSN = os.path.join(HPC, "dsn")
FAKE_PBS = os.path.join(PIPE, "tests", "fake_pbs")
SYNTH = os.path.join(PIPE, "projects", "synth", "synth.json")
GIULIA = os.path.join(PIPE, "projects", "giulia", "giulia.json")
PYTHON = sys.executable
BUILDER = os.path.join(STAGE1, "build_latent_bank.py")
RECORD_TOOL = os.path.join(PIPE, "tools", "synth_bank_record.py")
RUNNER = os.path.join(STAGE3, "run_joint_arms.py")

# The commits the "unchanged" oracles compare against (this repository's history):
# 59465e1 is "blocks 11-14" (its parent is the code before blocks 11-14);
# 44f812b is "blocks 25-26" (its parent is the code of blocks 11-24).
PRE_BLOCK11 = "59465e1^"
PRE_BLOCK25 = "44f812b^"


def add_paths(*paths):
    for p in paths:
        p = os.path.abspath(p)
        if p not in sys.path:
            sys.path.insert(0, p)


def pipeline_path():
    """The pipeline's modules import each other as top-level modules."""
    add_paths(PIPE)


def joint_paths():
    add_paths(JOINT, STAGE2, STAGE1, STAGE3)


def strip_docs(obj):
    """Documentation keys (sec. 3.1: a key starting with '_' is ignored),
    written here from the spec's rule, not taken from config.py."""
    if isinstance(obj, dict):
        return {k: strip_docs(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [strip_docs(v) for v in obj]
    return obj


def committed_synth():
    with open(SYNTH, "rb") as fh:
        return strip_docs(json.loads(fh.read().decode("ascii")))


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_TREES = {}


def old_tree(rev):
    """hpc/joint, hpc/dsn and hpc/npe_diagnostics.py of `rev`, extracted once
    per process into a temporary folder; returns the folder (its hpc/ inside)."""
    if rev in _TREES:
        return _TREES[rev]
    d = tempfile.mkdtemp(prefix="t1b_tree_")
    arc = subprocess.run(["git", "-C", REPO, "archive", rev, "hpc/joint", "hpc/dsn",
                          "hpc/npe_diagnostics.py"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, check=True)
    subprocess.run(["tar", "-x", "-C", d], input=arc.stdout, check=True)
    _TREES[rev] = d
    return d


def run_builder(out_dir, args, tree=None, timeout=600):
    """build_latent_bank.py as a job runs it: `python <builder> --out-dir <d>
    <args>` from its own folder. Returns (returncode, output)."""
    root = tree or REPO
    stage1 = os.path.join(root, "hpc", "joint", "stage1")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    r = subprocess.run([PYTHON, os.path.join(stage1, "build_latent_bank.py"), "--out-dir", out_dir]
                       + [str(a) for a in args], cwd=stage1, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout, env=env)
    return r.returncode, r.stdout


def load_shard(path):
    """(arrays, sidecar) read with numpy and json directly (not the code's reader)."""
    import numpy as np
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    with open(os.path.splitext(path)[0] + ".json") as fh:
        side = json.load(fh)
    return arrays, side


def fake_env(spool, extra=None):
    """The fake PBS (sec. 3.3) for this process and its children; returns the
    previous values so that a test can restore them."""
    old = dict(os.environ)
    env = {"PATH": FAKE_PBS + os.pathsep + os.environ.get("PATH", ""),
           "FAKE_PBS_SPOOL": spool, "FAKE_PBS_DELAY": "0.2", "FAKE_PBS_ENFORCE_WALLTIME": "0",
           "PIPELINE_SKIP_CONDA": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": PIPE + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for k in ("FAKE_PBS_FAIL_START", "FAKE_PBS_HOSTS", "FAKE_PBS_HOST_BY_NAME"):
        os.environ.pop(k, None)
    env.update(extra or {})
    os.environ.update(env)
    return old


def restore_env(old):
    os.environ.clear()
    os.environ.update(old)


def spool_records(spool):
    """The fake PBS's job records (<num>.json), in submission order."""
    out = []
    for p in glob.glob(os.path.join(spool, "*.json")):
        if os.path.basename(p)[:-5].isdigit():
            with open(p) as fh:
                out.append(json.load(fh))
    return sorted(out, key=lambda r: r["num"])


def reduced_synth(root, searches=None, real=None, sim=None, **over):
    """A reduced copy of the committed synth.json in `root` (the bench at
    f_s 10 Hz and T_win 12.8 s, so W = 128; small banks; a pinned search
    space; 1-core jobs). Every value changed is listed here; the rest is the
    committed file's. Returns (path, dict)."""
    d = committed_synth()
    for k in list(d["paths"]):
        d["paths"][k] = os.path.join(root, k)
    d["paths"]["sbi_dir"] = REPO
    d["synthetic"]["builder"].update({"fs": 10.0, "T_win": 12.8})
    d["synthetic"]["sim"] = sim or {"seed": 101, "n_theta": 12, "n_windows": 1,
                                    "traces_per_shard": 8}
    d["synthetic"]["real"] = real or [
        {"name": "R0", "seed": 202, "donors_per_class": 2, "n_windows": 2, "pi": 0.0,
         "gap_modes": ["range_shift", "drift", "contamination"]},
        {"name": "R3", "seed": 303, "donors_per_class": 2, "n_windows": 2, "pi": 0.3,
         "gap_modes": ["range_shift", "drift", "contamination"]}]
    d["synthetic"]["searches"] = searches or [
        {"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False}]
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
    for k, v in over.items():
        cur = d
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = v
    path = os.path.join(root, "synth.json")
    write_json(path, d)
    return path, d


def tmpdir(prefix="t1b_"):
    return tempfile.mkdtemp(prefix=prefix)


def rmtree(path):
    shutil.rmtree(path, ignore_errors=True)


def deepcopy(x):
    return copy.deepcopy(x)
