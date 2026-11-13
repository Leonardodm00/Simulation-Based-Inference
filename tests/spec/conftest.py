"""Shared set-up of the spec-derived tests (integration tester, run t1-C:
blocks 18-23 of specs/SPEC.md). Written from SPEC.md; the code under test is
only called, never used to compute an expected value.

The end-to-end tests need a reduced synthetic project run to PIPELINE DONE
(W 128, S-A1 and S-A2-R0 with alt_best_fit true, two members per ensemble):
the project the evaluation's smoke test builds. Either point
SPEC_EVAL_FIXTURE at a kept tree
    EVAL_KEEP=1 python3 hpc/pipeline/tests/test_smoke_evaluation.py test_e19a_rebuild
    SPEC_EVAL_FIXTURE=<the printed dir> python3 -m pytest tests/spec -q
or let the session build one (about 2 minutes; needs torch, sbi, zuko,
scikit-optimize, pytorch-metric-learning < 3). The tests write their own
evaluation under <fixture>/spec_eval_t1c (a fresh folder per session).

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

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HPC = os.path.join(REPO, "hpc")
PIPE = os.path.join(HPC, "pipeline")
EVAL = os.path.join(PIPE, "evaluation")
PY = sys.executable

if EVAL not in sys.path:
    sys.path.insert(0, EVAL)
import _evalpath  # noqa: E402,F401  (the pipeline's config first, then the joint stack)


def training_env():
    """True when torch, sbi, zuko and skopt import."""
    try:
        import torch  # noqa: F401
        import sbi  # noqa: F401
        import zuko  # noqa: F401
        import skopt  # noqa: F401
    except ImportError:
        return False
    return True


def run_py(args, cwd=REPO, env=None, timeout=1800):
    """(returncode, stdout+stderr) of `python3 <args>`."""
    r = subprocess.run([PY] + list(args), cwd=cwd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return r.returncode, r.stdout


def files_under(d):
    if not os.path.isdir(d):
        return []
    return sorted(os.path.relpath(os.path.join(a, f), d) for a, _b, fs in os.walk(d) for f in fs)


def seed_of_ref(base, *parts):
    """SPEC block 19 names seed_of(seed, D) without defining it; this is the
    text the module documents (sha256 of "<base>|<parts joined by |>", the
    first 32 bits), restated here so that the examples' draws can be redone
    without calling the code under test."""
    h = hashlib.sha256(("%d|" % int(base) + "|".join(str(p) for p in parts)).encode("ascii"))
    return int(h.hexdigest()[:8], 16)


# ------------------------------------------------------------ the fixture

_STATE = {}


def _build_fixture():
    sys.path.insert(0, os.path.join(PIPE, "tests"))
    import test_smoke_evaluation as TE       # the reduced project's builder (context only)
    root = tempfile.mkdtemp(prefix="spec_t1c_fixture_")
    TE._build_project(root)
    return root


@pytest.fixture(scope="session")
def project():
    """dict(root, cfg): the reduced synthetic project run to PIPELINE DONE."""
    if not training_env():
        pytest.skip("the training environment (torch, sbi, zuko, skopt) is needed")
    if "project" in _STATE:
        return _STATE["project"]
    root = os.environ.get("SPEC_EVAL_FIXTURE", "").strip()
    if root:
        root = os.path.abspath(root)
        if not os.path.isfile(os.path.join(root, "synth.json")):
            pytest.skip("SPEC_EVAL_FIXTURE=%s has no synth.json" % root)
    else:
        root = _build_fixture()
    _STATE["project"] = {"root": root, "cfg": os.path.join(root, "synth.json")}
    return _STATE["project"]


def eval_values(out_root, **over):
    """The reduced evaluation's values (W 128 project; small but every
    quantity defined). Independent of the smoke test's own file."""
    d = {"out_root": out_root, "seed": 777, "n_post": 40, "batch_rows": 16,
         "oracle_tol": 2e-3, "ppc_rows": 3, "ppc_draws": 4,
         "cal_glob": os.path.join(out_root, "calbank", "shard_*.npz"), "cal_n": 20,
         "cal_shards": 2, "cal_seed": 91, "lc2st_null": 5, "lc2st_eval_draws": 10,
         "bootstrap_n": 200, "level": 0.95, "dpi": 40, "formats": ["png"], "ncpus": 1,
         "jobs": {"queue": "test", "calbank": {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00"},
                  "sample": {"ncpus": 1, "mem": "2gb", "walltime": "00:20:00"},
                  "report": {"ncpus": 1, "mem": "2gb", "walltime": "00:30:00"},
                  "data": {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00"}},
         "data_view": {"sim_shards": 2, "examples_per_condition": 1, "sim_examples": 3,
                       "zoom_s": 4.0, "bins": 12}}
    d.update(over)
    return d


def write_eval_config(path, out_root, **over):
    with open(path, "w") as fh:
        json.dump(eval_values(out_root, **over), fh, indent=1, sort_keys=True)
    return path


def build_calbank(cfg_path, ecfg_path):
    """The calibration bank by hand, from ev_plan.calbank_build's argv (the
    launcher's own job does the same on davinci)."""
    import ev_config
    import ev_plan
    cfg = ev_plan.load_project(cfg_path)
    ecfg = ev_config.load(ecfg_path)
    build = ev_plan.calbank_build(cfg, ecfg)
    os.makedirs(build["out_dir"], exist_ok=True)
    builder = os.path.join(HPC, "joint", "stage1", "build_latent_bank.py")
    for k, argv in enumerate(build["argv"]):
        rc, out = run_py([builder, "--out-dir", build["out_dir"]] + list(argv), timeout=900)
        assert rc == 0, out[-3000:]
    return build


@pytest.fixture(scope="session")
def evaluation(project):
    """The evaluation run through its real entry points (the CLIs) on the
    reduced project: every ensemble sampled, the metrics, the figures. Returns
    dict(cfg, ecfg, out, logs)."""
    if "evaluation" in _STATE:
        return _STATE["evaluation"]
    out = os.path.join(project["root"], "spec_eval_t1c")
    done = os.path.join(project["root"], "spec_eval_t1c.done.json")
    if os.environ.get("SPEC_REUSE_EVAL", "") == "1" and os.path.isfile(done):
        _STATE["evaluation"] = load_json(done)      # a session's run kept for re-runs of the tests
        return _STATE["evaluation"]
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    ecfg = write_eval_config(os.path.join(project["root"], "spec_eval_t1c.json"), out)
    build_calbank(project["cfg"], ecfg)
    import ev_plan
    names = [e.name for e in ev_plan.ensembles(ev_plan.load_project(project["cfg"]))]
    logs = {}
    for n in names:
        rc, txt = run_py([os.path.join(EVAL, "ev_sample.py"), "--config", project["cfg"],
                          "--eval-config", ecfg, "--ensemble", n, "--ncpus", "1"])
        logs["sample:" + n] = (rc, txt)
    rc, txt = run_py([os.path.join(EVAL, "ev_metrics.py"), "--config", project["cfg"],
                      "--eval-config", ecfg, "--ncpus", "1"])
    logs["metrics"] = (rc, txt)
    rc, txt = run_py([os.path.join(EVAL, "ev_figures.py"), "--config", project["cfg"],
                      "--eval-config", ecfg])
    logs["figures"] = (rc, txt)
    _STATE["evaluation"] = {"cfg": project["cfg"], "ecfg": ecfg, "out": out, "logs": logs,
                            "names": names, "root": project["root"]}
    with open(done, "w") as fh:
        json.dump(_STATE["evaluation"], fh)
    return _STATE["evaluation"]


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def glob1(pattern):
    found = sorted(glob.glob(pattern))
    assert len(found) == 1, (pattern, found)
    return found[0]
