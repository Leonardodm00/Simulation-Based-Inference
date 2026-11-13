"""Spec-derived tests of block 17 (the synthetic project; SPEC.md secs. 2.4,
2.5, blocks 17, 25, 26) and of the directives the blocks' code is held to
(sec. 4: pure ASCII, LF, Python 3.6 syntax in the sequencer's modules; D-091:
no project value in the code). Oracles O17.1, O17.2 and sec. 2.4's theta
ceiling of the t1-B report.

The committed values are read with json (documentation keys dropped by the
spec's rule) and compared with the numbers SPEC.md states.
"""
import importlib.util
import itertools
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
from scipy import stats

import t1b_helpers as H


def _lbg():
    H.add_paths(H.DSN)
    spec = importlib.util.spec_from_file_location("t1b_lbg17",
                                                  os.path.join(H.DSN, "latent_burst_generator.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_o17_1_committed_values():
    d = H.committed_synth()
    assert d["kind"] == "synthetic" and d["dataset"] == "synth" and d["config_version"] == 10
    b = d["synthetic"]["builder"]
    assert (b["provider"], b["fs"], b["T_win"], b["n_neurons"]) == ("bench", 100.0, 180.0, 100)
    assert (b["wells_per_donor"], b["donors_per_batch"], b["n_classes"]) == (2, 8, 3)
    assert b["tau_ov"] == 0.10 and b["class_centres"] == "regular"
    assert b["label_axes"] == [0, 1, 2, 3, 4, 5, 6, 7] and b["centre_scale"] == 0.562743   # block 25 (v4)
    s = d["synthetic"]["sim"]
    assert (s["n_theta"], s["n_windows"], s["seed"], s["traces_per_shard"]) == (20000, 1, 1000, 2000)
    assert s["n_theta"] * b["wells_per_donor"] // s["traces_per_shard"] == 20            # K
    r0, r3 = d["synthetic"]["real"]
    modes = ["range_shift", "drift", "contamination"]
    assert (r0["name"], r0["seed"], r0["donors_per_class"], r0["n_windows"], r0["pi"]) == \
        ("R0", 2000, 9, 6, 0.0)
    assert (r3["name"], r3["seed"], r3["donors_per_class"], r3["n_windows"], r3["pi"]) == \
        ("R3", 3000, 9, 6, 0.3)
    assert sorted(r0["gap_modes"]) == sorted(modes) and sorted(r3["gap_modes"]) == sorted(modes)
    got = [(q["name"], q["campaign"], q["real"], q["alt_best_fit"]) for q in d["synthetic"]["searches"]]
    assert got == [("S-A1", "S-A1", "R0", False), ("S-A2-R0", "S-A2-only", "R0", False),
                   ("S-A5-R0", "S-A5-only", "R0", False), ("S-A2-R3", "S-A2-only", "R3", False),
                   ("S-A5-R3", "S-A5-only", "R3", False)]                                # block 26
    q = d["search"]
    assert (q["n_initial_points"], q["gp_rounds"], q["trials_per_round"]) == (20, 4, 10)  # N = 60
    t = q["trial"]
    assert (t["ncpus"], t["mem"], t["walltime"], t["passes"], t["epochs"]) == (8, "96gb", "24:00:00", 6.0, 20)
    assert q["space_overrides"] == {"depth_exponent": [3, 5], "embedding_size": [16, 64],
                                    "head_pool_ops": [0, 1]}
    assert q["shape_anchors"] == {"p": 10, "d_theta": 10, "embedding_dim": 64}
    f = d["best_fit"]
    assert f["member_seeds"] == [1, 2, 3, 4, 5] and f["split_seed"] == 1                 # sec. 2.5
    assert (f["ncpus"], f["mem"], f["walltime"], f["passes"], f["epochs"], f["patience_epochs"]) == \
        (8, "96gb", "120:00:00", 40.0, 40, 10)
    assert d["loader"]["buffer_shards"] == 20                                             # the whole arm S
    j = d["synthetic"]["bank_job"]
    assert (j["ncpus"], j["mem"], j["walltime"]) == (1, "8gb", "02:00:00")
    assert d["driver"]["ncpus"] == 4
    assert d["resources"]["max_cores"] == 120
    assert d["split"]["fractions"] == [0.85, 0.10, 0.05]                                  # D-088


def test_o17_2_centre_distances():
    """block 25: the three centres 0.441934 apart on 8 axes at 0.562743 (to
    1e-6, equal), the 7-axis distance at 0.6 the same."""
    lbg = _lbg()
    for L, s in ((8, 0.562743), (7, 0.6)):
        m = np.asarray(lbg._class_center_vectors(3, L, "simplex"), dtype=np.float64)
        mu = 0.5 + s * (m - 0.5)
        dist = [np.linalg.norm(mu[a] - mu[b]) for a, b in itertools.combinations(range(3), 2)]
        np.testing.assert_allclose(dist, 0.441934, rtol=0, atol=1e-6)


def _theta_ceiling(L, s, tau, n, seed):
    """sec. 2.4: A_theta = E[max_c P(c | theta)], equal condition
    probabilities, theta_k | c ~ TN(mu_{c,k}, tau^2; (0, 1)) on the L label
    axes (the free axes carry no information). Monte Carlo with scipy."""
    lbg = _lbg()
    m = np.asarray(lbg._class_center_vectors(3, L, "simplex"), dtype=np.float64)
    mu = 0.5 + s * (m - 0.5)
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 3, n)
    th = np.empty((n, L))
    for k in range(L):
        for cl in range(3):
            idx = c == cl
            law = stats.truncnorm((0 - mu[cl, k]) / tau, (1 - mu[cl, k]) / tau, loc=mu[cl, k], scale=tau)
            th[idx, k] = law.rvs(size=int(idx.sum()), random_state=rng)
    logp = np.zeros((n, 3))
    for cl in range(3):
        for k in range(L):
            law = stats.truncnorm((0 - mu[cl, k]) / tau, (1 - mu[cl, k]) / tau, loc=mu[cl, k], scale=tau)
            logp[:, cl] += law.logpdf(th[:, k])
    post = np.exp(logp - logp.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    mx = post.max(axis=1)
    return float(mx.mean()), float(mx.std() / np.sqrt(n))


def test_sec24_theta_ceiling():
    """sec. 2.4: at s = 0.6, tau = 0.10 on the 7 label axes, A_theta =
    0.973-0.976 [SANDBOX]. The population value by Monte Carlo (400,000
    draws, SE about 2.5e-4) lies within that range widened by 4 SE."""
    a, se = _theta_ceiling(7, 0.6, 0.10, 400000, 24)
    print("A_theta (7 axes, s 0.6): %.5f +- %.5f" % (a, se))
    assert 0.973 - 4 * se <= a <= 0.976 + 4 * se, (a, se)
    a8, se8 = _theta_ceiling(8, 0.562743, 0.10, 400000, 25)
    print("A_theta (8 axes, s 0.562743, synth.json v4): %.5f +- %.5f" % (a8, se8))


def test_o17_1_readme_run_sheets():
    text = open(os.path.join(H.PIPE, "README.md")).read()
    for head in ("synth: the first submission", "synth: the other arm's best fits",
                 "synth: the relaunch"):
        assert re.search(r"^#{2,4} " + re.escape(head), text, re.M), head


# ----------------------------------------------------------- directives ---

SCOPE_FILES = [
    "hpc/joint/stage1/build_latent_bank.py", "hpc/joint/stage1/latent_sbi_simulator.py",
    "hpc/joint/stage1/bench_burst_provider.py", "hpc/joint/stage2/joint_batches.py",
    "hpc/joint/stage3/run_joint_arms.py", "hpc/joint/stage3/smoke_test_blocks_11_14.py",
    "hpc/joint/stage3/smoke_test_joint_arms.py", "hpc/pipeline/config.py",
    "hpc/pipeline/stages/synthetic.py", "hpc/pipeline/stages/__init__.py",
    "hpc/pipeline/stages/common.py", "hpc/pipeline/stages/search.py",
    "hpc/pipeline/stages/best_fit.py", "hpc/pipeline/pipeline.py",
    "hpc/pipeline/tools/synth_bank_record.py", "hpc/pipeline/projects/synth/synth.json",
    "hpc/pipeline/tests/test_smoke_synthetic.py", "hpc/pipeline/README.md",
]
SEQUENCER_FILES = ["hpc/pipeline/config.py", "hpc/pipeline/stages/synthetic.py",
                   "hpc/pipeline/stages/__init__.py", "hpc/pipeline/stages/common.py",
                   "hpc/pipeline/stages/search.py", "hpc/pipeline/stages/best_fit.py",
                   "hpc/pipeline/pipeline.py"]


@pytest.mark.parametrize("rel", SCOPE_FILES)
def test_sec4_ascii_lf(rel):
    blob = open(os.path.join(H.REPO, rel), "rb").read()
    assert all(b < 128 for b in blob), rel
    assert b"\r" not in blob, rel


def test_sec4_sequencer_python36_and_stdlib():
    """sec. 4: the sequencer's modules are stdlib only and Python 3.6 syntax
    (vermin's minimum version)."""
    vermin = shutil.which("vermin")
    if not vermin:
        pytest.skip("vermin is not installed")
    files = [os.path.join(H.REPO, f) for f in SEQUENCER_FILES]
    r = subprocess.run([vermin, "--no-tips", "--violations", "-t=3.6-"] + files,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert r.returncode == 0, r.stdout[-3000:]
    import sys as _sys
    std = set(getattr(_sys, "stdlib_module_names", ()))
    if not std:
        pytest.skip("no stdlib list in this Python")
    import ast
    own = ("config", "pbs", "state", "envs", "stages", "pipeline")
    for f in files:
        tree = ast.parse(open(f).read(), filename=f)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module.split(".")[0]]
            else:
                continue
            for n in names:
                assert n in std or n in own, "%s:%d imports %s" % (f, node.lineno, n)


def test_d091_no_project_value_in_the_code():
    pat = re.compile(r'"synth"|\'synth\'|synth_(sim|real)|davinci-1|ldellamea|"R0"|"R3"|S-A2-R0|synth\.json')
    files = [os.path.join(H.PIPE, f) for f in os.listdir(H.PIPE) if f.endswith(".py")]
    files += [os.path.join(H.PIPE, "stages", f) for f in os.listdir(os.path.join(H.PIPE, "stages"))
              if f.endswith(".py")]
    files += [os.path.join(H.PIPE, "run_pipeline.sh"), H.RECORD_TOOL]
    for f in files:
        for i, line in enumerate(open(f), 1):
            assert not pat.search(line), "%s:%d: %s" % (f, i, line.strip())
