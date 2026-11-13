"""Silent-failure hunt, block 11 (and every bench bank): the box prior's clip
to [eps, 1 - eps], eps the float64 machine epsilon (block 11, "as
sample_prior"), and the bank's theta stored as float32 (sec. 2 notation:
theta (p,) float32). Every stored theta must lie in the open box (0, 1)^p
(sec. 2.4; the floor L_0 and the flow's density are defined there).

float32's largest value below 1 is 1 - 2^-24, so a draw u in (1 - 2^-25, 1)
is stored as exactly 1.0 -- the float64 clip at 1 - 2.2e-16 does not
survive the cast. Per coordinate the probability is 2^-25 (3.0e-8); over arm
S's 20,000 donors x 10 axes, 1 - (1 - 2^-25)^200000 = 0.6 % per bank.

The runner is then given a bank with one such row, in the report split and
in the training split, to see what the pipeline does with it.
"""
import json
import os
import subprocess

import numpy as np
import pytest

import t1b_helpers as H

H.joint_paths()
import latent_sbi_simulator as LS      # noqa: E402  (code under test)


def test_clip_does_not_survive_float32():
    eps = np.finfo(np.float64).eps
    assert np.float32(1.0 - eps) == 1.0
    assert np.nextafter(np.float32(1.0), np.float32(0.0)) == np.float32(1.0 - 2.0 ** -24)

    class NearOne(object):
        def uniform(self, lo, hi, size=None):
            return np.full(size, 1.0 - 1e-9)                # a legal uniform draw in (1 - 2^-25, 1)

    spec = LS.LatentSBISpec(n_latent=10, label_idx=tuple(range(7)), n_windows_per_trace=1,
                            T_win=12.8, fs=10.0)
    c, phi = LS.sample_box_prior(spec, 3, NearOne())
    assert np.all(phi < 1.0)                                # inside in float64, as the spec clips
    stored = phi.astype(np.float32)                         # build_latent_bank.py:370 stores float32
    assert np.all((stored > 0.0) & (stored < 1.0)), \
        "a box draw of 1 - 1e-9 is stored as theta = %r: outside the open box" % float(stored[0, 0])


def test_probability_per_arm_S_bank():
    p = 2.0 ** -25
    assert abs((1 - (1 - p) ** 200000) - 0.0059) < 5e-4


@pytest.fixture(scope="module")
def edge_bank():
    pytest.importorskip("torch")
    root = H.tmpdir("t1b_edge_")
    sim = os.path.join(root, "S")
    rc, out = H.run_builder(sim, ["--arm", "S", "--prior", "box", "--provider", "bench", "--fs", "10.0",
                                  "--T-win", "12.8", "--n-traces", "80", "--wells-per-donor", "2",
                                  "--n-windows", "2", "--seed", "41"])
    assert rc == 0, out
    yield root, os.path.join(sim, "shard_0000.npz")
    H.rmtree(root)


def _with_theta_one(root, src, part, tag):
    """A copy of the bank whose first donor of split `part` (0 train, 2
    report) has theta_0 = 1.0, as the builder stores a draw in (1 - 2^-25, 1)."""
    H.joint_paths()
    import run_joint_arms as RJA
    with np.load(src) as z:
        arrays = {k: z[k] for k in z.files}
    split, _h = RJA.grouped_split(arrays["donor"], fracs=(0.85, 0.10, 0.05), seed=0)
    d = arrays["donor"][np.flatnonzero(split == part)[0]]
    th = arrays["theta"].copy()
    th[arrays["donor"] == d, 0] = np.float32(1.0)
    arrays["theta"] = th
    out = os.path.join(root, tag)
    os.makedirs(out)
    np.savez_compressed(os.path.join(out, "shard_0000.npz"), **arrays)
    side = json.load(open(os.path.splitext(src)[0] + ".json"))
    json.dump(side, open(os.path.join(out, "shard_0000.json"), "w"))
    return os.path.join(out, "shard_*.npz")


def _run_a1(sim_glob, out_dir):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [H.PYTHON, H.RUNNER, "--arm", "A1", "--seed", "0", "--sim-shards", sim_glob, "--out-dir",
           out_dir, "--threads", "1", "--epochs", "2", "--steps-per-epoch", "3", "--b-sim", "24",
           "--n-posterior-draws", "16", "--embedding-size", "8", "--hidden-features", "24",
           "--num-transforms", "3", "--depth-exponent", "3", "--width-multiplier", "1.5",
           "--split-fracs", "0.85", "0.10", "0.05"]
    r = subprocess.run(cmd, cwd=H.STAGE3, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=1800)
    return r.returncode, r.stdout


@pytest.mark.parametrize("part,tag", [(2, "report"), (0, "train")])
def test_runner_on_a_stored_theta_of_one(edge_bank, part, tag):
    """A bank row with theta = 1.0 (the float32 edge): the runner should
    finish with finite numbers. Observed outcome reported by the assert."""
    root, src = edge_bank
    glob_ = _with_theta_one(root, src, part, tag)
    out_dir = os.path.join(root, "runs_" + tag)
    rc, txt = _run_a1(glob_, out_dir)
    tail = "\n".join(txt.strip().splitlines()[-4:])
    assert rc == 0, "the %s row with theta = 1.0: runner exit %d: %s" % (tag, rc, tail)
    rec = json.load(open(os.path.join(out_dir, "A1_seed0.json")))
    assert np.isfinite(rec["L"]) and np.isfinite(rec["L0"]), (rec["L"], rec["L0"])
