#!/usr/bin/env python3
"""Smoke test for blocks 4, 5 and 6 of specs/SPEC.md (the pipeline build):
the campaign prior, the streaming loader and the runner's flags, checkpoint
and resume.

Checks (the SPEC's oracles; B4 / B5 / B6 name the block)
    test_b5_index          ShardIndex against concat_shards; a foreign contract refused
    test_b5_equivalence    the stream at K >= n_shards draws the in-memory batches
    test_b5_uniformity     K = 2 on 8 unequal shards: coverage, mean, CV, per-shard
                           counts, steps per pass, at most K resident
    test_b5_state          a stream saved and reloaded continues identically
    test_b5_evaluate       evaluate_npe_shards == evaluate_npe (1e-6 nats)
    test_b4_runner         a training element on a campaign bank: L0 = 0,
                           delta_hat = -L, the replicate block with real shards
    test_b5_runner         --buffer-shards 99 == --buffer-shards 0 bit for bit;
                           --buffer-shards 1 runs; L(K=1) vs L(K=99) reported
    test_b6_flags          --passes, --split-fracs, --split-seed, --head-pool-ops,
                           --flow-hidden-layers, --threads, --patience
    test_b6_resume         stop after epoch 1 + --resume == the uninterrupted run
    test_b6c_damaged_checkpoint  [D-111] --resume on an empty, a truncated, a garbage
                         checkpoint: moved aside, the run from epoch 0 == no checkpoint
    test_convergence       skipped (not a discretisation)

Run
    cd hpc/joint/stage3 && python3 smoke_test_blocks_4_6.py
        About 3 minutes at one thread. Needs torch, sbi, zuko (sbi_env) and
        the in-repo DSN tree (hpc/dsn). Builds its own campaign-style banks
        (8 simulated shards of unequal sizes, W 128, p 23; a 12-row real
        bank) in a temporary folder; nothing is read from davinci.

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

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "stage1"), os.path.join(_HERE, "..", "stage2"),
           os.path.join(_HERE, "..")):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import latent_bank as LB            # noqa: E402

RUNNER = os.path.join(_HERE, "run_joint_arms.py")
PYTHON = sys.executable
P, W, FS = 23, 128, 100.0
SHARD_SIZES = (31, 17, 40, 9, 26, 22, 35, 14)     # unequal on purpose (SPEC block 5)
ROWS_PER_DONOR = 5


# --------------------------------------------------------------- helpers ---

def _campaign_sidecar(arm, kind, theta_withheld):
    names = ["p%02d" % i for i in range(P)]
    bounds = [[0.0, 1.0]] * P
    latent = {"kind": kind, "bank": "smoke", "param_names": names, "coord_physical": "unit",
              "units": ["-"] * P, "theta_map": "identity", "prior": "uniform on the unit box",
              "on_axis": names[0], "on_declared_box": [0.0, 1.0], "topology_axes": [names[-1]],
              "n_windows_per_trace": 1, "T_win": W / FS, "fs": FS, "n_e": 1,
              "floor_hz": 0.1, "floor_rtol": 1e-6, "label_freeze_sha256": "0" * 64,
              "cohort_manifest_sha256": "0" * 64}
    if kind == "campaign_real":
        latent = {"kind": "campaign_real", "classes": ["a", "b"]}
    return LB.build_sidecar(param_names=names, bounds_theta=bounds, coord="unit_box",
                            fs=FS, w_size=1.0 / FS, T_win=W / FS, W=W,
                            scale_convention="per_electrode_mean", latent_spec=latent,
                            nuisance_spec={"kind": "none", "components": ["a", "b", "c"]},
                            realisation_spec={"kind": "simulator_seed_run"},
                            gap_spec={"kind": "none"}, generator_sha256="1" * 64,
                            theta_withheld=theta_withheld, arm=arm)


def _rows(rng, n, donor0, cls_values):
    theta = rng.uniform(0.05, 0.95, size=(n, P)).astype(np.float32)
    # the window carries a signal about theta so the NLL can move
    t = np.linspace(0, 1, W, dtype=np.float32)
    x = (theta[:, :1] * 3.0 + theta[:, 1:2] * np.sin(2 * np.pi * (2 + 6 * theta[:, 2:3]) * t)
         + 0.1 * rng.standard_normal((n, W))).astype(np.float32)
    donor = donor0 + np.arange(n) // ROWS_PER_DONOR
    return {"x": x, "theta": theta,
            "cls": np.asarray(cls_values, dtype=np.int64),
            "nu": np.zeros((n, 3), dtype=np.float32),
            "realisation_id": np.arange(n, dtype=np.uint64),
            "donor": donor.astype(np.int64), "well": donor.astype(np.int64),
            "batch": np.zeros(n, dtype=np.int64), "subregion": np.zeros(n, dtype=np.int64),
            "window_idx": np.arange(n, dtype=np.int64),
            "contaminated": np.zeros(n, dtype=bool)}


def build_banks(root, seed=7):
    """8 simulated shards of unequal sizes and a 12-row real bank."""
    rng = np.random.default_rng(seed)
    sim = os.path.join(root, "S")
    os.makedirs(sim)
    side = _campaign_sidecar("S", "campaign", False)
    donor0 = 0
    for k, n in enumerate(SHARD_SIZES):
        arrays = _rows(rng, n, donor0, [-1] * n)
        donor0 += int(np.ceil(n / ROWS_PER_DONOR)) + 1
        LB.write_shard(os.path.join(sim, "shard_%04d.npz" % k), arrays, side)
    real = os.path.join(root, "R")
    os.makedirs(real)
    side_r = _campaign_sidecar("R", "campaign_real", False)
    arrays = _rows(rng, 12, 1000, [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    arrays["donor"] = np.repeat(np.arange(4), 3).astype(np.int64) + 1000
    arrays["well"] = arrays["donor"].copy()
    LB.write_shard(os.path.join(real, "shard_0000.npz"), arrays, side_r)
    return os.path.join(sim, "*.npz"), os.path.join(real, "*.npz")


_SHARED = {}


def shared():
    if "root" not in _SHARED:
        root = tempfile.mkdtemp(prefix="blocks46_")
        _SHARED["root"] = root
        _SHARED["sim"], _SHARED["real"] = build_banks(root)
    return _SHARED


def run_runner(out_dir, extra, env_extra=None, timeout=900):
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.update(env_extra or {})
    sh = shared()
    cmd = [PYTHON, RUNNER, "--arm", "A1", "--sim-shards", sh["sim"], "--out-dir", out_dir,
           "--threads", "1", "--n-posterior-draws", "16", "--embedding-size", "8",
           "--hidden-features", "24", "--num-transforms", "3", "--depth-exponent", "3",
           "--width-multiplier", "1.5"] + list(extra)
    r = subprocess.run(cmd, cwd=_HERE, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return r.returncode, r.stdout


def record(out_dir, stem="A1_seed0"):
    with open(os.path.join(out_dir, stem + ".json")) as fh:
        return json.load(fh)


def ckpt_sha(out_dir, stem="A1_seed0"):
    import torch
    ck = torch.load(os.path.join(out_dir, stem + "_ckpt.pt"), map_location="cpu",
                    weights_only=False)
    return ck["model_sha256"], ck


def _tiny_model(seed=0):
    """A small joint model on a fallback-style backbone, for the evaluators."""
    import torch
    import torch.nn as nn
    from sbi.utils import BoxUniform
    from joint_model import build_joint_model
    torch.manual_seed(seed)

    class Small(nn.Module):
        def __init__(self, E=6):
            super().__init__()
            self.c1 = nn.Conv1d(1, 8, 5, stride=4, padding=2)
            self.n1 = nn.GroupNorm(2, 8)
            self.fc = nn.Linear(8, E)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            h = torch.relu(self.n1(self.c1(x))).mean(-1)
            return torch.nn.functional.normalize(self.fc(h), dim=-1)

    prior = BoxUniform(torch.zeros(P), torch.ones(P))
    return build_joint_model(Small(), prior, torch.rand(8, P) * 0.9 + 0.05,
                             torch.rand(8, W), hidden_features=16, num_transforms=2,
                             num_bins=6)


# ---------------------------------------------------------------- checks ---

def test_b5_index():
    import torch
    sh = shared()
    paths = sorted(glob.glob(sh["sim"]))
    arrays, side = LB.concat_shards(paths)
    idx = LB.ShardIndex(paths)
    assert idx.n_rows == arrays["x"].shape[0] == sum(SHARD_SIZES)
    assert tuple(idx.n_rows_per_shard) == SHARD_SIZES
    assert idx.offsets.tolist() == [0] + list(np.cumsum(SHARD_SIZES))
    for k in LB.SMALL_ARRAYS:
        np.testing.assert_array_equal(getattr(idx, k), arrays[k], err_msg=k)
    for k in range(idx.n_shards):
        np.testing.assert_array_equal(idx.load_x(k), arrays["x"][idx.offsets[k]:idx.offsets[k + 1]])
    assert idx.shard_of_row(0) == 0 and idx.shard_of_row(idx.n_rows - 1) == idx.n_shards - 1
    assert idx.sidecar["contract_digest"] == side["contract_digest"]
    # a shard of another contract is refused
    tmp = tempfile.mkdtemp()
    try:
        foreign = _campaign_sidecar("S", "campaign", True)       # theta_withheld differs
        rng = np.random.default_rng(1)
        LB.write_shard(os.path.join(tmp, "shard_0000.npz"), _rows(rng, 5, 0, [-1] * 5), foreign)
        try:
            LB.ShardIndex([paths[0], os.path.join(tmp, "shard_0000.npz")])
            raise AssertionError("a foreign contract was accepted")
        except LB.ContractError as exc:
            assert "different contract" in str(exc), str(exc)
    finally:
        shutil.rmtree(tmp)
    assert torch  # imported for the side effect of the environment check


def test_b5_equivalence():
    import torch
    from joint_batches import BatchSpec, ShardStream, ThreeStreamBatcher
    sh = shared()
    paths = sorted(glob.glob(sh["sim"]))
    arrays, _ = LB.concat_shards(paths)
    idx = LB.ShardIndex(paths)
    n = idx.n_rows
    rng = np.random.default_rng(3)
    mask = rng.uniform(size=n) < 0.7
    theta = torch.as_tensor(arrays["theta"])
    x = torch.as_tensor(arrays["x"])
    mem = ThreeStreamBatcher(theta[mask], x[mask], spec=BatchSpec(13, 0, 0), seed=5)
    stream = ShardStream(idx, mask, 13, 99, seed=5)
    assert stream.whole
    st = ThreeStreamBatcher(None, None, spec=BatchSpec(13, 0, 0), seed=5, sim_stream=stream)
    masked_rows = np.flatnonzero(mask)
    for i in range(50):
        t1, x1, i1 = mem.sim_batch()
        t2, x2, r2 = st.sim_batch()
        np.testing.assert_array_equal(masked_rows[i1.numpy()], r2.numpy(), err_msg="batch %d rows" % i)
        assert torch.equal(t1, t2) and torch.equal(x1, x2), "batch %d tensors" % i


def test_b5_uniformity():
    import torch
    from joint_batches import ShardStream
    sh = shared()
    paths = sorted(glob.glob(sh["sim"]))
    idx = LB.ShardIndex(paths)
    n = idx.n_rows
    rng = np.random.default_rng(11)
    mask = rng.uniform(size=n) < 0.8
    mask[idx.offsets[3]:idx.offsets[4]] = False      # one shard entirely unmasked: skipped
    B, K, passes = 10, 2, 400
    stream = ShardStream(idx, mask, B, K, seed=2)
    assert not stream.whole
    n_masked = int(mask.sum())
    steps = int(round(passes * n_masked / B))
    counts = np.zeros(n, dtype=np.int64)
    max_resident = 0
    for _ in range(steps):
        _, _, rows = stream.next()
        np.add.at(counts, rows.numpy(), 1)
        max_resident = max(max_resident, len(stream.resident_shards))
    assert max_resident <= K, max_resident
    assert counts[~mask].sum() == 0, "an unmasked row was drawn"
    drawn = counts[mask]
    assert drawn.min() > 0, "a masked row was never drawn"
    mean, cv = drawn.mean(), drawn.std() / drawn.mean()
    assert abs(mean - passes) / passes < 0.02, "mean draws per row %.2f, expected %d" % (mean, passes)
    assert cv < 0.10, "CV of draws per row %.3f" % cv
    per_shard = []
    for k in range(idx.n_shards):
        lo, hi = idx.offsets[k], idx.offsets[k + 1]
        nk = int(mask[lo:hi].sum())
        if nk:
            got = counts[lo:hi].sum()
            per_shard.append((k, nk, got))
            assert abs(got - passes * nk) / (passes * nk) < 0.02, (k, nk, got)
    assert stream.steps == steps
    assert torch


def test_b5_state():
    import torch
    from joint_batches import ShardStream
    sh = shared()
    paths = sorted(glob.glob(sh["sim"]))
    idx = LB.ShardIndex(paths)
    mask = np.ones(idx.n_rows, dtype=bool)
    a = ShardStream(idx, mask, 7, 2, seed=9)
    for _ in range(37):
        a.next()
    st = a.state_dict()
    b = ShardStream(idx, mask, 7, 2, seed=9).load_state_dict(st)
    for i in range(20):
        ra = a.next()
        rb = b.next()
        assert torch.equal(ra[2], rb[2]) and torch.equal(ra[1], rb[1]), "batch %d after reload" % i
    # a state for another (seed, batch, K) is refused
    try:
        ShardStream(idx, mask, 8, 2, seed=9).load_state_dict(st)
        raise AssertionError("a state for another batch size was accepted")
    except ValueError:
        pass


def test_b5_evaluate():
    import torch
    from joint_train import evaluate_npe, evaluate_npe_shards
    sh = shared()
    paths = sorted(glob.glob(sh["sim"]))
    arrays, _ = LB.concat_shards(paths)
    idx = LB.ShardIndex(paths)
    model = _tiny_model()
    rng = np.random.default_rng(5)
    mask = rng.uniform(size=idx.n_rows) < 0.6
    a = evaluate_npe(model, torch.as_tensor(arrays["theta"][mask]), torch.as_tensor(arrays["x"][mask]))
    b = evaluate_npe_shards(model, idx, mask)
    assert abs(a - b) < 1e-6, (a, b)
    try:
        evaluate_npe_shards(model, idx, np.zeros(idx.n_rows, dtype=bool))
        raise AssertionError("an empty split was scored")
    except ValueError:
        pass


def test_b4_runner():
    sh = shared()
    out = os.path.join(sh["root"], "b4")
    rc, log = run_runner(out, ["--epochs", "2", "--steps-per-epoch", "3", "--b-sim", "16",
                               "--seed", "0"])
    assert rc == 0, log[-3000:]
    assert "prior            : campaign bank -- uniform on the unit box (D-078)" in log, log[-2000:]
    rec = record(out)
    assert rec["L0"] == 0.0, rec["L0"]
    assert abs(rec["delta_hat"] + rec["L"]) < 1e-12, (rec["delta_hat"], rec["L"])
    assert rec["prior"] == "campaign_unit_box"
    assert rec["endpoint_primary"] == "simulated (arm R absent)"
    assert rec["loader"] == {"buffer_shards": 0, "n_shards": 8, "streamed": False}
    assert rec["threads"] == 1
    assert rec["cluster_note"] == "too few rows", rec.get("cluster_note")
    assert rec["spectrum_source"].startswith("repo:"), rec["spectrum_source"]
    assert not os.path.exists(os.path.join(out, "A1_seed0_train_ckpt.pt")), "the training checkpoint must be removed"
    # with the real bank: the replicate block, no pseudo-real endpoint
    out2 = os.path.join(sh["root"], "b4r")
    rc, log = run_runner(out2, ["--epochs", "2", "--steps-per-epoch", "3", "--b-sim", "16",
                                "--seed", "0", "--real-shards", sh["real"]])
    assert rc == 0, log[-3000:]
    rec2 = record(out2)
    assert rec2["endpoint_primary"] == "simulated (arm R absent)", rec2["endpoint_primary"]
    assert rec2["L_pseudo_real"] is None
    assert "replicate" in rec2 and rec2["replicate"]["n_pairs"] == 12, rec2.get("replicate")
    assert rec2["L"] == rec["L"], "the real shards must not change the simulated training (%r vs %r)" % (rec2["L"], rec["L"])


def test_b5_runner():
    sh = shared()
    base = os.path.join(sh["root"], "b5_mem")
    rc, log = run_runner(base, ["--epochs", "3", "--steps-per-epoch", "4", "--b-sim", "16",
                                "--seed", "1", "--buffer-shards", "0"])
    assert rc == 0, log[-3000:]
    whole = os.path.join(sh["root"], "b5_whole")
    rc, log = run_runner(whole, ["--epochs", "3", "--steps-per-epoch", "4", "--b-sim", "16",
                                 "--seed", "1", "--buffer-shards", "99"])
    assert rc == 0, log[-3000:]
    assert "streamed: buffer 99 shard(s) of 8" in log, log[-2000:]
    a, b = record(base, "A1_seed1"), record(whole, "A1_seed1")
    assert a["split_hash"] == b["split_hash"]
    assert a["history"] == b["history"], "histories differ:\n%r\n%r" % (a["history"], b["history"])
    assert a["L"] == b["L"], (a["L"], b["L"])
    np.testing.assert_array_equal(np.load(os.path.join(base, "A1_seed1_perrow.npz"))["nll"],
                                  np.load(os.path.join(whole, "A1_seed1_perrow.npz"))["nll"])
    assert ckpt_sha(base, "A1_seed1")[0] == ckpt_sha(whole, "A1_seed1")[0]
    assert b["loader"] == {"buffer_shards": 99, "n_shards": 8, "streamed": True}
    # a buffer smaller than the bank runs; the measurement D-084 asks for
    small = os.path.join(sh["root"], "b5_small")
    rc, log = run_runner(small, ["--epochs", "3", "--steps-per-epoch", "4", "--b-sim", "16",
                                 "--seed", "1", "--buffer-shards", "1"])
    assert rc == 0, log[-3000:]
    c = record(small, "A1_seed1")
    assert c["split_hash"] == a["split_hash"]
    assert np.isfinite(c["L"])
    _SHARED["measurement"] = "L after 3 epochs: K=99 %.4f, K=1 %.4f (the fixture is too small to judge; the bench banks decide)" % (b["L"], c["L"])


def test_b6_flags():
    import torch
    from run_joint_arms import grouped_split
    sh = shared()
    # --split-fracs on 2,612 synthetic donor ids at seed 0 (check_config_v2's number)
    donors = np.arange(2612)
    idx, _ = grouped_split(donors, fracs=(0.85, 0.10, 0.05), seed=0)
    counts = [int((idx == k).sum()) for k in (0, 1, 2)]
    assert counts == [2220, 261, 131], counts
    idx_old, h_old = grouped_split(donors, seed=0)
    assert [int((idx_old == k).sum()) for k in (0, 1, 2)] == [1828, 392, 392]
    # fractions not summing to 1 are refused by the runner
    out = os.path.join(sh["root"], "b6_fracs")
    rc, log = run_runner(out, ["--epochs", "1", "--steps-per-epoch", "1", "--b-sim", "8",
                               "--split-fracs", "0.8", "0.1", "0.05"])
    assert rc != 0 and "--split-fracs must be" in log, log[-500:]
    # --passes: S = ceil(P / N_ep * n_train / B)
    out = os.path.join(sh["root"], "b6_passes")
    rc, log = run_runner(out, ["--epochs", "4", "--passes", "2", "--b-sim", "16", "--seed", "0",
                               "--split-fracs", "0.85", "0.10", "0.05"])
    assert rc == 0, log[-3000:]
    rec = record(out)
    n_train = rec["n_train"]
    S = int(np.ceil(2.0 / 4 * n_train / 16.0))
    assert rec["steps_per_epoch_effective"] == S, (rec["steps_per_epoch_effective"], S)
    assert rec["passes"] == 2.0 and rec["split_fracs"] == [0.85, 0.10, 0.05]
    drawn = 4 * S * 16
    assert 2 * n_train <= drawn < 2 * n_train + 4 * 16, (drawn, n_train)
    # --split-seed: two members on one split share the group array, differ in weights
    outs = {}
    for seed in (1, 2):
        o = os.path.join(sh["root"], "b6_member%d" % seed)
        rc, log = run_runner(o, ["--epochs", "1", "--steps-per-epoch", "2", "--b-sim", "16",
                                 "--seed", str(seed), "--split-seed", "1"])
        assert rc == 0, log[-3000:]
        outs[seed] = o
    assert os.path.isfile(os.path.join(outs[1], "A1_seed1.json")), "split seed == seed keeps today's stem"
    assert os.path.isfile(os.path.join(outs[2], "A1_split1_seed2.json")), os.listdir(outs[2])
    g1 = np.load(os.path.join(outs[1], "A1_seed1_perrow.npz"))["group"]
    g2 = np.load(os.path.join(outs[2], "A1_split1_seed2_perrow.npz"))["group"]
    np.testing.assert_array_equal(g1, g2)
    assert record(outs[1], "A1_seed1")["split_hash"] == record(outs[2], "A1_split1_seed2")["split_hash"]
    assert ckpt_sha(outs[1], "A1_seed1")[0] != ckpt_sha(outs[2], "A1_split1_seed2")[0]
    # --head-pool-ops 1: the head's in-features triple; code 0 is today's model
    heads = {}
    for code in (0, 1):
        o = os.path.join(sh["root"], "b6_head%d" % code)
        rc, log = run_runner(o, ["--epochs", "1", "--steps-per-epoch", "1", "--b-sim", "8",
                                 "--head-pool-ops", str(code)])
        assert rc == 0, log[-3000:]
        _, ck = ckpt_sha(o)
        heads[code] = tuple(ck["model_state"]["estimator._embedding_net.head.proj.weight"].shape)
    assert heads[1][1] == 3 * heads[0][1], heads
    # --flow-hidden-layers 2 at 4 transforms: 2 hidden layers per conditioner, 4 without
    layers = {}
    for flag in (None, 2):
        o = os.path.join(sh["root"], "b6_flow%s" % flag)
        rc, log = run_runner(o, ["--epochs", "1", "--steps-per-epoch", "1", "--b-sim", "8",
                                 "--num-transforms", "4"]
                             + ([] if flag is None else ["--flow-hidden-layers", "2"]))
        assert rc == 0, log[-3000:]
        _, ck = ckpt_sha(o)
        keys = [k for k in ck["model_state"]
                if k.startswith("estimator.net.transform.transforms.1.hyper.") and k.endswith(".weight")]
        layers[flag] = len(keys) - 1          # hidden layers = Linear layers minus the output one
    assert layers == {None: 4, 2: 2}, layers
    # --threads from NCPUS
    o = os.path.join(sh["root"], "b6_threads")
    env = {"NCPUS": "2"}
    cmd = [PYTHON, RUNNER, "--arm", "A1", "--sim-shards", sh["sim"], "--out-dir", o,
           "--epochs", "1", "--steps-per-epoch", "1", "--b-sim", "8", "--n-posterior-draws", "16",
           "--embedding-size", "8", "--hidden-features", "24", "--num-transforms", "3"]
    e = dict(os.environ)
    e.update(env)
    r = subprocess.run(cmd, cwd=_HERE, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:]
    assert "torch threads    : 2" in r.stdout and record(o)["threads"] == 2
    # --patience 2 with lr 0: the validation never improves after epoch 0 -> stop after epoch 2
    o = os.path.join(sh["root"], "b6_patience")
    rc, log = run_runner(o, ["--epochs", "6", "--steps-per-epoch", "2", "--b-sim", "16",
                             "--lr", "0", "--patience", "2"])
    assert rc == 0, log[-3000:]
    rec = record(o)
    assert len(rec["history"]) == 3, [h["epoch"] for h in rec["history"]]
    assert "early stop: no NPE validation improvement for 2 epochs" in log
    vals = [h["val_npe"] for h in rec["history"]]
    assert rec["val_npe"] >= min(vals) - 1e-12
    assert torch


def test_b6_resume():
    sh = shared()
    common = ["--epochs", "5", "--steps-per-epoch", "3", "--b-sim", "16", "--seed", "3",
              "--buffer-shards", "2", "--dropout", "0.1"]
    full = os.path.join(sh["root"], "b6_full")
    rc, log = run_runner(full, common)
    assert rc == 0, log[-3000:]
    part = os.path.join(sh["root"], "b6_part")
    rc, log = run_runner(part, common + ["--stop-after-epoch", "1"])
    assert rc == 0, log[-3000:]
    assert "STOPPED after epoch 1" in log, log[-1500:]
    assert os.path.isfile(os.path.join(part, "A1_seed3_train_ckpt.pt"))
    assert not os.path.isfile(os.path.join(part, "A1_seed3.json")), "no record after a stop"
    rc, log = run_runner(part, common + ["--resume"])
    assert rc == 0, log[-3000:]
    assert "resumed from" in log and "2 epoch(s) done" in log, log[-2000:]
    a, b = record(full, "A1_seed3"), record(part, "A1_seed3")
    assert a["history"] == b["history"], "histories differ:\n%r\n%r" % (a["history"], b["history"])
    assert a["L"] == b["L"] and a["split_hash"] == b["split_hash"]
    assert ckpt_sha(full, "A1_seed3")[0] == ckpt_sha(part, "A1_seed3")[0], "weights differ after resume"
    assert not os.path.isfile(os.path.join(part, "A1_seed3_train_ckpt.pt")), "the training checkpoint must be removed"
    # --resume with no checkpoint starts fresh (a job may always pass it)
    fresh = os.path.join(sh["root"], "b6_fresh")
    rc, log = run_runner(fresh, common + ["--resume"])
    assert rc == 0 and "resumed from" not in log, log[-1500:]
    assert record(fresh, "A1_seed3")["L"] == a["L"]
    # the early stop's counter survives a resume: lr 0, patience 2, stop after epoch 1
    stop_full = os.path.join(sh["root"], "b6_stop_full")
    cfg = ["--epochs", "6", "--steps-per-epoch", "2", "--b-sim", "16", "--lr", "0",
           "--patience", "2", "--seed", "4"]
    rc, log = run_runner(stop_full, cfg)
    assert rc == 0, log[-3000:]
    stop_part = os.path.join(sh["root"], "b6_stop_part")
    rc, log = run_runner(stop_part, cfg + ["--stop-after-epoch", "1"])
    assert rc == 0, log[-3000:]
    rc, log = run_runner(stop_part, cfg + ["--resume"])
    assert rc == 0, log[-3000:]
    assert [h["epoch"] for h in record(stop_full, "A1_seed4")["history"]] == \
        [h["epoch"] for h in record(stop_part, "A1_seed4")["history"]] == [0, 1, 2]


def test_b6c_damaged_checkpoint():
    """[D-111] --resume on a damaged checkpoint (empty, truncated, garbage):
    moved aside as .corrupt1, and the run from epoch 0 equals a run without
    a checkpoint, bit for bit at one thread (the pipeline passes --resume to
    every trial and member, so a torn checkpoint must not fail the run)."""
    sh = shared()
    common = ["--epochs", "3", "--steps-per-epoch", "3", "--b-sim", "16", "--seed", "5",
              "--buffer-shards", "2"]
    ref = os.path.join(sh["root"], "b6c_ref")
    rc, log = run_runner(ref, common)
    assert rc == 0, log[-3000:]
    part = os.path.join(sh["root"], "b6c_part")
    rc, log = run_runner(part, common + ["--stop-after-epoch", "1"])
    assert rc == 0, log[-3000:]
    with open(os.path.join(part, "A1_seed5_train_ckpt.pt"), "rb") as fh:
        good = fh.read()
    for kind, raw in (("empty", b""), ("truncated", good[:len(good) // 2]),
                      ("garbage", b"\x00not a checkpoint" * 40)):
        d = os.path.join(sh["root"], "b6c_" + kind)
        os.makedirs(d)
        p_ = os.path.join(d, "A1_seed5_train_ckpt.pt")
        with open(p_, "wb") as fh:
            fh.write(raw)
        rc, log = run_runner(d, common + ["--resume"])
        assert rc == 0, (kind, "the damaged checkpoint failed the run", log[-3000:])
        assert "checkpoint A1_seed5_train_ckpt.pt could not be loaded" in log and \
            "moved aside as A1_seed5_train_ckpt.pt.corrupt1; starting from epoch 0" in log, \
            (kind, log[-2000:])
        with open(p_ + ".corrupt1", "rb") as fh:
            assert fh.read() == raw, kind
        a, b = record(ref, "A1_seed5"), record(d, "A1_seed5")
        assert a["history"] == b["history"] and a["L"] == b["L"], kind
        assert ckpt_sha(ref, "A1_seed5")[0] == ckpt_sha(d, "A1_seed5")[0], kind
    # an intact checkpoint still resumes
    rc, log = run_runner(part, common + ["--resume"])
    assert rc == 0 and "resumed from" in log and "could not be loaded" not in log, log[-2000:]
    assert record(part, "A1_seed5")["L"] == record(ref, "A1_seed5")["L"]


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


# ---------------------------------------------------------------- runner ---

def main():
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    try:
        import torch
        import sbi
        print("python %s | torch %s | sbi %s" % (sys.version.split()[0], torch.__version__, sbi.__version__))
    except ImportError as exc:
        print("needs torch and sbi: %s" % exc)
        return 1
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(r[0]) for r in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline[:150]))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()[-4000:]))
    if "measurement" in _SHARED:
        print("\n-- D-084 measurement: %s" % _SHARED["measurement"])
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    if "root" in _SHARED:
        shutil.rmtree(_SHARED["root"], ignore_errors=True)
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
