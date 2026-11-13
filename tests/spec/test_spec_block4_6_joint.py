"""Spec-derived tests of blocks 4-6 under hpc/joint (SPEC sec. 2.3; blocks
4, 5, 6). Oracles O-G6, O-G7, O4.1, O5.1-O5.3, O6.1 of the 2026-10-20 report.

The synthetic banks are written with the repository's own contract writer
(`latent_bank.write_shard`): it is the fixture, not the code under test. The
expected values (row offsets, per-shard draws, split counts, the torch
draws of the whole-buffer case) are computed here from the spec's rules.
"""
import glob
import math
import os
import sys

import numpy as np
import pytest

from conftest import JOINT

torch = pytest.importorskip("torch")

for _p in (os.path.join(JOINT, "stage3"), os.path.join(JOINT, "stage2"),
           os.path.join(JOINT, "stage1"), JOINT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import latent_bank as LB            # noqa: E402
from joint_batches import ShardStream   # noqa: E402
import run_joint_arms as RJA        # noqa: E402

P = 5
W = 16


def _sidecar(arm="S", digest_salt="1"):
    names = ["p%d" % i for i in range(P)]
    return LB.build_sidecar(param_names=names, bounds_theta=[[0.0, 1.0]] * P, coord="unit_box",
                            fs=100.0, w_size=0.01, T_win=0.16, W=W,
                            scale_convention="per_electrode_mean",
                            latent_spec={"kind": "campaign"},
                            nuisance_spec={"kind": "none"}, realisation_spec={"kind": "x"},
                            gap_spec={"kind": "none"}, generator_sha256=digest_salt * 64,
                            theta_withheld=False, arm=arm)


def _bank(root, sizes, seed=0, salt_of=None):
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    donor0 = 0
    for k, n in enumerate(sizes):
        arrays = {"x": rng.standard_normal((n, W)).astype(np.float32),
                  "theta": rng.uniform(0.05, 0.95, (n, P)).astype(np.float32),
                  "cls": np.full(n, -1, dtype=np.int64), "nu": np.zeros((n, 2), np.float32),
                  "realisation_id": np.arange(n, dtype=np.uint64),
                  "donor": (donor0 + np.arange(n) // 3).astype(np.int64),
                  "well": np.zeros(n, np.int64), "batch": np.zeros(n, np.int64),
                  "subregion": np.zeros(n, np.int64), "window_idx": np.arange(n, dtype=np.int64),
                  "contaminated": np.zeros(n, bool)}
        donor0 += n // 3 + 1
        side = _sidecar(digest_salt=(salt_of or {}).get(k, "1"))
        LB.write_shard(os.path.join(root, "shard_%04d.npz" % k), arrays, side)
    return sorted(glob.glob(os.path.join(root, "shard_*.npz")))


SIZES = (31, 17, 40, 9, 26, 22, 35, 14)     # unequal, as the spec's oracle


# ------------------------------------------------------------ block 4 -------

def test_campaign_log_prob_open_box():
    th = np.array([[0.5] * 3, [1e-9] * 3, [0.0, 0.5, 0.5], [0.5, 1.0, 0.5], [1.2, 0.5, 0.5],
                   [-0.1, 0.5, 0.5], [1 - 1e-9] * 3])
    lp = RJA.campaign_log_prob(th)
    assert lp.shape == (7,) and lp.dtype == np.float64
    np.testing.assert_array_equal(lp, [0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.inf, 0.0])


# ------------------------------------------------------ split (sec. 2.3) ----

def _split_by_hand(donors, fracs, seed):
    """The spec's rule: permutation of the distinct donors seeded by the split
    seed; round(f_tr n_d) to train, round(f_sel n_d) to selection, the rest
    to the report split."""
    uniq = np.unique(donors)
    perm = np.random.default_rng(seed).permutation(uniq)
    n = len(perm)
    a, b = int(round(fracs[0] * n)), int(round(fracs[1] * n))
    lab = {}
    for i, g in enumerate(perm):
        lab[g] = 0 if i < a else (1 if i < a + b else 2)
    return np.array([lab[g] for g in donors])


def test_grouped_split_counts_2612():
    donors = np.repeat(np.arange(2612), 3)
    idx, h = RJA.grouped_split(donors, fracs=(0.85, 0.10, 0.05), seed=0)
    per = [len(np.unique(donors[idx == k])) for k in (0, 1, 2)]
    assert per == [2220, 261, 131]                     # sec. 2.3, block 6 oracle
    np.testing.assert_array_equal(idx, _split_by_hand(donors, (0.85, 0.10, 0.05), 0))
    idx2, h2 = RJA.grouped_split(donors, fracs=(0.85, 0.10, 0.05), seed=0)
    assert h == h2 and len(h) == 64
    _, h3 = RJA.grouped_split(donors, fracs=(0.85, 0.10, 0.05), seed=1)
    assert h3 != h
    # a donor never straddles two splits
    for g in np.unique(donors):
        assert len(set(idx[donors == g])) == 1


def test_steps_formula():
    """sec. 2.3: S = ceil(P / N_ep * n_train / B); N_ep S B >= P n_train and
    < P n_train + N_ep B (the spec's example: P 2, N_ep 4, n_train 16, B 4 -> 2)."""
    assert math.ceil(2 / 4 * 16 / 4) == 2
    for P_, N, n, B in [(2, 4, 16, 4), (6.0, 20, 123457, 256), (40.0, 40, 9999, 1024), (1, 3, 7, 5)]:
        S = math.ceil(P_ / N * n / B)
        assert N * S * B >= P_ * n and N * S * B < P_ * n + N * B


# ------------------------------------------------------------ block 5 -------

def test_shard_index_offsets_and_arrays(tmp_path):
    paths = _bank(str(tmp_path / "b"), SIZES)
    idx = LB.ShardIndex(paths)
    np.testing.assert_array_equal(idx.offsets, np.concatenate([[0], np.cumsum(SIZES)]))
    assert idx.n_rows == sum(SIZES) and idx.n_shards == len(SIZES)
    th = np.concatenate([np.load(p)["theta"] for p in paths])
    dn = np.concatenate([np.load(p)["donor"] for p in paths])
    np.testing.assert_array_equal(idx.theta, th)
    np.testing.assert_array_equal(idx.donor, dn)
    assert idx.theta.dtype == np.float32
    for k in (0, 3, 7):
        x = idx.load_x(k)
        np.testing.assert_array_equal(x, np.load(paths[k])["x"])
        assert x.dtype == np.float32 and x.shape == (SIZES[k], W)
        np.testing.assert_array_equal(idx.rows_of(k), np.arange(idx.offsets[k], idx.offsets[k + 1]))


def test_shard_index_refuses_other_contract(tmp_path):
    paths = _bank(str(tmp_path / "b"), (5, 6, 7), salt_of={2: "2"})
    with pytest.raises(LB.ContractError):
        LB.ShardIndex(paths)


def _draws(stream, steps, n_rows):
    counts = np.zeros(n_rows, dtype=np.int64)
    max_res = 0
    for _ in range(steps):
        _, x, rows = stream.next()
        assert x.shape[0] == stream.batch_size
        np.add.at(counts, rows.numpy(), 1)
        max_res = max(max_res, len(stream.resident_shards))
    return counts, max_res


@pytest.mark.parametrize("K", [2, 7])
def test_stream_quota_rule_per_shard(tmp_path, K):
    """Block 5: every row's expected number of draws per pass is exactly 1
    whatever the shards' sizes; shard k receives n_k picks per pass; at most
    K shards resident; no unmasked row drawn. K = 7 of 8 shards exercises
    the 'nearly every shard resident' case of the pass permutation."""
    paths = _bank(str(tmp_path / "b"), SIZES)
    idx = LB.ShardIndex(paths)
    mask = np.ones(idx.n_rows, dtype=bool)
    mask[idx.offsets[2] + np.arange(0, 40, 4)] = False         # some unmasked rows
    n_m = int(mask.sum())
    B, passes = 8, 300
    s = ShardStream(idx, mask, B, K, seed=3)
    steps = int(round(passes * n_m / B))
    counts, max_res = _draws(s, steps, idx.n_rows)
    assert max_res <= K
    assert counts[~mask].sum() == 0
    assert (counts[mask] > 0).all()
    drawn = steps * B
    for k in range(idx.n_shards):
        sl = slice(idx.offsets[k], idx.offsets[k + 1])
        n_k = int(mask[sl].sum())
        expected = drawn * n_k / n_m
        # 2 %: the spec's tolerance for the quota rule's exactness over 400 passes
        assert abs(counts[sl].sum() - expected) <= 0.02 * expected, (k, counts[sl].sum(), expected)
    per_row = counts[mask] / (drawn / n_m)
    assert abs(per_row.mean() - 1.0) < 0.02
    # Poisson spread at 300 expected draws ~ 0.058; twice that, as the spec does at 400
    assert per_row.std() / per_row.mean() < 0.12


def test_stream_whole_buffer_equals_randint(tmp_path):
    """Block 5: with K >= n_shards a batch is torch.randint(0, n_rows, (B,),
    generator) over the masked rows in concatenated order, the generator
    seeded with seed + 1."""
    paths = _bank(str(tmp_path / "b"), (7, 12, 5))
    idx = LB.ShardIndex(paths)
    mask = np.ones(idx.n_rows, bool)
    mask[[0, 8, 20]] = False
    rows_masked = np.flatnonzero(mask)
    x_all = np.concatenate([np.load(p)["x"] for p in paths])
    s = ShardStream(idx, mask, 6, 99, seed=11)
    g = torch.Generator().manual_seed(12)
    for _ in range(50):
        th, x, rows = s.next()
        j = torch.randint(0, len(rows_masked), (6,), generator=g).numpy()
        np.testing.assert_array_equal(rows.numpy(), rows_masked[j])
        np.testing.assert_array_equal(x.numpy(), x_all[rows_masked[j]])
        np.testing.assert_array_equal(th.numpy(), idx.theta[rows_masked[j]])


def test_stream_state_roundtrip(tmp_path):
    paths = _bank(str(tmp_path / "b"), SIZES)
    idx = LB.ShardIndex(paths)
    mask = np.ones(idx.n_rows, bool)
    a = ShardStream(idx, mask, 5, 2, seed=4)
    for _ in range(37):
        a.next()
    st = a.state_dict()
    b = ShardStream(idx, mask, 5, 2, seed=4).load_state_dict(st)
    for _ in range(20):
        ra, rb = a.next()[2], b.next()[2]
        np.testing.assert_array_equal(ra.numpy(), rb.numpy())


def test_stream_refuses_mask_of_wrong_length(tmp_path):
    paths = _bank(str(tmp_path / "b"), (4, 5))
    idx = LB.ShardIndex(paths)
    with pytest.raises(ValueError):
        ShardStream(idx, np.ones(idx.n_rows + 1, bool), 2, 1)


# ------------------------------------------------- block 6: the runner -----

import json        # noqa: E402
import subprocess  # noqa: E402

RUNNER = os.path.join(JOINT, "stage3", "run_joint_arms.py")


@pytest.fixture(scope="module")
def fixture_bank(tmp_path_factory):
    """The blocks 4-6 fixture banks (8 shards of unequal sizes), built by the
    repository's own fixture builder: an input, not an oracle."""
    sys.path.insert(0, os.path.join(JOINT, "stage3"))
    import smoke_test_blocks_4_6 as SM
    root = str(tmp_path_factory.mktemp("bank"))
    sim, real = SM.build_banks(root)
    return sim


def _run(out_dir, sim, extra):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.pop("NCPUS", None)
    cmd = [sys.executable, RUNNER, "--arm", "A1", "--sim-shards", sim, "--out-dir", out_dir,
           "--threads", "1", "--n-posterior-draws", "8", "--embedding-size", "8",
           "--hidden-features", "16", "--num-transforms", "2", "--depth-exponent", "3",
           "--width-multiplier", "1.5"] + list(extra)
    r = subprocess.run(cmd, cwd=os.path.dirname(RUNNER), env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=1200)
    return r.returncode, r.stdout


def test_runner_passes_split_and_stem(tmp_path, fixture_bank):
    """Block 6: --passes -> S = ceil(P / N_ep * n_train / B) in the record as
    steps_per_epoch_effective; --split-fracs; a split seed other than the
    seed gives the stem <arm>_split<s>_seed<k>."""
    out = str(tmp_path / "r")
    rc, log = _run(out, fixture_bank, ["--epochs", "2", "--passes", "1.5", "--b-sim", "8",
                                       "--seed", "2", "--split-seed", "1",
                                       "--split-fracs", "0.85", "0.10", "0.05"])
    assert rc == 0, log[-3000:]
    rec = json.load(open(os.path.join(out, "A1_split1_seed2.json")))
    donors = np.concatenate([np.load(p)["donor"] for p in sorted(glob.glob(fixture_bank))])
    lab = _split_by_hand(donors, (0.85, 0.10, 0.05), 1)
    n_train = int((lab == 0).sum())
    assert rec["steps_per_epoch_effective"] == math.ceil(1.5 / 2 * n_train / 8)
    assert rec["passes"] == 1.5
    assert os.path.isfile(os.path.join(out, "A1_split1_seed2_perrow.npz"))
    pr = np.load(os.path.join(out, "A1_split1_seed2_perrow.npz"))
    # the report rows are the donors of split 2, by the spec's rule
    assert set(np.unique(pr["group"])) == set(np.unique(donors[lab == 2]))
    assert pr["nll"].dtype == np.float64 and np.all(np.isfinite(pr["nll"]))


def test_runner_refuses_fractions_not_summing_to_1(tmp_path, fixture_bank):
    rc, log = _run(str(tmp_path / "r"), fixture_bank, ["--epochs", "1", "--steps-per-epoch", "1",
                                                      "--split-fracs", "0.80", "0.10", "0.05"])
    assert rc != 0
    assert not glob.glob(str(tmp_path / "r" / "*.json"))


def test_bank_split_hash_tool(tmp_path, fixture_bank):
    """Block 7: the split tool's hash equals grouped_split's on the bank, its
    contract digest the sidecar's; pass line `SPLIT OK: hash=... contract=...`;
    n_train the rows of split 0 by the spec's rule."""
    from conftest import PIPE
    out = str(tmp_path / "split.json")
    r = subprocess.run([sys.executable, os.path.join(PIPE, "tools", "bank_split_hash.py"),
                        "--sim-shards", fixture_bank, "--split-fracs", "0.85", "0.10", "0.05",
                        "--split-seed", "0", "--out", out],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip().split("\n")[-1].startswith("SPLIT OK: hash=")
    rec = json.load(open(out))
    paths = sorted(glob.glob(fixture_bank))
    donors = np.concatenate([np.load(p)["donor"] for p in paths])
    _, h = RJA.grouped_split(donors, fracs=(0.85, 0.10, 0.05), seed=0)
    assert rec["split_hash"] == h
    side = json.load(open(os.path.splitext(paths[0])[0] + ".json"))
    assert rec["contract_digest"] == side["contract_digest"]
    lab = _split_by_hand(donors, (0.85, 0.10, 0.05), 0)
    assert rec["n_train"] == int((lab == 0).sum()) and rec["n_report"] == int((lab == 2).sum())
    assert rec["n_rows"] == len(donors)


def test_runner_same_seed_same_result_streaming(tmp_path, fixture_bank):
    """sec. 4 (randomness: the configuration's seeds) and step 6's
    reproducibility check: two runs of the streaming path (K = 2 < 8
    shards) at one seed and one thread give the same record numbers."""
    recs = []
    for k in (1, 2):
        out = str(tmp_path / ("r%d" % k))
        rc, log = _run(out, fixture_bank, ["--epochs", "2", "--steps-per-epoch", "3", "--b-sim", "8",
                                           "--seed", "0", "--buffer-shards", "2"])
        assert rc == 0, log[-3000:]
        recs.append(json.load(open(os.path.join(out, "A1_seed0.json"))))
        recs[-1]["_model_sha256"] = _model_sha(out, "A1_seed0")
    a, b = recs
    assert a["L"] == b["L"] and a["split_hash"] == b["split_hash"]
    assert a["history"] == b["history"]
    # t1-A: the record has no model_sha256 (it is in <stem>_ckpt.pt); read it there
    assert a["_model_sha256"] == b["_model_sha256"]
    assert a["loader"]["buffer_shards"] == 2


# ---------------------------------------------------------------------------
# t1-A (2026-11-13, D-131): extensions for blocks 4-6.
# ---------------------------------------------------------------------------

def test_campaign_log_prob_nan_is_not_inside():
    """Block 4: 0 inside the OPEN unit box, -inf outside; a NaN coordinate is
    not inside (it must not read as 0, a prior floor of a missing value)."""
    lp = RJA.campaign_log_prob(np.array([[np.nan, 0.5], [0.5, 0.5], [0.5, np.inf]]))
    np.testing.assert_array_equal(lp, [-np.inf, 0.0, -np.inf])


def test_shards_without_masked_rows_never_loaded(tmp_path):
    """Block 5: shards with no masked row are skipped -- never resident,
    nothing drawn from them -- also when fewer shards hold rows than K."""
    paths = _bank(str(tmp_path / "b"), SIZES)
    idx = LB.ShardIndex(paths)
    mask = np.zeros(idx.n_rows, bool)
    for k in (1, 4, 6):
        mask[idx.offsets[k]:idx.offsets[k + 1]] = True
    for K in (2, 4):
        s = ShardStream(idx, mask, 8, K, seed=5)
        seen = set()
        for _ in range(60):
            rows = s.next()[2].numpy()
            assert mask[rows].all()
            seen.update(s.resident_shards)
        assert seen <= {1, 4, 6}, (K, seen)


def _model_sha(out, stem):
    """The final checkpoint's `model_sha256` (joint_model's own digest of the
    state dict): the record does not carry it, so the artefact is read."""
    ck = torch.load(os.path.join(out, stem + "_ckpt.pt"), map_location="cpu", weights_only=False)
    return ck["model_sha256"]


BASE6 = ["--epochs", "3", "--steps-per-epoch", "2", "--b-sim", "8", "--seed", "0"]


def test_resume_and_damaged_checkpoint(tmp_path, fixture_bank):
    """Block 6: a run stopped after an epoch (`--stop-after-epoch`, exit 0,
    `stopped after epoch` line, the training checkpoint kept) and re-run
    with --resume gives the uninterrupted run's history, L and model digest
    bit for bit at one thread; the training checkpoint is removed once the
    final record is written. D-111 (block 6, amended 2026-10-29): a
    checkpoint that cannot be loaded (garbage bytes) under --resume is moved
    aside as `<name>.corrupt1`, logged, and the run equals a run without a
    checkpoint."""
    stem = "A1_seed0"
    out_a, out_b, out_c = (str(tmp_path / n) for n in ("a", "b", "c"))
    rc, log = _run(out_a, fixture_bank, BASE6)
    assert rc == 0, log[-3000:]
    rec_a = json.load(open(os.path.join(out_a, stem + ".json")))
    rc, log = _run(out_b, fixture_bank, BASE6 + ["--stop-after-epoch", "1"])
    assert rc == 0 and "stopped after epoch 1" in log, log[-2000:]
    assert os.path.isfile(os.path.join(out_b, stem + "_train_ckpt.pt"))
    assert not os.path.isfile(os.path.join(out_b, stem + ".json"))
    rc, log = _run(out_b, fixture_bank, BASE6 + ["--resume"])
    assert rc == 0, log[-3000:]
    rec_b = json.load(open(os.path.join(out_b, stem + ".json")))
    assert rec_b["history"] == rec_a["history"] and rec_b["L"] == rec_a["L"]
    assert _model_sha(out_b, stem) == _model_sha(out_a, stem)
    assert not os.path.isfile(os.path.join(out_b, stem + "_train_ckpt.pt"))
    os.makedirs(out_c)
    with open(os.path.join(out_c, stem + "_train_ckpt.pt"), "wb") as fh:
        fh.write(b"not a checkpoint at all" * 10)
    rc, log = _run(out_c, fixture_bank, BASE6 + ["--resume"])
    assert rc == 0, log[-3000:]
    assert os.path.isfile(os.path.join(out_c, stem + "_train_ckpt.pt.corrupt1"))
    assert "moved aside as %s_train_ckpt.pt.corrupt1" % stem in log
    rec_c = json.load(open(os.path.join(out_c, stem + ".json")))
    assert rec_c["history"] == rec_a["history"] and rec_c["L"] == rec_a["L"]
    assert _model_sha(out_c, stem) == _model_sha(out_a, stem)


def test_streaming_whole_buffer_equals_in_memory(tmp_path, fixture_bank):
    """Block 5's equivalence oracle: `--buffer-shards 99` (K >= n_shards)
    gives the same split_hash, history and L as `--buffer-shards 0` at one
    thread, bit for bit."""
    recs = []
    for k in ("0", "99"):
        out = str(tmp_path / ("k" + k))
        rc, log = _run(out, fixture_bank, ["--epochs", "2", "--steps-per-epoch", "3", "--b-sim", "8",
                                           "--seed", "0", "--buffer-shards", k])
        assert rc == 0, log[-3000:]
        recs.append(json.load(open(os.path.join(out, "A1_seed0.json"))))
    a, b = recs
    assert a["split_hash"] == b["split_hash"]
    assert a["history"] == b["history"] and a["L"] == b["L"]


def test_threads_default_from_ncpus(tmp_path, fixture_bank):
    """Block 6: --threads defaults to the NCPUS environment variable when
    set; the record carries the thread count."""
    out = str(tmp_path / "t")
    env = dict(os.environ, OMP_NUM_THREADS="1", NCPUS="3")
    cmd = [sys.executable, RUNNER, "--arm", "A1", "--sim-shards", fixture_bank, "--out-dir", out,
           "--n-posterior-draws", "8", "--embedding-size", "8", "--hidden-features", "16",
           "--num-transforms", "2", "--depth-exponent", "3", "--width-multiplier", "1.5",
           "--epochs", "1", "--steps-per-epoch", "1", "--b-sim", "8"]
    r = subprocess.run(cmd, cwd=os.path.dirname(RUNNER), env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-3000:]
    rec = json.load(open(os.path.join(out, "A1_seed0.json")))
    assert rec["threads"] == 3


def test_bank_wide_donor_ids_through_the_index_and_split(tmp_path):
    """Boundary blocks 12 -> 5 / 7: the bank-wide ids of a bench bank
    (shard k's donors k * ID_STRIDE + the local id, ID_STRIDE = 1,000,000)
    are opaque to ShardIndex and grouped_split: the index concatenates them
    as written, and the split assigns whole donors by the sec. 2.3 rule."""
    root = str(tmp_path / "b")
    paths = _bank(root, (12, 9, 15))
    # rewrite each shard's donor ids to the bank-wide form (the arrays only)
    for k, p in enumerate(paths):
        z = dict(np.load(p))
        local = np.arange(z["x"].shape[0]) // 3
        z["donor"] = (k * 1000000 + local).astype(np.int64)
        side = json.load(open(os.path.splitext(p)[0] + ".json"))
        LB.write_shard(p, z, side)
    idx = LB.ShardIndex(sorted(paths))
    want = np.concatenate([k * 1000000 + np.arange(n) // 3 for k, n in enumerate((12, 9, 15))])
    np.testing.assert_array_equal(idx.donor, want)
    split, _h = RJA.grouped_split(idx.donor, fracs=(0.5, 0.25, 0.25), seed=3)
    np.testing.assert_array_equal(split, _split_by_hand(idx.donor, (0.5, 0.25, 0.25), 3))
    for g in np.unique(idx.donor):
        assert len(set(split[idx.donor == g])) == 1


def test_steps_per_epoch_when_the_quotient_is_an_integer(tmp_path):
    """sec. 2.3 / block 6: S = ceil(P / N_ep * n_train / B_sim), with N_ep S
    B < P n_train + N_ep B. At P = 2.1, N_ep = 3, n_train = 40, B = 4 the
    quotient is exactly 7 (2.1 / 3 * 40 / 4 = 7), so S = 7; float64
    evaluation of the same expression gives 7.000000000000001, whose ceil is
    8 -- one extra batch per epoch (3 * 8 * 4 = 96 = P n_train + N_ep B).
    The bank: 80 single-row donors, split (0.5, 0.25, 0.25): 40 training
    rows by the sec. 2.3 rule. The runner's own `steps per epoch` line is
    read (--stop-after-epoch 0: one epoch, no record)."""
    from fractions import Fraction
    assert Fraction("2.1") / 3 * 40 / 4 == 7
    root = str(tmp_path / "b")
    os.makedirs(root)
    Wb, n = 2048, 80
    rng = np.random.default_rng(0)
    arrays = {"x": rng.standard_normal((n, Wb)).astype(np.float32),
              "theta": rng.uniform(0.05, 0.95, (n, P)).astype(np.float32),
              "cls": np.full(n, -1, dtype=np.int64), "nu": np.zeros((n, 2), np.float32),
              "realisation_id": np.arange(n, dtype=np.uint64), "donor": np.arange(n, dtype=np.int64),
              "well": np.zeros(n, np.int64), "batch": np.zeros(n, np.int64),
              "subregion": np.zeros(n, np.int64), "window_idx": np.arange(n, dtype=np.int64),
              "contaminated": np.zeros(n, bool)}
    side = LB.build_sidecar(param_names=["p%d" % i for i in range(P)], bounds_theta=[[0.0, 1.0]] * P,
                            coord="unit_box", fs=100.0, w_size=0.01, T_win=Wb / 100.0, W=Wb,
                            scale_convention="per_electrode_mean", latent_spec={"kind": "campaign"},
                            nuisance_spec={"kind": "none"}, realisation_spec={"kind": "x"},
                            gap_spec={"kind": "none"}, generator_sha256="1" * 64,
                            theta_withheld=False, arm="S")
    LB.write_shard(os.path.join(root, "shard_0000.npz"), arrays, side)
    rc, log = _run(str(tmp_path / "r"), os.path.join(root, "shard_*.npz"),
                   ["--epochs", "3", "--passes", "2.1", "--b-sim", "4",
                    "--split-fracs", "0.5", "0.25", "0.25", "--stop-after-epoch", "0"])
    assert rc == 0, log[-3000:]
    line = [l for l in log.split("\n") if l.startswith("steps per epoch")]
    assert line and "over 40 training rows" in line[0], line
    S = int(line[0].split(":")[1].split()[0])
    assert S == 7, line[0]
    assert 3 * S * 4 < 2.1 * 40 + 3 * 4
