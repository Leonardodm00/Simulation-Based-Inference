"""Spec-derived tests of block 14 (arms A2 and A5 on a cohort; SPEC.md sec.
2.4 the searches, block 14: F-ab, F-aa, F-t, F-aw). Oracles O14.1-O14.6 of
the t1-B report.

The pairs are checked against a brute-force enumeration written here from
F-ab's definition; the runner's counts (48 cross-well pairs over 12 donors,
3 metric classes, L0 = 0) follow from the banks' shapes and sec. 2.4 / block
11 by hand.
"""
import itertools
import json
import os
import subprocess

import numpy as np
import pytest

import t1b_helpers as H

H.joint_paths()
torch = pytest.importorskip("torch")
import joint_batches as JB              # noqa: E402  (code under test)
import run_joint_arms as RJA            # noqa: E402  (code under test)


def _brute(donor, well=None):
    out = []
    for i, j in itertools.combinations(range(len(donor)), 2):
        if donor[i] == donor[j] and (well is None or well[i] != well[j]):
            out.append((i, j))
    return out


def _singletons(donor, well=None):
    n = 0
    for d in set(donor):
        rows = [i for i in range(len(donor)) if donor[i] == d]
        if len(rows) < 2 or (well is not None and len(set(well[i] for i in rows)) < 2):
            n += 1
    return n


def test_o14_1_by_hand_eight_rows():
    """8 rows: donor 0 in wells (0, 0, 1, 1), donor 1 in wells (2, 3), donor 2
    twice in well 4. Cross-well pairs: 4 + 1 + 0 = 5, one singleton donor;
    without wells C(4,2) + 1 + 1 = 8 pairs."""
    donor = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    well = np.array([0, 0, 1, 1, 2, 3, 4, 4])
    pairs, n_single = JB.enumerate_donor_pairs(donor, well)
    assert pairs.dtype == np.int64 and pairs.shape == (5, 2)
    assert sorted(map(tuple, pairs.tolist())) == [(0, 2), (0, 3), (1, 2), (1, 3), (4, 5)]
    assert n_single == 1
    pairs, n_single = JB.enumerate_donor_pairs(donor)
    assert pairs.shape == (8, 2) and n_single == 0


@pytest.mark.parametrize("seed", range(6))
def test_o14_1_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 60
    donor = rng.integers(0, 12, n)
    well = donor * 10 + rng.integers(0, 3, n)
    for w in (well, None):
        pairs, n_single = JB.enumerate_donor_pairs(donor, w)
        got = sorted(map(tuple, pairs.tolist()))
        assert got == _brute(donor.tolist(), None if w is None else w.tolist())
        assert all(i < j for i, j in got)
        assert n_single == _singletons(donor.tolist(), None if w is None else w.tolist())


def test_o14_2_spread_pairs_round_robin():
    """At most k pairs, a subset of the input, spread over the donors: a donor
    with pairs left never trails another by more than one; k pairs cover
    min(k, n_donors) donors."""
    rng = np.random.default_rng(3)
    donor = np.repeat(np.arange(30), rng.integers(2, 7, 30))
    pairs, _ = JB.enumerate_donor_pairs(donor)
    for k in (5, 30, 64, 10 ** 6):
        take = RJA.spread_pairs(pairs, donor, k)
        assert take.shape == (min(k, pairs.shape[0]), 2)
        s_all = set(map(tuple, pairs.tolist()))
        s_take = list(map(tuple, take.tolist()))
        assert set(s_take) <= s_all and len(set(s_take)) == len(s_take)
        if k >= pairs.shape[0]:
            continue
        cnt = {d: 0 for d in np.unique(donor)}
        avail = {d: 0 for d in np.unique(donor)}
        for i, _j in pairs.tolist():
            avail[donor[i]] += 1
        for i, _j in s_take:
            cnt[donor[i]] += 1
        for a in cnt:
            for b in cnt:
                if cnt[a] < avail[a]:
                    assert cnt[b] <= cnt[a] + 1, (k, a, b, cnt[a], cnt[b])
        assert sum(1 for d in cnt if cnt[d]) == min(k, len(cnt))


def test_o14_3_metric_class_count():
    assert RJA.metric_class_count(np.array([0, 1, 2, 0, -1, 2]), "real") == 3
    assert RJA.metric_class_count(np.array([1, 0]), "real") == 2
    for bad in ([-1, -1, -1], [0, 0, -1], [0, 2], [1, 2], []):
        with pytest.raises(SystemExit):
            RJA.metric_class_count(np.array(bad, dtype=np.int64), "simulated")


def test_o14_4_refuse_shared_bench_seed():
    s = {"latent_spec": {"seed": 5}}
    with pytest.raises(SystemExit) as e:
        RJA.refuse_shared_bench_seed(s, {"latent_spec": {"seed": 5}})
    assert "one --seed" in str(e.value)
    assert RJA.refuse_shared_bench_seed(s, {"latent_spec": {"seed": 6}}) is None
    assert RJA.refuse_shared_bench_seed({"latent_spec": {"kind": "campaign"}},
                                        {"latent_spec": {"seed": 5}}) is None
    assert RJA.refuse_shared_bench_seed(s, None) is None


def test_o14_1_batcher_pairs_and_report():
    donor = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    well = np.array([0, 0, 1, 1, 2, 3, 4, 4])
    th = torch.zeros(4, 3)
    x = torch.zeros(4, 5)
    rx = torch.zeros(8, 5)
    rc = torch.zeros(8, dtype=torch.long)
    b = JB.ThreeStreamBatcher(th, x, rx, rc, donor, None, JB.BatchSpec(2, 0, 2), seed=0,
                              real_well=well)
    assert sorted(map(tuple, b.pairs.tolist())) == _brute(donor.tolist(), well.tolist())
    assert "replicate pairs  : 5  (two wells of one donor)" in b.report()
    b0 = JB.ThreeStreamBatcher(th, x, rx, rc, donor, None, JB.BatchSpec(2, 0, 2), seed=0)
    assert b0.pairs.shape[0] == 8
    assert "replicate pairs  : 8  (any two rows of one donor)" in b0.report()
    for _ in range(20):                       # every replicate batch is a cross-well pair
        _xa, _xb, (ig, igp) = b.rep_batch()
        for i, j in zip(ig.tolist(), igp.tolist()):
            assert donor[i] == donor[j] and well[i] != well[j]


# --------------------------------------------------------------- the runner ---

BENCH = ["--provider", "bench", "--fs", "10.0", "--T-win", "12.8"]


@pytest.fixture(scope="module")
def banks():
    root = H.tmpdir("t1b_b14_")
    out = {}

    def build(name, args):
        d = os.path.join(root, name)
        rc, txt = H.run_builder(d, BENCH + args)
        assert rc == 0, txt
        out[name] = os.path.join(d, "shard_*.npz")

    build("S", ["--arm", "S", "--prior", "box", "--n-traces", "80", "--wells-per-donor", "2",
                "--n-windows", "2", "--seed", "41"])
    build("R", ["--arm", "R", "--donors-per-class", "4", "--n-traces", "24", "--wells-per-donor",
                "2", "--n-windows", "2", "--class-centres", "regular", "--centre-scale", "0.6",
                "--seed", "42"])
    build("R1", ["--arm", "R", "--donors-per-class", "4", "--n-traces", "12", "--wells-per-donor",
                 "1", "--n-per-theta", "1", "--n-windows", "2", "--seed", "43"])
    build("Rsame", ["--arm", "R", "--donors-per-class", "4", "--n-traces", "24", "--wells-per-donor",
                    "2", "--n-windows", "2", "--seed", "41"])
    out["root"] = root
    yield out
    H.rmtree(root)


def _run(arm, sim, real, out_dir):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [H.PYTHON, H.RUNNER, "--arm", arm, "--seed", "0", "--sim-shards", sim, "--out-dir",
           out_dir, "--threads", "1", "--epochs", "2", "--steps-per-epoch", "3", "--b-sim", "24",
           "--b-met", "12", "--b-rep", "4", "--n-posterior-draws", "16", "--embedding-size", "8",
           "--hidden-features", "24", "--num-transforms", "3", "--depth-exponent", "3",
           "--width-multiplier", "1.5"]
    if real:
        cmd += ["--real-shards", real]
    r = subprocess.run(cmd, cwd=H.STAGE3, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=1800)
    return r.returncode, r.stdout


@pytest.mark.parametrize("arm", ["A1", "A2", "A5"])
def test_o14_5_runner_arms_on_box_and_cohort(banks, arm):
    out_dir = os.path.join(banks["root"], "runs_" + arm)
    rc, txt = _run(arm, banks["S"], banks["R"], out_dir)
    assert rc == 0, txt[-3000:]
    with open(os.path.join(out_dir, "%s_seed0.json" % arm)) as fh:
        rec = json.load(fh)
    assert rec["prior"] == "box_unit_box"
    assert rec["L0"] == 0.0 and rec["L0_pseudo_real"] == 0.0
    assert rec["endpoint_primary"] == "pseudo_real"
    assert rec["L_pseudo_real"] is not None and np.isfinite(rec["L_pseudo_real"])
    pr = np.load(os.path.join(out_dir, "%s_seed0_perrow.npz" % arm))
    assert pr["nll_pseudo_real"].shape == (48,)                 # 3 x 4 donors x 2 wells x 2 windows
    np.testing.assert_allclose(np.mean(pr["nll_pseudo_real"]), rec["L_pseudo_real"], rtol=1e-12)
    rep = rec["replicate"]
    assert rep["pair_kind"] == "two wells of one donor"
    assert rep["n_pairs"] == 48 and rep["n_donors_scored"] == 12          # 2 x 2 per donor, <= 64
    assert "replicate pairs  : 48  (two wells of one donor)" in txt
    if arm == "A2":
        assert "metric classes   : 3 (on the real arm)" in txt
    else:
        assert "metric classes" not in txt


def test_o14_6_A5_refused_on_one_well_per_donor(banks):
    rc, txt = _run("A5", banks["S"], banks["R1"], os.path.join(banks["root"], "runs_r1_a5"))
    assert rc != 0 and "has no replicate pair" in txt, txt[-2000:]


def test_o14_6_A1_diagnostic_fallback_on_one_well(banks):
    out_dir = os.path.join(banks["root"], "runs_r1_a1")
    rc, txt = _run("A1", banks["S"], banks["R1"], out_dir)
    assert rc == 0, txt[-3000:]
    with open(os.path.join(out_dir, "A1_seed0.json")) as fh:
        rep = json.load(fh)["replicate"]
    assert rep["pair_kind"] == "rows of one well (no donor has two wells)"
    assert rep["n_pairs"] == 12 and rep["n_donors_scored"] == 12     # 1 pair (2 windows) per donor


def test_o14_6_one_seed_refused(banks):
    rc, txt = _run("A1", banks["S"], banks["Rsame"], os.path.join(banks["root"], "runs_same"))
    assert rc != 0 and "one --seed" in txt, txt[-2000:]


def test_o14_6_A2s_on_box_bank_refused(banks):
    rc, txt = _run("A2s", banks["S"], banks["R"], os.path.join(banks["root"], "runs_a2s"))
    assert rc != 0 and "at least two classes" in txt, txt[-2000:]


def test_o11_dry_run_names_the_box(banks):
    """Block 11's module list names the runner's dry run."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([H.PYTHON, H.RUNNER, "--arm", "A1", "--sim-shards", banks["S"],
                        "--real-shards", banks["R"], "--out-dir", os.path.join(banks["root"], "dry"),
                        "--dry-run"], cwd=H.STAGE3, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:]
    assert "box bank (unit box, D-118)" in r.stdout
    assert not os.path.exists(os.path.join(banks["root"], "dry", "A1_seed0.json"))
