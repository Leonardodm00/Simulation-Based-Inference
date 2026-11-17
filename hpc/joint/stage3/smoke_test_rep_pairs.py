"""Smoke test for the replicate pair rule -- Block 31 in specs/SPEC.md
(D-146, D-147, D-148 (3); wave W3).

The rule: a real row's group is the value of a real-bank id array (the KEY:
donor, rep_group, batch or well). `within_group` pairs any two different
non-surrogate rows of one group, any well: P = {(i, j) : i < j, g_i = g_j},
|P| = sum_h n_h (n_h - 1) / 2, each drawn pair uniform on P (a group h with
probability w_h = [n_h (n_h - 1) / 2] / |P|, then two distinct rows of h),
P never enumerated, the draws through the batcher's replicate generator.
`cross_well` is block 14's rule (two wells of one group) with the KEY in
place of the donor, and is the default.

Checks (SPEC's T31a-T31f; the template's oracle kinds in brackets)
    test_t31a_counts           [known answer, reference, invariants] 12 rows by
                               hand: n = (4, 3, 1, 2), |P| = 10, w = (0.6, 0.3,
                               0, 0.1), `replicate groups : 3 of 4`; 100,000
                               draws through rep_batch: inside P, no surrogate,
                               no repeated row, every pair of P drawn, the
                               group and pair frequencies
    test_t31b_law              [known answer, statistical] LindDUP's shape
                               (groups of 5, 12, 12 and 1 wells of 54 windows):
                               w_h exactly, the group frequencies on 200,000
                               draws, the share of pairs inside one well
                               (54 - 1) / (54 w - 1), the same-180-s share
                               8 / (54 w - 1), the per-row frequencies
    test_t31c_regression       [reference] cross_well with KEY donor gives B14's
                               pairs and the default call's draws; the KEY in
                               place of the donor; the stage 2 suites pass
    test_t31c_parent_batcher   [reference] the batcher of the parent commit
                               (88c32e5, block 14's) draws the same pairs,
                               the same 50 steps and prints the same report
    test_t31d_determinism      [determinism] the same seed gives the same pairs;
                               a state_dict saved (through torch.save, as the
                               checkpoint does) and restored draws the same
                               continuation; the other streams do not see the
                               rule
    test_t31e_runner           [contract] the runner: A5 within_group on a
                               cohort bank carrying rep_group (its report lines
                               and its diagnostic), within_group on donor, the
                               refusal without rep_group, F-aa under
                               within_group, the flags without a real bank
    test_t31e_parent_runner    [reference] A1 and A5 without the flags, and A5
                               with `--rep-pairs cross_well --rep-group-key
                               donor`, against the parent commit's runner: L,
                               history and the replicate block bit for bit
    test_t31e_b14_unchanged    B14's runner checks (smoke_test_blocks_11_14.py)
                               pass
    test_t31f_runner_extra     [known answer] replicate.runner_extra on the
                               committed configurations, with and without base,
                               through the tuner's split into the runner's parser
    test_contract              shapes and dtypes of the sampler and the batcher
    test_edge_cases            |P| = 0, no real arm, string ids, a bad rule or
                               length, KEY well under cross_well, the
                               diagnostic's round robin
    test_convergence           skipped (not a discretisation)

Run
    python3 hpc/joint/stage3/smoke_test_rep_pairs.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO. About two to three minutes at one
        thread (the runner checks start the runner 13 times). Needs the
        training environment (numpy, torch, sbi, zuko; the in-repo hpc/dsn)
        for the batcher and runner checks; T31f needs only the stdlib
        pipeline. The parent-commit checks need git and commit 88c32e5 in
        the checkout (they are SKIPPED with the reason otherwise). Builds its
        own banks in a temporary folder with the builder's reference
        provider; nothing is read from davinci.
    python -m pytest hpc/joint/stage3/smoke_test_rep_pairs.py -q
        The same checks through pytest.

Debug
    A failed check_close() saves both arrays to
        hpc/joint/stage3/artifacts/<check name>.npz

Pure ASCII, LF only.
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_JOINT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PIPELINE = os.path.join(_REPO, "hpc", "pipeline")
for _p in (os.path.join(_JOINT, "stage1"), os.path.join(_JOINT, "stage2"), _JOINT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 12345
PARENT = "88c32e5"          # block 31's parent: the contract commit (block 14's batcher and runner)
REPORT_PACKAGES = ("numpy", "scipy", "torch", "sbi")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []
_SHARED = {}

BUILDER = os.path.join(_JOINT, "stage1", "build_latent_bank.py")
RUNNER = os.path.join(_HERE, "run_joint_arms.py")
PYTHON = sys.executable
# small reference-provider shards (the fixture generator), W = 64 samples
REF = ["--provider", "reference", "--T-win", "1.28", "--fs", "50",
       "--n-neurons", "20", "--n-windows", "2"]


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def _need_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("needs torch: %s" % exc)


def _need_sbi():
    _need_torch()
    try:
        import sbi  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("needs sbi: %s" % exc)


def tmp_root():
    if "root" not in _SHARED:
        _SHARED["root"] = tempfile.mkdtemp(prefix="reppairs_")
    return _SHARED["root"]


def make_batcher(group_ids, surrogate=None, rep_pairs="within_group", key="rep_group",
                 b_rep=4, seed=SEED, well=None, donor=None, rep_group_given=True,
                 module=None, b_sim=2, b_met=0):
    """A ThreeStreamBatcher on a real arm of len(group_ids) rows whose window
    is its own row index (real_x[i] = i, W = 1), so a gathered window names
    its row; a 6-row simulated arm. `module`: another joint_batches (the
    parent commit's) to build it from."""
    import torch
    JB = module
    if JB is None:
        import joint_batches as JB
    n = len(group_ids)
    real_x = torch.arange(n, dtype=torch.float32)[:, None]
    cls = torch.zeros(n, dtype=torch.long)
    sim_t = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    sim_x = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    kw = {}
    if module is None:
        kw = {"rep_pairs": rep_pairs, "rep_group_key": key,
              "rep_group": (np.asarray(group_ids) if rep_group_given else None)}
    don = np.asarray(group_ids) if donor is None else np.asarray(donor)
    return JB.ThreeStreamBatcher(sim_t, sim_x, real_x, cls, don, surrogate,
                                 JB.BatchSpec(b_sim, b_met, b_rep), seed=seed,
                                 real_well=well, **kw)


def draw_steps(b, n_steps):
    """n_steps rep_batch() calls -> (n_steps * b_rep, 2) int64 rows; checks
    that every gathered window is its row's (real_x[i] = i)."""
    out = []
    for _ in range(n_steps):
        xg, xgp, (ig, igp) = b.rep_batch()
        assert np.array_equal(xg[:, 0].numpy(), ig.numpy().astype(np.float32))
        assert np.array_equal(xgp[:, 0].numpy(), igp.numpy().astype(np.float32))
        out.append(np.stack([ig.numpy(), igp.numpy()], axis=1))
    return np.concatenate(out, axis=0).astype(np.int64)


def lindup_shape():
    """LindDUP's real arm after D-145 as T31b states it: four groups of 5,
    12, 12 and 1 wells, each well 54 windows (9 subregions x 6 windows)."""
    wells_per_group = (5, 12, 12, 1)
    group, well, sub, win = [], [], [], []
    wid = 0
    for h, w in enumerate(wells_per_group):
        for _ in range(w):
            for s in range(9):
                for t in range(6):
                    group.append(h)
                    well.append(wid)
                    sub.append(s)
                    win.append(t)
            wid += 1
    return (wells_per_group, np.asarray(group, np.int64), np.asarray(well, np.int64),
            np.asarray(sub, np.int64), np.asarray(win, np.int64))


def _git(*args):
    return subprocess.run(["git", "-C", _REPO] + list(args), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)


def _parent_available():
    try:
        r = _git("rev-parse", "--verify", "--quiet", PARENT + "^{commit}")
    except OSError as exc:
        return "git not available: %s" % exc
    if r.returncode != 0:
        return "commit %s not in this checkout" % PARENT
    return None


def parent_module(rel, name):
    """The parent commit's version of a module (git show), loaded under
    another name from a temporary file."""
    r = _git("show", "%s:%s" % (PARENT, rel))
    if r.returncode != 0:
        raise unittest.SkipTest("git show %s:%s failed: %s" % (PARENT, rel, r.stderr.strip()))
    d = os.path.join(tmp_root(), "parent_mod")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".py")
    with open(path, "w") as fh:
        fh.write(r.stdout)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parent_tree():
    """The parent commit's hpc/ tree (git archive), once."""
    if "parent_tree" in _SHARED:
        return _SHARED["parent_tree"]
    d = os.path.join(tmp_root(), "parent_tree")
    os.makedirs(d, exist_ok=True)
    arch = subprocess.run(["git", "-C", _REPO, "archive", PARENT, "hpc"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if arch.returncode != 0:
        raise unittest.SkipTest("git archive %s failed: %s" % (PARENT, arch.stderr[-300:]))
    subprocess.run(["tar", "-x", "-C", d], input=arch.stdout, check=True)
    _SHARED["parent_tree"] = d
    return d


def build(out, args):
    cmd = [PYTHON, BUILDER, "--out-dir", out] + [str(a) for a in args]
    r = subprocess.run(cmd, cwd=os.path.dirname(BUILDER), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)
    if r.returncode != 0:
        raise AssertionError("builder failed (%d): %s" % (r.returncode, r.stdout.strip()[-600:]))


def with_rep_group(src, dst, fn):
    """Copy the bank `src` to `dst`, every shard gaining the array rep_group
    = fn(merged arrays) split back by shard (the shard contract's optional
    array; the sidecar, hence the contract digest, unchanged)."""
    import latent_bank as LB
    paths = sorted(p for p in os.listdir(src) if p.endswith(".npz"))
    parts = [LB.read_shard(os.path.join(src, p)) for p in paths]
    merged = dict((k, np.concatenate([a[k] for a, _s in parts], axis=0))
                  for k in parts[0][0])
    rg = np.asarray(fn(merged), dtype=np.int64)
    assert rg.shape == (merged["x"].shape[0],), rg.shape
    os.makedirs(dst, exist_ok=True)
    lo = 0
    for p, (a, side) in zip(paths, parts):
        n = a["x"].shape[0]
        b = dict(a)
        b["rep_group"] = rg[lo:lo + n]
        lo += n
        LB.write_shard(os.path.join(dst, p), b, side)
    return rg, merged


def runner_banks():
    """The runner checks' banks (B14's shapes): a box arm S, 40 donors x 2
    wells x 2 windows, seed 1; a cohort arm R, 3 classes x 4 donors x 2
    wells x 2 windows = 48 rows, seed 2, without rep_group; R_rg, R with
    rep_group = cls except the last row, alone in group 99 (groups of 16,
    16, 15 and 1 rows); R_single, R with every row its own group."""
    if "rb" in _SHARED:
        return _SHARED["rb"]
    root = os.path.join(tmp_root(), "banks")
    S, R, RG, R1 = (os.path.join(root, n) for n in ("S", "R", "R_rg", "R_single"))
    build(S, ["--arm", "S", "--prior", "box", "--n-traces", "80", "--n-classes", "3",
              "--seed", "1"] + REF)
    build(R, ["--arm", "R", "--n-classes", "3", "--donors-per-class", "4", "--n-traces",
              "24", "--class-centres", "regular", "--centre-scale", "0.6",
              "--seed", "2"] + REF)

    def cls_but_last(m):
        g = np.asarray(m["cls"], dtype=np.int64).copy()
        g[-1] = 99
        return g
    rg, merged = with_rep_group(R, RG, cls_but_last)
    with_rep_group(R, R1, lambda m: np.arange(m["x"].shape[0], dtype=np.int64))
    _SHARED["rb"] = dict(S=os.path.join(S, "*.npz"), R=os.path.join(R, "*.npz"),
                         RG=os.path.join(RG, "*.npz"), R1=os.path.join(R1, "*.npz"),
                         rg=rg, merged=merged, root=root)
    return _SHARED["rb"]


def run_arm(arm, sim, real, out_dir, extra=(), runner=RUNNER):
    """B14's runner call: 2 epochs of 3 steps, one thread."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    cmd = [PYTHON, runner, "--arm", arm, "--seed", "0", "--sim-shards", sim,
           "--out-dir", out_dir, "--threads", "1",
           "--epochs", "2", "--steps-per-epoch", "3", "--b-sim", "24",
           "--b-met", "12", "--b-rep", "4", "--n-posterior-draws", "16",
           "--embedding-size", "8", "--hidden-features", "24",
           "--num-transforms", "3", "--depth-exponent", "3",
           "--width-multiplier", "1.5"]
    if real is not None:
        cmd += ["--real-shards", real]
    cmd += list(extra)
    r = subprocess.run(cmd, cwd=os.path.dirname(runner), env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=1200)
    return r.returncode, r.stdout


def read_record(out_dir, arm):
    with open(os.path.join(out_dir, "%s_seed0.json" % arm)) as fh:
        return json.load(fh)


def comparable(rec):
    """The parts of a runner record that the rule could move, as one JSON
    string (NaN-safe: the history of an arm without the term carries NaN)."""
    keep = dict((k, rec.get(k)) for k in ("L", "L0", "L_pseudo_real", "split_hash",
                                            "history", "replicate", "val_npe",
                                            "endpoint_primary", "r_eff"))
    return json.dumps(keep, sort_keys=True)


# ---------------------------------------------------------------- checks ---

def test_t31a_counts():
    """T31a: 12 rows -- rows 0-5 in group 0 with rows 4 and 5 surrogates,
    rows 6-8 in group 1, row 9 in group 2, rows 10-11 in group 3."""
    _need_torch()
    g = np.array([0] * 6 + [1] * 3 + [2] + [3] * 2)
    sur = np.zeros(12, dtype=bool)
    sur[[4, 5]] = True
    b = make_batcher(g, surrogate=sur, b_rep=4, seed=SEED)
    gs = b.group_sampler
    # known answer, by hand
    assert gs.sizes.tolist() == [4, 3, 1, 2], gs.sizes
    assert b.n_pairs == 10, b.n_pairs                      # 6 + 3 + 0 + 1
    w = np.zeros(4)
    w[gs.paired] = gs.weights
    # 0.6, 0.3, 0.1 are k/10 in float64: one rounding of a division
    check_close("t31a_w", w, [0.6, 0.3, 0.0, 0.1], rtol=0, atol=1e-15)
    rep = b.report()
    assert "replicate pairs  : 10  (two rows of one rep_group group, any well)" in rep, rep
    assert "replicate groups : 3 of 4" in rep, rep
    assert b.pairs is None                                  # P is not enumerated
    # reference: P by brute force on the tiny input
    P = sorted((i, j) for i in range(12) for j in range(i + 1, 12)
               if g[i] == g[j] and not sur[i] and not sur[j])
    assert len(P) == 10, P
    # 100,000 draws through the training path: 25,000 steps of b_rep = 4
    d = draw_steps(b, 25000)
    assert d.shape == (100000, 2), d.shape
    # invariants: no pair crosses a group, contains a surrogate or repeats a row
    assert np.all(g[d[:, 0]] == g[d[:, 1]]), "a pair crosses a group"
    assert not np.any(sur[d[:, 0]] | sur[d[:, 1]]), "a pair contains a surrogate"
    assert np.all(d[:, 0] < d[:, 1]), "a pair repeats a row (or is not i < j)"
    drawn = set(map(tuple, d.tolist()))
    assert drawn == set(P), (sorted(drawn), P)               # inside P, all of P
    # group frequencies within 0.01 of w: the binomial SD at 100,000 draws is
    # at most sqrt(0.6 * 0.4 / 1e5) = 0.0015, so 0.01 is >= 6.4 SD
    freq = np.bincount(g[d[:, 0]], minlength=4) / float(d.shape[0])
    check_close("t31a_group_freq", freq, [0.6, 0.3, 0.0, 0.1], rtol=0, atol=0.01)
    # uniform on P (beyond the SPEC's oracle): each pair 1/|P| = 0.1, the
    # binomial SD sqrt(0.1 * 0.9 / 1e5) = 0.00095, so 0.005 is 5.3 SD
    idx = dict((p, k) for k, p in enumerate(P))
    pf = np.bincount([idx[t] for t in map(tuple, d.tolist())], minlength=10) / float(d.shape[0])
    check_close("t31a_pair_freq", pf, np.full(10, 0.1), rtol=0, atol=0.005)


def test_t31b_law():
    """T31b: the law on LindDUP's shape (groups of 5, 12, 12 and 1 wells of
    54 windows: 9 subregions x 6 windows)."""
    _need_torch()
    wpg, group, well, sub, win = lindup_shape()
    n_w = 54
    assert group.size == 30 * n_w
    b = make_batcher(group, well=well, b_rep=4, seed=SEED)
    gs = b.group_sampler
    # known answer: w_h = C_h / |P| exactly (fractions), the batcher's to
    # float64 roundoff (one division of integers below 2^53)
    C = [Fraction(n_w * w * (n_w * w - 1), 2) for w in wpg]
    Ptot = sum(C)
    assert b.n_pairs == int(Ptot) == 457002, (b.n_pairs, Ptot)
    w_exact = np.array([float(c / Ptot) for c in C])
    check_close("t31b_w", gs.weights, w_exact, rtol=1e-15, atol=0.0)
    share_exact = np.array([(n_w - 1) / (n_w * w - 1.0) for w in wpg])
    # the SPEC's values (pair_mix.py of 2026-11-17), to their printed digits
    assert np.round(share_exact, 3).tolist() == [0.197, 0.082, 0.082, 1.0], share_exact
    same_t_exact = np.array([8.0 / (n_w * w - 1.0) for w in wpg])   # D-147's Implies
    assert np.round(same_t_exact, 3).tolist() == [0.030, 0.012, 0.012, 0.151], same_t_exact
    rep = b.report()
    assert "replicate pairs  : 457002  (two rows of one rep_group group, any well)" in rep, rep
    assert "replicate groups : 4 of 4" in rep, rep

    # 200,000 draws (20 steps of 10,000: the draws are i.i.d., so the
    # chunking does not change the law): each group's frequency within 0.005
    # of w_h; the binomial SD is at most sqrt(0.4587 * 0.5413 / 2e5) = 0.0011,
    # so 0.005 is >= 4.5 SD for every group
    d = np.concatenate([b.draw_within_pairs(10000) for _ in range(20)], axis=0)
    gh = group[d[:, 0]]
    assert np.all(gh == group[d[:, 1]])
    freq = np.bincount(gh, minlength=4) / float(d.shape[0])
    check_close("t31b_group_freq", freq, w_exact, rtol=0, atol=0.005)

    # The share of pairs inside one well, per group, within 0.005 of
    # (54 - 1) / (54 w - 1). [ISSUE reported to the orchestrator, block 31:
    # at the SPEC's 200,000 draws the 5-well group receives about 15,900
    # draws, the share's binomial SD there is sqrt(0.197 * 0.803 / 15,900) =
    # 0.0032 and 0.005 is 1.6 SD -- a correct sampler fails 11 % of seeds
    # (measured: 38 of 300 seeds). With 2,000,000 draws the 5-well group's
    # SD is 0.0010 and 0.005 is 5.0 SD (the 12-well groups' 17 SD; the 1-well
    # group's share is exactly 1).]
    d2 = np.concatenate([d] + [b.draw_within_pairs(100000) for _ in range(18)], axis=0)
    assert d2.shape[0] == 2000000
    g2 = group[d2[:, 0]]
    same_well = well[d2[:, 0]] == well[d2[:, 1]]
    same_t = same_well & (win[d2[:, 0]] == win[d2[:, 1]])
    shares, shares_t, n_h = [], [], []
    for h in range(4):
        m = g2 == h
        n_h.append(int(m.sum()))
        shares.append(float(same_well[m].mean()))
        shares_t.append(float(same_t[m].mean()))
    _SHARED["t31b"] = (n_h, shares, shares_t)
    check_close("t31b_share_one_well", shares, share_exact, rtol=0, atol=0.005)
    # inside one well at the same 180 s (D-147's Implies, beyond the SPEC's
    # oracle): 8 / (54 w - 1); tolerance 5 binomial SDs at the group's draws
    sd_t = np.sqrt(same_t_exact * (1 - same_t_exact) / np.asarray(n_h, dtype=float))
    for h in range(4):
        assert abs(shares_t[h] - same_t_exact[h]) < 5 * sd_t[h], (h, shares_t[h], same_t_exact[h])
    # rows uniform within their group (beyond the SPEC's oracle): a row of
    # group h is in a draw of h with probability 2 / n_h; over 1,620 rows the
    # largest |deviation| of a binomial count stays below 5.5 SD (each row
    # exceeds it with probability 4e-8)
    cnt = np.bincount(d2.ravel(), minlength=group.size).astype(float)
    nh_rows = np.asarray([n_w * w for w in wpg])[group]
    trials = np.asarray(n_h, dtype=float)[group]
    p_row = 2.0 / nh_rows
    z = (cnt - trials * p_row) / np.sqrt(trials * p_row * (1 - p_row))
    assert np.max(np.abs(z)) < 5.5, float(np.max(np.abs(z)))


def test_t31c_regression():
    """T31c: cross_well with KEY donor is block 14's rule; the KEY in place of
    the donor; the stage 2 suites pass."""
    _need_torch()
    from joint_batches import enumerate_donor_pairs
    don = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    wel = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    b14 = [(0, 2), (0, 3), (1, 2), (1, 3), (6, 7)]          # B14's by hand
    default = make_batcher(don, well=wel, module=None, rep_pairs="cross_well", key="donor",
                           rep_group_given=False)
    explicit = make_batcher(don, well=wel, rep_pairs="cross_well", key="donor")
    for b in (default, explicit):
        assert [tuple(p) for p in b.pairs.tolist()] == b14, b.pairs
        assert b.n_pairs == 5 and b.n_singleton_donors == 1
        assert "replicate pairs  : 5  (two wells of one donor)" in b.report(), b.report()
        assert "singleton donors : 1   <- D12" in b.report()
    assert default.report() == explicit.report()
    assert_array_equal(draw_steps(default, 50), draw_steps(explicit, 50))
    # the call without any block-31 argument (as before block 31)
    import joint_batches as JB
    import torch
    plain = JB.ThreeStreamBatcher(torch.zeros(6, 2), torch.zeros(6, 3),
                                  torch.arange(8, dtype=torch.float32)[:, None],
                                  torch.zeros(8, dtype=torch.long), don, None,
                                  JB.BatchSpec(2, 0, 4), seed=SEED, real_well=wel)
    assert plain.rep_pairs == "cross_well" and plain.rep_group_key == "donor"
    assert_array_equal(plain.pairs, default.pairs)
    # the KEY in place of the donor: KEY batch joins the donors 0 and 1
    # (batch 5: rows 0-5 in wells 0, 0, 1, 1, 2, 2) and keeps donor 2 alone
    bat = np.array([5, 5, 5, 5, 5, 5, 6, 6])
    kb = make_batcher(bat, well=wel, donor=don, rep_pairs="cross_well", key="batch")
    ref, n1 = enumerate_donor_pairs(bat, wel)
    assert_array_equal(kb.pairs, ref)
    # by hand: batch 5, C(6, 2) = 15 pairs less the 3 inside one well = 12;
    # batch 6, wells 3 and 4: 1
    assert kb.n_pairs == 12 + 1 == ref.shape[0] and n1 == 0, kb.pairs
    assert "replicate pairs  : 13  (two wells of one batch group)" in kb.report(), kb.report()
    assert "singleton groups : 0" in kb.report(), kb.report()
    # the stage 2 suites pass
    s2 = os.path.join(_JOINT, "stage2")
    for suite, want in (("smoke_test_joint.py", re.compile(r"(\d+) passed, 0 failed")),
                        ("smoke_test_joint_losses.py", None)):
        r = subprocess.run([PYTHON, suite], cwd=s2, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=1800)
        assert r.returncode == 0, "%s exit %d: %s" % (suite, r.returncode, r.stdout[-1500:])
        if want is not None:
            m = want.search(r.stdout)
            assert m, r.stdout[-800:]
            _SHARED.setdefault("stage2", []).append("%s: %s" % (suite, m.group(0)))
        else:
            _SHARED.setdefault("stage2", []).append("%s: exit 0" % suite)


def test_t31c_parent_batcher():
    """T31c against the parent commit's batcher (block 14's code)."""
    _need_torch()
    why = _parent_available()
    if why:
        raise unittest.SkipTest(why)
    PB = parent_module("hpc/joint/stage2/joint_batches.py", "joint_batches_parent")
    don = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    wel = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    new = make_batcher(don, well=wel, rep_pairs="cross_well", key="donor", rep_group_given=False)
    old = make_batcher(don, well=wel, module=PB)
    assert_array_equal(new.pairs, old.pairs)
    assert new.report() == old.report(), (new.report(), old.report())
    assert_array_equal(draw_steps(new, 50), draw_steps(old, 50))
    # J4's setting: string-free donors with surrogates, no wells; and the
    # explicit KEY donor
    donor = np.repeat(np.arange(20), 2)
    sur = np.zeros(40, dtype=bool)
    sur[:6] = True
    a = make_batcher(donor, surrogate=sur, rep_pairs="cross_well", key="donor")
    o = make_batcher(donor, surrogate=sur, module=PB)
    assert_array_equal(a.pairs, o.pairs)
    assert a.report() == o.report()
    for _ in range(30):
        na, no = a.next(), o.next()
        for k in ("sim", "rep"):
            for u, v in zip(na[k][:2], no[k][:2]):
                assert np.array_equal(u.numpy(), v.numpy()), k
        assert np.array_equal(na["rep"][2][0].numpy(), no["rep"][2][0].numpy())
    assert_array_equal(a.state_dict()["g_rep"].numpy(), o.state_dict()["g_rep"].numpy())


def test_t31d_determinism():
    """T31d: the same seed gives the same pairs; a batcher saved with
    state_dict and restored draws the same continuation."""
    _need_torch()
    import torch
    wpg, group, well, _sub, _win = lindup_shape()
    a = make_batcher(group, well=well, b_rep=8, seed=7)
    b = make_batcher(group, well=well, b_rep=8, seed=7)
    assert_array_equal(draw_steps(a, 30), draw_steps(b, 30))
    c = make_batcher(group, well=well, b_rep=8, seed=8)
    assert not np.array_equal(draw_steps(c, 30), draw_steps(make_batcher(group, well=well, b_rep=8, seed=7), 30))
    for rule in ("within_group", "cross_well"):
        key = "rep_group" if rule == "within_group" else "donor"
        x = make_batcher(group, well=well, b_rep=8, seed=11, rep_pairs=rule, key=key)
        draw_steps(x, 13)
        x.next()                                   # a whole step, as train_joint takes
        buf = io.BytesIO()
        torch.save(x.state_dict(), buf)            # the checkpoint's route (block 6)
        buf.seek(0)
        st = torch.load(buf, weights_only=False)
        cont = draw_steps(x, 20)
        y = make_batcher(group, well=well, b_rep=8, seed=11, rep_pairs=rule, key=key)
        y.load_state_dict(st)
        assert_array_equal(draw_steps(y, 20), cont)
    # the rule does not move the other streams (one generator per stream, J9)
    p = make_batcher(group, well=well, b_rep=8, seed=3, rep_pairs="within_group", key="rep_group")
    q = make_batcher(group, well=well, b_rep=8, seed=3, rep_pairs="cross_well", key="donor")
    for _ in range(10):
        bp, bq = p.next(), q.next()
        assert np.array_equal(bp["sim"][2].numpy(), bq["sim"][2].numpy())


def test_t31e_runner():
    """T31e: the runner under the flags."""
    _need_sbi()
    B = runner_banks()
    runs = os.path.join(tmp_root(), "runs_e")
    # the cohort's rep_group: groups of 16, 16, 15 and 1 rows (by design: 4
    # donors x 2 wells x 2 windows per class, the last row alone)
    _u, counts = np.unique(B["rg"], return_counts=True)
    assert sorted(counts.tolist()) == [1, 15, 16, 16], counts
    P_rg = int(sum(c * (c - 1) // 2 for c in counts.tolist()))
    assert P_rg == 2 * 120 + 105 == 345, P_rg
    rc, txt = run_arm("A5", B["S"], B["RG"], runs,
                      ["--rep-pairs", "within_group", "--rep-group-key", "rep_group"])
    assert rc == 0, "A5 within_group exit %d: %s" % (rc, txt[-2000:])
    assert "replicate pairs  : 345  (two rows of one rep_group group, any well)" in txt, txt[-2500:]
    assert "replicate groups : 3 of 4" in txt, txt[-2500:]
    rec = read_record(runs, "A5")
    rep = rec["replicate"]
    assert rep["pair_kind"] == "two rows of one rep_group group, any well", rep["pair_kind"]
    assert rep["n_pairs"] == 64 and rep["n_groups_scored"] == 3, (rep["n_pairs"], rep["n_groups_scored"])
    assert len(rep["T"]) == 64
    assert rec["config"]["rep_pairs"] == "within_group" and rec["config"]["rep_group_key"] == "rep_group"
    assert rec["history"][-1]["rep"] != 0.0, rec["history"]       # the term trained
    # within_group on the donor (giulia's rule): 12 donors x C(4, 2) = 72 pairs
    rc, txt = run_arm("A5", B["S"], B["R"], os.path.join(tmp_root(), "runs_e_donor"),
                      ["--rep-pairs", "within_group", "--rep-group-key", "donor"])
    assert rc == 0, txt[-2000:]
    assert "replicate pairs  : 72  (two rows of one donor group, any well)" in txt, txt[-2500:]
    assert "replicate groups : 12 of 12" in txt, txt[-2500:]
    rep = read_record(os.path.join(tmp_root(), "runs_e_donor"), "A5")["replicate"]
    assert rep["n_pairs"] == 64 and rep["n_groups_scored"] == 12, rep
    # the refusal on a bank without rep_group (A1 too: the real arm is loaded)
    want = ("--rep-group-key rep_group: the real bank has no 'rep_group' array (a bank "
            "built before Sbi-extractor block 9 has no rep_group)")
    for arm in ("A1", "A5"):
        rc, txt = run_arm(arm, B["S"], B["R"], runs, ["--rep-group-key", "rep_group"])
        assert rc != 0 and want in txt, (arm, rc, txt[-800:])
    # F-aa under within_group: every row its own group, |P| = 0
    rc, txt = run_arm("A5", B["S"], B["R1"], runs,
                      ["--rep-pairs", "within_group", "--rep-group-key", "rep_group"])
    assert rc != 0 and "has no replicate pair" in txt, txt[-800:]
    assert "--rep-pairs within_group --rep-group-key rep_group" in txt, txt[-800:]
    assert "replicate pairs  : 0  (two rows of one rep_group group, any well)" in txt
    assert "replicate groups : 0 of 48" in txt, txt[-1500:]
    # F-aa under cross_well with KEY well: two wells of one well, P empty
    rc, txt = run_arm("A5", B["S"], B["R"], runs, ["--rep-group-key", "well"])
    assert rc != 0 and "has no replicate pair" in txt, txt[-800:]
    assert "--rep-pairs cross_well --rep-group-key well" in txt, txt[-800:]
    # without a real bank (S-A1's trials) the flags are accepted and unused
    rc, txt = run_arm("A1", B["S"], None, os.path.join(tmp_root(), "runs_e_noreal"),
                      ["--rep-pairs", "within_group", "--rep-group-key", "rep_group"])
    assert rc == 0, txt[-1500:]
    rec = read_record(os.path.join(tmp_root(), "runs_e_noreal"), "A1")
    assert "replicate" not in rec and rec["endpoint_primary"] == "simulated (arm R absent)", rec.keys()


def test_t31e_parent_runner():
    """T31e: A1 and A5 without the flags, and A5 with the defaults given
    explicitly, against the parent commit's runner on the same banks, one
    thread: the record's L, history, endpoint and replicate block bit for
    bit."""
    _need_sbi()
    why = _parent_available()
    if why:
        raise unittest.SkipTest(why)
    B = runner_banks()
    old_runner = os.path.join(parent_tree(), "hpc", "joint", "stage3", "run_joint_arms.py")
    for arm, extra in (("A1", ()), ("A5", ()),
                       ("A5", ("--rep-pairs", "cross_well", "--rep-group-key", "donor"))):
        tag = "%s_%s" % (arm, "explicit" if extra else "default")
        new_dir = os.path.join(tmp_root(), "par_new_" + tag)
        old_dir = os.path.join(tmp_root(), "par_old_" + tag)
        rc, txt = run_arm(arm, B["S"], B["R"], new_dir, extra)
        assert rc == 0, (tag, txt[-1500:])
        rc0, txt0 = run_arm(arm, B["S"], B["R"], old_dir, (), runner=old_runner)
        assert rc0 == 0, (tag, txt0[-1500:])
        a, o = read_record(new_dir, arm), read_record(old_dir, arm)
        assert comparable(a) == comparable(o), (tag, comparable(a)[:600], comparable(o)[:600])
        line = [ln for ln in txt.splitlines() if ln.startswith("replicate pairs")]
        line0 = [ln for ln in txt0.splitlines() if ln.startswith("replicate pairs")]
        assert line == line0 == ["replicate pairs  : 48  (two wells of one donor)"], (line, line0)
        _SHARED.setdefault("parent", []).append("%s L %.12g" % (tag, a["L"]))


def test_t31e_b14_unchanged():
    """T31e: B14's runner checks pass (A1, A2, A5 without the flags; the
    refusals)."""
    _need_sbi()
    import smoke_test_blocks_11_14 as B1114
    try:
        B1114.test_b14_runner_arms()
        B1114.test_b14_runner_refusals()
    finally:
        if "root" in B1114._SHARED:
            shutil.rmtree(B1114._SHARED["root"], ignore_errors=True)


def test_t31f_runner_extra():
    """T31f: replicate.runner_extra on the committed configurations."""
    # the pipeline's `config`, not the DSN tree's hpc/dsn/config.py
    if sys.path[0] != _PIPELINE:
        sys.path.insert(0, _PIPELINE)
    loaded = sys.modules.get("config")
    if loaded is not None and not os.path.abspath(getattr(loaded, "__file__", "")).startswith(_PIPELINE):
        del sys.modules["config"]
    import config as C
    assert os.path.abspath(C.__file__).startswith(_PIPELINE), C.__file__
    from stages import replicate as RP
    from stages.runs import RunView
    _s4 = os.path.join(_JOINT, "stage4")
    if _s4 not in sys.path:
        sys.path.append(_s4)
    expect = {"linddup": "--rep-pairs within_group --rep-group-key rep_group",
              "giulia": "--rep-pairs within_group --rep-group-key donor",
              "synth": "--rep-pairs cross_well --rep-group-key donor"}
    try:
        import npe_tune_joint as TUNE                        # the tuner's split
        tuner_split = TUNE.runner_extra
    except Exception as exc:                                 # its env absent
        tuner_split = None
        _SHARED["t31f_note"] = "the tuner did not import (%s): str.split used" % exc
    try:
        import run_joint_arms as RJA
        parser = RJA.build_parser()
    except Exception as exc:
        parser = None
        _SHARED["t31f_note"] = "the runner did not import (%s): its parser not checked" % exc
    for tag, want in sorted(expect.items()):
        path = os.path.join(_PIPELINE, "projects", tag, tag + ".json")
        cfg = C.load_config(path, synthetic_ok=(tag == "synth"))
        assert RP.runner_extra(cfg) == want, (tag, RP.runner_extra(cfg))
        assert RP.runner_extra(cfg, "") == want
        assert RP.runner_extra(cfg, base="--n-posterior-draws 512") == \
            "--n-posterior-draws 512 " + want
        got = RP.runner_extra(cfg, "  --probe-steps 3   --probe-val-rows 50 ")
        assert got == "--probe-steps 3 --probe-val-rows 50 " + want, repr(got)
        assert got == got.strip() and "  " not in got
        toks = tuner_split(got) if tuner_split else got.split()
        if parser is not None:
            ns = parser.parse_args(["--arm", "A5", "--sim-shards", "x", "--out-dir", "y"] + toks)
            assert (ns.rep_pairs, ns.rep_group_key) == (cfg.get("replicate.pairs"),
                                                        cfg.get("replicate.group_key"))
            assert (ns.probe_steps, ns.probe_val_rows) == (3, 50)
        if tag == "synth":
            for run in cfg.get("synthetic.searches"):
                assert RP.runner_extra(RunView(cfg, run)) == want, run
    # a hand-built configuration: a value of two words, a missing key
    bad = C.Config({"replicate": {"pairs": "within group", "group_key": "donor"}}, "x", "0")
    try:
        RP.runner_extra(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a two-word replicate.pairs was accepted")
    try:
        RP.runner_extra(C.Config({}, "x", "0"))
    except KeyError:
        pass
    else:
        raise AssertionError("a configuration without replicate.* was accepted")


def test_contract():
    """Shapes and dtypes: GroupPairSampler.draw -> (n, 2) int64, i < j; the
    batcher's rep batch; `pairs` None and `n_pairs` an int under
    within_group."""
    _need_torch()
    from joint_batches import GroupPairSampler
    g = np.array([3, 3, 3, 7, 7, 9])
    s = GroupPairSampler(g)
    assert s.groups.tolist() == [3, 7, 9] and s.sizes.tolist() == [3, 2, 1]
    assert s.n_pairs == 4 and s.n_groups == 3 and s.n_groups_paired == 2
    assert s.members(0).tolist() == [0, 1, 2] and s.members(2).tolist() == [5]
    p = s.draw(1000, make_rng())
    assert p.shape == (1000, 2) and p.dtype == np.int64, (p.shape, p.dtype)
    assert np.all(p[:, 0] < p[:, 1])
    z = s.draw(0, make_rng())
    assert z.shape == (0, 2) and z.dtype == np.int64
    # rows other than 0..n-1 are returned as given
    s2 = GroupPairSampler(g, rows=np.array([10, 11, 12, 20, 21, 30]))
    assert set(np.unique(s2.draw(500, make_rng())).tolist()) <= {10, 11, 12, 20, 21}
    b = make_batcher(np.repeat(np.arange(4), 3), b_rep=5)
    xg, xgp, (ig, igp) = b.rep_batch()
    assert tuple(xg.shape) == (5, 1) and tuple(xgp.shape) == (5, 1)
    import torch
    assert ig.dtype == torch.long and igp.dtype == torch.long
    assert b.pairs is None and isinstance(b.n_pairs, int) and b.n_pairs == 12


def test_edge_cases():
    """|P| = 0, no real arm, string ids, a bad rule or length, KEY well
    under cross_well, the diagnostic's round robin."""
    _need_torch()
    import torch
    import joint_batches as JB
    from run_joint_arms import spread_group_pairs
    # |P| = 0: every group of one row
    b = make_batcher(np.arange(5), b_rep=4)
    assert b.n_pairs == 0 and b.rep_batch() is None
    assert "replicate pairs  : 0  (two rows of one rep_group group, any well)" in b.report()
    assert "replicate groups : 0 of 5" in b.report()
    try:
        b.draw_within_pairs(3)
    except ValueError:
        pass
    else:
        raise AssertionError("a draw from an empty P did not raise")
    # no real arm
    nb = JB.ThreeStreamBatcher(torch.zeros(6, 2), torch.zeros(6, 3), spec=JB.BatchSpec(2, 0, 4),
                               rep_pairs="within_group", rep_group_key="rep_group")
    assert nb.n_pairs == 0 and nb.rep_batch() is None and "replicate groups : 0 of 0" in nb.report()
    # a bad rule, a KEY array of the wrong length
    for kw in ({"rep_pairs": "any"}, {"rep_group": np.zeros(3)}):
        try:
            JB.ThreeStreamBatcher(torch.zeros(6, 2), torch.zeros(6, 3), torch.zeros(5, 1),
                                  torch.zeros(5, dtype=torch.long), np.arange(5), None,
                                  JB.BatchSpec(2, 0, 4), **kw)
        except ValueError:
            continue
        raise AssertionError("accepted %r" % (kw,))
    # string ids of a narrow dtype with surrogates (J4e's trap): masked, not relabelled
    ids = np.array(["d%d" % (i // 2) for i in range(20)])
    sur = np.zeros(20, dtype=bool)
    sur[:12] = True
    sb = make_batcher(ids, surrogate=sur, b_rep=6)
    d = draw_steps(sb, 200)
    assert not np.any(sur[d.ravel()]) and np.all(ids[d[:, 0]] == ids[d[:, 1]])
    assert sb.n_pairs == 4 and "replicate groups : 4 of 4" in sb.report(), sb.report()
    # one group: every draw from it
    one = make_batcher(np.zeros(7, dtype=np.int64), b_rep=3)
    assert one.n_pairs == 21 and draw_steps(one, 50).max() < 7
    # KEY well under cross_well: two wells of one well -- P empty by definition
    wel = np.repeat(np.arange(4), 3)
    kw_ = make_batcher(wel, well=wel, rep_pairs="cross_well", key="well")
    assert kw_.n_pairs == 0 and "(two wells of one well group)" in kw_.report(), kw_.report()
    # the diagnostic's pairs: |P| <= k -> every pair once (brute force)
    g = np.array([5, 5, 5, 2, 2, 8, 9, 9, 9, 9])
    P = sorted((i, j) for i in range(10) for j in range(i + 1, 10) if g[i] == g[j])
    take, ng = spread_group_pairs(g, 64, make_rng(0))
    assert sorted(map(tuple, take.tolist())) == P and ng == 3, (take, ng)
    # round robin in increasing id, one fresh pair per group per turn: the
    # first turn takes groups 2, 5, 9 (8 has one row); group 2 is then spent
    take, ng = spread_group_pairs(g, 4, make_rng(1))
    assert [int(g[i]) for i in take[:, 0]] == [2, 5, 9, 5], take
    assert take.shape == (4, 2) and ng == 3 and np.all(take[:, 0] < take[:, 1])
    # 64 pairs over three large groups: 22, 21, 21; no pair twice; seeded
    big = np.repeat(np.array([4, 1, 7]), 40)
    take, ng = spread_group_pairs(big, 64, make_rng(2))
    assert np.bincount(big[take[:, 0]], minlength=8)[[1, 4, 7]].tolist() == [22, 21, 21]
    assert len(set(map(tuple, take.tolist()))) == 64 and ng == 3
    assert np.all(big[take[:, 0]] == big[take[:, 1]])
    assert_array_equal(take, spread_group_pairs(big, 64, make_rng(2))[0])
    e, ne = spread_group_pairs(np.arange(6), 64, make_rng(0))
    assert e.shape == (0, 2) and ne == 0


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


# ---------------------------------------------------------------- runner ---

def _environment():
    import importlib.metadata
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
    print(_environment())

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
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline[:150]}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()[-4000:]}")

    if "t31b" in _SHARED:
        n_h, shares, shares_t = _SHARED["t31b"]
        print("\n-- T31b (2,000,000 draws): draws per group %s; share inside one well %s; "
              "inside one well at the same 180 s %s"
              % (n_h, ["%.4f" % v for v in shares], ["%.4f" % v for v in shares_t]))
    for line in _SHARED.get("stage2", []):
        print("-- T31c stage 2 suite %s" % line)
    for line in _SHARED.get("parent", []):
        print("-- T31e against %s: %s (bit for bit)" % (PARENT, line))
    if "t31f_note" in _SHARED:
        print("-- T31f: %s" % _SHARED["t31f_note"])
    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    if "root" in _SHARED:
        shutil.rmtree(_SHARED["root"], ignore_errors=True)
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
