"""Block 31: the replicate pair rule (specs/SPEC.md block 31, amendment 2).

Oracles (from the spec, before the code was read):
  O31.8  runner_extra(cfg, base) = base's items, then `--rep-pairs <pairs>
         --rep-group-key <key>`, one space between items, none leading or
         trailing; the committed files give linddup within_group / rep_group,
         giulia within_group / donor, synth cross_well / donor.
  O31.2  within_group: P = {(i, j): i < j, i, j in R (not surrogates), g_i = g_j};
         a draw picks group h with w_h = C(n_h, 2) / |P|, then a uniform pair
         of two distinct rows of h; each pair of P has probability 1 / |P|.
  O31.9  T31a: n = (4, 3, 1, 2), |P| = 10, w = (0.6, 0.3, 0, 0.1), `replicate
         groups : 3 of 4`.
  O31.10 LindDUP's shape (groups of 5, 12, 12, 1 wells of 54 windows): the
         share of within-well pairs in group h is (54 - 1) / (54 w_h - 1).
  O31.6  report lines: `replicate pairs  : <|P|>  (two rows of one <KEY> group,
         any well)` and `replicate groups : <paired> of <groups>`; under
         cross_well with KEY donor block 14's texts unchanged; `two wells of
         one <KEY> group` for another KEY.
  O31.3  cross_well with KEY donor = block 14 (the parent's batcher) exactly.
  O31.2' determinism: the same seed, the same pairs; a restored state, the
         same continuation (the draws use g_rep).
  O31.4  |P| = 0: no replicate batch.
"""

import importlib.util
import itertools
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

import _spec_util as U

import config as C                  # noqa: E402
import stages.replicate as RP       # noqa: E402

torch = pytest.importorskip("torch")

STAGE2 = os.path.join(U.JOINT, "stage2")
BASE_STAGE2 = "/tmp/base_tree/hpc/joint/stage2"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


JB = _load("spec_joint_batches_head", os.path.join(STAGE2, "joint_batches.py"))


# ------------------------------------------------------------------ O31.8 ---

class _Cfg(object):
    def __init__(self, pairs, key):
        self.data = {"replicate": {"pairs": pairs, "group_key": key}}

    def get(self, dotted):
        cur = self.data
        for part in dotted.split("."):
            cur = cur[part]
        return cur


def test_runner_extra_committed():
    want = {U.LINDDUP: "--rep-pairs within_group --rep-group-key rep_group",
            U.GIULIA: "--rep-pairs within_group --rep-group-key donor",
            U.SYNTH: "--rep-pairs cross_well --rep-group-key donor"}
    for path, flags in want.items():
        cfg = C.load_config(path, synthetic_ok=True)
        assert RP.runner_extra(cfg) == flags
        assert RP.runner_extra(cfg, "") == flags
        assert RP.runner_extra(cfg, "--n-posterior-draws 512") == \
            "--n-posterior-draws 512 " + flags
        assert RP.runner_extra(cfg, "  --probe-steps 20   --probe-val-rows 5000 ") == \
            "--probe-steps 20 --probe-val-rows 5000 " + flags


def test_runner_extra_refuses_a_value_of_two_words():
    with pytest.raises(ValueError):
        RP.runner_extra(_Cfg("within_group --x", "donor"))
    with pytest.raises(ValueError):
        RP.runner_extra(_Cfg("within_group", ""))


# ------------------------------------------------- the batcher's law ---

def _batcher(mod, groups, surrogate=None, rep_pairs="within_group", key="rep_group",
             wells=None, seed=7, b_rep=100, donors=None):
    n = len(groups)
    real_x = torch.arange(n, dtype=torch.float32).reshape(n, 1).repeat(1, 3)
    real_cls = torch.zeros(n, dtype=torch.long)
    sim_theta = torch.zeros((5, 2))
    sim_x = torch.zeros((5, 3))
    spec = mod.BatchSpec(b_sim=2, b_met=0, b_rep=b_rep)
    kw = {}
    if rep_pairs is not None:
        kw = {"rep_pairs": rep_pairs, "rep_group": np.asarray(groups), "rep_group_key": key}
    return mod.ThreeStreamBatcher(sim_theta, sim_x, real_x, real_cls,
                                  np.asarray(donors if donors is not None else groups),
                                  surrogate, spec, seed=seed, real_well=wells, **kw)


def _pairs_of(b, steps):
    out = []
    for _ in range(steps):
        r = b.rep_batch()
        if r is None:
            return None
        ig, igp = r[2]
        out.append(np.stack([ig.numpy(), igp.numpy()], axis=1))
    return np.concatenate(out, axis=0)


T31A_GROUPS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 3]
T31A_SURR = np.zeros(12, dtype=bool)
T31A_SURR[[4, 5]] = True


def test_t31a_counts_and_report():
    b = _batcher(JB, T31A_GROUPS, surrogate=T31A_SURR)
    assert b.n_pairs == 10
    rep = b.report().splitlines()
    assert "replicate pairs  : 10  (two rows of one rep_group group, any well)" in rep, rep
    assert "replicate groups : 3 of 4" in rep, rep


def test_t31a_law():
    b = _batcher(JB, T31A_GROUPS, surrogate=T31A_SURR, b_rep=100, seed=11)
    p = _pairs_of(b, 1000)                                   # 100,000 draws
    assert p.shape == (100000, 2)
    g = np.asarray(T31A_GROUPS)
    assert np.all(p[:, 0] < p[:, 1])                         # two distinct rows, i < j
    assert np.all(g[p[:, 0]] == g[p[:, 1]])                  # never across groups
    assert not np.any(T31A_SURR[p].any(axis=1))              # never a surrogate
    freq = np.bincount(g[p[:, 0]], minlength=4) / float(p.shape[0])
    # 0.01 is 6.5 binomial SDs at w = 0.6 and N = 1e5 (sqrt(0.24 / 1e5) = 0.0015)
    assert_allclose(freq, [0.6, 0.3, 0.0, 0.1], atol=0.01)
    # each of the 10 pairs of P equally likely: chi-square over the pairs, 9 dof;
    # p > 1e-3 at this fixed seed (a wrong law, e.g. a group-uniform draw, gives
    # p ~ 0 at N = 1e5)
    allowed = [(i, j) for i, j in itertools.combinations(range(12), 2)
               if g[i] == g[j] and not T31A_SURR[i] and not T31A_SURR[j]]
    assert len(allowed) == 10
    idx = dict((pr, k) for k, pr in enumerate(allowed))
    counts = np.bincount([idx[(int(i), int(j))] for i, j in p], minlength=10)
    assert stats.chisquare(counts).pvalue > 1e-3, counts


def test_lindup_shape_within_well_share():
    """O31.10 on 2,000,000 draws (amendment 2): 0.005 is 5 binomial SDs for the
    5-well group, whose share is (54 - 1) / (54 * 5 - 1) = 0.19703."""
    wells_per_group = [5, 12, 12, 1]
    groups, wells = [], []
    w_id = 0
    for h, nw in enumerate(wells_per_group):
        for _ in range(nw):
            groups += [h] * 54
            wells += [w_id] * 54
            w_id += 1
    groups, wells = np.asarray(groups), np.asarray(wells)
    b = _batcher(JB, groups, wells=wells, b_rep=1, seed=5)
    p = b.draw_within_pairs(2000000)
    n_h = np.asarray([54 * w for w in wells_per_group], dtype=float)
    C2 = n_h * (n_h - 1) / 2
    w_h = C2 / C2.sum()
    h = groups[p[:, 0]]
    freq = np.bincount(h, minlength=4) / float(p.shape[0])
    assert_allclose(freq, w_h, atol=0.005)
    same_well = wells[p[:, 0]] == wells[p[:, 1]]
    for k, nw in enumerate(wells_per_group):
        share = same_well[h == k].mean()
        expected = (54.0 - 1.0) / (54.0 * nw - 1.0)
        assert abs(share - expected) < 0.005, (k, share, expected)


def test_determinism_and_resume():
    b1 = _batcher(JB, T31A_GROUPS, surrogate=T31A_SURR, b_rep=8, seed=3)
    b2 = _batcher(JB, T31A_GROUPS, surrogate=T31A_SURR, b_rep=8, seed=3)
    assert_array_equal(_pairs_of(b1, 20), _pairs_of(b2, 20))
    b3 = _batcher(JB, T31A_GROUPS, surrogate=T31A_SURR, b_rep=8, seed=4)
    assert not np.array_equal(_pairs_of(b3, 20), _pairs_of(
        _batcher(JB, T31A_GROUPS, surrogate=T31A_SURR, b_rep=8, seed=3), 20))
    # a state saved after 7 steps continues as the uninterrupted batcher
    a = _batcher(JB, T31A_GROUPS, surrogate=T31A_SURR, b_rep=8, seed=9)
    _pairs_of(a, 7)
    st = a.state_dict()
    cont = _pairs_of(a, 5)
    fresh = _batcher(JB, T31A_GROUPS, surrogate=T31A_SURR, b_rep=8, seed=9)
    fresh.load_state_dict(st)
    assert_array_equal(_pairs_of(fresh, 5), cont)


def test_no_pair_gives_no_batch():
    b = _batcher(JB, [0, 1, 2, 3])
    assert b.n_pairs == 0 and b.rep_batch() is None
    assert "replicate groups : 0 of 4" in b.report()
    # a group whose rows are all surrogates contributes nothing
    b = _batcher(JB, [0, 0, 1], surrogate=np.asarray([True, True, False]))
    assert b.n_pairs == 0 and b.rep_batch() is None


def test_cross_well_texts():
    groups = np.asarray([0, 0, 1, 1, 2])
    wells = np.asarray([0, 1, 2, 3, 4])
    b = _batcher(JB, groups, rep_pairs="cross_well", key="rep_group", wells=wells)
    rep = b.report().splitlines()
    assert "replicate pairs  : 2  (two wells of one rep_group group)" in rep, rep
    bd = _batcher(JB, groups, rep_pairs="cross_well", key="donor", wells=wells)
    rep = bd.report().splitlines()
    assert "replicate pairs  : 2  (two wells of one donor)" in rep, rep
    assert any(ln.startswith("singleton donors : 1") for ln in rep), rep


# ------------------------------------------------------------------ O31.3 ---

@pytest.mark.skipif(not os.path.isdir(BASE_STAGE2), reason="the base tree is not extracted")
def test_default_equals_the_base_commits_batcher():
    """Without the new arguments, and with cross_well / donor given, the
    batcher draws what the base commit's draws, bit for bit."""
    base = _load("spec_joint_batches_base", os.path.join(BASE_STAGE2, "joint_batches.py"))
    rng = np.random.default_rng(0)
    donors = rng.integers(0, 6, size=40)
    wells = donors * 10 + rng.integers(0, 3, size=40)
    surr = rng.random(40) < 0.1
    for seed in (0, 1, 5):
        ref = _batcher(base, donors, surrogate=surr, rep_pairs=None, wells=wells, seed=seed,
                       b_rep=6)
        new_default = _batcher(JB, donors, surrogate=surr, rep_pairs=None, wells=wells,
                               seed=seed, b_rep=6)
        new_explicit = _batcher(JB, donors, surrogate=surr, rep_pairs="cross_well",
                                key="donor", wells=wells, seed=seed, b_rep=6)
        assert_array_equal(ref.pairs, new_default.pairs)
        assert_array_equal(ref.pairs, new_explicit.pairs)
        r = _pairs_of(ref, 15)
        assert_array_equal(r, _pairs_of(new_default, 15))
        assert_array_equal(r, _pairs_of(new_explicit, 15))
        assert ref.report() == new_default.report() == new_explicit.report()
