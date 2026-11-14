"""Chain t2-C2, step 6: two runs of the same configuration (seeds included)
in two work roots, through the real entry point, give identical results
(C-O13): the banks' arrays and builds, the split, the proposals and trial
ids, every trial's nll and L_pseudo_real (the runner at one thread, block 6:
bit for bit), the finalists, the members' NLLs and the ensembles.
Nothing here calls the code under test: the files are compared."""

import glob
import os

import numpy as np
import pytest

import c2_helpers as H


def _arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return dict((k, z[k]) for k in z.files)


@pytest.fixture(scope="module")
def both(run1, run2):
    for r in (run1, run2):
        assert r[5] is not None and "done (exit 0)" in r[5], r[5]
    return run1[1], run2[1]


@pytest.mark.parametrize("b", ["sim", "R0", "R3"])
def test_banks_identical(both, b):
    c1, c2 = both
    p1 = sorted(glob.glob(os.path.join(H.bank_dir(c1, b), "shard_*.npz")))
    p2 = sorted(glob.glob(os.path.join(H.bank_dir(c2, b), "shard_*.npz")))
    assert [os.path.basename(p) for p in p1] == [os.path.basename(p) for p in p2]
    for a, c in zip(p1, p2):
        x, y = _arrays(a), _arrays(c)
        assert sorted(x) == sorted(y)
        for k in x:
            assert np.array_equal(x[k], y[k]), (os.path.basename(a), k)
        s1, s2 = H.read_json(a[:-4] + ".json"), H.read_json(c[:-4] + ".json")
        assert s1["contract_digest"] == s2["contract_digest"]
    r1 = H.read_json(os.path.join(H.bank_dir(c1, b), "BANK_RECORD.json"))
    r2 = H.read_json(os.path.join(H.bank_dir(c2, b), "BANK_RECORD.json"))
    assert r1["build"] == r2["build"] and r1["contract_digest"] == r2["contract_digest"]


@pytest.mark.parametrize("q", H.SEARCHES, ids=[q["name"] for q in H.SEARCHES])
def test_search_identical(both, q):
    c1, c2 = both
    b1 = H.read_json(os.path.join(H.campaign_dir(c1, q), "bank_split.json"))
    b2 = H.read_json(os.path.join(H.campaign_dir(c2, q), "bank_split.json"))
    for k in ("split_hash", "contract_digest", "n_train", "n_sel", "n_report"):
        assert b1[k] == b2[k], k
    rf1 = H.read_json(os.path.join(H.campaign_dir(c1, q), "pipeline_rounds.json"))
    rf2 = H.read_json(os.path.join(H.campaign_dir(c2, q), "pipeline_rounds.json"))
    assert [r["trial_ids"] for r in rf1["rounds"]] == [r["trial_ids"] for r in rf2["rounds"]]
    l1, l2 = H.ledger(c1, q), H.ledger(c2, q)
    assert sorted(l1) == sorted(l2)
    for tid in l1:
        assert l1[tid]["config"] == l2[tid]["config"]
        assert l1[tid]["nll"] == l2[tid]["nll"], (tid, l1[tid]["nll"], l2[tid]["nll"])
        assert l1[tid]["L_pseudo_real"] == l2[tid]["L_pseudo_real"], tid
    s1 = H.read_json(os.path.join(H.campaign_dir(c1, q), "SEARCH_COMPLETE.json"))
    s2 = H.read_json(os.path.join(H.campaign_dir(c2, q), "SEARCH_COMPLETE.json"))
    assert s1["finalist"] == s2["finalist"]


@pytest.mark.parametrize("q,alt", [(q, a) for q in H.SEARCHES for a in (False, True)
                                   if a is False or q["alt_best_fit"]],
                         ids=lambda v: v["name"] if isinstance(v, dict) else ("alt" if v else "main"))
def test_best_fit_identical(both, q, alt):
    c1, c2 = both
    e1 = H.read_json(os.path.join(H.best_dir(c1, q, alt), "BEST_FIT_COMPLETE.json"))
    e2 = H.read_json(os.path.join(H.best_dir(c2, q, alt), "BEST_FIT_COMPLETE.json"))
    assert e1["finalist"]["trial_id"] == e2["finalist"]["trial_id"]
    assert e1["ensemble"]["L_bar"] == e2["ensemble"]["L_bar"]
    assert e1["ensemble"]["L_members"] == e2["ensemble"]["L_members"]
    for k in (1, 2):
        a = sorted(glob.glob(os.path.join(H.best_dir(c1, q, alt), "member_seed%d" % k, "*_perrow.npz")))
        b = sorted(glob.glob(os.path.join(H.best_dir(c2, q, alt), "member_seed%d" % k, "*_perrow.npz")))
        x, y = _arrays(a[0]), _arrays(b[0])
        for key in x:
            assert np.array_equal(x[key], y[key]), (k, key)
