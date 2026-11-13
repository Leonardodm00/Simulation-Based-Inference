"""Step 6 of the tester's procedure on the synthetic kind: two runs of the
reduced project from scratch, with the same configuration (the same seeds),
give identical results (banks: every array and every contract digest; the
search: the same trials, configurations and nll; the best fit: the same
L_bar). One thread per job (search.trial.ncpus = best_fit.ncpus = 1), the
setting under which block 6 promises bit-for-bit runs.
"""
import glob
import json
import os

import numpy as np
import pytest

import t1b_helpers as H

H.pipeline_path()
import config as C                     # noqa: E402
import pipeline as P                   # noqa: E402
import stages as S                     # noqa: E402
from stages import synthetic as SY     # noqa: E402


def _one_run(tag):
    root = H.tmpdir("t1b_det_%s_" % tag)
    spool = os.path.join(root, "spool")
    os.makedirs(spool)
    old = H.fake_env(spool)
    try:
        path, d = H.reduced_synth(root)            # one search: S-A1 on R0
        cfg = C.load_config(path, synthetic_ok=True)
        lines = []
        rc = P.run(cfg, graph=S.default_graph(cfg), poll_seconds=0.2, log=lines.append,
                   max_ticks=3000)
        assert rc == 0, "\n".join(map(str, lines))[-4000:]
    finally:
        H.restore_env(old)
    return root, d


@pytest.fixture(scope="module")
def two_runs():
    pytest.importorskip("torch")
    pytest.importorskip("skopt")
    a = _one_run("a")
    b = _one_run("b")
    yield a, b
    H.rmtree(a[0])
    H.rmtree(b[0])


def _banks(root, d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d["paths"]["banks_root"], "*", "shard_*.npz"))):
        arrays, side = H.load_shard(p)
        out[os.path.relpath(p, d["paths"]["banks_root"])] = (arrays, side["contract_digest"])
    return out


def test_step6_banks_identical(two_runs):
    (ra, da), (rb, db) = two_runs
    A, B = _banks(ra, da), _banks(rb, db)
    assert sorted(A) == sorted(B) and len(A) == 5          # 3 + 1 + 1 shards
    for k in A:
        assert A[k][1] == B[k][1], k
        for name in A[k][0]:
            np.testing.assert_array_equal(A[k][0][name], B[k][0][name], err_msg="%s %s" % (k, name))


def test_step6_search_and_best_fit_identical(two_runs):
    (ra, da), (rb, db) = two_runs

    def ledger(d):
        camp = os.path.join(d["paths"]["tune_root"], "S-A1", "S-A1")
        recs = {}
        for p in glob.glob(os.path.join(camp, "trials", "*.json")):
            r = json.load(open(p))
            recs[r["trial_id"]] = (r["config"], r["nll"], r["L_pseudo_real"])
        return recs, json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))

    la, sa = ledger(da)
    lb, sb = ledger(db)
    assert sorted(la) == sorted(lb)
    for t in la:
        assert la[t][0] == lb[t][0], t
        assert la[t][1] == lb[t][1] and la[t][2] == lb[t][2], (t, la[t][1:], lb[t][1:])
    assert sa["finalist"]["trial_id"] == sb["finalist"]["trial_id"]

    def lbar(d):
        f = os.path.join(d["paths"]["runs_root"], "S-A1", d["dataset"] + "_best",
                         "BEST_FIT_COMPLETE.json")
        return json.load(open(f))

    ba, bb = lbar(da), lbar(db)

    def numbers(obj, prefix=""):
        """Every number of a JSON tree, by path (times and paths are not numbers here)."""
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "utc" in k or "time" in k or k.endswith("_s"):
                    continue
                out.update(numbers(v, prefix + "/" + k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                out.update(numbers(v, prefix + "/%d" % i))
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out[prefix] = obj
        return out

    na, nb = numbers(ba["ensemble"]), numbers(bb["ensemble"])
    assert na and sorted(na) == sorted(nb)
    for k in na:
        assert na[k] == nb[k], (k, na[k], nb[k])
