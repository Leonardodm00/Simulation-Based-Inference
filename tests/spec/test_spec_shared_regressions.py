"""The blocks whose modules changed through shared files (scope: 1, 2, 5, 11,
15-18): regression against the base commit bd7651f and the v11 additions.

Oracles (from the spec, before the code was read):
  O1.r  cores_in_flight(state) and the Submitter without a core group behave as
        at the base commit (block 1 (d)); `names=None` is the whole project.
  O2.v11 the bank profile carries `cohort_map` (real.cohort_map, null allowed)
        and `drop_partial_topologies` (bank.drop_partial_topologies), and loads
        through the Sbi-extractor checkout's bank_plan_config (block 2's known
        answer).
  O15.r the synthetic kind's graph, stage names, deps and run views are
        unchanged (block 29: "its graph, names and behaviour unchanged").
  O5.v11 the shard contract's optional `rep_group` (n,) int array: accepted when
        present, checked like the others; one concatenation holds the same
        optional arrays in every shard; a bank without it reads as before.
"""

import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import _spec_util as U

import config as C            # noqa: E402
import pipeline as P          # noqa: E402
import stages as S            # noqa: E402
import stages.bank as SB      # noqa: E402
import state as ST            # noqa: E402

BASE = "/tmp/base_tree/hpc"
needs_base = pytest.mark.skipif(not os.path.isdir(BASE), reason="the base tree is not extracted "
                                "(git archive bd7651f hpc | tar -x -C /tmp/base_tree)")


@pytest.fixture
def root():
    r = U.tmpdir("spec_shared_")
    yield r
    shutil.rmtree(r, ignore_errors=True)


# ---------------------------------------------------------------- O1.r ---

def _random_states(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        stages = {}
        jid = 0
        for name in ("search", "best_fit", "sim_bank", "real_bank", "bank_plan", "x_extra"):
            status = rng.choice(["pending", "running", "running", "done", "skipped", "failed"])
            jobs = []
            for _k in range(rng.randint(0, 5)):
                jid += 1
                j = {"id": "%d.srv" % jid, "role": "r", "finished": rng.random() < 0.3,
                     "cmd": ["qsub", "-l", "select=%d:ncpus=%d" % (rng.randint(1, 2),
                                                                   rng.choice([1, 4, 48]))]}
                if rng.random() < 0.6:
                    j["ncpus"] = rng.choice([1, 8, 16, 48])
                if jobs and rng.random() < 0.3:
                    j["after"] = rng.choice(jobs)["id"]
                jobs.append(j)
            stages[name] = {"status": status, "jobs": jobs}
        out.append({"schema": "pipeline_state/1", "stages": stages})
    return out


BASE_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import pipeline as P
import state as ST
states = json.load(open(sys.argv[2]))
out = []
for d in states:
    st = ST.State("/nonexistent", d)
    out.append(list(P.cores_in_flight(st)))
json.dump(out, open(sys.argv[3], "w"))
"""


@needs_base
def test_cores_in_flight_equals_the_base(root):
    states = _random_states(300)
    sp, op, scr = (os.path.join(root, n) for n in ("states.json", "out.json", "base.py"))
    json.dump(states, open(sp, "w"))
    open(scr, "w").write(BASE_SCRIPT)
    r = subprocess.run([U.PYTHON, scr, os.path.join(BASE, "pipeline"), sp, op], cwd=root,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert r.returncode == 0, r.stdout
    base = json.load(open(op))
    for d, b in zip(states, base):
        st = ST.State("/nonexistent", d)
        head = P.cores_in_flight(st)
        assert [head[0], head[1], list(head[2])] == b
        assert P.cores_in_flight(st, names=None) == head
        # the names filter: the sum over a partition is the whole when no job is
        # `after` a job of another stage (every `after` here is within a stage)
        parts = [P.cores_in_flight(st, names=(n,))[0] for n in d["stages"]]
        assert sum(parts) == head[0]


# ---------------------------------------------------------------- O2.v11 ---

def _profile(root, mutate):
    path, d = U.linddup_copy(root, mutate)
    cfg = C.load_config(path)
    for kind in ("sim_redetection", "label_freeze", "real_extraction"):
        p = S.artefact_of(kind, cfg)[1]
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write('{"x": 1}\n')
    return cfg, d, SB.BankPlanStage().write_profile(cfg)


def test_profile_carries_the_v11_keys(root):
    cfg, d, prof = _profile(root, None)
    assert prof["cohort_map"] == d["real"]["cohort_map"]
    assert prof["drop_partial_topologies"] is True
    written = json.load(open(os.path.join(d["paths"]["work_root"], "bank_profile_linddup.json")))
    assert written["cohort_map"] == d["real"]["cohort_map"]
    assert written["drop_partial_topologies"] is True

    def nomap(dd):
        dd["real"]["cohort_map"] = None
        dd["bank"]["drop_partial_topologies"] = False
    cfg2, d2, prof2 = _profile(os.path.join(root, "b"), nomap)
    assert prof2["cohort_map"] is None and prof2["drop_partial_topologies"] is False


@pytest.mark.skipif(not os.path.isfile(os.path.join(U.SBIX_CLONE, "export_windows",
                                                    "bank_plan_config.py")),
                    reason="no Sbi-extractor checkout")
def test_profile_loads_through_bank_plan_config(root):
    for sub, mutate in (("a", None),
                        ("b", lambda dd: (dd["real"].__setitem__("cohort_map", None),
                                          dd["bank"].__setitem__("drop_partial_topologies",
                                                                 False)))):
        cfg, d, prof = _profile(os.path.join(root, sub), mutate)
        ppath = os.path.join(d["paths"]["work_root"], "bank_profile_linddup.json")
        code = ("import sys; sys.path.insert(0, %r); sys.dont_write_bytecode = True\n"
                "import bank_plan_config as B\n"
                "c = B.load_profile(%r, repo_root=%r)\n"
                "print(repr(c.cohort_map), repr(c.drop_partial_topologies))\n"
                % (os.path.join(U.SBIX_CLONE, "export_windows"), ppath, U.SBIX_CLONE))
        r = subprocess.run([U.PYTHON, "-c", code], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        assert r.returncode == 0, r.stdout
        assert r.stdout.strip() == "%r %r" % (prof["cohort_map"], prof["drop_partial_topologies"])


# ---------------------------------------------------------------- O15.r ---

GRAPH_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import config as C
import stages as S
import stages.common as CM
cfg = C.load_config(sys.argv[2], synthetic_ok=True)
g = S.default_graph(cfg)
out = []
for s in g:
    rec = {"name": s.name, "deps": list(s.deps), "cls": type(s).__name__}
    v = None
    if hasattr(s, "_view"):
        v = s._view(cfg)
        rec["tune_root"] = v.data["paths"]["tune_root"]
        rec["runs_root"] = v.data["paths"]["runs_root"]
        rec["campaign"] = v.data["search"]["campaign"]
        rec["globs"] = list(CM.shard_globs(v))
        rec["artefact"] = S.artefact_of(getattr(s, "base_name", s.name), v)[1]
    out.append(rec)
json.dump(out, open(sys.argv[3], "w"), sort_keys=True)
"""


def _graph_of(tree, cfg_path, out):
    scr = out + ".py"
    open(scr, "w").write(GRAPH_SCRIPT)
    r = subprocess.run([U.PYTHON, scr, os.path.join(tree, "pipeline"), cfg_path, out],
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert r.returncode == 0, r.stdout
    return json.load(open(out))


@needs_base
@pytest.mark.parametrize("alt", [False, True])
def test_synthetic_graph_equals_the_base(root, alt):
    def v3like(d):
        if alt:                            # block 18's alternative fits on the mixed campaigns
            for q in d["synthetic"]["searches"]:
                if q["campaign"] in ("S-A2-only", "S-A5-only"):
                    q["campaign"] = q["campaign"][:-5]
                    q["alt_best_fit"] = True
    head_path, d = U.synth_copy(root, v3like, "synth_head.json")
    base_d = json.loads(json.dumps(d))
    del base_d["replicate"]
    base_d["config_version"] = 10
    base_path = U.write_json(os.path.join(root, "synth_base.json"), base_d)
    head = _graph_of(os.path.join(U.REPO, "hpc"), head_path, os.path.join(root, "head.json"))
    base = _graph_of(BASE, base_path, os.path.join(root, "base.json"))
    assert len(head) == (17 if alt else 13)
    assert head == base


@needs_base
def test_recorded_null_graph_equals_the_base(root):
    head_path, d = U.giulia_copy(root, name="g_head.json")
    base_d = json.loads(json.dumps(d))
    for k in ("searches", "replicate"):
        del base_d[k]
    del base_d["real"]["cohort_map"]
    del base_d["bank"]["drop_partial_topologies"]
    base_d["config_version"] = 10
    base_path = U.write_json(os.path.join(root, "g_base.json"), base_d)
    head = _graph_of(os.path.join(U.REPO, "hpc"), head_path, os.path.join(root, "head.json"))
    base = _graph_of(BASE, base_path, os.path.join(root, "base.json"))
    assert [g["name"] for g in head] == [g["name"] for g in base]
    assert [g["deps"] for g in head] == [g["deps"] for g in base]


# ---------------------------------------------------------------- O5.v11 ---

def _lb():
    sys.path.insert(0, os.path.join(U.JOINT, "stage1"))
    import latent_bank as LB
    return LB


def _arrays(n, W=8, rep_group=None):
    a = {"x": np.random.default_rng(0).random((n, W)).astype(np.float32),
         "theta": np.full((n, 3), 0.5, dtype=np.float32),
         "cls": np.zeros(n, dtype=np.int64), "nu": np.zeros((n, 2), dtype=np.float32),
         "realisation_id": np.arange(n, dtype=np.uint64),
         "donor": np.arange(n, dtype=np.int64), "well": np.arange(n, dtype=np.int64),
         "batch": np.zeros(n, dtype=np.int64), "subregion": np.zeros(n, dtype=np.int64),
         "window_idx": np.zeros(n, dtype=np.int64), "contaminated": np.zeros(n, dtype=bool)}
    if rep_group is not None:
        a["rep_group"] = rep_group
    return a


def test_optional_rep_group_contract():
    LB = _lb()
    assert "rep_group" in LB.OPTIONAL_ROW_ARRAYS
    assert LB.check_arrays(_arrays(4)) == 4
    assert LB.check_arrays(_arrays(4, rep_group=np.array([0, 0, 1, 1], dtype=np.int64))) == 4
    for bad in (np.array([0.0, 0, 1, 1]),               # not an int array
                np.zeros((4, 1), dtype=np.int64),         # not 1-D
                np.array([0, 0, 1], dtype=np.int64)):     # another row count
        with pytest.raises(LB.ContractError):
            LB.check_arrays(_arrays(4, rep_group=bad))
    other = _arrays(4)
    other["rep_groups"] = np.zeros(4, dtype=np.int64)     # an unknown array is still refused
    with pytest.raises(LB.ContractError):
        LB.check_arrays(other)


def test_concat_with_and_without_rep_group(root):
    LB = _lb()
    side = LB.build_sidecar(["a", "b", "c"], [[0, 1]] * 3, "unit", 10.0, 0.1, 0.8, 8,
                            "per_unit_mean", {"kind": "box"}, {}, {}, {}, "g" * 64, False, "R")
    p0, p1 = os.path.join(root, "shard_0000.npz"), os.path.join(root, "shard_0001.npz")
    LB.write_shard(p0, _arrays(3, rep_group=np.array([5, 5, 6], dtype=np.int64)), side)
    LB.write_shard(p1, _arrays(2, rep_group=np.array([6, 7], dtype=np.int64)), side)
    merged, _s = LB.concat_shards([p0, p1])
    assert_array_equal(merged["rep_group"], [5, 5, 6, 6, 7])
    p2 = os.path.join(root, "shard_0002.npz")
    LB.write_shard(p2, _arrays(2), side)
    with pytest.raises(LB.ContractError):
        LB.concat_shards([p0, p2])                     # one concatenation, one set of arrays
    merged2, _s = LB.concat_shards([p2])
    assert "rep_group" not in merged2
