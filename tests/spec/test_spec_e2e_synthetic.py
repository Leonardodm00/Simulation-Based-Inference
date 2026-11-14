"""End to end (step 6 of the tester's procedure): the real entry point
`hpc/pipeline/pipeline.py run` on a reduced synthetic project under the
fake PBS, twice with the same configuration in two fresh roots.

Oracles from specs/SPEC.md: the run ends with `PIPELINE DONE` (exit 0);
every bank's stored theta is float32 strictly inside (0, 1) (block 11's
amendment); SEARCH_COMPLETE.json names a finalist whose nll is the lowest
of the ledger's ok records (sec. 2.3); BEST_FIT_COMPLETE.json names that
finalist (block 7 (i)), M = len(member_seeds), a finite L_bar <= the mean
of the members' L (Jensen, sec. 2.3); two runs with the same seeds give
identical numbers (sec. 4: no randomness in the pipeline itself; the
seeds are the configuration's; one thread). After the run, `--retry
search_S-A1` with the finalist unchanged: the best fit is skipped and no
job is submitted (rule 5 + done predicate).

Needs the training environment. Run:
    python3 -m pytest -q tests/spec/test_spec_e2e_synthetic.py
"""

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
TESTS = os.path.join(PIPE, "tests")
for p in (PIPE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("torch")
pytest.importorskip("sbi")
pytest.importorskip("skopt")


def _run(root, *extra):
    import test_smoke_synthetic as T            # fixture builders only
    path = os.path.join(root, "synth.json")
    spool = os.path.join(root, "spool")
    env = T._fake_env(spool)
    r = subprocess.run([sys.executable, os.path.join(PIPE, "pipeline.py"), "run", "--config", path,
                        "--poll-seconds", "0.5"] + list(extra), cwd=root, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=3000)
    n_jobs = len([p for p in glob.glob(os.path.join(spool, "*.json"))
                  if os.path.basename(p)[:-5].isdigit()])
    return r.returncode, r.stdout, n_jobs


def _project(root):
    import test_smoke_synthetic as T
    path, d = T._small(root, searches=[{"name": "S-A1", "campaign": "S-A1", "real": "R0",
                                        "alt_best_fit": False}])
    return d


def _outputs(d):
    p = d["paths"]
    camp = os.path.join(p["tune_root"], "S-A1", "S-A1")
    sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
    recs = [json.load(open(t)) for t in sorted(glob.glob(os.path.join(camp, "trials", "*.json")))]
    bf = json.load(open(os.path.join(p["runs_root"], "S-A1", d["dataset"] + "_best",
                                     "BEST_FIT_COMPLETE.json")))
    thetas = {}
    for bank in sorted(glob.glob(os.path.join(p["banks_root"], "*"))):
        for sh in sorted(glob.glob(os.path.join(bank, "shard_*.npz"))):
            with np.load(sh) as z:
                thetas[os.path.relpath(sh, p["banks_root"])] = np.array(z["theta"])
    return sc, recs, bf, thetas


@pytest.fixture(scope="module")
def two_runs():
    out = []
    roots = []
    try:
        for _ in range(2):
            root = tempfile.mkdtemp(prefix="spec_e2e_")
            roots.append(root)
            d = _project(root)
            rc, text, n = _run(root)
            out.append((root, d, rc, text, n))
        yield out
    finally:
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)


def test_runs_finish(two_runs):
    for root, d, rc, text, n in two_runs:
        assert rc == 0 and "PIPELINE DONE" in text, text[-4000:]


def test_outputs_contract(two_runs):
    root, d, rc, text, n = two_runs[0]
    sc, recs, bf, thetas = _outputs(d)
    assert thetas, "no bank shard"
    for name, th in thetas.items():
        assert th.dtype == np.float32 and th.ndim == 2 and th.shape[1] == 10, name
        assert np.all(np.isfinite(th)) and np.all(th > 0) and np.all(th < 1), name
    ok = [r for r in recs if r.get("status") == "ok"]
    assert ok
    best = min(ok, key=lambda r: r["nll"])
    assert sc["finalist"]["trial_id"] == best["trial_id"]
    assert math.isfinite(sc["finalist"]["nll"]) and sc["finalist"]["nll"] == best["nll"]
    assert bf["finalist"]["trial_id"] == sc["finalist"]["trial_id"]
    ens = bf["ensemble"]
    assert ens["M"] == len(d["best_fit"]["member_seeds"])
    assert math.isfinite(ens["L_bar"])
    Ls = [m["L"] for m in bf["members"]] if isinstance(bf.get("members"), list) else None
    if Ls:
        # Jensen: the mixture's NLL is at most the members' mean (sec. 2.3); 1e-9 for roundoff
        assert ens["L_bar"] <= sum(Ls) / len(Ls) + 1e-9, (ens, Ls)


def test_determinism(two_runs):
    (r1, d1, _, _, n1), (r2, d2, _, _, n2) = two_runs
    sc1, recs1, bf1, th1 = _outputs(d1)
    sc2, recs2, bf2, th2 = _outputs(d2)
    assert sorted(th1) == sorted(th2)
    for k in th1:
        assert th1[k].tobytes() == th2[k].tobytes(), k
    assert sc1["finalist"]["trial_id"] == sc2["finalist"]["trial_id"]
    assert sc1["finalist"]["nll"] == sc2["finalist"]["nll"]
    assert [(r["trial_id"], r.get("nll")) for r in recs1] == [(r["trial_id"], r.get("nll")) for r in recs2]
    assert bf1["ensemble"]["L_bar"] == bf2["ensemble"]["L_bar"]
    assert n1 == n2


def test_retry_search_same_finalist_skips_best_fit(two_runs):
    root, d, _, _, n0 = two_runs[0]
    best = os.path.join(d["paths"]["runs_root"], "S-A1", d["dataset"] + "_best")
    before = dict((os.path.relpath(os.path.join(dp, f), best), open(os.path.join(dp, f), "rb").read())
                  for dp, _, fs in os.walk(best) for f in fs)
    rc, text, n1 = _run(root, "--retry", "search_S-A1")
    assert rc == 0 and "PIPELINE DONE" in text, text[-4000:]
    assert "best_fit_S-A1 returned to pending" in text, text[-4000:]
    assert n1 == n0, "jobs submitted: %d" % (n1 - n0)
    after = dict((os.path.relpath(os.path.join(dp, f), best), open(os.path.join(dp, f), "rb").read())
                 for dp, _, fs in os.walk(best) for f in fs)
    assert after == before
    assert not glob.glob(best + ".set_aside*")
