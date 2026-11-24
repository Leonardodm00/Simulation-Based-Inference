"""End to end through the real entry point (step 6 of the tester's procedure):
`hpc/pipeline/pipeline.py run` on the stand-in davinci (hpc/pipeline/tests/
standin.py: the Sbi-extractor checkout's real bank tools, the fake PBS) with a
recorded configuration v11 whose `searches` list holds S-A1 and S-A5-only, each
capped at one core, under the standard replicate rule (within_group). Trains
(the real tuner and runner): run it alone.

Oracles (from the spec):
  E.1 PIPELINE DONE; each search's SEARCH_COMPLETE.json and each best fit's
      BEST_FIT_COMPLETE.json in its own folder (<tune>/<q>/<campaign>,
      <runs>/<q>/<ds>_best); no single-search folder; every number finite.
  E.2 the A5 search's trials carry `--rep-pairs within_group --rep-group-key
      rep_group` (block 31) and the real bank (block 29's globs); the A1
      search's no real bank.
  E.3 the ensemble's L_bar is -(1/n) sum_i [logsumexp_m(-l_{m,i}) - log M] of the
      members' per-row NLLs (sec. 2.3), recomputed here with scipy.
  E.4 two runs from scratch in the same folder with the same seeds give the same
      identities, trials, nlls, finalists and L_bar (sec. 4: no randomness in the
      pipeline itself). Two folders are not compared: the stand-in's banks carry
      the sha256 of S1's manifest and S3's freeze (latent_spec), whose contents
      name the folder, so their contract digests -- and the trial ids -- differ.
  E.5 block 30 through the entry point: `stop-search --search A5` once A5 has an
      ok trial ends that search `user_stop`, its best fit then fitted on the
      best ok trial; the A1 search runs to its rounds.
"""

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

import _spec_util as U

pytest.importorskip("torch")
pytest.importorskip("sbi")
pytest.importorskip("skopt")

sys.path.insert(0, U.PTESTS)


def _project(root, searches, max_cores=1):
    import test_smoke_runs as TR             # the stand-in builder of block 29's suite
    path, cfg, env = TR._standin_project(root)
    cfg["searches"] = [dict(q, max_cores=max_cores) for q in searches]
    cfg["replicate"] = {"pairs": "within_group", "group_key": "rep_group"}
    # the posterior draws pinned at their lower bound (4 d_theta = 92): an A5 runner at
    # 324 draws on these 18,000-sample windows takes about 6 GB (measured here, under
    # either pair rule), past this sandbox's memory with two searches at once
    cfg["search"]["space_overrides"]["n_posterior_draws"] = [92, 92]
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return path, cfg, env


SEARCHES = [{"name": "A1", "campaign": "S-A1"}, {"name": "A5", "campaign": "S-A5-only"}]


def _run(path, env, root, timeout=3000):
    r = subprocess.run([U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "run", "--config", path,
                        "--poll-seconds", "0.5"], cwd=root, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return r.returncode, r.stdout


def _ledger(folder):
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "trials", "*.json"))):
        rec = json.load(open(p))
        out[rec["trial_id"]] = rec
    return out


def _summary(cfg):
    p = cfg["paths"]
    out = {}
    for q in SEARCHES:
        folder = os.path.join(p["tune_root"], q["name"], q["campaign"])
        sc = json.load(open(os.path.join(folder, "SEARCH_COMPLETE.json")))
        bf = json.load(open(os.path.join(p["runs_root"], q["name"], "synth_best",
                                         "BEST_FIT_COMPLETE.json")))
        led = _ledger(folder)
        out[q["name"]] = {"ended_by": sc["ended_by"], "finalist": sc["finalist"]["trial_id"],
                          "nll": dict((k, v.get("nll")) for k, v in led.items()),
                          "status": dict((k, v.get("status")) for k, v in led.items()),
                          "L_bar": bf["ensemble"]["L_bar"], "bf": bf, "folder": folder}
    return out


def _l_bar_from_members(bf_root):
    from scipy.special import logsumexp
    files = sorted(glob.glob(os.path.join(bf_root, "member_seed*", "*_perrow.npz")))
    assert files, bf_root
    ls = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            ls.append(np.asarray(z["nll"], dtype=np.float64))
    L = np.stack(ls)                                   # (M, n), rows in the same order
    M = L.shape[0]
    return float(-np.mean(logsumexp(-L, axis=0) - math.log(M))), M, L.shape[1]


@pytest.fixture(scope="module")
def first_run():
    root = U.tmpdir("spec_e2e1_")
    path, cfg, env = _project(root, SEARCHES)
    rc, out = _run(path, env, root)
    yield root, path, cfg, env, rc, out
    shutil.rmtree(root, ignore_errors=True)


def test_e1_done_and_folders(first_run):
    root, path, cfg, env, rc, out = first_run
    assert rc == 0 and "PIPELINE DONE" in out, out[-5000:]
    p = cfg["paths"]
    s = _summary(cfg)
    for q in SEARCHES:
        r = s[q["name"]]
        assert r["ended_by"] == "rounds"
        assert r["status"] and all(v == "ok" for v in r["status"].values()), r["status"]
        assert all(math.isfinite(v) for v in r["nll"].values()), r["nll"]
        assert math.isfinite(r["L_bar"])
        assert r["bf"]["finalist"]["trial_id"] == r["finalist"]
    assert not os.path.exists(os.path.join(p["tune_root"], "S-A1"))
    assert not os.path.exists(os.path.join(p["runs_root"], "synth_best"))


def test_e2_argv_of_the_trials(first_run):
    root, path, cfg, env, rc, out = first_run
    p = cfg["paths"]
    real = os.path.join(p["banks_root"], "synth_real", "shard_*.npz")
    for q in SEARCHES:
        folder = os.path.join(p["tune_root"], q["name"], q["campaign"])
        for tid, rec in _ledger(folder).items():
            argv = rec["argv"]
            got = argv[argv.index("--real-shards") + 1] if "--real-shards" in argv else None
            assert got == (None if q["campaign"] == "S-A1" else real), (q, got)
            joined = " ".join(argv)
            assert "--rep-pairs within_group --rep-group-key rep_group" in joined, joined
    # the real bank holds the optional rep_group array (Sbi-extractor block 9)
    with np.load(sorted(glob.glob(real))[0]) as z:
        assert "rep_group" in z.files


def test_e3_l_bar_recomputed(first_run):
    root, path, cfg, env, rc, out = first_run
    s = _summary(cfg)
    for q in SEARCHES:
        bf_root = os.path.join(cfg["paths"]["runs_root"], q["name"], "synth_best")
        lb, M, n = _l_bar_from_members(bf_root)
        assert M == 2
        # 1e-9: float64 sums of the same per-row float64 NLLs in another order
        assert abs(lb - s[q["name"]]["L_bar"]) < 1e-9, (lb, s[q["name"]]["L_bar"])


def test_e5_stop_search_through_the_entry_point():
    root = U.tmpdir("spec_e2e3_")
    try:
        path, cfg, env = _project(root, SEARCHES)
        p = cfg["paths"]
        a5 = os.path.join(p["tune_root"], "A5", "S-A5-only")
        proc = subprocess.Popen([U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "run", "--config",
                                 path, "--poll-seconds", "0.5"], cwd=root, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        t0 = time.time()
        stop_out = None
        while time.time() - t0 < 2400 and proc.poll() is None:
            oks = [r for r in _ledger(a5).values() if r.get("status") == "ok"] \
                if os.path.isdir(os.path.join(a5, "trials")) else []
            if oks and not os.path.exists(os.path.join(a5, "SEARCH_COMPLETE.json")):
                r = subprocess.run([U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "stop-search",
                                    "--config", path, "--search", "A5"], cwd=root, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, timeout=120)
                stop_out = r.stdout
                assert r.returncode == 0, stop_out
                break
            time.sleep(0.5)
        out, _ = proc.communicate(timeout=3000)
        assert stop_out is not None, "no ok trial of A5 was seen before the run ended:\n" + out[-3000:]
        assert proc.returncode == 0 and "PIPELINE DONE" in out, out[-5000:]
        sc = json.load(open(os.path.join(a5, "SEARCH_COMPLETE.json")))
        led = _ledger(a5)
        oks = dict((k, v["nll"]) for k, v in led.items() if v.get("status") == "ok")
        assert sc["ended_by"] in ("user_stop", "rounds"), sc["ended_by"]
        if sc["ended_by"] == "user_stop":
            assert sc["stop_request"]["schema"] == "search_stop/1"
            assert sc["finalist"]["trial_id"] == min(oks, key=oks.get)
            n = sc["n_trials"]
            assert n.get("ok", 0) == len(oks) and "cancelled" in n
            assert "search_A5: stop requested" in out
        bf = json.load(open(os.path.join(p["runs_root"], "A5", "synth_best",
                                         "BEST_FIT_COMPLETE.json")))
        assert bf["finalist"]["trial_id"] == sc["finalist"]["trial_id"]
        sc1 = json.load(open(os.path.join(p["tune_root"], "A1", "S-A1", "SEARCH_COMPLETE.json")))
        assert sc1["ended_by"] == "rounds"
        print("A5 ended_by %s; n_trials %s; stop line: %s"
              % (sc["ended_by"], sc["n_trials"], stop_out.strip().splitlines()[-1]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_e4b_same_path_same_results():
    """E.4 at one path: the stand-in's banks carry the digests of S1-S3's
    artefacts, which name their folder, so two runs are compared in the same
    folder (the second after the first's tree is removed)."""
    root = U.tmpdir("spec_e2e4_")
    try:
        summaries = []
        for attempt in range(2):
            if attempt:
                shutil.rmtree(root)
                os.makedirs(root)
            path, cfg, env = _project(root, SEARCHES)
            rc, out = _run(path, env, root)
            assert rc == 0 and "PIPELINE DONE" in out, out[-4000:]
            s = _summary(cfg)
            for q in SEARCHES:
                sc = json.load(open(os.path.join(s[q["name"]]["folder"], "SEARCH_COMPLETE.json")))
                s[q["name"]]["identity"] = (sc["identity"]["split_hash"],
                                            sc["identity"]["contract_digest"])
                s[q["name"]].pop("bf")
                s[q["name"]].pop("folder")
            summaries.append(s)
        a, b = summaries
        for q in SEARCHES:
            n = q["name"]
            assert a[n]["identity"] == b[n]["identity"], (n, a[n]["identity"], b[n]["identity"])
            assert a[n] == b[n], (n, a[n], b[n])
        print("same path: %s" % dict((q["name"], (len(a[q["name"]]["nll"]), a[q["name"]]["L_bar"]))
                                     for q in SEARCHES))
    finally:
        shutil.rmtree(root, ignore_errors=True)
