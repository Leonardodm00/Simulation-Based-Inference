"""Block 18 (the other arm's best fit), derived from specs/SPEC.md sec. 2.1,
2.5 and block 18 (oracles O18.1-O18.6 of the t1-C report).

Pure ASCII, LF only.
"""

import copy
import glob
import json
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import HPC, PIPE, PY, REPO, load_json, run_py

SYNTH = os.path.join(PIPE, "projects", "synth", "synth.json")
TUNER = os.path.join(HPC, "joint", "stage4", "npe_tune_joint.py")


def _arm(cfg):
    """sec. 2.5: dsn_on = 1 gives A2, rep_on = 1 gives A5, both 0 give A1;
    both on: no arm (block 18)."""
    dsn, rep = int(cfg["dsn_on"]), int(cfg["rep_on"])
    if dsn and rep:
        return None
    return "A2" if dsn else ("A5" if rep else "A1")


def _committed():
    with open(SYNTH) as fh:
        return json.load(fh)


def _load(data):
    import config as C
    d = tempfile.mkdtemp(prefix="t1c_b18_cfg_")
    p = os.path.join(d, "synth.json")
    with open(p, "w") as fh:
        json.dump(data, fh, indent=1)
    try:
        return C.load_config(p, synthetic_ok=True), None
    except C.ConfigError as exc:
        return None, str(exc)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _with_alt(campaign_by_name, alt_by_name):
    d = _committed()
    for q in d["synthetic"]["searches"]:
        if q["name"] in campaign_by_name:
            q["campaign"] = campaign_by_name[q["name"]]
        if q["name"] in alt_by_name:
            v = alt_by_name[q["name"]]
            if v is KeyError:
                del q["alt_best_fit"]
            else:
                q["alt_best_fit"] = v
    return d


# ------------------------------------------------------------- O18.3 config

def test_alt_arms_table():
    import config as C
    assert C.ALT_ARMS == {"S-A2": ("A1", "A2"), "S-A5": ("A1", "A5")}


@pytest.mark.parametrize("campaign,value,ok", [
    ("S-A2", True, True), ("S-A5", True, True), ("S-A2", False, True),
    ("S-A1", True, False), ("S-A25", True, False),
    ("S-A2-only", True, False), ("S-A5-only", True, False),       # block 26
    ("S-A2", "true", False), ("S-A2", 1, False), ("S-A2", None, False),
    ("S-A2", KeyError, False),                                    # the key missing
])
def test_alt_best_fit_key(campaign, value, ok):
    data = _with_alt({"S-A2-R0": campaign}, {"S-A2-R0": value})
    cfg, err = _load(data)
    if ok:
        assert cfg is not None, err
    else:
        assert cfg is None and "alt_best_fit" in (err or ""), err


# -------------------------------------------------------------- O18.4 graph

def _graph(data):
    import stages
    cfg, err = _load(data)
    assert cfg is not None, err
    return cfg, stages.default_graph(cfg)


def test_committed_v4_has_no_alt_stage():
    cfg, g = _graph(_committed())
    names = [s.name for s in g]
    assert not [n for n in names if n.startswith("best_fit_alt_")], names
    assert len(names) == 13, names                 # block 26: three banks, five searches and fits


def test_graph_alt_stage_order_deps_artefact():
    import stages
    from stages.synthetic import RunView
    data = _with_alt({"S-A2-R0": "S-A2", "S-A5-R3": "S-A5"},
                     {"S-A2-R0": True, "S-A5-R3": True})
    cfg, g = _graph(data)
    names = [s.name for s in g]
    want = ["sim_bank", "real_bank_R0", "real_bank_R3"]
    for q in data["synthetic"]["searches"]:
        want += ["search_" + q["name"], "best_fit_" + q["name"]]
        if q["alt_best_fit"]:
            want.append("best_fit_alt_" + q["name"])
    assert names == want
    by = dict((s.name, s) for s in g)
    for q in ("S-A2-R0", "S-A5-R3"):
        assert tuple(by["best_fit_alt_" + q].deps) == ("search_" + q,)
        view = RunView(cfg, [x for x in data["synthetic"]["searches"] if x["name"] == q][0])
        kind, path = stages.artefact_of("best_fit_alt", view)
        assert path == os.path.join(data["paths"]["runs_root"], q, data["dataset"] + "_best_alt",
                                    "BEST_FIT_COMPLETE.json"), path
        assert by["best_fit_alt_" + q].root_suffix == "_best_alt"
        assert by["best_fit_alt_" + q].job_prefix == "ba"


def test_member_job_names_fit_pbs():
    """`ba_<dataset[:6]>_s<k>`, within PBS's 15 characters for any dataset name
    and member seeds up to 99."""
    for ds in ("synth", "a_very_long_dataset_name"):
        for k in (1, 5, 10, 99):
            name = "ba_%s_s%d" % (ds[:6], k)
            assert len(name) <= 15, name


def test_dry_run_lists_alt_members(tmp_path):
    """The dry run of a configuration with alt_best_fit true prints the alt
    members' job names ba_synth_s<k> and no REFUSE line (davinci paths
    absent here: every stage not done)."""
    data = _with_alt({"S-A2-R0": "S-A2"}, {"S-A2-R0": True})
    data["paths"]["work_root"] = str(tmp_path / "work")
    p = tmp_path / "synth.json"
    p.write_text(json.dumps(data, indent=1))
    env = dict(os.environ, PIPELINE_SKIP_CONDA="1")
    rc, out = run_py([os.path.join(PIPE, "pipeline.py"), "run", "--config", str(p), "--dry-run"],
                     env=env, timeout=600)
    assert rc == 0, out[-3000:]
    assert "REFUSE" not in out, out[-3000:]
    assert "best_fit_alt_S-A2-R0: would wait for search_S-A2-R0" in out, out[-3000:]
    # the members' qsub line, as a template (`<k>` the member seed), in the alt folder
    alt = [l for l in out.splitlines() if "qsub -N ba_synth_s<k>" in l]
    assert len(alt) == 1 and "/S-A2-R0/synth_best_alt/jobs/" in alt[0], alt
    main = [l for l in out.splitlines() if "qsub -N bf_synth_s<k>" in l]
    assert len(main) == len(data["synthetic"]["searches"]), len(main)
    assert not os.path.exists(str(tmp_path / "work" / "state.json"))


# ------------------------------------------------------- O18.1-2 the tuner

def _ledger(root, campaign, base_cfg, rows):
    """rows: (trial_id, dsn_on, rep_on, nll, status, tag)."""
    d = os.path.join(root, campaign, "trials")
    os.makedirs(d, exist_ok=True)
    for tid, dsn, rep, nll, status, tag in rows:
        c = copy.deepcopy(base_cfg)
        c["dsn_on"], c["rep_on"] = dsn, rep
        c["lr"] = float(c.get("lr", 1e-4)) * (1.0 + 0.01 * len(tid))     # distinct configs
        c["_t1c_id"] = tid
        rec = {"trial_id": tid, "config": c, "nll": nll, "status": status, "tag": tag}
        with open(os.path.join(d, "%s.json" % tid), "w") as fh:
            json.dump(rec, fh)


def _base_config(project):
    p = sorted(glob.glob(os.path.join(project["root"], "tune_root", "S-A2-R0", "S-A2", "trials",
                                      "*.json")))[0]
    return load_json(p)["config"]


ROWS = [("a", 0, 0, 3.0, "ok", ""), ("b", 0, 0, 1.5, "ok", ""), ("c", 1, 0, 2.0, "ok", ""),
        ("d", 1, 0, 0.7, "ok", ""), ("e", 1, 1, 0.1, "ok", ""),          # both on: no arm
        ("f", 0, 0, 0.2, "failed", ""), ("g", 0, 0, 0.3, "ok", "lc"),     # failed; tagged
        ("h", 0, 0, float("nan"), "ok", "")]                               # not finite


def _finalists(root, *extra):
    return run_py([TUNER, "finalists", "--campaign", "S-A2", "--results-dir", root,
                   "--split-hash", "h", "--contract-digest", "c"] + list(extra), timeout=300)


@pytest.fixture()
def ledger_root(project, tmp_path):
    root = str(tmp_path / "res")
    _ledger(root, "S-A2", _base_config(project), ROWS)
    return root


def _expect(arm):
    ok = [(nll, tid) for tid, dsn, rep, nll, st, tag in ROWS
          if st == "ok" and not tag and math.isfinite(nll)
          and (arm is None or _arm({"dsn_on": dsn, "rep_on": rep}) == arm)]
    return [tid for nll, tid in sorted(ok)]


@pytest.mark.parametrize("arm", ["A1", "A2"])
def test_finalists_by_arm(ledger_root, tmp_path, arm):
    out = str(tmp_path / ("fin_%s.json" % arm))
    rc, txt = _finalists(ledger_root, "--top-k", "3", "--arm", arm, "--out", out)
    assert rc == 0, txt[-2000:]
    fin = load_json(out)
    assert fin["arm"] == arm
    got = [f["config"]["_t1c_id"] for f in fin["finalists"]]
    assert got == _expect(arm)[:3], (got, _expect(arm))
    assert not os.path.exists(os.path.join(ledger_root, "S-A2", "finalists.json"))


def test_finalists_no_trial_of_arm(ledger_root, tmp_path):
    out = str(tmp_path / "fin_A5.json")
    rc, txt = _finalists(ledger_root, "--arm", "A5", "--out", out)
    assert rc != 0 and "no completed trials of arm A5 to rank" in txt, txt[-1500:]
    assert not os.path.exists(out)


def test_finalists_without_flags(ledger_root):
    rc, txt = _finalists(ledger_root, "--top-k", "2")
    assert rc == 0, txt[-1500:]
    fin = load_json(os.path.join(ledger_root, "S-A2", "finalists.json"))
    assert "arm" in fin and fin["arm"] is None
    got = [f["config"]["_t1c_id"] for f in fin["finalists"]]
    assert got == _expect(None)[:2] == ["e", "d"], got      # the both-on trial ranks first


def test_finalists_arm_shuffled_is_accepted(ledger_root, tmp_path):
    """SPEC block 18, Public API: `finalists ... [--arm A1|A2|A5|shuffled]`."""
    out = str(tmp_path / "fin_sh.json")
    rc, txt = _finalists(ledger_root, "--arm", "shuffled", "--out", out)
    assert "invalid choice" not in txt, txt[-800:]


# --------------------------------------------- O18.5 the fixture's alt fit

def _search_dirs(project, q, campaign):
    root = project["root"]
    return (os.path.join(root, "tune_root", q, campaign),
            os.path.join(root, "runs_root", q, "synth_best"),
            os.path.join(root, "runs_root", q, "synth_best_alt"))


def test_alt_finalist_is_other_arms_best(project):
    tdir, main, alt = _search_dirs(project, "S-A2-R0", "S-A2")
    recs = [load_json(p) for p in sorted(glob.glob(os.path.join(tdir, "trials", "*.json")))]
    best = {}
    for r in recs:
        if r.get("status") != "ok" or r.get("tag") or r.get("nll") is None \
                or not math.isfinite(float(r["nll"])):
            continue
        a = _arm(r["config"])
        if a is None:
            continue
        if a not in best or float(r["nll"]) < best[a][0]:
            best[a] = (float(r["nll"]), r["trial_id"])
    sc = load_json(os.path.join(tdir, "SEARCH_COMPLETE.json"))
    own = [a for a in best if best[a][1] == sc["finalist"]["trial_id"]]
    assert len(own) == 1, (best, sc["finalist"]["trial_id"])
    other = [a for a in ("A1", "A2") if a != own[0]][0]
    fc = load_json(os.path.join(alt, "finalist_config.json"))
    assert fc["arm"] == other
    assert fc["trial_id"] == best[other][1]
    assert fc["search_finalist"] == {"trial_id": sc["finalist"]["trial_id"], "arm": own[0]}
    assert sorted(fc["finalists_files"]) == ["A1", "A2"]
    for a, p in fc["finalists_files"].items():
        assert os.path.isfile(p) and load_json(p)["arm"] == a
    assert _arm(fc["config"]) == other


def test_alt_members_share_the_main_fit_settings(project):
    tdir, main, alt = _search_dirs(project, "S-A2-R0", "S-A2")
    bm, ba = (load_json(os.path.join(f, "BEST_FIT_COMPLETE.json")) for f in (main, alt))
    for k in ("member_seeds", "split_seed", "epochs", "passes", "patience_epochs",
              "n_posterior_draws", "split_fracs"):
        assert bm[k] == ba[k], k
    cfg = load_json(project["cfg"])
    assert ba["member_seeds"] == cfg["best_fit"]["member_seeds"]
    fc = load_json(os.path.join(alt, "finalist_config.json"))
    hashes = set()
    for s in ba["member_seeds"]:
        r = load_json(glob.glob(os.path.join(alt, "member_seed%d" % s, "*_seed%d.json" % s))[0])
        assert r["arm"] == fc["arm"] and int(r["seed"]) == int(s)
        assert int(r["split_seed"]) == int(cfg["best_fit"]["split_seed"])
        hashes.add(r["split_hash"])
        rm = load_json(glob.glob(os.path.join(main, "member_seed%d" % s, "*_seed%d.json" % s))[0])
        assert rm["split_hash"] == r["split_hash"]          # one split for every member, both arms
        assert rm["passes"] == r["passes"]
    assert len(hashes) == 1


def test_alt_ensemble_nll_formula(project):
    """sec. 2.3: L_bar = -(1/n) sum_i [logsumexp_m(-l_mi) - log M], from the
    members' per-row files, against BEST_FIT_COMPLETE.json."""
    from scipy.special import logsumexp
    for folder in _search_dirs(project, "S-A2-R0", "S-A2")[1:]:
        b = load_json(os.path.join(folder, "BEST_FIT_COMPLETE.json"))
        L = []
        for s in b["member_seeds"]:
            pr = np.load(glob.glob(os.path.join(folder, "member_seed%d" % s,
                                                "*_seed%d_perrow.npz" % s))[0])
            L.append(np.asarray(pr["nll"], dtype=np.float64))
        L = np.stack(L)
        M, n = L.shape
        want = -np.mean(logsumexp(-L, axis=0) - math.log(M))
        assert b["ensemble"]["M"] == M and b["ensemble"]["n_report"] == n
        assert abs(b["ensemble"]["L_bar"] - want) <= 1e-9, (b["ensemble"]["L_bar"], want)


def test_alt_member_jobs_named_ba(project):
    names = []
    for p in glob.glob(os.path.join(project["root"], "spool", "*.json")):
        try:
            names.append(load_json(p).get("name"))
        except ValueError:
            continue
    alt = set(n for n in names if n and n.startswith("ba_"))
    cfg = load_json(project["cfg"])
    assert alt == set("ba_%s_s%d" % (cfg["dataset"][:6], s)
                      for s in cfg["best_fit"]["member_seeds"]), alt
