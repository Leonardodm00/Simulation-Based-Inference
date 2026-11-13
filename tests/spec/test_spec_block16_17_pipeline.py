"""Spec-derived end-to-end tests of blocks 15-17 under the fake PBS (SPEC.md
secs. 2.1, 3.3, blocks 15-17) and their boundaries with blocks 1, 7 and 19
(oracles O15.4, O15.7, O16.5, O16.6, O16.8, O17.3 and B1, B7, B19 of the t1-B
report).

The sequencer runs in this process on a reduced copy of the committed
project (helpers.reduced_synth); its jobs run under the fake PBS, the bank
jobs with the real builder and record tool, the trials and members with the
real tuner and runner. Expected outcomes are block 16's rules and block 17's
T17 contract, checked from the files and the fake PBS's job records.
"""
import glob
import json
import os
import re
import shutil

import numpy as np
import pytest

import t1b_helpers as H

H.pipeline_path()
import config as C                     # noqa: E402
import pipeline as P                   # noqa: E402
import stages as S                     # noqa: E402
from stages import common as CM        # noqa: E402
from stages import synthetic as SY     # noqa: E402


def _run(cfg_path, banks_only=False, retry=(), max_ticks=3000):
    cfg = C.load_config(cfg_path, synthetic_ok=True)
    graph = S.default_graph(cfg)
    if banks_only:
        graph = [s for s in graph if isinstance(s, SY.SynthBankStage)]
    lines = []
    rc = P.run(cfg, graph=graph, poll_seconds=0.2, retry=tuple(retry), log=lines.append,
               max_ticks=max_ticks)
    return rc, "\n".join(str(l) for l in lines)


def _cores(rec):
    for r in rec["resources"]:
        m = re.search(r"ncpus=(\d+)", r)
        if m:
            return int(m.group(1))
    return 0


def _max_cores_at_once(recs):
    """The largest sum of cores over jobs queued or running at one moment
    (each job counted from its submission to its end)."""
    ev = []
    for r in recs:
        if r.get("finished") is None:
            continue
        ev.append((r["submitted"], _cores(r)))
        ev.append((r["finished"], -_cores(r)))
    cur = best = 0
    for _t, c in sorted(ev, key=lambda e: (e[0], e[1])):
        cur += c
        best = max(best, cur)
    return best


def _fresh_state(d):
    for f in ("state.json", "state.json.prev"):
        p = os.path.join(d["paths"]["work_root"], f)
        if os.path.exists(p):
            os.remove(p)


@pytest.fixture(scope="module")
def banks_run():
    root = H.tmpdir("t1b_e2e_banks_")
    spool = os.path.join(root, "spool")
    os.makedirs(spool)
    old = H.fake_env(spool)
    path, d = H.reduced_synth(root)
    try:
        yield {"root": root, "spool": spool, "path": path, "d": d}
    finally:
        H.restore_env(old)
        H.rmtree(root)


def _bank(d, name):
    return os.path.join(d["paths"]["banks_root"], d["dataset"] + ("_sim" if name == "sim"
                                                                  else "_real_" + name))


def test_o16_5_banks_built_through_their_jobs(banks_run):
    d, spool = banks_run["d"], banks_run["spool"]
    rc, out = _run(banks_run["path"], banks_only=True)
    assert rc == 0, out[-4000:]
    recs = H.spool_records(spool)
    names = sorted(r["name"] for r in recs)
    evidence = "\n".join(["%s %s exit %s %s" % (r["num"], r["name"], r["exit_status"], r.get("comment"))
                          for r in recs] + [l for l in out.splitlines() if "shard" in l])
    assert names == sorted(["sb_sim_0", "sb_sim_1", "sb_sim_2", "sr_sim", "sb_R0_0", "sr_R0",
                            "sb_R3_0", "sr_R3"]), evidence
    wr = d["paths"]["work_root"]
    for r in recs:
        stage = "sim_bank" if "_sim" in r["name"] else "real_bank_" + r["name"].split("_")[1]
        assert r["queue"] == "test"
        assert _cores(r) == 1 and any("mem=1gb" in x for x in r["resources"])
        assert any("walltime=00:10:00" in x for x in r["resources"])
        v = r["vars"]
        assert v["ENV_NAME"] == d["envs"]["train"]
        assert v["WORKDIR"] == (H.STAGE1 if r["name"].startswith("sb_") else H.PIPE)
        assert v["AVOID_FILE"] == os.path.join(wr, "avoid_nodes.txt")
        assert os.path.dirname(r["out"]) == os.path.join(wr, "bank_jobs", stage)
        assert os.path.dirname(v["ARGV_FILE"]) == os.path.join(wr, "bank_jobs", stage)
        assert r["exit_status"] == 0
    # D-099: never more than resources.max_cores - driver.ncpus cores at once
    assert _max_cores_at_once(recs) <= d["resources"]["max_cores"] - d["driver"]["ncpus"]
    seeds = {"sim": d["synthetic"]["sim"]["seed"], "R0": 202, "R3": 303}
    for name, rows, kind in (("sim", 12 * 2 * 1, "box"), ("R0", 3 * 2 * 2 * 2, None),
                             ("R3", 3 * 2 * 2 * 2, None)):
        rec = json.load(open(os.path.join(_bank(d, name), "BANK_RECORD.json")))
        assert rec["n_rows"] == rows and rec["latent_kind"] == kind, name
        assert rec["build"] == SY.build_of(C.load_config(banks_run["path"], synthetic_ok=True),
                                           name)
        side = json.load(open(os.path.join(_bank(d, name), "shard_0000.json")))
        assert side["latent_spec"]["seed"] == seeds[name]
        assert side["gap_spec"]["pi"] == (0.3 if name == "R3" else 0.0)


def test_o16_8_rerun_on_fresh_state_submits_nothing(banks_run):
    d, spool = banks_run["d"], banks_run["spool"]
    n0 = len(H.spool_records(spool))
    _fresh_state(d)
    rc, out = _run(banks_run["path"], banks_only=True)
    assert rc == 0, out[-3000:]
    assert len(H.spool_records(spool)) == n0
    assert out.count("skipped") >= 3, out[-3000:]


def test_o16_3_only_the_missing_shard_and_the_record(banks_run):
    d, spool = banks_run["d"], banks_run["spool"]
    sim = _bank(d, "sim")
    keep = {k: H.sha256_file(os.path.join(sim, "shard_%04d.npz" % k)) for k in (0, 1)}
    for f in ("shard_0002.npz", "shard_0002.json", "BANK_RECORD.json", "BANK_RECORD.json.sha256"):
        os.remove(os.path.join(sim, f))
    n0 = len(H.spool_records(spool))
    _fresh_state(d)
    rc, out = _run(banks_run["path"], banks_only=True)
    assert rc == 0, out[-3000:]
    new = [r["name"] for r in H.spool_records(spool)[n0:]]
    assert sorted(new) == ["sb_sim_2", "sr_sim"], new
    for k in (0, 1):
        assert H.sha256_file(os.path.join(sim, "shard_%04d.npz" % k)) == keep[k]


def test_o16_6_failing_shard_stops_then_retry(banks_run):
    d, spool = banks_run["d"], banks_run["spool"]
    r0 = _bank(d, "R0")
    for f in os.listdir(r0):
        os.remove(os.path.join(r0, f))
    os.makedirs(os.path.join(r0, "shard_0000.npz"))         # the builder cannot write it
    n0 = len(H.spool_records(spool))
    _fresh_state(d)
    rc, out = _run(banks_run["path"], banks_only=True)
    assert rc == 3, out[-3000:]
    new = [r["name"] for r in H.spool_records(spool)[n0:]]
    assert new == ["sb_R0_0", "sb_R0_0"], new                # max_resubmits 1: two jobs, then stop
    stop = [l for l in out.splitlines() if "STOP" in l]
    assert stop and "real_bank_R0" in stop[-1], out[-3000:]
    assert "--retry real_bank_R0" in stop[-1]
    assert "IsADirectoryError" in stop[-1] or "Is a directory" in stop[-1], stop[-1]
    os.rmdir(os.path.join(r0, "shard_0000.npz"))
    n1 = len(H.spool_records(spool))
    rc, out = _run(banks_run["path"], banks_only=True, retry=("real_bank_R0",))
    assert rc == 0, out[-3000:]
    assert [r["name"] for r in H.spool_records(spool)[n1:]] == ["sb_R0_0", "sr_R0"]


def test_o16_6_start_failure_not_counted(banks_run):
    d, spool = banks_run["d"], banks_run["spool"]
    r3 = _bank(d, "R3")
    for f in os.listdir(r3):
        os.remove(os.path.join(r3, f))
    k = 1 + sum(1 for r in H.spool_records(spool) if r["name"] == "sb_R3_0")
    os.environ["FAKE_PBS_FAIL_START"] = "sb_R3_0:%d" % k     # Exit_status -2, no log
    try:
        n0 = len(H.spool_records(spool))
        _fresh_state(d)
        rc, out = _run(banks_run["path"], banks_only=True)
    finally:
        os.environ.pop("FAKE_PBS_FAIL_START", None)
    assert rc == 0, out[-3000:]
    new = H.spool_records(spool)[n0:]
    assert [r["name"] for r in new] == ["sb_R3_0", "sb_R3_0", "sr_R3"]
    assert new[0]["exit_status"] == -2
    st = json.load(open(os.path.join(d["paths"]["work_root"], "state.json")))
    assert st["stages"]["real_bank_R3"]["sub"]["shards"]["0"]["resubmits"] == 0


def test_o16_4_foreign_bank_refused_unchanged(banks_run):
    d, spool = banks_run["d"], banks_run["spool"]
    r3 = _bank(d, "R3")
    for f in os.listdir(r3):
        os.remove(os.path.join(r3, f))
    argv = SY.build_of(C.load_config(banks_run["path"], synthetic_ok=True), "R3")["argv"][0]
    argv = list(argv)
    argv[argv.index("--seed") + 1] = "999"
    rc, txt = H.run_builder(r3, argv)
    assert rc == 0, txt
    before = {f: H.sha256_file(os.path.join(r3, f)) for f in os.listdir(r3)}
    n0 = len(H.spool_records(spool))
    _fresh_state(d)
    rc, out = _run(banks_run["path"], banks_only=True)
    assert rc == 3, out[-3000:]
    assert "holds a bank of another build" in out and "--seed 999" in out
    assert len(H.spool_records(spool)) == n0
    assert {f: H.sha256_file(os.path.join(r3, f)) for f in os.listdir(r3)} == before


# ------------------------------------------------------ T17: end to end ---

def _arm(cfg_json):
    dsn, rep = int(cfg_json.get("dsn_on", 0)), int(cfg_json.get("rep_on", 0))
    if dsn and rep:
        return None
    return "A2" if dsn else ("A5" if rep else "A1")


@pytest.fixture(scope="module")
def project_run():
    pytest.importorskip("torch")
    pytest.importorskip("sbi")
    pytest.importorskip("zuko")
    pytest.importorskip("skopt")
    root = H.tmpdir("t1b_e2e_t17_")
    spool = os.path.join(root, "spool")
    os.makedirs(spool)
    old = H.fake_env(spool)
    searches = [{"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False},
                {"name": "S-A2-R0", "campaign": "S-A2-only", "real": "R0", "alt_best_fit": False}]
    path, d = H.reduced_synth(root, searches=searches)
    try:
        rc, out = _run(path)
        yield {"root": root, "spool": spool, "path": path, "d": d, "rc": rc, "out": out}
    finally:
        H.restore_env(old)
        H.rmtree(root)


def test_o17_3_pipeline_done(project_run):
    assert project_run["rc"] == 0, project_run["out"][-6000:]
    assert "PIPELINE DONE" in project_run["out"]


def test_o17_3_artefacts_and_trials(project_run):
    d = project_run["d"]
    cfg = C.load_config(project_run["path"], synthetic_ok=True)
    for q in d["synthetic"]["searches"]:
        v = SY.RunView(cfg, q)
        camp = os.path.join(v.data["paths"]["tune_root"], q["campaign"])
        assert os.path.isfile(os.path.join(camp, "SEARCH_COMPLETE.json"))
        best = os.path.join(v.data["paths"]["runs_root"], d["dataset"] + "_best")
        assert os.path.isfile(os.path.join(best, "BEST_FIT_COMPLETE.json"))
        trials = [json.load(open(p)) for p in glob.glob(os.path.join(camp, "trials", "*.json"))]
        assert len(trials) == d["search"]["n_initial_points"] + d["search"]["gp_rounds"] * \
            d["search"]["trials_per_round"]
        for t in trials:
            assert t["status"] == "ok", t.get("error")
            assert "L_pseudo_real" in t and np.isfinite(t["L_pseudo_real"]), sorted(t)
            arm = _arm(t["config"])
            assert arm == ("A1" if q["campaign"] == "S-A1" else "A2"), (q["name"], t["config"])
        members = sorted(glob.glob(os.path.join(best, "member_seed*", "*_seed*.json")))
        members = [m for m in members if not m.endswith("_perrow.json")]
        assert len(members) == 2, members
        arms = set(json.load(open(m))["arm"] for m in members)
        assert arms == {"A1" if q["campaign"] == "S-A1" else "A2"}
        for m in members:
            assert json.load(open(m))["split_seed"] == d["best_fit"]["split_seed"]


def test_b7_trials_get_the_views_banks(project_run):
    """D-122 and block 15: every trial (S-A1 included) gets the simulated bank
    and its search's real bank, the view's shard_globs."""
    d = project_run["d"]
    cfg = C.load_config(project_run["path"], synthetic_ok=True)
    for q in d["synthetic"]["searches"]:
        v = SY.RunView(cfg, q)
        sim, real = CM.shard_globs(v)
        camp = os.path.join(v.data["paths"]["tune_root"], q["campaign"])
        argvs = glob.glob(os.path.join(camp, "jobs", "*.argv"))
        trial_argvs = [a for a in argvs if not os.path.basename(a).startswith("gate")]
        assert trial_argvs, camp
        for a in trial_argvs:
            toks = open(a).read().split("\n")
            assert toks[toks.index("--sim-shards") + 1] == sim, a
            assert toks[toks.index("--real-shards") + 1] == real, a


def test_b7_identity_digest_is_the_banks(project_run):
    d = project_run["d"]
    cfg = C.load_config(project_run["path"], synthetic_ok=True)
    rec = json.load(open(os.path.join(_bank(d, "sim"), "BANK_RECORD.json")))
    for q in d["synthetic"]["searches"]:
        v = SY.RunView(cfg, q)
        ident = json.load(open(os.path.join(v.data["paths"]["tune_root"], q["campaign"],
                                            "bank_split.json")))
        assert ident.get("contract_digest") == rec["contract_digest"], sorted(ident)


def test_o15_7_log_lines_renamed(project_run):
    log = project_run["out"]          # the sequencer's log lines (run(..., log=...))
    bad = [l for l in log.splitlines() if re.match(r"^(search|best_fit): ", l)]
    assert not bad, bad[:5]
    assert re.search(r"^search_S-A1: ", log, re.M)
    assert re.search(r"^best_fit_S-A2-R0: ", log, re.M)


def test_b1_state_names_are_the_graphs(project_run):
    cfg = C.load_config(project_run["path"], synthetic_ok=True)
    st = json.load(open(os.path.join(project_run["d"]["paths"]["work_root"], "state.json")))
    assert sorted(st["stages"]) == sorted(s.name for s in S.default_graph(cfg))
    assert all(v["status"] in ("done", "skipped") for v in st["stages"].values())


def test_d099_cap_end_to_end(project_run):
    d = project_run["d"]
    recs = H.spool_records(project_run["spool"])
    assert _max_cores_at_once(recs) <= d["resources"]["max_cores"] - d["driver"]["ncpus"]


def test_b19_evaluation_reads_the_banks(project_run):
    """Block 19's readers on what blocks 15-16 wrote: the simulated glob is
    the views'; a cohort reads every row of its bank, theta withheld; the
    ensembles of the two searches are found with their arms and cohorts."""
    pytest.importorskip("torch")
    H.add_paths(os.path.join(H.PIPE, "evaluation"))
    import ev_plan
    import ev_data
    cfg = C.load_config(project_run["path"], synthetic_ok=True)
    d = project_run["d"]
    q0 = d["synthetic"]["searches"][0]
    assert ev_plan.sim_bank_glob(cfg) == CM.shard_globs(SY.RunView(cfg, q0))[0]
    for r in d["synthetic"]["real"]:
        c = ev_data.cohort(cfg, r["name"])
        assert c["x"].shape == (24, 128) and c["theta"].shape == (24, 10)
        a, side = H.load_shard(os.path.join(_bank(d, r["name"]), "shard_0000.npz"))
        np.testing.assert_array_equal(c["theta"], a["theta"].astype(np.float64))
        np.testing.assert_array_equal(c["donor"], a["donor"])
    ens = ev_plan.ensembles(cfg)
    got = sorted((e.name, e.cohorts) for e in ens)
    assert got == sorted([("S-A1:A1", ("R0", "R3")), ("S-A2-R0:A2", ("R0",))]), got
