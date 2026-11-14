"""Chain t2-C2, step 6: an upstream artefact missing or wrong, and where the
chain stops (C-O12), on small copies of the reduced project (arm S of 3
shards; one search each), through the real entry point:

- P_foreign: R3's folder holds a shard of another seed before the run:
  real_bank_R3 must STOP the run (block 16 "refused, and nothing in it is
  overwritten"), no search started.
- P_ident: the search folder holds a pipeline_rounds.json of another
  identity: search_S-A1 must STOP with REFUSE, nothing submitted for it
  (block 7, "Rule of the identity").
- P_A1 (S-A1 alone), after a complete run:
  (d1) a sim shard lost after its bank's record, then a fresh state and the
       search folders moved aside: the search must not silently run on a
       bank that is not the bank of record (sec. 2.1, "never a silent reuse
       of a half-written artefact"; t1-B's F205 at the bank stage);
  (d2) SEARCH_COMPLETE.json gone under a done search, best_fit retried: the
       chain stops at best_fit (the best fit reads the search's artefact);
  (d3) the finalist of SEARCH_COMPLETE.json changed (a search run again
       with another finalist) and BEST_FIT_COMPLETE.json removed, best_fit
       retried: the members must be the finalist's (C-O8);
  (d4) the same with BEST_FIT_COMPLETE.json kept: sec. 2.2 leaves it done.
- P_mix (the mixed S-A2 on R3), after a complete run with alt_best_fit
  false: (e1) alt_best_fit set true is refused without
  --accept-config-change and runs best_fit_alt alone with it, the other
  artefacts byte-identical (block 18); (e2) a search finalist that is the
  best trial of neither arm makes best_fit_alt STOP (block 18).
"""

import glob
import hashlib
import json
import os
import shutil

import pytest

import c2_helpers as H

SMALL = {"synthetic.sim": {"seed": 11, "n_theta": 30, "n_windows": 1, "traces_per_shard": 20},
         "loader.buffer_shards": 3}


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _tree_sha(root):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = _sha(p)
    return out


def _root(tmp_path_factory, name):
    base = os.environ.get("C2_ROOT")
    root = os.path.join(os.path.abspath(base), name) if base else str(tmp_path_factory.mktemp(name))
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    return root


def _builder_argv(cfg, bank_out, seed):
    """R3's shard argv by hand (block 16) with another --seed."""
    return ["python3", os.path.join(H.REPO, "hpc", "joint", "stage1", "build_latent_bank.py"),
            "--out-dir", bank_out, "--arm", "R", "--shard-index", "0", "--n-traces", "12",
            "--donors-per-class", "2", "--n-windows", "2", "--pi", "0.3",
            "--gap-modes", "range_shift,drift,contamination", "--seed", str(seed),
            "--provider", "bench", "--fs", "10.0", "--T-win", "12.8", "--n-neurons", "100",
            "--wells-per-donor", "2", "--donors-per-batch", "8", "--n-classes", "3",
            "--tau-ov", "0.1", "--class-centres", "regular", "--centre-scale", "0.562743",
            "--label-axes", "0,1,2,3,4,5,6,7"]


def _set_config(cfg_path, mutate):
    with open(cfg_path) as fh:
        d = json.load(fh)
    mutate(d)
    with open(cfg_path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    return H._strip(d)


# ------------------------------------------------------------ P_foreign ---

def test_foreign_bank_stops_the_chain(tmp_path_factory):
    root = _root(tmp_path_factory, "p_foreign")
    q = {"name": "S-A5-R3", "campaign": "S-A5-only", "real": "R3", "alt_best_fit": False}
    cfg_path, cfg = H.reduced(os.path.join(root, "project"), [q], **SMALL)
    spool = os.path.join(root, "spool")
    env = H.fake_env(spool)
    bank = H.bank_dir(cfg, "R3")
    rc, out = H.run(_builder_argv(cfg, bank, 34), env,
                    cwd=os.path.join(H.REPO, "hpc", "joint", "stage1"))
    assert rc == 0, out
    before = _tree_sha(bank)
    rc, out = H.entry(cfg_path, env)
    assert rc == 0, out
    ce = H.wait_chain(cfg, spool, timeout=900)
    log = H.log_text(cfg)
    assert ce is not None and "failed (exit 3)" in ce, ce
    stop = [l for l in log.splitlines() if "STOP:" in l]
    assert stop and "real_bank_R3" in stop[0] and "another build" in stop[0], stop
    assert "--seed 34" in log or "34" in stop[0]
    assert _tree_sha(bank) == before, "the foreign folder was changed"
    names = [j["name"] for j in H.jobs(spool)]
    assert not [n for n in names if n.startswith(("gate_", "t_", "bf_", "sb_R3", "sr_R3"))], names


# -------------------------------------------------------------- P_ident ---

def test_search_folder_of_another_identity_stops_the_chain(tmp_path_factory):
    root = _root(tmp_path_factory, "p_ident")
    q = {"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False}
    cfg_path, cfg = H.reduced(os.path.join(root, "project"), [q], **SMALL)
    spool = os.path.join(root, "spool")
    env = H.fake_env(spool)
    camp = H.campaign_dir(cfg, q)
    os.makedirs(camp)
    rf = os.path.join(camp, "pipeline_rounds.json")
    with open(rf, "w") as fh:
        json.dump({"schema": "pipeline_rounds/1", "identity": {"split_hash": "0" * 64},
                   "rounds": []}, fh)
    before = _sha(rf)
    rc, out = H.entry(cfg_path, env)
    assert rc == 0, out
    ce = H.wait_chain(cfg, spool, timeout=900)
    log = H.log_text(cfg)
    assert ce is not None and "failed (exit 3)" in ce, ce
    stop = [l for l in log.splitlines() if "STOP:" in l]
    assert stop and "search_S-A1" in stop[0] and "REFUSE" in stop[0], stop
    assert _sha(rf) == before
    assert not [j for j in H.jobs(spool) if j["out"].startswith(camp)], "a job of the refused search"


# ----------------------------------------------------------------- P_A1 ---

@pytest.fixture(scope="module")
def p_a1(tmp_path_factory):
    root = _root(tmp_path_factory, "p_a1")
    q = {"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False}
    cfg_path, cfg = H.reduced(os.path.join(root, "project"), [q], **SMALL)
    spool = os.path.join(root, "spool")
    env = H.fake_env(spool)
    rc, out = H.entry(cfg_path, env)
    assert rc == 0, out
    ce = H.wait_chain(cfg, spool, timeout=1800)
    assert ce and "done (exit 0)" in ce, ce
    return {"root": root, "cfg_path": cfg_path, "cfg": cfg, "spool": spool, "env": env, "q": q}


def _rerun(p, *flags, timeout=1800):
    rc, out = H.entry(p["cfg_path"], p["env"], *flags)
    if rc != 0:
        return rc, out, None
    ce = H.wait_chain(p["cfg"], p["spool"], timeout=timeout)
    return rc, out, ce


def test_d1_lost_shard_and_a_fresh_state(p_a1):
    """(d1): the record of 3 shards kept, shard 2 lost, a fresh state (the
    state file removed, as a new work root) and the search and best fit
    moved aside so they run again. Per sec. 2.1 a bank's artefact of record
    hands the next stage the bank it records; the search must not train on
    another bank without a word."""
    cfg, q = p_a1["cfg"], p_a1["q"]
    bank = H.bank_dir(cfg, "sim")
    for ext in ("npz", "json"):
        os.remove(os.path.join(bank, "shard_0002." + ext))
    wr = H.work_root(cfg)
    for f in ("state.json", "state.json.prev"):
        if os.path.exists(os.path.join(wr, f)):
            os.remove(os.path.join(wr, f))
    os.rename(os.path.join(cfg["paths"]["tune_root"], q["name"]),
              os.path.join(cfg["paths"]["tune_root"], q["name"] + ".aside_d1"))
    os.rename(os.path.join(cfg["paths"]["runs_root"], q["name"]),
              os.path.join(cfg["paths"]["runs_root"], q["name"] + ".aside_d1"))
    rc, out, ce = _rerun(p_a1)
    assert rc == 0, out
    rec = H.read_json(os.path.join(bank, "BANK_RECORD.json"))
    split = os.path.join(H.campaign_dir(cfg, q), "bank_split.json")
    used = H.read_json(split) if os.path.isfile(split) else None
    print("CHAIN_END:", ce, "| record:", rec["n_shards"], "shards", rec["n_rows"], "rows | the "
          "search's identity:", used and (used["n_shards"], used["n_rows"]))
    # the oracle: the chain stops, or the search's bank is the bank of record
    stopped = ce is not None and "failed" in ce
    assert stopped or (used is not None and used["n_shards"] == rec["n_shards"]
                       and used["n_rows"] == rec["n_rows"]), \
        "the search ran on %s shard(s) / %s rows, the bank's record holds %d / %d" % (
            used and used["n_shards"], used and used["n_rows"], rec["n_shards"], rec["n_rows"])


def test_d2_search_artefact_gone_under_a_done_search(p_a1):
    cfg, q = p_a1["cfg"], p_a1["q"]
    sc = os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json")
    bfc = os.path.join(H.best_dir(cfg, q), "BEST_FIT_COMPLETE.json")
    if not (os.path.isfile(sc) and os.path.isfile(bfc)):
        pytest.skip("needs the complete run of (d1)")
    shutil.copy(sc, sc + ".keep")
    shutil.copy(bfc, bfc + ".keep")
    os.remove(sc)
    os.remove(bfc)
    n_before = len(H.jobs(p_a1["spool"]))
    rc, out, ce = _rerun(p_a1, "--retry", "best_fit_" + q["name"])
    log = H.log_text(cfg)
    os.replace(sc + ".keep", sc)
    os.replace(bfc + ".keep", bfc)
    assert rc == 0, out
    assert ce is not None and "failed (exit 3)" in ce, ce
    stop = [l for l in log.splitlines() if "STOP:" in l]
    assert stop and ("best_fit_" + q["name"]) in stop[-1], stop[-1:]
    new = [j["name"] for j in H.jobs(p_a1["spool"])[n_before:]]
    assert not [n for n in new if n.startswith("bf_")], new


def _second_best(cfg, q):
    led = H.ledger(cfg, q)
    ranked = sorted((rec["nll"], tid) for tid, rec in led.items() if rec["status"] == "ok")
    return ranked[0][1], ranked[1][1], led


def test_d3_members_of_another_finalist_are_not_reused(p_a1):
    cfg, q = p_a1["cfg"], p_a1["q"]
    sc_path = os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json")
    bfc = os.path.join(H.best_dir(cfg, q), "BEST_FIT_COMPLETE.json")
    first, second, led = _second_best(cfg, q)
    sc = H.read_json(sc_path)
    assert sc["finalist"]["trial_id"] == first
    keep_sc, keep_bf = sc_path + ".keep3", bfc + ".keep3"
    shutil.copy(sc_path, keep_sc)
    shutil.copy(bfc, keep_bf)
    sc["finalist"] = {"trial_id": second, "nll": led[second]["nll"], "config": led[second]["config"]}
    with open(sc_path, "w") as fh:
        json.dump(sc, fh, indent=1, sort_keys=True)
    os.remove(bfc)
    n_before = len(H.jobs(p_a1["spool"]))
    rc, out, ce = _rerun(p_a1, "--retry", "best_fit_" + q["name"])
    try:
        assert rc == 0, out
        new = [j["name"] for j in H.jobs(p_a1["spool"])[n_before:]]
        got = H.read_json(bfc) if os.path.isfile(bfc) else None
        print("CHAIN_END:", ce, "| jobs submitted:", new, "| BEST_FIT_COMPLETE finalist:",
              got and got["finalist"]["trial_id"][:12])
        # the oracle (C-O8): the ensemble's members are the finalist trained
        # M times -- each member's configuration the finalist's
        assert got is not None
        assert got["finalist"]["trial_id"] == second
        for m in got["members"]:
            run = H.read_json(m["record"])
            for key, val in led[second]["config"].items():
                if key in run.get("config", {}):
                    assert run["config"][key] == val, (
                        "member seed %d: %s = %r, the finalist's %r (members of trial %s reused "
                        "for trial %s; %d member job(s) submitted)"
                        % (m["seed"], key, run["config"][key], val, first[:12], second[:12],
                           len([n for n in new if n.startswith("bf_")])))
    finally:
        os.replace(keep_sc, sc_path)
        os.replace(keep_bf, bfc)


def test_d4_best_fit_of_another_finalist_stays_done(p_a1):
    """sec. 2.2: a pending stage whose done(s) holds is skipped, whatever its
    dependency became; BEST_FIT_COMPLETE.json's finalist is then not the
    search's. Recorded as the spec's behaviour (S-item in the report)."""
    cfg, q = p_a1["cfg"], p_a1["q"]
    sc_path = os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json")
    bfc = os.path.join(H.best_dir(cfg, q), "BEST_FIT_COMPLETE.json")
    first, second, led = _second_best(cfg, q)
    keep = sc_path + ".keep4"
    shutil.copy(sc_path, keep)
    sc = H.read_json(sc_path)
    sc["finalist"] = {"trial_id": second, "nll": led[second]["nll"], "config": led[second]["config"]}
    with open(sc_path, "w") as fh:
        json.dump(sc, fh, indent=1, sort_keys=True)
    n_before = len(H.jobs(p_a1["spool"]))
    try:
        rc, out, ce = _rerun(p_a1, "--retry", "best_fit_" + q["name"])
        assert rc == 0 and ce and "done (exit 0)" in ce, (out, ce)
        st = H.state(cfg)["stages"]["best_fit_" + q["name"]]
        assert st["status"] == "skipped"
        assert len(H.jobs(p_a1["spool"])) - n_before <= 2       # drivers only
        assert H.read_json(bfc)["finalist"]["trial_id"] == first
    finally:
        os.replace(keep, sc_path)


# ---------------------------------------------------------------- P_mix ---

@pytest.fixture(scope="module")
def p_mix(tmp_path_factory):
    root = _root(tmp_path_factory, "p_mix")
    q = {"name": "S-A2m-R3", "campaign": "S-A2", "real": "R3", "alt_best_fit": False}
    cfg_path, cfg = H.reduced(os.path.join(root, "project"), [q], **SMALL)
    spool = os.path.join(root, "spool")
    env = H.fake_env(spool)
    rc, out = H.entry(cfg_path, env)
    assert rc == 0, out
    ce = H.wait_chain(cfg, spool, timeout=1800)
    assert ce and "done (exit 0)" in ce, ce
    return {"root": root, "cfg_path": cfg_path, "cfg": cfg, "spool": spool, "env": env, "q": q}


def test_e1_alt_best_fit_turned_on(p_mix):
    cfg_path, q = p_mix["cfg_path"], p_mix["q"]
    cfg = p_mix["cfg"]
    sc = os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json")
    bfc = os.path.join(H.best_dir(cfg, q), "BEST_FIT_COMPLETE.json")
    before = {"sc": _sha(sc), "bf": _sha(bfc), "camp": _tree_sha(H.campaign_dir(cfg, q))}

    def on(d):
        d["synthetic"]["searches"][0]["alt_best_fit"] = True
    p_mix["cfg"] = cfg = _set_config(cfg_path, on)
    rc, out = H.entry(cfg_path, p_mix["env"])
    assert rc == 2 and "REFUSE" in out and "configuration" in out, (rc, out[-800:])
    n_before = len(H.jobs(p_mix["spool"]))
    rc, out = H.entry(cfg_path, p_mix["env"], "--accept-config-change")
    assert rc == 0, out
    ce = H.wait_chain(cfg, p_mix["spool"], timeout=1800)
    assert ce and "done (exit 0)" in ce, ce
    new = [j["name"] for j in H.jobs(p_mix["spool"])[n_before:] if j["name"] != "pl_synth"]
    assert sorted(new) == ["ba_synth_s1", "ba_synth_s2"], new
    assert _sha(sc) == before["sc"] and _sha(bfc) == before["bf"]
    assert _tree_sha(H.campaign_dir(cfg, q)) == before["camp"]
    led = H.ledger(cfg, q)
    arm = dict((t, {(0, 0): "A1", (1, 0): "A2"}.get((int(r["config"].get("dsn_on", 0)),
                                                       int(r["config"].get("rep_on", 0)))))
               for t, r in led.items() if r["status"] == "ok")
    s = H.read_json(sc)
    own = arm[s["finalist"]["trial_id"]]
    other = ({"A1", "A2"} - {own}).pop()
    exp = min((led[t]["nll"], t) for t, a in arm.items() if a == other)[1]
    fc = H.read_json(os.path.join(H.best_dir(cfg, q, alt=True), "finalist_config.json"))
    assert fc["trial_id"] == exp and fc["arm"] == other


def test_e2_search_finalist_best_of_neither_arm(p_mix):
    cfg, q = p_mix["cfg"], p_mix["q"]
    alt = H.best_dir(cfg, q, alt=True)
    if not os.path.isfile(os.path.join(alt, "BEST_FIT_COMPLETE.json")):
        pytest.skip("needs (e1)")
    led = H.ledger(cfg, q)
    arm = dict((t, int(r["config"].get("dsn_on", 0))) for t, r in led.items() if r["status"] == "ok")
    best = {}
    for t, a in arm.items():
        if a not in best or led[t]["nll"] < led[best[a]]["nll"]:
            best[a] = t
    neither = [t for t in arm if t not in best.values()]
    if not neither:
        pytest.skip("every ok trial is its arm's best")
    sc_path = os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json")
    keep = sc_path + ".keep_e2"
    shutil.copy(sc_path, keep)
    aside = alt + ".aside_e2"
    os.rename(alt, aside)
    sc = H.read_json(sc_path)
    t = neither[0]
    sc["finalist"] = {"trial_id": t, "nll": led[t]["nll"], "config": led[t]["config"]}
    with open(sc_path, "w") as fh:
        json.dump(sc, fh, indent=1, sort_keys=True)
    try:
        n_before = len(H.jobs(p_mix["spool"]))
        rc, out = H.entry(p_mix["cfg_path"], p_mix["env"], "--retry", "best_fit_alt_" + q["name"])
        assert rc == 0, out
        ce = H.wait_chain(cfg, p_mix["spool"], timeout=900)
        log = H.log_text(cfg)
        assert ce and "failed (exit 3)" in ce, ce
        stop = [l for l in log.splitlines() if "STOP:" in l]
        assert stop and "best_fit_alt_" + q["name"] in stop[-1] and "exactly one" in stop[-1], stop[-1:]
        new = [j["name"] for j in H.jobs(p_mix["spool"])[n_before:]]
        assert not [n for n in new if n.startswith("ba_")], new
    finally:
        os.replace(keep, sc_path)
        if os.path.isdir(alt):
            shutil.rmtree(alt)
        os.rename(aside, alt)


# ---------------------------------------------------------- P_cfgchange ---

def test_f_identity_key_changed_while_the_search_runs(tmp_path_factory):
    """(f): block 7's Rule of the identity -- "a results folder whose file
    disagrees with the configuration is refused before anything is
    submitted". The search S-A1 is stopped while its round 0 runs; the
    configuration's split.fractions (an identity key) is changed; the
    submission without --accept-config-change is refused (block 1), and
    with it the chain goes on. Expected: a refusal before the next trial is
    submitted, or every trial of the search on the identity's split."""
    root = _root(tmp_path_factory, "p_cfg")
    q = {"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False}
    cfg_path, cfg = H.reduced(os.path.join(root, "project"), [q], **SMALL)
    spool = os.path.join(root, "spool")
    env = H.fake_env(spool)
    rc, out = H.entry(cfg_path, env)
    assert rc == 0, out
    ok = H.wait_until(lambda: any(j["name"].startswith("t_") and j["state"] in ("Q", "R")
                                  for j in H.jobs(spool)), timeout=900)
    assert ok
    rc, out = H.entry(cfg_path, env, "--stop", "--wait", "300")
    assert rc == 0, out

    def change(d):
        d["split"]["fractions"] = [0.8, 0.15, 0.05]
    cfg = _set_config(cfg_path, change)
    rc, out = H.entry(cfg_path, env)
    assert rc == 2 and "REFUSE" in out, (rc, out[-600:])
    rc, out = H.entry(cfg_path, env, "--accept-config-change")
    assert rc == 0, out
    ce = H.wait_chain(cfg, spool, timeout=1800)
    log = H.log_text(cfg)
    ident = H.read_json(os.path.join(H.campaign_dir(cfg, q), "pipeline_rounds.json"))["identity"]
    led = H.ledger(cfg, q)
    hashes = sorted(set(r.get("split_hash") for r in led.values() if r["status"] == "ok"))
    fracs = sorted(set(tuple(float(x) for x in
                             r["argv"][r["argv"].index("--split-fracs") + 1:
                                       r["argv"].index("--split-fracs") + 4])
                       for r in led.values()))
    print("CHAIN_END:", ce)
    print("identity split_hash %s fracs %s; the trials' split hashes %s; their --split-fracs %s"
          % (ident["split_hash"][:12], ident["split_fracs"], [h[:12] for h in hashes], fracs))
    refused = ce is not None and "failed" in ce and "REFUSE" in log
    assert refused or hashes == [ident["split_hash"]], (
        "no refusal, and the search's trials ran on %d split(s) %s while its identity is %s: "
        "SEARCH_COMPLETE ranks them together" % (len(hashes), [h[:12] for h in hashes],
                                                 ident["split_hash"][:12]))


# ------------------------------------------------------------ P_badcfg ---

def test_g_configuration_refused_before_anything_runs(tmp_path_factory):
    """C-O1 through the real entry point: a configuration sec. 3.1 refuses
    (arm S's seed shared with R0, D-120/D-121; and a search naming no real
    bank) -- exit 2 from run_pipeline.sh, every problem named, nothing
    submitted, no state written."""
    root = _root(tmp_path_factory, "p_badcfg")
    q = {"name": "S-A1", "campaign": "S-A1", "real": "R9", "alt_best_fit": False}
    cfg_path, cfg = H.reduced(os.path.join(root, "project"), [q], **SMALL)

    def bad(d):
        d["synthetic"]["sim"]["seed"] = d["synthetic"]["real"][0]["seed"]
    _set_config(cfg_path, bad)
    spool = os.path.join(root, "spool")
    env = H.fake_env(spool)
    for flags in ((), ("--dry-run",)):
        rc, out = H.entry(cfg_path, env, *flags)
        assert rc == 2, (flags, rc, out[-800:])
        assert "R9" in out, out[-800:]
    assert H.jobs(spool) == []
    assert not os.path.exists(os.path.join(H.work_root(cfg), "state.json"))
