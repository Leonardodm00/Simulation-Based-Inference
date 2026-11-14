"""Chain C1 (the real-data project), end to end -- integration tester, audit
D-131, run t2-C1. Oracles: the report's "Oracles derived from the spec"
(C1-G*, C1-H*, C1-R*, C1-F*, C1-Q*, C1-X*, C1-T*), from SPEC.md secs. 2.1-2.3,
3, 4 and blocks 1-10.

The chain runs through its real entry point, `hpc/pipeline/run_pipeline.sh
--config <file>` (the D-098 driver jobs) under the fake PBS, on the stand-in
of `chain_c1_harness.py`: a recorded data set whose eight stages S1-S8 all
run (S1-S2 through the repository's launcher stubs, S3-S6 Sbi-extractor's
real tools, S7-S8 SBI's real tuner and runner).

Run A is interrupted twice (`--stop --wait`) at stage boundaries and
resumed; run B is the same configuration in another folder, run straight
through: the same seeds must give the same numbers (step 6). The last tests
reuse run A's banks for two short searches and then purge run A's outputs
and rebuild them.

    SBIX_CLONE=<Sbi-extractor checkout> python3 -m pytest -q tests/spec/test_spec_chain_C1_e2e.py

About 25-35 minutes on 2 CPUs (torch, sbi, zuko, scikit-optimize needed).
C1_KEEP=1 keeps the stand-in folders.

Pure ASCII, LF only.
"""

import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import chain_c1_harness as H      # noqa: E402

pytest.importorskip("torch")
pytest.importorskip("sbi")
pytest.importorskip("skopt")
if H.ST.sbix_clone() is None:
    pytest.skip("no Sbi-extractor checkout (set SBIX_CLONE)", allow_module_level=True)

STAGES = ("real_extraction", "sim_redetection", "label_freeze", "bank_plan", "sim_bank",
          "real_bank", "search", "best_fit")
DEPS = {"real_extraction": (), "sim_redetection": ("real_extraction",), "label_freeze": (),
        "bank_plan": ("real_extraction", "sim_redetection", "label_freeze"),
        "sim_bank": ("bank_plan",), "real_bank": ("bank_plan",),
        "search": ("sim_bank", "real_bank"), "best_fit": ("search",)}
OBS = {}


def _utc_epoch(text):
    return time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone


def _wait_status(s, stage, wanted, timeout):
    return s.wait_for(lambda: s.statuses().get(stage) in wanted, timeout, every=1.0)


def _stop(s, wait=300):
    rc, out = s.sh(["--stop", "--wait", str(wait)], timeout=wait + 60)
    return rc, out


def _ledger(s, campaign="S-A1", tune_root=None):
    d = os.path.join(tune_root or s.cfg["paths"]["tune_root"], campaign, "trials")
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(p) as fh:
            out[os.path.basename(p)[:-5]] = json.load(fh)
    return out


def _bank_arrays(bank_dir):
    """{shard name: {member: array}} of the small arrays and x of every shard."""
    out = {}
    for p in sorted(glob.glob(os.path.join(bank_dir, "shard_*.npz"))):
        with np.load(p, allow_pickle=False) as z:
            out[os.path.basename(p)] = dict((k, np.asarray(z[k])) for k in z.files)
    return out


# ---------------------------------------------------------------------------
# Run A: the chain end to end, stopped and resumed twice
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def A():
    s = H.ChainStandin()
    obs = {"root": s.root}
    # C1-T1: the inputs check before the first submission (read-only)
    watched = [s.cfg["sim"]["sim_main"], s.sbix, s.ds["extract_root"], s.ds["out_root"]]
    before = dict((w, H.tree_snapshot(w)) for w in watched)
    obs["inputs_check"] = s.tool("tools/inputs_check.py", ["--config", s.path], timeout=600)
    after = dict((w, H.tree_snapshot(w)) for w in watched)
    obs["inputs_check_changed"] = dict(
        (w, sorted(set(before[w].items()) ^ set(after[w].items()))) for w in watched)
    obs["dry_run"] = s.sh(["--dry-run"], timeout=300)
    obs["n_jobs_after_dry_run"] = len(s.jobs())
    obs["submit1"] = s.sh([], timeout=300)
    # stop 1: once S4 (the bank plan) is done -- a stage boundary
    obs["reached_s4"] = bool(_wait_status(s, "bank_plan", ("done",), 900))
    obs["stop1"] = _stop(s)
    obs["statuses_at_stop1"] = s.statuses()
    obs["jobs_at_stop1"] = len(s.jobs())
    obs["submit2"] = s.sh([], timeout=300)
    # stop 2: while round 0's trials run (inside S7)
    def trials_running():
        st = s.state()
        if not st:
            return False
        sub = st["stages"]["search"].get("sub") or {}
        return any(t.get("status") == "running" for t in (sub.get("trials") or {}).values())
    obs["reached_trials"] = bool(s.wait_for(trials_running, 1500, every=1.0))
    obs["stop2"] = _stop(s)
    obs["statuses_at_stop2"] = s.statuses()
    # C1-T2 / C1-T3 while the trials' jobs still run and no sequencer ticks
    live = [j for j in s.live_jobs() if not j["name"].startswith("pl_")]
    obs["live_stage_jobs_at_stop2"] = [j["name"] for j in live]
    camp = os.path.join(s.cfg["paths"]["tune_root"], "S-A1")
    snap = H.tree_snapshot(camp)
    rc, lst = s.tool("tools/purge_outputs.py", ["--config", s.path, "--what", "search"])
    m = re.search(r"CODE ([0-9a-f]{8})", lst)
    obs["purge_listing_mid"] = (rc, lst)
    obs["purge_confirm_mid"] = s.tool("tools/purge_outputs.py",
                                      ["--config", s.path, "--what", "search",
                                       "--confirm", m.group(1) if m else "00000000"])
    obs["purge_mid_campaign_present"] = os.path.isdir(camp)
    obs["report_mid"] = s.tool("tools/search_report.py", ["--config", s.path])
    snap2 = H.tree_snapshot(camp)
    obs["camp_changed_by_tools_mid"] = [k for k in set(snap) | set(snap2)
                                        if k.startswith("F ") and k[2:].startswith(
                                            ("pipeline_rounds", "SEARCH_COMPLETE", "trials/",
                                             "pending/", "gate/"))
                                        and snap.get(k) != snap2.get(k)]
    obs["submit3"] = s.sh([], timeout=300)
    obs["chain_end"] = s.wait_chain_end(2400)
    obs["statuses_end"] = s.statuses()
    obs["log"] = s.log_text()
    obs["jobs"] = s.jobs()
    obs["state"] = s.state()
    obs["status_cmd"] = s.sh(["--status"], timeout=300)
    # a submission on the finished state does nothing but its driver
    n0 = len(s.jobs())
    obs["submit4"] = s.sh([], timeout=300)
    s.wait_for(lambda: s.chain_end() and not s.live_jobs()
               and len(s.jobs()) > n0, 600, every=1.0)
    s.wait_for(lambda: not s.live_jobs(), 300)
    obs["jobs_after_submit4"] = s.jobs()[n0:]
    obs["log_after_submit4"] = s.log_text()
    OBS["A"] = obs
    yield s, obs
    if os.environ.get("C1_KEEP") != "1":
        s.close()


def test_entry_point_dry_run_submits_nothing(A):
    s, obs = A
    rc, out = obs["dry_run"]
    assert rc == 0, out[-3000:]
    assert "REFUSE" not in out, out[-3000:]
    assert obs["n_jobs_after_dry_run"] == 0


def test_entry_point_submit_and_stop_lines(A):
    s, obs = A
    for k in ("submit1", "submit2", "submit3"):
        rc, out = obs[k]
        assert rc == 0 and "SUBMITTED the sequencer: job" in out, (k, out[-2000:])
    for k in ("stop1", "stop2"):
        rc, out = obs[k]
        assert rc == 0 and "stop: the sequencer has stopped" in out, (k, out[-2000:])
    assert obs["reached_s4"] and obs["reached_trials"]


def test_chain_ends_done_with_every_stage_run(A):
    """C1-G1, C1-R4: the eight stages, each run (none pre-made), PIPELINE DONE."""
    s, obs = A
    assert obs["chain_end"] is not None and " done (exit 0)" in obs["chain_end"], obs["chain_end"]
    assert "PIPELINE DONE: every stage done or skipped" in obs["log"]
    assert obs["statuses_end"] == dict((n, "done") for n in STAGES), obs["statuses_end"]
    assert list(obs["state"]["stages"].keys()) == sorted(STAGES) or \
        set(obs["state"]["stages"]) == set(STAGES)


def test_stage_order_respects_dependencies(A):
    """C1-G2: no job of a stage is submitted before each of its dependencies
    is done (the state's finished_utc against the fake PBS's submission
    times, at the state's one-second resolution)."""
    s, obs = A
    st = obs["state"]["stages"]
    by_id = dict((j["id"], j) for j in obs["jobs"])
    bad = []
    for name in STAGES:
        sub_times = [by_id[j["id"]]["submitted"] for j in st[name]["jobs"] if j["id"] in by_id]
        if not sub_times:
            continue
        first = min(sub_times)
        for d in DEPS[name]:
            fin = st[d].get("finished_utc")
            assert fin, (d, st[d])
            if first < _utc_epoch(fin) - 1.0:
                bad.append((name, d, first, fin))
    assert not bad, bad


def test_no_stage_job_submitted_twice(A):
    """C1-R2: across two stops and three submissions, every job the fake PBS
    ran belongs to a stage's record or to the sequencer, and no command (argv
    file, or a launcher job's output path) was submitted twice."""
    s, obs = A
    recorded = set()
    for name, st in obs["state"]["stages"].items():
        for j in st["jobs"]:
            recorded.add(j["id"])
    seen, dup, orphan = {}, [], []
    for j in obs["jobs"]:
        if j["name"].startswith("pl_"):
            continue
        if j["id"] not in recorded:
            orphan.append((j["id"], j["name"]))
        key = (j["vars"] or {}).get("ARGV_FILE") or j.get("out") or j["script"]
        if key in seen:
            dup.append((key, seen[key], j["id"]))
        seen[key] = j["id"]
    assert not orphan, orphan
    assert not dup, dup


def test_one_driver_at_a_time(A):
    """C1-R2 / D-098: the sequencer's jobs never run two at once."""
    s, obs = A
    drv = [j for j in obs["jobs"] if j["name"].startswith("pl_") and j.get("started")]
    assert len(drv) >= 3, [j["id"] for j in drv]
    iv = sorted((j["started"], j["finished"] or time.time(), j["id"]) for j in drv)
    over = [(a[2], b[2]) for a, b in zip(iv, iv[1:]) if b[0] < a[1] - 1e-6]
    assert not over, over


def test_stage_jobs_cores_never_exceed_cap(A):
    """C1-X2 / D-099: the cores of the jobs that run at once (fake PBS times)
    never exceed resources.max_cores."""
    s, obs = A
    cap = int(s.cfg["resources"]["max_cores"])
    ev = []
    for j in obs["jobs"]:
        if not j.get("started"):
            continue
        n = 1
        for r in j["resources"]:
            mm = re.search(r"ncpus=(\d+)", r)
            if mm:
                n = int(mm.group(1))
        k = len(j["indices"]) if j.get("indices") else 1
        ev.append((j["started"], n * k))
        ev.append((j["finished"] or time.time(), -n * k))
    cur, peak = 0, 0
    for t, d in sorted(ev, key=lambda e: (e[0], e[1])):
        cur += d
        peak = max(peak, cur)
    assert peak <= cap, (peak, cap)


def test_artefacts_of_record_and_sidecars(A):
    """C1-H1..H3: each stage's artefact of record, with its matching sidecar."""
    s, obs = A
    d = s.cfg
    for p in (os.path.join(d["real"]["extract_root"], "cohort_manifest.json"),
              os.path.join(d["sim"]["out_root"], "REEXTRACTION_RECORD.json"),
              os.path.join(d["paths"]["sbix_dir"], d["freeze"]["out"]),
              os.path.join(d["paths"]["sbix_dir"], "export_windows", "synth", "bank_plan.json"),
              os.path.join(d["paths"]["banks_root"], "synth_sim", "BANK_RECORD.json"),
              os.path.join(d["paths"]["banks_root"], "synth_sim", "BANK_VERIFIED.json")):
        assert os.path.isfile(p), p
        tok = open(p + ".sha256").read().split()[0]
        assert re.match(r"^[0-9a-f]{64}$", tok) and tok == H.sha256_file(p), p
    cert = json.load(open(os.path.join(d["paths"]["banks_root"], "synth_sim", "BANK_VERIFIED.json")))
    assert cert["record_sha256"] == H.sha256_file(
        os.path.join(d["paths"]["banks_root"], "synth_sim", "BANK_RECORD.json"))
    for p in (os.path.join(d["paths"]["banks_root"], "synth_real", "shard_0000.npz"),
              os.path.join(d["paths"]["banks_root"], "synth_real", "shard_0000.json"),
              os.path.join(d["paths"]["tune_root"], "S-A1", "SEARCH_COMPLETE.json"),
              os.path.join(d["paths"]["runs_root"], "synth_best", "BEST_FIT_COMPLETE.json")):
        assert os.path.isfile(p), p


def test_s1_manifest_reaches_s4_through_s2s_frozen_copy(A):
    """Step 7, S1 -> S2 -> S4: the frozen copy is S1's artefact byte for byte,
    and the bank profile names it, the S2 record and the S3 freeze by the
    sha256 prefixes of the files on disk."""
    s, obs = A
    d = s.cfg
    s1 = os.path.join(d["real"]["extract_root"], "cohort_manifest.json")
    assert H.sha256_file(s.frozen) == H.sha256_file(s1)
    prof = json.load(open(os.path.join(d["paths"]["work_root"], "bank_profile_synth.json")))
    text = json.dumps(prof)
    for p in (s.frozen, os.path.join(d["sim"]["out_root"], "REEXTRACTION_RECORD.json"),
              os.path.join(d["paths"]["sbix_dir"], d["freeze"]["out"])):
        assert p in text, (p, prof)
        assert H.sha256_file(p)[:16] in text, (p, prof)


def test_s4_plan_is_the_one_both_banks_used(A):
    """Step 7, S4 -> S5, S6: the plan's sha256 is the one the simulated bank's
    record and the real bank's sidecar name (where they name one)."""
    s, obs = A
    d = s.cfg
    plan = os.path.join(d["paths"]["sbix_dir"], "export_windows", "synth", "bank_plan.json")
    h = H.sha256_file(plan)
    rec = open(os.path.join(d["paths"]["banks_root"], "synth_sim", "BANK_RECORD.json")).read()
    side = open(os.path.join(d["paths"]["banks_root"], "synth_real", "shard_0000.json")).read()
    found = [name for name, t in (("record", rec), ("real sidecar", side)) if h[:16] in t]
    OBS.setdefault("notes", {})["plan_sha_named_by"] = found
    assert "record" in found, "the simulated bank's record does not name the plan's sha256"


def test_s5_to_s7_identity_matches_the_bank(A):
    """Step 7, S5 -> S7 (C1-Q3): bank_split.json and SEARCH_COMPLETE's identity
    against the shards as written: shard count, rows, W, d_theta, donors, the
    sidecars' contract digest; the donors per part by sec. 2.3's formula."""
    s, obs = A
    d = s.cfg
    bank = os.path.join(d["paths"]["banks_root"], "synth_sim")
    arrs = _bank_arrays(bank)
    n_rows = sum(a["theta"].shape[0] for a in arrs.values())
    W = set(a["x"].shape[1] for a in arrs.values())
    dth = set(a["theta"].shape[1] for a in arrs.values())
    donors = np.concatenate([a["donor"].ravel() for a in arrs.values()])
    digests = set(json.load(open(p))["contract_digest"]
                  for p in glob.glob(os.path.join(bank, "shard_*.json")))
    bs = json.load(open(os.path.join(d["paths"]["tune_root"], "S-A1", "bank_split.json")))
    sc = json.load(open(os.path.join(d["paths"]["tune_root"], "S-A1", "SEARCH_COMPLETE.json")))
    ident = sc["identity"]
    assert bs["n_shards"] == len(arrs) == ident["n_shards"]
    assert bs["n_rows"] == n_rows == ident["n_rows"]
    assert [bs["W"]] == sorted(W) and ident["W"] == bs["W"]
    assert [bs["d_theta"]] == sorted(dth) == [d["search"]["shape_anchors"]["d_theta"]]
    assert len(digests) == 1 and bs["contract_digest"] == digests.pop() == ident["contract_digest"]
    n_d = len(np.unique(donors))
    assert bs["n_donors"] == n_d
    f = d["split"]["fractions"]
    want = [int(round(f[0] * n_d)), int(round(f[1] * n_d))]
    want.append(n_d - want[0] - want[1])
    assert bs["n_donors_split"] == want, (bs["n_donors_split"], want, n_d)
    assert ident["split_hash"] == bs["split_hash"] and ident["split_fracs"] == f
    assert ident["split_seed"] == d["search"]["train_seed"]


def test_trial_records_carry_the_identity_and_the_configuration(A):
    """C1-Q1, C1-Q3, C1-Q5: every trial's runner record and its argv against
    the configuration and the search's identity."""
    s, obs = A
    d = s.cfg
    se = d["search"]
    sc = json.load(open(os.path.join(d["paths"]["tune_root"], "S-A1", "SEARCH_COMPLETE.json")))
    ident = sc["identity"]
    led = _ledger(s)
    n_total = se["n_initial_points"] + se["gp_rounds"] * se["trials_per_round"]
    assert len(led) == n_total and all(r["status"] == "ok" for r in led.values()), led.keys()
    for tid, r in led.items():
        assert r["split_hash"] == ident["split_hash"], tid
        assert r["contract_digest"] == ident["contract_digest"], tid
        run = json.load(open(r["runner_record"]))
        assert run["split_fracs"] == d["split"]["fractions"]
        assert run["seed"] == se["train_seed"] and run["split_seed"] == se["train_seed"]
        assert run["threads"] == se["trial"]["ncpus"]
        assert run["loader"]["buffer_shards"] == d["loader"]["buffer_shards"]
        assert float(run["passes"]) == float(se["trial"]["passes"])
        assert len(run["history"]) == se["trial"]["epochs"]
        assert run["n_train"] == ident["n_train"]
        B = int(r["config"]["batch_size_npe"])
        S = math.ceil(se["trial"]["passes"] / se["trial"]["epochs"] * run["n_train"] / B)
        assert run["steps_per_epoch_effective"] == S, (tid, run["steps_per_epoch_effective"], S)
        assert r["nll"] == run["L"]
        argv = r["argv"]
        assert H.flag_value(argv, "--split-fracs", 3) == [repr(float(x)) for x in d["split"]["fractions"]]
        assert H.flag_value(argv, "--passes") == repr(float(se["trial"]["passes"]))
        assert H.flag_value(argv, "--epochs") == str(se["trial"]["epochs"])
        assert H.flag_value(argv, "--buffer-shards") == str(d["loader"]["buffer_shards"])
        assert H.flag_value(argv, "--threads") == str(se["trial"]["ncpus"])
        assert H.flag_value(argv, "--flow-hidden-layers") == str(se["flow"]["conditioner_hidden_layers"])
        assert H.flag_value(argv, "--num-bins") == str(se["flow"]["num_bins"])
        assert H.flag_value(argv, "--patience") == "99"
        assert "--resume" in argv
        # S6 -> S7 for campaign S-A1 (D-039; block 4): the real bank is built but
        # not given to the trials; the campaign prior's floor is 0
        assert "--real-shards" not in argv
        assert run["endpoint_primary"] == "simulated (arm R absent)"
        assert run["L0"] == 0.0 and run["delta_hat"] == -run["L"]


def test_gp_rounds_use_gp_seed_plus_r(A):
    """C1-Q1: round r proposes with --gp-seed gp_seed + r (the search's log of
    its login-node tuner calls)."""
    s, obs = A
    lg = open(os.path.join(s.cfg["paths"]["tune_root"], "S-A1", "jobs", "login_node.log")).read()
    seeds = [int(x) for x in re.findall(r"propose .*?--gp-seed (\d+)", lg)]
    g = s.cfg["search"]["gp_seed"]
    assert seeds[:2] == [g, g + 1], seeds


def test_finalist_is_the_ledgers_lowest_nll(A):
    """C1-Q4: SEARCH_COMPLETE's finalist and the best fit's are the ok record
    of lowest nll, with its configuration."""
    s, obs = A
    d = s.cfg
    led = _ledger(s)
    best = min((r["nll"], tid) for tid, r in led.items() if r["status"] == "ok")
    sc = json.load(open(os.path.join(d["paths"]["tune_root"], "S-A1", "SEARCH_COMPLETE.json")))
    bf = json.load(open(os.path.join(d["paths"]["runs_root"], "synth_best", "BEST_FIT_COMPLETE.json")))
    fc = json.load(open(os.path.join(d["paths"]["runs_root"], "synth_best", "finalist_config.json")))
    assert sc["finalist"]["trial_id"] == best[1] and sc["finalist"]["nll"] == best[0]
    assert sc["finalist"]["config"] == led[best[1]]["config"]
    assert bf["finalist"]["trial_id"] == best[1] and fc["config"] == led[best[1]]["config"]
    assert sc["ended_by"] == "rounds" and sc["n_trials"]["ok"] == len(led)


def test_members_carry_the_best_fit_configuration(A):
    """C1-Q2: each member's runner argv and record."""
    s, obs = A
    d = s.cfg
    b = d["best_fit"]
    best = os.path.join(d["paths"]["runs_root"], "synth_best")
    fc = json.load(open(os.path.join(best, "finalist_config.json")))
    hashes = set()
    for k in b["member_seeds"]:
        argvs = sorted(glob.glob(os.path.join(best, "jobs", "member_seed%d.*.argv" % k)))
        assert len(argvs) == 1, argvs
        argv = [l.rstrip("\n") for l in open(argvs[0]) if l.strip()]
        assert H.flag_value(argv, "--seed") == str(k)
        assert H.flag_value(argv, "--split-seed") == str(b["split_seed"])
        assert H.flag_value(argv, "--patience") == str(b["patience_epochs"])
        assert H.flag_value(argv, "--epochs") == str(b["epochs"])
        assert H.flag_value(argv, "--passes") == repr(float(b["passes"]))
        assert H.flag_value(argv, "--n-posterior-draws") == str(b["n_posterior_draws"])
        assert H.flag_value(argv, "--threads") == str(b["ncpus"])
        assert H.flag_value(argv, "--buffer-shards") == str(d["loader"]["buffer_shards"])
        assert H.flag_value(argv, "--num-bins") == str(d["search"]["flow"]["num_bins"])
        assert "--resume" in argv
        bs_ = H.flag_value(argv, "--b-sim") or H.flag_value(argv, "--batch-size")
        assert bs_ is None or int(bs_) == int(fc["config"]["batch_size_npe"])
        rec = json.load(open(glob.glob(os.path.join(best, "member_seed%d" % k,
                                                   "*_seed%d.json" % k))[0]))
        assert rec["seed"] == k and rec["split_seed"] == b["split_seed"]
        hashes.add(rec["split_hash"])
    assert len(hashes) == 1, hashes
    sc = json.load(open(os.path.join(d["paths"]["tune_root"], "S-A1", "SEARCH_COMPLETE.json")))
    if b["split_seed"] != d["search"]["train_seed"]:
        assert hashes.pop() != sc["identity"]["split_hash"]


def test_ensemble_nll_recomputed_independently(A):
    """C1-Q4: L_bar from the members' per-row NLLs with scipy, and Jensen."""
    from scipy.special import logsumexp
    s, obs = A
    d = s.cfg
    best = os.path.join(d["paths"]["runs_root"], "synth_best")
    bf = json.load(open(os.path.join(best, "BEST_FIT_COMPLETE.json")))
    rows, grp = [], []
    for k in d["best_fit"]["member_seeds"]:
        p = glob.glob(os.path.join(best, "member_seed%d" % k, "*_seed%d_perrow.npz" % k))[0]
        with np.load(p, allow_pickle=True) as z:
            rows.append(np.asarray(z["nll"], dtype=np.float64))
            grp.append(np.asarray(z["group"]))
    a = np.stack(rows)
    M = a.shape[0]
    L_bar = float(-np.mean(logsumexp(-a, axis=0) - np.log(M)))
    e = bf["ensemble"]
    assert e["M"] == M == len(d["best_fit"]["member_seeds"]) and e["n_report"] == a.shape[1]
    # 1e-12 nats: the same float64 operations in another order on ~10 rows
    assert abs(e["L_bar"] - L_bar) <= 1e-12, (e["L_bar"], L_bar)
    np.testing.assert_allclose(e["L_members"], a.mean(axis=1), rtol=0, atol=1e-12)
    assert e["L_bar"] <= float(np.mean(e["L_members"])) + 1e-12
    assert all(np.array_equal(g, grp[0]) for g in grp)


def test_bank_window_and_floor_as_written(A):
    """Step 7, S4 -> S5 -> S7: the windows the bank wrote against the
    configuration (block 9's window rule: W = round(T / Delta_t) with
    |W Delta_t - T| <= 1e-9 s; D-075's floor, a row kept iff its mean rate >=
    bank.floor_hz). The rate of a row is taken as mean(x) * f_s: the
    sidecar's scale_convention is per_electrode_mean, x per bin of w_size
    (see the report: the spec's notation says Hz)."""
    s, obs = A
    d = s.cfg
    bank = os.path.join(d["paths"]["banks_root"], "synth_sim")
    side = json.load(open(os.path.join(bank, "shard_0000.json")))
    T, dt = float(d["bank"]["t_win_s"]), float(side["w_size"])
    W = int(round(T / dt))
    assert abs(W * dt - T) <= 1e-9 and math.floor(T / dt + 1e-12) == W
    assert side["W"] == W and abs(side["fs"] * dt - 1.0) <= 1e-12 and side["T_win"] == T
    arrs = _bank_arrays(bank)
    x = np.concatenate([a["x"] for a in arrs.values()])
    assert x.shape[1] == W and x.dtype == np.float32 and np.isfinite(x).all()
    rate = x.astype(np.float64).mean(axis=1) * side["fs"]
    floor = float(d["bank"]["floor_hz"]) * (1.0 - float(d["bank"]["floor_rtol"]))
    assert (rate >= floor).all(), (float(rate.min()), floor)
    OBS.setdefault("notes", {})["bank_row_mean_x_min"] = float(x.mean(axis=1).min())


def test_bank_theta_inside_the_runners_open_box(A):
    """Step 7, S5 -> S7 (sec. 2.3: theta in (0, 1)^p; block 4: the prior is
    the open unit box, a report row outside it is refused): every theta the
    simulated bank wrote lies in the open box."""
    s, obs = A
    bank = os.path.join(s.cfg["paths"]["banks_root"], "synth_sim")
    side = json.load(open(os.path.join(bank, "shard_0000.json")))
    th = np.concatenate([a["theta"] for a in _bank_arrays(bank).values()])
    r, c = np.where((th <= 0.0) | (th >= 1.0))
    OBS.setdefault("notes", {})["theta_on_boundary"] = [
        (int(i), side["param_names"][k], float(th[i, k])) for i, k in zip(r, c)]
    assert r.size == 0, OBS["notes"]["theta_on_boundary"]


def test_runner_accepts_every_split_of_the_bank(A):
    """Step 7, S5 -> S7: the runner, given the bank the chain built, finishes
    whatever donors its split puts in the report part. The split seed is
    chosen (with the runner's own grouped_split, used here only to pick the
    input) so that the rows of the previous test lie in the report split."""
    import subprocess
    s, obs = A
    d = s.cfg
    bank = os.path.join(d["paths"]["banks_root"], "synth_sim")
    th = np.concatenate([a["theta"] for a in _bank_arrays(bank).values()])
    donor = np.concatenate([a["donor"].ravel() for a in _bank_arrays(bank).values()])
    bad = np.where(((th <= 0.0) | (th >= 1.0)).any(axis=1))[0]
    if bad.size == 0:
        pytest.skip("no bank row on the box's boundary")
    st3 = os.path.join(H.REPO, "hpc", "joint", "stage3")
    sys.path[:0] = [os.path.join(H.REPO, "hpc", "joint", x) for x in ("stage1", "stage2", "")] + [st3]
    from run_joint_arms import grouped_split
    seed = None
    for k in range(200):
        sp, _ = grouped_split(donor, fracs=tuple(d["split"]["fractions"]), seed=k)
        if set(sp[bad].tolist()) == {2}:
            seed = k
            break
    assert seed is not None
    r = sorted(_ledger(s).values(), key=lambda q: q["nll"])[0]
    argv = [a for a in r["argv"][1:] if a != "--resume"]
    i = argv.index("--out-dir")
    argv[i + 1] = os.path.join(s.root, "boundary_split")
    argv += ["--split-seed", str(seed)]
    p = subprocess.run([sys.executable] + argv, cwd=st3, env=s.env(), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=1800)
    tail = [l for l in p.stdout.splitlines() if "Error" in l or "prior" in l][-3:]
    OBS.setdefault("notes", {})["runner_boundary_split"] = {"seed": seed, "exit": p.returncode,
                                                           "lines": tail}
    assert p.returncode == 0, OBS["notes"]["runner_boundary_split"]


def test_search_clock_starts_at_round_0s_first_trial(A):
    """Block 7 (proposal 1, sec. 2.3's elapsed clock): clock_start_utc is the
    first trial's submission, after the gate's probes; elapsed_h is
    (finished - start) / 3600."""
    s, obs = A
    d = s.cfg
    sc = json.load(open(os.path.join(d["paths"]["tune_root"], "S-A1", "SEARCH_COMPLETE.json")))
    st = obs["state"]["stages"]["search"]
    by_id = dict((j["id"], j) for j in obs["jobs"])
    trials = [by_id[j["id"]]["submitted"] for j in st["jobs"] if j["role"].startswith("trial")]
    probes = [by_id[j["id"]]["submitted"] for j in st["jobs"] if not j["role"].startswith("trial")]
    t0 = _utc_epoch(sc["clock_start_utc"])
    assert abs(t0 - min(trials)) <= 1.0, (sc["clock_start_utc"], min(trials))
    assert not probes or max(probes) <= min(trials)
    el = (_utc_epoch(sc["finished_utc"]) - t0) / 3600.0
    assert abs(el - sc["elapsed_h"]) <= 2.0 / 3600.0, (el, sc["elapsed_h"])


def test_status_after_the_run(A):
    s, obs = A
    rc, out = obs["status_cmd"]
    assert rc == 0, out[-2000:]
    assert out.count("| done     | yes") == 8, out[-3000:]


def test_submission_on_a_finished_run_submits_no_stage_job(A):
    """C1-H4 / C1-R1: a fresh submission on the finished state: the driver says
    PIPELINE DONE and no stage job is submitted."""
    s, obs = A
    rc, out = obs["submit4"]
    assert rc == 0, out[-2000:]
    names = [j["name"] for j in obs["jobs_after_submit4"]]
    assert names and all(n.startswith("pl_") for n in names), names
    tail = obs["log_after_submit4"][len(obs["log"]):]
    assert "PIPELINE DONE" in tail, tail[-2000:]


def test_inputs_check_before_the_run(A):
    """C1-T1: read-only, and the last line; its FAIL rows are named for the
    report (the Sbi-extractor checkout predates D-107/D-112/D-113)."""
    s, obs = A
    rc, out = obs["inputs_check"]
    last = [l for l in out.splitlines() if l.startswith("INPUTS CHECK:")]
    assert last, out[-3000:]
    fails = [l for l in out.splitlines() if l.startswith("FAIL")]
    OBS.setdefault("notes", {})["inputs_check_fails"] = fails
    assert rc == (1 if fails else 0), (rc, fails)
    changed = dict((w, c) for w, c in obs["inputs_check_changed"].items() if c)
    assert not changed, changed


def test_mid_run_purge_refused_and_report_read_only(A):
    """C1-T2, C1-T3 while trial jobs run and no sequencer ticks: the purge
    refuses (exit 4, nothing deleted); the search report writes nothing."""
    s, obs = A
    assert obs["live_stage_jobs_at_stop2"], "no stage job was live at the second stop"
    rc, out = obs["purge_listing_mid"]
    assert rc in (0, 4), out[-2000:]
    rc, out = obs["purge_confirm_mid"]
    assert rc == 4 and "PURGE DONE" not in out, out[-3000:]
    assert obs["purge_mid_campaign_present"]
    rc, out = obs["report_mid"]
    assert rc == 0 and "SEARCH REPORT OK:" in out and "nothing was written" in out, out[-2000:]
    assert not obs["camp_changed_by_tools_mid"], obs["camp_changed_by_tools_mid"]


def test_perrow_check_on_the_finished_search(A):
    """C1-T5: the ledger's nll equals the per-row mean (1e-6), nothing written."""
    s, obs = A
    camp = os.path.join(s.cfg["paths"]["tune_root"], "S-A1")
    snap = H.tree_snapshot(camp)
    rc, out = s.tool("tools/perrow_check.py", ["--config", s.path], timeout=600)
    assert rc == 0 and "PERROW CHECK OK:" in out, out[-3000:]
    assert "MISMATCH" not in out, out[-3000:]
    assert H.tree_snapshot(camp) == snap
    for tid, r in _ledger(s).items():
        p = glob.glob(os.path.join(camp, "runs", tid, "*_perrow.npz"))[0]
        with np.load(p, allow_pickle=True) as z:
            assert abs(float(np.mean(z["nll"])) - r["nll"]) <= 1e-6, tid


# ---------------------------------------------------------------------------
# Run B: the same configuration, run straight through (determinism, C1-Q7)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def B(A):
    s = H.ChainStandin()
    obs = {"root": s.root, "submit": s.sh([], timeout=300)}
    obs["chain_end"] = s.wait_chain_end(2400)
    obs["statuses"] = s.statuses()
    OBS["B"] = obs
    yield s, obs
    if os.environ.get("C1_KEEP") != "1":
        s.close()


def test_twin_run_same_banks(A, B):
    sa, _ = A
    sb, ob = B
    assert ob["chain_end"] and " done (exit 0)" in ob["chain_end"], ob
    for kind in ("synth_sim", "synth_real"):
        a = _bank_arrays(os.path.join(sa.cfg["paths"]["banks_root"], kind))
        b = _bank_arrays(os.path.join(sb.cfg["paths"]["banks_root"], kind))
        assert sorted(a) == sorted(b), kind
        for shard in a:
            assert sorted(a[shard]) == sorted(b[shard]), (kind, shard)
            for k in a[shard]:
                np.testing.assert_array_equal(a[shard][k], b[shard][k], err_msg="%s %s %s"
                                              % (kind, shard, k))


def test_twin_run_same_search_and_best_fit(A, B):
    """The numbers of the two runs, matched by configuration: the trial ids
    and the contract digest are not compared, because the two stand-ins'
    inputs differ by construction (the fixture's manifest holds its own
    absolute paths, and S3's freeze its `sim_main` path and `created_utc`),
    and the contract digest -- hence the trial id -- covers their sha256."""
    sa, _ = A
    sb, _ = B
    pa = json.load(open(os.path.join(sa.cfg["paths"]["tune_root"], "S-A1", "bank_split.json")))
    pb = json.load(open(os.path.join(sb.cfg["paths"]["tune_root"], "S-A1", "bank_split.json")))
    for k in ("split_hash", "n_rows", "n_train", "n_sel", "n_report", "n_donors_split"):
        assert pa[k] == pb[k], k
    la, lb = _ledger(sa), _ledger(sb)
    ka = sorted((json.dumps(r["config"], sort_keys=True), r["nll"]) for r in la.values())
    kb = sorted((json.dumps(r["config"], sort_keys=True), r["nll"]) for r in lb.values())
    # one thread, the same seeds: the same configurations and NLLs bit for bit
    assert ka == kb, (ka, kb)
    ba = json.load(open(os.path.join(sa.cfg["paths"]["runs_root"], "synth_best",
                                     "BEST_FIT_COMPLETE.json")))
    bb = json.load(open(os.path.join(sb.cfg["paths"]["runs_root"], "synth_best",
                                     "BEST_FIT_COMPLETE.json")))
    fa = json.load(open(os.path.join(sa.cfg["paths"]["runs_root"], "synth_best",
                                     "finalist_config.json")))
    fb = json.load(open(os.path.join(sb.cfg["paths"]["runs_root"], "synth_best",
                                     "finalist_config.json")))
    assert fa["config"] == fb["config"]
    assert ba["ensemble"]["L_bar"] == bb["ensemble"]["L_bar"]
    assert ba["ensemble"]["L_members"] == bb["ensemble"]["L_members"]


# ---------------------------------------------------------------------------
# Searches on run A's banks: S1-S6 skipped by their artefacts (C1-H4)
# ---------------------------------------------------------------------------

def _on_banks_of(sa, name, search=None, flow=None, split=None, loader=None, cap=None,
                 trial=None):
    """A configuration whose S1-S6 artefacts are run A's and whose work root,
    tune root and runs root are new (another folder under run A's root)."""
    cfg = json.loads(json.dumps(sa.cfg))
    base = os.path.join(sa.root, name)
    cfg["paths"]["work_root"] = os.path.join(base, "work_root")
    cfg["paths"]["tune_root"] = os.path.join(base, "tune")
    cfg["paths"]["runs_root"] = os.path.join(base, "runs")
    if search:
        cfg["search"].update(search)
    if flow:
        cfg["search"]["flow"].update(flow)
    if trial:
        cfg["search"]["trial"].update(trial)
    if split:
        cfg["split"] = split
    if loader:
        cfg["loader"] = loader
    if cap:
        cfg["resources"]["max_cores"] = cap
    os.makedirs(base)
    v = H.ChainStandin.__new__(H.ChainStandin)
    v.root, v.sbix, v.fakebin, v.ds = base, sa.sbix, sa.fakebin, sa.ds
    v.cfg, v.path = cfg, os.path.join(base, "synth.json")
    v.spool, v.work_root, v.frozen = os.path.join(base, "spool"), cfg["paths"]["work_root"], sa.frozen
    v.write_cfg()
    return v


def _first_trial_runner_argv(v, timeout=900):
    camp = os.path.join(v.cfg["paths"]["tune_root"], "S-A1")

    def got():
        for p in sorted(glob.glob(os.path.join(camp, "jobs", "*.0.log"))):
            a = H.runner_argv_from_log(open(p).read())
            if a:
                return a
        return None
    return v.wait_for(got, timeout, every=1.0)


def test_configuration_values_reach_the_runner(A):
    """C1-Q1 with values that differ from every default: flow.num_bins 6,
    conditioner_hidden_layers 3, passes 1.5 in 3 epochs, buffer_shards 3,
    split 0.5/0.3/0.2, train_seed 5. The first trial's runner command line
    (its job log) must carry each."""
    sa, _ = A
    v = _on_banks_of(sa, "values", search={"train_seed": 5, "gp_seed": 11},
                     flow={"num_bins": 6, "conditioner_hidden_layers": 3},
                     trial={"passes": 1.5, "epochs": 3},
                     split={"fractions": [0.5, 0.3, 0.2]}, loader={"buffer_shards": 3})
    rc, out = v.sh([], timeout=300)
    assert rc == 0, out[-2000:]
    argv = _first_trial_runner_argv(v)
    _stop(v)
    v.wait_for(lambda: not v.live_jobs(), 600)
    OBS.setdefault("notes", {})["values_runner_argv"] = argv
    assert argv, v.log_text()[-3000:]
    st = v.statuses()
    assert all(st[n] == "skipped" for n in STAGES[:6]), st
    got = {"--num-bins": H.flag_value(argv, "--num-bins"),
           "--flow-hidden-layers": H.flag_value(argv, "--flow-hidden-layers"),
           "--passes": H.flag_value(argv, "--passes"), "--epochs": H.flag_value(argv, "--epochs"),
           "--buffer-shards": H.flag_value(argv, "--buffer-shards"),
           "--split-fracs": H.flag_value(argv, "--split-fracs", 3),
           "--seed": H.flag_value(argv, "--seed")}
    want = {"--num-bins": "6", "--flow-hidden-layers": "3", "--passes": "1.5", "--epochs": "3",
            "--buffer-shards": "3", "--split-fracs": ["0.5", "0.3", "0.2"], "--seed": "5"}
    diff = dict((k, (got[k], want[k])) for k in want if got[k] != want[k])
    assert not diff, diff


def test_identity_change_while_the_search_runs_is_refused(A):
    """C1-F4, step 9 (an artefact reused after its upstream changed): a search
    whose round 0 runs; the sequencer stopped; split.fractions changed (an
    identity field, block 7's rule of the identity) and the submission made
    with --accept-config-change. The spec: a results folder whose identity
    disagrees with the configuration is refused before anything is submitted.
    Observed: whether any trial job is submitted after the change, and on
    which split it ran."""
    sa, _ = A
    v = _on_banks_of(sa, "identity", search={"n_initial_points": 3, "max_concurrent_trials": 1})
    rc, out = v.sh([], timeout=300)
    assert rc == 0, out[-2000:]

    def one_running():
        st = v.state()
        sub = (st or {}).get("stages", {}).get("search", {}).get("sub") or {}
        tr = sub.get("trials") or {}
        return sum(1 for t in tr.values() if t.get("status") == "running") >= 1
    assert v.wait_for(one_running, 1200, every=1.0)
    _stop(v)
    n_before = len([j for j in v.jobs() if j["name"].startswith("t_")])
    v.cfg["split"] = {"fractions": [0.5, 0.3, 0.2]}
    v.write_cfg()
    rc, out = v.sh(["--accept-config-change"], timeout=300)
    end = v.wait_chain_end(2400)
    trials = [j for j in v.jobs() if j["name"].startswith("t_")]
    led = _ledger(v, tune_root=v.cfg["paths"]["tune_root"])
    rf = json.load(open(os.path.join(v.cfg["paths"]["tune_root"], "S-A1", "pipeline_rounds.json")))
    hashes = sorted(set(r.get("split_hash") for r in led.values()))
    fr = []
    for r in led.values():
        fr.append(json.load(open(r["runner_record"]))["split_fracs"])
    OBS.setdefault("notes", {})["identity_change"] = {
        "submit_rc": rc, "chain_end": end, "trial_jobs_before": n_before,
        "trial_jobs_after": len(trials), "ledger_split_hashes": hashes,
        "identity_split_hash": rf["identity"]["split_hash"],
        "identity_split_fracs": rf["identity"]["split_fracs"],
        "runner_split_fracs": fr, "log_tail": v.log_text()[-4000:]}
    assert len(trials) == n_before and "REFUSE" in v.log_text(), \
        OBS["notes"]["identity_change"]


def test_cap_holds_the_queued_successor_too(A):
    """C1-X2 / D-099 (sec. 4: "the cores requested by the pipeline's jobs
    that are queued, held or running, the sequencer's own job included, never
    sum to more than resources.max_cores"), on a search whose cap leaves room
    for exactly two 1-core trials (max_cores 3 = the driver's 1 + 2). The fake
    PBS is sampled every 0.2 s: the cores of the jobs queued, held or running,
    and of those running."""
    import threading
    sa, _ = A
    v = _on_banks_of(sa, "cap", cap=3, search={"max_concurrent_trials": 3})
    peak = {"requested": 0, "running": 0, "at": None}
    stop = threading.Event()

    def ncpus(j):
        for r in j["resources"]:
            mm = re.search(r"ncpus=(\d+)", r)
            if mm:
                return int(mm.group(1))
        return 1

    def sample():
        while not stop.is_set():
            try:
                js = v.jobs()
            except Exception:
                js = []
            req = sum(ncpus(j) for j in js if j.get("state") in ("Q", "H", "W", "R", "B"))
            run = sum(ncpus(j) for j in js if j.get("state") in ("R", "B"))
            if req > peak["requested"]:
                peak["requested"] = req
                peak["at"] = sorted((j["name"], j["state"]) for j in js
                                    if j.get("state") in ("Q", "H", "W", "R", "B"))
            peak["running"] = max(peak["running"], run)
            time.sleep(0.2)
    th = threading.Thread(target=sample)
    th.start()
    try:
        rc, out = v.sh([], timeout=300)
        assert rc == 0, out[-2000:]
        end = v.wait_chain_end(2400)
    finally:
        stop.set()
        th.join()
    OBS.setdefault("notes", {})["cap_peak"] = dict(peak, end=end)
    assert end and " done (exit 0)" in end, end
    assert peak["running"] <= 3, peak
    assert peak["requested"] <= 3, peak


# ---------------------------------------------------------------------------
# After the run: the purge (block 10) and the rebuild (C1-T2, C1-R1)
# ---------------------------------------------------------------------------

def test_purge_then_retry_rebuilds_the_same_outputs(A):
    """C1-T2 then C1-R1: the purge of search, best_fit and sim_bank deletes
    exactly its targets; the state is untouched, so a plain submission does
    nothing (done is not re-checked) and --status shows the discrepancy;
    --retry of the three stages rebuilds them, and the rebuilt bank, ledger
    and ensemble equal the purged ones (the same seeds)."""
    s, obs = A
    d = s.cfg
    sim = os.path.join(d["paths"]["banks_root"], "synth_sim")
    camp = os.path.join(d["paths"]["tune_root"], "S-A1")
    best = os.path.join(d["paths"]["runs_root"], "synth_best")
    old_bank = _bank_arrays(sim)
    old_led = dict((k, (v["config"], v["nll"])) for k, v in _ledger(s).items())
    old_bf = json.load(open(os.path.join(best, "BEST_FIT_COMPLETE.json")))
    kept = [os.path.join(d["paths"]["banks_root"], "synth_real"),
            os.path.join(d["paths"]["sbix_dir"], "export_windows", "synth"),
            d["paths"]["work_root"]]
    snap_kept = dict((k, H.tree_snapshot(k)) for k in kept)
    rc, lst = s.tool("tools/purge_outputs.py", ["--config", s.path, "--what",
                                                "search,best_fit,sim_bank"])
    assert rc == 0, lst[-3000:]
    code = re.search(r"CODE ([0-9a-f]{8})", lst).group(1)
    rc, out = s.tool("tools/purge_outputs.py", ["--config", s.path, "--what",
                                                "search,best_fit,sim_bank", "--confirm", code])
    assert rc == 0 and "PURGE DONE:" in out, out[-3000:]
    for p in (sim, camp, best):
        assert not os.path.exists(p), p
    for k in kept[:2]:
        assert H.tree_snapshot(k) == snap_kept[k], k
    # the work root: only purge/ is new
    wr_new = H.tree_snapshot(d["paths"]["work_root"])
    changed = [k for k in set(wr_new) | set(snap_kept[d["paths"]["work_root"]])
               if wr_new.get(k) != snap_kept[d["paths"]["work_root"]].get(k)
               and "purge" not in k and k != "D ."]
    assert not changed, changed
    # a plain submission: the state says done; nothing re-runs (C1-R1)
    rc, st = s.sh(["--status"], timeout=300)
    assert rc == 0
    for n in ("sim_bank", "search", "best_fit"):
        line = [l for l in st.splitlines() if "| %s" % n in l][0]
        assert "| done     | no" in line, line
    # --retry of the three: rebuilt
    n0 = len(s.jobs())
    rc, out = s.sh(["--retry", "sim_bank", "--retry", "search", "--retry", "best_fit"],
                   timeout=300)
    assert rc == 0, out[-2000:]
    s.wait_for(lambda: (s.chain_end() or "") and len(s.jobs()) > n0 + 3, 2400, every=2.0)
    s.wait_chain_end(2400)
    assert s.statuses() == dict((n, "done") for n in STAGES), s.statuses()
    new_bank = _bank_arrays(sim)
    assert sorted(new_bank) == sorted(old_bank)
    for shard in old_bank:
        for k in old_bank[shard]:
            np.testing.assert_array_equal(new_bank[shard][k], old_bank[shard][k])
    new_led = dict((k, (v["config"], v["nll"])) for k, v in _ledger(s).items())
    assert new_led == old_led
    new_bf = json.load(open(os.path.join(best, "BEST_FIT_COMPLETE.json")))
    assert new_bf["ensemble"]["L_bar"] == old_bf["ensemble"]["L_bar"]


def test_search_reset_then_retry_reproposes_the_same_round_0(A):
    """C1-T4 on the finished (rebuilt) run (block 7, `tools/search_reset.py`):
    the listing moves nothing; --confirm moves the search's items to a
    set-aside folder byte for byte and keeps the gate and bank_split.json;
    then a submission with --retry search --accept-config-change runs the
    search again with no probe job (the gate's files reused) and proposes
    the same round-0 configurations (gp_seed unchanged), with the same NLLs."""
    s, obs = A
    d = s.cfg
    camp = os.path.join(d["paths"]["tune_root"], "S-A1")
    rf_old = json.load(open(os.path.join(camp, "pipeline_rounds.json")))
    led_old = dict((k, (v["config"], v["nll"])) for k, v in _ledger(s).items())
    snap = H.tree_snapshot(camp)
    hashes = dict((k, v[2]) for k, v in snap.items() if k.startswith("F ") and v[2])
    rc, lst = s.tool("tools/search_reset.py", ["--config", s.path])
    assert rc == 0, lst[-3000:]
    assert H.tree_snapshot(camp) == snap
    code = re.search(r"CODE ([0-9a-f]{8})", lst).group(1)
    rc, out = s.tool("tools/search_reset.py", ["--config", s.path, "--confirm", code])
    assert rc == 0 and "SEARCH RESET DONE" in out, out[-3000:]
    aside = glob.glob(os.path.join(d["paths"]["tune_root"], "S-A1_set_aside_*"))
    assert len(aside) == 1, aside
    moved = 0
    for k, h in hashes.items():
        rel = k[2:]
        if rel.startswith(("gate/", "bank_split.json")):
            assert os.path.isfile(os.path.join(camp, rel)), rel
            continue
        p = os.path.join(aside[0], rel)
        if os.path.isfile(p):
            assert H.sha256_file(p) == h, rel
            moved += 1
    assert moved >= len(led_old), moved
    assert not os.path.exists(os.path.join(camp, "SEARCH_COMPLETE.json"))
    n0 = len(s.jobs())
    rc, out = s.sh(["--retry", "search", "--accept-config-change"], timeout=300)
    assert rc == 0, out[-2000:]
    s.wait_for(lambda: len(s.jobs()) > n0 + 1 and s.chain_end(), 2400, every=2.0)
    s.wait_chain_end(2400)
    new_jobs = [j for j in s.jobs()[n0:] if not j["name"].startswith("pl_")]
    assert not [j for j in new_jobs if j["name"].startswith("gate_")], [j["name"] for j in new_jobs]
    rf_new = json.load(open(os.path.join(camp, "pipeline_rounds.json")))
    assert rf_new["rounds"][0]["trial_ids"] == rf_old["rounds"][0]["trial_ids"]
    led_new = dict((k, (v["config"], v["nll"])) for k, v in _ledger(s).items())
    assert led_new == led_old
    assert s.statuses()["search"] == "done" and "PIPELINE DONE" in s.log_text()[-3000:]


def teardown_module(module):
    out = os.environ.get("C1_OBS_OUT")
    if out:
        def clean(o):
            if isinstance(o, dict):
                return dict((k, clean(v)) for k, v in o.items() if k not in ("jobs",))
            if isinstance(o, (list, tuple)):
                return [clean(x) for x in o]
            return o
        with open(out, "w") as fh:
            json.dump(clean(OBS), fh, indent=1, default=str)
