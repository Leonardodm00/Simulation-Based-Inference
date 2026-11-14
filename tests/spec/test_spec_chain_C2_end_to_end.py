"""Chain t2-C2, step 6 and step 7: the synthetic experiment end to end
through its real entry point (`run_pipeline.sh --config <reduced synth>`:
`pipeline.py submit`, the sequencer as PBS jobs of the fake PBS, D-098),
then every producer-consumer pair along the chain read from what the
producers wrote.

Oracles C-O2 .. C-O10 of the report (specs/test_reports/
2026-11-14_0030_full_t2-C2.md). Every expected value below is computed here
from specs/SPEC.md and the reduced configuration (c2_helpers.reduced), with
numpy, scipy and hashlib -- never by the code under test.

The reduced configuration: arm S 40 theta x 2 wells x 1 window in 4 shards
of 20 wells (W = round(12.8 s x 10 Hz) = 128); R0 and R3 of 3 conditions x
2 donors x 2 wells x 2 windows (24 windows, R3 at pi 0.3); four searches
(S-A1 on R0; S-A2-only on R0; S-A5-only on R3; a mixed S-A2 on R3 with
alt_best_fit true), each 2 + 1 x 2 trials, batch sizes 16 and 32; 2 members
on split seed 1; every job 1 core under resources.max_cores 3.
"""

import glob
import hashlib
import json
import math
import os
import re
from datetime import datetime

import numpy as np
import pytest
from scipy.special import logsumexp

import c2_helpers as H

# ---- the reduced configuration's numbers, by hand (sec. 2.3, 2.4, 3.1) ----
N_THETA, WPD, TPS, SIM_WIN = 40, 2, 20, 1
K_SIM = N_THETA * WPD // TPS                    # 4 shards
ROWS_SIM_SHARD = TPS * SIM_WIN                  # 20 rows: wells x windows
W = int(round(12.8 * 10.0))                     # 128 samples
C, DPC, R_WIN = 3, 2, 2
ROWS_R = C * DPC * WPD * R_WIN                  # 24 windows
N_DONORS = N_THETA                              # arm S: one donor per theta
FR = (0.85, 0.10, 0.05)
D_TR = int(round(FR[0] * N_DONORS))             # 34 donors
D_SEL = int(round(FR[1] * N_DONORS))            # 4
D_REP = N_DONORS - D_TR - D_SEL                 # 2
N_TRAIN, N_SEL, N_REP = D_TR * WPD * SIM_WIN, D_SEL * WPD * SIM_WIN, D_REP * WPD * SIM_WIN
TRIALS = 2 + 1 * 2                              # n_initial_points + gp_rounds x per round
BATCHES = (16, 32)
SEEDS = {"sim": 11, "R0": 22, "R3": 33}
PI = {"R0": 0.0, "R3": 0.3}
MEMBERS = (1, 2)
SPLIT_SEED_BF = 1
REAL_OF = dict((q["name"], q["real"]) for q in H.SEARCHES)
STAGES = (["sim_bank", "real_bank_R0", "real_bank_R3"]
          + ["search_S-A1", "best_fit_S-A1", "search_S-A2-R0", "best_fit_S-A2-R0",
             "search_S-A5-R3", "best_fit_S-A5-R3", "search_S-A2m-R3", "best_fit_S-A2m-R3",
             "best_fit_alt_S-A2m-R3"])
DEPS = {"sim_bank": (), "real_bank_R0": (), "real_bank_R3": ()}
for _q in H.SEARCHES:
    DEPS["search_" + _q["name"]] = ("sim_bank", "real_bank_" + _q["real"])
    DEPS["best_fit_" + _q["name"]] = ("search_" + _q["name"],)
    if _q["alt_best_fit"]:
        DEPS["best_fit_alt_" + _q["name"]] = ("search_" + _q["name"],)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc(text):
    return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")


def _arm(config):
    """sec. 2.5: dsn_on -> A2, rep_on -> A5, both off -> A1, both on -> none."""
    d, r = int(config.get("dsn_on", 0)), int(config.get("rep_on", 0))
    return {(0, 0): "A1", (1, 0): "A2", (0, 1): "A5"}.get((d, r))


def _flag(argv, name):
    """The value after the LAST occurrence of `name` (the last one wins)."""
    idx = [i for i, a in enumerate(argv) if a == name]
    return argv[idx[-1] + 1] if idx else None


def _flags_n(argv, name, n):
    idx = [i for i, a in enumerate(argv) if a == name]
    return argv[idx[-1] + 1: idx[-1] + 1 + n] if idx else None


@pytest.fixture(scope="module")
def r1(run1):
    cfg_path, cfg, spool, env, out, ce = run1
    assert ce is not None, "the chain did not end (no driver/CHAIN_END): submit said:\n" + out
    return {"cfg_path": cfg_path, "cfg": cfg, "spool": spool, "env": env, "out": out, "ce": ce,
            "jobs": H.jobs(spool), "state": H.state(cfg), "log": H.log_text(cfg)}


# ---------------------------------------------------------------- C-O4 ---

def test_entry_point_and_the_end(r1):
    """`run_pipeline.sh` submitted the sequencer's first job and the chain
    ended on purpose with every stage done: CHAIN_END `done (exit 0)`, the
    log's `PIPELINE DONE`, no STOP and no traceback in any driver's log."""
    if r1["out"] != "(reused)":
        assert re.search(r"SUBMITTED the sequencer: job \S+ \(pl_synth\)", r1["out"]), r1["out"]
    assert r1["ce"].split()[-3:] == ["done", "(exit", "0)"] or "done (exit 0)" in r1["ce"], r1["ce"]
    assert "PIPELINE DONE" in r1["log"]
    assert "STOP:" not in r1["log"] and "REFUSE" not in r1["log"]
    for p in glob.glob(os.path.join(H.work_root(r1["cfg"]), "driver", "*.log")):
        with open(p, errors="replace") as fh:
            assert "Traceback" not in fh.read(), p


def test_stage_order_dependencies_and_artefacts(r1):
    """C-O2/C-O4: the graph's stages in sec. 2.1's order, every one done on a
    fresh run, none started before its dependencies finished; each artefact
    of record where sec. 2.1 puts it, a bank's record with a matching
    .sha256."""
    st = r1["state"]["stages"]
    assert list(st) == sorted(st)                       # sorted keys (sec. 3.2)
    assert set(st) == set(STAGES)
    for name in STAGES:
        assert st[name]["status"] == "done", (name, st[name]["status"], st[name].get("note"))
        for dep in DEPS[name]:
            assert _utc(st[name]["started_utc"]) >= _utc(st[dep]["finished_utc"]), (name, dep)
    cfg = r1["cfg"]
    for b in ("sim", "R0", "R3"):
        rec = os.path.join(H.bank_dir(cfg, b), "BANK_RECORD.json")
        with open(rec + ".sha256") as fh:
            tok = fh.read().split()[0]
        assert re.fullmatch(r"[0-9a-f]{64}", tok) and tok == _sha(rec), b
    for q in H.SEARCHES:
        assert os.path.isfile(os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json"))
        assert os.path.isfile(os.path.join(H.best_dir(cfg, q), "BEST_FIT_COMPLETE.json"))
        assert os.path.isfile(os.path.join(H.best_dir(cfg, q, alt=True), "BEST_FIT_COMPLETE.json")) \
            == bool(q["alt_best_fit"])


def test_the_jobs_once_each(r1):
    """C-O4/C-O11's baseline: every stage job submitted once (no resubmission
    on a clean run), with the job names of blocks 7, 16 and 18, exit 0."""
    stage_jobs = [j for j in r1["jobs"] if j["name"] != "pl_synth"]
    names = sorted(j["name"] for j in stage_jobs)
    expect = (["sb_sim_%d" % k for k in range(K_SIM)] + ["sb_R0_0", "sb_R3_0"]
              + ["sr_sim", "sr_R0", "sr_R3"]
              + ["gate_b%d" % b for b in BATCHES] * len(H.SEARCHES)
              + ["bf_synth_s%d" % k for k in MEMBERS] * len(H.SEARCHES)
              + ["ba_synth_s%d" % k for k in MEMBERS])
    trials = [n for n in names if n.startswith("t_")]
    assert sorted(n for n in names if not n.startswith("t_")) == sorted(expect)
    assert len(trials) == TRIALS * len(H.SEARCHES)
    outs = [j["out"] for j in stage_jobs]
    assert len(outs) == len(set(outs)), "a stage job's output file twice: a job submitted twice"
    for j in stage_jobs:
        assert j["state"] == "F" and j["exit_status"] == 0, (j["name"], j["exit_status"])


_EV = re.compile(r"^\S+ \S+ (search_\S+|best_fit_alt_\S+|best_fit_\S+|sim_bank|real_bank_\S+): "
                 r"(.*)$")


def test_earliest_search_gets_the_free_cores_first(r1):
    """sec. 2.1 (synthetic kind): "The sequencer ticks the stages in this
    order, and every stage's submissions share resources.max_cores (D-099),
    so the free cores go to the earliest search (and its best fit) first."
    Read from the log in its own order: a job submitted by a stage while an
    EARLIER search stage (or its best fit) had a proposed trial (member)
    waiting and room under max_concurrent_trials -- every job is 1 core
    here, so the earlier stage's job would have fitted the same core -- is
    an inversion of that priority."""
    order = [s for s in STAGES if s.startswith(("search_", "best_fit"))]
    waiting = dict((s, 0) for s in order)
    running = dict((s, 0) for s in order)
    started = set()
    inversions = []
    for line in r1["log"].splitlines():
        m = _EV.match(line)
        if not m or m.group(1) not in waiting:
            continue
        stg, text = m.group(1), m.group(2)
        ts = line[11:19]
        if stg.startswith("best_fit") and stg not in started and \
                (text.startswith("submitted member") or text.startswith("running")):
            started.add(stg)
            waiting[stg] += len(MEMBERS)                   # every member waits at the start
        if stg.startswith("search_"):
            if text.startswith("budget gate passed"):
                waiting[stg] += 2                           # round 0: n_initial_points
                continue
            p = re.match(r"round [0-9]+ proposed: ([0-9]+) trial", text)
            if p:
                waiting[stg] += int(p.group(1))
                continue
        s = re.match(r"submitted (trial \S+|member seed [0-9]+|gate_b[0-9]+) -> \S+", text)
        if s:
            for earlier in order[:order.index(stg)]:
                room = running[earlier] < 2 if earlier.startswith("search_") else True
                if waiting[earlier] > 0 and room:
                    inversions.append("%s %s submitted %s while %s had %d waiting, %d running"
                                      % (ts, stg, s.group(1), earlier, waiting[earlier],
                                         running[earlier]))
            if not s.group(1).startswith("gate"):
                waiting[stg] = max(0, waiting[stg] - 1)
                running[stg] += 1
            continue
        if re.match(r"trial \S+ (ok|failed|killed)", text) or re.match(r"member seed [0-9]+ done", text):
            running[stg] = max(0, running[stg] - 1)
    print("\n".join(inversions))
    assert not inversions, "%d inversion(s) of the search order's priority:\n" % len(inversions) + \
        "\n".join(inversions[:14])


def test_core_cap(r1):
    """C-O5 (D-099): the cores of the jobs queued, held or running never
    exceed resources.max_cores = 3, counting the running sequencer job. The
    successor driver waiting on its `afterany` dependency is reported apart
    (sec. 4's wording counts held jobs; see the report, S-items)."""
    worst, when, who = H.cores_profile(r1["jobs"], count_dependent=False)
    assert worst <= 3, (worst, who)
    worst_all, _, who_all = H.cores_profile(r1["jobs"], count_dependent=True)
    print("cores in flight: %d without the dependent successor, %d with it (%s)"
          % (worst, worst_all, ",".join(who_all)))


# ---------------------------------------------------------------- C-O6 ---

def _bank_arrays(cfg, b):
    paths = sorted(glob.glob(os.path.join(H.bank_dir(cfg, b), "shard_*.npz")))
    out = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            arrs = dict((k, z[k]) for k in z.files)
        out.append((p, arrs, H.read_json(p[:-4] + ".json")))
    return out


def _expected_argv(b):
    """sec. 5 block 16's argv of a shard, by hand: {flag: value} (the spec
    fixes the flags and values, not their order); floats as repr."""
    opts = {"--provider": "bench", "--fs": repr(10.0), "--T-win": repr(12.8),
            "--n-neurons": "100", "--wells-per-donor": "2", "--donors-per-batch": "8",
            "--n-classes": "3", "--tau-ov": repr(0.1), "--class-centres": "regular",
            "--centre-scale": repr(0.562743), "--label-axes": "0,1,2,3,4,5,6,7"}
    if b == "sim":
        return [dict(opts, **{"--arm": "S", "--prior": "box", "--shard-index": str(k),
                              "--n-traces": str(TPS), "--n-windows": str(SIM_WIN),
                              "--seed": str(SEEDS["sim"])}) for k in range(K_SIM)]
    return [dict(opts, **{"--arm": "R", "--shard-index": "0", "--n-traces": str(C * DPC * WPD),
                          "--donors-per-class": str(DPC), "--n-windows": str(R_WIN),
                          "--pi": repr(PI[b]), "--gap-modes": "range_shift,drift,contamination",
                          "--seed": str(SEEDS[b])})]


def _pairs(argv):
    """{flag: value} of a builder argv without --out-dir (every flag here
    takes one value)."""
    out, i = {}, 0
    a = [x for x in argv]
    while i < len(a):
        assert a[i].startswith("--"), a
        out[a[i]] = a[i + 1]
        i += 2
    out.pop("--out-dir", None)
    return out


@pytest.mark.parametrize("b", ["sim", "R0", "R3"])
def test_banks_against_the_spec(r1, b):
    """C-O6: the shards the chain built, read with numpy: the shard set, W,
    theta's width and box, rows per shard, wells per donor and shared theta,
    classes, the withheld flag and the latent kind, one contract digest,
    the bank-wide ids (block 12), the seed and pi carried into the sidecar;
    the record's numbers and its shards' sha256; its build equal to the
    hand-written one; the bank job's argv the same."""
    cfg = r1["cfg"]
    shards = _bank_arrays(cfg, b)
    k_exp = K_SIM if b == "sim" else 1
    assert [os.path.basename(p) for p, _, _ in shards] == ["shard_%04d.npz" % k for k in range(k_exp)]
    files = sorted(os.listdir(H.bank_dir(cfg, b)))
    assert files == sorted(["shard_%04d.%s" % (k, e) for k in range(k_exp) for e in ("npz", "json")]
                           + ["BANK_RECORD.json", "BANK_RECORD.json.sha256"]), files
    digests = set()
    exp_argv = _expected_argv(b)
    for k, (p, a, sc) in enumerate(shards):
        n = a["x"].shape[0]
        assert a["x"].shape == (n, W) and a["x"].dtype == np.float32
        assert a["theta"].shape == (n, 10)
        assert np.all(a["theta"] > 0) and np.all(a["theta"] < 1), "theta outside the open box"
        assert n == (ROWS_SIM_SHARD if b == "sim" else ROWS_R)
        digests.add(sc["contract_digest"])
        assert sc["arm"] == ("S" if b == "sim" else "R")
        assert bool(sc["theta_withheld"]) == (b != "sim")
        assert (sc["latent_spec"] or {}).get("kind") == ("box" if b == "sim" else None)
        assert sc["latent_spec"]["seed"] == SEEDS[b]
        assert _pairs(sc["provenance"]["argv"]) == exp_argv[k]
        # wells: each donor exactly WPD wells, theta shared by the donor's rows
        donors = np.unique(a["donor"])
        for d in donors:
            m = a["donor"] == d
            assert len(np.unique(a["well"][m])) == WPD
            assert np.all(a["theta"][m] == a["theta"][m][0])
        # bank-wide ids (block 12): shard k's ids in [k 1e6, (k + 1) 1e6)
        for key in ("donor", "well", "batch"):
            assert np.all(a[key] >= k * 1000000) and np.all(a[key] < (k + 1) * 1000000), key
        if b == "sim":
            assert np.all(a["cls"] == -1)
            assert len(donors) == TPS // WPD
        else:
            assert sorted(np.unique(a["cls"]).tolist()) == list(range(C))
            for c in range(C):
                assert int((a["cls"] == c).sum()) == DPC * WPD * R_WIN
            assert abs(float(sc["gap_spec"]["pi"]) - PI[b]) == 0.0
    assert len(digests) == 1
    rec_path = os.path.join(H.bank_dir(cfg, b), "BANK_RECORD.json")
    rec = H.read_json(rec_path)
    assert rec["n_shards"] == k_exp
    assert rec["n_rows"] == sum(a["x"].shape[0] for _, a, _ in shards)
    assert rec["W"] == W and rec["d_theta"] == 10
    assert rec["contract_digest"] == digests.pop()
    assert rec["arm"] == ("S" if b == "sim" else "R")
    assert bool(rec["theta_withheld"]) == (b != "sim")
    for s in rec["shards"]:
        assert s["sha256"] == _sha(os.path.join(H.bank_dir(cfg, b), s["file"]))
    bd = rec["build"]
    assert bd["schema"] == "synth_bank_build/1" and bd["arm"] == ("S" if b == "sim" else "R")
    assert bd["n_shards"] == k_exp
    assert bd["rows_per_shard"] == [ROWS_SIM_SHARD if b == "sim" else ROWS_R] * k_exp
    assert [_pairs(x) for x in bd["argv"]] == exp_argv
    # the job that built shard k ran exactly that argv (block 16), in envs.train
    stage = "sim_bank" if b == "sim" else "real_bank_" + b
    for k in range(k_exp):
        argv = H.read_argv(os.path.join(H.work_root(cfg), "bank_jobs", stage, "shard_%d.0.argv" % k))
        assert argv[1] == os.path.join(H.REPO, "hpc", "joint", "stage1", "build_latent_bank.py")
        assert _flag(argv, "--out-dir") == H.bank_dir(cfg, b)
        assert _pairs(argv[2:]) == exp_argv[k]
    jobs = [j for j in r1["jobs"] if j["name"] in (["sb_%s_%d" % ("sim" if b == "sim" else b, k)
                                                   for k in range(k_exp)] + ["sr_" + b])]
    for j in jobs:
        assert j["queue"] == "test"
        assert "select=1:ncpus=1:mem=1gb" in j["resources"] and "walltime=00:10:00" in j["resources"]
        assert j["vars"]["ENV_NAME"] == "sbi_env"
        assert j["vars"]["AVOID_FILE"].endswith("avoid_nodes.txt")


# ------------------------------------------------------- C-O7 / C-O10 ---

def _q(name):
    return [q for q in H.SEARCHES if q["name"] == name][0]


def test_search_identity_and_split(r1):
    """C-O7/C-O10: each search's bank_split.json is arm S's (4 shards, 80
    rows, W 128, d_theta 10) split by donor at the configuration's
    fractions with the search's split seed (train_seed 0): 34/4/2 donors,
    68/8/4 rows (sec. 2.3's rule, by hand); its contract digest the sim
    sidecars'; every ok trial's record carries that split hash (the runner's
    own, the tuner copying it from the runner's record)."""
    cfg = r1["cfg"]
    sim_digest = H.read_json(os.path.join(H.bank_dir(cfg, "sim"), "shard_0000.json"))["contract_digest"]
    hashes = set()
    for q in H.SEARCHES:
        bs = H.read_json(os.path.join(H.campaign_dir(cfg, q), "bank_split.json"))
        assert bs["n_shards"] == K_SIM and bs["n_rows"] == K_SIM * ROWS_SIM_SHARD
        assert (bs["n_train"], bs["n_sel"], bs["n_report"]) == (N_TRAIN, N_SEL, N_REP)
        assert bs["n_donors_split"] == [D_TR, D_SEL, D_REP]
        assert bs["W"] == W and bs["d_theta"] == 10 and bs["split_seed"] == 0
        assert bs["split_fracs"] == list(FR)
        assert bs["contract_digest"] == sim_digest
        assert bs["latent_kind"] == "box"
        hashes.add(bs["split_hash"])
        for tid, rec in H.ledger(cfg, q).items():
            assert rec["status"] == "ok", (q["name"], tid, rec.get("error"))
            assert rec["split_hash"] == bs["split_hash"], (q["name"], tid)
            assert rec["contract_digest"] == sim_digest
    assert len(hashes) == 1, "the four searches see one arm S and one split seed"


def test_trial_argv_carries_the_configuration(r1):
    """C-O7/C-O10: every trial job's argv (the tuner's evaluate) and the
    runner argv in its record carry the configuration's values: the epochs,
    passes, split fractions, the flow's layers and bins, the buffer, the
    threads, `--patience 99`, `--resume` (D-111 (e)), the training seed, the
    simulated glob and q.real's glob (D-122, S-A1 included)."""
    cfg = r1["cfg"]
    sim_glob = os.path.join(H.bank_dir(cfg, "sim"), "shard_*.npz")
    for q in H.SEARCHES:
        real_glob = os.path.join(H.bank_dir(cfg, q["real"]), "shard_*.npz")
        camp = H.campaign_dir(cfg, q)
        argvs = sorted(glob.glob(os.path.join(camp, "jobs", "*.0.argv")))
        assert len(argvs) == TRIALS
        for p in argvs:
            a = H.read_argv(p)
            assert a[2] == "evaluate"
            assert _flag(a, "--campaign") == q["campaign"]
            assert _flag(a, "--epochs") == "2" and float(_flag(a, "--passes")) == 1.0
            assert [float(x) for x in _flags_n(a, "--split-fracs", 3)] == list(FR)
            assert _flag(a, "--flow-hidden-layers") == "2" and _flag(a, "--buffer-shards") == "4"
            assert _flag(a, "--threads") == "1" and _flag(a, "--patience") == "99"
            assert "--resume" in a
            assert _flag(a, "--seed") == "0"
            assert _flag(a, "--sim-shards") == sim_glob and _flag(a, "--real-shards") == real_glob
        for tid, rec in H.ledger(cfg, q).items():
            ra = [str(x) for x in rec["argv"]]
            assert _flag(ra, "--num-bins") == "10"
            assert _flag(ra, "--flow-hidden-layers") == "2"
            assert float(_flag(ra, "--passes")) == 1.0 and _flag(ra, "--epochs") == "2"
            assert [float(x) for x in _flags_n(ra, "--split-fracs", 3)] == list(FR)
            assert _flag(ra, "--buffer-shards") == "4" and _flag(ra, "--threads") == "1"
            assert _flag(ra, "--patience") == "99" and "--resume" in ra
            assert _flag(ra, "--seed") == "0"
            assert _flag(ra, "--split-seed") in (None, "0")
            assert _flag(ra, "--real-shards") == real_glob
            assert _flag(ra, "--sim-shards") == sim_glob


def test_gp_seed_per_round(r1):
    """C-O7/C-O10: round r proposes with --gp-seed gp_seed + r (gp_seed 3):
    the tuner's propose calls in the search's login-node log, in order."""
    cfg = r1["cfg"]
    for q in H.SEARCHES:
        with open(os.path.join(H.campaign_dir(cfg, q), "jobs", "login_node.log")) as fh:
            text = fh.read()
        calls = [l for l in text.splitlines() if l.startswith("+ ") and " propose " in l]
        seeds = [l.split("--gp-seed")[1].split()[0] for l in calls]
        npts = [l.split("--n-points")[1].split()[0] for l in calls]
        assert seeds == ["3", "4"], (q["name"], seeds)
        assert npts == ["2", "2"], (q["name"], npts)
        rf = H.read_json(os.path.join(H.campaign_dir(cfg, q), "pipeline_rounds.json"))
        assert [len(r["trial_ids"]) for r in rf["rounds"]] == [2, 2]


def test_trial_records_and_runner_records(r1):
    """C-O7: the ledger of each search holds its rounds' trials, every one ok
    with a finite nll and a finite L_pseudo_real; the arms the campaign
    allows (S-A1 all A1; S-A2-only all A2; S-A5-only all A5); the runner's
    record: the box prior `box_unit_box`, L0 0, the ledger's nll its L,
    L_pseudo_real the mean of its per-row pseudo-real NLLs over the cohort's
    24 windows; steps per epoch ceil(P / N_ep * n_train / B) (sec. 2.3)."""
    cfg = r1["cfg"]
    allowed = {"S-A1": {"A1"}, "S-A2-only": {"A2"}, "S-A5-only": {"A5"}, "S-A2": {"A1", "A2"}}
    for q in H.SEARCHES:
        rf = H.read_json(os.path.join(H.campaign_dir(cfg, q), "pipeline_rounds.json"))
        ids = [t for r in rf["rounds"] for t in r["trial_ids"]]
        led = H.ledger(cfg, q)
        assert sorted(led) == sorted(ids)
        for tid, rec in led.items():
            assert rec["status"] == "ok"
            assert math.isfinite(rec["nll"]) and math.isfinite(rec["L_pseudo_real"])
            assert _arm(rec["config"]) == rec["arm"], (tid, rec["arm"])
            assert rec["arm"] in allowed[q["campaign"]], (q["name"], rec["arm"])
            run = H.read_json(rec["runner_record"])
            assert run["L"] == rec["nll"]
            assert run.get("prior") == "box_unit_box"
            assert run.get("L0") == 0.0
            stem = rec["runner_record"][:-5]
            with np.load(stem + "_perrow.npz", allow_pickle=False) as z:
                pr = dict((k, z[k]) for k in z.files)
            assert pr["nll"].shape == (N_REP,)
            assert abs(float(np.mean(pr["nll"])) - run["L"]) <= 1e-9 * max(1.0, abs(run["L"]))
            assert pr["nll_pseudo_real"].shape == (ROWS_R,)
            assert abs(float(np.mean(pr["nll_pseudo_real"])) - rec["L_pseudo_real"]) <= \
                1e-9 * max(1.0, abs(rec["L_pseudo_real"]))
            b = int(rec["config"]["batch_size_npe"])
            assert b in BATCHES
            s_exp = int(math.ceil(1.0 / 2 * N_TRAIN / b))
            assert run["steps_per_epoch_effective"] == s_exp, (tid, b, run["steps_per_epoch_effective"])


def test_runner_logs_of_the_terms(r1):
    """C-O7 (block 14 along the chain): an A2 trial counts 3 metric classes
    on the real arm; an A5 trial trains on the cohort's cross-well pairs:
    6 donors x (2 windows x 2 windows) = 24."""
    cfg = r1["cfg"]
    seen = {"A2": 0, "A5": 0}
    for q in H.SEARCHES:
        for tid, rec in H.ledger(cfg, q).items():
            logs = sorted(glob.glob(os.path.join(H.campaign_dir(cfg, q), "jobs", tid + ".*.log")))
            text = "".join(open(p, errors="replace").read() for p in logs)
            if rec["arm"] == "A2":
                assert re.search(r"metric classes\s*:\s*3 \(on the real arm\)", text), tid
                seen["A2"] += 1
            if rec["arm"] == "A5":
                m = re.search(r"replicate pairs\s*:\s*([0-9]+)\s+\(two wells of one donor\)", text)
                assert m and int(m.group(1)) == C * DPC * (R_WIN * R_WIN), (tid, m and m.group(0))
                seen["A5"] += 1
    assert seen["A2"] >= TRIALS and seen["A5"] >= TRIALS


def test_search_complete_ranks_on_nll(r1):
    """C-O7 (D-121): SEARCH_COMPLETE.json -- ended by its rounds, 2 rounds, 4
    ok; its finalist the ledger's lowest nll over the ok trials (computed
    here), never chosen on L_pseudo_real; its identity the bank_split's."""
    cfg = r1["cfg"]
    for q in H.SEARCHES:
        sc = H.read_json(os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json"))
        led = H.ledger(cfg, q)
        best = min((rec["nll"], tid) for tid, rec in led.items() if rec["status"] == "ok")
        assert sc["ended_by"] == "rounds" and sc["rounds_evaluated"] == 2
        assert sc["n_trials"]["ok"] == TRIALS
        assert sc["finalist"]["trial_id"] == best[1]
        assert sc["finalist"]["nll"] == best[0]
        bs = H.read_json(os.path.join(H.campaign_dir(cfg, q), "bank_split.json"))
        assert sc["identity"]["split_hash"] == bs["split_hash"]


# ---------------------------------------------------------------- C-O8 ---

def _members(cfg, q, alt=False):
    out = {}
    root = H.best_dir(cfg, q, alt)
    for k in MEMBERS:
        recs = sorted(glob.glob(os.path.join(root, "member_seed%d" % k, "*_seed%d.json" % k)))
        assert len(recs) == 1, recs
        out[k] = recs[0]
    return out


@pytest.mark.parametrize("alt", [False, True])
def test_best_fit_ensemble(r1, alt):
    """C-O8: the finalist trained once per member seed on split seed 1 (one
    split across the members), the stems of block 6, the best fit's budget in
    the members' argv; L_bar recomputed from the members' per-row NLLs with
    scipy's logsumexp, the rows identical and in one order across the
    members; L_bar <= the members' mean (Jensen)."""
    cfg = r1["cfg"]
    for q in H.SEARCHES:
        if alt and not q["alt_best_fit"]:
            continue
        root = H.best_dir(cfg, q, alt)
        bfc = H.read_json(os.path.join(root, "BEST_FIT_COMPLETE.json"))
        fc = H.read_json(os.path.join(root, "finalist_config.json"))
        recs = _members(cfg, q, alt)
        nll, theta0, group0, hashes, Ls = [], None, None, set(), []
        for k, rp in sorted(recs.items()):
            stem = os.path.basename(rp)[:-5]
            arm = fc.get("arm") or H.read_json(os.path.join(H.campaign_dir(cfg, q), "trials",
                                                            fc["trial_id"] + ".json"))["arm"]
            exp_stem = ("%s_seed%d" % (arm, k) if k == SPLIT_SEED_BF
                        else "%s_split%d_seed%d" % (arm, SPLIT_SEED_BF, k))
            assert stem == exp_stem, (stem, exp_stem)
            run = H.read_json(rp)
            hashes.add(run["split_hash"])
            Ls.append(run["L"])
            with np.load(rp[:-5] + "_perrow.npz", allow_pickle=False) as z:
                nll.append(np.asarray(z["nll"], dtype=np.float64))
                th, gr = np.asarray(z["theta"]), np.asarray(z["group"])
            if theta0 is None:
                theta0, group0 = th, gr
            else:
                assert np.array_equal(th, theta0) and np.array_equal(gr, group0), \
                    "the members' report rows differ"
            # block 7's contract: the finalist's lr / weight decay / pooling / fusion /
            # batch size, patience 10 and n_posterior_draws 16 in the member's run
            rc_ = run["config"]
            fcfg = fc["config"]
            assert rc_["lr"] == fcfg["lr"] and rc_["weight_decay"] == fcfg["weight_decay"]
            assert rc_["head_pool_ops"] == fcfg["head_pool_ops"]
            assert rc_["head_fusion"] == fcfg["head_fusion"]
            assert rc_["b_sim"] == fcfg["batch_size_npe"]
            assert rc_["patience"] == 10 and rc_["n_posterior_draws"] == 16
            assert run["split_seed"] == SPLIT_SEED_BF and run["seed"] == k
            assert run["steps_per_epoch_effective"] == int(math.ceil(1.0 / 2 * N_TRAIN
                                                                     / fcfg["batch_size_npe"]))
            argv = H.read_argv(os.path.join(root, "jobs", "member_seed%d.0.argv" % k))
            assert _flag(argv, "--seed") == str(k)
            assert _flag(argv, "--split-seed") == str(SPLIT_SEED_BF)
            assert float(_flag(argv, "--passes")) == 1.0 and _flag(argv, "--epochs") == "2"
            assert _flag(argv, "--patience") == "10"
            assert _flag(argv, "--n-posterior-draws") == "16"
            assert _flag(argv, "--threads") == "1" and _flag(argv, "--buffer-shards") == "4"
            assert "--resume" in argv
            assert _flag(argv, "--real-shards") == os.path.join(H.bank_dir(cfg, q["real"]),
                                                                "shard_*.npz")
        assert len(hashes) == 1, "the members do not share one split"
        sc = H.read_json(os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json"))
        assert hashes.pop() != sc["identity"]["split_hash"], "split seed 1 gives the search's split"
        L = np.vstack(nll)                                  # (M, n)
        M, n = L.shape
        assert (M, n) == (len(MEMBERS), N_REP)
        lbar = float(-np.mean(logsumexp(-L, axis=0) - np.log(M)))
        ens = bfc["ensemble"]
        assert ens["M"] == M and ens["n_report"] == n
        assert abs(ens["L_bar"] - lbar) <= 1e-12 * max(1.0, abs(lbar)), (ens["L_bar"], lbar)
        assert ens["L_bar"] <= float(np.mean(Ls)) + 1e-12
        assert [float(x) for x in ens["L_members"]] == [float(x) for x in Ls]
        assert bfc["split_seed"] == SPLIT_SEED_BF and bfc["member_seeds"] == list(MEMBERS)


# ---------------------------------------------------------------- C-O9 ---

def test_other_arm_best_fit(r1):
    """C-O9 (block 18, sec. 2.5): for the mixed search, the arm of the
    search's finalist; the other arm's finalist = the lowest nll among the
    ok trials of that arm (computed here); finalist_config.json naming it,
    with `arm`, `search_finalist` and `finalists_files`; its members `ba_*`
    of that arm, with the main best fit's split seed, seeds and budget."""
    cfg = r1["cfg"]
    q = _q("S-A2m-R3")
    led = H.ledger(cfg, q)
    arms = dict((tid, _arm(rec["config"])) for tid, rec in led.items() if rec["status"] == "ok")
    assert set(arms.values()) == {"A1", "A2"}, "the reduced search drew one arm only: " + str(arms)
    sc = H.read_json(os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json"))
    own = arms[sc["finalist"]["trial_id"]]
    other = ({"A1", "A2"} - {own}).pop()
    exp = min((led[t]["nll"], t) for t, a in arms.items() if a == other)[1]
    alt = H.best_dir(cfg, q, alt=True)
    fc = H.read_json(os.path.join(alt, "finalist_config.json"))
    assert fc["trial_id"] == exp and fc["arm"] == other
    assert fc["search_finalist"] == {"trial_id": sc["finalist"]["trial_id"], "arm": own}
    assert sorted(fc["finalists_files"]) == ["A1", "A2"]
    for k, rp in _members(cfg, q, alt=True).items():
        assert H.read_json(rp)["arm"] == other
    main = H.read_json(os.path.join(H.best_dir(cfg, q), "BEST_FIT_COMPLETE.json"))
    alt_c = H.read_json(os.path.join(alt, "BEST_FIT_COMPLETE.json"))
    for key in ("split_seed", "member_seeds", "passes", "epochs", "patience_epochs",
                "n_posterior_draws", "split_fracs"):
        assert main[key] == alt_c[key], key
    names = sorted(j["name"] for j in r1["jobs"] if j["name"].startswith("ba_"))
    assert names == ["ba_synth_s1", "ba_synth_s2"]


def test_rows_align_with_the_banks(r1):
    """Step 7, the banks -> the runner: every trial's and member's per-row
    file names bank rows that exist. The report rows: each (group, theta)
    pair is a donor of arm S with that donor's theta (rows of a donor share
    it, block 11), 2 donors x 2 wells; the pseudo-real rows: the cohort's
    24 windows, their donors (`group_pseudo_real`) the real bank's `donor`
    column in bank order (sec. 2.4: L_pseudo_real over the n_R windows of
    q.real, each at its donor's withheld theta)."""
    cfg = r1["cfg"]
    sim = {}
    for p in sorted(glob.glob(os.path.join(H.bank_dir(cfg, "sim"), "shard_*.npz"))):
        with np.load(p, allow_pickle=False) as z:
            for d, th in zip(z["donor"], z["theta"]):
                sim.setdefault(int(d), th)
    real = {}
    for b in ("R0", "R3"):
        with np.load(os.path.join(H.bank_dir(cfg, b), "shard_0000.npz"), allow_pickle=False) as z:
            real[b] = np.asarray(z["donor"]).astype(np.int64)
    files = []
    for q in H.SEARCHES:
        for rec in H.ledger(cfg, q).values():
            files.append((q, rec["runner_record"][:-5] + "_perrow.npz"))
        for alt in ((False, True) if q["alt_best_fit"] else (False,)):
            for p in glob.glob(os.path.join(H.best_dir(cfg, q, alt), "member_seed*", "*_perrow.npz")):
                files.append((q, p))
    assert len(files) == len(H.SEARCHES) * TRIALS + 2 * (len(H.SEARCHES) + 1)
    for q, p in files:
        with np.load(p, allow_pickle=False) as z:
            g, th = np.asarray(z["group"]).astype(np.int64), np.asarray(z["theta"])
            gpr = np.asarray(z["group_pseudo_real"]).astype(np.int64)
        assert len(np.unique(g)) == D_REP and len(g) == N_REP
        for d, t in zip(g, th):
            assert int(d) in sim and np.array_equal(sim[int(d)], t), (p, int(d))
        assert np.array_equal(gpr, real[q["real"]]), (p, "pseudo-real rows not in bank order")
