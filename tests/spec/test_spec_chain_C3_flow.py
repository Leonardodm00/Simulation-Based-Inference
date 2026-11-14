"""Chain C3 (the evaluation), run t2-C3 of D-131: the stages' order and
dependencies, the done predicates, the resume rule and the entry point's exit
codes, along `submit_eval.sh` from the best fits to REPORT_COMPLETE.json and
the data overview. Oracles: SPEC.md block 22's Rule and Public API, block 23's
Public API, sec. 2.5, sec. 3.3 (the fake PBS). The fixture
(tests/spec/chain_c3_support.py) runs the chain through its real entry points;
these tests read what it recorded and wrote.

    python3 -m pytest tests/spec/test_spec_chain_C3_flow.py -q
    C3_FIXTURE=<dir> python3 -m pytest tests/spec/test_spec_chain_C3_flow.py -q

Pure ASCII, LF only.
"""

import glob
import json
import os
import re

from chain_c3_support import fixture, sha256_file


# ------------------------------------------------------------------ helpers

def _cfg(F):
    with open(F["cfg"]) as fh:
        return json.load(fh)


def _ecfg(path):
    with open(path) as fh:
        return json.load(fh)


def _expected_ensembles(F):
    """[(name, stem, kind, search index)] from SPEC block 19's rule, read
    from the producer's own files (the members' records' `arm`), in the
    configuration's order, best fit before the other arm's."""
    cfg = _cfg(F)
    out = []
    for i, q in enumerate(cfg["synthetic"]["searches"]):
        kinds = [("best_fit", "_best")] + ([("best_fit_alt", "_best_alt")] if q["alt_best_fit"] else [])
        for kind, suffix in kinds:
            folder = os.path.join(cfg["paths"]["runs_root"], q["name"], cfg["dataset"] + suffix)
            with open(os.path.join(folder, "BEST_FIT_COMPLETE.json")) as fh:
                bf = json.load(fh)
            arms = set()
            for s in bf["member_seeds"]:
                recs = glob.glob(os.path.join(folder, "member_seed%d" % s, "*_seed%d.json" % s))
                assert len(recs) == 1, recs
                with open(recs[0]) as fh:
                    arms.add(json.load(fh)["arm"])
            assert len(arms) == 1, (folder, arms)
            arm = arms.pop()
            out.append(("%s:%s" % (q["name"], arm), "%s_%s" % (q["name"], arm), kind, i))
    return out


def _by_name(jobs):
    return dict((j["name"], j) for j in jobs)


def _summary(out):
    m = re.search(r"EVAL (SUBMIT OK|DRY RUN): (\d+) job\(s\) (?:would be )?submitted, (\d+) done, "
                  r"(\d+) live", out)
    assert m, out[-2000:]
    return int(m.group(2)), int(m.group(3)), int(m.group(4))


def _log_text(job):
    p = job.get("out")
    if not p or not os.path.isfile(p):
        return None
    with open(p, errors="replace") as fh:
        return fh.read()


# ------------------------------------------------------------------ producer

def test_producer_reached_pipeline_done():
    """Precondition: the chain's inputs were written by their producers
    (blocks 7, 16, 18), the reduced project to PIPELINE DONE."""
    F = fixture()
    p = F["rec"]["producer"]
    assert p["rc"] == 0 and "PIPELINE DONE" in p["tail"], p["tail"][-3000:]


def test_the_expected_ensembles_exist():
    """Block 19: one ensemble per best-fit folder; the term-only searches'
    members are of their term arm (block 26), the mixed search has both arms
    (block 18)."""
    F = fixture()
    ens = _expected_ensembles(F)
    names = [e[0] for e in ens]
    assert names[0] == "S-A1:A1" and names[1] == "S-A2-R0:A2" and names[2] == "S-A5-R3:A5", names
    assert sorted(names[3:]) == ["S-A2x-R0:A1", "S-A2x-R0:A2"], names


# ------------------------------------------------------------------ run A

def test_A_one_submission_plans_every_unit():
    """Block 22: calbank shards + one sampling per complete ensemble + the
    report + (named) data, all submitted, nothing done or live; exit 0."""
    F = fixture()
    A = F["rec"]["A"]
    s = A["submissions"][0]
    e = _ecfg(A["ecfg"])
    n_ens = len(_expected_ensembles(F))
    assert s["rc"] == 0, s["out"][-3000:]
    assert _summary(s["out"]) == (e["cal_shards"] + n_ens + 1 + 1, 0, 0), s["out"][-2000:]
    assert "not complete:" not in s["out"]
    assert "REFUSE" not in s["out"] and "ABORT" not in s["out"]


def test_A_job_names_dependencies_and_exits():
    """Block 22: job names (<= 15 characters) ev_cal<k>, ev_s<i><b|a>,
    ev_report, ev_data; every sampling afterok on every calibration job; the
    report afterok on every sampling; the data job after nothing; every job
    exit 0; ROADMAP_STAGE I19 / I22 / I23 (block 22, block 23)."""
    F = fixture()
    A = F["rec"]["A"]
    e = _ecfg(A["ecfg"])
    jobs = A["jobs"]
    by = _by_name(jobs)
    want = ["ev_cal%d" % k for k in range(e["cal_shards"])]
    want += ["ev_s%d%s" % (i, "a" if kind == "best_fit_alt" else "b")
             for _n, _s, kind, i in _expected_ensembles(F)]
    want += ["ev_report", "ev_data"]
    assert sorted(by) == sorted(want) and len(jobs) == len(want), [j["name"] for j in jobs]
    assert all(len(n) <= 15 for n in by), sorted(by)
    cal_ids = set(by["ev_cal%d" % k]["id"] for k in range(e["cal_shards"]))
    smp = [j for j in jobs if j["name"].startswith("ev_s")]
    for j in jobs:
        assert j["exit_status"] == 0, (j["name"], j.get("comment"), (_log_text(j) or "")[-2000:])
        stage = {"ev_report": "I22", "ev_data": "I23"}.get(j["name"], "I19")
        assert j["vars"].get("ROADMAP_STAGE") == stage, (j["name"], j["vars"])
        assert j["vars"].get("ENV_NAME") == _cfg(F)["envs"]["train"], j["vars"]
    for j in jobs:
        dep = set((j["depend"] or "").split(":")) - {""}
        if j["name"].startswith("ev_cal") or j["name"] == "ev_data":
            assert dep == set(), (j["name"], dep)
        elif j["name"].startswith("ev_s"):
            assert dep == cal_ids, (j["name"], dep, cal_ids)
        else:
            assert dep == set(x["id"] for x in smp), (dep, [x["id"] for x in smp])


def test_A_jobs_ran_in_dependency_order():
    """The fake PBS's start and end times: every sampling started after
    every calibration job ended; the report after every sampling."""
    F = fixture()
    jobs = F["rec"]["A"]["jobs"]
    cal_end = max(j["finished"] for j in jobs if j["name"].startswith("ev_cal"))
    smp = [j for j in jobs if j["name"].startswith("ev_s")]
    assert all(j["started"] >= cal_end for j in smp), [(j["name"], j["started"], cal_end) for j in smp]
    rep = _by_name(jobs)["ev_report"]
    assert rep["started"] >= max(j["finished"] for j in smp)


def test_A_job_logs_pass_lines_and_last_line():
    """Each job's log has its command's pass line and ends with `EVAL JOB
    EXIT: 0` (block 22; the markers are off here, cf. F303)."""
    F = fixture()
    A = F["rec"]["A"]
    cfg = _cfg(F)
    b = cfg["synthetic"]["builder"]
    n_R = dict((r["name"], b["n_classes"] * r["donors_per_class"] * b["wells_per_donor"]
                * r["n_windows"]) for r in cfg["synthetic"]["real"])
    e = _ecfg(A["ecfg"])
    n_S = e["data_view"]["sim_shards"] * cfg["synthetic"]["sim"]["traces_per_shard"] \
        * cfg["synthetic"]["sim"]["n_windows"]
    stems = dict(("ev_s%d%s" % (i, "a" if kind == "best_fit_alt" else "b"), stem)
                 for _n, stem, kind, i in _expected_ensembles(F))
    for j in A["jobs"]:
        # the log as it was when run A ended (a later job of the same unit -- W2's
        # resampling -- writes the same log path)
        txt = A["logs"].get(str(j["num"]))
        assert txt, j["name"]
        assert txt.rstrip().splitlines()[-1] == "EVAL JOB EXIT: 0", (j["name"], txt[-500:])
        if j["name"].startswith("ev_cal"):
            assert "rows written     : %d" % (e["cal_n"] // e["cal_shards"]) in txt, txt[-1500:]
        elif j["name"] in stems:
            assert "EVAL SAMPLE OK: %s " % stems[j["name"]] in txt, txt[-1500:]
        elif j["name"] == "ev_report":
            n = len(stems)
            for line in ("EVAL METRICS OK: %d ensemble(s)" % n, "EVAL FIGURES OK: ",
                         "EVAL REPORT OK: %d ensemble(s)" % n):
                assert line in txt, (line, txt[-2000:])
        else:
            assert ("EVAL DATA OK: %d simulated windows, %d + %d cohort windows"
                    % (n_S, n_R["R0"], n_R["R3"])) in txt, txt[-1500:]


def test_A_argv_files_and_ledger():
    """Block 22: every job's command in <out_root>/jobs/<unit>.argv, one
    argument per line, its log in <out_root>/jobs/logs/, recorded in the
    ledger under its unit with the job id the scheduler gave."""
    F = fixture()
    A = F["rec"]["A"]
    out = A["out_root"]
    led = A["ledger"]
    by_id = dict((j["id"], j) for j in A["jobs"])
    units = sorted(led)
    n_ens = len(_expected_ensembles(F))
    assert len(units) == 2 + n_ens + 2, units
    for unit, r in led.items():
        safe = unit.replace(":", "_")
        assert r["argv_file"] == os.path.join(out, "jobs", "%s.argv" % safe), r
        assert r["log"] == os.path.join(out, "jobs", "logs", "%s.log" % safe), r
        j = by_id[r["job_id"]]
        assert j["out"] == r["log"] and j["vars"]["ARGV_FILE"] == r["argv_file"], (j, r)
        with open(r["argv_file"]) as fh:
            argv = fh.read().splitlines()
        assert argv[0] == "python3" and os.path.isfile(argv[1]), argv[:3]
        assert os.path.commonpath([argv[1], _cfg(F)["paths"]["sbi_dir"]]) == _cfg(F)["paths"]["sbi_dir"]


def test_A_report_complete_records_its_inputs():
    """Block 22: REPORT_COMPLETE.json records the samples' provenance
    digests (every ensemble's <stem>_sample.json) and REPORT_PARAMS; as many
    figures as FIGURES.md lists (one file per figure and format)."""
    F = fixture()
    A = F["rec"]["A"]
    e = _ecfg(A["ecfg"])
    out = A["out_root"]
    with open(os.path.join(out, "REPORT_COMPLETE.json")) as fh:
        rc = json.load(fh)
    stems = [s for _n, s, _k, _i in _expected_ensembles(F)]
    want_samples = dict((s, sha256_file(os.path.join(out, "samples", "%s_sample.json" % s)))
                        for s in stems)
    assert rc["inputs"]["samples"] == want_samples
    params = dict((k, e[k]) for k in ("seed", "lc2st_null", "lc2st_eval_draws", "bootstrap_n",
                                      "level", "dpi", "formats"))
    assert rc["inputs"]["params"] == params, rc["inputs"]["params"]
    assert sorted(rc["ensembles"]) == sorted(n for n, _s, _k, _i in _expected_ensembles(F))
    with open(os.path.join(out, "figures", "FIGURES.md")) as fh:
        listed = [l for l in fh if l.startswith("- `")]
    files = [p for p in glob.glob(os.path.join(out, "figures", "**", "*"), recursive=True)
             if os.path.isfile(p) and not p.endswith("FIGURES.md")]
    assert rc["n_figures"] == len(listed) == len(files) // len(e["formats"]) > 0, \
        (rc["n_figures"], len(listed), len(files))


def test_A_data_view_records_its_inputs():
    """Block 23: DATA_VIEW.json records DATA_PARAMS and, per bank, each shard
    read with the sha256 of its sidecar (the first sim_shards of the
    simulated bank, sorted; every shard of each cohort)."""
    F = fixture()
    A = F["rec"]["A"]
    e = _ecfg(A["ecfg"])
    cfg = _cfg(F)
    with open(os.path.join(A["out_root"], "data", "DATA_VIEW.json")) as fh:
        dv = json.load(fh)
    assert dv["schema"] == "eval_data_view/1"
    banks = {}
    sim = sorted(glob.glob(os.path.join(cfg["paths"]["banks_root"], "%s_sim" % cfg["dataset"],
                                        "shard_*.npz")))[:e["data_view"]["sim_shards"]]
    banks["sim"] = [[os.path.basename(p), sha256_file(p[:-4] + ".json")] for p in sim]
    for r in cfg["synthetic"]["real"]:
        ps = sorted(glob.glob(os.path.join(cfg["paths"]["banks_root"],
                                           "%s_real_%s" % (cfg["dataset"], r["name"]), "shard_*.npz")))
        banks[r["name"]] = [[os.path.basename(p), sha256_file(p[:-4] + ".json")] for p in ps]
    assert dv["inputs"]["banks"] == banks
    assert dv["inputs"]["params"] == dict((k, e[k]) for k in ("seed", "dpi", "formats", "data_view"))


def test_A_second_submission_does_nothing():
    """Block 22's resume rule: the same command again finds every unit done
    and submits nothing."""
    F = fixture()
    A = F["rec"]["A"]
    s = A["submissions"][1]
    n = len(A["jobs"])
    assert s["rc"] == 0 and _summary(s["out"]) == (0, n, 0), s["out"][-2000:]
    assert A["n_jobs_after_rerun"] == n


def test_pipeline_state_never_touched():
    """Block 22: nothing of the pipeline's state is read or written -- the
    work root's files are byte-identical before run A and after every
    scenario."""
    F = fixture()
    rec = F["rec"]
    assert rec["work_root_before"] == rec["work_root_after_A"]
    assert rec["work_root_before"] == rec["work_root_after_all"]


def test_producer_artefacts_never_written():
    """The evaluation reads the banks, the searches and the best fits and
    writes only under its out_root (block 22: `<out_root>/...`; block 19:
    `<out_root>/samples/`): no regular file under banks_root, tune_root or
    runs_root is newer than the producer's run (5 s of margin for the file
    system's clock), after run A, run B and every scenario (each put back
    what it moved)."""
    F = fixture()
    rec = F["rec"]
    cfg = _cfg(F)
    end = rec["started"] + rec["producer"]["seconds"] + 5.0
    newer = []
    for key in ("banks_root", "tune_root", "runs_root"):
        for a, _b, fs in os.walk(cfg["paths"][key]):
            for f in fs:
                p = os.path.join(a, f)
                if os.path.isfile(p) and os.path.getmtime(p) > end:
                    newer.append(os.path.relpath(p, F["root"]))
    assert not newer, newer[:10]


def test_dropped_report_leaves_the_previous_log():
    """README E8 (block 22's run sheet) checks a report by grepping
    <out_root>/jobs/logs/report.log for EVAL REPORT OK. When the report is
    dropped (its sampling refused, W2), the log is not rewritten: record
    whether the log then still shows the previous report's pass line."""
    F = fixture()
    W = F["rec"]["W2"]
    rep = [j for j in W["jobs"] if j["name"] == "ev_report"][0]
    assert rep["exit_status"] != 0
    log = os.path.join(F["rec"]["A"]["out_root"], "jobs", "logs", "report.log")
    with open(log) as fh:
        txt = fh.read()
    stale = "EVAL REPORT OK" in txt and os.path.getmtime(log) < rep["submitted"]
    assert not stale, ("the report job %s was dropped (exit %s: %s), yet %s still ends with the "
                       "previous report's `EVAL REPORT OK` (written before the drop)"
                       % (rep["id"], rep["exit_status"], rep["comment"], log))


# ------------------------------------------------------------------ run B (interrupted)

def _steps(F):
    B = F["rec"]["B"]
    return dict((s["name"], s) for s in B["steps"])


def test_B_calibration_bank_alone():
    F = fixture()
    s = _steps(F)["B1_calbank"]
    e = _ecfg(F["rec"]["B"]["ecfg"])
    assert s["rc"] == 0 and _summary(s["out"]) == (e["cal_shards"], 0, 0), s["out"][-2000:]
    assert [j["name"] for j in s["new_jobs"]] == ["ev_cal%d" % k for k in range(e["cal_shards"])]


def test_B_samplings_after_a_done_bank_have_no_dependency():
    """The calibration bank done: the samplings are submitted at once (no
    afterok), the bank counted done."""
    F = fixture()
    s = _steps(F)["B2_samples_queued"]
    e = _ecfg(F["rec"]["B"]["ecfg"])
    n_ens = len(_expected_ensembles(F))
    assert s["rc"] == 0 and _summary(s["out"]) == (n_ens, e["cal_shards"], 0), s["out"][-2000:]
    assert all(not j["depend"] for j in s["new_jobs"]), [j["depend"] for j in s["new_jobs"]]
    assert s["qdel"]["rc"] == 0, s["qdel"]


def test_B_report_waits_for_live_and_new_samplings():
    """Block 22: with four samplings live and one deleted, the plan
    resubmits the deleted one and puts the report after it AND after the
    live ones (`-W depend=afterok:<live ids>:<new>`)."""
    F = fixture()
    st = _steps(F)
    s = st["B3_report_while_live"]
    e = _ecfg(F["rec"]["B"]["ecfg"])
    n_ens = len(_expected_ensembles(F))
    assert s["rc"] == 0, s["out"][-2000:]
    assert _summary(s["out"]) == (2, e["cal_shards"], n_ens - 1), s["out"][-2000:]
    deleted = st["B2_samples_queued"]["qdel_unit"]
    assert "%s would be submitted" % deleted in s["out"], s["out"][-3000:]
    live = [j["id"] for j in st["B2_samples_queued"]["new_jobs"]
            if j["id"] != st["B2_samples_queued"]["qdel"]["job"]]
    rep = [l for l in s["out"].splitlines() if l.strip().startswith("qsub -N ev_report")]
    assert len(rep) == 1, s["out"][-3000:]
    dep = re.search(r"depend=afterok:(\S+)", rep[0]).group(1).split(":")
    assert set(live) <= set(dep) and "<%s>" % deleted.replace(":", "_") in dep, (dep, live)


def test_B_rest_resubmits_only_the_deleted_sampling_and_the_report():
    F = fixture()
    st = _steps(F)
    s = st["B4_rest_and_report"]
    e = _ecfg(F["rec"]["B"]["ecfg"])
    n_ens = len(_expected_ensembles(F))
    assert s["rc"] == 0 and _summary(s["out"]) == (2, e["cal_shards"] + n_ens - 1, 0), \
        s["out"][-2000:]
    final = dict((j["num"], j) for j in F["rec"]["B"]["jobs"])     # the records at the end
    new = [final[j["num"]] for j in s["new_jobs"]]
    names = sorted(j["name"] for j in new)
    assert len(names) == 2 and "ev_report" in names, names
    smp = [j for j in new if j["name"] != "ev_report"][0]
    rep = [j for j in new if j["name"] == "ev_report"][0]
    assert rep["depend"] == smp["id"] and smp["exit_status"] == 0 and rep["exit_status"] == 0, \
        (rep["depend"], smp["id"], smp["exit_status"], rep["exit_status"])


def test_B_data_then_everything_done():
    F = fixture()
    st = _steps(F)
    s = st["B7_data"]
    assert s["rc"] == 0 and _summary(s["out"]) == (1, 0, 0), s["out"][-2000:]
    final = dict((j["num"], j) for j in F["rec"]["B"]["jobs"])
    assert [(final[j["num"]]["name"], final[j["num"]]["exit_status"]) for j in s["new_jobs"]] \
        == [("ev_data", 0)]
    s = st["B8_all_again"]
    n_ens = len(_expected_ensembles(F))
    e = _ecfg(F["rec"]["B"]["ecfg"])
    assert s["rc"] == 0 and _summary(s["out"]) == (0, e["cal_shards"] + n_ens + 2, 0), s["out"][-2000:]


def test_B_killed_resampling_leaves_a_resumable_state():
    """A resampling killed while it runs (qdel of a running job, exit 271):
    the next plan must not fail; by block 22's done predicate the ensemble
    is done when its <stem>_sample.json lists files that all exist with the
    current values and best fit -- recorded here, with which files the
    killed job had rewritten (cf. F301)."""
    F = fixture()
    st = _steps(F)
    s = st["B5_resample_killed"]
    assert s["rc"] == 0, s["out"][-2000:]
    plan = st["B6_plan_after_kill"]
    assert plan["rc"] == 0, plan["out"][-2000:]
    assert "REFUSE" not in plan["out"]


# ------------------------------------------------------------------ the entry point's refusals

def test_entry_point_refusals_exit_codes():
    """Block 22 Public API: 2 on the arguments or the configurations, 3 on a
    refusal with nothing submitted; run on the fixture's configuration."""
    import subprocess
    from chain_c3_support import SUBMIT, fake_env, REPO
    F = fixture()
    A = F["rec"]["A"]
    env = fake_env(os.path.join(F["root"], "spool_refusals"))
    before = sorted(os.listdir(os.path.join(A["out_root"], "jobs")))

    def run(*args):
        r = subprocess.run(["bash", SUBMIT, "--config", F["cfg"], "--eval-config", A["ecfg"]]
                           + list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, env=env, cwd=REPO, timeout=300)
        return r.returncode, r.stdout
    rc, out = run("--what", "calbank,nonsense")
    assert rc == 2 and "ABORT" in out, out
    rc, out = run("--resample", "S-A9:A1", "--what", "sample")
    assert rc == 3 and "REFUSE" in out, out
    assert sorted(os.listdir(os.path.join(A["out_root"], "jobs"))) == before
