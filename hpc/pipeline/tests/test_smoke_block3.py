#!/usr/bin/env python3
"""Smoke test for block 3 of specs/SPEC.md: the input stages S1-S3 on the
stand-in davinci under the fake PBS. S3 runs the REAL freeze tool
(`preflight_label_axes.py`) on the fixture's simulation tree; S1 and S2
run STUBS of the two launchers (tests/standin.py) that print the real
launchers' lines, submit the same job structure through qsub and write the
artefacts from the jobs -- the real launchers need the recordings and the
ANN tools, which exist on davinci alone (their own suites cover them).

Checks (specs/SPEC.md block 3, "Test oracles")
    test_known_answer   S3 on the fixture: the freeze's topology block is
                        ['conn_prob'], the three excluded axes carry the
                        configuration's reasons, the sidecar matches; the
                        argv carries every --exclude NAME=REASON
    test_reference      the Giulia dry run prints the three stages' lines
                        and no REFUSE line (the whole graph is built)
    test_convergence    skipped
    test_invariants     the launchers' variables all come from the
                        configuration (the stubs refuse an unset one);
                        S2 without the cohort's profile is refused before
                        anything is submitted
    test_contract       S1 (array + aggregation) and S3 run at once, S2
                        (plan, test, array + gate with RESUME=1) after S1;
                        the three artefacts with their sidecars; S4-S8
                        skipped (pre-made); 5 jobs in all; a re-run on a
                        fresh state skips everything
    test_determinism    a data set with one task: after the test task the
                        array has nothing and the gate runs alone
    test_edge_cases     the flags refusal of S1 (NEW COHORT ... COMMIT) stops
                        the run before any job; a refused aggregation and a
                        refused gate stop it with their REFUSED lines
    test_cores          D-099: S2's CONCURRENCY clamped by the cap (3 tasks of
                        3 cores in the 10 it leaves), its jobs' cores in the
                        state, the gate after the array; S1 refused when its
                        array (2 wells x 2 cores, no concurrency limit of its
                        own) exceeds the share; S1 recorded (4 cores, the
                        aggregation after it), then S2 refused at its start
                        when its gate (8 cores) can never fit

Run
    python3 hpc/pipeline/tests/test_smoke_block3.py [check ...]
        About 1 minute; names of checks run only those. Needs the Sbi-extractor checkout (SBIX_CLONE or a
        known place) and numpy.

Pure ASCII, LF only.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
_FAKE = os.path.join(_HERE, "fake_pbs")
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C          # noqa: E402
import standin as ST        # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
PYTHON = sys.executable


class Standin(object):
    """The stand-in with the data set's RAW inputs only: S1-S3's artefacts
    removed (the stages make them), S4-S8's pre-made."""

    def __init__(self, keep=(), pre_make_rest=True):
        self.root = tempfile.mkdtemp(prefix="pipeline_b3_")
        clone = ST.sbix_clone()
        if clone is None:
            raise unittest.SkipTest("no Sbi-extractor checkout (set SBIX_CLONE)")
        self.sbix = ST.build_sbix(self.root, clone)
        self.fakebin = ST.fake_conda(self.root)
        self.ds = ST.write_dataset(self.root, self.sbix)
        ST.stub_launchers(self.sbix)
        self.cfg = ST.synthetic_config(self.root, self.sbix, _REPO, self.ds)
        # the real tools' stand-ins: a cohort config file and a tools folder
        with open(self.cfg["real"]["cohort_config"], "w") as fh:
            fh.write('{"stub": true}\n')
        os.makedirs(self.cfg["sim"]["tools"], exist_ok=True)
        self.path = os.path.join(self.root, "synth.json")
        with open(self.path, "w") as fh:
            json.dump(self.cfg, fh, indent=1, sort_keys=True)
        # remove S1-S3's artefacts unless kept
        arts = {"real_extraction": os.path.join(self.ds["extract_root"], "cohort_manifest.json"),
                "sim_redetection": os.path.join(self.ds["out_root"], "REEXTRACTION_RECORD.json"),
                "label_freeze": os.path.join(self.sbix, self.ds["freeze_out"])}
        for name, p in arts.items():
            if name not in keep:
                for q in (p, p + ".sha256"):
                    if os.path.isfile(q):
                        os.remove(q)
        self.arts = arts
        if pre_make_rest:
            tag = "synth"
            pre = [os.path.join(self.sbix, "export_windows", tag, "bank_plan.json")]
            ST.premake_bank_record(os.path.join(self.cfg["paths"]["banks_root"], tag + "_sim"))
            for p in pre:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as fh:
                    fh.write('{"pre_made": true}\n')
                import hashlib
                h = hashlib.sha256(open(p, "rb").read()).hexdigest()
                with open(p + ".sha256", "w") as fh:
                    fh.write("%s  %s\n" % (h, os.path.basename(p)))
            rb = os.path.join(self.cfg["paths"]["banks_root"], tag + "_real")
            os.makedirs(rb, exist_ok=True)
            for f in ("shard_0000.npz", "shard_0000.json"):
                with open(os.path.join(rb, f), "w") as fh:
                    fh.write("pre-made\n")
            for p in (os.path.join(self.cfg["paths"]["tune_root"], "S-A1", "SEARCH_COMPLETE.json"),
                      os.path.join(self.cfg["paths"]["runs_root"], tag + "_best", "BEST_FIT_COMPLETE.json")):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as fh:
                    fh.write('{"pre_made": true}\n')
        self.spool = os.path.join(self.root, "spool")

    def env(self, delay="0.2", **extra):
        env = dict(os.environ)
        env["PATH"] = _FAKE + os.pathsep + self.fakebin + os.pathsep + env.get("PATH", "")
        env["FAKE_PBS_SPOOL"] = self.spool
        env["FAKE_PBS_DELAY"] = delay
        env["PIPELINE_SKIP_CONDA"] = "1"
        env["PYTHONPATH"] = _PIPE + os.pathsep + env.get("PYTHONPATH", "")
        for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "RESUME", "COHORT_TAG", "STUB_REFUSE",
                  "STUB_ONE_TASK"):
            env.pop(k, None)
        env.update(extra)
        return env

    def run(self, extra, timeout=600, **env_extra):
        cmd = [PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", self.path,
               "--poll-seconds", "0.5"] + list(extra)
        proc = subprocess.run(cmd, cwd=self.root, env=self.env(**env_extra), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
        return proc.returncode, proc.stdout

    def n_submitted(self):
        counter = os.path.join(self.spool, "counter")
        if not os.path.isfile(counter):
            return 0
        with open(counter) as fh:
            return int(fh.read().strip() or 0)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    s = Standin(keep=("real_extraction", "sim_redetection"))
    try:
        import stages.inputs as I
        stage = I.LabelFreezeStage()
        argv = stage._argv(C.load_config(GIULIA))
        assert argv.count("--exclude") == 3 and "p0_conn=Weibull kernel axis; never drawn under conn_rule=flat, recorded as NaN" in argv, argv
        assert argv[argv.index("--require-conn-rule") + 1] == "flat"
        assert argv.count("--exclude") == 3
        argv = stage._argv(C.load_config(s.path))
        assert "--exclude" not in argv and argv[argv.index("--campaigns") + 1] == "campaign_cadex_hhgap_v[125]"
        rc, out = s.run([], timeout=300)
        assert rc == 0 and "PIPELINE DONE" in out, out
        assert "label_freeze: theta topology block: ['conn_prob']  (p_topology = 1)" in out, out
        assert "label_freeze: wrote %s" % s.arts["label_freeze"] in out, out
        doc = json.load(open(s.arts["label_freeze"]))
        assert doc["topology_axes"] == ["conn_prob"], doc
        assert doc["conn_rule_observed"] == ["flat"], doc
        import stages as S
        assert S.sha256_sidecar_ok(s.arts["label_freeze"])
        assert s.n_submitted() == 0, "S3 runs on the login node"
    finally:
        s.close()


def _giulia_fresh_state(root):
    """The committed Giulia file with its work root under `root`: the dry run
    reads no state (a work root holding a run's state, as on davinci, would
    be read and, after a configuration change, refused); every other path is
    the committed one, which the checks read in the dry run's lines."""
    with open(GIULIA, "rb") as fh:
        raw = json.loads(fh.read().decode("ascii"))
    cfg = C._strip_docs(raw)
    cfg["paths"]["work_root"] = os.path.join(root, "work_root")
    path = os.path.join(root, "giulia.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    return path


def test_reference():
    root = tempfile.mkdtemp(prefix="pipeline_b3_")
    try:
        env = dict(os.environ, PATH=_FAKE + os.pathsep + os.environ.get("PATH", ""),
                   FAKE_PBS_SPOOL=os.path.join(root, "spool"), PIPELINE_SKIP_CONDA="1")
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config",
                            _giulia_fresh_state(root), "--dry-run"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=120)
        out = r.stdout
        assert r.returncode == 0 and "REFUSE" not in out, out
        assert "real_extraction: would run" in out and "label_freeze: would run" in out, out
        assert "CONFIG=/davinci-1/home/ldellamea/SBI/hpc/dsn/hpc/Config/config_giulia_cohort.davinci.json COHORT_TAG=giulia ENV_NAME=sbi_export bash /davinci-1/home/ldellamea/repos/Sbi-extractor/extractor/launch_stage_d.sh /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia" in out, out
        assert "COHORT_TAG=giulia COHORT_MANIFEST=/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia/cohort_manifest.json SIM_MAIN=/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro OUT_ROOT=/davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out_giulia_v2 TOOLS=/davinci-1/home/ldellamea/ANN/MEA_analysis CAMPAIGN_GLOB=campaign_cadex_hhgap_v* PLAN_ARGS=--allow-mixed-contract QUEUE=cpu NCPUS=48 CONCURRENCY=20 bash" in out, out
        assert "--campaigns campaign_cadex_hhgap_v[125] --require-conn-rule flat --out /davinci-1/home/ldellamea/repos/Sbi-extractor/artifacts/label_axes_giulia_hhgap.json" in out, out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    # (a) S2 without the cohort's profile: refused before anything is submitted
    s = Standin(keep=("real_extraction", "label_freeze"))
    try:
        os.remove(os.path.join(s.sbix, "sim_reextract", "profiles", "synth.sh"))
        rc, out = s.run([], timeout=300)
        assert rc == 3 and "STOP: stage sim_redetection refused: sim_redetection: the cohort's profile" in out, out
        assert s.n_submitted() == 0
    finally:
        s.close()
    # (b) every launcher variable comes from the configuration (the environment's
    #     own values are dropped): a stray RESUME/WALLTIME in the caller's shell
    #     does not reach the launcher
    import stages.inputs as I
    cfg = C.load_config(GIULIA)
    os.environ["RESUME"] = "1"
    os.environ["WALLTIME"] = "99:00:00"
    try:
        env = I.SimRedetectionStage()._env(cfg, **I.SimRedetectionStage()._vars(cfg))
        assert "RESUME" not in env and "WALLTIME" not in env
        assert env["SIM_MAIN"] == cfg.data["sim"]["sim_main"] and env["NCPUS"] == "48"
        assert env["SBI_HPC_DIR"] == os.path.join(cfg.data["paths"]["sbi_dir"], "hpc")
        assert env["ENV_NAME"] == "sbi_export"
    finally:
        del os.environ["RESUME"]
        del os.environ["WALLTIME"]


def test_contract():
    s = Standin()
    try:
        rc, out = s.run(["--dry-run"], timeout=120)
        assert rc == 0 and "REFUSE" not in out, out
        rc, out = s.run([], timeout=600)
        assert rc == 0 and "PIPELINE DONE" in out, out
        for name, p in s.arts.items():
            assert os.path.isfile(p) and os.path.isfile(p + ".sha256"), (name, p)
        assert "real_extraction: wrote %s" % s.arts["real_extraction"] in out, out
        assert "sim_redetection: [plan] tasks complete  : 1 kept as done (--resume)" in out, out
        assert "sim_redetection: wrote %s" % s.arts["sim_redetection"] in out, out
        assert "label_freeze: wrote" in out, out
        # S2 starts after S1's manifest; S3 waits for neither. (Until 2026-10-16
        # this also asserted S3 done before S2 started -- true only while S1's
        # fake jobs outlast the first tick; under load they finish inside it,
        # S1 is done and S2 starts before the tick reaches S3. Not a rule.)
        i1 = out.index("real_extraction: running")
        d1 = out.index("real_extraction: done")
        i2 = out.index("sim_redetection: running")
        assert i1 < d1 < i2, (i1, d1, i2)
        assert "label_freeze: done" in out, out
        # 5 jobs: the extraction array and aggregation; the test task, the array, the gate
        assert s.n_submitted() == 5, s.n_submitted()
        names = sorted(json.load(open(p))["name"] for p in glob.glob(os.path.join(s.spool, "*.json")))
        assert names == ["chsub_mea_array_synth", "cohort_manifest_synth", "g2_gate", "g2_mea", "g2_test"], names
        for stage in ("bank_plan", "sim_bank", "real_bank", "search", "best_fit"):
            assert "%s: skipped, artefact of record exists" % stage in out, stage
        # a re-run on a fresh state skips the whole graph
        os.remove(os.path.join(s.cfg["paths"]["work_root"], "state.json"))
        rc, out2 = s.run([], timeout=300)
        assert rc == 0 and out2.count("skipped, artefact of record exists") == 8 and s.n_submitted() == 5, out2
    finally:
        s.close()


def test_determinism():
    """One task: after the test task the array is empty and the gate runs alone."""
    s = Standin(keep=("real_extraction", "label_freeze"))
    try:
        rc, out = s.run([], timeout=600, STUB_ONE_TASK="1")
        assert rc == 0 and "PIPELINE DONE" in out, out
        assert "sim_redetection: every task complete after the test task; the gate alone" in out, out
        assert "sim_redetection: wrote" in out, out
        assert s.n_submitted() == 2, s.n_submitted()      # the test task and the gate
    finally:
        s.close()


def test_edge_cases():
    # (a) the flags refusal (a new cohort): nothing submitted, the text in the STOP line
    s = Standin()
    try:
        rc, out = s.run([], timeout=300, STUB_REFUSE="flags")
        # since D-099 the launcher's own dry run comes first (the array's size, for
        # the core cap): the flags refusal is that run's, as with the real launcher
        assert rc == 3 and ("STOP: stage real_extraction failed: the extraction could not be planned "
                            "(its dry run, exit 3)") in out, out
        assert "NEW COHORT: wrote extraction_flags_synth.sh" in out, out
        llog = open(os.path.join(s.cfg["paths"]["work_root"], "logs", "real_extraction_login_node.log")).read()
        assert "COMMIT it" in llog and "Nothing was submitted" in llog, llog
        assert s.n_submitted() == 0
    finally:
        s.close()
    # (b) a refused aggregation
    s = Standin()
    try:
        rc, out = s.run([], timeout=300, STUB_REFUSE="agg")
        assert rc == 3 and "STOP: stage real_extraction failed: the cohort manifest was refused" in out, out
        assert "REFUSED: 1 well(s) could not be used" in out and "well 1 (array index 1)" in out, out
        assert not os.path.isfile(s.arts["real_extraction"])
    finally:
        s.close()
    # (c) a refused gate
    s = Standin(keep=("real_extraction", "label_freeze"))
    try:
        rc, out = s.run([], timeout=300, STUB_REFUSE="gate")
        assert rc == 3 and "STOP: stage sim_redetection failed: the re-detection gate refused" in out, out
        assert "[gate] FAIL campaign_x/sweep_task0001" in out, out
        assert not os.path.isfile(s.arts["sim_redetection"])
        assert s.n_submitted() == 3
    finally:
        s.close()


def _cap(s, stages_share, sim_ncpus=1):
    """resources.max_cores = the sequencer's cores + `stages_share`; every
    sized job small enough for the load's cross-check."""
    c = s.cfg
    c["search"]["trial"]["ncpus"] = 1
    c["best_fit"]["ncpus"] = 1
    c["sim"]["ncpus"] = int(sim_ncpus)
    c["bank"]["verify"]["ncpus"] = 1
    c["resources"]["max_cores"] = int(c["driver"]["ncpus"]) + int(stages_share)
    with open(s.path, "w") as fh:
        json.dump(c, fh, indent=1, sort_keys=True)
    return c


def _fake_records(s):
    recs = [json.load(open(p)) for p in glob.glob(os.path.join(s.spool, "*.json"))
            if os.path.basename(p)[:-5].isdigit()]
    return sorted(recs, key=lambda r: r["num"])


def test_cores():
    # (a) S2 under a share of 10 with tasks of 3 cores: CONCURRENCY 3 = min(20, 10 // 3)
    s = Standin(keep=("real_extraction", "label_freeze"))
    try:
        _cap(s, 10, sim_ncpus=3)
        rc, out = s.run(["--dry-run"], timeout=120)
        assert rc == 0 and ("cores (D-099): the test task 3; the array at CONCURRENCY 3 = "
                            "min(sim.concurrency 20, the cap's room 3) of 3 cores each") in out, out
        rc, out = s.run([], timeout=300)
        assert rc == 0 and "PIPELINE DONE" in out, out
        assert ("sim_redetection: the array at CONCURRENCY 3 (sim.concurrency 20; the cap leaves "
                "room for 3 task(s) of 3 cores, D-099)") in out, out
        arrays = [r for r in _fake_records(s) if r["name"] == "g2_mea"]
        assert len(arrays) == 1 and arrays[0]["array_spec"] == "0-1%3", arrays
        st = json.load(open(os.path.join(s.cfg["paths"]["work_root"], "state.json")))
        jobs = st["stages"]["sim_redetection"]["jobs"]
        assert [(j["role"], j.get("ncpus")) for j in jobs] == [
            ("re-detection test task 0", 3), ("re-detection array", 3), ("re-detection gate", 8)], jobs
        assert jobs[2]["after"] == jobs[1]["id"] and "after" not in jobs[1]
    finally:
        s.close()
    # (b) S1's array, 2 wells x 2 cores at once, in a share of 3: refused, nothing submitted
    s = Standin()
    try:
        _cap(s, 3)
        rc, out = s.run([], timeout=300)
        assert rc == 3 and ("STOP: stage real_extraction refused: real_extraction: the extraction "
                            "array asks 2 x 2 = 4 cores at once and has no concurrency limit, more "
                            "than the 3 the cap leaves") in out, out
        assert s.n_submitted() == 0
    finally:
        s.close()
    # (c) a share of 4: S1 fits and is recorded; S2's gate (8) never can: refused at its start
    s = Standin()
    try:
        _cap(s, 4)
        rc, out = s.run([], timeout=300)
        assert rc == 3 and ("STOP: stage sim_redetection refused: sim_redetection: a job of 8 cores "
                            "can never start: the stages' jobs share 4 cores (resources.max_cores "
                            "%d less the sequencer's job, %d)"
                            % (int(s.cfg["driver"]["ncpus"]) + 4, int(s.cfg["driver"]["ncpus"]))) in out, out
        st = json.load(open(os.path.join(s.cfg["paths"]["work_root"], "state.json")))
        jobs = st["stages"]["real_extraction"]["jobs"]
        assert [(j["role"], j.get("ncpus")) for j in jobs] == [("extraction array", 4),
                                                               ("cohort manifest", 1)], jobs
        assert jobs[1]["after"] == jobs[0]["id"]
        assert st["stages"]["real_extraction"]["status"] == "done"
        assert s.n_submitted() == 2, "S2 submitted nothing"
    finally:
        s.close()


# ------------------------------------------------------------------ main ---

CHECKS = [("test_known_answer", test_known_answer), ("test_reference", test_reference),
          ("test_convergence", test_convergence), ("test_invariants", test_invariants),
          ("test_contract", test_contract), ("test_determinism", test_determinism),
          ("test_edge_cases", test_edge_cases), ("test_cores", test_cores)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        t1 = time.time()
        try:
            fn()
            rows.append((name, "PASS", "", time.time() - t1))
        except unittest.SkipTest as exc:
            rows.append((name, "SKIP", str(exc), time.time() - t1))
        except AssertionError as exc:
            bad += 1
            rows.append((name, "FAIL", str(exc)[:2500], time.time() - t1))
        except Exception:
            bad += 1
            rows.append((name, "ERROR", traceback.format_exc()[-2500:], time.time() - t1))
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
