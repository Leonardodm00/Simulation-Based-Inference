"""Spec-derived tests of the bank and input stages' contracts that can be
checked in process (SPEC blocks 2 and 3; amendment D-099 of each).

Oracles O2.1, O2.2, O3.1 of the 2026-10-20 report.
"""
import json
import os

import pytest

from conftest import GIULIA, SBIX

import config as C
import stages as S
from stages import bank as B
from stages import inputs as I


def _cfg(tmp_path, **paths):
    raw = json.load(open(GIULIA))
    raw["paths"]["sbix_dir"] = SBIX
    for k, v in paths.items():
        raw["paths"][k] = v
    p = tmp_path / "giulia_test.json"
    p.write_text(json.dumps(raw))
    return C.load_config(str(p))


def test_bank_env_rebuilt_from_config(tmp_path, monkeypatch):
    """Block 2: PROFILE, ENV_NAME = envs.extract, QUEUE = bank.queue,
    CONCURRENCY = bank.concurrency, SBI_HPC_DIR = <sbi_dir>/hpc; NCPUS and
    GATE_NCPUS the launcher's own defaults, never the environment's (D-099 (a))."""
    if not os.path.isfile(os.path.join(SBIX, "export_windows", "launch_bank.sh")):
        pytest.skip("no Sbi-extractor checkout")
    cfg = _cfg(tmp_path, work_root=str(tmp_path / "w"))
    for k, v in (("NCPUS", "4"), ("GATE_NCPUS", "99"), ("WALLTIME", "99:00:00"), ("DRYRUN", "1"),
                 ("TEST_INDEX", "7"), ("CONCURRENCY", "1000"), ("QUEUE", "wrong")):
        monkeypatch.setenv(k, v)
    env = S.BUILT["sim_bank"]()._env(cfg)
    assert env["ENV_NAME"] == "sbi_export"
    assert env["QUEUE"] == "cpu"
    assert env["CONCURRENCY"] == "20"
    assert env["SBI_HPC_DIR"] == os.path.join(cfg.data["paths"]["sbi_dir"], "hpc")
    assert env["PROFILE"] == os.path.join(str(tmp_path / "w"), "bank_profile_giulia.json")
    # the launcher's defaults at e9aeb55: NCPUS="${NCPUS:-1}", GATE_NCPUS="${GATE_NCPUS:-8}"
    assert env["NCPUS"] == "1" and env["GATE_NCPUS"] == "8"
    for k in ("WALLTIME", "DRYRUN", "TEST_INDEX"):
        assert k not in env, k


def test_inputs_env_drops_stray_variables(tmp_path, monkeypatch):
    """Block 3: every launcher's environment is rebuilt from the
    configuration (a stray RESUME or WALLTIME in the caller's shell is dropped)."""
    cfg = _cfg(tmp_path)
    for k in ("RESUME", "WALLTIME", "TEST_INDEX", "DRYRUN", "NCPUS", "CONCURRENCY", "PLAN_ARGS"):
        monkeypatch.setenv(k, "stray")
    stage = I.SimRedetectionStage()
    env = stage._env(cfg, **stage._vars(cfg))
    for k in ("RESUME", "WALLTIME", "TEST_INDEX", "DRYRUN"):
        assert k not in env, k
    assert env["NCPUS"] == "48" and env["CONCURRENCY"] == "20"
    assert env["PLAN_ARGS"] == "--allow-mixed-contract"
    assert env["ENV_NAME"] == "sbi_export"
    assert env["COHORT_MANIFEST"] == S.artefact_of("real_extraction", cfg)[1]


def test_label_freeze_argv(tmp_path):
    """Block 3: --sim_main, --campaigns, --require-conn-rule, one --exclude
    NAME=REASON per excluded axis, --out <sbix_dir>/<freeze.out>."""
    cfg = _cfg(tmp_path)
    argv = I.LabelFreezeStage()._argv(cfg)
    d = cfg.data
    assert argv[argv.index("--sim_main") + 1] == d["sim"]["sim_main"]
    assert argv[argv.index("--campaigns") + 1] == "campaign_cadex_hhgap_v[125]"
    assert argv[argv.index("--require-conn-rule") + 1] == "flat"
    assert argv[argv.index("--out") + 1] == os.path.join(SBIX, d["freeze"]["out"])
    ex = [argv[i + 1] for i, a in enumerate(argv) if a == "--exclude"]
    assert sorted(e.split("=", 1)[0] for e in ex) == ["beta_conn", "d0_conn", "p0_conn"]


class _FakeSub(object):
    """A stand-in for pipeline.Submitter: a scheduler answering 'finished'."""

    def __init__(self, exit_status):
        self.exit_status = exit_status
        self.lines = []

    def status(self, job_id):
        import pbs as PBS
        return PBS.JobStatus(job_id, "F", exit_status=self.exit_status)

    def log(self, t):
        self.lines.append(t)


@pytest.mark.parametrize("exit_status,passes", [(0, True), (1, False), (271, False), (None, False)])
def test_s2_test_task_pass_is_exit_0(tmp_path, monkeypatch, exit_status, passes):
    """Block 3: S2's test task -- 'the job's exit 0 is its pass'. A finished
    job with no Exit_status (deleted before it ran; an unknown, purged job) is
    not an exit 0."""
    cfg = _cfg(tmp_path)
    stage = I.SimRedetectionStage()
    started = []
    monkeypatch.setattr(stage, "_start_array", lambda cfg, st, pbs: started.append(1))
    st = {"jobs": [{"id": "7.s", "role": "test", "finished": False}],
          "sub": {"phase": "test", "test_job": "7.s"}}
    stage.poll(cfg, st, _FakeSub(exit_status))
    assert bool(started) == passes, (exit_status, st["sub"])


def test_giulia_bank_values_match_committed_profile():
    """Block 2: the committed Giulia configuration reproduces the committed
    profile's decided values (Sbi-extractor export_windows/profiles/giulia.json
    at e9aeb55), key by key."""
    prof_path = os.path.join(SBIX, "export_windows", "profiles", "giulia.json")
    if not os.path.isfile(prof_path):
        pytest.skip("no Sbi-extractor checkout")
    prof = json.load(open(prof_path))
    bank = json.load(open(GIULIA))["bank"]
    decided = ("on_axis", "on_declared_box", "topology_axes", "expected_p", "t_win_s", "n_e",
               "floor_hz", "floor_rtol", "sample_seed", "bootstrap_resamples", "bootstrap_seed",
               "bytes_sample_rows", "rows_per_shard")
    for k in decided:
        assert bank[k] == prof[k], k
    cfg = C.load_config(GIULIA)
    assert prof["sim_dir"] == cfg.data["sim"]["sim_main"]
    assert prof["bank_root"] == os.path.join(cfg.data["paths"]["banks_root"], "giulia_sim")
    assert prof["real_bank_root"] == os.path.join(cfg.data["paths"]["banks_root"], "giulia_real")


def test_launcher_line_formats_parse():
    """Block boundary: the lines the stages parse, against the real launchers
    at e9aeb55 (echo lines read from the scripts, not from the stages)."""
    import re
    bank = open(os.path.join(SBIX, "export_windows", "launch_bank.sh")).read()
    s2 = open(os.path.join(SBIX, "sim_reextract", "launch_sim_reextract.sh")).read()
    s1 = open(os.path.join(SBIX, "extractor", "launch_stage_d.sh")).read()
    assert 'echo "  job   : $JOB_ID"' in bank and 'echo "  gate  : $gate"' in bank
    assert 'echo "  job   : $arr"' in s2 and 'echo "  gate  : $gate"' in s2
    assert 'echo "  array : $arr"' in s1 and 'echo "  agg   : $agg' in s1
    # the stages' regexes on the rendered lines
    jobs = dict(B._JOB_LINE.findall("x\n  job   : 1845588[].davinci-mgt01\n  gate  : 1845589.davinci-mgt01\n"))
    assert jobs == {"job": "1845588[].davinci-mgt01", "gate": "1845589.davinci-mgt01"}
    s1jobs = dict(I._S1_LINE.findall("  array : 77[].d\n  agg   : 78.d   (runs once the array has ended)\n"))
    assert s1jobs == {"array": "77[].d", "agg": "78.d"}
    m = I._S1_QSUB_J.search("  qsub -J 0-68 -N chsub_mea_array_giulia -o out/x -v ENV_NAME=e run.pbs")
    assert m and int(m.group(1)) + 1 == 69
    # the launcher's default lines the bank stage reads (D-099 (a))
    assert re.search(r'^NCPUS="\$\{NCPUS:-1\}"', bank, re.M)
    assert re.search(r'^GATE_NCPUS="\$\{GATE_NCPUS:-8\}"', bank, re.M)
