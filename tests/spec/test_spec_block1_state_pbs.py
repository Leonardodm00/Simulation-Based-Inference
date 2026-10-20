"""Spec-derived tests of the state file, the scheduler adapter, the stage
graph, the done-predicate and the core cap (SPEC secs. 2.1, 3.2, 3.3;
block 1 and its amendments (c), (d)).

Oracles O-G3, O-G4, O1.3-O1.8 of the 2026-10-20 report. Expected values are
hand-written from the spec text.
"""
import hashlib
import json
import os
import stat
import sys

import pytest

from conftest import GIULIA, PIPE

import config as C
import pbs as PBS
import pipeline as P
import stages as S
from state import State


# ---------------------------------------------------------------- state ----

def _state(tmp_path):
    st = State.new(str(tmp_path), "toy", "/x/toy.json", "a" * 64)
    st.stage("search")["sub"]["rounds"] = [1, 2]
    st.add_job("search", "123.server", "trial", ["qsub", "-N", "t"], log="/l", ncpus=4)
    return st


def test_state_roundtrip_identity(tmp_path):
    st = _state(tmp_path)
    st.save()
    back = State.load(str(tmp_path))
    assert back.data == json.loads(json.dumps(st.data))


def test_state_file_sorted_ascii(tmp_path):
    st = _state(tmp_path)
    p = st.save()
    blob = open(p, "rb").read()
    assert all(b < 128 for b in blob)
    data = json.loads(blob)
    assert data["schema"] == "pipeline_state/1"
    # sorted keys: re-dumping with sort_keys and the same indent gives the bytes
    assert blob.decode().rstrip("\n") == json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True)
    for f in ("dataset", "config_path", "config_sha256", "started_utc", "updated_utc", "pid", "host",
              "stages"):
        assert f in data, f
    stg = data["stages"]["search"]
    for f in ("status", "started_utc", "finished_utc", "jobs", "resubmits", "note", "sub"):
        assert f in stg, f
    for f in ("id", "role", "submitted_utc", "cmd", "log"):
        assert f in stg["jobs"][0], f


def test_state_tmp_only_crash_leaves_old(tmp_path):
    st = _state(tmp_path)
    p = st.save()
    old = open(p, "rb").read()
    # a writer that died after writing its temporary file
    with open(p + ".%d.tmp" % 99999, "w") as fh:
        fh.write("{half")
    assert open(p, "rb").read() == old
    assert State.load(str(tmp_path)).data["config_sha256"] == "a" * 64


def test_state_prev_fallback(tmp_path):
    st = _state(tmp_path)
    st.save()
    st.data["dataset"] = "second"
    p = st.save()
    assert os.path.isfile(p + ".prev")
    assert json.load(open(p + ".prev"))["dataset"] == "toy"
    with open(p, "wb") as fh:
        fh.write(b"\0" * 100)
    msgs = []
    back = State.load(str(tmp_path), log=msgs.append)
    assert back.data["dataset"] == "toy"
    assert msgs and "does not parse" in msgs[0]


def test_state_neither_parses(tmp_path):
    st = _state(tmp_path)
    st.save()
    p = st.save()
    for q in (p, p + ".prev"):
        open(q, "w").write("{")
    with pytest.raises(ValueError, match="no state file parses"):
        State.load(str(tmp_path))


def test_state_missing_is_fresh(tmp_path):
    st = _state(tmp_path)
    p = st.save()
    st.save()
    os.remove(p)
    assert os.path.isfile(p + ".prev")
    assert State.load(str(tmp_path)) is None


# ---------------------------------------------------------------- pbs -------

DAVINCI_QSTAT = """Job Id: 1829506.davinci-mgt01
    Job_Name = tune_trial
    Job_Owner = someone@dvlogin01
    resources_used.cpupercent = 4700
    resources_used.walltime = 101:02:03
    job_state = F
    queue = cpu
    Exit_status = 271
    Output_Path = dvlogin02:/home/u/tune/jobs/abcdef.1.l
\tog
    array_state_count = Queued:0 Running:2 Exiting:0 Expired:5
"""


def test_parse_qstat_fields():
    js = PBS.parse_qstat("1829506.davinci-mgt01", DAVINCI_QSTAT)
    assert js.state == "F" and js.finished
    assert js.exit_status == 271
    assert js.walltime_s == 101 * 3600 + 2 * 60 + 3          # hh:mm:ss as seconds
    assert js.array_counts == {"Queued": 0, "Running": 2, "Exiting": 0, "Expired": 5}
    assert js.output_path == "/home/u/tune/jobs/abcdef.1.log"   # wrapped, host part dropped


@pytest.mark.parametrize("letter", list("QRHBEFXSTWM"))
def test_known_states(letter):
    js = PBS.parse_qstat("1", "Job Id: 1\n    job_state = %s\n" % letter)
    assert js.state == letter
    assert js.finished == (letter in ("F", "X"))


@pytest.mark.parametrize("text", ["Job Id: 1\n    job_state = Z\n", "garbage", ""])
def test_unknown_state(text):
    js = PBS.parse_qstat("1", text)
    assert js.state == "unknown" and js.finished


def _fake_bin(tmp_path, name, script):
    p = tmp_path / name
    p.write_text("#!/bin/bash\n" + script)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_submit_first_token(tmp_path, monkeypatch):
    q = _fake_bin(tmp_path, "qsub", 'echo "1829506[].davinci extra"\n')
    monkeypatch.setattr(PBS, "QSUB", q)
    assert PBS.submit(["qsub", "x.pbs"]) == "1829506[].davinci"


def test_submit_error_carries_output_never_retried(tmp_path, monkeypatch):
    cnt = tmp_path / "n"
    q = _fake_bin(tmp_path, "qsub", 'echo x >> %s; echo out-text; echo err-text >&2; exit 3\n' % cnt)
    monkeypatch.setattr(PBS, "QSUB", q)
    with pytest.raises(PBS.SubmitError) as ei:
        PBS.submit(["qsub", "x.pbs"])
    assert "out-text" in str(ei.value) and "err-text" in str(ei.value)
    assert open(str(cnt)).read().count("x") == 1


def test_qstat_retried_three_times(tmp_path, monkeypatch):
    cnt = tmp_path / "n"
    q = _fake_bin(tmp_path, "qstat", 'echo x >> %s; echo "cannot connect to server" >&2; exit 2\n' % cnt)
    monkeypatch.setattr(PBS, "QSTAT", q)
    with pytest.raises(PBS.SchedulerError):
        PBS.status("1.s", retry_wait_s=0.0)
    assert open(str(cnt)).read().count("x") == 3


def test_qstat_unknown_job(tmp_path, monkeypatch):
    q = _fake_bin(tmp_path, "qstat", 'echo "qstat: Unknown Job Id 5.s" >&2; exit 153\n')
    monkeypatch.setattr(PBS, "QSTAT", q)
    js = PBS.status("5.s", retry_wait_s=0.0)
    assert js.state == "unknown" and js.finished


def test_job_ids():
    assert PBS.job_seq("1829506.davinci-mgt01") == "1829506"
    assert PBS.job_seq("1829506[].davinci") == "1829506[]"
    assert PBS.job_number("1829506.davinci-mgt01") == 1829506
    assert PBS.job_number("abc") == -1
    assert PBS.same_job("1829506.davinci", "1829506.davinci-mgt01")
    assert not PBS.same_job("1829506.davinci", "1829507.davinci")
    assert not PBS.same_job("1829506[].davinci", "1829506.davinci")


# ------------------------------------------------- job_cores (amendment d) ---

@pytest.mark.parametrize("argv,expected", [
    (["qsub", "-l", "select=1:ncpus=48:mem=160gb", "x"], 48),
    (["qsub", "-l", "select=3:ncpus=8:mem=1gb", "x"], 24),            # K x N
    (["qsub", "-l", "select=ncpus=8:mem=1gb", "x"], 8),               # K = 1 when absent
    (["qsub", "-l", "select=2:ncpus=4,walltime=01:00:00", "x"], 8),   # select=...,walltime=...
    (["qsub", "-l", "select=1:ncpus=4", "-J", "0-68%20", "x"], 80),   # min(20, 69) x 4
    (["qsub", "-l", "select=1:ncpus=4", "-J", "0-2%9", "x"], 12),     # min(9, 3) x 4
    (["qsub", "-l", "select=1:ncpus=4", "-J", "0-9:2", "x"], 20),     # (9-0)//2+1 = 5
    (["qsub", "-l", "select=1:ncpus=4", "-J", "1-10:3%2", "x"], 8),   # count 4, min(2, 4)
    (["qsub", "-l", "select=1:ncpus=4", "-J", "0-4", "x"], 20),       # no %: all
    (["qsub", "-l", "walltime=01:00:00", "x"], None),                  # neither says
])
def test_job_cores_from_argv(argv, expected):
    assert P.job_cores({"cmd": argv}) == expected


def test_job_cores_recorded_wins():
    assert P.job_cores({"ncpus": 7, "cmd": ["qsub", "-l", "select=1:ncpus=48", "x"]}) == 7


class _St(object):
    def __init__(self, stages):
        self.data = {"stages": stages}


def _job(i, n=None, after=None, finished=False):
    j = {"id": i, "cmd": [], "finished": finished}
    if n is not None:
        j["ncpus"] = n
    if after:
        j["after"] = after
    return j


def test_cores_in_flight_rules():
    st = _St({
        "sim_bank": {"status": "running", "jobs": [_job("10", 80), _job("11", 16, after="10")]},
        "search": {"status": "running", "jobs": [_job("20", 48), _job("21", 48, finished=True),
                                                 _job("22")]},
        "real_bank": {"status": "done", "jobs": [_job("30", 99)]},        # done: holds none
        "bank_plan": {"status": "skipped", "jobs": [_job("31", 99)]},     # skipped: holds none
    })
    cores, n, unknown = P.cores_in_flight(st)
    # array 80 with its gate 16 after it: max(80, 16) = 80; trial 48; finished 21 not counted
    assert cores == 80 + 48
    assert n == 3
    assert unknown == ["22"]


def test_cores_in_flight_gate_larger_than_array():
    st = _St({"sim_bank": {"status": "running",
                           "jobs": [_job("10", 3), _job("11", 8, after="10")]}})
    assert P.cores_in_flight(st)[0] == 8          # max(3, 8), never 11


def test_cores_in_flight_two_dependents():
    st = _St({"s": {"status": "running",
                    "jobs": [_job("1", 4), _job("2", 3, after="1"), _job("3", 3, after="1")]}})
    assert P.cores_in_flight(st)[0] == 6          # max(4, 3 + 3)


def test_cores_in_flight_dependent_of_finished_job():
    st = _St({"s": {"status": "running",
                    "jobs": [_job("1", 40, finished=True), _job("2", 8, after="1")]}})
    assert P.cores_in_flight(st)[0] == 8


def test_cores_in_flight_after_id_spelled_differently():
    """sec. 3.3: job ids are never compared as whole strings (qsub, qselect and
    PBS_JOBID may spell the server part differently). An `after` naming the
    array with another server suffix must still group the gate with it."""
    st = _St({"s": {"status": "running",
                    "jobs": [_job("1845588[].davinci-mgt01", 80),
                             _job("1845589.davinci-mgt01", 16, after="1845588[].davinci")]}})
    assert P.cores_in_flight(st)[0] == 80


# ----------------------------------------------------- Submitter (d) --------

class _Cfg(object):
    def __init__(self, cap, drv):
        self.data = {"resources": {"max_cores": cap}, "driver": {"ncpus": drv}}


def _sub(cap, drv, stages):
    return P.Submitter(PBS, _St(stages), "search", lambda t: None, _Cfg(cap, drv))


def test_submitter_budget_free_fits():
    s = _sub(200, 4, {"search": {"status": "running", "jobs": [_job("1", 48), _job("2", 48)]}})
    assert s.budget() == 196
    assert s.in_flight() == 96
    assert s.free() == 100
    assert s.fits(100) and not s.fits(101)


def test_submitter_check_never_fits():
    s = _sub(200, 4, {})
    with pytest.raises(P.CapExceeded) as ei:
        s.check(197)
    assert "can never start" in str(ei.value)
    with pytest.raises(P.CapExceeded):
        s.fits(197)                     # fits asks check first
    assert s.check(196) is True


def test_submitter_without_cfg_everything_fits():
    s = P.Submitter(PBS, _St({}), "x", lambda t: None, None)
    assert s.fits(10 ** 6)


# --------------------------------------------- graph and done (sec. 2.1) ----

def test_graph_order_and_deps():
    cfg = C.load_config(GIULIA)
    g = S.default_graph(cfg)
    assert [s.name for s in g] == ["real_extraction", "sim_redetection", "label_freeze", "bank_plan",
                                   "sim_bank", "real_bank", "search", "best_fit"]
    deps = dict((s.name, set(s.deps)) for s in g)
    assert deps == {"real_extraction": set(), "sim_redetection": {"real_extraction"},
                    "label_freeze": set(),
                    "bank_plan": {"real_extraction", "sim_redetection", "label_freeze"},
                    "sim_bank": {"bank_plan"}, "real_bank": {"bank_plan"},
                    "search": {"sim_bank", "real_bank"}, "best_fit": {"search"}}
    assert not any(isinstance(s, S.NotBuiltStage) for s in g)       # every block built


def test_topological_refuses_cycle():
    class A(S.Stage):
        name, deps = "a", ("b",)

    class B(S.Stage):
        name, deps = "b", ("a",)
    with pytest.raises(ValueError):
        S.topological_order([A(), B()])


def _sidecar(tmp_path, token):
    f = tmp_path / "a.json"
    f.write_bytes(b'{"x": 1}\n')
    (tmp_path / "a.json.sha256").write_text(token)
    return str(f)


def test_sidecar_matching(tmp_path):
    d = hashlib.sha256(b'{"x": 1}\n').hexdigest()
    assert S.sha256_sidecar_ok(_sidecar(tmp_path, d + "  a.json\n"))


@pytest.mark.parametrize("token", ["", "0" * 64, "abc", "x" * 64])
def test_sidecar_not_matching(tmp_path, token):
    assert not S.sha256_sidecar_ok(_sidecar(tmp_path, token))


def test_sidecar_missing(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}")
    assert not S.sha256_sidecar_ok(str(f))


def test_artefacts_of_record():
    cfg = C.load_config(GIULIA)
    d = cfg.data
    j = os.path.join
    exp = {
        "real_extraction": ("sha256", j(d["real"]["extract_root"], "cohort_manifest.json")),
        "sim_redetection": ("sha256", j(d["sim"]["out_root"], "REEXTRACTION_RECORD.json")),
        "label_freeze": ("sha256", j(d["paths"]["sbix_dir"], d["freeze"]["out"])),
        "bank_plan": ("sha256", j(d["paths"]["sbix_dir"], "export_windows", "giulia", "bank_plan.json")),
        "real_bank": ("pair", j(d["paths"]["banks_root"], "giulia_real", "shard_0000.npz")),
        "search": ("file", j(d["paths"]["tune_root"], "S-A1", "SEARCH_COMPLETE.json")),
        "best_fit": ("file", j(d["paths"]["runs_root"], "giulia_best", "BEST_FIT_COMPLETE.json")),
    }
    for name, kp in exp.items():
        assert S.artefact_of(name, cfg) == kp, name


def test_sim_bank_needs_certificate(tmp_path):
    """Block 2 (c): BANK_RECORD.json + .sha256 AND BANK_VERIFIED.json +
    .sha256 whose record_sha256 is the record's sha256."""
    raw = json.load(open(GIULIA))
    raw["paths"]["banks_root"] = str(tmp_path)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw))
    cfg = C.load_config(str(p))
    stage = [s for s in S.default_graph(cfg) if s.name == "sim_bank"][0]
    root = tmp_path / "giulia_sim"
    root.mkdir()
    rec = root / "BANK_RECORD.json"
    rec.write_text('{"r": 1}\n')
    rh = hashlib.sha256(rec.read_bytes()).hexdigest()
    (root / "BANK_RECORD.json.sha256").write_text(rh + "\n")
    assert not stage.is_done(cfg)[0]                      # record alone: not done
    cert = root / "BANK_VERIFIED.json"
    cert.write_text(json.dumps({"record_sha256": "0" * 64}))
    (root / "BANK_VERIFIED.json.sha256").write_text(hashlib.sha256(cert.read_bytes()).hexdigest())
    assert not stage.is_done(cfg)[0]                      # certifies another record
    cert.write_text(json.dumps({"record_sha256": rh}))
    (root / "BANK_VERIFIED.json.sha256").write_text(hashlib.sha256(cert.read_bytes()).hexdigest())
    assert stage.is_done(cfg)[0]
