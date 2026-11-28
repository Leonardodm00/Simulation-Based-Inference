"""Block 30: stop a search by hand (specs/SPEC.md block 30, amendment 3).

Oracles (from the spec, before the code was read):
  O30.1 `pipeline.py stop-search` writes or removes one file and submits nothing.
  O30.2 the file <tune_root>/<campaign>/STOP_REQUEST.json of the search's view:
        {schema search_stop/1, requested_utc ISO 8601 UTC, kill_running bool,
        user str, host str}, ASCII JSON, sorted keys, via a temporary file.
  O30.3 exit 2 `REFUSED: <reason>`, nothing written: --search missing / unknown
        on a list, given on a null configuration; the search ended (naming
        ended_by); no record of status ok ("nothing to fit yet; stop the whole
        run with `stop` instead"); --cancel without a request or after the end.
  O30.4 a request present is replaced (the line names its time and
        kill_running); --cancel removes it (`STOP CANCELLED: ...`); the last line
        `STOP REQUESTED: search <name or campaign> (<campaign>); kill running:
        yes|no; the sequencer acts at its next tick (every driver.poll_seconds)`.
  O30.5 the stage under a request: nothing submitted; kill_running deletes the
        last job of each running trial once (stop_deleted); a job ended without
        a record -> killed with `stopped by the user (stop-search
        --kill-running)` (deleted) or `ended without a record after a stop
        request`; a record written before the deletion is kept; once nothing
        runs or is examined the waiting trials are `cancelled` and the search
        finishes: SEARCH_COMPLETE.json ended_by user_stop, `stop_request` the
        file's content, the cancelled trials counted in n_trials.
  O30.6 one log line when a request is first seen:
        `search: stop requested <utc> (kill running: yes|no): no new trial; <n> running`.
  O30.7 status_line: None without a request; `stop requested <utc> (kill
        running: yes|no)`, `; the search ended (user_stop)` once SEARCH_COMPLETE says so.
  O30.8 search_report: `next: nothing (cancelled by the stop request; never
        submitted)`; under a request no `next:` predicts a submission; the
        request's status line after the header.
  O30.9 the dry run's probe, trial and member argv show --runner-extra with
        the replicate flags.
"""

import glob
import json
import os
import re
import shutil
import time

import pytest

import _spec_util as U

import config as C                # noqa: E402
import pbs as PBS                 # noqa: E402
import stages.common as CM        # noqa: E402
import stages.search as SS        # noqa: E402
import stages.search_stop as SST  # noqa: E402

UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|\+00:00)$")


@pytest.fixture
def root():
    r = U.tmpdir("spec_b30_")
    yield r
    shutil.rmtree(r, ignore_errors=True)


def _null_cfg(root):
    """Giulia's configuration (searches null, campaign S-A1) under root."""
    path, d = U.giulia_copy(root)
    return path, d, os.path.join(d["paths"]["tune_root"], d["search"]["campaign"])


def _ok_record(camp, tid, nll=1.0, status="ok"):
    rec = {"trial_id": tid, "status": status, "nll": nll, "config": {"lr": 1e-3}}
    if status is None:
        del rec["status"]
    U.write_json(os.path.join(camp, "trials", tid + ".json"), rec)


def _cli(root, path, *extra):
    spool = os.path.join(root, "spool")
    os.makedirs(spool, exist_ok=True)
    env = U.env_with_fake_pbs(spool)
    rc, out = U.run([U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "stop-search", "--config",
                     path] + list(extra), env=env, cwd=root, timeout=120)
    jobs = [p for p in glob.glob(os.path.join(spool, "*.json"))
            if os.path.basename(p)[:-5].isdigit()]
    assert jobs == [], "stop-search submitted a job: %s" % jobs          # O30.1
    return rc, out


def _last(out):
    return out.rstrip().splitlines()[-1]


# --------------------------------------------------------- O30.1 - O30.4 ---

def test_request_written_with_its_content(root):
    path, d, camp = _null_cfg(root)
    _ok_record(camp, "a" * 64)
    before = U.tree_snapshot(root)
    rc, out = _cli(root, path)
    assert rc == 0, out
    assert _last(out) == ("STOP REQUESTED: search S-A1 (S-A1); kill running: no; the sequencer "
                          "acts at its next tick (every driver.poll_seconds)"), out
    req = os.path.join(camp, "STOP_REQUEST.json")
    raw = open(req, "rb").read()
    raw.decode("ascii")
    obj = json.loads(raw.decode("ascii"))
    assert sorted(obj) == ["host", "kill_running", "requested_utc", "schema", "user"]
    assert obj["schema"] == "search_stop/1" and obj["kill_running"] is False
    assert isinstance(obj["user"], str) and isinstance(obj["host"], str)
    assert UTC_RE.match(obj["requested_utc"]), obj["requested_utc"]
    # sorted keys as written
    keys_in_file = re.findall(rb'"([a-z_]+)":', raw)
    assert [k.decode() for k in keys_in_file] == sorted(obj)
    # one file written (and its temporary file gone)
    after = U.tree_snapshot(root)
    new = sorted(set(after) - set(before))
    assert [n for n in new if not n.startswith("spool")] == [os.path.relpath(req, root)], new


def test_replacement_and_cancel(root):
    path, d, camp = _null_cfg(root)
    _ok_record(camp, "a" * 64)
    assert _cli(root, path)[0] == 0
    first = json.load(open(os.path.join(camp, "STOP_REQUEST.json")))
    time.sleep(1.1)
    rc, out = _cli(root, path, "--kill-running")
    assert rc == 0
    line = _last(out)
    assert line.startswith("STOP REQUESTED: search S-A1 (S-A1); kill running: yes; the sequencer "
                           "acts at its next tick (every driver.poll_seconds)"), line
    assert first["requested_utc"] in line and "kill running: no" in line, line   # the previous one
    second = json.load(open(os.path.join(camp, "STOP_REQUEST.json")))
    assert second["kill_running"] is True and second["requested_utc"] != first["requested_utc"]
    rc, out = _cli(root, path, "--cancel")
    assert rc == 0 and _last(out).startswith("STOP CANCELLED:"), out
    assert not os.path.exists(os.path.join(camp, "STOP_REQUEST.json"))
    rc, out = _cli(root, path, "--cancel")                 # nothing to cancel
    assert rc == 2 and _last(out).startswith("REFUSED:"), out


def test_refusals_write_nothing(root):
    path, d, camp = _null_cfg(root)
    os.makedirs(os.path.join(camp, "trials"))
    # no completed trial: only a failed record, and an ok record set aside
    _ok_record(camp, "f" * 64, status="failed")
    U.write_json(os.path.join(camp, "trials", "e" * 64 + ".json.nodefault1"),
                 {"trial_id": "e" * 64, "status": "ok", "nll": 0.1})
    snap = U.tree_snapshot(camp)
    rc, out = _cli(root, path)
    assert rc == 2 and _last(out).startswith("REFUSED:"), out
    assert "nothing to fit yet; stop the whole run with `stop` instead" in out, out
    assert U.tree_snapshot(camp) == snap
    # --search on a configuration whose searches is null
    _ok_record(camp, "a" * 64)
    snap = U.tree_snapshot(camp)
    rc, out = _cli(root, path, "--search", "A1")
    assert rc == 2 and _last(out).startswith("REFUSED:") and U.tree_snapshot(camp) == snap, out
    # --cancel with --kill-running
    rc, out = _cli(root, path, "--cancel", "--kill-running")
    assert rc == 2 and _last(out).startswith("REFUSED:") and U.tree_snapshot(camp) == snap, out
    # the search has ended: refused naming its ended_by, request or cancel alike
    U.write_json(os.path.join(camp, "SEARCH_COMPLETE.json"), {"ended_by": "budget"})
    snap = U.tree_snapshot(camp)
    for extra in ([], ["--kill-running"], ["--cancel"]):
        rc, out = _cli(root, path, *extra)
        assert rc == 2 and _last(out).startswith("REFUSED:") and "budget" in out, (extra, out)
        assert U.tree_snapshot(camp) == snap


def test_record_without_status_counts_as_ok(root):
    """SearchStage's reading: str(rec.get("status", "ok")) (sec. 3 of block 30:
    "the records the stage reads as its ledger, SearchStage's own reading")."""
    path, d, camp = _null_cfg(root)
    _ok_record(camp, "b" * 64, status=None)
    rc, out = _cli(root, path)
    assert rc == 0, out


def test_list_configuration_names_the_search(root):
    path, d = U.linddup_copy(root)
    camp_a5 = os.path.join(d["paths"]["tune_root"], "A5", "S-A5-only")
    _ok_record(camp_a5, "c" * 64)
    for extra in ([], ["--search", "A9"]):
        rc, out = _cli(root, path, *extra)
        assert rc == 2 and "REFUSED:" in out and "A1, A2, A5" in out, (extra, out)
    rc, out = _cli(root, path, "--search", "A5")
    assert rc == 0, out
    assert _last(out).startswith("STOP REQUESTED: search A5 (S-A5-only); kill running: no;"), out
    assert os.path.isfile(os.path.join(camp_a5, "STOP_REQUEST.json"))
    # only that search's folder holds a request
    found = glob.glob(os.path.join(d["paths"]["tune_root"], "**", "STOP_REQUEST.json"),
                      recursive=True)
    assert found == [os.path.join(camp_a5, "STOP_REQUEST.json")], found
    # another search of the list has no completed trial: refused
    rc, out = _cli(root, path, "--search", "A2")
    assert rc == 2 and "nothing to fit yet" in out, out


# ------------------------------------------------------------------ O30.7 ---

def test_status_line(root):
    path, d, camp = _null_cfg(root)
    cfg = C.load_config(path)
    assert SST.status_line(cfg) is None
    _ok_record(camp, "a" * 64)
    assert _cli(root, path, "--kill-running")[0] == 0
    req = json.load(open(os.path.join(camp, "STOP_REQUEST.json")))
    assert SST.status_line(cfg) == "stop requested %s (kill running: yes)" % req["requested_utc"]
    U.write_json(os.path.join(camp, "SEARCH_COMPLETE.json"), {"ended_by": "rounds"})
    assert SST.status_line(cfg) == "stop requested %s (kill running: yes)" % req["requested_utc"]
    U.write_json(os.path.join(camp, "SEARCH_COMPLETE.json"), {"ended_by": "user_stop"})
    assert SST.status_line(cfg) == ("stop requested %s (kill running: yes); the search ended "
                                    "(user_stop)" % req["requested_utc"])


# ------------------------------------------------------- O30.5, O30.6 ---

class FakePBS(object):
    """The stage's recorder: statuses by job id, every call recorded; a
    submission is an error (nothing may be submitted under a request)."""

    def __init__(self, states, allow_submit=False):
        self.states = dict(states)       # job id -> (state letter, exit status)
        self.deleted, self.submitted, self.lines = [], [], []
        self.allow_submit = allow_submit
        self.n = 0

    def log(self, text):
        self.lines.append(str(text))

    def status(self, jid):
        state, ex = self.states[jid]
        return PBS.JobStatus(jid, state, exit_status=ex, walltime_s=10.0, exec_host="nodeA")

    def delete(self, jid):
        self.deleted.append(jid)
        return True

    def fits(self, n):
        return True

    def check(self, n):
        return True

    def submit(self, argv, role, log_path=None, **kw):
        if not self.allow_submit:
            raise AssertionError("submitted under a stop request: %s" % role)
        self.n += 1
        jid = "%d.srv" % (900 + self.n)
        self.submitted.append((jid, role))
        self.states[jid] = ("Q", None)
        return jid


def _stub_tuner(camp_holder):
    """CM.run_tuner's stand-in for `finalists`: the lowest-nll ok record of the
    ledger (the tuner's ranking is block 7's, unchanged by block 30)."""
    def run_tuner(cfg, args, log_path=None, timeout=None):
        assert args[0] == "finalists", args
        camp = camp_holder[0]
        best = None
        for p in sorted(glob.glob(os.path.join(camp, "trials", "*.json"))):
            rec = json.load(open(p))
            if rec.get("status", "ok") == "ok" and (best is None or rec["nll"] < best["nll"]):
                best = rec
        U.write_json(os.path.join(camp, "finalists.json"),
                     {"finalists": [{"trial_id": best["trial_id"], "nll_rank_split": best["nll"],
                                     "config": best.get("config", {})}]})
        return ""
    return run_tuner


def _search_setup(root, monkeypatch, kill, trials):
    """A search in phase rounds of the giulia-like configuration: `trials`
    {tid: (status, job id or None)}; a stop request standing (kill_running
    `kill`, None for no request). (cfg, stage, st, camp)."""
    path, d, camp = _null_cfg(root)
    cfg = C.load_config(path)
    stage = SS.SearchStage()
    ident = dict(SS.config_identity(cfg))
    ident.update({"split_hash": "h" * 64, "contract_digest": "c" * 64, "n_train": 10,
                  "n_sel": 2, "n_report": 2, "n_rows": 14, "n_shards": 1, "W": 8,
                  "d_theta": int(d["search"]["shape_anchors"]["d_theta"]), "latent_kind": None})
    sub = {"phase": "rounds", "round": 0, "identity": ident, "gate": {"verdict": {"ok": True}},
           "clock_start_utc": CM.utc_now(), "ended_by": None, "trials": {},
           "rounds": [{"round": 0, "trial_ids": sorted(trials)}]}
    jobs = []
    for tid, (status, jid) in trials.items():
        tr = {"round": 0, "jobs": [jid] if jid else [], "resubmits": 0, "status": status,
              "fault_checked": True}
        sub["trials"][tid] = tr
        if jid:
            jobs.append({"id": jid, "role": "trial " + tid[:12], "cmd": [], "finished": False,
                         "exit_status": None, "walltime_s": None, "ncpus": 48,
                         "log": os.path.join(camp, "jobs", tid + ".0.log")})
        if status == "waiting" or status == "running":
            U.write_json(os.path.join(camp, "pending", tid + ".json"),
                         {"trial_id": tid, "campaign": "S-A1", "config": {"lr": 1e-3}})
    os.makedirs(os.path.join(camp, "trials"), exist_ok=True)
    os.makedirs(os.path.join(camp, "jobs"), exist_ok=True)
    U.write_json(os.path.join(camp, "pipeline_rounds.json"),
                 {"schema": "pipeline_rounds/1", "identity": ident, "rounds": sub["rounds"]})
    st = {"status": "running", "jobs": jobs, "resubmits": 0, "note": "", "sub": sub}
    if kill is not None:
        U.write_json(os.path.join(camp, "STOP_REQUEST.json"),
                     {"schema": "search_stop/1", "requested_utc": "2026-11-17T10:00:00Z",
                      "kill_running": kill, "user": "u", "host": "h"})
    monkeypatch.setattr(CM, "run_tuner", _stub_tuner([camp]))
    return cfg, stage, st, camp


T = dict((k, k * 64) for k in "abcdefg")


def test_stage_without_kill(root, monkeypatch):
    cfg, stage, st, camp = _search_setup(root, monkeypatch, False, {
        T["a"]: ("ok", None), T["b"]: ("ok", None),
        T["c"]: ("running", "101.srv"), T["d"]: ("running", "102.srv"),
        T["e"]: ("waiting", None), T["f"]: ("waiting", None)})
    _ok_record(camp, T["a"], 1.0)
    _ok_record(camp, T["b"], 0.5)
    pbs = FakePBS({"101.srv": ("R", None), "102.srv": ("R", None)})
    assert stage.poll(cfg, st, pbs) == "running"
    assert pbs.deleted == [] and pbs.submitted == []
    seen = [ln for ln in pbs.lines if ln.startswith("search: stop requested")]
    assert seen == ["search: stop requested 2026-11-17T10:00:00Z (kill running: no): no new "
                    "trial; 2 running"], pbs.lines
    # the next poll: c ended with its record, d ended without one
    _ok_record(camp, T["c"], 0.7)
    pbs.states.update({"101.srv": ("F", 0), "102.srv": ("F", 271)})
    pbs.lines = []
    assert stage.poll(cfg, st, pbs) == "finished"
    assert not [ln for ln in pbs.lines if ln.startswith("search: stop requested")], pbs.lines
    tr = st["sub"]["trials"]
    assert tr[T["c"]]["status"] == "ok"
    assert tr[T["d"]]["status"] == "killed"
    rec = json.load(open(os.path.join(camp, "trials", T["d"] + ".json")))
    assert rec["status"] == "killed" and \
        "ended without a record after a stop request" in rec["error"], rec
    assert tr[T["e"]]["status"] == tr[T["f"]]["status"] == "cancelled"
    assert pbs.submitted == [] and pbs.deleted == []
    sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
    assert sc["ended_by"] == "user_stop"
    assert sc["stop_request"] == json.load(open(os.path.join(camp, "STOP_REQUEST.json")))
    assert sc["n_trials"]["cancelled"] == 2 and sc["n_trials"]["ok"] == 3 and \
        sc["n_trials"]["killed"] == 1, sc["n_trials"]
    assert sc["finalist"]["trial_id"] == T["b"]                 # the best ok trial (nll 0.5)
    # the cancelled trials have no ledger record (never submitted)
    assert not os.path.exists(os.path.join(camp, "trials", T["e"] + ".json"))


def test_stage_kill_running(root, monkeypatch):
    cfg, stage, st, camp = _search_setup(root, monkeypatch, True, {
        T["a"]: ("ok", None), T["c"]: ("running", "101.srv"), T["d"]: ("running", "102.srv"),
        T["e"]: ("waiting", None)})
    _ok_record(camp, T["a"], 1.0)
    pbs = FakePBS({"101.srv": ("R", None), "102.srv": ("Q", None)})
    assert stage.poll(cfg, st, pbs) == "running"
    assert sorted(pbs.deleted) == ["101.srv", "102.srv"]
    assert st["sub"]["trials"][T["c"]].get("stop_deleted")
    # the scheduler still shows them: not deleted twice
    assert stage.poll(cfg, st, pbs) == "running"
    assert sorted(pbs.deleted) == ["101.srv", "102.srv"]
    # c ends without a record; d had written its ok record before the deletion
    _ok_record(camp, T["d"], 0.2)
    pbs.states.update({"101.srv": ("F", 271), "102.srv": ("F", 271)})
    assert stage.poll(cfg, st, pbs) == "finished"
    tr = st["sub"]["trials"]
    rec_c = json.load(open(os.path.join(camp, "trials", T["c"] + ".json")))
    assert rec_c["status"] == "killed" and \
        "stopped by the user (stop-search --kill-running)" in rec_c["error"], rec_c
    assert tr[T["d"]]["status"] == "ok"
    rec_d = json.load(open(os.path.join(camp, "trials", T["d"] + ".json")))
    assert rec_d["status"] == "ok" and rec_d["nll"] == 0.2          # kept (D-154)
    assert tr[T["e"]]["status"] == "cancelled" and pbs.submitted == []
    sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
    assert sc["ended_by"] == "user_stop" and sc["finalist"]["trial_id"] == T["d"]
    assert sc["stop_request"]["kill_running"] is True


def test_stage_node_fault_under_a_request_is_not_resubmitted(root, monkeypatch):
    cfg, stage, st, camp = _search_setup(root, monkeypatch, False, {
        T["a"]: ("ok", None), T["c"]: ("running", "101.srv")})
    _ok_record(camp, T["a"], 1.0)
    pbs = FakePBS({"101.srv": ("F", 135)})
    U.write_json(os.path.join(camp, "trials", T["c"] + ".json"),
                 {"trial_id": T["c"], "status": "failed", "error": "runner exited -7",
                  "config": {"lr": 1e-3}})
    with open(os.path.join(camp, "jobs", T["c"] + ".0.log"), "w") as fh:
        fh.write("RUN_ARGV: host nodeA\nBus error (core dumped)\nRUN_ARGV EXIT: 135\n")
    out = stage.poll(cfg, st, pbs)
    if out == "running":                       # the examination may take a second poll
        out = stage.poll(cfg, st, pbs)
    assert out == "finished", pbs.lines
    assert pbs.submitted == []
    assert st["sub"]["trials"][T["c"]]["status"] in ("cancelled", "killed"), st["sub"]["trials"]
    sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
    assert sc["ended_by"] == "user_stop"


def test_withdrawn_request_resumes_but_a_deleted_trial_stays_killed(root, monkeypatch):
    cfg, stage, st, camp = _search_setup(root, monkeypatch, True, {
        T["a"]: ("ok", None), T["c"]: ("running", "101.srv"), T["e"]: ("waiting", None)})
    _ok_record(camp, T["a"], 1.0)
    pbs = FakePBS({"101.srv": ("R", None)}, allow_submit=True)
    assert stage.poll(cfg, st, pbs) == "running"
    assert pbs.deleted == ["101.srv"] and pbs.submitted == []
    os.remove(os.path.join(camp, "STOP_REQUEST.json"))          # stop-search --cancel
    pbs.states["101.srv"] = ("F", 271)
    stage.poll(cfg, st, pbs)
    tr = st["sub"]["trials"]
    assert tr[T["c"]]["status"] == "killed"                      # not resubmitted
    assert [r for _j, r in pbs.submitted if T["c"][:12] in r] == []
    assert [r for _j, r in pbs.submitted if T["e"][:12] in r], pbs.submitted   # e submitted again
    assert any("withdrawn" in ln for ln in pbs.lines), pbs.lines


def test_stage_without_request_unchanged(root, monkeypatch):
    cfg, stage, st, camp = _search_setup(root, monkeypatch, None, {
        T["a"]: ("ok", None), T["e"]: ("waiting", None), T["f"]: ("waiting", None)})
    _ok_record(camp, T["a"], 1.0)
    pbs = FakePBS({}, allow_submit=True)
    assert stage.poll(cfg, st, pbs) == "running"
    assert len(pbs.submitted) == 2 and not any("stop" in ln for ln in pbs.lines)


# ------------------------------------------------------------------ O30.8 ---

def _report(root, path, *extra):
    spool = os.path.join(root, "spool")
    os.makedirs(spool, exist_ok=True)
    return U.run([U.PYTHON, os.path.join(U.TOOLS, "search_report.py"), "--config", path]
                 + list(extra), env=U.env_with_fake_pbs(spool), cwd=root, timeout=120)


def _state_with_trials(d, cfg, trials, jobs):
    import state as ST
    wr = d["paths"]["work_root"]
    st = ST.State.new(wr, d["dataset"], cfg.path, cfg.sha256)
    rec = st.stage("search")
    rec["status"] = "running"
    rec["jobs"] = jobs
    rec["sub"] = {"phase": "rounds", "round": 0, "trials": trials,
                  "rounds": [{"round": 0, "trial_ids": sorted(trials)}]}
    st.save()


def _nexts(out):
    """The `next: ...` field of every trial line."""
    return [m.group(1) for m in re.finditer(r"\| (next: .*)$", out, re.M)]


def test_report_under_a_request(root):
    path, d, camp = _null_cfg(root)
    cfg = C.load_config(path)
    _ok_record(camp, T["a"], 1.0)
    for t in (T["e"], T["c"]):
        U.write_json(os.path.join(camp, "pending", t + ".json"),
                     {"trial_id": t, "campaign": "S-A1", "config": {"lr": 1e-3}})
    trials = {T["a"]: {"round": 0, "jobs": ["1.srv"], "resubmits": 0, "status": "ok"},
              T["c"]: {"round": 0, "jobs": ["2.srv"], "resubmits": 0, "status": "running"},
              T["e"]: {"round": 0, "jobs": [], "resubmits": 0, "status": "waiting"},
              T["f"]: {"round": 0, "jobs": [], "resubmits": 0, "status": "cancelled"}}
    jobs = [{"id": "1.srv", "role": "t", "cmd": [], "finished": True, "exit_status": 0},
            {"id": "2.srv", "role": "t", "cmd": [], "finished": False, "exit_status": None}]
    _state_with_trials(d, cfg, trials, jobs)
    rc, out = _report(root, path)
    assert rc == 0, out
    nexts = _nexts(out)
    assert "next: nothing (cancelled by the stop request; never submitted)" in nexts, nexts
    assert "next: waits for a slot and the cores" in nexts, nexts      # no request: submitted
    # no request yet: the waiting trial is predicted to be submitted
    U.write_json(os.path.join(camp, "STOP_REQUEST.json"),
                 {"schema": "search_stop/1", "requested_utc": "2026-11-17T10:00:00Z",
                  "kill_running": False, "user": "u", "host": "h"})
    rc, out = _report(root, path)
    assert rc == 0, out
    assert "stop requested 2026-11-17T10:00:00Z (kill running: no)" in out, out
    nexts = _nexts(out)
    assert len(nexts) == 4, nexts
    assert "next: waits for a slot and the cores" not in nexts, nexts
    for ln in nexts:
        low = ln.lower()
        assert not re.search(r"\b(submitted|resubmitted) (at once|with --resume)", low) or \
            "never submitted" in low or "not resubmitted" in low, ln
    assert any(ln.startswith("next: nothing (a stop request stands:") for ln in nexts), nexts
    # the header line comes before the trials
    head = out.split("\ntrial ")[0]
    assert "stop requested 2026-11-17T10:00:00Z" in head


# ------------------------------------------------------------------ O30.9 ---

def test_dry_run_argv_lines_carry_the_replicate_flags(root):
    path, d = U.linddup_copy(root)
    spool = os.path.join(root, "spool")
    os.makedirs(spool)
    rc, out = U.run([U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "run", "--config", path,
                     "--dry-run"], env=U.env_with_fake_pbs(spool), cwd=root, timeout=300)
    assert rc == 0, out[-2000:]
    flags = "--rep-pairs within_group --rep-group-key rep_group"
    probe = [ln for ln in out.splitlines() if "--probe-steps" in ln]
    trial = [ln for ln in out.splitlines() if " evaluate --campaign" in ln]
    member = [ln for ln in out.splitlines() if "--n-posterior-draws" in ln]
    assert len(probe) == 3 and len(trial) == 3 and len(member) == 3, out[-3000:]
    for ln in probe:
        assert "--probe-steps 20 --probe-val-rows 5000 " + flags in ln, ln
    for ln in trial:
        assert "--runner-extra '%s'" % flags in ln, ln
    for ln in member:
        assert "--n-posterior-draws 512 " + flags in ln, ln


# ------------------------------------------- boundary: block 30 x the reset tool ---
# [2026-11-17_1730, revised for amendment 5] The earlier form of this test showed
# F1 (the request left behind). Block 30's amendment 5: the reset tool lists the
# request (its size in the CODE), moves it into <campaign>_set_aside_<utc>/ and
# names it in RESET_RECORD.json; the reset search proposes round 0 as before.


class _Proposed(Exception):
    """Raised by the tuner's stand-in when the stage proposes a round."""


def test_reset_after_a_stopped_search_restarts_it(root, monkeypatch):
    """Block 7's reset tool after a search ended by stop-search (block 30,
    amendment 5): the request is a listing item, moved aside with its bytes,
    named in RESET_RECORD.json; the reset search proposes round 0 anew."""
    import hashlib
    path, d, camp = _null_cfg(root)
    _ok_record(camp, T["a"], 1.0)
    req = {"schema": "search_stop/1", "requested_utc": "2026-11-17T10:00:00Z",
           "kill_running": False, "user": "u", "host": "h"}
    for name, obj in (("pipeline_rounds.json", {"rounds": []}), ("finalists.json", {}),
                      ("SEARCH_COMPLETE.json", {"ended_by": "user_stop"}),
                      ("STOP_REQUEST.json", req)):
        U.write_json(os.path.join(camp, name), obj)
    req_path = os.path.join(camp, "STOP_REQUEST.json")
    req_sha = hashlib.sha256(open(req_path, "rb").read()).hexdigest()
    req_size = os.path.getsize(req_path)
    spool = os.path.join(root, "spool")
    os.makedirs(spool)
    env = U.env_with_fake_pbs(spool)
    tool = os.path.join(U.TOOLS, "search_reset.py")
    rc, out = U.run([U.PYTHON, tool, "--config", path], env=env, cwd=root)
    assert rc == 0, out
    # the listing names the request among the items it would set aside
    assert re.search(r"would set aside STOP_REQUEST\.json\s+1 file\(s\)", out), out
    code = re.search(r"^CODE ([0-9a-f]{8})", out, re.M).group(1)
    rc, out = U.run([U.PYTHON, tool, "--config", path, "--confirm", code], env=env, cwd=root)
    assert rc == 0 and "SEARCH RESET DONE" in out, out
    assert not os.path.exists(os.path.join(camp, "SEARCH_COMPLETE.json"))
    assert not os.path.exists(req_path), "the reset left the stop request in the campaign"
    asides = sorted(glob.glob(camp.rstrip("/") + "_set_aside_*"))
    assert len(asides) == 1, asides
    moved = os.path.join(asides[0], "STOP_REQUEST.json")
    assert os.path.getsize(moved) == req_size
    assert hashlib.sha256(open(moved, "rb").read()).hexdigest() == req_sha
    rec = json.load(open(os.path.join(asides[0], "RESET_RECORD.json")))
    assert "STOP_REQUEST.json" in rec["moved"], rec
    # the stage after the reset: no request stands, so its poll proposes round 0
    cfg, stage, st, camp2 = _search_setup(root, monkeypatch, None, {})
    assert camp2 == camp
    st["sub"]["rounds"] = []
    calls = []

    def run_tuner(cfg_, args, log_path=None, timeout=None):
        calls.append(list(args))
        if args[0] == "propose":
            raise _Proposed(args)
        raise AssertionError("unexpected tuner call %r" % (args,))
    monkeypatch.setattr(CM, "run_tuner", run_tuner)
    pbs = FakePBS({}, allow_submit=True)
    with pytest.raises(_Proposed):
        stage.poll(cfg, st, pbs)
    assert calls and calls[0][0] == "propose", calls
    a = calls[0]
    # round 0 (block 7: n_initial_points configurations, --gp-seed gp_seed + 0)
    assert a[a.index("--n-points") + 1] == str(d["search"]["n_initial_points"]), a
    assert a[a.index("--gp-seed") + 1] == str(d["search"]["gp_seed"]), a
    assert not [ln for ln in pbs.lines if "stop requested" in ln], pbs.lines
    assert st["sub"].get("ended_by") in (None, ""), st["sub"].get("ended_by")
