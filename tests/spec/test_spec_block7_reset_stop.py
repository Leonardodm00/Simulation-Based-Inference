"""Block 7's reset tool after block 30's amendment 5 (specs/SPEC.md block 7,
`tools/search_reset.py`; block 30, amendment 5).

Oracles (from the spec, before the code was read; tester run 2026-11-17_1730):
  O30r.1 STOP_REQUEST.json is a listing item when present (with its size) and
         enters the CODE: the CODE changes when a request appears.
  O30r.2 without a request the listing and its CODE are the base commit's
         (a915950's tool, same tree, same scheduler): nothing else changed.
  O30r.3 --confirm moves the request into <campaign>_set_aside_<utc>/ with its
         bytes, RESET_RECORD.json names it, the campaign keeps gate/ and
         bank_split.json and no request; a second listing no longer lists it.
  O30r.4 a request written between the listing and --confirm makes the CODE
         stale: exit 4, nothing moved.
  O30r.5 a list configuration: --search <q> acts on q's campaign folder only.
  O30r.6 through the commands: stop-search writes the request, the reset sets
         it aside, `stop-search --cancel` then finds none (exit 2).
"""

import glob
import hashlib
import json
import os
import re
import shutil

import pytest

import _spec_util as U

BASE_TOOLS = "/tmp/base_a915950/hpc/pipeline/tools"


@pytest.fixture
def root():
    r = U.tmpdir("spec_reset_")
    yield r
    shutil.rmtree(r, ignore_errors=True)


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _campaign(d, view=None):
    if view is None:
        return os.path.join(d["paths"]["tune_root"], d["search"]["campaign"])
    q = [s for s in d["searches"] if s["name"] == view][0]
    return os.path.join(d["paths"]["tune_root"], view, q["campaign"])


def _populate(camp, with_request=True, nll=1.0, tid="a" * 64):
    """A finished search folder: one ok record, the rounds, the finalists,
    SEARCH_COMPLETE (user_stop), the gate's probe file, bank_split.json, a job
    log, and (optionally) the stop request."""
    U.write_json(os.path.join(camp, "trials", tid + ".json"),
                 {"trial_id": tid, "status": "ok", "nll": nll, "config": {"lr": 1e-3}})
    U.write_json(os.path.join(camp, "pipeline_rounds.json"), {"rounds": [{"round": 0}]})
    U.write_json(os.path.join(camp, "finalists.json"), {"finalists": []})
    U.write_json(os.path.join(camp, "SEARCH_COMPLETE.json"), {"ended_by": "user_stop"})
    U.write_json(os.path.join(camp, "gate", "b16", "probe.json"), {"ok": True})
    U.write_json(os.path.join(camp, "bank_split.json"), {"split_hash": "h"})
    os.makedirs(os.path.join(camp, "jobs"), exist_ok=True)
    with open(os.path.join(camp, "jobs", tid + ".0.log"), "w") as fh:
        fh.write("RUN_ARGV EXIT: 0\n")
    if with_request:
        U.write_json(os.path.join(camp, "STOP_REQUEST.json"),
                     {"schema": "search_stop/1", "requested_utc": "2026-11-17T10:00:00Z",
                      "kill_running": False, "user": "u", "host": "h"})


def _tool(root, path, *extra, tools=U.TOOLS):
    spool = os.path.join(root, "spool")
    os.makedirs(spool, exist_ok=True)
    return U.run([U.PYTHON, os.path.join(tools, "search_reset.py"), "--config", path]
                 + list(extra), env=U.env_with_fake_pbs(spool), cwd=root, timeout=300)


def _code(out):
    return re.search(r"^CODE ([0-9a-f]{8})", out, re.M).group(1)


def test_listing_names_the_request_and_its_code_changes(root):
    path, d = U.giulia_copy(root)
    camp = _campaign(d)
    _populate(camp, with_request=False)
    rc, out0 = _tool(root, path)
    assert rc == 0 and "STOP_REQUEST.json" not in out0, out0
    U.write_json(os.path.join(camp, "STOP_REQUEST.json"),
                 {"schema": "search_stop/1", "requested_utc": "2026-11-17T10:00:00Z",
                  "kill_running": True, "user": "u", "host": "h"})
    rc, out1 = _tool(root, path)
    assert rc == 0, out1
    size = os.path.getsize(os.path.join(camp, "STOP_REQUEST.json"))
    m = re.search(r"would set aside STOP_REQUEST\.json\s+1 file\(s\)\s+([0-9.]+) MB", out1)
    assert m and abs(float(m.group(1)) - size / 1e6) < 0.05 + 1e-9, out1   # printed to 0.1 MB
    assert _code(out1) != _code(out0)
    # the rest of the listing is unchanged by the request
    strip = lambda o: [l for l in o.splitlines() if "STOP_REQUEST" not in l and not l.startswith("CODE")]
    assert strip(out1) == strip(out0)


def test_without_a_request_the_listing_is_the_base_commits(root):
    if not os.path.isdir(BASE_TOOLS):
        pytest.skip("no base tree (git archive a915950 hpc | tar -x -C /tmp/base_a915950)")
    path, d = U.giulia_copy(root)
    camp = _campaign(d)
    _populate(camp, with_request=False)
    rc_h, out_h = _tool(root, path)
    rc_b, out_b = _tool(root, path, tools=BASE_TOOLS)
    assert rc_h == rc_b == 0, (out_h, out_b)
    assert out_h == out_b, (out_h, out_b)


def test_confirm_sets_the_request_aside_with_its_bytes(root):
    path, d = U.giulia_copy(root)
    camp = _campaign(d)
    _populate(camp, with_request=True)
    before = dict((os.path.relpath(p, camp), _sha(p))
                  for p in glob.glob(os.path.join(camp, "**", "*"), recursive=True) if os.path.isfile(p))
    rc, out = _tool(root, path)
    rc, out = _tool(root, path, "--confirm", _code(out))
    assert rc == 0 and "SEARCH RESET DONE: 0 job(s) deleted" in out, out
    assert not os.path.exists(os.path.join(camp, "STOP_REQUEST.json"))
    asides = glob.glob(camp + "_set_aside_*")
    assert len(asides) == 1
    aside = asides[0]
    after = {}
    for base in (camp, aside):
        for p in glob.glob(os.path.join(base, "**", "*"), recursive=True):
            if os.path.isfile(p) and not p.endswith("RESET_RECORD.json"):
                after[os.path.relpath(p, base)] = _sha(p)
    assert after == before                       # every file kept, bytes unchanged (D-095)
    rec = json.load(open(os.path.join(aside, "RESET_RECORD.json")))
    assert set(rec["moved"]) == {"trials", "jobs", "pipeline_rounds.json", "finalists.json",
                                 "SEARCH_COMPLETE.json", "STOP_REQUEST.json"}, rec["moved"]
    assert sorted(os.listdir(camp)) == ["bank_split.json", "gate"], os.listdir(camp)
    rc, out2 = _tool(root, path)
    assert rc == 0 and "STOP_REQUEST" not in out2 and "nothing to set aside" in out2, out2


def test_a_request_written_after_the_listing_makes_the_code_stale(root):
    path, d = U.giulia_copy(root)
    camp = _campaign(d)
    _populate(camp, with_request=False)
    rc, out = _tool(root, path)
    code = _code(out)
    snap = U.tree_snapshot(camp)
    U.write_json(os.path.join(camp, "STOP_REQUEST.json"),
                 {"schema": "search_stop/1", "requested_utc": "2026-11-17T11:00:00Z",
                  "kill_running": False, "user": "u", "host": "h"})
    snap = U.tree_snapshot(camp)
    rc, out = _tool(root, path, "--confirm", code)
    assert rc == 4 and "SEARCH RESET REFUSED" in out, out
    assert U.tree_snapshot(camp) == snap and not glob.glob(camp + "_set_aside_*")


def test_a_list_configuration_resets_only_the_named_search(root):
    path, d = U.linddup_copy(root)
    a1, a5 = _campaign(d, "A1"), _campaign(d, "A5")
    _populate(a1, with_request=True)
    _populate(a5, with_request=True, tid="b" * 64)
    snap_a1 = U.tree_snapshot(a1)
    rc, out = _tool(root, path, "--search", "A5")
    assert rc == 0 and a5 in out and "would set aside STOP_REQUEST.json" in out, out
    rc, out = _tool(root, path, "--search", "A5", "--confirm", _code(out))
    assert rc == 0 and "--retry search_A5" in out, out
    assert not os.path.exists(os.path.join(a5, "STOP_REQUEST.json"))
    assert len(glob.glob(a5 + "_set_aside_*")) == 1 and not glob.glob(a1 + "_set_aside_*")
    assert U.tree_snapshot(a1) == snap_a1


def test_through_the_commands(root):
    """stop-search writes the request; the reset sets it aside; --cancel then
    has nothing to withdraw (block 30: refused, exit 2, nothing written)."""
    path, d = U.giulia_copy(root)
    camp = _campaign(d)
    U.write_json(os.path.join(camp, "trials", "c" * 64 + ".json"),
                 {"trial_id": "c" * 64, "status": "ok", "nll": 2.0, "config": {}})
    spool = os.path.join(root, "spool")
    os.makedirs(spool, exist_ok=True)
    env = U.env_with_fake_pbs(spool)
    stop = [U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "stop-search", "--config", path]
    rc, out = U.run(stop, env=env, cwd=root)
    assert rc == 0 and "STOP REQUESTED:" in out.strip().splitlines()[-1], out
    req = os.path.join(camp, "STOP_REQUEST.json")
    sha = _sha(req)
    rc, out = _tool(root, path)
    rc, out = _tool(root, path, "--confirm", _code(out))
    assert rc == 0, out
    moved = glob.glob(os.path.join(camp + "_set_aside_*", "STOP_REQUEST.json"))
    assert len(moved) == 1 and _sha(moved[0]) == sha
    rc, out = U.run(stop + ["--cancel"], env=env, cwd=root)
    assert rc == 2 and out.strip().splitlines()[-1].startswith("REFUSED:"), out
    assert not os.path.exists(req)
