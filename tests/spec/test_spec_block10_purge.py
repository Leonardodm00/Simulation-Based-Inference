"""Spec-derived tests of block 10, the purge tool `tools/purge_outputs.py`
(SPEC block 10; block 24 only where it states that block 10's behaviour
is unchanged and where it refuses block 10's kinds on a synthetic
configuration). t1-A run of 2026-11-13 (D-131).

A recorded project (a copy of giulia.json with every path under a temporary
folder) with its search, set-aside rounds, best fit, simulated bank and
quarantine, the kept real bank, plan folder and work root, and decoys (an
other campaign's folder, another data set's bank). The scheduler is the
fake PBS of hpc/pipeline/tests/fake_pbs on PATH (a fixture, sec. 3.3).
Expected values are the spec's: the targets and their order, the kept
paths, exit codes 0 / 2 / 4 / 5, the record, the pass line.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import pytest

from conftest import FAKE_PBS, GIULIA, PIPE, SYNTH

TOOL = os.path.join(PIPE, "tools", "purge_outputs.py")


def _w(path, data="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(data)


def _sha_tree(root):
    out = {}
    for d, _s, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            if os.path.islink(p):
                out[os.path.relpath(p, root)] = "link:" + os.readlink(p)
            else:
                out[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


@pytest.fixture
def project(tmp_path):
    raw = json.load(open(GIULIA))
    base = tmp_path / "dv"
    for k in ("sbix_dir", "work_root", "banks_root", "runs_root", "tune_root"):
        raw["paths"][k] = str(base / k)
    cfgp = tmp_path / "proj.json"
    cfgp.write_text(json.dumps(raw, indent=1))
    P = raw["paths"]
    tune, runs, banks = P["tune_root"], P["runs_root"], P["banks_root"]
    # targets
    camp = os.path.join(tune, "S-A1")
    _w(os.path.join(camp, "trials", "a.json"), json.dumps({"status": "ok", "nll": 1.0}))
    _w(os.path.join(camp, "trials", "b.json"), json.dumps({"status": "failed"}))
    _w(os.path.join(camp, "runs", "a", "A1_seed0.json"), "{}")
    _w(os.path.join(camp, "SEARCH_COMPLETE.json"), "{}")
    # set-aside folders made in the reverse of their name order
    _w(os.path.join(tune, "S-A1_set_aside_20261101T000000Z", "trials", "c.json"), "{}")
    time.sleep(0.01)
    _w(os.path.join(tune, "S-A1_set_aside_20261020T000000Z", "trials", "d.json"), "{}")
    _w(os.path.join(runs, "giulia_best", "BEST_FIT_COMPLETE.json"), "{}")
    rec = os.path.join(banks, "giulia_sim", "BANK_RECORD.json")
    _w(rec, "{}")
    _w(rec + ".sha256", hashlib.sha256(b"{}").hexdigest() + "  BANK_RECORD.json\n")
    _w(os.path.join(banks, "giulia_sim", "shard_0000.npz"), "z" * 100)
    _w(os.path.join(banks, "giulia_sim_quarantine", "20261018_000000_round0", "shard_0001.npz"), "q")
    # kept
    _w(os.path.join(banks, "giulia_real", "shard_0000.npz"), "r")
    _w(os.path.join(P["sbix_dir"], "export_windows", "giulia", "bank_plan.json"), "{}")
    _w(os.path.join(P["work_root"], "pipeline.log"), "log\n")
    # decoys: another campaign, another data set's bank, a name that only
    # starts like a set-aside folder
    _w(os.path.join(tune, "S-A2", "trials", "e.json"), "{}")
    _w(os.path.join(banks, "other_sim", "shard_0000.npz"), "o")
    _w(os.path.join(tune, "S-A1_set_aside", "keep.txt"), "k")
    env = dict(os.environ)
    env["PATH"] = FAKE_PBS + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = str(tmp_path / "spool")
    env["FAKE_PBS_DELAY"] = "0.1"
    return {"cfg": str(cfgp), "raw": raw, "base": str(base), "env": env, "camp": camp,
            "tune": tune, "runs": runs, "banks": banks}


def _tool(project, *args, cfg=None):
    r = subprocess.run([sys.executable, TOOL, "--config", cfg or project["cfg"]] + list(args),
                       env=project["env"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=300)
    return r.returncode, r.stdout


def _code(out):
    m = re.findall(r"^CODE ([0-9a-f]{8}) ", out, re.M)
    assert len(m) == 1, out
    return m[0]


ALL = "search,best_fit,sim_bank"


def _targets_in_order(project):
    t, r, b = project["tune"], project["runs"], project["banks"]
    return [os.path.join(t, "S-A1"),
            os.path.join(t, "S-A1_set_aside_20261020T000000Z"),
            os.path.join(t, "S-A1_set_aside_20261101T000000Z"),
            os.path.join(r, "giulia_best"),
            os.path.join(b, "giulia_sim"),
            os.path.join(b, "giulia_sim_quarantine")]


def test_listing_names_targets_in_order_and_deletes_nothing(project):
    before = _sha_tree(project["base"])
    rc, out = _tool(project, "--what", ALL)
    assert rc == 0, out
    dele = re.findall(r"^  would DELETE \[(\w+)\] (\S+?):", out, re.M)
    assert [p for _k, p in dele] == _targets_in_order(project)
    assert [k for k, _p in dele] == ["search"] * 3 + ["best_fit"] + ["sim_bank"] * 2
    kept = re.findall(r"^  kept: (\S+)", out, re.M)
    P = project["raw"]["paths"]
    assert kept == [os.path.join(project["banks"], "giulia_real"),
                    os.path.join(P["sbix_dir"], "export_windows", "giulia"), P["work_root"]]
    assert "ok 1" in out and "failed 1" in out                        # the ledger's counts
    assert "record sha256 %s" % hashlib.sha256(b"{}").hexdigest()[:16] in out
    _code(out)
    assert _sha_tree(project["base"]) == before


def test_code_is_stable_and_stale_on_a_change(project):
    """Block 10: the CODE is the same twice, and changes when a file is
    added, removed, resized or rewritten in a target -- not when a kept
    file changes."""
    c0 = _code(_tool(project, "--what", ALL)[1])
    assert _code(_tool(project, "--what", ALL)[1]) == c0
    # a kept file rewritten: no change
    _w(os.path.join(project["banks"], "giulia_real", "shard_0000.npz"), "rr" * 50)
    assert _code(_tool(project, "--what", ALL)[1]) == c0
    # added
    _w(os.path.join(project["camp"], "runs", "b", "x.json"), "{}")
    c1 = _code(_tool(project, "--what", ALL)[1])
    assert c1 != c0
    # resized
    _w(os.path.join(project["banks"], "giulia_sim", "shard_0000.npz"), "z" * 101)
    c2 = _code(_tool(project, "--what", ALL)[1])
    assert c2 not in (c0, c1)
    # removed
    os.remove(os.path.join(project["camp"], "runs", "b", "x.json"))
    c3 = _code(_tool(project, "--what", ALL)[1])
    assert c3 not in (c1, c2)
    # rewritten (same size, a later mtime in whole seconds)
    p = os.path.join(project["runs"], "giulia_best", "BEST_FIT_COMPLETE.json")
    st = os.stat(p)
    os.utime(p, (st.st_atime, st.st_mtime + 5))
    assert _code(_tool(project, "--what", ALL)[1]) != c3


def test_confirm_deletes_exactly_the_targets(project):
    before = _sha_tree(project["base"])
    code = _code(_tool(project, "--what", ALL)[1])
    rc, out = _tool(project, "--what", ALL, "--confirm", code)
    assert rc == 0, out
    for p in _targets_in_order(project):
        assert not os.path.lexists(p), p
    after = _sha_tree(project["base"])
    gone = set(before) - set(after)
    for rel in gone:                                 # only files under a target went
        assert any(os.path.join(project["base"], rel).startswith(t + os.sep)
                   for t in _targets_in_order(project)), rel
    for rel in after:                                # every other file unchanged
        if not rel.startswith(os.path.join("work_root", "purge")):
            assert after[rel] == before[rel], rel
    assert len(re.findall(r"^DELETED \[", out, re.M)) == 6
    last = [l for l in out.strip().split("\n") if l.startswith("PURGE DONE:")]
    assert last and last[0].startswith("PURGE DONE: 6 folder(s) deleted, ") \
        and last[0].endswith("; the state file is not touched")
    recs = os.listdir(os.path.join(project["raw"]["paths"]["work_root"], "purge"))
    assert len(recs) == 1 and re.match(r"^PURGE_.*\.json$", recs[0])
    rec = json.load(open(os.path.join(project["raw"]["paths"]["work_root"], "purge", recs[0])))
    assert rec["schema"] == "purge_outputs/1" and rec["status"] == "done" and rec["code"] == code
    assert [t["result"] for t in rec["targets"]] == ["deleted"] * 6
    rc2, out2 = _tool(project, "--what", ALL)
    # the four fixed targets (campaign, best fit, bank, quarantine) are absent;
    # the set-aside folders are found by name and are gone
    assert rc2 == 0 and len(re.findall(r"^  absent ", out2, re.M)) == 4
    assert "would DELETE" not in out2


def test_wrong_code_refused(project):
    before = _sha_tree(project["base"])
    rc, out = _tool(project, "--what", ALL, "--confirm", "00000000")
    assert rc == 4 and "PURGE REFUSED" in out and "CODE " in out
    assert _sha_tree(project["base"]) == before


def test_lock_refused(project):
    wr = project["raw"]["paths"]["work_root"]
    _w(os.path.join(wr, "pipeline.lock"), "job:9.fake host 2026-11-13T00:00:00Z\n")
    before = _sha_tree(project["base"])
    code = "deadbeef"
    rc, out = _tool(project, "--what", ALL, "--confirm", code)
    assert rc == 4 and "PURGE REFUSED" in out
    assert _sha_tree(project["base"]) == before


def test_symbolic_link_target_refused(project, tmp_path):
    camp = project["camp"]
    real = str(tmp_path / "elsewhere")
    os.rename(camp, real)
    os.symlink(real, camp)
    before = _sha_tree(real)
    rc, out = _tool(project, "--what", "search")
    assert rc == 4 and "symbolic link" in out
    assert _sha_tree(real) == before and os.path.islink(camp)


def test_target_not_a_directory_refused(project):
    p = os.path.join(project["runs"], "giulia_best")
    import shutil
    shutil.rmtree(p)
    _w(p, "a file where the folder should be")
    rc, out = _tool(project, "--what", "best_fit")
    assert rc == 4 and "not a directory" in out


@pytest.mark.parametrize("what", ["", ",", "bogus", "search,bogus"])
def test_bad_what_exit2(project, what):
    before = _sha_tree(project["base"])
    rc, out = _tool(project, "--what", what)
    assert rc == 2
    assert _sha_tree(project["base"]) == before


def test_config_that_does_not_load_exit2(project, tmp_path):
    bad = tmp_path / "bad.json"
    raw = dict(project["raw"])
    raw["config_version"] = 9
    bad.write_text(json.dumps(raw))
    rc, out = _tool(project, "--what", ALL, cfg=str(bad))
    assert rc == 2


def test_block10_kinds_refused_on_a_synthetic_configuration(project):
    """Block 24 (block 10's tool): block 10's kinds are refused (exit 2) for
    a configuration of kind synthetic."""
    rc, out = _tool(project, "--what", "search", cfg=SYNTH)
    assert rc == 2


def _qsub(project, name, vars_=None, seconds=60):
    script = os.path.join(project["base"], "sleep.sh")
    _w(script, "#!/bin/bash\nsleep %d\n" % seconds)
    argv = ["qsub", "-N", name, "-o", os.path.join(project["base"], name + ".log")]
    if vars_:
        argv += ["-v", vars_]
    argv.append(script)
    r = subprocess.run(argv, env=project["env"], stdout=subprocess.PIPE, universal_newlines=True)
    assert r.returncode == 0
    return r.stdout.split()[0]


def _qdel(project, jid):
    subprocess.run(["qdel", jid], env=project["env"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_sequencer_job_refused(project):
    jid = _qsub(project, "pl_giulia")
    try:
        rc, out = _tool(project, "--what", ALL)
        assert rc == 4 and "sequencer job %s" % jid in out
    finally:
        _qdel(project, jid)


def test_job_naming_the_chain_refused_another_chain_not(project):
    """Block 10: a job of the user whose qstat -f names the chain
    (CHAIN_NAME=pl_<tag> followed by a comma or the end of a line, the
    wrapped lines joined) is not finished -> refused; a job naming another
    chain (pl_giulia2) does not refuse."""
    j1 = _qsub(project, "trial_x", "ARGV_FILE=/a,CHAIN_NAME=pl_giulia2,ROLE=stage")
    try:
        rc, out = _tool(project, "--what", ALL)
        assert rc == 0, out
    finally:
        _qdel(project, j1)
    j2 = _qsub(project, "trial_y", "ARGV_FILE=/a,CHAIN_NAME=pl_giulia,ROLE=stage")
    try:
        rc, out = _tool(project, "--what", ALL)
        assert rc == 4 and j2.split(".")[0] in out
    finally:
        _qdel(project, j2)


def test_job_the_state_records_refused(project):
    """Block 10: a job that any stage of the state records is not finished
    per the scheduler -> refused (exit 4)."""
    jid = _qsub(project, "other_name")
    sys.path.insert(0, PIPE)
    from state import State
    wr = project["raw"]["paths"]["work_root"]
    st = State.new(wr, "giulia", project["cfg"], "0" * 64)
    st.add_job("search", jid, "trial", ["qsub", "x"])
    st.save()
    try:
        rc, out = _tool(project, "--what", ALL)
        assert rc == 4 and jid in out
    finally:
        _qdel(project, jid)
