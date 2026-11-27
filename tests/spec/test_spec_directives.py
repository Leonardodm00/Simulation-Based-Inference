"""Directive audit of the changed files (tester run 2026-11-17_1730 on 6c6c297).

  D.1 swappability: one value of the configuration changed (the pair rule, a
      search's core cap, the bank's partial-topology rule) and the dry run still
      exits 0, its lines following the new values (block 29's caps, block 31's
      runner_extra, block 2's profile).
  D.2 sec. 4: the changed sequencer modules are pure ASCII, LF, Python-3.6
      syntax (vermin, when installed); no project value in the code (block 1's
      agnostic grep over hpc/pipeline outside projects/ and tests/).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess

import pytest

import _spec_util as U

CHANGED = ["hpc/pipeline/stages/bank.py", "hpc/pipeline/config.py",
           "hpc/pipeline/stages/synthetic.py", "hpc/pipeline/tools/search_reset.py",
           "hpc/joint/stage3/run_joint_arms.py", "hpc/pipeline/README.md",
           "hpc/pipeline/tests/test_smoke_block2.py", "hpc/pipeline/tests/test_smoke_reset.py",
           "hpc/pipeline/tests/test_smoke_identity.py"]


@pytest.fixture
def root():
    r = U.tmpdir("spec_dir_")
    yield r
    shutil.rmtree(r, ignore_errors=True)


def _dry(root, mutate):
    path, d = U.linddup_copy(root, mutate)
    spool = os.path.join(root, "spool")
    os.makedirs(spool, exist_ok=True)
    rc, out = U.run([U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "run", "--config", path,
                     "--dry-run"], env=U.env_with_fake_pbs(spool), cwd=root, timeout=300)
    return rc, out, d


def test_swappability_of_the_pair_rule_and_a_cap(root):
    def mutate(d):
        d["replicate"] = {"pairs": "cross_well", "group_key": "donor"}
        d["searches"][1]["max_cores"] = 96
        d["bank"]["drop_partial_topologies"] = False
    rc, out, d = _dry(root, mutate)
    assert rc == 0 and not re.search(r"REFUSE|STOP|FAILED", out), out[-3000:]
    argv = [l for l in out.splitlines() if "argv:" in l and "--rep-pairs" in l]
    assert len(argv) == 9 and all("--rep-pairs cross_well --rep-group-key donor" in l for l in argv)
    assert "cores of search A2: at most 96 (searches[].max_cores)" in out
    a2 = [l for l in out.splitlines() if "the jobs of search_A2, best_fit_A2 share at most 96 cores" in l]
    assert len(a2) == 2 and all(l.endswith("at most 2 job(s) of 48 cores at once") for l in a2), a2
    assert "cores of search A1: at most 148" in out


def test_profile_follows_the_partial_topology_rule(root):
    import config as C
    import stages as S
    import stages.bank as B
    for flag in (True, False):
        path, d = U.linddup_copy(root, lambda x: x["bank"].__setitem__("drop_partial_topologies", flag),
                                 name="ld_%s.json" % flag)
        cfg = C.load_config(path)
        for name in ("real_extraction", "sim_redetection", "label_freeze"):
            p = S.artefact_of(name, cfg)[1]
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write("{}")
        prof = B.BankPlanStage().write_profile(cfg)
        assert prof["drop_partial_topologies"] is flag
        assert prof["cohort_map"] == d["real"]["cohort_map"]


def test_changed_files_ascii_lf():
    for rel in CHANGED:
        blob = open(os.path.join(U.REPO, rel), "rb").read()
        blob.decode("ascii")
        assert b"\r" not in blob, rel


def test_sequencer_modules_py36_syntax():
    if shutil.which("vermin") is None:
        pytest.skip("vermin not installed")
    files = [os.path.join(U.REPO, f) for f in CHANGED[:4]]
    r = subprocess.run(["vermin", "--no-tips", "-t=3.6-", "--violations"] + files,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert r.returncode == 0, r.stdout


def test_agnostic_grep():
    bad = []
    for dp, dn, fn in os.walk(U.PIPE):
        rel = os.path.relpath(dp, U.PIPE)
        if rel.split(os.sep)[0] in ("projects", "tests") or "__pycache__" in rel:
            continue
        for f in fn:
            if f.endswith((".py", ".sh")):
                text = open(os.path.join(dp, f), "rb").read().decode("ascii", "replace")
                for word in ("giulia", "ldellamea", "davinci-1/home", "cadex", "hhgap"):
                    if word in text:
                        bad.append((os.path.join(rel, f), word))
    assert bad == [], bad
