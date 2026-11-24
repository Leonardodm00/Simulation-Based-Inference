"""Block 31 at the runner (run_joint_arms.py; specs/SPEC.md block 31): the flags,
the report lines, the replicate diagnostic, the refusals and the default
unchanged. Trains tiny models (W 64, 2 epochs of 3 steps, one thread): run it
alone, never beside another training suite.

Oracles (from the spec, before the code was read):
  O31.1 without the flags the runner is the parent's: a run with no flag, one
        with `--rep-pairs cross_well --rep-group-key donor` and the base
        commit's runner give the same record (one thread, bit for bit).
  O31.5 a real bank without the KEY array stops the run with `--rep-group-key
        <KEY>: the real bank has no '<KEY>' array (a bank built before
        Sbi-extractor block 9 has no rep_group)`; without a real bank the flags
        are accepted.
  O31.4 |P| = 0 with lambda_rep > 0 stops (F-aa), the message naming the rule
        and the KEY.
  O31.6 the report lines `replicate pairs  : <|P|>  (two rows of one rep_group
        group, any well)` and `replicate groups : <paired> of <groups>`, |P| and
        the counts computed here from the rep_group array.
  O31.7 the diagnostic under within_group scores min(64, |P|) pairs, its
        pair_kind the rule's text and n_groups_scored the groups among them.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import _spec_util as U

pytest.importorskip("torch")
pytest.importorskip("sbi")
pytest.importorskip("zuko")

STAGE3 = os.path.join(U.JOINT, "stage3")
BUILDER = os.path.join(U.JOINT, "stage1", "build_latent_bank.py")
RUNNER = os.path.join(STAGE3, "run_joint_arms.py")
BASE_RUNNER = "/tmp/base_tree/hpc/joint/stage3/run_joint_arms.py"
REF = ["--provider", "reference", "--T-win", "1.28", "--fs", "50", "--n-neurons", "20",
       "--n-windows", "2"]
_BANKS = {}


def _build(out, args):
    r = subprocess.run([U.PYTHON, BUILDER, "--out-dir", out] + [str(a) for a in args],
                       cwd=os.path.dirname(BUILDER), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=900,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stdout[-1500:]


def _with_group(src, dst, fn):
    sys.path.insert(0, os.path.join(U.JOINT, "stage1"))
    import latent_bank as LB
    paths = sorted(p for p in os.listdir(src) if p.endswith(".npz"))
    os.makedirs(dst)
    lo = 0
    groups = []
    for p in paths:
        a, side = LB.read_shard(os.path.join(src, p))
        n = a["x"].shape[0]
        g = np.asarray(fn(a, lo), dtype=np.int64)
        lo += n
        b = dict(a)
        b["rep_group"] = g
        groups.append(g)
        LB.write_shard(os.path.join(dst, p), b, side)
    return np.concatenate(groups)


def banks():
    if _BANKS:
        return _BANKS
    root = U.tmpdir("spec_b31_banks_")
    S, R = os.path.join(root, "S"), os.path.join(root, "R")
    _build(S, ["--arm", "S", "--prior", "box", "--n-traces", "80", "--n-classes", "3",
               "--seed", "1"] + REF)
    _build(R, ["--arm", "R", "--n-classes", "3", "--donors-per-class", "4", "--n-traces", "24",
               "--class-centres", "regular", "--centre-scale", "0.6", "--seed", "2"] + REF)
    # rep_group: the class for classes 0 and 1; every row of class 2 its own group
    rg = _with_group(R, os.path.join(root, "RG"),
                     lambda a, lo: np.where(a["cls"] < 2, a["cls"],
                                            100 + lo + np.arange(a["cls"].size)))
    _with_group(R, os.path.join(root, "R1"), lambda a, lo: 1000 + lo + np.arange(a["cls"].size))
    _BANKS.update(root=root, S=os.path.join(S, "*.npz"), R=os.path.join(R, "*.npz"),
                  RG=os.path.join(root, "RG", "*.npz"), R1=os.path.join(root, "R1", "*.npz"),
                  rg=rg)
    return _BANKS


def run_arm(arm, sim, real, out_dir, extra=(), runner=RUNNER):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [U.PYTHON, runner, "--arm", arm, "--seed", "0", "--sim-shards", sim,
           "--out-dir", out_dir, "--threads", "1", "--epochs", "2", "--steps-per-epoch", "3",
           "--b-sim", "24", "--b-met", "12", "--b-rep", "4", "--n-posterior-draws", "16",
           "--embedding-size", "8", "--hidden-features", "24", "--num-transforms", "3",
           "--depth-exponent", "3", "--width-multiplier", "1.5"]
    if real is not None:
        cmd += ["--real-shards", real]
    cmd += list(extra)
    r = subprocess.run(cmd, cwd=os.path.dirname(runner), env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=1800)
    return r.returncode, r.stdout


def record(out_dir, arm):
    with open(os.path.join(out_dir, "%s_seed0.json" % arm)) as fh:
        return json.load(fh)


def comparable(rec):
    keep = dict((k, rec.get(k)) for k in ("L", "L0", "L_pseudo_real", "split_hash", "history",
                                            "replicate", "endpoint_primary", "r_eff"))
    return json.dumps(keep, sort_keys=True)


@pytest.fixture(scope="module")
def out_root():
    r = U.tmpdir("spec_b31_runs_")
    yield r
    shutil.rmtree(r, ignore_errors=True)
    if _BANKS:
        shutil.rmtree(_BANKS["root"], ignore_errors=True)


def test_within_group_report_and_diagnostic(out_root):
    b = banks()
    rg = b["rg"]
    ids, counts = np.unique(rg, return_counts=True)
    P = int(sum(c * (c - 1) // 2 for c in counts))
    paired = int(np.sum(counts >= 2))
    assert (P, paired, ids.size) == (240, 2, 18)          # the design: 16 + 16 rows, 16 singletons
    out = os.path.join(out_root, "wg")
    rc, text = run_arm("A5", b["S"], b["RG"], out,
                       ["--rep-pairs", "within_group", "--rep-group-key", "rep_group"])
    assert rc == 0, text[-3000:]
    lines = text.splitlines()
    assert "replicate pairs  : %d  (two rows of one rep_group group, any well)" % P in lines, \
        [ln for ln in lines if "replicate" in ln]
    assert "replicate groups : %d of %d" % (paired, ids.size) in lines, \
        [ln for ln in lines if "replicate" in ln]
    rep = record(out, "A5")["replicate"]
    assert rep["pair_kind"] == "two rows of one rep_group group, any well", rep
    assert rep["n_pairs"] == min(64, P) and rep["n_groups_scored"] == paired, rep


def test_missing_key_array_refused(out_root):
    b = banks()
    rc, text = run_arm("A5", b["S"], b["R"], os.path.join(out_root, "nokey"),
                       ["--rep-pairs", "within_group", "--rep-group-key", "rep_group"])
    assert rc != 0
    assert ("--rep-group-key rep_group: the real bank has no 'rep_group' array (a bank built "
            "before Sbi-extractor block 9 has no rep_group)") in text, text[-2000:]


def test_no_pair_refused_naming_rule_and_key(out_root):
    b = banks()
    rc, text = run_arm("A5", b["S"], b["R1"], os.path.join(out_root, "single"),
                       ["--rep-pairs", "within_group", "--rep-group-key", "rep_group"])
    assert rc != 0
    tail = text.strip().splitlines()[-1]
    assert "within_group" in tail and "rep_group" in tail and "F-aa" in tail, tail


def test_flags_without_a_real_bank_accepted(out_root):
    b = banks()
    rc, text = run_arm("A1", b["S"], None, os.path.join(out_root, "a1"),
                       ["--rep-pairs", "within_group", "--rep-group-key", "rep_group"])
    assert rc == 0, text[-2000:]


def test_a2_with_the_key_runs(out_root):
    b = banks()
    rc, text = run_arm("A2", b["S"], b["RG"], os.path.join(out_root, "a2"),
                       ["--rep-pairs", "within_group", "--rep-group-key", "rep_group"])
    assert rc == 0, text[-2000:]


def test_default_unchanged(out_root):
    b = banks()
    o1, o2 = os.path.join(out_root, "d1"), os.path.join(out_root, "d2")
    rc1, t1 = run_arm("A5", b["S"], b["R"], o1)
    rc2, t2 = run_arm("A5", b["S"], b["R"], o2, ["--rep-pairs", "cross_well",
                                                  "--rep-group-key", "donor"])
    assert rc1 == 0 and rc2 == 0, (t1[-1500:], t2[-1500:])
    assert comparable(record(o1, "A5")) == comparable(record(o2, "A5"))
    rep1 = [ln for ln in t1.splitlines() if ln.startswith("replicate pairs")]
    assert rep1 == ["replicate pairs  : 48  (two wells of one donor)"], rep1   # B14's 48
    if os.path.isfile(BASE_RUNNER):
        o3 = os.path.join(out_root, "d3")
        rc3, t3 = run_arm("A5", b["S"], b["R"], o3, runner=BASE_RUNNER)
        assert rc3 == 0, t3[-1500:]
        assert comparable(record(o1, "A5")) == comparable(record(o3, "A5"))
