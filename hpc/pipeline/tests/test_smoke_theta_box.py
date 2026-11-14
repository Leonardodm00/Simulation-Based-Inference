"""Smoke test for theta strictly inside the open unit box -- Block 11 of
specs/SPEC.md, "Amendment 2026-11-14" (D-134 (3); wave W1, block 1c; the
audit's F202 and F206), with its pointers in blocks 12 and 16.

What is tested
    `clip_open_box` and `BOX_DELTA` (hpc/joint/stage1/latent_sbi_simulator.py)
    and their use in `sample_prior`, `sample_prior_given_classes` and
    `sample_box_prior`; the builder's refusal of a stored theta outside the
    open box (hpc/joint/stage1/build_latent_bank.py); `write_shard`'s
    `exist_ok=True` (hpc/joint/stage1/latent_bank.py); the record tool's
    theta check (hpc/pipeline/tools/synth_bank_record.py).

Checks (the amendment's oracles)
    test_known_answer   clip_open_box on 0, 1e-300, 2^-25, 2^-24, 0.5,
                        1 - 2^-24, 1 - 2^-25, 1 - 1e-9, 1 - eps64, 1: float32
                        of every result in (0, 1), equal to the input in
                        [2^-24, 1 - 2^-24]; the old clip's case (float32 of
                        clip(1 - 1e-9, eps, 1 - eps) is 1.0); -0.2, 1.5, NaN
                        and inf each raise FloatingPointError
    test_reference      the three samplers at fixed seeds against 8d5b162's
                        (git show): equal to the old value clipped to
                        [BOX_DELTA, 1 - BOX_DELTA], element by element; a
                        generator returning chosen values (1 - 1e-9 among
                        them): the box sampler, the builder's cast and the
                        record tool leave a stored theta < 1 (8d5b162's
                        builder stores 1.0 there, which the tool refuses)
    test_convergence    not a discretisation: skipped
    test_invariants     a default (mixture) bank, a box bank and a cohort with
                        regular centres, small, built by this code and by
                        8d5b162's (git archive): every array byte for byte,
                        generator_sha256, contract digest and build equal
    test_contract       the record tool on a bank whose stored theta was set
                        to 1.0, 0.0 and NaN in one coordinate (arm S; 1.0
                        also on arm R): exit 4, the REFUSE line, nothing
                        written; the clean banks recorded with their pass line;
                        a shard of 0 rows: the row count's REFUSE line, exit 4,
                        no traceback, as 8d5b162's tool; the builder's cast
                        given a NaN theta refuses, nothing written
                        [contract amendment 2]
    test_determinism    the same argv twice gives the same bytes (every array)
    test_edge_cases     clip_open_box: non-finite -> FloatingPointError, the
                        input unchanged, a 0-d and an empty input; the
                        builder's refusal of a stored 1.0 (nothing written);
                        two shard builds started together into one new folder
                        with os.makedirs slowed by 0.5 s (F206): both exit 0,
                        both shards present; with 8d5b162's latent_bank.py one
                        exits 1

Run
    python3 hpc/pipeline/tests/test_smoke_theta_box.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO. About 1-2 minutes on one CPU; needs
        numpy and scipy, and git with the commit 8d5b162 in this repository's
        history (the reference code). Builds its banks in a temporary folder.
    python3 -m pytest hpc/pipeline/tests/test_smoke_theta_box.py -q
        The same checks through pytest.

Keep this file pure ASCII, LF only.
"""
from __future__ import annotations

import atexit
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
_STAGE1 = os.path.join(_REPO, "hpc", "joint", "stage1")
for _p in (_STAGE1, _HERE, _PIPE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import latent_bank as LB                     # noqa: E402
import latent_sbi_simulator as LSS           # noqa: E402

SEED = 12345
REF_COMMIT = "8d5b162"   # the reference code of the amendment's "Unchanged"
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []

PYTHON = sys.executable
BUILDER = os.path.join(_STAGE1, "build_latent_bank.py")
RECORD_TOOL = os.path.join(_PIPE, "tools", "synth_bank_record.py")
EPS64 = float(np.finfo(np.float64).eps)

# Small reference-provider shards (the fixture generator of
# hpc/joint/stage3/smoke_test_blocks_11_14.py), W = 64 samples.
REF = ["--provider", "reference", "--T-win", "1.28", "--fs", "50",
       "--n-neurons", "20", "--n-windows", "2"]

_SHARED = {}


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def tmp_root():
    if "root" not in _SHARED:
        _SHARED["root"] = tempfile.mkdtemp(prefix="theta_box_")
        atexit.register(shutil.rmtree, _SHARED["root"], True)
    return _SHARED["root"]


def old_tree():
    """The tree of hpc/joint and hpc/dsn at REF_COMMIT, extracted once."""
    if "old" not in _SHARED:
        dest = os.path.join(tmp_root(), "old_tree")
        os.makedirs(dest)
        arch = subprocess.run(["git", "-C", _REPO, "archive", REF_COMMIT, "hpc/joint", "hpc/dsn"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if arch.returncode != 0:
            raise AssertionError("git archive %s failed: %s" % (REF_COMMIT, arch.stderr[-400:]))
        tar = subprocess.run(["tar", "-x", "-C", dest], input=arch.stdout,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if tar.returncode != 0:
            raise AssertionError("tar failed: %s" % tar.stderr[-400:])
        _SHARED["old"] = dest
    return _SHARED["old"]


def old_stage1():
    return os.path.join(old_tree(), "hpc", "joint", "stage1")


def old_record_tool():
    """8d5b162's tools/synth_bank_record.py, beside the old tree's stage1."""
    path = os.path.join(old_tree(), "hpc", "pipeline", "tools", "synth_bank_record.py")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
        src = subprocess.run(["git", "-C", _REPO, "show",
                              REF_COMMIT + ":hpc/pipeline/tools/synth_bank_record.py"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if src.returncode != 0:
            raise AssertionError("git show failed: %s" % src.stderr[-400:])
        with open(path, "wb") as fh:
            fh.write(src.stdout)
    return path


def old_simulator():
    """8d5b162's latent_sbi_simulator.py, imported under another name (its
    own imports resolve to this tree's modules, unchanged since then)."""
    if "old_lss" not in _SHARED:
        path = os.path.join(old_stage1(), "latent_sbi_simulator.py")
        spec = importlib.util.spec_from_file_location("old_latent_sbi_simulator", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert not hasattr(mod, "clip_open_box"), "the reference is not the code before the amendment"
        _SHARED["old_lss"] = mod
    return _SHARED["old_lss"]


def build(out, args, stage1=_STAGE1, expect_ok=True):
    """Run a builder (this tree's, or another stage1's); (rc, output)."""
    cmd = [PYTHON, os.path.join(stage1, "build_latent_bank.py"), "--out-dir", out] + [str(a) for a in args]
    r = subprocess.run(cmd, cwd=stage1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=600)
    if expect_ok and r.returncode != 0:
        raise AssertionError("builder failed (%d): %s" % (r.returncode, r.stdout.strip()[-800:]))
    return r.returncode, r.stdout


# A builder run with its box sampler's generator replaced: its uniform draws
# are those of the real generator with chosen coordinates overwritten by
# 1 - 1e-9 (the old clip let it through, float32 rounds it to 1.0).
# MODE "clip": the sampler of the stage1 it runs in, as written;
# MODE "raw": the draw returned unclipped (the builder's own defence);
# MODE "nan_cast": the sampler as written, the simulation at its phi, and the
# phi_eff the builder casts of well 0 given a NaN on axis 1.
_FORCED_RUNNER = r'''
import os, sys
stage1, mode = sys.argv[1], sys.argv[2]
argv = sys.argv[3:]
sys.path.insert(0, stage1)
os.chdir(stage1)
import numpy as np
import build_latent_bank as B
import latent_sbi_simulator as L
FORCED = 1.0 - 1e-9

class Forced(object):
    def __init__(self, rng):
        self._rng = rng
    def uniform(self, low=0.0, high=1.0, size=None):
        u = np.array(self._rng.uniform(low, high, size=size), dtype=np.float64)
        flat = u.reshape(-1)
        flat[0] = FORCED
        flat[flat.size // 2] = FORCED
        return u
    def __getattr__(self, name):
        return getattr(self._rng, name)

real_box = L.sample_box_prior

def forced_box(spec, n, rng):
    if mode == "clip":
        return real_box(spec, n, Forced(rng))
    phi = Forced(rng).uniform(0.0, 1.0, size=(int(n), spec.n_latent))
    return np.full(int(n), L.BOX_CLASS, dtype=np.int64), phi

real_sim = B.simulate_windows

def nan_cast_sim(spec, phi, *a, **kw):
    x, info = real_sim(spec, phi, *a, **kw)
    if kw.get("well") == 0:
        eff = np.array(info["phi_eff"], dtype=np.float64)
        eff[1] = np.nan
        info = dict(info, phi_eff=eff)
    return x, info

if mode == "nan_cast":
    B.simulate_windows = nan_cast_sim
else:
    B.sample_box_prior = forced_box
sys.argv = [B.__file__] + argv
sys.exit(B.main(argv))
'''

# A builder run whose os.makedirs, for the bank's folder only, waits until the
# other build has reached the same line (a barrier on two marker files), then
# 0.5 s more, then creates it: the race of F206, made certain.
_RACE_RUNNER = r'''
import os, sys, time
stage1, target, barrier, me, other = sys.argv[1:6]
argv = sys.argv[6:]
sys.path.insert(0, stage1)
os.chdir(stage1)
_real = os.makedirs

def slow_makedirs(name, *a, **kw):
    if os.path.abspath(name) == os.path.abspath(target):
        open(os.path.join(barrier, me), "w").close()
        t0 = time.time()
        while not os.path.exists(os.path.join(barrier, other)) and time.time() - t0 < 120:
            time.sleep(0.01)
        time.sleep(0.5)
    return _real(name, *a, **kw)

os.makedirs = slow_makedirs
import build_latent_bank as B
sys.argv = [B.__file__] + argv
sys.exit(B.main(argv))
'''


def _runner(name, text):
    path = os.path.join(tmp_root(), name)
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(text)
    return path


def forced_build(out, args, stage1=_STAGE1, mode="clip"):
    cmd = [PYTHON, _runner("forced_runner.py", _FORCED_RUNNER), stage1, mode,
           "--out-dir", out] + [str(a) for a in args]
    r = subprocess.run(cmd, cwd=tmp_root(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=600)
    return r.returncode, r.stdout


def load_arrays(path):
    with np.load(path) as z:
        return dict((k, np.array(z[k])) for k in z.files)


def read_sidecar(path):
    with open(os.path.splitext(path)[0] + ".json") as fh:
        return json.load(fh)


def normalise(argv):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--out-dir":
            skip = True
            continue
        out.append(str(a))
    return out


def expect_for(bank_dir, n_shards, wells_per_donor, n_windows, n_classes=3, donors_per_class=0):
    """The record tool's expectation for a bank built by hand, its build read
    from the shards' own sidecars (the argv check is not what is tested)."""
    argvs, rows = [], []
    side = None
    for k in range(n_shards):
        npz = os.path.join(bank_dir, "shard_%04d.npz" % k)
        side = read_sidecar(npz)
        argvs.append(normalise(side["provenance"]["argv"]))
        with np.load(npz) as z:
            rows.append(int(z["x"].shape[0]))
    arm = side["arm"]
    return {"bank": "test", "dir": os.path.abspath(bank_dir), "arm": arm,
            "theta_withheld": arm == "R", "W": int(side["W"]),
            "d_theta": int(len(side["param_names"])),
            "latent_kind": (side.get("latent_spec") or {}).get("kind"),
            "wells_per_donor": int(wells_per_donor), "n_windows": int(n_windows),
            "n_classes": int(n_classes), "donors_per_class": int(donors_per_class),
            "build": {"schema": "synth_bank_build/1", "bank": "test", "arm": arm,
                      "n_shards": int(n_shards), "rows_per_shard": rows, "argv": argvs}}


def record_tool(bank_dir, expect, out, tool=RECORD_TOOL):
    exp_path = os.path.join(tmp_root(), "expect_%d.json" % (abs(hash(bank_dir)) % 10 ** 8))
    with open(exp_path, "w") as fh:
        json.dump(expect, fh)
    r = subprocess.run([PYTHON, tool, "--bank-dir", bank_dir, "--expect", exp_path,
                        "--out", out], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=300)
    return r.returncode, r.stdout


def box_bank():
    """A small box bank (arm S) built by this tree, shared by the checks."""
    if "box" not in _SHARED:
        out = os.path.join(tmp_root(), "box_S")
        build(out, ["--arm", "S", "--prior", "box", "--n-traces", "8", "--seed", "7"] + REF)
        _SHARED["box"] = out
    return _SHARED["box"]


def cohort_bank():
    """A small cohort (arm R, regular centres, 2 donors per class)."""
    if "cohort" not in _SHARED:
        out = os.path.join(tmp_root(), "cohort_R")
        build(out, ["--arm", "R", "--n-classes", "3", "--donors-per-class", "2",
                    "--n-traces", "12", "--wells-per-donor", "2", "--class-centres", "regular",
                    "--centre-scale", "0.6", "--seed", "8"] + REF)
        _SHARED["cohort"] = out
    return _SHARED["cohort"]


# ---------------------------------------------------------------- checks ---

KNOWN_INPUTS = np.array([0.0, 1e-300, 2.0 ** -25, 2.0 ** -24, 0.5, 1.0 - 2.0 ** -24,
                         1.0 - 2.0 ** -25, 1.0 - 1e-9, 1.0 - EPS64, 1.0], dtype=np.float64)


def test_known_answer():
    assert LSS.BOX_DELTA == 2.0 ** -24, LSS.BOX_DELTA
    # BOX_DELTA and 1 - BOX_DELTA are float32 numbers (exact round trip)
    assert float(np.float32(LSS.BOX_DELTA)) == LSS.BOX_DELTA
    assert float(np.float32(1.0 - LSS.BOX_DELTA)) == 1.0 - LSS.BOX_DELTA
    out = LSS.clip_open_box(KNOWN_INPUTS)
    assert isinstance(out, np.ndarray) and out.dtype == np.float64 and out.shape == KNOWN_INPUTS.shape
    f32 = out.astype(np.float32)
    assert np.all(f32 > 0) and np.all(f32 < 1), f32
    lo, hi = 2.0 ** -24, 1.0 - 2.0 ** -24
    inside = (KNOWN_INPUTS >= lo) & (KNOWN_INPUTS <= hi)
    assert inside.tolist() == [False, False, False, True, True, True, False, False, False, False]
    # exact: the clip changes nothing inside, bit for bit
    assert_array_equal(out[inside], KNOWN_INPUTS[inside])
    # exact values outside, by hand: the two ends
    expected = np.array([lo, lo, lo, lo, 0.5, hi, hi, hi, hi, hi])
    assert_array_equal(out, expected)
    # the old clip's case reproduced: [eps, 1 - eps] lets 1 - 1e-9 through, and
    # its float32 is 1.0, on the boundary of the box
    old = np.clip(1.0 - 1e-9, EPS64, 1.0 - EPS64)
    assert old == 1.0 - 1e-9 and float(np.float32(old)) == 1.0, (old, np.float32(old))
    # [contract amendment 2] below 0, above 1 and not finite: refused
    for bad in (-0.2, 1.5, float("nan"), float("inf")):
        try:
            LSS.clip_open_box(np.array([0.5, bad]))
            raise AssertionError("clip_open_box accepted %r" % bad)
        except FloatingPointError:
            pass


def test_reference():
    old = old_simulator()
    lo, hi = LSS.BOX_DELTA, 1.0 - LSS.BOX_DELTA
    spec = LSS.LatentSBISpec()
    n_moved = 0
    for seed in (1, 2, 3):
        # sample_prior
        c_new, p_new = LSS.sample_prior(spec, 3000, make_rng(seed))
        c_old, p_old = old.sample_prior(spec, 3000, make_rng(seed))
        assert_array_equal(c_new, c_old)
        assert_array_equal(p_new, np.clip(p_old, lo, hi))
        n_moved += int(np.sum(p_new != p_old))
        # sample_prior_given_classes
        cls = make_rng(seed + 100).permutation(np.repeat(np.arange(spec.n_classes), 1000))
        g_new = LSS.sample_prior_given_classes(spec, cls, make_rng(seed))
        g_old = old.sample_prior_given_classes(spec, cls, make_rng(seed))
        assert_array_equal(g_new, np.clip(g_old, lo, hi))
        n_moved += int(np.sum(g_new != g_old))
        # sample_box_prior
        b_c, b_new = LSS.sample_box_prior(spec, 3000, make_rng(seed))
        o_c, b_old = old.sample_box_prior(spec, 3000, make_rng(seed))
        assert_array_equal(b_c, o_c)
        assert_array_equal(b_new, np.clip(b_old, lo, hi))
        n_moved += int(np.sum(b_new != b_old))
    print("   reference: %d of %d coordinates moved by the new clip at seeds 1-3"
          % (n_moved, 3 * 3 * 3000 * spec.n_latent))

    # a generator returning chosen values: 1 - 1e-9 and 1 - 2^-25 near the top,
    # 0 and 2^-25 near the bottom, 0.5 in the middle
    chosen = np.array([[1.0 - 1e-9, 1.0 - 2.0 ** -25, 0.0, 2.0 ** -25, 0.5, 1.0 - EPS64]])

    class Chosen(object):
        def uniform(self, low=0.0, high=1.0, size=None):
            return np.broadcast_to(chosen, size).copy()

    spec6 = LSS.LatentSBISpec(n_latent=6)
    _c, ph = LSS.sample_box_prior(spec6, 2, Chosen())
    assert np.all(ph.astype(np.float32) < 1) and np.all(ph.astype(np.float32) > 0), ph
    _c, ph_old = old.sample_box_prior(spec6, 2, Chosen())
    assert float(ph_old.astype(np.float32)[0, 0]) == 1.0, "the old sampler's case is not reproduced"

    # the builder's cast and the record tool, with the box sampler's generator
    # forced (coordinates 0 and n/2 of its draw set to 1 - 1e-9)
    args = ["--arm", "S", "--prior", "box", "--n-traces", "8", "--seed", "5"] + REF
    new_dir = os.path.join(tmp_root(), "forced_new")
    rc, txt = forced_build(new_dir, args)
    assert rc == 0, txt[-800:]
    th = load_arrays(os.path.join(new_dir, "shard_0000.npz"))["theta"]
    assert th.dtype == np.float32
    assert th[0, 0] == np.float32(1.0 - LSS.BOX_DELTA) and np.all(th < 1) and np.all(th > 0), th[:2]
    rc, txt = record_tool(new_dir, expect_for(new_dir, 1, 2, 2), os.path.join(new_dir, "BANK_RECORD.json"))
    assert rc == 0 and "SYNTH BANK RECORD OK:" in txt, txt
    # 8d5b162's sampler and builder store 1.0 there, and the record tool refuses it
    old_dir = os.path.join(tmp_root(), "forced_old")
    rc, txt = forced_build(old_dir, args, stage1=old_stage1())
    assert rc == 0, txt[-800:]
    th_old = load_arrays(os.path.join(old_dir, "shard_0000.npz"))["theta"]
    assert th_old[0, 0] == np.float32(1.0), th_old[0]
    out = os.path.join(old_dir, "BANK_RECORD.json")
    rc, txt = record_tool(old_dir, expect_for(old_dir, 1, 2, 2), out)
    # 4 donors x 6 axes drawn; coordinates 0 and 12 forced: donors 0 and 2,
    # each of 2 wells x 2 windows, so 8 rows
    assert rc == 4 and ("REFUSE: shard 0: 8 row(s) with a theta outside the open unit box "
                        "(first: row 0, axis 0, value 1.0)") in txt, txt
    assert not os.path.exists(out)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def _same_bank(new_dir, old_dir, n_shards):
    for k in range(n_shards):
        npz_n = os.path.join(new_dir, "shard_%04d.npz" % k)
        npz_o = os.path.join(old_dir, "shard_%04d.npz" % k)
        a_n, a_o = load_arrays(npz_n), load_arrays(npz_o)
        assert sorted(a_n) == sorted(a_o), (sorted(a_n), sorted(a_o))
        for key in a_n:
            assert a_n[key].dtype == a_o[key].dtype and a_n[key].shape == a_o[key].shape, key
            assert a_n[key].tobytes() == a_o[key].tobytes(), "shard %d: array %s differs" % (k, key)
        s_n, s_o = read_sidecar(npz_n), read_sidecar(npz_o)
        assert s_n["generator_sha256"] == s_o["generator_sha256"], (s_n["generator_sha256"],
                                                                  s_o["generator_sha256"])
        assert s_n["contract_digest"] == s_o["contract_digest"] == LB.contract_digest(s_n)
        assert normalise(s_n["provenance"]["argv"]) == normalise(s_o["provenance"]["argv"])


def test_invariants():
    # The banks of the pipeline's stages (block 16) at test_smoke_synthetic's
    # reduced shape: the bench generator at W 128; arm S of 3 shards and R0.
    import test_smoke_synthetic as T
    from stages import synthetic as SY
    import config as C
    root = os.path.join(tmp_root(), "inv")
    os.makedirs(root)
    path, _d = T._small(root)
    cfg = C.load_config(path, synthetic_ok=True)
    for bank in ("sim", "R0"):
        b = SY.build_of(cfg, bank)
        exp = SY.SynthBankStage(bank).expect(cfg)
        new_dir = exp["dir"]
        old_dir = os.path.join(root, "old_" + bank)
        for argv in b["argv"]:
            build(new_dir, argv)
            build(old_dir, argv, stage1=old_stage1())
        _same_bank(new_dir, old_dir, b["n_shards"])
        # the build: the record tool's, on the bank of each code, equal to build_of's
        builds = []
        for d in (new_dir, old_dir):
            e = dict(exp)
            e["dir"] = os.path.abspath(d)
            out = os.path.join(d, "BANK_RECORD.json")
            rc, txt = record_tool(d, e, out)
            assert rc == 0 and "SYNTH BANK RECORD OK:" in txt, txt
            with open(out) as fh:
                rec = json.load(fh)
            builds.append((rec["build"], rec["contract_digest"]))
        assert builds[0] == builds[1] and builds[0][0] == b, builds
    # a default (mixture) bank, the reference provider
    args = ["--arm", "S", "--n-traces", "16", "--seed", "3"] + REF
    new_dir, old_dir = os.path.join(root, "mix_new"), os.path.join(root, "mix_old")
    build(new_dir, args)
    build(old_dir, args, stage1=old_stage1())
    _same_bank(new_dir, old_dir, 1)


def test_contract():
    for bank_dir, arm, values in ((box_bank(), "S", (1.0, 0.0, float("nan"))),
                                  (cohort_bank(), "R", (1.0,))):
        dpc = 2 if arm == "R" else 0
        exp = expect_for(bank_dir, 1, 2, 2, n_classes=3, donors_per_class=dpc)
        out = os.path.join(bank_dir, "BANK_RECORD.json")
        rc, txt = record_tool(bank_dir, exp, out)
        assert rc == 0 and ("SYNTH BANK RECORD OK: %s (1 shard(s), %d rows, digest "
                            % (out, sum(exp["build"]["rows_per_shard"]))) in txt, txt
        os.remove(out)
        os.remove(out + ".sha256")
        npz = os.path.join(bank_dir, "shard_0000.npz")
        keep = open(npz, "rb").read()
        clean = load_arrays(npz)
        assert clean["theta"].dtype == np.float32
        assert np.all(clean["theta"] > 0) and np.all(clean["theta"] < 1)
        for v in values:
            arrays = dict((k, a.copy()) for k, a in clean.items())
            arrays["theta"][3, 2] = v
            np.savez_compressed(npz[:-4], **arrays)
            rc, txt = record_tool(bank_dir, exp, out)
            want = ("REFUSE: shard 0: 1 row(s) with a theta outside the open unit box "
                    "(first: row 3, axis 2, value %r)" % float(np.float32(v)))
            assert rc == 4 and want in txt, (arm, v, txt)
            assert "SYNTH BANK RECORD REFUSED: 1 problem(s); nothing written" in txt, txt
            assert not os.path.exists(out) and not os.path.exists(out + ".sha256")
            with open(npz, "wb") as fh:
                fh.write(keep)
        rc, txt = record_tool(bank_dir, exp, out)
        assert rc == 0, txt

    # [contract amendment 2] a shard with 0 rows: the row count's refusal,
    # exit 4, no traceback, the same lines as 8d5b162's tool
    bank_dir = box_bank()
    exp = expect_for(bank_dir, 1, 2, 2)
    out = os.path.join(bank_dir, "BANK_RECORD.json")
    if os.path.exists(out):
        os.remove(out)
        os.remove(out + ".sha256")
    npz = os.path.join(bank_dir, "shard_0000.npz")
    keep = open(npz, "rb").read()
    clean = load_arrays(npz)
    try:
        np.savez_compressed(npz[:-4], **dict((k, a[:0]) for k, a in clean.items()))
        rc, txt = record_tool(bank_dir, exp, out)
        want = "REFUSE: shard 0: 0 rows, the build's %d" % exp["build"]["rows_per_shard"][0]
        assert rc == 4 and want in txt and "Traceback" not in txt, txt
        assert not os.path.exists(out) and not os.path.exists(out + ".sha256")
        rc_old, txt_old = record_tool(bank_dir, exp, out, tool=old_record_tool())
        assert rc_old == 4 and "Traceback" not in txt_old, txt_old
        refuse = [ln for ln in txt.splitlines() if ln.startswith("REFUSE")]
        refuse_old = [ln for ln in txt_old.splitlines() if ln.startswith("REFUSE")]
        assert refuse == refuse_old, (refuse, refuse_old)
        assert not os.path.exists(out)
    finally:
        with open(npz, "wb") as fh:
            fh.write(keep)

    # [contract amendment 2] the builder's cast given a NaN theta: refused,
    # nothing written
    nan_dir = os.path.join(tmp_root(), "nan_cast")
    rc, txt = forced_build(nan_dir, ["--arm", "S", "--prior", "box", "--n-traces", "8",
                                     "--seed", "5"] + REF, mode="nan_cast")
    assert rc != 0 and "outside the open unit box after the cast to float32" in txt and \
        "(first: row 0, axis 1, value nan)" in txt, txt[-800:]
    assert "Traceback" not in txt, txt[-800:]
    assert not os.path.exists(nan_dir) or not os.listdir(nan_dir), os.listdir(nan_dir)


def test_determinism():
    a = LSS.clip_open_box(KNOWN_INPUTS)
    b = LSS.clip_open_box(KNOWN_INPUTS)
    assert a.tobytes() == b.tobytes()
    args = ["--arm", "S", "--prior", "box", "--n-traces", "8", "--seed", "7"] + REF
    twin = os.path.join(tmp_root(), "box_S_twin")
    build(twin, args)
    a1 = load_arrays(os.path.join(box_bank(), "shard_0000.npz"))
    a2 = load_arrays(os.path.join(twin, "shard_0000.npz"))
    assert sorted(a1) == sorted(a2)
    for k in a1:
        assert a1[k].tobytes() == a2[k].tobytes(), k
    s1 = read_sidecar(os.path.join(box_bank(), "shard_0000.npz"))
    s2 = read_sidecar(os.path.join(twin, "shard_0000.npz"))
    assert s1["contract_digest"] == s2["contract_digest"]
    assert s1["generator_sha256"] == s2["generator_sha256"]


def test_edge_cases():
    # clip_open_box: non-finite input refused
    for bad in (np.nan, np.inf, -np.inf):
        x = np.array([0.5, bad])
        try:
            LSS.clip_open_box(x)
            raise AssertionError("clip_open_box accepted %r" % bad)
        except FloatingPointError:
            pass
    # the input is not changed, the result is a new array
    x = np.array([[0.0, 1.0], [0.25, 1.0 - 1e-9]])
    keep = x.copy()
    y = LSS.clip_open_box(x)
    assert_array_equal(x, keep)
    assert y is not x and not np.shares_memory(x, y)
    # a 0-d input: an ndarray of shape (); an empty input: an empty array
    z = LSS.clip_open_box(1.0)
    assert isinstance(z, np.ndarray) and z.shape == () and float(z) == 1.0 - LSS.BOX_DELTA
    e = LSS.clip_open_box(np.empty((0, 6)))
    assert e.shape == (0, 6) and e.dtype == np.float64

    # the builder's refusal: a draw returned unclipped with 1 - 1e-9 in it
    out = os.path.join(tmp_root(), "refused")
    rc, txt = forced_build(out, ["--arm", "S", "--prior", "box", "--n-traces", "8",
                                 "--seed", "5"] + REF, mode="raw")
    assert rc != 0 and "outside the open unit box after the cast to float32" in txt and \
        "(first: row 0, axis 0, value 1.0)" in txt, txt[-800:]
    assert not os.path.exists(out) or not os.listdir(out), os.listdir(out)

    # the race of F206: two shards of one new folder, os.makedirs slowed
    def race(stage1, tag):
        target = os.path.join(tmp_root(), "race_" + tag)
        barrier = os.path.join(tmp_root(), "barrier_" + tag)
        os.makedirs(barrier)
        runner = _runner("race_runner.py", _RACE_RUNNER)
        procs = []
        for k in (0, 1):
            argv = ["--out-dir", target, "--arm", "S", "--prior", "box", "--n-traces", "4",
                    "--shard-index", str(k), "--seed", "9"] + REF
            procs.append(subprocess.Popen(
                [PYTHON, runner, stage1, target, barrier, "p%d" % k, "p%d" % (1 - k)] + argv,
                cwd=tmp_root(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True))
        res = []
        for p in procs:
            txt, _ = p.communicate(timeout=600)
            res.append((p.returncode, txt))
        present = sorted(f for f in os.listdir(target)) if os.path.isdir(target) else []
        return res, present

    res, present = race(_STAGE1, "new")
    assert [r[0] for r in res] == [0, 0], [r[1][-600:] for r in res]
    assert present == ["shard_0000.json", "shard_0000.npz", "shard_0001.json",
                       "shard_0001.npz"], present
    res, present = race(old_stage1(), "old")
    codes = sorted(r[0] for r in res)
    assert codes == [0, 1], [(r[0], r[1][-400:]) for r in res]
    loser = [r[1] for r in res if r[0] == 1][0]
    assert "FileExistsError" in loser, loser[-600:]


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
    print(_environment())

    results = []
    try:
        for name, func in checks:
            start = time.perf_counter()
            try:
                func()
                status, detail = "PASS", ""
            except unittest.SkipTest as exc:
                status, detail = "SKIP", str(exc)
            except NotImplementedError as exc:
                status, detail = "TODO", str(exc)
            except AssertionError as exc:
                status, detail = "FAIL", str(exc)
            except Exception:  # unexpected errors are reported, never hidden
                status, detail = "ERROR", traceback.format_exc()
            results.append((name, status, time.perf_counter() - start, detail))
    finally:
        if "root" in _SHARED:
            shutil.rmtree(_SHARED["root"], ignore_errors=True)

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
