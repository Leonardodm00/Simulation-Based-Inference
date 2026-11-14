"""Spec-derived tests: block 11's amendment of 2026-11-14 (D-134 (3)), with
blocks 12 and 16's amendments of the same date and block 5's `write_shard`.

Oracles come from specs/SPEC.md (block 11, "Amendment 2026-11-14"):
BOX_DELTA = 2^-24; clip_open_box = clip(phi, BOX_DELTA, 1 - BOX_DELTA) in a
new float64 array, the input unchanged, a non-finite value raising
FloatingPointError; the three samplers return clip_open_box of their draw
(every draw in [BOX_DELTA, 1 - BOX_DELTA] unchanged bit for bit against the
code at 8d5b162); float32(theta) strictly inside (0, 1); the builder refuses
a stored coordinate <= 0 or >= 1; the record tool refuses a stored theta
<= 0, >= 1 or not finite with the stated REFUSE line, exit 4, nothing
written; write_shard creates its folder with exist_ok=True; a bank's
arrays, generator_sha256 and contract digest unchanged against 8d5b162.

Expected values are computed with NumPy by hand, never by the code under test.
Run: python3 -m pytest -q tests/spec/test_spec_block11_theta_box.py
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAGE1 = os.path.join(REPO, "hpc", "joint", "stage1")
PIPE = os.path.join(REPO, "hpc", "pipeline")
RECORD_TOOL = os.path.join(PIPE, "tools", "synth_bank_record.py")
BASE = "8d5b16285140f1a4181c0901bf535c7da041ae39"
PY = sys.executable
if STAGE1 not in sys.path:
    sys.path.insert(0, STAGE1)

import latent_sbi_simulator as L     # noqa: E402
import latent_bank as LB             # noqa: E402

D = 2.0 ** -24          # the spec's BOX_DELTA, written by hand
# a tiny reference-provider bank (W = 64): the builder's fixture generator
REF = ["--provider", "reference", "--T-win", "1.28", "--fs", "50",
       "--n-neurons", "20", "--n-windows", "2"]


@pytest.fixture(scope="module")
def tmp():
    d = tempfile.mkdtemp(prefix="spec_b11_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def old_tree(tmp):
    """The tree at the base commit (git archive), for the reference oracles."""
    dest = os.path.join(tmp, "old")
    os.makedirs(dest)
    a = subprocess.run(["git", "-C", REPO, "archive", BASE, "hpc"], stdout=subprocess.PIPE, check=True)
    subprocess.run(["tar", "-x", "-C", dest], input=a.stdout, check=True)
    return dest


@pytest.fixture(scope="module")
def old_L(old_tree):
    path = os.path.join(old_tree, "hpc", "joint", "stage1", "latent_sbi_simulator.py")
    spec = importlib.util.spec_from_file_location("old_lss_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "clip_open_box")
    return mod


# ------------------------------------------------------- the constant ---

def test_box_delta_constant():
    assert L.BOX_DELTA == D
    # both ends are float32 numbers (spec: "BOX_DELTA and 1 - BOX_DELTA are float32 numbers")
    assert float(np.float32(D)) == D and float(np.float32(1 - D)) == 1 - D
    # 1 - D is the float32 predecessor of 1: nothing float32 lies strictly between them
    assert np.nextafter(np.float32(1 - D), np.float32(2)) == np.float32(1)


# ------------------------------------------------------- clip_open_box ---

KNOWN = [0.0, 1e-300, 2.0 ** -25, 2.0 ** -24, 0.5, 1 - 2.0 ** -24, 1 - 2.0 ** -25,
         1 - 1e-9, 1 - np.finfo(np.float64).eps, 1.0, np.nextafter(1.0, 0.0),
         np.nextafter(D, 0.0), np.nextafter(D, 1.0), np.nextafter(1 - D, 1.0), 5e-324]


def test_clip_known_answers_and_float32_inside():
    phi = np.array(KNOWN, dtype=np.float64)
    out = L.clip_open_box(phi)
    expect = np.minimum(np.maximum(phi, D), 1 - D)          # by hand
    npt.assert_array_equal(out, expect)                     # exact: clip is exact arithmetic
    f32 = out.astype(np.float32)
    assert np.all(f32 > 0) and np.all(f32 < 1)
    inside = (phi >= D) & (phi <= 1 - D)
    # bit for bit where already inside
    assert np.array_equal(out[inside].view(np.uint64), phi[inside].view(np.uint64))
    # the old clip's failure, reproduced: float32 of clip(1 - 1e-9, eps, 1 - eps) is 1.0
    eps = np.finfo(np.float64).eps
    assert np.float32(np.clip(1 - 1e-9, eps, 1 - eps)) == np.float32(1.0)


def test_clip_dense_boundary_sweep():
    # every float64 within 4096 ulps of each end, plus a random sample: float32 of the result inside
    lo = D * (1 + np.arange(-4096, 4096) * 2.0 ** -52)
    hi = 1 - D * (1 + np.arange(-4096, 4096) * 2.0 ** -52)
    rnd = np.random.default_rng(0).uniform(0, 1, 100000)
    phi = np.clip(np.concatenate([lo, hi, rnd, [0.0, 1.0]]), 0.0, 1.0)
    f32 = L.clip_open_box(phi).astype(np.float32)
    assert np.all(f32 > 0) and np.all(f32 < 1)


def test_clip_new_array_input_unchanged_shapes():
    phi = np.array([[0.0, 0.3], [1.0, 0.7]])
    before = phi.copy()
    out = L.clip_open_box(phi)
    npt.assert_array_equal(phi, before)                 # input not changed
    assert out is not phi and not np.shares_memory(out, phi)
    assert out.dtype == np.float64 and out.shape == (2, 2)
    z = L.clip_open_box(np.float64(1.0))                # 0-d
    assert isinstance(z, np.ndarray) and z.shape == () and float(z) == 1 - D
    lst = L.clip_open_box([0.0, 1.0])                   # array_like
    npt.assert_array_equal(lst, [D, 1 - D])
    e = L.clip_open_box(np.zeros((0, 10)))
    assert e.shape == (0, 10)
    f32in = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    o = L.clip_open_box(f32in)
    assert o.dtype == np.float64 and not np.shares_memory(o, f32in)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_clip_non_finite_raises(bad):
    phi = np.array([0.5, bad, 0.2])
    with pytest.raises(FloatingPointError):
        L.clip_open_box(phi)


# ----------------------------------------------------------- samplers ---

def _spec(centre, tau, n_latent=4, label=(0, 1)):
    cc = np.full((3, len(label)), centre)
    return L.LatentSBISpec(n_latent=n_latent, label_idx=label, class_centres=cc, tau_ov=tau)


@pytest.mark.parametrize("centre", [1e-6, 1 - 1e-6])
def test_samplers_against_base_commit(old_L, centre):
    """Centres at the edges with tau 1e-6 push truncnorm draws into the
    float64 band the new clip moves (P(x < 2^-24) about 0.17 per draw)."""
    sp = _spec(centre, 1e-6)
    for name, call in (
            ("sample_prior", lambda m, s: m.sample_prior(s, 400, np.random.default_rng(5))[1]),
            ("sample_prior_given_classes",
             lambda m, s: m.sample_prior_given_classes(s, np.arange(400) % 3, np.random.default_rng(5))),
            ("sample_box_prior", lambda m, s: m.sample_box_prior(s, 400, np.random.default_rng(5))[1])):
        new = call(L, sp)
        old = call(old_L, sp)
        assert new.dtype == np.float64 and new.shape == old.shape
        inside = (old >= D) & (old <= 1 - D)
        assert np.array_equal(new[inside].view(np.uint64), old[inside].view(np.uint64)), name
        npt.assert_array_equal(new[~inside], np.clip(old[~inside], D, 1 - D), err_msg=name)
        f32 = new.astype(np.float32)
        assert np.all(f32 > 0) and np.all(f32 < 1), name
        if name != "sample_box_prior":
            # the edge case is really exercised: some draws were moved
            assert (~inside).sum() > 0, name


def test_sample_prior_classes_unchanged(old_L):
    sp = _spec(0.5, 0.1)
    c_new, _ = L.sample_prior(sp, 300, np.random.default_rng(9))
    c_old, _ = old_L.sample_prior(sp, 300, np.random.default_rng(9))
    npt.assert_array_equal(c_new, c_old)
    c_box, _ = L.sample_box_prior(sp, 10, np.random.default_rng(9))
    assert c_box.dtype == np.int64 and set(c_box.tolist()) == {-1}


class _ChosenRng(object):
    """A generator whose uniform draws are chosen values (spec's reference oracle)."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def uniform(self, lo, hi, size=None):
        n = int(np.prod(size))
        return np.resize(self.values, n).reshape(size)


def test_box_sampler_chosen_values():
    vals = [0.0, 1 - 1e-9, 1.0, 1e-300, 0.25, 1 - 2.0 ** -25]
    sp = _spec(0.5, 0.1, n_latent=6, label=(0, 1))
    c, phi = L.sample_box_prior(sp, 1, _ChosenRng(vals))
    npt.assert_array_equal(phi[0], np.minimum(np.maximum(vals, D), 1 - D))
    assert np.all(phi.astype(np.float32) < 1) and np.all(phi.astype(np.float32) > 0)


# ------------------------------------------------------- write_shard ---

def test_write_shard_folder_race(tmp, banks, monkeypatch):
    """Another job created the folder between isdir and makedirs (the spec's
    race of the K shard jobs): write_shard must not raise FileExistsError.
    The arrays and the sidecar are a real shard's, so the contract checks
    that precede the folder step pass and the folder step is reached."""
    src = os.path.join(banks["S"][0], "shard_0000")
    with np.load(src + ".npz") as z:
        arrays = dict((k, np.array(z[k])) for k in z.files)
    sidecar = json.load(open(src + ".json"))
    out = os.path.join(tmp, "race", "bank")       # absent at write_shard's isdir check
    real_makedirs = os.makedirs

    def racing_makedirs(name, *a, **k):
        # another shard job creates the folder first, then this one's call runs
        if os.path.abspath(name) == os.path.abspath(out) and not os.path.exists(name):
            real_makedirs(name)
        return real_makedirs(name, *a, **k)

    monkeypatch.setattr(LB.os, "makedirs", racing_makedirs)
    LB.write_shard(os.path.join(out, "shard_0000.npz"), arrays, sidecar)
    monkeypatch.undo()
    assert os.path.isfile(os.path.join(out, "shard_0000.npz"))
    assert os.path.isfile(os.path.join(out, "shard_0000.json"))


# ----------------------------------------------- builder: the defence ---

def _run_builder_patched(out, args, patch):
    code = ("import sys; sys.path.insert(0, %r)\n"
            "import numpy as np\n"
            "import latent_sbi_simulator as L\n"
            "import build_latent_bank as B\n"
            "%s\n"
            "sys.exit(B.main(%r))\n" % (STAGE1, patch, ["--out-dir", out] + args))
    return subprocess.run([PY, "-c", code], cwd=STAGE1, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)


def test_builder_refuses_stored_theta_on_the_boundary(tmp):
    out = os.path.join(tmp, "defence")
    patch = ("def _box(spec, n, rng):\n"
             "    phi = rng.uniform(0.0, 1.0, size=(int(n), spec.n_latent))\n"
             "    phi[0, 1] = 1 - 1e-9\n"
             "    return np.full(int(n), -1, dtype=np.int64), phi\n"
             "B.sample_box_prior = _box\n")
    r = _run_builder_patched(out, ["--arm", "S", "--prior", "box", "--n-traces", "4", "--seed", "3"] + REF,
                             patch)
    assert r.returncode != 0, r.stdout[-1500:]
    assert "outside the open unit box" in r.stdout and "row 0, axis 1" in r.stdout, r.stdout[-1500:]
    written = [f for f in os.listdir(out)] if os.path.isdir(out) else []
    assert not any(f.endswith(".npz") or f.endswith(".json") for f in written), written


def test_builder_nan_theta_probe(tmp):
    """Probe beyond the spec's wording (silent-failure hunt): a NaN
    coordinate fails both of the builder's comparisons (TH <= 0, TH >= 1).
    The samplers cannot produce one (clip_open_box raises), so this probes
    the defence alone, with a sampler patched past the clip. Recorded as a
    NOTE in the report; the assertion is the spec's reachability claim."""
    with pytest.raises(FloatingPointError):
        L.clip_open_box(np.array([[0.5, np.nan]]))
    out = os.path.join(tmp, "nan_probe")
    patch = ("def _box(spec, n, rng):\n"
             "    phi = rng.uniform(0.0, 1.0, size=(int(n), spec.n_latent))\n"
             "    phi[0, 1] = float('nan')\n"
             "    return np.full(int(n), -1, dtype=np.int64), phi\n"
             "B.sample_box_prior = _box\n")
    r = _run_builder_patched(out, ["--arm", "S", "--prior", "box", "--n-traces", "4", "--seed", "3"] + REF,
                             patch)
    wrote = os.path.isfile(os.path.join(out, "shard_0000.npz"))
    stored_nan = False
    if wrote:
        with np.load(os.path.join(out, "shard_0000.npz")) as z:
            stored_nan = bool(np.isnan(z["theta"]).any())
    print("NAN_PROBE rc=%d wrote=%s stored_nan=%s tail=%r" % (r.returncode, wrote, stored_nan,
                                                               r.stdout[-300:]))


# ----------------------------------------------------- the record tool ---

def _build(out, args, stage1=STAGE1):
    r = subprocess.run([PY, os.path.join(stage1, "build_latent_bank.py"), "--out-dir", out] + args,
                       cwd=stage1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=600)
    assert r.returncode == 0, r.stdout[-1500:]


def _norm(argv):
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


def _expect(bank, wells, n_windows, n_classes=3, dpc=0):
    side = json.load(open(os.path.join(bank, "shard_0000.json")))
    with np.load(os.path.join(bank, "shard_0000.npz")) as z:
        rows = int(z["x"].shape[0])
    arm = side["arm"]
    return {"bank": "t", "dir": bank, "arm": arm, "theta_withheld": arm == "R", "W": int(side["W"]),
            "d_theta": len(side["param_names"]),
            "latent_kind": (side.get("latent_spec") or {}).get("kind"),
            "wells_per_donor": wells, "n_windows": n_windows, "n_classes": n_classes,
            "donors_per_class": dpc,
            "build": {"schema": "synth_bank_build/1", "bank": "t", "arm": arm, "n_shards": 1,
                      "rows_per_shard": [rows], "argv": [_norm(side["provenance"]["argv"])]}}


def _record(bank, exp, tmp):
    ep = os.path.join(tmp, "exp_%d.json" % (abs(hash(bank)) % 10 ** 8))
    json.dump(exp, open(ep, "w"))
    out = os.path.join(bank, "BANK_RECORD.json")
    r = subprocess.run([PY, RECORD_TOOL, "--bank-dir", bank, "--expect", ep, "--out", out],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=300)
    return r.returncode, r.stdout, out


def _set_theta(src, dst, row, axis, value):
    shutil.copytree(src, dst)
    p = os.path.join(dst, "shard_0000.npz")
    with np.load(p) as z:
        arrs = dict((k, np.array(z[k])) for k in z.files)
    arrs["theta"][row, axis] = value
    np.savez_compressed(p, **arrs)


@pytest.fixture(scope="module")
def banks(tmp):
    s = os.path.join(tmp, "bankS")
    _build(s, ["--arm", "S", "--prior", "box", "--n-traces", "4", "--seed", "7"] + REF)
    r = os.path.join(tmp, "bankR")
    _build(r, ["--arm", "R", "--n-classes", "3", "--donors-per-class", "1", "--n-traces", "6",
               "--class-centres", "regular", "--centre-scale", "0.6", "--seed", "8"] + REF)
    return {"S": (s, _expect(s, 2, 2, dpc=0)), "R": (r, _expect(r, 2, 2, dpc=1))}


LINE = re.compile(r"^REFUSE: shard 0: (\d+) row\(s\) with a theta outside the open unit box "
                  r"\(first: row (\d+), axis (\d+), value (\S+)\)$", re.M)


def test_record_clean_banks_pass(banks, tmp):
    for arm in ("S", "R"):
        bank, exp = banks[arm]
        clean = os.path.join(tmp, "clean_" + arm)
        shutil.copytree(bank, clean)
        exp = dict(exp, dir=clean)
        rc, text, out = _record(clean, exp, tmp)
        assert rc == 0, text[-1500:]
        assert "SYNTH BANK RECORD OK" in text
        assert os.path.isfile(out) and os.path.isfile(out + ".sha256")


@pytest.mark.parametrize("arm", ["S", "R"])
@pytest.mark.parametrize("value,label", [(1.0, "one"), (0.0, "zero"), (np.nan, "nan"),
                                         (-0.0, "negzero"), (np.inf, "inf"), (1.5, "above")])
def test_record_refuses_theta_outside(banks, tmp, arm, value, label):
    bank, exp = banks[arm]
    dst = os.path.join(tmp, "bad_%s_%s" % (arm, label))
    _set_theta(bank, dst, 2, 3, value)
    rc, text, out = _record(dst, dict(exp, dir=dst), tmp)
    assert rc == 4, text[-1500:]
    m = LINE.search(text)
    assert m, text[-1500:]
    assert (m.group(1), m.group(2), m.group(3)) == ("1", "2", "3"), m.group(0)
    assert not os.path.exists(out) and not os.path.exists(out + ".sha256")


def test_record_accepts_float32_ends(banks, tmp):
    """The closest float32 values to the ends are inside: accepted."""
    bank, exp = banks["S"]
    dst = os.path.join(tmp, "ends_S")
    _set_theta(bank, dst, 0, 0, np.float32(1 - D))
    p = os.path.join(dst, "shard_0000.npz")
    with np.load(p) as z:
        arrs = dict((k, np.array(z[k])) for k in z.files)
    arrs["theta"][1, 0] = np.float32(D)
    arrs["theta"][1, 1] = np.nextafter(np.float32(0), np.float32(1))     # smallest subnormal
    np.savez_compressed(p, **arrs)
    rc, text, _ = _record(dst, dict(exp, dir=dst), tmp)
    assert rc == 0, text[-1500:]


# ------------------------------------------ invariants against the base ---

def test_banks_unchanged_against_base(tmp, old_tree):
    old_stage1 = os.path.join(old_tree, "hpc", "joint", "stage1")
    cases = {
        "mixture": ["--arm", "S", "--n-traces", "4", "--seed", "4"] + REF,
        "box": ["--arm", "S", "--prior", "box", "--n-traces", "4", "--seed", "4"] + REF,
        "cohort": ["--arm", "R", "--n-classes", "3", "--donors-per-class", "1", "--n-traces", "6",
                   "--class-centres", "regular", "--centre-scale", "0.6", "--seed", "5"] + REF,
    }
    for name, args in cases.items():
        a, b = os.path.join(tmp, "inv_new_" + name), os.path.join(tmp, "inv_old_" + name)
        _build(a, args)
        _build(b, args, stage1=old_stage1)
        with np.load(os.path.join(a, "shard_0000.npz")) as za, np.load(os.path.join(b, "shard_0000.npz")) as zb:
            assert sorted(za.files) == sorted(zb.files)
            for k in za.files:
                assert za[k].dtype == zb[k].dtype and za[k].tobytes() == zb[k].tobytes(), (name, k)
            th = za["theta"]
            assert th.dtype == np.float32 and np.all(th > 0) and np.all(th < 1)
        sa = json.load(open(os.path.join(a, "shard_0000.json")))
        sb = json.load(open(os.path.join(b, "shard_0000.json")))
        assert sa["generator_sha256"] == sb["generator_sha256"], name
        assert sa["contract_digest"] == sb["contract_digest"], name
        assert LB.contract_digest(sa) == sa["contract_digest"], name


def test_build_deterministic(tmp):
    args = ["--arm", "S", "--prior", "box", "--n-traces", "4", "--seed", "12"] + REF
    a, b = os.path.join(tmp, "det_a"), os.path.join(tmp, "det_b")
    _build(a, args)
    _build(b, args)
    with np.load(os.path.join(a, "shard_0000.npz")) as za, np.load(os.path.join(b, "shard_0000.npz")) as zb:
        for k in za.files:
            assert za[k].tobytes() == zb[k].tobytes(), k


def test_record_empty_shard_refused_not_crash(banks, tmp):
    """A shard whose arrays hold no row: the spec's checks (rows =
    rows_per_shard) make it a refusal -- REFUSE lines, exit 4, nothing
    written -- never a traceback (a crash is resubmitted as a job that
    ended without its record, block 16)."""
    bank, exp = banks["S"]
    dst = os.path.join(tmp, "empty_S")
    shutil.copytree(bank, dst)
    p = os.path.join(dst, "shard_0000.npz")
    with np.load(p) as z:
        arrs = dict((k, np.array(z[k])[:0]) for k in z.files)
    np.savez_compressed(p, **arrs)
    rc, text, out = _record(dst, dict(exp, dir=dst), tmp)
    assert "Traceback" not in text, text[-1500:]
    assert rc == 4 and "REFUSE:" in text, text[-1500:]
    assert not os.path.exists(out)
