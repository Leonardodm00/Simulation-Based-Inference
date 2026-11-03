#!/usr/bin/env python3
"""Smoke test for blocks 11-14 of specs/SPEC.md: the synthetic A2/A5
experiment's banks and arms (roadmap synth, steps S2 and S3; D-118..D-121).

Checks (the SPEC's oracles; B11..B14 name the block)
    test_b11_box_bank        --prior box: theta uniform on the box (KS per axis),
                             cls -1, latent_spec kind box, the wells of a donor
                             share theta; refused on arm R
    test_b11_mixture_keys    the default path's latent_spec has exactly
                             LatentSBISpec.to_dict()'s keys (digest unchanged)
    test_b12_cohort          --donors-per-class with regular centres: N donors
                             per class, the centres recomputed from the DSN's
                             simplex, equidistant, the per-class means of the
                             label axes at the truncated-normal means, batches
                             mixing classes, the design recorded; the refusals
    test_b12_ids             two shards: bank-wide donor, well and batch ids
                             (shard k offset by k * ID_STRIDE), none shared
    test_b13_duty_clip       the shifted duty by hand (clipped, unclipped, the
                             identity at shift 0); a pi = 0.3 bench cohort builds
    test_b14_pairs           pairs across two wells of one donor only; the old
                             pairing without wells; the round-robin spread
    test_b14_class_count     the metric term's classes counted on its source
    test_b14_runner_arms     A1, A2, A5 on a box arm S and a 3-class cohort: the
                             box prior, 3 metric classes, the cross-well pairs
    test_b14_runner_refusals F-aa (one well per donor), F-aw (one seed), F-t
                             (A2s on a box bank)
    test_convergence         skipped (not a discretisation)

Run
    cd hpc/joint/stage3 && python3 smoke_test_blocks_11_14.py
        About 2-4 minutes at one thread. Needs numpy and scipy for B11-B13;
        torch, sbi and zuko (sbi_env) and the in-repo DSN tree (hpc/dsn) for
        the runner checks of B14. Builds its own banks in a temporary folder
        with the builder's reference provider (and the bench provider for
        B13's shard); nothing is read from davinci.

Pure ASCII, LF only.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_S1 = os.path.abspath(os.path.join(_HERE, "..", "stage1"))
for _p in (_HERE, _S1, os.path.join(_HERE, "..", "stage2"), os.path.join(_HERE, "..")):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import latent_bank as LB            # noqa: E402

BUILDER = os.path.join(_S1, "build_latent_bank.py")
RUNNER = os.path.join(_HERE, "run_joint_arms.py")
PYTHON = sys.executable
DSN_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "dsn"))

_SHARED = {}


# --------------------------------------------------------------- helpers ---

def tmp_root():
    if "root" not in _SHARED:
        _SHARED["root"] = tempfile.mkdtemp(prefix="blocks1114_")
    return _SHARED["root"]


def build(out, args, expect_ok=True):
    """Run the builder; returns (rc, combined output)."""
    cmd = [PYTHON, BUILDER, "--out-dir", out] + [str(a) for a in args]
    r = subprocess.run(cmd, cwd=_S1, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=900)
    if expect_ok and r.returncode != 0:
        raise AssertionError("builder failed (%d): %s"
                             % (r.returncode, r.stdout.strip()[-600:]))
    return r.returncode, r.stdout


def read(out, k=0):
    return LB.read_shard(os.path.join(out, "shard_%04d.npz" % k))


# Small reference-provider shards (the fixture generator: plumbing, not
# science), W = 64 samples.
REF = ["--provider", "reference", "--T-win", "1.28", "--fs", "50",
       "--n-neurons", "20", "--n-windows", "2"]


# ---------------------------------------------------------------- checks ---

def test_b11_box_bank():
    from scipy import stats
    out = os.path.join(tmp_root(), "b11_box")
    build(out, ["--arm", "S", "--prior", "box", "--n-traces", "800",
                "--n-latent", "6", "--n-label-axes", "3", "--seed", "11"] + REF)
    a, side = read(out)
    th = np.asarray(a["theta"], dtype=np.float64)
    # one theta per donor (first row of each donor)
    don = np.asarray(a["donor"])
    first = np.unique(don, return_index=True)[1]
    td = th[first]
    assert td.shape == (400, 6), td.shape
    # oracle: U(0, 1) on every axis; 400 donors, KS p > 1e-3 per axis (the
    # bound is loose on purpose: 6 tests, a false alarm has p < 0.006)
    for k in range(6):
        p = stats.kstest(td[:, k], "uniform").pvalue
        assert p > 1e-3, "axis %d KS p = %.2e" % (k, p)
    assert set(np.unique(a["cls"]).tolist()) == {-1}, np.unique(a["cls"])
    assert side["latent_spec"].get("kind") == "box", side["latent_spec"]
    assert np.all((th > 0.0) & (th < 1.0))
    # the two wells of a donor share theta exactly (theta per donor)
    for d in np.unique(don)[:20]:
        rows = th[don == d]
        assert np.all(rows == rows[0]), d
        assert np.unique(np.asarray(a["well"])[don == d]).size == 2
    rc, txt = build(os.path.join(tmp_root(), "b11_boxR"),
                    ["--arm", "R", "--prior", "box", "--n-traces", "8"] + REF,
                    expect_ok=False)
    assert rc != 0 and "--prior box is arm S's" in txt, txt[-300:]


def test_b11_mixture_keys():
    from latent_sbi_simulator import LatentSBISpec
    out = os.path.join(tmp_root(), "b11_mix")
    build(out, ["--arm", "S", "--n-traces", "16", "--seed", "3"] + REF)
    _a, side = read(out)
    expected = set(LatentSBISpec().to_dict())
    got = set(side["latent_spec"])
    assert got == expected, (sorted(got ^ expected))


def _dsn_regular(C, L):
    sys.path.insert(0, DSN_DIR)
    import latent_burst_generator as lbg
    return np.asarray(lbg._class_center_vectors(C, L, "simplex"), dtype=np.float64)


def test_b12_cohort():
    from scipy import stats
    out = os.path.join(tmp_root(), "b12_cohort")
    # 3 classes x 150 donors x 2 wells, regular centres at scale 0.6
    build(out, ["--arm", "R", "--n-classes", "3", "--donors-per-class", "150",
                "--n-traces", "900", "--wells-per-donor", "2",
                "--class-centres", "regular", "--centre-scale", "0.6",
                "--n-latent", "6", "--n-label-axes", "3", "--seed", "21"] + REF)
    a, side = read(out)
    don, cls = np.asarray(a["donor"]), np.asarray(a["cls"])
    first = np.unique(don, return_index=True)[1]
    cd = cls[first]
    assert np.bincount(cd, minlength=3).tolist() == [150, 150, 150], np.bincount(cd)
    # centres: recomputed independently from the DSN's simplex
    cen = np.asarray(side["latent_spec"]["class_centres"], dtype=np.float64)
    expect = 0.5 + 0.6 * (_dsn_regular(3, 3) - 0.5)
    assert np.allclose(cen, expect, rtol=0, atol=1e-12), (cen, expect)
    dist = [np.linalg.norm(cen[i] - cen[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    assert max(dist) - min(dist) < 1e-12, dist          # float64 roundoff
    ls = side["latent_spec"]
    assert ls.get("class_centres_mode") == "regular" and ls.get("centre_scale") == 0.6
    assert ls.get("donors_per_class") == 150, ls
    # per-class means of the label axes against the truncated-normal means
    # (tau 0.10 on (0, 1)); 150 donors per class: 5 standard errors
    th = np.asarray(a["theta"], dtype=np.float64)[first]
    tau = float(ls["tau_ov"])
    for c in range(3):
        for j, ax in enumerate(ls["label_idx"]):
            mu = cen[c, j]
            d = stats.truncnorm((0 - mu) / tau, (1 - mu) / tau, loc=mu, scale=tau)
            m = th[cd == c, ax]
            se = d.std() / np.sqrt(m.size)
            assert abs(m.mean() - d.mean()) < 5 * se, (c, ax, m.mean(), d.mean(), se)
    # the wells of a donor share theta and class
    for d0 in np.unique(don)[:20]:
        assert np.unique(cls[don == d0]).size == 1
        assert np.all(np.asarray(a["theta"])[don == d0] == np.asarray(a["theta"])[don == d0][0])
    # batches of 8 consecutive donors mix the classes (a permutation, seed 21)
    bat = np.asarray(a["batch"])[first]
    mixed = [np.unique(cd[bat == b]).size for b in np.unique(bat)]
    assert max(mixed) >= 2 and np.mean(np.asarray(mixed) >= 2) > 0.9, mixed
    # refusals
    t = os.path.join(tmp_root(), "b12_ref")
    for extra, msg in (
            (["--donors-per-class", "4", "--n-traces", "20"], "is 24 wells"),
            (["--prior", "box", "--donors-per-class", "4", "--n-traces", "24"],
             "--prior box has none"),
            (["--centre-scale", "0.6", "--n-traces", "8"],
             "--centre-scale needs --class-centres regular")):
        arm = "S" if "--prior" in extra else "R"
        rc, txt = build(t, ["--arm", arm, "--n-classes", "3"] + extra + REF,
                        expect_ok=False)
        assert rc != 0 and msg in txt, (extra, txt[-300:])


def test_b12_ids():
    out = os.path.join(tmp_root(), "b12_ids")
    for k in (0, 1):
        build(out, ["--arm", "S", "--prior", "box", "--n-traces", "12",
                    "--donors-per-batch", "4", "--shard-index", str(k),
                    "--seed", "5"] + REF)
    from build_latent_bank import ID_STRIDE
    a0, _ = read(out, 0)
    a1, _ = read(out, 1)
    for f in ("donor", "well", "batch"):
        v0, v1 = np.asarray(a0[f]), np.asarray(a1[f])
        assert v0.min() >= 0 and v0.max() < ID_STRIDE, (f, v0.min(), v0.max())
        assert v1.min() >= ID_STRIDE and v1.max() < 2 * ID_STRIDE, (f, v1.min())
        assert np.array_equal(v1 - ID_STRIDE, v0), f   # same local layout
        assert not set(v0.tolist()) & set(v1.tolist()), f
    merged, _side = LB.concat_shards([os.path.join(out, "shard_0000.npz"),
                                      os.path.join(out, "shard_0001.npz")])
    assert np.unique(merged["donor"]).size == 12, np.unique(merged["donor"]).size


def test_b13_duty_clip():
    from bench_burst_provider import phi_to_bench_params
    phi = np.full(10, 0.5)
    shift = 0.3 * 0.35
    phi[7] = 0.95
    p = phi_to_bench_params(phi, n_neurons=10, duration_s=10.0, free_axis_shift=shift)
    assert p.duty == 1.0, p.duty        # 0.41825 * 0.05 + 1.06825 * 0.95 = 1.0357 -> 1
    phi[7] = 0.5
    p = phi_to_bench_params(phi, n_neurons=10, duration_s=10.0, free_axis_shift=shift)
    lo, hi = 0.35 + shift * 0.65, 1.00 + shift * 0.65
    assert abs(p.duty - (lo * 0.5 + hi * 0.5)) < 1e-15, p.duty
    phi[7] = 1.0
    p = phi_to_bench_params(phi, n_neurons=10, duration_s=10.0, free_axis_shift=0.0)
    assert p.duty == 1.0, p.duty        # the identity at shift 0: U_7 exactly
    # a pi = 0.3 bench cohort of 27 donors builds (it raised before, F-av)
    out = os.path.join(tmp_root(), "b13_pi03")
    build(out, ["--arm", "R", "--provider", "bench", "--n-classes", "3",
                "--donors-per-class", "9", "--n-traces", "54",
                "--wells-per-donor", "2", "--n-windows", "1", "--T-win", "6",
                "--fs", "100", "--class-centres", "regular", "--centre-scale",
                "0.6", "--pi", "0.3", "--gap-modes",
                "range_shift,drift,contamination", "--seed", "31"])
    a, side = read(out)
    th = np.asarray(a["theta"], dtype=np.float64)
    first = np.unique(np.asarray(a["donor"]), return_index=True)[1]
    n_clip = int(np.sum(th[first, 7] > 1.0 - shift))
    assert n_clip >= 1, "no donor exercised the clip at seed 31"
    _SHARED["b13_n_clip"] = n_clip


def test_b14_pairs():
    from joint_batches import enumerate_donor_pairs
    from run_joint_arms import spread_pairs
    don = [0, 0, 0, 0, 1, 1, 2, 2]
    wel = [0, 0, 1, 1, 2, 2, 3, 4]
    p, n1 = enumerate_donor_pairs(don, wel)
    assert sorted(map(tuple, p.tolist())) == [(0, 2), (0, 3), (1, 2), (1, 3), (6, 7)], p
    assert n1 == 1, n1                  # donor 1: one well
    p_old, n_old = enumerate_donor_pairs(don)
    assert p_old.shape[0] == 6 + 1 + 1 and n_old == 0, (p_old, n_old)
    # round-robin: 3 pairs over 2 donors take donor 0's first, donor 2's
    # first, then donor 0's second
    take = spread_pairs(p, np.asarray(don), 3)
    assert [tuple(r) for r in take.tolist()] == [(0, 2), (6, 7), (0, 3)], take


def test_b14_class_count():
    from run_joint_arms import metric_class_count
    assert metric_class_count([0, 1, 2, 2, -1], "real") == 3
    for bad in ([-1, -1, -1], [0, 0, -1], [0, 2, 2]):
        try:
            metric_class_count(bad, "real")
        except SystemExit:
            continue
        raise AssertionError("accepted %r" % (bad,))


def _need_torch():
    try:
        import torch  # noqa: F401
        import sbi  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("needs torch and sbi: %s" % exc)


def _runner_banks():
    """A box arm S (40 donors x 2 wells x 2 windows) on seed 1 and a 3-class
    cohort arm R (4 donors per class x 2 wells x 2 windows) on seed 2."""
    if "rb" in _SHARED:
        return _SHARED["rb"]
    root = os.path.join(tmp_root(), "runner")
    S, R, R1 = (os.path.join(root, n) for n in ("S", "R", "R1"))
    build(S, ["--arm", "S", "--prior", "box", "--n-traces", "80",
              "--n-classes", "3", "--seed", "1"] + REF)
    build(R, ["--arm", "R", "--n-classes", "3", "--donors-per-class", "4",
              "--n-traces", "24", "--class-centres", "regular",
              "--centre-scale", "0.6", "--seed", "2"] + REF)
    build(R1, ["--arm", "R", "--n-classes", "3", "--donors-per-class", "4",
               "--n-traces", "12", "--wells-per-donor", "1", "--n-per-theta", "1",
               "--seed", "3"] + REF)
    _SHARED["rb"] = (os.path.join(S, "*.npz"), os.path.join(R, "*.npz"),
                     os.path.join(R1, "*.npz"), root)
    return _SHARED["rb"]


def run_arm(arm, sim, real, out_dir, extra=()):
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = [PYTHON, RUNNER, "--arm", arm, "--seed", "0", "--sim-shards", sim,
           "--real-shards", real, "--out-dir", out_dir, "--threads", "1",
           "--epochs", "2", "--steps-per-epoch", "3", "--b-sim", "24",
           "--b-met", "12", "--b-rep", "4", "--n-posterior-draws", "16",
           "--embedding-size", "8", "--hidden-features", "24",
           "--num-transforms", "3", "--depth-exponent", "3",
           "--width-multiplier", "1.5"] + list(extra)
    r = subprocess.run(cmd, cwd=_HERE, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=1200)
    return r.returncode, r.stdout


def test_b14_runner_arms():
    _need_torch()
    sim, real, _r1, root = _runner_banks()
    runs = os.path.join(root, "runs")
    seen = {}
    for arm in ("A1", "A2", "A5"):
        rc, txt = run_arm(arm, sim, real, runs)
        assert rc == 0, "%s exit %d: %s" % (arm, rc, txt[-1500:])
        with open(os.path.join(runs, "%s_seed0.json" % arm)) as fh:
            rec = json.load(fh)
        assert rec["prior"] == "box_unit_box", rec["prior"]
        assert rec["L0"] == 0.0, rec["L0"]
        assert rec["endpoint_primary"] == "pseudo_real", rec["endpoint_primary"]
        assert rec["L_pseudo_real"] is not None
        # the cross-well pairs: 12 donors x (2 windows x 2 windows) = 48
        assert "replicate pairs  : 48  (two wells of one donor)" in txt, txt[-2000:]
        rep = rec["replicate"]
        assert rep["n_pairs"] == 48 and rep["n_donors_scored"] == 12, rep
        assert rep["pair_kind"] == "two wells of one donor", rep["pair_kind"]
        seen[arm] = txt
    assert "metric classes   : 3 (on the real arm)" in seen["A2"], seen["A2"][-2000:]
    assert "metric classes" not in seen["A1"] and "metric classes" not in seen["A5"]


def test_b14_runner_refusals():
    _need_torch()
    sim, real, r1, root = _runner_banks()
    runs = os.path.join(root, "runs_refuse")
    rc, txt = run_arm("A5", sim, r1, runs)
    assert rc != 0 and "has no replicate pair" in txt, txt[-800:]
    # the diagnostic alone falls back to rows of one well, and says so
    rc, txt = run_arm("A1", sim, r1, runs)
    assert rc == 0, txt[-1500:]
    with open(os.path.join(runs, "A1_seed0.json")) as fh:
        rep = json.load(fh)["replicate"]
    assert rep["pair_kind"] == "rows of one well (no donor has two wells)", rep
    assert rep["n_pairs"] == 12 and rep["n_donors_scored"] == 12, rep
    # one seed for both arms: a cohort on seed 1, the seed of arm S
    same = os.path.join(root, "R_seed1")
    build(same, ["--arm", "R", "--n-classes", "3", "--donors-per-class", "4",
                 "--n-traces", "24", "--seed", "1"] + REF)
    rc, txt = run_arm("A1", sim, os.path.join(same, "*.npz"), runs)
    assert rc != 0 and "built with one --seed (1)" in txt, txt[-800:]
    rc, txt = run_arm("A2s", sim, real, runs)
    assert rc != 0 and "needs at least two classes on its simulated source" in txt, \
        txt[-800:]


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


# ---------------------------------------------------------------- runner ---

def main():
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    try:
        import scipy
        vers = "numpy %s | scipy %s" % (np.__version__, scipy.__version__)
    except ImportError:
        vers = "numpy %s" % np.__version__
    try:
        import torch
        import sbi
        vers += " | torch %s | sbi %s" % (torch.__version__, sbi.__version__)
    except ImportError:
        vers += " | torch/sbi absent (the runner checks skip)"
    print("python %s | %s" % (sys.version.split()[0], vers))
    results = []
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
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(r[0]) for r in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline[:150]))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()[-4000:]))
    if "b13_n_clip" in _SHARED:
        print("\n-- B13: %d of 27 donors had the shifted duty clipped" % _SHARED["b13_n_clip"])
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    if "root" in _SHARED:
        shutil.rmtree(_SHARED["root"], ignore_errors=True)
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
