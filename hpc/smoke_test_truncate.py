#!/usr/bin/env python3
"""
smoke_test_truncate.py -- end-to-end checks for prior_truncate.py.

Builds a synthetic fixture with a KNOWN forward map (the same map as
npe_contract.make_synthetic_shard, replicated here for the real arm),
runs the real CLI in a subprocess, and asserts on the outputs.

Ground truth of the fixture: each synthetic "culture" has ONE true theta
drawn from a central sub-box of the prior; its windows are noisy pushes of
that theta through the map. A correct truncation box must therefore
contain the generating thetas, sit inside the prior, and respect the
window <= culture <= condition <= common hierarchy.

Checks
------
  S1  the CLI exits 0 and writes truncated_prior.json + truncate_arrays.npz
  S2  every box: finite, lo < hi, inside the prior box
  S3  hierarchy monotone on RAW bounds (window <= culture <= condition <=
      common), envelope mode
  S4  at least 5/6 culture ground-truth thetas inside the padded common box
  S5  eps is monotone: the raw eps=0.05 box lies inside the eps=1e-3 box
  S6  idempotence: a second run reuses the ensemble and reproduces the
      boxes exactly
  S7  pooled mode runs, sits inside the envelope box, and is informative
  S8  negative paths: rate filter without --activity fails; misaligned
      activity table fails

Cost: about 1-3 minutes on a login node (2 tiny ensemble members).
ASCII-only by policy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import npe_contract as C  # noqa: E402

SCRIPT = os.path.join(HERE, "prior_truncate.py")

N_SIM = 1600
P = 5
E = 6
N_LOG = 3
SEED = 0
CULTURES = ["c%02d" % i for i in range(6)]
COND = {c: ("0" if i < 3 else "1") for i, c in enumerate(CULTURES)}
WIN_PER_CULTURE = 10


def _forward(theta, lo, hi, seed, rng):
    """The make_synthetic_shard map, replicated. If that map changes
    upstream, S4 degrades visibly rather than silently."""
    u = (theta - lo[None, :]) / (hi - lo)[None, :]
    w_rng = np.random.default_rng(seed + 1000)
    W = w_rng.normal(size=(P, E))
    b = w_rng.normal(size=(E,))
    raw = u @ W + b[None, :] + 0.05 * rng.normal(size=(theta.shape[0], E))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def build_fixture(root):
    import pandas as pd

    z, theta, contract, sim_path = C.make_synthetic_shard(
        root, name="sim_000", n_rows=N_SIM, p=P, embedding_dim=E,
        n_log_axes=N_LOG, seed=SEED, write=True)
    lo, hi = contract.low, contract.high

    rng = np.random.default_rng(SEED + 7)
    truth = {}
    rows_z, rows_cu, rows_cl = [], [], []
    for cu in CULTURES:
        th = lo + rng.uniform(0.3, 0.6, size=P) * (hi - lo)
        truth[cu] = th
        zc = _forward(np.tile(th, (WIN_PER_CULTURE, 1)), lo, hi, SEED, rng)
        rows_z.append(zc)
        rows_cu += [cu] * WIN_PER_CULTURE
        rows_cl += [COND[cu]] * WIN_PER_CULTURE
    Zr = np.vstack(rows_z).astype(np.float32)

    cols = {c: Zr[:, j] for j, c in enumerate(contract.z_columns)}
    cols["culture"] = np.array(rows_cu, dtype=object)
    cols["condition"] = np.array(rows_cl, dtype=object)
    real_path = os.path.join(root, "real_cohort.parquet")
    pd.DataFrame(cols).to_parquet(real_path, index=False)
    with open(os.path.join(root, "real_cohort.json"), "w",
              encoding="utf-8") as fh:
        json.dump(contract.to_dict(), fh)

    n_real = Zr.shape[0]
    act_rng = np.random.default_rng(SEED + 11)
    sim_rate = act_rng.uniform(0.05, 5.0, size=N_SIM)   # ~1% below 0.1
    real_rate = act_rng.uniform(0.5, 5.0, size=n_real)  # all pass
    act_path = os.path.join(root, "activity.npz")
    np.savez(act_path, sim_rate=sim_rate, real_rate=real_rate)
    bad_act = os.path.join(root, "activity_bad.npz")
    np.savez(bad_act, sim_rate=sim_rate[:-5], real_rate=real_rate)

    return contract, truth, sim_path, real_path, act_path, bad_act


def run_cli(extra, expect_fail=False):
    cmd = [sys.executable, SCRIPT] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if expect_fail:
        assert r.returncode != 0, \
            "expected failure, got success:\n%s" % r.stdout[-2000:]
        return r
    assert r.returncode == 0, \
        "CLI failed (%d):\nSTDOUT:\n%s\nSTDERR:\n%s" \
        % (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    return r


def bounds(d, key="bounds_theta"):
    return np.asarray(d[key], dtype=np.float64)


def check_box(d, prior_lo, prior_hi, label):
    b = bounds(d)
    assert np.all(np.isfinite(b)), "%s: non-finite bounds" % label
    assert np.all(b[:, 0] < b[:, 1]), "%s: lo >= hi" % label
    assert np.all(b[:, 0] >= prior_lo - 1e-9), "%s: below prior" % label
    assert np.all(b[:, 1] <= prior_hi + 1e-9), "%s: above prior" % label
    assert 0.0 < d["prior_mass_fraction"] <= 1.0 + 1e-12, \
        "%s: bad mass fraction" % label


def main():
    root = tempfile.mkdtemp(prefix="smoke_truncate_")
    try:
        contract, truth, sim_path, real_path, act_path, bad_act = \
            build_fixture(root)
        prior_lo, prior_hi = contract.low, contract.high
        sim_glob = os.path.join(root, "sim_*.parquet")
        out1 = os.path.join(root, "out_envelope")

        base = ["--sim", sim_glob, "--real", real_path,
                "--activity", act_path, "--out", out1,
                "--n_members", "2", "--n_draws", "300",
                "--hidden_features", "32", "--num_transforms", "3",
                "--num_bins", "5", "--max_epochs", "60",
                "--stop_after", "8", "--batch_size", "128",
                "--n_cal", "100", "--chunk", "16", "--seed", "0",
                "--progress_every", "0"]

        # ---- S1: end-to-end run --------------------------------------
        run_cli(list(base))
        jpath = os.path.join(out1, "truncated_prior.json")
        apath = os.path.join(out1, "truncate_arrays.npz")
        assert os.path.isfile(jpath) and os.path.isfile(apath), \
            "S1: outputs missing"
        doc = json.load(open(jpath, "r", encoding="utf-8"))
        arrs = np.load(apath, allow_pickle=False)
        n_real = len(CULTURES) * WIN_PER_CULTURE
        assert arrs["window_lo"].shape == (n_real, P), "S1: window_lo shape"
        print("S1 PASS: CLI ran and wrote both outputs")

        # ---- S2: every box well-formed -------------------------------
        check_box(doc["boxes"]["common"], prior_lo, prior_hi, "common")
        for cl, d in doc["boxes"]["per_condition"].items():
            check_box(d, prior_lo, prior_hi, "condition %s" % cl)
        for cu, d in doc["boxes"]["per_culture"].items():
            check_box(d, prior_lo, prior_hi, "culture %s" % cu)
            assert d["condition"] == COND[cu], "S2: wrong condition tag"
        print("S2 PASS: all boxes finite, ordered, inside the prior")

        # ---- S3: hierarchy monotone on RAW bounds --------------------
        wlo = np.asarray(arrs["window_lo"], dtype=np.float64)
        whi = np.asarray(arrs["window_hi"], dtype=np.float64)
        cu_arr = np.asarray([str(x) for x in arrs["culture"]])
        cl_arr = np.asarray([str(x) for x in arrs["condition"]])
        raw_common = bounds(doc["boxes"]["common"], "raw_bounds_theta")
        tol = 1e-6
        for cu in CULTURES:
            m = cu_arr == cu
            raw_cu = bounds(doc["boxes"]["per_culture"][cu],
                            "raw_bounds_theta")
            assert np.all(wlo[m] >= raw_cu[None, :, 0] - tol) \
                and np.all(whi[m] <= raw_cu[None, :, 1] + tol), \
                "S3: window not inside culture %s" % cu
            raw_cl = bounds(doc["boxes"]["per_condition"][COND[cu]],
                            "raw_bounds_theta")
            assert np.all(raw_cu[:, 0] >= raw_cl[:, 0] - tol) \
                and np.all(raw_cu[:, 1] <= raw_cl[:, 1] + tol), \
                "S3: culture %s not inside its condition" % cu
        for cl, d in doc["boxes"]["per_condition"].items():
            raw_cl = bounds(d, "raw_bounds_theta")
            assert np.all(raw_cl[:, 0] >= raw_common[:, 0] - tol) \
                and np.all(raw_cl[:, 1] <= raw_common[:, 1] + tol), \
                "S3: condition %s not inside common" % cl
        assert np.all(cl_arr == np.asarray([COND[c] for c in cu_arr])), \
            "S3: condition labels misaligned"
        print("S3 PASS: window <= culture <= condition <= common")

        # ---- S4: ground truth covered --------------------------------
        b = bounds(doc["boxes"]["common"])
        n_in = sum(bool(np.all((t >= b[:, 0]) & (t <= b[:, 1])))
                   for t in truth.values())
        assert n_in >= 5, "S4: only %d/6 true thetas inside the box" % n_in
        print("S4 PASS: %d/6 generating thetas inside the padded common box"
              % n_in)

        # ---- S5: eps bites, monotonically ----------------------------
        # At eps = 1e-3 with a deliberately weak fixture flow the box may
        # legitimately saturate the prior (min/max extents measure the
        # flow's tails). The implementation invariant is monotonicity: a
        # larger eps raises every tau_j and can only tighten every raw
        # extent. Check that, plus informativeness at eps = 0.05.
        out3 = os.path.join(root, "out_eps")
        eps_args = list(base)
        eps_args[eps_args.index("--out") + 1] = out3
        eps_args += ["--eps", "0.05",
                     "--ensemble_dir", os.path.join(out1, "ensemble")]
        run_cli(eps_args)
        doce = json.load(open(os.path.join(out3, "truncated_prior.json"),
                              "r", encoding="utf-8"))
        re_ = bounds(doce["boxes"]["common"], "raw_bounds_theta")
        assert np.all(re_[:, 0] >= raw_common[:, 0] - tol) \
            and np.all(re_[:, 1] <= raw_common[:, 1] + tol), \
            "S5: eps=0.05 box exceeds the eps=1e-3 box"
        assert np.any(re_[:, 0] > raw_common[:, 0] + 1e-6) \
            or np.any(re_[:, 1] < raw_common[:, 1] - 1e-6), \
            "S5: raising eps tightened nothing"
        print("S5 PASS: eps monotone (raw eps=0.05 box inside eps=1e-3 box)")

        # ---- S6: idempotence -----------------------------------------
        r = run_cli(list(base))
        assert "loading existing ensemble" in r.stdout, \
            "S6: second run retrained instead of reusing"
        doc2 = json.load(open(jpath, "r", encoding="utf-8"))
        assert np.allclose(bounds(doc["boxes"]["common"]),
                           bounds(doc2["boxes"]["common"]),
                           rtol=0, atol=0), "S6: boxes not reproduced"
        print("S6 PASS: ensemble reused, boxes reproduced exactly")

        # ---- S7: pooled mode inside envelope -------------------------
        out2 = os.path.join(root, "out_pooled")
        pooled_args = list(base)
        pooled_args[pooled_args.index("--out") + 1] = out2
        pooled_args += ["--box_mode", "pooled", "--eps", "0.05",
                        "--ensemble_dir", os.path.join(out1, "ensemble")]
        run_cli(pooled_args)
        docp = json.load(open(os.path.join(out2, "truncated_prior.json"),
                              "r", encoding="utf-8"))
        rp = bounds(docp["boxes"]["common"], "raw_bounds_theta")
        assert np.all(rp[:, 0] >= raw_common[:, 0] - tol) \
            and np.all(rp[:, 1] <= raw_common[:, 1] + tol), \
            "S7: pooled box exceeds the envelope box"
        mfp = docp["boxes"]["common"]["prior_mass_fraction"]
        assert mfp < 0.9, \
            "S7: pooled box uninformative, retains %.3f of prior mass" % mfp
        print("S7 PASS: pooled common box inside the envelope box and "
              "informative (%.3g of prior mass)" % mfp)

        # ---- S8: negative paths --------------------------------------
        neg = [a for a in base if a != "--activity" and a != act_path]
        run_cli(neg + ["--out", os.path.join(root, "out_neg1")],
                expect_fail=True)
        bad = list(base)
        bad[bad.index("--activity") + 1] = bad_act
        bad[bad.index("--out") + 1] = os.path.join(root, "out_neg2")
        run_cli(bad, expect_fail=True)
        print("S8 PASS: missing --activity and misaligned table both refuse")

        print("ALL CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
