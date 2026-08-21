#!/usr/bin/env python3
"""Smoke test for gate_data / gate_run / gate_plots.

Builds a synthetic fixture in a temp dir that mimics the real export layout
(parquet shards with z_*, zraw_*, th_*, ident columns, plus JSON sidecars),
then runs the whole pipeline on it. It checks the PLUMBING and the selection
logic; it does NOT re-validate the gate's statistics, which have their own
suite (smoke_test_misspec.py, G0-G7).

Two cases are exercised deliberately:
  WELL-SPECIFIED  real drawn from the same distribution as sim -> expect the
                  gate to pass.
  MISSPECIFIED    real shifted -> expect the gate to reject.
A test that only ever runs the passing case cannot tell a working gate from
one wired to a constant.

USAGE
    python3 smoke_test_gate.py --misspec_dir /path/to/.../hpc
    python3 smoke_test_gate.py --misspec_dir ... --keep   # keep the fixture
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

E = 14
P = 26
PARAM_NAMES = ["a%02d" % i for i in range(P - 3)] + ["p0_conn", "d0_conn",
                                                    "beta_conn"]
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print("  %-4s %-34s %s" % ("PASS" if cond else "FAIL", name, detail))
    return bool(cond)


def _sidecar(path, n_rows, digest="a" * 64):
    doc = {
        "schema_version": 1, "n_rows": int(n_rows),
        "n_traces_used": int(n_rows), "n_traces_skipped_too_short": 0,
        "embedding": {"embedding_dim": E, "dsn_checkpoint_sha256": digest,
                      "l2_normalised": True, "zraw_available": True},
        "param_names": PARAM_NAMES,
        "coord": ["linear"] * P,
        "bounds_theta": [[-5.0, 5.0]] * P,
        "assertions_passed": ["A2", "A3", "A4", "A5", "A7", "A9"],
        "warnings": [],
    }
    with open(path, "w") as fh:
        json.dump(doc, fh)


def _write_shard(path, z, zraw, theta=None, ident=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = {}
    for j in range(z.shape[1]):
        cols["z_%03d" % j] = z[:, j]
    for j in range(zraw.shape[1]):
        cols["zraw_%03d" % j] = zraw[:, j]
    if theta is not None:
        for j, nm in enumerate(PARAM_NAMES):
            cols["th_%s" % nm] = theta[:, j]
    for k, v in (ident or {}).items():
        cols[k] = v
    pq.write_table(pa.table(cols), path)
    _sidecar(os.path.splitext(path)[0] + ".json", z.shape[0])


def build_fixture(root, shift=0.0, seed=0, n_topo=12, per_topo=40,
                  n_groups=8, n_sub=3, n_win=6, dup_factor=3):
    """Simulated shards (with deliberate duplicates) plus a real cohort."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)

    # --- simulated: n_topo topologies, per_topo rows each, then duplicated
    thetas, zs = [], []
    for t in range(n_topo):
        topo = rng.normal(size=3)
        base = rng.normal(size=E) * 0.3
        for _ in range(per_topo):
            th = np.concatenate([rng.normal(size=P - 3), topo])
            thetas.append(th)
            zs.append(base + rng.normal(scale=0.5, size=E))
    theta = np.asarray(thetas)
    z = np.asarray(zs)
    # replay the SAME rows dup_factor times, exactly as the seed collision does
    theta = np.vstack([theta] * dup_factor)
    z = np.vstack([z] * dup_factor)
    zn = z / np.linalg.norm(z, axis=1, keepdims=True)

    n = z.shape[0]
    half = n // 2
    _write_shard(os.path.join(root, "sbi_sim__task0000.parquet"),
                 zn[:half], z[:half], theta[:half],
                 {"campaign_id": ["c"] * half,
                  "topo_idx": np.arange(half) // per_topo,
                  "iter_idx": np.arange(half),
                  "seed_run": np.arange(half),
                  "window_idx": np.zeros(half, dtype=np.int64)})
    _write_shard(os.path.join(root, "sbi_sim__task0001.parquet"),
                 zn[half:], z[half:], theta[half:],
                 {"campaign_id": ["c"] * (n - half),
                  "topo_idx": np.arange(n - half) // per_topo,
                  "iter_idx": np.arange(n - half),
                  "seed_run": np.arange(n - half),
                  "window_idx": np.zeros(n - half, dtype=np.int64)})

    # --- real: n_groups cultures x n_sub subregions x n_win windows
    rz, groups, classes, subs, wins = [], [], [], [], []
    for g in range(n_groups):
        gbase = rng.normal(size=E) * 0.3 + shift
        for s in range(n_sub):
            sb = gbase + rng.normal(scale=0.1, size=E)
            for w in range(n_win):
                rz.append(sb + rng.normal(scale=0.4, size=E))
                groups.append("cult_%02d" % g)
                classes.append("0" if g < n_groups // 2 else "1")
                subs.append(s)
                wins.append(w)
    rz = np.asarray(rz)
    rzn = rz / np.linalg.norm(rz, axis=1, keepdims=True)
    _write_shard(os.path.join(root, "sbi_real_cohort.parquet"),
                 rzn, rz, None,
                 {"culture": groups, "condition": classes,
                  "subregion": subs, "name": ["%s_%d_%d" % (g, s, w)
                                              for g, s, w in
                                              zip(groups, subs, wins)],
                  "T_rec_s": np.full(len(rz), 1200.0),
                  "fs_ifr": np.full(len(rz), 100.0),
                  "window_idx": wins})
    return {"n_sim_rows": n, "n_sim_distinct": n // dup_factor,
            "n_topo": n_topo, "n_real": len(rz), "n_groups": n_groups}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misspec_dir", required=True)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.abspath(args.misspec_dir))

    tmp = tempfile.mkdtemp(prefix="gate_smoke_")
    print("fixture:", tmp)
    try:
        # ---------- A. loading and deduplication ----------
        print("\n[A] gate_data: loading and dedup")
        wd = os.path.join(tmp, "well")
        info = build_fixture(wd, shift=0.0, seed=1)
        import gate_data as D

        sim_raw = D.load_sim(os.path.join(wd, "sbi_sim__*.parquet"),
                             dedup_theta=False)
        sim = D.load_sim(os.path.join(wd, "sbi_sim__*.parquet"),
                         dedup_theta=True)
        check("A1_rows_raw", sim_raw.n == info["n_sim_rows"],
              "%d" % sim_raw.n)
        check("A2_dedup_removes_replay", sim.n == info["n_sim_distinct"],
              "%d -> %d" % (sim_raw.n, sim.n))
        check("A3_dup_factor_reported",
              abs(sim.meta["duplication_factor"] - 3.0) < 1e-9,
              "%.2f" % sim.meta["duplication_factor"])
        check("A4_theta_shape", sim.theta.shape[1] == P,
              str(sim.theta.shape))
        check("A5_zraw_present_and_aligned",
              sim.zraw is not None and sim.zraw.shape == sim.z.shape,
              str(None if sim.zraw is None else sim.zraw.shape))

        real = D.load_real(os.path.join(wd, "sbi_real_cohort.parquet"))
        check("A6_real_rows", real.n == info["n_real"], "%d" % real.n)
        check("A7_real_groups",
              len(set(real.groups.tolist())) == info["n_groups"],
              "%d" % len(set(real.groups.tolist())))
        check("A8_real_has_no_theta", real.theta is None)

        topo = D.topology_diversity(sim.theta, PARAM_NAMES)
        check("A9_topology_count", topo["n_topologies"] == info["n_topo"],
              "%d" % topo["n_topologies"])

        # column ORDER: z_10 must not sort before z_2
        cols = D._ordered_cols(["z_0", "z_10", "z_2", "z_1"], D._Z_RE)
        check("A10_numeric_column_order", cols == ["z_0", "z_1", "z_2", "z_10"],
              str(cols))

        r_eff, ev = D.effective_rank(sim.z)
        check("A11_effective_rank_sane", 1.0 <= r_eff <= E and ev.size == E,
              "r_eff=%.2f" % r_eff)

        # ---------- B. the gate fires correctly both ways ----------
        print("\n[B] gate_run: well-specified vs misspecified")
        md = os.path.join(tmp, "miss")
        build_fixture(md, shift=1.2, seed=1)

        outs = {}
        for tag, d in (("well", wd), ("miss", md)):
            stem = os.path.join(tmp, "res_%s" % tag)
            cmd = [sys.executable, os.path.join(here, "gate_run.py"),
                   "--real", os.path.join(d, "sbi_real_cohort.parquet"),
                   "--sim", os.path.join(d, "sbi_sim__*.parquet"),
                   "--misspec_dir", os.path.abspath(args.misspec_dir),
                   "--out", stem, "--quick", "--spaces", "z"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-3000:]); print(r.stderr[-3000:])
            check("B1_%s_exit0" % tag, r.returncode == 0,
                  "rc=%d" % r.returncode)
            if r.returncode != 0:
                continue
            with open(stem + "_results.json") as fh:
                outs[tag] = json.load(fh)

        if "well" in outs and "miss" in outs:
            pw = outs["well"]["gate"]["z::pooled"]["p_group"]
            pm = outs["miss"]["gate"]["z::pooled"]["p_group"]
            check("B2_well_specified_passes", pw >= 0.05, "p_group=%.4g" % pw)
            check("B3_misspecified_rejects", pm < 0.05, "p_group=%.4g" % pm)
            check("B4_separation", pw > pm, "%.4g vs %.4g" % (pw, pm))
            check("B5_dedup_recorded",
                  outs["well"]["sim"]["meta"].get("duplication_factor",
                                                  0) > 1.0)
            check("B6_topology_reported",
                  outs["well"]["sim"]["topology_diversity"]["n_topologies"]
                  == info["n_topo"])
            check("B7_caveats_present",
                  len(outs["well"].get("caveats", [])) >= 3)

        # ---------- C. figures ----------
        print("\n[C] gate_plots")
        figdir = os.path.join(tmp, "figs")
        r = subprocess.run(
            [sys.executable, os.path.join(here, "gate_plots.py"),
             "--stem", os.path.join(tmp, "res_miss"), "--outdir", figdir],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
        check("C1_plots_exit0", r.returncode == 0, "rc=%d" % r.returncode)
        made = sorted(os.listdir(figdir)) if os.path.isdir(figdir) else []
        check("C2_figures_written", len(made) >= 4, "%d files" % len(made))
        check("C3_nonempty",
              all(os.path.getsize(os.path.join(figdir, f)) > 2000
                  for f in made) if made else False,
              ", ".join(made[:6]))

    finally:
        if args.keep:
            print("\nfixture kept at", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 66)
    print("%d passed, %d failed, %d total"
          % (n_ok, len(RESULTS) - n_ok, len(RESULTS)))
    print("=" * 66)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
