#!/usr/bin/env python3
"""
smoke_witness_run.py -- correctness checks for witness_run.py, the driver
that turns gate_run.py's arrays into the witness analysis. Run from hpc/:

    python3 smoke_witness_run.py

Exit 0 and "ALL CHECKS PASSED" mean the driver is sound. Set KEEP_PLOTS=1
to keep the generated figures and the summary JSON. No cluster resources
and no real data needed: the fixture is a synthetic gate_arrays.npz built
in the same layout gate_run.py writes.

WHAT IS CHECKED
  1. The verdict rule is a pure function of the five diagnostics, and
     every branch of it is reachable: faithful+stable, faithful+unstable,
     unfaithful+stable, unfaithful+unstable, and the case where no slices
     were run (stability unknown).
  2. The driver runs end to end on a fixture with the exact key layout
     gate_run.py writes (sim_z, real_z, sim_zraw, real_zraw, real_groups,
     real_classes, plus the spectra it also stores).
  3. Every promised figure is written, non-trivially sized, for both
     spaces and every requested method.
  4. witness_summary.json is valid JSON, records both spaces, and carries
     the diagnostics actually used by the verdict (rho_sum, resid_median,
     slice min_corr, sign disagreement) rather than a summary of them.
  5. The reported witness gap equals npe_misspec's own witness_function
     output, so the JSON cannot drift from the library.
  6. --no_split makes mean(u) - mean(v) EXACTLY mmd2_multiscale, which is
     the gate's statistic; the JSON flags which regime it ran in.
  7. Negative paths: a missing space is skipped with a message rather
     than crashing; a group vector of the wrong length is ignored rather
     than mislabelling recordings; --quick drops t-SNE.
  8. Determinism: two runs with the same seed produce identical
     diagnostics.

ASCII-only by policy (HPC transfer safety).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from npe_misspec import mmd2_multiscale, witness_function   # noqa: E402
from witness_run import _verdict                            # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not cond:
        FAILED.append(name)


def sphere_blob(rng, n, center, spread):
    x = center[None, :] + spread * rng.normal(size=(n, center.size))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make_fixture(path, rng, E=16, n_real=30):
    """A gate_arrays.npz in exactly the layout gate_run.py writes."""
    c1, c2 = np.eye(E)[0], np.eye(E)[1]
    sim_z = np.concatenate([sphere_blob(rng, 250, c1, 0.12),
                            sphere_blob(rng, 250, c2, 0.12)])
    real_z = sphere_blob(rng, n_real, c1, 0.12)
    scale_s = 1.0 + 0.3 * rng.random((sim_z.shape[0], 1))
    scale_r = 1.0 + 0.3 * rng.random((real_z.shape[0], 1))
    # 5 recordings x 6 windows: real rows are clustered, as in the cohort
    groups = np.asarray(["cult%02d" % (i // 6) for i in range(n_real)])
    classes = np.asarray(["ctrl" if (i // 6) % 2 == 0 else "treat"
                          for i in range(n_real)])
    np.savez_compressed(
        path, sim_z=sim_z, real_z=real_z,
        sim_zraw=sim_z * scale_s, real_zraw=real_z * scale_r,
        real_groups=groups, real_classes=classes,
        sim_spectrum=np.linspace(1.0, 0.01, E),
        real_spectrum=np.linspace(1.0, 0.01, E))
    return sim_z, real_z, groups


def run_driver(arrays, out, extra=()):
    cmd = [sys.executable, os.path.join(HERE, "witness_run.py"),
           "--arrays", arrays, "--out", out, "--misspec_dir", HERE] \
        + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p


def main():
    rng = np.random.default_rng(23)
    tmp = tempfile.mkdtemp(prefix="witness_run_smoke_")
    try:
        print("check 1: the verdict rule covers every branch")
        good = [0.3, 0.6, 0.9]          # all scales resolve the data
        mixed = [1.7, 0.8, 0.4]         # the narrowest does not
        none_ = [1.7, 2.4, 3.1]         # no scale resolves the data
        v_ok = _verdict(0.97, good, 0.30, 0.95, 0.01)
        v_unstable = _verdict(0.97, good, 0.30, 0.40, 0.35)
        v_unfaith = _verdict(0.20, good, 0.30, 0.95, 0.01)
        v_both = _verdict(0.20, good, 0.30, 0.40, 0.35)
        v_noslice = _verdict(0.97, good, 0.30, float("nan"), float("nan"))
        check("faithful + stable -> READ THE PLANE",
              v_ok.startswith("READ THE PLANE"), v_ok[:48])
        check("faithful + unstable -> caution about translation",
              v_unstable.startswith("READ WITH CAUTION")
              and "translation" in v_unstable, v_unstable[:48])
        check("unfaithful + stable -> caution about the cut",
              v_unfaith.startswith("READ WITH CAUTION")
              and "track g" in v_unfaith, v_unfaith[:48])
        check("unfaithful + unstable -> DO NOT READ THE PLANE",
              v_both.startswith("DO NOT READ"), v_both[:48])
        check("no slices run -> still readable, no stability claim",
              v_noslice.startswith("READ THE PLANE")
              and "translation" not in v_noslice, v_noslice[:48])
        # The residual reports SCOPE; it must not veto a high rho on the
        # strength of the narrowest kernel alone. This is the regression
        # test for a real bug: the first version of the rule did exactly
        # that and turned rho=0.995 into READ WITH CAUTION.
        v_mixed = _verdict(0.97, mixed, 0.30, 0.95, 0.01)
        check("one unresolved scale does NOT veto a high rho",
              v_mixed.startswith("READ THE PLANE"), v_mixed[:48])
        check("the verdict states how many scales resolve the data",
              "2/3 bandwidths resolve" in v_mixed,
              v_mixed[v_mixed.find("["):][:52])
        check("all scales resolved -> 3/3 reported",
              "3/3 bandwidths resolve" in v_ok)
        check("NO scale resolving the data downgrades even a high rho",
              _verdict(0.97, none_, 0.30, 0.95, 0.01).startswith(
                  "READ WITH CAUTION"),
              _verdict(0.97, none_, 0.30, 0.95, 0.01)[:48])
        check("absent residual info -> no scope clause, rho alone decides",
              _verdict(0.97, None, 0.30, 0.95, 0.01).startswith(
                  "READ THE PLANE")
              and "bandwidths resolve" not in _verdict(0.97, None, 0.30,
                                                       0.95, 0.01))

        print("check 2: end-to-end run on a gate_run.py-shaped fixture")
        arrays = os.path.join(tmp, "gate_arrays.npz")
        sim_z, real_z, groups = make_fixture(arrays, rng)
        out = os.path.join(tmp, "witness")
        p = run_driver(arrays, out, ["--quick"])
        check("driver exits 0", p.returncode == 0,
              (p.stderr or "")[-300:] if p.returncode else "")
        if p.returncode != 0:
            print(p.stdout[-2000:])
            raise SystemExit(1)
        check("both spaces processed",
              "z: sim" in p.stdout and "zraw: sim" in p.stdout)
        check("a verdict was printed per space and linear method",
              p.stdout.count("VERDICT") >= 4,
              "%d verdict lines" % p.stdout.count("VERDICT"))

        print("check 3: every promised figure exists")
        want = ["witness_heat_z_pca_lift.png",
                "witness_heat_z_contrast_lift.png",
                "witness_heat_zraw_pca_lift.png",
                "witness_slices_z_pca.png",
                "witness_slices_z_contrast.png",
                "witness_map_z_pca.png",
                "witness_hist_z.png"]
        for w in want:
            fp = os.path.join(out, w)
            check("wrote %s" % w,
                  os.path.exists(fp) and os.path.getsize(fp) > 5000,
                  "%d bytes" % (os.path.getsize(fp)
                                if os.path.exists(fp) else 0))

        print("check 4: the summary JSON is complete and machine-readable")
        with open(os.path.join(out, "witness_summary.json")) as fh:
            doc = json.load(fh)
        check("both spaces in the summary",
              set(doc["spaces"].keys()) == {"z", "zraw"},
              str(sorted(doc["spaces"].keys())))
        zrec = doc["spaces"]["z"]
        check("heatmap diagnostics recorded per method",
              set(zrec["heatmaps"].keys()) >= {"pca", "contrast"},
              str(sorted(zrec["heatmaps"].keys())))
        for m in ("pca", "contrast"):
            h = zrec["heatmaps"][m]
            check("%s: rho_sum recorded" % m,
                  "rho_sum" in h and np.isfinite(h["rho_sum"]),
                  "rho_sum=%.3f" % h.get("rho_sum", float("nan")))
            check("%s: resid_median and per-sigma ratio recorded" % m,
                  "resid_median" in h and "resid_over_sigma" in h)
            s = zrec["slices"][m]
            check("%s: slice stability recorded" % m,
                  "min_corr" in s and "max_sign_disagreement" in s,
                  "min_corr=%.3f" % s.get("min_corr", float("nan")))
        check("verdicts present for both linear methods",
              set(zrec["verdicts"].keys()) == {"pca", "contrast"})
        check("caveats carried into the JSON, not just the prose",
              any("EXPLORATORY" in c for c in doc["caveats"]))
        check("the identity regime is flagged",
              zrec["witness_gap_is_exact_mmd2"] is False)

        print("check 5: reported gap matches the library, not a copy of it")
        res = witness_function(sim_z, real_z, space="z", split=True, seed=0)
        check("witness_gap equals witness_function's own value",
              abs(zrec["witness_gap"] - res.mmd2_from_witness()) < 1e-12,
              "|diff|=%.2e" % abs(zrec["witness_gap"]
                                  - res.mmd2_from_witness()))
        check("the 5 most negative real rows are named by group label",
              len(zrec["most_negative_real"]) == 5
              and str(zrec["most_negative_real"][0][0]).startswith("cult"),
              str(zrec["most_negative_real"][0]))

        print("check 6: --no_split gives the exact gate statistic")
        out2 = os.path.join(tmp, "witness_nosplit")
        p2 = run_driver(arrays, out2,
                        ["--quick", "--no_split", "--no_slices",
                         "--spaces", "z", "--methods", "pca"])
        check("driver exits 0 with --no_split", p2.returncode == 0,
              (p2.stderr or "")[-300:] if p2.returncode else "")
        with open(os.path.join(out2, "witness_summary.json")) as fh:
            doc2 = json.load(fh)
        z2 = doc2["spaces"]["z"]
        exact = mmd2_multiscale(sim_z, real_z, z2["bandwidths"])
        check("mean(u)-mean(v) is exactly mmd2_multiscale",
              abs(z2["witness_gap"] - exact) < 1e-10,
              "|diff|=%.2e" % abs(z2["witness_gap"] - exact))
        check("the JSON flags the exact regime",
              z2["witness_gap_is_exact_mmd2"] is True)

        print("check 7: negative paths")
        bare = os.path.join(tmp, "bare.npz")
        np.savez_compressed(bare, sim_z=sim_z, real_z=real_z)
        out3 = os.path.join(tmp, "witness_bare")
        p3 = run_driver(bare, out3,
                        ["--quick", "--no_slices", "--methods", "pca"])
        check("missing zraw is skipped, not fatal",
              p3.returncode == 0 and "SKIP zraw" in p3.stdout,
              (p3.stdout or "")[-200:] if p3.returncode else "")
        bad = os.path.join(tmp, "badgroups.npz")
        np.savez_compressed(bad, sim_z=sim_z, real_z=real_z,
                            real_groups=np.asarray(["a", "b", "c"]))
        out4 = os.path.join(tmp, "witness_badgroups")
        p4 = run_driver(bad, out4,
                        ["--quick", "--no_slices", "--spaces", "z",
                         "--methods", "pca"])
        check("mismatched group vector is ignored with a NOTE, not used",
              p4.returncode == 0 and "labels not used" in p4.stdout,
              (p4.stdout or "")[-200:] if p4.returncode else "")
        check("--quick drops t-SNE from the method list",
              not os.path.exists(os.path.join(out, "witness_heat_z_tsne_nw.png")))

        print("check 8: determinism")
        out5 = os.path.join(tmp, "witness_rep")
        p5 = run_driver(arrays, out5,
                        ["--quick", "--spaces", "z", "--methods", "pca"])
        check("repeat run exits 0", p5.returncode == 0)
        with open(os.path.join(out5, "witness_summary.json")) as fh:
            doc5 = json.load(fh)
        a = doc["spaces"]["z"]["heatmaps"]["pca"]
        b = doc5["spaces"]["z"]["heatmaps"]["pca"]
        check("rho_sum reproduces exactly across runs",
              a["rho_sum"] == b["rho_sum"],
              "%.15f vs %.15f" % (a["rho_sum"], b["rho_sum"]))
        check("resid_median reproduces exactly across runs",
              a["resid_median"] == b["resid_median"])

        if os.environ.get("KEEP_PLOTS") == "1":
            print("  outputs kept in %s" % tmp)
    finally:
        if os.environ.get("KEEP_PLOTS") != "1":
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print("FAILED: %d check(s): %s" % (len(FAILED), FAILED))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
