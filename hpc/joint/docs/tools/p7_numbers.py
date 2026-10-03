#!/usr/bin/env python3
"""p7_numbers.py -- every [RAN] number of P7_UPSTREAM_AND_JOBS.md, recomputed.

Torch-free. Like p6 this script IMPORTS the code it describes: the Stage 1
modules (`latent_bank`, `latent_sbi_simulator`, `latent_nuisance`,
`latent_gap`, `latent_realisation`, `bench_burst_provider`,
`bench_burst_generator`, `build_latent_bank`) and, through them, the DSN
generator modules under `hpc/dsn` (`latent_burst_generator`,
`generate_burst_data`: numpy, scipy, matplotlib). Two functions of modules
that import torch at the top are extracted from their source with `ast` and
executed alone: `joint_batches.enumerate_donor_pairs` (Stage 2) and
`run_joint_arms.grouped_split` (Stage 3). Small banks are built through the
real command line (`build_latent_bank.py`, by subprocess) in a temporary
directory that is deleted at the end. Nothing is trained and no torch
module is imported.

Run from hpc/joint/docs/tools:

    python p7_numbers.py

Every block prints only quantities that do not depend on the machine (no
paths, no timings), so two runs give identical output.

Pure ASCII, LF only.
"""

from __future__ import annotations

import ast
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))   # repo root
_JOINT = os.path.join(_ROOT, "hpc", "joint")
_STAGE1 = os.path.join(_JOINT, "stage1")
for _p in (_STAGE1, _JOINT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_latent_bank as BLB                      # noqa: E402
import latent_bank as LB                             # noqa: E402
import latent_gap as LG                              # noqa: E402
import latent_nuisance as LN                         # noqa: E402
import latent_realisation as LR                      # noqa: E402
import latent_sbi_simulator as LS                    # noqa: E402
import bench_burst_provider as BP                    # noqa: E402
import bench_burst_generator as BG                   # noqa: E402
from smoke_test_latent_sbi import ReferenceBurstProvider   # noqa: E402

JOBS = {
    "build_latent_bank.pbs": "stage1/jobs/build_latent_bank.pbs",
    "joint_arms.pbs": "stage3/jobs/joint_arms.pbs",
    "probe_conda_activation.pbs": "stage3/jobs/probe_conda_activation.pbs",
    "probe_dsn_runtime.pbs": "stage3/jobs/probe_dsn_runtime.pbs",
    "stage3b.pbs": "stage3b/jobs/stage3b.pbs",
    "stage3c.pbs": "stage3c/jobs/stage3c.pbs",
    "joint_tune.pbs": "stage4/jobs/joint_tune.pbs",
}
# the script each job calls, as inventory.json names it
JOB_SCRIPT = {
    "build_latent_bank.pbs": "joint/stage1/build_latent_bank.py",
    "joint_arms.pbs": "joint/stage3/run_joint_arms.py",
    "probe_dsn_runtime.pbs": "joint/stage3/probe_dsn_runtime.py",
    "stage3b.pbs": "joint/stage3b/run_stage3b.py",
    "stage3c.pbs": "joint/stage3c/run_stage3c.py",
}
# documented array ranges (job headers; usage v1.3 S3.2, S5.2)
ARRAYS = {"build_latent_bank.pbs": 32, "joint_arms.pbs": 45, "joint_tune.pbs": 8}


def hr(title: str) -> None:
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


def _extract(path: str, name: str, extra_ns=None):
    """Return function `name` from `path`, compiled alone (no module import)."""
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"np": np}
            ns.update(extra_ns or {})
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
            return ns[name]
    raise RuntimeError("%s not found in %s" % (name, path))


import hashlib  # noqa: E402  (grouped_split's namespace)

ENUM_PAIRS = _extract(os.path.join(_JOINT, "stage2", "joint_batches.py"),
                      "enumerate_donor_pairs")
GROUPED_SPLIT = _extract(os.path.join(_JOINT, "stage3", "run_joint_arms.py"),
                         "grouped_split", {"hashlib": hashlib, "json": json})


def _defaults():
    return BLB.build_parser().parse_args(["--out-dir", "unused"])


def _build(out_dir: str, *flags: str):
    """Run build_latent_bank.py; return (returncode, last stderr/stdout line)."""
    cmd = [sys.executable, "build_latent_bank.py", "--out-dir", out_dir] + list(flags)
    res = subprocess.run(cmd, cwd=_STAGE1, capture_output=True, text=True)
    text = (res.stdout + res.stderr).strip().splitlines()
    return res.returncode, (text[-1] if text else "")


def _synthetic_donors(n_shards: int, n_don: int, n_wd: int, J: int):
    """Row-level donor ids and (shard, donor) labels exactly as
    build_latent_bank.py lays a shard out (:216, :238-254): trace-major rows,
    J consecutive rows per trace, n_wd consecutive traces per donor, and the
    donor index restarting at 0 in every shard."""
    per_shard = np.repeat(np.repeat(np.arange(n_don), n_wd), J)
    donors = np.concatenate([per_shard] * n_shards)
    shard = np.repeat(np.arange(n_shards), per_shard.size)
    well = np.concatenate([np.repeat(np.arange(n_don * n_wd), J)] * n_shards)
    return donors, shard, well


# ---------------------------------------------------------------------------
# B1  shard geometry and pair counts, closed form, at the job defaults
# ---------------------------------------------------------------------------

def b1_geometry():
    hr("B1  shard geometry at the builder's and the job's defaults")
    a = _defaults()
    N_tr, n_wd, n_db, J = a.n_traces, a.wells_per_donor, a.donors_per_batch, a.n_windows
    W = int(round(a.T_win * a.fs))
    N_don = N_tr // n_wd
    n_bat = (N_don - 1) // n_db + 1
    rows = N_tr * J
    print("defaults: n_traces %d, wells_per_donor %d, donors_per_batch %d, n_windows %d, "
          "T_win %.1f s, fs %.1f Hz, n_neurons %d, n_latent %d, n_label_axes %d, "
          "n_classes %d, tau_ov %.2f, pi %.1f, gap_modes %r, n_per_theta %d, provider %r, "
          "seed %d" % (N_tr, n_wd, n_db, J, a.T_win, a.fs, a.n_neurons, a.n_latent,
                        a.n_label_axes, a.n_classes, a.tau_ov, a.pi, a.gap_modes,
                        a.n_per_theta, a.provider, a.seed))
    print("W = round(T_win * fs) = %d samples; one trace simulates J * W = %d samples = %.1f s"
          % (W, J * W, J * W / a.fs))
    print("per shard: %d traces = %d donors x %d wells; %d batches of %d donors; rows %d; "
          "rows per donor %d" % (N_tr, N_don, n_wd, n_bat, n_db, rows, n_wd * J))
    xb = rows * W * 4
    print("x as float32: %d bytes per shard (%.3f MB); 32 shards %.1f MB; the plan's 32000 "
          "windows %.1f MB" % (xb, xb / 1e6, 32 * xb / 1e6, 32000 * W * 4 / 1e6))
    print("bench provider duration per trace: (J*W + PAD_BINS) / fs = %.2f s (PAD_BINS %d)"
          % ((J * W + BP.BenchBurstProvider.PAD_BINS) / a.fs, BP.BenchBurstProvider.PAD_BINS))
    print("the plan's G = 4000 traces at %d per shard is %.2f shards; shard counts that divide "
          "4000 with an even n_traces: %s"
          % (N_tr, 4000 / N_tr,
             ", ".join("%d x %d" % (4000 // t, t) for t in (100, 80, 50, 40, 32, 20)
                       if 4000 % t == 0 and t % n_wd == 0)))
    print("")
    print("pairs enumerated by donor id after concat_shards of S identical-layout shards")
    print("(same-theta: the two rows come from one shard's donor; cross-well: and two wells)")
    for S in (1, 2, 4, 32):
        m = S * n_wd * J
        U = N_don * m * (m - 1) // 2
        same = S * N_don * (n_wd * J) * (n_wd * J - 1) // 2
        cross = S * N_don * (n_wd * (n_wd - 1) // 2) * J * J
        within = S * N_don * n_wd * J * (J - 1) // 2
        r_same = (n_wd * J - 1) / (S * n_wd * J - 1)
        r_well = (n_wd - 1) * J / (S * n_wd * J - 1)
        print("  S_sh = %2d: |U| = %9d; same-theta %7d (r_same %.4f = %.2f%%); within-well "
              "%6d; cross-well same-theta %6d (r_well %.4f = %.2f%%); different-theta %9d"
              % (S, U, same, r_same, 100 * r_same, within, cross, r_well, 100 * r_well,
                 U - same))
        assert same == cross + within


# ---------------------------------------------------------------------------
# B2  the job scripts: resources, forwarding, arrays, bash
# ---------------------------------------------------------------------------

def _pbs_text(rel):
    with open(os.path.join(_JOINT, rel), "r", encoding="utf-8") as fh:
        return fh.read()


def _resources(text):
    sel = re.search(r"^#PBS -l select=1:ncpus=(\d+):mem=(\d+)gb", text, re.M)
    wall = re.search(r"^#PBS -l walltime=(\d+):(\d+):(\d+)", text, re.M)
    h = int(wall.group(1)) + int(wall.group(2)) / 60.0 + int(wall.group(3)) / 3600.0
    return int(sel.group(1)), int(sel.group(2)), h, wall.group(0).split("=")[1]


def _flags_forwarded(text):
    out = set()
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("echo"):
            continue
        out.update(re.findall(r"(--[A-Za-z][-A-Za-z0-9]*)", s))
    return out


def b2_jobs():
    hr("B2  job scripts: resources, flag forwarding, the array index map, bash")
    with open(os.path.join(_HERE, "inventory.json"), "r", encoding="utf-8") as fh:
        inv = json.load(fh)["rows"]
    print("%-28s %5s %6s %9s" % ("job", "ncpus", "mem GB", "walltime"))
    res = {}
    for name, rel in JOBS.items():
        t = _pbs_text(rel)
        nc, mem, h, wall = _resources(t)
        res[name] = (nc, mem, h)
        print("%-28s %5d %6d %9s" % (name, nc, mem, wall))
    print("")
    for name, n in ARRAYS.items():
        nc, mem, h = res[name]
        print("documented array of %-22s: %3d elements x %d CPUs x %.0f h = %6.0f "
              "core-hours requested at most" % (name, n, nc, h, n * nc * h))
    print("")
    for name, script in JOB_SCRIPT.items():
        t = _pbs_text(JOBS[name])
        fwd = _flags_forwarded(t)
        cli = sorted({r["name"] for r in inv if r["file"] == script and r["surface"] == "cli"})
        if not cli:
            print("%s -> %s: no CLI rows in inventory.json (outside its scope)"
                  % (name, os.path.basename(script)))
            continue
        reach = [f for f in cli if f in fwd]
        miss = [f for f in cli if f not in fwd]
        print("%s -> %s: %d CLI flags; the job passes %d, %d have no job variable"
              % (name, os.path.basename(script), len(cli), len(reach), len(miss)))
        print("   no job variable: %s" % " ".join(miss))
    # the probe job's two variables
    t = _pbs_text(JOBS["probe_dsn_runtime.pbs"])
    print("probe_dsn_runtime.pbs passes: %s" % " ".join(sorted(_flags_forwarded(t))))
    print("")
    t = _pbs_text(JOBS["joint_arms.pbs"])
    arms = re.search(r"^ARMS=\(([^)]*)\)", t, re.M).group(1).split()
    n = len(arms)
    print("ARMS = %s (%d arms); index -> (ARMS[i %% %d], i // %d)" % (" ".join(arms), n, n, n))
    for arm in arms:
        idx = [i for i in range(45) if arms[i % n] == arm]
        print("  %-8s indices %s" % (arm, idx))
    print("")
    probe = ("set -u; EXTRA=(); f() { echo \"argc=$#\"; }; f \"${EXTRA[@]}\"; "
             "echo \"bash ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}: an empty array "
             "expands under set -u without error\"")
    out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)
    print("sandbox bash, empty \"${EXTRA[@]}\" under set -u: rc %d; %s"
          % (out.returncode, " | ".join(out.stdout.strip().splitlines())))


# ---------------------------------------------------------------------------
# B3  real banks through the real CLI: ids, pairs, the split, arm R vs arm S
# ---------------------------------------------------------------------------

def b3_banks(tmp):
    hr("B3  two-shard banks built by build_latent_bank.py (reference provider, defaults)")
    for arm, extra in (("S", []), ("R", ["--pi", "0.0"])):
        for k in (0, 1):
            rc, last = _build(os.path.join(tmp, arm), "--arm", arm, "--shard-index", str(k),
                              *extra)
            assert rc == 0, last
    S, sideS = LB.concat_shards(sorted(glob.glob(os.path.join(tmp, "S", "*.npz"))))
    R, sideR = LB.concat_shards(sorted(glob.glob(os.path.join(tmp, "R", "*.npz"))))
    a = _defaults()
    n_wd, J = a.wells_per_donor, a.n_windows
    print("arm S, 2 shards: rows %d; distinct donor ids %d, well ids %d, batch ids %d, "
          "realisation ids %d, distinct theta rows %d"
          % (S["x"].shape[0], len(np.unique(S["donor"])), len(np.unique(S["well"])),
             len(np.unique(S["batch"])), len(np.unique(S["realisation_id"])),
             len({tuple(r) for r in S["theta"]})))
    per_d = [len({tuple(r) for r in S["theta"][S["donor"] == d]}) for d in np.unique(S["donor"])]
    per_w = [len({tuple(r) for r in S["theta"][S["well"] == w]}) for w in np.unique(S["well"])]
    print("distinct theta per donor id: min %d max %d; per well id: min %d max %d"
          % (min(per_d), max(per_d), min(per_w), max(per_w)))
    # pairs as the replicate stream enumerates them (on the REAL arm)
    pairs, n_single = ENUM_PAIRS(R["donor"])
    th = R["theta"]
    same = np.array([np.array_equal(th[i], th[j]) for i, j in pairs])
    wl = R["well"]
    sh = np.repeat(np.arange(2), R["x"].shape[0] // 2)
    cross = same & (wl[pairs[:, 0]] != wl[pairs[:, 1]])
    print("enumerate_donor_pairs(arm R donor ids): |U| %d, singleton donors %d; same-theta "
          "%d (%.4f); cross-well same-theta %d (%.4f); different-theta %d"
          % (pairs.shape[0], n_single, int(same.sum()), same.mean(), int(cross.sum()),
             cross.mean(), int((~same).sum())))
    assert np.all(same == (sh[pairs[:, 0]] == sh[pairs[:, 1]]))
    first = pairs[:64]
    fs_ = same[:64]
    print("the Stage 3 diagnostic's first 64 pairs (run_joint_arms.py:612-617): same-theta "
          "%d, of them within-well %d; different-theta %d"
          % (int(fs_.sum()), int((fs_ & (wl[first[:, 0]] == wl[first[:, 1]])).sum()),
             int((~fs_).sum())))
    # the split
    one = sorted(glob.glob(os.path.join(tmp, "S", "*.npz")))[:1]
    S1, _ = LB.concat_shards(one)
    for lab, arr in (("1 shard ", S1["donor"]), ("2 shards", S["donor"])):
        idx, dig = GROUPED_SPLIT(arr, seed=0)
        groups = [len(np.unique(arr[idx == p])) for p in (0, 1, 2)]
        print("grouped_split(donor ids), %s: groups train/sel/report %s; rows %s; hash %s"
              % (lab, groups, [int((idx == p).sum()) for p in (0, 1, 2)], dig[:16]))
    # arm R against arm S, field by field
    print("")
    for k in ("theta", "cls", "nu", "realisation_id", "donor", "well", "x"):
        print("arm R (--pi 0, same --seed and --shard-index) == arm S on %-15s: %s"
              % (k, bool(np.array_equal(S[k], R[k]))))
    print("contract digests differ (arm, theta_withheld are contract fields): %s; "
          "base_seed S %d, R %d (shard 0)"
          % (sideS["contract_digest"] != sideR["contract_digest"],
             sideS["provenance"]["base_seed"], sideR["provenance"]["base_seed"]))
    idx, _ = GROUPED_SPLIT(S["donor"], seed=0)
    shares = [float((idx == p).mean()) for p in (0, 1, 2)]
    print("so the pseudo-real endpoint, scored on every arm-R row (run_joint_arms.py:552-556), "
          "scores copies of training / selection / report rows in the shares %.4f / %.4f / "
          "%.4f (the split of arm S's donor ids, seed 0)" % tuple(shares))
    # Stage 3c's culture means group by well id
    mism = 0
    for w in np.unique(S["well"]):
        cl = np.unique(S["cls"][S["well"] == w])
        mism += int(cl.size > 1)
    print("run_stage3c._culture_means groups by well id: %d groups for %d wells; groups "
          "mixing two classes %d of %d" % (len(np.unique(S["well"])), 2 * a.n_traces, mism,
                                           len(np.unique(S["well"]))))
    d0 = np.flatnonzero(S["donor"] == S["donor"][0])[:16]
    print("P10 (run_stage3c.py:304-305) takes rows %d..%d: all in shard 0: %s; one theta: %s"
          % (d0.min(), d0.max(), bool(d0.max() < a.n_traces * J),
             len({tuple(r) for r in S["theta"][d0]}) == 1))
    # the digest does not see the nesting knobs or the size
    rc1, _ = _build(os.path.join(tmp, "T1"), "--max-records", "2")
    rc2, _ = _build(os.path.join(tmp, "T2"), "--wells-per-donor", "4")
    d1 = json.load(open(os.path.join(tmp, "T1", "shard_0000.json")))["contract_digest"]
    d2 = json.load(open(os.path.join(tmp, "T2", "shard_0000.json")))["contract_digest"]
    print("digest at --max-records 2 == default: %s; at --wells-per-donor 4 == default: %s"
          % (d1 == sideS["contract_digest"], d2 == sideS["contract_digest"]))
    print("CONTRACT_FIELDS (%d): %s" % (len(LB.CONTRACT_FIELDS), ", ".join(LB.CONTRACT_FIELDS)))
    return S


def b3_synthetic():
    hr("B3b  32 shards, synthetic ids laid out as the builder lays them")
    a = _defaults()
    N_don, n_wd, J = a.n_traces // a.wells_per_donor, a.wells_per_donor, a.n_windows
    donors, shard, well = _synthetic_donors(32, N_don, n_wd, J)
    pairs, _ = ENUM_PAIRS(donors)
    same = shard[pairs[:, 0]] == shard[pairs[:, 1]]
    cross = same & (well[pairs[:, 0]] != well[pairs[:, 1]])
    print("|U| %d; same-theta %d (%.4f); cross-well same-theta %d (%.4f)"
          % (pairs.shape[0], int(same.sum()), same.mean(), int(cross.sum()), cross.mean()))
    f = pairs[:64]
    fs_ = same[:64]
    wi = fs_ & (well[f[:, 0]] == well[f[:, 1]])
    print("first 64 pairs: same-theta %d (within-well %d, cross-well %d); different-theta %d"
          % (int(fs_.sum()), int(wi.sum()), int(fs_.sum() - wi.sum()), int((~fs_).sum())))
    idx, dig = GROUPED_SPLIT(donors, seed=0)
    print("grouped_split on 32 shards: groups %s, hash %s (the 1-shard hash of B3)"
          % ([len(np.unique(donors[idx == p])) for p in (0, 1, 2)], dig[:16]))
    glob_ids = shard * N_don + donors
    idx2, _ = GROUPED_SPLIT(glob_ids, seed=0)
    print("with bank-global donor ids (shard * %d + donor): groups %s"
          % (N_don, [len(np.unique(glob_ids[idx2 == p])) for p in (0, 1, 2)]))
    print("rows of the first 512 of the concatenation (--max-probe-rows 512): shard 0 "
          "only: %s; donors %d, wells %d"
          % (bool(np.all(shard[:512] == 0)), len(np.unique(donors[:512])),
             len(np.unique(well[:512]))))


# ---------------------------------------------------------------------------
# B4  providers: the scale of x, the gap's range shift, Stage 3c's provider
# ---------------------------------------------------------------------------

def b4_providers(tmp):
    hr("B4  providers at phi = 0.5 (8 windows of 3000 samples, 100 neurons, seed 12345)")
    a = _defaults()
    J, W, fs, N = a.n_windows, int(round(a.T_win * a.fs)), a.fs, a.n_neurons
    ref = ReferenceBurstProvider()
    dsn = LS.load_dsn_provider()
    ben = BP.load_bench_provider()
    nus = LN.NuisanceSpec()
    sd_base = float(np.sqrt(sum(nus.scales[l][1] ** 2 for l in LN.NU_LEVELS)))
    for name, prov, p in (("reference", ref, 6), ("dsn", dsn, 6), ("bench", ben, 10)):
        x = prov(np.full(p, 0.5), J, W, fs, N, 12345)
        print("%-9s p = %2d: x mean %.5f, sd %.5f, min %.5f; additive nuisance sd %.4f = %.2f%% "
              "of the mean" % (name, p, x.mean(), x.std(), x.min(), sd_base,
                                100 * sd_base / x.mean()))
    params = BP.phi_to_bench_params(np.full(10, 0.5), n_neurons=N, duration_s=J * W / fs)
    print("bench expected_mfr(phi = 0.5) = %.4f spikes/s/neuron -> %.5f counts/bin/neuron at "
          "fs %.0f" % (BG.expected_mfr(params), BG.expected_mfr(params) / fs, fs))
    try:
        dsn(np.full(10, 0.5), 2, W, fs, N, 1)
        print("dsn provider with a 10-entry phi: accepted")
    except Exception as exc:   # noqa: BLE001
        print("dsn provider with a 10-entry phi (what run_stage3c.py:178-179 gives it on a "
              "bench bank): %s: %s" % (type(exc).__name__, exc))
    print("")
    shift_max = LG.GapSpec().shift_max
    lo, hi = [ax for ax in BP.BENCH_AXES if ax[0] == "fragment_duty"][0][1:]
    print("bench free axes %s; fragment_duty [%.2f, %.2f]; shift_max %.2f"
          % (list(BP.BENCH_FREE_IDX), lo, hi, shift_max))
    for pi in (0.05, 0.1, 0.25, 0.5, 1.0):
        d = pi * shift_max
        phi = np.full(10, 0.5)
        k = [i for i, ax in enumerate(BP.BENCH_AXES) if ax[0] == "fragment_duty"][0]
        phi[k] = 1.0 - d - 1e-9
        ok_below = True
        try:
            BP.phi_to_bench_params(phi, n_neurons=N, duration_s=10.0, free_axis_shift=d)
        except ValueError:
            ok_below = False
        phi[k] = 1.0 - d + 1e-9
        raised_above = False
        try:
            BP.phi_to_bench_params(phi, n_neurons=N, duration_s=10.0, free_axis_shift=d)
        except ValueError:
            raised_above = True
        N_don = a.n_traces // a.wells_per_donor
        print("  pi %.2f: duty > 1 iff phi_duty > 1 - pi*shift_max = %.4f (below accepted %s, "
              "above raises %s); P(a %d-donor shard raises) = 1 - (1 - %.4f)^%d = %.4f"
              % (pi, 1 - d, ok_below, raised_above, N_don, d, N_don,
                 1 - (1 - d) ** N_don))
    rc, last = _build(os.path.join(tmp, "BR"), "--arm", "R", "--provider", "bench",
                      "--pi", "0.1")
    print("build_latent_bank.py --provider bench --arm R --pi 0.1 (defaults otherwise): rc %d, "
          "%s" % (rc, last))
    rc, last = _build(os.path.join(tmp, "BR2"), "--arm", "R", "--provider", "bench",
                      "--pi", "0.5", "--gap-modes", "drift,contamination")
    arr, side = LB.read_shard(os.path.join(tmp, "BR2", "shard_0000.npz"))
    print("build_latent_bank.py --provider bench --arm R --pi 0.5 --gap-modes "
          "drift,contamination: rc %d; rows %d, contaminated windows %d, gap_spec modes %s"
          % (rc, arr["x"].shape[0], int(arr["contaminated"].sum()),
             side["gap_spec"]["modes"]))
    # the dsn provider at the defaults: no PAD_BINS, raises on a short trace
    # (latent_sbi_simulator.py:365-370); and with the gap (a) at pi = 0.5
    for tag, flags in (("DS", ("--provider", "dsn")),
                       ("DR", ("--provider", "dsn", "--arm", "R", "--pi", "0.5"))):
        rc, last = _build(os.path.join(tmp, tag), *flags)
        arr, side = LB.read_shard(os.path.join(tmp, tag, "shard_0000.npz"))
        print("build_latent_bank.py %s (defaults otherwise): rc %d; rows %d, W %d, "
              "gap_spec modes %s, pi %.1f"
              % (" ".join(flags), rc, arr["x"].shape[0], arr["x"].shape[1],
                 side["gap_spec"]["modes"], side["gap_spec"]["pi"]))
    # a shard stores x_obs, the window after the nuisance map (eq. P7.3): its
    # sign on a default bench shard and on the two dsn shards above
    rc, last = _build(os.path.join(tmp, "BS"), "--provider", "bench")
    assert rc == 0, last
    for tag in ("BS", "DS", "DR"):
        arr, side = LB.read_shard(os.path.join(tmp, tag, "shard_0000.npz"))
        x = np.asarray(arr["x"], dtype=np.float64)
        print("stored x of %s (%s, arm %s, pi %.1f): min %.5f; share of samples < 0: %.4f"
              % (tag, side["scale_convention"], side["arm"], side["gap_spec"]["pi"],
                 float(x.min()), float((x < 0).mean())))
    phi = np.full(6, 0.4)
    y1 = ref(phi, 2, 300, fs, N, 7, param_overrides={"free_axis_range_shift": 0.2})
    y2 = ref(phi + 0.2, 2, 300, fs, N, 7)
    print("reference fixture: shift 0.2 == phi + 0.2 on every axis, bitwise: %s"
          % bool(np.array_equal(y1, y2)))
    phi = np.full(6, 0.999)
    try:
        dsn(phi, 2, W, fs, N, 3, param_overrides={"free_axis_range_shift": shift_max})
        print("dsn provider at pi = 1 (shift %.2f), phi = 0.999 on every axis: accepted"
              % shift_max)
    except Exception as exc:   # noqa: BLE001
        print("dsn provider at pi = 1: %s: %s" % (type(exc).__name__, exc))


# ---------------------------------------------------------------------------
# B5  the nuisance: scales, the identity, the dropout threshold, the floor
# ---------------------------------------------------------------------------

def b5_nuisance(S):
    hr("B5  NuisanceSpec() at the defaults")
    sp = LN.NuisanceSpec()
    sc = {l: sp.scales[l] for l in LN.NU_LEVELS}
    tot = np.sqrt(sum(sc[l] ** 2 for l in LN.NU_LEVELS))
    for m, name in enumerate(LN.NU_COMPONENTS):
        share_b = sc["batch"][m] ** 2 / tot[m] ** 2
        share_fl = (sc["donor"][m] ** 2 + sc["well"][m] ** 2) / tot[m] ** 2
        print("%-13s batch %.2f donor %.2f well %.2f -> total sd %.4f; batch share of the "
              "variance %.4f; the floor's share (donor + well) %.4f; 2*well^2 == donor^2 + "
              "well^2: %s" % (name, sc["batch"][m], sc["donor"][m], sc["well"][m], tot[m],
                              share_b, share_fl,
                              bool(np.isclose(2 * sc["well"][m] ** 2,
                                              sc["donor"][m] ** 2 + sc["well"][m] ** 2))))
    t = np.arange(5, dtype=float)
    A0, b0 = LN.nuisance_affine(sp, np.zeros(5), t)
    print("nu = 0: gain %.6f, additive part max |b| %.1e (the identity)" % (A0, np.abs(b0).max()))
    thr = math.log(1.0 / 17.0) - sp.dropout_logit0
    for nu in (thr - 1e-6, thr + 1e-6):
        A, _ = LN.nuisance_affine(sp, np.array([0, 0, 0, nu, 0.0]), t)
        print("dropout_logit %.6f -> gain %.6f" % (nu, A))
    p = stats.norm.sf(thr / tot[3])
    print("an electrode is lost when dropout_logit > ln(1/17) - (%.1f) = %.4f = %.3f total "
          "sd; P = %.3e per well" % (sp.dropout_logit0, thr, thr / tot[3], p))
    thr2 = math.log(1.0 / 5.0) - sp.dropout_logit0
    print("a second electrode when dropout_logit > %.4f = %.2f total sd; P = %.1e"
          % (thr2, thr2 / tot[3], stats.norm.sf(thr2 / tot[3])))
    gains = np.array([LN.nuisance_affine(sp, r, t)[0] for r in np.unique(S["nu"], axis=0)])
    drop = np.array([LN.nuisance_affine(sp, np.array([0, 0, 0, r[3], 0.0]), t)[0]
                     for r in np.unique(S["nu"], axis=0)])
    print("two-shard bank: %d wells; dropout factor < 1 in %d; gain range [%.4f, %.4f]"
          % (len(gains), int((drop < 1).sum()), gains.min(), gains.max()))
    h = 0.5 * tot[3]
    A_p, _ = LN.nuisance_affine(sp, np.array([0, 0, 0, h, 0.0]), t)
    A_m, _ = LN.nuisance_affine(sp, np.array([0, 0, 0, -h, 0.0]), t)
    A_P, _ = LN.nuisance_affine(sp, np.array([0, 0, 0, 8 * h, 0.0]), t)
    A_M, _ = LN.nuisance_affine(sp, np.array([0, 0, 0, -8 * h, 0.0]), t)
    print("aliasing.jacobian_nu step for dropout: 0.5 * %.4f = %.4f -> gains %.4f / %.4f "
          "(equal); 8x step %.4f -> %.4f / %.4f (quantised flag)"
          % (tot[3], h, A_p, A_m, 8 * h, A_P, A_M))
    print("drift periods: nuisance %.0f s, gap %.0f s; a trace spans %.0f s"
          % (sp.drift_period_s, LG.GapSpec().drift_period_s,
             _defaults().n_windows * _defaults().T_win))


# ---------------------------------------------------------------------------
# B6  Stage 3b / 3c arithmetic and the demo generator
# ---------------------------------------------------------------------------

def b6_post_hoc(tmp):
    hr("B6  Stage 3b / 3c knobs, and the demo generator")
    a = _defaults()
    print("--max-probe-rows 512 vs rows per default shard %d" % (a.n_traces * a.n_windows))
    print("--n-post-draws: job 128, CLI 64 -> the Monte Carlo term of the floor covariance "
          "differs by the factor %.1f" % (128 / 64))
    for k, d in ((3, 26), (3, 6), (3, 10)):
        print("kernel concentration threshold 0.5 * (1 + k/d) at k = %d, d = %d: %.4f "
              "(uniform %.4f)" % (k, d, 0.5 * (1 + k / d), k / d))
    nu, parts = LN.sample_nuisance(LN.NuisanceSpec(), ["B"] * 4, ["D%d" % i for i in range(4)],
                                   ["W%d" % i for i in range(4)], base_seed=0)
    print("sample_nuisance returns a %s of length 2 (nu %s, parts %s)"
          % (type((nu, parts)).__name__, nu.shape, sorted(parts)))
    cmd = [sys.executable, "demo_classes_generate.py", "--out",
           os.path.join(tmp, "demo.npz"), "--n-per-class", "2", "--n-background", "0",
           "--n-windows", "2", "--T-win", "5", "--nuisance"]
    res = subprocess.run(cmd, cwd=_STAGE1, capture_output=True, text=True)
    last = (res.stdout + res.stderr).strip().splitlines()[-1]
    print("demo_classes_generate.py --nuisance: rc %d, %s" % (res.returncode, last))


if __name__ == "__main__":
    b1_geometry()
    b2_jobs()
    tmp = tempfile.mkdtemp(prefix="p7_")
    try:
        S = b3_banks(tmp)
        b3_synthetic()
        b4_providers(tmp)
        b5_nuisance(S)
        b6_post_hoc(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
