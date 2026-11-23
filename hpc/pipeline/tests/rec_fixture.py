#!/usr/bin/env python3
"""The fixture of the recorded evaluation's tests (block 33; specs/SPEC.md
block 33, "The fixture"): built once per process, a few minutes in the
sandbox.

    F = rec_fixture.fixture()
    F["root"]       the tree
    F["cfg"]        the reduced synthetic project's configuration (synth.json)
    F["ecfg_old"]   the synthetic evaluation's configuration (blocks 19-22)
    F["old_out"]    its out_root: the samples and metrics of blocks 19-20
    F["ecfg_as"]    rec_eval_as.json: the project read as recorded (D-162)
    F["cfg_rec"]    a recorded-kind configuration on the same banks and runs
    F["ecfg_rec"]   rec_eval.json: its evaluation's configuration

What is built, in `root`:
    synth.json               block 19's reduced project (S-A1; S-A2-R0 with
                             alt_best_fit; cohorts R0 and R3; two members)
                             with synthetic.sim.n_theta 160 (8 networks in
                             the report split), run to PIPELINE DONE under
                             the fake PBS
    synth_eval.json          the synthetic evaluation's configuration
                             (test_smoke_evaluation._eval_config), out_root
                             <root>/evaluation_test: its calibration bank,
                             the samples of every complete ensemble (block
                             19) and the metrics (block 20)
    rec_eval_as.json         as_recorded true, the synthetic evaluation's
                             seed, n_post, batch_rows and oracle_tol,
                             sim_draw_rows_per_donor null, gate.n_null_min
                             1 and gate.alpha 0.5 (amendment 2, D-165: the
                             null set has 2 rows, so k_gate = 1),
                             validate_against the synthetic out_root
    synth_rec.json           the committed linddup.json with `dataset`,
                             `paths` and `searches` replaced: dataset synth,
                             the synthetic project's roots, the searches
                             S-A1 (S-A1) and S-A2-R0 (S-A2), max_cores null
    banks_root/synth_real/   its recording bank: cohort R0's shards with
                             theta of zero columns and a sidecar of arm R
    rec_eval.json            as_recorded false, sim_draw_rows_per_donor 1,
                             gate.n_null_min 1, gate.alpha 0.5

Each step is skipped when its outputs exist, so a tree given by
REC_FIXTURE=<dir> is completed rather than rebuilt (it must hold a
synth.json with n_theta 160); the three configuration files the fixture
writes (rec_eval_as.json, synth_rec.json, rec_eval.json) are rewritten
when their JSON content is not the one it writes now (amendment 2). REC_KEEP=1 keeps the tree this process builds
and prints where. Needs the training environment (torch, sbi, zuko,
scikit-optimize); `need_training_env()` raises unittest.SkipTest without
it. Not pipeline code: a test fixture.

CONTRACT (wave A): the keys above are fixed by specs/SPEC.md block 33.

Pure ASCII, LF only.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_EVAL = os.path.join(_PIPE, "evaluation")
for _p in (_HERE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURE_KEYS = ("root", "cfg", "ecfg_old", "old_out", "ecfg_as", "cfg_rec", "ecfg_rec")

LINDDUP = os.path.join(_PIPE, "projects", "linddup", "linddup.json")
N_THETA = 160            # block 33: 8 networks in the report split (60 in block 19's fixture)

_FIX = {}


def need_training_env():
    """unittest.SkipTest when torch, sbi, zuko or skopt cannot be imported."""
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("the training environment is needed (%s)" % exc)


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


# ------------------------------------------------- (1) the reduced project ---

def _build_project(root):
    """Block 19's reduced project (test_smoke_evaluation._build_project) with
    synthetic.sim.n_theta 160, run to PIPELINE DONE; returns synth.json."""
    import test_smoke_synthetic as TS
    searches = [{"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False},
                {"name": "S-A2-R0", "campaign": "S-A2", "real": "R0", "alt_best_fit": True}]
    real = [{"name": "R0", "seed": 22, "donors_per_class": 2, "n_windows": 2, "pi": 0.0,
             "gap_modes": ["range_shift", "drift", "contamination"]},
            {"name": "R3", "seed": 34, "donors_per_class": 2, "n_windows": 2, "pi": 0.3,
             "gap_modes": ["range_shift", "drift", "contamination"]}]
    path, _d = TS._small(root, searches=searches,
                         **{"synthetic.sim": {"seed": 11, "n_theta": N_THETA, "n_windows": 1,
                                              "traces_per_shard": 40},
                            "synthetic.real": real})
    saved = dict(os.environ)
    try:
        env = TS._fake_env(os.path.join(root, "spool"))
        r = subprocess.run([sys.executable, os.path.join(_PIPE, "pipeline.py"), "run",
                            "--config", path, "--poll-seconds", "0.5"], cwd=root, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=3600)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    if not (r.returncode == 0 and "PIPELINE DONE: every stage done or skipped" in r.stdout):
        raise AssertionError("the fixture's project did not run to PIPELINE DONE:\n%s"
                             % r.stdout[-6000:])
    return path


# ------------------------------------------ (2) the synthetic evaluation ---

def _old_evaluation(root, cfg):
    """The synthetic evaluation (blocks 19-20) of every complete ensemble,
    with its calibration bank; returns (ecfg_old path, its out_root)."""
    import test_smoke_evaluation as TE
    import ev_config
    import ev_plan
    ecfg_path = os.path.join(root, "synth_eval.json")
    if not os.path.isfile(ecfg_path):
        TE._eval_config(root)
    ecfg = ev_config.load(ecfg_path)
    TE._build_calbank(cfg, ecfg_path)
    todo = [e.name for e in ev_plan.ensembles(ev_plan.load_project(cfg))
            if not os.path.isfile(ev_plan.sample_json(ecfg, e.stem))]
    if todo:
        import ev_sample
        for name in todo:
            lines = []
            ev_sample.run(cfg, ecfg_path, name, log=lines.append)
            if not any(l.startswith("EVAL SAMPLE OK: ") for l in lines):
                raise AssertionError("the synthetic sampling of %s failed:\n%s"
                                     % (name, "\n".join(lines)))
    if todo or not os.path.isfile(os.path.join(ecfg.sub("metrics"), "METRICS.json")):
        import ev_metrics
        lines = []
        ev_metrics.compute(cfg, ecfg_path, ncpus=2, log=lines.append)
        if not (lines and lines[-1].startswith("EVAL METRICS OK: ")):
            raise AssertionError("the synthetic metrics failed:\n%s" % "\n".join(lines))
    return ecfg_path, ecfg.out_root


# ----------------------------------------- (3) and (4) the configurations ---

def rec_eval_dict(out_root, **over):
    """A recorded evaluation's configuration (ev_rec_config.KEYS) for the
    fixture: D-163's and D-164's values where the fixture does not need its
    own; `over` replaces top-level keys (and `gate` keys under "gate")."""
    d = {"out_root": out_root, "seed": 4242, "n_post": 40, "batch_rows": 16,
         "oracle_tol": 2e-3, "as_recorded": False, "sim_draw_rows_per_donor": 1,
         "edge_eps": 0.006693,
         "gate": {"donor_fracs": [0.4, 0.25, 0.25, 0.1], "n_ref": 800, "n_rank": 1000,
                  "n_null": 1000, "n_null_min": 1, "rows_per_donor": 1,
                  "factors": [0.25, 0.5, 1, 2, 4], "alpha": 0.5},
         "bootstrap_n": 200, "level": 0.95, "dpi": 60, "formats": ["png"], "ncpus": 2,
         "validate_against": None,
         "jobs": {"queue": "test",
                  "sample": {"ncpus": 2, "mem": "2gb", "walltime": "00:20:00"},
                  "report": {"ncpus": 2, "mem": "2gb", "walltime": "00:30:00"},
                  "validate": {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00"}}}
    gate = over.pop("gate", None)
    d.update(over)
    if gate:
        d["gate"].update(gate)
    return d


def _write_if_changed(path, obj):
    """Writes `obj` unless the file holds that JSON content already
    (amendment 2: a reused tree's configuration of other values is
    rewritten)."""
    if os.path.isfile(path):
        try:
            if _read_json(path) == obj:
                return path
        except ValueError:
            pass
    _write_json(path, obj)
    return path


def _ecfg_as(root, ecfg_old):
    """rec_eval_as.json: the synthetic project read as recorded (D-162)."""
    old = _read_json(ecfg_old)
    return _write_if_changed(os.path.join(root, "rec_eval_as.json"), rec_eval_dict(
        os.path.join(root, "rec_eval_as_out"), seed=old["seed"], n_post=old["n_post"],
        batch_rows=old["batch_rows"], oracle_tol=old["oracle_tol"], as_recorded=True,
        sim_draw_rows_per_donor=None, validate_against=old["out_root"]))


def _cfg_rec(root, cfg):
    """synth_rec.json: the committed linddup.json with its dataset, paths
    and searches replaced by the synthetic project's."""
    path = os.path.join(root, "synth_rec.json")
    syn = _read_json(cfg)
    d = _read_json(LINDDUP)
    d["dataset"] = syn["dataset"]
    paths = dict(d["paths"])
    for k in paths:
        paths[k] = syn["paths"][k] if k in syn["paths"] else os.path.join(root, k)
    d["paths"] = paths
    d["searches"] = [{"name": q["name"], "campaign": q["campaign"], "max_cores": None}
                     for q in syn["synthetic"]["searches"]]
    return _write_if_changed(path, d)


def _real_bank(cfg):
    """`<banks_root>/<dataset>_real`: cohort R0's shards with theta of zero
    columns and a sidecar of arm R (an extractor's real bank's shape)."""
    import _evalpath  # noqa: F401
    import numpy as np
    from latent_bank import build_sidecar, read_shard, write_shard
    syn = _read_json(cfg)
    banks = syn["paths"]["banks_root"]
    src = sorted(glob.glob(os.path.join(banks, "%s_real_R0" % syn["dataset"], "shard_*.npz")))
    if not src:
        raise AssertionError("cohort R0 has no shard under %s" % banks)
    out = os.path.join(banks, "%s_real" % syn["dataset"])
    want = [os.path.join(out, os.path.basename(p)) for p in src]
    if all(os.path.isfile(p) for p in want):
        return out
    for p, q in zip(src, want):
        arrays, side = read_shard(p)
        arrays = dict(arrays)
        n = arrays["theta"].shape[0]
        arrays["theta"] = np.zeros((n, 0), dtype=arrays["theta"].dtype)
        new = build_sidecar(
            param_names=[], bounds_theta=[], coord=side["coord"], fs=side["fs"],
            w_size=side["w_size"], T_win=side["T_win"], W=side["W"],
            scale_convention=side["scale_convention"], latent_spec=side["latent_spec"],
            nuisance_spec=side["nuisance_spec"], realisation_spec=side["realisation_spec"],
            gap_spec=side["gap_spec"], generator_sha256=side["generator_sha256"],
            theta_withheld=True, arm="R",
            extra={"fixture": "block 33: cohort R0's shard %s with theta of zero columns"
                              % os.path.basename(p)})
        write_shard(q, arrays, new)
    return out


def _ecfg_rec(root):
    return _write_if_changed(os.path.join(root, "rec_eval.json"), rec_eval_dict(
        os.path.join(root, "rec_eval_out"), as_recorded=False, sim_draw_rows_per_donor=1))


# ------------------------------------------------------------- fixture ---

def fixture():
    """dict with FIXTURE_KEYS (and `built`, True when this process built the
    tree): the fixture of specs/SPEC.md block 33, built once per process."""
    if _FIX.get("root"):
        return _FIX
    need_training_env()
    given = os.environ.get("REC_FIXTURE", "").strip()
    t0 = time.time()
    if given:
        root = os.path.abspath(given)
        cfg = os.path.join(root, "synth.json")
        if not os.path.isfile(cfg):
            raise AssertionError("REC_FIXTURE=%s has no synth.json" % root)
        n = _read_json(cfg)["synthetic"]["sim"]["n_theta"]
        if n != N_THETA:
            raise AssertionError("REC_FIXTURE=%s: synthetic.sim.n_theta is %d, the fixture's %d "
                                 "(block 33)" % (root, n, N_THETA))
    else:
        root = tempfile.mkdtemp(prefix="rec_fixture_")
        _FIX["_tmp"] = root                    # cleanup() removes a tree half built too
        cfg = _build_project(root)
        print("[rec_fixture] the reduced project ran to PIPELINE DONE in %.0f s: %s"
              % (time.time() - t0, root), flush=True)
    ecfg_old, old_out = _old_evaluation(root, cfg)
    ecfg_as = _ecfg_as(root, ecfg_old)
    cfg_rec = _cfg_rec(root, cfg)
    _real_bank(cfg)
    ecfg_rec = _ecfg_rec(root)
    _FIX.clear()
    _FIX.update(root=root, cfg=cfg, ecfg_old=ecfg_old, old_out=old_out, ecfg_as=ecfg_as,
                cfg_rec=cfg_rec, ecfg_rec=ecfg_rec, built=not given)
    print("[rec_fixture] ready in %.0f s: %s" % (time.time() - t0, root), flush=True)
    return _FIX


def cleanup():
    """Removes the tree this process built, unless REC_KEEP=1 (then prints
    where it is). A tree given by REC_FIXTURE is never removed."""
    root = _FIX.get("root") or _FIX.get("_tmp")
    built = _FIX.get("built", "_tmp" in _FIX)
    if not root or not built:
        return
    if os.environ.get("REC_KEEP", "").strip() == "1":
        print("[rec_fixture] kept: %s (REC_FIXTURE=%s reuses it)" % (root, root), flush=True)
        return
    shutil.rmtree(root, ignore_errors=True)
    _FIX.clear()
