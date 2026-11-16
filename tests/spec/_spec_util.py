"""Shared helpers of the tester's spec-derived tests (tests/spec/).

Builds small synthetic projects in a temporary folder from the committed
`synth.json` / `synth_eval.json` (their values, the paths moved), and writes
block 19's sample files by hand so that block 27's paired comparisons can be
checked against answers computed here from specs/SPEC.md. Nothing here calls
the code under test to compute an expected value.

Pure ASCII, LF only.
"""

import copy
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
EVAL = os.path.join(PIPE, "evaluation")
TESTS = os.path.join(PIPE, "tests")
SYNTH = os.path.join(PIPE, "projects", "synth", "synth.json")
SYNTH_EVAL = os.path.join(PIPE, "projects", "synth", "synth_eval.json")

if EVAL not in sys.path:
    sys.path.insert(0, EVAL)
import _evalpath  # noqa: E402,F401  (side effect: the pipeline's config first, as the modules do)


def committed(path=SYNTH):
    with open(path) as fh:
        return json.load(fh)


def write_project(root, searches=None, real=None, bootstrap_n=400, seed=4242, level=0.95):
    """A copy of the committed synth.json / synth_eval.json with every path under
    `root` (sbi_dir: this checkout) and optionally other searches; returns
    (cfg_path, ecfg_path)."""
    d = committed(SYNTH)
    d["paths"]["sbi_dir"] = REPO
    for k in ("work_root", "banks_root", "runs_root", "tune_root"):
        d["paths"][k] = os.path.join(root, k)
    if searches is not None:
        d["synthetic"]["searches"] = copy.deepcopy(searches)
    if real is not None:
        d["synthetic"]["real"] = copy.deepcopy(real)
    e = committed(SYNTH_EVAL)
    e["out_root"] = os.path.join(root, "evaluation")
    e["cal_glob"] = os.path.join(e["out_root"], "calbank", "shard_*.npz")
    e["bootstrap_n"] = int(bootstrap_n)
    e["seed"] = int(seed)
    e["level"] = float(level)
    os.makedirs(root, exist_ok=True)
    cp, ep = os.path.join(root, "synth.json"), os.path.join(root, "synth_eval.json")
    with open(cp, "w") as fh:
        json.dump(d, fh, indent=1)
    with open(ep, "w") as fh:
        json.dump(e, fh, indent=1)
    return cp, ep


def search(name, campaign, real, alt=False):
    return {"name": name, "campaign": campaign, "real": real, "alt_best_fit": alt}


def make_complete(cfg_data, q, arm, seeds=(1, 2, 3), suffix="_best"):
    """A complete best fit on disk, as block 7 leaves it, for ev_plan.ensembles:
    BEST_FIT_COMPLETE.json with the member seeds and one record per member
    carrying the arm."""
    folder = os.path.join(cfg_data["paths"]["runs_root"], q, cfg_data["dataset"] + suffix)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "BEST_FIT_COMPLETE.json"), "w") as fh:
        json.dump({"member_seeds": list(seeds)}, fh)
    for s in seeds:
        md = os.path.join(folder, "member_seed%d" % s)
        os.makedirs(md, exist_ok=True)
        with open(os.path.join(md, "%s_seed%d.json" % (arm, s)), "w") as fh:
            json.dump({"arm": arm, "seed": s}, fh)
    return folder


def write_sample(ecfg, stem, datasets, params=None):
    """Block 19's sample of one ensemble written by hand: <stem>_<D>.npz with the
    arrays a comparison reads (datasets: [(D, dict of arrays)], in provenance
    order) and <stem>_sample.json listing them. Returns the provenance path."""
    import numpy as np
    sdir = os.path.join(ecfg.out_root, "samples")
    os.makedirs(sdir, exist_ok=True)
    files = []
    for D, arrays in datasets:
        p = os.path.join(sdir, "%s_%s.npz" % (stem, D))
        np.savez(p, **arrays)
        files.append(p)
    prov = {"schema": "eval_sample/1", "stem": stem, "files": files,
            "sample_params": ecfg.sample_params() if params is None else params}
    pj = os.path.join(sdir, "%s_sample.json" % stem)
    with open(pj, "w") as fh:
        json.dump(prov, fh)
    return pj
