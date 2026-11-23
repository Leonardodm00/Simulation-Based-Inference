#!/usr/bin/env python3
"""Block 36: the validation of the recorded path against the synthetic
evaluation (specs/SPEC.md sec. 2.6, "The validation", the cluster's
targets, and block 36; decisions D-162, D-163 (4), D-164).

    python3 hpc/pipeline/evaluation/ev_rec_validate.py \\
        --old <out_root of the synthetic evaluation> \\
        --new <out_root of the as-recorded evaluation> \\
        [--nll-tol 0.002] [--coverage-tol 0.01]

Compares, for each ensemble, on D_sim: the NLL of every row, the expected
coverage on the grid, the SBC verdicts and Holm rejections, TARP's two
verdicts. Exit 0 with `REC VALIDATION OK: <n> ensemble(s)`; exit 3 with one
line per failure otherwise; 2 on the arguments.

The old tree is block 20's (`<out_root>/metrics/METRICS.json` and
`<stem>_sim.json/.npz`), the new one block 36's. TARP's verdict for a
reference mode is "ks_p < 1 - level", `level` the new tree's
`<stem>_sim.json` level (block 20 writes no verdict for TARP); the SBC
verdicts are the `verdict` and `holm_rejected` lists of the two files.

Reads JSON and NPZ files only (numpy); no other part of the pipeline.

Pure ASCII, LF only.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

EXIT_ARGS, EXIT_FAIL = 2, 3
NLL_TOL = 0.002          # nats per row (D-163 (4))
COVERAGE_TOL = 0.01      # on every point of the grid (D-163 (4))
TARP_MODES = ("x", "random")


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _ensembles(root, label, failures):
    """{name: stem} of a tree's METRICS.json, or None (a failure line) when
    it cannot be read."""
    p = os.path.join(root, "metrics", "METRICS.json")
    if not os.path.isfile(p):
        failures.append("%s tree: %s absent" % (label, p))
        return None
    try:
        idx = _read_json(p)
        return dict((name, v["stem"]) for name, v in idx["ensembles"].items())
    except (ValueError, KeyError, TypeError) as exc:
        failures.append("%s tree: %s unreadable (%s)" % (label, p, exc))
        return None


def _files(root, stem):
    base = os.path.join(root, "metrics", "%s_sim" % stem)
    return base + ".json", base + ".npz"


def _fmt(v):
    return "null" if v is None or (isinstance(v, float) and math.isnan(v)) else repr(v)


def _tarp_verdict(js, mode, level):
    """True when TARP rejects in `mode` (ks_p < 1 - level); None when its
    ks_p is null."""
    ks = ((js.get("tarp") or {}).get(mode) or {}).get("ks_p")
    return None if ks is None else bool(float(ks) < 1.0 - float(level))


def _compare(name, old_root, new_root, so, sn, nll_tol, coverage_tol):
    """The failure lines of one ensemble."""
    out = []
    jo_p, zo_p = _files(old_root, so)
    jn_p, zn_p = _files(new_root, sn)
    for p in (jo_p, zo_p, jn_p, zn_p):
        if not os.path.isfile(p):
            return ["%s: %s absent" % (name, p)]
    jo, jn = _read_json(jo_p), _read_json(jn_p)
    with np.load(zo_p) as z:
        zo = dict((k, z[k]) for k in ("rows", "nll_rows", "coverage", "betas"))
    with np.load(zn_p) as z:
        zn = dict((k, z[k]) for k in ("rows", "nll_rows", "coverage", "betas"))
    # the rows
    if zo["rows"].shape != zn["rows"].shape or not np.array_equal(zo["rows"], zn["rows"]):
        diff = ("%d rows against %d" % (zo["rows"].size, zn["rows"].size)
                if zo["rows"].shape != zn["rows"].shape else
                "first difference at row %d: bank row %d against %d" % (
                    int(np.flatnonzero(zo["rows"] != zn["rows"])[0]),
                    int(zo["rows"][zo["rows"] != zn["rows"]][0]),
                    int(zn["rows"][zo["rows"] != zn["rows"]][0])))
        return ["%s: rows of D_sim differ (old %s, new %s): %s" % (name, zo_p, zn_p, diff)]
    # the NLL, row by row
    lo, ln = zo["nll_rows"].astype(np.float64), zn["nll_rows"].astype(np.float64)
    for i in range(lo.size):
        a, b = lo[i], ln[i]
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) or np.isnan(b) or abs(a - b) > nll_tol:
            out.append("%s: nll row %d (bank row %d): old %s new %s, |diff| %s > %g"
                       % (name, i, int(zo["rows"][i]), _fmt(float(a)), _fmt(float(b)),
                          _fmt(float(abs(a - b))), nll_tol))
    # the expected coverage on the grid
    if zo["coverage"].shape != zn["coverage"].shape or not np.allclose(zo["betas"], zn["betas"]):
        out.append("%s: coverage grids differ (%d against %d points)"
                   % (name, zo["coverage"].size, zn["coverage"].size))
    else:
        dev = np.abs(zo["coverage"].astype(np.float64) - zn["coverage"].astype(np.float64))
        k = int(np.argmax(dev))
        if not dev[k] <= coverage_tol:
            out.append("%s: coverage at beta %.2f: old %s new %s, largest |diff| over the grid "
                       "%s > %g" % (name, float(zn["betas"][k]), _fmt(float(zo["coverage"][k])),
                                    _fmt(float(zn["coverage"][k])), _fmt(float(dev[k])),
                                    coverage_tol))
    # SBC: each axis's verdict and the Holm rejections
    so_, sn_ = jo.get("sbc") or {}, jn.get("sbc") or {}
    for key in ("verdict", "holm_rejected"):
        vo, vn = so_.get(key), sn_.get(key)
        if not isinstance(vo, list) or not isinstance(vn, list) or len(vo) != len(vn):
            out.append("%s: sbc %s: old %s new %s" % (name, key, vo, vn))
            continue
        for k, (a, b) in enumerate(zip(vo, vn)):
            if a != b:
                out.append("%s: sbc %s axis %d: old %r new %r" % (name, key, k, a, b))
    # TARP's two verdicts, at the new tree's level
    level = jn.get("level")
    if level is None:
        out.append("%s: the new tree's %s has no level" % (name, jn_p))
    else:
        for mode in TARP_MODES:
            vo, vn = _tarp_verdict(jo, mode, level), _tarp_verdict(jn, mode, level)
            if vo != vn:
                out.append("%s: tarp %s verdict (ks_p < %g): old %s (ks_p %s) new %s (ks_p %s)"
                           % (name, mode, 1.0 - float(level), vo,
                              _fmt(((jo.get("tarp") or {}).get(mode) or {}).get("ks_p")), vn,
                              _fmt(((jn.get("tarp") or {}).get(mode) or {}).get("ks_p"))))
    return out


def validate(old_root, new_root, nll_tol=NLL_TOL, coverage_tol=COVERAGE_TOL):
    """(n_ensembles, failures): `failures` a list of strings, one per
    failure, each naming the ensemble, the quantity, the row or the beta,
    and the two values; empty when the new tree matches the old one. An
    ensemble in one tree only, or rows that differ, is a failure."""
    failures = []
    eo = _ensembles(old_root, "old", failures)
    en = _ensembles(new_root, "new", failures)
    if eo is None or en is None:
        return 0, failures
    for name in sorted(set(eo) - set(en)):
        failures.append("%s: in the old tree only" % name)
    for name in sorted(set(en) - set(eo)):
        failures.append("%s: in the new tree only" % name)
    both = sorted(set(eo) & set(en))
    if not eo and not en:
        failures.append("no ensemble in either tree")
    for name in both:
        failures += _compare(name, old_root, new_root, eo[name], en[name], nll_tol, coverage_tol)
    return len(both), failures


def main(argv=None):
    """The command line: 0 with the pass line, EXIT_ARGS, EXIT_FAIL."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="the out_root of the synthetic evaluation")
    ap.add_argument("--new", required=True, help="the out_root of the as-recorded evaluation")
    ap.add_argument("--nll-tol", type=float, default=NLL_TOL, help="nats per row")
    ap.add_argument("--coverage-tol", type=float, default=COVERAGE_TOL)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return EXIT_ARGS if exc.code else 0
    bad = [d for d in (args.old, args.new) if not os.path.isdir(d)]
    if bad or not (args.nll_tol >= 0 and args.coverage_tol >= 0):
        print("ARGS: %s" % ("not a directory: %s" % ", ".join(bad) if bad
                            else "the tolerances must be >= 0"), flush=True)
        return EXIT_ARGS
    n, failures = validate(args.old, args.new, args.nll_tol, args.coverage_tol)
    if failures:
        for f in failures:
            print("FAIL: %s" % f, flush=True)
        print("REC VALIDATION FAILED: %d failure(s) over %d ensemble(s)" % (len(failures), n),
              flush=True)
        return EXIT_FAIL
    print("REC VALIDATION OK: %d ensemble(s)" % n, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
