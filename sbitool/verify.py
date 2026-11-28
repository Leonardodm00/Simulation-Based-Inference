"""Block 47: a bundle verified against the evaluation it was exported from
(specs/SPEC.md block 47; decision D-181).

The tool's central guarantee (sec. 2.7) as a command: an inference from the
bundle (block 45) on the recording banks of the evaluated project, compared
with the files blocks 35 and 36 wrote for them. This module holds no
estimator: the inference is block 45's, the comparison numpy's.

    python3 -m sbitool verify --bundle FILE --config <tag>.json \
        --eval-config <tag>_rec_eval.json --ensemble <name> \
        [--bank NAME ...] [--max-rows N] [--out DIR]

Exit 0 `VERIFY OK: ...`; 1 `VERIFY FAILED: ...`; 2 the arguments; 3 a
refusal (`REFUSED: ...`, nothing written).

Pure ASCII, LF only.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from sbitool import _sources  # noqa: F401  (the evaluation's modules come through it)

VERIFY_SCHEMA = "sbitool_verify/1"
EXIT_FAILED, EXIT_ARGS, EXIT_REFUSE = 1, 2, 3
JSON_NAME = "VERIFY.json"
GATE_TOL = 1e-12        # block 45's T45a: the gate's float64 arrays, absolute

# what is compared with which file, and how
SAMPLE_ARRAYS = ("draws", "draw_member", "logq_draws", "logq_draws_own", "z")   # block 35, bitwise
SUMMARY_ARRAYS = ("mhat", "sigma", "q05", "q50", "q95", "contraction")          # block 36, bitwise
GATE_ARRAYS = ("gate_t", "gate_r", "gate_p_member", "gate_T", "gate_p")         # block 36, GATE_TOL
FLAG_ARRAYS = ("gate_flag",)                                                    # block 36, equal
COMPARED = SAMPLE_ARRAYS + SUMMARY_ARRAYS + GATE_ARRAYS + FLAG_ARRAYS
# the arrays whose window axis is axis 1 ((M, n, ...)); axis 0 for the others,
# and none for draw_member ((n_post,))
WINDOW_AXIS_1 = ("z", "gate_t", "gate_r", "gate_p_member")
NO_WINDOW_AXIS = ("draw_member",)

RECORD_KEYS = ("schema", "bundle", "project", "ensemble", "stem", "n_post", "batch_rows",
               "torch_threads", "max_rows", "banks", "identical", "tolerance_gate", "software",
               "seconds")
BANK_KEYS = ("n", "n_compared", "seed", "identical", "arrays")
ARRAY_KEYS = ("equal", "max_abs_diff", "shape", "dtype")


class VerifyRefused(ValueError):
    """A verification that is not made; the message says why."""


# ---------------------------------------------------------------- the comparison

def first_rows(name, a, n):
    """Array `name` of the evaluation restricted to its first n windows."""
    if name in NO_WINDOW_AXIS:
        return a
    return a[:, :n] if name in WINDOW_AXIS_1 else a[:n]


def _max_abs_diff(a, b):
    """The largest absolute difference of the float64 values; inf where the
    two disagree on a NaN or an infinity's sign; 0.0 for empty arrays."""
    a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a64.size == 0:
        return 0.0
    both_nan = np.isnan(a64) & np.isnan(b64)
    same = both_nan | (a64 == b64)
    with np.errstate(invalid="ignore"):
        d = np.abs(a64 - b64)
    d = np.where(same, 0.0, np.where(np.isfinite(d), d, np.inf))
    return float(d.max())


def _bitwise_equal(a, b):
    """Same shape, same dtype, every element bitwise equal (NaN to NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def compare(arrays, evaluation, n=None):
    """The comparison of one bank (SPEC block 47, "Rule").

    arrays      the inference's arrays (infer.run(...)["arrays"])
    evaluation  {name: array} of block 35's and block 36's files for the bank
    n           None, or the number of first windows the inference was run
                on: the evaluation's arrays are restricted to them

    Returns {name: {equal, max_abs_diff, shape, dtype}} for each name of
    COMPARED: SAMPLE_ARRAYS and SUMMARY_ARRAYS equal when bitwise equal,
    GATE_ARRAYS when the shapes agree and max_abs_diff <= GATE_TOL,
    FLAG_ARRAYS when equal elementwise. shape and dtype are the
    inference's. An array of another shape is not equal, max_abs_diff inf."""
    out = {}
    for name in COMPARED:
        a = np.asarray(arrays[name])
        b = np.asarray(evaluation[name])
        if n is not None:
            b = first_rows(name, b, n)
        if a.shape != b.shape:
            equal, diff = False, float("inf")
        elif name in GATE_ARRAYS:
            diff = _max_abs_diff(a, b)
            equal = diff <= GATE_TOL
        elif name in FLAG_ARRAYS:
            equal = bool(np.array_equal(a, b))
            diff = 0.0 if equal else 1.0       # a flag that differs differs by 1
        else:
            equal = _bitwise_equal(a, b)
            diff = 0.0 if equal else _max_abs_diff(a, b)
        out[name] = {"equal": bool(equal), "max_abs_diff": diff, "shape": list(a.shape),
                     "dtype": str(a.dtype)}
    return out


# ---------------------------------------------------------------- the evaluation's side

def _load_evaluation(ecfg, stem, bank):
    """{name: array} of COMPARED from samples/<stem>_<bank>.npz (block 35)
    and metrics/<stem>_<bank>.npz (block 36). VerifyRefused for an absent
    file or an absent array."""
    sp = os.path.join(ecfg.sub("samples"), "%s_%s.npz" % (stem, bank))
    mp = os.path.join(ecfg.sub("metrics"), "%s_%s.npz" % (stem, bank))
    out = {}
    for path, names in ((sp, SAMPLE_ARRAYS), (mp, SUMMARY_ARRAYS + GATE_ARRAYS + FLAG_ARRAYS)):
        if not os.path.isfile(path):
            raise VerifyRefused("bank %s: the evaluation's file %s is absent" % (bank, path))
        with np.load(path, allow_pickle=False) as z:
            for k in names:
                if k not in z.files:
                    raise VerifyRefused("bank %s: %s holds no array %r" % (bank, path, k))
                out[k] = z[k]
    return out


def _open_project(bundle, cfg_path, ecfg_path, name):
    """The evaluated project's side of the comparison: (P, ecfg, view, e,
    prov), after the refusals of SPEC block 47 that need no inference."""
    from sbitool import export as EXPORT
    P = EXPORT._pipeline()
    try:
        ecfg = P.ev_rec_config.load(ecfg_path)
        view = P.ev_rec_plan.load_view(cfg_path, ecfg)
        e = P.ev_rec_plan.find(view, name)
    except P.ev_plan.RefuseError as exc:
        raise VerifyRefused(str(exc))
    why = P.ev_rec_metrics.sample_left_out(ecfg, e)
    if why is not None:
        raise VerifyRefused("ensemble %s: its sample is absent or not current: %s" % (name, why))
    sample_json = P.ev_rec_plan.sample_json(ecfg, e.stem)
    why = EXPORT._metrics_left_out(P, ecfg, e, sample_json)
    if why is not None:
        raise VerifyRefused("ensemble %s: its metrics are absent or not current: %s"
                            % (name, why))
    man = bundle.manifest
    sha = EXPORT._sha256_file(sample_json)
    if man["provenance"]["sample_json_sha256"] != sha:
        raise VerifyRefused(
            "the bundle %s was not exported from this evaluation of ensemble %s: its "
            "provenance.sample_json_sha256 is %s, the sha256 of %s is %s"
            % (bundle.path, name, man["provenance"]["sample_json_sha256"], sample_json, sha))
    with open(sample_json) as fh:
        prov = json.load(fh)
    if int(man["sample"]["n_post"]) != int(prov["n_post"]):
        raise VerifyRefused("the bundle's sample.n_post is %s, the evaluation's n_post is %s: "
                            "the two cannot be compared"
                            % (man["sample"]["n_post"], prov["n_post"]))
    if int(man["sample"]["batch_rows"]) != int(ecfg.batch_rows):
        raise VerifyRefused("the bundle's sample.batch_rows is %s, the evaluation's batch_rows "
                            "is %s: the draws depend on it (D-179), the two cannot be compared"
                            % (man["sample"]["batch_rows"], ecfg.batch_rows))
    return P, ecfg, view, e, prov


# ---------------------------------------------------------------- the verification

def run(bundle, cfg_path, ecfg_path, name, banks=None, max_rows=None, log=print):
    """Verifies `bundle` (a sbitool.bundle.Bundle) against the evaluation of
    ensemble `name` of the project at `cfg_path` under `ecfg_path` (SPEC
    block 47).

    banks     None (every recording bank of the sample's `seeds`, sorted),
              or a list of bank names
    max_rows  None, or N: only the first floor(N / batch_rows) batch_rows
              windows of each bank (exact: a member's random numbers are
              drawn batch after batch from one seed)

    Returns the record (RECORD_KEYS). VerifyRefused before any inference
    for what cannot be compared."""
    import torch
    from sbitool import infer as INFER

    t0 = time.time()
    P, ecfg, view, e, prov = _open_project(bundle, cfg_path, ecfg_path, name)
    man = bundle.manifest
    batch = int(man["sample"]["batch_rows"])
    seeds = prov["seeds"]
    have = sorted(prov["banks"])          # the ensemble's recording banks (block 35)
    if banks is None:
        banks = have
    else:
        banks = list(banks)
        unknown = [r for r in banks if r not in have]
        if unknown:
            raise VerifyRefused("the sample of ensemble %s has no recording bank %s (it has: %s)"
                                % (name, ", ".join(repr(r) for r in unknown), ", ".join(have)))
    if not banks:
        raise VerifyRefused("the sample of ensemble %s has no recording bank" % name)
    n_first = None
    if max_rows is not None:
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
            raise VerifyRefused("--max-rows must be an integer >= 1, got %r" % (max_rows,))
        n_first = (max_rows // batch) * batch
        if n_first == 0:
            raise VerifyRefused("--max-rows %d is less than one batch of %d windows: nothing "
                                "can be compared exactly" % (max_rows, batch))

    threads = int(prov["torch_threads"])
    before = torch.get_num_threads()
    torch.set_num_threads(max(1, threads))
    record_banks = {}
    try:
        for r in banks:
            evaluation = _load_evaluation(ecfg, e.stem, r)
            try:
                x = np.asarray(P.ev_rec_data.recording(view, r)["x"])
            except P.ev_plan.RefuseError as exc:
                raise VerifyRefused("bank %s: %s" % (r, exc))
            n = int(evaluation["draws"].shape[0])
            if int(x.shape[0]) != n:
                raise VerifyRefused("bank %s has %d windows, the evaluation's files hold %d: "
                                    "the bank changed since the evaluation"
                                    % (r, x.shape[0], n))
            n_cmp = n if n_first is None else min(n_first, n)
            if n_first is not None and n_cmp < n:
                x = x[:n_cmp]
            log("bank %s: %d of %d window(s), seed %d, %d draw(s) each ..."
                % (r, n_cmp, n, int(seeds[r]), int(prov["n_post"])))
            try:
                res = INFER.run(bundle, np.ascontiguousarray(x), seed=int(seeds[r]),
                                log=lambda *a, **k: None)
            except INFER.InferRefused as exc:
                raise VerifyRefused("bank %s: the inference is refused: %s" % (r, exc))
            arrays = compare(res["arrays"], evaluation, None if n_cmp == n else n_cmp)
            record_banks[r] = {"n": n, "n_compared": n_cmp, "seed": int(seeds[r]),
                               "identical": all(a["equal"] for a in arrays.values()),
                               "arrays": arrays}
    finally:
        torch.set_num_threads(before)

    record = {
        "schema": VERIFY_SCHEMA,
        "bundle": {"name": man["name"], "sha256": bundle.sha256},
        "project": man["provenance"]["project"], "ensemble": e.name, "stem": e.stem,
        "n_post": int(prov["n_post"]), "batch_rows": batch, "torch_threads": threads,
        "max_rows": max_rows, "banks": record_banks,
        "identical": all(b["identical"] for b in record_banks.values()),
        "tolerance_gate": GATE_TOL, "software": INFER._software(),
        "seconds": round(time.time() - t0, 1)}
    assert tuple(sorted(record)) == tuple(sorted(RECORD_KEYS))
    return record


def differences(record):
    """The lines of a failed verification: one per bank and array that is
    not equal, with its largest absolute difference."""
    lines = []
    for r in sorted(record["banks"]):
        for name in COMPARED:
            a = record["banks"][r]["arrays"][name]
            if not a["equal"]:
                lines.append("bank %s: %s differs (largest absolute difference %.3g; %s %s)"
                             % (r, name, a["max_abs_diff"], a["dtype"], tuple(a["shape"])))
    return lines


def write(record, out_dir):
    """Writes the record as <out_dir>/VERIFY.json (sorted keys, ASCII),
    through a temporary name; creates the folder. Returns the path. A
    max_abs_diff that is not finite is written as the string "inf" (JSON
    has no token for it)."""
    def jsonable(v):
        if isinstance(v, dict):
            return dict((k, jsonable(x)) for k, x in v.items())
        if isinstance(v, float) and not np.isfinite(v):
            return "inf"
        return v
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, JSON_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii") as fh:
        json.dump(jsonable(record), fh, sort_keys=True, indent=1, ensure_ascii=True,
                  allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- the command

def _parser():
    ap = argparse.ArgumentParser(
        prog="python3 -m sbitool verify",
        description="A bundle's inference on the recording banks of the evaluated project it "
                    "was exported from, compared with the evaluation's own files.")
    ap.add_argument("--bundle", required=True, help="the .sbibundle file")
    ap.add_argument("--config", required=True, help="the project's configuration, <tag>.json")
    ap.add_argument("--eval-config", required=True,
                    help="the evaluation's configuration, <tag>_rec_eval.json")
    ap.add_argument("--ensemble", required=True, help="the ensemble's name in the project")
    ap.add_argument("--bank", action="append", default=None,
                    help="a recording bank to compare (repeatable; default: all)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="compare only the first floor(N / batch_rows) batch_rows windows of "
                         "each bank (exact)")
    ap.add_argument("--out", default=None, help="a folder for VERIFY.json")
    return ap


def main(argv=None):
    """`python3 -m sbitool verify ...` (the module's docstring): 0, 1
    (EXIT_FAILED), EXIT_ARGS, EXIT_REFUSE."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else EXIT_ARGS
    if args.max_rows is not None and args.max_rows < 1:
        sys.stderr.write("sbitool verify: --max-rows must be >= 1\n")
        return EXIT_ARGS
    from sbitool import bundle as bundle_mod
    try:
        try:
            b = bundle_mod.read(args.bundle)
        except OSError as exc:
            raise bundle_mod.BundleError("the bundle %s cannot be read: %s" % (args.bundle, exc))
        record = run(b, args.config, args.eval_config, args.ensemble, banks=args.bank,
                     max_rows=args.max_rows)
    except (VerifyRefused, bundle_mod.BundleError, _sources.SourceError) as exc:
        sys.stderr.write("REFUSED: %s\n" % " ".join(str(exc).split()))
        return EXIT_REFUSE
    if args.out is not None:
        print("written: %s" % write(record, args.out))
    n_total = sum(bk["n_compared"] for bk in record["banks"].values())
    for r in sorted(record["banks"]):
        bk = record["banks"][r]
        worst = max(a["max_abs_diff"] for a in bk["arrays"].values())
        print("bank %s: %d of %d window(s) compared, %d array(s), largest absolute difference "
              "%.3g: %s" % (r, bk["n_compared"], bk["n"], len(bk["arrays"]), worst,
                            "identical" if bk["identical"] else "DIFFERENT"))
    if record["identical"]:
        print("VERIFY OK: %d bank(s), %d window(s): the bundle reproduces the evaluation (the "
              "sample's arrays and the summaries bit for bit, the gate within %g)"
              % (len(record["banks"]), n_total, GATE_TOL))
        return 0
    lines = differences(record)
    for ln in lines:
        print(ln)
    print("VERIFY FAILED: %d array(s) differ over %d bank(s) (torch threads %d, as the "
          "evaluation's; another machine or another number of threads can change the last "
          "digits)" % (len(lines), len(record["banks"]), record["torch_threads"]))
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
