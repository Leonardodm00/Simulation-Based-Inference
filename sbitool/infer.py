"""Block 45: inference from a bundle (specs/SPEC.md sec. 2.7, "An
inference", and block 45 with its "Contract of wave C").

CONTRACT (wave C): the names, signatures, ARRAYS and JSON_KEYS below are
fixed by specs/SPEC.md; the bodies are block 45's. The numbers are the
evaluation's own (blocks 35 and 36), through sbitool._sources: this module
holds no estimator of its own.

Pure ASCII, LF only.
"""

import argparse
import json
import os
import platform
import sys
import types

import numpy as np

from sbitool import _sources  # noqa: F401  (the evaluation's modules come through it)

POSTERIOR_SCHEMA = "sbitool_posterior/1"
EXIT_ARGS, EXIT_REFUSE = 2, 3

# posterior.npz: name -> (shape, dtype). n windows, n_post draws per
# window, p parameters, M members, d_z the embedding's size. Beside these:
# `rows` (n,) int64 = arange(n), and the ids given to run(), as given.
ARRAYS = {
    "draws": ("(n, n_post, p)", "float32"),          # block 35's coordinate, bounds not applied
    "draw_member": ("(n_post,)", "int64"),
    "logq_draws": ("(n, n_post)", "float32"),        # log q_E at each draw (nats)
    "logq_draws_own": ("(n, n_post)", "float32"),    # log q_m of the member that drew it (nats)
    "z": ("(M, n, d_z)", "float32"),
    "mhat": ("(n, p)", "float32"), "sigma": ("(n, p)", "float32"),
    "q05": ("(n, p)", "float32"), "q50": ("(n, p)", "float32"), "q95": ("(n, p)", "float32"),
    "contraction": ("(n, p)", "float32"),
    "gate_t": ("(M, n)", "float64"), "gate_r": ("(M, n)", "float64"),
    "gate_p_member": ("(M, n)", "float64"),
    "gate_T": ("(n,)", "float64"), "gate_p": ("(n,)", "float64"), "gate_flag": ("(n,)", "bool"),
    "mix_mean": ("(p,)", "float64"), "mix_sd": ("(p,)", "float64"),
    "mix_q05": ("(p,)", "float64"), "mix_q50": ("(p,)", "float64"), "mix_q95": ("(p,)", "float64"),
}
# posterior.json: exactly these keys
JSON_KEYS = ("schema", "bundle", "n", "n_post", "M", "p", "seed", "batch_rows", "parameters",
             "gate", "ids", "mixture_note", "software")
JSON_GATE_KEYS = ("alpha", "k_gate", "n_null", "size_mean", "size_lo", "size_hi", "level",
                  "n_flagged", "share_flagged")
NPZ_NAME, JSON_NAME = "posterior.npz", "posterior.json"


MIXTURE_NOTE = ("the mixture over the recording's windows (R4): where the windows put their "
                "mass, every window weighing the same; it is not the posterior of the "
                "recording's parameter given its windows, and its width does not shrink with "
                "the number of windows (D-160)")
_WINDOWS_IDS = ("subregion", "window_idx", "t_start_s")    # block 44's windows.npz
_TMP = ".tmp"


class InferRefused(ValueError):
    """An inference that is not made; the message says why."""


# ---------------------------------------------------------------- the sources

def _evaluation():
    """The evaluation's modules (block 19's sampler, block 36's quantities,
    block 34's gate), through _sources (model and evaluation groups)."""
    _sources.use("model", "evaluation")
    import _evalpath  # noqa: F401  (the pipeline's config and stages on sys.path)
    import ev_gate
    import ev_rec_metrics
    import ev_sample
    return types.SimpleNamespace(ev_sample=ev_sample, ev_rec_metrics=ev_rec_metrics,
                                 ev_gate=ev_gate)


def _software():
    """The versions of this run, under the manifest's software keys."""
    import safetensors
    import sbi
    import torch
    import zuko
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "sbi": str(sbi.__version__), "zuko": str(zuko.__version__),
            "numpy": str(np.__version__), "safetensors": str(safetensors.__version__)}


# ---------------------------------------------------------------- the checks

def _int_or(value, default, name, minimum):
    """`default` for None; otherwise an int >= minimum, InferRefused else."""
    v = default if value is None else value
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise InferRefused("%s must be an integer, got %r" % (name, v))
    v = int(v)
    if v < minimum:
        raise InferRefused("%s must be >= %d, got %d" % (name, minimum, v))
    return v


def _checked_windows(windows, W):
    x = np.asarray(windows)
    if x.dtype != np.float32:
        raise InferRefused("the windows must be float32, got %s" % (x.dtype,))
    if x.ndim != 2 or x.shape[1] != W:
        raise InferRefused("the windows must be (n, W) with W = %d, the bundle's window length; "
                           "got shape %r" % (W, x.shape))
    if x.shape[0] < 1:
        raise InferRefused("no window: n = 0")
    bad = np.flatnonzero(~np.isfinite(x).all(axis=1))
    if bad.size:
        raise InferRefused("%d window(s) hold a non-finite value (the first: row %d)"
                           % (bad.size, int(bad[0])))
    return x


def _checked_ids(ids, n):
    if ids is None:
        return {}
    if not isinstance(ids, dict):
        raise InferRefused("ids must be None or a dict of (n,) arrays, got %s"
                           % type(ids).__name__)
    out = {}
    for name, v in ids.items():
        if not isinstance(name, str) or not name:
            raise InferRefused("an id's name must be a non-empty string, got %r" % (name,))
        if name in ARRAYS or name == "rows":
            raise InferRefused("the id %r has the name of an array of the result" % name)
        a = np.asarray(v)
        if a.shape != (n,):
            raise InferRefused("the id %r has shape %r, not (n,) = (%d,)" % (name, a.shape, n))
        out[name] = a
    return out


# ---------------------------------------------------------------- the inference

def run(bundle, windows, ids=None, n_post=None, seed=None, batch=None, log=print):
    """One inference (SPEC block 45, "Contract of wave C").

    bundle   a sbitool.bundle.Bundle
    windows  (n, W) float32, finite, n >= 1, W the manifest's
    ids      None, or {name: (n,) array}, written beside the arrays as given
    n_post, seed, batch
             None: the bundle's sample.n_post, sample.seed, sample.batch_rows.
             n_post a multiple of M; member k draws from seed + k.

    Returns {"arrays": {name: numpy array}, "json": dict}: ARRAYS, `rows`
    and the ids; JSON_KEYS. InferRefused for windows of another shape or
    dtype or with a non-finite value, n = 0, n_post not a positive multiple
    of M, ids of another length or under a name of ARRAYS or `rows`."""
    man = bundle.manifest
    M, W, p = int(man["M"]), int(man["W"]), int(man["p"])
    smp, gman = man["sample"], man["gate"]
    n_post = _int_or(n_post, smp["n_post"], "n_post", 1)
    if n_post % M:
        raise InferRefused("n_post %d is not a multiple of the bundle's %d members" % (n_post, M))
    seed = _int_or(seed, smp["seed"], "seed", 0)
    batch = _int_or(batch, smp["batch_rows"], "batch", 1)
    x = _checked_windows(windows, W)
    n = int(x.shape[0])
    ids = _checked_ids(ids, n)
    from sbitool import model
    P = _evaluation()
    models = model.load(bundle)
    # block 35's call for a recording bank: no reference parameter, none in the box
    s = P.ev_sample.sample_set(models, x, np.zeros((n, p), dtype=np.float64),
                               np.zeros(n, dtype=bool), n_post, seed, batch, log)
    arrays = dict((k, s[k]) for k in ("draws", "draw_member", "logq_draws", "logq_draws_own",
                                      "z"))
    # block 36's per-window summaries, written as float32 as block 36 writes them
    per = P.ev_rec_metrics.window_summary(arrays["draws"])
    for k in ("mhat", "sigma", "q05", "q50", "q95", "contraction"):
        arrays[k] = per[k].astype(np.float32)
    # the gate against the bundle's own sets
    g = bundle.gate_sets()
    gt = P.ev_gate.gate(g["z_ref"], g["z_rank"], g["z_null"], arrays["z"], list(gman["factors"]))
    arrays.update({"gate_t": gt["t"], "gate_r": gt["r"], "gate_p_member": gt["p_member"],
                   "gate_T": gt["T"], "gate_p": gt["p"]})
    arrays["gate_flag"] = np.asarray(gt["p"] < float(gman["alpha"]), dtype=bool)
    # (R4): the mixture over the windows, all of them one group
    ws = P.ev_rec_metrics.well_summary(arrays["draws"], np.zeros(n, dtype=np.int64))
    for k in ("mean", "sd", "q05", "q50", "q95"):
        arrays["mix_" + k] = np.asarray(ws["well_" + k][0], dtype=np.float64)
    arrays["rows"] = np.arange(n, dtype=np.int64)
    arrays.update(ids)
    n_flagged = int(arrays["gate_flag"].sum())
    gate = dict((k, gman[k]) for k in JSON_GATE_KEYS[:7])
    gate.update(n_flagged=n_flagged, share_flagged=n_flagged / float(n))
    js = {"schema": POSTERIOR_SCHEMA,
          "bundle": {"name": man["name"], "sha256": getattr(bundle, "sha256", None)},
          "n": n, "n_post": n_post, "M": M, "p": p, "seed": seed, "batch_rows": batch,
          "parameters": json.loads(json.dumps(man["parameters"])), "gate": gate,
          "ids": list(ids), "mixture_note": MIXTURE_NOTE, "software": _software()}
    assert tuple(js) == JSON_KEYS
    return {"arrays": arrays, "json": js}


def _ordered(arrays):
    """ARRAYS' order, then rows, then the ids in their order; each array
    contiguous (its values and dtype unchanged)."""
    names = [k for k in ARRAYS] + ["rows"] + [k for k in arrays if k not in ARRAYS
                                               and k != "rows"]
    return dict((k, np.ascontiguousarray(arrays[k])) for k in names)


def write(result, out_dir):
    """Writes run()'s result as <out_dir>/posterior.npz (ev_sample.write_npz)
    and <out_dir>/posterior.json (sorted keys, ASCII), each through a
    temporary name; creates the folder. Returns (npz path, json path)."""
    P = _evaluation()
    text = json.dumps(result["json"], indent=1, sort_keys=True, ensure_ascii=True,
                      allow_nan=False) + "\n"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, NPZ_NAME)
    jsn = os.path.join(out_dir, JSON_NAME)
    P.ev_sample.write_npz(npz, _ordered(result["arrays"]))
    tmp = jsn + _TMP
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, jsn)
    return npz, jsn


# ---------------------------------------------------------------- the command

def _parser():
    from sbitool import windows as WIN
    ap = argparse.ArgumentParser(
        prog="python3 -m sbitool infer",
        description="Posterior draws, summaries and the gate of a bundle on windows "
                    "(specs/SPEC.md block 45).")
    ap.add_argument("--bundle", required=True, help="the .sbibundle file")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--windows", help="a windows.npz written by `python3 -m sbitool windows`")
    src.add_argument("--recording", help="a recording's file or folder (with --adapter)")
    ap.add_argument("--adapter", help="the recording adapter (ptrain, spikeinterface)")
    ap.add_argument("--option", action="append", default=[], type=WIN.parse_option,
                    metavar="KEY=VALUE",
                    help="an adapter option; VALUE parsed as JSON when it parses (repeatable)")
    ap.add_argument("--out", required=True, help="the output folder")
    ap.add_argument("--n-post", type=int, default=None,
                    help="draws per window (default: the bundle's sample.n_post)")
    ap.add_argument("--seed", type=int, default=None,
                    help="the seed; member k draws from seed + k (default: the bundle's)")
    return ap


def _read_windows(path):
    """(windows, ids) of block 44's windows.npz: its `windows`, and as the
    ids those of subregion, window_idx, t_start_s that it holds."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if "windows" not in z.files:
                raise InferRefused("%s holds no array `windows` (it holds %s)"
                                   % (path, ", ".join(z.files) or "nothing"))
            x = z["windows"]
            ids = dict((k, z[k]) for k in _WINDOWS_IDS if k in z.files)
    except (OSError, ValueError) as exc:
        if isinstance(exc, InferRefused):
            raise
        raise InferRefused("%s cannot be read as a windows.npz: %s" % (path, exc))
    return x, ids


def main(argv=None):
    """`python3 -m sbitool infer --bundle FILE (--windows windows.npz |
    --recording PATH --adapter NAME [--option KEY=VALUE ...]) --out DIR
    [--n-post N] [--seed S]`: 0 with `INFER OK: <n> window(s), <k> flagged
    by the gate`; EXIT_ARGS for the arguments (--option detection
    included); EXIT_REFUSE with one `REFUSED:` line, nothing written."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = _parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else EXIT_ARGS
    if args.recording is not None and not args.adapter:
        sys.stderr.write("sbitool infer: --recording needs --adapter\n")
        return EXIT_ARGS
    if args.windows is not None and (args.adapter or args.option):
        sys.stderr.write("sbitool infer: --adapter and --option go with --recording, not "
                         "--windows\n")
        return EXIT_ARGS
    options = dict(args.option)
    if "detection" in options:
        sys.stderr.write("sbitool infer: --option detection is refused: the detection is the "
                         "bundle's (its extraction.detection), never the user's\n")
        return EXIT_ARGS
    from sbitool import bundle as bundle_mod
    from sbitool import recordings
    from sbitool import windows as WIN
    refusals = (InferRefused, bundle_mod.BundleError, WIN.WindowsError,
                recordings.RecordingError, _sources.SourceError)
    try:
        try:
            b = bundle_mod.read(args.bundle)
        except OSError as exc:
            raise bundle_mod.BundleError("the bundle %s cannot be read: %s" % (args.bundle, exc))
        if args.windows is not None:
            x, ids = _read_windows(args.windows)
        else:
            try:
                r = WIN.from_file(b.manifest, args.recording, args.adapter, options)
            except TypeError as exc:   # an option the adapter does not take: the arguments
                sys.stderr.write("sbitool infer: adapter %r: %s\n" % (args.adapter, exc))
                return EXIT_ARGS
            except OSError as exc:
                raise recordings.RecordingError("the recording %s cannot be read: %s"
                                                % (args.recording, exc))
            x, ids = r["windows"], r["ids"]
        result = run(b, x, ids=ids, n_post=args.n_post, seed=args.seed)
    except refusals as exc:
        sys.stderr.write("REFUSED: %s\n" % "; ".join(str(exc).splitlines()))
        return EXIT_REFUSE
    write(result, args.out)
    js = result["json"]
    print("INFER OK: %d window(s), %d flagged by the gate" % (js["n"], js["gate"]["n_flagged"]),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
