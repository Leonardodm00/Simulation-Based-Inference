"""Block 44: from a recording to the windows a bundle's networks take, by
the extractor's own rule (specs/SPEC.md sec. 2.7, "From a recording to
windows", and block 44).

CONTRACT (wave B): the signatures below are fixed.

The rule is the extractor's, imported through sbitool._sources and never
copied: this module only builds the extractor's own input objects (its
PtrainInventory and its coords dict) from a Recording and calls its
functions -- mean_firing_rates, partition_subregions, subregion_ifrs (which
calls compute_ifr_trace of hpc/dsn/generate_burst_data.py) -- and the real
bank's windows_of (export_windows/bank_real.py). Nothing of the rule is
computed here.

Pure ASCII, LF only.
"""

import argparse
import json
import os
import sys

import numpy as np

from sbitool import _sources, schema

EXIT_ARGS, EXIT_REFUSE = 2, 3
PITCH_TOL_UM = 1.0


class WindowsError(ValueError):
    """The recording cannot be cut into the bundle's windows: the message
    names the two values that differ, or what is missing."""


# ------------------------------------------------------------ the sources

def _cse():
    """The extractor's channel_subset_extraction, with the IFR primitive's
    folder (hpc/dsn) put on sys.path first (sec. 2.7, "The sources")."""
    _sources.use("extractor", "ifr")
    import channel_subset_extraction
    return channel_subset_extraction


def _windows_of():
    """The real bank's windows_of(K, W) (export_windows/bank_real.py)."""
    _sources.use("extractor")
    import bank_real
    return bank_real.windows_of


# ------------------------------------------------------------ the checks

def _checked_extraction(extraction):
    probs = schema.validate_extraction(extraction)
    if probs:
        raise WindowsError("the extraction is not valid: %s" % "; ".join(probs))
    return extraction


def _check_recording(recording):
    probs = recording.check()
    if probs:
        raise WindowsError("the recording %r is not valid: %s"
                           % (recording.name, "; ".join(probs)))


def _check_rate(recording, extraction):
    fs_rec, fs_ext = float(recording.fs_raw_hz), float(extraction["fs_raw_hz"])
    if fs_rec != fs_ext:
        raise WindowsError("the recording %r has fs_raw_hz %r, the extraction's fs_raw_hz is %r: "
                           "a bundle is not applied to a recording at another rate (Q-T2)"
                           % (recording.name, fs_rec, fs_ext))


def _placement(recording, extraction):
    """(coords, key): coords {electrode id: (row, col)} as grid_of returns
    it; key {electrode id: the partition's key}, the electrode's linear index
    on the grid (amendment 1 of wave B): the id itself for a recording with a
    grid, layout.as_grid's linear index for one with positions only."""
    from sbitool.recordings import layout
    _checked_extraction(extraction)
    _check_recording(recording)
    _check_rate(recording, extraction)
    if recording.grid is not None:
        width, base, pitch = recording.grid
        for name, rec_v, ext_v in (("grid_width", width, extraction["grid_width"]),
                                   ("index_base", base, extraction["index_base"])):
            if int(rec_v) != int(ext_v):
                raise WindowsError("the recording %r has %s %r, the extraction's %s is %r"
                                   % (recording.name, name, rec_v, name, ext_v))
        ext_pitch = extraction["pitch_um"]
        if pitch is not None and ext_pitch is not None and float(pitch) != float(ext_pitch):
            raise WindowsError("the recording %r has pitch_um %r, the extraction's pitch_um is %r"
                               % (recording.name, float(pitch), float(ext_pitch)))
        try:
            coords = layout.grid(recording)
        except layout.LayoutError as exc:
            raise WindowsError(str(exc)) from exc
        return coords, {int(e): int(e) for e in coords}
    if recording.positions_um is not None:
        try:
            width, pitch, lin = layout.as_grid(recording.positions_um)
        except layout.LayoutError as exc:
            raise WindowsError("the recording %r: its positions are not a regular square grid: %s"
                               % (recording.name, exc)) from exc
        ext_pitch = extraction["pitch_um"]
        if ext_pitch is not None and not abs(float(pitch) - float(ext_pitch)) <= PITCH_TOL_UM:
            raise WindowsError("the recording %r has a pitch of %.6g um (from its positions), the "
                               "extraction's pitch_um is %r (tolerance %g um)"
                               % (recording.name, pitch, float(ext_pitch), PITCH_TOL_UM))
        # the grid's width is the recording's own and is not compared (block 44's Rule)
        coords = {int(e): (int(k) // int(width), int(k) % int(width)) for e, k in lin.items()}
        return coords, {int(e): int(k) for e, k in lin.items()}
    raise WindowsError("the recording %r has neither a grid nor electrode positions: its "
                       "electrodes cannot be placed on the extraction's grid" % (recording.name,))


def grid_of(recording, extraction):
    """{electrode id: (row, col)} of the recording on the extraction's
    grid, after the comparison block 44's Rule states (the recording's own
    grid, or layout.as_grid of its positions); WindowsError naming both
    values on a mismatch of the rate, the grid or the pitch."""
    return _placement(recording, extraction)[0]


def _inventory(recording, index_base, key=None):
    """The extractor's PtrainInventory of a Recording: its spikes, n_samples,
    fs_raw, T_rec = n_samples / fs_raw (as load_ptrain_folder computes it);
    keyed by key[e] when `key` is given, by the electrode id otherwise."""
    cse = _cse()
    key = key or {int(e): int(e) for e in recording.spikes}
    spikes = {key[int(e)]: np.asarray(recording.spikes[e], dtype=np.int64)
              for e in sorted(recording.spikes, key=lambda e: key[int(e)])}
    return cse.PtrainInventory(spikes=spikes, n_samples=int(recording.n_samples),
                               fs_raw=float(recording.fs_raw_hz),
                               T_rec=float(recording.n_samples) / float(recording.fs_raw_hz),
                               index_base=int(index_base))


def _base_of(recording):
    return int(recording.grid[1]) if recording.grid is not None else 0


# ------------------------------------------------------------ the steps

def subregions(recording, extraction):
    """The extractor's Subregion objects for this recording
    (partition_subregions on grid_of's rows and columns and the
    electrodes' mean firing rates), keyed by the electrodes' linear index
    on the grid (its tie-break) and returned with the recording's electrode
    ids as members and centre (amendment 1 of wave B)."""
    import dataclasses
    coords_id, key = _placement(recording, extraction)
    cse = _cse()
    id_of = {k: e for e, k in key.items()}
    coords = {key[e]: rc for e, rc in coords_id.items()}
    inv = _inventory(recording, _base_of(recording), key)
    mfrs = cse.mean_firing_rates(inv, inv.T_rec)
    try:
        subs, _discarded = cse.partition_subregions(
            coords, mfrs, n_subsets=int(extraction["n_subsets"]),
            electrodes_per_subset=int(extraction["electrodes_per_subset"]),
            mfr_threshold=float(extraction["mfr_threshold_hz"]))
    except cse.InsufficientElectrodesError as exc:
        raise WindowsError("the recording %r: the extractor refuses the partition: %s"
                           % (recording.name, exc)) from exc
    return [dataclasses.replace(s, center=id_of[int(s.center)],
                                members=tuple(id_of[int(m)] for m in s.members)) for s in subs]


def traces(recording, subs, extraction):
    """(x, fs_ifr_hz): x (n_subregions, K) float32 [Hz per electrode], the
    extractor's subregion_ifrs."""
    _checked_extraction(extraction)
    _check_recording(recording)
    _check_rate(recording, extraction)
    if not subs:
        raise WindowsError("no subregion: nothing to extract")
    unknown = sorted({int(m) for s in subs for m in s.members} - set(int(e) for e in recording.spikes))
    if unknown:
        raise WindowsError("the subregions name electrodes the recording %r does not hold: %s"
                           % (recording.name, unknown[:10]))
    cse = _cse()
    inv = _inventory(recording, _base_of(recording))
    x, fs_ifr = cse.subregion_ifrs(inv, list(subs), w_size=float(extraction["w_size_s"]),
                                   gaussian_window=float(extraction["gaussian_window_s"]))
    return x, float(fs_ifr)


def cut(x, W, fs_ifr_hz):
    """(windows, ids): windows (n, W) float32, the rows of every subregion
    in subregion order, each subregion's windows by windows_of(K, W); ids a
    dict of (n,) arrays `subregion` (int64), `window_idx` (int64, within
    the subregion), `t_start_s` (float64, window_idx W / fs_ifr_hz)."""
    if isinstance(W, bool) or not isinstance(W, (int, np.integer)) or int(W) < 1:
        raise WindowsError("W must be an integer >= 1, got %r" % (W,))
    W = int(W)
    if isinstance(fs_ifr_hz, bool) or not np.isfinite(fs_ifr_hz) or not fs_ifr_hz > 0:
        raise WindowsError("fs_ifr_hz must be a positive finite number, got %r" % (fs_ifr_hz,))
    x = np.asarray(x)
    if x.ndim != 2:
        raise WindowsError("x must be (n_subregions, K), got shape %r" % (x.shape,))
    windows_of = _windows_of()
    rows, sub, win = [], [], []
    for s in range(x.shape[0]):
        for j, start in enumerate(windows_of(int(x.shape[1]), W)):
            rows.append(np.asarray(x[s, start:start + W], dtype=np.float32))
            sub.append(s)
            win.append(j)
    windows = (np.stack(rows).astype(np.float32) if rows
               else np.zeros((0, W), dtype=np.float32))
    window_idx = np.asarray(win, dtype=np.int64)
    ids = {"subregion": np.asarray(sub, dtype=np.int64), "window_idx": window_idx,
           "t_start_s": window_idx.astype(np.float64) * float(W) / float(fs_ifr_hz)}
    return windows, ids


def extract(recording, extraction, W):
    """(windows, ids, subs): the three steps above."""
    _checked_extraction(extraction)
    subs = subregions(recording, extraction)
    x, fs_ifr = traces(recording, subs, extraction)
    windows, ids = cut(x, W, fs_ifr)
    return windows, ids, subs


# ------------------------------------------------------------ the command

def _option(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("an option is key=value, got %r" % (text,))
    key, val = text.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("an option's key is empty in %r" % (text,))
    try:
        v = json.loads(val)
    except ValueError:
        return key, val
    # JSON has no tuple: a list (an adapter's grid, (grid_width, index_base,
    # pitch_um)) is passed as a tuple, the form a Recording holds
    return key, tuple(v) if isinstance(v, list) else v


def _parser():
    p = argparse.ArgumentParser(
        prog="python3 -m sbitool windows",
        description="Cut a recording into the windows of a bundle, by the extractor's rule.")
    p.add_argument("--bundle", required=True, help="the .sbibundle file")
    p.add_argument("--recording", required=True, help="the recording's file or folder")
    p.add_argument("--adapter", required=True, help="the recording adapter (ptrain, spikeinterface)")
    p.add_argument("--option", action="append", default=[], type=_option, metavar="KEY=VALUE",
                   help="an adapter option; VALUE parsed as JSON when it parses (repeatable)")
    p.add_argument("--out", required=True, help="the output windows.npz")
    return p


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(u) for k, u in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(u) for u in v]
    if isinstance(v, np.generic):
        return v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _read_recording(path, adapter, options, detection):
    """(recording, detection_applied) for the command (amendment 2 of wave B).
    The bundle's detection goes to the spikeinterface adapter only, and only
    when the file is raw voltage: the file is first read without it, which
    the adapter accepts for spike trains (a sorting) and refuses for raw
    voltage; then, on that refusal, with it. detection_applied is the
    bundle's detection when it was applied, None for spike trains. Without a
    detection (or for another adapter) one read, the adapter's refusals
    standing. RecordingError naming both reads when neither succeeds."""
    from sbitool import recordings
    if adapter != "spikeinterface" or detection is None:
        return recordings.read(path, adapter, **options), None
    try:
        return recordings.read(path, adapter, **options), None
    except recordings.RecordingError as first:
        try:
            rec = recordings.read(path, adapter, detection=dict(detection), **options)
        except recordings.RecordingError as second:
            raise recordings.RecordingError(
                "neither as spike trains (%s) nor as raw voltage with the bundle's detection "
                "(%s)" % (first, second)) from second
    if rec.source.get("kind") != "recording":
        raise recordings.RecordingError("the adapter read %r with the bundle's detection but "
                                        "reports kind %r, not raw voltage"
                                        % (path, rec.source.get("kind")))
    return rec, dict(detection)


def main(argv=None):
    """`python3 -m sbitool windows --bundle FILE --recording PATH --adapter
    NAME [--option key=value ...] --out windows.npz`: 0 with WINDOWS OK:
    ..., EXIT_ARGS, EXIT_REFUSE."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else EXIT_ARGS
    options = dict(args.option)
    if "detection" in options:
        sys.stderr.write("sbitool windows: --option detection is refused: the detection is the "
                         "bundle's (its extraction.detection), never the user's\n")
        return EXIT_ARGS
    from sbitool import bundle as bundle_mod
    from sbitool import recordings
    try:
        b = bundle_mod.read(args.bundle)
    except (bundle_mod.BundleError, OSError) as exc:
        sys.stderr.write("REFUSED: the bundle %s: %s\n" % (args.bundle, exc))
        return EXIT_REFUSE
    manifest = b.manifest
    extraction = manifest.get("extraction")
    if extraction is None:
        sys.stderr.write("REFUSED: the bundle %s has no extraction (null): it accepts windows "
                         "only, not a recording\n" % args.bundle)
        return EXIT_REFUSE
    W = manifest.get("W")
    detection = extraction.get("detection")
    try:
        recording, applied = _read_recording(args.recording, args.adapter, options, detection)
    except TypeError as exc:    # an option the adapter does not take: the arguments
        sys.stderr.write("sbitool windows: adapter %r: %s\n" % (args.adapter, exc))
        return EXIT_ARGS
    except (recordings.RecordingError, _sources.SourceError, OSError) as exc:
        sys.stderr.write("REFUSED: %s\n" % exc)
        return EXIT_REFUSE
    try:
        # extract()'s three steps, called one by one to keep the extractor's fs_ifr
        _checked_extraction(extraction)
        subs = subregions(recording, extraction)
        x, fs_ifr = traces(recording, subs, extraction)
        windows, ids = cut(x, W, fs_ifr)
    except (WindowsError, _sources.SourceError) as exc:
        sys.stderr.write("REFUSED: %s\n" % exc)
        return EXIT_REFUSE
    if windows.shape[0] == 0:
        sys.stderr.write("REFUSED: no window: every trace of the recording %r is shorter than "
                         "one window (K = %d < W = %d samples)\n"
                         % (recording.name, int(x.shape[1]), int(W)))
        return EXIT_REFUSE
    meta = {"recording": {"name": recording.name, "source": _jsonable(recording.source)},
            "extraction": _jsonable(extraction),
            "detection": _jsonable(detection), "detection_applied": _jsonable(applied),
            "subregions": [{"center": int(s.center), "members": [int(m) for m in s.members],
                            "mean_mfr": float(s.mean_mfr), "center_mfr": float(s.center_mfr)}
                           for s in subs]}
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, windows=windows, subregion=ids["subregion"], window_idx=ids["window_idx"],
                 t_start_s=ids["t_start_s"], fs_ifr_hz=np.float64(fs_ifr),
                 meta=np.str_(json.dumps(meta, sort_keys=True)))
    os.replace(tmp, args.out)
    if applied is None:
        print("the spikes of %r arrived as spike trains (adapter %s): they were detected outside "
              "the tool; the bundle's detection (%s) was not applied"
              % (recording.name, args.adapter,
                 "none" if detection is None else json.dumps(detection, sort_keys=True)))
    print("WINDOWS OK: %d window(s) of %d subregion(s)" % (windows.shape[0], len(subs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
