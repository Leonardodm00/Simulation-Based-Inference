"""Block 43: the files SpikeInterface reads as a Recording (specs/SPEC.md
block 43; D-173, D-176). spikeinterface and probeinterface are imported
inside the functions only.

CONTRACT (wave A): the signatures are fixed.

Pure ASCII, LF only.
"""

import math
import os

import numpy as np

from sbitool.interfaces import Recording
from sbitool.recordings import RecordingError, _checked
from sbitool.schema import DETECTION_KEYS

# The seed of the random chunks on which spikeinterface's get_noise_levels
# estimates each channel's MAD-based noise level (detect_peaks' default
# estimate; without a seed the chunks, and so the threshold, change from run
# to run). It fixes how the noise is sampled, not what the threshold means.
NOISE_SEED = 0
# spikeinterface's job_kwargs: one process, no progress bar (a laptop's run;
# the detection does not depend on them).
JOB_KWARGS = {"n_jobs": 1, "progress_bar": False}
# the options si.read takes for a sorting itself (from_sorting's arguments),
# never passed to the reader
_SORTING_OPTIONS = ("n_samples", "positions_um", "grid")


def _as_int(v):
    """v as an int when it is an integer (or a string of decimal digits); None
    otherwise."""
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, str) and v.isascii() and v.isdecimal():
        return int(v)
    return None


def _key_of(k):
    """k as an int when it is an integer (not a bool) or an integer's text
    (ASCII decimal digits, an optional leading minus); None otherwise."""
    if isinstance(k, (bool, np.bool_)):
        return None
    if isinstance(k, (int, np.integer)):
        return int(k)
    if isinstance(k, str) and k.isascii():
        digits = k[1:] if k.startswith("-") else k
        if digits.isdecimal():
            return int(k)
    return None


def _is_finite_number(v):
    """v is an int or a float (not a bool) whose float is finite; an int
    beyond the float range is not."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def _positions_option(positions_um):
    """positions_um as a Recording holds it, {int: (x, y) floats} [um]
    (block 43, fix round of 2026-11-28, the tester's F6): the option may
    arrive as a JSON object, whose keys are strings and whose pairs are
    lists; each key must be an integer or an integer's text and each value
    two finite numbers, or the option is refused naming the key. None stays
    None. Nothing is rescaled: the numbers are taken as um."""
    if positions_um is None:
        return None
    if not isinstance(positions_um, dict):
        raise RecordingError("spikeinterface: positions_um must be an object {electrode id: "
                             "[x, y]} [um], got a %s" % type(positions_um).__name__)
    out = {}
    for k, v in positions_um.items():
        e = _key_of(k)
        if e is None:
            raise RecordingError("spikeinterface: positions_um has the key %r, which is not an "
                                 "integer's text (an electrode id)" % (k,))
        if e in out:
            raise RecordingError("spikeinterface: positions_um names electrode %d twice (the key "
                                 "%r)" % (e, k))
        if (not isinstance(v, (list, tuple)) or len(v) != 2
                or not all(_is_finite_number(u) for u in v)):
            raise RecordingError("spikeinterface: positions_um[%r] is %r, not two finite numbers "
                                 "(x, y) [um]" % (k, v))
        out[e] = (float(v[0]), float(v[1]))
    return out


def from_sorting(sorting, n_samples, positions_um=None, grid=None, name=None, source=None):
    """A SpikeInterface sorting (spike trains) as a Recording: one unit per
    electrode required (the unit's id, or its channel property, an int);
    two units on one electrode are refused. `positions_um` may be a JSON
    object (string keys, list pairs; _positions_option), turned into
    {int: (x, y)} [um] before Recording.check."""
    positions_um = _positions_option(positions_um)
    if n_samples is None:
        raise RecordingError("spikeinterface: n_samples not given; a sorting does not hold the "
                             "recording's length, so it is an argument")
    if sorting.get_num_segments() != 1:
        raise RecordingError("spikeinterface: the sorting has %d segments; a Recording is one "
                             "segment" % sorting.get_num_segments())
    unit_ids = list(sorting.get_unit_ids())
    if not unit_ids:
        raise RecordingError("spikeinterface: the sorting has no unit")
    channel = sorting.get_property("channel")   # None when the sorting has no such property
    spikes, owner = {}, {}
    for i, u in enumerate(unit_ids):
        label = channel[i] if channel is not None else u
        e = _as_int(label)
        if e is None:
            raise RecordingError("spikeinterface: unit %r has the electrode %r (its %s), not an "
                                 "integer" % (u, label,
                                              "channel property" if channel is not None else "id"))
        if e in spikes:
            raise RecordingError("spikeinterface: units %r and %r are on electrode %d; one unit "
                                 "per electrode is required" % (owner[e], u, e))
        train = np.asarray(sorting.get_unit_spike_train(unit_id=u, segment_index=0))
        if not np.issubdtype(train.dtype, np.integer):
            raise RecordingError("spikeinterface: unit %r's spike train is of dtype %s, not "
                                 "integer sample indices" % (u, train.dtype))
        spikes[e] = train.astype(np.int64)
        owner[e] = u
    if name is None:
        name = "sorting"
    src = {"adapter": "spikeinterface", "kind": "sorting", "unit_of_electrode":
           {int(e): repr(u) for e, u in owner.items()}}
    src.update(source or {})
    rec = Recording(name=name, spikes=spikes, fs_raw_hz=float(sorting.get_sampling_frequency()),
                    n_samples=n_samples, positions_um=positions_um, grid=grid, source=src)
    return _checked(rec)


def _positions_of(recording):
    """{electrode id: (x, y)} [um] of the recording's probe, or None when it
    has none."""
    if not recording.has_probe():
        return None
    probe = recording.get_probe()
    scale = {"um": 1.0, "mm": 1000.0}.get(getattr(probe, "si_units", None))
    if scale is None:
        raise RecordingError("spikeinterface: the probe's unit %r is neither um nor mm"
                             % (getattr(probe, "si_units", None),))
    loc = np.asarray(recording.get_channel_locations(), dtype=np.float64)
    if loc.ndim != 2 or loc.shape[1] != 2:
        raise RecordingError("spikeinterface: the probe's positions have shape %r; a Recording "
                             "takes (x, y)" % (loc.shape,))
    out = {}
    for cid, (x, y) in zip(recording.get_channel_ids(), loc):
        out[_as_int(cid)] = (float(x * scale), float(y * scale))
    return out


def from_recording(recording, detection, name=None, source=None):
    """A SpikeInterface raw recording as a Recording: the peaks detected per
    channel by spikeinterface's detect_peaks with exactly the arguments of
    `detection` (schema.DETECTION_KEYS; None is refused: raw voltage needs
    a declared detection); the positions are the probe's [um]."""
    if detection is None:
        raise RecordingError("spikeinterface: a raw recording needs `detection` (the keys %s); "
                             "none was given" % ", ".join(DETECTION_KEYS))
    if not isinstance(detection, dict) or set(detection) != set(DETECTION_KEYS):
        raise RecordingError("spikeinterface: detection must have exactly the keys %s, got %r"
                             % (", ".join(DETECTION_KEYS), detection))
    if detection["method"] != "by_channel":
        raise RecordingError("spikeinterface: detection method %r; sec. 2.7 declares "
                             "\"by_channel\"" % (detection["method"],))
    if recording.get_num_segments() != 1:
        raise RecordingError("spikeinterface: the recording has %d segments; a Recording is one "
                             "segment" % recording.get_num_segments())
    chans = list(recording.get_channel_ids())
    ids = [_as_int(c) for c in chans]
    if any(e is None for e in ids):
        raise RecordingError("spikeinterface: channel ids %r are not all integers"
                             % ([c for c, e in zip(chans, ids) if e is None],))
    if len(set(ids)) != len(ids):
        raise RecordingError("spikeinterface: two channels have one electrode id: %r" % (chans,))
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    method_kwargs = {"peak_sign": detection["peak_sign"],
                     "detect_threshold": detection["detect_threshold"],
                     "exclude_sweep_ms": detection["exclude_sweep_ms"],
                     "random_slices_kwargs": {"seed": NOISE_SEED}}
    try:
        peaks = detect_peaks(recording, method=detection["method"], method_kwargs=method_kwargs,
                             job_kwargs=dict(JOB_KWARGS))
    except AssertionError as exc:   # spikeinterface asserts on its arguments
        raise RecordingError("spikeinterface: detect_peaks refuses the detection %r: %s"
                             % (detection, exc)) from exc
    spikes = {}
    for i, e in enumerate(ids):
        spikes[e] = np.sort(peaks["sample_index"][peaks["channel_index"] == i]).astype(np.int64)
    if name is None:
        name = "recording"
    src = {"adapter": "spikeinterface", "kind": "recording", "detection": dict(detection),
           "noise": "spikeinterface get_noise_levels, method mad, random chunks seed %d"
                    % NOISE_SEED}
    src.update(source or {})
    rec = Recording(name=name, spikes=spikes, fs_raw_hz=float(recording.get_sampling_frequency()),
                    n_samples=int(recording.get_num_samples(segment_index=0)),
                    positions_um=_positions_of(recording), grid=None, source=src)
    return _checked(rec)


def read(path, reader=None, detection=None, name=None, **reader_options):
    """The Recording of the file at `path`, read by the SpikeInterface
    extractor named `reader` (a key of its recording or sorting extractor
    lists; required). A sorting goes through from_sorting, a raw recording
    through from_recording."""
    import spikeinterface
    from spikeinterface.core import BaseRecording, BaseSorting
    from spikeinterface.extractors import (recording_extractor_full_dict,
                                           sorting_extractor_full_dict)
    if reader is None:
        raise RecordingError("spikeinterface: `reader` not given; the format is never guessed: "
                             "name a SpikeInterface extractor")
    in_rec = reader in recording_extractor_full_dict
    in_sort = reader in sorting_extractor_full_dict
    if not (in_rec or in_sort):
        raise RecordingError("spikeinterface: unknown reader %r; the readers are the keys of "
                             "spikeinterface.extractors' recording_extractor_full_dict and "
                             "sorting_extractor_full_dict" % (reader,))
    # a name in both lists: a declared detection says the file is raw voltage
    kind = "recording" if (in_rec and (not in_sort or detection is not None)) else "sorting"
    cls = (recording_extractor_full_dict if kind == "recording"
           else sorting_extractor_full_dict)[reader]
    own = {k: reader_options.pop(k) for k in _SORTING_OPTIONS if k in reader_options}
    if kind == "recording" and own:
        raise RecordingError("spikeinterface: %s given for a raw recording, which holds them"
                             % ", ".join(sorted(own)))
    if kind == "sorting" and detection is not None:
        raise RecordingError("spikeinterface: detection given for a sorting (reader %r); a "
                             "detection applies to raw voltage only" % (reader,))
    try:
        obj = cls(path, **reader_options)
    except Exception as exc:   # any reader's failure, named and passed on
        raise RecordingError("spikeinterface: reader %r cannot read %r: %s: %s"
                             % (reader, path, type(exc).__name__, exc)) from exc
    if name is None:
        name = os.path.basename(os.path.normpath(str(path)))
    source = {"path": os.path.abspath(str(path)), "reader": reader,
              "reader_options": dict(reader_options), "spikeinterface": spikeinterface.__version__}
    if isinstance(obj, BaseSorting):
        n_samples = own.get("n_samples")
        if n_samples is None and obj.has_recording():
            n_samples = int(obj._recording.get_num_samples(segment_index=0))
        return from_sorting(obj, n_samples, positions_um=own.get("positions_um"),
                            grid=own.get("grid"), name=name, source=source)
    if isinstance(obj, BaseRecording):
        return from_recording(obj, detection, name=name, source=source)
    raise RecordingError("spikeinterface: reader %r gave a %s, neither a recording nor a sorting"
                         % (reader, type(obj).__name__))
