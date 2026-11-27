"""Block 43: the extractor's per-electrode spike-train files as a Recording
(specs/SPEC.md block 43). The extractor's own readers, imported through
sbitool._sources.use("extractor"); nothing is re-detected or resampled.

CONTRACT (wave A): the signature is fixed.

Pure ASCII, LF only.
"""

import os
import sys

from sbitool import _sources
from sbitool.interfaces import Recording
from sbitool.recordings import RecordingError, _checked


def _extractor():
    """The extractor's module channel_subset_extraction, its folders put on
    sys.path by _sources.use("extractor") (sec. 2.7, "The sources the package
    imports"). The one place this package imports the extractor.

    Fallback while block 40's `use` is a stub (it raises NotImplementedError):
    the same folders, GROUPS["extractor"][1], relative to the first of
    _sources.EXTRACTOR_ENV that names a directory, are put on sys.path here.
    The fallback is taken only on NotImplementedError; any other error of
    `use` (its SourceError) is passed on unchanged."""
    try:
        _sources.use("extractor")
    except NotImplementedError:
        root = None
        for var in _sources.EXTRACTOR_ENV:
            val = os.environ.get(var)
            if val and os.path.isdir(val):
                root = os.path.abspath(val)
                break
        if root is None:
            raise _sources.SourceError(
                "the Sbi-extractor's checkout is not found: set %s to its folder"
                % " or ".join(_sources.EXTRACTOR_ENV))
        for rel in _sources.GROUPS["extractor"][1]:
            d = os.path.normpath(os.path.join(root, rel))
            if not os.path.isdir(d):
                raise _sources.SourceError("the extractor's folder %s is absent (from %s)"
                                           % (d, " or ".join(_sources.EXTRACTOR_ENV)))
            if d not in sys.path:
                sys.path.insert(0, d)
    import channel_subset_extraction
    return channel_subset_extraction


def read(folder, fs_raw_hz=None, ptrain_format="raster", ptrain_varname="ptrain",
         ptrain_name_pattern=None, grid_width=None, index_base=None, pitch_um=None, name=None):
    """The Recording of one folder of per-electrode files. `fs_raw_hz`,
    `grid_width` and `index_base` are required (a missing one is a
    RecordingError naming it: the files do not hold them); `pitch_um` None
    gives positions_um None and grid None. `ptrain_format` "raster" or
    "sparse_peaks", `ptrain_varname`, `ptrain_name_pattern` (None: the
    extractor's default) as the extractor declares them; its refusals are
    passed on as RecordingError."""
    missing = [k for k, v in (("fs_raw_hz", fs_raw_hz), ("grid_width", grid_width),
                              ("index_base", index_base)) if v is None]
    if missing:
        raise RecordingError("ptrain: %s not given; the files do not hold %s, so it is an "
                             "argument (sec. 2.7, block 43)"
                             % (", ".join(missing), "it" if len(missing) == 1 else "them"))
    if isinstance(grid_width, bool) or not isinstance(grid_width, int) or grid_width < 1:
        raise RecordingError("ptrain: grid_width must be an integer >= 1, got %r" % (grid_width,))
    if pitch_um is not None and (isinstance(pitch_um, bool)
                                 or not isinstance(pitch_um, (int, float)) or not pitch_um > 0):
        raise RecordingError("ptrain: pitch_um must be a positive number or None, got %r"
                             % (pitch_um,))
    cse = _extractor()
    try:
        inv = cse.load_ptrain_folder(folder, fs_raw=fs_raw_hz, index_base=index_base,
                                     name_pattern=ptrain_name_pattern, varname=ptrain_varname,
                                     fmt=ptrain_format)
        cse.validate_grid(inv.indices, width=grid_width, base=index_base)
    except ValueError as exc:   # PtrainLoadError and GeometryError are ValueErrors
        raise RecordingError("ptrain: the extractor refuses %r: %s" % (folder, exc)) from exc
    spikes = {int(k): inv.spikes[k] for k in inv.indices}
    positions, grid = None, None
    if pitch_um is not None:
        positions = {}
        for k in spikes:
            c = cse.electrode_coords(k, width=grid_width, base=index_base)
            positions[k] = (float(c.col * pitch_um), float(c.row * pitch_um))
        grid = (int(grid_width), int(index_base), float(pitch_um))
    if name is None:
        name = os.path.basename(os.path.normpath(folder))
    source = {"adapter": "ptrain", "path": os.path.abspath(folder),
              "options": {"fs_raw_hz": fs_raw_hz, "ptrain_format": ptrain_format,
                          "ptrain_varname": ptrain_varname,
                          "ptrain_name_pattern": ptrain_name_pattern,
                          "grid_width": grid_width, "index_base": index_base,
                          "pitch_um": pitch_um}}
    rec = Recording(name=name, spikes=spikes, fs_raw_hz=float(inv.fs_raw),
                    n_samples=int(inv.n_samples), positions_um=positions, grid=grid,
                    source=source)
    return _checked(rec)
