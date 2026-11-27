"""Block 43: the extractor's per-electrode spike-train files as a Recording
(specs/SPEC.md block 43). The extractor's own readers, imported through
sbitool._sources.use("extractor"); nothing is re-detected or resampled.

CONTRACT (wave A): the signature is fixed.

Pure ASCII, LF only.
"""


def read(folder, fs_raw_hz=None, ptrain_format="raster", ptrain_varname="ptrain",
         ptrain_name_pattern=None, grid_width=None, index_base=None, pitch_um=None, name=None):
    """The Recording of one folder of per-electrode files. `fs_raw_hz`,
    `grid_width` and `index_base` are required (a missing one is a
    RecordingError naming it: the files do not hold them); `pitch_um` None
    gives positions_um None and grid None. `ptrain_format` "raster" or
    "sparse_peaks", `ptrain_varname`, `ptrain_name_pattern` (None: the
    extractor's default) as the extractor declares them; its refusals are
    passed on as RecordingError."""
    raise NotImplementedError("block 43")
