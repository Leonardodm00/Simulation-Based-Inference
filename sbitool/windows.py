"""Block 44: from a recording to the windows a bundle's networks take, by
the extractor's own rule (specs/SPEC.md sec. 2.7, "From a recording to
windows", and block 44).

CONTRACT (wave B): the signatures below are fixed.

Pure ASCII, LF only.
"""

import sys

EXIT_ARGS, EXIT_REFUSE = 2, 3
PITCH_TOL_UM = 1.0


class WindowsError(ValueError):
    """The recording cannot be cut into the bundle's windows: the message
    names the two values that differ, or what is missing."""


def grid_of(recording, extraction):
    """{electrode id: (row, col)} of the recording on the extraction's
    grid, after the comparison block 44's Rule states (the recording's own
    grid, or layout.as_grid of its positions); WindowsError naming both
    values on a mismatch of the rate, the grid or the pitch."""
    raise NotImplementedError("block 44")


def subregions(recording, extraction):
    """The extractor's Subregion objects for this recording
    (partition_subregions on grid_of's rows and columns and the
    electrodes' mean firing rates)."""
    raise NotImplementedError("block 44")


def traces(recording, subs, extraction):
    """(x, fs_ifr_hz): x (n_subregions, K) float32 [Hz per electrode], the
    extractor's subregion_ifrs."""
    raise NotImplementedError("block 44")


def cut(x, W, fs_ifr_hz):
    """(windows, ids): windows (n, W) float32, the rows of every subregion
    in subregion order, each subregion's windows by windows_of(K, W); ids a
    dict of (n,) arrays `subregion` (int64), `window_idx` (int64, within
    the subregion), `t_start_s` (float64, window_idx W / fs_ifr_hz)."""
    raise NotImplementedError("block 44")


def extract(recording, extraction, W):
    """(windows, ids, subs): the three steps above."""
    raise NotImplementedError("block 44")


def main(argv=None):
    """`python3 -m sbitool windows --bundle FILE --recording PATH --adapter
    NAME [--option key=value ...] --out windows.npz`: 0 with WINDOWS OK:
    ..., EXIT_ARGS, EXIT_REFUSE."""
    raise NotImplementedError("block 44")


if __name__ == "__main__":
    sys.exit(main())
