"""Block 46: the device and the window as figures (specs/SPEC.md block 46
with its "Contract of wave C").

CONTRACT (wave C): the names, signatures and artist ids (gid) below are
fixed by specs/SPEC.md; the bodies are block 46's. The figures draw what
blocks 43 and 44 return and compute nothing: no statistic is taken in this
module. matplotlib and numpy only; block 21's colours are read from
ev_figures (through sbitool._sources), not copied.

Pure ASCII, LF only.
"""

import sys

from sbitool import _sources  # noqa: F401  (ev_figures comes through it)

EXIT_ARGS, EXIT_REFUSE = 2, 3
N_COLORS = 6      # subregion j is drawn in SUBREGION_COLORS[j % N_COLORS]
GID_ELECTRODES, GID_HIGHLIGHT, GID_IFR = "electrodes", "highlight", "ifr"
GID_SUBREGION, GID_RASTER = "subregion-%d", "raster-%d"
DPI = 150


class FigureError(ValueError):
    """A figure that cannot be drawn; the message says why."""


def subregion_colors():
    """The six colours of the subregions, block 21's: ev_figures'
    ARM_COLOR["A1"], ARM_COLOR["A2"], ARM_COLOR["A5"], OTHER_ARM,
    COND_COLOR[0], COND_COLOR[2], in this order."""
    raise NotImplementedError("block 46: device.subregion_colors")


def positions(recording):
    """({electrode id: (x, y)}, unit): the recording's positions_um with
    unit "um"; else, for a grid with a pitch, (column pitch, row pitch)
    with unit "um"; else, for a grid without a pitch, (column, row) with
    unit None. FigureError for a recording with neither."""
    raise NotImplementedError("block 46: device.positions")


def fig_device(recording, subregions=None, highlight=None):
    """The device: one axes, equal aspect; a scatter (gid GID_ELECTRODES)
    with one marker per electrode in sorted id order at positions(); an
    electrode outside every subregion in ev_figures.SIM_GRAY, subregion j's
    in subregion_colors()[j % N_COLORS]; an electrode without a spike
    hollow (no face colour); the index j as a text (gid GID_SUBREGION % j)
    at subregion j's centre electrode; highlight = j: a second scatter (gid
    GID_HIGHLIGHT), rings in ev_figures.INK around subregion j's
    electrodes. Axes labelled "x [um]", "y [um]", or "column", "row"
    without a pitch. Returns the Figure. FigureError for a highlight that
    is no subregion's index or a subregion naming an unknown electrode."""
    raise NotImplementedError("block 46: device.fig_device")


def fig_window(recording, subregion, x, window_idx, W, fs_ifr_hz):
    """One window of one subregion: two axes sharing the time axis. The
    window's span is [t0, t0 + W / fs_ifr_hz), t0 = window_idx W /
    fs_ifr_hz [s]. Above: one eventplot row per member electrode, in the
    subregion's member order (gid GID_RASTER % electrode id), with the
    electrode's spike times index / fs_raw_hz [s] inside the span. Below:
    one line (gid GID_IFR, drawstyle "steps-post"), x data t0 + k /
    fs_ifr_hz for k = 0..W-1, y data `x` ((W,), [Hz per electrode])
    unchanged. Labels "time [s]", "electrode", "IFR [Hz per electrode]".
    Returns the Figure. FigureError for x of another length than W."""
    raise NotImplementedError("block 46: device.fig_window")


def main(argv=None):
    """`python3 -m sbitool device-figure --bundle FILE --recording PATH
    --adapter NAME [--option KEY=VALUE ...] --out DIR [--subregion j
    --window i]`: writes <out>/device.png and, with --subregion and
    --window, <out>/window_s<j>_w<i>.png (dpi DPI); 0 with `DEVICE FIGURES
    OK: <n> file(s)`; EXIT_ARGS; EXIT_REFUSE with one `REFUSED:` line."""
    raise NotImplementedError("block 46: device.main")


if __name__ == "__main__":
    sys.exit(main())
