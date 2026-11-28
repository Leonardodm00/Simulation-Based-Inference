"""Block 46: the device and the window as figures (specs/SPEC.md block 46
with its "Contract of wave C").

CONTRACT (wave C): the names, signatures and artist ids (gid) below are
fixed by specs/SPEC.md; the bodies are block 46's. The figures draw what
blocks 43 and 44 return and compute nothing: no statistic is taken in this
module. matplotlib and numpy only; block 21's colours are read from
ev_figures (through sbitool._sources), not copied.

The figures are matplotlib.figure.Figure objects made without pyplot, so
that drawing them leaves no global figure state; ev_figures (imported for
its colours) selects the Agg backend at its import, and its style() is
applied by the command only.

Pure ASCII, LF only.
"""

import argparse
import os
import sys

import numpy as np

from sbitool import _sources  # noqa: F401  (ev_figures comes through it)

EXIT_ARGS, EXIT_REFUSE = 2, 3
N_COLORS = 6      # subregion j is drawn in SUBREGION_COLORS[j % N_COLORS]
GID_ELECTRODES, GID_HIGHLIGHT, GID_IFR = "electrodes", "highlight", "ifr"
GID_SUBREGION, GID_RASTER = "subregion-%d", "raster-%d"
DPI = 150

# drawing sizes (points^2 for the markers, points for the lines); style only
_S_ELECTRODE, _S_RING, _LW_EDGE, _LW_RING = 60.0, 160.0, 1.0, 1.2
_LABEL_OFFSET_PT = (6, 6)      # the subregion's index, beside its centre (amendment 1 of wave C)


class FigureError(ValueError):
    """A figure that cannot be drawn; the message says why."""


def _evf():
    """ev_figures (block 21), through the evaluation's source group."""
    _sources.use("evaluation")
    import ev_figures
    return ev_figures


def subregion_colors():
    """The six colours of the subregions, block 21's: ev_figures'
    ARM_COLOR["A1"], ARM_COLOR["A2"], ARM_COLOR["A5"], OTHER_ARM,
    COND_COLOR[0], COND_COLOR[2], in this order."""
    ev = _evf()
    return (ev.ARM_COLOR["A1"], ev.ARM_COLOR["A2"], ev.ARM_COLOR["A5"], ev.OTHER_ARM,
            ev.COND_COLOR[0], ev.COND_COLOR[2])


def __getattr__(name):
    # device.SUBREGION_COLORS (the SPEC's name) read lazily, so that importing
    # this module does not import ev_figures
    if name == "SUBREGION_COLORS":
        return subregion_colors()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def positions(recording):
    """({electrode id: (x, y)}, unit): the recording's positions_um with
    unit "um"; else, for a grid with a pitch, (column pitch, row pitch)
    with unit "um"; else, for a grid without a pitch, (column, row) with
    unit None. FigureError for a recording with neither."""
    if recording.positions_um is not None:
        return ({int(e): (float(xy[0]), float(xy[1])) for e, xy in recording.positions_um.items()},
                "um")
    if recording.grid is not None:
        from sbitool.recordings import layout
        try:
            rc = layout.grid(recording)
        except layout.LayoutError as exc:
            raise FigureError("the recording %r cannot be placed on its grid: %s"
                              % (recording.name, exc)) from exc
        pitch = recording.grid[2]
        if pitch is not None:
            p = float(pitch)
            return {e: (c * p, r * p) for e, (r, c) in rc.items()}, "um"
        return {e: (float(c), float(r)) for e, (r, c) in rc.items()}, None
    raise FigureError("the recording %r has neither electrode positions nor a grid: its "
                      "electrodes cannot be placed" % (recording.name,))


def _labels(ax, unit):
    if unit == "um":
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
    else:
        ax.set_xlabel("column")
        ax.set_ylabel("row")


def _subregion_of(recording, subregions):
    """{electrode id: subregion index}; FigureError for a subregion naming
    an electrode the recording does not hold, or an electrode in two."""
    known = set(int(e) for e in recording.spikes)
    owner = {}
    for j, s in enumerate(subregions):
        members = [int(m) for m in s.members]
        bad = sorted(set(members + [int(s.center)]) - known)
        if bad:
            raise FigureError("subregion %d names electrode(s) the recording %r does not hold: %s"
                              % (j, recording.name, bad[:10]))
        for m in members:
            if m in owner:
                raise FigureError("electrode %d is in subregions %d and %d" % (m, owner[m], j))
            owner[m] = j
    return owner


def _is_index(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def fig_device(recording, subregions=None, highlight=None):
    """The device: one axes, equal aspect; a scatter (gid GID_ELECTRODES)
    with one marker per electrode in sorted id order at positions(); an
    electrode outside every subregion in ev_figures.SIM_GRAY, subregion j's
    in subregion_colors()[j % N_COLORS]; an electrode without a spike
    hollow (no face colour); the index j as an annotation (gid
    GID_SUBREGION % j) whose xy is subregion j's centre electrode and whose
    text is offset by (6, 6) points, with no halo over the marker
    (amendment 1 of wave C); highlight = j: a second scatter (gid
    GID_HIGHLIGHT), rings in ev_figures.INK around subregion j's
    electrodes. Axes labelled "x [um]", "y [um]", or "column", "row"
    without a pitch. Returns the Figure. FigureError for a highlight that
    is no subregion's index or a subregion naming an unknown electrode."""
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure
    ev = _evf()
    subregions = list(subregions) if subregions is not None else []
    if highlight is not None and not (_is_index(highlight)
                                      and 0 <= int(highlight) < len(subregions)):
        raise FigureError("highlight %r is no subregion's index (%d subregion(s))"
                          % (highlight, len(subregions)))
    pos, unit = positions(recording)
    owner = _subregion_of(recording, subregions)
    colors = subregion_colors()
    ids = sorted(int(e) for e in recording.spikes)
    missing = [e for e in ids if e not in pos]
    if missing:
        raise FigureError("the recording %r has no position for electrode(s) %s"
                          % (recording.name, missing[:10]))
    xy = np.array([pos[e] for e in ids], dtype=np.float64).reshape(-1, 2)
    edge = np.array([to_rgba(colors[owner[e] % N_COLORS] if e in owner else ev.SIM_GRAY)
                     for e in ids]).reshape(-1, 4)
    face = edge.copy()
    silent = np.array([np.asarray(recording.spikes[e]).size == 0 for e in ids], dtype=bool)
    face[silent] = to_rgba("none")

    fig = Figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xy[:, 0], xy[:, 1], s=_S_ELECTRODE, facecolors=face, edgecolors=edge,
               linewidths=_LW_EDGE, gid=GID_ELECTRODES, zorder=2)
    for j, s in enumerate(subregions):
        cx, cy = pos[int(s.center)]
        ax.annotate(str(j), xy=(cx, cy), xytext=_LABEL_OFFSET_PT, textcoords="offset points",
                    gid=GID_SUBREGION % j, ha="left", va="bottom", color=ev.INK, fontsize=7,
                    fontweight="bold", zorder=4)
    if highlight is not None:
        hx = np.array([pos[int(m)] for m in subregions[int(highlight)].members],
                      dtype=np.float64).reshape(-1, 2)
        ax.scatter(hx[:, 0], hx[:, 1], s=_S_RING, facecolors="none", edgecolors=ev.INK,
                   linewidths=_LW_RING, gid=GID_HIGHLIGHT, zorder=3)
    ax.set_aspect("equal")
    _labels(ax, unit)
    title = "%s: %d electrode(s), %d subregion(s)" % (recording.name, len(ids), len(subregions))
    if highlight is not None:
        title += ", subregion %d ringed" % int(highlight)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _window_label(fs_ifr_hz):
    """The window axis's label (block 46, fix round of 2026-11-28, D-180):
    "x [spikes per electrode per <b>-ms bin]", <b> = 1000 / fs_ifr_hz with
    %g, block 23's wording for the same quantity. x is drawn unchanged: a
    window holds spikes per bin per electrode (x / Delta t is Hz per
    electrode), so the label names the bin, it does not rescale x."""
    return "x [spikes per electrode per %g-ms bin]" % (1000.0 / float(fs_ifr_hz))


def fig_window(recording, subregion, x, window_idx, W, fs_ifr_hz):
    """One window of one subregion: two axes sharing the time axis. The
    window's span is [t0, t0 + W / fs_ifr_hz), t0 = window_idx W /
    fs_ifr_hz [s]. Above: one eventplot row per member electrode, in the
    subregion's member order (gid GID_RASTER % electrode id), with the
    electrode's spike times index / fs_raw_hz [s] inside the span. Below:
    one Axes.stairs (a StepPatch, gid GID_IFR; amendment 1 of wave C) with
    values `x` ((W,), [spikes per bin per electrode]; D-180) unchanged and
    the W + 1 edges t0 + k / fs_ifr_hz, k = 0..W, each value over its whole
    bin. Labels "time [s]", "electrode", and _window_label(fs_ifr_hz) for x.
    Returns the Figure. FigureError for x of another length than W."""
    from matplotlib.figure import Figure
    ev = _evf()
    if not _is_index(W) or int(W) < 1:
        raise FigureError("W must be an integer >= 1, got %r" % (W,))
    if not _is_index(window_idx) or int(window_idx) < 0:
        raise FigureError("window_idx must be an integer >= 0, got %r" % (window_idx,))
    if isinstance(fs_ifr_hz, bool) or not np.isfinite(fs_ifr_hz) or not fs_ifr_hz > 0:
        raise FigureError("fs_ifr_hz must be a positive finite number, got %r" % (fs_ifr_hz,))
    W, fs_ifr_hz = int(W), float(fs_ifr_hz)
    x = np.asarray(x)
    if x.shape != (W,):
        raise FigureError("x must be the window, of shape (%d,), got shape %r" % (W, x.shape))
    members = [int(m) for m in subregion.members]
    unknown = [m for m in members if m not in recording.spikes]
    if unknown:
        raise FigureError("the subregion names electrode(s) the recording %r does not hold: %s"
                          % (recording.name, unknown[:10]))
    t0 = int(window_idx) * W / fs_ifr_hz
    t1 = t0 + W / fs_ifr_hz
    fs_raw = float(recording.fs_raw_hz)
    rows = []
    for m in members:
        t = np.asarray(recording.spikes[m], dtype=np.int64) / fs_raw
        rows.append(t[(t >= t0) & (t < t1)])

    fig = Figure(figsize=(5.2, 4.0))
    ax_r, ax_x = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]})
    colls = ax_r.eventplot(rows, lineoffsets=np.arange(len(members)), linelengths=0.8,
                           linewidths=1.0, colors=ev.INK)
    for m, c in zip(members, colls):
        c.set_gid(GID_RASTER % m)
    ax_r.set_yticks(np.arange(len(members)))
    ax_r.set_yticklabels([str(m) for m in members])
    ax_r.set_ylim(len(members) - 0.5, -0.5)       # the first member (the centre) on top
    ax_r.set_ylabel("electrode")
    ax_r.set_title("%s: subregion of centre %d, window %d"
                   % (recording.name, int(subregion.center), int(window_idx)))
    edges = t0 + np.arange(W + 1) / fs_ifr_hz
    # baseline None: an open path, no vertical strokes down to 0 at t0 and t1 (which
    # would read as the trace falling to 0 at the window's ends)
    ax_x.stairs(x, edges, baseline=None, color=ev.INK, lw=1.0, gid=GID_IFR)
    ax_x.set_xlim(t0, t1)
    ax_x.set_xlabel("time [s]")
    ax_x.set_ylabel(_window_label(fs_ifr_hz))
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ the command

def _parser():
    from sbitool import windows
    p = argparse.ArgumentParser(
        prog="python3 -m sbitool device-figure",
        description="Draw a recording's device, with the bundle's subregions, and one window.")
    p.add_argument("--bundle", required=True, help="the .sbibundle file")
    p.add_argument("--recording", required=True, help="the recording's file or folder")
    p.add_argument("--adapter", required=True, help="the recording adapter (ptrain, spikeinterface)")
    p.add_argument("--option", action="append", default=[], type=windows.parse_option,
                   metavar="KEY=VALUE",
                   help="an adapter option; VALUE parsed as JSON when it parses (repeatable)")
    p.add_argument("--out", required=True, help="the output folder")
    p.add_argument("--subregion", type=int, default=None, help="j, the subregion of the window")
    p.add_argument("--window", type=int, default=None, help="i, the window within subregion j")
    return p


def _refuse(msg):
    sys.stderr.write("REFUSED: %s\n" % " ".join(str(msg).split()))
    return EXIT_REFUSE


def _save(fig, path):
    tmp = path + ".part"
    fig.savefig(tmp, dpi=DPI, format="png")
    os.replace(tmp, path)


def main(argv=None):
    """`python3 -m sbitool device-figure --bundle FILE --recording PATH
    --adapter NAME [--option KEY=VALUE ...] --out DIR [--subregion j
    --window i]`: writes <out>/device.png and, with --subregion and
    --window, <out>/window_s<j>_w<i>.png (dpi DPI); 0 with `DEVICE FIGURES
    OK: <n> file(s)`; EXIT_ARGS; EXIT_REFUSE with one `REFUSED:` line."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else EXIT_ARGS
    if (args.subregion is None) != (args.window is None):
        sys.stderr.write("sbitool device-figure: --subregion and --window are given together\n")
        return EXIT_ARGS
    options = dict(args.option)
    if "detection" in options:
        sys.stderr.write("sbitool device-figure: --option detection is refused: the detection "
                         "is the bundle's (its extraction.detection), never the user's\n")
        return EXIT_ARGS
    from sbitool import bundle as bundle_mod
    from sbitool import recordings, windows
    try:
        b = bundle_mod.read(args.bundle)
    except (bundle_mod.BundleError, OSError) as exc:
        return _refuse("the bundle %s: %s" % (args.bundle, exc))
    try:
        r = windows.from_file(b.manifest, args.recording, args.adapter, options)
    except TypeError as exc:    # an option the adapter does not take: the arguments
        sys.stderr.write("sbitool device-figure: adapter %r: %s\n" % (args.adapter, exc))
        return EXIT_ARGS
    except (windows.WindowsError, recordings.RecordingError, _sources.SourceError,
            OSError) as exc:
        return _refuse(exc)
    rec, subs, ids = r["recording"], r["subregions"], r["ids"]
    j, i = args.subregion, args.window
    figs = []
    try:
        _evf().style()
        if j is not None:
            if not 0 <= j < len(subs):
                return _refuse("subregion %d does not exist: the recording %r has %d "
                               "subregion(s), 0..%d" % (j, rec.name, len(subs), len(subs) - 1))
            sel = np.flatnonzero((ids["subregion"] == j) & (ids["window_idx"] == i))
            n_win = int(np.count_nonzero(ids["subregion"] == j))
            if sel.size != 1:
                return _refuse("window %d of subregion %d does not exist: it has %d window(s), "
                               "0..%d" % (i, j, n_win, n_win - 1))
            figs.append(("device.png", fig_device(rec, subs, highlight=j)))
            figs.append(("window_s%d_w%d.png" % (j, i),
                         fig_window(rec, subs[j], r["windows"][int(sel[0])], i,
                                    int(b.manifest["W"]), r["fs_ifr_hz"])))
        else:
            figs.append(("device.png", fig_device(rec, subs)))
    except (FigureError, _sources.SourceError) as exc:
        return _refuse(exc)
    os.makedirs(args.out, exist_ok=True)
    for name, fig in figs:
        path = os.path.join(args.out, name)
        _save(fig, path)
        print(path)
    print("DEVICE FIGURES OK: %d file(s)" % len(figs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
