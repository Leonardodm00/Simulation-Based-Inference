#!/usr/bin/env python3
"""Smoke test of block 46: the device and the window as figures (specs/SPEC.md block 46, "Test
oracles" and "Contract of wave C", checks T46*).

A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t46a_device         one marker per electrode at its position, the subregions' colours
    test_t46b_window         the raster holds the span's spikes; the stairs' data are x and the edges
    test_t46c_command        PNG files with the PNG signature; no statistic in the module

Run
    SBIX_CLONE=<the Sbi-extractor's checkout> PYTHONDONTWRITEBYTECODE=1 \
        python3 sbitool/tests/test_smoke_tool_figures.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO.

Pure ASCII, LF only.
"""

import os
import shutil
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SEED = 20261127  # every random number of this file derives from it
_TMP = []


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="sbitool_%s_" % tag)
    _TMP.append(d)
    return d


def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


import ast  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import subprocess  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from sbitool import _sources  # noqa: E402
from sbitool.interfaces import Recording  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# the 6 x 6 device of T46a-T46c: ids 1..36 (index base 1), pitch 60 um, 10 kHz, 2 s
GW, BASE, PITCH, FS, NSAMP = 6, 1, 60.0, 10000.0, 20000
NE, NSUB, THR = 4, 2, 2.0          # two subregions of 4 electrodes, 2 Hz (4 spikes in 2 s)
DT, SIG, WIN = 0.02, 0.04, 7       # fs_ifr = 50 Hz, K = 100 bins, 14 windows of 7 bins


def _need_extractor():
    try:
        _sources.extractor_dir()
    except _sources.SourceError as exc:
        raise unittest.SkipTest("the Sbi-extractor's checkout is absent (set SBIX_DIR or "
                                "SBIX_CLONE): %s" % exc)


def _device():
    from sbitool.plotting import device
    return device


def _evf():
    """Block 21's colours, read by the test from ev_figures itself."""
    _sources.use("evaluation")
    import ev_figures
    return ev_figures


def _rgba(c):
    from matplotlib.colors import to_rgba
    return np.array(to_rgba(c))


def _wtest():
    sys.path.insert(0, _HERE)
    try:
        import test_smoke_tool_windows as tw
    finally:
        sys.path.remove(_HERE)
    return tw


def _extraction(**kw):
    e = dict(electrodes_per_subset=NE, n_subsets=NSUB, mfr_threshold_hz=THR, w_size_s=DT,
             gaussian_window_s=SIG, grid_width=GW, index_base=BASE, pitch_um=PITCH,
             fs_raw_hz=FS, detection=None)
    e.update(kw)
    from sbitool import schema as S
    probs = S.validate_extraction(e)
    assert not probs, "the test's extraction is not valid: %s" % probs
    return e


_FX = {}


def _fixture():
    """A ptrain folder of the 6 x 6 device, its trains and its Recording
    (block 43's ptrain.read), cached. Counts per electrode: 8 electrodes
    silent (hollow), 4 with 1..3 spikes (below THR), the others 4..27."""
    if not _FX:
        from sbitool.recordings import ptrain
        rng = np.random.default_rng(SEED)
        ids = list(range(BASE, BASE + GW * GW))
        vals = [0] * 8 + [1, 2, 3, 3] + list(range(4, 28))
        order = rng.permutation(len(ids))
        counts = {ids[i]: vals[order[i]] for i in range(len(ids))}
        trains = {e: np.sort(rng.choice(NSAMP, size=k, replace=False)).astype(np.int64)
                  for e, k in counts.items()}
        folder = os.path.join(_tmpdir("f46"), "rec")
        _wtest()._write_raster(folder, trains, NSAMP)
        rec = ptrain.read(folder, fs_raw_hz=FS, grid_width=GW, index_base=BASE, pitch_um=PITCH)
        _FX.update(counts=counts, trains=trains, folder=folder, rec=rec)
    return _FX


def _by_gid(artists, gid):
    got = [a for a in artists if a.get_gid() == gid]
    assert len(got) == 1, "%d artists with gid %r, expected 1" % (len(got), gid)
    return got[0]


def _expected_xy(e, pitch=PITCH):
    """The grid's position of electrode e, independently: row-major linear index."""
    row, col = (e - BASE) // GW, (e - BASE) % GW
    return (col * pitch, row * pitch) if pitch is not None else (float(col), float(row))


def _refused(fn, words):
    dev = _device()
    try:
        fn()
    except dev.FigureError as exc:
        msg = str(exc)
        missing = [w for w in words if w not in msg]
        assert not missing, "refused, but %r does not name %s" % (msg, missing)
        return
    raise AssertionError("not refused (expected a FigureError naming %s)" % (words,))


def _check_device(fig, rec, subs, highlight, unit_um=True, pitch=PITCH):
    """The oracle of T46a on one device figure."""
    ev = _evf()
    six = [ev.ARM_COLOR["A1"], ev.ARM_COLOR["A2"], ev.ARM_COLOR["A5"], ev.OTHER_ARM,
           ev.COND_COLOR[0], ev.COND_COLOR[2]]
    assert len(fig.axes) == 1, "%d axes, expected one" % len(fig.axes)
    ax = fig.axes[0]
    sc = _by_gid(ax.collections, "electrodes")
    ids = sorted(rec.spikes)
    off = np.asarray(sc.get_offsets(), dtype=np.float64)
    want = np.array([_expected_xy(e, pitch) for e in ids], dtype=np.float64)
    # exact: the module and the test multiply the same integers by the same pitch
    assert off.shape == (len(ids), 2), "offsets %s, expected %d markers" % (off.shape, len(ids))
    assert np.array_equal(off, want), "the markers are not at the electrodes' positions"
    owner = {int(m): j for j, s in enumerate(subs) for m in s.members}
    face, edge = np.asarray(sc.get_facecolors()), np.asarray(sc.get_edgecolors())
    assert face.shape == edge.shape == (len(ids), 4), (face.shape, edge.shape)
    for i, e in enumerate(ids):
        col = _rgba(six[owner[e] % 6] if e in owner else ev.SIM_GRAY)
        assert np.allclose(edge[i], col, atol=0, rtol=0), \
            "electrode %d: edge %s, expected %s" % (e, edge[i], col)
        if rec.spikes[e].size == 0:
            assert face[i][3] == 0.0, "electrode %d has no spike but a face %s" % (e, face[i])
        else:
            assert np.array_equal(face[i], col), "electrode %d: face %s, expected %s" % (
                e, face[i], col)
    for j, s in enumerate(subs):
        # amendment 1 of wave C: an annotation whose xy is the centre electrode, its text
        # offset by (6, 6) points, with no halo over the marker
        from matplotlib.text import Annotation
        t = _by_gid(ax.texts, "subregion-%d" % j)
        assert isinstance(t, Annotation), "subregion %d's index is a %s, not an annotation" % (
            j, type(t).__name__)
        assert t.get_text() == str(j), (j, t.get_text())
        assert tuple(map(float, t.xy)) == _expected_xy(int(s.center), pitch), \
            "subregion %d's index does not point at its centre electrode %d" % (j, s.center)
        assert t.anncoords == "offset points" and tuple(map(float, t.xyann)) == (6.0, 6.0), \
            "subregion %d's index offset %r %r, expected (6, 6) points" % (j, t.anncoords,
                                                                           t.xyann)
        assert not t.get_path_effects(), "subregion %d's index has a halo" % j
    assert len([t for t in ax.texts if str(t.get_gid()).startswith("subregion-")]) == len(subs)
    rings = [c for c in ax.collections if c.get_gid() == "highlight"]
    if highlight is None:
        assert not rings, "a highlight is drawn without being asked"
    else:
        assert len(rings) == 1, "%d highlight scatters" % len(rings)
        hr = rings[0]
        ho = np.asarray(hr.get_offsets(), dtype=np.float64)
        hw = np.array([_expected_xy(int(m), pitch) for m in subs[highlight].members])
        assert np.array_equal(ho, hw), "the rings are not around subregion %d" % highlight
        he = np.asarray(hr.get_edgecolors())
        assert np.array_equal(he, np.tile(_rgba(ev.INK), (len(hw), 1))) or \
            np.array_equal(he, _rgba(ev.INK)[None, :]), "the rings are not in INK: %s" % he
        assert np.all(np.asarray(hr.get_facecolors())[:, 3] == 0.0), "the rings are filled"
    labels = (ax.get_xlabel(), ax.get_ylabel())
    assert labels == (("x [um]", "y [um]") if unit_um else ("column", "row")), labels
    assert ax.get_aspect() == 1.0, "aspect %r, expected equal" % (ax.get_aspect(),)


def test_t46a_device():
    """T46a: on a 6 x 6 recording with two subregions, one marker per electrode at its position
    and the subregions' electrodes in their colours, read back from the artists; hollow
    markers for the electrodes without a spike; the highlight's rings."""
    _need_extractor()
    dev = _device()
    from sbitool import windows
    fx = _fixture()
    rec, counts = fx["rec"], fx["counts"]
    subs = windows.subregions(rec, _extraction())
    assert len(subs) == NSUB, subs
    n_hollow = sum(1 for e in rec.spikes if rec.spikes[e].size == 0)
    assert n_hollow == 8 and n_hollow == sum(1 for k in counts.values() if k == 0)
    # (i) block 44's subregions, with and without the highlight
    for hl in (None, 0, 1):
        _check_device(dev.fig_device(rec, subs, highlight=hl), rec, subs, hl)
    # no subregion: every electrode in SIM_GRAY
    _check_device(dev.fig_device(rec), rec, [], None)
    # (ii) the colours by j mod 6 and a silent electrode inside a subregion (hollow in its
    # colour): seven one-electrode subregions, hand-made with block 44's two fields
    silent = sorted(e for e in rec.spikes if rec.spikes[e].size == 0)
    live = sorted(e for e in rec.spikes if rec.spikes[e].size > 0)
    seven = [SimpleNamespace(members=(e,), center=e) for e in [silent[0]] + live[:6]]
    _check_device(dev.fig_device(rec, seven, highlight=6), rec, seven, 6)
    # (iii) the positions' three sources: no positions_um but a grid with a pitch; a grid
    # without one ("column", "row"); neither (refused)
    rec_g = Recording(name="g", spikes=rec.spikes, fs_raw_hz=FS, n_samples=NSAMP,
                      positions_um=None, grid=(GW, BASE, PITCH), source={})
    _check_device(dev.fig_device(rec_g, subs, 1), rec_g, subs, 1)
    rec_n = Recording(name="n", spikes=rec.spikes, fs_raw_hz=FS, n_samples=NSAMP,
                      positions_um=None, grid=(GW, BASE, None), source={})
    _check_device(dev.fig_device(rec_n, subs, 0), rec_n, subs, 0, unit_um=False, pitch=None)
    assert dev.positions(rec_n)[1] is None and dev.positions(rec_g)[1] == "um"
    # positions_um prevail over the grid: positions that are not the grid's are drawn as given
    odd = {e: (float(e), -2.5 * e) for e in rec.spikes}
    rec_p = Recording(name="p", spikes=rec.spikes, fs_raw_hz=FS, n_samples=NSAMP,
                      positions_um=odd, grid=(GW, BASE, PITCH), source={})
    sc = _by_gid(dev.fig_device(rec_p).axes[0].collections, "electrodes")
    assert np.array_equal(np.asarray(sc.get_offsets()),
                          np.array([odd[e] for e in sorted(odd)])), "positions_um not used"
    rec_0 = Recording(name="bare", spikes=rec.spikes, fs_raw_hz=FS, n_samples=NSAMP,
                      positions_um=None, grid=None, source={})
    _refused(lambda: dev.fig_device(rec_0), ["neither"])
    # (iv) refusals: a highlight that is no subregion's index; an unknown electrode
    _refused(lambda: dev.fig_device(rec, subs, highlight=2), ["highlight", "2"])
    _refused(lambda: dev.fig_device(rec, None, highlight=0), ["highlight"])
    _refused(lambda: dev.fig_device(rec, subs, highlight=True), ["highlight"])
    _refused(lambda: dev.fig_device(rec, [SimpleNamespace(members=(1, 99), center=1)]), ["99"])
    # the SPEC's name for the six colours
    ev = _evf()
    assert tuple(dev.SUBREGION_COLORS) == (ev.ARM_COLOR["A1"], ev.ARM_COLOR["A2"],
                                           ev.ARM_COLOR["A5"], ev.OTHER_ARM, ev.COND_COLOR[0],
                                           ev.COND_COLOR[2])
    print("[T46a] %d electrodes (%d hollow), %d subregions %s; 9 figures read back"
          % (len(rec.spikes), n_hollow, len(subs), [list(s.members) for s in subs]))


def _check_window(fig, rec, sub, x, i, Wn, fs_ifr):
    """The oracle of T46b on one window figure; returns the spikes drawn."""
    assert len(fig.axes) == 2, "%d axes, expected two" % len(fig.axes)
    ax_r, ax_x = fig.axes
    t0 = i * Wn / fs_ifr                    # the span, as SPEC block 46 states it
    t1 = t0 + Wn / fs_ifr
    rasters = [c for c in ax_r.collections if str(c.get_gid()).startswith("raster-")]
    assert len(rasters) == len(sub.members), "%d raster rows, %d members" % (len(rasters),
                                                                              len(sub.members))
    n_drawn = 0
    for row, (m, c) in enumerate(zip(sub.members, rasters)):
        assert c.get_gid() == "raster-%d" % m, "row %d has gid %r, member %d" % (
            row, c.get_gid(), m)
        assert c.get_lineoffset() == row, "member %d at offset %r" % (m, c.get_lineoffset())
        t = np.asarray(rec.spikes[m], dtype=np.float64) / rec.fs_raw_hz
        want = t[(t >= t0) & (t < t1)]
        got = np.sort(np.asarray(c.get_positions(), dtype=np.float64))
        # exact: the same division index / fs_raw_hz on both sides
        assert np.array_equal(got, want), "electrode %d: drawn %s, in the span %s" % (
            m, got.tolist(), want.tolist())
        n_drawn += got.size
    # amendment 1 of wave C: one StepPatch (Axes.stairs), values x, the W + 1 bin edges
    from matplotlib.patches import StepPatch
    st = _by_gid(ax_x.patches, "ifr")
    assert isinstance(st, StepPatch), "the IFR is a %s, not a StepPatch" % type(st).__name__
    assert not [a for a in ax_x.lines if a.get_gid() == "ifr"], "an IFR line is drawn too"
    data = st.get_data()
    edges = t0 + np.arange(Wn + 1) / fs_ifr
    assert np.array_equal(np.asarray(data.edges, dtype=np.float64), edges), \
        "the IFR's edges are not t0 + k / fs_ifr, k = 0..W"
    assert np.array_equal(np.asarray(data.values), np.asarray(x)), \
        "the IFR's values are not the window's x"
    assert data.baseline is None, "the stairs are closed down to %r" % (data.baseline,)
    assert ax_x.get_xlabel() == "time [s]" and ax_r.get_ylabel() == "electrode" and \
        ax_x.get_ylabel() == "IFR [Hz per electrode]", (ax_x.get_xlabel(), ax_r.get_ylabel(),
                                                         ax_x.get_ylabel())
    assert ax_x.get_shared_x_axes().joined(ax_r, ax_x), "the two axes do not share time"
    return n_drawn


def test_t46b_window():
    """T46b: the raster holds exactly the spikes of the subregion's electrodes inside the
    window's span; the lower panel's stairs have the window's x as values and the W + 1 bins'
    edges as edges (amendment 1 of wave C)."""
    _need_extractor()
    dev = _device()
    from sbitool import windows
    fx = _fixture()
    rec = fx["rec"]
    ext = _extraction()
    subs = windows.subregions(rec, ext)
    x, fs_ifr = windows.traces(rec, subs, ext)
    win, ids = windows.cut(x, WIN, fs_ifr)
    total, checked = 0, 0
    for row in range(win.shape[0]):
        j, i = int(ids["subregion"][row]), int(ids["window_idx"][row])
        if i % 3:                           # every third window of each subregion: small and fast
            continue
        fig = dev.fig_window(rec, subs[j], win[row], i, WIN, fs_ifr)
        total += _check_window(fig, rec, subs[j], win[row], i, WIN, fs_ifr)
        checked += 1
    assert total > 0, "no spike fell in the windows checked: the case tests nothing"
    # the span's two ends, on spikes placed there: [t0, t1) holds t0 and not t1
    fs_i, Wn, i = 50.0, 5, 3                 # t0 = 0.3 s, t1 = 0.4 s; fs_raw 1 kHz
    t0 = i * Wn / fs_i
    t1 = t0 + Wn / fs_i
    s_lo, s_hi = int(round(t0 * 1000)), int(round(t1 * 1000))
    edge = {0: np.array([s_lo - 1, s_lo, s_lo + 1, s_hi - 1, s_hi, s_hi + 1], np.int64),
            1: np.array([], np.int64), 2: np.array([s_hi + 50], np.int64)}
    rec_e = Recording(name="edges", spikes=edge, fs_raw_hz=1000.0, n_samples=1000,
                      grid=(3, 0, None), source={})
    sub = SimpleNamespace(members=(2, 0, 1), center=2)
    xe = np.arange(Wn, dtype=np.float32) * 1.5
    n = _check_window(dev.fig_window(rec_e, sub, xe, i, Wn, fs_i), rec_e, sub, xe, i, Wn, fs_i)
    # by hand: t0 = 15 / 50 = 300 / 1000 and t1 = 0.3 + 0.1 = 400 / 1000 in float64, so the
    # span holds the samples 300, 301 and 399 of electrode 0 and nothing of the others
    assert (t0, t1) == (0.3, 0.4) and n == 3, (t0, t1, n)
    fig = dev.fig_window(rec_e, sub, xe, i, Wn, fs_i)
    r0 = _by_gid(fig.axes[0].collections, "raster-0")
    assert np.array_equal(np.sort(r0.get_positions()), np.array([300, 301, 399]) / 1000.0)
    # refusals: x of another length than W, an unknown electrode
    _refused(lambda: dev.fig_window(rec_e, sub, xe[:-1], i, Wn, fs_i), ["shape", "5"])
    _refused(lambda: dev.fig_window(rec_e, SimpleNamespace(members=(0, 7), center=0), xe, i, Wn,
                                    fs_i), ["7"])
    print("[T46b] %d windows of %d subregions checked, %d spikes drawn; the span's ends: %d of "
          "6 edge spikes drawn" % (checked, len(subs), total, n))


def _run_main(argv):
    dev = _device()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = dev.main(argv)
    return code, out.getvalue(), err.getvalue()


def _static_problems(path):
    """T46c's static check: imports of scipy, sklearn, statistics or pandas, and calls
    named mean, std, var, median, quantile, percentile, average, histogram or bincount."""
    banned_mod = {"scipy", "sklearn", "statistics", "pandas"}
    banned_call = {"mean", "std", "var", "median", "quantile", "percentile", "average",
                   "histogram", "bincount"}
    with open(path) as fh:
        tree = ast.parse(fh.read())
    probs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in banned_mod:
                    probs.append("line %d imports %s" % (node.lineno, a.name))
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in banned_mod:
                probs.append("line %d imports from %s" % (node.lineno, node.module))
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute)
                                                         else None)
            if name in banned_call:
                probs.append("line %d calls %s" % (node.lineno, name))
    # style() is applied by the command, not at import: every call of style is in main
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for fn in funcs:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    node.func.attr == "style" and fn.name != "main":
                probs.append("line %d: style() called in %s, not in main" % (node.lineno,
                                                                             fn.name))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and \
                        sub.func.attr == "style":
                    probs.append("line %d: style() called at import" % sub.lineno)
    return probs


def _png(path):
    with open(path, "rb") as fh:
        return fh.read(8) == PNG_SIGNATURE


def test_t46c_command():
    """T46c: the command writes PNG files with the PNG signature; device.py imports no statistics
    library and calls no mean, std, var, median, quantile, percentile, average, histogram or
    bincount (static)."""
    dev = _device()
    # the static part first: it needs nothing
    probs = _static_problems(dev.__file__)
    assert not probs, "device.py: %s" % "; ".join(probs)
    # the static check itself catches what it is meant to catch
    probe = os.path.join(_tmpdir("f46s"), "probe.py")
    with open(probe, "w") as fh:
        fh.write("import numpy as np\nfrom scipy import stats\ndef f(a):\n    return "
                 "np.mean(a) + a.std()\n")
    assert len(_static_problems(probe)) == 3, _static_problems(probe)
    _need_extractor()
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("the static part passes; a real bundle needs torch")
    tw = _wtest()
    fx = _fixture()
    work = _tmpdir("f46c")
    ext = _extraction()
    path_ext = tw._real_bundle(work, ext, WIN)
    path_none = tw._real_bundle(work, None, WIN)
    opts = ["--option", "fs_raw_hz=%r" % FS, "--option", "grid_width=%d" % GW,
            "--option", "index_base=%d" % BASE, "--option", "pitch_um=%r" % PITCH]
    base = ["--bundle", path_ext, "--recording", fx["folder"], "--adapter", "ptrain"] + opts

    # the device only, then the device and one window
    out1 = os.path.join(work, "o1")
    code, so, se = _run_main(base + ["--out", out1])
    assert code == 0 and so.strip().splitlines()[-1] == "DEVICE FIGURES OK: 1 file(s)", (
        code, so, se)
    assert sorted(os.listdir(out1)) == ["device.png"] and _png(os.path.join(out1, "device.png"))
    out2 = os.path.join(work, "o2")
    code, so, se = _run_main(base + ["--out", out2, "--subregion", "1", "--window", "13"])
    assert code == 0 and so.strip().splitlines()[-1] == "DEVICE FIGURES OK: 2 file(s)", (
        code, so, se)
    assert sorted(os.listdir(out2)) == ["device.png", "window_s1_w13.png"], os.listdir(out2)
    for name in os.listdir(out2):
        assert _png(os.path.join(out2, name)), "%s has no PNG signature" % name
    from matplotlib.image import imread
    h, w = imread(os.path.join(out2, "device.png")).shape[:2]
    assert (w, h) != (0, 0)

    # refusals: exit 3 with one REFUSED: line and nothing written
    def refused(argv, words):
        out = os.path.join(work, "r%d" % len(os.listdir(work)))
        code, so, se = _run_main(argv + ["--out", out])
        lines = [ln for ln in se.splitlines() if ln.startswith("REFUSED:")]
        assert code == 3 and len(lines) == 1 and all(w in lines[0] for w in words) and \
            not os.path.exists(out), (argv[-4:], code, so, se)
    refused(base + ["--subregion", "2", "--window", "0"], ["subregion 2"])
    refused(base + ["--subregion", "-1", "--window", "0"], ["subregion -1"])
    refused(base + ["--subregion", "0", "--window", "14"], ["window 14"])
    refused(["--bundle", path_none, "--recording", fx["folder"], "--adapter", "ptrain"] + opts,
            ["extraction"])
    refused(["--bundle", os.path.join(work, "absent.sbibundle"), "--recording", fx["folder"],
             "--adapter", "ptrain"] + opts, ["absent.sbibundle"])
    refused(base[:-8] + ["--option", "fs_raw_hz=20000.0"] + opts[2:], ["20000.0", "10000.0"])
    # a recording that gives no figure: a device.FigureError is a refusal too
    from unittest import mock
    with mock.patch.object(dev, "positions",
                           side_effect=dev.FigureError("no place for the electrodes")):
        refused(base, ["no place"])
    # the arguments: exit 2
    for argv in (base + ["--out", work, "--subregion", "0"],
                 base + ["--out", work, "--window", "0"],
                 base + ["--out", work, "--option", "detection={}"],
                 base + ["--out", work, "--option", "no_such_option=1"],
                 ["--bundle", path_ext]):
        code, so, se = _run_main(argv)
        assert code == 2, (argv[-2:], code, so, se)

    # through the package's dispatcher, in a fresh interpreter
    out3 = os.path.join(work, "o3")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "sbitool", "device-figure"] + base
                       + ["--out", out3, "--subregion", "0", "--window", "0"], cwd=_REPO,
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == \
        "DEVICE FIGURES OK: 2 file(s)", (r.returncode, r.stdout[-1500:], r.stderr[-1500:])
    assert all(_png(os.path.join(out3, n)) for n in ("device.png", "window_s0_w0.png"))
    # for the orchestrator to look at (not committed): SBITOOL_FIG46_DIR=<folder>
    keep = os.environ.get("SBITOOL_FIG46_DIR")
    if keep:
        os.makedirs(keep, exist_ok=True)
        for n in os.listdir(out2):
            shutil.copy(os.path.join(out2, n), os.path.join(keep, n))
    print("[T46c] static check clean; 2 + 1 + 2 PNG files written; 7 refusals exit 3, 5 "
          "argument errors exit 2")


CHECKS = [("test_t46a_device", test_t46a_device),
          ("test_t46b_window", test_t46b_window),
          ("test_t46c_command", test_t46c_command)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    try:
        for name, fn in CHECKS:
            if only and name not in only:
                continue
            t1 = time.time()
            try:
                fn()
                rows.append((name, "PASS", "", time.time() - t1))
            except NotImplementedError as exc:
                bad += 1
                rows.append((name, "TODO", str(exc), time.time() - t1))
            except unittest.SkipTest as exc:
                rows.append((name, "SKIP", str(exc), time.time() - t1))
            except AssertionError as exc:
                bad += 1
                rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
            except Exception:
                bad += 1
                rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    finally:
        _cleanup()
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
