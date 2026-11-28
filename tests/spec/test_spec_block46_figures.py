"""Spec tests of block 46 (the device and the window as figures; specs/SPEC.md
block 46 with its wave-C contract and amendment 1). Integration tester,
2026-11-28, incremental run on 39e6f66.

Every expected value is read from the inputs the test builds (positions,
spikes, the window's values) and from block 21's colours in ev_figures (the
spec's source of the colours), then compared with what the artists hold.
Needs SBIX_CLONE for the command's check.
"""

import ast
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import pytest
from matplotlib.colors import to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import _tool_util as U  # noqa: E402

EVAL = os.path.join(U.REPO, "hpc", "pipeline", "evaluation")


def _colors():
    sys.path.insert(0, EVAL)
    import ev_figures as ev
    return ev, (ev.ARM_COLOR["A1"], ev.ARM_COLOR["A2"], ev.ARM_COLOR["A5"], ev.OTHER_ARM,
                ev.COND_COLOR[0], ev.COND_COLOR[2])


def _sub(center, members):
    return types.SimpleNamespace(center=center, members=tuple(members))


def _artist(fig, gid):
    out = [a for ax in fig.axes for a in ax.get_children() if a.get_gid() == gid]
    out += [a for ax in fig.axes for a in ax.texts if a.get_gid() == gid and a not in out]
    return out


def _rec36(silent=(5, 30)):
    rng = np.random.default_rng(4601)
    sp = {e: (np.array([], dtype=np.int64) if e in silent else
              np.sort(rng.choice(20000, size=20, replace=False))) for e in range(36)}
    return U.grid_recording(sp, 20000, width=6, pitch=60.0), sp


def test_t46a_device_figure():
    from sbitool.plotting import device
    ev, cols = _colors()
    rec, sp = _rec36()
    subs = [_sub(7, (7, 0, 1, 6)), _sub(28, (28, 29, 34, 35, 30))]   # 30 is silent
    fig = device.fig_device(rec, subs, highlight=1)
    [sc] = _artist(fig, "electrodes")
    off = sc.get_offsets()
    ids = sorted(sp)
    # one marker per electrode in sorted id order at x = col pitch, y = row pitch [um]
    np.testing.assert_array_equal(off, [((e % 6) * 60.0, (e // 6) * 60.0) for e in ids])
    owner = {m: j for j, s in enumerate(subs) for m in s.members}
    ec, fc = sc.get_edgecolors(), sc.get_facecolors()
    assert ec.shape[0] == 36 and fc.shape[0] == 36
    for i, e in enumerate(ids):
        want = to_rgba(cols[owner[e] % 6] if e in owner else ev.SIM_GRAY)
        np.testing.assert_allclose(ec[i], want, atol=1e-12)
        if sp[e].size == 0:
            assert fc[i][3] == 0.0, e                    # hollow: no face colour
        else:
            np.testing.assert_allclose(fc[i], want, atol=1e-12)
    # the index beside each subregion's centre: xy the centre's position, offset (6, 6) pt
    for j, s in enumerate(subs):
        [a] = _artist(fig, "subregion-%d" % j)
        assert a.get_text() == str(j)
        assert tuple(a.xy) == ((s.center % 6) * 60.0, (s.center // 6) * 60.0)
        assert tuple(a.xyann) == (6, 6) and a.anncoords == "offset points"
    # the ring: subregion 1's electrodes
    [ring] = _artist(fig, "highlight")
    np.testing.assert_array_equal(ring.get_offsets(),
                                  [((m % 6) * 60.0, (m // 6) * 60.0) for m in subs[1].members])
    ax = fig.axes[0]
    assert ax.get_xlabel() == "x [um]" and ax.get_ylabel() == "y [um]"
    assert ax.get_aspect() == 1.0
    # the seventh subregion takes the first colour again (j mod 6)
    subs7 = [_sub(e, (e,)) for e in (0, 2, 4, 12, 14, 16, 24)]
    [sc7] = _artist(device.fig_device(rec, subs7), "electrodes")
    np.testing.assert_allclose(sc7.get_edgecolors()[24], to_rgba(cols[0]), atol=1e-12)
    # a grid without a pitch: column and row, no um
    rec_np = U.grid_recording(sp, 20000, width=6, pitch=None)
    f2 = device.fig_device(rec_np, subs)
    [sc2] = _artist(f2, "electrodes")
    np.testing.assert_array_equal(sc2.get_offsets(), [(e % 6, e // 6) for e in ids])
    assert f2.axes[0].get_xlabel() == "column" and f2.axes[0].get_ylabel() == "row"
    # neither positions nor a grid: FigureError
    from sbitool.interfaces import Recording
    bare = Recording(name="b", spikes=dict(sp), fs_raw_hz=1e4, n_samples=20000)
    with pytest.raises(device.FigureError):
        device.fig_device(bare, subs)


def test_t46b_window_figure_half_open_span():
    from sbitool.plotting import device
    W, fs_ifr, fs = 50, 100.0, 10000.0
    j = 2                                     # the window [1.0, 1.5) s
    t0, t1 = j * W / fs_ifr, (j + 1) * W / fs_ifr
    sp = {e: np.array([], dtype=np.int64) for e in range(36)}
    # electrode 0: at t0 exactly (in), at t1 exactly (out), one inside, one before
    sp[0] = np.array([9999, 10000, 12345, 15000], dtype=np.int64)
    sp[1] = np.array([14999, 15001], dtype=np.int64)
    sp[6] = np.array([3], dtype=np.int64)
    rec = U.grid_recording(sp, 30000)
    sub = _sub(1, (1, 0, 6))
    x = np.linspace(0.0, 3.0, W).astype(np.float32)
    fig = device.fig_window(rec, sub, x, j, W, fs_ifr)
    want = {0: [10000 / fs, 12345 / fs], 1: [14999 / fs], 6: []}
    for e, w in want.items():
        [c] = _artist(fig, "raster-%d" % e)
        np.testing.assert_array_equal(np.sort(c.get_positions()), w)
    [st] = _artist(fig, "ifr")
    d = st.get_data()
    np.testing.assert_array_equal(d.values, x)                 # the values unchanged
    np.testing.assert_allclose(d.edges, t0 + np.arange(W + 1) / fs_ifr, rtol=0, atol=1e-12)
    assert d.baseline is None
    labels = [ax.get_ylabel() for ax in fig.axes] + [fig.axes[1].get_xlabel()]
    assert "electrode" in labels and "IFR [Hz per electrode]" in labels and "time [s]" in labels
    with pytest.raises(device.FigureError):
        device.fig_window(rec, sub, x[:-1], j, W, fs_ifr)


def test_t46c_static_no_statistics():
    src = open(os.path.join(U.REPO, "sbitool", "plotting", "device.py")).read()
    tree = ast.parse(src)
    banned_mod = {"scipy", "sklearn", "statistics", "pandas"}
    banned_call = {"mean", "std", "var", "median", "quantile", "percentile", "average",
                   "histogram", "bincount", "nanmean", "nanmedian", "nanstd"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] in banned_mod for a in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned_mod
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            assert name not in banned_call, name


@pytest.mark.skipif(not os.path.isdir(U.SBIX), reason="no SBIX_CLONE")
def test_t46c_command_writes_pngs():
    d = tempfile.mkdtemp(prefix="f46_")
    rng = np.random.default_rng(4603)
    n = 20000
    sp = {e: np.sort(rng.choice(n, size=int(rng.integers(20, 60)), replace=False))
          for e in range(36)}
    U.write_ptrain(os.path.join(d, "rec"), sp, n)
    b = U.mini_bundle(os.path.join(d, "b.sbibundle"),
                      U.extraction(n_subsets=2, electrodes_per_subset=9), W=10)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SBIX_CLONE=U.SBIX)
    opts = ["--option", "fs_raw_hz=10000.0", "--option", "grid_width=6", "--option",
            "index_base=0", "--option", "pitch_um=60.0"]
    base = [sys.executable, "-m", "sbitool", "device-figure", "--bundle", b, "--recording",
            os.path.join(d, "rec"), "--adapter", "ptrain"] + opts
    r = subprocess.run(base + ["--out", os.path.join(d, "o"), "--subregion", "1", "--window",
                               "3"], cwd=U.REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "DEVICE FIGURES OK: 2 file(s)" in r.stdout, r.stdout + r.stderr
    for f in ("device.png", "window_s1_w3.png"):
        with open(os.path.join(d, "o", f), "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    # a window that does not exist: exit 3, one REFUSED line, nothing written
    r = subprocess.run(base + ["--out", os.path.join(d, "o2"), "--subregion", "1", "--window",
                               "20"], cwd=U.REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 3 and r.stderr.count("REFUSED:") == 1, r.stdout + r.stderr
    assert not os.path.exists(os.path.join(d, "o2"))
    r = subprocess.run(base + ["--out", os.path.join(d, "o3"), "--subregion", "5", "--window",
                               "0"], cwd=U.REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 3 and not os.path.exists(os.path.join(d, "o3"))
    # --subregion without --window: an argument error
    r = subprocess.run(base + ["--out", os.path.join(d, "o4"), "--subregion", "1"],
                       cwd=U.REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 2
