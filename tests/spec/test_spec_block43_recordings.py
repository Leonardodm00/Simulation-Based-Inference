"""Spec tests of block 43 (recordings and the device's layout; specs/SPEC.md
sec. 2.7, "A recording", and block 43). Integration tester, 2026-11-28,
incremental run on 39e6f66.

The inputs are written here (ptrain folders with scipy.io.savemat in the two
formats the extractor declares; SpikeInterface objects built by the
library); the expected values are the planted ones. Needs SBIX_CLONE;
spikeinterface for the si tests.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import _tool_util as U  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(U.SBIX), reason="no SBIX_CLONE")


def _trains(rng, ids, n):
    return {e: np.sort(rng.choice(n, size=int(rng.integers(0, 30)), replace=False)).astype(
        np.int64) for e in ids}


@pytest.mark.parametrize("fmt", ["raster", "sparse_peaks"])
@pytest.mark.parametrize("base", [0, 1])
def test_ptrain_reads_back_the_planted_samples(fmt, base):
    from sbitool import recordings
    rng = np.random.default_rng(43 + base)
    n, width, pitch = 5000, 6, 60.0
    ids = [base + k for k in (0, 1, 5, 6, 7, 20, 35)]     # 35 = the last cell of a 6 x 6 grid
    tr = _trains(rng, ids, n)
    tr[ids[0]] = np.array([0, n - 1], dtype=np.int64)     # both ends of [0, n_samples)
    tr[ids[1]] = np.array([], dtype=np.int64)             # a silent electrode
    d = U.write_ptrain(tempfile.mkdtemp(prefix="p43_"), tr, n, fmt=fmt)
    rec = recordings.read(d, "ptrain", fs_raw_hz=10000.0, ptrain_format=fmt, grid_width=width,
                          index_base=base, pitch_um=pitch)
    assert sorted(rec.spikes) == sorted(ids)
    for e in ids:
        np.testing.assert_array_equal(rec.spikes[e], tr[e])
        assert rec.spikes[e].dtype == np.int64
    assert rec.n_samples == n and rec.fs_raw_hz == 10000.0
    assert rec.duration_s == n / 10000.0
    assert rec.grid == (width, base, pitch)
    # row-major linear index (the extractor's): row = (e - base) // width, col = ... % width;
    # x = column x pitch, y = row x pitch [um] (block 46's contract states the same mapping)
    for e in ids:
        i0 = e - base
        assert rec.positions_um[e] == ((i0 % width) * pitch, (i0 // width) * pitch), e
    # without a pitch: no positions, no grid (block 43's build note)
    r2 = recordings.read(d, "ptrain", fs_raw_hz=10000.0, ptrain_format=fmt, grid_width=width,
                         index_base=base)
    assert r2.positions_um is None and r2.grid is None


def test_ptrain_refusals():
    from sbitool import recordings
    rng = np.random.default_rng(431)
    tr = _trains(rng, [0, 1, 2], 1000)
    d = U.write_ptrain(tempfile.mkdtemp(prefix="p43r_"), tr, 1000, fmt="raster")
    # the other format: refused
    with pytest.raises(recordings.RecordingError):
        recordings.read(d, "ptrain", fs_raw_hz=1e4, ptrain_format="sparse_peaks",
                        grid_width=6, index_base=0)
    # a missing rate (or grid) is a refusal naming it: never guessed
    for miss in ("fs_raw_hz", "grid_width", "index_base"):
        kw = dict(fs_raw_hz=1e4, grid_width=6, index_base=0)
        kw.pop(miss)
        with pytest.raises(recordings.RecordingError) as ei:
            recordings.read(d, "ptrain", **kw)
        assert miss in str(ei.value)
    # electrodes of unequal length
    U.write_ptrain(d, {3: np.array([1], dtype=np.int64)}, 999, fmt="raster")
    with pytest.raises(recordings.RecordingError):
        recordings.read(d, "ptrain", fs_raw_hz=1e4, grid_width=6, index_base=0)
    # an index outside the grid (base 1 with an electrode file 0)
    d2 = U.write_ptrain(tempfile.mkdtemp(prefix="p43g_"), {0: [1], 1: [2]}, 100)
    with pytest.raises(recordings.RecordingError):
        recordings.read(d2, "ptrain", fs_raw_hz=1e4, grid_width=6, index_base=1)
    # an unknown adapter
    with pytest.raises(recordings.RecordingError):
        recordings.read(d2, "neuroscope")


def _si():
    return pytest.importorskip("spikeinterface")


def test_si_sorting_and_probe_units():
    _si()
    import probeinterface
    from spikeinterface.core import NpzSortingExtractor, NumpyRecording, NumpySorting
    from sbitool import recordings
    from sbitool.schema import DETECTION_DEFAULT
    rng = np.random.default_rng(4310)
    fs, n = 30000.0, 20000
    tr = {e: np.sort(rng.choice(n, size=10, replace=False)).astype(np.int64)
          for e in (2, 9, 40)}
    s = NumpySorting.from_unit_dict([tr], sampling_frequency=fs)
    p = os.path.join(tempfile.mkdtemp(prefix="s43_"), "s.npz")
    NpzSortingExtractor.write_sorting(s, p)
    rec = recordings.read(p, "spikeinterface", reader="npz", n_samples=n)
    for e in tr:
        np.testing.assert_array_equal(rec.spikes[e], tr[e])
    assert rec.n_samples == n and rec.fs_raw_hz == fs
    # a probe in mm: positions in um (x 1000; block 43's build note: probes in um or mm)
    xy_mm = np.array([[0.0, 0.0], [0.06, 0.0], [0.0, 0.06], [0.06, 0.06]])
    traces = rng.normal(0.0, 1.0, size=(3000, 4)).astype(np.float32)
    raw = NumpyRecording([traces], sampling_frequency=fs, channel_ids=np.array([0, 1, 2, 3]))
    probe = probeinterface.Probe(ndim=2, si_units="mm")
    probe.set_contacts(positions=xy_mm, shapes="circle", shape_params={"radius": 0.005})
    probe.set_device_channel_indices(np.arange(4))
    raw.set_probe(probe, in_place=True)
    folder = os.path.join(tempfile.mkdtemp(prefix="s43p_"), "raw")
    raw.save(folder=folder, n_jobs=1, progress_bar=False, verbose=False)
    r = recordings.read(folder, "spikeinterface", reader="binaryfolder",
                        detection=dict(DETECTION_DEFAULT))
    got = np.array([r.positions_um[e] for e in sorted(r.positions_um)])
    # 1e-9 um: x 1000 of a float in mm, a few ulps
    np.testing.assert_allclose(got, xy_mm * 1000.0, atol=1e-9)
    # a raw file without a detection is refused
    with pytest.raises(recordings.RecordingError):
        recordings.read(folder, "spikeinterface", reader="binaryfolder")


def test_si_raw_detection_exclude_sweep():
    """D-176: negative peaks below detect_threshold x the channel's MAD noise, at least
    exclude_sweep_ms apart on one channel. Two planted troughs 0.5 ms apart on one channel
    give one peak; two 3 ms apart give two; a positive peak gives none."""
    _si()
    from spikeinterface.core import NumpyRecording
    from sbitool.recordings import si
    fs = 20000.0
    rng = np.random.default_rng(4320)
    x = rng.normal(0.0, 1.0, size=(40000, 2)).astype(np.float32)
    shape = np.array([-0.3, -1.0, -0.3], dtype=np.float32)
    close = [10000, 10010]          # 10 samples = 0.5 ms < 1.0 ms
    apart = [20000, 20060]          # 60 samples = 3 ms
    for t in close + apart:
        x[t - 1:t + 2, 0] += 30.0 * shape
    x[30000 - 1:30000 + 2, 1] -= 30.0 * shape     # a positive peak on channel 1
    raw = NumpyRecording([x], sampling_frequency=fs, channel_ids=np.array([0, 1]))
    det = {"method": "by_channel", "peak_sign": "neg", "detect_threshold": 5.0,
           "exclude_sweep_ms": 1.0}
    r = si.from_recording(raw, det)
    got0 = r.spikes[0]
    near = lambda t: np.any(np.abs(got0 - t) <= 1)   # noqa: E731  (the trough, +-1 sample)
    assert near(apart[0]) and near(apart[1])
    assert sum(near(t) for t in close) == 1, got0
    assert got0.size == 3, got0                    # nothing else above 5 MADs of N(0, 1)
    assert r.spikes[1].size == 0, r.spikes[1]


def test_as_grid():
    from sbitool.recordings import layout
    rng = np.random.default_rng(4330)
    pitch, tol = 60.0, 1.0
    cells = [(r, c) for r in range(4) for c in range(4)]
    # jittered within the tolerance, offset origin, ids not in grid order
    ids = rng.permutation(len(cells)) + 100
    pos = {int(ids[i]): (500.0 + c * pitch + rng.uniform(-0.4, 0.4),
                         -30.0 + r * pitch + rng.uniform(-0.4, 0.4))
           for i, (r, c) in enumerate(cells)}
    width, p, lin = layout.as_grid(pos, tol)
    assert width == 4 and abs(p - pitch) < 0.5      # the fitted pitch within the jitter
    for i, (r, c) in enumerate(cells):
        assert lin[int(ids[i])] == r * width + c
    # a rectangular 2 x 5 patch: the smallest square grid that holds it, width 5
    pos2 = {i: (c * pitch, r * pitch) for i, (r, c) in
            enumerate([(r, c) for r in range(2) for c in range(5)])}
    w2, _, lin2 = layout.as_grid(pos2, tol)
    assert w2 == 5 and sorted(lin2.values()) == list(range(10))
    # a hexagonal lattice and a jitter beyond the tolerance: LayoutError
    hexa = {i: ((c + 0.5 * (r % 2)) * pitch, r * pitch * np.sqrt(3) / 2)
            for i, (r, c) in enumerate(cells)}
    with pytest.raises(layout.LayoutError):
        layout.as_grid(hexa, tol)
    bad = dict(pos)
    k0 = next(iter(bad))
    bad[k0] = (bad[k0][0] + 5.0, bad[k0][1])
    with pytest.raises(layout.LayoutError):
        layout.as_grid(bad, tol)
    # round trip with ptrain's positions: the linear index of a full 6 x 6 grid
    rec = U.grid_recording({e: [] for e in range(36)}, 100, width=6, pitch=60.0)
    w3, p3, lin3 = layout.as_grid(rec.positions_um, tol)
    assert w3 == 6 and p3 == pytest.approx(60.0, abs=1e-9)
    assert lin3 == {e: e for e in range(36)}
    assert layout.grid(rec) == {e: (e // 6, e % 6) for e in range(36)}


# ============ fix round of 2026-11-28 (the tester's F6): positions_um as a JSON object
# [2026-11-28_0151]

def _npz_sorting(rng, ids, n, fs=10000.0):
    from spikeinterface.core import NpzSortingExtractor, NumpySorting
    tr = {e: np.sort(rng.choice(n, size=int(rng.integers(20, 80)), replace=False)).astype(
        np.int64) for e in ids}
    s = NumpySorting.from_unit_dict([tr], sampling_frequency=fs)
    p = os.path.join(tempfile.mkdtemp(prefix="s43j_"), "s.npz")
    NpzSortingExtractor.write_sorting(s, p)
    return p, tr


def test_si_positions_um_json_object():
    """Block 43, Rule (fix round): a positions_um option that is a JSON object (string keys,
    list pairs) becomes {int: (x, y) floats} before Recording.check; a key that is not an
    integer's text, or a value that is not two finite numbers, is refused naming it; a
    non-object is refused; two keys naming one electrode are refused; a key with a leading
    minus parses and then fails Recording.check (another electrode than the spikes')."""
    _si()
    import json
    from sbitool import recordings
    rng = np.random.default_rng(4320)
    p, tr = _npz_sorting(rng, range(9), 20000)
    pos = json.loads(json.dumps({str(e): [60 * (e % 3), 60 * (e // 3)] for e in range(9)}))
    rec = recordings.read(p, "spikeinterface", reader="npz", n_samples=20000, positions_um=pos)
    assert set(rec.positions_um) == set(range(9))
    for e in range(9):
        v = rec.positions_um[e]
        assert isinstance(v, tuple) and all(type(u) is float for u in v)
        assert v == (float(60 * (e % 3)), float(60 * (e // 3)))
        np.testing.assert_array_equal(rec.spikes[e], tr[e])
    assert rec.check() == []
    bad = {
        "a key not an integer's text": ({**pos, "x1": [0, 0]}, "x1"),
        "a float key text": (dict((("%d.0" % e) if e == 4 else str(e), v) for e, v in
                                  ((int(k), v) for k, v in pos.items())), "4.0"),
        "three numbers": ({**pos, "4": [0, 0, 0]}, "4"),
        "a NaN": ({**pos, "4": [float("nan"), 0]}, "4"),
        "an inf": ({**pos, "4": [0, float("inf")]}, "4"),
        "a string number": ({**pos, "4": ["60", 0]}, "4"),
        "a bool": ({**pos, "4": [True, 0]}, "4"),
        "an int beyond float range": ({**pos, "4": [10 ** 400, 0]}, "4"),
        "two keys for one electrode": ({**pos, "04": [0, 0]}, "04"),
        "not an object": ([[0, 0]] * 9, "positions_um"),
        "a leading minus": ({**dict((k, v) for k, v in pos.items() if k != "4"), "-4": [0, 0]},
                            "positions_um"),
    }
    passed = []
    for what, (opt, word) in bad.items():
        try:
            recordings.read(p, "spikeinterface", reader="npz", n_samples=20000,
                            positions_um=opt)
            passed.append(what)
        except recordings.RecordingError as exc:
            assert word in str(exc), (what, str(exc))
    assert not passed, passed


def test_si_positions_um_through_the_windows_command():
    """T43b's route (fix round): `windows --option positions_um=<JSON>` on a sorting without
    a probe exits 0, and gives the windows of the same device given with a grid (D-177: one
    device, one partition), computed by `windows.extract` on a Recording built here."""
    _si()
    import json
    import subprocess
    from sbitool import windows
    rng = np.random.default_rng(4321)
    n = 40000
    p, tr = _npz_sorting(rng, range(9), n)
    d = os.path.dirname(p)
    ext = U.extraction(n_subsets=1, electrodes_per_subset=9, grid_width=3)
    b = U.mini_bundle(os.path.join(d, "b.sbibundle"), ext, W=10)
    pos = {str(e): [60.0 * (e % 3), 60.0 * (e // 3)] for e in range(9)}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SBIX_CLONE=U.SBIX)
    out = os.path.join(d, "w.npz")
    r = subprocess.run([sys.executable, "-m", "sbitool", "windows", "--bundle", b,
                        "--recording", p, "--adapter", "spikeinterface", "--option", "reader=npz",
                        "--option", "n_samples=%d" % n, "--option",
                        "positions_um=" + json.dumps(pos), "--out", out], cwd=U.REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "WINDOWS OK" in r.stdout + r.stderr, r.stdout + r.stderr
    rec = U.grid_recording(tr, n, width=3)
    w, ids, _ = windows.extract(rec, ext, 10)
    assert w.shape[0] == 40                      # floor(400 / 10) windows of one subregion
    with np.load(out) as z:
        np.testing.assert_array_equal(z["windows"], w)
        np.testing.assert_array_equal(z["window_idx"], ids["window_idx"])
    # a bad key through the command: exit 3, one REFUSED line, nothing written
    pos2 = dict(pos)
    pos2["e4"] = pos2.pop("4")
    out2 = os.path.join(d, "w2.npz")
    r = subprocess.run([sys.executable, "-m", "sbitool", "windows", "--bundle", b,
                        "--recording", p, "--adapter", "spikeinterface", "--option", "reader=npz",
                        "--option", "n_samples=%d" % n, "--option",
                        "positions_um=" + json.dumps(pos2), "--out", out2], cwd=U.REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    txt = r.stdout + r.stderr
    assert r.returncode == 3 and "REFUSED" in txt and "e4" in txt and not os.path.exists(out2), txt
