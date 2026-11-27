#!/usr/bin/env python3
"""Smoke test of block 43: recordings and the device's layout (specs/SPEC.md block 43, "Test
oracles", checks T43*).

CONTRACT SKELETON (wave A): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 43's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t43a_ptrain       the two spike-train formats read back exactly; the refusals
    test_t43b_si_sorting   a SpikeInterface sorting read back exactly; the probe's positions
    test_t43c_si_raw       raw voltage: known spikes found by the declared detection
    test_t43d_layout       as_grid on a jittered grid, LayoutError otherwise; the round trip

Run
    python3 sbitool/tests/test_smoke_tool_recordings.py [check ...]
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


import numpy as np


def _need_extractor():
    """SkipTest naming SBIX_DIR when no extractor checkout is named."""
    from sbitool import _sources
    for var in _sources.EXTRACTOR_ENV:
        v = os.environ.get(var)
        if v and os.path.isdir(v):
            return
    raise unittest.SkipTest("the Sbi-extractor's checkout is not named: set SBIX_DIR (or "
                            "SBIX_CLONE in the tests)")


def _need_si():
    try:
        import spikeinterface  # noqa: F401
        import probeinterface  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("spikeinterface / probeinterface not installed (%s)" % exc)


def _refused(fn, *words):
    """fn() must raise RecordingError whose message holds every word."""
    from sbitool.recordings import RecordingError
    try:
        fn()
    except RecordingError as exc:
        msg = str(exc)
        for w in words:
            assert w in msg, "the refusal does not name %r: %s" % (w, msg)
        return msg
    raise AssertionError("not refused (expected a RecordingError naming %s)" % (words,))


# the ptrain fixture: a full 6 x 6 grid, linear indices 1..36 (index_base 1)
GW, BASE, PITCH, FS, NSAMP = 6, 1, 60.0, 10000.0, 5000


def _trains(rng):
    """{electrode id: sorted unique int64 samples}; electrode 1 is silent and
    electrode 36 has a spike at the last sample (the interval's closed end)."""
    out = {}
    for e in range(BASE, BASE + GW * GW):
        k = 0 if e == BASE else int(rng.integers(1, 25))
        out[e] = np.sort(rng.choice(NSAMP, size=k, replace=False)).astype(np.int64)
    out[BASE + GW * GW - 1] = np.union1d(out[BASE + GW * GW - 1], [0, NSAMP - 1]).astype(np.int64)
    return out


def _write_raster(folder, trains, n=NSAMP, varname="ptrain"):
    import scipy.io as sio
    os.makedirs(folder, exist_ok=True)
    for e, t in trains.items():
        r = np.zeros((n, 1), dtype=np.uint8)
        r[t, 0] = 1
        sio.savemat(os.path.join(folder, "ptrain_%d.mat" % e), {varname: r}, format="5",
                    do_compression=True)


def _write_sparse(folder, trains, rng, n=NSAMP, varname="peak_train"):
    import scipy.io as sio
    from scipy import sparse
    os.makedirs(folder, exist_ok=True)
    for e, t in trains.items():
        amp = -rng.uniform(20.0, 80.0, size=t.size)   # the values are ignored by the format
        m = sparse.csc_matrix((amp, (t, np.zeros(t.size, np.int64))), shape=(n, 1))
        sio.savemat(os.path.join(folder, "well_A1_%d.mat" % e), {varname: m}, format="5")


def _expected_xy(e):
    # the row-major decoding of EXTRACTOR_USAGE.md sec. 15.2, computed here
    i0 = e - BASE
    return (float((i0 % GW) * PITCH), float((i0 // GW) * PITCH))


def _same_trains(rec, trains, tag):
    assert sorted(rec.spikes) == sorted(trains), "%s: electrodes %r != %r" % (
        tag, sorted(rec.spikes), sorted(trains))
    for e, t in trains.items():
        got = rec.spikes[e]
        # exact: sample indices are integers, nothing may be resampled or shifted
        assert got.dtype == np.int64 and np.array_equal(got, t), \
            "%s: electrode %d: %r != %r" % (tag, e, got[:8], t[:8])


def test_t43a_ptrain():
    """the two spike-train formats read back exactly; the refusals (specs/SPEC.md block 43)."""
    _need_extractor()
    from sbitool import recordings
    rng = np.random.default_rng(SEED)
    trains = _trains(rng)
    d_r = os.path.join(_tmpdir("t43a"), "raster")
    d_s = os.path.join(_tmpdir("t43a"), "sparse")
    _write_raster(d_r, trains)
    _write_sparse(d_s, trains, rng)
    pat = r"^well_A1_(\d+)\.mat$"
    common = dict(fs_raw_hz=FS, grid_width=GW, index_base=BASE, pitch_um=PITCH)
    r1 = recordings.read(d_r, "ptrain", ptrain_format="raster", **common)
    r2 = recordings.read(d_s, "ptrain", ptrain_format="sparse_peaks", ptrain_varname="peak_train",
                         ptrain_name_pattern=pat, **common)
    for rec, tag in ((r1, "raster"), (r2, "sparse_peaks")):
        _same_trains(rec, trains, tag)
        assert rec.n_samples == NSAMP and rec.fs_raw_hz == FS, (rec.n_samples, rec.fs_raw_hz)
        # exact: duration is one float64 division done the same way
        assert rec.duration_s == NSAMP / FS
        # exact: (col, row) times the pitch, products of small integers and 60.0
        exp = {e: _expected_xy(e) for e in trains}
        assert rec.positions_um == exp, "%s: positions differ, e.g. %r vs %r" % (
            tag, rec.positions_um.get(2), exp[2])
        assert rec.grid == (GW, BASE, PITCH), rec.grid
        assert rec.check() == []
        assert rec.source["adapter"] == "ptrain"
    # without the pitch: no geometry, trains unchanged
    r3 = recordings.read(d_r, "ptrain", fs_raw_hz=FS, grid_width=GW, index_base=BASE)
    assert r3.positions_um is None and r3.grid is None
    _same_trains(r3, trains, "raster without pitch")
    # refusals
    _refused(lambda: recordings.read(d_r, "ptrain", ptrain_format="sparse_peaks", **common),
             "sparse_peaks", "raster")   # a file of the other format, the extractor's message
    _refused(lambda: recordings.read(d_s, "ptrain", ptrain_format="raster",
                                     ptrain_varname="peak_train", ptrain_name_pattern=pat,
                                     **common), "sparse")
    _refused(lambda: recordings.read(d_r, "ptrain", grid_width=GW, index_base=BASE,
                                     pitch_um=PITCH), "fs_raw_hz")
    _refused(lambda: recordings.read(d_r, "ptrain", fs_raw_hz=FS, index_base=BASE,
                                     pitch_um=PITCH), "grid_width")
    _refused(lambda: recordings.read(d_r, "ptrain", fs_raw_hz=FS, grid_width=GW,
                                     pitch_um=PITCH), "index_base")
    # index_base 0 puts linear index 36 on row 6 of a 6-wide grid: the extractor refuses
    _refused(lambda: recordings.read(d_r, "ptrain", fs_raw_hz=FS, grid_width=GW, index_base=0,
                                     pitch_um=PITCH), "outside")
    d_u = os.path.join(_tmpdir("t43a"), "unequal")
    _write_raster(d_u, {1: trains[2], 2: trains[3]})
    _write_raster(d_u, {3: np.array([5, 9], np.int64)}, n=NSAMP + 1)
    _refused(lambda: recordings.read(d_u, "ptrain", **common), "n_samples")
    _refused(lambda: recordings.read(d_r, "nosuch"), "nosuch", "ptrain", "spikeinterface")


def test_t43b_si_sorting():
    """a SpikeInterface sorting read back exactly; the probe's positions (specs/SPEC.md block 43)."""
    _need_si()
    import probeinterface
    from spikeinterface.core import NpzSortingExtractor, NumpyRecording, NumpySorting
    from sbitool import recordings
    from sbitool.recordings import si
    from sbitool.schema import DETECTION_DEFAULT
    rng = np.random.default_rng(SEED + 1)
    fs, n = 20000.0, 30000
    trains = {e: np.sort(rng.choice(n, size=int(rng.integers(1, 40)), replace=False)).astype(
        np.int64) for e in (0, 3, 4, 11, 17)}
    sorting = NumpySorting.from_unit_dict([trains], sampling_frequency=fs)
    path = os.path.join(_tmpdir("t43b"), "sorting.npz")
    NpzSortingExtractor.write_sorting(sorting, path)
    rec = recordings.read(path, "spikeinterface", reader="npz", n_samples=n)
    _same_trains(rec, trains, "npz sorting")
    assert rec.fs_raw_hz == fs and rec.n_samples == n and rec.positions_um is None
    assert rec.source["reader"] == "npz" and rec.source["kind"] == "sorting"
    _refused(lambda: recordings.read(path, "spikeinterface", reader="npz"), "n_samples")
    _refused(lambda: recordings.read(path, "spikeinterface"), "reader")
    _refused(lambda: recordings.read(path, "spikeinterface", reader="npz", n_samples=n,
                                     detection=dict(DETECTION_DEFAULT)), "detection")
    # the unit's channel property names the electrode; two units on one electrode are refused
    s2 = NumpySorting.from_unit_dict([{100: trains[0], 101: trains[3]}], sampling_frequency=fs)
    s2.set_property("channel", np.array([7, 9]))
    r2 = si.from_sorting(s2, n)
    assert sorted(r2.spikes) == [7, 9] and np.array_equal(r2.spikes[9], trains[3])
    s2.set_property("channel", np.array([7, 7]))
    _refused(lambda: si.from_sorting(s2, n), "electrode 7")
    # a recording with a probe gives the probe's positions [um]
    xy = np.array([[0.0, 0.0], [17.5, 0.0], [0.0, 42.0], [17.5, 42.0]])
    traces = rng.normal(0.0, 10.0, size=(4000, 4)).astype(np.float32)
    raw = NumpyRecording([traces], sampling_frequency=fs, channel_ids=np.array([5, 6, 7, 8]))
    probe = probeinterface.Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=xy, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(4))
    raw.set_probe(probe)
    folder = os.path.join(_tmpdir("t43b"), "raw")
    raw.save(folder=folder, n_jobs=1, progress_bar=False, verbose=False)
    r3 = recordings.read(folder, "spikeinterface", reader="binaryfolder",
                         detection=dict(DETECTION_DEFAULT))
    # exact: the positions are copied, not computed
    assert r3.positions_um == {5 + i: (float(xy[i, 0]), float(xy[i, 1])) for i in range(4)}, \
        r3.positions_um
    assert r3.n_samples == 4000 and r3.fs_raw_hz == fs


# T43c's synthetic trace
FS_C, N_C, SIGMA_C, AMP_SD, TOL_SAMPLES = 20000.0, 40000, (5.0, 10.0, 20.0, 40.0), 15.0, 1
SHAPE_C = np.array([-0.1, -0.5, -1.0, -0.5, -0.1])   # the spike's waveform, its trough at offset 0


def _raw_c(rng):
    """(recording, planted {channel id: samples}): white noise of SD
    SIGMA_C[c] on channel c plus negative spikes of amplitude AMP_SD SDs."""
    import probeinterface
    from spikeinterface.core import NumpyRecording
    nc = len(SIGMA_C)
    x = rng.normal(0.0, 1.0, size=(N_C, nc)) * np.array(SIGMA_C)[None, :]
    planted = {}
    for c in range(nc):
        # 12 spikes, at least 100 samples (5 ms, five times exclude_sweep_ms) apart
        # and 300 samples from either end
        slots = np.sort(rng.choice((N_C - 600) // 100, size=12, replace=False))
        t = 300 + slots * 100 + rng.integers(0, 40, size=12)
        for s in t:
            x[s - 2:s + 3, c] += AMP_SD * SIGMA_C[c] * SHAPE_C
        planted[c] = t.astype(np.int64)
    rec = NumpyRecording([x.astype(np.float32)], sampling_frequency=FS_C,
                         channel_ids=np.arange(nc))
    probe = probeinterface.Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=np.array([[0.0, 0.0], [60.0, 0.0], [0.0, 60.0], [60.0, 60.0]]),
                       shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(nc))
    rec.set_probe(probe)
    return rec, planted


def _match(planted, found, tol):
    """(n matched, missed, spurious) with a found peak matched to at most one
    planted spike within tol samples."""
    used = np.zeros(found.size, bool)
    missed = 0
    for s in planted:
        d = np.abs(found - s)
        d[used] = tol + 1
        j = int(np.argmin(d)) if found.size else -1
        if j >= 0 and d[j] <= tol:
            used[j] = True
        else:
            missed += 1
    return int(used.sum()), missed, int((~used).sum())


def test_t43c_si_raw():
    """raw voltage: known spikes found by the declared detection (specs/SPEC.md block 43)."""
    _need_si()
    from sbitool.recordings import si
    from sbitool.schema import DETECTION_DEFAULT
    rng = np.random.default_rng(SEED + 2)
    raw, planted = _raw_c(rng)
    # Tolerance TOL_SAMPLES = 1: the planted trough is deeper than its neighbours
    # by 0.5 x 15 = 7.5 noise SDs, so noise moves the minimum by one sample only
    # if two independent samples differ by 7.5 SDs (5.3 SDs of their
    # difference, p ~ 6e-8 per pair); and the detector reports the trough's own
    # sample (by_channel: a local minimum within exclude_sweep_ms). Spurious
    # peaks: P(noise < -5 SD) = 2.9e-7 per sample, 1.6e5 samples: 0.05 expected.
    tot = [0, 0, 0, 0]
    for det, expect_all in ((dict(DETECTION_DEFAULT), True),
                            (dict(DETECTION_DEFAULT, detect_threshold=10.0), True),
                            (dict(DETECTION_DEFAULT, detect_threshold=20.0), False)):
        rec = si.from_recording(raw, det)
        assert rec.n_samples == N_C and rec.fs_raw_hz == FS_C and sorted(rec.spikes) == [0, 1, 2, 3]
        assert rec.positions_um[3] == (60.0, 60.0)
        for c, t in planted.items():
            got, missed, spur = _match(t, rec.spikes[c], TOL_SAMPLES)
            if expect_all:
                assert missed == 0 and spur == 0, (
                    "threshold %g, channel %d (SD %g): %d planted, %d found, %d missed, %d "
                    "spurious" % (det["detect_threshold"], c, SIGMA_C[c], t.size,
                                  rec.spikes[c].size, missed, spur))
                if det["detect_threshold"] == 5.0:
                    off = rec.spikes[c] - t
                    tot[0] += t.size
                    tot[1] += rec.spikes[c].size
                    tot[2] += missed
                    tot[3] += spur
                    tot.append(int(np.abs(off).max()))
                    assert np.all(np.abs(off) <= TOL_SAMPLES)
            else:
                # 20 noise levels is beyond the 15-SD spikes on every channel, whatever its
                # SD: the threshold is a multiple of each channel's own noise level
                assert rec.spikes[c].size == 0, (
                    "threshold 20: channel %d found %d peaks" % (c, rec.spikes[c].size))
    print("T43c: planted %d, found %d, missed %d, spurious %d (tolerance %d sample; largest "
          "offset %d)" % (tot[0], tot[1], tot[2], tot[3], TOL_SAMPLES, max(tot[4:])))
    # without detection a raw file is refused
    folder = os.path.join(_tmpdir("t43c"), "raw")
    raw.save(folder=folder, n_jobs=1, progress_bar=False, verbose=False)
    _refused(lambda: si.from_recording(raw, None), "detection")
    from sbitool import recordings
    _refused(lambda: recordings.read(folder, "spikeinterface", reader="binaryfolder"),
             "detection")
    _refused(lambda: si.from_recording(raw, dict(DETECTION_DEFAULT, method="locally_exclusive")),
             "by_channel")


def test_t43d_layout():
    """as_grid on a jittered grid, LayoutError otherwise; the round trip (specs/SPEC.md block 43)."""
    from sbitool.recordings import layout
    rng = np.random.default_rng(SEED + 3)
    w, p, x0, y0, jit, tol = 7, 42.0, 13.0, -7.0, 0.4, 1.0
    ids = rng.permutation(1000)[:w * w]     # electrode ids unrelated to the cells
    cells = rng.permutation(w * w)
    pos, exp = {}, {}
    for e, k in zip(ids, cells):
        r, c = divmod(int(k), w)
        pos[int(e)] = (x0 + c * p + rng.uniform(-jit, jit), y0 + r * p + rng.uniform(-jit, jit))
        exp[int(e)] = int(k)
    gw, pitch, idx = layout.as_grid(pos, tol_um=tol)
    assert gw == w and idx == exp, (gw, sum(idx[e] != exp[e] for e in exp))
    # 0.1 um: the jitter is uniform +-0.4 (SD 0.23); the least-squares pitch from
    # 98 coordinates over 7 levels has an SD near 0.01 um, so 0.1 is ~8 SDs
    assert abs(pitch - p) < 0.1, pitch
    # a rectangle of 3 rows x 5 columns: the width is the longer side
    rect = {r * 5 + c: (c * 30.0, r * 30.0) for r in range(3) for c in range(5)}
    gw2, p2, i2 = layout.as_grid(rect)
    assert gw2 == 5 and abs(p2 - 30.0) < 1e-9 and i2 == {k: k for k in rect}, (gw2, p2)
    bad = {}
    # hexagonal: rows sqrt(3)/2 apart, odd rows shifted by half the pitch
    bad["hexagonal"] = {i * 8 + j: ((j + 0.5 * (i % 2)) * p, i * p * np.sqrt(3) / 2)
                        for i in range(6) for j in range(8)}
    bad["irregular"] = {i: tuple(rng.uniform(0, 300, size=2)) for i in range(40)}
    bad["jitter beyond the tolerance"] = {k: (v[0] + rng.uniform(2.5, 3.0), v[1])
                                         if k == ids[0] else v for k, v in pos.items()}
    bad["two in one cell"] = dict(rect)
    bad["two in one cell"][99] = (0.2, 0.1)
    for tag, pp in bad.items():
        try:
            layout.as_grid(pp, tol_um=tol)
        except layout.LayoutError:
            continue
        raise AssertionError("%s positions gave a grid" % tag)
    # the round trip with ptrain's positions
    _need_extractor()
    from sbitool import recordings
    rng2 = np.random.default_rng(SEED)
    trains = _trains(rng2)
    d = os.path.join(_tmpdir("t43d"), "raster")
    _write_raster(d, trains)
    rec = recordings.read(d, "ptrain", fs_raw_hz=FS, grid_width=GW, index_base=BASE,
                          pitch_um=PITCH)
    gw3, p3, i3 = layout.as_grid(rec.positions_um)
    # exact: noiseless positions on the grid, the least squares solves them exactly
    # up to float64 roundoff (1e-9 um)
    assert gw3 == GW and abs(p3 - PITCH) < 1e-9 and i3 == {e: e - BASE for e in trains}, \
        (gw3, p3)
    rc = layout.grid(rec)
    assert rc == {e: ((e - BASE) // GW, (e - BASE) % GW) for e in trains}
    no_grid = recordings.read(d, "ptrain", fs_raw_hz=FS, grid_width=GW, index_base=BASE)
    try:
        layout.grid(no_grid)
        raise AssertionError("layout.grid accepted a recording without a grid")
    except layout.LayoutError:
        pass


CHECKS = [("test_t43a_ptrain", test_t43a_ptrain),
          ("test_t43b_si_sorting", test_t43b_si_sorting),
          ("test_t43c_si_raw", test_t43c_si_raw),
          ("test_t43d_layout", test_t43d_layout)]


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
