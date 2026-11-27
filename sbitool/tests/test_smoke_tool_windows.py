#!/usr/bin/env python3
"""Smoke test of block 44: from a recording to windows (specs/SPEC.md block 44, "Test
oracles", checks T44*).

A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why. Every check is SKIP,
naming SBIX_DIR, where the Sbi-extractor's checkout is absent.

Checks
    test_t44a_ifr            the pooled IFR by its definition, computed independently
    test_t44b_partition      the subregions equal the extractor's command-line result
    test_t44c_windows        the windows' count, order, start times and dtype
    test_t44d_refusals       another rate, grid or pitch; a bundle without extraction

Run
    SBIX_CLONE=<the Sbi-extractor's checkout> PYTHONDONTWRITEBYTECODE=1 \
        python3 sbitool/tests/test_smoke_tool_windows.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO.

Pure ASCII, LF only.
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from unittest import mock

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


import numpy as np  # noqa: E402

from sbitool import _sources  # noqa: E402
from sbitool import schema as S  # noqa: E402
from sbitool.interfaces import Recording  # noqa: E402


# ------------------------------------------------------------ helpers

def _need_extractor():
    try:
        _sources.extractor_dir()
    except _sources.SourceError as exc:
        raise unittest.SkipTest("the Sbi-extractor's checkout is absent (set SBIX_DIR or "
                                "SBIX_CLONE): %s" % exc)


def _W():
    from sbitool import windows
    return windows


def _extraction(**kw):
    e = {"electrodes_per_subset": 9, "n_subsets": 1, "mfr_threshold_hz": 0.0,
         "w_size_s": 0.02, "gaussian_window_s": 0.04, "grid_width": 3, "index_base": 0,
         "pitch_um": 60.0, "fs_raw_hz": 1000.0, "detection": None}
    e.update(kw)
    probs = S.validate_extraction(e)
    assert not probs, "the test's extraction is not valid: %s" % probs
    return e


def _rec(spikes, fs, n, grid=None, positions=None, name="toy"):
    rec = Recording(name=name, spikes={int(k): np.asarray(v, dtype=np.int64)
                                       for k, v in spikes.items()},
                    fs_raw_hz=float(fs), n_samples=int(n), positions_um=positions, grid=grid,
                    source={"adapter": "test"})
    probs = rec.check()
    assert not probs, "the test's recording is not valid: %s" % probs
    return rec


def _refused(fn, words):
    """fn() raises WindowsError whose message holds every one of `words`."""
    W = _W()
    try:
        fn()
    except W.WindowsError as exc:
        msg = str(exc)
        missing = [w for w in words if w not in msg]
        assert not missing, "refused, but the message %r does not name %s" % (msg, missing)
        return msg
    raise AssertionError("not refused (expected a WindowsError naming %s)" % (words,))


# ------------------------------------------------------------ T44a

def _gaussian_reflect(c, sigma_bins, truncate=4.0):
    """The Gaussian smoothing eq. (2) of EXTRACTOR_USAGE.md names
    (scipy.ndimage.gaussian_filter1d, its defaults), written out in numpy:
    the kernel exp(-x^2 / (2 sigma^2)) on x = -r..r, r = int(truncate sigma +
    0.5), normalised to sum 1; the boundary 'reflect' (d c b a | a b c d | d c
    b a), numpy's 'symmetric' pad."""
    r = int(truncate * float(sigma_bins) + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * x ** 2 / float(sigma_bins) ** 2)
    g /= g.sum()
    padded = np.pad(np.asarray(c, dtype=np.float64), r, mode="symmetric")
    return np.array([np.dot(padded[k:k + 2 * r + 1], g) for k in range(c.size)])


def _ifr_reference(spikes, members, fs, n, dt, sigma):
    """eq. (1)-(3) of EXTRACTOR_USAGE.md, independently: C[k] the spikes of
    the members counted in [k dt, (k + 1) dt) by numpy.histogram, k in
    0..K-1, K = floor(T_rec / dt), T_rec = n / fs; smoothed, clipped at 0,
    divided by n_e = len(members). float64."""
    T = n / fs
    K = int(np.floor(T / dt))
    edges = np.arange(K + 1, dtype=np.float64) * dt
    c = np.zeros(K)
    for e in members:
        c += np.histogram(np.asarray(spikes[e], dtype=np.float64) / fs, bins=edges)[0]
    return np.clip(_gaussian_reflect(c, sigma / dt), 0.0, None) / float(len(members)), K


def test_t44a_ifr():
    """the pooled IFR by its definition, computed independently (specs/SPEC.md block 44)."""
    _need_extractor()
    W = _W()
    worst = []
    cases = [
        # (fs, n, dt, sigma, spikes): 9 electrodes of a 3 x 3 grid, 6 of them silent;
        # n_e = 9 (the threshold 0 keeps the silent ones), so a division by the
        # number of firing electrodes would be off by 3; a spike in bin 0 and in the
        # last bin exercise the reflecting boundary
        (1000.0, 2000, 0.02, 0.04,
         {0: [5, 310, 311, 1995], 4: [305, 1000], 8: [1990]}),
        # T_rec = 2.055 s is not a multiple of dt = 0.01 s: K = 205, and the spike at
        # t = 2.0525 s lies beyond K dt and is not counted
        (2000.0, 4110, 0.01, 0.03,
         {1: [10, 11, 12, 2001], 3: [2003, 4105], 7: [3001, 3002], 2: [999]}),
    ]
    for fs, n, dt, sigma, firing in cases:
        spikes = {e: firing.get(e, []) for e in range(9)}
        rec = _rec(spikes, fs, n, grid=(3, 0, 60.0))
        ext = _extraction(w_size_s=dt, gaussian_window_s=sigma, fs_raw_hz=fs)
        subs = W.subregions(rec, ext)
        assert len(subs) == 1 and sorted(subs[0].members) == list(range(9)), \
            "the one subregion must hold the 9 electrodes, got %r" % (subs,)
        x, fs_ifr = W.traces(rec, subs, ext)
        ref, K = _ifr_reference(spikes, subs[0].members, fs, n, dt, sigma)
        assert x.dtype == np.float32 and x.shape == (1, K), \
            "x must be (1, %d) float32, got %s %s" % (K, x.shape, x.dtype)
        assert fs_ifr == 1.0 / dt, "fs_ifr %r, 1 / dt = %r" % (fs_ifr, 1.0 / dt)
        # 1e-5 of the largest value (SPEC block 44, T44a): the extractor rounds a
        # float64 trace to float32 (relative error <= 2^-24 = 6e-8 per value);
        # every other difference (the order of the kernel's sums) is float64
        # roundoff, ~1e-16 -- far below the bound, and a dropped n_e (factor 9),
        # another sigma or a bin shifted by one are differences of order 1e-1
        scale = float(np.max(np.abs(ref)))
        assert scale > 0, "the reference trace is zero: the case tests nothing"
        diff = float(np.max(np.abs(x[0].astype(np.float64) - ref))) / scale
        worst.append(diff)
        assert diff <= 1e-5, ("the trace differs from the definition by %.3g of its largest "
                              "value (fs %g, n %d, dt %g, sigma %g)" % (diff, fs, n, dt, sigma))
    print("[T44a] largest relative difference per case: %s"
          % ", ".join("%.3g" % d for d in worst))


# ------------------------------------------------------------ T44b

GW, BASE, PITCH, FS, NSAMP = 6, 1, 60.0, 10000.0, 20000      # T_rec = 2 s
THR, NE, NSUB = 2.0, 4, 3                                     # 2 Hz = 4 spikes in 2 s


def _counts(rng):
    """Spike counts per electrode of the 6 x 6 grid (ids 1..36): every count
    distinct (no tie in the ranking by rate); 12 electrodes below 4 spikes
    (below THR), the others 4..27 spikes."""
    ids = list(range(BASE, BASE + GW * GW))
    order = rng.permutation(len(ids))
    low = [0, 1, 2, 3] * 3                       # 12 electrodes below the threshold
    high = list(range(4, 28))                    # 24 electrodes at or above it
    vals = low + high
    return {ids[i]: vals[order[i]] for i in range(len(ids))}


def _trains(rng, counts):
    return {e: np.sort(rng.choice(NSAMP, size=k, replace=False)).astype(np.int64)
            for e, k in counts.items()}


def _write_raster(folder, trains, n=NSAMP):
    import scipy.io as sio
    os.makedirs(folder, exist_ok=True)
    for e, t in trains.items():
        r = np.zeros((n, 1), dtype=np.uint8)
        r[t, 0] = 1
        sio.savemat(os.path.join(folder, "ptrain_%d.mat" % e), {"ptrain": r}, format="5",
                    do_compression=True)


def _grid_ext(**kw):
    e = dict(electrodes_per_subset=NE, n_subsets=NSUB, mfr_threshold_hz=THR, grid_width=GW,
             index_base=BASE, pitch_um=PITCH, fs_raw_hz=FS)
    e.update(kw)
    return _extraction(**e)


_FIXTURE = {}


def _fixture():
    """A ptrain folder of the 6 x 6 grid, its trains and its Recording (cached)."""
    if not _FIXTURE:
        from sbitool.recordings import ptrain
        rng = np.random.default_rng(SEED)
        counts = _counts(rng)
        trains = _trains(rng, counts)
        folder = os.path.join(_tmpdir("w44"), "rec")
        _write_raster(folder, trains)
        rec = ptrain.read(folder, fs_raw_hz=FS, grid_width=GW, index_base=BASE, pitch_um=PITCH)
        _FIXTURE.update(counts=counts, trains=trains, folder=folder, rec=rec)
    return _FIXTURE


def _cli(folder, out, ext):
    """The extractor's command line on `folder` (per_region_single: one trace per
    subregion, as the real bank reads them), its files read back."""
    script = os.path.join(_sources.extractor_dir(), "extractor", "run_channel_subset_extraction.py")
    env = dict(os.environ)
    env.setdefault("SBI_HPC_DIR", os.path.join(_REPO, "hpc"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, script, folder, "--out-dir", out, "--mode", "per_region_single",
           "--n-subsets", str(ext["n_subsets"]),
           "--electrodes-per-subset", str(ext["electrodes_per_subset"]),
           "--mfr-threshold", repr(float(ext["mfr_threshold_hz"])),
           "--fs-raw", repr(float(ext["fs_raw_hz"])), "--base", str(ext["index_base"]),
           "--grid-width", str(ext["grid_width"]), "--w-size", repr(float(ext["w_size_s"])),
           "--gaussian-window", repr(float(ext["gaussian_window_s"])), "--no-plots"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, "the extractor's command line failed (%d): %s\n%s" % (
        r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    with np.load(os.path.join(out, "traces.npz"), allow_pickle=False) as z:
        got = {k: z[k] for k in ("X", "centers", "center_mfr", "discarded", "fs_ifr")}
    return got


def test_t44b_partition():
    """the subregions equal the extractor's command-line result (specs/SPEC.md block 44)."""
    _need_extractor()
    W = _W()
    fx = _fixture()
    rec, counts = fx["rec"], fx["counts"]
    ext = _grid_ext()
    subs = W.subregions(rec, ext)
    T = NSAMP / FS
    above = {e for e, k in counts.items() if k / T >= THR}
    # the invariants: NSUB subregions of NE electrodes, disjoint, each above the threshold
    assert len(subs) == NSUB, "%d subregions, expected %d" % (len(subs), NSUB)
    seen = set()
    for s in subs:
        m = [int(e) for e in s.members]
        assert len(m) == NE, "subregion %r has %d electrodes, expected %d" % (s, len(m), NE)
        assert not (set(m) & seen), "subregions overlap at %s" % sorted(set(m) & seen)
        seen |= set(m)
        assert set(m) <= above, "electrodes below the threshold in %r: %s" % (
            s, sorted(set(m) - above))
        assert m[0] == s.center
    # the centres: the hottest electrode first (every count distinct)
    hottest = max(counts, key=lambda e: counts[e])
    assert subs[0].center == hottest, "subregion 0's centre %d, the hottest electrode %d" % (
        subs[0].center, hottest)
    # the extractor's command line on the same folder: its files hold the centres,
    # their rates, the discarded electrodes and the traces -- not the members, which
    # are compared through the traces (each trace pools exactly its members' spikes)
    got = _cli(fx["folder"], _tmpdir("w44cli"), ext)
    x, fs_ifr = W.traces(rec, subs, ext)
    centers = np.array([s.center for s in subs], dtype=np.int64)
    cmfr = np.array([s.center_mfr for s in subs], dtype=np.float64)
    assert np.array_equal(got["centers"], centers), "centres %s, the command line's %s" % (
        centers.tolist(), got["centers"].tolist())
    # exact: the same float64 division N / T_rec on both sides
    assert np.array_equal(got["center_mfr"], cmfr), "centre rates %s, the command line's %s" % (
        cmfr.tolist(), got["center_mfr"].tolist())
    disc = sorted(e for e in counts if e not in above)
    assert got["discarded"].tolist() == disc, "the command line discards %s, expected %s" % (
        got["discarded"].tolist(), disc)
    # bit for bit: both are the extractor's float32 subregion_ifr of the same spikes
    assert got["X"].dtype == np.float32 and got["X"].shape == x.shape, (got["X"].shape, x.shape)
    assert np.array_equal(got["X"], x), "the traces differ from the command line's by %.3g" % (
        float(np.max(np.abs(got["X"].astype(np.float64) - x))))
    assert float(got["fs_ifr"]) == fs_ifr
    # the same recording with positions and no grid (the spikeinterface adapter's form):
    # layout.as_grid places the electrodes on the same rows and columns, so the
    # partition is the same
    rec_pos = Recording(name=rec.name, spikes=rec.spikes, fs_raw_hz=rec.fs_raw_hz,
                        n_samples=rec.n_samples, positions_um=rec.positions_um, grid=None,
                        source={"adapter": "test"})
    subs_pos = W.subregions(rec_pos, ext)
    assert [tuple(s.members) for s in subs_pos] == [tuple(s.members) for s in subs], \
        "positions-only partition %r differs from the grid's %r" % (subs_pos, subs)
    # amendment 1 of wave B: for positions only the partition is keyed by as_grid's
    # linear index and mapped back to the electrode ids. (i) the 3 x 3 case of block
    # 44's report: id = 100 - linear index, the centre hottest, every other rate equal
    lin_spikes = {k: np.arange(20 if k == 4 else 10, dtype=np.int64) * 50 for k in range(9)}
    small = _extraction(electrodes_per_subset=3, n_subsets=1, grid_width=3, index_base=0,
                        pitch_um=60.0, fs_raw_hz=1000.0)
    rec_g = _rec(lin_spikes, 1000.0, 2000, grid=(3, 0, 60.0))
    rec_p = _rec({100 - k: v for k, v in lin_spikes.items()}, 1000.0, 2000,
                 positions={100 - k: (60.0 * (k % 3), 60.0 * (k // 3)) for k in range(9)})
    sg, sp = W.subregions(rec_g, small)[0], W.subregions(rec_p, small)[0]
    assert sg.members == (4, 1, 3), "the grid's partition %r, expected (4, 1, 3)" % (sg.members,)
    assert sp.center == 96 and sp.members == (96, 99, 97), \
        "positions only: %r, expected the linear-index partition (96, 99, 97)" % (sp,)
    assert (sp.mean_mfr, sp.center_mfr) == (sg.mean_mfr, sg.center_mfr)
    assert np.array_equal(W.traces(rec_p, [sp], small)[0], W.traces(rec_g, [sg], small)[0])
    # (ii) the 6 x 6 device with its electrodes renamed at random: the same partition
    # (renamed) and the same traces as the device read with a grid
    perm = np.random.default_rng(SEED + 4).permutation(GW * GW)
    new = {e: 1000 + int(perm[e - BASE]) for e in rec.spikes}
    rec_r = Recording(name="renamed", spikes={new[e]: t for e, t in rec.spikes.items()},
                      fs_raw_hz=FS, n_samples=NSAMP,
                      positions_um={new[e]: xy for e, xy in rec.positions_um.items()}, grid=None,
                      source={"adapter": "test"})
    subs_r = W.subregions(rec_r, ext)
    assert [s.members for s in subs_r] == [tuple(new[m] for m in s.members) for s in subs], \
        "renamed positions-only partition %r differs from the grid's %r" % (subs_r, subs)
    x_r, _fs = W.traces(rec_r, subs_r, ext)
    # bit for bit: the same spikes pooled, in the same member order
    assert np.array_equal(x_r, x), "the renamed device's traces differ from the grid's"
    print("[T44b] centres %s, members %s; command line: centres, rates, %d discarded and "
          "the (%d, %d) traces equal" % (centers.tolist(), [list(s.members) for s in subs],
                                         len(disc), x.shape[0], x.shape[1]))


# ------------------------------------------------------------ T44c

def test_t44c_windows():
    """the windows' count, order, start times and dtype (specs/SPEC.md block 44)."""
    _need_extractor()
    W = _W()
    # (1) cut on a float64 trace: floor(K / W) windows per subregion, back to back
    # from the start, subregion-major; float32 of the float64 values
    rng = np.random.default_rng(SEED + 1)
    for K, Wn in ((23, 5), (25, 5), (4, 5), (5, 5)):
        x = rng.uniform(0.0, 30.0, size=(3, K))
        fs = 50.0
        win, ids = W.cut(x, Wn, fs)
        per = K // Wn
        n = 3 * per
        assert win.shape == (n, Wn) and win.dtype == np.float32, (K, Wn, win.shape, win.dtype)
        for key, dt in (("subregion", np.int64), ("window_idx", np.int64),
                        ("t_start_s", np.float64)):
            assert ids[key].shape == (n,) and ids[key].dtype == dt, (key, ids[key].shape,
                                                                      ids[key].dtype)
        exp_sub = np.repeat(np.arange(3), per)
        exp_j = np.tile(np.arange(per), 3)
        assert np.array_equal(ids["subregion"], exp_sub), (ids["subregion"], exp_sub)
        assert np.array_equal(ids["window_idx"], exp_j), (ids["window_idx"], exp_j)
        # exact: the same float64 product j W / fs
        assert np.array_equal(ids["t_start_s"], exp_j * float(Wn) / fs), ids["t_start_s"]
        for i in range(n):
            s, j = exp_sub[i], exp_j[i]
            assert np.array_equal(win[i], x[s, j * Wn:(j + 1) * Wn].astype(np.float32)), \
                "window %d (subregion %d, j %d) is not x[%d, %d:%d]" % (i, s, j, s, j * Wn,
                                                                       (j + 1) * Wn)
    # (2) extract on the 6 x 6 recording: the rows of the extractor's trace
    fx = _fixture()
    ext = _grid_ext()
    Wn = 7
    win, ids, subs = W.extract(fx["rec"], ext, Wn)
    x, fs_ifr = W.traces(fx["rec"], subs, ext)
    K = x.shape[1]
    per = K // Wn
    assert K == 100 and per == 14, (K, per)
    assert win.shape == (NSUB * per, Wn) and win.dtype == np.float32
    for i in range(win.shape[0]):
        s, j = int(ids["subregion"][i]), int(ids["window_idx"][i])
        assert s == i // per and j == i % per, (i, s, j)
        assert np.array_equal(win[i], x[s, j * Wn:(j + 1) * Wn])
        assert ids["t_start_s"][i] == j * Wn / fs_ifr
    # W = 1 and W = K: every bin a window; one window
    assert W.cut(x, 1, fs_ifr)[0].shape == (NSUB * K, 1)
    assert W.cut(x, K, fs_ifr)[0].shape == (NSUB, K)
    _refused(lambda: W.cut(x, 0, fs_ifr), ["W"])
    print("[T44c] K %d, W %d: %d windows per subregion, %d in all" % (K, Wn, per, win.shape[0]))


# ------------------------------------------------------------ T44d

def _stand_in_bundle(manifest):
    class _B(object):
        pass
    b = _B()
    b.manifest = manifest
    b.path = "stand-in"
    return b


def _run_main(argv):
    W = _W()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = W.main(argv)
    return code, out.getvalue(), err.getvalue()


def _last(text):
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


def _need_si():
    try:
        import spikeinterface  # noqa: F401
        import probeinterface  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("spikeinterface or probeinterface not installed: %s" % exc)


def _raw_folder(folder, rng, fs, n, side, pitch):
    """A raw recording saved by SpikeInterface (binaryfolder): side x side
    channels (ids 0.., row-major, `pitch` um apart on a probe), white noise
    of SD 1 plus 10 negative spikes per channel of 15 SDs, 300 samples or
    more apart. Returns the folder."""
    import probeinterface
    from spikeinterface.core import NumpyRecording
    nc = side * side
    x = rng.normal(0.0, 1.0, size=(n, nc))
    shape = np.array([0.3, 0.7, 1.0, 0.7, 0.3])
    for c in range(nc):
        slots = np.sort(rng.choice((n - 600) // 300, size=10, replace=False))
        for t in 300 + slots * 300 + rng.integers(0, 50, size=10):
            x[t - 2:t + 3, c] -= 15.0 * shape
    rec = NumpyRecording([x.astype(np.float32)], sampling_frequency=fs,
                         channel_ids=np.arange(nc))
    probe = probeinterface.Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=np.array([[pitch * (c % side), pitch * (c // side)]
                                           for c in range(nc)]),
                       shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(nc))
    rec.set_probe(probe)
    rec.save(folder=folder, n_jobs=1, progress_bar=False, verbose=False)
    return folder


def _detection_cases(fx, ext, out_dir):
    """Amendment 2 of wave B through the command (bundle.read patched): the
    bundle's detection is applied to raw voltage only; spike trains are
    accepted whatever it declares; --option detection is refused."""
    from sbitool import bundle as bundle_mod
    from sbitool import recordings
    W = _W()
    det = dict(S.DETECTION_DEFAULT)
    ext_det = dict(ext, detection=det)
    opts = ["--option", "fs_raw_hz=%r" % FS, "--option", "grid_width=%d" % GW,
            "--option", "index_base=%d" % BASE, "--option", "pitch_um=%r" % PITCH]

    def run(extraction, rec_path, adapter, options, out):
        with mock.patch.object(bundle_mod, "read",
                               lambda path: _stand_in_bundle({"extraction": extraction, "W": 7})):
            return _run_main(["--bundle", "b.sbibundle", "--recording", rec_path, "--adapter",
                              adapter, "--out", out] + options)

    def meta_of(out):
        with np.load(out, allow_pickle=False) as z:
            return json.loads(str(z["meta"])), z["windows"].copy()

    win_ref = W.extract(fx["rec"], ext, 7)[0]
    # (a) ptrain spike trains, the bundle declaring a detection: accepted, not applied
    out = os.path.join(out_dir, "a.npz")
    code, so, se = run(ext_det, fx["folder"], "ptrain", opts, out)
    assert code == 0 and "detected outside the tool" in so and _last(so).startswith(
        "WINDOWS OK: 42 window(s)"), (code, so, se)
    meta, win = meta_of(out)
    assert meta["detection"] == det and meta["detection_applied"] is None, meta
    assert np.array_equal(win, win_ref)
    # (e) a detection among the adapter's options: the arguments, exit 2
    code, so, se = run(ext_det, fx["folder"], "ptrain",
                       opts + ["--option", "detection=%s" % json.dumps(det)],
                       os.path.join(out_dir, "e.npz"))
    assert code == 2 and "detection" in se and not os.path.exists(os.path.join(out_dir, "e.npz")), \
        (code, so, se)
    _need_si()
    from spikeinterface.core import NpzSortingExtractor, NumpySorting
    # (b) a spikeinterface sorting of the same trains, the bundle declaring a detection:
    # accepted, not applied, the same windows as the ptrain folder's
    sorting = NumpySorting.from_unit_dict([{int(e): t for e, t in fx["trains"].items()}],
                                          sampling_frequency=FS)
    spath = os.path.join(out_dir, "sorting.npz")
    NpzSortingExtractor.write_sorting(sorting, spath)
    out = os.path.join(out_dir, "b.npz")
    code, so, se = run(ext_det, spath, "spikeinterface",
                       ["--option", "reader=npz", "--option", "n_samples=%d" % NSAMP,
                        "--option", "grid=[%d, %d, %r]" % (GW, BASE, PITCH)], out)
    assert code == 0 and "detected outside the tool" in so, (code, so, se)
    meta, win = meta_of(out)
    assert meta["detection"] == det and meta["detection_applied"] is None, meta
    assert np.array_equal(win, win_ref), "the sorting's windows differ from the ptrain folder's"
    # (c) raw voltage: the bundle's detection is applied
    fs_raw, n_raw = 10000.0, 20000
    rfolder = _raw_folder(os.path.join(out_dir, "raw"), np.random.default_rng(SEED + 3),
                          fs_raw, n_raw, 3, 60.0)
    ext_raw = _extraction(electrodes_per_subset=9, n_subsets=1, mfr_threshold_hz=0.0,
                          grid_width=3, index_base=0, pitch_um=60.0, fs_raw_hz=fs_raw,
                          detection=det)
    out = os.path.join(out_dir, "c.npz")
    code, so, se = run(ext_raw, rfolder, "spikeinterface", ["--option", "reader=binaryfolder"], out)
    assert code == 0 and "detected outside" not in so and _last(so) == \
        "WINDOWS OK: 14 window(s) of 1 subregion(s)", (code, so, se)
    meta, win = meta_of(out)
    assert meta["detection"] == det and meta["detection_applied"] == det, meta
    direct = recordings.read(rfolder, "spikeinterface", reader="binaryfolder", detection=det)
    assert sum(t.size for t in direct.spikes.values()) == 90, "the detection found %d of 90" % (
        sum(t.size for t in direct.spikes.values()))
    assert np.array_equal(win, W.extract(direct, ext_raw, 7)[0])
    # (d) raw voltage, the bundle's detection null: the adapter's refusal, exit 3
    out = os.path.join(out_dir, "d.npz")
    code, so, se = run(dict(ext_raw, detection=None), rfolder, "spikeinterface",
                       ["--option", "reader=binaryfolder"], out)
    assert code == 3 and "detection" in se and not os.path.exists(out), (code, so, se)


def _real_bundle(folder, extraction, Wn):
    """A bundle written by block 41's bundle.write from the manifest block 40's
    test writes from sec. 2.7, with `extraction` and W set."""
    import torch
    from sbitool import bundle
    sys.path.insert(0, _HERE)
    try:
        from test_smoke_tool_schema import _valid_manifest
    finally:
        sys.path.remove(_HERE)
    man = _valid_manifest()
    man["extraction"] = extraction
    man["W"] = int(Wn)
    man["card_complete"] = False
    M, d_z, g = man["M"], man["d_z"], man["gate"]
    states = [{"w": torch.zeros(2)} for _ in range(M)]
    gate_sets = {"z_ref": np.zeros((M, g["n_ref"], d_z), np.float32),
                 "z_rank": np.zeros((M, g["n_rank"], d_z), np.float32),
                 "z_null": np.zeros((M, g["n_null"], d_z), np.float32)}
    path = os.path.join(folder, "toy_%s.sbibundle" % ("none" if extraction is None else "ext"))
    bundle.write(path, man, "---\ntitle: TODO\n---\n", states, gate_sets, b"{}")
    return path


def test_t44d_refusals():
    """another rate, grid or pitch; a bundle without extraction (specs/SPEC.md block 44)."""
    _need_extractor()
    W = _W()
    fx = _fixture()
    rec = fx["rec"]
    ext = _grid_ext()
    base = W.grid_of(rec, ext)
    assert base == {e: ((e - BASE) // GW, (e - BASE) % GW) for e in rec.spikes}, \
        "grid_of is not the row-major decoding of the linear index"
    # the rate, the grid's width, its index base, its pitch: each refused, naming both
    _refused(lambda: W.grid_of(rec, _grid_ext(fs_raw_hz=10001.0)),
             ["fs_raw_hz", "10000.0", "10001.0"])
    _refused(lambda: W.extract(rec, _grid_ext(fs_raw_hz=9999.5), 5), ["9999.5", "10000.0"])
    _refused(lambda: W.grid_of(rec, _grid_ext(grid_width=8)), ["grid_width", "6", "8"])
    _refused(lambda: W.grid_of(rec, _grid_ext(index_base=0)), ["index_base", "1", "0"])
    _refused(lambda: W.grid_of(rec, _grid_ext(pitch_um=65.0)), ["pitch_um", "60.0", "65.0"])
    _refused(lambda: W.subregions(rec, _grid_ext(pitch_um=60.5)), ["60.0", "60.5"])
    # a null pitch on the recording's side is not compared
    rec_np = Recording(name="np", spikes=rec.spikes, fs_raw_hz=FS, n_samples=NSAMP,
                       positions_um=None, grid=(GW, BASE, None), source={})
    assert W.grid_of(rec_np, _grid_ext(pitch_um=65.0)) == base
    # positions only: the pitch within 1 um, the width not compared
    rec_pos = Recording(name="pos", spikes=rec.spikes, fs_raw_hz=FS, n_samples=NSAMP,
                        positions_um=rec.positions_um, grid=None, source={})
    _refused(lambda: W.grid_of(rec_pos, _grid_ext(pitch_um=61.5)), ["61.5", "60"])
    _refused(lambda: W.grid_of(rec_pos, _grid_ext(pitch_um=58.9)), ["58.9", "60"])
    ok = W.grid_of(rec_pos, _grid_ext(pitch_um=60.8, grid_width=64, index_base=0))
    assert ok == base, "positions-only rows and columns differ from the grid's"
    _refused(lambda: W.grid_of(rec_pos, _grid_ext(fs_raw_hz=20000.0)), ["20000.0", "10000.0"])
    # neither a grid nor positions
    rec_none = Recording(name="bare", spikes=rec.spikes, fs_raw_hz=FS, n_samples=NSAMP,
                         positions_um=None, grid=None, source={})
    _refused(lambda: W.grid_of(rec_none, ext), ["neither a grid nor electrode positions"])
    # an invalid extraction, too few electrodes above the threshold
    bad = dict(ext)
    del bad["pitch_um"]
    _refused(lambda: W.extract(rec, bad, 5), ["pitch_um"])
    _refused(lambda: W.subregions(rec, _grid_ext(n_subsets=7)), ["partition"])

    # the command, with a stand-in for bundle.read (the command's logic, whatever
    # block 41 is): a bundle without extraction exits 3; one with it exits 0 and
    # writes extract()'s windows
    from sbitool import bundle as bundle_mod
    out_dir = _tmpdir("w44cmd")
    out = os.path.join(out_dir, "windows.npz")
    opts = ["--option", "fs_raw_hz=%r" % FS, "--option", "grid_width=%d" % GW,
            "--option", "index_base=%d" % BASE, "--option", "pitch_um=%r" % PITCH]
    argv = ["--bundle", "b.sbibundle", "--recording", fx["folder"], "--adapter", "ptrain",
            "--out", out] + opts
    with mock.patch.object(bundle_mod, "read",
                           lambda path: _stand_in_bundle({"extraction": None, "W": 7})):
        code, so, se = _run_main(argv)
    assert code == 3 and "extraction" in se and not os.path.exists(out), (code, so, se)
    with mock.patch.object(bundle_mod, "read",
                           lambda path: _stand_in_bundle({"extraction": ext, "W": 7})):
        code, so, se = _run_main(argv)
        assert code == 0 and _last(so) == "WINDOWS OK: 42 window(s) of 3 subregion(s)" and "detected outside the tool" in so, \
            (code, so, se)
        code_rate, _so, se_rate = _run_main(argv[:-8] + ["--option", "fs_raw_hz=20000.0"]
                                            + opts[2:])
        assert code_rate == 3 and "20000.0" in se_rate and "10000.0" in se_rate, (code_rate,
                                                                                  se_rate)
        code_opt, _so, _se = _run_main(argv + ["--option", "no_such_option=1"])
        assert code_opt == 2, code_opt
    code_args, _so, _se = _run_main(["--bundle", "b.sbibundle"])
    assert code_args == 2, code_args
    win, ids, subs = W.extract(rec, ext, 7)
    with np.load(out, allow_pickle=False) as z:
        assert sorted(z.files) == sorted(["windows", "subregion", "window_idx", "t_start_s",
                                          "fs_ifr_hz", "meta"]), z.files
        assert z["windows"].dtype == np.float32 and np.array_equal(z["windows"], win)
        for k in ("subregion", "window_idx", "t_start_s"):
            assert np.array_equal(z[k], ids[k]), k
        assert float(z["fs_ifr_hz"]) == 1.0 / ext["w_size_s"]
        meta = json.loads(str(z["meta"]))
    assert meta["extraction"] == ext and meta["recording"]["name"] == rec.name
    assert [m["members"] for m in meta["subregions"]] == [list(s.members) for s in subs]

    _detection_cases(fx, ext, _tmpdir("w44det"))

    # the same with a real bundle (block 41's bundle.write and bundle.read)
    try:
        bundle_mod.read(os.path.join(out_dir, "absent.sbibundle"))
    except NotImplementedError:
        raise unittest.SkipTest("every other part passes; the part with a real bundle needs "
                                "block 41 (sbitool.bundle.read is a stub)")
    except Exception:                # noqa: BLE001 -- a refusal of an absent file is expected
        pass
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("every other part passes; a real bundle needs torch")
    path_none = _real_bundle(out_dir, None, 7)
    path_ext = _real_bundle(out_dir, ext, 7)
    out2 = os.path.join(out_dir, "windows2.npz")
    code, so, se = _run_main(["--bundle", path_none, "--recording", fx["folder"], "--adapter",
                              "ptrain", "--out", out2] + opts)
    assert code == 3 and "extraction" in se and not os.path.exists(out2), (code, so, se)
    code, so, se = _run_main(["--bundle", path_ext, "--recording", fx["folder"], "--adapter",
                              "ptrain", "--out", out2] + opts)
    assert code == 0 and _last(so) == "WINDOWS OK: 42 window(s) of 3 subregion(s)" and "detected outside the tool" in so, (code, so, se)
    with np.load(out2, allow_pickle=False) as z:
        assert np.array_equal(z["windows"], win)


CHECKS = [("test_t44a_ifr", test_t44a_ifr),
          ("test_t44b_partition", test_t44b_partition),
          ("test_t44c_windows", test_t44c_windows),
          ("test_t44d_refusals", test_t44d_refusals)]


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
