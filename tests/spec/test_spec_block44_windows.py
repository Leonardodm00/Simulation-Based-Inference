"""Spec tests of block 44 (from a recording to windows; specs/SPEC.md sec.
2.7, "From a recording to windows", and block 44 with its wave-B
amendments, D-177). Integration tester, 2026-11-28, incremental run on
39e6f66.

Oracles: the IFR written here from sec. 2.7's words [as corrected by D-180,
2026-11-28_0151: the spikes of the n_e electrodes counted in the bins of
edges k Delta t (float64), K = int(T_rec / Delta t); smoothed by the Gaussian
of width sigma_sm, scipy's gaussian_filter1d; divided by n_e: spikes per bin
per electrode];
floor(K / W) consecutive windows from the start; t_start = window_idx W /
fs_ifr; the partition's invariants and D-177's "one device gives one
partition whatever its channels are called". Needs SBIX_CLONE.
"""

import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import _tool_util as U  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(U.SBIX), reason="no SBIX_CLONE")


def _K(n_samples, fs, dt):
    """The number of whole bins of width dt in T_rec = n_samples / fs, in exact
    arithmetic on the decimal values given (sec. 2.7's first wording; kept to show where
    the imported rule differs from it)."""
    return int(Fraction(n_samples) / Fraction(repr(fs)) // Fraction(repr(dt)))


def _K_d180(n_samples, fs, dt):
    """K as sec. 2.7 states it since D-180 (2026-11-28): int(T_rec / Delta t) in float64."""
    return int((float(n_samples) / float(fs)) / float(dt))


def _ifr_spec(rec, members, dt, sigma, per_hz=False):
    """sec. 2.7 as corrected by D-180: the members' spikes counted in the bins whose edges
    are k Delta t in float64, k = 0..K, K = int(T_rec / Delta t) (numpy.histogram on those
    edges, the reference implementation of "counted in bin k"), smoothed by the Gaussian
    of width sigma_sm (scipy's gaussian_filter1d, sigma_sm / Delta t bins), clipped at 0,
    divided by n_e: spikes per bin per electrode. per_hz=True also divides by Delta t (the
    first wording, Hz per electrode)."""
    K = _K_d180(rec.n_samples, rec.fs_raw_hz, dt)
    edges = np.arange(K + 1, dtype=np.float64) * dt
    c = np.zeros(K)
    for e in members:
        t = rec.spikes[e].astype(np.float64) / rec.fs_raw_hz
        c += np.histogram(t, bins=edges)[0]
    c /= len(members) * (dt if per_hz else 1.0)
    return np.clip(gaussian_filter1d(c, sigma / dt), 0.0, None), K


def _nine(rng, n, rate_hz=40.0, fs=10000.0):
    """A 6 x 6 grid recording; the first subregion's electrodes active."""
    sp = {}
    for e in range(36):
        m = rng.poisson(rate_hz * n / fs)
        # no spike on a bin boundary (a multiple of fs dt = 100 samples): the boundary is
        # tested apart (test_t44a_spike_on_a_bin_boundary)
        cand = np.setdiff1d(np.arange(n), np.arange(0, n, 100))
        sp[e] = np.sort(rng.choice(cand, size=m, replace=False)).astype(np.int64)
    return U.grid_recording(sp, n, fs=fs)


def test_t44a_ifr_units_by_the_spec():
    """Sec. 2.7 and T44a as corrected by D-180: x is the smoothed count per bin per
    electrode (EXTRACTOR_USAGE.md eq. (3)); x / Delta t is Hz per electrode."""
    from sbitool import windows
    rng = np.random.default_rng(4401)
    rec = _nine(rng, 20000)
    ext = U.extraction(n_subsets=1, electrodes_per_subset=9)
    subs = windows.subregions(rec, ext)
    x, fs_ifr = windows.traces(rec, subs, ext)
    assert x.dtype == np.float32 and fs_ifr == pytest.approx(100.0)
    want, K = _ifr_spec(rec, [int(m) for m in subs[0].members], 0.01, 0.02)
    assert x.shape == (1, K)
    # 1e-5 relative to the largest value: the extractor returns float32 (block 44's T44a)
    np.testing.assert_allclose(x[0], want, rtol=0, atol=1e-5 * np.max(want))
    hz, _ = _ifr_spec(rec, [int(m) for m in subs[0].members], 0.01, 0.02, per_hz=True)
    # and the trace is not the first wording's Hz per electrode (100 times the values)
    assert np.max(hz) / np.max(x[0]) == pytest.approx(100.0, rel=1e-5)


def test_t44a_number_of_bins_d180():
    """D-180 (the tester's F3): K = int(T_rec / Delta t) in float64. 2900 samples at 10 kHz
    with Delta t = 0.01 s: 0.29 / 0.01 = 28.999999999999996, K = 28 (29 whole bins in
    exact arithmetic), so a window of 29 bins does not fit (NoWindow by `extract`)."""
    from sbitool import windows
    n, fs, dt = 2900, 10000.0, 0.01
    assert _K(n, fs, dt) == 29 and _K_d180(n, fs, dt) == 28
    sp = {e: np.array([80 * e + 5], dtype=np.int64) for e in range(36)}
    rec = U.grid_recording(sp, n, fs=fs)
    ext = U.extraction(n_subsets=1, electrodes_per_subset=9)
    subs = windows.subregions(rec, ext)
    x, _ = windows.traces(rec, subs, ext)
    assert x.shape[1] == 28
    w, _ids = windows.cut(x, 28, 100.0)
    assert w.shape[0] == 1
    w, _ids = windows.cut(x, 29, 100.0)
    assert w.shape[0] == 0


def test_t44a_spike_on_a_bin_boundary_d180():
    """D-180 (the tester's F4): the bins' edges are k Delta t in float64, so a spike at
    sample 11300 of a 10 kHz recording (t = 1.13 s) is in bin 112, since 113 * 0.01 =
    1.1300000000000001 > 1.13. With sigma_sm / Delta t = 1e-4 the trace is the counts."""
    from sbitool import windows
    n = 20000
    sp = {e: np.array([11300], dtype=np.int64) for e in range(36)}
    rec = U.grid_recording(sp, n)
    ext = U.extraction(n_subsets=1, electrodes_per_subset=1, gaussian_window_s=1e-6)
    subs = windows.subregions(rec, ext)
    x, _ = windows.traces(rec, subs, ext)
    want, _K2 = _ifr_spec(rec, [int(m) for m in subs[0].members], 0.01, 1e-6)
    assert list(np.flatnonzero(want)) == [112]
    assert list(np.flatnonzero(x[0])) == [112]


def test_t44a_spike_on_the_last_edge():
    """A spike exactly at K Delta t of a recording longer than K bins (20050 samples at
    10 kHz, K = 200, a spike at sample 20000: t = 2.0 = 200 * 0.01 exactly). By the bins
    [k Dt, (k + 1) Dt), k < K, it is in no bin; numpy.histogram on the edges k Dt (the
    bins sec. 2.7 now names) closes its last bin and counts it in bin K - 1. The oracle
    is numpy's; the half-open reading is recorded beside it (the report's S1)."""
    from sbitool import windows
    n = 20050
    sp = {e: np.array([20000], dtype=np.int64) for e in range(36)}
    rec = U.grid_recording(sp, n)
    ext = U.extraction(n_subsets=1, electrodes_per_subset=1, gaussian_window_s=1e-6)
    subs = windows.subregions(rec, ext)
    x, _ = windows.traces(rec, subs, ext)
    want, K = _ifr_spec(rec, [int(m) for m in subs[0].members], 0.01, 1e-6)
    assert K == 200 and 200 * 0.01 == 20000 / 10000.0
    assert list(np.flatnonzero(want)) == [199]                    # numpy closes the last bin
    assert list(np.flatnonzero(x[0])) == [199], list(np.flatnonzero(x[0]))


def test_t44c_windows_from_the_start():
    from sbitool import windows
    rng = np.random.default_rng(4403)
    x = rng.random((3, 107)).astype(np.float32)
    for W in (1, 10, 53, 107, 108):
        w, ids = windows.cut(x, W, 100.0)
        nw = 107 // W
        assert w.shape == (3 * nw, W) and w.dtype == np.float32
        for s in range(3):
            for j in range(nw):
                r = s * nw + j
                np.testing.assert_array_equal(w[r], x[s, j * W:(j + 1) * W])
                assert ids["subregion"][r] == s and ids["window_idx"][r] == j
                # t_start = window_idx W / fs_ifr [s], exact in float64 here
                assert ids["t_start_s"][r] == j * W / 100.0
        assert ids["subregion"].dtype == np.int64 and ids["window_idx"].dtype == np.int64
    # float32 of a float64 trace
    x64 = rng.random((1, 20))
    w, _ = windows.cut(x64, 10, 100.0)
    np.testing.assert_array_equal(w, x64[:, :20].reshape(2, 10).astype(np.float32))


def test_t44b_partition_invariants():
    from sbitool import windows
    rng = np.random.default_rng(4404)
    n, fs = 50000, 10000.0
    rates = rng.uniform(0.0, 20.0, size=36)
    sp = {e: np.sort(rng.choice(n, size=int(rates[e] * n / fs), replace=False)).astype(np.int64)
          for e in range(36)}
    rec = U.grid_recording(sp, n, fs=fs)
    thr = 2.0
    ext = U.extraction(n_subsets=3, electrodes_per_subset=4, mfr_threshold_hz=thr)
    subs = windows.subregions(rec, ext)
    assert len(subs) == 3
    seen = set()
    for s in subs:
        m = [int(e) for e in s.members]
        assert len(m) == 4 and not (set(m) & seen)
        seen |= set(m)
        assert int(s.center) in m
        for e in m:
            assert sp[e].size / (n / fs) >= thr          # every member above the threshold


def _device(rng, n=20000, fs=10000.0, ties=True):
    """{(x, y): train} of a 5 x 5 patch at pitch 60 um; with ties every electrode has
    the same number of spikes (the partition breaks the ties by its key)."""
    out = {}
    for r in range(5):
        for c in range(5):
            m = 20 if ties else int(rng.integers(5, 40))
            out[(c * 60.0, r * 60.0)] = np.sort(rng.choice(n, size=m, replace=False))
    return out


def _by_pos(rec, subs):
    return [(rec.positions_um[int(s.center)],
             sorted(rec.positions_um[int(m)] for m in s.members)) for s in subs]


@pytest.mark.parametrize("ties", [True, False])
def test_d177_one_device_one_partition(ties):
    """D-177: for a recording with positions and no grid, the partition does not depend on
    the names its file gives the electrodes; and it is the partition of the same device
    read with a grid (the key is the electrode's place on the device)."""
    from sbitool import windows
    rng = np.random.default_rng(4405)
    dev = _device(rng, ties=ties)
    ext = U.extraction(n_subsets=2, electrodes_per_subset=4, grid_width=5, pitch_um=60.0)
    a = U.positions_recording(dev, 20000)
    order = sorted(dev)
    names = dict((xy, 1000 - 7 * i) for i, xy in enumerate(order))      # reversed, sparse
    b = U.positions_recording(dev, 20000, rename=names)
    pa, pb = _by_pos(a, windows.subregions(a, ext)), _by_pos(b, windows.subregions(b, ext))
    assert pa == pb
    # the same device with a grid of width 5: electrode id = row * 5 + col
    g = U.grid_recording({int(round(y / 60.0)) * 5 + int(round(x / 60.0)): t
                          for (x, y), t in dev.items()}, 20000, width=5, pitch=60.0)
    pg = _by_pos(g, windows.subregions(g, ext))
    assert pa == pg
    # and the traces: the same subregions give the same windows
    xa, _ = windows.traces(a, windows.subregions(a, ext), ext)
    xb, _ = windows.traces(b, windows.subregions(b, ext), ext)
    np.testing.assert_array_equal(xa, xb)


def test_t44d_refusals_name_both_values():
    from sbitool import windows
    rng = np.random.default_rng(4406)
    rec = _nine(rng, 5000)
    ext = U.extraction(n_subsets=1)
    cases = [
        (U.extraction(n_subsets=1, fs_raw_hz=20000.0), ("10000", "20000")),
        (U.extraction(n_subsets=1, grid_width=48), ("6", "48")),
        (U.extraction(n_subsets=1, index_base=1), ("0", "1")),
        (U.extraction(n_subsets=1, pitch_um=42.0), ("60", "42")),
    ]
    for e, vals in cases:
        with pytest.raises(windows.WindowsError) as ei:
            windows.extract(rec, e, 10)
        msg = str(ei.value)
        assert all(v in msg for v in vals), msg
    assert windows.extract(rec, ext, 10)[0].shape[1] == 10
    # positions only: the pitch within 1 um is accepted, beyond refused
    dev = {(c * 60.8, r * 60.8): np.arange(0, 5000, 50) for r in range(3) for c in range(3)}
    p = U.positions_recording(dev, 5000)
    e9 = U.extraction(n_subsets=1, electrodes_per_subset=9, grid_width=3)
    assert windows.extract(p, e9, 10)[0].shape[0] == 5
    dev2 = {(c * 61.5, r * 61.5): np.arange(0, 5000, 50) for r in range(3) for c in range(3)}
    with pytest.raises(windows.WindowsError) as ei:
        windows.extract(U.positions_recording(dev2, 5000), e9, 10)
    assert "61.5" in str(ei.value) and "60" in str(ei.value)


def _cli(args, timeout=300):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SBIX_CLONE=U.SBIX)
    r = subprocess.run([sys.executable, "-m", "sbitool"] + args, cwd=U.REPO, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def test_command_windows():
    from sbitool import windows
    rng = np.random.default_rng(4407)
    d = tempfile.mkdtemp(prefix="w44_")
    n = 20000
    sp = {e: np.sort(rng.choice(n, size=int(rng.integers(20, 80)), replace=False))
          for e in range(36)}
    U.write_ptrain(os.path.join(d, "rec"), sp, n)
    ext = U.extraction(n_subsets=2, electrodes_per_subset=9)
    b = U.mini_bundle(os.path.join(d, "b.sbibundle"), ext, W=10)
    opts = ["--option", "fs_raw_hz=10000.0", "--option", "grid_width=6", "--option",
            "index_base=0", "--option", "pitch_um=60.0"]
    out = os.path.join(d, "w.npz")
    rc, txt = _cli(["windows", "--bundle", b, "--recording", os.path.join(d, "rec"),
                    "--adapter", "ptrain", "--out", out] + opts)
    assert rc == 0 and "WINDOWS OK: 40 window(s) of 2 subregion(s)" in txt, txt
    rec = U.grid_recording(sp, n)
    w, ids, _ = windows.extract(rec, ext, 10)
    with np.load(out) as z:
        np.testing.assert_array_equal(z["windows"], w)
        for k in ("subregion", "window_idx", "t_start_s"):
            np.testing.assert_array_equal(z[k], ids[k])
    # the detection is the bundle's: an option detection is an argument error (exit 2)
    rc, txt = _cli(["windows", "--bundle", b, "--recording", os.path.join(d, "rec"),
                    "--adapter", "ptrain", "--out", out + "2", "--option",
                    'detection={"method": "by_channel"}'] + opts)
    assert rc == 2 and not os.path.exists(out + "2"), txt
    # a bundle without extraction refuses a recording (exit 3)
    b0 = U.mini_bundle(os.path.join(d, "b0.sbibundle"), None, W=10)
    rc, txt = _cli(["windows", "--bundle", b0, "--recording", os.path.join(d, "rec"),
                    "--adapter", "ptrain", "--out", out + "3"] + opts)
    assert rc == 3 and "REFUSED" in txt and not os.path.exists(out + "3"), txt
    # a recording at another rate (exit 3, both values named)
    rc, txt = _cli(["windows", "--bundle", b, "--recording", os.path.join(d, "rec"),
                    "--adapter", "ptrain", "--out", out + "4", "--option",
                    "fs_raw_hz=20000.0", "--option", "grid_width=6", "--option",
                    "index_base=0"])
    assert rc == 3 and "20000" in txt and "10000" in txt and not os.path.exists(out + "4"), txt
