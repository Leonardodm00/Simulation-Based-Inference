#!/usr/bin/env python3
"""Smoke test of block 49: the session engine (specs/SPEC.md block 49, "Test
oracles", checks T49*).

A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t49a_describe       T49a: the electrodes and their spikes equal the folder's; spans
    test_t49b_statistics     T49b: counts, rates and the CV against numpy on planted trains
    test_t49c_exclusion      T49c: excluded electrodes as silent ones for the partition
    test_t49d_windows        T49d: the windows of a span; infer equals block 45's on them
    test_t49e_record         T49e: refusals; the result's keys; JSON-serialisable

Run
    SBIX_CLONE=<the Sbi-extractor's checkout> PYTHONDONTWRITEBYTECODE=1 \
        python3 sbitool/tests/test_smoke_tool_session.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<a kept fixture> reuses it.

Pure ASCII, LF only.
"""

import json
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

import numpy as np  # noqa: E402

SEED = 20261128  # every random number of this file derives from it
_TMP = []
_CASE = {}       # the folder, the bundle and the session, made once per process


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="sbitool_%s_" % tag)
    _TMP.append(d)
    return d


def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)
    if _T45:                 # block 45's test cleans block 41's project
        _T45[0]._cleanup()


# ------------------------------------------------------------ the case

GW, BASE, PITCH, FS = 4, 1, 60.0, 1000.0     # a 4 x 4 grid, ids 1..16
MFR_THR = 0.5      # Hz: > 0, so that a silent electrode is under the threshold (T49c's oracle)
EQUAL_ID, EQUAL_STEP, EQUAL_FIRST = 1, 250, 100   # a train of equal intervals (samples)
FEW_ID, SILENT_ID, FOUR_ID = 2, 3, 4              # 3 spikes; none; exactly 4
_T45 = []


def _t45():
    """Block 45's test module: its fixture's sidecar, extraction and export
    helpers (T45d), reused by import."""
    if not _T45:
        from sbitool.tests import test_smoke_tool_infer as T45
        _T45.append(T45)
    return _T45[0]


def _need_extractor():
    from sbitool import _sources
    try:
        _sources.extractor_dir()
    except _sources.SourceError as exc:
        raise unittest.SkipTest("the Sbi-extractor's checkout is absent (set SBIX_DIR or "
                                "SBIX_CLONE): %s" % exc)


def _quiet(*_a, **_k):
    return None


def _plant(rng, n_samples):
    """{electrode id: sorted sample indices}: a train of equal intervals, one
    of 3 spikes, a silent one, one of exactly 4, and twelve random trains of
    distinct counts (no tie in the ranking by rate)."""
    trains = {EQUAL_ID: np.arange(EQUAL_FIRST, n_samples, EQUAL_STEP, dtype=np.int64),
              FEW_ID: np.array([10, 2000, 15000], dtype=np.int64),
              SILENT_ID: np.zeros(0, dtype=np.int64),
              FOUR_ID: np.array([500, 1500, 4000, 20000], dtype=np.int64)}
    others = [e for e in range(BASE, BASE + GW * GW) if e not in trains]
    counts = 40 + 9 * rng.permutation(len(others))
    for e, k in zip(others, counts):
        trains[e] = np.sort(rng.choice(n_samples, size=int(k), replace=False)).astype(np.int64)
    return trains


def _write_folder(folder, trains, n_samples):
    """The extractor's raster format: ptrain_<id>.mat, an (n_samples, 1)
    array named ptrain, 1 at each spike."""
    import scipy.io as sio
    os.makedirs(folder, exist_ok=True)
    for e, t in trains.items():
        r = np.zeros((n_samples, 1), dtype=np.uint8)
        r[t, 0] = 1
        sio.savemat(os.path.join(folder, "ptrain_%d.mat" % e), {"ptrain": r}, format="5",
                    do_compression=True)
    return folder


def _options(**over):
    o = {"fs_raw_hz": FS, "grid_width": GW, "index_base": BASE, "pitch_um": PITCH}
    o.update(over)
    return o


def _case():
    """The planted folder, the bundle exported with an extraction at the
    bank's bin width (as T45d's, on a 4 x 4 grid with a positive rate
    threshold), the Recording read back by block 43, and one Session."""
    if _CASE:
        return _CASE
    _need_extractor()
    from sbitool import bundle, session
    from sbitool.recordings import ptrain
    T45 = _t45()
    rng = np.random.default_rng(SEED)
    side = T45._sim_sidecar()
    n_samples = T45._n_samples(side)
    trains = _plant(rng, n_samples)
    folder = _write_folder(os.path.join(_tmpdir("s49"), "rec"), trains, n_samples)
    ext = dict(T45._extraction(side), grid_width=GW, index_base=BASE, pitch_um=PITCH,
               fs_raw_hz=FS, mfr_threshold_hz=MFR_THR)
    bpath = T45._bundle_with_extraction(ext)
    b = bundle.read(bpath)
    assert b.manifest["extraction"] == ext, b.manifest["extraction"]
    rec = ptrain.read(folder, **_options())
    S = session.Session(bpath, folder, "ptrain", _options())
    _CASE.update(side=side, n_samples=n_samples, trains=trains, folder=folder, ext=ext,
                 bpath=bpath, bundle=b, rec=rec, S=S, W=int(b.manifest["W"]))
    return _CASE


def _session():
    from sbitool import session
    c = _case()
    return session.Session(c["bpath"], c["folder"], "ptrain", _options())


def _refused(fn, words=()):
    """fn() raises SessionError whose message holds every one of `words`."""
    from sbitool import session
    try:
        fn()
    except session.SessionError as exc:
        msg = str(exc)
        missing = [w for w in words if w not in msg]
        assert not missing, "refused, but the message %r does not name %s" % (msg, missing)
        return msg
    raise AssertionError("not refused (expected a SessionError naming %s)" % (list(words),))


def _times(c):
    """The planted spike times [s], computed by the test: sample / fs."""
    return {e: np.asarray(t, dtype=np.float64) / FS for e, t in c["trains"].items()}


def _bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ------------------------------------------------------------ T49a

def test_t49a_describe():
    """T49a: the electrodes and their spikes equal the folder's; spans (SPEC block 49)."""
    c = _case()
    S = c["S"]
    d = S.describe()
    json.dumps(d, allow_nan=False)
    T = c["n_samples"] / FS
    r = d["recording"]
    ids = sorted(c["trains"])
    assert r["duration_s"] == T and r["fs_raw_hz"] == FS, (r["duration_s"], r["fs_raw_hz"])
    assert r["n_electrodes"] == len(ids) == GW * GW and r["position_unit"] == "um", r
    assert [e["id"] for e in r["electrodes"]] == ids, [e["id"] for e in r["electrodes"]]
    for e in r["electrodes"]:
        i0 = e["id"] - BASE
        assert (e["x"], e["y"]) == ((i0 % GW) * PITCH, (i0 // GW) * PITCH), e
        assert e["n_spikes"] == c["trains"][e["id"]].size, e
    man = c["bundle"].manifest
    b = d["bundle"]
    assert b["name"] == man["name"] and b["sha256"] == c["bundle"].sha256, b["name"]
    assert b["card_complete"] == man["card_complete"]
    for k in ("parameters", "extraction", "sample", "gate"):
        assert b[k] == json.loads(json.dumps(man[k])), k
    assert isinstance(b["card"], dict) and "title" in b["card"], b["card"]
    assert d["W"] == c["W"] and abs(d["window_s"] - c["W"] * c["ext"]["w_size_s"]) < 1e-12, d
    times = _times(c)
    sp = S.spikes()
    assert sorted(sp) == ids
    for e in ids:
        assert np.array_equal(np.asarray(sp[e], dtype=np.float64), times[e]), e
    # a span whose ends are spike times: t0 included, t1 excluded
    eq = times[EQUAL_ID]
    t0, t1 = float(eq[3]), float(eq[40])
    sp = S.spikes(t0, t1)
    n_span = 0
    for e in ids:
        want = times[e][(times[e] >= t0) & (times[e] < t1)]
        assert np.array_equal(np.asarray(sp[e], dtype=np.float64), want), e
        n_span += want.size
    assert sp[EQUAL_ID][0] == t0 and sp[EQUAL_ID][-1] == float(eq[39]), sp[EQUAL_ID][-1]
    sub = S.spikes(t0, t1, electrodes=[7, 5])
    assert sorted(sub) == [5, 7] and sub[5] == sp[5] and sub[7] == sp[7]
    _refused(lambda: S.spikes(electrodes=[99]), ["99"])
    _refused(lambda: S.spikes(1.0, 0.5))
    print("[T49a] %d electrodes, %d spikes, T_rec %g s; span [%g, %g) s: %d spikes, t0 kept, t1 "
          "left out" % (len(ids), sum(t.size for t in c["trains"].values()), T, t0, t1, n_span))


# ------------------------------------------------------------ T49b

def test_t49b_statistics():
    """T49b: counts, rates and the CV against numpy on planted trains (SPEC block 49)."""
    from sbitool import session
    c = _case()
    S = c["S"]
    T = c["n_samples"] / FS
    times = _times(c)
    worst, n_cv = 0.0, 0
    spans = [(None, None, 0.0, T), (2.5, 17.25, 2.5, 17.25), (0.0, 1.6, 0.0, 1.6)]
    for a, b, t0, t1 in spans:
        st = S.statistics(a, b)
        assert sorted(st) == sorted(times)
        for e, t in times.items():
            v = st[e]
            assert tuple(v) == session.STAT_KEYS, tuple(v)
            tt = t[(t >= t0) & (t < t1)]
            assert v["n"] == tt.size and isinstance(v["n"], int), (e, v, tt.size)
            assert v["rate_hz"] == tt.size / (t1 - t0), (e, v["rate_hz"])
            if tt.size < 4:
                assert v["isi_cv"] is None, (e, v)
            else:
                isi = np.diff(tt)
                ref = float(np.std(isi) / np.mean(isi))
                err = abs(v["isi_cv"] - ref)
                assert err <= 1e-12, (e, v["isi_cv"], ref)
                worst, n_cv = max(worst, err), n_cv + 1
    st = S.statistics()
    assert st[FEW_ID]["isi_cv"] is None and st[SILENT_ID]["isi_cv"] is None
    assert st[SILENT_ID]["n"] == 0 and st[SILENT_ID]["rate_hz"] == 0.0
    assert st[FOUR_ID]["n"] == 4 and st[FOUR_ID]["isi_cv"] is not None
    cv_eq = st[EQUAL_ID]["isi_cv"]
    assert abs(cv_eq) <= 1e-9, cv_eq
    # exactly 3 of electrode FOUR_ID's spikes in a span: None
    assert S.statistics(0.0, 10.0)[FOUR_ID] == {"n": 3, "rate_hz": 0.3, "isi_cv": None}
    for a, b in ((-0.1, 1.0), (1.0, 1.0), (2.0, 1.0), (0.0, T + 0.5), (float("nan"), 1.0),
                 (0.0, float("inf")), ("0", 1.0)):
        _refused(lambda: S.statistics(a, b))
    print("[T49b] %d CVs against numpy, largest difference %.3g; equal intervals: CV %.3g; "
          "3 spikes and 0 spikes: None; 7 bad spans refused" % (n_cv, worst, cv_eq))


# ------------------------------------------------------------ T49c

def _silenced(rec, ids):
    """A copy of the Recording in which the electrodes `ids` have no spike."""
    import dataclasses
    sp = dict(rec.spikes)
    for e in ids:
        sp[e] = np.zeros(0, dtype=np.int64)
    return dataclasses.replace(rec, spikes=sp)


def _as_dicts(subs):
    return [{"index": j, "center": int(s.center), "members": [int(m) for m in s.members],
             "mean_mfr": float(s.mean_mfr), "center_mfr": float(s.center_mfr)}
            for j, s in enumerate(subs)]


def _extractor_message(rec, ext, silent):
    """The extractor's own InsufficientElectrodesError message for the
    recording with `silent` electrodes without spikes, by the test: the
    coordinates and rates built from the planted trains, partition_subregions
    called directly."""
    from sbitool import windows
    cse = windows._cse()
    T = rec.n_samples / rec.fs_raw_hz
    coords, mfrs = {}, {}
    for e in sorted(rec.spikes):
        i0 = e - BASE
        coords[e] = (i0 // GW, i0 % GW)
        mfrs[e] = 0.0 if e in silent else rec.spikes[e].size / T
    try:
        cse.partition_subregions(coords, mfrs, n_subsets=ext["n_subsets"],
                                 electrodes_per_subset=ext["electrodes_per_subset"],
                                 mfr_threshold=ext["mfr_threshold_hz"])
    except cse.InsufficientElectrodesError as exc:
        return str(exc)
    raise AssertionError("the extractor formed the partition with %s silent" % sorted(silent))


def test_t49c_exclusion():
    """T49c: excluded electrodes as silent ones for the partition (SPEC block 49)."""
    from sbitool import windows
    c = _case()
    S = _session()
    rec, ext = c["rec"], c["ext"]
    base = _as_dicts(windows.subregions(rec, ext))
    assert S.excluded == [] and S.subregions() == base
    S.set_excluded([])
    assert S.excluded == [] and S.subregions() == base, "excluding nothing changed the partition"
    base_x = S.traces()["x"]
    # the centres of the base partition, then an electrode of the second subregion
    cases = [[base[0]["center"]], [base[0]["center"], base[1]["members"][1]],
             [base[1]["center"], base[0]["members"][2], 16]]
    changed = 0
    for ex in cases:
        S.set_excluded(ex)
        assert S.excluded == sorted(ex), S.excluded
        got = S.subregions()
        want = _as_dicts(windows.subregions(_silenced(rec, ex), ext))
        assert got == want, (ex, got, want)
        members = set(m for s in got for m in s["members"]) | set(s["center"] for s in got)
        assert not members & set(ex), (ex, got)
        x = S.traces()["x"]
        x_want, _fs = windows.traces(_silenced(rec, ex), windows.subregions(_silenced(rec, ex),
                                                                           ext), ext)
        assert _bits_equal(x, x_want), ex
        changed += got != base
    assert changed == len(cases), "an exclusion left the partition as it was"
    # back to nothing: the base again
    S.set_excluded(())
    assert S.subregions() == base and _bits_equal(S.traces()["x"], base_x)
    # unknown ids: refused, by the session and by block 44
    _refused(lambda: S.set_excluded([99]), ["99"])
    _refused(lambda: S.set_excluded([True]))
    assert S.excluded == []
    try:
        windows.subregions(rec, ext, exclude=[0])
    except windows.WindowsError as exc:
        assert "0" in str(exc)
    else:
        raise AssertionError("windows.subregions took an unknown id")
    # too many excluded: the extractor's own refusal
    valid = sorted((e for e in rec.spikes if rec.spikes[e].size / (rec.n_samples / FS)
                    >= MFR_THR), key=lambda e: -rec.spikes[e].size)
    need = ext["n_subsets"] * ext["electrodes_per_subset"]
    ex = valid[:len(valid) - need + 1]
    S.set_excluded(ex)
    msg = _refused(lambda: S.subregions())
    want = _extractor_message(rec, ext, set(ex))
    assert want in msg, (msg, want)
    _refused(lambda: S.windows())
    # at a threshold of 0 (where a silent electrode is valid) an excluded one is still in no
    # subregion: the rule, not only the oracle
    ext0 = dict(ext, mfr_threshold_hz=0.0)
    ex0 = [base[0]["center"]]
    subs0 = windows.subregions(rec, ext0, exclude=ex0)
    assert not set(ex0) & set(m for s in subs0 for m in s.members)
    print("[T49c] %d valid electrodes, %d needed; %d exclusions equal the silenced copies' "
          "partition and traces; unknown ids refused; %d excluded: %r"
          % (len(valid), need, len(cases), len(ex), msg[:90]))


# ------------------------------------------------------------ T49d

def _block44(c, exclude=()):
    """Block 44's windows of the whole recording, by the test: subregions,
    traces and cut called one by one."""
    from sbitool import windows
    subs = windows.subregions(c["rec"], c["ext"], exclude=exclude)
    x, fs = windows.traces(c["rec"], subs, c["ext"])
    w, ids = windows.cut(x, c["W"], fs)
    return w, ids, fs, len(subs)


def _filtered(ids, W, fs, t0, t1):
    ts = ids["t_start_s"]
    return (ts >= t0) & (ts + W / fs <= t1)


def _phys(u, bounds, j):
    u = np.asarray(u, dtype=np.float64)
    if bounds is None:
        return u
    lo, hi = float(bounds[j][0]), float(bounds[j][1])
    return lo + u * (hi - lo)


def _compare_infer(S, c, t0, t1, n_post=None, seed=None, tag="a"):
    """S.infer on [t0, t1] against infer.run on block 44's windows of the
    span, computed by the test: the files bit for bit, the summary's numbers
    as the result's arrays (mapped where bounds are declared)."""
    from sbitool import infer
    w, ids, fs, _ns = _block44(c, S.excluded)
    sel = _filtered(ids, c["W"], fs, t0, t1)
    out = _tmpdir("s49d_" + tag)
    got = S.infer(t0, t1, n_post=n_post, seed=seed, out_dir=out)
    ref = infer.run(S._bundle, w[sel], ids={k: v[sel] for k, v in ids.items()}, n_post=n_post,
                    seed=seed, log=_quiet)
    A = ref["arrays"]
    with np.load(os.path.join(out, "posterior.npz")) as z:
        assert sorted(z.files) == sorted(A), (sorted(z.files), sorted(A))
        for k in A:
            assert _bits_equal(z[k], A[k]), "the session's %s is not infer.run's" % k
    assert got["files"] == [os.path.join(out, "posterior.npz"),
                            os.path.join(out, "posterior.json")], got["files"]
    par = S._bundle.manifest["parameters"]
    bounds = par["bounds"]
    assert got["coordinate"] == ("unit_box" if bounds is None else "physical")
    assert got["n"] == int(sel.sum()) and got["n_post"] == ref["json"]["n_post"]
    assert got["seed"] == ref["json"]["seed"] and got["batch_rows"] == ref["json"]["batch_rows"]
    for j, pj in enumerate(got["parameters"]):
        assert pj["name"] == par["names"][j] and pj["unit"] == par["units"][j]
        assert pj["bounds"] == (None if bounds is None else list(bounds[j]))
        for q in ("q05", "q50", "q95"):
            assert pj["pooled"][q] == float(_phys(A["mix_" + q][j], bounds, j)), (j, q)
        assert pj["windows_q50"] == _phys(A["q50"][:, j], bounds, j).tolist(), j
    g = got["gate"]
    assert g["flag"] == A["gate_flag"].tolist() and g["p"] == A["gate_p"].tolist()
    assert g["n_flagged"] == int(A["gate_flag"].sum()) == ref["json"]["gate"]["n_flagged"]
    assert g["alpha"] == S._bundle.manifest["gate"]["alpha"]
    assert g["share_flagged"] == ref["json"]["gate"]["share_flagged"]
    for k in ("subregion", "window_idx", "t_start_s"):
        assert got["windows"][k] == ids[k][sel].tolist(), k
    return got, ref


def test_t49d_windows():
    """T49d: the windows of a span; infer equals block 45's on them (SPEC block 49)."""
    c = _case()
    S = _session()
    T = c["n_samples"] / FS
    w, ids, fs, n_sub = _block44(c)
    ws = c["W"] / fs
    spans = [(0.0, T), (0.0, ws), (0.0, ws - 1e-9), (ws, T), (ws - 1e-9, T), (0.5, T),
             (0.1, 0.2), (0.0, 2 * ws)]
    counts = []
    for t0, t1 in spans:
        got = S.windows(t0, t1)
        sel = _filtered(ids, c["W"], fs, t0, t1)
        assert got["n"] == int(sel.sum()), (t0, t1, got["n"])
        for k in ("subregion", "window_idx", "t_start_s"):
            assert got[k] == ids[k][sel].tolist(), (t0, t1, k)
        assert got["per_subregion"] == [int(np.sum(ids["subregion"][sel] == s))
                                        for s in range(n_sub)], got["per_subregion"]
        counts.append(got["n"])
    assert S.windows() == S.windows(0.0, T)
    n_win = int(ids["subregion"].size)
    assert counts[0] == n_win and counts[1] == n_sub and counts[2] == 0, counts
    assert counts[3] == n_win - n_sub and counts[6] == 0, counts
    # the grid is the recording's: a window starts at i window_s whatever the span
    got = S.windows(0.5, T)
    assert all(abs(t - i * ws) < 1e-12 for t, i in zip(got["t_start_s"], got["window_idx"]))
    # under an exclusion the windows are those of the new subregions
    ex = [S.subregions()[0]["center"]]
    S.set_excluded(ex)
    w2, ids2, fs2, _n = _block44(c, ex)
    assert S.windows()["subregion"] == ids2["subregion"].tolist()
    S.set_excluded([])
    # the inference on a span
    M = int(S._bundle.manifest["M"])
    r1, _ = _compare_infer(S, c, ws, T, tag="a")
    r2, _ = _compare_infer(S, c, 0.0, T, n_post=2 * M, seed=7, tag="b")
    # the map to physical coordinates, on sessions whose manifest is set by the test: bounds
    # with lo != 0 on every axis (the bundle's may start at 0), and no bounds
    S2 = _session()
    par = S2._bundle.manifest["parameters"]
    p = len(par["names"])
    par["bounds"] = [[-1.0 - j, 2.0 + 3.0 * j] for j in range(p)]
    r3, _ = _compare_infer(S2, c, 0.0, T, tag="c")
    S3 = _session()
    S3._bundle.manifest["parameters"]["bounds"] = None
    r4, _ = _compare_infer(S3, c, 0.0, T, tag="d")
    assert (r3["coordinate"], r4["coordinate"]) == ("physical", "unit_box")
    assert r3["parameters"][0]["pooled"] != r4["parameters"][0]["pooled"]
    print("[T49d] %d windows of %d subregions, window_s %g s; spans gave %s windows; infer on "
          "[%g, %g]: %d windows, files bit for bit, %d flagged; with n_post %d seed 7: %d; "
          "coordinates: the bundle's %s (bounds %s), set by the test %s and %s"
          % (n_win, n_sub, ws, counts, ws, T, r1["n"], r1["gate"]["n_flagged"], 2 * M, r2["n"],
             r1["coordinate"], c["bundle"].manifest["parameters"]["bounds"], r3["coordinate"],
             r4["coordinate"]))


# ------------------------------------------------------------ T49e

def test_t49e_record():
    """T49e: refusals; the result's keys; JSON-serialisable (SPEC block 49)."""
    from sbitool import session
    c = _case()
    folder, bpath = c["folder"], c["bpath"]
    Sess = session.Session
    # a device that does not fit the bundle: the rate, the pitch
    _refused(lambda: Sess(bpath, folder, "ptrain", _options(fs_raw_hz=2000.0)),
             ["2000.0", "1000.0"])
    _refused(lambda: Sess(bpath, folder, "ptrain", _options(pitch_um=50.0)), ["50.0", "60.0"])
    # an unreadable recording: no such folder, a folder of something else, an unknown adapter,
    # a missing rate, an option the adapter does not take
    _refused(lambda: Sess(bpath, os.path.join(folder, "nope"), "ptrain", _options()))
    junk = _tmpdir("s49e_junk")
    with open(os.path.join(junk, "ptrain_1.mat"), "w") as fh:
        fh.write("not a mat file\n")
    _refused(lambda: Sess(bpath, junk, "ptrain", _options()))
    _refused(lambda: Sess(bpath, folder, "nope", _options()), ["nope"])
    _refused(lambda: Sess(bpath, folder, "ptrain", {"grid_width": GW}), ["fs_raw_hz"])
    _refused(lambda: Sess(bpath, folder, "ptrain", _options(colour="red")))
    _refused(lambda: Sess(bpath, folder, "ptrain", _options(detection={})), ["detection"])
    # a bundle that is not one; a bundle without extraction
    _refused(lambda: Sess(os.path.join(junk, "ptrain_1.mat"), folder, "ptrain", _options()))
    _refused(lambda: Sess(os.path.join(junk, "absent.sbibundle"), folder, "ptrain",
                          _options()))
    T41 = _t45()._t41()
    plain = T41._ctx()["bundles"][T41.ENSEMBLES[0]][0]
    from sbitool import bundle
    if bundle.read(plain).manifest["extraction"] is None:
        _refused(lambda: Sess(plain, folder, "ptrain", _options()), ["extraction"])
        no_ext = "refused"
    else:
        no_ext = "not tested (the fixture's bundle has an extraction)"
    # infer with no window
    S = _session()
    T = c["n_samples"] / FS
    _refused(lambda: S.infer(0.1, 0.2), ["no window"])
    _refused(lambda: S.infer(0.0, T + 1.0))
    M = int(S._bundle.manifest["M"])
    _refused(lambda: S.infer(n_post=M + 1), ["multiple"])
    # the record
    ex = [S.subregions()[0]["center"], 16]
    S.set_excluded(ex)
    ws = S.describe()["window_s"]
    res = S.infer(0.0, T, n_post=M)
    assert tuple(res) == session.RESULT_KEYS, tuple(res)
    text = json.dumps(res, allow_nan=False)
    assert json.loads(text) == res, "the result does not survive JSON"
    assert res["excluded"] == sorted(ex) and res["span"] == [0.0, T], (res["excluded"],
                                                                      res["span"])
    assert res["files"] is None
    for pj in res["parameters"]:
        assert tuple(pj) == session.PARAMETER_KEYS and tuple(pj["pooled"]) == session.POOLED_KEYS
        assert len(pj["windows_q50"]) == res["n"]
    assert tuple(res["gate"]) == session.GATE_KEYS
    assert tuple(res["windows"]) == session.WINDOW_KEYS
    assert len(res["gate"]["flag"]) == len(res["gate"]["p"]) == res["n"]
    assert res["n"] == S.windows()["n"] and res["n_post"] == M
    print("[T49e] the rate, the pitch, 5 unreadable recordings, the user's detection, 2 bad "
          "bundles refused; no extraction: %s; no window and a bad span refused; the record: "
          "%d keys, JSON round trip, excluded %s, span [0, %g], window_s %g"
          % (no_ext, len(res), res["excluded"], T, ws))


CHECKS = [("test_t49a_describe", test_t49a_describe),
          ("test_t49b_statistics", test_t49b_statistics),
          ("test_t49c_exclusion", test_t49c_exclusion),
          ("test_t49d_windows", test_t49d_windows),
          ("test_t49e_record", test_t49e_record)]


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
