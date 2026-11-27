#!/usr/bin/env python3
"""Smoke test of block 45: inference from a bundle (specs/SPEC.md block 45, "Test
oracles" and "Contract of wave C", checks T45*).

A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t45a_evaluation     the evaluation's numbers: block 35's file bit for bit, block 36's
    test_t45b_mixture        the mixture over the windows, by numpy on the pooled draws
    test_t45c_contract       keys, shapes, dtypes; one seed, one result; seed and batch matter
    test_t45d_end_to_end     a ptrain folder through windows and infer by the command line
    test_t45e_refusals       n_post not a multiple of M, another W, a broken bundle

Run
    SBIX_CLONE=<the Sbi-extractor's checkout> PYTHONDONTWRITEBYTECODE=1 \
        python3 sbitool/tests/test_smoke_tool_infer.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<dir> reuses a fixture tree built before
        (hpc/pipeline/tests/rec_fixture.py). The evaluated project and its
        bundles are block 41's test's (test_smoke_tool_bundle._ctx()).

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
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

SEED = 20261127  # every random number of this file derives from it
_TMP = []
_RUN = {}        # T45a's inferences, reused by T45b and T45c


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="sbitool_%s_" % tag)
    _TMP.append(d)
    return d


def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)
    if _T41:                 # block 41's evaluated project and bundles
        _T41[0]._cleanup()


# ------------------------------------------------------------ helpers

_T41 = []


def _t41():
    """Block 41's test module: its _ctx() is the evaluated fixture project and
    its bundles, built once per process (not built again here)."""
    if not _T41:
        from sbitool.tests import test_smoke_tool_bundle as T41
        _T41.append(T41)
    return _T41[0]


def _ctx():
    return _t41()._ctx()


def _quiet(*_a, **_k):
    return None


def _bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _max_abs(a, b):
    return _t41()._max_abs(a, b)


def _bank_case(name):
    """(bundle, e, prov, r, rec) of ensemble `name` on the fixture's one
    recording bank: the bundle of block 41's test, the sample's provenance,
    the bank's rows by ev_rec_data.recording."""
    import ev_rec_data
    from sbitool import bundle
    c = _ctx()
    e, prov = _t41()._ensemble(name)
    banks = sorted(c["view"].banks)
    assert len(banks) == 1, "the fixture has banks %s, expected one" % banks
    r = banks[0]
    rec = ev_rec_data.recording(c["view"], r)
    return bundle.read(c["bundles"][name][0]), e, prov, r, rec


def _inference(name):
    """infer.run on the bank's windows with block 35's seed for that bank,
    n_post and batch left to the bundle (cached)."""
    if name not in _RUN:
        from sbitool import infer
        b, e, prov, r, rec = _bank_case(name)
        res = infer.run(b, rec["x"], seed=int(prov["seeds"][r]), log=_quiet)
        _RUN[name] = dict(bundle=b, e=e, prov=prov, r=r, rec=rec, res=res)
    return _RUN[name]


def _shape(text, dims):
    return tuple(int(dims[t.strip()]) for t in text.strip("()").split(",") if t.strip())


def _npz_members(path):
    """{member name: uncompressed bytes} of an npz (a zip): the arrays' .npy
    files, header and data."""
    with zipfile.ZipFile(path) as zf:
        return dict((n, zf.read(n)) for n in zf.namelist())


def _main(fn, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(argv)
    return code, out.getvalue(), err.getvalue()


def _env():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("SBI_HPC_DIR", os.path.join(_REPO, "hpc"))
    return env


def _cli(argv):
    """(exit code, stdout, stderr) of `python3 -m sbitool <argv>`, a process of
    its own, from the repository's root."""
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "sbitool"] + list(argv),
                       cwd=_REPO, capture_output=True, text=True, env=_env(), timeout=600)
    return r.returncode, r.stdout, r.stderr


def _last(text):
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


def _need_extractor():
    from sbitool import _sources
    try:
        _sources.extractor_dir()
    except _sources.SourceError as exc:
        raise unittest.SkipTest("the Sbi-extractor's checkout is absent (set SBIX_DIR or "
                                "SBIX_CLONE): %s" % exc)


# ------------------------------------------------------------ T45a

_ESAMPLE = ("draws", "draw_member", "logq_draws", "logq_draws_own", "z")
_SUMMARY = ("mhat", "sigma", "q05", "q50", "q95", "contraction")
_GATE = ("gate_t", "gate_r", "gate_p_member", "gate_T", "gate_p")
GATE_TOL = 1e-12   # contract: the gate's float64 arrays; both sides run ev_gate.gate on
#                    float32 sets of the same bits, so any difference would be roundoff


def test_t45a_evaluation():
    """T45a: on the fixture's bank, with the sample's seed and the bundle's n_post and batch, draws,
    draw_member, logq_draws, logq_draws_own and z equal block 35's file bit for bit; the
    summaries, the gate's arrays and the flags equal block 36's."""
    T41 = _t41()
    c = _ctx()
    lines = []
    for name in T41.ENSEMBLES:
        run = _inference(name)
        e, r, res, b = run["e"], run["r"], run["res"], run["bundle"]
        A = res["arrays"]
        man = b.manifest
        assert res["json"]["n_post"] == man["sample"]["n_post"] == run["prov"]["n_post"]
        assert res["json"]["batch_rows"] == man["sample"]["batch_rows"]
        assert A["draws"].shape[0] > res["json"]["batch_rows"], \
            "the bank's %d windows do not exceed the batch %d" % (A["draws"].shape[0],
                                                                 res["json"]["batch_rows"])
        sp = os.path.join(c["out_root"], "samples", "%s_%s.npz" % (e.stem, r))
        mp = os.path.join(c["out_root"], "metrics", "%s_%s.npz" % (e.stem, r))
        with np.load(sp) as z:
            for k in _ESAMPLE:
                assert _bits_equal(A[k], z[k]), "%s: %s differs from block 35's %s: %s %s vs %s " \
                    "%s, max |diff| %.3g" % (name, k, sp, A[k].dtype, A[k].shape, z[k].dtype,
                                             z[k].shape, _max_abs(A[k], z[k]))
        worst = 0.0
        with np.load(mp) as z:
            for k in _SUMMARY:
                assert _bits_equal(A[k], z[k]), "%s: %s differs from block 36's %s: max |diff| " \
                    "%.3g (%s vs %s)" % (name, k, mp, _max_abs(A[k], z[k]), A[k].dtype,
                                         z[k].dtype)
            for k in _GATE:
                assert A[k].dtype == np.float64 and A[k].shape == z[k].shape, (k, A[k].dtype,
                                                                              A[k].shape)
                d = _max_abs(A[k], z[k])
                worst = max(worst, d)
                assert d <= GATE_TOL, "%s: %s differs from block 36's by %.3g" % (name, k, d)
            assert A["gate_flag"].dtype == np.bool_
            assert np.array_equal(A["gate_flag"], z["gate_flag"]), (name, A["gate_flag"],
                                                                     z["gate_flag"])
            n_fl = int(z["gate_flag"].sum())
        assert res["json"]["gate"]["n_flagged"] == n_fl
        lines.append("%s: %d windows x %d draws, the 5 sample arrays and 6 summaries bit for bit, "
                     "the gate's arrays max |diff| %.3g, %d flagged"
                     % (name, A["draws"].shape[0], A["draws"].shape[1], worst, n_fl))
    print("[T45a] " + "; ".join(lines))


# ------------------------------------------------------------ T45b

MIX_TOL = 1e-12   # contract: float64 of the same pooled float32 draws; numpy's own


def test_t45b_mixture():
    """T45b: mix_* equal numpy's mean, SD (ddof 1) and quantiles of the result's own draws pooled
    over the windows as float64, within 1e-12."""
    worst, one_well = 0.0, []
    for name in _t41().ENSEMBLES:
        A = _inference(name)["res"]["arrays"]
        d = A["draws"]
        p = d.shape[2]
        pooled = d.reshape(-1, p).astype(np.float64)
        q = np.quantile(pooled, [0.05, 0.5, 0.95], axis=0)
        want = {"mix_mean": pooled.mean(axis=0), "mix_sd": pooled.std(axis=0, ddof=1),
                "mix_q05": q[0], "mix_q50": q[1], "mix_q95": q[2]}
        for k, v in want.items():
            assert A[k].dtype == np.float64 and A[k].shape == (p,), (k, A[k].dtype, A[k].shape)
            dd = _max_abs(A[k], v)
            worst = max(worst, dd)
            assert dd <= MIX_TOL, "%s: %s differs from numpy's by %.3g" % (name, k, dd)
        # the fixture's recording is one well: block 36's well_summary of that well too
        rec = _inference(name)["rec"]
        if np.unique(rec["well"]).size == 1:
            mp = os.path.join(_ctx()["out_root"], "metrics", "%s_%s.npz"
                              % (_inference(name)["e"].stem, _inference(name)["r"]))
            with np.load(mp) as z:
                for k in want:
                    dd = _max_abs(A[k], z["well_" + k[4:]][0])
                    assert dd <= MIX_TOL, "%s: %s differs from block 36's by %.3g" % (name, k, dd)
            one_well.append(name)
    print("[T45b] the mixture over the windows: max |diff| to numpy %.3g; equal to block 36's "
          "well_summary of the one well for %s" % (worst, one_well or "none (several wells)"))


# ------------------------------------------------------------ T45c

def test_t45c_contract():
    """T45c: every key, shape and dtype of ARRAYS and JSON_KEYS; two runs with one seed give
    identical files; another seed gives other draws and the same z; another batch gives other
    draws when n exceeds it."""
    from sbitool import infer
    name = _t41().ENSEMBLES[0]
    run = _inference(name)
    b, rec, res = run["bundle"], run["rec"], run["res"]
    man = b.manifest
    seed = int(run["prov"]["seeds"][run["r"]])
    n = rec["x"].shape[0]
    ids = {"well": rec["well"].astype(np.int32), "label": np.array(["w%d" % i for i in range(n)])}
    res1 = infer.run(b, rec["x"], ids=ids, seed=seed, log=_quiet)
    A, js = res1["arrays"], res1["json"]
    dims = {"n": n, "n_post": man["sample"]["n_post"], "p": man["p"], "M": man["M"],
            "d_z": man["d_z"]}
    want_keys = set(infer.ARRAYS) | {"rows"} | set(ids)
    assert set(A) == want_keys, "arrays %s, expected %s" % (sorted(A), sorted(want_keys))
    for k, (shape, dtype) in infer.ARRAYS.items():
        assert A[k].shape == _shape(shape, dims) and str(A[k].dtype) == dtype, \
            "%s is %s %s, expected %s %s" % (k, A[k].dtype, A[k].shape, dtype, _shape(shape, dims))
    assert A["rows"].dtype == np.int64 and np.array_equal(A["rows"], np.arange(n))
    for k, v in ids.items():
        assert _bits_equal(A[k], v), "id %s not written as given" % k
    assert np.all(np.isfinite(A["draws"])) and np.all(np.isfinite(A["logq_draws"]))
    assert np.array_equal(A["draw_member"], np.repeat(np.arange(dims["M"]),
                                                      dims["n_post"] // dims["M"]))
    assert tuple(js) == infer.JSON_KEYS, tuple(js)
    assert tuple(js["gate"]) == infer.JSON_GATE_KEYS, tuple(js["gate"])
    assert js["schema"] == infer.POSTERIOR_SCHEMA
    assert js["bundle"] == {"name": man["name"], "sha256": b.sha256}
    assert (js["n"], js["n_post"], js["M"], js["p"], js["seed"], js["batch_rows"]) == (
        n, dims["n_post"], dims["M"], dims["p"], seed, man["sample"]["batch_rows"])
    assert js["parameters"] == man["parameters"]
    for k in infer.JSON_GATE_KEYS[:7]:
        assert js["gate"][k] == man["gate"][k], k
    assert js["gate"]["n_flagged"] == int(A["gate_flag"].sum())
    assert js["gate"]["share_flagged"] == int(A["gate_flag"].sum()) / float(n)
    assert js["ids"] == list(ids)
    assert "not the posterior of the recording" in js["mixture_note"], js["mixture_note"]
    from sbitool import schema as S
    assert sorted(js["software"]) == sorted(man["software"]) == sorted(S.SOFTWARE_KEYS), \
        (js["software"], man["software"])
    assert all(isinstance(v, str) and v for v in js["software"].values()), js["software"]
    # the files: two runs with one seed, the same content; the file read back is the result
    d1, d2 = _tmpdir("i45c1"), os.path.join(_tmpdir("i45c2"), "new", "folder")
    npz1, json1 = infer.write(res1, d1)
    res2 = infer.run(b, rec["x"], ids=ids, seed=seed, log=_quiet)
    npz2, json2 = infer.write(res2, d2)
    assert sorted(os.listdir(d1)) == sorted(os.listdir(d2)) == [infer.JSON_NAME, infer.NPZ_NAME]
    m1, m2 = _npz_members(npz1), _npz_members(npz2)
    assert list(m1) == list(m2) and all(m1[k] == m2[k] for k in m1), \
        "two runs with one seed differ in %s" % [k for k in m1 if m1.get(k) != m2.get(k)]
    with open(json1, "rb") as f1, open(json2, "rb") as f2:
        t1, t2 = f1.read(), f2.read()
    assert t1 == t2, "the two JSON files differ"
    t1.decode("ascii")
    assert t1.decode("ascii") == json.dumps(js, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    with np.load(npz1, allow_pickle=False) as z:
        assert sorted(z.files) == sorted(A), (sorted(z.files), sorted(A))
        for k in A:
            assert _bits_equal(z[k], A[k]), "the file's %s is not the result's" % k
    same_bytes = open(npz1, "rb").read() == open(npz2, "rb").read()
    # the same result as T45a's run without ids
    for k in infer.ARRAYS:
        assert _bits_equal(A[k], res["arrays"][k]), k
    # another seed: other draws, the same embeddings (they do not depend on the seed)
    res3 = infer.run(b, rec["x"], seed=seed + 1000, log=_quiet)
    assert not np.array_equal(res3["arrays"]["draws"], A["draws"]), "another seed, same draws"
    assert _bits_equal(res3["arrays"]["z"], A["z"]), "another seed changed z"
    for k in _GATE + ("gate_flag",):
        assert _bits_equal(res3["arrays"][k], A[k]), "another seed changed %s" % k
    # another batch (n exceeds it): other draws (D-179), the same z
    nb = man["sample"]["batch_rows"] // 2
    assert n > nb
    res4 = infer.run(b, rec["x"], seed=seed, batch=nb, log=_quiet)
    assert res4["json"]["batch_rows"] == nb
    assert not np.array_equal(res4["arrays"]["draws"], A["draws"]), "another batch, same draws"
    assert _max_abs(res4["arrays"]["z"], A["z"]) <= 1e-5, "another batch changed z"
    # the flag is p < alpha_gate, strictly (sec. 2.6): p takes the values (1 + c) / (1 +
    # N_null), so p = alpha occurs (alpha 0.05, N_null 199: p = 10 / 200). The gate is
    # replaced for one call by the real one with p set at, just below and just above alpha
    # on windows 0, 1, 2 (one ulp each side), and 1 on the others
    import unittest.mock as mock
    import ev_gate
    alpha = float(man["gate"]["alpha"])
    real_gate = ev_gate.gate
    tie_p = np.array([alpha, np.nextafter(alpha, -np.inf), np.nextafter(alpha, np.inf)])

    def tie_gate(*a, **k):
        out = dict(real_gate(*a, **k))
        pp = np.ones_like(out["p"])
        pp[:3] = tie_p
        out["p"] = pp
        return out

    nt = 5
    with mock.patch.object(ev_gate, "gate", tie_gate):
        rt = infer.run(b, rec["x"][:nt], seed=seed, log=_quiet)
    assert ev_gate.gate is real_gate
    At, jt = rt["arrays"], rt["json"]
    assert _bits_equal(At["gate_p"][:3], tie_p), (At["gate_p"][:3], tie_p)
    assert At["gate_flag"].tolist() == [False, True, False] + [False] * (nt - 3), \
        "at the tie p = alpha = %r the flags are %s, expected [False, True, False, ...] " \
        "(p < alpha strictly)" % (alpha, At["gate_flag"].tolist())
    assert jt["gate"]["n_flagged"] == 1 and jt["gate"]["share_flagged"] == 1.0 / nt, jt["gate"]
    # n_post given: other per-member counts
    np2 = 2 * dims["M"]
    res5 = infer.run(b, rec["x"][:3], n_post=np2, seed=seed, log=_quiet)
    assert res5["arrays"]["draws"].shape == (3, np2, dims["p"])
    assert res5["json"]["n_post"] == np2 and res5["json"]["n"] == 3
    print("[T45c] %d arrays + rows + %d ids, %d JSON keys; two runs: identical members and JSON "
          "(whole npz files %s); seed + 1000 and batch %d: other draws, z the same; at p = alpha "
          "not flagged, one ulp below flagged, one above not"
          % (len(infer.ARRAYS), len(ids), len(js),
             "identical" if same_bytes else "differ only in the zip's time stamps", nb))


# ------------------------------------------------------------ T45d

GW, BASE, PITCH, FS, NSAMP = 3, 1, 60.0, 1000.0, 6000        # T_rec = 6 s
NE, NSUB, WSIZE, GWIN = 4, 2, 0.01, 0.03


def _ptrain_folder(folder, rng):
    """A ptrain folder of the 3 x 3 grid (ids 1..9), raster format, every
    electrode its own count of spikes (no tie in the ranking by rate)."""
    import scipy.io as sio
    os.makedirs(folder, exist_ok=True)
    counts = 40 + 7 * rng.permutation(GW * GW)
    for i, k in enumerate(counts):
        t = np.sort(rng.choice(NSAMP, size=int(k), replace=False))
        r = np.zeros((NSAMP, 1), dtype=np.uint8)
        r[t, 0] = 1
        sio.savemat(os.path.join(folder, "ptrain_%d.mat" % (BASE + i)), {"ptrain": r},
                    format="5", do_compression=True)
    return folder


def _extraction():
    from sbitool import schema as S
    e = {"electrodes_per_subset": NE, "n_subsets": NSUB, "mfr_threshold_hz": 0.0,
         "w_size_s": WSIZE, "gaussian_window_s": GWIN, "grid_width": GW, "index_base": BASE,
         "pitch_um": PITCH, "fs_raw_hz": FS, "detection": None}
    assert not S.validate_extraction(e), S.validate_extraction(e)
    return e


def _bundle_with_extraction(ext):
    """A bundle of the fixture's first ensemble exported by block 41's command
    with this extraction and a card that agrees with it."""
    from sbitool import export
    T41 = _t41()
    c = _ctx()
    name = T41.ENSEMBLES[0]
    e, prov = T41._ensemble(name)
    d = _tmpdir("i45d_bundle")
    ext_path = os.path.join(d, "extraction.json")
    with open(ext_path, "w") as fh:
        json.dump(ext, fh)
    card = os.path.join(d, "CARD.md")
    with open(card, "w") as fh:
        fh.write(T41._card_text(e.arm, len(e.records), prov["axes"]["names"], extraction=ext))
    out = os.path.join(d, "ext.sbibundle")
    code, text = T41._main(export.main, T41._export_argv(name, out, card=card,
                                                          extraction=ext_path, ctx=c))
    assert code == 0, text[-3000:]
    return out


def test_t45d_end_to_end():
    """T45d: a ptrain folder written by the test, through `windows` and `infer` by the command
    line with a bundle whose extraction fits it: exit 0, n windows as T44c counts them; the
    same through `infer --recording`."""
    _need_extractor()
    from sbitool import bundle, windows
    from sbitool.recordings import ptrain
    rng = np.random.default_rng(SEED + 45)
    folder = _ptrain_folder(os.path.join(_tmpdir("i45d"), "rec"), rng)
    ext = _extraction()
    bpath = _bundle_with_extraction(ext)
    man = bundle.read(bpath).manifest
    assert man["extraction"] == ext
    Wn = int(man["W"])
    # n as T44c counts it: floor(K / W) windows per subregion, K the trace's bins
    rec = ptrain.read(folder, fs_raw_hz=FS, grid_width=GW, index_base=BASE, pitch_um=PITCH)
    subs = windows.subregions(rec, ext)
    x, _fs = windows.traces(rec, subs, ext)
    n_exp = len(subs) * (x.shape[1] // Wn)
    assert len(subs) == NSUB and n_exp >= NSUB, (len(subs), x.shape, Wn)
    opts = ["--option", "fs_raw_hz=%r" % FS, "--option", "grid_width=%d" % GW,
            "--option", "index_base=%d" % BASE, "--option", "pitch_um=%r" % PITCH]
    wdir = _tmpdir("i45d_out")
    wnpz = os.path.join(wdir, "windows.npz")
    code, out, err = _cli(["windows", "--bundle", bpath, "--recording", folder, "--adapter",
                           "ptrain", "--out", wnpz] + opts)
    assert code == 0, (code, out[-2000:], err[-2000:])
    assert _last(out) == "WINDOWS OK: %d window(s) of %d subregion(s)" % (n_exp, NSUB), out
    o1 = os.path.join(wdir, "from_windows")
    code, out, err = _cli(["infer", "--bundle", bpath, "--windows", wnpz, "--out", o1])
    assert code == 0, (code, out[-2000:], err[-2000:])
    with np.load(os.path.join(o1, "posterior.npz")) as z:
        k = int(z["gate_flag"].sum())
        A1 = dict((key, z[key]) for key in z.files)
    assert _last(out) == "INFER OK: %d window(s), %d flagged by the gate" % (n_exp, k), out
    assert A1["draws"].shape == (n_exp, man["sample"]["n_post"], man["p"])
    with np.load(wnpz) as z:
        for key in ("subregion", "window_idx", "t_start_s"):
            assert _bits_equal(A1[key], z[key]), "id %s not the windows file's" % key
    o2 = os.path.join(wdir, "from_recording")
    code, out, err = _cli(["infer", "--bundle", bpath, "--recording", folder, "--adapter",
                           "ptrain", "--out", o2] + opts)
    assert code == 0, (code, out[-2000:], err[-2000:])
    assert _last(out) == "INFER OK: %d window(s), %d flagged by the gate" % (n_exp, k), out
    with np.load(os.path.join(o2, "posterior.npz")) as z:
        assert sorted(z.files) == sorted(A1)
        for key in A1:
            assert _bits_equal(z[key], A1[key]), "--recording and --windows differ in %s" % key
    with open(os.path.join(o2, "posterior.json")) as fh:
        js = json.load(fh)
    assert js["ids"] == ["subregion", "window_idx", "t_start_s"] and js["n"] == n_exp
    print("[T45d] %d subregion(s), K %d, W %d: %d window(s), %d flagged; windows -> infer and "
          "infer --recording give the same files' arrays" % (NSUB, x.shape[1], Wn, n_exp, k))


# ------------------------------------------------------------ T45e

def _refused(fn, words):
    from sbitool import infer
    try:
        fn()
    except infer.InferRefused as exc:
        msg = str(exc)
        for w in words:
            assert w in msg, "refusal %r does not name %r" % (msg, w)
        return msg
    raise AssertionError("not refused (expected words %s)" % (words,))


def _cli_refused(argv, out_dir, code_want, words=()):
    from sbitool import infer
    code, out, err = _main(infer.main, argv)
    assert code == code_want, (argv, code, out[-1500:], err[-1500:])
    assert not os.path.exists(out_dir), "%s written on a refusal" % out_dir
    if code_want == infer.EXIT_REFUSE:
        lines = [x for x in err.splitlines() if x.startswith("REFUSED:")]
        assert len(lines) == 1, err
        for w in words:
            assert w in lines[0], (w, lines[0])
    assert "INFER OK" not in out
    return err


def test_t45e_refusals():
    """T45e: n_post not a multiple of M, windows of another length than W, a bundle that fails
    its check, --option detection: refused, nothing written."""
    from sbitool import infer
    T41 = _t41()
    name = T41.ENSEMBLES[0]
    b, e, prov, r, rec = _bank_case(name)
    man = b.manifest
    M, W = man["M"], man["W"]
    x = rec["x"][:4]
    # run: refused before any network is built
    _refused(lambda: infer.run(b, x, n_post=M + 1, log=_quiet), ["n_post", "multiple"])
    _refused(lambda: infer.run(b, x, n_post=0, log=_quiet), ["n_post"])
    _refused(lambda: infer.run(b, x[:, :W - 1], log=_quiet), ["W = %d" % W])
    _refused(lambda: infer.run(b, np.zeros((4, W + 1), np.float32), log=_quiet), ["W"])
    _refused(lambda: infer.run(b, x.astype(np.float64), log=_quiet), ["float32"])
    _refused(lambda: infer.run(b, x[:0], log=_quiet), ["n = 0"])
    bad = x.copy()
    bad[2, 5] = np.nan
    _refused(lambda: infer.run(b, bad, log=_quiet), ["non-finite", "row 2"])
    _refused(lambda: infer.run(b, x, ids={"well": np.zeros(3)}, log=_quiet), ["well"])
    _refused(lambda: infer.run(b, x, ids={"draws": np.zeros(4)}, log=_quiet), ["draws"])
    _refused(lambda: infer.run(b, x, ids={"rows": np.zeros(4)}, log=_quiet), ["rows"])
    _refused(lambda: infer.run(b, x, batch=0, log=_quiet), ["batch"])
    # the command: exit 3 with one REFUSED: line, nothing written
    d = _tmpdir("i45e")
    wnpz = os.path.join(d, "windows.npz")
    np.savez(wnpz, windows=x, subregion=np.zeros(4, np.int64))
    short = os.path.join(d, "short.npz")
    np.savez(short, windows=x[:, :W - 1])
    bpath = _ctx()["bundles"][name][0]
    broken = T41._rezip(bpath, os.path.join(d, "broken.sbibundle"),
                        edit=T41._flip_last(_member_file(man)))
    out = os.path.join(d, "out")
    _cli_refused(["--bundle", bpath, "--windows", wnpz, "--out", out, "--n-post", str(M + 1)],
                 out, infer.EXIT_REFUSE, ["n_post"])
    _cli_refused(["--bundle", bpath, "--windows", short, "--out", out], out, infer.EXIT_REFUSE,
                 ["W = %d" % W])
    _cli_refused(["--bundle", broken, "--windows", wnpz, "--out", out], out, infer.EXIT_REFUSE,
                 ["sha256"])
    _cli_refused(["--bundle", os.path.join(d, "absent.sbibundle"), "--windows", wnpz, "--out",
                  out], out, infer.EXIT_REFUSE, ["no such file"])
    _cli_refused(["--bundle", bpath, "--windows", os.path.join(d, "absent.npz"), "--out", out],
                 out, infer.EXIT_REFUSE, ["absent.npz"])
    # a bundle without extraction takes windows only
    _cli_refused(["--bundle", bpath, "--recording", d, "--adapter", "ptrain", "--out", out],
                 out, infer.EXIT_REFUSE, ["no extraction"])
    # the arguments: exit 2
    err = _cli_refused(["--bundle", bpath, "--recording", d, "--adapter", "ptrain", "--option",
                        "detection={}", "--out", out], out, infer.EXIT_ARGS)
    assert "detection" in err, err
    _cli_refused(["--bundle", bpath, "--windows", wnpz, "--recording", d, "--out", out], out,
                 infer.EXIT_ARGS)
    _cli_refused(["--bundle", bpath, "--out", out], out, infer.EXIT_ARGS)
    _cli_refused(["--bundle", bpath, "--recording", d, "--out", out], out, infer.EXIT_ARGS)
    _cli_refused(["--bundle", bpath, "--windows", wnpz, "--out", out, "--n-post", "x"], out,
                 infer.EXIT_ARGS)
    # the same command with a valid n_post is not refused (the refusals above are the n_post's)
    code, so, se = _main(infer.main, ["--bundle", bpath, "--windows", wnpz, "--out", out,
                                      "--n-post", str(2 * M)])
    assert code == 0 and _last(so).startswith("INFER OK: 4 window(s), "), (code, so, se)
    print("[T45e] 11 refusals of run, 6 of the command (exit 3, one REFUSED: line, nothing "
          "written), 5 of the arguments (exit 2)")


def _member_file(man):
    """The name of member 0's state file in the bundle."""
    return man["members"][0]["file"]


CHECKS = [("test_t45a_evaluation", test_t45a_evaluation),
          ("test_t45b_mixture", test_t45b_mixture),
          ("test_t45c_contract", test_t45c_contract),
          ("test_t45d_end_to_end", test_t45d_end_to_end),
          ("test_t45e_refusals", test_t45e_refusals)]


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
