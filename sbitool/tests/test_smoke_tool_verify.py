#!/usr/bin/env python3
"""Smoke test of block 47: a bundle verified against the evaluation it was
exported from (specs/SPEC.md block 47, "Test oracles", checks T47*).

A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t47a_identical      both ensembles of the fixture verify; the record's keys
    test_t47b_max_rows       the first batches alone are identical; less than a batch refused
    test_t47c_difference     one changed element is found and named; the gate's tolerance
    test_t47d_refusals       another ensemble's bundle, another n_post, an unknown bank, no file

Run
    SBIX_CLONE=<the Sbi-extractor's checkout> PYTHONDONTWRITEBYTECODE=1 \
        python3 sbitool/tests/test_smoke_tool_verify.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<a kept fixture> reuses it.

Pure ASCII, LF only.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import unittest
import zipfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SEED = 20261128  # every random number of this file derives from it
_TMP = []
_REC = {}        # the verification of each ensemble, made once


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="sbitool_%s_" % tag)
    _TMP.append(d)
    return d


def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)
    try:
        from sbitool.tests import test_smoke_tool_bundle as T41
        T41._cleanup()
    except Exception:           # the helper's folders: nothing to report
        pass


def _t41():
    """Block 41's test: its `_ctx()` is the fixture's recorded-kind project,
    sampled (block 35) and evaluated (block 36), with one bundle per
    ensemble. Built once per process."""
    from sbitool.tests import test_smoke_tool_bundle as T41
    return T41


def _run_cli(argv):
    """(exit code, stdout, stderr) of verify.main in this process."""
    from sbitool import verify
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = verify.main(argv)
    return code, out.getvalue(), err.getvalue()


def _argv(name, bundle_path=None, extra=()):
    c = _t41()._ctx()
    return ["--bundle", bundle_path or c["bundles"][name][0], "--config", c["cfg"],
            "--eval-config", c["ecfg"], "--ensemble", name] + list(extra)


def _record(name):
    from sbitool import bundle, verify
    if name not in _REC:
        c = _t41()._ctx()
        _REC[name] = verify.run(bundle.read(c["bundles"][name][0]), c["cfg"], c["ecfg"], name,
                                log=lambda t: None)
    return _REC[name]


# ------------------------------------------------------------ T47a

def test_t47a_identical():
    """T47a: both ensembles of the fixture verify: identical, exit 0, the pass line, VERIFY.json
    with exactly RECORD_KEYS and every array's max_abs_diff 0.0 (the gate's at most 1e-12)."""
    from sbitool import verify
    T41 = _t41()
    c = T41._ctx()
    lines = []
    for name in T41.ENSEMBLES:
        rec = _record(name)
        assert tuple(sorted(rec)) == tuple(sorted(verify.RECORD_KEYS)), sorted(rec)
        assert rec["schema"] == verify.VERIFY_SCHEMA and rec["identical"] is True, rec
        assert rec["ensemble"] == name and rec["max_rows"] is None
        assert rec["batch_rows"] == c["ecfg_obj"].batch_rows
        assert len(rec["banks"]) == 1, sorted(rec["banks"])
        for r, bk in rec["banks"].items():
            assert tuple(sorted(bk)) == tuple(sorted(verify.BANK_KEYS)), sorted(bk)
            assert bk["identical"] is True and bk["n_compared"] == bk["n"] > rec["batch_rows"], bk
            assert tuple(bk["arrays"]) == verify.COMPARED, tuple(bk["arrays"])
            for k, a in bk["arrays"].items():
                assert tuple(sorted(a)) == tuple(sorted(verify.ARRAY_KEYS)), (k, sorted(a))
                assert a["equal"] is True, (k, a)
                assert a["max_abs_diff"] <= (verify.GATE_TOL if k in verify.GATE_ARRAYS else 0.0), \
                    (k, a)
            # the shapes are the evaluation's own, read here from block 35's file
            with np.load(os.path.join(c["out_root"], "samples",
                                      "%s_%s.npz" % (rec["stem"], r))) as z:
                assert bk["arrays"]["draws"]["shape"] == list(z["draws"].shape)
                assert bk["n"] == z["draws"].shape[0] and rec["n_post"] == z["draws"].shape[1]
            lines.append("%s/%s: %d windows x %d draws, %d arrays identical"
                         % (name, r, bk["n"], rec["n_post"], len(bk["arrays"])))
    # the command, with --out
    name = T41.ENSEMBLES[0]
    out = os.path.join(_tmpdir("v47a"), "verify")
    code, text, err = _run_cli(_argv(name, extra=["--out", out]))
    assert code == 0, (code, text[-1500:], err[-1500:])
    last = text.strip().splitlines()[-1]
    assert last.startswith("VERIFY OK: 1 bank(s), %d window(s): the bundle reproduces the "
                           "evaluation" % _record(name)["banks"][sorted(_record(name)["banks"])[0]]["n"]), last
    with open(os.path.join(out, verify.JSON_NAME), "rb") as fh:
        raw = fh.read()
    raw.decode("ascii")
    disk = json.loads(raw)
    assert tuple(sorted(disk)) == tuple(sorted(verify.RECORD_KEYS)) and disk["identical"] is True
    assert sorted(os.listdir(out)) == [verify.JSON_NAME], os.listdir(out)
    # torch's number of threads is restored
    import torch
    before = torch.get_num_threads()
    other = 1 if int(_record(name)["torch_threads"]) != 1 else 2
    torch.set_num_threads(other)      # another number than the evaluation's
    calls = []
    real_set = torch.set_num_threads

    def recording_set(k):
        calls.append(int(k))
        return real_set(k)
    torch.set_num_threads = recording_set
    try:
        code, text, err = _run_cli(_argv(name))
        assert code == 0, (code, text[-800:], err[-800:])
        assert torch.get_num_threads() == other, (other, torch.get_num_threads())
        # set to the evaluation's number for the inference, then back
        assert calls == [int(_record(name)["torch_threads"]), other], calls
    finally:
        torch.set_num_threads = real_set
        torch.set_num_threads(before)
    print("[T47a] " + "; ".join(lines))


# ------------------------------------------------------------ T47b

def test_t47b_max_rows():
    """T47b: --max-rows N with N = batch_rows and N = batch_rows + 4: the first batch_rows windows
    are compared and are identical (n_compared = batch_rows < n); N < batch_rows is refused."""
    from sbitool import bundle, verify
    T41 = _t41()
    c = T41._ctx()
    name = T41.ENSEMBLES[1]
    B = int(c["ecfg_obj"].batch_rows)
    b = bundle.read(c["bundles"][name][0])
    for N in (B, B + 4):
        rec = verify.run(b, c["cfg"], c["ecfg"], name, max_rows=N, log=lambda t: None)
        assert rec["identical"] is True and rec["max_rows"] == N, rec
        for r, bk in rec["banks"].items():
            assert bk["n_compared"] == B < bk["n"], bk
            assert bk["arrays"]["draws"]["shape"][0] == B
            assert bk["arrays"]["z"]["shape"][1] == B
            assert bk["arrays"]["draw_member"]["shape"] == [rec["n_post"]]
    # the restriction itself, against numpy on block 35's file: the first B rows
    full = _record(name)
    r = sorted(full["banks"])[0]
    with np.load(os.path.join(c["out_root"], "samples", "%s_%s.npz" % (full["stem"], r))) as z:
        assert verify.first_rows("draws", z["draws"], B).shape == (B,) + z["draws"].shape[1:]
        assert np.array_equal(verify.first_rows("z", z["z"], B), z["z"][:, :B])
        assert verify.first_rows("draw_member", z["draw_member"], B).shape == z["draw_member"].shape
    # a max-rows of more than the bank compares the whole bank
    rec = verify.run(b, c["cfg"], c["ecfg"], name, max_rows=10 ** 6, log=lambda t: None)
    assert rec["identical"] and rec["banks"][r]["n_compared"] == rec["banks"][r]["n"]
    # less than one batch: refused
    for N in (B - 1, 1):
        try:
            verify.run(b, c["cfg"], c["ecfg"], name, max_rows=N, log=lambda t: None)
            raise AssertionError("max_rows %d accepted" % N)
        except verify.VerifyRefused as exc:
            assert "less than one batch" in str(exc), exc
    code, text, err = _run_cli(_argv(name, extra=["--max-rows", str(B - 1)]))
    assert code == verify.EXIT_REFUSE and err.startswith("REFUSED: ") and err.count("\n") == 1, \
        (code, err)
    code, text, err = _run_cli(_argv(name, extra=["--max-rows", "0"]))
    assert code == verify.EXIT_ARGS, (code, err)
    print("[T47b] batch %d: max_rows %d and %d compare the first %d of %d windows, identical; "
          "max_rows %d refused" % (B, B, B + 4, B, full["banks"][r]["n"], B - 1))


# ------------------------------------------------------------ T47c

def _case():
    """(inference arrays, evaluation arrays, ecfg, stem, bank) of one ensemble."""
    from sbitool import bundle, infer, verify
    T41 = _t41()
    c = T41._ctx()
    import ev_rec_data      # on sys.path once the context is built
    name = T41.ENSEMBLES[0]
    e, prov = T41._ensemble(name)
    r = sorted(prov["banks"])[0]
    ev = verify._load_evaluation(c["ecfg_obj"], e.stem, r)
    x = ev_rec_data.recording(c["view"], r)["x"]
    res = infer.run(bundle.read(c["bundles"][name][0]), x, seed=int(prov["seeds"][r]),
                    log=lambda *a, **k: None)
    return name, res["arrays"], ev, r


def test_t47c_difference():
    """T47c: one draw of the evaluation changed by one float32 step: `draws` not equal with a
    max_abs_diff of that step, every other array equal; one gate p-value changed by 1e-9: gate_p
    not equal; by 1e-13: equal; a flag flipped; another shape or dtype; the command exits 1 and
    names the bank and the array."""
    from sbitool import verify
    name, arrays, ev, r = _case()
    base = verify.compare(arrays, ev)
    assert all(a["equal"] for a in base.values()), base

    def changed(key, fn):
        ev2 = dict(ev)
        a = np.array(ev[key], copy=True)
        fn(a)
        ev2[key] = a
        return verify.compare(arrays, ev2)

    # one float32 step on one draw
    v = np.float32(ev["draws"][3, 5, 1])
    step = float(np.abs(np.float64(np.nextafter(v, np.float32(np.inf))) - np.float64(v)))

    def bump(a):
        a[3, 5, 1] = np.nextafter(a[3, 5, 1], np.float32(np.inf))
    res = changed("draws", bump)
    assert res["draws"]["equal"] is False and res["draws"]["max_abs_diff"] == step, \
        (res["draws"], step)
    assert all(a["equal"] for k, a in res.items() if k != "draws"), res
    # the gate's tolerance: 1e-9 is found, 1e-13 is not
    def shift(d):
        def f(a):
            a[2] = a[2] + d
        return f
    res = changed("gate_p", shift(1e-9))
    assert res["gate_p"]["equal"] is False and abs(res["gate_p"]["max_abs_diff"] - 1e-9) < 1e-15
    res = changed("gate_p", shift(1e-13))
    assert res["gate_p"]["equal"] is True and 0.0 < res["gate_p"]["max_abs_diff"] <= 1e-12
    # a flag flipped
    def flip(a):
        a[0] = not a[0]
    res = changed("gate_flag", flip)
    assert res["gate_flag"]["equal"] is False and res["gate_flag"]["max_abs_diff"] == 1.0
    # a NaN against a number; the same NaN on both sides
    def nan(a):
        a[1, 0] = np.nan
    res = changed("mhat", nan)
    assert res["mhat"]["equal"] is False and res["mhat"]["max_abs_diff"] == float("inf")
    arr2 = dict(arrays)
    m2 = np.array(arrays["mhat"], copy=True)
    m2[1, 0] = np.nan
    arr2["mhat"] = m2
    ev2 = dict(ev)
    ev2["mhat"] = m2.copy()
    assert verify.compare(arr2, ev2)["mhat"]["equal"] is True
    # another dtype (the same values) and another shape are not equal
    ev2 = dict(ev)
    ev2["sigma"] = ev["sigma"].astype(np.float64)
    res = verify.compare(arrays, ev2)
    assert res["sigma"]["equal"] is False and res["sigma"]["max_abs_diff"] == 0.0
    ev2 = dict(ev)
    ev2["logq_draws"] = ev["logq_draws"][:-1]
    res = verify.compare(arrays, ev2)
    assert res["logq_draws"]["equal"] is False and res["logq_draws"]["max_abs_diff"] == float("inf")
    # draw_member of another order
    def roll(a):
        a[:] = np.roll(a, 1)
    assert changed("draw_member", roll)["draw_member"]["equal"] is False

    # the command: the evaluation's file read with one draw changed -> exit 1, named
    real = verify._load_evaluation

    def altered(ecfg, stem, bank):
        out = real(ecfg, stem, bank)
        a = np.array(out["draws"], copy=True)
        bump(a)
        out["draws"] = a
        return out
    out_dir = os.path.join(_tmpdir("v47c"), "verify")
    verify._load_evaluation = altered
    try:
        code, text, err = _run_cli(_argv(name, extra=["--out", out_dir]))
    finally:
        verify._load_evaluation = real
    assert code == verify.EXIT_FAILED, (code, text[-1500:], err[-1500:])
    assert "bank %s: draws differs (largest absolute difference %.3g" % (r, step) in text, text
    assert text.strip().splitlines()[-1].startswith("VERIFY FAILED: 1 array(s) differ over 1 "
                                                    "bank(s)"), text
    with open(os.path.join(out_dir, verify.JSON_NAME)) as fh:
        disk = json.load(fh)
    assert disk["identical"] is False and disk["banks"][r]["arrays"]["draws"]["equal"] is False
    assert disk["banks"][r]["arrays"]["draws"]["max_abs_diff"] == step
    # a non-finite difference is written as the string "inf" (valid JSON)
    rec = dict(_record(name))
    rec["banks"] = {r: dict(rec["banks"][r], arrays=dict(
        rec["banks"][r]["arrays"], mhat=dict(rec["banks"][r]["arrays"]["mhat"],
                                             equal=False, max_abs_diff=float("inf"))))}
    p = verify.write(rec, _tmpdir("v47c_inf"))
    with open(p) as fh:
        txt = fh.read()
    assert "Infinity" not in txt and json.loads(txt)["banks"][r]["arrays"]["mhat"][
        "max_abs_diff"] == "inf"
    print("[T47c] one float32 step (%.3g) on one draw: draws not equal, the other %d arrays "
          "equal; gate_p + 1e-9 found, + 1e-13 within tolerance; the command exits 1 and names "
          "bank %s, draws" % (step, len(verify.COMPARED) - 1, r))


# ------------------------------------------------------------ T47d

def test_t47d_refusals():
    """T47d: the bundle of the other ensemble; a bundle whose manifest says another n_post; an
    unknown --bank; an absent sample file: exit 3, one REFUSED: line, no VERIFY.json."""
    from sbitool import bundle, verify
    T41 = _t41()
    c = T41._ctx()
    a, b2 = T41.ENSEMBLES
    work = _tmpdir("v47d")
    cases = []
    # the other ensemble's bundle
    cases.append(("another ensemble's bundle", _argv(a, bundle_path=c["bundles"][b2][0]),
                  "was not exported from this evaluation"))
    # a bundle whose manifest says another n_post (a valid bundle: the manifest rewritten)
    src = c["bundles"][a][0]
    edited = os.path.join(work, "n_post.sbibundle")
    with zipfile.ZipFile(src) as zf:
        entries = [(i.filename, zf.read(i.filename)) for i in zf.infolist()]
    man = json.loads(dict(entries)["manifest.json"])
    man["sample"]["n_post"] = int(man["sample"]["n_post"]) * 2
    with zipfile.ZipFile(edited, "w", zipfile.ZIP_STORED) as zf:
        for fn, data in entries:
            zf.writestr(fn, json.dumps(man, sort_keys=True).encode("ascii")
                        if fn == "manifest.json" else data)
    assert bundle.check(edited) == [], bundle.check(edited)
    cases.append(("another n_post", _argv(a, bundle_path=edited), "sample.n_post"))
    # ... and another batch
    edited_b = os.path.join(work, "batch.sbibundle")
    man2 = json.loads(dict(entries)["manifest.json"])
    man2["sample"]["batch_rows"] = int(man2["sample"]["batch_rows"]) + 1
    with zipfile.ZipFile(edited_b, "w", zipfile.ZIP_STORED) as zf:
        for fn, data in entries:
            zf.writestr(fn, json.dumps(man2, sort_keys=True).encode("ascii")
                        if fn == "manifest.json" else data)
    cases.append(("another batch_rows", _argv(a, bundle_path=edited_b), "sample.batch_rows"))
    cases.append(("an unknown bank", _argv(a, extra=["--bank", "no_such_bank"]),
                  "no recording bank 'no_such_bank'"))
    cases.append(("an unknown ensemble", ["--bundle", src, "--config", c["cfg"],
                                          "--eval-config", c["ecfg"], "--ensemble", "nobody"],
                  "nobody"))
    cases.append(("an absent bundle", _argv(a, bundle_path=os.path.join(work, "no.sbibundle")),
                  "no such file"))
    # a bundle that fails its check (one byte of a member changed)
    broken = os.path.join(work, "broken.sbibundle")
    with zipfile.ZipFile(broken, "w", zipfile.ZIP_STORED) as zf:
        for fn, data in entries:
            if fn.endswith("member_0.safetensors"):
                data = data[:-1] + bytes([data[-1] ^ 1])
            zf.writestr(fn, data)
    cases.append(("a broken bundle", _argv(a, bundle_path=broken), "not a valid bundle"))
    n = 0
    for label, argv, needle in cases:
        out = os.path.join(work, "out_%d" % n)
        code, text, err = _run_cli(argv + ["--out", out])
        assert code == verify.EXIT_REFUSE, (label, code, text[-600:], err[-600:])
        assert err.startswith("REFUSED: ") and err.count("\n") == 1, (label, err)
        assert needle in err, (label, needle, err)
        assert not os.path.exists(out), (label, os.listdir(out))
        n += 1
    # an absent evaluation file: the comparison's loader refuses, naming it
    e, prov = T41._ensemble(a)
    r = sorted(prov["banks"])[0]

    class _Ecfg:
        def __init__(self, root):
            self.root = root

        def sub(self, kind):
            return os.path.join(self.root, kind)
    try:
        verify._load_evaluation(_Ecfg(os.path.join(work, "nowhere")), e.stem, r)
        raise AssertionError("an absent file accepted")
    except verify.VerifyRefused as exc:
        assert "is absent" in str(exc) and e.stem in str(exc), exc
    # the arguments
    code, text, err = _run_cli(["--bundle", src])
    assert code == verify.EXIT_ARGS, (code, err)
    # the command is registered
    from sbitool import __main__ as CLI
    assert CLI.COMMANDS["verify"] == "sbitool.verify:main"
    print("[T47d] %d refusals (exit 3, one REFUSED: line, nothing written): %s; an absent "
          "evaluation file refused by name" % (len(cases), ", ".join(c0[0] for c0 in cases)))


CHECKS = [("test_t47a_identical", test_t47a_identical),
          ("test_t47b_max_rows", test_t47b_max_rows),
          ("test_t47c_difference", test_t47c_difference),
          ("test_t47d_refusals", test_t47d_refusals)]


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
