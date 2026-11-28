#!/usr/bin/env python3
"""Smoke test of block 48: the model's training in the bundle (specs/SPEC.md block 48, "Test
oracles", checks T48*).

A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t48a_export         T48a: the member files equal the records' history; the search's trials
    test_t48b_round_trip     T48b: Bundle.training(), the check, a changed byte, unequal lists
    test_t48c_no_training    T48c: --no-training gives training null and no member
    test_t48d_schema         T48d: format 1 refused naming both; the earlier suites pass

Run
    SBIX_CLONE=<the Sbi-extractor's checkout> PYTHONDONTWRITEBYTECODE=1 \
        python3 sbitool/tests/test_smoke_tool_training.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<a kept fixture> reuses it.

Pure ASCII, LF only.
"""

import copy
import glob
import json
import math
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

SEED = 20261128  # every random number of this file derives from it
_TMP = []
_T41 = []        # block 41's test module: its evaluated project and bundles (_ctx)
# T48d: every earlier suite of the tool, run with the new format
EARLIER_SUITES = ("schema", "bundle", "card", "recordings", "windows", "infer", "figures",
                  "verify")
SUITE_TIMEOUT_S = 3600


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="sbitool_%s_" % tag)
    _TMP.append(d)
    return d


def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)
    if _T41:
        _T41[0]._cleanup()


def _t41():
    if not _T41:
        from sbitool.tests import test_smoke_tool_bundle as T41
        _T41.append(T41)
    return _T41[0]


# ------------------------------------------------------------------ the test's own readings

def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _num_or_none(v):
    """The test's own reading of a curve's entry: NaN (and an infinity) are null."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _history_lists(rec):
    """The five lists of a record's history, read by the test."""
    hist = rec["history"]
    return dict((key, [_num_or_none(h.get(key)) for h in hist])
                for key in ("epoch", "npe", "val_npe", "dsn", "rep"))


def _tune_folder(c, e):
    """The search's tune folder, from the project's configuration as the test
    reads it: paths.tune_root joined with the search's name (the run's view),
    then its campaign."""
    d = _read_json(c["cfg"])
    q = [x for x in d["searches"] if x["name"] == e.search]
    assert len(q) == 1, (e.search, d["searches"])
    return os.path.join(d["paths"]["tune_root"], q[0]["name"], q[0]["campaign"])


def _zip(path):
    with zipfile.ZipFile(path) as zf:
        return dict((n, zf.read(n)) for n in zf.namelist())


def _pieces(path):
    """(manifest, card text, states, gate sets, sim bytes) of a bundle, to
    write it again by hand with another training."""
    from sbitool import bundle
    b = bundle.read(path)
    states = [b.state(k) for k in range(b.manifest["M"])]
    return (copy.deepcopy(b.manifest), b.card, states, b.gate_sets(),
            _zip(path)["evaluation/sim.json"])


def _check_refuses(path, what, names):
    """bundle-check exits 3 naming each of `names`; read() refuses."""
    T41 = _t41()
    from sbitool import bundle
    return T41._refused_naming(bundle, path, what, names)


# ------------------------------------------------------------------ checks

def test_t48a_export():
    """T48a: the member files equal the records' history; the search's trials (SPEC block 48)."""
    T41 = _t41()
    c = T41._ctx()
    for name in T41.ENSEMBLES:
        e, _prov = T41._ensemble(name)
        path = c["bundles"][name][0]
        zm = _zip(path)
        man = json.loads(zm["manifest.json"].decode("ascii"))
        M = len(e.records)
        assert man["schema"] == "sbitool_bundle/2", man["schema"]
        assert man["training"] == {"members": M, "search": True}, (name, man["training"])
        tr = sorted(n for n in zm if n.startswith("training/"))
        assert tr == sorted(["training/member_%d.json" % k for k in range(M)]
                            + ["training/search.json"]), (name, tr)
        # the M member files: the records' history, read here
        for k, rpath in enumerate(e.records):
            rec = _read_json(rpath)
            got = json.loads(zm["training/member_%d.json" % k].decode("ascii"))
            want = _history_lists(rec)
            assert sorted(got) == sorted(["seed"] + list(want)), sorted(got)
            assert got["seed"] == rec["seed"] == man["members"][k]["seed"], (got["seed"], k)
            E = len(rec["history"])
            assert E >= 1 and all(len(got[key]) == E for key in want), (name, k, E)
            for key in want:
                assert got[key] == want[key], (name, k, key, got[key], want[key])
            assert man["files"]["training/member_%d.json" % k] == T41._sha(
                zm["training/member_%d.json" % k])
            print("[T48a] %s member %d (seed %d): %d epoch(s), npe %s, val_npe %s"
                  % (name, k, got["seed"], E, got["npe"], got["val_npe"]), flush=True)
        # the search's summary: one entry per file of the trials folder
        folder = _tune_folder(c, e)
        tdir = os.path.join(folder, "trials")
        files = sorted(os.listdir(tdir))
        s = json.loads(zm["training/search.json"].decode("ascii"))
        assert sorted(s) == sorted(["campaign", "n_trials", "objective", "trials", "chosen"]), s
        assert s["n_trials"] == len(files) == len(s["trials"]) >= 1, (s["n_trials"], files)
        assert s["campaign"] == e.campaign and s["objective"] == "nll", s
        ids = [t["trial_id"] for t in s["trials"]]
        assert ids == sorted(ids), ids
        by_id = {}
        for f in files:
            d = _read_json(os.path.join(tdir, f))
            by_id[d["trial_id"]] = d
        assert sorted(by_id) == ids, (sorted(by_id), ids)
        for t in s["trials"]:
            d = by_id[t["trial_id"]]
            assert sorted(t) == sorted(["trial_id", "status", "nll", "config"]), t
            assert t["status"] == d["status"] and t["nll"] == d["nll"], (t, d["status"], d["nll"])
            want_cfg = dict((k, v) for k, v in d["config"].items() if not k.startswith("_")
                            and (v is None or isinstance(v, (bool, int, float, str))))
            assert t["config"] == want_cfg, (t["config"], want_cfg)
        assert s["chosen"] in ids, (s["chosen"], ids)
        fin = _read_json(os.path.join(folder, "finalists.json"))["finalists"][0]
        match = [i for i in ids if by_id[i]["config"] == fin["config"]]
        assert match == [s["chosen"]], (match, s["chosen"])
        # in the fixture the first finalist's own trial_id names that trial
        assert s["chosen"] == fin["trial_id"], (s["chosen"], fin["trial_id"])
        assert man["files"]["training/search.json"] == T41._sha(zm["training/search.json"])
        print("[T48a] %s: search %s, %d trial(s) (files %d), chosen %s, nll %s"
              % (name, s["campaign"], s["n_trials"], len(files), s["chosen"],
                 [t["nll"] for t in s["trials"]]), flush=True)
    # collect itself: the search's folder absent is a None and one log line, not a refusal
    from sbitool import export, training_info as TI
    P = export._pipeline()
    e, _prov = T41._ensemble(T41.ENSEMBLES[0])
    recs = [_read_json(r) for r in e.records]
    lines = []
    members, search = TI.collect(P, c["view"], e._replace(search="no-such-search"), recs,
                                 log=lines.append)
    assert search is None and len(lines) == 1 and "no search summary" in lines[0], lines
    assert [m["seed"] for m in members] == [r["seed"] for r in recs]
    work = _tmpdir("t48a")
    gone = os.path.join(work, "cfg_no_tune.json")
    d = _read_json(c["cfg"])
    d["paths"]["tune_root"] = os.path.join(work, "absent")
    with open(gone, "w") as fh:
        json.dump(d, fh)
    import config as C
    view2 = c["view"]._replace(cfg=C.load_config(gone, synthetic_ok=True))
    lines = []
    _m, search = TI.collect(P, view2, e, recs, log=lines.append)
    assert search is None and len(lines) == 1 and "trials" in lines[0], lines
    # a record without history is refused (it has no curve to pack)
    try:
        TI.collect(P, c["view"], e, [dict(recs[0], history=[])] + recs[1:], log=lines.append)
        raise AssertionError("a record without history was taken")
    except TI.TrainingInfoError as exc:
        assert "history" in str(exc), str(exc)


def test_t48b_round_trip():
    """T48b: Bundle.training(), the check, a changed byte, unequal lists (SPEC block 48)."""
    T41 = _t41()
    from sbitool import bundle
    c = T41._ctx()
    name = T41.ENSEMBLES[0]
    e, _prov = T41._ensemble(name)
    good = c["bundles"][name][0]
    zm = _zip(good)
    man = json.loads(zm["manifest.json"].decode("ascii"))
    M = man["M"]
    b = bundle.read(good)
    t = b.training()
    assert t == {"members": [json.loads(zm["training/member_%d.json" % k]) for k in range(M)],
                 "search": json.loads(zm["training/search.json"])}, t
    for k, rpath in enumerate(e.records):
        lists = _history_lists(_read_json(rpath))
        assert all(t["members"][k][key] == lists[key] for key in lists), k
    code, text = T41._main(bundle.main, [good])
    assert code == 0 and "BUNDLE CHECK OK: " in text, (code, text)
    work = _tmpdir("t48b")
    # written by hand with a training given: Bundle.training() returns what was written
    # (a NaN written null), and the curves of two lengths are refused at writing
    man0, card, states, gates, sim = _pieces(good)
    members = [dict(m) for m in t["members"]]
    members[0]["npe"] = [float("nan")] + list(members[0]["npe"][1:])
    search = copy.deepcopy(t["search"])
    hand = os.path.join(work, "hand.sbibundle")
    final = bundle.write(hand, man0, card, states, gates, sim, training=(members, search))
    assert final["training"] == {"members": M, "search": True}, final["training"]
    assert bundle.check(hand) == [], bundle.check(hand)
    t2 = bundle.read(hand).training()
    want0 = dict(members[0], npe=[None] + list(members[0]["npe"][1:]))
    assert t2["members"] == [want0] + members[1:] and t2["search"] == search, t2
    # without the search's summary: search false, no search.json
    nos = os.path.join(work, "nosearch.sbibundle")
    final = bundle.write(nos, man0, card, states, gates, sim, training=(t["members"], None))
    assert final["training"] == {"members": M, "search": False}
    assert "training/search.json" not in _zip(nos) and bundle.check(nos) == []
    assert bundle.read(nos).training() == {"members": t["members"], "search": None}
    # lists of two lengths: refused at writing, nothing written
    for key in ("dsn", "epoch"):
        bad = [dict(m) for m in t["members"]]
        bad[M - 1][key] = list(bad[M - 1][key]) + [0.0]
        out = os.path.join(work, "two_lengths.sbibundle")
        try:
            bundle.write(out, man0, card, states, gates, sim, training=(bad, search))
            raise AssertionError("curves of two lengths (%s) were written" % key)
        except bundle.BundleError as exc:
            assert "not of one length" in str(exc) and "member_%d" % (M - 1) in str(exc), str(exc)
        assert not os.path.exists(out) and not os.path.exists(out + ".part"), os.listdir(work)
    # a seed that is not the manifest's member's, M - 1 curves: refused at writing
    for bad, says in (([dict(t["members"][0], seed=t["members"][0]["seed"] + 100)]
                       + t["members"][1:], "seed"),
                      (t["members"][:-1], "member curve")):
        try:
            bundle.write(os.path.join(work, "x.sbibundle"), man0, card, states, gates, sim,
                         training=(bad, search))
            raise AssertionError("training %s was written" % says)
        except bundle.BundleError as exc:
            assert says in str(exc), str(exc)
    # one byte changed in training/member_0.json (and in search.json): exit 3 naming it
    for target in ("training/member_0.json", "training/search.json"):
        bad = T41._rezip(good, os.path.join(work, "bad.sbibundle"), edit=T41._flip_last(target))
        _check_refuses(bad, "a byte of %s" % target, [target])
    # a member file of another seed, its digest made to match: refused naming the seed
    m0 = json.loads(zm["training/member_0.json"])
    m0["seed"] += 100
    T41._rebundle(good, os.path.join(work, "seed.sbibundle"),
                  members={"training/member_0.json": (json.dumps(m0) + "\n").encode("ascii")})
    _check_refuses(os.path.join(work, "seed.sbibundle"), "a member's seed",
                   ["training/member_0.json", "seed"])
    # a training/ member not announced: on a bundle whose training is null, on one
    # without search, and one member beyond M
    cases = [(nos, "training/search.json"), (good, "training/member_%d.json" % M),
             (good, "training/notes.txt")]
    nul = os.path.join(work, "null.sbibundle")
    bundle.write(nul, man0, card, states, gates, sim, training=None)
    assert bundle.check(nul) == [] and bundle.read(nul).training() is None
    cases.append((nul, "training/member_0.json"))
    for src, extra in cases:
        bad = T41._rezip(src, os.path.join(work, "extra.sbibundle"),
                         add={extra: zm["training/member_0.json"]})
        text = _check_refuses(bad, "%s not announced" % extra, [extra, "does not announce"])
    # announced but absent: missing member
    miss = os.path.join(work, "miss.sbibundle")
    with zipfile.ZipFile(good) as zin, zipfile.ZipFile(miss, "w") as zout:
        for n in zin.namelist():
            if n != "training/member_1.json":
                zout.writestr(n, zin.read(n))
    _check_refuses(miss, "an announced member absent", ["missing member training/member_1.json"])


def test_t48c_no_training():
    """T48c: --no-training gives training null and no member (SPEC block 48)."""
    T41 = _t41()
    from sbitool import bundle, export
    c = T41._ctx()
    name = T41.ENSEMBLES[1]
    card = c["bundles"][name][1]
    out = os.path.join(_tmpdir("t48c"), "nt.sbibundle")
    code, text = T41._main(export.main, T41._export_argv(name, out, card=card)
                           + ["--no-training"])
    assert code == 0 and "BUNDLE EXPORT OK: " in text, text[-2000:]
    zm = _zip(out)
    man = json.loads(zm["manifest.json"].decode("ascii"))
    assert man["training"] is None, man["training"]
    assert not [n for n in zm if n.startswith("training/")], sorted(zm)
    assert not [n for n in man["files"] if n.startswith("training/")], man["files"]
    assert bundle.check(out) == [], bundle.check(out)
    code, text = T41._main(bundle.main, [out])
    assert code == 0 and "BUNDLE CHECK OK: " in text, text
    assert bundle.read(out).training() is None
    # everything else is the default export's: the same members, card, sim.json, gate
    zd = _zip(c["bundles"][name][0])
    for n in zm:
        if n != "manifest.json":
            assert zm[n] == zd[n], n


def test_t48d_schema():
    """T48d: format 1 refused naming both; the earlier suites pass (SPEC block 48)."""
    T41 = _t41()
    from sbitool import schema as S
    assert S.BUNDLE_SCHEMA == "sbitool_bundle/2"
    from sbitool.tests import test_smoke_tool_schema as T40
    m = T40._valid_manifest()
    assert S.validate_manifest(m) == [], S.validate_manifest(m)
    m1 = dict(m, schema="sbitool_bundle/1")
    probs = S.validate_manifest(m1)
    assert len(probs) == 1 and "sbitool_bundle/1" in probs[0] and "sbitool_bundle/2" in probs[0], \
        probs
    # a manifest of the first format as it was (no training key): refused too, naming both
    m1b = dict((k, v) for k, v in m1.items() if k != "training")
    probs = S.validate_manifest(m1b)
    assert any("sbitool_bundle/1" in p and "sbitool_bundle/2" in p for p in probs), probs
    # the training key's own faults
    for bad, says in (({"members": 2, "search": True}, "training.members"),
                      ({"members": 3, "search": 1}, "training.search"),
                      ({"members": 3}, "training.search"), ([3, True], "training")):
        probs = S.validate_manifest(dict(m, training=bad))
        assert len(probs) == 1 and says in probs[0], (bad, probs)
    # a bundle whose manifest says "sbitool_bundle/1": bundle-check exits 3 naming both
    c = T41._ctx()
    good = c["bundles"][T41.ENSEMBLES[0]][0]
    old = os.path.join(_tmpdir("t48d"), "old.sbibundle")

    def to_1(man):
        man["schema"] = "sbitool_bundle/1"
    with zipfile.ZipFile(good) as zf:
        man = json.loads(zf.read("manifest.json"))
    to_1(man)
    T41._rezip(good, old, edit=lambda n, d: T41._manifest_text(man) if n == "manifest.json" else d)
    _check_refuses(old, "a first-format bundle", ["sbitool_bundle/1", "sbitool_bundle/2"])
    # every earlier suite of the tool passes with the new format (each in its own process,
    # on the fixture this process built or was given)
    import rec_fixture
    root = rec_fixture.fixture()["root"]
    env = dict(os.environ, REC_FIXTURE=root, PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    env.pop("REC_KEEP", None)
    failed = []
    for suite in EARLIER_SUITES:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-W", "ignore",
                            os.path.join(_HERE, "test_smoke_tool_%s.py" % suite)],
                           cwd=_REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=SUITE_TIMEOUT_S)
        table = [l for l in r.stdout.splitlines() if l.startswith("| ")
                 or " checks pass in " in l]
        print("[T48d] %s: exit %d in %.0f s\n  %s" % (suite, r.returncode, time.time() - t0,
                                                     "\n  ".join(table)), flush=True)
        if r.returncode != 0:
            failed.append((suite, r.stdout[-3000:]))
    assert not failed, "\n\n".join("%s:\n%s" % f for f in failed)


CHECKS = [("test_t48a_export", test_t48a_export),
          ("test_t48b_round_trip", test_t48b_round_trip),
          ("test_t48c_no_training", test_t48c_no_training),
          ("test_t48d_schema", test_t48d_schema)]


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
