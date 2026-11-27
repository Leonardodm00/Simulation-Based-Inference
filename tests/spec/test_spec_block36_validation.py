"""Spec tests of D-162's validation (sec. 2.6 "The validation"; block 36's
`ev_rec_validate.py`). Integration tester, 2026-11-23, incremental run on
dc096e5.

Needs REC_FIXTURE and the as-recorded run REC_SPEC_OUT/asA (samples and
report; produced by the CLIs when absent). (1) The in-sandbox target checked
directly with numpy on the two trees' files: the NLL per row within 1e-9, the
coverage, its band, gamma, the SBC ranks, TARP's curves and the recovery
within 1e-12. (2) The validator on perturbed copies of the trees.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL = os.path.join(REPO, "hpc", "pipeline", "evaluation")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

FIX = os.environ.get("REC_FIXTURE", "").strip()
OUT = os.environ.get("REC_SPEC_OUT", "/tmp/tst/e2e")
pytestmark = pytest.mark.skipif(not (FIX and os.path.isdir(FIX)), reason="REC_FIXTURE not set")
VALIDATE = os.path.join(EVAL, "ev_rec_validate.py")


def _as_tree():
    from test_spec_block35_sample import _ensure, _ecfg
    v, ec = _ensure("asA", "synth.json", "rec_eval_as.json")
    if not os.path.isfile(os.path.join(OUT, "asA", "REC_REPORT_COMPLETE.json")):
        r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_report.py"), "--config",
                            os.path.join(FIX, "synth.json"), "--eval-config",
                            _ecfg("asA", "rec_eval_as.json"), "--ncpus", "1"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    return v, ec, os.path.join(OUT, "asA"), ec.validate_against


def _load(path):
    with np.load(path) as z:
        return dict((k, z[k]) for k in z.files)


def test_in_sandbox_target_direct():
    v, ec, new, old = _as_tree()
    for e in v.ensembles:
        a = _load(os.path.join(new, "metrics", "%s_sim.npz" % e.stem))
        o = _load(os.path.join(old, "metrics", "%s_sim.npz" % e.stem))
        np.testing.assert_array_equal(a["rows"], o["rows"])
        # 1e-9 nats on the NLL (D-162), 1e-12 on the rest (the same functions on the same arrays)
        for k, tol in (("nll_rows", 1e-9), ("gamma", 1e-12), ("coverage", 1e-12),
                       ("coverage_lo", 1e-12), ("coverage_hi", 1e-12),
                       ("coverage_members", 1e-12), ("sbc_ranks", 0), ("sbc_ecdf_diff", 1e-12),
                       ("tarp_x_f", 1e-12), ("tarp_random_f", 1e-12), ("mhat", 1e-12),
                       ("sigma", 1e-12), ("q05", 1e-12), ("q95", 1e-12), ("zeta", 1e-12)):
            assert k in a and k in o, k
            x, y = a[k].astype(np.float64), o[k].astype(np.float64)
            assert x.shape == y.shape, k
            np.testing.assert_array_equal(np.isnan(x), np.isnan(y), err_msg=k)
            m = ~np.isnan(x)
            assert np.max(np.abs(x[m] - y[m]), initial=0) <= tol, k
        ja = json.load(open(os.path.join(new, "metrics", "%s_sim.json" % e.stem)))
        jo = json.load(open(os.path.join(old, "metrics", "%s_sim.json" % e.stem)))
        assert ja["sbc"]["ks_p"] == jo["sbc"]["ks_p"]
        assert ja["sbc"]["holm_rejected"] == jo["sbc"]["holm_rejected"]
        for mode in ("x", "random"):
            assert ja["tarp"][mode]["ks_p"] == jo["tarp"][mode]["ks_p"]
    # the paired file on D_sim equals block 27's
    pa = json.load(open(os.path.join(new, "metrics", "paired_S-A2-R0_sim.json")))
    import glob
    po_path = glob.glob(os.path.join(old, "metrics", "paired_S-A2-R0_sim*.json"))
    assert po_path, os.listdir(os.path.join(old, "metrics"))
    po = json.load(open(po_path[0]))
    for k in ("Dbar", "ci", "se", "sigma_seed", "n_rows", "n_donors", "resolved"):
        assert np.allclose(pa[k], po[k], rtol=0, atol=1e-12), (k, pa[k], po[k])


def _run(old, new, *extra):
    r = subprocess.run([sys.executable, VALIDATE, "--old", old, "--new", new] + list(extra),
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def _copy(src, dst):
    shutil.copytree(os.path.join(src, "metrics"), os.path.join(dst, "metrics"))


def _edit_npz(path, key, fn):
    z = _load(path)
    z[key] = fn(z[key].copy())
    np.savez(path, **z)


def test_validator_tolerances(tmp_path):
    v, ec, new, old = _as_tree()
    rc, out = _run(old, new)
    assert rc == 0 and "REC VALIDATION OK: 3 ensemble(s)" in out, out
    stem = v.ensembles[0].stem
    name = v.ensembles[0].name
    # an NLL moved by 0.0015 (inside 0.002) passes; by 0.003 fails, naming ensemble and row
    for delta, ok in ((0.0015, True), (0.003, False)):
        d = tmp_path / ("n%g" % delta); _copy(new, d)
        _edit_npz(str(d / "metrics" / ("%s_sim.npz" % stem)), "nll_rows",
                  lambda a: (a.__setitem__(3, a[3] + delta), a)[1])
        rc, out = _run(old, str(d))
        assert (rc == 0) == ok, out
        if not ok:
            assert rc == 3 and name in out and "row 3" in out
    # the coverage moved by 0.009 (inside 0.01) passes; by 0.02 fails
    for delta, ok in ((0.009, True), (0.02, False)):
        d = tmp_path / ("c%g" % delta); _copy(new, d)
        _edit_npz(str(d / "metrics" / ("%s_sim.npz" % stem)), "coverage",
                  lambda a: (a.__setitem__(50, a[50] + delta), a)[1])
        rc, out = _run(old, str(d))
        assert (rc == 0) == ok, out
    # an SBC verdict flipped fails; a NaN NLL on one side fails
    d = tmp_path / "sbc"; _copy(new, d)
    jp = d / "metrics" / ("%s_sim.json" % stem)
    js = json.load(open(jp))
    js["sbc"]["verdict"][0] = "FLIPPED"
    json.dump(js, open(jp, "w"))
    rc, out = _run(old, str(d))
    assert rc == 3 and "sbc verdict axis 0" in out, out
    d = tmp_path / "nan"; _copy(new, d)
    _edit_npz(str(d / "metrics" / ("%s_sim.npz" % stem)), "nll_rows",
              lambda a: (a.__setitem__(0, np.nan), a)[1])
    rc, out = _run(old, str(d))
    assert rc == 3, out
    # an ensemble missing from one tree fails; the record is written either way
    d = tmp_path / "miss"; _copy(new, d)
    ip = d / "metrics" / "METRICS.json"
    idx = json.load(open(ip)); idx["ensembles"].pop(name); json.dump(idx, open(ip, "w"))
    rec = tmp_path / "rec.json"
    rc, out = _run(old, str(d), "--record", str(rec))
    assert rc == 3 and name in out
    r = json.load(open(rec))
    assert r["ok"] is False and r["schema"] == "rec_eval_validation/1" and r["failures"]
    assert set(r) >= {"schema", "old", "new", "ok", "n_ensembles", "failures", "nll_tol",
                      "coverage_tol", "metrics_index_sha256", "finished"}
    import hashlib
    assert r["metrics_index_sha256"] == hashlib.sha256(open(ip, "rb").read()).hexdigest()
