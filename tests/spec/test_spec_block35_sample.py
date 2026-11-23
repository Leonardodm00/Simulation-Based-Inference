"""Spec tests of block 35, the sampler on a recorded project, through its
command line (`ev_rec_sample.py`). Integration tester, 2026-11-23,
incremental run on dc096e5.

Needs REC_FIXTURE (a built rec_fixture tree). The samples are written under
REC_SPEC_OUT (default /tmp/tst/e2e): runA and runB, two runs of the recorded
project with the same configuration, and asA, the as-recorded run; each is
produced by the CLI when absent. Expected values come from block 35's
contract, from block 34's rule written by hand, from the members' own
`_perrow.npz` records and from block 19's sample files, all read with numpy.
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL = os.path.join(REPO, "hpc", "pipeline", "evaluation")
sys.path.insert(0, HERE)
if EVAL not in sys.path:
    sys.path.insert(0, EVAL)
sys.dont_write_bytecode = True

FIX = os.environ.get("REC_FIXTURE", "").strip()
OUT = os.environ.get("REC_SPEC_OUT", "/tmp/tst/e2e")
pytestmark = pytest.mark.skipif(not (FIX and os.path.isdir(FIX)), reason="REC_FIXTURE not set")

SIM_KEYS = {"rows": ((1,), "i8"), "donor": ((1,), "i8"), "well": ((1,), "i8"),
            "cls": ((1,), "i8"), "theta_ref": ((2,), "f8"), "in_box": ((1,), "b1"),
            "logq_ref_members": ((2,), "f8"), "logq_ref": ((1,), "f8"), "calib": ((1,), "b1"),
            "draw": ((1,), "b1"), "gate_role": ((1,), "i1"), "draw_idx": ((1,), "i8"),
            "draws": ((3,), "f4"), "draw_member": ((1,), "i8"), "logq_draws": ((2,), "f4"),
            "logq_draws_own": ((2,), "f4"), "stats": ((2,), "f4"), "z_idx": ((1,), "i8"),
            "z": ((3,), "f4")}
REC_KEYS = {"rows": "i8", "cls": "i8", "donor": "i8", "well": "i8", "batch": "i8",
            "subregion": "i8", "window_idx": "i8", "rep_group": "i8", "contaminated": "b1",
            "draws": "f4", "draw_member": "i8", "logq_draws": "f4", "logq_draws_own": "f4",
            "z": "f4", "stats": "f4"}
PROV_KEYS = {"schema", "ensemble", "stem", "search", "campaign", "arm", "folder", "view_kind",
             "members", "banks", "bank_globs", "sim_bank", "best_fit_complete_sha256",
             "sample_params", "seeds", "oracle_max_abs_nats", "oracle_mean_abs_nats",
             "oracle_tol", "gate", "n", "n_draw", "n_z", "M", "n_post", "p", "axes", "files",
             "eval_config", "config", "config_sha256", "seconds", "ncpus", "torch_threads"}


def _ecfg(run, base):
    p = os.path.join(OUT, "%s_%s" % (run, base))
    if not os.path.isfile(p):
        d = json.load(open(os.path.join(FIX, base)))
        d["out_root"] = os.path.join(OUT, run)
        os.makedirs(OUT, exist_ok=True)
        json.dump(d, open(p, "w"), indent=1)
    return p


def _view(cfg, ecfg):
    import _evalpath  # noqa: F401
    import ev_rec_config as RC
    import ev_rec_plan as RP
    return RP.load_view(cfg, RC.load(ecfg)), RC.load(ecfg)


def _ensure(run, cfg_name, ecfg_name):
    cfg = os.path.join(FIX, cfg_name)
    ecfg = _ecfg(run, ecfg_name)
    v, ec = _view(cfg, ecfg)
    for e in v.ensembles:
        if not os.path.isfile(os.path.join(OUT, run, "samples", "%s_sample.json" % e.stem)):
            r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_sample.py"), "--config",
                                cfg, "--eval-config", ecfg, "--ensemble", e.name],
                               capture_output=True, text=True)
            assert r.returncode == 0 and "REC SAMPLE OK" in r.stdout, r.stdout + r.stderr
    return v, ec


def _load(path):
    with np.load(path) as z:
        return dict((k, z[k]) for k in z.files)


def test_contract_and_identities_recorded():
    v, ec = _ensure("runA", "synth_rec.json", "rec_eval.json")
    import ev_plan
    sys.path.insert(0, HERE)
    from test_spec_block34_gate import roles_oracle
    for e in v.ensembles:
        sd = os.path.join(OUT, "runA", "samples")
        z = _load(os.path.join(sd, "%s_sim.npz" % e.stem))
        prov = json.load(open(os.path.join(sd, "%s_sample.json" % e.stem)))
        assert set(prov) == PROV_KEYS, set(prov) ^ PROV_KEYS
        assert prov["schema"] == "rec_eval_sample/1"
        n, M, n_post, p = len(z["rows"]), prov["M"], prov["n_post"], prov["p"]
        nd, nz = len(z["draw_idx"]), len(z["z_idx"])
        assert set(z) == set(SIM_KEYS), set(z) ^ set(SIM_KEYS)
        for k, (nd_, dt) in SIM_KEYS.items():
            assert z[k].ndim == nd_[0] and z[k].dtype == np.dtype(dt), (k, z[k].shape, z[k].dtype)
        assert z["theta_ref"].shape == (n, p) and z["logq_ref_members"].shape == (n, M)
        assert z["draws"].shape == (nd, n_post, p) and z["logq_draws"].shape == (nd, n_post)
        assert z["stats"].shape == (nd, 8) and z["z"].shape[:2] == (M, nz)
        # n_post / M draws per member, in member order
        assert_array_equal(z["draw_member"], np.repeat(np.arange(M), n_post // M))
        # k_draw = 1: draw = calib = the first row of each network (by hand)
        seen, first = set(), []
        for d in z["donor"]:
            first.append(d not in seen); seen.add(d)
        assert_array_equal(z["calib"], first)
        assert_array_equal(z["draw"], z["calib"])
        assert_array_equal(z["draw_idx"], np.flatnonzero(z["draw"]))
        assert_array_equal(z["z_idx"], np.flatnonzero(z["draw"] | (z["gate_role"] != 0)))
        # the roles: block 34's rule by hand, seed seed_of(seed, "gate")
        g = ec.gate
        exp, _ = roles_oracle(z["donor"], g["donor_fracs"], g["n_ref"], g["n_rank"],
                              g["n_null"], g["rows_per_donor"], ev_plan.seed_of(ec.seed, "gate"))
        assert_array_equal(z["gate_role"], exp)
        # (E1): the mixture at theta_i from the members, -inf outside the box
        ib = z["in_box"]
        assert_array_equal(ib, np.all((z["theta_ref"] > 0) & (z["theta_ref"] < 1), axis=1))
        mix = logsumexp(z["logq_ref_members"], axis=1) - np.log(M)
        # 1e-12: one logsumexp of M = 2 float64 terms, two summation orders
        assert np.max(np.abs(z["logq_ref"][ib] - mix[ib])) < 1e-12
        assert np.all(np.isneginf(z["logq_ref"][~ib]))
        # the oracle against each member's own record: -log q_m(theta_i | x_i) = perrow nll
        for m, rec in enumerate(e.records):
            pr = glob.glob(os.path.join(os.path.dirname(rec), "*_perrow.npz"))
            assert len(pr) == 1
            nll = np.load(pr[0])["nll"]
            assert nll.shape == (n,)
            assert np.max(np.abs(-z["logq_ref_members"][:, m] - nll)) <= ec.oracle_tol
        # draws inside the unit box, finite
        assert np.all(np.isfinite(z["draws"])) and np.all((z["draws"] >= 0) & (z["draws"] <= 1))
        # the recording bank's file
        zr = _load(os.path.join(sd, "%s_real.npz" % e.stem))
        assert set(zr) == set(REC_KEYS), set(zr) ^ set(REC_KEYS)
        nr = len(zr["rows"])
        for k, dt in REC_KEYS.items():
            assert zr[k].dtype == np.dtype(dt), k
        assert zr["draws"].shape == (nr, n_post, p) and zr["z"].shape[:2] == (M, nr)
        assert zr["z"].shape[2] == z["z"].shape[2] and zr["stats"].shape == (nr, 8)
        bank = [np.load(f) for f in sorted(glob.glob(os.path.join(
            FIX, "banks_root", "synth_real", "shard_*.npz")))]
        for k in ("cls", "donor", "well", "batch", "subregion", "window_idx"):
            assert_array_equal(zr[k], np.concatenate([b[k] for b in bank]), err_msg=k)
        assert_array_equal(zr["rows"], np.arange(nr))
        assert np.all(zr["rep_group"] == -1)


def test_same_seed_identical_files():
    _ensure("runA", "synth_rec.json", "rec_eval.json")
    v, _ = _ensure("runB", "synth_rec.json", "rec_eval.json")
    for e in v.ensembles:
        for suf in ("sim", "real"):
            a = _load(os.path.join(OUT, "runA", "samples", "%s_%s.npz" % (e.stem, suf)))
            b = _load(os.path.join(OUT, "runB", "samples", "%s_%s.npz" % (e.stem, suf)))
            assert set(a) == set(b)
            for k in a:
                assert_array_equal(a[k], b[k], err_msg="%s %s %s" % (e.stem, suf, k))


def test_as_recorded_equals_block19():
    """D-162's first target: the as-recorded samples equal block 19's
    (`<old_out>/samples/<stem>_sim.npz` and `<stem>_<cohort>.npz`)."""
    v, ec = _ensure("asA", "synth.json", "rec_eval_as.json")
    old = os.path.join(ec.validate_against, "samples")
    for e in v.ensembles:
        a = _load(os.path.join(OUT, "asA", "samples", "%s_sim.npz" % e.stem))
        o = _load(os.path.join(old, "%s_sim.npz" % e.stem))
        assert_array_equal(a["draws"], o["draws"])                    # bit for bit
        for k in ("theta_ref", "in_box", "rows", "donor", "well", "cls", "logq_ref_members",
                  "logq_ref", "draw_member", "logq_draws", "logq_draws_own", "calib",
                  "stats", "z"):
            if k in o:
                # 1e-9: D-162's in-sandbox tolerance for the NLL; equal arrays expected
                if np.issubdtype(o[k].dtype, np.floating):
                    fin = np.isfinite(o[k])
                    assert_array_equal(np.isfinite(a[k]), fin, err_msg=k)
                    assert np.max(np.abs(a[k][fin] - o[k][fin]), initial=0) <= 1e-9, k
                else:
                    assert_array_equal(a[k], o[k], err_msg=k)
        assert_array_equal(a["draw_idx"], np.arange(len(a["rows"])))
        for c in e.banks:
            a = _load(os.path.join(OUT, "asA", "samples", "%s_%s.npz" % (e.stem, c)))
            o = _load(os.path.join(old, "%s_%s.npz" % (e.stem, c)))
            for k in ("draws", "logq_draws", "logq_draws_own", "z", "stats", "cls", "donor",
                      "well"):
                assert_array_equal(a[k], o[k], err_msg="%s %s" % (c, k))
            assert "theta_ref" not in a and "theta" not in a


def test_refusal_writes_nothing(tmp_path):
    d = json.load(open(os.path.join(FIX, "rec_eval.json")))
    d["out_root"] = str(tmp_path / "out")
    d["n_post"] = 41                                       # not a multiple of M = 2
    p = tmp_path / "e.json"; json.dump(d, open(p, "w"))
    r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_sample.py"), "--config",
                        os.path.join(FIX, "synth_rec.json"), "--eval-config", str(p),
                        "--ensemble", "S-A1:A1"], capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    assert not os.path.exists(tmp_path / "out" / "samples") or \
        not os.listdir(tmp_path / "out" / "samples")
    r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_sample.py"), "--config",
                        os.path.join(FIX, "synth_rec.json"), "--eval-config", str(p)],
                       capture_output=True, text=True)
    assert r.returncode == 2, r.stdout + r.stderr          # --ensemble missing: arguments
