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
             "eval_config", "config", "config_sha256", "seconds", "ncpus", "torch_threads",
             # added to block 35's provenance list on 2026-11-23 (D-167)
             "n_out_of_box"}


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


# ---------------------------------------------------------------- 2026-11-23_1906: rows outside the open box
# Oracles (sec. 2.6 "The rows outside the open box", block 35 Rule, R35f):
# a row with a coordinate <= 0 or >= 1 keeps log q = -inf (mixture and
# members), in_box false, is left out of the oracle and counted in
# `n_out_of_box`; it gets draws like any row; nothing that does not hold
# theta changes; with every row outside the box the sampling refuses (exit
# 3) and writes nothing.

def _sim_rows():
    """The recorded project's D_sim as runA sampled it: bank rows, draw rows."""
    v, ec = _ensure("runA", "synth_rec.json", "rec_eval.json")
    e = v.ensembles[0]
    z = _load(os.path.join(OUT, "runA", "samples", "%s_sim.npz" % e.stem))
    return v, z


def _box_tree(name, edits):
    """A boundary copy (rec_fixture.boundary_copy) under OUT/<name>, with its
    own evaluation configuration (out_root OUT/<name>/out). Returns (cfg,
    ecfg, out_root)."""
    sys.path.insert(0, os.path.join(REPO, "hpc", "pipeline", "tests"))
    os.environ["REC_FIXTURE"] = FIX
    import rec_fixture
    root = os.path.join(OUT, name)
    cfgp = os.path.join(root, "synth_rec_box.json")
    if not os.path.isfile(cfgp):
        os.makedirs(root, exist_ok=True)
        cfgp = rec_fixture.boundary_copy(root, edits)
    d = json.load(open(os.path.join(FIX, "rec_eval.json")))
    d["out_root"] = os.path.join(root, "out")
    ep = os.path.join(root, "rec_eval_box.json")
    json.dump(d, open(ep, "w"), indent=1)
    return cfgp, ep, d["out_root"]


def _sample_cli(cfg, ecfg, name):
    return subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_sample.py"), "--config",
                           cfg, "--eval-config", ecfg, "--ensemble", name],
                          capture_output=True, text=True)


def box_run():
    """The two-row edit of R35f: one draw row with a coordinate set to 1.0,
    one non-draw row with another set to 0.0; every ensemble sampled."""
    v, z = _sim_rows()
    r_draw = int(z["rows"][z["draw_idx"][1]])
    r_other = int(z["rows"][np.flatnonzero(~z["draw"])[2]])
    p = z["theta_ref"].shape[1]
    edits = [(r_draw, 2 % p, 1.0), (r_other, 5 % p, 0.0)]
    cfg, ecfg, out = _box_tree("box2", edits)
    for e in v.ensembles:
        if not os.path.isfile(os.path.join(out, "samples", "%s_sample.json" % e.stem)):
            r = _sample_cli(cfg, ecfg, e.name)
            assert r.returncode == 0 and "REC SAMPLE OK" in r.stdout, r.stdout + r.stderr
    return v, cfg, ecfg, out, (r_draw, r_other)


def test_out_of_box_rows_sampled_not_refused():
    v, cfg, ecfg, out, (r_draw, r_other) = box_run()
    for e in v.ensembles:
        a = _load(os.path.join(OUT, "runA", "samples", "%s_sim.npz" % e.stem))
        b = _load(os.path.join(out, "samples", "%s_sim.npz" % e.stem))
        prov = json.load(open(os.path.join(out, "samples", "%s_sample.json" % e.stem)))
        assert prov["n_out_of_box"] == 2
        assert set(prov) == PROV_KEYS
        bad = np.isin(b["rows"], [r_draw, r_other])
        assert bad.sum() == 2
        assert_array_equal(b["in_box"], a["in_box"] & ~bad)
        assert np.all(np.isneginf(b["logq_ref"][bad]))
        assert np.all(np.isneginf(b["logq_ref_members"][bad]))
        # every array that does not hold theta: bit for bit (draws, embeddings,
        # statistics and densities at the draws do not read theta)
        for k in ("rows", "donor", "well", "cls", "calib", "draw", "gate_role", "draw_idx",
                  "draws", "draw_member", "logq_draws", "logq_draws_own", "stats", "z_idx",
                  "z"):
            assert_array_equal(b[k], a[k], err_msg=k)
        # the other rows' densities at theta_i: the unmodified run's. Tolerance
        # 1e-4 nats (R35b's: float32 roundoff when a batch's composition
        # changes); exact equality is reported separately below
        ok = ~bad
        d = np.max(np.abs(b["logq_ref_members"][ok] - a["logq_ref_members"][ok]))
        assert d <= 1e-4, d
        assert np.max(np.abs(b["logq_ref"][ok] - a["logq_ref"][ok])) <= 1e-4
        # the oracle over the rows in the box, recomputed from each member's record
        for k, info in enumerate(prov["members"]):
            nll = np.load(info["record"][:-len(".json")] + "_perrow.npz")["nll"].astype(float)
            dif = np.abs(nll[ok] + b["logq_ref_members"][ok, k])
            assert abs(prov["oracle_max_abs_nats"][k] - dif.max()) < 1e-12
            assert abs(prov["oracle_mean_abs_nats"][k] - dif.mean()) < 1e-12
            assert dif.max() <= prov["oracle_tol"]


def test_every_row_out_of_box_refused_nothing_written():
    v, z = _sim_rows()
    edits = [(int(r), 0, 1.0) for r in z["rows"]]
    cfg, ecfg, out = _box_tree("boxall", edits)
    r = _sample_cli(cfg, ecfg, v.ensembles[0].name)
    assert r.returncode == 3, r.stdout + r.stderr
    assert not os.path.isdir(os.path.join(out, "samples")) or \
        os.listdir(os.path.join(out, "samples")) == []


def test_slightly_negative_and_one_row_left_in_box():
    """All rows but one outside (a coordinate -1e-7 or exactly 1): not a
    refusal (one row in the box)."""
    v, z = _sim_rows()
    keep = int(z["rows"][np.flatnonzero(~z["draw"])[0]])
    edits = [(int(r), 1, (-1e-7 if i % 2 else 1.0)) for i, r in enumerate(z["rows"])
             if int(r) != keep]
    cfg, ecfg, out = _box_tree("boxone", edits)
    e = v.ensembles[0]
    if not os.path.isfile(os.path.join(out, "samples", "%s_sample.json" % e.stem)):
        r = _sample_cli(cfg, ecfg, e.name)
        assert r.returncode == 0, r.stdout + r.stderr
    prov = json.load(open(os.path.join(out, "samples", "%s_sample.json" % e.stem)))
    b = _load(os.path.join(out, "samples", "%s_sim.npz" % e.stem))
    assert prov["n_out_of_box"] == len(z["rows"]) - 1
    assert b["in_box"].sum() == 1 and b["rows"][b["in_box"]][0] == keep


def test_nan_theta_row_is_a_refusal_exit_3():
    """A NaN coordinate is neither <= 0 nor >= 1 (sec. 2.6), so the spec gives
    it no rule as a row; the bank reader rejects it (latent_bank's contract).
    Block 35's CLI contract: exit 3 on a refusal, which writes nothing; any
    other outcome is reported."""
    v, z = _sim_rows()
    r_other = int(z["rows"][np.flatnonzero(~z["draw"])[3]])
    cfg, ecfg, out = _box_tree("boxnan", [(r_other, 0, float("nan"))])
    e = v.ensembles[0]
    r = _sample_cli(cfg, ecfg, e.name)
    assert not os.path.isdir(os.path.join(out, "samples")) or \
        os.listdir(os.path.join(out, "samples")) == []
    assert r.returncode == 3, (r.returncode, r.stderr[-400:])


# ---------------------------------------------------------------- 2026-11-27_1832: D-168 (sec. 2.6)
# Oracle O35.7: "A theta with a NaN or an infinite coordinate is not such a
# row: the bank breaks its contract and the sampling refuses (exit 3), naming
# it." Any non-finite label of the simulated bank (a report row or not, NaN,
# +inf or -inf) gives exit 3, a one-line REFUSE (no traceback) and no file.

def _nonfinite_case(tag, row, axis, value):
    v, z = _sim_rows()
    cfg, ecfg, out = _box_tree(tag, [(row, axis, value)])
    r = _sample_cli(cfg, ecfg, v.ensembles[0].name)
    sdir = os.path.join(out, "samples")
    files = os.listdir(sdir) if os.path.isdir(sdir) else []
    return r, files


@pytest.mark.parametrize("tag,value", [("boxpinf", float("inf")), ("boxninf", float("-inf"))])
def test_inf_theta_is_a_refusal_exit_3(tag, value):
    v, z = _sim_rows()
    row = int(z["rows"][z["draw_idx"][0]])           # a draw (and calibration) row
    r, files = _nonfinite_case(tag, row, 1, value)
    assert files == [], files
    assert r.returncode == 3, (r.returncode, r.stdout[-400:], r.stderr[-400:])
    assert "Traceback" not in r.stderr, r.stderr[-400:]
    last = [l for l in r.stdout.splitlines() if l.startswith("REFUSE")]
    assert last and "contract" in last[-1] and "theta" in last[-1], r.stdout[-400:]


def test_nan_theta_outside_d_sim_is_a_refusal_exit_3():
    """A NaN on a bank row that is not a report row (a training row): the bank
    as a whole breaks its contract, so the sampling refuses all the same."""
    v, z = _sim_rows()
    import glob as _g
    n_bank = 0
    for p in sorted(_g.glob(os.path.join(v.sim_glob))):
        with np.load(p) as s:
            n_bank += int(s["theta"].shape[0])
    other = sorted(set(range(n_bank)) - set(int(x) for x in z["rows"]))
    assert other, "every bank row is a report row"
    r, files = _nonfinite_case("boxnantrain", other[0], 0, float("nan"))
    assert files == [], files
    assert r.returncode == 3, (r.returncode, r.stdout[-400:], r.stderr[-400:])
    assert "Traceback" not in r.stderr


def test_nan_refusal_names_the_offending_row_or_shard():
    """'refuses (exit 3), naming it': the refusal line should let the user find
    the broken label -- the shard, the bank row or at least the bank's path.
    Reported, not a contract of block 35's own entry (see the report's S1)."""
    v, z = _sim_rows()
    r_other = int(z["rows"][np.flatnonzero(~z["draw"])[3]])
    cfg, ecfg, out = _box_tree("boxnan", [(r_other, 0, float("nan"))])
    r = _sample_cli(cfg, ecfg, v.ensembles[0].name)
    line = [l for l in r.stdout.splitlines() if l.startswith("REFUSE")][-1]
    assert ("shard_" in line) or (str(r_other) in line) or ("_sim" in line), line


# ---------------------------------------------------------------- 2026-11-28 (39e6f66)
# O35.9 [block 36's Rule, amended 2026-11-27]: "a bank with a non-finite label is refused
# naming the bank's glob and the first such label's shard, row and axis". "First": the
# shards in sorted order (the bank's order), then the row, then the axis. The expected
# shard, row of the shard and axis are found here by reading the copied bank's shards with
# numpy.

def _first_nonfinite(sim_glob):
    for path in sorted(glob.glob(sim_glob)):
        with np.load(path) as zz:
            th = np.asarray(zz["theta"], dtype=np.float64)
        bad = np.argwhere(~np.isfinite(th))
        if bad.size:
            return os.path.basename(path), int(bad[0][0]), int(bad[0][1])
    return None


def test_nan_refusal_names_glob_and_the_first_label():
    v, z = _sim_rows()
    rows = z["rows"]
    # two bad labels: a later bank row on axis 0, an earlier one on axis 3 and 1 (the
    # first of the earlier row's axes is 1)
    r_late, r_early = int(rows[-2]), int(rows[1])
    cfg, ecfg, out = _box_tree("boxnan2", [(r_late, 0, float("nan")),
                                           (r_early, 3, float("inf")),
                                           (r_early, 1, float("-inf"))])
    r = _sample_cli(cfg, ecfg, v.ensembles[0].name)
    msg = r.stdout + r.stderr
    assert r.returncode == 3 and "Traceback" not in msg, msg[-1500:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("REFUSE")][-1]
    c = json.load(open(cfg))
    sim_glob = None
    for dp, dn, fn in os.walk(os.path.dirname(cfg)):
        if any(f.startswith("shard_") and f.endswith(".npz") for f in fn) and \
                dp.endswith("_sim"):
            sim_glob = os.path.join(dp, "shard_*.npz")
    assert sim_glob is not None
    shard, row, axis = _first_nonfinite(sim_glob)
    assert os.path.dirname(sim_glob) in line, line           # the bank's glob
    assert ("shard %s" % shard) in line and ("row %d " % row) in line and \
        ("axis %d " % axis) in line, (line, shard, row, axis)
    assert axis == 1, axis                                    # the oracle's own sanity
    assert not os.path.isdir(os.path.join(out, "samples")) or \
        os.listdir(os.path.join(out, "samples")) == []
