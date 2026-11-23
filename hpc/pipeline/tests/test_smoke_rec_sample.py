#!/usr/bin/env python3
"""Smoke test of block 35: the sampler on a recorded project (specs/SPEC.md block 35, "Test
oracles", checks R35*).

CONTRACT SKELETON (wave B): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 35's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_r35a_as_recorded    D-162's first target: the arrays equal block 19's, the draws bit for bit
    test_r35b_recorded       the recorded-kind project with k_draw = 1: draw rows, z_idx, roles, the bank's file
    test_r35c_contract       every key, shape and dtype; the same seed, identical files
    test_r35d_refusals       five refusals that write nothing
    test_r35e_mixture        the mixture density at the draws, from the members in float64
    test_r35f_out_of_box     rows of D_sim outside the open box: left out of the oracle, counted

Run
    python3 hpc/pipeline/tests/test_smoke_rec_sample.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<dir> reuses a fixture tree built
        before (rec_fixture.py); REC_KEEP=1 keeps the one this run builds.

Pure ASCII, LF only.
"""

import os
import sys
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_EVAL = os.path.join(_PIPE, "evaluation")
for _p in (_HERE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rec_fixture  # noqa: E402

SEED = 20261123  # every random number of this file derives from it

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

PYTHON = sys.executable
SAMPLER = os.path.join(_EVAL, "ev_rec_sample.py")
AS_NAMES = ("S-A1:A1", "S-A2-R0:A1", "S-A2-R0:A2")   # the fixture's complete ensembles
FORBIDDEN_REC = ("theta", "theta_ref", "in_box", "logq_ref", "logq_ref_members")

# every output of this file goes under temporary folders of its own, never
# into the fixture's out_roots (another process may share the tree)
_TMP = []
_RUNS = {}


def _tmpdir(prefix):
    d = tempfile.mkdtemp(prefix="r35_%s_" % prefix)
    _TMP.append(d)
    return d


def _cleanup_tmp():
    while _TMP:
        shutil.rmtree(_TMP.pop(), ignore_errors=True)


def _ecfg_copy(src, d, fname, **over):
    """A copy of the evaluation configuration `src` in folder `d` with
    top-level keys (and `gate` keys under "gate") replaced."""
    with open(src) as fh:
        data = json.load(fh)
    gate = over.pop("gate", None)
    data.update(over)
    if gate:
        data["gate"].update(gate)
    path = os.path.join(d, fname)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    return path


def _sampled(kind, name, ecfg=None, tag="main"):
    """(out, provenance) of block 35's sampler (in this process) on ensemble
    `name` of the fixture's `kind` ("as": the synthetic project read as
    recorded; "rec": the recorded-kind project), cached per arguments."""
    import ev_rec_sample
    F = rec_fixture.fixture()
    cfg = F["cfg"] if kind == "as" else F["cfg_rec"]
    ecfg = ecfg or (F["ecfg_as"] if kind == "as" else F["ecfg_rec"])
    key = (kind, name, ecfg, tag)
    if key not in _RUNS:
        out = _tmpdir("%s_%s" % (kind, tag))
        lines = []
        prov = ev_rec_sample.run(cfg, ecfg, name, out=out, log=lines.append)
        assert any(l.startswith("REC SAMPLE OK: ") for l in lines), "\n".join(lines)
        _RUNS[key] = (out, prov)
    return _RUNS[key]


def _cli(*argv):
    r = subprocess.run([PYTHON, SAMPLER] + list(argv), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)
    return r.returncode, r.stdout


def _files_under(d):
    return sorted(os.path.relpath(os.path.join(a, f), d) for a, _b, fs in os.walk(d) for f in fs) \
        if os.path.isdir(d) else []


def _members(kind, name):
    """(view, index, models) of an ensemble of the fixture, rebuilt."""
    import ev_models
    import ev_rec_config
    import ev_rec_data
    import ev_rec_plan
    F = rec_fixture.fixture()
    cfg = F["cfg"] if kind == "as" else F["cfg_rec"]
    ecfg = ev_rec_config.load(F["ecfg_as"] if kind == "as" else F["ecfg_rec"])
    view = ev_rec_plan.load_view(cfg, ecfg)
    e = ev_rec_plan.find(view, name)
    index, _paths = ev_rec_data.sim_index(view)
    return view, index, [ev_models.rebuild(r, index)[0] for r in e.records]


def _close(a, b, tol):
    """The largest |a - b| over finite entries; inf when the infinite
    entries or the shapes differ."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not (np.array_equal(fa, fb) and np.array_equal(a[~fa], b[~fb])):
        return float("inf")
    return float(np.max(np.abs(a[fa] - b[fb]))) if fa.any() else 0.0


def _check_contract(path, table, dims):
    """Every key of `table` (and no other), with its shape (in the symbols of
    `dims`) and dtype."""
    import numpy as np
    z = np.load(path)
    assert sorted(z.files) == sorted(table), (os.path.basename(path), sorted(z.files))
    for k, (shape, dt) in table.items():
        want = eval(shape, {}, dict(dims))   # noqa: S307 (the contract's own symbols)
        want = want if isinstance(want, tuple) else (want,)
        assert z[k].shape == want and z[k].dtype == np.dtype(dt), (
            os.path.basename(path), k, z[k].shape, want, z[k].dtype, dt)
    return z


# ---------------------------------------------------------------- checks ---

def test_r35a_as_recorded():
    """D-162's first target: the arrays equal block 19's, the draws bit for bit (specs/SPEC.md block 35)."""
    import numpy as np
    F = rec_fixture.fixture()
    old_dir = os.path.join(F["old_out"], "samples")
    # floats within 1e-9: the same functions on the same rows in the same
    # batches give the same floats; 1e-9 is the SPEC's bound for l_i and
    # leaves room only for the order of a sum, not for another computation
    tol = 1e-9
    keys_sim = ("theta_ref", "in_box", "rows", "donor", "well", "cls", "logq_ref_members",
                "logq_ref", "draw_member", "logq_draws", "logq_draws_own", "calib", "stats", "z")
    keys_coh = ("draws", "logq_draws", "logq_draws_own", "z", "stats", "rows", "donor", "well",
                "cls", "draw_member")
    worst = 0.0
    for name in AS_NAMES:
        out, prov = _sampled("as", name)
        stem = prov["stem"]
        with open(os.path.join(old_dir, "%s_sample.json" % stem)) as fh:
            old_prov = json.load(fh)
        assert list(prov["banks"]) == list(old_prov["cohorts"]), (prov["banks"], old_prov["cohorts"])
        for ds in ["sim"] + list(prov["banks"]):
            assert prov["seeds"][ds] == old_prov["seeds"][ds], (name, ds)
        new = np.load(os.path.join(out, "%s_sim.npz" % stem))
        old = np.load(os.path.join(old_dir, "%s_sim.npz" % stem))
        n = old["rows"].shape[0]
        assert new["draws"].dtype == old["draws"].dtype and np.array_equal(
            new["draws"], old["draws"]), (name, "sim draws not bit for bit")
        for k in keys_sim:
            d = _close(new[k], old[k], tol)
            worst = max(worst, d)
            assert d <= tol, (name, "sim", k, d)
        assert np.array_equal(new["draw_idx"], np.arange(n)), (name, new["draw_idx"])
        assert np.array_equal(new["z_idx"], np.arange(n)), (name, new["z_idx"])
        assert new["draw"].all(), name
        for c in prov["banks"]:
            new = np.load(os.path.join(out, "%s_%s.npz" % (stem, c)))
            old = np.load(os.path.join(old_dir, "%s_%s.npz" % (stem, c)))
            assert np.array_equal(new["draws"], old["draws"]), (name, c, "draws not bit for bit")
            for k in keys_coh:
                d = _close(new[k], old[k], tol)
                worst = max(worst, d)
                assert d <= tol, (name, c, k, d)
    print("[r35a] %d ensembles as recorded: draws bit for bit on D_sim and on every cohort; "
          "largest difference of the other arrays %.2g" % (len(AS_NAMES), worst))


def test_r35b_recorded():
    """the recorded-kind project with k_draw = 1: draw rows, z_idx, roles, the bank's file (specs/SPEC.md block 35)."""
    import numpy as np
    import ev_gate
    import ev_rec_config
    import ev_rec_sample
    import ev_sample
    import run_joint_arms as R
    F = rec_fixture.fixture()
    ecfg = ev_rec_config.load(F["ecfg_rec"])
    assert ecfg.sim_draw_rows_per_donor == 1 and not ecfg.as_recorded
    out, prov = _sampled("rec", "S-A1:A1")
    out_as, _p = _sampled("as", "S-A1:A1")
    assert prov["view_kind"] == "recorded" and prov["banks"] == ["real"], prov["banks"]
    z = np.load(os.path.join(out, "S-A1_A1_sim.npz"))
    za = np.load(os.path.join(out_as, "S-A1_A1_sim.npz"))
    n = z["rows"].shape[0]
    assert np.array_equal(z["rows"], za["rows"])
    assert np.array_equal(z["draw"], z["calib"]), "the draw rows are not the calibration rows"
    assert np.array_equal(z["draw_idx"], np.flatnonzero(z["calib"]))
    n_d = int(z["calib"].sum())
    assert n_d < n, "the fixture's D_sim has one row per network: k_draw = 1 tests nothing"
    # the densities at theta_i: draw rows through sample_set, the others by
    # logq_at in other batches. 1e-4 nats: float32 roundoff of batches of
    # another shape (block 19 measured 1.1e-5; the oracle's 1e-4)
    d_lq = _close(z["logq_ref_members"], za["logq_ref_members"], 1e-4)
    assert d_lq <= 1e-4, d_lq
    assert _close(z["logq_ref"], za["logq_ref"], 1e-4) <= 1e-4
    # the draws: n_post / M per member, in member order, member k from seed + k
    M, n_post, p = prov["M"], prov["n_post"], prov["p"]
    assert z["draws"].shape == (n_d, n_post, p)
    per = n_post // M
    assert np.array_equal(z["draw_member"], np.repeat(np.arange(M), per))
    view, index, models = _members("rec", "S-A1:A1")
    x_d = R._rows_x(index, z["rows"][z["draw_idx"]]).numpy()
    for k, m in enumerate(models):
        mine = ev_sample.draws_of(m, x_d, per, prov["seeds"]["sim"] + k, ecfg.batch_rows)
        assert np.array_equal(z["draws"][:, k * per:(k + 1) * per], mine), (
            "member %d's draws are not its own from seed_of(seed, 'sim') + %d" % (k, k))
    # the gate's roles and z_idx
    want = ev_gate.assign_roles(z["donor"], ecfg.gate["donor_fracs"], ecfg.gate["n_ref"],
                                ecfg.gate["n_rank"], ecfg.gate["n_null"],
                                ecfg.gate["rows_per_donor"],
                                ev_rec_sample.seed_of(ecfg.seed, "gate"))
    assert z["gate_role"].dtype == np.int8 and np.array_equal(z["gate_role"], want), (
        z["gate_role"], want)
    g = prov["gate"]
    for key, code in (("n_ref", 1), ("n_rank", 2), ("n_null", 3), ("n_check", 4)):
        assert g[key] == int(np.sum(want == code)), (key, g)
    assert sum(g["networks"]) == np.unique(z["donor"]).size and min(g["networks"]) >= 1, g
    union = np.union1d(np.flatnonzero(z["draw"]), np.flatnonzero(z["gate_role"] != 0))
    assert np.array_equal(z["z_idx"], union), (z["z_idx"], union)
    assert union.size > n_d, "no gate row outside the draw rows: their embeddings untested"
    # the embeddings of every row of z_idx equal the as-recorded run's (every
    # row embedded there): float32 roundoff of batches of another shape
    d_z = float(np.max(np.abs(z["z"] - za["z"][:, z["z_idx"]])))
    scale = float(np.max(np.abs(za["z"])))
    # 1e-5 of the embeddings' scale: float32 has 6e-8 relative precision, a
    # few hundred products' roundoff stays well below; a row placed wrongly
    # differs at the scale itself
    assert d_z <= 1e-5 * max(1.0, scale), (d_z, scale)
    # the file of bank `real`
    zr = np.load(os.path.join(out, "S-A1_A1_real.npz"))
    assert sorted(zr.files) == sorted(ev_rec_sample.REC_ARRAYS), sorted(zr.files)
    assert not [k for k in FORBIDDEN_REC if k in zr.files], zr.files
    old_r0 = np.load(os.path.join(F["old_out"], "samples", "S-A1_A1_R0.npz"))
    assert np.array_equal(zr["stats"], old_r0["stats"]), "the bank's statistics are not R0's"
    for k in ("donor", "well", "cls"):
        assert np.array_equal(zr[k], old_r0[k]), k
    print("[r35b] %d of %d rows drawn; logq_ref_members within %.2g nats of the as-recorded "
          "run's; z within %.2g (scale %.2g); %d rows embedded; the gate %d/%d/%d/%d"
          % (n_d, n, d_lq, d_z, scale, union.size, g["n_ref"], g["n_rank"], g["n_null"], g["n_check"]))


def test_r35c_contract():
    """every key, shape and dtype; the same seed, identical files (specs/SPEC.md block 35)."""
    import numpy as np
    import ev_rec_sample
    F = rec_fixture.fixture()
    for kind, name in (("rec", "S-A1:A1"), ("as", "S-A1:A1"), ("rec", "S-A2-R0:A2")):
        out, prov = _sampled(kind, name)
        stem = prov["stem"]
        with open(os.path.join(out, "%s_sample.json" % stem)) as fh:
            pj = json.load(fh)
        assert sorted(pj) == sorted(ev_rec_sample.PROVENANCE_KEYS), sorted(
            set(pj) ^ set(ev_rec_sample.PROVENANCE_KEYS))
        assert pj["schema"] == ev_rec_sample.SCHEMA == "rec_eval_sample/1"
        assert len(pj["oracle_max_abs_nats"]) == pj["M"] == len(pj["members"])
        assert max(pj["oracle_max_abs_nats"]) <= pj["oracle_tol"]
        for mem in pj["members"]:
            assert all(k in mem for k in ("record", "ckpt", "ckpt_sha256", "seed")), mem
        assert sorted(pj["gate"]) == ["n_check", "n_null", "n_rank", "n_ref", "networks", "seed"]
        assert len(pj["axes"]["names"]) == pj["p"]
        files = sorted(os.path.basename(f) for f in pj["files"])
        assert files == sorted(["%s_sim.npz" % stem] + ["%s_%s.npz" % (stem, r)
                                                        for r in pj["banks"]]), files
        assert sorted(_files_under(out)) == sorted(files + ["%s_sample.json" % stem]), \
            _files_under(out)
        z = np.load(os.path.join(out, "%s_sim.npz" % stem))
        dims = {"n": pj["n"], "n_d": pj["n_draw"], "n_z": pj["n_z"], "M": pj["M"], "p": pj["p"],
                "n_post": pj["n_post"], "d_z": z["z"].shape[2]}
        _check_contract(os.path.join(out, "%s_sim.npz" % stem), ev_rec_sample.SIM_ARRAYS, dims)
        assert np.all(np.diff(z["rows"]) > 0) and np.all(np.diff(z["draw_idx"]) > 0) and \
            np.all(np.diff(z["z_idx"]) > 0), "rows, draw_idx or z_idx not ascending"
        assert np.array_equal(z["draw_idx"], np.flatnonzero(z["draw"]))
        assert np.all((z["draws"] > 0) & (z["draws"] < 1)) and np.all(np.isfinite(z["z"]))
        for r in pj["banks"]:
            zr = np.load(os.path.join(out, "%s_%s.npz" % (stem, r)))
            dims_r = dict(dims, n=zr["rows"].shape[0])
            _check_contract(os.path.join(out, "%s_%s.npz" % (stem, r)), ev_rec_sample.REC_ARRAYS,
                            dims_r)
            assert np.array_equal(zr["rows"], np.arange(dims_r["n"]))
    # the same seed: identical arrays; another seed: other draws, the same densities at theta_i
    out1, prov1 = _sampled("rec", "S-A1:A1")
    out2, _p2 = _sampled("rec", "S-A1:A1", tag="again")
    for f in prov1["files"]:
        a, b = np.load(f), np.load(os.path.join(out2, os.path.basename(f)))
        assert a.files == b.files
        for k in a.files:
            assert np.array_equal(a[k], b[k]), (os.path.basename(f), k)
    d = _tmpdir("seed")
    other = _ecfg_copy(F["ecfg_rec"], d, "rec_eval_seed.json", seed=SEED)
    out3, prov3 = _sampled("rec", "S-A1:A1", ecfg=other)
    assert prov3["seeds"]["sim"] != prov1["seeds"]["sim"]
    for f in prov1["files"]:
        a, b = np.load(f), np.load(os.path.join(out3, os.path.basename(f)))
        assert not np.array_equal(a["draws"], b["draws"]), os.path.basename(f)
        if "logq_ref_members" in a.files:
            assert np.array_equal(a["logq_ref_members"], b["logq_ref_members"])
    assert prov3["gate"]["seed"] != prov1["gate"]["seed"]
    print("[r35c] the three files of 3 samplings hold the contract; a second run gives the same "
          "arrays; another seed other draws")


def test_r35d_refusals():
    """five refusals that write nothing (specs/SPEC.md block 35)."""
    F = rec_fixture.fixture()
    d = _tmpdir("refuse")
    cases = []
    o = os.path.join(d, "o1")
    # its out_root a folder of this test's: the last case below runs without --out
    cases.append((F["cfg_rec"], _ecfg_copy(F["ecfg_rec"], d, "npost.json", n_post=41,
                                           out_root=os.path.join(d, "out_root")),
                  "S-A1:A1", ("REFUSE: n_post 41 is not a multiple of the 2 members",), o))
    # 1e-9 nats: under the rebuild's float32 roundoff on the recorded kind
    # (draw rows and the others in batches of other shapes, about 1e-5)
    o = os.path.join(d, "o2")
    cases.append((F["cfg_rec"], _ecfg_copy(F["ecfg_rec"], d, "tight.json", oracle_tol=1e-9),
                  "S-A1:A1", ("does not reproduce its report-split NLL", "above oracle_tol 1e-09"),
                  o))
    # a member's record of another split hash: a copy of S-A1's best fit whose
    # records name another hash (the checkpoints and per-row files linked)
    runs = os.path.join(d, "runs")
    with open(F["cfg_rec"]) as fh:
        cfg = json.load(fh)
    src = os.path.join(cfg["paths"]["runs_root"], "S-A1", "synth_best")
    dst = os.path.join(runs, "S-A1", "synth_best")
    os.makedirs(dst)
    shutil.copy(os.path.join(src, "BEST_FIT_COMPLETE.json"), dst)
    for mdir in sorted(os.listdir(src)):
        if not mdir.startswith("member_seed"):
            continue
        os.makedirs(os.path.join(dst, mdir))
        for f in os.listdir(os.path.join(src, mdir)):
            a, b = os.path.join(src, mdir, f), os.path.join(dst, mdir, f)
            if f.endswith("_ckpt.pt") or f.endswith("_perrow.npz"):
                os.symlink(a, b)
            elif f.endswith(".json"):
                with open(a) as fh:
                    rec = json.load(fh)
                rec["split_hash"] = "0" * 64
                with open(b, "w") as fh:
                    json.dump(rec, fh)
    cfg["paths"]["runs_root"] = runs
    cfg_split = os.path.join(d, "synth_rec_split.json")
    with open(cfg_split, "w") as fh:
        json.dump(cfg, fh, indent=1)
    o = os.path.join(d, "o3")
    cases.append((cfg_split, _ecfg_copy(F["ecfg_rec"], d, "split.json"), "S-A1:A1",
                  ("the bank's split", "is not the record's"), o))
    o = os.path.join(d, "o4")
    cases.append((F["cfg_rec"], F["ecfg_rec"], "S-A9:A1", ("REFUSE: no complete ensemble 'S-A9:A1'",),
                  o))
    o = os.path.join(d, "o5")
    cases.append((F["cfg_rec"], _ecfg_copy(F["ecfg_rec"], d, "nullmin.json", gate={"n_null_min": 3}),
                  "S-A1:A1", ("the gate's null set has 2 row(s), below gate.n_null_min 3",), o))
    for cfg_p, ec, name, needles, odir in cases:
        rc, text = _cli("--config", cfg_p, "--eval-config", ec, "--ensemble", name, "--out", odir)
        assert rc == 3 and all(s in text for s in needles), (name, ec, rc, text[-2000:])
        assert "REC SAMPLE OK" not in text and _files_under(odir) == [], (odir, _files_under(odir))
    # the out_root of a refused run is not written either
    rc, text = _cli("--config", F["cfg_rec"], "--eval-config", cases[0][1], "--ensemble",
                    "S-A1:A1")
    assert rc == 3, (rc, text[-800:])
    o = os.path.join(d, "out_root")
    assert _files_under(o) == [], _files_under(o)
    rc, text = _cli("--config", F["cfg_rec"], "--eval-config", F["ecfg_rec"])
    assert rc == 2 and "--ensemble" in text, (rc, text[-800:])
    print("[r35d] 5 refusals exit 3 and write nothing; the argument error exits 2")


def test_r35e_mixture():
    """the mixture density at the draws, from the members in float64 (specs/SPEC.md block 35)."""
    import numpy as np
    import torch
    import ev_rec_data
    out, prov = _sampled("rec", "S-A1:A1")
    view, _index, models = _members("rec", "S-A1:A1")
    z = np.load(os.path.join(out, "S-A1_A1_real.npz"))
    x = torch.as_tensor(ev_rec_data.recording(view, "real")["x"])
    draws = torch.as_tensor(z["draws"])                                   # (n, S, p)
    with torch.no_grad():
        lq = np.stack([m.estimator.log_prob(draws.transpose(0, 1), condition=x).numpy().T
                       for m in models]).astype(np.float64)               # (M, n, S), sbi's path
    mix = np.log(np.mean(np.exp(lq), axis=0))                             # float64, directly
    own = lq[z["draw_member"], :, np.arange(z["draws"].shape[1])].T
    d_mix = float(np.max(np.abs(mix - z["logq_draws"])))
    d_own = float(np.max(np.abs(own - z["logq_draws_own"])))
    # 1e-4 nats: float32 storage of values of order 10 (1e-6) and the
    # embed-once path against sbi's in batches of another shape (block 19's
    # E19c: 1.1e-5)
    assert d_mix <= 1e-4 and d_own <= 1e-4, (d_mix, d_own)
    # log mean_m exp(l_m) >= l_own - log M, within the same float32 storage
    assert np.all(np.isfinite(z["logq_draws"])) and np.all(z["logq_draws"] >= z["logq_draws_own"]
                                                            - np.log(len(models)) - 1e-4)
    print("[r35e] on bank real (%d windows x %d draws): mixture %.2g, own member %.2g nats from "
          "the members in float64" % (z["draws"].shape[0], z["draws"].shape[1], d_mix, d_own))


def test_r35f_out_of_box():
    """rows of D_sim outside the open box are left out of the oracle, not refused (specs/SPEC.md block 35)."""
    import numpy as np
    import ev_plan
    import ev_rec_sample
    F = rec_fixture.fixture()
    out0, prov0 = _sampled("rec", "S-A1:A1")
    z0 = np.load(os.path.join(out0, "S-A1_A1_sim.npz"))
    assert prov0["n_out_of_box"] == 0 and z0["in_box"].all()
    n = z0["rows"].shape[0]
    i_d = int(np.flatnonzero(z0["draw"])[1])          # a draw (calibration) row
    i_o = int(np.flatnonzero(~z0["draw"])[0])         # a row without draws
    d = _tmpdir("box")
    # exactly 1 and exactly 0, as a label clipped to [0, 1] and stored in float32
    cfg = rec_fixture.boundary_copy(d, [(z0["rows"][i_d], 3, 1.0), (z0["rows"][i_o], 8, 0.0)])
    out = os.path.join(d, "samples")
    lines = []
    prov = ev_rec_sample.run(cfg, F["ecfg_rec"], "S-A1:A1", out=out, log=lines.append)
    assert prov["n_out_of_box"] == 2, prov["n_out_of_box"]
    assert any("2 row(s) with theta outside the open box" in l for l in lines), lines
    assert max(prov["oracle_max_abs_nats"]) <= prov["oracle_tol"], prov["oracle_max_abs_nats"]
    z = np.load(os.path.join(out, "S-A1_A1_sim.npz"))
    want_out = np.zeros(n, dtype=bool)
    want_out[[i_d, i_o]] = True
    assert np.array_equal(~z["in_box"], want_out), np.flatnonzero(~z["in_box"])
    assert np.all(np.isneginf(z["logq_ref"][want_out]))
    assert np.all(np.isneginf(z["logq_ref_members"][want_out]))
    assert np.all(np.isfinite(z["logq_ref"][~want_out]))
    assert z["theta_ref"][i_d, 3] == 1.0 and z["theta_ref"][i_o, 8] == 0.0
    keep = np.ones(z0["theta_ref"].shape, dtype=bool)
    keep[i_d, 3] = keep[i_o, 8] = False
    assert np.array_equal(z["theta_ref"][keep], z0["theta_ref"][keep])
    # what does not depend on theta is the unmodified run's, the draws bit for bit
    for k in ("rows", "donor", "well", "cls", "calib", "draw", "gate_role", "draw_idx",
              "draws", "draw_member", "logq_draws", "logq_draws_own", "stats", "z_idx", "z"):
        assert np.array_equal(z[k], z0[k]), k
    # the members' densities on the other rows: 1e-4 nats, float32 roundoff of batches
    # of another composition (the rows outside the box leave them), as in R35b
    d_lq = float(np.max(np.abs(z["logq_ref_members"][~want_out]
                               - z0["logq_ref_members"][~want_out])))
    assert d_lq <= 1e-4, d_lq
    zr = np.load(os.path.join(out, "S-A1_A1_real.npz"))
    zr0 = np.load(os.path.join(out0, "S-A1_A1_real.npz"))
    assert np.array_equal(zr["draws"], zr0["draws"])
    # every row of D_sim outside the box: a refusal that writes nothing
    d2 = _tmpdir("box_all")
    cfg2 = rec_fixture.boundary_copy(d2, [(r, 0, 1.0) for r in z0["rows"]])
    out2 = os.path.join(d2, "samples")
    try:
        ev_rec_sample.run(cfg2, F["ecfg_rec"], "S-A1:A1", out=out2, log=lambda t: None)
        raise AssertionError("no row in the open box was not refused")
    except ev_plan.RefuseError as exc:
        assert "no row of D_sim has its theta in the open box" in str(exc), exc
    assert _files_under(out2) == [], _files_under(out2)
    print("[r35f] 2 of %d rows outside the open box (a draw row and another): sampled, "
          "n_out_of_box 2, log q = -inf there, the draws bit for bit the unmodified run's, "
          "the other rows' densities within %.2g nats; every row outside: refused" % (n, d_lq))


CHECKS = [("test_r35a_as_recorded", test_r35a_as_recorded),
          ("test_r35b_recorded", test_r35b_recorded),
          ("test_r35c_contract", test_r35c_contract),
          ("test_r35d_refusals", test_r35d_refusals),
          ("test_r35e_mixture", test_r35e_mixture),
          ("test_r35f_out_of_box", test_r35f_out_of_box)]


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
        _cleanup_tmp()
        rec_fixture.cleanup()
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
