#!/usr/bin/env python3
"""Smoke test for blocks 19-22 of specs/SPEC.md: the synthetic experiment's
evaluation (the roadmap synth, step S6).

Checks (specs/SPEC.md blocks 19-22, "Test oracles")
    test_e19a_rebuild    the ensembles found (best fit and the other arm's,
                         S-A1's two cohorts); every member rebuilt reproduces
                         its _perrow.npz row for row (nll on D_sim,
                         nll_pseudo_real on its cohort); a record of another
                         split, a folder of two arms, an incomplete folder:
                         refused or listed
    test_e19b_dagger     theta-dagger through the unshifted affine map gives
                         the shifted map's physics at theta* (float64
                         roundoff), the duty capped, in_box exact, delta =
                         0.35 pi; R0 (pi 0) has none
    test_e19c_mixture    the stored mixture log density equals log mean_m
                         exp(log q_m) computed in float64 through sbi's own
                         log_prob; the own-member densities; the embed-once
                         densities equal sbi's; log q at theta^ref -inf
                         exactly outside the open box
    test_e19d_contract   every file's keys, shapes and dtypes; n_post / M
                         draws per member; the same run twice gives the
                         same arrays; the seeds common to the ensembles;
                         refusals (n_post not a multiple of M, a calibration
                         bank at a bank's seed or of another size, an unknown
                         ensemble, a bad configuration) write nothing; exit 2
                         on the arguments
    test_e19e_ppc        the posterior-predictive statistics recomputed from
                         the stored draws at one process equal the sampler's
                         (two processes); the cohort's every window, none for
                         theta-dagger; the calibration file's theta and
                         statistics are the bank's, one window per theta,
                         built with ev_plan.calbank_build's arguments
    test_e20a_quantities the quantities on cases with a known answer: the
                         HPD rank and coverage by hand; a conjugate Gaussian
                         model (10 axes) whose exact posterior gives the
                         diagonal within the KS bound, a too narrow one a
                         negative area, a too wide one a positive area; the
                         binomial band; the donor bootstrap unchanged by
                         copies of a donor's rows; SBC rejecting only the
                         shifted axis; TARP's x-dependent references seeing
                         a posterior equal to the prior; the recovery of a
                         known posterior; the NLL, the PPC, E3's verdicts,
                         the running minimum, the embeddings' MMD by hand
                         or by construction
    test_e20b_lc2st      sbi's LC2ST through lc2st_fit and lc2st_eval: draws
                         of the prior in place of the posterior give a larger
                         statistic and smaller p-values than the exact
                         posterior's (the Gaussian model, 2 axes)
    test_e20c_metrics    ev_metrics on the fixture's three ensembles: every
                         file of the index; the NLL, the coverage curve and
                         E3 recomputed from the samples; theta-dagger's rows
                         out of the NLL and in SBC; the search's trials in the
                         rounds' order; the same files from one worker and
                         from three; a sample of other values left out, an
                         unknown ensemble refused; the example donors the
                         median of their condition under the A1 ensemble
    test_e21a_figures    ev_figures on the fixture's metrics: the figures
                         the rules give (per ensemble and data set, per
                         search with both arms, across searches), each in
                         FIGURES.md with a caption and none unlisted, every
                         PNG readable; the corner plots at the example donors
    test_e21b_static     ev_figures computes nothing: no statistics library
                         imported, no mean, SD, quantile or median called;
                         the arms' and conditions' colours the validated
                         palette's; a figure to PDF; no metrics refused
    test_e22a_dry_run    the launcher on the committed synth.json and
                         synth_eval.json (no davinci here): five calibration
                         shards with the committed resources, the builder's
                         arguments of ev_plan.calbank_build and stage I19,
                         every ensemble named not complete, nothing written;
                         the wrapper's --project; a bad --what (skipped where
                         the project's runs root exists: davinci)
    test_e22b_job_script eval_run.pbs: bash -n, ASCII and LF, the activation
                         of activate_env.sh before the markers, no bare conda
                         activate; exits 90, 91, 95, 96 and the command's own;
                         the roadmap's markers pushed to a stand-in origin,
                         done for exit 0 and failed for exit 7
    test_e22c_launcher   the fixture under the fake PBS: the calibration bank
                         and the three samplings submitted, each sampling
                         after both shards, every job exit 0, the job's draws
                         equal to the sampler's in this process; run again:
                         everything done; a job queued is live, not submitted
                         twice; n_post changed: stale; --resample; an unknown
                         ensemble and a calibration bank of another seed
                         refused with nothing written; a changed best fit:
                         stale
    test_e22d_end_to_end the evaluation end to end under the fake PBS: the
                         calibration bank, three samplings and the report
                         after them (stage I22), every job exit 0, the
                         report's three pass lines, REPORT_COMPLETE.json
                         with the current inputs; run again: six done; a
                         resampling puts the report after it; a report value
                         changed: the report alone stale
    test_convergence     not a discretisation: skipped

Run
    python3 hpc/pipeline/tests/test_smoke_evaluation.py [check ...]
        Builds once a reduced project -- T17's and T18's: S-A1 and S-A2-R0
        with alt_best_fit, run to PIPELINE DONE under the fake PBS, about 1
        minute in the sandbox -- and its calibration bank, then checks the
        evaluation on it (about 4 minutes more). EVAL_FIXTURE=<dir> reuses a tree built before (its synth.json,
        banks and runs; the evaluation's files are written under
        <dir>/evaluation_test), EVAL_KEEP=1 keeps the tree it builds and
        prints where. Needs the training environment (torch, sbi, zuko,
        scikit-optimize); without it every check skips. Nothing touches
        davinci.

Pure ASCII, LF only.
"""

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
_EVAL = os.path.join(_PIPE, "evaluation")
for _p in (_HERE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_smoke_synthetic as TS      # noqa: E402  (the reduced project; puts the pipeline first)

PYTHON = sys.executable
SAMPLER = os.path.join(_EVAL, "ev_sample.py")
BUILDER = os.path.join(_REPO, "hpc", "joint", "stage1", "build_latent_bank.py")
SUBMIT = os.path.join(_EVAL, "submit_eval.sh")
JOB = os.path.join(_EVAL, "jobs", "eval_run.pbs")
SYNTH_EVAL = os.path.join(_PIPE, "projects", "synth", "synth_eval.json")

_FIX = {}


# --------------------------------------------------------------- helpers ---

def _need_training_env():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("the training environment is needed (%s)" % exc)


def _build_project(root):
    """T17's and T18's reduced project in `root`, run to PIPELINE DONE: S-A1
    and S-A2-R0 (alt_best_fit true), two members each. Returns the path of
    its synth.json."""
    searches = [{"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False},
                {"name": "S-A2-R0", "campaign": "S-A2", "real": "R0", "alt_best_fit": True}]
    path, _d = TS._small(root, searches=searches,
                         **{"synthetic.sim": {"seed": 11, "n_theta": 60, "n_windows": 1,
                                              "traces_per_shard": 40}})
    saved = dict(os.environ)
    try:
        env = TS._fake_env(os.path.join(root, "spool"))
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", path,
                            "--poll-seconds", "0.5"], cwd=root, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=3600)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert r.returncode == 0 and "PIPELINE DONE: every stage done or skipped" in r.stdout, \
        r.stdout[-6000:]
    return path


def _eval_config(root, name="synth_eval.json", out=None, **over):
    """A reduced evaluation configuration in `root`; returns its path."""
    out = out or os.path.join(root, "evaluation_test")
    d = {"out_root": out, "seed": 4242, "n_post": 40, "batch_rows": 16, "oracle_tol": 1e-4,
         "ppc_rows": 3,
         "ppc_draws": 4, "cal_glob": os.path.join(out, "calbank", "shard_*.npz"), "cal_n": 20,
         "cal_shards": 2, "cal_seed": 55, "lc2st_null": 5, "lc2st_eval_draws": 10,
         "bootstrap_n": 200, "level": 0.95, "dpi": 60, "formats": ["png"], "ncpus": 2,
         "jobs": {"queue": "test", "calbank": {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00"},
                  "sample": {"ncpus": 2, "mem": "2gb", "walltime": "00:20:00"},
                  "report": {"ncpus": 2, "mem": "2gb", "walltime": "00:30:00"}}}
    d.update(over)
    path = os.path.join(root, name)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    return path


def _build_calbank(cfg_path, ecfg_path, seed=None):
    """The calibration bank of ev_plan.calbank_build, by hand (block 22's
    job builds it on davinci); `seed` overrides the builder's seed."""
    import ev_config
    import ev_plan
    cfg = ev_plan.load_project(cfg_path)
    ecfg = ev_config.load(ecfg_path)
    build = ev_plan.calbank_build(cfg, ecfg)
    os.makedirs(build["out_dir"], exist_ok=True)
    for k, argv in enumerate(build["argv"]):
        if os.path.isfile(os.path.join(build["out_dir"], "shard_%04d.npz" % k)):
            continue
        argv = list(argv)
        if seed is not None:
            argv[argv.index("--seed") + 1] = str(int(seed))
        r = subprocess.run([PYTHON, BUILDER, "--out-dir", build["out_dir"]] + argv,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=600)
        assert r.returncode == 0, r.stdout[-3000:]
    return build


def _fixture():
    """dict(root, cfg, ecfg): the project and the reduced evaluation, built
    once per process."""
    if "root" in _FIX:
        return _FIX
    _need_training_env()
    given = os.environ.get("EVAL_FIXTURE", "").strip()
    if given:
        root = os.path.abspath(given)
        cfg = os.path.join(root, "synth.json")
        assert os.path.isfile(cfg), "EVAL_FIXTURE=%s has no synth.json" % root
    else:
        root = tempfile.mkdtemp(prefix="eval_fixture_")
        t0 = time.time()
        cfg = _build_project(root)
        print("[fixture] the reduced project ran to PIPELINE DONE in %.0f s: %s"
              % (time.time() - t0, root), flush=True)
    ecfg = _eval_config(root)
    _build_calbank(cfg, ecfg)
    _FIX.update(root=root, cfg=cfg, ecfg=ecfg, built=not given, samples={})
    return _FIX


def _sampled(name, out=None, ecfg=None):
    """Runs the sampler on ensemble `name` (in this process) into `out` and
    returns (out, provenance); cached per (name, out)."""
    import ev_sample
    F = _fixture()
    key = (name, out)
    if key not in F["samples"]:
        lines = []
        prov = ev_sample.run(F["cfg"], ecfg or F["ecfg"], name, out=out, log=lines.append)
        assert any(l.startswith("EVAL SAMPLE OK: ") for l in lines), "\n".join(lines)
        F["samples"][key] = (os.path.dirname(prov["files"][0]), prov)
    return F["samples"][key]


def _cli(*argv):
    r = subprocess.run([PYTHON, SAMPLER] + list(argv), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)
    return r.returncode, r.stdout


def _files_under(d):
    return sorted(os.path.relpath(os.path.join(a, f), d) for a, _b, fs in os.walk(d) for f in fs) \
        if os.path.isdir(d) else []


# ---------------------------------------------------------------- checks ---

def test_e19a_rebuild():
    import numpy as np
    import torch
    import ev_data
    import ev_models
    import ev_plan
    import run_joint_arms as R
    F = _fixture()
    cfg = ev_plan.load_project(F["cfg"])
    found, missing = ev_plan.ensembles(cfg, with_missing=True)
    names = sorted(e.name for e in found)
    assert names == ["S-A1:A1", "S-A2-R0:A1", "S-A2-R0:A2"] and missing == [], (names, missing)
    by = dict((e.name, e) for e in found)
    assert by["S-A1:A1"].cohorts == ("R0", "R3") and by["S-A1:A1"].kind == "best_fit"
    alt = [e for e in found if e.kind == "best_fit_alt"]
    assert len(alt) == 1 and alt[0].search["name"] == "S-A2-R0" and alt[0].cohorts == ("R0",), alt
    assert alt[0].folder.endswith(os.path.join("S-A2-R0", "synth_best_alt")), alt[0].folder
    index, _paths = ev_data.sim_index(cfg)
    worst = 0.0
    for e in found:
        coh = ev_data.cohort(cfg, e.search["real"])
        for rec_path in e.records:
            model, info = ev_models.rebuild(rec_path, index)
            assert info["arm"] == e.arm and not model.training, info
            rec = json.load(open(rec_path))
            pr = np.load(rec_path[:-len(".json")] + "_perrow.npz")
            rows = ev_data.report_rows(index, rec["split_fracs"], rec["split_seed"],
                                       rec["split_hash"])
            th = torch.as_tensor(np.asarray(index.theta)[rows], dtype=torch.float32)
            with torch.no_grad():
                nll = model.npe_loss(th, R._rows_x(index, rows)).numpy().astype(np.float64)
                nll_r = model.npe_loss(torch.as_tensor(coh["theta"], dtype=torch.float32),
                                       torch.as_tensor(coh["x"])).numpy().astype(np.float64)
            assert nll.shape == pr["nll"].shape and nll_r.shape == pr["nll_pseudo_real"].shape
            d1 = float(np.max(np.abs(nll - pr["nll"])))
            d2 = float(np.max(np.abs(nll_r - pr["nll_pseudo_real"])))
            worst = max(worst, d1, d2)
            # 1e-4 nats: float32 batches of another size than the runner's
            assert d1 <= 1e-4 and d2 <= 1e-4, (rec_path, d1, d2)
    print("[e19a] %d members rebuilt; largest |nll difference| %.2g nats"
          % (sum(len(e.records) for e in found), worst))
    # a record of another split is refused
    tmp = tempfile.mkdtemp(prefix="e19a_")
    try:
        src = by["S-A1:A1"].records[0]
        rec = json.load(open(src))
        rec["split_seed"] = int(rec["split_seed"]) + 6
        dst = os.path.join(tmp, os.path.basename(src))
        with open(dst, "w") as fh:
            json.dump(rec, fh)
        os.symlink(src[:-len(".json")] + "_ckpt.pt", dst[:-len(".json")] + "_ckpt.pt")
        try:
            ev_models.rebuild(dst, index)
            raise AssertionError("a record of another split was rebuilt")
        except ev_plan.RefuseError as exc:
            assert "the bank's split" in str(exc) and "another bank?" in str(exc), exc
        # a folder of two arms is refused; an incomplete one is listed
        d = json.load(open(F["cfg"]))
        d["paths"]["runs_root"] = os.path.join(tmp, "runs")
        bad = os.path.join(tmp, "runs", "S-A1", "synth_best")
        for k, arm in ((1, "A1"), (2, "A2")):
            os.makedirs(os.path.join(bad, "member_seed%d" % k))
            with open(os.path.join(bad, "member_seed%d" % k, "x_seed%d.json" % k), "w") as fh:
                json.dump({"arm": arm}, fh)
        with open(os.path.join(bad, "BEST_FIT_COMPLETE.json"), "w") as fh:
            json.dump({"member_seeds": [1, 2]}, fh)
        p2 = os.path.join(tmp, "synth.json")
        with open(p2, "w") as fh:
            json.dump(d, fh)
        cfg2 = ev_plan.load_project(p2)
        try:
            ev_plan.ensembles(cfg2)
            raise AssertionError("a folder of two arms was taken as an ensemble")
        except ev_plan.RefuseError as exc:
            assert "the members' arms differ: ['A1', 'A2']" in str(exc), exc
        os.remove(os.path.join(bad, "BEST_FIT_COMPLETE.json"))
        found2, missing2 = ev_plan.ensembles(cfg2, with_missing=True)
        assert found2 == [] and len(missing2) == 3, (found2, missing2)
        assert all(w == "no BEST_FIT_COMPLETE.json" for _q, _f, w in missing2), missing2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_e19b_dagger():
    import numpy as np
    from bench_burst_provider import BENCH_AXES, phi_to_bench_params
    from latent_bank import ShardIndex
    import ev_data
    import ev_plan
    F = _fixture()
    cfg = ev_plan.load_project(F["cfg"])
    side3 = ShardIndex(ev_data.cohort_paths(cfg, "R3")).sidecar
    side0 = ShardIndex(ev_data.cohort_paths(cfg, "R0")).sidecar
    pi3 = [r["pi"] for r in cfg.data["synthetic"]["real"] if r["name"] == "R3"][0]
    rng = np.random.default_rng(7)
    n = 4000
    theta = rng.uniform(1e-9, 1.0 - 1e-9, size=(n, len(BENCH_AXES)))
    dag, in_box, delta = ev_data.pseudo_true(theta, side3["gap_spec"])
    assert abs(delta - 0.35 * float(pi3)) <= 1e-15, (delta, pi3)
    fields = ("lambda_b", "sigma_d", "ibi_cv", "n_frag_mean", "lambda_burst", "p_mean", "d_med",
              "duty", "kappa", "lambda_bg")
    worst = 0.0
    for i in range(n):
        a = phi_to_bench_params(theta[i], n_neurons=100, duration_s=10.0, free_axis_shift=delta)
        b = phi_to_bench_params(dag[i], n_neurons=100, duration_s=10.0, free_axis_shift=0.0)
        for f in fields:
            va, vb = getattr(a, f), getattr(b, f)
            worst = max(worst, abs(va - vb) / max(1.0, abs(va)))
    # float64 roundoff of two affine maps of values below 150: 1e-12 relative
    assert worst <= 1e-12, worst
    assert np.all(dag[:, 7] <= 1.0) and np.allclose(dag[:, 7], np.minimum(theta[:, 7] + delta, 1.0),
                                                     rtol=0, atol=1e-15)
    assert np.array_equal(dag[:, :7], theta[:, :7])
    assert np.array_equal(in_box, np.all((dag > 0) & (dag < 1), axis=1))
    expect = (1.0 - delta) ** 3
    frac = float(in_box.mean())
    # binomial SD sqrt(0.72 * 0.28 / 4000) = 0.007; 0.03 is above 4 SD
    assert abs(frac - expect) <= 0.03, (frac, expect)
    dag0, in0, d0 = ev_data.pseudo_true(theta, side0["gap_spec"])
    assert d0 == 0.0 and np.array_equal(dag0, theta) and in0.all(), d0
    print("[e19b] delta %.4f; largest relative difference of the physics %.2g; %.3f of theta* "
          "with theta-dagger in the box (expected %.3f)" % (delta, worst, frac, expect))


def test_e19c_mixture():
    import numpy as np
    import torch
    import ev_data
    import ev_models
    import ev_plan
    import ev_sample
    import run_joint_arms as R
    F = _fixture()
    out, prov = _sampled("S-A1:A1")
    cfg = ev_plan.load_project(F["cfg"])
    e = ev_plan.find(cfg, "S-A1:A1")
    index, _ = ev_data.sim_index(cfg)
    models = [ev_models.rebuild(r, index)[0] for r in e.records]
    M = len(models)
    worst = {"mix": 0.0, "own": 0.0, "embed_once": 0.0, "embed_once_batched": 0.0}
    for ds in ("sim", "R3"):
        z = np.load(os.path.join(out, "S-A1_A1_%s.npz" % ds))
        if ds == "sim":
            x = R._rows_x(index, z["rows"])
        else:
            x = torch.as_tensor(ev_data.cohort(cfg, "R3")["x"])[torch.as_tensor(z["rows"])]
        draws = torch.as_tensor(z["draws"])                          # (n, S, d)
        with torch.no_grad():
            lq = np.stack([m.estimator.log_prob(draws.transpose(0, 1), condition=x).numpy().T
                           for m in models]).astype(np.float64)       # (M, n, S), sbi's path
        mix = np.log(np.mean(np.exp(lq), axis=0))                     # float64, directly
        worst["mix"] = max(worst["mix"], float(np.max(np.abs(mix - z["logq_draws"]))))
        own = lq[z["draw_member"], :, np.arange(z["draws"].shape[1])].T
        worst["own"] = max(worst["own"], float(np.max(np.abs(own - z["logq_draws_own"]))))
        for k, m in enumerate(models):
            one = ev_sample.logq_draws_of(m, z["draws"], x.numpy(), batch=x.shape[0])
            worst["embed_once"] = max(worst["embed_once"], float(np.max(np.abs(one - lq[k]))))
            five = ev_sample.logq_draws_of(m, z["draws"], x.numpy(), batch=5)
            worst["embed_once_batched"] = max(worst["embed_once_batched"],
                                              float(np.max(np.abs(five - lq[k]))))
        # theta^ref: the members' densities, and their mixture
        ref = np.where(z["in_box"], np.log(np.mean(np.exp(z["logq_ref_members"]), axis=1)), -np.inf)
        fin = np.isfinite(ref)
        assert np.array_equal(fin, np.isfinite(z["logq_ref"])) and np.array_equal(fin, z["in_box"])
        assert np.allclose(ref[fin], z["logq_ref"][fin], rtol=0, atol=1e-9)
    # float32 storage of values of order 10: 1e-4 nats. The embed-once path in
    # one batch evaluates the same float32 products as sbi's (equal in the
    # sandbox, 0.0; 1e-6 leaves room for another BLAS); in batches of 5 rows
    # the matrix products change shape and differ by float32 roundoff (1e-5
    # nats in the sandbox), bounded by the oracle's 1e-4
    assert worst["mix"] <= 1e-4 and worst["own"] <= 1e-4, worst
    assert worst["embed_once"] <= 1e-6 and worst["embed_once_batched"] <= 1e-4, worst
    # -inf exactly where theta-dagger leaves the open box, the flow never read
    z = np.load(os.path.join(out, "S-A1_A1_R3dag.npz"))
    zr = np.load(os.path.join(out, "S-A1_A1_R3.npz"))
    out_rows = ~z["in_box"]
    assert out_rows.any(), "the fixture's R3 has no donor with theta-dagger outside the box"
    assert np.all(np.isneginf(z["logq_ref"][out_rows]))
    assert np.all(np.isneginf(z["logq_ref_members"][out_rows]))
    assert np.all(np.isfinite(z["logq_ref"][~out_rows]))
    for k in ("draws", "logq_draws", "logq_draws_own", "z", "stats"):
        assert np.array_equal(z[k], zr[k]), k              # R3's draws and windows
    assert M == 2 and not np.array_equal(z["theta_ref"], zr["theta_ref"])
    print("[e19c] largest differences: mixture %.2g, own member %.2g, embed-once vs sbi %.2g "
          "(one batch) and %.2g (batches of 5) nats; %d of %d R3 rows with theta-dagger outside "
          "the box at -inf" % (worst["mix"], worst["own"], worst["embed_once"],
                               worst["embed_once_batched"], int(out_rows.sum()), out_rows.size))


_NPZ = {"theta_ref": (2, "f8"), "in_box": (1, "b1"), "rows": (1, "i8"), "donor": (1, "i8"),
        "well": (1, "i8"), "cls": (1, "i8"), "logq_ref_members": (2, "f8"), "logq_ref": (1, "f8"),
        "draws": (3, "f4"), "draw_member": (1, "i8"), "logq_draws": (2, "f4"),
        "logq_draws_own": (2, "f4"), "z": (3, "f4"), "stats": (2, "f4"), "calib": (1, "b1"),
        "ppc_rows": (1, "i8"), "ppc_draw_idx": (1, "i8"), "ppc_stats": (3, "f4")}
_CAL = {"theta": (2, "f8"), "stats": (2, "f4"), "draws": (2, "f4"), "member": (1, "i8")}


def _check_npz(path, n_post, M, ppc_draws):
    import numpy as np
    z = np.load(path)
    assert sorted(z.files) == sorted(_NPZ), (path, sorted(z.files))
    for k, (nd, dt) in _NPZ.items():
        assert z[k].ndim == nd and z[k].dtype == np.dtype(dt), (path, k, z[k].shape, z[k].dtype)
    n = z["theta_ref"].shape[0]
    assert z["theta_ref"].shape == (n, 10) and z["draws"].shape == (n, n_post, 10)
    assert z["logq_ref_members"].shape == (n, M) and z["z"].shape[:2] == (M, n)
    for k in ("in_box", "rows", "donor", "well", "cls", "logq_ref", "calib"):
        assert z[k].shape == (n,), (k, z[k].shape)
    for k in ("logq_draws", "logq_draws_own"):
        assert z[k].shape == (n, n_post), (k, z[k].shape)
    assert np.array_equal(z["draw_member"], np.repeat(np.arange(M), n_post // M))
    assert z["stats"].shape == (n, 8) and z["ppc_draw_idx"].shape == (ppc_draws,)
    assert z["ppc_stats"].shape == (z["ppc_rows"].size, ppc_draws, 8)
    assert np.all((z["draws"] > 0) & (z["draws"] < 1)), "a draw outside the open box"
    assert np.all(np.isfinite(z["logq_draws"])) and np.all(np.isfinite(z["z"]))
    return z


def test_e19d_contract():
    import numpy as np
    import ev_config
    F = _fixture()
    ecfg = ev_config.load(F["ecfg"])
    out, prov = _sampled("S-A1:A1")
    names = sorted(os.path.basename(p) for p in prov["files"])
    assert names == ["S-A1_A1_R0.npz", "S-A1_A1_R3.npz", "S-A1_A1_R3dag.npz", "S-A1_A1_cal.npz",
                     "S-A1_A1_sim.npz"], names
    M = len(prov["members"])
    for p in prov["files"]:
        if p.endswith("_cal.npz"):
            z = np.load(p)
            assert sorted(z.files) == sorted(_CAL), sorted(z.files)
            for k, (nd, dt) in _CAL.items():
                assert z[k].ndim == nd and z[k].dtype == np.dtype(dt), (k, z[k].dtype)
            assert z["theta"].shape == (ecfg.cal_n, 10) and z["member"].shape == (ecfg.cal_n,)
        else:
            _check_npz(p, ecfg.n_post, M, ecfg.ppc_draws)
    sim = np.load(os.path.join(out, "S-A1_A1_sim.npz"))
    calib = sim["calib"]
    assert np.unique(sim["donor"][calib]).size == calib.sum() == np.unique(sim["donor"]).size
    assert np.array_equal(sim["ppc_rows"], np.flatnonzero(calib)[:ecfg.ppc_rows])
    assert np.all(np.diff(sim["rows"]) > 0), "D_sim's rows not in global order"
    pj = json.load(open(os.path.join(out, "S-A1_A1_sample.json")))
    assert pj["schema"] == "eval_sample/1" and pj["oracle_tol"] == 1e-4
    assert max(pj["oracle_max_abs_nats"].values()) <= 1e-4 and len(pj["oracle_max_abs_nats"]) == 2 * M
    # the same run twice: the same arrays
    out2, _ = _sampled("S-A1:A1", out=os.path.join(F["root"], "evaluation_test", "samples_again"))
    for p in prov["files"]:
        a, b = np.load(p), np.load(os.path.join(out2, os.path.basename(p)))
        for k in a.files:
            assert np.array_equal(a[k], b[k]), (os.path.basename(p), k)
    # common random numbers: the same seeds for every ensemble
    _o, prov_b = _sampled("S-A2-R0:A2")
    assert prov_b["seeds"]["sim"] == prov["seeds"]["sim"] and prov_b["seeds"]["R0"] == prov["seeds"]["R0"]
    assert prov_b["seeds"]["cal"] == prov["seeds"]["cal"]
    # refusals: nothing written
    root = F["root"]
    tmp = tempfile.mkdtemp(prefix="e19d_", dir=root)
    try:
        cases = []
        e1 = _eval_config(tmp, "n_post.json", out=os.path.join(tmp, "o1"), n_post=41,
                          cal_glob=ecfg.cal_glob)
        cases.append((e1, "S-A1:A1", 3, "REFUSE: n_post 41 is not a multiple of the 2 members",
                      os.path.join(tmp, "o1")))
        e2 = _eval_config(tmp, "calseed.json", out=os.path.join(tmp, "o2"), cal_n=10, cal_shards=1)
        sim_seed = json.load(open(F["cfg"]))["synthetic"]["sim"]["seed"]
        _build_calbank(F["cfg"], e2, seed=sim_seed)
        cases.append((e2, "S-A1:A1", 3, "is a bank's seed", os.path.join(tmp, "o2", "samples")))
        e3 = _eval_config(tmp, "caln.json", out=os.path.join(tmp, "o3"), cal_n=30, cal_shards=2,
                          cal_glob=ecfg.cal_glob)
        cases.append((e3, "S-A1:A1", 3, "one window per theta", os.path.join(tmp, "o3")))
        cases.append((F["ecfg"], "S-A9:A1", 3, "REFUSE: no complete ensemble 'S-A9:A1'", None))
        e5 = _eval_config(tmp, "bad.json", out=os.path.join(tmp, "o5"), n_post=0, level=1.5)
        cases.append((e5, "S-A1:A1", 3, "n_post must be an integer >= 1, got 0; level must be",
                      os.path.join(tmp, "o5")))
        # the oracle's tolerance (2026-11-12): below the rebuild's roundoff it refuses, naming
        # the tolerance, before anything is written; out of range it is a configuration error
        e6 = _eval_config(tmp, "tight.json", out=os.path.join(tmp, "o6"), oracle_tol=1e-6,
                          cal_glob=ecfg.cal_glob)
        cases.append((e6, "S-A1:A1", 3, ("does not reproduce its report-split NLL",
                                         "above oracle_tol 1e-06"), os.path.join(tmp, "o6")))
        e7 = _eval_config(tmp, "loose.json", out=os.path.join(tmp, "o7"), oracle_tol=0.5)
        cases.append((e7, "S-A1:A1", 3, "oracle_tol must be a number of nats in (0, 0.1], got 0.5",
                      os.path.join(tmp, "o7")))
        for ec, name, code, needle, odir in cases:
            rc, text = _cli("--config", F["cfg"], "--eval-config", ec, "--ensemble", name)
            needles = needle if isinstance(needle, tuple) else (needle,)
            assert rc == code and all(n in text for n in needles), (name, ec, rc, text[-2000:])
            if odir is not None:
                assert _files_under(odir) == [], (odir, _files_under(odir))
        rc, text = _cli("--config", F["cfg"], "--eval-config", F["ecfg"])
        assert rc == 2 and "--ensemble" in text, (rc, text[-800:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[e19d] 5 files of the contract; identical on a second run; 7 refusals and the "
          "argument error write nothing")


def test_e19e_ppc():
    import numpy as np
    from latent_bank import ShardIndex
    import ev_config
    import ev_plan
    import ev_sample
    import ev_simulate
    F = _fixture()
    ecfg = ev_config.load(F["ecfg"])
    cfg = ev_plan.load_project(F["cfg"])
    out, prov = _sampled("S-A1:A1")
    sim_side = json.load(open(sorted(glob.glob(ev_plan.sim_bank_glob(cfg)))[0][:-4] + ".json"))
    latent = sim_side["latent_spec"]
    for ds in ("sim", "R0"):
        z = np.load(os.path.join(out, "S-A1_A1_%s.npz" % ds))
        assert np.all(np.isfinite(z["ppc_stats"])), ds
        idx = z["ppc_draw_idx"]
        assert np.array_equal(idx, np.arange(ecfg.ppc_draws) * (ecfg.n_post // ecfg.ppc_draws))
        assert set(z["draw_member"][idx]) == set(range(len(prov["members"])))   # spread over members
        th = z["draws"][z["ppc_rows"]][:, idx].reshape(-1, 10).astype(np.float64)
        xs = ev_simulate.windows_at(th, latent, ev_sample.seed_of(ecfg.seed, ds, "ppc"), ncpus=1)
        assert xs.shape == (th.shape[0], int(sim_side["W"]))
        st = ev_simulate.fixed_stats(xs).reshape(z["ppc_stats"].shape)
        assert np.array_equal(st, z["ppc_stats"]), ds            # one process vs the sampler's two
        if ds == "R0":
            assert np.array_equal(z["ppc_rows"], np.arange(z["rows"].size))
    z = np.load(os.path.join(out, "S-A1_A1_R3dag.npz"))
    assert z["ppc_rows"].size == 0 and z["ppc_stats"].shape == (0, ecfg.ppc_draws, 8)
    # the calibration bank and file
    paths = sorted(glob.glob(ecfg.cal_glob))
    idx = ShardIndex(paths)
    lat = idx.sidecar["latent_spec"]
    assert idx.n_rows == ecfg.cal_n == np.unique(idx.donor).size == np.unique(idx.well).size
    assert lat["kind"] == "box" and int(lat["seed"]) == ecfg.cal_seed, lat
    build = ev_plan.calbank_build(cfg, ecfg)
    import stages.synthetic as SY
    for k, p in enumerate(paths):
        side = json.load(open(p[:-4] + ".json"))
        assert SY._normalise(side["provenance"]["argv"]) == build["argv"][k], k
    cal = np.load(os.path.join(out, "S-A1_A1_cal.npz"))
    assert np.array_equal(cal["theta"], np.asarray(idx.theta, dtype=np.float64))
    x = np.concatenate([idx.load_x(k) for k in range(idx.n_shards)])
    assert np.array_equal(cal["stats"], ev_simulate.fixed_stats(x))
    assert np.all((cal["draws"] > 0) & (cal["draws"] < 1))
    assert set(np.unique(cal["member"])) <= set(range(len(prov["members"])))
    print("[e19e] PPC statistics reproduced from the stored draws; the calibration file is the "
          "bank's %d windows, one per theta" % idx.n_rows)


def _gauss(n, d, rng, s=0.5, n_post=200, scale=1.0, shift=None):
    """The conjugate Gaussian model theta ~ N(0, I_d), x = theta + N(0, s^2 I_d),
    whose exact posterior is N(x / (1 + s^2), s^2 / (1 + s^2) I_d); draws and
    log densities from N(mean + shift, scale^2 times that variance) -- the
    exact posterior at scale 1 and no shift. Returns (theta, x, draws,
    logq_draws, logq_ref)."""
    import numpy as np
    th = rng.normal(size=(n, d))
    x = th + s * rng.normal(size=(n, d))
    sd = scale * math.sqrt(s ** 2 / (1.0 + s ** 2))
    mu = x / (1.0 + s ** 2) + (0.0 if shift is None else np.asarray(shift))
    draws = mu[:, None, :] + sd * rng.normal(size=(n, n_post, d))

    def logq(t, m):
        return (-0.5 * np.sum(((t - m) / sd) ** 2, axis=-1) - d * math.log(sd)
                - 0.5 * d * math.log(2.0 * math.pi))
    return th, x, draws, logq(draws, mu[:, None, :]), logq(th, mu)


def test_e20a_quantities():
    import numpy as np
    from scipy.stats import kstwo
    import ev_metrics as EM
    # the HPD rank and the coverage curve by hand
    g = EM.hpd_ranks(np.array([[1.0, 2.0, 3.0, 4.0]] * 3), np.array([2.5, -np.inf, 10.0]))
    assert np.array_equal(g, [0.5, 1.0, 0.0]), g
    c = EM.coverage_curve(np.array([0.1, 0.5, 0.9]))
    assert abs(c[50] - 2.0 / 3.0) < 1e-15 and c[0] == 0.0 and c[100] == 1.0 and c[9] == 0.0
    # the Gaussian model: the exact posterior's curve is the diagonal within
    # the KS bound at 0.999; too narrow and too wide on either side
    rng = np.random.default_rng(1)
    n = 2000
    th, x, dr, lqd, lqr = _gauss(n, 10, rng)
    curve = EM.coverage_curve(EM.hpd_ranks(lqd, lqr))
    dev = float(np.max(np.abs(curve - EM.BETAS)))
    # the empirical CDF of n uniform ranks (200 draws: a grid of 1/200)
    assert dev <= kstwo.ppf(0.999, n) + 1.0 / 200, dev
    area0 = EM.coverage_summary(curve)["signed_area"]
    narrow = EM.coverage_summary(EM.coverage_curve(EM.hpd_ranks(*_gauss(n, 10, rng, scale=0.5)[3:])))
    wide = EM.coverage_summary(EM.coverage_curve(EM.hpd_ranks(*_gauss(n, 10, rng, scale=2.0)[3:])))
    assert abs(area0) < 0.02 and narrow["signed_area"] < -0.1 and wide["signed_area"] > 0.1, (
        area0, narrow["signed_area"], wide["signed_area"])
    # the binomial band; the donor bootstrap: 12 copies of each donor's row
    # change nothing (the same donors, the same resamples)
    lo, hi = EM.binomial_band(n, EM.BETAS, 0.95)
    assert lo[0] == hi[0] == 0.0 and lo[100] == hi[100] == 1.0
    assert abs((hi[50] - lo[50]) / 2 - 1.96 * math.sqrt(0.25 / n)) < 0.002, (lo[50], hi[50])
    gam = EM.hpd_ranks(lqd, lqr)
    donor = np.arange(n)
    b1 = EM.donor_bootstrap_band(gam, donor, EM.BETAS, 0.95, 2000, seed=5)
    b12 = EM.donor_bootstrap_band(np.repeat(gam, 12), np.repeat(donor, 12), EM.BETAS, 0.95, 2000,
                                  seed=5)
    assert np.allclose(b1[0], b12[0], atol=1e-12) and np.allclose(b1[1], b12[1], atol=1e-12)
    half = (b1[1][50] - b1[0][50]) / 2
    assert abs(half - 1.96 * math.sqrt(curve[50] * (1 - curve[50]) / n)) < 0.003, half
    # SBC: only the shifted axis rejected
    shift = np.zeros(10)
    shift[0] = 0.5
    th, x, dr, _a, _b = _gauss(1000, 10, np.random.default_rng(2), shift=shift)
    sb, sa = EM.sbc_block(th, dr, 0.95, seed=3)
    assert sb["holm_rejected"][0] and sb["n_rejected"] == 1, (sb["n_rejected"], sb["ks_p"])
    assert np.all(sa["sbc_hist_counts"].sum(axis=1) == 1000)
    assert abs(sa["sbc_hist_expected"].sum() - 1000) < 1e-9
    assert np.all(sa["sbc_hist_lo"] <= sa["sbc_hist_expected"]) and np.all(
        sa["sbc_hist_expected"] <= sa["sbc_hist_hi"])
    md = np.max(np.abs(sa["sbc_ecdf_diff"]), axis=1)
    assert md[0] > sb["ks_band_halfwidth"] and md[0] == md.max(), md
    # TARP: a posterior equal to the prior, seen by the x-dependent references only
    th, x, _d, _a, _b = _gauss(1000, 10, np.random.default_rng(4))
    prior = np.random.default_rng(5).normal(size=(1000, 200, 10))
    tb, ta = EM.tarp_block(th, prior, x, seed=6)
    assert tb["x"]["reference_mode"] == "x-dependent" and tb["x"]["ks_p"] < 1e-3, tb
    assert tb["random"]["ks_p"] > 1e-3, tb
    assert ta["tarp_x_f"].shape == (1000,)
    # the recovery of a known posterior N(mu_i, 0.1^2) per axis
    rng = np.random.default_rng(7)
    mu = rng.uniform(0.2, 0.8, size=(500, 10))
    drw = mu[:, None, :] + 0.1 * rng.normal(size=(500, 400, 10))
    rec = EM.recovery_arrays(mu, drw)
    rs = EM.recovery_summary(mu, rec, np.ones(500, dtype=bool))
    # 4 standard errors of a mean of 400 draws; of an SD, 0.1 / sqrt(2 * 400) each
    assert np.max(np.abs(rec["mhat"] - mu)) < 4 * 0.1 / 20 * 1.5
    assert np.max(np.abs(np.array(rs["contraction"]) - (1 - 0.01 * 12))) < 0.01, rs["contraction"]
    assert min(rs["interval90_coverage"]) == 1.0 and max(np.abs(rs["zeta_mean"])) < 0.05
    rec2 = EM.recovery_arrays(mu + 0.2, drw)
    rs2 = EM.recovery_summary(mu + 0.2, rec2, np.ones(500, dtype=bool))
    assert max(rs2["interval90_coverage"]) < 0.05 and max(np.abs(np.array(rs2["zeta_mean"]) + 2.0)) < 0.1
    # the NLL by hand
    s_, a_ = EM.nll_block(np.array([-1.0, -2.0, -np.inf, -4.0]),
                          np.array([[-1, -1], [-2, -2], [-9, -9], [-3, -5]], dtype=float),
                          np.array([True, True, False, True]), np.array([0, 0, 1, 1]))
    assert abs(s_["mean"] - 7.0 / 3.0) < 1e-12 and s_["n_out_of_box"] == 1
    assert np.allclose(a_["donor_nll"], [1.5, 4.0]) and np.allclose(s_["members"], [2.0, 8.0 / 3.0])
    # the PPC by hand
    sim = np.tile(np.arange(20, dtype=float)[None, :, None], (3, 1, 8))
    obs = np.stack([np.full(8, 9.5), np.full(8, 25.0), np.full(8, np.nan)])
    ps, pa = EM.ppc_block(obs, sim)
    assert pa["ppc_rank"][0, 0] == 10 and pa["ppc_rank"][1, 0] == 20 and not pa["ppc_outside"][0, 0]
    assert pa["ppc_outside"][1].all() and ps["n_nonfinite"] == 1 and ps["frac_outside_90"][0] == 0.5
    # E3's verdicts
    rng = np.random.default_rng(8)
    donor = np.repeat(np.arange(27), 12)
    # (Dbar, noise, sigma_seed) -> (beyond 2 sigma_seed, the interval excluding 0)
    for dbar, noise, sd, beyond, excl in ((-0.5, 0.05, 0.1, True, True),
                                          (-0.05, 0.05, 0.1, False, True),
                                          (0.05, 2.0, 0.001, True, False)):
        d = dbar + noise * rng.normal(size=donor.size)
        d += dbar - d.mean()                       # the mean exactly dbar
        ok = np.ones(donor.size, dtype=bool)
        r, ra = EM.paired_block(d, np.zeros(donor.size), donor, ok, sd, sd / 2, 2000, 0.95, 9)
        assert abs(r["Dbar"] - dbar) < 1e-12 and r["sigma_seed"] == sd, r
        assert (r["beyond_2_sigma_seed"], r["ci_excludes_0"]) == (beyond, excl), (dbar, noise, r)
        assert r["resolved"] == (beyond and excl) and ra["delta_d"].shape == (27,)
        assert (r["lower_nll"] == "term arm") == (beyond and excl)
    assert EM.running_min([3, None, 2, 5, float("nan"), 1]) == [3, 3, 2, 2, 2, 1]
    # the embeddings: one law, two laws
    rng = np.random.default_rng(10)
    zs = rng.normal(size=(2, 600, 4))
    same, _sa = EM.embedding_block(zs, rng.normal(size=(2, 100, 4)), np.zeros(600), np.zeros(100), 11)
    moved, ma = EM.embedding_block(zs, 1.0 + rng.normal(size=(2, 100, 4)), np.zeros(600),
                                   np.zeros(100), 12)
    assert min(m["p"] for m in same["members"]) > 0.01, same
    assert max(m["p"] for m in moved["members"]) <= 0.01, moved
    assert ma["pca_coords"].shape == (2, 700, 2) and int(ma["pca_source"].sum()) == 100
    print("[e20a] Gaussian model: diagonal within the KS bound (max dev %.3f), areas %.3f / %.3f / "
          "%.3f; SBC, TARP, recovery, NLL, PPC, E3, MMD as constructed"
          % (dev, area0, narrow["signed_area"], wide["signed_area"]))


def test_e20b_lc2st():
    _need_training_env()
    import numpy as np
    import ev_metrics as EM
    rng = np.random.default_rng(13)
    th, x, dr, _a, _b = _gauss(2000, 2, rng, n_post=1)
    prior = rng.normal(size=(2000, 2))
    tt, tx, tdr, _c, _d = _gauss(20, 2, rng, n_post=100)
    stats = {}
    for name, draws in (("exact", dr[:, 0, :]), ("prior", prior)):
        lc = EM.lc2st_fit({"theta": th, "stats": x, "draws": draws}, 10, seed=14)
        test = tdr if name == "exact" else rng.normal(size=tdr.shape)
        t, p = EM.lc2st_eval(lc, test, tx, np.arange(20), 50)
        stats[name] = (float(t.mean()), float(np.mean(p < 0.05)))
    assert stats["prior"][0] > stats["exact"][0], stats
    assert stats["prior"][1] > stats["exact"][1], stats
    print("[e20b] L-C2ST mean statistic / fraction p < 0.05: exact %.4f / %.2f, prior %.4f / %.2f"
          % (stats["exact"] + stats["prior"]))


def test_e20c_metrics():
    import numpy as np
    import ev_config
    import ev_metrics as EM
    import ev_plan
    F = _fixture()
    for name in ("S-A1:A1", "S-A2-R0:A1", "S-A2-R0:A2"):
        _sampled(name)
    ecfg = ev_config.load(F["ecfg"])
    lines = []
    idx = EM.compute(F["cfg"], F["ecfg"], ncpus=3, log=lines.append)
    assert lines[-1].startswith("EVAL METRICS OK: 3 ensemble(s), 8 data-set file(s), 2 paired "
                                "comparison(s), 2 search(es), 0 left out"), lines[-1]
    md = ecfg.sub("metrics")
    want = ["S-A1_A1_%s" % D for D in ("R0", "R3", "R3dag", "sim")] + [
        "S-A2-R0_%s_%s" % (a, D) for a in ("A1", "A2") for D in ("R0", "sim")] + [
        "paired_S-A2-R0_R0", "paired_S-A2-R0_sim"]
    for w in want:
        for ext in (".json", ".npz"):
            assert os.path.join(md, w + ext) in idx["files"] and os.path.isfile(os.path.join(md, w + ext)), w
    for q in ("S-A1", "S-A2-R0"):
        assert os.path.isfile(os.path.join(md, "search_%s.json" % q))
    sd = ecfg.sub("samples")
    for stem, D in (("S-A1_A1", "sim"), ("S-A1_A1", "R3dag"), ("S-A2-R0_A2", "R0")):
        z = np.load(os.path.join(sd, "%s_%s.npz" % (stem, D)))
        js = json.load(open(os.path.join(md, "%s_%s.json" % (stem, D))))
        ar = np.load(os.path.join(md, "%s_%s.npz" % (stem, D)))
        ok = z["in_box"]
        assert abs(js["nll"]["mean"] - float(np.mean(-z["logq_ref"][ok]))) < 1e-12, (stem, D)
        assert js["nll"]["n_out_of_box"] == int((~ok).sum())
        rows = z["calib"] & ok
        curve = EM.coverage_curve(EM.hpd_ranks(z["logq_draws"][rows], z["logq_ref"][rows]))
        assert np.array_equal(curve, ar["coverage"]), (stem, D)
        # one row per donor on D_sim (the fixture's gammas are all 1, so the
        # curve alone cannot tell the rows apart)
        assert js["coverage"]["n_rows"] == int(rows.sum()), (stem, D, js["coverage"]["n_rows"])
        assert js["sbc"]["n_rows"] == int(z["calib"].sum()) and ar["sbc_ranks"].shape[0] == int(
            z["calib"].sum())
        if D == "R3dag":
            assert (~ok).any() and np.all(np.isnan(ar["gamma"][~ok]))
    zt = np.load(os.path.join(sd, "S-A2-R0_A2_R0.npz"))
    za = np.load(os.path.join(sd, "S-A2-R0_A1_R0.npz"))
    pj = json.load(open(os.path.join(md, "paired_S-A2-R0_R0.json")))
    d = -zt["logq_ref"] + za["logq_ref"]
    assert abs(pj["Dbar"] - float(d.mean())) < 1e-12 and pj["term_arm"] == "A2", pj
    sds = [float(np.std(-np.mean(z["logq_ref_members"], axis=0), ddof=1)) for z in (zt, za)]
    assert abs(pj["sigma_seed"] - max(sds)) < 1e-12, (pj["sigma_seed"], sds)
    srch = json.load(open(os.path.join(md, "search_S-A2-R0.json")))
    cfg = ev_plan.load_project(F["cfg"])
    rd = os.path.join(F["root"], "tune_root", "S-A2-R0", "S-A2")
    order = [t for r in json.load(open(os.path.join(rd, "pipeline_rounds.json")))["rounds"]
             for t in r["trial_ids"]]
    assert [t["trial_id"] for t in srch["trials"]] == order
    for t in srch["trials"]:
        assert t["arm"] == json.load(open(os.path.join(rd, "trials", t["trial_id"] + ".json")))["arm"]
    assert srch["finalist"] == json.load(open(os.path.join(rd, "SEARCH_COMPLETE.json")))["finalist"][
        "trial_id"]
    assert sorted(srch["ensembles"]) == ["S-A2-R0:A1", "S-A2-R0:A2"]
    # the example donors: per condition the median lbar_d under the search's A1 ensemble
    ex = srch["example_donors"]["R0"]
    assert ex["chosen_by"] == "S-A2-R0:A1", ex
    ma = np.load(os.path.join(md, "S-A2-R0_A1_R0.npz"))
    for x in ex["donors"]:
        same = [d for d, i in zip(ma["donor_ids"], range(ma["donor_ids"].size))
                if ma["cls"][list(ma["donor"]).index(d)] == x["cls"]]
        vals = sorted(float(ma["donor_nll"][list(ma["donor_ids"]).index(d)]) for d in same)
        assert x["lbar"] == vals[(len(vals) - 1) // 2], (x, vals)
        assert ma["donor"][x["row"]] == x["donor"] and x["row"] == list(ma["donor"]).index(x["donor"])
    assert sorted(x["cls"] for x in ex["donors"]) == sorted(set(ma["cls"].tolist()))
    exs = json.load(open(os.path.join(md, "search_S-A1.json")))["example_donors"]
    assert sorted(exs) == ["R0", "R3"] and exs["R3"]["chosen_by"] == "S-A1:A1", exs
    # one worker gives the same files as three
    keep = dict((f, open(f, "rb").read()) for f in idx["files"] if f.endswith(".npz"))
    jkeep = dict((f, json.load(open(f))) for f in idx["files"] if f.endswith(".json"))
    EM.compute(F["cfg"], F["ecfg"], ncpus=1, log=lambda t: None)
    for f, b in keep.items():
        a1, a2 = np.load(f), np.load(__import__("io").BytesIO(b))
        for k in a1.files:
            assert np.array_equal(a1[k], a2[k], equal_nan=a1[k].dtype.kind == "f"), (f, k)
    for f, j in jkeep.items():
        assert json.load(open(f)) == j, f
    # a sample of other values is left out; nothing current: refused; an unknown ensemble
    ec2 = _eval_config(F["root"], "eval_e20c.json", out=ecfg.out_root, n_post=20)
    try:
        EM.compute(F["cfg"], ec2, log=lambda t: None)
        raise AssertionError("samples of other values were reported")
    except ev_plan.RefuseError as exc:
        assert "no ensemble has a current sample" in str(exc), exc
    r = subprocess.run([PYTHON, os.path.join(_EVAL, "ev_metrics.py"), "--config", F["cfg"],
                        "--eval-config", F["ecfg"], "--ensemble", "S-A9:A1"], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=300)
    assert r.returncode == 3 and "REFUSE: no complete ensemble S-A9:A1" in r.stdout, r.stdout[-800:]
    print("[e20c] 8 data-set files, 2 paired, 2 searches; recomputed from the samples; the same "
          "files from one worker and from three")


def _expected_figures(md):
    """The figures block 21's rules give for the metrics in `md`, by
    relative path (an independent statement of SPEC block 21's list)."""
    idx = json.load(open(os.path.join(md, "METRICS.json")))
    want = set()
    for name, info in idx["ensembles"].items():
        stem, dss = info["stem"], info["datasets"]
        base = "ensembles/%s/" % stem
        want |= set(base + n for n in ("cal_coverage", "cal_tarp", "acc_sigma", "acc_nll",
                                       "post_lc2st"))
        coh = [D for D in dss if D != "sim" and not D.endswith("dag")]
        for D in dss:
            want |= set(base + n % D for n in ("cal_sbc_ecdf_%s", "cal_sbc_hist_%s",
                                                "acc_recovery_%s", "acc_zscore_%s"))
        for D in ["sim"] + coh:
            want.add(base + "post_ppc_%s" % D)
        for D in coh:
            want.add(base + "emb_pca_%s" % D)
        ex = json.load(open(os.path.join(md, "search_%s.json" % info["search"])))["example_donors"]
        for D in coh:
            for x in ex.get(D, {}).get("donors", []):
                want.add(base + "post_corner_%s_c%d" % (D, x["cls"]))
    by = {}
    for name, info in idx["ensembles"].items():
        by.setdefault(info["search"], {})[info["arm"]] = info
    for q, arms in by.items():
        base = "searches/%s/" % q
        want |= set(base + n for n in ("search_trials", "search_nll_vs_lpr", "search_learning"))
        if len(arms) == 2 and "A1" in arms:
            want.add(base + "arms_coverage")
            term = [a for a in arms if a != "A1"][0]
            shared = [D for D in arms["A1"]["datasets"] if D in arms[term]["datasets"]]
            for D in shared:
                if D != "sim":
                    want.add(base + "arms_nll_%s" % D)
                if D != "sim" and not D.endswith("dag"):
                    want |= {base + "arms_ppc_%s" % D, base + "arms_lc2st_%s" % D}
                want.add(base + "arms_forest_%s" % D)
    if any(os.path.basename(f).startswith("paired_") for f in idx["files"]):
        want.add("across/dbar")
    want.add("across/lpr")
    return want


def test_e21a_figures():
    import ev_config
    import ev_figures as FG
    import ev_metrics as EM
    F = _fixture()
    for name in ("S-A1:A1", "S-A2-R0:A1", "S-A2-R0:A2"):
        _sampled(name)
    ecfg = ev_config.load(F["ecfg"])
    md = ecfg.sub("metrics")
    if not os.path.isfile(os.path.join(md, "METRICS.json")):
        EM.compute(F["cfg"], F["ecfg"], ncpus=3, log=lambda t: None)
    lines = []
    index = FG.make_all(F["cfg"], F["ecfg"], log=lines.append)
    want = _expected_figures(md)
    got = set(r for _f, r, _c in index)
    assert got == want, ("missing %s; extra %s" % (sorted(want - got)[:6], sorted(got - want)[:6]))
    assert lines[-1].startswith("EVAL FIGURES OK: %d figure(s)" % len(want)), lines[-1]
    fdir = ecfg.sub("figures")
    listed = []
    for l in open(os.path.join(fdir, "FIGURES.md")).read().splitlines():
        if l.startswith("- `"):
            rel = l[3:l.index(".{")]
            cap = l[l.index("`: ") + 3:]
            assert cap.endswith(".") and len(cap) > 30, l
            listed.append(rel)
    assert sorted(listed) == sorted(want), "FIGURES.md does not list exactly the figures"
    on_disk = set(os.path.relpath(os.path.join(a, f), fdir)[:-4] for a, _d, fs in os.walk(fdir)
                  for f in fs if f.endswith(".png"))
    assert on_disk == want, sorted(on_disk ^ want)[:6]
    for rel in want:
        with open(os.path.join(fdir, rel + ".png"), "rb") as fh:
            head = fh.read(8)
        assert head == b"\x89PNG\r\n\x1a\n", rel
    # the corner plots: at the example donors of the search, in the caption
    ex = json.load(open(os.path.join(md, "search_S-A1.json")))["example_donors"]["R3"]["donors"]
    caps = dict((r, c) for _f, r, c in index)
    for x in ex:
        cap = caps["ensembles/S-A1_A1/post_corner_R3_c%d" % x["cls"]]
        assert ("donor %d (condition %d" % (x["donor"], x["cls"])) in cap, cap
    print("[e21a] %d figures, each listed with its caption; the corner plots at the example donors"
          % len(want))


def test_e21b_static():
    import ast
    src = open(os.path.join(_EVAL, "ev_figures.py")).read()
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= set(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    banned = {"scipy", "sklearn", "torch", "sbi", "npe_diagnostics", "bootstrap_paired",
              "ev_metrics", "ev_sample"}
    assert not (mods & banned), mods & banned
    calls = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            calls.add(n.func.attr)
    assert not (calls & {"quantile", "percentile", "mean", "std", "var", "median", "nanmean",
                         "nanstd", "average"}), calls & {"quantile", "percentile", "mean", "std"}
    _need_training_env()
    import ev_figures as FG
    # the colours: the validated palette (dataviz validator, PASS all-pairs, light)
    assert FG.ARM_COLOR == {"A1": "#2a78d6", "A2": "#eb6834", "A5": "#1baf7a"}, FG.ARM_COLOR
    assert FG.COND_COLOR == ("#e87ba4", "#6250d6", "#008300") and len(set(FG.COND_MARKER)) == 3
    F = _fixture()
    import ev_config
    ecfg = ev_config.load(F["ecfg"])
    if not os.path.isfile(os.path.join(ecfg.sub("metrics"), "METRICS.json")):
        import ev_metrics as EM
        for name in ("S-A1:A1", "S-A2-R0:A1", "S-A2-R0:A2"):
            _sampled(name)
        EM.compute(F["cfg"], F["ecfg"], ncpus=3, log=lambda t: None)
    st = FG.Store(ecfg)
    FG.style()
    info = st.index["ensembles"]["S-A1:A1"]
    fig, cap = FG.fig_coverage(st, dict(info, name="S-A1:A1"), info["datasets"])
    tmp = tempfile.mkdtemp(prefix="e21b_")
    try:
        p = os.path.join(tmp, "c.pdf")
        fig.savefig(p)
        assert open(p, "rb").read(5) == b"%PDF-", p
        ec = _eval_config(tmp, "none.json", out=os.path.join(tmp, "nothing"))
        r = subprocess.run([PYTHON, os.path.join(_EVAL, "ev_figures.py"), "--config", F["cfg"],
                            "--eval-config", ec], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=300)
        assert r.returncode == 3 and "REFUSE: no metrics at" in r.stdout, r.stdout[-600:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[e21b] no statistics in the figures' code; the validated colours; PDF; refusal")


def _submit(*argv, env=None, timeout=600):
    """The wrapper with SKIP_CONDA=1; (exit code, output)."""
    e = dict(env or os.environ)
    e["SKIP_CONDA"] = "1"
    e.setdefault("ROADMAP_DISABLE", "1")
    r = subprocess.run(["bash", SUBMIT] + list(argv), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout, env=e)
    return r.returncode, r.stdout


def test_e22a_dry_run():
    import ev_config
    import ev_plan
    cfg = ev_plan.load_project(TS.SYNTH)
    ecfg = ev_config.load(SYNTH_EVAL)
    if os.path.isdir(cfg.data["paths"]["runs_root"]):
        raise unittest.SkipTest("the project's runs root %s exists (davinci): its dry run is the run "
                                "sheet's" % cfg.data["paths"]["runs_root"])
    # the committed configuration with paths.sbi_dir at this checkout (davinci's
    # SBI is not here); every other value the committed one
    tmp = tempfile.mkdtemp(prefix="e22a_")
    d = json.load(open(TS.SYNTH))
    d["paths"]["sbi_dir"] = _REPO
    here_cfg = os.path.join(tmp, "synth.json")
    with open(here_cfg, "w") as fh:
        json.dump(d, fh, indent=1)

    def direct(skip):
        e = dict(os.environ)
        e["PIPELINE_SKIP_CONDA"] = skip
        e.pop("ROADMAP_DISABLE", None)
        r = subprocess.run([PYTHON, os.path.join(_EVAL, "ev_submit.py"), "--config", here_cfg,
                            "--eval-config", SYNTH_EVAL, "--dry-run"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, env=e, timeout=120)
        return r.returncode, r.stdout

    rc, out = direct("0")                          # the lines davinci's launcher submits
    assert rc == 0 and "REFUSE" not in out and "ABORT" not in out, out[-3000:]
    assert out.count("SKIP_CONDA=0,") == 5 and "ROADMAP_DISABLE" not in out, out[-2000:]
    assert "EVAL DRY RUN: 5 job(s) would be submitted, 0 done, 0 live; nothing written" in out, out[-800:]
    assert "EVAL SUBMIT: report: nothing to report yet (no ensemble sampled or being sampled)" in out
    names = [q["name"] for q in cfg.data["synthetic"]["searches"]]
    n_alt = sum(1 for q in cfg.data["synthetic"]["searches"] if q["alt_best_fit"])
    assert out.count("EVAL SUBMIT: not complete: ") == len(names) + n_alt == 9, out[-3000:]
    build = ev_plan.calbank_build(cfg, ecfg)
    sbi = _REPO
    j = ecfg.jobs["calbank"]
    for k, argv in enumerate(build["argv"]):
        assert ("calbank:%d would be submitted:" % k) in out, k
        qline = ("qsub -N ev_cal%d -q %s -l select=1:ncpus=%d:mem=%s -l walltime=%s -j oe -o %s"
                 % (k, ecfg.jobs["queue"], j["ncpus"], j["mem"], j["walltime"],
                    os.path.join(ecfg.out_root, "jobs", "logs", "calbank_%d.log" % k)))
        assert qline in out, (qline, out[-2000:])
        cmd = "command: python %s --out-dir %s %s" % (
            os.path.join(sbi, "hpc", "joint", "stage1", "build_latent_bank.py"), build["out_dir"],
            " ".join(argv))
        assert cmd in out, (cmd, out[-2000:])
    assert out.count("ROADMAP_STAGE=I19") == 5 and out.count(
        os.path.join(sbi, "hpc", "pipeline", "evaluation", "jobs", "eval_run.pbs")) == 5
    assert "--n-per-theta 1 --seed %d" % ecfg.cal_seed in out and "--wells-per-donor 1 " in out
    assert not os.path.exists(ecfg.out_root), "the dry run wrote %s" % ecfg.out_root
    e = dict(os.environ)
    e["ROADMAP_DISABLE"] = "0"
    rc2, out2 = _submit("--config", here_cfg, "--eval-config", SYNTH_EVAL, "--dry-run", env=e)
    rc1, out1 = direct("1")
    shutil.rmtree(tmp, ignore_errors=True)
    strip = lambda t: [l for l in t.splitlines() if not l.startswith("[evaluation]")]   # noqa: E731
    assert rc2 == 0 and rc1 == 0 and strip(out2) == strip(out1), "the wrapper gave another plan"
    # --project names the committed files: their sbi_dir is davinci's, absent here
    rc5, out5 = _submit("--project", "synth", "--what", "calbank,sample", "--dry-run")
    assert rc5 == 3 and ("REFUSE: calbank:0's program /davinci-1/home/ldellamea/SBI/hpc/joint/"
                         "stage1/build_latent_bank.py is not in this checkout (git pull)") in out5, out5
    rc3, out3 = _submit("--project", "synth", "--what", "calbank,bogus", "--dry-run")
    assert rc3 == 2 and "ABORT: --what takes a subset of calbank,sample,report" in out3, out3
    rc4, out4 = _submit("--project", "nosuch")
    assert rc4 == 2 and "ABORT: no configuration at" in out4, out4
    print("[e22a] the committed plan: 5 calibration shards, 9 folders not complete, nothing written")


def test_e22b_job_script():
    import stat
    raw = open(JOB, "rb").read()
    for f in (JOB, SUBMIT):
        b = open(f, "rb").read()
        assert b.count(b"\r") == 0 and all(c < 128 for c in b), f
        assert os.stat(f).st_mode & stat.S_IXUSR, "%s is not executable" % f
        r = subprocess.run(["bash", "-n", f], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
        assert r.returncode == 0, (f, r.stdout)
    text = raw.decode("ascii")
    body = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert not any("conda activate" in l for l in body), "a bare conda activate in eval_run.pbs"
    i_act = body.index('. "$JOBS_DIR/activate_env.sh"')
    i_pa = [k for k, l in enumerate(body) if l.startswith("if ! pipeline_activate")][0]
    i_mark = body.index("roadmap_job_start")
    assert i_act < i_pa < i_mark < body.index("\"${ARGV[@]}\"")
    assert "export ROADMAP_N_TASKS=1" in body and "trap 'roadmap_job_end $?' EXIT" in body
    tmp = tempfile.mkdtemp(prefix="e22b_")
    try:
        def job(cmd_lines, **v):
            argv_file = os.path.join(tmp, "cmd.argv")
            with open(argv_file, "w") as fh:
                fh.write("\n".join(cmd_lines) + "\n")
            env = dict(os.environ)
            env.update({"ARGV_FILE": argv_file, "WORKDIR": _REPO, "SKIP_CONDA": "1",
                        "JOBS_DIR": os.path.join(_PIPE, "jobs"), "WORK_ROOT": tmp,
                        "ROADMAP_STAGE": "", "ROADMAP_DISABLE": "1", "PBS_JOBID": "77.fake"})
            env.update(v)
            r = subprocess.run(["bash", JOB], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, env=env, timeout=120)
            return r.returncode, r.stdout
        py = [PYTHON, "-c", "import sys; print('ARGS', sys.argv[1:]); sys.exit(int(sys.argv[2]))",
              "a b", "7"]
        rc, out = job(py)
        assert rc == 7 and "ARGS ['a b', '7']" in out and out.rstrip().endswith("EVAL JOB EXIT: 7"), out
        rc, out = job(py, WORKDIR=os.path.join(tmp, "nowhere"))
        assert rc == 90 and "EVAL JOB EXIT: 90" in out, out
        rc, out = job(py, ARGV_FILE=os.path.join(tmp, "none.argv"))
        assert rc == 91 and "EVAL JOB EXIT: 91" in out, out
        rc, out = job(py, SKIP_CONDA="0", JOBS_DIR=tmp)
        assert rc == 95 and "no activate_env.sh" in out, out
        e = dict(os.environ)
        if not any(os.path.isfile(os.path.join(d, "conda")) for d in e.get("PATH", "").split(":")):
            rc, out = job(py, SKIP_CONDA="0", ENV_NAME="no_such_env_e22b",
                          HOME=os.path.join(tmp, "home"), CONDA_ENVS_PATH="")
            assert rc == 96 and "the environment no_such_env_e22b cannot start" in out, out
        # the markers, pushed to a stand-in origin
        origin = os.path.join(tmp, "origin.git")
        clone = os.path.join(tmp, "clone")
        git = lambda *a: subprocess.run(["git"] + list(a), stdout=subprocess.PIPE,  # noqa: E731
                                        stderr=subprocess.STDOUT, universal_newlines=True)
        assert git("init", "-q", "--bare", origin).returncode == 0
        assert git("clone", "-q", origin, clone).returncode == 0
        assert git("-C", clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                   "--allow-empty", "-m", "init").returncode == 0
        assert git("-C", clone, "push", "-q", "origin", "HEAD").returncode == 0
        for jid, code, state in (("81.fake", "0", "done"), ("82.fake", "7", "failed")):
            rc, out = job([PYTHON, "-c", "import sys; sys.exit(%s)" % code], ROADMAP_DISABLE="0",
                          ROADMAP_STAGE="I19", ROADMAP_REPO=clone, PBS_JOBID=jid)
            assert rc == int(code), out
            r = git("-C", origin, "show", "hpc-status:runs/I19/%s.json" % jid)
            assert r.returncode == 0, (jid, r.stdout, out)
            m = json.loads(r.stdout)
            assert m["state"] == state and m["n_tasks"] == 1 and m["last_exit"] == int(code), m
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[e22b] eval_run.pbs: exits 90, 91, 95, 96 and 7; markers done and failed pushed")


def _fake_pbs_env(spool, delay="0.2"):
    env = dict(os.environ)
    env.update({"PATH": TS._FAKE + os.pathsep + env.get("PATH", ""), "FAKE_PBS_SPOOL": spool,
                "FAKE_PBS_DELAY": delay, "FAKE_PBS_ENFORCE_WALLTIME": "0", "OMP_NUM_THREADS": "1",
                "PIPELINE_QSTAT_RETRY_WAIT_S": "0.2", "ROADMAP_DISABLE": "1"})
    os.makedirs(spool, exist_ok=True)
    return env


def test_e22c_launcher():
    import numpy as np
    F = _fixture()
    root = F["root"]
    out_root = os.path.join(root, "ev_e22c")
    shutil.rmtree(out_root, ignore_errors=True)
    ec = _eval_config(root, "eval_e22c.json", out=out_root)
    spool = os.path.join(root, "spool_e22c")
    shutil.rmtree(spool, ignore_errors=True)
    env = _fake_pbs_env(spool)
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec, "--what", "calbank,sample", env=env)
    assert rc == 0 and "EVAL SUBMIT OK: 5 job(s) submitted, 0 done, 0 live" in out, out[-3000:]
    t0 = time.time()
    want = ["S-A1_A1", "S-A2-R0_A1", "S-A2-R0_A2"]
    while time.time() - t0 < 900:
        recs = TS._records(spool)
        if len(recs) == 5 and all(r["finished"] for r in recs):
            break
        time.sleep(2)
    recs = dict((r["name"], r) for r in TS._records(spool))
    assert sorted(recs) == ["ev_cal0", "ev_cal1", "ev_s0b", "ev_s1a", "ev_s1b"], sorted(recs)
    cal_ids = "%s:%s" % (recs["ev_cal0"]["id"], recs["ev_cal1"]["id"])
    for n, r in recs.items():
        assert r["exit_status"] == 0, (n, r.get("comment"), open(r["out"]).read()[-2000:])
        tail = open(r["out"]).read()
        assert tail.rstrip().endswith("EVAL JOB EXIT: 0"), (n, tail[-500:])
        if n.startswith("ev_s"):
            assert r["depend"] == cal_ids, (n, r["depend"])
            assert "EVAL SAMPLE OK: " in tail and r["vars"]["ROADMAP_STAGE"] == "I19", n
        else:
            assert r["depend"] is None, (n, r["depend"])
    ledger = json.load(open(os.path.join(out_root, "jobs", "ledger.json")))
    assert sorted(ledger) == ["calbank:0", "calbank:1", "sample:S-A1:A1", "sample:S-A2-R0:A1",
                              "sample:S-A2-R0:A2"], sorted(ledger)
    # the job's sampling is the sampler's: same draws as in this process
    here, _p = _sampled("S-A1:A1")
    a = np.load(os.path.join(out_root, "samples", "S-A1_A1_sim.npz"))
    b = np.load(os.path.join(here, "S-A1_A1_sim.npz"))
    for k in ("draws", "logq_draws", "logq_ref", "ppc_stats"):
        assert np.array_equal(a[k], b[k]), k
    # run again: done
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec, "--what", "calbank,sample", env=env)
    assert rc == 0 and "EVAL SUBMIT OK: 0 job(s) submitted, 5 done, 0 live" in out, out[-2000:]
    # a queued job is live: not submitted twice
    slow = _fake_pbs_env(spool, delay="60")
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec, "--what", "sample",
                      "--resample", "S-A1:A1", env=slow)
    assert rc == 0 and "EVAL SUBMIT OK: 1 job(s) submitted, 2 done, 0 live" in out, out[-2000:]
    assert "sample:S-A1:A1 submitted" in out and "(stale: --resample)" in out, out[-2000:]
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec, "--what", "sample",
                      "--resample", "S-A1:A1", env=slow)
    assert rc == 0 and "EVAL SUBMIT OK: 0 job(s) submitted, 2 done, 1 live" in out, out[-2000:]
    led = json.load(open(os.path.join(out_root, "jobs", "ledger.json")))
    subprocess.run(["qdel", led["sample:S-A1:A1"]["job_id"]], env=slow, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT)
    # n_post changed: the three samplings are stale
    ec2 = _eval_config(root, "eval_e22c_n20.json", out=out_root, n_post=20)
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec2, "--what", "calbank,sample",
                      "--dry-run", env=env)
    assert rc == 0 and out.count("(stale: n_post was 40, now 20)") == 3, out[-3000:]
    assert "EVAL DRY RUN: 3 job(s) would be submitted, 2 done, 0 live" in out, out[-1000:]
    # refusals, nothing written
    before = _files_under(out_root)
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec, "--resample", "S-A9:A1",
                      "--what", "sample", env=env)
    assert rc == 3 and "REFUSE: --resample names no complete ensemble: S-A9:A1" in out, out
    ec3 = _eval_config(root, "eval_e22c_seed.json", out=out_root, cal_seed=56)
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec3, "--what", "calbank,sample",
                      env=env)
    assert rc == 3 and "holds a calibration bank of another build" in out and "--seed 56" in out, out
    assert _files_under(out_root) == before
    # a changed best fit: stale
    import ev_plan
    cfg = ev_plan.load_project(F["cfg"])
    e = ev_plan.find(cfg, "S-A2-R0:A1")
    bfc = os.path.join(e.folder, "BEST_FIT_COMPLETE.json")
    keep = open(bfc, "rb").read()
    try:
        with open(bfc, "wb") as fh:
            fh.write(keep + b"\n")
        rc, out = _submit("--config", F["cfg"], "--eval-config", ec, "--what", "sample",
                          "--dry-run", env=env)
        assert rc == 0 and ("sample:S-A2-R0:A1 would be submitted (stale: its best fit's "
                            "BEST_FIT_COMPLETE.json changed") in out, out[-2000:]
        assert "EVAL DRY RUN: 1 job(s) would be submitted, 2 done" in out, out[-600:]
    finally:
        with open(bfc, "wb") as fh:
            fh.write(keep)
    print("[e22c] 5 jobs through the fake PBS in %.0f s; the samplings after both shards; "
          "done, live, stale and refused as specified" % (time.time() - t0))


def test_e22d_end_to_end():
    import ev_config
    import ev_plan
    F = _fixture()
    root = F["root"]
    out_root = os.path.join(root, "ev_e22d")
    shutil.rmtree(out_root, ignore_errors=True)
    ec = _eval_config(root, "eval_e22d.json", out=out_root)
    spool = os.path.join(root, "spool_e22d")
    shutil.rmtree(spool, ignore_errors=True)
    env = _fake_pbs_env(spool)
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec, env=env)
    assert rc == 0 and "EVAL SUBMIT OK: 6 job(s) submitted, 0 done, 0 live" in out, out[-3000:]
    t0 = time.time()
    while time.time() - t0 < 1500:
        recs = TS._records(spool)
        if len(recs) == 6 and all(r["finished"] for r in recs):
            break
        time.sleep(3)
    recs = dict((r["name"], r) for r in TS._records(spool))
    assert sorted(recs) == ["ev_cal0", "ev_cal1", "ev_report", "ev_s0b", "ev_s1a", "ev_s1b"], sorted(recs)
    for n, r in recs.items():
        assert r["exit_status"] == 0, (n, r.get("comment"), open(r["out"]).read()[-3000:])
    rep = recs["ev_report"]
    smp = sorted(recs[n]["id"] for n in ("ev_s0b", "ev_s1a", "ev_s1b"))
    assert sorted(rep["depend"].split(":")) == smp and rep["vars"]["ROADMAP_STAGE"] == "I22", rep
    log = open(rep["out"]).read()
    for line in ("EVAL METRICS OK: 3 ensemble(s)", "EVAL FIGURES OK: ", "EVAL REPORT OK: 3 ensemble(s)"):
        assert line in log, (line, log[-2000:])
    cfg = ev_plan.load_project(F["cfg"])
    ecfg = ev_config.load(ec)
    rc_ = json.load(open(os.path.join(out_root, "REPORT_COMPLETE.json")))
    assert rc_["inputs"] == ev_plan.report_inputs(cfg, ecfg) and len(rc_["ensembles"]) == 3, rc_
    n_listed = sum(1 for l in open(rc_["figures_index"]) if l.startswith("- `"))
    assert rc_["n_figures"] == n_listed > 0, (rc_["n_figures"], n_listed)
    # run again: all done; a resampling puts the report after it; a report value changed
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec, env=env)
    assert rc == 0 and "EVAL SUBMIT OK: 0 job(s) submitted, 6 done, 0 live" in out, out[-2000:]
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec, "--resample", "S-A1:A1",
                      "--dry-run", env=env)
    assert rc == 0 and "EVAL DRY RUN: 2 job(s) would be submitted, 4 done, 0 live" in out, out[-2000:]
    assert "report would be submitted (stale: the sampling of this run comes first)" in out
    assert "depend=afterok:<sample_S-A1_A1>" in out, out[-2000:]
    ec2 = _eval_config(root, "eval_e22d_dpi.json", out=out_root, dpi=61)
    rc, out = _submit("--config", F["cfg"], "--eval-config", ec2, "--dry-run", env=env)
    assert rc == 0 and "EVAL DRY RUN: 1 job(s) would be submitted, 5 done, 0 live" in out, out[-2000:]
    assert "report would be submitted (stale: its samples or its values changed" in out, out[-1500:]
    print("[e22d] calibration bank, three samplings and the report through the fake PBS in %.0f s; "
          "REPORT_COMPLETE.json current; done, resampled and stale as specified" % (time.time() - t0))


def test_convergence():
    raise unittest.SkipTest("not a discretisation: no convergence order")


# ------------------------------------------------------------------ main ---

CHECKS = [("test_e19a_rebuild", test_e19a_rebuild), ("test_e19b_dagger", test_e19b_dagger),
          ("test_e19c_mixture", test_e19c_mixture), ("test_e19d_contract", test_e19d_contract),
          ("test_e19e_ppc", test_e19e_ppc), ("test_e20a_quantities", test_e20a_quantities),
          ("test_e20b_lc2st", test_e20b_lc2st), ("test_e20c_metrics", test_e20c_metrics),
          ("test_e21a_figures", test_e21a_figures), ("test_e21b_static", test_e21b_static),
          ("test_e22a_dry_run", test_e22a_dry_run),
          ("test_e22b_job_script", test_e22b_job_script), ("test_e22c_launcher", test_e22c_launcher),
          ("test_e22d_end_to_end", test_e22d_end_to_end),
          ("test_convergence", test_convergence)]


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
            except unittest.SkipTest as exc:
                rows.append((name, "SKIP", str(exc), time.time() - t1))
            except AssertionError as exc:
                bad += 1
                rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
            except Exception:
                bad += 1
                rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    finally:
        if _FIX.get("built"):
            if os.environ.get("EVAL_KEEP", "").strip() == "1":
                print("[fixture] kept: %s" % _FIX["root"])
            else:
                shutil.rmtree(_FIX["root"], ignore_errors=True)
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
