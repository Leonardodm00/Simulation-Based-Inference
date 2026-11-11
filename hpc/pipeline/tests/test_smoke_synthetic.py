#!/usr/bin/env python3
"""Smoke test for blocks 15-17 of specs/SPEC.md: the pipeline's synthetic
kind -- its configuration, its graph and run views, its banks, and the
committed project `synth` (the roadmap synth, step S4).

Checks (specs/SPEC.md blocks 15-17, "Test oracles")
    test_t15_config     the committed synth.json loads; one refusal per
                        rule of sec. 3.1's synthetic kind; three problems
                        of one file listed at once; load_config refuses
                        the file without synthetic_ok
    test_t15_graph      the stages (13, and since block 18 the four best
                        fits of the other arm: 17) and their dependencies
                        in sec. 2.1's order; a recorded configuration's graph unchanged;
                        the run views' folders, campaign and real glob
                        (S-A1's included), their idempotence; the renamed
                        log lines
    test_t15_dry_run    the committed file's dry run: its stages, a qsub
                        line per kind of job, no REFUSE line, exit 0
                        (skipped where the project's work root exists:
                        davinci, whose run sheet has its own dry run)
    test_t16_record     the record tool by hand: a shard of another digest,
                        a cohort with one class short, a shard index too
                        many, a corrupt zip: each refused, nothing written;
                        the clean bank recorded with its build
    test_t16_banks      the fake PBS, the bench generator at W 128: the
                        three banks through their jobs and records; a
                        re-run skips them; a shard deleted is the only one
                        rebuilt; a shard whose job always fails stops the
                        stage after max_resubmits, --retry after the fix
                        completes it without rebuilding the others; a job
                        that cannot start (PBS -2) is submitted again, not
                        counted; a folder of another build is refused and
                        left as it was; a corrupt shard makes the record
                        job refuse
    test_convergence    not a discretisation: skipped
    test_t17_end_to_end the training environment (torch, sbi, zuko,
                        scikit-optimize): a reduced copy of the project with
                        two searches (S-A1 and S-A2 on R0) runs to PIPELINE
                        DONE; both searches and both best fits complete in
                        their run folders; every trial given R0's windows
                        and its record carrying L_pseudo_real; at least one
                        S-A2 trial with the metric term on; nothing over
                        the core cap
    test_t18_config     [block 18] the committed v3 loads; alt_best_fit
                        true on S-A1, true on an S-A25 search, not a bool,
                        missing: each refused
    test_t18_graph      the committed v3's 17 stages, each best_fit_alt_<q>
                        after best_fit_<q>, its dependency, folder, job
                        prefix and renamed log lines; S-A1's view has no
                        other arm
    test_t18_dry_run    the committed v3's dry run: four best fits of the
                        other arm, their ba_ qsub lines, no REFUSE (skipped
                        as test_t15_dry_run)
    test_t18_end_to_end the training environment: a reduced project with
                        the search S-A2-R0 runs to PIPELINE DONE; with
                        alt_best_fit turned on, the run refuses without
                        --accept-config-change, then runs the other arm's
                        best fit alone: two ba_ member jobs, the search's
                        and the best fit's artefacts unchanged, the finalist
                        the lowest-nll trial of the arm the search's
                        finalist is not (from the ledger), members of that
                        arm

Run
    python3 hpc/pipeline/tests/test_smoke_synthetic.py [check ...]
        About 1 minute without the training environment (T17 then skips);
        about 5 more minutes with it (T17, T18 end to end). Needs numpy and scipy (the bench
        generator); names of checks run only those. Nothing touches davinci.

Pure ASCII, LF only.
"""

import copy
import glob
import hashlib
import json
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
_FAKE = os.path.join(_HERE, "fake_pbs")
_STAGE1 = os.path.join(_REPO, "hpc", "joint", "stage1")
for _p in (_STAGE1, _HERE, _PIPE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C      # noqa: E402  (the pipeline's, before anything else)

SYNTH = os.path.join(_PIPE, "projects", "synth", "synth.json")
PYTHON = sys.executable
BUILDER = os.path.join(_STAGE1, "build_latent_bank.py")
RECORD_TOOL = os.path.join(_PIPE, "tools", "synth_bank_record.py")


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_synth_")


def _committed():
    with open(SYNTH, "rb") as fh:
        return C._strip_docs(json.loads(fh.read().decode("ascii")))


def _small(root, searches=None, **over):
    """A reduced synthetic project in `root`: the bench generator at f_s 10
    Hz and T_win 12.8 s (W 128); arm S 12 theta in 3 shards of 8 wells; R0
    and R3 of 3 classes x 2 donors x 2 wells x 2 windows; the space of block
    7's fixture on the bench's anchors. Returns (path, dict)."""
    d = _committed()
    for k in list(d["paths"]):
        d["paths"][k] = os.path.join(root, k)
    d["paths"]["sbi_dir"] = _REPO
    d["synthetic"]["builder"].update({"fs": 10.0, "T_win": 12.8})
    d["synthetic"]["sim"] = {"seed": 11, "n_theta": 12, "n_windows": 1, "traces_per_shard": 8}
    d["synthetic"]["real"] = [
        {"name": "R0", "seed": 22, "donors_per_class": 2, "n_windows": 2, "pi": 0.0,
         "gap_modes": ["range_shift", "drift", "contamination"]},
        {"name": "R3", "seed": 33, "donors_per_class": 2, "n_windows": 2, "pi": 0.3,
         "gap_modes": ["range_shift", "drift", "contamination"]}]
    d["synthetic"]["searches"] = searches or [
        {"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False},
        {"name": "S-A2-R0", "campaign": "S-A2", "real": "R0", "alt_best_fit": False}]
    d["synthetic"]["bank_job"] = {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00",
                                  "queue": "test", "max_resubmits": 1}
    d["search"].update({
        "shape_anchors": {"p": 10, "d_theta": 10, "embedding_dim": 8},
        "space_overrides": {"depth_exponent": [3, 3], "width_multiplier": [1.5, 1.5],
                            "embedding_size": [8, 8], "hidden_features": [24, 24],
                            "num_transforms": [3, 3], "batch_size_npe": [16, 32],
                            "head_pool_ops": [0, 1]},
        "n_initial_points": 2, "gp_rounds": 1, "trials_per_round": 1,
        "max_concurrent_trials": 4, "gp_seed": 3, "train_seed": 0,
        "budget_gate": {"probe_steps": 2, "probe_val_rows": 10, "safety_factor": 1.5}})
    d["search"]["trial"].update({"passes": 1.0, "epochs": 2, "ncpus": 1, "mem": "4gb",
                                 "walltime": "00:10:00", "queue": "test", "max_resubmits": 1})
    d["best_fit"].update({"member_seeds": [1, 2], "passes": 1.0, "epochs": 2,
                          "n_posterior_draws": 16, "ncpus": 1, "mem": "4gb",
                          "walltime": "00:10:00", "queue": "test", "max_resubmits": 1})
    d["loader"] = {"buffer_shards": 3}
    d["driver"] = {"queue": "test", "ncpus": 1, "mem": "1gb", "walltime": "01:00:00",
                   "poll_seconds": 1}
    d["resources"] = {"max_cores": 5}
    for k, v in over.items():
        cur = d
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = v
    path = os.path.join(root, "synth.json")
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    return path, d


def _fake_env(spool, extra=None):
    """The fake PBS for this process and its children (the jobs get the
    fake's environment)."""
    env = {"PATH": _FAKE + os.pathsep + os.environ.get("PATH", "").replace(_FAKE + os.pathsep, ""),
           "FAKE_PBS_SPOOL": spool, "FAKE_PBS_DELAY": "0.2", "FAKE_PBS_ENFORCE_WALLTIME": "0",
           "PIPELINE_SKIP_CONDA": "1", "OMP_NUM_THREADS": "1",
           "PYTHONPATH": _PIPE + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for k in ("FAKE_PBS_FAIL_START",):
        os.environ.pop(k, None)
    env.update(extra or {})
    os.environ.update(env)
    return dict(os.environ)


def _run_banks(cfg_path, retry=(), max_ticks=900):
    """The sequencer in this process on the bank stages alone; (rc, text)."""
    import pipeline as P
    import stages as S
    from stages import synthetic as SY
    cfg = C.load_config(cfg_path, synthetic_ok=True)
    graph = [s for s in S.default_graph(cfg) if isinstance(s, SY.SynthBankStage)]
    lines = []

    def log(text):
        lines.append(str(text))

    rc = P.run(cfg, graph=graph, poll_seconds=0.2, retry=tuple(retry), log=log,
               max_ticks=max_ticks)
    return rc, "\n".join(lines)


def _records(spool):
    out = []
    for p in glob.glob(os.path.join(spool, "*.json")):
        if os.path.basename(p)[:-5].isdigit():
            with open(p) as fh:
                out.append(json.load(fh))
    return sorted(out, key=lambda r: r["num"])


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _build_by_hand(out_dir, argv):
    r = subprocess.run([PYTHON, BUILDER, "--out-dir", out_dir] + list(argv), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=300)
    assert r.returncode == 0, r.stdout
    return r.stdout


def _record_tool(bank_dir, expect, out):
    exp_path = os.path.join(os.path.dirname(out), "expect_test.json")
    with open(exp_path, "w") as fh:
        json.dump(expect, fh)
    r = subprocess.run([PYTHON, RECORD_TOOL, "--bank-dir", bank_dir, "--expect", exp_path,
                        "--out", out], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=300)
    return r.returncode, r.stdout


# ---------------------------------------------------------------- checks ---

def test_t15_config():
    cfg = C.load_config(SYNTH, synthetic_ok=True)
    assert C.is_synthetic(cfg.data) and cfg.tag == "synth"
    try:
        C.load_config(SYNTH)
        raise AssertionError("load_config accepted a synthetic file without synthetic_ok")
    except C.ConfigError as exc:
        assert "a synthetic data set's configuration" in exc.problems[0], exc.problems

    def problems(mutate):
        d = _committed()
        mutate(d)
        return C.validate(d)

    def refused(mutate, needle):
        p = problems(mutate)
        assert any(needle in x for x in p), (needle, p)
        return p

    syn = lambda d: d["synthetic"]                                         # noqa: E731
    refused(lambda d: syn(d)["real"][0].update(seed=syn(d)["sim"]["seed"]),
            "synthetic.sim and R0 share the seed 1000")
    refused(lambda d: syn(d)["real"][1].update(seed=2000), "R0 and R3 share the seed 2000")
    refused(lambda d: syn(d)["searches"][1].update(real="R9"),
            "names the real bank 'R9', which synthetic.real does not list")
    refused(lambda d: syn(d)["searches"][2].update(name="S-A1"), "the names must be unique, S-A1")
    refused(lambda d: syn(d)["searches"][2].update(campaign="S-A2"),
            "are the same search (S-A2 on R0)")
    refused(lambda d: syn(d)["sim"].update(traces_per_shard=3000), "must be a multiple of")
    refused(lambda d: syn(d)["builder"].update(wells_per_donor=1), "wells_per_donor: must be >= 2")
    refused(lambda d: syn(d)["builder"].update(n_classes=1), "n_classes: must be >= 2")
    refused(lambda d: syn(d)["builder"].update(class_centres="simplex"),
            "needs class_centres 'regular'")
    refused(lambda d: syn(d)["real"][1].update(gap_modes=[]),
            "gap_modes must be a non-empty list")
    refused(lambda d: syn(d)["real"][1].update(gap_modes=["drift", "drift"]),
            "gap_modes must be distinct modes")
    refused(lambda d: syn(d)["real"][1].update(gap_modes=["tilt"]),
            "gap_modes must be distinct modes")
    refused(lambda d: syn(d)["real"][1].update(pi=1.5), "pi must be a number in [0, 1]")
    refused(lambda d: syn(d)["builder"].update(provider="reference"), "must be 'bench'")
    refused(lambda d: syn(d)["real"][0].update(extra=1), "must have exactly the keys")
    p = problems(lambda d: d.update(kind="recorded"))
    assert len(p) == 1 and "kind: must be 'synthetic'" in p[0], p
    refused(lambda d: d.update(bank={"on_axis": "O_N"}), "bank.on_axis: unknown key")
    refused(lambda d: d["search"].update(campaign="S-A1"), "search.campaign: unknown key")
    refused(lambda d: syn(d)["sim"].pop("seed"), "synthetic.sim.seed: missing")
    refused(lambda d: (syn(d)["bank_job"].update(ncpus=200)),
            "cannot hold the sequencer's job (driver.ncpus 4) and the largest job "
            "(synthetic.bank_job.ncpus 200)")
    # three problems of one file, all listed
    p = problems(lambda d: (syn(d)["real"][1].update(seed=2000), syn(d)["builder"].update(n_classes=1),
                            syn(d)["sim"].pop("n_windows")))
    assert len(p) == 3, p
    # a documentation key inside a list item is ignored
    d = _committed()
    syn(d)["real"][0]["_why"] = "documented"
    assert C.validate(C._strip_docs(d)) == [], C.validate(C._strip_docs(d))
    # the recorded files are unchanged
    for name in ("giulia", "linddup"):
        C.load_config(os.path.join(_PIPE, "projects", name, name + ".json"))


def test_t15_graph():
    import stages as S
    from stages import common as CM
    from stages import synthetic as SY
    cfg = C.load_config(SYNTH, synthetic_ok=True)
    graph = S.topological_order(S.default_graph(cfg))
    names = [s.name for s in graph]
    want = ["sim_bank", "real_bank_R0", "real_bank_R3"]
    for q in ("S-A1", "S-A2-R0", "S-A5-R0", "S-A2-R3", "S-A5-R3"):
        want += ["search_" + q, "best_fit_" + q] + (["best_fit_alt_" + q] if q != "S-A1" else [])
    assert names == want, names
    deps = dict((s.name, tuple(s.deps)) for s in graph)
    assert deps["search_S-A1"] == ("sim_bank", "real_bank_R0")
    assert deps["search_S-A5-R3"] == ("sim_bank", "real_bank_R3")
    assert deps["best_fit_S-A2-R3"] == ("search_S-A2-R3",)
    assert deps["sim_bank"] == () and deps["real_bank_R3"] == ()
    rec = C.load_config(os.path.join(_PIPE, "projects", "giulia", "giulia.json"))
    assert [s.name for s in S.default_graph(rec)] == list(S.STAGE_ORDER)
    assert CM.shard_globs(rec)[1] is None, "the recorded S-A1 search gets no real bank"
    try:
        CM.shard_globs(cfg)
        raise AssertionError("shard_globs accepted a synthetic configuration without a run")
    except ValueError:
        pass
    p = cfg.data["paths"]
    by = dict((s.name, s) for s in graph)
    for q, camp, real in (("S-A1", "S-A1", "R0"), ("S-A2-R3", "S-A2", "R3")):
        st = by["search_" + q]
        v = st._view(cfg)
        assert st._view(v) is v and st._view(cfg) is v, "a view of a view is itself (cached)"
        assert v.data["paths"]["tune_root"] == os.path.join(p["tune_root"], q)
        assert v.data["paths"]["runs_root"] == os.path.join(p["runs_root"], q)
        assert v.data["search"]["campaign"] == camp and v.sha256 == cfg.sha256
        assert cfg.data["paths"]["tune_root"] == p["tune_root"], "the base is not changed"
        sim, rg = CM.shard_globs(v)
        assert sim == os.path.join(p["banks_root"], "synth_sim", "shard_*.npz")
        assert rg == os.path.join(p["banks_root"], "synth_real_" + real, "shard_*.npz"), rg
        # the reason names the artefact, absent or present (amended 2026-11-11: on davinci the
        # run of 2026-11-07 made them, and this check runs there too)
        why = st.is_done(cfg)[1]
        art = os.path.join(p["tune_root"], q, camp, "SEARCH_COMPLETE.json")
        assert why in (art, art + " absent"), why
        bf = by["best_fit_" + q]
        why = bf.is_done(cfg)[1]
        art = os.path.join(p["runs_root"], q, "synth_best", "BEST_FIT_COMPLETE.json")
        assert why in (art, art + " absent"), why
        try:
            by["search_S-A5-R0"]._view(v)
            raise AssertionError("a stage accepted another run's view")
        except ValueError:
            pass
    # the renamed log lines
    seen = []

    class Rec(object):
        def log(self, t):
            seen.append(t)

        def fits(self, n):
            return "fits %d" % n

    r = SY._StageLog(Rec(), "search", "search_S-A1")
    r.log("search: trial abc ok")
    r.log("search_S-A1: host x listed")
    r.log("other line")
    assert seen == ["search_S-A1: trial abc ok", "search_S-A1: host x listed", "other line"], seen
    assert r.fits(3) == "fits 3", "everything else is the recorder's"
    # the builds: K shards of the configured size, R's cohort
    b = SY.build_of(cfg, "sim")
    assert b["n_shards"] == 20 and b["rows_per_shard"] == [2000] * 20
    assert b["argv"][7][:6] == ["--arm", "S", "--prior", "box", "--shard-index", "7"], b["argv"][7]
    r3 = SY.build_of(cfg, "R3")
    assert r3["n_shards"] == 1 and r3["rows_per_shard"] == [324], r3
    a = r3["argv"][0]
    for flag, val in (("--n-traces", "54"), ("--donors-per-class", "9"), ("--n-windows", "6"),
                      ("--pi", "0.3"), ("--seed", "3000"), ("--class-centres", "regular"),
                      ("--centre-scale", "0.6"), ("--T-win", "180.0"), ("--fs", "100.0"),
                      ("--gap-modes", "range_shift,drift,contamination")):
        assert a[a.index(flag) + 1] == val, (flag, a)
    assert json.loads(json.dumps(b)) == b, "the build survives its JSON copy unchanged"


def _skip_where_run():
    """[amended 2026-11-11] The committed project's dry run reads its state file where its work
    root exists (on davinci, since the run of 2026-11-03): there the run sheet's own dry run is the
    check, not this one."""
    if os.path.isdir(cfg_work_root()):
        raise unittest.SkipTest("%s exists (the committed project ran here): the run sheet's dry "
                                "run is the check" % cfg_work_root())


def test_t15_dry_run():
    _skip_where_run()
    root = _tmp()
    try:
        env = dict(os.environ)
        env["PIPELINE_SKIP_CONDA"] = "1"
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", SYNTH,
                            "--dry-run"], cwd=root, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        out = r.stdout
        assert r.returncode == 0, out
        assert "REFUSE" not in out, out
        for name in ("sim_bank", "real_bank_R0", "real_bank_R3", "search_S-A1", "best_fit_S-A5-R3"):
            assert "| %s " % name in out, (name, out)
        assert out.count("qsub -N sb_") == 3 and out.count("qsub -N sr_") == 3, out
        assert out.count("qsub -N t_<trial>") == 5 and out.count("qsub -N gate_b<B>") == 5, out
        assert out.count("qsub -N bf_synth_s<k>") == 5, out
        assert "bank sim: 20 shard(s) of 2000 row(s) (arm S)" in out, out
        assert "bank R3: 1 shard(s) of 324 row(s) (arm R)" in out, out
        assert "the real bank /davinci-1/home/ldellamea/banks/synth_real_R0/shard_*.npz" in out
        assert "cores: at most 120 at once (resources.max_cores, D-099)" in out, out
        assert "at most 14 trial(s) at once" in out, out
        assert not os.path.exists(cfg_work_root()), "a dry run creates nothing"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def cfg_work_root():
    return _committed()["paths"]["work_root"]


def test_t16_record():
    from stages import synthetic as SY
    root = _tmp()
    try:
        path, d = _small(root)
        cfg = C.load_config(path, synthetic_ok=True)
        st = SY.SynthBankStage("sim")
        exp = st.expect(cfg)
        bdir = exp["dir"]
        for k, argv in enumerate(SY.build_of(cfg, "sim")["argv"]):
            _build_by_hand(bdir, argv)
        out = os.path.join(bdir, "BANK_RECORD.json")
        rc, text = _record_tool(bdir, exp, out)
        assert rc == 0 and "SYNTH BANK RECORD OK: %s (3 shard(s), 24 rows" % out in text, text
        rec = json.load(open(out))
        assert rec["build"] == SY.build_of(cfg, "sim") and rec["n_rows"] == 24 and rec["W"] == 128
        assert rec["latent_kind"] == "box" and rec["theta_withheld"] is False
        assert [s["rows"] for s in rec["shards"]] == [8, 8, 8]
        assert st.is_done(cfg)[0], st.is_done(cfg)
        os.remove(out)
        os.remove(out + ".sha256")
        # (a) a shard of another digest: its sidecar's fields changed and re-stamped
        import latent_bank as LB
        js = os.path.join(bdir, "shard_0001.json")
        keep = open(js).read()
        side = json.loads(keep)
        side["latent_spec"]["tau_ov"] = 0.2
        side["contract_digest"] = LB.contract_digest(side)
        with open(js, "w") as fh:
            fh.write(LB.canonical_json(side))
        rc, text = _record_tool(bdir, exp, out)
        assert rc == 4 and "REFUSE: the shards carry 2 contract digests" in text, text
        assert not os.path.exists(out), "nothing written on a refusal"
        with open(js, "w") as fh:
            fh.write(keep)
        # (b) a shard index the build does not have
        shutil.copy(os.path.join(bdir, "shard_0000.json"), os.path.join(bdir, "shard_0003.json"))
        rc, text = _record_tool(bdir, exp, out)
        assert rc == 4 and "shard files of an index >= 3" in text, text
        os.remove(os.path.join(bdir, "shard_0003.json"))
        # (c) a corrupt zip: bytes flipped in the middle of shard 2's archive
        npz = os.path.join(bdir, "shard_0002.npz")
        blob = bytearray(open(npz, "rb").read())
        mid = len(blob) // 2
        for i in range(mid, mid + 64):
            blob[i] ^= 0xFF
        with open(npz, "wb") as fh:
            fh.write(bytes(blob))
        rc, text = _record_tool(bdir, exp, out)
        assert rc == 4 and "REFUSE: shard 2:" in text and ("CRC" in text or "zip" in text), text
        # (d) a cohort with one class short: R0 rebuilt by hand, one row's class changed
        import numpy as np
        rst = SY.SynthBankStage("R0")
        rexp = rst.expect(cfg)
        _build_by_hand(rexp["dir"], SY.build_of(cfg, "R0")["argv"][0])
        rout = os.path.join(rexp["dir"], "BANK_RECORD.json")
        rc, text = _record_tool(rexp["dir"], rexp, rout)
        assert rc == 0 and "(1 shard(s), 24 rows" in text, text
        os.remove(rout)
        rnpz = os.path.join(rexp["dir"], "shard_0000.npz")
        with np.load(rnpz) as z:
            arrays = dict((k, np.array(z[k])) for k in z.files)
        i0 = int(np.flatnonzero(arrays["cls"] == 0)[0])
        arrays["cls"][i0] = 1
        np.savez_compressed(rnpz[:-4], **arrays)
        rc, text = _record_tool(rexp["dir"], rexp, rout)
        assert rc == 4 and "7 rows of class 0, expected 8" in text and \
            "9 rows of class 1, expected 8" in text, text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_t16_banks():
    from stages import synthetic as SY
    root = _tmp()
    saved = dict(os.environ)
    try:
        path, d = _small(root)
        spool = os.path.join(root, "spool")
        _fake_env(spool)
        # (a) the three banks through their jobs and records
        rc, out = _run_banks(path)
        assert rc == 0 and "PIPELINE DONE" in out, out
        cfg = C.load_config(path, synthetic_ok=True)
        for bank, n_rows, kind in (("sim", 24, "box"), ("R0", 24, None), ("R3", 24, None)):
            rec = json.load(open(os.path.join(SY.bank_dir(cfg, bank), "BANK_RECORD.json")))
            assert rec["n_rows"] == n_rows and rec["latent_kind"] == kind, rec
            assert rec["build"] == SY.build_of(cfg, bank)
        assert "sim_bank: BANK RECORDED:" in out and "real_bank_R3: BANK RECORDED:" in out, out
        recs = _records(spool)
        names = [r["name"] for r in recs]
        assert sorted(names) == sorted(["sb_sim_0", "sb_sim_1", "sb_sim_2", "sb_R0_0", "sb_R3_0",
                                        "sr_sim", "sr_R0", "sr_R3"]), names
        r0 = json.load(open(os.path.join(SY.bank_dir(cfg, "R0"), "shard_0000.json")))
        r3 = json.load(open(os.path.join(SY.bank_dir(cfg, "R3"), "shard_0000.json")))
        assert r0["gap_spec"]["pi"] == 0.0 and r3["gap_spec"]["pi"] == 0.3
        assert r0["latent_spec"]["seed"] == 22 and r3["latent_spec"]["seed"] == 33
        # (b) a re-run on the same state ends at once; on a fresh state it skips the three
        n0 = len(_records(spool))
        rc, out = _run_banks(path)
        assert rc == 0 and "PIPELINE DONE" in out, out
        shutil.rmtree(d["paths"]["work_root"])
        rc, out = _run_banks(path)
        assert rc == 0 and out.count(": skipped, artefact of record exists") == 3, out
        assert len(_records(spool)) == n0, "nothing submitted"
        # (c) shard 2 and the record deleted: only shard 2 rebuilt (a fresh work root)
        sdir = SY.bank_dir(cfg, "sim")
        shas = [_sha(os.path.join(sdir, "shard_%04d.npz" % k)) for k in (0, 1)]
        for f in ("shard_0002.npz", "shard_0002.json", "BANK_RECORD.json", "BANK_RECORD.json.sha256"):
            os.remove(os.path.join(sdir, f))
        shutil.rmtree(d["paths"]["work_root"])
        rc, out = _run_banks(path)
        assert rc == 0 and "sim_bank: 2 of 3 shard(s) already built" in out, out
        new = [r["name"] for r in _records(spool)][n0:]
        assert sorted(new) == ["sb_sim_2", "sr_sim"], new
        assert [_sha(os.path.join(sdir, "shard_%04d.npz" % k)) for k in (0, 1)] == shas
        # (d) a shard whose job always fails: max_resubmits 1, then the stop; --retry after the fix
        root2 = os.path.join(root, "second")
        os.makedirs(root2)
        path2, d2 = _small(root2)
        cfg2 = C.load_config(path2, synthetic_ok=True)
        sdir2 = SY.bank_dir(cfg2, "sim")
        os.makedirs(os.path.join(sdir2, "shard_0001.npz.tmp.npz"))    # the builder cannot write
        spool2 = os.path.join(root2, "spool")
        _fake_env(spool2, {"FAKE_PBS_FAIL_START": "sb_R0_0:1"})
        rc, out1 = _run_banks(path2)
        assert rc == 3 and "STOP: stage sim_bank failed: shard 1: 2 job(s) ended without the " \
                           "shard" in out1, out1
        assert "--retry sim_bank (the shards built are kept)" in out1, out1
        assert "IsADirectoryError" in out1 or "Is a directory" in out1, "the log's tail is quoted"
        names2 = [r["name"] for r in _records(spool2)]
        assert names2.count("sb_sim_1") == 2 and names2.count("sb_sim_0") == 1, names2
        os.rmdir(os.path.join(sdir2, "shard_0001.npz.tmp.npz"))
        _fake_env(spool2)
        n2 = len(_records(spool2))
        rc, out2 = _run_banks(path2, retry=("sim_bank",))
        assert rc == 0 and "PIPELINE DONE" in out2, out2
        new2 = [r["name"] for r in _records(spool2)][n2:]
        assert sorted(n for n in new2 if n.endswith("_sim") or "_sim_" in n) == ["sb_sim_1", "sr_sim"], \
            new2
        # the R0 job that PBS could not start (-2) was submitted again, not counted (D-111)
        names2 = [r["name"] for r in _records(spool2)]
        assert names2.count("sb_R0_0") == 2, names2
        assert "an environment fault, not the bank's (D-111)" in out1 + out2, out1 + out2
        st2 = json.load(open(os.path.join(d2["paths"]["work_root"], "state.json")))
        r0s = st2["stages"]["real_bank_R0"]
        assert r0s["status"] == "done" and r0s["sub"]["shards"]["0"]["resubmits"] == 0, r0s
        assert len(r0s["sub"]["shards"]["0"]["env_faults"]) == 1, r0s
        # (e) a folder of another build: refused, left as it was
        root3 = os.path.join(root, "third")
        os.makedirs(root3)
        path3, d3 = _small(root3)
        cfg3 = C.load_config(path3, synthetic_ok=True)
        argv = list(SY.build_of(cfg3, "R3")["argv"][0])
        argv[argv.index("--seed") + 1] = "9999"
        r3dir = SY.bank_dir(cfg3, "R3")
        _build_by_hand(r3dir, argv)
        before = dict((f, _sha(os.path.join(r3dir, f))) for f in os.listdir(r3dir))
        spool3 = os.path.join(root3, "spool")
        _fake_env(spool3)
        rc, out = _run_banks(path3)
        assert rc == 3 and "STOP: stage real_bank_R3 refused: REFUSE: %s holds a bank of another " \
                           "build" % r3dir in out, out
        assert "--seed 9999" in out, out
        assert dict((f, _sha(os.path.join(r3dir, f))) for f in os.listdir(r3dir)) == before
        assert not any(r["name"] == "sb_R3_0" for r in _records(spool3))
        # (f) a corrupt shard: the record job refuses, the stage stops
        shutil.rmtree(r3dir)
        sdir3 = SY.bank_dir(cfg3, "sim")
        for f in ("BANK_RECORD.json", "BANK_RECORD.json.sha256"):
            p_ = os.path.join(sdir3, f)
            if os.path.exists(p_):
                os.remove(p_)
        rc, out = _run_banks(path3, retry=("real_bank_R3",))
        assert rc == 0, out
        os.remove(os.path.join(sdir3, "BANK_RECORD.json"))
        os.remove(os.path.join(sdir3, "BANK_RECORD.json.sha256"))
        npz = os.path.join(sdir3, "shard_0000.npz")
        blob = bytearray(open(npz, "rb").read())
        for i in range(len(blob) // 2, len(blob) // 2 + 64):
            blob[i] ^= 0xFF
        with open(npz, "wb") as fh:
            fh.write(bytes(blob))
        shutil.rmtree(d3["paths"]["work_root"])
        rc, out = _run_banks(path3)
        assert rc == 3 and "STOP: stage sim_bank failed: the record tool refused the bank" in out, out
        assert "REFUSE: shard 0:" in out, out
    finally:
        os.environ.clear()
        os.environ.update(saved)
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the stages run tools; no convergence order")


def _peak(records):
    ev = []
    import pbs as PBS
    for r in records:
        argv = ["qsub"]
        for x in r.get("resources", []):
            argv += ["-l", x]
        w = PBS.cores_of_argv(argv + ["x"]) or 0
        end = r["finished"] if r["finished"] else float("inf")
        if r["submitted"] < end:
            ev += [(r["submitted"], 1, w), (end, 0, -w)]
    peak, now = 0, 0
    for _t, _k, w in sorted(ev):
        now += w
        peak = max(peak, now)
    return peak


def test_t17_end_to_end():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("the training environment is needed (%s)" % exc)
    root = _tmp()
    saved = dict(os.environ)
    try:
        path, d = _small(root, **{"synthetic.sim": {"seed": 11, "n_theta": 60, "n_windows": 1,
                                                    "traces_per_shard": 40}})
        spool = os.path.join(root, "spool")
        env = _fake_env(spool)
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", path,
                            "--poll-seconds", "0.5"], cwd=root, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=1800)
        out = r.stdout
        assert r.returncode == 0 and "PIPELINE DONE: every stage done or skipped" in out, out[-6000:]
        p = d["paths"]
        for q, camp in (("S-A1", "S-A1"), ("S-A2-R0", "S-A2")):
            sc = json.load(open(os.path.join(p["tune_root"], q, camp, "SEARCH_COMPLETE.json")))
            assert sc["campaign"] == camp and sc["ended_by"] == "rounds", sc
            assert sc["n_trials"]["ok"] == 3, sc["n_trials"]
            bf = json.load(open(os.path.join(p["runs_root"], q, "synth_best",
                                             "BEST_FIT_COMPLETE.json")))
            assert bf["ensemble"]["M"] == 2, bf
            ledger = sorted(glob.glob(os.path.join(p["tune_root"], q, camp, "trials", "*.json")))
            assert len(ledger) == 3, ledger
            arms = []
            for lp in ledger:
                rec = json.load(open(lp))
                assert rec["status"] == "ok" and "L_pseudo_real" in rec, rec
                real = rec["argv"][rec["argv"].index("--real-shards") + 1]
                assert real == os.path.join(p["banks_root"], "synth_real_R0", "shard_*.npz"), real
                arms.append(rec["arm"])
            if camp == "S-A1":
                assert set(arms) == {"A1"}, arms
            else:
                assert "A2" in arms, ("no S-A2 trial with the metric term on at gp_seed %d"
                                      % d["search"]["gp_seed"], arms)
        recs = _records(spool)
        assert _peak(recs) <= 4, "the jobs held %d cores at once, over the 4 the cap leaves" % _peak(recs)
        assert "search_S-A2-R0: SEARCH COMPLETE" in out and "best_fit_S-A1: BEST FIT COMPLETE" in out
    finally:
        os.environ.clear()
        os.environ.update(saved)
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------- block 18 (T18) ---

def test_t18_config():
    cfg = C.load_config(SYNTH, synthetic_ok=True)
    flags = dict((q["name"], q["alt_best_fit"]) for q in cfg.data["synthetic"]["searches"])
    assert flags == {"S-A1": False, "S-A2-R0": True, "S-A5-R0": True, "S-A2-R3": True,
                     "S-A5-R3": True}, flags
    assert C.ALT_ARMS == {"S-A2": ("A1", "A2"), "S-A5": ("A1", "A5")}, C.ALT_ARMS

    def problems(mutate):
        d = _committed()
        mutate(d)
        return C.validate(d)

    def refused(mutate, needle):
        p = problems(mutate)
        assert any(needle in x for x in p), (needle, p)

    q = lambda d, i: d["synthetic"]["searches"][i]                         # noqa: E731
    refused(lambda d: q(d, 0).update(alt_best_fit=True),
            "S-A1: alt_best_fit true needs a campaign of two arms (S-A2, S-A5), got S-A1")
    refused(lambda d: d["synthetic"]["searches"].append(
        {"name": "S-A25-R0", "campaign": "S-A25", "real": "R0", "alt_best_fit": True}),
        "S-A25-R0: alt_best_fit true needs a campaign of two arms")
    refused(lambda d: q(d, 1).update(alt_best_fit="yes"),
            "S-A2-R0: alt_best_fit must be true or false, got 'yes'")
    refused(lambda d: q(d, 1).update(alt_best_fit=1),
            "S-A2-R0: alt_best_fit must be true or false, got 1")
    refused(lambda d: q(d, 2).pop("alt_best_fit"), "must have exactly the keys")
    # an S-A25 search without the alternative is accepted, as before
    p = problems(lambda d: d["synthetic"]["searches"].append(
        {"name": "S-A25-R0", "campaign": "S-A25", "real": "R0", "alt_best_fit": False}))
    assert p == [], p


def test_t18_graph():
    import stages as S
    from stages import synthetic as SY
    cfg = C.load_config(SYNTH, synthetic_ok=True)
    graph = S.topological_order(S.default_graph(cfg))
    names = [s.name for s in graph]
    assert len(names) == 17, names
    by = dict((s.name, s) for s in graph)
    p = cfg.data["paths"]
    for q in ("S-A2-R0", "S-A5-R0", "S-A2-R3", "S-A5-R3"):
        st = by["best_fit_alt_" + q]
        assert isinstance(st, SY.RunAltBestFitStage), type(st)
        assert names.index("best_fit_alt_" + q) == names.index("best_fit_" + q) + 1, names
        assert tuple(st.deps) == ("search_" + q,), st.deps
        root = os.path.join(p["runs_root"], q, "synth_best_alt")
        done, why = st.is_done(cfg)
        art = os.path.join(root, "BEST_FIT_COMPLETE.json")
        assert why == (art if done else art + " absent"), why
        v = st._view(cfg)
        assert st._dirs(v)["root"] == root and st._dirs(v)["complete"] == os.path.join(
            root, "BEST_FIT_COMPLETE.json"), st._dirs(v)
        assert st._dirs(v)["search_complete"] == by["best_fit_" + q]._dirs(v)["search_complete"]
        assert by["best_fit_" + q]._dirs(v)["root"] == os.path.join(p["runs_root"], q, "synth_best")
        assert st.job_prefix == "ba" and by["best_fit_" + q].job_prefix == "bf"
        assert st._arms(v) == C.ALT_ARMS[v.data["search"]["campaign"]]
    assert "best_fit_alt_S-A1" not in by
    # S-A1's view has no other arm: the alternative refuses before any tuner call
    alt = SY.RunAltBestFitStage(dict(by["search_S-A1"].run))
    try:
        alt._arms(alt._view(cfg))
        raise AssertionError("an S-A1 view gave arms for the other arm's best fit")
    except RuntimeError as exc:
        assert "campaign S-A1 has no other arm" in str(exc), exc
    # S8's lines carry the stage's own name
    seen = []

    class Rec(object):
        def log(self, t):
            seen.append(t)

    r = by["best_fit_alt_S-A5-R3"]._rec(Rec())
    r.log("best_fit: member seed 2 done: L 1.0 (x)")
    r.log("best_fit_alt: not a prefix of S8")
    assert seen == ["best_fit_alt_S-A5-R3: member seed 2 done: L 1.0 (x)",
                    "best_fit_alt: not a prefix of S8"], seen


def test_t18_dry_run():
    _skip_where_run()
    root = _tmp()
    try:
        env = dict(os.environ)
        env["PIPELINE_SKIP_CONDA"] = "1"
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", SYNTH,
                            "--dry-run"], cwd=root, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        out = r.stdout
        assert r.returncode == 0 and "REFUSE" not in out, out[-4000:]
        assert out.count("qsub -N ba_synth_s<k>") == 4, out.count("qsub -N ba_synth_s<k>")
        assert out.count("qsub -N bf_synth_s<k>") == 5, out.count("qsub -N bf_synth_s<k>")
        assert out.count("the best fit of the other arm (D-126) in ") == 4, out
        for q, arms in (("S-A2-R0", "A1/A2"), ("S-A5-R3", "A1/A5")):
            assert "best_fit_alt_%s: would wait for search_%s" % (q, q) in out, (q, out[-3000:])
            assert ("the arm of %s that the finalist of /davinci-1/home/ldellamea/tune_joint_synth/"
                    "%s/%s/SEARCH_COMPLETE.json is not" % (arms, q, q[:4])) in out, (q, out[-3000:])
            assert ("/davinci-1/home/ldellamea/runs/synth/%s/synth_best_alt/finalists_<a>.json"
                    % q) in out, q
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_t18_end_to_end():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("the training environment is needed (%s)" % exc)
    root = _tmp()
    saved = dict(os.environ)
    try:
        search = {"name": "S-A2-R0", "campaign": "S-A2", "real": "R0", "alt_best_fit": False}
        path, d = _small(root, searches=[search],
                         **{"synthetic.sim": {"seed": 11, "n_theta": 60, "n_windows": 1,
                                              "traces_per_shard": 40}})
        spool = os.path.join(root, "spool")
        env = _fake_env(spool)

        def run(*extra):
            r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", path,
                                "--poll-seconds", "0.5"] + list(extra), cwd=root, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, timeout=1800)
            return r.returncode, r.stdout

        rc, out = run()
        assert rc == 0 and "PIPELINE DONE: every stage done or skipped" in out, out[-6000:]
        p = d["paths"]
        camp_dir = os.path.join(p["tune_root"], "S-A2-R0", "S-A2")
        best_dir = os.path.join(p["runs_root"], "S-A2-R0", "synth_best")
        alt_dir = os.path.join(p["runs_root"], "S-A2-R0", "synth_best_alt")
        assert not os.path.exists(alt_dir), "alt_best_fit false made the other arm's folder"
        before = dict((f, _sha(f)) for f in (os.path.join(camp_dir, "SEARCH_COMPLETE.json"),
                                             os.path.join(best_dir, "BEST_FIT_COMPLETE.json")))
        n_jobs = len(_records(spool))
        recs = [json.load(open(f)) for f in glob.glob(os.path.join(camp_dir, "trials", "*.json"))]
        arms = sorted(set(r["arm"] for r in recs if r["status"] == "ok"))
        assert arms == ["A1", "A2"], ("the fixture needs trials of both arms at gp_seed %d, got %s"
                                      % (d["search"]["gp_seed"], arms))
        # the configuration change: alt_best_fit turned on
        cfg_d = json.load(open(path))
        cfg_d["synthetic"]["searches"][0]["alt_best_fit"] = True
        with open(path, "w") as fh:
            json.dump(cfg_d, fh, indent=1, sort_keys=True)
        rc, out = run()
        assert rc == 2 and "REFUSE: the configuration changed since this run started" in out, out[-3000:]
        assert len(_records(spool)) == n_jobs, "a refused run submitted a job"
        rc, out = run("--accept-config-change")
        assert rc == 0 and "PIPELINE DONE: every stage done or skipped" in out, out[-6000:]
        assert "best_fit_alt_S-A2-R0: BEST FIT COMPLETE: M=2 members" in out, out[-3000:]
        new = _records(spool)[n_jobs:]
        assert sorted(r["name"] for r in new) == ["ba_synth_s1", "ba_synth_s2"], [r["name"] for r in new]
        for f, h in before.items():
            assert _sha(f) == h, "%s changed" % f
        # the oracle: the lowest-nll trial of the arm the search's finalist is not
        sc = json.load(open(os.path.join(camp_dir, "SEARCH_COMPLETE.json")))
        fin_arm = [r["arm"] for r in recs if r["trial_id"] == sc["finalist"]["trial_id"]][0]
        other = min((r for r in recs if r["status"] == "ok" and r["arm"] != fin_arm),
                    key=lambda r: r["nll"])
        fc = json.load(open(os.path.join(alt_dir, "finalist_config.json")))
        assert fc["trial_id"] == other["trial_id"] and fc["arm"] == other["arm"], (fc, other)
        assert fc["search_finalist"] == {"trial_id": sc["finalist"]["trial_id"], "arm": fin_arm}, fc
        bf = json.load(open(os.path.join(alt_dir, "BEST_FIT_COMPLETE.json")))
        assert bf["ensemble"]["M"] == 2 and bf["finalist"]["arm"] == other["arm"], bf["finalist"]
        main = json.load(open(os.path.join(best_dir, "BEST_FIT_COMPLETE.json")))
        assert bf["split_seed"] == main["split_seed"] and bf["member_seeds"] == main["member_seeds"]
        for k in (1, 2):
            mrec = sorted(glob.glob(os.path.join(alt_dir, "member_seed%d" % k, "*_seed%d.json" % k)))
            assert len(mrec) == 1, mrec
            m = json.load(open(mrec[0]))
            assert m["arm"] == other["arm"] and m["seed"] == k and m["split_seed"] == 1, (
                m["arm"], m["seed"], m["split_seed"])
        assert _peak(_records(spool)) <= 4, "over the 4 cores the cap leaves"
    finally:
        os.environ.clear()
        os.environ.update(saved)
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ main ---

CHECKS = [("test_t15_config", test_t15_config), ("test_t15_graph", test_t15_graph),
          ("test_t15_dry_run", test_t15_dry_run), ("test_t16_record", test_t16_record),
          ("test_t16_banks", test_t16_banks), ("test_convergence", test_convergence),
          ("test_t17_end_to_end", test_t17_end_to_end), ("test_t18_config", test_t18_config),
          ("test_t18_graph", test_t18_graph), ("test_t18_dry_run", test_t18_dry_run),
          ("test_t18_end_to_end", test_t18_end_to_end)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
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
