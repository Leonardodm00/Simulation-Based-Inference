"""Block 26 -- one search per arm, the term-only campaigns (SPEC.md sec. 5,
block 26; D-132). Spec-derived tests of the integration tester, run t1-D of
2026-11-13. Oracles O26.1-O26.11 of the report.

The references: the campaigns S-A2 and S-A5 as they stood at 0e62645 (before
blocks 25-26; the spec says the old four are unchanged and the new two are
their pins plus the term's switch), the committed synth.json at 0e62645 (the
search and best-fit values unchanged), and the spec's own lists.

    python3 -m pytest tests/spec/test_spec_block26_campaigns.py -v
"""

import glob
import importlib.util
import json
import os
import re
import shutil
import tempfile
import types

import numpy as np
import pytest

import spec_helpers as H

H.add_paths(H.PIPE, H.PTESTS, H.STAGE4, H.EVAL)

import config as C          # noqa: E402

NEW = {"S-A2-only": ("S-A2", "dsn_on", "A2"), "S-A5-only": ("S-A5", "rep_on", "A5")}
V4_SEARCHES = [("S-A1", "S-A1", "R0", False), ("S-A2-R0", "S-A2-only", "R0", False),
               ("S-A5-R0", "S-A5-only", "R0", False), ("S-A2-R3", "S-A2-only", "R3", False),
               ("S-A5-R3", "S-A5-only", "R3", False)]


def fresh(path, name):
    """A private copy of a module (joint_space keeps its campaigns in a
    module-level dict that apply_overrides rewrites)."""
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


def js_now(tag):
    return fresh(os.path.join(H.STAGE4, "joint_space.py"), "joint_space_" + tag)


def js_before():
    return H.module_at(H.BEFORE_B25_26, "hpc/joint/stage4/joint_space.py", "joint_space_0e62645",
                       [H.STAGE4])


def _strip_docs(x):
    if isinstance(x, dict):
        return dict((k, _strip_docs(v)) for k, v in x.items() if not k.startswith("_"))
    if isinstance(x, list):
        return [_strip_docs(v) for v in x]
    return x


# ------------------------------------------------------------------- checks ---

def test_o26_1_every_campaign_partitions_the_space():
    """O26.1: pinned and free disjoint, their union every axis."""
    JS = js_now("o261")
    for name in JS.CAMPAIGNS:
        c = JS.campaign(name)
        pinned, free = set(c.pinned), set(c.free())
        assert not pinned & free and pinned | free == set(JS.JOINT_KNOB_ORDER), name
    assert {"S-A2-only", "S-A5-only"} <= set(JS.CAMPAIGNS)


def test_o26_2_4_pins_relation_and_old_campaigns_unchanged():
    """O26.2: S-A2-only = S-A2's pins + dsn_on 1; S-A5-only = S-A5's pins +
    rep_on 1 (the term's own axes free). O26.4: S-A1, S-A2, S-A5, S-A25 as at
    0e62645."""
    JS, old = js_now("o262"), js_before()
    for n in ("S-A1", "S-A2", "S-A5", "S-A25"):
        assert JS.campaign(n).to_dict() == old.campaign(n).to_dict(), n
    for new, (base, switch, _arm) in NEW.items():
        want = dict(old.campaign(base).pinned)
        want[switch] = 1
        assert JS.campaign(new).pinned == want, (new, JS.campaign(new).pinned, want)
        assert set(JS.campaign(new).free()) == set(old.campaign(base).free()) - {switch}
    a2, a5 = JS.campaign("S-A2-only"), JS.campaign("S-A5-only")
    for axis in ("log10_lambda_dsn", "loss_type", "mining_strategy", "margin", "lambda_sep"):
        assert axis in a2.free() and axis in a5.pinned, axis
    for axis in ("log10_lambda_rep",):
        assert axis in a5.free() and axis in a2.pinned, axis


def _random_point(JS, spec, name, rng):
    pt = []
    for axis in JS.free_axes(name):
        r = spec.range_of(axis)
        if axis in JS._STR_AXES:
            pt.append(str(rng.choice(list(r))))
        elif axis in ("block_family", "head_fusion", "batch_size_npe", "head_pool_ops"):
            pt.append(int(rng.choice(list(r))))
        elif axis in JS._INT_AXES:
            pt.append(int(rng.integers(int(r[0]), int(r[1]) + 1)))
        else:
            pt.append(float(rng.uniform(float(r[0]), float(r[1]))))
    return pt


@pytest.mark.parametrize("seed", [26, 27])
def test_o26_3_every_configuration_maps_to_the_terms_arm(seed):
    """O26.3: 300 random points of each new campaign, and 40 of skopt's own
    proposals over its dimensions, all map to A2 (S-A2-only) or A5
    (S-A5-only); the term's switch is 1 and the other's 0 in every one."""
    JS = js_now("o263_%d" % seed)
    H.add_paths(H.STAGE4)
    import npe_tune_joint as T
    from skopt import Optimizer
    spec = JS.default_joint_space(p=10, embedding_dim=64, d_theta=10)
    rng = np.random.default_rng(seed)
    for name, (_base, switch, arm) in NEW.items():
        other = "rep_on" if switch == "dsn_on" else "dsn_on"
        pts = [_random_point(JS, spec, name, rng) for _ in range(300)]
        pts += Optimizer(JS.space_dimensions(spec, name), random_state=seed).ask(40)
        for pt in pts:
            cfg = JS.config_from_point(pt, name, spec=spec)
            assert int(cfg[switch]) == 1 and int(cfg[other]) == 0, (name, cfg)
            assert T.arm_for_config(cfg) == arm, (name, cfg)


def test_o26_5_overrides_act_on_the_new_campaigns_as_on_the_old():
    """O26.5: synth's space overrides applied: each new campaign's pins are
    still its base's plus the switch, and its free axes its base's minus the
    switch."""
    with open(H.SYNTH) as fh:
        ov = json.load(fh)["search"]["space_overrides"]
    ov = dict((k, v) for k, v in ov.items() if not k.startswith("_"))
    JS = js_now("o265")
    spec = JS.default_joint_space(p=10, embedding_dim=64, d_theta=10)
    JS.apply_overrides(spec, ov)
    for new, (base, switch, _arm) in NEW.items():
        want = dict(JS.campaign(base).pinned)
        want[switch] = 1
        assert JS.campaign(new).pinned == want, (new, JS.campaign(new).pinned, want)
        assert set(JS.campaign(new).free()) == set(JS.campaign(base).free()) - {switch}
        dims = [d.name for d in JS.space_dimensions(spec, new)]
        assert dims == [a for a in JS.free_axes(new)], dims


@pytest.mark.parametrize("campaign", ["S-A1", "S-A2", "S-A5", "S-A25", "S-A2-only", "S-A5-only"])
def test_o26_6_configuration_accepts_each_campaign(campaign):
    import test_smoke_synthetic as TS
    d = TS._committed()
    d["synthetic"]["searches"][1]["campaign"] = campaign
    assert not [p for p in C.validate(d) if "campaign" in p or "alt_best_fit" in p], C.validate(d)


@pytest.mark.parametrize("mutate,needle", [
    (lambda q: q.update(campaign="S-A2-ONLY"), "campaign"),
    (lambda q: q.update(campaign="S-A2-only", alt_best_fit=True), "alt_best_fit"),
    (lambda q: q.update(campaign="S-A5-only", alt_best_fit=True), "alt_best_fit")])
def test_o26_6_configuration_refusals(mutate, needle):
    """O26.6: an unknown campaign; alt_best_fit true on a term-only campaign."""
    import test_smoke_synthetic as TS
    d = TS._committed()
    mutate(d["synthetic"]["searches"][1])
    probs = [p for p in C.validate(d) if needle in p]
    assert len(probs) == 1, C.validate(d)


def test_o26_6_alt_arms_unchanged():
    assert C.ALT_ARMS == {"S-A2": ("A1", "A2"), "S-A5": ("A1", "A5")}


def test_o26_7_committed_v4_searches_and_values():
    """O26.7: v4's five searches in order, alt_best_fit false; every value
    but the label axes, the centre scale and the searches' campaigns and
    alt_best_fit as in 0e62645's file (D-123, D-125 unchanged)."""
    import subprocess
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    got = [(q["name"], q["campaign"], q["real"], q["alt_best_fit"]) for q in
           cfg.data["synthetic"]["searches"]]
    assert got == V4_SEARCHES, got
    old = json.loads(subprocess.run(["git", "-C", H.REPO, "show", "%s:hpc/pipeline/projects/synth/"
                                     "synth.json" % H.BEFORE_B25_26], stdout=subprocess.PIPE,
                                    check=True).stdout)
    new = _strip_docs(json.load(open(H.SYNTH)))
    old = _strip_docs(old)
    assert [(q["name"], q["real"]) for q in new["synthetic"]["searches"]] == \
        [(q["name"], q["real"]) for q in old["synthetic"]["searches"]]
    for d in (new, old):
        d["synthetic"]["builder"].pop("label_axes", None)
        d["synthetic"]["builder"].pop("centre_scale")
        d["synthetic"].pop("searches")
    assert new == old


def test_o26_8_graph_views_and_folders():
    """O26.8: 13 stages in order with their dependencies, no other arm's fit;
    each view's campaign, folders, banks and artefact path."""
    import stages as S
    from stages import common as CM
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    p = cfg.data["paths"]
    graph = S.topological_order(S.default_graph(cfg))
    names = [s.name for s in graph]
    want = ["sim_bank", "real_bank_R0", "real_bank_R3"]
    for q, _c, _r, _a in V4_SEARCHES:
        want += ["search_" + q, "best_fit_" + q]
    assert names == want, names
    by = dict((s.name, s) for s in graph)
    for q, camp, real, _a in V4_SEARCHES:
        assert set(by["search_" + q].deps) == {"sim_bank", "real_bank_" + real}, by["search_" + q].deps
        assert set(by["best_fit_" + q].deps) == {"search_" + q}
        v = by["search_" + q]._view(cfg)
        assert v.data["search"]["campaign"] == camp
        assert v.data["paths"]["tune_root"] == os.path.join(p["tune_root"], q)
        assert v.data["paths"]["runs_root"] == os.path.join(p["runs_root"], q)
        sim_g, real_g = CM.shard_globs(v)
        assert sim_g == os.path.join(p["banks_root"], "synth_sim", "shard_*.npz")
        assert real_g == os.path.join(p["banks_root"], "synth_real_" + real, "shard_*.npz")
        art = S.artefact_of("search", v)[1]
        assert art == os.path.join(p["tune_root"], q, camp, "SEARCH_COMPLETE.json"), art


def test_o26_9_committed_dry_run():
    """O26.9: the committed v4's dry run: exit 0, no REFUSE or STOP, the 13
    stages, five searches' trial lines and five best fits' member lines, no
    other arm's fit, the three bank commands with the label axes 0-7."""
    root = tempfile.mkdtemp(prefix="t1d_b26_dry_")
    try:
        env = dict(os.environ)
        env["PIPELINE_SKIP_CONDA"] = "1"
        rc, out = H.run([H.PYTHON, os.path.join(H.PIPE, "pipeline.py"), "run", "--config", H.SYNTH,
                         "--dry-run"], env, cwd=root, timeout=300)
        assert rc == 0, out
        assert not re.search(r"^(REFUSE|STOP)", out, re.M), out
        for q, _c, _r, _a in V4_SEARCHES:
            assert "| search_%s " % q in out and "| best_fit_%s " % q in out, q
        assert "best_fit_alt" not in out and "qsub -N ba_" not in out
        assert out.count("qsub -N t_<trial>") == 5 and out.count("qsub -N bf_synth_s<k>") == 5, out
        assert out.count("--label-axes 0,1,2,3,4,5,6,7") == 3, out.count("--label-axes 0,1,2,3,4,5,6,7")
        assert not os.path.exists("/davinci-1/home/ldellamea/pipeline/synth")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _training_env():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
        from pytorch_metric_learning.utils import loss_and_miner_utils  # noqa: F401
    except ImportError as exc:
        pytest.skip("the training environment is not installed here (%s)" % exc)


def test_o26_10_end_to_end_a5_only():
    """O26.10 (beyond the spec's T26c, which runs S-A2-only): a reduced
    project with one search on S-A5-only runs to PIPELINE DONE; every ledger
    trial is arm A5 with rep_on 1, and the best fit's members are A5."""
    _training_env()
    import test_smoke_synthetic as TS
    root = tempfile.mkdtemp(prefix="t1d_b26_e2e_")
    try:
        path, d = TS._small(root, searches=[
            {"name": "S-A5-R0", "campaign": "S-A5-only", "real": "R0", "alt_best_fit": False}])
        env = H.fake_env(root, extra={"FAKE_PBS_ENFORCE_WALLTIME": "0", "OMP_NUM_THREADS": "1",
                                      "PYTHONPATH": H.PIPE})
        rc, out = H.run([H.PYTHON, os.path.join(H.PIPE, "pipeline.py"), "run", "--config", path,
                         "--poll-seconds", "0.5"], env, cwd=root, timeout=1800)
        assert rc == 0 and "PIPELINE DONE" in out, out[-4000:]
        p = d["paths"]
        camp = os.path.join(p["tune_root"], "S-A5-R0", "S-A5-only")
        sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
        assert sc["campaign"] == "S-A5-only", sc
        trials = sorted(glob.glob(os.path.join(camp, "trials", "*.json")))
        assert trials
        for tp in trials:
            rec = json.load(open(tp))
            assert rec["status"] == "ok" and rec["arm"] == "A5", (tp, rec.get("arm"), rec.get("status"))
            conf = rec.get("config") or {}
            assert int(conf.get("rep_on", 1)) == 1 and int(conf.get("dsn_on", 0)) == 0, conf
        members = glob.glob(os.path.join(p["runs_root"], "S-A5-R0", "synth_best", "member_seed*",
                                         "*_seed*.json"))
        assert len(members) == 2, members
        for mp in members:
            assert json.load(open(mp))["arm"] == "A5", mp
    finally:
        rc, out = H.run(["qselect"], H.fake_env(root))
        for j in out.split():
            H.run(["qdel", j], H.fake_env(root))
        shutil.rmtree(root, ignore_errors=True)


def test_o26_evaluation_pairs_each_term_arm_under_v4():
    """The boundary with blocks 19-20 (sec. 2.5, E3; synth.json v4's own
    `_searches` note: "the evaluation pairs each term arm's ensemble with
    S-A1's"): with v4's five ensembles reported, the paired comparison of
    each term arm is made. Expected by the note; sec. 2.5 defines E3 within a
    search only."""
    _training_env()
    import ev_config
    import ev_metrics
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    ecfg = ev_config.load(H.SYNTH_EVAL)
    arm = {"S-A1": "A1", "S-A2-only": "A2", "S-A5-only": "A5"}
    reported = [types.SimpleNamespace(search=q, arm=arm[q["campaign"]],
                                      stem="%s_%s" % (q["name"], arm[q["campaign"]]),
                                      name="%s:%s" % (q["name"], arm[q["campaign"]]))
                for q in cfg.data["synthetic"]["searches"]]
    mdir = tempfile.mkdtemp(prefix="t1d_b26_ev_")
    try:
        lines = []
        files = ev_metrics.paired_all(cfg, ecfg, reported, mdir, lines.append)
    except Exception as exc:          # a pairing that reads the samples: it tried
        files = ["(tried: %s)" % exc]
    finally:
        shutil.rmtree(mdir, ignore_errors=True)
    assert files, "under v4 no search has two arms: paired_all makes no comparison, silently"


def test_o26_11_recorded_kind_campaigns():
    """O26.11: sec. 3.1 lists {S-A1, S-A2, S-A5, S-A25} for a recorded data
    set's search.campaign; block 26 adds the two names to the shared
    _CAMPAIGNS. Recorded as found (the spec is ambiguous after block 26)."""
    with open(os.path.join(H.PIPE, "projects", "giulia", "giulia.json")) as fh:
        d = json.load(fh)
    d["search"]["campaign"] = "S-A2-only"
    root = tempfile.mkdtemp(prefix="t1d_b26_rec_")
    try:
        path = os.path.join(root, "giulia.json")
        H.write(path, json.dumps(d))
        try:
            C.load_config(path)
            probs = []
        except C.ConfigError as exc:
            probs = [p for p in exc.problems if "campaign" in p]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    assert probs, "a recorded configuration accepts search.campaign S-A2-only (sec. 3.1 lists four)"
