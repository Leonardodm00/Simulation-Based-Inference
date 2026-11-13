"""Spec-derived tests of block 15 (the synthetic kind: configuration, graph
and run views; SPEC.md secs. 2.1, 3.1, block 15). Oracles O15.1-O15.9 of the
t1-B report.

Every configuration here is the committed synth.json (v4) with one change
made by the test; the expected outcome (accepted, or refused with the key
named) is sec. 3.1's synthetic table and its cross-checks.
"""
import copy
import json
import os
import subprocess

import pytest

import t1b_helpers as H

H.pipeline_path()
import config as C                     # noqa: E402  (code under test)
import stages as S                     # noqa: E402
from stages import common as CM        # noqa: E402
from stages import synthetic as SY     # noqa: E402


def _write(tmp_path, d, text=None):
    p = os.path.join(str(tmp_path), "synth.json")
    with open(p, "w") as fh:
        fh.write(text if text is not None else json.dumps(d, indent=1))
    return p


def _problems(tmp_path, d=None, text=None):
    """[] when the file loads (synthetic_ok), else the ConfigError's problems."""
    p = _write(tmp_path, d, text)
    try:
        C.load_config(p, synthetic_ok=True)
        return []
    except C.ConfigError as exc:
        return exc.problems


def _named(problems, key):
    return [p for p in problems if p.startswith(key + ":") or key in p]


def test_o15_1_committed_loads(tmp_path):
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    assert C.is_synthetic(cfg.data)
    assert cfg.sha256 == H.sha256_file(H.SYNTH)
    with pytest.raises(C.ConfigError) as e:
        C.load_config(H.SYNTH)
    assert "synthetic" in str(e.value)
    g = C.load_config(H.GIULIA)
    assert not C.is_synthetic(g.data)


def _committed():
    return H.committed_synth()


CASES = [
    # (id, mutation, the key the refusal must name)
    ("seed_S_R0", lambda d: d["synthetic"]["real"][0].update(seed=d["synthetic"]["sim"]["seed"]),
     "synthetic.real"),
    ("seed_R0_R3", lambda d: d["synthetic"]["real"][1].update(seed=d["synthetic"]["real"][0]["seed"]),
     "synthetic.real"),
    ("unknown_real", lambda d: d["synthetic"]["searches"][1].update(real="R9"), "synthetic.searches"),
    ("dup_search_name", lambda d: d["synthetic"]["searches"][2].update(
        name=d["synthetic"]["searches"][1]["name"]), "synthetic.searches"),
    ("same_campaign_real", lambda d: d["synthetic"]["searches"].append(
        dict(d["synthetic"]["searches"][1], name="twin")), "synthetic.searches"),
    ("dup_real_name", lambda d: d["synthetic"]["real"][1].update(name="R0"), "synthetic.real"),
    ("tps_not_multiple", lambda d: d["synthetic"]["sim"].update(traces_per_shard=2001),
     "synthetic.sim.traces_per_shard"),
    ("tps_not_dividing", lambda d: d["synthetic"]["sim"].update(traces_per_shard=14),
     "synthetic.sim.traces_per_shard"),
    ("one_well", lambda d: d["synthetic"]["builder"].update(wells_per_donor=1),
     "synthetic.builder.wells_per_donor"),
    ("one_class", lambda d: d["synthetic"]["builder"].update(n_classes=1),
     "synthetic.builder.n_classes"),
    ("scale_without_regular", lambda d: d["synthetic"]["builder"].update(class_centres="simplex"),
     "synthetic.builder.centre_scale"),
    ("pi_without_mode", lambda d: d["synthetic"]["real"][1].update(gap_modes=[]), "synthetic.real"),
    ("pi0_without_mode", lambda d: d["synthetic"]["real"][0].update(gap_modes=[]), "synthetic.real"),
    ("unknown_mode", lambda d: d["synthetic"]["real"][1].update(gap_modes=["jitter"]),
     "synthetic.real"),
    ("dup_mode", lambda d: d["synthetic"]["real"][1].update(gap_modes=["drift", "drift"]),
     "synthetic.real"),
    ("pi_above_1", lambda d: d["synthetic"]["real"][1].update(pi=1.5), "synthetic.real"),
    ("pi_below_0", lambda d: d["synthetic"]["real"][1].update(pi=-0.1), "synthetic.real"),
    ("pi_bool", lambda d: d["synthetic"]["real"][1].update(pi=True), "synthetic.real"),
    ("pi_string", lambda d: d["synthetic"]["real"][1].update(pi="0.3"), "synthetic.real"),
    ("real_name_hyphen", lambda d: d["synthetic"]["real"][1].update(name="R-3"), "synthetic.real"),
    ("search_name_space", lambda d: d["synthetic"]["searches"][0].update(name="S A1"),
     "synthetic.searches"),
    ("search_name_lead_dash", lambda d: d["synthetic"]["searches"][0].update(name="-SA1"),
     "synthetic.searches"),
    ("campaign_unknown", lambda d: d["synthetic"]["searches"][0].update(campaign="S-A9"),
     "synthetic.searches"),
    ("real_item_extra_key", lambda d: d["synthetic"]["real"][0].update(extra=1), "synthetic.real"),
    ("search_item_missing_key", lambda d: d["synthetic"]["searches"][0].pop("alt_best_fit"),
     "synthetic.searches"),
    ("kind_other", lambda d: d.update(kind="recorded"), "kind"),
    ("recorded_key_sim", lambda d: d.update(sim={"seed": 1}), "sim"),
    ("recorded_key_campaign", lambda d: d["search"].update(campaign="S-A1"), "search.campaign"),
    ("recorded_key_arm", lambda d: d["search"].update(arm="A1"), "search.arm"),
    ("recorded_key_sbix", lambda d: d["paths"].update(sbix_dir="/x"), "paths.sbix_dir"),
    ("missing_key", lambda d: d["synthetic"]["bank_job"].pop("queue"), "synthetic.bank_job.queue"),
    ("cap_below_bank_job", lambda d: d["synthetic"]["bank_job"].update(ncpus=200),
     "resources.max_cores"),
    ("seed_bool", lambda d: d["synthetic"]["sim"].update(seed=True), "synthetic.sim.seed"),
    ("n_neurons_bool", lambda d: d["synthetic"]["builder"].update(n_neurons=True),
     "synthetic.builder.n_neurons"),
    ("real_seed_bool", lambda d: d["synthetic"]["real"][0].update(seed=False), "synthetic.real"),
    ("dpc_zero", lambda d: d["synthetic"]["real"][0].update(donors_per_class=0), "synthetic.real"),
    ("provider_dsn", lambda d: d["synthetic"]["builder"].update(provider="dsn"),
     "synthetic.builder.provider"),
    ("centres_other", lambda d: d["synthetic"]["builder"].update(class_centres="hexagon"),
     "synthetic.builder.class_centres"),
    ("real_empty", lambda d: d["synthetic"].update(real=[]), "synthetic.real"),
    ("searches_empty", lambda d: d["synthetic"].update(searches=[]), "synthetic.searches"),
    ("alt_on_S_A1", lambda d: d["synthetic"]["searches"][0].update(alt_best_fit=True),
     "synthetic.searches"),
    ("alt_on_A2_only", lambda d: d["synthetic"]["searches"][1].update(alt_best_fit=True),
     "synthetic.searches"),
    ("alt_not_bool", lambda d: d["synthetic"]["searches"][1].update(alt_best_fit="yes"),
     "synthetic.searches"),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_o15_2_one_refusal_each(tmp_path, case):
    _id, mutate, key = case
    d = _committed()
    mutate(d)
    probs = _problems(tmp_path, d)
    assert probs, "accepted: %s" % _id
    assert _named(probs, key), probs


@pytest.mark.parametrize("field,text", [("fs", "NaN"), ("T_win", "Infinity"),
                                        ("tau_ov", "-Infinity"), ("centre_scale", "NaN")])
def test_o15_3_nonfinite_floats_refused(tmp_path, field, text):
    d = _committed()
    d["synthetic"]["builder"][field] = 12345.5
    raw = json.dumps(d, indent=1).replace("12345.5", text)
    probs = _problems(tmp_path, text=raw)
    assert _named(probs, "synthetic.builder." + field), probs


def test_o15_3_pi_nan_refused(tmp_path):
    d = _committed()
    d["synthetic"]["real"][1]["pi"] = 12345.5
    probs = _problems(tmp_path, text=json.dumps(d, indent=1).replace("12345.5", "NaN"))
    assert _named(probs, "synthetic.real"), probs


@pytest.mark.parametrize("where,name", [("real", "R3\n"), ("searches", "S-A1\n")])
def test_o15_3_name_with_trailing_newline_refused(tmp_path, where, name):
    """sec. 3.1: a real bank's name matches [A-Za-z0-9]+ and a search's
    [A-Za-z0-9][A-Za-z0-9_-]*; "R3\\n" matches neither (it becomes a folder
    name and a stage name)."""
    d = _committed()
    if where == "real":
        d["synthetic"]["real"][1]["name"] = name
        for q in d["synthetic"]["searches"]:
            if q["real"] == "R3":
                q["real"] = name
    else:
        d["synthetic"]["searches"][0]["name"] = name
    probs = _problems(tmp_path, d)
    assert _named(probs, "synthetic." + where), "accepted a name with a newline: %r" % name


def test_o15_2_cap_exactly_enough(tmp_path):
    d = _committed()
    need = d["driver"]["ncpus"] + max(d["search"]["trial"]["ncpus"], d["best_fit"]["ncpus"],
                                      d["synthetic"]["bank_job"]["ncpus"])
    d["resources"]["max_cores"] = need
    assert _problems(tmp_path, d) == []
    d["resources"]["max_cores"] = need - 1
    assert _named(_problems(tmp_path, d), "resources.max_cores")


def test_o15_2_three_problems_at_once(tmp_path):
    d = _committed()
    d["synthetic"]["builder"]["wells_per_donor"] = 1
    d["synthetic"]["real"][1]["gap_modes"] = ["jitter"]
    d["synthetic"]["bank_job"].pop("queue")
    probs = _problems(tmp_path, d)
    assert _named(probs, "synthetic.builder.wells_per_donor")
    assert _named(probs, "synthetic.real")
    assert _named(probs, "synthetic.bank_job.queue")


def test_o15_2_every_problem_at_once_seed_with_search_problem(tmp_path):
    """Validation reports every problem at once; the cross-checks run on the
    keys that passed their own checks. A seed shared by arm S and R0 (keys
    synthetic.sim.seed and synthetic.real, both valid on their own) next to
    an unrelated problem in synthetic.searches: both are expected."""
    d = _committed()
    d["synthetic"]["real"][0]["seed"] = d["synthetic"]["sim"]["seed"]
    d["synthetic"]["searches"][0]["campaign"] = "S-A9"
    probs = _problems(tmp_path, d)
    assert _named(probs, "synthetic.searches"), probs
    assert any("seed" in p and "synthetic.real" in p for p in probs), probs


# -------------------------------------------------------------------- graph ---

V4_ORDER = ["sim_bank", "real_bank_R0", "real_bank_R3",
            "search_S-A1", "best_fit_S-A1", "search_S-A2-R0", "best_fit_S-A2-R0",
            "search_S-A5-R0", "best_fit_S-A5-R0", "search_S-A2-R3", "best_fit_S-A2-R3",
            "search_S-A5-R3", "best_fit_S-A5-R3"]


def _deps_expected(d):
    out = {"sim_bank": (), }
    for r in d["synthetic"]["real"]:
        out["real_bank_" + r["name"]] = ()
    for q in d["synthetic"]["searches"]:
        out["search_" + q["name"]] = ("sim_bank", "real_bank_" + q["real"])
        out["best_fit_" + q["name"]] = ("search_" + q["name"],)
        if q.get("alt_best_fit"):
            out["best_fit_alt_" + q["name"]] = ("search_" + q["name"],)
    return out


def test_o15_4_committed_graph():
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    g = SY.synthetic_set_graph(cfg)
    assert [s.name for s in g] == V4_ORDER
    want = _deps_expected(cfg.data)
    for s in g:
        assert tuple(s.deps) == want[s.name], s.name
    assert [s.name for s in S.default_graph(cfg)] == V4_ORDER
    assert [s.name for s in S.topological_order(g)] == V4_ORDER


def test_o15_4_alt_best_fit_follows_its_best_fit(tmp_path):
    d = _committed()
    d["synthetic"]["searches"][1].update(campaign="S-A2", alt_best_fit=True)
    p = _write(tmp_path, d)
    cfg = C.load_config(p, synthetic_ok=True)
    names = [s.name for s in SY.synthetic_set_graph(cfg)]
    i = names.index("best_fit_S-A2-R0")
    assert names[i + 1] == "best_fit_alt_S-A2-R0"
    alt = [s for s in SY.synthetic_set_graph(cfg) if s.name == "best_fit_alt_S-A2-R0"][0]
    assert tuple(alt.deps) == ("search_S-A2-R0",)
    assert len(names) == 14


def test_o15_8_recorded_graph_unchanged():
    g = S.default_graph(C.load_config(H.GIULIA))
    assert [s.name for s in g] == ["real_extraction", "sim_redetection", "label_freeze",
                                   "bank_plan", "sim_bank", "real_bank", "search", "best_fit"]
    want = {"real_extraction": (), "sim_redetection": ("real_extraction",), "label_freeze": (),
            "bank_plan": ("real_extraction", "sim_redetection", "label_freeze"),
            "sim_bank": ("bank_plan",), "real_bank": ("bank_plan",),
            "search": ("sim_bank", "real_bank"), "best_fit": ("search",)}
    for s in g:
        assert tuple(s.deps) == want[s.name]


# ------------------------------------------------------------------- views ---

def test_o15_5_run_view():
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    before = copy.deepcopy(cfg.data)
    for q in cfg.data["synthetic"]["searches"]:
        v = SY.RunView(cfg, q)
        assert v.path == cfg.path and v.sha256 == cfg.sha256
        assert v.data["paths"]["tune_root"] == os.path.join(before["paths"]["tune_root"], q["name"])
        assert v.data["paths"]["runs_root"] == os.path.join(before["paths"]["runs_root"], q["name"])
        assert v.data["search"]["campaign"] == q["campaign"]
        assert v.data["synthetic"]["run"] == q
        for k in before["paths"]:
            if k not in ("tune_root", "runs_root"):
                assert v.data["paths"][k] == before["paths"][k]
        v.data["search"]["trial"]["ncpus"] = -1          # a deep copy: the base never moves
    assert cfg.data == before


def test_o15_5_view_idempotent():
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    q0, q1 = cfg.data["synthetic"]["searches"][:2]
    st = SY.RunSearchStage(q0)
    v = st._view(cfg)
    assert st._view(v) is v                                # a view is returned as it is
    assert st._view(cfg) is v                              # cached for the same configuration
    bf = SY.RunBestFitStage(q0)
    assert bf._view(v) is v
    with pytest.raises(ValueError):
        SY.RunSearchStage(q1)._view(v)                       # another run's view


def test_o15_6_shard_globs():
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    br, tag = cfg.data["paths"]["banks_root"], cfg.data["dataset"]
    for q in cfg.data["synthetic"]["searches"]:
        sim, real = CM.shard_globs(SY.RunView(cfg, q))
        assert sim == os.path.join(br, tag + "_sim", "shard_*.npz")
        assert real == os.path.join(br, "%s_real_%s" % (tag, q["real"]), "shard_*.npz")
    with pytest.raises(Exception):
        CM.shard_globs(cfg)
    g = C.load_config(H.GIULIA)
    gb, gt = g.data["paths"]["banks_root"], g.data["dataset"]
    sim, real = CM.shard_globs(g)
    assert sim == os.path.join(gb, gt + "_sim", "shard_*.npz")
    assert real == (None if g.data["search"]["campaign"] == "S-A1"
                    else os.path.join(gb, gt + "_real", "shard_*.npz"))


def test_o15_7_log_prefix():
    class Rec(object):
        def __init__(self):
            self.lines = []

        def log(self, t):
            self.lines.append(t)

        def fits(self, n):
            return True

    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    q = cfg.data["synthetic"]["searches"][0]
    for stage, old in ((SY.RunSearchStage(q), "search"), (SY.RunBestFitStage(q), "best_fit")):
        r = Rec()
        lg = stage._rec(r)
        lg.log(old + ": trial abc done")
        lg.log("other line")
        lg.log("  " + old + ": indented")
        assert r.lines[0] == stage.name + ": trial abc done"
        assert r.lines[1] == "other line"
        assert lg.fits(3) is True                       # everything else is the recorder's
    assert SY.RunSearchStage(q).name == "search_" + q["name"]
    assert SY.RunBestFitStage(q).name == "best_fit_" + q["name"]


def test_o15_4_is_done_on_absent_artefacts(tmp_path):
    path, d = H.reduced_synth(str(tmp_path))
    cfg = C.load_config(path, synthetic_ok=True)
    for s in SY.synthetic_set_graph(cfg):
        ok, why = s.is_done(cfg)
        assert ok is False and why, s.name


def test_o15_9_committed_dry_run():
    """sec. 6: `run_pipeline.sh --project synth --dry-run` -- every stage, a qsub
    line per kind of job, no REFUSE line; exit 0; nothing submitted (no qsub
    on PATH here: a submission would fail loudly)."""
    if os.path.isdir(C.load_config(H.SYNTH, synthetic_ok=True).data["paths"]["work_root"]):
        pytest.skip("the committed project's work root exists here")
    env = dict(os.environ, SKIP_CONDA="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(["bash", os.path.join(H.PIPE, "run_pipeline.sh"), "--project", "synth",
                        "--dry-run"], cwd=H.REPO, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    out = r.stdout
    assert r.returncode == 0, out[-3000:]
    assert "REFUSE" not in out
    for name in V4_ORDER:
        assert name in out, name
    for job in ("-N sb_sim_", "-N sr_sim", "-N sb_R0_", "-N sr_R0", "-N sb_R3_", "-N sr_R3",
                "-N bf_synth_s"):
        assert job in out, job
