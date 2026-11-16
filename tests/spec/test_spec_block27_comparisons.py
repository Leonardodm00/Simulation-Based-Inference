"""Spec-derived tests of block 27 (specs/SPEC.md sec. 2.5 "The comparisons",
block 27's Rule and its paired-file contract), written by the integration
tester from the spec alone.

Expected values come from the spec's text: the comparison lists are written
out by hand; Dbar, sigma_seed, the verdict and the keys are computed here
from hand-made samples with numpy; the interval is the library's
`bootstrap_paired.paired_bootstrap` called with the arguments the spec names.

Run: python3 -m pytest -q tests/spec/test_spec_block27_comparisons.py
Pure ASCII, LF only.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _spec_util as U  # noqa: E402

import ev_config  # noqa: E402
import ev_plan  # noqa: E402

S = U.search
FIELDS = ("name", "kind", "E", "E_ref", "search", "search_ref", "arm", "arm_ref")
WHY_S_A25 = "campaign S-A25: its arms are not defined against A1 (D-136)"

# sec. 2.5: "Under synth.json v4 they are S-A2-R0, S-A5-R0, S-A2-R3 and S-A5-R3,
# each against S-A1:A1, then S-A2-R0_vs_S-A5-R0 and S-A2-R3_vs_S-A5-R3."
V4_WANT = [
    ("S-A2-R0", "vs_A1", "S-A2-R0:A2", "S-A1:A1", "S-A2-R0", "S-A1", "A2", "A1"),
    ("S-A5-R0", "vs_A1", "S-A5-R0:A5", "S-A1:A1", "S-A5-R0", "S-A1", "A5", "A1"),
    ("S-A2-R3", "vs_A1", "S-A2-R3:A2", "S-A1:A1", "S-A2-R3", "S-A1", "A2", "A1"),
    ("S-A5-R3", "vs_A1", "S-A5-R3:A5", "S-A1:A1", "S-A5-R3", "S-A1", "A5", "A1"),
    ("S-A2-R0_vs_S-A5-R0", "A2_vs_A5", "S-A2-R0:A2", "S-A5-R0:A5", "S-A2-R0", "S-A5-R0", "A2", "A5"),
    ("S-A2-R3_vs_S-A5-R3", "A2_vs_A5", "S-A2-R3:A2", "S-A5-R3:A5", "S-A2-R3", "S-A5-R3", "A2", "A5"),
]


def _cfg(tmp_path, searches, **kw):
    cp, ep = U.write_project(str(tmp_path), searches=searches, **kw)
    return ev_plan.load_project(cp), ev_config.load(ep)


def _tuples(cs):
    return [tuple(getattr(c, f) for f in FIELDS) for c in cs]


# ------------------------------------------------------------------ O27.1, O27.2

def test_comparison_namedtuple_fields():
    assert ev_plan.Comparison._fields == FIELDS


def test_committed_v4_gives_the_six_comparisons_in_order():
    cfg = ev_plan.load_project(U.SYNTH)
    assert _tuples(ev_plan.comparisons(cfg)) == V4_WANT
    cs, unc = ev_plan.comparisons(cfg, with_uncompared=True)
    assert _tuples(cs) == V4_WANT
    assert list(unc) == []


# ------------------------------------------------------------------ O27.3 - O27.5

def test_v3_design_nesting_searches_take_their_own_A1_and_pair_A2_with_A5(tmp_path):
    # v3's layout (block 18): S-A2 / S-A5 nest A1; case (b) applies to any A2 / A5 term arm
    cfg, _ = _cfg(tmp_path, [S("S-A1", "S-A1", "R0"), S("S-A2-R0", "S-A2", "R0", True),
                             S("S-A5-R0", "S-A5", "R0", True), S("S-A2-R3", "S-A2", "R3", True),
                             S("S-A5-R3", "S-A5", "R3", True)])
    want = [(q, "vs_A1", "%s:%s" % (q, a), "%s:A1" % q, q, q, a, "A1")
            for q, a in (("S-A2-R0", "A2"), ("S-A5-R0", "A5"), ("S-A2-R3", "A2"), ("S-A5-R3", "A5"))]
    want += [("S-A2-R0_vs_S-A5-R0", "A2_vs_A5", "S-A2-R0:A2", "S-A5-R0:A5", "S-A2-R0", "S-A5-R0",
              "A2", "A5"),
             ("S-A2-R3_vs_S-A5-R3", "A2_vs_A5", "S-A2-R3:A2", "S-A5-R3:A5", "S-A2-R3", "S-A5-R3",
              "A2", "A5")]
    assert _tuples(ev_plan.comparisons(cfg)) == want


def test_fixture_v3_like_one_within_search_comparison(tmp_path):
    cfg, _ = _cfg(tmp_path, [S("S-A1", "S-A1", "R0"), S("S-A2-R0", "S-A2", "R0", True)])
    assert _tuples(ev_plan.comparisons(cfg)) == [
        ("S-A2-R0", "vs_A1", "S-A2-R0:A2", "S-A2-R0:A1", "S-A2-R0", "S-A2-R0", "A2", "A1")]


def test_term_only_without_S_A1_is_refused_naming_the_search(tmp_path):
    cfg, _ = _cfg(tmp_path, [S("T2", "S-A2-only", "R0")])
    with pytest.raises(ev_plan.RefuseError) as ei:
        ev_plan.comparisons(cfg)
    assert "T2" in str(ei.value)
    with pytest.raises(ev_plan.RefuseError):
        ev_plan.comparisons(cfg, with_uncompared=True)


def test_term_only_with_two_S_A1_is_refused_naming_them(tmp_path):
    cfg, _ = _cfg(tmp_path, [S("A1a", "S-A1", "R0"), S("A1b", "S-A1", "R3"),
                             S("T5", "S-A5-only", "R3")])
    with pytest.raises(ev_plan.RefuseError) as ei:
        ev_plan.comparisons(cfg)
    msg = str(ei.value)
    assert "T5" in msg and "A1a" in msg and "A1b" in msg, msg


def test_nesting_campaigns_never_refuse_on_the_S_A1_count(tmp_path):
    cfg, _ = _cfg(tmp_path / "zero", [S("N2", "S-A2", "R0")])
    assert _tuples(ev_plan.comparisons(cfg)) == [
        ("N2", "vs_A1", "N2:A2", "N2:A1", "N2", "N2", "A2", "A1")]
    cfg, _ = _cfg(tmp_path / "two", [S("A1a", "S-A1", "R0"), S("A1b", "S-A1", "R3"),
                                     S("N5", "S-A5", "R0")])
    assert _tuples(ev_plan.comparisons(cfg)) == [
        ("N5", "vs_A1", "N5:A5", "N5:A1", "N5", "N5", "A5", "A1")]


def test_S_A25_is_uncompared_with_its_why_and_S_A1_yields_nothing(tmp_path):
    cfg, _ = _cfg(tmp_path, [S("S-A1", "S-A1", "R0"), S("X25", "S-A25", "R0"),
                             S("T2", "S-A2-only", "R3")])
    cs, unc = ev_plan.comparisons(cfg, with_uncompared=True)
    assert _tuples(cs) == [("T2", "vs_A1", "T2:A2", "S-A1:A1", "T2", "S-A1", "A2", "A1")]
    assert [tuple(x) for x in unc] == [("X25", WHY_S_A25)]
    assert _tuples(ev_plan.comparisons(cfg)) == _tuples(cs)


def test_case_b_A2_is_E_whichever_comes_first(tmp_path):
    cfg, _ = _cfg(tmp_path, [S("T5", "S-A5-only", "R0"), S("S-A1", "S-A1", "R0"),
                             S("T2", "S-A2-only", "R0")])
    assert _tuples(ev_plan.comparisons(cfg)) == [
        ("T5", "vs_A1", "T5:A5", "S-A1:A1", "T5", "S-A1", "A5", "A1"),
        ("T2", "vs_A1", "T2:A2", "S-A1:A1", "T2", "S-A1", "A2", "A1"),
        ("T2_vs_T5", "A2_vs_A5", "T2:A2", "T5:A5", "T2", "T5", "A2", "A5")]


def test_case_b_every_pair_on_one_cohort_in_the_order_of_q_then_q2(tmp_path):
    # two A2 searches (nesting and term-only) and two A5 searches on R0; one more cohort
    cfg, _ = _cfg(tmp_path, [S("S-A1", "S-A1", "R0"), S("N2", "S-A2", "R0", True),
                             S("T2", "S-A2-only", "R0"), S("N5", "S-A5", "R0", True),
                             S("T5", "S-A5-only", "R0"), S("U5", "S-A5-only", "R3")])
    got = _tuples(ev_plan.comparisons(cfg))
    assert [g[0] for g in got] == ["N2", "T2", "N5", "T5", "U5", "N2_vs_N5", "N2_vs_T5",
                                   "T2_vs_N5", "T2_vs_T5"]
    assert got[1][3] == "S-A1:A1" and got[0][3] == "N2:A1" and got[2][3] == "N5:A1"
    for g in got[5:]:
        assert g[1] == "A2_vs_A5" and g[2].endswith(":A2") and g[3].endswith(":A5")
        assert g[6:] == ("A2", "A5")


def test_case_b_needs_one_cohort(tmp_path):
    cfg, _ = _cfg(tmp_path, [S("S-A1", "S-A1", "R0"), S("T2", "S-A2-only", "R0"),
                             S("T5", "S-A5-only", "R3")])
    assert [c.kind for c in ev_plan.comparisons(cfg)] == ["vs_A1", "vs_A1"]


# ------------------------------------------------------------------ O27.6, O27.7

def test_comparisons_reads_no_file_and_needs_no_numpy():
    code = r'''
import builtins, sys
class Block(object):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("numpy", "scipy", "torch", "sbi", "sklearn", "matplotlib"):
            raise ImportError("blocked by the test: " + name)
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import ev_plan
cfg = ev_plan.load_project(sys.argv[2])
real_open = builtins.open
def no_open(*a, **k):
    raise AssertionError("comparisons opened a file: %r" % (a,))
builtins.open = no_open
try:
    cs, unc = ev_plan.comparisons(cfg, with_uncompared=True)
finally:
    builtins.open = real_open
print("N=%d U=%d" % (len(cs), len(unc)))
'''
    r = subprocess.run([sys.executable, "-c", code, U.EVAL, U.SYNTH], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "N=6 U=0" in r.stdout


def test_report_inputs_carries_the_comparisons_in_order(tmp_path):
    cfg, ecfg = _cfg(tmp_path, None)          # v4's searches, nothing on disk
    ri = ev_plan.report_inputs(cfg, ecfg)
    assert ri["comparisons"] == [[w[0], w[2], w[3]] for w in V4_WANT]
    assert ri["samples"] == {}


# ------------------------------------------------------------------ paired_all (O27.8 - O27.12)

def _ds(donors, rows_per_donor, d_by_donor, off_E, off_R, seed=0, in_box_E=None, lqR_inf=()):
    """Two samples (E, E_ref) of one data set: per row a base NLL b_i, E's NLL b_i +
    d_{donor(i)} (+ a per-row jitter j_i, mean 0 within each donor), members shifted by
    off_E / off_R (the members' mean NLLs are therefore the ensemble's + off). Returns
    (arrays_E, arrays_R, expected d per row)."""
    rng = np.random.default_rng(seed)
    donor = np.repeat(np.asarray(donors, dtype=np.int64), rows_per_donor)
    n = donor.size
    rows = np.arange(1000, 1000 + n, dtype=np.int64)
    theta = rng.uniform(0.05, 0.95, size=(n, 10))
    b = rng.uniform(1.0, 3.0, size=n)
    jit = np.zeros(n)
    for x in np.unique(donor):
        k = np.flatnonzero(donor == x)
        if k.size > 1:
            v = rng.normal(0.0, 0.05, size=k.size)
            jit[k] = v - v.mean()
    d = np.array([d_by_donor[int(x)] for x in donor]) + jit
    lE, lR = b + d, b.copy()
    in_box = np.ones(n, dtype=bool) if in_box_E is None else np.asarray(in_box_E, dtype=bool)
    lqR = -lR
    for i in lqR_inf:
        lqR[i] = -np.inf
    aE = {"rows": rows, "theta_ref": theta, "donor": donor, "in_box": in_box, "logq_ref": -lE,
          "logq_ref_members": -(lE[:, None] + np.asarray(off_E)[None, :])}
    aR = {"rows": rows.copy(), "theta_ref": theta.copy(), "donor": donor.copy(),
          "in_box": np.ones(n, dtype=bool), "logq_ref": lqR,
          "logq_ref_members": -(lR[:, None] + np.asarray(off_R)[None, :])}
    return aE, aR, d


def _expected(aE, aR, ecfg, name, D):
    """E3 and G.2 from the spec, here: ok rows, d_i, donor means, Dbar (the mean over
    the donors), sigma_seed (SD over members, ddof 1, of each member's mean NLL on the
    ensemble's in-box finite rows; the larger of the two), the library's interval."""
    import bootstrap_paired as BP
    ok = aE["in_box"] & np.isfinite(aE["logq_ref"]) & np.isfinite(aR["logq_ref"])
    d = (-aE["logq_ref"][ok]) - (-aR["logq_ref"][ok])
    g = aE["donor"][ok]
    donors = np.unique(g)
    delta = np.array([d[g == x].mean() for x in donors])

    def msd(a):
        o = a["in_box"] & np.isfinite(a["logq_ref"])
        m = (-a["logq_ref_members"][o]).mean(axis=0)
        return float(np.std(m, ddof=1))
    seed = ev_plan.seed_of(ecfg.seed, "paired", name, D)
    res = BP.paired_bootstrap(d, groups=g, within="all", n_boot=int(ecfg.bootstrap_n),
                              level=ecfg.level, seed=int(seed))
    sig = max(msd(aE), msd(aR))
    dbar = float(delta.mean())
    lo, hi = float(res.ci[0]), float(res.ci[1])
    beyond = abs(dbar) > 2.0 * sig
    excl = lo > 0.0 or hi < 0.0
    return {"n_rows": int(ok.sum()), "n_donors": int(donors.size), "Dbar": dbar, "ci": [lo, hi],
            "sigma_seed": sig, "beyond": beyond, "excludes": excl, "resolved": beyond and excl,
            "seed": seed, "donors": donors, "delta": delta, "d_rowmean": float(d.mean())}


KEYS_BLOCK20 = {"schema", "dataset", "n_rows", "n_donors", "Dbar", "ci", "se", "level", "n_boot",
                "scheme", "sigma_seed", "two_sigma_seed", "beyond_2_sigma_seed", "ci_excludes_0",
                "resolved"}
KEYS_EVERY = {"comparison", "kind", "E", "E_ref", "search", "search_ref", "arm", "arm_ref",
              "sd_members_E", "sd_members_ref", "lower_nll_ensemble", "bootstrap_seed"}
KEYS_LEGACY = {"term_arm", "ensembles", "sd_members", "lower_nll"}
PAIRED_LINE = re.compile(r"^  paired (\S+) on (\S+): Dbar [+-]\d+\.\d{4} \[[+-]\d+\.\d{4}, "
                         r"[+-]\d+\.\d{4}\], 2 sigma_seed \d+\.\d{4} -> (resolved|not resolved)$")


@pytest.fixture
def v4_project(tmp_path):
    """v4's searches; complete best fits on disk for S-A1:A1, S-A2-R0:A2, S-A5-R0:A5,
    S-A2-R3:A2 and S-A5-R3:A5 (three members each)."""
    cfg, ecfg = _cfg(tmp_path, None, bootstrap_n=300)
    for q, a in (("S-A1", "A1"), ("S-A2-R0", "A2"), ("S-A5-R0", "A5"), ("S-A2-R3", "A2"),
                 ("S-A5-R3", "A5")):
        U.make_complete(cfg.data, q, a)
    return cfg, ecfg


def _scenarios():
    """Four data sets of one comparison with known verdicts (sigma_seed 0.3 from E's
    members at -0.3, 0, +0.3; E_ref's 0.1): resolved with E lower; resolved with E_ref
    lower; inside 2 sigma_seed; beyond 2 sigma_seed with an interval holding 0."""
    donors = [10, 11, 12, 13, 14, 15]
    offE, offR = [-0.3, 0.0, 0.3], [-0.1, 0.0, 0.1]
    sc = {}
    sc["sim"] = _ds(donors, 2, dict((x, -1.0 + 0.01 * i) for i, x in enumerate(donors)), offE, offR, 1)
    sc["R0"] = _ds(donors, 4, dict((x, +1.0 + 0.01 * i) for i, x in enumerate(donors)), offE, offR, 2)
    sc["Xa"] = _ds(donors, 2, dict((x, 0.5) for x in donors), offE, offR, 3)
    sc["Xb"] = _ds(donors, 2, dict(zip(donors, [4.0, -2.6, 4.0, -2.6, 4.0, -2.6])), offE, offR, 4)
    return sc


def test_paired_known_answers_and_contract(v4_project):
    import ev_metrics as EM
    cfg, ecfg = v4_project
    sc = _scenarios()
    # E's provenance order sim, R0, Xa, Xb; S-A1:A1's another order and one more data set
    U.write_sample(ecfg, "S-A2-R0_A2", [(D, sc[D][0]) for D in ("sim", "R0", "Xa", "Xb")])
    U.write_sample(ecfg, "S-A1_A1", [(D, sc[D][1]) for D in ("Xb", "R0", "sim", "Xa")]
                   + [("R3", sc["sim"][1])])
    # S-A5-R0:A5 against... it is E_ref of S-A2-R0_vs_S-A5-R0 and E of S-A5-R0 (vs S-A1:A1)
    U.write_sample(ecfg, "S-A5-R0_A5", [("sim", sc["sim"][1]), ("R0", sc["R0"][1])])
    # S-A2-R3:A2: current sample but not requested; S-A5-R3:A5: complete, never sampled
    U.write_sample(ecfg, "S-A2-R3_A2", [("sim", sc["sim"][0])])
    found = ev_plan.ensembles(cfg)
    reported = [e for e in found if e.name in ("S-A1:A1", "S-A2-R0:A2", "S-A5-R0:A5")]
    assert len(reported) == 3
    mdir = os.path.join(ecfg.out_root, "metrics")
    os.makedirs(mdir)
    lines = []
    files, made, left = EM.paired_all(cfg, ecfg, reported, mdir, lines.append)

    # what is made, in comparisons' order, on E's data sets shared with E_ref, in E's order
    assert [m["name"] for m in made] == ["S-A2-R0", "S-A5-R0", "S-A2-R0_vs_S-A5-R0"]
    assert made[0]["datasets"] == ["sim", "R0", "Xa", "Xb"]
    assert made[1]["datasets"] == ["sim", "R0"]
    assert made[2]["datasets"] == ["sim", "R0"]
    for m, w in zip(made, (V4_WANT[0], V4_WANT[1], V4_WANT[4])):
        assert tuple(m[f] for f in FIELDS) == w
        assert set(m["paired"]) == set(m["datasets"])
        for D, p in m["paired"].items():
            assert os.path.isabs(p) and p == os.path.join(os.path.abspath(mdir),
                                                          "paired_%s_%s.json" % (m["name"], D))
    want_files = []
    for m in made:
        for D in m["datasets"]:
            base = os.path.join(os.path.abspath(mdir), "paired_%s_%s" % (m["name"], D))
            want_files += [base + ".json", base + ".npz"]
    assert sorted(files) == sorted(want_files) and all(os.path.isfile(f) for f in files)

    # the left out, each named once with its why, E's first (contract amendment 1)
    sj = ev_plan.sample_json(ecfg, "S-A5-R3_A5")
    want_left = [
        ("S-A2-R3", "S-A2-R3:A2", "S-A1:A1", "S-A2-R3:A2 was not requested (--ensemble)"),
        ("S-A5-R3", "S-A5-R3:A5", "S-A1:A1",
         "S-A5-R3:A5 is not reported (not sampled (%s absent))" % sj),
        ("S-A2-R3_vs_S-A5-R3", "S-A2-R3:A2", "S-A5-R3:A5", "S-A2-R3:A2 was not requested (--ensemble)")]
    assert [(x["comparison"], x["E"], x["E_ref"], x["why"]) for x in left] == want_left
    assert all(set(x) == {"comparison", "E", "E_ref", "why"} for x in left)
    for c, _e, _r, why in want_left:
        assert lines.count("  comparison %s left out: %s" % (c, why)) == 1

    # every paired file against the spec's numbers
    n_paired_lines = 0
    for m in made:
        for D in m["datasets"]:
            base = os.path.join(mdir, "paired_%s_%s" % (m["name"], D))
            js = json.load(open(base + ".json"))
            if m["name"] == "S-A2-R0":
                aE, aR = sc[D][0], sc[D][1]
            elif m["name"] == "S-A5-R0":
                aE, aR = sc[D][1], sc[D][1]
            else:
                aE, aR = sc[D][0], sc[D][1]
            x = _expected(aE, aR, ecfg, m["name"], D)
            keys = KEYS_BLOCK20 | KEYS_EVERY | (KEYS_LEGACY if m["kind"] == "vs_A1" else set())
            assert set(js) == keys, (m["name"], D, sorted(set(js) ^ keys))
            assert js["schema"] == "eval_metrics/1" and js["dataset"] == D
            assert (js["comparison"], js["kind"], js["E"], js["E_ref"]) == (m["name"], m["kind"],
                                                                           m["E"], m["E_ref"])
            assert js["n_rows"] == x["n_rows"] and js["n_donors"] == x["n_donors"]
            # 1e-12 nats: float64 sums of a few dozen O(1) terms in another order
            npt.assert_allclose(js["Dbar"], x["Dbar"], rtol=0, atol=1e-12)
            npt.assert_allclose(js["sigma_seed"], x["sigma_seed"], rtol=0, atol=1e-12)
            npt.assert_allclose(js["two_sigma_seed"], 2 * x["sigma_seed"], rtol=0, atol=1e-12)
            npt.assert_allclose(js["ci"], x["ci"], rtol=0, atol=1e-12)
            assert js["bootstrap_seed"] == x["seed"]
            assert js["beyond_2_sigma_seed"] == x["beyond"]
            assert js["ci_excludes_0"] == x["excludes"]
            assert js["resolved"] == x["resolved"]
            lower = (m["E"] if x["Dbar"] < 0 else m["E_ref"]) if x["resolved"] else None
            assert js["lower_nll_ensemble"] == lower
            assert js["level"] == ecfg.level and js["n_boot"] == ecfg.bootstrap_n
            assert js["scheme"] == "all"
            if m["kind"] == "vs_A1":
                assert js["term_arm"] == m["arm"]
                assert js["ensembles"] == {"term": m["E"], "A1": m["E_ref"]}
                assert js["sd_members"] == {m["arm"]: js["sd_members_E"], "A1": js["sd_members_ref"]}
                want_l = (("term arm" if x["Dbar"] < 0 else "A1") if x["resolved"] else None)
                assert js["lower_nll"] == want_l
            z = np.load(base + ".npz")
            assert set(z.files) == {"donor_ids", "delta_d", "se_d"}
            assert z["donor_ids"].dtype == np.int64 and z["delta_d"].dtype == np.float64
            assert z["se_d"].dtype == np.float64
            assert z["donor_ids"].shape == z["delta_d"].shape == z["se_d"].shape == (x["n_donors"],)
            npt.assert_array_equal(z["donor_ids"], x["donors"])
            npt.assert_allclose(z["delta_d"], x["delta"], rtol=0, atol=1e-12)
            n_paired_lines += 1
    # the four verdicts the scenarios were built for (comparison S-A2-R0)
    v = dict((D, json.load(open(os.path.join(mdir, "paired_S-A2-R0_%s.json" % D))))
             for D in ("sim", "R0", "Xa", "Xb"))
    assert v["sim"]["resolved"] and v["sim"]["lower_nll_ensemble"] == "S-A2-R0:A2"
    assert v["sim"]["lower_nll"] == "term arm"
    assert v["R0"]["resolved"] and v["R0"]["lower_nll_ensemble"] == "S-A1:A1"
    assert v["R0"]["lower_nll"] == "A1"
    assert not v["Xa"]["beyond_2_sigma_seed"] and v["Xa"]["ci_excludes_0"] and not v["Xa"]["resolved"]
    assert v["Xa"]["lower_nll_ensemble"] is None and v["Xa"]["lower_nll"] is None
    assert v["Xb"]["beyond_2_sigma_seed"] and not v["Xb"]["ci_excludes_0"] and not v["Xb"]["resolved"]
    # the log: one paired line per file pair, in block 20's format
    got = [l for l in lines if l.startswith("  paired ")]
    assert len(got) == n_paired_lines and all(PAIRED_LINE.match(l) for l in got), got


def test_A2_vs_A5_seed_is_the_comparisons_not_Es_search(v4_project):
    import ev_metrics as EM
    import bootstrap_paired as BP
    cfg, ecfg = v4_project
    donors = list(range(20, 32))
    rng = np.random.default_rng(7)
    aE, aR, _ = _ds(donors, 2, dict((x, float(rng.normal(0.1, 1.0))) for x in donors),
                    [-0.2, 0.0, 0.2], [-0.2, 0.0, 0.2], 9)
    U.write_sample(ecfg, "S-A2-R0_A2", [("sim", aE)])
    U.write_sample(ecfg, "S-A5-R0_A5", [("sim", aR)])
    reported = [e for e in ev_plan.ensembles(cfg) if e.name in ("S-A2-R0:A2", "S-A5-R0:A5")]
    mdir = os.path.join(ecfg.out_root, "metrics")
    os.makedirs(mdir)
    EM.paired_all(cfg, ecfg, reported, mdir, lambda t: None)
    js = json.load(open(os.path.join(mdir, "paired_S-A2-R0_vs_S-A5-R0_sim.json")))
    s_name = ev_plan.seed_of(ecfg.seed, "paired", "S-A2-R0_vs_S-A5-R0", "sim")
    s_search = ev_plan.seed_of(ecfg.seed, "paired", "S-A2-R0", "sim")
    assert s_name != s_search
    assert js["bootstrap_seed"] == s_name
    d = (-aE["logq_ref"]) - (-aR["logq_ref"])
    for s, same in ((s_name, True), (s_search, False)):
        r = BP.paired_bootstrap(d, groups=aE["donor"], within="all", n_boot=ecfg.bootstrap_n,
                                level=ecfg.level, seed=s)
        assert np.allclose(js["ci"], r.ci, rtol=0, atol=1e-12) == same
    assert set(js) == KEYS_BLOCK20 | KEYS_EVERY          # no legacy key for A2_vs_A5


def test_ok_rows_use_Es_in_box_and_both_finite_densities(v4_project):
    import ev_metrics as EM
    cfg, ecfg = v4_project
    donors = [1, 2, 3, 4, 5]
    in_box = np.ones(10, dtype=bool)
    in_box[0:2] = False                              # donor 1 out of E's box: left out
    aE, aR, d = _ds(donors, 2, {1: 9.0, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8}, [0, 0.1], [0, 0.1], 5,
                    in_box_E=in_box, lqR_inf=(2, 3))  # donor 2's E_ref density -inf: left out
    aR["in_box"][4:6] = False                        # E_ref's in_box is not read (donor 3 stays)
    U.write_sample(ecfg, "S-A2-R0_A2", [("sim", aE)])
    U.write_sample(ecfg, "S-A1_A1", [("sim", aR)])
    reported = [e for e in ev_plan.ensembles(cfg) if e.name in ("S-A2-R0:A2", "S-A1:A1")]
    mdir = os.path.join(ecfg.out_root, "metrics")
    os.makedirs(mdir)
    EM.paired_all(cfg, ecfg, reported, mdir, lambda t: None)
    js = json.load(open(os.path.join(mdir, "paired_S-A2-R0_sim.json")))
    z = np.load(os.path.join(mdir, "paired_S-A2-R0_sim.npz"))
    assert js["n_rows"] == 6 and js["n_donors"] == 3
    npt.assert_array_equal(z["donor_ids"], [3, 4, 5])
    npt.assert_allclose(js["Dbar"], np.mean([0.4, 0.6, 0.8]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("what", ["rows", "theta_ref", "donor", "length"])
def test_rows_that_differ_are_refused_naming_both_and_D(v4_project, what):
    import ev_metrics as EM
    cfg, ecfg = v4_project
    aE, aR, _ = _ds([1, 2, 3], 2, {1: 0.1, 2: 0.2, 3: 0.3}, [0, 0.1], [0, 0.1], 11)
    if what == "rows":
        aR["rows"][3] += 1
    elif what == "theta_ref":
        aR["theta_ref"][2, 4] += 1e-9
    elif what == "donor":
        aR["donor"][5] = 99
    else:
        aR = dict((k, v[:-1]) for k, v in aR.items())
    U.write_sample(ecfg, "S-A2-R0_A2", [("sim", aE)])
    U.write_sample(ecfg, "S-A1_A1", [("sim", aR)])
    reported = [e for e in ev_plan.ensembles(cfg) if e.name in ("S-A2-R0:A2", "S-A1:A1")]
    mdir = os.path.join(ecfg.out_root, "metrics")
    os.makedirs(mdir)
    with pytest.raises(ev_plan.RefuseError) as ei:
        EM.paired_all(cfg, ecfg, reported, mdir, lambda t: None)
    assert str(ei.value) == "S-A2-R0:A2 and S-A1:A1 do not share the rows of sim"


def test_left_out_not_complete_other_values_and_nothing_shared(tmp_path):
    import ev_metrics as EM
    cfg, ecfg = _cfg(tmp_path, None, bootstrap_n=200)
    # S-A1:A1 not complete (no folder); S-A2-R0:A2 and S-A5-R0:A5 complete; S-A5-R0:A5's
    # sample of other values; S-A2-R3:A2 and S-A5-R3:A5 complete, sampled on no common set
    for q, a in (("S-A2-R0", "A2"), ("S-A5-R0", "A5"), ("S-A2-R3", "A2"), ("S-A5-R3", "A5")):
        U.make_complete(cfg.data, q, a)
    aE, aR, _ = _ds([1, 2, 3], 2, {1: 0.1, 2: 0.2, 3: 0.3}, [0, 0.1], [0, 0.1], 13)
    U.write_sample(ecfg, "S-A2-R0_A2", [("sim", aE)])
    other = dict(ecfg.sample_params(), n_post=7)
    U.write_sample(ecfg, "S-A5-R0_A5", [("sim", aR)], params=other)
    U.write_sample(ecfg, "S-A2-R3_A2", [("R3", aE)])
    U.write_sample(ecfg, "S-A5-R3_A5", [("R3dag", aR)])
    reported = [e for e in ev_plan.ensembles(cfg) if e.name in ("S-A2-R0:A2", "S-A2-R3:A2",
                                                                 "S-A5-R3:A5")]
    mdir = os.path.join(ecfg.out_root, "metrics")
    os.makedirs(mdir)
    lines = []
    files, made, left = EM.paired_all(cfg, ecfg, reported, mdir, lines.append)
    assert files == [] and made == []
    why = dict((x["comparison"], x["why"]) for x in left)
    assert [x["comparison"] for x in left] == [w[0] for w in V4_WANT]
    assert why["S-A2-R0"] == "S-A1:A1 is not reported (not complete)"
    assert why["S-A5-R0"].startswith("S-A5-R0:A5 is not reported (sampled with other values: ")
    assert why["S-A2-R3"] == "S-A1:A1 is not reported (not complete)"
    assert why["S-A5-R3"] == "S-A1:A1 is not reported (not complete)"
    assert why["S-A2-R0_vs_S-A5-R0"].startswith("S-A5-R0:A5 is not reported (sampled with other")
    assert why["S-A2-R3_vs_S-A5-R3"] == "no data set is sampled for both"
    assert len([l for l in lines if " left out: " in l]) == 6


def test_Dbar_is_E3s_mean_over_donors_when_donors_have_unequal_rows(v4_project):
    """E3 (sec. 2.5): Dbar(E, E') is the mean of Delta_d over the donors. Donors with
    1, 2 and 6 ok rows: the mean over donors and the mean over rows differ."""
    import ev_metrics as EM
    cfg, ecfg = v4_project
    rows_per = [1, 2, 6]
    donor = np.repeat([1, 2, 3], rows_per)
    n = donor.size
    aE, aR, _ = _ds([1], n, {1: 0.0}, [0.0, 0.05], [0.0, 0.05], 17)
    aE["donor"] = donor.copy()
    aR["donor"] = donor.copy()
    dd = {1: 1.0, 2: 0.0, 3: -0.5}
    lR = -aR["logq_ref"]
    aE["logq_ref"] = -(lR + np.array([dd[int(x)] for x in donor]))
    U.write_sample(ecfg, "S-A2-R0_A2", [("sim", aE)])
    U.write_sample(ecfg, "S-A1_A1", [("sim", aR)])
    reported = [e for e in ev_plan.ensembles(cfg) if e.name in ("S-A2-R0:A2", "S-A1:A1")]
    mdir = os.path.join(ecfg.out_root, "metrics")
    os.makedirs(mdir)
    EM.paired_all(cfg, ecfg, reported, mdir, lambda t: None)
    js = json.load(open(os.path.join(mdir, "paired_S-A2-R0_sim.json")))
    z = np.load(os.path.join(mdir, "paired_S-A2-R0_sim.npz"))
    donor_mean = float(np.mean([1.0, 0.0, -0.5]))              # 0.1667: E3's Dbar
    row_mean = float((1 * 1.0 + 2 * 0.0 + 6 * -0.5) / 9.0)     # -0.2222
    npt.assert_allclose(z["delta_d"], [1.0, 0.0, -0.5], rtol=0, atol=1e-12)
    assert abs(donor_mean - row_mean) > 0.3
    npt.assert_allclose(js["Dbar"], donor_mean, rtol=0, atol=1e-12,
                        err_msg="Dbar is the mean over rows (%r), not E3's mean over donors (%r)"
                        % (row_mean, donor_mean))
