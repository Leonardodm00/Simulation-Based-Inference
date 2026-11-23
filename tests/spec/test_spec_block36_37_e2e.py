"""End-to-end spec tests of blocks 36 and 37 on the recorded project of
rec_fixture, through `ev_rec_report.py` (the report job's entry). Integration
tester, 2026-11-23, incremental run on dc096e5.

Needs REC_FIXTURE and the samples of test_spec_block35_sample.py under
REC_SPEC_OUT/runA and runB; the report is produced here by the CLI when
absent. Every expected value is recomputed from the sample files with numpy
and scipy, independently of block 36's functions, or read from the spec.
"""

import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL = os.path.join(REPO, "hpc", "pipeline", "evaluation")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

FIX = os.environ.get("REC_FIXTURE", "").strip()
OUT = os.environ.get("REC_SPEC_OUT", "/tmp/tst/e2e")
pytestmark = pytest.mark.skipif(not (FIX and os.path.isdir(FIX)), reason="REC_FIXTURE not set")
BETAS = np.round(np.linspace(0.0, 1.0, 101), 10)


def _load(path):
    with np.load(path) as z:
        return dict((k, z[k]) for k in z.files)


def _report(run):
    from test_spec_block35_sample import _ensure, _ecfg
    v, ec = _ensure(run, "synth_rec.json", "rec_eval.json")
    if not os.path.isfile(os.path.join(OUT, run, "REC_REPORT_COMPLETE.json")):
        r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_report.py"), "--config",
                            os.path.join(FIX, "synth_rec.json"), "--eval-config",
                            _ecfg(run, "rec_eval.json"), "--ncpus", "1"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    return v, ec


def test_sim_metrics_recomputed_from_samples():
    v, ec = _report("runA")
    for e in v.ensembles:
        z = _load(os.path.join(OUT, "runA", "samples", "%s_sim.npz" % e.stem))
        js = json.load(open(os.path.join(OUT, "runA", "metrics", "%s_sim.json" % e.stem)))
        a = _load(os.path.join(OUT, "runA", "metrics", "%s_sim.npz" % e.stem))
        ib, cal, didx = z["in_box"], z["calib"], z["draw_idx"]
        l = -z["logq_ref"]
        # the NLL over every row in the box (D-163 (1)), 1e-12: one mean of 16 values
        assert abs(js["nll"]["mean"] - np.mean(l[ib])) < 1e-12
        # gamma on the draw rows: the share of draws with a STRICTLY higher density (sec. 2.5)
        g = np.full(len(l), np.nan)
        g[didx] = np.mean(z["logq_draws"] > z["logq_ref"][didx][:, None], axis=1)
        assert_allclose(a["gamma"], g, rtol=0, atol=0, equal_nan=True)
        s_cal = cal & ib
        assert js["n_cal"] == int(s_cal.sum())
        cov = np.array([np.mean(g[s_cal] <= b) for b in BETAS])
        assert_allclose(a["coverage"], cov, atol=0)
        # the edge rows (R1) and the split over S_cal
        th = z["theta_ref"]
        edge = np.min(np.minimum(th, 1 - th), axis=1) < ec.edge_eps
        assert_array_equal(a["edge"], edge)
        assert js["edge"]["n_cal"] == int(s_cal.sum())
        assert js["edge"]["n_edge"] == int((edge & s_cal).sum())
        p = th.shape[1]
        assert abs(js["edge"]["pi_edge_expected"] - (1 - (1 - 2 * ec.edge_eps) ** p)) < 1e-15
        assert js["edge"]["nll_int_n"] == int((ib & ~edge).sum())
        # the recovery (sec. 2.5): mean and SD of the draws on the draw rows, NaN elsewhere
        assert_allclose(a["mhat"][didx], z["draws"].astype(np.float64).mean(1), rtol=1e-6)
        assert np.all(np.isnan(a["mhat"][np.setdiff1d(np.arange(len(l)), didx)]))
        # the contraction on D_sim: 1 - mean_cal sigma^2 / (1/12) (block 20's)
        sig = a["sigma"][cal].astype(np.float64)
        assert_allclose(js["recovery"]["contraction"], 1 - np.mean(sig ** 2, 0) * 12,
                        rtol=1e-5)
        # the axis names are the bank's (block 36, R36f)
        names = json.load(open(os.path.join(OUT, "runA", "samples",
                                            "%s_sample.json" % e.stem)))["axes"]["names"]
        assert js["recovery"]["axes"] == names and js["axes"]["names"] == names


def test_gate_metrics_from_definition():
    """The check rows and the bank's windows: T and p from the sample's
    embeddings and roles, by test_spec_block34_gate's loop oracle (R5)-(R8)."""
    from test_spec_block34_gate import gate_oracle
    from scipy import stats
    v, ec = _report("runA")
    alpha = ec.gate["alpha"]
    fac = list(ec.gate["factors"])
    for e in v.ensembles:
        z = _load(os.path.join(OUT, "runA", "samples", "%s_sim.npz" % e.stem))
        a = _load(os.path.join(OUT, "runA", "metrics", "%s_sim.npz" % e.stem))
        js = json.load(open(os.path.join(OUT, "runA", "metrics", "%s_sim.json" % e.stem)))
        pos = dict((int(i), j) for j, i in enumerate(z["z_idx"]))
        emb = lambda role: z["z"][:, [pos[i] for i in np.flatnonzero(z["gate_role"] == role)], :].astype(np.float64)  # noqa
        ref, rank, null, chk = emb(1), emb(2), emb(3), emb(4)
        o = gate_oracle(ref, rank, null, chk, fac)
        assert_allclose(a["gate_T_null"], o["T_null"], atol=1e-12)
        assert_allclose(a["gate_T_check"], o["T"], atol=1e-12)
        assert_allclose(a["gate_p_check"], o["p"], atol=1e-12)
        N = null.shape[1]
        k = int(np.ceil(alpha * (N + 1) - 1e-9)) - 1
        assert js["gate"]["k_gate"] == k and js["gate"]["n_null"] == N
        b = stats.beta(k, N + 1 - k)
        assert abs(js["gate"]["size_lo"] - b.ppf((1 - ec.level) / 2)) < 1e-12
        assert abs(js["gate"]["size_hi"] - b.ppf((1 + ec.level) / 2)) < 1e-12
        nf = int(np.sum(o["p"] < alpha))
        assert js["gate"]["check"]["n_flagged"] == nf
        ci = stats.binomtest(nf, chk.shape[1]).proportion_ci(ec.level, method="exact")
        assert_allclose(js["gate"]["check"]["ci"], [ci.low, ci.high], rtol=1e-12)
        # the bank
        zr = _load(os.path.join(OUT, "runA", "samples", "%s_real.npz" % e.stem))
        ar = _load(os.path.join(OUT, "runA", "metrics", "%s_real.npz" % e.stem))
        jr = json.load(open(os.path.join(OUT, "runA", "metrics", "%s_real.json" % e.stem)))
        ob = gate_oracle(ref, rank, null, zr["z"].astype(np.float64), fac)
        assert_allclose(ar["gate_T"], ob["T"], atol=1e-12)
        assert_allclose(ar["gate_p"], ob["p"], atol=1e-12)
        flag = ob["p"] < alpha
        assert_array_equal(ar["gate_flag"], flag)
        assert jr["gate"]["n_flagged"] == int(flag.sum())
        for c in jr["gate"]["per_class"]:
            sel = zr["cls"] == c["cls"]
            assert c["n"] == int(sel.sum()) and c["n_flagged"] == int(flag[sel].sum())
        # the wells (R4): pooled draws of each well's windows, equal weights
        d = zr["draws"].astype(np.float64)
        for gi, w in enumerate(ar["well_ids"]):
            pooled = d[zr["well"] == w].reshape(-1, d.shape[2])
            assert_allclose(ar["well_mean"][gi], pooled.mean(0), rtol=1e-12)
            assert_allclose(ar["well_sd"][gi], pooled.std(0, ddof=1), rtol=1e-12)
            assert_allclose(ar["well_q50"][gi], np.quantile(pooled, 0.5, axis=0), rtol=1e-12)
        # (R3) per window
        assert_allclose(ar["contraction"], 1 - 12 * ar["sigma"].astype(np.float64) ** 2,
                        atol=1e-5)
        # the example well: median of the wells' mean T per class (lower middle when even)
        ex = dict((x["cls"], x["well"]) for x in jr["example_wells"])
        for c in np.unique(zr["cls"]):
            ws = np.unique(zr["well"][zr["cls"] == c])
            mt = sorted((float(np.mean(ob["T"][zr["well"] == w])), int(w)) for w in ws)
            assert ex[int(c)] == mt[(len(mt) - 1) // 2][1], (c, mt, ex)


def test_paired_from_samples():
    v, ec = _report("runA")
    idx = json.load(open(os.path.join(OUT, "runA", "metrics", "METRICS.json")))
    import _evalpath  # noqa: F401
    import ev_rec_plan as RP
    made, _ = RP.comparisons(v)
    assert [c["name"] for c in idx["comparisons"]] == [c.name for c in made] == ["S-A2-R0"]
    by = dict((e.name, e) for e in v.ensembles)
    for c in idx["comparisons"]:
        zE = _load(os.path.join(OUT, "runA", "samples", "%s_sim.npz" % by[c["E"]].stem))
        zR = _load(os.path.join(OUT, "runA", "samples", "%s_sim.npz" % by[c["E_ref"]].stem))
        js = json.load(open(c["paired"]["sim"]))
        ok = zE["in_box"] & np.isfinite(zE["logq_ref"]) & np.isfinite(zR["logq_ref"])
        d = -zE["logq_ref"][ok] + zR["logq_ref"][ok]
        # Dbar = the mean of d_i over the ok rows (sec. 2.5, F1 of wave W2)
        assert abs(js["Dbar"] - d.mean()) < 1e-12
        sdE = np.std(-zE["logq_ref_members"][ok].mean(0), ddof=1)
        sdR = np.std(-zR["logq_ref_members"][ok].mean(0), ddof=1)
        # sigma_seed: the larger of the two ensembles' SDs over members of their mean NLL
        assert abs(js["sigma_seed"] - max(sdE, sdR)) < 1e-9, (js["sigma_seed"], sdE, sdR)
        assert js["n_donors"] == len(np.unique(zE["donor"][ok]))


def test_reproducible_metrics():
    _report("runA")
    _report("runB")
    ma, mb = os.path.join(OUT, "runA", "metrics"), os.path.join(OUT, "runB", "metrics")
    fa = sorted(os.path.basename(f) for f in glob.glob(ma + "/*"))
    assert fa == sorted(os.path.basename(f) for f in glob.glob(mb + "/*"))
    for f in fa:
        if f.endswith(".npz"):
            a, b = _load(os.path.join(ma, f)), _load(os.path.join(mb, f))
            for k in a:
                assert_array_equal(a[k], b[k], err_msg="%s %s" % (f, k))
        elif f != "METRICS.json":
            ja = open(os.path.join(ma, f)).read().replace("/runA/", "/runX/")
            jb = open(os.path.join(mb, f)).read().replace("/runB/", "/runX/")
            assert ja == jb, f


def test_report_wording_and_figures():
    v, ec = _report("runA")
    rep = open(os.path.join(OUT, "runA", "REC_REPORT.md")).read()
    figs = open(os.path.join(OUT, "runA", "figures", "FIGURES.md")).read()
    for txt in (rep, figs):
        assert "well's posterior" not in txt and "misspecified" not in txt.lower()
    assert "the mixture over the well's windows" in rep
    # the figure set of block 37 for this view (2 ensembles, one bank "real" with 3 classes,
    # one comparison), written from the entry
    exp = set()
    for e in v.ensembles:
        for f in ("cal_coverage", "cal_sbc_ecdf", "cal_sbc_hist", "cal_tarp", "acc_recovery",
                  "acc_zscore", "acc_nll", "acc_contraction", "gate_null", "gate_wells_real",
                  "gate_windows_real", "gate_members_real", "emb_pca_real", "rec_wells_real"):
            exp.add("ensembles/%s/%s" % (e.stem, f))
        zr = _load(os.path.join(OUT, "runA", "samples", "%s_real.npz" % e.stem))
        for c in np.unique(zr["cls"]):
            exp.add("ensembles/%s/rec_corner_real_c%d" % (e.stem, c))
    exp |= {"comparisons/S-A2-R0/cmp_nll", "comparisons/S-A2-R0/cmp_forest", "across/dbar"}
    listed = set(re.findall(r"- `([^`]+)\.\{", figs))
    assert listed == exp, (listed ^ exp)
    for name in exp:
        for fmt in ec.formats:
            path = os.path.join(OUT, "runA", "figures", name + "." + fmt)
            assert os.path.isfile(path), path
            if fmt == "png":
                assert open(path, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
    comp = json.load(open(os.path.join(OUT, "runA", "REC_REPORT_COMPLETE.json")))
    assert comp["schema"] == "rec_eval_report/1" and comp["n_figures"] == len(exp)
    assert [l.split(":")[0] for l in comp["pass_lines"]] == ["REC METRICS OK", "REC FIGURES OK",
                                                              "REC REPORT OK"]
