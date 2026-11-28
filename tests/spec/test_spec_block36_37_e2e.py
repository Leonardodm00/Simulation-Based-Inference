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


# ---------------------------------------------------------------- 2026-11-23_1906: block 36's fix round
# Oracles (sec. 2.6 "The rows outside the open box", block 36 Rule, R36h,
# R36g's addition): on the two-row boundary sample the NLL, gamma, C_E and
# the edge split exclude the rows outside the box; SBC, TARP and the
# recovery keep every calibration / draw row; a comparison is over the rows
# in the box; a sample made from another best fit is left out and named.

def _box_report():
    from test_spec_block35_sample import box_run
    v, cfg, ecfg, out, rr = box_run()
    if not os.path.isfile(os.path.join(out, "REC_REPORT_COMPLETE.json")):
        r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_report.py"), "--config",
                            cfg, "--eval-config", ecfg, "--ncpus", "1"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "REC REPORT OK" in r.stdout, r.stdout + r.stderr
    return v, cfg, ecfg, out, rr


def test_box_metrics_from_samples():
    _report("runA")
    v, cfg, ecfg, out, (r_draw, r_other) = _box_report()
    for e in v.ensembles:
        z = _load(os.path.join(out, "samples", "%s_sim.npz" % e.stem))
        js = json.load(open(os.path.join(out, "metrics", "%s_sim.json" % e.stem)))
        a = _load(os.path.join(out, "metrics", "%s_sim.npz" % e.stem))
        a0 = _load(os.path.join(OUT, "runA", "metrics", "%s_sim.npz" % e.stem))
        js0 = json.load(open(os.path.join(OUT, "runA", "metrics", "%s_sim.json" % e.stem)))
        n = len(z["rows"])
        ib, cal, didx = z["in_box"], z["calib"], z["draw_idx"]
        bad = ~ib
        assert bad.sum() == 2
        l = -z["logq_ref"]
        assert js["nll"]["n_out_of_box"] == 2 and js["nll"]["n_rows"] == n - 2
        assert js["n_in_box"] == n - 2
        # 1e-12: one mean of 14 float64 values, two summation orders
        assert abs(js["nll"]["mean"] - np.mean(l[ib])) < 1e-12
        assert np.all(np.isnan(a["nll_rows"][bad]))
        # the other rows' NLL: the unmodified run's (1e-4 nats, R35b's float32 tolerance)
        assert np.nanmax(np.abs(a["nll_rows"][ib] - a0["nll_rows"][ib])) <= 1e-4
        # each member's mean over the rows in the box
        assert_allclose(js["nll"]["members"], (-z["logq_ref_members"][ib]).mean(0), rtol=0,
                        atol=1e-12)
        # gamma: NaN on the draw row outside the box, the strict-> share elsewhere
        g = np.full(n, np.nan)
        g[didx] = np.mean(z["logq_draws"] > z["logq_ref"][didx][:, None], axis=1)
        g[~ib] = np.nan
        assert_allclose(a["gamma"], g, rtol=0, atol=0, equal_nan=True)
        assert np.isnan(a["gamma"][z["rows"] == r_draw][0])
        s_cal = cal & ib
        assert js["n_cal"] == int(cal.sum()) - 1 and js["n_calib"] == int(cal.sum())
        assert js["coverage"]["n_rows"] == js["n_cal"]
        assert_allclose(a["coverage"], [np.mean(g[s_cal] <= b) for b in BETAS], atol=0)
        assert js["edge"]["n_cal"] == js["n_cal"]
        # [2026-11-27, D-168] SBC and TARP are over S_cal (the calibration rows in
        # the box), no longer over every calibration row: sbc.n_rows and the first
        # axis of sbc_ranks, tarp_x_f, tarp_random_f are n_cal
        assert js["sbc"]["n_rows"] == js["n_cal"] == int(s_cal.sum())
        assert a["sbc_ranks"].shape[0] == js["n_cal"]
        assert a["tarp_x_f"].shape[0] == js["n_cal"]
        assert a["tarp_random_f"].shape[0] == js["n_cal"]
        # the ranks by definition, r_{i,k} = #{s : theta^(s)_{i,k} < theta_{i,k}},
        # on the S_cal rows in global row order (exact: integer counts)
        pos = np.full(n, -1)
        pos[didx] = np.arange(didx.size)
        rk = np.stack([(z["draws"][pos[i]] < z["theta_ref"][i]).sum(0)
                       for i in np.flatnonzero(s_cal)])
        assert_array_equal(a["sbc_ranks"], rk)
        # the boundary row is not among them; the ranks kept are the unmodified
        # run's on the same rows (the draws are bit for bit; exact)
        row = int(np.flatnonzero(z["rows"] == r_draw)[0])
        assert cal[row] and not s_cal[row]
        keep = s_cal[cal]
        assert_array_equal(a["sbc_ranks"], a0["sbc_ranks"][keep])
        # TARP's per-row coverage values of the kept rows: tarp_block's seeded
        # reference points depend on the set of rows, so only the shape is fixed
        # here; the wiring (block 20's functions on S_cal with the seed of block
        # 20's form) is checked in test_dcal_sbc_tarp_wiring
        # the recovery on every draw row, the boundary one included
        assert np.all(np.isfinite(a["mhat"][didx]))
        # the edge NLL / interior NLL split over the rows in the box only
        edge = a["edge"]
        assert js["edge"]["nll_edge_n"] + js["edge"]["nll_int_n"] == n - 2
        # the gate does not read theta: its numbers equal the unmodified run's
        assert js["gate"]["check"] == js0["gate"]["check"]
        assert_array_equal(a["gate_T_null"], a0["gate_T_null"])
        del edge


def test_box_paired_over_rows_in_box():
    v, cfg, ecfg, out, rr = _box_report()
    idx = json.load(open(os.path.join(out, "metrics", "METRICS.json")))
    by = dict((e.name, e) for e in v.ensembles)
    assert idx["comparisons"], idx["comparisons_left_out"]
    for c in idx["comparisons"]:
        zE = _load(os.path.join(out, "samples", "%s_sim.npz" % by[c["E"]].stem))
        zR = _load(os.path.join(out, "samples", "%s_sim.npz" % by[c["E_ref"]].stem))
        js = json.load(open(c["paired"]["sim"]))
        ok = zE["in_box"] & zR["in_box"]
        assert js["n_rows"] == int(ok.sum()) == len(zE["rows"]) - 2
        d = -zE["logq_ref"][ok] + zR["logq_ref"][ok]
        assert abs(js["Dbar"] - d.mean()) < 1e-12
        assert np.isfinite(js["Dbar"])


def test_box_report_written():
    v, cfg, ecfg, out, rr = _box_report()
    done = json.load(open(os.path.join(out, "REC_REPORT_COMPLETE.json")))
    assert len(done["pass_lines"]) == 3
    assert os.path.isfile(os.path.join(out, "REC_REPORT.md"))


def test_stale_best_fit_left_out_and_named(tmp_path):
    """Block 36 Rule (D-167): a sample whose best_fit_complete_sha256 is not
    the sha256 of its folder's current BEST_FIT_COMPLETE.json is left out
    and named. Simulated without touching the fixture: the samples copied to
    a new out_root, one provenance's digest changed."""
    import shutil
    v, ec = _report("runA")
    src = os.path.join(OUT, "runA", "samples")
    d = json.load(open(os.path.join(FIX, "rec_eval.json")))
    d["out_root"] = str(tmp_path / "out")
    ep = str(tmp_path / "e.json")
    json.dump(d, open(ep, "w"))
    dst = os.path.join(d["out_root"], "samples")
    shutil.copytree(src, dst)
    stale = v.ensembles[1]
    pj = os.path.join(dst, "%s_sample.json" % stale.stem)
    prov = json.load(open(pj))
    prov["best_fit_complete_sha256"] = "0" * 64
    json.dump(prov, open(pj, "w"))
    r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_metrics.py"), "--config",
                        os.path.join(FIX, "synth_rec.json"), "--eval-config", ep, "--ncpus", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    idx = json.load(open(os.path.join(d["out_root"], "metrics", "METRICS.json")))
    left = dict((x["ensemble"], x["why"]) for x in idx["left_out"])
    assert stale.name in left and "best fit" in left[stale.name], left
    assert stale.name not in idx["ensembles"]
    assert set(idx["ensembles"]) == set(e.name for e in v.ensembles) - {stale.name}
    # every sample stale: a refusal (exit 3), "no ensemble with a current sample"
    for e in v.ensembles:
        pj = os.path.join(dst, "%s_sample.json" % e.stem)
        prov = json.load(open(pj)); prov["best_fit_complete_sha256"] = "1" * 64
        json.dump(prov, open(pj, "w"))
    r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_metrics.py"), "--config",
                        os.path.join(FIX, "synth_rec.json"), "--eval-config", ep, "--ncpus", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr


def test_empty_s_cal_refuses_the_metrics():
    """[2026-11-27, D-168] Sec. 2.6: a row outside the box is never a reason to
    refuse 'unless no row of D_sim is in the open box (the sampler) or no
    calibration row is (the metrics)'; block 36: 'an ensemble with an empty
    S_cal refuses the metrics'. Every calibration row (= draw row at k_draw 1)
    outside the box, the other rows inside: the sampling completes (exit 0),
    the metrics refuse (exit 3, a REFUSE line naming the calibration rows) and
    write no METRICS.json. (Replaces the 2026-11-23 test of the opposite
    rule, which the spec has since changed.)"""
    from test_spec_block35_sample import _sim_rows, _box_tree, _sample_cli
    v, z = _sim_rows()
    edits = [(int(r), 0, 1.0) for r in z["rows"][z["calib"]]]
    cfg, ecfg, out = _box_tree("boxcal", edits)
    for e in v.ensembles:
        if not os.path.isfile(os.path.join(out, "samples", "%s_sample.json" % e.stem)):
            r = _sample_cli(cfg, ecfg, e.name)
            assert r.returncode == 0, r.stdout + r.stderr
    import shutil
    shutil.rmtree(os.path.join(out, "metrics"), ignore_errors=True)
    r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_metrics.py"), "--config",
                        cfg, "--eval-config", ecfg, "--ncpus", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 3, (r.returncode, (r.stdout + r.stderr)[-3000:])
    assert "Traceback" not in r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("REFUSE")][-1]
    assert "calibration" in line and "open box" in line, line
    assert not os.path.isfile(os.path.join(out, "metrics", "METRICS.json"))
    # recorded for the report: what the refused run left in metrics/ (the spec
    # does not say the metrics write nothing on a refusal)
    left = sorted(os.listdir(os.path.join(out, "metrics"))) \
        if os.path.isdir(os.path.join(out, "metrics")) else []
    print("metrics/ after the refusal:", left)


def test_dcal_sbc_tarp_wiring():
    """[D-168] Block 36 Rule: SBC and TARP are block 20's sbc_block and
    tarp_block (imported, the spec's named reference functions), applied to
    S_cal's rows with the seeds seed_of(seed, "metrics", <stem>, "sim",
    "sbc" | "tarp"). Their JSON values and arrays equal a call made here on
    the rows selected by hand (1e-12: the same functions on the same arrays,
    the JSON's float round trip)."""
    sys.path.insert(0, EVAL)
    import _evalpath  # noqa: F401
    import ev_metrics as EM
    from ev_plan import seed_of
    v, cfg, ecfg, out, rr = _box_report()
    ec = json.load(open(ecfg))
    for e in v.ensembles:
        z = _load(os.path.join(out, "samples", "%s_sim.npz" % e.stem))
        js = json.load(open(os.path.join(out, "metrics", "%s_sim.json" % e.stem)))
        a = _load(os.path.join(out, "metrics", "%s_sim.npz" % e.stem))
        s_cal = z["calib"] & z["in_box"]
        pos = np.full(len(z["rows"]), -1)
        pos[z["draw_idx"]] = np.arange(z["draw_idx"].size)
        pc = pos[s_cal]
        names = js["axes"]["names"]
        S = lambda *p: seed_of(ec["seed"], "metrics", e.stem, *p)   # noqa: E731
        s, sa = EM.sbc_block(z["theta_ref"][s_cal], z["draws"][pc], ec["level"], S("sim", "sbc"),
                             axes=names)
        assert_allclose(js["sbc"]["ks_p"], s["ks_p"], rtol=0, atol=1e-12)
        assert js["sbc"]["holm_rejected"] == list(s["holm_rejected"])
        for k, val in sa.items():
            assert_allclose(a[k], val, rtol=0, atol=1e-12, err_msg=k)
        t, ta = EM.tarp_block(z["theta_ref"][s_cal], z["draws"][pc], z["stats"][pc],
                              S("sim", "tarp"))
        for mode in ("x", "random"):
            assert abs(js["tarp"][mode]["ks_p"] - t[mode]["ks_p"]) <= 1e-12
        for k, val in ta.items():
            assert_allclose(a[k], val, rtol=0, atol=1e-12, err_msg=k)


def _fresh_samples(tag):
    """Every ensemble of the recorded project sampled by the CLI into a new
    out_root OUT/<tag> (so that a file can be deleted in place)."""
    from test_spec_block35_sample import _sample_cli
    from test_spec_block35_sample import _view
    d = json.load(open(os.path.join(FIX, "rec_eval.json")))
    d["out_root"] = os.path.join(OUT, tag)
    ep = os.path.join(OUT, "%s_rec_eval.json" % tag)
    json.dump(d, open(ep, "w"), indent=1)
    cfg = os.path.join(FIX, "synth_rec.json")
    v, ec = _view(cfg, ep)
    for e in v.ensembles:
        if not os.path.isfile(os.path.join(d["out_root"], "samples", "%s_sample.json" % e.stem)):
            r = _sample_cli(cfg, ep, e.name)
            assert r.returncode == 0, r.stdout + r.stderr
    return v, cfg, ep, d["out_root"]


def _metrics_cli(cfg, ep):
    return subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_metrics.py"), "--config",
                           cfg, "--eval-config", ep, "--ncpus", "1"],
                          capture_output=True, text=True)


def test_missing_sample_file_in_place_left_out_and_named():
    """[2026-11-24] Block 36 Rule: 'a sample one of whose files (its
    provenance's files) is missing counts as absent: left out and named'.
    One ensemble's recording-bank file deleted in place: exit 0, that
    ensemble in left_out with a reason naming the file, not reported, its
    comparison named as not made; every sample with a file missing: exit 3."""
    v, cfg, ep, out = _fresh_samples("runF")
    sd = os.path.join(out, "samples")
    gone_e = v.ensembles[1]
    gone = os.path.join(sd, "%s_real.npz" % gone_e.stem)
    os.rename(gone, gone + ".away")
    try:
        r = _metrics_cli(cfg, ep)
        assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
        idx = json.load(open(os.path.join(out, "metrics", "METRICS.json")))
        left = dict((x["ensemble"], x["why"]) for x in idx["left_out"])
        assert gone_e.name in left and os.path.basename(gone) in left[gone_e.name], left
        assert gone_e.name not in idx["ensembles"]
        assert set(idx["ensembles"]) == set(e.name for e in v.ensembles) - {gone_e.name}
        assert any(gone_e.name in (c["E"], c["E_ref"]) for c in idx["comparisons_left_out"])
        # the sim file of the other ensemble missing as well: no current sample
        other = os.path.join(sd, "%s_sim.npz" % v.ensembles[0].stem)
        os.rename(other, other + ".away")
        try:
            r = _metrics_cli(cfg, ep)
            assert r.returncode == 3, (r.stdout + r.stderr)[-3000:]
            assert "Traceback" not in r.stderr
        finally:
            os.rename(other + ".away", other)
    finally:
        os.rename(gone + ".away", gone)


def test_missing_sample_file_in_relocated_tree_not_a_traceback(tmp_path):
    """The 2026-11-23_1906 report's F3 reproducer: the samples COPIED to a new
    out_root and one file removed from the copy. The provenance's `files`
    still name the original (existing) paths, while the metrics read
    <out_root>/samples/. Block 36's CLI contract allows exit 0 (left out and
    named) or exit 3 (a refusal); a traceback (exit 1) is reported."""
    import shutil
    v, cfg, ep0, out0 = _fresh_samples("runF")
    d = json.load(open(ep0))
    d["out_root"] = str(tmp_path / "out")
    ep = str(tmp_path / "e.json")
    json.dump(d, open(ep, "w"))
    dst = os.path.join(d["out_root"], "samples")
    shutil.copytree(os.path.join(out0, "samples"), dst)
    os.remove(os.path.join(dst, "%s_real.npz" % v.ensembles[1].stem))
    r = _metrics_cli(cfg, ep)
    assert r.returncode in (0, 3), (r.returncode, (r.stdout + r.stderr)[-1500:])


# ---------------------------------------------------------------- 2026-11-27_1832: D-172 end to end
# Oracle O38.2 / step 6: the recorded project sampled and reported through the
# real command lines with the new default fractions (0.25, 0.25, 0.4, 0.1):
# the networks dealt by block 34's rule written here (permutation of the
# sorted distinct networks by default_rng(seed_of(seed, "gate")), floor(f G +
# 1e-9) per group, the rest to the check group; round-robin by occurrence),
# the provenance's gate.networks, and two runs with the same seed identical.

def _d172_run(tag):
    from test_spec_block35_sample import _sample_cli, _view
    d = json.load(open(os.path.join(FIX, "rec_eval.json")))
    d["out_root"] = os.path.join(OUT, tag)
    d["gate"]["donor_fracs"] = [0.25, 0.25, 0.4, 0.1]
    ep = os.path.join(OUT, "%s_rec_eval.json" % tag)
    json.dump(d, open(ep, "w"), indent=1)
    cfg = os.path.join(FIX, "synth_rec.json")
    v, ec = _view(cfg, ep)
    for e in v.ensembles:
        if not os.path.isfile(os.path.join(d["out_root"], "samples", "%s_sample.json" % e.stem)):
            r = _sample_cli(cfg, ep, e.name)
            assert r.returncode == 0 and "REC SAMPLE OK" in r.stdout, r.stdout + r.stderr
    if not os.path.isfile(os.path.join(d["out_root"], "REC_REPORT_COMPLETE.json")):
        r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_report.py"), "--config",
                            cfg, "--eval-config", ep, "--ncpus", "1"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "REC REPORT OK" in r.stdout, (r.stdout + r.stderr)[-2000:]
    return v, d


def _roles_by_hand(donor, fracs, caps, rpd, seed):
    nets = np.unique(donor)
    G = nets.size
    perm = nets[np.random.default_rng(seed).permutation(G)]
    sizes = [int(np.floor(f * G + 1e-9)) for f in fracs[:3]]
    groups = [perm[:sizes[0]], perm[sizes[0]:sizes[0] + sizes[1]],
              perm[sizes[0] + sizes[1]:sum(sizes)], perm[sum(sizes):]]
    occ = np.zeros(donor.size, dtype=int)
    seen = {}
    for i, dn in enumerate(donor):
        occ[i] = seen.get(int(dn), 0)
        seen[int(dn)] = occ[i] + 1
    role = np.zeros(donor.size, dtype=int)
    for g, (members, cap) in enumerate(zip(groups, caps)):
        idx = [i for i in range(donor.size) if donor[i] in set(members.tolist())]
        if g > 0:
            idx = [i for i in idx if occ[i] < rpd]
        idx.sort(key=lambda i: (occ[i], i))
        if cap is not None:
            idx = idx[:cap]
        role[idx] = g + 1
    return role, [len(x) for x in groups]


def test_d172_fracs_end_to_end_and_reproducible():
    sys.path.insert(0, EVAL)
    import _evalpath  # noqa: F401
    from ev_plan import seed_of          # block 19's seed function, outside this scope
    v, d = _d172_run("d172A")
    v2, d2 = _d172_run("d172B")
    g = d["gate"]
    for e in v.ensembles:
        z = _load(os.path.join(d["out_root"], "samples", "%s_sim.npz" % e.stem))
        prov = json.load(open(os.path.join(d["out_root"], "samples", "%s_sample.json" % e.stem)))
        role, nets = _roles_by_hand(z["donor"], g["donor_fracs"],
                                    [g["n_ref"], g["n_rank"], g["n_null"], None],
                                    g["rows_per_donor"], seed_of(d["seed"], "gate"))
        assert_array_equal(z["gate_role"], role)
        assert prov["gate"]["networks"] == nets, (prov["gate"]["networks"], nets)
        assert prov["sample_params"]["gate"]["donor_fracs"] == [0.25, 0.25, 0.4, 0.1] \
            if isinstance(prov["sample_params"].get("gate"), dict) else True
        # the null set is now the largest of the three capped groups' networks
        assert nets[2] >= nets[0]
        js = json.load(open(os.path.join(d["out_root"], "metrics", "%s_sim.json" % e.stem)))
        assert js["gate"]["n_null"] == int(np.sum(role == 3))
        # same seed, same files: every array of the sample and of the metrics
        for sub, f in (("samples", "%s_sim.npz"), ("samples", "%s_real.npz"),
                       ("metrics", "%s_sim.npz"), ("metrics", "%s_real.npz")):
            a = _load(os.path.join(d["out_root"], sub, f % e.stem))
            b = _load(os.path.join(d2["out_root"], sub, f % e.stem))
            assert set(a) == set(b)
            for k in a:
                assert_array_equal(a[k], b[k], err_msg="%s %s" % (f, k))
            # every float array finite where the spec gives no NaN
            for k in ("draws", "z", "logq_draws"):
                if k in a:
                    assert np.all(np.isfinite(a[k])), k
    print("D-172 networks per group:", nets)


# ---------------------------------------------------------------- 2026-11-28 (39e6f66)
# O36.12 [block 36's Rule, amended 2026-11-27]: "the files are looked for where the run reads
# them, <out_root>/samples/<name>, not at the provenance's recorded paths". (a) A complete
# copy of the samples under a new out_root is current: every ensemble reported, nothing left
# out. (b) The copy with one file removed: that ensemble left out with a reason naming the
# file under the NEW out_root, exit 0.

def test_relocated_samples_read_where_the_run_reads_them(tmp_path):
    import shutil
    v, cfg, ep0, out0 = _fresh_samples("runF")
    for case in ("complete", "one_missing"):
        d = json.load(open(ep0))
        d["out_root"] = str(tmp_path / case)
        ep = str(tmp_path / ("%s.json" % case))
        json.dump(d, open(ep, "w"))
        dst = os.path.join(d["out_root"], "samples")
        shutil.copytree(os.path.join(out0, "samples"), dst)
        gone = os.path.join(dst, "%s_real.npz" % v.ensembles[1].stem)
        if case == "one_missing":
            os.remove(gone)
        r = _metrics_cli(cfg, ep)
        assert r.returncode == 0, (case, (r.stdout + r.stderr)[-1500:])
        idx = json.load(open(os.path.join(d["out_root"], "metrics", "METRICS.json")))
        left = dict((x["ensemble"], x["why"]) for x in idx["left_out"])
        if case == "complete":
            assert left == {} and set(idx["ensembles"]) == set(e.name for e in v.ensembles)
        else:
            assert set(left) == {v.ensembles[1].name}
            assert gone in left[v.ensembles[1].name], left
            assert out0 not in left[v.ensembles[1].name]
