"""Block 21 (the evaluation's figures), derived from specs/SPEC.md block 21
(oracles O21.1-O21.2 of the t1-C report): the set of files the rules give,
stated here independently, the index, and the module computing nothing.

Pure ASCII, LF only.
"""

import ast
import os
import re

import numpy as np
import pytest

from conftest import EVAL, files_under, load_json, run_py, write_eval_config

FIG = os.path.join(EVAL, "ev_figures.py")
STATS_LIBS = {"scipy", "sklearn", "torch", "sbi", "npe_diagnostics", "bootstrap_paired",
              "ev_metrics"}
STAT_CALLS = {"mean", "std", "var", "quantile", "percentile", "median", "nanmean", "nanstd",
              "nanmedian", "nanquantile", "nanpercentile", "average"}


def _tree():
    return ast.parse(open(FIG).read())


def test_no_statistics_library_imported():
    mods = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            mods |= set(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add(node.module.split(".")[0])
    assert not (mods & STATS_LIBS), sorted(mods & STATS_LIBS)


def test_no_mean_sd_quantile_median_called():
    bad = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in STAT_CALLS:
            bad.append((node.lineno, node.func.attr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in STAT_CALLS:
            bad.append((node.lineno, node.func.id))
    assert not bad, bad


def test_validated_colours():
    import ev_figures as F
    assert F.ARM_COLOR == {"A1": "#2a78d6", "A2": "#eb6834", "A5": "#1baf7a"}
    assert tuple(F.COND_COLOR) == ("#e87ba4", "#6250d6", "#008300")
    assert tuple(F.COND_MARKER) == ("o", "s", "^")


def _expected(evaluation):
    """The figures block 21's file list gives for the fixture's ensembles,
    data sets and searches (an independent statement of the rules)."""
    cfg = load_json(evaluation["cfg"])
    out = set()
    by_search = {}
    for n in evaluation["names"]:
        stem = n.replace(":", "_")
        prov = load_json(os.path.join(evaluation["out"], "samples", "%s_sample.json" % stem))
        dss = [os.path.basename(f)[len(stem) + 1:-4] for f in prov["files"]
               if not f.endswith("_cal.npz")]
        cohorts = list(prov["cohorts"])
        q = n.split(":")[0]
        by_search.setdefault(q, []).append((n.split(":")[1], dss, cohorts))
        e = "ensembles/%s/" % stem
        out |= {e + "cal_coverage", e + "cal_tarp", e + "acc_sigma", e + "acc_nll",
                e + "post_lc2st"}
        for D in dss:
            out |= {e + "cal_sbc_ecdf_" + D, e + "cal_sbc_hist_" + D, e + "acc_recovery_" + D,
                    e + "acc_zscore_" + D}
        for D in ["sim"] + cohorts:
            out.add(e + "post_ppc_" + D)
        for c in cohorts:
            out.add(e + "emb_pca_" + c)
            m = load_json(os.path.join(evaluation["out"], "metrics", "search_%s.json" % q))
            for ex in m["example_donors"][c]["donors"]:
                out.add(e + "post_corner_%s_c%d" % (c, ex["cls"]))
    for q, arms in by_search.items():
        s = "searches/%s/" % q
        out |= {s + "search_trials", s + "search_nll_vs_lpr", s + "search_learning"}
        if len(arms) == 2:
            dss = [d for d in arms[0][1] if d in arms[1][1]]
            cohorts = [c for c in arms[0][2] if c in arms[1][2]]
            out.add(s + "arms_coverage")
            for D in dss:
                if D != "sim":
                    out.add(s + "arms_nll_" + D)
                out.add(s + "arms_forest_" + D)
            for c in cohorts:
                out |= {s + "arms_ppc_" + c, s + "arms_lc2st_" + c}
    out |= {"across/dbar", "across/lpr"}
    return out


def test_figure_set_and_index(evaluation):
    rc, txt = evaluation["logs"]["figures"]
    assert rc == 0, txt[-3000:]
    m = re.search(r"EVAL FIGURES OK: (\d+) figure\(s\)", txt)
    assert m, txt[-1500:]
    fdir = os.path.join(evaluation["out"], "figures")
    ecfg = load_json(evaluation["ecfg"])
    files = [f for f in files_under(fdir) if f != "FIGURES.md"]
    got = set(os.path.splitext(f)[0] for f in files)
    want = _expected(evaluation)
    assert got == want, ("unexpected", sorted(got - want), "missing", sorted(want - got))
    assert int(m.group(1)) == len(want)
    assert len(files) == len(want) * len(ecfg["formats"])
    idx = open(os.path.join(fdir, "FIGURES.md")).read().splitlines()
    listed = {}
    for ln in idx:
        mm = re.match(r"^- `([^`]+)\.\{([^}]+)\}`: (.+)$", ln)
        if mm:
            assert mm.group(1) not in listed, mm.group(1)
            listed[mm.group(1)] = mm.group(3)
    assert set(listed) == want
    assert all(len(c.strip(" .")) > 10 for c in listed.values())
    for f in files:
        if f.endswith(".png"):
            with open(os.path.join(fdir, f), "rb") as fh:
                assert fh.read(8) == b"\x89PNG\r\n\x1a\n", f


def test_no_metrics_refused(evaluation, tmp_path):
    out = str(tmp_path / "empty_eval")
    ec = write_eval_config(str(tmp_path / "e.json"), out)
    rc, txt = run_py([FIG, "--config", evaluation["cfg"], "--eval-config", ec])
    assert rc == 3, (rc, txt[-1500:])
    assert not os.path.isdir(os.path.join(out, "figures")) or not files_under(
        os.path.join(out, "figures"))


def test_swap_the_plot_style(evaluation, tmp_path):
    """Directive audit, swappability: another file format and dpi through the
    configuration alone; the figures' rules unchanged."""
    import shutil
    out = str(tmp_path / "ev")
    for d in ("samples", "metrics"):
        shutil.copytree(os.path.join(evaluation["out"], d), os.path.join(out, d))
    ec = write_eval_config(str(tmp_path / "e.json"), out, formats=["svg"], dpi=30)
    rc, txt = run_py([FIG, "--config", evaluation["cfg"], "--eval-config", ec])
    assert rc == 0, txt[-2000:]
    files = [f for f in files_under(os.path.join(out, "figures")) if f != "FIGURES.md"]
    assert files and all(f.endswith(".svg") for f in files)
    assert set(os.path.splitext(f)[0] for f in files) == _expected(evaluation)
