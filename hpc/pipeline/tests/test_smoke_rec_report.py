#!/usr/bin/env python3
"""Smoke test of block 37: the figures and the report of a recorded project (specs/SPEC.md block 37, "Test
oracles", checks R37*).

Every check runs on rec_fixture's recorded-kind project (cfg_rec, one
recording bank `real`, two ensembles, one comparison) and on its as-recorded
one (cfg with rec_eval_as.json: three ensembles, the cohorts R0 and R3 as
recording banks, the comparisons of block 27), each sampled here by block
35's sampler and measured by block 36's compute into an out_root of this
test's own (never the fixture's).

Checks
    test_r37a_figures          the set of figures, FIGURES.md, the PNG signatures, a figure to PDF
    test_r37b_static           the figures' module computes nothing scientific; block 21's colours
    test_r37c_report           every number of the report equals the metrics' JSON value it names; the wording
    test_r37d_end_to_end       ev_rec_report.py: the three pass lines, REC_REPORT_COMPLETE.json; no metrics refused

Run
    python3 hpc/pipeline/tests/test_smoke_rec_report.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<dir> reuses a fixture tree built
        before (rec_fixture.py); REC_KEEP=1 keeps the one this run builds
        and this test's temporary directories.

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

import ast  # noqa: E402
import atexit  # noqa: E402
import decimal  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402

SEED = 20261123  # every random number of this file derives from it
PYTHON = sys.executable

# ---------------------------------------------------------------- the lists of block 37
# An independent statement of the entry's figures (R37a): not read from the
# module's constants, which the test compares with these.
ENSEMBLE_NAMES = ("cal_coverage", "cal_sbc_ecdf", "cal_sbc_hist", "cal_tarp", "acc_recovery",
                  "acc_zscore", "acc_nll", "acc_contraction", "gate_null")
BANK_NAMES = ("gate_wells_%s", "gate_windows_%s", "gate_members_%s", "emb_pca_%s",
              "rec_wells_%s")
CORNER_NAME = "rec_corner_%s_c%d"
COMPARISON_NAMES = ("cmp_nll", "cmp_forest")
# block 21's validated colours (dataviz validator, PASS all-pairs, light)
ARM_COLOR = {"A1": "#2a78d6", "A2": "#eb6834", "A5": "#1baf7a"}
COND_COLOR = ("#e87ba4", "#6250d6", "#008300")
# R37b: the statistics libraries and the calls the figures' module must not use
BANNED_MODULES = {"scipy", "sklearn", "torch", "sbi", "npe_diagnostics", "bootstrap_paired",
                  "ev_metrics", "ev_rec_metrics", "ev_gate"}
BANNED_CALLS = {"quantile", "nanquantile", "percentile", "nanpercentile", "mean", "nanmean",
                "std", "nanstd", "var", "nanvar", "median", "nanmedian", "average"}
# R37c: what a quantity/value table's label names, as the last keys of its
# source (an independent statement of the report's tables)
LABEL_KEY = {"alpha_gate": "alpha", "k_gate (R9)": "k_gate", "N_null, null rows": "n_null",
             "N_ref, reference rows": "n_ref", "N_rank, rank rows": "n_rank",
             "N_chk, check rows": "n_check", "size band, lower end": "size_lo",
             "size band, upper end": "size_hi",
             "size band's mean, k_gate / (N_null + 1)": "size_mean",
             "check rows' share flagged": "share", "its exact interval, lower end": 0,
             "its exact interval, upper end": 1,
             "pi_edge, the share of edge rows in S_cal": "pi_edge",
             "its value under the box prior, 1 - (1 - 2 eps)^p": "pi_edge_expected",
             "eps, the edge width": "eps",
             "largest absolute difference between the two sides of (R2)": "r2_max_abs_dev",
             "number of axes rejected": "n_rejected", "calibration rows": "n_rows"}
# R37c: what a column header names, as the last key (or the one before an
# axis index) of its source
HEADER_KEY = {"median": "q50", "5 %": "q05", "95 %": "q95", "mean": "mean", "SD": "sd",
              "mean NLL": ("mean", "nll_int_mean", "nll_edge_mean", "members", "members_sd"),
              "Dbar (nats)": "Dbar", "2 sigma_seed": "two_sigma_seed",
              "share flagged": ("share", "share_flagged"), "mean T": "mean_T",
              "C at 0.50": "0.50", "C at 0.68": "0.68", "C at 0.90": "0.90", "C at 0.95": "0.95",
              "SBC KS p": "ks_p", "Holm-adjusted p": "holm_adjusted_p",
              "contraction, D_sim": "contraction", "KS p": "ks_p", "max deviation": "max_dev",
              "interval, lower end": 0, "interval, upper end": 1, "windows": "n",
              "flagged": "n_flagged", "class": "cls", "well": "well", "batch": "batch"}

_TMP = []
_TREES = {}


def _cleanup_tmp():
    if os.environ.get("REC_KEEP", "").strip() == "1":
        for d in _TMP:
            print("[test_smoke_rec_report] kept: %s" % d, flush=True)
        return
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_tmp)


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="r37%s_" % tag)
    _TMP.append(d)
    return d


def _write_ecfg(src, out_root, name, **over):
    """A copy of the evaluation configuration `src` with another out_root (and
    `over`), written beside out_root; returns its path."""
    with open(src) as fh:
        d = json.load(fh)
    d["out_root"] = out_root
    d.update(over)
    path = os.path.join(os.path.dirname(out_root), name)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    return path


def _tree(kind):
    """(cfg, ecfg_path): the fixture's project of `kind` ("recorded" or
    "as_recorded") sampled by block 35 and measured by block 36 into an
    out_root of this test's own; built once per process. The as-recorded
    tree's figures are written as PNG and PDF (R37a: a figure to PDF)."""
    if kind in _TREES:
        return _TREES[kind]
    import ev_rec_config
    import ev_rec_metrics
    import ev_rec_plan
    import ev_rec_sample
    F = rec_fixture.fixture()
    cfg, src = (F["cfg_rec"], F["ecfg_rec"]) if kind == "recorded" else (F["cfg"], F["ecfg_as"])
    root = _tmpdir(kind[:3])
    over = {"formats": ["png", "pdf"]} if kind == "as_recorded" else {}
    ecfg_path = _write_ecfg(src, os.path.join(root, "out"), "rec_eval.json", **over)
    ecfg = ev_rec_config.load(ecfg_path)
    view = ev_rec_plan.load_view(cfg, ecfg)
    t0 = time.time()
    for e in view.ensembles:
        ev_rec_sample.run(cfg, ecfg_path, e.name, log=lambda t: None)
    ev_rec_metrics.compute(cfg, ecfg_path, ncpus=1, log=lambda t: None)
    print("[r37] %s project: %d ensemble(s) sampled and measured in %.0f s"
          % (kind, len(view.ensembles), time.time() - t0), flush=True)
    _TREES[kind] = (cfg, ecfg_path)
    return _TREES[kind]


_FIGS = {}


def _figured(kind):
    """(cfg, ecfg_path, index, log lines, warnings): make_all once per
    process on the tree of `kind`, the warnings it raised recorded."""
    if kind in _FIGS:
        return _FIGS[kind]
    import ev_rec_config
    import ev_rec_figures as RF
    cfg, ecfg_path = _tree(kind)
    fdir = ev_rec_config.load(ecfg_path).sub("figures")
    shutil.rmtree(fdir, ignore_errors=True)          # no figure of an earlier run on disk
    lines = []
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index = RF.make_all(cfg, ecfg_path, log=lines.append)
    # ResourceWarning: block 21's Store.js leaves its JSON files to the
    # collector (not this block's code, and nothing on a figure)
    caught = [w for w in caught if not issubclass(w.category, ResourceWarning)]
    print("[r37] %s project: %d figures in %.0f s" % (kind, len(index), time.time() - t0),
          flush=True)
    _FIGS[kind] = (cfg, ecfg_path, index, lines, caught)
    return _FIGS[kind]


def _expected_figures(mdir):
    """The figures block 37's rules give for the metrics in `mdir` (the
    independent statement of R37a)."""
    def js(name):
        with open(os.path.join(mdir, name)) as fh:
            return json.load(fh)

    idx = js("METRICS.json")
    want = []
    for name in sorted(idx["ensembles"]):
        info = idx["ensembles"][name]
        base = "ensembles/%s/" % info["stem"]
        want += [base + n for n in ENSEMBLE_NAMES]
        for r in info["datasets"]:
            if r == "sim":
                continue
            want += [base + n % r for n in BANK_NAMES]
            jr = js("%s_%s.json" % (info["stem"], r))
            want += [base + CORNER_NAME % (r, k) for k in jr["classes"]]
    for c in idx.get("comparisons") or []:
        want += ["comparisons/%s/%s" % (c["name"], n) for n in COMPARISON_NAMES]
    if idx.get("comparisons"):
        want.append("across/dbar")
    return want


# ---------------------------------------------------------------- R37a

def _check_figures(kind):
    import ev_rec_config
    cfg, ecfg_path, index, lines, caught = _figured(kind)
    ecfg = ev_rec_config.load(ecfg_path)
    mdir, fdir = ecfg.sub("metrics"), ecfg.sub("figures")
    want = _expected_figures(mdir)
    got = [n for n, _c in index]
    assert len(got) == len(set(got)), "a figure drawn twice"
    assert set(got) == set(want), "%s: missing %s; extra %s" % (
        kind, sorted(set(want) - set(got))[:6], sorted(set(got) - set(want))[:6])
    assert not caught, "%s: warnings while drawing: %s" % (
        kind, ["%s: %s" % (w.category.__name__, w.message) for w in caught][:4])
    assert lines[-1].startswith("REC FIGURES OK: %d figure(s)" % len(want)), lines[-1]
    # FIGURES.md: one line per figure, in make_all's order, a caption each
    md = open(os.path.join(fdir, "FIGURES.md")).read()
    md.encode("ascii")
    fm = ",".join(ecfg.formats)
    listed = []
    for l in md.splitlines():
        if l.startswith("- `"):
            name = l[3:l.index(".{")]
            assert l[l.index(".{"):].startswith(".{%s}`: " % fm), l
            cap = l[l.index("`: ") + 3:]
            # block 28's caption rule: one sentence, at least 30 characters, a full stop
            assert cap.endswith(".") and len(cap) >= 30, l
            assert cap[:-1].rstrip(".") == dict(index)[name], (cap, dict(index)[name])
            listed.append(name)
    assert listed == got, "FIGURES.md does not list exactly the figures, in make_all's order"
    head = md.split("\n## ")[0]
    for need in (mdir, "%d dpi" % ecfg.dpi, ", ".join(ecfg.formats), "Left out of the metrics",
                 "Comparisons left out"):
        assert need in head, "FIGURES.md's header lacks %r" % need
    # the files on disk: exactly the figures, in every format; their signatures
    for f in ecfg.formats:
        on_disk = set(os.path.relpath(os.path.join(a, x), fdir)[:-len(f) - 1]
                      for a, _d, xs in os.walk(fdir) for x in xs if x.endswith("." + f))
        assert on_disk == set(want), (f, sorted(on_disk ^ set(want))[:6])
    sig = {"png": b"\x89PNG\r\n\x1a\n", "pdf": b"%PDF-"}
    for name in want:
        for f in ecfg.formats:
            with open(os.path.join(fdir, name + "." + f), "rb") as fh:
                assert fh.read(len(sig[f])) == sig[f], (name, f)
    # the corners at the class's example well (block 36's choice)
    caps = dict(index)
    for name in sorted(json.load(open(os.path.join(mdir, "METRICS.json")))["ensembles"]):
        info = json.load(open(os.path.join(mdir, "METRICS.json")))["ensembles"][name]
        for r in info["datasets"][1:]:
            jr = json.load(open(os.path.join(mdir, "%s_%s.json" % (info["stem"], r))))
            for ex in jr["example_wells"]:
                cap = caps["ensembles/%s/%s" % (info["stem"], CORNER_NAME % (r, ex["cls"]))]
                assert "class %d's example well %d of bank %s" % (ex["cls"], ex["well"], r) in cap, cap
    return len(want), ecfg.formats


class _FakeStore(object):
    """js/arr of a bank and of D_sim, built in memory (the scale check)."""

    def __init__(self, js, arr):
        self._js, self._arr = js, arr

    def js(self, name):
        return self._js[name]

    def arr(self, name):
        return self._arr[name]


def _scale_check():
    """The per-bank figures and a corner on a bank of the size of a real
    project: 14 axes, 300 wells in 4 classes and 5 batches, 1,500 windows,
    10 members; drawn without an error or a warning (the layout follows the
    number of axes, wells, classes and members, never the fixture's)."""
    import numpy as np
    import matplotlib.pyplot as plt
    import ev_rec_figures as RF
    rng = np.random.default_rng(SEED)
    p, G, C, B, M, per = 14, 300, 4, 5, 10, 5
    n = G * per
    well = np.repeat(np.arange(G), per)
    wcls = rng.integers(0, C, G)
    wbat = rng.integers(0, B, G)
    cls = wcls[well]
    q = np.sort(rng.random((G, 3, p)), axis=1)      # q05 <= q50 <= q95 (test data, not the module)
    flag = rng.random(n) < 0.05
    wells = [{"well": int(g), "cls": int(wcls[g]), "batch": int(wbat[g]), "n": per,
              "q05": q[g, 0].tolist(), "q50": q[g, 1].tolist(), "q95": q[g, 2].tolist(),
              "share_flagged": float(flag[well == g].sum()) / per} for g in range(G)]
    jr = {"wells": wells, "classes": list(range(C)),
          "gate": {"alpha": 0.05, "size_lo": 0.033, "size_hi": 0.071, "size_mean": 0.05,
                   "level": 0.95, "n_null": 500},
          "axes": {"names": ["axis_%d" % k for k in range(p)],
                   "bounds": [[-1.0, 2.0]] * p}, "pca": {"n_ref": 800}}
    ar = {"well": well, "cls": cls, "gate_p": rng.random(n) + 1e-3, "gate_flag": flag,
          "gate_r": rng.random((M, n)),
          "pca_coords": rng.normal(size=(M, 800 + n, 2)).astype(np.float32),
          "pca_evr": np.full((M, 2), 0.3),
          "pca_source": np.r_[np.zeros(800, dtype=np.int64), np.ones(n, dtype=np.int64)]}
    st = _FakeStore({"S_real": jr}, {"S_real": ar})
    e = {"name": "big:A2", "stem": "S", "arm": "A2"}
    RF.F21.style()
    tmp = _tmpdir("scale")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for fn in (RF.fig_gate_wells, RF.fig_gate_windows, RF.fig_gate_members, RF.fig_emb_pca,
                   RF.fig_rec_wells):
            fig, cap = fn(st, e, "real")
            fig.savefig(os.path.join(tmp, fn.__name__ + ".png"), dpi=40)
            plt.close(fig)
        draws = rng.random((per, 1000, p)).astype(np.float32)
        fig, cap = RF.fig_rec_corner(e, "real", {"cls": 0, "well": 0}, draws, wells[0],
                                     jr["axes"]["names"])
        fig.savefig(os.path.join(tmp, "corner.png"), dpi=40)
        plt.close(fig)
    caught = [w for w in caught if not issubclass(w.category, ResourceWarning)]
    assert not caught, ["%s: %s" % (w.category.__name__, w.message) for w in caught][:4]


def test_r37a_figures():
    """the set of figures, FIGURES.md, the PNG signatures, a figure to PDF (specs/SPEC.md block 37)."""
    import ev_rec_figures as RF
    # the module's names are the entry's
    assert RF.ENSEMBLE_FIGURES == ENSEMBLE_NAMES, RF.ENSEMBLE_FIGURES
    assert RF.BANK_FIGURES == BANK_NAMES and RF.CORNER_FIGURE == CORNER_NAME
    assert RF.COMPARISON_FIGURES == COMPARISON_NAMES and RF.ACROSS_FIGURES == ("dbar",)
    out = []
    for kind in ("recorded", "as_recorded"):
        n, fm = _check_figures(kind)
        out.append("%s %d (%s)" % (kind, n, ",".join(fm)))
    _scale_check()
    print("[r37a] figures as the rules give them: %s; each listed in FIGURES.md with its caption, "
          "none unlisted, signatures checked, no warning; the scale check (14 axes, 300 wells, 10 "
          "members) drawn" % "; ".join(out))


# ---------------------------------------------------------------- R37b

def _static_findings(src):
    """(imported top-level modules, called names): the AST of the figures'
    module."""
    tree = ast.parse(src)
    mods, calls = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= set(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
            calls |= set(a.name for a in n.names) & BANNED_CALLS     # from numpy import mean
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                calls.add(f.attr)
            elif isinstance(f, ast.Name):
                calls.add(f.id)
    return mods, calls


def test_r37b_static():
    """the figures' module computes nothing scientific; block 21's colours (specs/SPEC.md block 37)."""
    path = os.path.join(_EVAL, "ev_rec_figures.py")
    src = open(path).read()
    mods, calls = _static_findings(src)
    assert not (mods & BANNED_MODULES), "the figures' module imports %s" % (mods & BANNED_MODULES)
    assert not (calls & BANNED_CALLS), "the figures' module calls %s" % (calls & BANNED_CALLS)
    # the same at run time, transitively: importing it loads no statistics library
    code = ("import sys; sys.path.insert(0, %r); import ev_rec_figures; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in %r))"
            % (_EVAL, sorted(BANNED_MODULES)))
    r = subprocess.run([PYTHON, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "[]", r.stdout[-800:]
    # the colours are block 21's: none defined here, block 21's the validated ones
    own = sorted(set(re.findall(r"#[0-9a-fA-F]{6}\b", src)))
    assert not own, "the figures' module defines colours of its own: %s" % own
    import ev_figures as F21
    import ev_rec_figures as RF
    assert RF.F21 is F21
    assert F21.ARM_COLOR == ARM_COLOR, F21.ARM_COLOR
    assert tuple(F21.COND_COLOR) == COND_COLOR and len(set(F21.COND_MARKER)) == 3
    for c in range(6):
        assert RF._cls_style(c) == (COND_COLOR[c % 3], F21.COND_MARKER[c % 3]), c
    for arm in ARM_COLOR:
        assert F21.arm_color(arm) == ARM_COLOR[arm]
    print("[r37b] no statistics library imported (statically and at run time), no mean, SD, "
          "quantile or median called; block 21's colours, none of its own")


# ---------------------------------------------------------------- R37c

def _parse_tables(text):
    """[(specs, headers, rows)] of every table preceded by its cells comment."""
    import ev_rec_report as RR
    lines = text.splitlines()
    out = []
    for i, l in enumerate(lines):
        if not l.startswith(RR.CELLS_TAG):
            continue
        assert l.endswith(" -->"), l[:80]
        specs = json.loads(l[len(RR.CELLS_TAG):-4])["cells"]
        hdr = [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
        assert set(lines[i + 2].replace("|", "").replace("-", "")) == set(), lines[i + 2]
        rows, j = [], i + 3
        while j < len(lines) and lines[j].startswith("|"):
            rows.append([c.strip() for c in lines[j].strip().strip("|").split(" | ")])
            j += 1
        assert len(rows) == len(specs), "a table of %d rows has %d cell lists" % (len(rows),
                                                                                    len(specs))
        for r, s in zip(rows, specs):
            assert len(r) == len(hdr) == len(s), (hdr, r, s)
        out.append((specs, hdr, rows))
    # no table without its comment
    n_tables = sum(1 for i, l in enumerate(lines) if l.startswith("|---") or l.startswith("|--"))
    assert n_tables == len(out), "%d tables, %d with a cells comment" % (n_tables, len(out))
    return out


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _check_cell(text, spec, js, where):
    """One printed cell against the JSON value its spec names. Numbers at
    the printed precision: |printed - value| <= half a unit of the printed
    last digit (the rounding of %-formatting), with 1e-12 relative slack for
    the binary value of the decimal string."""
    def get(fp):
        obj = js(fp[0])
        for k in fp[1]:
            obj = obj[k]
        return obj

    if isinstance(spec, dict):
        if "ks_lt" in spec:
            v, lev = get(spec["ks_lt"]), get(spec["level"])
            want = "-" if v is None else ("rejected" if v < 1.0 - lev else "not rejected")
        elif "names_where" in spec:
            sel = [n for n, f in zip(get(spec["names"]), get(spec["names_where"])) if f]
            want = ", ".join(sel) if sel else "none"
        elif "verdict" in spec:
            j = js(spec["verdict"])
            want = ("resolved, %s lower" % j["lower_nll_ensemble"]) if j["resolved"] \
                else "not resolved"
        else:
            raise AssertionError("%s: unknown derived cell %r" % (where, spec))
        assert text == want, "%s: %r, its source gives %r" % (where, text, want)
        return 0
    v = get(spec)
    if v is None:
        assert text == "-", "%s: %r where the metrics hold null" % (where, text)
        return 0
    if isinstance(v, bool):
        assert text == ("yes" if v else "no"), "%s: %r for %r" % (where, text, v)
        return 0
    if isinstance(v, int):
        assert text == "%d" % v, "%s: %r for the integer %r" % (where, text, v)
        return 1
    if isinstance(v, float):
        d = decimal.Decimal(text)
        half = decimal.Decimal(5).scaleb(d.as_tuple().exponent - 1)
        assert abs(float(d) - v) <= float(half) + 1e-12 * abs(v), (
            "%s: printed %s, the metrics' value %r (half a unit %s)" % (where, text, v, half))
        return 1
    assert text == str(v), "%s: %r for %r" % (where, text, v)
    return 0


def _semantics(specs, hdr, rows, js, where):
    """The labels and headers name what the cells read (LABEL_KEY,
    HEADER_KEY: the test's own statement); a cell whose value is null is
    printed "-" and checked by _check_cell alone."""
    def value(sp):
        obj = js(sp[0])
        for k in sp[1]:
            obj = obj[k]
        return obj

    n = 0
    if hdr[:2] == ["quantity", "value"]:
        for r, s in zip(rows, specs):
            if r[0] in LABEL_KEY and isinstance(s[1], list):
                assert s[1][1][-1] == LABEL_KEY[r[0]], "%s: %r reads %r" % (where, r[0], s[1])
                n += 1
        return n
    for col, h in enumerate(hdr):
        want = HEADER_KEY.get(h)
        if want is None:
            continue
        ok = want if isinstance(want, tuple) else (want,)
        for r, s in zip(rows, specs):
            sp = s[col]
            if not isinstance(sp, list) or value(sp) is None:
                continue
            path = sp[1]
            key = path[-1]
            if isinstance(key, int) and h not in ("interval, lower end", "interval, upper end"):
                key = path[-2]                     # an axis's or a member's index
            assert key in ok, "%s: column %r reads %r" % (where, h, sp)
            n += 1
    return n


def _check_report(kind):
    import ev_rec_config
    import ev_rec_report as RR
    from ev_plan import sha256_file
    cfg, ecfg_path, _index, _lines, _w = _figured(kind)
    ecfg = ev_rec_config.load(ecfg_path)
    mdir = ecfg.sub("metrics")
    path = RR.write_report(cfg, ecfg_path, log=lambda t: None)
    assert path == os.path.join(ecfg.out_root, RR.REPORT_MD), path
    text = open(path).read()
    text.encode("ascii")
    cache = {}

    def js(f):
        if f not in cache:
            with open(os.path.join(mdir, f)) as fh:
                cache[f] = json.load(fh)
        return cache[f]

    n_num = n_sem = 0
    tables = _parse_tables(text)
    for t, (specs, hdr, rows) in enumerate(tables):
        for i, (r, s) in enumerate(zip(rows, specs)):
            for k, (cell, spec) in enumerate(zip(r, s)):
                where = "%s table %d (%s) row %d column %r" % (kind, t, hdr[0], i, hdr[k])
                if spec is None:
                    # every number of a table names its source
                    assert not _is_number(cell), "%s: the number %r names no source" % (where,
                                                                                       cell)
                    continue
                n_num += _check_cell(cell, spec, js, where)
        n_sem += _semantics(specs, hdr, rows, js, "%s table %d" % (kind, t))
        if hdr[:1] == ["well"] and "share flagged" in hdr:        # decreasing order of share
            sh = [float(r[hdr.index("share flagged")]) for r in rows]
            assert sh == sorted(sh, reverse=True), sh
    # the header's digests are the files'
    digests = re.findall(r"`([^`]+)` \(sha256 `([0-9a-f]{64})`\)", text)
    assert len(digests) == 3, digests
    for p, h in digests:
        assert sha256_file(p) == h, (p, h)
    assert os.path.abspath(cfg) in [p for p, _h in digests]
    assert ecfg.path in [p for p, _h in digests]
    # the wording (D-160; "flagged by the gate"; D-165; sec. 2.6's caution)
    fig_md = open(os.path.join(ecfg.sub("figures"), "FIGURES.md")).read()
    n_ens = len(js("METRICS.json")["ensembles"])
    for name, body in (("REC_REPORT.md", text), ("FIGURES.md", fig_md)):
        low = body.lower()
        assert "well's posterior" not in low and "misspecified" not in low, name
        assert "the mixture over the well's windows" in body, name
        assert "flagged by the gate" in body, name
    assert text.count(RR.CAUTION) == n_ens, "the caution once per ensemble"
    assert text.count("not against alpha_gate alone") == n_ens
    # every number from the metrics' JSON files alone: without the samples
    # and the metrics' .npz files the same report is written
    side = _tmpdir("side")
    moved = []
    try:
        shutil.move(ecfg.sub("samples"), os.path.join(side, "samples"))
        moved.append((os.path.join(side, "samples"), ecfg.sub("samples")))
        for f in os.listdir(mdir):
            if f.endswith(".npz"):
                shutil.move(os.path.join(mdir, f), os.path.join(side, f))
                moved.append((os.path.join(side, f), os.path.join(mdir, f)))
        again = open(RR.write_report(cfg, ecfg_path, log=lambda t: None)).read()
    finally:
        for a, b in moved:
            shutil.move(a, b)
    assert again == text, "the report changes without the samples and the .npz files"
    return n_num, n_sem, len(tables)


def test_r37c_report():
    """every number of the report equals the metrics' JSON value it names; the wording (specs/SPEC.md block 37)."""
    out = []
    for kind in ("recorded", "as_recorded"):
        n_num, n_sem, n_t = _check_report(kind)
        out.append("%s: %d numbers in %d tables, %d labels checked" % (kind, n_num, n_t, n_sem))
    print("[r37c] %s; the digests the files'; no \"well's posterior\" or \"misspecified\"; the "
          "same report without the samples and the .npz files" % "; ".join(out))


# ---------------------------------------------------------------- R37d

def _copy_samples(kind, tag):
    """An out_root of its own holding a copy of the tree's samples (block
    35's), and its configuration; nothing else."""
    import ev_rec_config
    cfg, ecfg_path = _tree(kind)
    src = ev_rec_config.load(ecfg_path).sub("samples")
    root = _tmpdir(tag)
    out = os.path.join(root, "out")
    shutil.copytree(src, os.path.join(out, "samples"))
    with open(ecfg_path) as fh:
        d = json.load(fh)
    return cfg, _write_ecfg(ecfg_path, out, "rec_eval.json", formats=d["formats"])


def _check_complete(cfg, ecfg_path, rec, lines):
    import ev_rec_config
    import ev_rec_plan
    import ev_rec_report as RR
    ecfg = ev_rec_config.load(ecfg_path)
    pre = ("REC METRICS OK", "REC FIGURES OK", "REC REPORT OK")
    hits = [next(i for i, l in enumerate(lines) if l.startswith(p)) for p in pre]
    assert hits == sorted(hits), "the pass lines out of order: %s" % hits
    assert lines[-1].startswith("REC REPORT OK"), lines[-1]
    path = os.path.join(ecfg.out_root, RR.REPORT_COMPLETE)
    with open(path) as fh:
        disk = json.load(fh)
    assert rec is None or disk == json.loads(json.dumps(rec)), "run's record and the file differ"
    assert tuple(sorted(disk)) == tuple(sorted(RR.COMPLETE_KEYS)), sorted(disk)
    assert disk["schema"] == "rec_eval_report/1"
    view = ev_rec_plan.load_view(cfg, ecfg)
    assert disk["inputs"] == ev_rec_plan.report_inputs(view, ecfg), "not the current inputs"
    assert disk["view_kind"] == view.kind
    assert [p_[:len(q)] for p_, q in zip(disk["pass_lines"], pre)] == list(pre), disk["pass_lines"]
    assert disk["pass_lines"] == [lines[i] for i in hits], (disk["pass_lines"], hits)
    with open(disk["metrics_index"]) as fh:
        idx = json.load(fh)
    assert disk["ensembles"] == sorted(idx["ensembles"]) and disk["left_out"] == idx["left_out"]
    assert disk["comparisons"] == [c["name"] for c in idx["comparisons"]]
    listed = [l for l in open(disk["figures_index"]).read().splitlines() if l.startswith("- `")]
    assert disk["n_figures"] == len(listed) > 0, (disk["n_figures"], len(listed))
    for k in ("report", "metrics_index", "figures_index"):
        assert os.path.isfile(disk["report"][k]), disk["report"]
    assert disk["report"]["report"] == os.path.join(ecfg.out_root, RR.REPORT_MD)
    assert sorted(disk["seconds"]) == ["figures", "metrics", "report"]
    # written last
    t_last = os.path.getmtime(path)
    for p_ in (disk["report"]["report"], disk["metrics_index"], disk["figures_index"]):
        assert os.path.getmtime(p_) <= t_last, p_
    return disk


def test_r37d_end_to_end():
    """ev_rec_report.py: the three pass lines, REC_REPORT_COMPLETE.json; no metrics refused (specs/SPEC.md block 37)."""
    import ev_rec_report as RR
    prog = os.path.join(_EVAL, "ev_rec_report.py")
    # the recorded-kind project through the command line
    cfg, ecfg_path = _copy_samples("recorded", "d1")
    r = subprocess.run([PYTHON, prog, "--config", cfg, "--eval-config", ecfg_path, "--ncpus", "1"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:]
    d1 = _check_complete(cfg, ecfg_path, None, r.stdout.strip().splitlines())
    # the as-recorded project through run() in the process
    cfg2, ecfg2 = _copy_samples("as_recorded", "d2")
    lines = []
    rec = RR.run(cfg2, ecfg2, ncpus=1, log=lines.append)
    d2 = _check_complete(cfg2, ecfg2, rec, lines)
    # no metrics: refused, exit 3, by both programs; the arguments: exit 2
    root = _tmpdir("d3")
    empty = _write_ecfg(ecfg_path, os.path.join(root, "nothing"), "none.json")
    r = subprocess.run([PYTHON, prog, "--config", cfg, "--eval-config", empty],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=600)
    assert r.returncode == 3 and "REFUSE: " in r.stdout, (r.returncode, r.stdout[-800:])
    assert not os.path.exists(os.path.join(root, "nothing", RR.REPORT_COMPLETE))
    r = subprocess.run([PYTHON, os.path.join(_EVAL, "ev_rec_figures.py"), "--config", cfg,
                        "--eval-config", empty], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=600)
    assert r.returncode == 3 and "REFUSE: no metrics at" in r.stdout, (r.returncode,
                                                                       r.stdout[-800:])
    try:
        RR.write_report(cfg, empty, log=lambda t: None)
        raise AssertionError("a report written without metrics")
    except RR.RefuseError as exc:
        assert "no metrics at" in str(exc), exc
    r = subprocess.run([PYTHON, prog, "--config", cfg], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    assert r.returncode == 2, (r.returncode, r.stdout[-400:])
    print("[r37d] ev_rec_report.py on the recorded project (%d figures, %d ensemble(s)) and run() "
          "on the as-recorded one (%d figures, %d ensemble(s), %d comparison(s)): the three pass "
          "lines in order, REC_REPORT_COMPLETE.json with the current inputs, written last; no "
          "metrics refused (exit 3), no arguments exit 2"
          % (d1["n_figures"], len(d1["ensembles"]), d2["n_figures"], len(d2["ensembles"]),
             len(d2["comparisons"])))


CHECKS = [("test_r37a_figures", test_r37a_figures),
          ("test_r37b_static", test_r37b_static),
          ("test_r37c_report", test_r37c_report),
          ("test_r37d_end_to_end", test_r37d_end_to_end)]


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
