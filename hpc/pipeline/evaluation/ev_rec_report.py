#!/usr/bin/env python3
"""Block 37: the report job of a recorded project's evaluation -- block
36's metrics, then the figures, then REC_REPORT.md and, last,
REC_REPORT_COMPLETE.json (specs/SPEC.md sec. 2.6 and block 37; decisions
D-159, D-160, D-165).

    python3 hpc/pipeline/evaluation/ev_rec_report.py --config <tag>.json \\
        --eval-config <tag>_rec_eval.json [--ncpus N]

Submitted by the launcher after the samplings (block 38:
submit_rec_eval.sh); it can also be run by hand. REC_REPORT.md states
numbers read from the metrics' JSON files alone and what each is; it draws
no conclusion about a culture, a condition or an arm, writes "the mixture
over the well's windows" and never "the well's posterior" (D-160), and
"flagged by the gate", never "misspecified".

Every table of REC_REPORT.md is preceded by an HTML comment (invisible in a
rendered page) that names, cell by cell, the metrics' JSON file and the key
each printed value was read from: {"cells": [[spec, ...], ...]}, one list
per row, `spec` null for a label or [file, [key, ...]] (or a derived
verdict, below), so that every number can be checked against its source
(R37c). The derived cells are the report's one comparison (block 37):
{"ks_lt": [file, path], "level": [file, path]}, TARP's verdict
"ks_p < 1 - level"; {"names_where": [file, path], "names": [file, path]},
the names of the axes whose flag is true; {"verdict": file}, block 20's
verdict line of a paired file.

Pass line: REC REPORT OK: ...; exit 2 on the arguments, 3 on a refusal (no
ensemble with a current sample, no metrics).

CONTRACT (wave C): the constants and the signatures below are fixed by
specs/SPEC.md block 37; the bodies are the implementer's.

Pure ASCII, LF only.
"""

import argparse
import datetime
import json
import os
import sys
import time

import _evalpath  # noqa: F401

import ev_rec_config
from ev_plan import RefuseError, sha256_file

EXIT_ARGS, EXIT_REFUSE = 2, 3
SCHEMA = "rec_eval_report/1"
REPORT_MD = "REC_REPORT.md"
REPORT_COMPLETE = "REC_REPORT_COMPLETE.json"

# the keys of REC_REPORT_COMPLETE.json (block 37, "the keys"); block 38's
# launcher reads `inputs`
COMPLETE_KEYS = (
    "schema", "inputs", "config", "eval_config", "view_kind", "ensembles", "left_out",
    "comparisons", "n_figures", "pass_lines", "metrics_index", "figures_index", "report",
    "seconds", "finished")

METRICS_INDEX = "METRICS.json"
BETA_KEYS = ("0.50", "0.68", "0.90", "0.95")      # block 36's coverage `at`
CELLS_TAG = "<!-- cells "                         # the comment before each table
# the printed precision of each kind of number; R37c compares at it
F_NLL, F_COV, F_FRAC, F_P, F_EPS = "%.4f", "%.3f", "%.4f", "%#.3g", "%.6f"

# fixed sentences (D-160, D-165, sec. 2.6)
CAUTION = ("Caution (sec. 2.6): an estimator equal to the simulator's posterior has C_E(beta) = "
           "beta, but it need not have C_E^edge(beta) = beta or C_E^int(beta) = beta, the "
           "expected coverage being an average over the prior and not over a part of it; the "
           "split shows where a deficit sits, read through (R2), and is not two further "
           "calibration tests.")
SHARE_LINE = ("A share of windows flagged by the gate is read against the size band -- the range "
              "in which the flagged share of independent windows of the simulator's law lies, "
              "for a null set of this size -- and not against alpha_gate alone; the windows of "
              "one well share a culture, and no test is made on a share (D-165; sec. 8, Q-R1).")
MIXTURE_LINE = ("Each row below is the mixture over the well's windows (R4): the law of a draw "
                "taken from a window of the well picked uniformly at random, every window "
                "weighing the same. It describes where the windows of the well put their mass; "
                "it is not the posterior of the well's parameter given all its windows, and its "
                "width does not shrink with the number of windows (D-160). Values are in the "
                "unit box.")


# ---------------------------------------------------------------- reading

def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _get(obj, path):
    for k in path:
        obj = obj[k]
    return obj


class _Metrics(object):
    """The metrics' JSON files (block 36's keys), by base name; never an
    .npz and never a sample file."""

    def __init__(self, mdir):
        self.mdir = mdir
        self.index = _read_json(os.path.join(mdir, METRICS_INDEX))
        self._js = {}

    def js(self, fname):
        if fname not in self._js:
            self._js[fname] = _read_json(os.path.join(self.mdir, fname))
        return self._js[fname]


# ---------------------------------------------------------------- formatting

def fmt_value(v, spec=F_FRAC):
    """A JSON value as the report prints it: null as '-', a bool as yes/no,
    an int as itself, a float with `spec`, a string as it is."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return "%d" % v
    if isinstance(v, float):
        return spec % v
    return str(v).replace("|", "/")


def tarp_verdict(ks_p, level):
    """TARP's verdict for one reference mode (block 37, wave C's contract):
    'ks_p < 1 - level', ev_rec_validate's definition; '-' when ks_p is
    null."""
    if ks_p is None:
        return "-"
    return "rejected" if ks_p < 1.0 - level else "not rejected"


def names_where(flags, names):
    sel = [n for n, f in zip(names, flags) if f]
    return ", ".join(sel) if sel else "none"


def paired_verdict(j):
    """Block 20's verdict line of a paired file."""
    return ("resolved, %s lower" % j["lower_nll_ensemble"]) if j["resolved"] else "not resolved"


class _Table(object):
    """A Markdown table whose cells each name their source (CELLS_TAG)."""

    def __init__(self, M, headers):
        self.M = M
        self.headers = headers
        self.rows, self.specs = [], []

    def row(self, cells):
        """cells: each a label (a str, printed as it is, spec null) or a
        tuple (file, path[, format]) read from the metrics, or a dict of a
        derived verdict."""
        out, specs = [], []
        for c in cells:
            if isinstance(c, str):
                out.append(c)
                specs.append(None)
            elif isinstance(c, dict):
                out.append(self._derived(c))
                specs.append(c)
            else:
                fname, path = c[0], list(c[1])
                spec = c[2] if len(c) > 2 else F_FRAC
                out.append(fmt_value(_get(self.M.js(fname), path), spec))
                specs.append([fname, path])
        self.rows.append(out)
        self.specs.append(specs)

    def _derived(self, c):
        if "ks_lt" in c:
            return tarp_verdict(_get(self.M.js(c["ks_lt"][0]), c["ks_lt"][1]),
                                _get(self.M.js(c["level"][0]), c["level"][1]))
        if "names_where" in c:
            return names_where(_get(self.M.js(c["names_where"][0]), c["names_where"][1]),
                               _get(self.M.js(c["names"][0]), c["names"][1]))
        if "verdict" in c:
            return paired_verdict(self.M.js(c["verdict"]))
        raise ValueError("unknown derived cell %r" % (c,))

    def lines(self):
        # "-" escaped so that the comment never holds "--" (an HTML comment's end)
        blob = json.dumps({"cells": self.specs}, sort_keys=True,
                          separators=(",", ":")).replace("-", "\\u002d")
        out = [CELLS_TAG + blob + " -->",
               "| " + " | ".join(self.headers) + " |",
               "|" + "---|" * len(self.headers)]
        out += ["| " + " | ".join(r) + " |" for r in self.rows]
        out.append("")
        return out


# ---------------------------------------------------------------- the sections

def _header(M, cfg_path, ecfg, mdir):
    idx = M.index
    try:
        project = _read_json(cfg_path).get("dataset") or os.path.basename(cfg_path)
    except (OSError, ValueError):
        project = os.path.basename(cfg_path)
    reported = sorted(idx.get("ensembles") or {})
    left = idx.get("left_out") or []
    cleft = idx.get("comparisons_left_out") or []
    out = ["# The evaluation of a recorded project: %s" % project, "",
           "Written by `hpc/pipeline/evaluation/ev_rec_report.py` (block 37) from the metrics' "
           "JSON files alone (block 36). Every number below is a value of those files; the "
           "comment before each table names the file and the key of each cell.", "",
           "- Project: `%s`" % project,
           "- Configuration: `%s` (sha256 `%s`)" % (os.path.abspath(cfg_path),
                                                    sha256_file(cfg_path)),
           "- Evaluation's configuration: `%s` (sha256 `%s`)" % (ecfg.path,
                                                                 sha256_file(ecfg.path)),
           "- Metrics' index: `%s` (sha256 `%s`)" % (os.path.join(mdir, METRICS_INDEX),
                                                     sha256_file(os.path.join(mdir,
                                                                              METRICS_INDEX))),
           "- Figures: `%s`" % os.path.join(ecfg.sub("figures"), "FIGURES.md"),
           "- The view's kind: %s" % idx.get("view_kind"),
           "- Ensembles reported: %s" % (", ".join(reported) or "none"),
           "- Ensembles left out: %s" % ("; ".join("%s (%s)" % (x["ensemble"], x["why"])
                                                    for x in left) or "none"),
           "- Comparisons made: %s" % (", ".join(c["name"] for c in idx.get("comparisons") or [])
                                       or "none"),
           "- Comparisons left out: %s" % ("; ".join("%s (%s)" % (x["comparison"], x["why"])
                                                      for x in cleft) or "none"),
           "",
           "Words used: \"flagged by the gate\" is p(x) < alpha_gate (R8), a statement about the "
           "window's embedding and the gate's null set only; \"the mixture over the well's "
           "windows\" is (R4). The report states numbers and what each is; it draws no "
           "conclusion about a culture, a condition or an arm.", ""]
    return out


def _ensemble(M, name, info):
    stem = info["stem"]
    S = "%s_sim.json" % stem
    js = M.js(S)
    banks = [D for D in info["datasets"] if D != "sim"]
    p = js["p"]
    out = ["## Ensemble %s" % name, "",
           "Arm %s, search %s (campaign %s); %d member(s), %d draws per row; D_sim: %d rows, %d "
           "in the box, %d calibration rows in the box (S_cal), %d draw rows, %d networks; "
           "recording banks: %s." % (info["arm"], info["search"] or "-",
                                     info["campaign"] or "-", js["M"],
                                     js["n_post"], js["n"], js["n_in_box"], js["n_cal"],
                                     js["n_d"], js["n_networks"], ", ".join(banks) or "none"),
           ""]
    # the NLL
    out += ["### The NLL on D_sim (nats per row)", ""]
    t = _Table(M, ["rows", "number of rows", "mean NLL"])
    t.row(["every row in the box", (S, ["nll", "n_rows"]), (S, ["nll", "mean"], F_NLL)])
    t.row(["interior rows", (S, ["edge", "nll_int_n"]), (S, ["edge", "nll_int_mean"], F_NLL)])
    t.row(["edge rows (R1)", (S, ["edge", "nll_edge_n"]),
           (S, ["edge", "nll_edge_mean"], F_NLL)])
    for m in range(len(js["nll"]["members"])):
        t.row(["member %d, every row in the box" % (m + 1), (S, ["nll", "n_rows"]),
               (S, ["nll", "members", m], F_NLL)])
    t.row(["SD of the members' means", "-", (S, ["nll", "members_sd"], F_NLL)])
    out += t.lines()
    # the expected coverage and its edge split
    out += ["### The expected coverage on D_sim", ""]
    t = _Table(M, ["curve", "rows"] + ["C at %s" % b for b in BETA_KEYS]
               + ["signed area", "max abs deviation"])
    for label, n, base in (("C_E (S_cal)", ["coverage", "n_rows"], ["coverage"]),
                           ("C_E^int", ["edge", "n_int"], ["edge", "interior"]),
                           ("C_E^edge", ["edge", "n_edge"], ["edge", "edge"])):
        cur = _get(js, base)
        if cur is None:
            t.row([label, (S, n)] + [(S, base)] * 6)       # null: printed "-"
        else:
            t.row([label, (S, n)] + [(S, base + ["at", b], F_COV) for b in BETA_KEYS]
                  + [(S, base + ["signed_area"], F_FRAC), (S, base + ["max_abs_dev"], F_FRAC)])
    out += t.lines()
    t = _Table(M, ["quantity", "value"])
    t.row(["pi_edge, the share of edge rows in S_cal", (S, ["edge", "pi_edge"], F_FRAC)])
    t.row(["its value under the box prior, 1 - (1 - 2 eps)^p",
           (S, ["edge", "pi_edge_expected"], F_FRAC)])
    t.row(["eps, the edge width", (S, ["edge", "eps"], F_EPS)])
    t.row(["p, the axes", (S, ["edge", "p"])])
    t.row(["largest absolute difference between the two sides of (R2)",
           (S, ["edge", "r2_max_abs_dev"], F_P)])
    out += t.lines()
    out += [CAUTION, ""]
    # SBC and TARP
    out += ["### SBC and TARP over the calibration rows", ""]
    t = _Table(M, ["quantity", "value"])
    t.row(["calibration rows", (S, ["sbc", "n_rows"])])
    t.row(["Holm family's level", (S, ["sbc", "holm_alpha"], F_FRAC)])
    t.row(["axes rejected by SBC (Holm)", {"names_where": [S, ["sbc", "holm_rejected"]],
                                            "names": [S, ["recovery", "axes"]]}])
    t.row(["number of axes rejected", (S, ["sbc", "n_rejected"])])
    out += t.lines()
    t = _Table(M, ["TARP reference", "mode used", "KS p", "max deviation",
                   "verdict (KS p < 1 - level)"])
    for mode in ("x", "random"):
        t.row([mode, (S, ["tarp", mode, "reference_mode"]), (S, ["tarp", mode, "ks_p"], F_P),
               (S, ["tarp", mode, "max_dev"], F_FRAC),
               {"ks_lt": [S, ["tarp", mode, "ks_p"]], "level": [S, ["level"]]}])
    out += t.lines()
    # per axis: SBC, the contraction on D_sim and on each bank
    out += ["### Per axis: SBC and the contraction", "",
            "The contraction on D_sim is 1 - (mean sigma^2 over the calibration rows) / (1/12); "
            "on a recording bank, chi (R3) per window, its mean and its 5 % and 95 % quantiles "
            "over the bank's windows.", ""]
    hdr = ["axis", "SBC KS p", "Holm-adjusted p", "rejected", "contraction, D_sim"]
    for r in banks:
        hdr += ["%s: mean chi" % r, "%s: chi 5 %%" % r, "%s: chi 95 %%" % r]
    t = _Table(M, hdr)
    for k in range(p):
        cells = [(S, ["recovery", "axes", k]), (S, ["sbc", "ks_p", k], F_P),
                 (S, ["sbc", "holm_adjusted_p", k], F_P), (S, ["sbc", "holm_rejected", k]),
                 (S, ["recovery", "contraction", k], F_FRAC)]
        for r in banks:
            B = "%s_%s.json" % (stem, r)
            cells += [(B, ["posterior", "contraction_mean", k], F_FRAC),
                      (B, ["posterior", "contraction_q05", k], F_FRAC),
                      (B, ["posterior", "contraction_q95", k], F_FRAC)]
        t.row(cells)
    out += t.lines()
    # the gate
    out += ["### The gate", ""]
    t = _Table(M, ["quantity", "value"])
    for label, path, f in (("alpha_gate", ["gate", "alpha"], F_FRAC),
                           ("N_ref, reference rows", ["gate", "n_ref"], None),
                           ("N_rank, rank rows", ["gate", "n_rank"], None),
                           ("N_null, null rows", ["gate", "n_null"], None),
                           ("N_chk, check rows", ["gate", "n_check"], None),
                           ("k_gate (R9)", ["gate", "k_gate"], None),
                           ("size band's mean, k_gate / (N_null + 1)", ["gate", "size_mean"], F_FRAC),
                           ("size band, lower end", ["gate", "size_lo"], F_FRAC),
                           ("size band, upper end", ["gate", "size_hi"], F_FRAC),
                           ("size band's level", ["gate", "level"], F_FRAC),
                           ("check rows flagged by the gate", ["gate", "check", "n_flagged"], None),
                           ("check rows", ["gate", "check", "n"], None),
                           ("check rows' share flagged", ["gate", "check", "share"], F_FRAC),
                           ("its exact interval, lower end", ["gate", "check", "ci", 0], F_FRAC),
                           ("its exact interval, upper end", ["gate", "check", "ci", 1], F_FRAC),
                           ("the interval's level", ["gate", "check", "level"], F_FRAC)):
        t.row([label, (S, path, f) if f else (S, path)])
    out += t.lines()
    out += [SHARE_LINE, ""]
    for r in banks:
        B = "%s_%s.json" % (stem, r)
        jr = M.js(B)
        out += ["#### Bank %s: windows flagged by the gate" % r, ""]
        t = _Table(M, ["windows", "flagged", "share flagged", "size band, lower end",
                       "size band, upper end", "size band's mean", "alpha_gate"])
        t.row([(B, ["gate", "n"]), (B, ["gate", "n_flagged"]), (B, ["gate", "share"], F_FRAC),
               (B, ["gate", "size_lo"], F_FRAC), (B, ["gate", "size_hi"], F_FRAC),
               (B, ["gate", "size_mean"], F_FRAC), (B, ["gate", "alpha"], F_FRAC)])
        out += t.lines()
        t = _Table(M, ["class", "windows", "flagged", "share flagged"])
        for i in range(len(jr["gate"]["per_class"])):
            t.row([(B, ["gate", "per_class", i, "cls"]), (B, ["gate", "per_class", i, "n"]),
                   (B, ["gate", "per_class", i, "n_flagged"]),
                   (B, ["gate", "per_class", i, "share"], F_FRAC)])
        out += t.lines()
        out += ["The wells, in decreasing order of their share of windows flagged by the gate "
                "(ties by the well's id):", ""]
        wells = jr["wells"]
        order = sorted(range(len(wells)), key=lambda i: (-wells[i]["share_flagged"],
                                                         wells[i]["well"]))
        t = _Table(M, ["well", "class", "batch", "windows", "flagged", "share flagged",
                       "mean T"])
        for i in order:
            t.row([(B, ["wells", i, "well"]), (B, ["wells", i, "cls"]),
                   (B, ["wells", i, "batch"]), (B, ["wells", i, "n"]),
                   (B, ["wells", i, "n_flagged"]), (B, ["wells", i, "share_flagged"], F_FRAC),
                   (B, ["wells", i, "mean_T"], F_FRAC)])
        out += t.lines()
    # the wells' mixtures
    for r in banks:
        B = "%s_%s.json" % (stem, r)
        jr = M.js(B)
        wells = jr["wells"]
        names, bounds = jr["axes"]["names"], jr["axes"]["bounds"]
        order = sorted(range(len(wells)), key=lambda i: (wells[i]["cls"], wells[i]["batch"],
                                                         wells[i]["well"]))
        out += ["### Bank %s: the mixture over the well's windows, per axis" % r, "",
                MIXTURE_LINE, ""]
        for k in range(len(names)):
            rng = ("; its declared range [%s, %s] in its declared coordinate, a quantile u "
                   "there being lo + u (hi - lo)" % (fmt_value(bounds[k][0], "%.6g"),
                                                     fmt_value(bounds[k][1], "%.6g"))
                   if bounds is not None else "")
            out += ["Axis %s%s." % (names[k], rng), ""]
            t = _Table(M, ["well", "class", "batch", "windows", "median", "5 %", "95 %", "mean",
                           "SD"])
            for i in order:
                t.row([(B, ["wells", i, "well"]), (B, ["wells", i, "cls"]),
                       (B, ["wells", i, "batch"]), (B, ["wells", i, "n"]),
                       (B, ["wells", i, "q50", k], F_FRAC), (B, ["wells", i, "q05", k], F_FRAC),
                       (B, ["wells", i, "q95", k], F_FRAC), (B, ["wells", i, "mean", k], F_FRAC),
                       (B, ["wells", i, "sd", k], F_FRAC)])
            out += t.lines()
    return out


def _comparisons(M):
    idx = M.index
    comps = idx.get("comparisons") or []
    out = ["## The comparisons (on D_sim)", ""]
    if not comps:
        out += ["No comparison was made.", ""]
    else:
        out += ["For each comparison (E, E_ref): Dbar, the mean over the simulated networks of "
                "the paired difference of the two ensembles' NLLs (E3), its bootstrap interval "
                "over the networks, 2 sigma_seed and the two clauses, with block 20's verdict "
                "line; nothing more is derived from them (D-163 (3)).", ""]
        t = _Table(M, ["comparison", "E", "E_ref", "rows", "networks", "Dbar (nats)",
                       "interval, lower end", "interval, upper end", "level", "2 sigma_seed",
                       "interval excludes 0", "Dbar beyond 2 sigma_seed", "verdict"])
        for c in comps:
            P = "paired_%s_sim.json" % c["name"]
            t.row([(P, ["comparison"]),
                   (P, ["E"]), (P, ["E_ref"]), (P, ["n_rows"]), (P, ["n_donors"]),
                   (P, ["Dbar"], "%+.4f"), (P, ["ci", 0], "%+.4f"), (P, ["ci", 1], "%+.4f"),
                   (P, ["level"], F_FRAC), (P, ["two_sigma_seed"], F_NLL),
                   (P, ["ci_excludes_0"]), (P, ["beyond_2_sigma_seed"]), {"verdict": P}])
        out += t.lines()
    left = idx.get("comparisons_left_out") or []
    if left:
        out += ["Comparisons not made:", ""]
        out += ["- %s: %s" % (x["comparison"], x["why"]) for x in left]
        out.append("")
    return out


def write_report(cfg_path, ecfg_path, log=print):
    """Writes <out_root>/REC_REPORT.md from <out_root>/metrics/METRICS.json
    and the JSON files it names (never from a sample file or an .npz);
    returns its path. ev_plan.RefuseError when there are no metrics."""
    ecfg = ev_rec_config.load(ecfg_path)
    mdir = ecfg.sub("metrics")
    if not os.path.isfile(os.path.join(mdir, METRICS_INDEX)):
        raise RefuseError("no metrics at %s (block 36's ev_rec_metrics.py first)" % mdir)
    M = _Metrics(mdir)
    ens = M.index.get("ensembles") or {}
    if not ens:
        raise RefuseError("%s lists no ensemble" % os.path.join(mdir, METRICS_INDEX))
    lines = _header(M, cfg_path, ecfg, mdir)
    for name in sorted(ens):
        lines += _ensemble(M, name, ens[name])
    lines += _comparisons(M)
    text = "\n".join(lines)
    text.encode("ascii")                      # the report is ASCII (it ships with the run)
    path = os.path.join(ecfg.out_root, REPORT_MD)
    with open(path + ".tmp", "w") as fh:
        fh.write(text)
    os.replace(path + ".tmp", path)
    log("report: %s (%d ensemble(s), %d line(s))" % (path, len(ens), len(lines)))
    return path


def run(cfg_path, ecfg_path, ncpus=None, log=print):
    """The report job: ev_rec_plan.report_inputs read before anything is
    computed; ev_rec_metrics.compute; ev_rec_figures.make_all;
    write_report; then REC_REPORT_COMPLETE.json (COMPLETE_KEYS), written to
    a temporary name and renamed. Returns that record."""
    import ev_rec_figures
    import ev_rec_metrics
    import ev_rec_plan
    t0 = time.time()
    ecfg = ev_rec_config.load(ecfg_path)
    view = ev_rec_plan.load_view(cfg_path, ecfg)
    inputs = ev_rec_plan.report_inputs(view, ecfg)       # before anything is read or computed
    seen = []

    def tee(t):
        seen.append(t)
        log(t)

    def pass_line(prefix):
        hit = [t for t in seen if t.startswith(prefix)]
        if not hit:
            raise RefuseError("no %r line was printed" % prefix)
        return hit[-1]

    idx = ev_rec_metrics.compute(cfg_path, ecfg.path, ncpus=ncpus, log=tee)
    t1 = time.time()
    figs = ev_rec_figures.make_all(cfg_path, ecfg.path, log=tee)
    t2 = time.time()
    md = write_report(cfg_path, ecfg.path, log=tee)
    t3 = time.time()
    mpath = os.path.join(ecfg.sub("metrics"), METRICS_INDEX)
    fpath = os.path.join(ecfg.sub("figures"), ev_rec_figures.FIGURES_INDEX)
    own = ("REC REPORT OK: %d ensemble(s), %d comparison(s), %d figure(s), %d left out; %s"
           % (len(idx["ensembles"]), len(idx["comparisons"]), len(figs), len(idx["left_out"]),
              md))
    rec = {"schema": SCHEMA, "inputs": inputs, "config": os.path.abspath(cfg_path),
           "eval_config": ecfg.path, "view_kind": view.kind,
           "ensembles": sorted(idx["ensembles"]), "left_out": idx["left_out"],
           "comparisons": [c["name"] for c in idx["comparisons"]], "n_figures": len(figs),
           "pass_lines": [pass_line("REC METRICS OK"), pass_line("REC FIGURES OK"), own],
           "metrics_index": mpath, "figures_index": fpath,
           "report": {"report": md, "metrics_index": mpath, "figures_index": fpath},
           "seconds": {"metrics": round(t1 - t0, 1), "figures": round(t2 - t1, 1),
                       "report": round(t3 - t2, 1)},
           "finished": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}
    assert tuple(sorted(rec)) == tuple(sorted(COMPLETE_KEYS)), sorted(rec)
    path = os.path.join(ecfg.out_root, REPORT_COMPLETE)
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)                       # last
    log(own)
    return rec


def main(argv=None):
    """The command line: 0 with the pass line, EXIT_ARGS, EXIT_REFUSE."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the project's configuration (<tag>.json)")
    ap.add_argument("--eval-config", required=True,
                    help="the evaluation's configuration (<tag>_rec_eval.json)")
    ap.add_argument("--ncpus", type=int, default=None, help="default: the configuration's ncpus")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return EXIT_ARGS if exc.code else 0
    if args.ncpus is not None and args.ncpus < 1:
        print("--ncpus must be >= 1, got %d" % args.ncpus, flush=True)
        return EXIT_ARGS
    try:
        run(args.config, args.eval_config, ncpus=args.ncpus, log=lambda t: print(t, flush=True))
    except (RefuseError, ev_rec_config.RecEvalConfigError) as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
