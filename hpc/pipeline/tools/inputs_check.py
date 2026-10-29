#!/usr/bin/env python3
"""The inputs check (block 9 of specs/SPEC.md): a read-only look at a data
set's inputs before its first submission.

    python3 hpc/pipeline/tools/inputs_check.py --project <tag> \\
        [--max-files N] [--quiet-hours H] [--json OUT] [--no-qstat]

Run on the login node in the configuration's envs.extract (numpy). Reads the
configuration (the driver's own validation), the simulation campaign under
sim.sim_main (its sweep tasks' job_args.json and manifest.json, the names of
their iteration files, the modification times of their folders, and the 0-d
members of at most N raw iteration files, one per topology), the real arm's
cohort manifest, sim.out_root, and from the Sbi-extractor checkout the S2
profile and sbi_labels. Writes nothing (but --json).

Every check prints one line, OK / FAIL / WARN / NOTE: a FAIL means a stage
would refuse, or run on inputs the configuration does not describe; a WARN
is for the user to read (above all: the campaign changed recently, so a job
may still be writing it -- D-060, D-073). When sim.plan_args gives S2's plan
--quiet-hours H (D-112), the tasks changed less than H hours ago are the
ones the plan leaves out: listed, kept out of the sample, and every task so
young is a FAIL (the plan refuses); the pace of the iteration files (the
gaps between two files of one task) says whether H is long enough. The job
protocol (D-113): the SBI checkout's jobs/stage_job.sh and the Sbi-extractor
checkout's side of it (env.sh's sbix_pipeline_v, run_mea_task.pbs, the job
scripts) must be there. The last line:

    INPUTS CHECK: <f> FAIL, <w> WARN

Exit 0 without a FAIL, 1 with one, 2 on a refused configuration or bad
arguments.

Pure ASCII, LF only; Python 3.6 syntax; numpy.
"""

import argparse
import getpass
import glob
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
import config as C          # noqa: E402

# Sbi-extractor's own patterns (sim_reextract_plan.SWEEP_RE, bank_plan_load,
# bank_plan.CAMPAIGN_RE): copied, since that repository is not a package
SWEEP_RE = re.compile(r"^sweep_[^_]+_task\d+$")
TOPO_RE = re.compile(r"^topo_(\d+)$")
ITER_RE = re.compile(r"^iter_(\d+)\.npz$")
CAMPAIGN_RE = re.compile(r"_v(\d+)$")
BOX_RTOL = 1e-12            # float64 roundoff on a JSON-read bound (the bank plan's)
EXIT_FAIL, EXIT_USAGE = 1, 2
GAP_FILES_CAP = 20000       # iteration files whose times the pace reads, at most
# D-113: the Sbi-extractor job scripts that start through the job protocol
SBIX_PROTOCOL_SCRIPTS = ("export_windows/run_bank_task.pbs", "export_windows/run_bank_gate.pbs",
                         "sim_reextract/run_sim_reextract_gate.pbs", "sim_reextract/run_mea_task.pbs",
                         "extractor/run_extractor_array_mea.pbs", "extractor/run_cohort_manifest.pbs")


class Findings(object):
    """The checks' lines, printed as they come."""

    def __init__(self, out=None):
        self.rows = []
        self.out = out if out is not None else sys.stdout    # at call time, not at import

    def add(self, status, label, text):
        self.rows.append({"status": status, "check": label, "text": text})
        self.out.write("%-4s  %-28s %s\n" % (status, label, text))

    def count(self, status):
        return sum(1 for r in self.rows if r["status"] == status)


def _read_json(path):
    try:
        with open(path, "r") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _mtime(path):
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def scan_campaign(sim_main, campaign_glob):
    """The sweep tasks of the campaigns under sim_main matching the glob.

    Each task: campaign, sweep, dir, job_args (dict or None), manifest (dict
    or None), topo {k: sorted iteration indices}, newest (the latest
    modification time of the task folder, its topo folders and its JSON)."""
    camps = sorted(d for d in glob.glob(os.path.join(sim_main, campaign_glob)) if os.path.isdir(d))
    tasks = []
    for cd in camps:
        for sw in sorted(os.listdir(cd)):
            td = os.path.join(cd, sw)
            if not (SWEEP_RE.match(sw) and os.path.isdir(td)):
                continue
            times = [_mtime(td)]
            ja_p, mf_p = os.path.join(td, "job_args.json"), os.path.join(td, "manifest.json")
            times += [_mtime(ja_p), _mtime(mf_p)]
            topo = {}
            for e in sorted(os.listdir(td)):
                m = TOPO_RE.match(e)
                p = os.path.join(td, e)
                if m is None or not os.path.isdir(p):
                    continue
                times.append(_mtime(p))
                topo[int(m.group(1))] = sorted(int(m2.group(1)) for m2 in
                                               (ITER_RE.match(n) for n in os.listdir(p)) if m2)
            tasks.append({"campaign": os.path.basename(cd), "sweep": sw, "dir": td,
                          "job_args": _read_json(ja_p) if os.path.isfile(ja_p) else None,
                          "manifest": _read_json(mf_p) if os.path.isfile(mf_p) else None,
                          "topo": topo, "newest": max(t for t in times if t is not None)})
    return [os.path.basename(c) for c in camps], tasks


def plan_quiet_hours(plan_args):
    """S2's --quiet-hours from sim.plan_args: (hours, None), (0.0, None)
    when absent, or (None, the reason) when it cannot be read (D-112)."""
    try:
        words = shlex.split(plan_args or "")
    except ValueError as exc:
        return None, "sim.plan_args cannot be split: %s" % exc
    value = None
    for i, w in enumerate(words):
        if w == "--quiet-hours":
            if i + 1 >= len(words):
                return None, "--quiet-hours without a value"
            value = words[i + 1]
        elif w.startswith("--quiet-hours="):
            value = w.split("=", 1)[1]
    if value is None:
        return 0.0, None
    try:
        h = float(value)
    except ValueError:
        return None, "--quiet-hours %r is not a number" % value
    if not (math.isfinite(h) and h >= 0):
        return None, "--quiet-hours %r: a positive number of hours, or 0" % value
    return h, None


def iteration_gaps(tasks, cap=GAP_FILES_CAP):
    """The pace of the iteration files: the gaps (hours) between two files
    of one task written one after the other, by modification time, over
    the tasks in order until `cap` files are read. Returns (gaps, the
    longest gap's task, the number of files read)."""
    gaps, worst, worst_gap, n = [], None, -1.0, 0
    for t in tasks:
        times = []
        for k in sorted(t["topo"]):
            for i in t["topo"][k]:
                if n >= cap:
                    break
                tm = _mtime(os.path.join(t["dir"], "topo_%05d" % k, "iter_%05d.npz" % i))
                n += 1
                if tm is not None:
                    times.append(tm)
        times.sort()
        for a, b in zip(times, times[1:]):
            g = (b - a) / 3600.0
            gaps.append(g)
            if g > worst_gap:
                worst_gap, worst = g, "%s/%s" % (t["campaign"], t["sweep"])
        if n >= cap:
            break
    return gaps, worst, n


def sample_files(tasks, max_files):
    """The first iteration of each topology, round-robin over the tasks in
    sorted order, at most max_files: [(task index, k, path)]."""
    queues = [[(n, k, os.path.join(t["dir"], "topo_%05d" % k, "iter_%05d.npz" % t["topo"][k][0]))
               for k in sorted(t["topo"]) if t["topo"][k]] for n, t in enumerate(tasks)]
    out = []
    while len(out) < max_files and any(queues):
        for q in queues:
            if q and len(out) < max_files:
                out.append(q.pop(0))
    return out


def read_scalars(path):
    """{member: value} for the 0-d members of an .npz, without decompressing
    the others (their headers are read, not their data)."""
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for name in z.files:
            with z.zip.open(name + ".npy") as fh:
                version = np.lib.format.read_magic(fh)
                if version == (1, 0):
                    shape, _f, _d = np.lib.format.read_array_header_1_0(fh)
                else:
                    shape, _f, _d = np.lib.format.read_array_header_2_0(fh)
            if shape != ():
                continue
            v = z[name][()]
            if isinstance(v, (bool, np.bool_)):
                out[name] = bool(v)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                out[name] = float(v)
            else:
                out[name] = str(v)
    return out


def _single(values):
    """The distinct JSON values of a list (by canonical JSON), in first order."""
    seen, out = set(), []
    for v in values:
        key = json.dumps(v, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _finite(v):
    return (not isinstance(v, bool)) and isinstance(v, (int, float)) and math.isfinite(v)


def _sha_ok(path):
    side = path + ".sha256"
    if not (os.path.isfile(path) and os.path.isfile(side)):
        return False
    with open(side, "rb") as fh:
        tokens = fh.read().decode("ascii", "replace").split()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return bool(tokens) and tokens[0].lower() == h.hexdigest()


def _sbi_labels(sbix_dir):
    """Sbi-extractor's sbi_labels, or (None, the reason)."""
    if sbix_dir not in sys.path:
        sys.path.insert(0, sbix_dir)
    try:
        import sbi_labels as SL     # noqa: E402
        return SL, None
    except Exception as exc:       # noqa: BLE001 -- reported, never hidden
        return None, "%s: %s" % (type(exc).__name__, exc)


def run_checks(cfg, max_files, quiet_hours, use_qstat, out=None):
    out = out if out is not None else sys.stdout              # at call time, not at import
    d = cfg.data
    F = Findings(out)
    b, s = d["bank"], d["sim"]
    out.write("inputs check: %s (%s), sha256 %s\n" % (d["dataset"], cfg.path, cfg.sha256[:16]))

    # --- the campaign ------------------------------------------------------
    camps, tasks = scan_campaign(s["sim_main"], s["campaign_glob"])
    n_topo = sum(len(t["topo"]) for t in tasks)
    n_iter = sum(len(v) for t in tasks for v in t["topo"].values())
    plan_q, why_q = plan_quiet_hours(s["plan_args"])
    now = time.time()
    active = [t for t in tasks if plan_q and (now - t["newest"]) / 3600.0 < plan_q]
    if why_q is not None:
        F.add("FAIL", "plan args", "%s (sim.plan_args %r)" % (why_q, s["plan_args"]))
    if not tasks:
        F.add("FAIL", "campaign", "no sweep task under %s matching %s"
              % (s["sim_main"], s["campaign_glob"]))
    else:
        F.add("NOTE", "campaign", "%d campaign(s) %s: %d task(s), %d topologies, %d iteration files"
              % (len(camps), ", ".join(camps), len(tasks), n_topo, n_iter))
        newest = max(t["newest"] for t in tasks)
        age_h = (now - newest) / 3600.0
        last = [t for t in tasks if t["newest"] == newest][0]
        text = ("last change %.1f h ago (%s), in %s/%s"
                % (age_h, time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)),
                   last["campaign"], last["sweep"]))
        planned_young = [t for t in tasks if t not in active and (now - t["newest"]) / 3600.0 < quiet_hours]
        if planned_young:
            F.add("WARN", "campaign", text + ": younger than %g h -- a job may still be writing "
                  "it; S2 must not plan a campaign being written (D-060, D-073)%s"
                  % (quiet_hours, ("; the plan leaves out only the tasks changed within %g h "
                                   "(sim.plan_args --quiet-hours, D-112): %d task(s) younger than "
                                   "%g h are planned" % (plan_q, len(planned_young), quiet_hours))
                     if plan_q else ""))
        elif age_h < quiet_hours:
            F.add("NOTE", "campaign", text + ": younger than %g h, and S2's plan leaves out every task "
                  "changed within %g h (sim.plan_args --quiet-hours, D-112)" % (quiet_hours, plan_q))
        else:
            F.add("OK", "campaign", text)
        if plan_q:
            names = ["%s/%s" % (t["campaign"], t["sweep"]) for t in active]
            if active and len(active) == len(tasks):
                F.add("FAIL", "being written", "every task (%d) changed less than %g h ago: S2's plan "
                      "refuses (no runnable task); wait, or lower --quiet-hours in sim.plan_args"
                      % (len(tasks), plan_q))
            elif active:
                F.add("NOTE", "being written", "%d of %d task(s) changed less than %g h ago: S2's plan "
                      "leaves them out (D-112), they are not sampled here: %s%s"
                      % (len(active), len(tasks), plan_q, ", ".join(names[:8]),
                         " ..." if len(names) > 8 else ""))
            else:
                F.add("OK", "being written", "no task changed within %g h (sim.plan_args "
                      "--quiet-hours, D-112): S2 plans all %d" % (plan_q, len(tasks)))
        gaps, worst, n_files = iteration_gaps(tasks)
        if gaps:
            srt = sorted(gaps)
            p95 = srt[min(len(srt) - 1, int(math.ceil(0.95 * len(srt))) - 1)]
            text = ("the gaps between two iteration files of one task: 95%% under %.2f h, the longest "
                    "%.2f h (in %s), from %d file(s)" % (p95, srt[-1], worst, n_files))
            if plan_q and p95 >= plan_q:
                F.add("WARN", "pace", text + ": --quiet-hours %g is not longer than most gaps, so a "
                      "task still running may look finished (D-112)" % plan_q)
            else:
                F.add("NOTE", "pace", text)
    missing = ["%s/%s" % (t["campaign"], t["sweep"]) for t in tasks
               if t["job_args"] is None or t["manifest"] is None]
    if missing:
        F.add("WARN", "task files", "%d task(s) without job_args.json or manifest.json (S2's plan "
              "excludes them by name): %s" % (len(missing), ", ".join(missing[:6])))
    elif tasks:
        F.add("OK", "task files", "every task has job_args.json and manifest.json")
    jas = [t["job_args"] for t in tasks if t["job_args"] is not None]
    mfs = [t["manifest"] for t in tasks if t["manifest"] is not None]

    # --- simtime and the window ------------------------------------------------
    simtimes = _single([ja.get("simtime") for ja in jas])
    if len(simtimes) == 1 and _finite(simtimes[0]) and float(simtimes[0]) == float(b["t_win_s"]):
        F.add("OK", "simtime", "%r s in every task = bank.t_win_s" % (simtimes[0],))
    else:
        F.add("FAIL", "simtime", "job_args simtime %r, bank.t_win_s %r (one window per simulation: "
              "they must be equal)" % (simtimes, b["t_win_s"]))

    # --- the real arm's manifest --------------------------------------------
    man_path = os.path.join(d["real"]["extract_root"], "cohort_manifest.json")
    man = _read_json(man_path) if os.path.isfile(man_path) else None
    if man is None:
        F.add("NOTE", "cohort manifest", "%s absent: S1 runs (a first extraction)" % man_path)
    elif _sha_ok(man_path):
        F.add("OK", "cohort manifest", "%s matches its .sha256: S1 is done (skipped); %s wells, "
              "classes %s" % (man_path, man.get("n_wells"), man.get("classes")))
    else:
        F.add("FAIL", "cohort manifest", "%s does not match a .sha256: S1 would run, and its "
              "launcher refuses a root that holds a manifest" % man_path)
    pre = (man or {}).get("preprocessing") or {}
    w_size = pre.get("w_size")
    W = None
    if _finite(w_size) and w_size > 0:
        # the bank plan's own rule (Sbi-extractor bank_plan.py): W = round(T / dt),
        # |W dt - T| <= 1e-9 s and int(T / dt) == W
        t_win, dt = float(b["t_win_s"]), float(w_size)
        w_try = int(round(t_win / dt))
        if abs(w_try * dt - t_win) <= 1e-9 and int(t_win / dt) == w_try:
            W = w_try
            F.add("OK", "window", "t_win_s %r s / w_size %r s = W %d" % (b["t_win_s"], w_size, W))
        else:
            F.add("FAIL", "window", "t_win_s %r s is not a whole number of %r s bins"
                  % (b["t_win_s"], w_size))
    elif man is not None:
        F.add("FAIL", "window", "the manifest has no usable preprocessing.w_size: %r" % (w_size,))
    eps = pre.get("electrodes_per_subset")
    if man is not None:
        if eps == b["n_e"]:
            F.add("OK", "n_e", "bank.n_e %d = the manifest's electrodes_per_subset" % b["n_e"])
        else:
            F.add("FAIL", "n_e", "bank.n_e %r, the manifest's electrodes_per_subset %r"
                  % (b["n_e"], eps))

    # --- the label's width and the O_N box -------------------------------------
    actives = _single([m.get("active_indices") for m in mfs])
    sa = d["search"]["shape_anchors"]
    if len(actives) == 1 and isinstance(actives[0], list):
        p = len(actives[0]) + len(b["topology_axes"])
        widths = (b["expected_p"], sa["p"], sa["d_theta"])
        if all(w == p for w in widths):
            F.add("OK", "label width", "%d active axes + %d topology axes = %d = bank.expected_p = "
                  "shape_anchors p = d_theta" % (len(actives[0]), len(b["topology_axes"]), p))
        else:
            F.add("FAIL", "label width", "%d active axes + %d topology axes = %d; bank.expected_p "
                  "%r, shape_anchors p %r, d_theta %r" % (len(actives[0]), len(b["topology_axes"]),
                                                         p, widths[0], widths[1], widths[2]))
    else:
        F.add("FAIL", "label width", "active_indices differs between tasks or is absent: %d "
              "value(s)" % len(actives))
    lo_d, hi_d = float(b["on_declared_box"][0]), float(b["on_declared_box"][1])
    boxes, bad_on = {}, []
    for t in tasks:
        m = t["manifest"] or {}
        names, bounds = m.get("param_names"), m.get("param_bounds")
        if not (isinstance(names, list) and b["on_axis"] in names and isinstance(bounds, list)
                and len(bounds) == len(names)):
            bad_on.append("%s/%s" % (t["campaign"], t["sweep"]))
            continue
        row = bounds[names.index(b["on_axis"])]
        boxes.setdefault((float(row[0]), float(row[1])), []).append(t["sweep"])
    outside = [bx for bx in boxes if not (bx[0] >= lo_d * (1.0 - BOX_RTOL)
                                          and bx[1] <= hi_d * (1.0 + BOX_RTOL) and bx[0] < bx[1])]
    if bad_on or outside or not boxes:
        F.add("FAIL", "on axis", "%s: tasks without it in param_names/param_bounds %s; boxes "
              "outside the declared [%r, %r]: %s" % (b["on_axis"], bad_on[:4], lo_d, hi_d,
                                                     [list(bx) for bx in outside]))
    else:
        F.add("OK", "on axis", "%s boxes %s inside the declared [%r, %r]"
              % (b["on_axis"], [list(bx) for bx in sorted(boxes)], lo_d, hi_d))
        if all(bx[0] > lo_d * (1.0 + BOX_RTOL) or bx[1] < hi_d * (1.0 - BOX_RTOL) for bx in boxes):
            F.add("NOTE", "on axis", "the declared box is wider than every task's: the prior is wider "
                  "than the sampled box (D-078)")

    # --- the connectivity rule ----------------------------------------------
    rules = _single([(t["job_args"] or {}).get("conn_rule", (t["manifest"] or {}).get("conn_rule"))
                     for t in tasks])
    want = d["freeze"]["require_conn_rule"]
    if rules == [want]:
        F.add("OK", "conn_rule", "%r in every task = freeze.require_conn_rule" % want)
    else:
        F.add("FAIL", "conn_rule", "the tasks hold %r, freeze.require_conn_rule is %r" % (rules, want))

    # --- the sampled iteration files ----------------------------------------
    quiet = [t for t in tasks if t not in active] or tasks
    picks = sample_files(quiet, max_files)
    stats, unreadable = {}, []
    for n, k, path in picks:
        try:
            sc = read_scalars(path)
        except Exception as exc:   # noqa: BLE001 -- reported, never hidden
            unreadable.append("%s (%s)" % (path, exc))
            continue
        for key, v in sc.items():
            st = stats.setdefault(key, {"files": 0, "values": [], "nan": 0, "task_k": []})
            st["files"] += 1
            st["task_k"].append((n, k))
            if isinstance(v, float) and math.isnan(v):
                st["nan"] += 1
            else:
                st["values"].append(v)
    n_read = len(picks) - len(unreadable)
    if unreadable:
        F.add("FAIL", "iteration files", "%d of %d sampled file(s) unreadable: %s"
              % (len(unreadable), len(picks), "; ".join(unreadable[:3])))
    else:
        F.add("OK" if picks else "FAIL", "iteration files", "%d sampled (the first iteration of "
              "each topology, at most %d)" % (len(picks), max_files))

    SL, why = _sbi_labels(d["paths"]["sbix_dir"])
    known = tuple(getattr(SL, "KNOWN_TOPOLOGY_AXES", ())) if SL else ()
    jab = dict(getattr(SL, "JOB_ARGS_TOPOLOGY_BOUNDS", {})) if SL else {}
    for axis in b["topology_axes"]:
        label = "topology axis %s" % axis
        if SL is None:
            F.add("FAIL", label, "Sbi-extractor's sbi_labels cannot be imported from %s: %s"
                  % (d["paths"]["sbix_dir"], why))
            continue
        if axis not in known:
            F.add("FAIL", label, "not in the Sbi-extractor checkout's KNOWN_TOPOLOGY_AXES %r: "
                  "pull it (the D-107 change) or fix bank.topology_axes" % (known,))
            continue
        st = stats.get(axis)
        problems = []
        if st is None or st["files"] < n_read:
            problems.append("present in %d of %d sampled files" % (st["files"] if st else 0, n_read))
        if st and st["nan"]:
            problems.append("%d NaN value(s)" % st["nan"])
        vals = [v for v in (st["values"] if st else []) if isinstance(v, float)]
        if len(set(vals)) < 2:
            problems.append("%d distinct value(s) over the sampled topologies" % len(set(vals)))
        if axis in jab:
            k_lo, k_hi = jab[axis]
            pairs = _single([[ja.get(k_lo, "absent"), ja.get(k_hi, "absent")] for ja in jas])
            if not (len(pairs) == 1 and all(_finite(x) for x in pairs[0])
                    and pairs[0][0] < pairs[0][1]):
                problems.append("job_args %s / %s over the tasks: %r" % (k_lo, k_hi, pairs))
            elif vals and not all(pairs[0][0] <= v <= pairs[0][1] for v in vals):
                problems.append("sampled values outside [%r, %r]" % (pairs[0][0], pairs[0][1]))
        text = ("%d distinct sampled values in [%.6g, %.6g]" % (len(set(vals)), min(vals), max(vals))
                if vals else "no value")
        if axis in jab and not problems:
            text += "; box [%r, %r] from job_args" % tuple(pairs[0])
        if problems:
            F.add("FAIL", label, "; ".join(problems))
        else:
            F.add("OK", label, text)
    for axis in sorted(d["freeze"]["exclude"]):
        if axis in stats:
            F.add("OK", "freeze exclude %s" % axis, "present in the sampled files (%d NaN of %d)"
                  % (stats[axis]["nan"], stats[axis]["files"]))
        else:
            F.add("FAIL", "freeze exclude %s" % axis, "absent from the sampled files: S3 refuses an "
                  "exclusion of an axis the data does not hold")
    for axis in known:
        st = stats.get(axis)
        if (st and st["nan"] == st["files"] and axis not in b["topology_axes"]
                and axis not in d["freeze"]["exclude"]):
            F.add("WARN", "freeze exclude %s" % axis, "present with NaN values only and not excluded: "
                  "the freeze counts NaN values as distinct (EXTRACTOR_USAGE trap 6.5)")

    # --- the batch numbers (D-108) ------------------------------------------
    by = {}
    for c in camps:
        m = CAMPAIGN_RE.search(c)
        by.setdefault(int(m.group(1)) if m else 0, []).append(c)
    clash = dict((k, v) for k, v in by.items() if len(v) > 1)
    if clash:
        F.add("FAIL", "batches", "campaigns sharing one batch number (N of _v<N>, else 0): %r" % clash)
    elif camps:
        F.add("OK", "batches", ", ".join("%s %d" % (c, k) for k, v in sorted(by.items()) for c in v))

    # --- S2's profile and root --------------------------------------------------
    rec = os.path.join(s["out_root"], "REEXTRACTION_RECORD.json")
    profile = os.path.join(d["paths"]["sbix_dir"], "sim_reextract", "profiles",
                           "%s.sh" % d["real"]["cohort_tag"])
    if _sha_ok(rec):
        F.add("NOTE", "S2 root", "%s matches its .sha256: S2 is done" % rec)
    elif not os.path.isdir(s["out_root"]) or not os.listdir(s["out_root"]):
        F.add("NOTE", "S2 root", "%s absent or empty: S2 re-detects into a new root" % s["out_root"])
    else:
        F.add("WARN", "S2 root", "%s holds files but no record: S2 resumes (RESUME=1)" % s["out_root"])
    if os.path.isfile(profile):
        F.add("OK", "S2 profile", profile)
    elif not _sha_ok(rec):
        F.add("FAIL", "S2 profile", "%s absent: S2 refuses (pull Sbi-extractor)" % profile)
    if plan_q:
        plan_py = os.path.join(d["paths"]["sbix_dir"], "sim_reextract", "sim_reextract_plan.py")
        try:
            has = "--quiet-hours" in open(plan_py).read()
        except OSError:
            has = False
        if has:
            F.add("OK", "plan option", "%s takes --quiet-hours (D-112)" % plan_py)
        else:
            F.add("FAIL", "plan option", "%s has no --quiet-hours, which sim.plan_args gives it: pull "
                  "Sbi-extractor (D-112)" % plan_py)

    # --- the job protocol (D-113) ----------------------------------------------
    jobs_dir = os.path.join(d["paths"]["sbi_dir"], "hpc", "pipeline", "jobs")
    miss = [f for f in ("stage_job.sh", "activate_env.sh", "chain.sh")
            if not os.path.isfile(os.path.join(jobs_dir, f))]
    bad_v = []
    for k, v in (("work_root", d["paths"]["work_root"]), ("sbi_dir", d["paths"]["sbi_dir"])):
        if "," in v or any(c.isspace() for c in v):
            bad_v.append("paths.%s %r" % (k, v))
    if miss or bad_v:
        F.add("FAIL", "protocol (SBI)", "%s%s" % (
            ("%s lacks %s: paths.sbi_dir is a checkout before D-113; " % (jobs_dir, ", ".join(miss)))
            if miss else "",
            ("qsub -v cannot carry a comma or a blank: %s" % ", ".join(bad_v)) if bad_v else ""))
    else:
        recs = [e for e in sorted(set(d["envs"].values()))
                if os.path.isfile(os.path.join(d["paths"]["work_root"], "envs", e + ".prefix"))]
        F.add("OK", "protocol (SBI)", "%s/stage_job.sh; environment prefixes recorded under %s/envs: %s"
              % (jobs_dir, d["paths"]["work_root"],
                 ", ".join(recs) if recs else "none yet (the submission records them, D-111)"))
    sbix = d["paths"]["sbix_dir"]
    lacks = []
    try:
        if "sbix_pipeline_v()" not in open(os.path.join(sbix, "env.sh")).read():
            lacks.append("env.sh's sbix_pipeline_v")
    except OSError:
        lacks.append("env.sh")
    for rel in SBIX_PROTOCOL_SCRIPTS:
        try:
            if "pipeline_stage_job_start" not in open(os.path.join(sbix, rel)).read():
                lacks.append(rel)
        except OSError:
            lacks.append(rel + " (absent)")
    if lacks:
        F.add("FAIL", "protocol (Sbi-extractor)", "%s lacks %s: a checkout before D-113, whose jobs "
              "would start without the job protocol -- pull it" % (sbix, ", ".join(lacks)))
    else:
        F.add("OK", "protocol (Sbi-extractor)", "env.sh's sbix_pipeline_v and the %d job scripts "
              "start through the job protocol (D-113)" % len(SBIX_PROTOCOL_SCRIPTS))

    # --- the registry ---------------------------------------------------------
    widths = _single([len(m.get("param_names") or []) for m in mfs])
    if SL is not None:
        try:
            reg = SL.load_registry(s["sim_main"])
            n = len(reg.param_names)
            if widths == [n]:
                F.add("OK", "registry", "%d axes in %s, as the manifests' param_names" % (n, s["sim_main"]))
            else:
                F.add("FAIL", "registry", "%d axes in %s, the manifests' param_names %r"
                      % (n, s["sim_main"], widths))
        except Exception as exc:   # noqa: BLE001 -- reported, never hidden
            F.add("NOTE", "registry", "load_registry(%s) failed: %s: %s"
                  % (s["sim_main"], type(exc).__name__, exc))

    # --- size and the queue ------------------------------------------------------
    if W:
        F.add("NOTE", "size", "%d rows before the floor, %.1f GB of raw windows (W %d, float32)"
              % (n_iter, n_iter * W * 4 / 1e9, W))
    if use_qstat and shutil.which("qstat"):
        user = os.environ.get("USER") or getpass.getuser()
        try:
            r = subprocess.run(["qstat", "-u", user], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
            lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
            F.add("NOTE", "qstat", "%d line(s) for %s; read the job names below" % (len(lines), user))
            for ln in lines[:40]:
                out.write("      %s\n" % ln)
        except (OSError, subprocess.SubprocessError) as exc:
            F.add("NOTE", "qstat", "failed: %s" % exc)

    # --- the sampled members --------------------------------------------------------
    out.write("\nsampled 0-d members (%d file(s)):\n" % n_read)
    out.write("  %-24s %6s %9s %12s %12s %5s\n" % ("member", "files", "distinct", "min", "max", "nan"))
    for key in sorted(stats):
        st = stats[key]
        nums = [v for v in st["values"] if isinstance(v, float)]
        if nums:
            out.write("  %-24s %6d %9d %12.6g %12.6g %5d\n"
                      % (key, st["files"], len(set(nums)), min(nums), max(nums), st["nan"]))
        else:
            out.write("  %-24s %6d %9d %12s %12s %5d\n"
                      % (key, st["files"], len(set(map(str, st["values"]))),
                         str(st["values"][0])[:12] if st["values"] else "-", "", st["nan"]))
    return F, {"campaigns": camps, "n_tasks": len(tasks), "n_topologies": n_topo,
               "n_iterations": n_iter, "sampled": n_read,
               "plan_quiet_hours": plan_q, "being_written": sorted("%s/%s" % (t["campaign"], t["sweep"])
                                                                   for t in active),
               "members": dict((k, {"files": v["files"], "nan": v["nan"],
                                    "distinct": len(set(map(str, v["values"])))})
                               for k, v in stats.items())}


def build_parser():
    p = argparse.ArgumentParser(description="The inputs check: a read-only look at a data set's "
                                            "inputs before its first submission (block 9).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--project", help="tag: hpc/pipeline/projects/<tag>/<tag>.json")
    g.add_argument("--config", help="a configuration file")
    p.add_argument("--max-files", type=int, default=200,
                   help="raw iteration files to open, one per topology (default 200)")
    p.add_argument("--quiet-hours", type=float, default=6.0,
                   help="WARN when a task S2 would plan changed more recently than this (default "
                        "6 h); the plan's own window is sim.plan_args --quiet-hours (D-112)")
    p.add_argument("--json", default=None, help="write the findings here")
    p.add_argument("--no-qstat", action="store_true", help="do not call qstat")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.max_files < 1 or a.quiet_hours < 0:
        print("--max-files must be >= 1 and --quiet-hours >= 0")
        return EXIT_USAGE
    path = a.config or os.path.join(_PIPE, "projects", a.project, "%s.json" % a.project)
    try:
        cfg = C.load_config(path)
    except C.ConfigError as exc:
        print("the configuration %s is refused:\n%s" % (path, exc))
        return EXIT_USAGE
    except OSError as exc:
        print("no configuration: %s" % exc)
        return EXIT_USAGE
    F, summary = run_checks(cfg, a.max_files, a.quiet_hours, not a.no_qstat)
    n_fail, n_warn = F.count("FAIL"), F.count("WARN")
    if a.json:
        tmp = a.json + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"config": cfg.path, "config_sha256": cfg.sha256, "checks": F.rows,
                       "summary": summary, "fail": n_fail, "warn": n_warn}, fh, indent=1,
                      sort_keys=True)
        os.replace(tmp, a.json)
    print("INPUTS CHECK: %d FAIL, %d WARN" % (n_fail, n_warn))
    return EXIT_FAIL if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
