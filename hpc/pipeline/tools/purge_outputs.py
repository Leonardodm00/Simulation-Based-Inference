#!/usr/bin/env python3
"""Delete a project's search, best fit and simulated bank (specs/SPEC.md
block 10, added 2026-11-02: a data set whose simulated data are
corrupted). A list-then-delete tool with a CODE: the project's rule that
Claude never deletes data and the user deletes through such a tool.

    python3 hpc/pipeline/tools/purge_outputs.py --config <project>.json --what search,best_fit,sim_bank
        lists what would be deleted and prints the CODE; deletes nothing
    python3 hpc/pipeline/tools/purge_outputs.py --config <project>.json --what ... --confirm <CODE>
        deletes it, when the CODE still matches what is there

The targets, from the configuration:
    search    <tune_root>/<search.campaign> and every
              <tune_root>/<campaign>_set_aside_* (search_reset.py's)
    best_fit  <runs_root>/<dataset>_best
    sim_bank  <banks_root>/<dataset>_sim and <banks_root>/<dataset>_sim_quarantine
Never touched (printed as kept): the real bank <banks_root>/<dataset>_real,
the bank plan's folder, the work root (state, log, reports), S1-S3.

Refused (exit 4, nothing deleted): the work root's lock; a sequencer job
pl_<tag> queued, held or running; a job a stage of the state records that
the scheduler reports not finished; a job of the user whose qstat names
the chain (CHAIN_NAME=pl_<tag>); a target that is a symbolic link, not a
directory, or outside its root; the scheduler not answering; a CODE that
does not match. Exit 2: the configuration, or --what. Exit 5: a deletion
failed (the others were attempted; the record says which). Exit 0 with
the pass line

    PURGE DONE: <k> folder(s) deleted, <n> file(s), <x> GB; the state file is not touched

Stdlib only. Pure ASCII, LF only; Python 3.6 syntax.
"""

import argparse
import getpass
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)

import config as C                      # noqa: E402
import pbs as PBS                       # noqa: E402
import pipeline as P                    # noqa: E402
import state as ST                      # noqa: E402
import stages as S                      # noqa: E402
import stages.common as CM              # noqa: E402
from stages.bank import SimBankStage    # noqa: E402

KINDS = ("search", "best_fit", "sim_bank")
EXIT_OK, EXIT_ARGS, EXIT_REFUSED, EXIT_INCOMPLETE = 0, 2, 4, 5
SCHEMA = "purge_outputs/1"


# ---------------------------------------------------------------- targets ---

def targets_of(cfg, kinds):
    """[(kind, root, path, note)] in the order the deletion follows: the
    folders each kind owns, from the configuration; `root` is the
    configured folder the path must stay under."""
    p = cfg.data["paths"]
    out = []
    if "search" in kinds:
        camp = os.path.dirname(S.artefact_of("search", cfg)[1])
        out.append(("search", p["tune_root"], camp, "the search's campaign folder"))
        pat = re.compile("^" + re.escape(os.path.basename(camp)) + r"_set_aside_[0-9A-Za-z:._-]+$")
        try:
            names = sorted(n for n in os.listdir(p["tune_root"]) if pat.match(n))
        except OSError:
            names = []
        for n in names:
            out.append(("search", p["tune_root"], os.path.join(p["tune_root"], n),
                        "rounds set aside by search_reset.py"))
    if "best_fit" in kinds:
        out.append(("best_fit", p["runs_root"], os.path.dirname(S.artefact_of("best_fit", cfg)[1]),
                    "the best fit's folder"))
    if "sim_bank" in kinds:
        out.append(("sim_bank", p["banks_root"], os.path.dirname(S.artefact_of("sim_bank", cfg)[1]),
                    "the simulated bank"))
        out.append(("sim_bank", p["banks_root"], SimBankStage()._quarantine_root(cfg),
                    "the simulated bank's quarantine"))
    return out


def kept_of(cfg):
    """The paths printed as kept: never touched by this tool."""
    return [os.path.dirname(S.artefact_of("real_bank", cfg)[1]),
            os.path.dirname(S.artefact_of("bank_plan", cfg)[1]),
            cfg.data["paths"]["work_root"]]


def _tree(path):
    """(files, bytes, newest mtime in whole seconds) under a directory:
    lstat, links not followed (a link counts as a file of its own size)."""
    n, b, newest = 0, 0, int(os.lstat(path).st_mtime)
    for dpath, _subs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                s = os.lstat(os.path.join(dpath, f))
            except OSError:
                continue
            n += 1
            b += s.st_size
            newest = max(newest, int(s.st_mtime))
    return n, b, newest


def _problem(root, path):
    """Why a present target must not be deleted, or None."""
    if os.path.islink(path):
        return "is a symbolic link (delete it by hand if that is meant)"
    if not os.path.isdir(path):
        return "is not a directory"
    r = os.path.realpath(root)
    q = os.path.realpath(path)
    if q == r or not q.startswith(r.rstrip(os.sep) + os.sep):
        return "resolves to %s, outside its root %s" % (q, r)
    return None


def _ledger(camp):
    """The campaign's ledger counts by status, and its run folders."""
    counts = {}
    tdir = os.path.join(camp, "trials")
    if os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            if not f.endswith(".json"):
                continue
            rec, _why = CM.try_read_json(os.path.join(tdir, f))
            k = "unreadable" if rec is None else str(rec.get("status", "ok"))
            counts[k] = counts.get(k, 0) + 1
    rdir = os.path.join(camp, "runs")
    runs = len(os.listdir(rdir)) if os.path.isdir(rdir) else 0
    return counts, runs


def _record_sha(bank):
    """The first 16 hex digits of the bank record's sha256, from its .sha256."""
    try:
        with open(os.path.join(bank, "BANK_RECORD.json.sha256")) as fh:
            tok = fh.read().split()
        return tok[0][:16] if tok else None
    except OSError:
        return None


# ------------------------------------------------------------------- jobs ---

_CHAIN_RE = "CHAIN_NAME=%s(,|$)"


def _user_jobs(user):
    """The ids qselect lists for the user (not finished, without -x)."""
    try:
        proc = subprocess.run([PBS.QSELECT, "-u", user], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)
    except OSError as exc:
        raise PBS.SchedulerError("cannot run %s: %s" % (PBS.QSELECT, exc))
    if proc.returncode != 0:
        raise PBS.SchedulerError("qselect exited %d: %s" % (proc.returncode,
                                                            (proc.stderr or proc.stdout).strip()))
    return [tok for tok in proc.stdout.split() if tok]


def names_chain(raw, chain):
    """True when a qstat -f record names the chain in a variable."""
    joined = raw.replace("\n\t", "")
    return re.search(_CHAIN_RE % re.escape(chain), joined, re.M) is not None


def jobs_check(cfg, pbs_module=PBS, user=None):
    """[why] the deletion must wait for: the lock, the sequencer's jobs, the
    jobs the state records, the user's jobs that name the chain; and a
    one-line summary of what was checked."""
    wr = cfg.data["paths"]["work_root"]
    chain = CM.chain_name(cfg)
    why = []
    lock = P.read_lock(wr)
    if lock is not None:
        why.append("the lock %s names %s on %s: a sequencer holds the work root"
                   % (os.path.join(wr, P.LOCK_FILE), lock[0], lock[1]))
    drivers = P._drivers(cfg, pbs_module)
    for j, s in drivers:
        why.append("the sequencer job %s is %s" % (j, s))
    recorded, live = 0, set(PBS.job_seq(j) for j, _s in drivers)
    state = ST.State.load(wr) if os.path.isfile(ST.State.path_of(wr)) else None
    if state is not None:
        for name, st in sorted(state.data.get("stages", {}).items()):
            for job in (st or {}).get("jobs", []):
                recorded += 1
                if job.get("finished"):
                    continue
                js = pbs_module.status(job["id"])
                if not js.finished:
                    live.add(PBS.job_seq(job["id"]))
                    why.append("the job %s (%s, %s) is %s" % (job["id"], name, job.get("role", ""),
                                                              js.state))
    mine = _user_jobs(user or getpass.getuser())
    for j in mine:
        if PBS.job_seq(j) in live:
            continue
        js = pbs_module.status(j)
        if not js.finished and names_chain(js.raw, chain):
            why.append("the job %s names the chain %s and is %s" % (j, chain, js.state))
    summary = ("lock: %s; sequencer jobs (%s): %d; the state's jobs: %d recorded; the user's "
               "jobs: %d listed" % ("none" if lock is None else "%s on %s" % lock, chain,
                                    len(drivers), recorded, len(mine)))
    return why, summary


# ---------------------------------------------------------------- listing ---

def listing(cfg, kinds):
    items = []
    for kind, root, path, note in targets_of(cfg, kinds):
        it = {"kind": kind, "path": path, "note": note, "present": os.path.lexists(path),
              "files": 0, "bytes": 0, "newest": 0, "problem": None, "extra": ""}
        if it["present"]:
            it["problem"] = _problem(root, path)
            if it["problem"] is None:
                it["files"], it["bytes"], it["newest"] = _tree(path)
                if kind == "search" and note == "the search's campaign folder":
                    counts, runs = _ledger(path)
                    it["extra"] = "the ledger: %s; %d run folder(s)" % (
                        ", ".join("%s %d" % kv for kv in sorted(counts.items())) or "empty", runs)
                elif kind == "sim_bank" and note == "the simulated bank":
                    sha = _record_sha(path)
                    it["extra"] = ("record sha256 %s (its .sha256)" % sha) if sha else "no record"
        items.append(it)
    return items


def code_of(items):
    """8 hex characters of the targets' canonical text: stale once a file in
    a target is added, removed, resized or rewritten."""
    canon = json.dumps([{"kind": it["kind"], "path": it["path"], "present": it["present"],
                         "files": it["files"], "bytes": it["bytes"], "newest": it["newest"]}
                        for it in items], sort_keys=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()[:8]


def _gb(b):
    return b / 1e9


def _utc(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else "-"


def print_listing(cfg, items, summary, code):
    print("PURGE (nothing deleted yet): project %s, work root %s"
          % (cfg.data["dataset"], cfg.data["paths"]["work_root"]))
    print("  jobs: none of the chain %s queued, held or running (%s)"
          % (CM.chain_name(cfg), summary))
    n = b = k = 0
    for it in items:
        if not it["present"]:
            print("  absent       [%s] %s (%s)" % (it["kind"], it["path"], it["note"]))
            continue
        k += 1
        n += it["files"]
        b += it["bytes"]
        print("  would DELETE [%s] %s: %d file(s), %.2f GB, newest %s (%s%s)"
              % (it["kind"], it["path"], it["files"], _gb(it["bytes"]), _utc(it["newest"]),
                 it["note"], ("; " + it["extra"]) if it["extra"] else ""))
    for p in kept_of(cfg):
        print("  kept: %s%s" % (p, "" if os.path.exists(p) else " (absent)"))
    print("  the state file and the log are not touched")
    print("  in all: %d folder(s), %d file(s), %.2f GB" % (k, n, _gb(b)))
    print("CODE %s -- run again with --confirm %s to delete the folders above" % (code, code))


# ---------------------------------------------------------------- delete ---

def _write_record(path, rec):
    CM.write_json(path, rec)


def do_purge(cfg, items, code, summary):
    wr = cfg.data["paths"]["work_root"]
    rdir = os.path.join(wr, "purge")
    os.makedirs(rdir, exist_ok=True)
    rpath = os.path.join(rdir, "PURGE_%s.json" % CM.utc_now().replace(":", ""))
    rec = {"schema": SCHEMA, "code": code, "config_sha256": cfg.sha256, "host": socket.gethostname(),
           "started_utc": CM.utc_now(), "jobs_checked": summary, "status": "started",
           "targets": [dict(it, result="pending" if it["present"] else "absent") for it in items]}
    _write_record(rpath, rec)
    print("PURGE: record %s written (the listing and the CODE)" % rpath)
    n = b = k = 0
    failed = []
    for it, r in zip(items, rec["targets"]):
        if not it["present"]:
            continue
        t0 = time.time()
        try:
            shutil.rmtree(it["path"])
        except OSError as exc:
            r["result"] = "failed: %s" % exc
            failed.append(it["path"])
            print("FAILED [%s] %s: %s" % (it["kind"], it["path"], exc))
            continue
        r["result"] = "deleted"
        k += 1
        n += it["files"]
        b += it["bytes"]
        print("DELETED [%s] %s: %d file(s), %.2f GB (%.0f s)"
              % (it["kind"], it["path"], it["files"], _gb(it["bytes"]), time.time() - t0))
    rec["status"] = "failed" if failed else "done"
    rec["finished_utc"] = CM.utc_now()
    _write_record(rpath, rec)
    if failed:
        print("PURGE INCOMPLETE: %d folder(s) deleted, %d failed (%s); the record %s says which; "
              "list again to see what is left" % (k, len(failed), ", ".join(failed), rpath))
        return EXIT_INCOMPLETE
    print("PURGE DONE: %d folder(s) deleted, %d file(s), %.2f GB; the state file is not touched"
          % (k, n, _gb(b)))
    return EXIT_OK


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--what", required=True, help="a comma-separated subset of " + ",".join(KINDS))
    ap.add_argument("--confirm", default=None, metavar="CODE")
    args = ap.parse_args(argv)
    kinds = [k.strip() for k in args.what.split(",") if k.strip()]
    bad = [k for k in kinds if k not in KINDS]
    if not kinds or bad:
        print("PURGE: --what must name one or more of %s (got %r)" % (", ".join(KINDS), args.what))
        return EXIT_ARGS
    try:
        cfg = C.load_config(args.config)
    except C.ConfigError as exc:
        print("PURGE: the configuration %s does not load: %s" % (args.config, exc.problems))
        return EXIT_ARGS
    try:
        why, summary = jobs_check(cfg)
    except PBS.SchedulerError as exc:
        print("PURGE REFUSED: the scheduler did not answer (%s); nothing deleted; try again" % exc)
        return EXIT_REFUSED
    if why:
        for w in why:
            print("  %s" % w)
        print("PURGE REFUSED: the pipeline's jobs must have ended first (stop the sequencer with "
              "run_pipeline.sh --project <tag> --stop --wait 180, then delete its jobs); nothing "
              "deleted")
        return EXIT_REFUSED
    items = listing(cfg, kinds)
    bad_items = [it for it in items if it["problem"]]
    if bad_items:
        for it in bad_items:
            print("  [%s] %s %s" % (it["kind"], it["path"], it["problem"]))
        print("PURGE REFUSED: a target is not a plain folder under its root; nothing deleted")
        return EXIT_REFUSED
    code = code_of(items)
    if args.confirm is None:
        print_listing(cfg, items, summary, code)
        return EXIT_OK
    if args.confirm != code:
        print_listing(cfg, items, summary, code)
        print("PURGE REFUSED: the CODE %s does not match %s (a file changed since the listing, or "
              "a typo); nothing deleted" % (args.confirm, code))
        return EXIT_REFUSED
    return do_purge(cfg, items, code, summary)


if __name__ == "__main__":
    sys.exit(main())
