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

A synthetic project (block 24, added 2026-11-13; D-132): for a
configuration of kind `synthetic` the kinds are these five, listed and
deleted in this order (block 10's kinds are refused for it, these for any
other kind):
    evaluation  the evaluation's out_root (--eval-config, by default
                <the configuration's folder>/<dataset>_eval.json)
    best_fits   <runs_root>/<q> for each search q (its best fit and the
                other arm's)
    searches    <tune_root>/<q> for each q (the campaign, rounds set aside)
    banks       <banks_root>/<dataset>_sim and <dataset>_real_<name>
    state       every entry of the work root but purge/ (the state file
                and its .prev, the log, the reports): the next submission
                starts a new run
When `evaluation` is named and its out_root exists, --archive <file.tar.gz>
is required (or --no-archive): every file under out_root except samples/
and calbank/ is packed there, re-read and compared with the listing and
given a .sha256, before anything is deleted. Refused besides: a job of the
evaluation's ledger not finished (exit 4); an archive path that exists,
lies under out_root or a target, or whose folder does not exist (exit 2).

    python3 hpc/pipeline/tools/purge_outputs.py --config <synth.json> \\
        --what evaluation,best_fits,searches,banks,state --archive ~/<name>.tar.gz

The pass line then

    PURGE DONE: <k> target(s) deleted, <n> file(s), <x> GB; archive <path> (<m> file(s), sha256 <16 hex>)[; the work root keeps purge/ only: ...]

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
import stat
import subprocess
import sys
import tarfile
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
SYNTH_KINDS = ("evaluation", "best_fits", "searches", "banks", "state")   # block 24
ARCHIVE_SKIP = ("samples", "calbank")       # the evaluation's subtrees not archived
EXIT_OK, EXIT_ARGS, EXIT_REFUSED, EXIT_INCOMPLETE = 0, 2, 4, 5
SCHEMA = "purge_outputs/1"
_TAR_OPEN = tarfile.open                    # the archive's writer (a test replaces it)


class ArchiveError(Exception):
    """The archive does not hold what the listing promised."""


def is_synthetic(cfg):
    return cfg.data.get("kind") == "synthetic"


# ---------------------------------------------------------------- targets ---

def eval_config_path(cfg, given=None):
    """--eval-config, or <the configuration's folder>/<dataset>_eval.json."""
    if given:
        return os.path.abspath(os.path.expanduser(given))
    return os.path.join(os.path.dirname(os.path.abspath(cfg.path)),
                        "%s_eval.json" % cfg.data["dataset"])


def ev_modules():
    """The evaluation's configuration loader and launcher (stdlib only),
    imported on demand: block 10's projects never need them."""
    ev = os.path.join(_PIPE, "evaluation")
    if ev not in sys.path:
        sys.path.insert(0, ev)
    import ev_config            # noqa: E402
    import ev_submit            # noqa: E402
    return ev_config, ev_submit


def _synth_targets(cfg, kinds, ecfg):
    """[(kind, root, path, note)] of a synthetic project (block 24), in the
    deletion's order: evaluation, best_fits, searches, banks, state."""
    from stages.synthetic import RunView, bank_dir
    p = cfg.data["paths"]
    syn = cfg.data["synthetic"]
    runs = [(q, RunView(cfg, q)) for q in syn["searches"]]
    out = []
    if "evaluation" in kinds:
        ev = os.path.normpath(ecfg.out_root)
        out.append(("evaluation", os.path.dirname(ev), ev, "the evaluation's out_root"))
    if "best_fits" in kinds:
        for q, v in runs:
            out.append(("best_fits", p["runs_root"], v.data["paths"]["runs_root"],
                        "search %s: its best fit and the other arm's" % q["name"]))
    if "searches" in kinds:
        for q, v in runs:
            out.append(("searches", p["tune_root"], v.data["paths"]["tune_root"],
                        "search %s: the campaign %s and any rounds set aside"
                        % (q["name"], q["campaign"])))
    if "banks" in kinds:
        for b in ["sim"] + [r["name"] for r in syn["real"]]:
            out.append(("banks", p["banks_root"], bank_dir(cfg, b), "the bank %s" % b))
    if "state" in kinds:
        wr = p["work_root"]
        try:
            names = sorted(n for n in os.listdir(wr) if n != "purge")
        except OSError:
            names = []
        for n in names:
            out.append(("state", wr, os.path.join(wr, n), "the work root's %s" % n))
    return out


def targets_of(cfg, kinds, ecfg=None):
    """[(kind, root, path, note)] in the order the deletion follows: the
    folders each kind owns, from the configuration; `root` is the
    configured folder the path must stay under."""
    if is_synthetic(cfg):
        return _synth_targets(cfg, kinds, ecfg)
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
    if is_synthetic(cfg):
        return [os.path.join(cfg.data["paths"]["work_root"], "purge")]
    return [os.path.dirname(S.artefact_of("real_bank", cfg)[1]),
            os.path.dirname(S.artefact_of("bank_plan", cfg)[1]),
            cfg.data["paths"]["work_root"]]


def _tree(path):
    """(files, bytes, newest mtime in whole seconds) under a directory, or
    of a file itself: lstat, links not followed (a link counts as a file of
    its own size)."""
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        return 1, st.st_size, int(st.st_mtime)
    n, b, newest = 0, 0, int(st.st_mtime)
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


def _problem(root, path, allow_file=False):
    """Why a present target must not be deleted, or None; a regular file is
    a target only where `allow_file` (block 24's `state`)."""
    if os.path.islink(path):
        return "is a symbolic link (delete it by hand if that is meant)"
    if not os.path.isdir(path) and not (allow_file and os.path.isfile(path)):
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


def eval_jobs_check(ecfg, ev_submit, pbs_module=PBS):
    """[why] for the jobs of the evaluation's ledger not finished, and a
    one-line summary (block 24)."""
    ledger = ev_submit.read_ledger(ecfg)
    why, n = [], 0
    for unit, rec in sorted(ledger.items()):
        jid = (rec or {}).get("job_id")
        if not jid:
            continue
        n += 1
        js = pbs_module.status(jid)
        if not js.finished:
            why.append("the evaluation's job %s (%s) is %s" % (jid, unit, js.state))
    return why, "the evaluation's ledger: %d job(s)" % n


# ---------------------------------------------------------------- listing ---

def _campaign_extra(camp):
    counts, runs = _ledger(camp)
    return "the ledger: %s; %d run folder(s)" % (
        ", ".join("%s %d" % kv for kv in sorted(counts.items())) or "empty", runs)


def listing(cfg, kinds, ecfg=None):
    items = []
    for kind, root, path, note in targets_of(cfg, kinds, ecfg):
        it = {"kind": kind, "path": path, "note": note, "present": os.path.lexists(path),
              "files": 0, "bytes": 0, "newest": 0, "problem": None, "extra": ""}
        if it["present"]:
            it["problem"] = _problem(root, path, allow_file=(kind == "state"))
            if it["problem"] is None:
                it["files"], it["bytes"], it["newest"] = _tree(path)
                if kind == "search" and note == "the search's campaign folder":
                    it["extra"] = _campaign_extra(path)
                elif kind == "sim_bank" and note == "the simulated bank":
                    sha = _record_sha(path)
                    it["extra"] = ("record sha256 %s (its .sha256)" % sha) if sha else "no record"
                elif kind == "searches":
                    camps = sorted(n for n in os.listdir(path)
                                   if os.path.isdir(os.path.join(path, n, "trials")))
                    it["extra"] = "; ".join("%s: %s" % (n, _campaign_extra(os.path.join(path, n)))
                                            for n in camps)
                elif kind == "banks":
                    sha = _record_sha(path)
                    it["extra"] = ("record sha256 %s (its .sha256)" % sha) if sha else "no record"
        items.append(it)
    return items


def archive_plan(out_root):
    """The evaluation's archive (block 24): (members, skipped, problems).
    members: [(path, arcname, size, mtime)] of every regular file under
    out_root except its top-level ARCHIVE_SKIP subtrees, arcnames relative
    to out_root's parent, sorted by arcname; skipped: {name: (files,
    bytes)}; problems: [why] (a symbolic link or a non-regular file)."""
    root = os.path.normpath(out_root)
    base = os.path.dirname(root)
    members, skipped, problems = [], {}, []
    for dpath, subs, files in os.walk(root, followlinks=False):
        if dpath == root:
            for s in sorted(x for x in subs if x in ARCHIVE_SKIP):
                sp = os.path.join(dpath, s)
                if os.path.islink(sp):
                    problems.append("%s is a symbolic link" % sp)
                    continue
                n, b, _t = _tree(sp)
                skipped[s] = (n, b)
            subs[:] = [x for x in subs if x not in ARCHIVE_SKIP]
        for s in subs:
            if os.path.islink(os.path.join(dpath, s)):
                problems.append("%s is a symbolic link" % os.path.join(dpath, s))
        subs.sort()
        for f in sorted(files):
            fp = os.path.join(dpath, f)
            st = os.lstat(fp)
            if stat.S_ISLNK(st.st_mode):
                problems.append("%s is a symbolic link" % fp)
            elif not stat.S_ISREG(st.st_mode):
                problems.append("%s is not a regular file" % fp)
            else:
                members.append((fp, os.path.relpath(fp, base), st.st_size, int(st.st_mtime)))
    members.sort(key=lambda m: m[1])
    return members, skipped, problems


def code_of(items, extra=None):
    """8 hex characters of the targets' canonical text: stale once a file in
    a target is added, removed, resized or rewritten. `extra` (block 24:
    the archive's path and members) joins the text when given; without it
    the text is block 10's."""
    targets = [{"kind": it["kind"], "path": it["path"], "present": it["present"],
                "files": it["files"], "bytes": it["bytes"], "newest": it["newest"]}
               for it in items]
    canon = json.dumps(targets if extra is None else {"targets": targets, "extra": extra},
                       sort_keys=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()[:8]


def archive_extra(archive):
    """The archive's part of the CODE's canonical text, or None."""
    if archive is None:
        return None
    return {"archive": archive["path"],
            "members": [[a, s, t] for _p, a, s, t in archive["members"]]}


def _gb(b):
    return b / 1e9


def _utc(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else "-"


def print_listing(cfg, items, summary, code, archive=None, kinds=()):
    synth = is_synthetic(cfg)
    print("PURGE (nothing deleted yet): project %s, work root %s"
          % (cfg.data["dataset"], cfg.data["paths"]["work_root"]))
    print("  jobs: none of the chain %s queued, held or running (%s)"
          % (CM.chain_name(cfg), summary))
    if archive is not None:
        print("  would ARCHIVE %d file(s), %.2f GB uncompressed, into %s (then its .sha256)"
              % (len(archive["members"]), _gb(sum(m[2] for m in archive["members"])),
                 archive["path"]))
        for s in ARCHIVE_SKIP:
            if s in archive["skipped"]:
                n_s, b_s = archive["skipped"][s]
                print("    not archived: %s/ (%d file(s), %.2f GB)" % (s, n_s, _gb(b_s)))
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
    if synth and "state" in kinds:
        print("  the state file, its .prev and the log go with the work root's entries above: "
              "the next submission starts a new run")
    else:
        print("  the state file and the log are not touched")
    print("  in all: %d %s, %d file(s), %.2f GB"
          % (k, "target(s)" if synth else "folder(s)", n, _gb(b)))
    print("CODE %s -- run again with --confirm %s to delete the %s above"
          % (code, code, "targets" if synth else "folders"))


# ---------------------------------------------------------------- delete ---

def _write_record(path, rec):
    CM.write_json(path, rec)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_archive(archive):
    """Write the archive (block 24): `<path>.tmp` (gzip level 6) with every
    member in order, re-read and compared with the listing's names and
    sizes, renamed to `<path>`, then `<path>.sha256`. Returns the sha256.
    Raises (OSError, tarfile.TarError, ArchiveError) with the .tmp
    removed."""
    path = archive["path"]
    tmp = path + ".tmp"
    try:
        tf = _TAR_OPEN(tmp, "w:gz", compresslevel=6)
        try:
            for p, arc, _size, _mtime in archive["members"]:
                tf.add(p, arcname=arc, recursive=False)
        finally:
            tf.close()
        with tarfile.open(tmp, "r:gz") as rf:
            got = [(ti.name, ti.size) for ti in rf.getmembers()]
        want = [(arc, size) for _p, arc, size, _m in archive["members"]]
        if got != want:
            diffs = [(w, g) for w, g in zip(want, got) if w != g][:3]
            raise ArchiveError("the archive holds %d member(s) where the listing has %d; first "
                               "differences (listing, archive): %s" % (len(got), len(want), diffs))
        os.replace(tmp, path)
    except BaseException:
        if os.path.lexists(tmp):
            os.remove(tmp)
        raise
    sha = _sha256_file(path)
    with open(path + ".sha256", "w") as fh:
        fh.write("%s  %s\n" % (sha, os.path.basename(path)))
    return sha


def _remove(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def do_purge(cfg, items, code, summary, archive=None, kinds=()):
    synth = is_synthetic(cfg)
    wr = cfg.data["paths"]["work_root"]
    rdir = os.path.join(wr, "purge")
    os.makedirs(rdir, exist_ok=True)
    rpath = os.path.join(rdir, "PURGE_%s.json" % CM.utc_now().replace(":", ""))
    rec = {"schema": SCHEMA, "code": code, "config_sha256": cfg.sha256, "host": socket.gethostname(),
           "started_utc": CM.utc_now(), "jobs_checked": summary, "status": "started",
           "targets": [dict(it, result="pending" if it["present"] else "absent") for it in items]}
    if archive is not None:
        rec["archive"] = {"path": archive["path"], "members": len(archive["members"]),
                          "bytes": sum(m[2] for m in archive["members"]), "result": "pending"}
    _write_record(rpath, rec)
    print("PURGE: record %s written (the listing and the CODE)" % rpath)
    sha = None
    if archive is not None:
        t0 = time.time()
        try:
            sha = write_archive(archive)
        except (OSError, tarfile.TarError, ArchiveError) as exc:
            rec["archive"]["result"] = "failed: %s" % exc
            rec["status"] = "archive failed"
            rec["finished_utc"] = CM.utc_now()
            _write_record(rpath, rec)
            print("PURGE ARCHIVE FAILED: %s; nothing deleted (the record %s)" % (exc, rpath))
            return EXIT_INCOMPLETE
        rec["archive"].update(result="written", sha256=sha)
        _write_record(rpath, rec)
        print("ARCHIVED %d file(s), %.2f GB uncompressed, into %s: sha256 %s (%.0f s)"
              % (len(archive["members"]), _gb(rec["archive"]["bytes"]), archive["path"],
                 sha[:16], time.time() - t0))
    n = b = k = 0
    failed = []
    for it, r in zip(items, rec["targets"]):
        if not it["present"]:
            continue
        t0 = time.time()
        try:
            _remove(it["path"])
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
        print("PURGE INCOMPLETE: %d %s deleted, %d failed (%s); the record %s says which; "
              "list again to see what is left" % (k, "target(s)" if synth else "folder(s)",
                                                  len(failed), ", ".join(failed), rpath))
        return EXIT_INCOMPLETE
    if not synth:
        print("PURGE DONE: %d folder(s) deleted, %d file(s), %.2f GB; the state file is not touched"
              % (k, n, _gb(b)))
        return EXIT_OK
    arch = ("archive %s (%d file(s), sha256 %s)" % (archive["path"], len(archive["members"]), sha[:16])
            if archive is not None else "no archive")
    print("PURGE DONE: %d target(s) deleted, %d file(s), %.2f GB; %s%s"
          % (k, n, _gb(b), arch, "; the work root keeps purge/ only: the next submission starts "
             "a new run" if "state" in kinds else ""))
    return EXIT_OK


def _archive_args(args, cfg, kinds, ecfg, items):
    """Block 24's archive: (archive or None, None) or (None, why) -- why
    an exit-2 message. The archive is {path, members, skipped, problems}."""
    if args.archive and args.no_archive:
        return None, "--archive and --no-archive exclude each other"
    if args.archive and "evaluation" not in kinds:
        return None, "--archive needs --what evaluation (the archive is the evaluation's)"
    if "evaluation" not in kinds or args.no_archive:
        return None, None
    out_root = os.path.normpath(ecfg.out_root)
    if not os.path.isdir(out_root):
        return None, None
    if not args.archive:
        return None, ("--what evaluation needs --archive <file.tar.gz> (its results are packed "
                      "there before anything is deleted) or --no-archive")
    path = os.path.abspath(os.path.expanduser(args.archive))
    if os.path.lexists(path):
        return None, "the archive %s exists: name another file" % path
    if not os.path.isdir(os.path.dirname(path)):
        return None, "the archive's folder %s does not exist" % os.path.dirname(path)
    real = os.path.realpath(path)
    for it in items:
        if not it["present"]:
            continue
        tp = os.path.realpath(it["path"])
        if real == tp or real.startswith(tp.rstrip(os.sep) + os.sep):
            return None, "the archive %s lies under the target %s" % (path, it["path"])
    ro = os.path.realpath(out_root)
    if real.startswith(ro.rstrip(os.sep) + os.sep):
        return None, "the archive %s lies under the evaluation's out_root %s" % (path, out_root)
    members, skipped, problems = archive_plan(out_root)
    return {"path": path, "members": members, "skipped": skipped, "problems": problems}, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--what", required=True,
                    help="a comma-separated subset of %s; for a synthetic project of %s"
                    % (",".join(KINDS), ",".join(SYNTH_KINDS)))
    ap.add_argument("--eval-config", default=None,
                    help="a synthetic project's evaluation file (default <folder>/<dataset>_eval.json)")
    ap.add_argument("--archive", default=None, metavar="FILE.tar.gz",
                    help="where the evaluation's results are packed before the deletion")
    ap.add_argument("--no-archive", action="store_true",
                    help="delete the evaluation without packing its results")
    ap.add_argument("--confirm", default=None, metavar="CODE")
    args = ap.parse_args(argv)
    kinds = [k.strip() for k in args.what.split(",") if k.strip()]
    bad = [k for k in kinds if k not in KINDS + SYNTH_KINDS]
    if not kinds or bad:
        print("PURGE: --what must name one or more of %s, or for a synthetic project of %s "
              "(got %r)" % (", ".join(KINDS), ", ".join(SYNTH_KINDS), args.what))
        return EXIT_ARGS
    try:
        cfg = C.load_config(args.config, synthetic_ok=True)     # block 24: a synthetic project too
    except C.ConfigError as exc:
        print("PURGE: the configuration %s does not load: %s" % (args.config, exc.problems))
        return EXIT_ARGS
    synth = is_synthetic(cfg)
    allowed = SYNTH_KINDS if synth else KINDS
    wrong = [k for k in kinds if k not in allowed]
    if wrong:
        print("PURGE: --what for %s project must name one or more of %s (got %s)"
              % ("a synthetic" if synth else "this", ", ".join(allowed), ", ".join(wrong)))
        return EXIT_ARGS
    ecfg = ev_submit = None
    if synth:
        epath = eval_config_path(cfg, args.eval_config)
        if os.path.isfile(epath) or "evaluation" in kinds:
            ev_config, ev_submit = ev_modules()
            try:
                ecfg = ev_config.load(epath)
            except (ev_config.EvalConfigError, OSError, ValueError) as exc:
                print("PURGE: the evaluation's configuration %s does not load: %s" % (epath, exc))
                return EXIT_ARGS
    elif args.archive or args.no_archive or args.eval_config:
        print("PURGE: --eval-config, --archive and --no-archive are for a synthetic project")
        return EXIT_ARGS
    try:
        why, summary = jobs_check(cfg)
        if ecfg is not None:
            why_e, summary_e = eval_jobs_check(ecfg, ev_submit)
            why += why_e
            summary += "; " + summary_e
    except PBS.SchedulerError as exc:
        print("PURGE REFUSED: the scheduler did not answer (%s); nothing deleted; try again" % exc)
        return EXIT_REFUSED
    if why:
        for w in why:
            print("  %s" % w)
        print("PURGE REFUSED: the pipeline's jobs must have ended first (stop the sequencer with "
              "run_pipeline.sh --project <tag> --stop --wait 180, then delete its jobs%s); "
              "nothing deleted" % (" and the evaluation's" if synth else ""))
        return EXIT_REFUSED
    items = listing(cfg, kinds, ecfg)
    archive, msg = (_archive_args(args, cfg, kinds, ecfg, items) if synth else (None, None))
    if msg:
        print("PURGE: %s; nothing deleted" % msg)
        return EXIT_ARGS
    bad_items = [it for it in items if it["problem"]]
    if bad_items or (archive is not None and archive["problems"]):
        for it in bad_items:
            print("  [%s] %s %s" % (it["kind"], it["path"], it["problem"]))
        for p in (archive or {}).get("problems", []):
            print("  [archive] %s" % p)
        if synth:
            print("PURGE REFUSED: a target is not a plain folder or file under its root, or the "
                  "archive's folders hold a link; nothing deleted")
        else:
            print("PURGE REFUSED: a target is not a plain folder under its root; nothing deleted")
        return EXIT_REFUSED
    extra = archive_extra(archive)
    code = code_of(items) if extra is None else code_of(items, extra)
    if args.confirm is None:
        print_listing(cfg, items, summary, code, archive, kinds)
        return EXIT_OK
    if args.confirm != code:
        print_listing(cfg, items, summary, code, archive, kinds)
        print("PURGE REFUSED: the CODE %s does not match %s (a file changed since the listing, or "
              "a typo); nothing deleted" % (args.confirm, code))
        return EXIT_REFUSED
    return do_purge(cfg, items, code, summary, archive, kinds)


if __name__ == "__main__":
    sys.exit(main())
