#!/usr/bin/env python3
"""Verify the simulated bank shard by shard (block 2, the self-healing bank;
D-094; specs/SPEC.md, block 2, rule (a)). A PBS job, never the login node:
the stage submits it through jobs/run_argv.pbs after every round's gate.

    python3 bank_verify.py --plan <bank_plan.json> --bank-root <dir> \\
        --out <report.json> [--workers N]

For every task of the plan: a marker of this plan (its sha256) listing
exactly the plan's chunks (shard, start, stop); for every chunk the marker
gives a file, the .npz and the .json present, both sha256 equal to the
marker's, every zip member's CRC (each member read to its end, as
zipfile.ZipFile.testzip does), the sidecar valid JSON; for a chunk the marker gives no
file, no .npz present. Then the bank folder's listing: a shard_NNNN.* or
tasks/task_NNNN.* file no chunk or marker names belongs to that chunk's or
that index's task; a shard_* file of no planned chunk, or a marker of
another plan, is a GLOBAL problem; other files are reported, not judged.
Then the record (BANK_RECORD.json: present, its sidecar, its plan).

Why a separate job: a marker's sha256 is taken after its task wrote the
shard, on the node that wrote it, so it cannot vouch for what a failing
node put on disk; the gate reads only the small arrays. This job reads the
bytes on storage from its own node and decompresses every member.

The report (JSON, written atomically to --out) lists each bad task with its
problems and its files on disk (paths relative to the bank folder), the
global problems, the record's state, the verdict OK or BAD, the host and
the seconds. The last line printed:

    VERIFY OK: <n> tasks, <k> shard files (<m> members' CRCs) checked on <host> in <s> s
    VERIFY BAD: <b> task(s) [<i>, ...], <g> global problem(s), on <host>

Exit 0 when the report is written, whatever it found; 2 on bad arguments
or a plan that does not match its .sha256; 75 when the plan or the bank
folder cannot be read (an I/O error: the node this runs on is suspect).

Stdlib only (zipfile, hashlib, json, multiprocessing). Pure ASCII, LF only;
Python 3.6 syntax.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import socket
import sys
import time
import zipfile
import zlib

TOOL_VERSION = 1
EXIT_BAD_ARGS = 2
EXIT_SUSPECT = 75
_SHARD = re.compile(r"^shard_(\d{4,})\.")
_MARKER = re.compile(r"^task_(\d{4,})\.")
RECORD_FILES = ("BANK_RECORD.json", "BANK_RECORD.json.sha256",
                "BANK_VERIFIED.json", "BANK_VERIFIED.json.sha256")


class SuspectNode(RuntimeError):
    """The plan or the bank folder could not be read: an I/O error here."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_npz(path):
    """(sha256 or None, members checked, problem kind, problem text): kind
    None (intact), 'io' (an I/O error reading it) or 'content'."""
    try:
        digest = sha256_file(path)
    except OSError as exc:
        return None, 0, "io", "unreadable (%s)" % exc
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        return digest, 0, "content", "not a zip file (%s)" % exc
    except OSError as exc:
        return digest, 0, "io", "unreadable while opening (%s)" % exc
    # Custom: zipfile.ZipFile.testzip's loop -- read every member to its end,
    # which checks its CRC-32 -- written out so that every failure names its
    # member: testzip names a CRC failure but lets a corrupt deflate stream
    # (zlib.error) escape unnamed. Checked: zipfile.ZipFile.testzip.
    n = 0
    try:
        for zi in z.infolist():
            try:
                with z.open(zi, "r") as fh:
                    while fh.read(1 << 20):
                        pass
            except zipfile.BadZipFile as exc:
                return digest, n, "content", "member %s: %s" % (zi.filename, exc)
            except (zlib.error, EOFError, NotImplementedError, ValueError) as exc:
                return digest, n, "content", "member %s cannot be decompressed (%s: %s)" % (
                    zi.filename, type(exc).__name__, exc)
            except OSError as exc:
                return digest, n, "io", "member %s unreadable (%s)" % (zi.filename, exc)
            n += 1
    finally:
        z.close()
    return digest, n, None, None


def check_json(path):
    """(sha256 or None, 0, problem kind, problem text) for a sidecar."""
    try:
        digest = sha256_file(path)
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError as exc:
        return None, 0, "io", "unreadable (%s)" % exc
    try:
        json.loads(blob.decode("ascii"))
    except (ValueError, UnicodeDecodeError) as exc:
        return digest, 0, "content", "not valid JSON (%s)" % exc
    return digest, 0, None, None


def _check(item):
    kind, path = item
    return check_npz(path) if kind == "npz" else check_json(path)


def load_plan(path):
    """(plan, sha256); SystemExit(2) when absent or not matching its
    .sha256; SuspectNode on an I/O error."""
    side = path + ".sha256"
    if not (os.path.isfile(path) and os.path.isfile(side)):
        print("ERROR: no plan, or no .sha256 beside it, at %s" % path)
        raise SystemExit(EXIT_BAD_ARGS)
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
        with open(side, "rb") as fh:
            tokens = fh.read().decode("ascii", "replace").split()
    except OSError as exc:
        raise SuspectNode("the plan %s cannot be read: %s" % (path, exc))
    digest = hashlib.sha256(blob).hexdigest()
    if not tokens or tokens[0].lower() != digest:
        print("ERROR: the plan %s does not match its .sha256" % path)
        raise SystemExit(EXIT_BAD_ARGS)
    return json.loads(blob.decode("ascii")), digest


def read_record(bank_root, plan_digest):
    """The record's state, as the report gives it."""
    p = os.path.join(bank_root, "BANK_RECORD.json")
    out = {"present": False, "sidecar_ok": False, "sha256": None, "plan_matches": None}
    if not os.path.isfile(p):
        return out
    out["present"] = True
    try:
        digest = sha256_file(p)
        out["sha256"] = digest
        side = p + ".sha256"
        if os.path.isfile(side):
            with open(side, "rb") as fh:
                tokens = fh.read().decode("ascii", "replace").split()
            out["sidecar_ok"] = bool(tokens) and tokens[0].lower() == digest
        with open(p, "rb") as fh:
            rec = json.loads(fh.read().decode("ascii"))
        out["plan_matches"] = rec.get("plan_sha256") == plan_digest
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        out["error"] = str(exc)
    return out


class _Bad(object):
    """The problems of the bad tasks: index -> list of (kind, text)."""

    def __init__(self):
        self.items = {}

    def add(self, i, kind, text):
        self.items.setdefault(int(i), []).append((kind, text))


def verify(plan, digest, bank_root, workers):
    """The report (a dict). SuspectNode when the bank folder cannot be listed."""
    t0 = time.time()
    tasks = plan["tasks"]
    try:
        listing = set(os.listdir(bank_root))
        tdir = os.path.join(bank_root, "tasks")
        tlisting = set(os.listdir(tdir)) if os.path.isdir(tdir) else set()
    except OSError as exc:
        raise SuspectNode("the bank folder %s cannot be listed: %s" % (bank_root, exc))
    chunk_task = {}
    for i, t in enumerate(tasks):
        for c in t["shards"]:
            chunk_task[int(c["shard"])] = i
    bad = _Bad()
    global_problems, other_files = [], []
    jobs = []                       # (kind, path, task index, marker sha256, name)
    for i, t in enumerate(tasks):
        mname = "task_%04d.json" % i
        if mname not in tlisting:
            bad.add(i, "missing", "no marker (the task did not finish)")
            continue
        try:
            with open(os.path.join(tdir, mname), "rb") as fh:
                m = json.loads(fh.read().decode("ascii"))
        except OSError as exc:
            bad.add(i, "io", "its marker is unreadable (%s)" % exc)
            continue
        except (ValueError, UnicodeDecodeError) as exc:
            bad.add(i, "content", "its marker is not valid JSON (%s)" % exc)
            continue
        if m.get("plan_sha256") != digest:
            global_problems.append("tasks/%s belongs to another plan (sha256 %s)"
                                   % (mname, str(m.get("plan_sha256"))[:16]))
            continue
        want = [(int(c["shard"]), int(c["start"]), int(c["stop"])) for c in t["shards"]]
        try:
            got = [(int(s["shard"]), int(s["start"]), int(s["stop"])) for s in m["shards"]]
        except (KeyError, TypeError, ValueError):
            got = None
        if got != want:
            bad.add(i, "content", "its marker lists the chunks %r, the plan %r" % (got, want))
            continue
        for s in m["shards"]:
            if s.get("file") is None:
                npz = "shard_%04d.npz" % int(s["shard"])
                if npz in listing:
                    bad.add(i, "content", "%s exists, but the marker says the chunk keeps no row" % npz)
                continue
            for name, want_sha, kind in ((s["file"], s.get("sha256"), "npz"),
                                         (s["json"], s.get("json_sha256"), "json")):
                if name not in listing:
                    bad.add(i, "missing", "%s missing" % name)
                else:
                    jobs.append((kind, os.path.join(bank_root, name), i, want_sha, name))
    # sha256 and CRC of every file the markers name, in parallel
    items = [(k, p) for k, p, _, _, _ in jobs]
    if items:
        pool = multiprocessing.Pool(max(1, int(workers)))
        try:
            results = pool.map(_check, items, chunksize=1)
        finally:
            pool.close()
            pool.join()
    else:
        results = []
    n_npz, n_members = 0, 0
    for (kind, path, i, want_sha, name), (got_sha, members, pkind, ptext) in zip(jobs, results):
        if kind == "npz":
            n_npz += 1
            n_members += members
        if got_sha is None:
            bad.add(i, pkind or "io", "%s %s" % (name, ptext))
            continue
        if got_sha != want_sha:
            bad.add(i, "content", "%s changed since its marker (sha256 %s, the marker's %s)"
                    % (name, got_sha[:16], str(want_sha)[:16]))
        if pkind is not None:
            bad.add(i, pkind, "%s: %s" % (name, ptext))
    # who owns each file on disk
    named = set()
    for t in tasks:
        for c in t["shards"]:
            named.add("shard_%04d.npz" % int(c["shard"]))
            named.add("shard_%04d.json" % int(c["shard"]))
    leftovers = {}                  # task index -> [relative paths]
    for name in sorted(listing):
        if name in named or name == "tasks" or name in RECORD_FILES:
            continue
        mm = _SHARD.match(name)
        if mm and int(mm.group(1)) in chunk_task:
            i = chunk_task[int(mm.group(1))]
            bad.add(i, "content", "%s: a leftover of chunk %s" % (name, mm.group(1)))
            leftovers.setdefault(i, []).append(name)
        elif mm:
            global_problems.append("%s: a shard file of no chunk of the plan" % name)
        else:
            other_files.append(name)
    for name in sorted(tlisting):
        mm = _MARKER.match(name)
        if mm and int(mm.group(1)) < len(tasks):
            i = int(mm.group(1))
            if name == "task_%04d.json" % i:
                continue
            bad.add(i, "content", "tasks/%s: a leftover" % name)
            leftovers.setdefault(i, []).append("tasks/" + name)
        else:
            other_files.append("tasks/" + name)
    # each bad task's files on disk (what a quarantine moves)
    bad_tasks = {}
    for i in sorted(bad.items):
        files = set(leftovers.get(i, []))
        if ("task_%04d.json" % i) in tlisting:
            files.add("tasks/task_%04d.json" % i)
        for c in tasks[i]["shards"]:
            for ext in (".npz", ".json"):
                n = "shard_%04d%s" % (int(c["shard"]), ext)
                if n in listing:
                    files.add(n)
        problems = bad.items[i]
        bad_tasks[str(i)] = {"name": tasks[i]["name"],
                             "problems": [text for _, text in problems],
                             "io_only": all(kind == "io" for kind, _ in problems),
                             "files": sorted(files)}
    verdict = "OK" if not bad_tasks and not global_problems else "BAD"
    return {"tool": "bank_verify", "version": TOOL_VERSION, "verdict": verdict,
            "plan_sha256": digest, "bank_root": os.path.abspath(bank_root),
            "host": socket.gethostname(), "n_tasks": len(tasks),
            "n_files_checked": len(jobs), "n_npz_checked": n_npz, "n_members_checked": n_members,
            "bad_tasks": bad_tasks, "global_problems": global_problems,
            "other_files": other_files, "record": read_record(bank_root, digest),
            "seconds": round(time.time() - t0, 1)}


def write_report(path, report):
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="the plan of record (bank_plan.json, with its .sha256)")
    ap.add_argument("--bank-root", required=True, help="the simulated bank's folder")
    ap.add_argument("--out", required=True, help="the report (JSON) to write")
    ap.add_argument("--workers", type=int, default=4, help="processes for the sha256 and CRC checks")
    args = ap.parse_args(argv)
    if args.workers < 1:
        print("ERROR: --workers must be >= 1")
        return EXIT_BAD_ARGS
    host = socket.gethostname()
    print("[verify] host %s, plan %s, bank %s, %d worker(s)"
          % (host, args.plan, args.bank_root, args.workers))
    try:
        plan, digest = load_plan(args.plan)
        report = verify(plan, digest, args.bank_root, args.workers)
    except SuspectNode as exc:
        print("SUSPECT NODE: %s" % exc)
        return EXIT_SUSPECT
    report["plan"] = os.path.abspath(args.plan)
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for key in sorted(report["bad_tasks"], key=int):
        b = report["bad_tasks"][key]
        print("[verify] BAD task %s (%s): %s" % (key, b["name"], "; ".join(b["problems"][:4])))
    for g in report["global_problems"]:
        print("[verify] GLOBAL %s" % g)
    write_report(args.out, report)
    print("[verify] report %s" % args.out)
    if report["verdict"] == "OK":
        print("VERIFY OK: %d tasks, %d shard files (%d members' CRCs) checked on %s in %.0f s"
              % (report["n_tasks"], report["n_npz_checked"], report["n_members_checked"], host,
                 report["seconds"]))
    else:
        print("VERIFY BAD: %d task(s) [%s], %d global problem(s), on %s"
              % (len(report["bad_tasks"]), ", ".join(sorted(report["bad_tasks"], key=int)),
                 len(report["global_problems"]), host))
    return 0


if __name__ == "__main__":
    sys.exit(main())
