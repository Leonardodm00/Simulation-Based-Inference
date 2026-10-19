#!/usr/bin/env python3
"""Set a search's rounds aside and start it again from nothing (SPEC block
7, amended 2026-10-19, the user's call of 17:19: "remove everything from
the round 0 of S7"). A list-then-confirm tool with a CODE: the project's
rule that Claude never deletes data and the user deletes through such a
tool. Nothing is deleted here either: the round's folders are MOVED aside
(D-095), the trial jobs still queued or running are deleted from the
scheduler (qdel; their work stops), and the state file is left to the
submission's own `--retry search`.

    python3 hpc/pipeline/tools/search_reset.py --config <project>.json
        lists what the reset would do and prints the CODE; does nothing
    python3 hpc/pipeline/tools/search_reset.py --config <project>.json --confirm <CODE>
        does it, when the CODE still matches what is there

What is set aside, to `<campaign>_set_aside_<utc>/` beside the campaign
folder: `trials/` (the ledger), `pending/`, `runs/`, `jobs/`,
`pipeline_rounds.json`, `finalists.json`, `SEARCH_COMPLETE.json` -- each
only if present -- with a `RESET_RECORD.json` naming the jobs deleted and
the time. Kept in place: `gate/` (the budget gate's probe files, reused by
the next search: no probe jobs again) and `bank_split.json` (rewritten by
the next search anyway).

Refused (exit 4, nothing done): a sequencer job of the project
(`pl_<tag>`) queued or running -- stop it first (`--stop`); a CODE that no
longer matches (the folder or the jobs changed since the listing). Exit 2:
the configuration does not load. Exit 0 with the pass line

    SEARCH RESET DONE: <n> job(s) deleted, round(s) set aside as <dir>; next:
    submit with --retry search --accept-config-change

Stdlib only. Pure ASCII, LF only; Python 3.6 syntax.
"""

import argparse
import hashlib
import json
import os
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
import stages.common as CM              # noqa: E402
from stages.search import SearchStage   # noqa: E402

MOVED = ("trials", "pending", "runs", "jobs", "pipeline_rounds.json", "finalists.json",
         "SEARCH_COMPLETE.json")
KEPT = ("gate", "bank_split.json")
WAIT_S = 180.0


def _tree_size(path):
    """(entries, bytes) under a directory, or (1, bytes) of a file."""
    if os.path.isfile(path):
        return 1, os.path.getsize(path)
    n, b = 0, 0
    for d, _subs, files in os.walk(path):
        for f in files:
            n += 1
            try:
                b += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return n, b


def listing(cfg, pbs_module=PBS):
    """What the reset would act on: the campaign's items, the ledger's
    counts, the trial jobs the state records that the scheduler reports
    not finished, and the sequencer jobs (which must be none)."""
    stage = SearchStage()
    dirs = stage._dirs(cfg)
    camp = dirs["campaign"]
    items = []
    for name in MOVED:
        p = os.path.join(camp, name)
        if os.path.lexists(p):
            n, b = _tree_size(p)
            items.append({"name": name, "entries": n, "bytes": b})
    kept = [name for name in KEPT if os.path.lexists(os.path.join(camp, name))]
    ledger = {"ok": 0, "failed": 0, "killed": 0, "other": 0, "unreadable": 0}
    tdir = os.path.join(camp, "trials")
    if os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            if not f.endswith(".json"):
                continue
            rec, _problem = CM.try_read_json(os.path.join(tdir, f))
            if rec is None:
                ledger["unreadable"] += 1
            else:
                k = str(rec.get("status", "ok"))
                ledger[k if k in ledger else "other"] += 1
    wr = cfg.data["paths"]["work_root"]
    state = ST.State.load(wr) if os.path.isdir(wr) else None
    st = (state.data.get("stages", {}).get("search") if state is not None else None) or {}
    live = []
    for job in st.get("jobs", []):
        if job.get("finished"):
            continue
        js = pbs_module.status(job["id"])
        if not js.finished:
            live.append({"id": job["id"], "role": job.get("role", ""), "state": js.state})
    drivers = P._drivers(cfg, pbs_module)
    return {"campaign": camp, "items": items, "kept": kept, "ledger": ledger, "live": live,
            "drivers": drivers, "state_status": st.get("status"),
            "rounds": len((st.get("sub") or {}).get("rounds", []) or [])}


def code_of(lst):
    """8 hex characters of the listing's canonical text: a CODE that no
    longer matches once a file or a job has changed."""
    canon = json.dumps({"campaign": lst["campaign"], "items": lst["items"],
                        "live": sorted(j["id"] for j in lst["live"]),
                        "ledger": lst["ledger"]}, sort_keys=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()[:8]


def print_listing(lst, code):
    print("SEARCH RESET (nothing done yet): campaign folder %s" % lst["campaign"])
    print("  the state's search stage: %s, %d round(s) recorded"
          % (lst["state_status"], lst["rounds"]))
    print("  the ledger: %s" % ", ".join("%s %d" % (k, v) for k, v in sorted(lst["ledger"].items())
                                        if v))
    for it in lst["items"]:
        print("  would set aside %-24s %7d file(s) %10.1f MB" % (it["name"], it["entries"],
                                                                  it["bytes"] / 1e6))
    if not lst["items"]:
        print("  nothing to set aside: the campaign folder holds no round")
    print("  kept in place: %s" % (", ".join(lst["kept"]) or "nothing (no gate yet)"))
    for j in lst["live"]:
        print("  would DELETE job %s (%s, state %s): its work stops"
              % (j["id"], j["role"], j["state"]))
    if not lst["live"]:
        print("  no trial job queued or running per the state and the scheduler")
    print("  the state file is not touched: the submission's --retry search resets the stage")
    print("CODE %s -- run again with --confirm %s to do the above" % (code, code))


def wait_finished(ids, pbs_module=PBS, timeout=WAIT_S):
    end = time.time() + timeout
    left = list(ids)
    while left and time.time() < end:
        left = [j for j in left if not pbs_module.status(j).finished]
        if left:
            time.sleep(2.0)
    return left


def do_reset(cfg, lst, code, pbs_module=PBS):
    camp = lst["campaign"]
    deleted = []
    for j in lst["live"]:
        pbs_module.delete(j["id"])
        deleted.append(j["id"])
        print("deleted job %s (%s)" % (j["id"], j["role"]))
    left = wait_finished(deleted, pbs_module)
    if left:
        print("SEARCH RESET: %d job(s) still not finished %d s after qdel (%s); the folders are "
              "not moved -- list again when they have ended"
              % (len(left), int(WAIT_S), ", ".join(left)))
        return 4
    aside = "%s_set_aside_%s" % (camp.rstrip("/"), CM.utc_now().replace(":", ""))
    os.makedirs(aside)
    moved = []
    for it in lst["items"]:
        src = os.path.join(camp, it["name"])
        dst = os.path.join(aside, it["name"])
        os.replace(src, dst)
        moved.append(it["name"])
        print("set aside %s -> %s" % (src, dst))
    CM.write_json(os.path.join(aside, "RESET_RECORD.json"),
                  {"schema": "search_reset/1", "campaign": camp, "set_aside_utc": CM.utc_now(),
                   "code": code, "moved": moved, "kept": lst["kept"], "jobs_deleted": deleted,
                   "ledger_before": lst["ledger"], "config_sha256": cfg.sha256,
                   "host": os.uname()[1]})
    print("SEARCH RESET DONE: %d job(s) deleted, round(s) set aside as %s; next: submit with "
          "--retry search --accept-config-change" % (len(deleted), aside))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--confirm", default=None, metavar="CODE")
    args = ap.parse_args(argv)
    try:
        cfg = C.load_config(args.config)
    except C.ConfigError as exc:
        print("SEARCH RESET: the configuration %s does not load: %s" % (args.config, exc.problems))
        return 2
    try:
        lst = listing(cfg)
    except PBS.SchedulerError as exc:
        print("SEARCH RESET: the scheduler did not answer (%s); try again" % exc)
        return 4
    if lst["drivers"]:
        print("SEARCH RESET REFUSED: the sequencer is queued or running (%s); stop it first "
              "(run_pipeline.sh --project <tag> --stop --wait 180)"
              % ", ".join("%s %s" % d for d in lst["drivers"]))
        return 4
    code = code_of(lst)
    if args.confirm is None:
        print_listing(lst, code)
        return 0
    if args.confirm != code:
        print_listing(lst, code)
        print("SEARCH RESET REFUSED: the CODE %s does not match %s (the folder or the jobs changed "
              "since the listing, or a typo); nothing done" % (args.confirm, code))
        return 4
    return do_reset(cfg, lst, code)


if __name__ == "__main__":
    sys.exit(main())
