"""Stop a best fit by hand (block 30, amendment 6; D-185; specs/SPEC.md block 30).

    python3 hpc/pipeline/pipeline.py stop-best-fit --config <cfg> [--search <name>]

The command writes one file, STOP_REQUEST.json, in the best fit's root
`<runs_root>/<dataset>_best` of the search's view (the folder of its
finalist_config.json, its members and BEST_FIT_COMPLETE.json), creating
the root when it does not exist yet; it runs on any login node and submits
nothing (D-098). The sequencer reads the file at its next tick (stages/
best_fit.py): the best fit is abandoned -- the running members' jobs are
deleted, no member is submitted or resubmitted, the waiting members are
cancelled, no ensemble is computed -- and once no member job runs the
stage writes BEST_FIT_STOPPED.json and ends done, so that its cores go
back to the stages still running.

    request_path(view)                          the request's file
    stopped_path(view)                          the stopped record's file
    read_request(view)                          its content, or None
    write_request(view, finalist, user, host, now_utc)   atomic; the content
    read_stopped(view)                          (record, problem), or (None, None)
    status_line(view)                           one line for `status`, or None
    cli(cfg, search, out=print)                 the exit code
    delete_job(pbs, job_id)                     the stage's deletion (block 30's rule)

A file that exists and cannot be read as a JSON object (written by hand, or
damaged) still stands as a request: read_request gives it with its
`problem` and no finalist, and the stage acts on it as a request for the
trial it fits (and logs the problem).

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import getpass
import os
import socket

from . import common as CM

__all__ = ["STOP_FILE", "STOPPED_FILE", "SCHEMA", "STOPPED_SCHEMA", "STOP_KILLED",
           "STOP_ENDED", "request_path", "stopped_path", "read_request", "read_request_at",
           "write_request", "read_stopped", "status_line", "cli", "delete_job"]

STOP_FILE = "STOP_REQUEST.json"
STOPPED_FILE = "BEST_FIT_STOPPED.json"
SCHEMA = "best_fit_stop/1"
STOPPED_SCHEMA = "best_fit_stopped/1"
# the reasons of a member killed under a stop request (its `why`)
STOP_KILLED = "stopped by the user (stop-best-fit)"
STOP_ENDED = "ended without a record after a stop request"
# the search's and the best fit's artefacts of record (stages/search.py and
# stages/best_fit.py; not imported from there: best_fit.py imports this module)
SEARCH_COMPLETE_FILE = "SEARCH_COMPLETE.json"
COMPLETE_FILE = "BEST_FIT_COMPLETE.json"
ROOT_SUFFIX = "_best"           # BestFitStage.root_suffix (S8, F_q)

EXIT_OK, EXIT_REFUSED = 0, 2


def _root(view):
    d = view.data
    return os.path.join(d["paths"]["runs_root"], d["dataset"] + ROOT_SUFFIX)


def _search_complete_path(view):
    d = view.data
    return os.path.join(d["paths"]["tune_root"], d["search"]["campaign"], SEARCH_COMPLETE_FILE)


def request_path(view):
    """`<runs_root>/<dataset>_best/STOP_REQUEST.json` of the view."""
    return os.path.join(_root(view), STOP_FILE)


def stopped_path(view):
    """`<runs_root>/<dataset>_best/BEST_FIT_STOPPED.json` of the view."""
    return os.path.join(_root(view), STOPPED_FILE)


def _trial_of(fin):
    """The `trial_id` of a finalist object, or None (best_fit._trial_of)."""
    if isinstance(fin, dict) and fin.get("trial_id"):
        return fin["trial_id"]
    return None


def read_request(view):
    """The request's content (a dict with schema, requested_utc, finalist,
    user, host), or None when there is no file. A file that cannot be read
    as a JSON object gives {schema, requested_utc None, finalist None, user
    None, host None, problem}; one removed while it was read gives None."""
    return read_request_at(request_path(view))


def read_request_at(p):
    """read_request on the file `p` (the stage reads its own root's)."""
    if not os.path.lexists(p):
        return None
    rec, problem = CM.try_read_json(p)
    if rec is None:
        if not os.path.lexists(p):
            return None
        return {"schema": SCHEMA, "requested_utc": None, "finalist": None, "user": None,
                "host": None, "problem": problem}
    return rec


def write_request(view, finalist, user, host, now_utc):
    """Write the request atomically (common.write_json: a temporary file in
    the same folder, fsynced, then os.replace; the folder created when
    absent); return its content."""
    content = {"schema": SCHEMA, "requested_utc": str(now_utc), "finalist": str(finalist),
               "user": str(user), "host": str(host)}
    CM.write_json(request_path(view), content)
    return content


def read_stopped(view):
    """(record, None) when BEST_FIT_STOPPED.json can be read; (None, problem)
    when it exists and cannot; (None, None) when there is none."""
    p = stopped_path(view)
    if not os.path.lexists(p):
        return None, None
    rec, problem = CM.try_read_json(p)
    if rec is None:
        return None, problem or "unreadable"
    return rec, None


def status_line(view):
    """`stop requested <utc> (finalist <tid>)` while a request stands, with
    `; stopped (BEST_FIT_STOPPED.json)` once a stopped record names that
    finalist; the file's problem when it cannot be read; None without a
    request."""
    req = read_request(view)
    if req is None:
        return None
    line = "stop requested %s (finalist %s)" % (req.get("requested_utc") or "?",
                                               req.get("finalist") or "?")
    rec, _ = read_stopped(view)
    if rec is not None and (req.get("finalist") is None
                            or _trial_of(rec.get("finalist")) == req.get("finalist")):
        line += "; stopped (%s)" % STOPPED_FILE
    if req.get("problem"):
        line += ("; its file %s cannot be read (%s): taken as a request for the trial the best "
                 "fit fits" % (request_path(view), req["problem"]))
    return line


def delete_job(pbs, job_id):
    """pbs.delete: the recorder's own when it has one, else the scheduler
    adapter's (`pbs.py` delete: qdel; False for a job already finished) --
    block 30's rule (stages/search.py `_delete_job`). True when the
    scheduler accepted the deletion. A scheduler that cannot be asked
    raises SchedulerError: the sequencer retries the poll at its next tick,
    and the deletion with it."""
    fn = getattr(pbs, "delete", None)
    if not callable(fn):
        fn = CM.PBS.delete
    return bool(fn(job_id))


def _user():
    try:
        return getpass.getuser()
    except (KeyError, OSError, ImportError):
        return os.environ.get("USER") or "?"


def cli(cfg, search, out=print):
    """The command (specs/SPEC.md block 30, amendment 6): 0 with the line `STOP REQUESTED:
    ...`, 2 with `REFUSED: ...` (nothing written)."""
    # imported here: runs.py builds the views and imports the stages
    from . import runs as RU
    import config as C

    def refuse(text):
        out("REFUSED: %s; nothing was written" % text)
        return EXIT_REFUSED

    if C.is_synthetic(cfg.data):
        return refuse("%s is a synthetic configuration: its runs carry a second best fit (block "
                      "18) which this command does not reach; stop-best-fit is for a recorded "
                      "configuration" % cfg.path)
    views = RU.search_views(cfg)
    names = [n for n, _ in views if n is not None]
    if names:
        if search is None or search not in names:
            return refuse("%sthis configuration has the searches %s: give --search <name>"
                          % ("" if search is None else "no search is named %r; " % search,
                             ", ".join(names)))
        view = dict(views)[search]
    else:
        view = views[0][1]
        if search is not None:
            return refuse("--search %s: this configuration has a single search (searches null; "
                          "campaign %s): do not give --search"
                          % (search, view.data["search"]["campaign"]))
    camp = view.data["search"]["campaign"]
    label = "best fit of search %s" % (search if search is not None else camp)
    sc_path = _search_complete_path(view)
    if not os.path.lexists(sc_path):
        return refuse("%s: the search has not ended (no %s), so its best fit has not started; "
                      "stop the search first (stop-search)" % (label, sc_path))
    sc, problem = CM.try_read_json(sc_path)
    if sc is None:
        return refuse("%s: %s cannot be read (%s): the finalist is not known"
                      % (label, sc_path, problem))
    finalist = _trial_of(sc.get("finalist"))
    if finalist is None:
        return refuse("%s: %s names no finalist" % (label, sc_path))
    complete = os.path.join(_root(view), COMPLETE_FILE)
    if os.path.lexists(complete):
        rec, _ = CM.try_read_json(complete)
        if rec is not None and _trial_of(rec.get("finalist")) == finalist:
            return refuse("%s: the best fit of trial %s is complete (%s): there is nothing to "
                          "stop" % (label, finalist, complete))
    rec, _ = read_stopped(view)
    if rec is not None and _trial_of(rec.get("finalist")) == finalist:
        return refuse("%s: the best fit of trial %s was stopped already (%s)"
                      % (label, finalist, stopped_path(view)))
    tail = ("at its next tick (every driver.poll_seconds) the sequencer deletes the running "
            "member jobs, submits none and ends the stage without an ensemble")
    prev = read_request(view)
    if prev is not None and not prev.get("problem") and prev.get("finalist") == finalist:
        out("STOP REQUESTED: %s (finalist %s); %s (already requested %s; unchanged)"
            % (label, finalist, tail, prev.get("requested_utc") or "?"))
        return EXIT_OK
    write_request(view, finalist, _user(), socket.gethostname(), CM.utc_now())
    line = "STOP REQUESTED: %s (finalist %s); %s" % (label, finalist, tail)
    if prev is not None:
        if prev.get("problem"):
            line += ("; it replaces a request file that could not be read (%s)"
                     % prev["problem"])
        else:
            line += ("; it replaces the request of %s for trial %s"
                     % (prev.get("requested_utc") or "?", prev.get("finalist") or "?"))
    out(line)
    return EXIT_OK
