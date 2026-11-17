"""Stop a search by hand (block 30; D-148 (2), D-141, D-150; specs/SPEC.md
block 30).

    python3 hpc/pipeline/pipeline.py stop-search --config <cfg>
        [--search <name>] [--kill-running] [--cancel]

The command writes (or with --cancel removes) one file, STOP_REQUEST.json,
in the search's campaign folder `<tune_root>/<campaign>` of its view (the
folder of its ledger and of SEARCH_COMPLETE.json); it runs on any login
node and submits nothing (D-098). The sequencer reads the file at its next
tick (stages/search.py): no new trial, no resubmission and no new round;
with kill_running the running trials' jobs are deleted; when no trial runs
any more, the waiting trials are cancelled and the search finishes as at
the clock cap, ended_by `user_stop`, so its best fit starts.

    request_path(view)                   the file
    read_request(view)                   its content, or None
    write_request(view, kill_running, user, host, now_utc)   atomic; the content
    clear_request(view)                  True when a file was removed
    status_line(view)                    one line for `status`, or None
    cli(cfg, search, kill_running, cancel, out=print)        the exit code
    kills_running(req)                   the request's kill_running as the stage reads it

A file that exists and cannot be read as a JSON object (written by hand, or
damaged) still stands as a request, without kill_running: read_request gives
it with its `problem`, and the stage acts on it (and logs the problem).

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import getpass
import os
import socket

from . import common as CM

__all__ = ["STOP_FILE", "SCHEMA", "request_path", "read_request", "write_request",
           "clear_request", "status_line", "cli", "kills_running"]

STOP_FILE = "STOP_REQUEST.json"
SCHEMA = "search_stop/1"
# the search's artefact of record (stages/search.py COMPLETE_FILE; not imported
# from there: search.py imports this module)
COMPLETE_FILE = "SEARCH_COMPLETE.json"

EXIT_OK, EXIT_REFUSED = 0, 2


def _campaign_dir(view):
    d = view.data
    return os.path.join(d["paths"]["tune_root"], d["search"]["campaign"])


def request_path(view):
    """`<tune_root>/<search.campaign>/STOP_REQUEST.json` of the view."""
    return os.path.join(_campaign_dir(view), STOP_FILE)


def kills_running(req):
    """The request's kill_running as the stage acts on it: True only for a
    JSON true (a missing, damaged or other value is no)."""
    return isinstance(req, dict) and req.get("kill_running") is True


def read_request(view):
    """The request's content (a dict with schema, requested_utc,
    kill_running, user, host), or None when there is no file. A file that
    cannot be read as a JSON object gives {schema, requested_utc None,
    kill_running False, user None, host None, problem}; one removed while it
    was read (a --cancel) gives None."""
    p = request_path(view)
    if not os.path.lexists(p):
        return None
    rec, problem = CM.try_read_json(p)
    if rec is None:
        if not os.path.lexists(p):
            return None
        return {"schema": SCHEMA, "requested_utc": None, "kill_running": False,
                "user": None, "host": None, "problem": problem}
    return rec


def write_request(view, kill_running, user, host, now_utc):
    """Write the request atomically (a temporary file, then os.replace);
    return its content."""
    content = {"schema": SCHEMA, "requested_utc": str(now_utc),
               "kill_running": bool(kill_running), "user": str(user), "host": str(host)}
    # common.write_json: ASCII JSON, sorted keys, `<file>.tmp` in the same
    # folder, fsynced, then os.replace (the stage never reads a half file)
    CM.write_json(request_path(view), content)
    return content


def clear_request(view):
    """Remove the request; True when a file was removed."""
    try:
        os.remove(request_path(view))
    except FileNotFoundError:
        return False
    return True


def _ended(view):
    """(True, its ended_by) when the search's SEARCH_COMPLETE.json exists;
    (False, None) otherwise. A file that cannot be read has ended all the
    same; its ended_by is then named as unknown, with the problem."""
    p = os.path.join(_campaign_dir(view), COMPLETE_FILE)
    if not os.path.lexists(p):
        return False, None
    rec, problem = CM.try_read_json(p)
    if rec is None:
        return True, "unknown (%s cannot be read: %s)" % (p, problem)
    return True, rec.get("ended_by")


def _yes_no(flag):
    return "yes" if flag else "no"


def status_line(view):
    """`stop requested <utc> (kill running: yes|no)` while a request stands,
    with `; the search ended (user_stop)` once SEARCH_COMPLETE.json says so;
    None without a request."""
    req = read_request(view)
    if req is None:
        return None
    line = "stop requested %s (kill running: %s)" % (req.get("requested_utc") or "?",
                                                     _yes_no(kills_running(req)))
    ended, how = _ended(view)
    if ended and how == "user_stop":
        line += "; the search ended (user_stop)"
    if req.get("problem"):
        line += "; its file %s cannot be read (%s): taken as kill running: no" % (
            request_path(view), req["problem"])
    return line


def _user():
    try:
        return getpass.getuser()
    except (KeyError, OSError, ImportError):
        # no login name and no password entry for the uid (getpass's failures)
        return os.environ.get("USER") or "?"


def cli(cfg, search, kill_running, cancel, out=print):
    """The command (specs/SPEC.md block 30): 0 with the line `STOP REQUESTED:
    ...` or `STOP CANCELLED: ...`, 2 with `REFUSED: ...`."""
    # imported here: stages/search.py imports this module; runs.py is the views'
    from . import runs as RU
    from . import search as SS

    def refuse(text):
        out("REFUSED: %s" % text)
        return EXIT_REFUSED

    if cancel and kill_running:
        return refuse("--cancel withdraws a stop request and --kill-running asks one: give "
                      "one of them; nothing was written")
    views = RU.search_views(cfg)
    names = [n for n, _ in views if n is not None]
    if names:
        if search is None or search not in names:
            return refuse("%sthis configuration has the searches %s: give --search <name>; "
                          "nothing was written"
                          % ("" if search is None else "no search is named %r; " % search,
                             ", ".join(names)))
        view = dict(views)[search]
    else:
        view = views[0][1]
        if search is not None:
            return refuse("--search %s: this configuration has a single search (searches null; "
                          "campaign %s): do not give --search; nothing was written"
                          % (search, view.data["search"]["campaign"]))
    camp = view.data["search"]["campaign"]
    label = "search %s (%s)" % (search if search is not None else camp, camp)
    ended, how = _ended(view)
    if ended:
        return refuse("%s has ended (ended_by %s; %s): %s; nothing was written"
                      % (label, how, os.path.join(_campaign_dir(view), COMPLETE_FILE),
                         "its stop request, if any, stays" if cancel else "there is nothing to stop"))
    if cancel:
        req = read_request(view)
        if req is None:
            return refuse("%s has no stop request (%s): nothing to cancel"
                          % (label, request_path(view)))
        clear_request(view)
        out("STOP CANCELLED: %s: the request of %s (kill running: %s) is withdrawn (%s removed); "
            "the sequencer submits trials again at its next tick; a trial already deleted by it "
            "stays killed" % (label, req.get("requested_utc") or "?", _yes_no(kills_running(req)),
                              request_path(view)))
        return EXIT_OK
    if not SS.completed_trials(view):
        return refuse("%s has no completed trial yet (no record of status ok in %s): nothing to "
                      "fit yet; stop the whole run with `stop` instead; nothing was written"
                      % (label, SS.SearchStage()._dirs(view)["ledger"]))
    prev = read_request(view)
    content = write_request(view, kill_running, _user(), socket.gethostname(), CM.utc_now())
    line = ("STOP REQUESTED: %s; kill running: %s; the sequencer acts at its next tick (every "
            "driver.poll_seconds)" % (label, _yes_no(content["kill_running"])))
    if prev is not None:
        line += ("; it replaces the request of %s (kill running: %s)"
                 % (prev.get("requested_utc") or "?", _yes_no(kills_running(prev))))
    out(line)
    return EXIT_OK
