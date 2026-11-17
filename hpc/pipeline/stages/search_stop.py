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

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

__all__ = ["STOP_FILE", "SCHEMA", "request_path", "read_request", "write_request",
           "clear_request", "status_line", "cli"]

STOP_FILE = "STOP_REQUEST.json"
SCHEMA = "search_stop/1"


def request_path(view):
    """`<tune_root>/<search.campaign>/STOP_REQUEST.json` of the view."""
    raise NotImplementedError("block 30")


def read_request(view):
    """The request's content (a dict with schema, requested_utc,
    kill_running, user, host), or None when there is no file."""
    raise NotImplementedError("block 30")


def write_request(view, kill_running, user, host, now_utc):
    """Write the request atomically (a temporary file, then os.replace);
    return its content."""
    raise NotImplementedError("block 30")


def clear_request(view):
    """Remove the request; True when a file was removed."""
    raise NotImplementedError("block 30")


def status_line(view):
    """`stop requested <utc> (kill running: yes|no)` while a request stands,
    with `; the search ended (user_stop)` once SEARCH_COMPLETE.json says so;
    None without a request."""
    raise NotImplementedError("block 30")


def cli(cfg, search, kill_running, cancel, out=print):
    """The command (specs/SPEC.md block 30): 0 with the line `STOP REQUESTED:
    ...` or `STOP CANCELLED: ...`, 2 with `REFUSED: ...`."""
    raise NotImplementedError("block 30")
