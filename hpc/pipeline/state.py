"""The run's state file `<work_root>/state.json` (specs/SPEC.md sec. 3.2).

Written atomically: the whole document to a temporary file, then
`os.replace`, so a sequencer killed mid-write leaves the previous state
intact. ASCII JSON with sorted keys, so two states can be diffed.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import json
import os
import socket
import time

__all__ = ["State", "STATUSES", "utc_now", "STATE_SCHEMA"]

STATE_SCHEMA = "pipeline_state/1"
STATUSES = ("pending", "running", "done", "failed", "skipped")
STATE_FILE = "state.json"


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_stage():
    return {"status": "pending", "started_utc": None, "finished_utc": None,
            "jobs": [], "resubmits": 0, "note": "", "sub": {}}


class State(object):
    """In-memory state with load/save; one instance per sequencer."""

    def __init__(self, work_root, data):
        self.work_root = work_root
        self.data = data

    # -- construction --------------------------------------------------------

    @classmethod
    def path_of(cls, work_root):
        return os.path.join(work_root, STATE_FILE)

    @classmethod
    def new(cls, work_root, dataset, config_path, config_sha256):
        now = utc_now()
        return cls(work_root, {
            "schema": STATE_SCHEMA,
            "dataset": dataset,
            "config_path": config_path,
            "config_sha256": config_sha256,
            "started_utc": now,
            "updated_utc": now,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "stages": {},
            "events": [],
        })

    @classmethod
    def load(cls, work_root):
        """The state on disk, or None when there is none yet."""
        p = cls.path_of(work_root)
        if not os.path.isfile(p):
            return None
        with open(p, "rb") as fh:
            data = json.loads(fh.read().decode("ascii"))
        if data.get("schema") != STATE_SCHEMA:
            raise ValueError("%s: unknown state schema %r" % (p, data.get("schema")))
        return cls(work_root, data)

    def save(self):
        self.data["updated_utc"] = utc_now()
        self.data["pid"] = os.getpid()
        self.data["host"] = socket.gethostname()
        if not os.path.isdir(self.work_root):
            os.makedirs(self.work_root)
        p = self.path_of(self.work_root)
        tmp = p + ".tmp"
        blob = json.dumps(self.data, indent=1, sort_keys=True,
                          ensure_ascii=True).encode("ascii")
        with open(tmp, "wb") as fh:
            fh.write(blob)
            fh.write(b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
        return p

    # -- stages --------------------------------------------------------------

    def stage(self, name):
        """The stage's record, created as pending when absent."""
        st = self.data["stages"].get(name)
        if st is None:
            st = _new_stage()
            self.data["stages"][name] = st
        return st

    def set_status(self, name, status, note=None):
        if status not in STATUSES:
            raise ValueError("unknown status %r" % status)
        st = self.stage(name)
        st["status"] = status
        now = utc_now()
        if status == "running" and st["started_utc"] is None:
            st["started_utc"] = now
        if status in ("done", "failed", "skipped"):
            st["finished_utc"] = now
        if note is not None:
            st["note"] = note
        self.event("%s -> %s%s" % (name, status, (": " + note) if note else ""))

    def add_job(self, name, job_id, role, cmd, log=None):
        st = self.stage(name)
        st["jobs"].append({"id": str(job_id), "role": role,
                           "submitted_utc": utc_now(), "cmd": list(cmd),
                           "log": log, "finished": False, "exit_status": None,
                           "walltime_s": None})
        self.event("%s: submitted %s as %s" % (name, role, job_id))

    def reset_stage(self, name):
        """Back to pending with no jobs (the user's --retry)."""
        self.data["stages"][name] = _new_stage()
        self.event("%s reset to pending" % name)

    def event(self, text):
        self.data["events"].append([utc_now(), text])

    def statuses(self):
        return dict((k, v["status"]) for k, v in self.data["stages"].items())
