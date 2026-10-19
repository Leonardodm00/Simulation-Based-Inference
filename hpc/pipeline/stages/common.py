"""What the search, best-fit and bank stages share (blocks 7, 2): where the tuner,
the runner and the tools are, the tuner's common arguments from the
configuration, the qsub line of a one-command job, the argv files, the
JSON helpers and the clock; since 2026-10-18 the node-fault signature
(D-096) and the checkpoint check (D-097) that the trials and the members
share.

Nothing here names a project (D-091): every path and value comes from the
configuration.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import calendar
import errno
import glob
import json
import os
import re
import signal
import time
import zipfile
import zlib

import envs as E
import pbs as PBS

__all__ = ["paths_of", "tuner_common_args", "shard_globs", "run_tuner", "qsub_line",
           "write_argv_file", "read_argv_file", "read_json", "write_json", "utc_now",
           "utc_to_epoch", "elapsed_hours", "mem_to_gb", "job_name", "ToolError",
           "node_fault", "read_log_tail", "host_of_log", "checkpoint_problem",
           "set_aside_bad_checkpoints", "LOG_GRACE_S"]


class ToolError(RuntimeError):
    """A login-node tool (the tuner, a tool script) failed; carries its output."""


# ---------------------------------------------------------------------------
# Where the code is
# ---------------------------------------------------------------------------

def paths_of(cfg):
    """The code paths the stages call, all under the configuration's
    `paths.sbi_dir` (the SBI checkout the jobs run from)."""
    sbi = cfg.data["paths"]["sbi_dir"]
    pipe = os.path.join(sbi, "hpc", "pipeline")
    return {
        "sbi_dir": sbi,
        "stage4": os.path.join(sbi, "hpc", "joint", "stage4"),
        "tuner": os.path.join(sbi, "hpc", "joint", "stage4", "npe_tune_joint.py"),
        "runner": os.path.join(sbi, "hpc", "joint", "stage3", "run_joint_arms.py"),
        "split_tool": os.path.join(pipe, "tools", "bank_split_hash.py"),
        "ensemble_tool": os.path.join(pipe, "tools", "ensemble_nll.py"),
        "verify_tool": os.path.join(pipe, "tools", "bank_verify.py"),
        "job_script": os.path.join(pipe, "jobs", "run_argv.pbs"),
    }


def shard_globs(cfg):
    """(sim glob, real glob or None): the real bank goes to the runner only
    for a campaign whose arm needs it (S-A1 trains on simulated windows
    alone, D-039; A2/A5 need the recordings)."""
    d = cfg.data
    tag = d["dataset"]
    sim = os.path.join(d["paths"]["banks_root"], tag + "_sim", "shard_*.npz")
    real = os.path.join(d["paths"]["banks_root"], tag + "_real", "shard_*.npz")
    return sim, (real if d["search"]["campaign"] != "S-A1" else None)


def tuner_common_args(cfg, identity):
    """The tuner's `common` arguments, identical on EVERY call of a search
    (the space, the identity inputs of the trial ids, the training seed):
    a difference between two calls would orphan the pending specs."""
    d = cfg.data
    s = d["search"]
    return ["--campaign", s["campaign"],
            "--results-dir", d["paths"]["tune_root"],
            "--p", str(int(s["shape_anchors"]["p"])),
            "--embedding-dim", str(int(s["shape_anchors"]["embedding_dim"])),
            "--d-theta", str(int(s["shape_anchors"]["d_theta"])),
            "--n-train", str(int(identity["n_train"])),
            "--split-hash", identity["split_hash"],
            "--contract-digest", identity["contract_digest"],
            "--seed", str(int(s["train_seed"])),
            "--space-overrides", json.dumps(s["space_overrides"], sort_keys=True,
                                            separators=(",", ":"))]


def run_tuner(cfg, args, log_path=None, timeout=None):
    """Run `npe_tune_joint.py <args>` in `envs.train` from the stage4
    directory; return its output; raise ToolError on a non-zero exit."""
    p = paths_of(cfg)
    argv = [E.python_in_env(), p["tuner"]] + [str(a) for a in args]
    rc, out = E.run_in_env(cfg.data["envs"]["train"], argv, cwd=p["stage4"],
                           log_path=log_path, timeout=timeout)
    if rc != 0:
        raise ToolError("npe_tune_joint.py %s exited %d:\n%s"
                        % (args[0] if args else "", rc, out.strip()[-2000:]))
    return out


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

_NAME_OK = re.compile(r"[^A-Za-z0-9_]")


def job_name(text, limit=15):
    """A PBS job name: [A-Za-z0-9_], at most `limit` characters (PBS Pro's
    classic limit), starting with a letter."""
    t = _NAME_OK.sub("_", str(text))[:limit]
    if not t or not t[0].isalpha():
        t = ("j" + t)[:limit]
    return t


def qsub_line(cfg, name, queue, ncpus, mem, walltime, log_path, argv_file, workdir,
              env_name=None):
    """The complete qsub command line of one run_argv.pbs job, in the
    environment `env_name` (default the configuration's envs.train: the
    trials, the probes and the members; the bank's verification job passes
    envs.extract)."""
    p = paths_of(cfg)
    skip = "1" if E.skip_conda() else "0"
    variables = "ARGV_FILE=%s,ENV_NAME=%s,WORKDIR=%s,SKIP_CONDA=%s" % (
        argv_file, env_name or cfg.data["envs"]["train"], workdir, skip)
    for v in (argv_file, workdir):
        if "," in v:
            raise ValueError("a path with a comma cannot be passed through qsub -v: %r" % v)
    return ["qsub", "-N", job_name(name), "-q", str(queue),
            "-l", "select=1:ncpus=%d:mem=%s" % (int(ncpus), mem),
            "-l", "walltime=%s" % walltime, "-j", "oe", "-o", log_path,
            "-v", variables, p["job_script"]]


def write_argv_file(path, argv):
    """One token per line, atomically; a token with a newline is refused."""
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    for a in argv:
        if "\n" in a:
            raise ValueError("an argument with a newline cannot go in an argv file: %r" % a)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(str(a) for a in argv) + "\n")
    os.replace(tmp, path)
    return path


_PBS_SELECT_LINE = re.compile(r"^#PBS\s+-l\s+(\S*select=\S+|\S*ncpus=\S+)")


def pbs_header_ncpus(path):
    """The cores a job script asks for in its `#PBS -l select=...` header
    (D-099: a launcher whose qsub line passes no -l uses its script's).
    RuntimeError when the file has none."""
    try:
        with open(path) as fh:
            for line in fh:
                m = _PBS_SELECT_LINE.match(line.strip())
                if m:
                    n = PBS.cores_of_select(m.group(1))
                    if n is not None:
                        return n
    except OSError as exc:
        raise RuntimeError("cannot read the cores of %s: %s" % (path, exc))
    raise RuntimeError("cannot read the cores of %s: no #PBS -l select=...:ncpus=N line" % path)


def shell_default(path, var):
    """The default a launcher gives `var`, from its `VAR="${VAR:-value}"`
    line (D-099: the cores it asks for unless told otherwise). RuntimeError
    when the file has no such line."""
    pat = re.compile(r'^\s*%s="\$\{%s:-([^}]*)\}"' % (re.escape(var), re.escape(var)))
    try:
        with open(path) as fh:
            for line in fh:
                m = pat.match(line)
                if m:
                    return m.group(1)
    except OSError as exc:
        raise RuntimeError("cannot read the default of %s in %s: %s" % (var, path, exc))
    raise RuntimeError("cannot read the default of %s in %s: no %s=\"${%s:-...}\" line"
                       % (var, path, var, var))


def stage_budget(cfg):
    """The cores the stages' jobs may hold at once: resources.max_cores less
    the sequencer's own job (D-099); None without a cap."""
    d = cfg.data
    if "resources" not in d:
        return None
    return int(d["resources"]["max_cores"]) - int(d["driver"]["ncpus"])


def cap_room(cfg, ncpus, limit=None):
    """How many jobs of `ncpus` cores the cap holds at once, at most `limit`
    (the dry runs' line; D-099)."""
    b = stage_budget(cfg)
    k = (b // int(ncpus)) if b is not None else None
    if limit is None:
        return k
    return int(limit) if k is None else min(int(limit), k)


def read_argv_file(path):
    with open(path) as fh:
        return [line.rstrip("\n") for line in fh if line.rstrip("\n") != ""]


# ---------------------------------------------------------------------------
# JSON, time, memory
# ---------------------------------------------------------------------------

def read_json(path):
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("ascii", "replace"))


def write_json(path, obj):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    tmp = path + ".tmp"
    blob = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True)
    with open(tmp, "w") as fh:
        fh.write(blob + "\n")
    os.replace(tmp, path)
    return path


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_to_epoch(text):
    return float(calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))


def elapsed_hours(start_utc, now_utc=None):
    now = utc_to_epoch(now_utc) if now_utc else time.time()
    return (now - utc_to_epoch(start_utc)) / 3600.0


_MEM = re.compile(r"^([0-9]+)(kb|mb|gb|tb)$")


def mem_to_gb(text):
    m = _MEM.match(str(text))
    if not m:
        raise ValueError("not a PBS memory: %r" % (text,))
    n, unit = int(m.group(1)), m.group(2)
    return n * {"kb": 1.0 / (1024.0 ** 2), "mb": 1.0 / 1024.0, "gb": 1.0, "tb": 1024.0}[unit]


def walltime_seconds(text):
    return PBS.walltime_to_seconds(text)


# ---------------------------------------------------------------------------
# Node faults and checkpoints (D-096, D-097; SPEC block 7, amended 2026-10-18)
# ---------------------------------------------------------------------------

# A failed trial's job log is awaited this long after the failure is seen,
# for the scheduler's copy of the log to land (as the bank stage's grace).
LOG_GRACE_S = 300.0

_SIGBUS = int(getattr(signal, "SIGBUS", 7))
# SIGBUS seen through an exit code: bash's 128 + n for a direct child (a
# member's runner), Python's sys.exit(-n) & 0xFF for the tuner's own exit.
_SIGBUS_EXITS = (128 + _SIGBUS, 256 - _SIGBUS)
_FS_ERRNOS = tuple(sorted(set(getattr(errno, n) for n in
                              ("EIO", "ENOTCONN", "ESHUTDOWN", "ESTALE", "EREMOTEIO")
                              if hasattr(errno, n))))
_FS_ERRNO_RE = re.compile(r"\[Errno (?:%s)\]" % "|".join(str(n) for n in _FS_ERRNOS))
_FS_TEXTS = ("Input/output error", "Transport endpoint is not connected",
             "Cannot send after transport endpoint shutdown", "Stale file handle",
             "Could not chdir to home directory", "Bus error")
_RUNNER_EXITED = re.compile(r"^runner exited (-?[0-9]+)$")
_RUN_ARGV_EXIT = re.compile(r"^RUN_ARGV EXIT: ([0-9]+)\s*$", re.M)
_RUN_ARGV_HOST = re.compile(r"^RUN_ARGV: host ([^,\s]+)", re.M)


def _line_at(text, i):
    a = text.rfind("\n", 0, i) + 1
    b = text.find("\n", i)
    return (text[a:] if b < 0 else text[a:b]).strip()[:200]


def node_fault(record_error=None, log_text=None):
    """The node-fault signature: a short reason when a failure is the
    node's -- the runner killed by SIGBUS, or a file-system error (EIO,
    ENOTCONN, ESHUTDOWN, ESTALE, EREMOTEIO, or their texts) -- else None.

    `record_error` is a ledger record's `error` (the tuner writes `runner
    exited <returncode>`, negative for a signal); `log_text` the job's log.
    Not node faults: any other exit, SIGKILL (memory, an operator), a full
    disk (ENOSPC, EDQUOT), a missing file (ENOENT)."""
    m = _RUNNER_EXITED.match(str(record_error or "").strip())
    if m and int(m.group(1)) == -_SIGBUS:
        return "SIGBUS: runner exited %s" % m.group(1)
    text = log_text or ""
    for mm in _RUN_ARGV_EXIT.finditer(text):
        if int(mm.group(1)) in _SIGBUS_EXITS:
            return "SIGBUS: RUN_ARGV EXIT %s" % mm.group(1)
    m = _FS_ERRNO_RE.search(text)
    if m:
        return _line_at(text, m.start())
    for t in _FS_TEXTS:
        i = text.find(t)
        if i >= 0:
            return _line_at(text, i)
    return None


def read_log_tail(path, max_bytes=262144):
    """The last `max_bytes` of a job log as text, or None when it is absent
    or unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            n = fh.tell()
            fh.seek(max(0, n - int(max_bytes)))
            return fh.read().decode("ascii", "replace")
    except OSError:
        return None


def host_of_log(text):
    """The node a run_argv.pbs job ran on, from its `RUN_ARGV: host` line."""
    m = _RUN_ARGV_HOST.search(text or "")
    return m.group(1) if m else None


def checkpoint_problem(path):
    """(kind, why) for a training checkpoint, which torch.save writes as a
    zip archive of stored members with their CRC-32: kind None when the
    archive opens and every member's CRC matches; "content" when it is not
    a readable zip archive or a member fails; "io" when this node cannot
    read the file (an OSError: the file is not judged)."""
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except zipfile.BadZipFile as exc:
        return "content", "not a readable zip archive: %s" % exc
    except (zlib.error, EOFError, NotImplementedError, ValueError) as exc:
        return "content", "%s: %s" % (type(exc).__name__, exc)
    except OSError as exc:
        return "io", "unreadable here: %s" % exc
    if bad is not None:
        return "content", "member %s fails its CRC-32" % bad
    return None, ""


def set_aside_bad_checkpoints(run_dir, log, stage, who):
    """D-097: every `*_train_ckpt.pt` of `run_dir` whose archive is damaged
    is moved to `<name>.corrupt<k>` (kept; never deleted), so the job that
    follows starts from epoch 0; a file this node cannot read is left in
    place. Returns the paths moved to."""
    moved = []
    for p in sorted(glob.glob(os.path.join(glob.escape(run_dir), "*_train_ckpt.pt"))):
        kind, why = checkpoint_problem(p)
        if kind is None:
            continue
        if kind == "io":
            log("%s: %s checkpoint %s could not be checked (%s); left in place"
                % (stage, who, os.path.basename(p), why))
            continue
        k = len(glob.glob(glob.escape(p) + ".corrupt*")) + 1
        dst = "%s.corrupt%d" % (p, k)
        os.replace(p, dst)
        log("%s: %s checkpoint %s failed its CRC check (%s); moved aside as %s; it restarts "
            "from epoch 0" % (stage, who, os.path.basename(p), why, os.path.basename(dst)))
        moved.append(dst)
    return moved
