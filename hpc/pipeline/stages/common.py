"""What the search, best-fit and bank stages share (blocks 7, 2): where the tuner,
the runner and the tools are, the tuner's common arguments from the
configuration, the qsub line of a one-command job, the argv files, the
JSON helpers and the clock; since 2026-10-18 the node-fault signature
(D-096) and the checkpoint check (D-097) that the trials and the members
share; since 2026-10-19 the reading of a record a node fault may have
left without its bytes (`try_read_json`, `unreadable_record`) and durable
writes (`write_json` fsyncs before its rename); since 2026-10-21 the
bad-node avoid list (D-102): a host with AVOID_FAULTS node faults in a
stage's run is written to `<work_root>/avoid_nodes.txt`, a job that lands
there holds its slot AVOID_HOLD_S seconds and exits AVOID_EXIT, and the
stage submits it again without counting a resubmission; since 2026-10-29
environment faults (D-111): a trial's or a member's job that could not
start its environment, or that PBS could not start, is submitted again
without counting a resubmission, and a stage whose jobs fault so on many
nodes in a row stops (`env_fault`, `note_env_fault`); every job carries
the variables of the chain's helpers (`qsub_line`). Since 2026-10-29 the
same for the jobs Sbi-extractor's launchers submit (D-113): the
variables that put them under the job protocol (`stage_job_protocol`,
jobs/stage_job.sh), and the free rounds of a stage whose tasks could not
start (`free_round`, `counted_round`).

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
           "write_argv_file", "read_argv_file", "read_json", "try_read_json", "write_json",
           "set_aside", "utc_now", "utc_to_epoch", "elapsed_hours", "mem_to_gb", "job_name",
           "ToolError", "node_fault", "read_log_tail", "host_of_log", "checkpoint_problem",
           "set_aside_bad_checkpoints", "unreadable_record", "LOG_GRACE_S", "RECORD_GRACE_S",
           "AVOID_FAULTS", "AVOID_HOLD_S", "AVOID_EXIT", "AVOID_FILE_NAME", "avoid_file",
           "read_avoid_list", "add_to_avoid_list", "note_node_fault", "sync_avoid_list",
           "held_off_node", "chain_name", "ENV_FAULT_EXIT", "JOB_EXEC_FAULTS",
           "ENV_FAULT_STREAK", "ENV_FAULT_HOSTS", "env_fault", "note_env_fault",
           "clear_env_fault_streak", "stage_job_protocol", "PROTOCOL_VARS", "FREE_ROUND_STALL",
           "free_round", "counted_round", "unrun_dependent"]


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
        "jobs_dir": os.path.join(pipe, "jobs"),
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


def chain_name(cfg):
    """The PBS job name of the project's sequencer jobs, `pl_<tag>` (D-098;
    the tag is the configuration's `dataset`)."""
    return job_name("pl_" + cfg.data["dataset"])


def qsub_line(cfg, name, queue, ncpus, mem, walltime, log_path, argv_file, workdir,
              env_name=None, avoid_file=None, role="stage"):
    """The complete qsub command line of one run_argv.pbs job, in the
    environment `env_name` (default the configuration's envs.train: the
    trials, the probes and the members; the bank's verification job passes
    envs.extract). With `avoid_file` (the trials and the members, D-102)
    the job reads the avoid list at its start and, on a listed host, holds
    its slot AVOID_HOLD_S seconds and exits AVOID_EXIT. Every job carries
    the helpers' folder, the work root, the chain's name and its role
    (`driver` for the sequencer's jobs, `stage` otherwise), D-111: the
    activation's fallback, the driver's replacement, the watchdog."""
    p = paths_of(cfg)
    skip = "1" if E.skip_conda() else "0"
    work_root = cfg.data["paths"]["work_root"]
    if role not in ("stage", "driver"):
        raise ValueError("role %r: stage or driver" % (role,))
    variables = "ARGV_FILE=%s,ENV_NAME=%s,WORKDIR=%s,SKIP_CONDA=%s" % (
        argv_file, env_name or cfg.data["envs"]["train"], workdir, skip)
    variables += ",JOBS_DIR=%s,WORK_ROOT=%s,CHAIN_NAME=%s,ROLE=%s" % (
        p["jobs_dir"], work_root, chain_name(cfg), role)
    if avoid_file:
        variables += ",AVOID_FILE=%s,AVOID_HOLD_S=%d" % (avoid_file, int(AVOID_HOLD_S))
    for v in (argv_file, workdir, avoid_file or "", p["jobs_dir"], work_root):
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


def try_read_json(path):
    """(object, None) when `path` holds a JSON object, else (None, problem);
    never raises. `problem` names the error and the file's size, `<Exc>:
    <message> (<n> bytes[, all NUL])`, or `not a JSON object (<type>; <n>
    bytes)`; a file that cannot be opened gives `<OSError>: ...` (SPEC
    block 7, an unreadable record, amended 2026-10-19)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)
    try:
        obj = json.loads(raw.decode("ascii", "replace"))
    except ValueError as exc:
        nul = ", all NUL" if raw and raw.count(b"\0") == len(raw) else ""
        return None, "%s: %s (%d bytes%s)" % (type(exc).__name__, exc, len(raw), nul)
    if not isinstance(obj, dict):
        return None, "not a JSON object (%s; %d bytes)" % (type(obj).__name__, len(raw))
    return obj, None


def write_json(path, obj):
    """Atomic and durable: a temporary file, flushed and fsynced, then
    renamed -- a node that goes down after the rename leaves the file with
    its bytes (amended 2026-10-19)."""
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    tmp = path + ".tmp"
    blob = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True)
    with open(tmp, "w") as fh:
        fh.write(blob + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def set_aside(path, suffix):
    """Move `path` to `<path>.<suffix><k>`, the first k >= 1 not taken (a
    move aside is not a deletion, D-095; an earlier one is never
    overwritten); return the new path."""
    k = 1
    while os.path.lexists("%s.%s%d" % (path, suffix, k)):
        k += 1
    aside = "%s.%s%d" % (path, suffix, k)
    os.replace(path, aside)
    return aside


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

# A record that exists but cannot be read is read again this long from the
# first failed read, then set aside (SPEC block 7, an unreadable record,
# amended 2026-10-19); the tests shorten it through the environment.
RECORD_GRACE_S = float(os.environ.get("PIPELINE_RECORD_GRACE_S", "300"))


def unreadable_record(holder, path, problem, log, who, what="record", grace=True,
                      host=None, then="the job is taken as one that ended without a record"):
    """A record that exists and cannot be read (empty, zeros, cut short:
    what a node that goes down after the rename leaves). Returns True while
    it is read again -- the first failed read is logged and its time kept
    in `holder["record_unreadable_since"]` (`holder` is the trial's or the
    member's state) -- and False once it has been moved to
    `<path>.unreadable<k>` (kept, out of every record glob; listed in
    `holder["unreadable_records"]`): the caller then goes on by `then`.
    `grace` False (no job of the stage runs: a fresh state, a stage's
    start): moved aside at once. A move that fails is logged and tried
    again at the next poll (True), never raised."""
    now = time.time()
    if grace:
        since = holder.get("record_unreadable_since")
        if since is None:
            holder["record_unreadable_since"] = now
            log("%s: its %s %s cannot be read (%s); read again for up to %d s"
                % (who, what, path, problem, int(RECORD_GRACE_S)))
            return True
        if now - float(since) < RECORD_GRACE_S:
            return True
    try:
        aside = set_aside(path, "unreadable")
    except OSError as exc:
        log("%s: its %s %s cannot be read (%s) and could not be set aside (%s); tried again "
            "at the next poll" % (who, what, path, problem, exc))
        holder.setdefault("record_unreadable_since", now)
        return True
    holder.pop("record_unreadable_since", None)
    holder.setdefault("unreadable_records", []).append(
        {"record": os.path.basename(aside), "problem": problem, "host": host,
         "utc": utc_now()})
    log("%s: its %s could not be read%s (%s); set aside as %s%s; %s"
        % (who, what, (" for %d s" % int(RECORD_GRACE_S)) if grace else "", problem,
           os.path.basename(aside), (" (last job on %s)" % host) if host else "", then))
    return False

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
_RUN_ARGV_HOST = re.compile(r"^RUN_ARGV(?: AVOID| ENV| ENV FAULT)?: host ([^,:\s]+)", re.M)


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
    """The node a run_argv.pbs job ran on, from its `RUN_ARGV: host` line
    (or its `RUN_ARGV AVOID: host` line, a job held off a listed node)."""
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
        dst = set_aside(p, "corrupt")
        log("%s: %s checkpoint %s failed its CRC check (%s); moved aside as %s; it restarts "
            "from epoch 0" % (stage, who, os.path.basename(p), why, os.path.basename(dst)))
        moved.append(dst)
    return moved


# ---------------------------------------------------------------------------
# The bad-node avoid list (D-102; SPEC block 7, amended 2026-10-21)
# ---------------------------------------------------------------------------

# A host with this many node faults in a stage's run (since its base, which a
# clear by hand resets) is written to the avoid list; the environment
# overrides are for the tests. No configuration key: the configuration's
# digest does not change, a running chain needs no --accept-config-change.
AVOID_FAULTS = int(os.environ.get("PIPELINE_AVOID_NODE_FAULTS", "3"))
# Seconds a job holds its slot on a listed host before it exits, so that the
# scheduler does not refill the node with the next queued job at once.
AVOID_HOLD_S = int(os.environ.get("PIPELINE_AVOID_HOLD_S", "600"))
# The exit code of such a job (jobs/run_argv.pbs); reserved: the tuner exits
# 0-3, the runner 0-2, a signal 129-255, the script's own refusals 90-98.
AVOID_EXIT = 94
AVOID_FILE_NAME = "avoid_nodes.txt"
AVOID_HEADER = ("# avoid list of the pipeline (D-102): one host per line, as `hostname` prints\n"
                "# it on the node. A trial or member job that starts on a listed host holds\n"
                "# its slot %d s and exits %d; the pipeline submits it again at once. The\n"
                "# pipeline appends a host after %d node faults in a stage's run and never\n"
                "# removes one: delete a host's line (or this file) to lift its ban.\n")


def avoid_file(cfg):
    """`<work_root>/avoid_nodes.txt`."""
    return os.path.join(cfg.data["paths"]["work_root"], AVOID_FILE_NAME)


def read_avoid_list(path):
    """The listed hosts, in file order: one per line, stripped; blank lines
    and lines starting with `#` ignored; `[]` when the file is absent or
    cannot be read."""
    try:
        with open(path, "rb") as fh:
            lines = fh.read().decode("ascii", "replace").splitlines()
    except OSError:
        return []
    hosts = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#"):
            hosts.append(s)
    return hosts


def add_to_avoid_list(path, host, note):
    """Append `host` (with a comment line `# <note>` above it) unless it is
    listed already; the header is written when the file is new. Returns
    True when the host was written. Never rewrites or removes a line."""
    host = str(host).strip()
    if not host or host == "?" or host.startswith("#"):
        return False
    if host in read_avoid_list(path):
        return False
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    new = not os.path.isfile(path)
    with open(path, "a") as fh:
        if new:
            fh.write(AVOID_HEADER % (AVOID_HOLD_S, AVOID_EXIT, AVOID_FAULTS))
        fh.write("# %s\n%s\n" % (str(note).replace("\n", " "), host))
        fh.flush()
        os.fsync(fh.fileno())
    return True


def note_node_fault(sub, host, why, stage, log, path):
    """Count a node fault on `host` in the stage's run (`sub["node_fault_
    hosts"]`, which the log lines print) and, at AVOID_FAULTS faults since
    the host's base, list it: returns (faults on the host in all, listed
    now). A host already listed (by the other stage, or by hand) is noted
    as the stage's, so that a clear by hand is seen; `?` (no host read) is
    counted and never listed."""
    hosts = sub.setdefault("node_fault_hosts", {})
    hosts[host] = hosts.get(host, 0) + 1
    n = hosts[host]
    if host == "?":
        return n, False
    base = sub.setdefault("avoid_base", {})
    avoided = sub.setdefault("avoided", {})
    since = n - int(base.get(host, 0))
    if since < AVOID_FAULTS or host in avoided:
        return n, False
    note = "%s %s: %d node fault(s) in this run, the last: %s" % (utc_now(), stage, n,
                                                                  str(why)[:120])
    written = add_to_avoid_list(path, host, note)
    avoided[host] = {"faults": n, "utc": utc_now(), "written": written}
    if written:
        log("%s: host %s has %d node fault(s) in this run: added to the avoid list %s (D-102); a "
            "job that lands there holds its slot %d s and is submitted again; delete its line from "
            "that file to lift the ban" % (stage, host, n, path, AVOID_HOLD_S))
    else:
        log("%s: host %s has %d node fault(s) in this run; it is on the avoid list %s already"
            % (stage, host, n, path))
    return n, True


def sync_avoid_list(sub, path, log, stage):
    """Every poll: a host the stage listed that is no longer in the file was
    cleared by hand -- its base is set to its count (the next AVOID_FAULTS
    faults list it again) and the clear is logged once. Returns the hosts
    cleared."""
    avoided = sub.get("avoided") or {}
    if not avoided:
        return []
    listed = set(read_avoid_list(path))
    cleared = []
    for host in sorted(avoided):
        if host in listed:
            continue
        sub.setdefault("avoid_base", {})[host] = int(sub.get("node_fault_hosts", {}).get(host, 0))
        del sub["avoided"][host]
        cleared.append(host)
        log("%s: host %s is no longer on the avoid list %s (cleared by hand); its node faults are "
            "counted again from now" % (stage, host, path))
    return cleared


def held_off_node(job):
    """True when a run_argv.pbs job (a sequencer job record) ended by the
    avoid list: exit status AVOID_EXIT from the scheduler (its log, when it
    has landed, says `RUN_ARGV AVOID:` and gives the host)."""
    try:
        return int(job.get("exit_status")) == AVOID_EXIT
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Environment faults (D-111; SPEC block 7, amended 2026-10-29)
# ---------------------------------------------------------------------------

# run_argv.pbs's exit when the job's environment cannot start on its node
# (jobs/activate_env.sh: neither conda nor the environment's own folder).
ENV_FAULT_EXIT = 96
# The job script before D-111: the conda hook (97) or activation (98) failed.
_OLD_CONDA_EXITS = (97, 98)
# PBS Pro's Exit_status for a job its node could not start or keep (exec
# failures, MoM aborts, a bad uid or gid, a rerun, a sister or the MoM lost,
# a hook's requeue) [proposed: PBS Pro's documented codes; davinci's use of
# them is known for -2 only, 2026-10-16]. The limits' kills (-24..-29) and a
# hook's deletion (-17, -19) are not here.
JOB_EXEC_FAULTS = frozenset((-1, -2, -3, -4, -5, -6, -7, -10, -11, -14, -16, -18, -20))
# The guard: this many environment faults in a row, on at least this many
# hosts, stop the stage (a broken environment would otherwise loop for ever,
# the faults being free). Environment overrides for the tests.
ENV_FAULT_STREAK = int(os.environ.get("PIPELINE_ENV_FAULT_STREAK", "12"))
ENV_FAULT_HOSTS = int(os.environ.get("PIPELINE_ENV_FAULT_HOSTS", "3"))
_ENV_FAULT_LINE = re.compile(r"^RUN_ARGV ENV FAULT: (.*)$", re.M)


def env_fault(job, log_text=None):
    """A short reason when a job that ended without a record could not start
    its command for its node's sake (D-111) -- its environment (exit
    ENV_FAULT_EXIT, or 97 / 98 of the job script before D-111, or the
    `RUN_ARGV ENV FAULT:` line of its log) or PBS itself (Exit_status in
    JOB_EXEC_FAULTS) -- else None. `job` is a sequencer job record (its
    `exit_status` from the scheduler); `log_text` the job's log, if any."""
    code = None
    try:
        code = int(job.get("exit_status"))
    except (TypeError, ValueError):
        code = None
    m = _ENV_FAULT_LINE.search(log_text or "")
    line = m.group(1).strip()[:200] if m else None
    if code == ENV_FAULT_EXIT:
        return line or "exit %d: the environment could not start (no log)" % ENV_FAULT_EXIT
    if code in _OLD_CONDA_EXITS:
        return "exit %d: the conda %s failed (a job script before D-111)" % (
            code, "hook" if code == 97 else "activation")
    if code in JOB_EXEC_FAULTS:
        return "PBS could not start or keep the job on its node (Exit_status %d)" % code
    if code is None and line:
        return line
    return None


def note_env_fault(sub, host, why, stage, log, path):
    """Count an environment fault: a node fault on `host` (note_node_fault,
    D-102: listed at AVOID_FAULTS) and one more in the stage's streak.
    Returns (faults on the host, the stop reason or None): the reason when
    the streak reaches ENV_FAULT_STREAK faults on at least ENV_FAULT_HOSTS
    hosts -- the environment or the cluster, not a node."""
    n, _ = note_node_fault(sub, host, why, stage, log, path)
    streak = sub.setdefault("env_fault_streak", {"n": 0, "hosts": []})
    streak["n"] = int(streak.get("n", 0)) + 1
    hosts = list(streak.get("hosts", []))
    if host not in hosts:
        hosts.append(host)
    streak["hosts"] = hosts
    streak["last"] = str(why)[:200]
    real = [h for h in hosts if h != "?"]
    if streak["n"] >= ENV_FAULT_STREAK and len(set(hosts)) >= ENV_FAULT_HOSTS:
        listed = sorted(h for h in (sub.get("avoided") or {}) if h in real)
        return n, ("%d job(s) in a row could not start (environment faults on %d node(s): %s; "
                   "the last: %s): the environment or the cluster, not a node -- fix it, delete "
                   "from %s the hosts this run listed (%s), then --retry %s"
                   % (streak["n"], len(set(hosts)), ", ".join(hosts), streak["last"], path,
                      ", ".join(listed) or "none", stage))
    return n, None


def clear_env_fault_streak(sub):
    """A job of the stage ran (a record, a kill, a refusal): the streak of
    environment faults in a row starts again (D-111)."""
    if sub.get("env_fault_streak", {}).get("n"):
        sub["env_fault_streak"] = {"n": 0, "hosts": []}


# ---------------------------------------------------------------------------
# The jobs of Sbi-extractor's launchers (D-113; SPEC blocks 2 and 3, amended
# 2026-10-29 (b))
# ---------------------------------------------------------------------------

# The variables a stage gives a launcher so that every job it submits starts
# through jobs/stage_job.sh (the avoid list, the activation's folder fallback,
# exit 94 / 96, the chain's watchdog); env.sh's sbix_pipeline_v passes them on
# with qsub -v. Popped first: a value inherited from the caller never leaks.
PROTOCOL_VARS = ("PIPELINE_JOBS_DIR", "WORK_ROOT", "CHAIN_NAME", "AVOID_FILE", "AVOID_HOLD_S")
# Free rounds (a round whose bad tasks all never started) in a row that leave
# at least as many tasks unstarted as the round before stop the stage.
FREE_ROUND_STALL = int(os.environ.get("PIPELINE_FREE_ROUND_STALL", "3"))


def stage_job_protocol(cfg):
    """{name: value} of PROTOCOL_VARS for a launcher's environment: this
    checkout's jobs/ folder, the work root, the chain's name, the avoid list
    and its hold. Refuses a value qsub -v cannot carry (a comma or a blank)."""
    v = {"PIPELINE_JOBS_DIR": paths_of(cfg)["jobs_dir"],
         "WORK_ROOT": cfg.data["paths"]["work_root"],
         "CHAIN_NAME": chain_name(cfg), "AVOID_FILE": avoid_file(cfg),
         "AVOID_HOLD_S": str(int(AVOID_HOLD_S))}
    for k in PROTOCOL_VARS:
        if "," in v[k] or any(c.isspace() for c in v[k]):
            raise ValueError("%s=%r: qsub -v cannot carry a comma or a blank (D-113)" % (k, v[k]))
    return v


def free_round(sub, n_unstarted, stage, what="task(s)"):
    """D-113: a round whose bad tasks all never started (their jobs could not
    start: an environment fault, a hold off a listed node, PBS) is free, not
    counted against the stage's rerun rounds. Counts it in the stage's state
    and returns the stop reason, or None: FREE_ROUND_STALL free rounds in a
    row that leave at least as many unstarted as the round before."""
    prev = sub.get("free_last")
    stall = int(sub.get("free_stall", 0))
    stall = stall + 1 if (prev is not None and int(n_unstarted) >= int(prev)) else 0
    sub["free_stall"] = stall
    sub["free_last"] = int(n_unstarted)
    sub["free_rounds"] = int(sub.get("free_rounds", 0)) + 1
    if stall >= FREE_ROUND_STALL:
        return ("%d %s still could not start after %d free rounds in a row, no fewer each round "
                "(D-113): the environment or the cluster, not a node -- read the jobs' logs (their "
                "RUN_ARGV lines), fix it, then --retry %s" % (int(n_unstarted), what, stall, stage))
    return None


def unrun_dependent(job, log_fresh):
    """D-113: True for a finished job that waited on another (`after`) and
    that PBS ended with one of JOB_EXEC_FAULTS before it wrote a log of its
    own -- its dependency failed (afterok on a plain job: davinci deletes the
    dependent) or PBS never started it. Never a node's fault: the caller's
    'never ran' path applies, and nothing is counted against a host."""
    try:
        code = int(job.get("exit_status"))
    except (TypeError, ValueError):
        return False
    return bool(job.get("after")) and not log_fresh and code in JOB_EXEC_FAULTS


def counted_round(sub):
    """A round counted against the rerun rounds (a task started and failed):
    the free rounds' stall starts again (D-113)."""
    sub["free_stall"] = 0
    sub.pop("free_last", None)
