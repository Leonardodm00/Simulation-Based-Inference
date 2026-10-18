"""What the search, best-fit and bank stages share (blocks 7, 2): where the tuner,
the runner and the tools are, the tuner's common arguments from the
configuration, the qsub line of a one-command job, the argv files, the
JSON helpers and the clock.

Nothing here names a project (D-091): every path and value comes from the
configuration.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import calendar
import json
import os
import re
import time

import envs as E
import pbs as PBS

__all__ = ["paths_of", "tuner_common_args", "shard_globs", "run_tuner", "qsub_line",
           "write_argv_file", "read_argv_file", "read_json", "write_json", "utc_now",
           "utc_to_epoch", "elapsed_hours", "mem_to_gb", "job_name", "ToolError"]


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
