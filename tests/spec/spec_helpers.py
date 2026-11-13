"""Shared helpers of the tester's spec-derived tests (tests/spec/).

Written by the integration tester (run t1-D of 2026-11-13, blocks 24-26).
Nothing here is the code under test: paths, the fake PBS's environment, tree
digests computed with hashlib/os.walk, and loaders of a module as it stood at
an earlier commit (git show), used as the reference where the spec says
"unchanged".
"""

import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
TOOLS = os.path.join(PIPE, "tools")
PTESTS = os.path.join(PIPE, "tests")
EVAL = os.path.join(PIPE, "evaluation")
FAKE = os.path.join(PTESTS, "fake_pbs")
STAGE1 = os.path.join(REPO, "hpc", "joint", "stage1")
STAGE4 = os.path.join(REPO, "hpc", "joint", "stage4")
SYNTH = os.path.join(PIPE, "projects", "synth", "synth.json")
SYNTH_EVAL = os.path.join(PIPE, "projects", "synth", "synth_eval.json")
PURGE = os.path.join(TOOLS, "purge_outputs.py")
BUILDER = os.path.join(STAGE1, "build_latent_bank.py")
PYTHON = sys.executable

# the commits before the blocks under test (the spec's "unchanged" clauses)
BEFORE_B24 = "4d10214"
BEFORE_B25_26 = "0e62645"


def add_paths(*paths):
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


def fake_env(root, delay="0.2", extra=None):
    """An environment whose qsub/qstat/qselect/qdel are the fake PBS's."""
    env = dict(os.environ)
    env["PATH"] = FAKE + os.pathsep + env.get("PATH", "").replace(FAKE + os.pathsep, "")
    env["FAKE_PBS_SPOOL"] = os.path.join(root, "spool")
    env["FAKE_PBS_DELAY"] = str(delay)
    env["FAKE_PBS_QSELECT_SUFFIX"] = ".cluster"
    env["PIPELINE_SKIP_CONDA"] = "1"
    env["PIPELINE_PYTHON"] = PYTHON
    for k in ("PBS_JOBID", "MUTATION"):
        env.pop(k, None)
    env.update(extra or {})
    return env


def run(argv, env, cwd=None, timeout=600):
    """(exit status, stdout+stderr) of a command."""
    p = subprocess.run(argv, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=timeout)
    return p.returncode, p.stdout


def snapshot(top, skip=()):
    """{relative path: (kind, sha256 or link target, size, mtime_ns)} of every
    entry under `top` but those under `skip` -- the oracle of "nothing
    changed": bytes AND modification times."""
    out = {}
    skip = tuple(os.path.abspath(s).rstrip(os.sep) + os.sep for s in skip)
    for d, subs, files in os.walk(top, followlinks=False):
        if (os.path.abspath(d).rstrip(os.sep) + os.sep).startswith(skip):
            subs[:] = []
            continue
        for name in subs + files:
            q = os.path.join(d, name)
            if (os.path.abspath(q) + os.sep).startswith(skip):
                continue
            st = os.lstat(q)
            if os.path.islink(q):
                out[os.path.relpath(q, top)] = ("link", os.readlink(q), st.st_size, st.st_mtime_ns)
            elif os.path.isdir(q):
                out[os.path.relpath(q, top)] = ("dir", None, 0, None)
            elif os.path.isfile(q):
                with open(q, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                out[os.path.relpath(q, top)] = ("file", h, st.st_size, st.st_mtime_ns)
            else:
                out[os.path.relpath(q, top)] = ("other", None, st.st_size, st.st_mtime_ns)
    return out


def walk_counts(path):
    """(files, bytes) under a folder by os.walk + lstat (links not
    followed), or (1, size) for a file."""
    st = os.lstat(path)
    if not os.path.isdir(path) or os.path.islink(path):
        return 1, st.st_size
    n = b = 0
    for d, _subs, files in os.walk(path, followlinks=False):
        for f in files:
            n += 1
            b += os.lstat(os.path.join(d, f)).st_size
    return n, b


def write(path, data):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "wb") as fh:
        fh.write(data if isinstance(data, bytes) else data.encode("ascii"))


def module_at(commit, relpath, name, extra_paths=()):
    """The module `relpath` as it stood at `commit`, imported under `name`
    from a temporary file (its own imports resolve on sys.path)."""
    src = subprocess.run(["git", "-C", REPO, "show", "%s:%s" % (commit, relpath)],
                         stdout=subprocess.PIPE, check=True).stdout
    d = tempfile.mkdtemp(prefix="tester_old_")
    path = os.path.join(d, name + ".py")
    with open(path, "wb") as fh:
        fh.write(src)
    add_paths(*extra_paths)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


def tree_at(commit, subdir):
    """A temporary folder holding `subdir` of the repository at `commit`."""
    d = tempfile.mkdtemp(prefix="tester_tree_")
    arch = subprocess.run(["git", "-C", REPO, "archive", commit, subdir],
                          stdout=subprocess.PIPE, check=True).stdout
    subprocess.run(["tar", "-x", "-C", d], input=arch, check=True)
    return d
