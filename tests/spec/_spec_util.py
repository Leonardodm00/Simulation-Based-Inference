"""Helpers of the tester's spec tests (2026-11-17, incremental run on a915950).

Copies of the committed project files with their paths moved under a
temporary folder, the pipeline's modules on sys.path, the fake PBS of
hpc/pipeline/tests/fake_pbs on PATH for subprocesses. Nothing here is the code
under test; expected values in the tests come from specs/SPEC.md.
"""

import contextlib
import copy
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
TOOLS = os.path.join(PIPE, "tools")
PTESTS = os.path.join(PIPE, "tests")
FAKE_PBS = os.path.join(PTESTS, "fake_pbs")
JOINT = os.path.join(REPO, "hpc", "joint")
PYTHON = sys.executable

GIULIA = os.path.join(PIPE, "projects", "giulia", "giulia.json")
LINDDUP = os.path.join(PIPE, "projects", "linddup", "linddup.json")
SYNTH = os.path.join(PIPE, "projects", "synth", "synth.json")
COHORT_MAP = os.path.join(PIPE, "projects", "linddup", "linddup_cohort_map.csv")
SBIX_CLONE = os.environ.get("SBIX_CLONE", "/home/claude/test-sbix-for-sbi")

for _p in (TOOLS, PIPE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True


def raw_json(path):
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("ascii"))


def strip_docs(obj):
    """The configuration without its `_` documentation keys (sec. 3.1: they
    are ignored), written independently of config._strip_docs."""
    if isinstance(obj, dict):
        return dict((k, strip_docs(v)) for k, v in obj.items() if not k.startswith("_"))
    if isinstance(obj, list):
        return [strip_docs(v) for v in obj]
    return obj


def moved(d, root):
    """A recorded configuration with every path under `root` (the committed
    files name davinci)."""
    d = copy.deepcopy(d)
    for k in list(d["paths"]):
        d["paths"][k] = os.path.join(root, k)
    if "real" in d:
        d["real"]["cohort_config"] = os.path.join(root, "cohort.json")
        d["real"]["extract_root"] = os.path.join(root, "extracted")
        if d["real"].get("cohort_map"):
            d["real"]["cohort_map"] = os.path.join(root, "cohort_map.csv")
    if "sim" in d:
        for k in ("sim_main", "out_root", "tools"):
            d["sim"][k] = os.path.join(root, k)
    return d


def write_json(path, obj):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    return path


def tmpdir(prefix="spec_"):
    return tempfile.mkdtemp(prefix=prefix)


def linddup_copy(root, mutate=None, name="linddup.json"):
    """(path, dict): the committed linddup.json (docs stripped), paths moved."""
    d = moved(strip_docs(raw_json(LINDDUP)), root)
    if mutate is not None:
        mutate(d)
    return write_json(os.path.join(root, name), d), d


def giulia_copy(root, mutate=None, name="giulia.json"):
    d = moved(strip_docs(raw_json(GIULIA)), root)
    if mutate is not None:
        mutate(d)
    return write_json(os.path.join(root, name), d), d


def synth_copy(root, mutate=None, name="synth.json"):
    d = strip_docs(raw_json(SYNTH))
    for k in list(d["paths"]):
        d["paths"][k] = os.path.join(root, k)
    if mutate is not None:
        mutate(d)
    return write_json(os.path.join(root, name), d), d


def env_with_fake_pbs(spool, extra=None):
    env = dict(os.environ)
    env["PATH"] = FAKE_PBS + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = spool
    env["FAKE_PBS_DELAY"] = "0.2"
    env["PIPELINE_SKIP_CONDA"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PBS_JOBID", None)
    if extra:
        env.update(extra)
    return env


def run(argv, env=None, cwd=None, timeout=600):
    """(returncode, stdout+stderr) of a subprocess."""
    r = subprocess.run(argv, cwd=cwd or REPO, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return r.returncode, r.stdout


def tree_snapshot(root):
    """{relative path: (size, mtime_ns)} of every file under root."""
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            st = os.lstat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


@contextlib.contextmanager
def env_vars(**kw):
    saved = dict(os.environ)
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
