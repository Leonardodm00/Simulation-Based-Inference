"""Session fixtures of the chain run t2-C2: the reduced synthetic chain run
through its real entry point under the fake PBS.

C2_ROOT=<dir> keeps the runs there and reuses a finished one (each run's
folder holds its own configuration, spool and outputs); without it the runs
go under pytest's temporary folder.
"""

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_helpers as H  # noqa: E402


def _root(tmp_path_factory, name):
    base = os.environ.get("C2_ROOT")
    if base:
        return os.path.join(os.path.abspath(base), name)
    return str(tmp_path_factory.mktemp(name))


def _finished(root):
    cfg_path = os.path.join(root, "project", "synth.json")
    if not os.path.isfile(cfg_path):
        return None
    import json
    with open(cfg_path) as fh:
        cfg = H._strip(json.load(fh))
    ce = H.chain_end(cfg)
    if ce is None or H.live_jobs(os.path.join(root, "spool")):
        return None
    return cfg_path, cfg, os.path.join(root, "spool"), H.fake_env(os.path.join(root, "spool")), \
        "(reused)", ce


def _run(tmp_path_factory, name, **over):
    root = _root(tmp_path_factory, name)
    got = _finished(root)
    if got is not None:
        return got
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    return H.full_run(root, **over)


@pytest.fixture(scope="session")
def run1(tmp_path_factory):
    """The reduced chain, uninterrupted: (cfg_path, cfg, spool, env, submit
    output, CHAIN_END)."""
    return _run(tmp_path_factory, "run1")


@pytest.fixture(scope="session")
def run2(tmp_path_factory):
    """The same configuration again, in another root (C-O13)."""
    return _run(tmp_path_factory, "run2")
