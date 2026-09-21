"""
dsn_locate.py -- where the Deep Summary Network lives: hpc/dsn, in this repo.

Migration step 2 (2026-09-19). Until step 1 the DSN was a separate repository
and every consumer here reached it the same way:

    d = dsn_main_dir or os.environ.get("DSN_MAIN_DIR")
    sys.path.insert(0, d)

That made the encoder's location a property of the SHELL, and the failure
modes were the shell's: the variable unset, the $HOME/dsn_main symlink
deleted, a checkout that predated condition_space.py (probe_dsn_runtime P3,
run_tests.sh, three smoke suites all grew separate diagnostics for the same
three states). Since step 1 the DSN is a byte-identical mirror at hpc/dsn
(see hpc/dsn/README.md and ORIGIN_MANIFEST.tsv), so its location is a
property of this CHECKOUT and none of those states can occur.

Rules, in order of precedence:

  1. An EXPLICIT path always wins: the `dsn_main_dir=` parameter of the
     loaders and the `--dsn-main-dir` flag of the entry points. That is how a
     smoke test points at a stub and how a deliberate A/B against another
     tree is run. Explicit means typed into a call or a command line.
  2. Otherwise the tree is hpc/dsn, resolved relative to THIS file.
  3. DSN_MAIN_DIR is NOT consulted. If it is set and names a different tree
     than the one resolved, that is reported once, on stdout, and ignored --
     a stale export in a login shell or a job's -v list must not redirect a
     run silently, which is exactly what the old fallback allowed.

Every consumer goes through `dsn_dir()` / `add_dsn_to_path()`; nothing else
in hpc/joint reads DSN_MAIN_DIR any more (grep is the check).

Pure ASCII, LF only.
"""

from __future__ import annotations

import os
import sys

JOINT_DIR = os.path.dirname(os.path.abspath(__file__))       # <hpc>/joint
HPC_DIR = os.path.dirname(JOINT_DIR)                          # <hpc>
DSN_DIR = os.path.join(HPC_DIR, "dsn")                        # <hpc>/dsn

# The modules the joint stack imports from the tree. A directory that lacks
# any of them is not a usable DSN tree, whatever it is called.
SENTINELS = ("backbone.py", "checkpoint.py", "condition_space.py", "train.py",
             "latent_burst_generator.py", "generate_burst_data.py")

_warned_env = False


class DSNTreeMissing(RuntimeError):
    """The resolved DSN tree does not hold the modules the caller needs."""


def _report_ignored_env(resolved):
    """Say once that DSN_MAIN_DIR is set and not used. stdout, like [warn]."""
    global _warned_env
    if _warned_env:
        return
    env = os.environ.get("DSN_MAIN_DIR")
    if not env:
        return
    try:
        same = os.path.realpath(env) == os.path.realpath(resolved)
    except OSError:
        same = False
    if same:
        return
    _warned_env = True
    print("[dsn_locate] DSN_MAIN_DIR=%r is set but IGNORED: since migration "
          "step 2 the DSN is the in-repo tree %r. Unset the variable."
          % (env, resolved), flush=True)


def dsn_dir(explicit=None, require=True):
    """Absolute path of the DSN tree.

    Parameters
    ----------
    explicit : str or None
        A path typed by the caller. Wins over the in-repo tree when given.
    require : bool
        When True (default) raise DSNTreeMissing unless every SENTINEL file
        is present, naming the missing ones. When False return the path
        regardless, for callers that have their own fallback and want to
        decide themselves (make_backbone's GroupNorm CNN).
    """
    d = os.path.abspath(explicit) if explicit else DSN_DIR
    if not explicit:
        _report_ignored_env(d)
    if require:
        missing = [s for s in SENTINELS
                   if not os.path.isfile(os.path.join(d, s))]
        if missing:
            src = ("--dsn-main-dir / dsn_main_dir=%r" % explicit if explicit
                   else "the in-repo tree hpc/dsn")
            raise DSNTreeMissing(
                "DSN tree %r (%s) is missing %s. If this is a fresh clone, "
                "hpc/dsn should hold every file listed in ORIGIN_MANIFEST.tsv; "
                "run `git status` there." % (d, src, ", ".join(missing)))
    return d


def add_dsn_to_path(explicit=None, require=True):
    """dsn_dir(), then put the tree first on sys.path. Returns the path."""
    d = dsn_dir(explicit, require=require)
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def dsn_status(explicit=None):
    """Why the DSN is or is not reachable. Empty string means it is.

    The one function the smoke suites share for their SKIP reasons, so the
    reason a suite prints is the reason the loaders would raise with.
    """
    try:
        d = dsn_dir(explicit, require=True)
    except DSNTreeMissing as exc:
        return str(exc)
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import condition_space  # noqa: F401
    except ImportError as exc:
        return "condition_space present at %r but will not import (%s)" % (
            d, exc)
    return ""


def dsn_available(explicit=None):
    return dsn_status(explicit) == ""


if __name__ == "__main__":
    status = dsn_status()
    print("DSN_DIR  : %s" % DSN_DIR)
    print("resolves : %s" % ("yes" if status == "" else "NO -- " + status))
    sys.exit(0 if status == "" else 1)
