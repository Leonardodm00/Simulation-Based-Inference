"""Block 40: the existing code the tool imports and never copies, until the
release packages it (specs/SPEC.md sec. 2.7, "The sources the package
imports"; D-176).

CONTRACT (wave A): GROUPS and the signatures are fixed; the bodies are the
implementer's. Standard library only.

Pure ASCII, LF only.
"""

import hashlib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXTRACTOR_ENV = ("SBIX_DIR", "SBIX_CLONE")     # the Sbi-extractor's checkout, first one set

# group -> (root, the folders put on sys.path, the files whose sha256 is recorded);
# paths relative to the root: REPO for the first three, the extractor's checkout for the last
GROUPS = {
    "model": ("repo", ("hpc/joint/stage2", "hpc/joint/stage3", "hpc/dsn"),
              ("hpc/joint/stage2/joint_model.py", "hpc/joint/stage3/run_joint_arms.py",
               "hpc/dsn/backbone.py", "hpc/dsn/condition_space.py")),
    "evaluation": ("repo", ("hpc/pipeline/evaluation",),
                   ("hpc/pipeline/evaluation/ev_sample.py", "hpc/pipeline/evaluation/ev_gate.py",
                    "hpc/pipeline/evaluation/ev_rec_metrics.py",
                    "hpc/pipeline/evaluation/ev_models.py")),
    "ifr": ("repo", ("hpc/dsn",), ("hpc/dsn/generate_burst_data.py",)),
    "extractor": ("extractor", ("extractor", "export_windows", "."),
                  ("extractor/channel_subset_extraction.py", "export_windows/bank_real.py")),
}


class SourceError(RuntimeError):
    """A source group's folder or file is absent."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extractor_dir():
    """The Sbi-extractor's checkout: the first of EXTRACTOR_ENV that is set
    and is a directory; SourceError naming the variables otherwise."""
    seen = []
    for var in EXTRACTOR_ENV:
        val = os.environ.get(var)
        if val:
            if os.path.isdir(val):
                return os.path.abspath(val)
            seen.append("%s=%s is not a directory" % (var, val))
    raise SourceError("the Sbi-extractor's checkout is not found: set %s to its folder "
                      "(the tests read %s)%s"
                      % (EXTRACTOR_ENV[0], " or ".join(EXTRACTOR_ENV),
                         "" if not seen else "; %s" % "; ".join(seen)))


def use(*names):
    """Puts the folders of the named groups on sys.path (once, in GROUPS'
    order) and returns {relative file: {"path": absolute, "sha256": hex}}
    for the groups' recorded files. SourceError for an unknown group or an
    absent folder or file."""
    unknown = [n for n in names if n not in GROUPS]
    if unknown:
        raise SourceError("unknown source group(s) %s; the groups are %s"
                          % (", ".join(repr(n) for n in unknown), ", ".join(GROUPS)))
    folders, files = [], {}
    for name in GROUPS:                 # GROUPS' order, whatever the arguments' order
        if name not in names:
            continue
        root_kind, dirs, recorded = GROUPS[name]
        root = REPO if root_kind == "repo" else extractor_dir()
        for d in dirs:
            path = os.path.normpath(os.path.join(root, d))
            if not os.path.isdir(path):
                raise SourceError("source group %r: the folder %s is absent" % (name, path))
            if path not in folders:
                folders.append(path)
        for rel in recorded:
            path = os.path.normpath(os.path.join(root, rel))
            if not os.path.isfile(path):
                raise SourceError("source group %r: the file %s is absent" % (name, path))
            files[rel] = {"path": path, "sha256": _sha256(path)}
    # in front of sys.path, the first group's first folder first, so that the
    # sources and not a same-named module elsewhere are imported; a folder
    # already on sys.path stays where it is
    new = [f for f in folders if f not in sys.path]
    sys.path[0:0] = new
    return files
