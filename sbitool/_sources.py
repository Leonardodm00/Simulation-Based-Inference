"""Block 40: the existing code the tool imports and never copies, until the
release packages it (specs/SPEC.md sec. 2.7, "The sources the package
imports"; D-176).

CONTRACT (wave A): GROUPS and the signatures are fixed; the bodies are the
implementer's. Standard library only.

Pure ASCII, LF only.
"""

import os

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


def extractor_dir():
    """The Sbi-extractor's checkout: the first of EXTRACTOR_ENV that is set
    and is a directory; SourceError naming the variables otherwise."""
    raise NotImplementedError("block 40")


def use(*names):
    """Puts the folders of the named groups on sys.path (once, in GROUPS'
    order) and returns {relative file: {"path": absolute, "sha256": hex}}
    for the groups' recorded files. SourceError for an unknown group or an
    absent folder or file."""
    raise NotImplementedError("block 40")
