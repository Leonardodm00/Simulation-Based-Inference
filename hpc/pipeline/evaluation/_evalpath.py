"""sys.path for the evaluation's modules (block 19): the pipeline first, so
that `config` is the pipeline's, then the joint stack's stages and `hpc/`
(npe_diagnostics, bootstrap_paired). Imported for its side effect.

Pure ASCII, LF only.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.abspath(os.path.join(HERE, ".."))
HPC = os.path.abspath(os.path.join(PIPE, ".."))
REPO = os.path.abspath(os.path.join(HPC, ".."))
JOINT = os.path.join(HPC, "joint")

for _p in (os.path.join(JOINT, "stage1"), os.path.join(JOINT, "stage2"),
           os.path.join(JOINT, "stage3"), JOINT, HPC, PIPE, HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import config as _pipeline_config  # noqa: E402,F401  (the pipeline's, before anything else)

if not hasattr(_pipeline_config, "SCHEMA_SYNTHETIC"):
    raise ImportError("the evaluation imported %s as `config`, not the pipeline's config.py"
                      % getattr(_pipeline_config, "__file__", "?"))
