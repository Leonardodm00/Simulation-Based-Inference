"""The values of a recorded project's evaluation (block 33; specs/SPEC.md sec.
2.6 and block 33; decisions D-159..D-164): `<tag>_rec_eval.json`.

    ecfg = ev_rec_config.load(".../<tag>_rec_eval.json")
    ecfg.sub("samples")          # a folder under out_root
    ecfg.sample_params()         # what a sampling depends on
    ecfg.report_params()         # what a report depends on beyond its samples

Standard library only: the launcher (block 38) imports this module on the
login node's python. Keys of the file that start with `_` are comments.

CONTRACT (wave A): the constants below are fixed by specs/SPEC.md block 33
and are shared by blocks 33, 35, 36 and 38. The functions are stubs.

Pure ASCII, LF only; Python 3.6 syntax.
"""

import json  # noqa: F401
import os
from collections import namedtuple

FORMATS = ("png", "pdf", "svg")
JOB_KINDS = ("sample", "report", "validate")

# the file's keys, in order (block 33, "The configuration")
KEYS = (
    "out_root",                  # absolute: samples/, metrics/, figures/, jobs/ under it
    "seed",                      # int >= 0: the base of every seed of the evaluation
    "n_post",                    # int > 0: posterior draws per row, a multiple of M (D-163: 1000)
    "batch_rows",                # int > 0: windows per batch through a member
    "oracle_tol",                # nats, in (0, 0.1]: the rebuild oracle's tolerance (2e-3)
    "as_recorded",               # bool: a synthetic project read as recorded (D-162)
    "sim_draw_rows_per_donor",   # null (every row) or int >= 1: k_draw (D-163 (1): 1)
    "edge_eps",                  # in (0, 0.5): the edge width eps of (R1) (D-163 (4): 0.006693)
    "gate",                      # dict with exactly GATE_KEYS
    "bootstrap_n",               # int > 0: resamples of every bootstrap
    "level",                     # in (0, 1): the bands' and intervals' level
    "dpi",                       # int > 0: the figures'
    "formats",                   # non-empty list of distinct FORMATS
    "ncpus",                     # int > 0: worker processes when a command is run by hand
    "validate_against",          # null, or the absolute out_root of the synthetic evaluation
                                 # the validation compares with; not null only with as_recorded
    "jobs",                      # queue, and ncpus/mem/walltime for each of JOB_KINDS
)

# the keys of `gate` (sec. 2.6, "The gate"; D-161, D-163 (4), D-164)
GATE_KEYS = (
    "donor_fracs",               # four positive numbers summing to 1 within 1e-9:
                                 # (f_ref, f_rank, f_null, f_check) (D-164: 0.4, 0.25, 0.25, 0.1)
    "n_ref",                     # int >= 2: the reference's cap (D-163 (4): 800)
    "n_rank",                    # int >= 1: the rank set's cap (D-164: 1000)
    "n_null",                    # int >= 1: the null set's cap (D-164: 1000)
    "n_null_min",                # int >= 1: a smaller null set is a refusal (D-164: 50)
    "rows_per_donor",            # int >= 1: rows of a network in the rank, null, check sets (1)
    "factors",                   # non-empty list of positive numbers (0.25, 0.5, 1, 2, 4)
    "alpha",                     # in (0, 1): alpha_gate (D-163 (4): 0.05)
)

# what a sampling depends on: a sample of other values is stale (block 38)
SAMPLE_PARAMS = ("seed", "n_post", "as_recorded", "sim_draw_rows_per_donor")
SAMPLE_GATE_PARAMS = ("donor_fracs", "n_ref", "n_rank", "n_null", "n_null_min", "rows_per_donor")
# what a report depends on beyond its samples; oracle_tol is in neither list
REPORT_PARAMS = ("seed", "edge_eps", "bootstrap_n", "level", "dpi", "formats")
REPORT_GATE_PARAMS = ("factors", "alpha")


class RecEvalConfigError(ValueError):
    """The configuration is not usable; the message lists every problem."""


class RecEvalConfig(namedtuple("RecEvalConfig", KEYS + ("path",))):
    """The evaluation's values (KEYS; `formats` a tuple, `gate` a dict whose
    `donor_fracs` and `factors` are tuples) and the file they were read from."""

    __slots__ = ()

    def sub(self, name):
        """A folder under out_root."""
        return os.path.join(self.out_root, name)

    def sample_params(self):
        """dict: SAMPLE_PARAMS' values and, under "gate", SAMPLE_GATE_PARAMS'
        (JSON types: lists, not tuples)."""
        raise NotImplementedError("block 33")

    def report_params(self):
        """dict: REPORT_PARAMS' values and, under "gate", REPORT_GATE_PARAMS'
        (JSON types)."""
        raise NotImplementedError("block 33")


def validate(data):
    """The problems of a configuration dict, every one, as strings (empty
    when usable). `data` has no `_` keys."""
    raise NotImplementedError("block 33")


def load(path):
    """The RecEvalConfig of a file (ASCII JSON); RecEvalConfigError listing
    every problem, prefixed with the path."""
    raise NotImplementedError("block 33")
