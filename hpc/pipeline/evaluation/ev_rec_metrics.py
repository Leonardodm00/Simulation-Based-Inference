#!/usr/bin/env python3
"""Block 36: the quantities of a recorded project's evaluation, from the
samples of block 35 (specs/SPEC.md sec. 2.6 and block 36; decisions
D-160..D-165).

    python3 hpc/pipeline/evaluation/ev_rec_metrics.py --config <tag>.json \\
        --eval-config <tag>_rec_eval.json [--ensemble <name> ...] [--ncpus N]

Writes under <out_root>/metrics/: per ensemble <stem>_sim.json/.npz and
<stem>_<r>.json/.npz for each recording bank; per comparison
paired_<name>_sim.json/.npz; METRICS.json, the index. Pass line: REC METRICS
OK ...; exit 2 on the arguments, 3 on a refusal.

The quantities of D_sim are block 20's functions (ev_metrics), imported,
never copied. The functions below are the ones no library has.

CONTRACT (wave B): the constants and the signatures below are fixed by
specs/SPEC.md block 36. The functions are stubs.

Pure ASCII, LF only.
"""

import _evalpath  # noqa: F401

EXIT_ARGS, EXIT_REFUSE = 2, 3
SCHEMA = "rec_eval_metrics/1"
INDEX_SCHEMA = "rec_eval_metrics_index/1"


def edge_rows(theta, eps):
    """(n,) bool: (R1). theta (n, p) float in the unit box; row i is an edge
    row iff min_k min(theta_ik, 1 - theta_ik) < eps (strict)."""
    raise NotImplementedError("block 36")


def edge_split(gamma, edge, eps, p, betas=None):
    """(summary, arrays): the edge split of the expected coverage over S_cal.

    gamma (n_cal,) float64: the HPD ranks of the rows of S_cal. edge (n_cal,)
    bool: their edge flags. eps, p: the edge width and the number of axes
    (for the expected share 1 - (1 - 2 eps)^p). betas: the grid (default
    ev_metrics.BETAS).

    summary: n_cal, n_edge, n_int, pi_edge, pi_edge_expected, eps, `edge`
    and `interior` (ev_metrics.coverage_summary of each curve, None for an
    empty set), `r2_max_abs_dev` (the largest absolute difference between
    the two sides of (R2) over the grid; None when a set is empty).
    arrays: coverage_edge, coverage_int (len(betas),) float64, NaN-filled
    for an empty set.
    """
    raise NotImplementedError("block 36")


def well_summary(draws, well):
    """dict: (R4) from the pooled draws of each well's windows. draws (n,
    n_post, p); well (n,) int. Returns well_ids (G,) int64 sorted, well_n
    (G,) int64, and well_mean, well_sd (ddof 1), well_q05, well_q50,
    well_q95, each (G, p) float64."""
    raise NotImplementedError("block 36")


def flag_shares(flag, group):
    """(ids, n, k): for each distinct value of group (n,) int, sorted: the
    number of rows and the number of them with flag (n,) bool true; each
    (G,) int64."""
    raise NotImplementedError("block 36")


def gate_size(alpha, n_null, level):
    """dict: k_gate = ceil(alpha (n_null + 1) - 1e-9) - 1 (R9) and, when
    k_gate >= 1, the size's law (R10) Beta(k_gate, n_null + 1 - k_gate):
    size_mean = k_gate / (n_null + 1), size_lo and size_hi the ends of its
    central interval at `level`. With k_gate = 0 the three are None."""
    raise NotImplementedError("block 36")


def compute(cfg_path, ecfg_path, names=None, ncpus=None, log=print):
    """Every quantity of every ensemble with a current sample (or of those
    named), the comparisons, and METRICS.json; returns the index dict.
    ev_plan.RefuseError when no ensemble has a current sample, or for
    k_gate = 0."""
    raise NotImplementedError("block 36")


def main(argv=None):
    """The command line: 0 with the pass line, EXIT_ARGS, EXIT_REFUSE."""
    raise NotImplementedError("block 36")


if __name__ == "__main__":
    import sys
    sys.exit(main())
