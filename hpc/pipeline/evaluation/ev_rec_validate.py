#!/usr/bin/env python3
"""Block 36: the validation of the recorded path against the synthetic
evaluation (specs/SPEC.md sec. 2.6, "The validation", the cluster's
targets, and block 36; decisions D-162, D-163 (4), D-164).

    python3 hpc/pipeline/evaluation/ev_rec_validate.py \\
        --old <out_root of the synthetic evaluation> \\
        --new <out_root of the as-recorded evaluation> \\
        [--nll-tol 0.002] [--coverage-tol 0.01]

Compares, for each ensemble, on D_sim: the NLL of every row, the expected
coverage on the grid, the SBC verdicts and Holm rejections, TARP's two
verdicts. Exit 0 with `REC VALIDATION OK: <n> ensemble(s)`; exit 3 with one
line per failure otherwise; 2 on the arguments.

CONTRACT (wave B): the constants and the signatures below are fixed by
specs/SPEC.md block 36. The functions are stubs.

Pure ASCII, LF only.
"""

EXIT_ARGS, EXIT_FAIL = 2, 3
NLL_TOL = 0.002          # nats per row (D-163 (4))
COVERAGE_TOL = 0.01      # on every point of the grid (D-163 (4))


def validate(old_root, new_root, nll_tol=NLL_TOL, coverage_tol=COVERAGE_TOL):
    """(n_ensembles, failures): `failures` a list of strings, one per
    failure, each naming the ensemble, the quantity, the row or the beta,
    and the two values; empty when the new tree matches the old one. An
    ensemble in one tree only, or rows that differ, is a failure."""
    raise NotImplementedError("block 36")


def main(argv=None):
    """The command line: 0 with the pass line, EXIT_ARGS, EXIT_FAIL."""
    raise NotImplementedError("block 36")


if __name__ == "__main__":
    import sys
    sys.exit(main())
