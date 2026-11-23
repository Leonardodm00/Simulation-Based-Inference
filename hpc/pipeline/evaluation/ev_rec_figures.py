#!/usr/bin/env python3
"""Block 37: the figures of a recorded project's evaluation (specs/SPEC.md
sec. 2.6 and block 37; decisions D-159, D-160, D-165).

    python3 hpc/pipeline/evaluation/ev_rec_figures.py --config <tag>.json \\
        --eval-config <tag>_rec_eval.json

Reads <out_root>/metrics/ (block 36: METRICS.json and the files it names)
and, for the corner figures alone, the example wells' draws in
<out_root>/samples/ (block 35); writes <out_root>/figures/<family>/<name>.
<fmt> in the configuration's formats and FIGURES.md, one line per figure
with a one-sentence caption. It draws what the metrics hold and computes
nothing scientific: no mean, SD, quantile or median is taken here, and no
statistics library is imported (R37b).

Pass line: REC FIGURES OK: <n> figure(s); exit 3 when there are no metrics.

CONTRACT (wave C): the constants and the signatures below are fixed by
specs/SPEC.md block 37; the bodies are the implementer's.

Pure ASCII, LF only.
"""

import sys

EXIT_REFUSE = 3
FIGURES_INDEX = "FIGURES.md"

# The names of the figures (block 37, "The figures"); <stem> an ensemble's
# stem, <r> one of its recording banks, <k> a class of that bank, <name> a
# comparison's name.
#   ensembles/<stem>/<f>            f in ENSEMBLE_FIGURES
#   ensembles/<stem>/<f % r>        f in BANK_FIGURES, for each bank r
#   ensembles/<stem>/<CORNER % (r, k)>  for each class k of bank r
#   comparisons/<name>/<f>          f in COMPARISON_FIGURES
#   across/<f>                      f in ACROSS_FIGURES, when there is a comparison
ENSEMBLE_FIGURES = ("cal_coverage", "cal_sbc_ecdf", "cal_sbc_hist", "cal_tarp", "acc_recovery",
                    "acc_zscore", "acc_nll", "acc_contraction", "gate_null")
BANK_FIGURES = ("gate_wells_%s", "gate_windows_%s", "gate_members_%s", "emb_pca_%s",
                "rec_wells_%s")
CORNER_FIGURE = "rec_corner_%s_c%d"
COMPARISON_FIGURES = ("cmp_nll", "cmp_forest")
ACROSS_FIGURES = ("dbar",)


def make_all(cfg_path, ecfg_path, log=print):
    """Draws every figure of the evaluation at `ecfg_path` for the project
    at `cfg_path` from the metrics present; writes them and FIGURES.md.
    Returns the list of (name, caption): `name` the figure's path relative
    to <out_root>/figures/ without its extension, with "/" as separator
    (ensembles/<stem>/cal_coverage), in the order of FIGURES.md; `caption`
    one sentence. ev_plan.RefuseError when <out_root>/metrics/METRICS.json
    is absent or lists no ensemble."""
    raise NotImplementedError("block 37")


def main(argv=None):
    """The command line: 0 with the pass line, 2 on the arguments,
    EXIT_REFUSE on a refusal."""
    raise NotImplementedError("block 37")


if __name__ == "__main__":
    sys.exit(main())
