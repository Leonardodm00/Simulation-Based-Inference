#!/usr/bin/env python3
"""Block 37: the report job of a recorded project's evaluation -- block
36's metrics, then the figures, then REC_REPORT.md and, last,
REC_REPORT_COMPLETE.json (specs/SPEC.md sec. 2.6 and block 37; decisions
D-159, D-160, D-165).

    python3 hpc/pipeline/evaluation/ev_rec_report.py --config <tag>.json \\
        --eval-config <tag>_rec_eval.json [--ncpus N]

Submitted by the launcher after the samplings (block 38:
submit_rec_eval.sh); it can also be run by hand. REC_REPORT.md states
numbers read from the metrics' JSON files alone and what each is; it draws
no conclusion about a culture, a condition or an arm, writes "the mixture
over the well's windows" and never "the well's posterior" (D-160), and
"flagged by the gate", never "misspecified".

Pass line: REC REPORT OK: ...; exit 2 on the arguments, 3 on a refusal (no
ensemble with a current sample, no metrics).

CONTRACT (wave C): the constants and the signatures below are fixed by
specs/SPEC.md block 37; the bodies are the implementer's.

Pure ASCII, LF only.
"""

import sys

EXIT_ARGS, EXIT_REFUSE = 2, 3
SCHEMA = "rec_eval_report/1"
REPORT_MD = "REC_REPORT.md"
REPORT_COMPLETE = "REC_REPORT_COMPLETE.json"

# the keys of REC_REPORT_COMPLETE.json (block 37, "the keys"); block 38's
# launcher reads `inputs`
COMPLETE_KEYS = (
    "schema", "inputs", "config", "eval_config", "view_kind", "ensembles", "left_out",
    "comparisons", "n_figures", "pass_lines", "metrics_index", "figures_index", "report",
    "seconds", "finished")


def write_report(cfg_path, ecfg_path, log=print):
    """Writes <out_root>/REC_REPORT.md from <out_root>/metrics/METRICS.json
    and the JSON files it names (never from a sample file or an .npz);
    returns its path. ev_plan.RefuseError when there are no metrics."""
    raise NotImplementedError("block 37")


def run(cfg_path, ecfg_path, ncpus=None, log=print):
    """The report job: ev_rec_plan.report_inputs read before anything is
    computed; ev_rec_metrics.compute; ev_rec_figures.make_all;
    write_report; then REC_REPORT_COMPLETE.json (COMPLETE_KEYS), written to
    a temporary name and renamed. Returns that record."""
    raise NotImplementedError("block 37")


def main(argv=None):
    """The command line: 0 with the pass line, EXIT_ARGS, EXIT_REFUSE."""
    raise NotImplementedError("block 37")


if __name__ == "__main__":
    sys.exit(main())
