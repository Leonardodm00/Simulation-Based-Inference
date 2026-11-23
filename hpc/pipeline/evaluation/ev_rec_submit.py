#!/usr/bin/env python3
"""Block 38: the launcher of a recorded project's evaluation on davinci's
PBS -- one sampling job per complete ensemble, the report after them, and
for a synthetic project read as recorded the validation after the report
(specs/SPEC.md sec. 2.6 and block 38; decisions D-159, D-162).

    bash hpc/pipeline/evaluation/submit_rec_eval.sh --project <tag> [--dry-run] \\
        [--what sample,report,validate] [--resample <name>,...|all]

(the wrapper runs `python3 hpc/pipeline/evaluation/ev_rec_submit.py
--config <tag>.json --eval-config <tag>_rec_eval.json ...` in
envs.extract). Each unit is done, live (its job in <out_root>/jobs/
ledger.json queued, held or running) or submitted; the rule is block 22's
(ev_submit.py), for these units:

    sample:<name>  one complete ensemble of the view (ev_rec_sample.py)
    report         ev_rec_report.py, after the samplings
    validate       ev_rec_validate.py --old <validate_against> --new
                   <out_root> --record <out_root>/VALIDATION.json, after the
                   report; only for an as-recorded view with validate_against

Every job is jobs/eval_run.pbs (block 22's, unchanged) in envs.train.
--dry-run prints the plan and every qsub line and writes nothing.

Exit 0 (REC EVAL SUBMIT OK / REC EVAL DRY RUN), 2 on the arguments or the
configurations, 3 on a refusal (nothing submitted), 4 when a qsub fails.

CONTRACT (wave C): the constants and the signatures below are fixed by
specs/SPEC.md block 38; the bodies are the implementer's.

Standard library and the pipeline's own modules only (the login node's
python). Pure ASCII, LF only.
"""

import sys

EXIT_ARGS, EXIT_REFUSE, EXIT_QSUB = 2, 3, 4
KINDS = ("sample", "report", "validate")
STAGE_OF = {"sample": "I35", "report": "I37", "validate": "I36"}   # specs/roadmap/tool.yaml
REPORT_COMPLETE = "REC_REPORT_COMPLETE.json"      # block 37's (ev_rec_report.REPORT_COMPLETE)
VALIDATION_RECORD = "VALIDATION.json"             # block 36's ev_rec_validate.py --record
JOB_NAMES = {"report": "er_report", "validate": "er_valid"}        # a sampling: er_s<i>


class Refuse(RuntimeError):
    """The launcher cannot go on; nothing is submitted (exit 3)."""


def default_what(view, ecfg):
    """The units of a run without --what: ("sample", "report"), and
    "validate" as well for an as-recorded view whose evaluation has
    `validate_against`."""
    raise NotImplementedError("block 38")


def plan(cfg_path, ecfg_path, what=None, resample=(), log=print):
    """(view, ecfg, units): every unit as dict(unit, kind, name, state, why,
    argv, live) -- nothing submitted, nothing written. `what` None is
    default_what. Refuse (or ev_plan.RefuseError) when the launcher cannot
    go on: an unknown --resample name, `validate` named for a view that
    has none, a program or the job script missing from the checkout, the
    scheduler not answering for a recorded job."""
    raise NotImplementedError("block 38")


def run(cfg_path, ecfg_path, what=None, resample=(), dry_run=False, log=print):
    """Submits what the plan says; returns the exit code."""
    raise NotImplementedError("block 38")


def main(argv=None):
    """The command line; returns the exit code."""
    raise NotImplementedError("block 38")


if __name__ == "__main__":
    sys.exit(main())
