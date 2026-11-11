#!/usr/bin/env python3
"""Block 22: the report job -- block 20's metrics, then block 21's figures,
then REPORT_COMPLETE.json (specs/SPEC.md block 22; D-126, D-127).

    python3 hpc/pipeline/evaluation/ev_report.py --config hpc/pipeline/projects/synth/synth.json \\
        --eval-config hpc/pipeline/projects/synth/synth_eval.json [--ncpus 8]

Submitted by the launcher after the samplings (submit_eval.sh); it can also
be run by hand. <out_root>/REPORT_COMPLETE.json records what the report is
made of -- ev_plan.report_inputs, read before anything is computed: the
digest of every sample provenance present and the report's values -- and
the ensembles reported and left out, the figures' count and the times; the
launcher compares those inputs with the current ones to tell a current
report from a stale one.

Pass line: EVAL REPORT OK: ...; exit 2 on the arguments, 3 on a refusal
(no current sample, no metrics).

Pure ASCII, LF only.
"""

import argparse
import datetime
import json
import os
import sys
import time

import _evalpath  # noqa: F401

import ev_config
import ev_figures
import ev_metrics
import ev_plan

EXIT_ARGS, EXIT_REFUSE = 2, 3
REPORT_COMPLETE = "REPORT_COMPLETE.json"


def run(cfg_path, ecfg_path, ncpus=None, log=print):
    t0 = time.time()
    cfg = ev_plan.load_project(cfg_path)
    ecfg = ev_config.load(ecfg_path)
    inputs = ev_plan.report_inputs(cfg, ecfg)        # before anything is read or computed
    idx = ev_metrics.compute(cfg.path, ecfg.path, ncpus=ncpus, log=log)
    t1 = time.time()
    figs = ev_figures.make_all(cfg.path, ecfg.path, log=log)
    rec = {"schema": "eval_report/1", "inputs": inputs, "config": cfg.path,
           "eval_config": ecfg.path, "ensembles": sorted(idx["ensembles"]),
           "left_out": idx["left_out"], "n_figures": len(figs),
           "metrics_index": os.path.join(ecfg.sub("metrics"), "METRICS.json"),
           "figures_index": os.path.join(ecfg.sub("figures"), "FIGURES.md"),
           "seconds": {"metrics": round(t1 - t0, 1), "figures": round(time.time() - t1, 1)},
           "finished": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}
    path = os.path.join(ecfg.out_root, REPORT_COMPLETE)
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)
    log("EVAL REPORT OK: %d ensemble(s), %d figure(s), %d left out (%.0f s); %s"
        % (len(rec["ensembles"]), len(figs), len(idx["left_out"]), time.time() - t0,
           rec["figures_index"]))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the pipeline's project file (<tag>.json)")
    ap.add_argument("--eval-config", required=True, help="the evaluation's file (<tag>_eval.json)")
    ap.add_argument("--ncpus", type=int, default=None, help="default: the eval config's ncpus")
    args = ap.parse_args(argv)
    try:
        run(args.config, args.eval_config, ncpus=args.ncpus, log=lambda t: print(t, flush=True))
    except (ev_plan.RefuseError, ev_config.EvalConfigError) as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
