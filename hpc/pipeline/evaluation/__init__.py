"""The synthetic experiment's evaluation (step S6 of the roadmap synth;
specs/SPEC.md sec. 2.5 and blocks 19-22; decisions D-126, D-127).

A standalone tool, not pipeline stages: it reads what the pipeline's runs
left (the best fits' members, the banks) and writes under the evaluation's
own `out_root`. The sequencer imports nothing from here.

    ev_sample.py    one ensemble -> its posterior draws, densities and
                    embeddings per data set, as .npz (block 19)
    ev_metrics.py   the quantities of sec. 2.5 from the .npz (block 20)
    ev_figures.py   the figures (block 21)
    ev_report.py    the report job: metrics, then figures (block 22)

Every module is named `ev_*`: the DSN tree puts itself on sys.path at run
time and has its own `metrics.py` and `evaluate.py`.

Runs in `envs.train` (torch, sbi). Pure ASCII, LF only.
"""
