# hpc/pipeline -- one command per data set

From the simulated and the recorded spike trains to the hyper-parameter
search and the best fit, never pausing, resumable (D-081). Every value of
the search and the training comes from the project's configuration file
(D-082); the posteriors and the report are a separate command (D-083). The
code is data-set-agnostic: a project's paths, manifests, digests and
resources live in its folder `projects/<tag>/` (D-091). The specification is
`specs/SPEC.md` at the repository's root.

## The one command

```
bash hpc/pipeline/run_pipeline.sh --project giulia --dry-run    # 1. read it: the stage table, every qsub line, every REFUSE line
bash hpc/pipeline/run_pipeline.sh --project giulia              # 2. the run: starts the sequencer under nohup and returns
bash hpc/pipeline/run_pipeline.sh --project giulia --status     # 3. any time: the stages, their jobs, the last events
```

`--project <tag>` names `projects/<tag>/<tag>.json`; `--config <path>` names
any configuration. The sequencer runs on the login node under the
configuration's `envs.extract` conda environment and needs only Python's
standard library. Further options go through to `pipeline.py run`:
`--retry <stage>` (a stage recorded as failed is reset to pending),
`--accept-config-change` (continue a run under an edited configuration),
`--break-lock` (a lock left by a sequencer on another host), `--poll-seconds
N` (default 60).

Pass lines, as lines that must appear: the dry run ends WITHOUT a `REFUSE:`
line and exits 0; the start prints `started : pid <n>`; `--status` shows
every stage `done` or `skipped` at the end; the log's last line is
`PIPELINE DONE: every stage done or skipped`. A stop is
`STOP: stage <name> failed: <reason>` (exit 3): fix the cause, then re-run
the same command with `--retry <name>`.

## The stages

| stage | artefact of record (done when it exists; `+sha256` = with a matching `.sha256` sidecar) | built in |
|---|---|---|
| `real_extraction` (S1) | `<real.extract_root>/cohort_manifest.json` +sha256 | block 3 |
| `sim_redetection` (S2, after S1) | `<sim.out_root>/REEXTRACTION_RECORD.json` +sha256 | block 3 |
| `label_freeze` (S3) | `<paths.sbix_dir>/<freeze.out>` +sha256 | block 3 |
| `bank_plan` (S4, after S1-S3) | `<paths.sbix_dir>/export_windows/<tag>/bank_plan.json` +sha256 | block 2 |
| `sim_bank` (S5, after S4) | `<paths.banks_root>/<tag>_sim/BANK_RECORD.json` +sha256 | block 2 |
| `real_bank` (S6, after S4) | `<paths.banks_root>/<tag>_real/shard_0000.npz` + `.json` | block 2 |
| `search` (S7, after S5, S6) | `<paths.tune_root>/<search.campaign>/SEARCH_COMPLETE.json` | block 7 |
| `best_fit` (S8, after S7) | `<paths.runs_root>/<tag>_best/BEST_FIT_COMPLETE.json` | block 7 |

A stage whose artefact exists when the run starts is skipped, so the command
starts wherever the data set stands. A stage whose runner is not built yet
is reported as `REFUSE: stage <name> is not built yet (block N)`, and a
configuration key that no built block honours as `REFUSE: this code cannot
honour <key> (needs block N: ...)`; both stop the run before anything is
submitted (exit 2).

## The run's state

`<paths.work_root>/state.json` records every stage's status (`pending`,
`running`, `done`, `failed`, `skipped`), its submitted jobs (id, role, the
qsub line, the log path, the exit status once finished), its resubmissions
and its notes; `pipeline.log` is the sequencer's log; `pipeline.lock` holds
the pid and host of the running sequencer (a second start on the same work
root exits 4); `pipeline.nohup` catches anything the sequencer prints
outside its log (a traceback). A re-run reads the state file, re-attaches
to the recorded jobs through `qstat -x -f`, and goes on. One known window:
a sequencer killed between `qsub` returning and the state file's write
loses that job id and would submit it again on resume; the runners refuse
to overwrite their artefacts, so the duplicate stops with a refusal rather
than corrupting anything.

## The project folder

`projects/<tag>/<tag>.json` -- every key required, documented in the file by
the `_<key>` string beside it; the schema is `specs/SPEC.md` sec. 3.1 and
`config.py`. The Giulia file is config v4 (2026-10-15): v3's values with the
search's wall budget at 400 h (D-090).

## Layout

```
pipeline.py        the sequencer: run (--dry-run), status, paths
config.py          the schema, the loader, the NEEDS-CODE capability table
state.py           the state file
pbs.py             qsub / qstat -x -f, the only scheduler calls
stages/__init__.py the Stage interface, the graph, the registry
run_pipeline.sh    the one command (nohup, conda, --dry-run, --status)
projects/<tag>/    the project's configuration
tests/             the smoke test, the fake PBS, the synthetic graph
```

## Tests

```
python3 hpc/pipeline/tests/test_smoke_pipeline.py
```

Block 1's smoke test under the fake PBS; blocks 4-6 (the runner) are tested by
`hpc/joint/stage3/smoke_test_blocks_4_6.py` (about 3 minutes; needs torch and sbi).
Block 1's test (`tests/fake_pbs/`): the schema's
refusals, the state file, the qstat parser against a recorded davinci
record, and the sequencer on a synthetic three-stage graph -- the dry run,
a full run, a kill-and-resume with exactly one submission, a refusing
verifier, a silent kill resubmitted once, the lock, the configuration-change
guard, and the grep that keeps project values out of the code. About 25 s.

## Build status

| block | what | status |
|---|---|---|
| 1 | the sequencer, the schema, the state, the scheduler adapter, the one command, config v4 | built 2026-10-15, smoke-tested |
| 2 | the bank stages S4-S6 | planned |
| 3 | the input stages S1-S3 | planned |
| 4 | the campaign prior in the runner (`run_joint_arms.py`: a `latent_spec.kind` `campaign` bank uses the unit box, L0 = 0) | built 2026-10-15, smoke-tested |
| 5 | the streaming loader (D-084): `--buffer-shards K` streams the bank through a rolling buffer of K shards with the quota rule; K >= the shard count reproduces the in-memory run bit for bit | built 2026-10-15, smoke-tested |
| 6 | runner flags `--passes`, `--patience`, `--split-fracs`, `--split-seed`, `--head-pool-ops`, `--flow-hidden-layers`, `--threads`; a per-epoch checkpoint, `--resume`, `--stop-after-epoch` | built 2026-10-15, smoke-tested |
| 7 | the search and the best fit S7-S8 | planned |
| 8 | end to end on the stand-in; delivery | planned |
