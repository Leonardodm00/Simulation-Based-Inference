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
| `real_extraction` (S1) | `<real.extract_root>/cohort_manifest.json` +sha256 | block 3, built |
| `sim_redetection` (S2, after S1) | `<sim.out_root>/REEXTRACTION_RECORD.json` +sha256 | block 3, built |
| `label_freeze` (S3) | `<paths.sbix_dir>/<freeze.out>` +sha256 | block 3, built |
| `bank_plan` (S4, after S1-S3) | `<paths.sbix_dir>/export_windows/<tag>/bank_plan.json` +sha256 | block 2, built |
| `sim_bank` (S5, after S4) | `<paths.banks_root>/<tag>_sim/BANK_RECORD.json` +sha256 | block 2, built |
| `real_bank` (S6, after S4) | `<paths.banks_root>/<tag>_real/shard_0000.npz` + `.json` | block 2, built |
| `search` (S7, after S5, S6) | `<paths.tune_root>/<search.campaign>/SEARCH_COMPLETE.json` | block 7, built |
| `best_fit` (S8, after S7) | `<paths.runs_root>/<tag>_best/BEST_FIT_COMPLETE.json` | block 7, built |

A stage whose artefact exists when the run starts is skipped, so the command
starts wherever the data set stands (for Giulia, S1-S6 are done: the command
starts at the search). Every stage is built (blocks 1-8, 2026-10-16); the
sequencer still refuses, before anything is submitted (exit 2), a stage
registered as not built or a configuration key no block honours, should a
future block add one.

## The input and bank stages (blocks 3 and 2)

S1-S3 drive Sbi-extractor's committed launchers as their headers say (D-062:
the pipeline changes none of them): `extractor/launch_stage_d.sh` (the
extraction array and the aggregation job; its own refusals -- a NEW cohort
whose flags file must be committed first, a root already holding a manifest
-- stop the run with their text), `sim_reextract/launch_sim_reextract.sh`
`plan` / `test` / `array` with RESUME=1 (the cohort's `profiles/<tag>.sh`
must exist in Sbi-extractor: the device geometry lives there), and
`preflight_label_axes.py` on the login node. S4-S6 drive `export_windows/
launch_bank.sh`: the pipeline writes the bank profile `<work_root>/bank_
profile_<tag>.json` from the configuration's `bank` block and the digests of
S1-S3's artefacts, then `plan` (pass line `BANK PLAN OK`), the test task
(`TASK 0 OK`), the array and its gate (`wrote <bank_root>/BANK_RECORD.json`),
and `real` (`REAL BANK OK`) -- the chain line BC of the Giulia build. The
launchers' variables (queues, cores, walltimes, concurrency) come from the
configuration's `sim` and `bank` blocks; `SBI_HPC_DIR` is passed as
`<paths.sbi_dir>/hpc`.

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

## The search and the best fit (block 7)

The search stage runs on the login node (in `envs.train`, through the
sequencer) the bank's split hash (`tools/bank_split_hash.py`, the runner's
own `grouped_split`), the tuner's `costliest`, `propose` and `finalists`;
and submits one job per probe and one job per trial (`jobs/run_argv.pbs`,
which reads the command from an argv file and runs it in `envs.train` with
the resources of `search.trial`). Its phases, logged as `search: ...` lines:

1. the budget gate: the space's costliest configuration at each searched
   batch size runs `budget_gate.probe_steps` steps and `probe_val_rows`
   validation rows on the trial's resources; the projected trial time x
   `safety_factor` must fit `search.trial.walltime` and the peak memory
   `search.trial.mem`, else `STOP: stage search failed: budget gate: ...`;
   the pass line is `search: budget gate passed: ...`;
2. round 0 (`n_initial_points` random configurations) and the GP rounds
   (`trials_per_round` each, expected improvement with the constant liar,
   the GP refitted on the ledger before each round), at most
   `max_concurrent_trials` trial jobs at once; a trial whose job ends with
   no ledger record is resubmitted with `--resume` up to
   `search.trial.max_resubmits` times, then recorded `killed`; the elapsed
   clock (D-090) starts at round 0's first submission and no new round
   starts past `max_total_walltime_h`;
3. `finalists --top-k 1` and `SEARCH_COMPLETE.json` (`ended_by` rounds,
   budget or no_new_proposals).

Everything lives under `<paths.tune_root>/<search.campaign>/`: `pending/`
and `trials/` (the tuner's), `runs/<trial>/` (the runner's outputs),
`gate/`, `jobs/` (argv files and job logs), `pipeline_rounds.json` (the
rounds, the identity, the gate, the clock -- it survives `--retry search`,
and a folder holding a search of another bank, split, space or campaign is
refused; a `--retry search` after a stop revives the trials recorded
`killed` -- raise `search.trial.walltime` or `max_resubmits` first, and
pass `--accept-config-change`). The best-fit stage reads the finalist from `SEARCH_COMPLETE.json`,
submits one job per `best_fit.member_seeds` entry on the split of
`best_fit.split_seed` (a member whose record exists is not re-run; a member
killed is resubmitted with `--resume` up to `best_fit.max_resubmits` times),
then computes the ensemble's report-split NLL (`tools/ensemble_nll.py`,
D-089) and writes `<paths.runs_root>/<tag>_best/BEST_FIT_COMPLETE.json`.

## The project folder

`projects/<tag>/<tag>.json` -- every key required, documented in the file by
the `_<key>` string beside it; the schema is `specs/SPEC.md` sec. 3.1 and
`config.py`. The Giulia file is config v5 (2026-10-16): v4's values (the
search's wall budget at 400 h, D-090) plus the queue of the trial and
member jobs (`search.trial.queue`, `best_fit.queue`: `cpu`).

## Layout

```
pipeline.py        the sequencer: run (--dry-run), status, paths
config.py          the schema, the loader, the NEEDS-CODE capability table (empty since block 7)
state.py           the state file
pbs.py             qsub / qstat -x -f, the only scheduler calls
envs.py            a command in a conda environment on the login node
stages/__init__.py the Stage interface, the graph, the registry
stages/common.py   what the stages share: the tuner's arguments, the qsub line, the clock
stages/inputs.py   S1-S3 (block 3)
stages/bank.py     S4-S6 (block 2)
stages/search.py   S7 (block 7)
stages/best_fit.py S8 (block 7)
jobs/run_argv.pbs  one job = one command line from an argv file
tools/             bank_split_hash.py, ensemble_nll.py (run in envs.train by the stages)
run_pipeline.sh    the one command (nohup, conda, --dry-run, --status)
projects/<tag>/    the project's configuration
tests/             the smoke tests, the fake PBS, the synthetic graph, the stand-in davinci (standin.py)
```

## On davinci (the first run)

Each line is one line; run them in this order and read each result before
the next. The pass condition of each is a line that must APPEAR.

```
# 1. the code and the cluster-side check (LF, ASCII, syntax, the imports); pass line: PIPELINE CODE OK
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && git fetch origin && git checkout sci/pipeline && git pull --ff-only origin sci/pipeline && git log --oneline -1 && bash -c 'set -e; cd hpc/pipeline; for f in $(find . -name "*.py" -o -name "*.sh" -o -name "*.pbs" -o -name "*.json" | grep -v __pycache__); do if LC_ALL=C grep -q $'"'"'\r'"'"' "$f"; then echo "CR in $f"; exit 1; fi; if LC_ALL=C grep -qP "[^\x00-\x7F]" "$f"; then echo "non-ASCII in $f"; exit 1; fi; done; python3 -m py_compile pipeline.py config.py state.py pbs.py envs.py stages/*.py tools/*.py tests/*.py; python3 -c "import sys; sys.path.insert(0, \".\"); import pipeline, stages; print(sorted(stages.BUILT))"; echo PIPELINE CODE OK'

# 2. the block-1 test under the fake PBS (stdlib; about 25 s); pass line: 6 pass, 0 fail
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tests/test_smoke_pipeline.py 2>&1 | tail -3

# 3. the dry run (submits nothing); pass: no REFUSE line, exit 0, S1-S6 "would be skipped", the gate's and the trials' qsub lines
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --dry-run 2>&1 | tee ~/pipeline_giulia_dry.log | grep -E "REFUSE|would|qsub -N|identity|clock|exit" ; echo "exit ${PIPESTATUS[0]}"

# 4. the run (nohup; returns at once); pass line: started : pid <n>
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia

# 5. any time: the status; the log's lines to watch, in order: `search: budget gate passed: ...`,
#    `search: round 0 proposed` / `submitted trial ...`, `search: round r complete (...)`,
#    `search: SEARCH COMPLETE (...)`, `best_fit: BEST FIT COMPLETE: ...`, `PIPELINE DONE: every stage done or skipped`
bash ~/SBI/hpc/pipeline/run_pipeline.sh --project giulia --status; tail -n 5 /davinci-1/home/ldellamea/pipeline/giulia/pipeline.log
```

A `STOP: stage search failed: budget gate: ...` line names the projected
trial time and memory against `search.trial.walltime` and `mem`: raise them
(or narrow the space) in `projects/giulia/giulia.json`, then re-run step 4
with `--retry search --accept-config-change` appended. A sequencer that
died (the login node rebooted) is resumed by step 4 again; it re-attaches
to the recorded jobs.

## Tests

```
python3 hpc/pipeline/tests/test_smoke_pipeline.py      # block 1, about 25 s, stdlib
python3 hpc/pipeline/tests/test_smoke_block2.py        # block 2, about 20 s, needs the Sbi-extractor checkout (SBIX_CLONE)
python3 hpc/pipeline/tests/test_smoke_block3.py        # block 3, about 7 s, the same
python3 hpc/pipeline/tests/test_smoke_block7.py        # block 7, about 2 min, needs torch, sbi, skopt
python3 hpc/pipeline/tests/test_smoke_block8.py        # block 8, about 1 min, end to end on the stand-in, needs all of the above
```

Block 1's smoke test under the fake PBS; blocks 4-6 (the runner) are tested by
`hpc/joint/stage3/smoke_test_blocks_4_6.py` (about 3 minutes; needs torch and sbi);
block 7's test runs the search and the best fit end to end on the fixture
banks under the fake PBS (the real tuner and runner in the trial jobs; a
trial killed at its walltime; the budget stop; the sequencer stopped and
re-run), with `PIPELINE_SKIP_CONDA=1`.
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
| 2 | the bank stages S4-S6 through `launch_bank.sh` (the profile from the configuration; the chain line BC as phases) | built 2026-10-16, smoke-tested on the stand-in |
| 3 | the input stages S1-S3 through `launch_stage_d.sh`, `launch_sim_reextract.sh`, `preflight_label_axes.py` | built 2026-10-16, smoke-tested (S3 on the real tool; S1-S2 on stubs of the launchers) |
| 4 | the campaign prior in the runner (`run_joint_arms.py`: a `latent_spec.kind` `campaign` bank uses the unit box, L0 = 0) | built 2026-10-15, smoke-tested |
| 5 | the streaming loader (D-084): `--buffer-shards K` streams the bank through a rolling buffer of K shards with the quota rule; K >= the shard count reproduces the in-memory run bit for bit | built 2026-10-15, smoke-tested |
| 6 | runner flags `--passes`, `--patience`, `--split-fracs`, `--split-seed`, `--head-pool-ops`, `--flow-hidden-layers`, `--threads`; a per-epoch checkpoint, `--resume`, `--stop-after-epoch` | built 2026-10-15, smoke-tested |
| 7 | the search and the best fit S7-S8: the budget gate, the rounds with the throttle and the resubmission with `--resume`, the elapsed clock, the finalists, the members and the ensemble NLL; config v5 | built 2026-10-16, smoke-tested |
| 8 | end to end on the stand-in (the real bank tools, tuner and runner; 11 jobs; the sequencer stopped and re-run); the davinci lines above | rehearsed 2026-10-16 |
