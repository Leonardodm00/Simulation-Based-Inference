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
bash hpc/pipeline/run_pipeline.sh --project giulia              # 2. the run: submits the sequencer's first PBS job and returns
bash hpc/pipeline/run_pipeline.sh --project giulia --status     # 3. any time: the stages, their jobs, the sequencer's jobs
bash hpc/pipeline/run_pipeline.sh --project giulia --stop --wait 180   # the sequencer stops at its next tick; the jobs keep running
```

**Nothing runs on a login node (D-098, 2026-10-18).** The user's logins land
on any login node, so no step may depend on one: the sequencer is a chain of
PBS jobs on compute nodes, named `pl_<tag>` (the configuration's `driver`
block: queue, cores, memory, walltime, the seconds between two ticks). Each
of them submits its successor first (`-W depend=afterany`: it starts when
this one ends, whatever ended it), ticks until its walltime less a margin,
and exits; the last one deletes its successor. Every command above, and
every line below, works from any login node and leaves nothing running
there: they submit, read, and write files on the shared file system. Never
start the sequencer in the background on a login node (`nohup`, `&`), and
never manage it with `ps` or `kill`: use `--status` and `--stop`.

`--project <tag>` names `projects/<tag>/<tag>.json`; `--config <path>` names
any configuration. The commands and the sequencer's jobs run under the
configuration's `envs.extract` conda environment and need only Python's
standard library. Further options go to the first sequencer job:
`--retry <stage>` (a stage recorded as failed is reset to pending),
`--accept-config-change` (continue a run under an edited configuration),
`--break-lock` (replace a lock at once instead of proving it stale).

Pass lines, as lines that must appear: the dry run ends WITHOUT a `REFUSE:`
line, prints `the sequencer: PBS jobs named pl_<tag> ...` and exits 0; the
start prints `SUBMITTED the sequencer: job <id> (pl_<tag>)`; the log then
shows `driver <id> on <node>: started; its successor <id> waits for it
(afterany)` and `sequencer started: job <id> on <node>`; `--status` lists
`sequencer jobs (pl_<tag>): <id> R, <id> H` while it runs; the log's last
lines at the end are `PIPELINE DONE: every stage done or skipped` and
`driver <id>: the chain ends (done, exit 0)`. A stop is
`STOP: stage <name> failed: <reason>` (exit 3): fix the cause, then submit
again with `--retry <name>`.

The sequencer's jobs. A job that ends at its budget hands over: `driver
<id>: time budget reached; the successor <id> continues`. A job killed (its
node, a qdel, its memory) is followed by its successor, which takes the
dead job's lock (`stale lock of driver job <id> (finished) replaced`) and
re-attaches to every recorded job; a job of the same name already queued or
running is adopted, never submitted twice (`<stage>: adopted ...`). Three
sequencer jobs in a row that end uncleanly before completing a tick end
the chain (`unclean end 3 in a row; the chain ends`, or `the last 3
sequencer jobs ended uncleanly ...`): read their logs in
`<work_root>/driver/`, fix, submit again (a submission resets the count). A second submission while the
chain runs is refused (`REFUSE: the sequencer is already queued or
running`, exit 4).

To load new code into a running pipeline: `--stop --wait 180` writes
`<work_root>/STOP_REQUEST` and deletes the waiting sequencer job; the
running one stops after its polls and the state's save -- never between a
submission and its record -- logs `stopped on request (...); resumable:
<n> job(s) still queued or running keep running`, deletes its successor
and releases its lock (`stop: the sequencer has stopped`). Pull, then
submit again (with `--accept-config-change` when the configuration
changed): the new chain re-attaches to the jobs. `--retry <stage>` is
refused while that stage's jobs still run (`REFUSE: --retry <stage>: the
stage has <n> job(s) still queued or running ...`): the reset forgets them
and would submit their work again.

## The core cap (D-099, 2026-10-19)

The pipeline's jobs never ask for more than `resources.max_cores` cores at
once (Giulia: 200, the user's allowance), from the extraction to the best
fit, counting every job queued, held or running -- a queued job can start
at any moment -- and the sequencer's own job (`driver.ncpus`). A job that
depends on another (a gate after its array, the aggregation after the
extraction array) starts when that one ends: the two hold the larger of
their cores, never both. A job waits until it fits: the search's trials and
the budget gate's probes, the best fit's members, the bank's and the
re-detection's test task, array and gate, the verification job. An array
with a concurrency limit is launched with `CONCURRENCY` = min(its configured
value, the room the cap leaves); the extraction array, which has none, only
when all of it fits (refused when it alone exceeds the share). Nothing
submitted is ever deleted for the cap: jobs submitted before it keep
running, and no new one starts until the pipeline is under it. A job that
can never fit (more than `max_cores` less `driver.ncpus`) stops its stage:
`<stage>: a job of <n> cores can never start ...; raise resources.max_cores`.
With Giulia's 200: 196 cores for the stages, so 4 trials (or members, or
re-detection tasks) of 48 cores at once.

The dry run and `--status` print `cores: at most <cap> at once
(resources.max_cores, D-099): the sequencer's job <d>, the stages' jobs <b>;
in flight per the state file: <n> in <k> job(s)`; each stage's dry-run
lines say what the cap leaves it (`cores (D-099): at most 4 trial(s) at
once -- ...`). A stage that waits says so once: `<stage>: waits for cores:
...`, or its phase's line (`sim_bank: the array (its gate after it) waits
for cores (...)`).

To raise the cap: edit `resources.max_cores` in `projects/<tag>/<tag>.json`
(the load refuses a value below `driver.ncpus` + the largest job the file
sizes), `--stop --wait 180`, then submit again with
`--accept-config-change`: the new chain re-attaches to every recorded job
and fills the larger room.

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
`preflight_label_axes.py` in the sequencer's job. S4-S6 drive `export_windows/
launch_bank.sh`: the pipeline writes the bank profile `<work_root>/bank_
profile_<tag>.json` from the configuration's `bank` block and the digests of
S1-S3's artefacts, then `plan` (pass line `BANK PLAN OK`), the test task
(`TASK 0 OK`), the array and its gate (`wrote <bank_root>/BANK_RECORD.json`),
and `real` (`REAL BANK OK`) -- the chain line BC of the Giulia build. After
every round's gate a verification JOB (`tools/bank_verify.py`, in
`envs.extract`, `bank.verify`'s resources) re-reads every shard and checks
its sha256 against its task's marker and every zip member's CRC; a bad or
unfinished task is MOVED (never deleted) to `<banks_root>/<tag>_sim_quarantine/
<time>_round<r>/` with a `QUARANTINE.json`, and the array runs again, at most
`bank.max_rerun_rounds` times (D-094, D-095); the stage is done when
`BANK_VERIFIED.json` certifies `BANK_RECORD.json`. A gate the scheduler ends
without running it (no record, no log of its own: davinci, 2026-10-16) is
run alone once its round verifies; a gate log left by an earlier run is not
read. The
launchers' variables (queues, cores, walltimes, concurrency) come from the
configuration's `sim` and `bank` blocks; `SBI_HPC_DIR` is passed as
`<paths.sbi_dir>/hpc`.

## The run's state

`<paths.work_root>/state.json` records every stage's status (`pending`,
`running`, `done`, `failed`, `skipped`), its submitted jobs (id, role, the
qsub line, the log path, the exit status once finished), its resubmissions
and its notes; `state.json.prev` is the state before the last write (read
when `state.json` does not parse); `pipeline.log` is the sequencer's log;
`pipeline.lock` names the sequencer that ticks (`job:<id> <node>`; a second
sequencer on the same work root exits 4); `driver/driver.<k>.log` is the
k-th sequencer job's own log (a traceback lands there), `driver/
driver.<k>.argv` its command line. A new sequencer job reads the state file,
re-attaches to the recorded jobs through `qstat -x -f`, and goes on. The
window of a sequencer killed between `qsub` returning and the state file's
write is closed by adoption: the job, queued or running under its name, is
recorded instead of submitted again.

## The search and the best fit (block 7)

The search stage runs in the sequencer's job (in `envs.train`, through the
sequencer; on the login node before D-098) the bank's split hash (`tools/bank_split_hash.py`, the runner's
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
   `search.trial.max_resubmits` times, then recorded `killed`; a trial
   recorded `failed` whose runner was killed by SIGBUS or whose job log
   shows a file-system error (EIO, ENOTCONN, ESHUTDOWN, ESTALE, EREMOTEIO)
   is a node fault (D-096, `search.resubmit_node_faults`): `search: trial
   <id> NODE FAULT on <node> (<why>); record set aside as <id>.json.
   nodefault<k>; queued again with --resume (<r> of <max>)`, within the
   same `max_resubmits`, then `killed`; any other failure is final,
   `search: trial <id> failed: not a node fault (...); final`; before any
   submission a checkpoint whose zip CRCs fail is moved aside (`<name>.
   corrupt<k>`, D-097) and the trial starts from epoch 0; the elapsed
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
killed, or failed of a node fault (D-096, `best_fit.resubmit_node_faults`),
is resubmitted with `--resume` up to `best_fit.max_resubmits` times),
then computes the ensemble's report-split NLL (`tools/ensemble_nll.py`,
D-089) and writes `<paths.runs_root>/<tag>_best/BEST_FIT_COMPLETE.json`.

## The project folder

`projects/<tag>/<tag>.json` -- every key required, documented in the file by
the `_<key>` string beside it; the schema is `specs/SPEC.md` sec. 3.1 and
`config.py`. The Giulia file is config v9 (2026-10-19): v8's values plus
`resources.max_cores` 200 (D-099, the user's); a run started under v6, v7
or v8 needs `--accept-config-change` once. v8 (2026-10-18): v7's values plus
the `driver` block (`cpu`, 4 cores, `16gb`, `24:00:00`, a tick every 60 s;
D-098, Claude's values, proposed). v7 (2026-10-18): v6's values plus
`search.resubmit_node_faults` and `best_fit.resubmit_node_faults`, both
true (D-096).
v6 (2026-10-18): v5's values plus the
self-healing bank's keys (`bank.verify`: 8 cores, 16gb, 02:00:00;
`bank.max_rerun_rounds`: 2; D-094). v5 (2026-10-16): v4's values (the
search's wall budget at 400 h, D-090) plus the queue of the trial and
member jobs (`search.trial.queue`, `best_fit.queue`: `cpu`). A run started
under v5 needs `--accept-config-change` once.

## Layout

```
pipeline.py        the sequencer: submit, drive (its PBS job), run (--dry-run; the loop: tests), status, stop, retire, paths; the core cap's count (D-099)
config.py          the schema, the loader, the NEEDS-CODE capability table (empty since block 7)
state.py           the state file
pbs.py             qsub, qstat -x -f, qselect, qdel: the only scheduler calls
envs.py            a command in a conda environment (in the sequencer's job)
stages/__init__.py the Stage interface, the graph, the registry
stages/common.py   what the stages share: the tuner's arguments, the qsub line, the clock
stages/inputs.py   S1-S3 (block 3)
stages/bank.py     S4-S6 (block 2)
stages/search.py   S7 (block 7)
stages/best_fit.py S8 (block 7)
jobs/run_argv.pbs  one job = one command line from an argv file
tools/             bank_split_hash.py, ensemble_nll.py, bank_verify.py (run by the stages); probe_compute_qsub.pbs
run_pipeline.sh    the one command (conda; submit, --dry-run, --status, --stop, --retire-login-sequencer)
projects/<tag>/    the project's configuration
tests/             the smoke tests, the fake PBS, the synthetic graph, the stand-in davinci (standin.py)
```

## On davinci

Each line is one line, for ANY login node; run them in this order and read
each result before the next. The pass condition of each is a line that must
APPEAR.

Done once, 2026-10-19, not to repeat: the probe of the scheduler from a
compute node (`qsub -q cpu -o $HOME/probe_compute_qsub.log
~/SBI/hpc/pipeline/tools/probe_compute_qsub.pbs`, then `cat
$HOME/probe_compute_qsub.log $HOME/probe_compute_qsub_child.log`: `PROBE:
QSUB OK`, `QSTAT OK`, `QSELECT OK`, `QDEL OK`, `PROBE: DONE`, `CHILD: ran on
<node>` -- passed on dvnode002, job 1851748, 10:52), and the retirement of the
sequencer started on a login node by the code before D-098
(`run_pipeline.sh --project giulia --retire-login-sequencer`; its lock was
already gone). A new cluster, or a new queue for the sequencer's jobs, runs
the probe again first (its header names the queue).

```
# 1. the code and the cluster-side check (LF, ASCII, syntax, the imports); pass line: PIPELINE CODE OK
#    (`module load proxy`: git reaches GitHub through the cluster's proxy; skipped where there is no such module)
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && { module load proxy 2>/dev/null || true; } && git fetch origin && git checkout sci/pipeline && git pull --ff-only origin sci/pipeline && git log --oneline -1 && bash -c 'set -e; cd hpc/pipeline; for f in $(find . -name "*.py" -o -name "*.sh" -o -name "*.pbs" -o -name "*.json" | grep -v __pycache__); do if LC_ALL=C grep -q $'"'"'\r'"'"' "$f"; then echo "CR in $f"; exit 1; fi; if LC_ALL=C grep -qP "[^\x00-\x7F]" "$f"; then echo "non-ASCII in $f"; exit 1; fi; done; python3 -m py_compile pipeline.py config.py state.py pbs.py envs.py stages/*.py tools/*.py tests/*.py; python3 -c "import sys; sys.path.insert(0, \".\"); import pipeline, stages; print(sorted(stages.BUILT))"; echo PIPELINE CODE OK'

# 2. the block-1 tests under the fake PBS (stdlib, about 6 min in all); pass lines, one per test:
#    -- 6 pass, 0 fail, 0 error, 0 todo, 1 skip      (the sequencer, about 25 s)
#    -- 6 pass, 0 fail, 0 error, 0 todo, 1 skip      (the sequencer's jobs, D-098, about 3 min)
#    -- 7 pass, 0 fail, 0 error, 0 todo, 1 skip      (the core cap, D-099, about 2 min; the launchers' cores from this checkout)
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tests/test_smoke_pipeline.py 2>&1 | tail -1 && python3 hpc/pipeline/tests/test_smoke_block1b.py 2>&1 | tail -1 && SBIX_CLONE=/davinci-1/home/ldellamea/repos/Sbi-extractor python3 hpc/pipeline/tests/test_smoke_block1c.py 2>&1 | tail -1

# 3. the dry run (submits nothing); pass: no REFUSE or STOP line, exit 0, the line "the sequencer: PBS jobs named
#    pl_giulia ...", the line "cores: at most 200 at once ... in flight per the state file: <n> in <k> job(s)" (the
#    jobs the state file records as queued or running), S1-S6 "would be skipped", "search: would run" with its
#    node-fault lines and "cores (D-099): at most 4 trial(s) at once ...". A stage the state file records as failed
#    prints "<stage>: FAILED in the state file ... add --retry <stage>" and exit 3: add that --retry to this line
#    and to step 4. --accept-config-change: the run was started under an earlier configuration version.
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --dry-run --accept-config-change 2>&1 | tee ~/pipeline_giulia_dry.log | grep -E "REFUSE|STOP|FAILED|would|sequencer|node fault|cores|exit" ; echo "exit ${PIPESTATUS[0]}"

# 4. the run: submits the sequencer's first job and returns; pass line: SUBMITTED the sequencer: job <id> (pl_giulia)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --accept-config-change

# 5. any time, from any login node: the status and the log. Lines to watch: `driver <id> on <node>: started; its
#    successor <id> waits for it (afterany)`, `sequencer started: job <id> on <node>`, the search's lines
#    (`search: trial <id> ... was never examined for a node fault; examining it`, `search: trial <id> NODE FAULT on
#    <node> (...); ... queued again with --resume (r of 2)`, `submitted trial ...`, `search: round r complete (...)`,
#    `search: SEARCH COMPLETE (...)`), `driver <id>: time budget reached; the successor <id> continues` once a day,
#    `best_fit: BEST FIT COMPLETE: ...`, `PIPELINE DONE: every stage done or skipped`; in --status, the `cores:`
#    line (at most 200 in flight). The first sequencer job waits in the queue while the trials submitted before
#    the cap fill the user's running-job limit; it starts when one of them ends.
bash ~/SBI/hpc/pipeline/run_pipeline.sh --project giulia --status; tail -n 12 /davinci-1/home/ldellamea/pipeline/giulia/pipeline.log
```

A `STOP: stage search failed: budget gate: ...` line names the projected
trial time and memory against `search.trial.walltime` and `mem`: raise them
(or narrow the space) in `projects/giulia/giulia.json`, then submit again
(step 4) with `--retry search --accept-config-change` appended. A sequencer
job that died is followed by its successor; a chain that ended (`the chain
ends (...)` in the log, `none queued or running` in `--status`) is resumed by
step 4: it re-attaches to the recorded jobs.

## Tests

```
python3 hpc/pipeline/tests/test_smoke_pipeline.py      # block 1, about 25 s, stdlib
python3 hpc/pipeline/tests/test_smoke_block1b.py       # block 1's sequencer jobs (D-098), about 3 min, stdlib
python3 hpc/pipeline/tests/test_smoke_block1c.py       # block 1's core cap (D-099), about 2 min, stdlib (SBIX_CLONE for its launcher check)
python3 hpc/pipeline/tests/test_smoke_block2.py        # block 2, about 2 min, needs the Sbi-extractor checkout (SBIX_CLONE)
python3 hpc/pipeline/tests/test_smoke_block3.py        # block 3, about 7 s, the same
python3 hpc/pipeline/tests/test_smoke_block7.py        # block 7, about 2 min, needs torch, sbi, skopt
python3 hpc/pipeline/tests/test_smoke_block7b.py       # block 7's node faults and checkpoints, block 1's stop and retry guard, about 6 min, the same
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
| 1 | the sequencer, the schema, the state, the scheduler adapter, the one command, config v4; the sequencer as a chain of PBS jobs on compute nodes, the lock and the state hardened, adoption, the transition (D-098, config v8); the core cap (D-099, config v9) | built 2026-10-15, smoke-tested; amended 2026-10-18 (`test_smoke_block1b.py`) and 2026-10-19 (`test_smoke_block1c.py`); on davinci 2026-10-19: the probe passed (a compute node drives jobs), the three block-1 tests pass |
| 2 | the bank stages S4-S6 through `launch_bank.sh` (the profile from the configuration; the chain line BC as phases; a gate that never ran is run alone; the self-healing simulated bank: the verification job, the quarantine, the reruns, the certificate; config v6) | built 2026-10-16, smoke-tested on the stand-in; amended 2026-10-16 (the dropped gate), 2026-10-18 (D-094, D-095) and 2026-10-19 (D-099: the cap, the launcher's NCPUS passed; `test_cores`) |
| 3 | the input stages S1-S3 through `launch_stage_d.sh`, `launch_sim_reextract.sh`, `preflight_label_axes.py` | built 2026-10-16, smoke-tested (S3 on the real tool; S1-S2 on stubs of the launchers); amended 2026-10-19 (D-099: the cap; `test_cores`) |
| 4 | the campaign prior in the runner (`run_joint_arms.py`: a `latent_spec.kind` `campaign` bank uses the unit box, L0 = 0) | built 2026-10-15, smoke-tested |
| 5 | the streaming loader (D-084): `--buffer-shards K` streams the bank through a rolling buffer of K shards with the quota rule; K >= the shard count reproduces the in-memory run bit for bit | built 2026-10-15, smoke-tested |
| 6 | runner flags `--passes`, `--patience`, `--split-fracs`, `--split-seed`, `--head-pool-ops`, `--flow-hidden-layers`, `--threads`; a per-epoch checkpoint, `--resume`, `--stop-after-epoch` | built 2026-10-15, smoke-tested |
| 7 | the search and the best fit S7-S8: the budget gate, the rounds with the throttle and the resubmission with `--resume`, the elapsed clock, the finalists, the members and the ensemble NLL; config v5; node faults healed and corrupt checkpoints moved aside (D-096, D-097; config v7); the requested stop and the --retry guard (block 1) | built 2026-10-16, smoke-tested; amended 2026-10-18 (`test_smoke_block7b.py`) and 2026-10-19 (D-099: the throttle under the cap; `test_cores`) |
| 8 | end to end on the stand-in (the real bank tools, tuner and runner; 11 jobs; the sequencer stopped and re-run); the davinci lines above | rehearsed 2026-10-16 |
