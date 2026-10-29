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
running`, exit 4). A job whose node has lost the scheduler (every tick
`<stage>: scheduler unreachable (...)`, as on 2026-10-21 when a compute
node ran out of memory and its `qstat` answered `illegally formed job
identifier` for every id) hands over after 10 such ticks in a row, about
17 min, instead of waiting for its budget: `driver <id>: the scheduler was
unreachable for 10 consecutive tick(s); unclean end <n> in a row (3 end
the chain); the successor <id> continues` -- the successor starts where
PBS puts it; three such hand-overs in a row with no sound tick between
them end the chain as above (2026-10-21).

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
With Giulia's 400 (config v10, D-101; 200 under v9): 396 cores for the
stages, so 8 trials (or members, or re-detection tasks) of 48 cores at once.

The dry run and `--status` print `cores: at most <cap> at once
(resources.max_cores, D-099): the sequencer's job <d>, the stages' jobs <b>;
in flight per the state file: <n> in <k> job(s)`; each stage's dry-run
lines say what the cap leaves it (`cores (D-099): at most 8 trial(s) at
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
   `search: trial <id> failed: not a node fault (...); final`; a ledger
   record that exists but cannot be read (empty, or zeros: what a node
   that goes down after the record's rename leaves; davinci, 2026-10-19)
   is read again for 300 s (`PIPELINE_RECORD_GRACE_S`) -- `search: trial
   <id>: its ledger record ... cannot be read (...); read again for up to
   300 s` -- then set aside as `<id>.json.unreadable<k>` (kept) and the
   trial is taken as a job that ended without a record; a trial with no
   pending spec left (the tuner reads its configuration there) is recorded
   `killed` at once; before any
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
is resubmitted with `--resume` up to `best_fit.max_resubmits` times; a
member record that cannot be read is set aside as `<name>.unreadable<k>`
after the same 300 s and the member taken as killed),
then computes the ensemble's report-split NLL (`tools/ensemble_nll.py`,
D-089) and writes `<paths.runs_root>/<tag>_best/BEST_FIT_COMPLETE.json`.
The pipeline's own JSON files and the tuner's ledger records are fsynced
before their rename (2026-10-19).

**The avoid list (D-102, 2026-10-21).** A node that keeps killing jobs is
avoided by the pipeline itself: a host with 3 node faults in the search's
run (or in the best fit's) is appended to `<paths.work_root>/avoid_nodes.txt`
-- `search: host <node> has 3 node fault(s) in this run: added to the avoid
list ... (D-102)` -- one host per line, as `hostname` prints it, with a
comment above each. A trial or member job that starts on a listed host runs
nothing: its log says `RUN_ARGV AVOID: host <node> is on the avoid list ...;
holding this slot for 600 s, then exiting 94`, it holds the slot 10 minutes
(so the scheduler does not put the next queued job on the same node at
once) and exits 94, and the sequencer submits the trial again at once --
`search: trial <id> landed on <node>, which is on the avoid list (D-102):
its job <id> held the slot and exited; submitted again at once (not a
resubmission)` -- without counting a resubmission (`max_resubmits`
untouched). The file is yours to read (`cat`) and to clear: delete a host's
line, or the file, and the pipeline logs `host <node> is no longer on the
avoid list ... (cleared by hand); its node faults are counted again from
now` and lists it again only after 3 more faults; it never removes a host
by itself. The threshold and the hold are constants
(`PIPELINE_AVOID_NODE_FAULTS`, `PIPELINE_AVOID_HOLD_S` in the sequencer's
environment), not configuration keys, so a running chain picks the rule up
at its next hand-over after the pull with no `--accept-config-change`.
Not covered: the gate's probes, the bank's verification job, S1, S2 and the
bank tasks (short jobs; the bank quarantines bad shards, D-095).

**Environment faults (D-111, 2026-10-29).** A node without the conda
installation (davinci, 2026-10-27: PBS ran the sequencer and two trials again
on such nodes; all exited 98 before any command, then the successor too, and
the chain ended with no line in the log) no longer stops anything. Every job,
and every in-environment command of the sequencer, activates its environment
through `jobs/activate_env.sh`: conda first; when conda is absent or fails on
the node, the environment's own folder -- the prefix recorded in
`<paths.work_root>/envs/<env>.prefix` (written whenever conda works, and by
the submission on the login node), `$HOME/.conda/envs/<env>`, then
`CONDA_ENVS_PATH` -- whose python must start there (`RUN_ARGV ENV: host
<node>: conda unavailable (...); <env> activated from its folder <path>` in
the job's log). When nothing works the job cannot start (`RUN_ARGV ENV FAULT:
host <node>: ...`, exit 96): a trial's or a member's job holds its slot 10
minutes and the sequencer submits it again at once, not counted as a
resubmission (`search: trial <id> could not start on <node> (...): an
environment fault, not the trial's (D-111); submitted again at once (not a
resubmission)`), counted as a node fault on that host (so the avoid list
takes it at its third); a job PBS itself could not start (an `Exit_status`
-2 and its kin) is treated the same. A driver job that cannot start submits a
replacement of itself, delayed 5 minutes and twice that at each further
attempt up to an hour (`driver <id>: could not start on <node> (...); its
replacement <id> starts in 300 s (replacement 1 since the last full tick;
D-111)`), unless another job of the chain is queued, held or running; and
every trial or member job, at its end, checks that the chain is there and
revives it when it is gone without having ended on purpose (`watchdog: job
<id> on <node>: no sequencer job was queued, held or running ...; submitted
<id> to revive it (D-111)`). The chain ends on purpose only when the
pipeline is done, stopped, failed or refused, or after three unclean ends in
a row: then `<paths.work_root>/driver/CHAIN_END` holds why (`--status`
prints `chain: ended on purpose at ...`), no watchdog revives it, and a
submission removes it. Twelve environment faults in a row on three or more
nodes stop the stage instead (`STOP: ... the environment or the cluster, not
a node -- fix it, ... then --retry <stage>`): a broken environment, not a
node. Every trial and member runs with `--resume`, so a job PBS runs again
continues from its checkpoint; a checkpoint that cannot be loaded is moved
aside (`<name>.corrupt<k>`) and the run starts from epoch 0.

**Every job of the pipeline (D-113, 2026-10-29).** The same holds for the jobs
the input and bank stages submit through Sbi-extractor's launchers, whatever
the data set: the stages give the launchers `PIPELINE_JOBS_DIR` (this
checkout's `jobs/`), `WORK_ROOT`, `CHAIN_NAME`, `AVOID_FILE` and
`AVOID_HOLD_S`, which Sbi-extractor's `env.sh` passes to every job with qsub
-v, and every job script (S1's extraction array and aggregation, S2's tasks
through `run_mea_task.pbs` around ANN's `submit_mea_array.sh`, S2's gate, the
bank's tasks and gate) starts through `jobs/stage_job.sh`: the avoid list
(exit 94), the activation above (exit 96 when nothing starts, after holding
the slot), the chain's watchdog at its end. S2 and S5 submit again, free, a
test task, a gate or a verification job that could not start, and run free
rounds (not counted against their rerun rounds) for the tasks their gate or
verifier finds never started; three free rounds in a row with no fewer such
tasks stop the stage. S2's tasks that started and failed get two rerun rounds.
S1 has no rounds: a well whose job could not start stops it at the
aggregation (SPEC sec. 8).

`tools/search_report.py --config <file> [--qstat] [--trial <id>]` reads
(never writes) the state, the ledger, the pending specs, the run folders and
the job logs and prints, per trial, what the stage's next poll does with it
by the stage's own rules (`next: ...`), why the failed records failed (per
cause, one trial's traceback tail; with `--qstat` each job's node, exit,
walltime and memory), the knobs of the failed records against the ok ones
and the open trials, the failures grouped by cause and the unreadable
records; its last line is `SEARCH REPORT OK: <n> trial(s) read; nothing was
written`.

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

`projects/linddup/linddup.json` (2026-10-29, config v10): the LindDUP data set (D-104) -- the
MEAMA campaign `campaign_meama_rho1600_n01` against the DUP15HD recordings (`extracted_v2`,
DATA_C / DATA_P), S-A1 (D-106), the contact gate `stoa_gate_p` in theta (D-107: p = 34),
`resources.max_cores` 200 (D-109); its other values Giulia's, each `_` key saying which.
Values not yet read on the data carry `[to verify]`: `tools/inputs_check.py` reads them.
v2 of the same day (17:37): `sim.plan_args` `--quiet-hours 6` -- the campaign is still
being written, and S2 runs on the simulations already written, leaving out every task
changed in the last 6 h (D-112); the search's `gp_seed` 1 and `max_concurrent_trials`
25, Giulia's (D-114: the optimisation and training values are Giulia's; the cap still
runs 4 trials at once). Every job the pipeline submits, Sbi-extractor's and ANN's
included, starts through the job protocol `jobs/stage_job.sh` (D-113).

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
jobs/activate_env.sh, chain.sh  the activation's fallback, the chain's replacement and watchdog (D-111)
jobs/stage_job.sh  the job protocol of the other repositories' job scripts (D-113)
tools/             bank_split_hash.py, ensemble_nll.py, bank_verify.py (run by the stages); probe_compute_qsub.pbs;
                   read-only: search_report.py, perrow_check.py, inputs_check.py (block 9: a data set's inputs before its first submission)
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

# 2. the block-1 tests and the report's under the fake PBS (stdlib, about 7 min in all); pass lines, one per test:
#    -- 6 pass, 0 fail, 0 error, 0 todo, 1 skip      (the sequencer, about 25 s)
#    -- 6 pass, 0 fail, 0 error, 0 todo, 1 skip      (the sequencer's jobs, D-098, about 4 min)
#    -- 7 pass, 0 fail, 0 error, 0 todo, 1 skip      (the core cap, D-099, about 2 min; the launchers' cores from this checkout)
#    -- 5 pass, 0 fail, 0 error, 1 skip              (the search report, a few seconds)
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tests/test_smoke_pipeline.py 2>&1 | tail -1 && python3 hpc/pipeline/tests/test_smoke_block1b.py 2>&1 | tail -1 && SBIX_CLONE=/davinci-1/home/ldellamea/repos/Sbi-extractor python3 hpc/pipeline/tests/test_smoke_block1c.py 2>&1 | tail -1 && python3 hpc/pipeline/tests/test_smoke_report.py 2>&1 | tail -1

# 3. the search's trials, read-only (writes nothing; qstat is only asked); pass line: SEARCH REPORT OK: <n> trial(s)
#    read; nothing was written. Per trial `next:` says what the sequencer's next poll does with it; then the
#    failures grouped by cause and the unreadable records. Paste it whole.
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tools/search_report.py --config hpc/pipeline/projects/giulia/giulia.json --qstat 2>&1 | tee ~/search_report_giulia.txt

# 4. the dry run (submits nothing); pass: no REFUSE or STOP line, exit 0, the line "the sequencer: PBS jobs named
#    pl_giulia ...", the line "cores: at most 200 at once ... in flight per the state file: <n> in <k> job(s)" (the
#    jobs the state file records as queued or running), S1-S6 "would be skipped", "search: would run" with its
#    node-fault lines and "cores (D-099): at most 8 trial(s) at once ...". A stage the state file records as failed
#    prints "<stage>: FAILED in the state file ... add --retry <stage>" and exit 3: add that --retry to this line
#    and to step 5. --accept-config-change: the run was started under an earlier configuration version.
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --dry-run --accept-config-change 2>&1 | tee ~/pipeline_giulia_dry.log | grep -E "REFUSE|STOP|FAILED|would|sequencer|node fault|cores|exit" ; echo "exit ${PIPESTATUS[0]}"

# 5. the run: submits the sequencer's first job and returns; pass line: SUBMITTED the sequencer: job <id> (pl_giulia)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --accept-config-change

# 6. any time, from any login node: the status and the log. Lines to watch: `driver <id> on <node>: started; its
#    successor <id> waits for it (afterany)`, `sequencer started: job <id> on <node>`, the search's lines
#    (`search: trial <id> ... was never examined for a node fault; examining it`, `search: trial <id> NODE FAULT on
#    <node> (...); ... queued again with --resume (r of 2)`, `search: trial <id>: its ledger record ... cannot be
#    read (...); read again for up to 300 s` and, 300 s later, `... set aside as <id>.json.unreadable1 ...`,
#    `submitted trial ...`, `search: round r complete (...)`,
#    `search: SEARCH COMPLETE (...)`), `driver <id>: time budget reached; the successor <id> continues` once a day,
#    `best_fit: BEST FIT COMPLETE: ...`, `PIPELINE DONE: every stage done or skipped`; in --status, the `cores:`
#    line (at most 200 in flight). The first sequencer job waits in the queue while the trials submitted before
#    the cap fill the user's running-job limit; it starts when one of them ends.
bash ~/SBI/hpc/pipeline/run_pipeline.sh --project giulia --status; tail -n 12 /davinci-1/home/ldellamea/pipeline/giulia/pipeline.log
```

Restarting the search from nothing (the user's call of 2026-10-19 17:19):
`tools/search_reset.py --config <file>` lists what a reset would do -- the
campaign's `trials/`, `pending/`, `runs/`, `jobs/`, `pipeline_rounds.json`,
`finalists.json`, `SEARCH_COMPLETE.json` to set aside (moved to
`<campaign>_set_aside_<utc>/`, never deleted), `gate/` and `bank_split.json`
kept (the budget gate's probe files are reused: no probe jobs again), and the
trial jobs still queued or running, to be deleted (qdel) -- and prints a CODE;
with `--confirm <CODE>` it does it (pass line `SEARCH RESET DONE: ...`),
refused when a sequencer job is queued or running or the CODE no longer
matches. The state file is not touched: the submission that follows carries
`--retry search --accept-config-change`, and the search starts again at the
gate (reused) and round 0. Round 0's design follows `search.gp_seed`: the
same seed redraws the same configurations (verified: two proposals with seed
0 give the same 50 ids and configurations, seed 1 gives 50 others), which is
why config v10 (D-101) moved Giulia to seed 1 for the restart of 2026-10-19.
The lines, from any login node, after step 1 of the run sheet:

```
# R1. the listing (nothing done); pass line: CODE <8 hex> -- run again with --confirm <CODE> ...
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tools/search_reset.py --config hpc/pipeline/projects/giulia/giulia.json

# R2. the reset, with the CODE of R1; pass line: SEARCH RESET DONE: <n> job(s) deleted, round(s) set aside as ...
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tools/search_reset.py --config hpc/pipeline/projects/giulia/giulia.json --confirm <CODE>

# R3. the dry run; pass: exit 0, no REFUSE, "search reset to pending (--retry)", "search: would run", "at most 8 trial(s) at once"
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --dry-run --retry search --accept-config-change 2>&1 | tee ~/pipeline_giulia_dry.log | grep -E "REFUSE|STOP|FAILED|reset|would|sequencer|cores|exit" ; echo "exit ${PIPESTATUS[0]}"

# R4. the submission; pass line: SUBMITTED the sequencer: job <id> (pl_giulia)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --retry search --accept-config-change
```

Reading the trials' NLL (2026-10-20): `tools/perrow_check.py --config <file>
[--campaign-dir <dir>]... [--trial <id>]... [--bound 5.0]` reads (never
writes) every trial's per-row file `runs/<id>/*_perrow.npz` and its ledger
record and splits the report-split NLL as L = T + R. T is the flow's edge
term: a coordinate with |logit theta| > 5 (theta within 0.0067 of an edge of
the unit box, past the splines of zuko's NSF) adds
tau = u^2/2 + ln(2 pi)/2 + ln(theta (1 - theta)) nats in every trial, so T
is the same for every trial on the same rows; R = L - T is the part the
trial controls. It prints, per row set, T and a table per coordinate (the
shares past the bound, the part of T, the edge zone in physical units from
the bank's sidecar); per trial L, T, R and whether the ledger's nll equals L;
per trial its training curve from the runner record `runs/<id>/<stem>.json`
beside the per-row file (added 2026-10-20 19:15: the best epoch of the
selection-split NLL by the runner's rule, whether it is the last epoch --
still improving when the epochs ran out -- the gain of the last 5 epochs,
p_eff) and the line `TRAINING CURVES: best epoch the last in <k> of <n>
trial(s), in the last 5 in <m>; median gain of the last 5 epochs <g>
nats/row (<j> without a curve)`; and the pass line `PERROW CHECK OK: ...;
nothing was written`. The pull of
step 1 changes no module the running sequencer or the trial jobs use. The
lines, after step 1:

```
# P0. its test; pass line: -- 4 pass, 0 fail, 0 error, 2 skip (5 pass, 1 skip where torch, sbi and zuko import)
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tests/test_smoke_perrow.py 2>&1 | tail -1

# P1. read-only, both runs (the first one set aside); pass line: PERROW CHECK OK: <k> trial(s) read, <s> skipped, <m> without a per-row file; <r> row set(s); T = <t> nats/row; nothing was written -- above it the line TRAINING CURVES: ...
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tools/perrow_check.py --config hpc/pipeline/projects/giulia/giulia.json --campaign-dir /davinci-1/home/ldellamea/tune_joint_giulia/S-A1_set_aside_2026-10-19T161735Z 2>&1 | tee ~/perrow_check_giulia.txt

# P2. the bound of the trials' flows (read-only); expected: sbi 0.27.0 zuko 1.6.0 bound 5.0 -- another bound: P1 with --bound <it>; another sbi: the bound sbi passes was read at 0.27.0
~/.conda/envs/sbi_env/bin/python -c "import inspect, sbi, zuko; from zuko.transforms import MonotonicRQSTransform as M; print('sbi', sbi.__version__, 'zuko', zuko.__version__, 'bound', inspect.signature(M.__init__).parameters['bound'].default)"
```

A `STOP: stage search failed: budget gate: ...` line names the projected
trial time and memory against `search.trial.walltime` and `mem`: raise them
(or narrow the space) in `projects/giulia/giulia.json`, then submit again
(step 5) with `--retry search --accept-config-change` appended. A sequencer
job that died is followed by its successor; a chain that ended (`the chain
ends (...)` in the log, `none queued or running` in `--status`) is resumed by
step 5: it re-attaches to the recorded jobs. An accepted configuration change
(and a `--retry`) is saved before the first driver's first tick, so a first
tick that raises no longer leaves its successor refusing (2026-10-19).

### LindDUP: the first submission (2026-10-29)

The same discipline: one line at a time, each read before the next. [2026-10-29 19:30,
D-112..D-114: L0 is new (the two Sbi-extractor commits of the day, through the laptop);
L1-L6 keep their commands with new pass lines; S2 runs on a campaign still being
written (D-112) and every job of S1, S2 and S5 runs under the job protocol (D-113). The
version of 17:30 is in git history.]

```
# L0. on the laptop (Git Bash), once: the two Sbi-extractor commits of 2026-10-29 (D-062), from
#     the two files sent in the chat, saved in Downloads. The working clone is ~/Sbi-extractor
#     (C:\Users\leona\Sbi-extractor); ~/repos/ holds only the retired one (HPC_PATHS sec. L).
#     The line picks the file from the clone's tree: at e9aeb55 (tree dc10065f) both commits,
#     sbix_linddup_d107_to_d113.mbox; at the D-107 commit (tree 413e707a) the second only,
#     sbix_d112_d113.patch; any other state applies nothing and prints UNEXPECTED HEAD (paste it).
#     Pass line: the tree 9d8ea296cfa6bc5bed557cb707bee1431f216608; then the push.
cd ~/Sbi-extractor && git checkout feat/real-arm-parity && git pull --ff-only && case "$(git rev-parse HEAD^{tree})" in dc10065fc25a35c7ac0a84d237530cbb5c68e537) git am ~/Downloads/sbix_linddup_d107_to_d113.mbox ;; 413e707addd88ce37bda8c4b11d4fd91aea0d285) git am ~/Downloads/sbix_d112_d113.patch ;; *) echo "UNEXPECTED HEAD: $(git log -1 --format='%h %s')"; false ;; esac && git rev-parse HEAD^{tree}
git push origin feat/real-arm-parity

# L1. this code and its cluster-side check: step 1 above, then the job protocol's units; pass lines:
#     the D-112..D-114 commit or later from git log, PIPELINE CODE OK, then
#     "3 of 3 checks pass" (activate_units, job_units, stage_job_units: bash and stdlib).
#     The pull also updates the running Giulia chain at its next tick: nothing it runs changed
#     (blocks 2, 3 and 9 and new helpers only; giulia.json's digest unchanged).
cd ~/SBI && python3 hpc/pipeline/tests/test_smoke_block7c.py activate_units job_units stage_job_units 2>&1 | tail -1

# L2. the Sbi-extractor pull, its tree and its suites (SBI_HPC_DIR -> ~/SBI/hpc, pulled in L1);
#     pass lines, in order: the tree 9d8ea296..., then
#     -- 8 pass, 0 fail, 0 error, 0 todo, 2 skip / -- 41 pass, 0 fail, ... 2 skip /
#     -- 18 pass, 0 fail, ... 2 skip / ALL 40 CHECKS PASSED / ALL STAGE-D JOB CHECKS PASSED
cd ~/repos/Sbi-extractor && eval "$(conda shell.bash hook)" && conda activate sbi_export && { module load proxy 2>/dev/null || true; } && git pull --ff-only origin feat/real-arm-parity && git rev-parse HEAD^{tree} && source env.sh && python3 smoke_test_topology_axes.py 2>&1 | tail -1 && python3 export_windows/smoke_test_bank_plan.py 2>&1 | tail -1 && python3 export_windows/smoke_test_bank_export.py 2>&1 | tail -1 && ANN_TOOLS=~/ANN/MEA_analysis python3 sim_reextract/smoke_test_sim_reextract.py 2>&1 | tail -1 && (cd extractor && python3 smoke_test_stage_d_jobs.py 2>&1 | tail -1)

# L3. the inputs check (read-only; block 9); pass line: INPUTS CHECK: 0 FAIL, <w> WARN. Read the rows
#     `being written` (the tasks S2's plan leaves out now: changed in the last 6 h, D-112; a FAIL if
#     every task is), `pace` (a WARN means 6 h is short for this campaign's pace: say so before L5),
#     `plan option`, `protocol (SBI)` and `protocol (Sbi-extractor)` (both OK: D-113), and the qstat
#     lines (a `meama_cam*` array may still run: D-112 handles it). Paste it whole.
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tools/inputs_check.py --project linddup --json ~/inputs_check_linddup.json 2>&1 | tee ~/inputs_check_linddup.txt

# L4. the dry run; pass: exit 0, no REFUSE or STOP line, real_extraction "would be skipped", sim_redetection
#     and label_freeze "would run", "at most 4 trial(s) at once -- max_concurrent_trials 25", two
#     "environment faults (D-111)" lines, three "job protocol (D-113)" lines (S1, S2, S5), S2's
#     "rounds (D-113)" and S5's "jobs that cannot start (D-113)", the sequencer pl_linddup
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project linddup --dry-run 2>&1 | tee ~/pipeline_linddup_dry.log | grep -E "REFUSE|STOP|FAILED|would|sequencer|cores|exit|D-111|D-113" ; echo "exit ${PIPESTATUS[0]}"

# L5. the run; pass lines, in order: environment sbi_export: <prefix> (recorded for nodes without conda,
#     D-111), the same for sbi_env (both under /davinci-1/home/ldellamea/.conda/envs), then SUBMITTED the
#     sequencer: job <id> (pl_linddup)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project linddup

# L6. any time: the status and the log. Expected in the log: "sim_redetection: [plan] being written : <n>
#     task(s) changed less than 6 h before the plan, left out (D-112)"; S2's and S5's jobs print
#     "RUN_ARGV: host <h>, job <id>, env sbi_export (the pipeline's job protocol, D-113)" in their own logs.
#     A job that cannot start costs no round (D-113): "could not start on <host> (...): ... submitted
#     again", "a free round (<k>), not counted ...". STOPs that remain: a task failing after its rerun
#     rounds (S2 2, S5 bank.max_rerun_rounds 2), 12 environment faults in a row on 3 nodes, 3 free rounds
#     with no fewer unstarted tasks, a refusal no rerun fixes; each ends the chain on purpose: read the
#     STOP line and the job's log, then the dry run and the submission with --retry <stage>
bash ~/SBI/hpc/pipeline/run_pipeline.sh --project linddup --status; tail -n 12 /davinci-1/home/ldellamea/pipeline/linddup/pipeline.log
```

## Tests

```
python3 hpc/pipeline/tests/test_smoke_pipeline.py      # block 1, about 25 s, stdlib
python3 hpc/pipeline/tests/test_smoke_block1b.py       # block 1's sequencer jobs (D-098), about 3 min, stdlib
python3 hpc/pipeline/tests/test_smoke_block1c.py       # block 1's core cap (D-099), about 2 min, stdlib (SBIX_CLONE for its launcher check)
python3 hpc/pipeline/tests/test_smoke_block2.py        # block 2, about 2 min, needs the Sbi-extractor checkout (SBIX_CLONE) at or after D-113
python3 hpc/pipeline/tests/test_smoke_block3.py        # block 3, about 30 s, the same
python3 hpc/pipeline/tests/test_smoke_block7.py        # block 7, about 2 min, needs torch, sbi, skopt
python3 hpc/pipeline/tests/test_smoke_block7b.py       # block 7's node faults, checkpoints, unreadable records and the avoid list, block 1's stop and retry guard, about 10 min, the same
python3 hpc/pipeline/tests/test_smoke_block7c.py       # D-111: the environment's fallback, environment faults, the driver's replacement, the watchdog; D-113's stage_job_units; about 3 min, the same
python3 hpc/pipeline/tests/test_smoke_report.py        # tools/search_report.py, a few seconds, stdlib
python3 hpc/pipeline/tests/test_smoke_reset.py         # tools/search_reset.py, about 20 s, stdlib and the fake PBS
python3 hpc/pipeline/tests/test_smoke_perrow.py        # tools/perrow_check.py, a few seconds, numpy (with torch, sbi and zuko also the real-flow oracle)
python3 hpc/pipeline/tests/test_smoke_block8.py        # block 8, about 1 min, end to end on the stand-in, needs all of the above
python3 hpc/pipeline/tests/test_smoke_inputs_check.py  # block 9, tools/inputs_check.py, about 1 s, numpy and the Sbi-extractor checkout (SBIX_CLONE)
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
| 1 | the sequencer, the schema, the state, the scheduler adapter, the one command, config v4; the sequencer as a chain of PBS jobs on compute nodes, the lock and the state hardened, adoption, the transition (D-098, config v8); the core cap (D-099, config v9); the prepared state saved before the first tick (2026-10-19 (e)); a driver that loses the scheduler hands over (2026-10-21 (f)); an environment fault never ends the chain: the driver's replacement, the stage jobs' watchdog, the deliberate end (D-111, 2026-10-29 (g)) | built 2026-10-15, smoke-tested; amended 2026-10-18 (`test_smoke_block1b.py`) and 2026-10-19 (`test_smoke_block1c.py`; (e) in `test_smoke_block1b.py`); on davinci 2026-10-19: the probe passed (a compute node drives jobs), the three block-1 tests pass |
| 2 | the bank stages S4-S6 through `launch_bank.sh` (the profile from the configuration; the chain line BC as phases; a gate that never ran is run alone; the self-healing simulated bank: the verification job, the quarantine, the reruns, the certificate; config v6); the job protocol, and the jobs that cannot start cost no round or try (D-113) | built 2026-10-16, smoke-tested on the stand-in; amended 2026-10-16 (the dropped gate), 2026-10-18 (D-094, D-095), 2026-10-19 (D-099: the cap, the launcher's NCPUS passed; `test_cores`) and 2026-10-29 (b) (D-113: `test_env_faults`, `test_env_decisions`) |
| 3 | the input stages S1-S3 through `launch_stage_d.sh`, `launch_sim_reextract.sh`, `preflight_label_axes.py`; the job protocol for S1 and S2 (D-113); S2's free resubmissions, free rounds and rerun rounds (D-113); the plan's `--quiet-hours` through `sim.plan_args` (D-112) | built 2026-10-16, smoke-tested (S3 on the real tool; S1-S2 on stubs of the launchers); amended 2026-10-19 (D-099: the cap; `test_cores`) and 2026-10-29 (b) (D-113: `test_env_rounds`) |
| 4 | the campaign prior in the runner (`run_joint_arms.py`: a `latent_spec.kind` `campaign` bank uses the unit box, L0 = 0) | built 2026-10-15, smoke-tested |
| 5 | the streaming loader (D-084): `--buffer-shards K` streams the bank through a rolling buffer of K shards with the quota rule; K >= the shard count reproduces the in-memory run bit for bit | built 2026-10-15, smoke-tested |
| 6 | runner flags `--passes`, `--patience`, `--split-fracs`, `--split-seed`, `--head-pool-ops`, `--flow-hidden-layers`, `--threads`; a per-epoch checkpoint, `--resume`, `--stop-after-epoch`; the checkpoint fsynced, a damaged one moved aside on resume (D-111) | built 2026-10-15, smoke-tested; amended 2026-10-29 (`smoke_test_blocks_4_6.py` B6c) |
| 7 | the search and the best fit S7-S8: the budget gate, the rounds with the throttle and the resubmission with `--resume`, the elapsed clock, the finalists, the members and the ensemble NLL; config v5; node faults healed and corrupt checkpoints moved aside (D-096, D-097; config v7); the requested stop and the --retry guard (block 1); an unreadable record set aside and taken as none, the read-only `tools/search_report.py`; the read-only `tools/perrow_check.py` (the NLL split L = T + R); the bad-node avoid list (D-102); environment faults and reruns: the activation's fallback, free resubmissions, the guard, `--resume` on every trial and member (D-111) | built 2026-10-16, smoke-tested; amended 2026-10-18 (`test_smoke_block7b.py`) and 2026-10-19 (D-099: the throttle under the cap; `test_cores`; the unreadable records: `test_smoke_block7b.py`, `test_smoke_report.py`); 2026-10-20 (`test_smoke_perrow.py`); 2026-10-21 (the avoid list: `test_smoke_block7b.py` `avoid_units`, `avoid_list`); 2026-10-29 (D-111: `test_smoke_block7c.py`) |
| 8 | end to end on the stand-in (the real bank tools, tuner and runner; 11 jobs; the sequencer stopped and re-run); the davinci lines above | rehearsed 2026-10-16 |
| 9 | the inputs check `tools/inputs_check.py` (read-only, any data set; LindDUP's first, 2026-10-29); the plan's `--quiet-hours`, the tasks being written, the pace, the job protocol's two sides (D-112, D-113) | built 2026-10-29, smoke-tested (`test_smoke_inputs_check.py`; 12 mutations, each caught); amended 2026-10-29 (b) (`test_being_written`; 5 more mutations); not run on davinci |
