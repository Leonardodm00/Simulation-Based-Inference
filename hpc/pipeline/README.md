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
once (Giulia: 300 since 2026-10-30, D-115 -- 200 under D-099, 400 under
D-101; LindDUP: 200, D-109), from the extraction to the best
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
With Giulia's 300 (D-115, 2026-10-30; 400 under D-101, 200 under v9): 296
cores for the stages, so 6 trials (or members, or re-detection tasks) of 48
cores at once.

The dry run and `--status` print `cores: at most <cap> at once
(resources.max_cores, D-099): the sequencer's job <d>, the stages' jobs <b>;
in flight per the state file: <n> in <k> job(s)`; each stage's dry-run
lines say what the cap leaves it (`cores (D-099): at most 6 trial(s) at
once -- ...`). A stage that waits says so once: `<stage>: waits for cores:
...`, or its phase's line (`sim_bank: the array (its gate after it) waits
for cores (...)`).

To change the cap: edit `resources.max_cores` in `projects/<tag>/<tag>.json`
(the load refuses a value below `driver.ncpus` + the largest job the file
sizes), `--stop --wait 180`, then submit again with
`--accept-config-change`: the new chain re-attaches to every recorded job
and fills the room the cap leaves. A lower cap deletes nothing: the jobs in
flight finish, and no new one starts until the pipeline is under the cap
(Giulia's 300 of 2026-10-30, loaded by the lines "Giulia: a configuration
change in the running chain" under "On davinci").

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

## The synthetic kind (blocks 15-17, 2026-11-03)

A data set whose parameters are known: `"kind": "synthetic"` in the
configuration (specs/SPEC.md secs. 1.1, 2.1, 2.4, 3.1; D-118..D-123). No
recordings and no simulation campaign: the bench generator builds both arms
from `synthetic.*`, and every entry of `synthetic.searches` is one search
and its best fit. Its graph, in this order:

| stage | artefact of record | after |
|---|---|---|
| `sim_bank` | `<banks_root>/<tag>_sim/BANK_RECORD.json` +sha256, its `build` the configuration's | -- |
| `real_bank_<r>`, each entry r of `synthetic.real` | `<banks_root>/<tag>_real_<r>/BANK_RECORD.json`, the same | -- |
| `search_<q>`, each entry q of `synthetic.searches` | `<tune_root>/<q>/<campaign>/SEARCH_COMPLETE.json` | `sim_bank`, `real_bank_<q.real>` |
| `best_fit_<q>` | `<runs_root>/<q>/<tag>_best/BEST_FIT_COMPLETE.json` | `search_<q>` |
| `best_fit_alt_<q>`, each q with `alt_best_fit` true (block 18) | `<runs_root>/<q>/<tag>_best_alt/BEST_FIT_COMPLETE.json` | `search_<q>` |

A bank is one `jobs/run_argv.pbs` job per shard of
`hpc/joint/stage1/build_latent_bank.py` (the resources of
`synthetic.bank_job`, in `envs.train`), then one job of
`tools/synth_bank_record.py`, which checks every shard (CRCs, digests, rows,
wells per donor, classes) and writes the record, or prints `REFUSE:` lines.
A shard already built with the configuration's arguments is never built
again; a folder holding a bank of other arguments is refused, never
overwritten. A search and a best fit are S7 and S8 on the search's view of
the configuration: its own folders under `tune_root` and `runs_root`, its
campaign, the simulated bank and its real bank (for S-A1 too: its trials are
scored on R0, D-122). Their log lines start with the stage's name
(`search_S-A2-R0: round 1 complete ...`). [block 18, D-126] A search of
S-A2 or S-A5 with `alt_best_fit` true also fits the best trial (on the same
report-split NLL) of the arm its finalist is not -- A1, or the term's arm --
as `best_fit_alt_<q>`: S8 again with the same `best_fit` values, its members
named `ba_<tag>_s<k>`, its finalist chosen by the tuner's `finalists --arm`
for each of the two arms (the files `finalists_<arm>.json` in its folder), so
that every such search has an ensemble per arm for the evaluation. The stages share
`resources.max_cores`, and each tick fills the free cores in the graph's
order: the first search and its best fit first. The tools that read a
recorded data set's single search (`search_report.py`, `search_reset.py`,
`perrow_check.py`, `inputs_check.py`) refuse a synthetic configuration;
`purge_outputs.py` takes one with its own kinds (block 24, 2026-11-13:
`evaluation`, `best_fits`, `searches`, `banks`, `state`, and the archive of
the evaluation's results; the run sheet "synth: the relaunch" below).

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

`projects/synth/synth.json` (2026-11-03, the synthetic kind, config v10): the synthetic
A2/A5 experiment (D-118..D-123) -- arm S, 20,000 theta on the box in 20 shards; the
cohorts R0 (pi = 0) and R3 (pi = 0.3) of 3 conditions x 9 donors x 2 wells x 6 windows;
the five searches S-A1, S-A2-R0, S-A5-R0, S-A2-R3, S-A5-R3 of 20 + 4 x 10 trials (8 cores,
96gb, 24 h; 96gb since 2026-11-04, D-125, was 32gb) and their best fits of 5 members (96gb too); `resources.max_cores` 120 (D-120). The
values marked `[Claude's]` in its `_` keys are logistics proposed with the file (seeds,
shard size, the bank jobs, the members' memory and walltime, the loader's buffer).

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
stages/synthetic.py the synthetic kind: its banks, the run views, its graph (blocks 15, 16)
jobs/run_argv.pbs  one job = one command line from an argv file
jobs/activate_env.sh, chain.sh  the activation's fallback, the chain's replacement and watchdog (D-111)
jobs/stage_job.sh  the job protocol of the other repositories' job scripts (D-113)
tools/             bank_split_hash.py, ensemble_nll.py, bank_verify.py, synth_bank_record.py (run by the stages); probe_compute_qsub.pbs;
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
#    pl_giulia ...", the line "cores: at most 300 at once ... in flight per the state file: <n> in <k> job(s)" (300
#    since D-115, 2026-10-30; the jobs the state file records as queued or running), S1-S6 "would be skipped", "search: would run" with its
#    node-fault lines and "cores (D-099): at most 6 trial(s) at once ..." (300 cores since D-115; 8 at 400). A stage the state file records as failed
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

# R3. the dry run; pass: exit 0, no REFUSE, "search reset to pending (--retry)", "search: would run", "at most 6 trial(s) at once" (8 before D-115)
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

### Giulia: a configuration change in the running chain (D-115, 2026-10-30)

A changed `projects/giulia/giulia.json` (here `resources.max_cores` 400 ->
300, D-115) has a new digest: once it is on davinci, the next sequencer job
that starts without `--accept-config-change` -- the daily hand-over's
successor included -- refuses it and ends the chain on purpose. So the pull
(any pull of `~/SBI`, LindDUP's lines included) and the reload go together,
C1 to C4 in one sitting. Nothing running is deleted: the trials in flight
finish under the old room, and a new one starts only when it fits the new
cap (at 300: 6 trials of 48 at once, the sequencer's 4 beside them).

```
# C1. the code: step 1 above, then the value; pass lines: PIPELINE CODE OK, then max_cores 300
cd ~/SBI && python3 -c "import json; print('max_cores', json.load(open('hpc/pipeline/projects/giulia/giulia.json'))['resources']['max_cores'])"

# C2. stop the running sequencer; its trials keep running; pass line: stop: the sequencer has stopped
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --stop --wait 180 2>&1 | tail -n 3

# C3. the dry run with the change accepted; pass: exit 0, "search: would run", "cores: at most 300 at once", "at most 6 trial(s) at once"
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --dry-run --accept-config-change 2>&1 | tee ~/pipeline_giulia_dry_cap.log | grep -E "REFUSE|STOP|FAILED|would|cores" ; echo "exit ${PIPESTATUS[0]}"

# C4. the submission, only on that dry run; pass line: SUBMITTED the sequencer: job <id> (pl_giulia)
cd ~/SBI && [ ! -f /davinci-1/home/ldellamea/tune_joint_giulia/S-A1/SEARCH_COMPLETE.json ] && grep -q "search: would run" ~/pipeline_giulia_dry_cap.log && grep -q "cores: at most 300 at once" ~/pipeline_giulia_dry_cap.log && ! grep -qE "REFUSE|STOP:" ~/pipeline_giulia_dry_cap.log && bash hpc/pipeline/run_pipeline.sh --project giulia --accept-config-change || echo "STOP: not submitted (SEARCH_COMPLETE.json exists, or the dry run was not clean)"

# C5. about 10 minutes later, read-only; pass: today's "driver <id> on <node>: started; its successor ...", the event "configuration changed: <old> -> <new>", "cores: at most 300 at once"
W=/davinci-1/home/ldellamea/pipeline/giulia; qstat -u $USER | grep -E " pl_| t_"; grep -E ": started; its successor|could not start|watchdog|NODE FAULT|REFUSE" $W/pipeline.log | tail -n 4; cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --status 2>&1 | grep -E "configuration changed|cores: at most"
```

### Giulia: the halt and the deletion of the search and the simulated bank (D-116, D-117, 2026-11-02)

The simulated data the search used are corrupted (the user, 2026-11-02). First
the halt (D-116): the sequencer is stopped with `--stop`, so the chain ends on
purpose (`driver/CHAIN_END`) and no stage job's watchdog revives it; only then
are the jobs left deleted -- a plain `qdel` of the sequencer is an unclean end
that hands over, writes no `CHAIN_END`, and the next stage job to end submits
a sequencer again, which resubmits the deleted trials with `--resume`. H1 and
H3 select the jobs whose variables name the chain (`CHAIN_NAME=pl_giulia`), so
no other job is touched. Then the deletion (D-117) through
`tools/purge_outputs.py` (block 10): it lists, prints a CODE, and deletes only
with that CODE; it refuses while any job of the chain exists. Deleted: the
search's campaign folder `S-A1` (the ledger, the runs and their checkpoints,
the trial jobs' logs, the gate's probes), the rounds set aside on 2026-10-19
(`S-A1_set_aside_*`), the best fit's folder if any, the simulated bank
`giulia_sim` and its quarantine. Kept: the real bank `giulia_real`, the bank
plan's folder, the work root (state, log, reports), S1-S3 and S2's outputs.
The state file is not touched: a later run of these stages needs their
`--retry`. Do NOT submit the pipeline afterwards until what is rebuilt has
been decided: a submission would start the search again from the state.

```
# H1. read-only: the jobs of the chain; pass line: LISTED: <n> Giulia job(s), the jobs above it
n=0; while read -r j; do f=$(qstat -f "$j" 2>/dev/null | sed ':a;N;$!ba;s/\n\t//g'); echo "$f" | grep -qE 'CHAIN_NAME=pl_giulia(,|$)' || continue; n=$((n+1)); echo "GIULIA $j $(echo "$f" | awk -F' = ' '/^ *Job_Name = /{print $2; exit}') $(echo "$f" | awk -F' = ' '/^ *job_state = /{print $2; exit}')"; done < <(qselect -u "$USER" | tr -s ' \t' '\n'); echo "LISTED: $n Giulia job(s)"

# H2. stop the sequencer (up to 3 min); pass lines: stop: deleted the waiting sequencer job <id> ..., then stop: the
#     sequencer has stopped (still running after 180 s: run H2 again; once stopped it says stop: no sequencer is running)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --stop --wait 180 2>&1

# H3. delete the chain's jobs left; refused while a pl_giulia job exists; pass line: DELETE DONE: <n> Giulia job(s) deleted
s=(); g=(); while read -r j; do f=$(qstat -f "$j" 2>/dev/null | sed ':a;N;$!ba;s/\n\t//g'); echo "$f" | grep -qE 'CHAIN_NAME=pl_giulia(,|$)' || continue; if echo "$f" | grep -qE '^ *Job_Name = pl_giulia$'; then s+=("$j"); else g+=("$j"); fi; done < <(qselect -u "$USER" | tr -s ' \t' '\n'); if [ ${#s[@]} -gt 0 ]; then echo "REFUSED: the sequencer job(s) ${s[*]} still there -- run the stop line again first; nothing deleted"; else c=0; for j in "${g[@]}"; do qdel "$j" && { echo "DELETED $j"; c=$((c+1)); }; done; echo "DELETE DONE: $c Giulia job(s) deleted"; fi

# H4. the check; pass lines: lock: none (no sequencer ticking), sequencer jobs (pl_giulia): none queued or running,
#     chain: ended on purpose at ... stopped (exit 0): STOP_REQUEST: asked ... (D-111), Giulia jobs left: 0
#     (a LEFT <id> E line is a job still exiting: H4 again in a minute)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project giulia --status 2>&1 | grep -E 'lock: |sequencer jobs|chain: ended on purpose'; n=0; while read -r j; do f=$(qstat -f "$j" 2>/dev/null | sed ':a;N;$!ba;s/\n\t//g'); echo "$f" | grep -qE 'CHAIN_NAME=pl_giulia(,|$)' || continue; n=$((n+1)); echo "LEFT $j $(echo "$f" | awk -F' = ' '/^ *job_state = /{print $2; exit}')"; done < <(qselect -u "$USER" | tr -s ' \t' '\n'); echo "Giulia jobs left: $n"

# P1. the code: step 1 above, then; pass lines: PIPELINE CODE OK, then PURGE TOOL PRESENT
cd ~/SBI && test -f hpc/pipeline/tools/purge_outputs.py && echo PURGE TOOL PRESENT

# P2. the listing, deletes nothing; pass: the "would DELETE" lines (paths, files, GB), the "kept:" lines, and CODE <8 hex>
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tools/purge_outputs.py --config hpc/pipeline/projects/giulia/giulia.json --what search,best_fit,sim_bank 2>&1 | tee ~/purge_giulia_list.txt

# P3. the deletion, with the CODE that P2 printed in place of <CODE>; in the background (hundreds of GB take minutes;
#     the log survives a closed session); prints nothing here
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && nohup python3 hpc/pipeline/tools/purge_outputs.py --config hpc/pipeline/projects/giulia/giulia.json --what search,best_fit,sim_bank --confirm <CODE> > ~/purge_giulia_delete.log 2>&1 &

# P4. read-only, any time after P3; pass lines: the log's PURGE DONE: <k> folder(s) deleted, ..., and in the listing
#     only "absent" lines and "in all: 0 folder(s), 0 file(s), 0.00 GB" (while P3 still runs: DELETED lines so far)
tail -n 8 ~/purge_giulia_delete.log; cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tools/purge_outputs.py --config hpc/pipeline/projects/giulia/giulia.json --what search,best_fit,sim_bank 2>&1 | grep -E "REFUSED|absent|would DELETE|in all"
```

### LindDUP: the first submission (2026-10-29)

The same discipline: one line at a time, each read before the next. [2026-10-29 19:30,
D-112..D-114: L0 is new (the two Sbi-extractor commits of the day, through the laptop);
L1-L6 keep their commands with new pass lines; S2 runs on a campaign still being
written (D-112) and every job of S1, S2 and S5 runs under the job protocol (D-113). The
version of 17:30 is in git history.]

```
# L0. ON THE LAPTOP, in Git Bash -- not on davinci, whose clone (~/repos/Sbi-extractor) only
#     pulls, in L2 (D-062; HPC_PATHS sec. L). Once: the two Sbi-extractor commits of 2026-10-29,
#     from the two files sent in the chat, saved in Downloads. The laptop's working clone is
#     ~/Sbi-extractor (C:\Users\leona\Sbi-extractor); its ~/repos/ holds only the retired one.
#     The first line picks the file from the clone's tree: at e9aeb55 (tree dc10065f) both commits,
#     sbix_linddup_d107_to_d113.mbox; at the D-107 commit (tree 413e707a) the second only,
#     sbix_d112_d113.patch; any other state applies nothing and prints UNEXPECTED HEAD (paste it).
#     Pass line: the tree 9d8ea296cfa6bc5bed557cb707bee1431f216608. The second line pushes only
#     that tree (else NOT PUSHED); its pass line is git's "e9aeb55..<new> feat/real-arm-parity ->
#     feat/real-arm-parity". [2026-10-30: the push guarded by the tree, after L0 was run on davinci,
#     where nothing happened: no ~/Sbi-extractor there, so the chain stopped at its cd.]
cd ~/Sbi-extractor && git checkout feat/real-arm-parity && git pull --ff-only && case "$(git rev-parse HEAD^{tree})" in dc10065fc25a35c7ac0a84d237530cbb5c68e537) git am ~/Downloads/sbix_linddup_d107_to_d113.mbox ;; 413e707addd88ce37bda8c4b11d4fd91aea0d285) git am ~/Downloads/sbix_d112_d113.patch ;; *) echo "UNEXPECTED HEAD: $(git log -1 --format='%h %s')"; false ;; esac && git rev-parse HEAD^{tree}
cd ~/Sbi-extractor && [ "$(git rev-parse HEAD^{tree})" = 9d8ea296cfa6bc5bed557cb707bee1431f216608 ] && git push origin feat/real-arm-parity || echo "NOT PUSHED: the first line must print the tree 9d8ea296... first"

# L1. this code and its cluster-side check: step 1 above, then the job protocol's units; pass lines:
#     the D-112..D-114 commit or later from git log, PIPELINE CODE OK, then
#     "3 of 3 checks pass" (activate_units, job_units, stage_job_units: bash and stdlib).
#     The pull also reaches the running Giulia chain: its sequencer job keeps the code it loaded,
#     its successor and the jobs it submits read the new files; nothing they run changed (blocks
#     2, 3 and 9 and new helpers only; giulia.json untouched, so its digest is unchanged).
#     [2026-10-30, D-115: from the commit of D-115 on, giulia.json IS changed (resources.max_cores
#     300), so after this pull load it in the Giulia chain at once -- C2-C5 of "Giulia: a
#     configuration change in the running chain" -- or the Giulia chain's next hand-over refuses
#     the new digest and ends the chain on purpose.]
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
#     "environment faults (D-111)" lines, two "job protocol (D-113)" lines (S2, S5; a skipped stage
#     prints none, so S1's is absent), S2's "rounds (D-113)" and S5's "jobs that cannot start
#     (D-113)", the sequencer pl_linddup
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

### synth: the first submission (2026-11-03)

The synthetic A2/A5 experiment (D-118..D-123; the roadmap synth, step S5). The same
discipline: one line at a time, each read before the next. The pull of Y1 changes nothing
for the Giulia and LindDUP chains: their graphs, their configurations' digests and their
log lines are the same (blocks 15-17 add the synthetic kind; the recorded kind is untouched).

```
# Y1. the code: step 1 above (pass line PIPELINE CODE OK), then the synthetic kind's tests under the fake PBS
#     in sbi_env (the bench generator needs numpy and scipy; about 1 min; T17, which trains, ran in the sandbox);
#     pass line: 5 of 5 checks pass
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_env && python3 hpc/pipeline/tests/test_smoke_synthetic.py test_t15_config test_t15_graph test_t15_dry_run test_t16_record test_t16_banks 2>&1 | tail -3

# Y2. the dry run; pass: exit 0, no REFUSE or STOP line, "sim_bank: would run" with "bank sim: 20 shard(s) of 2000
#     row(s) (arm S)" and "folder: 0 of 20 shard(s) built already", real_bank_R0 and real_bank_R3 "would run" (324 rows
#     each), five "search_<q>: would wait for sim_bank, real_bank_<R>", "at most 14 trial(s) at once", "cores: at
#     most 120 at once", the sequencer pl_synth
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project synth --dry-run 2>&1 | tee ~/pipeline_synth_dry.log | grep -E "REFUSE|STOP|FAILED|would|bank |folder:|sequencer|cores: at most|trial\(s\) at once" ; echo "exit ${PIPESTATUS[0]}"

# Y3. the run; pass line: SUBMITTED the sequencer: job <id> (pl_synth)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project synth

# Y4. any time: the status and the log. Expected, in order: "sim_bank: submitted shard <k> -> <id>", "sim_bank:
#     shard <k> built (job <id>)", "sim_bank: every shard built (20); the record job checks the bank", "sim_bank:
#     BANK RECORDED: ...", the same for real_bank_R0 and real_bank_R3 (one shard each); then "search_S-A1:
#     budget gate passed: ...", its rounds, "search_S-A1: SEARCH COMPLETE (rounds) ...", "best_fit_S-A1: BEST
#     FIT COMPLETE ...", and so on for S-A2-R0, S-A5-R0, S-A2-R3, S-A5-R3; at the end "PIPELINE DONE: every stage
#     done or skipped". A STOP names the stage and its cause (a bank's: the shard, its job and its log's last
#     lines, or the record tool's REFUSE lines); after the fix: Y2 and Y3 with --retry <stage>
bash ~/SBI/hpc/pipeline/run_pipeline.sh --project synth --status; tail -n 12 /davinci-1/home/ldellamea/pipeline/synth/pipeline.log
```

### synth: the restart after the budget gate (2026-11-04)

The first submission stopped at `search_S-A2-R0`'s budget gate: the costliest configuration
peaks at 59.3 GB against `search.trial.mem` 32gb. The five gates measured the same (19.0, 32.4
and 59.2-59.3 GB at batch 256, 512 and 1024; every trial projected within 6.7 h with the
safety factor, against 24 h). D-125 raises the trials' and the members' memory to 96gb (1.5 x
59.3 GB, rounded up to 8 GB). The restart reuses the 15 probe files: no probe job is submitted
again, and each gate decides at the sequencer's first ticks (rehearsed on the stand-in in this
situation, one search failed at its gate and one with its probes ended unread: the same two
lines, then `PIPELINE DONE`, four probe jobs in all).

```
# Z1. the code: step 1 above; pass line PIPELINE CODE OK, git log at the D-125 commit or later

# Z2. the dry run; pass: exit 0, no REFUSE or STOP line, "search_S-A2-R0 reset to pending (--retry)",
#     the other four searches "running", and "search_<q>: would run" for all five
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project synth --dry-run --retry search_S-A2-R0 --accept-config-change 2>&1 | tee ~/pipeline_synth_dry2.log | grep -E "REFUSE|STOP|FAILED|reset to pending|would|\| search" ; echo "exit ${PIPESTATUS[0]}"

# Z3. the run; pass lines: "search_S-A2-R0: --retry accepted; the first sequencer job resets it to pending",
#     then "SUBMITTED the sequencer: job <id> (pl_synth)"
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project synth --retry search_S-A2-R0 --accept-config-change

# then Y4 at any time; next in the log: five "search_<q>: budget gate passed: worst projected ... <= 24:00:00;
# peak 59.xx GB <= 96gb", then each search's round 0 (at most 14 trials at once under the 120-core cap)
```

### synth: the other arm's best fits (2026-11-11, block 18, D-126)

The run of v2 ended `PIPELINE DONE` on 2026-11-07. For the evaluation (the roadmap's step S6)
every search with a term needs an ensemble of five members for each of its two arms; the run fitted
one, its finalist's (A1 in S-A2-R0 and S-A5-R3, A5 in S-A5-R0, A2 in S-A2-R3). v3 of `synth.json`
sets `alt_best_fit` true on those four searches: four new stages `best_fit_alt_<q>`, 20 member jobs
of 8 cores, 96gb and 120 h each (the `best_fit` values), at most 14 at once under the 120-core cap.
The banks, the searches and the five best fits are done and are not run again; nothing of theirs is
rewritten (rehearsed on the stand-in: the old artefacts byte-identical, only the new members'
jobs submitted). The configuration changed, so every line below carries `--accept-config-change`.

```
# X1. the code: step 1 above; pass line PIPELINE CODE OK, git log at the block-18 commit or later

# X2. the checks that need no training, in sbi_env (about 30 s); pass lines "6 of 6 checks pass" and "J37   PASS"
#     (the two dry-run checks skip where the project's work root exists, so they are not named here: X3 is the dry run)
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_env && python3 hpc/pipeline/tests/test_smoke_synthetic.py test_t15_config test_t15_graph test_t16_record test_t16_banks test_t18_config test_t18_graph 2>&1 | tail -1 && (cd hpc/joint/stage4 && python3 smoke_test_joint_tune.py -k J37 2>&1 | grep J37)

# X3. the dry run; pass: exit 0, no REFUSE or STOP line, the five searches and five best fits "done", four
#     "best_fit_alt_<q>: would run",
#     each with "run <q>: the best fit of the other arm (D-126) in .../runs/synth/<q>/synth_best_alt" and a
#     "qsub -N ba_synth_s<k>" line, "at most 5 member(s) at once" per stage and "cores: at most 120 at once"
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project synth --dry-run --accept-config-change 2>&1 | tee ~/pipeline_synth_dry3.log | grep -E "REFUSE|STOP|FAILED|would|\| (search|best_fit)|other arm|qsub -N ba|member\(s\) at once|cores: at most" ; echo "exit ${PIPESTATUS[0]}"

# X4. the run; pass line: SUBMITTED the sequencer: job <id> (pl_synth)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project synth --accept-config-change

# X5. any time: the status and the log. Expected, in order: four "best_fit_alt_<q>: submitted member seed <k> -> <id>"
#     groups (14 members at first, the other 6 as cores free up), then per stage five "member seed <k> done: L ...",
#     "best_fit_alt_<q>: BEST FIT COMPLETE: M=5 members, ensemble L_bar ...", at the end "PIPELINE DONE: every stage
#     done or skipped". Each folder <runs_root>/<q>/synth_best_alt holds finalists_A1.json and finalists_<A2|A5>.json
#     (the tuner's best trial of each arm) and finalist_config.json (the other arm's, with "arm" and "search_finalist").
#     A STOP names the stage and its cause; after the fix: X3 and X4 with --retry best_fit_alt_<q>
bash ~/SBI/hpc/pipeline/run_pipeline.sh --project synth --status; tail -n 12 /davinci-1/home/ldellamea/pipeline/synth/pipeline.log
```

### synth: the evaluation (2026-11-11, blocks 19-22, D-126, D-127)

The evaluation is a tool beside the pipeline, not a stage of it (`hpc/pipeline/evaluation/`): it reads
the best fits' members and the banks and writes under `/davinci-1/home/ldellamea/runs/synth/evaluation`
(the `out_root` of `projects/synth/synth_eval.json`). The running chain's `synth.json` and its code are
not touched, so the pull below does not make the chain refuse. One command, `submit_eval.sh`, submits
what is missing: the calibration bank of L-C2ST (5 jobs of 1 core, 2,000 windows each, a few minutes),
one sampling job per complete ensemble (8 cores, 64gb, at most 24 h; block 19) after the bank, and the
report (blocks 20-21) after the samplings. It can be run again at any time: what is done or queued is
not submitted twice, and an ensemble whose best fit is not complete yet is named and left for a later
run. The evaluation's jobs are outside the pipeline's 120-core cap (`resources.max_cores` counts the
pipeline's jobs only).

```
# E1. the code; pass: the last line names the block-22 commit or a later one
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && { module load proxy 2>/dev/null || true; } && git pull --ff-only origin sci/pipeline && git log --oneline -1

# E2. the job script's check (needs no training; a few seconds); pass line "1 of 1 checks pass"
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_env && python3 hpc/pipeline/tests/test_smoke_evaluation.py test_e22b_job_script 2>&1 | tail -1

# E3. the dry run of the calibration bank and the samplings; pass: exit 0, no REFUSE or ABORT line, five
#     "calbank:<k> would be submitted", five "sample:<q>:<arm> would be submitted" (S-A1:A1, S-A2-R0:A1,
#     S-A5-R0:A5, S-A2-R3:A2, S-A5-R3:A1), four "not complete:" lines naming the synth_best_alt folders still
#     training, and "EVAL DRY RUN: 10 job(s) would be submitted, 0 done, 0 live; nothing written"
cd ~/SBI && bash hpc/pipeline/evaluation/submit_eval.sh --project synth --what calbank,sample --dry-run 2>&1 | grep -E "REFUSE|ABORT|would be submitted|not complete|DRY RUN"; echo "exit ${PIPESTATUS[0]}"

# E4. the submission; pass line: EVAL SUBMIT OK: 10 job(s) submitted, 0 done, 0 live
cd ~/SBI && bash hpc/pipeline/evaluation/submit_eval.sh --project synth --what calbank,sample

# E5. any time: the jobs, then their pass lines. Each log ends with "EVAL JOB EXIT: 0"; a sampling job's
#     also has "EVAL SAMPLE OK: <stem> (<n> file(s), M=5, n_post=1000, oracle max ... nats, ... s)", 5 files
#     for S-A1 (sim, R0, R3, R3dag, cal) and 3 or 4 for the others. A REFUSE line names its cause (a member
#     that does not reproduce its own per-row NLL, a calibration bank of another build)
qstat -u ldellamea | grep ev_ ; grep -h "EVAL SAMPLE OK\|REFUSE\|EVAL JOB EXIT" /davinci-1/home/ldellamea/runs/synth/evaluation/jobs/logs/*.log

# E6. the report (blocks 20-21: metrics, then figures), queued after every sampling job still queued or
#     running; the default --what is calbank,sample,report. Dry run, pass: exit 0, no REFUSE or ABORT, "report
#     would be submitted" with "-W depend=afterok:<the sampling jobs' ids>" while they run (none once they ended)
cd ~/SBI && bash hpc/pipeline/evaluation/submit_eval.sh --project synth --dry-run 2>&1 | grep -E "REFUSE|ABORT|report|DRY RUN"; echo "exit ${PIPESTATUS[0]}"

# E7. its submission; pass line: EVAL SUBMIT OK: 1 job(s) submitted, ... (the samplings counted done or live)
cd ~/SBI && bash hpc/pipeline/evaluation/submit_eval.sh --project synth

# E8. the report's pass lines, in its log: EVAL METRICS OK: <n> ensemble(s), ..., EVAL FIGURES OK: <m> figure(s),
#     EVAL REPORT OK; then the figures' index, one line of caption per figure (open the PNGs from there)
grep -h "EVAL METRICS OK\|EVAL FIGURES OK\|EVAL REPORT OK\|REFUSE\|EVAL JOB EXIT" /davinci-1/home/ldellamea/runs/synth/evaluation/jobs/logs/report.log; head -n 20 /davinci-1/home/ldellamea/runs/synth/evaluation/figures/FIGURES.md

# E9. once the four best fits of the other arm are complete (the pipeline's log has four "best_fit_alt_<q>:
#     BEST FIT COMPLETE"): the dry run shows the five samplings and the report "done" and four "sample:<q>:<arm>
#     would be submitted" with the report after them; the submission "EVAL SUBMIT OK: 5 job(s) submitted"
cd ~/SBI && bash hpc/pipeline/evaluation/submit_eval.sh --project synth --dry-run 2>&1 | grep -E "REFUSE|ABORT| done |would be submitted|not complete|DRY RUN"; echo "exit ${PIPESTATUS[0]}"
cd ~/SBI && bash hpc/pipeline/evaluation/submit_eval.sh --project synth

# E10. the data overview (block 23, D-130): the windows, the eight statistics, theta and their PCA of the
#      synthetic data set, nine figures; it reads the banks only, so it needs no sampling and waits for none.
#      Dry run, pass: exit 0, no REFUSE or ABORT, "data would be submitted" with "ncpus=2:mem=8gb" and
#      "EVAL DRY RUN: 1 job(s) would be submitted, 0 done, 0 live; nothing written"; then the submission,
#      pass line "EVAL SUBMIT OK: 1 job(s) submitted, 0 done, 0 live"
cd ~/SBI && bash hpc/pipeline/evaluation/submit_eval.sh --project synth --what data --dry-run 2>&1 | grep -E "REFUSE|ABORT|data|DRY RUN"; echo "exit ${PIPESTATUS[0]}"
cd ~/SBI && bash hpc/pipeline/evaluation/submit_eval.sh --project synth --what data

# E11. its pass lines, a minute or two after it starts: "EVAL DATA OK: 4000 simulated windows, 324 + 324
#      cohort windows, 9 figure(s)" and "EVAL JOB EXIT: 0"; then the figures' index
grep -h "EVAL DATA\|REFUSE\|EVAL JOB EXIT" /davinci-1/home/ldellamea/runs/synth/evaluation/jobs/logs/data.log; cat /davinci-1/home/ldellamea/runs/synth/evaluation/data/FIGURES.md
```

A sampling that refuses (a `REFUSE:` line in its log, exit 3) writes
nothing, and PBS then drops the report queued after it. Once the cause is
fixed, the dry run lists that sampling again under "would be submitted",
with a new report after it, and the submission sends both; the samplings
already done stay done. (2026-11-12: four samplings refused on the rebuild
oracle's 1e-4 nats; the tolerance is now `oracle_tol` in `synth_eval.json`,
2e-3 -- `specs/SPEC.md` block 19.)

What is drawn and where (`<out_root>/figures/`, FIGURES.md lists each with its caption): per ensemble
`ensembles/<q>_<arm>/` -- `cal_*` the HPD expected coverage, SBC and TARP; `acc_*` the recovery, the z-scores
against the contraction, the posterior SDs and the NLL; `post_*` the corner plots of one example donor per
condition, the posterior-predictive ranks and L-C2ST; `emb_*` the members' embeddings -- per search
`searches/<q>/` the two arms side by side (`arms_*`, the forest plot of E3 among them) and the search itself
(`search_*`); `across/` every search's Dbar and every ensemble's NLL on R0 and R3. The numbers behind each
figure are in `<out_root>/metrics/` (`METRICS.json` is their index).

The data overview (E10, block 23) writes `<out_root>/data/`: `figures/` with `windows_cohorts` and
`windows_cohorts_zoom` (example windows of R0 and R3 by condition, full and around their maximum),
`windows_sim` and `windows_contaminated` (the simulated bank's and the contaminated windows), `stats_hist`
and `stats_strip` (the eight statistics of the first 4,000 simulated windows against the cohorts'),
`theta_cohorts` and `theta_dagger` (the cohorts' theta* on the unit box; theta-dagger on R3's shifted axes),
`stats_pca` (the statistics on the simulated windows' first two principal components); `FIGURES.md` with a
caption per figure, `data_view.npz` with every number drawn, `DATA_VIEW.json` with the counts and what it
read. `--what data` is never part of the default plan: E3-E9 are unchanged.

### synth: the relaunch -- the halt, the archive and the deletion (2026-11-13, block 24, D-132)

The synthetic experiment is relaunched with `fragment_duty` class-bearing and one search per
arm (D-132), so the last run's outputs go first. The halt as for Giulia (D-116): the
sequencer stopped with `--stop` (the chain ends on purpose, no watchdog revives it), then the
jobs left deleted -- the chain's (`CHAIN_NAME=pl_synth`, among them S-A2-R0's fit of the other
arm, which ends unfinished) and the evaluation's (`ev_*`). Then `tools/purge_outputs.py`
(blocks 10 and 24): it lists, prints a CODE, and with that CODE first packs the evaluation's
results into one archive in your home -- everything under `runs/synth/evaluation` but the
posterior samples and the calibration bank: the metrics, the figures, the report, the data
overview, the jobs' logs -- re-reads it and writes its `.sha256`, and only then deletes: the
evaluation, the five searches' best-fit folders (`runs/synth/<q>`), the five searches'
folders (`tune_joint_synth/<q>`), the banks `synth_sim`, `synth_real_R0`, `synth_real_R3`,
and every entry of the work root but `purge/` (the state file, the log): the next
submission starts a new run. It refuses while any job of the chain or of the evaluation's
ledger exists. Do NOT submit the pipeline afterwards until the relaunch's blocks (25-26)
are pulled: a submission would rebuild the old design.

```
# R1. read-only: the chain's jobs and the evaluation's; pass line: LISTED: <n> synth job(s), the jobs above it
n=0; while read -r j; do f=$(qstat -f "$j" 2>/dev/null | sed ':a;N;$!ba;s/\n\t//g'); nm=$(echo "$f" | awk -F' = ' '/^ *Job_Name = /{print $2; exit}'); if echo "$f" | grep -qE 'CHAIN_NAME=pl_synth(,|$)' || [[ "$nm" == ev_* ]]; then n=$((n+1)); echo "SYNTH $j $nm $(echo "$f" | awk -F' = ' '/^ *job_state = /{print $2; exit}')"; fi; done < <(qselect -u "$USER" | tr -s ' \t' '\n'); echo "LISTED: $n synth job(s)"

# R2. stop the sequencer (up to 3 min); pass lines: stop: deleted the waiting sequencer job <id> ..., then stop: the
#     sequencer has stopped (still running after 180 s: run R2 again; once stopped it says stop: no sequencer is running)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project synth --stop --wait 180 2>&1

# R3. delete the jobs left (the chain's and the evaluation's); refused while a pl_synth job exists;
#     pass line: DELETE DONE: <n> synth job(s) deleted
s=(); g=(); while read -r j; do f=$(qstat -f "$j" 2>/dev/null | sed ':a;N;$!ba;s/\n\t//g'); nm=$(echo "$f" | awk -F' = ' '/^ *Job_Name = /{print $2; exit}'); echo "$f" | grep -qE 'CHAIN_NAME=pl_synth(,|$)' || [[ "$nm" == ev_* ]] || continue; if [ "$nm" = "pl_synth" ]; then s+=("$j"); else g+=("$j"); fi; done < <(qselect -u "$USER" | tr -s ' \t' '\n'); if [ ${#s[@]} -gt 0 ]; then echo "REFUSED: the sequencer job(s) ${s[*]} still there -- run R2 again first; nothing deleted"; else c=0; for j in "${g[@]}"; do qdel "$j" && { echo "DELETED $j"; c=$((c+1)); }; done; echo "DELETE DONE: $c synth job(s) deleted"; fi

# R4. the check; pass lines: lock: none (no sequencer ticking), sequencer jobs (pl_synth): none queued or running,
#     chain: ended on purpose at ... stopped, synth jobs left: 0 (a LEFT <id> ... E line is a job still exiting: R4
#     again in a minute)
cd ~/SBI && bash hpc/pipeline/run_pipeline.sh --project synth --status 2>&1 | grep -E 'lock: |sequencer jobs|chain: ended on purpose'; n=0; while read -r j; do f=$(qstat -f "$j" 2>/dev/null | sed ':a;N;$!ba;s/\n\t//g'); nm=$(echo "$f" | awk -F' = ' '/^ *Job_Name = /{print $2; exit}'); echo "$f" | grep -qE 'CHAIN_NAME=pl_synth(,|$)' || [[ "$nm" == ev_* ]] || continue; n=$((n+1)); echo "LEFT $j $nm $(echo "$f" | awk -F' = ' '/^ *job_state = /{print $2; exit}')"; done < <(qselect -u "$USER" | tr -s ' \t' '\n'); echo "synth jobs left: $n"

# P1. the code (block 24) and its test under the fake PBS (stdlib and find, about 20 s); pass line:
#     -- 11 pass, 0 fail, 0 error, 1 skip
cd ~/SBI && git pull && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tests/test_smoke_purge.py 2>&1 | tail -n 1

# P2. the listing, deletes nothing; pass: "would ARCHIVE <m> file(s), <x> GB uncompressed, into
#     /davinci-1/home/ldellamea/synth_eval_2026-11-13.tar.gz", the two "not archived:" lines (samples/, calbank/),
#     the "would DELETE" lines (the evaluation, five runs/synth/<q>, five tune_joint_synth/<q>, three banks, the
#     work root's entries), "kept: /davinci-1/home/ldellamea/pipeline/synth/purge", and CODE <8 hex>
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tools/purge_outputs.py --config hpc/pipeline/projects/synth/synth.json --what evaluation,best_fits,searches,banks,state --archive ~/synth_eval_2026-11-13.tar.gz 2>&1 | tee ~/purge_synth_list.txt

# P3. the archive, then the deletion, with the CODE that P2 printed in place of <CODE>; in the background (the
#     checkpoints take minutes; the log survives a closed session); prints nothing here
cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && nohup python3 hpc/pipeline/tools/purge_outputs.py --config hpc/pipeline/projects/synth/synth.json --what evaluation,best_fits,searches,banks,state --archive ~/synth_eval_2026-11-13.tar.gz --confirm <CODE> > ~/purge_synth_delete.log 2>&1 &

# P4. read-only, any time after P3; pass lines: the log's "ARCHIVED <m> file(s) ..." and "PURGE DONE: <k> target(s)
#     deleted, ..., archive ... the next submission starts a new run"; the archive's member count <m>; its
#     checksum "OK"; the listing with only "absent" lines and "in all: 0 target(s), 0 file(s), 0.00 GB" (while
#     P3 still runs: DELETED lines so far)
tail -n 6 ~/purge_synth_delete.log; tar tzf ~/synth_eval_2026-11-13.tar.gz | wc -l; (cd ~ && sha256sum -c synth_eval_2026-11-13.tar.gz.sha256); cd ~/SBI && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 hpc/pipeline/tools/purge_outputs.py --config hpc/pipeline/projects/synth/synth.json --what evaluation,best_fits,searches,banks,state 2>&1 | grep -E "REFUSED|absent|would DELETE|in all"
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
python3 hpc/pipeline/tests/test_smoke_purge.py         # blocks 10 and 24, tools/purge_outputs.py, about 20 s, stdlib and the fake PBS
python3 hpc/pipeline/tests/test_smoke_perrow.py        # tools/perrow_check.py, a few seconds, numpy (with torch, sbi and zuko also the real-flow oracle)
python3 hpc/pipeline/tests/test_smoke_block8.py        # block 8, about 1 min, end to end on the stand-in, needs all of the above
python3 hpc/pipeline/tests/test_smoke_inputs_check.py  # block 9, tools/inputs_check.py, about 1 s, numpy and the Sbi-extractor checkout (SBIX_CLONE)
python3 hpc/pipeline/tests/test_smoke_synthetic.py     # blocks 15-18, the synthetic kind: about 30 s with numpy and scipy, 2 min more with torch, sbi, zuko, skopt (T17 and T18 end to end)
python3 hpc/pipeline/tests/test_smoke_evaluation.py    # blocks 19-23, the evaluation and the data overview: builds a reduced project once (about 1 min, torch, sbi, zuko, skopt), then about 5 min; EVAL_FIXTURE=<dir> reuses one, EVAL_KEEP=1 keeps it
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
| 10 | the purge tool `tools/purge_outputs.py`: the list-then-delete with a CODE of a project's search (with its set-aside rounds), best fit and simulated bank (with its quarantine), refused while any job of the chain exists (D-117) | built 2026-11-02, smoke-tested |
| 11-14 | the synthetic experiment's banks and arms in `hpc/joint/` (the box prior, the cohort and bank-wide ids, the shifted duty clipped at 1, A2 and A5 on a cohort) | built 2026-11-03, smoke-tested (`hpc/joint/stage3/smoke_test_blocks_11_14.py`) |
| 15 | the synthetic kind: its schema and cross-checks, `load_config(..., synthetic_ok)`, the graph, the run views, S7 and S8 per search | built 2026-11-03, smoke-tested (`test_smoke_synthetic.py` T15) |
| 16 | the synthetic banks: a job per shard, the record job (`tools/synth_bank_record.py`), the refusal of a folder of another build | built 2026-11-03, smoke-tested (T16) |
| 17 | the project `synth` (D-118..D-123) and its run sheet; end to end on a reduced copy under the fake PBS | built 2026-11-03, smoke-tested (T17); ran on davinci 2026-11-04..07 (`PIPELINE DONE`) |
| 18 | the other arm's best fit `best_fit_alt_<q>` (D-126): the tuner's `finalists --arm/--out`, the key `alt_best_fit`, synth.json v3 and its run sheet X1-X5 | built 2026-11-11, smoke-tested (T18; the tuner's J37) |
