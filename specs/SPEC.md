# SPEC -- Simulation-Based-Inference: the one-command pipeline (`hpc/pipeline/`)

Single source of truth for what the code must do. The code follows this file;
when the two disagree, fix whichever is wrong and record why in the decision
log (`claude/SBI_decisions_and_ideas_log.md` in the project's knowledge base;
decisions are cited here by their `D-nnn` IDs and their binding content is
restated, so the tester never needs the log). The integration tester reads
only the repository, so every constraint the code must satisfy is stated here,
including those that come from decisions.

Notation rules: every symbol is defined at first use; a conditional quantity
keeps its conditioning every time it appears (write F(y | x), never F(y));
quantifiers ("for each fixed x", "for all alpha") are stated explicitly.

The branch `sci/spec-docs-tools` holds another `specs/SPEC.md`, for the
joint-docs number scripts (Stage 5). The two are sections of one file once
both are on `main`; until then each covers its own paths (sec. 0).

Pure ASCII, LF only (the file ships to a PBS cluster with the code).

Last reviewed at commit: 9fcbc88 (the base of `sci/pipeline`; this file is
new on that branch)

## 0. Coverage

Paths this spec covers (files, directories ending in `/`, or globs):

- `hpc/pipeline/` -- the pipeline: `pipeline.py`, `config.py`, `state.py`,
  `pbs.py`, `envs.py`, `stages/`, `jobs/`, `tools/`, `run_pipeline.sh`,
  `README.md`, `tests/`
- `hpc/pipeline/projects/giulia/giulia.json` -- the Giulia project's
  configuration (config v6)
- the changes the later blocks make under `hpc/joint/` (listed per block in
  sec. 5 as they are written; the rest of `hpc/joint/` is their input, not
  under test here): `hpc/joint/stage1/latent_bank.py` (`ShardIndex`,
  `npz_member_shape`), `stage2/joint_batches.py` (`ShardStream`, the
  batcher's `sim_stream` and state), `stage2/joint_train.py` (`evaluate_npe_
  shards`, the checkpoint, `StoppedAfterEpoch`), `stage2/joint_model.py`
  (`flow_hidden_layers`), `stage3/run_joint_arms.py` (the campaign prior,
  the streaming path, the flags, the probe mode), `stage3/smoke_test_blocks_4_6.py`,
  `stage4/joint_space.py` (the `head_pool_ops` axis, `apply_overrides`),
  `stage4/npe_tune_joint.py` (the runner flags, `--space-overrides`,
  `--gp-seed`, `--runner-extra`, `costliest`)

Everything else in the repository is not specified yet and is out of scope
for the tester.

## 1. Scientific goal

One bash command takes a data set -- a folder of simulated spike trains (the
raw campaign folders of the ANN simulator) and a folder of recorded spike
trains (`ptrain_*.mat` files of one real cohort) -- to a trained and
optimised amortised posterior: the real-arm extraction, the simulated-arm
re-detection, the label freeze, the bank plan, the simulated and the real
window banks, a Gaussian-process search over the joint DSN+NPE stack's
hyper-parameters, and the best configuration fitted as an ensemble. It never
pauses for the user; a stage that refuses stops the run with its report; the
same command re-run resumes from the first stage not done (D-081). Every
optimisation and training value comes from one configuration file (D-082).
The posteriors on the real windows and the report are a separate command
(D-083), not covered here.

Deliverables, consumed by the posteriors command and by the user:

- `<work_root>/state.json` -- the run's state (sec. 3.2), and
  `<work_root>/pipeline.log`;
- the artefacts of record of each stage (sec. 2.1), in particular
  `<tune_root>/<campaign>/SEARCH_COMPLETE.json` (the search's ledger summary)
  and `<runs_root>/<dataset>_best/BEST_FIT_COMPLETE.json` (the ensemble's
  members and its report-split NLL).

The pipeline code is data-set-agnostic (D-091): no project name, path, glob,
digest or resource may appear in `hpc/pipeline/*.py`, `stages/*.py` or
`run_pipeline.sh`; every such value is read from the project's configuration
`hpc/pipeline/projects/<tag>/<tag>.json`. A new data set is a new project
folder and the same command.

## 2. Mathematical formulation

### 2.1 The stage graph

The pipeline is a directed acyclic graph of eight stages. Each stage s has a
set of dependencies dep(s), a predicate done(s) read from the file system
(its artefact of record, with its digest where one exists), a runner, and a
verifier that reads the runner's refusal lines. The sequencer's invariant: a
stage starts only when every stage in dep(s) is done; a stage whose done(s)
holds at the start is skipped; the run stops at the first stage whose
verifier refuses.

| s | name | dep(s) | done(s): the artefact of record | runner (repository; conda env of the config) |
|---|---|---|---|---|
| S1 | `real_extraction` | -- | `<real.extract_root>/cohort_manifest.json` with a matching `.sha256` sidecar | Sbi-extractor `extractor/launch_stage_d.sh` (`envs.extract`) -- block 3 |
| S2 | `sim_redetection` | S1 | `<sim.out_root>/REEXTRACTION_RECORD.json` with a matching `.sha256` | Sbi-extractor `sim_reextract/launch_sim_reextract.sh` plan, test, array (`envs.extract`) -- block 3 |
| S3 | `label_freeze` | -- | `<paths.sbix_dir>/<freeze.out>` with a matching `.sha256` | Sbi-extractor `preflight_label_axes.py` (`envs.extract`, login node) -- block 3 |
| S4 | `bank_plan` | S1, S2, S3 | `<paths.sbix_dir>/export_windows/<dataset>/bank_plan.json` with a matching `.sha256` | Sbi-extractor `export_windows/launch_bank.sh plan` with the profile the pipeline writes (`envs.extract`, login node) -- block 2 |
| S5 | `sim_bank` | S4 | `<paths.banks_root>/<dataset>_sim/BANK_RECORD.json` with a matching `.sha256` | `launch_bank.sh test`, then `array` (the array and its gate job) -- block 2 |
| S6 | `real_bank` | S4 | `<paths.banks_root>/<dataset>_real/shard_0000.npz` and `shard_0000.json` | `launch_bank.sh real` (login node) -- block 2 |
| S7 | `search` | S5, S6 | `<paths.tune_root>/<search.campaign>/SEARCH_COMPLETE.json` | SBI `hpc/joint/stage4/npe_tune_joint.py` propose / evaluate, one job per trial through `hpc/pipeline/jobs/run_argv.pbs` (`envs.train`) -- block 7 [corrected 2026-10-16: not the tuner's array job] |
| S8 | `best_fit` | S7 | `<paths.runs_root>/<dataset>_best/BEST_FIT_COMPLETE.json` | SBI `hpc/joint/stage3/run_joint_arms.py`, one job per ensemble member (`envs.train`) -- block 7 |

"A matching `.sha256`" means: the sidecar `<file>.sha256` exists, its first
whitespace-separated token is 64 hex characters, and it equals the SHA-256
hex digest of the file's bytes. A file without its sidecar, or with a
non-matching one, is NOT done: the stage would re-run and its runner refuses
to overwrite (the Sbi-extractor tools' own rule), which stops the pipeline
with the reason -- never a silent reuse of a half-written artefact.

For the Giulia data set on 2026-10-15, S1-S6 are done (their artefacts exist
on davinci; the bank chain's last line `BANK CHAIN DONE` was reported by the
user), so the first real run starts at S7.

### 2.2 The resume rule

The state file (sec. 3.2) records, per stage, its status in {`pending`,
`running`, `done`, `failed`, `skipped`} and the scheduler job ids it
submitted. On every start the sequencer reads the state file and, for each
stage in dependency order:

1. `skipped` or `done`: left alone (done(s) is NOT re-checked, so that a
   later deletion of an artefact does not silently re-run a stage; `--status`
   reports the discrepancy instead).
2. `pending`: if done(s) holds, the stage becomes `skipped` (the artefact
   pre-exists); else, once dep(s) are all done or skipped, the runner starts
   it and it becomes `running`.
3. `running`: the stage re-attaches to its recorded jobs through the
   scheduler (sec. 3.3). Jobs still queued or running are waited for. When
   every job has finished (or is unknown to the scheduler, which a purged
   history makes indistinguishable from finished), the verifier runs: it
   reads the artefact and the refusal lines and sets `done` or `failed`. A
   job that ended without its artefact and without a refusal line is
   `killed`: the stage's policy resubmits it (at most the configured number
   of times, with the resume flag where the runner has one) or fails.
4. `failed`: the run stops at the first failed stage with its report (D-081:
   "stop on a refusal"); a re-run after the user's repair starts that stage
   again from `pending` only if the user removed it from the state file or
   passed `--retry <stage>`; otherwise it stops again with the same report.

The sequencer loop: every `poll_seconds` it ticks each stage that is not
done, skipped or failed, in dependency order; it exits 0 when every stage is
done or skipped, 3 when a stage failed, 2 on a configuration refusal, and
4 when another sequencer holds the project's lock.

### 2.3 Quantities fixed by the configuration (used by blocks 5-7)

Let n_train be the number of rows of the training split, B_sim the trial's
batch size (the searched `batch_size_npe`), P the passes over the training
split (`search.trial.passes`, `best_fit.passes`) and N_ep the epochs
(`search.trial.epochs`, `best_fit.epochs`). The steps per epoch are

    S = ceil(P / N_ep * n_train / B_sim),

so that N_ep * S * B_sim >= P * n_train, within one batch per epoch (block 6).

The grouped split by donor (D-076; `run_joint_arms.grouped_split`) takes the
fractions (f_tr, f_sel, f_rep) = `split.fractions` with f_tr + f_sel + f_rep
= 1 (D-088: (0.85, 0.10, 0.05)): of the n_d distinct donors, round(f_tr n_d)
go to training, round(f_sel n_d) to selection, the rest to the report split,
in the order of a permutation seeded by the split seed; at n_d = 2,612 and
seed 0 that is 2,220 / 261 / 131 donors (block 6).

The search (block 7): N = n_initial_points + gp_rounds * trials_per_round
trials, round 0 of n_initial_points random configurations, then gp_rounds
rounds of trials_per_round configurations proposed by expected improvement
with the constant liar, the GP refitted on every finished trial before each
round; at most `max_concurrent_trials` trial jobs at once; the objective is
the ledger's `nll`, the mean of -log q_omega(theta_i | h_psi(x_i)) over the
report split's rows i at the best selection-split epoch, in nats per row.
The elapsed clock: the search ends when gp_rounds rounds are evaluated or
when the time since its first submission, queue waits included, exceeds
`max_total_walltime_h` (D-090: 400 h); at the cap no new round is proposed
and the running trials finish.

The best fit (block 7; D-089): the finalist c* = the configuration of lowest
`nll` in the ledger is trained M = len(`best_fit.member_seeds`) times, member
m with training seed `member_seeds[m]` and the donor split of
`best_fit.split_seed` (one split for every member), giving pairs (psi_m,
omega_m); the ensemble is the uniform mixture

    q_bar(theta | x) = (1/M) sum_{m=1}^{M} q_{omega_m}(theta | h_{psi_m}(x)),
    theta in (0, 1)^p, x in R^W,

with p = `search.shape_anchors.d_theta` and W the window length of the bank.
Its report-split NLL, from the members' per-row NLLs
l_{m,i} = -log q_{omega_m}(theta_i | h_{psi_m}(x_i)) on the same report rows
i = 1..n in the same order (which the shared split guarantees), is

    L_bar = -(1/n) sum_{i=1}^{n} [ logsumexp_m( -l_{m,i} ) - log M ].

Early stopping (blocks 6-7; D-087): with L_sel(e) the selection-split NLL
after epoch e, epoch e improves iff L_sel(e) < min_{e' < e} L_sel(e') - 1e-6;
e_best is the last improving epoch; the loop stops at the first e with
e - e_best >= `patience_epochs` and restores e_best's weights. The search
trials have no patience (every epoch runs; the best epoch is restored); the
best fit's patience is `best_fit.patience_epochs` (10).

### Notation

| Symbol | Meaning | Type / shape | Units |
|---|---|---|---|
| s, dep(s), done(s) | a stage, its dependencies, its done-predicate | -- | -- |
| n_train, n_d | training rows; distinct donors in the bank | int | rows; donors |
| B_sim, P, N_ep, S | batch size, passes, epochs, steps per epoch | int, float, int, int | rows; passes; epochs; steps |
| (f_tr, f_sel, f_rep) | split fractions by donor | floats summing to 1 | -- |
| theta, x | a parameter vector in the unit box; a window | (p,), (W,) float32 | -- ; Hz (IFR) |
| q_omega(theta \| h_psi(x)) | the flow's density at theta given the encoder's summary of x | density on (0, 1)^p | -- |
| l_{m,i}, L_bar | member m's NLL on report row i; the ensemble's mean NLL | float64 | nats per row |
| L_sel(e), e_best | selection-split NLL after epoch e; the best epoch | float; int | nats per row; epochs |

## 3. Data contracts

### 3.1 The project configuration `hpc/pipeline/projects/<tag>/<tag>.json`

One JSON object per data set. Every key below is required; a missing key, an
unknown key, or a value of the wrong type or range is a refusal before
anything runs (exit 2, every problem listed). A key starting with `_`
documents the key after it and is ignored. Paths are absolute (no `~`).
Config v4 for Giulia is the committed file; the values are the user's or
Claude's as the log records (D-085..D-090).

| Key | Type | Constraint |
|---|---|---|
| `config_version` | int | 8 (v8, 2026-10-18: the `driver` block, D-098; v7, 2026-10-18: `search.resubmit_node_faults` and `best_fit.resubmit_node_faults`; v6, 2026-10-18: `bank.verify` and `bank.max_rerun_rounds`; v5, 2026-10-16: the two queue keys below) |
| `dataset` | str | `[a-z0-9_]+`; the `<tag>` of the folder and of the artefact names |
| `paths.{sbix_dir, sbi_dir, work_root, banks_root, runs_root, tune_root}` | str | absolute paths |
| `envs.{extract, train}` | str | conda environment names |
| `real.{cohort_config, cohort_tag, extract_root}` | str | `cohort_config` an absolute path |
| `sim.{sim_main, campaign_glob, out_root, tools, plan_args, test_walltime, walltime, queue}` | str | walltimes `hh:mm:ss` |
| `sim.{ncpus, concurrency}` | int | >= 1 |
| `freeze.{out, campaigns, require_conn_rule}` | str | `out` relative to `paths.sbix_dir` |
| `freeze.exclude` | object str -> str | axis name -> reason |
| `bank.{on_axis, queue, walltime}` | str | `walltime` `hh:mm:ss` or `auto` |
| `bank.on_declared_box` | [float, float] | lo < hi |
| `bank.topology_axes` | [str] | non-empty |
| `bank.{expected_p, n_e, sample_seed, bootstrap_resamples, bootstrap_seed, bytes_sample_rows, rows_per_shard, concurrency}` | int | >= 1 (seeds >= 0) |
| `bank.{t_win_s, floor_hz, floor_rtol}` | float | > 0 |
| `bank.verify.{ncpus}`, `bank.verify.{mem, walltime}` | int; str | >= 1; PBS memory, `hh:mm:ss` -- the verification job (v6) |
| `bank.max_rerun_rounds` | int | >= 0 -- reruns of the array after a quarantine (v6) |
| `split.fractions` | [float, float, float] | each in (0, 1), sum 1 within 1e-9 |
| `search.{campaign, arm}` | str | `campaign` in {S-A1, S-A2, S-A5, S-A25}; `arm` in the runner's arms |
| `search.shape_anchors.{p, d_theta, embedding_dim}` | int | >= 1 |
| `search.space_overrides` | object | axis -> [lo, hi] (numeric axes) or [levels] (categorical); axes must exist in the joint space (block 7) |
| `search.flow.{conditioner_hidden_layers, num_bins}` | int | >= 1 |
| `search.{n_initial_points, gp_rounds, trials_per_round, max_concurrent_trials}` | int | >= 1 |
| `search.{gp_seed, train_seed}` | int | >= 0 |
| `search.gp_noise_sigma` | float or null | > 0 when given |
| `search.trial.{passes}` | float | > 0 |
| `search.trial.{epochs, ncpus, max_resubmits}` | int | >= 1; `max_resubmits` >= 0 |
| `search.trial.{mem, walltime}` | str | PBS memory (`<n>gb`), `hh:mm:ss` |
| `search.trial.queue` | str | the trial jobs' and the probes' queue (v5) |
| `search.budget_gate.{probe_steps, probe_val_rows}` | int | >= 1 |
| `search.budget_gate.safety_factor` | float | >= 1 |
| `search.max_total_walltime_h` | float or null | > 0; null = no cap |
| `search.walltime_reading` | str | `elapsed` |
| `search.resubmit_killed` | bool | -- |
| `search.resubmit_node_faults` | bool | a trial that failed of a node fault is resubmitted (D-096; v7) |
| `best_fit.split_seed` | int | >= 0 |
| `best_fit.member_seeds` | [int] | non-empty, distinct |
| `best_fit.passes` | float | > 0 |
| `best_fit.{epochs, patience_epochs, n_posterior_draws, ncpus, max_resubmits}` | int | >= 1; `max_resubmits` >= 0 |
| `best_fit.{mem, walltime}` | str | as for the trial |
| `best_fit.queue` | str | the member jobs' queue (v5) |
| `best_fit.resubmit_node_faults` | bool | a member that failed of a node fault is resubmitted (D-096 applied to the members; v7) |
| `loader.buffer_shards` | int | >= 1 |
| `driver.{queue, mem, walltime}` | str | the sequencer's PBS jobs (D-098; v8): queue, PBS memory, `hh:mm:ss` |
| `driver.{ncpus, poll_seconds}` | int | >= 1: the jobs' cores; seconds between two ticks |

Keys whose code is not built yet (the NEEDS-CODE keys) are refused by
`pipeline.py run` with the block that builds them named, until that block
lands (sec. 5, the capability table of `config.py`); a key is refused only
when a stage that uses it is still to run (a finished search does not block
the best fit on a search-only key). `--dry-run` prints the same refusals
after the stage table, so the user sees both.

### 3.2 The state file `<work_root>/state.json`

Written atomically (temporary file, then `os.replace`), ASCII JSON, sorted
keys. Fields: `schema` ("pipeline_state/1"); `dataset`; `config_path`;
`config_sha256` (the configuration's bytes; a run under a configuration of
another digest is refused unless `--accept-config-change` is given, and the
change is logged); `started_utc`, `updated_utc` (ISO 8601, UTC); `pid`,
`host` of the sequencer that last wrote; `stages`: an object stage name ->
{`status`, `started_utc`, `finished_utc`, `jobs`: [{`id`, `role`,
`submitted_utc`, `cmd`, `log`}], `resubmits`, `note`, `sub`: a stage-private
object for its own state (the search's rounds and clock, block 7)}.

### 3.3 The scheduler adapter (`pbs.py`)

`submit(argv) -> job_id`: runs `argv` (a complete `qsub` command line), the
job id is the first whitespace-separated token of its stdout (`1829506.
davinci` or an array's `1829506[].davinci`); a non-zero exit is a
`SubmitError` carrying stdout and stderr. `status(job_id) -> JobStatus`:
runs `qstat -x -f <job_id>` and parses the PBS Pro record: `job_state` (one
of Q, R, H, B, E, F, X, S, T, W, M; anything else or an unknown job gives
`unknown`), `Exit_status` (int, when present), `resources_used.walltime`
(`hh:mm:ss`, as seconds), `array_state_count` (when present). Finished means
`job_state` in {F, X} or `unknown`. The adapter retries a `qstat` that fails
to run (the server unreachable) three times, 20 s apart, before raising; it
never retries `qsub`. [amended 2026-10-18, D-098] `select(name) ->
[job_id]`: `qselect -u <user> -N <name>`, the user's jobs of that name that
have not finished (PBS Pro lists no finished job without `-x`), in
qselect's order -- the sequencer reads each one's state with `status` and
drops a finished or unknown one all the same, so a server that lists
finished jobs cannot block a submission; `delete(job_id) -> bool`: `qdel`, False for a job already
finished. Both raise `SchedulerError` when the command cannot run.
`job_seq(id)` is the part before the first dot (`1829506` of `1829506.
davinci-mgt01`, `1829506[]` of an array), `job_number(id)` its integer (-1
without one), `same_job(a, b)` compares the sequence parts: qsub, qselect
and PBS_JOBID may spell the server part differently, so job ids are never
compared as whole strings. The sequencer's jobs call all four commands from
compute nodes; `tools/probe_compute_qsub.pbs` checks that davinci allows it
(qsub with `-W depend=afterany` and `-h`, qstat -x -f, qselect -u -N, qdel).

The fake PBS of `tests/fake_pbs/` provides the same commands for the
tests: `qsub` records the job and runs its script after `FAKE_PBS_DELAY`
seconds in a detached process (array elements in parallel, `depend=afterok`
and, for a plain job, `depend=afterany` honoured; `-h` holds it), `qstat -x
-f` prints a PBS-like record with `Output_Path` (wrapped as davinci's),
`qselect [-N] [-s] [-u] [-x]` lists (no finished job without `-x`, unless
`FAKE_PBS_QSELECT_HISTORY=1`; `FAKE_PBS_QSELECT_SUFFIX` appends a server
suffix to the ids it prints), `qdel` ends a waiting job unrun and kills a
running one (Exit_status 271). A plain job ends when its script ends: what
is left of its process group is killed before the job is recorded finished,
as PBS Pro does. The fake hands its own environment to its jobs (PBS passes
only `-v`): the tests' switches reach the jobs that way. It is a test
fixture, not part of the pipeline.

## 4. Global conventions

- Units used internally: seconds for wall times (printed as `hh:mm:ss` where
  PBS wants them), nats per row for NLLs, Hz for rates.
- Floating-point precision: float64 for every number the pipeline computes
  (the ensemble NLL); the trainers' own precision is theirs.
- Randomness: none in the pipeline itself; the seeds it passes are the
  configuration's (`gp_seed`, `train_seed`, `split_seed`, `member_seeds`).
- Python: the stdlib only in the sequencer's process (`hpc/pipeline/*.py`,
  `stages/`; json, os, subprocess, time, hashlib, argparse); syntax
  compatible with Python 3.6 (no f-strings, no walrus, no postponed
  annotations), since the sequencer runs on the login node under the
  project's `envs.extract` environment; pure ASCII, LF. [corrected
  2026-10-18 by D-098: the sequencer runs under `envs.extract` in its own
  PBS jobs on compute nodes; only the short commands (`submit`, the dry run,
  `status`, `stop`, `retire`) run on whichever login node the user is on.] The tools of
  `hpc/pipeline/tools/` are run by the stages inside `envs.train` (numpy,
  scipy, torch) and are not held to the stdlib rule.
- Binding decisions (their content, so the tester needs no log):
  - D-039: the training is arm `A1` of the joint stack (encoder and flow on
    simulated windows alone); the search is campaign `S-A1`.
  - D-062: Claude may push to this repository (branch `sci/pipeline`);
    Sbi-extractor and Astro-Neuron-Network only through the user's laptop,
    so the pipeline drives Sbi-extractor's committed tools without changing
    them.
  - D-074, D-075, D-076, D-078: the bank -- all record tasks; the on-axis's
    declared box; a row kept iff its window's mean rate >= `bank.floor_hz`;
    donor = one network; the prior is the declared box (the unit box for
    the runner), no mixture correction.
  - D-081: one command, never pausing, stop on a refusal, resumable, a
    background process on the login node (`nohup`). [corrected 2026-10-18 by
    D-098: no process on a login node; the sequencer is a chain of PBS jobs
    on compute nodes.]
  - D-094, D-095: the simulated bank verified by a job after every round,
    bad tasks moved to a quarantine folder and rerun (block 2).
  - D-096, D-097: a trial (and a member) that dies of a node fault is
    resubmitted, its record set aside; a checkpoint whose zip CRCs fail is
    moved aside (block 7).
  - D-098: nothing in the pipeline or in the lines given to the user may
    depend on a particular login node -- the user's logins land on any
    login node: no process left running on a login node, no `ps`, `kill`
    or `nohup` there, no step that must return to the node it started on.
    Everything runs as PBS jobs on compute nodes; the user's commands only
    submit, read and write files on the shared file system, from any login
    node (block 1, amended 2026-10-18 (c)).
  - D-082: every search and training value from the configuration.
  - D-083: the posteriors and the report are not started by this command.
  - D-084: the runner streams the bank through a buffer of
    `loader.buffer_shards` shards; no row dropped (block 5).
  - D-085..D-088: the configuration's values (sec. 3.1; the file).
  - D-086: `head_pool_ops` is a searched axis, levels {0, 1} (block 7).
  - D-087: the early-stopping margin is 1e-6 nats per row (sec. 2.3).
  - D-089: the finalist is an ensemble of 10 members, the uniform mixture,
    on one shared split (sec. 2.3).
  - D-090: `max_total_walltime_h` = 400, elapsed.
  - D-091: no project value in the code; the project folder in the repo.
  - D-092: the bank tools' defaults (Sbi-extractor `specs/SPEC.md` sec. 8
    Q6-Q8, Q10-Q14, the empty-topology rule) stand; the pipeline's bank
    stage changes none of them.
- Every new flag of an existing runner or tuner (blocks 4-7) defaults to the
  behaviour at `9fcbc88`, so a call without the flag is unchanged
  (regression test per flag).
- Pass conditions are lines that must APPEAR (a `DONE`, an `OK`, a digest
  line), never a warning that must be absent.

## 5. Blocks

### Block 1: the sequencer (driver skeleton)

- Module: `hpc/pipeline/pipeline.py` (CLI and the loop), `config.py` (the
  schema of sec. 3.1 and the capability table), `state.py` (sec. 3.2),
  `pbs.py` (sec. 3.3), `stages/__init__.py` (the `Stage` interface, the
  eight stage names with their dependencies and done-predicates, the
  registry), `run_pipeline.sh` (the one command), `projects/giulia/giulia.
  json` (config v4; v5 since block 7), `README.md`.
- Public API: `config.load_config(path) -> Config` (raises `ConfigError`
  listing every problem); `config.unhonoured_keys(cfg) -> [(key, block,
  what)]`; `state.State.load(work_root)`, `.save()`, `.stage(name)`;
  `pbs.submit(argv, cwd=None) -> str`, `pbs.status(job_id) -> JobStatus`;
  `stages.Stage` with `name`, `deps`, `is_done(cfg) -> (bool, str)`,
  `start(cfg, st, pbs) -> None` (records jobs in `st`), `poll(cfg, st, pbs)
  -> str` (`running` | `finished`), `verify(cfg, st) -> (bool, str)`,
  `on_killed(cfg, st, pbs) -> bool` (resubmitted or not), `dry_run(cfg,
  st) -> [str]` (the submission lines it would print); `stages.default_
  graph(cfg) -> [Stage]`; `pipeline.main(argv, graph=None) -> int`.
- Inputs: the configuration file; `<work_root>/state.json` if present; the
  scheduler's answers.
- Outputs: `<work_root>/state.json`, `<work_root>/pipeline.log`,
  `<work_root>/pipeline.lock` (the pid; removed at exit); the stage table on
  stdout; exit codes 0 / 2 / 3 / 4 (sec. 2.2).
- Parameters: `--poll-seconds` (default 60; the tests use 0.2),
  `--dry-run`, `--status`, `--retry <stage>`, `--accept-config-change`,
  `--project <tag>` or `--config <path>`.
- Library calls relied on: `json`, `subprocess.run`, `os.replace`,
  `hashlib.sha256`, `fcntl`-free locking (the pid file plus `os.kill(pid,
  0)`).
- Custom code: the sequencer loop and the resume rule (sec. 2.2) -- no
  library schedules PBS jobs from a state file; checked: `subprocess`,
  `concurrent.futures` (not a workflow engine); a workflow engine (Snakemake,
  Nextflow) is not installed on davinci's login node and would be a new
  dependency for a stdlib-sized problem.
- Test oracles (`tests/test_smoke_pipeline.py`, the fake PBS):
  - config: the committed `giulia.json` loads with no schema problem; a
    missing key, an unknown key, a wrong type, fractions not summing to 1,
    `walltime_reading` != `elapsed` each produce one named problem; the
    NEEDS-CODE keys are listed with their block numbers (exact list).
  - state: save then load is the identity (JSON equality); the file is
    sorted-key ASCII; a crash between the temporary file and the rename
    leaves the old file intact (simulated by writing the temporary file
    only).
  - pbs: the parser reproduces `job_state`, `Exit_status` and
    `resources_used.walltime` from a recorded `qstat -x -f` text (a
    davinci-format sample in the test); an unknown job gives `unknown`.
  - sequencer, on a synthetic three-stage graph A -> B -> C under the fake
    PBS: `--dry-run` prints A's and B's qsub lines and submits nothing; a
    full run ends with A, B done and C `skipped` when C's artefact
    pre-exists; a run killed (SIGKILL) while A's job runs and re-run
    resumes: A is verified from its artefact without a second submission
    (the fake PBS's job count is 1), B then runs; a stage whose verifier
    refuses stops the run with exit 3 and the `STOP:` line; a second
    sequencer on the same work root exits 4; a job that ends without an
    artefact is resubmitted once (`max_resubmits` 1) and then the stage
    fails.
  - agnostic: no code file (`*.py`, `*.sh`) under `hpc/pipeline/` outside
    `projects/` and `tests/` contains the strings `giulia`, `ldellamea`,
    `davinci-1/home`, `cadex`, `hhgap` (a grep in the test; the tests name
    the Giulia file as their known-answer fixture, the README gives it as
    the example of the command).
- Data flow: the stages of blocks 2, 3 and 7 plug into the registry; the
  configuration feeds every block.
- Known window: a sequencer killed between `qsub` returning and the state
  file's write loses that job id and submits it again on resume; the
  runners' refusal to overwrite their artefacts turns the duplicate into a
  stop, not a corruption. Accepted for block 1; a submission journal written
  before `qsub` would close it (block 8 if it ever bites).
- Amended 2026-10-16 (the first davinci run): the dry run reports a stage
  the state file records as failed (`<stage>: FAILED in the state file
  (<note>); the run would stop here -- add --retry <stage>`), ends with a
  `STOP:` line and exits 3; with `--retry <stage>` it shows the run that
  would follow and exits 0 (the reset is not saved by a dry run). Test:
  `test_edge_cases` (a).
- Amended 2026-10-18 (restarting the sequencer under a running search, to
  load new code): (a) `--retry <stage>` forgets the stage's jobs
  (`reset_stage`), so on a stage whose jobs still run it would submit
  their work a second time; it is refused -- exit 2, `REFUSE: --retry
  <stage>: the stage has <n> job(s) still queued or running (<ids>); a
  retry forgets them and would submit their work again -- let them end
  (or qdel them) first` -- when a job recorded for the stage is not
  finished by the scheduler's answer (an unknown job counts as finished);
  the dry run prints the same line and exits 2. (b) A requested stop:
  `pipeline.py stop --config <file>` (`run_pipeline.sh ... --stop`) writes
  `<work_root>/STOP_REQUEST`; SIGTERM and SIGINT set the same request in
  the running sequencer. The request is honoured at a tick boundary only
  -- after every running stage's poll and the state's save; the sleep
  between ticks is cut short by it -- so a stop never falls between a
  submission and its record; the sequencer removes the file, logs
  `stopped on request (<how>); resumable: <n> job(s) still queued or
  running keep running`, releases the lock and exits 0. The jobs are
  untouched; a re-run re-attaches to them. Before this amendment a `kill`
  could land inside a poll (the known window above). Tests:
  `tests/test_smoke_block7b.py` `test_edge_cases` (its stop and retry part;
  `stop_and_retry` when run alone).
- Amended 2026-10-18 (c) (D-098; the user, 16:33: "I always log in from
  a different node so do not use this approach ... only submissions to
  compute nodes as usual"): the sequencer runs as a chain of PBS jobs on
  compute nodes; nothing runs or is managed on a login node, and every
  command the user is given works from any login node.
  - `pipeline.py submit --config <file> [--retry S] [--accept-config-change]
    [--break-lock]` (`run_pipeline.sh --project <tag>`, from any login
    node): the refusals a run would make, checked here with the run's own
    code (`_prepare_state`: a configuration change without the flag, a
    `--retry` naming no stage or a stage whose jobs still run; then the
    NEEDS-CODE keys and a stage not built: exit 2), then `REFUSE: the
    sequencer is already queued or running: <ids>` (exit 4) when a job of
    the project's sequencer (`pl_<tag>`) has not finished, else one `qsub`
    and `SUBMITTED the sequencer: job <id> (pl_<tag>) ...`, exit 0. The job:
    `jobs/run_argv.pbs` in `envs.extract` with the resources of `driver`,
    the argv file `<work_root>/driver/driver.<k>.argv` (`pipeline.py drive
    --config <file>` and the one-time flags), the log `<work_root>/driver/
    driver.<k>.log`. A stop request left behind is removed, and the count of
    unclean ends below is reset.
  - `pipeline.py drive` (the sequencer's job; refused, exit 2, without
    PBS_JOBID): (1) one driver at a time -- when another `pl_<tag>` job runs
    (R, E, B) with a smaller job number, `driver <me> on <host>: another
    driver <id> is running; this one exits` (exit 0, no successor); the
    waiting ones are deleted (`deleted the waiting driver`); a younger one
    that runs exits by the same rule, so of two drivers released together
    the older stays; (2) a lock left by a finished driver job means that
    job was killed (`the previous driver <id> ended without releasing the
    lock ...`): an unclean end; (3) the count of unclean ends in a row
    (`<work_root>/driver/unclean_ends.json`) at 3 (DRIVER_MAX_UNCLEAN) ends
    the chain at once (`the last 3 sequencer jobs ended uncleanly before
    completing a tick ...`, exit 3), so a sequencer that crashes cannot loop;
    (4) the successor, submitted at once: the same line with `-W
    depend=afterany:<me>` and without the one-time flags, so a driver that
    dies anywhere -- its node, its walltime -- is followed; a failed
    submission is retried at the budget; (5) the loop of `run` with the
    budget walltime - min(max(900 s, 5 x poll_seconds), walltime / 4); its
    first tick boundary resets the unclean count; (6) the end: the budget
    -> `driver <me>: time budget reached; the successor <id> continues`
    (exit 0); a signal (PBS ending the job: walltime, qdel; honoured at a
    tick boundary) -> an unclean end, `ended by signal <n> ...; the
    successor <id> continues` (exit 0); an exception -> its traceback in the
    job's log, an unclean end, the successor continues below the limit
    (`unclean end <n> of at most 3 in a row`), else is deleted (exit 1); any
    other end (PIPELINE DONE, a STOP, a REFUSE, a requested stop, another
    sequencer's lock) -> the successor deleted, `driver <me>: the chain ends
    (<why>, exit <n>); its successor <id> was deleted`.
  - `run` takes the lock FIRST and reads the state after it (a sequencer
    that wrote the state while this one waited for the lock is not
    overwritten), then the configuration change and the `--retry` resets.
  - The lock names its holder: `job:<PBS_JOBID> <host> <utc>`, `pid:<pid>
    <host> <utc>` (a foreground run: the tests), or `<pid> <host> <utc>`
    (written by the code before this amendment). Stale: a job holder the
    scheduler reports finished or unknown; a pid holder on this host not
    alive (pids 0 and -1 never are); a pid holder on another host that has
    not written `state.json` for `PIPELINE_HEARTBEAT_QUIET_S` (default 600)
    seconds, watched by this sequencer's clock (two mtime readings, so the
    hosts' clocks need not agree) -- else `another sequencer (...) holds` /
    `is still writing the state` and exit 4. Where there is no lock it is
    created with O_EXCL; a stale one is replaced and read back 1 s later,
    so of two sequencers that replaced it at once the later keeps it and
    the other yields (exit 4). `--break-lock` replaces any lock at once. A
    sequencer removes only its own lock.
  - A job about to be submitted is ADOPTED when a job of the same name
    (`-N`) AND the same output file (`-o`, against qstat's `Output_Path`
    without its host part) is queued, held or running (`select`, `status`):
    recorded, not submitted again, `<stage>: adopted <role> -> <id> (a job
    of that name and output file was queued or running but not recorded)`
    -- the known window above, for a driver killed between a `qsub` and the
    state's save. A name alone is not enough: `bank_verify` and `gate_b<k>`
    are the same in every project, the output files are not. A command line
    without `-N` or `-o`, or a scheduler that does not answer: submitted as
    before. `JobStatus.output_path` is that path (None when qstat gives
    none).
  - The state file: each write goes to `state.json.<pid>.tmp`, the current
    file is kept as `state.json.prev` (a hard link, then a rename: no
    moment without `state.json`), then the temporary file is renamed; a
    `state.json` that does not parse is read from `state.json.prev` (`the
    state file ... does not parse ...; the previous state ... is used`);
    neither -> `ValueError: no state file parses`. A MISSING `state.json` is
    no state, a fresh start, as before (a save never leaves a moment
    without it, so only a removal makes it go; `.prev` is then not read).
    The state records the writer's job (`job`).
  - `pipeline.py stop [--wait S]` (`--stop`): with no `pl_<tag>` job and no
    pid lock that may be live (a pid on another host counts as live), `stop:
    no sequencer is running` (a stop request is removed); else
    `STOP_REQUEST` is written FIRST, then the waiting `pl_<tag>` jobs are
    deleted, so a waiting job released at that moment stops too: the
    running sequencer stops at its next tick boundary and deletes its
    successor; a driver job does not remove a request at
    its start. `--wait S` waits for no running driver and no lock (`stop:
    the sequencer has stopped`, exit 0; else exit 4).
  - `pipeline.py retire [--wait S]` (`--retire-login-sequencer`; the
    transition, once): for a lock in the old format (a bare pid), creates
    the directory `<work_root>/state.json.tmp` -- the old code writes its
    state through that name after every tick, so its next save raises and
    it leaves through its `finally`, which removes its lock -- and waits up
    to S s (default 300) for the lock to go: `RETIRE: OLD SEQUENCER
    STOPPED`; else `RETIRE: the lock is still there after S s ...` (the old
    sequencer was already gone, or is in a step longer than S; the first
    driver takes the lock after the heartbeat's quiet time). The directory
    stays: this code never uses that name. A temporary FILE of that name
    still there after 10 s (a crash's leftover) is moved aside to
    `state.json.tmp.stale-<utc>`, not deleted. A job or `pid:` lock:
    `nothing to retire; --stop stops it`. No lock: `nothing to retire`.
    State written during the wait: `RETIRE: UNEXPECTED ...`, exit 3.
  - `pipeline.py status`: the state's last write (age by this node's clock,
    writer), the lock, every `pl_<tag>` job not finished with its state,
    the stage table.
  - The dry run prints `the sequencer: PBS jobs named pl_<tag> on queue
    <q> (<n> cores, <mem>, walltime <w>, a tick every <p> s), one at a time,
    each submitting its successor (afterany); nothing runs on a login node
    (D-098)`.
  - Config v8: `driver.{queue, ncpus, mem, walltime, poll_seconds}`;
    Giulia: `cpu`, 4, `16gb`, `24:00:00`, 60 `[proposed]` (a budget of
    85,500 s per driver job).
  - `pipeline.main(argv, graph=None, entry=None)`: `entry` is the command
    line before the verb in the drivers' argv files (default: the
    environment's python and this `pipeline.py`); the tests run their
    synthetic graph's `drive` through it.
  - Test oracles (`tests/test_smoke_block1b.py`, the fake PBS with
    `FAKE_PBS_QSELECT_SUFFIX=.cluster` and the walltime enforced, the
    synthetic graph A -> B -> C): known answer (Giulia v8: the driver block,
    `pl_giulia`, 85,500 s, the dry run's line); reference (job ids, the
    budget for four walltimes, the driver's qsub line and argv files, the
    three lock formats, all hand-written); invariants (the state's own
    temporary name beside a `state.json.tmp` directory, `.prev`, the
    fallback; O_EXCL; a live job's lock refused, a finished one replaced; a
    remote pid lock refused while the state is written, replaced after the
    quiet time; release never removes another's lock; no `nohup` in the
    code); contract (the chain hands over by budget, one driver at a time,
    every stage job once, the last successor deleted unrun, a second
    submission refused); determinism (a driver killed by qdel: the
    successor finishes, nothing twice; a queued job of the same name and
    output file adopted, one of the same name and another output file
    not); edge cases (drive without PBS_JOBID; stop from any node, then
    submit again, qselect listing the finished jobs too; `--retry` of a
    stage whose job runs and a configuration change refused at submit, the
    flag in the first driver's argv only; a
    crash loop ends after 3 drivers, a submission resets the count; two
    drivers released together, the older stays; SIGTERM keeps the chain;
    retire against a stand-in of the old sequencer, then the takeover of a
    stale old-format lock). Mutations (each makes its check fail):
    successor, unclean_limit, lock_check, prev_fallback, heartbeat,
    tiebreak, retry_at_submit, adopt, adopt_output, finished_filter.
  - The transition from the old code, rehearsed on the code of `f1a2b7a`
    itself (not in the repository's tests): see the build handoff.
  - Confirmed: the direction, yes 2026-10-18 (D-098: "only submissions to
    compute nodes as usual"); the driver block's values, the limit of 3
    unclean ends, the 600 s quiet time and the budget's margin rule are
    Claude's `[proposed]`.
- Confirmed: yes 2026-10-15 (the user's answers to the build handoff's Q1-Q7
  and the follow-up; D-090..D-093)
- Status: smoke-tested 2026-10-15 (6 pass, 1 skip: convergence, not a
  discretisation; nothing skipped for want of davinci -- the scheduler is
  the fake PBS, which is what the sandbox can test)
  [2026-10-18, amendment (c): `test_smoke_block1b.py` 6 pass, 1 skip
  (convergence); ten mutations each caught; blocks 1, 2, 3, 7, 7b, 8 pass;
  the transition rehearsed on the code of `f1a2b7a` itself, its sequencer
  alive and already dead (the SIGBUS trial healed, the running trial
  re-attached, no other job twice). Skipped for want of davinci: PBS Pro
  answering qsub, qstat, qselect and qdel from a compute node (the probe
  `tools/probe_compute_qsub.pbs` is the user's to run first), `Output_Path`
  as qstat prints it for a job submitted from a compute node, the real
  dependency hold (H) and kill delay.]

### Block 2: the bank stages S4-S6 (`stages/bank.py`)

- Module: `hpc/pipeline/stages/bank.py` (`BankPlanStage`, `SimBankStage`,
  `RealBankStage`), through Sbi-extractor's committed `export_windows/
  launch_bank.sh` (D-062: driven, not changed; D-092: its rules stand);
  `pipeline.Submitter.record` (a job a launcher submitted, recorded like
  one of ours); `envs.run_in_env(..., env=)`; `tests/standin.py` (the
  stand-in davinci: a copy of the Sbi-extractor tools, a fake conda, the
  bank fixture's synthetic data set).
- Public API: `BankPlanStage.write_profile(cfg) -> dict` (the bank profile,
  `bank_plan_config.BankPlanConfig`'s keys, written to `<work_root>/bank_
  profile_<tag>.json`: the configuration's `bank` block, the paths from
  `paths` and `sim`, the record / freeze / frozen-manifest paths with the
  16-hex prefixes of their sha256 -- the frozen copy `<sbix_dir>/artifacts/
  cohort_manifest_<cohort_tag>/cohort_manifest.json` when it exists, else
  S1's artefact); the launcher's variables PROFILE, ENV_NAME = `envs.
  extract`, QUEUE = `bank.queue`, CONCURRENCY = `bank.concurrency`,
  WALLTIME = `bank.walltime` (the array), TEST_INDEX 0, SBI_HPC_DIR =
  `<sbi_dir>/hpc`.
- Rules: S4 runs `plan` on the login node and needs the pass line `BANK
  PLAN OK` (the tool writes the plan and its .sha256); S5 is the chain line
  BC as phases -- `test` (skipped when `<bank_root>/tasks/task_0000.json`
  carries this plan's sha256 and a `finished` time), passed when that
  marker appears after the job (its log line `TASK <i> OK` is reported
  with it), then `array` (the array and its gate, recorded from the
  launcher's `  job   :` and `  gate  :` lines), passed when `BANK_RECORD.
  json` + a matching .sha256 exist, else the gate log's `[gate] FAIL` /
  `REFUSED` lines stop the run; S6 runs `real` on the login node and needs
  `REAL BANK OK`. [amended 2026-10-16, before the first bank run on
  davinci: success is read from the files the jobs write, not from their
  PBS logs (delivered at the job's end, possibly with site epilogue lines
  after the pass line); the whole log is searched for the failure text;
  when a job has finished with neither its file nor its script's own end
  line `[job] <name>_exit=<rc>` in the log, the stage waits up to
  `PIPELINE_LOG_GRACE_S` seconds (default 300, kept in the stage's state)
  before deciding. The same rule holds for S1's aggregation and S2's gate.]
  A launcher's `ABORT:` / `REFUSED:` lines are the refusal text. The
  committed Giulia configuration reproduces the committed profile's
  decided values (the test checks every key).
- Test oracles (`tests/test_smoke_block2.py`, the stand-in, the fake PBS):
  - known answer: the profile loads through `bank_plan_config` with the
    fixture's digests and the configuration's values; the Giulia
    configuration against `profiles/giulia.json` key by key.
  - reference: the Giulia dry run prints the three stages' lines.
  - invariants: the `job` / `gate` lines are parsed; a refusal is read
    from ABORT / REFUSED lines.
  - contract: S4 -> S5 (the test task, the array of 3 and its gate) -> S6
    on the fixture: `BANK PLAN OK`, `TASK 0 OK`, the gate's `wrote`, `REAL
    BANK OK`, the artefacts with their sidecars, 3 jobs; a re-run on the
    state does nothing, on a fresh state skips all eight stages.
  - determinism: the sequencer stopped while the test task is queued (the
    fake PBS's 4 s start delay) and re-run re-attaches: 3 jobs in all.
  - edge: a task's detections removed before the array -> the gate
    (ordering only after an array in the fake PBS too, D-006) refuses, the
    run stops with the `[gate] FAIL` line and no record; a wrong
    `expected_p` -> the plan's REFUSED stops the run before any job.
  - dropped gate (the fake PBS's `FAKE_PBS_ARRAY_DEPEND=gating`: the
    afterok dependent of an array with a failed element ends unrun, with
    no Exit_status and no log; `FAKE_PBS_DROP_DEPENDENTS=1` ends it unrun
    whatever the elements did): (i) a task's detections removed after the
    plan -> its element fails, the gate never runs, the stage logs "never
    ran" and runs the gate alone, which refuses with `[gate] FAIL` -> the
    run stops with those lines, 4 jobs, no record; (ii) every element fine,
    the gate dropped, a stale `bank_gate.log` holding `REFUSED` from an
    earlier run -> the stale log is not read, the gate alone writes the
    record, the run ends `PIPELINE DONE`, 4 jobs.
  - log timing (in process, a fake scheduler): a test log saying `TASK 0
    OK` without the marker waits; the marker with 15 epilogue lines after
    the pass line passes; a finished gate with neither record nor log
    waits, then fails at once when the log arrives with `[gate] FAIL` and
    its end line; a record with no log passes; the grace run out with
    neither record nor fresh log runs the gate alone [amended 2026-10-16;
    was: fails with "no log"], and that gate also without a log fails with
    "the gate did not run"; a stale log (its stat unchanged since the
    submission) counts as no log; a gate alone waits for the array's end.
- Defect found on davinci (2026-10-16 10:35, the first run): with the plan
  of record (S4 skipped) and a fresh work root, no stage had written the
  profile and `launch_bank.sh test` aborted on the missing file. Fixed the
  same morning: every bank stage (re)writes the profile from the
  configuration before its first launcher call (`_BankBase.ensure_profile`,
  idempotent; a changed file is logged with the keys that changed), and the
  dry run reads a dependency as satisfied when its artefact exists. The
  contract test now repeats the run with the work root and the banks
  removed and the plan kept (S4 skipped, S5 and S6 run).
- Defect found on davinci (2026-10-16 12:29, the second run): the array
  `1845588[]` (69 elements) ended and its gate `1845589` (afterok on the
  array) never ran -- no log, no record, no exit status in the stage's
  `qstat -x -f` -- and the stage stopped with "the gate refused (...): no
  log". D-006's probe (an element exiting 1; the dependent ran) does not
  cover whatever ended this gate unrun (the diagnosis on davinci is
  pending). [diagnosed 2026-10-18 from the user's read-only check of the
  PBS records, the task logs and the markers: 66 elements exited 0; three
  failed, all on `dvnode012` within about seven minutes -- elements 45
  and 46 exited 135 (a SIGBUS, `Bus error (core dumped)`, after `[Errno
  5] Input/output error` reading the simulation tree, and a DSN tree
  reported missing), element 63 never started (Exit_status -2,
  `resources_used.walltime` 00:00:00, its log only `Could not chdir to
  home directory`): the node lost the home file system. The array's
  record reads Exit_status 0 with the comment `Job Array Began ... and
  failed`; the gate's has no start time and no Exit_status. Reading, not
  verified in the server's log: an element exiting with a code (1 in the
  probe, 135 here) leaves the dependent to run; the element that failed
  to start (-2) is what ended it unrun.] Rule since [amended 2026-10-16]: (a) the gate's log is THIS
  gate's only when it is fresh: absent at the gate's submission and
  present now, or present then with a (size, mtime) that has changed since
  (the stat is taken before the launcher call and kept in the stage's
  state; a state without it counts any log as fresh); a stale log is read
  neither for the pass nor for the failure text. (b) A finished gate with
  no record and no fresh log, once its grace period has run out, NEVER
  RAN: the stage logs "the gate <id> never ran", waits until the array
  has finished, and runs `launch_bank.sh gate` (the gate alone, no
  dependency) once, recorded from its `  gate  :` line, with its own grace
  period. (c) If the gate alone does not run either, the stage fails with
  "the gate did not run" (both job ids, the log path). (d) A gate that ran
  is judged as before: the record -> done; else its `[gate] FAIL` /
  `REFUSED` lines stop the run. The stage does not resubmit failed tasks:
  the gate names them, and `--retry sim_bank` reruns the array, whose
  finished elements skip on their markers (the launcher's rule).
- Status: smoke-tested 2026-10-16 (7 pass, 1 skip, about 30 s; the Sbi-
  extractor bank suite untouched). The first davinci run (10:35) passed
  the cluster check, the block-1 test and the dry run; it found the
  profile defect above and that the Giulia banks had never been built
  (`~/banks/giulia_sim`, `giulia_real` absent, no `chain.log`), which the
  stage now builds.
  [2026-10-16 13:15, the dropped gate: smoke-tested, 8 pass, 1 skip,
  about 45 s (70 s under a 6-process CPU load); with the never-ran branch
  removed, `test_dropped_gate` and `test_log_timing` fail (the new checks
  bite). Regression: block 1 6 pass 1 skip; block 3 6 pass 1 skip (its
  contract test's ordering assertion corrected: it assumed S1's fake jobs
  outlast the first tick, which fails under load -- now S2 after S1's
  `done`); block 7 6 pass 1 skip; block 8 1 pass 1 skip.]
  [2026-10-18, the self-healing bank (D-094, D-095): smoke-tested, 11
  pass, 1 skip, about 2 min; five mutations each caught by a check (no
  quarantine; no member read in the verifier; any record taken as this
  gate's; no reuse of round 0's walltime; done without the certificate).
  Regression: block 1 6 pass 1 skip; block 3 6 pass 1 skip; block 7 6
  pass 1 skip; block 8 1 pass 1 skip (12 jobs: the verification job
  added). Not run here: davinci's PBS, Lustre, a real failing node.]
- Self-healing simulated bank [amended 2026-10-18; the user's answers
  D-094 and D-095]. Found by the read-only check of 2026-10-18: two tasks
  that FINISHED on the failing node (exit 0, markers written) left one
  shard each that is not a zip file -- a marker's sha256 is taken after
  the write, so it cannot vouch for what a failing node wrote, and the
  gate reads only the small arrays, so damage inside `x` passes it.
  Rules:
  (a) Every round of the simulated bank ends with a verification JOB,
      never on the login node: `tools/bank_verify.py --plan <the plan of
      record> --bank-root <bank_root> --out <work_root>/bank_verify/
      round<r>_try<a>_<YYYYmmdd_HHMMSS>.json --workers <bank.verify.ncpus>`
      (a new name each try, so no earlier report is read), through
      `jobs/run_argv.pbs` in `envs.extract`, on `bank.queue` with
      `bank.verify.{ncpus, mem, walltime}`. For every task of the plan it
      checks: a marker of this plan (its sha256) listing exactly the
      plan's chunks (shard, start, stop); for every chunk the marker gives
      a file, the `.npz` and the `.json` present, both sha256 equal to the
      marker's, every zip member's CRC (`zipfile.ZipFile.testzip`, which
      decompresses every member), the sidecar valid JSON; for a chunk the
      marker gives no file, no `.npz` present. It lists the bank folder: a
      `shard_NNNN.*` or `tasks/task_NNNN.*` file the plan's chunks or
      markers do not name belongs to that chunk's or that index's task; a
      `shard_*` file of no planned chunk, or a marker of another plan, is
      a GLOBAL problem; other files are reported, not judged. It also
      reads the record (`BANK_RECORD.json`: present, its sidecar, its
      plan). It writes the report -- each bad task with its problems and
      its files on disk (paths relative to the bank folder), the global
      problems, the record's state, the verdict `OK` or `BAD`, the host,
      the seconds -- and prints `VERIFY OK: ...` or `VERIFY BAD: ...` last.
      Exit 0 whatever it found; 2 on bad arguments or a plan that does not
      match its .sha256; 75 when it cannot read the plan or list the bank
      folder (an I/O error: the node it runs on is suspect). Stdlib only.
  (b) The round: `launch_bank.sh array` (the array and its gate); once the
      gate has ended -- passed, refused, or never ran (the rule above) --
      and the array has ended, the verification job. The gate PASSED only
      when the record was written after its submission (the record's
      stat, like the log's, is taken before the launcher call): a record
      left by an earlier round or run is not this gate's. Its report decides:
      - BAD with a global problem: STOP with the problems; nothing moved.
      - BAD with tasks only, and more than half of the plan's tasks (and
        at least 3) bad by an I/O error alone (`unreadable`): the
        verifier is suspect; verify again, nothing moved.
      - BAD with tasks only, otherwise, and fewer than
        `bank.max_rerun_rounds` reruns made: QUARANTINE -- every file the
        report lists for each bad task, and `BANK_RECORD.json`,
        `BANK_VERIFIED.json` and their sidecars when present, are moved
        with `os.rename` (never copied, never deleted; a move across file
        systems is refused) into `<paths.banks_root>/<tag>_sim_quarantine/
        <YYYYmmdd_HHMMSS>_round<r>/` under their names relative to the
        bank folder, with `QUARANTINE.json` there (each move's from, to,
        bytes, task and reasons; the report and its sha256; the time);
        then `launch_bank.sh array` again, round r + 1, with WALLTIME the
        value round 0 used (read from the launcher's `WALLTIME=auto ...
        -> hh:mm:ss` line, so a quarantined test-task marker does not stop
        the auto rule); finished tasks skip on their markers.
      - BAD after `bank.max_rerun_rounds` reruns: STOP with each task
        still bad, its problems and its task log's last `FAILED:` line.
      - OK and the record present with its sidecar: the stage writes
        `<bank_root>/BANK_VERIFIED.json` + .sha256 (the record's sha256,
        the report's path and sha256, the verifying host, the rounds, the
        quarantine folders) and is done.
      - OK and no record: the gate never ran -> the gate alone (once),
        then the certificate; the gate ran and refused -> STOP with its
        `[gate] FAIL` lines (a problem the verifier does not test: the
        contract, the donors, a repeated row).
      A verification job that ends without its report after the grace
      period is submitted again; at most 3 tries per round, then STOP.
  (c) S5's artefact of record is `BANK_RECORD.json` + .sha256 AND
      `BANK_VERIFIED.json` + .sha256 whose `record_sha256` is the
      record's sha256: a record without a matching certificate is not
      done (a `--retry sim_bank` or a fresh state verifies the bank).
  Config v6 adds `bank.verify.{ncpus, mem, walltime}` (Giulia: 8, 16gb,
  02:00:00 -- the gate's resources) and `bank.max_rerun_rounds` (Giulia:
  2, as the trials' and the members' `max_resubmits`); Claude's values
  `[proposed]`, D-094's "at most 2 rounds".
  Oracles (`tests/test_smoke_block2.py`):
  - verifier, in process, on the stand-in's bank built by the real
    tools: the intact bank -> OK, every shard tested, exit 0; a shard
    truncated -> BAD `not a zip file`; one byte flipped inside `x`'s
    compressed data with the marker's sha256 rewritten to match (bytes a
    failing node wrote and hashed) -> BAD, a CRC problem naming `x.npy`,
    while the gate passes that bank; a marker removed -> BAD `no marker`
    with the task's shards as its files; a `shard_NNNN.npz.tmp.npz` ->
    one of its task's files; a `shard_0999.npz` -> a global problem; a
    directory in place of a shard -> BAD `unreadable`; no plan -> exit
    2; no bank folder -> exit 75.
  - healing, end to end (the stand-in, the fake PBS): (h1) the gate
    passes a bank whose test task's shard is CRC-bad (marker rewritten):
    verify BAD -> the task and the record quarantined -> rerun with the
    round-0 walltime -> gate, verify OK, certificate, done; the
    quarantine folder holds the moved files and `QUARANTINE.json`; (h2)
    the davinci case: one task's element never starts in round 0
    (`FAKE_PBS_FAIL_ONCE`, exit -2, under the gating switch) and another
    task's shard truncated with its marker rewritten -> the gate dropped,
    verify names both, quarantine, rerun, done; (h3) a task that fails
    every time -> STOP after `max_rerun_rounds` reruns with its problem
    and its `FAILED:` line; (h4) a `shard_0999.npz` in the bank folder ->
    STOP with the global problem, nothing moved; the dropped gate's two
    cases (a dropped gate now leads to verify; with every task good, the
    gate alone, then the certificate).
  - decisions, in process (`test_decisions`): each kind of report --
    OK with this round's record (the certificate, then `is_done`; a record
    rewritten after it is not done), OK with an older record and an unrun
    gate (the gate alone; unrun twice: STOP), OK after a refusal (STOP
    with the gate's lines), a global problem, an I/O majority (verify
    again; the third time STOP), bad tasks with and without a rerun left
    (the quarantine's moves and its QUARANTINE.json; the STOP with the
    task log's `FAILED:` line) -- and a verification job that leaves no
    report (again; the third time STOP).

### Block 3: the input stages S1-S3 (`stages/inputs.py`)

- Module: `hpc/pipeline/stages/inputs.py` (`RealExtractionStage`,
  `SimRedetectionStage`, `LabelFreezeStage`) through Sbi-extractor's
  `extractor/launch_stage_d.sh`, `sim_reextract/launch_sim_reextract.sh`
  and `preflight_label_axes.py` (read at `e9aeb55`, driven as their
  headers say). For Giulia all three artefacts exist and the stages skip.
- Rules: S1 runs the launcher with CONFIG = `real.cohort_config`,
  COHORT_TAG = `real.cohort_tag` and the root `real.extract_root`, records
  its `  array :` and `  agg   :` jobs, and needs `wrote <root>/cohort_
  manifest.json` in `extractor/out/cohort_manifest_<tag>.log`; the
  launcher's own refusals (a NEW cohort whose flags file must be
  COMMITTED; a root already holding a manifest; flags differing from the
  tracked file) stop the run with their text. S2 needs `sim_reextract/
  profiles/<tag>.sh` (refused before anything runs when absent), runs
  `plan`, then `test` (WALLTIME = `sim.test_walltime`, TEST_INDEX 0; the
  job's exit 0 is its pass), then `array` with RESUME=1 and WALLTIME =
  `sim.walltime` (or `gate` alone when the launcher says `tasks.tsv is
  empty`), with COHORT_MANIFEST = S1's artefact, SIM_MAIN, OUT_ROOT, TOOLS,
  CAMPAIGN_GLOB, PLAN_ARGS, QUEUE, NCPUS, CONCURRENCY from `sim`, and
  needs `wrote <out_root>/REEXTRACTION_RECORD.json` in `sim_reextract/out/
  sim_reextract_gate_<tag>.log`. S3 runs the freeze tool on the login node
  with `--sim_main sim.sim_main --campaigns freeze.campaigns --require-
  conn-rule freeze.require_conn_rule --exclude NAME=REASON ... --out
  <sbix_dir>/<freeze.out>` and needs its `wrote` line (the tool writes the
  sidecar). Every launcher's environment is rebuilt from the configuration
  (a stray RESUME or WALLTIME in the caller's shell is dropped).
- Test oracles (`tests/test_smoke_block3.py`): S3 against the REAL tool
  on the fixture's simulation tree (topology block `['conn_prob']`, the
  sidecar matches; the Giulia argv carries the three `--exclude`); S1 and
  S2 against STUBS of the launchers (tests/standin.py) that print the real
  launchers' lines, submit the same jobs through qsub and write the
  artefacts from the jobs -- the real launchers need the recordings and
  the ANN tools (davinci) and have their own suites: the full chain (5
  jobs, S1 and S3 at once, S2 after S1), a one-task data set (the gate
  alone after the test task), the missing-profile refusal, the flags
  refusal, a refused aggregation and a refused gate (each with its REFUSED
  text in the STOP line); the Giulia dry run with every stage built and no
  REFUSE line.
- Not rehearsed: the real S1 and S2 tools end to end (davinci).
- Status: smoke-tested 2026-10-16 (6 pass, 1 skip, about 7 s).

### Block 4: the campaign prior in the runner

- Module: `hpc/joint/stage3/run_joint_arms.py` (the prior block of `main`).
- Public API: `campaign_log_prob(theta) -> (n,) float64` (0 inside the open
  unit box (0, 1)^p, -inf outside), module-level; the runner's behaviour.
- Rule: when the simulated sidecar's `latent_spec["kind"]` is `campaign`
  (the Sbi-extractor bank, D-078: the prior is the declared box, the unit
  box in the bank's coordinate), the runner builds no `LatentSBISpec` and
  the prior floor is L_0 = -E[log p(theta)] = 0 nats on the report rows
  (every row inside the box; a row outside is the existing refusal of
  `mc_prior_floor`); the information gain is then Delta_hat = -L. Any other
  kind (the bench banks' mixture prior) takes today's path unchanged. The
  real bank's `latent_spec["kind"]` `campaign_real` with `theta_withheld`
  false skips the pseudo-real endpoint as today; `cluster_scores` on a bank
  whose `cls` is one value returns its "too few rows" note.
- `--sbi-hpc-dir` defaults to this repository's `hpc/` (the runner lives in
  it) when that folder holds `npe_diagnostics.py`, so the spectrum's source
  is `repo:` rather than the local fallback unless the flag names another
  tree (the usage guide's open point, sec. 10).
- Test oracles (`hpc/joint/stage3/smoke_test_blocks_4_6.py`, section B4): on the
  Sbi-extractor fixture's banks (3 shards, 24 rows, W 18000, p 23; 72 real
  rows) a training element of arm A1 runs to completion and writes its
  record with `L0` = 0.0 exactly, `delta_hat` = -`L`, `endpoint_primary` =
  "simulated (arm R absent)" when no real shards are given and the
  replicate block present when they are; the Stage 3 suite still passes
  (33 checks: the bench path unchanged).
- Data flow: consumed by block 7's trials; feeds nothing else.
- Confirmed: yes 2026-10-15 (D-078; the user's "GO on with 4 5 and 6").
- Status: smoke-tested 2026-10-15 (`smoke_test_blocks_4_6.py` B4: pass; the Stage 3 suite 33/33 with `L_primary` 1.658521682024002 unchanged from `9fcbc88`)

### Block 5: the streaming loader (D-084)

- Modules: `hpc/joint/stage1/latent_bank.py` (`ShardIndex`),
  `stage2/joint_batches.py` (`ShardStream`; `ThreeStreamBatcher(sim_stream=)`),
  `stage2/joint_train.py` (`train_joint(val_eval=)`, `evaluate_npe_shards`),
  `stage3/run_joint_arms.py` (`--buffer-shards`).
- Public API: `ShardIndex(paths)`: `n_rows` (total), `n_shards`, `offsets`
  (shard k holds rows `offsets[k]:offsets[k+1]` of the concatenated order,
  which is the sorted path order, the same as `concat_shards`), the small
  per-row arrays concatenated (`theta` (n, p) float32, `cls`, `donor`,
  `well`, `batch`, `subregion`, `window_idx`, `realisation_id`,
  `contaminated`, `nu`), `sidecar` (the first shard's; every shard's
  `contract_digest` must equal it, else `ContractError`), `load_x(k) ->
  (n_k, W) float32` (the shard's windows, read on demand; `x` is never held
  for more than the shards a caller keeps), `rows_of(k)`.
  `ShardStream(index, row_mask, batch_size, buffer_shards, seed)`: `next()
  -> (theta (B, p), x (B, W), rows (B,) int64 global row ids)`;
  `state_dict()` / `load_state_dict()`; `n_rows` (the masked rows);
  `resident_shards` (the shards whose `x` is in memory now).
- The sampling scheme (D-084's rolling buffer, the quota rule): the stream
  walks the shards in passes. In each pass the shard order is a permutation
  drawn from the stream's own generator (seed + 1, the simulated stream's
  generator as today; a shard still resident from the previous pass is
  moved behind the first later non-resident one so it is not loaded twice);
  the buffer holds the masked rows of at most K = `buffer_shards` shards;
  every resident shard k carries a quota q_k, set to its number of masked
  rows n_k when it is loaded (plus the overshoot carried from its previous
  pass, below). Each of the B rows of a batch picks a resident shard with
  probability q_k / sum_j q_j over the resident shards, then one of that
  shard's masked rows uniformly, and q_k decreases by one per pick (the
  picks of one batch are drawn from the quotas as they stand at the start
  of the batch, so a quota can end a batch below zero: that overshoot is
  carried into the shard's quota at its next load, which keeps the draws
  per pass exact over many passes). A shard whose quota is spent is evicted
  and the next shard of the pass loaded. Consequences: every row's expected
  number of draws per pass is exactly 1 whatever the shards' sizes (shard
  k receives n_k picks per pass, spread uniformly over its n_k rows); a pass
  is n_rows / B steps; at most K shards' windows are ever resident. Shards
  with no masked row are skipped. When K >= n_shards the buffer holds every
  shard in the natural order, never evicts and never permutes: the row
  order is the concatenated order of the masked rows and `next()` draws
  `torch.randint(0, n_rows, (B,), generator)` exactly as `ThreeStreamBatcher.
  sim_batch` draws from the in-memory tensors, so the two paths produce
  identical batches at the same seed (the equivalence oracle below). With K
  < n_shards consecutive batches come from the same few shards (one task's
  simulations each); D-084 leaves the size of that effect on training to be
  measured on the bench (the measurement oracle below), not decided here.
  A first design (advance every n_oldest / B steps) was replaced before it
  was tested because its expected draws per row depend on the neighbouring
  shards' sizes (4/13 + 9/25 for sizes 4, 9, 16 at K = 2), and Giulia's
  shards range from a task's full 10,000-row chunk to its remainder.
- `train_joint(..., val_eval=None)`: when given, `val_eval()` (a callable
  returning the selection-split NLL in nats per row) replaces the
  in-memory `evaluate_npe(model, val_theta, val_x)`; `evaluate_npe_shards
  (model, index, row_mask, batch_size)` computes it shard by shard: the
  mean over the masked rows of -log q_omega(theta_i | h_psi(x_i)), the same
  sum and count as `evaluate_npe` on the concatenated tensors (float64
  accumulation of per-row float32 losses in both).
- `run_joint_arms.py --buffer-shards K` (int, default 0 = today's in-memory
  path): with K >= 1 the runner builds a `ShardIndex`, the grouped split
  from `index.donor` (the same `grouped_split`, the same hash), the stream
  over the training rows, the selection NLL by `evaluate_npe_shards`, the
  report-split per-row NLL and the embedding `z` shard by shard in row
  order, and the diagnostics' posterior samples on the first 256 report rows
  in row order (today's cap, `:561`), loading only the shards those rows
  lie in. The record gains `loader: {"buffer_shards": K, "n_shards": ...}`.
  Memory: at most K shards' `x` plus one shard being evaluated, plus the
  small arrays; never the bank.
- Library calls relied on: `numpy.load` (an `.npz` member is decompressed
  only when accessed, so the small arrays are read without `x`),
  `torch.randint`, `torch.randperm` with a `torch.Generator`.
- Custom code: the rolling-buffer schedule (checked: `torch.utils.data`'s
  `DataLoader` with a `Sampler` cannot bound memory to K shards while
  sampling with replacement across them without a custom dataset of the
  same size as this code, and it would add worker processes the PBS job
  does not want).
- Test oracles (`hpc/joint/stage3/smoke_test_blocks_4_6.py`, section B5), on the
  fixture banks and on a synthetic 6-shard bank of unequal sizes:
  - index: `ShardIndex` reproduces `concat_shards`' small arrays exactly
    (`assert_array_equal`) and its `offsets`; `load_x(k)` equals the k-th
    slice of the concatenated `x`; a shard of another contract is refused.
  - equivalence (K >= n_shards): the stream's first 50 batches equal the
    in-memory batcher's at the same seed (row ids and tensors, bit for bit);
    `run_joint_arms.py --buffer-shards 99` on the fixture gives the same
    `split_hash`, the same `history` and the same `L` as `--buffer-shards 0`
    at one thread (bit for bit: `torch.set_num_threads(1)`).
  - coverage and uniformity (K = 2 on 8 shards of unequal sizes): over 400
    passes every masked row is drawn, no unmasked row is ever drawn, the
    draws per row have mean 400 within 2 % and a coefficient of variation
    below 0.10 (the Poisson spread at 400 expected draws is 0.05; the
    tolerance is twice that), the draws per shard equal 400 n_k within 2 %
    (the quota rule's exactness), and the steps taken equal 400 n_rows / B
    to within one batch; `resident_shards` never exceeds K.
  - `evaluate_npe_shards` equals `evaluate_npe` on the concatenated tensors
    to 1e-6 nats (float64 sums of float32 terms in another order; 1e-6 is
    far above the roundoff of 24 rows and far below any real difference).
  - the measurement D-084 asks for: on the fixture, `L` at K = 1 against K
    = 99 after 4 epochs at one seed, reported (a number, not a pass/fail:
    the fixture is too small to decide it; the bench banks on davinci are
    the place, block 8's rehearsal log).
  - state: a stream saved after 37 batches and reloaded into a fresh
    stream yields the same next 20 batches as the uninterrupted one.
- Data flow: consumed by block 6 (the checkpoint carries the stream's
  state) and by block 7's trials (`--buffer-shards` from `loader.buffer_
  shards`).
- Cost of the streaming path, measured here on one 10,000 x 18,000 float32
  shard of random data (667 MB compressed): 2 ms to read the row count from
  the `.npy` header against 4.7 s to decompress the windows `[SANDBOX
  2026-10-15]`. So the index starts in seconds on 155 shards; each pass of
  the stream, each selection-split evaluation (every epoch) and each of the
  report-split passes decompresses every shard once more, a few minutes
  per epoch against the hours an epoch of the costliest configuration
  trains [estimate: the plan's sec. 4b].
- Confirmed: yes 2026-10-15 (D-084 as decided; the memory-mapped
  alternative Claude proposed on 2026-10-15 was not taken up and is not
  built; it stays a proposal in sec. 8).
- Status: smoke-tested 2026-10-15 (`smoke_test_blocks_4_6.py` B5: 5 checks
  pass; the D-084 measurement on the 8-shard fixture after 3 epochs: L at
  K = 99 2.9094, at K = 1 3.1878 -- a number, not a verdict; the bench
  banks on davinci decide, block 8)

### Block 6: runner flags, the per-epoch checkpoint and `--resume`

- Modules: `hpc/joint/stage3/run_joint_arms.py`, `stage2/joint_train.py`,
  `stage2/joint_model.py`.
- New runner flags, every default today's behaviour: `--passes P` (float;
  with it the steps per epoch are S = ceil(P / N_ep * n_train / B_sim),
  sec. 2.3, computed after the split and written to the record as
  `steps_per_epoch_effective` with `passes`; without it `--steps-per-epoch`
  as today); `--patience` (int, default 99: today's `TrainConfig(patience=
  99)`); `--split-fracs TR SEL REP` (default 0.7 0.15 0.15; refused unless
  they sum to 1 within 1e-9); `--split-seed` (default: `--seed`; drives
  `grouped_split`'s permutation alone, `--seed` keeps torch, the stream
  generators and the shuffle); `--head-pool-ops CODE` (0 or 1, default 0 =
  `BackboneConfig`'s default `("mean",)`; decoded by `condition_space.
  decode_head_pool_ops` and passed into `BackboneConfig(head_pool_ops=)`:
  D-086); `--flow-hidden-layers K` (default None = sbi's int behaviour, F-ac;
  with K the flow is built with `hidden_features=[h] * K`); `--threads N`
  (default: the `NCPUS` environment variable when set, else torch's own;
  applied by `torch.set_num_threads` and printed); `--resume`; `--buffer-
  shards` (block 5). The output stem is `<arm>_seed<k>` when the split seed
  equals the seed (unchanged) and `<arm>_split<s>_seed<k>` otherwise (it
  still ends in `_seed<k>`, which the tuner's glob `*_seed%d.json` reads).
- The checkpoint: `train_joint(..., ckpt_path=None, resume=False)` writes,
  at the end of every epoch, `<ckpt_path>` (atomic: `.tmp` then `os.
  replace`) holding `format` "joint_train_ckpt/1", `epoch` (the last
  completed), `step`, `model_state`, `optimizer_state`, `history`, `best`,
  `best_epoch`, `best_state`, `torch_rng` (`torch.get_rng_state()`),
  `batcher_state` (every stream generator's state and, under streaming, the
  `ShardStream` state), and `config` (`TrainConfig.to_dict()`). With
  `resume=True` and the file present, the loop restores all of it and
  continues from `epoch + 1`; absent, it starts fresh (so a job may always
  pass `--resume`). The early-stopping counter (best, best_epoch,
  best_state) is part of the state, so a resumed run stops at the epoch the
  uninterrupted run stops at (D-087's rule, sec. 2.3). The runner's
  checkpoint path is `<stem>_train_ckpt.pt`; the final `<stem>_ckpt.pt`
  (the model alone, today's) is unchanged, and the training checkpoint is
  removed once the final record is written.
- Test oracles (`hpc/joint/stage3/smoke_test_blocks_4_6.py`, section B6):
  - regression: a call without any new flag gives today's stem, split
    hash, `steps_per_epoch` and (on the fixture, 1 thread) today's `L` bit
    for bit against the block-4 run.
  - `--passes`: S equals the formula on the fixture's split (e.g. P 2, N_ep
    4, n_train 16, B 4: S = 2); the rows drawn over N_ep epochs are N_ep *
    S * B >= P * n_train and < P * n_train + N_ep * B.
  - `--split-fracs 0.85 0.10 0.05` on 2,612 synthetic donor ids at seed 0:
    2,220 / 261 / 131 donors (the check_config_v2 number); fractions
    summing to 0.95 are refused; a call without the flag gives today's hash.
  - `--split-seed`: two runs at seeds 1 and 2 with `--split-seed 1` share
    the `group` array of their per-row files and differ in `model_sha256`;
    `--split-seed` equal to `--seed` gives today's stem.
  - `--head-pool-ops 1`: the head's in-features are three times the
    summed stage widths (the head's Linear `in_features` read from the
    model); code 0 equals today's model (same parameter count and names).
  - `--flow-hidden-layers 2` at 12 transforms: every conditioner has 2
    hidden layers (the zuko MLP's Linear count), against 12 without the
    flag (F-ac).
  - `--threads`: the record and the log carry the thread count; with
    `NCPUS=3` in the environment and no flag, 3.
  - `--patience 2` on a run whose validation stops improving stops early
    (the history is shorter than `--epochs`) and restores the best epoch's
    weights (`val_npe` of the last history entry >= the minimum).
  - resume: a run with `--epochs 5` killed after epoch 2 (simulated: `--stop-after-epoch 2`, which ends the process after the
    checkpoint of epoch 2 with exit 0 and a `stopped after epoch 2` line -- a
    SIGKILL would do the same, the flag makes the test deterministic and
    lets a long fit be split by hand) and re-run with `--resume --epochs 5` gives the
    uninterrupted 5-epoch run's `history` (every number) and `model_sha256`
    bit for bit at one thread, and its stop epoch under `--patience 2`.
- Not in the checkpoint: the DSN loss adapter's separation-ramp counter
  (arms with `lambda_dsn > 0`: A2, A2s, A3) and the two-stage arms'
  pre-training; a resumed run of those arms restarts the ramp. Arm A1 (the
  Giulia arm, D-039) and A5 carry no such state and resume exactly.
- Data flow: block 7 passes every flag from the configuration; the
  pipeline's resubmission adds `--resume`.
- Confirmed: yes 2026-10-15 (D-086..D-089; the user's "GO on with 4 5 and
  6").
- Status: smoke-tested 2026-10-15 (`smoke_test_blocks_4_6.py` B6: the flags
  and the resume pass bit for bit at one thread, dropout 0.1 included; the
  Stage 2 suite 40/40 + 1 skip, the Stage 1 suite 46/46 + 1 skip, the
  Stage 4 suites 9/9 and 8/8 under `SKIP_CONDA=1`, the Sbi-extractor bank
  suite 15/15 + 1 skip, all unchanged)
### Block 7: the search and the best fit S7-S8

- Modules: `hpc/pipeline/stages/search.py` (`SearchStage`), `stages/best_fit.py`
  (`BestFitStage`), `stages/common.py` (the tuner's common arguments, the
  qsub line, the argv files, the clock), `envs.py` (`run_in_env`: a command
  in a conda environment on the login node, `PIPELINE_SKIP_CONDA=1` for the
  tests), `jobs/run_argv.pbs` (one job = one command line read from a file),
  `tools/bank_split_hash.py`, `tools/ensemble_nll.py`; `hpc/joint/stage4/
  joint_space.py` (`head_pool_ops` as an axis, `apply_overrides`: a range
  per axis from the configuration, an axis freed by two levels or pinned by
  one), `stage4/npe_tune_joint.py` (the runner flags passed through
  `build_argv`, `--space-overrides`, `--gp-seed`, `--runner-extra`, `argv
  --argv-file`, `evaluate` exit 3 on a checkpoint without a record, the
  `costliest` subcommand), `stage3/run_joint_arms.py` (`--probe-steps`,
  `--probe-val-rows`: the probe mode); `config.py` (v5: `search.trial.
  queue`, `best_fit.queue`; `NEEDS_CODE` empty); `pipeline.py` (a
  `SchedulerError` in a poll is logged and retried next tick; the status
  table summarises a stage with more than eight jobs).
- Public API: `stages.search.SearchStage`, `stages.best_fit.BestFitStage`
  (the `Stage` interface of block 1, `max_resubmits` 0: the trials and the
  members carry their own); `envs.run_in_env(env_name, argv, cwd, log_path)
  -> (rc, text)`; `stages.common.tuner_common_args(cfg, identity) -> [str]`,
  `qsub_line(cfg, name, queue, ncpus, mem, walltime, log, argv_file, workdir)
  -> [str]`, `elapsed_hours(start_utc)`, `mem_to_gb`; the tuner's
  `costliest --out-dir D` (writes `costliest_b<B>.json` per searched batch
  size), `--runner-extra "<args>"` (appended last to the runner's argv; the
  last occurrence of a flag wins); `tools/bank_split_hash.py --sim-shards
  G --split-fracs a b c --split-seed s --out F` (the pass line `SPLIT OK:
  hash=... contract=...`), `tools/ensemble_nll.py --members F... --out F`
  (`ENSEMBLE OK: M=.. n=.. L_bar=..`).
- Inputs: the configuration; the two banks (S5, S6); the scheduler. Outputs
  under `<tune_root>/<campaign>/`: `bank_split.json` (the identity),
  `gate/costliest_b<B>.json` and `gate/b<B>/<stem>_probe.json`,
  `pending/`, `trials/` (the tuner's ledger; a trial the pipeline gives up
  on is a record with `status` `killed`, written by the pipeline),
  `runs/<trial_id>/` (the runner's outputs), `jobs/` (every argv file, job
  log and the login-node log), `pipeline_rounds.json` (the identity, the
  gate's verdict, the rounds with their trial ids, the clock; survives a
  `--retry`), `finalists.json`, `SEARCH_COMPLETE.json` (the artefact of
  record: `ended_by` in {rounds, budget, no_new_proposals}, the counts, the
  elapsed hours, the finalist's id, nll and configuration). Under
  `<runs_root>/<tag>_best/`: `finalist_config.json`, `member_seed<k>/`
  (the runner's outputs, stem `<arm>_seed<k>` when k equals the split seed
  and `<arm>_split<s>_seed<k>` otherwise), `jobs/`, `ensemble_nll.json`,
  `BEST_FIT_COMPLETE.json` (the artefact of record: the members' records
  and L_m, the ensemble's L_bar, M, n).
- Parameters: every value of `search`, `best_fit`, `split`, `loader` and
  `envs` (sec. 3.1); the elapsed clock starts at round 0's first
  submission (not at the gate's probes: block 7's proposal 1); GP round r
  proposes with `--gp-seed gp_seed + r`; the trials' `--patience` is 99
  (no patience, D-088's reading); the members' `--n-posterior-draws` is
  `best_fit.n_posterior_draws` through `--runner-extra`.
- Library calls relied on: the tuner's `propose` (`npe_tune_search.
  propose`: skopt's `Optimizer.ask(n_points, strategy="cl_min")`), `evaluate`,
  `finalists`; `scipy.special.logsumexp`; `run_joint_arms.grouped_split`
  (the split hash: the pipeline computes it with the runner's function, not
  a copy); the runner's checkpoint and `--resume` (block 6).
- Custom code: the round loop and the throttle (no library drives PBS
  jobs from a ledger); the budget gate's rule; the ensemble formula (three
  lines over `logsumexp`); the `costliest` rule table (which end of each
  axis costs the most, stated per axis in the tuner).
- Rule of the budget gate: with T_B the runner's projection at batch size
  B, T_B = N_ep (S_B t_step + n_sel t_val_row) from `probe_steps` steps
  (the first excluded as warm-up) and `probe_val_rows` rows, the gate
  passes iff max_B T_B x safety_factor <= `search.trial.walltime` and
  max_B maxrss_B <= `search.trial.mem`; it STOPs the run otherwise, with
  the numbers; an existing probe file is reused rather than re-probed.
- Rule of the rounds: a trial's job ending with a ledger record is `ok`
  or `failed` as the record says (its pending file is removed); ending
  without one, it is resubmitted with `--resume` while `resubmits <
  trial.max_resubmits` and `resubmit_killed`, else recorded `killed`;
  `evaluate` exits 3 when a training checkpoint exists without a record
  (a stop, not a failure). A round is complete when every trial of it is
  terminal; then the search ends by `rounds` (r == gp_rounds), by `budget`
  (the elapsed clock >= `max_total_walltime_h`; the running trials had
  finished by then) or by `no_new_proposals`, else round r + 1 is proposed
  with `trials_per_round` points and `n_initial_points` as skopt's initial
  design size (so if fewer than `n_initial_points` trials are `ok`, the
  surrogate is not consulted yet and the proposals are random: the
  initial design continues). A finished search with no `ok` trial fails
  the stage.
- Rule of the retry: `--retry search` on a search that did not complete
  restores the rounds from `pipeline_rounds.json`, treats a trial with a
  ledger record as terminal, and revives every trial the pipeline recorded
  `killed` (the record set aside as `<id>.json.killed<n>`, the pending spec
  rewritten from it, the resubmission counter reset, `ended_by` cleared),
  so raising the walltime or `max_resubmits` and retrying re-runs them;
  a completed search is skipped by its artefact, so its killed trials
  stay killed unless `SEARCH_COMPLETE.json` is removed first.
- Rule of the identity: `pipeline_rounds.json` carries the split hash,
  the contract digest, n_train, the campaign, the anchors, the overrides
  and the flow settings; a results folder whose file disagrees with the
  configuration is refused before anything is submitted (proposal 2).
- Test oracles (`hpc/pipeline/tests/test_smoke_block7.py`, the fake PBS
  with `FAKE_PBS_ENFORCE_WALLTIME`, the blocks 4-6 bank fixture: 194 rows,
  8 shards, 44 donors; the trial jobs run the real tuner and runner):
  - known answer: the split tool's hash equals `grouped_split`'s on the
    fixture and its contract digest the sidecar's; `ensemble_nll.py` on
    three copies of one member gives that member's mean within 1e-12, on
    two members L_bar <= the members' mean (Jensen), and refuses members
    of different row counts; `costliest` on the Giulia space gives depth
    5, width 3.0, ResNet, embedding 64, fusion, pooling 1, hidden 512, 12
    transforms at each of 256 / 512 / 1024; the committed configuration
    has no NEEDS-CODE key and `config_version` 6 [5 until 2026-10-18].
  - reference: the Giulia dry run prints the gate's, the trials' and the
    members' qsub lines (`cpu`, 48 cores, 160gb / 200gb, 120 / 300 h) and
    no REFUSE line for S7 / S8 (blocks 2-3's not-built lines remain).
  - invariants: `tuner_common_args` is a pure function of the
    configuration and the identity; a `pipeline_rounds.json` of another
    identity stops the run (exit 3, `REFUSE:` in the STOP line, nothing
    submitted); no project value in the code.
  - contract: on the fixture (overrides pinning the encoder and the flow,
    batch levels 16 / 32, 2 initial points, 1 GP round of 2, throttle 1,
    passes 1 in 2 epochs, members seeds 1 and 2 on split seed 1): the gate
    submits 2 probes and passes; round 0 and round 1 run their 2 trials
    each with no two trial jobs overlapping; `SEARCH_COMPLETE.json` says
    `rounds`, 4 ok; every record's `split_hash` is the identity's; the
    runner's argv in the job log carries `--num-bins 10`, `--head-pool-
    ops`, `--passes 1.0`, `--split-fracs 0.85 0.1 0.05`, `--buffer-shards
    2`, `--threads 1`, `--flow-hidden-layers 2`, `--patience 99`; the best
    fit's two members share the split hash (seed 1, different from the
    search's), carry `n_posterior_draws` 16, `patience` 10, the finalist's
    lr / weight decay / pooling / fusion / batch size; L_bar <= the
    members' mean; the sequencer stopped after 12 polls and re-run submits
    no job twice (8 submissions in all); `PIPELINE DONE`.
  - determinism (the kill path): under a 3 s trial walltime enforced by
    the fake PBS (exit 271) with pre-seeded probe files, every trial is
    resubmitted twice with `--resume` (the first argv file without, the
    later ones with) and then recorded `killed`; the search ends with no
    ok trial and the run stops with exit 3 and the STOP line; 12
    submissions; then, with the walltime raised to 10 min, `--retry
    search --accept-config-change` revives the 4 killed trials (their
    records set aside as `.json.killed1`), runs them, and the search ends
    `rounds` with 4 ok, the best fit following (18 submissions in all).
  - edge: `max_total_walltime_h` 1e-6 ends the search after round 0
    (`budget`, 1 round evaluated, 2 ok), the GP round never proposed; a
    pre-made `BEST_FIT_COMPLETE.json` skips S8.
- Not in the sandbox: davinci's queues and conda (the tests run with
  `PIPELINE_SKIP_CONDA=1`); the first run on davinci is the dry run, then
  the plain command; the probes of the real space at 48 cores.
- Confirmed: the values yes (D-085..D-093); the two proposals (the clock
  starts at round 0's first submission; a results folder of another
  identity is refused) built as proposed, to confirm.
- Status: smoke-tested 2026-10-16 (`test_smoke_block7.py` 6 pass, 1 skip,
  about 2 minutes; `test_smoke_pipeline.py` 6 pass, 1 skip; the Stage 4
  suites 8/8 under `SKIP_CONDA=1` and 9/9; the Stage 3 suite and
  `smoke_test_blocks_4_6.py` unchanged)
- Node faults and corrupt checkpoints [amended 2026-10-18; the user's
  answers D-096 and D-097]. Found on davinci 2026-10-18 (the read-only
  check of 14:28): trials `16641a4d79ce`, `4006d8ff495f` and
  `42f269b8ecb8` ran one after another on `dvnode012` and failed 12-14
  min in, when the node lost the home file system -- two runners killed
  by SIGBUS (the ledger's `runner exited -7`, after `BrokenPipeError:
  [Errno 108] Cannot send after transport endpoint shutdown`), one
  `OSError: [Errno 5] Input/output error` reading a shard the
  verification job had read intact on `dvnode001`. The tuner records any
  non-zero runner exit as `failed`, which the rounds took as final: three
  design points lost to a node's fault, not to their configurations.
  - The signature (`stages/common.py`, `node_fault(record_error,
    log_text) -> str or None`): (a) the runner killed by SIGBUS -- the
    record's error `runner exited -<SIGBUS>` (-7 on Linux), or in the log
    `RUN_ARGV EXIT: <128 + SIGBUS>` (135) or `Bus error`; (b) a file-system
    error in the log -- `[Errno <n>]` for EIO (5), ENOTCONN (107),
    ESHUTDOWN (108), ESTALE (116), EREMOTEIO (121), the numbers from
    Python's `errno`, or the texts `Input/output error`, `Transport
    endpoint is not connected`, `Cannot send after transport endpoint
    shutdown`, `Stale file handle`, `Could not chdir to home directory`.
    Not node faults: any other exit (a traceback without those errors,
    SIGKILL -- memory or an operator --, ENOSPC and EDQUOT -- storage
    full --, ENOENT).
  - Rule of the search (D-096): a trial whose job ended with a ledger
    record of status `failed` is `examining` (not terminal: its round
    waits) until its evidence is read -- the record's error, and its
    latest job log `jobs/<id>.<n>.log` once that log has its `RUN_ARGV
    EXIT:` line or `LOG_GRACE_S` (300 s) have passed since the failure was
    seen. With `search.resubmit_node_faults` and a node fault: the record
    is moved to `trials/<id>.json.nodefault<k>` (k = 1, 2, ...; kept, out
    of the ledger's `*.json`), the pending spec is rewritten from the
    record's configuration if absent, and, while `resubmits <
    trial.max_resubmits` (the counter the walltime kills use), the trial is
    `waiting` again, to be submitted with `--resume` when the throttle has
    a slot (earliest round first): `search: trial <id12> NODE FAULT on
    <host> (<why>); record set aside as <name>; queued again with --resume
    (<r> of <max>); <n> node fault(s) on <host> so far`; with no
    resubmission left, the pipeline's `killed` record (error `killed: node
    fault on <host> (<why>) after <r> resubmission(s)`, revivable by
    `--retry search` as any killed trial) and `... no resubmission left:
    recorded killed; ...`. Otherwise the trial is `failed`, final, as
    before: `search: trial <id12> failed: not a node fault (<error>);
    final`. A record that no longer says `failed` when it is read decides
    as any record. A trial already `failed` and never
    examined -- recorded before this code, or restored by a fresh state --
    is examined at the next poll (the three of 2026-10-18 come back so).
    With `search.resubmit_node_faults` false nothing is examined (the
    behaviour before D-096).
  - Rule of the best fit (D-096 applied to the members: Claude's reading,
    `[proposed]`): a member's job that ended without a record and whose
    log shows a node fault is resubmitted with `--resume` up to
    `best_fit.max_resubmits` -- `best_fit: member seed <k> NODE FAULT on
    <host> (<why>); resubmitted with --resume (<r> of <max>)` -- instead
    of failing the stage as a refusal; with `best_fit.resubmit_node_faults`
    false, the refusal rule as before.
  - Rule of the checkpoints (D-097): before a trial or a member is
    submitted, each `*_train_ckpt.pt` of its run folder is opened as a zip
    archive and every member's CRC-32 checked (`zipfile.ZipFile.testzip`);
    a file that is not a readable zip archive, or a member that fails, is
    moved to `<name>.corrupt<k>` (kept) and logged `<stage>: <who>
    checkpoint <name> failed its CRC check (<why>); moved aside as
    <name>.corrupt<k>; it restarts from epoch 0`; a file the login node
    cannot read (an OSError) is left in place and logged. The job is then
    submitted as before (with `--resume` on a resubmission: the runner,
    finding no checkpoint, starts from epoch 0). Why: torch.save writes a
    zip archive of stored members with their CRC-32; in the sandbox (torch
    2.14.1) `testzip` passes a fresh checkpoint, names the member after one
    flipped byte -- which `torch.load` reads WITHOUT an error, returning
    altered weights -- and a truncated file is not a zip archive
    (`torch.load` raises) `[SANDBOX 2026-10-18]`.
  - Config v7: `search.resubmit_node_faults`, `best_fit.resubmit_node_
    faults` (bool; true for Giulia). The tuner and the runner are not
    changed: the trial jobs on davinci keep their code.
  - Test oracles (`tests/test_smoke_block7b.py`; the blocks 4-6 fixture
    under the fake PBS; the real tuner, and the real runner behind a
    wrapper that injects the faults -- a copy of `hpc/` whose
    `stage3/run_joint_arms.py` is the wrapper; the plan per trial in the
    order of first call, the throttle at 1, the gate's probe files
    pre-seeded):
    - known answer: `node_fault` names a node fault on the three davinci
      lines of 2026-10-18 (verbatim) and on `RUN_ARGV EXIT: 135`; it
      returns None on a ValueError traceback, `[Errno 2]`, `[Errno 28]`,
      `[Errno 122]`, `runner exited -9` and `runner exited 1` without a
      log.
    - contract (the healing): trial 1 SIGBUS then ok; trial 2 EIO then ok;
      trial 3 a ValueError (the negative control): `failed`, final; trial
      4 ESHUTDOWN three times: resubmitted twice, then `killed` with `node
      fault` in its error; the search completes with 2 ok, 1 failed, 1
      killed; `trials/<id>.json.nodefault1` for trials 1, 2 and 4 and
      `.nodefault2` for 4; the second argv of trials 1 and 2 carries
      `--resume`; member seed 1 EIO then ok, member seed 2 ok; `PIPELINE
      DONE`.
    - the legacy sweep: with `resubmit_node_faults` false a SIGBUS trial
      is recorded `failed`; the sequencer is stopped by `pipeline.py
      stop`; with the key true and `--accept-config-change` the restarted
      sequencer heals it (its record set aside) and the search completes
      with every trial ok, no job submitted twice.
    - the checkpoint guard: a trial whose first job stops after epoch 1
      and leaves a checkpoint with one flipped byte (a CRC failure), and a
      trial whose checkpoint is truncated: both checkpoints moved aside as
      `.corrupt1` with the log line, both trials complete `ok`.
    - the stop and the retry guard (block 1's amendment): SIGTERM, and
      then `pipeline.py stop --wait`, each end a running sequencer at a
      tick boundary (exit 0, `stopped on request (...); resumable: 1
      job(s) still queued or running keep running`); `--retry search`
      while a trial job runs is refused, in the dry run and the run (exit
      2, nothing submitted), and accepted once the job has ended (the
      search completes, 4 submissions in all); `stop` with no sequencer
      says so.
    - mutations: with the signature disabled the healing test fails; with
      the sweep disabled the legacy test fails; with the guard disabled
      the checkpoint test fails; with the retry guard disabled the stop
      and retry test fails.
  - Status: smoke-tested 2026-10-18 (`test_smoke_block7b.py` 6 pass, 1
    skip, about 6 minutes; the four mutations each make their check FAIL:
    signature -> test_contract, sweep -> test_determinism, guard -> the
    checkpoint check, retry_guard -> the stop and retry check; regression
    on the final code: block 1 6 pass 1 skip, block 2 11 pass 1 skip,
    block 3 6 pass 1 skip, block 7 6 pass 1 skip, block 8 1 pass 1 skip).
    Not in the sandbox: davinci's PBS (its job logs land when the job
    ends, which the 300 s wait for `RUN_ARGV EXIT:` covers), its Lustre
    errors (the signature's lines are davinci's own of 2026-10-18).
### Block 8: end to end on the stand-in and the delivery

- Test (`tests/test_smoke_block8.py`): the one command from an empty work
  root on the stand-in in the Giulia situation (S1-S3's artefacts exist;
  the bank fixture with 12 more topologies per task, 104 simulations, W
  18,000, p 23): S4-S6 with the REAL Sbi-extractor bank tools (the plan,
  the test task, the array of 3 and its gate, the real bank), S7-S8 with
  the REAL tuner and runner (2 probes, 2 + 2 trials, the finalists, 2
  members on split seed 1, the ensemble NLL), the sequencer stopped after
  3, 8 and 15 polls and re-run each time; 11 jobs, none twice; `--status`
  shows 3 skipped and 5 done; a last run on the state does nothing. About
  50 s.
- Delivery: `sci/pipeline` pushed by D-062 (a dry run, then the push);
  the davinci lines in the README (the cluster-side check block, the dry
  run, the run, the status), each with the lines that must appear.
- Not rehearsed: davinci's queues and conda, the 48-core probes on the
  real space, the real S1 and S2 tools.
- Status: smoke-tested 2026-10-16 (1 pass, 1 skip).

Every block is written (2026-10-16); the constraints are those of secs. 2-4
above.

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `bash hpc/pipeline/run_pipeline.sh --project giulia --dry-run` | the stage table and every qsub line; submits nothing | the committed `giulia.json`; davinci paths need not exist (each stage reports `not done`) | seconds |
| `bash hpc/pipeline/run_pipeline.sh --project giulia` | the one command: starts the sequencer under `nohup`, prints the log and state paths | davinci | days |
| `bash hpc/pipeline/run_pipeline.sh --project giulia --status` | the stage table from the state file and the live scheduler | -- | seconds |
| `python3 hpc/pipeline/tests/test_smoke_pipeline.py` | block 1's smoke test under the fake PBS | none (synthetic project in a temporary directory) | about 30 s |
| `python3 hpc/pipeline/tests/test_smoke_block7.py` | block 7's smoke test: the search and the best fit under the fake PBS on the fixture banks | the training environment (torch, sbi, skopt) | about 2 min |
| `python3 hpc/pipeline/tests/test_smoke_block2.py`, `test_smoke_block3.py` | blocks 2 and 3: the bank stages and the input stages on the stand-in davinci | the Sbi-extractor checkout (`SBIX_CLONE`), numpy | about 20 s, 7 s |
| `python3 hpc/pipeline/tests/test_smoke_block8.py` | block 8: the one command end to end on the stand-in with the real tools | both of the above | about 1 min |

## 7. Not executable in the sandbox

davinci itself: the real `qsub`/`qstat`, the conda environments, the banks
(45 GB of windows), the Sbi-extractor artefacts of record. The substitute:
the fake PBS of `tests/fake_pbs/`, synthetic artefacts in a temporary work
root, and -- for blocks 2-3 -- the Sbi-extractor fixtures (`export_windows/
bank_plan_fixture.py`, `sim_reextract/smoke_test_sim_reextract.py`, `extractor/
smoke_test_stage_d_jobs.py`) on a stand-in `/davinci-1` tree. The first real
run on davinci is `--dry-run`, then the plain command; the pass lines are
stated in the README.

## 8. Open questions

- Block 5's alternative, proposed 2026-10-15 and not taken up: decompress
  the windows once into a plain float32 file (about 45 GB on disk for
  Giulia) and read it through a memory map, which gives exactly i.i.d.
  batches with bounded memory; D-084's rolling buffer stands until the bench
  measurement (block 8) says the within-buffer correlation matters (raised
  2026-10-15).

- Whether PBS Pro on davinci honours an array throttle (`-J 0-49%25`) for
  the search's `max_concurrent_trials` (the bank's array used `%20`, so it
  is likely; `[NOT VERIFIED]` for the tune job) -- block 7 (raised
  2026-10-15). [closed 2026-10-16: block 7 submits one job per trial and
  throttles in the sequencer, so no array throttle is needed; the tuner's
  array job (`joint_tune.pbs`) is not used by the pipeline, because the
  pending set is its index map and shrinks as elements finish.]
- The `NCPUS` job variable for torch's thread count (block 6; `[NOT
  VERIFIED on davinci]`) (raised 2026-10-15). [narrowed 2026-10-16: the
  pipeline passes `--threads` from `search.trial.ncpus` / `best_fit.ncpus`
  explicitly, so the variable is not relied on; `run_argv.pbs` prints it.]
- Block 7's two proposals, built as proposed and to confirm: the elapsed
  clock starts at round 0's first submission (the gate's probes are
  before it); a results folder holding a search of another identity is
  refused rather than merged (raised 2026-10-16).
- Config v5 added `search.trial.queue` and `best_fit.queue` (both `cpu`,
  the queue whose per-job cap is 300:00:00 [KB, the C8 handoff]); the
  user to confirm the queue (raised 2026-10-16).
- The dropped gate (block 2, 2026-10-16): what ended `1845589` unrun --
  the davinci diagnosis decides whether D-006's Statement needs a
  qualifier (for instance: ordering only when the elements exit with a
  code of their own; an element the scheduler kills ends the dependent
  unrun). [answered 2026-10-18: an element that never started (Exit_status
  -2, its node `dvnode012` without the home file system) -- not a kill;
  block 2's defect entry has the records. The qualifier: afterok on an
  array is ordering only for elements that run and exit with a code; an
  element PBS fails to start ends the dependent unrun (a reading of the
  records, not of the server's log).] Two options for the user, neither taken: (1) the launcher's
  gate dependency `afterok` -> `afterany` (Sbi-extractor, a laptop push
  by D-062; `afterany:<array>[]` was accepted by davinci in the D-006
  sweep, and is ordering by definition); (2) the stage resubmits failed
  tasks itself, bounded by a new `bank.max_resubmits`, instead of
  stopping with the gate's names (raised 2026-10-16).
