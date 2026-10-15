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
  `pbs.py`, `stages/`, `run_pipeline.sh`, `README.md`, `tests/`
- `hpc/pipeline/projects/giulia/giulia.json` -- the Giulia project's
  configuration (config v4)
- the changes the later blocks make under `hpc/joint/` (listed per block in
  sec. 5 as they are written; the rest of `hpc/joint/` is their input, not
  under test here): `hpc/joint/stage1/latent_bank.py` (`ShardIndex`,
  `npz_member_shape`), `stage2/joint_batches.py` (`ShardStream`, the
  batcher's `sim_stream` and state), `stage2/joint_train.py` (`evaluate_npe_
  shards`, the checkpoint, `StoppedAfterEpoch`), `stage2/joint_model.py`
  (`flow_hidden_layers`), `stage3/run_joint_arms.py` (the campaign prior,
  the streaming path, the flags), `stage3/smoke_test_blocks_4_6.py`

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
| S7 | `search` | S5, S6 | `<paths.tune_root>/<search.campaign>/SEARCH_COMPLETE.json` | SBI `hpc/joint/stage4/npe_tune_joint.py` propose / evaluate through `jobs/launch_joint_tune.sh` (`envs.train`) -- block 7 |
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
| `config_version` | int | 4 |
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
| `search.budget_gate.{probe_steps, probe_val_rows}` | int | >= 1 |
| `search.budget_gate.safety_factor` | float | >= 1 |
| `search.max_total_walltime_h` | float or null | > 0; null = no cap |
| `search.walltime_reading` | str | `elapsed` |
| `search.resubmit_killed` | bool | -- |
| `best_fit.split_seed` | int | >= 0 |
| `best_fit.member_seeds` | [int] | non-empty, distinct |
| `best_fit.passes` | float | > 0 |
| `best_fit.{epochs, patience_epochs, n_posterior_draws, ncpus, max_resubmits}` | int | >= 1; `max_resubmits` >= 0 |
| `best_fit.{mem, walltime}` | str | as for the trial |
| `loader.buffer_shards` | int | >= 1 |

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
never retries `qsub`.

The fake PBS of `tests/fake_pbs/` provides the same two commands for the
tests: `qsub` records the job and runs its script after `FAKE_PBS_DELAY`
seconds in a detached process (array elements in parallel, `depend=afterok`
honoured), `qstat -x -f` prints a PBS-like record. It is a test fixture, not
part of the pipeline.

## 4. Global conventions

- Units used internally: seconds for wall times (printed as `hh:mm:ss` where
  PBS wants them), nats per row for NLLs, Hz for rates.
- Floating-point precision: float64 for every number the pipeline computes
  (the ensemble NLL); the trainers' own precision is theirs.
- Randomness: none in the pipeline itself; the seeds it passes are the
  configuration's (`gp_seed`, `train_seed`, `split_seed`, `member_seeds`).
- Python: the stdlib only in `hpc/pipeline/` (json, os, subprocess, time,
  hashlib, argparse); syntax compatible with Python 3.6 (no f-strings, no
  walrus, no postponed annotations), since the sequencer runs on the login
  node under the project's `envs.extract` environment; pure ASCII, LF.
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
    background process on the login node (`nohup`).
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
  json` (config v4), `README.md`.
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
- Confirmed: yes 2026-10-15 (the user's answers to the build handoff's Q1-Q7
  and the follow-up; D-090..D-093)
- Status: smoke-tested 2026-10-15 (6 pass, 1 skip: convergence, not a
  discretisation; nothing skipped for want of davinci -- the scheduler is
  the fake PBS, which is what the sandbox can test)

### Block 2: the bank stages S4-S6 -- planned (`stages/bank.py`)
### Block 3: the input stages S1-S3 -- planned (`stages/real_extraction.py`, `stages/sim_redetection.py`, `stages/label_freeze.py`)
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
### Block 7: the search and the best fit S7-S8 -- planned (`stages/search.py`, `stages/best_fit.py`, `hpc/joint/stage4/npe_tune_joint.py`, `stage4/joint_space.py`)
### Block 8: end to end on the stand-in and the delivery -- planned

Each planned block gets its entry in the format of block 1 when it is
written; its constraints are those of secs. 2-4 above.

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `bash hpc/pipeline/run_pipeline.sh --project giulia --dry-run` | the stage table and every qsub line; submits nothing | the committed `giulia.json`; davinci paths need not exist (each stage reports `not done`) | seconds |
| `bash hpc/pipeline/run_pipeline.sh --project giulia` | the one command: starts the sequencer under `nohup`, prints the log and state paths | davinci | days |
| `bash hpc/pipeline/run_pipeline.sh --project giulia --status` | the stage table from the state file and the live scheduler | -- | seconds |
| `python3 hpc/pipeline/tests/test_smoke_pipeline.py` | block 1's smoke test under the fake PBS | none (synthetic project in a temporary directory) | about 30 s |

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
  2026-10-15).
- The `NCPUS` job variable for torch's thread count (block 6; `[NOT
  VERIFIED on davinci]`) (raised 2026-10-15).
