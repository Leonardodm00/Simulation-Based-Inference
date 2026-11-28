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
  configuration (config v6 [2026-10-30: v10 since 2026-10-19, D-101;
  `resources.max_cores` 300 since 2026-10-30, D-115])
- `hpc/pipeline/projects/linddup/linddup.json` -- the LindDUP project's
  configuration (config v10; added 2026-10-29, D-104..D-109; v2 of the
  file the same day: `sim.plan_args` `--quiet-hours 6` (D-112), the
  search's `gp_seed` 1 and `max_concurrent_trials` 25, Giulia's (D-114);
  [2026-11-18: `sim.plan_args` `--quiet-hours 1`, D-155])
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

- the synthetic A2/A5 experiment (D-118..D-120; roadmap
  `specs/roadmap/synth.yaml`): its formulation is sec. 2.4; the paths its
  blocks change (the bank builder under `hpc/joint/stage1/`, the batcher and
  the runner under `hpc/joint/stage2/` and `stage3/`, the pipeline's
  synthetic kind and `hpc/pipeline/projects/synth/`) are listed per block in
  sec. 5 as they are written (added 2026-11-03)
- the synthetic kind of the pipeline (blocks 15-17, the roadmap's step S4;
  added 2026-11-03): `hpc/pipeline/stages/synthetic.py`,
  `hpc/pipeline/tools/synth_bank_record.py`,
  `hpc/pipeline/projects/synth/synth.json`,
  `hpc/pipeline/tests/test_smoke_synthetic.py`, and the changes they make to
  `config.py`, `stages/__init__.py`, `stages/common.py`, `stages/search.py`,
  `stages/best_fit.py` and `pipeline.py`
- the synthetic experiment's evaluation (blocks 18-22, the roadmap's step
  S6; added 2026-11-11): block 18's other-arm best fits -- the tuner's
  `finalists --arm/--out` in `hpc/joint/stage4/npe_tune_joint.py` with its
  check J37 in `hpc/joint/stage4/smoke_test_joint_tune.py`, and the changes
  to `hpc/pipeline/` listed in its entry; blocks 19-22's
  `hpc/pipeline/evaluation/` once built (drafted, sec. 8 Q-E1..Q-E6)
- the synthetic experiment's relaunch (D-132; blocks 24-27; added
  2026-11-13): the purge of a synthetic project with its archive (block
  24, `hpc/pipeline/tools/purge_outputs.py`), and the changes blocks 25-27
  make, listed per block in sec. 5 as they are written [2026-11-16, D-136:
  blocks 24-28; 27 and 28, the comparisons and their figures, are wave W2,
  with the new `hpc/pipeline/tests/test_smoke_comparisons.py`]

- LindDUP's three arms and the pipeline's standard (D-138..D-151; blocks
  29-32, wave W3; added 2026-11-17): `hpc/pipeline/stages/runs.py`,
  `stages/search_stop.py`, `stages/replicate.py`, the changes blocks 29-31
  make to `pipeline.py`, `stages/synthetic.py`, `stages/search.py`,
  `tools/search_report.py`, `search_reset.py`, `perrow_check.py`,
  `purge_outputs.py`, `inputs_check.py` (block 32), `hpc/joint/stage2/joint_batches.py`,
  `hpc/joint/stage3/run_joint_arms.py`, their tests
  (`hpc/pipeline/tests/test_smoke_runs.py`, `test_smoke_stop.py`,
  `hpc/joint/stage3/smoke_test_rep_pairs.py`), the contract's
  `OPTIONAL_ROW_ARRAYS` in `hpc/joint/stage1/latent_bank.py`, config v11
  (`config.py`, the three projects' files) and
  `hpc/pipeline/projects/linddup/linddup_cohort_map.csv`

- the evaluation of a recorded project (D-159..D-163; roadmap
  `specs/roadmap/tool.yaml`, step S2, wave 1; blocks 33-38; added
  2026-11-23, on the branch `sci/tool`): its formulation is sec. 2.6; the
  new files under `hpc/pipeline/evaluation/` (`ev_rec_config.py`,
  `ev_rec_plan.py`, `ev_rec_data.py`, `ev_gate.py`, `ev_rec_sample.py`,
  `ev_rec_metrics.py`, `ev_rec_validate.py`, `ev_rec_figures.py`,
  `ev_rec_report.py`, `ev_rec_submit.py`, `submit_rec_eval.sh`), their tests
  and fixture (`hpc/pipeline/tests/test_smoke_rec_*.py`,
  `hpc/pipeline/tests/rec_fixture.py`),
  `hpc/pipeline/projects/synth/synth_rec_eval.json`, and one additive
  change to block 20's `ev_metrics.py` (block 36). The synthetic
  evaluation's other files are not changed

Everything else in the repository is not specified yet and is out of scope
for the tester.

- [2026-11-27] The tool (`sbitool/`, roadmap `tool`, steps S1, S3 and S4): sec. 2.7 and
  blocks 40-46 -- the package, its schemas and interfaces, the model bundle, the card,
  the recording adapters and the layout, the windows, the inference from a bundle, the
  device's figures. [2026-11-28] Built and smoke-tested in the sandbox (blocks 40-46,
  31 checks); tested by the integration tester at 39e6f66 (report 2026-11-28_0013);
  nothing run on davinci.

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

### 1.1 The synthetic A2/A5 experiment (D-118..D-120; added 2026-11-03)

The same command, run on a synthetic data set whose parameters are known, to
test arms A2 and A5 -- the two joint arms whose extra term reads the real
windows (A2: the DSN metric loss on the real windows' condition labels; A5:
the replicate-consistency term on pairs of real windows from two wells of one
donor) -- end to end, through the search and the best fit (D-118). The data
set has two arms, both built by the bench generator (sec. 2.4): a simulated
arm S whose theta covers the whole parameter box, and a synthetic "real" arm
R of three conditions on donors of two wells, whose theta is recorded and
withheld from training, at gap severities pi = 0 and pi = 0.3 (D-119, D-120).
Deliverables: the five searches (S-A2 and S-A5 on each arm-R bank, and one
S-A1, D-122) and their best fits, as above for any data set, every trial's record carrying the
pseudo-real endpoint L_pseudo_real (sec. 2.4) beside the ledger's `nll`. The
evaluation (the roadmap's step S6) is specified when it starts.

### 1.2 The evaluation of a recorded project (D-159; added 2026-11-23)

For a data set whose real arm is a set of recordings the pipeline ends at
the best fit (D-083), and the evaluation of sec. 2.5 exists for the
synthetic kind alone. This step answers, for any recorded project and each
of its ensembles:

- (i) is the ensemble faithful to its simulator on simulations it never
  trained on: the NLL, the expected coverage, SBC, TARP and the contraction,
  with the rows near an edge of the prior box read apart;
- (ii) what does it return on each recorded window, and where do the
  windows of each well put their mass;
- (iii) does each recorded window look like anything the simulator
  produces, as the ensemble's encoders see it (the gate);
- (iv) do the project's arms differ on the held-out simulations.

The predictive check through the project's own simulator is wave 2 and is
not specified here (D-159). L-C2ST is not part of this step (D-163 (2)).

Deliverables, per ensemble: the samples, the metrics, the figures and one
report. They are read by the user, by the model bundle of the roadmap's
step S3 (the held-out evaluation and the gate's reference embeddings) and
by the viewer of step S6. LindDUP is the test of the whole chain (the
roadmap's step S9), not the design target: no value of a particular
project is in the code (D-159).

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

**The synthetic kind (block 15; D-118..D-123; added 2026-11-03).** A
configuration with `"kind": "synthetic"` (sec. 3.1) has no S1-S6: its graph
is built from the configuration. With R the entries of `synthetic.real` and
Q the entries of `synthetic.searches`, its stages are, in this order:

| s | name | dep(s) | done(s): the artefact of record |
|---|---|---|---|
| B_S | `sim_bank` | -- | `<banks_root>/<dataset>_sim/BANK_RECORD.json` with a matching `.sha256` whose `build` equals the configuration's (block 16) |
| B_r, each r in R | `real_bank_<r.name>` | -- | `<banks_root>/<dataset>_real_<r.name>/BANK_RECORD.json`, the same rule |
| S_q, each q in Q | `search_<q.name>` | B_S, B_{q.real} | `<tune_root>/<q.name>/<q.campaign>/SEARCH_COMPLETE.json` |
| F_q, each q in Q | `best_fit_<q.name>` | S_q | `<runs_root>/<q.name>/<dataset>_best/BEST_FIT_COMPLETE.json` |
| G_q, each q in Q with `alt_best_fit` true (block 18, D-126; added 2026-11-11) | `best_fit_alt_<q.name>` | S_q | `<runs_root>/<q.name>/<dataset>_best_alt/BEST_FIT_COMPLETE.json` |

S_q and F_q of one q are adjacent, in the order of `synthetic.searches`
[2026-11-11: and G_q, when q has one, follows F_q; G_q is S8 for the best
trial of the arm S_q's finalist is not (sec. 2.5, block 18)].
They are stages S7 and S8 run on q's view of the configuration:
`paths.tune_root` and `paths.runs_root` joined with q.name, `search.campaign`
= q.campaign, the simulated bank B_S and the real bank B_{q.real} -- passed
for every campaign, S-A1 included (D-122: its trials are scored on R0). The
sequencer ticks the stages in this order, and every stage's submissions
share `resources.max_cores` (D-099), so the free cores go to the earliest
search (and its best fit) first.

**A recorded data set with several searches (block 29; D-148 (1); added
2026-11-17).** A recorded configuration whose `searches` is a list Q (v11,
sec. 3.1) has S1-S6 as above, then, for each q in Q in its order:

| s | name | dep(s) | done(s): the artefact of record |
|---|---|---|---|
| S7_q | `search_<q.name>` | S5, S6 | `<tune_root>/<q.name>/<q.campaign>/SEARCH_COMPLETE.json` |
| S8_q | `best_fit_<q.name>` | S7_q | `<runs_root>/<q.name>/<dataset>_best/BEST_FIT_COMPLETE.json` |

S7_q and S8_q are S7 and S8 run on q's view (`stages/runs.py` RunView:
`paths.tune_root` and `paths.runs_root` joined with q.name,
`search.campaign` = q.campaign), the synthetic kind's construction. Every
stage's jobs count against `resources.max_cores` (D-099); the jobs of q's
stages also count against q.max_cores when it is an integer (block 29), so
each search keeps its own share (LindDUP: 148 cores each, D-141). With
`searches` null the graph is the eight stages above (the single search of
`search.campaign`, as before v11). A search stops by hand with
`pipeline.py stop-search` (block 30): its best fit then starts on the best
trial so far.

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
5. [2026-11-14, D-134; block 7 (i)] `--retry <s>` also returns to `pending`
   every stage downstream of s -- a stage that has s among its dependencies,
   directly or through other stages -- whose status is `done` or `skipped`,
   so that rule 2 reads its done(s) again when its turn comes: this is the
   one case in which a done stage is re-checked. A downstream stage with a
   job still queued or running refuses the retry as s's own live jobs do
   (exit 2, nothing changed); a downstream stage that is `pending` or
   `failed`, or `running` with no job live (rule 3 verifies it at its next
   tick), is left as it is. One log line per stage returned. A stage
   returned to `pending` has its record reset as `--retry` resets s's own
   (its finished jobs and its stage data are forgotten), and every check
   comes before any reset, so a refusal changes nothing.

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

### 2.4 The synthetic A2/A5 experiment (D-118..D-120; added 2026-11-03)

**The generator.** The bench provider (`hpc/joint/stage1/
bench_burst_provider.py`, the builder's `--provider bench`; D-119). A
parameter vector theta = (theta_0, ..., theta_9) in the open unit box
(0, 1)^10 is the coordinate of the affine map

    a_k(theta_k) = L_k (1 - theta_k) + U_k theta_k,   k = 0, ..., 9,

onto axis k's physical range [L_k, U_k] (`BENCH_AXES`). The class-bearing
axes are K_L = {0, ..., 6} (`burst_rate`, `irregularity`, `ibi_cv`,
`n_fragments`, `intraburst_rate`, `participation_mean`, `burst_duration`),
the free axes K_F = {7, 8, 9} (`fragment_duty`, `participation_kappa`,
`background`). For one well g with parameter theta, realisation seed r_g and
nuisance vector nu_g, the builder's `simulate_windows` returns J windows
x_{g,j} in R^W, j = 1, ..., J: the provider's per-unit-mean IFR trace (the
spikes of the burst process at a(theta), smoothed by `compute_ifr_trace`
with a 40-ms Gaussian and divided by the n_n = 100 units) at f_s = 100 Hz,
cut into J disjoint windows of T_win = 180 s, W = f_s T_win = 18,000
samples (D-120), then mapped by the observation model

    x_obs(t) = A(nu_g) x(t) + b(nu_g, t)        (`latent_nuisance.py`, eq. N1)

with the default `NuisanceSpec`, nested batch : donor : well (batches of 8
consecutive donor indices). In arm R at pi > 0 the gap acts before the
nuisance (below).

**Arm S, the simulated bank.** N_S = 20,000 parameter vectors theta_d,
d = 1, ..., N_S, i.i.d. from the box prior

    p_S(theta) = 1  for theta in (0, 1)^10   (D-118, D-120);

each "donor" d of arm S has two wells at theta_d, each with its own
realisation seed, and J_S = 1 window per well: 2 N_S J_S = 40,000 windows;
pi = 0. Every row's condition label is cls = -1 (arm S has no conditions).
The bank's sidecar declares the box prior (`latent_spec["kind"]` = `box`),
which the runner reads as block 4 reads `campaign`: no `LatentSBISpec`, the
prior floor L_0 = 0 on the report rows (the roadmap's step S2).

**Arm R, the synthetic cohort.** C = 3 conditions c in {0, 1, 2}, n_c = 9
donors per condition, N_R = 27 donors (D-120). The condition c_d of donor d
is fixed by the design, not drawn: each condition is assigned to exactly 9
donor indices, in the order of a permutation seeded by the arm's seed, so
that the batches of 8 consecutive donor indices mix the conditions. For each
fixed c in {0, 1, 2}, independently over the donors d with c_d = c,

    theta_d | c_d = c  ~  p_R(theta | c)
        = prod_{k in K_L} TN(theta_k ; mu_{c,k}, tau^2 ; (0, 1))
          * prod_{k in K_F} 1{0 < theta_k < 1},

where TN(. ; mu, tau^2 ; (0, 1)) is the density of a normal law with mean mu
and standard deviation tau truncated to (0, 1) and renormalised, tau =
tau_ov = 0.10, and the condition centres are (D-120)

    mu_c = 0.5 * 1_7 + s (m_c - 0.5 * 1_7),   s = 0.6,

with 1_7 the all-ones vector of R^7 and m_c in [0, 1]^7 the DSN generator's
regular simplex centres for three classes on seven axes
(`hpc/dsn/latent_burst_generator.py`, `_class_center_vectors(3, 7,
"simplex")`): for every c != c', ||m_c - m_c'||_2 = 0.7366 and the centred
centres have cosine -1/2, so ||mu_c - mu_c'||_2 = 0.442 `[SANDBOX
2026-11-03]`. Each donor has two wells at theta_d, each with its own
realisation seed, and J_R = 6 windows per well (1080 s per well; DUP15HD's
six windows per subregion trace): 2 N_R = 54 wells, 2 N_R J_R = 324
windows. Every row carries cls = c_d. The bank records theta and marks it
withheld (`theta_withheld` true): the runner never trains on arm R's theta
and scores the pseudo-real endpoint on it.

**The gap (D-119, D-120).** Two arm-R banks: R0 at pi = 0, and R3 at
pi = 0.3 with the three modes of `latent_gap.py`:

- (a) range shift: for k in K_F the range is displaced by
  delta = 0.35 pi = 0.105 of its width w_k = U_k - L_k,
  a_k = (L_k + delta w_k)(1 - theta_k) + (U_k + delta w_k) theta_k, and for
  `fragment_duty` (k = 7) the shifted value is clipped at 1:
  a_7 = min((L_7 + delta w_7)(1 - theta_7) + (U_7 + delta w_7) theta_7, 1)
  (D-120), so a donor with theta_7 > 0.895 gets duty 1 and a smaller shift;
- (c) drift: for each well g one phase phi_g ~ U(0, 2 pi), and
  x_g(t) <- x_g(t) + 0.5 pi xbar_g sin(2 pi t / (120 s) + phi_g) over the
  well's absolute time t, xbar_g the mean of x_g over its J_R windows;
- (d) contamination: for each well g, n_g = floor(f J_R) + B_g of its
  windows, f = 0.2 pi = 0.06, B_g ~ Bernoulli(f J_R - floor(f J_R)) -- at
  J_R = 6 one window with probability 0.36, none otherwise -- chosen
  uniformly; a chosen window becomes max(u x + e, 0) elementwise, u ~
  U(2, 5), e_t ~ N(0, sd(x)^2) i.i.d. over its samples t.

At pi = 0 every mode is the identity (`GapSpec.active`). The gap acts on
arm R only (the builder refuses pi != 0 on arm S).

**Seeds (D-120, D-121; F-aw).** Arm S is built with seed s_S, R0 with
s_R0 and R3 with s_R3, pairwise distinct: R0 and R3 are independent cohorts
of 27 donors (D-121), so the comparison of pi = 0 with pi = 0.3 carries the
cohort-to-cohort spread of the separability (below) as well as the gap.

**Separability, the reason for s (D-119, D-120).** For a cohort built as
above at pi = 0, three accuracies of the condition, chance being 1/3:

- the theta ceiling A_theta = E[ max_c P(c_d = c | theta_d) ], the Bayes
  accuracy of the condition given the donor's theta under equal condition
  probabilities; by the data-processing inequality along
  c_d -> theta_d -> x (the nuisance independent of (c_d, theta_d)), no
  classifier of a donor's windows exceeds it;
- the one-window accuracy A_win of a fixed reference classifier: the
  runner's eight fixed window statistics (`FixedStatsSummary`, arm A_ref's
  summary), standardised, multinomial logistic regression, five folds
  grouped by donor;
- the one-donor accuracy A_don: the same classifier's log-probabilities
  summed over the donor's 2 J_R windows.

At s = 0.6, tau = 0.10 and 180-s windows, over six distinct cohorts of 72 to
150 donors, A_win = 0.648-0.730 (mean 0.68), A_don = 0.650-0.825 (mean 0.73)
and A_theta = 0.973-0.976 `[SANDBOX 2026-11-03]`; "medium" (D-119) is this
setting -- one window near 2/3, halfway between chance and certainty. A
cohort of 27 donors is one draw of this law, so the built R0's own
separability is measured on the bank and recorded (the roadmap's step S2).

**The searches (D-119, D-120).** For each arm-R bank b in {R0, R3} and each
campaign k in {S-A2, S-A5}, one search (sec. 2.3) with arm S as the
simulated stream and b's windows as the real stream: n_initial_points = 20,
gp_rounds = 4, trials_per_round = 10, so N = 60 trials, then the best fit of
the search. In S-A2 a trial with dsn_on = 1 adds lambda_dsn L_DSN on b's
windows labelled by condition, over the C = 3 classes counted from b's
labels (F-t; the roadmap's step S3); in S-A5 a trial with rep_on = 1 adds
lambda_rep L_rep on pairs of windows from the two different wells of one
donor of b (F-ab; step S3), and a cohort with no such pair refuses (F-aa); a
trial with dsn_on = 0 (S-A2) or rep_on = 0 (S-A5) is arm A1. A fifth
search, S-A1 (D-122), optimises arm A1 alone with the same budget: A1
trains on arm S only, so one S-A1 search serves both cohorts -- its trials
are run with R0 as the real stream (scored on R0, trained without it), and
its best fit is scored on R3 at the evaluation. The project's core cap is
120 (D-120). Every trial's record carries L, the simulated
report split's NLL (the ledger's `nll`, sec. 2.3), and the pseudo-real
endpoint

    L_pseudo_real = (1/n_R) sum_{i=1}^{n_R} -log q_omega(theta_{d(i)} | h_psi(x_i)),

over the n_R windows x_i of b, d(i) the donor of window i and theta_{d(i)}
its withheld parameter. The search ranks on L, as on real data whose theta
is unknown (D-121); L_pseudo_real ranks nothing and is read by the
evaluation, which compares, within each search, the best trial with the
extra term on (dsn_on = 1 in S-A2, rep_on = 1 in S-A5) and the best A1
trial, each picked on L, on L_pseudo_real paired by donor (the roadmap's
step S6 specifies it). [2026-11-11: sec. 2.5.]

### 2.5 The evaluation (step S6; D-126; added 2026-11-11)

**What is compared (D-121, D-122, D-126).** For a search q of campaign S-A2
or S-A5 (sec. 2.4) let A(q) be its term arm (A2 for S-A2, A5 for S-A5); a
completed trial t of q has the arm a(t) in {A1, A(q)} given by its
configuration's switches (the tuner's `arm_for_config`: dsn_on = 1 gives
A2, rep_on = 1 gives A5, both 0 give A1). For each arm a in {A1, A(q)}, the
arm's finalist is

    t*_{q,a} = argmin { L(t) : t a completed, untagged trial of q with a(t) = a },

L(t) the trial's simulated report-split NLL (the ledger's `nll`): as the
search's own finalist t*_q = argmin_t L(t), it is ranked on L and never on
L_pseudo_real (D-121). t*_q = t*_{q,a(t*_q)}, and the search's best fit
(block 7, step S5) is t*_q's ensemble. Block 18 fits the other arm's
finalist t*_{q,abar}, abar the arm of {A1, A(q)} that is not a(t*_q), with
the same members' settings, so that every such q has one ensemble per arm,
E_{q,A1} and E_{q,A(q)}. S-A1 has one arm and one ensemble, E_{S-A1,A1};
its members train on arm S only, and it is scored on R0 and on R3 (D-122).

[amended 2026-11-16, D-132 (5) and D-136; blocks 27 and 28] **The
comparisons.** The relaunch (D-132, `synth.json` v4) adds the campaigns
S-A2-only and S-A5-only (block 26): the term always on, no A1 trial, so
such a search has one ensemble, E_{q,A(q)}, and no E_{q,A1}. A comparison
kappa is an ordered pair (E_kappa, E'_kappa) of ensembles, read
donor-paired as E3 below with E = E_kappa and E' = E'_kappa. For each
search q whose campaign has a term -- S-A2 and S-A5, which nest A1, and
S-A2-only and S-A5-only, which do not -- A(q) is its term arm (A2 for S-A2
and S-A2-only, A5 for S-A5 and S-A5-only), and:

- (a) against A1, kappa named q: E_kappa = E_{q,A(q)} and E'_kappa = q's
  A1 reference -- E_{q,A1} when q's campaign nests A1 (the finalist's or
  block 18's ensemble, D-126, as before), and E_{s,A1} when it does not, s
  the configuration's one search of campaign S-A1 (no such search, or two
  or more, is a refusal: the reference would be missing or ambiguous);
- (b) A2 against A5 (D-136), kappa named `<q>_vs_<q'>`: for every pair of
  searches (q, q') with A(q) = A2, A(q') = A5 and q.real = q'.real
  (whichever comes first in the configuration), E_kappa = E_{q,A2} and
  E'_kappa = E_{q',A5}, so that Dbar < 0 favours A2.

A search of campaign S-A1 enters only as a reference; one of campaign
S-A25 takes part in no comparison (its arms are not defined against A1
here) and is named as such. The comparisons are listed (a) first, in the
configuration's order of q, then (b), in the order of q and then of q'.
Under `synth.json` v4 they are S-A2-R0, S-A5-R0, S-A2-R3 and S-A5-R3, each
against S-A1:A1, then S-A2-R0_vs_S-A5-R0 and S-A2-R3_vs_S-A5-R3. A
comparison is made on every data set D both of its ensembles are sampled on
(block 19: D_sim for every ensemble -- one report split for every member,
arm and search, sec. 2.3 -- and every cohort each is scored on, with R3's
theta-dagger set), with the two samples' rows, theta^ref and donors equal on
D (otherwise a refusal); its paired draws share their seeds as within one
search (block 19's common random numbers). sigma_seed and G.2's verdict are
read per comparison and data set, with no adjustment for the number of
comparisons (D-136). A comparison that cannot be made -- an ensemble not
complete, not sampled with the current values, or not requested -- is
named, never dropped silently (the audit's F705).

**The ensembles.** E_{q,a} has M = 5 members m = 1, ..., M. Member m is
trained with train seed m on the donor split of split seed 1 (sec. 2.3: the
same split for every member, every arm and every search), 40 passes in 40
epochs, patience 10 (block 17's values). Member m's posterior density is
q_{q,a,m}(theta | x) = q_{omega_m}(theta | h_{psi_m}(x)), a density on
(0, 1)^10 that is 0 outside the open box (the flow acts on the logit of the
box: `hpc/joint/stage2/joint_model.py`,
`z_score_theta="transform_to_unconstrained"`). [amended 2026-11-11: 0 as
mathematics; the code returns finite values outside the box (-114.8 nats
with theta_8 = 1.05, -116.8 with theta_7 = 1, `[SANDBOX]`), its transform
clamping, so the evaluation never reads the flow there (block 19).] The ensemble's density is
the equal-weight mixture: for each fixed window x,

    q_{q,a}(theta | x) = (1/M) sum_{m=1}^{M} q_{q,a,m}(theta | x),     (E1)

whose mean NLL over the report split is block 7's L_bar.

**The data sets (D-126).** Four, each a list of rows i = 1, ..., n, row i a
window x_i with a reference parameter theta^ref_i:

- D_sim: the report split of arm S under split seed 1, the rows every
  member scores in its `_perrow.npz`; theta^ref_i = theta_i, the row's own
  parameter, drawn from the box prior p_S(theta) (sec. 2.4);
- D_R0 and D_R3: the cohort's n_R = 324 windows; theta^ref_i =
  theta*_{d(i)}, the withheld parameter of the window's donor d(i), whose
  condition is c_{d(i)};
- D_R3+: D_R3's windows with the pseudo-true parameter of the range shift,
  theta^ref_i = theta^dag_{d(i)}, where for each donor d of R3

      theta^dag_{d,k} = theta*_{d,k}                  for k in K_L,
      theta^dag_{d,7} = min(theta*_{d,7} + delta, 1),
      theta^dag_{d,k} = theta*_{d,k} + delta          for k in {8, 9},

  with delta = 0.35 pi = 0.105: the unshifted affine map at theta^dag_d
  gives the physical values the shifted map gives at theta*_d (sec. 2.4
  (a), the clip of `fragment_duty` included; the identity checked to float64
  roundoff on random ranges `[SANDBOX 2026-11-11]`). The drift (c) and the
  contamination (d) have no coordinate in the box, so theta^dag_d is
  pseudo-true for mode (a) alone. For a donor with theta*_{d,k} >= 1 -
  delta for some k in K_F, theta^dag_d lies outside the open box or on its
  boundary, where every member's density is 0 (sec. 8, Q-E3).
  [amended 2026-11-16, D-132 (3), block 25: K_L and K_F are the bank's own
  split, as its sidecar records it (`free_idx`); the formulas above are
  `synth.json` v3's (K_L = {0, ..., 6}, K_F = {7, 8, 9}). Under v4 K_L =
  {0, ..., 7} and K_F = {8, 9}: theta^dag_{d,k} = theta*_{d,k} + delta for
  k in {8, 9} and theta*_{d,k} elsewhere, and axis 7's cap at 1 applies only
  where 7 is in K_F; a donor leaves the box with probability 1 - (1 -
  delta)^2 = 0.20 under the uniform free axes.]

**Quantities.** For an ensemble E (or a single member, read with M = 1) and
a data set D, with n_post draws theta^(s)_i ~ q_E(theta | x_i), s = 1, ...,
n_post, per row i:

- the NLL l_i(E) = -log q_E(theta^ref_i | x_i), its mean over the rows,
  and on a cohort the donor mean lbar_d(E) = (1/n_d) sum_{i : d(i) = d}
  l_i(E) over the donor's n_d = 12 windows (2 wells x 6);
- the HPD rank gamma_i(E) = (1/n_post) sum_s 1{ q_E(theta^(s)_i | x_i) >
  q_E(theta^ref_i | x_i) }, the q_E-mass, estimated by the draws, of the
  region where q_E( . | x_i) exceeds its value at theta^ref_i; theta^ref_i
  lies in the HPD region of credibility beta iff gamma_i(E) <= beta (up to
  the draws' Monte Carlo error), and the expected coverage is, for each
  fixed beta in (0, 1),

      C_E(beta) = (1/n) sum_{i=1}^{n} 1{ gamma_i(E) <= beta };      (E2)

  on D_sim, whose (theta^ref_i, x_i) are draws of the joint p_S(theta) p(x
  | theta), C_E(beta) -> beta for every beta when q_E(theta | x) = p(theta
  | x) for p-almost every x (Hermans et al., TMLR 11/2022, Definition 2.1
  and eq. 5, `[KB, full text]`); a C_E(beta) below beta is overconfidence,
  above it underconfidence; on a cohort the same curve reads the
  calibration on that cohort's law, not on p_S;
- the marginal ranks r_{i,k}(E) = sum_s 1{ theta^(s)_{i,k} < theta^ref_{i,k}
  }, k = 0, ..., 9, each uniform on {0, ..., n_post} on D_sim under the
  same condition (simulation-based calibration; the practical guide in the
  project's knowledge base, "Simulation Based Inference A Practical Guide",
  p. 38, `[KB, full text]`, which also lists the local tests, L-C2ST among
  them; its case study, Fig. 10, checks a five-member ensemble with expected
  coverage, SBC, L-C2ST and posterior-predictive statistics);
- TARP's expected coverage (Lemos et al., ICML 2023, Algorithm 2, `[KB,
  full text]`) with the reference points drawn uniformly from the box
  [amended 2026-11-11, block 20: and, as the primary form, with reference
  points read out linearly from the windows' statistics
  (`npe_diagnostics.tarp`, `reference_mode="x"`), since uniform points cannot
  see a posterior that ignores the data];
- for each axis k: the posterior mean mhat_{i,k}(E) = (1/n_post) sum_s
  theta^(s)_{i,k}, the marginal SD sigma_{i,k}(E), the central 90 %
  interval; the z-score zeta_{i,k}(E) = (mhat_{i,k}(E) - theta^ref_{i,k}) /
  sigma_{i,k}(E) and the contraction 1 - sigma_{i,k}(E)^2 / sigma_0^2,
  sigma_0^2 = 1/12 the box prior's marginal variance;
- on a cohort, the donor-paired difference of two ensembles E and E',

      Delta_d(E, E') = lbar_d(E) - lbar_d(E'),   d = 1, ..., N_R,   (E3)

  its mean Dbar(E, E') over the N_R = 27 donors and a donor-cluster
  bootstrap interval (`hpc/bootstrap_paired.py`), read by the deck pack's
  two-clause rule (claude/deck_pack/08_SEC_G, G.0 and G.2, project
  knowledge): |Dbar| beyond 2 sigma_seed, sigma_seed the larger of the two
  arms' across-seed SDs of their members' L_pseudo_real, and the interval
  excluding 0; one copy of each donor enters the bootstrap (the ensembles'
  donor means, so G.2's pooling of five seeds' copies, finding F-bn, does
  not arise); within a search q, E = E_{q,A(q)} and E' = E_{q,A1}
  [amended 2026-11-16, D-136: E and E' are the two ensembles of a
  comparison kappa, "The comparisons" above, of which the within-search
  pair is case (a) for a campaign that nests A1; on D_sim the donors are
  the report split's] [2026-11-16, the final check of wave W2, F1:
  blocks 20 and 27 compute Dbar as `paired_bootstrap(..., within="all")`'s
  estimate, the mean of d_i over the ok rows; it equals the mean over the
  donors of Delta_d when every donor has the same number of ok rows, as on
  every data set of synth (2 rows per donor on D_sim, 12 on a cohort, all or
  none on R3dag, since theta-dagger is one per donor); the estimator for
  unequal rows is open (the hardening wave)];
- each member's embeddings z = h_{psi_m}(x) on D_sim and on a cohort, and
  their squared MMD with a Gaussian kernel at the median bandwidth
  (`hpc/npe_diagnostics.py`, `mmd2_biased`).

The quantities' estimators, n_post, the posterior-predictive check and the
L-C2ST settings are blocks 19 and 20's, open in sec. 8 (Q-E1..Q-E6).

### 2.6 The evaluation of a recorded project (roadmap `tool`, step S2, wave 1; D-159..D-163; added 2026-11-23)

Symbols of sec. 2.5 keep their meaning here: an ensemble E and its density
q_E(theta | x), the equal-weight mixture (E1) of its members' densities;
n_post; the NLL l_i(E); the HPD rank gamma_i(E); the expected coverage
C_E(beta) (E2); the paired difference (E3). The symbols introduced here are
in the Notation table. Nothing in this section holds a value of a
particular project (D-159): the number of axes, the window length, the
members, the wells and the classes are read from the configuration and the
banks.

**The view of a project.** The evaluation reads a project as a view: its
ensembles, its simulated bank B_S and its recording banks.

- A recorded configuration (sec. 3.1; not synthetic). The searches are
  `stages.runs.search_views(cfg)`. Search q's ensemble is its best fit,
  `<runs_root of q's view>/<dataset>_best`, once that folder has its
  `BEST_FIT_COMPLETE.json`; the ensemble's arm a is its members' records'
  `arm`, which must agree. Its name is `<q>:<a>` (stem `<q>_<a>`) when the
  configuration has a `searches` list, and `best:<a>` (stem `best_<a>`)
  when `searches` is null. B_S is `<banks_root>/<dataset>_sim`. There is
  one recording bank, named `real`: `<banks_root>/<dataset>_real`.
- A synthetic configuration read as recorded (D-162; for the validation,
  and only when the evaluation's configuration has `as_recorded` true).
  The ensembles are block 19's (name, stem, folder, members). B_S is the
  synthetic simulated bank. An ensemble's recording banks are its cohorts
  (block 19's rule), each under its cohort's name. The data module returns
  no parameter for a recording bank, so a cohort's withheld theta* never
  reaches the code under test.

An ensemble E has M >= 1 members m = 1, ..., M, each rebuilt as block 19
rebuilds it. p is the number of columns of B_S's `theta`, so that theta
lies in the unit box (0, 1)^p; W is the window length in samples; d_z is
the dimension of a member's embedding (the E of block 19's array shapes;
written d_z here because E is the ensemble).

**The data sets of an ensemble E.**

- D_sim: the report rows of the members' split of B_S, in global row
  order: rows i = 1, ..., n. Row i is a window x_i with its own parameter
  theta_i in the unit box and its simulated network d(i) (the bank's
  `donor`). Three subsets of D_sim are used:
  - the calibration rows: the first row of each network (sec. 8, Q-E2);
  - the draw rows (D-163 (1)): for each network its first k_draw rows,
    k_draw = `sim_draw_rows_per_donor`, an integer >= 1, or every row when
    that value is null; n_d is their number. Every calibration row is a
    draw row;
  - the gate rows: the rows that have a role in the gate (below).
- D_r, for each recording bank r of E: every row of the bank, in bank
  order: rows j = 1, ..., n_r. Row j is a window x_j with its ids -- its
  class c(j), its well g(j), and its donor, batch, subregion, window index
  and pair group as the bank has them -- and no parameter.

**On D_sim.** l_i(E), gamma_i(E), C_E(beta), the marginal ranks, TARP and
the recovery are those of sec. 2.5, read with theta^ref_i = theta_i.

- The NLL l_i(E) = -log q_E(theta_i | x_i), for every row i of D_sim whose
  theta_i is in the open box (every row, not the draw rows alone: D-163
  (1)); its mean over those rows, each member's mean, and the mean per
  network.
- On each draw row i: n_post draws theta^(s)_i ~ q_E(theta | x_i), s = 1,
  ..., n_post, n_post / M from each member, in member order; n_post is a
  multiple of M (D-163 (4): n_post = 1,000).
- Over the calibration rows whose theta_i is in the open box: C_E(beta) on
  the grid beta = 0, 0.01, ..., 1, with the binomial band of a calibrated
  estimator on that many independent rows; the same curve for each member;
  SBC and TARP over the calibration rows. Over the draw rows: the recovery
  (for each row and axis the mean mhat_{i,k}(E), the SD sigma_{i,k}(E), the
  5 % and 95 % quantiles and the z-score), summarised over the calibration
  rows.
- The edge rows (D-163 (4)). With eps in (0, 1/2) the edge width
  (`edge_eps`), row i of D_sim is an edge row if and only if

      min_{k in {1, ..., p}} min(theta_{i,k}, 1 - theta_{i,k}) < eps,     (R1)

  and an interior row otherwise. Let S_cal be the calibration rows whose
  theta_i is in the open box, S_edge and S_int its edge rows and its
  interior rows, n_cal = |S_cal| and pi_edge = |S_edge| / n_cal. For each
  fixed beta of the grid,

      C_E^edge(beta) = (1 / |S_edge|) sum_{i in S_edge} 1{ gamma_i(E) <= beta },
      C_E^int(beta)  = (1 / |S_int|)  sum_{i in S_int}  1{ gamma_i(E) <= beta },

  each written null when its set is empty. For each fixed beta, when both
  sets are non-empty,

      C_E(beta) = pi_edge C_E^edge(beta) + (1 - pi_edge) C_E^int(beta),     (R2)

  exactly (the two sets partition S_cal). A caution: an estimator equal to
  the simulator's posterior has C_E(beta) = beta, but it need not have
  C_E^edge(beta) = beta or C_E^int(beta) = beta, since the expected
  coverage is an average over the prior and not over a part of it. The
  split shows where a deficit sits, read through (R2); it is not two
  further calibration tests. The evaluation reports the three
  curves, pi_edge beside its value under the box prior, 1 - (1 - 2 eps)^p,
  and the mean of l_i(E) over the edge rows and over the interior rows of
  all of D_sim's rows whose theta_i is in the open box. eps = 0.006693 is
  the logistic function at -5, 1 / (1 + exp(5)): the flow's splines act on
  [-5, 5] in logit space (the decision log's open call on the flow's edge
  support), so a flow of that family puts almost no mass within eps of an
  edge. A project whose flow has the bound b sets eps = 1 / (1 + exp(b)).
- The contraction of axis k on D_sim is block 20's: 1 - (the mean over the
  calibration rows i of sigma_{i,k}(E)^2) / sigma_0^2, sigma_0^2 = 1/12 the
  box prior's marginal variance.
- The rows outside the open box [2026-11-23, after the final check, D-167]. A row of D_sim one of whose
  theta_{i,k} is <= 0 or >= 1 is outside the open box; a bank whose labels
  are clipped to [0, 1] and stored in float32 (the extractor's) can hold a
  coordinate exactly 0 or 1. For such a row q_E(theta_i | x_i) = 0 under
  the box prior, so: its log q_E(theta_i | x_i) is written -inf; it is in
  none of the quantities defined above over "the rows whose theta_i is in
  the open box" (the NLL, gamma_i and C_E, the edge split, the comparison
  of two ensembles), and their number is reported (`n_out_of_box`); it is
  in the quantities that do not use the density at theta_i and do not
  compare the draws with theta_i as a draw of the prior (the draws
  themselves, the recovery, the gate's roles). [2026-11-24, after the second check, D-168] It is
  also left out of SBC and TARP, which are over S_cal, the calibration
  rows in the open box, and no longer over every calibration row: a
  coordinate clipped to exactly 0 or 1 has an SBC rank fixed by the label
  (every draw of a continuous posterior lies on one side of it), so such a
  row would add a count to the first or the last rank whatever the
  calibration; with no row outside the box S_cal is every calibration row
  and the numbers are block 20's. It is never a reason to refuse the
  evaluation, unless no row of D_sim is in the open box (the sampler) or
  no calibration row is (the metrics). A theta with a NaN or an infinite
  coordinate is not such a row: the bank breaks its contract and the
  sampling refuses (exit 3), naming it.

**On a recording bank D_r.** On each row j: n_post draws theta^(s)_j ~
q_E(theta | x_j) as above, and, for each axis k in {1, ..., p}, the draws'
mean mhat_{j,k}(E), SD sigma_{j,k}(E), 5 %, 50 % and 95 % quantiles, and
the contraction

    chi_{j,k}(E) = 1 - sigma_{j,k}(E)^2 / sigma_0^2,     (R3)

for each fixed j and k: 0 when the window's posterior on axis k is as wide
as the prior, near 1 when it is much narrower, negative when it is wider.

- The well's summary (D-160). For a well g of D_r, with its windows J_g =
  {j : g(j) = g} and n_g = |J_g| >= 1,

      mix_g(theta) = (1 / n_g) sum_{j in J_g} q_E(theta | x_j),     (R4)

  for each fixed g (the s_g of D-160): the law of a draw taken from a
  window of g picked uniformly at random. It is estimated by the pooled
  draws {theta^(s)_j : j in J_g, s = 1, ..., n_post}, in which every window
  weighs the same, each having n_post draws; the evaluation reports their
  mean, SD and 5 %, 50 % and 95 % quantiles per axis. mix_g describes where
  the windows of the well put their mass. It is NOT the posterior of the
  well's parameter given all its windows: its width does not shrink with
  n_g, and no interval, statistic or test is built on it as if it were
  that posterior (D-160). Every output calls it "the mixture over the
  well's windows", never "the well's posterior".
- Coordinates. Every quantity above is in the unit box. The simulated
  bank's sidecar gives each axis its name and its declared range [lo_k,
  hi_k] in its declared coordinate (`param_names`, `bounds_theta`; the
  natural logarithm of the parameter on a log axis, the parameter itself
  elsewhere). A quantile u of axis k maps to lo_k + u (hi_k - lo_k) in that
  coordinate; means and SDs are not mapped. The metrics carry the names and
  the ranges beside the unit-box values.

**The gate (D-161, D-163 (4)).** For member m and a window x, z_m(x) =
h_{psi_m}(x) in R^{d_z} is the member's embedding of x
(`JointDSNNPE.encode`).

- The gate's sets. The networks of D_sim are dealt, by a seeded
  permutation, into four groups in the proportions `gate.donor_fracs` =
  (f_ref, f_rank, f_null, f_check), four positive numbers that sum to 1; no
  network is in two groups. Each set takes rows from its group in
  round-robin order -- every network's first row in global row order, then
  every network's second row, and so on -- until it has its cap or the
  group has no row left: the reference rows I_ref (cap `gate.n_ref`), the
  rank rows I_rank (cap `gate.n_rank`), the null rows I_null (cap
  `gate.n_null`) and the check rows I_chk (no cap). The last three take at
  most `gate.rows_per_donor` rows of a network; the reference takes any
  number. N_ref, N_rank, N_null and N_chk are the sizes reached. N_null
  below `gate.n_null_min` is a refusal.
- The kernel of member m. Let med_m be the median of the Euclidean
  distances between the pairs of distinct reference embeddings {z_m(x_i) :
  i in I_ref}, and (f_1, ..., f_{n_h}) = `gate.factors` (D-163 (4): (0.25,
  0.5, 1, 2, 4), n_h = 5). With the bandwidths h_{m,b} = f_b med_m, b = 1,
  ..., n_h,

      k_m(z, z') = sum_{b=1}^{n_h} exp( - ||z - z'||^2 / (2 h_{m,b}^2) )     (R5)

  for all z, z' in R^{d_z}: the sum of Gaussian kernels of
  `hpc/npe_misspec.py` (`bandwidth_grid` on the reference embeddings alone,
  `mmd2_multiscale`).
- The member's statistic of a window x is the biased squared MMD between
  the single point z_m(x) and the reference embeddings: for each fixed m
  and x,

      t_m(x) = (1 / N_ref^2) sum_{i in I_ref} sum_{i' in I_ref} k_m(z_m(x_i), z_m(x_{i'}))
               + k_m(z_m(x), z_m(x))
               - (2 / N_ref) sum_{i in I_ref} k_m(z_m(x_i), z_m(x)).     (R6)

  Its first term does not depend on x and its second is the constant n_h,
  so t_m(x) is large exactly when z_m(x) is, on average over the reference,
  far from the reference embeddings.
- The member's rank of x among the rank rows: for each fixed m and x,
  r_m(x) = (1 / N_rank) #{ a in I_rank : t_m(x_a) < t_m(x) }, in [0, 1].
- The ensemble statistic (D-161): for each fixed x,

      T(x) = (1 / M) sum_{m=1}^{M} r_m(x).     (R7)

  The ranks put the members' statistics on one scale, their embedding
  spaces and bandwidths being different.
- The p-values: for each fixed x and m,

      p(x)   = (1 + #{ v in I_null : T(x_v) >= T(x) }) / (1 + N_null),     (R8)
      p_m(x) = (1 + #{ v in I_null : t_m(x_v) >= t_m(x) }) / (1 + N_null),

  the repository's convention (`npe_misspec._pvalue`). x is flagged if and
  only if p(x) < alpha_gate, alpha_gate = `gate.alpha` (D-163 (4): 0.05, a
  value of the configuration). r_m(x), p_m(x) and T(x) are reported beside
  p(x).
- What p(x) is. I_ref, I_rank and I_null share no network, and T is the
  same function of (I_ref, I_rank) for a tested window x and for each row
  of I_null. If x is a draw of the simulator's law of one window,
  independent of the gate's rows, and the rows of I_null are draws of that
  law from distinct networks (`gate.rows_per_donor` = 1), then T(x) and
  (T(x_v))_{v in I_null} are exchangeable, and Pr[p(x) <= a] <= a for every
  a in [0, 1]. With several rows of a network in I_null its values are
  still draws of that law but no longer independent, and the bound holds
  approximately. p(x) is blind to what the embeddings do not carry
  (`hpc/npe_misspec.py`: the amplitude, lost where an embedding is
  normalised).
- The size given the gate's sets [added 2026-11-23, D-165,
  `[orchestrator]`: block 34's ISSUE on R34d]. The bound Pr[p(x) <= a] <= a
  averages over the draw of the gate's rows as well as of x. Once the rows
  of I_ref, I_rank and I_null are fixed, the probability that a window
  drawn from the simulator's law is flagged -- the gate's size q_gate -- is
  a number that depends on the null set. A window x is flagged, p(x) <
  alpha_gate, if and only if fewer than k_gate null values are at least
  T(x), with

      k_gate = ceil(alpha_gate (N_null + 1)) - 1,     (R9)

  that is, if and only if T(x) exceeds the k_gate-th largest of the null
  values. If the null rows are independent draws of the simulator's law of
  one window and T has no ties, then, over the draw of the null set,

      q_gate ~ Beta(k_gate, N_null + 1 - k_gate),     (R10)

  with mean m_gate = k_gate / (N_null + 1) <= alpha_gate and SD
  sqrt(m_gate (1 - m_gate) / (N_null + 2)) exactly, which is close to
  sqrt(alpha_gate (1 - alpha_gate) / (N_null + 2)) when m_gate is close
  to alpha_gate [corrected 2026-11-23 after the final check, S2: the
  second form alone stood here; at N_null = 50 and alpha_gate = 0.05,
  k_gate = 2, the SD is 0.0269, and the second form gives 0.0302]. The
  same holds for
  Pr[p(x) <= a] with k_a = floor(a (N_null + 1)) in the place of k_gate. T
  takes multiples of 1 / (M N_rank), so ties occur and make the gate
  slightly more conservative than (R10); t_m is continuous, and (R10) holds
  for the member's own p_m(x) without that caveat. At alpha_gate = 0.05 the
  central 95 % range of q_gate is [0.005, 0.107] with N_null = 50, [0.012,
  0.094] with 90, [0.033, 0.071] with 500 and [0.037, 0.064] with 1,000
  `[SANDBOX 2026-11-23]`: a small null set leaves the gate's actual size
  on a project poorly known. The size band is the central interval at
  `level` of (R10). k_gate = 0, that is alpha_gate (N_null + 1) <= 1, means
  that no window can be flagged, and is a refusal.
- The check rows I_chk are held-out simulated windows the gate did not
  use. The share of them flagged, with its exact binomial interval at
  `level`, estimates q_gate, the gate's size on the project's own simulator
  given its sets. It is reported beside k_gate / (N_null + 1) and the size
  band, not beside alpha_gate alone [amended 2026-11-23, D-165].
- On a recording bank: per window j, p(x_j) and the flag; per well g, the
  number and the share of its windows flagged; the same per class and for
  the whole bank. The windows of one well share a culture, so a well's or a
  class's share is reported beside the size band -- the range in which the
  flagged share of independent windows of the simulator's law lies, for a
  null set of this size [amended 2026-11-23, D-165: was "beside
  alpha_gate"] -- and no test is made on it (sec. 8, Q-R1).
- The synthetic evaluation's set-level embedding check (block 20's
  `embedding_overlap`) is not computed on recordings: its null is built
  from independent simulated windows, and it rejects a well-specified
  cohort whose windows share a culture (D-159; `hpc/npe_misspec.py`).

**The comparisons (D-163 (3)).** For a view with several searches the
comparisons are sec. 2.5's, on D_sim alone. For each search q whose
campaign has a term arm A(q): (a) q's ensemble of arm A(q) against its A1
reference -- q's own A1 ensemble when the view has one, otherwise the
ensemble of the view's one search of campaign S-A1; (b) for every pair of
searches (q, q') with A(q) = A2 and A(q') = A5, the ensemble of arm A2 of
q against the ensemble of arm A5 of q'. For a comparison (E, E'), the two
ensembles' rows of
D_sim must be the same rows; on the rows whose theta_i is in the open box,
d_i = l_i(E) - l_i(E'), its mean Dbar(E, E'), the bootstrap over the
simulated networks, sigma_seed and the two clauses are block 20's (E3, the
networks in the place of the donors). The numbers are reported with block
20's verdict line and nothing more is derived from them (D-163 (3)). A
comparison that cannot be made is named with its reason, never dropped
silently.

**The validation (D-162, D-163 (4)).** A synthetic project read as
recorded, with `sim_draw_rows_per_donor` null and the synthetic
evaluation's `seed`, `n_post` and `batch_rows`, gives the new path the rows
and the seeds of blocks 19 and 20: member k's draws on data set D start
from `seed_of(seed, D) + k` in both, D being `sim` or the bank's name. The
targets:

- in one environment (the sandbox; the reduced project of block 19's
  tests): the draws of every row of D_sim and of every cohort equal block
  19's bit for bit; l_i(E) equals block 19's within 1e-9 nats; gamma_i(E),
  C_E(beta), the SBC ranks and p-values, TARP's curves and p-values, the
  recovery arrays and the paired quantities equal block 20's within 1e-12
  (the same functions on the same arrays; the tolerance covers the order
  of floating-point sums only);
- on the cluster (the full synthetic project; `ev_rec_validate.py`):
  |l_i(new) - l_i(old)| <= 0.002 nats for every row i; the largest over
  the grid of |C_new(beta) - C_old(beta)| at most 0.01, for every
  ensemble; and (Claude's proposal, sec. 8 Q-R3) each axis's SBC verdict,
  the Holm family's rejections and TARP's two verdicts unchanged.

The quantities the existing evaluation does not have -- the edge split
(R1), (R2), the well's summary (R4), the gate (R5)-(R8) -- have oracles of
their own, fixed in their blocks' entries before the code.

### 2.7 The tool's contracts (roadmap `tool`, steps S1, S3, S4; D-158, D-172..D-175; added 2026-11-27)

Proposed 2026-11-27 from the user's decisions D-172 (order of work), D-173
(the card's form, the recording readers, the code's home), D-174
(safetensors) and D-175 (a self-contained bundle, the card's fields, the
working name `sbitool`); every value marked "Claude's" below is a proposal
until the user confirms it (sec. 8, Q-T*).

**What the tool is.** A Python package, `sbitool/` at the repository's
root, that turns a trained ensemble E of a recorded project (sec. 2.6) into
a file another person can load safely, and that, given such a file and a
recording, returns the ensemble's posterior on the recording's windows
with the gate (sec. 2.6, (R5)-(R10)), on a laptop, without the simulated
bank and without the cluster. Three objects are fixed here: the model
bundle, its card, and a recording.

**The sources the package imports (never copies).** Until the release step
(S10) packages them, `sbitool` imports three bodies of existing code
through one module, `sbitool/_sources.py`, which puts their folders on
`sys.path` and records each imported file's sha256:
- from this repository's `hpc/`: the estimator (`hpc/joint/stage2/
  joint_model.py`, `hpc/dsn/backbone.py` and `condition_space.py`), the
  runner's builders (`hpc/joint/stage3/run_joint_arms.py`:
  `make_backbone`, `FixedStatsSummary`), the IFR primitive
  (`hpc/dsn/generate_burst_data.py`: `compute_ifr_trace`), and the
  evaluation's functions (`hpc/pipeline/evaluation/ev_sample.py`,
  `ev_gate.py`, `ev_rec_metrics.py`);
- from a checkout of the Sbi-extractor (the folder named by the
  environment variable `SBIX_DIR`, or `SBIX_CLONE` in the tests): the
  per-electrode spike-train readers and the partition into subregions
  (`extractor/channel_subset_extraction.py`) and the windows of a trace
  (`export_windows/bank_real.py`: `windows_of`). Only blocks 43 and 44
  need it; a bundle is exported, checked and used on windows without it.
  [2026-11-27, at block 40's gate] The extractor's modules find this
  repository's `hpc/` tree through the environment variable
  `SBI_HPC_DIR`; `_sources.use("extractor")` sets it to this repository's
  `hpc/` when it is unset.
So that one function stays in one place: the density a bundle returns is
the pipeline's, by construction and by test (T41).

**The model bundle.** One zip file, `<name>.sbibundle`, whose members are,
exactly:

| member | content |
|---|---|
| `manifest.json` | the bundle's description, below |
| `CARD.md` | the card, below |
| `members/member_<k>.safetensors`, k = 0..M-1 | member k's `state_dict` (every tensor, the integer and boolean ones included), k in the ensemble's member order (sec. 2.5: the seeds ascending) |
| `evaluation/sim.json` | block 36's `<stem>_sim.json` of E: its quantities on the held-out simulations D_sim |
| `evaluation/gate.safetensors` | the gate's sets as block 35 embedded them: `z_ref` (M, N_ref, d_z), `z_rank` (M, N_rank, d_z), `z_null` (M, N_null, d_z), float32 |

No member is a pickle: the weights and the arrays are safetensors files,
the rest is JSON and Markdown; reading a bundle never calls `torch.load`,
`pickle` or `numpy.load`.

`manifest.json` has exactly the keys `MANIFEST_KEYS` (block 40):
- `schema`: "sbitool_bundle/1"; `name`; `created` (ISO time);
  `card_complete` (bool: false for a draft, see the card);
- `arm`, `M`, `p` (the number of parameters), `d_z` (the embedding's
  dimension), `W` (the window's length in samples of the IFR);
- `members`: a list of M dicts `file`, `sha256` (of the safetensors file),
  `seed`, `state_sha256` (the pipeline's digest of the member's state,
  `joint_model.state_sha256`, the checkpoint's `model_sha256`),
  `split_seed`, `split_hash`;
- `architecture`: what rebuilds a member with no data: `backbone` ("dsn",
  "fixed-stats" or "fallback"), `embedding_size`, the encoder's
  `depth_exponent`, `width_multiplier`, `block_family`, `head_fusion`,
  `dropout`, `head_pool_ops`, the flow's `hidden_features`,
  `num_transforms`, `num_bins`, `hidden_layers`, `z_score_theta`,
  `z_score_x`, and `backbone_repr` (the checkpoint's, compared after the
  rebuild);
- `parameters`: `names` (p), `units` (p; "1" for a dimensionless one),
  `bounds` (p pairs [lo_k, hi_k] in the declared coordinate, or null);
  the networks work in the unit box, u in (0, 1)^p, and the declared
  value of axis k is lo_k + u_k (hi_k - lo_k);
- `extraction`: null, or exactly `EXTRACTION_KEYS`, the parameters that
  turn a recording into windows (below);
- `gate`: `factors`, `alpha`, `n_ref`, `n_rank`, `n_null`, `k_gate`,
  `size_mean`, `size_lo`, `size_hi`, `level` (block 36's `gate` of
  `sim.json`, so that the tool flags as the evaluation did);
- `sample`: `n_post`, `seed` and `batch_rows` [`batch_rows`, 2026-11-27,
  D-179: the windows per batch through a member, block 35's; a member's
  draws depend on it, measured], the defaults of an inference;
- `provenance`: the project's tag, the ensemble's name, the search and the
  campaign, the sha256 of the best fit's `BEST_FIT_COMPLETE.json` and of
  block 35's `<stem>_sample.json`, the repository's commit, and the sha256
  of each source file the rebuild imports;
- `software`: the versions of python, torch, sbi, zuko, numpy,
  safetensors at the export;
- `files`: a dict from the name of each fixed member other than the manifest
  (`CARD.md`, `evaluation/sim.json`, `evaluation/gate.safetensors`) to its sha256;
  the networks' files are digested in `members[k].sha256` [2026-11-28, the tester's
  S2: as built].

**Rebuilding a member from a bundle.** With theta in the unit box and x a
window in R^W, member m's density q_m(theta | x) is the pipeline's (sec.
2.5, (E1) for the mixture q_E): the tool builds the network with the
pipeline's own builders from `architecture` and loads the safetensors
state into it. The builders need example tensors only for their shapes
(the runner passed 64 training rows); the tool passes tensors of the same
shapes drawn from a fixed generator, and the statement that this changes
nothing is a test, not an assumption: after the load the state's digest
equals the manifest's `state_sha256`, and log q_m(theta | x) and the draws
under a fixed seed equal the pipeline's rebuild (block 19's
`ev_models.rebuild`) bit for bit (T41a).

**The card.** `CARD.md`: a header between two `---` lines, in YAML, then
free Markdown. The header has exactly the top-level keys `CARD_KEYS`
(block 42), the user's eight (D-158) and the sharing fields (D-175):
`title`; `scientific_question`; `simulator` (`model_type`, `reference`,
`fixed_parameters`: name -> value with its unit, `searched_parameters`:
the manifest's `parameters.names`); `training` (`arm`, `members`,
`n_networks`, `n_windows`, `split_fracs`, `hyperparameters`: the
runner's, `hyperparameter_search`: how they were chosen); `extraction`
(the manifest's, or "not applicable"); `references`; `authors` (each
`name`, `contact`); `licence`; `cite`; `data` (`simulated_bank` and
`recordings`: each `location`, `public`, and for the bank `size_bytes`
and `sha256` or null); `software`; `limitations`. The export pre-fills
every field the pipeline's records hold and writes the literal `TODO` in
the others. A card is complete when every key is present, no value is
empty or `TODO`, and `training.arm`, `training.members`,
`simulator.searched_parameters`, `extraction` and `software` agree with
the manifest. A bundle whose card is not complete is a draft
(`card_complete` false): it can be made and used locally, and the hub
(step S8) will not take it.

**A recording.** For the tool a recording is, whatever file it came from
(block 40, `Recording`): for each electrode e of a set of electrodes, the
sorted sample indices of its detected spikes, s_{e,1} < s_{e,2} < ...,
integers in [0, n_samples); the raw sampling rate fs_raw [Hz]; n_samples,
so that the duration is T_rec = n_samples / fs_raw [s] and the spike
times are t_{e,j} = s_{e,j} / fs_raw [s]; and the device's layout: each
electrode's position (x_e, y_e) [um], and when the device is a regular
grid its `grid_width`, `index_base` and `pitch_um`. Two adapters produce
it (D-173): `ptrain` (the extractor's per-electrode spike-train files,
its two declared formats) and `spikeinterface` (the files that library
reads).

**From a recording to windows (the extraction).** `EXTRACTION_KEYS`:
`electrodes_per_subset` (n_e), `n_subsets`, `mfr_threshold_hz`,
`w_size_s` (the bin width Delta t), `gaussian_window_s` (sigma_sm),
`grid_width`, `index_base`, `pitch_um`, `fs_raw_hz`, and `detection`
(null for a recording that arrives as spike trains; for raw voltage
exactly `DETECTION_KEYS` (D-176): `method` "by_channel", `peak_sign`
"neg", `detect_threshold` (the multiple of the channel's noise level, its
MAD-based estimate, below which a negative peak is a spike; 5),
`exclude_sweep_ms` (the least time between two peaks of one channel; 1.0):
the arguments of SpikeInterface's `detect_peaks`, written by the bundle's
author). The rule is the extractor's, imported:
the electrodes are partitioned into `n_subsets` disjoint subregions of
n_e electrodes (`partition_subregions`, its mean-firing-rate threshold);
for a subregion S with electrodes e in S and for each bin index k in
{0, ..., K-1}, K = the number of bins of width Delta t in T_rec, the
pooled instantaneous firing rate is the extractor's (EXTRACTOR_USAGE.md,
eq. (3)): the spikes of the n_e electrodes counted in bin k, smoothed
over k by the Gaussian of width sigma_sm (`compute_ifr_trace`) and
divided by n_e. Its unit [corrected 2026-11-28, D-180, after the
tester's F1; this section first said "divided by n_e Delta t [Hz per
electrode]", which neither eq. (3) nor the code does]: x_S[k] is in
spikes per bin of width Delta t per electrode, the unit of the training
banks' windows (`scale_convention` "per_unit_mean"); x_S[k] / Delta t is
the same quantity in Hz per electrode. No number of the tool changes
with this correction: the windows were and are the extractor's. K and
the bins are the imported function's, and where its arithmetic differs
from this section's wording the imported function governs, because a
recording must be cut exactly as the training banks were (D-180):
K = int(T_rec / Delta t) in float64, which is floor(T_rec / Delta t)
except where the quotient falls just below an integer (2900 samples at
10 kHz with Delta t = 0.01 s: K = 28, not 29; the tester's F3), and the
bins' edges are k Delta t in float64 (and `numpy.histogram` closes its
last bin, so that a spike exactly at K Delta t is counted in bin K - 1
rather than dropped; the second tester's S2), so that a spike exactly on
an edge can be counted in the bin before it (the tester's F4: about 0.13 % of
the spikes of a recording sampled at a multiple of 1 / Delta t). The
trace x_S in R_{>=0}^K is cut
into floor(K / W) consecutive windows of W bins from its start
(`windows_of`), each a row x in R^W of the bundle's networks. The tool
adds nothing to this rule and changes nothing in it; a bundle whose
`extraction` is null accepts windows only.

**An inference.** Given a bundle of ensemble E and n windows x_1..x_n (of
one recording, or given as an array): for each window, n_post draws of
q_E(. | x_i), n_post / M per member in member order with the seeds block
19 uses (member k: seed + k), their densities, the members' embeddings,
the summaries of block 36 (per window: mean, SD, quantiles 0.05, 0.50,
0.95 per axis, contraction (R3); per recording: the mixture over its
windows (R4), called so and never "the recording's posterior"), and the
gate against the bundle's own sets ((R5)-(R8), p(x_i) and the flag p <
alpha_gate, read against the bundle's size band (R10)). With the seed
block 35 used for a recording bank r, the result on that bank's windows
is block 35's and block 36's, file for file (T45a): the tool is another
route to the same numbers, not another estimator.

### Notation

| Symbol | Meaning | Type / shape | Units |
|---|---|---|---|
| s, dep(s), done(s) | a stage, its dependencies, its done-predicate | -- | -- |
| n_train, n_d | training rows; distinct donors in the bank | int | rows; donors |
| B_sim, P, N_ep, S | batch size, passes, epochs, steps per epoch | int, float, int, int | rows; passes; epochs; steps |
| (f_tr, f_sel, f_rep) | split fractions by donor | floats summing to 1 | -- |
| theta, x | a parameter vector in the unit box; a window | (p,), (W,) float32 | -- ; Hz (IFR) [corrected 2026-11-12: in the bench banks of the synthetic data set, x is the Gaussian-smoothed population spike count per bin of w_size = 1/f_s s divided by the units (sidecar `scale_convention` per_unit_mean; `bench_burst_provider.py`), so spikes per unit per bin and x f_s in spikes/s per unit; the extractor's banks not checked here] [checked 2026-11-28, D-180: the extractor's windows are the same quantity, the smoothed spike count per bin of w_size divided by the subregion's n_e electrodes (`subregion_ifr`), so spikes per bin per electrode there too; "Hz (IFR)" at the head of this cell is superseded by both corrections] |
| q_omega(theta \| h_psi(x)) | the flow's density at theta given the encoder's summary of x | density on (0, 1)^p | -- |
| l_{m,i}, L_bar | member m's NLL on report row i; the ensemble's mean NLL | float64 | nats per row |
| L_sel(e), e_best | selection-split NLL after epoch e; the best epoch | float; int | nats per row; epochs |
| theta_k, a_k, [L_k, U_k] | sec. 2.4: bench coordinate k, its physical value, its range | float in (0, 1); per axis | -- ; the axis's units |
| K_L, K_F | the bench's class-bearing and free axes | index sets {0..6}, {7, 8, 9} [amended 2026-11-16: the bank's own, from its sidecar (block 25); `synth.json` v4: {0..7}, {8, 9}] | -- |
| c, c_d, C, n_c | a condition; donor d's condition; the number of conditions; donors per condition | int | -- |
| mu_c, m_c, s, tau | condition c's centre; the DSN's regular centre; the spacing factor; tau_ov | (7,), (7,), float, float | -- (unit box) |
| N_S, N_R, J_S, J_R | arm S's and arm R's donors; windows per well in each arm | int | donors; windows |
| pi, delta, f | gap severity; range shift; contaminated fraction | float | --; fraction of an axis's width; fraction of windows |
| A_theta, A_win, A_don | theta ceiling; one-window and one-donor accuracy of the condition | float in [0, 1] | -- |
| L_pseudo_real, n_R | mean NLL of arm R's withheld theta over its windows; arm R's windows | float; int | nats per row; windows |
| A(q), a(t), abar | sec. 2.5: search q's term arm; trial t's arm; the arm a search's finalist is not | arm names A1, A2, A5 | -- |
| t*_q, t*_{q,a} | search q's finalist; the best trial of arm a in q, both on L | trial | -- |
| E_{q,a}, q_{q,a,m}, q_{q,a} | arm a's ensemble in q; its member m's density; the mixture (E1) | densities on (0, 1)^10 | -- |
| D_sim, D_R0, D_R3, D_R3+ | the evaluation's data sets | rows (x_i, theta^ref_i) | -- |
| theta*_d, theta^dag_d | donor d's withheld parameter; its pseudo-true parameter under the range shift | (10,) float64 | -- (unit box) |
| n_post, theta^(s)_i | posterior draws per row; draw s for row i | int; (10,) | draws; -- |
| l_i(E), lbar_d(E) | ensemble E's NLL at row i; its donor mean | float64 | nats per row |
| gamma_i(E), C_E(beta) | the HPD rank of theta^ref_i; the expected coverage at credibility beta (E2) | float in [0, 1] | -- |
| r_{i,k}(E), zeta_{i,k}(E) | the marginal rank on axis k; the z-score | int in {0..n_post}; float | --; marginal SDs |
| mhat_{i,k}, sigma_{i,k}, sigma_0^2 | posterior mean and SD on axis k; the box prior's variance 1/12 | float | -- (unit box) |
| Delta_d(E, E'), Dbar(E, E') | donor d's paired difference of the two ensembles' donor NLLs (E3); its mean over the donors | float64 | nats per window |
| sigma_seed | the larger of two arms' across-seed SDs of their members' L_pseudo_real (G.2) | float64 | nats per window |
| kappa; E_kappa, E'_kappa | [2026-11-16, D-136] a comparison (sec. 2.5, "The comparisons"); its ensemble and its reference, E and E' of E3 | name `<q>` or `<q>_vs_<q'>`; ensembles `<q>:<arm>` | -- |
| p, W, M, d_z | sec. 2.6: the axes of the simulated bank's theta; the window length; an ensemble's members; the dimension of a member's embedding | int | axes; samples; members; -- |
| B_S, D_sim, D_r | sec. 2.6: a project's simulated bank; an ensemble's held-out simulated rows (the report split); the rows of its recording bank r | rows | -- |
| d(i), g(j), c(j) | the simulated network of row i of D_sim (the bank's `donor`); the well and the class of row j of a recording bank | int64 | -- |
| k_draw, n_d | the draw rows per network on D_sim (`sim_draw_rows_per_donor`; null: every row); the number of draw rows | int or null; int | rows |
| eps; S_cal, S_edge, S_int; n_cal, pi_edge | the edge width (`edge_eps`); the calibration rows with theta in the open box, its edge rows and interior rows (R1); the size of S_cal and the share of edge rows in it | float in (0, 1/2); row sets; int, float in [0, 1] | unit box; --; rows, -- |
| C_E^edge(beta), C_E^int(beta) | the expected coverage of ensemble E over S_edge and over S_int, for each fixed beta (R2) | float in [0, 1] or null | -- |
| chi_{j,k}(E) | the contraction of window j of a recording bank on axis k (R3) | float <= 1 | -- |
| J_g, n_g, mix_g(theta) | the windows of well g; their number; the mixture over the well's windows (R4), for each fixed g | row set; int >= 1; density on (0, 1)^p | --; windows; -- |
| z_m(x) | member m's embedding of window x | (d_z,) float32 | -- |
| I_ref, I_rank, I_null, I_chk; N_ref, N_rank, N_null, N_chk | the gate's reference, rank, null and check rows, subsets of D_sim that share no network; their sizes | row sets; int | --; rows |
| med_m, h_{m,b}, n_h, k_m(z, z') | member m's median distance between reference embeddings; its bandwidths f_b med_m; their number; its kernel (R5) | float > 0; float > 0; int; float in (0, n_h] | embedding units; embedding units; --; -- |
| t_m(x), r_m(x), T(x) | member m's statistic of window x (R6); its rank among the rank rows; the ensemble statistic (R7), for each fixed x | float >= 0; float in [0, 1]; float in [0, 1] | -- |
| p(x), p_m(x), alpha_gate | the gate's p-value of window x and member m's own (R8); the gate's threshold (`gate.alpha`) | float in (0, 1]; float in (0, 1) | -- |
| k_gate, q_gate | [2026-11-23, D-165] the rank among the null values that a flagged window's T must exceed (R9); the gate's size given its sets, random through the null set (R10) | int >= 1; float in (0, 1) | -- |

### 2.8 The layers of use (roadmap `tool`, step S7a; D-158 (5), D-183, D-184; added 2026-11-28)

A design section, no code: what the viewer (S6) shows and lets a user change, in which layer. It binds
S6's SPEC entries. Status: the layers and the four choices below are the user's (D-184); the assignment
of each item to a layer is Claude's proposal, to be corrected on the mockups before S6 starts.

**The layers.** Four, each containing the one before it; one switch in the interface chooses the layer,
and nothing a lower layer shows changes its value when a higher one is chosen.

| layer | who it is for | what they come to do |
|---|---|---|
| guided | an experimentalist with a recording and a published model | load both, run, read one result per recording |
| standard | a computational user | choose windows and subregions, read each window's posterior and the checks, compare recordings |
| expert | someone auditing a model or a result | see every array, every setting and the model's training |
| author | someone preparing a model for others | export a bundle, write and check its card, verify it |

**The user's choices (D-184).** (1) Four layers, the fourth for the author. (2) The viewer shows the
model's training beyond the card: the export also packs each network's training curve and a summary of
the hyper-parameter search's trials into the bundle (a change of the bundle's format; a block of its
own, to be specified before S6). (3) The guided layer has a few named settings, not none: the number of
posterior draws and the part of the recording to use. (4) The guided layer's headline for one
recording is, per parameter, the pooled median and 90% interval over all windows with the windows' own
medians drawn beside it, and the share of windows the gate flags; the windows' detail is one step away.

**Standing decisions the interface keeps.** The pooled result over the windows of a recording or a well
is the equal-weight mixture of the windows' ensemble posteriors, a description of where the windows put
their mass, not the posterior of the culture given all its windows (D-160): every layer labels it so.
The share of windows the gate flags is shown as a number, with no verdict on the recording (D-172). The
extraction and the spike detection are the bundle's and are never editable (D-176). A device that does
not fit the bundle is refused, naming both (D-176).

**Inventory.** Every setting and every output the tool exposes today (blocks 40-47), with the first
layer that shows it (G guided, S standard, E expert, A author). "new" marks what S6 or a block before
it has to add.

| item | first layer | note |
|---|---|---|
| the bundle file; the recording file and its format | G | the tool asks for a layout (grid or positions) only when the file does not carry one |
| number of posterior draws n_post | G | a multiple of the bundle's M members; default the bundle's |
| part of the recording to use (a time range) | G | new: a selection of windows by their start time |
| the recording's reader options (grid, positions, sampling, reader) | S | block 43's adapter options |
| which subregion and which window is looked at | S | G shows one example window |
| seed; batch size | E | defaults the bundle's (D-179) |
| the extraction and the detection (electrodes per subregion, number of subregions, rate threshold, bin width, smoothing width, grid, pitch, detection) | S (summary), E (all) | read-only at every layer |
| the device with its electrodes, those used, the subregions | G | block 46 |
| a window: the spike raster and x [spikes per electrode per bin] | G (one example), S (any) | block 46 |
| per parameter, the pooled median and 90% interval over the windows, with the windows' medians beside it | G | block 45's `mix_*` and `q50`; labelled a description (D-160) |
| the share of windows the gate flags | G | block 45's `gate_flag`; no verdict (D-172); with one sentence on what a flag means |
| what the model is: the card's title, question, simulator, the parameters it infers and their units | G | block 42 |
| the whole card | S | block 42 |
| each window's posterior: marginals and pairs, median, intervals, contraction | S | block 45's `draws`, `mhat`, `sigma`, `q05`, `q50`, `q95`, `contraction` |
| the gate per window: statistic T, p-value, flag | S | block 45's `gate_T`, `gate_p`, `gate_flag` |
| the model's held-out evaluation: coverage, SBC, TARP, NLL, recovery | S (summary), E (tables) | the bundle's `evaluation/sim.json` (block 36) |
| several recordings (wells, conditions) side by side | S | new |
| the results as files (`posterior.npz`, `posterior.json`, the figures) | S | block 45 |
| per member: the gate's values, which member drew each draw, the member's own log-density; the embeddings z | E | block 45's `gate_t`, `gate_r`, `gate_p_member`, `draw_member`, `logq_draws_own`, `z` |
| the model's training: each network's training curve, the search's trials | E | new: not in a bundle today (choice (2)) |
| the manifest, the bundle's sha256, the bundle check, the software's versions | E | block 41 |
| the command line that reproduces what the interface did | E | new |
| export; the card's template, editing and check; `bundle-check --rebuild`; `verify`; the units and extraction files | A | blocks 41, 42, 47 |

**Display of a parameter (proposal, to be confirmed on the mockups).** The networks work in the unit
box: u_j in [0, 1] for each inferred parameter j in {1, ..., p}. Where the bundle declares bounds
[lo_j, hi_j] for axis j, the guided and standard layers show theta_j = lo_j + u_j (hi_j - lo_j), the
banks' map, with the bundle's unit for axis j; where it declares none they show u_j. The expert layer
can show either.

**Not yet built, with or before S6.** The time-range selection of windows; the comparison of several
recordings; the training curves and the trials' summary in the bundle; the reproducing command line;
the third check of the roadmap's S6 summary, the predictive check, which is wave 2 of S2 and does not
exist: until it does the viewer shows two checks (the model's held-out calibration and the per-window
gate).

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
| `config_version` | int | 11 (v11, 2026-11-17, D-148: the keys `searches`, `replicate.pairs`, `replicate.group_key`, `real.cohort_map`, `bank.drop_partial_topologies` below) [v10's text follows] 10 (v10, 2026-10-19 17:44, D-101: values only -- `search.gp_seed` 1, `resources.max_cores` 400 [2026-10-30, D-115: Giulia's `resources.max_cores` 300, a value under v10, the version unchanged]; v9, 2026-10-19: `resources.max_cores`, D-099; v8, 2026-10-18: the `driver` block, D-098; v7, 2026-10-18: `search.resubmit_node_faults` and `best_fit.resubmit_node_faults`; v6, 2026-10-18: `bank.verify` and `bank.max_rerun_rounds`; v5, 2026-10-16: the two queue keys below) |
| `dataset` | str | `[a-z0-9_]+`; the `<tag>` of the folder and of the artefact names |
| `paths.{sbix_dir, sbi_dir, work_root, banks_root, runs_root, tune_root}` | str | absolute paths |
| `envs.{extract, train}` | str | conda environment names |
| `real.{cohort_config, cohort_tag, extract_root}` | str | `cohort_config` an absolute path |
| `real.cohort_map` | str or null | v11 (D-148 (4)): an absolute path of the cohort map (Sbi-extractor block 9: one row per well of the manifest -- `out_dir`, `donor`, `rep_group`, `include`), written into the bank profile as `cohort_map`; null: no map (donor = well; rep_group = donor) |
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
| `bank.drop_partial_topologies` | bool | v11 (D-148 (5)): written into the bank profile; true drops whole every topology with fewer simulations than its task dispatched (Sbi-extractor block 10) |
| `split.fractions` | [float, float, float] | each in (0, 1), sum 1 within 1e-9 |
| `searches` | null or [object] | v11 (D-148 (1), block 29): null -- the single search of `search.campaign`; a list -- each item exactly {`name` `[A-Za-z0-9][A-Za-z0-9_-]*`, `campaign` among the campaigns below, `max_cores` an integer >= 1 or null}; the names unique, the campaigns unique; an integer `max_cores` at least the larger of `search.trial.ncpus` and `best_fit.ncpus` and at most `resources.max_cores` - `driver.ncpus`. With a list, `search.campaign` and `search.arm` are refused (each search names its campaign; the arm is the campaign's). [amendment 5, 2026-11-17, the final check's S1 and F5: a list is non-empty (an empty list is refused); every name rule of the configuration -- `name` here, `synthetic.searches[].name`, `synthetic.real[].name`, the dataset tag, the axis names of `search.space_overrides` -- matches the whole string (`\Z`, not `$`, so a trailing newline is refused) [corrected 2026-11-17, the second final check's S1: "the label axes" named no string rule]. The value formats (`hh:mm:ss`, PBS memory, the queue and environment names) still end in `$`: a value with a trailing newline loads (the second final check's F1, a NOTE; a JSON value needs an explicit \n), a known gap left to the next change of config.py] |
| `replicate.pairs` | str | v11 (D-148 (3), block 31): `within_group` (the standard: two different real windows of one group, within-well pairs allowed) or `cross_well` (block 14's: two rows of one group in two different wells) |
| `replicate.group_key` | str | v11: the real bank's id array that names a row's group: `rep_group` (the standard; Sbi-extractor block 9), `donor`, `batch` or `well` |
| `search.{campaign, arm}` | str | `campaign` in {S-A1, S-A2, S-A5, S-A25} [and S-A2-only, S-A5-only since block 26]; `arm` in the runner's arms; [v11] only with `searches` null |
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
| `resources.max_cores` | int | >= 1 (D-099; v9): the most cores the pipeline's jobs request at once, queued, held or running, the sequencer's job included; at least `driver.ncpus` + the largest single job of the configuration. Giulia: 300 (D-115, 2026-10-30; 400 under D-101, 200 under D-099); LindDUP: 200 (D-109) |

**The synthetic kind (block 15; added 2026-11-03).** `"kind": "synthetic"`
selects this schema: every key above except `real.*`, `sim.*`, `freeze.*`,
`bank.*`, `search.campaign`, `search.arm`, `paths.sbix_dir` and [v11]
`searches` (it keeps `replicate.*`: synth.json pins `cross_well`, `donor`,
the rule it ran with), plus

| Key | Type | Constraint |
|---|---|---|
| `kind` | str | `synthetic`; absent for a recorded data set (any other value is refused) |
| `synthetic.builder.provider` | str | `bench` (the bench generator, D-119) |
| `synthetic.builder.{fs, T_win, tau_ov, centre_scale}` | float | > 0 (Hz, s, --, --); a `centre_scale` other than 1 needs `class_centres` `regular` |
| `synthetic.builder.{n_neurons, wells_per_donor, donors_per_batch, n_classes}` | int | >= 1; `wells_per_donor` >= 2 (the builder's two realisations per theta; A5's pairs); `n_classes` >= 2 (F-t) |
| `synthetic.builder.class_centres` | str | `simplex` or `regular` |
| `synthetic.sim.{seed, n_theta, n_windows, traces_per_shard}` | int | `seed` >= 0, the rest >= 1; `traces_per_shard` a multiple of `wells_per_donor` that divides `n_theta * wells_per_donor` |
| `synthetic.real` | [object] | non-empty; each exactly {`name` `[A-Za-z0-9]+`, `seed` int >= 0, `donors_per_class` int >= 1, `n_windows` int >= 1, `pi` a number in [0, 1], `gap_modes` a non-empty list of distinct modes among `range_shift`, `drift`, `contamination`}; the names unique |
| `synthetic.searches` | [object] | non-empty; each exactly {`name` `[A-Za-z0-9][A-Za-z0-9_-]*`, `campaign` in {S-A1, S-A2, S-A5, S-A25}, `real` the name of an entry of `synthetic.real`}; the names unique, no (campaign, real) twice |
| `synthetic.bank_job.{ncpus, max_resubmits}` | int | >= 1; >= 0 |
| `synthetic.bank_job.{mem, walltime, queue}` | str | PBS memory, `hh:mm:ss`, a queue |

`synthetic.sim.seed` and the seeds of `synthetic.real` are pairwise distinct
(D-120, D-121; F-aw), and `resources.max_cores` holds the sequencer's job
and the largest of `search.trial.ncpus`, `best_fit.ncpus` and
`synthetic.bank_job.ncpus`. A list item may carry `_` documentation keys
too. The tools that read a recorded data set's single search or its S1-S6
(`tools/inputs_check.py`, `perrow_check.py`, `purge_outputs.py`,
`search_report.py`, `search_reset.py`) refuse a synthetic configuration:
`load_config(path)` refuses one unless called with `synthetic_ok=True`, as
`pipeline.py` calls it.

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
[amended 2026-10-29, D-111] `JobStatus.exec_host`: the first host of the
record's `exec_host` (`<host>/0*48` -> `<host>`), None when absent; the
stages keep it as a finished job's `host`.

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
    them. [2026-11-17, D-151: Claude pushes to Sbi-extractor directly
    (branch `feat/real-arm-parity`); Astro-Neuron-Network unchanged.]
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
  - D-099: the cores requested by the pipeline's jobs that are queued, held
    or running, the sequencer's own job included, never sum to more than
    `resources.max_cores` (200 for Giulia, raisable [400 from 2026-10-19,
    D-101; 300 from 2026-10-30, D-115]); a job waits until it
    fits; nothing submitted is deleted for it (block 1, amended 2026-10-19
    (d); blocks 2, 3 and 7, amended 2026-10-19).
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
  - D-138..D-151 (LindDUP, 2026-11-16 and 17): the simulated arm is the
    campaign `lcdup15_rho1300_neur_n01` (p = 14: 13 axes and conn_prob;
    180 s), trained as it is; arms A1, A2 and A5 each searched and fitted
    (block 29), 148 cores each, on intel, Giulia's budget; a search stopped
    by hand (block 30) and no clock cap (`max_total_walltime_h` null);
    topologies cut short dropped whole (Sbi-extractor block 10); the real
    cohort's donor is its condition, an A5 pair two windows of one condition
    x batch group drawn at random, within-well pairs allowed (block 31), the
    five copied wells left out (the cohort map, Sbi-extractor block 9); the
    term weights searched as now (D-149). D-148: these features are the
    pipeline's standard; a new project gets `replicate` = within_group,
    rep_group and a cohort map.
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
- Amended 2026-10-19 (d) (D-099; the user, 11:50: "You will have available
  200 cores so put this cap (that can be risen) for the whole pipeline"):
  the cores the pipeline's jobs request at once are counted and capped.
  - `pipeline.job_cores(job) -> int | None`: the job's `ncpus`, recorded at
    submission since this amendment, else read from its qsub line: `-l
    select=K:ncpus=N` gives K x N (K = 1 when absent; a `select=...,
    walltime=...` argument is read the same way), times, for an array `-J
    a-b[:s][%c]`, min(c, count) with count = (b - a) // s + 1 (no `%`: all
    of them); None when neither says.
  - `pipeline.cores_in_flight(state) -> (cores, n, unknown)`: over the
    jobs not marked finished of every stage not done or skipped (a done
    stage's record its polls never marked finished holds no cores), the
    sum of the footprints of the jobs recorded after no such job: a job's
    `job_cores`, or, when jobs are recorded `after` it (a gate, an
    aggregation: they start when it ends), the larger of its cores and the
    sum of theirs (never both at once); n, the jobs counted; `unknown`,
    the ids of the jobs no record gives the cores of (counted 0, named by
    the dry run and `--status`).
  - `Submitter(pbs, state, stage, log, cfg)`: `budget()` =
    `resources.max_cores` - `driver.ncpus` (the sequencer's own job is
    always counted); `in_flight()`; `free()` = budget - in flight;
    `check(n)` raises `CapExceeded` (a RuntimeError) when n exceeds the
    whole budget, a job that can never start (`<stage>: a job of <n> cores
    can never start: the stages' jobs share <b> cores (resources.max_cores
    <m> less the sequencer's job, <d>); raise resources.max_cores ... or
    ask fewer cores (D-099)`); `fits(n)` asks `check` first, then whether n
    <= free; `submit(argv, role, ..., ncpus=None, after=None)` records
    `ncpus` (given, else read from the argv) and `after`, and raises
    `CapExceeded` when the job does not fit -- the stages ask `fits` first,
    so this is a guard; `record(job_id, role, cmd, ..., ncpus=None,
    after=None)`: a launcher's job, with the cores the stage worked out
    before it ran the launcher (no check here). A stage's start asks
    `check` of the jobs it submits later (S2's and the bank's array with
    its gate), so it stops before its first job. `run` fails the stage on
    a `CapExceeded` from its poll as from its start (`STOP: stage <s>
    failed: ...`), never a crash of every driver. Without a configuration
    (the unit tests' Submitter) everything fits.
  - `stages.Deferred`: raised by a stage's `start` when its first job does
    not fit; `run` leaves the stage pending, logs `<stage>: waits for cores:
    <reason>` once per reason (kept as the stage's `waiting_note`), and
    calls `start` again at the next tick.
  - Config v9: `resources.max_cores`; the load refuses a value below
    `driver.ncpus` + the largest job the configuration sizes
    (`search.trial.ncpus`, `best_fit.ncpus`, `sim.ncpus`,
    `bank.verify.ncpus`): `resources.max_cores: <n> cannot hold the
    sequencer's job (driver.ncpus <d>) and the largest job (<key> <v>)`.
    Giulia: 200 `[user]`. Raising it: the value, `--stop`, a submission
    with `--accept-config-change`.
  - The dry run and `--status` print `cores: at most <cap> at once
    (resources.max_cores, D-099): the sequencer's job <d>, the stages' jobs
    <b>; in flight per the state file: <n> in <k> job(s)` (and the unknown
    ones).
  - Tests: `tests/test_smoke_block1c.py`.
- Amended 2026-10-19 (e) (found on davinci 2026-10-19 15:31; Claude's fix,
  `[proposed]`): `run` saves the state once its refusals are checked and
  before its first tick (`_keep_prepared`), so an accepted configuration
  change and the `--retry` resets reach the state file even when the first
  tick raises; a refused run writes nothing. Before, the first save came
  with the first stage's poll: driver 1852795 accepted v9 and raised in its
  first tick (block 7, an unreadable record); its successor, without the
  one-time flag, read the old digest and refused (`REFUSE: the configuration
  changed ...`), and the chain ended. Test oracle (`test_smoke_block1b.py`,
  edge cases): a configuration change accepted at submission and a first
  driver that raises in its first tick (the synthetic graph's
  `SYNTH_CRASH_ONCE`): one `raised`, no `REFUSE`, `PIPELINE DONE`, the
  state's digest the file's; mutation `early_save` (the save removed) -> the
  check FAILS.
- Amended 2026-10-21 (f) (found on davinci 2026-10-21: driver 1854083 on
  dvnode001 lost the scheduler at 11:59 -- `cannot run qstat: [Errno 12]
  Cannot allocate memory`, then `qstat: illegally formed job identifier`
  at every tick, the node's fault, the node offline by 15:30 -- and sat
  through 105 failing ticks, every finished trial unrecorded and its slot
  idle, until the user's `qdel` at 15:11; the user's "Solve this";
  Claude's design, `[proposed]`): a driver that cannot reach the scheduler
  hands over to its successor instead of waiting for its budget. (1) `run`
  counts the ticks in a row in which a running stage's poll raised
  `SchedulerError`; a tick in which no poll raised resets the count to 0
  and is the only kind of tick that calls `on_tick` (so clears the unclean
  count, (5) above: a driver whose every tick fails never clears its
  predecessors'). (2) When the count reaches SCHEDULER_FAIL_TICKS (10;
  `PIPELINE_SCHEDULER_FAIL_TICKS` overrides it for the tests -- about 17
  min on davinci: a failing tick is 3 qstat attempts 20 s apart plus the
  60 s poll) and the driver's `hand_over()` returns True, `run` ends at
  that tick boundary with EXIT_BUDGET and `_END["why"] = "scheduler"`
  (`the scheduler was unreachable for <n> consecutive tick(s) (<last
  error>); stopping (resumable): the successor driver job continues`),
  after the stop request is honoured and before the time budget. (3)
  `hand_over()` is True when a successor is queued, or when one can be
  submitted now (logged: `its successor <id> was submitted now`); when
  none can be, it is logged once (`no successor could be submitted (...);
  this driver keeps trying until its time budget`) and the driver keeps
  ticking -- it never exits without a successor. A `run` without a
  `hand_over` (a foreground run, the tests) never ends for this reason.
  (4) The end is noted as an unclean end (`driver <me>: the scheduler was
  unreachable for <n> consecutive tick(s); unclean end <k> in a row; the
  successor <id> continues`, exit 0), so three drivers in a row that hand
  over without a healthy tick between them end the chain by (3) above
  (the scheduler itself down, not a node). The successor starts where PBS
  puts it; the pipeline cannot choose its node. Not covered: a successor
  that cannot reach the scheduler at its start AND finds the lock of a
  killed predecessor -- it yields (`cannot tell: do not take it`, exit 4)
  and the chain ends, as before. (5) `pbs.status` takes its retry wait
  from `PIPELINE_QSTAT_RETRY_WAIT_S` (default 20) for the tests; the fake
  PBS's `FAKE_PBS_QSTAT_FAIL_FILE=<path>` makes every `qstat` print
  `qstat: illegally formed job identifier: <id>` and exit 1 while that
  file exists (davinci's error, verbatim). Test oracles
  (`test_smoke_block1b.py`, edge cases, with SCHEDULER_FAIL_TICKS 2 and
  the retry wait 0.1 s): (a) the scheduler lost while stage A runs: the
  first driver hands over after 2 failing ticks (its line, `unclean end 1
  in a row`), the scheduler back for the successor: `PIPELINE DONE`, the
  unclean count 0 at the end, every stage job once, no two drivers at
  once; (b) the scheduler never back: 3 drivers hand over, the fourth
  ends the chain (`the last 3 sequencer jobs ended uncleanly`), count 3,
  A's job submitted once; the scheduler back and a submission resets the
  count and finishes. Mutations: `sched_handover` (the threshold never
  reached) -> (a) FAILS; `sched_unclean` (a hand-over not noted as an
  unclean end) -> (a) FAILS at its count line and (b) at the chain's end;
  `sched_clear` (`on_tick` called on failing ticks too) -> (b) FAILS.
  Status: smoke-tested 2026-10-21 `[SANDBOX]`: `test_smoke_block1b.py` 6
  pass, 1 skip; the three mutations caught as stated, the eleven earlier
  ones unchanged; regression blocks 1, 1c, 2, 3, 7, 7b, 8, report and
  reset pass. Skipped for want of davinci: the real node's failure mode
  (`Cannot allocate memory`, then every qstat refused) -- the fake PBS
  models the refusal only -- and the 17-minute scale (the tests use 2
  ticks of about 1.6 s).
- Amended 2026-10-29 (g) (D-111, the user's "Fix this it must be bullet
  proof as the whole pipeline once started MUST BE"; the records are in
  block 7's "Environment faults and reruns"; the design Claude's,
  `[proposed]`): an environment fault never ends the chain. (1)
  `submit_driver` writes, at every submission, `<work_root>/driver/
  chain.qsub` -- the qsub argument lines of a driver job, one per line,
  its argv file `@ARGV@` and its log `@LOG@`, no dependency -- and
  `driver/revive.argv`, the plain `drive --config <cfg>` of a successor;
  the driver's job carries `ROLE=driver`. (2) A driver job that cannot
  start its command (status 96, block 7's (a); or 90, 91: its folder or
  its argv file unreachable) and sees no other job named `CHAIN_NAME`
  queued, held or running (`qselect -u <user> -N <name> -s HQRSTUW`,
  itself left out; a qselect that fails counts as none: a duplicate driver
  exits by itself, (c) above) submits a replacement from the template --
  its own argv file (it never ran, so its flags still apply) and a new log
  `driver/replace.<utc>.<pid>.log` -- with `-a <now + d>`, d = min(3600,
  300 x 2^(n-1)) s for its n-th replacement since a driver last completed
  a full tick (`driver/env_replacements`, which `on_tick` removes;
  `PIPELINE_REPLACE_BASE_S`, `PIPELINE_REPLACE_MAX_S` for the tests), qsub
  tried up to 10 times 30 s apart (`PIPELINE_CHAIN_QSUB_WAIT_S`); printed
  `RUN_ARGV CHAIN: ...` and logged in `pipeline.log`: `driver <id>: could
  not start on <host> (<why>); its replacement <id> starts in <d> s
  (replacement <n> since the last full tick; D-111)`. Another job of the
  chain queued, held or running: nothing submitted (`RUN_ARGV CHAIN: the
  sequencer's chain goes on with <ids>`). No template (code before this
  amendment): `RUN_ARGV CHAIN: ... the chain ends here`. (3) The watchdog,
  at the end of every stage job (block 7's (b)): when neither
  `<work_root>/STOP_REQUEST` nor `driver/CHAIN_END` exists, the template
  and `revive.argv` do, and qselect (as in (2); one that fails: nothing)
  lists no job named `CHAIN_NAME`, the job takes `driver/watchdog.lock`
  (`mkdir`; stale after 600 s, `PIPELINE_WATCHDOG_STALE_S`), checks the
  same again and submits a driver from the template with `revive.argv` and
  a log `driver/revive.<utc>.<pid>.log`, at once; printed `RUN_ARGV CHAIN:
  ...` and logged `watchdog: job <id> on <host>: no sequencer job was
  queued, held or running and the chain had not ended on purpose;
  submitted <id> to revive it (D-111)`. So the chain also comes back after
  a driver job PBS could not start (an `Exit_status` -2 on a bad node), or
  one that died before submitting its successor, as soon as a stage job
  ends. (4) The deliberate end, `driver/CHAIN_END` (one line: utc, the
  job, the reason), written by a driver that ends the chain on purpose --
  `done`, `stopped` on request (not by a signal), `failed`, `refused`,
  `tick limit`, the unclean limit of (c), a raise that reaches it -- and by
  a driver job's script when its command exits 2 (a refusal before its
  first tick) and none is there; not when the chain ends because no
  successor could be submitted (the scheduler: a watchdog revives it once
  qsub works again) nor when another driver holds the lock. `cmd_submit`
  removes it and `env_replacements`, logged. (5) `run`: an `EnvError` (both
  routes of block 7's (a) failed for an in-environment command) at a
  stage's start is not a refusal: it propagates, as one in a poll does, to
  `cmd_drive`'s handler (an unclean end; the successor goes on). (6)
  `cmd_submit`, on the login node, records the prefixes of `envs.extract`
  and `envs.train` (`python -c "import sys; print(sys.prefix)"` through
  `run_in_env`, whose activation writes `<work_root>/envs/<env>.prefix`)
  before the first driver, logged `environment <env>: <prefix> (recorded
  for nodes without conda, D-111)`; a failure is logged and the submission
  goes on. (7) `--status` prints `chain: ended on purpose at <utc> by
  <job>: <reason>` when `CHAIN_END` exists, and `chain: <n> replacement(s)
  after environment faults since the last full tick` when
  `env_replacements` holds n > 0. (8) `run` exports `WORK_ROOT` (the
  configuration's work root) into its own environment, so the
  in-environment commands of any sequencer record the prefix conda gives.
  Test oracles and status: block 7's entry (one suite,
  `tests/test_smoke_block7c.py`, covers both); smoke-tested 2026-10-29.
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
  [2026-10-19, amendment (d): `test_smoke_block1c.py` 7 pass, 1 skip
  (convergence); five mutations each caught (fits_always, footprint_sum,
  done_counted, never_fits, guard); the `test_cores` checks of blocks 2,
  3 and 7 pass, seven one-off mutations each caught (the bank's NCPUS from
  the environment, its CONCURRENCY unclamped, its array asking room for
  its tasks alone, S2's CONCURRENCY unclamped, S2's start check removed,
  S1's own refusal removed, the throttle without the cap); blocks 1, 1b,
  2, 3, 7, 7b, 8 pass. Skipped for want of davinci: how the cluster
  accounts the 200 cores (whether PBS enforces them, whether the user's
  jobs outside the pipeline count) and the launchers of the davinci
  checkout (step 3's `test_launchers` reads them there). The known-answer
  dry runs of `test_smoke_block1b.py` and `test_smoke_block1c.py` run on
  a copy of the Giulia file with its paths under a temporary folder, the
  reference dry runs of blocks 2, 3, 7 and 7b on a copy with its work root
  there: on davinci the committed file's work root holds the run's state,
  which a dry run reads and, after the v9 change, refuses -- rehearsed
  with such a state in place: the six checks of `e449fb5` FAIL there, the
  corrected ones pass, and step 3's three tests pass (6, 6 and 7 pass).]
  [2026-10-19, davinci (the user's paste of 14:51): the probe
  `tools/probe_compute_qsub.pbs` PASSED on dvnode002 (job 1851748, queue
  cpu, 10:52): qsub with afterany, qstat, qselect and qdel answer from a
  compute node and the child ran after its parent -- the scheduler checks
  listed above as skipped for want of davinci are now verified; the
  block-1 tests there: `test_smoke_pipeline.py` 6 pass, `test_smoke_block1c.py`
  7 pass, `test_smoke_block1b.py` 4 pass, 2 FAIL on dvlogin04 --
  `test_invariants` (iii) and `_edge_retire` (c) wrote a pid lock naming
  `dvlogin04` as "another host", which on dvlogin04 itself is judged by the
  local pid (dead: replaced at once), not by the state's heartbeat;
  reproduced in the sandbox with `socket.gethostname` patched to
  `dvlogin04` (the same 4 pass, 2 fail), corrected: the lock names
  `OTHER_HOST = gethostname() + "-elsewhere"`, which is never the running
  host; 6 pass, 1 skip under both names, the `heartbeat` and `lock_check`
  mutations still caught under both. The code was right.]
  [2026-10-19, amendment (e): `test_smoke_block1b.py` 6 pass, 1 skip with
  `_edge_accept_then_crash` in `test_edge_cases`; under the mutation
  `early_save` it FAILS with davinci's three lines of 15:31-15:32 (`the
  sequencer raised RuntimeError ... the successor ... continues`, `REFUSE:
  the configuration changed ...`, `the chain ends (refused, exit 2)`).]

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
  WALLTIME = `bank.walltime` (the array), TEST_INDEX 0 (since amendment 4:
  the plan's first task with simulations), SBI_HPC_DIR =
  `<sbi_dir>/hpc`. [contract amendment 4, 2026-11-17, Sbi-extractor's final
  check F3 (report 2026-11-17_1500 there): TEST_INDEX is the plan's first
  task that holds simulations -- the smallest index i of the plan's
  `tasks` (bank_plan.json, written by S4) whose `n_iters` is above 0
  (Sbi-extractor Block 10 counts them after dropping the partial
  topologies, so a task may hold none) -- passed to `test` and to `array`
  alike (the launcher's WALLTIME=auto reads that task's marker; its
  amendment 2 refuses a test task of 0 simulations). The test is skipped
  when `<bank_root>/tasks/task_<iiii>.json` carries this plan's sha256 and
  a `finished` time; the test log is `bank_test_<i>.log`; the record names
  `bank test task <i>`; the dry run, which runs before the plan exists,
  prints `TEST_INDEX=<the plan's first task with simulations>`. [The
  second final check's S4, accepted: `tasks` is read up to the first task
  with simulations; a malformed task after it is left to the launcher and
  the plan's own checks.] A plan
  that cannot be read, or holds no such task, stops the stage with
  `the bank plan <path> has no task with simulations` (or the read's
  problem) before any job. With every task holding simulations, i = 0 and
  every line is as before.]
- Rules: S4 runs `plan` on the login node and needs the pass line `BANK
  PLAN OK` (the tool writes the plan and its .sha256); S5 is the chain line
  BC as phases -- `test` (skipped when `<bank_root>/tasks/task_0000.json`
  -- since amendment 4 `task_<iiii>.json` of the test task --
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
  - [amendment 4] the test task: a plan whose task 0 holds no simulation
    (on the stand-in, task 0's topologies made partial with
    `drop_partial_topologies` true; where the stand-in cannot, a plan JSON
    whose tasks' `n_iters` are [0, k, ...] given to the stage with the
    launcher's calls recorded) -> `test` and `array` get TEST_INDEX 1, the
    log `bank_test_1.log`, the skip rule reads `task_0001.json`; a plan
    whose every task holds 0, or that cannot be read, stops the stage
    before any job with the line above; on the unchanged fixture every
    line and argument is as before (TEST_INDEX 0) and the end to end
    passes as above.
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
- Status: [Reopened 2026-11-17 by contract amendment 4 (the test task);
  smoke-tested again 2026-11-17 (merged 03a0085): `test_smoke_block2.py` 15
  of 16 pass, 1 skip, its new `test_test_task` (a plan [0, 5, 3]: test and
  array with TEST_INDEX 1, `bank_test_1.log`, `bank test task 1`, the skip
  rule on task_0001.json; plans [0, 0, 0], [], a non-integer n_iters, no
  `tasks`, not JSON, absent: the stage stops before any launcher call; [3,
  5, 2]: every argument as before; through the sequencer: STOP, exit 3, 0
  jobs; end to end on the stand-in with task 0's topologies partial: `TASK
  1 OK`, `WALLTIME=auto: test task 1 read 10 simulations`, VERIFIED), also
  against Sbi-extractor 5935eb0, whose launcher refuses a test task of 0
  simulations; twelve mutants caught; on the unchanged fixture the run's
  lines identical to the parent's; block1c 7/1, inputs check 8 of 9
  `[SANDBOX 2026-11-17]`. Accepted deviations: the read's problem worded
  `the bank plan <path> cannot be read: <problem>`; the stop in start, after
  D-099's check, through the stage's failure (STOP, exit 3); a state
  without `test_index` (written before) polls task 0's marker; the dry
  run's test line always prints the placeholder.] Before: smoke-tested 2026-10-16 (7 pass, 1 skip, about 30 s; the Sbi-
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

- Amended 2026-10-19 (D-099: the core cap, block 1 (d)). (a) The bank
  launcher gets `NCPUS` from the pipeline -- the launcher's own default,
  read from its `NCPUS="${NCPUS:-<n>}"` line -- never from the
  environment: since D-098 the stage runs in the sequencer's PBS job, where
  PBS sets `NCPUS` to that job's cores, and the launcher reads it (its
  tasks would ask for the sequencer's cores). (b) Each of the stage's
  jobs waits in a phase of its own until it fits (`test_waiting`,
  `array_waiting`, `gate_alone_waiting`, `verify_waiting`: logged once,
  tried again at every poll): the test task (`NCPUS`); the array when
  max(`NCPUS`, `GATE_NCPUS`) fits (the array and its gate hold the larger
  of the two), with `CONCURRENCY` = min(`bank.concurrency`, floor(free /
  `NCPUS`)); recorded with min(`CONCURRENCY`, the plan's tasks) x
  `NCPUS`, and its gate with
  `GATE_NCPUS` (the launcher's default, read likewise) `after` the array;
  the gate alone (`GATE_NCPUS`) and the verification job
  (`bank.verify.ncpus`) when they fit. A launcher whose `NCPUS` or
  `GATE_NCPUS` line cannot be read stops the stage with the reason (in a
  poll as at the start). At its start the stage asks
  `check(max(NCPUS, GATE_NCPUS))`: a cap that can never hold the array
  with its gate stops it before its test task. The dry run prints the
  `CONCURRENCY` the cap leaves. Tests: `test_smoke_block2.py` `test_cores`
  (a share of 9 with NCPUS=4 and GATE_NCPUS=16 in the sequencer's
  environment: the tasks 1 core, the gate 8, `-J 0-2%9`; in process, the
  array waits while 7 are free; a share of 7 stops the stage at its start).

- Amended 2026-10-29 (b) (D-113; block 3's amendment (a) gives the
  launcher the job protocol's variables, so the test task, the array's
  tasks and the gates start through `jobs/stage_job.sh`). A job that could
  not start its command costs no round and no try:
  - (a) the test task held off a listed host (94) or with an environment
    fault (`common.env_fault` on the job and its log `logs/bank_test_0.
    log`): submitted again (`_start_test`), the fault counted on its host
    (`_env_fault_counted`: D-102's count and D-111's guard);
  - (b) a gate (the round's or the gate alone) held off or with an
    environment fault is a gate that never ran, at once (no grace):
    `gate_not_started` keeps its job, host and reason; the fault is
    counted; a verified round then runs the gate alone, again as often as
    it cannot start (the one-try rule for a gate that never ran stays for
    the others); a dependent gate PBS ended unrun (`common.
    unrun_dependent`) never ran, nothing counted;
  - (c) the verification job (run_argv.pbs) with an environment fault:
    submitted again without counting a try;
  - (d) the rounds: a bad task of the report never started when it has no
    marker and no file on disk, and its log `logs/bank_task_<i>.log` is
    absent, older than this round's array (`array_submitted` - 60 s), lacks
    the job script's `[job] host=` line, or shows `RUN_ARGV ENV FAULT` /
    `RUN_ARGV AVOID` (its environment fault counted on the log's host).
    When every bad task never started, the round is free: nothing is
    quarantined (but a record or certificate, if any), `round` + 1,
    `free_rounds` + 1, the array again; FREE_ROUND_STALL (3) free rounds in
    a row with no fewer unstarted stop the stage. Otherwise the round
    counts against `bank.max_rerun_rounds` as before (`after <r - free>
    rerun round(s), ...`).
  - Test oracles (`tests/test_smoke_block2.py`): `test_env_faults` (the real
    launcher and job scripts, the fake conda absent on host `broken`,
    `bank.max_rerun_rounds` 0): the test task's first job and the round's
    gate on `broken`, an element PBS never starts -> the test submitted
    again, the gate taken as never run, a free round, VERIFIED, 8 jobs,
    `node_fault_hosts` {broken: 2}, the certificate's `quarantines` empty;
    `test_env_decisions` (in process): the free round with no rerun left
    (no log; a stale log; a log with `RUN_ARGV ENV FAULT`, counted on its
    host), a counted round when the log shows a start or the task has
    files, the stall STOP after 3 free rounds, a verification job exiting
    96 (no try counted, its host noted), a gate exiting 96 with its own log
    (never ran, host noted), a dependent gate ended unrun with -1 and no
    log (never ran, nothing noted). Mutations caught `[SANDBOX
    2026-10-29]`: no task ever unstarted; a verification job's fault
    counted as a try; the test task's fault a failure; a stale log read as
    this round's; no job protocol in the launcher's environment.

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

- Amended 2026-10-19 (D-099: the core cap, block 1 (d)). (a) S1's array
  has no concurrency limit (the launcher's `-J 0-<n-1>`): its cores are n x
  the element's (the `#PBS -l select` header of
  `extractor/run_extractor_array_mea.pbs`), n read from the launcher's own
  dry run (`DRYRUN=1`, its printed qsub line), once per stage start; it is
  launched when all of it and its aggregation (the larger of the two)
  fit, else `Deferred`; refused when it alone exceeds the stages' share
  (`the extraction array asks <n> x <c> = <t> cores at once and has no
  concurrency limit, more than the <b> the cap leaves ...; a CONCURRENCY in
  Sbi-extractor's launcher (a laptop change, D-062) or a larger
  resources.max_cores`); the aggregation (its header's cores) `after` the
  array. A launcher's refusal now comes from its dry run (`the extraction
  could not be planned (its dry run, exit <rc>): ...`). (b) S2: at its
  start, `check(max(sim.ncpus, the gate's cores))` (a cap that can never
  hold them stops the stage before its test task); the test task
  (`sim.ncpus`) when it fits (else `Deferred`, before the plan runs); the
  array when max(`sim.ncpus`, its gate's cores) fits, else it waits
  (`array_waiting`), with `CONCURRENCY` = min(`sim.concurrency`, floor(free
  / `sim.ncpus`)), recorded with min(`CONCURRENCY`, the plan's tasks) x
  `sim.ncpus`; its gate (the header of
  `sim_reextract/run_sim_reextract_gate.pbs`) `after` the array; a header
  that cannot be read stops the stage (`cannot read the cores of <file>`),
  in a poll as at the start. The tests' stub of `launch_stage_d.sh` honours
  `DRYRUN=1` as the real one does. Tests: `test_smoke_block3.py`
  `test_cores` (S2's CONCURRENCY 3 in a share of 10 with tasks of 3 cores;
  S1 refused in a share of 3; S1 recorded in a share of 4, then S2
  refused at its start: its gate of 8 can never fit).

- Amended 2026-10-29 (b) (D-113, the user's "This should be agnostic
  (independent) from the subproject run", 17:37; D-112, "we will run on
  the available simulations"; the design Claude's, `[proposed]`). Every
  job the pipeline submits starts through its job protocol, whatever the
  data set and whichever repository's job script it is.
  - (a) The launchers' environment (`_InputBase._env`, and block 2's
    `_BankBase._env`): the variables of `common.stage_job_protocol(cfg)`
    -- `PIPELINE_JOBS_DIR` (this checkout's `hpc/pipeline/jobs`),
    `WORK_ROOT`, `CHAIN_NAME` (`pl_<tag>`), `AVOID_FILE` (D-102's list)
    and `AVOID_HOLD_S` (`common.AVOID_HOLD_S`) -- each first removed from
    the caller's environment; a value with a comma or a blank is refused
    (`ValueError`, the stage stops). Sbi-extractor's env.sh
    (`sbix_pipeline_v`) passes them to every job its launchers submit with
    `qsub -v`, and every Sbi-extractor job script -- `extractor/
    run_extractor_array_mea.pbs`, `run_cohort_manifest.pbs`,
    `sim_reextract/run_sim_reextract_gate.pbs`, `export_windows/
    run_bank_task.pbs`, `run_bank_gate.pbs`, and S2's task wrapper
    `sim_reextract/run_mea_task.pbs` around Astro-Neuron-Network's
    `submit_mea_array.sh` (ANN unchanged) -- then starts through
    `jobs/stage_job.sh` (block 7, "Environment faults and reruns" (g)). By
    hand (no `PIPELINE_JOBS_DIR`) the scripts and the launchers are as
    before.
  - (b) S2's jobs that could not start their command -- the test task or a
    gate held off a listed host (exit 94; `common.held_off_node`) or with
    an environment fault (`common.env_fault`: exit 96, 97 / 98, a code of
    `JOB_EXEC_FAULTS`, the log's `RUN_ARGV ENV FAULT` line) -- are
    submitted again (the test task: `launch test`; a gate: `launch gate`
    with RESUME=1, the gate alone), free; an environment fault counts as a
    node fault of the job's host (the scheduler's `exec_host`, kept as the
    job's `host`) and in D-111's guard (`common.note_env_fault`; at 12 in a
    row on 3 hosts the stage stops). A gate that waited on the array
    (`after`) and that PBS ended with a code of `JOB_EXEC_FAULTS` before it
    wrote a log of its own (`common.unrun_dependent`: a one-task round is
    a plain job, whose failure ends its afterok gate unrun) is a gate that
    never ran, no node's fault: as one that finished with no record and no
    log of its own after the grace period, the gate alone, at most
    `GATE_UNRUN_MAX` (3) times in a row. Only the gate's own log counts
    (`gate_log_stat` at its submission, as block 2's).
  - (c) S2's rounds: a gate that ran and refused is read
    (`inputs.parse_gate_refusal`: the `[gate] FAIL <campaign>/<sweep>`
    blocks and their `[gate]      - ` problems). A refusal no rerun fixes
    (`[gate] REFUSED:` a precondition, `[gate] FAIL environment`, `[gate]
    FAIL no mea_env.json read`) or one without task blocks stops the stage
    as before. A task with the problem `no mea_env.json` never started (its
    job never ran `submit_mea_array.sh`); every other bad task started and
    failed. When every bad task never started, the round is free
    (`common.free_round`): not counted, but `FREE_ROUND_STALL` (3) free
    rounds in a row that leave at least as many unstarted stop the stage
    (`<n> task(s) still could not start after 3 free rounds in a row, no
    fewer each round (D-113): the environment or the cluster, not a node
    ...`). Otherwise the round counts: at most `S2_RERUN_ROUNDS` (2,
    `PIPELINE_S2_RERUN_ROUNDS`), then `after <k> rerun round(s), <n>
    task(s) still bad (job <gate>; <log>): <task>: <problems> | ...`. Each
    round is `launch array` with RESUME=1 (the plan keeps the complete
    tasks and re-plans the rest, D-112 included), recorded as
    `re-detection array (round <r>)` and `re-detection gate (round <r>)`.
  - (d) D-112 in the plan (Sbi-extractor `sim_reextract_plan.py
    --quiet-hours H`, through `sim.plan_args`): a task whose folder, JSON
    files or `topo_*/` folders changed less than H hours before the plan
    is left out, named in the plan's `excluded` (`still being written
    (D-112): changed <age> h before the plan, less than --quiet-hours H`)
    and `excluded_active`, before the replay check reads its files; every
    task so young: refused (no runnable task); such a task with detections
    in the output root: refused. S2 plans at every launch, so a task gone
    quiet joins the next round and the record names the tasks it holds.
  - Constants, not configuration keys (the digest does not change):
    `inputs.S2_RERUN_ROUNDS` 2, `inputs.GATE_UNRUN_MAX` 3,
    `common.FREE_ROUND_STALL` 3 (`PIPELINE_FREE_ROUND_STALL`).
  - S1 is not changed beyond (a): its jobs start through the protocol (the
    folder fallback included), but a well whose job could not start
    (exit 96 or 94, PBS) stops S1 at the aggregation's refusal, since the
    launcher refuses a refused root (sec. 8).
  - Test oracles (`tests/test_smoke_block3.py` `test_env_rounds`; the S2
    stub's tracking mode `STUB_TRACK=1`: tasks complete by a file, jobs
    through `jobs/stage_job.sh` when `PIPELINE_JOBS_DIR` is set, the gate's
    refusal in the real format; `tests/fake_conda` absent on host
    `broken`, fake environment folders whose python fails there): (a) the
    test task's first job and the first gate on `broken` (exit 96): both
    submitted again, `node_fault_hosts` {broken: 2}; an array element PBS
    never starts: the gate names it never started, a free round runs it;
    PIPELINE DONE, 7 jobs; the launcher's jobs print the protocol's line
    and the prefix is recorded; (b) a task failing whenever it starts:
    rounds 1 and 2, the one-task rounds' gates ended unrun (-1) taken as
    never run (the gate alone, no node fault), then the STOP line, 9 jobs;
    (c) a task whose job never starts: free rounds 1-3, then the stall
    STOP, 13 jobs. Mutations caught (`MUTATIONS` in the scratch runner,
    `[SANDBOX 2026-10-29]`): a test task that could not start taken as a
    failure; free rounds counted; the gate's environment fault taken as a
    refusal; no stall guard; no job protocol in the launcher's
    environment.

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
- Amended 2026-10-29 (D-111; block 7's "Environment faults and reruns",
  (e)): the pipeline passes `--resume` to every trial and member, so a run
  whose checkpoint is damaged must not fail for it. (1) `_save_checkpoint`
  writes the temporary file, flushes and fsyncs it, renames it and fsyncs
  the folder (best effort): a node that goes down after the rename leaves
  the old or the new checkpoint, not an empty one (the ledger's empty
  record of 2026-10-19 was a rename without an fsync). (2) `train_joint`
  with `resume` and the file present: a checkpoint `load_train_checkpoint`
  cannot load -- any exception but an OSError, which is the file system's
  (a node fault) and is raised as before -- is moved aside as
  `<name>.corrupt<k>` (the first free k; never deleted, as D-097's) and
  the run starts from epoch 0, logged `checkpoint <name> could not be
  loaded (<error>); moved aside as <name>.corrupt<k>; starting from epoch
  0`; a checkpoint of another TrainConfig still raises (it is not
  damaged). Test oracle (`smoke_test_blocks_4_6.py`, B6c): an empty, a
  truncated and a garbage checkpoint under `--resume`: each moved aside,
  the run then equal to a run without a checkpoint (history and model bit
  for bit at one thread); an intact one resumes as before. Mutation: the
  catch removed -> B6c FAILS. Status: smoke-tested 2026-10-29 `[SANDBOX]`
  (B6c pass in 38 s; the mutation made it fail on the empty checkpoint: the
  run exited 1).
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
- Amendment (i), 2026-11-14 (D-134 (2); wave W1, block 1a; the audit's
  F409, F504, S502): the best fit follows its search's finalist.
  - Modules: `hpc/pipeline/stages/best_fit.py` (the best-fit stage in its
    three uses: S8, F_q of a synthetic run view, G_q of block 18),
    `hpc/pipeline/pipeline.py` (`--retry`, sec. 2.2 rule 5),
    `hpc/pipeline/stages/synthetic.py` (`RunBestFitStage.is_done`: the
    run stages' own `is_done` came first in the lookup and hid the
    predicate below from F_q and G_q [contract amendment 1, 2026-11-14,
    D-134 `[orchestrator]`]), `hpc/pipeline/tests/test_smoke_finalist.py`,
    `hpc/pipeline/tests/test_smoke_block7.py` (`test_edge_cases`' pre-made
    best fit, below), and the pre-made artefacts and the counts that follow
    from them in `test_smoke_block2.py`, `test_smoke_block3.py` and
    `test_smoke_block7b.py` [contract amendment 2].
  - Rule, the done predicate: done(S8) and done(F_q) hold when
    `BEST_FIT_COMPLETE.json` exists, `SEARCH_COMPLETE.json` exists, and the
    trial the artefact names as its finalist (`finalist.trial_id`) is
    `SEARCH_COMPLETE.json`'s `finalist.trial_id`. done(G_q) holds when its
    artefact exists and the search's finalist it recorded (block 18's
    `search_finalist.trial_id`) is `SEARCH_COMPLETE.json`'s. The reason
    `is_done` returns names the condition that fails (the two trial ids, or
    the missing file), so `--status` shows it. done(G_q) does not rank the
    other arm's trials again (that needs the tuner, which the done predicate
    never runs): after a search that ran again and kept its finalist, a G_q
    whose trial is no longer the other arm's best stays done. Its start does
    compare (the rule below), so moving G_q's `BEST_FIT_COMPLETE.json` aside
    and `--retry best_fit_alt_<q>` refits it. Known limit [contract amendment
    2, 2026-11-14; the final check's F3]; no search of the committed
    `synth.json` v4 has a G_q.
  - Rule, the start: before the stage writes or submits anything it reads
    its current finalist (S8, F_q: `SEARCH_COMPLETE.json`'s; G_q: the other
    arm's, block 18) and examines its root (`<runs_root>/<dataset>_best`,
    `_best_alt`). The root BELONGS TO ANOTHER FINALIST when it holds a
    `finalist_config.json` or a `BEST_FIT_COMPLETE.json` that names another
    trial, or holds any member record while no readable `finalist_config.json`
    names the current finalist. Such a root is moved aside whole with
    `stages.common.set_aside(root, "set_aside")` (`<root>.set_aside<k>`;
    D-095: a move, never a deletion; an earlier one is never overwritten),
    with one log line naming the two trial ids and the new path, and the
    finalist is fitted again in a new root: M member jobs, as on a first
    run. A root of the current finalist is kept and its member records are
    reused as before. No member record of another finalist is ever reused.
    An unreadable `finalist_config.json` or `BEST_FIT_COMPLETE.json`, or one
    that names no trial, cannot be attributed and counts as another
    finalist's, with or without member records: it is moved aside, never
    overwritten [contract amendment 1, 2026-11-14, D-134 `[orchestrator]`].
    So a hand-made `BEST_FIT_COMPLETE.json` that names no finalist is no
    longer done: `test_smoke_block7.py`'s edge case that pre-made one (`{"pre_made":
    true}`) and expected `best_fit: skipped` now expects the root moved aside
    with that file unchanged, the M member jobs and `PIPELINE DONE`; the
    skipped case (an artefact naming the search's finalist) is
    `test_smoke_finalist.py`'s. [contract amendment 2, 2026-11-14, D-134
    `[orchestrator]`; the final check's F1] The other suites that pre-make
    S7's and S8's artefacts by hand follow the same rule:
    `test_smoke_block2.py` and `test_smoke_block3.py`, which pre-make both
    to end after the banks and the inputs, write a `SEARCH_COMPLETE.json`
    and a `BEST_FIT_COMPLETE.json` that name one finalist (both stages
    `skipped`, their checks and counts as before); `test_smoke_block7b.py`,
    whose searches run and whose best fit was pre-made through
    `test_smoke_block7.py`'s `pre_make_best`, expects the pre-made root
    moved aside and the M member jobs in its counts.
  - Rule, the retry: sec. 2.2 rule 5. With it, `--retry search` (the next
    step `tools/search_reset.py` prints) makes the best fit follow: when the
    search that ran again has the same finalist the best fit is `skipped`
    on its artefact, and when it has another the old root is set aside and
    the new finalist is fitted.
  - The dry run prints, for a best fit whose root belongs to another
    finalist, that the root would be moved aside (the two trial ids) and the
    finalist fitted again; it moves and writes nothing.
  - State: no field is added that an earlier state file lacks; a chain
    started at `8d5b162` resumes under this code with no change of
    behaviour unless a search runs again or `--retry` is passed.
  - Test oracles (`test_smoke_finalist.py`; the fake PBS):
    - known answer: `is_done`'s table -- no artefact; the artefact naming
      the search's finalist; naming another trial; no `SEARCH_COMPLETE.json`
      -- for S8 and for G_q; on a chain a -> b -> c with every stage done,
      `--retry a` returns b and c to `pending`, `--retry b` returns c alone,
      `--retry c` returns none;
    - reference, end to end on a reduced project: a finished run; then the
      search's finalist changed (the search reset and run again with another
      `gp_seed`, or `SEARCH_COMPLETE.json` rewritten to name the second-best
      trial of the ledger, as the audit's t2-C2 did) and `--retry search`
      (recorded kind) or `--retry search_<q>` (synthetic kind): the best fit
      returns to `pending`; the old root is found as `<root>.set_aside1`
      with every file's sha256 unchanged; M member jobs are submitted;
      `BEST_FIT_COMPLETE.json` names the search's finalist and every member
      record carries its configuration; `PIPELINE DONE`. With the finalist
      unchanged: the best fit `skipped`, no member job, the root untouched;
    - the audit's F504 case: `BEST_FIT_COMPLETE.json` removed, the member
      folders of another finalist kept, `--retry best_fit`: the root is set
      aside and M member jobs run (0 before this amendment);
    - invariants: no file is deleted by any case (every file present
      before is present after, at its path or under the set-aside root);
      `--retry <s>` with a downstream job live exits 2 and leaves the state
      file byte for byte; a retry of a stage with nothing downstream behaves
      as before;
    - edge cases: an unreadable `finalist_config.json` beside member
      records (the root is set aside); a second set-aside (`.set_aside2`);
      the dry run after a changed finalist (the line, nothing moved).
  - Confirmed: yes. The choice 2026-11-14 (D-134 (2): set aside and
    refit); the done predicate and sec. 2.2's rule 5 2026-11-16 (D-135).
  - Status: smoke-tested 2026-11-14 `[SANDBOX]` (`test_smoke_finalist.py` 6
    pass, 2 skip -- convergence, determinism; `test_smoke_block7.py` 7 of 8,
    `test_smoke_block7b.py` 6 of 7, `test_smoke_block2.py` 13 of 15 (its
    `test_env_faults` needs the Sbi-extractor checkout of D-113)
    [2026-11-16: 14 of 15 with that checkout, pushed that day (D-133;
    Sbi-extractor `4955e8d`, tree `9d8ea296`), and `test_smoke_inputs_check.py`
    7 of 8, each with its convergence skip `[SANDBOX]`. `test_env_faults`
    and `test_smoke_block7c.py`'s three checks use the tests' fake conda
    and fail when the caller exports `PIPELINE_SKIP_CONDA=1`, as the two
    checking runs of 2026-11-14 did: run plain, block 7c is 6 of 7],
    `test_smoke_block3.py` 8 of 9, each with its convergence skip, after
    contract amendments 1 and 2); checked by an independent agent on
    `e6e7af0` and `57e294d` (`specs/test_reports/2026-11-14_1558_incremental.md`,
    `2026-11-14_1659_incremental.md`). Not run on davinci.
- Amendment (j), 2026-11-14 (D-134 (1); wave W1, block 1b; the audit's
  F401, F503): the identity is re-checked whenever the configuration may
  have changed.
  - Modules: `hpc/pipeline/stages/search.py`,
    `hpc/pipeline/tests/test_smoke_identity.py`.
  - Rule: the identity's fields that come from the configuration are
    K_cfg = {`split_fracs` <- `split.fractions`, `split_seed` <-
    `search.train_seed`, `campaign` <- `search.campaign` (the run view's on
    a synthetic data set), `shape_anchors`, `space_overrides`, `flow` <-
    `search.shape_anchors`, `search.space_overrides`, `search.flow`}. At the
    stage's start, once the identity is read from the state or computed, and
    at the top of every poll, before anything is proposed or submitted in
    that call, each field of K_cfg in the state's identity is compared with
    the value the current configuration gives (numbers as floats, lists
    element by element in order, dicts key by key). On any difference the
    stage refuses with `REFUSE: the configuration no longer matches this
    search's identity (<key>: search <old>, configuration <new>; ...); a
    search is not continued under another identity -- restore the value(s),
    or reset the search first`: the stage is `failed` with that text, the
    run stops by sec. 2.2 rule 4 (the `STOP:` line, exit 3), nothing is
    proposed or submitted in that call, `pipeline_rounds.json` is not
    rewritten, and the jobs already queued or running are neither deleted
    nor forgotten. After the values are restored, `--retry search` (once
    those jobs have ended) continues the search by the rule of the retry.
    The fields that come from the bank (`split_hash`, `contract_digest`,
    the counts, `W`, `d_theta`, `latent_kind`) are not recomputed at a
    poll: the start's check of `pipeline_rounds.json` covers them.
  - A change of a key outside K_cfg accepted while the search runs (a
    walltime, `poll_seconds`, `max_concurrent_trials`, `resources.*`)
    continues the search as before.
  - The dry run prints the `REFUSE:` line for a search whose identity the
    configuration no longer matches.
  - State: no field is added; an identity written at `8d5b162` from a
    configuration compares equal to that configuration.
  - Test oracles (`test_smoke_identity.py`; the fake PBS):
    - known answer: the comparison on a table -- each field of K_cfg
      changed alone (that key named, with both values), two changed (both
      named), none changed (no refusal), `0.85` written `0.8500` (no
      refusal), the fractions in another order (refused);
    - contract: for each committed project file (`giulia.json`,
      `linddup.json`, and every run view of `synth.json`) the K_cfg part of
      the identity as `_identity` writes it from that file compares equal to
      the file -- no refusal on the running chains; [2026-11-17, the final
      check's F2: every search view of each file -- linddup.json v11's three
      run views included -- one identity per view, counted per file through
      block 29's `search_views`];
    - reference, end to end on a reduced search of at least two rounds at
      one trial at a time: stopped while round 0 runs, `split.fractions`
      changed, submitted again with `--accept-config-change`: the `STOP:`
      line carries the `REFUSE:` text naming `split_fracs`, exit 3, no trial
      job is submitted after the change (the fake PBS's count),
      `pipeline_rounds.json` byte for byte; then the value restored and
      `--retry search`: `PIPELINE DONE`, and every record of the ledger has
      the identity's `split_hash`. The same refusal for
      `search.flow.conditioner_hidden_layers`; a changed
      `search.trial.walltime` continues to `PIPELINE DONE`;
    - invariants: after every case, every ok record's `split_hash` is the
      identity's;
    - edge cases: the change accepted before the stage has started (the
      start refuses a folder of another identity as before); the dry run's
      line.
  - Confirmed: yes. The refusal 2026-11-14 (D-134 (1), the plan's 1b; it
    applies the rule of the identity above); the place of the check, the
    start and every poll, 2026-11-16 (D-135).
  - Status: smoke-tested 2026-11-14 `[SANDBOX]` (`test_smoke_identity.py` 5
    of 7, 2 skip -- convergence, determinism; the poll's and the start's
    checks each removed are caught); checked by an independent agent on
    `e6e7af0` (`specs/test_reports/2026-11-14_1558_incremental.md`). Not run
    on davinci. [2026-11-17: `test_contract` failed at a915950 against
    linddup.json v11 (the final check's F2); 5 of 7 again since 6382a02.]
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
- The core cap [amended 2026-10-19, D-099; block 1 (d)]. The search's
  throttle submits a waiting trial only while fewer than
  `max_concurrent_trials` run AND `fits(search.trial.ncpus)`: at most
  min(`max_concurrent_trials`, floor(free / ncpus)) at once (Giulia, 200 -
  4 = 196 cores: 4 trials of 48). The budget gate's probes are prepared at
  the gate's start and submitted as they fit (a probe still waiting counts
  as running). A trial whose job ended without a record is resubmitted at
  once when it fits, else it waits (`waiting`, `--resume` next) for the
  throttle, within the same `max_resubmits`. The best fit submits its
  members as they fit, at its start and at every poll, the others
  `waiting`; a member's resubmission likewise. The dry run prints
  `cores (D-099): at most <n> trial(s) at once -- max_concurrent_trials
  <m>, and the cap leaves room for <k> of <c> cores each; the gate's
  probes likewise`, and `cores (D-099): at most <n> member(s) at once --
  the cap leaves room for <k> of <c> cores each`. Trials submitted before
  the cap keep running; no new one starts until the pipeline is under it.
  Tests: `test_smoke_block7.py` `test_cores` (a cap that leaves room for
  two 1-core jobs, `max_concurrent_trials` 3: the gate's 3 probes and the
  trials never more than two at once, the members of 2 cores one at a
  time, nothing over the cap in the fake PBS's records).
- An unreadable record [amended 2026-10-19; D-096 applied to a record that
  a node fault left without its bytes: Claude's reading, `[proposed]`].
  Found on davinci 2026-10-19 (the read-only check of 15:48): the first
  driver job (1852795 on dvnode002, `--accept-config-change`) marked four
  trials ok and then raised `json.decoder.JSONDecodeError: Expecting
  value: line 1 column 1 (char 0)` in `_finish_trial_job`, reading the
  ledger record of a trial whose job had ended: the file existed and held
  no JSON (empty, or zeros) -- what a write through a temporary file and a
  rename WITHOUT an fsync leaves when the node goes down after the rename
  and before the data reach the disk `[reasoning]`. The login-node
  sequencer of `f1a2b7a`, which read the record the same way, last wrote
  the state at 05:52Z; it most likely raised on the same file
  `[reasoning; not confirmed]`.
  - `common.try_read_json(path) -> (dict, None) | (None, problem)`: never
    raises; `problem` names the error and the size, `<Exc>: <message> (<n>
    bytes[, all NUL])`; a JSON value that is not an object is a problem
    (`not a JSON object (<type>)`); a file that cannot be opened is
    `OSError: ...`.
  - `common.RECORD_GRACE_S`: `PIPELINE_RECORD_GRACE_S` from the
    environment, default 300 (as `LOG_GRACE_S`; the tests set 2).
    `common.set_aside(path, suffix) -> new path`: `<path>.<suffix><k>`, the
    first k >= 1 not taken (an earlier one is never overwritten); the
    node-fault, killed and corrupt-checkpoint moves use it too (the same
    names as before when the numbering has no gap).
  - The search: a trial whose last job ended and whose ledger record
    exists but cannot be read stays `running` (its slot held), read again
    at every poll; the first failed read logs `search: trial <id12>: its
    ledger record <path> cannot be read (<problem>); read again for up to
    <g> s`. A read that succeeds in the meantime decides, as any record.
    After RECORD_GRACE_S from the first failed read the record is moved to
    `trials/<id>.json.unreadable<k>` (k = 1, 2, ...; kept, out of the
    ledger's `*.json`; D-095), the move recorded in the trial's
    `unreadable_records` with its host (the last job log's), and logged
    `search: trial <id12>: its ledger record could not be read for <g> s
    (<problem>); set aside as <name> (last job on <host>); the trial is
    taken as a job that ended without a record` -- then the rule of such a
    job: resubmitted
    with `--resume` (at once when it fits, else `waiting`) within
    `trial.max_resubmits`, else the pipeline's `killed` record. The same
    in `_examine` (a failed trial whose record cannot be read when it is
    examined: the grace from the examination's start, then the record set
    aside and the same rule); in `_refresh_from_ledger` (a fresh state, a
    `--retry search`: no job of the stage runs, so no grace -- the record
    set aside at once, the trial left `waiting`, or recorded `killed` when
    no pending spec is left); `_revive_killed` and the killed record's
    configuration read an unreadable file as absent; the budget gate reads
    an unreadable probe file as a failed probe (`probe b<k>: <path> cannot
    be read (<problem>)`).
  - A trial that ended without a record and has no pending spec cannot be
    resubmitted (the tuner reads its configuration there): the pipeline's
    `killed` record at once, `search: trial <id12> ended without a record
    and has no pending spec to resubmit from (job <id> exit <e>); recorded
    killed` (before: resubmitted up to `max_resubmits` times, each job
    failing at its start). An ok record whose bytes were lost leaves no
    pending spec (the tuner removes it after the write), so such a trial is
    recorded killed: its configuration is nowhere else.
  - The best fit: a member's record (`*_seed<k>.json`, the runner's) that
    cannot be read keeps the member `running` for RECORD_GRACE_S, then is
    moved to `<name>.unreadable<k>` (out of the record's glob) and the
    member follows the rule of a job that ended without a record (a node
    fault in its log, a refusal, else `--resume` within
    `best_fit.max_resubmits`); at the stage's start (no member job runs)
    at once, the member then `waiting`.
  - The writers: `common.write_json` and the tuner's `write_record` flush
    and fsync the temporary file before the rename, so a record that names
    an ended job holds its bytes on the file system. The runner's own
    record is not changed (block 6; the member rule covers it). The trial
    jobs that run on davinci keep the tuner they started with.
  - `tools/search_report.py --config <file>` (read-only; stdlib; writes
    nothing): per trial the state's status, round, jobs and resubmissions,
    the last job's exit and log (host, `RUN_ARGV EXIT`, the last error
    line), the ledger record (status, nll, error) or why it cannot be
    read, the set-aside records, the pending spec, the run folder's
    checkpoint and runner record, and what the next poll does with it by
    the stage's own rules (`common.node_fault`, `try_read_json`): ok, final
    failure, a node fault healed, an unreadable record set aside and
    resubmitted or killed; with `--qstat`, one `qstat -x -f` of every
    trial's last job (its state, node, exit, walltime, memory used and
    asked); then why the failed records failed -- per cause (a heuristic
    label: a node fault by `node_fault`, SIGKILL, another signal, a
    walltime, the log's last exception) one trial's traceback tail -- and
    the knobs of the failed records against the ok ones and the open
    trials (their pending specs); the counts, the failures' errors grouped,
    and `SEARCH REPORT OK: <n> trial(s) read; nothing was written`
    [extended 2026-10-19 17:00, the user's "Why did they fail?"].
  - Test oracles: `test_smoke_block7b.py` (the fault wrapper's
    `empty_record` and `nul_record`: a zero-byte, or NUL-filled, ledger
    record written by the runner's wrapper, which then kills the tuner and
    itself with SIGKILL -- a node that went down after the rename; for a
    member, its own record): `unreadable_record` -- trial 1 `empty_record`
    then ok, trial 2 `nul_record` then ok, two trials ok, member seed 1
    `empty_record` then ok: no traceback, `PIPELINE DONE`, the records set
    aside as `.unreadable1` (0 bytes; NUL bytes), the second argv of trials
    1 and 2 with `--resume`, 4 records ok, 6 trial and 3 member
    submissions; `unreadable_units` (no scheduler) -- try_read_json's known
    answers; the grace then the move; an unreadable record and no pending
    spec: killed at once with the reason; a failed record never examined
    that cannot be read: set aside, queued again; a fresh state's refresh:
    set aside at once, no crash; write_json fsyncs; mutation `unreadable`
    (the read that raises restored) -> `unreadable_record` FAILS.
    `tools/search_reset.py --config <file> [--confirm <CODE>]` [the user's
    call of 2026-10-19 17:19, "remove everything from the round 0 of S7"]:
    a list-then-confirm tool (the project's delete rule; nothing deleted).
    The listing: the campaign's `trials`, `pending`, `runs`, `jobs`,
    `pipeline_rounds.json`, `finalists.json`, `SEARCH_COMPLETE.json` (those
    present, with their sizes) [and, amendment 5, `STOP_REQUEST.json`
    (block 30) when present; the CODE digests the items' sizes, so a
    request replaced by one of the same size between the listing and
    `--confirm` keeps the CODE -- accepted, the second final check's S3], `gate` and `bank_split.json` kept, the
    ledger's counts, the trial jobs the state records that the scheduler
    reports not finished, and `CODE <8 hex>` -- the first 8 hex digits of
    the sha256 of the listing's items, live jobs and ledger counts, so that
    a file or a job that changed makes the CODE stale. `--confirm <CODE>`:
    refused (exit 4) when a `pl_<tag>` job is queued or running or the CODE
    no longer matches; else the live trial jobs are deleted (qdel) and
    awaited up to 180 s (still live: nothing moved, exit 4), the items are
    moved to `<campaign>_set_aside_<utc>/` with a `RESET_RECORD.json` (the
    jobs deleted, the ledger's counts, the CODE, the configuration's
    digest), and the pass line `SEARCH RESET DONE: <n> job(s) deleted,
    round(s) set aside as <dir>; next: submit with --retry search
    --accept-config-change`. The state is left to that submission's
    `--retry search` (accepted once the jobs have ended). The search then
    starts again: the identity recomputed, the gate's probe files reused
    (no probe job), round 0 proposed anew -- with `search.gp_seed` unchanged
    the same configurations as before (checked by computation: `propose`
    with seed 0 twice gives the same 50 ids and configurations, seed 1 gives
    50 others, none in common `[SANDBOX 2026-10-19]`). Test oracles
    (`tests/test_smoke_reset.py`, the report test's fixture under the fake
    PBS with two running trial jobs and one finished): the listing (items,
    kept, counts, the two live jobs, the finished one not, nothing moved);
    the CODE a function of the listing (a new record or an ended job
    changes it); bytes preserved (every file's sha256 the same in the
    set-aside folder, the gate untouched); the confirm (jobs F, the folder,
    the record, the pass line, then the dry run with `--retry search
    --accept-config-change` accepted, a second listing empty); a wrong
    CODE, a sequencer job, a bad configuration refused; mutation `code`
    (the CODE constant) -> the wrong-CODE case acts: FAIL. Rehearsed on the
    stand-in (`rehearse_reset.py`, not in the repository): a search stopped
    mid round 0, reset with the CODE (3 jobs deleted, round 0 set aside, the
    gate kept), submitted again with the two flags: no probe job, the same
    round-0 ids, 5 ok records, `PIPELINE DONE` `[SANDBOX 2026-10-19]`.
    Config v10 [2026-10-19 17:44, D-101, the user: "Do it" (a new design)
    and "Increase to 400 the number of cores available"]: values only,
    `search.gp_seed` 1 (the restarted search draws 50 new configurations,
    none of the first run's) and `resources.max_cores` 400 (396 for the
    stages: 8 trials, 8 members or 8 re-detection tasks of 48 cores at
    once; `CONFIG_VERSION` 10 in `config.py`). The known answers of
    `test_smoke_block1c.py` and the `config_version` checks of blocks 1,
    1b, 2, 7, 7b follow.
    [2026-10-30, D-115, the user: "Can we please reduce the cores used to
    300?"]: Giulia's `resources.max_cores` 300 (296 for the stages: 6
    trials, 6 members or 6 re-detection tasks of 48 cores at once), a value
    under v10 -- `CONFIG_VERSION` stays 10, since no key changed and
    `linddup.json` is v10 too; the configuration's digest changes, so the
    running chain loads it by `--stop` and a submission with
    `--accept-config-change`; the jobs in flight when it loads are not
    deleted (D-099): the pipeline holds more than 300 cores until enough of
    them end, and no new job starts until it fits. The known answers of
    `test_smoke_block1c.py` (`test_known_answer`, `_edge_switch_to_cap`)
    follow.
    `tests/test_smoke_report.py` (stdlib; runs on davinci): a hand-made
    campaign folder and state (ok, a SIGBUS failure, a ValueError failure,
    an empty record with its pending spec, a NUL record without, a running
    trial): the report's rows and counts, its predictions, the pass line,
    and every file's bytes and mtime unchanged; mutation `classify` -> its
    check FAILS.
  - `tools/perrow_check.py --config <file> [--campaign-dir <dir>]...
    [--trial <id>]... [--bound <B>]` [added 2026-10-20, the user's "Write
    it" after the comments on the tail NLLs] (read-only; numpy; writes
    nothing): how much of each trial's report-split NLL is the flow's edge
    term. Every trial's flow (`joint_model.build_joint_model`: `zuko_nsf`
    behind sbi's `transform_to_unconstrained` for the unit box) models
    u = logit(theta_k) with rational-quadratic splines on [-B, B] and the
    identity outside, over a standard normal base (B = 5, zuko 1.6.0's
    default; sbi 0.27.0 passes none). A coordinate with |u_ik| > B adds to
    the row's -log q(theta_i | z_i), whatever the trial learned,
        tau(theta) = u^2/2 + ln(2 pi)/2 + ln(theta (1 - theta)),
        u = logit(theta) = ln(theta) - ln(1 - theta)
    nats (checked by computation with sbi 0.27.0 and zuko 1.6.0: four flows
    agree to 1e-6 nats at theta 1e-4 to 0.006 and 0.994 to 0.999; additive
    over the coordinates of a row; no longer true just inside |u| = 5; the
    flow's draws never fall past the bound, at most P(|z| > 5) = 5.7e-7 per
    coordinate `[SANDBOX 2026-10-20, flow_box_check.py, not in the
    repository]`). Over the n report rows i and the d coordinates k:
        T_i = sum_k tau(theta_ik) 1{|u_ik| > B},   T = (1/n) sum_i T_i,
        L = (1/n) sum_i l_i = T + R,   R = L - T,
    with l_i = -log q(theta_i | z_i) the trial's per-row NLL (nats). T is
    the same for every trial scored on the same rows -- the part of L no
    flow of the family can lower; R is the part the trial controls, and
    L_a - L_b = R_a - R_b for two trials on the same rows. tau is evaluated
    in float64 at the stored float32 theta. Inputs: every
    `<campaign>/runs/<id>/*_perrow.npz` (written by the runner: `nll` (n,)
    float64 nats, `theta` (n, d) float32 in (0, 1), `group` (n,) the donor)
    with the ledger record `<campaign>/trials/<id>.json` (`status`, `nll`);
    the campaigns are the configuration's `<tune_root>/<campaign>` and every
    `--campaign-dir` (a set-aside run, say); `--trial` keeps the ids that
    start with one of the given prefixes. Optional: the bank's sidecar
    `<banks_root>/<tag>_sim/shard_0000.json` (else the first `shard_*.json`)
    for `param_names`, `bounds_theta` and `latent_spec.coord_physical` /
    `units`, to name the coordinates and give the edge zone in physical
    units (`ln` axes: exp(lo + s (hi - lo)); `linear`: lo + s (hi - lo);
    s = sigmoid(-B) and sigmoid(B)). A row set is identified by a sha256 of
    theta as stored and of the donors; trials on the same rows share it.
    Output (stdout): the bound and the formula; per row set n, d, the
    donors, the trials on it, T, the share of rows with an edge coordinate,
    and per coordinate the share of rows with u < -B and u > B, its part of
    T, its distinct values, min, max and the physical edge zone; per trial
    (sorted by L) L, T, R, the median l_i, the share of rows with l_i < 0,
    the mean l_i over the rows with no edge coordinate, and the ledger's
    `nll` -- equal to L within 1e-6 nats (the runner writes
    `L = float(mean(nll))` of the same array) or MISMATCH; R over the
    trials (min, median, max, how many below 0); the pass line
    `PERROW CHECK OK: <k> trial(s) read, <s> skipped, <m> without a
    per-row file; <r> row set(s); T = <t> nats/row; nothing was written`
    (several row sets: `T = <t1>, <t2>, ... nats/row (one per row set)`).
    The training curves [added 2026-10-20 19:15, the user's "add the
    check": is a trial still improving when its epochs end?]: beside
    every per-row file `<stem>_perrow.npz` the runner record `<stem>.json`
    holds `history`, one entry per epoch e = 0, ..., E-1 with `val_npe`,
    the selection-split NLL in nats/row, and `p_eff_spectrum`. The best
    epoch by the runner's own rule (`joint_train.py:204, 315-316`,
    D-087): e* = -1 and best = inf to start; epoch e becomes e* when
    val_npe(e) < best - 1e-6, and best becomes val_npe(e) (epoch 0 always
    does; the runner then scores the report rows with e*'s weights,
    `joint_train.py:375-376`). Per trial: e* + 1 of E; whether
    e* = E - 1 (still improving when the epochs ran out); the gain of the
    last 5 epochs g = min_{e < E-5} val_npe(e) - min_e val_npe(e) >= 0
    (none when E <= 5); p_eff. Over the trials with a curve: how many
    have e* = E - 1, how many e* >= E - 5, the median g; the line
    `TRAINING CURVES: best epoch the last in <k> of <n> trial(s), in the
    last 5 in <m>; median gain of the last 5 epochs <g> nats/row (<j>
    without a curve)`, g printed to 4 decimals (2 significant digits where
    4 decimals would print a nonzero g as 0). A missing or unreadable
    record, or a history without a finite val_npe in every entry, is a
    trial without a curve, its reason printed.
    A per-row file that cannot be read, lacks a key, has inconsistent
    shapes, non-finite values or theta outside the open box is SKIPPED with
    its reason. Exit 0; 2 the configuration does not load; 3 no per-row
    file read.
  - Test oracles (`tests/test_smoke_perrow.py`, stdlib and numpy; runs on
    davinci): tau's known answers (theta = 0.003: 11.962413 nats; the two
    edges symmetric; 0 inside the bound) against a scalar `math`
    transcription; a hand-made campaign folder (a copy of the Giulia
    configuration under a temporary folder; 200 rows on 20 donors, d = 3,
    edge coordinates planted at 0.003, 0.999, 1e-4 and 0.006, per-row nll
    T_i + r_i with r_i known per trial): T, every trial's R and L, the edge
    shares and the table, a ledger MISMATCH, a trial on other rows (a second
    row set), a trial with no per-row file, a corrupt file SKIPPED, a
    pickled donor array read, the pass line, the physical edge zone from a
    hand-made sidecar; the training curves of planted runner records,
    written as the runner writes them (a steady fall over 20 epochs:
    e* = 19, g = 0.25; a minimum at epoch 9 then a rise: e* = 9, g = 0; a
    fall of 9e-7 after epoch 3: e* = 3 by the margin, best = the value at
    epoch 3; a creep of 9e-7 per epoch after epoch 3: e* = 19, every
    second epoch 1.8e-6 below the best so far, g = 4.5e-6; 4 epochs:
    g none; a record without `history` and a trial without a record: no
    curve, with their reasons; p_eff a number, None, a bool, NaN, absent)
    and their summary line (3 of 5 the last, 3 in the last 5, median g
    2.25e-6, 2 without); curve_summary's refusals (not a list, an epoch
    without a finite number, an int too large for a float) and
    boundaries (1 epoch; g none at E = 5, defined at E = 6; a flat
    curve: e* = 0); nothing written (every file's bytes and mtime
    unchanged); `--trial`; exit 2 and 3; with torch, sbi and zuko
    importable, the tool's T_i against the real flow's loss (both
    coordinates past the bound; skipped otherwise); mutations `sign` (the
    ln(theta (1 - theta)) term negated), `bound` (|u| >= 4), `crosscheck`
    (the ledger comparison always equal), `margin` (the best epoch
    without the 1e-6 margin) and `rule` (e* by "below every earlier
    epoch", not the best so far) must each make the known answer FAIL.
  - Status (perrow_check): smoke-tested 2026-10-20 (`test_smoke_perrow.py`
    5 pass, 1 skip (convergence, not a discretisation) with torch, sbi
    0.27.0 and zuko 1.6.0 importable; 4 pass, 2 skip without torch, the
    davinci `sbi_export` case; the mutations `sign`, `bound` and
    `crosscheck` each make `test_known_answer` FAIL). Rehearsed on a per-row
    file written by the real runner (`run_joint_arms.py`, arm A1, the
    blocks 4-6 fixture bank with edge coordinates planted, 31 report rows,
    d = 23): the file's keys and dtypes as above (`group` int64 there), T
    equal to the scalar transcription, the ledger's nll equal to L, the pass
    line `[SANDBOX 2026-10-20, rehearse_perrow.py, not in the repository]`.
    Not in the sandbox: davinci's per-row files and bank sidecar, and the
    versions of sbi and zuko in `envs.train` (the README's line P2 prints
    them; the bound sbi passes was read at 0.27.0).
    [2026-10-20 18:24, davinci `[CLUSTER, the user's pastes]`: P2 printed
    `sbi 0.27.0 zuko 1.6.0 bound 5.0`, the sandbox's versions; P1 printed
    `PERROW CHECK OK: 40 trial(s) read, 0 skipped, 35 without a per-row
    file; 1 row set(s); T = 3.9821 nats/row` -- both runs on the same
    29,639 report rows of 131 donors, 26.06% of them with an edge
    coordinate; R from -3.7927 to -0.8911, below 0 in all 40 trials.]
    [2026-10-20 19:15, the training curves: `test_smoke_perrow.py` 5 pass,
    1 skip with torch, sbi and zuko; 4 pass, 2 skip without torch; the
    mutations `margin` and `rule` each make `test_known_answer` FAIL, as
    `sign`, `bound` and `crosscheck` still do. Rehearsed on three records
    written by the real runner (arm A1, 8 epochs, seeds 0-2, stopped after
    epoch 7 with its per-epoch checkpoint kept, then `--resume`): e*
    equal to the runner's own `best_epoch` read from the checkpoint in all
    three (epochs 5, 6 and 1 of 0-7: none the last), the record's history
    equal to the checkpoint's `[SANDBOX 2026-10-20, rehearse_curves.py,
    not in the repository]`. Not yet run on davinci's records.]
    [2026-10-21 14:12, davinci `[CLUSTER, the user's paste]`, at
    `164ef07`, both runs: `TRAINING CURVES: best epoch the last in 13 of
    50 trial(s), in the last 5 in 39; median gain of the last 5 epochs
    0.0992 nats/row (0 without a curve)`; `PERROW CHECK OK: 50 trial(s)
    read, 0 skipped, 34 without a per-row file; 1 row set(s); T = 3.9821
    nats/row`. Read from the paste `[computed]`: 36 of 50 best epochs in
    18-20, 11 at 15 or earlier (2, 2 and 6 for the three worst by L);
    after the best epoch the 37 others rose by a median 0.15 nats/row;
    p_eff 2.52-3.07 in the 40 best by L. The label "plateaued or rising
    after it" also covers best epochs 18 and 19 after a descent.]
  - Status: smoke-tested 2026-10-19 (`test_smoke_block7b.py` 6 pass, 1 skip,
    about 4 minutes -- `unreadable_units` and `unreadable_record` run in
    `test_edge_cases`; the mutation `unreadable` makes `unreadable_record`
    FAIL: the sequencer raises, as on davinci; `test_smoke_report.py` 5
    pass, 1 skip, the mutation `classify` caught; the regression: blocks 1 6
    pass, 1b 6 pass, 1c 7 pass, 2 12 pass, 3 7 pass, 7 7 pass, 8 1 pass).
    Rehearsed with the code of `f1a2b7a` itself (not in the repository's
    tests): its login-node sequencer read a trial's empty ledger record,
    raised `JSONDecodeError` and left no lock -- what davinci showed at
    10:28 -- and the new chain, submitted with `--accept-config-change`,
    read the record again for the grace, set it aside, ran the trial again
    with `--resume` and finished (PIPELINE DONE, no driver raised); the
    report, run on the state the old code left, said so before the
    submission. Not in the sandbox: davinci's file system (whether its lost
    writes leave zero bytes or zeros: both are tested), the cost of an fsync
    there.
- The bad-node avoid list [amended 2026-10-21; D-102, the user's "Avoid this
  for the next timesss" and the card's "Build it as proposed"]. By 2026-10-21
  15:30 `dvnode012` had killed 13 trial attempts and 2 trials (`killed` after
  `max_resubmits`) since 2026-10-18; the node stays in service, PBS Pro gives
  a user no way to exclude it in `select`, and D-100 had left it unhandled.
  The pipeline now avoids such a node itself: a job that lands there spends
  a short hold instead of hours, and the trial loses nothing.
  - Constants (`stages/common.py`; environment overrides for the tests, no
    configuration key, so the configuration's digest does not change and a
    running chain needs no `--accept-config-change`): `AVOID_FAULTS` = 3
    (`PIPELINE_AVOID_NODE_FAULTS`), the node faults a host accumulates in a
    stage's run before it is listed; `AVOID_HOLD_S` = 600
    (`PIPELINE_AVOID_HOLD_S`), the seconds a job holds its slot on a listed
    host; `AVOID_EXIT` = 94, the exit code of such a job (reserved: the
    tuner exits 0-3, the runner 0-2, a signal 129-255); `AVOID_FILE_NAME` =
    `avoid_nodes.txt`, the list, in the configuration's `paths.work_root`
    (`common.avoid_file(cfg)`).
  - The file: plain text, one host per line, as `hostname` prints it on the
    node (the `RUN_ARGV: host <h>` line of every job log); a line that is
    blank or starts with `#` is ignored (the pipeline writes a header and,
    above each host, a comment with the date, the stage and the fault
    count). The user reads it as it is and clears it by deleting a host's
    line or the file; the pipeline appends, never rewrites or removes.
    `common.read_avoid_list(path) -> [hosts]` (absent or unreadable: `[]`);
    `common.add_to_avoid_list(path, host, note)` appends (a host already
    listed is not written twice).
  - The rule of the stages (the search's campaign and the best fit each
    keep their own counts in their state, `sub["node_fault_hosts"]`;
    `common.note_node_fault(sub, host, why, stage, log, path)` is called at
    every heal, D-096): when a host's faults since its base reach
    `AVOID_FAULTS` and it is not listed, it is appended and logged
    `<stage>: host <h> has <n> node fault(s) in this run: added to the avoid
    list <path> (D-102); a job that lands there holds its slot <s> s and is
    submitted again; delete its line from that file to lift the ban`; a host
    already listed (by the other stage, or by hand) is only noted. The host
    `?` (no `RUN_ARGV: host` line read) is never listed. At every poll
    (`common.sync_avoid_list(sub, path, log, stage)`) a host the stage listed
    that is no longer in the file was cleared by the user: logged `<stage>:
    host <h> is no longer on the avoid list <path> (cleared by hand); its
    node faults are counted again from now`, and the host's base is set to
    its count, so the next `AVOID_FAULTS` faults list it again.
  - The job (`jobs/run_argv.pbs`; variables `AVOID_FILE` and `AVOID_HOLD_S`
    through `qsub -v`, passed by `common.qsub_line(..., avoid_file=<path>)`;
    the trials and the members only -- not the gate's probes, the bank's
    verification job or the driver, whose stages do not read the exit):
    after `cd WORKDIR` and before the conda activation, when `AVOID_FILE`
    names a file and `hostname` matches a line of it exactly, the job prints
    `RUN_ARGV AVOID: host <h> is on the avoid list <path> (D-102); holding
    this slot for <s> s, then exiting 94`, sleeps `AVOID_HOLD_S` seconds (the
    scheduler does not refill the node with the next queued job at once),
    prints `RUN_ARGV EXIT: 94` and exits 94. No conda, no python, no
    command: the check is `grep -qxF` on the file.
  - The search: a trial whose last job ended with exit status 94 and no
    ledger record (the exit status from the scheduler; the log, when it has
    landed, gives the host: `common.host_of_log` reads the `RUN_ARGV AVOID:
    host <h>` line as it reads `RUN_ARGV: host <h>`) is submitted again at
    once when it fits under
    the cap, else `waiting` for the throttle -- NOT a resubmission
    (`resubmits` unchanged; `max_resubmits` untouched; the pending spec
    untouched), the next argv carrying `--resume` exactly when the held
    job's did; the hold recorded in the trial's `avoid_holds` (host, job,
    utc) and logged `search: trial <id12> landed on <host>, which is on the
    avoid list (D-102): its job <id> held the slot and exited; submitted
    again at once (not a resubmission)`. A job that exited 94 WITH a record
    is decided by the record, as any job.
  - The best fit: a member's job that ended with exit 94 and no record,
    likewise (`best_fit: member seed <k> landed on <host>, which is on the
    avoid list (D-102): ...`), before the node-fault and refusal rules;
    `resubmits` unchanged.
  - `tools/search_report.py`: a trial whose last job exited 94 without a
    record reads `held off an avoided node (exit 94): submitted again at
    once, not a resubmission`. The dry runs print `avoid list (D-102):
    <path>; a host with >= <k> node faults in this run is added; a trial
    [member] that lands there holds its slot <s> s and is submitted again at
    once (not a resubmission); delete its line to lift the ban`.
  - Not changed: the tuner and the runner; the hold is not a node fault
    (never counted); S1, S2 and the bank stages do not read the list (their
    tasks are short, and D-095 quarantines their outputs); a listed host is
    never removed by the pipeline; the fake PBS gains
    `FAKE_PBS_HOSTS=<h1>,<h2>,...`: the k-th job submitted runs with a
    `hostname` that prints h_k (the last entry for every later job), through
    a `hostname` stub first on the job's PATH.
  - Test oracles (`tests/test_smoke_block7b.py`):
    - known answer (`avoid_units`, no scheduler): `read_avoid_list` on an
      absent file, a file with a header, blanks and two hosts; `add_to_
      avoid_list` appends once (a second call does not duplicate; the first
      host's line is kept byte for byte); `note_node_fault` lists a host at
      its third fault and not before, never lists `?`, notes a host already
      listed; a fourth fault on the listed host is counted, not listed
      again; `sync_avoid_list` after the host's line is deleted by hand: the
      clear logged, the base set to 4, the fifth and sixth faults do not
      list it again, the seventh does; `held_off_node` on 94 (int or text)
      and on 0, 1, 93, 135, 271, None; the report's `next_action` on a job
      that exited 94 without a record, and on one that exited 137; the qsub
      line with and without `avoid_file`.
    - contract (`avoid_list`, end to end on the fault fixture, the throttle
      at 1, `PIPELINE_AVOID_HOLD_S` 2, `FAKE_PBS_HOSTS` nodeX for the first
      four jobs then nodeY): trial 1 SIGBUS three times on nodeX (killed
      after 2 resubmissions, the third fault lists nodeX); trial 2's first
      job lands on nodeX, holds 2 s, exits 94 and is submitted again at once
      (its state: `resubmits` 0, one `avoid_holds` entry; its second argv
      without `--resume`; its record `ok`); trial 3 ok; member seed 1's first
      job on nodeX likewise held, then ok on nodeY, `resubmits` 0; member
      seed 2 ok; the file holds `nodeX` once, with the pipeline's comment;
      the log lines above; `PIPELINE DONE`; 6 trial and 3 member
      submissions; the job log of a held job has `RUN_ARGV AVOID:` and
      `RUN_ARGV EXIT: 94` and no `RUN_ARGV: <command>` line.
    - mutations: `avoid_add` (`add_to_avoid_list` disabled): no host is ever
      listed, trial 2 is not held -> the contract check FAILS; `avoid_hold`
      (the stages take exit 94 as a job that ended without a record): the
      hold counts as a resubmission -> FAILS; `avoid_sync` (`sync_avoid_list`
      disabled, patched in the test's own process since the known answer
      runs no sequencer): the clear is never seen -> the known answer FAILS.
    - the other checks of the suite run with `FAKE_PBS_HOSTS` set to sixty
      distinct names (one per job), as a cluster spreads jobs over nodes, so
      no host of theirs reaches `AVOID_FAULTS`; `PIPELINE_AVOID_HOLD_S` 2
      throughout.
  - Status: smoke-tested 2026-10-21 (`test_smoke_block7b.py` 6 pass, 1
    skip, 462 s; `avoid_units` and `avoid_list` 64 s; the three mutations
    each make their check FAIL: `avoid_add` and `avoid_hold` -> `avoid_list`
    (nothing listed; `resubmits` 1), `avoid_sync` -> `avoid_units`; the
    regression: blocks 1 6 pass 1 skip, 1b 6 pass 1 skip, 1c 7 pass 1 skip
    (with `SBIX_CLONE`), 2 12 pass 1 skip, 3 7 pass 1 skip, 7 7 pass 1 skip,
    8 1 pass 1 skip, report 5 pass 1 skip (its mutation `classify` caught),
    reset 5 pass 1 skip, perrow 5 pass 1 skip; the job script by hand with a
    `hostname` stub: a listed host held and 94, a prefix, a suffix and an
    unlisted host run, an absent list runs; vermin 3.6, pyflakes, ASCII, LF
    `[SANDBOX 2026-10-21]`). Not in the sandbox: davinci's `hostname` on a
    compute node (the `RUN_ARGV: host` lines of its logs print the short
    name the faults are counted under, so the two agree), its PBS passing
    `-v` values with no comma (as every other variable), and whether a
    10-minute hold keeps the scheduler from refilling a node at once (the
    hold's length is the user's to change through `PIPELINE_AVOID_HOLD_S`).

- Environment faults and reruns [amended 2026-10-29; D-111, the user's
  "Fix this it must be bullet proof as the whole pipeline once started
  MUST BE"; the design Claude's, `[proposed]`]. Found on davinci 2026-10-29
  (the read-only check pasted at 14:59 [corrected 2026-11-02: not 14:52]): the sequencer `1904837` stopped ticking
  at 2026-10-26 21:43Z on its first node; PBS ran it again, with the two
  trial jobs then running (`1905463`, `1905466`, both with checkpoints),
  on 2026-10-27 at 12:36:58, on three nodes where the conda installation
  was not mounted: `conda: command not found` at `run_argv.pbs:64-65`,
  exit 98 within 1-2 s, before any command. The successor `1906402` then
  ran on one of them and exited 98 as well: no driver was left, the chain
  ended with no line in the log, and the trials' jobs ended without a
  record. Had conda been there, the trials' reruns would have started from
  epoch 0 (their argv had no `--resume`). Such faults now cost a delay,
  never the chain, a resubmission or a trial's progress.
  - (a) The activation, `jobs/activate_env.sh` (`pipeline_activate <env>`,
    0 or 96; sourced by `run_argv.pbs` and by `envs.run_in_env`). With
    `SKIP_CONDA=1` nothing (the tests, as before). First conda, as the job
    scripts always did: `set +u` across `eval "$(conda shell.bash hook)"`
    and `conda activate <env>`, the caller's `-u` restored after; a
    success records `$CONDA_PREFIX` in `<WORK_ROOT>/envs/<env>.prefix`
    (rewritten only when it changed, through a temporary name). When the
    hook or the activation fails, the environment's own folder: the first
    usable of the recorded prefix, `$HOME/.conda/envs/<env>` and
    `<d>/<env>` for each `d` of `CONDA_ENVS_PATH` -- usable when
    `<p>/conda-meta` is a directory, `<p>/bin/python` is executable and
    `<p>/bin/python -c "import sys"` exits 0 on this node within 120 s
    --, activated as conda activates one environment: `CONDA_PREFIX=<p>`,
    `CONDA_DEFAULT_ENV=<env>`, `CONDA_SHLVL=1`, `PATH=<p>/bin:$PATH`, every
    `<p>/etc/conda/activate.d/*.sh` sourced in name order under `set +u`;
    printed `RUN_ARGV ENV: host <h>: conda unavailable (<its first error
    line>); <env> activated from its folder <p>`. None usable: printed
    `RUN_ARGV ENV FAULT: host <h>: the environment <env> cannot start
    (conda: <error>; folders: <p> (<why>), ...)`, status 96.
  - (b) The job, `run_argv.pbs`, with four more variables through `qsub
    -v` (`common.qsub_line`): `JOBS_DIR` (this checkout's
    `hpc/pipeline/jobs`), `WORK_ROOT`, `CHAIN_NAME` (the sequencer's job
    name, `pl_<tag>`) and `ROLE` (`driver` for the sequencer's jobs,
    `stage` for every other). With `JOBS_DIR` the activation (a); without
    it -- a job submitted by code before this amendment -- the two lines of
    before (exit 97, 98). Status 96: a stage job holds its slot
    `AVOID_HOLD_S` s (D-102's hold: the scheduler does not refill the node
    at once), runs the watchdog of block 1 (g), prints `RUN_ARGV EXIT: 96`
    and exits 96; a driver job keeps the chain going (block 1 (g)) and
    exits 96 at once. A stage job runs the watchdog also after its command
    and after D-102's hold.
  - (c) The stages: a trial's or a member's last job that ended without a
    record (a record decides as before) is an environment fault
    (`common.env_fault(job, log_text) -> why or None`) when its exit
    status is 96, or 97 / 98 (the job script before this amendment), or
    one of `JOB_EXEC_FAULTS` = {-1, -2, -3, -4, -5, -6, -7, -10, -11, -14,
    -16, -18, -20} (PBS Pro's codes for a job its node could not start or
    keep: exec failures, MoM aborts, a bad uid, a rerun, a sister or the
    MoM lost, a hook's requeue), or when its log has a `RUN_ARGV ENV
    FAULT:` line. After D-102's hold rule and before D-096's and the
    refusal rules: the fault is counted on its host as a node fault
    (`note_node_fault`, so D-102 lists the host at its third) -- the host
    of the log's `RUN_ARGV ... host <h>` line, else the scheduler's
    `exec_host` (sec. 3.3, recorded as the job's `host` when it finishes),
    else `?` --, recorded in the trial's (member's) `env_faults` (host,
    job, why, utc), and the trial (member) is submitted again at once when
    it fits, else `waiting`: NOT a resubmission (`resubmits` unchanged).
    Logged `search: trial <id12> could not start on <host> (<why>): an
    environment fault, not the trial's (D-111); submitted again at once
    (not a resubmission); <n> node fault(s) on <host> so far`
    (`best_fit: member seed <k> ...`).
  - (d) The guard against a broken environment (the faults are free, so a
    broken environment would loop for ever): each stage counts its
    environment faults in a row (`sub["env_fault_streak"]`: the count and
    the hosts), reset by any job of the stage that ended otherwise (a
    record, a kill, a refusal; D-102's holds leave it as it is). At
    `ENV_FAULT_STREAK` (12) faults in a row on at least `ENV_FAULT_HOSTS`
    (3) hosts the stage fails, the trial (member) left `waiting`: `STOP:
    stage <s> failed: <n> job(s) in a row could not start (environment
    faults on <k> node(s): <hosts>; the last: <why>): the environment or
    the cluster, not a node -- fix it, delete from <avoid file> the hosts
    this run listed (<hosts>), then --retry <s>`.
  - (e) Reruns: every trial's and member's argv carries `--resume` (the
    runner continues from the run's checkpoint when there is one and
    starts fresh otherwise, block 6), so a job PBS runs again continues
    where its first run stopped; block 6's amendment of 2026-10-29 makes a
    checkpoint that cannot be loaded start the run fresh instead of
    failing it.
  - (f) `tools/search_report.py`: a trial whose last job is an environment
    fault reads `could not start (an environment fault: <why>): submitted
    again at once, not a resubmission (D-111)`. The dry runs print
    `environment faults (D-111): a trial [member] whose job cannot start
    its environment, or that PBS cannot start, is submitted again at once
    (not a resubmission); <n> in a row on <k> node(s) stop the stage; every
    trial [member] runs with --resume`.
  - Constants (`stages/common.py`; no configuration key, so the digest does
    not change and a running chain needs no `--accept-config-change`):
    `ENV_FAULT_EXIT` 96, `JOB_EXEC_FAULTS`, `ENV_FAULT_STREAK` 12
    (`PIPELINE_ENV_FAULT_STREAK`), `ENV_FAULT_HOSTS` 3
    (`PIPELINE_ENV_FAULT_HOSTS`).
  - Not changed: D-096's rule for a failed record; D-102's list, hold and
    counts; `max_resubmits`; Sbi-extractor's job scripts (S1, S2 and the
    bank's tasks keep their own activation: done for Giulia; a laptop
    push, D-062). [corrected 2026-10-29 (b): D-113 brings them under the
    protocol, (g)]
  - (g) [amended 2026-10-29 (b), D-113] `jobs/stage_job.sh`, the job
    protocol of the job scripts of other repositories, sourced by absolute
    path when the launcher passed `PIPELINE_JOBS_DIR`:
    `pipeline_stage_job_start <env>` prints `RUN_ARGV: host <h>, job <id>,
    env <env> (the pipeline's job protocol, D-113)` (the stages read the
    host there, `common.host_of_log`); on a host of `AVOID_FILE` (D-102)
    `RUN_ARGV AVOID: host <h> is on the avoid list <file> (D-102); holding
    this slot for <s> s, then exiting 94`, the hold, the watchdog, status
    94; else `pipeline_activate` (a); when it fails, `RUN_ARGV: holding
    this slot for <s> s (D-113), then exiting <rc>`, the hold, the
    watchdog, status 96; `pipeline_stage_job_end` (the scripts' EXIT trap)
    runs the watchdog of block 1 (g) whatever the job's exit, which it
    never changes. The stages' side: blocks 2 and 3, amended 2026-10-29
    (b). Tests: `test_smoke_block7c.py` `stage_job_units` (conda; the
    folder; nothing usable: 96 after the hold, the command not run; a
    listed host: 94, neither the activation nor the command; the command's
    own exit; SKIP_CONDA=1; the watchdog at the job's end); mutations
    `stage_avoid` and `stage_fault` each caught `[SANDBOX 2026-10-29]`.
  - Test oracles (`tests/test_smoke_block7c.py`; the jobs under the fake PBS,
    `hostname` stubs for the node names, a fake conda (`tests/fake_conda/`:
    `FAKE_CONDA_ABSENT_HOSTS` behaves as no conda installed,
    `FAKE_CONDA_ACTIVATE_FAILS`) and fake environment folders whose python
    fails on the hosts of `FAKE_ENV_BROKEN_HOSTS`; the fake PBS gains `-a`,
    `FAKE_PBS_FAIL_START=<name>:<k>` (Exit_status -2, no log),
    `FAKE_PBS_HOST_BY_NAME=<name>:<k>=<host>` and `exec_host` in qstat, and no
    longer passes a submitter's node name on to its job):
    - known answer (`activate_units`, no scheduler): `pipeline_activate`
      under `set -u` -- conda works (0; the env's python first on PATH; the
      prefix recorded; nothing printed); conda absent (the recorded prefix;
      `RUN_ARGV ENV: host noconda: conda unavailable (bash: conda: command
      not found); myenv activated from its folder <p>`; its activate.d
      script sourced; `-u` restored); conda's activation failing (the same,
      its EnvironmentNameNotFound line as the reason); only
      `$HOME/.conda/envs`; only `CONDA_ENVS_PATH` (an absent folder first);
      nothing usable (96; `RUN_ARGV ENV FAULT: ...` naming each folder and
      why: `no conda-meta`, `its python does not start on this node`);
      `SKIP_CONDA=1` (0, nothing, PATH unchanged). `run_in_env` through the
      fallback (the command's output and the `RUN_ARGV ENV:` line) and with
      nothing usable (`EnvError`, exit 96). `env_fault` on 96, 97, 98, -2,
      -11, "96" as text, the marker with no exit status; None on 0, 1, 2, 3,
      94, 135, 271, -29, None; `host_of_log` on the two new lines. The qsub
      line's variables (a stage job's, a driver's; a role `x` refused); the
      template equal to a driver's qsub line with `@LOG@` and `@ARGV@` and no
      `-W`; `revive.argv`. The guard: 11 faults on 3 hosts do not stop, the
      12th does (its text, the hosts this run listed), cleared by a job that
      ran; 12 on 2 hosts do not stop.
    - reference (`job_units`, `run_argv.pbs` and `chain.sh` by hand, the fake
      PBS for qsub and qselect): a good node (conda, the command, `RUN_ARGV
      EXIT: 0` last, the prefix recorded); no conda (the folder, the
      command); a stage job where nothing starts (`RUN_ARGV ENV FAULT:`, held
      1 s, `RUN_ARGV EXIT: 96` last, the command not run; no template:
      nothing submitted); with a template and no job of the chain, the
      watchdog revives it once (`revive.argv`, no `-a`, the line in
      `pipeline.log`) and not again while the revived job waits; nothing with
      `STOP_REQUEST`, nothing with `CHAIN_END`. A driver job where nothing
      starts: a replacement with `-a` about 1 s ahead and its own argv file
      (`env_replacements` 1, the line in `pipeline.log`); the next one 2 s;
      nothing while a held job of the chain exists; nothing with
      `STOP_REQUEST`. A driver's command that exits 2 writes `CHAIN_END`. A job
      without `JOBS_DIR` on a node without conda exits 98, as before.
    - invariants (`chain_end`, the synthetic graph, drivers as jobs): stage B
      refuses: `the chain ends (failed, exit 3)`, `CHAIN_END` written,
      `--status` prints `chain: ended on purpose at ...`; a submission with
      `--retry B` logs its removal and the pipeline finishes (`CHAIN_END`
      then `done (exit 0)`).
    - contract (`env_faults`, end to end on the blocks 4-6 fixture, the real
      tuner and runner in the jobs, `SKIP_CONDA=0`): the first trial job on
      host `noconda` runs through the folder (`RUN_ARGV ENV:` in its log; ok;
      `resubmits` 0); the second trial's first job on host `broken` exits 96
      after its hold and the trial is submitted again at once (`resubmits` 0;
      `env_faults` [(broken, its job)]; the job's `host` broken from
      `exec_host`; the log line) and finishes ok; member seed 1's first job on
      `broken` likewise (`best_fit: member seed 1 could not start on broken
      ...`); `node_fault_hosts` {broken: 1} in each stage; the streak 0 at the
      end; every trial and member argv with `--resume`; the training
      environment's prefix recorded by the sequencer's own commands;
      `PIPELINE DONE`.
    - determinism (`chain_replace`, the synthetic graph, drivers as jobs under
      the fake conda, `FAKE_PBS_HOST_BY_NAME=pl_synth:2=broken`): the
      submission records both prefixes (`environment sbi_export: ...
      (recorded for nodes without conda, D-111)`); the second driver job
      exits 96 on `broken` (`RUN_ARGV ENV FAULT:`, `RUN_ARGV CHAIN: submitted
      the replacement ...`, `RUN_ARGV EXIT: 96` last), its replacement
      carries `-a`, its own argv file and a `driver/replace.*` log, starts
      (`driver <id> on ...`) and finishes the pipeline; `env_replacements`
      gone; `CHAIN_END` `done (exit 0)`; no two drivers at once; every stage
      job once.
    - edge cases (`watchdog`, the synthetic graph with `SYNTH_RUN_ARGV=1`,
      `FAKE_PBS_FAIL_START=pl_synth:2`, stage A's job 30 s past the first
      driver's budget): the second driver ends with Exit_status -2 and no
      log while A runs; A's end revives the chain (one `watchdog: job <A>
      on ...` line, `RUN_ARGV CHAIN: no sequencer job was queued ...` in A's
      log), `PIPELINE DONE`, each stage job once. The same with `stop`
      between the two: `stop: no sequencer is running`, `CHAIN_END` stopped
      on request, A's end revives nothing (two driver jobs in all); a
      submission removes the marker and finishes.
    - mutations: `fallback` (the folder route disabled) -> `activate_units`,
      `job_units` and `env_faults` FAIL; `env_free` (an environment fault
      taken as a kill) -> `env_faults` FAILS (`resubmits` 1);
      `always_resume` (`--resume` only on a resubmission) -> `env_faults`
      FAILS; `streak` (the guard never stops) -> `activate_units` FAILS;
      `replace` (the driver's replacement disabled) -> `job_units` and
      `chain_replace` FAIL; `watchdog` (disabled) -> `job_units` and
      `watchdog` FAIL; `chain_end` (the marker ignored by the watchdog) ->
      `job_units` and the stop run of `watchdog` FAIL.
  - Status: smoke-tested 2026-10-29 `[SANDBOX]`: `test_smoke_block7c.py` 6
    pass, 1 skip (164 s), the seven mutations each make their checks FAIL as
    listed; `smoke_test_blocks_4_6.py` 10 pass, 1 skip (B6c new; its
    mutation caught); the regression: blocks 1 6 pass 1 skip, 1b 6 pass 1
    skip, 1c 7 pass 1 skip, 2 12 pass 1 skip, 3 7 pass 1 skip, 7 7 pass 1
    skip, 7b 6 pass 1 skip, 8 1 pass 1 skip, report 5 pass 1 skip, reset 5
    pass 1 skip, perrow 5 pass 1 skip; the Stage 2 suite 40/40 (1 skip), the
    Stage 3 suite 33/33, the Stage 4 suites 9/9 and 8/8 (`SKIP_CONDA=1`);
    vermin (3.5 minimum on the changed modules), pyflakes, ASCII, LF, the
    agnostic grep. Not in the sandbox: davinci's PBS (`qselect -s` with
    these letters, `-a`'s form, which `Exit_status` it reports for which
    failure, qsub and qselect from a job on a node without conda), a real
    conda installation's hook and activate.d scripts, and the 300-3600 s
    backoff and the 600 s hold at their own scale (the tests use 1-2 s).
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
- Status: smoke-tested 2026-10-16 (1 pass, 1 skip). [2026-10-19: on
  davinci, the scheduler probe passed from a compute node and the three
  block-1 tests pass (block 1's Status); the README's run sheet is the
  final one: the pull and check, the tests, the dry run, the submission,
  the status.]

Every block is written (2026-10-16); the constraints are those of secs. 2-4
above.

### Block 9: the inputs check (`tools/inputs_check.py`; added 2026-10-29)

A read-only look at a data set's inputs before its first submission, for
the LindDUP project (D-104) and any later one: whether the simulation
campaign is still being written (D-060, D-073: a campaign still written
must not be re-detected), and whether the configuration's values agree
with the data they describe -- what S2-S7 would otherwise find one
submission at a time. [proposed 2026-10-29 by Claude; Confirmed: no]

- Command: `python3 hpc/pipeline/tools/inputs_check.py --project <tag>`
  (or `--config <path>`) `[--max-files N] [--quiet-hours H] [--json OUT]
  [--no-qstat]`; run on the login node in `envs.extract` (numpy).
- Reads, never writes (but `--json`): the configuration, through
  `config.load_config` (the driver's own validation; a refused
  configuration exits 2); under `sim.sim_main`, the campaign folders that
  `sim.campaign_glob` matches and in each its sweep tasks (folders
  `sweep_<x>_task<NNNN>`, Sbi-extractor's pattern): `job_args.json`,
  `manifest.json`, the `topo_*` folders and the names of their
  `iter_*.npz` (counted, not opened), and the modification times of the
  task folder, its `topo_*` folders and its two JSON files; at most
  `max_files` (default 200) raw iteration files, the first iteration of
  each topology taken round-robin over the tasks in sorted order, each
  opened with `numpy.load(allow_pickle=False)` and only its 0-d members
  read; `<real.extract_root>/cohort_manifest.json` and its `.sha256`;
  `sim.out_root`; under `paths.sbix_dir` the S2 profile
  `sim_reextract/profiles/<real.cohort_tag>.sh` and the module
  `sbi_labels` (its `KNOWN_TOPOLOGY_AXES`, `JOB_ARGS_TOPOLOGY_BOUNDS` and
  `load_registry(sim.sim_main)`); `qstat -u $USER` when present.
- Checks. Each prints one line `OK`, `FAIL`, `WARN` or `NOTE` with its
  label; a FAIL means a stage would refuse, or run on inputs the
  configuration does not describe. With m(t) the value of a key in task t
  and Tasks the sweep tasks found:
  - `campaign`: at least one campaign and one task (FAIL otherwise); the
    counts of campaigns, tasks, topologies and iteration files (NOTE); the
    age of the last change, the maximum modification time over the read
    times (WARN when younger than `quiet_hours`, default 6 h).
  - `task files`: tasks without `job_args.json` or `manifest.json` (WARN:
    S2's plan excludes them, naming them).
  - `simtime`: m(t) of `simtime` takes one value over Tasks, equal to
    `bank.t_win_s` (FAIL).
  - `cohort manifest`: present with a matching `.sha256` (OK: S1 done);
    present without one (FAIL: S1 would run and its launcher refuses a
    root that holds a manifest); absent (NOTE: S1 runs, a first
    extraction).
  - `window`: with T = `bank.t_win_s`, Delta_t the manifest's
    `preprocessing.w_size` and W = round(T / Delta_t): |W Delta_t - T| <=
    1e-9 s and floor(T / Delta_t) = W, the bank plan's own rule (FAIL
    otherwise).
  - `n_e`: `bank.n_e` equals `preprocessing.electrodes_per_subset` (FAIL).
  - `label width`: `active_indices` takes one value over Tasks and
    |A| + |`bank.topology_axes`| = `bank.expected_p` =
    `search.shape_anchors.p` = `search.shape_anchors.d_theta` (FAIL).
  - `on axis`: `bank.on_axis` is in `param_names` and each task's row of
    `param_bounds` for it lies inside `bank.on_declared_box` (relative
    tolerance 1e-12, as the bank plan) (FAIL); a declared box wider than
    every task's box is a NOTE (the prior is then wider than the sampled
    box, D-078).
  - `conn_rule`: one value over Tasks (job_args, else manifest), equal to
    `freeze.require_conn_rule` (FAIL).
  - `iteration files`: at least one sampled, and every sampled file
    readable (FAIL).
  - `topology axis <a>`, for each a of `bank.topology_axes`: present in
    every sampled file, finite, more than one distinct value over the
    sampled files (FAIL); when `JOB_ARGS_TOPOLOGY_BOUNDS` names a, its two
    job_args keys finite with lo < hi and one pair over Tasks, and every
    sampled value inside [lo, hi] (FAIL); a not in `KNOWN_TOPOLOGY_AXES`
    (FAIL: the Sbi-extractor checkout cannot place it; pull it); sbi_labels
    not importable (FAIL).
  - `freeze exclude <a>`, for each a of `freeze.exclude`: a present in the
    sampled files (FAIL: S3 refuses an exclusion of an axis the data does
    not hold); a known topology axis present with only NaN values, not in
    `bank.topology_axes` and not excluded (WARN: the freeze counts NaN
    values as distinct, EXTRACTOR_USAGE trap 6.5).
  - `batches`: N of a `_v<N>` suffix, else 0 (Sbi-extractor's D-108 rule):
    two campaigns with one number (FAIL).
  - `S2 profile`: the profile file exists, unless S2's record does (FAIL).
  - `S2 root`: `sim.out_root` absent or empty (NOTE: a new root); with
    `REEXTRACTION_RECORD.json` and a matching `.sha256` (NOTE: S2 done);
    otherwise (WARN: detections without a record, S2 resumes).
  - `registry`: `load_registry(sim.sim_main)`'s width equals the
    manifests' `param_names` length (FAIL); its failure to import is a
    NOTE (the bank plan imports it the same way).
  - `size`: the iteration count (rows before the floor) and its raw bytes,
    count x W x 4 (NOTE).
  - `qstat`: the user's jobs (name, state), at most 40 lines (NOTE).
- Output: the lines above, then the sampled scalar members as a table
  (key, files holding it, distinct values, min, max, NaN count); last
  line `INPUTS CHECK: <f> FAIL, <w> WARN`; exit 0 without a FAIL, 1 with
  one, 2 on a refused configuration or bad arguments.
- Library calls relied on: `numpy.load`; `config.load_config`; Sbi-
  extractor's `sbi_labels`. Custom code: the scan and the comparisons.
- Test oracles (`tests/test_smoke_inputs_check.py`, a MEAMA-like tree of
  two tasks, 3 + 2 topologies of 2 iterations, a cohort manifest and a
  configuration derived from `linddup.json`, all in a temporary folder):
  the consistent tree gives 0 FAIL, the counts 1 campaign, 2 tasks, 5
  topologies, 10 iteration files, and stoa_gate_p's 5 distinct sampled
  values; each of these makes the named check FAIL and no other:
  simtime 180 in one task (`simtime`), the manifest's w_size 0.03 s,
  re-signed (`window`), n_e 1 (`n_e`), expected_p 33 (`label width`), a
  task's O_N box [0.01, 3.0]
  against the declared [0.03, 3.0] (`on axis`), conn_rule weibull in one
  task (`conn_rule`), stoa_gate_p absent from one topology's files
  (`topology axis stoa_gate_p`), stoa_gate_hi absent from one task
  (`topology axis stoa_gate_p`), an axis gate_x in topology_axes with the
  widths raised to 35 (`topology axis gate_x`), p0_conn in freeze.exclude
  (`freeze exclude p0_conn`), a
  second campaign without `_v<N>` (`batches`), the profile removed (`S2
  profile`), a manifest whose `.sha256` does not match (`cohort
  manifest`); a quiet-hours window covering the tree's age gives the
  `campaign` WARN, and 0 h none; NaN-valued p0_conn in every file gives
  the `freeze exclude` WARN; the tree's listing and modification times are
  unchanged by a run (read-only).
- Status: built 2026-10-29; smoke-tested in the sandbox; not run on
  davinci.
- Amended 2026-10-29 (b) (D-112, D-113). `sim.plan_args` is read for S2's
  `--quiet-hours H` (`plan_quiet_hours`; an unreadable value: FAIL `plan
  args`). With H: the tasks changed less than H h ago (`newest`, as the
  plan's own rule) are the ones the plan leaves out: `being written` NOTE
  naming them (OK when none; FAIL when every task is, the plan refuses);
  they are not sampled (unless every task is); the `campaign` WARN is kept
  for the tasks younger than the tool's `--quiet-hours` that the plan
  would still plan, else a NOTE; `plan option`: the Sbi-extractor
  checkout's `sim_reextract_plan.py` takes `--quiet-hours` (else FAIL).
  `pace` (`iteration_gaps`, at most 20,000 files): the gaps between two
  iteration files of one task by modification time, the 95th percentile
  and the longest; WARN when the 95th percentile is not under H (a task
  still running may look finished), else NOTE. D-113: `protocol (SBI)`:
  `<sbi_dir>/hpc/pipeline/jobs` holds stage_job.sh, activate_env.sh,
  chain.sh, and `paths.work_root` / `paths.sbi_dir` carry no comma or
  blank (else FAIL), with the environment prefixes recorded so far under
  `<work_root>/envs`; `protocol (Sbi-extractor)`: env.sh defines
  `sbix_pipeline_v` and the six job scripts call
  `pipeline_stage_job_start` (else FAIL: a checkout before D-113). The
  summary gains `plan_quiet_hours` and `being_written`. Tests:
  `test_being_written` (one task changed now: named, not sampled -- 3
  files, stoa_gate_p's 3 values --, no WARN; iteration files 10 h apart:
  the `pace` WARN, the longest 10.00 h); the consistent tree adds `being
  written` OK, `plan option` OK, `pace` NOTE, both `protocol` rows OK;
  the mutations every task changed now (`being written`), a plan without
  the option (`plan option`), plan_args `--quiet-hours soon` (`plan
  args`), stage_job.sh removed (`protocol (SBI)`), a job script without
  the protocol (`protocol (Sbi-extractor)`), each its named FAIL alone.

### Block 10: the purge tool (`tools/purge_outputs.py`; added 2026-11-02)

The deletion of a project's search, best fit and simulated bank, for a
data set whose simulated data are corrupted (the user's call of
2026-11-02, after the Giulia pipeline's halt). A list-then-delete tool
with a CODE -- the project's rule that Claude never deletes data and the
user deletes through such a tool. [the user's choice of 2026-11-02 10:22,
"Search + simulated bank"; the targets and the refusals proposed by
Claude; Confirmed: no]

- Command: `python3 hpc/pipeline/tools/purge_outputs.py --config <file>
  --what <kinds> [--confirm <CODE>]`, on a login node, stdlib only;
  `<kinds>` a comma-separated subset of `search`, `best_fit`, `sim_bank`,
  required (no default).
- Targets, from the configuration: `search` -- `<tune_root>/<search.
  campaign>` (the ledger, the pending specs, the runs and their
  checkpoints, the trial jobs' logs and argv files, the gate's probe
  files, `bank_split.json`, the rounds file, `finalists.json`,
  `SEARCH_COMPLETE.json`) and every `<tune_root>/<campaign>_set_aside_*`
  folder (rounds set aside by `tools/search_reset.py`); `best_fit` --
  `<runs_root>/<dataset>_best`; `sim_bank` -- `<banks_root>/<dataset>_sim`
  (the shards, sidecars, task markers, the record and the certificate,
  each with its `.sha256`) and `<banks_root>/<dataset>_sim_quarantine`.
  Never touched, and printed as kept: `<banks_root>/<dataset>_real` (S6),
  the bank plan's folder (S4, with the bank launcher's logs) and the work
  root (the state, the log, the verification reports, the bank profile);
  never touched either, and not listed: S1-S3's artefacts and S2's
  outputs.
- The listing (no `--confirm`; deletes nothing; exit 0): per target its
  kind, its path, and either `absent` or the files under it (lstat sizes;
  links not followed), their bytes and the newest modification time of
  the folder itself and its files (UTC); for the campaign the ledger's
  counts by status and the number of
  run folders; for the bank the record's sha256 prefix from its
  `.sha256`; the jobs check; the kept paths; the totals; and `CODE <8
  hex>` -- the first 8 hex digits of the sha256 of the canonical JSON of
  the targets (kind, path, present, files, bytes, that newest time in
  whole seconds), so that a file added, removed, resized or rewritten in
  a target makes the CODE stale. A job the refusal names once is not
  named again by the chain scan.
- Refused (exit 4, nothing deleted): the work root's lock exists (a
  sequencer holds it); a sequencer job `pl_<tag>` is queued, held or
  running; a job that any stage of the state records is not finished per
  the scheduler; a job of the user whose `qstat -f` names the chain
  (`CHAIN_NAME=pl_<tag>` followed by a comma or the end of a line, the
  wrapped lines joined) is not finished; a target is a symbolic link, is
  not a directory, or resolves outside its configured root; the scheduler
  does not answer; `--confirm` with a CODE other than the listing's (the
  listing is printed again, with its CODE). Exit 2: the configuration
  does not load, or `--what` is empty or names an unknown kind.
- `--confirm <CODE>` with the matching CODE: first the record
  `<work_root>/purge/PURGE_<utc>.json` (schema `purge_outputs/1`: the
  listing, the CODE, the configuration's sha256, the host, status
  `started`); then each present target removed by `shutil.rmtree`, in
  the listing's order, one line each, `DELETED [<kind>] <path>: <n>
  file(s), <x> GB (<s> s)`; a failure prints `FAILED [<kind>] <path>:
  <error>` and the next target is attempted; the record is rewritten with
  each target's result and status `done` or `failed`; the pass line
  `PURGE DONE: <k> folder(s) deleted, <n> file(s), <x> GB; the state file
  is not touched` (exit 0), else `PURGE INCOMPLETE: ...` (exit 5). The
  state file, the log and the chain's files are not touched: a later run
  of the stages concerned needs their `--retry`.
- Test oracles (`tests/test_smoke_purge.py`; a synthetic project under the
  fake PBS -- the report test's campaign and state, a set-aside folder, a
  simulated bank with a record and its `.sha256`, a quarantine, a real
  bank, a plan folder): the listing names every target with its counts
  and the kept paths, prints a CODE and deletes nothing (every file's
  sha256 unchanged); the CODE is the same twice and another once a file
  in any target changes; with the CODE, exactly the targets are gone,
  every kept file's sha256 is unchanged, the record says `done` with each
  target `deleted`, and the pass line is printed; a second listing finds
  every target absent; the targets come in the documented order (the
  set-aside folders by name) whatever the order they were made in;
  refused, each with nothing deleted: a wrong CODE,
  a lock, a sequencer job, a job the state records still running, a job
  naming the chain (the fake PBS prints `Variable_List`, wrapped like
  davinci's) -- a job naming another chain does not refuse --, a target
  that is a symbolic link, an empty or unknown `--what`, a configuration
  that does not load; the mutations `code` (a constant CODE) and `chain`
  (the chain scan off) each make a check FAIL.

### Block 11: the box prior for arm S (added 2026-11-03; roadmap synth, step S2)

- Modules: `hpc/joint/stage1/build_latent_bank.py` (`--prior`, the draw,
  the sidecar's latent_spec), `hpc/joint/stage1/latent_sbi_simulator.py`
  (`sample_box_prior`, `BOX_CLASS`), `hpc/joint/stage3/run_joint_arms.py`
  (`UNIT_BOX_KINDS`, `unit_box_kind`, the prior block, the dry run, the
  record's `prior`).
- Public API: `build_latent_bank.py --prior {mixture, box}` (default
  `mixture`); `sample_box_prior(spec, n, rng) -> (c, phi)`, c an (n,) int64
  array all `BOX_CLASS` = -1, phi an (n, n_latent) float64 array;
  `unit_box_kind(sidecar) -> "campaign" | "box" | None`.
- Rule (sec. 2.4, arm S; D-118, D-120): with `--prior box`, refused on arm
  R, each donor's theta_d ~ U((0, 1)^p) i.i.d., clipped to [eps, 1 - eps]
  with eps the float64 machine epsilon (as `sample_prior`), its wells
  sharing it; cls = -1 on every row; the sidecar's latent_spec is
  `LatentSBISpec.to_dict()` plus `"kind": "box"`. The runner reads kind
  `box` as block 4 reads `campaign`: prior_fn = `campaign_log_prob`, the
  floors L_0 on the report rows and on arm R's rows are 0, no
  `LatentSBISpec` is built; the record's `prior` is `box_unit_box`
  (`campaign_unit_box` and `bench_mixture` as before). The default path is
  unchanged draw for draw, and its latent_spec has exactly `to_dict()`'s
  keys, so its contract digest is unchanged.
- Test oracles (`hpc/joint/stage3/smoke_test_blocks_11_14.py`): B11 box
  bank -- a KS test against U(0, 1) on each of 6 axes over 400 donors, p >
  1e-3; cls = {-1}; kind `box`; the two wells of a donor share theta; the
  refusal on arm R; B11 mixture keys; B14 runner arms -- the record's prior
  `box_unit_box` and L0 = 0.
- Data flow: arm S's bank of a synthetic data set (the roadmap's step S4
  builds it as an array); consumed by the runner (blocks 4-7).
- Amendment 2026-11-14 (D-134 (3); wave W1, block 1c; the audit's F202
  and F206): theta strictly inside the open unit box in every bank.
  [corrected 2026-11-14: this supersedes the Rule's "clipped to [eps, 1 -
  eps] with eps the float64 machine epsilon": a draw u in (1 - 2^-25, 1)
  passed that clip and was stored as the float32 1.0.]
  - Modules: `hpc/joint/stage1/latent_sbi_simulator.py` (`BOX_DELTA`,
    `clip_open_box`, and its use in `sample_prior`,
    `sample_prior_given_classes`, `sample_box_prior`),
    `hpc/joint/stage1/build_latent_bank.py` (the stored theta's check),
    `hpc/joint/stage1/latent_bank.py` (`write_shard`'s folder),
    `hpc/pipeline/tools/synth_bank_record.py` (the theta check),
    `hpc/pipeline/tests/test_smoke_theta_box.py`.
  - Public API: `BOX_DELTA = 2.0 ** -24`, the spacing of float32 in
    [1/2, 1); `clip_open_box(phi) -> ndarray`: phi any shape, float64, each
    value in [0, 1]; returns a new float64 array of that shape,
    clip(phi, BOX_DELTA, 1 - BOX_DELTA); the input is not changed; a
    non-finite value, or a value below 0 or above 1 (no draw of the three
    samplers is: it would be a fault upstream, not something to clip away),
    raises `FloatingPointError` [the range: contract amendment 2,
    2026-11-14, D-134 `[orchestrator]`; the final check's F5].
  - Rule: the three samplers return `clip_open_box` of their draw, in place
    of the clip at the float64 epsilon. `BOX_DELTA` and `1 - BOX_DELTA` are
    float32 numbers and rounding to nearest is monotone, so for every phi a
    sampler returns float32(phi) lies in [BOX_DELTA, 1 - BOX_DELTA], inside
    (0, 1): the theta a shard stores (float32, sec. 2 Notation) is strictly
    inside the open box. A draw the new clip moves has probability
    2 BOX_DELTA = 1.2e-7 per coordinate under the uniform prior; every draw
    in [BOX_DELTA, 1 - BOX_DELTA] is unchanged bit for bit. The simulation
    is run at the float64 phi, as before.
  - The builder, after casting theta to float32 for the shard, refuses a
    stored coordinate <= 0, >= 1 or not finite (the count and the first row
    and axis; nothing written): a defence, not reachable through the
    samplers [not finite: contract amendment 2; the final check's F4].
  - The record tool (block 16) refuses, per shard, a stored theta with a
    coordinate <= 0, >= 1 or not finite: `REFUSE: shard <k>: <n> row(s) with
    a theta outside the open unit box (first: row <i>, axis <j>, value
    <v>)`, exit 4 and nothing written as its other refusals, on arm S and
    arm R alike. A shard whose x has no row is refused by the tool's row
    count as before (`REFUSE: shard <k>: 0 rows, the build's <n>`, exit 4):
    the theta check adds no problem for an empty array and never raises
    [contract amendment 2; the final check's F2] [corrected 2026-11-14: "a
    shard with no row" said more than the tool does -- its row count reads
    x; see the Status]. -0.0 is not below 0 for `clip_open_box`: it is
    clipped to BOX_DELTA.
  - `write_shard` creates the bank's folder with `exist_ok=True`: the K
    shard jobs of one bank (block 16) may reach that line together.
  - Unchanged: for the same arguments, a bank's `generator_sha256` and
    contract digest, and its `build`, are those of the code at `8d5b162`,
    so shards built before and after pool in one bank, and a calibration
    bank built after is of the generator of a simulated bank built before.
  - Test oracles (`test_smoke_theta_box.py`):
    - known answer: `clip_open_box` on 0, 1e-300, 2^-25, 2^-24, 0.5,
      1 - 2^-24, 1 - 2^-25, 1 - 1e-9, 1 - (float64 epsilon), 1: the float32
      of every result is > 0 and < 1, and the result equals the input where
      the input lies in [2^-24, 1 - 2^-24]; the old clip's case reproduced
      (float32 of clip(1 - 1e-9, eps, 1 - eps) is 1.0); -0.2, 1.5, NaN and
      inf each raise `FloatingPointError` [contract amendment 2];
    - reference: at fixed seeds the three samplers against the functions of
      `8d5b162` (`git show`), element by element: equal where the old value
      lies in [BOX_DELTA, 1 - BOX_DELTA], and the clipped old value
      elsewhere; with a generator returning chosen values (1 - 1e-9 among
      them) the box sampler, the builder's cast and the record tool leave a
      stored theta < 1;
    - invariants: a default (mixture) bank, a box bank and a cohort with
      regular centres, small, built by this code and by `8d5b162`'s
      (`git archive`): every array byte for byte, `generator_sha256`,
      contract digest and `build` equal;
    - contract: the record tool on a bank whose stored theta was set to
      1.0, to 0.0 and to NaN in one coordinate: exit 4, the `REFUSE:` line,
      nothing written; the clean bank recorded with its pass line; a shard
      with 0 rows: exit 4 with the row count's `REFUSE:` line, as at
      `8d5b162`, no traceback; the builder's cast given a NaN theta refuses
      and writes nothing [both: contract amendment 2];
    - determinism: the same argv twice gives the same bytes;
    - edge cases: two shard builds started together into one new folder
      with `os.makedirs` slowed by 0.5 s (the audit's F206): both exit 0,
      both shards present; with `8d5b162`'s `latent_bank.py` one exits 1.
  - Confirmed: yes. The clip 2026-11-14 (D-134 (3)); the builder's and
    the record tool's refusals 2026-11-16 (D-135).
  - Status: smoke-tested 2026-11-14 `[SANDBOX]` (`test_smoke_theta_box.py` 6
    pass, 1 skip -- convergence; the stage 1 suites, blocks 11-14's and the
    synthetic suite at their earlier counts); checked by an independent
    agent on `e6e7af0` and `57e294d` (the two reports of 2026-11-14). Open,
    for the hardening wave (the second report's F1, present at `8d5b162`
    too): the record tool takes a shard's rows from x alone and does not
    check that theta, cls, donor and well have as many (`latent_bank`'s
    `check_arrays` does; the builder cannot write such a shard).
- Confirmed: yes 2026-11-03 (D-118; D-120: arm S = 20,000 theta on the box,
  2 wells x 1 window).
- Status: smoke-tested 2026-11-03 (9 pass, 1 skip; the stage 1, 2 and 3
  suites and the block 4-6 suite pass, sec. 7 lists what they skip).

### Block 12: arm R's cohort and bank-wide ids (added 2026-11-03; roadmap synth, step S2)

- Modules: `hpc/joint/stage1/build_latent_bank.py` (`--class-centres`,
  `--centre-scale`, `--donors-per-class`, `class_centres_for`, `ID_STRIDE`,
  the ids written), `hpc/joint/stage1/latent_sbi_simulator.py`
  (`sample_prior_given_classes`, `regular_centres`).
- Public API: `--class-centres {simplex, regular}` (default `simplex`, the
  bench's `simplex_centres`), `--centre-scale s` (default 1.0; `regular`
  only), `--donors-per-class N` (default 0: each donor's class drawn as eq.
  (7) draws it); `regular_centres(lbg, C, L, s) -> (C, L)` float64;
  `sample_prior_given_classes(spec, c, rng) -> phi`, (n, n_latent) float64;
  `ID_STRIDE` = 1,000,000.
- Rule (sec. 2.4, arm R; D-120): the centres are mu_c = 0.5 + s (m_c - 0.5),
  m_c = `latent_burst_generator._class_center_vectors(C, L, "simplex")` (a
  private function of the DSN module, used so that the bench and the DSN
  generator share one simplex; F-ba); with N > 0 the classes are assigned
  to the donor indices as `rng.permutation(repeat(arange(C), N))` with the
  shard's generator, and phi_d given c_d is drawn by
  `sample_prior_given_classes` (TN(mu_{c,k}, tau^2; (0, 1)) on the label
  axes, U(0, 1) on the free axes); N > 0 requires --n-traces = C N
  --wells-per-donor and refuses `--prior box` and `--max-records`;
  `--provider dsn` refuses both centre options (its centres are the DSN
  generator's own). Bank-wide ids (F-at): the arrays' `donor`, `well` and
  `batch` of shard k are k * ID_STRIDE + the local id; the nuisance keys and
  the realisation seeds still use the local ids, so every shard's `x`,
  `theta`, `cls`, `nu` and `realisation_id` are unchanged byte for byte
  (shard 0: every array). The design is recorded in latent_spec
  (`class_centres_mode`, `centre_scale`, `donors_per_class`) only when it
  departs from the defaults, so a default bank's digest is unchanged.
- Test oracles: B12 cohort -- 150 donors per class exactly; the centres
  recomputed from the DSN's simplex (to 1e-12) and equidistant (to 1e-12);
  per class and label axis, the mean of theta within 5 standard errors of
  the truncated normal's mean (scipy); every batch of 8 consecutive donors
  but at most one in ten mixing the classes; the design keys; three
  refusals. B12 ids -- on two shards, ids in [k ID_STRIDE, (k + 1)
  ID_STRIDE), none shared, the local layouts equal, `concat_shards` finding
  12 donors. One-off `[SANDBOX 2026-11-03]`: a default bench shard 0 and 1
  built by the code before and after this block are equal in every array
  but shard 1's ids, which are the old ones + 10^6, and have one digest.
- Data flow: arm R's banks R0 and R3 (step S4); consumed by the runner.
- Amendment 2026-11-14 (D-134 (3)): `sample_prior_given_classes` clips with
  `clip_open_box` (block 11's amendment of this date); the cohort's theta*
  is stored strictly inside the open box.
- Confirmed: yes 2026-11-03 (D-120: the regular simplex at 0.6 of the DSN
  spacing, 9 donors per condition; D-118's Implies for F-at).
- Status: smoke-tested 2026-11-03.

### Block 13: the shifted duty clipped at 1 (added 2026-11-03; roadmap synth, step S2)

- Module: `hpc/joint/stage1/bench_burst_provider.py` (`phi_to_bench_params`).
- Rule (sec. 2.4, the gap (a); D-120; F-av): after the affine map, shifted
  or not, a_7 = min(a_7, 1). At shift delta = 0.35 pi a donor with
  theta_7 > 1 - delta gets duty 1, a smaller shift, where `BenchBurstParams`
  refused the value and the shard; at shift 0 the clip is the identity (the
  range ends at U_7 = 1). The generator hash `provider_sha256` (the source
  of the `BenchBurstProvider` class) does not see the function and is
  unchanged, as is every shard the code could build before: the clip acts
  only where the old code refused.
- Test oracles: B13 -- by hand, 0.41825 x 0.05 + 1.06825 x 0.95 = 1.0357
  clipped to 1, 0.74325 unclipped (to 1e-15), duty 1 at theta_7 = 1 and
  shift 0; a pi = 0.3 bench cohort of 3 x 9 donors x 2 wells with the three
  modes builds, 4 of its 27 donors clipped at seed 31.
- Confirmed: yes 2026-11-03 (D-120).
- Status: smoke-tested 2026-11-03.

### Block 14: arms A2 and A5 on a cohort (added 2026-11-03; roadmap synth, step S3)

- Modules: `hpc/joint/stage2/joint_batches.py` (`enumerate_donor_pairs(
  donor_ids, well_ids=None)`, `ThreeStreamBatcher(..., real_well=None)`, the
  report line), `hpc/joint/stage3/run_joint_arms.py`
  (`refuse_shared_bench_seed`, `metric_class_count`, `spread_pairs`, the
  batcher's `real_well`, the A5 refusal, the replicate diagnostic),
  `hpc/joint/stage3/smoke_test_joint_arms.py` (R5's two seeds).
- Rule (sec. 2.4, the searches):
  - F-ab: with well ids, a replicate pair is two rows of one donor in two
    different wells; the runner passes the real arm's `well` to the batcher
    (the training term of A5) and to the replicate diagnostic; without well
    ids `enumerate_donor_pairs` pairs any two rows of one donor, as before.
    The diagnostic scores at most 64 pairs taken round-robin over the
    donors (`spread_pairs`; before, the first 64 in donor order) and, when
    no donor has two wells (a real bank whose donor is its well), falls back
    to rows of one donor; its block records `pair_kind` and
    `n_donors_scored`.
  - F-aa: an arm with lambda_rep > 0 whose batcher holds no pair stops
    ("has no replicate pair").
  - F-t: the DSN loss's class count is counted on its source's labels (the
    real `cls` for A0, A2, A3; the simulated `cls` for A0s, A2s), negative
    labels excluded; the classes must be 0..C-1 with C >= 2, or the run
    stops; the count is printed (`metric classes   : C (on the <domain>
    arm)`).
  - F-aw: a bench arm S and a bench arm R whose latent_spec seeds are equal
    stop the run ("built with one --seed"); a bank whose latent_spec has no
    seed (a campaign bank, block 4) is not checked.
  - Seen on the other projects' banks (diagnostic and refusal only; the
    training of arm A1 is unchanged): on a real bank whose donor is its
    well (DUP15HD as banked) the replicate diagnostic keeps within-well
    pairs, now labelled and spread over the donors instead of the first 64
    of the first donor, and arm A5 refuses instead of training on
    within-well pairs.
- Test oracles: B14 pairs -- by hand on 8 rows (5 cross-well pairs, one
  singleton donor; 8 pairs without wells; the round-robin order); B14 class
  count; B14 runner arms -- A1, A2 and A5 to completion on a box arm S (40
  donors x 2 wells x 2 windows) and a cohort (3 x 4 donors x 2 wells x 2
  windows): prior `box_unit_box`, L0 = 0, the pseudo-real endpoint scored,
  `replicate pairs  : 48  (two wells of one donor)`, 48 pairs over 12
  donors in the replicate block, `metric classes   : 3 (on the real arm)`
  for A2 alone; B14 runner refusals -- A5 on one well per donor (and the
  diagnostic's fallback for A1 there: 12 within-well pairs over 12 donors),
  A1 on one seed, A2s on a box bank. Regression `[SANDBOX 2026-11-03]`:
  the Stage 3 suite (R5 on seeds 1 and 2), the stage 2 suites, the block
  4-6 suite (B4's 12 pairs through the fallback) pass. End to end on the
  bench generator at the decided shape (180-s windows at 100 Hz, W 18,000;
  arm S 120 theta x 2 wells x 1 window; R0 and R3 of 3 x 9 donors x 2
  wells x 6 windows, regular x 0.6, R3 at pi = 0.3 with the three modes):
  A1, A2 and A5 on R0 and A5 on R3 exit 0 in 13-17 s each at 2 epochs of
  3 steps, with the DSN backbone, 972 cross-well pairs and the diagnostic's
  64 over all 27 donors `[SANDBOX 2026-11-03]`.
- Confirmed: yes 2026-11-03 (D-118's Implies: F-ab, F-aa; D-119's: F-t;
  D-120: distinct seeds for S and R).
- Status: smoke-tested 2026-11-03.

### Block 15: the synthetic kind -- configuration, graph and run views (added 2026-11-03; roadmap synth, step S4)

- Modules: `hpc/pipeline/config.py` (`SCHEMA_SYNTHETIC`, `is_synthetic`,
  the synthetic kinds of `_check_value`, `validate`'s cross-checks,
  `load_config(path, synthetic_ok=False)`, `_strip_docs` into lists),
  `hpc/pipeline/stages/synthetic.py` (new: `RunView`, `RunSearchStage`,
  `RunBestFitStage`, `synthetic_set_graph`), `stages/__init__.py`
  (`default_graph` dispatching on the kind), `stages/common.py`
  (`shard_globs`' synthetic branch), `stages/search.py` and
  `stages/best_fit.py` (the stage's own `name` passed to the fault and
  avoid-list helpers), `pipeline.py` (`_load` with `synthetic_ok=True`).
- Public API: `config.is_synthetic(data) -> bool`; `config.load_config(
  path, synthetic_ok=False) -> Config`; `synthetic.RunView(base, run)`: a
  `Config` with `base`'s path and sha256 whose data is a copy of
  `base.data` with `paths.tune_root` and `paths.runs_root` joined with
  `run["name"]`, `search.campaign` = `run["campaign"]` and `synthetic.run`
  = run; `RunSearchStage(run)`, `RunBestFitStage(run)`;
  `synthetic_set_graph(cfg) -> [Stage]`, the stages of sec. 2.1's synthetic
  table in that order.
- Rule (secs. 2.1 and 3.1, the synthetic kind):
  - validation reports every problem at once, as for a recorded data set;
    the cross-checks run on the keys that passed their own checks;
  - S_q and F_q run the code of S7 and S8 on q's view; a view passed to
    `RunSearchStage._view` or `RunBestFitStage._view` is returned as it is
    (S7's and S8's methods call the stage's own `is_done` with the view);
    their log lines (`search: ...`, `best_fit: ...`) carry the stage's name
    (`search_<q.name>: ...`), and the fault and avoid-list helpers are given
    the stage's name, so a stop's hint says `--retry search_<q.name>`;
  - `shard_globs(view)` = (`<banks_root>/<dataset>_sim/shard_*.npz`,
    `<banks_root>/<dataset>_real_<q.real>/shard_*.npz`) for every campaign;
    a synthetic configuration that is not a view raises (a programming
    error);
  - on a recorded configuration nothing changes: `default_graph` returns
    the eight stages, `shard_globs` its pair, every log line is the same
    (the names passed to the helpers are the literals of before).
- Test oracles (`hpc/pipeline/tests/test_smoke_synthetic.py`): T15 config --
  the committed `synth.json` loads; one refusal each for a shared seed (S
  and R0; R0 and R3), an unknown real bank, a duplicated search name, the
  same (campaign, real) twice, traces that do not divide, one well per
  donor, one class, a centre scale without regular centres, pi > 0 without
  a mode, an unknown mode, a `kind` other than synthetic, a recorded key in
  a synthetic file, a missing key, a cap below the bank job; every problem
  of a file with three listed at once; `load_config` refuses the file
  without `synthetic_ok`; T15 graph -- 13 stage names and dependencies in
  the order of sec. 2.1, `default_graph` of a recorded configuration the
  eight of before; the views' folders, campaign, real glob (S-A1's
  included) and idempotence; the log prefix; T15 dry run -- the committed
  file's dry run lists the 13 stages, a qsub line per kind of job and no
  REFUSE line. [amended 2026-11-11, block 18: the committed v3 has 17
  stages; the graph check accepts each artefact absent or present (on
  davinci the run of 2026-11-07 made them), and the dry run is skipped
  where the project's work root exists, the run sheet's dry run being the
  check there.]
- Data flow: the configuration -> the graph -> the sequencer (block 1).
- Confirmed: yes 2026-11-03 (D-122: five searches; D-123: one project,
  one command, one core cap).
- Status: smoke-tested 2026-11-03 (`test_smoke_synthetic.py`: T15 config,
  graph and dry run pass; with blocks 16-17, 6 pass and 1 skip in 124 s
  `[SANDBOX]`). Regression `[SANDBOX 2026-11-03]`: the recorded kind
  unchanged -- `test_smoke_pipeline` 6 pass 1 skip, `block1b` 6 pass 1
  skip, `block1c` 6 pass 2 skip (its launcher check needs SBIX_CLONE),
  `block2` 14 of 15, `block3` 8 of 9, `block7` 7 of 8 (now with torch,
  sbi and skopt), `block7b` 6 of 7, `block7c` 6 of 7, `block8` 1 of 2,
  `inputs_check` 7 of 8, `purge` 6 pass 1 skip, `report`, `reset` and
  `perrow` pass; every skip is a convergence check or the SBIX_CLONE one.

### Block 16: the synthetic banks (added 2026-11-03; roadmap synth, step S4)

- Modules: `hpc/pipeline/stages/synthetic.py` (`SynthBankStage`, `build_of`),
  `hpc/pipeline/tools/synth_bank_record.py` (new; run in `envs.train`).
- Public API: `SynthBankStage(bank)`, bank `"sim"` or the name of an entry
  of `synthetic.real`; `build_of(cfg, bank) -> dict`; `synth_bank_record.py
  --bank-dir <dir> --expect <json> --out <dir>/BANK_RECORD.json`.
- Rule (sec. 2.4's arms S and R; D-120's shapes):
  - B_S: n_w = `n_theta` * `wells_per_donor` wells in K = n_w /
    `traces_per_shard` shards; shard k = 0, ..., K - 1 is one job of
    `hpc/joint/stage1/build_latent_bank.py --out-dir <dir> --arm S --prior
    box --shard-index k --n-traces <traces_per_shard> --n-windows
    <sim.n_windows> --seed <sim.seed>` and the builder options (`--provider`,
    `--fs`, `--T-win`, `--n-neurons`, `--wells-per-donor`,
    `--donors-per-batch`, `--n-classes`, `--tau-ov`, `--class-centres`,
    `--centre-scale`, from `synthetic.builder`): theta i.i.d. on the box per
    donor (block 11), bank-wide ids (block 12); the centres are recorded in
    its latent_spec and unused by the box's draw.
  - B_r: one shard (k = 0): `--arm R --shard-index 0 --n-traces <n_classes
    * r.donors_per_class * wells_per_donor> --donors-per-class
    <r.donors_per_class> --n-windows <r.n_windows> --pi <r.pi> --gap-modes
    <r.gap_modes joined by commas> --seed <r.seed>` and the builder options.
  - Every job: `jobs/run_argv.pbs` in `envs.train` (working directory
    `hpc/joint/stage1` for a shard, `hpc/pipeline` for the record), the
    resources of `synthetic.bank_job`, the avoid list (D-102), the job
    names `sb_<bank>_<k>` and `sr_<bank>` (bank `sim` or r.name); a job
    is submitted only when it fits under
    `resources.max_cores` (D-099); logs and argv files in
    `<work_root>/bank_jobs/<stage>/`.
  - build_of(cfg, b) = {`schema` `synth_bank_build/1`, `bank`, `arm`,
    `n_shards` K, `rows_per_shard` (per shard, its wells times its
    windows), `argv` (per shard, the builder's arguments without
    `--out-dir` and its value; floats as Python's repr)}, all strings and
    integers.
    A shard is built when its `.npz` and its `.json` exist and the
    sidecar's `provenance.argv` without `--out-dir` and its value equals the
    shard's argv of build_of.
  - start: a bank folder holding a sidecar of another argv, a shard index
    >= K, or a `BANK_RECORD.json` whose `build` differs is refused, and
    nothing in it is overwritten (move it aside or change
    `paths.banks_root`, then `--retry <stage>`); a shard already built is
    not built again; an `.npz` without its `.json` is built again, and a
    `.json` that cannot be read is set aside as `.unreadable<k>` first.
  - A shard job that ends without its shard: held off a listed node (exit
    94) or an environment fault (exit 96, PBS's exec faults; D-111) is
    submitted again, not counted (D-111's streak guard stops the stage);
    otherwise it is submitted again up to `bank_job.max_resubmits` times,
    then the stage fails with the job's id, its exit and the tail of its
    log.
  - When every shard is built: one job of `tools/synth_bank_record.py`,
    same resources, which checks the bank and writes the record, or prints
    `REFUSE: <problem>` lines, writes nothing and exits 4. The checks:
    exactly the K pairs `shard_%04d.{npz,json}`, k < K (no other index);
    every zip member's CRC (`ZipFile.testzip`); every sidecar's digest
    recomputed (`latent_bank.contract_digest`) and one digest for the bank;
    `arm`; `theta_withheld` true exactly on arm R; `W` = round(T_win f_s);
    theta's width = `search.shape_anchors.d_theta`; latent_spec `kind`
    `box` on arm S and absent on arm R; each sidecar's argv = build_of's;
    per shard, its rows = rows_per_shard and every donor with exactly
    `wells_per_donor` wells; arm S: every cls = -1; arm R: for c = 0, ...,
    n_classes - 1, exactly donors_per_class * wells_per_donor * n_windows
    rows of cls c. The record: {`schema` `synth_bank_record/1`, `bank`,
    `dir`, `n_shards`, `n_rows`, `W`, `d_theta`, `contract_digest`, `arm`,
    `latent_kind`, `theta_withheld`, `shards` [{`file`, `sha256`, `bytes`,
    `rows`, `sidecar`, `sidecar_sha256`, `contract_digest`}], `build`,
    `created_utc`, `tool`}, written atomically and fsynced, then
    `BANK_RECORD.json.sha256` the same way; the pass line `SYNTH BANK
    RECORD OK: <path> (<K> shard(s), <n> rows, digest <16 hex>)`. A record job that refused
    fails the stage with its lines (it is not resubmitted); one that ended
    otherwise without the record is resubmitted as a shard job is.
  - done(b): the record with a matching `.sha256` and its `build` =
    build_of(cfg, b).
- Test oracles: T16 banks (the fake PBS, the bench generator at f_s = 10
  Hz, T_win = 12.8 s, W = 128) -- B_S of 3 shards and B_R0, B_R3 built
  through their 8 jobs, their records' rows, kinds and builds, R3's pi 0.3
  and the cohorts' seeds in their sidecars; a re-run on a fresh state skips
  the three stages and submits nothing; with shard 2 deleted only shard 2
  and the record are submitted, shards 0 and 1 unchanged byte for byte; a
  shard the builder cannot write (`IsADirectoryError`) stops the stage
  after max_resubmits + 1 = 2 jobs with the log's tail quoted, and `--retry
  sim_bank` after the fix submits only that shard and the record; a job
  PBS cannot start (Exit_status -2) is submitted again with no resubmission
  counted; a folder holding a shard of another seed is refused ("holds a
  bank of another build", the other `--seed` quoted) and its files are
  unchanged; a corrupt shard makes the record job refuse and the stage
  stop. T16 record tool, by hand -- a shard of another digest (two digests
  in one bank), a shard index too many, bytes flipped in an archive (its
  CRC or its deflate stream), one row of a cohort moved from class 0 to
  class 1 (7 and 9 rows, 8 expected): each refused with exit 4 and nothing
  written; the clean banks recorded with their builds.
- Data flow: `synthetic.*` -> the banks -> S_q and F_q (block 15) -> the
  runner (blocks 4-7, 11-14).
- Amendment 2026-11-14 (D-134 (3)): the record tool also refuses a shard
  whose stored theta leaves the open unit box, and the shard jobs no longer
  race to create the bank's folder (block 11's amendment of this date;
  `test_smoke_theta_box.py`).
- Confirmed: yes 2026-11-03 (D-120: arm S 20,000 theta x 2 wells x 1
  window, arm R 3 x 9 donors x 2 wells x 6 windows, W 18,000; D-121: two
  independent cohorts; D-123: the bank jobs' layout is Claude's, below).
- Status: smoke-tested 2026-11-03 (T16 record 7.9 s and T16 banks 32.4 s
  pass `[SANDBOX]`; the builder at the decided shape takes about 0.1 s per
  well of one window, so a shard of 2,000 wells about 3.5 min, and 5 s
  for 6 wells of 6 windows `[SANDBOX 2026-11-03]`). Not run on davinci.
  [2026-11-04: run on davinci 2026-11-03, the three banks built and
  recorded -- 20 shards and 40,000 rows for arm S, 324 rows each for R0
  and R3, every record's sha256 matching its sidecar `[CLUSTER]`.]

### Block 17: the synthetic project (added 2026-11-03; roadmap synth, step S4)

- Modules: `hpc/pipeline/projects/synth/synth.json`,
  `hpc/pipeline/tests/test_smoke_synthetic.py`, `hpc/pipeline/README.md`
  (its run sheet "synth: the first submission").
- Values (D-118..D-123; each key's `_` note names its source): the bench
  generator at f_s = 100 Hz, T_win = 180 s, n_n = 100, 2 wells per donor,
  batches of 8 donors, 3 classes, tau_ov = 0.10, regular centres x 0.6; arm
  S: 20,000 theta, 1 window, seed 1000, 2,000 wells per shard (K = 20);
  R0: seed 2000, 9 donors per class, 6 windows, pi = 0; R3: seed 3000, the
  same at pi = 0.3; both with the three modes; the five searches S-A1 (on
  R0), S-A2-R0, S-A5-R0, S-A2-R3, S-A5-R3, in that order; 20 + 4 x 10
  trials of 8 cores, 96gb, 24 h (6 passes in 20 epochs, Giulia's)
  [amended 2026-11-04, D-125: 32gb at first; on davinci the budget gate
  measured the costliest configuration's peak at 19.0, 32.4 and 59.3 GB
  for batch 256, 512 and 1024 and its trial within 4.5 h, x 1.5 within
  6.7 h; 96gb is 1.5 x 59.3 GB rounded up to 8 GB]; the
  space: Giulia's overrides (depth_exponent [3, 5], embedding_size [16,
  64], head_pool_ops [0, 1]) on the anchors p = d_theta = 10,
  embedding_dim 64; the best fit: 5 members of 8 cores (40 passes in 40
  epochs, patience 10, Giulia's), 96gb (D-125; 32gb at first) and 120 h
  (Claude's); the loader's
  buffer the whole arm S (20 shards); the bank jobs 1 core, 8gb, 2 h
  (Claude's); the sequencer Giulia's (4 cores); `resources.max_cores` 120
  (D-120).
- Test oracles: T17 end to end (the training environment: torch, sbi,
  zuko, scikit-optimize) -- a reduced copy of the project (W 128, a
  restricted space, 3 trials per search, 2 members) with two searches,
  S-A1 and S-A2 on R0, runs to `PIPELINE DONE` under the fake PBS: both
  SEARCH_COMPLETE.json and BEST_FIT_COMPLETE.json in their run folders,
  every trial record with `L_pseudo_real`, the S-A1 trials given R0's
  windows and every S-A1 trial arm A1, at least one S-A2 trial arm A2 (the
  metric term on; gp_seed 3 of the reduced copy), both ensembles of 2
  members, nothing over the cap in the fake PBS's records.
- Confirmed: yes 2026-11-03 (D-120, D-122, D-123; the values marked
  Claude's in the file are proposals, D-123's Statement).
- Status: smoke-tested 2026-11-03 (T17 end to end in 83.5 s `[SANDBOX]`:
  S-A1's 3 trials all A1, S-A2's 3 trials with one A2, every trial record
  with L_pseudo_real, both ensembles of 2 members). Not run on davinci (the
  roadmap's step S5).
  [2026-11-04: submitted on davinci 2026-11-03 (step S5): the banks built,
  then the run stopped at search_S-A2-R0's budget gate on memory (59.3 GB
  against 32gb), the other four gates undecided `[CLUSTER]`; D-125's
  96gb; the restart (`--retry search_S-A2-R0 --accept-config-change`,
  README Z1-Z3) rehearsed on the stand-in in that situation: the probe
  files reused, no probe job submitted twice, `PIPELINE DONE` in 76 s
  `[SANDBOX]`.]

### Block 18: the other arm's best fit (added 2026-11-11; roadmap synth, step S6)

- Modules: `hpc/joint/stage4/npe_tune_joint.py` (`finalists --arm`,
  `--out`), `hpc/pipeline/config.py` (`ALT_ARMS`, the key
  `synthetic.searches[i].alt_best_fit`), `hpc/pipeline/stages/__init__.py`
  (`artefact_of("best_fit_alt", cfg)`), `hpc/pipeline/stages/best_fit.py`
  (the class attributes `root_suffix`, `job_prefix`), `hpc/pipeline/stages/
  synthetic.py` (`RunAltBestFitStage`, `synthetic_set_graph`),
  `hpc/pipeline/projects/synth/synth.json` (v3),
  `hpc/pipeline/tests/test_smoke_synthetic.py` (T18),
  `hpc/joint/stage4/smoke_test_joint_tune.py` (J37), `hpc/pipeline/README.md`
  (the run sheet "synth: the other arm's best fits").
- Public API: `npe_tune_joint.py finalists ... [--arm A1|A2|A5|shuffled]
  [--out <path>]`; `config.ALT_ARMS` = {"S-A2": ("A1", "A2"), "S-A5":
  ("A1", "A5")}; `synthetic.RunAltBestFitStage(run)`, the stage
  `best_fit_alt_<q.name>`.
- Rule (sec. 2.5, D-126):
  - the tuner's `finalists --arm a` ranks only the completed, untagged
    trials whose configuration `arm_for_config` maps to a (a configuration
    with both terms on maps to no arm and is left out), on the same `nll`
    as without the flag; with no such trial it exits non-zero, printing
    `no completed trials of arm <a> to rank`; `--out p` writes the file at
    p (default, as before, `<results_dir>/<campaign>/finalists.json`); the
    file records `"arm"` (null without `--arm`); without the two flags the
    ranking and the file are those of before plus `"arm": null`;
  - `synthetic.searches[i].alt_best_fit` is required, true or false; true
    is refused for a campaign not in `ALT_ARMS` (S-A1 has one arm; S-A25's
    other arm is not defined);
  - the graph: for each q with `alt_best_fit` true, the stage
    `best_fit_alt_<q>` follows `best_fit_<q>` in the list and depends on
    `search_<q>` alone;
  - `best_fit_alt_<q>` is S8 on q's view (block 15) with the root
    `<runs_root>/<q>/<dataset>_best_alt`, the artefact of record
    `<root>/BEST_FIT_COMPLETE.json`, the members' job names
    `ba_<dataset[:6]>_s<k>` (S8's are `bf_...`) and the log lines of S8
    under its own name; its finalist: for each arm a of
    `ALT_ARMS[campaign]`, the tuner's `finalists --top-k 1 --rank-split
    report --gate-split report --arm a --out <root>/finalists_<a>.json`
    with the search's identity (SEARCH_COMPLETE.json's); the trial of
    exactly one of the two files must be the search's finalist
    (SEARCH_COMPLETE.json's `finalist.trial_id`), else the stage refuses;
    the other file's trial is t*_{q,abar}, written to
    `<root>/finalist_config.json` in the tuner's `--config-json` shape with
    the keys `arm`, `search_finalist` ({`trial_id`, `arm`}) and
    `finalists_files`; everything after (the members, their resubmissions
    and faults, the ensemble, the artefact) is S8's, with the same
    `best_fit.*` values as `best_fit_<q>`, so E_{q,A1} and E_{q,A(q)} share
    the split seed, the member seeds and the training budget;
  - turning `alt_best_fit` on in a run is a configuration change
    (`--accept-config-change`, block 1): the stages done keep their
    artefacts and are not run again, and `best_fit_alt_<q>` enters the
    state as pending.
- Test oracles: J37 (`smoke_test_joint_tune.py`) -- on a ledger of eight
  S-A2 trials (four A1, three A2, one with both terms on), `--arm A1` and
  `--arm A2` return their arm's lowest `nll` (found in the test from the
  records), `--arm A5` exits non-zero with the message, the both-on trial is
  never returned with `--arm` and is returned without it when it is the
  lowest, `--out` writes where it says and no `finalists.json`, no flag
  gives the old file plus `"arm": null`. T18 (`test_smoke_synthetic.py`) --
  config: the committed v3 loads; `alt_best_fit` true on S-A1, true on an
  S-A25 search, a string, a missing key: each refused. Graph: the committed
  file's 17 stages in order, each `best_fit_alt_<q>` after `best_fit_<q>`
  with its dependency and artefact path, its log lines renamed. Dry run: the
  four alternative best fits' `ba_synth_s<k>` lines, no REFUSE. End to end
  (the training environment): a reduced project with the search S-A2-R0
  runs to `PIPELINE DONE` with `alt_best_fit` false; set true, the run
  REFUSES without `--accept-config-change` and with it runs
  `best_fit_alt_S-A2-R0` alone to `PIPELINE DONE`: no job of the search or
  of `best_fit_S-A2-R0` submitted again, their artefacts unchanged
  (sha256), the alternative finalist the lowest-`nll` trial of the arm the
  search's finalist is not (found in the test from the ledger), its members'
  records of that arm, an ensemble of 2 members, the member jobs `ba_*`.
  The committed file's graph checks (T15, T18) accept each artefact absent
  or present, and its dry-run checks skip where the project's work root
  exists, so that the run sheet's X2 runs on davinci after the run.
- Data flow: SEARCH_COMPLETE.json and the ledger -> the tuner's
  `finalists --arm` -> `finalist_config.json` -> the members ->
  `BEST_FIT_COMPLETE.json`, read with `best_fit_<q>`'s by block 19.
- Confirmed: yes 2026-11-11 (D-126: an ensemble of five members for each
  arm of every search with a term; the other arm's finalist picked on L as
  the search's own, D-121).
- Status: smoke-tested 2026-11-11 (`test_smoke_synthetic.py` 10 pass and 1
  skip, the convergence check, in 129 s; T18 end to end in 43 s; the
  tuner's J37 passes, its J36 needing conda's `sbi_env` fails in the
  sandbox as before `[SANDBOX]`). Seven mutations, each caught: in the
  tuner, the both-on trial counted as A2 and the arm filter dropped (J37);
  in the pipeline, the stage left out of the graph (T18 graph, dry run),
  `alt_best_fit` accepted on S-A1 and a non-bool accepted (T18 config), the
  main best fit's folder reused (T18 graph), the search finalist's own arm
  fitted (T18 end to end). The committed file's checks run on a stand-in
  `/davinci-1` tree holding the five searches' and best fits' artefacts:
  the graph checks pass, the two dry runs skip. Regression `[SANDBOX
  2026-11-11]`, the recorded kind unchanged: `test_smoke_pipeline` 6 pass 1
  skip, `block7` 7 of 8, `purge` 6 pass 1 skip, `block7b` 6 of 7,
  `block7c` 6 of 7 (every skip a convergence check). Rehearsed on a copy of
  T17's tree in the davinci situation (the run without the key done, then
  the key on S-A2-R0): the dry run without `--accept-config-change`
  refused, with it exit 0 and only `best_fit_alt_S-A2-R0` to run; the run
  to `PIPELINE DONE` in 23 s with two jobs, `ba_synth_s1` and
  `ba_synth_s2`; the search's and the best fit's artefacts byte-identical;
  the alternative finalist the ledger's lowest-`nll` A1 trial (the search's
  finalist A2), its members' records arm A1 at split seed 1. Not run on
  davinci (README X1-X5).

### Block 19: the evaluation's sampler (added 2026-11-11; confirmed 2026-11-11, D-127; roadmap synth, step S6)

- Modules: `hpc/pipeline/evaluation/__init__.py`, `ev_config.py`,
  `ev_plan.py`, `ev_models.py`, `ev_data.py`, `ev_simulate.py`,
  `ev_sample.py`, `_evalpath.py` (new; the `ev_` prefix because the DSN
  tree, which puts itself on `sys.path` at run time, has its own
  `metrics.py` and `evaluate.py`; they run in `envs.train` and never in the
  sequencer, which imports none of them; `ev_config` and `ev_plan` import
  the standard library and the pipeline's own modules only, so that block
  22's launcher runs on the login node's python),
  `hpc/pipeline/projects/synth/synth_eval.json` (new: the evaluation's
  values; a file of its own, so that the running chain's `synth.json` is
  never touched), `hpc/pipeline/tests/test_smoke_evaluation.py` (new;
  checks E19*).
- Public API: `ev_config.load(path) -> EvalConfig`; `ev_plan.ensembles(cfg)
  -> [Ensemble]` (name `<q>:<arm>`, stem `<q>_<arm>`, search q, arm, folder,
  member records, cohorts), `ev_plan.find(cfg, name)`,
  `ev_plan.calbank_build(cfg, ecfg)` (the calibration bank's builder
  arguments per shard), `ev_plan.sim_bank_glob(cfg)`;
  `ev_models.rebuild(record_path, index) -> (JointDSNNPE in eval mode,
  info)`; `ev_data.sim_index(cfg)`,
  `ev_data.report_rows(index, fracs, seed, split_hash)`,
  `ev_data.calibration_rows(donor)`, `ev_data.cohort(cfg, name)` ([2026-11-12,
  block 23] also `window_idx` and `contaminated`, additive),
  `ev_data.pseudo_true(theta, gap_spec) -> (theta_dag, in_box, delta)`;
  `ev_simulate.fixed_stats(x) -> (n, 8)`, `ev_simulate.windows_at(theta,
  latent_spec, seed, ncpus) -> (n, W)`;
  `python3 hpc/pipeline/evaluation/ev_sample.py --config <tag>.json
  --eval-config <tag>_eval.json --ensemble <q>:<arm>` (exit 0 with the pass
  line `EVAL SAMPLE OK: <stem>`, 2 on its arguments, 3 on a refusal).
- Rule (sec. 2.5; D-126, D-127):
  - the ensembles: for each search q, `<runs_root>/<q>/<tag>_best` and,
    once it has its `BEST_FIT_COMPLETE.json`, `<tag>_best_alt` (block 18);
    an ensemble's arm is its members' records' `arm`, which must agree; its
    cohorts are every entry of `synthetic.real` for campaign S-A1 (D-122)
    and q.real otherwise;
  - a member is rebuilt from its record's `config` and `<stem>_ckpt.pt`:
    the backbone by the runner's `make_backbone` (A_ref's
    `FixedStatsSummary`), the flow by `build_joint_model` with the prior
    BoxUniform on (0, 1)^d and, for sbi's shapes, the first 64 training rows
    of the record's split (its seed and fractions; the split's hash must
    equal the record's), then `load_checkpoint` with its digest check;
  - the oracle: each member's -log q at theta^ref on D_sim equals its
    `_perrow.npz` `nll`, and on the search's own cohort its
    `nll_pseudo_real`, within `oracle_tol` nats row for row, or the job
    refuses naming the member, the largest and the mean difference and the
    tolerance (the rebuild reproduced a member exactly, 0.0,
    `[SANDBOX 2026-11-11]`). [amended 2026-11-12: the tolerance was the
    constant 1e-4 at the build; on davinci four of the five finalists'
    samplings refused at 1.2e-4 to 5.6e-4 nats, S-A1's passed at 9.5e-6
    `[CLUSTER 2026-11-12, the user's paste of 15:38]` -- float32 roundoff of
    the full-size networks scored in batches of another size and with
    another thread count than the runner's (256 rows); in the sandbox a
    wrong member differs by 0.30 to 2.4 nats and rows misaligned by one by
    9 to 11 nats, a correct rebuild by 0 to 9.5e-6 whatever the batch size
    (64, 256, 7) `[SANDBOX 2026-11-12]`. So the tolerance is a value of the
    configuration, `oracle_tol`, 2e-3 nats in synth: about 4 times the
    largest roundoff seen, 150 times under the smallest wrong rebuild
    measured. It is not one of SAMPLE_PARAMS: it changes no output, so a
    sample made under another tolerance stays current.];
  - the data sets: D_sim the report rows of the members' split, in global
    row order; its calibration rows the first row of each donor (Q-E2); its
    PPC rows the first `ppc_rows` calibration rows (Q-E4); D_<r> every window
    of cohort r in bank order, theta^ref the bank's theta; D_<r>+ (files
    `<stem>_<r>dag.npz`) for a cohort
    whose sidecar `gap_spec` has the range shift at pi > 0: theta^dag with
    delta = `latent_gap.free_axis_range_shift(GapSpec(**gap_spec))`, theta*_k
    + delta on `BENCH_FREE_IDX`, axis 7 capped at (1 - L_7) / (U_7 - L_7)
    (= 1) as `phi_to_bench_params` clips the duty, `in_box` = all(0 <
    theta^dag < 1), the draws and the windows D_<r>'s;
  - for each data set and member m: log q_m(theta^ref | x) where theta^ref is
    in the open box and -inf elsewhere -- the flow is never read there, as
    it returns finite values outside the box (-114.8 nats with theta_8 =
    1.05, `[SANDBOX 2026-11-11]`: its logit transform clamps); n_post / M
    draws from each member, in member order (`torch.manual_seed` per data
    set and member); the mixture's log density (E1) at every draw, every
    member evaluated at every draw, and the log density of each draw under
    its own member; each member's embedding; the eight fixed statistics of
    every window (`FixedStatsSummary`); [2026-11-11, at the build] the
    densities at the draws embed each window once and evaluate the flow's
    distribution, conditioned on the embeddings, at the draws by
    broadcasting (sbi 0.27's `ZukoFlow.log_prob` expands the condition
    before its embedding net and would embed each window n_post times;
    the numbers are the same, E19c);
  - the seeds are common to the ensembles (common random numbers, so that
    two arms' paired differences carry no independent Monte Carlo noise of
    the draws): member k's draws on data set D from `seed_of(seed, D) + k`,
    D's posterior-predictive windows from `seed_of(seed, D, "ppc")`, the
    calibration draws from `seed_of(seed, "cal")` [2026-11-11, at the
    build];
  - the posterior-predictive check (Q-E4): for each PPC row (the cohort's
    windows; D_sim's PPC rows), the draws s = j n_post / ppc_draws, j = 0,
    ..., ppc_draws - 1 (spread over the members), each simulated as one
    window by the bench provider with the bank's latent spec (T_win, fs,
    n_neurons), the default `NuisanceSpec` at ids of its own, no gap; the
    eight statistics of each;
  - L-C2ST's calibration (Q-E5): the calibration bank, `cal_n` windows of
    arm S on the box prior, one well per donor, at its own seed (block
    22's job, the arguments of `ev_plan.calbank_build`), one mixture draw
    per window (a member uniformly at random, then its draw), and the
    windows' statistics; the bank is checked before any sampling (the box
    prior, a builder seed no bank of the project has, `cal_n` windows of as
    many donors), so a refusal writes nothing;
  - outputs, under `<out_root>/samples/`: `<stem>_<D>.npz` per data set,
    `<stem>_cal.npz`, `<stem>_sample.json` (the records, the checkpoints'
    sha256, the seeds, the oracle's largest difference, the timings).
- The `.npz` of a data set: `theta_ref` (n, 10) f64, `in_box` (n,) bool,
  `rows` (n,) int64 (bank rows), `donor`, `well`, `cls` (n,) int64,
  `logq_ref_members` (n, M) f64, `logq_ref` (n,) f64, `draws` (n, n_post,
  10) f32, `draw_member` (n_post,) int64, `logq_draws` (n, n_post) f32,
  `logq_draws_own` (n, n_post) f32, `z` (M, n, E) f32, `stats` (n, 8) f32,
  `calib` (n,) bool, `ppc_rows` (k,) int64 (indices into the n rows),
  `ppc_draw_idx` (ppc_draws,) int64, `ppc_stats` (k, ppc_draws, 8) f32; the
  calibration file: `theta` (cal_n, 10), `stats` (cal_n, 8), `draws`
  (cal_n, 10), `member` (cal_n,).
- EvalConfig (`synth_eval.json`): `out_root`, `seed`, `n_post` 1000 (Q-E1),
  `batch_rows`, `oracle_tol` 2e-3 nats (in (0, 0.1]; 2026-11-12, the
  oracle above), `ppc_rows` 324 and `ppc_draws` 20 (Q-E4), `cal_glob`,
  `cal_n` 10000, `cal_shards`, `cal_seed`, `lc2st_null` 100 and
  `lc2st_eval_draws` (Q-E5), `bootstrap_n`, `level` 0.95, `dpi` 200,
  `formats` png and pdf (Q-E6), `ncpus`; [2026-11-12, block 23] `data_view`
  and `jobs.data`.
- Test oracles (`test_smoke_evaluation.py`, on the reduced project of T18,
  W 128): E19a the rebuild reproduces every member's `_perrow.npz` on D_sim
  and on its cohort; E19b theta^dag maps through the unshifted affine map
  onto the shifted map's values at theta* (`phi_to_bench_params`, float64
  roundoff), axis 7 capped, `in_box` exact; E19c the mixture log density
  equals log mean_m exp(log q_m) computed directly in float64; E19d the
  npz contract (keys, shapes, dtypes), the draws n_post / M per member,
  the same seed giving identical files; E19e the PPC windows' statistics
  finite, one window per draw; a calibration bank of one window per theta.
  [2026-11-11, at the build] E19a also refuses a record of another split
  and a folder of two arms and lists an incomplete one; E19c also compares
  the embed-once densities with sbi's `log_prob` (equal in one batch, 0.0
  in the sandbox; float32 roundoff in batches of another shape, 1.1e-5
  nats, under the oracle's 1e-4); E19d also the seeds common to the
  ensembles and five refusals that write nothing [2026-11-12: seven -- a
  tolerance under the rebuild's roundoff, 1e-6 nats, refuses naming
  `oracle_tol` and writes nothing; 0.5 is a configuration error]; E19e recomputes the PPC
  statistics from the stored draws in one process (the sampler's two give
  the same bytes) and checks the calibration bank's arguments against
  `calbank_build`. Six mutations of the code (the mixture's -log M, the
  duty's cap, the members' order of draws, the PPC draws' spread, the flow
  read outside the box, windows misaligned with their draws) each fail a
  check.
- Confirmed: yes 2026-11-11 (D-127: Q-E1..Q-E6 as proposed).
- Status: smoke-tested 2026-11-11 (`test_smoke_evaluation.py` E19a-e: 5
  pass and 1 skip (convergence, not a discretisation) in 100 s
  `[SANDBOX]`, the reduced project built from scratch in 66 s; the six
  mutations above each caught). Not run on davinci: the sampling of the
  real ensembles is block 22's job (W 18,000, n_post 1,000, M 5; a bench
  window of that size simulates in 0.05 s `[SANDBOX]`). [2026-11-12, on
  davinci `[CLUSTER, the user's paste of 15:38]`: S-A1's sampling done in
  756 s, oracle 9.5e-6 nats; the four others refused on the then 1e-4
  (the oracle above). Amended the same day: `oracle_tol` in the
  configuration, two refusal cases in E19d; `test_smoke_evaluation.py`
  14 of 15 checks pass, the convergence check skipped, from a fresh reduced
  project in 278 s; the oracle disabled fails E19d `[SANDBOX 2026-11-12]`.]

### Block 20: the evaluation's quantities (added 2026-11-11; confirmed 2026-11-11, D-127; roadmap synth, step S6)

- Modules: `hpc/pipeline/evaluation/ev_metrics.py` (new), checks E20* of
  `test_smoke_evaluation.py`.
- Public API: `python3 hpc/pipeline/evaluation/ev_metrics.py --config
  <tag>.json --eval-config <tag>_eval.json [--ensemble <q>:<arm> ...]
  [--ncpus N]` writes, under `<out_root>/metrics/`, `<stem>_<D>.json` per
  ensemble and data set, `paired_<q>_<D>.json` per search with a term and
  data set [amended 2026-11-16, block 27: `paired_<name>_<D>.json` per
  comparison of sec. 2.5 and data set], and `search_<q>.json` per search (pass line `EVAL METRICS OK`);
  [amended 2026-11-11 at the build] beside each `<stem>_<D>.json` and
  `paired_<q>_<D>.json` an `.npz` of the per-row arrays and curves the
  figures draw (block 21 computes nothing), and `METRICS.json`, the index
  (the files, the ensembles reported, those left out and why, the report's
  inputs); `ev_metrics.compute(cfg_path, ecfg_path, names, ncpus)` the
  same in a process. Each ensemble's quantities in a worker of its own
  (one thread), the paired comparisons and the searches after them; an
  ensemble not sampled, or sampled with other values than
  `ev_config.SAMPLE_PARAMS`, is left out and named; none current is a
  refusal (exit 3).
- Rule (sec. 2.5's quantities; estimators):
  - NLL: l_i(E) = -logq_ref; the mean over the in-box rows; the donor
    means lbar_d(E); each member's mean;
  - HPD: gamma_i(E) = (1/n_post) #{s : logq_draws[i,s] > logq_ref[i]}; C_E(beta)
    on beta = 0, 0.01, ..., 1 (E2); D_sim over its calibration rows (Q-E2),
    with a binomial 95 % band; a cohort over every window, with a 95 %
    band from `bootstrap_n` resamples of the donors; the same per member
    from `logq_draws_own` and `logq_ref_members` on its own draws;
  - SBC: `npe_diagnostics.simulation_based_calibration` (ranks, KS p-values)
    on the same rows, the ten axes' family by `holm_bonferroni`;
  - TARP: `npe_diagnostics.tarp` twice, with reference points read out
    linearly from the windows' eight statistics (`reference_mode="x"`,
    primary) and drawn uniformly (`"random"`, Lemos et al.'s experiments);
    the x-dependent form because x-independent points cannot see a
    posterior that ignores the data (the function's docstring: 1 of 12
    detections against 12 of 12, on a posterior equal to the prior) -- an
    amendment to sec. 2.5, which named the uniform points only;
  - recovery: mhat, sigma, the 5 % and 95 % quantiles per row and axis, the
    90 % interval's coverage per axis, zeta, the contraction 1 -
    mean_i sigma_{i,k}^2 / (1/12);
  - paired (E3) [amended 2026-11-16, block 27, D-136: the pairs are sec.
    2.5's comparisons, E_kappa and E'_kappa in place of E_{q,A(q)} and
    E_{q,A1}; the rest of this rule stands]: for each search q with a term and each data set D it
    shares, d_i = l_i(E_{q,A(q)}) - l_i(E_{q,A1}) on the in-box rows,
    `bootstrap_paired.paired_bootstrap(d, groups=donor, within="all")` (the
    primary scheme; one copy of each donor), sigma_seed the larger over the
    two arms of the SD over members of the members' mean NLL on D, and the
    verdict of G.2: |Dbar| > 2 sigma_seed and the interval excluding 0;
  - embeddings: for each member, `npe_diagnostics.embedding_overlap(z on
    D_sim, z on the cohort)` (MMD^2 at the median bandwidth, geodesic, with
    its permutation null), and the first two principal components of the
    pooled embeddings (`sklearn.decomposition.PCA`);
  - PPC: for each PPC row and statistic, the rank of the observed value
    among the `ppc_draws` simulated ones and whether it lies outside their
    5-95 % range; per data set the fraction outside, per statistic;
  - L-C2ST (Q-E5): sbi's `LC2ST` on the calibration file (thetas, the
    statistics as xs, the draws), `lc2st_null` permutation trials, the
    statistic and p-value at each cohort window with `lc2st_eval_draws` of
    its draws, and at D_sim's PPC rows as the reference; per donor the mean;
    [2026-11-11 at the build] the features z-scored (`z_score=True`: the
    eight statistics have different units), the seed given to
    `train_on_observed_data(seed=...)` -- in sbi 0.27 the constructor's seed
    reaches the folds alone and the classifiers were unseeded, so two runs
    differed (`[SANDBOX]`); with it, the same statistics in one worker and
    in three; theta-dagger's data set (the windows of its cohort) is not
    tested again;
  - the search: trials in proposal order (`pipeline_rounds.json`), each
    with its arm, `nll` and `L_pseudo_real`; every member's `history`;
    [2026-11-11 at the build] the running minimum of `nll` per arm over the
    trials that ended ok, and the search's finalist;
  - [2026-11-11 at the build] the summaries: C(beta)'s signed area against
    the diagonal (negative: overconfident), its largest deviation, C at
    0.50, 0.68, 0.90, 0.95, and the expected coverage's KS test
    (`npe_diagnostics.expected_coverage`); SBC's ECDF differences of the
    normalised ranks on the beta grid with the KS band
    (`scipy.stats.kstwo.ppf(level, n)`) and a 20-bin rank histogram with its
    per-bin binomial 99 % band; the MMD^2 and its square root; NaN and inf
    written as null; [2026-11-11, for block 21] the per-row contraction
    1 - sigma^2 / sigma_0^2 (`contraction` in the `.npz`), L-C2ST's 95th
    percentile of the statistic (`stat_q95`, D_sim's being the figures'
    reference), and in `search_<q>.json` the corner plots' example donors
    (`example_donors`: per cohort and condition the donor of median lbar_d
    under the search's A1 ensemble, or S-A1's while that one is not
    reported, with the donor's first row).
- Test oracles (`test_smoke_evaluation.py`): E20a the quantities on cases
  with a known answer -- the HPD rank and C(beta) by hand; the conjugate
  Gaussian model on 10 axes (theta ~ N(0, I), x = theta + N(0, 0.25 I), the
  exact posterior N(x / 1.25, 0.2 I)): its curve within the KS bound at
  0.999 of the diagonal, a posterior half as wide gives a signed area below
  -0.1 and one twice as wide above 0.1; the binomial band's half-width
  1.96 sqrt(0.25 / n); the donor bootstrap's band unchanged by twelve
  copies of each donor's row; SBC rejecting the one shifted axis alone;
  TARP's x-dependent references seeing a posterior equal to the prior that
  the random ones do not; the recovery of N(mu_i, 0.1^2) (contraction 0.88,
  coverage 1, z-scores 0; shifted by 2 SDs: coverage 0, z-scores -2); the
  NLL, the PPC, E3's two clauses, the running minimum by hand; the
  embeddings' MMD p-value above 0.01 for one law and at most 0.01 for two;
  E20b LC2ST through `lc2st_fit` and `lc2st_eval`: the prior's draws give a
  larger statistic and more p-values below 0.05 than the exact
  posterior's; E20c the reduced project's three ensembles: every file of
  the index, the NLL, C(beta), its rows and E3 recomputed from the samples,
  theta-dagger's out-of-box rows out of the NLL and in SBC, the search's
  trials in the rounds' order with their arms and finalist, the same files
  from one worker and from three, a sample of other values left out and
  none current refused, an unknown ensemble refused.
- Confirmed: yes 2026-11-11 (D-127).
- Status: smoke-tested 2026-11-11 (E20a-c pass `[SANDBOX]`: 3.3 s, 14.5 s,
  10 s on the kept fixture; five mutations -- C(beta) with <, a bootstrap
  of rows, the prior's variance 1/6, sigma_seed the smaller, the D_sim
  curve over both windows -- each caught, the last after E20c learned to
  count the rows, the fixture's gammas being all 1). Not run on davinci.

### Block 21: the evaluation's figures (added 2026-11-11; confirmed 2026-11-11, D-127; roadmap synth, step S6)

- Modules: `hpc/pipeline/evaluation/ev_figures.py` (new; matplotlib, Agg;
  reads the samples and the metrics and computes nothing scientific),
  checks E21* of `test_smoke_evaluation.py`.
- Public API: `python3 hpc/pipeline/evaluation/ev_figures.py --config ...
  --eval-config ...` writes `<out_root>/figures/<family>/<name>.<fmt>`, a
  `FIGURES.md` index with one line per figure, pass line `EVAL FIGURES OK:
  <n> figure(s)`; `ev_figures.make_all(cfg_path, ecfg_path)` the same in a
  process; exit 3 when there are no metrics.
- The figures (D-126, D-127; one sentence of caption each in the index):
  per ensemble -- calibration: coverage curves (members thin, ensemble
  thick, the band) on D_sim and each cohort, SBC ECDF differences with the
  KS band and rank histograms, TARP (both reference modes); accuracy and
  sharpness: recovery per axis, zeta against contraction, sigma against
  sigma_0, l_i per window and lbar_d per donor; posterior views and checks:
  corner plots of the example donors (one per condition, median lbar_d
  under the search's A1 ensemble, Q-E6), the PPC per statistic and its
  ranks, L-C2ST per window and per donor; embeddings: the first two
  principal components of D_sim and the cohort by condition, with MMD^2;
  [amended 2026-11-16, block 28, D-136: the two ensembles of each of sec.
  2.5's comparisons, under `comparisons/<name>/`, in place of the per-search
  `arms_*`; `across/dbar` over the comparisons, one panel per kind]
  per search -- the two arms' coverage, NLL, PPC and L-C2ST overlaid, the
  donor-paired forest plot (E3) with 2 sigma_seed; across searches -- Dbar
  with its interval and 2 sigma_seed for every search and data set, the
  ensembles' L_pseudo_real on R0 and R3; the search -- L in proposal order
  with its running minimum per arm, L against L_pseudo_real over all
  trials, the members' learning curves.
- [2026-11-11 at the build] The files: `ensembles/<stem>/` with the four
  families as prefixes -- `cal_coverage`, `cal_tarp`, `cal_sbc_ecdf_<D>`,
  `cal_sbc_hist_<D>` (every data set); `acc_recovery_<D>`, `acc_zscore_<D>`
  (every data set), `acc_sigma`, `acc_nll`; `post_ppc_<D>` (D_sim and the
  cohorts), `post_lc2st`, `post_corner_<cohort>_c<k>` (the example donor
  of condition k, its first window, theta* marked, and theta-dagger where it
  differs); `emb_pca_<cohort>` -- then `searches/<q>/` with `arms_coverage`,
  `arms_nll_<D>` (every data set but D_sim), `arms_ppc_<D>` and
  `arms_lc2st_<D>` (the cohorts), `arms_forest_<D>` (every paired data
  set), `search_trials`, `search_nll_vs_lpr`, `search_learning`, and
  `across/dbar`, `across/lpr`. [superseded 2026-11-16 by block 28: no
  `arms_*` figure; each comparison's figures under `comparisons/<name>/`
  (`cmp_*`); `searches/<q>/` keeps `search_*`; `across/dbar` over the
  comparisons, one panel per kind] The example donors are block 20's
  (`search_<q>.json`'s `example_donors`: per condition the donor of median
  lbar_d under the search's A1 ensemble, S-A1's while that one is not
  reported); the per-row contraction and the D_sim reference of L-C2ST
  (its 95th percentile) are block 20's too, so that nothing is computed
  here. The colours follow the entity in every figure, after the dataviz
  method: an arm its hue (A1 #2a78d6, A2 #eb6834, A5 #1baf7a), a condition
  its hue and marker (#e87ba4 circle, #6250d6 square, #008300 triangle;
  both triplets PASS the palette validator all-pairs in light mode, the
  contrast WARN of aqua and magenta relieved by the legends and the
  metrics' files as the table view, the tritan 7.6 of the conditions by
  their markers), the text in the ink colours, gray for references and
  D_sim as context, one y-axis per panel, the legend in a row below the
  panels.
- Test oracles: E21a on the reduced project's metrics: the set of figures
  equals the one the rules above give (an independent statement in the
  test), each listed in `FIGURES.md` with a caption and none unlisted,
  every PNG's signature, the corner plots at the example donors; E21b the
  module imports no statistics library (scipy, sklearn, torch, sbi,
  npe_diagnostics, bootstrap_paired, ev_metrics) and calls no mean, SD,
  quantile or median; the colours the validated ones; a figure to PDF; no
  metrics refused.
- Confirmed: yes 2026-11-11 (D-127).
- Status: smoke-tested 2026-11-11 (E21a-b pass `[SANDBOX]`: 84 figures in
  60 s at 60 dpi on the kept fixture; looked at by Claude at 120 dpi --
  coverage, recovery, the corner plot, the forest plot, the NLL, the PCA,
  the PPC comparison, L-C2ST, SBC -- and corrected: legends moved below
  the panels, the corner's theta-dagger marked only where it differs, the
  PPC comparison as dots, integer trial ticks; two mutations -- the
  embedding figures dropped, a quantile computed in the figures -- each
  caught). Not run on davinci.

### Block 22: the evaluation's jobs and run sheet (added 2026-11-11; confirmed 2026-11-11, D-127; roadmap synth, step S6)

- Modules: [amended 2026-11-11 at the build: one job script instead of
  three, and the launcher in Python] `hpc/pipeline/evaluation/jobs/eval_run.pbs`
  (one job = one command line, as the pipeline's `run_argv.pbs`: the
  command one argument per line in `ARGV_FILE`, the activation by
  `hpc/pipeline/jobs/activate_env.sh` (D-111: conda with `set -u` off across
  the hook and the activate, then the environment's folder), then the
  roadmap's markers with `ROADMAP_STAGE` from the launcher, I19 for the
  calibration bank and the samplings, I22 for the report; exits 90, 91, 95,
  96 or the command's; last line `EVAL JOB EXIT: <n>`),
  `hpc/pipeline/evaluation/ev_submit.py` (the launcher, standard library
  only) and `submit_eval.sh` (its wrapper on the login node: `--project
  <tag>` reads `<tag>.json` and `<tag>_eval.json`, runs the launcher in
  `envs.extract` as `run_pipeline.sh` does), `ev_report.py` (the report
  job's entry: blocks 20 and 21, then `REPORT_COMPLETE.json`), the README's
  run sheet "synth: the evaluation", `synth_eval.json`'s `jobs` (the queue
  and each kind's ncpus, mem, walltime: calbank 1, 16gb, 4 h; sample 8,
  64gb, 24 h; report 8, 64gb, 24 h, Claude's), checks E22* of
  `test_smoke_evaluation.py`; `tests/fake_pbs/fake_pbs.py` learns
  `afterok:<id>:<id>...` (several ids).
- Public API: `bash hpc/pipeline/evaluation/submit_eval.sh --project <tag>
  [--dry-run] [--what calbank,sample,report] [--resample <q>:<arm>,...|all]`
  ([2026-11-12, block 23] `--what` also takes `data`, block 23's job, which
  the default `calbank,sample,report` leaves out)
  (= `ev_submit.py --config ... --eval-config ...`); exit 0 with `EVAL SUBMIT
  OK: <n> job(s) submitted, <d> done, <l> live` (`EVAL DRY RUN: ... nothing
  written`), 2 on the arguments or the configurations, 3 on a refusal
  (nothing submitted), 4 when a qsub fails.
- Rule: the units, each done, live (its job in `<out_root>/jobs/ledger.json`
  queued, held or running, by `qstat -x -f`) or submitted:
  - `calbank:<k>`, shard k of `ev_plan.calbank_build` (`build_latent_bank.py
    --arm S --prior box --wells-per-donor 1 --n-per-theta 1 --n-windows 1`
    at `cal_seed`, `cal_n / cal_shards` donors per shard, the simulated
    bank's other options; [2026-11-11: `--n-per-theta 1` added, the builder
    refusing its default 2 with one well per donor]): done when the shard
    and its sidecar exist with the build's arguments; a sidecar of other
    arguments, an unreadable one, or a shard index beyond the build is a
    refusal, never overwritten;
  - `sample:<q>:<arm>`, one per complete ensemble (`ev_plan.ensembles`):
    done when its `<stem>_sample.json` lists files that all exist, with the
    current `ev_config.SAMPLE_PARAMS` and its best fit's current
    `BEST_FIT_COMPLETE.json` sha256; otherwise absent or stale (the reason
    printed), or `--resample`; after every calibration job submitted now or
    live (`-W depend=afterok:<id>:...`); an ensemble whose best fit is not
    complete is named (`not complete: <q> (<folder>: ...)`), not sampled;
  - `report`, one: submitted when a sampling job is submitted now or live,
    or when `REPORT_COMPLETE.json` records other inputs than
    `ev_plan.report_inputs` (the samples' provenance digests and
    `ev_config.REPORT_PARAMS`; [amended 2026-11-16, block 27] and the
    comparisons); after the sampling jobs; a live report is
    not submitted twice (the launcher says the sampling submitted now is not
    in it);
  - every job is `eval_run.pbs` in `envs.train` with its command in
    `<out_root>/jobs/<unit>.argv` and its log in `<out_root>/jobs/logs/`,
    job names of at most 15 characters (`ev_cal<k>`, `ev_s<i><b|a>` for
    search i's best fit or the other arm's, `ev_report`); a program or the
    job script missing from the checkout (`paths.sbi_dir`) is a refusal;
    nothing of the pipeline's state is read or written.
- Test oracles: E22a the launcher's dry run on the committed `synth.json`
  (its `sbi_dir` at this checkout) and `synth_eval.json`: five calibration
  qsub lines with the committed resources, `ROADMAP_STAGE=I19`,
  `SKIP_CONDA=0` and the builder's arguments of `calbank_build`, nine
  folders not complete [amended 2026-11-16: five under `synth.json` v4,
  one per search and no other arm's fit; the check counts five since block
  26], nothing written; the wrapper's plan equal to the
  launcher's; `--project`, a bad `--what`, an absent project (skipped where
  the runs root exists); E22b `eval_run.pbs`: bash -n, ASCII and LF,
  executable, the activation before the markers, no bare `conda activate`;
  exits 90, 91, 95, 96 and the command's own (7), an argument with a space
  kept; the markers pushed to a stand-in origin, `done` for exit 0 and
  `failed` for exit 7; E22c the reduced project under the fake PBS: five
  jobs, each sampling after both calibration jobs, every exit 0, the job's
  draws equal to the sampler's in the test process; run again: five done; a
  queued job live and not submitted twice; `n_post` changed: three stale;
  `--resample`; an unknown ensemble and a calibration bank of another seed
  refused with nothing written; a changed `BEST_FIT_COMPLETE.json`: stale.
  E22d the evaluation end to end on the reduced project under the fake
  PBS: two calibration jobs, three samplings and the report after them
  (`afterok` on the three, stage I22), every exit 0, the report's three
  pass lines, `REPORT_COMPLETE.json` with the current
  `ev_plan.report_inputs` and as many figures as `FIGURES.md` lists; run
  again, six done; `--resample` of one ensemble puts the report after its
  sampling; a report value changed (dpi) makes the report alone stale.
  [2026-11-11 at the build] The report is not submitted while no ensemble
  is sampled or being sampled (`report: nothing to report yet`), so the
  committed dry run on a fresh davinci shows five jobs (E22a).
- Confirmed: yes 2026-11-11 (D-127).
- Status: smoke-tested 2026-11-11: part 1, the calibration bank and the
  samplings (E22a-c, four mutations -- the samplings not after the bank,
  the sampling's values ignored, live jobs ignored, the markers before the
  activation -- each caught; the launcher's modules need Python 3.3 or
  later, `vermin`); part 2, the report (`ev_report.py`, E22d; four
  mutations -- the report submitted with nothing sampled, not after the
  samplings, recording its samples only, a staleness test blind to the
  report's values -- each caught). The whole suite from a fresh reduced
  project: 14 of 15 checks pass, the convergence check skipped, in 262 s
  `[SANDBOX 2026-11-11]`. Not run on davinci: part 1 handed over at
  59d7f90 (the run sheet's E1-E5), part 2 with the run sheet's E6-E9.

### Block 23: the data overview (added 2026-11-12; confirmed 2026-11-12, D-130; roadmap synth, step S6)

- Modules: `hpc/pipeline/evaluation/ev_dataview.py` (new: the quantities
  and the job's entry; numpy, torch through `ev_simulate.fixed_stats`,
  scikit-learn's PCA), `ev_data_figures.py` (new: the figures; matplotlib,
  Agg; reads `data_view.npz` and computes nothing scientific; block 21's
  style and colours), `ev_data.py` (block 19's `cohort()` also returns
  `window_idx` and `contaminated`), `ev_config` (`data_view`, `jobs.data`,
  `DATA_PARAMS`),
  `ev_plan` (`data_dir`, `data_view_json`, `cohort_glob`,
  `data_bank_paths`, `data_inputs`), `ev_submit` and `submit_eval.sh` (the
  kind `data`), `synth_eval.json` (`data_view`, `jobs.data`), checks E23* of
  `test_smoke_evaluation.py`.
- Public API: `python3 hpc/pipeline/evaluation/ev_dataview.py --config
  <tag>.json --eval-config <tag>_eval.json [--ncpus n]` writes
  `<out_root>/data/data_view.npz`, the figures
  `<out_root>/data/figures/<name>.<fmt>` with `<out_root>/data/FIGURES.md`,
  and last `<out_root>/data/DATA_VIEW.json` (schema `eval_data_view/1`: the
  inputs, the counts, the PCA's ratios and loadings); pass line `EVAL DATA
  OK: <n_S> simulated windows, <n_R0> + <n_R3> cohort windows, <k>
  figure(s)`; exit 2 on the arguments, 3 on a refusal (fewer shards than
  `sim_shards`, a zoom longer than a window, a statistic constant over
  S_sub), nothing written then. `ev_dataview.compute(cfg, ecfg) -> (arrays,
  info)`; `ev_data_figures.make_all(data_dir, ecfg) -> [(name, caption)]`.
  `bash hpc/pipeline/evaluation/submit_eval.sh --project <tag> --what data
  [--dry-run]`: one job `ev_data` (`jobs.data`: synth 2 cores, 8gb, 1 h,
  Claude's), stage I23, after nothing; done when `DATA_VIEW.json` records
  the current `ev_plan.data_inputs` (`ev_config.DATA_PARAMS` -- seed, dpi,
  formats, data_view -- and, per bank, each shard read with the sha256 of
  its sidecar), stale otherwise; `data` is not in the default `--what`, so
  the evaluation's plan (block 22) is unchanged unless it is named.
- The data (D-130): S_sub, the first `data_view.sim_shards` shards, sorted,
  of the simulated bank -- synth: 2 of 20, n_S = 4,000 windows of 2,000
  donors, each donor's theta, drawn from the box prior p_S(theta) =
  U(0, 1)^10, shared by its two wells; every window of S_sub is therefore a
  draw of the prior predictive p_S(x) = int p(x | theta) p_S(theta) dtheta,
  the realisation and the nuisance included, two windows of a donor not
  independent; R0 and R3, the cohorts of `synthetic.real`, n_R = 324
  windows each (27 donors, 9 per condition, 2 wells, 6 windows per well),
  theta*_d read here although the networks never see it.
- The quantities (`compute`; D-130, the finer choices [Claude's]):
  - T(x) in R^8, the eight fixed statistics of every window
    (`ev_simulate.fixed_stats`, block 19: mean, SD, max, 90 % quantile,
    burst fraction, lag-1 autocorrelation, skewness, roughness);
  - the reference standardisation on S_sub: m_j and s_j, the mean and the
    SD (ddof 1) of T_j over S_sub in float64, and z(x) = (T(x) - m) / s,
    componentwise; a statistic constant over S_sub is a refusal;
  - PCA: scikit-learn's `PCA(n_components=2, random_state=seed)` fitted on
    z of S_sub alone; every window's two coordinates, the explained variance
    ratios, the loadings (2 x 8);
  - per statistic, S_sub's quantiles at 0.01, 0.5 and 0.99 (numpy, linear)
    and each cohort's counts of windows below the first and above the last;
  - the frames: per statistic, the range of T_j over S_sub and every
    uncontaminated cohort window (the bank's `contaminated`: gap mode (d)),
    widened by 2 % of its width on each side; the same for the PCA plane;
    a window outside a frame is necessarily contaminated (asserted),
    counted, and drawn at the frame's edge;
  - the histograms: `bins` equal bins over each frame; heights count /
    (n_D x width), so that a data set's area over the frame is the fraction
    of its windows inside;
  - theta: per cohort donor d, theta*_d (every row of d carries it,
    asserted) and c_d; theta^dag_d, in_box and delta by
    `ev_data.pseudo_true` (sec. 2.5: R3 delta = 0.35 x 0.3 = 0.105; R0
    theta^dag_d = theta*_d); the conditions' centres mu_c (the sidecar's
    `class_centres`, C x 7, equal in both cohorts, asserted);
  - the examples (D1): per cohort and condition c, `examples_per_condition`
    donors of c drawn without replacement by
    `numpy.random.default_rng(ev_plan.seed_of(seed, "data_view", <cohort>,
    c))`, sorted, each shown by the first window of its first well; per
    cohort with contaminated windows, as many contaminated windows drawn the
    same way (key "contaminated"); `sim_examples` rows of S_sub (key "sim");
    each example's zoom start z0 = min(max(argmax_t x_t - floor(L/2), 0), W
    - L), L = round(zoom_s f_s) samples (a zoom longer than the window is a
    refusal); the strips' jitter uniform in (-0.28, 0.28), seeded the same
    way (keys "jitter", "jitter_d").
- The figures (D-130; formats and dpi of the evaluation's configuration;
  one caption line each in `FIGURES.md`; the conditions' validated hues and
  markers of block 21, S_sub gray, the cohorts told apart by line style or
  panel, text in the ink colours, one y-axis per panel, the legend below):
  - D1: `windows_cohorts` (rows: condition x example, columns: R0, R3; the
    full window, the zoom shaded; y shared by a row), `windows_cohorts_zoom`
    (the same windows, zoom_s around each maximum), `windows_sim` (S_sub's
    examples, full and zoomed side by side), `windows_contaminated` (the
    contaminated examples, full and zoomed; only where a cohort has some)
    [Claude's];
  - D2: `stats_hist` (one panel per statistic: S_sub's histogram as a gray
    fill, each cohort's as an ink step line, solid R0, dashed R3; the
    windows beyond the frame counted in the panel's title), `stats_strip`
    (one panel per statistic: every cohort window as a point in its
    cohort-and-condition row, contaminated windows ringed, S_sub's 1-99 %
    band and median; the counts below and above the band in the title);
  - D3: `theta_cohorts` (one panel per axis: each cohort donor's theta*_{d,k}
    in its cohort-and-condition row on [0, 1], where p_S is uniform; the
    conditions' centres as bars on the label axes), `theta_dagger` (the free
    axes 7-9 of each range-shifted cohort: theta^dag_{d,k} against
    theta*_{d,k}, with y = x, y = x + delta and the box's edge 1; only where
    a cohort has delta > 0);
  - D4: `stats_pca` (S_sub's cloud in gray with each cohort's windows by
    condition, one panel per cohort on the same frame; a full-extent panel
    for a cohort with windows beyond the frame; the loadings).
- Test oracles: E23a on the reduced project (W 128, f_s 10 Hz; S_sub the
  first 2 of 3 shards, 80 windows): S_sub's rows those of the first two
  sorted shards read with numpy; the eight statistics against their
  definitions recomputed in float64 (1e-5 on the means and SDs, one sample
  in W on the burst fraction); m and s against numpy; the PCA's ratios
  against the top eigenvalues of the covariance of z over its trace, its
  loadings against the eigenvectors up to sign, every coordinate against
  (z - mean z) times the loadings; the quantiles and the counts outside
  them; the frames holding S_sub and every uncontaminated window; each
  data set's histogram area equal to its fraction inside; the helpers on
  known cases (half the values beyond the edges leave half the area; a
  constant widened; the zoom at both ends); theta*_d the rows' own,
  theta^dag_d `pseudo_true`'s, delta = pi x shift_max of the sidecar (R3
  0.105, R0 0); every cohort example the first window of a donor's first
  well, one per condition, contaminated examples contaminated, every zoom
  holding its window's maximum; the same arrays twice, other examples under
  another seed; refusals (3 shards for 4; a zoom of 20 s in 12.8-s windows)
  and the configuration's checks; E23b the job's entry: its pass line, the
  figures the rules give (an independent statement in the test) in PNG and
  PDF, each in `FIGURES.md` with a caption, `DATA_VIEW.json` with the
  current `ev_plan.data_inputs`; `ev_data_figures.py` imports no statistics
  library and calls no mean, SD, quantile, median, histogram, fit,
  transform, argmax or random draw; exit 2 on the arguments, 3 on a
  refusal with nothing written; E23c `--what data` on the committed
  configuration: one qsub line with `jobs.data`'s resources, stage I23, the
  command, nothing written; no data unit in the default plan; on the
  reduced project under the fake PBS: the job's pass line and exit 0; run
  again, done; a `data_view` value changed, and a cohort's sidecar changed,
  stale.
- Confirmed: yes 2026-11-12 (D-130: D1-D4 and the subset of 2 shards are
  the user's; the finer choices marked [Claude's]).
- Status: smoke-tested 2026-11-12 (E23a-c pass `[SANDBOX]`; twelve
  mutations -- ddof 0, the last window instead of the first, the zoom on
  the minimum, the PCA on raw statistics, theta-dagger dropped, the windows
  scaled, the histograms normalised by the windows inside, a mean in the
  figures, the record's inputs emptied, staleness blind to the inputs, data
  in the default plan, the job at stage I22 -- each caught); on the
  full-size banks rebuilt here with davinci's seeds (R0, R3 and the first
  two simulated shards) the job takes 13 s and 1.2 GB, its nine figures
  looked at by Claude. The whole suite from a fresh reduced project in a
  clean clone of `bb37614`: 17 of 18 checks pass, the convergence check
  skipped, in 319 s `[SANDBOX 2026-11-12]`. Not run on davinci.

### Block 24: the purge of a synthetic project, with the archive (added 2026-11-13; D-132; roadmap synth, step S5)

The deletion of everything a synthetic project's last run made, before
its relaunch (D-132: "remove everything from the last session of
synthetic data generation and optimization and training"; "Archive
results, delete rest"). Block 10's tool, extended: the same
list-then-delete discipline with a CODE (Claude never deletes data; the
user deletes through the tool), five kinds for a configuration of kind
`synthetic`, and an archive of the evaluation's results written and
verified before anything is deleted. [the user's choices of 2026-11-13
(D-132); the targets, the archive's content and the refusals proposed by
Claude; Confirmed: no]

- Modules: `hpc/pipeline/tools/purge_outputs.py` (block 10's tool: the
  synthetic kinds, the archive, the evaluation's ledger in the jobs
  check), `hpc/pipeline/tests/test_smoke_purge.py` (checks T24a-T24e, the
  mutation `evaljobs`); the README's run sheet "synth: the relaunch".
- Command: `python3 hpc/pipeline/tools/purge_outputs.py --config
  <synth.json> --what <kinds> [--eval-config <file>] [--archive
  <file.tar.gz> | --no-archive] [--confirm <CODE>]`, on a login node,
  stdlib only. For a configuration of kind `synthetic`, `<kinds>` is a
  comma-separated subset of `evaluation`, `best_fits`, `searches`,
  `banks`, `state`; block 10's kinds are refused for it, and these five
  for any other kind (exit 2). `--eval-config` defaults to
  `<the configuration's folder>/<dataset>_eval.json`; it must load when
  `evaluation` is named or when the file exists (exit 2 otherwise), and
  its ledger enters the jobs check whatever the kinds; `--eval-config`,
  `--archive` and `--no-archive` are refused for any other kind (exit 2).
- Targets, from the configurations, listed and deleted in this order:
  `evaluation` -- the evaluation's `out_root`; `best_fits` --
  `<runs_root>/<q>` for each search q of `synthetic.searches` (the run
  view's `runs_root`: the best fit `<dataset>_best` and the other arm's
  `<dataset>_best_alt`, block 18); `searches` -- `<tune_root>/<q>` for
  each q (the run view's `tune_root`: the campaign folder and any rounds
  set aside); `banks` -- `bank_dir` of `sim` and of each real bank
  (`<banks_root>/<dataset>_sim`, `<banks_root>/<dataset>_real_<name>`);
  `state` -- every entry of the work root except `purge/`, by name (the
  state file and its `.prev`, the log, the reports, the sequencer's
  folders), files allowed. Never touched, printed as kept:
  `<work_root>/purge` (the purge records). A missing state file is a
  fresh start (`state.py`), so after `state` the next submission starts a
  new run.
- The archive (`--archive <path>`, `~` expanded): required when
  `evaluation` is named and its `out_root` exists, unless `--no-archive`
  (exit 2 otherwise); every regular file under `out_root` except its
  top-level `samples/` and `calbank/` subtrees, member names relative to
  `out_root`'s parent (so they start with its folder name), sorted. The
  listing prints its file count, bytes (uncompressed) and the two skipped
  subtrees' counts and bytes. Exit 2: the path exists; it lies under
  `out_root` or under a target; its folder does not exist; `--archive`
  without `evaluation`; `--archive` with `--no-archive`. Exit 4 (refused):
  a symbolic link or a non-regular file among the members' folders. With
  `--confirm`, before the first deletion: written to `<path>.tmp` (gzip,
  level 6), re-read and every member's name and size compared with the
  listing, renamed to `<path>`, and `<path>.sha256` written (`<sha256>
  <name>`); a failure prints `PURGE ARCHIVE FAILED: <why>; nothing
  deleted`, removes the `.tmp` and exits 5.
- The CODE: block 10's, with the archive's path and its members (name,
  size, mtime in whole seconds) added to the canonical text when an
  archive is asked; block 10's canonical text unchanged otherwise.
- Refused (exit 4, nothing deleted), besides block 10's refusals: a job of
  the evaluation's ledger (`<out_root>/jobs/ledger.json`) not finished per
  the scheduler. A target that is a symbolic link or resolves outside its
  root is refused as in block 10 (for `evaluation` the root is
  `out_root`'s parent).
- `--confirm <CODE>`: the record `<work_root>/purge/PURGE_<utc>.json`
  (block 10's, with `archive`: its path, member count, bytes, result and
  sha256); the archive; then each present target removed in the listing's
  order (`shutil.rmtree` for a folder, `os.remove` for a file), one
  `DELETED` line each; the pass line `PURGE DONE: <k> target(s) deleted,
  <n> file(s), <x> GB; archive <path> (<m> file(s), sha256 <16 hex>)`, the
  archive clause `no archive` when none was written, and `; the work root
  keeps purge/ only: the next submission starts a new run` when `state`
  was named (exit 0); `PURGE INCOMPLETE: ...` (exit 5) as in block 10.
  Block 10's pass line, kinds and behaviour for other configurations are
  unchanged.
- Test oracles (`tests/test_smoke_purge.py`, checks T24a-T24e; a reduced
  synthetic project from `test_smoke_synthetic._small` with two searches
  and an evaluation configuration, under the fake PBS): the trees -- the
  three banks with a record and its `.sha256`, each search's campaign
  folder with a ledger and a set-aside folder, each search's best fit and
  other-arm best fit, the evaluation (samples, calbank, metrics, figures,
  data, jobs with a ledger of finished jobs), the work root (the state
  file, its `.prev`, the log, a reports folder, an older purge record) --
  and decoys (another data set's bank, a file beside the banks): T24a the
  listing names the targets in the documented order with `find`'s counts,
  the archive's members (none under `samples/` or `calbank/`) and the
  skipped subtrees, prints a CODE and deletes nothing; the CODE equals the
  formula computed here, is the same twice, and changes when a result
  file changes; T24b with the CODE: the archive's members are the
  listing's names and sizes with the listing's bytes, its `.sha256`
  matches, every target is gone, `purge/` keeps the older record unchanged
  and a new one `done`, the decoys are unchanged, the pass line; a second
  listing finds every target absent and asks for no archive; T24c refused
  with exit 2 and nothing written: `evaluation` without `--archive`; an
  archive path that exists, under `out_root`, under a target, in a missing
  folder; `--archive` without `evaluation`; `--archive` with
  `--no-archive`; block 10's kinds on the synthetic configuration and the
  synthetic kinds on block 10's; T24d refused with exit 4 and nothing
  deleted: a live job in the evaluation's ledger, a wrong CODE, a symbolic
  link among the archive's members; T24e an archive whose writing fails
  half-way (the tar writer raising after one member, in-process): `PURGE
  ARCHIVE FAILED`, exit 5, nothing deleted, no `.tmp` and no archive
  left, the record `archive failed`. Mutations: `code` (a constant
  CODE) and the new `evaljobs` (the ledger check off: T24d's live job does
  not refuse) each make a check FAIL.
- Confirmed: no (the targets, the archive's content and the refusals are
  Claude's proposal on the user's call of 2026-11-13, D-132).
- Status: smoke-tested 2026-11-13 (T24a-T24e with block 10's checks: 11
  pass, 1 skip; the mutations `code`, `chain` and `evaljobs` each caught
  `[SANDBOX]`); the run sheet's R1, R3, R4 and P1-P4 rehearsed as written
  on a stand-in of the committed `synth.json`'s tree under the fake PBS
  and the fake conda -- R1 lists the chain's and the evaluation's jobs and
  not another chain's, R3 refuses while the sequencer's job exists and
  then deletes the two others, P1's pull and test pass, P2-P4 archive 9
  files and delete 17 targets, the other projects' banks, runs and work
  root untouched (R2 replaced by a `qdel`: no real sequencer there; R4's
  status lines not reproduced) `[SANDBOX 2026-11-13]`. Not run on davinci.

### Block 25: the bench's label axes from the configuration (added 2026-11-13; D-132; roadmap synth, step S2)

`fragment_duty` (axis 7) becomes class-bearing in the synth project
(D-132): the bench builder takes its class-bearing axes from the
configuration instead of the provider's constant, gap mode (a) displaces
the bank's own free axes, and every consumer of the split reads it from
the bank's sidecar. [the user's choices of 2026-11-13, D-132 (1)-(3); the
configuration key, the defaults and the refusals proposed by Claude;
Confirmed: no]

- Modules: `hpc/joint/stage1/bench_burst_provider.py` (`bench_free_idx`;
  `phi_to_bench_params(..., free_idx)`; `BenchBurstProvider(...,
  label_idx)`; `load_bench_provider(..., label_idx)`),
  `hpc/joint/stage1/build_latent_bank.py` (`--label-axes`),
  `hpc/pipeline/config.py` (`synthetic.builder.label_axes`),
  `hpc/pipeline/stages/synthetic.py` (`_builder_options`),
  `hpc/pipeline/projects/synth/synth.json` (v4: the label axes 0-7,
  `centre_scale` 0.562743), `hpc/pipeline/evaluation/ev_data.py`
  (`free_axes_of`, `pseudo_true(..., free_idx)`), `ev_dataview.py`,
  `ev_sample.py`, `ev_data_figures.py` (theta-dagger's caption), checks
  T25a-T25e of `hpc/pipeline/tests/test_smoke_synthetic.py`, and the
  evaluation's fixture in `test_smoke_evaluation.py` (R3's seed 34: under
  v4's split the seed 33 leaves all 6 of its donors' theta-dagger inside
  the box, and e19c needs some outside; 34 puts 3 outside).
- The split: the label axes L, a non-empty strictly increasing list of
  distinct indices in {0, ..., 9} that leaves at least one free axis; the
  free axes F = {0, ..., 9} \ L in increasing order -- `bench_free_idx(L)`,
  ValueError otherwise. The default L = (0, ..., 6), F = (7, 8, 9)
  (`BENCH_LABEL_IDX`, `BENCH_FREE_IDX`), unchanged for every caller that
  passes none.
- Gap mode (a): `phi_to_bench_params(phi, ..., free_axis_shift=delta,
  free_idx=F)` displaces the ranges [L_k, U_k] of the axes k in F only, by
  delta (U_k - L_k); the duty's clip at 1 stays and is the identity when
  the duty is not in F (its unshifted range ends at U_7 = 1).
  `BenchBurstProvider(gbd, ..., label_idx=L)` holds F = `bench_free_idx(L)`
  and passes it on every call.
- The builder: `build_latent_bank.py --label-axes i,j,...` (`--provider
  bench` only; SystemExit for another provider or a list `bench_free_idx`
  refuses): the spec's `label_idx` = L, its `class_centres` (C, |L|) from
  `class_centres_for` (the DSN generator's regular simplex on |L| axes,
  scaled about 0.5 by `--centre-scale`), the provider built with L.
  Without the flag, the bench's default. The sidecar's `latent_spec`
  records `label_idx` and `free_idx` (as before).
- The configuration: `synthetic.builder.label_axes`, required, the rule
  above (one problem per violation); `_builder_options` passes
  `--label-axes` to the shards of every bank, so arm S's sidecars record
  the same split -- its theta, drawn uniformly on the whole box, does not
  depend on it. `synth.json` v4: `label_axes` [0, ..., 7] and
  `centre_scale` 0.562743, which puts the three centres 0.441934 apart on
  8 axes, their distance on 7 axes at 0.6 (D-132 (2)).
- The evaluation: `ev_data.free_axes_of(sidecar)` is the bank's
  `latent_spec.free_idx`, or `BENCH_FREE_IDX` for a sidecar without one;
  `pseudo_true(theta, gap_spec, free_idx)` adds delta on those axes only
  and caps the duty at (1 - L_7) / (U_7 - L_7) only when the duty is among
  them; its three callers pass the cohort's own.
- Test oracles (`tests/test_smoke_synthetic.py`, T25a-T25d; numpy and
  scipy): T25a the split -- `bench_free_idx` on the default and on 0-7;
  refused: empty, duplicates, unsorted, out of range, all ten axes;
  `phi_to_bench_params` at phi 0.9 on every axis with delta 0.105: with F
  = (8, 9) the duty equals the unshifted map's and axes 8 and 9 the
  shifted map's; with the default F the parameters equal those of the
  call without `free_idx` (regression); the provider built with L = 0-7
  holds F = (8, 9). T25b the centres -- `class_centres_for` on 8 axes at
  0.562743: shape (3, 8), the three pairwise distances 0.441934 within
  1e-6 and equal; on 7 axes at 0.6 the same distance. T25c the banks --
  the builder by hand at W 128 with L = 0-7: arm R's sidecar records
  `label_idx` 0-7, `free_idx` [8, 9] and centres (3, 8); arm S's theta and
  x byte-identical to the same build with the default L (the box prior
  does not depend on the split); `--label-axes` with `--provider
  reference` and an invalid list each refused. T25d the configuration --
  the committed `synth.json` loads with L = 0-7 and the scale 0.562743;
  refused: the key missing, empty, duplicates, unsorted, all ten, out of
  range, not integers; every bank's build argv carries `--label-axes
  0,1,2,3,4,5,6,7`; `pseudo_true` with F = (8, 9) shifts those axes only
  and leaves a duty of 0.95 at 0.95, with the default F it equals the old
  function's output. Mutations: `shift` (`phi_to_bench_params` shifts the
  default F whatever it is given) fails T25a; `scale` (the committed file
  read at the scale 0.6) fails T25d. T25e (`pseudo_true`, the
  evaluation's modules; skipped without the training environment):
  with F = (8, 9) only those axes move by delta and a duty of 0.95 stays
  0.95; with the default F the output equals blocks 19-23's function
  (theta* + delta on 7-9, the duty capped); `free_axes_of` on a sidecar
  with and without `free_idx`.
- Confirmed: no (the configuration key, the defaults, the refusals and the
  sidecar as the evaluation's source of the split are Claude's proposal on
  the user's call of 2026-11-13, D-132 (1)-(3)).
- Status: smoke-tested 2026-11-13: T25a-T25e pass in the synthetic suite
  (18 of 19, the convergence check skipped, 207 s); the mutations `shift`
  and `scale` each caught (T25a, respectively T25d, FAIL); the evaluation's
  suite 17 of 18 (the convergence check skipped, 382 s) on R3's seed 34;
  stage 1's smoke tests: the bench provider 19/19, the latent SBI 46 pass
  and 1 skip, the demo classes 14/14, the burst generator 69/69 with
  `scipy.stats.kstest(x, "norm", args=...)` rerouted to the frozen
  distribution's cdf by a sandbox-only driver (scipy 1.18.1 here raises
  TypeError inside `kstest` at check B4, the same on `0e62645`; the test
  file unchanged); the other pipeline suites (blocks 1-3, 7-10, 24 and the
  tools) pass with only their convergence skips and block 1c's launcher
  skip (no Sbi-extractor checkout); the committed v4's dry run exits 0 with
  `--label-axes 0,1,2,3,4,5,6,7` on its three bank commands `[SANDBOX
  2026-11-13]`. Not run on davinci.

### Block 26: one search per arm -- the term-only campaigns (added 2026-11-13; D-132; roadmap synth, step S4)

Each arm is optimised by its own search (D-132 (4)): A1 by S-A1, as
before, and A2 and A5 by searches in which their term is always on, so no
A1 trial is nested in them; each search's best trial is refitted as a
5-seed ensemble by its best fit, and no fit of another arm is made.
[the user's choice of 2026-11-13, D-132 (4); the campaigns' names and
pins proposed by Claude; Confirmed: no]

- Modules: `hpc/joint/stage4/joint_space.py` (the campaigns `S-A2-only`
  and `S-A5-only`), `hpc/pipeline/config.py` (`_CAMPAIGNS`),
  `hpc/pipeline/projects/synth/synth.json` (v4: the five searches),
  checks T26a-T26c of `hpc/pipeline/tests/test_smoke_synthetic.py`, and
  T15's and T18's expectations moved to v4 (T18 reads a v3-like copy:
  block 18's other-arm fits stay tested on the mixed campaigns).
- `S-A2-only`: S-A2's pins (the replicate axes off, `head_pool_ops` 1)
  and `dsn_on` = 1; free: everything else, the DSN term's axes among them.
  `S-A5-only`: S-A5's pins (the DSN axes off, `head_pool_ops` 1) and
  `rep_on` = 1. `campaign()` verifies each partitions the space; every
  configuration of `S-A2-only` maps to arm A2 and of `S-A5-only` to A5
  (`npe_tune_joint.arm_for_config`). `apply_overrides` frees or pins
  their axes as it does the others'. S-A1, S-A2, S-A5 and S-A25 are
  unchanged.
- The configuration: `_CAMPAIGNS` gains the two names; `alt_best_fit` true
  stays refused for any campaign outside `ALT_ARMS` (S-A2, S-A5), so for
  the two new ones. `synth.json` v4's searches, in this order (the core
  cap's priority): S-A1 (`S-A1`, R0), S-A2-R0 (`S-A2-only`, R0), S-A5-R0
  (`S-A5-only`, R0), S-A2-R3 (`S-A2-only`, R3), S-A5-R3 (`S-A5-only`, R3),
  every `alt_best_fit` false: the graph's 13 stages (three banks, then
  each search and its best fit). The search's and the best fit's values
  are unchanged (D-123, D-125); the searches' folders keep their names,
  `<tune_root>/<q>/<campaign>` now with the new campaign names.
- Test oracles (`tests/test_smoke_synthetic.py`): T26a the campaigns --
  each partitions the space; its pins (`dsn_on` 1 and the replicate axes
  off; `rep_on` 1 and the DSN axes off); the term's own axes free; 200
  random points of each over the default space's free axes (seed 26) all
  map to A2, respectively A5; the four old campaigns' pins unchanged.
  T26b the committed v4 -- the five searches with their campaigns and
  cohorts, `alt_best_fit` false on all five, the 13 stages in order, each
  view's campaign; refused: `alt_best_fit` true on an `S-A2-only` search,
  an unknown campaign. T26c (the training environment; skipped
  otherwise) a reduced project with the one search S-A2-R0 on `S-A2-only`
  runs to PIPELINE DONE: every trial of the ledger has `dsn_on` 1 and arm
  A2, the best fit's members are A2.
- Confirmed: no (the campaigns' names and pins and the searches' order are
  Claude's proposal on the user's call of 2026-11-13, D-132 (4)).
- Status: smoke-tested 2026-11-13: T26a-T26c pass in the synthetic suite
  (T26c's reduced project to PIPELINE DONE in 41 s), T15's and T18's moved
  expectations pass (T18 on the v3-like copy); stage 4's joint-space test
  9/9 and joint-tune test 9/9 (the latter with `SKIP_CONDA=1`: its J36 runs
  the PBS element, and the sandbox has no conda); the committed v4's dry
  run exits 0 with the five searches on S-A1, S-A2-only (R0, R3) and
  S-A5-only (R0, R3), five best fits and no other arm's fit `[SANDBOX
  2026-11-13]`. Not run on davinci.

### Block 27: the comparisons across searches (added 2026-11-16; D-132 (5), D-136; roadmap synth, step S6; wave W2)

- Modules: `hpc/pipeline/evaluation/ev_plan.py` (`Comparison`,
  `comparisons`, `report_inputs`), `hpc/pipeline/evaluation/ev_metrics.py`
  (`paired_all`, `compute`'s index and log lines),
  `hpc/pipeline/evaluation/ev_submit.py` (the comparisons in the launcher's
  plan), `hpc/pipeline/tests/test_smoke_comparisons.py` (new; checks E27*).
  Amends block 19 (`report_inputs`), block 20 (E3's pairs: from "each
  search q with a term", within the search, to the comparisons of sec. 2.5)
  and block 22 (the launcher's lines and refusal).
- Public API:
  - `ev_plan.Comparison`, a namedtuple with the fields `name` (`<q>` or
    `<q>_vs_<q'>`), `kind` (`"vs_A1"` or `"A2_vs_A5"`), `E`, `E_ref` (the
    two ensembles' names, `<search>:<arm>`), `search`, `search_ref` (their
    searches' names), `arm`, `arm_ref` (their arms);
  - `ev_plan.comparisons(cfg, with_uncompared=False)` -> `[Comparison]`, or
    with `with_uncompared=True` -> `([Comparison], [(q, why)])`, `q` the
    name of a search that has a term and takes part in no comparison (S-A25)
    and `why` one line; standard library only, reads no file (the
    ensembles' names follow from the configuration: the arm of a
    nesting campaign's A1 reference is A1, a term search's term arm is
    A(q)); raises `RefuseError` when a search of a term-only campaign has no
    A1 reference (no search of campaign S-A1, or two or more), naming the
    search and the S-A1 searches found;
  - `ev_plan.report_inputs(cfg, ecfg)` -> block 19's dict plus the key
    `"comparisons"`: `[[name, E, E_ref], ...]` in `comparisons`' order, so
    that a report made before this block is stale (block 22's rule);
  - `ev_metrics.paired_all(cfg, ecfg, reported, mdir, log)` -> `(files,
    made, left_out)`: `files` the paired files written (absolute paths,
    `.json` and `.npz`), `made` and `left_out` as METRICS.json's
    "comparisons" and "comparisons_left_out" below;
  - `ev_metrics.compute(cfg_path, ecfg_path, names=None, ncpus=None,
    log=print)` and the CLI: block 20's, unchanged in signature.
- Rule:
  - `comparisons`: sec. 2.5's "The comparisons": (a) for each search q, in
    the configuration's order, of campaign S-A2, S-A5, S-A2-only or
    S-A5-only (A(q) = A2 for S-A2 and S-A2-only, A5 for S-A5 and
    S-A5-only): `Comparison(name=q, kind="vs_A1", E="<q>:<A(q)>",
    E_ref=...)`, `E_ref` `"<q>:A1"` for S-A2 and S-A5 and `"<s>:A1"` for
    S-A2-only and S-A5-only, s the one search of campaign S-A1; then (b) for
    each search q of term arm A2 and each search q' of term arm A5 with
    `q.real == q'.real`, in the order of q and then of q':
    `Comparison(name="<q>_vs_<q'>", kind="A2_vs_A5", E="<q>:A2",
    E_ref="<q'>:A5")`. A search of campaign S-A1 yields no comparison of its
    own; one of S-A25 is returned among the uncompared with the why
    "campaign S-A25: its arms are not defined against A1 (D-136)".
  - `paired_all`, for each comparison in `comparisons` order:
    - made when E and E_ref are both in `reported` (block 20's ensembles
      with a current sample, among `names` when `--ensemble` restricts),
      on each data set D of E's sample provenance (`datasets_of`, its order)
      that E_ref's provenance has too; none shared: left out;
    - on each D, the two samples' `rows`, `theta_ref` and `donor` are equal,
      or `RefuseError("<E> and <E_ref> do not share the rows of <D>")`;
      ok = E's `in_box` and both `logq_ref` finite; d_i = l_i(E) -
      l_i(E_ref) on the ok rows; sigma_seed = max(`member_sd` of E on D,
      of E_ref on D) [2026-11-16, the final check's S2: `member_sd` is the
      sample SD (ddof 1) over the members of their mean NLL on that
      ensemble's own in-box rows with a finite density -- on synth's data
      sets the comparison's ok rows]; `paired_block` with the seed `ev_plan.seed_of(ecfg.seed,
      "paired", name, D)` (for a comparison named q: block 20's seed, so a
      project of v3's design keeps its numbers);
    - written to `<mdir>/paired_<name>_<D>.json` and `.npz` (the contract
      below) and logged by block 20's line, `  paired <name> on <D>: Dbar
      <+.4f> [<+.4f>, <+.4f>], 2 sigma_seed <.4f> -> resolved|not resolved`;
    - not made: one entry of `left_out` and the log line `  comparison
      <name> left out: <why>`, why one of `"<ensemble> is not reported
      (<reason>)"` (the reason METRICS.json's "left_out" gives for it, or
      `"not complete"` when it is not among the ensembles found),
      `"<ensemble> was not requested (--ensemble)"`, `"no data set is
      sampled for both"`. [contract amendment 1, 2026-11-16, wave W2: the
      two first overlapped for a complete ensemble that `--ensemble` did
      not name and that has no current sample -- block 20's `compute` skips
      such an ensemble before reading its sample, so "left_out" gives it no
      reason, and `paired_all` does not receive the names. The why is now,
      for E first and then E_ref: `"<ensemble> is not reported (not
      complete)"` when it is not among the ensembles found; `"<ensemble> is
      not reported (<reason>)"`, block 20's reason ("not sampled (<path>
      absent)" or "sampled with other values: ..."), when its sample is
      absent or not current, whether or not `--ensemble` named it; `"<ensemble>
      was not requested (--ensemble)"` only for an ensemble with a current
      sample that `--ensemble` left out. Blocks affected: 27 (block 28 shows
      the why as text).]
  - The paired files (block 28 reads them; block 20's E20c reads the
    legacy keys):
    - JSON: block 20's keys `schema` (unchanged, "eval_metrics/1"),
      `dataset`, `n_rows`, `n_donors`, `Dbar`, `ci`, `se`, `level`,
      `n_boot`, `scheme`, `sigma_seed`, `two_sigma_seed`,
      `beyond_2_sigma_seed`, `ci_excludes_0`, `resolved`; for every
      comparison also `comparison`, `kind`, `E`, `E_ref`, `search`,
      `search_ref`, `arm`, `arm_ref`, `sd_members_E`, `sd_members_ref`
      (the two members' SDs entering sigma_seed), `lower_nll_ensemble` (the
      name of the ensemble with the lower donor-mean NLL when `resolved`,
      else null) and `bootstrap_seed`; for kind "vs_A1" also block 20's
      legacy keys with their meaning unchanged: `term_arm` (= `arm`),
      `ensembles` (`{"term": E, "A1": E_ref}`), `sd_members` (`{arm:
      sd_members_E, "A1": sd_members_ref}`), `lower_nll` (`"term arm"`,
      `"A1"` or null); kind "A2_vs_A5" has none of the four;
    - `.npz`: `donor_ids` (n_donors,) int64, `delta_d` (n_donors,) float64
      (the donor's mean of d_i over its ok rows), `se_d` (n_donors,) float64,
      as block 20's.
  - `compute`: METRICS.json gains `"comparisons"`, the comparisons made, in
    `comparisons` order, each `{"name", "kind", "E", "E_ref", "search",
    "search_ref", "arm", "arm_ref", "datasets": [D, ...] (the paired data
    sets, in order), "paired": {D: <absolute path of paired_<name>_<D>.json>}}`;
    `"comparisons_left_out"`, `[{"comparison", "E", "E_ref", "why"}]`; and
    `"uncompared"`, `[{"search", "why"}]`. Before the pass line, the log
    line `comparisons: <k> made, <m> left out, <u> search(es) in none`; the
    pass line keeps block 20's format and stays the last line (E20c reads
    it): its "paired comparison(s)" counts the (comparison, data set) pairs
    written, as before.
  - `ev_submit`: when `report` is in `--what`, the plan calls
    `ev_plan.comparisons(cfg, with_uncompared=True)`; the run logs, after
    the report's unit, one line per comparison, `EVAL SUBMIT: comparison
    <name>: <E> against <E_ref> (<kind>)`, and one per uncompared search,
    `EVAL SUBMIT: no comparison for <q>: <why>`, in a dry run as in a
    submission; its `RefuseError` is the launcher's refusal (`REFUSE: ...`,
    exit 3, nothing written or submitted). Without `report` in `--what` the
    comparisons are never consulted: the calibration bank, the samplings and
    the data overview neither list nor refuse on them (`run`'s count of the
    samples present must not call `comparisons`).
  - A project of v3's design (S-A2 or S-A5 searches with `alt_best_fit`
    true): its comparisons are block 20's pairs, under the same names and
    seeds, with the same numbers; `test_smoke_evaluation.py`'s checks pass
    unchanged. [2026-11-16, the final check's S4: this holds for its
    comparisons against A1; rule (b) adds A2_vs_A5 comparisons where an S-A2
    and an S-A5 search share a cohort, as the committed v3 had on R0 and R3,
    which block 20 never made.]
- Custom code: none (the bootstrap is `bootstrap_paired.paired_bootstrap`,
  as in block 20).
- Test oracles (`test_smoke_comparisons.py`, pure ASCII; it may import the
  helpers of `test_smoke_evaluation.py` and `test_smoke_synthetic.py`
  without changing them; checks needing the training environment skip
  without it, as block 19's):
  - E27a, `comparisons` on configurations alone (no training environment):
    the committed `synth.json` v4 gives exactly the six comparisons of sec.
    2.5, in order, with their names, kinds, E and E_ref; the evaluation
    fixture's v3-like configuration (S-A1, and S-A2-R0 of campaign S-A2)
    gives S-A2-R0 against `S-A2-R0:A1` and no A2_vs_A5; term-only searches
    without an S-A1 search, and with two, refused naming the search; an
    S-A25 search among the uncompared; an A2 and an A5 search on different
    cohorts give no A2_vs_A5;
  - E27b, a reduced v4-like project -- S-A1 on R0, S-A2-only and S-A5-only
    on R3, two members each, run to PIPELINE DONE under the fake PBS as the
    evaluation's fixture, its calibration bank and samplings made in the
    test's process -- through `ev_metrics.compute`: the paired files of the
    three comparisons (S-A2-R3 and S-A5-R3 against S-A1:A1,
    S-A2-R3_vs_S-A5-R3) on sim, R3 and R3dag, nine of each extension; each
    JSON's keys as the contract says, per kind; Dbar and sigma_seed
    recomputed in the test from the two samples (`logq_ref`,
    `logq_ref_members`, the ok rows) within 1e-12; the interval equal to
    `paired_bootstrap` called in the test with `seed_of(seed, "paired",
    name, D)`; METRICS.json's "comparisons" (order, data sets, paths),
    "comparisons_left_out" (empty) and "uncompared"; the log lines and the
    pass line last; S-A1's sample provenance removed: the two comparisons
    against A1 left out and named, the A2_vs_A5 one made; one sample's
    `rows` altered: a refusal naming both ensembles and the data set;
  - E27c: `report_inputs` carries the comparisons; a `REPORT_COMPLETE.json`
    recording inputs without them makes `ev_submit.plan`'s report unit
    stale; the launcher's dry run with `report` lists the three comparisons'
    lines; a configuration with term-only searches and no S-A1 search: the
    dry run with `--what report` exits 3 with nothing written, and with
    `--what calbank,sample` it does not refuse;
  - the mutations to try, each failing a check: a term-only search's
    reference taken from its own search; the bootstrap seed of an A2_vs_A5
    comparison taken from E's search; a comparison left out with no entry
    and no line; the rows check dropped;
  - regression: `test_smoke_evaluation.py` passes unchanged (its fixture is
    v3's design).
- Data flow: consumes block 19's samples (their provenance and `.npz`) and
  the configuration; runs inside block 20's `compute`, after the
  per-ensemble quantities; feeds block 28 (METRICS.json's "comparisons" and
  "comparisons_left_out", the paired files) and block 22's report staleness
  (`report_inputs`).
- Confirmed: yes 2026-11-16 (D-136: the comparisons, A2 against A5, the
  names left out, the build); the kind names, the S-A25 rule and the
  launcher's lines are Claude's proposal within it.
- Status: smoke-tested 2026-11-16 (wave W2): built by an implementer agent
  on Opus in its own worktree against the contract `cea2373` and its
  amendment 1 `c4257ba`; `test_smoke_comparisons.py` 3 of 4 (the
  convergence check skipped; E27b on a reduced v4-like project -- S-A1 on
  R0, S-A2-only and S-A5-only on R3), its 15 mutations each caught;
  `test_smoke_evaluation.py` 18 of 19 after the merge with block 28;
  vermin's minimum 3.3 for `ev_plan.py` and `ev_submit.py` `[SANDBOX
  2026-11-16]`; read against this entry at the gate and merged (`13d21aa`);
  the final check by an independent agent on Opus at `bd7651f`,
  `specs/test_reports/2026-11-16_1018_incremental.md`: PASS WITH ISSUES, no
  BLOCKER or MAJOR, the nine paired files of the v4-like run recomputed from
  the samples, a v3 project's numbers equal to block 20's at the base. Not
  run on davinci.
- Accepted at the gate (2026-11-16; plumbing, the orchestrator's): the
  launcher's why "its comparisons changed since <finished>" when only the
  comparisons differ; `compute` lists the comparisons first, so a
  configuration whose comparisons cannot be listed is refused before any
  per-ensemble work; with both ensembles missing the why names E;
  `lower_nll_ensemble` is E when Dbar < 0; the comparison lines follow the
  report's unit whatever its state and are not printed when its qsub fails
  (exit 4); the report's unit carries its comparisons; the run's count of
  samples is "a complete ensemble's provenance exists", without hashing;
  the refusal's text "search <q> (campaign <c>, no A1 trial) has no A1
  reference: ..."; the helpers `sample_left_out`, `_read_json`,
  `_paired_arrays` and the constants `TERM_ARM`, `NESTS_A1`, `UNCOMPARED`;
  the test's `CMP_FIXTURE` and `CMP_KEEP`.
- Open (the final check of 2026-11-16, left for the hardening wave): F4, a
  paired data set with fewer than two donors ends the metrics run with a
  traceback (about 1e-17 on the committed R3 of 27 donors); F5, an
  unreadable sample provenance does too (block 20's path, and block 27's
  under `--ensemble`); F6, a one-member ensemble's sigma_seed is 0; F7, the
  files of an earlier report stay beside a new one's in `metrics/` and
  `figures/`, unlisted; F8, the launcher exits 1 with a traceback, not 2, on
  a project file `config.py` refuses (the same at the base); F1's estimator
  for donors of unequal rows (sec. 2.5).

### Block 28: the comparisons' figures (added 2026-11-16; D-132 (5), D-136; roadmap synth, step S6; wave W2)

- Modules: `hpc/pipeline/evaluation/ev_figures.py`; the checks E21a
  (restated), E21b and E28* of `hpc/pipeline/tests/test_smoke_evaluation.py`.
  Amends block 21 (its per-search `arms_*` figures and `across/dbar`).
- Public API: block 21's, unchanged: `python3
  hpc/pipeline/evaluation/ev_figures.py --config ... --eval-config ...`,
  `ev_figures.make_all(cfg_path, ecfg_path, log=print)` -> the index
  `[(family, rel, caption)]`, the pass line `EVAL FIGURES OK: <n>
  figure(s) in <dir> (<t> s)`, exit 3 without metrics.
- Rule (from METRICS.json's "comparisons" and "comparisons_left_out", the
  paired files as block 27's contract gives them, block 20's per-ensemble
  files and `search_<q>.json`; it computes nothing, as block 21):
  - `ensembles/<stem>/`: block 21's figures, names, rules and captions,
    unchanged;
  - `comparisons/<name>/`, for each comparison in METRICS.json's
    "comparisons" (none when the key is absent), with E and E_ref the
    entries of METRICS.json's "ensembles" named by the comparison:
    `cmp_coverage` (one panel per paired data set, in the comparison's
    order: both ensembles' C(beta) with their bands, as block 21's
    `arms_coverage`); `cmp_nll_<D>` for each paired D other than sim (both
    ensembles' lbar_d per donor, each donor's pair joined, ordered by
    E_ref's, as `arms_nll`); `cmp_ppc_<D>` for each paired cohort D (not sim
    and not ending in "dag") where both ensembles' metrics have "ppc", and
    `cmp_lc2st_<D>` where both have "lc2st" (as `arms_ppc` and
    `arms_lc2st`); `cmp_forest_<D>` for each paired D, from
    `paired_<name>_<D>` (as block 21's forest plot, the x label `Delta_d =
    lbar_d(E) - lbar_d(E_ref) (nats)` with the two ensembles named, the
    verdict "resolved, <lower_nll_ensemble> lower" or "not resolved");
  - `searches/<q>/`: `search_trials`, `search_nll_vs_lpr` and
    `search_learning` for every search with its `search_<q>.json`, as block
    21's; no `arms_*` figure is drawn any more;
  - `across/dbar`, when "comparisons" is not empty: one row per comparison
    and paired data set, labelled `<name>, <D>`, in the comparisons' order
    and each comparison's data-set order; one panel per kind present, left
    "against A1" (x: Dbar = term arm - A1) and right "A2 against A5" (x:
    Dbar = A2 - A5), each row with its interval and its +-2 sigma_seed bar
    as block 21's `fig_dbar_all`, the marker in E's arm's hue; `across/lpr`
    unchanged;
  - colours: each ensemble in its arm's hue (A1 #2a78d6, A2 #eb6834, A5
    #1baf7a; a comparison's two ensembles never share an arm); legends name
    the arm and the search ("A2 (S-A2-R3)");
  - captions: one sentence each, at least 30 characters and ending with a
    full stop, naming both ensembles with their searches and the data set;
  - FIGURES.md: the families in the order calibration, accuracy, posterior,
    embeddings, comparisons, searches, across; its header line names the
    ensembles left out (block 21's) and then "Comparisons left out: <name>
    (<why>); ..." or "Comparisons left out: none", and the uncompared
    searches the same way.
- Custom code: none (matplotlib, as block 21).
- Test oracles (`test_smoke_evaluation.py`):
  - E21a restated: `_expected_figures` derives the comparisons' figures from
    METRICS.json's "comparisons" by the rules above (none when the key is
    absent, as a metrics tree of block 20 alone has it) and expects no
    `arms_*` figure; the rest as before;
  - E21b unchanged (no statistics library imported; no mean, SD, quantile
    or median called; the validated colours; a figure to PDF; no metrics
    refused);
  - E28a, on a metrics tree holding comparisons of both kinds, one of them
    against another search's A1 ensemble, built in the test by copying the
    fixture's real metrics (block 20's files of S-A1:A1, S-A2-R0:A1 and
    S-A2-R0:A2) into a directory of its own, relabelling one copy as an A5
    ensemble of another search, and writing METRICS.json's "comparisons",
    "comparisons_left_out" (one entry) and the paired files as block 27's
    contract gives them: the figure set equals the rules', each listed in
    FIGURES.md under "comparisons" with a caption naming both ensembles,
    the left-out comparison named in the header, `across/dbar` with two
    panels and one row per (comparison, data set);
  - the mutations to try, each failing a check: the reference ensemble
    taken from the term search's own entries; a comparison figure drawn for
    a data set that is not paired; the two kinds in one panel of
    `across/dbar`.
- Data flow: consumes block 27's METRICS.json and paired files and block
  20's per-ensemble files; feeds the report (block 22's `ev_report.py`) and
  FIGURES.md.
- Confirmed: yes 2026-11-16 (D-136); the families, names, panels and
  captions are Claude's proposal within it.
- Status: smoke-tested 2026-11-16 (wave W2): built by an implementer agent
  on Opus in its own worktree against the contract `cea2373`;
  `test_smoke_evaluation.py` 18 of 19 (the convergence check skipped; E21a
  restated, E21b, E28a on a metrics tree of both kinds, 108 figures), its
  three mutations each caught; the figures looked at by the implementer (a
  one-panel title cut off, fixed); read against this entry at the gate and
  merged (`ddfec22`); the final check (block 27's report): the v4-like
  report draws 113 figures, 24 of them the three comparisons', none
  `arms_*`, the two panels of `across/dbar` `[SANDBOX 2026-11-16]`. Not run
  on davinci.
- Accepted at the gate (2026-11-16; plumbing, the orchestrator's):
  refusals on a breach of block 27's contract -- a comparison naming an
  ensemble METRICS.json does not report, listing no paired data set, paired
  on a data set where E or E_ref has no metrics, or with a paired file
  absent: `RefuseError`, exit 3; titles that name two ensembles wrapped to
  the figure's width (14 characters per inch at 9 pt), `finish` giving each
  extra line 0.15 in; FIGURES.md's header "Uncompared searches: ..." and
  "(blocks 21 and 28)"; `across/dbar`'s x labels and its legend, one entry
  per arm of E with that arm's searches; a kind other than the two in a
  panel of its own; donors matched by id in `cmp_nll` and `cmp_lc2st`;
  `cmp_ppc`'s D_sim rings only where D_sim has a PPC; the forest plot in E's
  arm's hue; the mutations under `EVAL_MUTATION` (`MUTATION` is the
  synthetic test's); E21a clears the figures folder before drawing. [The
  final check's S5] A caption is FIGURES.md's line, which ends with a full
  stop; the index `make_all` returns carries it without one (block 21's
  convention).

### Block 29: several searches in one recorded project, each with its own core cap (added 2026-11-17; D-148 (1), D-140, D-141; wave W3)

A recorded data set may search and fit several arms on one set of banks
(D-140, D-148 (1): LindDUP's A1, A2 and A5), each search with its own core
cap inside `resources.max_cores` (D-141: 148 cores per arm). [the user's
call of 2026-11-17, D-148; the design (the views, the stage names, the
group cap) Claude's, approved with the plan; Confirmed: yes 2026-11-17]

- Modules: `hpc/pipeline/stages/runs.py` (`recorded_set_graph`,
  `core_group_of`, and the run stages, moved here from `stages/synthetic.py`
  or wrapped, as the implementer chooses; `RunView` and `search_views` are
  the contract's and stay as committed), `hpc/pipeline/stages/synthetic.py`
  (uses the run classes of runs.py; its graph, names and behaviour
  unchanged), `hpc/pipeline/pipeline.py` (`cores_in_flight(state,
  names=None)`; the Submitter's group cap; the per-search lines of `status`
  and of the dry run -- not the `stop-search` parser and `cmd_stop_search`,
  which are the contract's and stay), `hpc/pipeline/tools/search_report.py`
  and `search_reset.py` (`--search <name>`), `hpc/pipeline/tools/
  perrow_check.py` and `purge_outputs.py` (a clear refusal on a
  configuration with a `searches` list), `hpc/pipeline/tests/
  test_smoke_runs.py` (new). Not this block's: `config.py`,
  `stages/__init__.py`, `stages/search.py`, `stages/best_fit.py`,
  `stages/replicate.py`, `stages/search_stop.py` (the contract's and blocks
  30-31's).
- Configuration (sec. 3.1, v11; the contract): `searches` null or a list of
  {`name`, `campaign`, `max_cores`}; with a list, `search.campaign` and
  `search.arm` are refused.
- Graph (sec. 2.1, "A recorded data set with several searches"): with a
  list Q, S1-S6 as in the default graph, then for each q in Q in order
  `search_<q.name>` (deps `sim_bank`, `real_bank`) and `best_fit_<q.name>`
  (deps `search_<q.name>`); no stage named `search` or `best_fit`. A stage
  of q runs S7's or S8's code on q's view (`runs.RunView`): `paths.
  tune_root` and `paths.runs_root` joined with q.name, `search.campaign` =
  q.campaign, so q's folders are `<tune_root>/<q.name>/<q.campaign>/` and
  `<runs_root>/<q.name>/<dataset>_best/`, its log lines carry the stage's
  name (as the synthetic kind's), and `common.shard_globs(view)` gives the
  recorded banks of S5 and S6 -- the real one for every campaign but S-A1
  (the rule of block 7, unchanged). With `searches` null the graph is the
  eight stages, unchanged (regression).
- The group cap: `core_group_of(stage)` = (q.max_cores, the names of q's
  stages: `search_<q>`, `best_fit_<q>` and, on the synthetic kind,
  `best_fit_alt_<q>` when it exists) for a stage of q with an integer
  `max_cores`; None for every other stage, and for q with `max_cores` null
  (every synthetic search, whose entries have no such key). The
  sequencer's Submitter of a stage with a group (cap c, names G):
  - `fits(n)` is true iff n fits under `resources.max_cores` as now (D-099)
    AND n <= c - (the cores in flight of the jobs of the stages in G, counted
    by `cores_in_flight(state, names=G)` exactly as `cores_in_flight`
    counts the whole project's, a group of jobs `after` another counted
    once);
  - `check(n)` raises CapExceeded when n > c (such a job can never start),
    naming the search and its `max_cores`, so the stage stops at its start
    instead of waiting for ever;
  - jobs of stages outside G never count against c; G's jobs always count
    against `resources.max_cores`.
  The search's throttle (`pbs.fits(trial_cores)`) and the best fit's need
  no change: they ask `fits`.
- Status and dry run: for a configuration with a searches list, after the
  project's cores line, one line per search with a cap: `cores of search
  <q>: at most <c> (searches[].max_cores); in flight <x> in <k> job(s)`, and
  the search's stop line `search_stop.status_line(view)` when it returns
  one (block 30; the stub raises until block 30 lands: call it inside a
  try that prints nothing on NotImplementedError).
- Tools: `search_report.py --config C --search NAME` reports search NAME of
  a list configuration on its view (folders, ledger, trials), last line
  unchanged (`SEARCH REPORT OK: ...`); on a list configuration without
  `--search`, or with an unknown name, exit 2 with `REFUSED: this
  configuration has the searches <names>: give --search <name>`; on a
  configuration with `searches` null, `--search` is refused the same way.
  `search_reset.py` takes `--search` by the same rule. `perrow_check.py` and
  `purge_outputs.py` refuse a list configuration (exit 2, one `REFUSED:`
  line naming the reason: not supported for several searches); their
  behaviour on a null configuration is unchanged.
- Test oracles (`tests/test_smoke_runs.py`, the fake PBS): T29a the graph --
  a recorded configuration with the three searches A1 (S-A1), A2
  (S-A2-only), A5 (S-A5-only) gives the 12 stages real_extraction ...
  real_bank, search_A1, best_fit_A1, search_A2, best_fit_A2, search_A5,
  best_fit_A5 in that order with the deps above; `searches` null gives the
  eight stages (regression); the committed `linddup.json` gives the 12, and
  its dry run exits 0. T29b the views -- each view's tune_root, runs_root,
  campaign; `shard_globs` gives no real bank on A1's view and the real bank
  on A2's and A5's. T29c the group cap -- searches capped at 148 with 48-core
  jobs, `resources.max_cores` 444, driver 4: at every tick no search has more
  than 3 trial jobs in flight, the project never more than 440 cores of
  stage jobs, and each of the three searches holds a running trial while the
  others hold theirs (none starves); with c = 40 < 48 the stage stops at its
  start with CapExceeded naming the search; a search with max_cores null is
  limited by the project's cap only. T29d end to end on the stand-in: a
  recorded project with two searches (S-A1 and S-A2-only, small budgets)
  runs to PIPELINE DONE; each search's SEARCH_COMPLETE.json and each best
  fit's BEST_FIT_COMPLETE.json in its own folder. T29e the tools: the report
  with and without `--search`; the refusals.
- Regression: `test_smoke_pipeline.py`, `test_smoke_block1b.py`,
  `test_smoke_synthetic.py` (T15, T18's graphs), `test_smoke_report.py`,
  `test_smoke_reset.py`, `test_smoke_purge.py`, `test_smoke_perrow.py`
  pass as at the contract commit.
- Custom code: none.
- [2026-11-17, the final check's S2, a design question the orchestrator defers: `replicate.*`
  is not in the search identity K_cfg (block 7 (j)), so changing it while a search
  that trains the replicate term runs (A5) would continue that search under another
  rule. Not changed in this wave: a changed configuration already needs
  `--accept-config-change`, and the run sheet says not to change `replicate.*` while an
  A5 search runs. Adding it to K_cfg, with identities written before accepted, is
  an open idea.]
- Status: smoke-tested 2026-11-17 (wave W3, merged e4c85d9): `tests/
  test_smoke_runs.py` 5 pass, 1 skip (convergence) -- T29c's ticks show
  each search at 3 jobs of 48 at once and the project at 432 of 440 cores;
  T29d end to end on the stand-in with two searches to PIPELINE DONE; five
  mutations, each caught. Regression as at the contract commit:
  test_smoke_pipeline 6/1, block1b 6/1, synthetic 18 of 19, report 5/1,
  reset 5/1, purge 11/1, perrow 5/1, block1c 7/1, block8 1/1; the
  synthetic and Giulia dry runs byte-identical to the contract commit's;
  LindDUP's dry run exits 0 with the 12 stages and a cores line per search
  `[SANDBOX 2026-11-17]`. Accepted deviations: the run classes are built
  lazily in runs.py (`stage_classes()`; synthetic.py binds them under their
  old names) so that runs.py imports neither search.py nor best_fit.py at
  its top (block 30's search.py imports search_stop, which imports runs);
  helpers `run_core_group`, `pick_search`; a stop line per search
  `search <q>: <status_line>`; the run stages' dry run adds a line per
  capped search (`at most 3 job(s) of 48 cores at once`). After the wave
  (the orchestrator's fix, plumbing): D-099's own dry-run lines of a run
  with a cap state the smaller room too -- `stages/common.py` `cap_room`
  takes min(the project's room, the run's `max_cores` // ncpus), so
  LindDUP's trial and member lines read `at most 3` where they read 9; a
  configuration without a searches list and the synthetic runs print as
  before (Giulia's and synth's dry runs unchanged); checked in T29a. Not
  run on davinci.

### Block 30: stop a search by hand (added 2026-11-17; D-148 (2), D-141, D-150; wave W3)

The user watches the interim results and ends a search when it has done
enough; its best fit then trains the best configuration found so far
(D-141, D-150). [the user's call of 2026-11-17; the command, the file and
the rules Claude's, approved with the plan; Confirmed: yes 2026-11-17]

- Modules: `hpc/pipeline/stages/search_stop.py` (the contract's stub; the
  bodies), `hpc/pipeline/stages/search.py` (the poll honours a request),
  `hpc/pipeline/tests/test_smoke_stop.py` (new). The parser of
  `stop-search` and `cmd_stop_search` in `pipeline.py` are the contract's
  (they call `search_stop.cli`); the lines of `stages/search.py` that call
  `replicate.runner_extra` are the contract's and stay.
- Command: `python3 hpc/pipeline/pipeline.py stop-search --config <cfg>
  [--search <name>] [--kill-running] [--cancel]`; on any login node; it
  writes or removes one file and submits nothing (D-098).
- The request file: `<campaign folder>/STOP_REQUEST.json`, the campaign
  folder being `<paths.tune_root>/<search.campaign>` of the search's view
  (`runs.search_views`; the folder of its ledger and SEARCH_COMPLETE.json):
  {"schema": "search_stop/1", "requested_utc": ISO 8601 UTC, "kill_running":
  bool, "user": str, "host": str}; ASCII JSON, sorted keys, written to a
  temporary file in the same folder then `os.replace`d.
- The command's rules (exit 0 with its last line; exit 2 with `REFUSED:
  <reason>` and nothing written):
  - with a searches list, `--search` names one of its searches; with
    `searches` null it is not given (the single search);
  - the search has ended (its SEARCH_COMPLETE.json exists): refused, naming
    its `ended_by`;
  - the search has no completed trial yet (no record of status `ok` among
    the records the stage reads as its ledger, `SearchStage`'s own reading):
    refused -- "nothing to fit yet; stop the whole run with `stop` instead";
  - a request already present is replaced (the line names the previous
    one's time and kill_running);
  - `--cancel`: removes the file when present and the search has not ended
    (`STOP CANCELLED: ...`), refused otherwise; a trial already deleted by
    the request stays killed.
  - last line `STOP REQUESTED: search <name or the campaign> (<campaign>);
    kill running: yes|no; the sequencer acts at its next tick (every
    driver.poll_seconds)`.
- The stage (stages/search.py), at every poll in the phase `rounds` while a
  request stands:
  - nothing is submitted: no waiting trial, no resubmission of any kind
    (killed, node fault, hold, environment fault), no new round proposed;
  - with kill_running: the last job of each running trial is deleted once
    (`pbs.delete`; the trial records `stop_deleted`); when the job has
    ended, the trial gets the stage's killed record with the reason
    `stopped by the user (stop-search --kill-running)`, status `killed`,
    and is not resubmitted; [amendment 3, 2026-11-17, D-154: a job that
    wrote its ledger record before the deletion took effect keeps that
    record -- ok or failed -- which a killed record never overwrites;]
  - without kill_running: the running trials run to their end; a trial
    whose job ends without its record is killed with the reason `ended
    without a record after a stop request` and is not resubmitted;
  - when no trial is running or under examination: every waiting trial
    becomes `cancelled` (a terminal status, never submitted, counted in
    `n_trials`), and the search finishes as at the clock cap: `_finish`
    with ended_by `user_stop` -- finalists over the ledger, SEARCH_COMPLETE
    .json with `"ended_by": "user_stop"` and `"stop_request"` (the file's
    content); the best fit then starts as after any finished search;
  - one log line when a request is first seen: `search: stop requested
    <utc> (kill running: yes|no): no new trial; <n> running`.
  In the phase `gate` (the budget probes) a request is acted on as above
  once the phase ends; with no completed trial `_finish` fails as it does
  today (the command refuses such a request, so only a hand-made file gets
  there). Without a request the stage is unchanged (regression).
  [amendment 5, 2026-11-17, the final check's F1 (MAJOR), S4, S5: the
  request file outlives its search -- it stays beside SEARCH_COMPLETE.json
  as the record of the stop (`--cancel` is refused once the search has
  ended) -- so `tools/search_reset.py` sets it aside with the rounds: it is
  one of the listing's items (with its size, in the CODE's digest) and is
  moved into `<campaign>_set_aside_<utc>/`; a reset search therefore starts
  with no request and proposes round 0 as before. The stage itself is
  unchanged.]
- `status_line(view)`: None without a request; else `stop requested <utc>
  (kill running: yes|no)`, with `; the search ended (user_stop)` once its
  SEARCH_COMPLETE.json says so. [2026-11-17, the final check's S3 and S6,
  the code's rules recorded and accepted: on a synthetic configuration
  `--search` takes the names of `synthetic.searches`; with `searches` null
  the `status` prints no stop line (the search report prints the request).]
- [amendment 3, 2026-11-17, D-154; a follow-up of the wave, block "30b"]
  `tools/search_report.py`: a trial `cancelled` reads `next: nothing
  (cancelled by the stop request; never submitted)`, and while the search's
  STOP_REQUEST.json stands no trial's `next:` predicts a submission or a
  resubmission (it reads `next: nothing (a stop request stands: ...)` with
  what the stage does instead -- the job deleted under kill_running, the
  trial killed when its job ends without a record); the report prints the
  request's `status_line` after its header. The dry-run lines of the search
  and of the best fit that describe a trial's, a probe's and a member's
  argv show the `--runner-extra` text with the replicate flags (block 31)
  that the submitted argv carries.
- Test oracles (`tests/test_smoke_stop.py`, the fake PBS and the stand-in
  runner): T30a the command -- each refusal (ended search, no completed
  trial, `--search` missing or unknown on a list configuration, given on a
  null one), the request's file and content, the replacement, `--cancel`;
  nothing written on a refusal. T30b a search with trials ok, running and
  waiting: after a request without kill, nothing new is submitted, the
  running trials finish, the waiting ones become `cancelled`, SEARCH_COMPLETE
  has `ended_by` `user_stop` and its finalist is the best ok trial, and the
  best fit starts. T30c with `--kill-running`: the running trials' jobs are
  deleted (the fake qdel records it), their records are killed with the
  reason, none is resubmitted though `resubmit_killed` is true. T30d a
  trial failing of a node fault after a request is not resubmitted. T30e a
  configuration with a searches list: the request stops that search only;
  the other searches go on.
- Regression: `test_smoke_block7.py`, `test_smoke_identity.py`,
  `test_smoke_finalist.py`, `test_smoke_report.py` pass as at the contract
  commit.
- Custom code: none.
- Status: [Reopened 2026-11-17 by contract amendment 5 (F1: the reset tool sets the request aside);
  smoke-tested again 2026-11-17 (block 30c, merged 79f882d): `test_smoke_reset.py`
  6 pass, 1 skip, its new `test_stop_request` (the CODE changes when a request
  appears; a search ended user_stop reset with --confirm: the request listed, moved
  into the set-aside folder with its sha256 and named in RESET_RECORD.json, gone
  from the campaign; the stage then passes the gate on the kept probe file and
  proposes round 0); the mutation `stop_left` (F1's symptom) caught; stop 6/2,
  report 6/1, T29e as before `[SANDBOX 2026-11-17]`.] Before: smoke-tested 2026-11-17 (wave W3, merged b1e53d1): `tests/
  test_smoke_stop.py` 5 pass, 2 skip (convergence, determinism); T30b-T30d
  end to end on the fake PBS (`SEARCH COMPLETE (user_stop)`, the deleted
  jobs' Exit_status 271, no resubmission though `resubmit_killed` is true);
  six mutations, each caught; regression as at the contract commit: report
  5/1, identity 4 pass 2 skip 1 TODO (block 29's, then), finalist 6/2,
  block7 7 of 8 `[SANDBOX 2026-11-17]`. Accepted deviations: the stage
  deletes through the recorder's `delete` when it has one, else the
  adapter's; a node fault under a request heals the record but never
  resubmits (the trial ends `cancelled`); a hold or an environment fault
  under a request is `ended without a record after a stop request`; a
  record written before the deletion is kept (D-154, amendment 3);
  `cancelled` is terminal and gets no ledger record; `--cancel` with
  `--kill-running` is refused; an unreadable request is a request without
  kill_running. Follow-up 30b (amendment 3): the report and the dry run.
  Not run on davinci (PBS Pro's qdel of a running trial).

### Block 31: the replicate pair rule (added 2026-11-17; D-148 (3), D-146, D-147; wave W3)

Arm A5's replicate pairs become two different real windows drawn at random
within a group named by one of the real bank's id arrays, pairs within one
well allowed (D-146, D-147); the pipeline passes the rule of the
configuration (`replicate.*`) to every runner, and the new rule is the
standard of a new project (D-148). [the user's calls of 2026-11-17, D-146,
D-147, D-148; the sampler's construction Claude's; Confirmed: yes
2026-11-17]

- Modules: `hpc/joint/stage2/joint_batches.py` (the within-group sampler;
  `ThreeStreamBatcher(..., rep_pairs="cross_well", rep_group=None,
  rep_group_key="donor")` [amendment 2, 2026-11-17: the KEY's name, which
  the report lines print]; the report lines), `hpc/joint/stage3/run_joint_arms.py` (the flags, the A5
  refusal, the replicate diagnostic), `hpc/joint/stage3/
  smoke_test_rep_pairs.py` (new), `hpc/pipeline/stages/replicate.py`
  (`runner_extra`'s body). The shard contract's optional `rep_group` array
  (`hpc/joint/stage1/latent_bank.py`, `OPTIONAL_ROW_ARRAYS`) is the
  contract's and stays as committed.
- Flags (run_joint_arms.py; the tuner passes them through `--runner-extra`):
  `--rep-pairs {cross_well,within_group}`, default `cross_well`;
  `--rep-group-key {donor,rep_group,batch,well}`, default `donor`. Without
  the flags the runner is exactly the parent commit's (regression: block
  14's rule). [amendment 5, 2026-11-17, the final check's F3: the two
  choices are literal tuples inside `build_parser`, which the stage-4
  suites (`smoke_test_joint_tune.py` J28, `smoke_test_joint_space.py` J35)
  lift out of the runner by AST and run without its module; the runner
  asserts at import that they equal `joint_batches.REP_PAIRS` and its own
  `REP_GROUP_KEYS`, so the lists stay one.]
- Mathematics. Let R be the real rows that are not surrogates, g_i the
  value of the KEY array at row i, and for each group h, n_h = |{i in R :
  g_i = h}|.
  - `within_group`: the pair set is P = {(i, j) : i, j in R, i < j, g_i =
    g_j}, |P| = sum over h of n_h (n_h - 1) / 2; two rows of one well are in
    P. Each pair the batcher draws is uniform on P: a group h with
    probability w_h = [n_h (n_h - 1) / 2] / |P|, then two distinct rows of
    h, uniform among its n_h (n_h - 1) / 2 unordered pairs. The b_rep pairs
    of one step are independent draws, so one step may repeat a pair (with
    |P| >> b_rep, rarely). P is never enumerated (its size grows as the
    square of the groups' sizes); the draws use the batcher's replicate
    generator (`g_rep`), so the checkpoint's state resumes the same
    sequence.
  - `cross_well`: block 14's rule (F-ab) with the KEY array in place of the
    donor: P = {(i, j) : g_i = g_j, well_i != well_j}, enumerated, drawn as
    now (b_rep distinct pairs per step). With KEY = donor it is block 14's
    exactly.
  - Under either rule |P| = 0 with lambda_rep > 0 stops the run (F-aa), the
    message naming the rule and KEY.
- The real bank must hold the KEY array when the real arm is loaded;
  otherwise the run stops: `--rep-group-key <KEY>: the real bank has no
  '<KEY>' array (a bank built before Sbi-extractor block 9 has no
  rep_group)`. Without a real bank (S-A1's trials) the flags are accepted
  and unused.
- Report lines (the batcher's report): `replicate pairs  : <|P|>  (<rule>)`,
  where <rule> is `two rows of one <KEY> group, any well` (within_group) or
  `two wells of one <KEY> group` (cross_well); under within_group also
  `replicate groups : <number of h with n_h >= 2> of <number of groups>`.
  [amendment 2, 2026-11-17, D-152: under cross_well with KEY donor the
  texts are block 14's unchanged (`two wells of one donor`, the singleton
  donors line), since the default's output must stay the parent commit's
  and block 14's runner check reads it; `two wells of one <KEY> group`
  holds for any other KEY.] [2026-11-17, the final check's S8, the code's
  text recorded: under cross_well with another KEY the singleton line reads
  `singleton groups : <n>   <- D12: pairs are impossible for these (<KEY>
  groups)`.]
- The replicate diagnostic (run_joint_arms, after training): under
  `within_group` it scores at most 64 pairs of P drawn with a generator
  seeded by `--seed`, spread over the groups (round robin over the groups
  in increasing id, one fresh pair per group per turn, a group skipped once
  it has no unscored pair), its record's `pair_kind` the rule's text, plus
  `n_groups_scored`; under `cross_well`, block 14's diagnostic unchanged
  (its fallback included). [amendment 2, 2026-11-17, D-152: under
  cross_well the diagnostic groups by the KEY array as the term does --
  with KEY donor exactly block 14's diagnostic -- so the pairs scored and
  the pairs trained follow one rule.]
- `replicate.runner_extra(cfg, base="")`: `base` followed by `--rep-pairs
  <replicate.pairs> --rep-group-key <replicate.group_key>`, one space
  between items, no leading or trailing space; the stages already pass it
  to every trial, budget probe and best-fit member (the contract).
- Test oracles (`hpc/joint/stage3/smoke_test_rep_pairs.py`): T31a counts by
  hand -- 12 rows: rows 0-5 in group 0 with rows 4 and 5 surrogates, rows
  6-8 in group 1, row 9 in group 2, rows 10-11 in group 3: n = (4, 3, 1, 2),
  |P| = 6 + 3 + 0 + 1 = 10, w = (0.6, 0.3, 0, 0.1), `replicate groups : 3 of
  4`; in 100,000 draws no pair crosses a group, contains a surrogate or
  repeats a row, and each group's frequency is within 0.01 of its w. T31b the
  law -- the 30 wells of LindDUP's shape (groups of 5, 12, 12 and 1 wells,
  54 windows per well: 9 subregions x 6 windows), 200,000 draws: each
  group's frequency within 0.005 of w_h, the share of pairs within one well
  within 0.005 of its exact value per group ((54 - 1) / (54 w - 1)), as
  computed by `pair_mix.py` of 2026-11-17 (0.197, 0.082, 0.082, 1.000).
  [amendment 2, 2026-11-17, D-152: the group frequencies on 200,000 draws
  within 0.005, the within-well shares on 2,000,000 draws within 0.005. At
  200,000 draws the 5-well group gets about 15,900 of them, so its share's
  binomial SD is sqrt(0.197 x 0.803 / 15,900) = 0.0032 and 0.005 is 1.6 SD
  (12.7 % of seeds fail, measured by the implementer); at 2,000,000 it is
  5.0 SD.]
  T31c regression -- `cross_well` with KEY donor equals block 14's pairs on
  B14's example; the stage 2 suites pass. T31d determinism -- the same
  seed gives the same pairs; a batcher saved with `state_dict` and restored
  draws the same continuation. T31e the runner -- A5 to completion at 2
  epochs of 3 steps on a small cohort bank whose shard carries `rep_group`,
  under `--rep-pairs within_group --rep-group-key rep_group`, its report
  line counting |P|; the refusal on a bank without `rep_group`; A1 and A5
  without the flags unchanged (B14's runner tests pass). T31f
  `runner_extra` on the committed configurations: linddup `--rep-pairs
  within_group --rep-group-key rep_group`, giulia `... within_group ...
  donor`, synth `... cross_well ... donor`, with and without `base`.
- Custom code: the within-group sampler (no library draws a uniform pair
  from a union of groups without enumerating it; checked:
  `numpy.random.Generator.choice` over groups then over rows, which is what
  it uses).
- Status: [Reopened 2026-11-17 by contract amendment 5 (F3: the parser's choices);
  smoke-tested again 2026-11-17 (merged 6382a02): the stage-4 suites
  `smoke_test_joint_tune.py` and `smoke_test_joint_space.py` 9 of 9 again (J28,
  J35); `smoke_test_rep_pairs.py` 12/1 with T31e's bit-for-bit lines unchanged;
  `smoke_test_joint_arms.py` 33; the parser unchanged (53 actions equal on every
  attribute, the same help and usage, 12 command lines alike); five mutants of the
  lists, each stopping the import `[SANDBOX 2026-11-17]`. Accepted deviations: the
  check raises AssertionError explicitly (not stripped by -O) and reads the
  parser's actions.] Before: smoke-tested 2026-11-17 (wave W3, merged d44a76d):
  `hpc/joint/stage3/smoke_test_rep_pairs.py` 12 pass, 1 skip
  (convergence); T31b at 2,000,000 draws: within-well shares 0.1969,
  0.0818, 0.0824, 1.0000 against 0.1970, 0.0819, 0.0819, 1 (amendment 2);
  against 88c32e5's runner with KEY donor, L, history, endpoint and the
  replicate block bit for bit equal (A1 6.79678972562, A5 6.79841470718),
  block 14's fallback included; the four stage suites as at the contract
  commit (joint 40/1, blocks_11_14 9/1, joint_arms 33, blocks_4_6 10/1);
  mutations of the sampler and of the cross_well diagnostic, each caught
  `[SANDBOX 2026-11-17]`. Accepted deviations: each step seeds a numpy
  Generator from 62 bits of `g_rep` (resume exact within one environment;
  numpy does not promise its streams across versions); under within_group
  `batcher.pairs` is None and `n_pairs` gives |P|; the KEY refusal fires
  whenever a real bank is loaded; `runner_extra` refuses a value that is
  not one word. Not run on davinci.

### Block 32: the inputs check on configuration v11 (added 2026-11-17; D-148, D-143, D-145, D-142; wave W3)

The inputs check (block 9) is the user's first command on davinci for a new
campaign; on v11 it also checks the cohort map against the manifest, the
topology folders a kill left without metadata, and the partial topologies.
[the user's calls D-143, D-145, D-148; the rows Claude's, approved with the
plan; Confirmed: yes 2026-11-17]

- Modules: `hpc/pipeline/tools/inputs_check.py`, `hpc/pipeline/tests/
  test_smoke_inputs_check.py` (its fixture moved from the MEAMA-like tree
  to an lcdup15-like one: the committed `linddup.json` v11 describes the
  campaign `lcdup15_rho1300_neur_n01`). Not this block's: `config.py`,
  `linddup.json` and the cohort map (the contract's).
- The fixture (the test's tree): a campaign `campaign_lcdup15_rho1300_neur_n01`
  with tasks `sweep_cfd_task0000` and `sweep_cpu_task0001` (or the like), a
  37-axis registry of which 13 are active, ln coordinates but one, the
  on-axis `gbarA` with box [0.01, 10] in every task, `conn_rule` flat,
  `conn_prob` the only topology axis, each topology holding K iteration
  files with their `iter_<i>.json` (K = `_resolved_n_workers` x
  `n_params_per_worker` of the task's job_args.json), `topology_meta.json`
  in each topology folder; a cohort manifest whose wells have the
  `out_dir` of the committed cohort map (or a small map of its own,
  written by the test); the configuration derived from the committed
  `linddup.json` as now (paths into the tree).
- Rows added (each one line, as the others):
  - `cohort map`: `real.cohort_map` null -- NOTE `no cohort map: donor =
    well`. Else the file read by Sbi-extractor's `export_windows/
    cohort_map.read_cohort_map(path, manifest)` imported from
    `paths.sbix_dir` (Block 9; the rules in one place), against the
    manifest the check reads: OK with `<n> wells, <k> excluded; <D>
    donors; <G> pairing groups; included per class <class>:<count>, ...`;
    FAIL with the error's text (a well without a row, a row of no well, a
    bad value); FAIL when the module is absent or raises
    NotImplementedError (`the Sbi-extractor checkout has no Block 9: pull
    it`); NOTE when the manifest is absent (nothing to check against).
  - `topology metadata`: the raw `topo_*` folders without
    `topology_meta.json`: FAIL with their count and the first ten paths
    relative to `sim.sim_main` -- "move them out of their sweep folder
    before S2: the virtual MEA stops on the whole task (ANN reply
    2026-11-17, sec. 2.2)"; OK when none.
  - `partial topologies`: with `bank.drop_partial_topologies` true, by
    Sbi-extractor Block 10's rule (K(t) from job_args; f(t, k) = the
    topology's `iter_*.json` files + the non-blank lines of its
    `_failures.jsonl`): NOTE `<n> partial topologies in <m> task(s),
    <s> simulations: the bank plan drops them whole` (<s> counts the
    partial topologies' `iter_*.npz` files: the final check's S7); FAIL when a task's
    job_args lacks a positive integer `_resolved_n_workers` or
    `n_params_per_worker`, or when f(t, k) > K(t); with false: NOTE `not
    checked (drop_partial_topologies false)`.
  - `searches`: with a `searches` list, one NOTE naming each search, its
    campaign and its cap, and the cores the caps add up to against
    `resources.max_cores`.
  The existing rows are unchanged; their reading of `stoa_gate_p` is the
  configuration's (`bank.topology_axes`), not a name in the code.
  [2026-11-17, the final check's F6, a NOTE accepted: the rows match the
  folders by `^topo_(\d+)$` and rebuild their paths as `topo_%05d`, so a
  folder named otherwise (`topo_1`) is misread; the simulator writes only
  `topo_%05d`. Importing Sbi-extractor Block 10's `dispatch_size` and
  `raw_finished_counts`, so that the rule lives in one place, is an open
  improvement.]
- Test oracles: T32a the consistent lcdup15-like tree: 0 FAIL, 0 WARN;
  the counts; `topology axis conn_prob` OK; the `cohort map` row OK with its
  counts (a fake `cohort_map` module in the test's Sbi-extractor tree
  implementing Block 9's contract is allowed while the real one raises
  NotImplementedError; with SBIX_CLONE at or after Block 9 the real one);
  `partial topologies` NOTE with 0. T32b mutations, each making its named
  row FAIL and no other: a manifest well missing from the map; a map row of
  no well; a topology folder without `topology_meta.json`; a job_args
  without `n_params_per_worker`; f(t, k) > K(t). T32c a partial topology (2
  of K files): NOTE counting it, no FAIL. The suite's existing checks
  (being written, reference, invariants, contract, determinism, edge
  cases) moved to the new tree and passing.
- Custom code: none.
- Status: smoke-tested 2026-11-17 (wave W3, merged bdbd8e2): `tests/
  test_smoke_inputs_check.py` 8 pass, 1 skip (convergence) on the
  lcdup15-like tree, `INPUTS CHECK: 0 FAIL, 0 WARN` with the cohort map's
  row `35 wells, 5 excluded; 2 donors; 4 pairing groups; included per class
  control:17, pathological:13`; 16 injected bugs, each caught
  `[SANDBOX 2026-11-17]`. Accepted deviations: the map file absent is a
  FAIL; a task without job_args.json is skipped by the partial count;
  `<s> simulations` counts the partial topologies' iter_*.npz; the module
  is loaded from its file with no bytecode written into the checkouts.
  Not run on davinci.

### Block 33: the view of a recorded project, its data and the evaluation's configuration (added 2026-11-23; D-159, D-162, D-163; roadmap tool, step S2, wave 1)

- Modules: `hpc/pipeline/evaluation/ev_rec_config.py` (new; standard
  library only), `ev_rec_plan.py` (new; the standard library and the
  pipeline's own `config`, `stages.runs`, `ev_plan`: it runs on the login
  node's python, as `ev_plan` does), `ev_rec_data.py` (new; numpy and the
  runner's bank readers, so `envs.train`),
  `hpc/pipeline/tests/rec_fixture.py` (new; the fixture of every
  `test_smoke_rec_*.py`), `hpc/pipeline/tests/test_smoke_rec_plan.py` (new;
  checks R33*).
- Public API:
  - `ev_rec_config.load(path) -> RecEvalConfig` (a namedtuple of KEYS and
    `path`, with `.sub(name)`, `.sample_params()`, `.report_params()`);
    `ev_rec_config.validate(data) -> [str]`, every problem found;
    `RecEvalConfigError`; `KEYS`, `GATE_KEYS`, `SAMPLE_PARAMS`,
    `REPORT_PARAMS`, `JOB_KINDS = ("sample", "report", "validate")`.
  - `ev_rec_plan.load_view(cfg_path, ecfg) -> View`; `View` is a namedtuple
    `(cfg, kind, ensembles, missing, sim_glob, banks)`: `kind` "recorded"
    or "as_recorded"; `ensembles` a tuple of `RecEnsemble(name, stem,
    search, campaign, arm, folder, seeds, records, banks)` in the
    configuration's order (`search` the search's name, or None for a
    recorded configuration whose `searches` is null; `banks` the names of
    its recording banks); `missing` a tuple of `(search, folder, why)` for
    each configured best fit that is not complete; `sim_glob` the simulated
    bank's shards; `banks` a dict from a recording bank's name to its
    shards' glob.
    `ev_rec_plan.find(view, name) -> RecEnsemble`;
    `ev_rec_plan.comparisons(view) -> ([Comparison], [(name, why)])`,
    `Comparison` being `ev_plan`'s namedtuple;
    `ev_rec_plan.sample_json(ecfg, stem)`;
    `ev_rec_plan.report_inputs(view, ecfg)`. `RefuseError`, `seed_of` and
    `sha256_file` are `ev_plan`'s, imported, never defined again.
  - `ev_rec_data.sim_index(view) -> (ShardIndex, paths)`;
    `ev_rec_data.report_rows` and `calibration_rows`: `ev_data`'s, imported;
    `ev_rec_data.draw_rows(donor, k) -> (n,) bool`;
    `ev_rec_data.recording(view, name) -> dict` with `x` (n, W) float32,
    `cls`, `donor`, `well`, `batch`, `subregion`, `window_idx`, `rep_group`
    (n,) int64, `contaminated` (n,) bool, `sidecar`, `paths`;
    `ev_rec_data.axes_of(sidecar, p) -> (names, bounds)`: `names` a list of
    p strings, `bounds` a (p, 2) float64 array or None.
  - `rec_fixture.fixture() -> dict` (built once per process; `REC_FIXTURE`
    reuses a tree built before, `REC_KEEP=1` keeps the one it builds).
- Rule (sec. 2.6, "The view of a project" and "The data sets"):
  - a recorded configuration with `as_recorded` true, or a synthetic one
    with `as_recorded` false, is a refusal that names the other evaluation
    (`submit_eval.sh` for a synthetic project read as itself);
  - an ensemble is complete when its folder has `BEST_FIT_COMPLETE.json`,
    whose `member_seeds` give the members; a member without its record is a
    refusal, as are members whose `arm`, `split_seed`, `split_fracs` or
    `split_hash` differ; two complete ensembles of one name are a refusal;
  - the comparisons (sec. 2.6): with `TERM_ARM` and `NESTS_A1` of `ev_plan`;
    for an as-recorded view exactly `ev_plan.comparisons`; for a recorded
    view with a `searches` list, (a) then (b) in the configuration's order,
    a search whose ensemble is not of its term arm, or whose reference is
    missing, named with the reason; no search of campaign S-A1, or two, when
    a reference is needed, is a refusal (block 27's rule); a recorded
    configuration with `searches` null has no comparison;
  - `draw_rows(donor, k)`: True on the first k rows of each donor in the
    given order; every row when k is None; `draw_rows(donor, 1)` equals
    `calibration_rows(donor)`;
  - `recording`: the bank's shards sorted, read by `R.concat_shards`; a
    bank whose sidecar `arm` is not "R" is a refusal; `rep_group` is -1 on
    every row of a bank without that array; the returned dict has no
    `theta` key (an extractor's real bank has zero columns of it; a
    synthetic cohort's theta* is withheld from the evaluation, D-162);
  - `axes_of`: the sidecar's `param_names` and `bounds_theta` when both
    have p entries; otherwise the names `theta_0`, ..., `theta_{p-1}` and
    bounds None.
- The configuration `<tag>_rec_eval.json` (`ev_rec_config`; keys starting
  with `_` are comments, as in the project's configuration). KEYS, with
  the value D-163 fixed or Claude's proposal (sec. 8, Q-R2, Q-R5):
  `out_root` (absolute); `seed` (int >= 0); `n_post` (int > 0; 1000);
  `batch_rows` (int > 0); `oracle_tol` (nats, in (0, 0.1]; 2e-3);
  `as_recorded` (bool); `sim_draw_rows_per_donor` (null or an int >= 1; 1);
  `edge_eps` (in (0, 0.5); 0.006693); `gate` (exactly GATE_KEYS:
  `donor_fracs`, four positive numbers summing to 1 within 1e-9, [0.25,
  0.25, 0.4, 0.1] [2026-11-27, D-172 [user]: the null set gets 0.4 of the
  networks and the reference set 0.25; D-164's value was [0.4, 0.25, 0.25,
  0.1], which the tests' fixture keeps]; `n_ref` int >= 2 and <= 1000, 800 [the upper bound
  added 2026-11-23 after the final check, F4, D-167: above 1,000
  reference rows `npe_misspec.bandwidth_grid` takes med_m on a subsample,
  which is not (R5)]; `n_rank` int >= 1, 1000;
  `n_null` int >= 1, 1000; `n_null_min` int >= 1, 50; `rows_per_donor` int
  >= 1, 1; `factors`, a non-empty list of positive finite numbers
  [finite: 2026-11-23, F2], [0.25, 0.5, 1,
  2, 4]; `alpha` in (0, 1), 0.05); `bootstrap_n` (int > 0); `level` (in (0,
  1); 0.95); `dpi`; `formats` (among png, pdf, svg); `ncpus`;
  `validate_against` (null, or the absolute `out_root` of the synthetic
  evaluation the validation compares with; not null only with
  `as_recorded`); `jobs` (`queue`, and `ncpus`, `mem`, `walltime` for each
  of JOB_KINDS). SAMPLE_PARAMS = `seed`, `n_post`, `as_recorded`,
  `sim_draw_rows_per_donor` and the gate's `donor_fracs`, `n_ref`,
  `n_rank`, `n_null`, `n_null_min`, `rows_per_donor` (what a sampling
  depends on: a sample of other values is stale). REPORT_PARAMS = `seed`,
  `edge_eps`, the gate's `factors` and `alpha`, `bootstrap_n`, `level`,
  `dpi`, `formats`. `oracle_tol` is in neither: it changes no output.
- The fixture (`rec_fixture.fixture()`), built once, a few minutes in the
  sandbox: (1) the reduced synthetic project of block 19's tests
  (`test_smoke_evaluation._build_project`'s searches S-A1 and S-A2-R0 with
  `alt_best_fit`, cohorts R0 and R3, two members) with `synthetic.sim.
  n_theta` 160 instead of 60, so that the report split has 8 networks and
  the gate's four groups are not empty; (2) the synthetic evaluation's
  samples and metrics of it (blocks 19-20, with their calibration bank);
  (3) `rec_eval_as.json`, the as-recorded configuration with the synthetic
  evaluation's `seed`, `n_post`, `batch_rows`, `sim_draw_rows_per_donor`
  null and `gate.n_null_min` 1; [amended 2026-11-23, contract amendment 2,
  after D-165: both of the fixture's configurations have `gate.n_null_min`
  1 and `gate.alpha` 0.5 -- with 8 report networks the null set has 2 rows,
  and k_gate (R9) is 0 at alpha 0.05, a refusal, and 1 at 0.5; a
  configuration file of a reused tree whose content is not the one the
  fixture writes is rewritten] (4) a recorded-kind project on the same
  runs: a configuration made from the committed `linddup.json` with its
  `dataset`, `paths` and `searches` replaced (the synthetic project's
  dataset, roots and its two searches as recorded searches), and its
  recording bank `<dataset>_real`, written from cohort R0's shards with
  `theta` of zero columns and a sidecar of arm R; with `rec_eval.json`
  (`as_recorded` false, `sim_draw_rows_per_donor` 1). Keys: `root`, `cfg`
  (the synthetic configuration), `ecfg_old`, `ecfg_as`, `cfg_rec`,
  `ecfg_rec`, `old_out` (the synthetic evaluation's `out_root`).
- Custom code: none.
- Test oracles (`test_smoke_rec_plan.py`):
  - R33a the configuration: the fixture's two files load; each missing and
    each unknown key is named; one check per value rule above (a fraction
    list summing to 0.9, `edge_eps` 0.5, `alpha` 1, `n_post` 0, an empty
    `factors`, `validate_against` without `as_recorded`, a `jobs` kind
    missing); `sample_params` and `report_params` hold exactly the keys
    listed;
    [2026-11-23, after the final check: also `gate.n_ref` 1001, a
    non-finite factor, a `mem` and a `walltime` with a trailing newline,
    `n_post` 1000.0, each refused; `n_ref` 1000 accepted];
  - R33b the as-recorded view equals `ev_plan.ensembles` on the fixture:
    names, stems, folders, seeds, records and cohorts, in order, and
    `comparisons` equals `ev_plan.comparisons`; the two kind mismatches
    refused;
  - R33c the recorded view on the fixture: names `S-A1:A1` and
    `S-A2-R0:<arm>`, one recording bank `real`, `sim_glob` the simulated
    bank's; with `searches` set to null and one best fit, `best:<arm>`; a
    best fit without `BEST_FIT_COMPLETE.json` listed in `missing`; a member
    record of another arm, and one of another split hash, refused;
  - R33d the comparisons of a recorded view, on a tree of empty best-fit
    folders with hand-written records: searches A1 (S-A1), A2 (S-A2-only),
    A5 (S-A5-only) give `A2`, `A5` (each against `A1:A1`) and `A2_vs_A5`,
    in that order; without the S-A1 search, a refusal; a term search whose
    best fit is not complete, named and left out;
  - R33e `draw_rows` by hand on donor = (5, 5, 7, 5, 7, 9): k = 1 gives
    rows 0, 2, 5; k = 2 gives 0, 1, 2, 4, 5; None gives all; k = 1 equals
    `calibration_rows`;
  - R33f `recording` on the fixture's recorded bank: the keys above and
    their dtypes, the rows in bank order, x equal to cohort R0's, no
    `theta` key, `rep_group` -1; on the as-recorded view the same keys and
    no `theta`; a bank of arm S refused; `axes_of` on the simulated bank's
    sidecar and on a sidecar without names.
- Build notes (2026-11-23, accepted at the gate; none changes the rule):
  a number of a non-integer key (`oracle_tol`, `edge_eps`, `level`, the
  gate's `donor_fracs`, `factors` and `alpha`) is accepted as an int or a
  float, a bool never; an integer key requires an int (`n_post` 1000.0 is
  refused) [clarified 2026-11-23 after the final check, S3]; a job's `mem`
  and `walltime` are matched whole, so a trailing newline is refused
  [2026-11-23, F3]; the fixture's values the entry did not state are seed 4242,
  `n_post` 40, `batch_rows` 16, `bootstrap_n` 200, dpi 60, formats png,
  queue `test`, `oracle_tol` the synthetic evaluation's (1e-4) as recorded
  and 2e-3 for the recorded kind, the `out_root`s `<root>/rec_eval_as_out`
  and `<root>/rec_eval_out`, `validate_against` the synthetic evaluation's
  `out_root` in `rec_eval_as.json`; the recorded configuration is
  `<root>/synth_rec.json`, its searches S-A1 (S-A1) and S-A2-R0 (S-A2) with
  `max_cores` null; the recorded bank keeps cohort R0's sidecar but for
  `param_names` [], `bounds_theta` [] and arm R; a tree given by
  `REC_FIXTURE` is completed, not rebuilt, and one whose `n_theta` is not
  160 is refused; in a recorded view a search of a campaign that nests A1
  takes its own ensemble as its A1 reference only when that ensemble is of
  arm A1, and the S-A1 search's otherwise (sec. 2.6); the test's runner
  calls `rec_fixture.cleanup()`.
- Confirmed: yes 2026-11-23 (D-159, D-162, D-163; the configuration's
  values that were Claude's, D-164).
- Fix round of 2026-11-28 (the tester's F7, plumbing): `ev_rec_data`'s
  search for the first non-finite theta of a refused bank skips a shard it
  cannot read on the reader's errors only (OSError, ValueError, KeyError,
  EOFError, zipfile.BadZipFile), not on every exception; the refusal's
  message is otherwise unchanged.
- Status: smoke-tested 2026-11-23 (`test_smoke_rec_plan.py` R33a-f: 6 of 6
  pass `[SANDBOX]`, 86 s with the fixture built from scratch -- the reduced
  project to PIPELINE DONE in 76 s, 8 networks and 16 rows in its report
  split --, 3 s on a kept tree; `ev_rec_config` and `ev_rec_plan` import
  with numpy, torch and scipy blocked; five mutations -- `draw_rows` with k
  + 1, the comparisons reversed, a `theta` key in `recording`, the split
  not checked, the fractions' tolerance 0.2 -- each caught). Read against
  this entry by the orchestrator and merged at `df02780`.
- Fix round (2026-11-23, after the final check, report 2026-11-23_1730: F2,
  F3, F4, S3; D-167): `ev_rec_config.validate` refuses a non-finite
  factor, `gate.n_ref` above 1,000 and a `mem` or `walltime` with a
  trailing newline; `rec_fixture.boundary_copy(dst, edits)` added (a copy
  of the recorded-kind project whose simulated bank has chosen theta
  entries set, for R35f and R36h). 6 of 6 checks, R33a extended.

### Block 34: the gate (added 2026-11-23; D-161, D-163 (4); roadmap tool, step S2, wave 1)

- Modules: `hpc/pipeline/evaluation/ev_gate.py` (new; numpy and
  `npe_misspec`; no torch, no file is read or written),
  `hpc/pipeline/tests/test_smoke_rec_gate.py` (new; checks R34*; needs
  numpy and scipy only).
- Public API:
  - `ROLE_NONE, ROLE_REF, ROLE_RANK, ROLE_NULL, ROLE_CHECK = 0, 1, 2, 3, 4`;
  - `assign_roles(donor, fracs, n_ref, n_rank, n_null, rows_per_donor,
    seed) -> role`: `donor` (n,) int, the networks of D_sim's rows in global
    row order; `role` (n,) int8;
  - `member_statistic(z_ref, z, factors) -> (t, bandwidths, median)`:
    `z_ref` (N_ref, d_z), `z` (n, d_z); `t` (n,) float64, `bandwidths`
    (n_h,) float64, `median` float;
  - `gate(z_ref, z_rank, z_null, z_test, factors) -> dict`: inputs of shapes
    (M, N_ref, d_z), (M, N_rank, d_z), (M, N_null, d_z), (M, n, d_z), n >=
    0; returns `t`, `r`, `p_member` (M, n) float64, `T`, `p` (n,) float64,
    `t_rank` (M, N_rank), `t_null` (M, N_null), `r_null` (M, N_null),
    `T_null` (N_null,), `bandwidths` (M, n_h), `median` (M,).
- Rule (sec. 2.6, "The gate"):
  - `assign_roles`: the distinct networks, sorted, are permuted by
    `numpy.random.default_rng(seed).permutation`; with G their number, the
    first floor(f_ref G) of the permuted list are the reference group, the
    next floor(f_rank G) the rank group, the next floor(f_null G) the null
    group, the rest the check group; fewer than one network in a group is a
    ValueError naming G and the fractions. [2026-11-23, at the build,
    D-165: floor(f G) is computed as floor(f G + 1e-9), binary roundoff
    giving 0.29 x 100 = 28.99...; and a ValueError for `fracs` that are not
    four positive numbers summing to 1 within 1e-9, a negative cap,
    `rows_per_donor` below 1, or a `donor` that is not a 1-D integer
    array.] A row's occurrence is its
    position among its own network's rows in the given order (0 for the
    first). Within a group the rows are ordered by (occurrence, position in
    `donor`); the reference takes the first `n_ref` of them; the rank, null
    and check sets take the rows of occurrence below `rows_per_donor`, the
    first `n_rank` and `n_null` of them for the first two, all of them for
    the check set. Every other row has role 0;
  - `member_statistic`: (R5), (R6), with `npe_misspec.bandwidth_grid(z_ref,
    z_ref[:0], factors=factors)` (the reference alone; that function
    subsamples, with its own fixed generator, only above 1,000 points, and
    the configuration bounds `gate.n_ref` at 1,000 (block 33; 2026-11-23,
    F4), so med_m is the median over all pairs, as (R5) states) and the
    kernel sums of
    `npe_misspec.mmd2_multiscale`; computed for all n points at once;
  - `gate`: for each member, `t` on the rank, the null and the test rows
    by `member_statistic` against that member's reference; `r` by (the
    count of rank values strictly below) / N_rank, for the test rows and,
    by the same expression, for the null rows (`r_null`); `T` and `T_null`
    the means over the members (R7); `p` and `p_member` by (R8) with
    `npe_misspec._pvalue`'s convention. Non-finite embeddings are a
    ValueError: a NaN would pass every comparison silently. [2026-11-23,
    at the build, D-165: T is computed from the integer counts, (the sum
    over the members of the counts) / (M N_rank), which is (R7) exactly and
    makes equal values of T equal floats, as the ties of (R8) need; the
    embeddings are cast to float64 before the bandwidths; a ValueError
    when M, N_ref, N_rank or N_null is 0, when M or d_z differs between
    the four arrays, or for a factor that is not positive.]
- Custom code: the member's statistic for many single points at once (one
  kernel matrix between the reference and the points), since
  `npe_misspec.mmd2_multiscale` takes one sample per call and a call per
  window and member would dominate the report's time. Checked:
  `mmd2_multiscale`, `_self_term` (used for the first term). Oracle R34a.
- Test oracles (`test_smoke_rec_gate.py`; seeds fixed in the file):
  - R34a `member_statistic` equals `npe_misspec.mmd2_multiscale(z_ref,
    z[i][None, :], bandwidths, kaa)` for every point i of 50 within 1e-12
    (float64 roundoff of two orders of the same sums), and orders the
    points exactly as their mean kernel similarity to the reference does,
    reversed (Spearman -1);
  - R34b by hand, M = 1, d_z = 2, `factors` (1.0,): three reference points,
    two rank points, two null points and one test point with coordinates
    written in the test; t, r, T and p from loops over the definitions
    (R5)-(R8) in float64, equal within 1e-12; a second case with M = 2
    whose members disagree, for the mean of the ranks;
  - R34c `gate` called with the null rows as test rows returns `T` =
    `T_null`, `r` = `r_null`, and p_j = (1 + #{v : T_v >= T_j}) / (1 +
    N_null) by hand (the null rows are ranked by the function that ranks a
    tested window);
  - R34d size, on toy embeddings of one law: a latent u ~ N(e_1, 0.35^2
    I_16), member m's embedding (Q_m u + 0.08 xi) / ||Q_m u + 0.08 xi||
    with Q_m a fixed random rotation and xi standard normal, M = 5, 800
    reference, 500 rank, 500 null and 4,000 test points, every point with
    a latent of its own: for a in {0.05, 0.1, 0.5}, with k_a = floor(a
    (N_null + 1)) and s_a = sqrt(a (1 - a) (1 / 4000 + 1 / (N_null + 2))),
    the share of test points with p <= a is at most k_a / (N_null + 1) + 3
    s_a and at least k_a / (N_null + 1) - 3 s_a - 0.01 [amended 2026-11-23,
    D-165: the bound was a +- 3 sqrt(a (1 - a) / 4000), which left out the
    spread of the size over the null set, (R10); the implementer measured,
    over 100 toys, SDs of the share of 0.0093, 0.0145 and 0.0228 against
    0.0034, 0.0047 and 0.0079 for the binomial term alone, and 45 of the
    100 toys outside the old bound although the mean shares were 0.0502,
    0.1003 and 0.4960]. The 0.01 on the lower side is for the ties of T,
    which takes multiples of 1 / 2500 and so can equal a null value; a tie
    counts against the window in (R8);
  - R34h [added 2026-11-23, D-165] the size given the null set, (R10), on
    the member's own p-value, whose statistic has no ties: the toy of R34d
    with M = 1, one reference set of 800 points and one rank set of 500;
    300 independent null sets of N_null = 99 points; one set of 2,000 test
    points, the same for every null set; for each null set the share of
    the test points with p_member <= 0.1. With k = floor(0.1 x 100) = 10,
    (R10) is Beta(10, 90), of mean 0.1 and SD 0.0299: the mean of the 300
    shares is within 0.026 of 0.1 (3 x 0.0299 / sqrt(300) = 0.0052 for the
    null sets, plus 3 sqrt(0.09 / 2000) = 0.020 for the one set of test
    points, which shifts every share alike), and their SD is within 20 %
    of 0.0299 (the relative standard error of an SD over 300 sets is about
    4 %; the shared test points move the mean, not the spread);
  - R34e power, the same toy with the test latents displaced by delta
    times 0.35 along e_2: the flagged share at alpha 0.05 is non-decreasing
    in delta over (0, 2, 4, 8, 16) and above 0.9 at 16 (the single-member
    gate of the tool's document reached 0.979 there; the bound leaves room
    for the toy's seed);
  - R34f `assign_roles`: on 40 networks with 1 to 5 rows each, no network
    in two roles; the groups' sizes floor(f G) and the rest; the caps; at
    most `rows_per_donor` rows of a network in the rank, null and check
    sets; with `n_ref` below the reference group's size, only first rows;
    the same roles from the same seed, others from another; with caps
    that do not bind, the roles unchanged when the rows of one network are
    moved together (they then depend on each row's network and occurrence
    alone; with a binding cap the position in `donor` decides which rows
    of an occurrence are taken) [qualified 2026-11-23, D-165]; 3 networks
    refused;
  - R34g non-finite input refused; n = 0 test rows returns empty arrays of
    the right shapes.
  Mutations, each of which must fail a check: `<=` for `<` in the rank;
  the null rows ranked against themselves in place of the rank rows; `>`
  for `>=` in (R8); the missing +1; the bandwidths from the reference and
  the test points pooled; the rank and null groups sharing networks.
- Confirmed: yes 2026-11-23 (D-161, D-163 (4); the dealing of the
  networks and the other caps, D-164).
- Status: smoke-tested 2026-11-23 (`test_smoke_rec_gate.py` R34a-h: 8 of 8
  pass in 22 s `[SANDBOX]`; R34a within 1e-12 of `mmd2_multiscale`,
  Spearman -1; R34d's shares 0.0462, 0.0953, 0.5262 at a = 0.05, 0.1, 0.5
  inside the amended bounds; R34e's flagged shares 0.0525, 0.0695, 0.1255,
  0.3967, 0.9680 at 0, 2, 4, 8, 16 SDs; R34h's mean 0.0969 and SD 0.0298
  against Beta(10, 90)'s 0.1 and 0.0299; the six mutations each caught, by
  R34b, R34b-c-e, R34b-c, R34b-c, R34a-b and R34f). The implementer stopped
  once, on R34d's bound (D-165, contract amendment 1). Read against this
  entry by the orchestrator and merged at `f1544ec`.

### Block 35: the sampler on a recorded project (added 2026-11-23; D-159, D-160, D-162, D-163; roadmap tool, step S2, wave 1)

- Modules: `hpc/pipeline/evaluation/ev_rec_sample.py` (new),
  `hpc/pipeline/tests/test_smoke_rec_sample.py` (new; checks R35*).
- Public API: `python3 hpc/pipeline/evaluation/ev_rec_sample.py --config
  <tag>.json --eval-config <tag>_rec_eval.json --ensemble <name> [--out
  DIR] [--ncpus N]`: exit 0 with the pass line `REC SAMPLE OK: <stem>`, 2
  on its arguments, 3 on a refusal, which writes nothing;
  `ev_rec_sample.run(cfg_path, ecfg_path, name, out=None, ncpus=None,
  log=print) -> provenance dict`.
- Rule (sec. 2.6):
  - the ensemble by `ev_rec_plan.find`, its members by
    `ev_models.rebuild(record, index)` on the simulated bank's index;
    `n_post` not a multiple of M is a refusal;
  - the per-member functions are `ev_sample`'s (`logq_at`, `draws_of`,
    `logq_draws_of`, `embed`, `sample_set`), imported, never copied; the
    eight statistics `ev_simulate.fixed_stats`;
  - the oracle (block 19's, on the rows it can be made on): each member's
    -log q at theta_i equals its `_perrow.npz` `nll` on every row of D_sim
    whose theta_i is in the open box, within `oracle_tol`, row for row, or
    the sampling refuses, naming the member, the largest and the mean
    difference and the tolerance. [2026-11-23, after the final check, D-167] The rows outside
    the open box are left out of the oracle (the runner's record holds the
    flow's finite density there, the evaluation's density is 0 under the
    box prior: sec. 2.6, "The rows outside the open box"); their number is
    logged and written (`n_out_of_box`); they keep log q = -inf and get
    draws like any other row. No row of D_sim in the open box is a
    refusal. Block 19 compares on every row and so refuses such a row; the
    bench's theta is in the open box by construction (block 11's
    amendment), so the two samplers agree on every synthetic project and
    D-162's validation is unchanged. The length of `_perrow.npz` `nll`
    must still be D_sim's n;
  - D_sim. The rows: `ev_rec_data.report_rows`; `calib` and `draw` by
    `calibration_rows` and `draw_rows(donor, k_draw)`; `gate_role` by
    `ev_gate.assign_roles(donor, ...)` with the configuration's gate values
    and the seed `seed_of(seed, "gate")`; N_null below `gate.n_null_min` is
    a refusal. On the draw rows, `sample_set` with the seed `seed_of(seed,
    "sim")` (member k's draws from that seed + k), which also gives their
    densities at theta_i and their embeddings; on the other rows, log q at
    theta_i by `logq_at`, the windows read block by block (never all of
    D_sim's windows in memory when `sim_draw_rows_per_donor` is an
    integer); the embeddings of the gate rows that are not draw rows by
    `embed`. With `sim_draw_rows_per_donor` null every row is a draw row
    and one `sample_set` call on D_sim's rows gives the arrays block 19
    writes (the validation, sec. 2.6);
  - D_r, for each recording bank r of the ensemble: `ev_rec_data.recording`;
    the draws, their mixture densities, their own-member densities and the
    embeddings by `sample_set` on every row with the seed `seed_of(seed,
    r)`, called with no row in the box so that no density at a parameter
    is computed; the eight statistics;
  - outputs under `<out_root>/samples/` (or `--out`), each written to a
    temporary name and renamed: `<stem>_sim.npz`, `<stem>_<r>.npz`,
    `<stem>_sample.json` last.
- `<stem>_sim.npz` (n rows of D_sim; n_d draw rows; n_z rows with an
  embedding, the draw rows and the gate rows): `rows` (n,) int64 (bank
  rows), `donor`, `well`, `cls` (n,) int64, `theta_ref` (n, p) float64,
  `in_box` (n,) bool, `logq_ref_members` (n, M) float64, `logq_ref` (n,)
  float64 (-inf where not in the box), `calib`, `draw` (n,) bool,
  `gate_role` (n,) int8, `draw_idx` (n_d,) int64 (indices into the n rows,
  ascending), `draws` (n_d, n_post, p) float32, `draw_member` (n_post,)
  int64, `logq_draws`, `logq_draws_own` (n_d, n_post) float32, `stats`
  (n_d, 8) float32, `z_idx` (n_z,) int64 (ascending), `z` (M, n_z, d_z)
  float32.
- `<stem>_<r>.npz` (n rows of the bank): `rows`, `cls`, `donor`, `well`,
  `batch`, `subregion`, `window_idx`, `rep_group` (n,) int64,
  `contaminated` (n,) bool, `draws` (n, n_post, p) float32, `draw_member`
  (n_post,) int64, `logq_draws`, `logq_draws_own` (n, n_post) float32, `z`
  (M, n, d_z) float32, `stats` (n, 8) float32. No `theta_ref`, `in_box` or
  `logq_ref*`.
- `<stem>_sample.json`: `schema` "rec_eval_sample/1", the ensemble (name,
  stem, search, campaign, arm, folder), the members' records and their
  checkpoints' sha256, the best fit's `BEST_FIT_COMPLETE.json` sha256, the
  view's kind, the banks' names and globs, `sample_params`, the seeds, the
  oracle's largest and mean difference per member, the gate's sizes
  (N_ref, N_rank, N_null, N_chk, the networks per group), n, n_d, n_z, the
  axes' names and bounds (`axes_of`), `files`, the timings. [2026-11-23,
  the contract of wave B: the keys, block 19's where it has them --
  `schema`, `ensemble`, `stem`, `search`, `campaign`, `arm`, `folder`,
  `view_kind`, `members` (per member: `record`, `ckpt`, `ckpt_sha256`,
  `seed`, as `ev_models.rebuild`'s info), `banks` (the names),
  `bank_globs`, `sim_bank`, `best_fit_complete_sha256`, `sample_params`
  (= `ecfg.sample_params()`), `seeds`, `oracle_max_abs_nats` and
  `oracle_mean_abs_nats` (one value per member), `oracle_tol`, `gate`
  (`n_ref`, `n_rank`, `n_null`, `n_check`, `networks`: the four groups'
  numbers of networks, `seed`), `n`, `n_draw`, `n_z`, `n_out_of_box` [2026-11-23, D-167], `M`, `n_post`, `p`,
  `axes` (`names`, `bounds` or null), `files`, `eval_config`, `config`,
  `config_sha256`, `seconds`, `ncpus`, `torch_threads`.]
- Custom code: none.
- Test oracles (`test_smoke_rec_sample.py`, on `rec_fixture`):
  - R35a the validation's first target (D-162): the as-recorded sampling
    of `S-A1:A1` and of `S-A2-R0`'s two ensembles gives, on D_sim, `draws`
    equal bit for bit to block 19's `<stem>_sim.npz` `draws`, and
    `theta_ref`, `in_box`, `rows`, `donor`, `well`, `cls`,
    `logq_ref_members`, `logq_ref`, `draw_member`, `logq_draws`,
    `logq_draws_own`, `calib`, `stats`, `z` equal to block 19's (floats
    within 1e-9; `draw_idx` and `z_idx` the whole range); on each cohort,
    `draws`, `logq_draws`, `logq_draws_own`, `z`, `stats` and the ids
    equal to block 19's `<stem>_<cohort>.npz`;
  - R35b the recorded-kind project with k_draw = 1: `draw` equals `calib`;
    `logq_ref_members` on every row within 1e-4 nats of the as-recorded
    run's (other batches: float32 roundoff; block 19 measured 1.1e-5);
    `draws` (n_d, n_post, p) with n_post / M per member in member order;
    `z_idx` the sorted union of the draw rows and the gate rows; the roles
    those `ev_gate.assign_roles` gives for the same arguments; the file of
    bank `real` with every key of the contract and none of `theta`,
    `theta_ref`, `in_box`, `logq_ref`, `logq_ref_members`; its windows'
    eight statistics equal to cohort R0's (the bank was written from it);
  - R35c the contract: every key, shape and dtype of the three files; the
    same seed gives identical files; another seed other draws;
  - R35d refusals that write nothing: `n_post` not a multiple of M;
    `oracle_tol` 1e-9 (a member then does not reproduce its record: the
    message names `oracle_tol`); a member's record of another split hash;
    an unknown ensemble; `gate.n_null_min` above the null set's size;
  - R35e the mixture density at the draws equals log mean_m exp(log q_m)
    computed directly from the members in float64 (block 19's E19c, on a
    recording bank).
  - R35f a row outside the open box [2026-11-23, after the final check, D-167]: on a copy of the
    recorded-kind project's banks in which one draw row and one other
    row of D_sim have a coordinate of theta set to exactly 1 and 0, the
    sampling does not refuse; `n_out_of_box` is 2; `in_box` is false,
    `logq_ref` and `logq_ref_members` -inf on those two rows; every
    array of the unmodified run that does not hold theta is reproduced
    (the draws bit for bit: they do not depend on theta), and
    `logq_ref_members` on the other rows; with every row of D_sim
    outside the box the sampling refuses and writes nothing.
  Mutations, each of which must fail a check: the oracle on every row of
  D_sim [2026-11-23]; the seed without its `+ k`;
  the members drawn in another order than their seeds'; `draw_idx`
  unsorted; the gate's
  roles from the calibration rows alone; a `theta` key written to a
  recording's file.
- Confirmed: yes 2026-11-23 (D-159, D-160, D-162, D-163).
- Status: smoke-tested 2026-11-23 (5 of 5 checks, `test_smoke_rec_sample.py`; the five
  mutations each caught: the seed without its `+ k` by R35a and R35b; the
  members in another order by R35a and R35b, or refused by the oracle;
  `draw_idx` unsorted by R35a, R35b and R35c; the roles from the
  calibration rows alone by R35b; a `theta` key by R35b and R35c).
  Measured on the fixture: as recorded, `draws` bit for bit and every
  other array of R35a equal to block 19's (largest difference 0); with
  k_draw = 1, `logq_ref_members` within 1.1e-5 nats and `z` within 2.7e-7
  (scale 0.78) of the as-recorded run's; the mixture at the draws within
  9e-6 nats of the members' in float64.
- Fix round (2026-11-23, after the final check, report 2026-11-23_1730: F1,
  S1; D-167): the oracle on the rows in the open box, `n_out_of_box` in
  the provenance, no row in the box a refusal; R35f added, 6 of 6 checks;
  the mutation "the oracle on every row" is caught by R35f. Measured: 2 of
  16 rows outside the box, the draws bit for bit the unmodified run's, the
  other rows' densities equal (difference 0).
- Build notes (2026-11-23, plumbing; none changes the science):
  - a gate group that receives no network (`ev_gate.assign_roles` raises
    ValueError) is a refusal, exit 3, nothing written;
  - `gate.networks` in `<stem>_sample.json` is the list (reference, rank,
    null, check) of the numbers of networks DEALT to the groups (block
    34's rule: floor(f G + 1e-9) for the first three, the rest to the
    fourth); the number of networks that received a row is smaller only
    where a cap (`n_ref`, `n_rank`, `n_null`) binds;
  - `oracle_max_abs_nats`, `oracle_mean_abs_nats`: lists in member order,
    the mean of the absolute differences, over the rows in the open box
    [corrected 2026-11-23 after the final check, F1 and S1: this note
    said the oracle runs on every row, so a row outside the open box
    refuses, as in block 19; that contradicted sec. 2.6];
  - an error of the project's configuration (`config.ConfigError`) exits
    3 like a refusal;
  - `members` holds `ev_models.rebuild`'s whole info dict; `sim_bank` is
    the list of the simulated bank's shard paths, `bank_globs` maps each
    recording bank's name to its glob;
  - the rows of D_sim that are not draw rows are read one shard of the
    bank at a time.

### Block 36: the quantities of a recorded project and the validation tool (added 2026-11-23; D-160..D-163; roadmap tool, step S2, wave 1)

- Modules: `hpc/pipeline/evaluation/ev_rec_metrics.py` (new),
  `ev_rec_validate.py` (new), one additive change to
  `hpc/pipeline/evaluation/ev_metrics.py` (block 20's), and
  `hpc/pipeline/tests/test_smoke_rec_metrics.py` (new; checks R36*).
- The change to `ev_metrics.py`: `sbc_block(theta_ref, draws, level, seed,
  axes=None)` and `recovery_summary(theta_ref, rec, rows, axes=None)`; with
  `axes` None both name the bench's ten axes as before (`AXES`), otherwise
  the given names, one per column. Nothing else in that file changes, and
  no number of the synthetic evaluation changes.
- Public API: `python3 hpc/pipeline/evaluation/ev_rec_metrics.py --config
  <tag>.json --eval-config <tag>_rec_eval.json [--ensemble <name> ...]
  [--ncpus N]` writes under `<out_root>/metrics/`, per ensemble,
  `<stem>_sim.json` and `.npz`, `<stem>_<r>.json` and `.npz` for each of its
  recording banks; per comparison `paired_<name>_sim.json` and `.npz`; and
  `METRICS.json`, the index (the files, the ensembles reported, those left
  out and why, the comparisons made and those not made and why, the
  report's inputs); pass line `REC METRICS OK`; exit 2 on the arguments, 3
  on a refusal (no ensemble with a current sample).
  `ev_rec_metrics.compute(cfg_path, ecfg_path, names=None, ncpus=None,
  log=print)` the same in a process. Pure functions, no file read or
  written: `edge_rows(theta, eps) -> (n,) bool`; `edge_split(gamma, edge,
  betas) -> dict`; `well_summary(draws, well) -> dict`;
  `flag_shares(flag, group) -> (ids, n, k)`; [2026-11-23, the contract of
  wave B] `gate_size(alpha, n_null, level) -> dict` with `k_gate`,
  `size_mean`, `size_lo`, `size_hi`; and the exact signatures
  `edge_split(gamma, edge, eps, p, betas=BETAS) -> (summary, arrays)`,
  `well_summary(draws, well) -> dict` of `well_ids` (G,) int64 sorted,
  `well_n` (G,) int64, `well_mean`, `well_sd`, `well_q05`, `well_q50`,
  `well_q95` (G, p) float64.
  `python3 hpc/pipeline/evaluation/ev_rec_validate.py --old <out_root of
  the synthetic evaluation> --new <out_root of the as-recorded evaluation>
  [--nll-tol 0.002] [--coverage-tol 0.01]`: exit 0 with `REC VALIDATION OK:
  <n> ensemble(s)`, 3 with one line per failure otherwise.
  [2026-11-23, the contract of wave C] `--record FILE` writes the verdict
  to FILE (a temporary name, then renamed), whether it passes or fails:
  `schema` "rec_eval_validation/1", `old`, `new` (absolute), `ok` (bool),
  `n_ensembles`, `failures` (the failure lines), `nll_tol`,
  `coverage_tol`, `metrics_index_sha256` (the sha256 of
  `<new>/metrics/METRICS.json`, read before anything is compared; null
  when that file is absent), `finished`. Block 38's launcher tells a
  current validation from a stale one by this file; the exit codes and
  the printed lines are the same with and without it.
- Rule (sec. 2.6; every seed is `seed_of(seed, "metrics", <stem>, <data
  set>, <what>)`, block 20's, so that the as-recorded numbers are block
  20's [corrected 2026-11-23 at block 36's gate: the paired bootstrap's
  seed is block 27's, `seed_of(seed, "paired", <name>, "sim")`, which R36d
  requires; the sentence above named one form for all]):
  - [2026-11-24, after the second check, D-168] SBC and TARP are over S_cal (sec. 2.6, "The rows
    outside the open box"), `sbc.n_rows` and the first axis of `sbc_ranks`,
    `tarp_x_f`, `tarp_random_f` being n_cal; an ensemble with an empty
    S_cal refuses the metrics; a sample one of whose files (its
    provenance's `files`) is missing counts as absent: left out and named;
    [2026-11-27, after the third check] the files are looked for where the
    run reads them, `<out_root>/samples/<name>`, not at the provenance's
    recorded paths; a bank with a non-finite label is refused naming the
    bank's glob and the first such label's shard, row and axis;
    in R35f, R36h and the fix-round notes "equal" for the densities of the
    rows in the box means within 1e-4 nats (float32 roundoff of batches of
    another composition; the second tester measured up to 1.3e-5);
  - an ensemble whose sample is absent, or was made with other
    `sample_params`, is left out and named; [2026-11-23, after the final check, D-167]
    so is one whose sample was made from another best fit (its
    `best_fit_complete_sha256` is not the sha256 of the folder's current
    `BEST_FIT_COMPLETE.json`): the launcher already calls that sample
    stale (block 38), and a report run by hand must not use it
    silently;
  - D_sim, with block 20's functions (`nll_block`, `hpd_ranks`,
    `coverage_curve`, `binomial_band`, `coverage_summary`,
    `ND.expected_coverage`, `sbc_block`, `tarp_block`, `recovery_arrays`,
    `recovery_summary`), imported, never copied: the NLL over every row in
    the box, the networks as donors; gamma_i on the draw rows; C_E(beta),
    its binomial band, the members' curves and the summaries over S_cal
    (the calibration rows in the box); SBC and TARP over the calibration
    rows with the project's axis names; the recovery on the draw rows,
    summarised over the calibration rows;
  - the edge split: `edge_rows` is (R1); `edge_split` gives pi_edge, its
    expected value 1 - (1 - 2 eps)^p, C_E^edge and C_E^int on the grid
    (null for an empty set), their `coverage_summary`, and the largest
    absolute difference between the two sides of (R2); the mean NLL over
    the edge rows and over the interior rows of every row in the box, with
    their counts;
  - the gate: `ev_gate.gate` on the embeddings of the roles (`gate_role`
    1, 2, 3 of `<stem>_sim.npz`), once with the check rows as test rows and
    once per recording bank; the flag is p < `gate.alpha`; k_gate (R9), its
    value 0 a refusal naming `gate.alpha` and N_null, computed as
    ceil(alpha_gate (N_null + 1) - 1e-9) - 1 [2026-11-23, the contract of
    wave B: the 1e-9 guards the ceiling against binary roundoff, 0.05 x 100
    being a hair above 5 in some products; it agrees with the flag's own
    comparison, since (1 + c) / (N_null + 1) and alpha_gate are the same
    float whenever they are the same rational]; the size band, the
    central interval at `level` of Beta(k_gate, N_null + 1 - k_gate)
    (`scipy.stats.beta.ppf`), with its mean k_gate / (N_null + 1); the
    check rows' flagged share with its exact (Clopper-Pearson) interval at
    `level` (`scipy.stats.binomtest(...).proportion_ci`) [amended
    2026-11-23, D-165: k_gate and the size band];
  - a recording bank: per window and axis the mean, the SD, the 5 %, 50 %
    and 95 % quantiles of its draws and chi (R3) (`recovery_arrays` with a
    NaN reference for all but the median; its z-score is not written); per
    axis the mean of chi over the windows and its 5 %, 50 % and 95 %
    quantiles over the windows; per well, `well_summary`: n_g and, from the
    pooled draws of its windows (R4), the mean, the SD (ddof 1) and the 5
    %, 50 % and 95 % quantiles per axis, with the well's class, donor and
    batch; `flag_shares` per well, per class and for the bank; per class
    the example well (sec. 8, Q-R4): the well whose mean T over its windows
    is the median over the class's wells (the lower middle one for an even
    number), with its first window; for each member the first two
    principal components of the bank's and the reference rows' embeddings
    pooled (`sklearn.decomposition.PCA`, for the figure);
  - the comparisons: `ev_rec_plan.comparisons`; for each, block 20's
    `paired_block` on D_sim's rows in the box under both ensembles, the
    networks as the bootstrap's groups, sigma_seed the larger of the two
    ensembles' SDs over their members of the members' mean NLL on D_sim;
    the two ensembles' `rows` arrays must be equal, otherwise the
    comparison is not made and is named;
  - NaN and inf are written as null in the JSON files (block 20's rule).
- [2026-11-23, the contract of wave B] In `<stem>_sim.npz` every per-row
  array has the n rows of D_sim, with NaN on a row that has no draws or
  whose theta is outside the box where the quantity needs them (`gamma`
  (n,), `mhat`, `sigma`, `q05`, `q95`, `zeta`, `contraction` (n, p)), so
  that with every row a draw row the arrays are block 20's; `draw` (n,)
  bool is written beside `calib`. The SBC and TARP arrays are block 20's,
  on the calibration rows.
- The files: `<stem>_sim.json` (`schema` "rec_eval_metrics/1"; the
  ensemble; n, n_cal, n_d, M, n_post, p; the axes' names and bounds; `nll`,
  `coverage`, `sbc`, `tarp`, `recovery` as block 20 writes them; `edge`;
  `gate` with the sizes, the bandwidths, alpha, `k_gate`, `size_mean`,
  `size_lo`, `size_hi` and `check`) and its `.npz`
  (block 20's arrays, on their rows, with `edge` (n,) bool, `coverage_edge`,
  `coverage_int`, `gate_T_null`, `gate_T_check`, `gate_p_check`);
  `<stem>_<r>.json` (the bank; n_r; the wells, classes and batches;
  `posterior`, the per-axis contraction; `wells`, one entry per well;
  `gate`, the shares; `example_wells`) and its `.npz` (`rows` and the ids;
  `mhat`, `sigma`, `q05`, `q50`, `q95`, `contraction` (n_r, p) float32;
  `well_ids`, `well_n`, `well_mean`, `well_sd`, `well_q05`, `well_q50`,
  `well_q95`; `gate_t`, `gate_r`, `gate_p_member` (M, n_r), `gate_T`,
  `gate_p` (n_r,), `gate_flag` (n_r,) bool; `pca_coords`, `pca_evr`,
  `pca_source`); `paired_<name>_sim.json` and `.npz` as block 27's.
- `ev_rec_validate.py` (sec. 2.6, "The validation", the cluster's targets):
  for each ensemble both trees report, on D_sim: every row's |nll_new -
  nll_old| at most `--nll-tol`; the largest |C_new - C_old| over the grid
  at most `--coverage-tol`; the SBC verdict of each axis, the Holm
  rejections and TARP's two verdicts equal (sec. 8, Q-R3) [2026-11-23, the
  contract of wave B: block 20 writes no verdict for TARP, so its verdict
  for a reference mode is here "ks_p < 1 - level", with `level` the new
  tree's `<stem>_sim.json` `level`; the SBC verdicts are the `verdict` and
  `holm_rejected` lists of the two JSON files]. A failure line
  names the ensemble, the quantity, the row or the beta, and the two
  values. An ensemble in one tree only, or rows that differ, is a failure.
- Custom code: `edge_rows`, `edge_split`, `well_summary`, `flag_shares`:
  a few lines of numpy each; no library computes them. Oracles R36a-c.
- Test oracles (`test_smoke_rec_metrics.py`):
  - R36a the edge split: `edge_rows` by hand on six rows of p = 3 (a
    coordinate at eps exactly is interior, (R1) being strict); on the toy
    of the tool's document -- theta uniform on (0, 1)^10, x = theta + 0.15
    xi with xi standard normal, 4,000 rows, 1,000 draws per row, seed in
    the file -- with the exact posterior (a product of normals truncated to
    [0, 1]): C_E within the Kolmogorov bound at 0.999 of the diagonal (no
    such claim for C_E^edge or C_E^int: sec. 2.6's caution); with the
    posterior's support cut to [eps, 1 - eps]^10, eps = 0.006693: pi_edge
    within 3 standard errors of 1 - (1 - 2 eps)^10 = 0.1261, C_E^edge(beta)
    = 0 for every beta < 1 of the grid (the truth has density 0), C_E^int
    within 0.03 of the diagonal (the tool document's Check 1.1 measured
    0.0069 on this toy; a property of the toy, not a theorem), and
    C_E(0.95) within 0.02 of 0.95 (1 - pi_edge); (R2) holds within 1e-12
    in both cases; an empty edge set gives null and no error;
  - R36b a recording bank's quantities by definition: for windows whose
    draws are N(mu_j, s_j^2) per axis, 4,000 draws: the mean, the SD and
    the three quantiles within 4 Monte Carlo standard errors; chi = 1 - 12
    sigma^2 exactly from the returned sigma; `well_summary`: the pooled
    mean equals the mean of the windows' means (1e-6); the pooled variance
    equals the mean of the windows' variances plus the variance of their
    means within 1e-3 relative (the law of total variance, the finite-
    sample factors stated in the test); for windows of one law N(mu, s^2)
    the pooled SD of a well of 8 windows and that of a well of 2 are both
    within 2 % of s (it does not shrink like s / sqrt(n_g)); a well of one
    window equals that window;
  - R36c the gate's wiring: on arrays built from the contract, the metrics'
    `gate_T`, `gate_p` and the check rows' values equal `ev_gate.gate` on
    the same embeddings; `flag_shares` by hand; the check share's interval
    equal to `scipy.stats.binomtest`'s; k_gate by hand for (alpha_gate,
    N_null) = (0.05, 500), (0.05, 99) and (0.05, 19): 25, 4 and 0, the last
    a refusal; on random T values, the windows flagged by p < alpha_gate
    are exactly those whose T exceeds the k_gate-th largest null value
    (R9); the size band equal to the quantiles of `scipy.stats.beta(k,
    N + 1 - k)` [2026-11-23, D-165]; the example well by hand on 5 wells
    (the median of the mean T) and on 4 (the lower middle one);
  - R36d the validation's second target (D-162): on `rec_fixture`, the
    as-recorded metrics of each ensemble on D_sim equal block 20's
    `<stem>_sim.json` and `.npz`: the NLL per row within 1e-9, `gamma`,
    `coverage`, its band and the members' curves, the SBC ranks and
    p-values, TARP's curves and p-values, the recovery arrays and summary
    within 1e-12; the paired files of every comparison equal block 27's on
    D_sim within 1e-12. The test's inputs are the samples block 35 writes
    when that block is merged, and until then block 19's `<stem>_sim.npz`
    recast in block 35's contract by the test (every row a draw row);
  - R36e `ev_rec_validate.py`: two identical trees pass; one NLL moved by
    0.003 fails naming the ensemble and the row; the coverage of one
    ensemble moved by 0.02 fails; an SBC verdict flipped fails; an
    ensemble missing from one tree fails; exit codes 0 and 3;
  - R36f the change to `ev_metrics.py`: with `axes` None the two functions
    return what they returned before on a fixed input (the values written
    in the test from the unchanged code); with 14 names, 14 named axes; the
    synthetic suite's E20a-c still pass;
  - R36g the recorded-kind project end to end (after block 35's merge):
    every file of `METRICS.json` exists; the comparison list equals
    `ev_rec_plan.comparisons`; one worker and three give the same files; a
    sample of other values is left out and named, and none current is
    refused.
  - R36h the rows outside the open box [2026-11-23, after the final check, D-167]: on R35f's
    sample, `nll.n_out_of_box` is 2 and `nll.n_rows` n - 2; the NLL of
    every other row equals the unmodified run's; n_cal is one less than
    the calibration rows; `gamma` is NaN on the draw row outside the
    box; `coverage.n_rows` is n_cal; SBC still has every calibration
    row; the metrics and the report are written without an error;
    R36g's addition: a sample whose `best_fit_complete_sha256` is not
    its folder's is left out and named.
  Mutations, each of which must fail a check: `<=` in (R1); the edge
  split over the draw rows in place of S_cal; the pooled draws weighted by
  window length in place of equally; the well's SD divided by sqrt(n_g);
  the gate's reference taken from the null rows; `axes` ignored.
- Confirmed: yes 2026-11-23 (D-160..D-163; the example well and the SBC
  and TARP criterion of the validation, D-164).
- Status: smoke-tested 2026-11-23 (7 of 7 checks, `test_smoke_rec_metrics.py`,
  R36d and R36g with block 35's sampler on the merged tree; the six
  mutations each caught: `<=` in (R1) by R36a; the edge split over the
  draw rows by R36c and R36d; the pooled draws weighted unequally by R36b;
  the well's SD divided by sqrt(n_g) by R36b; the gate's reference from
  the null rows by R36c; `axes` ignored by R36d and R36f; and, for
  `--record`, `ok` always true by R36e). Measured: R36a, exact posterior,
  max |C_E - beta| 0.0100 against the bound 0.0318; cut support, pi_edge
  0.1328 against 0.1261 (1.3 standard errors), C_E^edge = 0 for every beta
  < 1, C_E^int within 0.0085 of the diagonal, C_E(0.95) 0.8227 against
  0.8239; (R2) within 1.1e-16. R36b: the worst quantile 3.38 Monte Carlo
  standard errors from its value; the pooled SD of a well of 8 windows
  0.05013 and of 2 windows 0.05022 against s = 0.05. R36d (D-162's second
  target, in the sandbox, through block 35's sampler): 3 ensembles, the
  largest NLL difference 0 nats, every other difference 0, the paired
  files equal to block 27's; `ev_rec_validate` passes on the two trees.
- Fix round (2026-11-23, after the final check, report 2026-11-23_1730: F5
  and F1's consequences; D-167): a sample made from another best fit is
  left out and named (R36g's addition; the mutation is caught); R36h added
  (the rows outside the open box through the metrics and the report job);
  8 of 8 checks. Measured in R36h: the NLL over 14 of 16 rows, n_cal 7 of
  8 calibration rows, SBC on 8, the gate's statistics unchanged, the
  report written.
- Build notes (2026-11-23, plumbing; none changes the science):
  - `edge_split`'s `betas=None` means `ev_metrics.BETAS`;
  - the example well is chosen by `ev_metrics.example_donors` on the
    wells' mean T (ties by the well's id);
  - `ev_rec_validate`: a tree without `metrics/METRICS.json` is a failure
    line (exit 3), a path that is not a directory an argument error (exit
    2); two trees without any ensemble are a failure; TARP's verdict is
    null when its `ks_p` is null; `--record` (the contract of wave C)
    written by the orchestrator at the gate, checked in R36e;
  - the workers' thread limit is `threadpoolctl`'s, one thread per worker;
  - a comparison whose two samples differ in `rows`, `theta_ref` or
    `donor` is not made and is named; the comparisons the view does not
    make (`ev_rec_plan.comparisons`' second list) are listed in
    `comparisons_left_out` with `E` and `E_ref` null.
- The keys as written (2026-11-23; wave C's contract, part 2: block 37
  reads these and nothing else of block 36):
  - `METRICS.json`: `schema` "rec_eval_metrics_index/1", `config`,
    `eval_config`, `view_kind`, `ensembles` (a dict from the ensemble's
    name to `stem`, `arm`, `search`, `campaign`, `banks`, `datasets` =
    ["sim"] + the banks), `left_out` (a list of `ensemble`, `why`),
    `files`, `comparisons` (a list of `name`, `kind`, `E`, `E_ref`,
    `search`, `search_ref`, `arm`, `arm_ref`, `datasets` = ["sim"],
    `paired` = {"sim": the JSON's path}), `comparisons_left_out` (a list
    of `comparison`, `E`, `E_ref`, `why`), `report_inputs`, `seconds`;
  - `<stem>_sim.json`: `schema`, `ensemble`, `stem`, `search`,
    `campaign`, `arm`, `dataset` "sim", `n`, `n_cal` (S_cal: the
    calibration rows in the box), `n_calib`, `n_d`, `n_in_box`,
    `n_networks`, `M`, `n_post`, `p`, `axes` (`names`, `bounds` or null),
    `level`, `sample`; `nll` (`mean`, `members`, `members_sd`, `n_rows`,
    `n_donors`, `n_out_of_box`); `coverage` (`at` with the keys "0.50",
    "0.68", "0.90", "0.95", `signed_area`, `max_abs_dev`, `n_rows`,
    `band`, `ks_p`, `verdict`, `members`: per member `at`, `signed_area`,
    `max_abs_dev`); `sbc` (`ks_p`, `verdict`, `holm_adjusted_p`,
    `holm_rejected`, each a list over the axes, `holm_alpha`,
    `n_rejected`, `n_rows`, `n_post`, `mean_rank_frac`,
    `ks_band_halfwidth`, `hist_level`); `tarp` (`x` and `random`, each
    with `ks_p`, `max_dev`, `reference_mode`, `notes`); `recovery`
    (`axes`, `n_rows`, `contraction`, `interval90_coverage`, `zeta_mean`,
    `zeta_sd`, `rmse`, `sigma_mean`, lists over the axes, `sigma0`);
    `edge` (`n_cal`, `n_edge`, `n_int`, `pi_edge`, `pi_edge_expected`,
    `eps`, `p`, `edge` and `interior`: each null or `at`, `signed_area`,
    `max_abs_dev`; `r2_max_abs_dev`, `nll_edge_n`, `nll_edge_mean`,
    `nll_int_n`, `nll_int_mean`, `caution`); `gate` (`alpha`, `k_gate`,
    `level`, `size_mean`, `size_lo`, `size_hi`, `n_ref`, `n_rank`,
    `n_null`, `n_check`, `factors`, `bandwidths` and `median` (per
    member), `check`: `n`, `n_flagged`, `share`, `ci` [lo, hi], `level`;
    `note`);
  - `<stem>_sim.npz`: `rows`, `donor`, `cls`, `well`, `in_box`, `calib`,
    `draw`, `edge` (n,); `theta_ref` (n, p); `betas` (101,); `nll_rows`
    (n,), `donor_ids`, `donor_nll` (the networks); `gamma` (n,);
    `coverage`, `coverage_lo`, `coverage_hi`, `coverage_edge`,
    `coverage_int` (101,), `coverage_members` (M, 101); `sbc_ranks`
    (n_calib, p), `sbc_ecdf_diff` (p, 101), `sbc_hist_counts` (p, 20),
    `sbc_hist_edges` (21,), `sbc_hist_expected`, `sbc_hist_lo`,
    `sbc_hist_hi` (20,); `tarp_x_f`, `tarp_random_f` (n_calib,); `mhat`,
    `sigma`, `q05`, `q95`, `zeta`, `contraction` (n, p) float32, NaN on
    the rows without draws; `gate_role` (n,) int8, `gate_check_rows`
    (N_chk,) (indices into the n rows), `gate_T_null` (N_null,),
    `gate_T_check`, `gate_p_check` (N_chk,);
  - `<stem>_<r>.json`: `schema`, `ensemble`, `stem`, `dataset` and `bank`
    (the bank's name), `n_r`, `n_post`, `M`, `p`, `axes`, `level`,
    `n_wells`, `well_ids`, `classes`, `batches`, `sample`; `posterior`
    (`contraction_mean`, `contraction_q05`, `contraction_q50`,
    `contraction_q95`, lists over the axes, `what`); `wells` (a list in
    the order of `well_ids`: `well`, `n`, `cls`, `donor`, `batch`,
    `first_window`, `mean`, `sd`, `q05`, `q50`, `q95` (lists over the
    axes), `n_flagged`, `share_flagged`, `mean_T`, `what`);
    `mixture_note`; `gate` (`alpha`, `k_gate`, `level`, `n_null`,
    `size_mean`, `size_lo`, `size_hi`, `n`, `n_flagged`, `share`,
    `per_class`: a list of `cls`, `n`, `n_flagged`, `share`; `per_well`:
    a list of `well`, `n`, `n_flagged`, `share`; `note`);
    `example_wells` (a list of `cls`, `well`, `mean_T`, `first_window`);
    `pca` (`n_ref`, `source`);
  - `<stem>_<r>.npz`: `rows`, `cls`, `donor`, `well`, `batch`,
    `subregion`, `window_idx`, `rep_group`, `contaminated` (n_r,); `mhat`,
    `sigma`, `q05`, `q50`, `q95`, `contraction` (n_r, p) float32;
    `well_ids`, `well_n` (G,), `well_mean`, `well_sd`, `well_q05`,
    `well_q50`, `well_q95` (G, p); `gate_t`, `gate_r`, `gate_p_member` (M,
    n_r), `gate_T`, `gate_p` (n_r,), `gate_flag` (n_r,) bool;
    `pca_coords` (M, N_ref + n_r, 2) float32, `pca_evr` (M, 2),
    `pca_source` (N_ref + n_r,) (0: a reference row of D_sim, 1: a window
    of the bank);
  - `paired_<name>_sim.json`: block 27's keys (`Dbar`, `ci`, `se`,
    `sigma_seed`, `two_sigma_seed`, `ci_excludes_0`,
    `beyond_2_sigma_seed`, `resolved`, `lower_nll_ensemble`, `n_rows`,
    `n_donors`, `n_boot`, `level`, `E`, `E_ref`, `arm`, `arm_ref`, `kind`,
    ...) with `schema` "rec_eval_metrics/1" and `groups`; its `.npz`:
    `donor_ids`, `delta_d`, `se_d`.

### Block 37: the figures and the report of a recorded project (added 2026-11-23; D-159, D-160; roadmap tool, step S2, wave 1)

- Modules: `hpc/pipeline/evaluation/ev_rec_figures.py` (new; matplotlib,
  Agg; reads the metrics and the samples and computes nothing scientific),
  `ev_rec_report.py` (new; the report job's entry),
  `hpc/pipeline/tests/test_smoke_rec_report.py` (new; checks R37*).
- Public API: `python3 hpc/pipeline/evaluation/ev_rec_figures.py --config
  ... --eval-config ...` writes `<out_root>/figures/<family>/<name>.<fmt>`
  and `FIGURES.md`, one line per figure with a one-sentence caption; pass
  line `REC FIGURES OK: <n> figure(s)`; exit 3 when there are no metrics.
  `python3 hpc/pipeline/evaluation/ev_rec_report.py --config ...
  --eval-config ... [--ncpus N]` runs block 36's `compute`, the figures,
  then writes `<out_root>/REC_REPORT.md` and, last,
  `REC_REPORT_COMPLETE.json` (`ev_rec_plan.report_inputs`, the three pass
  lines, the counts); pass line `REC REPORT OK`.
  [2026-11-23, the contract of wave C]
  `ev_rec_figures.make_all(cfg_path, ecfg_path, log=print) -> [(name,
  caption)]`, `name` the figure's path relative to `<out_root>/figures/`
  without its extension (`ensembles/<stem>/cal_coverage`), in the order
  of `FIGURES.md`; the figures' names are the module's constants
  `ENSEMBLE_FIGURES`, `BANK_FIGURES` (one per recording bank),
  `CORNER_FIGURE` (one per class of a bank), `COMPARISON_FIGURES`,
  `ACROSS_FIGURES`; `FIGURES.md` holds one line per figure in block 21's
  form, `` - `<name>.{<formats>}`: <caption>. ``, under a header that names
  the metrics' folder, the formats, the dpi and what the metrics left
  out. `ev_rec_report.write_report(cfg_path, ecfg_path, log=print) ->
  path` writes `REC_REPORT.md`; `ev_rec_report.run(cfg_path, ecfg_path,
  ncpus=None, log=print) -> record` is the report job, in this order:
  `report_inputs` (before anything is computed), `ev_rec_metrics.compute`,
  `make_all`, `write_report`, `REC_REPORT_COMPLETE.json`. That file's
  keys (`ev_rec_report.COMPLETE_KEYS`): `schema` "rec_eval_report/1",
  `inputs` (`ev_rec_plan.report_inputs(view, ecfg)`), `config`,
  `eval_config`, `view_kind`, `ensembles` (the names reported, sorted),
  `left_out` (the metrics index's), `comparisons` (the names made),
  `n_figures`, `pass_lines` (the metrics', the figures' and the report's,
  in this order), `metrics_index`, `figures_index`, `report` (the three
  files' paths), `seconds` (`metrics`, `figures`, `report`), `finished`.
- The figures. Per ensemble, under `ensembles/<stem>/`: `cal_coverage`
  (C_E with its band and the members' curves; C_E^edge and C_E^int beside
  it), `cal_sbc_ecdf`, `cal_sbc_hist`, `cal_tarp`, `acc_recovery`,
  `acc_zscore`, `acc_nll` (every row, the interior rows, the edge rows),
  `acc_contraction` (per axis: D_sim's and each recording bank's);
  `gate_null` (the distribution of T on the null rows, the check rows and
  each bank's windows); and for each recording bank r: `gate_wells_<r>`
  (each well's share of flagged windows, by class, with alpha_gate and the
  size band [2026-11-23, D-165]),
  `gate_windows_<r>` (p per window, grouped by well, the flagged ones
  marked), `gate_members_<r>` (r_m per member and window), `emb_pca_<r>`,
  `rec_wells_<r>` (per axis, each well's mixture: its median and its 5 % to
  95 % range, by class and batch, on the prior's range),
  `rec_corner_<r>_c<k>` (the pooled draws of class k's example well). Per
  comparison, under `comparisons/<name>/`: `cmp_nll` and `cmp_forest`
  (block 28's, on D_sim). Across: `across/dbar` when there is a comparison.
  The colours are block 21's (an arm its hue, a class its hue and marker,
  gray for references and for D_sim as context), one y-axis per panel, the
  legend below the panels, no title inside the image.
- The report `REC_REPORT.md`: a header (the project, the configuration's
  and the evaluation's files and digests, the view's kind, the ensembles
  reported and those left out with the reason); per ensemble a section
  with tables read from the metrics' JSON files alone -- the NLL (every
  row, interior, edge, each member), C_E at 0.50, 0.68, 0.90, 0.95 for the
  three curves with pi_edge and its expected value and, in one line, sec.
  2.6's caution on the split curves, SBC's rejected axes,
  TARP's two verdicts [2026-11-23, wave C's contract: block 36 writes
  each mode's `ks_p` and no verdict, so the report prints the `ks_p` and
  the verdict `ks_p < 1 - level`, `ev_rec_validate`'s definition, the one
  comparison the report makes], the contraction per axis on D_sim and on
  each bank;
  the gate's sizes, k_gate, the size band and its mean, its check share
  with its interval, and per bank the flagged share per class and the
  wells in decreasing order of their share, with one line saying that a
  share is read against the size band and not against alpha_gate alone
  [2026-11-23, D-165]; the wells' mixtures per axis; then the comparisons' table (Dbar,
  its interval, 2 sigma_seed, the two clauses). The report states numbers
  and what each is; it draws no conclusion about a culture, a condition or
  an arm. It writes "the mixture over the well's windows", never "the
  well's posterior" (D-160), and "flagged by the gate", never
  "misspecified".
- Custom code: none.
- Test oracles (`test_smoke_rec_report.py`, on `rec_fixture`'s recorded-
  kind project and its as-recorded one):
  - R37a the set of figures equals the one the rules above give (an
    independent statement in the test), each listed in `FIGURES.md` with a
    caption and none unlisted; every PNG's signature; a figure to PDF;
  - R37b the figures' module imports no statistics library (scipy,
    sklearn, torch, sbi, npe_diagnostics, bootstrap_paired, ev_metrics,
    ev_rec_metrics, ev_gate) and calls no mean, SD, quantile or median;
    the colours are block 21's;
  - R37c every number of the report's tables, parsed back, equals the
    metrics' JSON value it names at the printed precision; the header's
    digests are the files'; neither "well's posterior" nor "misspecified"
    occurs in the report or in `FIGURES.md`;
  - R37d `ev_rec_report.py` end to end: the three pass lines,
    `REC_REPORT_COMPLETE.json` with the current `report_inputs` and as
    many figures as `FIGURES.md` lists; no metrics refused (exit 3).
  Mutations: the gate's figures dropped; a quantile computed in the
  figures; a table reading a sample file in place of the metrics.
- Confirmed: yes 2026-11-23 (D-159, D-160 for the report's wording; the
  plan of the wave, D-164). The list of figures is Claude's and can change
  with the user's reading of the first report.
- Status: smoke-tested 2026-11-23 (4 of 4 checks, `test_smoke_rec_report.py`;
  the three mutations each caught: the gate's figures dropped by R37a; a
  quantile computed in the figures by R37b; a table reading a sample file
  by R37c, whose run without the samples and the `.npz` files also catches
  the variant that keeps the JSON key in the table's comment). Measured on
  the fixture: the recorded-kind project 37 figures, the as-recorded one
  62 (png and pdf); the report 2,597 numbers in 41 tables and 5,105 in 74,
  each equal to the JSON value it names at the printed precision; the
  scale check draws the per-bank figures for 14 axes, 300 wells in 4
  classes and 10 members.
- Build notes (2026-11-23, plumbing; none changes the science):
  - every table of `REC_REPORT.md` is preceded by an HTML comment `<!--
    cells {...} -->` that names, cell by cell, the metrics' JSON file and
    key the printed value was read from (R37c parses it); the three
    derived cells are TARP's verdict (`ks_p < 1 - level`), the names of
    the axes SBC rejects, and block 20's verdict line of a paired file;
  - "no title inside the image" is read as no figure title (suptitle);
    panel titles stay;
  - `cal_sbc_hist`, `acc_recovery`, `acc_zscore` are block 21's functions
    on D_sim, `cmp_nll`, `cmp_forest` and `across/dbar` block 28's, with
    "donor" relabelled "simulated network"; `cal_sbc_ecdf` and `cal_tarp`
    are redrawn here so that a null `ks_p` does not stop the figure;
  - a class's hue and marker are block 21's three, by the class id modulo
    3: more than three classes repeat them (a palette of more classes is
    the user's call when a project has them);
  - the wells' mixtures are printed and drawn in the unit box, each axis's
    declared range beside its name; no value is mapped to the declared
    coordinate (the metrics do not hold it);
  - `gate_null` draws the distributions of T without a threshold line
    (the k_gate-th largest null value is not in the metrics); the windows
    flagged by the gate are marked in `gate_windows` and `gate_members`
    from `gate_flag`;
  - `REC_REPORT_COMPLETE.json`'s `report` is the path of `REC_REPORT.md`
    (the block wrote a dict of the three paths; corrected by the
    orchestrator at the gate, with R37d).

### Block 38: the jobs and the run sheet of a recorded project's evaluation (added 2026-11-23; D-159, D-162; roadmap tool, step S2, wave 1)

- Modules: `hpc/pipeline/evaluation/ev_rec_submit.py` (new; the launcher,
  standard library only), `submit_rec_eval.sh` (new; its wrapper on the
  login node: `--project <tag>` reads `<tag>.json` and
  `<tag>_rec_eval.json`), `hpc/pipeline/projects/synth/synth_rec_eval.json`
  (new: the as-recorded evaluation of the synthetic project, D-162's
  validation on davinci: `as_recorded` true, `sim_draw_rows_per_donor`
  null, `seed`, `n_post`, `batch_rows` and `oracle_tol` those of
  `synth_eval.json`, `validate_against` its `out_root`, an `out_root` of
  its own beside it; its other values in the next item), the README's
  (`hpc/pipeline/README.md`) run sheets "the evaluation of a recorded
  project" and "synth: the validation of the recorded path (D-162)",
  `hpc/pipeline/tests/test_smoke_rec_jobs.py` (new; checks R38*). The job
  script is block 22's `jobs/eval_run.pbs`, unchanged.
- The values of `synth_rec_eval.json` beyond those named above (2026-11-23,
  wave C's contract): `out_root`
  /davinci-1/home/ldellamea/runs/synth/evaluation_rec; `edge_eps` and the
  `gate` those of block 33's list (0.006693; [0.25, 0.25, 0.4, 0.1], 800,
  1000, 1000, 50, 1, [0.25, 0.5, 1, 2, 4], 0.05); `bootstrap_n`, `level`,
  `dpi`, `formats`, `ncpus` those of `synth_eval.json`; `jobs`: its queue
  (`intel`, D-137), `sample` and `report` 8 cores, 64gb, 24 h (Q-R5),
  `validate` 1 core, 8gb, 1 h (Claude's: it reads two trees of JSON and
  NPZ files).
- Public API: `bash hpc/pipeline/evaluation/submit_rec_eval.sh --project
  <tag> [--dry-run] [--what sample,report,validate] [--resample
  <name>,...|all]` (= `ev_rec_submit.py --config ... --eval-config ...`);
  exit 0 with `REC EVAL SUBMIT OK: <n> job(s) submitted, <d> done, <l>
  live` (`REC EVAL DRY RUN: ... nothing written`), 2 on the arguments or
  the configurations, 3 on a refusal (nothing submitted), 4 when a qsub
  fails.
  [2026-11-23, the contract of wave C] In a process:
  `ev_rec_submit.plan(cfg_path, ecfg_path, what=None, resample=(),
  log=print) -> (view, ecfg, units)`, each unit a dict of `unit`, `kind`,
  `name`, `state`, `why`, `argv`, `live`, nothing submitted or written;
  `run(cfg_path, ecfg_path, what=None, resample=(), dry_run=False,
  log=print) -> exit code`; `default_what(view, ecfg)`; the constants
  `KINDS = ("sample", "report", "validate")`, `STAGE_OF` (I35, I37, I36),
  `REPORT_COMPLETE`, `VALIDATION_RECORD = "VALIDATION.json"`, `JOB_NAMES`;
  `Refuse`. `what` None is `default_what`: sample and report, and
  validate as well for an as-recorded view whose evaluation has
  `validate_against`; `--what validate` for any other view is a refusal
  that says why.
- Rule (block 22's, for these units): each unit is done, live (its job in
  `<out_root>/jobs/ledger.json` queued, held or running) or submitted;
  - `sample:<name>`, one per complete ensemble of the view: done when its
    `<stem>_sample.json` lists files that all exist, with the current
    `sample_params` and its best fit's current `BEST_FIT_COMPLETE.json`
    sha256; otherwise absent or stale (the reason printed), or
    `--resample`; an ensemble whose best fit is not complete is named, not
    sampled;
  - `report`, one: submitted when a sampling is submitted now or live, or
    when `REC_REPORT_COMPLETE.json` records other inputs than
    `ev_rec_plan.report_inputs`; after the samplings (`-W
    depend=afterok:...`); not submitted while no ensemble is sampled or
    being sampled;
  - `validate`, one, only for an as-recorded view with `validate_against`:
    `ev_rec_validate.py --old <validate_against> --new <out_root>`, after
    the report; left out of the default `--what` otherwise;
    [2026-11-23, the contract of wave C] its command ends with `--record
    <out_root>/VALIDATION.json` (block 36). The unit is done when that
    record exists with `old` equal to `validate_against` and
    `metrics_index_sha256` equal to the sha256 of the current
    `<out_root>/metrics/METRICS.json`: the launcher then prints its
    verdict, `validated: OK, <n> ensemble(s)` or `validated: FAILED, <k>
    failure(s), see <record>`, and does not submit it again (a failed
    comparison of unchanged files fails again; to repeat it, delete the
    record). Otherwise it is absent or stale and is submitted after the
    report's job of this run or its live one (`afterok`), or at once when
    the report is done; while the report is neither done, live nor
    submitted now, the launcher says there is nothing to validate yet.
    [2026-11-23, at block 38's gate] A validation that is done is
    submitted again, after the report (`afterok`), whenever the report
    is submitted in this run or is live: that report rewrites the
    metrics its record names, so the record would be stale the moment
    the report ends (the rule of the report after a sampling of this
    run, one step down); a live validation is left as it is, with a
    note that it does not cover the report submitted now;
  - every job is `eval_run.pbs` in `envs.train` with its command in
    `<out_root>/jobs/<unit>.argv` and its log in `<out_root>/jobs/logs/`,
    job names of at most 15 characters (`er_s<i>`, `er_report`,
    `er_valid`), the roadmap's markers with `ROADMAP_STAGE` I35 for the
    samplings, I37 for the report and I36 for the validation
    (`specs/roadmap/tool.yaml`); a program or the job script missing from
    the checkout is a refusal; nothing of the pipeline's state is read or
    written.
- Custom code: none.
- Test oracles (`test_smoke_rec_jobs.py`):
  - R38a the launcher's dry run on the committed `synth.json` (its
    `sbi_dir` at this checkout) and `synth_rec_eval.json`: the committed
    resources, `ROADMAP_STAGE` per unit, the folders not complete named,
    nothing written; the wrapper's plan equal to the launcher's; a bad
    `--what`, an absent project; `synth_rec_eval.json` loads, with the
    `seed`, `n_post`, `batch_rows` and `oracle_tol` of `synth_eval.json`;
  - R38b under the fake PBS on `rec_fixture`'s as-recorded project: the
    samplings, the report after them and the validation after the report,
    every exit 0, the validation's pass line (the new path against the
    fixture's synthetic evaluation: D-162's first target, end to end
    through the jobs); run again: every unit done; `n_post` changed: the
    samplings stale; a report value changed (dpi): the report stale and the
    validation after it [2026-11-23, at the gate], the samplings done;
    `--resample`; a queued job live and not submitted twice;
  - R38c under the fake PBS on the recorded-kind project: one sampling per
    complete ensemble and the report, no `validate` unit; an ensemble
    whose `BEST_FIT_COMPLETE.json` is removed is named and not sampled;
  - R38d the launcher and the wrapper are ASCII and LF, the wrapper passes
    `bash -n`, and the launcher imports the standard library and the
    pipeline's own modules only.
  [2026-11-23, the contract of wave C] R38b and R38c run the real
  programs when `ev_rec_report.run` and `ev_rec_validate.validate` are
  implemented in the checkout. While either is a stub (block 37 is
  written in parallel), they run on a copy of the checkout's
  `hpc/pipeline` in a temporary directory in which the test replaces
  `ev_rec_report.py` and `ev_rec_validate.py` by stand-ins of a few
  lines (a report that writes `REC_REPORT_COMPLETE.json` with the current
  `report_inputs` and a `metrics/METRICS.json`, and prints `REC REPORT
  OK`; a validation that writes its record and prints `REC VALIDATION OK:
  <n> ensemble(s)`), so that the launcher's rules -- the order, done,
  stale, live, `--resample`, the validation's record -- are tested in the
  block's own worktree; the samplings are block 35's real ones in both
  cases. The summary line of each check says which programs ran.
  Mutations: the report not after the samplings; the sampling's values
  ignored in the staleness test; the validation submitted for a recorded
  view; live jobs ignored.
- Confirmed: yes 2026-11-23 (D-159, D-162; the resources, D-164).
- Status: smoke-tested 2026-11-23 (4 of 4 checks, `test_smoke_rec_jobs.py`,
  on the merged tree with the real programs: R38b, three samplings, the
  report after them and the validation after the report through the fake
  PBS, every exit 0, `REC VALIDATION OK: 3 ensemble(s)` -- D-162's two
  targets end to end through the jobs, in the sandbox; R38c, two samplings
  and the report, no validation unit. The four mutations each caught: the
  report not after the samplings by R38a, R38b and R38c; the sampling's
  values ignored by R38b; the validation for a recorded view by R38c; live
  jobs ignored by R38b; and, for the gate's rule, the validation not after
  a report of the same run by R38b).
- Build notes (2026-11-23, plumbing; none changes the science):
  - `er_s<i>`: i is the ensemble's position among the view's complete
    ensembles, in the configuration's order;
  - a validation record whose `metrics/METRICS.json` is absent is stale;
    `old` is compared with `validate_against` as absolute paths;
  - `--what validate` without `report` reads the report's state and its
    live job to place the validation, and never submits the report;
  - `--resample` without `sample` in `--what` is ignored (block 22's);
  - R38a's resources and `ROADMAP_STAGE` per unit are checked on a second
    dry run, the committed `synth_rec_eval.json`'s values on the fixture's
    as-recorded project, the committed `synth.json` having no complete
    best fit in the sandbox;
  - the run sheets pull the branch `sci/tool`, where blocks 33-38 are
    (a fast-forward of `sci/pipeline`); which branch the cluster's
    checkout follows is the user's call.

### Block 40: the tool's package, its schemas and its interfaces (added 2026-11-27; D-172..D-175; roadmap tool, step S1)

- Modules: `sbitool/__init__.py`, `sbitool/__main__.py` (the command
  line's dispatcher), `sbitool/_sources.py`, `sbitool/schema.py`,
  `sbitool/interfaces.py`, `sbitool/tests/test_smoke_tool_schema.py` (all
  new; standard library, plus numpy for `Recording`).
- Public API:
  - `sbitool._sources.use(*names) -> dict`: puts on `sys.path` the folders
    of the named source groups (`"model"`, `"evaluation"`, `"ifr"`,
    `"extractor"`) and returns, per imported file of the group, its path
    and sha256; `SourceError` when a folder is absent (for `"extractor"`:
    naming `SBIX_DIR`);
  - `sbitool.schema`: `BUNDLE_SCHEMA = "sbitool_bundle/1"`,
    `MANIFEST_KEYS`, `MEMBER_KEYS`, `ARCHITECTURE_KEYS`, `PARAMETER_KEYS`,
    `EXTRACTION_KEYS`, `DETECTION_KEYS`, `GATE_KEYS`, `CARD_KEYS` and the
    nested card keys;
    `validate_manifest(data) -> [str]`, `validate_extraction(data) ->
    [str]`, `validate_card_header(data, manifest=None) -> [str]`: every
    problem found, an empty list when valid; never an exception on any
    JSON value;
  - `sbitool.interfaces`: `Recording` (a frozen dataclass: `name`,
    `spikes`: dict electrode id (int) -> sorted int64 sample indices,
    `fs_raw_hz`, `n_samples`, `positions_um`: dict electrode id -> (x, y)
    or None, `grid`: None or (`grid_width`, `index_base`, `pitch_um`),
    `source`: dict) with `duration_s` and `check() -> [str]`; the
    protocols `RecordingAdapter` (`read(path, **options) -> Recording`),
    `Simulator` (`simulate(theta, seed) -> x`, theta (p,) in the unit box,
    x (n_windows, W): the interface wave 2 of step S2 will call, D-172)
    and `Scheduler` (`submit(job) -> id`, `state(id) -> str`: step S5's);
  - `python3 -m sbitool <command> ...`: the commands of blocks 41-46,
    each exiting 0 with a pass line, 2 on the arguments, 3 on a refusal.
- Rule: sec. 2.7. The validators check keys (missing, unknown), types,
  ranges and the cross-conditions sec. 2.7 states (M members, p names,
  units and bounds, lo < hi, the files' digests' form, a complete card's
  agreement with the manifest).
- Libraries: standard library; numpy. Custom code: none.
- Test oracles (`test_smoke_tool_schema.py`, T40*):
  - T40a a valid manifest and card header (written by the test from sec.
    2.7) give []; each of a table of single faults (a missing key, an
    unknown key, a wrong type, M not the members' number, lo >= hi, a
    non-finite number, an integer beyond float range, `TODO` in a complete
    card, an arm that differs from the manifest's) gives exactly one
    problem naming the key;
  - T40b `Recording.check`: unsorted or out-of-range sample indices, a
    non-positive rate, positions for other electrodes than the spikes'
    are each named; `duration_s` = n_samples / fs_raw_hz;
  - T40c `_sources.use("model", "evaluation", "ifr")` imports the modules
    sec. 2.7 names and returns their digests (recomputed by the test);
    `use("extractor")` without the folder raises `SourceError`;
  - T40d `python3 -m sbitool` without a command exits 2 and lists the
    commands; the package's own modules import nothing outside the
    standard library, numpy and the libraries their block names (static).
- Status: smoke-tested 2026-11-27 (4 of 4 checks, `test_smoke_tool_schema.py`;
  six mutations each caught, one of them as a hang: T40a a table of 54
  single faults and a fuzz of more than 5,000 odd values, none raising).
- Build notes (2026-11-27, plumbing): the card's `training.members` is the
  integer M; the members' seeds are strictly ascending; `provenance`'s
  `project`, `ensemble`, `search`, `campaign`, `commit` are non-empty
  strings; a digest is 64 lower-case hex characters; `created` parses as
  an ISO time; the ranges of `architecture` are the runner's own; not
  checked in the manifest: `n_post` a multiple of M (block 45 refuses it),
  the order size_lo <= size_mean <= size_hi; `python3 -m sbitool -h` lists
  the commands and exits 0; `use("extractor")` sets `SBI_HPC_DIR` when
  unset (the orchestrator's, at the gate, after the block's report).
- Confirmed: 2026-11-27 by D-176 (the plan and the key lists as written).

### Block 41: the model bundle -- export, check, rebuild (added 2026-11-27; D-174, D-175; roadmap tool, step S3)

- Modules: `sbitool/bundle.py`, `sbitool/model.py`, `sbitool/export.py`,
  `sbitool/tests/test_smoke_tool_bundle.py` (new; torch, safetensors,
  and through `_sources` the estimator and the evaluation's modules).
- Public API:
  - `python3 -m sbitool export --config <tag>.json --eval-config
    <tag>_rec_eval.json --ensemble <name> --out <file>.sbibundle
    [--extraction FILE.json] [--units FILE.json] [--card CARD.md]
    [--name NAME]`: pass line
    `BUNDLE EXPORT OK: <file> (<M> members, <bytes> bytes, card complete |
    draft)`;
  - `python3 -m sbitool bundle-check <file>.sbibundle [--rebuild]`: pass
    line `BUNDLE CHECK OK: ...`; exit 3 with one line per problem;
  - `bundle.read(path) -> Bundle` (the manifest, the card's text, lazily
    the members' states and the gate's sets; every member's sha256
    verified when it is read); `bundle.check(path) -> [str]`;
  - `model.build(architecture, p, W) -> model` (untrained, eval mode);
    `model.load(bundle) -> [model_0, ..., model_{M-1}]` (each state
    loaded, its `state_sha256` and `backbone_repr` verified).
- Inputs: the project's configuration and the evaluation's (block 33);
  the ensemble's best fit (its records and checkpoints, the one place a
  `.pt` is read, with the pipeline's own loader); block 35's
  `<stem>_sample.json` and `<stem>_sim.npz`; block 36's `<stem>_sim.json`
  and `METRICS.json`; the parameters' names and bounds from the sample's
  `axes`; their units [2026-11-27, after block 41's ISSUE, D-178]: the file
  of `--units FILE.json` (a list of p non-empty strings) when given; else
  the simulated bank's sidecar `latent_spec.units` when it holds p
  non-empty strings; else "1" on every axis when the declared coordinate
  is the unit box itself (no bounds, or every bound [0, 1]: the value
  reported is u, a pure number); else a refusal naming `--units` (a unit
  is never guessed); the number of networks of the training split,
  counted on the bank with the members' own split, which the export
  hands to the card's template; optional `--extraction` (exactly `EXTRACTION_KEYS`) and
  `--card`.
- Rule (sec. 2.7): the export refuses (exit 3, nothing written) when the
  ensemble is not complete, its sample or its metrics are absent or not
  current (block 36's `sample_left_out`), the gate's sets are not in the
  sample, or `--extraction` is not valid. The gate's sets are the rows of
  `z` at the roles 1, 2, 3 of `gate_role`, in row order. Without `--card`
  the card is the pre-filled template (block 42) and the bundle a draft;
  with it the card is validated against the manifest, and the bundle is
  complete only if the card is. The zip is written to a temporary name,
  re-read and checked, then renamed. `bundle.check`: the members are
  exactly sec. 2.7's; the manifest validates; every digest holds; no
  member's name ends in `.pt`, `.pkl`, `.npy` or `.npz`.
- Libraries: `zipfile`, `json`, `hashlib`; `safetensors.torch`,
  `safetensors.numpy`; the pipeline's `build_joint_model`,
  `make_backbone`, `state_sha256`. Custom code: none.
- Test oracles (`test_smoke_tool_bundle.py`, T41*; on `rec_fixture`'s
  recorded-kind project, sampled and evaluated by blocks 35-36 into the
  test's own folder):
  - T41a the rebuild is the pipeline's: for each member of an exported
    bundle, the loaded state's digest equals the checkpoint's
    `model_sha256`; on 16 rows of D_sim, log q_m(theta_i | x_i) equals
    `ev_models.rebuild`'s member bit for bit, and so do 8 draws per row
    under one seed and the embeddings;
  - T41b the bundle's content: the members are exactly sec. 2.7's; the
    manifest validates and its `parameters`, `gate`, `members` equal the
    sample's `axes`, the metrics' `gate` and the records; `z_ref`,
    `z_rank`, `z_null` equal the sample's `z` at the three roles;
    `evaluation/sim.json` is the metrics' file byte for byte;
  - T41c safety and integrity: one byte changed in a member's weights, in
    the manifest or in the gate's file makes `bundle-check` exit 3 naming
    the member [fix round of 2026-11-28, D-180, after the tester's F2:
    for the manifest this holds where the change breaks its JSON, its
    schema, a digest, or its agreement with the members it describes
    (below); a manifest edited into another one that is valid and agrees
    with every member passes: the file cannot vouch for itself, its
    sha256 is its identity]; a zip with an extra member is refused; the package's
    reading path contains no call of `torch.load`, `pickle` or
    `numpy.load` (static, over `bundle.py` and `model.py`), and `export.py`
    reads a checkpoint only through `ev_models.rebuild`;
  - T41d the refusals, each exit 3 and nothing written: an unknown
    ensemble, no sample, a sample of other values, an invalid
    `--extraction`; the same export twice gives members with the same
    digests;
  - T41e a complete card (written by the test from `CARD_KEYS`, in
    agreement with the manifest) gives `card_complete` true; a card with
    one `TODO` gives a draft; without `--card` the bundle carries block
    42's template and is a draft (SKIP, naming block 42, while
    `card.template` is a stub: blocks 41 and 42 are written in parallel).
    [2026-11-27, D-178, on the merged tree] the template's
    `training.n_networks` is an integer and, in the fixture, where every
    network has the same number of windows, n_networks times that number
    equals `training.n_windows` and the record's `n_train` (counted by
    the test with numpy on the bank's donors); `parameters.units` is "1"
    on every axis of the fixture (unit box); `--units` with p strings
    replaces them; `--units` with p - 1 strings exits 3, names `--units`
    and writes nothing.
  Mutations, each of which must fail a check: the members written in
  another order; a state tensor dropped; the digest not verified at
  reading; the gate's sets taken from the wrong roles.
- Fix round of 2026-11-28 (D-180; the tester's F2, S4, S6; these clauses
  complete the entry above):
  - `bundle.check` also compares the manifest with the members it
    describes, one problem per disagreement: `card_complete` with what
    `schema.validate_card_header(header, manifest, complete=True)` gives
    for the bundle's own CARD.md (a card that does not parse is not
    complete); `gate` with the `gate` of `evaluation/sim.json` on
    `schema.GATE_KEYS`, the keys the export copies from it (a
    `sim.json` without `gate` is a problem); `parameters.names` and
    `parameters.bounds` with `evaluation/sim.json`'s `axes` where it
    holds `axes`; `M` and `d_z` with the shapes of the gate's sets. The
    tests' hand-made bundles therefore carry an `evaluation/sim.json`
    that holds the manifest's `gate`.
  - `bundle-check`'s pass line ends with the file's sha256 in full: the
    bundle's identity, which a publication or a hub holds outside the
    file. A bundle cannot vouch for itself; authenticity against a
    deliberate edit is the sharing step's (sec. 8, Q-T5).
  - The export refuses (exit 3, nothing written) an `--extraction` whose
    `w_size_s` is not the simulated bank's sidecar `w_size` (compared
    with math.isclose, rel_tol 1e-12): the bin width the networks were
    trained on. [Corrected 2026-11-28, D-182, after the second tester's
    F1] Where the sidecar's `latent_spec` holds `n_e` (the banks the
    Sbi-extractor's `export_windows` writes), the same for
    `electrodes_per_subset`. `gaussian_window_s` is compared with
    nothing: no bank's sidecar holds the smoothing width. It and the
    other extraction values are the author's declaration. [As first
    written, 2026-11-28: "where the sidecar holds `gaussian_window` or
    `electrodes_per_subset`", top-level keys that no bank writer
    writes, so that the two comparisons could never run.]
  - Tests: [D-182] T41d compares `electrodes_per_subset` on a sidecar
    with `latent_spec.n_e`, and shows that a `gaussian_window_s` of
    another value is accepted. T41c gains the three edits of the tester's F2 (the gate's
    alpha, a parameter's name, `card_complete` on a draft), each exit 3
    naming what disagrees, and the sha256 on the pass line; T41d gains
    the `--extraction` of another `w_size_s`.
- Status: smoke-tested 2026-11-27 (5 of 5 checks,
  `test_smoke_tool_bundle.py`, on the merged tree with blocks 40, 42, 43,
  44; the four mutations each caught, implementer's report). Measured in
  T41a on the fixture's ensemble: largest absolute difference 0 between
  the pipeline's networks and the bundle's for log q, the draws, the
  densities at the draws and the embeddings, also in a separate process
  in which `torch.load`, `numpy.load` and `pickle` raise; bundle of
  2.43 MB.
- Build notes (2026-11-27): science-side, D-178 [orchestrator]: the
  source of `parameters.units` (Inputs, above), after the implementer's
  ISSUE that the entry named none. Plumbing: the card's header is parsed
  by block 42's `card.parse`; the export counts the networks of the
  training split with the pipeline's `grouped_split` on the bank's
  donors and the record's fractions and seed, refuses when the split's
  hash is not the record's, and hands the count to `card.template`;
  `safetensors.numpy.save_file` writes a non-contiguous array's raw
  buffer, so every array is made contiguous before it is saved (block 45
  does the same).
- Confirmed: 2026-11-27 by D-176.
- Fix round of 2026-11-28, built: smoke-tested 2026-11-28 (5 of 5 checks
  on the merged tree; ten mutations, nine caught by T41c or T41d, the
  tenth -- the except of block 33 widened back -- by no suite, checked by
  the implementer by hand). Accepted at the gate (plumbing): the
  comparisons are by JSON equality; a member whose own digest fails is
  not compared again; a simulated bank's sidecar without `w_size` is a
  refusal; `gaussian_window` and `electrodes_per_subset` are looked for
  at the sidecar's top level (the fixture's sidecar holds neither, so
  those two comparisons are tested on a hand-written sidecar only).

### Block 42: the card (added 2026-11-27; D-158 (4), D-173, D-175; roadmap tool, step S3)

- Modules: `sbitool/card.py`, `sbitool/tests/test_smoke_tool_card.py`
  (new; PyYAML `safe_load` and `safe_dump`).
- Public API: `card.template(manifest, records, project=None,
  n_networks=None) -> str` [`n_networks`, 2026-11-27: the networks of the
  training split, which no record holds; block 41's export counts them on
  the bank and passes them; None leaves the field `TODO`]
  [`records`: the members' record dicts in member order; `project`: the
  project's tag or None] (a
  `CARD.md` with the header pre-filled and `TODO` elsewhere, and a body of
  headed sections that say what to write); `card.parse(text) -> (header,
  body)`; `card.check(text, manifest) -> [str]` (through
  `schema.validate_card_header`); `python3 -m sbitool card-template
  --bundle FILE --out CARD.md` [2026-11-28, the tester's F5: the entry
  first said "<bundle or export arguments>"; the export's own route to
  the template is `export` without `--card`]; `python3 -m sbitool
  card-check CARD.md [--bundle FILE]`: pass line `CARD OK: complete` or
  `CARD OK: draft, <k> field(s) to fill: <names>`; exit 3 on a malformed
  card (no header, YAML that does not parse, an unknown or missing key).
- Rule (sec. 2.7, "The card"): pre-filled from the records and the
  project: `training` (arm, members, the networks and windows of the
  training split, `split_fracs`, the runner's hyperparameters from a
  member's record `config`, restricted to the keys that describe the
  model and its optimisation), `simulator.searched_parameters`,
  `extraction`, `software`; `TODO` in `title`, `scientific_question`,
  `simulator.model_type`, `simulator.reference`,
  `simulator.fixed_parameters`, `training.hyperparameter_search`,
  `references`, `authors`, `licence`, `cite`, `data`, `limitations`. The
  header is written with `safe_dump` in `CARD_KEYS`' order; a value is
  never guessed.
- Libraries: PyYAML. Custom code: none.
- Test oracles (`test_smoke_tool_card.py`, T42*):
  - T42a the template of the fixture's ensemble parses; its pre-filled
    values equal the records' and the manifest's (computed by the test);
    `card-check` says draft and names exactly the `TODO` fields;
  - T42b every `TODO` replaced: complete; then each of a table of single
    faults (a key removed, an unknown key added, an empty list, `arm`
    changed, a searched parameter renamed, a `TODO` left in a nested
    value) is named;
  - T42c malformed cards (no header, one `---`, YAML with a tab error, a
    header that is a list) exit 3 with one line; a card with YAML tags
    that would build Python objects is refused (`safe_load`);
  - T42d parse(template) then dump gives the same header (a round trip).
- Status: smoke-tested 2026-11-27 (6 of 6 checks, `test_smoke_tool_card.py`,
  on the merged tree; the two checks of the commands, T42e
  `card-template` and T42f `card-check` on a bundle written by block 41's
  `bundle.write`, were added by the implementer).
- Build notes (2026-11-27, plumbing): `training.n_networks` is no
  record's (a record holds `n_train`, the training windows), so
  `template` takes it as `n_networks` from the export and leaves `TODO`
  without it; an `n_networks` that is not an integer >= 1 is a
  ValueError; T42f's bundle carries gate sets of the sizes its manifest
  declares, which `bundle.check` compares (the test first wrote sets of
  4 rows, written before block 41 existed).
- Confirmed: 2026-11-27 by D-176.

### Block 43: recordings and the device's layout (added 2026-11-27; D-158 (1), D-173; roadmap tool, step S4)

- Modules: `sbitool/recordings/__init__.py` (the adapters' registry),
  `sbitool/recordings/ptrain.py`, `sbitool/recordings/si.py`,
  `sbitool/recordings/layout.py`,
  `sbitool/tests/test_smoke_tool_recordings.py` (new; numpy, scipy;
  `spikeinterface` and `probeinterface` imported inside `si.py` only; the
  extractor through `_sources.use("extractor")` in `ptrain.py` only).
- Public API:
  - `recordings.read(path, adapter, **options) -> Recording`;
    `recordings.ADAPTERS` ("ptrain", "spikeinterface");
  - `ptrain.read(folder, fs_raw_hz, ptrain_format="raster",
    ptrain_varname="ptrain", ptrain_name_pattern=..., grid_width,
    index_base, pitch_um) -> Recording`: the extractor's
    `load_ptrain_folder` and `electrode_coords`, its refusals passed on;
  - `si.read(path, reader, detection=None, **reader_options) ->
    Recording`: `reader` the name of a SpikeInterface extractor; a file of
    spike trains (a sorting) gives its units' spike samples, one unit per
    electrode required; a raw recording needs `detection` (Q-T1) and gives
    the detected peaks per channel; the positions are the probe's;
  - `layout.grid(recording) -> (rows, cols)` per electrode for a regular
    grid, `layout.as_grid(positions_um, tol_um) -> (grid_width, pitch_um,
    index of each electrode)` or `LayoutError` when the positions are not
    a regular square grid within `tol_um`.
- Rule: sec. 2.7, "A recording". Nothing is resampled, filtered or
  re-detected in the `ptrain` adapter. An adapter never guesses a format,
  a rate or a geometry: what the file does not hold is an argument, and a
  missing one is a refusal naming it.
  [Fix round of 2026-11-28, the tester's F6] `positions_um` given to the
  `spikeinterface` adapter as an option may be a JSON object, whose keys
  are strings and whose pairs are lists: the adapter turns the keys into
  integers and the pairs into (x, y) floats before `Recording.check`; a
  key that is not an integer's text, or a value that is not two finite
  numbers, is a refusal naming it. T43b gains this route through the
  `windows` command's `--option positions_um=<JSON>`.
- Libraries: scipy.io (through the extractor), spikeinterface,
  probeinterface. Custom code: `layout.as_grid` (no library function
  turns positions into the extractor's grid index; checked:
  probeinterface's `Probe` holds positions and has no grid index).
- Test oracles (`test_smoke_tool_recordings.py`, T43*):
  - T43a `ptrain`: a folder written by the test with `scipy.io.savemat`
    in each of the two formats (known spike samples per electrode) reads
    back to exactly those samples, n_samples and positions (row, column)
    x pitch; a file of the other format, a missing rate, electrodes of
    unequal length are refused;
  - T43b `spikeinterface`: a sorting built in memory by the library
    (`NumpySorting`, known trains) and saved in a format it reads comes
    back to exactly those samples; a recording with a probe gives the
    probe's positions; two units on one electrode are refused;
  - T43c raw voltage (Q-T1's detection): on a synthetic trace with spikes
    of known times and amplitude well above the noise, every spike is
    found within the stated tolerance and no other; without `detection`
    a raw file is refused;
  - T43d `layout.as_grid`: a jittered regular grid within the tolerance
    gives the grid; a hexagonal or irregular one raises `LayoutError`;
    round trip with `ptrain`'s positions.
  T43b and T43c are SKIP, naming the library, where spikeinterface is not
  installed.
- Status: smoke-tested 2026-11-27 (4 of 4 checks,
  `test_smoke_tool_recordings.py`; three mutations each caught). Measured
  in T43c: 48 spikes planted on 4 channels of noise SD 5 to 40, 48 found
  at offset 0, none spurious, at thresholds 5 and 10; none found at 20.
- Build notes (2026-11-27, plumbing): `detect_threshold` multiplies the
  channel's MAD-based noise level (read in spikeinterface 0.106.0's
  source); the noise is estimated on random chunks with a fixed seed (0),
  so that one recording gives one set of spikes; no filter is applied
  before the detection; a peak's sample index is the detector's (the
  trough); `ptrain.read` without a pitch gives `grid` None; the
  `spikeinterface` adapter gives positions and no grid (block 44 derives
  it); a unit's electrode is its `channel` property or its id; probes in
  um or mm, 2-D, one segment; `as_grid`'s width is the larger extent in
  cells, so it equals the device's width only when the electrodes span it.
- Confirmed: 2026-11-27 by D-176 (the detection: Q-T1).
- Fix round of 2026-11-28, built: smoke-tested 2026-11-28 (4 of 4 checks
  on the merged tree; seven mutations of the positions option each
  caught). Accepted at the gate (plumbing): a key may carry a leading
  minus (and then fails `Recording.check`); two keys naming one electrode
  ("5" and "05") are refused; a `positions_um` that is not an object is
  refused naming it.

### Block 44: from a recording to windows (added 2026-11-27; D-158 (1); roadmap tool, step S4)

- Modules: `sbitool/windows.py`,
  `sbitool/tests/test_smoke_tool_windows.py` (new; numpy; the extractor
  and the IFR primitive through `_sources`).
- Public API: `windows.subregions(recording, extraction) -> [Subregion]`
  (the extractor's objects: the electrodes of each, its centre);
  `windows.traces(recording, subregions, extraction) -> (x, fs_ifr_hz)`,
  x (n_subregions, K) [spikes per bin per electrode; D-180], float32 as
  the extractor's
  `subregion_ifrs` returns it [2026-11-27, wave B's contract; the first
  draft said float64]; `windows.cut(x, W) ->
  (windows, ids)`: windows (n, W) float32, ids a dict of (n,) arrays
  `subregion`, `window_idx`, `t_start_s`; `windows.extract(recording,
  extraction, W) -> (windows, ids, subregions)`; `python3 -m sbitool
  windows --bundle FILE --recording PATH --adapter NAME [adapter options]
  --out windows.npz`: pass line `WINDOWS OK: <n> window(s) of <n_s>
  subregion(s)`.
- Rule: sec. 2.7, "From a recording to windows": the extractor's
  functions, called with the bundle's `extraction`; the recording's
  `fs_raw_hz` and grid must equal the extraction's or the tool refuses,
  naming both values (a bundle is not applied to a device it was not
  trained for without the user saying so, Q-T2).
  [2026-11-27, wave B's contract, after block 43's report] The grid of a
  recording: its `grid` when it has one (`grid_width`, `index_base` and
  `pitch_um` each equal to the extraction's; a null pitch on either side
  is not compared); for a recording with positions and no grid (the
  `spikeinterface` adapter's), `layout.as_grid(positions_um)` gives the
  linear index of each electrode on the smallest square grid that holds
  them, its pitch must equal the extraction's `pitch_um` within 1 um, and
  the grid's width is then the recording's own, not compared (the
  partition uses rows and columns, which do not depend on it); a
  recording with neither is refused. The rows and columns go to
  `partition_subregions` with the electrodes' mean firing rates
  (`mean_firing_rates`) and the extraction's `n_subsets`,
  `electrodes_per_subset`, `mfr_threshold_hz`; the traces are
  `subregion_ifrs` with `w_size_s` and `gaussian_window_s`; the windows'
  starts are `windows_of(K, W)`.
  [2026-11-27, contract amendment 1 of wave B, after block 44's ISSUE;
  D-177] The key of the partition. `partition_subregions` breaks ties, in
  rate and in distance, by the lower key of its dicts, which the
  extractor defines as the electrode's linear index on the grid. For a
  recording with a grid the electrode's id is that index. For a recording
  with positions and no grid the dicts are keyed by `as_grid`'s linear
  index (the electrode's place on the device, not the name its file gives
  it), and the subregions' electrodes are mapped back to the recording's
  electrode ids before they are returned, so that one device gives one
  partition whatever its channels are called. `Subregion` members are
  always electrode ids of the Recording.
  The detection. When the bundle's `extraction.detection` is not null,
  the `windows` command passes it to the `spikeinterface` adapter as its
  `detection`; a `detection` among the command's adapter options is
  refused (exit 2: the detection is the bundle's, never the user's). A
  recording that arrives as spike trains is accepted whatever the bundle
  declares, and `meta` then records `detection_applied` null beside the
  bundle's `detection`, with one printed line saying that the spikes
  were detected outside the tool. `cut(x, W, fs_ifr_hz)` takes the IFR's
  rate (the Public API's line above omits it).
- Libraries: the extractor (`partition_subregions`, `subregion_ifrs`),
  `compute_ifr_trace`, `windows_of`. Custom code: none.
- Test oracles (`test_smoke_tool_windows.py`, T44*):
  - T44a the IFR by its definition: on a recording of 9 electrodes with
    a few spikes at chosen times, the trace equals an independent
    computation by the test (the counts per bin by `numpy.histogram`,
    smoothed as eq. (3) of EXTRACTOR_USAGE.md states and divided by n_e
    [D-180; the entry first said "divided by n_e Delta t", the test was
    written to eq. (3)]) within 1e-5 relative to the trace's largest value (the
    extractor returns float32) [2026-11-27; the first draft said 1e-12];
  - T44b the partition: on a 6 x 6 grid with chosen rates, the subregions
    are disjoint, each of n_e electrodes above the threshold, and equal
    the extractor's command-line result on the same folder (its own
    output files read back);
  - T44c the windows: floor(K / W) per subregion, consecutive, from the
    start; `t_start_s` = window_idx W / fs_ifr; float32 of the float64
    trace;
  - T44d a recording whose rate or grid differs from the extraction's is
    refused; a bundle without `extraction` is refused by the command.
  SKIP, naming `SBIX_DIR`, where the extractor's checkout is absent.
- Status: smoke-tested 2026-11-27 (4 of 4 checks,
  `test_smoke_tool_windows.py`, on the merged tree, T44d's part with a
  bundle written by block 41 included; extractor checkout
  `feat/real-arm-parity` 0ff531d).
- Build notes (2026-11-27): science-side, D-177 [orchestrator]: the
  partition's key for a recording with positions and no grid, and the
  detection being the bundle's (Rule, above). Plumbing: the traces are
  float32 as the extractor returns them, so T44a compares at 1e-5, not
  1e-12; the command reads a file first without detection and, when the
  adapter refuses it as raw voltage, again with the bundle's detection,
  and writes which was applied in `meta`. Added with wave C's contract
  (2026-11-27, plumbing, the suite rerun: 4 of 4):
  `windows.from_file(manifest, path, adapter, options) -> dict`
  (`recording`, `detection_applied`, `subregions`, `x`, `fs_ifr_hz`,
  `windows`, `ids`, `meta`), the command's work as one function that
  `infer --recording` and `device-figure` call too; `windows.NoWindow`
  (a `WindowsError`); `windows.parse_option` (the command line's
  KEY=VALUE).
- Confirmed: 2026-11-27 by D-176.

### Block 45: inference from a bundle (added 2026-11-27; D-158 (3); roadmap tool, steps S3 and S4)

- Modules: `sbitool/infer.py`, `sbitool/tests/test_smoke_tool_infer.py`
  (new; torch, numpy; the evaluation's functions through `_sources`).
- Public API: `infer.run(bundle, windows, ids=None, n_post=None,
  seed=None, batch=None, log=print) -> dict` (arrays and JSON); `infer.write(result,
  out_dir)`; `python3 -m sbitool infer --bundle FILE (--windows
  windows.npz | --recording PATH --adapter NAME [options]) --out DIR
  [--n-post N] [--seed S]`: pass line `INFER OK: <n> window(s), <k>
  flagged by the gate`.
- Outputs (`<out>/posterior.npz` and `<out>/posterior.json`; arrays named
  as blocks 35 and 36 name them): `draws` (n, n_post, p) float32,
  `draw_member` (n_post,), `logq_draws`, `logq_draws_own` (n, n_post)
  float32, `z` (M, n, d_z) float32; `mhat`, `sigma`, `q05`, `q50`, `q95`,
  `contraction` (n, p); `gate_t`, `gate_r`, `gate_p_member` (M, n),
  `gate_T`, `gate_p` (n,), `gate_flag` (n,) bool; the ids; and for the
  recording the mixture over its windows, `mix_mean`, `mix_sd`,
  `mix_q05`, `mix_q50`, `mix_q95` (p,). The JSON: the bundle's name and
  sha256, n, n_post, the seed, the parameters' names, units and bounds,
  the gate's `alpha`, `k_gate` and size band, the flagged share, and the
  note that the mixture over the windows is not a posterior of the
  recording (D-160).
- Rule: sec. 2.7, "An inference": `ev_sample.draws_of`, `logq_draws_of`
  and `embed` per member with the seed + k; `ev_rec_metrics`'
  `window_summary` and `well_summary`; `ev_gate.gate` on the bundle's
  three sets. `n_post` a multiple of M or a refusal. No window is
  dropped: a window the gate flags keeps its draws and carries its flag.
- Libraries: the evaluation's functions. Custom code: none.
- Test oracles (`test_smoke_tool_infer.py`, T45*):
  - T45a the same numbers as the evaluation: for a bundle exported from
    the fixture's ensemble, an inference on the recording bank's windows
    with block 35's seed for that bank gives `draws`, `draw_member`,
    `logq_draws`, `logq_draws_own` and `z` equal to block 35's file bit
    for bit, and the summaries, the gate's arrays and the flags equal to
    block 36's within 1e-12;
  - T45b the mixture over the windows equals block 36's `well_summary` of
    the pooled draws when the recording is one well;
  - T45c the contract: every key, shape and dtype; two runs with one seed
    give identical files, another seed other draws and the same `z`;
  - T45d end to end from files: a `ptrain` folder written by the test,
    through `windows` and `infer` by the command line, with a bundle whose
    `extraction` fits it: exit 0, n windows as T44c counts them;
  - T45e refusals: `n_post` not a multiple of M, windows of another
    length than W, a bundle that fails its check.
- Contract of wave C (2026-11-27; these clauses complete the entry above
  and prevail where the two differ):
  - Defaults: `infer.run(bundle, windows, ids=None, n_post=None,
    seed=None, batch=None, log=print)`; None for `n_post`, `seed` or
    `batch` means the bundle's `sample.n_post`, `sample.seed`,
    `sample.batch_rows` (D-179: a member's draws depend on the batch).
    `bundle` is a `bundle.Bundle`; `windows` is (n, W) float32, finite,
    n >= 1; `ids` None or a dict of (n,) arrays, written as given.
  - The numbers are block 35's by construction: one call
    `ev_sample.sample_set(models, windows, zeros (n, p) float64, zeros
    (n,) bool, n_post, seed, batch, log)`, the call block 35 makes for a
    recording bank, with `models = model.load(bundle)`; of its result
    `draws`, `draw_member`, `logq_draws`, `logq_draws_own`, `z` are kept
    and the two reference arrays dropped. Then
    `ev_rec_metrics.window_summary(draws)`, each array cast to float32 as
    block 36 writes it; `ev_gate.gate(z_ref, z_rank, z_null, z,
    gate.factors)` on `bundle.gate_sets()`, its `t`, `r`, `p_member`, `T`,
    `p` written as `gate_t`, `gate_r`, `gate_p_member`, `gate_T`, `gate_p`
    (float64) and `gate_flag = gate_p < gate.alpha` (bool);
    `ev_rec_metrics.well_summary(draws, zeros (n,))`, its one row written
    as `mix_mean`, `mix_sd`, `mix_q05`, `mix_q50`, `mix_q95` ((p,)
    float64). The draws are in the coordinate block 35 writes them in; the
    manifest's bounds are reported, not applied.
  - `run` returns `{"arrays": {...}, "json": {...}}`. The arrays:
    `infer.ARRAYS` (name -> (shape, dtype)), plus `rows` = arange(n) int64
    and the ids. The JSON: exactly `infer.JSON_KEYS` = `schema`
    ("sbitool_posterior/1"), `bundle` (`name`, `sha256`), `n`, `n_post`,
    `M`, `p`, `seed`, `batch_rows`, `parameters` (the manifest's), `gate`
    (the manifest's `alpha`, `k_gate`, `n_null`, `size_mean`, `size_lo`,
    `size_hi`, `level`, and `n_flagged`, `share_flagged` of this
    inference), `ids` (the ids' names), `mixture_note`, `software` (as
    the manifest's keys, of this run).
  - `infer.write(result, out_dir)` writes `posterior.npz`
    (`ev_sample.write_npz`) then `posterior.json` (sorted keys, ASCII),
    each through a temporary name; the folder is created.
  - The command: `--windows FILE` reads block 44's `windows.npz` (its
    `windows`; its `subregion`, `window_idx`, `t_start_s` as the ids);
    `--recording PATH --adapter NAME [--option KEY=VALUE ...]` goes
    through `windows.from_file` (block 44; `--option detection` is
    refused, exit 2). Exit 3 with one `REFUSED:` line and nothing written
    for an `infer.InferRefused`, a `bundle.BundleError`, a
    `windows.WindowsError`, a `recordings.RecordingError`, a
    `_sources.SourceError`; exit 2 for the arguments.
  - T45a reads the bank's windows with `ev_rec_data.recording`, the seed
    from the sample's JSON (`seeds[r]`), leaves `n_post` and `batch` to
    the bundle, and compares with `samples/<stem>_<r>.npz` (bit for bit)
    and `metrics/<stem>_<r>.npz` (the summaries bit for bit as float32,
    the gate's arrays within 1e-12, the flags equal). T45b's oracle is
    numpy on the result's own draws pooled over the windows as float64
    (mean, SD with ddof 1, quantiles 0.05, 0.50, 0.95; within 1e-12). In
    T45c "another seed" also checks that another `batch` gives other
    draws (the reason for D-179) when n exceeds the batch. The evaluated
    project and its bundles are those of block 41's test
    (`test_smoke_tool_bundle._ctx()`), not built again.
- Status: smoke-tested 2026-11-28 (5 of 5 checks,
  `test_smoke_tool_infer.py`, on the merged tree; eleven mutations each
  caught, implementer's report). Measured in T45a for the fixture's two
  ensembles (A1; A2 of the search S-A2-R0), on the recording bank's 24
  windows with 40 draws each and the bundle's batch of 16: `draws`,
  `draw_member`, `logq_draws`, `logq_draws_own`, `z` and the six
  summaries equal to block 35's and block 36's files bit for bit; the
  gate's arrays with a largest absolute difference of 0; the flags equal
  (2 and 0 windows flagged). T45b: the mixture against numpy on the
  pooled draws, largest difference 0.
- Build notes (2026-11-28, plumbing; the implementer's choices accepted
  at the gate): T45c also checks the flag's strict inequality at the tie
  (the gate replaced for one call by the real one with p set to alpha,
  one ulp below and one ulp above: not flagged, flagged, not flagged),
  added at the gate because p = (1 + c) / (1 + N_null) takes the value
  alpha in practice; "two runs give identical files" is tested on the
  npz members' bytes and the JSON's bytes (the zip's time stamps may
  differ); of a `--windows` file `windows` is required and the ids
  present are taken; an absent bundle or recording file is a refusal
  (exit 3); `--adapter` or `--option` with `--windows`, and `--recording`
  without `--adapter`, are argument errors (exit 2); `seed` >= 0 and
  `batch` >= 1 are checked; the command has no `--batch` (the bundle's is
  used); `mixture_note` is worded for a recording (block 36's speaks of
  a well); T45b's comparison with block 36's `well_*` did not apply (the
  fixture's bank has several wells).
- Confirmed: 2026-11-27 by D-176.
- Fix round of 2026-11-28 (D-180): T45d's extraction takes its `w_size_s`
  from the fixture's simulated sidecar (0.1 s), since the export now
  refuses another; 29,800 raw samples at 1 kHz, K 298, W 128, 2
  subregions, 4 windows; 5 of 5 checks on the merged tree. `infer.py`
  unchanged.

### Block 46: the device and the window as figures (added 2026-11-27; D-158 (1); roadmap tool, step S4)

- Modules: `sbitool/plotting/device.py`,
  `sbitool/tests/test_smoke_tool_figures.py` (new; matplotlib, numpy).
- Public API: `device.fig_device(recording, subregions=None,
  highlight=None) -> Figure` (every electrode at its position in um, the
  electrodes without a spike drawn hollow, each subregion's electrodes in
  one colour with its index, one subregion emphasised);
  `device.fig_window(recording, subregion, x, window_idx, W, fs_ifr_hz)
  -> Figure` (above, the spike raster of the subregion's n_e electrodes
  over the window's time span; below, the window's x [spikes per bin
  per electrode; D-180, the entry first said Hz per electrode] against
  time [s]); `python3 -m sbitool device-figure
  --bundle FILE --recording PATH --adapter NAME [options] --out DIR
  [--subregion j --window i]`: pass line `DEVICE FIGURES OK: <n> file(s)`.
- Rule: the figures draw what blocks 43 and 44 return and compute
  nothing (no statistic is taken in this module); the axes carry their
  units; the colours are block 21's.
- Libraries: matplotlib. Custom code: none.
- Test oracles (`test_smoke_tool_figures.py`, T46*): T46a on a 6 x 6
  recording with two subregions, the device figure has one marker per
  electrode at its position and the subregions' electrodes in their
  colours (read back from the artists); T46b the window figure's raster
  holds exactly the spikes of the subregion's electrodes inside the
  window's span, and its lower panel's data are the window's x; T46c the
  command writes PNG files with the PNG signature; the module imports no
  statistics library and calls no mean, SD, quantile or median (static).
- Contract of wave C (2026-11-27; these clauses complete the entry above
  and prevail where the two differ):
  - Positions: `recording.positions_um` when present [um]; else, for a
    grid with a pitch, x = column times pitch, y = row times pitch [um]
    from `layout.grid`; else, for a grid without a pitch, x = column,
    y = row and the axes say "column" and "row" (no um). A recording with
    neither is a `device.FigureError`. Equal aspect.
  - `fig_device`: one axes; one scatter (gid `electrodes`) with one
    marker per electrode in sorted id order; an electrode outside every
    subregion in `SIM_GRAY`; subregion j's electrodes in
    `device.SUBREGION_COLORS[j mod 6]`, the six being block 21's
    `ARM_COLOR["A1"]`, `ARM_COLOR["A2"]`, `ARM_COLOR["A5"]`, `OTHER_ARM`,
    `COND_COLOR[0]`, `COND_COLOR[2]` (read from `ev_figures`, not
    copied); an electrode with no spike has no face colour (hollow), its
    edge in its colour; the index j written beside subregion j's centre
    electrode, so that a subregion is not told by its colour alone
    [amendment 1 of wave C, 2026-11-28: an annotation with gid
    `subregion-<j>` whose `xy` is the centre's position and whose text is
    offset by (6, 6) points, with no halo over the marker: written on the
    marker with a white halo, as first specified and built, the label hid
    the marker's face and a centre read as an electrode without a spike]; `highlight` = j draws a ring in `INK`
    around each electrode of subregion j (a second scatter, gid
    `highlight`). `subregions` are block 44's objects (`members`,
    `center`: electrode ids).
  - `fig_window`: `x` is the subregion's window, (W,) [spikes per bin
    per electrode; D-180]; the window's span is [t0, t0 + W / fs_ifr_hz) with
    t0 = window_idx W / fs_ifr_hz [s]. Upper axes: one `eventplot` row
    per member electrode in the subregion's member order (gid
    `raster-<electrode id>`), holding the electrode's spike times
    index / fs_raw_hz that fall in the span. Lower axes, sharing the time
    axis: [amendment 1 of wave C, 2026-11-28] one `Axes.stairs` (a
    `StepPatch`, gid `ifr`) with values `x` unchanged and W + 1 edges
    t0 + k / fs_ifr_hz for k = 0..W, so that each of the W values is
    drawn over its whole bin [t0 + k / fs_ifr_hz, t0 + (k + 1) /
    fs_ifr_hz) (read back with `get_data()`); as first specified, a line
    with `drawstyle="steps-post"` on W points, the last bin had no
    width and showed as a vertical stroke only. Labels: "time
    [s]", "electrode", and for the window [fix round of 2026-11-28,
    D-180] "x [spikes per electrode per <b>-ms bin]" with <b> =
    1000 / fs_ifr_hz written with %g (block 23's wording for the same
    quantity); the entry first said "IFR [Hz per electrode]", which was
    100 times off at a 10 ms bin.
  - The command reads the bundle and the recording through
    `windows.from_file` and writes `<out>/device.png` and, when
    `--subregion j --window i` are given, `<out>/window_s<j>_w<i>.png`
    (dpi 150); exit 3 with one `REFUSED:` line for a refusal (as block
    45's list, with `device.FigureError`, a subregion or window that does
    not exist included), exit 2 for the arguments. `ev_figures.style()`
    is applied by the command, not at import.
  - Accepted at the gate (2026-11-28, plumbing): the y axis increases
    upward, in plain data coordinates (row 0 at the bottom on a grid);
    in the raster the first member, the centre, is the top row and the
    ticks carry the electrode ids; with `--subregion j --window i` the
    command's `device.png` rings subregion j; an electrode in two
    subregions is a `FigureError`; the figures are made without pyplot;
    `device.SUBREGION_COLORS` is read lazily (the function
    `subregion_colors()`), so that importing the module does not import
    `ev_figures`; a refusal's message is folded onto one line.
  - T46c's static check reads `device.py`'s source: no import of scipy,
    sklearn, statistics or pandas, and no call named mean, std, var,
    median, quantile, percentile, average, histogram or bincount.
- Status: smoke-tested 2026-11-28 (3 of 3 checks,
  `test_smoke_tool_figures.py`, on the merged tree; eighteen mutations
  each caught, implementer's report; the two figures looked at by the
  orchestrator before and after amendment 1).
- Build notes (2026-11-28, plumbing): the stairs are drawn with
  `baseline=None` (an open path: matplotlib's default adds strokes down
  to 0 at the window's ends, which read as the rate falling to 0);
  T46b checks it.
- Confirmed: 2026-11-27 by D-176.
- Fix round of 2026-11-28, built: smoke-tested 2026-11-28 (3 of 3 checks
  on the merged tree; T46b checks the label at bins of 20, 10, 2.5, 30
  and 1000 ms; four mutations of the label and of the drawn values each
  caught). The figure looked at by the orchestrator: "x [spikes per
  electrode per 20-ms bin]", the values unchanged.

### Block 47: a bundle verified against the evaluation it was exported from

- Purpose: the tool's central guarantee (sec. 2.7; T45a in the sandbox) as
  a command that runs on a real evaluated project: an inference from the
  bundle on the project's recording banks, compared with the files blocks
  35 and 36 wrote. It is the check on davinci of D-172 for the tool
  (D-181), where the ensemble has five members, n_post is 1,000 and the
  null set has its real size.
- Module: `sbitool/verify.py`; command `verify` of `sbitool/__main__.py`.
- Public API: `verify.run(bundle, cfg_path, ecfg_path, name, banks=None,
  max_rows=None, log=print) -> dict` (the record, `verify.RECORD_KEYS`);
  `verify.write(record, out_dir) -> path` (`VERIFY.json`, sorted keys,
  ASCII, through a temporary name); `verify.main(argv)`;
  `verify.VerifyRefused`; and, as built, the comparison as functions of
  arrays: `verify.compare(arrays, evaluation, n=None) -> {name: {equal,
  max_abs_diff, shape, dtype}}` over `verify.COMPARED`,
  `verify.first_rows(name, a, n)`, `verify.differences(record) -> lines`.
- CLI: `python3 -m sbitool verify --bundle FILE --config <tag>.json
  --eval-config <tag>_rec_eval.json --ensemble <name> [--bank NAME ...]
  [--max-rows N] [--out DIR]`: exit 0 with `VERIFY OK: <b> bank(s),
  <n> window(s): the bundle reproduces the evaluation (the sample's
  arrays and the summaries bit for bit, the gate within 1e-12)`; exit 1
  with `VERIFY FAILED: ...` naming each bank and array that differs and
  its largest absolute difference; exit 3 with one `REFUSED:` line; exit 2
  for the arguments. With `--out`, `VERIFY.json` is written on exit 0 and
  on exit 1.
- Inputs: a bundle (block 41); the project's configuration and its
  evaluation's (blocks 33, 34); of the evaluation's `out_root`, the
  sample's JSON of the ensemble (block 35: `seeds`, `n_post`,
  `torch_threads`), `samples/<stem>_<r>.npz` (block 35) and
  `metrics/<stem>_<r>.npz` (block 36) for each recording bank r; the
  bank's windows by `ev_rec_data.recording`.
- Rule. For each recording bank r (every bank of the sample's `seeds`
  other than `sim` and `gate`, in sorted order, or those named with
  `--bank`): x = the bank's windows, (n, W) float32, in bank order;
  `infer.run(bundle, x, seed=seeds[r])` with `n_post` and `batch` left to
  the bundle, after `torch.set_num_threads(torch_threads)` of the
  sample's JSON (a result of floating-point arithmetic may depend on the
  number of threads; the previous number is restored at the end). Each
  array of the result is compared with the evaluation's of the same name:
  `draws`, `draw_member`, `logq_draws`, `logq_draws_own`, `z` with block
  35's file and `mhat`, `sigma`, `q05`, `q50`, `q95`, `contraction` with
  block 36's, equal when shape and dtype agree and every element is
  bitwise equal (NaN equal to NaN); `gate_t`, `gate_r`, `gate_p_member`,
  `gate_T`, `gate_p` with block 36's, equal when shape agrees and the
  largest absolute difference is at most 1e-12 (T45a's tolerance);
  `gate_flag` equal elementwise. The verdict `identical` is true when
  every array of every bank is equal in this sense.
  `--max-rows N`: only the first N' = floor(N / batch_rows) batch_rows
  windows of each bank are passed and compared with the first N' rows of
  the evaluation's arrays (the window axis is axis 0, axis 1 for `z`,
  `gate_t`, `gate_r`, `gate_p_member`). This is exact, not an
  approximation: a member's random numbers are drawn batch after batch
  from one seed, so the first batches do not depend on the later ones
  (T47b tests it). N' = 0 is a refusal.
- Refusals (`VerifyRefused`, exit 3, nothing written): a bundle that
  fails its check; an ensemble the project does not have; a sample or
  metrics file absent or not current (block 36's rule, as the export's);
  a bundle whose `provenance.sample_json_sha256` is not the sha256 of the
  ensemble's sample JSON (the bundle was not exported from this
  evaluation); a bundle whose `sample.n_post` or `sample.batch_rows` is
  not the evaluation's; a `--bank` the sample does not have; a bank whose
  number of windows is not the evaluation's.
- Outputs: the record, exactly `verify.RECORD_KEYS`: `schema`
  ("sbitool_verify/1"), `bundle` (`name`, `sha256`), `project`,
  `ensemble`, `stem`, `n_post`, `batch_rows`, `torch_threads`,
  `max_rows` (null or N), `banks` (per bank: `n` the evaluation's
  windows, `n_compared`, `seed`, `identical`, `arrays`: per array
  `equal`, `max_abs_diff` (0.0 for equal arrays; the largest absolute
  difference of the float64 values otherwise, NaN positions that differ
  counted as inf), `shape`, `dtype`), `identical`, `tolerance_gate`
  (1e-12), `software` (`schema.SOFTWARE_KEYS`, of this run), `seconds`.
- Libraries: numpy, torch (through block 45). Custom code: none (the
  inference is block 45's, the comparison numpy's).
- Test oracles (`sbitool/tests/test_smoke_tool_verify.py`, T47a-d; the
  evaluated project and its bundles are block 41's test's, `_ctx()`):
  - T47a both ensembles of the fixture verify: `identical` true, exit 0,
    the pass line, `VERIFY.json` with exactly RECORD_KEYS and every
    array's `max_abs_diff` 0.0 (the gate's at most 1e-12).
  - T47b `--max-rows`: with N = batch_rows and N = batch_rows + 4 the
    first batch_rows windows are compared and are identical
    (`n_compared` = batch_rows < n); N < batch_rows is refused.
  - T47c a difference is found and named [as built, 2026-11-28: on the
    evaluation's arrays in memory, through `verify.compare`, rather than
    in a copy of the tree, whose digests would all have to be rewritten]:
    one draw changed by one float32 step gives `draws` not equal with a
    `max_abs_diff` of that step and every other array equal; one gate
    p-value changed by 1e-9 gives `gate_p` not equal, by 1e-13 equal; a
    flag flipped, a NaN against a number, another dtype, another shape
    and a `draw_member` in another order are each not equal; the command,
    with the evaluation's `draws` read changed by that one step, exits 1,
    names the bank and `draws`, and writes `VERIFY.json` with
    `identical` false.
  - T47d refusals: the bundle of the other ensemble (another sample
    JSON); a bundle whose manifest says another `n_post`; an unknown
    `--bank`; an absent sample file: exit 3, one `REFUSED:` line, no
    `VERIFY.json`.
- Status: smoke-tested 2026-11-28 (4 of 4 checks,
  `test_smoke_tool_verify.py`, written and run by the orchestrator, a
  single block built inline). Measured: both ensembles of the fixture
  identical (24 windows x 40 draws, 17 arrays each); with `--max-rows` 16
  and 20 the first 16 of the 24 windows identical, so the first batches
  do not depend on the later ones; one float32 step (2.98e-08) on one
  draw found. Mutations: the member's seed + 1; the gate's tolerance
  1e-6; bitwise replaced by numpy.allclose; `--max-rows` not rounded to
  whole batches; the sample's digest not compared; `z` cut on axis 0;
  the threads not restored; the threads not set to the evaluation's; a
  failed verification exiting 0 -- each caught. One more, the dtype not
  compared, changes nothing (arrays of another dtype have other bytes).
- Build notes (2026-11-28, plumbing): the recording banks are the
  sample's `banks`; a flag that differs has `max_abs_diff` 1.0; a
  `max_abs_diff` that is not finite is written to `VERIFY.json` as the
  string "inf"; the module calls `export._pipeline`,
  `export._metrics_left_out`, `export._sha256_file` and
  `infer._software` (private names of blocks 41 and 45, to be made
  public when the package is cut at the release); the refusal for a bank
  whose number of windows changed is not covered by a check.
- Confirmed: 2026-11-28 by D-181.

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `bash hpc/pipeline/run_pipeline.sh --project giulia --dry-run` | the stage table and every qsub line; submits nothing | the committed `giulia.json`; davinci paths need not exist (each stage reports `not done`) | seconds |
| `bash hpc/pipeline/run_pipeline.sh --project giulia` | the one command: submits the sequencer's first PBS job (D-098) and prints its id, the log and state paths | davinci | days |
| `bash hpc/pipeline/run_pipeline.sh --project giulia --status` | the stage table from the state file and the live scheduler | -- | seconds |
| `python3 hpc/pipeline/tests/test_smoke_pipeline.py`, `test_smoke_block1b.py`, `test_smoke_block1c.py` | block 1's smoke tests under the fake PBS: the sequencer; its jobs (D-098); the core cap (D-099) | none (synthetic project in a temporary directory; `SBIX_CLONE` for 1c's launcher check) | about 25 s, 3 min, 2 min |
| `python3 hpc/pipeline/tests/test_smoke_block7.py` | block 7's smoke test: the search and the best fit under the fake PBS on the fixture banks | the training environment (torch, sbi, skopt) | about 2 min |
| `python3 hpc/pipeline/tests/test_smoke_block2.py`, `test_smoke_block3.py` | blocks 2 and 3: the bank stages and the input stages on the stand-in davinci; since 2026-10-29 (b) their jobs through the job protocol (D-113) | the Sbi-extractor checkout (`SBIX_CLONE`) at or after D-113, numpy | about 2 min, 30 s |
| `python3 hpc/pipeline/tests/test_smoke_block8.py` | block 8: the one command end to end on the stand-in with the real tools | both of the above | about 1 min |
| `python3 hpc/pipeline/tests/test_smoke_block7c.py` | D-111: the environment's fallback, environment faults, the chain's replacement and watchdog; D-113's `stage_job_units` | the training environment (torch, sbi, skopt) | a few minutes |
| `python3 hpc/pipeline/tests/test_smoke_inputs_check.py` | blocks 9 and 32: the inputs check (D-112's and D-113's rows included; since block 32 on an lcdup15-like tree, with the cohort map, topology metadata, partial topologies and searches rows) | the Sbi-extractor checkout at or after D-113 (Block 9 for the real cohort-map reader; a stand-in otherwise), numpy | 2-7 s |
| `python3 hpc/pipeline/tools/purge_outputs.py --config <file> --what search,best_fit,sim_bank [--confirm <CODE>]` | block 10: the list-then-delete of a project's search, best fit and simulated bank | davinci (the tests: a synthetic project) | seconds to list; minutes to delete hundreds of GB |
| `python3 hpc/pipeline/tests/test_smoke_purge.py` | block 10's smoke test under the fake PBS | none (stdlib) | about 15 s |
| `bash hpc/pipeline/run_pipeline.sh --project synth --dry-run` | blocks 15-17: the synthetic experiment's 13 stages and every kind of qsub line; submits nothing [2026-11-11, block 18: 17 stages with v3; on the running state with `--accept-config-change`] | the committed `synth.json`; davinci paths need not exist | seconds |
| `python3 hpc/pipeline/tests/test_smoke_synthetic.py` | blocks 15-18's smoke test: T15, T16 and T18's config, graph and dry run under the fake PBS, T17 and T18 end to end | numpy and scipy (the bench generator); T17 and T18's end to end the training environment (torch, sbi, zuko, scikit-optimize) | about 30 s; about 2 min more for T17 and T18 |
| `python3 hpc/pipeline/tests/test_smoke_finalist.py` | block 7 (i), 2026-11-14: the best fit follows its search's finalist; `--retry` and the downstream stages | the training environment for its end-to-end checks; the fake PBS | minutes |
| `python3 hpc/pipeline/tests/test_smoke_identity.py` | block 7 (j), 2026-11-14: the search's identity re-checked after an accepted configuration change | the training environment for its end-to-end checks; the fake PBS | minutes |
| `python3 hpc/pipeline/tests/test_smoke_theta_box.py` | block 11's amendment of 2026-11-14: theta inside the open box (the samplers, the builder, the record tool), the shard folder's race | numpy, scipy; the builder's environment | about a minute |
| `qsub -q cpu -o $HOME/w1_checks.log -v W1_REPO=<checkout> <checkout>/hpc/pipeline/tests/run_w1_checks.pbs` | the cluster-side check of the wave W1 (2026-11-16): the three suites above and `test_smoke_pipeline.py`, `test_smoke_block7.py`, `test_smoke_synthetic.py` in one job on a compute node, from a checkout other than the running chain's; pass line `W1CHECK: ALL 6 SUITES PASSED` | davinci: the training environment (`W1_ENV`, default `sbi_env`); here: `W1_SKIP_CONDA=1 bash <script>` | about 15 minutes |
| `python3 hpc/joint/stage4/smoke_test_joint_tune.py -k J37` | block 18's tuner check: `finalists --arm/--out` | numpy, scikit-optimize | seconds |
| `python3 hpc/pipeline/pipeline.py stop-search --config <cfg> [--search <name>] [--kill-running] [--cancel]` | block 30: stop a search by hand (one file; the sequencer acts at its next tick) | a project's configuration | seconds |
| `python3 hpc/pipeline/tests/test_smoke_runs.py`, `test_smoke_stop.py` | blocks 29 and 30 under the fake PBS | none for the config, graph and cap checks; the training environment for the end to end | minutes |
| `python3 hpc/joint/stage3/smoke_test_rep_pairs.py` | block 31: the replicate pair rule and the runner's flags | the training environment (torch, sbi, zuko) | about a minute |
| `bash hpc/pipeline/evaluation/submit_eval.sh --project synth --what data [--dry-run]` | block 23: the data overview's job (`ev_dataview.py`): the windows, the eight statistics, theta and the PCA of the synthetic data set, nine figures | davinci (the tests: the reduced project under the fake PBS) | about a minute on 4,000 + 648 windows |
| `python3 hpc/pipeline/tests/test_smoke_rec_plan.py`, `test_smoke_rec_gate.py`, `test_smoke_rec_sample.py`, `test_smoke_rec_metrics.py`, `test_smoke_rec_report.py`, `test_smoke_rec_jobs.py` | blocks 33-38 (2026-11-23): the evaluation of a recorded project -- the view and the fixture, the gate, the sampler, the quantities and the validation tool, the figures and the report, the jobs under the fake PBS; each prints a PASS / FAIL table and its last line `<k> of <n> checks pass` | the training environment (torch, sbi, zuko, scikit-optimize, scikit-learn, matplotlib); `REC_FIXTURE=<dir>` reuses a fixture tree built before, `REC_KEEP=1` keeps the one a run builds | the fixture about 90 s once; then seconds to a few minutes each |
| `python3 hpc/pipeline/evaluation/ev_rec_sample.py --config <tag>.json --eval-config <tag>_rec_eval.json --ensemble <name> [--out DIR]`, then `ev_rec_report.py --config ... --eval-config ...` | blocks 35-37 by hand: one ensemble's samples, then the metrics, the figures, `REC_REPORT.md` and `REC_REPORT_COMPLETE.json`; pass lines `REC SAMPLE OK`, `REC METRICS OK`, `REC FIGURES OK`, `REC REPORT OK` | `rec_fixture.fixture()`'s `cfg_rec` and `ecfg_rec` (the recorded kind) or `cfg` and `ecfg_as` (a synthetic project read as recorded), with `out_root` in a temporary directory | seconds per ensemble on the fixture |
| `python3 hpc/pipeline/evaluation/ev_rec_validate.py --old <synthetic out_root> --new <as-recorded out_root> [--record FILE]` | block 36: D-162's validation of the recorded path against the synthetic evaluation; pass line `REC VALIDATION OK: <n> ensemble(s)` | the fixture's `old_out` and an as-recorded run of `cfg` with `ecfg_as` | under a second |
| `bash hpc/pipeline/evaluation/submit_rec_eval.sh --project <tag> [--dry-run] [--what sample,report,validate] [--resample <name>,...]` | block 38: the launcher of a recorded project's evaluation on PBS; `--project synth` is D-162's validation on davinci (`synth_rec_eval.json`) | davinci (the tests: the fixture under the fake PBS, `SKIP_CONDA=1`) | seconds; the jobs hours |
| `python3 -m sbitool export --config <tag>.json --eval-config <tag>_rec_eval.json --ensemble <name> --out <file>.sbibundle [--extraction FILE.json] [--units FILE.json] [--card CARD.md]`; `bundle-check <file>`; `card-template`, `card-check` | blocks 41 and 42 (2026-11-27): a trained and evaluated ensemble as a model bundle with its card; run from the repository's root | the training environment, safetensors, PyYAML; blocks 35 and 36's outputs of the project | seconds to a minute |
| `python3 -m sbitool windows --bundle FILE --recording PATH --adapter ptrain\|spikeinterface [--option KEY=VALUE] --out windows.npz`; `infer --bundle FILE (--windows windows.npz \| --recording PATH --adapter NAME) --out DIR [--n-post N] [--seed S]`; `device-figure --bundle FILE --recording PATH --adapter NAME --out DIR [--subregion j --window i]` | blocks 43 to 46 (2026-11-27/28): a recording cut into a bundle's windows, the inference, the device and one window as figures | the Sbi-extractor's checkout (`SBIX_DIR` or `SBIX_CLONE`); spikeinterface for its adapter; the training environment for `infer` | seconds |
| `python3 -m sbitool verify --bundle FILE --config <tag>.json --eval-config <tag>_rec_eval.json --ensemble <name> [--bank NAME] [--max-rows N] [--out DIR]`; `python3 sbitool/tests/test_smoke_tool_verify.py` | block 47 (2026-11-28, D-181): a bundle's inference on the project's recording banks compared with the evaluation's files; the tool's check on davinci | the training environment, safetensors; blocks 35 and 36's outputs of the project; torch's threads set to the evaluation's | the recording banks' sampling again (seconds on the fixture; with `--max-rows`, a few batches) |
| `python3 sbitool/tests/test_smoke_tool_schema.py`, `test_smoke_tool_recordings.py`, `test_smoke_tool_card.py`, `test_smoke_tool_windows.py`, `test_smoke_tool_bundle.py`, `test_smoke_tool_infer.py`, `test_smoke_tool_figures.py` | blocks 40 to 46: the tool's smoke tests (4, 4, 6, 4, 5, 5, 3 checks) | `SBIX_CLONE=<the Sbi-extractor's checkout>`; the training environment; the fixture of blocks 33-38 (`REC_FIXTURE` to reuse a kept one) | about a minute in all |

## 7. Not executable in the sandbox

davinci itself: the real `qsub`/`qstat`, the conda environments, the banks
(45 GB of windows), the Sbi-extractor artefacts of record. The substitute:
the fake PBS of `tests/fake_pbs/`, synthetic artefacts in a temporary work
root, and -- for blocks 2-3 -- the Sbi-extractor fixtures (`export_windows/
bank_plan_fixture.py`, `sim_reextract/smoke_test_sim_reextract.py`, `extractor/
smoke_test_stage_d_jobs.py`) on a stand-in `/davinci-1` tree. The first real
run on davinci is `--dry-run`, then the plain command; the pass lines are
stated in the README.

Blocks 11-14 (2026-11-03): the sandbox installs torch 2.10.0, sbi 0.27.0,
zuko 1.6.0 and pytorch-metric-learning 1.6.3 (CPU) [2026-11-14: the audit's
sandboxes ran 2.9.0; 3.0.3 has no `utils.loss_and_miner_utils`, which
`hpc/dsn/dsn_joint_loss.py` imports (the audit's F207, F306, F708).
Corrected 2026-11-16: davinci's `sbi_env` holds 1.6.3 on python 3.11.15 and
imports that module `[CLUSTER 2026-11-16, the user's paste]`, so the bound
`>=2.3` the files carried never described the environment that runs:
`hpc/requirements.txt` and `hpc/environment.yml` now ask `>=1.6.3,<3`, the
range actually run -- 1.6.3 on davinci and here on 2026-11-03, 2.9.0 in the
audit's sandboxes] and runs the arms at
2 epochs of 3 steps; the searches and the full banks run on davinci (the
roadmap's step S5). `hpc/joint/stage1/smoke_test_bench_burst.py` stops at
its check B4 in the sandbox -- scipy 1.18.1's `kstest(x, "norm",
args=...)` raises a TypeError, on the code before block 11 as after it --
so its checks from B4 on were not run here; blocks 11-14 do not touch the
generator it tests (`bench_burst_generator.py`).

Blocks 33-38 (2026-11-23): the sandbox installs python 3.13, torch 2.15.0, sbi 0.27.0,
zuko 1.6.0, scikit-optimize 0.10.2, pytorch-metric-learning 2.9.0, numpy 2.5, scipy 1.18,
scikit-learn 1.9 and matplotlib 3.11 (CPU) and runs them on `rec_fixture`'s reduced
synthetic project (8 report networks, 16 rows, 2 members, n_post 40; a null set of 2 rows,
so `gate.alpha` 0.5 and `gate.n_null_min` 1 there). Not executable here: a recorded
project's real banks (the extractor's, on davinci), and D-162's validation at full size
(`submit_rec_eval.sh --project synth`: five ensembles of five members, n_post 1,000,
against `/davinci-1/home/ldellamea/runs/synth/evaluation`), whose cluster tolerances
(sec. 2.6, "The validation") are therefore untested until that run; the sandbox tests
its bit-for-bit form on the fixture (R35a, R36d, R38b).

Blocks 40-46 (2026-11-28): the sandbox adds safetensors 0.8.0, PyYAML 6.0.3,
spikeinterface 0.106.0 and probeinterface 0.5.0 and the Sbi-extractor's checkout
(`feat/real-arm-parity` 0ff531d). Not executable here: an export and an inference on a
real project's ensemble (five members, n_post 1,000, a null set of the real size), which
is the check on davinci of D-172; a real recording file of any vendor (the adapters are
tested on files the tests write: `ptrain` folders, and SpikeInterface's own generated
sortings and recordings); whether the default spike detection of raw voltage matches the
simulator's virtual MEA (no pair of raw voltage and reference spikes is in the
repository).

## 8. Open questions

- The evaluation of a recorded project (sec. 2.6, blocks 33-38; raised
  2026-11-23). Decided by the user: D-159 (first step, any recorded
  project, two waves), D-160 (the well's summary), D-161 (the gate's
  ensemble statistic), D-162 (the validation target), D-163 (the draw rows,
  no L-C2ST, the comparisons, n_post, the edge width, the gate's reference
  and the disjoint sets, the threshold as a value of the configuration, the
  tolerances), D-164 (the build plan; Q-R2..Q-R5 confirmed as proposed,
  2026-11-23). Q-R1 is closed by D-172 (2026-11-27); Q-R6 (wave 2, the
  predictive check) is built after the tool's step S1 fixes the
  simulator's interface (D-172); Claude's proposals in
  brackets:
  - Q-R1 how the share of flagged windows of a well, of a class or of a
    bank is read, the windows of a well sharing a culture [no test in wave
    1: the share is reported beside alpha_gate; D-161 leaves it open]
    [2026-11-23, D-165: reported beside the gate's size band (R10), which
    is wide for a small null set; still no test]
    [closed 2026-11-27 by D-172 [user]: the share is reported beside the
    size band and no verdict is given on a well, a class or a bank];
  - [closed 2026-11-23 by D-164: confirmed as proposed] Q-R2 the gate's values the user did not fix: the proportions in which
    the networks are dealt, (0.4, 0.25, 0.25, 0.1) for the reference, the
    rank, the null and the check rows; the caps of the rank and null sets,
    1,000 each; one row per network in the rank, null and check sets; the
    smallest null set accepted, 50 rows [values of the configuration, so
    changing one changes no code];
  - [closed 2026-11-23 by D-164: confirmed as proposed] Q-R3 how SBC and TARP are compared on the cluster in the validation
    [each axis's verdict, the Holm family's rejections and TARP's two
    verdicts unchanged; the user fixed the NLL's and the coverage's
    tolerances only];
  - [closed 2026-11-23 by D-164: confirmed as proposed] Q-R4 the example well of each class in the figures [the well whose mean
    T over its windows is the median over the class's wells: an ordinary
    well for the gate, chosen without looking at the posteriors];
  - [closed 2026-11-23 by D-164: confirmed as proposed] Q-R5 the jobs' resources [sample 8 cores, 64gb, 24 h; report 8 cores,
    64gb, 24 h: block 22's];
  - Q-R6 the predictive check on recordings, wave 2 [not specified: the
    contract "a simulation at a given theta, re-detected as the bank's
    simulations were" and its cost come first, D-159].

- [closed 2026-11-11 by D-127: the six proposals below confirmed as they
  stand] The evaluation's open choices (sec. 2.5, blocks 19-22; raised
  2026-11-11; Claude's proposals in brackets, none decided):
  - Q-E1 n_post, the posterior draws per row [1,000 on every data set, 200
    from each member; the HPD rank's Monte Carlo SD is then at most 0.016
    per row];
  - Q-E2 the rows [D_sim: the whole report split of split seed 1 -- about
    1,000 donors of arm S (5 % of 20,000), two windows each, the rows the
    members' own L_bar is computed on --, the NLL over both windows and the
    calibration tests (HPD coverage, SBC, TARP) over one window per donor,
    since rows that share theta make the uniformity tests anti-conservative
    (the deck pack's G.6 (i): an exact posterior fails G2 in 0.975 of
    replicates with 16 rows per theta, in 0.055 with one); the cohorts:
    every window, the calibration curves' bands from a donor-cluster
    bootstrap (12 windows share each theta*)];
  - Q-E3 theta^dag outside the open box or on its boundary (D_R3+; in
    expectation 1 - 0.895^3 = 0.28 of R3's donors, the free axes being
    uniform under p_R; the bank's theta* gives the count) [the NLL and the
    HPD coverage over the donors whose theta^dag is inside, with their
    count; the marginal ranks, the recovery and the z-scores over every
    donor, where an axis outside the box gives the extreme rank by
    construction];
  - Q-E4 the posterior-predictive check [for each cohort window, 20 draws of
    the ensemble, each simulated by the bench generator with a fresh
    nuisance and no gap (the model's own world), compared with the window
    on the eight fixed window statistics of arm A_ref's summary; the same
    on 324 rows of D_sim as the reference];
  - Q-E5 L-C2ST [sbi 0.27's `LC2ST`, calibrated on 10,000 fresh windows of
    arm S, one per theta, simulated in the evaluation job by the bench
    generator at a seed of its own (the practical guide's case study, Fig.
    10, trained it on 20k additional simulations `[KB, full text]`; D_sim's
    1,000 donors would be few), one ensemble draw per window;
    features theta and the eight fixed window statistics (not an embedding,
    so one classifier design serves both arms); sbi's default MLP; the null
    by 100 permutations; the local statistic at every cohort window and its
    donor mean];
  - Q-E6 where the figures are drawn and kept [one CPU job on davinci after
    the sampling jobs; PNG at 200 dpi and PDF under
    `<runs_root>/evaluation/`; the corner plots' example donors one per
    condition, the donor of median lbar_d under the A1 ensemble, so the
    choice is not tuned to an arm].

- [closed 2026-11-03 by D-121: the searches rank on L, the evaluation reads
  L_pseudo_real] The synthetic experiment's search objective (sec. 2.4;
  raised 2026-11-03, the roadmap's step S1): the search ranks trials on the
  ledger's `nll`, the
  simulated report split's NLL L (`npe_tune_joint.py:623`, `rec["nll"] =
  run["L"]`), which A2's and A5's real-data terms do not improve by
  construction; the pseudo-real endpoint L_pseudo_real is recorded for every
  trial and ranks nothing. Either the searches rank on L, as the pipeline
  does on real data (whose theta is unknown), and the evaluation reads
  L_pseudo_real of every trial and of the best fits; or they rank on
  L_pseudo_real of a selection part of arm R's donors and the evaluation
  reads the rest (an oracle only synthetic data allow). The user's call.
- [closed 2026-11-03 by D-121: two seeds, independent cohorts] R0 and R3
  on one arm-R seed (sec. 2.4; raised 2026-11-03): with one s_R
  the two banks hold the same donors, theta, conditions, nuisance and
  realisation seeds, and the comparison of pi = 0 with pi = 0.3 is paired
  by donor; with two seeds they are independent cohorts. The user's call.

- D-111's residuals (raised 2026-10-29): a driver job PBS cannot start
  while no stage job runs (between two stages, or once every trial of the
  round has ended) leaves the chain dead until a stage job ends or the
  user submits again; a cluster on which no job can start anywhere stops
  everything until it recovers (the replacement loop then tries once an
  hour at most); a broken environment stops the stage after
  `ENV_FAULT_STREAK` faults on `ENV_FAULT_HOSTS` nodes, and the hosts
  D-102 listed during that streak stay listed until the user deletes them.
  `JOB_EXEC_FAULTS` is PBS Pro's documented set, `[proposed]`: which of
  the codes davinci's server reports for which failure is known only for
  -2 (2026-10-16).
  [2026-10-29, the LindDUP chat: a data set whose S2 and S5 still run
  (`projects/linddup/`) meets them outside D-111. Their test tasks, arrays
  and gates run Sbi-extractor's and ANN's job scripts (S2:
  `launch_sim_reextract.sh` with ANN's `submit_mea_array.sh` and
  `run_sim_reextract_gate.pbs`; S5: `launch_bank.sh` with
  `run_bank_task.pbs` and `run_bank_gate.pbs`), which keep their own
  activation (block 7's "Not changed"), so a node without conda fails
  them: S2's test task or gate then stops the stage, as do the bank's test
  task and a bank gate that does not run alone either, while the bank's
  array tasks heal through the rerun rounds (D-094); each STOP ends the
  chain on purpose (`driver/CHAIN_END`), and a submission with `--retry
  <stage>` resumes it. Of the jobs of S2-S6 only the bank's verification
  job runs the watchdog, so the first residual above covers those stages
  almost whole. Whether D-111 is extended to them is the user's call (the
  decisions log, Open calls).] [closed 2026-10-29 (b) by D-113: every job
  of S1, S2 and S5 starts through `jobs/stage_job.sh` (block 7 (g)) and
  runs the watchdog at its end; S2 and S5 submit again, free, the jobs
  that could not start and run free rounds for the tasks that never
  started (blocks 2 and 3, amended 2026-10-29 (b)).]
- S1's residual (D-113, 2026-10-29): S1's jobs start through the job
  protocol, so a node without conda activates the environment from its
  folder; but a well whose job could not start at all (exit 96 or 94, a
  PBS start failure) is named by the aggregation's refusal and stops S1,
  and the launcher refuses a refused root (`holds no
  cohort_manifest.json`), so the remedy is by hand: the root moved aside,
  then `--retry real_extraction`. A resume mode of `launch_stage_d.sh`
  (the missing wells only) would let S1 heal as S2 and S5 do (a
  Sbi-extractor change, D-062). Not needed for LindDUP (S1 skipped).

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

- The tool, steps S1, S3, S4 (sec. 2.7, blocks 40-46; proposed 2026-11-27 from D-172..D-175).
  Open, Claude's proposals in brackets:
  - [Q-T1 to Q-T4 closed 2026-11-27 by D-176: confirmed as proposed, the
    detection in the tool with its parameters declared in the bundle]
  - Q-T1 raw voltage in the `spikeinterface` adapter: how spikes are detected, since the
    networks were trained on the simulator's detections [per channel, negative peaks
    below 5 times the channel's noise level (MAD), 1 ms between two peaks of a channel,
    SpikeInterface's `detect_peaks`; the values written in the bundle's
    `extraction.detection` by the bundle's author; or: spike trains only for now];
  - Q-T2 a recording whose sampling rate or grid differs from the bundle's extraction
    [refused, naming both];
  - Q-T3 the key lists of the manifest and of the card, and which card fields the
    export pre-fills [sec. 2.7 and block 42 as written];
  - Q-T4 the sources imported and not copied until the release (this repository's
    `hpc/` and the Sbi-extractor's checkout) [as sec. 2.7 states].
  - Q-T5 [open, raised 2026-11-28 by the tester's F2; for the sharing step of the
    roadmap] the authenticity of a shared bundle: the file's digests protect against
    damage and against a manifest that disagrees with its members, not against a
    deliberate, consistent edit. Options: the hub or the publication holds the
    bundle's sha256 (the tool then checks a bundle against a digest it is given); a
    signature of the author. Not decided.
