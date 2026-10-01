# Joint DSN + NPE documentation set -- index, status ledger and inventory

| date | change |
|---|---|
| 2026-10-01 | v1. Stage 1 of `claude/JOINT_DOCS_BUILD_PLAN_v1.md` (v1.1): this index; the inventory extractor `tools/inventory_joint_knobs.py` with its smoke test `tools/smoke_test_inventory.py` (31/31, run twice in the sandbox) and its output `tools/inventory.json`; the inventory block of S7 generated from the repository at `834eb41`; the findings of the plan's S4 re-checked against the extraction and extended (F-m .. F-p). Delivered as the first patch of the set (D-035). |
| 2026-10-01 | v1.1. Stage 2: `P0_PARAMETERS_OVERVIEW.md` drafted, its three generated tables rendered by the new `tools/p0_tables.py` (tests T6.1-T6.6 added to the suite: 38/38, run twice `[RAN]`) and checked with `--check-doc`; `--n-post-draws` re-owned to P7 (`stage3c`), `inventory.json` and the S7 block regenerated (`--check-index` OK); findings F-q and F-r added from the P0 reading, F-a's `build_argv` line reference corrected; S9 rewritten for D-052 (one push at the end, no per-turn delivery; first written as "D-047" and renumbered the same turn, that number having been taken by the Giulia chat's entry of about 16:10); status rows updated. |
| 2026-10-01 | v1.2. Stage 3: `E0_READERS_GUIDE_NOTATION.md` drafted -- the master table (179 declared symbols `[RAN]`), the conventions with seventeen overload repairs beyond the plan's two, the glossary by first appearance, the reading map, the prerequisites, the running example at the DUP15HD and bench shapes, the spaces and maps, the analytic/computed pairs; the checker `tools/check_notation.py` with `tools/smoke_test_notation.py` (29/29, run twice `[RAN]`) passes on E0 and on P0. P0 v1.1: bare `T` and `\lambda` aligned with E0 (`T_{gg'}`, `\lambda_{\rm dsn}`), no value changed. S3: the plan's stage status is tracked here between re-issues of the plan. |
| 2026-10-01 | v1.4. Stage 4 continues: `P2_DSN_LOSS_AXES.md` drafted (the seven loss axes and the ten fixed knobs of the metric term; the loss written out from the code as one function of the knobs, eq. (P2.1)-(P2.13), with the miners read from the installed library's source, `pytorch_metric_learning` 1.6.3, and every constant recomputed `[RAN]`; the D-036 differences table; owns F-f, F-h, F-q with P5, and the new F-t to F-x). `E0_READERS_GUIDE_NOTATION.md` v1.2: the DSN-loss group appended (26 rows, 31 symbols). `tools/check_notation.py` extended for two index forms P2 needs (a primed declared index inside a script; a relation or a comma list inside a script) with a recursion guard for a parenthesised list before `\in`; tests T2.10-T2.12 and T4.4-T4.5 added (every written document is now in the suite): 34/34 twice with `--docs-dir ..`, 28/28 fixture-only `[RAN]`. S6: F-f extended (a second stale docstring, `joint_space.py:780-782`); F-t to F-x added. S8: the P2 constants. |
| 2026-10-01 | v1.3. Stage 4 begins: `P1_ENCODER_AXES.md` drafted (the six encoder axes and the twelve fixed `BackboneConfig` knobs, the architecture as a closed-form function of the knobs with the stage layouts, output lengths and parameter counts over the searched grid `[RAN]`, the D-036 differences table, findings F-b, F-c, F-g, F-m and the new F-s). The parameter count reconciled: the cluster's 359708 is the count at `E = 12` (the runtime probe's default), 359450 at the runner's `E = 10` `[RAN]` (S8 corrected). E0 v1.1 appends 29 encoder-architecture symbols; `check_notation.py` reads `\text{}`/environments as prose, set-membership and arithmetic in scripts, and lets a declared base fall through to an indexed match (29/29 twice). Grounding searches for P1 run and reported in its S6 (PubMed: four queries, one PMC full text used; bioRxiv: no keyword search exists, a 30-day slice inspected). |

**What this is.** The index of the joint DSN + NPE documentation set: where
the documents live and in what state (S1-S3), what they were written
against (S4), the conventions and claim tags they share (S5), the findings
about the code that Set P carries (S6), the machine-generated inventory of
every tunable parameter of the joint stack -- the ledger the parameter
documents are written from and checked against (S7) -- the numbers
reproduced in the sandbox (S8), and how each document reaches the
repository (S9). The two sets themselves: **Set P** (parameters,
`P0`-`P7`) explains every tunable parameter of the joint optimisation and
training; **Set E** (environment, `E0`-`E9`) explains the whole joint
environment for learning. Nothing here is a result: no job of `hpc/joint/`
has run on the cluster (`JOINT_DSN_NPE_USAGE_v1.md` S9, `[KB]`).

Claim tags, as in the plan: `[KB]` project knowledge base document;
`[KB-PDF p.n]` a project PDF, full text; `[REPO file:line]` read from the
repository at the freeze commit; `[RAN]` verified by running code in the
sandbox; `[PubMed full text]`; `[PubMed abstract only]`; `[bioRxiv preprint,
abstract only]`; `[textbook, from memory]`; `[reasoning]`; `[STATED]` decided
by the user, with its D-number.

---

## 1. Where the set lives (D-035)

| what | where | note |
|---|---|---|
| the documents, the tools, this index | repository `Simulation-Based-Inference`, `hpc/joint/docs/` | the copy of record; delivered as patches the user applies and pushes |
| this index and `P0_PARAMETERS_OVERVIEW.md` | project knowledge, `claude/joint_docs/` | copies of the repository versions, written in the same turn |
| the inventory of S7, machine-readable | `hpc/joint/docs/tools/inventory.json` | regenerated with the block of S7; stamped with the commit |
| the build plan and its decisions | project knowledge, `claude/JOINT_DOCS_BUILD_PLAN_v1.md`; `claude/SBI_decisions_and_ideas_log.md` D-035..D-038 | not in the repository |

The folder is self-contained: every path in the documents is relative to
`hpc/`, and the tools read the repository's own source, never a copy.

## 2. Reading map

Set P, in the order a reader who wants a knob's meaning should open them;
Set E, in the order the chapters build. `E0` holds the master notation table
and glossary for both sets. Full chapter plans: the build plan, S1.

| id | title | the question it answers |
|---|---|---|
| P0 | Parameters overview | which parameters exist, where each lives, its default and range, and which document explains it |
| P1 | Encoder axes | what `depth_exponent`, `width_multiplier`, `block_family`, `embedding_size`, `head_fusion`, `dropout` and the fixed backbone knobs do to $h_\psi$ |
| P2 | DSN-loss axes | what the composite metric loss and its seven axes do, with the legality projection and the margin convention |
| P3 | Replicate axes | what `rep_on`, `log10_lambda_rep`, `warmup_frac_rep`, `n_posterior_draws` and the loss constants do; the $S_{\rm mc}$ floor and the $\kappa_S$ bias |
| P4 | Flow axes | what `hidden_features`, `num_transforms`, `num_bins` and the z-scoring choices do inside the spline flow |
| P5 | Optimiser and schedule | what `lr`, `one_minus_beta1`, `weight_decay`, the three batch sizes, epochs, steps, clipping, patience and the split do |
| P6 | Search driver | campaigns, the shape anchors, canonicalisation, the ledger, proposals, finalists and per-finalist controls |
| P7 | Upstream and jobs | the bank and bench knobs that shape the problem; the job variables; the Stage 3b/3c flags |
| E0 | Reader's guide and notation | how to read the set; the master notation table and glossary |
| E1 | The problem | from recordings to posteriors over mechanism: objects, spaces, maps, the two data domains, the standing aim |
| E2 | NPE and flows | neural posterior estimation and neural spline flows, built up |
| E3 | The summary network | the DSN, metric learning, the information argument and collapse |
| E4 | Joint objective and loop | eq. (1)-(2), three streams, gradient reach, the explicit loop, the arms |
| E5 | Replicate consistency | from two wells to one number: $T$, $M$, $p_{\rm eff}$, $H_0$, the corrections, the escape routes |
| E6 | The bench | the generator, arms S and R, $\pi$, $\nu$, $\mathcal{G}$, providers, shards |
| E7 | Diagnostics and decision | $L$, $L_0$, $\hat\Delta$, $r_{\rm eff}$, $p_{\rm eff}$, contraction, the bootstrap, Stage 3b/3c, the gates |
| E8 | Hyper-parameter search | GP Bayesian optimisation, constant liar, canonicalisation, nested campaigns, controls, escalation |
| E9 | Operating and status | running it, what has and has not run, the open decisions, D-031/D-033/I-001 |

## 3. Status ledger

`planned`: not written. `drafted`: written, delivered, not yet reviewed by
a fresh agent (plan Stage 6). `reviewed`: both reviews returned empty. The
commit column is the repository commit that carries the file, filled in once
the user's push is seen from the sandbox (`git fetch`), never assumed.
Between re-issues of `claude/JOINT_DOCS_BUILD_PLAN_v1.md` (v1.2 marks
Stages 0-2 done), this ledger is the record of which stage is done; the
plan is re-issued when its content changes, not for status alone.

| id | file | status | repo commit | KB copy | notes |
|---|---|---|---|---|---|
| 00 | `00_INDEX.md` | drafted | pending push (D-052) | `claude/joint_docs/00_INDEX.md` | this file; v1.4 |
| tools | `tools/inventory_joint_knobs.py`, `tools/smoke_test_inventory.py`, `tools/inventory.json`, `tools/p0_tables.py`, `tools/check_notation.py`, `tools/smoke_test_notation.py`, `tools/p2_numbers.py` | drafted | pending push (D-052) | -- | 31/31 twice `[RAN]` at v1 [corrected 2026-10-01: 38/38 twice at v1.1, `p0_tables.py` and tests T6.1-T6.6 added; at v1.2 `check_notation.py` with its own suite, 29/29 twice; at v1.4 the checker extended and the suite at 34/34 twice] |
| P0 | `P0_PARAMETERS_OVERVIEW.md` | drafted | pending push (D-052) | `claude/joint_docs/P0_PARAMETERS_OVERVIEW.md` | v1.1; tables A, F, K generated, `--check-doc` OK x3; notation check OK |
| P1 | `P1_ENCODER_AXES.md` | drafted | pending push (D-052) | -- | v1; notation check OK; owns F-b, F-c, F-g, F-m (encoder rows), F-s |
| P2 | `P2_DSN_LOSS_AXES.md` | drafted | pending push (D-052) | -- | v1; notation check OK; owns F-f, F-h, F-q (with P5), F-t, F-u, F-v, F-w, F-x |
| P3 | `P3_REPLICATE_AXES.md` | planned | -- | -- | next turn |
| P4 | `P4_FLOW_AXES.md` | planned | -- | -- | |
| P5 | `P5_OPTIMISER_AND_SCHEDULE.md` | planned | -- | -- | |
| P6 | `P6_SEARCH_DRIVER.md` | planned | -- | -- | |
| P7 | `P7_UPSTREAM_AND_JOBS.md` | planned | -- | -- | |
| E0 | `E0_READERS_GUIDE_NOTATION.md` | drafted | pending push (D-052) | -- | v1.2; 239 declared symbols `[RAN]`; `--self` check OK |
| E1 | `E1_THE_PROBLEM.md` | planned | -- | -- | |
| E2 | `E2_NPE_AND_FLOWS.md` | planned | -- | -- | |
| E3 | `E3_THE_SUMMARY_NETWORK.md` | planned | -- | -- | |
| E4 | `E4_JOINT_OBJECTIVE_AND_LOOP.md` | planned | -- | -- | |
| E5 | `E5_REPLICATE_CONSISTENCY.md` | planned | -- | -- | |
| E6 | `E6_THE_BENCH.md` | planned | -- | -- | |
| E7 | `E7_DIAGNOSTICS_AND_DECISION.md` | planned | -- | -- | |
| E8 | `E8_HYPERPARAMETER_SEARCH.md` | planned | -- | -- | |
| E9 | `E9_OPERATING_AND_STATUS.md` | planned | -- | -- | |

## 4. Source freeze

- Repository `main` @ `834eb41` (2026-09-30 15:33 +0200), re-fetched
  2026-10-01 from `origin/main`: unchanged `[RAN]`. The joint plan in the
  repository is v0.6.5; "v0.6.6" (the $\kappa_S$ correction named by
  `JOINT_DSN_NPE_USAGE_v1.md` v1.3 and `claude/METRIC_REPLICATE_v1_4.md`) is
  not in the repository, and `joint/stage4/joint_space.py:276-281` still says
  the `n_posterior_draws` axis "controls variance and never bias"
  `[REPO]`. The set states the corrected claim from
  `claude/METRIC_REPLICATE_v1_4.md` S3.13.1 `[KB]` and flags the repository
  text.
- Knowledge base as of 2026-10-01: `JOINT_DSN_NPE_USAGE_v1.md` v1.3;
  `claude/METRIC_REPLICATE_v1_4.md`; `claude/FINITE_DRAW_CORRECTION_v1.md`;
  `claude/NULL_TAIL_AND_PVALUE_v1.md`; the deck pack; `SBI_PIPELINE.md`;
  `EXTRACTOR_USAGE.md`; `HPC_PATHS.md`; the handoffs; the decision log v1.10
  (D-033: Stage C ends at the re-extraction record; D-035..D-038: this set's
  decisions).
- DSN documentation in the repository: `dsn/Documentation/TUNING_1_searched_axes.md`
  revision 3, `TUNING_2_fixed_knobs.md`, `CONFIG_REFERENCE.md`.
- Not available anywhere: `JOINT_DSN_NPE_DEVIATIONS_v1.md`,
  `HANDOFF_D17_option_c.md` (usage v1.3 S10; `find` over the clone,
  2026-10-01). Nothing is cited from them.

## 5. Conventions shared by every document

The build plan's S2 in full. In short: the project's topic-document
skeleton (header with abstract and exclusions; notation table with a
Conventions subsection; glossary; body; summary of results; open points;
references); Set P's fixed per-parameter fields (where it lives with
`[REPO file:line]`; type and domain; every default it has anywhere in the
code; range and prior with provenance; Status in the Provenance model's
sense; plain meaning; analytic effect; legality and interactions; failure
modes and the diagnostic that reveals them; source tag); Set E's plain-first
then analytic sections with one running example at the DUP15HD shapes
($d_\theta = 26$; the search anchors $p = 26$, $E = 12$; $S_{\rm mc} = 128$
`[REPO]`); the Precision model's pre-send check (R1-R8) with its one-line
footer in every document; numbers only with a tag; ASCII and LF only; math
in `$...$`; Mermaid diagrams inline with an ASCII fallback; equation
numbers of the plan and of the metric document kept, never renumbered.

**Symbols in this index** (the master table is E0): $h_\psi$ the encoder,
weights $\psi$; $q_\omega(\theta \mid z)$ the flow, weights $\omega$;
$\theta \in \Theta \subset \mathbb{R}^{d_\theta}$ the inference parameters;
$x$ one IFR window; $z = h_\psi(x) \in S^{E-1}$ the embedding, $E$ its
dimension; $p$ the latent dimension of a bank; $S_{\rm mc}$ posterior draws
per well; $T$ the replicate statistic, $M$ its metric, $p_{\rm eff}$ its
target, $H_0$ the null; $\kappa_S$ the finite-draw inflation factor of $T$;
$L, L_0, \hat\Delta$ held-out NLL, prior floor, information gain;
$r_{\rm eff}$ the effective rank of an embedding cloud; $\pi, \nu,
\mathcal{G}$ the bench's gap severity, nuisance latent and realised graph.

## 6. Findings the set carries (code read at `834eb41`; none is a result)

Each finding has an owner document that states it next to the parameter,
and a resolution: `report` (Set P states it), `patch` (D-038: a fix patch is
prepared, Stage 8 of the plan), `open` (needs a decision, listed in the
decision log's Open calls).

| id | finding | evidence | owner | resolution |
|---|---|---|---|---|
| F-a | `num_bins`: runner default **8**; `build_joint_model` default 10; the space declares it FIXED at 10; `build_argv` passes no `--num-bins`, so a Stage 4 trial trains an 8-bin flow while its ledger spec says 10 | `[REPO]` `joint/stage3/run_joint_arms.py:229`; `joint/stage2/joint_model.py:195-197`; `joint/stage4/joint_space.py:322-324`; `joint/stage4/npe_tune_joint.py:86-106, 206-217` [corrected 2026-10-01: was `198-203`] | P4, P6 | patch (D-038) |
| F-b | `head_pool_ops`: the space declares code **1** (`("mean","max","std")`) fixed; `make_backbone` never sets it, so `BackboneConfig`'s default `("mean",)` (code 0) is built | `[REPO]` `joint/stage4/joint_space.py:322-324`; `joint/stage3/run_joint_arms.py:301-308`; `dsn/backbone.py:76`; `dsn/condition_space.py:113` | P1, P6 | patch (D-038) |
| F-c | `RANGE_PROVENANCE` names `SearchConfig.weight_decay_range` for `weight_decay` and `SearchConfig.dropout_range` for `dropout`; `SearchConfig` has `weight_decay_range = (1e-4, 1e-2)` and no `dropout_range`; the values the space carries, `(1e-5, 1e-2)` and `(0.0, 0.3)`, are `RegularizationConfig`'s | `[REPO]` `joint/stage4/joint_space.py:121-145, 221, 240`; `dsn/config.py:934, 1183-1184` | P1, P5 | report; open whether the two strings get fixed in the D-038 stream |
| F-d | `patience=99` in the runner: early stopping cannot fire in any run shorter than 99 epochs (default `epochs=10`); the best-validation state is still restored. `control_config_from`'s docstring expects early stopping to fire on shuffled data | `[REPO]` `joint/stage3/run_joint_arms.py:519-524`; `joint/stage4/joint_space.py:712-717` | P5, P6 | report; open |
| F-e | the `n_posterior_draws` floor's stated reason (rank; "variance, never bias") in `joint_space.py` and plan S5.1 is superseded by `claude/METRIC_REPLICATE_v1_4.md` S3.13.1 | `[REPO]` `joint/stage4/joint_space.py:276-281`; `[KB]` usage v1.3 S5.3 | P3, E5 | report; open (the $\kappa_S$ repair is an Open call of the log) |
| F-f | `canonicalise_config`'s docstring quotes margin 0.3 (the DSN `TrainConfig`) while `INACTIVE_CANONICAL` pins 0.2 after the `[CORRECTION]` [extended 2026-10-01: `boundary_axes`'s docstring says the same, `:780-782`] | `[REPO]` `joint/stage4/joint_space.py:165-176, 536-538, 780-782` | P2 | report; open |
| F-g | two sources for the DSN-inherited ranges: the `SearchConfig` dataclass defaults, which the space copies (`depth_exponent (3, 6)`, `width_multiplier (1.5, 3.0)`, `lambda_sep (1e-3, 1.0)`, `lr (1e-4, 0.2)`), and the JSON that `TUNING_1` documents (`{2..5}`, `[1.5, 5]`, `[1e-2, 20]`) | `[REPO]` `dsn/config.py:893-934`; `dsn/Documentation/TUNING_1_searched_axes.md` S1 | P1, P2, P5 | report (both stated) |
| F-h | the joint stack's base loss differs from the standalone DSN's: `DSNLossConfig` (margin 0.2, `joint_sep`, `easy_pos_semihard_neg`, `strict_semihard=True`) against the DSN `TrainConfig` (0.3, `triplet`, `hard`, `False`) | S7, "Same knob, different defaults" | P2 | report |
| F-i | configured-by-code, never by flag: `TrainConfig.grad_clip=5.0`, `beta2=0.999`; `ReplicateConsistencyLoss` `jitter=1e-6`, `t_floor=1e-8`, `p_eff_min=1e-3`, `correct_mc=True`; `grouped_split` fractions `(0.7, 0.15, 0.15)`; `stem_width=16`; diagnostics on at most 256 report rows and 64 replicate pairs | `[REPO]` `joint/stage2/joint_train.py:41-44`; `joint/stage2/joint_losses.py:79-83, 264-266`; `joint/stage3/run_joint_arms.py:118, 307, 562, 614` | P3, P5 | report |
| F-j | resolved ranges at the DUP15HD shapes $(p, E, d_\theta) = (26, 12, 26)$: `hidden_features` `[52, 256]`, `n_posterior_draws` `[104, 400]`; on the bench $(10, 10, 10)$: `[32, 256]`, `[40, 400]`; free axes S-A1 12, S-A2 19, S-A5 16, S-A25 23 | `[RAN]` S8 | P4, P6 | report |
| F-k | `README_joint.md`'s test counts (47 / 40+1 / 27 / 28 / 23 / 24 / 14) and usage v1.3 S9's (46/0/1, 69/69, 30, 27, 33, ...) are of different dates | `[REPO]` `joint/README_joint.md`; `[KB]` usage v1.3 S9 | E9 | report (each with its date) |
| F-l | `batch_size_npe` reaches the runner as `--b-sim`; `b_met` (32) and `b_rep` (4) are not searched and not job variables | `[REPO]` `joint/stage4/npe_tune_joint.py:99`; `joint/stage3/jobs/joint_arms.pbs:50-62` | P5 | report |
| F-m | library defaults and runner defaults disagree for the same knobs: `TrainConfig` (20 epochs, 50 steps, lr 5e-4, patience 5) vs the runner (10, 25, 1e-3, 99); `BatchSpec` (512/64/8) vs `--b-sim/--b-met/--b-rep` (128/32/4); `build_joint_model` (64/5/10) vs `--hidden-features/--num-transforms/--num-bins` (48/3/8); `train_joint` `n_posterior_draws=256` vs `--n-posterior-draws 128`; `BackboneConfig` (`depth_exponent` 4, `embedding_size` 16) vs the runner (3, 10). The runner passes every value explicitly, so a Stage 3 run gets the runner's; a direct caller of the library gets the library's | S7, "Same knob, different defaults" `[REPO]` | P1, P4, P5 | report |
| F-n | `stage3c.pbs` defaults `N_POST_DRAWS=128` while `run_stage3c.py --n-post-draws` defaults to 64: the two entry points run different draw counts unless the variable is passed | `[REPO]` `joint/stage3c/jobs/stage3c.pbs:49`; `joint/stage3c/run_stage3c.py:66` | P7 | report; open |
| F-o | `JointSpaceSpec.n_posterior_draws` defaults to `(100, 400)`, below the $4 d_\theta = 104$ floor at $d_\theta = 26$; only `default_joint_space` resolves the floor, so a `JointSpaceSpec()` built directly carries a range the floor forbids | `[REPO]` `joint/stage4/joint_space.py:233, 308-316` | P3, P6 | report |
| F-p | `warmup_frac_rep`: the runner's default is 0.3 (`--warmup-frac-rep`), the space's inactive pin 0.0 and `ReplicateConsistencyLoss` default 0.0: a Stage 3 arm A5 ramps the term over the first 30 % of training, a Stage 4 trial with `rep_on = 0` records 0.0 | `[REPO]` `joint/stage3/run_joint_arms.py:224`; `joint/stage4/joint_space.py:178`; `joint/stage2/joint_losses.py:264` | P3 | report |
| F-q | the encoder-only pre-training of arms `A0`, `A0s` builds `torch.optim.AdamW(backbone.parameters(), lr=lr)`: torch's own defaults for `weight_decay` and `betas`, not the runner's `--weight-decay` and `--one-minus-beta1`, which reach only the joint loop's optimiser; the values of those torch defaults are not read here (torch is not installed in the sandbox) | `[REPO]` `joint/stage3/run_joint_arms.py:154`, `:458-459`; `joint/stage2/joint_train.py:144-147` | P2, P5 | report; open whether the pre-training should take the runner's optimiser flags (candidate for the D-038 stream) |
| F-r | four of the runner's defaults lie outside the ranges the space samples, so a Stage 3 arm at default hyper-parameters is not a point of the Stage 4 space: `--b-sim 128` not in `{256, 512, 1024}`; `--hidden-features 48` below the lower bound 52 at $(26, 12, 26)$ (inside the bench's `[32, 256]`); `--num-transforms 3` below `[4, 12]`; `--weight-decay 0.0` below `[1e-5, 1e-2]` | `[reasoning]` over P0 Table A (`[REPO]` `joint/stage3/run_joint_arms.py:217, 221, 227-228`; `[RAN]` S8) | P0, P4, P5, P6 | report |
| F-s | the joint space's `depth_exponent` range `(3, 6)` is the DSN `SearchConfig` dataclass default, not the DSN's configured `[2, 5]`; `TUNING_1` S3.1 says a bound of 6 doubles the block count to 64 and must not be searched without re-running its budget gate; at that bound the searched encoders have 49 M to 215 M parameters `[RAN]` (P1 S3.2), against the standalone study's documented maximum of 31.6 M over its own ranges; the joint driver has no parameter-count or budget guard and the runner records no count | `[REPO]` `joint/stage4/joint_space.py:221`; `dsn/config.py:893`; `dsn/Documentation/TUNING_1_searched_axes.md` S3.1; `[RAN]` P1 S3.2 | P1, P6 | report; open whether the range is narrowed to `{3, ..., 5}` or a guard added (Open calls) |
| F-t | the class count of the separation target and of the DSN loss's class mask is `len(unique(sim["cls"]))`, the **simulated** bank's, even when the metric stream is the real cohort (`A0`, `A2`, `A3`); equal on the bench by construction, unchecked on a cohort bank, where a real label outside `0..C-1` contributes to no class statistic and no error is raised | `[REPO]` `joint/stage3/run_joint_arms.py:402`; `dsn/dsn_joint_loss.py:112-123, 176-184` | P2, P7 | report; open (take the count from the metric source's labels) |
| F-u | with `strict_semihard` fixed at 1 (`spec.fixed`) and legality-projected, 9 of the DSN's 13 legal (mining, loss, filter) conditions are reachable from the joint space: the four cells of the two easy-positive miners under `joint`/`joint_sep` with the filter off are not; a scope statement of plan S5.1, which does not state the fixed value (`default_joint_space` docstring) | `[RAN]` P2 S3.3; `[REPO]` `joint/stage4/joint_space.py:287-291, 322-324`; `dsn/condition_space.py:222-255` | P2, P6 | report |
| F-v | the composite loss's per-batch counts `n_mined`, `n_strict`, `n_active` and the separation statistics `sep_mean_cos`, `sep_n_classes` are exposed by `DSNLossAdapter.stats()` and never written to the history; a metric term whose strict set is empty every step trains nothing and shows only as `history[].dsn = 0.0` with `rho_grad = NaN` | `[REPO]` `joint/stage2/dsn_loss_adapter.py:109-111`; `joint/stage2/joint_train.py:170-176, 250-252`; `dsn/dsn_joint_loss.py:320-334` | P2, E7 | report; open (log `stats()` per epoch; D-038 stream candidate) |
| F-w | `build_argv` passes `--strict-semihard` from `spec.fixed` but not `--sep-warmup-frac`; the space's fixed 0.0 and the runner's default 0.0 agree, so the ledger and the trained loss coincide by coincidence of defaults (the F-a pattern, one default change away) | `[REPO]` `joint/stage4/npe_tune_joint.py:206-211`; `joint/stage4/joint_space.py:322-324`; `joint/stage3/run_joint_arms.py:256` | P2, P6 | report; open (pass it as `strict_semihard` is passed; D-038 stream candidate) |
| F-x | `dsn/dsn_joint_loss.py` calls `warnings.warn` without importing `warnings`: a `SepWarmup` with a positive fraction and no horizon raises `NameError` instead of warning; unreachable from the joint path, whose adapter refuses that case first | `[REPO]` `dsn/dsn_joint_loss.py:152-157, 638`; `joint/stage2/dsn_loss_adapter.py:169-173` | P2 | report (DSN tree; out of scope for fixes, D-037) |

## 7. Inventory ledger

Generated from the repository by `tools/inventory_joint_knobs.py`, which
reads the joint stack by AST (CLI flags with their subcommand context,
keyword defaults, dataclass fields, ALL_CAPS constants) and the job scripts
by regex (`#PBS` directives, `-v` defaults, env defaults, arrays), and
assigns every row to its owner document by an explicit rule table (plumbing
= paths, selectors, switches and outputs, listed in P0 without a deep dive).
Scope per D-037: `joint/` plus the DSN and NPE-tuner objects the joint
stack reads its fixed values and ranges from.

To regenerate and to check this block against the code:

```
cd hpc/joint/docs/tools
python3 smoke_test_inventory.py --hpc-dir ../../..                 # expect 31/31 passed
python3 inventory_joint_knobs.py --hpc-dir ../../.. --strict --commit <id> --out-json inventory.json --out-md /tmp/inv.md
python3 inventory_joint_knobs.py --hpc-dir ../../.. --check-index ../00_INDEX.md   # expect "OK: index block matches"
```

A `FAIL` from `--check-index` names the first differing line: a parameter
was added, removed or re-defaulted in the code since this block was
generated, and P0 and the owner document are due an update.

<!-- inventory:begin -->
Generated by `tools/inventory_joint_knobs.py` at commit `834eb41`; 546 rows. Do not edit by hand: regenerate, then run `--check-index`. Required (default-less) function parameters are kept in the JSON only.

### CLI flags (164)

| file | name | context | type | default | choices | required | owner |
|---|---|---|---|---|---|---|---|
| joint/stage1/build_latent_bank.py | `--out-dir` |  | str |  |  | yes | plumbing |
| joint/stage1/build_latent_bank.py | `--arm` |  | str | `'S'` | `('S', 'R')` |  | plumbing |
| joint/stage1/build_latent_bank.py | `--shard-index` |  | int | `0` |  |  | plumbing |
| joint/stage1/build_latent_bank.py | `--n-traces` |  | int | `64` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--wells-per-donor` |  | int | `2` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--donors-per-batch` |  | int | `8` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--n-windows` |  | int | `8` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--T-win` |  | float | `60.0` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--fs` |  | float | `50.0` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--n-neurons` |  | int | `100` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--n-latent` |  | int | `6` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--n-label-axes` |  | int | `3` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--n-classes` |  | int | `3` |  |  | P2 |
| joint/stage1/build_latent_bank.py | `--tau-ov` |  | float | `0.1` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--seed` |  | int | `0` |  |  | P5 |
| joint/stage1/build_latent_bank.py | `--pi` |  | float | `0.0` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--gap-modes` |  | str | `'range_shift'` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--n-per-theta` |  | int | `2` |  |  | P7 |
| joint/stage1/build_latent_bank.py | `--provider` |  | str | `'reference'` | `('reference', 'dsn', 'bench')` |  | P7 |
| joint/stage1/build_latent_bank.py | `--dsn-main-dir` |  | str | `None` |  |  | plumbing |
| joint/stage1/build_latent_bank.py | `--max-records` |  | int | `0` |  |  | plumbing |
| joint/stage1/build_latent_bank.py | `--dry-run` |  | flag | `False` |  |  | plumbing |
| joint/stage1/demo_classes_generate.py | `--out` |  | str | `'demo_classes.npz'` |  |  | plumbing |
| joint/stage1/demo_classes_generate.py | `--n-per-class` |  | int | `30` |  |  | P7 |
| joint/stage1/demo_classes_generate.py | `--n-classes` |  | int | `3` |  |  | P2 |
| joint/stage1/demo_classes_generate.py | `--n-windows` |  | int | `4` |  |  | P7 |
| joint/stage1/demo_classes_generate.py | `--T-win` |  | float | `15.0` |  |  | P7 |
| joint/stage1/demo_classes_generate.py | `--fs` |  | float | `50.0` |  |  | P7 |
| joint/stage1/demo_classes_generate.py | `--n-neurons` |  | int | `100` |  |  | P7 |
| joint/stage1/demo_classes_generate.py | `--tau-ov` |  | float | `0.1` |  |  | P7 |
| joint/stage1/demo_classes_generate.py | `--seed` |  | int | `0` |  |  | P5 |
| joint/stage1/demo_classes_generate.py | `--n-background` |  | int | `60` |  |  | P7 |
| joint/stage1/demo_classes_generate.py | `--nuisance` |  | flag | `False` |  |  | plumbing |
| joint/stage1/demo_classes_generate.py | `--dsn-main-dir` |  | str | `None` |  |  | plumbing |
| joint/stage1/demo_classes_plot.py | `--demo` |  | str | `'demo_classes.npz'` |  |  | plumbing |
| joint/stage1/demo_classes_plot.py | `--out-dir` |  | str | `'.'` |  |  | plumbing |
| joint/stage1/demo_classes_plot.py | `--n-show` |  | int | `3` |  |  | P7 |
| joint/stage1/demo_classes_plot.py | `--zoom-s` |  | float | `6.0` |  |  | P7 |
| joint/stage1/demo_classes_plot.py | `--zoom-start` |  | float | `0.0` |  |  | P7 |
| joint/stage1/demo_classes_plot.py | `--embed` |  | str | `'all'` | `('phi', 'x', 'psd', 'all')` |  | plumbing |
| joint/stage1/demo_classes_plot.py | `--perplexity` |  | float | `25.0` |  |  | P7 |
| joint/stage1/demo_classes_plot.py | `--pca-dim` |  | int | `50` |  |  | P7 |
| joint/stage1/demo_classes_plot.py | `--seed` |  | int | `0` |  |  | P5 |
| joint/stage3/probe_dsn_runtime.py | `--dsn-main-dir` |  | str | `None` |  |  | plumbing |
| joint/stage3/probe_dsn_runtime.py | `--embedding-size` |  | int | `12` |  |  | P1 |
| joint/stage3/probe_dsn_runtime.py | `--window` |  | int | `3000` |  |  | P7 |
| joint/stage3/probe_dsn_runtime.py | `--n-classes` |  | int | `3` |  |  | P2 |
| joint/stage3/probe_dsn_runtime.py | `--total-steps` |  | int | `2` |  |  | P2 |
| joint/stage3/probe_dsn_runtime.py | `--mining-strategy` |  | str | `'easy_pos_semihard_neg'` |  |  | P2 |
| joint/stage3/probe_dsn_runtime.py | `--loss-type` |  | str | `'joint_sep'` |  |  | P2 |
| joint/stage3/probe_dsn_runtime.py | `--strict-semihard` |  | int | `1` | `(0, 1)` |  | P2 |
| joint/stage3/probe_dsn_runtime.py | `--margin` |  | float | `0.2` |  |  | P2 |
| joint/stage3/probe_dsn_runtime.py | `--angular-alpha-deg` |  | float | `18.0` |  |  | P2 |
| joint/stage3/probe_dsn_runtime.py | `--lambda-sep` |  | float | `0.1` |  |  | P2 |
| joint/stage3/probe_dsn_runtime.py | `--sep-warmup-frac` |  | float | `0.0` |  |  | P2 |
| joint/stage3/probe_dsn_runtime.py | `--seed` |  | int | `0` |  |  | P5 |
| joint/stage3/report_joint_arms.py | `--runs-dir` |  | str |  |  | yes | plumbing |
| joint/stage3/report_joint_arms.py | `--out` |  | str | `None` |  |  | plumbing |
| joint/stage3/report_joint_arms.py | `--bootstrap-dir` |  | str | `None` |  |  | plumbing |
| joint/stage3/run_joint_arms.py | `--arm` |  | str |  | `ARMS` | yes | plumbing |
| joint/stage3/run_joint_arms.py | `--seed` |  | int | `0` |  |  | P5 |
| joint/stage3/run_joint_arms.py | `--sim-shards` |  | str |  |  | yes | plumbing |
| joint/stage3/run_joint_arms.py | `--real-shards` |  | str | `None` |  |  | plumbing |
| joint/stage3/run_joint_arms.py | `--out-dir` |  | str |  |  | yes | plumbing |
| joint/stage3/run_joint_arms.py | `--epochs` |  | int | `10` |  |  | P5 |
| joint/stage3/run_joint_arms.py | `--steps-per-epoch` |  | int | `25` |  |  | P5 |
| joint/stage3/run_joint_arms.py | `--b-sim` |  | int | `128` |  |  | P5 |
| joint/stage3/run_joint_arms.py | `--b-met` |  | int | `32` |  |  | P2 |
| joint/stage3/run_joint_arms.py | `--b-rep` |  | int | `4` |  |  | P3 |
| joint/stage3/run_joint_arms.py | `--lr` |  | float | `0.001` |  |  | P5 |
| joint/stage3/run_joint_arms.py | `--weight-decay` |  | float | `0.0` |  |  | P5 |
| joint/stage3/run_joint_arms.py | `--lambda-dsn` |  | float | `0.1` |  |  | P2 |
| joint/stage3/run_joint_arms.py | `--lambda-rep` |  | float | `0.05` |  |  | P3 |
| joint/stage3/run_joint_arms.py | `--warmup-frac-rep` |  | float | `0.3` |  |  | P3 |
| joint/stage3/run_joint_arms.py | `--n-posterior-draws` |  | int | `128` |  |  | P3 |
| joint/stage3/run_joint_arms.py | `--encoder-steps` |  | int | `200` |  |  | P2 |
| joint/stage3/run_joint_arms.py | `--hidden-features` |  | int | `48` |  |  | P4 |
| joint/stage3/run_joint_arms.py | `--num-transforms` |  | int | `3` |  |  | P4 |
| joint/stage3/run_joint_arms.py | `--num-bins` |  | int | `8` |  |  | P4 |
| joint/stage3/run_joint_arms.py | `--embedding-size` |  | int | `10` |  |  | P1 |
| joint/stage3/run_joint_arms.py | `--depth-exponent` |  | int | `3` |  |  | P1 |
| joint/stage3/run_joint_arms.py | `--width-multiplier` |  | float | `2.0` |  |  | P1 |
| joint/stage3/run_joint_arms.py | `--block-family` |  | int | `0` | `(0, 1)` |  | P1 |
| joint/stage3/run_joint_arms.py | `--head-fusion` |  | int | `0` | `(0, 1)` |  | P1 |
| joint/stage3/run_joint_arms.py | `--dropout` |  | float | `0.0` |  |  | P1 |
| joint/stage3/run_joint_arms.py | `--loss-type` |  | str | `'joint_sep'` | `('triplet', 'joint', 'joint_sep')` |  | P2 |
| joint/stage3/run_joint_arms.py | `--mining-strategy` |  | str | `'easy_pos_semihard_neg'` | `('hard', 'easy_positive', 'easy_pos_semihard_neg')` |  | P2 |
| joint/stage3/run_joint_arms.py | `--margin` |  | float | `0.2` |  |  | P2 |
| joint/stage3/run_joint_arms.py | `--angular-alpha-deg` |  | float | `18.0` |  |  | P2 |
| joint/stage3/run_joint_arms.py | `--lambda-sep` |  | float | `0.1` |  |  | P2 |
| joint/stage3/run_joint_arms.py | `--sep-warmup-frac` |  | float | `0.0` |  |  | P2 |
| joint/stage3/run_joint_arms.py | `--strict-semihard` |  | int | `1` | `(0, 1)` |  | P2 |
| joint/stage3/run_joint_arms.py | `--one-minus-beta1` |  | float | `0.1` |  |  | P5 |
| joint/stage3/run_joint_arms.py | `--warm-start-ckpt` |  | str | `None` |  |  | plumbing |
| joint/stage3/run_joint_arms.py | `--dsn-main-dir` |  | str | `None` |  |  | plumbing |
| joint/stage3/run_joint_arms.py | `--sbi-hpc-dir` |  | str | `None` |  |  | plumbing |
| joint/stage3/run_joint_arms.py | `--dry-run` |  | flag | `False` |  |  | plumbing |
| joint/stage3b/run_stage3b.py | `--runs-dir` |  | str |  |  | yes | plumbing |
| joint/stage3b/run_stage3b.py | `--sim-shards` |  | str | `None` |  |  | plumbing |
| joint/stage3b/run_stage3b.py | `--real-shards` |  | str | `None` |  |  | plumbing |
| joint/stage3b/run_stage3b.py | `--probe-ckpt` |  | str | `None` |  |  | plumbing |
| joint/stage3b/run_stage3b.py | `--max-probe-rows` |  | int | `512` |  |  | P7 |
| joint/stage3b/run_stage3b.py | `--bootstrap-dir` |  | str | `None` |  |  | plumbing |
| joint/stage3b/run_stage3b.py | `--dsn-main-dir` |  | str | `None` |  |  | plumbing |
| joint/stage3b/run_stage3b.py | `--out` |  | str | `None` |  |  | plumbing |
| joint/stage3b/run_stage3b.py | `--out-json` |  | str | `None` |  |  | plumbing |
| joint/stage3b/run_stage3b.py | `--dry-run` |  | flag | `False` |  |  | plumbing |
| joint/stage3c/run_stage3c.py | `--ckpt` |  | str |  |  | yes | plumbing |
| joint/stage3c/run_stage3c.py | `--sim-shards` |  | str |  |  | yes | plumbing |
| joint/stage3c/run_stage3c.py | `--out-dir` |  | str |  |  | yes | plumbing |
| joint/stage3c/run_stage3c.py | `--n-floor-draws` |  | int | `48` |  |  | P7 |
| joint/stage3c/run_stage3c.py | `--n-post-draws` |  | int | `64` |  |  | P7 |
| joint/stage3c/run_stage3c.py | `--kernel-axes` |  | str | `''` |  |  | P7 |
| joint/stage3c/run_stage3c.py | `--spread-axes` |  | str | `''` |  |  | P7 |
| joint/stage3c/run_stage3c.py | `--fd-step` |  | float | `0.02` |  |  | P7 |
| joint/stage3c/run_stage3c.py | `--fd-seeds` |  | int | `3` |  |  | P7 |
| joint/stage3c/run_stage3c.py | `--allow-real` |  | flag | `False` |  |  | plumbing |
| joint/stage3c/run_stage3c.py | `--validation` |  | str | `None` |  |  | plumbing |
| joint/stage3c/run_stage3c.py | `--dsn-main-dir` |  | str | `None` |  |  | plumbing |
| joint/stage3c/run_stage3c.py | `--seed` |  | int | `0` |  |  | P5 |
| joint/stage3c/run_stage3c.py | `--dry-run` |  | flag | `False` |  |  | plumbing |
| joint/stage4/joint_space.py | `--campaign` |  | str | `None` |  |  | plumbing |
| joint/stage4/joint_space.py | `--p` |  | int | `26` |  |  | P6 |
| joint/stage4/joint_space.py | `--embedding-dim` |  | int | `12` |  |  | P6 |
| joint/stage4/joint_space.py | `--d-theta` |  | int | `26` |  |  | P6 |
| joint/stage4/joint_space.py | `--n-train` |  | int | `None` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--campaign` | common | str | `'S-A1'` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--results-dir` | common | str | `'tune_joint'` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--p` | common | int | `26` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--embedding-dim` | common | int | `12` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--d-theta` | common | int | `26` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--n-train` | common | int | `None` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--split-hash` | common | str | `''` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--contract-digest` | common | str | `''` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--seed` | common | int | `0` |  |  | P5 |
| joint/stage4/npe_tune_joint.py | `--sim-shards` | common[need_shards] | str |  |  | yes | plumbing |
| joint/stage4/npe_tune_joint.py | `--real-shards` | common[need_shards] | str | `None` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--out-dir` | common[need_shards] | str | `'runs_joint'` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--epochs` | common[need_shards] | int | `10` |  |  | P5 |
| joint/stage4/npe_tune_joint.py | `--steps-per-epoch` | common[need_shards] | int | `25` |  |  | P5 |
| joint/stage4/npe_tune_joint.py | `--runner` | common[need_shards] | str | `DEFAULT_RUNNER` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--dsn-main-dir` | common[need_shards] | str | `None` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--sbi-hpc-dir` | common[need_shards] | str | `None` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--dry-run` | common[need_shards] | flag | `False` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--n-points` | propose | int | `8` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--n-initial-points` | propose | int | `12` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--sigma-seed` | propose | float | `None` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--config-json` | argv | str | `None` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--pending-id` | argv | str | `None` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--config-json` | evaluate | str | `None` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--pending-id` | evaluate | str | `None` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--tag` | evaluate | str | `''` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--top` | status | int | `10` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--sigma-seed` | status | float | `None` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--top-k` | finalists | int | `3` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--rank-split` | finalists | str | `'sel'` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--gate-split` | finalists | str | `'gate'` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--plan` | controls | flag | `False` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--score` | controls | flag | `False` |  |  | plumbing |
| joint/stage4/npe_tune_joint.py | `--n-control` | controls | int | `5` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--n-seeds` | controls | int | `1` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--alpha` | controls | float | `0.05` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--floor` | controls | float | `0.0` |  |  | P6 |
| joint/stage4/npe_tune_joint.py | `--delta-min-provisional` | controls | float | `float('nan')` |  |  | P6 |

### Function and `__init__` keyword defaults (104)

| file | name | default | owner |
|---|---|---|---|
| joint/stage1/latent_gap.py | `GapSpec.__init__.contam_frac_max` | `0.2` | P7 |
| joint/stage1/latent_gap.py | `GapSpec.__init__.drift_amp_max` | `0.5` | P7 |
| joint/stage1/latent_gap.py | `GapSpec.__init__.drift_period_s` | `120.0` | P7 |
| joint/stage1/latent_gap.py | `GapSpec.__init__.modes` | `('range_shift',)` | P7 |
| joint/stage1/latent_gap.py | `GapSpec.__init__.pi` | `0.0` | P7 |
| joint/stage1/latent_gap.py | `GapSpec.__init__.shift_max` | `0.35` | P7 |
| joint/stage1/latent_nuisance.py | `NuisanceSpec.__init__.drift_period_s` | `600.0` | P7 |
| joint/stage1/latent_nuisance.py | `NuisanceSpec.__init__.dropout_logit0` | `-4.0` | P7 |
| joint/stage1/latent_nuisance.py | `NuisanceSpec.__init__.kappa` | `0.5` | P7 |
| joint/stage1/latent_nuisance.py | `NuisanceSpec.__init__.n_electrodes` | `9` | P7 |
| joint/stage1/latent_nuisance.py | `NuisanceSpec.__init__.scales` | `None` | P7 |
| joint/stage1/latent_realisation.py | `RealisationSpec.__init__.n_per_theta` | `2` | P7 |
| joint/stage1/latent_realisation.py | `RealisationSpec.__init__.share_within` | `('subregion',)` | P7 |
| joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.T_win` | `60.0` | P7 |
| joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.class_centres` | `None` | P7 |
| joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.fs` | `50.0` | P7 |
| joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.label_idx` | `(0, 1, 2)` | P7 |
| joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.n_latent` | `6` | P7 |
| joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.n_neurons` | `100` | P7 |
| joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.n_windows_per_trace` | `8` | P7 |
| joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.seed` | `0` | P7 |
| joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.tau_ov` | `0.1` | P7 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.angular_alpha_deg` | `18.0` | P2 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.lambda_sep` | `0.1` | P2 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.loss_type` | `'joint_sep'` | P2 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.margin` | `0.2` | P2 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.mining_strategy` | `'easy_pos_semihard_neg'` | P2 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.sep_centre_means` | `None` | P2 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.sep_gate_threshold` | `None` | P2 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.sep_warmup_frac` | `0.0` | P2 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.strict_semihard` | `True` | P2 |
| joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.swap` | `True` | P2 |
| joint/stage2/dsn_loss_adapter.py | `build_dsn_loss.cfg` | `None` | plumbing |
| joint/stage2/dsn_loss_adapter.py | `build_dsn_loss.dsn_main_dir` | `None` | plumbing |
| joint/stage2/dsn_loss_adapter.py | `build_dsn_loss.total_steps` | `None` | plumbing |
| joint/stage2/joint_batches.py | `BatchSpec.__init__.b_met` | `64` | P2 |
| joint/stage2/joint_batches.py | `BatchSpec.__init__.b_rep` | `8` | P3 |
| joint/stage2/joint_batches.py | `BatchSpec.__init__.b_sim` | `512` | P5 |
| joint/stage2/joint_batches.py | `ThreeStreamBatcher.__init__.met_cls` | `None` | plumbing |
| joint/stage2/joint_batches.py | `ThreeStreamBatcher.__init__.met_x` | `None` | plumbing |
| joint/stage2/joint_batches.py | `ThreeStreamBatcher.__init__.real_cls` | `None` | plumbing |
| joint/stage2/joint_batches.py | `ThreeStreamBatcher.__init__.real_donor` | `None` | plumbing |
| joint/stage2/joint_batches.py | `ThreeStreamBatcher.__init__.real_x` | `None` | plumbing |
| joint/stage2/joint_batches.py | `ThreeStreamBatcher.__init__.seed` | `0` | plumbing |
| joint/stage2/joint_batches.py | `ThreeStreamBatcher.__init__.spec` | `None` | plumbing |
| joint/stage2/joint_batches.py | `ThreeStreamBatcher.__init__.surrogate` | `None` | plumbing |
| joint/stage2/joint_losses.py | `box_prior_covariance.device` | `None` | plumbing |
| joint/stage2/joint_losses.py | `box_prior_covariance.dtype` | `torch.float64` | P3 |
| joint/stage2/joint_losses.py | `replicate_statistic.correct_mc` | `False` | P3 |
| joint/stage2/joint_losses.py | `replicate_statistic.detach_metric` | `True` | P3 |
| joint/stage2/joint_losses.py | `replicate_statistic.jitter` | `DEFAULT_JITTER` | P3 |
| joint/stage2/joint_losses.py | `replicate_statistic.n_draws` | `None` | P3 |
| joint/stage2/joint_losses.py | `replicate_loss.t_floor` | `DEFAULT_T_FLOOR` | P3 |
| joint/stage2/joint_losses.py | `ReplicateConsistencyLoss.__init__.correct_mc` | `True` | P3 |
| joint/stage2/joint_losses.py | `ReplicateConsistencyLoss.__init__.jitter` | `DEFAULT_JITTER` | P3 |
| joint/stage2/joint_losses.py | `ReplicateConsistencyLoss.__init__.p_eff_min` | `DEFAULT_P_EFF_MIN` | P3 |
| joint/stage2/joint_losses.py | `ReplicateConsistencyLoss.__init__.t_floor` | `DEFAULT_T_FLOOR` | P3 |
| joint/stage2/joint_losses.py | `ReplicateConsistencyLoss.__init__.warmup` | `0.0` | P3 |
| joint/stage2/joint_model.py | `build_joint_model.hidden_features` | `64` | P4 |
| joint/stage2/joint_model.py | `build_joint_model.meta` | `None` | plumbing |
| joint/stage2/joint_model.py | `build_joint_model.num_bins` | `10` | P4 |
| joint/stage2/joint_model.py | `build_joint_model.num_transforms` | `5` | P4 |
| joint/stage2/joint_model.py | `build_joint_model.z_score_theta` | `'transform_to_unconstrained'` | P4 |
| joint/stage2/joint_model.py | `build_joint_model.z_score_x` | `'none'` | P4 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.beta1` | `0.9` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.beta2` | `0.999` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.epochs` | `20` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.grad_clip` | `5.0` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.lambda_dsn` | `0.0` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.lambda_rep` | `0.0` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.lr` | `0.0005` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.patience` | `5` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.rho_grad_probe` | `True` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.steps_per_epoch` | `50` | P5 |
| joint/stage2/joint_train.py | `TrainConfig.__init__.weight_decay` | `0.0` | P5 |
| joint/stage2/joint_train.py | `evaluate_npe.batch_size` | `512` | plumbing |
| joint/stage2/joint_train.py | `train_joint.dsn_loss_fn` | `None` | plumbing |
| joint/stage2/joint_train.py | `train_joint.log_fn` | `print` | plumbing |
| joint/stage2/joint_train.py | `train_joint.n_posterior_draws` | `256` | P3 |
| joint/stage2/joint_train.py | `train_joint.rep_criterion` | `None` | plumbing |
| joint/stage2/joint_train.py | `train_joint.seed` | `0` | plumbing |
| joint/stage2/joint_train.py | `train_joint.val_theta` | `None` | plumbing |
| joint/stage2/joint_train.py | `train_joint.val_x` | `None` | plumbing |
| joint/stage3/run_joint_arms.py | `grouped_split.fracs` | `(0.7, 0.15, 0.15)` | P5 |
| joint/stage3/run_joint_arms.py | `grouped_split.seed` | `0` | plumbing |
| joint/stage3/run_joint_arms.py | `train_encoder_only.log_fn` | `None` | plumbing |
| joint/stage3/run_joint_arms.py | `train_encoder_only.seed` | `0` | plumbing |
| joint/stage3/run_joint_arms.py | `make_backbone.encoder` | `None` | plumbing |
| joint/stage4/joint_space.py | `default_joint_space.n_posterior_draws_max` | `400` | P6 |
| joint/stage4/joint_space.py | `default_joint_space.n_train` | `None` | plumbing |
| joint/stage4/joint_space.py | `default_joint_space.strict_semihard` | `1` | P6 |
| joint/stage4/joint_space.py | `boundary_axes.campaign_name` | `None` | plumbing |
| joint/stage4/joint_space.py | `boundary_axes.rel_tol` | `1e-06` | P6 |
| joint/stage4/npe_tune_joint.py | `build_argv.dry_run` | `False` | plumbing |
| joint/stage4/npe_tune_joint.py | `build_argv.dsn_main_dir` | `None` | plumbing |
| joint/stage4/npe_tune_joint.py | `build_argv.epochs` | `10` | P6 |
| joint/stage4/npe_tune_joint.py | `build_argv.extra` | `None` | plumbing |
| joint/stage4/npe_tune_joint.py | `build_argv.python` | `None` | plumbing |
| joint/stage4/npe_tune_joint.py | `build_argv.real_shards` | `None` | plumbing |
| joint/stage4/npe_tune_joint.py | `build_argv.runner` | `DEFAULT_RUNNER` | plumbing |
| joint/stage4/npe_tune_joint.py | `build_argv.sbi_hpc_dir` | `None` | plumbing |
| joint/stage4/npe_tune_joint.py | `build_argv.spec_fixed` | `None` | plumbing |
| joint/stage4/npe_tune_joint.py | `build_argv.steps_per_epoch` | `25` | P6 |
| npe_tune_search.py | `default_space.n_train` | `None` | plumbing |

### Dataclass fields (82)

| file | name | annotation | default | owner |
|---|---|---|---|---|
| dsn/backbone.py | `BackboneConfig.depth_exponent` | `int` | `4` | P1 |
| dsn/backbone.py | `BackboneConfig.width_multiplier` | `float` | `2.0` | P1 |
| dsn/backbone.py | `BackboneConfig.stem_width` | `int` | `16` | P1 |
| dsn/backbone.py | `BackboneConfig.in_channels` | `int` | `1` | P1 |
| dsn/backbone.py | `BackboneConfig.block_family` | `int` | `0` | P1 |
| dsn/backbone.py | `BackboneConfig.group_width` | `int` | `16` | P1 |
| dsn/backbone.py | `BackboneConfig.embedding_size` | `int` | `16` | P1 |
| dsn/backbone.py | `BackboneConfig.l2_normalize` | `bool` | `True` | P1 |
| dsn/backbone.py | `BackboneConfig.head_fusion` | `bool` | `False` | P1 |
| dsn/backbone.py | `BackboneConfig.head_pool_ops` | `Tuple[str, ...]` | `('mean',)` | P1 |
| dsn/backbone.py | `BackboneConfig.head_prenorm` | `bool` | `True` | P1 |
| dsn/backbone.py | `BackboneConfig.norm_target_cpg` | `int` | `16` | P1 |
| dsn/backbone.py | `BackboneConfig.norm_g_max` | `int` | `32` | P1 |
| dsn/backbone.py | `BackboneConfig.stem_kernel` | `int` | `5` | P1 |
| dsn/backbone.py | `BackboneConfig.stem_stride` | `int` | `4` | P1 |
| dsn/backbone.py | `BackboneConfig.stage_kernel` | `int` | `3` | P1 |
| dsn/backbone.py | `BackboneConfig.downsampling_rate` | `int` | `2` | P1 |
| dsn/backbone.py | `BackboneConfig.dropout` | `float` | `0.0` | P1 |
| dsn/config.py | `TrainConfig.loss_type` | `str` | `'triplet'` | P2 |
| dsn/config.py | `TrainConfig.margin` | `float` | `0.3` | P2 |
| dsn/config.py | `TrainConfig.swap` | `bool` | `True` | P2 |
| dsn/config.py | `TrainConfig.mining_strategy` | `str` | `'hard'` | P2 |
| dsn/config.py | `TrainConfig.angular_alpha_deg` | `float` | `18.0` | P2 |
| dsn/config.py | `TrainConfig.strict_semihard` | `bool` | `False` | P2 |
| dsn/config.py | `TrainConfig.lambda_sep` | `float` | `0.1` | P2 |
| dsn/config.py | `TrainConfig.sep_warmup_frac` | `float` | `0.0` | P2 |
| dsn/config.py | `TrainConfig.lr` | `float` | `0.0003` | P5 |
| dsn/config.py | `TrainConfig.weight_decay` | `float` | `0.0001` | P5 |
| dsn/config.py | `TrainConfig.max_epochs` | `int` | `100` | P5 |
| dsn/config.py | `TrainConfig.patience` | `int` | `10` | P5 |
| dsn/config.py | `SearchConfig.depth_exponent_range` | `Tuple[int, int]` | `(3, 6)` | P1 |
| dsn/config.py | `SearchConfig.width_multiplier_range` | `Tuple[float, float]` | `(1.5, 3.0)` | P1 |
| dsn/config.py | `SearchConfig.block_family_choices` | `Tuple[int, ...]` | `(0, 1)` | P1 |
| dsn/config.py | `SearchConfig.embedding_size_range` | `Tuple[int, int]` | `(8, 16)` | P1 |
| dsn/config.py | `SearchConfig.margin_range` | `Tuple[float, float]` | `(0.1, 1.0)` | P2 |
| dsn/config.py | `SearchConfig.angular_alpha_deg_range` | `Tuple[float, float]` | `(2.0, 20.0)` | P2 |
| dsn/config.py | `SearchConfig.lambda_sep_range` | `Tuple[float, float]` | `(0.001, 1.0)` | P2 |
| dsn/config.py | `SearchConfig.sep_warmup_frac_range` | `Tuple[float, float]` | `(0.0, 0.5)` | P2 |
| dsn/config.py | `SearchConfig.lr_range` | `Tuple[float, float]` | `(0.0001, 0.2)` | P5 |
| dsn/config.py | `SearchConfig.one_minus_beta1_range` | `Tuple[float, float]` | `(0.01, 0.1)` | P5 |
| dsn/config.py | `SearchConfig.one_minus_beta2_range` | `Tuple[float, float]` | `(0.0001, 0.01)` | P5 |
| dsn/config.py | `SearchConfig.weight_decay_range` | `Tuple[float, float]` | `(0.0001, 0.01)` | P5 |
| dsn/config.py | `SearchConfig.mining_strategy_choices` | `Tuple[str, ...]` | `('hard', 'easy_positive', 'easy_pos_semihard_neg')` | P2 |
| dsn/config.py | `SearchConfig.loss_type_choices` | `Tuple[str, ...]` | `('triplet', 'joint', 'joint_sep')` | P2 |
| dsn/config.py | `SearchConfig.strict_semihard_choices` | `Tuple[int, ...]` | `(0, 1)` | P2 |
| dsn/config.py | `SearchConfig.head_fusion_choices` | `Tuple[int, ...]` | `(0, 1)` | P1 |
| dsn/config.py | `SearchConfig.head_pool_ops_choices` | `Tuple[int, ...]` | `(0, 1)` | P1 |
| dsn/config.py | `SearchConfig.sep_centre_means_choices` | `Tuple[int, ...]` | `(0, 1)` | P2 |
| dsn/config.py | `RegularizationConfig.dropout_range` | `Tuple[float, float]` | `(0.0, 0.3)` | P1 |
| dsn/config.py | `RegularizationConfig.weight_decay_range` | `Tuple[float, float]` | `(1e-05, 0.01)` | P5 |
| joint/stage4/joint_space.py | `JointSpaceSpec.depth_exponent` | `Tuple[int, int]` | `(3, 6)` | P1 |
| joint/stage4/joint_space.py | `JointSpaceSpec.width_multiplier` | `Tuple[float, float]` | `(1.5, 3.0)` | P1 |
| joint/stage4/joint_space.py | `JointSpaceSpec.block_family` | `Tuple[int, ...]` | `(0, 1)` | P1 |
| joint/stage4/joint_space.py | `JointSpaceSpec.embedding_size` | `Tuple[int, int]` | `(8, 16)` | P1 |
| joint/stage4/joint_space.py | `JointSpaceSpec.head_fusion` | `Tuple[int, ...]` | `(0, 1)` | P1 |
| joint/stage4/joint_space.py | `JointSpaceSpec.dropout` | `Tuple[float, float]` | `(0.0, 0.3)` | P1 |
| joint/stage4/joint_space.py | `JointSpaceSpec.log10_lambda_dsn` | `Tuple[float, float]` | `(-3.0, 1.0)` | P2 |
| joint/stage4/joint_space.py | `JointSpaceSpec.loss_type` | `Tuple[str, ...]` | `('triplet', 'joint', 'joint_sep')` | P2 |
| joint/stage4/joint_space.py | `JointSpaceSpec.mining_strategy` | `Tuple[str, ...]` | `('hard', 'easy_positive', 'easy_pos_semihard_neg')` | P2 |
| joint/stage4/joint_space.py | `JointSpaceSpec.margin` | `Tuple[float, float]` | `(0.1, 1.0)` | P2 |
| joint/stage4/joint_space.py | `JointSpaceSpec.angular_alpha_deg` | `Tuple[float, float]` | `(2.0, 20.0)` | P2 |
| joint/stage4/joint_space.py | `JointSpaceSpec.lambda_sep` | `Tuple[float, float]` | `(0.001, 1.0)` | P2 |
| joint/stage4/joint_space.py | `JointSpaceSpec.log10_lambda_rep` | `Tuple[float, float]` | `(-3.0, 1.0)` | P3 |
| joint/stage4/joint_space.py | `JointSpaceSpec.warmup_frac_rep` | `Tuple[float, float]` | `(0.0, 0.5)` | P3 |
| joint/stage4/joint_space.py | `JointSpaceSpec.n_posterior_draws` | `Tuple[int, int]` | `(100, 400)` | P3 |
| joint/stage4/joint_space.py | `JointSpaceSpec.hidden_features` | `Tuple[int, int]` | `(64, 256)` | P4 |
| joint/stage4/joint_space.py | `JointSpaceSpec.num_transforms` | `Tuple[int, int]` | `(4, 12)` | P4 |
| joint/stage4/joint_space.py | `JointSpaceSpec.lr` | `Tuple[float, float]` | `(0.0001, 0.002)` | P5 |
| joint/stage4/joint_space.py | `JointSpaceSpec.one_minus_beta1` | `Tuple[float, float]` | `(0.01, 0.1)` | P5 |
| joint/stage4/joint_space.py | `JointSpaceSpec.weight_decay` | `Tuple[float, float]` | `(1e-05, 0.01)` | P5 |
| joint/stage4/joint_space.py | `JointSpaceSpec.batch_size_npe` | `Tuple[int, ...]` | `(256, 512, 1024)` | P5 |
| joint/stage4/joint_space.py | `JointSpaceSpec.fixed` | `Dict[str, Any]` | `field(default_factory=dict)` | P6 |
| joint/stage4/joint_space.py | `JointSpaceSpec.anchored_to` | `Dict[str, Any]` | `field(default_factory=dict)` | P6 |
| joint/stage4/joint_space.py | `Campaign.name` | `str` |  | P6 |
| joint/stage4/joint_space.py | `Campaign.pinned` | `Dict[str, Any]` |  | P6 |
| joint/stage4/joint_space.py | `Campaign.note` | `str` | `''` | P6 |
| npe_tune_search.py | `SpaceSpec.hidden_features` | `Tuple[int, int]` | `(64, 256)` | P4 |
| npe_tune_search.py | `SpaceSpec.num_transforms` | `Tuple[int, int]` | `(4, 12)` | P4 |
| npe_tune_search.py | `SpaceSpec.num_bins` | `Tuple[int, int]` | `(6, 16)` | P4 |
| npe_tune_search.py | `SpaceSpec.learning_rate` | `Tuple[float, float]` | `(0.0001, 0.002)` | P5 |
| npe_tune_search.py | `SpaceSpec.batch_sizes` | `Tuple[int, ...]` | `(256, 512, 1024)` | P5 |
| npe_tune_search.py | `SpaceSpec.anchored_to` | `Dict[str, Any]` | `field(default_factory=dict)` | P6 |

### Module and class constants (46)

| file | name | default | owner |
|---|---|---|---|
| dsn/condition_space.py | `MINING_STRATEGIES` | `('hard', 'easy_positive', 'easy_pos_semihard_neg')` | P2 |
| dsn/condition_space.py | `LOSS_TYPES` | `('triplet', 'joint', 'joint_sep')` | P2 |
| dsn/condition_space.py | `HEAD_POOL_OPS_LEVELS` | `(('mean',), ('mean', 'max', 'std'))` | P1 |
| dsn/condition_space.py | `LOSS_HP_SUPERSET` | `('margin', 'angular_alpha_deg', 'lambda_sep', 'sep_warmup_frac')` | P2 |
| dsn/condition_space.py | `_ACTIVE` | `{'triplet': ('margin',), 'joint': ('angular_alpha_deg',), 'joint_sep'...` | P2 |
| dsn/condition_space.py | `_MINING_TAG` | `{'hard': 'h', 'easy_positive': 'ep', 'easy_pos_semihard_neg': 'epsh'}` | P2 |
| dsn/condition_space.py | `_LOSS_TAG` | `{'triplet': 'trip', 'joint': 'joint', 'joint_sep': 'jsep'}` | P2 |
| joint/stage1/bench_burst_provider.py | `_HERE` | `os.path.dirname(os.path.abspath(__file__))` | plumbing |
| joint/stage1/bench_burst_provider.py | `BENCH_AXES` | `(('burst_rate', 0.1, 0.4), ('irregularity', 0.3, 0.9), ('ibi_cv', 0.2...` | P7 |
| joint/stage1/bench_burst_provider.py | `BENCH_LABEL_IDX` | `(0, 1, 2, 3, 4, 5, 6)` | P7 |
| joint/stage1/bench_burst_provider.py | `BENCH_FREE_IDX` | `(7, 8, 9)` | P7 |
| joint/stage1/bench_burst_provider.py | `BenchBurstProvider.PAD_BINS` | `2` | P7 |
| joint/stage1/latent_gap.py | `GAP_MODES` | `('range_shift', 'drift', 'contamination')` | P7 |
| joint/stage1/latent_nuisance.py | `NU_COMPONENTS` | `('log_gain', 'baseline', 'dthr', 'dropout_logit', 'drift_amp')` | P7 |
| joint/stage1/latent_nuisance.py | `N_COMPONENTS` | `len(NU_COMPONENTS)` | P7 |
| joint/stage1/latent_nuisance.py | `NU_LEVELS` | `('batch', 'donor', 'well')` | P7 |
| joint/stage2/joint_losses.py | `DEFAULT_JITTER` | `1e-06` | P3 |
| joint/stage2/joint_losses.py | `DEFAULT_T_FLOOR` | `1e-08` | P3 |
| joint/stage2/joint_losses.py | `DEFAULT_P_EFF_MIN` | `0.001` | P3 |
| joint/stage3/run_joint_arms.py | `_HERE` | `os.path.dirname(os.path.abspath(__file__))` | plumbing |
| joint/stage3/run_joint_arms.py | `ARMS` | `('A0', 'A0s', 'A1', 'A2', 'A2s', 'A3', 'A5', 'A_ref', 'shuffled')` | P6 |
| joint/stage3/run_joint_arms.py | `FixedStatsSummary.N_STATS` | `8` | P5 |
| joint/stage3c/run_stage3c.py | `_HERE` | `os.path.dirname(os.path.abspath(__file__))` | plumbing |
| joint/stage4/joint_space.py | `_JOINT_DIR` | `os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__...` | plumbing |
| joint/stage4/joint_space.py | `JOINT_KNOB_ORDER` | `('depth_exponent', 'width_multiplier', 'block_family', 'embedding_siz...` | P6 |
| joint/stage4/joint_space.py | `BLOCKS` | `{'encoder': ('depth_exponent', 'width_multiplier', 'block_family', 'e...` | P6 |
| joint/stage4/joint_space.py | `RANGE_PROVENANCE` | `{'depth_exponent': 'DSN config.SearchConfig.depth_exponent_range', 'w...` | P6 |
| joint/stage4/joint_space.py | `INACTIVE_CANONICAL` | `{'log10_lambda_dsn': 0.0, 'loss_type': 'triplet', 'mining_strategy': ...` | P6 |
| joint/stage4/joint_space.py | `_INT_AXES` | `frozenset(('depth_exponent', 'block_family', 'embedding_size', 'head_...` | P6 |
| joint/stage4/joint_space.py | `_STR_AXES` | `frozenset(('loss_type', 'mining_strategy'))` | P6 |
| joint/stage4/joint_space.py | `_DSN_AXES_OFF` | `{'dsn_on': 0, 'log10_lambda_dsn': INACTIVE_CANONICAL['log10_lambda_ds...` | P6 |
| joint/stage4/joint_space.py | `_REP_AXES_OFF` | `{'rep_on': 0, 'log10_lambda_rep': INACTIVE_CANONICAL['log10_lambda_re...` | P6 |
| joint/stage4/joint_space.py | `CAMPAIGNS` | `{'S-A1': Campaign(name='S-A1', pinned=dict(**_DSN_AXES_OFF, **_REP_AX...` | P6 |
| joint/stage4/joint_space.py | `SHUFFLE_FIELDS` | `('shuffle_pairs', 'shuffle_seed')` | P6 |
| joint/stage4/joint_space.py | `_NO_BOUNDARY` | `frozenset(('block_family', 'head_fusion', 'dsn_on', 'rep_on', 'loss_t...` | P6 |
| joint/stage4/npe_tune_joint.py | `_HERE` | `os.path.dirname(os.path.abspath(__file__))` | plumbing |
| joint/stage4/npe_tune_joint.py | `PENDING_FILE` | `'pending'` | plumbing |
| joint/stage4/npe_tune_joint.py | `LEDGER_FILE` | `'trials'` | plumbing |
| joint/stage4/npe_tune_joint.py | `CONTROLS_FILE` | `'controls.json'` | plumbing |
| joint/stage4/npe_tune_joint.py | `FINALISTS_FILE` | `'finalists.json'` | plumbing |
| joint/stage4/npe_tune_joint.py | `DEFAULT_RUNNER` | `os.path.abspath(os.path.join(_HERE, '..', 'stage3', 'run_joint_arms.p...` | plumbing |
| joint/stage4/npe_tune_joint.py | `AXIS_TO_FLAG` | `{'depth_exponent': '--depth-exponent', 'width_multiplier': '--width-m...` | P6 |
| joint/stage4/npe_tune_joint.py | `DERIVED` | `{'dsn_on': 'gates --lambda-dsn: 0 when off, 10 ** log10_lambda_dsn wh...` | P6 |
| joint/stage4/npe_tune_joint.py | `UNREACHABLE` | `{}` | P6 |
| npe_tune_search.py | `KNOB_ORDER` | `('hidden_features', 'num_transforms', 'num_bins', 'learning_rate', 't...` | P4 |
| npe_tune_search.py | `NPE_ADAPTER` | `SpaceAdapter(name='npe_flow_only', dimensions=space_dimensions, confi...` | P6 |

### Job-script variables and directives (101)

| file | name | kind | default | owner |
|---|---|---|---|---|
| joint/stage1/jobs/build_latent_bank.pbs | `PBS_-N` | directive | `build_latent_bank` | plumbing |
| joint/stage1/jobs/build_latent_bank.pbs | `PBS_-l` | directive | `select=1:ncpus=2:mem=8gb` | plumbing |
| joint/stage1/jobs/build_latent_bank.pbs | `PBS_-l` | directive | `walltime=02:00:00` | plumbing |
| joint/stage1/jobs/build_latent_bank.pbs | `PBS_-j` | directive | `oe` | plumbing |
| joint/stage1/jobs/build_latent_bank.pbs | `ARM` | -v default | `S` | plumbing |
| joint/stage1/jobs/build_latent_bank.pbs | `OUT_DIR` | -v default |  | plumbing |
| joint/stage1/jobs/build_latent_bank.pbs | `N_TRACES` | -v default | `64` | P7 |
| joint/stage1/jobs/build_latent_bank.pbs | `WELLS_PER_DONOR` | -v default | `2` | P7 |
| joint/stage1/jobs/build_latent_bank.pbs | `N_WINDOWS` | -v default | `8` | P7 |
| joint/stage1/jobs/build_latent_bank.pbs | `T_WIN` | -v default | `60.0` | P7 |
| joint/stage1/jobs/build_latent_bank.pbs | `FS` | -v default | `50.0` | P7 |
| joint/stage1/jobs/build_latent_bank.pbs | `PI` | -v default | `0.0` | P7 |
| joint/stage1/jobs/build_latent_bank.pbs | `GAP_MODES` | -v default | `range_shift` | P7 |
| joint/stage1/jobs/build_latent_bank.pbs | `N_PER_THETA` | -v default | `2` | P7 |
| joint/stage1/jobs/build_latent_bank.pbs | `PROVIDER` | -v default | `reference` | plumbing |
| joint/stage1/jobs/build_latent_bank.pbs | `SEED` | -v default | `0` | P5 |
| joint/stage1/jobs/build_latent_bank.pbs | `MAX_RECORDS` | -v default | `0` | P7 |
| joint/stage1/jobs/build_latent_bank.pbs | `DRYRUN` | -v default | `0` | plumbing |
| joint/stage1/jobs/build_latent_bank.pbs | `ENV_NAME` | -v default | `sbi_env` | plumbing |
| joint/stage1/jobs/build_latent_bank.pbs | `IDX` | env default | `0` | P7 |
| joint/stage3/jobs/joint_arms.pbs | `PBS_-N` | directive | `joint_arms` | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `PBS_-l` | directive | `select=1:ncpus=4:mem=16gb` | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `PBS_-l` | directive | `walltime=06:00:00` | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `PBS_-j` | directive | `oe` | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `SIM_SHARDS` | -v default |  | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `REAL_SHARDS` | -v default |  | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `OUT_DIR` | -v default |  | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `EPOCHS` | -v default | `10` | P5 |
| joint/stage3/jobs/joint_arms.pbs | `STEPS_PER_EPOCH` | -v default | `25` | P5 |
| joint/stage3/jobs/joint_arms.pbs | `B_SIM` | -v default | `128` | P5 |
| joint/stage3/jobs/joint_arms.pbs | `LAMBDA_DSN` | -v default | `0.1` | P2 |
| joint/stage3/jobs/joint_arms.pbs | `LAMBDA_REP` | -v default | `0.05` | P3 |
| joint/stage3/jobs/joint_arms.pbs | `N_DRAWS` | -v default | `128` | P3 |
| joint/stage3/jobs/joint_arms.pbs | `ENCODER_STEPS` | -v default | `200` | P2 |
| joint/stage3/jobs/joint_arms.pbs | `WARM_START_CKPT` | -v default |  | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `ENV_NAME` | -v default | `sbi_env` | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `DRYRUN` | -v default | `0` | plumbing |
| joint/stage3/jobs/joint_arms.pbs | `ARMS` | array | `A1 A0 A0s A2 A2s A3 A5 A_ref shuffled` | P6 |
| joint/stage3/jobs/joint_arms.pbs | `IDX` | env default | `0` | P7 |
| joint/stage3/jobs/joint_arms.pbs | `EXTRA` | array |  | P7 |
| joint/stage3b/jobs/stage3b.pbs | `PBS_-N` | directive | `stage3b` | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `PBS_-l` | directive | `select=1:ncpus=2:mem=8gb` | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `PBS_-l` | directive | `walltime=00:30:00` | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `PBS_-j` | directive | `oe` | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `RUNS_DIR` | -v default |  | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `SIM_SHARDS` | -v default |  | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `REAL_SHARDS` | -v default |  | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `PROBE_CKPT` | -v default |  | P7 |
| joint/stage3b/jobs/stage3b.pbs | `BOOTSTRAP_DIR` | -v default |  | P7 |
| joint/stage3b/jobs/stage3b.pbs | `OUT` | -v default | `stage3b_report.md` | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `OUT_JSON` | -v default |  | P7 |
| joint/stage3b/jobs/stage3b.pbs | `MAX_PROBE_ROWS` | -v default | `512` | P7 |
| joint/stage3b/jobs/stage3b.pbs | `ENV_NAME` | -v default | `sbi_env` | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `DRYRUN` | -v default | `0` | plumbing |
| joint/stage3b/jobs/stage3b.pbs | `EXTRA` | array |  | P7 |
| joint/stage3c/jobs/stage3c.pbs | `PBS_-N` | directive | `stage3c` | plumbing |
| joint/stage3c/jobs/stage3c.pbs | `PBS_-l` | directive | `select=1:ncpus=4:mem=16gb` | plumbing |
| joint/stage3c/jobs/stage3c.pbs | `PBS_-l` | directive | `walltime=04:00:00` | plumbing |
| joint/stage3c/jobs/stage3c.pbs | `PBS_-j` | directive | `oe` | plumbing |
| joint/stage3c/jobs/stage3c.pbs | `CKPT` | -v default |  | plumbing |
| joint/stage3c/jobs/stage3c.pbs | `SIM_SHARDS` | -v default |  | plumbing |
| joint/stage3c/jobs/stage3c.pbs | `OUT_DIR` | -v default |  | plumbing |
| joint/stage3c/jobs/stage3c.pbs | `KERNEL_AXES` | -v default |  | P7 |
| joint/stage3c/jobs/stage3c.pbs | `SPREAD_AXES` | -v default |  | P7 |
| joint/stage3c/jobs/stage3c.pbs | `N_FLOOR_DRAWS` | -v default | `48` | P7 |
| joint/stage3c/jobs/stage3c.pbs | `N_POST_DRAWS` | -v default | `128` | P7 |
| joint/stage3c/jobs/stage3c.pbs | `FD_SEEDS` | -v default | `3` | P7 |
| joint/stage3c/jobs/stage3c.pbs | `FD_STEP` | -v default | `0.02` | P7 |
| joint/stage3c/jobs/stage3c.pbs | `SEED` | -v default | `0` | P5 |
| joint/stage3c/jobs/stage3c.pbs | `ALLOW_REAL` | -v default | `0` | P7 |
| joint/stage3c/jobs/stage3c.pbs | `VALIDATION` | -v default |  | P7 |
| joint/stage3c/jobs/stage3c.pbs | `ENV_NAME` | -v default | `sbi_env` | plumbing |
| joint/stage3c/jobs/stage3c.pbs | `DRYRUN` | -v default | `0` | plumbing |
| joint/stage3c/jobs/stage3c.pbs | `EXTRA` | array |  | P7 |
| joint/stage4/jobs/joint_tune.pbs | `PBS_-N` | directive | `joint_tune` | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `PBS_-l` | directive | `select=1:ncpus=4:mem=16gb` | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `PBS_-l` | directive | `walltime=08:00:00` | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `PBS_-j` | directive | `oe` | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `CAMPAIGN` | -v default | `S-A1` | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `RESULTS_DIR` | -v default |  | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `SIM_SHARDS` | -v default |  | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `REAL_SHARDS` | -v default |  | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `OUT_DIR` | -v default |  | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `EPOCHS` | -v default | `10` | P5 |
| joint/stage4/jobs/joint_tune.pbs | `STEPS_PER_EPOCH` | -v default | `25` | P5 |
| joint/stage4/jobs/joint_tune.pbs | `SEED` | -v default | `0` | P5 |
| joint/stage4/jobs/joint_tune.pbs | `TAG` | -v default |  | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `SPLIT_HASH` | -v default |  | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `CONTRACT_DIGEST` | -v default |  | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `P` | -v default | `26` | P6 |
| joint/stage4/jobs/joint_tune.pbs | `EMBEDDING_DIM` | -v default | `12` | P6 |
| joint/stage4/jobs/joint_tune.pbs | `D_THETA` | -v default | `26` | P6 |
| joint/stage4/jobs/joint_tune.pbs | `DRYRUN` | -v default | `0` | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `ENV_NAME` | -v default | `sbi_env` | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `SKIP_CONDA` | -v default | `0` | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `PYBIN` | -v default | `python` | plumbing |
| joint/stage4/jobs/joint_tune.pbs | `IDX` | env default | `0` | P7 |
| joint/stage4/jobs/joint_tune.pbs | `IDS` | array |  | P7 |
| joint/stage4/jobs/launch_joint_tune.sh | `CAMPAIGN` | positional $1 |  | plumbing |
| joint/stage4/jobs/launch_joint_tune.sh | `RESULTS_DIR` | positional $2 |  | plumbing |
| joint/stage4/jobs/launch_joint_tune.sh | `PYBIN` | env default | `python` | plumbing |

### Same knob, different defaults (46 knobs)

One normalised knob name, every place it carries a default, where the defaults disagree. The runner's flag wins at run time wherever the runner passes the value explicitly; the library default is what a direct caller gets. Grouping is by name alone, so a row can be a name collision rather than one knob: read it as a prompt, not a verdict.

| knob | file | name | default |
|---|---|---|---|
| `T_win` | joint/stage1/build_latent_bank.py | `--T-win` | `60.0` |
|  | joint/stage1/demo_classes_generate.py | `--T-win` | `15.0` |
|  | joint/stage1/latent_sbi_simulator.py | `LatentSBISpec.__init__.T_win` | `60.0` |
| `allow_real` | joint/stage3c/jobs/stage3c.pbs | `ALLOW_REAL` | `0` |
|  | joint/stage3c/run_stage3c.py | `--allow-real` | `False` |
| `angular_alpha_deg` | dsn/config.py | `SearchConfig.angular_alpha_deg_range` | `(2.0, 20.0)` |
|  | dsn/config.py | `TrainConfig.angular_alpha_deg` | `18.0` |
|  | joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.angular_alpha_deg` | `18.0` |
|  | joint/stage3/probe_dsn_runtime.py | `--angular-alpha-deg` | `18.0` |
|  | joint/stage3/run_joint_arms.py | `--angular-alpha-deg` | `18.0` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.angular_alpha_deg` | `(2.0, 20.0)` |
| `arm` | joint/stage1/build_latent_bank.py | `--arm` | `'S'` |
|  | joint/stage1/jobs/build_latent_bank.pbs | `ARM` | `S` |
| `b_met` | joint/stage2/joint_batches.py | `BatchSpec.__init__.b_met` | `64` |
|  | joint/stage3/run_joint_arms.py | `--b-met` | `32` |
| `b_rep` | joint/stage2/joint_batches.py | `BatchSpec.__init__.b_rep` | `8` |
|  | joint/stage3/run_joint_arms.py | `--b-rep` | `4` |
| `b_sim` | joint/stage2/joint_batches.py | `BatchSpec.__init__.b_sim` | `512` |
|  | joint/stage3/jobs/joint_arms.pbs | `B_SIM` | `128` |
|  | joint/stage3/run_joint_arms.py | `--b-sim` | `128` |
| `block_family` | dsn/backbone.py | `BackboneConfig.block_family` | `0` |
|  | dsn/config.py | `SearchConfig.block_family_choices` | `(0, 1)` |
|  | joint/stage3/run_joint_arms.py | `--block-family` | `0` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.block_family` | `(0, 1)` |
| `campaign` | joint/stage4/jobs/joint_tune.pbs | `CAMPAIGN` | `S-A1` |
|  | joint/stage4/joint_space.py | `--campaign` | `None` |
|  | joint/stage4/npe_tune_joint.py | `--campaign` | `'S-A1'` |
| `correct_mc` | joint/stage2/joint_losses.py | `ReplicateConsistencyLoss.__init__.correct_mc` | `True` |
|  | joint/stage2/joint_losses.py | `replicate_statistic.correct_mc` | `False` |
| `depth_exponent` | dsn/backbone.py | `BackboneConfig.depth_exponent` | `4` |
|  | dsn/config.py | `SearchConfig.depth_exponent_range` | `(3, 6)` |
|  | joint/stage3/run_joint_arms.py | `--depth-exponent` | `3` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.depth_exponent` | `(3, 6)` |
| `drift_period_s` | joint/stage1/latent_gap.py | `GapSpec.__init__.drift_period_s` | `120.0` |
|  | joint/stage1/latent_nuisance.py | `NuisanceSpec.__init__.drift_period_s` | `600.0` |
| `dropout` | dsn/backbone.py | `BackboneConfig.dropout` | `0.0` |
|  | dsn/config.py | `RegularizationConfig.dropout_range` | `(0.0, 0.3)` |
|  | joint/stage3/run_joint_arms.py | `--dropout` | `0.0` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.dropout` | `(0.0, 0.3)` |
| `embedding_size` | dsn/backbone.py | `BackboneConfig.embedding_size` | `16` |
|  | dsn/config.py | `SearchConfig.embedding_size_range` | `(8, 16)` |
|  | joint/stage3/probe_dsn_runtime.py | `--embedding-size` | `12` |
|  | joint/stage3/run_joint_arms.py | `--embedding-size` | `10` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.embedding_size` | `(8, 16)` |
| `epochs` | joint/stage2/joint_train.py | `TrainConfig.__init__.epochs` | `20` |
|  | joint/stage3/jobs/joint_arms.pbs | `EPOCHS` | `10` |
|  | joint/stage3/run_joint_arms.py | `--epochs` | `10` |
|  | joint/stage4/jobs/joint_tune.pbs | `EPOCHS` | `10` |
|  | joint/stage4/npe_tune_joint.py | `--epochs` | `10` |
|  | joint/stage4/npe_tune_joint.py | `build_argv.epochs` | `10` |
| `gap_modes` | joint/stage1/build_latent_bank.py | `--gap-modes` | `'range_shift'` |
|  | joint/stage1/jobs/build_latent_bank.pbs | `GAP_MODES` | `range_shift` |
| `head_fusion` | dsn/backbone.py | `BackboneConfig.head_fusion` | `False` |
|  | dsn/config.py | `SearchConfig.head_fusion_choices` | `(0, 1)` |
|  | joint/stage3/run_joint_arms.py | `--head-fusion` | `0` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.head_fusion` | `(0, 1)` |
| `head_pool_ops` | dsn/backbone.py | `BackboneConfig.head_pool_ops` | `('mean',)` |
|  | dsn/config.py | `SearchConfig.head_pool_ops_choices` | `(0, 1)` |
| `hidden_features` | joint/stage2/joint_model.py | `build_joint_model.hidden_features` | `64` |
|  | joint/stage3/run_joint_arms.py | `--hidden-features` | `48` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.hidden_features` | `(64, 256)` |
|  | npe_tune_search.py | `SpaceSpec.hidden_features` | `(64, 256)` |
| `lambda_dsn` | joint/stage2/joint_train.py | `TrainConfig.__init__.lambda_dsn` | `0.0` |
|  | joint/stage3/jobs/joint_arms.pbs | `LAMBDA_DSN` | `0.1` |
|  | joint/stage3/run_joint_arms.py | `--lambda-dsn` | `0.1` |
| `lambda_rep` | joint/stage2/joint_train.py | `TrainConfig.__init__.lambda_rep` | `0.0` |
|  | joint/stage3/jobs/joint_arms.pbs | `LAMBDA_REP` | `0.05` |
|  | joint/stage3/run_joint_arms.py | `--lambda-rep` | `0.05` |
| `lambda_sep` | dsn/config.py | `SearchConfig.lambda_sep_range` | `(0.001, 1.0)` |
|  | dsn/config.py | `TrainConfig.lambda_sep` | `0.1` |
|  | joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.lambda_sep` | `0.1` |
|  | joint/stage3/probe_dsn_runtime.py | `--lambda-sep` | `0.1` |
|  | joint/stage3/run_joint_arms.py | `--lambda-sep` | `0.1` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.lambda_sep` | `(0.001, 1.0)` |
| `log_fn` | joint/stage2/joint_train.py | `train_joint.log_fn` | `print` |
|  | joint/stage3/run_joint_arms.py | `train_encoder_only.log_fn` | `None` |
| `loss_type` | dsn/config.py | `SearchConfig.loss_type_choices` | `('triplet', 'joint', 'joint_sep')` |
|  | dsn/config.py | `TrainConfig.loss_type` | `'triplet'` |
|  | joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.loss_type` | `'joint_sep'` |
|  | joint/stage3/probe_dsn_runtime.py | `--loss-type` | `'joint_sep'` |
|  | joint/stage3/run_joint_arms.py | `--loss-type` | `'joint_sep'` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.loss_type` | `('triplet', 'joint', 'joint_sep')` |
| `lr` | dsn/config.py | `SearchConfig.lr_range` | `(0.0001, 0.2)` |
|  | dsn/config.py | `TrainConfig.lr` | `0.0003` |
|  | joint/stage2/joint_train.py | `TrainConfig.__init__.lr` | `0.0005` |
|  | joint/stage3/run_joint_arms.py | `--lr` | `0.001` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.lr` | `(0.0001, 0.002)` |
| `margin` | dsn/config.py | `SearchConfig.margin_range` | `(0.1, 1.0)` |
|  | dsn/config.py | `TrainConfig.margin` | `0.3` |
|  | joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.margin` | `0.2` |
|  | joint/stage3/probe_dsn_runtime.py | `--margin` | `0.2` |
|  | joint/stage3/run_joint_arms.py | `--margin` | `0.2` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.margin` | `(0.1, 1.0)` |
| `mining_strategy` | dsn/config.py | `SearchConfig.mining_strategy_choices` | `('hard', 'easy_positive', 'easy_pos_semihard_neg')` |
|  | dsn/config.py | `TrainConfig.mining_strategy` | `'hard'` |
|  | joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.mining_strategy` | `'easy_pos_semihard_neg'` |
|  | joint/stage3/probe_dsn_runtime.py | `--mining-strategy` | `'easy_pos_semihard_neg'` |
|  | joint/stage3/run_joint_arms.py | `--mining-strategy` | `'easy_pos_semihard_neg'` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.mining_strategy` | `('hard', 'easy_positive', 'easy_pos_semihard_neg')` |
| `n_draws` | joint/stage2/joint_losses.py | `replicate_statistic.n_draws` | `None` |
|  | joint/stage3/jobs/joint_arms.pbs | `N_DRAWS` | `128` |
| `n_post_draws` | joint/stage3c/jobs/stage3c.pbs | `N_POST_DRAWS` | `128` |
|  | joint/stage3c/run_stage3c.py | `--n-post-draws` | `64` |
| `n_posterior_draws` | joint/stage2/joint_train.py | `train_joint.n_posterior_draws` | `256` |
|  | joint/stage3/run_joint_arms.py | `--n-posterior-draws` | `128` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.n_posterior_draws` | `(100, 400)` |
| `n_windows` | joint/stage1/build_latent_bank.py | `--n-windows` | `8` |
|  | joint/stage1/demo_classes_generate.py | `--n-windows` | `4` |
|  | joint/stage1/jobs/build_latent_bank.pbs | `N_WINDOWS` | `8` |
| `num_bins` | joint/stage2/joint_model.py | `build_joint_model.num_bins` | `10` |
|  | joint/stage3/run_joint_arms.py | `--num-bins` | `8` |
|  | npe_tune_search.py | `SpaceSpec.num_bins` | `(6, 16)` |
| `num_transforms` | joint/stage2/joint_model.py | `build_joint_model.num_transforms` | `5` |
|  | joint/stage3/run_joint_arms.py | `--num-transforms` | `3` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.num_transforms` | `(4, 12)` |
|  | npe_tune_search.py | `SpaceSpec.num_transforms` | `(4, 12)` |
| `one_minus_beta1` | dsn/config.py | `SearchConfig.one_minus_beta1_range` | `(0.01, 0.1)` |
|  | joint/stage3/run_joint_arms.py | `--one-minus-beta1` | `0.1` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.one_minus_beta1` | `(0.01, 0.1)` |
| `out` | joint/stage1/demo_classes_generate.py | `--out` | `'demo_classes.npz'` |
|  | joint/stage3/report_joint_arms.py | `--out` | `None` |
|  | joint/stage3b/jobs/stage3b.pbs | `OUT` | `stage3b_report.md` |
|  | joint/stage3b/run_stage3b.py | `--out` | `None` |
| `out_dir` | joint/stage1/demo_classes_plot.py | `--out-dir` | `'.'` |
|  | joint/stage4/npe_tune_joint.py | `--out-dir` | `'runs_joint'` |
| `patience` | dsn/config.py | `TrainConfig.patience` | `10` |
|  | joint/stage2/joint_train.py | `TrainConfig.__init__.patience` | `5` |
| `provider` | joint/stage1/build_latent_bank.py | `--provider` | `'reference'` |
|  | joint/stage1/jobs/build_latent_bank.pbs | `PROVIDER` | `reference` |
| `sep_centre_means` | dsn/config.py | `SearchConfig.sep_centre_means_choices` | `(0, 1)` |
|  | joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.sep_centre_means` | `None` |
| `sep_warmup_frac` | dsn/config.py | `SearchConfig.sep_warmup_frac_range` | `(0.0, 0.5)` |
|  | dsn/config.py | `TrainConfig.sep_warmup_frac` | `0.0` |
|  | joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.sep_warmup_frac` | `0.0` |
|  | joint/stage3/probe_dsn_runtime.py | `--sep-warmup-frac` | `0.0` |
|  | joint/stage3/run_joint_arms.py | `--sep-warmup-frac` | `0.0` |
| `steps_per_epoch` | joint/stage2/joint_train.py | `TrainConfig.__init__.steps_per_epoch` | `50` |
|  | joint/stage3/jobs/joint_arms.pbs | `STEPS_PER_EPOCH` | `25` |
|  | joint/stage3/run_joint_arms.py | `--steps-per-epoch` | `25` |
|  | joint/stage4/jobs/joint_tune.pbs | `STEPS_PER_EPOCH` | `25` |
|  | joint/stage4/npe_tune_joint.py | `--steps-per-epoch` | `25` |
|  | joint/stage4/npe_tune_joint.py | `build_argv.steps_per_epoch` | `25` |
| `strict_semihard` | dsn/config.py | `SearchConfig.strict_semihard_choices` | `(0, 1)` |
|  | dsn/config.py | `TrainConfig.strict_semihard` | `False` |
|  | joint/stage2/dsn_loss_adapter.py | `DSNLossConfig.__init__.strict_semihard` | `True` |
|  | joint/stage3/probe_dsn_runtime.py | `--strict-semihard` | `1` |
|  | joint/stage3/run_joint_arms.py | `--strict-semihard` | `1` |
|  | joint/stage4/joint_space.py | `default_joint_space.strict_semihard` | `1` |
| `total_steps` | joint/stage2/dsn_loss_adapter.py | `build_dsn_loss.total_steps` | `None` |
|  | joint/stage3/probe_dsn_runtime.py | `--total-steps` | `2` |
| `warmup_frac_rep` | joint/stage3/run_joint_arms.py | `--warmup-frac-rep` | `0.3` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.warmup_frac_rep` | `(0.0, 0.5)` |
| `weight_decay` | dsn/config.py | `RegularizationConfig.weight_decay_range` | `(1e-05, 0.01)` |
|  | dsn/config.py | `SearchConfig.weight_decay_range` | `(0.0001, 0.01)` |
|  | dsn/config.py | `TrainConfig.weight_decay` | `0.0001` |
|  | joint/stage2/joint_train.py | `TrainConfig.__init__.weight_decay` | `0.0` |
|  | joint/stage3/run_joint_arms.py | `--weight-decay` | `0.0` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.weight_decay` | `(1e-05, 0.01)` |
| `width_multiplier` | dsn/backbone.py | `BackboneConfig.width_multiplier` | `2.0` |
|  | dsn/config.py | `SearchConfig.width_multiplier_range` | `(1.5, 3.0)` |
|  | joint/stage3/run_joint_arms.py | `--width-multiplier` | `2.0` |
|  | joint/stage4/joint_space.py | `JointSpaceSpec.width_multiplier` | `(1.5, 3.0)` |
<!-- inventory:end -->

## 8. Numbers reproduced in the sandbox `[RAN]`, 2026-10-01

Run against the clone at `834eb41` with Python 3.11.15, numpy 2.4.4; the
sandbox has no `torch`, `skopt` or `sbi`, so only the torch-free objects
were exercised.

| what | value | how |
|---|---|---|
| joint search space | 23 axes over 5 blocks (encoder 6, DSN loss 7, replicate 4, flow 2, optimiser 4) | `joint_space.provenance_report` |
| free axes per campaign | S-A1 12, S-A2 19, S-A5 16, S-A25 23 | `joint_space.free_axes` |
| resolved ranges at $(p, E, d_\theta) = (26, 12, 26)$ | `hidden_features` `[52, 256]`; `n_posterior_draws` `[104, 400]` | `default_joint_space(26, 12, 26)` |
| resolved ranges on the bench $(10, 10, 10)$ | `hidden_features` `[32, 256]`; `n_posterior_draws` `[40, 400]` | `default_joint_space(10, 10, 10)` |
| fixed by S5.1, as the space reports them | `head_pool_ops=1`, `num_bins=10`, `one_minus_beta2=1e-3`, `sep_warmup_frac=0.0`, `strict_semihard=1` | `provenance_report` |
| inventory | 546 rows: cli 164, signature 153 (104 with defaults), dataclass 82, constant 46, job_var 101; 46 knobs with more than one default | `inventory_joint_knobs.py --strict` |
| backbone parameter count at the runner's defaults | **not computed** (torch absent); usage v1.3 S8 states 359708 `[CLUSTER]` [corrected 2026-10-01: computed analytically from a torch-free replica of `backbone.py`'s helpers, P1 eq. (P1.10): 359450 at the runner's `E = 10`; 359708 at `E = 12`, which is the configuration `probe_dsn_runtime.py` builds and the cluster's number; 360224 at `E = 16`] | P1 S3.2 `[RAN]` |
| DSN-loss constants | $4\tan^2\!\alpha$ = 0.00488 / 0.4223 / 0.5299 at $\alpha$ = 2 / 18 / 20 deg; the DSN's silhouette reading $1 - 4\sin^2\!\alpha$ = 0.995 / 0.618 / 0.532, and 0 at 30 deg; $m_{\rm sq} = 2 m_{\cos}$ = 0.2 / 0.4 / 0.6 / 2.0 at $m_{\cos}$ = 0.1 / 0.2 / 0.3 / 1.0; ETF target $-1$ / $-0.5$ / $-1/3$ at $K$ = 2 / 3 / 4 | P2 S3.2, Table P2.1 `[RAN]` (`tools/p2_numbers.py`) |
| identities behind the loss | $\lVert z - z' \rVert_2^2 = 2 d_{\cos}(z, z')$ on unit rows (residual $10^{-15}$); the midpoint identity $Q_{\rm mid} = (2 Q(i,i'') + 2 Q(i',i'') - Q(i,i'))/4$ (residual $10^{-15}$); the angular hinge is zero exactly at $Q(i,i')/Q(i,i'') = 4\sin^2\!\alpha$ in the isosceles construction | P2 S3.2 `[RAN]` |
| metric batch and triplet pools | $B_{\rm met} = 32$: 16 rows per class and at most 7680 `hard` triplets at $C = 2$; 10 per class (30 rows) at $C = 3$; the easy-positive miners return at most one triplet per anchor | P2 S3.2 `[RAN]` |
| legal and reachable loss conditions | 13 legal (mining, loss, filter) triples; 9 reachable with `strict_semihard` fixed at 1 (F-u) | P2 S3.3 `[RAN]` |

## 9. How a document reaches the repository (D-035, D-052)

**One push at the end (D-052, 2026-10-01).** Every document of the set is
written turn by turn on the sandbox branch `docs/joint-docs`, forked from
`origin/main` at `834eb41`; nothing is applied or pushed while the set is
incomplete, and the Stage 1 tarball that was sent before D-052 is not to be
applied on its own. When the set is complete (plan Stage 7), `origin/main`
is fetched again, the branch is rebased if `main` moved, and ONE tarball
holding the whole `git format-patch` series from the base is sent; the
reply then gives the laptop commands: extract, `git am` (or `git apply` per
patch), review, `git push`. The gates of the `hpc-git-delivery` skill run
on every turn's files (the real file fetched from `origin/main` where one
exists; `py_compile`; `args.*` resolution; byte safety: pure ASCII and
LF-only for every `.py` and `.md`; the behavioural smoke test; the patch
re-applied in a fresh clone and compared byte for byte with the tested
files) and again on the whole series before it is sent. The KB copies of
this index and of P0 are still written in the turn that changes them.

[corrected 2026-10-01] v1 of this section described a per-turn delivery
(one tarball per turn, `git apply`, push, and a `git fetch` at the start of
the next turn to check the previous patch landed); D-052 replaced it with
the single delivery above. The fetch-before-build rule survives: the final
series is built against the re-fetched `origin/main`, never an assumed
state.
