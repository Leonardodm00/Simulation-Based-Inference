# Joint DSN + NPE documentation set -- index, status ledger and inventory

| date | change |
|---|---|
| 2026-10-01 | v1. Stage 1 of `claude/JOINT_DOCS_BUILD_PLAN_v1.md` (v1.1): this index; the inventory extractor `tools/inventory_joint_knobs.py` with its smoke test `tools/smoke_test_inventory.py` (31/31, run twice in the sandbox) and its output `tools/inventory.json`; the inventory block of S7 generated from the repository at `834eb41`; the findings of the plan's S4 re-checked against the extraction and extended (F-m .. F-p). Delivered as the first patch of the set (D-035). |
| 2026-10-01 | v1.1. Stage 2: `P0_PARAMETERS_OVERVIEW.md` drafted, its three generated tables rendered by the new `tools/p0_tables.py` (tests T6.1-T6.6 added to the suite: 38/38, run twice `[RAN]`) and checked with `--check-doc`; `--n-post-draws` re-owned to P7 (`stage3c`), `inventory.json` and the S7 block regenerated (`--check-index` OK); findings F-q and F-r added from the P0 reading, F-a's `build_argv` line reference corrected; S9 rewritten for D-052 (one push at the end, no per-turn delivery; first written as "D-047" and renumbered the same turn, that number having been taken by the Giulia chat's entry of about 16:10); status rows updated. |
| 2026-10-01 | v1.2. Stage 3: `E0_READERS_GUIDE_NOTATION.md` drafted -- the master table (179 declared symbols `[RAN]`), the conventions with seventeen overload repairs beyond the plan's two, the glossary by first appearance, the reading map, the prerequisites, the running example at the DUP15HD and bench shapes, the spaces and maps, the analytic/computed pairs; the checker `tools/check_notation.py` with `tools/smoke_test_notation.py` (29/29, run twice `[RAN]`) passes on E0 and on P0. P0 v1.1: bare `T` and `\lambda` aligned with E0 (`T_{gg'}`, `\lambda_{\rm dsn}`), no value changed. S3: the plan's stage status is tracked here between re-issues of the plan. |
| 2026-10-02 | v1.8. Stage 4 continues: the P4 commit is `1f8e904` on `docs/joint-docs` (ledger filled). `P5_OPTIMISER_AND_SCHEDULE.md` drafted (the four optimiser axes and the schedule the runner fixes around them: `epochs`, `steps_per_epoch`, `b_met`, `b_rep`, `encoder_steps`, `grad_clip`, `beta2`, `patience`, the grouped split, the seed; one optimiser step and one run written out from `joint_train.py`, `joint_batches.py`, `run_joint_arms.py` and torch 2.10.0's own `adam.py` / `adamw.py` / `clip_grad.py` (read from the GitHub tag; torch is not installed here) as eq. (P5.1)-(P5.9), with sbi 0.27.0's `train()` and the DSN's `TrainConfig` / `SearchConfig` / `RegularizationConfig` and its 20 JSON configs beside them; the horizons $1/\upsilon_1 \in [10, 100]$, $1/\upsilon_2 = 1000$ and $1/(\eta \gamma_{\rm wd}) \ge 5 \times 10^{4}$ steps against the planned 250; the schedule arithmetic of Table P5.1; every number recomputed by the new torch-free `tools/p5_numbers.py` `[RAN]`; owns F-c with P1, F-d with P6, F-g with P1 and P2, F-i with P3, F-l, F-m and F-r's optimiser rows, F-q with P2, and the new F-ag to F-al). `E0_READERS_GUIDE_NOTATION.md` v1.5: the optimiser-and-schedule group appended (15 rows; 297 declared `[RAN]`); the $L_{\rm sel}$ row annotated for F-al, not renamed. `tools/smoke_test_notation.py` gains T4.8 (P5): 37/37 twice with `--docs-dir ..`; the inventory suite 38/38 twice; `p0_tables.py --check-doc` and `inventory_joint_knobs.py --check-index` OK `[RAN]`. S6: F-ag to F-al added; F-c, F-d, F-i, F-l, F-m, F-q, F-r extended. S8: the P5 numbers. P0 Table C's wording for `grad_clip` ("over all trainable parameters"; the call is over `model.parameters()`, F-ah) noted for the Stage 6 review. |
| 2026-10-02 | v1.7. Stage 4 continues, in the next chat: the seven-commit series is on `origin/main` (`d991836` .. `ab7ea82`, directly above `745edce`; `git fetch` 2026-10-02 13:17 `[RAN]`) and the status ledger's commit column is filled from it; `docs/joint-docs` re-forked from `ab7ea82`; both suites re-run on the pushed state before any edit (35/35, 38/38 `[RAN]`). `P4_FLOW_AXES.md` drafted (the two flow axes and the fixed `num_bins`, the two standardisation choices and the library constants; the flow written out from the `sbi` 0.27.0 and `zuko` 1.6.0 wheels as eq. (P4.1)-(P4.12), the weight count in closed form and checked against the layer shapes, the box-coordinate anchor $L_0 = 0$ and the identity-initialised loss $0.564\, d_\theta$, every number recomputed by the new torch-free `tools/p4_numbers.py` `[RAN]`; the differences table against the standalone NPE tuner; owns F-a with P6, F-j with P6, F-m and F-r's flow rows, and the new F-ac to F-af). `E0_READERS_GUIDE_NOTATION.md` v1.4: the flow group appended (15 rows, 15 symbols; 272 declared `[RAN]` -- the v1.3 count was 257 as `--list` reports it, not 256); convention 14's reserved flow symbols are now declared. `tools/smoke_test_notation.py` gains T4.7 (P4): 36/36 twice with `--docs-dir ..`; the inventory suite 38/38 twice; `p0_tables.py --check-doc` and `inventory_joint_knobs.py --check-index` OK `[RAN]`. S6: F-ac to F-af added, F-a and F-m extended. S8: the P4 numbers. P0 S3.2's line reference for the width rule (`:297-300`) is off by two (`:299-302`), noted for the Stage 6 review. |
| 2026-10-02 | v1.6. Delivery at the chat change (D-054): the seven commits of `docs/joint-docs` (`b961784` .. this one) leave as one `git format-patch` series for the user to apply with `git am` and push; S9 annotated; S4 records that `origin/main` moved to `745edce` (`hpc/dsn/` only, none of the files the set reads) `[RAN]`. The handoff for the next chat is `claude/HANDOFF_2026-10-02_joint_docs_P3_done.md` `[KB]`. No document content changed. |
| 2026-10-02 | v1.5. Stage 4 continues: `P3_REPLICATE_AXES.md` drafted (the four replicate axes and the nine fixed knobs of the term; the term written out from the code as one function of the knobs, eq. (P3.1)-(P3.12), with the two samplers checked against the `sbi` 0.27.0 wheel and every number recomputed by the new torch-free `tools/p3_numbers.py` `[RAN]`; the three readings of the $4 d_\theta$ floor and the $\kappa_S$ bias, cited from `claude/METRIC_REPLICATE_v1_4.md` and `claude/FINITE_DRAW_CORRECTION_v1.md` with the repository plan flagged as v0.6.5; owns F-e, F-i with P5, F-o with P6, F-p, and the new F-y to F-ab). `E0_READERS_GUIDE_NOTATION.md` v1.3: the replicate-term group appended (13 rows, 13 symbols; 256 declared `[RAN]`). `tools/smoke_test_notation.py` gains T4.6 (P3): 35/35 twice with `--docs-dir ..`; the inventory suite 38/38 twice; `p0_tables.py --check-doc` and `inventory_joint_knobs.py --check-index` OK `[RAN]`. S6: F-y to F-ab added. S8: the P3 numbers. |
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
| 00 | `00_INDEX.md` | drafted | v1.6 at `ab7ea82`; v1.7 at `1f8e904`; v1.8 pending push (D-052) | `claude/joint_docs/00_INDEX.md` | this file; v1.8 |
| tools | `tools/inventory_joint_knobs.py`, `tools/smoke_test_inventory.py`, `tools/inventory.json`, `tools/p0_tables.py`, `tools/check_notation.py`, `tools/smoke_test_notation.py`, `tools/p2_numbers.py`, `tools/p3_numbers.py`, `tools/p4_numbers.py`, `tools/p5_numbers.py` | drafted | inventory tools and `p0_tables.py` at `ef7b2ef`; `check_notation.py`, `p2_numbers.py` at `9fac378`; `smoke_test_notation.py`, `p3_numbers.py` at `5097728`; `smoke_test_notation.py` (T4.7) and `p4_numbers.py` at `1f8e904`; `smoke_test_notation.py` (T4.8) and `p5_numbers.py` pending push (D-052) | -- | 31/31 twice `[RAN]` at v1 [corrected 2026-10-01: 38/38 twice at v1.1, `p0_tables.py` and tests T6.1-T6.6 added; at v1.2 `check_notation.py` with its own suite, 29/29 twice; at v1.4 the checker extended and the suite at 34/34 twice; at v1.5 `p3_numbers.py` added and the notation suite at 35/35 twice (T4.6); at v1.7 `p4_numbers.py` added and the notation suite at 36/36 twice (T4.7); at v1.8 `p5_numbers.py` added and the notation suite at 37/37 twice (T4.8)] |
| P0 | `P0_PARAMETERS_OVERVIEW.md` | drafted | `3adb82f` (v1.1; first at `ef7b2ef`) | `claude/joint_docs/P0_PARAMETERS_OVERVIEW.md` | v1.1; tables A, F, K generated, `--check-doc` OK x3; notation check OK |
| P1 | `P1_ENCODER_AXES.md` | drafted | `f2bc923` | -- | v1; notation check OK; owns F-b, F-c, F-g, F-m (encoder rows), F-s |
| P2 | `P2_DSN_LOSS_AXES.md` | drafted | `9fac378` | -- | v1; notation check OK; owns F-f, F-h, F-q (with P5), F-t, F-u, F-v, F-w, F-x |
| P3 | `P3_REPLICATE_AXES.md` | drafted | `5097728` | -- | v1; notation check OK; owns F-e, F-i (with P5), F-o (with P6), F-p, F-y, F-z, F-aa, F-ab |
| P4 | `P4_FLOW_AXES.md` | drafted | `1f8e904` | -- | v1; notation check OK; owns F-a (with P6), F-j (with P6), F-m and F-r (flow rows), F-ac, F-ad, F-ae (with P7, P1), F-af |
| P5 | `P5_OPTIMISER_AND_SCHEDULE.md` | drafted | pending push (D-052) | -- | v1; notation check OK; owns F-c (with P1), F-d (with P6), F-g (with P1, P2), F-i (with P3), F-l, F-m and F-r (optimiser rows), F-q (with P2), F-ag, F-ah, F-ai, F-aj, F-ak, F-al |
| P6 | `P6_SEARCH_DRIVER.md` | planned | -- | -- | next turn |
| P7 | `P7_UPSTREAM_AND_JOBS.md` | planned | -- | -- | |
| E0 | `E0_READERS_GUIDE_NOTATION.md` | drafted | v1.3 at `5097728` (first at `3adb82f`); v1.4 at `1f8e904`; v1.5 pending push (D-052) | -- | v1.5; 297 declared symbols `[RAN]` [corrected 2026-10-02: the v1.2 count was 244 as `--list` reports it, not 239; the v1.3 count 257, not 256]; `--self` check OK |
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
  2026-10-01 from `origin/main`: unchanged `[RAN]`. [2026-10-02] Re-fetched
  again: `origin/main` is at `745edce` (one commit on `834eb41`, `hpc/dsn/`
  cohort fields and the Giulia cohort config; it touches none of the files
  the set reads or the inventory extracts, so the freeze stays at `834eb41`
  until the Stage 7 rebase) `[RAN]`. [2026-10-02, the P4 turn] Re-fetched
  in the next chat: `origin/main` is at `ab7ea82`, the seven commits of the
  series applied on `745edce` with `git am` and nothing else; the freeze
  stays at `834eb41` `[RAN]`. The joint plan in the
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
| F-a | `num_bins`: runner default **8**; `build_joint_model` default 10; the space declares it FIXED at 10; `build_argv` passes no `--num-bins`, so a Stage 4 trial trains an 8-bin flow while its ledger spec says 10 [extended 2026-10-02: the difference is 598 against 754 conditioner outputs per transform and 167960 against 201032 flow weights at the space's lower corner `[RAN]`; the trained value is recorded in the trial's `argv` (no `--num-bins`), the runner record's `config.num_bins` and the checkpoint's `meta["flow"]`; the standalone NPE tuner searches the knob in `[6, 16]`] | `[REPO]` `joint/stage3/run_joint_arms.py:229`; `joint/stage2/joint_model.py:195-197`; `joint/stage4/joint_space.py:322-324`; `joint/stage4/npe_tune_joint.py:86-106, 206-217` [corrected 2026-10-01: was `198-203`]; P4 S3.1, S3.8 | P4, P6 | patch (D-038) |
| F-b | `head_pool_ops`: the space declares code **1** (`("mean","max","std")`) fixed; `make_backbone` never sets it, so `BackboneConfig`'s default `("mean",)` (code 0) is built | `[REPO]` `joint/stage4/joint_space.py:322-324`; `joint/stage3/run_joint_arms.py:301-308`; `dsn/backbone.py:76`; `dsn/condition_space.py:113` | P1, P6 | patch (D-038) |
| F-c | `RANGE_PROVENANCE` names `SearchConfig.weight_decay_range` for `weight_decay` and `SearchConfig.dropout_range` for `dropout`; `SearchConfig` has `weight_decay_range = (1e-4, 1e-2)` and no `dropout_range`; the values the space carries, `(1e-5, 1e-2)` and `(0.0, 0.3)`, are `RegularizationConfig`'s [extended 2026-10-02: `TUNING_1` S3.7 documents both weight-decay ranges for the standalone search and says the `search` one is inert there, so the joint space copied the active range under the inert range's name, P5 S3.8] | `[REPO]` `joint/stage4/joint_space.py:121-145, 221, 240`; `dsn/config.py:934, 1183-1184`; `dsn/Documentation/TUNING_1_searched_axes.md` S3.7 | P1, P5 | report; open whether the two strings get fixed in the D-038 stream |
| F-d | `patience=99` in the runner: early stopping cannot fire in any run shorter than 99 epochs (default `epochs=10`); the best-validation state is still restored. `control_config_from`'s docstring expects early stopping to fire on shuffled data [extended 2026-10-02: the rule itself is sbi's (`epoch - best_epoch >= patience` against `_converged`'s `> stop_after_epochs - 1`; the same stopping epoch on 2000 of 2000 random validation curves `[RAN]`), so the value and not the rule is the problem; the earliest epoch it can fire is index 99, i.e. 100 epochs; the DSN's `TrainConfig.__post_init__` warns on exactly `max_epochs <= patience` and the joint `TrainConfig` does not; the library default 5 fires from epoch index 5, P5 S3.2, S3.8] | `[REPO]` `joint/stage3/run_joint_arms.py:519-524`; `joint/stage4/joint_space.py:712-717`; `joint/stage2/joint_train.py:262-267`; `dsn/config.py:869-875`; `[REPO sbi wheel]` `inference/trainers/base.py:1243-1253` | P5, P6 | report; open (a `--patience` flag with the library's 5 and a `TrainConfig` warning are D-038 candidates) |
| F-e | the `n_posterior_draws` floor's stated reason (rank; "variance, never bias") in `joint_space.py` and plan S5.1 is superseded by `claude/METRIC_REPLICATE_v1_4.md` S3.13.1: $\bar C$ is full rank from $S_{\rm mc} = 14$ and the floor's content is $\kappa_S \le 1.151$; the documentation patch (plan v0.6.6, two docstrings, one error message) is not applied at `834eb41` | `[REPO]` `joint/stage4/joint_space.py:276-281, 312-316`; plan S5.1 `:1082-1087` (v0.6.5); `[KB]` usage v1.3 S5.3; `[KB]` `FINITE_DRAW_CORRECTION_v1.md` S3.9.8 item 4; P3 S3.3.4 `[RAN]` | P3, E5 | report; open (the $\kappa_S$ repair is an Open call of the log; D-038 candidate) |
| F-f | `canonicalise_config`'s docstring quotes margin 0.3 (the DSN `TrainConfig`) while `INACTIVE_CANONICAL` pins 0.2 after the `[CORRECTION]` [extended 2026-10-01: `boundary_axes`'s docstring says the same, `:780-782`] | `[REPO]` `joint/stage4/joint_space.py:165-176, 536-538, 780-782` | P2 | report; open |
| F-g | two sources for the DSN-inherited ranges: the `SearchConfig` dataclass defaults, which the space copies (`depth_exponent (3, 6)`, `width_multiplier (1.5, 3.0)`, `lambda_sep (1e-3, 1.0)`, `lr (1e-4, 0.2)`), and the JSON that `TUNING_1` documents (`{2..5}`, `[1.5, 5]`, `[1e-2, 20]`) | `[REPO]` `dsn/config.py:893-934`; `dsn/Documentation/TUNING_1_searched_axes.md` S1 | P1, P2, P5 | report (both stated) |
| F-h | the joint stack's base loss differs from the standalone DSN's: `DSNLossConfig` (margin 0.2, `joint_sep`, `easy_pos_semihard_neg`, `strict_semihard=True`) against the DSN `TrainConfig` (0.3, `triplet`, `hard`, `False`) | S7, "Same knob, different defaults" | P2 | report |
| F-i | configured-by-code, never by flag: `TrainConfig.grad_clip=5.0`, `beta2=0.999`; `ReplicateConsistencyLoss` `jitter=1e-6`, `t_floor=1e-8`, `p_eff_min=1e-3`, `correct_mc=True`; `grouped_split` fractions `(0.7, 0.15, 0.15)`; `stem_width=16`; diagnostics on at most 256 report rows and 64 replicate pairs [extended 2026-10-02: the optimiser rows also include `patience=99` written into the runner's `TrainConfig(...)` call, the `1e-6` improvement margin of the stopping rule, and torch's `eps=1e-8`, `amsgrad=False` below any flag; `beta2=0.999` equals the space's fixed `one_minus_beta2=1e-3` by two independent defaults, not by wiring, P5 S3.4] | `[REPO]` `joint/stage2/joint_train.py:41-44`; `joint/stage2/joint_losses.py:79-83, 264-266`; `joint/stage3/run_joint_arms.py:118, 307, 562, 614` | P3, P5 | report |
| F-j | resolved ranges at the DUP15HD shapes $(p, E, d_\theta) = (26, 12, 26)$: `hidden_features` `[52, 256]`, `n_posterior_draws` `[104, 400]`; on the bench $(10, 10, 10)$: `[32, 256]`, `[40, 400]`; free axes S-A1 12, S-A2 19, S-A5 16, S-A25 23 | `[RAN]` S8 | P4, P6 | report |
| F-k | `README_joint.md`'s test counts (47 / 40+1 / 27 / 28 / 23 / 24 / 14) and usage v1.3 S9's (46/0/1, 69/69, 30, 27, 33, ...) are of different dates | `[REPO]` `joint/README_joint.md`; `[KB]` usage v1.3 S9 | E9 | report (each with its date) |
| F-l | `batch_size_npe` reaches the runner as `--b-sim`; `b_met` (32) and `b_rep` (4) are not searched and not job variables [extended 2026-10-02: because an epoch is a fixed count of 25 steps, the axis moves the three streams' row ratio from 32 : 8 : 1 at 128 to 256 : 8 : 1 at 1024 and the simulated rows per trial from 32 000 to 256 000 `[RAN]`, P5 S3.3.4, Table P5.1] | `[REPO]` `joint/stage4/npe_tune_joint.py:99`; `joint/stage3/jobs/joint_arms.pbs:50-62`; `joint/stage2/joint_train.py:157`; `joint/stage2/joint_batches.py:172` | P5 | report |
| F-m | library defaults and runner defaults disagree for the same knobs: `TrainConfig` (20 epochs, 50 steps, lr 5e-4, patience 5) vs the runner (10, 25, 1e-3, 99); `BatchSpec` (512/64/8) vs `--b-sim/--b-met/--b-rep` (128/32/4); `build_joint_model` (64/5/10) vs `--hidden-features/--num-transforms/--num-bins` (48/3/8); `train_joint` `n_posterior_draws=256` vs `--n-posterior-draws 128`; `BackboneConfig` (`depth_exponent` 4, `embedding_size` 16) vs the runner (3, 10). The runner passes every value explicitly, so a Stage 3 run gets the runner's; a direct caller of the library gets the library's [extended 2026-10-02: below the library's flow defaults sit sbi's `posterior_nn` (50/5/10), sbi's `build_zuko_nsf` (50/5/8) and zuko's own (3 transforms, two hidden layers of 64); the three flows are 340730, 107634 and 1742736 (the standalone `NPEConfig`'s 128/8/10) weights at $(26, 12)$ `[RAN]`, P4 Table P4.1] [extended 2026-10-02: below the library's optimiser defaults sit sbi's `train()` (lr 5e-4, batch 200, 20 fruitless epochs, `Adam` without decay, 10 % of rows for validation) and torch's `AdamW` (lr 1e-3, `weight_decay` 1e-2, betas (0.9, 0.999), `eps` 1e-8), both read from source; the torch values P0 Table C marked "not read here" are now read, P5 S3.1, S3.5] | S7, "Same knob, different defaults" `[REPO]`; P4 S3.1; `[REPO sbi wheel]` `inference/trainers/npe/npe_base.py:252-259`; `[REPO torch v2.10.0]` `torch/optim/adamw.py:20-28` | P1, P4, P5 | report |
| F-n | `stage3c.pbs` defaults `N_POST_DRAWS=128` while `run_stage3c.py --n-post-draws` defaults to 64: the two entry points run different draw counts unless the variable is passed | `[REPO]` `joint/stage3c/jobs/stage3c.pbs:49`; `joint/stage3c/run_stage3c.py:66` | P7 | report; open |
| F-o | `JointSpaceSpec.n_posterior_draws` defaults to `(100, 400)`, below the $4 d_\theta = 104$ floor at $d_\theta = 26$; only `default_joint_space` resolves the floor, so a `JointSpaceSpec()` built directly carries a range the floor forbids | `[REPO]` `joint/stage4/joint_space.py:233, 308-316` | P3, P6 | report |
| F-p | `warmup_frac_rep`: the runner's default is 0.3 (`--warmup-frac-rep`), the space's inactive pin 0.0 and `ReplicateConsistencyLoss` default 0.0: a Stage 3 arm A5 ramps the term over the first 30 % of training, a Stage 4 trial with `rep_on = 0` records 0.0 | `[REPO]` `joint/stage3/run_joint_arms.py:224`; `joint/stage4/joint_space.py:178`; `joint/stage2/joint_losses.py:264` | P3 | report |
| F-q | the encoder-only pre-training of arms `A0`, `A0s` builds `torch.optim.AdamW(backbone.parameters(), lr=lr)`: torch's own defaults for `weight_decay` and `betas`, not the runner's `--weight-decay` and `--one-minus-beta1`, which reach only the joint loop's optimiser; the values of those torch defaults are not read here (torch is not installed in the sandbox) [extended 2026-10-02: read from the torch 2.10.0 source at the GitHub tag: `weight_decay=1e-2`, `betas=(0.9, 0.999)`, `eps=1e-8`; at the runner's own defaults the betas coincide and the decay differs (1e-2 against 0), a 0.20 % shrink of every encoder weight over the 200 pre-training steps `[RAN]` -- the only decay a default Stage 3 campaign applies to anything; under a searched `one_minus_beta1` or `weight_decay` the two optimisers of an `A0` run differ in both; the pre-training's last gradients also stay on the encoder tensors, F-ah, P5 S3.6, S3.8] | `[REPO]` `joint/stage3/run_joint_arms.py:154`, `:458-459`; `joint/stage2/joint_train.py:144-147`; `[REPO torch v2.10.0]` `torch/optim/adamw.py:24-27` | P2, P5 | report; open whether the pre-training should take the runner's optimiser flags (candidate for the D-038 stream) |
| F-r | four of the runner's defaults lie outside the ranges the space samples, so a Stage 3 arm at default hyper-parameters is not a point of the Stage 4 space: `--b-sim 128` not in `{256, 512, 1024}`; `--hidden-features 48` below the lower bound 52 at $(26, 12, 26)$ (inside the bench's `[32, 256]`); `--num-transforms 3` below `[4, 12]`; `--weight-decay 0.0` below `[1e-5, 1e-2]` [extended 2026-10-02: of the other two optimiser defaults, `--lr 1e-3` is inside its range with 77 % of the log-uniform prior's mass below it, and `--one-minus-beta1 0.1` is the closed upper edge of `[1e-2, 1e-1]`, inside the range but where `boundary_axes` would flag a best configuration `[RAN]`, P5 S3.1] | `[reasoning]` over P0 Table A (`[REPO]` `joint/stage3/run_joint_arms.py:217, 221, 227-228`; `[RAN]` S8) | P0, P4, P5, P6 | report |
| F-s | the joint space's `depth_exponent` range `(3, 6)` is the DSN `SearchConfig` dataclass default, not the DSN's configured `[2, 5]`; `TUNING_1` S3.1 says a bound of 6 doubles the block count to 64 and must not be searched without re-running its budget gate; at that bound the searched encoders have 49 M to 215 M parameters `[RAN]` (P1 S3.2), against the standalone study's documented maximum of 31.6 M over its own ranges; the joint driver has no parameter-count or budget guard and the runner records no count | `[REPO]` `joint/stage4/joint_space.py:221`; `dsn/config.py:893`; `dsn/Documentation/TUNING_1_searched_axes.md` S3.1; `[RAN]` P1 S3.2 | P1, P6 | report; open whether the range is narrowed to `{3, ..., 5}` or a guard added (Open calls) |
| F-t | the class count of the separation target and of the DSN loss's class mask is `len(unique(sim["cls"]))`, the **simulated** bank's, even when the metric stream is the real cohort (`A0`, `A2`, `A3`); equal on the bench by construction, unchecked on a cohort bank, where a real label outside `0..C-1` contributes to no class statistic and no error is raised | `[REPO]` `joint/stage3/run_joint_arms.py:402`; `dsn/dsn_joint_loss.py:112-123, 176-184` | P2, P7 | report; open (take the count from the metric source's labels) |
| F-u | with `strict_semihard` fixed at 1 (`spec.fixed`) and legality-projected, 9 of the DSN's 13 legal (mining, loss, filter) conditions are reachable from the joint space: the four cells of the two easy-positive miners under `joint`/`joint_sep` with the filter off are not; a scope statement of plan S5.1, which does not state the fixed value (`default_joint_space` docstring) | `[RAN]` P2 S3.3; `[REPO]` `joint/stage4/joint_space.py:287-291, 322-324`; `dsn/condition_space.py:222-255` | P2, P6 | report |
| F-v | the composite loss's per-batch counts `n_mined`, `n_strict`, `n_active` and the separation statistics `sep_mean_cos`, `sep_n_classes` are exposed by `DSNLossAdapter.stats()` and never written to the history; a metric term whose strict set is empty every step trains nothing and shows only as `history[].dsn = 0.0` with `rho_grad = NaN` | `[REPO]` `joint/stage2/dsn_loss_adapter.py:109-111`; `joint/stage2/joint_train.py:170-176, 250-252`; `dsn/dsn_joint_loss.py:320-334` | P2, E7 | report; open (log `stats()` per epoch; D-038 stream candidate) |
| F-w | `build_argv` passes `--strict-semihard` from `spec.fixed` but not `--sep-warmup-frac`; the space's fixed 0.0 and the runner's default 0.0 agree, so the ledger and the trained loss coincide by coincidence of defaults (the F-a pattern, one default change away) | `[REPO]` `joint/stage4/npe_tune_joint.py:206-211`; `joint/stage4/joint_space.py:322-324`; `joint/stage3/run_joint_arms.py:256` | P2, P6 | report; open (pass it as `strict_semihard` is passed; D-038 stream candidate) |
| F-x | `dsn/dsn_joint_loss.py` calls `warnings.warn` without importing `warnings`: a `SepWarmup` with a positive fraction and no horizon raises `NameError` instead of warning; unreachable from the joint path, whose adapter refuses that case first | `[REPO]` `dsn/dsn_joint_loss.py:152-157, 638`; `joint/stage2/dsn_loss_adapter.py:169-173` | P2 | report (DSN tree; out of scope for fixes, D-037) |
| F-y | the clamp before the logarithm (`t_floor` $= 10^{-8}$) zeroes the gradient of every replicate pair with $\hat T_{gg'} \le T_{\rm floor}$, which under `correct_mc = True` is every pair with $\hat T^{\rm raw}_{gg'} \le d_\theta/S_{\rm mc} + 10^{-8}$: the collapse signature itself; such a pair adds a constant 381 (at $\hat p_{\rm eff} = 3$) to the logged loss and nothing to the gradient; at a collapsed pair the fraction is 0.36 / 0.39 / 0.48 at $S_{\rm mc}$ = 104 / 128 / 256 (Gaussian draws, realised metric), against $\Pr(\chi^2_{26} \le 26) = 0.54$ with the exact metric; J11 tests loss values at a floor of $10^{-30}$ and no test covers the gradient; nothing counts the case in training | `[REPO]` `joint/stage2/joint_losses.py:232-236, 324`; `joint/stage2/smoke_test_joint_losses.py:82-106`; `[RAN]` `tools/p3_numbers.py` B5-B6; P3 S3.2, S3.7 | P3 | report; open (D-038 candidate: mask dead-zone pairs out of the batch mean and count them) |
| F-z | a non-positive target $\hat p_{\rm eff}$ is replaced by `p_eff_min` $= 10^{-3}$ and the pair stays in the batch mean, so the loss's slope in $\hat T_{gg'}$ is positive for every $\hat T_{gg'} > 10^{-3}$: an "undefined" target (the code's own word) becomes a pull toward agreement, weighted by the ramp; the count `n_p_eff_invalid` is the only trace | `[REPO]` `joint/stage2/joint_losses.py:312-325`; `[RAN]` `tools/p3_numbers.py` B5; P3 S3.2, S3.7 | P3 | report; open (D-038 candidate: zero the per-pair loss when the target is non-positive) |
| F-aa | arm `A5` on a real bank with no same-donor pair trains as `A1` silently: `needs_real` checks only that real shards exist, `rep_batch` returns `None` when $N_{\rm pair} = 0$, the loop skips the term; the run logs `rep` 0.0, `T` `nan`, `ramp` 1.0 from epoch 0 (`set_progress` never called) and writes a record without a `replicate` block; the batcher's report line `replicate pairs : 0` is the one visible trace; on the bench $N_{\rm pair} > 0$ by construction (`--wells-per-donor 2`), on the cohort it is D12 | `[REPO]` `joint/stage3/run_joint_arms.py:342-344, 613`; `joint/stage2/joint_batches.py:192-193, 213`; `joint/stage2/joint_train.py:178-180, 224`; `joint/stage2/joint_losses.py:278, 284-290`; P3 S3.3.1, S3.8 | P3, E5 | report; open (D-038 candidate: raise when `lambda_rep > 0` and $N_{\rm pair} = 0$) |
| F-ab | `enumerate_donor_pairs` pairs rows by `donor` only, and a row of every bank built so far is a window ($J$ windows per trace, one `donor` per donor, `--wells-per-donor` traces per donor): pairs of two windows of one well are enumerated as replicate pairs; at the Stage 1 job defaults (64 traces, 2 wells per donor, 8 windows) a donor has 16 rows and 120 pairs, 56 (47%) within one well and 64 across wells, $N_{\rm pair} = 3840$ against 32 pairs of wells; the bank's `well` field is not read; the plan's pair is two wells and the target $p_{\rm eff}$ is derived for two conditionally independent recordings, which two windows of one well (shared $\mathcal{G}$, shared $\nu$) are not, so for those pairs the expected statistic sits below the target [reasoning] and the two-sided loss pulls toward more disagreement; on the cohort the fraction depends on how the real bank's `donor` field is populated (Stage 6, not written) | `[REPO]` `joint/stage2/joint_batches.py:37-58, 160-164`; `joint/stage1/build_latent_bank.py:216, 224-252`; `joint/stage1/jobs/build_latent_bank.pbs:44-46`; `[RAN]` P3 S3.8; `[KB]` plan S2.5(b), METRIC S3.7.3 | P3, E5, E6 | report; open (D-038 candidate: enumerate pairs across distinct `well` values of one donor, or record the within-well fraction per run) |
| F-ac | `sbi` 0.27.0 passes `hidden_features` to zuko as the list `[hidden_features] * num_transforms`, and zuko's `MaskedMLP` reads it as its hidden-layer widths: `num_transforms` sets both the number of stacked transforms and the number of hidden layers of every conditioner; at the space's upper bound each conditioner is 12 plain ReLU layers deep and the chain 156 linear layers, with no residual connection or normalisation; the weight count is quadratic in `num_transforms` at fixed width (P4 eq. (P4.9)); the search's partial dependence on the axis conflates stages and depth; the plan's S5.1 and `JointSpaceSpec` describe a transform count only | `[REPO sbi wheel]` `neural_nets/net_builders/flow.py:1142-1143`; `[REPO zuko wheel]` `nn.py:255, 276-293`; `[RAN]` `tools/p4_numbers.py` B3 (four `MaskedLinear` layers at the runner's 3 transforms); P4 S3.2, S3.3.2 | P4, E2 | report; open (D-038 candidate: pass an explicit list of hidden widths of fixed depth from `build_joint_model`; `residual` is not exposed by sbi) |
| F-ad | both `default_joint_space` and `npe_tune_search.default_space` record `"width_rule": "clamp([2,8] * max(p,E), [64,256])"` in `anchored_to`, which as a function (a clamp into `[64, 256]`) disagrees with the code -- `lo = clamp(2 max(p,E), 32, 64)`, `hi = max(8 max(p,E), 256)` -- at 127 of the 128 values `max(p,E) = 1..128`: `[64, 208]` against the coded `[52, 256]` at the DUP15HD shapes; the resolved bounds are recorded beside the string, so nothing trains wrong | `[REPO]` `joint/stage4/joint_space.py:299-302, 328`; `npe_tune_search.py:131-134, 148`; `[RAN]` `tools/p4_numbers.py` B1; P4 S3.3.1 | P4, P6 | report; open (fix or drop the string; the F-c kind of repair) |
| F-ae | `stage3c._rebuild_model` and `stage3b.rebuild_encoder` rebuild the backbone with `make_backbone(W, E, dsn_main_dir, seed)` and no `encoder` dict, so the five encoder axes take the runner's defaults whatever the checkpoint was trained at, while the flow's three knobs are restored from `meta["flow"]` (with a refusal when absent) and the encoder's `meta["backbone_repr"]` is read by nothing: a checkpoint off the encoder defaults fails to load with a size mismatch (`depth_exponent`, `width_multiplier`, `block_family`, `head_fusion`) or loads silently (`dropout`, inert in `eval()`) | `[REPO]` `joint/stage3c/run_stage3c.py:336-373`; `joint/stage3b/run_stage3b.py:66-90`; `joint/stage3/run_joint_arms.py:282-308`; `joint/stage2/joint_model.py:227`; P4 S3.8 | P7, P1 | report; open (D-038 candidate: store the encoder dict in `meta` and read it back as the flow's is) |
| F-af | plan v0.6.5 S8 lists D6 (conditioner normalisation: `z_score_x = "none"` recommended, or sbi's `"structured"`) and D7 (ensemble semantics) as open; `build_joint_model` has taken D6's recommended option as its default since Stage 2, the Stage 0 contract names it, and the decisions log has no D-number for it; the choice is load-bearing: under any other `z_score_x` sbi wraps the backbone in a `Sequential` and the checkpoint's `encoder_state`, `freeze_encoder` and the Stage 3b encoder-alone loader no longer address the backbone | `[REPO]` plan S8 `:1571-1574`, Stage 0 `:1146-1147`; `joint/stage2/joint_model.py:198, 111-117, 170-181`; `[REPO sbi wheel]` `flow.py:1409-1411`; P4 S3.4 | P4, E9 | report; open (a status line in the plan's S8 or a D-number in the log) |
| F-ag | the code's epoch is `steps_per_epoch` (25) optimiser steps on `b_sim` rows drawn i.i.d. WITH replacement, whatever `b_sim` or the split's size; the plan's S5.1 ("One epoch is one pass over the simulated bank; at $B_{\rm sim} = 512$ that is about 58 steps"), the `batch_size_npe` trim rule of `default_joint_space` ("a batch that swallows the epoch makes early stopping meaningless", copied from `npe_tune_search.default_space`) and sbi's `train()` all mean a pass; at the runner's schedule ten epochs are 1.5 pass-equivalents of the DUP15HD training rows (taken as $0.7 \times 29\,616 = 20\,731$) at 128 and 12.3 at 1024, with 21 % of the rows never drawn at 128 `[RAN]`; `batch_size_npe` is thereby a data-budget axis; the trim thresholds (5 120 / 10 240 / 20 480 rows) refer to a loop the stack does not run, and the jobs never pass `--n-train` | `[REPO]` `joint/stage2/joint_train.py:157`; `joint/stage2/joint_batches.py:172`; `joint/stage4/joint_space.py:283-285, 304-306`; `npe_tune_search.py:126-129, 137-138`; `[KB]` plan S5.1; `[RAN]` `tools/p5_numbers.py` B2, P5 S3.2, Table P5.1 | P5 | report; open (state the epoch's definition in the plan and the docstring, or make the rule read `steps_per_epoch`) |
| F-ah | `clip_grad_norm_(model.parameters(), grad_clip)` measures every parameter whose `.grad` is not `None`; on `A0` and `A0s` the encoder tensors still carry the last pre-training step's gradients (`train_encoder_only` zeroes before each backward, not after the last, then sets `requires_grad_(False)`, which does not clear `.grad` [textbook, from memory]; the loop's `opt.zero_grad` reaches only the `trainable` tensors; `freeze_encoder` touches the flag and the mode only), so the clipped norm is the root of the squared flow-gradient norm plus the squared stale norm: a flow gradient of norm 5 beside a stale norm of 5 is scaled by 0.71, of norm 1 beside 10 by 0.50 `[RAN]`, and each firing also rescales the stale tensors in place; `A3` (fresh backbone) and `A_ref` (one frozen parameter, never a gradient) are unaffected; the `trainable` filter's comment says torch would decay a tensor whose gradient is `None`, which torch does not (`adam.py:151` skips it) -- the filter is right for the stale-gradient reason instead | `[REPO]` `joint/stage2/joint_train.py:135-147, 159, 208-210`; `joint/stage3/run_joint_arms.py:161-167`; `joint/stage2/joint_model.py:80-95`; `[REPO torch v2.10.0]` `torch/nn/utils/clip_grad.py:157, 165-182, 230`, `torch/optim/adam.py:151`; P5 S3.2, S3.8 | P5 | report; open (D-038 candidate: `backbone.zero_grad(set_to_none=True)` at the end of `train_encoder_only`, or clip the `trainable` list; not run, torch absent) |
| F-ai | within the searched box the decoupled decay is nearly inert at the runner's schedule: the cumulative factor $(1 - \eta \gamma_{\rm wd})^{250}$ lies between $1 - 2.5 \times 10^{-7}$ and 0.995 (a 0.50 % shrink at the strongest corner, 1.98 % over the library's 1000 steps), the horizon $1/(\eta \gamma_{\rm wd})$ between $5 \times 10^{4}$ and $10^{9}$ steps `[RAN]`; the DSN searched the same range with up to $10^{4}$ steps per run; the partial dependence on `weight_decay` will read as flat for this reason [reasoning on the closed form] | `[RAN]` `tools/p5_numbers.py` B1; `[REPO]` `joint/stage4/joint_space.py:240`; `dsn/Documentation/TUNING_2_fixed_knobs.md` S3.2; P5 S3.3.3, S3.8 | P5 | report; open (drop the axis, lengthen the schedule, or widen the range upward: a decision) |
| F-aj | the runner's record stores `"val_npe": history[-1].get("val_npe")`, the LAST epoch's validation score, while the weights every recorded diagnostic, `L`, $\hat\Delta$ and the checkpoint are computed on are the BEST epoch's (`load_state_dict(best_state)`); the two coincide only when the last epoch is the best; the full series is in `history`; the tuner reads `L`, so the search is unaffected; no `best_epoch` / epochs-to-stop field exists, which `control_config_from`'s docstring asks for | `[REPO]` `joint/stage3/run_joint_arms.py:594`; `joint/stage2/joint_train.py:229-232, 269-270`; `joint/stage4/npe_tune_joint.py:517-525`; `joint/stage4/joint_space.py:713-717`; P5 S3.7, S3.8 | P5 | report; open (store the minimum of the series and `best_epoch`) |
| F-ak | when `lambda_dsn > 0` the gradient-cosine probe draws one extra `batcher.next()` per epoch, advancing the simulated, metric and replicate generators by one batch each, so two runs at one `--seed` that differ only in whether the DSN term is on (`A1` against `A2`; an S-A1 trial against an S-A2 trial with `dsn_on = 1`) share the simulated batches of epoch 0 and none after (1 280 rows ahead after ten epochs at 128 `[RAN]`); J9's independence holds within a run; across arms the seed does not fix the data sequence, which the plan's S2.4a already assumes when it declines to pair seeds across arms | `[REPO]` `joint/stage2/joint_train.py:234-249`; `joint/stage2/joint_batches.py:126-128, 201-203`; `[KB]` plan S2.4a; `[RAN]` `tools/p5_numbers.py` B2; P5 S3.8 | P5 | report (an observation on comparability; a fourth generator for the probe would remove it) |
| F-al | the runner scores `L` -- the ledger's objective `nll` and the simulated-arm endpoint -- on the REPORT split (`per_row_nll(model, theta[rp], x[rp])`), and the SELECTION split is read only by `evaluate_npe` once per epoch for the stopping rule and the best-state restore; the plan assigns the roles the other way (its decision rule compares arms "on the frozen selection split", S2.4, and the report split "is touched once, at the end", S4.4), and E0's $L_{\rm sel}$ row follows the plan ("the search objective"); across a Stage 4 campaign the report split is therefore scored once per trial and selected on, and the selection split's 15 % of donors decide only the restored epoch (nothing at all under F-d's inoperative stopping beyond that) | `[REPO]` `joint/stage3/run_joint_arms.py:369-370, 536-538, 587`; `joint/stage4/npe_tune_joint.py:524-525`; `joint/stage2/joint_train.py:227-232`; `[KB]` plan S2.4, S4.4; E0 S1 (annotated at v1.5); P5 S3.2, S3.7, S3.8 | P5, E8 | report; open (which assignment is wanted is a decision for the log; E0 annotates, does not rename) |

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
| replicate-term constants and the draw floor | $\Sigma_0 = I_{d_\theta}/12$ on the unit cube; floor $4 d_\theta$ = 104 (26) / 40 (10); $\lambda_{\rm rep} \in [10^{-3}, 10]$; $\kappa_S$ finite from $S_{\rm mc} = 15$, $\hat C_g$ PD from 27, $\bar C$ from 14; $\kappa_S$ = 1.151 / 1.119 / 1.056 / 1.050 / 1.035 / 1.014 at $S_{\rm mc}$ = 104 / 128 / 256 / 285 / 400 / 1000, the floor identity $(8 d_\theta - 2)/(7 d_\theta - 3)$ = 1.151 (26), 1.164 (10); $\mathrm{res}_S$ at $p_{\rm eff} = 3$ = 0.490 / 0.381 / 0.173 / 0.154 / 0.107 / 0.041, 1.6-2.0 times $d_\theta/S_{\rm mc}$; $p_\times$ = 1.41 / 1.51 / 1.72 / 1.74 / 1.79 / 1.87; the loss minimised at 2.57 / 2.66 / 2.84 / 2.85 / 2.90 / 2.96; all agreeing with `[KB]` `FINITE_DRAW_CORRECTION_v1.md` S3.9.5 | `tools/p3_numbers.py` B1-B2, P3 S3.3.4 `[RAN]` |
| the replicate ramp and the step's cost at the runner's defaults | 250 planned steps; the ramp complete at step 75 (epoch 3); logged `ramp` 0.320 / 0.653 / 0.987 / 1 for epochs 0-3; 1024 draws of $d_\theta$ reals and 8 covariances per step ($2 B_{\rm rep} S_{\rm mc}$ at 4 and 128) | `tools/p3_numbers.py` B3-B4, P3 S3.2 `[RAN]` |
| the two clamps | loss plateau $(\ln 10^{-8} - \ln \hat p_{\rm eff})^2$ = 381 (3) / 440 (13); $\Pr(\chi^2_{26} \le 26)$ = 0.537; gradient weight $2 \lvert \ln(\hat T_{gg'}/\hat p_{\rm eff}) \rvert / \hat T_{gg'}$ = 1141 / 68.0 / 2.20 at $\hat T_{gg'}$ = 0.01 / 0.1 / 1 against 3; the invalid-target slope positive at 0.1, 1, 3; at a collapsed pair with the realised metric (4000 pairs, Gaussian draws) the mean $\hat T_{gg'}$ = +0.039 / +0.024 / +0.005 against eq. (72)'s +0.038 / +0.024 / +0.006 and the fraction with $\hat T_{gg'} \le 0$ = 0.358 / 0.389 / 0.481 at $S_{\rm mc}$ = 104 / 128 / 256 (F-y, F-z) | `tools/p3_numbers.py` B5-B6, P3 S3.3.4, S3.8 `[RAN]` |
| bench replicate pairs at the Stage 1 defaults | 16 rows per donor, 120 pairs per donor, 56 within one well (47%) and 64 across wells; 32 donors, $N_{\rm pair}$ = 3840 against 32 pairs of wells (F-ab) | `tools/p3_numbers.py` B7, P3 S3.8 `[RAN]` |
| the flow's width rule and prior | `hidden_features` in $[52, 256]$ at $(26, 12)$ and $(26, 10)$, $[32, 256]$ on the bench; the coded rule `lo = clamp(2 max(p,E), 32, 64)`, `hi = max(8 max(p,E), 256)` differs from the recorded string at 127 of 128 values of `max(p,E)` (F-ad); log-uniform mass below 64 / 128: 0.130 / 0.565 on $[52, 256]$, 0.333 / 0.667 on $[32, 256]$ | `tools/p4_numbers.py` B1, P4 S3.3.1 `[RAN]` |
| the spline | $3 K_{\rm bins} - 1$ = 17 / 23 / 29 / 47 parameters per axis at 6 / 8 / 10 / 16 bins; width and height logits clipped to $\pm 3.454$, log-slopes to $\pm 6.908$ (bin ratios at most 1000, slopes in $[10^{-3}, 10^{3}]$, boundary slopes 1); narrowest bin 0.00143 / 0.00111 of the domain at 8 / 10 bins; domain $[-5, 5]$ in $\mathrm{logit}\,\theta$ is $\theta \in [0.00669, 0.99331]$; under the uniform prior 0.0134 of each axis's mass is in the tails, 0.296 of 26-axis rows and 0.126 of 10-axis rows have at least one axis there; the standard logistic's sd 1.814; at zero parameters the spline is the identity to $10^{-16}$ | `tools/p4_numbers.py` B2, P4 S3.2, S3.4 `[RAN]` |
| the conditioner and the weight count | `MaskedLinear` shapes (48, 38), (48, 48), (48, 48), (598, 48) at the runner's defaults and $(26, 12)$: $n_{\rm tf}$ hidden layers per conditioner (F-ac); $n_\omega$ = 107634 (runner 48/3/8), 340730 (library 64/5/10), 1742736 (standalone 128/8/10), 201032 / 1482632 / 11129688 at the space's lower corner, geometric middle and upper corner (52/4/10, 115/8/10, 256/12/10), 167960 at the lower corner as trained with 8 bins (F-a), 50946 / 53640 / 9643416 on the bench (48/3/8, 32/4/10, 256/12/10); live share 0.52-0.59; the chain's depth $n_{\rm tf}(n_{\rm tf} + 1)$ = 20 / 156 at 4 / 12 | `tools/p4_numbers.py` B3, B6, P4 Table P4.1 `[RAN]` |
| conditioner passes | log_prob: one per transform; a sample: $d_\theta$ per transform -- 3 against 78 at the runner's defaults, 12 against 312 at 12 transforms ($d_\theta$ = 26); 30 / 40 / 120 on the bench at 3 / 4 / 12 | `tools/p4_numbers.py` B4, P4 S3.2 `[RAN]` |
| the box anchor | the standard logistic density at $\mathrm{logit}\,\theta$ equals $\theta(1 - \theta)$ (residual $10^{-18}$), so $L_0$ = 0 on the unit cube; the identity-initialised flow's loss is $\mathrm{KL}(\text{logistic} \,\|\, \text{normal})$ = 0.5639 nats per axis (closed form and quadrature agree), 14.66 / 5.64 nats/row at $d_\theta$ = 26 / 10, Monte Carlo 14.63 +- 0.02 on 200000 rows; the box term is at most $-1.386\, d_\theta$ and averages $-2\, d_\theta$ | `tools/p4_numbers.py` B5, P4 S3.2 `[RAN]` |
| the optimiser's horizons and corrections (torch 2.10.0 replica) | averaging horizon $1/\upsilon_1$ = 10 / 100 steps at `one_minus_beta1` 0.1 / 0.01 (half-lives 6.6 / 69), $1/\upsilon_2$ = 1000 at the fixed 1e-3 (half-life 693); planned steps 250 (runner) / 1000 (`TrainConfig`) / 200 (pre-training); bias corrections at step 250: $1 - 0.9^{250}$ = 1.0000, $1 - 0.99^{250}$ = 0.919, $1 - 0.999^{250}$ = 0.221 (0.632 at 1000); the second moment's weights on the first and last gradient of a 250-step run 4.5e-3 and 3.5e-3; a common rescaling of the loss by $10^{3}$ moves a 250-step, 50-weight trajectory by $5 \times 10^{-10}$ against a norm of 0.11, an alternating per-step factor 1 / 0.1 by $3 \times 10^{-2}$ | `tools/p5_numbers.py` B1, P5 S3.2 `[RAN]` |
| the decoupled decay | per-step factor and cumulative factor over 250 steps: 0.99998 and 0.995 at the strongest corner $(2 \times 10^{-3}, 10^{-2})$ (horizon $5 \times 10^{4}$ steps; 0.980 over 1000 steps), 0.9975 at $(10^{-3}, 10^{-2})$ (horizon $10^{5}$), $1 - 2.5 \times 10^{-7}$ at the weakest corner (horizon $10^{9}$); torch's defaults on the pre-training: $(1 - 10^{-3} \cdot 10^{-2})^{200}$ = 0.99800, a 0.20 % shrink (F-q) | `tools/p5_numbers.py` B1, B4, P5 S3.3.3, S3.6 `[RAN]` |
| the clip | coefficient 1 / 1 / 0.1 / 0.01 at gradient norms 1 / 5 / 50 / 500; with a stale norm beside a flow norm, (5, 5) gives 0.707 and (1, 10) gives 0.498 (F-ah) | `tools/p5_numbers.py` B1, P5 S3.2 `[RAN]` |
| the schedule and the split | bench split at seed 0: 32 donors to 22 / 5 / 5 = 352 / 80 / 80 rows (hash `930462872c62...`); 383 groups to 268 / 57 / 58; at `b_sim` 128 / 256 / 512 / 1024: 3 200 / 6 400 / 12 800 / 25 600 rows per epoch, 32 000 / 64 000 / 128 000 / 256 000 per run, 1.54 / 3.09 / 6.17 / 12.35 pass-equivalents of 20 731 training rows, coverage per epoch 0.143 / 0.266 / 0.461 / 0.709 and per run 0.786 / 0.954 / 0.998 / 1.000, 9.1 / 18.2 / 36.4 / 72.7 visits per bench row per epoch; the plan's pass at 512 is 57.8 steps over the bank and 40.5 over the training rows; trim thresholds 5 120 / 10 240 / 20 480 rows (bench: `[256]`; DUP15HD: all three, 1024 by 251 rows); a 256 batch is 73 % of the bench's training rows | `tools/p5_numbers.py` B2, P5 S3.2, Table P5.1, S3.3.4 `[RAN]` |
| early stopping and the probe | earliest firing after 100 epochs at patience 99 (cannot at `epochs=10`), 6 at the library's 5, 21 at sbi's 20; the joint rule and sbi's `_converged` stop at the same epoch on 2000 / 2000 random curves; the probe leaves an `A2` run's simulated stream 10 batches (1 280 rows at 128) ahead of an `A1` run after ten epochs | `tools/p5_numbers.py` B2, P5 S3.2, S3.8 `[RAN]` |
| the optimiser surfaces | log-uniform mass of `lr` below 1e-3: 0.769, below 5e-4: 0.537; the DSN's `lr` top 0.2 is 100 times the joint 2e-3; `beta1` range [0.90, 0.99] with the runner's 0.9 on its edge; stream ratios 32 : 8 : 1 (runner), 64 : 8 : 1 (`BatchSpec`), 256 : 8 : 1 (1024 with the runner's `b_met`, `b_rep`) | `tools/p5_numbers.py` B3, P5 S3.1, S3.3 `[RAN]` |

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

[2026-10-02, D-054] The chat that wrote P0-P3 ended before the set was
complete, so the seven commits written so far (`b961784`, `308ccc0`,
`e0da315`, `922aac9`, `f9867f7`, `f37f4ca` and the commit carrying this
note) were delivered as one `git format-patch` series against the
re-fetched `origin/main` (`745edce`), applied by the user with `git am` and
pushed; the next chat forks `docs/joint-docs` from the pushed state and
continues under the same rule -- write, then one delivery per chat segment or
at the end, never per turn. The Stage 7 rebase-and-recheck of the whole
series stands. [2026-10-02, the P4 turn: the series is on `origin/main` as
`d991836` .. `ab7ea82`; `docs/joint-docs` was re-forked from `ab7ea82` and
the P4 commit is the first on top of it, pending the next delivery.]
[2026-10-02, the P5 turn: the P4 commit is `1f8e904`; the P5 commit is
the second on `docs/joint-docs` above `ab7ea82`, pending the same
delivery.]

[corrected 2026-10-01] v1 of this section described a per-turn delivery
(one tarball per turn, `git apply`, push, and a `git fetch` at the start of
the next turn to check the previous patch landed); D-052 replaced it with
the single delivery above. The fetch-before-build rule survives: the final
series is built against the re-fetched `origin/main`, never an assumed
state.
