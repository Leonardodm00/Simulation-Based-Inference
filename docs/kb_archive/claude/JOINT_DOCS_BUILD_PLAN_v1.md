# Joint DSN + NPE documentation set -- staged plan (parameter reference + learning set)

| date | change |
|---|---|
| 2026-10-02 | v1.3. D-054 (09:58, the documentation chat: "Do we need to change chat? if so deliver a handoff for the other chat and also all the files to push plus the code to do it"): the chat changes after P3, and the seven commits written so far on `docs/joint-docs` (`b961784` Stage 1, `308ccc0` P0, `e0da315` E0 and the notation checker, `922aac9` P1, `f9867f7` P2, `f37f4ca` P3, `c21cb31` index v1.6) leave as ONE `git format-patch` series, `joint_docs_series_2026-10-02.tar.gz` (sha256 `ca1bd248e5158cdf1d5f5e44dfcd8b0b7a55c963ae66d64eb830b7947b334f7b`), built against the re-fetched `origin/main` `745edce` and verified by `git am` in a fresh clone (14/14 files byte-identical; every gate re-run); the user applies it with `git am` on `main` and pushes, and the next chat forks from the pushed state. S2.8, Stage 7, S8 and Q10 gain a dated note: D-052's "one delivery at the end" is kept per chat segment. State recorded here, with the index as the ledger of record (`hpc/joint/docs/00_INDEX.md` v1.6): Stage 3 done 2026-10-01 (E0 v1.3, 256 declared symbols; `tools/check_notation.py` and `tools/smoke_test_notation.py`, 35/35); Stage 4 three of seven done (P1, P2, P3; P3 adds `tools/p3_numbers.py`); the next document is P4. The handoff for the next chat is `claude/HANDOFF_2026-10-02_joint_docs_P3_done.md`. No change to the stages' content, to S6 or to the findings table (its rows F-s..F-ab live in the index). |
| 2026-10-01 | v1.2. D-052 (16:16; first logged by this chat as "D-047" at 16:2x, renumbered after the log was re-read: D-047..D-051 are the Giulia chat's entries of about 16:10): "first write everything and then push" -- the per-turn patch delivery of v1.1 is replaced by ONE delivery at the end; S2.8, Stage 7 and the abstract are rewritten and the superseded text is kept as [corrected]. Stage 2 executed: `hpc/joint/docs/P0_PARAMETERS_OVERVIEW.md` (seven tables A-F, K; the five surfaces and the value chain; the shape anchors) with `tools/p0_tables.py` rendering and checking its three generated tables (tests T6.1-T6.6; suite 38/38 twice), `--n-post-draws` re-owned to P7, `inventory.json` and the index block regenerated, `00_INDEX.md` v1.1 (findings F-q, F-r added from the P0 reading; F-a's line reference corrected; S9 per D-052); commit `308ccc0` on the sandbox branch, not pushed. S1 and S4 updated accordingly; S8 records the Giulia project's shapes (D-039..D-046) as the first place the anchors will move; S9 gains Q10 (cadence of delivery, D-052). |
| 2026-10-01 | v1.1. Stage 0 closed: D-035 (the set lives in the repository under `hpc/joint/docs/`; the KB keeps `00_INDEX` and `P0`), D-036 (the DSN-inherited axes explained in full), D-037 (joint stack only), D-038 (fix patches for F-a, F-b: a new Stage 8); Q4, Q5, Q7 and Q9 stay at their defaults, listed in the decision log's Open calls. Stage 1 executed in the same turn: `hpc/joint/docs/00_INDEX.md`, `tools/inventory_joint_knobs.py`, `tools/smoke_test_inventory.py` (32/32 twice), `tools/inventory.json` (546 rows at `834eb41`), delivered as `joint_docs_stage1_20261001.tar.gz` (commit `b961784` in the sandbox clone, pending the user's push). The findings table of record moves to the index (S6 there; F-a..F-p); S4 here keeps ids and owners only. The deliverable tree (S1), the per-turn delivery protocol (S2.8), Stage 7 and the new Stage 8 (S5), the dependency diagram (S7) and the open points (S8) are updated; S9 lists what is decided and what is in force as a default. Numbering corrected: the Stage 0 answers are D-035 onwards, not D-033 (taken by a concurrent chat on 2026-10-01; D-033 narrows Stage C to the re-extraction record, which chapter E9 reflects). |
| 2026-10-01 | v1. Written after reading, today: the decision log (D-001..D-032, I-001); `JOINT_DSN_NPE_USAGE_v1.md` v1.3; `claude/DECK_PACK_BUILD_PLAN_v1.md` (the format precedent) and `claude/deck_pack/01_DECK_OUTLINE.md`; the repository `Simulation-Based-Inference` `main` @ `834eb41` (2026-09-30 15:33 +0200): `hpc/joint/README_joint.md`, `JOINT_DSN_NPE_PLAN_v0_6.md` S1, S2.1-2.2, S5, the heading lists of S6-S11, `stage2/{joint_model,joint_train,joint_losses,joint_batches,dsn_loss_adapter}.py` in full, `stage3/run_joint_arms.py` in full, `stage4/joint_space.py` in full, `stage4/npe_tune_joint.py` S1-S330, the four PBS scripts and the launcher, `stage1/{build_latent_bank,latent_nuisance,latent_gap,latent_realisation,bench_burst_provider}.py` (argument surfaces and spec classes), `hpc/dsn/config.py` (`SearchConfig` ranges), `hpc/dsn/backbone.py` (`BackboneConfig`), `hpc/dsn/condition_space.py` (API), the heading lists of `hpc/dsn/Documentation/TUNING_1_searched_axes.md`, `TUNING_2_fixed_knobs.md`, `CONFIG_REFERENCE.md`, and `hpc/npe_tune_search.py::default_space`. `joint_space.provenance_report` run in the sandbox `[RAN]`. Nothing in this plan is a result; it is a build order and a list of what must be decided. |

**Abstract.** Two document sets are requested. **Set P (parameters):** one
overview document that presents and explains every tunable parameter that
configures the joint DSN + NPE optimisation and training, and a related set of
documents that explain each parameter group comprehensively -- what the
parameter represents, where it lives, what it changes in the objective or the
optimiser, its available options and ranges, its interactions, and the failure
modes it produces. **Set E (environment):** a complete, pedagogical set that
explains the whole joint DSN + NPE environment plainly and analytically, fit
for learning, with the difficult topics (normalising flows and NPE, metric
learning and collapse, the replicate-consistency statistic, Bayesian
hyper-parameter search, grouped bootstraps) built up rather than asserted.
This plan fixes the deliverable tree, the shared conventions, the source
freeze, the findings the documents must carry, the stages (one document per
turn), the grounding work each document owes, the review that closes the
work, and the decisions taken. The set lives in the repository (D-035), is
written turn by turn on a sandbox branch, and is delivered once, as one patch
series, when complete (D-052) [corrected 2026-10-01: v1.1 said "delivered
turn by turn as patches"]. Deliberately excluded: any change to the
code outside the D-038 patch stream (findings are reported; other fixes are
separate decisions); any new run on the cluster; a re-derivation of the
replicate metric (`claude/METRIC_REPLICATE_v1_4.md` is transcribed, not
re-derived); the standalone DSN search and the standalone NPE tuner (D-037),
which get only a mapping section.

**Claim tags used throughout both sets:** `[KB]` project knowledge base
document; `[KB-PDF p.n]` a project PDF, full text, page given; `[REPO
file:line]` read from the repository at `834eb41`; `[RAN]` verified by
running code in the sandbox; `[PubMed full text]`; `[PubMed abstract only]`;
`[bioRxiv preprint, abstract only]` / `[bioRxiv preprint, full text]`;
`[textbook, from memory]`; `[reasoning]`; `[STATED]` decided by the user, with
the D-number. No number appears without one of `[KB]`, `[KB-PDF]`, `[REPO]`,
`[RAN]`, `[PubMed full text]` or `[STATED]`.

---

## 0. What exists already, and what the two sets add

Not duplicated, only pointed to:

| exists | where | what it is | what it does not do |
|---|---|---|---|
| `JOINT_DSN_NPE_PLAN_v0_6.md` (v0.6.5 in the repo) | repo `hpc/joint/` | the design: objective, arms, metric, bench, stages, decisions D1-D18 | not a reference of knobs; not written to be learned from |
| `JOINT_DSN_NPE_USAGE_v1.md` v1.3 | KB | how to RUN each stage on davinci; troubleshooting; verification state | lists the `-v` variables, not what they mean or do |
| `METRIC_REPLICATE_v1_4.md`, `FINITE_DRAW_CORRECTION_v1.md`, `NULL_TAIL_AND_PVALUE_v1.md` | KB `claude/` | the derivations behind the replicate term | derivations, not an introduction |
| deck pack `02`-`12` | KB `claude/deck_pack/` | slide content for a group-meeting audience, with figures F01-F20 (figures not in the KB) | written as slides, not as chapters; no exercises; no parameter reference |
| `TUNING_1_searched_axes.md`, `TUNING_2_fixed_knobs.md`, `CONFIG_REFERENCE.md`, `THEORY_joint_condition_search.md` | repo `hpc/dsn/Documentation/` | the STANDALONE DSN's 18 searched axes and fixed knobs, in the project's document format | the joint stack reuses ten of those axes with different defaults, a different base config and a different trainer; nothing documents that |
| `hpc/docs/npe_tuning_usage.md`, `npe_tune_search.py` | repo | the standalone NPE tuner | the joint tuner reuses its mechanics through an adapter; nothing documents the joint space |

The gap both sets fill: no document inventories the joint stack's tunable
surface (the 23 searched axes `[RAN]`, the runner's flags, the constants
hard-coded in `stage2/`, the search driver's knobs, the job variables -- now
546 rows in `hpc/joint/docs/tools/inventory.json` `[RAN]`) and no text
explains the joint environment to someone learning it. Since Stage 2, the
inventory half of that gap is closed by `P0_PARAMETERS_OVERVIEW.md`; the
explanations are Stages 4 and 5.

---

## 1. Deliverable tree (D-035)

Everything lands in the repository under `hpc/joint/docs/`, beside the stack
it documents; project knowledge keeps copies of two documents. Filenames are
fixed so documents can cross-reference each other from the first turn; ids
are stable, additions append.

```
hpc/joint/docs/
  00_INDEX.md                     reading map for both sets; status ledger (one row per doc:
                                  planned / drafted / reviewed, with the repo commit); the source
                                  freeze; the shared conventions; the findings table of record
                                  (F-a..F-r); the generated inventory block; the sandbox numbers;
                                  the delivery route                                   [also KB]
  tools/
    inventory_joint_knobs.py      AST/regex inventory of every tunable parameter; owner rules;
                                  --check-index against the index block
    p0_tables.py                  renders P0's three generated tables (A searched axes, F same knob
                                  different defaults, K alphabetical index) from the space and the
                                  inventory; --check-doc against P0's marker blocks     [Stage 2]
    smoke_test_inventory.py       38 checks: fixture tree, negative paths, determinism, index
                                  check, the real tree, the P0 tables (T6.1-T6.6)
    inventory.json                the extraction at the freeze commit (546 rows)
  -- Set P: parameters ------------------------------------------------------------------------
  P0_PARAMETERS_OVERVIEW.md       the first document asked for: every tunable parameter, one table
                                  per surface (A searched axes / B runner flags outside the space /
                                  C code constants / D search-driver knobs / E job variables),
                                  each with type, default(s), range and prior, the flag or field
                                  it reaches, its Status, pointer to its P-document; F "the same
                                  knob, several names and defaults"; K the alphabetical index;
                                  the five surfaces and the value chain campaign -> GP point ->
                                  config -> build_argv -> runner flags -> library  [also KB; done]
  P1_ENCODER_AXES.md              encoder block: depth_exponent, width_multiplier, block_family,
                                  embedding_size, head_fusion, dropout + the fixed BackboneConfig
                                  knobs (stem_width, group_width, head_pool_ops, GroupNorm, kernels)
  P2_DSN_LOSS_AXES.md             DSN-loss block: dsn_on, log10_lambda_dsn, loss_type,
                                  mining_strategy, margin, angular_alpha_deg, lambda_sep + fixed
                                  strict_semihard, sep_warmup_frac, swap; the legality projection
                                  and the activity mask; the margin unit conversion; the
                                  encoder-only pre-training of A0/A0s and its optimiser (F-q)
  P3_REPLICATE_AXES.md            replicate block: rep_on, log10_lambda_rep, warmup_frac_rep,
                                  n_posterior_draws + the constants jitter, t_floor, p_eff_min,
                                  correct_mc; the S_mc floor and the kappa_S bias (corrected story)
  P4_FLOW_AXES.md                 flow block: hidden_features, num_transforms + num_bins,
                                  z_score_theta, z_score_x; what each does inside a zuko NSF
  P5_OPTIMISER_AND_SCHEDULE.md    optimiser block: lr, one_minus_beta1, weight_decay,
                                  batch_size_npe (-> --b-sim) + epochs, steps_per_epoch, b_met,
                                  b_rep, grad_clip, patience, beta2, encoder_steps, seed, the
                                  grouped split fractions; the runner's defaults as a point
                                  outside the space (F-r)
  P6_SEARCH_DRIVER.md             Stage 4: campaigns S-A1/S-A2/S-A5/S-A25, the shape anchors
                                  (p, d_theta, embedding_dim, n_train), propose / evaluate /
                                  status / finalists / controls knobs, canonicalisation and the
                                  inactive pins, the ledger and trial ids, the control recipe;
                                  the mapping to the standalone DSN search and NPE tuner (D-037)
  P7_UPSTREAM_AND_JOBS.md         what shapes the problem before training: Stage 1 bank knobs
                                  (provider, arm, sizes, pi, gap modes, n_per_theta, nuisance
                                  scales); the PBS -v variables and resources of all four jobs;
                                  appendix: Stage 3b/3c flags (post-hoc, not training)
  -- Set E: the environment, for learning -----------------------------------------------------
  E0_READERS_GUIDE_NOTATION.md    how the chapters build; prerequisites; the MASTER notation table
                                  and glossary for both sets; the running example carried across
                                  chapters
  E1_THE_PROBLEM.md               from MEA recordings to posteriors over mechanism: the objects
                                  (theta, x, z, c, nu, G), the spaces and the maps (simulator,
                                  encoder, flow) with one column per space; the standing aim
                                  (patient-specific, label-free); the two data domains
  E2_NPE_AND_FLOWS.md             neural posterior estimation built up: Bayes without a
                                  likelihood, amortisation, the NLL objective and what it bounds,
                                  normalising flows from the change of variables to rational-
                                  quadratic spline coupling/autoregressive transforms (zuko NSF),
                                  z-scoring and the unconstrained transform, what sbi provides
  E3_THE_SUMMARY_NETWORK.md       the DSN: why a learned summary, the RegNet-style GroupNorm 1D
                                  CNN, metric learning (triplet, angular, ETF separation, mining),
                                  the information argument c -> theta -> x -> z, the collapse
                                  result and the log C ceiling, r_eff
  E4_JOINT_OBJECTIVE_AND_LOOP.md  eq. (1)-(2); three streams, two domains; gradient reach and the
                                  two stop-gradients; why an explicit AdamW loop; early stopping
                                  on the NPE validation score only; the nine arms as an ablation
                                  design and what each isolates
  E5_REPLICATE_CONSISTENCY.md     from two wells to one number: T, the metric M, the target
                                  p_eff, the composite null H0, the finite-draw correction and the
                                  residual kappa_S bias, the three escape routes and their blockers,
                                  the per-direction decomposition -- transcribed from the KB
                                  derivations with a worked numeric example at d_theta = 26
  E6_THE_BENCH.md                 the generator phi -> burst parameters -> spike trains -> IFR;
                                  arms S and R from one generator; the gap knob pi; the nuisance
                                  latent nu; the realisation latent G; providers; the shard and
                                  sidecar contract; what the bench can and cannot adjudicate
  E7_DIAGNOSTICS_AND_DECISION.md  L, L0, Delta_hat; r_eff; the p_eff spectrum; per-axis
                                  contraction; cluster scores; the paired cluster bootstrap and
                                  DEFF; the decision rule; Stage 3b's decomposition; Stage 3c's
                                  floors, aliasing, stratification; gates G1-G3 and why they need
                                  theta*
  E8_HYPERPARAMETER_SEARCH.md     Bayesian optimisation with a GP surrogate built up: surrogate,
                                  acquisition, batching by constant liar, the stateless optimiser
                                  replayed from the ledger, canonicalisation and legality, nested
                                  campaigns and partial dependence, per-finalist controls with
                                  Holm, escalation and boundary axes; the mapping to the
                                  standalone NPE tuner (D-037)
  E9_OPERATING_AND_STATUS.md      orientation to running it (pointers into JOINT_DSN_NPE_USAGE_v1,
                                  not a copy); what has and has not run; the open decisions
                                  (D1-D4, D12-D16, kappa_S); how D-031 (retrain with
                                  optimisation), D-033 (Stage C ends at the re-extraction record),
                                  D-039..D-046 (the Giulia project trains arm A1 through S-A1)
                                  and I-001 relate to the joint stack

claude/joint_docs/                project knowledge: copies of 00_INDEX.md and P0, same turn
```

Nineteen documents plus the tools: the index, eight in Set P, ten in Set E.

---

## 2. Shared conventions (fixed now, applied in every document)

1. **Skeleton.** The project's mandatory topic-document structure: 0 header
   (title, date, abstract with scope and exclusions); 1 notation and symbols
   table (Symbol | Name / Meaning | Type & domain | Units | First used in S)
   with a Conventions subsection; 2 glossary (ordering stated); 3 main body,
   each section opening with what it establishes; 4 summary of results with
   cross-references; 5 open points, caveats, assumptions; 6 references, with
   anything from memory marked. E0 holds the master table; every other
   document carries its own table restricted to the symbols it uses, byte-
   consistent with E0 (checked by the Stage 6 script).
2. **Set P, per parameter**, the same fields in the same order: name and
   symbol | where it lives (search axis / runner flag / dataclass field /
   module constant / job variable, with `[REPO file:line]`) | type and domain
   | default, and EVERY other default the same parameter has elsewhere in the
   code (the inventory's "same knob, different defaults" table) | range and
   prior if searched, with its provenance | Status (configured / analytic /
   computed / measured / derivation-only; the Provenance model) | what it
   represents, plainly | what it changes in the objective or the optimiser,
   analytically | legality and interactions | failure modes and the
   diagnostic that reveals them | source tag. The ten DSN-inherited axes get
   the full treatment plus a differences table and a pointer to `TUNING_1`
   / `TUNING_2` (D-036).
3. **Set E, per section**: the plain mechanism first (invariant + consequence,
   real objects with their types), then the analytic treatment with numbered
   equations in full notation; one running example carried across chapters
   (the DUP15HD shapes: d_theta = 26, the 17-log / 9-linear coordinates, the
   search anchors p = 26, E = 12, S_mc = 128 `[REPO]`), switched to the bench
   shapes only where said; a "common confusions" list where a word carries two
   senses (R6); "check yourself" questions with folded answers (Q5 default);
   sections hand off with one bridging sentence (Style model pattern 6).
4. **The Precision model's pre-send check** (R1-R8) runs on every document and
   its outcome is recorded in a one-line footer. Types before relations;
   every "is" with its hypothesis; no transplant without the check (the
   kappa_S case is the standing example); one name per object and one object
   per name, with the level stated where an object exists on two levels
   (analytic `C` vs computed `C_hat`; `m_g` vs `m_hat_g`).
5. **Numbers.** Only with a tag from the list above; the no-numbers-from-
   abstracts rule and the metadata-is-not-data rule apply unchanged. A number
   that can be reproduced in the sandbox is reproduced and tagged `[RAN]`
   (resolved ranges, parameter counts, kappa_S at given S_mc).
6. **Hygiene.** ASCII only, LF only, math in `$...$`; no unicode arrows or
   dashes. Diagrams as Mermaid blocks with an ASCII fallback for the key ones
   (Q4 default). Equation numbers of the plan and of the metric document are
   kept, never renumbered.
7. **One document per turn.** Each turn: re-read the KB documents the chapter
   depends on; run the grounding searches it owes (S6) and report them,
   including empty returns; run any sandbox check that produces a `[RAN]`
   number; write the document; update its status row in `00_INDEX.md`;
   commit on the sandbox branch; end the reply naming the documents written.
8. **Delivery (D-035, D-052).** The files are written in the sandbox clone on
   the branch `docs/joint-docs`, cut from `main` @ `834eb41`, each turn's
   commit on top of the previous one. **Nothing is applied or pushed until
   the set is complete** (D-052, 2026-10-01): the Stage 1 tarball sent before
   D-052 is not to be applied on its own. Every turn still runs the gates of
   the `hpc-git-delivery` skill on its files (`py_compile`; `args.*`
   resolution; byte safety -- pure ASCII, LF only, for every `.py` and `.md`;
   the behavioural smoke test; the generated blocks checked against the code
   by `--check-index` and `--check-doc`). At Stage 7, `origin/main` is
   re-fetched, the branch rebased if `main` moved, the whole series
   re-checked (every patch applied in a fresh clone and compared byte for
   byte with the tested files), and ONE tarball holding the `git
   format-patch` series is sent as a file with the laptop commands (extract,
   `git am` or `git apply` per patch, review, `git push`). The documents land
   on `main` as documentation-only commits (default); the D-038 code patches
   go on a branch of their own (default). The KB copies of `00_INDEX.md` and
   `P0` are written in the same turn as the repository versions.
   [corrected 2026-10-01] v1.1 of this item described a tarball per turn with
   `git apply` and push each time, and a `git fetch` at the next turn's start
   to check the previous patch landed; D-052 replaced it with the single
   delivery above. The fetch-before-build rule survives at Stage 7.
   [2026-10-02, D-054] "Until the set is complete" is kept per chat segment:
   when the chat changes before the set is complete, the commits written so
   far leave as one `git format-patch` series built against the re-fetched
   `origin/main` and verified by `git am` in a fresh clone of it, with the
   laptop commands (`git am` on `main`, review, `git push`); the next chat
   forks `docs/joint-docs` from the pushed state and continues under the
   same rule. The first such series is `joint_docs_series_2026-10-02.tar.gz`
   (commits 1-7, P0-P3 and E0 with the index and the tools). After `git am`
   the commits on `main` carry new hashes: the index's commit column is
   filled from `git fetch`, never from the sandbox hashes.

---

## 3. Source freeze

- Repository `Simulation-Based-Inference` `main` @ `834eb41`, 2026-09-30;
  re-fetched 2026-10-01 at the start of Stage 1, unchanged `[RAN]`. The joint
  plan in the repo is **v0.6.5** (changelog read `[REPO]`).
- KB documents as of 2026-10-01: `JOINT_DSN_NPE_USAGE_v1.md` v1.3;
  `claude/METRIC_REPLICATE_v1_4.md`; `claude/FINITE_DRAW_CORRECTION_v1.md`;
  `claude/NULL_TAIL_AND_PVALUE_v1.md`; the deck pack; `SBI_PIPELINE.md`;
  `EXTRACTOR_USAGE.md`; `HPC_PATHS.md`; the handoffs; the decision log
  v1.12. D-033 (2026-10-01 14:22) narrows Stage C to the re-extraction
  record: no embedding export before the retrain; E9 states it so.
  D-039..D-051 (2026-10-01 15:35 and about 16:10, the Giulia project) fix that the joint
  stack is trained there on arm `A1` through campaign `S-A1`, at shapes of
  its own (`claude/PLAN_2026-10-01_giulia_hpc_stages.md`); E9 and P6 state
  it, P0 S5 records it.
- DSN documentation in the repo: `TUNING_1` revision 3 (18 axes),
  `TUNING_2`, `CONFIG_REFERENCE`.
- **Known staleness, to be carried not hidden.** "Plan v0.6.6" (the kappa_S
  correction named by `JOINT_DSN_NPE_USAGE_v1.md` v1.3 and
  `METRIC_REPLICATE_v1_4.md`) is **not** in the repository at `834eb41`, and
  `joint_space.py:276-281` still states that `n_posterior_draws` "controls
  variance and never bias" `[REPO]`. Both sets state the corrected claim from
  `METRIC_REPLICATE_v1_4.md` S3.13.1 `[KB]` and flag the repository text as
  superseded (Q9 default). `JOINT_DSN_NPE_DEVIATIONS_v1.md` and
  `HANDOFF_D17_option_c.md` are in neither the repository nor the KB (`find`
  over the clone; usage doc S10), so nothing is cited from them.

---

## 4. Findings from the reading, carried by Set P

The table of record is `hpc/joint/docs/00_INDEX.md` S6 (and its KB copy
`claude/joint_docs/00_INDEX.md`): eighteen findings F-a..F-r, each with its
evidence (`[REPO file:line]` or the inventory's "same knob, different
defaults" table), its owner document and its resolution. Here only the ids
and owners, so this plan cannot drift from the index:

| id | in one line | owner | resolution |
|---|---|---|---|
| F-a | `num_bins` 8 trained vs 10 declared in a Stage 4 trial | P4, P6 | patch (D-038) |
| F-b | `head_pool_ops` code 0 built vs code 1 declared | P1, P6 | patch (D-038) |
| F-c | `RANGE_PROVENANCE` names `SearchConfig` for `weight_decay` and `dropout`; the values are `RegularizationConfig`'s | P1, P5 | report; open |
| F-d | `patience=99`: early stopping cannot fire under 99 epochs | P5, P6 | report; open |
| F-e | the `n_posterior_draws` floor's stated reason superseded by the kappa_S derivation | P3, E5 | report; open (log) |
| F-f | stale margin-0.3 docstring in `canonicalise_config` | P2 | report; open |
| F-g | two sources for the DSN-inherited ranges (dataclass vs JSON) | P1, P2, P5 | report |
| F-h | the joint base loss differs from the standalone DSN's | P2 | report |
| F-i | constants configured by code, never by flag | P3, P5 | report |
| F-j | resolved ranges and free-axis counts at the DUP15HD and bench shapes `[RAN]` | P4, P6 | report |
| F-k | two dates' test counts in README vs usage | E9 | report |
| F-l | `batch_size_npe` reaches the runner as `--b-sim`; `b_met`, `b_rep` unsearched | P5 | report |
| F-m | library defaults vs runner defaults for the same knobs (TrainConfig, BatchSpec, build_joint_model, train_joint, BackboneConfig) | P1, P4, P5 | report |
| F-n | `stage3c.pbs` `N_POST_DRAWS=128` vs `--n-post-draws` default 64 | P7 | report; open |
| F-o | `JointSpaceSpec.n_posterior_draws` dataclass default `(100, 400)` is below the floor at d_theta = 26 | P3, P6 | report |
| F-p | `warmup_frac_rep` 0.3 in the runner vs 0.0 pinned and in the loss default | P3 | report |
| F-q | the encoder-only pre-training of `A0`/`A0s` builds AdamW on torch's own `weight_decay` and `betas`, not the runner's `--weight-decay` / `--one-minus-beta1` (added at Stage 2) | P2, P5 | report; open (candidate for the D-038 stream) |
| F-r | four runner defaults lie outside the searched ranges (`--b-sim 128`, `--hidden-features 48` at the DUP15HD shapes, `--num-transforms 3`, `--weight-decay 0.0`): a Stage 3 arm at default hyper-parameters is not a point of the Stage 4 space (added at Stage 2) | P0, P4, P5, P6 | report |

---

## 5. Stages

Sizes are effort per turn; "one turn" means one reply of this conversation.

### Stage 0 -- Decisions (done 2026-10-01)

D-035 (location), D-036 (DSN-inherited axes in full), D-037 (joint stack
only), D-038 (fix patches for F-a, F-b). In force as defaults, overridable
at any turn and listed in the decision log's Open calls: Q4 Mermaid
diagrams with an ASCII fallback; Q5 three to six "check yourself" items per
Set E chapter, answers folded; Q6 English; Q7 one document per turn in the
order of S7; Q9 cite the KB for the kappa_S correction and flag the repo plan
as v0.6.5. D-052 (16:16) added the cadence of delivery: one push at the end
(Q10 in S9).

### Stage 1 -- Inventory ledger and the checker script (done 2026-10-01)

Delivered: `hpc/joint/docs/00_INDEX.md` (reading map, status ledger, freeze,
conventions, findings F-a..F-p, the generated inventory block, the sandbox
numbers, the delivery route); `tools/inventory_joint_knobs.py` (AST
extraction of every `add_argument` with its subcommand context, every
keyword default of the named functions and `__init__`s, every field of the
named dataclasses, every ALL_CAPS constant, every job-script variable and
`#PBS` directive; owner assignment by an explicit rule table with no
catch-all; `--check-index`); `tools/smoke_test_inventory.py` (32 checks on a
fixture tree and on the real tree, 32/32 twice `[RAN]`); `tools/inventory.json`
(546 rows at `834eb41`: cli 164, signature 153, dataclass 82, constant 46,
job_var 101; 46 knobs with more than one default). Gates 1-6 passed; the
patch re-applied on a fresh clone of `834eb41` reproduces the tested bytes.
Commit `b961784` on the sandbox branch. The tarball sent that turn is
superseded by the single delivery of D-052 and is not to be applied alone.
The backbone parameter count at the runner's defaults was not computed (no
torch in the sandbox); usage v1.3 S8's 359708 `[CLUSTER]` stands.

Acceptance met: every parameter the script finds has a ledger row with an
owner; the index block equals a fresh extraction (`--check-index` OK).

### Stage 2 -- `P0_PARAMETERS_OVERVIEW.md` (done 2026-10-01)

Written: the five surfaces and the value chain (S3.1, Mermaid with ASCII
fallback); Table A, the 23 searched axes with type, range at the DUP15HD
shapes `[RAN]`, prior, the three campaigns' pins, the flag reached and the
runner default (generated); Table B, the eleven runner flags outside the
space plus the plumbing, with the F-r observation; Table C, nineteen rows of
constants configured by code; Table D, the driver's knobs; Table E, the job
variables; Table F, the same knob with different defaults (generated; 28
knobs after the plumbing rows and the range-bearing dataclasses are dropped,
against the index's 46 over all rows); the upstream pointers (S3.8); Table K,
the alphabetical index of 182 owned knobs (generated). Findings F-q and F-r
came out of the reading and went into the index. `tools/p0_tables.py`
renders A, F and K from `joint_space` and `npe_tune_joint` (imported,
torch-free) and from the inventory, and `--check-doc` compares P0's three
marker blocks with a fresh rendering; tests T6.1-T6.6 cover it (38/38 twice
`[RAN]`). Every `file:line` in the prose was re-read against the clone
before commit (two control passes; a dozen line references and one
claim -- the pre-training optimiser's weight decay -- corrected in the
second). Commit `308ccc0` on the sandbox branch; KB copies written.

Acceptance met in the form the tool gives it: every owned, non-plumbing knob
of `inventory.json` appears once in Table K (T6.6), the three generated
blocks equal a fresh rendering (`--check-doc` OK x3), every searched axis
reaches a runner flag (Table A), the pre-send check footer is present. The
acceptance as first written ("every non-plumbing row appears in P0") is met
through Table K rather than through the per-surface tables, which list
knobs, not rows.

### Stage 3 -- `E0_READERS_GUIDE_NOTATION.md` (one turn; next) [done 2026-10-01; "next" corrected 2026-10-02: commit `e0da315`, E0 at v1.3 after P1-P3; the ledger is the index]

Blocking for both sets: the master notation table (the plan's S1 symbols
with its v0.6 collision repairs, plus the search-space symbols of Set P and
the level-separated symbols of R8), the glossary ordered by first appearance
in the E-set, the reading map (which chapter needs which), the prerequisites,
and the running example.

Acceptance: every symbol used in P0 is in the master table; the Stage 6
checker finds no symbol in any later document that is absent here.

### Stage 4 -- Set P deep dives, `P1`..`P7` (seven turns) [2026-10-02: P1 `922aac9`, P2 `f9867f7`, P3 `f37f4ca` done 2026-10-01/02; P4 next, in the next chat]

Order P1, P2, P3, P4, P5, P6, P7. Each turn owes the grounding of S6 for the
named methods it introduces, the sandbox checks for its `[RAN]` numbers, and
the findings rows it owns (S4).

Acceptance per document: every parameter of its block has all S2.2 fields;
every default is quoted with `[REPO file:line]`; every differing default is
listed; the pre-send check footer is present.

### Stage 5 -- Set E chapters, `E1`..`E9` (nine turns)

Order E1 to E9. Each chapter transcribes the KB derivations it needs and
builds up the rest; each carries the exercises of Q5 and the common-confusions
list; each cites full texts only (S6).

Acceptance per chapter: one thread from its opening discrepancy to its closing
line; every equation numbered as in its source; every borrowed phrasing with
its caveat; the pre-send check footer.

### Stage 6 -- Review by agents that have not seen the work (one to two turns)

1. **Fidelity review** of Set P against the code at `834eb41` and of Set E
   against the KB sources: a fresh agent lists every difference (value, line,
   equation, tag, status) and every symbol used in two senses; findings fixed,
   re-checked.
2. **Consumer dry run**: a second fresh agent, given only the two sets,
   answers a fixed list of questions (ten on parameters, ten on the
   environment -- e.g. "which flag does `batch_size_npe` reach, and what does
   a control run differ in", "why is the replicate target not zero") without
   any other source; what it had to invent is a gap, what it got wrong is a
   clarity defect; both fixed.
3. **Scripted checks**: notation coverage against E0; `--check-index`;
   `--check-doc` on P0; ASCII/LF on every file.

Acceptance: both reviews return an empty list on the second pass.

### Stage 7 -- Closing and the single delivery (one turn)

The status ledger of `00_INDEX.md` set to `reviewed`; `origin/main`
re-fetched and the branch rebased if it moved; the whole series re-checked
(every patch applied in a fresh clone, bytes compared with the tested
files); ONE tarball with the `git format-patch` series sent as a file with
the laptop commands; this plan re-issued as executed. The repository is the
copy of record; the status ledger's commit column is filled in once the push
is seen from the sandbox. [corrected 2026-10-01] v1.1 of this stage sent a
zip of `hpc/joint/docs/` for convenience and relied on per-turn delivery;
under D-052 the series is the delivery and the zip is dropped.
[2026-10-02, D-054] The closing delivery is the series of the LAST chat
segment, from the last pushed state; the series of 2026-10-02 (commits 1-7)
is the first segment's. The re-fetch, the rebase if `main` moved and the
fresh-clone `git am` check are run for every series, not only the last.

### Stage 8 -- The D-038 patch stream (one to two turns; after P6 by default, earlier on request)

Separate from the documents, on its own branch: fix patches for F-a and F-b
with smoke tests, through gates 1-6 and a cluster dry run before any
submission. The design is prepared after P6 (which analyses `build_argv`,
the fixed values and the legality projection) and put to you with a
recommendation on the one open part -- whether the Stage 3 runner defaults
move to the space's fixed values (`--num-bins` 8 -> 10; `head_pool_ops` code
0 -> 1), which changes what the arms train at "default hyper-parameters", or
only the tuner passes them. The minor findings F-c, F-d, F-f, F-n and now
F-q are listed in the same proposal as optional items (Open calls). After a
patch lands, usage v1.3 S5.2 and plan v0.6.5 S5.1 get a note. Whether the
D-038 patches join the single delivery of D-052 or travel separately is a
"does not decide" of D-052 (default: separately).

---

## 6. Grounding owed by each document

Knowledge base first, then PubMed, then bioRxiv (published-version lookup
before citing a preprint), then repositories where a claim concerns data. A
search that returns nothing is reported as such in the document's S6. Full
text only for any claim; abstract-only flagged in the same sentence; no
numbers from abstracts. P0 owes none: it names methods only as the names of
the code's objects and says so in its S6; the grounding of each method is
its owner document's.

| document | KB sources (full text available in the project) | named methods to search (PubMed, then bioRxiv) |
|---|---|---|
| P1, E3 | `SBI_PIPELINE.md`; deck `02`, `04`; Goncalves et al. 2020 (embedding nets) `[KB-PDF]`; BayesFlow `[KB-PDF]` | GroupNorm; RegNet design spaces; ResNeXt grouped convolutions; dropout; triplet loss / semi-hard mining; angular loss; neural collapse and the simplex ETF; Barber-Agakov bound |
| P2 | `TUNING_1` S3.9-3.17, `THEORY_joint_condition_search.md` (repo) | as E3, plus the equiangular tight frame literature |
| P3, E5 | `METRIC_REPLICATE_v1_4.md`, `FINITE_DRAW_CORRECTION_v1.md`, `NULL_TAIL_AND_PVALUE_v1.md`; deck `05`; Sloppiness `[KB-PDF]` | Hotelling's T^2 and the Mahalanobis form; the inverse-Wishart bias of an inverted sample covariance; the Gil-Pelaez inversion; Satterthwaite |
| P4, E2 | Normalizing Flows review `[KB-PDF]`; Practical Guide `[KB-PDF]`; APT `[KB-PDF]`; sbi toolkit `[KB-PDF]`; Flow Matching `[KB-PDF]` (as the alternative, one paragraph) | neural spline flows (rational-quadratic splines); coupling vs autoregressive transforms; z-scoring / unconstrained transforms in sbi |
| P5, E4 | plan S2.2, S2.4; deck `03`, `08` | AdamW decoupled weight decay (and its library defaults, F-q); gradient clipping; early stopping; the (1 - beta) reparameterisation |
| P6, E8 | `npe_tune_search.py`, `npe_tuning_usage.md`, `HANDOFF_DELTA_MIN_PER_CONFIG_v1.md` (repo); `TUNING_2` S3.1-3.2 | Bayesian optimisation with GP surrogates; expected improvement / acquisition; constant-liar batching (kriging believer); Holm's step-down; partial dependence |
| P7, E6 | plan S4; `STAGE_A_BENCH_GENERATOR_SPEC_v1` if present in the repo (not yet located); deck `07`; Multilevel neuronal SBI `[KB-PDF]` | Poisson network-burst models; truncated-normal label axes (textbook); nested random effects |
| E1 | `SBI_PIPELINE.md`, `EXTRACTOR_USAGE.md`, `HPC_PATHS.md`; Frontier of SBI `[KB-PDF]`; Practical Guide `[KB-PDF]`; deck `02` | amortised vs sequential inference; covariate shift |
| E7 | Trust Crisis, Diagnostics (conditional density), TARP, Misspecification detection, Compositional SBI `[KB-PDF]`; deck `08`; `hpc/docs/diagnostics_reference_handoff.md` (repo) | simulation-based calibration; TARP coverage; design effect (Kish); paired cluster bootstrap |
| E9 | usage v1.3; the Stage C, C8 and Giulia handoffs; the decision log | none (status, not claims) |

The biomedical indexing of PubMed is a fact about what it returns, not a
reason to skip a query; every row above is queried and the return, empty or
not, is recorded.

---

## 7. Dependencies and order

```
Stage 0 (done) -> Stage 1 (done) -> Stage 2 (P0, done) -> Stage 3 (E0, next)
                                                              |
                                            +-----------------+-----------------+
                                            v                                   v
                                   Stage 4 (P1..P7, 7 turns)          Stage 5 (E1..E9, 9 turns)
                                            |        |                          |
                                            |        +--> Stage 8 (D-038 patches, after P6)
                                            +-----------------+-----------------+
                                                              v
                                                 Stage 6 (two reviews + scripts)
                                                              v
                                       Stage 7 (closing: the ONE delivery, D-052)
```

Default order of the remaining turns: E0, P1..P7, E1..E9, Stage 8, review,
closing: about nineteen turns. Stages 4 and 5 do not depend on each other
after E0; the order between them can be changed at any turn (Q7).

---

## 8. Open points, caveats, assumptions

- **Nothing in `hpc/joint/` has run on the cluster** (usage v1.3 S9; README
  `[REPO]`). Every statement about behaviour is read from source or from the
  plan, and both sets say so; no "impact" claim is a measurement.
- **The defaults may move.** D-031 (the DSN retrain "with optimization") and
  I-001 (the Giulia project) can change the DSN-side base configuration, the
  shapes (`d_theta`, `p`, `E`) and the campaign set. The Giulia project is the
  first concrete case: arm `A1` through `S-A1` (D-039), windows of 180 s at
  `W = 18000` (D-041), one electrode per subregion (D-043), a label freeze
  with `p = 23` (`claude/PLAN_2026-10-01_giulia_hpc_stages.md` S4 `[KB]`), its
  joint configuration "afterwards" (D-039). Set P is written against the
  shape anchors as parameters, with the resolved values at the DUP15HD shapes
  given as `[RAN]` examples, so a change of shape is a re-run of
  `default_joint_space` and of the checkers, not a rewrite; `--check-index`
  and `--check-doc` are the tripwires.
- **Code findings are not fixed in the document stream** (S4); F-a and F-b
  are the D-038 stream (Stage 8); the rest are Open calls (F-q joins them).
- **Capacity.** With D-035 the knowledge base receives two documents only
  (`00_INDEX`, about 79 KB with its inventory block, and `P0`, about 72 KB),
  so the cap of v1's S8 no longer binds.
- **Figures.** The deck pack's F01-F20 were delivered as a zip and are not in
  the KB or the repository; both sets reference them by id only if you still
  hold the zip. Q4's default draws Mermaid diagrams inline.
- **The usage reference is not duplicated.** E9 and P7 point into
  `JOINT_DSN_NPE_USAGE_v1.md` for commands, dry runs and troubleshooting.
- **Branch.** Documents on `main`; the D-038 patches on a branch of their
  own (D-035 and D-052, "Does not decide"; Claude's defaults).
- **The sandbox branch is the only copy of the work until Stage 7.** Under
  D-052 nothing reaches the repository before the end; the KB copies of the
  index and of P0 are the only off-sandbox copies, and the status ledger's
  "pending push" rows stay so until the single delivery. If the sandbox is
  lost before Stage 7, the KB copies and the per-turn replies are what
  survives; the tools and P1-E9 would have to be rewritten.
  [2026-10-02, D-054] Narrowed: a series leaves at every chat change, so
  what is at risk is one chat segment's work, not the whole set. As of
  2026-10-02 the risk window closes once the user applies and pushes
  `joint_docs_series_2026-10-02.tar.gz` (the tools, E0, P0-P3, the index);
  the tarball itself is the off-sandbox copy until then.
- **Assumed:** English; the reader is you (PhD level) and later a lab-mate;
  the `Explanation modes (v8)` instruction's Style, Provenance, Precision and
  Grounding models are the writing standard for Set E.

---

## 9. Decisions and defaults

| id | question | status |
|---|---|---|
| Q1 | where the documents live | **D-035**: repository `hpc/joint/docs/`; KB keeps `00_INDEX` and `P0` |
| Q2 | the ten DSN-inherited axes | **D-036**: fully, self-contained, with a differences table and pointers |
| Q3 | scope | **D-037**: the joint stack only; one mapping section in P6/E8 |
| Q4 | diagrams | default in force: Mermaid blocks inline, ASCII fallback for the key ones |
| Q5 | exercises in Set E | default in force: three to six per chapter, answers folded |
| Q6 | language | default in force: English |
| Q7 | cadence and order | default in force: one document per turn, order of S7 |
| Q8 | the code findings F-a, F-b | **D-038**: fix patches prepared (Stage 8); alignment direction open |
| Q9 | plan v0.6.6 | default in force: cite the KB for kappa_S; flag the repo plan as v0.6.5 |
| Q10 | cadence of delivery | **D-052**: everything written first on `docs/joint-docs`, ONE push at the end (Stage 7); the Stage 1 tarball not applied alone; branch `main` and separate D-038 patches as defaults. **D-054** (2026-10-02): one series per chat segment as well -- at a chat change the commits so far leave as one `git format-patch` series the user applies with `git am` and pushes; the next chat forks from the pushed state |
