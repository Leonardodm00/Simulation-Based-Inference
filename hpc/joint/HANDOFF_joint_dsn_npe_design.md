# HANDOFF -- joint DSN+NPE session (2026-08-28 -> 09-02)

**Purpose.** Everything decided, built, corrected and left open in the session
that designed the joint DSN+NPE programme. Written to be pasted into a new
chat. Append to the changelog at the bottom rather than editing history.

**Confidence legend** (extends the one in `HPC_PATHS.md`):

- **[KB]** -- read from the project knowledge base this session.
- **[REPO]** -- read from repository source this session (tarball of the named
 branch, not the cluster checkout).
- **[RAN]** -- a command was run in the analysis sandbox and its output seen.
 *Not* the cluster: nothing in this session touched the HPC.
- **[DERIVED]** -- my reasoning, not read anywhere. Check before relying on it.
- **[STATED]** -- asserted by the user, not independently verified.
- **[OPEN]** -- not established; decision or measurement outstanding.

---

## Changelog

| date | change |
|---|---|
| 2026-09-02 | This handoff written. Smoke tests re-run [RAN]; two KB numeric inconsistencies found (S9). |
| 2026-09-01 | `METRIC_REPLICATE_v1.md`, `PROJECT_IDEAS.md`, `EXPLANATION_MODES.md`. Three corrections to earlier metric claims. |
| 2026-08-31 | `GATES_v1.md`. Four docstring-vs-code discrepancies found (S11). |
| 2026-08-30 | Plan v0.3 -> v0.4. Drift argument withdrawn. `bootstrap_paired.py` + smoke test. |
| 2026-08-29 | `INFO_LOSS_THEORY_v1.md` -- label ceiling, `r_eff = C-1`, cardinality-vs-dimension. |
| 2026-08-28 | Plan v0.1 -> v0.2. Repos read; two-domain asymmetry identified. |

---

## 1. Documents produced, and which are current

All in `/mnt/user-data/outputs/` [RAN, listed]. All ASCII/LF-clean.

| file | role | status |
|---|---|---|
| `JOINT_DSN_NPE_PLAN_v0.4.md` | the staged plan: objective, arms, bench, search, Stages 0-8, decisions D1-D14 | **current** |
| `JOINT_DSN_NPE_PLAN_v0.3.md` | superseded by v0.4 | keep for history only |
| `JOINT_DSN_NPE_PLAN_v0.2.md` | superseded; **contains the withdrawn drift argument** (S6) | do not cite |
| `INFO_LOSS_THEORY_v1.md` | Propositions 1-15: the identity, label ceiling, `r_eff = C-1`, cardinality-vs-dimension | current; **S3.7.2 / Prop 12 superseded**, see S6 |
| `GATES_v1.md` | G1/G2/G3 as implemented, what each misses, four code discrepancies | current |
| `METRIC_REPLICATE_v1.md` | the metric in the replicate statistic; three corrections to earlier claims | current |
| `PROJECT_IDEAS.md` | parked ideas: IDEA-001, IDEA-002 | living |
| `EXPLANATION_MODES.md` | instruction block for project settings: comprehensive vs brief | to be pasted into project instructions |
| `bootstrap_paired.py` | paired bootstrap, four resampling schemes, ICC, tail diagnostic | delivered, verified |
| `smoke_test_bootstrap_paired.py` | 10 fast tests + coverage test, validated against analytic truth | delivered, verified |

**Not yet folded in:** the corrections in `METRIC_REPLICATE_v1.md` S5 and the
A5 rewrite discussed at the end of the session are **not** in plan v0.4. A
v0.5 is owed -- see S12.

---

## 2. The question the project is now answering

It started as "does training the DSN and NPE jointly beat training them
separately". It became broader, and the broader form is the paper:

> **Is the diagnosis the right teacher for a mechanistic summary statistic,
> when real data offer labels but no parameters?**

Three deliverables serve one claim: a summary statistic for mechanistic
inference should be supervised by the simulator's parameters and by the
experiment's replicate structure, **never by the diagnosis**, and doing so
recovers within-diagnosis mechanistic heterogeneity that the
diagnosis-supervised statistic provably destroys.

1. **Theory** -- the label ceiling (S4).
2. **Bench + real data** -- nuisance/mechanism separation, stratification in
 parameter space (S4).
3. **Validation** -- drug response as an external test, if such data exist
 [OPEN, D14].

The standing aim behind all of it is IDEA-001: **drop the discrete diagnostic
classification entirely; the pipeline should be patient-specific and
bias-free.** The label moves from *teacher* to *referee*.

---

## 3. Arms

| arm | encoder psi fitted by | flow omega | note |
|---|---|---|---|
| **A0** | `L_DSN` on **real**, then frozen | NPE on sim at frozen z | the status quo; encoder used out of domain |
| **A0s** | `L_DSN` on **sim** (generator labels), frozen | NPE on sim | bench only; isolates domain from objective |
| **A1** | `L_NPE^sim` only | joint | BayesFlow / sbi-embedding-net regime |
| **A2** | NPE + lambda_dsn.`L_DSN` | joint | status quo made joint; nests A1 |
| **A2s** | both terms on **sim** | joint | bench only |
| **A3** | A2 warm-started at A0's psi* | joint | is A0's encoder a good basin? |
| **A5** | NPE + lambda_rep.`L_rep` | joint | replicate consistency; label never used |
| ~~A4~~ | *withdrawn v0.3* | -- | alignment/MMD arm; see S6 |

A0s and A2s exist only where simulated windows carry a phenotype label -- the
bench. On the real bank they cannot be built [KB, O4].

---

## 4. Established results

| # | result | basis |
|---|---|---|
| R1 | **Label ceiling.** At the global minimum of the composite metric loss, `z` is a relabelling of `c`, so `Delta_hat <= log C` regardless of E, d_theta, capacity or bank size; free axes carry exactly zero. For C=2, **0.693 nats for all 26 axes together**. | [DERIVED], Prop 5 |
| R2 | **`r_eff = C-1`** at the ETF minimum (means sum to zero, C-1 equal non-zero eigenvalues). Measured real-arm `r_eff = 1.000` at C=2 **is** that value. Reframes r2 as *better optimised*, not more broken. | [DERIVED] + [KB] |
| R3 | **Cardinality bounds information; dimension does not.** A scalar summary has unbounded capacity. So the log C ceiling holds on the real arm (2 points) but does **not** transfer to the sim arm by `r_eff = 1.017` alone. | [DERIVED], Prop 8 |
| R4 | **Sufficiency asymmetry.** A theta-sufficient `z` keeps `I(c;z) = I(c;x)`; the converse fails -- `z = c` carries <= log C about theta. The objectives are not symmetric alternatives. | [DERIVED] |
| R5 | **NPE training = variational MI maximisation** (Barber-Agakov). `min L = H[theta] - max_psi I(theta; h_psi(x))`. The stack's `Delta_hat` *is* the bound. | [DERIVED], Prop 4 |
| R6 | **Paired cluster bootstrap required.** At rho=0.15 a nominal 95% row bootstrap covered **56.8%**; grouped schemes 93.2%. `se(none)/se(all)` tracks `1/sqrtDEFF`. | [RAN] |
| R7 | **Replicate disagreement is an inverted U** in lambda_j, peaking where F ~ Sigma0^-1. Unweighted loss targets accidentally-intermediate directions. | [DERIVED] + [RAN] |
| R8 | **Target is `p_eff`, not zero**, and `p_eff` = sum of the generalised eigenvalues `information_spectrum` already reports. Verified numerically in the non-diagonal correlated case. | [DERIVED] + [RAN] |
| R9 | **No gate can run on real data** -- G1/G2/G3 all need theta*. Replicate consistency is the only label-free, truth-free calibration check available. | [REPO] |
| R10 | **Simulator parameters are log-transformed** by a mechanical rule: both bounds > 0 and span >= 1 decade; natural log; prior box in theta-coordinates. | [REPO] |

**Literature anchors** (full text read): Min et al., Genetics 2026,
`10.1093/genetics/iyag107` -- joint embedding+flow training, and the frozen
pre-trained network as summaries (the A0 analogue). Hommersom et al., Brain
2025;148(4):1286-1301, `10.1093/brain/awae330` -- biophysical MEA model + SBI
chosen for parameter degeneracy; 4-AP and NS309 with differential and partly
**opposite** effects; states its own limitation as one genetic background and
calls for multiple patient-derived models. Abstract-only, flagged: Van Lent
2021 `10.1093/brain/awab226`; Pavlinek 2026 `10.1016/j.crmeth.2026.101371`;
Ronchi 2021 `10.1002/adbi.202000223`.

**bioRxiv/medRxiv: the connector errored on every attempt across three
sessions.** No preprint check supports any document produced here.

---

## 5. Predictions, with numbering preserved

P5 was withdrawn; numbering was **not** re-flowed, so cross-references between
documents stay valid.

| # | prediction | cheapest test |
|---|---|---|
| P1 | free-axis gain lower under A0 than A1 | bench, per-axis Delta_hat |
| P2 | label axes comparable; A0 may lead at small n | bench |
| P3 | `r_eff -> C-1` at large lambda_dsn | gate's own report |
| P4 | rho_grad < 0 on a non-trivial fraction of steps in A2 | per-epoch probe |
| ~~P5~~ | *withdrawn v0.3* | -- |
| P6 | sim-arm and pseudo-real rankings need not agree | bench, both endpoints |
| P7 | selected m_cos, alpha below the clustering-only optimum | two campaigns |
| **P8** | **Delta_hat(A0) <= log 2 = 0.693 nats on the real bank** | **one `evaluate` run, existing data** |
| P9 | Delta_hat -> log C as ARI -> 1, monotone | bench |
| P10 | window-aggregation curve flat under collapse, rising otherwise | real bank, no ground truth |
| P11 | aliasing worst along sloppy directions | simulator Jacobians |
| P12 | pre-treatment posterior predicts drug response better than label/z/baseline | needs D14 |

---

## 6. Withdrawn and corrected -- read before reusing anything

**Do not resurrect these.** Each was argued in this session and then retracted.

| what | why withdrawn | where it still appears |
|---|---|---|
| **The O(E) drift-invariance argument** -- that the joint objective is blind to sim-real separation | Algebra correct; **realisability never shown**. A domain-conditional rotation is far stronger than a map whose outputs merely differ. Nothing in either objective aims at separation. | plan v0.2 S2.3b; `INFO_LOSS_THEORY_v1.md` S3.7.2 / Prop 12 -- **flagged in v0.4 S9, not yet demoted in the theory doc** |
| **Arm A4**, decision **D8**, prediction **P5**, MMD/kappa as drift monitors | Consequences of the above | v0.4 S9 and D8-closed only |
| "Within-group rows are near-duplicates" | Wrong: `derive_groups` keys on the connectivity axes only, **3 of 26**; the bank is theta-deduplicated, so the other 23 vary. Genuine distinct realisations. | corrected in v0.4 S2.4a |
| "On a sloppy axis the posterior mean wanders over the prior range" | **Backwards.** If the posterior ~ prior, the mean ~ prior mean -- a constant -- so replicates agree *exactly*. Confused posterior width with posterior-mean variability across datasets. | corrected in `METRIC_REPLICATE_v1.md` S3.4.4 |
| "The information metric" (implying Fisher `F`) | Conflates `F` with posterior precision `C^-1 = Sigma0^-1 + F`. Equal only in the diffuse-prior limit -- false on sloppy directions. | corrected in `METRIC_REPLICATE_v1.md` S3.4.1 |
| "The metric protects against collapse" | It does not. As F->0, V->0 and T->0 under any fixed metric. Only the non-zero target and the NPE term prevent degeneracy. | corrected in `METRIC_REPLICATE_v1.md` S3.9 |

**Retained and independent of the withdrawal:** the two-label-source asymmetry
(theta only on sim, c only on real) is a fact about data availability. And A0's
encoder being fitted on real windows and applied to simulated ones is plain
covariate shift -- Stage 3b decomposes A0's deficit into a *domain* effect and
an *objective* effect, and `r_eff(sim) = 1.017` is an out-of-domain number
with **three** candidate explanations nobody has separated: transfer of
collapse, off-support degeneracy, or simulator poverty (= O2).

---

## 7. Numbers of record

All [KB] unless noted. **Two inconsistencies found in the KB itself -- see S9.**

| quantity | value |
|---|---|
| parameter width | `p = 26` (17 log, 9 linear) -- but see S9 |
| simulated arm, exported | 86,251 rows, 54 shards, **383 topology draws** |
| simulated arm, after MFR floor 0.1 Hz/electrode | **29,616 rows (34.3%)** -- but see S9 |
| real arm | 315 npz = **35 cultures x 9 subregions**; f_s=100 Hz, T_rec=1200 s; **1890 windows** (54/culture); 1890/1890 kept |
| encoder in use | `refit_mea_joint_full_r2_l0_t82`, E = 10 |
| effective rank | sim **1.017**, real **1.000**; earlier E=14 encoder 1.483 |
| gate verdict | reject at permutation floor everywhere, `p_group = p_iid = 0.001996`, MDE 0.02 at power 1.0 |
| witness (E=14) | 5% deletion keeps 99.0%; 46.6% at degenerate max vs 19.0% in-band |
| truncation | 26/26 axes at bounds; at epsilon=0.05, median single-window width **0.9788**, envelope 0.9856 |
| NPE defaults | 128 hidden, 8 transforms, 10 bins, batch 512, early stop 20 of max 500 epochs; ensembles currently M=2 |
| grouping key, sim | connectivity-kernel axes of theta, via `np.unique` -- **not** the parquet `topo_idx` [REPO] |
| grouping key, real | the **specs file's `culture`** field, never the npz `culture_id` (repeats across batches) |
| consistency check | `-26.ln(0.9788) = 0.557` nats vs ceiling `ln 2 = 0.693` [RAN] |

---

## 8. Code delivered

`bootstrap_paired.py` -- paired bootstrap for `D = L(A0) - L(A1)`.

```python
from bootstrap_paired import compare_schemes, format_comparison
# d = nll_A0 - nll_A1 per row (positive => A0 worse); group from the split manifest
print(format_comparison(compare_schemes(d, group, label="A0 vs A1, selection split")))
```

Four schemes, all first-class: `none` (rows), `all` (groups, every row -- 
**primary**), `mean` (groups, group mean), `one` (groups, one random row).
Plus `intraclass_correlation` (rho, n0, design effect) and `tail_share`.

```bash
python3 smoke_test_bootstrap_paired.py # 10 tests, ~10 s
python3 smoke_test_bootstrap_paired.py --full # adds coverage, ~2 min
```

Verified at hand-over and **re-run 2026-09-02** [RAN]: `py_compile`, pure
ASCII, LF-only, all tests pass. Key outputs: T10 coverage
`none=0.568 all=0.932 mean=0.932 one=1.000`; demo `rho(d)=0.0706, DEFF=6.72,
se(none)/se(all)=0.372` against predicted `0.386`. In that demo `none`
declares significance **with the wrong sign** while all grouped schemes
correctly return "no" -- the silent failure this exists to prevent.

**Not run on real data.** No `d` array from the actual pipeline has been
scored.

---

## 9. Two inconsistencies found in the knowledge base

Both surfaced while verifying numbers for this handoff [KB]. **Neither is
resolved**; do not silently pick one.

1. **Bank row count: 29,616 vs 29,416.** `SBI_PIPELINE.md` S6 and S12.4 both
 say the MFR floor keeps **29,616** rows. `SBI_PIPELINE.md` O1 says
 "29,416 pairs for p = 26", and the `HPC_PATHS.md` changelog says "the real
 29,416-row bank". A 200-row difference. Check the actual parquet before
 quoting either.
2. **Whether `p = 26` is fixed.** `SBI_PIPELINE.md` S1 states `p = 26`
 throughout. `HPC_PATHS.md` S8 says: *"theta label width p: do not assume a
 fixed number -- it is now determined by `artifacts/label_axes.json`, which
 itself depends on which campaigns are included and their `conn_rule`.
 Regenerate/re-check it if the campaign set changes."* Every document
 produced this session assumes 26.

**Also [DERIVED], not confirmed:** the **23 global + 3 topology** split used
throughout the A5 design. The 3 follows from the Weibull kernel
`p(d) = p0_conn.exp(-(d/d0_conn)^beta_conn)` having three parameters, and 23 =
26 - 3. Confirm against `label_axes.json` before building on it.

---

## 10. Open decisions

**Blocking.**

- **D12 -- the design table.** How many distinct **donors** behind the 35
 cultures, and does any donor appear in more than one well or batch? Gates
 arm A5 entirely and all of the stratification (`Sigma_rep` needs same-donor
 pairs). If every culture is one donor with one well, `Sigma_rep` has no pairs at
 the level that matters and the simulated nuisance floor becomes essential
 rather than supplementary. Also: are batches nested within diagnosis? What
 is known: 315 npz = 35 cultures x 9 subregions, one condition per culture,
 and `ptrain_A1` exists in more than one batch [KB] -- so batches exist, but
 the donor->culture mapping is [OPEN].
- **D14 -- pharmacology.** Any paired pre/post-compound recordings, or a
 well-characterised blocker across a subset of cultures? Gates Stage 7's
 second half and P12. **Asked twice, not answered.**
- **D13 -- replicate consistency: loss or metric?** Train on it (A5) or report
 it for every arm? My recommendation is *both*, and the diagnostic use is the
 stronger of the two (R9).

**Others** (full text in plan v0.4 S8): D1 loss composition; D2 bench window
60 s vs 180 s; D3 C and tau_ov; D4 campaign count; D5 bench scale convention;
D6 conditioner normalisation; D7 ensemble semantics; D9 real-stream
discipline; D10 primary endpoint; D11 gap severities. **D8 closed** (no
alignment term, no arm A4).

---

## 11. Discrepancies found in the code, none reported upstream

**In `npe_tune_gates.py` / `npe_diagnostics.py`** [REPO], full detail in
`GATES_v1.md` S5:

1. **G3's coverage-deficit branch appears unreachable.** `gate_g3` reads
 `getattr(cov, "levels", getattr(cov, "ecdf_x", []))`, but
 `expected_coverage` returns a `RankResult` whose fields are
 `name, ranks, n_draws, ks_pvalue, mean_rank_frac, verdict` -- verified
 programmatically [RAN]. Both chains return `[]`, so control always falls to
 the KS branch. **Consequence: the documented tolerance of over-coverage is
 not in force, and `coverage_tol` does nothing.**
2. **Two significance thresholds coexist.** `RankResult.passes` and
 `TARPResult.passes` hardcode `ks_pvalue > 0.005` (uncorrected); the gates
 use Holm at alpha=0.05. A per-axis report and the gate verdict can disagree.
3. **A NaN bootstrap bound passes G1 condition (b) vacuously** -- 
 `ok_ci = (not isfinite(...)) or (... > delta_min)`.
4. **Latent uninitialised memory in TARP's fold loop** when a calibration half
 is smaller than `d_theta + 1`; the post-loop guard tests only the last fold.

**Prerequisite, not a bug:** `npe_tune_score.py` persists only aggregates. The
**per-row log-density array and the group label per row** must be saved beside
each trial's ledger entry, or `D` cannot be paired and no interval can be
built after the fact. Small change; **gates Stage 3**.

**Not checked:** the cluster checkout (everything here is from branch
tarballs), `npe_local.py`, `prior_truncate.py`, the DSN trainer's optimiser
internals, and the ANN simulator beyond `HPC_main_sweep.py`'s parameter block.

---

## 12. Next actions, cheapest first

1. **Run P8.** `Delta_hat(A0)` on the existing 29,616-row bank must not exceed
 `log 2 = 0.693` nats. One `npe_tune.py evaluate` run, no new simulation,
 tooling already validated. **The cheapest decisive experiment in the
 programme.** If it fails, R1 does not apply to the real arm and the account
 needs revision before anything else proceeds.
2. **Add per-row persistence to `npe_tune_score.py`** (S11). Blocks Stage 3.
3. **Run P10**, the window-aggregation curve on the real bank. Flat means
 collapse, rising means information. No ground truth, no labels.
4. **Two direct probes of the domain effect** (Stage 3b), no training needed:
 per-layer activation statistics on simulated vs real inputs, and the
 marginal distribution of simulated embeddings along the real cloud's
 surviving direction.
5. **Answer D12 and D14.** Everything in S2's second and third deliverables
 waits on these.
6. **Write plan v0.5**, folding in: the A5 rewrite (parameter-space
 constraint, `p_eff` target not zero, stop-gradient on omega, warm-up ramp tied
 to a measured criterion), IDEA-001 as an explicit stated aim, and the
 measure-then-constrain ordering for which axes replicate.
7. **Demote Prop 12 in `INFO_LOSS_THEORY_v1.md`** to a caveat, matching plan
 v0.4 S9. Currently the two documents disagree.
8. **Retry bioRxiv.** Nothing has ever been checked there.

---

## 13. Conventions established this session

- **Explanation modes.** `EXPLANATION_MODES.md` to be pasted into project
 instructions: "explain comprehensively" -> pedagogical *and* fully technical,
 with failure mode and counterexample by default, and errors in the passage
 being explained flagged rather than restated; "briefly" -> answer in the
 first sentence, one short paragraph, no headings, one caveat maximum.
- **Project ideas notes.** "Add to the project ideas notes" appends the idea
 under discussion to `PROJECT_IDEAS.md`. Currently IDEA-001 (drop diagnostic
 classification) and IDEA-002 (morphing without retraining).
- **Notation deviations from `SBI_PIPELINE.md`,** used in all documents
 produced here and flagged inside each: `d_theta` for the parameter dimension
 (not `p`, which is needed for densities), and `[a_k, b_k]` for prior box
 bounds (not `[L_k, U_k]`, since `L` is the NLL).
- **Prediction numbering is never re-flowed.** P5 stays withdrawn-in-place.
- **Two document genres, kept separate.** Formal (notation table, glossary,
 abstract) for mathematical content aimed at a human reader; technical (this
 one -- changelog, greppable headers, tables, no glossary) when the reader is
 a future session.

---

## 14. Questions asked and not answered

Carry these forward:

1. **D12** -- donors behind the 35 cultures; same donor in >1 well or batch?
 (asked once)
2. **D14** -- any pharmacology data? (asked twice)
3. **Stage 3b** -- keep it? (asked once; the user then asked for it to be
 explained better, which I read as *keep*, but it was never confirmed)
4. **D13** -- replicate consistency as loss, metric, or both?
5. Whether to resolve G3's coverage branch by returning the coverage curve
 from `expected_coverage`, or by changing the docstring to match the
 two-sided KS behaviour now in force.

---

## 15. What this session did not do

Scope discipline, so coverage is not over-read:

- **Nothing ran on the cluster.** No PBS job, no `qsub`, no `check_env.py`.
 Every `[RAN]` tag refers to the analysis sandbox.
- **No new code was delivered to any repository.** `bootstrap_paired.py` and
 its smoke test exist as files only; no branch, no commit, no patch.
- **No real data was touched.** No bank was loaded, no posterior scored, no
 `d` array computed. Every quantitative claim about the pipeline is from the
 KB, not from a fresh measurement.
- **Repositories were read from branch tarballs** -- `Deep-Summary-Network@main`,
 `Simulation-Based-Inference@feat/misspec-gate`,
 `Sbi-extractor@feat/real-arm-parity`, `Astro-Neuron-Network@main` -- not from
 the cluster checkout. They may have diverged.
- **The bench does not exist.** Every Stage 1+ deliverable in plan v0.4 is
 designed, not built.
- **No literature check on bioRxiv/medRxiv**, connector failing throughout.
