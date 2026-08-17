# Validating an Amortized SBI Pipeline: Results of the Synthetic Trial

**Date:** 2026-08-12
**Run:** `synthetic_trial_20260811_152440`
**Figures:** `presentation_figures/`

## Abstract

**Scientific question.** Before applying an amortized neural posterior estimator to real multi-electrode recordings, can we demonstrate that the inference pipeline is correct, and that the diagnostic battery attached to it is capable of detecting the ways it could fail?

**What this run is.** A full end-to-end dress rehearsal at production shapes — 27 parameters, a 12-dimensional embedding, 20 000 training simulations, a 5-member ensemble — on synthetic data whose generating map is known. It exercises every stage the real data will pass through and produces the same twelve diagnostic figures that will accompany the real analysis.

**What this run is not.** It says nothing about neuronal biophysics. The parameters are synthetic and the forward map is a fixed linear projection onto the unit sphere. Every number below is a statement about the *machinery*, not about a culture.

**Headline result.** The pipeline works, the overlap gate fires correctly in both directions, and every parameter is informed by the data. But the joint posterior is substantially overconfident: at a nominal 90% credible level it delivers 61% coverage. Critically, **marginal simulation-based calibration passed on all 27 axes** — the standard diagnostic in most published applications would have declared this model sound. The failure is visible only in the joint coverage analysis, and ensembling is shown to reduce it without removing it.

**Scope of this document.** Interpretation of the twelve figures, the disagreement between diagnostic layers, and what each result implies for the real-data stage. Excluded: implementation details of the estimator, the mathematics of each diagnostic (see the diagnostics reference handoff), and any biological interpretation.

---

## 1. Notation and Symbols

| Symbol | Name / Meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $\theta$ | Parameter vector (inference coordinates) | $\theta \in \Theta \subset \mathbb{R}^{p}$ | mixed | §3 |
| $\theta^{*}_{i}$ | Ground truth for calibration draw $i$ | $\theta^{*}_{i} \in \Theta$ | mixed | §5 |
| $p$ | Parameter dimension | $p = 27$ | dimensionless | §3 |
| $z$ | Embedding of an observation | $z \in \mathbb{S}^{E-1} \subset \mathbb{R}^{E}$ | dimensionless | §3 |
| $E$ | Embedding dimension | $E = 12$ | dimensionless | §3 |
| $q_\phi(\theta \mid z)$ | Estimated posterior | density on $\Theta$, for each fixed $z$ | dimensionless | §3 |
| $p(\theta)$ | Prior | density on $\Theta$ | dimensionless | §3 |
| $N$ | Calibration observations | $N = 500$ | dimensionless | §5 |
| $M$ | Posterior draws per observation | $M = 128$ | dimensionless | §5 |
| $n$ | Ensemble members | $n = 5$ | dimensionless | §8 |
| $\alpha$ | Significance / credibility complement | $\alpha \in (0,1)$ | dimensionless | §6 |
| $1-\alpha$ | Nominal credibility level | $\in (0,1)$ | dimensionless | §6 |
| $u_{i}$ | Jittered fractional rank of the truth | $u_{i} \in [0,1]$ | dimensionless | §5, eq. (1) |
| $r_{i}$ | Integer rank of the truth among draws | $r_{i} \in \{0,\dots,M\}$ | dimensionless | §5 |
| $c_{k}$ | Posterior contraction on axis $k$ | $c_{k} \in (-\infty, 1]$ | dimensionless | §7, eq. (3) |
| $\lambda_{j}$ | $j$-th eigenvalue of the information spectrum | $\lambda_{j} \in [0,1]$ | dimensionless | §7, eq. (4) |
| $\mathrm{MMD}$ | Maximum mean discrepancy in summary space | $\ge 0$ | dimensionless | §4 |
| $\tilde{p}_{k}$ | Holm-Bonferroni adjusted $p$-value | $\in [0,1]$ | dimensionless | §5 |
| $W$ | Fixed random matrix, bilinear test quantity | $W \in \mathbb{R}^{p \times E}$ | dimensionless | §6.3 |
| $\theta_{r}(z)$ | TARP reference point as a function of the observation | $\mathbb{R}^{E} \to \Theta$ | mixed | §6.4 |

### 1.1 Conventions

- All $p$-values are two-sided Kolmogorov-Smirnov against uniformity unless stated otherwise; the per-test threshold is $0.005$, and per-axis families are additionally controlled at $\alpha = 0.05$ by Holm-Bonferroni.
- "Nominal level" always means $1-\alpha$, the stated credibility of a region; "empirical coverage" means the observed frequency with which the truth falls inside it.
- Axis names `ax00` ... `ax26` are synthetic placeholders; in the real run they carry the biophysical parameter names from the campaign manifest.
- Figure numbers refer to files in `presentation_figures/` and are used as section anchors throughout.

---

## 2. Glossary

Ordered by first appearance.

- **Amortized posterior estimator.** A network trained once over the whole prior that returns a posterior for any observation without retraining. Makes it affordable to assess calibration over hundreds of observations. Operative in §3.
- **Embedding / summary.** The fixed-dimensional compression $z$ of a recording used as the conditioning variable. **Everyday-meaning warning:** "summary" does not imply *sufficient* — a summary can discard the information the parameters live in. Operative in §3 and §7.
- **Calibration.** Agreement between stated and realised uncertainty. **Everyday-meaning warning:** a calibrated posterior need not be informative; the prior is perfectly calibrated and useless. Operative in §5 and §7.
- **Marginal SBC.** A rank-uniformity test applied one parameter at a time. Sees each marginal; blind to correlations between them. Operative in §5.
- **Expected coverage.** The frequency with which the truth falls inside the estimator's own $(1-\alpha)$ highest-density region. Sensitive to the joint structure. Operative in §6.
- **Overconfident.** Empirical coverage below nominal — credible regions too small. The dangerous direction of error, because it produces false certainty. Operative in §6.
- **Conservative.** Empirical coverage above nominal — regions too large. The safe direction. Operative in §6.
- **TARP.** A coverage test using spherical regions around reference points, requiring no density evaluation. Its power depends critically on the reference points being a function of the observation. Operative in §6.4.
- **Data-dependent test quantity.** A statistic $f(\theta, z)$ depending on both the parameters and the observation, as opposed to $f(\theta)$ alone. Operative in §6.3.
- **Posterior contraction.** One minus the ratio of posterior to prior variance on an axis; measures information gain rather than correctness. Operative in §7.
- **Deep ensemble.** Several independently trained estimators combined as a mixture. Operative in §8.
- **Overlap gate / MMD test.** A two-sample test asking whether query embeddings lie inside the distribution of simulated ones. Operative in §4.

---

## 3. What was run

**This section establishes the configuration and what it does and does not demonstrate.**

| Setting | Value |
|---|---|
| Parameter dimension $p$ | 27 |
| Embedding dimension $E$ | 12 (unit-normalised, so 11 degrees of freedom) |
| Training simulations | 20 000, written as 2 Parquet shards |
| Ensemble members $n$ | 5 |
| Calibration set | $N = 500$ held-out observations, $M = 128$ draws each |
| Query sets | 36 matched (held out) and 36 deliberately shifted |

The synthetic generator maps $\theta$ through a fixed linear projection into $E$ dimensions, adds noise, and normalises onto the sphere. The relationship between $\theta$ and $z$ is therefore known to exist and to be learnable, which is exactly the property needed for a validation run: any diagnostic failure is attributable to the pipeline rather than to an unlearnable problem.

Total wall-clock: **3.9 hours**, of which training was 10.5 minutes and posterior sampling 4.3 minutes. The remaining 3.7 hours were consumed by a single avoidable step (§10).

---

## 4. The overlap gate fires correctly in both directions

**Figures 01a, 01b. This section establishes that the go/no-go gate is usable.**

Before any posterior is trusted, the query embeddings must lie inside the distribution of simulated ones. If they do not, the estimator is extrapolating and no calibration result computed on simulations applies.

| query set | MMD $p$-value | verdict |
|---|---|---|
| held-out, same process (fig. 01a) | $0.67$ | no gap detected — correct |
| deliberately shifted (fig. 01b) | $0.0040$ | **rejects** — correct |

Both directions matter. A gate that always fires is as useless as one that never does, and only testing both establishes that it discriminates.

Figure 01 has three panels: the MMD against its bootstrap null, the distribution of geodesic nearest-neighbour distances on the sphere, and a two-dimensional projection of both clouds.

**One caution about the third panel, worth stating whenever it is shown.** It projects onto the *most separating* direction, not onto the leading principal components. An earlier version used PCA and was misleading: a query set the gate correctly rejected looked perfectly well mixed, because PCA selects directions of greatest *simulated* variance, which need not be where the query data differs. The consequence of the current choice is an asymmetry — **overlap in this view is strong evidence of no gap; apparent separation is weak evidence**, because a discriminative axis can overfit at small query-set size. Read the $p$-value, use the panel for intuition.

---

## 5. Every marginal is calibrated

**Figures 02, 03. This section establishes what the standard diagnostic reports.**

For each axis $k$ we rank the truth among the posterior draws and test the ranks for uniformity, using

$$u_{i} = \frac{r_{i} + \varepsilon_{i}}{M+1}, \qquad \varepsilon_{i} \sim \mathrm{Unif}(0,1), \qquad \text{for each fixed } i \in \{1,\dots,N\}. \tag{1}$$

The jitter is not cosmetic: ranks take $M+1$ discrete values, and comparing them against a continuous uniform without it inflates the test statistic and manufactures rejections.

**Result: 0 of 27 axes reject.** The smallest Holm-Bonferroni adjusted $p$-value is $0.166$, comfortably above the $0.05$ family-wise threshold. Figure 02 shows all 27 rank ECDFs inside the simultaneous 95% band; figure 03 shows the individual histograms, all flat.

Family-wise correction matters here and is applied. With 27 axes tested at $0.005$ each, a perfectly calibrated posterior would trip something roughly 13% of the time; two axes do show raw $p$-values near $0.006$ and $0.012$, and without correction those would have been reported as failures.

**Taken alone, this figure says the estimator is fine.** §6 shows that conclusion is wrong.

---

## 6. The joint posterior is substantially overconfident

**Figures 04, 05, 06. This is the central result.**

### 6.1 What the coverage analysis shows

Using the estimator's own log-density as the test quantity converts the rank statistic into the expected coverage probability. The truth lies inside the $(1-\alpha)$ highest-density region exactly when the fraction $u_{i}$ of posterior draws with lower density than the truth exceeds $\alpha$, so, writing $c = 1-\alpha$ for the nominal level,

$$\widehat{\mathrm{coverage}}(c) = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}\big(u_{i} > 1-c\big), \qquad \text{for each fixed } c \in (0,1). \tag{2}$$

**Result: $p = 2.9 \times 10^{-25}$ (KS, $N=250$).** The curve in figure 04 lies below the diagonal at every level:

| nominal credibility | empirical coverage |
|---|---|
| 68% | **51%** |
| 90% | **61%** |
| 95% | **63%** |

The truth falls into the lowest-density decile of the estimated posterior **39% of the time**, against 10% expected. Credible regions are too small by a wide margin.

This is the dangerous direction of error. An overconfident posterior does not merely lose information; it makes confident claims that are wrong, and a 90% interval that contains the truth 61% of the time would invalidate any parameter comparison built on it.

### 6.2 Why marginal SBC missed it

The two results are not contradictory. Marginal SBC integrates over all other parameters, so a posterior with correct marginals and wrong *dependence structure* passes it. The joint coverage analysis is the only layer in the battery that sees correlations. Since the true posterior here has strong correlations induced by projecting 27 parameters through an 11-degree-of-freedom summary, this is precisely the structure a partially-trained flow would get wrong.

**Practical consequence for the real analysis: never report marginal SBC alone.** It is the standard diagnostic in most published applications, and on this run it would have certified a model whose 90% intervals are 61% intervals.

### 6.3 The posterior is nevertheless using the data

**Figure 05.** A separate concern is whether the estimator is using the observation at all — the limiting failure of an insufficient summary is a posterior that reverts to the prior, and such a posterior passes marginal SBC and coverage alike. This is tested with data-dependent quantities $f_{W}(\theta, z) = \theta^{\mathsf{T}} W z$ for eight independent random $W$.

**Result: 0 of 8 reject.** Combined with the contraction results in §7, the estimator is demonstrably extracting information from $z$. The overconfidence is a calibration problem, not an information problem.

### 6.4 TARP: the two reference modes disagree

**Figure 06.** TARP places spherical credible regions around reference points and asks how often the truth is closer to the reference than the posterior draws are.

| reference points | $p$-value | verdict |
|---|---|---|
| random, independent of the observation | $9.3 \times 10^{-5}$ | fail |
| positioned as a function of $z$ | $0.042$ | pass |

Both are legitimate tests, and they probe different geometry: random references test coverage of spheres placed anywhere in parameter space, while $z$-dependent references test spheres placed near the predicted value. The random-reference version corroborates the coverage failure; the $z$-dependent version is less sensitive to this particular defect, though its $p$-value is not comfortable either.

**Report both.** The theory requires $z$-dependent references for TARP's completeness guarantee, but this run demonstrates that the random-reference variant can be the more sensitive of the two against some failure modes.

---

## 7. All 27 parameters are informed

**Figures 07, 08, 09, 10. This section establishes that the posterior carries information, which calibration alone cannot show.**

Posterior contraction on axis $k$,

$$c_{k} = 1 - \frac{\mathbb{E}_{i}\big[\mathrm{Var}(\theta_{k} \mid z_{i})\big]}{\mathrm{Var}_{p(\theta)}(\theta_{k})}, \qquad \text{for each fixed } k \in \{1,\dots,27\}, \tag{3}$$

is 0 when the posterior marginal is as wide as the prior and 1 when the parameter is determined.

**Result (figure 07): mean 0.42, range 0.20 to 0.65, and 0 of 27 axes below the 0.05 threshold.** Six axes exceed 0.5. Every parameter is informed; none has reverted to the prior.

This layer is not redundant with §5 and §6. A posterior exactly equal to the prior is perfectly calibrated and completely uninformative, and only contraction distinguishes the two. Reporting calibration without contraction — or contraction without calibration — leaves the obvious question unanswered in each case.

The information spectrum (figure 08) reports the eigenvalues

$$\lambda_{j} = \mathrm{eig}_{j}\Big(\mathrm{Cov}_{p(\theta)}(\theta)^{-1/2}\,\mathrm{Cov}_{i}\big(\mathbb{E}[\theta \mid z_{i}]\big)\,\mathrm{Cov}_{p(\theta)}(\theta)^{-1/2}\Big), \tag{4}$$

measured here as $[0.99, 0.98, 0.96, 0.94, 0.92, 0.90, 0.89, 0.87, 0.85, 0.85, 0.81, 0.12, \dots]$ — eleven large values, then a sharp drop by nearly a factor of seven at the twelfth.

**A caution on reading this.** It is tempting to treat the count of large eigenvalues as a hard test of "an $E$-dimensional embedding constrains at most $E-1$ directions". That reading is wrong for a nonlinear estimator: (4) measures the *linear* rank of a possibly curved image, and a curved $d$-dimensional manifold spans more than $d$ linear dimensions. The cliff after eleven values is suggestive and consistent with $E-1 = 11$, but it is descriptive, not a test. Judge informativeness from (3).

Figures 09 and 10 support this: recovery plots show true-versus-inferred alignment for the best-constrained axes, and the pair plot shows the joint structure and the truth's position within it for a single observation.

---

## 8. Ensembling helps substantially and does not suffice

**Figure 11. This section establishes what the ensemble contributes.**

Comparing a single member against the 5-member mixture on the same observations:

| | truth in lowest-density decile | mean rank |
|---|---|---|
| single member | **55%** | 0.215 |
| ensemble of 5 | **39%** | 0.366 |
| correct value | 10% | 0.500 |

Ensembling closes roughly a third of the gap. This reproduces, on our own pipeline, the finding in the peer-reviewed literature that ensembling raises expected coverage while providing no guarantee of conservativeness — the extra spread between independently trained members is genuine epistemic uncertainty that a single network hides, but it is not a substitute for a well-trained estimator.

Figure 11 makes the mechanism visible: the members disagree noticeably on the best-constrained axes, and that disagreement is what widens the mixture.

**Actionable consequence.** Raising the ensemble from 5 to 10 members is the cheapest available improvement, and the trend above suggests it will help. It will not, on its own, close a gap this large.

---

## 9. Summary of results

| # | Statement | Established in |
|---|---|---|
| S1 | The overlap gate rejects a shifted query set ($p = 0.004$) and passes a matched one ($p = 0.67$). | §4 |
| S2 | Marginal SBC passes on all 27 axes; min adjusted $p = 0.166$. | §5 |
| S3 | The joint posterior is overconfident: nominal 90% delivers 61% coverage, $p = 2.9\times10^{-25}$. | §6.1, eq. (2) |
| S4 | S2 and S3 are consistent: marginal SBC cannot see correlations. Reporting it alone would have certified this model. | §6.2 |
| S5 | The posterior does use the observation: 0 of 8 data-dependent quantities reject. | §6.3 |
| S6 | TARP's two reference modes disagree; the random-reference variant is the more sensitive here. Report both. | §6.4 |
| S7 | All 27 parameters are informed; contraction mean 0.42, none below 0.05. | §7, eq. (3) |
| S8 | The information spectrum shows eleven large eigenvalues then a sevenfold drop, consistent with $E-1 = 11$ but not a test of it. | §7, eq. (4) |
| S9 | Ensembling moves the lowest-decile occupancy from 55% to 39% against an ideal of 10%. | §8 |
| S10 | The pipeline runs end to end at production shapes in 3.9 hours, of which 3.7 were avoidable overhead. | §3, §10 |

---

## 10. Open points, caveats, and assumptions

1. **The overconfidence has a plausible and testable cause that has not been confirmed.** Five members, 20 000 training rows, 27 dimensions and a 60-epoch cap is a modest budget for learning a joint distribution. The candidate remedies — more training rows, more epochs, more members, a larger flow — are untested, and it is not established which of them binds.
2. **3.7 of the 3.9 hours were consumed by a library default.** The `sbi` posterior evaluates a leakage correction by drawing 10 000 rejection samples per observation per member. Under the bounded-support transform used here that correction is provably a no-op — the measured in-box fraction is 1.0000 — and it has since been disabled. A rerun should take well under half an hour.
3. **This is synthetic data with a linear forward map.** Nothing here establishes that a 27-parameter biophysical model with a learned encoder will behave similarly. The demonstration is that the pipeline and the diagnostics work, not that the science will.
4. **Local calibration is not assessed.** Every test here is global, averaging over observations. A failure specific to one real recording can hide inside an average that looks acceptable. This is a real gap for the real-data stage, where the object of interest is a specific culture.
5. **Posterior predictive checks are absent**, since they require the simulator in the loop. They are the only layer that tests the model rather than the inference.
6. **The coverage figure was computed on $N = 250$ of the 500 calibration observations** for cost reasons. The original full-set run gave $p = 1.7\times10^{-49}$, consistent in direction and magnitude.
7. **Three plotting defects were found and corrected while preparing this document**, all of which had produced valid-looking but wrong figures: a PCA projection that hid a real distribution shift, a coverage plot whose axis was $1-\mathrm{coverage}$ while labelled "coverage" (so the curve sat above the diagonal while the verdict said overconfident), and a recovery plot that crashed on skewed marginals. Figures that render without error are not thereby correct; each was found by looking at the image.
8. **The verdict text attached to the coverage test was also wrong** and has been corrected. It inherited the wording written for parameter marginals and reported "biased high" where the correct reading is "overconfident".

---

## 11. Recommended next steps

In order of value per unit effort:

1. **Rerun with the leakage correction disabled.** Recovers 3.7 hours and makes iteration practical.
2. **Retrain with a larger budget** — 10 members, more epochs, and either more training rows or a larger flow — and re-measure the coverage curve. This directly targets S3.
3. **If overconfidence persists, apply post-training calibration**: replace the level-$\alpha$ region with the level whose empirical coverage is the desired one. This is a documented remedy and cheap, though it treats the symptom.
4. **Run the overlap gate on the real embeddings** as soon as they exist. It requires no training and is a hard go/no-go: if the real embeddings fall outside the simulated cloud, nothing downstream is meaningful.
5. **Add local calibration** (L-C2ST or a local coverage test) before drawing conclusions about any individual recording.

---

## 12. Figure index

| File | Shows | Key number |
|---|---|---|
| `01_embedding_overlap.png` | Gate, matched query set | $p = 0.67$, no gap |
| `01_embedding_overlap_shifted.png` | Gate, shifted query set | $p = 0.0040$, rejects |
| `02_sbc_ecdf.png` | All 27 rank ECDFs with band | 0/27 reject |
| `03_sbc_histograms.png` | Per-axis rank histograms | all flat |
| `04_coverage_pp.png` | **Nominal vs empirical coverage** | **90% → 61%** |
| `05_data_dependent_sbc.png` | Is the observation being used | 0/8 reject |
| `06_tarp.png` | TARP, $z$-dependent references | $p = 0.042$ |
| `07_contraction.png` | **Per-axis information gain** | mean 0.42, 0 uninformed |
| `08_information_spectrum.png` | Eigenvalue decay | 11 large, then ×7 drop |
| `09_recovery.png` | True vs inferred, 68% intervals | best-constrained axes |
| `10_posterior_pairs.png` | Joint structure, one observation | truth marked |
| `11_ensemble_spread.png` | Member disagreement | mechanism behind §8 |

Figures 04 and 07 are the two that carry the talk: one says the uncertainty is understated, the other says the information is real.

---

## 13. References

**Peer-reviewed, consulted directly:**

- Hermans, Delaunoy, Rozet, Wehenkel, Begy, Louppe (2022). *A Trust Crisis in Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful.* TMLR. — Expected coverage, the conservative/overconfident distinction, ensembling as partial mitigation, post-training calibration (§6, §8, §11).
- Lemos, Coogan, Hezaveh, Perreault-Levasseur. *Sampling-Based Accuracy Testing of Posterior Estimators for General Inference.* — TARP, positionable region generators, the requirement that reference points depend on the observation (§6.4).
- Modrák, Moon, Kim, Bürkner, Huurre, Faltejsková, Gelman, Vehtari (2025). *Simulation-Based Calibration Checking for Bayesian Computation.* Bayesian Analysis 20(2):461-488. doi:10.1214/23-ba1404 — Data-dependent test quantities (§6.3).
- Schmitt, Bürkner, Köthe, Radev. *Detecting Model Misspecification in Amortized Bayesian Inference with Neural Networks.* — Summary-space MMD testing (§4).

**Measured in this run, not from any source:** every number in §4 through §8 and §9, all reproducible from `diagnostics_data.npz` and the saved ensemble. The single-member versus ensemble comparison in §8 was computed specifically for this document and is not part of the standard diagnostic output.

**Stated from reasoning rather than a source:** the explanation in §6.2 for why marginal SBC and joint coverage disagree, and the caution in §7 about linear rank versus intrinsic dimension.
