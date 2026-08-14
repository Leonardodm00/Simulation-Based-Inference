# Handoff: Local Calibration Diagnostics for Amortized SBI

**Date:** 2026-08-14
**From:** SBI pipeline chat (built `npe_contract`, `npe_model`, `npe_diagnostics`, `npe_plots` and their test suites; ran the synthetic trial)
**To:** Next chat, tasked with implementing local calibration
**Repo:** `hpc/` in `sbi_hpc.tar.gz`, branch `feat/npe-stage` of `Leonardodm00/Simulation-Based-Inference`

## Abstract

**Scientific question.** Every diagnostic currently implemented is *global*: it averages calibration over hundreds of calibration observations. The scientific claim this project will make is *local* — a statement about one specific in-vitro culture, from one specific embedding $z_o$. A miscalibration confined to the neighbourhood of $z_o$ can hide inside a global average that looks acceptable, and nothing in the present battery would reveal it.

**Task.** Implement local calibration diagnostics — primarily Local Classifier Two-Sample Testing (L-C2ST), secondarily the Local Coverage Test (LCT) — as a new module `npe_local.py`, with a validation suite `smoke_test_local.py` in the style of the existing suites. Then implement posterior predictive checks as `npe_ppc.py`.

**Scope of this document.** The definition and algorithm for L-C2ST and LCT, why they are needed here specifically, how they must fit the existing interfaces, what the validation suite has to prove (including the one test that matters), and the known traps. It records the conventions the existing modules follow so the new one does not diverge from them.

**Deliberately excluded.** The estimator architecture, the simulator, the data export contract (separate handoff), and the theory of the *global* diagnostics (separate handoff: `diagnostics_reference_handoff.md`, which should be read first). This document assumes that one has been read.

**Critical context.** The synthetic trial found the joint posterior substantially overconfident (nominal 90% delivering 61% coverage) while marginal SBC passed on all 27 axes. That is a *global* failure caught only because two global layers disagreed. The local layer is the next thing that could be silently wrong, and it is the one that bears directly on a per-culture claim.

---

## 1. Notation and Symbols

| Symbol | Name / Meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $\theta$ | Parameter vector, inference coordinates | $\theta \in \Theta \subset \mathbb{R}^{p}$ | mixed | §3 |
| $p$ | Parameter dimension | $p = 27$ in production | dimensionless | §3 |
| $x$ | Raw observation (a recording window) | $x \in \mathcal{X}$ | mixed | §3 |
| $z$ | Embedding, the conditioning variable | $z \in \mathbb{S}^{E-1} \subset \mathbb{R}^{E}$ | dimensionless | §3 |
| $z_{o}$ | The specific observation being diagnosed | $z_{o} \in \mathbb{S}^{E-1}$ | dimensionless | §4 |
| $E$ | Embedding dimension | $E \in \mathbb{N}$, 12 in the trial | dimensionless | §3 |
| $q_\phi(\theta \mid z)$ | Estimated posterior | density on $\Theta$, for each fixed $z$ | dimensionless | §3 |
| $p(\theta \mid z)$ | True posterior | density on $\Theta$, for each fixed $z$ | dimensionless | §3 |
| $N_{\text{cal}}$ | Calibration set size | $N_{\text{cal}} \in \mathbb{N}$ | dimensionless | §4.2 |
| $\theta^{*}_{n}$ | Ground truth for calibration draw $n$ | $\theta^{*}_{n} \in \Theta$ | mixed | §4.2 |
| $\theta^{q}_{n}$ | A single posterior draw at $z_{n}$ | $\theta^{q}_{n} \sim q_\phi(\theta \mid z_{n})$ | mixed | §4.2 |
| $C$ | Binary class label in the C2ST construction | $C \in \{0,1\}$ | dimensionless | §4.2, eq. (1) |
| $d_{\omega}$ | Trained binary classifier with weights $\omega$ | $d_{\omega} : \Theta \times \mathbb{R}^{E} \to (0,1)$ | dimensionless | §4.2 |
| $\hat{t}(z_{o})$ | L-C2ST test statistic at $z_{o}$ | $\hat{t}(z_{o}) \in [0, \tfrac{1}{4}]$ for MSE form | dimensionless | §4.3, eq. (2) |
| $\hat{t}_{b}$ | $b$-th null-distribution replicate of the statistic | $\hat{t}_{b} \ge 0$ | dimensionless | §4.4 |
| $B$ | Number of null replicates | $B \in \mathbb{N}$ | dimensionless | §4.4 |
| $\alpha$ | Significance level | $\alpha \in (0,1)$ | dimensionless | §4.4 |
| $T$ | Base-space (flow) representation of $\theta$ | $T = f_\phi^{-1}(\theta; z) \in \mathbb{R}^{p}$ | dimensionless | §4.5 |
| $f_\phi(\cdot\,; z)$ | The flow's conditional bijection | $\mathbb{R}^{p} \to \Theta$, for each fixed $z$ | — | §4.5 |
| $r_{n}$ | Rank statistic for calibration draw $n$ | $r_{n} \in \{0,\dots,M\}$ | dimensionless | §5.1 |
| $M$ | Posterior draws per calibration observation | $M \in \mathbb{N}$ | dimensionless | §5.1 |
| $g(z)$ | Regressed local coverage function | $g : \mathbb{R}^{E} \to [0,1]$, for each fixed level | dimensionless | §5.1 |
| $W_{r}$ | Windows per real recording | $W_{r} = 6$ | dimensionless | §7 |
| $s(x)$ | A summary statistic used for posterior predictive checks | $s : \mathcal{X} \to \mathbb{R}^{d_{s}}$ | mixed | §6 |

### 1.1 Conventions

- The existing codebase calls the conditioning variable `Z` in code and $z$ in prose, never `x`. The literature calls it $x$. **This document uses $z$ throughout**; when quoting a paper's equation, the paper's $x$ is our $z$. Do not mix them in code.
- All new code is ASCII-only, LF line endings. This is enforced by `run_tests.sh` stage 0 and is not negotiable: the project's cluster transfers have corrupted files three times.
- Test-suite naming follows the existing convention: `T*` for pipeline, `G*` for the Gaussian benchmark, `D*` for diagnostics. Use `L*` for local.
- Every diagnostic returns a dataclass with a `passes` property and a `summary()` method, matching `RankResult`, `TARPResult`, `MMDResult`.
- Modules compute; they do not plot. Plotting lives in `npe_plots.py` and receives already-computed objects.

---

## 2. Glossary

Ordered by first appearance, because the concepts build on each other.

- **Global diagnostic.** A calibration test whose statistic averages over many observations — SBC, expected coverage, TARP. Answers "is the estimator calibrated on average?". Operative in §3.
- **Local diagnostic.** A calibration test evaluated at one specific observation $z_{o}$. Answers "is the estimator calibrated *here*?". Operative throughout.
- **C2ST (classifier two-sample test).** Train a binary classifier to separate two samples; if the classifier cannot beat chance, the samples are consistent with coming from one distribution. Already used in `smoke_test_gmm.py` for the *global* comparison of NPE draws against exact posterior draws. Operative in §4.1.
- **L-C2ST.** The local variant: rather than reading the classifier's accuracy, evaluate its predicted class probability at a fixed $z_{o}$ and compare it to $\tfrac{1}{2}$. **Everyday-meaning warning:** despite the name, L-C2ST does not report an accuracy; the statistic is a deviation from chance level, so *large is bad* and zero is perfect — the opposite reading from the C2ST accuracy already used elsewhere in this repo. Operative in §4.
- **Calibration set.** Pairs $(\theta^{*}_{n}, z_{n})$ drawn from the joint, plus posterior draws at each $z_{n}$. Shared by every diagnostic in the repo and already persisted per run. Operative in §4.2.
- **Null distribution by permutation.** The distribution of the test statistic under the hypothesis of perfect calibration, obtained by destroying the association the test is looking for and recomputing. Operative in §4.4.
- **L-C2ST-NF.** A variant exploiting a normalizing flow's invertibility: map $\theta$ into the flow's base space, where under a correct posterior the transformed calibration parameters are standard normal *independently of $z$*. Removes the need to sample the estimator during the null computation. Operative in §4.5.
- **LCT (Local Coverage Test).** Regress the SBC rank statistic on the observation and read off coverage as a function of $z$. Simpler and cheaper than L-C2ST, less sensitive. Operative in §5.
- **Posterior predictive check.** Simulate at posterior draws and compare simulated summaries to the observed ones. **The only diagnostic that tests the model rather than the inference.** Operative in §6.
- **Exchangeable set.** A group of observations sharing one unknown parameter vector, whose order carries no information. Arises here because each real recording yields 6 windows. Operative in §7.

---

## 3. Why this is the right next task

**This section establishes the motivation, so the implementer can judge trade-offs rather than follow instructions blindly.**

Three facts, in combination, make local calibration the highest-value remaining diagnostic:

1. **The claim is local.** The project will report a posterior over biophysical parameters for a particular culture, and compare cultures between conditions. Global calibration averaged over 500 synthetic observations is weak evidence for a statement about one real $z_{o}$.

2. **Global tests have already been observed to disagree with each other on this pipeline.** In the synthetic trial, marginal SBC passed on all 27 axes while joint coverage failed at $p = 2.9\times10^{-25}$. That is direct, in-house evidence that a passing diagnostic layer does not imply a correct posterior. The local layer is untested and could be wrong in the same way.

3. **It requires no new simulations.** The calibration set already exists in every run's `diagnostics_data.npz`. This is the defining property that makes it the right thing to build *while waiting for data*: it is the most valuable thing that can be done with zero input from the wet lab and zero cluster time on the simulator.

The reference literature in the project knowledge base is explicit that local coverage checks are "generally more powerful for pinpointing subtle failures that global checks might miss", while noting the cost: typically more simulations, an additional neural network, and the possibility of the diagnostic introducing its own errors through imperfect convergence. That last caveat is why §8 demands the validation suite it does.

---

## 4. L-C2ST — the primary task

**This section establishes the algorithm to implement.**

### 4.1 The idea

We want to test $q_\phi(\theta \mid z_{o}) = p(\theta \mid z_{o})$ for one fixed $z_{o}$. We cannot sample the true posterior. The trick is that we *can* sample the true **joint**, because the calibration set is exactly that: $\theta^{*}_{n} \sim p(\theta)$, then $z_{n}$ simulated from $\theta^{*}_{n}$, gives $(\theta^{*}_{n}, z_{n}) \sim p(\theta, z)$.

### 4.2 Construction

Build a two-class dataset over $\Theta \times \mathbb{R}^{E}$:

$$\big(\theta^{*}_{n},\, z_{n}\big) \,\big|\, (C=0) \;\sim\; p(\theta, z), \qquad \big(\theta^{q}_{n},\, z_{n}\big) \,\big|\, (C=1) \;\sim\; q_\phi(\theta \mid z)\, p(z), \tag{1}$$

where $\theta^{q}_{n} \sim q_\phi(\theta \mid z_{n})$ is **one** draw per calibration observation, at the same $z_{n}$. Note the two classes share the marginal over $z$ by construction; they differ only in how $\theta$ was produced. Therefore, if and only if $q_\phi(\theta \mid z) = p(\theta \mid z)$ for almost every $z$, the two joints coincide and no classifier can beat chance *anywhere*.

Train a binary classifier $d_{\omega}(\theta, z) \approx \Pr(C=1 \mid \theta, z)$ on this dataset. An MLP is adequate; the existing repo already depends on `scikit-learn`, whose `MLPClassifier` is used for the global C2ST in `smoke_test_gmm.py` and should be reused for consistency.

### 4.3 The local statistic

Global C2ST would stop here and report accuracy. L-C2ST instead *freezes* $z = z_{o}$ and asks whether the classifier can separate the classes **at that observation**:

$$\hat{t}(z_{o}) \;=\; \frac{1}{M}\sum_{m=1}^{M}\Big(d_{\omega}\big(\theta^{q}_{o,m},\, z_{o}\big) - \tfrac{1}{2}\Big)^{2}, \qquad \theta^{q}_{o,m} \sim q_\phi(\theta \mid z_{o}), \tag{2}$$

for each fixed $z_{o}$. Under perfect local calibration $d_{\omega}(\cdot, z_{o}) \equiv \tfrac{1}{2}$ and $\hat{t}(z_{o}) = 0$. Any systematic deviation from chance level indicates flawed inference at that observation.

Implement the mean-absolute variant $\frac{1}{M}\sum_{m}\lvert d_{\omega}(\theta^{q}_{o,m}, z_{o}) - \tfrac{1}{2}\rvert$ as an option; it is more interpretable on the probability scale, and reporting both costs nothing.

**A second, more useful output.** Because $d_{\omega}$ is a function of $\theta$ as well, the map $\theta \mapsto d_{\omega}(\theta, z_{o})$ evaluated over posterior draws shows *where in parameter space* the estimated posterior is wrong — too much mass here, too little there. Return the per-draw probabilities, not just the scalar. This is what turns the diagnostic from a verdict into a debugging tool, and the reference literature calls it out specifically as an advantage over P-P plots.

### 4.4 The null distribution

$\hat{t}(z_{o})$ has no closed-form null, so obtain it by permutation. Under $H_{0}$ the class label carries no information, so:

- Repeat $B$ times ($B \ge 100$): shuffle the labels $C$ across the pooled dataset, retrain the classifier, recompute $\hat{t}_{b}(z_{o})$.
- $p$-value $= \big(\#\{b : \hat{t}_{b} \ge \hat{t}\} + 1\big) / (B+1)$.

The $+1$ in both places matches the convention already used in `embedding_overlap` and keeps the $p$-value from ever being exactly zero.

**This is the expensive part** — $B$ classifier trainings. Two mitigations: keep the classifier small (the discriminative task is low-dimensional relative to the data), and note that the null does not depend on $z_{o}$, so **one null distribution serves every observation**. Compute it once, cache it, reuse across all 216 real embeddings. Make the caching explicit in the API.

### 4.5 L-C2ST-NF, if time allows

When the estimator is a normalizing flow, $\theta$ can be mapped into the flow's base space, $T_{n} = f_\phi^{-1}(\theta^{*}_{n}; z_{n})$. Under a correct posterior, $T_{n} \sim \mathcal{N}(0, I_{p})$ **independently of $z_{n}$**. The class-1 sample can then be drawn directly from the base normal rather than from the estimator, which makes the null distribution analytic in the sense that no estimator sampling is needed per replicate.

Treat this as an optimisation, not the primary deliverable. Implement §4.2–4.4 first, verify it, then add this if the runtime is uncomfortable. `zuko` flows expose the inverse transform; check the installed API rather than assuming, since this repo has already been bitten twice by assumed `sbi`/`zuko` signatures.

---

## 5. LCT — the secondary task

**This section establishes a cheaper alternative worth having alongside.**

### 5.1 Construction

LCT extends expected coverage by conditioning it on the observation. Compute the usual SBC rank $r_{n}$ for each calibration draw, then **regress the rank on $z$**:

$$g(z) \;=\; \mathbb{E}\big[\,\mathbb{I}(r \le \text{level}) \,\big|\, z\,\big], \qquad \text{for each fixed level and each fixed } z. \tag{3}$$

Any regressor works — random forest, MLP. Evaluating $g$ at $z_{o}$ gives local coverage, amortized: once fit, diagnosing a new observation is a forward pass.

### 5.2 Why implement both

LCT is cheaper and simpler but less sensitive: it looks only at a rank, so it inherits the blindness of whatever test quantity generated that rank. Given that this project has already documented parameter-only rank statistics being blind to a data-ignoring posterior, LCT should be built with **data-dependent** test quantities available, exactly as `data_dependent_sbc` is in `npe_diagnostics.py`. Reuse `make_bilinear_test_quantity` rather than writing a new one.

---

## 6. Posterior predictive checks — the third task

**This section establishes what to build once local calibration is done.**

Every diagnostic in this repo, local and global, tests the *inference*. None tests the *model*. Posterior predictive checks close that gap:

1. Draw $\theta_{1},\dots,\theta_{K} \sim q_\phi(\theta \mid z_{o})$.
2. Simulate $x_{k}$ from each $\theta_{k}$.
3. Compare summaries $s(x_{k})$ to $s(x_{o})$.

Requires the simulator callable — Brian2 is available on the cluster and one simulation is affordable, so $K \sim 100$ is realistic. The comparison should use burst statistics that the DSN's training objective did **not** have access to; matching on statistics the encoder was optimised for is close to circular.

Deliver as `npe_ppc.py` with the simulator behind an injected callable, so the module is testable without Brian2 installed. The existing modules all follow this pattern of keeping the expensive dependency at the boundary.

---

## 7. Two facts about the real data that affect this work

**This section establishes constraints the implementer will otherwise not know.**

**Duration mismatch, unresolved.** Simulations were generated with `--simtime 180` (seconds). The real recordings are 20 minutes, to be cut into **200 s non-overlapping windows**, giving $W_{r} = 6$ windows per recording. 180 and 200 are not the same. The two inputs to $h_\psi$ therefore differ in length by 11% before any biology enters, which alone could displace the simulated and real embedding clouds. Flag this to the project owner; the fix is to re-run the campaign at 200 s or to window the real data at 180 s. **Do not build around the mismatch — record it and let it be fixed upstream.**

**The real data has group structure.** With 36 recordings per class and 6 windows each, there are 216 real embeddings, and each group of 6 shares one unknown $\theta$. Consequences for this task:

- Local diagnostics are naturally per-window, so 6 statistics per culture. Decide and document whether to report them separately (showing within-culture variability, which is informative) or aggregate.
- If the six windows are later combined into a single per-culture posterior via a permutation-invariant aggregator, the local diagnostic must be applied to *that* posterior, not to the per-window ones. Do not assume the aggregation exists yet — it does not.

---

## 8. What the validation suite must prove

**This section establishes the acceptance criteria. It is the most important section of this handoff.**

Follow the pattern of `smoke_test_diagnostics.py`: score against **exact analytic posteriors** from `gmm_benchmark.py`, so no network training is required and any failure is attributable to the diagnostic rather than to an approximate reference.

Required tests, in `smoke_test_local.py`:

- **L1 — chance level on an exact posterior.** With $q_\phi$ set to the exact analytic posterior, $\hat{t}(z_{o})$ must be consistent with its null at many $z_{o}$. Report the **rate** over $\ge 12$ seeds, not a single-seed pass: a single-seed assertion on a statistical test fails at rate $\alpha$ by construction, and a flaky test teaches the reader to ignore it. This lesson was learned the hard way in `smoke_test_diagnostics.py` D3 and D5.

- **L2 — detects a global failure.** A uniformly too-narrow posterior must be caught at essentially every $z_{o}$.

- **L3 — THE TEST THAT JUSTIFIES THE WHOLE MODULE. Detects a *local* failure that the global battery misses.** Construct a posterior that is exactly correct for most of the observation space but wrong in a restricted region — for instance, correct except where $\lVert z \rVert$ or some coordinate falls in the top decile. Then assert **all** of the following:
  - marginal SBC passes (global, averaged, and the defect is diluted),
  - expected coverage passes, or is at most marginal,
  - L-C2ST **passes** at $z_{o}$ drawn from the healthy region,
  - L-C2ST **fails** at $z_{o}$ drawn from the corrupted region.

  If L3 cannot be made to pass, the module has no reason to exist and that finding should be reported rather than worked around. Tune the size of the corrupted region so that the global tests genuinely pass — if they also fail, the defect is too large and the test proves nothing.

- **L4 — false-positive rate.** On an exact posterior, the rejection rate at nominal $\alpha$ must be near $\alpha$ over many seeds. A local test that fires everywhere is worse than none, because it will fire on the real data and be believed.

- **L5 — the null distribution is correctly calibrated.** Verify the permutation null empirically: the observed statistic under $H_{0}$ should sit in the bulk, and its quantile should be approximately uniform across seeds.

- **L6 — LCT agrees with L-C2ST on gross failures** and is measurably less sensitive on subtle ones. This quantifies the trade-off rather than asserting it.

- **L7 — reuse of the cached null is valid.** Assert that a null computed once and reused across observations gives the same decisions as one recomputed per observation, since §4.4 recommends the reuse for cost reasons and it must be shown safe.

Add a negative control wherever a threshold is asserted, as `smoke_test_gmm.py` G6 does: apply the same criterion to a deliberately wrong answer and require it to fail. A threshold loose enough to accommodate training noise may also accommodate an estimator that learned nothing.

---

## 9. Interfaces to respect

**This section establishes how the new module fits the existing code.**

Inputs are already persisted by every run in `diagnostics_data.npz`:

| key | shape | meaning |
|---|---|---|
| `theta_cal` | $(N_{\text{cal}}, p)$ | $\theta^{*}_{n}$, class 0 |
| `posterior_samples` | $(N_{\text{cal}}, M, p)$ | draws at each $z_{n}$; take one per $n$ for class 1 |
| `Z_cal` | $(N_{\text{cal}}, E)$ | $z_{n}$ |
| `z_sim`, `z_real_ok`, `z_real_shifted` | $(\cdot, E)$ | embedding pools |
| `param_names`, `coord`, `bounds_theta`, `embedding_dim` | — | the contract |

So a first working version needs **no model reload and no simulator**. Load the npz, build (1), train, test. That is the whole primary deliverable.

Follow these conventions, which the existing modules all obey:

- A results dataclass with `passes`, `summary()`, and the raw arrays needed for plotting.
- Pure numpy after the classifier; keep any torch or sklearn dependency at the boundary so the statistics can be unit-tested on synthetic arrays.
- No plotting in the module. Add figures to `npe_plots.py` as `12_local_c2st.png` and `13_lct.png`, continuing the numbering, and register them in `save_all_plots`.
- Wire the new suite into `run_tests.sh` as an additional stage, and into `jobs/smoke_test.pbs` by extension (that script delegates to the runner).

---

## 10. Traps recorded from this project's history

**This section establishes what has already gone wrong, so it does not go wrong again.**

1. **Assert on the anchor when patching programmatically.** Two `str.replace` operations in this project silently did nothing because the anchor did not match, and one slice ran to end-of-file and deleted half a module. Both produced plausible-looking wrong results. Always `assert old in s`.
2. **Look at every figure.** Three plotting defects passed all automated checks and were found only by opening the PNG: a projection that hid a real distribution shift, a coverage plot whose axis was $1-\text{coverage}$ while labelled "coverage", and a plot that crashed on skewed marginals.
3. **Jitter discrete rank statistics before any continuous goodness-of-fit test.** Ranks take $M+1$ values; an unjittered KS test against a continuous uniform manufactures rejections.
4. **Apply family-wise correction when testing many axes or many observations.** At 27 axes and $\alpha = 0.005$, a correct posterior trips something ~13% of the time. `family_verdict()` exists; use it. This applies with force here: 216 real embeddings means 216 local tests.
5. **`sbi`'s `log_prob` defaults to `norm_posterior=True`**, which draws 10 000 rejection samples per observation per ensemble member. It consumed 3.7 of the synthetic trial's 3.9 hours and is a no-op under the bounded-support transform. `posterior_log_probs` in `npe_diagnostics.py` already defaults it to `False`.
6. **Verify library APIs against the installed version.** Pinned: `sbi==0.27.0`, `zuko==1.6.0`. This project has twice been wrong about a signature recalled rather than checked.
7. **ASCII-only, LF endings.** Non-negotiable; see §1.1.

---

## 11. Summary of results

| # | Statement | Established in |
|---|---|---|
| S1 | Local calibration is the highest-value remaining diagnostic because the scientific claim is per-culture and global tests average the evidence away. | §3 |
| S2 | L-C2ST trains a classifier to separate the true joint from the estimated joint, then evaluates its output at a fixed $z_{o}$. | §4.2, eq. (1) |
| S3 | The statistic is a deviation from chance level; zero is perfect and large is bad — the opposite reading from a C2ST accuracy. | §4.3, eq. (2) |
| S4 | The null is obtained by label permutation and does not depend on $z_{o}$, so one null serves all observations. | §4.4 |
| S5 | Returning per-draw classifier probabilities turns the test into a map of *where* in parameter space the posterior is wrong. | §4.3 |
| S6 | LCT is the cheaper alternative and must use data-dependent test quantities to avoid a known blindness. | §5 |
| S7 | The whole module is justified by test L3 and by nothing else. | §8 |
| S8 | No new simulations are required; the calibration set already exists in `diagnostics_data.npz`. | §9 |
| S9 | Simulations are 180 s but real windows will be 200 s; this must be reconciled upstream. | §7 |
| S10 | The real data has 6 windows per culture sharing one $\theta$, so local statistics come in groups of 6. | §7 |

---

## 12. Open points, caveats, and assumptions

1. **The classifier's capacity is a free parameter with no principled setting.** Too weak and it misses real differences; too strong and it overfits the calibration set, inflating the statistic. The permutation null partially absorbs this, since the null classifiers have the same capacity, but the trade-off is not eliminated. Test sensitivity to capacity in L4.
2. **$N_{\text{cal}} = 500$ may be too small.** The reference worked example used 20 000 additional simulations for L-C2ST. Existing runs have 500. Determining the minimum viable $N_{\text{cal}}$ is a worthwhile early experiment and costs nothing but compute on synthetic data.
3. **The diagnostic can fail through its own convergence** rather than the estimator's, a caveat the reference literature states explicitly. L1 and L4 are the guard, but a null result should always be reported with the classifier's training diagnostics attached.
4. **L3's construction is not specified in detail** because the right corrupted region depends on the benchmark's geometry. Getting it to the point where global tests genuinely pass and the local test genuinely fires may take iteration; that iteration is the substance of the task, not an obstacle to it.
5. **L-C2ST-NF assumes the estimator is an invertible flow.** True for the current `zuko_nsf` estimator but would break if the architecture changed. Keep it behind a capability check rather than assuming.
6. **Nothing here addresses model misspecification** — the possibility that no $\theta$ reproduces the real data. Local calibration tests the inference against the simulator's own joint. The summary-space MMD gate (`embedding_overlap`) and posterior predictive checks are the layers for that, and the gate is still the first thing to run when real embeddings arrive.
7. **Ordering assumption.** This document assumes the overlap gate has not yet been run on real data, because the real embeddings do not exist yet. If they arrive mid-task, **stop and run the gate first** — it is a hard go/no-go, needs no training, and can invalidate everything downstream.

---

## 13. References

**Project knowledge base, read directly:**

- *Simulation-Based Inference: A Practical Guide.* — The L-C2ST construction quoted as eq. (1) here, the description of LCT as rank regression on observation space, the statement that local checks are more powerful but costlier and can introduce their own errors, and the worked 31-parameter example in which L-C2ST was trained with 20k additional simulations and the ensemble posterior fell inside the confidence region of the test statistic.
- Zhao, Dalmasso, Izbicki, Lee. *Diagnostics for Conditional Density Models and Bayesian Inference Algorithms.* — The Local Coverage Test and Amortized Local P-P plots.
- Hermans et al. (2022). *A Trust Crisis in Simulation-Based Inference?* TMLR. — Global coverage, conservativeness, ensembling.

**Cited by the above, not read directly — obtain before implementing:**

- Linhart, Gramfort, Rodrigues. *L-C2ST: Local diagnostics for posterior approximations in simulation-based inference.* NeurIPS 36 (2024). arXiv:2306.03580. — **The primary reference for this task.** The construction in §4 is reconstructed from the Practical Guide's summary of it, not from the paper itself, and the paper should be read before implementation for the details of the null distribution and the NF variant.
- Lopez-Paz, Oquab. *Revisiting classifier two-sample tests.* ICLR 2017. arXiv:1610.06545.

**Measured in this project, not from any source:** the synthetic trial results quoted in the abstract and §3 (nominal 90% delivering 61% coverage; marginal SBC passing 0/27), reproducible from `synthetic_trial_20260811_152440`.

**Stated from reasoning rather than a source:** the argument in §3 for prioritising this task, the caching argument in §4.4, the design of test L3, and every item in §10.
