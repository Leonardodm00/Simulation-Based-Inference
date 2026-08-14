# Diagnostic Metrics for Amortized Neural Posterior Estimation

**Date:** 2026-08-12
**Purpose:** Reference document explaining, in depth, every diagnostic metric implemented in `npe_diagnostics.py`, what each one can and cannot detect, and the measured evidence for those claims.

## Abstract

**Scientific question.** Given an amortized neural posterior estimator $q_\phi(\theta \mid z)$ trained on simulations and intended for use on real recordings, what evidence can be gathered that its output is trustworthy, and what failure modes remain invisible to that evidence?

**Scope.** This document covers the five diagnostic layers implemented in `npe_diagnostics.py`: summary-space overlap, rank-based calibration (marginal SBC, expected coverage, data-dependent SBC), TARP, informativeness (contraction and the information spectrum), and multiple-testing control. For each: the definition, the theorem or argument that motivates it, what it detects, what it provably cannot detect, and the empirical measurements made on a benchmark with an exact analytic posterior.

**Deliberately excluded.** The estimator architecture, the training procedure, sequential (TSNPE) methods, local calibration methods (L-C2ST, LC2ST-NF, local coverage tests), posterior predictive checks that require the simulator, and the scientific interpretation of any particular posterior. Also excluded: instructions for running the code, which are in `README.md`.

**The single most important claim.** The standard diagnostic battery — marginal SBC, expected coverage, and TARP as usually implemented — is *provably blind* to a posterior that ignores the observation. Two of these are proved blind in the peer-reviewed literature; the third failure is a property of how reference points are chosen and is measured here. Detecting this failure requires a test quantity that depends on the observation, and §6 gives two that work.

---

## 1. Notation and Symbols

| Symbol | Name / Meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $\theta$ | Parameter vector (inference coordinates) | $\theta \in \Theta \subseteq \mathbb{R}^{p}$ | mixed | §3 |
| $\theta^{*}_{i}$ | Ground-truth parameter for calibration draw $i$ | $\theta^{*}_{i} \in \Theta$ | mixed | §5.1 |
| $\theta^{q}_{i,m}$ | $m$-th posterior draw for observation $i$ | $\theta^{q}_{i,m} \in \Theta$ | mixed | §5.1 |
| $p$ | Parameter dimension | $p \in \mathbb{N}$ | dimensionless | §3 |
| $x$ | Raw observation (a recording) | $x \in \mathcal{X}$ | mixed | §3 |
| $z$ | Summary / embedding of $x$ | $z = h_\psi(x) \in \mathbb{S}^{E-1} \subset \mathbb{R}^{E}$ | dimensionless | §4 |
| $h_\psi$ | Frozen encoder with weights $\psi$ | $h_\psi : \mathcal{X} \to \mathbb{S}^{E-1}$ | — | §4 |
| $E$ | Embedding dimension | $E \in \mathbb{N}$ | dimensionless | §4 |
| $p(\theta)$ | Prior | density on $\Theta$ | dimensionless | §3 |
| $p(\theta \mid z)$ | True posterior given the summary | density on $\Theta$, for each fixed $z$ | dimensionless | §3 |
| $q_\phi(\theta \mid z)$ | Estimated posterior, weights $\phi$ | density on $\Theta$, for each fixed $z$ | dimensionless | §3 |
| $N$ | Number of calibration observations | $N \in \mathbb{N}$ | dimensionless | §5.1 |
| $M$ | Posterior draws per observation | $M \in \mathbb{N}$ | dimensionless | §5.1 |
| $f$ | Test quantity | $f : \Theta \to \mathbb{R}$ or $f : \Theta \times \mathbb{R}^{E} \to \mathbb{R}$ | varies | §5.1 |
| $r_{i}(f)$ | Rank of the truth among draws, under $f$ | $r_{i}(f) \in \{0, \dots, M\}$ | dimensionless | §5.1, eq. (2) |
| $u_{i}$ | Jittered fractional rank | $u_{i} \in [0,1]$ | dimensionless | §5.2, eq. (3) |
| $\alpha$ | Credibility level / significance level | $\alpha \in (0,1)$ | dimensionless | §5.3 |
| $\Theta_{q}(1-\alpha)$ | $(1-\alpha)$ highest-posterior-density region of $q_\phi$ | subset of $\Theta$, for each fixed $z$ | — | §5.3, eq. (4) |
| $\theta_{r}$ | Reference point for a positionable region | $\theta_{r} \in \Theta$ | mixed | §5.5 |
| $\theta_{r}(z)$ | Reference point as a *function of* the observation | $\theta_{r} : \mathbb{R}^{E} \to \Theta$ | mixed | §5.5 |
| $W$ | Fixed random matrix defining a bilinear test quantity | $W \in \mathbb{R}^{p \times E}$ | dimensionless | §6.2, eq. (8) |
| $k(\cdot,\cdot)$ | Gaussian RBF kernel | $k : \mathbb{R}^{E} \times \mathbb{R}^{E} \to (0,1]$ | dimensionless | §4.1, eq. (1) |
| $\sigma$ | Kernel bandwidth (median heuristic) | $\sigma \in \mathbb{R}_{>0}$ | dimensionless | §4.1 |
| $\mathrm{MMD}$ | Maximum mean discrepancy | $\mathrm{MMD} \in \mathbb{R}_{\ge 0}$ | dimensionless | §4.1 |
| $c_{k}$ | Posterior contraction on axis $k$ | $c_{k} \in (-\infty, 1]$ | dimensionless | §7.1, eq. (9) |
| $\mu_{i}$ | Posterior mean for observation $i$ | $\mu_{i} \in \mathbb{R}^{p}$ | mixed | §7.2 |
| $\lambda_{j}$ | $j$-th generalised eigenvalue of the information spectrum | $\lambda_{j} \in [0,1]$ | dimensionless | §7.2, eq. (10) |
| $I(\theta; z)$ | Mutual information between parameters and summary | $\ge 0$ | nats | §7.2 |
| $\tilde{p}_{k}$ | Holm-Bonferroni adjusted $p$-value for axis $k$ | $\tilde{p}_{k} \in [0,1]$ | dimensionless | §8, eq. (11) |

### 1.1 Conventions

- **Log base** is natural throughout.
- **Ranks** are counts in $\{0, \dots, M\}$, so a rank statistic has $M+1$ possible values; this discreteness is why jitter is required before any continuous goodness-of-fit test (§5.2).
- **"Blind"** means: the test statistic has the same distribution under the broken estimator as under the correct one, so no sample size helps. It is a statement about the *test*, not about statistical power.
- **"Detects at chance"** means the empirical rejection rate equals the nominal significance level.
- **Quantifiers** are stated explicitly: a claim "for each fixed $z$" is not the same as one holding "in expectation over $z$", and that distinction is the whole content of §6.
- Conditioning is never suppressed: $q_\phi(\theta \mid z)$ is always written with its conditioning variable, even where the context would make it obvious.

---

## 2. Glossary

Ordered by first appearance, because the concepts build on one another.

- **Amortized inference.** A single estimator trained once over the whole prior, then evaluated on any observation without retraining. Its practical consequence for diagnostics is decisive: calibration can be assessed over hundreds of observations at no simulation cost, whereas a non-amortized method must repeat the whole training pipeline per observation. Operative in §3.
- **Summary / embedding.** A learned, fixed-dimensional compression $z = h_\psi(x)$ of a raw recording, used as the conditioning variable. **Everyday-meaning warning:** "summary" does not imply *sufficient*; a summary trained for some other objective (classification, say) can discard exactly the information the parameters live in. Operative in §4 and §6.
- **Calibration.** The property that stated uncertainty matches realised frequency. **Everyday-meaning warning:** a calibrated posterior need not be *informative* — the prior itself is perfectly calibrated. Operative in §5 and §7.
- **Simulation-based calibration (SBC).** A rank-uniformity test: for a correct posterior, the rank of the truth among posterior draws is uniform. Operative in §5.1.
- **Expected coverage probability.** The probability that the true parameter falls inside the estimator's $(1-\alpha)$ credible region, averaged over the joint $p(\theta, x)$. Operative in §5.3.
- **Conservative estimator.** One whose expected coverage is at least the credibility level, i.e. credible regions that are too *large*. This is the safe direction of error: it fails to reject implausible values rather than wrongly rejecting plausible ones. Operative in §5.3.
- **Overconfident estimator.** Coverage below the credibility level; credible regions too small. The dangerous direction. Operative in §5.3.
- **HPD region.** The smallest-volume region containing $(1-\alpha)$ of the posterior mass. Operative in §5.3.
- **Positionable credible region generator.** A region generator that can be placed at an arbitrary reference point $\theta_{r}$, shrinking to $\{\theta_{r}\}$ as $\alpha \to 1$. HPD regions are **not** positionable, and that single fact is the source of the most important blind spot in this document. Operative in §5.5 and §6.1.
- **TARP (Tests of Accuracy with Random Points).** A coverage test built on positionable spherical regions around reference points, requiring no density evaluation. Operative in §5.5.
- **Data-dependent test quantity.** A test quantity $f(\theta, z)$ depending on both parameters and observation, as opposed to $f(\theta)$. The distinction is what separates a test that can see a data-ignoring posterior from one that cannot. Operative in §6.
- **MMD (maximum mean discrepancy).** A kernel two-sample statistic; zero if and only if two distributions coincide, for a characteristic kernel. Operative in §4.1.
- **Posterior contraction.** One minus the ratio of posterior to prior variance on an axis. Measures information gain, not correctness. Operative in §7.1.
- **Deep ensemble.** Several independently initialised and trained estimators combined as a mixture. Raises expected coverage but does not guarantee conservativeness. Operative in §9.
- **Family-wise error rate.** The probability of at least one false rejection across a family of tests. Relevant because a per-axis diagnostic on $p$ axes is $p$ simultaneous tests. Operative in §8.

---

## 3. The framework, and why diagnostics are needed at all

**This section establishes what is being tested and what "correct" would mean.**

We have a simulator $\theta \mapsto x$, a fixed encoder $z = h_\psi(x)$, and an estimator $q_\phi(\theta \mid z)$ trained on pairs $(\theta_i, z_i)$ drawn from the joint. The object of interest is the true posterior $p(\theta \mid z)$, and the estimator is correct if

$$q_\phi(\theta \mid z) = p(\theta \mid z) \quad \text{for almost every } z \text{ and all } \theta \in \Theta. \tag{0}$$

The difficulty specific to this setting is that neither the true posterior nor the likelihood is available, so (0) cannot be checked directly. Every diagnostic below is a *necessary condition* for (0) — a test the estimator would pass if it were correct. None is sufficient on its own, and §6 shows that a large subset of them, taken together, is still not sufficient in a way that matters practically.

A prior consideration, which is why this document treats the peer-reviewed evidence as load-bearing rather than decorative: the literature in the project knowledge base establishes empirically that **all benchmarked SBI algorithms produced non-conservative posterior approximations on at least one problem setting**, with the pathology most prominent at small simulation budgets, and that a large budget does not guarantee conservativeness either (Hermans et al., *A Trust Crisis in Simulation-Based Inference*, TMLR 2022). Diagnostics are therefore not a formality; the failure they look for is the observed default rather than an edge case.

---

## 4. Layer 1 — Summary-space overlap (the gate)

**This section establishes whether the estimator is being asked a question it was trained to answer.**

### 4.1 Definition

Given simulated embeddings $\{z^{\text{sim}}_{i}\}_{i=1}^{n_{\text{sim}}}$ and real embeddings $\{z^{\text{real}}_{j}\}_{j=1}^{n_{\text{real}}}$, with the Gaussian RBF kernel $k(a,b) = \exp\!\big(-\lVert a-b\rVert^{2} / (2\sigma^{2})\big)$, the biased squared MMD estimate is

$$\widehat{\mathrm{MMD}}^{2} = \frac{1}{n_{\text{sim}}^{2}}\sum_{i,i'} k(z^{\text{sim}}_{i}, z^{\text{sim}}_{i'}) + \frac{1}{n_{\text{real}}^{2}}\sum_{j,j'} k(z^{\text{real}}_{j}, z^{\text{real}}_{j'}) - \frac{2}{n_{\text{sim}} n_{\text{real}}}\sum_{i,j} k(z^{\text{sim}}_{i}, z^{\text{real}}_{j}). \tag{1}$$

The null distribution is obtained by repeatedly drawing subsets of size $n_{\text{real}}$ from a *disjoint* part of the simulated pool and recomputing (1) against a fixed simulated reference. The $p$-value is $(\#\{\text{null} \ge \text{observed}\} + 1)/(n_{\text{null}} + 1)$; the $+1$ in both places reflects that the observed value is itself a draw under the null and keeps the $p$-value from ever being exactly zero.

Two implementation choices, each with a reason:

- **The biased estimator is used deliberately.** The unbiased version is undefined for a single observation, and a single real recording is an important practical case. Excluding it to gain unbiasedness would trade away the use case for a property that does not matter here.
- **Reference and null draws come from disjoint halves of the simulated pool.** Reusing the same points on both sides biases the null low and makes the test anticonservative.

A second, non-kernel view is also computed: the geodesic nearest-neighbour distance $\arccos\big(\max_{i} \langle z^{\text{real}}_{j}, z^{\text{sim}}_{i}\rangle\big)$, compared against the simulation-to-simulation baseline. Geodesic rather than Euclidean because the embedding is L2-normalised onto $\mathbb{S}^{E-1}$.

### 4.2 Why this must run first

The encoder in this project was trained on **real** recordings. The usual SBI concern is therefore inverted: it is the *simulations* that may be out of distribution for the encoder, not the reverse. An estimator trained on simulated embeddings will be internally calibrated on simulations and will report nothing about this — every diagnostic in §5 through §7 is computed on simulated draws and cannot see the gap. If $z^{\text{real}}$ falls off the simulated manifold, the flow extrapolates and the posterior is arbitrary while every other test passes.

This is the same concern the misspecification literature in the project knowledge base addresses by prescribing a structured summary space and testing real data against it (Schmitt et al.), with the MMD acting as a proxy for posterior estimation error — a quantity otherwise unknowable without ground truth.

### 4.3 Measured behaviour

Both directions were tested, because a gate that always fires is as useless as one that never does (test D1):

| query set | $p$-value | verdict |
|---|---|---|
| drawn from the same process | 0.223 | no rejection (correct) |
| concentrated on a spherical cap | 0.0033 | rejection (correct) |

### 4.4 A visualisation trap worth recording

The natural way to picture this is a PCA projection of both clouds. **That is misleading and was corrected after inspection.** PCA on the simulated cloud picks the directions of greatest *simulated* variance, which need not be the direction in which the real data differs: a query set the gate correctly rejected at $p = 0.0033$ looked perfectly well mixed in the top two PCs, because the shift lay in a low-variance direction.

The figure now projects onto the whitened difference-of-means direction instead. Note the asymmetry this creates, which must be stated wherever the figure appears: **overlap in this view is strong evidence of no gap; separation is weak evidence**, because a discriminative axis can overfit at small $n_{\text{real}}$.

---

## 5. Layer 2 and 3 — Rank-based calibration

**This section establishes the standard battery: what each test is, and what each one sees.**

### 5.1 The common structure

Every test in this section is a rank-uniformity test and differs from the others *only in the choice of test quantity $f$*. Draw $\theta^{*}_{i} \sim p(\theta)$, simulate $z_{i}$, sample $\{\theta^{q}_{i,m}\}_{m=1}^{M} \sim q_\phi(\theta \mid z_{i})$, and form

$$r_{i}(f) = \sum_{m=1}^{M} \mathbb{I}\big(f(\theta^{q}_{i,m}) < f(\theta^{*}_{i})\big) \in \{0, 1, \dots, M\}, \qquad \text{for each fixed } i \in \{1,\dots,N\}. \tag{2}$$

If $q_\phi(\theta \mid z) = p(\theta \mid z)$ for almost every $z$, then $\theta^{*}_{i}$ and the draws are exchangeable given $z_{i}$, so $r_{i}(f)$ is uniform on $\{0,\dots,M\}$ for every $f$. Non-uniformity for *any* $f$ is evidence against (0).

Recognising that SBC, expected coverage, and TARP are one procedure under three choices of $f$ is what makes §6 comprehensible: they share a common blind spot precisely because they share this structure and all three choose $f$ badly for one particular failure.

### 5.2 Testing uniformity, and why jitter is mandatory

Ranks take $M+1$ discrete values, so their exact null is discrete uniform. Comparing them against a continuous uniform inflates the Kolmogorov-Smirnov statistic and produces spurious rejections. The fix is to jitter before testing:

$$u_{i} = \frac{r_{i}(f) + \varepsilon_{i}}{M+1}, \qquad \varepsilon_{i} \sim \mathrm{Unif}(0,1) \text{ independent}, \qquad \text{for each fixed } i. \tag{3}$$

Then $u_{i} \sim \mathrm{Unif}(0,1)$ exactly under the null, and a KS test is valid. Measured false-positive rate on an exact analytic posterior: **0 rejections in 240 tests** (40 replicates $\times$ 6 marginals) against a nominal 0.5% — confirming the implementation is not anticonservative (test D3).

### 5.3 Marginal SBC — $f(\theta) = \theta_{k}$

One rank per parameter coordinate. Detects which *marginals* are too narrow, too broad, or biased, and the shape of the rank histogram names which:

| histogram shape | meaning |
|---|---|
| U-shaped (truth lands in the tails too often) | posterior too narrow — overconfident |
| dome-shaped (truth lands centrally too often) | posterior too broad — underconfident |
| sloped / shifted mean | posterior biased |
| flat | consistent with correct |

Blind to parameter *correlations*: a posterior with correct marginals and entirely wrong dependence structure passes. Verified to detect and correctly *name* all three failure shapes (test D4).

### 5.4 Expected coverage — $f(\theta) = \log q_\phi(\theta \mid z)$

Using the estimator's own log-density as the test quantity makes the rank statistic equivalent to the HPD expected coverage probability,

$$\mathrm{ECP}(\alpha) = \mathbb{E}_{p(\theta, x)}\Big[\mathbb{I}\big(\theta \in \Theta_{q_\phi(\theta \mid z)}(1-\alpha)\big)\Big], \tag{4}$$

which for a correct posterior equals $1-\alpha$ for all $\alpha \in (0,1)$. Plotted against $\alpha$, a correct estimator traces the diagonal; **curves above the diagonal are conservative, curves below are overconfident**. This is the joint counterpart to §5.3: it is sensitive to wrong correlations, which marginal SBC cannot see, but it cannot say *which* parameter is responsible.

A practical note from the project knowledge base: a global coverage analysis also permits post-training calibration, by replacing the level-$\alpha$ region with the level whose empirical coverage is the desired one (Hermans et al.).

### 5.5 TARP — positionable regions, and why the reference points matter enormously

TARP replaces HPD regions with spheres around a reference point $\theta_{r}$:

$$\mathcal{D}_{\theta_{r}}(\alpha, z) = \{\theta \in \Theta : d(\theta, \theta_{r}) \le R(\alpha, z)\}, \tag{5}$$

and computes, for each observation, the fraction of posterior draws closer to $\theta_{r}$ than the truth:

$$f_{i} = \frac{1}{M}\sum_{m=1}^{M} \mathbb{I}\big(\lVert \theta^{q}_{i,m} - \theta_{r,i} \rVert < \lVert \theta^{*}_{i} - \theta_{r,i} \rVert\big), \qquad \text{for each fixed } i. \tag{6}$$

For a correct posterior the $f_{i}$ are uniform on $[0,1]$. Two properties make this attractive: it needs no density evaluation, so it applies to estimators that cannot be evaluated pointwise; and unlike HPD regions, spherical regions around a free point *are* positionable.

The theorem that motivates TARP (Lemos et al., *Sampling-Based Accuracy Testing of Posterior Estimators for General Inference*, in the project knowledge base) states that correct expected coverage for a positionable generator, **for all functions $\theta_{r}(x)$ assigning positions as a function of the observation**, implies $q_\phi(\cdot \mid x) = p(\cdot \mid x)$.

**The quantified clause is doing all the work, and it is easy to miss.** The theorem requires the reference points to be a function *of the observation*. A natural implementation draws them uniformly at random, independently of $x$ — and that implementation does not satisfy the hypothesis. §6.3 measures exactly how much this costs.

---

## 6. The blind spot

**This section establishes the central claim: what the standard battery cannot see, why, and what to use instead.**

### 6.1 The failure mode

Consider an estimator that ignores the observation entirely:

$$q_\phi(\theta \mid z) = p(\theta) \qquad \text{for every } z. \tag{7}$$

This is not a contrived example. It is the limiting case of an *insufficient summary*: if $z$ carries little information about $\theta$, the trained posterior approaches the prior, and the same argument applies in weakened form to any estimator that uses only part of the information in $x$. For a project whose encoder was trained by metric learning on class labels rather than for parameter recovery, this is the live risk, not a hypothetical one.

Three of the standard tests pass on (7), each for a slightly different reason:

**Marginal SBC passes.** Under (7), both $\theta^{*}_{i}$ and every draw $\theta^{q}_{i,m}$ are marginally draws from $p(\theta)$. Marginally over $z$ they are exchangeable, so the rank (2) is uniform for any $f$ depending on $\theta$ alone. This is the content of Theorem 7 of Modrák et al. (*Bayesian Analysis*, doi:10.1214/23-ba1404): SBC with test quantities that do not depend on the data cannot detect a posterior that uses only some function of the data — including none of it.

**Expected coverage passes.** Under (7), $f(\theta) = \log q_\phi(\theta \mid z) = \log p(\theta)$ carries no $z$-dependence at all, so the HPD region is the prior's region for every observation, and a draw $\theta^{*} \sim p(\theta)$ lands inside the $(1-\alpha)$ region at rate exactly $1-\alpha$. The TARP paper derives precisely this: for $\hat{p}(\theta \mid x) = p(\theta)$, the HPD generator becomes independent of $x$ and the expected coverage is exactly $1-\alpha$ — perfect, and perfectly uninformative. The root cause is named there too: HPD generators are **not positionable**, which is exactly the hypothesis their Theorem 3 requires.

**TARP with $x$-independent reference points also passes.** Given a reference point $\theta_{r}$ that does not depend on the observation, $\theta^{*}_{i}$ and a prior draw remain exchangeable, so (6) is uniform. Positionability of the *region shape* is not sufficient; the *position* must vary with the observation.

### 6.2 What does work

Two test quantities detect (7). Both work for the same underlying reason: **$\theta^{*}_{i}$ is correlated with $z_{i}$ because $z_{i}$ was generated from it, whereas a draw from a data-ignoring $q_\phi$ is not.** Any statistic that couples $\theta$ to $z$ breaks the exchangeability that the blind tests rely on.

**(a) A data-dependent bilinear test quantity.** For a fixed random $W \in \mathbb{R}^{p \times E}$,

$$f_{W}(\theta, z) = \theta^{\mathsf{T}} W z. \tag{8}$$

Modrák et al. recommend the joint log-likelihood as the default data-dependent quantity; in simulation-based inference the likelihood is unavailable by construction, so (8) is the substitute used here. Each $W$ is fixed once and reused across all observations, so $f_{W}$ is a genuine function of $(\theta, z)$ rather than something refitted per observation.

**(b) TARP with $x$-dependent reference points.** Set $\theta_{r}(z_{i})$ to a cheap readout of $z_{i}$ — in the implementation, a least-squares linear map fit on one half of the calibration set and applied to the other, with the halves swapped so every observation receives an out-of-fit reference. The split is not optional: fitting and evaluating on the same rows would let the readout memorise $\theta^{*}$ and manufacture apparent miscalibration.

### 6.3 Measured evidence

All measurements on a 3-component Gaussian mixture benchmark with an *exact analytic* posterior, $p = 6$, observation dimension 3, $N = 600$, $M = 128$, 12 independent replicates, Holm-Bonferroni corrected at $\alpha = 0.05$ (test D5).

| test | test quantity | detections on (7) | verdict |
|---|---|---|---|
| Marginal SBC | $f(\theta) = \theta_{k}$ | **0 / 12** | blind |
| Expected coverage | $f(\theta) = \log q_\phi(\theta \mid z)$ | **0 / 12** | blind |
| TARP, random references | $f$ via $\theta_{r} \perp z$ | **0 / 12** | blind |
| TARP, $x$-dependent references | $f$ via $\theta_{r}(z)$ | **12 / 12** | detects |
| Data-dependent SBC | $f_{W}(\theta, z) = \theta^{\mathsf{T}} W z$ | **12 / 12**, median adjusted $p = 1.8 \times 10^{-6}$ | detects |
| Posterior contraction | not a rank test | $\max_{k} c_{k} = 0.086$ | detects |

Both working tests were checked for false positives on the *exact* posterior: **0 / 12** each. A test that fires on everything would detect (7) trivially and mean nothing.

A separate 40-replicate run measured the marginal-SBC detection rate on (7) as **0.0042** against a chance level of 0.005 — that is, indistinguishable from chance, which is the strongest possible statement of blindness.

For TARP specifically, a direct head-to-head at $N = 600$ over 12 replicates: random references detected 1/12 with median $p = 0.29$; $x$-dependent references detected 12/12 with median $p = 3.8 \times 10^{-10}$ and 0 false positives.

### 6.4 Practical consequence

**Never run marginal SBC alone, and never treat expected coverage as a safety net for it.** The minimum defensible battery is:

1. marginal SBC (which axes are miscalibrated),
2. expected coverage (joint structure and correlations),
3. **data-dependent SBC** and/or **TARP with $x$-dependent references** (is the observation being used at all),
4. **posterior contraction** (is the answer informative, §7).

Items 3 and 4 are the ones that carry information the others structurally cannot.

---

## 7. Layer 4 — Informativeness

**This section establishes that calibration and informativeness are different properties, and how to measure the second.**

### 7.1 Posterior contraction

$$c_{k} = 1 - \frac{\mathbb{E}_{i}\big[\mathrm{Var}(\theta_{k} \mid z_{i})\big]}{\mathrm{Var}_{p(\theta)}(\theta_{k})}, \qquad \text{for each fixed } k \in \{1,\dots,p\}. \tag{9}$$

$c_{k} = 0$ means the posterior marginal is as wide as the prior — the data said nothing about that axis. $c_{k} = 1$ means it is fully determined. Negative values mean the posterior is *wider* than the prior, which is possible and worth flagging.

This is the layer calibration cannot reach. A posterior equal to the prior is perfectly calibrated and completely uninformative; only (9) separates the two. The project knowledge base makes the same point from the coverage side: coverage is limited in its ability to determine information gain, and an estimator whose posteriors equal the prior has no gain while showing coverage exactly at the credibility level — so a complete analysis must be complemented with a measure such as the expected information gain $\mathbb{E}_{p(\theta,x)}[\log p(\theta \mid x) - \log p(\theta)]$ (Hermans et al.).

Measured: $\max_{k} c_{k} = 0.041$ on a prior-as-posterior, versus $\max_{k} c_{k} = 0.429$ on the exact posterior for the same problem (test D7).

### 7.2 The information spectrum, and a correction

Let $\mu_{i} = \mathbb{E}[\theta \mid z_{i}]$. Since $\mu_{i}$ is a function of $z_{i}$ alone, the covariance of $\{\mu_{i}\}$ across the calibration set is the part of the prior covariance the observation can account for. The generalised eigenvalues

$$\lambda_{j} = \mathrm{eig}_{j}\Big(\mathrm{Cov}_{p(\theta)}(\theta)^{-1/2}\, \mathrm{Cov}_{i}(\mu_{i})\, \mathrm{Cov}_{p(\theta)}(\theta)^{-1/2}\Big) \in [0,1] \tag{10}$$

give the fraction of prior variance explained along each eigendirection. Whitening by the prior makes them dimensionless and comparable across axes with different units — essential when some axes are log-coordinates and others linear.

**A correction to an intuitive but wrong reading.** It is tempting to treat the count $\#\{j : \lambda_{j} > \tau\}$ as a hard test of the bound "an embedding of dimension $E$ constrains at most $E-1$ parameter directions." That reading is **wrong for a nonlinear estimator**, and the error was made and then corrected during development.

The map $z \mapsto \mathbb{E}[\theta \mid z]$ does have an image of *intrinsic* dimension at most $E-1$, since $z$ carries only that many degrees of freedom. But (10) measures the *linear* rank of the covariance of that image, and a curved $d$-dimensional manifold spans more than $d$ linear dimensions. Measured directly: for $z$ on $\mathbb{S}^{11}$ and a linear $f(z)$, the covariance rank is exactly 11; for a nonlinear $f(z)$ on the same $z$, it is 12 or more. A normalizing flow is nonlinear, so an effective rank above $E-1$ is expected behaviour, not evidence of a leak.

What survives is the information-theoretic statement: by the data-processing inequality $I(\theta; z) \le I(\theta; x)$, and $z$ carries at most $E-1$ real numbers, so the total information is capped however it is distributed. A nonlinear map can spread that budget thinly across all $p$ linear directions rather than concentrating it in $E-1$ of them.

**Use the spectrum descriptively** — how fast the constrained variance decays — and judge "is this parameter informed?" from the per-axis contraction (9).

The bound *is* assertable in one place: on a conjugate linear-Gaussian benchmark, each posterior mean is an affine function of the observation, so the image is a flat subspace and its linear rank equals its intrinsic dimension. On such a benchmark with 6 parameters observed through a rank-3 map, the measured spectrum is $[0.927, 0.839, 0.460, 0.009, 0.008, 0.008]$ — effective rank 3, a clean cliff exactly at the observation dimension (test D8).

---

## 8. Multiple-testing control

**This section establishes why a per-axis diagnostic needs correction, and which one.**

Running one rank test per parameter axis and rejecting whenever any single $p$-value falls below $\alpha$ inflates the false-positive rate roughly $p$-fold. At $p = 27$ axes and $\alpha = 0.005$ per axis, a *perfectly calibrated* posterior trips something about 13% of the time. Chasing those phantom miscalibrations is a real and avoidable waste.

Holm-Bonferroni controls the family-wise error rate and is uniformly more powerful than plain Bonferroni. With $p$-values sorted ascending as $p_{(1)} \le \dots \le p_{(p)}$,

$$\tilde{p}_{(k)} = \min\Big(1, \max_{j \le k}\big\{(p - j + 1)\, p_{(j)}\big\}\Big), \qquad \text{for each fixed } k, \tag{11}$$

the inner maximum enforcing monotonicity. Reject axis $k$ iff $\tilde{p}_{k} < \alpha$.

Measured family-wise false-positive rate on an exact posterior: **0 / 12 replicates** at $\alpha = 0.05$, median adjusted minimum $p$-value 1.00 (test D3).

---

## 9. What the ensemble contributes, and what it does not

Deep ensembling — several independently initialised estimators combined as the *arithmetic mixture*

$$\bar{q}(\theta \mid z) = \frac{1}{n}\sum_{j=1}^{n} q_{\phi_{j}}(\theta \mid z) = \exp\Big(\operatorname{logsumexp}_{j} \log q_{\phi_{j}}(\theta \mid z) - \log n\Big) \tag{12}$$

— raises expected coverage relative to the average individual model, and the effect grows with ensemble size. The project knowledge base attributes this to the extra uncertainty the ensemble captures: individual estimators capture data uncertainty, while an ensemble also partially captures epistemic uncertainty, inflating credible regions. **But the ensemble can still be non-conservative** (Hermans et al.), so it is a mitigation, not a guarantee, and does not remove the need for any diagnostic above.

The arithmetic mixture in (12) is not interchangeable with the geometric mean $\frac{1}{n}\sum_{j} \log q_{\phi_{j}}$: the latter is a product of experts, which is *sharper* than any member and would make overconfidence worse. It also admits no closed-form sampler.

---

## 10. Summary of results

| # | Statement | Established in |
|---|---|---|
| S1 | SBC, expected coverage, and TARP are one rank-uniformity procedure under three choices of test quantity. | §5.1, eq. (2) |
| S2 | Rank jitter (3) is mandatory; without it the discreteness of ranks inflates KS rejections. | §5.2 |
| S3 | Marginal SBC names the failure shape: U-shaped is overconfident, dome is underconfident, sloped is biased. | §5.3 |
| S4 | Expected coverage below the diagonal is overconfidence — the dangerous direction. | §5.4 |
| S5 | Marginal SBC, expected coverage, and TARP with $x$-independent references are all blind to a data-ignoring posterior (0/12 each). | §6.3 |
| S6 | Blindness follows from exchangeability: parameter-only quantities cannot distinguish $\theta^{*}$ from a prior draw. | §6.1 |
| S7 | Data-dependent quantities detect it: bilinear 12/12, TARP with $x$-dependent references 12/12, both with 0 false positives. | §6.3 |
| S8 | The TARP theorem's power comes from reference points positioned *as a function of the observation*; random points forfeit it entirely (1/12 vs 12/12). | §5.5, §6.3 |
| S9 | Posterior contraction is the only layer that separates "calibrated" from "informative". | §7.1, eq. (9) |
| S10 | The effective rank of the information spectrum is a *linear* measure of a possibly curved image; a nonlinear estimator may legitimately exceed $E-1$. | §7.2 |
| S11 | Per-axis testing requires family-wise correction: at $p=27$, $\alpha=0.005$, a correct posterior trips ~13% of the time. | §8, eq. (11) |
| S12 | Ensembling raises coverage but does not guarantee conservativeness. | §9 |

---

## 11. Open points, caveats, and assumptions

1. **Local calibration is not implemented.** Every test here is *global*: it averages over observations. A failure specific to one real recording can hide inside an average that looks fine. L-C2ST and local coverage tests address this and are the natural next addition; they are also the most expensive, typically requiring an extra network and thousands of additional simulations.
2. **Posterior predictive checks are not implemented**, because they require the simulator to be callable. They are the only layer that tests the *model* rather than the *inference*, and their absence is a real gap in any claim that the model describes the data.
3. **The bilinear test quantity (8) is a substitute, not the recommended default.** Modrák et al. recommend the joint log-likelihood; (8) is used because the likelihood is unavailable in SBI. Its power against failure modes other than (7) has not been characterised, and a poorly chosen $W$ could in principle be insensitive to some deviations. Using several independent $W$ mitigates but does not eliminate this.
4. **The $x$-dependent TARP reference uses a linear readout of $z$.** A nonlinear relationship between $z$ and $\theta$ may be poorly captured, weakening the test. It was validated on a linear-Gaussian and a Gaussian-mixture problem; behaviour with a strongly nonlinear encoder is untested.
5. **All measurements come from analytic benchmarks**, chosen so the reference posterior is exact and any diagnostic failure is attributable to the diagnostic rather than to an approximate reference. Whether the detection rates transfer to a 27-parameter biophysical problem with a learned encoder is an assumption, not a result.
6. **The blindness results are stated for the exact case $q_\phi = p(\theta)$.** Real insufficiency is partial, and the tests will have *some* power against partial insufficiency. How much, as a function of how much information the summary retains, has not been characterised and would be a worthwhile experiment.
7. **MMD depends on the kernel and bandwidth.** The median heuristic is standard but not optimal, and a shift confined to a direction the kernel weights weakly could evade detection.
8. **Sample-size requirements are not established.** $N = 600$ was sufficient for every detection reported here; the minimum $N$ for a 27-dimensional problem is unknown, and the per-observation posterior sampling cost scales as $N \times n_{\text{members}}$, which is the practical bottleneck.

---

## 12. References

**Project knowledge base, read directly:**

- Hermans, Delaunoy, Rozet, Wehenkel, Begy, Louppe (2022). *A Trust Crisis in Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful.* Transactions on Machine Learning Research. — Expected coverage, the conservative/overconfident distinction, the empirical finding that all benchmarked algorithms can be non-conservative, the observation that a prior-equal posterior has nominal coverage with no information gain, ensembling as mitigation, post-training calibration.
- Lemos, Coogan, Hezaveh, Perreault-Levasseur. *Sampling-Based Accuracy Testing of Posterior Estimators for General Inference.* — Positionable credible region generators, Theorem 3, the explicit derivation that HPD ECP equals $1-\alpha$ when $\hat{p}(\theta \mid x) = p(\theta)$, and TARP.
- Schmitt, Bürkner, Köthe, Radev. *Detecting Model Misspecification in Amortized Bayesian Inference with Neural Networks.* — Structured summary space, MMD as a proxy for posterior error, sampling-based critical values.
- *Simulation-Based Inference: A Practical Guide.* — Embedding networks, the broad-marginals-versus-conditionals point.
- Transtrum et al. *Sloppiness and Emergent Theories in Physics, Biology and Beyond.* — Background for the spectrum interpretation in §7.2.

**PubMed, full text retrieved and read:**

- Modrák, Moon, Kim, Bürkner, Huurre, Faltejsková, Gelman, Vehtari (2025). *Simulation-Based Calibration Checking for Bayesian Computation: The Choice of Test Quantities Shapes Sensitivity.* Bayesian Analysis 20(2):461-488. https://doi.org/10.1214/23-ba1404 — Theorem 7 on incomplete use of data, the recommendation of data-dependent test quantities with the joint log-likelihood as default, and the worked case where a prior-equal posterior passes on all parameter-only quantities.

**Measured in this project, not from any source** (all on analytic benchmarks, reproducible via `smoke_test_diagnostics.py`):

- Every detection rate and $p$-value in §6.3 and the tables of §4.3, §5.2, §7.1, §7.2, §8.
- The argument in §6.1 that expected coverage degenerates under (7) was derived independently before the corroborating derivation was located in the TARP paper; both now agree.
- The finding in §5.5 and §6.3 that $x$-independent TARP references forfeit the theorem's guarantee, quantified as 1/12 versus 12/12.
- The correction in §7.2 that linear rank may exceed intrinsic dimension for a nonlinear estimator, with the measured 11-versus-12 demonstration.

**Not verified:** no claim in this document rests on a source that was not read in full. Where a statement is my own reasoning rather than a source's, §12 says so explicitly.
