# E3 -- The summary network: why a learned summary, what the metric loss asks of the geometry, and how much a collapsed code can carry

**Document E3 of the joint documentation set; the third chapter of Set E.**
What the encoder $h_\psi$ is for, what the DSN's label loss asks of the
embedding geometry, how much a code shaped by it can carry about $\theta$,
and what the participation-ratio rank $r_{\rm eff}$ can and cannot tell.
Index and status: `00_INDEX.md`; notation: `E0_READERS_GUIDE_NOTATION.md`;
the encoder's knobs: `P1_ENCODER_AXES.md`; the loss's knobs:
`P2_DSN_LOSS_AXES.md`. **Date:** 2026-10-05 (v1). **Applies to:** the
repository `Simulation-Based-Inference` at `834eb41` (`hpc/joint/`, D-037),
and the project documents and PDFs named in S6.

| date | change |
|---|---|
| 2026-10-05 | v1. Written from P1 S3.2 (the backbone, eqs. (P1.1)-(P1.10), cited and not repeated), P2 S3.2-S3.7 (the loss, eqs. (P2.1)-(P2.13)), E2 (eqs. (E2.4)-(E2.5)), the plan `JOINT_DSN_NPE_PLAN_v0_6.md` (repository, v0.6.5: S2.3, lines 251-311; the approximations, line 1702), the design handoff `HANDOFF_joint_dsn_npe_design.md` (R1-R4, lines 104-107; S12 item 1, lines 312-317), `hpc/dsn/Documentation/THEORY_joint_condition_search.md` S3.7.1 and S3.7.5, `hpc/dsn/metrics.py:1-131`, `stage3/joint_diagnostics.py:107-146`, `stage3b/encoder_probes.py:156-163`, `stage3/run_joint_arms.py:560, 595, 607-608`, `stage3/smoke_test_joint_arms.py:174-183`, `stage1/latent_sbi_simulator.py:128-237, 314-343`, `stage1/build_latent_bank.py:211-248` and the r2 encoder's configuration `hpc/dsn/hpc/Config/refit_mea_joint_full_r2_l0_t82.json`; the deck's section C (KB, v1.1); `SBI_PIPELINE.md` S4-S6; the project PDFs of BayesFlow, the Practical Guide, Goncalves et al. and the Frontier of SBI, at the pages cited; five PubMed Central full texts (S6); PubMed and bioRxiv searched (S6). Every `[RAN]` number is printed by the new torch-free `tools/e3_numbers.py` (two identical runs). New symbols $\hat c$, $\Sigma_z$ and $\hat\Sigma_z$, $h_{\rm b}$, $\rho_{\rm cap}$ (E0 v1.10, convention 14). Two findings: F-bb (r_eff is scale-free and is logged without a scale) and F-bc (at $C \ge 3$ the pre-training's uniform batches can change the separation target), both owned here. Dated notes added to E1 S3.8 (b), E2 S3.8 and P2 (the glossary's "Collapse", S3.2's closing reading, S3.6 and two rows of the S3.7 table). |

**Abstract.** The encoder reduces a window of $W = 18000$ samples to $E$
numbers on a unit sphere, and in arm `A0` it is fitted with the class label
alone -- $C = 2$ on the cohort -- before a flow over the 26 axes of $\theta$
is trained on what it outputs. The question this chapter answers is what the
label loss asks of the embedding, and how much of $\theta$ an embedding
shaped by it can carry. **Covered:** why a learned summary at all, and the
three signals a summary network can be trained on (S3.2, eq. (E3.1)); the
network as built, cited from P1 (S3.3); what the composite metric loss asks
of the geometry -- its zero sets under each `loss_type`, the simplex ETF,
and the separation term's weak pull on within-class spread (S3.4, eqs.
(E3.2)-(E3.3)); the information argument along $c \to \theta \to x \to z$ --
the asymmetry, the label ceiling of a code with at most $C$ values, the
split of what passes beyond it, and what the free axes can leak through a
code's errors, with the numbers on the bench (S3.5, eqs. (E3.4)-(E3.8));
where collapse happens in the stack and what transfers to the law the gain
is scored on, and why the size of a residual is not its information (S3.6,
eq. (E3.9)); $r_{\rm eff}$ -- what it measures on the sphere, what a printed
1.000 allows, unequal class masses, and finding F-bb (S3.7, eqs.
(E3.10)-(E3.11)); and the r2 encoder's measured numbers read with all of it
(S3.8). **Deliberately excluded:** the encoder's and the loss's knobs as
parameters (P1, P2, cited); the joint objective, the arms' training loop and
the trade between the two terms (E4); the bench generator (E6); the
statistics of the diagnostics and the decision rule (E7). No number here is
a result of a training run: nothing in `hpc/joint/` has run on the cluster.

---

## 1. Notation and symbols

A subset of E0's master table, in order of first use, plus the symbols E3
adds (marked *new*; E0 v1.10 declares them under convention 14).

| Symbol | Name / Meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| $x$ | one window (simulated or real) | $x \in \mathbb{R}^{W}$ | Hz per electrode on the cohort (E0) | S3.1 |
| $W$ | samples per window; 18000 on the cohort | $\mathbb{N}$ | samples | S3.1 |
| $h_\psi$ | the encoder, $h_\psi : \mathbb{R}^{W} \to S^{E-1}$ | map; weights $\psi$ | -- | S3.1 |
| $\psi$ | the encoder's weights | $\mathbb{R}^{n_\psi}$ | -- | S3.1 |
| $n_\psi$ | number of encoder weights | $\mathbb{N}$ | -- | S3.3 |
| $z$ | the embedding of a window, $z = h_\psi(x)$ | $z \in S^{E-1}$ | dimensionless | S3.1 |
| $S^{E-1}$ | the unit sphere of $\mathbb{R}^{E}$ | manifold | -- | S3.1 |
| $E$ | embedding dimension; 12 at the search anchors, 10 for the r2 encoder | $\mathbb{N}$ | -- | S3.1 |
| $c$ | class label | $c \in \{0, \dots, C - 1\}$ | -- | S3.1 |
| $C$ | number of classes; 2 on the cohort, 3 at the bench's bank-job default | $\mathbb{N}$ | -- | S3.1 |
| $\theta$ | the inference parameters of one row (E0); on the bench $\theta := \phi$ | $\theta \in \Theta \subset \mathbb{R}^{d_\theta}$ | mixed (E0) | S3.1 |
| $\Theta$ | the prior box | subset of $\mathbb{R}^{d_\theta}$ | -- | S3.2 |
| $d_\theta$ | parameter-space dimension; 26 on the DUP15HD bank, 10 on the bench | $\mathbb{N}$ | -- | S3.1 |
| $q_\omega$ | the conditional flow, $q_\omega(\theta \mid z)$, a density on $\Theta$ for each fixed $z$ | conditional density | as $p_\Theta$ | S3.1 |
| $p_\Theta$ | the prior density; on the bench the class mixture of plan eq. (7), with components $p_\Theta(\theta \mid c)$ | density on $\Theta$ | -- | S3.2 |
| $p_{\rm sim}$ | the simulator's joint law; on the bench it includes the class (S1.1); conditionals and marginals named by their arguments | law | -- | S3.2 |
| $p_{\rm real}$ | the law of real windows | law on $\mathbb{R}^{W}$ | -- | S3.6 |
| $p_{\rm ev}$ | the joint law of $(\theta, z)$ on the rows a score is computed on (E2) | law on $\Theta \times S^{E-1}$ | -- | S3.5 |
| $I$ | mutual information under the law named in the subscript | $\mathbb{R}_{\ge 0}$ | nats | S3.2 |
| $H$ | entropy under the law named in the subscript: Shannon entropy for a discrete argument, differential entropy for a continuous one (S1.1) | $\mathbb{R}$; $\ge 0$ when discrete | nats | S3.5 |
| $\ell_{\rm DSN}$ | the composite metric loss of one metric batch, eq. (P2.10) (computed level) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.4 |
| $\mathcal{L}^{\rm real}_{\rm DSN}$ | its expectation over the metric stream's law (analytic level); the stream holds simulated rows for `A0s`, `A2s` (S1.1) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.4 |
| $\mathcal{L}_{\rm joint}$ | the margin-plus-angular part of one batch's loss, eq. (P2.6) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.4 |
| $\mathcal{L}_{\rm sep}$ | the centroid-separation penalty of one batch, eq. (P2.8) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.4 |
| $\lambda_{\rm sep}$ | the separation term's weight (`lambda_sep`); constant in the joint space | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.4 |
| $\lambda_{\rm dsn}$ | the DSN term's weight in the joint objective | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.6 |
| $m_{\cos}, m_{\rm sq}$ | the triplet margin in cosine-distance and in squared-Euclidean units, $m_{\rm sq} = 2 m_{\cos}$ | $(0, 1)$; $(0, 2)$ | dimensionless | S3.4 |
| $\alpha$ | the angular hinge's half-angle (`angular_alpha_deg`) | $(0^\circ, 90^\circ)$ | degrees | S3.4 |
| $Q$ | squared Euclidean distance between two batch rows, $Q(i, i') = 2 d_{\cos}(z_i, z_{i'})$ on the sphere | $[0, 4]$ | dimensionless | S3.4 |
| $d_{\cos}$ | cosine distance, $1 - z^\top z'$ for unit rows | $[0, 2]$ | dimensionless | S3.4 |
| $\mathcal{T}_{\rm mined}, \mathcal{T}_{\rm strict}$ | the mined triplets of a batch, and those surviving the strict semi-hard filter | sets of index triples | -- | S3.4 |
| $B_{\rm met}$ | rows requested per step in the metric stream; 32 at the runner's default | $\mathbb{N}$ | rows | S3.4 |
| $n_c$ | rows of class $c$ in one metric batch | $\mathbb{N}_0$ | rows | S3.4 |
| $K$ | classes present in one metric batch with at least two rows (computed level) | $\mathbb{N}_0$ | -- | S3.4 |
| $\rho_{\rm ETF}$ | the separation term's target cosine, $-1/(K-1)$ | $[-1, 0)$ | dimensionless | S3.4 |
| $\bar z_c$ | mean embedding of class $c$ (analytic level); at collapse the class's single point | $\mathbb{R}^{E}$ | dimensionless | S3.4 |
| $\bar z^{(B)}_c, v_c$ | class-$c$ mean over one metric batch (computed level), and its unit direction | $\mathbb{R}^{E}$; $S^{E-1}$ | dimensionless | S3.4 |
| $\xi$ | within-class residual, $\xi = z - \bar z_c$ (the plan's and deck's `\eta`) | $\mathbb{R}^{E}$ | dimensionless | S3.5 |
| $\hat c$ | *new.* The class read off an embedding: $\hat c(z)$ is the label of the class region $z$ falls in (at exact collapse, of the class point it equals); an estimate of $c$ | map $S^{E-1} \to \{0, \dots, C-1\}$ | -- | S3.5 |
| $h_{\rm b}$ | *new.* The binary entropy function in nats: for a probability, minus the probability times its logarithm, minus its complement times the logarithm of the complement | map $[0, 1] \to [0, \ln 2]$ | nats | S3.5 |
| $\mathcal{A}_{\rm lab}, \mathcal{A}_{\rm free}$ | the bench's label-carrying and label-free axes; $\theta_{\mathcal{A}_{\rm free}}$ is the sub-vector on the free axes | disjoint index sets | -- | S3.5 |
| $\phi$ | the bench generator's latent vector; on the bench $\theta := \phi$ | $(0, 1)^{d_\theta}$ | dimensionless | S3.5 |
| $\tau_{\rm ov}$ | within-class spread of the bench's label axes; 0.10 at the bank job's default | $\mathbb{R}_{>0}$ | dimensionless | S3.5 |
| $\Delta, \hat\Delta$ | the population information gain (analytic level) and its computed estimate $L_0 - L$ (E2) | $\mathbb{R}$ | nats/row | S3.5 |
| $\Sigma_z, \hat\Sigma_z$ | *new.* The covariance of $z$ under the law a cloud is drawn from (analytic level), and the sample covariance (`ddof=1`) of the cloud's rows (computed level) | positive semi-definite $E \times E$ | dimensionless | S3.7 |
| $r_{\rm eff}$ | participation-ratio effective rank of an embedding cloud, eq. (E3.10); written $r_{\rm eff}(\Sigma_z)$ when applied to the analytic covariance | $[1, E]$ for a non-zero covariance (F-bb for zero) | -- | S3.1 |
| $\rho_{\rm cap}$ | *new.* The angle between a point of a spherical cap and the cap's centre, the cap's angular radius | $[0^\circ, 90^\circ]$ | degrees | S3.4 |
| $S_{\rm sil}$ | cosine silhouette of a cloud against its labels | $[-1, 1]$ | dimensionless | S3.7 |

### 1.1 Conventions

- E0 S1.1 applies in full: conditional quantities are written conditionally
  every time; every density carries a subscript naming its law; the analytic
  and computed levels carry different symbols (convention 4).
- **The bench's three-variable law.** On the bench the generator draws the
  class first, with equal probabilities, then $\theta$ from that class's
  component of the prior, then the window from $\theta$ alone:
  $p_{\rm sim}(c, \theta, x) = C^{-1}\, p_\Theta(\theta \mid c)\, p_{\rm sim}(x \mid \theta)$
  (`latent_sbi_simulator.py:165-196`; the provider is called with
  `condition=0`, `:341-343`; the nuisance and the seeds are drawn without
  the class, `build_latent_bank.py:211-248`) `[REPO]`. On the cohort no such
  law is given: $c$ is the donor's class, and a chain $c \to \theta \to x$
  there is a hypothesis about the world (S3.5).
- **Two entropies (R6).** $H$ of a discrete variable -- a class $c$, a code
  $\hat c$, a collapsed $z$ -- is Shannon's entropy, non-negative and at
  most $\ln$ of the number of values; $H$ of a continuous variable is
  differential (E2 S1.1) and can be negative. A mutual information is
  non-negative in both cases.
- **Levels in this chapter (R8).** Analytic: the laws, $I$, $H$,
  $\mathcal{L}^{\rm real}_{\rm DSN}$, $\Delta$, $\bar z_c$, $\Sigma_z$ and
  $r_{\rm eff}(\Sigma_z)$. Computed: $\ell_{\rm DSN}$ and its parts (one
  batch), $\bar z^{(B)}_c$, $v_c$, $K$, $\hat\Sigma_z$, $r_{\rm eff}$ as the
  code returns it, $\hat\Delta$, ARI and $S_{\rm sil}$. The metric stream's
  superscript "real" names the stream, not the domain: for `A0s` and `A2s`
  it carries simulated rows with the generator's labels (P2 S1).
- **Status (Provenance model).** In the stack, $r_{\rm eff}$, ARI and
  $S_{\rm sil}$ are `computed` once per run on the **simulated** report
  split (`run_joint_arms.py:560, 595, 607-608`) `[REPO]`; $\hat\Sigma_z$ is
  computed inside $r_{\rm eff}$ and its trace is never reported (F-bb);
  $\hat c$ is `derivation-only`, except as the k-means labels inside the
  ARI; $\mathcal{L}^{\rm real}_{\rm DSN}$ is `derivation-only` (its batch
  estimate trains, P2); the measured r2 values (S3.8) are `measured` by the
  pipeline's gate, outside this repository `[KB]`.
- **Caps.** A "cap of radius $\rho_{\rm cap}$" in `tools/e3_numbers.py` is a
  cloud of unit vectors around a centre whose angles from it are uniform on
  $[0, \rho_{\rm cap}]$ and whose tangent directions are uniform; a "rim
  cap" puts every point at angle exactly $\rho_{\rm cap}$. These are test
  clouds, not models of the r2 embedding.
- **Equations.** The plan's are "plan eq. (n)"; P2's "(P2.n)", P1's
  "(P1.n)", E2's "(E2.n)". E3's own are (E3.1) to (E3.11).
- Logarithms are natural; information is in nats. ASCII only, LF only.

---

## 2. Glossary

Ordered by first appearance; the section where each term becomes operative
is given last.

- **Summary statistic / embedding network** -- a map from a window to a
  short vector that the posterior estimator conditions on; an *embedding
  network* is one learned by a neural network. The Practical Guide's name
  for the network trained end-to-end with NPE. S3.2.
- **Sufficient statistic** -- a summary that keeps everything the window
  says about $\theta$. *Two senses:* classical (the conditional law of the
  window given the summary does not depend on $\theta$) and Bayesian (the
  posterior given the summary equals the posterior given the window, under
  one prior, eq. (E3.1)); this chapter uses the Bayesian one. S3.2.
- **Curse of dimensionality** -- the growth, in the worst case exponential
  in the data's dimension, of the simulations a non-parametric method needs.
  S3.2.
- **Metric learning** -- training an embedding so that distances between
  embedded examples reflect a relation between them, here "same class".
  S3.4.
- **Triplet, margin, mining** -- an anchor, a same-class positive and an
  other-class negative; the hinge that is zero once the negative is farther
  than the positive by the margin; the rule choosing which triplets of a
  batch are scored (P2). S3.4.
- **Zero set** -- the set of embedding geometries at which a non-negative
  loss is exactly zero; a loss minimised to zero lands somewhere in it, and
  which point depends on the optimiser, not on the loss. S3.4.
- **Simplex equiangular tight frame (ETF)** -- $C$ unit vectors with equal
  pairwise cosines $-1/(C-1)$: antipodal at $C = 2$, $120^\circ$ apart at
  $C = 3$. S3.4.
- **Neural collapse (NC1-NC4)** -- Papyan, Han and Donoho's four
  observations on classifiers trained by cross-entropy beyond zero training
  error: within-class variability of the last-layer activations vanishes,
  class means approach a simplex ETF, the classifiers align with the means,
  and the decision becomes nearest-class-mean. *Not the same as* class
  collapse in this stack, which a loss term imposes (S3.9). S3.4.
- **Class collapse** -- every window of a class mapped to one point; what
  the plan and the deck call "collapse". S3.4.
- **Terminal phase of training** -- Papyan et al.'s name for training past
  the epoch at which training error first vanishes. S3.4.
- **Minority collapse** -- Fang et al.'s name for the classifiers of
  minority classes becoming indistinguishable once class imbalance passes a
  threshold. S3.4.
- **Data-processing inequality** -- a function of a variable cannot carry
  more information about a third variable than the variable itself. S3.5.
- **Label code** -- an embedding that takes at most $C$ values, one per
  class region; $\hat c$ reads it. S3.5.
- **Label ceiling** -- the bound $\ln C$ on what a label code carries about
  $\theta$, eq. (E3.5). S3.5.
- **Fano's inequality** -- a bound on the entropy left in one discrete
  variable given another in terms of how often a guess of it is wrong. S3.5.
- **Information bottleneck** -- *two senses:* the method of Tishby et al.,
  an objective that compresses one variable while keeping information about
  another, over stochastic encoders; and the plan's metaphor for the margin
  and the half-angle. S3.6.
- **Participation ratio** -- the square of the sum of a covariance's
  eigenvalues over the sum of their squares; how many directions the cloud
  spreads along, in effect. $r_{\rm eff}$ is this number. S3.7.
- **Cardinality versus dimension** -- how many distinct values a variable
  takes, versus how many directions its values span. The first bounds mutual
  information; the second does not. S3.7.

---

## 3. Main body

### 3.1 The discrepancy: a summary fitted to two labels, asked for 26 axes

This section establishes the question the chapter answers and the running
example it answers it on.

A flow can only return what its conditioning variable carries: E2 showed
that the NPE loss is bounded below by the posterior's entropy given $z$, and
that an encoder in front of the flow can only lower the information
available, eq. (E2.4). In arm `A0` -- the status quo -- the encoder never
saw $\theta$. It was fitted to real windows with the class label, $C = 2$ on
the cohort, then frozen, and the flow over the 26 axes of DUP15HD was
trained on the simulated windows it embeds (plan S2.1) `[REPO]`. One would
expect a network like the stack's default -- about 0.36 M weights mapping a
three-minute window to twelve numbers (P1) -- to be able to carry far more
than one class bit. What the DSN's loss asks for is different: that windows
of one class lie close and windows of different classes far apart, with the
class means spread as far as $C$ unit vectors can be. If the loss gets
exactly that, each class lands on one point and $z$ can tell at most $C$
things apart -- at most $\ln 2 = 0.693$ nats about all 26 axes together, the
plan's label ceiling (plan S2.3, lines 269-277) `[REPO]`. And the r2
encoder's measured clouds look like that end state: participation-ratio rank
1.000 of $E = 10$ on the real windows and 1.017 on the simulated ones `[KB]`
(`SBI_PIPELINE.md` S4).

The chapter asks three things the plan's statement leaves open. When is the
loss's optimum exactly a $C$-point code, and on which windows? What exactly
is bounded, under which law, and what passes beyond the bound when the code
is not exact? And what can $r_{\rm eff} = 1.000$ say about it? The plan
itself lists "the label ceiling assumes exact collapse" among its
approximations (line 1702) `[REPO]`; the proofs it cites,
`INFO_LOSS_THEORY_v1.md` Propositions 5-7, are in neither the repository nor
the knowledge base, so every statement below is derived or cited here.

**Running example.** DUP15HD: $d_\theta = 26$, $W = 18000$, $E = 12$ at the
search anchors, $E = 10$ for the r2 encoder, $C = 2$; 1890 real windows and
29,616 simulated ones after the activity floor `[KB]` (pipeline S5-S6). The
bench, where a number needs $\theta$ or a known class: $d_\theta = 10$,
seven label axes and three free ones (`bench_burst_provider.py:82-96`),
$C = 3$ and $\tau_{\rm ov} = 0.10$ at the bank job's defaults `[REPO]`.

### 3.2 Why a learned summary

This section establishes what a summary must keep, why prescribed summaries
lose it, and the three signals a summary network can be trained on -- of
which only one is $\theta$'s posterior.

The question of S3.1 is about what $z$ keeps of $x$, so it starts with what
$z$ would have to keep. A window is 18,000 samples. Non-parametric
simulation-based methods cannot condition on that directly: the Frontier
review states that both rejection ABC and density-estimation-based inference
"suffer from the curse of dimensionality: In the worst case, the required
number of simulations increases exponentially with the dimension of the
data", so both "rely on low-dimensional summary statistics y(x) and the
quality of inference is tied to how well those summaries retain information
about the parameters" (p.3) `[KB-PDF p.3]`. A summary keeps everything when
it is sufficient. In the Bayesian sense used here, for the simulator's law
and its prior,

$$p_{\rm sim}(\theta \mid x) = p_{\rm sim}(\theta \mid h_\psi(x)) \ \text{ for } p_{\rm sim}\text{-almost every } x, \qquad \text{equivalently, when } I_{p_{\rm sim}}(\theta; x) < \infty, \qquad I_{p_{\rm sim}}(\theta; h_\psi(x)) = I_{p_{\rm sim}}(\theta; x), \tag{E3.1}$$

the second form being the equality case of the data-processing inequality
`[textbook, from memory]`. The Practical Guide asks the same of any summary
-- "that the posterior derived from the summary statistics approximates the
true posterior" -- and adds that for complex nonlinear simulators
"sufficient statistics are generally unknown" (p.8) `[KB-PDF p.8]`. The
Frontier review's verdict on prescribed summaries, that "the reduction of
the data to low-dimensional summary statistics invariably discards some of
the information in the data about" $\theta$ (p.3) `[KB-PDF p.3]`, carries
its condition with it (R7): it holds for any summary that is not sufficient,
which is the generic case for hand-chosen low-dimensional features, not a
theorem about every low-dimensional map.

A network can learn the summary instead, and what it learns depends on what
it is trained to do. Three signals are on record.

1. **Regression onto $\theta$.** According to PubMed, Akesson et al. train a
   convolutional network on time series from stochastic gene-regulatory
   models to output "the estimated posterior mean", with "the mean squared
   error (MSE)" as the loss, and then use "the predicted posterior mean ...
   as a summary statistic within the framework of ABC rejection sampling"
   `[PubMed full text]` ([DOI](https://doi.org/10.1109/TCBB.2021.3108695)).
   The summary has $d_\theta$ components and targets the posterior mean,
   which is all a squared loss can ask for; it is not the posterior.
2. **End-to-end with the density estimator.** BayesFlow trains its summary
   network jointly with the invertible network on the posterior's likelihood
   and wants it to learn "the most informative summary statistics directly
   from data" rather than lose information "through restrictive handcrafted
   summary statistics" (p.5) `[KB-PDF p.5]`; the Practical Guide calls such
   a network an embedding network and notes that it "can be trained
   end-to-end with the inference network (e.g., using the log-likelihood
   loss for NPE ...)" (p.8) `[KB-PDF p.8]`. By (E2.4) this is the one signal
   whose objective is a bound on $I_{p_{\rm sim}}(\theta; z)$ itself. Arm
   `A1` is this regime.
3. **A proxy task with a label.** The DSN is trained by metric learning on
   the class label (P2): its objective mentions neither $\theta$ nor the
   flow. Arm `A0` is this regime, two-stage in the Practical Guide's sense
   ("First learning summaries, then performing inference", p.8), with a
   first stage whose target is $c$.

The Practical Guide's remark that fixed, pre-computed embeddings "typically
underperform end-to-end learning" (p.31) `[KB-PDF p.31]` is made about
neural likelihood estimation, which cannot train an embedding end to end; it
is quoted here for the direction of the comparison, with that context, and
it carries no number. Goncalves et al. add the caution that "care is needed
when interpreting models fit to summary features, as choice of features can
influence the results" (p.17) `[KB-PDF p.17]`.

So the summary is learned because no sufficient one is known for this
simulator; whether the learned one approaches sufficiency depends on the
signal, and the next sections ask what the third signal does.

### 3.3 The network as built

This section establishes the object $h_\psi$ the rest of the chapter reasons
about, by pointer to P1, and the arms in which the label loss acts on it.

Before asking what the label loss does to $z$, the map that produces $z$
must be fixed. P1 writes it out as a closed-form function of its knobs, eqs.
(P1.1)-(P1.10): a one-dimensional residual network with GroupNorm, whose
stages at the runner's defaults have 16, 32, 64 and 128 channels, a total
stride of 64, $n_{\rm out} = 282$ output positions on a cohort window and a
receptive field of 1133 samples (11.3 s at 100 Hz); a head that pools
statistics over the positions and projects to $E$ numbers; and an L2
normalisation, $z = \tilde z / \lVert \tilde z \rVert_2 \in S^{E-1}$, eq.
(P1.9). It has $n_\psi = 359708$ weights at $E = 12$ `[RAN]` `[KB]` (P1
S3.2, S4). Two properties matter here. The pooling over positions makes the
summary a window-wide statistic: a feature counts the same wherever in the
three minutes it occurs, an invariance BayesFlow asks a summary network to
match to "the probabilistic symmetry of the observed data", naming a 1-D
convolutional network as a choice for time series (p.5) `[KB-PDF p.5]`;
whether time-shift invariance is the right symmetry for these windows is a
modelling choice the code makes, not a property it checks `[reasoning]`. And
the normalisation puts every embedding on the unit sphere, which the cosine
geometry of the losses uses (P2 eq. (P2.2)) and which S3.7 will need.

The label loss $\ell_{\rm DSN}$ acts on $\psi$ in five of the arms, through
the metric stream (P2 S1; plan S2.2) `[REPO]`: `A0` (real windows with the
cohort's labels, an encoder-only pre-training of `--encoder-steps` steps,
200 at the runner's default, then frozen); `A0s` (simulated windows with the
generator's labels, the same pre-training; bench only); `A2` and `A3` (real
windows, summed with the NPE term in every step; `A3` starts from `A0`'s
weights); `A2s` (both terms on simulated windows; bench only). In `A1` and
`A5` no label enters training.

### 3.4 What the metric loss asks of the geometry

This section establishes the zero sets of $\ell_{\rm DSN}$ under each
`loss_type`, the simplex ETF, and how weakly the separation term pulls on
within-class spread; it finds that "the global minimum is class collapse"
holds for the expected `joint_sep` loss and not for the other two types.

With $h_\psi$ fixed as a map into $S^{E-1}$, the label loss can be read as a
statement about the geometry of the cloud it produces.

**The plain mechanism.** A hinge on distances does not ask for positions; it
asks for an ordering. Regardless of how a class is spread, if every negative
an anchor is scored against lies far enough beyond the anchor's positive,
then the hinge is zero -- so classes that occupy separated regions of any
shape cost nothing. The separation term asks about one thing only, the
directions of the batch's class means, and asks them to sit at a simplex
ETF. Regardless of where a class's mean points on average, if the class is
spread, then two batches of its windows have mean directions that differ, so
the term cannot be zero on every batch; only a class that is a single point
gives the same direction every time. Hence the separation term, and only it,
makes exact class collapse the zero set.

**The simplex ETF.** For $C$ unit vectors $\bar z_0, \dots, \bar z_{C-1}$ in
$\mathbb{R}^{E}$ with $E \ge C - 1$,

$$\bar z_c^\top \bar z_{c'} = -\frac{1}{C-1} \ \ (c \ne c') \quad \implies \quad \sum_{c} \bar z_c = 0, \qquad \sum_{c} \bar z_c \bar z_c^\top \ \text{has } C - 1 \text{ eigenvalues equal to } \frac{C}{C-1} \text{ and the rest } 0, \tag{E3.2}$$

the first because the squared norm of the sum is $C$ plus $C(C-1)$ times the
common cosine, which is 0 (P2 S3.2) `[reasoning]`; the eigenvalues `[RAN]`
B1 for $C = 2, 3, 4$ (2, 1.5, 1.333; pairwise angles $180^\circ$,
$120^\circ$, $109.47^\circ$). $-1/(C-1)$ is the most negative common cosine
$C$ unit vectors can share. According to PubMed, Papyan, Han and Donoho
report that during the terminal phase of training of classification
networks, "The class means collapse to the vertices of a simplex equiangular
tight frame (ETF)" (NC2), together with "Cross-example within-class
variability of last-layer training activations collapses to zero" (NC1);
their class means are globally centred, the phase "begins at the epoch where
training error first vanishes", and the datasets they study are
class-balanced `[PubMed full text]`
([DOI](https://doi.org/10.1073/pnas.2015509117)). The transplant check (R3):
that geometry *emerges* there from cross-entropy training past zero error;
here it is *imposed*, by a penalty on the raw (uncentred) batch means, so
the paper's measurements say nothing about how fast or how far a
metric-learning run approaches it. The DSN's own reading makes the same
point about budget: NC is "a long-training phenomenon", reported at 300 to
350 epochs (THEORY S3.7.1) `[REPO]`; the joint stack's encoder-only
pre-training runs 200 steps (P2 S3.2). Fang et al., also according to
PubMed, prove the ETF for "class-balanced datasets" in their layer-peeled
model and identify "Minority Collapse" -- minority classes' classifiers
becoming indistinguishable once the imbalance ratio passes a finite
threshold -- for cross-entropy training `[PubMed full text]`
([DOI](https://doi.org/10.1073/pnas.2103091118)); the stack's joint batcher
draws equal rows per class (eq. (P2.1)), so the separation term sees
balanced batches there, and their imbalance result is not transplanted to
it.

**The zero sets.** Write the metric stream's law of $z$ given the class as
"class $c$'s law". With class-balanced batches holding at least two rows per
class -- the joint batcher, eq. (P2.1) -- and `loss_type = joint_sep` with
$\lambda_{\rm sep} > 0$,

$$\mathcal{L}^{\rm real}_{\rm DSN} = 0 \quad \iff \quad \Pr(z = \bar z_c \mid c) = 1 \ \text{ for every } c, \ \text{ with } \bar z_0, \dots, \bar z_{C-1} \text{ a simplex ETF.} \tag{E3.3}$$

*Why* `[reasoning]`: the expectation of a non-negative batch loss is 0 only
if the loss is 0 on almost every batch. $\mathcal{L}_{\rm sep}$ of eq.
(P2.8) is 0 only if every pair of batch-mean directions $v_c, v_{c'}$ has
cosine $\rho_{\rm ETF}$; with $K = C$ on every batch the directions then sum
to zero, so each $v_c$ is fixed by the others, and since each class's rows
are drawn independently of the others' this forces every $v_c$ to be the
same on almost every batch; a class law with two distinct points in its
support gives, with positive probability, batches whose mean directions
differ. Conversely, at an ETF of single points the margin hinge has nothing
to score: under `easy_pos_semihard_neg` no negative falls inside the strict
band, under `hard` no negative is as close as a positive (P2 eqs. (P2.3),
(P2.6a)), and $\mathcal{L}_{\rm sep} = 0$ `[RAN]` B4
($\mathcal{L}_{\rm joint}$ exactly 0 and $\mathcal{L}_{\rm sep}$ below
$10^{-31}$ on 300 of 300 batches, $C = 2$ and 3). Under
`loss_type = triplet` or `joint` the separation term is absent, and the zero
set is far larger: on caps of radius $\rho_{\rm cap}$ up to $30^\circ$
around the ETF vertices, at the joint defaults ($m_{\cos} = 0.2$,
$\alpha = 18^\circ$, `easy_pos_semihard_neg`, strict filter) and with r2's
constants (`hard`, filter off, $m_{\cos} = 0.3$), $\mathcal{L}_{\rm joint}$
and the plain triplet loss of eq. (P2.7) under the same miners were exactly
0 on every one of 300 batches at $C = 2$ and $C = 3$ `[RAN]` B4. The reason
is visible in (P2.3)-(P2.6a): the easy-positive miner pairs each anchor with
its *closest* positive, and the strict filter keeps a triplet only if the
negative lies within $m_{\rm sq}$ of that positive's distance, so separated
caps produce no kept triplet; under `hard` mining a triplet is mined only
when a negative is at least as close as a positive, which separated caps
never produce, and the margin does not enter that condition at all. A loss
with a large zero set does not say which point of it training reaches: that
is the optimiser's business, and the plan's "global minimum" (line 269)
names one point of the set under `triplet` and `joint`, and the whole set
only under `joint_sep`.

**How hard the separation term pulls.** Mirrored for the two class counts
(pattern 3). At $C = 2$ the target cosine $-1$ is the lowest a cosine can
take, so a small wobble of the two batch-mean directions raises their cosine
only at second order in the wobble and the squared penalty at fourth.
Conversely, at $C \ge 3$ the target $-1/(C-1)$ is interior, so the wobble
moves each cosine at first order and the penalty at second. The expected
batch value of $\mathcal{L}_{\rm sep}$ on caps of radius 2, 4 and $8^\circ$
grows with log-log slope 4.00 at $C = 2$ and 2.00 at $C = 3$ `[RAN]` B4
(4000 batches each). In absolute terms the pull is small:

| configuration | $C$ | rows per class | $\rho_{\rm cap} = 5^\circ$ | $15^\circ$ | $30^\circ$ |
|---|---|---|---|---|---|
| joint defaults, $\lambda_{\rm sep} = 0.1$: $\lambda_{\rm sep}$ times mean $\mathcal{L}_{\rm sep}$ | 2 | 16 | $2.9 \times 10^{-9}$ | $2.6 \times 10^{-7}$ | $3.8 \times 10^{-6}$ |
| the same | 3 | 10 | $3.8 \times 10^{-6}$ | $2.7 \times 10^{-5}$ | $1.4 \times 10^{-4}$ |
| r2's constants, $\lambda_{\rm sep} = 16.87$ | 2 | 9 | $1.6 \times 10^{-6}$ | $1.3 \times 10^{-4}$ | $2.4 \times 10^{-3}$ |
| the same | 3 | 9 | $6.5 \times 10^{-4}$ | $6.0 \times 10^{-3}$ | $2.3 \times 10^{-2}$ |

(`[RAN]` B4, a numpy transcription of eqs. (P2.3)-(P2.10), not the
repository's torch code; $E = 12$; 300 batches per cell; in every cell
$\mathcal{L}_{\rm joint} = 0$, so the composite is the separation term.) The
DSN's own documents call the separation term "collapse-seeking" and the
easy-positive miners "anti-collapse by construction" (THEORY S3.7.5)
`[REPO]`; the table puts a size on the first word: at the joint defaults and
$C = 2$, a class spread over a $15^\circ$ cap costs $2.6 \times 10^{-7}$ per
batch.

**A detail of the pre-training (finding F-bc).** The encoder-only
pre-training of `A0`/`A0s` draws $B_{\rm met}$ rows uniformly from the whole
source rather than per class (P2 S3.2, `run_joint_arms.py:158`) `[REPO]`. At
$C \ge 3$ a batch can then hold only two valid classes, and for that pair
the target becomes $\rho_{\rm ETF} = -1$ instead of $-1/(C-1)$: no
configuration zeroes the separation term on every batch, so at $C \ge 3$ the
expected pre-training loss has no exact zero. At $C = 3$, $B_{\rm met} = 32$
and equal class masses such a batch has probability $1.18 \times 10^{-4}$,
and at the exact ETF collapse it scores $\mathcal{L}_{\rm sep} = 0.25$, an
expected excess of $3 \times 10^{-5}$ `[RAN]` B4 -- negligible at the
defaults, larger at a small $B_{\rm met}$ or with a rare class. At $C = 2$
the same event leaves fewer than two classes and the term is simply 0 (P2
S3.7).

So the loss asks for exact class collapse onto a simplex ETF only through
its separation term, and asks for it gently at $C = 2$; under the other two
loss types it is satisfied by any geometry with separated classes. What a
geometry of either kind lets $z$ carry is the next question.

### 3.5 The information argument: $c \to \theta \to x \to z$

This section establishes the asymmetry between $\theta$-sufficiency and
$c$-sufficiency, the label ceiling of a code with at most $C$ values, the
split of what passes beyond it, and what the free axes can leak through a
code's errors, with the bench's numbers.

S3.4 described geometries; information is a property of laws, so the
argument now moves from where the embeddings lie to what they tell about
$\theta$.

**The asymmetry.** If $c \to \theta \to x$ is a Markov chain -- the window
depends on the class only through $\theta$ -- and $z = h_\psi(x)$ is
sufficient for $\theta$ in the sense of (E3.1), then for
$p_{\rm sim}$-almost every $x$

$$p_{\rm sim}(c \mid x) = \int_\Theta p_{\rm sim}(c \mid \theta)\, p_{\rm sim}(\theta \mid x)\, d\theta = \int_\Theta p_{\rm sim}(c \mid \theta)\, p_{\rm sim}(\theta \mid h_\psi(x))\, d\theta = p_{\rm sim}(c \mid h_\psi(x)), \tag{E3.4}$$

so $I_{p_{\rm sim}}(c; z) = I_{p_{\rm sim}}(c; x)$: a $\theta$-sufficient
summary loses nothing about the class (plan S2.3, line 256; deck C.1)
`[REPO]` `[KB]`. The first equality uses the chain ($c$ and $x$ independent
given $\theta$), the second (E3.1), the third the chain again for $z$, a
function of $x$ `[reasoning]`. The converse fails: $z = c$ is sufficient for
$c$, yet

$$I_{p_{\rm sim}}(\theta; c) = H_{p_{\rm sim}}[c] - H_{p_{\rm sim}}[c \mid \theta] \le \ln C. \tag{E3.5}$$

What bounds the right-hand side is the number of values $c$ takes, not
sufficiency: the window $x$ is sufficient for $c$ too, and so is the
posterior vector $p_{\rm sim}(c \mid x)$, and either can carry far more than
$\ln C$ about $\theta$ -- in a toy with a scalar parameter observed without
noise ($W = d_\theta = 1$, $x = \theta$) and
$p_{\rm sim}(c = 1 \mid \theta)$ strictly monotone in $\theta$ at $C = 2$,
the posterior vector is an injective function of $x$ and carries what $x$
carries, unbounded by $\ln 2$ `[reasoning]`. The deck's "a summary that is
sufficient for the diagnosis carries at most $\log C$ nats about the
parameters" (C.0, and the caption of its figure F04) `[KB]` therefore holds
for a summary that takes at most $C$ values -- $z = c$, or the exact code of
(E3.6) below -- and not for every summary sufficient for $c$ (R2).

On the bench the chain holds by construction (S1.1) `[REPO]`. On the cohort
it is a hypothesis -- "if the simulator mediates the pathology" (deck C.1,
O2) `[KB]` -- and the asymmetry's first half is conditional on it.

**The label ceiling.** Now let the embedding itself be a code. If, under the
law $p_{\rm ev}$ of the rows a score is computed on, $z$ takes at most $C$
distinct values almost surely, then $z$ is a discrete variable and

$$I_{p_{\rm ev}}(\theta; z) \;\le\; H_{p_{\rm ev}}[z] \;\le\; \ln C, \qquad \text{and in the first case of eq. (E2.5)} \qquad \Delta \;\le\; I_{p_{\rm ev}}(\theta; z) \;\le\; \ln C, \tag{E3.6}$$

whatever $E$, $d_\theta$, the encoder's capacity or the bank's size
`[textbook, from memory]` for the first inequality (a discrete variable's
Shannon entropy bounds its information about anything, and is at most $\ln$
of the number of its values); the second line is (E2.5) with its marginal
term 0, so the flow's expected Kullback-Leibler error only lowers $\Delta$
further. $\hat\Delta$ estimates $\Delta$, so the bound on the printed number
holds up to its Monte Carlo error, and only in E2's first case (the rows'
$\theta$-marginal is the floor's prior); E2 showed that against the box
floor on the activity-filtered bank the marginal term alone can reach 1.069
nats/row (E2 S3.3). Two hypotheses travel with (E3.6) (R2): the code must be
exact -- at most $C$ values, not $C$ tight clusters (S3.6) -- and it must be
exact *under $p_{\rm ev}$*, the law of the rows the gain is scored on, which
for every arm of the stack is a split of windows the simulator made: the
simulated report split, or on the bench the pseudo-real arm, which the
perturbed generator also simulates (E2 S3.3).

**What passes beyond it.** When the embedding is not a code, read it through
one: let $\hat c(z)$ be the class region $z$ falls in. Because $\hat c(z)$
is a function of $z$, the chain rule gives

$$I_{p_{\rm ev}}(\theta; z) \;=\; I_{p_{\rm ev}}(\theta; \hat c(z)) + I_{p_{\rm ev}}(\theta; z \mid \hat c(z)) \;\le\; \ln C + I_{p_{\rm ev}}(\theta; z \mid \hat c(z)), \tag{E3.7}$$

`[textbook, from memory]` for the chain rule. This is the exact form of the
plan's "everything beyond the label must travel through the within-class
residual" (lines 273-274): what passes beyond $\ln C$ passes through the
position of $z$ within the region the code reads, given that region. Note
whose region (R4): $\hat c(z)$ is the class read off the embedding, not the
window's true class $c$, and the residual that matters is measured from the
point of the class $\hat c(z)$; the plan's $\eta = z - \mu_c$ and E0's
$\xi = z - \bar z_c$ use $c$, and the two agree only where the code makes no
error.

**What an exact code carries, and what it leaks.** Two cases, the second
mirroring the first. If the code reproduces the true class, $\hat c = c$
almost surely, then by (E3.5) it carries
$I_{p_{\rm sim}}(\theta; c) = \ln C - H_{p_{\rm sim}}[c \mid \theta]$ with
equal class probabilities, and nothing about the free axes, since on the
bench $\theta_{\mathcal{A}_{\rm free}}$ is drawn independently of $c$
(`latent_sbi_simulator.py:187-188`) `[REPO]`. If instead the code errs, its
errors can depend on the free axes, because $\hat c$ is a function of $x$
and $x$ depends on them; then

$$I(\theta; \hat c) \;\le\; I(\theta; c) + H[\hat c \mid c], \qquad I(\theta_{\mathcal{A}_{\rm free}}; \hat c) \;\le\; H[\hat c \mid c] \;\le\; h_{\rm b}\big(\Pr(\hat c \ne c)\big) + \Pr(\hat c \ne c) \ln(C - 1), \tag{E3.8}$$

all under the bench's law $p_{\rm sim}$ (subscripts dropped inside this
display for readability and restored below) `[reasoning]` from
$I(\theta; \hat c) \le I(\theta; \hat c, c) = I(\theta; c) + I(\theta; \hat c \mid c)$,
$I(\theta_{\mathcal{A}_{\rm free}}; c) = 0$, and Fano's inequality for the
last step `[textbook, from memory]`. No code beats the Bayes error from
$\theta$ itself: a code computed from $x$ can be imitated from $\theta$ by
simulating $x$ first, so its error is at least the best error from $\theta$
`[reasoning]`. On the bench `[RAN]` B5 (two million prior draws per row;
bench centres from `simplex_centres`, DSN centres from the DSN generator's
regular simplex):

| $C$ | centres | $\tau_{\rm ov}$ | $H_{p_{\rm sim}}[c \mid \theta]$ | $I_{p_{\rm sim}}(\theta; c)$ | Bayes error from $\theta$ | bound (E3.8) at that error |
|---|---|---|---|---|---|---|
| 3 | bench | 0.10 | 0.0006 | 1.0980 | 0.0002 | 0.0019 |
| 3 | bench | 0.20 | 0.1450 | 0.9536 | 0.0551 | 0.2515 |
| 3 | bench | 0.30 | 0.5403 | 0.5583 | 0.2258 | 0.6907 |
| 3 | DSN | 0.10 | 0.0007 | 1.0979 | 0.0002 | 0.0023 |
| 3 | DSN | 0.20 | 0.1972 | 0.9014 | 0.0741 | 0.3156 |
| 3 | DSN | 0.30 | 0.6316 | 0.4670 | 0.2646 | 0.7613 |
| 2 | bench | 0.30 | 0.0510 | 0.6421 | 0.0190 | 0.0940 |
| 2 | DSN | 0.30 | 0.2851 | 0.4080 | 0.1214 | 0.3696 |

($\ln 3 = 1.0986$, $\ln 2 = 0.6931$; Monte Carlo standard errors of the
conditional entropy at most 0.0005; at $C = 2$ and $\tau_{\rm ov} \le 0.20$
both centre sets give at most 0.0494 nats of conditional entropy, listed in
B5.) Three readings follow. The plan's "the free axes carry exactly zero"
(line 272) holds for an error-free code. A code that errs leaks at most the
bound (E3.8) at its own error rate, which grows with the error: at the Bayes
error, the smallest any code can have, it is 0.0019 nats at the default
spread and 0.69 at $\tau_{\rm ov} = 0.30$ -- bounds, not estimates of what a
trained encoder leaks. Prediction P9 -- that the gain of `A0` tends to
$\log C$ from below as the ARI tends to 1 (line 306) -- has, more exactly,
its limit at $I_{p_{\rm sim}}(\theta; c)$: by (E3.8) and its mirror image
$I(\theta; c) \le I(\theta; \hat c) + H[c \mid \hat c]$, a code's
information lies within the Fano bound of that value on either side, and the
flow's error lowers the gain further. The value sits 0.0006 nats under
$\ln 3$ at the default and 0.54 under it at $\tau_{\rm ov} = 0.30$, where an
ARI near 1 is out of reach because the Bayes error is 0.2258. And on F-ba's
information side (E2 S5): the bench's irregular simplex places its three
centres 1.065, 0.805 and 0.697 apart (10.65, 8.05 and 6.97 times the default
$\tau_{\rm ov}$), the DSN's regular one 0.737 apart each (7.37 times)
`[RAN]` B5, and the irregular one carries *more* class information at
$\tau_{\rm ov} \ge 0.20$ (0.9536 against 0.9014; 0.5583 against 0.4670); at
the default the two agree to 0.0001 nats. What the irregularity does to the
DSN loss's arms is a separate question: the ETF target lives in $S^{E-1}$
and does not see the centres, which live in $\Theta$ (R5); unequal
separations make some class pairs harder to tell apart from $x$ than others
`[reasoning]`, which E6 owns.

So the ceiling is a statement about codes on a named law: exact, it caps the
information at $\ln C$; inexact, it caps the code's part and leaves the rest
to the position within each region. Which law, and which kind of code, the
stack produces is the next question.

### 3.6 Collapse in the stack: where it happens, and what transfers

This section establishes on which windows each arm's label loss acts, why a
collapse there does not by itself bound the information the flow uses, and
why the size of a residual is not its information.

(E3.6) needs a code under $p_{\rm ev}$, the law of the scored simulated
rows; (E3.3) produces one, at best, under the metric stream's law. The two
are the same law only in some arms.

**The two pre-trained arms, mirrored.** In `A0` the metric stream is the
real cohort: the loss constrains $h_\psi$ on real windows, and nothing in it
constrains $h_\psi$ on simulated windows, which are what the flow is trained
and scored on; a collapse of the real windows onto two points bounds nothing
about $I_{p_{\rm sim}}(\theta; z)$ unless the simulated windows land on
those two points too. Conversely, in `A0s` the metric stream is simulated
with the generator's labels: the loss constrains $h_\psi$ on the domain the
flow uses, and a collapse of the training windows bounds the scored split's
information as far as the encoder maps held-out simulated windows the way it
maps training ones `[reasoning]`. The design handoff draws the first half of
this line -- "the log C ceiling holds on the real arm (2 points) but does
not transfer to the sim arm by `r_eff = 1.017` alone" (R3) `[REPO]`. One
clause of that sentence needs its own condition (R2): on the real arm no
gain can be computed at all, since real windows carry no $\theta$, so "holds
on the real arm" is a statement about the real windows' code, true where
that code is exact. The same reading applies to the handoff's next action,
"If it fails, R1 does not apply to the real arm" (S12 item 1, line 316)
`[REPO]`: P8 is measured on the simulated bank, so a gain above $\ln 2$
there would say that the code does not transfer -- or, against the box
floor, that E2's marginal term is in it (E2 S5) -- not that the real arm
escaped collapse.

**The joint arms.** In `A2`, `A3` and `A2s` the metric term and the NPE term
act in the same step. The NPE term rewards information about $\theta$ on
simulated windows, eq. (E2.4); the metric term rewards a code on its stream.
Plan P3 predicts $r_{\rm eff} \to C - 1$ under any arm with large
$\lambda_{\rm dsn}$ (line 300) `[REPO]`; how the two terms trade is E4's
subject.

**Size is not information.** Mutual information does not change when an
injective map is applied to either variable: for any injective measurable
map taking $z$ to $z'$,

$$I(\theta; z') = I(\theta; z), \qquad \text{while for a scalar Gaussian signal in additive Gaussian noise} \qquad I = \frac{1}{2} \ln(1 + \text{SNR}), \tag{E3.9}$$

`[textbook, from memory]`, with SNR the ratio of the signal's variance to
the noise's. According to PubMed, Wieczorek and Roth note, citing earlier
work, that the information bottleneck's solution "should be invariant to
monotonic transformations" of its variables, "since the problem is defined
only in terms of mutual information which exhibits such invariance"
`[PubMed full text]` ([DOI](https://doi.org/10.3390/e22020131)). Shrinking
every class's residual by the same factor is injective as long as the
classes stay apart, so a cloud of tight clusters carries exactly what the
same cloud with wide clusters carries. What turns small into uninformative
is a noise or a resolution scale: with additive noise, an amplitude equal to
the noise's carries 0.347 nats, one tenth of it 0.00498, one hundredth
$5 \times 10^{-5}$ `[RAN]` B6. The encoder is deterministic and adds no
noise; the scales that matter in practice are the flow's ability to resolve
fine structure from a finite bank and the conditioner's smoothness, which
are properties of training, not of information `[reasoning]`. Two
consequences. The plan's "the margin $m_{\cos}$ and angular half-angle
$\alpha$ penalise [the residual] -- making them the width of an information
bottleneck" (lines 274-276) holds with a noise or resolution scale attached
(R7): without one, $m_{\cos}$ and $\alpha$ are geometric constraints on the
residual's size, and by S3.4 under the strict filter and the easy-positive
miners they constrain only how close the nearest negative may come. And a
near-collapse is not a ceiling: the bound (E3.6) needs exactly $C$ values,
and a cloud of $C$ clusters, however tight, carries by (E3.7) whatever its
within-cluster positions carry. The standalone pipeline makes the point
concrete: it standardises the frozen embedding per component before its flow
(`z_score_x = "independent"`, P4 S3.5) -- an affine, invertible map that
leaves the information unchanged and stretches every coordinate to unit
variance `[reasoning]`. "Information bottleneck" in the plan's sentence is
also not the method of that name (R6): Wieczorek and Roth state the method
as a search for a third random vector that compresses one variable while
preserving the information it contains about a second, its solution an
optimal conditional distribution -- a stochastic encoder -- under a
trade-off parameter `[PubMed full text]`.

So whether the stack's flows face a ceiling depends on whether the simulated
windows they see are mapped to an exact code, which the label loss does not
ask for in `A0`, and which a tight-but-continuous cloud does not provide.
The stack's one number about the cloud's shape is $r_{\rm eff}$, which the
next section takes apart.

### 3.7 $r_{\rm eff}$: what the participation ratio measures

This section establishes what $r_{\rm eff}$ computes, that it measures the
dimension of the cloud's linear span and not the number of its points, what
it reads at an exact collapse with equal and unequal class masses, how close
to collapse a printed 1.000 is on the sphere, and finding F-bb.

S3.6 left one question for the cloud itself: from the embeddings alone, can
one tell an exact code from tight clusters, or from a thin continuum?

**Definition.** All three implementations in the repository compute

$$r_{\rm eff} = \frac{(\mathrm{tr}\, \hat\Sigma_z)^2}{\mathrm{tr}(\hat\Sigma_z^2)}, \qquad \text{the computed level of} \qquad r_{\rm eff}(\Sigma_z) = \frac{(\mathrm{tr}\, \Sigma_z)^2}{\mathrm{tr}(\Sigma_z^2)}, \tag{E3.10}$$

the square of the sum of the covariance's eigenvalues over the sum of their
squares (`metrics.py:115-131`, `joint_diagnostics.py:107-121`,
`encoder_probes.py:156-163`) `[REPO]`. It lies in $[1, E]$ for a non-zero
covariance, equals $k$ for $k$ equal non-zero eigenvalues, and does not
change when the cloud is scaled `[reasoning]`. According to PubMed,
Recanatesi et al. describe the participation ratio as counting "the
effective dimensions along which data are spread as a ratio of the square of
the first moment and the second moment of the eigenvalue probability density
function", and their noisy spiral -- a one-dimensional curve -- reads 2 at
large scales, "reflecting the overall embedding of the spiral"
`[PubMed full text]` ([DOI](https://doi.org/10.1016/j.patter.2022.100555)):
the global participation ratio measures the dimension of the span a cloud
occupies, not its intrinsic dimension and not its number of points
`[reasoning]`.

**At an exact collapse.** At the ETF with equal class probabilities the mean
is 0 by (E3.2), so $\Sigma_z$ is $1/C$ times the sum in (E3.2): $C - 1$
equal non-zero eigenvalues, and $r_{\rm eff}(\Sigma_z) = C - 1$
`[reasoning]` (plan S2.3; design handoff R2) `[REPO]`; `[RAN]` B2 gives 1,
2, 3 at $C = 2, 3, 4$ in all three implementations. Unequal masses lower it
at $C \ge 3$: 1.96 for the counts 72, 108, 108 -- the Giulia cohort's
windows per class under D-056's tiling `[KB]` -- and 1.71 for 1, 1, 8; at
$C = 2$ two points always give a rank-one covariance, so $r_{\rm eff} = 1$
for any masses (10 against 90: 1.000) `[RAN]` B2. This is the deck's
"recompute with empirical masses" (C.3) made concrete.

**On the sphere, rank one means at most two points.** A cloud whose
covariance has rank one lies on an affine line, and a line meets a sphere in
at most two points `[reasoning]`. So an *exact* $r_{\rm eff} = 1$ on unit
embeddings is a code with at most two values: a two-point collapse, or a
single point. Rank two allows a circle, a continuum, so at $C = 3$ an exact
$r_{\rm eff} = 2$ does not imply three points. Near 1 the reading is
quantitative: for a small spread off the main axis, $r_{\rm eff} - 1$ is
about twice the variance off the axis over the variance along it
`[reasoning]`, and for two antipodal rim caps of equal mass

$$r_{\rm eff}(\Sigma_z) = \frac{1}{\cos^4 \rho_{\rm cap} + \sin^4 \rho_{\rm cap} / (E - 1)} \tag{E3.11}$$

`[reasoning]`, `[RAN]` B2 (Monte Carlo agrees to five decimals). A value
that prints as 1.000 -- below 1.0005 -- therefore allows, at $E = 10$, two
antipodal rim caps of radius under $0.906^\circ$, or one uniform arc under
$7.02^\circ$ long (half-angle $3.508^\circ$) `[RAN]` B2. The two differ in
scale, which $r_{\rm eff}$ discards: the arc's total variance
$\mathrm{tr}\,\Sigma_z$ is 0.00125, against a total variance of 1 for two
antipodal points of equal mass. The run record logs the k-means ARI and the
cosine silhouette $S_{\rm sil}$ beside $r_{\rm eff}$ (P2 S3.7), and they
separate the cases $r_{\rm eff}$ merges `[RAN]` B3:

| cloud ($E = 10$, 400 rows, two labels) | $r_{\rm eff}$ | $\mathrm{tr}\,\hat\Sigma_z$ | ARI | $S_{\rm sil}$ |
|---|---|---|---|---|
| one point, labels split in half | 1.0000 | 0.00000 | 0.000 | 0.000 |
| two antipodal points (an exact $C = 2$ code) | 1.0000 | 1.00251 | 1.000 | 1.000 |
| a $7^\circ$ arc, labels its two halves (a continuum) | 1.0004 | 0.00140 | 0.990 | 0.836 |
| a $180^\circ$ arc, labels its two halves | 1.3559 | 0.60793 | 0.980 | 0.769 |

**Finding F-bb.** $r_{\rm eff}$ is scale-free and the stack logs it without
a scale. The joint stack's `effective_rank` and its Stage 3b duplicate
return 1.0 when the covariance is exactly zero; the DSN's returns 0.0, and
its docstring calls that case "total collapse" `[REPO]`. In floating point a
constant cloud rarely has an exactly zero covariance: fifty copies of a
generic unit vector leave entries of order $10^{-31}$, and all three
implementations return 1.0; fifty copies of a coordinate axis give exactly
zero, and the DSN's returns 0.0 while the other two return 1.0 `[RAN]` B2.
The joint stack's own smoke test checks "effective_rank is 1 for a rank-one
cloud" on a Gaussian line in $\mathbb{R}^8$ -- 200 distinct rows, a
continuum -- and labels the value "the collapse value C-1 at C=2"
(`smoke_test_joint_arms.py:174-178`); the function's docstring says the same
of any rank-one cloud, "the collapse value C - 1 at C = 2, which is what the
r2 encoder measures" (`joint_diagnostics.py:111-112`) `[REPO]`. At $C = 2$,
then, the record's $r_{\rm eff} = 1$ is the same number for a constant
encoder (information 0), a two-point code (at most $\ln 2$) and a thin
continuum (no ceiling), and plan P3's test is met by all three; the ARI and
the silhouette separate them, the trace would too, and the trace is not
logged. Report; open (log $\mathrm{tr}\,\hat\Sigma_z$ beside $r_{\rm eff}$,
and read P3 with the cluster scores). The search's ledger keeps
$r_{\rm eff}$ among its recorded fields (`npe_tune_joint.py:518`) `[REPO]`.

So $r_{\rm eff}$ answers "how many directions" and not "how many points"; on
the sphere an exact 1 does bound the points to two, but a printed 1.000
cannot tell tight clusters from a short arc without a second number. The r2
encoder's measurements can now be read.

### 3.8 Back to the r2 numbers

This section establishes what the measured $r_{\rm eff}$ values of the r2
encoder imply and do not imply, and what would discriminate between the
readings.

The chapter opened on two measured numbers that look like the label
ceiling's end state; with S3.4-S3.7 they can be read term by term.

**What r2 was trained to do.** The configuration file names
`loss_type = joint_sep`, `hard` mining with the strict filter off,
$m_{\cos} = 0.3$, $\alpha = 10.69^\circ$, $\lambda_{\rm sep} = 16.87$ with a
warm-up over 0.288 of training, $E = 10$, and nine cultures per class per
batch with one window each
(`refit_mea_joint_full_r2_l0_t82.json:57-58, 97, 113-131`) `[REPO]`. It was
trained by the standalone DSN, outside D-037; whether the code that trained
it equals the freeze's is not established here. Under (E3.3) the expected
loss of that objective is zero only at an exact two-point code on its
training windows; under S3.4's table the separation term's pull on a
$15^\circ$ spread is $1.3 \times 10^{-4}$ per batch at these constants --
small, but about 500 times the joint defaults'.

**The real arm, 1.000.** On unit embeddings this is consistent with two
antipodal clusters of equal mass -- the most lenient two-cluster case, since
it maximises the variance along the main axis -- whose points lie within
about $0.9^\circ$ of their centres in root-mean-square angle (rim caps of
radius under $0.906^\circ$; by the first-order relation of S3.7 the
root-mean-square angle is what counts, whatever the clusters' shape)
`[reasoning]`, or with one arc under about $7^\circ$ (S3.7); the pipeline's
own reading, "the real arm lies essentially on a line" (`SBI_PIPELINE.md`
S4) `[KB]`, is the dimension statement, and the deck's "is that value" (C.3)
holds in the sense of being consistent with, not of identifying, an exact
two-point code. Neither the trace nor the cluster scores of the real cloud
are recorded in the knowledge base; the r2 `results.json` values are
"[TO VERIFY]" (`HPC_PATHS.md`, 2026-08-25b) `[KB]`.

**The simulated arm, 1.017.** By (E3.11) this is two antipodal rim caps of
$5.26^\circ$, or one arc of $40.8^\circ$ (half-angle $20.39^\circ$); the
off-axis variance is about 0.85% of the main axis's `[RAN]` B2. This is the
cloud that bears on reading (b) of E1 S3.8, because the flow is trained and
scored on simulated $z$; the deck lists three unseparated explanations for
it -- transfer of the collapse, off-support degeneracy, simulator poverty
(C.3; O2) `[KB]` -- and none of them is a statement about cardinality.

**What would discriminate.** For the geometry: $\mathrm{tr}\,\hat\Sigma_z$,
the silhouette against the labels and the number of distinct clusters on
each arm (S3.7). For the information: the simulated-arm gain against the
shuffled control on the same rows (E2 S3.3; E7), and Stage 3b's split of
`A0` against `A0s` into a domain part and an objective part (plan Stage 3b;
deck G) `[KB]`.

So the thread ends where it began, sharpened: the label loss's optimum is a
$C$-point code only under `joint_sep`, and only on its own stream; what
bounds the information is the cardinality of $z$ on the scored simulated
law, which no measured number yet establishes; and the measured 1.000 is a
statement about dimension. How the joint objective trades the label term
against the likelihood term is E4's subject.

### 3.9 Common confusions

Words and statements that carry two senses in this chapter (R6), each with
the sense the set uses.

- **"Collapse."** Four objects: *neural collapse* (Papyan et al.'s NC1-NC4,
  observed in cross-entropy classifiers past zero training error, on
  balanced data); *class collapse* (every window of a class at one point;
  the plan's and the deck's sense, imposed here by $\mathcal{L}_{\rm sep}$);
  *dimensional collapse* (a cloud of low rank; the DSN `metrics.py`
  docstring's "collapse to a line / point", a VICReg-style health check);
  *total collapse* (a constant encoder). $r_{\rm eff}$ measures the third
  and conflates the second with the fourth at $C = 2$ (F-bb).
- **"The global minimum."** Of one batch's loss or of its expectation; a
  zero *set* or a unique minimiser. Under `joint_sep` with
  $\lambda_{\rm sep} > 0$ the expected loss's zero set is exactly the
  collapsed ETF, (E3.3); under `triplet` and `joint` it contains every
  geometry with separated classes (S3.4).
- **"Sufficient."** Classical versus Bayesian (S2); and sufficient *for
  $\theta$* versus *for $c$* -- the first implies the second along the
  chain, (E3.4), and not conversely, (E3.5). Sufficiency for $c$ bounds
  nothing about $\theta$ (the window is sufficient for $c$); taking at most
  $C$ values bounds it by $\ln C$, (E3.6).
- **"The label."** The window's true class $c$, versus the class $\hat c(z)$
  read off the embedding; the ceiling and the residual of (E3.7) are about
  $\hat c$.
- **"The ceiling."** $\ln C$ (the bound on any code, (E3.6)), versus
  $I_{p_{\rm sim}}(\theta; c) = \ln C - H_{p_{\rm sim}}[c \mid \theta]$
  (what an error-free code carries, (E3.5)), versus a bound on $\hat\Delta$
  (only in E2's first case and up to Monte Carlo error).
- **"Information" versus "size."** Mutual information is invariant under
  injective maps; a residual's size is not information until a noise or
  resolution scale is named, (E3.9).
- **"Information bottleneck."** The method (an objective over stochastic
  encoders trading compression against preserved information) versus the
  plan's metaphor for $m_{\cos}$ and $\alpha$ (S3.6).
- **"Dimension" versus "cardinality."** $r_{\rm eff}$ counts directions;
  information about $\theta$ through a code is bounded by its number of
  values. On the sphere an exact $r_{\rm eff} = 1$ means at most two values,
  and nothing of the kind follows for 1.000 or for $r_{\rm eff} = 2$ (S3.7).
- **"$r_{\rm eff} = C - 1$."** Holds at an exact ETF collapse with equal
  class masses; at $C = 2$ for any masses; and at $C = 2$ also for a
  constant cloud in the joint stack's implementation and for any thin line
  (S3.7).
- **"ETF of the class means."** Papyan et al.'s NC2 is about globally
  centred means; the stack's $\mathcal{L}_{\rm sep}$ targets the raw batch
  means' directions; for unit vectors at the common cosine $-1/(C-1)$ the
  two coincide, because such vectors sum to zero, (E3.2).
- **"Free axes carry zero."** Through the true class, yes, on the bench by
  construction; through a code, only up to its error entropy, (E3.8).

### 3.10 Check yourself

1. Under `loss_type = joint` (no separation term) a colleague says the DSN's
   global minimum is class collapse. Give a geometry that refutes it and say
   why the loss is zero there.
2. A bench `A0s` run at $C = 3$ and $\tau_{\rm ov} = 0.30$ maps every
   simulated window, training and held-out, to exactly one of three points.
   What is the most its bench gain can be? Can the free axes contribute, and
   how much at most?
3. A $C = 3$ arm reports $r_{\rm eff} = 2.000$. Does that confirm a
   three-point code? What would $r_{\rm eff}$ be for an exact three-point
   code with 72, 108 and 108 windows per class?
4. The r2 encoder gives $r_{\rm eff} = 1.000$ on real windows and 1.017 on
   simulated ones. Which number bears on what an `A0` flow can learn about
   $\theta$, and does either bound it?
5. Why does shrinking every class's residual by a factor of a hundred leave
   $I_{p_{\rm ev}}(\theta; z)$ unchanged, and what would have to be present
   for it to lower it?
6. Why is the separation term's pull on within-class spread so much weaker
   at $C = 2$ than at $C = 3$?

<details>
<summary>Answers (folded)</summary>

1. Two antipodal caps of radius $30^\circ$ at $C = 2$. With the
   easy-positive miners each anchor is paired with its closest positive, and
   the strict filter keeps a triplet only if the negative lies within
   $m_{\rm sq}$ of that positive's distance -- far from true for separated
   caps -- so no triplet is kept and $\mathcal{L}_{\rm joint} = 0$; under
   `hard` mining no negative is as close as a positive, so nothing is mined
   (S3.4, `[RAN]` B4: 300 of 300 batches at zero). Only the separation term,
   absent under `joint`, makes spread cost anything.
2. By (E3.6), $\Delta \le I_{p_{\rm ev}}(\theta; z) \le \ln 3 = 1.0986$ nats
   (first case of (E2.5)). The code cannot reproduce the class: the Bayes
   error from $\theta$ is 0.2258 there `[RAN]` B5, so it errs at least that
   often, and by (E3.8) what the free axes can carry through it is bounded
   by $h_{\rm b}(\Pr(\hat c \ne c)) + \Pr(\hat c \ne c)\ln 2$, a bound of at
   least 0.69 nats at such an error -- inside the overall $\ln 3$. Only an
   error-free code, out of reach here, would carry exactly
   $I_{p_{\rm sim}}(\theta; c) = 0.5583$ nats and nothing on the free axes.
3. No: an exact rank-two covariance on the sphere allows a circle, a
   continuum (S3.7). Unequal masses lower the exact code's value: 1.96 for
   72, 108, 108 `[RAN]` B2.
4. The simulated arm's 1.017, because the flow is trained and scored on
   simulated $z$ (S3.6, S3.8); and neither bounds it, because $r_{\rm eff}$
   measures dimension, and only an exact code -- a cardinality -- caps the
   information (E3.6). The real arm's 1.000 says the real windows are close
   to a two-point code or a short arc (S3.7), not what the simulated windows
   carry.
5. The shrinkage is injective while the classes stay apart, and mutual
   information is invariant under injective maps (E3.9). An additive noise
   of fixed scale, or a resolution limit of the flow, would make the
   shrunken residual less informative: at a hundredth of a fixed noise's
   amplitude a Gaussian residual carries $5 \times 10^{-5}$ nats `[RAN]` B6.
6. At $C = 2$ the target cosine $-1$ is the minimum, so a wobble of the
   batch means changes the cosine at second order and the squared penalty at
   fourth; at $C = 3$ the target $-0.5$ is interior and the penalty is
   second order. Measured slopes 4.00 and 2.00 `[RAN]` B4.

</details>

---

## 4. Summary of results

- A summary must be sufficient to lose nothing, (E3.1); none is known for
  this simulator. Of the three training signals on record -- regression onto
  $\theta$, end-to-end NPE, a label proxy -- only end-to-end NPE optimises a
  bound on $I_{p_{\rm sim}}(\theta; z)$ (S3.2).
- The network is P1's: a GroupNorm 1-D residual network, pooled over 282
  positions, L2-normalised onto $S^{E-1}$, 359708 weights at $E = 12$; the
  label loss acts on it in `A0`, `A0s`, `A2`, `A3`, `A2s` (S3.3).
- The simplex ETF, (E3.2), sums to zero and has $C - 1$ equal frame
  eigenvalues; here it is imposed by $\mathcal{L}_{\rm sep}$, not emergent
  as in Papyan et al. (S3.4, `[RAN]` B1).
- The expected `joint_sep` loss is zero exactly at a collapsed ETF, (E3.3);
  `triplet` and `joint` are zero on separated caps of any radius tested up
  to $30^\circ$ (S3.4, `[RAN]` B4).
- The separation term's pull on spread is fourth order at $C = 2$ and second
  at $C = 3$ -- $2.6 \times 10^{-7}$ per batch for a $15^\circ$ cap at the
  joint defaults and $C = 2$ (S3.4, `[RAN]` B4). F-bc: at $C \ge 3$ the
  pre-training's uniform batches leave the expected loss without an exact
  zero (immaterial at the defaults).
- Along $c \to \theta \to x \to z$, $\theta$-sufficiency implies
  $c$-sufficiency, (E3.4), and not conversely, (E3.5); what caps a summary's
  information about $\theta$ at $\ln C$ is its having at most $C$ values,
  not its sufficiency for $c$ (S3.5).
- A code with at most $C$ values under the scored law carries at most
  $\ln C$, and bounds $\Delta$ by it in E2's first case, (E3.6); what passes
  beyond $\ln C$ passes through the position within the code's regions,
  (E3.7) (S3.5).
- An error-free code carries $\ln C - H_{p_{\rm sim}}[c \mid \theta]$; an
  erring one can leak free-axis information up to its error entropy, (E3.8);
  on the bench 1.0980 nats and a 0.0019-nat leak bound at the default,
  0.5583 and 0.69 at $\tau_{\rm ov} = 0.30$; the bench's irregular centres
  carry slightly more class information than the DSN's regular ones (S3.5,
  `[RAN]` B5).
- Collapse on the real stream (`A0`) does not bound the simulated law's
  information; size is not information, (E3.9), so near-collapse is no
  ceiling and the margin is a bottleneck only with a noise or resolution
  scale (S3.6, `[RAN]` B6).
- $r_{\rm eff}$, (E3.10), measures the dimension of the cloud's span:
  $C - 1$ at an equal-mass ETF collapse, 1.96 at 72:108:108; on the sphere
  an exact 1 means at most two points, while 1.000 allows rim caps under
  $0.906^\circ$ or an arc under $7.02^\circ$, (E3.11); F-bb: scale-free and
  logged without a scale (S3.7, `[RAN]` B2-B3).
- The r2 numbers are consistent with near-collapse on real windows and say
  nothing yet about the simulated law's cardinality (S3.8).

## 5. Open points, caveats, assumptions

- **Whether training reaches the zero set is not addressed.** (E3.3)
  characterises where the expected loss is zero; the stack's pre-training
  runs 200 steps, and nothing in `hpc/joint/` has run (S3.4).
- **The zero-set results are for the transcription.** `tools/e3_numbers.py`
  B4 implements P2's eqs. (P2.3)-(P2.10) in numpy; the torch code
  (`dsn_joint_loss.py`, `pytorch_metric_learning` 1.6.3) was read for P2 and
  not executed here. Ties in the miners' argmin are broken by index in the
  transcription and by the library's own rule in the code; on continuous
  clouds ties have probability 0.
- **The caps are test clouds.** Uniform angles and directions, or rims; real
  class clouds need not be caps, and the thresholds of S3.7 (0.906 and
  $3.508^\circ$) are for the two named shapes at $E = 10$.
- **(E3.8) is a bound, not an estimate.** What a trained `A0s` encoder leaks
  through its errors is not computed; the Bayes error from $\theta$ bounds
  every code's error from below only along the chain $c \to \theta \to x$.
- **The cohort's chain is a hypothesis.** (E3.4)'s first half needs
  $c \to \theta \to x$; on real data that is O2 (deck C.1, C.7).
- **F-bb (found in this chapter's reading; owner E3, E7).**
  `joint_diagnostics.effective_rank` (`:107-121`) and
  `encoder_probes.effective_rank` (`:156-163`) return 1.0 for an exactly
  zero covariance, `metrics.effective_rank` (`:115-131`) returns 0.0; in
  floating point a constant cloud generally returns 1.0 in all three; the
  run record carries $r_{\rm eff}$, ARI and $S_{\rm sil}$ but no trace; P3
  is met at $C = 2$ by a constant encoder, a two-point code and a line
  alike; the smoke test R3d calls a Gaussian line, and the function's
  docstring any rank-one cloud, "the collapse value" `[RAN]` B2-B3,
  `[REPO]`. Report; open (log the trace; read P3 with the cluster scores).
- **F-bc (found here; owner E3; extends P2 S3.7's "pre-training without
  balance").** At $C \ge 3$ the encoder-only pre-training's uniform batches
  can hold two valid classes, and the separation target for that pair is
  $-1$; the expected loss has no exact zero; probability
  $1.18 \times 10^{-4}$ per batch and expected excess $3 \times 10^{-5}$ at
  the exact collapse at $C = 3$, $B_{\rm met} = 32$ `[RAN]` B4. Report
  (negligible at the defaults).
- **The r2 encoder's training code** is the standalone DSN's (outside
  D-037); its configuration is read at the freeze, its training run is not
  re-examined, and its gate's $r_{\rm eff}$ implementation (in
  `Sbi-extractor`) is not read.
- **Not read, so not used for any claim:** `INFO_LOSS_THEORY_v1.md` (in
  neither the repository nor the knowledge base; its Propositions 5-8 are
  cited only as named by the plan and the handoff); the easy-positive mining
  paper the DSN's THEORY S3.7.5 cites; Tishby et al.'s information
  bottleneck paper; Fearnhead and Prangle's posterior-mean summaries (named
  by Akesson et al.).
- **Sources' status.** BayesFlow (IEEE TNNLS 2022), Goncalves et al. (eLife
  2020) and the Frontier review (PNAS 2020) are peer-reviewed; the Practical
  Guide (arXiv 2508.12939v1) is a preprint as held in the project. In the
  five PubMed Central full texts the connector dropped displayed formulas;
  only prose is quoted, and Wieczorek and Roth's variable letters, restored
  by the subagent from the publisher's page, are not quoted.

## 6. References / further reading

**Project knowledge base `[KB]`.** `SBI_PIPELINE.md` S4 (r2's $r_{\rm eff}$:
real 1.000, simulated 1.017, of $E = 10$; "the real arm lies essentially on
a line"), S5-S6 (1890 real windows, 29,616 simulated rows, $W = 18000$);
`claude/deck_pack/04_SEC_C_information_argument.md` (v1.1) C.1-C.3, C.6,
C.7; `claude/deck_pack/02_SEC_A_pipeline_today.md` A.8; `HPC_PATHS.md`
(2026-08-25b, the r2 `results.json` "[TO VERIFY]");
`claude/SBI_decisions_and_ideas_log.md` D-056 (72 and 108 windows per class,
by the tiling rule); E0 v1.9 (conventions, master table); E1 S3.1, S3.8; E2
eqs. (E2.4)-(E2.5), S3.3, S5; P1 S3.2, S4; P2 S1, S3.2-S3.7; P4 S3.5.

**Repository `[REPO 834eb41]`.** `hpc/joint/JOINT_DSN_NPE_PLAN_v0_6.md`
(v0.6.5) S2.1-S2.3 (lines 251-311), the approximations (line 1702), Stage
3b; `hpc/joint/HANDOFF_joint_dsn_npe_design.md` R1-R4 (lines 104-107), S12
item 1 (lines 312-317);
`hpc/dsn/Documentation/THEORY_joint_condition_search.md` S3.7.1, S3.7.5;
`hpc/dsn/metrics.py:1-131`;
`hpc/dsn/hpc/Config/refit_mea_joint_full_r2_l0_t82.json:57-58, 97, 113-131`;
`hpc/dsn/latent_burst_generator.py:460-527`;
`stage1/latent_sbi_simulator.py:128-237, 314-343`;
`stage1/build_latent_bank.py:211-248`;
`stage1/bench_burst_provider.py:82-96`;
`stage3/joint_diagnostics.py:107-146`;
`stage3/run_joint_arms.py:158, 560, 595, 607-608`;
`stage3/smoke_test_joint_arms.py:174-183`;
`stage3b/encoder_probes.py:156-163`; `stage4/npe_tune_joint.py:518`.

**Project PDFs `[KB-PDF]`, full text, page numbers of the PDF.** Radev ST,
Mertens UK, Voss A, Ardizzone L, Kothe U. *BayesFlow.* IEEE TNNLS
2022;33:1452-1466: p.2 (the summary network's role), p.5 (Proposition 2, "D.
Summary Network": avoiding information loss through handcrafted summaries;
aligning the architecture with the data's symmetry; 1-D convolutional
networks for time series). Deistler M, Boelts J, et al. *Simulation-Based
Inference: A Practical Guide* (arXiv 2508.12939v1, preprint): pp.7-8
(summary statistics and embedding networks; sufficiency; two-stage versus
end-to-end), p.31 (fixed embeddings versus end-to-end, in the context of
NLE). Goncalves PJ, Lueckmann J-M, Deistler M, et al. *Training deep neural
density estimators to identify mechanistic models of neural dynamics.* eLife
2020: p.5 (a CNN embedding for 1681-dimensional data), p.17 ("Use of summary
features"). Cranmer K, Brehmer J, Louppe G. *The frontier of
simulation-based inference.* PNAS 2020;117:30055-30062: p.3 (the curse of
dimensionality; summaries discard information), p.6 (learned summaries). No
number from these PDFs is used.

**Peer-reviewed literature.** According to PubMed, read in full text from
PubMed Central (by a subagent of this session, with verbatim quotes
returned; formulas were dropped by the connector): Papyan V, Han XY, Donoho
DL. *Prevalence of neural collapse during the terminal phase of deep
learning training.* PNAS 2020; PMID 32958680, PMC7547234,
[DOI](https://doi.org/10.1073/pnas.2015509117) -- NC1-NC4, the terminal
phase, centred means, balanced datasets (S3.4). Fang C, He H, Long Q, Su WJ.
*Exploring deep neural networks via layer-peeled model: minority collapse in
imbalanced training.* PNAS 2021; PMID 34675075, PMC8639364,
[DOI](https://doi.org/10.1073/pnas.2103091118) -- the ETF for class-balanced
data, minority collapse (S3.4). Recanatesi S, Bradde S, Balasubramanian V,
Steinmetz NA, Shea-Brown E. *A scale-dependent measure of system
dimensionality.* Patterns 2022; PMID 36033586, PMC9403367,
[DOI](https://doi.org/10.1016/j.patter.2022.100555) -- the participation
ratio; the spiral at large scales (S3.7). Wieczorek A, Roth V. *On the
difference between the information bottleneck and the deep information
bottleneck.* Entropy 2020; PMID 33285906, PMC7516540,
[DOI](https://doi.org/10.3390/e22020131) -- the IB objective; invariance of
mutual information (S3.6). Akesson M, Singh P, Wrede F, Hellander A.
*Convolutional neural networks as summary statistics for approximate
Bayesian computation.* IEEE/ACM TCBB 2022; PMID 34460381, PMC9847490,
[DOI](https://doi.org/10.1109/TCBB.2021.3108695) -- a CNN regressing the
posterior mean as a learned summary (S3.2). No number from these papers is
used. Records with a PMC copy that were screened by title and abstract and
not opened, having no content on the claims made here: Gaskell J, et al. J R
Soc Interface 2023 (graph-network summaries for ABC; PMC9810425,
[DOI](https://doi.org/10.1098/rsif.2022.0676)); Voznica J, et al. Nat Commun
2022 (deep learning from phylogenies; PMC9258765,
[DOI](https://doi.org/10.1038/s41467-022-31511-0)); Cao Y, et al. PNAS 2025
(simplex compression under adversarial training; PMC12054840,
[DOI](https://doi.org/10.1073/pnas.2421593122)); Lyu Z, et al. Entropy 2023
(IB-like training dynamics; PMC10377965,
[DOI](https://doi.org/10.3390/e25071063)); Du X, et al. Entropy 2021 (IB and
cascade learning; PMC8535168, [DOI](https://doi.org/10.3390/e23101360));
Litwin-Kumar A, et al. Neuron 2017 (located by citation lookup, not opened;
PMC5379477, [DOI](https://doi.org/10.1016/j.neuron.2017.01.030)). Min J, et
al. Genetics 2026 (PMC13334119,
[DOI](https://doi.org/10.1093/genetics/iyag107)) is cited by the plan from
an earlier session's reading and is not used here; its bioRxiv version (PMID
41573957) is superseded by it. Retrieved, abstract only, **not used for any
claim**: Sanchez T, et al. Mol Ecol Resour 2021
([DOI](https://doi.org/10.1111/1755-0998.13224)); Kimpson T, et al. J Theor
Biol 2026 ([DOI](https://doi.org/10.1016/j.jtbi.2026.112475)); Pinotti F, et
al. Proc Biol Sci 2026 ([DOI](https://doi.org/10.1098/rspb.2026.1059)); Lai
YL, et al. Neural Netw 2026
([DOI](https://doi.org/10.1016/j.neunet.2026.109224)); Wang S, et al. IEEE
TNNLS 2026 ([DOI](https://doi.org/10.1109/TNNLS.2025.3620798)).

**Searches run `[RAN]`, 2026-10-05.**

| source | query | result |
|---|---|---|
| PubMed | neural collapse simplex equiangular tight frame | 5 records: Papyan 2020 and Fang 2021 (PMC, read in full and used); Cao 2025 (PMC, adversarial simplex compression, not opened); two without a PMC copy (bipolar collapse in face verification; progressive feedforward collapse), abstract only, not used |
| PubMed | summary statistics neural network approximate Bayesian computation | 13 records, all screened by title and abstract: Akesson 2022 (PMC, read in full and used); Gaskell 2023 and Voznica 2022 (PMC, learned or deep-learning summaries, not opened); Min 2026 and its bioRxiv version (cited by the plan; not used here); Sanchez 2021, Kimpson 2026, Pinotti 2026 (no PMC copy, abstract only, not used); Polson 2025, Quelin 2025, Fortes-Lima 2021 (other uses of networks with ABC); two records on GWAS "summary statistics" (another sense of the term) |
| PubMed | participation ratio dimensionality neural representation | 6 records, none on the participation ratio of an embedding cloud; none used |
| PubMed | participation ratio dimensionality covariance eigenvalues neural activity | 0 records |
| PubMed | "participation ratio" dimensionality | 62 records; the first 20 screened by title and abstract: 16 use the participation ratio of a physical eigenstate, wave function or phonon mode (a localisation measure, another object); 4 use it as a dimensionality -- of fMRI signals, a single-cell embedding, a neural network's dynamics and a dynamical system's trajectories; none used |
| PubMed | citation lookup: Litwin-Kumar et al., Neuron 2017; Recanatesi et al., Patterns 2022 | both found; Recanatesi read in full and used; Litwin-Kumar not opened |
| PubMed | information bottleneck deep neural network representation mutual information | 9 records: Wieczorek and Roth 2020 (PMC, read in full and used); two IB-dynamics studies (PMC, not opened); six applications (registration, entity recognition, reinforcement learning, signal classification), not used |
| PubMed | triplet loss metric learning embedding class collapse | 0 records (P2's searches on triplet and semi-hard mining: Kertesz 2022, Chung and Lee 2023, read there) |
| bioRxiv | neuroscience, 2026-09-05 to 2026-10-04, first page of 30 records (the connector has no keyword search; records come in date order, so the page covers 2026-09-05 and 09-06 only) | none on learned summaries, metric learning, neural collapse or the participation ratio |
| bioRxiv | bioinformatics, same window, first page of 30 records (2026-09-05 to 09-07) | one on contrastive metric learning for protein-domain embeddings (10.64898/2026.08.28.747864, preprint, abstract only; no published version listed), not used; no preprint cited |
| data repositories | -- | E3 makes no claim about a public dataset; its data numbers concern the project's bank and the r2 export (KB) and the bench prior (code); no data-repository connector is among this session's tools; none queried |
| KB PDFs | BayesFlow, the Practical Guide, Goncalves et al., the Frontier review, searched for the passages cited | as cited above, with pages |

**Textbook, from memory** (tagged where used): the data-processing
inequality and its equality case; the chain rule of mutual information; the
bound of a discrete variable's entropy by $\ln$ of its number of values;
Fano's inequality; the invariance of mutual information under injective
maps; the Gaussian channel's $\frac{1}{2}\ln(1 + \text{SNR})$; the classical
and Bayesian senses of sufficiency.

---

### Pre-send check (Precision model)

R1 types: every symbol of S1 has its type; $\hat c$ is a map into the
labels, $h_{\rm b}$ a function on $[0, 1]$, $\Sigma_z$ and $\hat\Sigma_z$
positive semi-definite matrices, $r_{\rm eff}$ a number in $[1, E]$ for a
non-zero covariance; $H$ of a discrete argument is Shannon's and of a
continuous one differential; caps are measured in degrees. R2 hypotheses:
(E3.1)'s equivalence carries finite $I_{p_{\rm sim}}(\theta; x)$; (E3.3)
carries `joint_sep`, $\lambda_{\rm sep} > 0$, class-balanced batches with
two rows per class and the expectation over the metric stream's law; (E3.4)
carries the Markov chain and sufficiency, and the chain is a hypothesis on
the cohort; (E3.6) carries an exact code under $p_{\rm ev}$ and, for
$\Delta$, E2's first case; (E3.8) carries $\theta_{\mathcal{A}_{\rm free}}$
independent of $c$ (the bench); $r_{\rm eff} = C - 1$ carries equal masses
and an exact collapse; "rank one means two points" carries unit-norm
embeddings and exact rank one; the bench numbers carry their $C$, centres
and $\tau_{\rm ov}$. R3 transplants: Papyan et al.'s ETF (cross-entropy,
terminal phase, centred means) is not carried to an imposed penalty on raw
means; Fang et al.'s minority collapse is not carried to the separation
term; the Practical Guide's p.31 remark keeps its NLE context; the handoff's
"holds on the real arm" is restricted to the real windows' code; the plan's
ceiling is not carried from the real arm to the simulated law. R4 names:
$\hat c$ (read off $z$) and $c$ (true) are two objects, and the residual of
(E3.7) is measured from $\hat c$'s point; the plan's $\eta$ and $\mu_c$ are
E0's $\xi$ and $\bar z_c$; $\rho_{\rm cap}$ is not $\rho_{\rm ETF}$. R5
maps: $h_\psi : \mathbb{R}^{W} \to S^{E-1}$; the ETF lives in $S^{E-1}$, the
bench centres in $\Theta$, and F-ba's geometry is not compared with the ETF
target. R6 senses: collapse, global minimum, sufficient, the label, the
ceiling, information and size, information bottleneck, dimension and
cardinality, ETF, participation ratio (S3.9; S6's search row). R7 borrowed
phrasing: the deck's "a summary that is sufficient for the diagnosis carries
at most $\log C$" keeps "that takes at most $C$ values"; the Frontier
review's "invariably discards" keeps "not sufficient"; the plan's "width of
an information bottleneck" keeps "with a noise or resolution scale"; the
deck's "is that value" keeps "consistent with"; the DSN's "collapse-seeking"
is given its size. R8 levels: $\mathcal{L}^{\rm real}_{\rm DSN}$, $\Delta$,
$I$, $H$, $\Sigma_z$ and $r_{\rm eff}(\Sigma_z)$ analytic; $\ell_{\rm DSN}$,
$\bar z^{(B)}_c$, $K$, $\hat\Sigma_z$, the code's $r_{\rm eff}$,
$\hat\Delta$, ARI and $S_{\rm sil}$ computed, and the move between them
named where made (a bound on $\Delta$ is a bound on $\hat\Delta$ only up to
Monte Carlo error). Fixes made while writing: F-bb was first drafted as "the
joint stack returns 1.0 where the DSN returns 0.0" and narrowed after B2
showed that a constant cloud returns 1.0 in all three implementations unless
its covariance is exactly zero; the plan's "free axes carry exactly zero"
was given its error-free condition and the (E3.8) bound; P9's limit was
moved from $\ln C$ to $I_{p_{\rm sim}}(\theta; c)$; E2 S3.8's "is that
collapse's signature" and P2's "the composite loss's minimum is the
collapse" were given dated notes; the deck's C.0 sentence on summaries
sufficient for the diagnosis, and its F04 caption, were restricted to
summaries with at most $C$ values.
