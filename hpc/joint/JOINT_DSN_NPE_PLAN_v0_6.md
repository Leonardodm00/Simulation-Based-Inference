# Joint DSN + NPE training, nuisance separation, and mechanistic stratification

## Staged implementation plan

**Version** v0.6, 2026-09-02. Supersedes v0.5. Section 11 records what changed
and, in particular, what was **withdrawn**. Append changelog entries rather
than editing history in place.

---

**Abstract.** Two questions are joined here because they turn out to be the
same question asked twice. The first: the Deep Summary Network (DSN, encoder
$h_\psi$) is currently fitted on a phenotype metric-learning objective,
frozen, and the neural posterior estimator (NPE, flow $q_\omega(\theta\mid z)$)
is then fitted on the frozen embedding; does that two-stage, two-objective
procedure cost posterior quality relative to training the two jointly under
the NPE likelihood? The second: within one diagnosis, how much of the
culture-to-culture variation is nuisance (batch, plate, setup, detection) and
how much is genuine mechanistic heterogeneity -- the kind that could explain
why a compound acts on some cultures and not others? The link is that a
diagnosis-supervised summary statistic provably discards exactly what the
second question needs, and that a summary trained on the simulator's
parameters and on the experiment's replicate structure does not. This
document specifies the objective, the synthetic test-bed that adjudicates it,
the diagnostics, and eight implementation stages. Deliberately excluded: the
biophysics of the ANN simulator, spike detection, and the internals of the
misspecification gate (pointers given).

**The standing aim, stated explicitly rather than left implicit in the arm
definitions (IDEA-001).** The pipeline should be **patient-specific and free of
diagnostic bias**. The diagnosis is a coarse quantisation of a continuous
mechanistic state, applied by clinical criteria that need not track the biology
the simulator models; two patients under one label may differ mechanistically
as much as two patients across labels, which is a candidate explanation for
differential drug response. So the label is not to be used as a training input
at all. Each culture gets its own posterior over mechanism, and any grouping of
patients is an **output** of that, discovered in parameter space, never an
assumption fed in. The methodological force of this is in S2.5: if the encoder
is trained on the label, "patients cluster by diagnosis in parameter space" is
circular; removing it converts the label into a held-out evaluation variable
and the statement into a measurement. Arms A2 and A0 remain in the study
precisely as the comparators that test whether removing it costs anything.

Claims are tagged: **[KB]** project knowledge base (full text), **[REPO]**
read from the repositories, **[PubMed full text]**, **[PubMed abstract
only]**, or **[reasoning]** (mine, no source).

---

## 1. Notation

| Symbol | Meaning | Type / domain | Units | First used |
|---|---|---|---|---|
| $x$ | one IFR window | $x \in \mathbb{R}_{\ge 0}^{W}$ | Hz | S2 |
| $W$ | window length in samples, $W = \mathrm{round}(T_{\rm win} f_s)$ | $\mathbb{N}$ | samples | S2 |
| $\theta$ | inference parameters | $\theta \in \Theta \subset \mathbb{R}^{d_\theta}$ | mixed | S2 |
| $d_\theta$ | parameter-space dimension; $26$ on the real bank | $\mathbb{N}$ | -- | S2 |
| $\theta^{\rm ns}$ | the neuron/synapse block (23 axes); **descriptive only from v0.6, no longer a constrained/unconstrained partition** | subvector of $\theta$ | mixed | S2.5 |
| $\theta^{\rm topo}$ | the Weibull connectivity-kernel axes (`p0_conn`, `d0_conn`, `beta_conn`; 3 axes) | subvector of $\theta$ | mixed | S2.5 |
| $\mathcal{G}$ | a realised connectivity graph, drawn from the kernel at fixed $\theta^{\rm topo}$; a latent nuisance, **not** a parameter | adjacency | -- | S2.5 |
| $\nu$ | nuisance variables: act on the observation map, not the dynamics; outside $\Theta$ | $\nu \in \mathcal{N}$ | mixed | S2.5 |
| $\phi$ | latent factor vector of the synthetic generator; on the bench $\theta := \phi$ | $\phi \in (0,1)^n$, $n=6$ | dimensionless | S4 |
| $S$, $F$ | label-carrying and label-irrelevant latent axes | index sets, disjoint | -- | S4 |
| $c$ | phenotype (diagnosis) label | $c \in \{0,\dots,C-1\}$; $C = 2$ on the cohort | -- | S2 |
| $h_\psi$ | encoder, $h_\psi : \mathbb{R}^{W} \to S^{E-1}$ | weights $\psi$ | -- | S2 |
| $z$ | embedding, $z = h_\psi(x)$, L2-normalised | $z \in S^{E-1} \subset \mathbb{R}^E$ | -- | S2 |
| $E$ | embedding dimension; searched in $\{8..16\}$ | $\mathbb{N}$ | -- | S2 |
| $q_\omega(\theta \mid z)$ | conditional flow (zuko NSF via sbi) | density on $\Theta$ for each fixed $z$ | -- | S2 |
| $\mathcal{L}^{\rm sim}_{\rm NPE}$ | NLL of $\theta$ given $h_\psi(x)$, under the simulator's joint law | $\mathbb{R}$ | nats/row | S2.2 |
| $\ell_{\rm DSN}$ | composite metric loss (triplet + angular + ETF separation) on real class-labelled batches | $\mathbb{R}_{\ge 0}$ | -- | S2.2 |
| $\mathcal{L}^{\rm real}_{\rm rep}$ | replicate-consistency loss on same-donor pairs, eq. (3c) | $\mathbb{R}_{\ge 0}$ | dimensionless | S2.5 |
| $\lambda_{\rm dsn}$, $\lambda_{\rm rep}$ | weights of $\ell_{\rm DSN}$ and $\mathcal{L}^{\rm real}_{\rm rep}$ | $\ge 0$; searched | -- | S2.2, S2.5 |
| $g, g'$ | two wells (cultures) of one donor $P$ | index | -- | S2.5 |
| $m_g$ | culture-level posterior mean, $m_g = \mathbb{E}_{q_\omega}[\theta \mid x_g]$, for each fixed $x_g$; **over all $d_\theta = 26$ axes from v0.6** | $\mathbb{R}^{26}$ | param units | S2.5 |
| $\hat m_g$ | the $S_{\rm mc}$-draw Monte Carlo estimate of $m_g$ | $\mathbb{R}^{26}$ | param units | S2.5 |
| $S_{\rm mc}$ | posterior draws used to form one $\hat m_g$ | $\mathbb{N}$ | -- | S2.5 |
| $\Delta_{gg'}$ | replicate disagreement, $\Delta_{gg'} = m_g - m_{g'}$ | $\mathbb{R}^{26}$ | param units | S2.5 |
| $M$ | metric: symmetric positive-definite quadratic form | $26\times26$ | (param units)$^{-2}$ | S2.5 |
| $T_{gg'}$ | disagreement scalar, $T_{gg'} = \Delta_{gg'}^\top M \Delta_{gg'}$ | $\ge 0$ | dimensionless | S2.5 |
| $H_0$ | the null hypothesis behind the target, eq. (3e); a **conjunction** of two claims | -- | -- | S2.5 |
| $C_g$, $C_{g'}$ | posterior covariance $\mathrm{Cov}_{q_\omega}(\theta\mid x_g)$, for one fixed $x_g$ | PSD $26\times26$ | (param units)$^2$ | S2.5 |
| $\bar C$ | symmetrised posterior covariance, $\bar C = (C_g + C_{g'})/2$ | PSD | (param units)$^2$ | S2.5 |
| $V$ | $\mathrm{Var}(\Delta_{gg'}\mid\theta^*)$ over replicate datasets at fixed $\theta^*$; $\ne C$; **derivation-only** | PSD | (param units)$^2$ | S2.5 |
| $F$ | Fisher information of one well's data about $\theta$, at $\theta^*$; **derivation-only** | PSD | (param units)$^{-2}$ | S2.5 |
| $\Sigma_0$ | prior covariance; diagonal and analytic for a box, $(b_k-a_k)^2/12$ | PSD $26\times26$ | (param units)$^2$ | S2.5 |
| $\lambda_j$ | $j$-th generalised eigenvalue of $(F, \Sigma_0^{-1})$; data-to-prior precision ratio | $\ge 0$ | dimensionless | S2.5 |
| $c_j$ | prior's share of the posterior precision along direction $j$, $c_j = 1/(1+\lambda_j)$ | $(0,1]$ | dimensionless | S2.5 |
| $p_{\rm eff}$ | effective number of constrained directions, $\sum_j \lambda_j/(1+\lambda_j)$ | $[0, 26]$ | dimensionless | S2.5 |
| $N_{\rm pair}$ | same-donor well pairs available | $\mathbb{N}$ | pairs | S2.5 |
| $t_{\rm warm}$ | warm-up: fraction of training before $\lambda_{\rm rep}$ ramps in | $[0,1)$; searched | -- | S2.5 |
| $B_{\rm sim}$, $B_{\rm met}$, $B_{\rm rep}$ | rows per step in the simulated, metric and replicate streams | $\mathbb{N}$ | rows | S2.2 |
| $M_{\rm ens}$ | ensemble members (independent (encoder, flow) pairs); renamed from $M$ in v0.6 to stop clashing with the metric | $\mathbb{N}$ | -- | S3 |
| $L$, $L_0$, $\hat\Delta$ | held-out NLL, prior floor, information gain $\hat\Delta = L_0 - L$ | $\mathbb{R}$ | nats/row | S3 |
| $d_i$ | paired per-row NLL difference between two arms, same row | $\mathbb{R}$ | nats | S2.4 |
| $D$ | population mean of $d_i$; $D > 0$ means the first arm is worse | $\mathbb{R}$ | nats/row | S2.4 |
| $g_{\rm grp}$ | resampling group: topology draw (simulated bank) or culture (real bank) | index | -- | S2.4 |
| $n_g$, $\bar n$, $n_0$ | group sizes; mean; the unequal-size correction of the design effect | $\mathbb{N}$, $\mathbb{R}_{>0}$ | rows | S2.4 |
| $\rho(d)$ | intraclass correlation of $d$ within groups | $[0,1]$ | -- | S2.4 |
| $\mathrm{DEFF}$ | design effect, $1 + (n_0 - 1)\rho(d)$ | $\ge 1$ | -- | S2.4 |
| $\sigma_b^2$, $\sigma_w^2$ | between- and within-group variance components of $d$ | $\ge 0$ | nats$^2$ | S2.4 |
| $\hat\Delta^{(k)}$, $\hat\Delta_{S\mid c}$ | per-axis gain; within-class gain on the label axes | $\mathbb{R}$ | nats | S2.3 |
| $\kappa_k$ | posterior contraction on axis $k$ | $(-\infty, 1]$ | -- | S3 |
| $r_{\rm eff}$ | participation-ratio effective rank of an embedding cloud | $[1, E]$ | -- | S3 |
| $\rho_{\rm grad}$ | cosine between the two loss terms' gradients w.r.t. $\psi$ | $[-1,1]$ | -- | S3 |
| $\pi$ | simulation-gap severity of the pseudo-real bench arm | $[0,1]$; $0$ = no gap | -- | S4.0 |
| $J_\theta$, $J_\nu$ | Jacobians $\partial z/\partial\theta_k$, $\partial z/\partial\nu_m$ | $E \times d_\theta$, $E \times d_\nu$ | -- | S2.6 |
| $a_m$ | aliasing coefficient of nuisance direction $m$ | $[0,1]$ | -- | S2.6 |
| $\delta\theta_m$ | parameter shift that nuisance direction $m$ is mistaken for | $\mathbb{R}^{d_\theta}$ | param units | S2.6 |
| $\Sigma_{\rm rep}$ | covariance of $m_g - m_{g'}$ over same-donor pairs: the empirical nuisance floor | PSD $26\times26$ | (param units)$^2$ | S2.7 |
| $\Sigma_{\rm pat}$ | covariance of donor means within one diagnosis | PSD $26\times26$ | (param units)$^2$ | S2.7 |
| $\Sigma_{\rm dx}$ | covariance of diagnosis means | PSD $26\times26$ | (param units)$^2$ | S2.7 |
| $\mu_j$ | generalised eigenvalue of $(\Sigma_{\rm pat}, \Sigma_{\rm rep})$; **renamed from $\lambda_j$ in v0.6** to stop clashing with the Fisher spectrum | $\ge 0$ | -- | S2.7 |
| $\delta_d$ | intervention of compound $d$ on the mechanism, $\theta \mapsto \theta \oplus \delta_d$ | $\mathbb{R}^{d_\theta}$ | param units | S2.8 |
| $G$, $G_{\rm don}$ | cultures (wells); distinct donors behind them | $\mathbb{N}$; $G = 35$ | -- | S5.1 |

**Conventions.** (i) Conditional quantities are written conditionally
throughout. (ii) The parameter-space dimension is $d_\theta$, not $p$; $p$ is
reserved for densities. This deviates from `SBI_PIPELINE.md`, which overloads
$p$; the referent is unchanged. (iii) Inference coordinates as in the pipeline
document: 17 axes as $\ln$, 9 linear. (iv) One *culture* is one well; one
*donor* may contribute several cultures (this is an open design question, D12).
Windows of one culture never split across train / selection / report.
(v) $\psi$ and $\omega$ are network weights, $\phi$ is the bench latent vector,
$\nu$ is nuisance -- four different symbols on purpose.
(vi) **New in v0.6 -- evaluated versus derivation-only.** Every quantity in
S2.5 is marked **[EVALUATED]** if the pipeline computes it or **[DERIVATION
ONLY]** if it appears solely in the reasoning that justifies a formula.
$V$, $F$ and $\theta^*$ are derivation-only on real data. Conflating the two
categories is what makes the replicate construction look impossible when it is
not.
(vii) **New in v0.6 -- two symbol collisions repaired.** The metric is $M$ and
the ensemble size is now $M_{\rm ens}$; the Fisher-versus-prior generalised
eigenvalues are $\lambda_j$ and the stratification ones are $\mu_j$. v0.5
overloaded both letters silently.

---

## 2. Model and objective

### 2.1 What exists (read from the repositories and the knowledge base)

- **Who is fitted on what, today.** The encoder is fitted by
  `run_optimization.py` on the REAL cohort, `data_mode = "numpy"`, with the
  `condition` labels; $\theta$ appears nowhere in that run [REPO,
  `config_mea_joint_full.json`]. The NPE is then fitted on SIMULATED
  $(\theta, z)$ pairs, $z$ produced by that frozen encoder applied to
  simulated windows [REPO, `export_embeddings.py`; KB, `SBI_PIPELINE.md` S5].
  **So under A0 the encoder is fitted on one distribution and evaluated on
  another, and the flow never sees a real window during training.** This is
  ordinary covariate shift, and it is the subject of Stage 3b.
- **Encoder.** `backbone.py::OneDCNNBackbone` is a pure tensor-to-tensor
  RegNet-style 1D CNN, GroupNorm throughout (no BatchNorm), L2-normalised
  output, built from a frozen `BackboneConfig` [REPO]. Usable as an sbi
  `embedding_net` without modification.
- **DSN objective.** `CompositeDSNLoss` = triplet margin + angular hinge +
  $\lambda_{\rm sep}(t)$ times a simplex-ETF penalty on normalised class
  means, supervised by true class labels, with surrogate rows masked out of
  class statistics [REPO]. Batches from
  `ConditionBalancedBatchSampler` + `TripletCollator`; miners from
  `pytorch_metric_learning` under `CosineSimilarity` [REPO, `train.py`].
  `positives_mode: cross_culture` excludes same-culture positives, which is
  already a nuisance-invariance device [REPO].
- **NPE.** `npe_model.py`: `sbi 0.27.0`, `posterior_nn(model="zuko_nsf",
  z_score_theta="transform_to_unconstrained", z_score_x="independent")`,
  arithmetic-mixture ensembles of independently trained members [REPO].
- **Tuning stack.** `npe_tune*.py`: frozen grouped 3-way split with hash,
  atomic per-trial JSON ledger, stateless `skopt.Optimizer` replayed from the
  ledger with constant-liar batching, held-out NLL in nats/row as objective,
  gates G1 (informativeness) / G2 (marginal SBC, Holm) / G3 (coverage, TARP)
  on finalists, and a Stage-1 harness against an exact-posterior GMM
  benchmark [REPO; KB, `HPC_PATHS.md` S5c]. Weight decay and LR schedules are
  absent from the space because sbi's training loop does not expose them
  [REPO, `npe_tune_search.py`].
- **Bench generator.** `latent_burst_generator.py`: $\phi \in [0,1]^n$,
  $n = 6$; class centres on a regular simplex; label axes $S$, free axes
  $F$; $\phi \mapsto$ `BurstParams` $\mapsto$ Poisson network-burst spike
  trains $\mapsto$ Gaussian-smoothed cumulative IFR [REPO].
- **Nuisance already exists in the ANN pipeline.** The virtual-MEA stage
  (`ProbeConfig`, electrode positions, detection threshold, active-electrode
  count) is an observation-level transformation applied after the dynamics
  [REPO]. It is a nuisance generator that nobody has used as one.
- **The bank's topology structure** [REPO, `HPC_PATHS.md` S4a; KB,
  `SBI_PIPELINE.md` S3]. Simulations are stored as `topo_*/iter_*`: the three
  Weibull kernel axes are drawn **per topology directory** and the 23
  neuron/synapse axes are swept per iteration inside it (`manifest.json`:
  23 `active_indices`, group `neuron_synapse`). The campaign of record spans
  383 distinct topology draws. `npe_tune_data.derive_groups` keys on exactly
  those kernel axes. **This becomes load-bearing in v0.6 (S2.5c) and is the
  origin of new decision D17.**
- **The standing caveat.** The r2 encoder has real-arm $r_{\rm eff} = 1.000$
  of $E = 10$, simulated-arm $1.017$, and every per-window posterior is
  prior-like (median HPR width 0.9788 of the prior at $\varepsilon = 0.05$
  over 1890 windows) [KB, `SBI_PIPELINE.md` S4, S11].

### 2.2 The joint objective

$$\mathcal{L}(\psi,\omega) \;=\; \mathcal{L}^{\rm sim}_{\rm NPE}(\psi,\omega) \;+\; \lambda_{\rm dsn}\,\mathcal{L}^{\rm real}_{\rm DSN}(\psi) \;+\; \lambda_{\rm rep}\,\mathcal{L}^{\rm real}_{\rm rep}(\psi,\omega), \tag{1}$$

$$\mathcal{L}^{\rm sim}_{\rm NPE}(\psi,\omega) = \mathbb{E}_{(\theta,x)\sim p_{\rm sim}(\theta, x)}\big[-\log q_\omega\big(\theta \mid h_\psi(x)\big)\big], \qquad \mathcal{L}^{\rm real}_{\rm DSN}(\psi) = \mathbb{E}_{(x,c)\sim p_{\rm real}(x, c)}\big[\ell_{\rm DSN}\big(h_\psi(x),\, c\big)\big]. \tag{1a}$$

The third term is defined in S2.5. Each term is fed by whichever arm carries
the label it needs: $\theta$ is observed only under $p_{\rm sim}$; $c$ and the
replicate structure only under $p_{\rm real}$. Per optimiser step,

$$\hat{\mathcal{L}}_{\rm step} = \frac{1}{B_{\rm sim}}\sum_{i \in \mathcal{B}_{\rm sim}} -\log q_\omega\big(\theta_i \mid h_\psi(x_i^{\rm sim})\big) \;+\; \lambda_{\rm dsn}\, \ell_{\rm DSN}\big(h_\psi(X^{\rm real}_{\rm met}),\, y^{\rm real}_{\rm met},\, \mathcal{T}_{\rm mined}\big) \;+\; \lambda_{\rm rep}\,\hat{\mathcal{L}}^{\rm real}_{\rm rep}, \tag{2}$$

with $\mathcal{B}_{\rm sim}$ an i.i.d. prior-faithful minibatch of simulated
rows and $(X^{\rm real}_{\rm met}, y^{\rm real}_{\rm met})$ the DSN's own
class-balanced real batch, built by the existing sampler and collator.

Two mechanical facts:

- **The streams interact only through the weights.** The backbone is
  GroupNorm throughout with no BatchNorm [REPO], so every layer is
  sample-wise: $z_i = h_\psi(x_i)$ exactly, for each fixed $\psi$ and each
  fixed $i$, whatever else occupies the batch. Under a BatchNorm backbone the
  streams would also mix through running statistics -- a stateful channel
  invisible to any analysis of $\psi$. Stage 0 asserts the absence of
  BatchNorm modules.
- **Gradient reach differs by term.** $\ell_{\rm DSN}$ touches $\psi$ only.
  $\mathcal{L}^{\rm sim}_{\rm NPE}$ and $\mathcal{L}^{\rm real}_{\rm rep}$
  touch both $\psi$ and $\omega$, the latter because it evaluates the
  posterior -- except that S2.5d stops the gradient into $\omega$ for the
  replicate term, so in the implementation the third term reaches $\psi$ only.
  Augmented surrogate rows carry no valid $(\theta,x)$ pair, are real-arm rows
  anyway, and so are excluded from the first term automatically [reasoning].

**Arms.**

| arm | encoder $\psi$ fitted by | flow $\omega$ | what it is |
|---|---|---|---|
| **A0** (current) | $\ell_{\rm DSN}$ on **real**, then frozen | NPE on sim at frozen $z$ | two-stage; encoder used out of domain (S2.1, Stage 3b) |
| **A0s** (bench control) | $\ell_{\rm DSN}$ on **sim** with generator labels, then frozen | NPE on sim | same objective, no domain shift; isolates the two effects |
| **A1** | $\mathcal{L}^{\rm sim}_{\rm NPE}$ only | joint | sbi embedding-net / BayesFlow regime |
| **A2** | NPE $+\ \lambda_{\rm dsn}\ell_{\rm DSN}$ | joint | the status quo made joint; nests A1 |
| **A2s** (bench control) | both terms on **sim** | joint | A2 without the domain asymmetry |
| **A3** | A2 warm-started at A0's $\psi^\star$ | joint | is A0's encoder a good basin? |
| **A5** | NPE $+\ \lambda_{\rm rep}\mathcal{L}^{\rm real}_{\rm rep}$ | joint | replicate-consistency supervision (S2.5); label never used in training |

A0s and A2s exist only where simulated windows carry a phenotype label. On the
bench they do; on the real bank they do not, and no proxy is proposed -- the
condition-to-biology mapping is itself unestablished [KB, O4].

Precedent for joint training: backpropagating the posterior loss into the
summary network is the BayesFlow construction [KB]. According to PubMed, Min
et al. train the embedding and the flow jointly by SGD on the negative log
posterior, and separately use a frozen network pre-trained on another
objective, with its last layer removed, as learned summaries -- the published
analogue of A0 [PubMed full text, [DOI](https://doi.org/10.1093/genetics/iyag107)].

### 2.3 Why label supervision can lose information

Write $\theta = (\theta_S, \theta_F)$. On the bench $c \to \phi_S \to x$ and
$\phi_F \to x$ with $\phi_F$ independent of $c$ [REPO, `sample_latents`].

**The asymmetry (the core of the argument).** For the chain
$c \to \theta \to x \to z$:

- if $z$ is sufficient for $\theta$, then $p(c \mid x) = \int p(c\mid\theta)\,
  p(\theta\mid z)\,d\theta = p(c\mid z)$, so $I(c;z) = I(c;x)$: **a
  $\theta$-sufficient summary loses nothing about the diagnosis**;
- the converse fails: $z = c$ is perfectly $c$-sufficient and carries at most
  $\log C$ nats about $\theta$ -- all of $\theta$, label axes included.

So the two objectives are not symmetric alternatives. Likelihood supervision
retains the label for free *if the simulator mediates the pathology*;
label supervision does not retain the parameters under any circumstance.

**The label ceiling.** At the global minimum of $\ell_{\rm DSN}$ -- every
window of class $c$ mapped to one point $\mu_c$, the means on a simplex ETF --
$z$ is a relabelling of $c$, so $\hat\Delta \le \log C$ regardless of $E$,
$d_\theta$, capacity or bank size, and the free axes carry exactly zero. For
$C = 2$ that is $0.693$ nats for all 26 axes together. Everything beyond the
label must travel through the within-class residual $\eta = z - \mu_c$, which
the margin $m_{\cos}$ and angular half-angle $\alpha$ penalise -- making them
the width of an information bottleneck. Proofs: Propositions 5-7 of
`INFO_LOSS_THEORY_v1.md`.

The measured $r_{\rm eff}(\text{real}) = 1.000$ at $C = 2$ is exactly the
collapse prediction $C - 1$, which reframes the r2 encoder as **better
optimised**, not more broken [KB; theory doc S3.5.3].

**What is discarded is not "the nuisance".** The label partitions variation
into "the label" and "everything else", and everything else includes the
within-class variation of the very mechanisms the label is about: severity,
penetrance, which mechanism is affected in which patient. On the bench that is
$\tau_{\rm ov} > 0$, the within-class spread of the label axes themselves. The
reductio: if the class determined the parameters ($H[\theta\mid c] = 0$) there
would be no inference problem -- a classifier and a two-row lookup table would
be the whole method. The premise of NPE is $H[\theta\mid c] > 0$.

**Pre-registered predictions.** Numbering is preserved across versions so that
cross-references to the theory document stay valid; **P5 was withdrawn in
v0.3** (S11).

| # | prediction | measured by | falsified if |
|---|---|---|---|
| P1 | free-axis gain lower under A0 than A1, beyond seed spread | per-axis $\hat\Delta^{(k)}$, bench | free-axis gains equal or higher under A0 |
| P2 | on label axes the arms are comparable; A0 may lead at small bank size | per-axis $\hat\Delta^{(k)}$ vs $n_{\rm tr}$ | A0 loses on label axes too |
| P3 | $r_{\rm eff}\to C-1$ under any arm with large $\lambda_{\rm dsn}$ | gate's $r_{\rm eff}$ report | stays well above $C-1$ |
| P4 | $\rho_{\rm grad} < 0$ on a non-trivial fraction of steps in A2 | per-epoch probe | stays non-negative |
| ~~P5~~ | *withdrawn v0.3* | -- | -- |
| P6 | arm ranking by simulated-arm NLL need not match ranking by pseudo-real NLL | both endpoints, bench | rankings coincide across all arms |
| P7 | at the joint optimum with $\lambda_{\rm dsn}>0$, selected $m_{\cos}$, $\alpha$ sit below the clustering-only optimum | two campaigns | selected margins equal or larger |
| P8 | $\hat\Delta(\mathrm{A0})$ on the real bank does not exceed $\log 2 = 0.693$ nats | one `evaluate` run, existing bank | materially above 0.693 |
| P9 | on the bench, $\hat\Delta(\mathrm{A0}) \to \log C$ from below as ARI $\to 1$, monotonically | $\hat\Delta$ vs ARI across seeds | no relationship, or above $\log C$ |
| P10 | marginal value of extra windows per culture is zero under a collapsed encoder, positive otherwise | $\hat\Delta$ vs windows aggregated (S2.7) | A0 gains from aggregation as much as A1 |
| P11 | aliasing is worst along sloppy directions | $a_m$, $\delta\theta_m$ vs information spectrum (S2.6) | no relationship |
| P12 | pre-treatment posterior predicts post-treatment change better than label, $z_g$, or baseline statistics | held-out $R^2$ / AUC by batch (S2.8) | no advantage over baselines |
| P13 | same-donor embedding distances order A5 $<$ A2 $<$ A1, with A0(real) near A2 | the S3 distance table, Stage 3 | no ordering, or A5 not tightest |
| P14 | under a well-calibrated arm, $T_{gg'}$ concentrates near $p_{\rm eff}$; under A0 it sits far below (collapse) | eq. (3c) statistic, all arms | $T$ near $p_{\rm eff}$ under A0 too |
| **P15** | moving the constraint from 23 to 26 axes changes $p_{\rm eff}$ by much less than 3, because the kernel axes are sloppy: each direction contributes $1-c_j \in [0,1)$, not 1 | $p_{\rm eff}$ computed on both index sets, same posterior | $p_{\rm eff}(26) - p_{\rm eff}(23) \approx 3$, i.e. the kernel axes are stiff -- in which case D17 becomes urgent, not precautionary |
| **P16** | the same-donor disagreement on the three kernel axes, normalised per direction by $1-c_j$, is not systematically larger than on the neuron/synapse axes | per-direction decomposition of $T_{gg'}$, real cohort, arm A1 | kernel axes dominate $T_{gg'}$ across donors, which is evidence against the shared-kernel assumption of S2.5c |

**P8 deserves emphasis: it is a quantitative prediction about data already in
hand, checkable with tooling already written, requiring no new simulation.**
**P15 is nearly as cheap: it needs one `information_spectrum` call and no
training at all, and it is the check that says whether v0.6's central change
costs anything.**

### 2.4 Decision rule for "does A0 degrade the NPE"

On the frozen selection split, $n_{\rm seed} \ge 5$ per arm, with the
**pseudo-real held-out NLL** as primary endpoint (S4.0) and the simulated-arm
NLL reported alongside:

1. **Pair at the row level.** Both arms are scored on the same held-out
   rows, so form $d_i = \ell_i^{\rm A0} - \ell_i^{\rm A1}$ with
   $\ell_i = -\log q_\omega(\theta_i\mid z_i)$, and estimate
   $D = \mathbb{E}[d]$. Row difficulty is largely a property of the row and
   cancels exactly in $d_i$; the covariance between arms is then never
   estimated because it is never needed. Pairing is on the row
   $(\theta_i, x_i)$, which is identical across arms even though $z_i$ is not.
   The ensemble size $M_{\rm ens}$ is held fixed across arms, since $L$ is
   computed from the mixture density.
2. **Interval by paired bootstrap over the resampling unit**, using
   `bootstrap_paired.py` (S2.4a). Primary: `within="all"` -- resample groups
   with replacement, take every row of each drawn group, ratio estimator
   $\sum d_i / \sum n_g$. Report $\rho(d)$, $n_0$, $\mathrm{DEFF}$ and the
   tail share alongside, and the other three schemes as the comparison.
3. **The decision.** "A0 degrades" iff $D - 2\sigma_{\rm seed} > 0$ **and**
   the CI of $D$ excludes 0, with $\sigma_{\rm seed}$ the across-seed SD
   measured as the tuning stack measures it. The two clauses answer different
   questions and neither substitutes for the other: the bootstrap is
   conditional on the trained networks and varies only the evaluation sample,
   while $\sigma_{\rm seed}$ is training stochasticity. A narrow bootstrap
   interval on one lucky initialisation would declare a winner with no
   justification. Same rule per axis, and for $\hat\Delta_{S\mid c}$ (the
   within-class gain on the label axes), which is the decisive quantity for
   the heterogeneity question rather than the free-axis gain.
4. Every arm must pass G1 and G2. An arm failing G1 has "learned nothing", not
   "calibrated well" -- a prior-equal posterior is perfectly calibrated [KB].

### 2.4a The resampling unit, and why all four schemes are reported

`bootstrap_paired.py` (**written and verified, S6 Stage 1**) implements four
schemes and the module makes no choice among them; the diagnostic stage does,
on measured $\rho(d)$.

| scheme | unit and content | weighting | when it is right |
|---|---|---|---|
| `none` | rows, with replacement | row | only when $\rho(d)\approx 0$ |
| `all` | groups; **every row** of each drawn group | row | **primary** |
| `mean` | groups; the group mean $\bar d_g$ | group | when the culture is the unit of interest |
| `one` | groups; one random row per drawn group | group | never preferred: same estimand as `mean`, variance larger by $n_g$ |

Four facts that fix the choice, each verified numerically in the smoke test
rather than asserted:

- **The unit, not the replacement, is what matters.** Both `none` and `all`
  draw with replacement. They differ because the multiplicity of a group in a
  resample is $\mathrm{Poisson}(1)$-like under `all` (relative fluctuation
  $O(1)$: groups appear twice or not at all) but
  $\mathrm{Binomial}(N, n_g/N)$ under `none` (relative fluctuation
  $O(1/\sqrt{\bar n})$). `none` therefore varies the group weights by about
  $1/\sqrt{\bar n}$ of what the sampling design does, and averages the
  group-level variability away before it can reach the interval. Measured
  ratio $\mathrm{se(none)}/\mathrm{se(all)} = 1/\sqrt{\mathrm{DEFF}}$ to two
  decimals across $\rho \in \{0, 0.05, 0.3, 1\}$.
- **The cost is coverage, and it is silent.** At $\rho = 0.15$, $n = 40$,
  a nominal 95% `none` interval covered the truth **57%** of the time against
  93% for the grouped schemes. Nothing in the output of the failing run
  indicates anything is wrong.
- **`all` costs nothing when `none` would have been right.** At $\rho = 0$ the
  two agreed to three decimals. There is no regime in which `none` is
  preferable, only regimes in which it is harmless.
- **`all` and `mean` differ in estimand**, not merely in variance, whenever
  group sizes are unequal -- which they are here. This matters concretely:
  the MFR filter strips rows from quiet networks, so group size correlates
  with the biology, and `all` silently upweights the more active topologies.
  Report both; agreement is a sentence, disagreement is a finding.

**Why $\rho(d) > 0$ is expected here, but small.** `derive_groups` keys on the
connectivity-kernel axes of $\theta$ [REPO], and the bank is
$\theta$-deduplicated, so rows within a group share **3 of 26 axes** and vary
independently on the other 23. They are genuine distinct realisations, not
near-duplicates -- an earlier draft of this plan mischaracterised them.
Consequently $\rho(d)$ should be small. It is not zero, because $d$ is a
function of $\theta$ and three components of $\theta$ are constant within a
group by construction; if any of the arm difference is driven by connectivity,
it is shared across all rows of a group. With $\bar n \approx 77$, even
$\rho = 0.05$ gives $\mathrm{DEFF}\approx 4.8$.

**The design effect is axis-dependent, so compute it per axis group.** The 3
connectivity axes were drawn $G$ times whatever the row count, so a per-axis
statistic on them has effective sample size $G$; the 23 neuron/synapse axes
were drawn once per row. Run the comparison on the aggregate *and* separately
on the two axis groups. **v0.6 note:** this axis-group split survives the
removal of the constrained/unconstrained partition in S2.5 -- it is a statement
about how the bank was *generated*, not about which axes the loss touches, and
the two must not be conflated.

**Preconditions, both checked by the module.** $d$ must be finite -- a flow
assigning zero density at a held-out $\theta$ gives $-\infty$ and silently
destroys the mean. And the tail share of the largest 1% of $|d_i|$ must be
small: the bootstrap for a mean needs a finite second moment, and if a handful
of rows carry the mass then no interval for the mean is trustworthy, clustered
or not -- report a trimmed or median-based statistic and say why. Pairing
cancels the *common* component of row difficulty, not arm-specific
catastrophes, so this check runs on $d_i$ and not only on $\ell_i$.

**Use the group key the split was frozen on**, read from the split manifest;
do not reconstruct a plausible-looking one at scoring time. If the manifest
records `grouping == "iid_fallback"` the split was never grouped, only `none`
is meaningful for that bank, and the fallback itself needs chasing --
it means `param_names` lacked the topology axes.

**Not folded into one interval.** A two-level bootstrap over (seed, group)
would answer the procedure-level question directly, but with
$n_{\rm seed} = 5$ the seed level has five atoms and bootstrap asymptotics
fail. The seed level also cannot be paired: seed $k$ of A0 has no
correspondence with seed $k$ of A1, the architectures and objectives differing.
That is why it is the noisier component, and why extra seeds buy more than
extra evaluation rows. At $n_{\rm seed}\gtrsim 20$ revisit this.

### 2.5 What should real data teach the encoder?

Real recordings carry no $\theta$. So the question is not "phenotype loss:
yes or no" but **by what channel should real data constrain $\psi$ at all**.
Three candidates, and they are the arms.

**(i) Nothing (A1).** The encoder is fitted exclusively to simulator output.
The gate says the simulator is decisively misspecified relative to the cohort
[KB, S8], so A1's features may be simulator artefacts with no stable
counterpart in real recordings, and nothing in A1 would reveal it. This is a
real exposure, not a rhetorical one; the gate and witness are the monitors.

**(ii) The phenotype (A2).** The status quo made joint. Its cost is S2.3. Its
benefit is that it forces $h_\psi$ to compute something well defined and
discriminative on real recordings, and `cross_culture` positives make it
partly a nuisance-invariance device already.

**(iii) Replicate consistency (A5) -- recommended.** You do not know $\theta_g$
for a culture, but you know that two wells from the same donor must have the
*same* $\theta$. That is a trainable constraint requiring no ground
truth and no label:

$$\mathcal{L}^{\rm real}_{\rm rep}(\psi,\omega) \;=\; \sum_{\text{donor } P}\ \sum_{(g,g')\in P}\ \Big(\log T_{gg'} - \log p_{\rm eff}\Big)^2, \qquad T_{gg'} = \Delta_{gg'}^\top M\, \Delta_{gg'}, \tag{3}$$

summed over same-donor well pairs. The design of the three free choices --
**which axes**, **which metric $M$**, and **which target** -- is where the
content sits, and each was got wrong in an earlier draft. Full derivations:
`METRIC_REPLICATE_v1_1.md`.

#### (a) The metric is not decoration

A loss needs a scalar, and turning the 26-vector $\Delta_{gg'}$ into one
requires a quadratic form. The Euclidean default fails twice. It is
dimensionally incoherent -- the parameters are 17 natural-log axes and 9 linear
ones [REPO, `HPC_main_sweep.py`: log iff both bounds positive and span $\ge 1$
decade], so a squared log-ratio is being added to a squared quantity carrying
units. And, less obviously, it concentrates the gradient on an accident: in the
conjugate Gaussian case, for each fixed $\theta^*$,

$$\mathrm{Var}(m \mid \theta^*) = C F C = C - C\Sigma_0^{-1}C, \qquad V = 2CFC, \tag{3a}$$

so along the generalised eigendirection $j$ with $c_j = 1/(1+\lambda_j)$,

$$\mathrm{Var}(m\mid\theta^*)_j = c_j(1-c_j) = \lambda_j/(1+\lambda_j)^2, \tag{3b}$$

which is an **inverted U**, maximal at $\lambda_j = 1$ and vanishing at both
extremes. Replicate disagreement is therefore largest where the data happens
to be exactly as informative as the prior -- a regime of no scientific
interest -- and an unweighted loss spends its gradient there.

**[CORRECTION carried from v0.5]** v0.4 said the metric makes divergence
"along stiff directions expensive and along sloppy directions free", justified
by the claim that a sloppy-direction posterior mean wanders over the prior
range. That is **backwards**: if the posterior equals the prior, the posterior
*mean* equals the prior mean, a constant, so replicates agree *exactly*.
Eq. (3b) is the correct statement. v0.4 also conflated the Fisher information
$F$ with the posterior precision $C^{-1} = \Sigma_0^{-1} + F$; they coincide
only in the diffuse-prior limit.

The admissibility criterion is **invariance**: for $\theta' = A\theta$ with
$A$ invertible, $T' = \Delta^\top A^\top (AVA^\top)^{-1} A\Delta = T$, so a
Mahalanobis form is invariant under any invertible linear reparameterisation
while the Euclidean one is not. This also dissolves the axis-versus-direction
worry: sloppy directions are generalised eigenvectors, not axes, and the
statistic's value does not depend on the basis.

| $M$ | admissible? | $\mathbb{E}[T_{gg'}\mid\theta^*]$ under $H_0$ | full null law | use |
|---|---|---|---|---|
| $I_{26}$ | no -- not invariant | $\mathrm{tr}(V)$, no clean form | none | -- |
| $\Sigma_0^{-1}$ | yes; diagonal and analytic for a box | $\mathrm{tr}(\Sigma_0^{-1}V)$, no clean form | none | cheap fallback, no target |
| $(2\bar C)^{-1}$ | yes | $p_{\rm eff}$ | approximate | **practical default** |
| $V^{-1}$ | yes | exactly $26$ | $\chi^2_{26}$ exactly | exact, needs $V$ |

**What the metric actually does, stated mechanically (expanded in v0.6).**
Divide the wobble of eq. (3b) by the metric weight $M_{jj} = (2c_j)^{-1}$:

$$\frac{V_{jj}}{2C_{jj}} = \frac{2c_j(1-c_j)}{2c_j} = 1 - c_j = \frac{\lambda_j}{1+\lambda_j} \in [0,1). \tag{3f}$$

The $c_j$ cancels. Each direction therefore contributes to
$\mathbb{E}[T_{gg'}]$ the **data's share of the posterior precision along that
direction**: near 1 where the data dominates the prior, near 0 where the prior
dominates, exactly $1/2$ where they balance. Two consequences, both new to
v0.6 as explicit statements:

- eq. (3f) is **monotone** in $\lambda_j$ where the numerator (3b) is not, so
  normalising by $\bar C$ does not merely restore invariance, it also converts
  a physically meaningless weighting into one where "more constrained by data"
  always means "contributes more";
- **a sloppy direction contributes neither mean nor variance.** In the
  whitened eigenbasis $T_j = (1-c_j)\chi^2_1$, so
  $\mathbb{E}[T_j] = 1-c_j$ and $\mathrm{Var}[T_j] = 2(1-c_j)^2$. The statistic
  is self-limiting with respect to uninformative axes: widening the parameter
  space cannot dilute it. That is the technical justification for (c) below
  [reasoning]. Note this holds for $M = (2\bar C)^{-1}$ and **fails** for
  $M = V^{-1}$, where every direction contributes exactly 1 to the mean and 2
  to the variance whatever its $\lambda_j$, so three extra axes there cost
  three degrees of freedom of dilution.

#### (b) The target is not zero, and the null it comes from is composite

The null hypothesis behind the target is a **conjunction**, and writing it out
matters because rejecting it does not say which conjunct failed:

$$H_0:\quad \text{(i) } \theta^*_g = \theta^*_{g'} = \theta^* \quad\wedge\quad \text{(ii) } q_\omega(\theta\mid x) = p(\theta\mid x) \text{ at the relevant } x. \tag{3e}$$

Under $H_0$, $\mathbb{E}[\Delta_{gg'}\mid\theta^*] = 0$ -- **not**
$\Delta_{gg'} = 0$. Shared $\theta^*$ centres the difference; it does not
annihilate it, because $m_g = m_g(x_g)$ and $x_g \ne x_{g'}$ by noise. With
$M = (2\bar C)^{-1}$,

$$\mathbb{E}[T_{gg'}\mid\theta^*] = \mathrm{tr}\big((2C)^{-1} \cdot 2CFC\big) = \mathrm{tr}(FC) = 26 - \mathrm{tr}(\Sigma_0^{-1}C) = p_{\rm eff}, \tag{3g}$$

**and $p_{\rm eff}$ is the sum of the generalised eigenvalues that
`information_spectrum` already reports** [REPO] -- verified numerically in the
non-diagonal correlated case. Two honest replicates of one donor *should*
disagree by that much. So the loss is two-sided about the target,

$$\mathcal{L}^{\rm real}_{\rm rep} \;=\; \big(\log T_{gg'} - \log p_{\rm eff}\big)^2, \tag{3c}$$

with $T \gg p_{\rm eff}$ meaning overconfidence on real data and
$T \ll p_{\rm eff}$ the collapse signature.

**[CORRECTION carried from v0.5]** v0.4's eq. (3) drove $T \to 0$, which trains
the posterior to be more self-consistent than its own stated uncertainty
permits -- and v0.4 implied the metric protects against the degenerate
solution. It does not: as $F \to 0$ both $V \to 0$ and $T \to 0$ under any
fixed metric. Only the non-zero target of (3c) and the NPE term do.

**[CORRECTION new in v0.6]** v0.5 justified $\mathbb{E}[T] = \mathrm{tr}(MV)$
"for $\Delta \sim \mathcal{N}(0,V)$". Gaussianity is **not** required:
$\mathbb{E}[\Delta^\top M\Delta] = \mathrm{tr}(MV) + \mu^\top M\mu$ needs only
a finite second moment, since $\Delta^\top M\Delta = \mathrm{tr}(M\Delta\Delta^\top)$
and trace commutes with expectation. Gaussianity is needed only for the
$\chi^2_{26}$ law. The practical consequence is that the $p_{\rm eff}$ target
survives the non-Gaussian case better than v0.5 implied -- what fails outside
the conjugate model is the closed form $V = 2CFC$, not the trace identity.

**What is actually evaluated.** Neither $V$ nor $F$ nor $\theta^*$ survives to
the right-hand side of (3g). The evaluated formula is
$p_{\rm eff} = 26 - \mathrm{tr}(\Sigma_0^{-1}\bar C)$, with $\Sigma_0$ analytic
from the prior box **[EVALUATED]** and $\bar C$ from one forward pass per well
**[EVALUATED]**. $V$, $F$, $\theta^*$ are **[DERIVATION ONLY]**: they justify
*why* the target is $p_{\rm eff}$ and are never computed on real data. This
distinction is new in v0.6 and was the single largest source of confusion about
the construction.

#### (c) Which axes -- all of them (changed in v0.6)

**[SUPERSEDES v0.5 S2.5(c) and the measure-then-constrain ordering.]** v0.5
required estimating $\Sigma_{\rm rep}$ under arm A1, reading off which axes
replicate, and constraining only those. That ordering is withdrawn. The
constraint acts on the **full 26-axis parameter vector**.

The reason is that $\theta^{\rm topo}$ parameterises the Weibull kernel
$p(d) = p_0\exp(-(d/d_0)^\beta)$, which is a **distribution over connections**
-- a statistical property of the preparation. Two wells of one donor plausibly
share that distribution even though their realised graphs $\mathcal{G}_g$,
$\mathcal{G}_{g'}$ certainly differ. The realisation is a latent nuisance
marginalised inside $p(x\mid\theta)$, not a parameter. v0.4's exemption
conflated the two; v0.5 identified the conflation but then hedged with a
measurement gate that is no longer thought necessary.

Three further reasons to take the whole space [reasoning]:

1. **It costs nothing if the kernel axes are sloppy.** By eq. (3f) each
   direction contributes $1-c_j \in [0,1)$ to both mean and variance, so
   including a prior-dominated direction adds essentially zero. This is P15.
2. **It removes a dependency on an unverified split.** v0.5's S9 flagged that
   the 23+3 partition is derived, not read from `label_axes.json`. Constraining
   everything makes the plan insensitive to that.
3. **It is consistent with S2.7.** The stratification already works on the
   full $\theta$; having the loss act on a subvector and the analysis on the
   whole vector was an inconsistency.

**The risk this creates, and it is a real one (new decision D17).** The bank
draws one realised graph per topology directory, i.e. essentially **one
realisation per kernel-parameter value** across 383 draws [REPO, S2.1]. So the
likelihood the flow learns does not marginalise over $\mathcal{G}$ at fixed
$\theta^{\rm topo}$: it confounds "what this kernel does" with "what this
particular graph did". Consequences:

- $F$ on the kernel axes is **overstated**, hence $p_{\rm eff}$ is overstated;
- an overstated target is *not* conservative in the loss. Eq. (3c) would then
  push $T_{gg'}$ **upward**, i.e. actively train the encoder to make same-donor
  wells disagree more on the connectivity axes than they should. As a
  diagnostic the same bias only causes under-flagging, which is benign.

**Required before A5 trains on the kernel axes:** simulate at least two
independent graph realisations per $\theta^{\rm topo}$ value on the bench and
confirm $\mathbb{E}[T_{gg'}] = p_{\rm eff}$ still holds (smoke test J13b). If
the bank cannot support it, keep all 26 axes in the **diagnostic** and mask the
three kernel axes out of the **loss** until it can -- a different exemption
from v0.4's, and for a reason that is about the bank rather than about the
biology.

**What replaces the measurement gate.** $\Sigma_{\rm rep}$ is still estimated
under arm A1 and still reported per axis; it is now a **monitored diagnostic**
rather than a gate on the loss's index set. Together with the per-direction
decomposition of $T_{gg'}$ (P16), it keeps the shared-$\theta$ assumption
falsifiable: if one or two directions dominate $T_{gg'}$ systematically across
donors, that is evidence against (3e)(i) and the exemption question reopens
with data behind it.

#### (d) Implementation of the statistic (rewritten in v0.6)

Per donor, per training step:

1. Draw $\{\theta^{(s)}_g\}_{s=1}^{S_{\rm mc}} \sim q_\omega(\theta\mid z_g)$
   and likewise for $g'$. One forward pass each.
2. $\hat m_g$, $\hat C_g$ as the sample mean and covariance of the draws.
3. **Symmetrise:** $\bar C = (\hat C_g + \hat C_{g'})/2$. *New in v0.6.* v0.5
   wrote a single $C$ without saying at which well it is evaluated; since
   $C = C(x_g)$ for a trained flow this is a genuine ambiguity, and
   symmetrising makes $T_{gg'}$ exchangeable in $(g,g')$, which the
   unsymmetrised form is not [reasoning].
4. $\Delta_{gg'} = \hat m_g - \hat m_{g'}$; solve
   $(2\bar C)\, z = \Delta_{gg'}$ by Cholesky and set
   $T_{gg'} = \Delta_{gg'}^\top z$. Never form $\bar C^{-1}$ explicitly.
5. **Finite-$S_{\rm mc}$ correction.** *New in v0.6.* Each mean is estimated
   from $S_{\rm mc}$ draws, so $\hat m = m + e$ with $\mathrm{Var}(e) = C/S_{\rm mc}$
   and $\Delta$ carries an extra covariance $2C/S_{\rm mc}$, giving

   $$\mathbb{E}[\hat T_{gg'}] = \mathbb{E}[T_{gg'}] + \mathrm{tr}\big((2C)^{-1}\cdot 2C/S_{\rm mc}\big) = \mathbb{E}[T_{gg'}] + \frac{d_\theta}{S_{\rm mc}}. \tag{3h}$$

   Subtract $26/S_{\rm mc}$. At $S_{\rm mc} = 1000$ this is $0.026$, negligible
   against $p_{\rm eff}\sim 3$; at $S_{\rm mc} = 100$ it is $0.26$, of order
   10%, which is not. Verified numerically (smoke test J15).
6. $p_{\rm eff} = 26 - \mathrm{tr}(\Sigma_0^{-1}\bar C)$, cross-checked against
   `information_spectrum`.
7. Eq. (3c), with the warm-up ramp and the stop-gradients of (e).

#### (e) Warm-up and stop-gradient, both required

Two independent reasons for a warm-up. $\bar C$ is estimated from the
posterior, so before the flow is meaningful the metric is undefined. And at
initialisation an untrained flow barely depends on its conditioner, so
$m_g \approx m_{g'}$ already: the replicate term **starts on the collapse side
of its target** and from step one opposes the NPE term over exactly the
interval where that term does its only important work. Ramp
$\lambda_{\rm rep}$ in over $t_{\rm warm}$ (the DSN's `sep_warmup_frac` is the
existing convention), and end the warm-up on a *measured* criterion --
validation NLL plateaued, or G1 passing -- not an epoch count.

**Three escape routes, three distinct mechanisms.** *Clarified in v0.6; v0.5
ran two of these together.*

| escape route | what the model would do | what blocks it |
|---|---|---|
| **Collapse** | drive $F\to 0$, so $V\to 0$ and $T_{gg'}\to 0$ under any fixed metric | the **non-zero target** in (3c), which diverges as $T\to 0^+$ |
| **Inflation** | widen $\bar C$ so the metric shrinks and $T_{gg'}$ falls | the **NPE likelihood term**: an inflated $\bar C$ scores badly on held-out $(\theta,x)$ |
| **Desensitisation** | make $q_\omega$ less sensitive to $z$, so $m_g\approx m_{g'}$ whatever the input -- precisely a G1 failure | the **stop-gradient on $\omega$** in the replicate term: compute $\mathbb{E}[\theta\mid z_g]$ with the current flow, backpropagate only through $z_g$ into $\psi$ |

Also stop-gradient through $\bar C$, recomputed per epoch and held fixed within
it. A weak self-limiting effect exists but is not a guarantee: widening
$\bar C$ raises $p_{\rm eff}$ as well as lowering $T_{gg'}$, since both come
from the same matrix [reasoning].

#### (f) Multimodality breaks the summary, not the metric

If $p(\theta\mid x_g)$ is multimodal the posterior *mean* sits between modes,
in a region of low mass, and two replicates may land on different modes -- an
enormous $\Delta$ reflecting degeneracy, not miscalibration. No metric repairs
this. The principled fix is to stop comparing point estimates: the wells are
conditionally independent given $\theta$, so

$$p(\theta \mid x_g, x_{g'}) \;\propto\; p(\theta\mid x_g)\,p(\theta\mid x_{g'})\,/\,p(\theta), \tag{3d}$$

whose normalising constant is a **Bayes factor for "same $\theta$" against
"different $\theta$"**, correct under multimodality. A cheaper intermediate:
draw $\theta \sim p(\theta\mid x_g)$, evaluate its log-density under
$p(\theta\mid x_{g'})$, symmetrise. Whether posteriors here are actually
multimodal is unchecked [reasoning].

#### (g) Estimating $M$ with the data available

$V$ is $26\times26$: **351** free parameters, and a sample covariance from
$N_{\rm pair}$ pairs has rank at most $N_{\rm pair}-1$, so with fewer than
**27** pairs it is singular and cannot be inverted. (v0.5 said 276 and 24, for
23 axes.) Whether the cohort supplies that many is D12, open. Preference
order: estimate $V$ on the bench where replicate wells with a known shared
$\theta^*$ can be simulated in any number; else $(2\bar C)^{-1}$, which needs
no pairs; else a Ledoit-Wolf-shrunk sample $V$. A **diagonal** metric estimated
reliably may beat a full metric estimated badly at $N_{\rm pair}\sim 10$.

*Plain.* Instead of teaching the encoder the diagnosis, teach it that the same
preparation must give the same answer twice -- and judge "the same" against the
error bars the posterior itself quotes, not against zero, because two honest
instruments should not agree perfectly. Agreeing too well is as much a symptom
as disagreeing. And the right amount of disagreement turns out to be a count:
each of the 26 directions is owned partly by the prior and partly by the data,
and $p_{\rm eff}$ adds up the data's shares.

**Two cautions retained.**

- **Do not declare all replicate variation nuisance.** Wells differ in plating
  density; batches differ in effective maturation, which acts through synaptic
  strength. Only the measurement layer is pure nuisance. A blanket
  *embedding-space* invariance term would train the encoder to be blind to
  parameters it is meant to infer; eq. (3) lives in parameter space for exactly
  this reason -- two wells may legitimately produce different $z$ while
  implying the same $\theta$ [reasoning].
- **Every invariance objective has the constant map as a global minimum.**
  Formally: for any loss
  $\mathcal{L}_{\rm inv}(f) = \mathbb{E}_{(x,x')\sim\mathcal{E}}[\rho(f(x),f(x'))]$
  with $\rho \ge 0$ and $\rho(a,a) = 0$ for all $a$, the constant map
  $f \equiv c$ gives $\rho = 0$ pointwise and attains the infimum -- for any
  $\rho$, any equivalence set $\mathcal{E}$, and any function class containing
  the constants. No property of the data or the architecture is used, so this
  is not a local trap a better schedule avoids; it is a property of the
  objective as written. Hence $\lambda_{\rm rep}$ must be searched, never set.
  Under (3c) the point is sharper still: the two-sided target sends
  $\mathcal{L}\to+\infty$ as $T\to 0^+$, so the former global minimum becomes a
  divergence rather than merely a penalised point.

**Equation (3c) may be worth more as a diagnostic than as a loss.** Every gate
requires $\theta^*$, so **G1, G2 and G3 can only run on simulated data**
[REPO; `GATES_v1.md` S3.2] -- and so can SBC, expected coverage, TARP and
L-C2ST [KB]. The pipeline currently has no calibration check of any kind on the
real cohort. Replicate consistency needs no $\theta^*$, and normalised by
$p_{\rm eff}$ it is a $\chi^2$-like statistic with a reference value. Report it
for every arm whether or not it is trained on (D13). Note what this implies for
(3e): conjunct (ii) is **assumed** on real data, verified only on the bench,
and model misspecification is exactly the failure of that transfer.

**The label becomes a referee, not a teacher.** Under A1 and A5 the diagnosis
enters no training objective, which is what makes $I(c; z)$ on held-out
cultures a *measurement* rather than a tautology: if a $\theta$-supervised
summary separates the conditions as well as a label-supervised one, the
simulator's mechanisms mediate the pathology; if not, the pathology acts
through something the simulator lacks -- open point O2, restated as a number.
A2 remains in the study precisely as the arm that tests whether removing the
label costs anything.

**Alignment terms are not pursued.** An MMD between the simulated and real
summary clouds would optimise the misspecification gate's own statistic and
so destroy it as a test; it is excluded and no arm uses it (S11).

### 2.6 Separating nuisance from mechanism: the transversality test

Nuisance and within-class mechanism both produce within-class variation in
$x$, and from single windows they are confounded. Three sources of structure
break it.

**Lever 1 -- the simulator's support.** Mechanism variation moves along the
prior-predictive manifold; nuisance moves off it. Already instrumented as the
gate and witness. This conflates nuisance with misspecification, which is the
correct conflation for inference: anything the simulator cannot produce cannot
be attributed to $\theta$.

**Lever 2 -- the nesting.** batch $\to$ donor $\to$ culture (well) $\to$
subregion $\to$ window. Subregions of one well share every nuisance factor but
differ in local graph realisation $\mathcal{G}$; windows of one subregion share
everything. Note the v0.6 reading: the realisation differs, the *kernel* does
not, and only the kernel is $\theta$.

**Lever 3 -- simulate the nuisance.** Fix one $\theta$, vary only
`ProbeConfig`, electrode positions, detection threshold, active-electrode
count and gain, push through the entire pipeline, and measure the dispersion
of the resulting posterior means. **That is the nuisance floor**: how much
apparent mechanistic heterogeneity pure nuisance manufactures. It is the exact
analogue of measuring $\delta_{\min}$ from a shuffled control rather than
guessing it, and it needs no new biology. **Extended in v0.6:** the same
machinery, with $\mathcal{G}$ redrawn at fixed $\theta$ and everything else
held, measures the *realisation floor* -- the dispersion attributable to graph
sampling alone. That is the quantity D17 turns on.

**The separability condition.** Build $J_\theta = [\partial z/\partial\theta_1,
\dots]$ and $J_\nu$ by finite differences on the simulator. Separability is
transversality of their column spans, measured by principal angles. Per
nuisance direction,

$$a_m = \frac{\big\lVert P_{\mathrm{col}(J_\theta)}\,\partial z/\partial\nu_m \big\rVert}{\big\lVert \partial z/\partial\nu_m \big\rVert}, \qquad \delta\theta_m = J_\theta^{+}\,\frac{\partial z}{\partial \nu_m}. \tag{4}$$

$\delta\theta_m$ is directly readable: "a 10% gain drift is read by the
pipeline as 0.3 log-units of synaptic strength". Computable today from
simulator plus encoder, no new data. And $a_m$ predicts whether
$\mathcal{L}^{\rm real}_{\rm rep}$ will fight $\mathcal{L}^{\rm sim}_{\rm NPE}$:
the replicate term suppresses directions the likelihood term does not use
**iff** the two subspaces are transverse. That is the sense in which the
nuisance objective is compatible with the NPE loss where the phenotype
objective is not [reasoning].

Because $J_\theta^{+}$ amplifies along poorly-constrained directions, aliasing
is worst where the information spectrum is smallest: **the parameters least
constrained by the data are the most contaminated by nuisance** (P11).

### 2.7 The culture as inferential unit, and mechanistic stratification

**Partial pooling.** Windows of a subregion share $\theta$; subregions of a
culture share $\theta$ and differ in their realised graph $\mathcal{G}$. The
current pipeline computes *per-window* posteriors; the correct object is
$p(\theta_g \mid \text{all windows of culture } g)$. For conditionally
independent windows given $\theta$, the factorised-NPE composition
$p(\theta\mid x_{1:N}) \propto p(\theta)^{1-N}\prod_i p(\theta\mid x_i)$
applies; NLE/NRE combine i.i.d. observations by multiplying likelihood terms
[KB, Practical Guide; Compositional SBI]. Caveat: windows of one subregion are
exchangeable but not independent (shared realisation $\mathcal{G}$), so the
naive composition overstates and the correction must be validated on the bench
before use on real data.

**A free discriminator (P10).** Under a collapsed encoder all 54 windows of a
culture give the same $z$, so aggregating them adds exactly nothing. Under a
$\theta$-informative encoder the gain compounds. Plot $\hat\Delta$ against the
number of windows aggregated per culture: **flat means collapse, growing means
information.** No ground truth, no labels, works on the real bank.

**Stratification.** With $m_g = \mathbb{E}[\theta\mid x_g]$ over all 26 axes,
form $\Sigma_{\rm rep}$ (covariance of $m_g - m_{g'}$ over same-donor pairs --
the empirical nuisance floor, covering differentiation, plating, recording,
detection, encoding and inference), $\Sigma_{\rm pat}$ (covariance of donor
means within one diagnosis), $\Sigma_{\rm dx}$ (covariance of diagnosis
means). Then solve

$$\Sigma_{\rm pat}\,v_j \;=\; \mu_j\,\Sigma_{\rm rep}\,v_j. \tag{5}$$

Directions with $\mu_j \gg 1$ are those along which donors differ **more
than replicates of one donor do**: mechanistic heterogeneity above the measured
noise floor. $\mu_j \approx 1$ is indistinguishable from replication. This
is discriminant analysis whitened by the empirical replicate covariance, which
is the right noise model because it includes the whole measurement chain.

Project donor means onto the leading $v_j$ and ask where the diagnosis sits.
Three outcomes, all reportable: the diagnosis aligns with the leading direction
(the label is a good one-dimensional summary of mechanism); it is one of
several $\mu_j \gg 1$ directions (within-diagnosis heterogeneity exists and
the collapsed encoder discarded it); it cuts across them (the label is
misleading). **This analysis is only interpretable if the encoder was not
trained on the label** -- which is the strongest single argument for A1/A5 over
A0/A2 [reasoning].

**$\Sigma_{\rm rep}$ is a monitor, not a gate (changed in v0.6).** v0.5 made
the per-axis diagonal of $\Sigma_{\rm rep}$ decide which axes eq. (3)
constrains, and therefore required it to be measured under A1 before A5 could
be configured. With the full-space constraint of S2.5c that dependency is
gone: A5 can be configured immediately. $\Sigma_{\rm rep}$ is still estimated
under A1 -- estimating it under A5 would be circular -- and still reported per
axis, now as the falsifier for the shared-$\theta$ assumption (P16) rather than
as a switch.

**Closing the loop.** The leading eigenvectors of $\Sigma_{\rm rep}$ are an
empirical estimate of the nuisance subspace *in parameter coordinates* -- the
nuisance directions are discovered, not named. Compare them against the
simulated $\delta\theta_m$ of eq. (4), and against the realisation floor of
S2.6 Lever 3. Alignment with a nuisance direction identifies the factor (gain,
threshold, electrode yield); alignment with the realisation floor says the
disagreement is graph sampling and bears directly on D17; non-alignment with
either says the replicate variation is biological and the "same donor"
assumption is doing less work than assumed.

Grounding: according to PubMed, Van Lent et al. report that in CMT type 2,
genetic and clinical heterogeneity under one diagnostic label complicates
diagnosis and has inhibited therapy development, and that differentiating the
same patient iPSC lines into sensory neurons produced cellular phenotypes only
in subtypes with sensory involvement -- one label, subtype-dependent phenotype
[PubMed abstract only, [DOI](https://doi.org/10.1093/brain/awab226); no numbers
used]. Pavlinek et al. report MEA variability linked to donor line and batch
effects requiring explicit design treatment [PubMed abstract only,
[DOI](https://doi.org/10.1016/j.crmeth.2026.101371)]. Ronchi et al. report that
the ability to detect drug effects and the observed culture-to-culture
variability both depend on the number of recording electrodes [PubMed abstract
only, [DOI](https://doi.org/10.1002/adbi.202000223)] -- relevant to
$n_e = 9$.

### 2.8 Drug response as external validation (conditional on data existing)

Treat a compound as an intervention on the mechanism, $\theta \mapsto \theta
\oplus \delta_d$. The predicted response of culture $g$ is a posterior
predictive of that intervention,

$$\Delta y_g \;=\; \mathbb{E}_{\theta \sim p(\theta \mid x_g)}\big[\, f(\theta \oplus \delta_d) - f(\theta) \,\big]. \tag{6}$$

Two properties do the work: cultures differing only in $\nu$ have identical
predicted response, because $\nu$ acts on the observation and not the
dynamics; and because $f$ is nonlinear, $\partial f/\partial\theta \cdot
\delta_d$ varies with $\theta_g$, so responders and non-responders follow from
the model with a mechanism attached.

**Validation criterion (P12).** Does the *pre-treatment* posterior
$p(\theta_g\mid x_g)$ predict the *post-treatment* observed change, out of
sample, better than (i) the phenotype label, (ii) the raw summary $z_g$,
(iii) baseline burst statistics? Controls: dose-response, replicate wells of
the same culture, cross-validation held out by batch -- differential response
can also arise from dose delivery and cell density.

It also adjudicates the heterogeneity question independently: **if
within-class variation were pure nuisance, responder status within a class
would be unpredictable from $\theta$.**

Precedent, full text read: according to PubMed, Hommersom et al. used a
biophysical MEA network model with SBI for parameter estimation, chosen
explicitly because the same activity can arise from very different parameter
combinations, and identified the parameter combinations able to reproduce the
mutant phenotype. Their pharmacology is the phenomenon in miniature: 4-AP
restored time to burst peak but not ISI CoV within network burst, NS309 did the
converse, the combination partly restored both, and 4-AP's effect on AHP
amplitude ran in opposite directions in control versus mutant networks -- same
compound, different mechanistic state, opposite effect. They state their own
limitation as one genetic background and call for multiple patient-derived
models [PubMed full text, [DOI](https://doi.org/10.1093/brain/awae330)]. That
gap is this cohort.

---

## 3. Scoring and diagnostics

| quantity | tool | note |
|---|---|---|
| $L$, $L_0$, $\hat\Delta$, Jensen check | `npe_tune_score.py` | floor by Monte Carlo on the bench (mixture prior) |
| per-axis $\hat\Delta^{(k)}$, $\hat\Delta_{S\mid c}$, contraction, information spectrum | `npe_diagnostics.py` | $\hat\Delta_{S\mid c}$ is new and is the decisive quantity for S2.3 |
| G1 / G2 / G3 | `npe_tune_gates.py` | unchanged |
| C2ST between arms | `npe_tune_benchmark.c2st` | no exact posterior for the burst model |
| paired CI on $D$, all four schemes | `bootstrap_paired.paired_bootstrap` / `compare_schemes` | primary `within="all"`; per axis group as well as aggregate (S2.4a) |
| $\rho(d)$, $n_0$, $\mathrm{DEFF}$ | `bootstrap_paired.intraclass_correlation` | the number that justifies whichever scheme is reported |
| tail share of $|d_i|$ | `bootstrap_paired.tail_share` | precondition: if large, no interval for the mean is trustworthy |
| ARI, silhouette, $r_{\rm eff}$ (per arm) | DSN `metrics.py` | logged, never optimised |
| $\rho_{\rm grad}$ | new, small | two backward passes on one probe batch per epoch |
| replicate consistency $T_{gg'}$, eqs. (3), (3c) | **`replicate_statistic.py` -- delivered and verified (v0.6)** | reported for **every** arm whether trained on or not; against the $p_{\rm eff}$ target, not zero; all 26 axes |
| **per-direction decomposition of $T_{gg'}$** | `replicate_statistic.py` | terms normalised by $1-c_j$ so directions are comparable; the falsifier for the shared-$\theta$ assumption (P16) |
| $p_{\rm eff}$ | `npe_diagnostics.information_spectrum`, cross-checked by `replicate_statistic.p_eff_from_trace` | the target value for $T$; two independent routes must agree (smoke test J13) |
| **$p_{\rm eff}(26)$ vs $p_{\rm eff}(23)$** | `replicate_statistic.p_eff_from_trace` on both index sets | P15; costs one call, no training |
| $\lVert z_g - z_{g'}\rVert$ distribution over same-donor pairs | new, small | compared across A0(real), A2 and A5: three teachers, three routes to culture-invariance. A0/A2 get it implicitly through `cross_culture` positives, A5 explicitly through the posterior (P13) |
| aliasing $a_m$, $\delta\theta_m$, eq. (4) | new | finite differences on the simulator |
| nuisance floor; **realisation floor** | new | Levers 3 of S2.6; the second is new in v0.6 and gates D17 |
| $\mu_j$ spectrum, eq. (5) | new | the stratification figure |
| support overlap of real $z$ in the simulated cloud | new, small | an O1(c) diagnostic: is the flow evaluated where it has training mass |
| activation statistics, sim vs real inputs | new, small | detects the off-support degeneracy branch of Stage 3b |
| gate MMD, witness | `npe_misspec.py` | unchanged; a misspecification test, **not** reframed as anything else |

---

## 4. The bench

### 4.0 Two arms, a gap knob, and a nuisance latent

A bench where every window carries both $\theta$ and $c$ tests an easier
problem than the one deployed. Two arms from one generator:

| arm | drawn from | $\theta = \phi$ | $c$ | role |
|---|---|---|---|---|
| $\mathcal{S}$ | base generator $p_0$ | **available** | withheld (except A0s/A2s) | stands for the ANN campaign |
| $\mathcal{R}$ | perturbed generator $p_\pi$ | **recorded but withheld** | available | stands for the MEA cohort |

$\pi$ sets the simulation gap; at $\pi = 0$ the arms share a generator and
differ only in which label is available, isolating the availability asymmetry
from any distribution gap. Perturbations follow the severity design of Schmitt
et al. [KB]: (a) a free-axis range shifted partly outside the simulated prior
box; (b) heavy-tailed burst durations; (c) a slow background drift absent from
$p_0$; (d) a contaminated fraction of windows.

**A nuisance latent $\nu$**, applied to both arms as an observation-level
transformation *outside* $\theta$ -- gain, baseline offset, detection-threshold
shift, electrode dropout, slow drift -- with a donor/well/batch nesting
mirroring the cohort's, so that eq. (3) and eq. (5) can be exercised where the
ground truth is known. And $\tau_{\rm ov} > 0$ is kept deliberately, so that
within-class mechanistic spread exists by construction: the bench must be able
to *fail* the user's hypothesis, not just illustrate it.

**New in v0.6 -- a realisation latent.** The bench generator must expose a
per-well draw that is *inside* the likelihood but *outside* $\theta$, standing
for the graph $\mathcal{G}$: two wells of one bench donor share $\theta$
including the "kernel" axes, and draw independent realisations. Without it the
bench cannot exercise S2.5c at all, and J13b (the D17 check) cannot run.

**Primary endpoint:** the pseudo-real held-out NLL, because $\phi$ is known on
$\mathcal{R}$ but withheld from every training loss. On the real cohort this
measurement is impossible; that is why the bench earns its cost.

### 4.1 Generative model

$$c_g \sim \mathrm{Unif}\{0,\dots,C-1\}; \quad \phi_{g,k}\mid c_g \sim \mathcal{TN}\big(m_{c_g,k}, \tau_{\rm ov}^2; (0,1)\big),\ k \in S; \quad \phi_{g,k}\sim U(0,1),\ k \in F, \tag{7}$$

$$\nu_g \sim p(\nu), \quad \mathcal{G}_{g,w} \sim p(\mathcal{G}\mid\phi_g)\ \text{independently per well } w, \qquad x_{g,j}\mid \phi_g, \nu_g, \mathcal{G}_{g,w} \sim T_{\nu_g}\big[\text{burst model}(\phi_g, \mathcal{G}_{g,w})\big],\ j = 1..J. \tag{8}$$

Two deviations from `sample_latents`, confined to the new wrapper (the DSN
module is not edited): a **truncated** rather than clipped normal, because
clipping puts atoms on the box boundary which map to $\pm\infty$ under
`transform_to_unconstrained`; and the explicit statement that the NPE prior
$p(\phi) = C^{-1}\sum_c p(\phi\mid c)$ is a mixture, so $L_0$ needs Monte
Carlo. **Third, new in v0.6:** the per-well realisation draw in (8), which
makes the bench likelihood marginalise over $\mathcal{G}$ the way the real one
must.

### 4.2 Modules

```
LatentSBISpec        : LatentSpec + n_windows_per_trace, T_win, nuisance_spec,
                       gap_spec, realisation_spec, seed
sample_prior         -> (c, phi)                  # eq. (7)
sample_nuisance      -> nu, with donor/well/batch nesting
sample_realisation   -> G, one draw per well at fixed phi   # new in v0.6
simulate_windows     -> x (J, W) float32          # eq. (8)
prior_log_prob       -> log p(phi)                # mixture density, for the MC floor
class_posterior      -> p(c | phi)                # for p(c | x) from posterior samples
```

Signal synthesis delegates to `latent_burst_generator` and
`generate_burst_data`, imported through `DSN_MAIN_DIR` as
`Sbi-extractor/dsn_frozen.py` already does [REPO]. No spike model is
reimplemented.

### 4.3 Bank format

`npz` shards (a window is a $(W,)$ array; parquet columns per sample are the
wrong shape): `x`, `theta`, `cls`, `nu`, `realisation_id`, `donor`, `well`,
`batch`, `subregion`, `window_idx`, plus a JSON sidecar carrying `param_names`,
`bounds_theta`, `coord`, `fs`, `w_size`, `T_win`, `scale_convention`,
`latent_spec`, `nuisance_spec`, `gap_spec`, `realisation_spec`,
`generator_sha256`, `theta_withheld` (true on $\mathcal{R}$), and `split_hash`
once frozen. A `Bank` adapter presents `x` where the stack expects `z`, so
`npe_tune_data.make_split / check_split / shape_report` run unmodified.

### 4.4 Sizes

Iteration bank $G = 4000$ traces $\times$ $J = 8$ windows of $T_{\rm win} = 60$
s at $f_s = 50$ Hz ($W = 3000$), $N = 100$ neurons, $C = 3$,
$\tau_{\rm ov} = 0.10$: 32 000 windows. Parity bank at $T_{\rm win} = 180$ s.
Generation is embarrassingly parallel by PBS array; cost to be measured, not
assumed. The report split is touched once, at the end. **v0.6:** at least two
realisations per $\theta$ for a designated subset, sized so that J13b has
Monte Carlo error well below the $p_{\rm eff}$ difference it must resolve.

### 4.5 Controls

Shuffled-pairs control (gives $\delta_{\min}$ for G1); a hand-crafted summary
reference `A_ref` (NPE on fixed burst statistics from the ANN repo's
`burst_metrics.py` / `network_burst_detector.py`) -- if A1 cannot beat it, the
encoder rather than the objective is the problem; and the oracle-dimension
check, $r_{\rm eff}$ of the learned $z$ against the generator's $n = 6$.

---

## 5. Search space and the GP campaigns

### 5.1 One space, one objective, one ledger

$\mathcal{X}$ unions the DSN's searched axes and the NPE stack's, plus the
loss weights. Blocks: encoder (`depth_exponent`, `width_multiplier`,
`block_family`, `embedding_size`, `head_fusion`, `dropout`); DSN loss
(`dsn_on`, `log10_lambda_dsn`, `loss_type`, `mining_strategy`, `margin`,
`angular_alpha_deg`, `lambda_sep`); **replicate term (`rep_on`,
`log10_lambda_rep`, `warmup_frac_rep`, `n_posterior_draws`)**; flow
(`hidden_features`, `num_transforms`); optimiser (`lr`, `one_minus_beta1`,
`weight_decay`, `batch_size_npe`). Fixed: `stem_width`, `group_width`,
GroupNorm settings, kernels and strides, `head_pool_ops`, `strict_semihard`,
`sep_warmup_frac`, `num_bins`, `one_minus_beta2`.

**`n_posterior_draws` is new in v0.6** ($S_{\rm mc}$ of S2.5d). It is a
searched axis rather than a constant because it trades compute against the
$d_\theta/S_{\rm mc}$ inflation of eq. (3h); the correction is applied
regardless, so the axis controls variance, not bias. Lower bound at least
$4 d_\theta \approx 100$, or $\hat C_g$ is rank-deficient and the Cholesky
solve fails.

**What the weights absorb.** $\lambda_{\rm dsn}$ carries the loss-scale ratio,
a row-count ratio (29 616 simulated against 1 890 real [KB]) and an
effective-sample-size ratio far worse (383 topology draws against 35 cultures,
and fewer distinct donors -- D12). Range $\log_{10}\lambda_{\rm dsn} \in
[-3,1]$, with each trial recording the realised gradient-norm ratio in the
ledger. Same treatment for $\lambda_{\rm rep}$. Note that eq. (3c) is now
dimensionless and $O(1)$ near its target, so $\lambda_{\rm rep}$ carries a
smaller scale burden than v0.4's squared-distance form did.

**Epoch semantics and the small-cohort problem.** One epoch is one pass over
the simulated bank; at $B_{\rm sim} = 512$ that is about 58 steps, so the real
cohort is revisited roughly 58 times per epoch and can be overfitted long
before the NPE term converges. Mitigations: hold out cultures for a monitored
DSN-validation ARI and for a monitored replicate-consistency value; keep early
stopping on the NPE validation score; and use the **same** held-out cultures
for any later gate or witness run, or the circularity of O5 returns through a
new door [KB].

Mechanics reused from `npe_tune_search.py`: stateless `skopt.Optimizer`
rebuilt from the ledger each round, constant-liar batching, escalation verdict
against measured $\sigma_{\rm seed}$. `KNOB_ORDER` becomes `JOINT_KNOB_ORDER`;
the legality projection and activity mask are imported from
`condition_space.py` and extended with the `dsn_on` / `rep_on` clamps so that
points differing only in inactive coordinates build byte-identical configs.

**Objective**: held-out NLL. **Gates gate, NLL ranks** -- ARI, silhouette and
replicate consistency are logged and never optimised, for the reason the stack
already records: a clustering or calibration statistic as objective rewards
label-only or prior-like codes.

### 5.2 Campaigns

S-A0 (the DSN's own `joint_conditions` campaign, unchanged, then the existing
`npe_tune.py` on frozen $z$), S-A1 (`dsn_on = rep_on = 0`), S-A2 (`dsn_on`
free), S-A5 (`rep_on` free, `dsn_on = 0`), at equal budget on the same frozen
split. Since A2 and A5 nest A1, the informative outputs are where the winners
sit in the weight axes and the GP surrogate's partial dependence on
`dsn_on` / `rep_on`, which is the cleanest statement of "does this term help
at the optimum".

---

## 6. Stages

Every stage ends with the `hpc-git-delivery` gates (fetch the real file,
`py_compile`, `args.*` resolution, byte safety, behavioural smoke test on a
fixture, patch-reproduces-tested-bytes) and ships as a git patch or tarball via
`present_files`, never pasted text. Cluster-side: verify, dry-run, one probe,
then one array. Environment `sbi_env` [KB, `HPC_PATHS.md` S7];
`pytorch_metric_learning` added to `check_env.py`. Branch
`feat/joint-dsn-npe` off `feat/misspec-gate`; the DSN and ANN repos are
imported, not edited, until Stage 6.

### Stage 0 -- Contract and decisions

`check_env.py` additions: `pytorch_metric_learning` importable; DSN repo
importable through `DSN_MAIN_DIR` (use the `dsn_main` symlink, the path has a
space); `posterior_nn(model="zuko_nsf", embedding_net=OneDCNNBackbone(cfg),
z_score_x="none")` builds on a $(B,W)$ batch and `.loss(theta, condition=x)`
backpropagates into the backbone (assert non-zero backbone gradient); a
`DirectPosterior` builds from the trained estimator plus box prior and its
`.log_prob` / `.sample` work on raw windows; `EnsemblePosterior.log_prob` is
still the arithmetic mixture; **no `BatchNorm*` module in the built backbone**;
and read $f_s^{\rm IFR}$, `w_size`, `window_s`, hence $W$, **from the r2
checkpoint's embedded `ExperimentConfig`** via `dsn_frozen.load_frozen_dsn`.
That last is not bookkeeping: `config_mea_joint_full.json` is a template with
placeholder paths stating `cohort.w_size = 0.02` (50 Hz) while the extracted
archives are 100 Hz with $K = 120\,000$ over 1200 s [KB, S3]. At
$T_{\rm win} = 180$ s that is $W = 9\,000$ or $18\,000$ -- a factor of two in
every convolution in Stage 6. The checkpoint is authoritative.

**Added in v0.6:** read `artifacts/label_axes.json` and assert
$d_\theta = 26$ with the three kernel axes present by name, since every
$p_{\rm eff}$ number in this plan assumes it and S2.5c now makes the whole
vector load-bearing rather than just the 23-axis subvector.

Decide D1-D4, D12-D14 and **D17** before Stage 1 is written.

### Stage 1 -- Simulator wrapper, nuisance, realisation, bank builder, floor

Deliverables: `latent_sbi_simulator.py`, `latent_nuisance.py`,
`latent_realisation.py` (**new in v0.6**), `latent_gap.py`, `latent_bank.py`,
`build_latent_bank.py` (one shard per call, PBS-array friendly),
`jobs/build_latent_bank.pbs`, `smoke_test_latent_sbi.py`.

**Already delivered and verified: `bootstrap_paired.py` +
`smoke_test_bootstrap_paired.py`** (S2.4a). Pure numpy, no torch, no bank
dependency, so it ships ahead of everything else and can be exercised on any
stored pair of per-row NLL arrays. Verification passed at hand-over:
`py_compile`, pure ASCII, LF-only, ten fast tests plus a coverage test, each
validating against a truth known analytically from a generative model rather
than against another estimator. Tests cover ICC recovery, SE matching the
analytic SD per scheme, the $1/\sqrt{\mathrm{DEFF}}$ error law for `none`,
`mean` dominating `one`, `all` $=$ `mean` under balance and their divergence
under size-effect confounding, interval collapse on constant $d$, the guards,
determinism, and the tail diagnostic. Run
`python3 smoke_test_bootstrap_paired.py` (fast, ~10 s) or `--full` for
coverage.

**Also delivered and verified in v0.6: `replicate_statistic.py` +
`smoke_test_replicate_statistic.py`.** Pure numpy/scipy, no torch, no bank
dependency, so it too ships ahead of the training loop and can be exercised on
any stored pair of posterior sample arrays. It contains estimator logic only --
no I/O, no flow evaluation, no plotting -- and provides `posterior_moments`,
`replicate_statistic` (Cholesky solve on $2\bar C$, optional
$d_\theta/S_{\rm mc}$ correction), `mahalanobis_statistic`, `p_eff_from_trace`,
`p_eff_from_spectrum`, `generalised_spectrum`, `box_prior_covariance`,
`replicate_loss`, and the bench-only helpers
`conjugate_posterior_covariance`, `simulate_replicate_pairs`,
`sampling_covariance_V`. Fourteen tests pass: the two routes to $p_{\rm eff}$
agree on a non-diagonal correlated model; $\mathbb{E}[T] = p_{\rm eff}$ under
$M = (2C)^{-1}$; $\mathbb{E}[T] = d$, $\mathrm{Var}[T] = 2d$ and
$T \sim \chi^2_d$ by Kolmogorov-Smirnov under $M = V^{-1}$; invariance under
$\theta\mapsto A\theta$ with the Euclidean form failing the same test; the
$d/S_{\rm mc}$ inflation and its correction; the sloppy limit
($p_{\rm eff}\to 0$ and machine-level $\Delta$) and the stiff limit
($p_{\rm eff}\to d$); the two-sided loss shape; and an end-to-end run from
posterior samples. Verification at hand-over: `py_compile`, pure ASCII,
LF-only. Run `python3 smoke_test_replicate_statistic.py` (fast; exits non-zero
on failure, so it drops into the cluster-side verification block unchanged).

**What the scorer must now persist**, which it currently does not: the
**per-row** log-density array for each arm on the frozen split, plus the group
label per row, saved beside the trial's ledger entry. Without them $D$ cannot
be paired and no interval can be built after the fact. This is a small change
to `npe_tune_score.py`'s output, and it is a prerequisite for Stage 3.
**Added in v0.6:** persist $\hat m_g$ and $\hat C_g$ per culture as well, since
$T_{gg'}$, $p_{\rm eff}$ and the per-direction decomposition are all
reconstructible from them after the fact and none of them is otherwise
recoverable without re-running the flow.

Smoke tests: (S1) `sample_prior` reproduces `sample_latents` when no rejection
occurs and never returns a boundary coordinate; (S2) `prior_log_prob`
integrates to 1 on a grid for $n = 2$; (S3) `class_posterior` rows sum to 1 and
match a hand-computed Bayes rule; (S4) `simulate_windows` matches
`LatentBurstProvider.__call__` bit-for-bit at $\nu = 0$ and a fixed
realisation seed on the same seed, and the windowing matches
`sim_observable.window_trace`; (S5) shard round-trip, contract digest stable,
`check_split` passes, `theta_withheld` set on $\mathcal{R}$; (S6) MC floor
converges (two seeds within 3 SE); (S7) at $\pi = 0$ the two arms are
indistinguishable by the numpy permutation MMD on raw window statistics;
(S8) at $\pi > 0$ they are, monotonically; (S9) the nuisance transformation is
invertible in distribution; **(S10, new) two wells of one bench donor share
$\theta$ exactly and carry different `realisation_id`, and the marginal law of
$x$ at fixed $\theta$ is invariant to the realisation seed** -- the property
S2.5c needs and the bank currently lacks.

### Stage 2 -- Joint model and training loop

Deliverables: `joint_model.py` (`JointDSNNPE`: backbone + flow, checkpoint
embedding a DSN-format encoder so `dsn_frozen.load_frozen_dsn` can still load
the encoder alone), `joint_batches.py` (three-stream builder: i.i.d. simulated,
class-balanced metric, donor-paired replicate), `joint_losses.py` (eq. (2) and
eq. (3c), **calling `replicate_statistic.py` for the statistic and the target
rather than reimplementing them**), `joint_train.py` (explicit loop: AdamW,
grad clip, early stopping on validation $\mathcal{L}^{\rm sim}_{\rm NPE}$,
convergence capture, per-epoch log of every quantity in S3),
`smoke_test_joint.py`.

Why an explicit loop: sbi's loop cannot take a second or third loss with
labels, and weight decay and schedules are unreachable through it [REPO]. The
flow is still built by `posterior_nn`, so posterior objects, gates and
ensembles are untouched.

The torch side of `joint_losses.py` must mirror the numpy reference
bit-for-bit in value (smoke test J16), not merely in spirit: the reference is
the specification.

Smoke tests: (J1) at $\lambda_{\rm dsn} = \lambda_{\rm rep} = 0$ with a frozen
encoder the loop reproduces `npe_model.train_single`'s validation NLL on the
GMM benchmark within seed noise; (J2) with a trainable encoder the validation
NLL falls below the shuffled control within 3 epochs on a tiny bank; (J3) at
large $\lambda_{\rm dsn}$ with the flow frozen, the loop reproduces DSN
`train()`'s first-epoch $\ell_{\rm DSN}$ on the same batch and seed; (J4)
surrogate rows never reach the NPE term; (J5) one host sync per epoch on the
training path; (J6) checkpoint round-trip, encoder reloadable by
`load_frozen_dsn`, flow into a `DirectPosterior`, sha256 recorded; (J7)
$\rho_{\rm grad}$ matches a finite-difference cosine on a 2-parameter toy;
(J8) the metric stream contributes gradient to $\psi$ and **exactly none** to
$\omega$; (J9) the streams interact only through $\psi$ -- re-ordering or
duplicating the metric batch leaves the simulated-term value bit-identical
(fork a `torch.Generator` per stream, or run at `dropout = 0`, or this fails
for reasons unrelated to what it tests); (J10) $T_{gg'}$ is exactly zero when
both members of a pair are the same window, and is invariant to swapping the
pair order -- **the second clause is now guaranteed by construction through the
symmetrised $\bar C$, so J10 tests the implementation, not the definition**;
(J11) the degenerate check -- a constant encoder gives $T = 0$, which (3c)
sends to $+\infty$ and the NPE term makes catastrophic; (J12) **the metric is
invariant**: rescaling any axis of $\theta$ (or switching it between log and
linear) leaves $T$ unchanged to floating tolerance, while the Euclidean form
does not -- the behavioural test of S2.5a; (J13) on bench pairs with a known
shared $\theta^*$ and a well-calibrated posterior, $\mathbb{E}[T]$ matches
$p_{\rm eff}$ from `information_spectrum` within Monte Carlo error, **on all 26
axes**; (J13b, **new**) the same with two *independent graph realisations* per
well, which is the D17 check: if $\mathbb{E}[T]$ matches on 23 axes but not on
26, the kernel axes are confounded with their realisation and must be masked
out of the loss; (J14) **stop-gradient holds**: the replicate term contributes
zero gradient to $\omega$ and to $\bar C$; (J15, **new**) the
$d_\theta/S_{\rm mc}$ correction removes the finite-draw inflation, checked by
varying $S_{\rm mc}$ over a decade at fixed everything else; (J16, **new**) the
torch implementation of $T_{gg'}$, $p_{\rm eff}$ and eq. (3c) agrees with
`replicate_statistic.py` to floating tolerance on a fixture.

### Stage 3 -- The hypothesis test at default hyper-parameters

Run A0, A0s, A1, A2 ($\lambda_{\rm dsn} \in \{0.1, 1\}$), A2s, A3, A5, `A_ref`
and the shuffled control on the two-arm bench at $\pi = 0$ and one $\pi > 0$,
$n_{\rm seed} = 5$, $M_{\rm ens} = 1$, and apply S2.4 with the pseudo-real
endpoint.

Deliverables `run_joint_arms.py`, `jobs/joint_arms.pbs` (one job per
(arm, seed)), `report_joint_arms.py`: tables of $L$, $\hat\Delta$, per-axis
$\hat\Delta^{(k)}$ split by $S$/$F$, $\hat\Delta_{S\mid c}$, $r_{\rm eff}$,
ARI, replicate consistency, G1/G2, $\rho_{\rm grad}$; the paired-bootstrap
comparison table of S2.4a for every arm pair, on the aggregate and per axis
group; figures for per-axis gain, $z$ PCA per arm, coverage curves.

The same-donor embedding-distance comparison across A0(real), A2 and A5 (P13).
All three arms already exist, so this costs one extra table: it measures what
three different teachers produce for the same quantity, A0 and A2 obtaining
culture-invariance implicitly through `cross_culture` positives and A5
explicitly through the posterior.

**Added in v0.6:** the $p_{\rm eff}(26)$ versus $p_{\rm eff}(23)$ comparison
(P15) and the per-direction decomposition of $T_{gg'}$ (P16), both per arm.
Neither needs extra training.

**Exit criterion.** A written verdict on P1-P4, P6, P9, P13, P15, P16 and on
the degradation rule, with numbers, appended to the changelog. If A1 fails G1
on this easy bench, stop and fix the encoder or the loop before any search.

### Stage 3b -- Decomposing A0's deficit

A0 differs from A1 in objective **and** in training domain. A0s separates them.
With every score on the pseudo-real endpoint,

$$\underbrace{L(\mathrm{A0}) - L(\mathrm{A1})}_{\text{total}} = \underbrace{\big[L(\mathrm{A0}) - L(\mathrm{A0s})\big]}_{\text{domain effect}} + \underbrace{\big[L(\mathrm{A0s}) - L(\mathrm{A1})\big]}_{\text{objective effect}}, \tag{9}$$

each term with a paired cluster-bootstrap CI (S2.4a) and against
$\sigma_{\rm seed}$, and the same split per axis. The analogous joint-side
split is $L(\mathrm{A2}) - L(\mathrm{A2s})$.

**Plus two cheap direct probes of the domain effect**, which need no new
training: per-layer activation statistics of the r2 encoder on simulated versus
real inputs (dead or saturated units on simulated input is off-support
degeneracy, directly visible); and the marginal distribution of simulated
embeddings along the single surviving direction of the real cloud (piling at
one end is off-support, spreading across it is not).

Why it matters: if the domain term dominates, the fix is to fit the encoder on
simulated windows and the phenotype weight is second-order; if the objective
term dominates, $\lambda_{\rm dsn}$ and Stage 4 are the whole answer. The
recorded $r_{\rm eff}(\text{sim}) = 1.017$ under an encoder that never saw a
simulated window in training is consistent with either, plus a third
possibility -- that the prior predictive genuinely varies less than the cohort
in the features this encoder responds to, which is O2 and needs simulator work,
not encoder work.

### Stage 3c -- Nuisance floor, realisation floor, aliasing, stratification

On the bench first, where $\nu$, $\mathcal{G}$ and $\theta$ are all known:
`nuisance_floor.py` (Lever 3), **`realisation_floor.py` (new in v0.6: redraw
$\mathcal{G}$ at fixed $\theta$ and everything else, measure the dispersion of
$m_g$)**, `aliasing.py` (eq. (4): $J_\theta$, $J_\nu$ by finite differences,
principal angles, $a_m$, $\delta\theta_m$), `stratify.py` (eq. (5): the three
covariances and the generalised eigenproblem). Validation: on the bench the
recovered nuisance subspace must match the injected $\nu$ directions, the
realisation floor must concentrate on the kernel axes, and $\mu_j$ must exceed
1 exactly on the axes given genuine within-class spread. **The machinery is not
applied to real data until it passes this test.**

Also here: the window-aggregation curve (P10) and the factorised-NPE
composition, validated against a bench culture whose $\theta$ is known.

### Stage 4 -- The search campaigns

Deliverables `joint_space.py`, `npe_tune_joint.py` (subcommands mirroring
`npe_tune.py`, plus `partial-dependence`), `jobs/joint_tune.pbs`,
`jobs/launch_joint_tune.sh` (dry run by default), `smoke_test_joint_tune.py`.
Order: `baseline` (measures $\delta_{\min}$, $\sigma_{\rm seed}$), then S-A1 /
S-A2 / S-A5 in batches of 8, `status` escalation, `finalists --top-k 3` at
$M_{\rm ens} = 5$, gates, `partial-dependence`, `report` once.

### Stage 5 -- Robustness

Learning curves in $n_{\rm tr}$; $\tau_{\rm ov} \in \{0.05, 0.10, 0.20\}$;
$C \in \{2,3\}$; $T_{\rm win} \in \{60, 180\}$ s; nuisance severity sweep;
**realisation-variance sweep (new in v0.6): how far the D17 confound can be
pushed before J13b fails**; ablations (augmentation surrogates on/off, L2
normalisation on/off, $E \in \{6,10,16\}$); and a variant with
$\theta := \phi_S$ only, which tests whether the free axes should have been in
$\theta$ at all.

### Stage 6 -- Port to the real pipeline

- `Sbi-extractor/export_windows.py`: new mode writing **raw windows** for the
  simulated arm with the same sidecar contract as `export_embeddings.py`,
  reading $\Delta t$, $\sigma_{\rm sm}$, $T_{\rm win}$, $n_e$ from a config
  rather than a frozen checkpoint. Real-arm windows through `real_source.py`.
  Scale parity stays the per-electrode-mean convention [REPO].
- The real arm now enters the **training** loop for A2 and A5: read real
  windows with `donor`, `well`, `batch`, `subregion`, `condition`, with the
  holdout frozen and hashed **before** training and reused by every later gate
  or witness run.
- Same arms on the MFR-filtered, $\theta$-deduplicated 29 616-row bank with the
  existing grouped split. Cost: $W$ per Stage 0, $\sim 3\times10^4$ rows per
  member on CPU; budget from one probe epoch before any array.
- **New in v0.6, amended by D17's closure as (c):** before A5 trains on the
  real cohort, run the D17 audit on the *real* bank's terms -- count distinct
  realisations per kernel-parameter value -- and **record the count
  regardless of its value**. All 26 axes stay in the loss either way (D17,
  option (c)); the count goes into the Stage 3c report via
  `d17_realisation_audit()`, which is informational and cannot gate the run.
- Downstream: the gate and witness consume $z$ arrays and are
  encoder-agnostic, so re-export to a **new stem** and re-run; every witness
  number is encoder-conditional [KB]. TSNPE is untouched.

### Stage 7 -- Stratification on the cohort, and the drug arm

Culture-level posteriors, eq. (5) on real $m_g$ over all 26 axes, the diagnosis
as referee, the replicate statistic reported per donor with its per-direction
decomposition, and -- conditional on D14 -- eq. (6) and P12.

### Stage 8 -- Documentation and hand-off

Protocol and usage documents for the joint stack; changelog entries in
`SBI_PIPELINE.md` (S4, S10, O1, O2, O4, O5) and `HPC_PATHS.md`; frozen-artifact
conventions for the joint checkpoint (copy + sha256, never a symlink).

---

## 7. Cluster and delivery conventions

Pure ASCII, LF-only for every shipped `.py`, `.sh`, `.pbs`, `.md`; byte scan
before hand-over. PBS scripts: `eval "$(conda shell.bash hook)"` then absolute
interpreter path; self-locate from `BASH_SOURCE[0]` / `PBS_O_WORKDIR`; fail
loudly on a missing contract or shard. One job per trial; atomic ledger writes;
the results directory is the campaign state. Frozen artifacts in the
gitignored `artifacts/` with a `.sha256` beside each and a tracked README.
Never submit a full array before one probe run whose expected output lines were
stated in advance.

---

## 8. Decisions

**Open.**

- **D1 -- Loss composition.** Three-stream batches with the surrogate mask
  (recommended) versus one class-balanced batch for all terms.
- **D2 -- Bench window.** Iterate at 60 s / $W = 3000$, confirm once at 180 s?
- **D3 -- Classes and overlap.** $C = 3$, $\tau_{\rm ov} = 0.10$ primary with
  $C = 2$ as a fidelity replicate, or $C = 2$ throughout?
- **D4 -- Search design.** Four campaigns at equal budget, or S-A2/S-A5 only
  with partial-dependence read-outs?
- **D5 -- Bench scale convention.** Divide the cumulative IFR by a nominal
  $n_e = 9$ (recommended, so ranges transfer to Stage 6) or keep summed counts?
- **D6 -- Conditioner normalisation.** `z_score_x = "none"` (recommended) or
  sbi's `"structured"`?
- **D7 -- Ensemble semantics.** $M_{\rm ens}$ independent (encoder, flow) pairs
  (recommended) or one encoder shared by $M_{\rm ens}$ flows?
- **D9 -- Real-stream discipline.** Cycle the real cohort every step
  (recommended; ~54 rows, negligible) with a frozen culture holdout?
- **D10 -- Primary endpoint.** Pseudo-real held-out NLL (recommended) or the
  simulated-arm score?
- **D11 -- Gap severities.** Which perturbations, how many $\pi$ levels?
  Minimum viable: $\pi = 0$ plus one moderate level of (a).
- **D12 -- The design table (gates A5 and all of S2.7).** How many distinct
  donors behind the 35 cultures, and does any donor appear in more than one
  well or more than one batch? If every culture is a distinct donor with one
  well, $\Sigma_{\rm rep}$ has no same-donor pairs at the level that matters,
  eq. (3) degrades to subregion contrasts that miss differentiation and plating
  variance, and the simulated nuisance floor becomes essential rather than
  supplementary. Also: are batches nested within diagnosis? If plates are
  confounded with condition, only conditional forms of any invariance are
  admissible.
- **D13 -- Replicate consistency: loss or metric?** Train on it (arm A5) or
  report it only, for every arm? Recommendation is **both**, with the
  diagnostic use the stronger of the two: no gate and no standard calibration
  diagnostic can run on real data (S2.5), so this is the only calibration
  statistic available there.
- **D14 -- Pharmacology.** Are there paired pre/post-compound recordings, or
  any well-characterised blocker across a subset of cultures? Gates Stage 7's
  second half and P12.
- **D15 -- Which metric for eq. (3c)?** $(2\bar C)^{-1}$ (needs no replicate
  pairs; practical default, and the only one under which extra sloppy axes are
  free -- S2.5a), $V^{-1}$ estimated on the bench (exact null law, but needs
  $N_{\rm pair}\ge 27$ and transfers under an assumption the gate questions),
  or a shrunk/diagonal estimate? Decided by $N_{\rm pair}$, i.e. by D12.
- **D16 -- Point estimates or full posteriors?** Keep eq. (3c) on posterior
  means, or move to the Bayes-factor form (3d), which is correct under
  multimodality but costs density evaluations? Settle by first checking
  whether the posteriors are in fact multimodal -- nothing currently does.
**Closed.**

- **D17 -- Realisation marginalisation on the kernel axes (closed as option
  (c): accept and record, not mask).** The bank draws one connectivity
  realisation per $\theta^{\rm topo}$ value, so $F$ and hence $p_{\rm eff}$
  are overstated on the 3 kernel axes and eq. (3c) pushes $T_{gg'}$ upward
  there. Options were: (a) extend the campaign with repeated realisations;
  (b) keep all 26 axes in the diagnostic and mask the 3 kernel axes out of
  the loss until (a); (c) accept the bias and record it. **Why (c) and not
  (b), the previous default:** (b) was never implemented -- `joint_losses.py`
  has no axis-subset parameter, and its one call site builds $\Sigma_0$ from
  all 26 axes unconditionally -- so the runtime behaviour of (b)-pending and
  (c) is identical; (c) is the honest label for what runs, costs no masking
  code, and removes the invitation to "finish" a masking feature nothing
  depends on. The bias is recorded, not gated: `d17_realisation_audit()` in
  `run_stage3c.py` reports realisations-per-kernel-value informationally
  (structurally unable to fail the run -- it emits no `passed` key).
  **Residual work (the actual content of (c)):**
  `distinct_realisations_per_theta` has never been run against the real/ANN
  campaign bank; whether that bank's export even carries a
  `realisation_id` field, and whether the 1-realisation claim is a
  measurement or a directory-structure inference, are both unsettled -- see
  `HANDOFF_D17_option_c.md` S5. **Reopen condition:** if the Stage 3c
  concentration check finds realisation noise NOT concentrated on the kernel
  axes, D17's premise fails and "accept the bias on these 3 axes" stops
  being coherent; D17 reopens regardless of this closure.

- **D8 -- Alignment.** Closed in v0.3: no alignment term, no arm A4, no
  research thread. An MMD between the two summary clouds would optimise the
  gate's own statistic and destroy it as a test.
- **D18 -- Which axes does eq. (3) constrain? (closed in v0.6.)** All 26.
  The Weibull kernel parameterises a distribution over connections, which is a
  property of the preparation and is plausibly shared by two wells of one
  donor; the realised graph is a latent nuisance marginalised inside the
  likelihood, not a parameter. The measure-then-constrain ordering of v0.5 is
  withdrawn. Reasons and residual risk: S2.5c. The residual risk is D17, which
  is about the *bank*, not about the biology.

---

## 9. Open points, caveats, assumptions

**On the withdrawn drift argument.** v0.2 contained an argument (its S2.3b)
that the joint objective is invariant under an orthogonal transformation
applied to the real embedding cloud alone, and therefore blind to sim-real
separation. The algebra is correct: every constituent of $\ell_{\rm DSN}$ is a
function of distances, angles or inner products among real embeddings, all
preserved by $R \in O(E)$, while $\mathcal{L}^{\rm sim}_{\rm NPE}$ never
evaluates a real window. **The argument was nevertheless withdrawn**, because
realisability was asserted rather than shown: a domain-conditional rotation is
a far stronger object than a map whose outputs merely differ between domains,
the gate's rejection is evidence only for the latter, and a weight-shared
smooth CNN has no demonstrated route to the former. Nothing in either objective
aims at separation, and v0.2's suggestion that optimisation pressure favours it
was a conjecture, not a theorem. Recorded here rather than deleted so that
nobody re-derives it. Consequences: arm A4, decision D8, prediction P5 and the
"alignment trap" section are all removed.

**On the withdrawn measure-then-constrain ordering (new in v0.6).** v0.5's
S2.5(c) required $\Sigma_{\rm rep}$ to be measured under A1 before A5's index
set could be fixed. Withdrawn for the reasons in S2.5c: the kernel axes
describe a distribution, not a realisation, so the biological premise for
exempting them was never sound; and by eq. (3f) including a sloppy direction
costs essentially nothing in either mean or variance, so the gate protected
against a cost that does not exist. $\Sigma_{\rm rep}$ is retained as a
monitored diagnostic. What the withdrawal does **not** dispose of is D17, which
is a different objection with a different remedy.

**Assumed without proof.**

- **(3e)(i): that two wells of one donor share $\theta$ exactly, on all 26
  axes.** This is now the premise of the whole A5 construction rather than a
  hypothesis to be tested first. It remains falsifiable through P16 and through
  $\Sigma_{\rm rep}$, and if one direction dominates $T_{gg'}$ across donors the
  question reopens with data behind it.
- **(3e)(ii): that calibration verified on the bench transfers to real $x$.**
  Load-bearing and untestable on the cohort; model misspecification is exactly
  its failure. Every $T_{gg'}$ read on real data is conditional on it.
- The bench's exact independence of the free axes from the class (asserted by
  smoke test, not assumed).
- The observation-level character of $\nu$: nuisance is modelled as acting
  after the dynamics, which is what makes eq. (6) nuisance-invariant; a
  nuisance factor that alters the dynamics violates this and would have to move
  into $\theta$. Note that $\mathcal{G}$ is *not* of this kind -- it acts on the
  dynamics -- which is why it is marginalised inside the likelihood rather than
  treated as $\nu$.
- That windows of one subregion are exchangeable given $\theta$ -- used by the
  factorised composition of S2.7 and validated on the bench before real use.
- **Wells of one donor are conditionally independent given $\theta$**, used in
  the factor 2 of (3a) and in (3d). Shared plate, medium batch and session
  correlate them, in which case the factor under-estimates $V$ and $T$ is
  inflated.

**Approximations.**

- The label ceiling assumes exact collapse; the residual channel bounds rather
  than computes the interpolation. $r_{\rm eff} = C-1$ assumes equal class
  masses; recompute with empirical masses before comparing with a measurement.
- $\bar C$ is estimated from the current posterior, so eq. (3c) is a moving
  target during training -- fix it per epoch, not per step.
- **The closed forms (3a)-(3b) and (3f)-(3g) are exact only in the conjugate
  Gaussian case.** Elsewhere they are qualitative guidance; $V$ itself needs no
  distributional assumption and the trace identity survives without
  Gaussianity, but the $p_{\rm eff}$ target and the $\chi^2$ reference must be
  validated on the bench (J13, J13b).
- $(2\bar C)^{-1}$ approximates $V^{-1}$ well when $F \gg \Sigma_0^{-1}$ and is
  conservative by a factor $1/(1-c_j)$ per direction otherwise.
- **$M = (2\bar C)^{-1}$ is data-dependent (new caveat in v0.6).** Eq. (3g)
  assumes a *fixed* $M$. In the conjugate Gaussian model $C$ does not depend on
  $x$ and the issue does not arise, but for a trained flow $\bar C$ and
  $\Delta_{gg'}$ are functions of the same data, so
  $\mathbb{E}[T] = \mathrm{tr}(MV)$ holds only to the extent $C(x_g)$
  concentrates. The size of the resulting bias is unquantified; a bench
  experiment with a trained flow rather than the analytic model would settle
  it.
- **The prior is treated as Gaussian with covariance $\Sigma_0$ (new caveat in
  v0.6).** It is a uniform box; $(b_k-a_k)^2/12$ is its second moment, not a
  Gaussian covariance. Substituting it in (3g) is a separate approximation from
  the conjugacy one and was not flagged before v0.6.

**Left unresolved.**

- Whether the collapse observed on the real arm transfers to the simulated arm
  (Stage 3b), and which of the three explanations of $r_{\rm eff}(\text{sim}) =
  1.017$ holds.
- The magnitude of $I(\theta; x)$, the outer ceiling. Nothing estimates it, so
  no statement is possible about what fraction of the achievable information
  any encoder captures. A deliberately over-complete A1 at large $E$ gives a
  lower bound.
- **D17 in full.** How many realisations per kernel value the bank actually
  contains, and the size of the resulting bias in $p_{\rm eff}$.
- **Power.** Thirty-five cultures, fewer donors. Stratifying donors within a
  diagnosis in a stiff subspace of dimension 3-5 is hypothesis-generating, not
  powered clustering, and the paper must say so before a reviewer does. No
  power analysis exists for $T_{gg'}$ either: the probability that it detects a
  given degree of overconfidence at the available $N_{\rm pair}$ is unknown.
- **Anchoring.** Under A1 and A5 the diagnosis enters no training objective,
  and A5 anchors the encoder only through a consistency constraint, so nothing
  anchors the *mechanism* directions to real data in an absolute sense. With a
  simulator the gate rejects at the permutation floor, this is a real exposure;
  the gate and witness must be reported per arm, not run once.
- The bench's $\pi$ is not calibrated to the cohort's actual gap. Conclusions
  transfer as orderings and mechanisms, not as magnitudes.
- **The bootstrap's validity rests on groups being i.i.d.** Dependence
  *between* groups -- shared random seeds across topology draws, a systematic
  drift over the simulation campaign, or any batch structure in how the bank
  was generated -- is not handled by any scheme here and would need a
  different resampling unit again. Nothing currently checks for it.
- $\rho(d)$ is estimated by a one-way ANOVA decomposition, which assumes the
  group effect is additive and homoscedastic. With strongly unequal group
  sizes the estimator is noisy; treat it as an order of magnitude that
  selects a scheme, not as a quantity to report to three decimals.
- **On the real bank the group count is small.** Thirty-five cultures means a
  bootstrap distribution built by resampling 35 units, so a 2.5% percentile is
  coarse and the interval will be honestly wide. That is not a defect of the
  method. It also means group count, not row count, is what to read off the
  split manifest before interpreting any $D$.
- $W$ is unresolved until Stage 0 reads the checkpoint (9 000 vs 18 000), a
  factor of two in Stage 6 compute.
- **Two numeric inconsistencies in the knowledge base itself**, found while
  verifying for the handoff and **not resolved**; do not silently pick one.
  (i) The MFR-filtered bank is **29,616** rows in `SBI_PIPELINE.md` S6 and
  S12.4 but **29,416** in its O1 and in the `HPC_PATHS.md` changelog.
  (ii) `SBI_PIPELINE.md` fixes $d_\theta = 26$ throughout, while
  `HPC_PATHS.md` S8 says not to assume a fixed width -- it is determined by
  `artifacts/label_axes.json` and depends on which campaigns are included and
  their `conn_rule`. Every number in this plan assumes 26, and Stage 0 now
  asserts it.
- **The 23 + 3 split is [reasoning], not read.** The 3 follows from the Weibull
  kernel $p(d) = p_0\exp(-(d/d_0)^\beta)$ having three parameters, and
  $23 = 26-3$. **v0.6 lowers the stakes of this considerably**: with the
  constraint acting on all 26 axes, no loss depends on the partition. It still
  matters for the per-axis-group design effect (S2.4a) and for D17, so it
  should still be confirmed against `label_axes.json`.
- **Four discrepancies between docstring and running code in the gate stack**
  are recorded in `GATES_v1.md` S5 and are not fixed. The consequential one:
  G3's coverage-deficit branch appears unreachable, so the documented
  tolerance of over-coverage is not in force and `coverage_tol` has no effect.
  Gate verdicts in this plan are conditional on that.

---

## 10. References and provenance

**Knowledge base, full text.** `SBI_PIPELINE.md`, `HPC_PATHS.md`,
`witness_usage.md` (pipeline state, effective ranks, gate verdict, truncation
results, open points). Radev et al., *BayesFlow* (IEEE TNNLS 2022) -- joint
summary/inference training. Schmitt et al., *Detecting model misspecification
in amortized Bayesian inference* -- the weighted auxiliary term on the summary
space and the misspecification-severity design used for $\pi$. *Simulation
Based Inference: A Practical Guide* and *Compositional simulation-based
inference for time series* -- combining conditionally independent observations;
also the source for the standard calibration diagnostics (SBC, expected
coverage, TARP, L-C2ST) all requiring $\theta^*$. Goncalves et al. 2020; the
sloppiness review (orientation only, no numerical claim); TSNPE; TARP.

**PubMed, full text read.** Min J et al., *Neural posterior estimation for
population genetics*, Genetics 2026;233(3),
[DOI](https://doi.org/10.1093/genetics/iyag107) -- joint embedding + flow
training by SGD on the NLL, and the frozen pre-trained network as learned
summaries (the A0 analogue). Hommersom MP et al., Brain 2025;148(4):1286-1301,
[DOI](https://doi.org/10.1093/brain/awae330) -- biophysical MEA network model
with SBI, chosen for parameter degeneracy; the differential and partly opposite
effects of 4-AP and NS309; the stated need for multiple patient-derived models.

**PubMed, abstract only, flagged.** Van Lent J et al., Brain 2021,
[DOI](https://doi.org/10.1093/brain/awab226) -- heterogeneity under one
diagnostic label, subtype-dependent cellular phenotype. Pavlinek A et al., Cell
Rep Methods 2026, [DOI](https://doi.org/10.1016/j.crmeth.2026.101371) -- donor
line and batch effects in MEA. Ronchi S et al., Adv Biol 2021,
[DOI](https://doi.org/10.1002/adbi.202000223) -- drug-effect detection and
culture-to-culture variability depend on electrode count. Xie Z et al., Sensors
2026, [DOI](https://doi.org/10.3390/s26072088) -- negative transfer monitored
by gradient cosine. No numbers used from any of these.

**Searches run and reported.** PubMed on jointly trained summary and inference
networks; multi-task negative transfer; GP hyper-parameter search;
hiPSC/MEA line variability and drug-response heterogeneity; within-diagnosis
mechanistic heterogeneity. **New for v0.6:** PubMed on Mahalanobis-form null
distributions for replicate/reproducibility statistics; calibration diagnostics
with a stated null for neural posterior estimators; effective degrees of
freedom / effective number of parameters; well-to-well and culture-to-culture
variability in iPSC-derived MEA networks. The first three returned **zero
records**; the fourth returned 43, none of which addresses whether
connectivity-kernel parameters are a donor-level property shared across wells
-- the specific claim S2.5c rests on. **So the shared-kernel premise is made on
mechanistic grounds, not on a citable measurement, and is recorded in S9 as an
assumption.** bioRxiv/medRxiv: the connector now responds (it errored in every
earlier session) but supports only category and date filtering with **no
keyword search**, so no targeted preprint query is possible; this is an
observed tool limitation, not a statement about coverage. **No claim in this
plan is sourced from a preprint.**

**Stated from memory, not verified against a source.** That
$\mathbb{E}[\Delta^\top M\Delta] = \mathrm{tr}(MV) + \mu^\top M\mu$ for
$\Delta$ with mean $\mu$ and covariance $V$; that
$\sum_j \lambda_j/(1+\lambda_j)$ is the standard effective-degrees-of-freedom
expression familiar from ridge regression and DIC; that the Fisher information
of a Gaussian location family is the inverse covariance; the Ledoit-Wolf
shrinkage estimator and its availability in `scikit-learn`; the law of total
variance in the form
$\mathrm{Cov}(\theta) = \mathbb{E}[\mathrm{Cov}(\theta\mid z)] +
\mathrm{Cov}(\mathbb{E}[\theta\mid z])$. All are textbook, and the derivations
that depend on them are given in `METRIC_REPLICATE_v1_1.md`, so the
attributions can be checked independently of the mathematics.

**Repositories, read from source.** `Deep-Summary-Network@main`,
`Simulation-Based-Inference@feat/misspec-gate`,
`Sbi-extractor@feat/real-arm-parity`, `Astro-Neuron-Network@main`.

**Companions.**
`INFO_LOSS_THEORY_v1.md` -- Propositions 1-15: the label ceiling, the residual
channel, $r_{\rm eff} = C-1$, the cardinality-versus-dimension correction. Its
S3.7.2 (Proposition 12) is superseded by S9 here and should be read with that
caveat attached.
`GATES_v1.md` -- G1/G2/G3 as implemented, what each gate cannot see, and four
docstring-versus-code discrepancies.
`METRIC_REPLICATE_v1_1.md` -- the full derivation behind S2.5: the three aims,
the which-randomness table, the composite null, invariance, eqs. (3a)-(3h),
$p_{\rm eff}$ as a count of data-owned directions, the evaluated-versus-
derivation-only table, and the estimation constraints. **Supersedes
`METRIC_REPLICATE_v1.md`.**
`PROJECT_IDEAS.md` -- IDEA-001 (drop the diagnostic classification; the aim
stated in the abstract here) and IDEA-002 (morphing without full retraining).
`HANDOFF_joint_dsn_npe.md` -- the session handoff: what was and was not
verified, the withdrawn material, and the next actions in cost order.
`bootstrap_paired.py` + `smoke_test_bootstrap_paired.py` -- the interval
machinery of S2.4a, delivered and verified.
`replicate_statistic.py` + `smoke_test_replicate_statistic.py` -- the
replicate-statistic machinery of S2.5, delivered and verified in v0.6.

---

## 11. Changelog

- **2026-09-02 v0.6.** Changes the index set the replicate constraint acts on,
  and delivers the machinery. **Withdrawn:** the measure-then-constrain
  ordering of v0.5 S2.5(c) and S2.7, and with it the requirement that
  $\Sigma_{\rm rep}$ be measured under A1 before A5 can be configured
  (reasons in S9). **Changed:** eq. (3) now constrains all $d_\theta = 26$
  axes, because the Weibull kernel parameterises a *distribution* over
  connections -- a property of the preparation that two wells of one donor
  plausibly share -- while the realised graph is a latent nuisance
  marginalised inside the likelihood; $\Sigma_{\rm rep}$ becomes a monitored
  diagnostic; $p_{\rm eff}$, $\chi^2$ and the $N_{\rm pair}$ bound move from
  $23/276/24$ to $26/351/27$; $\bar C$ is symmetrised across the two wells;
  the finite-draw inflation $d_\theta/S_{\rm mc}$ is corrected, eq. (3h), and
  `n_posterior_draws` enters the search space. **Added:** the null hypothesis
  written out as a composite statement, eq. (3e); the per-direction share
  identity, eq. (3f), and the observation that sloppy axes are free under
  $(2\bar C)^{-1}$ but not under $V^{-1}$; the formal statement of the
  constant-map minimum and why the two-sided target turns it into a
  divergence; the three-escape-route table separating what the non-zero
  target, the NPE term and the stop-gradient each block; the
  evaluated-versus-derivation-only convention; a realisation latent on the
  bench and the realisation floor in Stage 3c; predictions P15, P16; decision
  D17 and the closure of D18; smoke tests J13b, J15, J16 and S10; and the
  delivered, verified `replicate_statistic.py` with its 14-test smoke test.
  **Corrections marked inline:** $\mathbb{E}[\Delta^\top M\Delta] =
  \mathrm{tr}(MV)$ does not require Gaussianity (v0.5 said it did); shared
  $\theta^*$ does not mean shared posterior mean; the stop-gradient does not
  guard against covariance inflation -- the NPE term does. **New caveats:**
  $M$ is data-dependent so the fixed-$M$ assumption behind (3g) is violated by
  an unquantified amount; the uniform box prior is being used as a Gaussian
  covariance. **Housekeeping:** ensemble size renamed $M_{\rm ens}$ and the
  stratification eigenvalues renamed $\mu_j$, to end two silent symbol
  collisions in v0.5.
- **2026-09-02 v0.5.** Rewrites S2.5(iii), arm A5, around three choices that
  v0.4 got wrong, with **two corrections marked inline**: the direction of the
  metric argument (replicate disagreement is an inverted U in $\lambda_j$,
  eq. (3b), largest where the data is as informative as the prior, not on
  sloppy directions), and the target (eq. (3c) is two-sided about
  $p_{\rm eff}$, not driven to zero; the metric does not protect against
  collapse). Adds: the invariance criterion that selects a Mahalanobis form
  and dissolves the axis-versus-direction question; the identification of
  $p_{\rm eff}$ with the sum of the generalised eigenvalues
  `information_spectrum` already reports; measure-then-constrain ordering for
  the axis partition (S2.5c, S2.7) -- **withdrawn in v0.6**; warm-up plus
  stop-gradient on $\omega$ and on $C$ (S2.5d); the multimodality limitation
  and the Bayes-factor repair (3d); the estimation constraints at small
  $N_{\rm pair}$ (S2.5f). States IDEA-001 explicitly as the standing aim in
  the abstract. New predictions P13, P14; new decisions D15, D16; new smoke
  tests J12-J14; the same-donor distance comparison in Stage 3. Records the
  two knowledge-base numeric inconsistencies and the derived status of the
  23+3 split.
- **2026-08-30 v0.4.** Fixes the inference procedure behind the central
  comparison. S2.4 rewritten: pairing at the row level made explicit, and the
  interval built by paired bootstrap over the resampling unit rather than the
  "row-bootstrap" of v0.2-v0.3, which was wrong for a grouped split. New
  S2.4a specifies all four schemes, the primary choice, the numerical evidence
  for it (the $1/\sqrt{\mathrm{DEFF}}$ error law and the 57%-versus-93%
  coverage result), the estimand difference between row- and group-weighting
  under unequal group sizes, the finiteness and tail preconditions, and the
  requirement to take the group key from the split manifest. Records the
  correction that rows within a topology group share **3 of 26** axes and are
  genuine distinct realisations, not near-duplicates as earlier drafts said --
  which lowers the expected $\rho(d)$ without making it zero, and makes the
  design effect axis-dependent. `bootstrap_paired.py` and its smoke test are
  delivered and verified (Stage 1). Adds the requirement that the scorer
  persist per-row log-densities and group labels. New caveats on
  between-group independence, ANOVA assumptions, and the small group count on
  the real bank.
- **2026-08-30 v0.3.** *Withdrawn:* the domain-drift argument (v0.2 S2.3b), the
  alignment trap (v0.2 S2.5), arm A4, decision D8, prediction P5, and
  $\mathrm{MMD}^2_\Sigma$/support-overlap as drift monitors -- reasons in S9.
  The gate's MMD remains what it always was, a misspecification test.
  *Retained and sharpened:* the two-label-source asymmetry, which is a fact
  about data availability and does not depend on the withdrawn argument; and
  the out-of-domain character of A0's encoder, which is plain covariate shift
  (S2.1, Stage 3b) and likewise independent of it. *Added:* the sufficiency
  asymmetry (S2.3); replicate-consistency supervision and arm A5, eq. (3), with
  the constant-map caveat (S2.5); the label as referee rather than teacher
  (S2.5); the transversality/aliasing test, eq. (4), and the nuisance floor
  (S2.6); culture-level hierarchical inference, the window-aggregation
  discriminator, and the three-covariance stratification, eq. (5) (S2.7); the
  drug-response validation, eq. (6) (S2.8); a nuisance latent $\nu$ on the
  bench (S4.0); $\lambda_{\rm rep}$ in the search space and campaign S-A5
  (S5); Stage 3c; two direct probes of the domain effect in Stage 3b; Stage 7;
  predictions P10-P12; decisions D12-D14. Prediction numbering is preserved
  across versions so that cross-references to the theory document remain valid.
- **2026-08-28 v0.2.** The two-domain amendment: the NPE term can only be fed
  by simulated windows and the DSN term only by real ones. Arms A0s, A2s;
  pseudo-real bench arm and gap knob $\pi$; endpoint moved to the pseudo-real
  NLL; Stage 3b; decisions D8-D11. (Parts withdrawn in v0.3; see S9.)
- **2026-08-28 v0.1.** Initial plan. Decisions D1-D7 open.
