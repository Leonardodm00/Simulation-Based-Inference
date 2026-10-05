# E4 -- The joint objective and the loop: three terms, three streams, what each weight reaches, and what the arms isolate

**Document E4 of the joint documentation set; the fourth chapter of Set E.**
What the joint objective's three terms estimate in the code, which weights
each term moves, what a loss weight does to an AdamW update, what the loop
records about the interaction of two terms, and what each of the nine arms
isolates. Index and status: `00_INDEX.md`; notation:
`E0_READERS_GUIDE_NOTATION.md`; the loop's arithmetic and the optimiser's
knobs: `P5_OPTIMISER_AND_SCHEDULE.md`; the metric term:
`P2_DSN_LOSS_AXES.md`, `E3_THE_SUMMARY_NETWORK.md`; the replicate term:
`P3_REPLICATE_AXES.md` (E5 next). **Date:** 2026-10-05 (v1). **Applies
to:** the repository `Simulation-Based-Inference` at `834eb41`
(`hpc/joint/`, D-037), and the project documents and PDFs named in S6.

| date | change |
|---|---|
| 2026-10-05 | v1. Written from the plan `JOINT_DSN_NPE_PLAN_v0_6.md` (repository, v0.6.5: S2.2, lines 195-249; the predictions, 296-316; S5.1, 1069-1117; Stages 2, 3, 3b and 6, lines 1236-1395; D9 and D10, 1575-1578), `stage2/joint_train.py`, `joint_batches.py`, `joint_model.py`, `joint_losses.py` and `dsn_loss_adapter.py` in full, `stage3/run_joint_arms.py` in full, `stage3/jobs/joint_arms.pbs`, `stage3/report_joint_arms.py:34-117`, `stage4/npe_tune_joint.py:517-528`, sbi 0.27.0 (`posterior_nn` and the y-embedding), torch 2.10.0's documentation of `std` and `quantile`, and scipy 1.18.1's exact binomial interval; P2 S3.3.2 and S3.5, P3, P5 S3.2 and S3.6-S3.8, P7 (F-at, F-aw), E1 S3.6, E3 S3.4 and S3.6; the deck's sections B and G (KB); `HPC_PATHS.md` and the decision log's D-056 for the cohorts' class counts; the BayesFlow PDF at the pages cited; five PubMed Central full texts (S6); PubMed and bioRxiv searched (S6). Every `[RAN]` number is printed by the new torch-free `tools/e4_numbers.py` (two identical runs). New symbols $N_c$, $\mathsf{w}_c$, $\mathcal{Q}_{\rm met}$, the three per-term gradients, $\mathsf{u}_\tau$, $\rho_{\rm norm}$, $f_-$ and $\hat f_-$, $n_{\rm cos}$ (E0 v1.11, convention 14). Five findings, F-bd to F-bh, all owned here. Dated notes added to E1 S3.7 (the grouped split is the simulated arm's), E3 S3.6 (`A0s`'s pre-training reads the scored split), P2 S3.3.2 ("dominates every step" given the AdamW reading) and P7's convention 5 (for which arms arm R's rows are held out). |

**Abstract.** The plan writes one objective with three terms and two
weights, and a reader takes the runner's $\lambda_{\rm dsn} = 0.1$ to mean
that the class label pulls on the encoder a tenth as hard as the likelihood.
The question this chapter answers is what the terms and the weights are in
the loop that actually trains the stack -- what each term estimates, which
weights it moves, what a weight does once AdamW and the gradient clip have
acted on the gradient, what the loop records about it -- and, given that,
what each arm can isolate. **Covered:** the objective and the three
sampling laws the loop estimates it under, including what the metric term's
expectation is an expectation over (S3.2, eqs. (E4.1)-(E4.2)); which weights
each term reaches and the two places the stack cuts a gradient (S3.3, eq.
(E4.3)); what a loss weight does under AdamW and the clip -- for a given
encoder trajectory, nothing on the flow's weights except through the clip's
variation from step to step, and on the encoder's a mixture that saturates
at both ends (S3.4, eqs. (E4.4)-(E4.6)); the gradient-cosine
record, the norms it computes and drops, and what ten probes per run can say
about prediction P4 (S3.5, eqs. (E4.7)-(E4.8)); the loop as built, and what
"early stopping on the NPE validation score only" means at the runner's
settings (S3.6); the two data domains in one loop and the sense in which the
primary endpoint is held out (S3.7); the nine arms as an ablation design,
the contrasts they serve and what else differs inside each contrast (S3.8,
eq. (E4.9)). **Deliberately excluded:** the replicate statistic and its
target (E5; only its gradient reach and its ramp appear here); the metric
loss's geometry (E3); the optimiser's and the schedule's knobs one by one
(P5, cited); the bench generator (E6); the decision rule and the bootstrap
(E7); the search (E8). No number here is a result of a training run: nothing
in `hpc/joint/` has run on the cluster, torch is absent from the sandbox,
and the `[RAN]` numbers are the arithmetic of the code's sampling laws, a
numpy replica of torch's AdamW on a toy, and a numpy transcription of the
reference arm's features.

---

## 1. Notation and symbols

A subset of E0's master table, in order of first use, plus the symbols E4
adds (marked *new*; E0 v1.11 declares them under convention 14).

| Symbol | Name / Meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| $n_\psi, n_\omega$ | the numbers of the encoder's and the flow's weights | $\mathbb{N}$ | -- | S1.1 |
| $n_\varpi$ | the length of $\varpi$, the loop's trainable weights | $\mathbb{N}$ | -- | S1.1 |
| $\psi$, $\omega$ | the encoder's and the flow's weights | $\mathbb{R}^{n_\psi}$, $\mathbb{R}^{n_\omega}$ | -- | S3.1 |
| $\lambda_{\rm dsn}, \lambda_{\rm rep}$ | the weights of the metric and replicate terms in $\mathcal{L}$ (configured) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.1 |
| $\theta$, $\Theta$, $d_\theta$ | the inference parameters of one row, the prior box, its dimension | $\theta \in \Theta \subset \mathbb{R}^{d_\theta}$ | mixed (E0) | S3.1 |
| $x$, $W$ | one window, and its length in samples | $x \in \mathbb{R}^{W}$ | as E0 | S3.1 |
| $S^{E-1}$, $E$ | the unit sphere of $\mathbb{R}^{E}$, and the embedding dimension | manifold; $\mathbb{N}$ | -- | S3.1 |
| $c$, $C$ | the class label and the number of classes | $c \in \{0, \dots, C - 1\}$; $\mathbb{N}$ | -- | S3.1 |
| $B_{\rm sim}, B_{\rm met}, B_{\rm rep}$ | rows per step in the simulated and metric streams, pairs in the replicate stream (runner: 128, 32, 4) | $\mathbb{N}$ | rows; pairs | S3.1 |
| $n_{\rm plan}$ | planned steps, $n_{\rm ep} n_{\rm step}$ in the joint loop | $\mathbb{N}$ | steps | S3.1 |
| $n_{\rm ep}, n_{\rm step}$ | epochs, and steps per epoch | $\mathbb{N}$ | -- | S3.1 |
| $\mathcal{L}$ | the joint objective, plan eq. (1), $\mathcal{L}(\psi, \omega)$ (analytic level) | $\mathbb{R}$ | nats/row plus dimensionless terms | S3.2 |
| $\mathcal{L}^{\rm sim}_{\rm NPE}$ | the expected NLL of $\theta$ given $h_\psi(x)$ under $p_{\rm sim}$, plan eq. (1a) (analytic level) | $\mathbb{R}$ | nats/row | S3.2 |
| $\mathcal{L}^{\rm real}_{\rm DSN}$ | the expectation of $\ell_{\rm DSN}$ over the metric stream's batch law, eq. (E4.2) (analytic level) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\mathcal{L}^{\rm real}_{\rm rep}$ | the replicate term, plan eq. (3c) (E5) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $h_\psi$, $z$ | the encoder and the embedding $z = h_\psi(x)$ | $h_\psi : \mathbb{R}^{W} \to S^{E-1}$ | dimensionless | S3.2 |
| $q_\omega$ | the conditional flow, $q_\omega(\theta \mid z)$, a density on $\Theta$ for each fixed $z$ | conditional density | -- | S3.2 |
| $p_{\rm sim}$, $p_{\rm real}$ | the simulator's joint law and the law of real windows | laws | -- | S3.2 |
| $\ell_{\rm DSN}$ | the composite metric loss of one metric batch (computed level) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $X^{\rm real}_{\rm met}, y^{\rm real}_{\rm met}$ | one metric batch and its labels; "real" names the stream, not the domain (S1.1) | batch of $C n_c$ rows; labels in $\{0, \dots, C - 1\}$ | -- | S3.2 |
| $\mathcal{T}_{\rm mined}$ | the triplets the miner selects from a batch | set of index triples | -- | S3.2 |
| $\hat{\mathcal{L}}_{\rm step}$ | the per-step estimate, plan eq. (2) as P5 eq. (P5.1) writes it (computed level) | $\mathbb{R}$ | mixed | S3.2 |
| $\ell_i$ | the NLL of row $i$ under the current weights, $-\log q_\omega(\theta_i \mid h_\psi(x_i))$ | $\mathbb{R}$ | nats | S3.2 |
| $i$ | row index | index | -- | S3.2 |
| $\varpi$ | the trainable weights of the loop, the `requires_grad` part of $(\psi, \omega)$ | $\mathbb{R}^{n_\varpi}$ | mixed | S3.2 |
| $N_{\rm train}$ | rows of the training split | $\mathbb{N}$ | rows | S3.2 |
| $\mathcal{B}_{\rm sim}$, $\mathcal{B}_{\rm rep}$ | one step's simulated rows and replicate pairs | index sets | -- | S3.2 |
| $N_{\rm pair}$ | the same-donor pairs the replicate stream enumerates | $\mathbb{N}$ | pairs | S3.2 |
| $g, g'$ | the two wells of one pair | indices into the real bank | -- | S3.2 |
| $\ell_{\rm rep}$ | the replicate loss of one pair (P3) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $r_{\rm rep}$ | the replicate term's ramp, $r_{\rm rep}(t)$ (P3) | $[0, 1]$ | dimensionless | S3.2 |
| $t$ | training progress, the fraction of planned steps done | $[0, 1]$ | -- | S3.2 |
| $n_c$ | rows of class $c$ in one metric batch, $\max(1, \lfloor B_{\rm met}/C \rfloor)$ in the stack | $\mathbb{N}$ | rows | S3.2 |
| $N_c$ | *new.* Rows of class $c$ in the metric stream's source: the real bank (`A2`, `A3`, and `A0`'s pre-training) or the simulated training split (`A2s`; `A0s`'s pre-training reads the whole simulated bank, F-be) | $\mathbb{N}$ | rows | S3.2 |
| $\mathsf{w}_c$ | *new.* The factor by which class $c$'s share of every metric batch exceeds its share of the source, $(1/C) / (N_c / \sum_{c'} N_{c'})$ | $\mathbb{R}_{> 0}$ | dimensionless | S3.2 |
| $\mathcal{Q}_{\rm met}$ | *new.* The law of one metric batch: $n_c$ rows of each class, uniform with replacement within the class (analytic level) | law on batches | -- | S3.2 |
| $\lambda_{\rm sep}$ | the separation term's weight inside $\ell_{\rm DSN}$ (P2) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $s_{\rm seed}$ | the run seed (P5) | $\mathbb{N}_0$ | -- | S3.2 |
| $\bar C$ | the symmetrised posterior covariance of a pair (P3, E5), detached in the replicate term | positive definite $d_\theta \times d_\theta$ | as $\theta^2$ | S3.3 |
| $\tau$, $\tau'$ | optimiser step index, from 1; $\tau' \le \tau$ an earlier step | $\{1, \dots, n_{\rm plan}\}$ | steps | S3.3 |
| $\Gamma_\tau$, $\tilde\Gamma_\tau$ | the gradient of $\hat{\mathcal{L}}_{\rm step}$ with respect to $\varpi$ at step $\tau$, before and after the clip (P5) | $\mathbb{R}^{n_\varpi}$ | mixed | S3.3 |
| $\Gamma^{\rm NPE}_\tau, \Gamma^{\rm DSN}_\tau, \Gamma^{\rm rep}_\tau$; $\Gamma^{\rm NPE}, \Gamma^{\rm DSN}, \Gamma^{\rm rep}$ | *new.* The gradients with respect to $\varpi$ of the unweighted NPE, metric and replicate terms of $\hat{\mathcal{L}}_{\rm step}$ at step $\tau$; written $\Gamma^{\rm NPE}, \Gamma^{\rm DSN}$ without a step index on the probe batch (S3.5) | $\mathbb{R}^{n_\varpi}$ | mixed | S3.3 |
| $\varrho_\tau$ | the clip coefficient of step $\tau$ (P5) | $(0, 1]$ | dimensionless | S3.4 |
| $\gamma_{\rm clip}$ | the global gradient-norm clip threshold (5.0) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.4 |
| $\eta$, $\gamma_{\rm wd}$ | AdamW's learning rate and decoupled decay | $\mathbb{R}_{> 0}$; $\mathbb{R}_{\ge 0}$ | dimensionless | S3.4 |
| $\beta_1, \beta_2$ | AdamW's decay rates | $(0, 1)$ | dimensionless | S3.4 |
| $\bar\mu^{(1)}_\tau, \bar\mu^{(2)}_\tau$ | AdamW's bias-corrected moving averages of the clipped gradient and of its square (P5) | $\mathbb{R}^{n_\varpi}$ | mixed | S3.4 |
| $\epsilon_{\rm adam}$ | AdamW's denominator constant ($10^{-8}$) | $\mathbb{R}_{> 0}$ | mixed | S3.4 |
| $\mathsf{u}_\tau$ | *new.* AdamW's normalised step, $\bar\mu^{(1)}_\tau / (\sqrt{\bar\mu^{(2)}_\tau} + \epsilon_{\rm adam})$ elementwise; written $\mathsf{u}_\tau[\cdot]$ when computed from the gradient history named in the bracket | $\mathbb{R}^{n_\varpi}$ | dimensionless | S3.4 |
| $\rho_{\rm grad}$ | the cosine of the NPE and metric terms' gradients on the coordinates of $\psi$, on one probe batch (computed level) | $[-1, 1]$ | -- | S3.5 |
| $\rho_{\rm norm}$ | *new.* The realised gradient-norm ratio on the coordinates of $\psi$, on one probe batch: $\lambda_{\rm dsn}$ times the metric term's gradient norm over the NPE term's (computed level; computed by the code and not recorded, F-bd) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.5 |
| $f_-$, $\hat f_-$ | *new.* The probability that one probe gives $\rho_{\rm grad} < 0$ (analytic level, assumed common to a run's probes), and the observed fraction of a run's probes that do (computed level) | $[0, 1]$ | -- | S3.5 |
| $n_{\rm cos}$ | *new.* The number of $\rho_{\rm grad}$ probes counted: $n_{\rm ep}$ per run, $n_{\rm ep} n_{\rm seed}$ pooled over an arm's seeds | $\mathbb{N}$ | -- | S3.5 |
| $n_{\rm seed}$ | seeds per arm (5 in the plan's Stage 3) | $\mathbb{N}$ | -- | S3.5 |
| $L$, $L_{\rm sel}$ | an arm's mean held-out NLL over the scored rows, and the same on the selection split (computed level) | $\mathbb{R}$ | nats/row | S3.6 |
| $\iota$, $\iota_{\rm best}$, $n_{\rm pat}$ | epoch index, the best epoch so far, the patience (P5) | integers | epochs | S3.6 |
| $\mathcal{S}$, $\mathcal{R}$ | the bench's simulated arm and its pseudo-real arm, whose $\theta$ is recorded and withheld (plan S4.0) | sets of rows | -- | S3.6 |
| $\pi$ | the bench's gap severity | $[0, 1]$ | -- | S3.7 |
| $\psi^\star$ | the encoder weights arm `A0` ends with, the warm start of `A3` | $\mathbb{R}^{n_\psi}$ | -- | S3.7 |
| $\hat\Delta$, $\delta_{\min}$ | the information gain, the prior floor minus $L$, and the shuffled control's gain, the "learned nothing" floor (E2, E7) | $\mathbb{R}$ | nats/row | S3.8 |
| $\sigma_{\rm seed}$ | the across-seed standard deviation of an arm's held-out NLL (E7) | $\mathbb{R}_{\ge 0}$ | nats/row | S3.8 |

### 1.1 Conventions

- E0 S1.1 applies in full: conditional quantities are written conditionally
  every time; every density carries a subscript naming its law; the analytic
  and computed levels carry different symbols (convention 4).
- **Levels in this chapter (R8).** Analytic: $\mathcal{L}$ and its three
  terms as expectations, $\mathcal{Q}_{\rm met}$, the expectation
  $\mathbb{E}[\hat{\mathcal{L}}_{\rm step}]$ over one step's draws, $f_-$.
  Computed: $\hat{\mathcal{L}}_{\rm step}$, $\ell_i$, $\ell_{\rm DSN}$ (one
  batch), $\ell_{\rm rep}$ (one pair), the per-term gradients, $\varrho_\tau$,
  $\mathsf{u}_\tau$, $\rho_{\rm grad}$ and $\rho_{\rm norm}$ (one probe),
  $\hat f_-$, $L$, $L_{\rm sel}$. A move between the two is named where it is
  made ("estimates", "in expectation over the draws").
- **Coordinates.** A vector in $\mathbb{R}^{n_\varpi}$ has one coordinate per
  scalar weight. "On the coordinates of $\psi$" restricts it to the encoder's
  $n_\psi$ weights, "on the coordinates of $\omega$" to the flow's; a norm or
  an inner product with subscript $\psi$ is taken over the encoder's
  coordinates only.
- **The metric stream's superscript.** "real" in $X^{\rm real}_{\rm met}$
  and $\mathcal{L}^{\rm real}_{\rm DSN}$ names the stream (E3 S1.1): for
  `A0s` and `A2s` the stream holds simulated rows with the generator's labels.
- **Status (Provenance model).** $\lambda_{\rm dsn}$, $\lambda_{\rm rep}$,
  $B_{\rm sim}$, $B_{\rm met}$, $B_{\rm rep}$, $\gamma_{\rm clip}$ are
  `configured`; $N_c$ and $N_{\rm pair}$ are `measured` (counted from the
  bank); $\mathcal{Q}_{\rm met}$ is `analytic` (the sampler's law);
  $\hat{\mathcal{L}}_{\rm step}$, the per-term gradients, $\varrho_\tau$ and
  $\mathsf{u}_\tau$ are `computed` at every step and never recorded;
  $\rho_{\rm grad}$ is `computed` once per epoch and recorded; $\rho_{\rm norm}$
  is `computed` -- both norms exist in `gradient_cosine` -- and discarded
  (F-bd); $f_-$ is `derivation-only`. Read from source (`[REPO]`, S6).
- **The toy of `tools/e4_numbers.py` B2.** Fixed gradient sequences that do
  not depend on the weights (open loop): 250 steps, 40 encoder coordinates
  whose metric gradients have scales spread log-uniformly over two decades,
  20 flow coordinates. Its numbers describe AdamW's arithmetic on given
  gradients, not a training run.
- **Equations.** The plan's are "plan eq. (n)", displayed with the tag
  (plan n); P5's are (P5.n), P3's (P3.n), E3's (E3.n). E4's own are (E4.1) to
  (E4.9).
- Logarithms are natural. ASCII only, LF only.

---

## 2. Glossary

Ordered by first appearance; the section where each term becomes operative
is given last.

- **Stream** -- one of the three batch sources a step draws from (simulated,
  metric, replicate), each with its own random generator. S3.2.
- **Empirical risk** -- the mean loss over a finite set of rows; what a loop
  that samples a fixed bank minimises in expectation, as distinct from the
  expectation under the law the bank was drawn from. S3.2.
- **Class-balanced sampling** -- drawing the same number of rows from each
  class for every batch, whatever the classes' sizes. *Not* a change of the
  classes' counts in the bank; it reweights each class's share of the batch.
  S3.2.
- **Sample-wise layer** -- a layer whose output for one row depends on that
  row alone; GroupNorm is one, BatchNorm in training is not. S3.3.
- **Gradient reach** -- the weights on which a term's gradient can be
  non-zero. S3.3.
- **Stop-gradient** -- treating a quantity as a constant in the backward
  pass. *Two senses* (S3.9): the stack's, which holds the flow's weights and
  $\bar C$ fixed inside the replicate term; and the Siamese self-supervised
  device against representational collapse. S3.3.
- **Reparameterised draw** -- a sample written as a differentiable function
  of the distribution's parameters and of independent noise, so a gradient
  passes through it (`rsample`); `.sample()` detaches. S3.3.
- **Decoupled weight decay (AdamW)** -- the shrink of every weight applied
  directly to the weights rather than through the gradient (P5). S3.4.
- **Global-norm gradient clipping** -- rescaling the whole gradient vector,
  all parameters together, so that its 2-norm does not exceed a threshold
  (P5). S3.4.
- **Scale invariance of Adam** -- multiplying every gradient of a run by one
  positive constant leaves the update unchanged up to $\epsilon_{\rm adam}$
  (P5 eq. (P5.4)). S3.4.
- **Saturation** -- here, the flattening of the encoder's update as a
  function of $\lambda_{\rm dsn}$ at both ends of its range. S3.4.
- **Gradient conflict, negative transfer** -- two terms whose gradients on
  shared weights point in opposed directions (a negative cosine); the loss of
  performance on the primary task this can cause (Xie et al., S6). S3.5.
- **Clopper-Pearson interval** -- the exact binomial confidence interval,
  obtained by inverting two one-sided binomial tests, each at half the
  complement of the confidence level (2.5 % for a 95 % interval); it keeps
  both one-sided non-coverage rates at or below that level for every true
  proportion, and is wide for it (Lyles et al., S6). S3.5.
- **Early stopping, best-state restore** -- ending training after a number
  of epochs without improvement of a validation score, and loading back the
  weights of the best epoch (P5). S3.6.
- **Online learning (BayesFlow's sense)** -- simulating fresh training data
  at every step, so that no training example is seen twice; the stack
  instead samples a fixed bank with replacement. S3.6.
- **Transductive (domain adaptation)** -- a training scheme that includes
  the target domain's unlabelled inputs; Kushibar et al.'s definition: "the
  images without expert annotations from unseen domain are included in the
  training process" (S6). S3.7.
- **Pseudo-real endpoint** -- the held-out NLL on the bench's arm
  $\mathcal{R}$, whose $\theta$ is recorded and withheld from every training
  loss (plan D10). S3.7.
- **Ablation** -- a run that removes or replaces one component of the
  objective or of its training, to measure what that component contributes.
  S3.8.
- **Warm start** -- initialising a network with another run's weights
  (`A3` from `A0`'s $\psi^\star$). S3.8.
- **Domain effect, objective effect** -- the two terms of plan eq. (9): what
  changes when the metric loss moves from real to simulated windows at fixed
  objective, and what changes when the objective changes at fixed (simulated)
  domain. S3.8.

---

## 3. Main body

### 3.1 The discrepancy: one objective, two weights, and a loop that does not read them as weights

This section establishes the question the chapter answers and the running
example it answers it on.

The plan's objective is one sum, eq. (plan 1) below: the NPE term on
simulated windows, plus $\lambda_{\rm dsn}$ times the metric term on
labelled windows, plus $\lambda_{\rm rep}$ times the replicate term on
same-donor pairs (plan S2.2, lines 195-209) `[REPO]`, and the deck reads the
weights as deciding "how hard each pull is" (deck B.1, speaker notes)
`[KB]`. At the runner's default
$\lambda_{\rm dsn} = 0.1$ (`run_joint_arms.py:222`) `[REPO]` the natural
reading is that the label pulls on the encoder a tenth as hard as the
likelihood, and that the search's range $[10^{-3}, 10]$ (P2 S3.3.2) spans
four decades of that pull. What the loop does with the number differs in
three ways, each established below. The flow's weights $\omega$ never see
it, except through the gradient clip (S3.4). The encoder's weights $\psi$
see it only against the sizes of the two terms' gradients, coordinate by
coordinate, so that its effect flattens at both ends (S3.4). And the one
quantity that would place a run on that curve -- the realised gradient-norm
ratio the plan asks every trial to record (plan S5.1, lines 1092-1094)
`[REPO]` -- is computed inside the loop and dropped (S3.5, F-bd).

Behind this sits a second gap, of level (R8): eq. (plan 1) is an
expectation under three laws, and the loop estimates it under laws of its
own -- the rows of a training split, a class-balanced batch law, the
enumerated pairs (S3.2). The arms are ablations of this objective (S3.8),
so what each can isolate depends on all of the above.

**Running example.** DUP15HD: $d_\theta = 26$, $E = 12$ at the search
anchors, $C = 2$, 1890 real windows -- 918 of the control class and 972 of
the pathological one (17 and 18 wells of 54 windows each) -- and 29,616
simulated rows after the activity floor `[KB]` (`HPC_PATHS.md`, the cohort
manifest's rows; `SBI_PIPELINE.md` S5-S6). The Giulia cohort where class
sizes matter: $C = 3$ with 72, 108 and 108 windows per class under D-056's
tiling `[KB]` (decision log D-056, counts by the tiling rule there). The
runner's schedule: $B_{\rm sim} = 128$, $B_{\rm met} = 32$,
$B_{\rm rep} = 4$, $n_{\rm ep} = 10$, $n_{\rm step} = 25$, so
$n_{\rm plan} = 250$ (`run_joint_arms.py:215-219`) `[REPO]`, `[RAN]` B0. The
bench where a number needs a built bank: one default shard of the `bench`
provider, 512 rows, $W = 3000$, $d_\theta = 10$, $C = 3$ with 160, 160 and
192 rows per class `[RAN]` B1, B4.

### 3.2 The objective, and the laws the loop estimates it under

This section establishes what eq. (plan 1) is an expectation of, what the
loop's per-step sum estimates instead, and the three sampling laws behind
it.

S3.1 left the question of what the weights multiply; the first thing they
multiply is a set of expectations, and the loop never draws from the laws
those expectations are taken under. The plan's objective and its first two
terms are (plan S2.2) `[REPO]`

$$\mathcal{L}(\psi, \omega) \;=\; \mathcal{L}^{\rm sim}_{\rm NPE}(\psi, \omega) \;+\; \lambda_{\rm dsn}\, \mathcal{L}^{\rm real}_{\rm DSN}(\psi) \;+\; \lambda_{\rm rep}\, \mathcal{L}^{\rm real}_{\rm rep}(\psi, \omega), \tag{plan 1}$$

$$\mathcal{L}^{\rm sim}_{\rm NPE}(\psi, \omega) = \mathbb{E}_{p_{\rm sim}}\big[-\log q_\omega\big(\theta \mid h_\psi(x)\big)\big], \qquad \mathcal{L}^{\rm real}_{\rm DSN}(\psi) = \mathbb{E}_{p_{\rm real}}\big[\ell_{\rm DSN}\big(h_\psi(x),\, c\big)\big], \tag{plan 1a}$$

the first expectation over $(\theta, x)$ drawn from $p_{\rm sim}(\theta, x)$,
the second over $(x, c)$ drawn from $p_{\rm real}(x, c)$. The loop sums, at
every step, one batch from each of three streams -- P5 eq. (P5.1), which is
plan eq. (2) as P3 eq. (P3.9) writes it: the mean of $\ell_i$ over
$\mathcal{B}_{\rm sim}$, plus $\lambda_{\rm dsn}\, \ell_{\rm DSN}$ on one
metric batch, plus $\lambda_{\rm rep}\, r_{\rm rep}(t)$ times the mean of
$\ell_{\rm rep}$ over the step's pairs (`joint_train.py:164-201`) `[REPO]`.
The streams draw as follows (`joint_batches.py:168-199`; the sources from
`run_joint_arms.py:496-508`) `[REPO]`:

| stream | per step | drawn from | within the step | feeds | generator |
|---|---|---|---|---|---|
| simulated | $B_{\rm sim}$ rows | the $N_{\rm train}$ rows of the simulated training split (`:172`) | uniform, with replacement | the NPE term | $s_{\rm seed} + 1$ |
| metric | $C n_c$ rows, $n_c = \max(1, \lfloor B_{\rm met}/C \rfloor)$ | every row of the real bank (`A2`, `A3`); the simulated training split with the generator's labels (`A2s`) (`:176-188`) | $n_c$ rows of each class, uniform with replacement within the class | $\ell_{\rm DSN}$ | $s_{\rm seed} + 2$ |
| replicate | $\min(B_{\rm rep}, N_{\rm pair})$ pairs | the $N_{\rm pair}$ same-donor pairs enumerated over every row of the real bank (`:37-58, 148-164`) | uniform, without replacement | the replicate term | $s_{\rm seed} + 3$ |

Successive steps draw independently, each stream from its own generator
(`:126-128`) `[REPO]`. The encoder-only pre-training of `A0` and `A0s` is
not a stream of the loop: it draws $B_{\rm met}$ rows uniformly from every
row of its source, the real bank or the whole simulated bank, before the
loop starts (S3.3; F-be, S3.8). Taking the expectation of the step's sum
over these draws, for each fixed $\varpi$,

$$\mathbb{E}\big[\hat{\mathcal{L}}_{\rm step}(\varpi)\big] \;=\; \frac{1}{N_{\rm train}} \sum_{i=1}^{N_{\rm train}} \ell_i(\varpi) \;+\; \lambda_{\rm dsn}\, \mathcal{L}^{\rm real}_{\rm DSN}(\psi) \;+\; \lambda_{\rm rep}\, r_{\rm rep}(t)\, \frac{1}{N_{\rm pair}} \sum_{(g, g')} \mathbb{E}\big[\ell_{\rm rep}\big], \tag{E4.1}$$

with $i$ over the training rows, $(g, g')$ over the enumerated pairs, the
inner expectation over the posterior draws of the replicate term, and the
metric term defined by

$$\mathcal{L}^{\rm real}_{\rm DSN}(\psi) \;=\; \mathbb{E}_{\mathcal{Q}_{\rm met}}\Big[\ell_{\rm DSN}\big(h_\psi(X^{\rm real}_{\rm met}),\, y^{\rm real}_{\rm met},\, \mathcal{T}_{\rm mined}\big)\Big], \qquad \mathsf{w}_c \;=\; \frac{1/C}{N_c \big/ \sum_{c'} N_{c'}}, \tag{E4.2}$$

where under $\mathcal{Q}_{\rm met}$ each class contributes $n_c$ rows drawn
uniformly with replacement from its $N_c$ rows, and $\mathsf{w}_c$ is the
factor by which a class's share of every batch exceeds its share of the
source `[reasoning, from the sampler at :176-188]`. The first term of
(E4.1) follows because each simulated draw is uniform over the training rows
(the mean of a uniform draw is the mean of the rows) and the third because a
uniform draw without replacement makes every pair equally likely to sit in
the batch, so the batch mean is unbiased for the mean over pairs
`[textbook, from memory]`.

Three readings follow, one per term.

**The NPE term is an empirical risk.** The loop minimises, in expectation
over its draws, the mean loss over the training rows. That mean estimates
$\mathcal{L}^{\rm sim}_{\rm NPE}$, the expectation under $p_{\rm sim}$, as
any sample mean estimates its law's expectation (R8: "estimates", not
"equals"); the gap between the two is what the held-out scores measure
(E2, S3.6 below). On DUP15HD a run at the runner's defaults makes 1.54
pass-equivalents of the training rows and leaves 21 % of them unseen (P5
Table P5.1) `[RAN]` (P5).

**The metric term's expectation needs a batch law, and the code's is
class-balanced.** Plan eq. (1a) applies $\ell_{\rm DSN}$ to one window and
its label, but $\ell_{\rm DSN}$ is a function of a batch: it mines triplets
inside the batch and forms the batch's class means (P2 eq. (P2.10); E3 S3.4)
`[REPO]`. Its expectation is therefore defined only together with the law of
the batch it is computed on (R1), and in the stack that law is
$\mathcal{Q}_{\rm met}$, eq. (E4.2) -- not the plan's "DSN's own
class-balanced real batch, built by the existing sampler and collator" (plan
S2.2) `[REPO]`, which carries augmentation surrogates and, under
`cross_culture`, a rule against same-culture positives (P2 S3.5's
differences table). Class balance makes every class weigh the same in every
batch whatever its size. On DUP15HD that barely moves anything: the factors
are 1.0294 for the control class and 0.9722 for the pathological one. On
the Giulia cohort it does: class 0's share is raised by a third, 1.3333,
and each AraC class's lowered to 0.8889 `[RAN]` B1. So
$\mathcal{L}^{\rm real}_{\rm DSN}$ as trained is the plan's expectation
under $p_{\rm real}(x, c)$ only when the classes are equally represented in
the source `[reasoning]`.

**The batch is not a set of distinct windows, and the real windows are
revisited.** Within a class the draws are with replacement, so one window
can appear twice in a batch: on DUP15HD a class's 16 draws repeat a window
with probability 0.123 (control) and 0.117 (pathological), on the Giulia
cohort 0.480 for class 0's ten draws and 0.349 for an AraC class's
`[RAN]` B1. Over a run's 250 steps one window of a class is drawn on average
$n_{\rm plan} n_c / N_c$ times: 4.36 and 4.12 on DUP15HD, 34.7 and 23.1 on
the Giulia cohort, 15.6 and 13.0 on a bench shard -- 4 % more where the
gradient-cosine probe draws one more batch per epoch (S3.5) `[RAN]` B1. The
plan's "the real cohort is revisited roughly 58 times per epoch" (S5.1,
lines 1098-1101) `[REPO]` describes an epoch the code does not run (F-ag,
P5); the per-window count over a run is the quantity the overfitting worry
of that paragraph needs, and the record keeps nothing that would show it
(S3.6).

The objective is also not constant in time: the replicate term is ramped by
$r_{\rm rep}(t)$, which reaches 1 at step 75 of 250 at the runner's
`--warmup-frac-rep 0.3` (P3; P5 S3.6), and $\ell_{\rm DSN}$ carries its own
ramp $\lambda_{\rm sep}(t)$, constant in the joint space (P2). Before the
ramp completes, the third term of (E4.1) is a fraction of its weight.

### 3.3 Gradient reach

This section establishes which weights each term's gradient can move, the
two places the stack cuts a gradient on purpose, and why the three streams
can interact only through the weights.

S3.2 fixed what each term estimates; the weights then move along the
gradient of the sum, and a term moves only the weights it reaches. Three
facts settle the reach.

**The layers are sample-wise.** The backbone is GroupNorm throughout, and
the stack refuses any BatchNorm at construction
(`joint_model.py:40-54`, called at `:78` and `:208`) `[REPO]`, so the
embedding of a row depends on $\psi$ and on that row alone, whatever else
occupies the batch; the three streams are encoded in separate calls
(`joint_train.py:166, 173, 186-187`) `[REPO]`. They therefore meet only
through the shared weights -- the plan's first "mechanical fact" (S2.2,
lines 211-219) `[REPO]`, tested by J9 (`smoke_test_joint.py:431-513`; not
run here) `[REPO]`.

**Each term reaches a definite set of weights.** The metric term reads the
embedding alone, so it reaches $\psi$ and never $\omega$ (J8,
`smoke_test_joint.py:395-424`) `[REPO]`. The NPE term evaluates the flow at
the embedding, so it reaches both. The replicate term evaluates the flow
too, and would reach both, but the loop computes it inside
`stop_grad_params(model.flow_parameters())`, which switches the flow's
weights to `requires_grad=False` for that forward pass
(`joint_train.py:182-188`; `joint_losses.py:91-122`) `[REPO]`, and the
covariance $\bar C$ that builds its metric and target is detached
(`joint_losses.py:205-206, 309`) `[REPO]`. What is left is a gradient into
$\psi$ through the difference of the two wells' posterior means, carried by
reparameterised draws -- `rsample_posterior` reaches past sbi's public API
to call the flow's `.rsample()`, because `.sample()` detaches
(`joint_model.py:143-166`) -- and the loop raises if the term arrives
without a gradient (`joint_train.py:189-195`) `[REPO]`. Why the two cuts
exist -- the escape routes each blocks -- is P3's and E5's subject.

In symbols, writing $\Gamma^{\rm NPE}_\tau$, $\Gamma^{\rm DSN}_\tau$ and
$\Gamma^{\rm rep}_\tau$ for the gradients of the three unweighted terms with
respect to $\varpi$ at step $\tau$, P5's gradient is their weighted sum,
$\Gamma_\tau = \Gamma^{\rm NPE}_\tau + \lambda_{\rm dsn}\, \Gamma^{\rm DSN}_\tau + \lambda_{\rm rep}\, r_{\rm rep}(t)\, \Gamma^{\rm rep}_\tau$,
and its two blocks are

$$\nabla_\omega \hat{\mathcal{L}}_{\rm step} \;=\; \nabla_\omega\Big(\frac{1}{B_{\rm sim}} \sum_{i \in \mathcal{B}_{\rm sim}} \ell_i\Big), \qquad \nabla_\psi \hat{\mathcal{L}}_{\rm step} \;=\; \nabla_\psi\Big(\frac{1}{B_{\rm sim}} \sum_{i \in \mathcal{B}_{\rm sim}} \ell_i\Big) + \lambda_{\rm dsn}\, \nabla_\psi \ell_{\rm DSN} + \lambda_{\rm rep}\, r_{\rm rep}(t)\, \nabla_\psi\Big(\frac{1}{\lvert \mathcal{B}_{\rm rep} \rvert} \sum_{(g, g') \in \mathcal{B}_{\rm rep}} \ell_{\rm rep}\Big), \tag{E4.3}$$

the last gradient taken with the flow's weights and $\bar C$ held fixed,
so $\Gamma^{\rm DSN}_\tau$ and $\Gamma^{\rm rep}_\tau$ are zero on the
coordinates of $\omega$. The two zeros have different standing (R2): the
metric term's is a property of the objective, since $\ell_{\rm DSN}$ is a
function of $z$ alone; the replicate term's is a property of the
implementation, the stop-gradient, and plan eq. (1) writes the term with
both arguments, $\mathcal{L}^{\rm real}_{\rm rep}(\psi, \omega)$.

**The frozen arms train the flow alone.** On `A0`, `A0s` and `A_ref` the
encoder is frozen before the loop starts -- weights and evaluation mode
(`joint_model.py:80-107`) -- and only the `requires_grad` tensors enter
AdamW (`joint_train.py:140-147`) `[REPO]`; the loop then has only the NPE
term (both weights are 0 for these arms, S3.8). `A0`'s and `A0s`'s encoder
is fitted earlier by a second optimiser, `train_encoder_only`
(`run_joint_arms.py:150-168`): 200 steps of $\ell_{\rm DSN}$ alone on
batches of $B_{\rm met}$ rows drawn uniformly from every row of the source
(F-bc, E3), AdamW at torch's own decay and betas (F-q, P5), no clip, no
validation `[REPO]`. Its last gradients stay on the frozen tensors and enter
the joint loop's clip (F-ah, P5).

### 3.4 What a loss weight does under AdamW and the clip

This section establishes what $\lambda_{\rm dsn}$ and $\lambda_{\rm rep}$
change in an AdamW update: for given gradient histories, nothing on the
flow's weights except through the clip's variation from step to step, and
on the encoder's a mixture that saturates at both ends, with the transition
set coordinate by coordinate by the sizes of the two terms' gradients.

S3.3 says which coordinates a weight multiplies; what the update does with
the product is AdamW's arithmetic, and AdamW does not add gradients the way
the weights suggest. Adam divides each coordinate's averaged gradient by the
root of its averaged square (P5 eq. (P5.4)), so it reads the shape of a
coordinate's gradient history, not its scale: multiplying every gradient of
a run by one positive constant leaves the update unchanged up to
$\epsilon_{\rm adam}$ (P5 S3.2) `[RAN]` (P5). The two blocks of (E4.3) then
behave as mirror images. On a coordinate of $\omega$, a loss weight
multiplies nothing: the gradient there is the NPE term's alone, whatever
$\lambda_{\rm dsn}$ and $\lambda_{\rm rep}$, so the update is the same
function of the same history -- unless the clip rescales that history by
factors the weights influence. On a coordinate of $\psi$, conversely, the
weight multiplies one of two summands: where $\lambda_{\rm dsn}$ times the
metric gradient is small against the NPE gradient, the update is the NPE
term's own; where it is large, the update is the metric term's own, and the
exact value of $\lambda_{\rm dsn}$ no longer matters, because Adam forgets
the scale.

Write AdamW's step (P5 eqs. (P5.3)-(P5.4)) with the normalised step named,

$$\varpi_\tau = (1 - \eta \gamma_{\rm wd})\, \varpi_{\tau - 1} - \eta\, \mathsf{u}_\tau, \qquad \mathsf{u}_\tau = \frac{\bar\mu^{(1)}_\tau}{\sqrt{\bar\mu^{(2)}_\tau} + \epsilon_{\rm adam}} \ \ \text{elementwise}. \tag{E4.4}$$

**On the flow's coordinates.** The moments average the clipped gradient
$\tilde\Gamma_{\tau'} = \varrho_{\tau'} \Gamma_{\tau'}$, whose $\omega$-block
is $\varrho_{\tau'} \Gamma^{\rm NPE}_{\tau'}$ by (E4.3). So, coordinate by
coordinate of $\omega$,

$$\mathsf{u}_\tau = \frac{\sum_{\tau'=1}^{\tau} (1 - \beta_1)\, \beta_1^{\tau - \tau'}\, \varrho_{\tau'}\, \Gamma^{\rm NPE}_{\tau'} \,\big/\, (1 - \beta_1^{\tau})}{\Big(\sum_{\tau'=1}^{\tau} (1 - \beta_2)\, \beta_2^{\tau - \tau'}\, \varrho_{\tau'}^{2}\, \big(\Gamma^{\rm NPE}_{\tau'}\big)^{2} \,\big/\, (1 - \beta_2^{\tau})\Big)^{1/2} + \epsilon_{\rm adam}} \quad \text{on the coordinates of } \omega, \tag{E4.5}$$

and the loss weights appear nowhere in it except through $\varrho_{\tau'}$.
A clip coefficient that is the same at every step cancels between
numerator and denominator, up to $\epsilon_{\rm adam}$, so only its
variation from step to step reaches $\omega$. The statement holds for given
gradient histories (R2); in a run, $\Gamma^{\rm NPE}_{\tau'}$ on $\omega$
is evaluated at embeddings $h_\psi(x)$ whose $\psi$ the weights did move,
so the loss weights reach the flow a second way, through the encoder's
trajectory `[reasoning]`. On the toy of S1.1 `[RAN]` B2, whose weights start
at zero so that the final weights are the displacement: without the clip,
the flow's final weights are bitwise equal for $\lambda_{\rm dsn} = 0$,
0.1, 1, 10 and 1000. With the clip at $\gamma_{\rm clip} = 5$ over all
parameters, as the loop applies it (`joint_train.py:208-210`) `[REPO]`: at
$\lambda_{\rm dsn} \le 0.1$ it never fires; at 0.3 it fires on 4.4 % of the
steps and moves the flow's displacement by 0.04 %; from
$\lambda_{\rm dsn} = 1$ it fires on every step, with a coefficient whose
standard deviation is 15.5 % to 17.0 % of its mean, and moves the
displacement by 2.4 % to 2.6 % (cosine with the clip-free displacement
0.99984 or more) -- the same at 10 and at 100, where the coefficient's level
is ten times lower each time and its relative variation the same.

**On the encoder's coordinates.** The history the moments average is
$\varrho_{\tau'}\big(\Gamma^{\rm NPE}_{\tau'} + \lambda_{\rm dsn}\Gamma^{\rm DSN}_{\tau'} + \lambda_{\rm rep}\, r_{\rm rep}(t)\, \Gamma^{\rm rep}_{\tau'}\big)$,
with $t = (\tau' - 1)/n_{\rm plan}$ at step $\tau'$ (P5 eq. (P5.1)). With
the clip off and the histories given -- a bracket names a history, its step
index and $t$ running together --

$$\mathsf{u}_\tau \;\to\; \mathsf{u}_\tau\big[\Gamma^{\rm NPE} + \lambda_{\rm rep}\, r_{\rm rep}(t)\, \Gamma^{\rm rep}\big] \ \ (\lambda_{\rm dsn} \to 0), \qquad \mathsf{u}_\tau \;\to\; \mathsf{u}_\tau\big[\Gamma^{\rm DSN}\big] \ \ (\lambda_{\rm dsn} \to \infty), \quad \text{on the coordinates of } \psi, \tag{E4.6}$$

the second on every coordinate where the metric history is not identically
zero, because (E4.4) is homogeneous of degree zero in the history up to
$\epsilon_{\rm adam}$ `[reasoning, from (P5.3)-(P5.4)]`. Between the two
limits a coordinate changes regime where $\lambda_{\rm dsn}$ times the size
of its metric gradient is comparable with the size of its NPE gradient, so
the transition is spread over as many decades as the ratio of the two sizes
spreads across coordinates `[reasoning]`. On the toy `[RAN]` B2, whose
per-coordinate ratios of root-mean-square NPE to metric gradient run from
0.085 to 15.4 (median 0.80): one coordinate of 40 is nearer the metric-only
update than the NPE-only one at $\lambda_{\rm dsn} = 10^{-1.5}$, half of them
at 0.734, all of them at $10^{2}$; the cosine of the update with the
NPE-only update falls from 1.0000 at $10^{-3}$ to 0.0766 at $10^{3}$, where
its cosine with the metric-only update is 1.0000, and the two cosines cross
at 0.449. The ratio of the two gradients' median per-step norms, 0.304,
sits off the half-way point by a factor 2.4: a norm is dominated by the
coordinates where its gradient is largest. A global ratio summarises where
a run sits; the per-coordinate ratios decide it.

Four consequences for the stack. First, $\lambda_{\rm dsn}$ means something
only against the gradient sizes of the run it is set in, coordinate by
coordinate; P2 said that the scale is bank-specific (S3.3.2), and AdamW
adds that it is coordinate-specific. Second, whether the ends of the
searched range $[10^{-3}, 10]$ fall inside the two flat regions depends on
those sizes, which no record keeps (S3.5, F-bd); a flat partial dependence
on `log10_lambda_dsn` at one end of the range can be saturation rather than
indifference `[reasoning]`. Third, a large $\lambda_{\rm dsn}$ makes the
clip fire at every step and couples the flow to the metric term through the
coefficient's variation -- weakly in the toy. Fourth, the same arithmetic
holds for $\lambda_{\rm rep}$, with the ramp $r_{\rm rep}(t)$ making the
replicate summand's size change during the first 30 % of training; under
Adam the ramp is not a linear fade-in of that summand's share of the update
`[reasoning]`.

### 3.5 The record of the two terms: $\rho_{\rm grad}$, the dropped norms, and what P4 can resolve

This section establishes what the loop measures about the two terms'
interaction, what it computes and discards, and how finely ten probes per
run can answer prediction P4.

S3.4 placed $\lambda_{\rm dsn}$'s effect on a curve whose position depends
on the two terms' gradient sizes; the loop measures the two gradients once
per epoch, and what it keeps of them decides what can be said afterwards.
When $\lambda_{\rm dsn} > 0$, after each epoch's validation, the loop draws
one more batch from every stream and computes, by two separate backward
passes on that probe batch, the NPE term's and the metric term's gradients
with respect to the encoder's weights, with the separation ramp's counter
rewound after the probe, so that the probe does not advance it -- the
context manager is named `frozen_warmup`, and its docstring warns that the
counter moves inside it and is restored on exit (`joint_train.py:77-101,
234-249`; `dsn_loss_adapter.py:113-147`) `[REPO]`:

$$\rho_{\rm grad} \;=\; \frac{\big\langle \Gamma^{\rm NPE},\, \Gamma^{\rm DSN} \big\rangle_\psi}{\lVert \Gamma^{\rm NPE} \rVert_\psi\, \lVert \Gamma^{\rm DSN} \rVert_\psi}, \qquad \rho_{\rm norm} \;=\; \lambda_{\rm dsn}\, \frac{\lVert \Gamma^{\rm DSN} \rVert_\psi}{\lVert \Gamma^{\rm NPE} \rVert_\psi}, \tag{E4.7}$$

both on the probe batch, at the epoch's end weights. The code computes both
norms (`joint_train.py:98`) and returns the cosine only (`:101`); the epoch
record keeps `rho_grad` (`:246`) `[REPO]`. $\rho_{\rm norm}$ is the plan's
"realised gradient-norm ratio", which "each trial" is to record "in the
ledger. Same treatment for $\lambda_{\rm rep}$" (S5.1, lines 1092-1094)
`[REPO]`; the ledger's fields (`npe_tune_joint.py:517-528`) carry neither it
nor $\rho_{\rm grad}$, and the replicate term's gradient is never computed
apart from the others `[REPO]`. Restricting the cosine to the encoder is, in
Xie et al.'s words, "standard practice in multi-task gradient analysis": they
compute it "over the shared backbone parameters" only, "excluding the
task-specific classification heads", because "the task-specific heads receive
single-task gradients" `[PubMed full text]`
([DOI](https://doi.org/10.3390/s26072088)); the flow is the stack's
task-specific head in this sense `[reasoning]`. They also show how far one
batch's cosine is from its average: over steps 1750 to 4250 of their run the
per-step cosine ranged from about -0.55 to +0.62 while its rolling average
stayed between 0.05 and 0.15 `[PubMed full text]` (ibid.). That is their model
and task; carried here is only that a single batch's cosine can be far from
its average -- the stack's own spread is unmeasured (R3).

**Finding F-bd** (owner E4; the ledger side with P6). The two records of
the two terms' interaction that the plan names are not produced. (a)
$\rho_{\rm norm}$ is computed and discarded (above), so no record places a
run on the curve of S3.4, and neither weight can be read as an effective
weight after the fact. (b) Prediction P4 -- "$\rho_{\rm grad} < 0$ on a
non-trivial fraction of steps in A2", measured by "per-epoch probe",
falsified if it "stays non-negative" (plan line 301) `[REPO]` -- is a
fraction, and nothing computes it: the report prints each run's last probe
only, `r["history"][-1].get("rho_grad")` (`report_joint_arms.py:85`)
`[REPO]`; the ten values per run are in each record's `history`.
Resolution: report; open (a D-038 candidate: return the norms with the
cosine, log $\rho_{\rm norm}$ and its replicate analogue per epoch, carry
their medians and the fraction of negative probes into the ledger and the
report).

**What ten probes can resolve.** Treat a run's probes as independent draws
with a common probability $f_-$ of a negative cosine -- an assumption, since
the weights change between epochs `[reasoning]`. Then

$$\Pr\big(\text{no probe of } n_{\rm cos} \text{ gives } \rho_{\rm grad} < 0\big) = (1 - f_-)^{n_{\rm cos}}, \qquad \hat f_- = \frac{\#\{\text{probes with } \rho_{\rm grad} < 0\}}{n_{\rm cos}}, \tag{E4.8}$$

with $n_{\rm cos} = n_{\rm ep} = 10$ per run at the runner's settings and
$n_{\rm ep} n_{\rm seed} = 50$ pooled over the plan's $n_{\rm seed} = 5$
seeds `[RAN]` B3. The exact interval for $f_-$
from $\hat f_-$ is the Clopper-Pearson interval, obtained "by inverting
the two one-sided tests" of the binomial distribution, each at half the
complement of the confidence level -- to the authors' knowledge "the only
viable CI to maintain strict control of both lower and upper lack of
coverage rates" at that half level for every proportion, and for that
"excessively wide relative to the others" `[PubMed full text]` (Lyles et al.,
[DOI](https://doi.org/10.1080/00949655.2019.1672695)); scipy 1.18.1's
`binomtest(...).proportion_ci(method="exact")` computes it by that
inversion and agrees with the beta-quantile form to $10^{-9}$ `[RAN]` B3. At
95 %: zero negatives of 10 leave $f_-$ anywhere in $[0, 0.3085]$; one of 10
gives $[0.0025, 0.4450]$, three $[0.0667, 0.6525]$, five
$[0.1871, 0.8129]$; pooled over 50 probes, zero gives $[0, 0.0711]$, five
$[0.0333, 0.2181]$ `[RAN]` B3. And P4's falsifier, "stays non-negative",
occurs by chance with probability 0.5987 at $f_- = 0.05$ and 0.3487 at 0.10
over one run's ten probes, 0.0769 and 0.0052 over fifty; the smallest $f_-$
seen at least once with probability 0.95 is 0.2589 in ten probes and 0.0582
in fifty `[RAN]` B3. Since "non-trivial" has no threshold, one run cannot
falsify P4 at any $f_-$ below about a quarter; five seeds pooled can, down
to about 6 %. The implementation session saw one probe at $-0.376$ on a
two-epoch flow (deck ledger D28, `[RAN]` in that session, not a result)
`[KB]`.

### 3.6 The loop as built: one step, one epoch, selection and restore

This section establishes the order of the loop's operations, why the loop is
written out rather than taken from sbi, and what "early stopping on the NPE
validation score only" amounts to at the runner's settings.

S3.5 described the one measurement the loop makes between epochs; the rest
of the epoch boundary is selection, and it is there that the three terms
stop being treated alike. P5 S3.2 writes the step and the epoch as
equations, (P5.1)-(P5.9); in order (`joint_train.py:153-270`) `[REPO]`:
the three batches; the three terms, the ramp set from the steps done before
the replicate forward pass (`:181`); one backward pass; the global clip at
5.0 over every parameter that holds a gradient (`:208-210`); one AdamW step.
After $n_{\rm step}$ steps the epoch record keeps the unweighted means of
the three terms, the replicate statistic and target, the invalid-target
count and the ramp (`:215-225`); the NLL on the selection split, $L_{\rm sel}$
(P5 eq. (P5.7)), and a copy of the weights whenever it improves by more
than $10^{-6}$ (`:227-232`); then the probe (S3.5); then the stopping test
(`:262-267`). At the end the best weights are loaded back (`:269-270`).

**Why an explicit loop.** sbi's `train()` "takes one loss and no labels,
and weight decay and LR schedules are unreachable through it"
(`joint_train.py:3-6`; plan Stage 2, lines 1248-1251) `[REPO]`; the flow is
still built by sbi's `posterior_nn`, so posterior objects downstream are
unchanged. P5 S3.5 compares the two loops knob by knob. BayesFlow, the
end-to-end regime `A1` belongs to, trains its summary and inference networks
"jointly via backpropagation" with the Adam optimiser at a starter
learning rate of $10^{-3}$ and an exponential decay rate of 0.95, over
50,000 to 100,000 iterations, simulating "on demand" so that "the network
never experiences the same input data twice" `[KB-PDF p.6]` (BayesFlow, the
PDF's sixth page, journal p.1457). The stack trains 250 steps at a constant
$\eta$ on a fixed bank sampled with replacement (P5), so its arms are
compared at a far shorter budget than the regime they are named after, and
overfitting the bank is possible where BayesFlow's online scheme excludes it
`[reasoning]`.

**Early stopping on the NPE validation score only.** The loop's reason is
in its docstring: "Stopping on the total loss would let a shrinking
replicate term buy patience for a worsening posterior"
(`joint_train.py:18-20`) `[REPO]`. The rule breaks after epoch $\iota$
when $\iota - \iota_{\rm best} \ge n_{\rm pat}$ (P5 eq. (P5.8)); at the
runner's `patience=99`, $n_{\rm pat} = 99$ (`run_joint_arms.py:522`), it
can first fire after 100 epochs, so at $n_{\rm ep} = 10$ it never does
(F-d, P5); what remains of "early stopping
on the NPE score" is the best-state restore, which still selects the epoch
by $L_{\rm sel}$ alone `[REPO]`. Two consequences. The metric and replicate
terms never choose an epoch, as intended. And nothing watches them on data
they did not train on: the plan's mitigation for the small cohort -- "hold
out cultures for a monitored DSN-validation ARI and for a monitored
replicate-consistency value" (S5.1, lines 1101-1105) `[REPO]` -- is not
built at Stage 3; the metric stream reads every real window (S3.2), which
an `A2` or `A3` run draws 4.3 to 4.5 times each on DUP15HD and 24 to 36
times on the Giulia cohort, the probe's batches included `[RAN]` B1, and an
overfitted metric term would show only as a falling training mean `dsn`
`[reasoning]`. Sahu et al., training a
segmentation network jointly on labelled simulated and unlabelled real
images, give the reason the supervised term must stay primary: "If the
unsupervised consistency loss ... outweighs the supervised loss, the learning
process is not effective and leads to a sub-optimal performance", which they
handle with a weight on the unlabelled term "linearly increased over the
training" `[PubMed full text]`
([DOI](https://doi.org/10.1007/s11548-021-02383-4)) -- the device of the
replicate ramp, though the ramp's stated reason in the stack is that the
replicate statistic starts on its collapse side (`joint_losses.py:259-261`)
`[REPO]`, and their consistency term is a perturbation consistency on
images, not a replicate statistic (R3).

**After the loop.** The runner scores $L$ on the report split and the
pseudo-real endpoint on arm $\mathcal{R}$ (`run_joint_arms.py:536-558`),
and records the last epoch's validation score where the weights are the best
epoch's (F-aj) and the search optimises the report split while the
selection split only picks the epoch (F-al) `[REPO]` (P5).

### 3.7 Two domains in one loop, and the endpoints

This section establishes which rows of which domain each arm's terms read,
what the primary endpoint scores, and in what sense it is held out.

S3.6 named the report split and arm $\mathcal{R}$ as the places an arm is
scored; whether those rows are new to an arm depends on which streams it
draws, and the arms differ there more than their names suggest. On the
bench, arm $\mathcal{S}$ is simulated with $\theta$ available and arm
$\mathcal{R}$ carries $\theta$ recorded but withheld from every training
loss (plan S4.0 and D10; the sidecar's `theta_withheld`,
`run_joint_arms.py:552`) `[REPO]`. The runner splits arm $\mathcal{S}$ by
donor into training, selection and report parts (`:118-143, 369`) and does
not split arm $\mathcal{R}$: every real-arm row feeds whichever real stream
the arm draws, and every real-arm row is scored by the primary endpoint
(`:491-508, 553-556`) `[REPO]`. The table reads arm $\mathcal{R}$ as a
sample of its own, which it is when the two bench arms are built at
different `SEED`s -- F-aw's operating rule (P7); at the job's defaults it is
not, and the end of this section says what changes.

| arm | simulated rows its training reads | real-arm rows its training reads | what of the scored real rows it has seen |
|---|---|---|---|
| `A1`, `shuffled`, `A_ref` | the training split, with $\theta$ (permuted for `shuffled`) | none | nothing |
| `A0` | the training split (flow) | every row, with $c$, in the pre-training | $x$ and $c$ |
| `A0s` | every row with the generator's labels (pre-training, F-be); the training split (flow) | none | nothing |
| `A2`, `A3` | the training split | every row, with $c$, in the metric stream (`A3` also through $\psi^\star$) | $x$ and $c$ |
| `A2s` | the training split, with labels in the metric stream | none | nothing |
| `A5` | the training split | every row in a same-donor pair, in the replicate stream | $x$ and the pairing |

So for `A0`, `A2`, `A3` and `A5` the primary endpoint is held out in $\theta$
and not in $x$: on one bench shard the 6400 draws of `A0`'s pre-training reach
a given real-arm row with probability 0.999996, and an `A2` run draws each
real-arm row about 15 times on average (260 batches of 30 rows over 512 rows)
`[RAN]` B1. This is the transductive scheme of domain adaptation in Kushibar
et al.'s sense: "In the transductive scenario, the images without expert
annotations from unseen domain are included in the training process with the
aim to minimise the domain-shift effect" `[PubMed full text]`
([DOI](https://doi.org/10.3389/fnins.2021.608808)), and joint training on
labelled simulations with unlabelled real data is the simulation-to-real form
of it (Sahu et al., S3.6) `[PubMed full text]`. Here the target domain's
inputs are unlabelled only in $\theta$: `A0`, `A2` and `A3` also use their
class labels. On the cohort this mirrors deployment -- the recordings and
their labels are in hand, $\theta$ never is -- so the endpoint asks the
deployment question `[reasoning]`. What it does not measure is generalisation
to cultures an arm has not seen; that needs the culture holdout of plan D9
("Cycle the real cohort every step ... with a frozen culture holdout?", open,
line 1575) and of Stage 6, which reads real windows "with the holdout frozen
and hashed **before** training" (lines 1389-1392) `[REPO]` -- not built at
Stage 3. The word carries both senses (S3.9).

Two further facts change what the endpoint scores at the bank's defaults.
Bench arms built at one `SEED` share every draw -- $\theta$, class, the
realisation seeds -- so at $\pi = 0$ arm $\mathcal{R}$'s windows are byte
copies of arm $\mathcal{S}$'s, and 68.75 % of the pseudo-real rows are
copies of rows every arm's flow trains on, $\theta$ included (F-aw, P7)
`[RAN]` (P7); at $\pi > 0$ the two arms differ only through the gap, at the
same $\theta$. The copies reach the pre-trainings too: `A0s` reads every
arm-$\mathcal{S}$ row, so at $\pi = 0$ its pre-training and `A0`'s read
byte-identical arrays with one seed, and in the code the two runs then
coincide step for step -- the simulated stream's generator does not depend
on the metric source (`joint_batches.py:126-128`), and both arms draw their
metric batches unused -- so eq. (9)'s domain term is zero by construction
there: the right value without a gap, but not a measurement of it
`[reasoning, from the source; not run]`. And the bank's identity fields
restart in every shard (F-at, P7), so on a bank of several shards the
replicate stream pairs unrelated donors.

The split of the domains is also why eq. (9) exists: `A0` differs from `A1`
in objective and in the domain its metric loss reads, and `A0s` moves the
domain at fixed objective (S3.8).

### 3.8 The arms as an ablation design

This section establishes what each arm changes against `A1`, which contrast
each serves, what else differs between the two members of each contrast,
and four findings.

S3.7 sorted the arms by what they read; an ablation design reads them in
pairs, and a pair isolates one cause only when everything else is held
fixed. The arms as the runner configures them, from `arm_config`
(`run_joint_arms.py:175-202`) executed at the freeze with the runner's
weights, and the job's index map (`joint_arms.pbs:69-73`) `[RAN]` B5:

| arm | encoder | $\lambda_{\rm dsn}$ | $\lambda_{\rm rep}$ | metric stream in the loop | replicate stream | probe | real bank needed | indices in `-J 0-44` |
|---|---|---|---|---|---|---|---|---|
| `A1` | trained, from scratch | 0 | 0 | none | none | no | no | 0, 9, 18, 27, 36 |
| `A0` | pre-trained by $\ell_{\rm DSN}$ on every real row, frozen | 0 | 0 | drawn, unused | none | no | yes | 1, 10, 19, 28, 37 |
| `A0s` | pre-trained by $\ell_{\rm DSN}$ on every simulated row, frozen | 0 | 0 | drawn, unused | none | no | no | 2, 11, 20, 29, 38 |
| `A2` | trained | 0.1 | 0 | real, every row | none | yes | yes | 3, 12, 21, 30, 39 |
| `A2s` | trained | 0.1 | 0 | simulated training split | none | yes | no | 4, 13, 22, 31, 40 |
| `A3` | trained, from `A0`'s $\psi^\star$ | 0.1 | 0 | real, every row | none | yes | yes | 5, 14, 23, 32, 41 |
| `A5` | trained | 0 | 0.05 | none | pairs over every real row | no | yes | 6, 15, 24, 33, 42 |
| `A_ref` | the eight fixed statistics, no weights | 0 | 0 | none | none | no | no | 7, 16, 25, 34, 43 |
| `shuffled` | trained; $\theta$ permuted in the training split | 0 | 0 | none | none | no | no | 8, 17, 26, 35, 44 |

"Drawn, unused": `A0` and `A0s` keep a metric stream because their metric
domain is set (`:505-506`), but their loop weight is 0, so the batches are
drawn from the stream's own generator and never used `[REPO]`. The contrasts
the plan reads off these arms, with $L(\cdot)$ an arm's primary score
(Stage 3b, lines 1320-1330, and S2.2's table) `[REPO]`:

$$L(\text{A0}) - L(\text{A1}) \;=\; \big[L(\text{A0}) - L(\text{A0s})\big] + \big[L(\text{A0s}) - L(\text{A1})\big], \qquad L(\text{A2}) - L(\text{A2s}), \tag{E4.9}$$

the first being plan eq. (9) -- the total difference as a domain effect plus
an objective effect, in that order -- and the second the joint side's domain
effect, which the plan names after it. The others follow from the
configuration table `[reasoning]`:

| contrast | isolates | what else differs inside it | role |
|---|---|---|---|
| `A0` against `A0s` | the domain of the label loss, at a frozen encoder | pre-training rows: every real row against every simulated row, selection and report splits included (F-be); the real arm's rows seen by `A0` (S3.7) | eq. (9) |
| `A0s` against `A1` | the objective (label loss then frozen, against end-to-end NPE), on simulated windows | the pre-training optimiser (F-q), its stale gradients in the clip (F-ah), its batches (F-bc); `A0s`'s encoder has seen the report split's windows and labels (F-be) | eq. (9) |
| `A2` against `A2s` | the domain of the metric term inside joint training | `A2` has seen the scored real windows, `A2s` has not (S3.7) | joint side |
| `A2` against `A1` | the metric term's addition at $\lambda_{\rm dsn}$ | the probe's extra draws shift `A2`'s simulated batches after the first epoch (F-ak) | -- |
| `A3` against `A2` | the starting point ($\psi^\star$ against a fresh encoder) at one objective | one checkpoint for all of `A3`'s seeds (F-bf) | basin |
| `A3` against `A0` | joint fine-tuning against freezing, from one starting point | as above | basin |
| `A5` against `A1` | the replicate term's addition | the replicate draws' cost and noise (P3) | -- |
| `A1` against `A_ref` | a learned encoder against fixed statistics | the statistics are not the plan's and enter the flow unstandardised (F-bg) | encoder worth |
| every arm against `shuffled` | having learned anything ($\hat\Delta$ against $\delta_{\min}$) | a control of a finalist with either term on still trains with both weights at 0 (F-ao, P6) | G1 |

`A2` nests `A1` in the objective (plan S2.2) `[REPO]`, and in the code at
$\lambda_{\rm dsn} = 0$ an `A2` run would reproduce `A1`'s: the encoder and
the flow are initialised at the same point of torch's stream (`make_backbone`
seeds before building, `:293`), the simulated stream's generator is the
same, the metric stream's draws come from a generator of their own, and with
the weight at 0 the term and the probe are both skipped
(`joint_train.py:170, 234`) `[reasoning, from the source; not run]`. At
$\lambda_{\rm dsn} > 0$ the probe's extra batch per epoch advances the
simulated stream, so the two runs share their first epoch's batches and no
others (F-ak, P5). The same holds of `A5` at $\lambda_{\rm rep} = 0$.

**Precedents.** `A1` is not new. BayesFlow optimises "the parameters
$\psi$ of the summary network jointly with those of the cINN chain via
backpropagation. Thus, training remains completely end-to-end" -- its
$\psi$ in the role of this set's -- and argues
that "Since the summary network is optimized jointly with the inference
network, the learned data representation is encouraged to be maximally
informative for inferring the parameters' posterior" `[KB-PDF p.6, p.13]`
(the PDF's pages; journal pp.1457, 1464); the Practical Guide describes the
same embedding networks (E3 S3.2) `[KB-PDF]`. According to PubMed, Min et
al. train "the parameters of both the embedding network and the normalizing
flow using stochastic gradient descent" -- the `A1` regime -- and, to compare
with a published network, "froze the trained ReLERNN network without dropout
and removed the last fully-connected layer", using its 64-dimensional
output as summary statistics for a separately trained NPE
`[PubMed full text]` ([DOI](https://doi.org/10.1093/genetics/iyag107)).
That frozen network had been trained "with mean squared error loss against"
the true recombination rate, on simulations `[PubMed full text]` (ibid.).
The plan calls this "the published analogue of A0" (S2.2, lines 244-249)
`[REPO]`; it is `A0`'s analogue in schedule -- fit, freeze, decapitate, then
NPE on the frozen features -- and not in signal or domain: its pre-training
signal is regression onto $\theta$ on simulated data, E3's first signal, not
a class loss on real windows (R7) `[reasoning]`.

**Finding F-be** (owner E4; the pre-training with P2). `A0s`'s encoder is
pre-trained on every row of the simulated bank -- the source is the whole
`sim` array (`run_joint_arms.py:452-459`) -- while `A2s`'s metric stream reads
the training split only (`:498-502`) `[REPO]`. On one default bench shard the
split is 352, 80 and 80 rows at every seed from 0 to 4; the pre-training's
6400 draws reach each row 12.5 times on average, so all 80 report rows are
expected among them; on the bank job's 32 shards, at 0.39 draws per row, about
828 of the 2560 report rows are expected among them `[RAN]` B1. So `A0s`'s
secondary endpoint, $L$ on the simulated report split, is scored on windows
whose $x$ and class label trained its encoder, and so is the selection split
that picks its epoch; `A2s`'s encoder never saw either. The class label is
informative about $\theta$ on the bench -- 1.0980 nats of class information at
the bank job's defaults (E3 S3.5) `[RAN]` (E3) -- so this is label information
about the scored rows, not $\theta$ itself. The primary endpoint is untouched
when arm $\mathcal{R}$ is drawn at its own `SEED` (`A0s` never reads it), and
so is eq. (9) on it -- at the job's defaults F-aw decides instead (S3.7); P6,
which compares the rankings by the two endpoints (plan line 303), is not
`[reasoning]`. Resolution: report; open (a D-038 candidate: pre-train `A0s` on
the training split, as `A2s`'s stream reads it).

**Finding F-bf** (owner E4; the job layer with P7). One submission of the
arms job fixes one value of each job variable, and the plan's Stage 3 needs
more than one. `LAMBDA_DSN` is read by `A2`, `A2s` and `A3`, and the plan
asks for `A2` at $\lambda_{\rm dsn} \in \{0.1, 1\}$ (Stage 3, lines
1292-1296); the record is named `<arm>_seed<k>` (`run_joint_arms.py:625`),
with $\lambda_{\rm dsn}$ only inside `config`, and the report reads every
`*_seed*.json` of one directory and groups by arm
(`report_joint_arms.py:34-55, 106-117`) `[REPO]`, `[RAN]` B5: a second
submission at 1 into the same `OUT_DIR` overwrites the 0.1 records of the
three arms, and into another directory needs a report of its own.
`WARM_START_CKPT` is one path for every `A3` element; without it `A3` exits
before training (`:476-478`), and the job's documented array command
passes none (`joint_arms.pbs:22`) `[REPO]`, `[RAN]` B5. The checkpoint
`A3` needs is written by an `A0` element (`run_joint_arms.py:634`), so `A3`
cannot run in the array that trains the `A0` it should start from, and with
one path its five seeds share one $\psi^\star$: its $\sigma_{\rm seed}$ then
omits the variability of `A0`'s pre-training, and only one `A3` seed is
paired with its own `A0` seed `[reasoning]`. Resolution: report; open (run
`A0` first; then `A3` per seed with the checkpoint of the same seed, or let
the runner derive the path from the seed; put $\lambda_{\rm dsn}$ in the record's
name).

**Finding F-bg** (owner E4; the flow's inputs with P4). Two things separate
the reference arm the runner builds from the one the plan describes. The
plan's `A_ref` is "NPE on fixed burst statistics from the ANN repo's
`burst_metrics.py` / `network_burst_detector.py`" (S4.5, lines 1059-1062)
`[REPO]`; the runner's are eight generic statistics of the window computed in
place, "Deliberately crude and deliberately fixed"
(`run_joint_arms.py:75-111`) `[REPO]` -- mean, standard deviation, maximum,
90th percentile, burst fraction, lag-1 autocorrelation, skewness and
roughness. And they reach the flow's conditioners as computed.
`build_joint_model` builds with `z_score_x="none"` (`joint_model.py:198`),
which the runner does not override (`run_joint_arms.py:463-474`), and in sbi
0.27.0 the data z-scoring, when on, is a standardising layer placed *before*
the embedding net (`factory.py:291-292` passes `z_score_x` as the builder's
`z_score_y`; `flow.py:1156-1157, 1391-1412`) `[REPO]`, so no option of the
builder standardises an embedding's output. On one default bench shard, with
torch's `std` (Bessel's correction, the default) and `quantile` (linear
interpolation, the default) (torch 2.10.0
`_torch_docs.py:7122-7124, 10687, 10709-10711`) `[REPO]` transcribed in numpy,
the statistics' standard deviations across rows run from 0.0052 (roughness) to
1.031 (skewness), a factor of 199.7, their means from 0.0128 to 3.757, and the
autocorrelation sits 168 of its standard deviations away from zero `[RAN]` B4.
A learned arm's input to the flow is a unit vector, its coordinates in
$[-1, 1]$ (E1, P1). So the reference arm's conditioners start on inputs whose
scales and offsets differ by two orders of magnitude, an optimisation handicap
the learned arms do not carry, and the runner's "If A1 cannot beat this, the
encoder is the problem, not the objective" (`:22-23`; plan S4.5, lines
1061-1062, says the same) is read against a reference that is not the plan's
and is weakened by its preprocessing as well as by its statistics
`[reasoning]`. Resolution: report; open (a fixed standardisation of the
statistics by the training split's means and standard deviations is not a
tuned reference, and keeps `A_ref` free of trainable weights `[reasoning]`;
whether to use the ANN repository's burst statistics is the plan's call).

**Finding F-bh** (owner E4; P13's reading with E7). Prediction P13 orders
same-donor embedding distances "A5 $<$ A2 $<$ A1, with A0(real) near A2"
(plan line 310), and the plan gives its mechanism: "A0 and A2 obtaining
culture-invariance implicitly through `cross_culture` positives and A5
explicitly through the posterior" (Stage 3, lines 1306-1310) `[REPO]`. The
arms as built have no such positives: the metric loss is called on an
embedding and a class label only (`dsn_loss_adapter.py:105-107`), the
pre-training draws rows uniformly (`run_joint_arms.py:158`) and the metric
stream by class (`joint_batches.py:176-188`) -- no culture, no positive rule
(P2 S3.5) `[REPO]`. The `cross_culture` rule is the standalone DSN's
collator's, which trained the r2 encoder, not the Stage 3 arm named `A0`.
The ordering P13 predicts can still be measured; its stated cause for `A0`
and `A2` is absent `[reasoning]`. Resolution: report (E7 reads P13 with
it).

### 3.9 Common confusions

Words and statements that carry two senses in this chapter (R6), each with
the sense the set uses.

- **"$\lambda_{\rm dsn} = 0.1$ is a tenth of the pull."** Under AdamW a
  weight changes the encoder's update only against the two gradients' sizes,
  coordinate by coordinate, and saturates at both ends, (E4.6); on the
  flow's weights, for a given encoder trajectory, it acts only through the
  clip's variation, (E4.5) (S3.4).
- **"The replicate term trains the flow."** Its gradient reaches $\psi$ only,
  through the difference of the two wells' posterior means, because the
  flow's weights and $\bar C$ are held fixed in its forward pass, (E4.3); by
  definition, plan eq. (1) writes it with both arguments (S3.3).
- **"Stop-gradient."** The stack's: a mask on a partial derivative, which
  keeps the flow's weights and $\bar C$ out of the replicate term's gradient
  (S3.3). The Siamese self-supervised device: a branch of a twin network
  whose gradient is blocked so that two augmented views cannot both collapse
  to a constant -- according to PubMed, in abstracts only, ULD-Net "relies
  solely on stop-gradient operation preventing the network from collapsing"
  and ASTCL "alternately uses the predictor and the stop-gradient to avoid
  model collapse" `[PubMed abstract only]` (S6). The two have different
  objects, and nothing here transfers between them.
- **"Held out."** Held out in $\theta$ -- the pseudo-real endpoint's sense:
  $\theta$ withheld from every training loss -- against held out as a
  window or a culture never seen in training. With arm $\mathcal{R}$ drawn
  at its own `SEED`, the primary endpoint is the first for every arm and
  the second only for the arms that draw no real stream; at the job's
  defaults and $\pi = 0$ even the first fails on 68.75 % of its rows,
  copies of rows every arm's flow trains on, $\theta$ included (S3.7,
  F-aw).
- **"Joint."** Three senses: joint training (one optimiser over $\psi$ and
  $\omega$ together, the arms `A1`-`A5`); the joint law $p_{\rm sim}(\theta, x)$;
  and the `joint` loss type of P2 (margin plus angular hinge). The arm `A2`
  is joint in the first sense whatever its `loss_type`.
- **"Class-balanced."** The batch is balanced; the bank is not. The metric
  term's expectation is over the balanced batch law, (E4.2), which equals
  the plan's expectation under $p_{\rm real}(x, c)$ only for equal classes
  (S3.2).
- **"The DSN's own class-balanced batch."** Plan eq. (2) and deck B.2 name
  the standalone DSN's sampler and collator; the joint stack draws its own
  metric batch, without surrogates and without a positive rule (S3.2; P2
  S3.5).
- **"Epoch."** The plan's pass over the simulated bank against the code's
  25 steps whatever the batch size (F-ag, P5).
- **"Early stopping."** Configured, and unable to fire before 100 epochs at
  the runner's patience; the best-state restore is what selects (S3.6).
- **"A2 nests A1."** In the objective, and in the run at
  $\lambda_{\rm dsn} = 0$; at $\lambda_{\rm dsn} > 0$ the probe's extra draws
  make the runs differ beyond the weight (S3.8).
- **"$\rho_{\rm grad}$."** The cosine of two gradients on one probe batch at
  an epoch's end weights; not a per-step statistic, and not the gradient
  norm ratio $\rho_{\rm norm}$, which is computed and dropped (S3.5).
- **"The gradient-norm ratio."** Weighted by $\lambda_{\rm dsn}$ (the
  plan's "realised", $\rho_{\rm norm}$) or not; global over the encoder or per
  coordinate -- the global one summarises, the per-coordinate ones decide
  (S3.4).
- **"A0."** The status-quo r2 encoder, trained by the standalone DSN, against
  the Stage 3 arm re-created by the joint stack -- another sampler, other
  loss defaults (F-h, P2), 200 steps (S3.3, F-bh).

### 3.10 Check yourself

1. In an `A2` run the clip never fires, and $\lambda_{\rm dsn}$ is doubled
   from 0.1 to 0.2. For a fixed encoder trajectory, what changes in the
   flow's update? And in the encoder's?
2. A Stage 3 `A2` run is over. Why can its record not tell whether
   $\lambda_{\rm dsn} = 0.1$ put the encoder's update near the NPE-only
   limit, near the metric-only limit, or between them -- and what would have
   had to be logged?
3. All ten probes of an `A2` run give $\rho_{\rm grad} \ge 0$. What does that
   say about P4, and what would five seeds with no negative probe say?
4. On the Giulia cohort at the runner's defaults, by what factor does the
   metric stream over-represent class 0 relative to its share of the
   windows, and how many times is one class-0 window drawn over a run?
5. Which arms have seen the pseudo-real endpoint's windows during training,
   and with what labels? In what sense is the endpoint held out for them?
6. Why is `A0s`'s simulated-arm score not comparable with `A2s`'s in the way
   their pseudo-real scores are?

<details>
<summary>Answers (folded)</summary>

1. On the flow, nothing: its gradient is the NPE term's alone, and with the
   clip off the loss weight appears nowhere in (E4.5) (S3.4; `[RAN]` B2:
   bitwise equal across $\lambda_{\rm dsn}$). On the encoder it depends on
   where the coordinates sit: those whose metric gradient is small against the
   NPE gradient barely move; those already metric-dominated barely move
   either; only those in the transition, (E4.6), change direction.
2. Because the two norms on which the position depends are computed in
   `gradient_cosine` and discarded, so $\rho_{\rm norm}$, (E4.7), is in no
   record (F-bd); and even a global ratio would only summarise the
   per-coordinate ratios that decide it (S3.4: on the toy, the ratio of median
   norms 0.304 against the half-way point 0.734). Logging $\rho_{\rm norm}$
   per probe, with a few quantiles of the per-coordinate ratios, would.
3. That $f_-$ is below 0.3085 at 95 % (Clopper-Pearson) -- compatible with
   $f_- = 0.10$, which gives no negative probe in ten with probability
   0.3487 (S3.5; `[RAN]` B3). Five seeds, 50 probes, without a negative
   would bound $f_-$ below 0.0711. P4 has no threshold for "non-trivial", so
   only the pooled record can falsify it, and only for fractions above about
   6 %.
4. By 1.3333: class 0 is a quarter of the windows and a third of every
   batch; a class-0 window is drawn 34.7 times over the 250 steps, 36.1 in
   an arm with the probe (S3.2; `[RAN]` B1).
5. With arm $\mathcal{R}$ drawn at its own `SEED`: `A0` (pre-training: $x$
   and the class), `A2` and `A3` (metric stream: $x$ and the class), `A5`
   (replicate stream: $x$ and the pairing). None saw $\theta$; the endpoint
   is held out in $\theta$, transductive in $x$ (S3.7). At the job's
   defaults and $\pi = 0$ every arm's flow has also trained on copies of
   68.75 % of the scored rows, $\theta$ included (F-aw).
6. `A0s`'s encoder was pre-trained on every simulated row with its label,
   the report split's included, while `A2s`'s metric stream read the
   training split only (F-be). With arm $\mathcal{R}$ at its own `SEED`
   neither reads it, so on the pseudo-real endpoint both are scored on
   windows new to their encoders; at the job's defaults and $\pi = 0$ the
   scored windows are copies of arm $\mathcal{S}$'s, all read by `A0s`'s
   pre-training and 68.75 % by `A2s`'s metric stream (S3.7).

</details>

---

## 4. Summary of results

- The loop minimises, in expectation over its draws, the training split's
  mean NLL, the metric loss under a class-balanced batch law and the mean
  over enumerated pairs, (E4.1); plan eq. (1a)'s metric expectation needs
  the batch law, (E4.2), and equals the plan's only for equal classes --
  factors 1.0294 and 0.9722 on DUP15HD, 1.3333 and 0.8889 on the Giulia
  cohort (S3.2, `[RAN]` B1).
- Within a class the draws repeat a window (0.12 on DUP15HD, up to 0.48 on
  the Giulia cohort), and over a run's 250 steps one real window is drawn
  4.1 to 4.4 times on DUP15HD and 23 to 35 times on the Giulia cohort, 4 %
  more with the probe (S3.2, `[RAN]` B1).
- The streams meet only through the weights (GroupNorm); the metric term
  reaches $\psi$ only by its definition, the replicate term by the
  stop-gradient on the flow's weights and on $\bar C$, (E4.3) (S3.3).
- Under AdamW, for given gradient histories, the loss weights reach the
  flow only through the clip's step-to-step variation, (E4.5) -- in a run
  also through the encoder's trajectory -- and the encoder's update moves
  between the NPE-only and the metric-only limits, (E4.6), coordinate by
  coordinate; on the toy, bitwise equal flows without the clip, a 2.4 % to
  2.6 % shift with it, and a transition from $10^{-1.5}$ to $10^{2}$ with
  its midpoint at 0.734 against a ratio of median norms of 0.304 (S3.4,
  `[RAN]` B2).
- $\rho_{\rm grad}$ is recorded once per epoch and $\rho_{\rm norm}$ is
  computed and dropped, (E4.7); P4's fraction is computed nowhere; ten
  probes bound $f_-$ only below 0.31 and fifty below 0.071, (E4.8) (S3.5,
  `[RAN]` B3; F-bd).
- Early stopping never fires at the runner's patience; selection is the
  best-state restore on $L_{\rm sel}$; no held-out culture watches the real
  terms (S3.6).
- With bench arm $\mathcal{R}$ drawn at its own `SEED`, the primary
  endpoint is held out in $\theta$ and, for `A0`, `A2`, `A3`, `A5`,
  transductive in $x$; at the job's defaults and $\pi = 0$ its rows are
  copies of simulated rows (F-aw), and `A0` and `A0s` coincide in the code,
  so eq. (9)'s domain term is zero by construction (S3.7; read, not run).
- The arms' contrasts, (E4.9), each with what else differs: F-be (`A0s`
  pre-trains on every simulated row), F-bf (one $\lambda_{\rm dsn}$ and one
  checkpoint per submission), F-bg (`A_ref`'s statistics are not the plan's
  and enter the flow unstandardised, their spreads differing by a factor 200),
  F-bh (P13's cross-culture mechanism absent); `A2` reproduces `A1` at
  $\lambda_{\rm dsn} = 0$ (S3.8).

## 5. Open points, caveats, assumptions

- **Nothing here ran in torch.** The loop, the probe and the arms were read,
  not executed: torch is absent from the sandbox and nothing in `hpc/joint/`
  has run on the cluster. The `[RAN]` numbers are sampling arithmetic, a
  numpy replica of torch 2.10.0's AdamW and clip (from `p5_numbers.py`), a
  toy, and a numpy transcription of `FixedStatsSummary` in float64 where the
  runner computes in float32.
- **The toy is open loop.** Its gradients do not depend on the weights, so
  (E4.5)-(E4.6) and B2's numbers describe the optimiser's arithmetic on given
  histories; the size of the clip's coupling and the width of the
  transition in a real run are unknown, and F-bd is why.
- **(E4.8)'s independence is an assumption.** The probes of a run are taken
  at different weights; $f_-$ is a summary of a fraction that can drift over
  training.
- **(E4.1)'s third term is an expectation over draws of a non-linear loss.**
  The replicate loss of a pair depends on its posterior draws through a
  logarithm and an inverse covariance; its expectation is not the loss at
  the exact moments (E5, the inflation from inverting an estimated
  covariance).
- **The culture holdout (plan D9, Stage 6) is a design decision, not a
  defect of Stage 3**: D9 is open, and the endpoint's transductive reading
  is the plan's D10 choice. Generalisation to unseen cultures is not
  measured by any arm at Stage 3.
- **The cohorts' class counts** are the knowledge base's: DUP15HD's 918 and
  972 from 17 and 18 wells of 54 windows (`HPC_PATHS.md`), the Giulia
  cohort's 72, 108 and 108 by D-056's tiling rule, computed there, not
  counted from a bank.
- **F-bd (found in this chapter's reading; owner E4, ledger side P6).** The
  realised gradient-norm ratio is computed and dropped; P4's fraction is not
  computed; the report shows each run's last probe (S3.5). Report; open.
- **F-be (owner E4, with P2).** `A0s` pre-trains on every simulated row;
  `A2s`'s metric stream reads the training split (S3.8). Report; open.
- **F-bf (owner E4, with P7).** One $\lambda_{\rm dsn}$ and one warm-start
  checkpoint per submission; records named by arm and seed (S3.8). Report;
  open.
- **F-bg (owner E4, with P4).** `A_ref`'s statistics are eight generic
  window statistics, not the plan's burst statistics, and reach the flow
  unstandardised (S3.8). Report; open.
- **F-bh (owner E4, with E7).** P13's stated mechanism for `A0` and `A2` is
  absent from the joint stack (S3.8). Report.
- **F-aw at the job's defaults (owner P7).** S3.7's table and the claim
  that the primary endpoint is held out in $\theta$ assume bench arm
  $\mathcal{R}$ drawn at its own `SEED`, F-aw's operating rule. At the job's
  defaults and $\pi = 0$, 68.75 % of the scored rows are copies of rows every
  flow trains on, and `A0` and `A0s` read byte-identical pre-training arrays,
  so eq. (9)'s domain term is zero by construction -- read from the source,
  not run (S3.7).
- **Not read, so not used for any claim:** the PDFs of Twist, ULD-Net and
  ASTCL (abstracts only); the standalone DSN's `ConditionBalancedBatchSampler`
  and `TripletCollator` (P2 did not read them either); Xie et al.'s figures,
  beyond the sentences quoted.
- **Sources' status.** BayesFlow (IEEE TNNLS 2022) is peer-reviewed; the
  five PubMed Central papers are peer-reviewed journal articles; Min et al.'s
  bioRxiv version (PMID 41573957) is superseded by the Genetics article and
  not used. In the PubMed Central texts the connector dropped displayed
  formulas; only prose is quoted.

## 6. References / further reading

**Project knowledge base `[KB]`.** `claude/deck_pack/03_SEC_B_joint_objective.md`
(v1) B.1-B.6; `claude/deck_pack/08_SEC_G_decision_and_diagnostics.md` G.5,
G.6; `claude/deck_pack/12_SOURCES_LEDGER.md` rows B1-B17, D28, G17, G18;
`HPC_PATHS.md` (the cohort manifest's rows: 35 wells, 17 control and 18
pathological; the real arm's 1890 rows, 54 per culture); `SBI_PIPELINE.md`
S5-S6; `claude/SBI_decisions_and_ideas_log.md` D-056; E0 v1.11
(conventions, master table, the eight rows added for E4); E1 S3.6; E2;
E3 S3.2, S3.4-S3.6; P2 S3.3.2, S3.5; P3; P5 S3.2, S3.5-S3.8 and Table P5.1;
P6 (F-ao); P7 (F-at, F-aw).

**Repository `[REPO 834eb41]`.** `hpc/joint/JOINT_DSN_NPE_PLAN_v0_6.md`
(v0.6.5) S2.2 (lines 195-249), predictions (296-316), S4.0 and D10 (999-1001,
1577-1578), S5.1 (1069-1117), Stage 2 (1236-1290), Stage 3 (1292-1318),
Stage 3b (1320-1345), Stage 6 (1382-1395), D9 (1575);
`stage2/joint_train.py:1-271`; `stage2/joint_batches.py:1-216`;
`stage2/joint_model.py:40-233`; `stage2/joint_losses.py:18-37, 91-122,
205-206, 255-336`; `stage2/dsn_loss_adapter.py:89-177`;
`stage2/smoke_test_joint.py:340-513` (J7, J8, J9, J14r);
`stage2/smoke_test_joint_losses.py:177-244` (J14);
`stage3/run_joint_arms.py:1-659`; `stage3/jobs/joint_arms.pbs:1-117`;
`stage3/report_joint_arms.py:34-117`; `stage4/npe_tune_joint.py:517-528`.

**Library sources `[REPO]`, as installed or unpacked.** sbi 0.27.0:
`neural_nets/factory.py:239-248, 291-292`;
`neural_nets/net_builders/flow.py:1078-1170, 1391-1412`;
`utils/sbiutils.py:154-185`. torch 2.10.0: `_torch_docs.py:7122-7124,
7152-7154` (`quantile`, interpolation `linear` by default), `:10687,
10709-10711` (`std`, `correction=1` by default). scipy 1.18.1:
`stats/_binomtest.py` (`proportion_ci(method="exact")`, the
Clopper-Pearson interval by test inversion).

**Project PDFs `[KB-PDF]`, full text, page numbers of the PDF.** Radev ST,
Mertens UK, Voss A, Ardizzone L, Kothe U. *BayesFlow.* IEEE TNNLS
2022;33:1452-1466: p.4 (the joint objective over the summary and inference
networks, their eq. (16)), p.6 (joint optimisation by backpropagation; Adam,
starter learning rate $10^{-3}$, decay 0.95, 50,000-100,000 iterations;
online learning), p.13 (joint optimisation makes the representation
maximally informative). Deistler M, Boelts J, et al. *Simulation-Based
Inference: A Practical Guide* (arXiv 2508.12939v1, preprint): pp.7-8, via
E3. The numbers quoted from BayesFlow are its own training settings, not a
result about this stack.

**Peer-reviewed literature.** According to PubMed, read in full text from
PubMed Central (formulas dropped by the connector; only prose quoted): Min
J, et al. *Neural posterior estimation for population genetics.* Genetics
2026; PMID 42032815, PMC13334119,
[DOI](https://doi.org/10.1093/genetics/iyag107) -- embedding network and
flow trained jointly by SGD; the frozen, decapitated ReLERNN, trained by
mean squared error on the recombination rate, as summaries (S3.8). Xie Z,
et al. *AviaTAD-LGH: a multi-task spatio-temporal action detector with
lightweight gradient harmonization.* Sensors 2026; PMID 41977872,
PMC13074830, [DOI](https://doi.org/10.3390/s26072088) -- gradient cosine on
the shared backbone only; per-step range against rolling average (S3.5).
Sahu M, Mukhopadhyay A, Zachow S. *Simulation-to-real domain adaptation with
teacher-student learning for endoscopic instrument segmentation.* Int J
Comput Assist Radiol Surg 2021; PMID 33982232, PMC8134307,
[DOI](https://doi.org/10.1007/s11548-021-02383-4) -- joint supervised loss
on simulated and consistency loss on unlabelled real data; the ramp on the
unlabelled term (S3.6, S3.7). Kushibar K, et al. *Transductive transfer
learning for domain adaptation in brain magnetic resonance image
segmentation.* Front Neurosci 2021; PMID 33994917, PMC8116893,
[DOI](https://doi.org/10.3389/fnins.2021.608808) -- the transductive
scenario (S3.7). Lyles RH, Weiss P, Waller LA. *Calibrated Bayesian credible
intervals for binomial proportions.* J Stat Comput Simul 2020; PMID
33012882, PMC7531056, [DOI](https://doi.org/10.1080/00949655.2019.1672695)
-- the Clopper-Pearson interval's construction and its two-sided control
(S3.5). No number from these papers enters a result here; Xie et al.'s
cosine range is quoted as theirs. **Abstract only, flagged where used:**
Tian Y, et al. *ULD-Net.* J Opt Soc Am A 2022
([DOI](https://doi.org/10.1364/JOSAA.473657)); Wang N, et al. *Adversarial
spatiotemporal contrastive learning for electrocardiogram signals* (ASTCL).
IEEE TNNLS 2024 ([DOI](https://doi.org/10.1109/TNNLS.2023.3272153)); Wang
F, et al. *Self-supervised learning by estimating twin class distribution*
(Twist). IEEE TIP 2023 ([DOI](https://doi.org/10.1109/TIP.2023.3266169)) --
the Siamese sense of "stop-gradient" (S3.9); Twist avoids collapse
"without specific designs such as asymmetric network, stop-gradient
operation, or momentum encoder" `[PubMed abstract only]`. Retrieved,
abstract only, **not used for any claim**: Aritake T, Hino H. Neural Comput
2026 ([DOI](https://doi.org/10.1162/NECO.a.1525)), a transductive
domain-adaptation definition.

**Searches run `[RAN]`, 2026-10-05.** Earlier exploratory queries of this
turn whose strings were not kept are not listed; every source used is
reached by a listed query.

| source | query | result |
|---|---|---|
| PubMed | "neural posterior estimation" | 21 records, all screened by title: Min et al. 2026 (PMC, read in full and used) and its bioRxiv version (superseded); the other 19 apply or evaluate NPE (epidemiology, population genetics, cell migration, MRI, reflectometry, gravitational waves, missing data) and none trains a summary network with an auxiliary loss; none opened |
| PubMed | gradient conflict cosine similarity multi-task learning | 1 record: Xie et al. 2026 (PMC, read in full and used) |
| PubMed | stop-gradient representation collapse self-supervised | 3 records (ULD-Net, ASTCL, Twist), none with a PMC copy; abstract only, used only as flagged in S3.9 |
| PubMed | embedding network normalizing flow trained jointly summary statistics posterior | 0 records |
| PubMed | auxiliary loss weighting multi-task negative transfer shared encoder | 0 records |
| PubMed | class-balanced sampling mini-batch metric learning triplet loss | 0 records |
| PubMed | transductive learning unlabeled target domain test data seen during training | 0 records |
| PubMed | transductive AND "domain adaptation" | 13 records: Kushibar et al. 2021 (PMC, read in full and used); Aritake and Hino 2026 (abstract only, not used); the others applications (OCT segmentation, structural damage, fault diagnosis, electronic nose, micro-expressions, zero-shot learning) or surveys, not opened |
| PubMed | "unsupervised domain adaptation" simulation-to-real | 1 record: Sahu et al. 2021 (PMC, read in full and used) |
| PubMed | Clopper-Pearson exact binomial confidence interval coverage | 6 records: Lyles et al. 2019 (PMC, read in full and used); Rahbek et al. 2025 and Wang et al. 2024 (PMC, not opened); three without a PMC copy, not used |
| PubMed | multi-task learning early stopping primary task validation loss auxiliary task | 0 records |
| PubMed | domain adaptation simulated and real data shared feature extractor joint training unlabeled real | 0 records |
| bioRxiv | bioinformatics, 2026-09-28 to 2026-10-04, first page of 30 records (the connector has no keyword search; all dated 2026-09-28) | none on joint objectives, multi-task gradients or NPE; no preprint cited |
| bioRxiv | neuroscience, 2026-09-28 to 2026-10-05, first page of 30 records (all dated 2026-09-28) | none on topic; one on transductive population graph networks for multisite fMRI (10.64898/2026.09.15.751719, preprint, abstract preview only; "transductive" in the graph sense), not used |
| data repositories | -- | E4 makes no claim about a public dataset; its data numbers concern the project's banks (KB, code); no data-repository connector is among this session's tools; none queried |
| KB PDFs | BayesFlow, read in full for the passages cited | as cited above, with pages |

**Textbook, from memory** (tagged where used): the mean of a uniform draw
from a finite set is the set's mean; a uniform draw without replacement
gives each member the same inclusion probability, so its sample mean is
unbiased for the population mean.

---

### Pre-send check (Precision model)

R1 types: every symbol of S1 has its type; $\ell_{\rm DSN}$ is a function of a
batch, so its expectation is over a batch law, $\mathcal{Q}_{\rm met}$, and
plan eq. (1a)'s per-window form is read through it; $\mathsf{w}_c$ is a ratio
of shares, $\rho_{\rm norm}$ a ratio of norms on the encoder's coordinates,
$f_-$ a probability and $\hat f_-$ a fraction; the toy's "global ratio" is
named as what it is, a ratio of median per-step norms. R2 hypotheses: (E4.1)
is for each fixed $\varpi$ and over one step's draws; the replicate term's
zero on $\omega$ carries "by the stop-gradient", the metric term's "by
definition"; (E4.5) and (E4.6) carry given gradient histories and, for (E4.6),
the clip off and a non-zero metric history, and the abstract, S3.4's opening,
S3.9 and S4 now say so; (E4.8) carries independent probes with a common $f_-$;
"A2 reproduces A1" carries $\lambda_{\rm dsn} = 0$ and is read from source,
not run; S3.7's table, F-be's "the primary endpoint is untouched" and the
answers to questions 5 and 6 carry bench arm $\mathcal{R}$ drawn at its own
`SEED` (F-aw's operating rule), and "A0 and A0s coincide at $\pi = 0$" carries
the job's shared `SEED` and is read, not run; the toy's numbers carry the open
loop. R3 transplants: Xie et al.'s cosine range is theirs and only "a batch's
cosine can be far from its average" is carried; Sahu et al.'s ramp is the same
device with another reason; the Siamese stop-gradient is not the stack's;
BayesFlow's schedule is not the stack's; Min et al.'s frozen network is `A0`'s
analogue in schedule only. R4 names: $\rho_{\rm norm}$ is not
$\rho_{\rm grad}$; $\mathsf{u}_\tau$ is P5's quotient named, not a new update;
$N_c$ (source rows) is not $n_c$ (batch rows); "the status-quo A0" and "the
Stage 3 arm A0" are distinguished; BayesFlow's $\psi$ is quoted in the role of
this set's. R5 maps: $h_\psi : \mathbb{R}^{W} \to S^{E-1}$; gradients live in
$\mathbb{R}^{n_\varpi}$ and are restricted to the coordinates of $\psi$ or
$\omega$ explicitly. R6 senses: stop-gradient, held out, joint,
class-balanced, epoch, early stopping, A0, the gradient-norm ratio (S3.9). R7
borrowed phrasing: the deck's "the weights decide how hard each pull is" is
given the AdamW reading; the plan's "the published analogue of A0" keeps "in
schedule"; "pseudo-real held-out NLL" keeps "in $\theta$"; "early stopping on
the NPE validation score" keeps "never fires at the runner's patience"; Lyles
et al.'s "only viable CI" keeps their "to our knowledge", Xie et al.'s
"standard practice" stays theirs, and the probe's "frozen" ramp is described
as the code does it, rewound on exit. R8 levels: $\mathcal{L}$ and its terms,
$\mathcal{Q}_{\rm met}$ and $f_-$ analytic; $\hat{\mathcal{L}}_{\rm step}$,
the per-term gradients, $\varrho_\tau$, $\mathsf{u}_\tau$, $\rho_{\rm grad}$,
$\rho_{\rm norm}$, $\hat f_-$, $L$ and $L_{\rm sel}$ computed; the training
split's mean NLL estimates $\mathcal{L}^{\rm sim}_{\rm NPE}$, and $\hat f_-$
estimates $f_-$; the 828 report rows of F-be are an expectation over the
pre-training's draws, not a count. Fixes made while writing: the A0s finding
was first drafted as a leak into eq. (9) and narrowed after the table of S3.7
showed that `A0s` never reads arm $\mathcal{R}$, and narrowed again on
re-reading: that holds with arm $\mathcal{R}$ at its own `SEED`, while at the
job's defaults F-aw, not F-be, decides what eq. (9) measures; the real-arm
culture holdout was first drafted as a finding and is recorded as plan D9's
open design (S5); the toy's clip coupling was first expected to peak where the
clip fires on some steps only, and is stated as B2 shows it: flat from
$\lambda_{\rm dsn} = 1$, where the clip fires at every step; the per-window
draw counts of S3.6 now include the probe's batches, which an arm with the
metric term always draws.
