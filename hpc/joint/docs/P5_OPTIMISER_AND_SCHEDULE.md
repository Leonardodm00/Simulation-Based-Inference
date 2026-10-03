# P5 -- The optimiser block: `lr`, `one_minus_beta1`, `weight_decay`, `batch_size_npe`, and the schedule the runner fixes around them

**Document P5 of the joint documentation set.** Owner of the `optimiser`
block of `JOINT_KNOB_ORDER` (`lr`, `one_minus_beta1`, `weight_decay`,
`batch_size_npe`) and of the schedule knobs the joint stack holds fixed or
sets by code: `epochs`, `steps_per_epoch`, the two unsearched batch sizes
`b_met` and `b_rep` (owned by P2 and P3, placed here in the step they share),
`encoder_steps`, the gradient-norm clip `grad_clip`, `beta2` (the space's
fixed `one_minus_beta2`), the early-stopping `patience`, the grouped
train / selection / report split and its fractions, the run `seed` and the
generators it feeds, and the constants torch fixes below any flag. Master
notation: E0. The chapter that explains the loop as an experimental design
-- three streams, two domains, the arms -- is E4. **Date:** 2026-10-03 (v1.1).
**Applies to:** the repository `Simulation-Based-Inference` at `834eb41`,
`hpc/joint/` (D-037): `stage2/joint_train.py`, `stage2/joint_batches.py`,
`stage3/run_joint_arms.py`, `stage3/jobs/joint_arms.pbs`,
`stage4/joint_space.py`, `stage4/npe_tune_joint.py`,
`stage4/jobs/joint_tune.pbs`, and the objects the joint stack reads its
ranges and defaults from, `dsn/config.py` (`TrainConfig`, `SearchConfig`,
`RegularizationConfig`), `npe_tune_search.py`, `npe_model.py`,
`npe_tune_train.py`; the plan `JOINT_DSN_NPE_PLAN_v0_6.md` at its repository
version **v0.6.5**; the installed `sbi` 0.27.0 wheel (its training loop, the
one the explicit loop replaces: `inference/trainers/base.py`,
`inference/trainers/npe/npe_base.py`); and the optimiser and the clip of
`torch` 2.10.0, read from the GitHub source at tag `v2.10.0`
(`torch/optim/adam.py`, `torch/optim/adamw.py`,
`torch/nn/utils/clip_grad.py`); `sbi_env` imports torch 2.10.0 (`[KB]`
`HPC_PATHS.md` sec. 7, 2026-10-01). The sandbox has no torch, so nothing
here was run through the optimiser itself: every number is a closed form or a
numpy replica of the torch source, `tools/p5_numbers.py` `[RAN]`.

| date | change |
|---|---|
| 2026-10-03 | v1.1. One correction, nothing else changed: the multilevel SBI paper (Hikida et al.) was flagged PREPRINT, not peer-reviewed, at its two citations (S3.5, S6); the project PDF's p.1 carries the NeurIPS 2025 conference line, so both are marked [corrected 2026-10-03]. Evidence: `[KB-PDF p.1]`, read in the P7 turn (P7 S6). |
| 2026-10-02 | v1. Written from `stage2/joint_train.py` and `stage2/joint_batches.py` (read in full), `stage3/run_joint_arms.py:75-168, 205-271, 334-382, 440-534, 578-634`, `stage3/jobs/joint_arms.pbs:3-4, 50-117`, `stage4/jobs/joint_tune.pbs:3-4, 60-161`, `stage4/joint_space.py:30-50, 119-145, 212-331, 441-472, 697-722, 761-814`, `stage4/npe_tune_joint.py:82-253, 266-296, 342-348, 444-535`, `dsn/config.py:695-770, 835-875, 882-934, 1176-1197`, `dsn/Documentation/TUNING_1_searched_axes.md` S3.5-S3.7, `TUNING_2_fixed_knobs.md` S3.2, S3.4, the DSN JSON configs under `dsn/hpc/Config/` (the optimiser ranges and the refit values), `npe_tune_search.py:95-170, 295-330`, `npe_model.py:76-100, 196-215`, `npe_tune_train.py:60-100, 225-260`; the `sbi` 0.27.0 wheel (`inference/trainers/base.py:280-290, 384-400, 474-540, 1037-1110, 1121-1165, 1225-1255`; `inference/trainers/npe/npe_base.py:252-320`); torch 2.10.0 (`optim/adamw.py:20-97`, `optim/adam.py:139-160, 396-440, 528-547`, `nn/utils/clip_grad.py:121-232`); the plan S2.2, S2.4, S5.1 ("Epoch semantics"), Stage 2, Stage 3; `[KB]` `JOINT_DSN_NPE_USAGE_v1.md` S5, deck `03_SEC_B` B.4; `[KB-PDF]` the practical guide, Goncalves et al. 2020, BayesFlow, the compositional-SBI paper, the flow-matching paper, the trust-crisis paper, the multilevel preprint and the RVNP paper, pages as cited in S6. Every number of S3.1-S3.5 recomputed by the new torch-free `tools/p5_numbers.py` `[RAN]`, run twice with identical output. Findings F-ag to F-al added; F-c, F-d, F-g, F-i, F-l, F-m, F-q, F-r owned or co-owned. Grounding searches of S6 run and reported. |

**Abstract.** Four axes of the joint search -- the learning rate, the
complement of AdamW's first decay rate, the decoupled weight decay and the
simulated batch size -- and a schedule the runner fixes around them decide
how far the weights $(\psi, \omega)$ travel per step, how many steps a run
takes, which rows feed them and which epoch's weights are kept. The question
this document answers is, for each of these and for the constants torch and
the runner fix below them: where the value is set and by which surface it
reaches the trainer (S3.1); what one optimiser step and one run compute,
written out from `joint_train.py` and from torch 2.10.0's own source as one
explicit function of the knobs, from the three-stream step loss through the
clip, the two moment averages, the bias correction and the decoupled decay to
the per-epoch validation, the stopping rule and the restore,
eq. (P5.1)-(P5.9) (S3.2); what each axis changes in that function (S3.3);
what the fixed knobs commit a run to (S3.4); how the block relates to the
two stacks it borrows from -- the standalone DSN search, whose `SearchConfig`
supplies two of the ranges (D-036), and sbi's own `train()`, whose loop the
explicit loop replaces (S3.5); how it interacts with the other blocks (S3.6);
how it fails and which number reveals each failure (S3.7); and the findings
this document owns (S3.8): F-c, F-d, F-g, F-i, F-l, F-m, F-q, F-r, and six
new ones -- the code's epoch is a fixed number of steps drawn with
replacement while the plan's, the batch-size trim rule's and sbi's is a pass
over the data (F-ag); on the two-stage arms the gradient clip measures the
frozen encoder's stale pre-training gradients together with the flow's
(F-ah); within the searched ranges the decoupled decay shrinks a weight by
at most half a percent over the runner's 250 steps (F-ai); the record's
`val_npe` is the last epoch's score while the saved weights are the best
epoch's (F-aj); the per-epoch gradient-cosine probe advances the
simulated stream, so arms with and without the DSN term see different
simulated batches at the same seed (F-ak); and the search objective and
the arm endpoint `L` are scored on the report split, the selection split
serving early stopping alone, where the plan and E0 assign the selection
split to the comparison and touch the report split once (F-al). **Deliberately excluded:** why
maximum likelihood on simulated pairs targets the posterior and what the
held-out NLL bounds (E2, E4); the three loss terms beyond how they enter one
step (P2, P3, P4); the search mechanics that propose the configurations
(P6), except where the schedule enters a control run; the job scripts'
resources and the Stage 3b/3c flags (P7); any claim about how fast or how
well a configuration trains -- no job of `hpc/joint/` has run on the cluster
(`[KB]` usage v1.3 S9), and every number here is a property of the loop as
written, recomputed from the code and the two libraries' source or a closed
form marked so.

---

## 1. Notation and symbols

A subset of E0's master table (same symbol, same type, same units) plus the
fifteen symbols this document adds, which E0 v1.5 declares in its
"Optimiser and schedule (P5)" group. Status in the Provenance model's sense
is given per knob in S3.1 and S3.4.

| Symbol | Name / Meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| $\psi$, $\omega$ | encoder and flow weights | $\mathbb{R}^{n_\psi}$, $\mathbb{R}^{n_\omega}$ | mixed | S3.2 |
| $n_\psi, n_\omega$ | number of encoder and flow weights | $\mathbb{N}$ | -- | S3.2 |
| $\varpi$ | the trainable weight vector: the concatenation of the `requires_grad` tensors of $(\psi, \omega)$; $\varpi_\tau$ its value after step $\tau$, $\varpi_0$ the initial value; not $\varphi$ (spline parameters), not $\pi$ | $\varpi \in \mathbb{R}^{n_\varpi}$ | mixed | S3.2 |
| $n_\varpi$ | its length: $n_\psi + n_\omega$ when nothing is frozen, $n_\omega$ on the frozen arms | $\mathbb{N}$ | -- | S3.2 |
| $\tau$, $\tau'$ | optimiser step index: the $\tau$-th call of `opt.step()` of one run, AdamW's own counter; $\tau' \le \tau$ an earlier step of the same run; $t = (\tau - 1)/n_{\rm plan}$ is the progress the ramps read; not $\tau_j$, $\tau_{\rm sep}$, $\tau_{\rm ov}$ | $\tau, \tau' \in \{1, \dots, n_{\rm plan}\}$ | steps | S3.2 |
| $t$ | training progress, as a fraction of the planned optimiser steps | $t \in [0, 1]$ | -- | S3.2 |
| $n_{\rm ep}, n_{\rm step}$ | epochs, and optimiser steps per epoch (`epochs`, `steps_per_epoch`) | $\mathbb{N}$ | -- | S3.1 |
| $n_{\rm plan}$ | the planned step budget, $n_{\rm ep} n_{\rm step}$ in the joint loop, `--encoder-steps` in the pre-training | $\mathbb{N}$ | steps | S3.2 |
| $n_{\rm done}$ | steps completed so far, $n_{\rm done} = \tau - 1$ at step $\tau$ | $\mathbb{N}_0$ | steps | S3.2 |
| $\iota$ | epoch index of the joint loop (the loop's `epoch`) | $\iota \in \{0, \dots, n_{\rm ep} - 1\}$ | -- | S3.2 |
| $\iota_{\rm best}$ | the epoch with the lowest $L_{\rm sel}$ so far (`best_epoch`); $-1$ before any validation | $\{-1, 0, \dots, n_{\rm ep} - 1\}$ | -- | S3.2 |
| $n_{\rm pat}$ | early-stopping patience in epochs (`patience`) | $\mathbb{N}$ | epochs | S3.2 |
| $\eta$ | the AdamW learning rate (`lr`) | $\mathbb{R}_{>0}$ | dimensionless | S3.1 |
| $\beta_1, \beta_2$ | AdamW exponential decay rates | $(0, 1)$ | dimensionless | S3.2 |
| $\upsilon_1, \upsilon_2$ | their complements, $\upsilon_1 = 1 - \beta_1$ (`one_minus_beta1`, searched), $\upsilon_2 = 1 - \beta_2$ (`one_minus_beta2`, fixed); TUNING_1 writes $u_1, u_2$, letters taken here by the Cholesky vector $u$ | $(0, 1)$ | dimensionless | S3.1 |
| $\gamma_{\rm wd}$ | the AdamW decoupled weight-decay coefficient (`weight_decay`) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.1 |
| $\gamma_{\rm clip}$ | the global gradient-norm clip threshold (`grad_clip`) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\epsilon_{\rm adam}$ | AdamW's denominator constant (torch `eps`); not $\epsilon_{\rm jit}$, not $\varepsilon$ | $\mathbb{R}_{>0}$ | mixed | S3.2 |
| $\mathcal{H}_1, \mathcal{H}_2, \mathcal{H}_{\rm wd}$ | averaging horizons of the two moment averages, $\mathcal{H}_1 = 1/\upsilon_1$, $\mathcal{H}_2 = 1/\upsilon_2$, and the decay horizon $\mathcal{H}_{\rm wd} = 1/(\eta \gamma_{\rm wd})$ | $\mathbb{R}_{>0}$ | steps | S3.2 |
| $B_{\rm sim}, B_{\rm met}, B_{\rm rep}$ | rows per optimiser step in the simulated, metric and replicate streams (`--b-sim`, `--b-met`, `--b-rep`); `batch_size_npe` is $B_{\rm sim}$ | $\mathbb{N}$ | rows; pairs for $B_{\rm rep}$ | S3.1 |
| $\mathcal{B}_{\rm sim}$ | one i.i.d. prior-faithful minibatch of simulated rows | index set of size $B_{\rm sim}$ | -- | S3.2 |
| $\hat{\mathcal{L}}_{\rm step}$ | the per-step estimate of $\mathcal{L}$, plan eq. (2): the three batch terms of one optimiser step | $\mathbb{R}$ | nats/row plus dimensionless terms | S3.2 |
| $\mathcal{L}$, $\mathcal{L}^{\rm sim}_{\rm NPE}$ | the joint objective, plan eq. (1), and its NPE term (analytic level: expectations) | $\mathbb{R}$ | nats/row | S3.2 |
| $\ell_{\rm DSN}$, $\lambda_{\rm dsn}$, $\lambda_{\rm rep}$ | the metric loss of one batch and the two weights of plan eq. (2) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\ell_{\rm rep}$ | the per-pair replicate loss as the code evaluates it, P3 eq. (P3.7) (computed level) | $\mathbb{R}$ | dimensionless | S3.2 |
| $r_{\rm rep}$ | the warm-up ramp of the replicate term, $r_{\rm rep}(t) = \min(1, t / t_{\rm warm})$ for $t_{\rm warm} > 0$ and $1$ for $t_{\rm warm} = 0$ | $[0, 1]$ | dimensionless | S3.2 |
| $\mathcal{B}_{\rm rep}$ | the replicate minibatch: the pairs drawn at one optimiser step, $\lvert \mathcal{B}_{\rm rep} \rvert = \min(B_{\rm rep}, N_{\rm pair})$ | set of pairs | -- | S3.2 |
| $g, g'$ | two wells (cultures) of one donor | indices into the real bank | -- | S3.2 |
| $N_{\rm pair}$ | same-donor well pairs available | $\mathbb{N}$ | pairs | S3.2 |
| $C$ | number of classes (`n_classes`) | $\mathbb{N}$ | -- | S3.2 |
| $\Gamma_\tau, \tilde\Gamma_\tau$ | the gradient of $\hat{\mathcal{L}}_{\rm step}$ with respect to $\varpi$ at step $\tau$ before and after clipping; not $\gamma_{\rm wd}$, $\gamma_{\rm sep}$, not $\mathcal{G}$ | $\mathbb{R}^{n_\varpi}$ | mixed | S3.2 |
| $\varrho_\tau$ | the clip coefficient of step $\tau$; not $\rho_{\rm grad}$ | $(0, 1]$ | dimensionless | S3.2 |
| $\mu^{(1)}_\tau, \mu^{(2)}_\tau, \bar\mu^{(1)}_\tau, \bar\mu^{(2)}_\tau$ | AdamW's exponential moving averages of the clipped gradient and of its elementwise square after step $\tau$, and their bias-corrected forms (the bar is torch's normalisation, not E0's computed-level hat); not $\mu_j$ | $\mathbb{R}^{n_\varpi}$ | mixed | S3.2 |
| $\rho_{\rm grad}$ | cosine between the NPE and DSN gradients with respect to $\psi$ on one probe batch | $[-1, 1]$ | -- | S3.2 |
| $\ell_i$, $L$, $L_{\rm sel}$ | per-row held-out NLL; its mean over a split; the mean over the selection split (the per-epoch `val_npe`, the quantity early stopping watches) | $\mathbb{R}$ | nats, nats/row | S3.2 |
| $L_0$ | the prior floor, $L_0 = -\mathbb{E}_{p_\Theta} \log p_\Theta(\theta)$ | $\mathbb{R}$ | nats/row | S3.3.1 |
| $\hat\Delta$ | the information gain, $\hat\Delta = L_0 - L$ | $\mathbb{R}$ | nats/row | S3.8 |
| $\hat T_{gg'}$, $\hat p_{\rm eff}$ | the replicate statistic and the effective-direction count as the code computes them per pair (computed level; P3), whose batch means the epoch record carries | $\mathbb{R}$ | dimensionless | S3.2 |
| $N_{\rm train}$ | rows of the training split (the search's `n_train` anchor) | $\mathbb{N}$ | rows | S3.2 |
| $G_{\rm don}$ | number of distinct donors (groups of the split) | $\mathbb{N}$ | -- | S3.2 |
| $\varsigma_{\rm tr}, \varsigma_{\rm sel}, \varsigma_{\rm rep}$ | the three split fractions of `grouped_split`, applied to the count of groups, not of rows; not $\sigma_b$, $\sigma_w$, $\sigma_{\rm seed}$ | $[0, 1]$, summing to 1 | -- | S3.2 |
| $s_{\rm seed}$ | the run seed (`--seed`); the stream generators take $s_{\rm seed} + 1$, $+ 2$, $+ 3$; not $s$ (draw index) | $\mathbb{N}_0$ | -- | S3.2 |
| $\sigma_{\rm seed}$, $n_{\rm seed}$ | across-seed standard deviation of an arm's held-out NLL; seeds per arm | $\mathbb{R}_{\ge 0}$, $\mathbb{N}$ | nats/row; -- | S3.6 |
| $t_{\rm warm}$ | fraction of training before $\lambda_{\rm rep}$ ramps in (`warmup_frac_rep`) | $[0, 1)$ | -- | S3.6 |
| $S_{\rm mc}$ | posterior draws per well in the replicate term | $\mathbb{N}$ | draws | S3.6 |
| $d_\theta$, $E$, $W$ | parameter dimension, embedding dimension, window length | $\mathbb{N}$ | -- | S3.2 |
| $h_\psi$, $q_\omega$ | the encoder and the flow | maps | -- | S3.2 |
| $\theta_i$, $x_i$, $z_i$ | one simulated row's parameters, window and embedding $z_i = h_\psi(x_i)$ | $\Theta$, $\mathbb{R}^{W}$, $S^{E-1}$ | -- | S3.2 |
| $n$ | a generic count, always qualified in prose | $\mathbb{N}$ | -- | S3.2 |

### 1.1 Conventions

1. **Two senses of "epoch", declared.** In the joint loop an epoch is
   $n_{\rm step}$ optimiser steps, each on $B_{\rm sim}$ rows drawn
   independently *with replacement* from the training split
   (`joint_train.py:157`; `joint_batches.py:172`) -- a fixed count of steps
   whatever the split's size or the batch size. In sbi's `train()`, in the
   standalone DSN trainer at its default `batches_per_epoch = 0`, and in
   the plan's S5.1 ("One epoch is one pass over the simulated bank"), an
   epoch is one pass over the training rows. The document writes "epoch"
   for the code's sense and "pass" for the other, and keeps the two apart
   wherever a count of epochs is compared across stacks (S3.2, S3.5, F-ag).
2. **Step index versus progress.** $\tau$ counts calls of `opt.step()` from
   1 (torch's `state["step"]`); the loop's own `step` variable is
   $n_{\rm done} = \tau - 1$, and the progress handed to the replicate ramp
   is $t = n_{\rm done}/n_{\rm plan}$ (`joint_train.py:181`), so the first
   step runs at $t = 0$ and the last at $t = 1 - 1/n_{\rm plan}$. Bias
   corrections use $\tau$.
3. **"Weight decay" is the decoupled kind.** AdamW multiplies every
   trainable weight by $(1 - \eta \gamma_{\rm wd})$ before the gradient step
   (torch `adam.py:417-419`); it is not an $L_2$ penalty added to the loss,
   and it does not enter the moment averages. Where a source means the
   penalty, the document says so.
4. **"Clip" is the global 2-norm rescale** of torch's `clip_grad_norm_`:
   one coefficient $\varrho_\tau \le 1$ multiplies every gradient tensor the
   call sees (`clip_grad.py:165-182`). Not a per-coordinate clamp
   (`clip_grad_value_`), which nothing in the stack calls.
5. **Levels (R8).** $\mathcal{L}$ and $\mathcal{L}^{\rm sim}_{\rm NPE}$ are
   expectations (analytic level); $\hat{\mathcal{L}}_{\rm step}$ is one
   step's batch estimate (computed level, a different batch every step);
   $L_{\rm sel}$ is the mean of $\ell_i$ over the whole selection split at
   the end of an epoch (computed level, one number per epoch); $L$ is the
   same mean over the report split at the end of the run. The move from
   $\hat{\mathcal{L}}_{\rm step}$ to $\mathcal{L}$ is "in expectation over
   the batch draws at fixed weights"; the move from $L_{\rm sel}$ to
   $\mathcal{L}^{\rm sim}_{\rm NPE}$ is "in expectation over the rows of the
   split". The moments $\mu^{(1)}_\tau$ and $\mu^{(2)}_\tau$ are computed
   objects of the run (optimiser state); their bias-corrected forms carry a
   bar, which here is torch's normalisation by $1 - \beta_1^{\tau}$ or
   $1 - \beta_2^{\tau}$ and not the
   set's computed-level hat.
6. **Status, per surface.** A knob's status (configured / analytic /
   computed / derivation-only) is given per surface in S3.1 and S3.4; the
   same name can carry a different default on each surface (F-m), and the
   runner's value is what a Stage 3 or Stage 4 run trains with.
7. **Defaults per surface, in a fixed order:** the space (range and prior),
   the runner flag, the job variable, the library `TrainConfig`
   (`joint_train.py`), sbi's `train()`, torch's `AdamW`, the DSN's
   `TrainConfig` / `SearchConfig` / `RegularizationConfig`. A value named
   without its surface is the runner's.
8. **Tags.** As the set uses them: `[REPO file:line]` at `834eb41`;
   `[REPO sbi wheel]`, `[REPO torch v2.10.0]` for the two libraries' source;
   `[RAN]` with the block of `tools/p5_numbers.py` that prints the number;
   `[KB]`, `[KB-PDF p.n]`, `[PubMed full text]`, `[PubMed abstract only]`,
   `[textbook, from memory]`, `[reasoning]`.

## 2. Glossary

Ordered by first appearance.

- **Explicit loop.** `joint_train.train_joint`: the stack's own training
  loop, written because sbi's `train()` takes one loss and no labels and
  exposes neither weight decay nor a schedule (`joint_train.py:1-6`; plan
  Stage 2). The flow is still built by sbi's `posterior_nn` (P4). (S3.1)
- **Surface.** A place a knob's value can be set: the search space, a
  runner flag, a job variable, a library default (P0 S3.1). (S3.1)
- **Stream.** One of the three batches an optimiser step consumes: `sim`
  ($B_{\rm sim}$ simulated rows, the NPE term), `met` ($B_{\rm met}$
  class-balanced rows, the DSN term), `rep` ($B_{\rm rep}$ same-donor pairs,
  the replicate term), each drawn from its own generator (P2, P3). (S3.1)
- **AdamW.** torch's `Adam` with `decoupled_weight_decay=True`
  (`adamw.py:20-49`): per coordinate, a bias-corrected average of past
  gradients divided by the square root of a bias-corrected average of past
  squared gradients, times $\eta$; the weight is first multiplied by
  $(1 - \eta \gamma_{\rm wd})$. (S3.2)
- **Exponential moving average, averaging horizon.** The recursion
  $\mu^{(1)}_\tau = \beta_1 \mu^{(1)}_{\tau - 1} + (1 - \beta_1) \tilde\Gamma_\tau$
  weights the gradient of step $\tau'$ by $(1 - \beta_1)\beta_1^{\tau - \tau'}$;
  the weights sum to one over an infinite past, and $\mathcal{H}_1 = 1/(1 - \beta_1)$ steps is the horizon
  over which they are spent (the half-life is $\ln 2 / (-\ln \beta_1)$); the
  same with $\beta_2$ and $\mathcal{H}_2$ for the squared gradient. (S3.2)
- **Bias correction.** Division by $1 - \beta_1^\tau$ (and $1 - \beta_2^\tau$),
  which makes the average of the first $\tau$ gradients unbiased when the
  moments start at zero. (S3.2)
- **Decoupled weight decay.** The multiplicative shrink of every trainable
  weight by $(1 - \eta \gamma_{\rm wd})$ per step, applied outside the moment
  averages. Its *horizon* is the step count after which a weight untouched
  by gradients would have shrunk by the factor $\exp(-1)$:
  $\mathcal{H}_{\rm wd}$. (S3.2)
- **Gradient-norm clipping.** Scaling the whole gradient so that its 2-norm
  is at most $\gamma_{\rm clip}$; a no-op when the norm is already below.
  (S3.2)
- **Loss-scale invariance.** The property that multiplying every term of
  the loss by one positive constant leaves the AdamW update unchanged up to
  $\epsilon_{\rm adam}$; a *per-step* rescaling, which is what clipping does,
  does not cancel. (S3.2)
- **Epoch, pass.** See convention 1. (S3.2)
- **Selection split, report split.** The 15 % of donors whose rows score
  each epoch ($L_{\rm sel}$) and the 15 % whose rows score the run ($L$,
  which the tuner reads as its objective `nll`); the training split is the
  other 70 % (`grouped_split`). The plan assigns the roles the other way
  round (F-al). (S3.2)
- **Early stopping, patience, best-state restore.** The loop breaks when
  $n_{\rm pat}$ epochs have passed since the best $L_{\rm sel}$, and at the
  end loads the weights of the best epoch whether or not it broke early.
  (S3.2)
- **Coverage.** The expected fraction of the training rows that $n$
  independent draws with replacement visit at least once,
  $1 - (1 - 1/N_{\rm train})^{n}$. (S3.2)
- **Warm-up ramp.** The linear ramp of $\lambda_{\rm rep}$ over the first
  $t_{\rm warm}$ of the planned steps (P3), and of the DSN separation term
  over `sep_warmup_frac` (P2, fixed 0); both read $n_{\rm plan}$. (S3.6)
- **Gradient-cosine probe.** `gradient_cosine`: once per epoch, on one extra
  batch, the cosine $\rho_{\rm grad}$ between the NPE term's and the DSN
  term's gradients with respect to $\psi$ (plan P4). (S3.2)
- **Grouped split.** `grouped_split`: a permutation of the distinct donors
  at the run seed, cut at $\mathrm{round}(\varsigma_{\rm tr} G_{\rm don})$
  and $\mathrm{round}(\varsigma_{\rm sel} G_{\rm don})$; every row follows
  its donor. (S3.2)
- **Seed reach.** The set of random objects one `--seed` fixes: the split,
  the encoder's initialisation, the three stream generators, the dropout
  masks and posterior draws of the loop, the shuffled control's
  permutation. (S3.4)
- **Control run.** A finalist's twin with the $(\theta, z)$ pairing permuted
  (`control_config_from`); the recipe copies every optimiser and schedule
  value (P6). (S3.6)
- **Linear scaling rule (SGD).** The observation, for plain stochastic
  gradient descent, that training is nearly unchanged when the learning
  rate and the batch size change in proportion; stated here only as the
  literature states it, for SGD. (S3.3.4, S3.6)
- **Trim rule.** `default_joint_space`'s removal of a batch size $b$ from
  `batch_size_npe` when `n_train` $< 20 b$ (`joint_space.py:304-306`),
  active only when `--n-train` is passed. (S3.3.4)

## 3. Main body

### 3.1 Where the optimiser knobs live, and what reaches the trainer

*Establishes, for the four searched axes and the schedule knobs around
them, the surfaces each lives on, what a Stage 3 and a Stage 4 run actually
train with, and the four readings of the table that the rest of the document
uses.*

One would expect the optimiser of a training stack to be configured in one
place. Here a learning rate has six homes with five values (space, runner,
library `TrainConfig`, sbi's loop, torch's default, the DSN's base config),
the batch size five, and the early-stopping patience three -- and the runner,
which passes every value explicitly, decides which one trains (P0 S3.1).

| knob | symbol | space (`JointSpaceSpec`, `joint_space.py:238-241`; prior `:456-471`) | reaches the runner as (`AXIS_TO_FLAG`, `npe_tune_joint.py:97-100`) | runner default (`run_joint_arms.py`) | job variable (`joint_arms.pbs:50-62`; `joint_tune.pbs:60-76`) | library `TrainConfig` (`joint_train.py:41-44`) | sbi `train()` (`npe_base.py:254-259`) | torch `AdamW` (`adamw.py:24-27`) | status |
|---|---|---|---|---|---|---|---|---|---|
| `lr` | $\eta$ | $[10^{-4}, 2 \times 10^{-3}]$, log-uniform | `--lr` | `1e-3` (`:220`) | none | `5e-4` | `5e-4` | `1e-3` | configured (space / flag) |
| `one_minus_beta1` | $\upsilon_1$ | $[10^{-2}, 10^{-1}]$, log-uniform | `--one-minus-beta1` | `0.1` (`:258`) | none | `beta1 = 0.9` | not exposed (Adam's 0.9) | `0.9` | configured |
| `weight_decay` | $\gamma_{\rm wd}$ | $[10^{-5}, 10^{-2}]$, log-uniform | `--weight-decay` | `0.0` (`:221`) | none | `0.0` | not exposed (Adam, none) | `1e-2` | configured |
| `batch_size_npe` | $B_{\rm sim}$ | $\{256, 512, 1024\}$, categorical; trimmed by `--n-train` | `--b-sim` | `128` (`:217`) | `B_SIM=128` (Stage 3 only) | `BatchSpec.b_sim = 512` | `training_batch_size = 200` | -- | configured |
| `one_minus_beta2` | $\upsilon_2$ | **fixed** $10^{-3}$ (`spec.fixed`, `:324`), not passed by `build_argv` | no flag | none (F-i) | none | `beta2 = 0.999` | Adam's 0.999 | `0.999` | configured by code |
| `epochs` | $n_{\rm ep}$ | not an axis | `--epochs`, passed by `build_argv` (`:192`) | `10` (`:215`) | `EPOCHS=10` (both jobs) | `20` | `max_num_epochs = 2^{31} - 1` | -- | configured (job) |
| `steps_per_epoch` | $n_{\rm step}$ | not an axis | `--steps-per-epoch` (`:193`) | `25` (`:216`) | `STEPS_PER_EPOCH=25` (both jobs) | `50` | not a notion of sbi's loop (a pass) | -- | configured (job) |
| `b_met` | $B_{\rm met}$ | not an axis (F-l) | not passed | `32` (`:218`) | none | `BatchSpec.b_met = 64` | -- | -- | configured (P2) |
| `b_rep` | $B_{\rm rep}$ | not an axis (F-l) | not passed | `4` (`:219`) | none | `BatchSpec.b_rep = 8` | -- | -- | configured (P3) |
| `encoder_steps` | $n_{\rm plan}$ of the pre-training | not an axis | not passed | `200` (`:226`) | `ENCODER_STEPS=200` (Stage 3) | -- | -- | -- | configured (P2) |
| `grad_clip` | $\gamma_{\rm clip}$ | not an axis | no flag (F-i) | -- | none | `5.0` | `clip_max_norm = 5.0` | -- | configured by code |
| `patience` | $n_{\rm pat}$ | not an axis | no flag (F-i) | hard-coded `99` (`:522`, F-d) | none | `5` | `stop_after_epochs = 20` | -- | configured by code |
| the split | $\varsigma_{\rm tr}, \varsigma_{\rm sel}, \varsigma_{\rm rep}$ | not an axis | no flag (F-i) | `(0.7, 0.15, 0.15)` by donor (`:118`) | none | -- | `validation_fraction = 0.1` by row | -- | configured by code |
| `seed` | $s_{\rm seed}$ | not an axis | `--seed`, passed by `build_argv` (`:191`) | `0` (`:209`) | `SEED = IDX / 9` (Stage 3, `joint_arms.pbs:73`); `SEED=0` (Stage 4) | -- | -- | -- | configured |
| `eps`, `amsgrad` | $\epsilon_{\rm adam}$ | -- | -- | -- | -- | not set (torch's `1e-8`, `False`) | not set | `1e-8`, `False` | configured by torch |

Four readings of this table carry through the document.

1. **The four axes reach the trainer by flag, with the value the space
   drew.** `build_argv` formats every float at `repr` precision
   (`npe_tune_joint.py:197-204`), so the ledger's configuration and the
   trained value agree for `lr`, `one_minus_beta1` and `weight_decay`; the
   runner turns `one_minus_beta1` into $\beta_1 = 1 - \upsilon_1$
   (`run_joint_arms.py:524`), and `TrainConfig` refuses a $\beta_1$ or $\beta_2$ outside
   $[0, 1)$ (`joint_train.py:62-64`). `batch_size_npe` becomes `--b-sim`,
   and `b_met`, `b_rep` are not touched by the search (F-l).
2. **The schedule travels from the job, not from the space.** `EPOCHS` and
   `STEPS_PER_EPOCH` are `-v` variables of both jobs and arguments of
   `build_argv` with defaults 10 and 25; the space never sees them, so every
   trial of a campaign trains for the same $n_{\rm plan} = n_{\rm ep} n_{\rm step}$
   unless the job is launched with other values. The trial id does not
   encode them (P6).
3. **Three knobs are fixed by code with no flag at all** (F-i): the clip at
   5, $\beta_2$ at 0.999 -- which is the space's fixed `one_minus_beta2 =
   10^{-3}`, so the ledger's fixed value and the trained one agree by
   coincidence of two independent defaults, the F-w pattern -- and the
   patience at 99, written into the runner's `TrainConfig(...)` call over
   the library's 5 (`run_joint_arms.py:519-524`).
4. **The runner's defaults are not a point of the space** (F-r): `--b-sim
   128` is not in $\{256, 512, 1024\}$ and `--weight-decay 0.0` lies below
   $[10^{-5}, 10^{-2}]$; `--lr 1e-3` is inside its range, below which the
   log-uniform prior puts 77 % of its mass `[RAN]` B3; `--one-minus-beta1
   0.1` sits exactly on the upper edge of $[10^{-2}, 10^{-1}]$, inside the
   closed range but where `boundary_axes` would report a best configuration
   as "on edge" (`joint_space.py:807-813`). A Stage 3 arm at default flags
   therefore trains an optimiser the Stage 4 search cannot propose.

The DSN's own surfaces, for the two axes copied from it (D-036): `SearchConfig.one_minus_beta1_range = (1e-2, 1e-1)` and `one_minus_beta2_range = (1e-4, 1e-2)` (`dsn/config.py:932-933`), `RegularizationConfig.weight_decay_range = (1e-5, 1e-2)` (`:1184`) against `SearchConfig.weight_decay_range = (1e-4, 1e-2)` (`:934`, the inert one, TUNING_1 S3.7; the provenance string names the wrong class, F-c), and the DSN `TrainConfig`'s `lr = 3e-4`, `beta1 = 0.9`, `beta2 = 0.999`, `weight_decay = 1e-4`, `max_epochs = 100`, `patience = 10`, `use_scheduler = False` (`:738-745, 761`). Every DSN JSON config in `dsn/hpc/Config/` carries the same four optimiser ranges `[REPO]` (the 20 JSON files read); the configured training values differ between them (`patience` 40 of 100 epochs in the joint-search and refit configs, 20 of 60 in `config_l2c_joint_search_e60.json`, 10 of 50 in the `l3c` screens, 60 of 60 in `config_l3c_simplex_jointsep.json`, the configuration the DSN's own `TrainConfig` warns about).

### 3.2 The loop as run

*Establishes what one optimiser step and one run compute, written out from
`joint_train.py`, `joint_batches.py`, `run_joint_arms.py` and the torch
2.10.0 source as one explicit function of the knobs; every later section
points into these equations.*

**The step loss.** At step $\tau$ the batcher returns one batch per stream
from its own generator (`joint_batches.py:201-203`): $\mathcal{B}_{\rm sim}$,
$B_{\rm sim}$ row indices drawn uniformly with replacement from the
$N_{\rm train}$ training rows (`:172`); a metric batch of
$\max(1, \lfloor B_{\rm met} / C \rfloor)$ rows per class, with replacement
within the class (`:181-186`); and $\min(B_{\rm rep}, N_{\rm pair})$ same-donor pairs,
without replacement within the step (`:194-196`). The loop sums the active
terms (`joint_train.py:164-201`), which is plan eq. (2) as P3 eq. (P3.9)
writes it, evaluated at the weights the previous step left,

$$\hat{\mathcal{L}}_{\rm step}(\varpi_{\tau - 1}) = \frac{1}{B_{\rm sim}} \sum_{i \in \mathcal{B}_{\rm sim}} \ell_i \;+\; \lambda_{\rm dsn}\, \ell_{\rm DSN} \;+\; \lambda_{\rm rep}\, r_{\rm rep}(t)\, \frac{1}{\lvert \mathcal{B}_{\rm rep} \rvert} \sum_{(g, g') \in \mathcal{B}_{\rm rep}} \ell_{\rm rep}, \qquad t = \frac{\tau - 1}{n_{\rm plan}}, \tag{P5.1}$$

with $\ell_i = -\log q_\omega(\theta_i \mid h_\psi(x_i))$ the NLL of
training row $i$ under the current weights, $\ell_{\rm DSN}$ the metric term
of P2 on the metric batch and $\ell_{\rm rep}$ the per-pair replicate loss of
P3 on the pairs, whose ramp $r_{\rm rep}(t)$ reads $t$ through
`set_progress` (`:181`); a term whose weight is zero or whose stream is
absent is skipped, and a step with no active term raises (`:203-206`).
$n_{\rm plan} = \max(1, n_{\rm ep} n_{\rm step})$ (`:148`).

**The gradient and the clip.** `total.backward()` fills `.grad` on every
tensor that requires it; `clip_grad_norm_(model.parameters(), grad_clip)`
then rescales *every parameter of the model whose `.grad` is not `None`*
(`:208-210`; `clip_grad.py:230`), not only the trainable ones:

$$\Gamma_\tau = \nabla_\varpi \hat{\mathcal{L}}_{\rm step}(\varpi_{\tau - 1}), \qquad \varrho_\tau = \min\!\Big(1,\ \frac{\gamma_{\rm clip}}{\lVert \Gamma_\tau \rVert_2 + 10^{-6}}\Big), \qquad \tilde\Gamma_\tau = \varrho_\tau \Gamma_\tau, \tag{P5.2}$$

where the norm is the 2-norm of the concatenation of all the gradient
tensors the call sees (`clip_grad.py:106-108, 157-169`). On the arms whose
encoder is trained and frozen in the same process (`A0`, `A0s`) that
concatenation includes the encoder's gradients left by the last
pre-training step (F-ah, S3.8); on every other arm it is $\Gamma_\tau$
itself. At $\gamma_{\rm clip} = 5$: $\varrho_\tau = 1, 1, 0.1, 0.01$ at
$\lVert \Gamma_\tau \rVert_2 = 1, 5, 50, 500$ `[RAN]` B1.

**The AdamW step**, as torch 2.10.0 applies it to each trainable tensor
(`adam.py:417-419, 528-547`; `adamw.py:48` selects the decoupled branch;
parameters with `.grad is None` are skipped, `:151`):

$$\mu^{(1)}_\tau = \beta_1 \mu^{(1)}_{\tau - 1} + (1 - \beta_1)\, \tilde\Gamma_\tau, \qquad \mu^{(2)}_\tau = \beta_2 \mu^{(2)}_{\tau - 1} + (1 - \beta_2)\, (\tilde\Gamma_\tau)^{2}, \qquad \mu^{(1)}_0 = \mu^{(2)}_0 = 0, \tag{P5.3}$$

$$\bar\mu^{(1)}_\tau = \frac{\mu^{(1)}_\tau}{1 - \beta_1^{\tau}}, \qquad \bar\mu^{(2)}_\tau = \frac{\mu^{(2)}_\tau}{1 - \beta_2^{\tau}}, \qquad \varpi_\tau = (1 - \eta \gamma_{\rm wd})\, \varpi_{\tau - 1} - \eta\, \frac{\bar\mu^{(1)}_\tau}{\sqrt{\bar\mu^{(2)}_\tau} + \epsilon_{\rm adam}}, \tag{P5.4}$$

the square, the square root and the division taken elementwise, with
$\beta_1 = 1 - \upsilon_1$ from the flag, $\beta_2 = 0.999$ and
$\epsilon_{\rm adam} = 10^{-8}$ from the defaults, and the decay applied to
the weight *before* the moment update of the same step (`:417-419`
precede `:528-547`). `amsgrad` is off, so there is no running maximum of
$\mu^{(2)}_\tau$. Three closed forms follow from (P5.3)-(P5.4) and are what the
axes of S3.3 act on:

$$\text{the weight of the gradient of step } \tau' \text{ in } \bar\mu^{(1)}_\tau \text{ is } \frac{(1 - \beta_1)\beta_1^{\tau - \tau'}}{1 - \beta_1^{\tau}}, \qquad \mathcal{H}_1 = \frac{1}{\upsilon_1}, \quad \mathcal{H}_2 = \frac{1}{\upsilon_2}; \tag{P5.5}$$

$$\text{the decay alone leaves } (1 - \eta \gamma_{\rm wd})^{n_{\rm plan}} \approx \exp(-n_{\rm plan} / \mathcal{H}_{\rm wd}) \text{ of a weight, } \mathcal{H}_{\rm wd} = \frac{1}{\eta \gamma_{\rm wd}}; \tag{P5.6}$$

and, because $\bar\mu^{(1)}_\tau$ and $\sqrt{\bar\mu^{(2)}_\tau}$ are both
homogeneous of degree one in the gradient history, multiplying every term
of $\hat{\mathcal{L}}_{\rm step}$ by one positive constant at every step
leaves the update unchanged up to $\epsilon_{\rm adam}$ (`[RAN]` B1: a
factor $10^{3}$ moves the 250-step trajectory of a 50-weight replica by
$5 \times 10^{-10}$ against a norm of 0.11), whereas a factor that changes
from step to step -- which is what $\varrho_\tau$ is -- does not cancel
(alternating factors 1 and 0.1 move it by $3 \times 10^{-2}$). So
$\lambda_{\rm dsn}$ and $\lambda_{\rm rep}$ matter through the *relative*
weights of the three terms, not through the overall scale of the step, and
the clip matters only on the steps where it fires.

**Horizons at the configured values** `[RAN]` B1. $\mathcal{H}_1 = 10$
steps at the runner's $\upsilon_1 = 0.1$ (half-life 6.6 steps) and 100 at
the space's lower end $\upsilon_1 = 0.01$ (half-life 69);
$\mathcal{H}_2 = 1000$ steps at the fixed $\upsilon_2 = 10^{-3}$ (half-life
693), against $n_{\rm plan} = 250$ at the runner's schedule and 1000 at the
library's. The
bias corrections at $\tau = 250$ are $1 - 0.9^{250} = 1.0000$,
$1 - 0.99^{250} = 0.919$ and $1 - 0.999^{250} = 0.221$: the first moment has
forgotten the start of the run many times over at the default $\beta_1$,
the second moment has not -- at $\tau = 250$ its weights on the first and
the last gradient of the run are $4.5 \times 10^{-3}$ and
$3.5 \times 10^{-3}$, a nearly flat average of every squared gradient the
run has seen. A run of 250 steps at $\beta_2 = 0.999$ never operates the
second moment as a *moving* average; it operates it as the running mean of
the whole history, and the bias correction is what keeps that mean on the
right scale.

**The epoch.** $n_{\rm step}$ such steps, then (`:215-232`) the epoch
record -- the means of the three terms over the epoch's steps, the mean
$\hat T_{gg'}$ and $\hat p_{\rm eff}$ of the replicate batches, the invalid
count, the ramp -- and, when a selection split exists, the validation score

$$L_{\rm sel}(\iota) = \frac{1}{n} \sum_{i = 1}^{n} \ell_i(\varpi), \qquad \ell_i(\varpi) = -\log q_\omega\big(\theta_i \mid h_\psi(x_i)\big), \tag{P5.7}$$

over all $n$ rows of the selection split in batches of 512 with the model in `eval()`
(`evaluate_npe`, `:104-121`; an empty split raises rather than scoring 0,
`:106-110`). If $L_{\rm sel}(\iota) < L_{\rm sel}(\iota_{\rm best}) - 10^{-6}$
the epoch becomes $\iota_{\rm best}$ and a detached copy of the whole state
dict is kept (`:229-232`). When the DSN term is on, one more
`batcher.next()` feeds the gradient-cosine probe (`:234-249`) -- one extra
draw from every stream generator per epoch, with the separation ramp frozen
during the probe (F-ak, S3.8). Then the stopping test (`:262-267`):

$$\text{break after epoch } \iota \text{ iff a selection split exists, } n_{\rm pat} > 0 \text{ and } \iota - \iota_{\rm best} \ge n_{\rm pat}, \tag{P5.8}$$

and at the end of the loop, broken or not, the best state is loaded back
(`:269-270`). The earliest epoch at which (P5.8) can hold is
$\iota = n_{\rm pat}$, i.e. after $n_{\rm pat} + 1$ epochs: 100 epochs at the
runner's `patience=99`, so at `epochs=10` the rule never fires and every Stage 3 and
Stage 4 run trains the full $n_{\rm plan}$ steps (F-d); 6 epochs at the
library's 5 of 20 `[RAN]` B2. Rule (P5.8) and sbi's `_converged` (stop when
the count of fruitless epochs exceeds `stop_after_epochs - 1`,
`base.py:1243-1253`) stop at the same epoch on every one of 2000 random
validation curves `[RAN]` B2; the two differ only in the $10^{-6}$ margin,
which sbi does not have.

**The run.** `grouped_split` (`run_joint_arms.py:118-143`) permutes the
distinct donors with `numpy.random.default_rng(seed)` and assigns the first
$\mathrm{round}(\varsigma_{\rm tr} G_{\rm don})$ to training, the next
$\mathrm{round}(\varsigma_{\rm sel} G_{\rm don})$ to selection and the rest to
report; its sha256 is the record's `split_hash`. The selection rows score
every epoch, eq. (P5.7), and nothing else; the report rows score the
finished run -- `per_row_nll` on `theta[rp]`, whose mean is the record's
`L` and the ledger's `nll` (`:536-538, 587`; `npe_tune_joint.py:524-525`)
-- so the split the search optimises over is the report split, and the
selection split's only consumer is the stopping rule (F-al). On the bench
at the Stage 1 defaults (64 traces, 2 wells per donor, 8 windows: 32 donors of 16 rows) the
split at seed 0 is 22 / 5 / 5 donors, $352 / 80 / 80$ rows `[RAN]` B2; if
the DUP15HD bank were grouped by its 383 topology draws (`[KB]` plan S5.1)
it would be 268 / 57 / 58 groups. The number of steps, draws and passes a
run makes is then arithmetic (Table P5.1): $n_{\rm plan} = 250$ at the
runner's schedule, $B_{\rm sim} n_{\rm step}$ rows per epoch, and the
coverage of the training split by $n$ draws with replacement

$$1 - \Big(1 - \frac{1}{N_{\rm train}}\Big)^{n}, \qquad n = B_{\rm sim}\, n_{\rm step} \text{ per epoch},\ B_{\rm sim}\, n_{\rm plan} \text{ per run}. \tag{P5.9}$$

**Table P5.1 -- the schedule at the runner's $n_{\rm ep} = 10$, $n_{\rm step} = 25$, for the four batch sizes** `[RAN]` B2. The DUP15HD training rows are taken as $0.7 \times 29\,616 = 20\,731$ (`[KB]` plan S5.1 for the bank size; the group split only approximates the row fraction); the bench's are the 352 of the split above.

| $B_{\rm sim}$ | rows per epoch | rows per run | pass-equivalents of 20 731 rows (the expected visits per row per run) | coverage per epoch, eq. (P5.9) | coverage per run | bench: visits per row per epoch (352 rows) |
|---|---|---|---|---|---|---|
| 128 (runner) | 3 200 | 32 000 | 1.54 | 0.143 | 0.786 | 9.1 |
| 256 | 6 400 | 64 000 | 3.09 | 0.266 | 0.954 | 18.2 |
| 512 | 12 800 | 128 000 | 6.17 | 0.461 | 0.998 | 36.4 |
| 1 024 | 25 600 | 256 000 | 12.35 | 0.709 | 1.000 | 72.7 |

Two readings. At the runner's defaults a run of the DUP15HD bank sees each
training row 1.5 times on average and leaves 21 % of the rows unseen; the
same ten "epochs" at 1 024 are twelve pass-equivalents. On the bench ten
epochs are 91 visits per row at 128. The plan's "one epoch is one pass over
the simulated bank; at $B_{\rm sim} = 512$ that is about 58 steps"
(S5.1, $29\,616 / 512 = 57.8$ `[RAN]`) describes a loop the code does not
run: the code's epoch is 25 steps at any batch size, and ten of them at 512
are 6.2 passes of the training rows, not 10 (F-ag).

**Where the seed reaches** (`[REPO]` as cited; the consequences are
[reasoning from the source], none run):

| random object | generator | seeded where |
|---|---|---|
| the donor permutation of the split | `numpy.random.default_rng(seed)` | `grouped_split` (`run_joint_arms.py:127`, called `:369`) -- the split changes with the seed, so two seeds of one configuration train, select and report on different rows |
| the encoder's initial weights | torch's global generator | `make_backbone` calls `torch.manual_seed(seed)` before building (`:293`) |
| the flow's initial weights | torch's global generator, wherever the previous consumers left it | `build_joint_model` (`:463`) seeds nothing itself; on `A0`/`A0s` the pre-training's `torch.manual_seed(seed)` (`:153`) has reset the stream in between, so the flow of an `A0` run and of an `A1` run at one seed are initialised from different points of the stream |
| the simulated, metric and replicate batches | three `torch.Generator`s at `seed + 1`, `+ 2`, `+ 3` | `ThreeStreamBatcher` (`joint_batches.py:126-128`); the pre-training's batch generator also uses `seed + 1` (`run_joint_arms.py:155`), a separate object with the same stream |
| dropout masks and the `rsample` noise of the replicate term | torch's global generator | `train_joint` calls `torch.manual_seed(seed)` once at its start (`joint_train.py:134`) |
| the shuffled control's pairing | `torch.Generator` at `seed + 777` | `run_joint_arms.py:381` |
| the cluster scores of the report | `cluster_scores(..., seed=seed)` | `:607-608` |

### 3.3 The four searched axes

*Establishes, per axis, the fields of plan S2.2: where it is set, what it
changes in eq. (P5.1)-(P5.9), what the range covers and how it is sampled,
how it fails and what reveals the failure.*

#### 3.3.1 `lr` ($\eta$)

- **Set on:** the space, $[10^{-4}, 2 \times 10^{-3}]$ log-uniform
  (`joint_space.py:238`, `:465-469`), the range of the standalone NPE
  tuner (`npe_tune_search.py:105, 144`; `RANGE_PROVENANCE`, `:141`) and
  **not** the DSN's `SearchConfig.lr_range = (1e-4, 0.2)` (`dsn/config.py:931`),
  whose upper end is 100 times higher `[RAN]` B3; reaches the runner as
  `--lr`; runner default $10^{-3}$, library $5 \times 10^{-4}$, sbi
  $5 \times 10^{-4}$, torch $10^{-3}$, DSN `TrainConfig` $3 \times 10^{-4}$;
  not a job variable. Free in every campaign (P0 Table A). Status:
  configured.
- **Changes:** the scale of every step in eq. (P5.4), of both the gradient
  part and the decay part, for $\psi$ and $\omega$ alike -- one learning
  rate for two networks of different depth and normalisation (the
  GroupNorm CNN of P1 and the plain ReLU conditioners of P4). Through
  eq. (P5.6) it also sets the decay horizon: at fixed $\gamma_{\rm wd}$ a
  larger $\eta$ decays faster. It does not change the direction of the
  step, which eq. (P5.4) fixes per coordinate up to scale.
- **Range and prior:** three decades below the DSN's range; under the
  log-uniform prior 54 % of the mass lies below sbi's default
  $5 \times 10^{-4}$ and 77 % below the runner's $10^{-3}$ `[RAN]` B3.
  There is no schedule: the module docstring and the plan name "LR
  schedules" among what sbi's loop hides (`joint_train.py:3-4`; plan
  Stage 2), and the explicit loop implements none, so $\eta$ is constant
  from $\tau = 1$ to $n_{\rm plan}$ -- the same constraint TUNING_1 S3.5
  states for the standalone DSN with `use_scheduler = False`: one value
  must be small enough to be stable at the first step and large enough to
  make progress by the last. Among the training recipes of the knowledge
  base, the compositional-SBI paper's score network used AdamW at
  $5 \times 10^{-4}$ *with a cosine schedule* (`[KB-PDF p.25]`), BayesFlow
  an exponential decay of 0.95 from $10^{-3}$ (`[KB-PDF p.6]`), the
  flow-matching paper a constant $5 \times 10^{-4}$ (`[KB-PDF p.27]`, Table
  3), the trust-crisis benchmarks $10^{-3}$ throughout (`[KB-PDF p.17]`,
  Table 2); the practical guide lists the learning rate with the batch
  size and the early-stopping schedule among the hyper-parameters worth
  tuning once the defaults are outgrown (`[KB-PDF p.11]`, `[KB-PDF p.32]`).
  None of these is a statement about this bank.
- **Interacts with:** $\gamma_{\rm wd}$ through the product
  $\eta \gamma_{\rm wd}$ (S3.3.3); $B_{\rm sim}$ (S3.3.4); $\upsilon_1$: a
  larger $\mathcal{H}_1$ smooths the direction, and the stable range of
  $\eta$ moves with it [reasoning]; the clip: at a large $\eta$ the first
  steps' gradients are the ones most likely to be clipped, and
  eq. (P5.2)'s rescaling then shapes $\mu^{(2)}_\tau$ for the rest of the run
  (S3.2, "nearly flat average"); the flow's depth (P4, F-ac): twelve plain
  layers per conditioner at the upper bound are the regime in which a
  constant $\eta$ chosen for the lower bound is least likely to transfer
  [reasoning].
- **Fails as:** too large, divergence or a loss that plateaus above the
  identity-initialised value $0.56\, d_\theta$ (P4 eq. (P4.10)); too small,
  250 steps that leave the flow near its initialisation. With no stopping
  rule in force (F-d) a diverged run trains to the end and reports its
  best epoch, which may be epoch 0.
- **Revealed by:** the history's `npe` series against $0.56\, d_\theta$ and
  against $L_0$; `val_npe` per epoch; `L` of the record; the ledger's
  partial dependence on `lr` (P6).

#### 3.3.2 `one_minus_beta1` ($\upsilon_1$)

- **Set on:** the space, $[10^{-2}, 10^{-1}]$ log-uniform
  (`joint_space.py:239`, `:465-469`), copied from the DSN's
  `SearchConfig.one_minus_beta1_range` (`dsn/config.py:932`;
  `RANGE_PROVENANCE`, `:142`); reaches the runner as `--one-minus-beta1`,
  which the runner turns into $\beta_1 = 1 - \upsilon_1$
  (`run_joint_arms.py:524`); runner default 0.1, i.e. $\beta_1 = 0.9$ --
  "AdamW's own default" (the flag's help, `:258-263`), torch's, sbi's and
  the DSN `TrainConfig`'s too. Free in every campaign. Status:
  configured. The axis exists in the trainer only since Stage 4:
  `TrainConfig` took AdamW's betas until the `[CHANGE Stage 4]` note
  (`joint_train.py:54-61`), which says why: a search over an axis that
  never reaches the trainer "reads as 'this axis does not matter' in the
  partial dependence".
- **Changes:** the horizon $\mathcal{H}_1 = 1/\upsilon_1$ of the first
  moment, eq. (P5.5): 10 steps at the runner's default, 100 at the lower
  end of the range `[RAN]` B1. In the direction of eq. (P5.4) a short
  horizon follows the current batch, a long one averages the last
  $\mathcal{H}_1$ batches; at $\mathcal{H}_1 = 100$ and $n_{\rm plan} = 250$
  the first moment has forgotten the run's start only twice over by the
  end, and its bias correction is still $1 - 0.99^{25} = 0.22$ at the end
  of the first epoch `[RAN]` B1. It does not change the second moment or
  the decay.
- **Range and prior:** the parameterisation is the DSN's argument, quoted
  from TUNING_1 S3.6 with its hypothesis: searching $\upsilon_1$ (TUNING_1's
  $u_1$) on a log scale makes the prior "uniform in timescale", because the
  quantity with meaning is the averaging horizon, "roughly $1/(1-\beta_1)$
  steps", and in the coordinates of the rate itself $0.999$ and $0.9999$ are
  adjacent while their horizons differ tenfold;
  the runner's flag help says the same in fewer words. The range is
  $\beta_1 \in [0.9, 0.99]$; the runner's default is the upper edge of the
  range (S3.1, reading 4).
- **Interacts with:** $\eta$ (above); the batch size: a smaller
  $B_{\rm sim}$ gives noisier $\Gamma_\tau$, and $\mathcal{H}_1$ is the
  only averaging the direction gets [reasoning]; the replicate term's ramp
  (P3): $\lambda_{\rm rep}$ grows over $t_{\rm warm} n_{\rm plan}$ steps and
  the first moment lags it by about $\mathcal{H}_1$ steps; the probe
  $\rho_{\rm grad}$, which is computed on raw gradients, not on
  $\mu^{(1)}_\tau$ (`gradient_cosine`, `:77-101`).
- **Fails as:** a horizon longer than the ramp or than an epoch blurs the
  change of regime the ramp introduces; a horizon of a few steps on a
  32-row metric batch and a 4-pair replicate batch (S3.1) follows the
  batch noise of the two small streams.
- **Revealed by:** the history's `npe`, `dsn`, `rep` series epoch by
  epoch (25-step means, coarser than $\mathcal{H}_1$ at the default); the
  partial dependence on the axis (P6).

#### 3.3.3 `weight_decay` ($\gamma_{\rm wd}$)

- **Set on:** the space, $[10^{-5}, 10^{-2}]$ log-uniform
  (`joint_space.py:240`, `:465-469`), the value of the DSN's
  `RegularizationConfig.weight_decay_range` (`dsn/config.py:1184`) under a
  provenance string that names `SearchConfig.weight_decay_range`, whose
  value is $(10^{-4}, 10^{-2})$ (`:934`; F-c); reaches the runner as
  `--weight-decay`; runner default 0.0 (below the range, F-r), library
  0.0, torch's `AdamW` $10^{-2}$, DSN `TrainConfig` $10^{-4}$; sbi's loop
  uses `Adam` and has none. Free in every campaign. Status: configured;
  on the frozen arms' pre-training it is torch's $10^{-2}$ (F-q).
- **Changes:** the factor $(1 - \eta \gamma_{\rm wd})$ of eq. (P5.4), applied
  to every trainable weight every step, outside the moment averages
  (convention 3). TUNING_1 S3.7 notes that with L2-normalised embeddings
  decay "acts mostly on the interior layers; its effect on the embedding
  geometry is indirect" -- a statement about the standalone DSN's output
  layer, which the joint encoder shares (P1); the flow's conditioners
  (P4) are not normalised and feel it directly.
- **Range and prior, and what the horizon says about the range.**
  Eq. (P5.6) at the corners of the space and the runner's horizon `[RAN]`
  B1: at the strongest corner, $\eta = 2 \times 10^{-3}$ with
  $\gamma_{\rm wd} = 10^{-2}$, the per-step factor is $0.99998$, the
  cumulative factor over 250 steps $0.995$ -- a 0.50 % shrink -- and
  $\mathcal{H}_{\rm wd} = 5 \times 10^{4}$ steps, 200 times the planned run;
  at $\eta = 10^{-3}$ with $\gamma_{\rm wd} = 10^{-2}$ the shrink is 0.25 %
  over 250 steps and 1.0 % over the library's 1000; at the weakest corner,
  $\eta = 10^{-4}$ with $\gamma_{\rm wd} = 10^{-5}$, the horizon is $10^{9}$
  steps.
  The decay's own contribution to any weight is therefore below one
  percent everywhere in the searched box at the schedules the stack runs,
  and its indirect effect through the trajectory is bounded by the same
  per-step factor; the axis is, at $n_{\rm plan} = 250$, a weak one whose
  partial dependence the seed spread will dominate [reasoning on
  eq. (P5.6); F-ai, S3.8]. The DSN searched the same range with
  `max_epochs = 100` of 100 steps each (`batches_per_epoch = 100`; TUNING_2
  S3.2), a horizon of up to $10^{4}$ steps, forty times longer. According
  to PubMed, the one peer-reviewed theory paper on AdamW that PubMed
  indexes (Zhou P et al., IEEE TPAMI 2024,
  [DOI](https://doi.org/10.1109/TPAMI.2024.3382294)) is **abstract only**
  (no PMC full text) and is therefore not used for any claim here; its
  abstract states that the decoupled decay makes AdamW minimise a
  dynamically regularised loss different from the $L_2$-penalised one,
  which is the distinction convention 3 draws and no more.
- **Interacts with:** $\eta$ (the product); the frozen arms: the encoder
  of `A0`/`A0s` is excluded from the loop's optimiser (`joint_train.py:140`)
  and decayed only during its pre-training, by torch's default
  $10^{-2}$ at $\eta = 10^{-3}$ for 200 steps, a 0.20 % shrink `[RAN]` B4
  -- at the runner's defaults the only tensor weight decay ever touches in
  a Stage 3 campaign (F-q); the dropout axis of P1, with which the DSN's
  staged pipeline tuned it last, "in a separate regularisation phase, after
  the architecture and loss were frozen" (TUNING_1 S3.8; `RegularizationConfig`,
  `dsn/config.py:1180-1184`); the control recipe (P6),
  which copies it.
- **Fails as:** inert at the horizon (above); at a long schedule
  ($n_{\rm plan}$ of the order of $\mathcal{H}_{\rm wd}$ or longer) the familiar shrink of
  weights that the gradient does not hold up -- not a regime the current
  jobs reach.
- **Revealed by:** the partial dependence on the axis against
  $\sigma_{\rm seed}$ (P6); the weight norms, which nothing in the stack
  records (S5).

#### 3.3.4 `batch_size_npe` ($B_{\rm sim}$)

- **Set on:** the space, the categorical $\{256, 512, 1024\}$
  (`joint_space.py:241`, `:456-457`), copied from the standalone NPE
  tuner (`npe_tune_search.py:106`; `RANGE_PROVENANCE`, `:144`), trimmed
  when `--n-train` is passed: a batch size stays only if `n_train` is at
  least twenty times it, and the smallest stays regardless (`:304-306`);
  reaches the runner as `--b-sim` (F-l); runner default 128 (not in the
  set, F-r), job
  variable `B_SIM=128` in the Stage 3 job only, `BatchSpec` 512, sbi 200,
  the standalone `NPEConfig` 512 (`npe_model.py:87`). Free in every
  campaign; `boundary_axes` never reports it (categorical,
  `joint_space.py:761-765`). Status: configured.
- **Changes:** the number of simulated rows in eq. (P5.1), hence the
  variance of the NPE term's gradient within $\Gamma_\tau$; the rows per
  epoch and per run, Table P5.1; the memory and time of a step. It does
  **not** change the number of steps: an epoch is $n_{\rm step}$ steps at
  any batch size (convention 1), so a larger batch is strictly more data
  per run, not the same data in fewer steps -- the opposite of sbi's
  pass-based loop, where a larger batch means fewer steps per epoch. It
  does not change $B_{\rm met}$ or $B_{\rm rep}$ (F-l), so the three
  streams' ratio moves with it: 32 : 8 : 1 at the runner's 128 / 32 / 4,
  64 : 8 : 1 at `BatchSpec`'s 512 / 64 / 8, 256 : 8 : 1 at the space's
  1024 with the runner's two `[RAN]` B3.
- **Range and prior:** three values, uniform. The trim rule's thresholds
  are 5 120 / 10 240 / 20 480 training rows `[RAN]` B2 -- on the bench's
  352 rows only 256 survives, and at the DUP15HD training rows
  ($20\,731$) all three, 1024 by 251 rows. The rule's docstring gives a
  pass-based reason, "a batch that swallows the epoch makes early stopping
  meaningless" (`:283-285`), inherited verbatim from the standalone tuner
  (`npe_tune_search.py:126-129`), where the epoch *is* a pass; in the
  joint loop no batch can swallow an epoch of 25 steps, and the twenty
  steps per pass the rule enforces are not a quantity the loop has (F-ag).
  The jobs pass no `--n-train`, so in practice the set is always the full
  three.
- **Interacts with:** $\eta$. According to PubMed, read in full text from
  PubMed Central, two recent papers state the coupling for plain
  stochastic gradient descent: Sclocchi and Wyart (PNAS 2024,
  [DOI](https://doi.org/10.1073/pnas.2316301121)) describe small-batch SGD
  as a noisy gradient descent whose noise amplitude is governed by the
  "temperature" learning rate over batch size, with a noise-dominated
  regime in which the outcome depends on that ratio alone, a
  first-step-dominated regime at large batches and a gradient-descent
  regime at low temperature -- and state explicitly that their analysis
  "does not consider variations of SGD sometimes used in practice, such
  as the addition of momentum, adaptive learning rates, or weight decay";
  Zeng et al. (J Am Stat Assoc 2026,
  [DOI](https://doi.org/10.1080/01621459.2026.2644611)) restate the
  linear scaling rule -- keeping the ratio of learning rate to batch size
  fixed leaves SGD training nearly unchanged over a broad range of batch
  sizes -- as a reason to tune one of the two and fix the other, for SGD
  on a batch-size-dependent objective. Both statements are about SGD; the
  R3 check fails for eq. (P5.4), whose per-coordinate normalisation is
  exactly what the SGD noise argument lacks, so the document transplants
  neither and records only that the joint space searches $\eta$ and
  $B_{\rm sim}$ as independent axes with no coupling built into the prior
  [reasoning]. With $\upsilon_1$: the batch noise $\mathcal{H}_1$ averages
  (S3.3.2). With the schedule: at fixed $n_{\rm plan}$ the axis is the only
  way the search changes how much simulated data a trial sees
  (Table P5.1), which makes it a budget axis as much as an optimisation
  axis; a trial at 1024 costs eight times the forward passes of one at
  128 per step [reasoning from the counts].
- **Fails as:** at 256 on the bench's 352 training rows a batch is 73 %
  of the split `[RAN]` B2 and consecutive batches overlap heavily, so the step
  gradients are nearly those of full-batch descent on a tiny set; at 1024
  on a bank whose selection split is small, overfitting the 128 000
  draws per run (Table P5.1) is the risk, with early stopping inoperative
  (F-d).
- **Revealed by:** the gap between the history's training `npe` and
  `val_npe`; `L` against $L_0$; the runtime per epoch of the job log; the
  partial dependence on the axis (P6).

### 3.4 The fixed knobs

*Establishes what the joint stack holds fixed around the four axes, where,
and what each fixed value commits a run to.*

| knob | value | where | what it commits to | note |
|---|---|---|---|---|
| `--epochs` $n_{\rm ep}$ | 10 (runner, both jobs); 20 (`TrainConfig`) | `run_joint_arms.py:215`; `joint_arms.pbs:53`; `joint_tune.pbs:65`; `build_argv` default 10 | with $n_{\rm step}$, $n_{\rm plan} = 250$: the horizon of eq. (P5.1)-(P5.9), of both warm-up ramps (P2, P3) and of the decay (P5.6); the number of validation scores early stopping can compare | not in the trial id (P6): two campaigns at different `EPOCHS` share a ledger silently |
| `--steps-per-epoch` $n_{\rm step}$ | 25 (runner, both jobs); 50 (`TrainConfig`) | `:216`; `joint_arms.pbs:54`; `joint_tune.pbs:66` | the epoch's length in steps, convention 1; the granularity of the history (25-step means) and of the stopping test | the plan's pass semantics would make it $N_{\rm train}/B_{\rm sim}$, 40.5 at 512 on the DUP15HD training rows `[RAN]` B2 (F-ag) |
| `--b-met` $B_{\rm met}$ | 32 (runner); 64 (`BatchSpec`) | `:218`; `joint_batches.py:64` | $\lfloor B_{\rm met}/C \rfloor$ rows per class per step; also the batch of the encoder pre-training (`:459`) | not searched, not a job variable (F-l); the metric term's pools (P2) |
| `--b-rep` $B_{\rm rep}$ | 4 (runner); 8 (`BatchSpec`) | `:219`; `joint_batches.py:64` | pairs per step, $2 B_{\rm rep} S_{\rm mc}$ posterior draws per step (P3) | not searched, not a job variable (F-l) |
| `--encoder-steps` | 200 (runner; `ENCODER_STEPS`) | `:226`; `joint_arms.pbs:59` | $n_{\rm plan}$ of the pre-training of `A0`/`A0s` and the horizon of its separation ramp (P2) | its optimiser is torch's `AdamW` defaults at `--lr` (F-q): a 0.20 % decay over the 200 steps `[RAN]` B4 |
| `grad_clip` $\gamma_{\rm clip}$ | 5.0 | `TrainConfig` (`joint_train.py:42`), applied `:208-210`; no flag (F-i) | eq. (P5.2) over `model.parameters()`; sbi's `clip_max_norm` is also 5.0 (`npe_base.py:259`; `base.py:1154`) | on `A0`/`A0s` the norm includes the frozen encoder's stale gradients (F-ah); `0` would disable it |
| `beta2`, `one_minus_beta2` $\upsilon_2$ | 0.999, $10^{-3}$ | `TrainConfig` (`:44`); `spec.fixed` (`joint_space.py:324`); no flag (F-i) | $\mathcal{H}_2 = 1000$ steps, eq. (P5.5): the second moment is a running mean of the whole 250-step run (S3.2) | the DSN searches $\upsilon_2 \in [10^{-4}, 10^{-2}]$ (S3.5); the two fixed values agree by independent defaults, not by wiring |
| `patience` $n_{\rm pat}$ | **99** (runner); 5 (`TrainConfig`) | `run_joint_arms.py:522`; `joint_train.py:43` | eq. (P5.8) cannot fire below 100 epochs: every run trains $n_{\rm plan}$ steps and restores its best epoch (F-d) | `control_config_from`'s docstring expects early stopping to "fire almost immediately on shuffled data" and asks for epochs-to-stop to be recorded (`joint_space.py:713-717`): neither can happen; the DSN's `TrainConfig` *warns* when `max_epochs <= patience` ("early stopping can never fire", `dsn/config.py:869-875`), the joint `TrainConfig` does not |
| the improvement margin | $10^{-6}$ nats/row | `joint_train.py:229` | an epoch counts as better only by more than $10^{-6}$ | sbi uses a strict `<` with no margin (`base.py:1243`); inert while F-d stands |
| the split fractions $\varsigma_{\rm tr}, \varsigma_{\rm sel}, \varsigma_{\rm rep}$ | $(0.7, 0.15, 0.15)$ of the donors | `grouped_split` default (`:118`); no flag (F-i) | 22 / 5 / 5 donors on the bench at seed 0 `[RAN]` B2; a bank with fewer than 7 donors gets an empty selection or report part and the runner refuses or warns (`:371-377`) | by group, not by row (plan S2.4, E0 convention 6); sbi splits 10 % of rows with no grouping (`base.py:505-513`) |
| `rho_grad_probe` | on iff $\lambda_{\rm dsn} > 0$ | `run_joint_arms.py:523`; `joint_train.py:234-249` | one extra `batcher.next()` per epoch, two extra backward passes on it | advances all three stream generators (F-ak) |
| `--seed` $s_{\rm seed}$ | 0 (runner; Stage 4 job); `IDX / 9` (Stage 3 job) | `:209`; `joint_tune.pbs:67`; `joint_arms.pbs:73` | the table of S3.2: split, initialisation, streams, dropout, draws, shuffle | one value per Stage 4 trial; `--n-seeds` of the controls varies it, and with it the split |
| `eps`, `amsgrad`, `foreach`/`fused` | $10^{-8}$, `False`, torch's choice | torch `adamw.py:24-28` | eq. (P5.4)'s denominator constant; no running maximum; the implementation path | unreachable from the stack |
| the `trainable` filter | `requires_grad` tensors only | `joint_train.py:140-147` | $\varpi$: the frozen encoder is outside the optimiser and outside the decay | the comment's rationale ("weight decay ... decays a frozen tensor ... even though its gradient is None", `:135-139`) does not describe torch, which skips a tensor with `.grad is None` (`adam.py:151`); the filter is still right, for the stale-gradient reason of F-ah |
| schedule of $\eta$ | none | -- | $\eta$ constant over the run | the loop's own motivation names "LR schedules" (`joint_train.py:3-4`); the DSN's `use_scheduler = False` likewise (`dsn/config.py:761`) |
| `evaluate_npe` batch | 512 rows | `joint_train.py:104` | the validation's forward batches; no effect on the score | -- |

### 3.5 What is inherited, and what differs: the DSN search and sbi's loop

*Establishes that the block borrows from two stacks at once -- two of its
ranges from the standalone DSN's `SearchConfig`, its remaining range and
its batch set from the standalone NPE tuner, and its loop from a rewrite of
sbi's -- and sets each pair side by side so a reader of one is not misled by
the other.*

**The D-036 differences table** for the two DSN-inherited axes, and for
`lr`, which the DSN also searches although the joint range is the NPE
tuner's. TUNING_1 documents `config_l3c_joint_search.json`; its optimiser
ranges equal the dataclass defaults and are the same in every DSN config of
the tree `[REPO]`, so for these three axes F-g's two sources agree.

| | joint space (P5) | DSN joint-condition search (`TUNING_1` S3.5-S3.7; `SearchConfig`) | DSN base config (`TrainConfig`) | note |
|---|---|---|---|---|
| `lr` $\eta$ | $[10^{-4}, 2 \times 10^{-3}]$ log-uniform; default $10^{-3}$ | $[10^{-4}, 0.2]$ log-uniform, "the upper end, 0.2, is aggressive for Adam and will produce divergent or degenerate runs ... intentional" (S3.5) | $3 \times 10^{-4}$ | the joint range is the NPE tuner's (`npe_tune_search.py:105`); the DSN's refit configs in `dsn/hpc/Config/` record selected values from $1.5 \times 10^{-3}$ to $4.1 \times 10^{-2}$ `[REPO]` (`refit_mea_*.json`, `train.lr`), five of the seven above the joint upper bound -- the standalone encoder was trained at learning rates the joint space cannot propose |
| `one_minus_beta1` $\upsilon_1$ | $[10^{-2}, 10^{-1}]$ log-uniform; default 0.1 | the same range, the same prior, the same argument (S3.6) | $\beta_1 = 0.9$ | identical by copy |
| `one_minus_beta2` $\upsilon_2$ | **fixed** $10^{-3}$ | searched, $[10^{-4}, 10^{-2}]$ log-uniform (S3.6: $\beta_2$ near 0.9999 "means a 10 000-step second-moment memory, which is comparable to the whole planned run") | $\beta_2 = 0.999$ | the joint fixes the geometric midpoint of the DSN's range; at $n_{\rm plan} = 250$ the fixed $\mathcal{H}_2 = 1000$ already exceeds the run (S3.2) |
| `weight_decay` $\gamma_{\rm wd}$ | $[10^{-5}, 10^{-2}]$ log-uniform; default 0 | $[10^{-5}, 10^{-2}]$ from `regularization.weight_decay_range`, with `search.weight_decay_range` $[10^{-4}, 10^{-2}]$ inert (S3.7, "a trap") | $10^{-4}$ | identical range; the joint's provenance string names the inert one (F-c); the refit configs record $1.4 \times 10^{-4}$ to $6.4 \times 10^{-3}$ `[REPO]`, all inside |
| epochs, steps | $10 \times 25 = 250$ steps, fixed count | `max_epochs` 100 (the design document recommends 60 and 20 but records the recommendation as "not yet accepted", TUNING_2 S3.2) of `batches_per_epoch = 100` steps | 100 epochs, `batches_per_epoch = 0`: one nominal pass, the training windows divided by the $C$ times `windows_per_condition` rows of a batch, rounded up (`dsn/config.py:730-735`) | the DSN's planned run is up to $10^{4}$ steps (TUNING_1 S3.6) against 250 |
| early stopping | on $L_{\rm sel}$, patience 99 of 10 (F-d) | on the held-out silhouette, patience 40, an improvement counted only beyond two standard deviations of a label-shuffled null (TUNING_2 S3.4, eq. (3)) | patience 10, `min_delta` 0 | the DSN's config warns when the rule cannot fire; the joint's does not |
| schedule | none | none (`use_scheduler = False`, TUNING_1 S3.5) | none | agree |
| clip | 5.0, global norm | no clip field in `TrainConfig` `[REPO]` `dsn/config.py:615-770` | -- | the joint adds it, from sbi's default |

**The joint loop against sbi's `train()`** (`[REPO sbi wheel]`
`base.py:384-400, 500-538, 1064-1110, 1143-1159, 1243-1253`;
`npe_base.py:252-259`), the loop `npe_tune_train.py` and `npe_model.py` run
for the standalone tuner (`npe_tune_train.py:242-248`; `npe_model.py:86-91,
204-211`):

| | joint loop (`joint_train.py`) | sbi 0.27.0 `train()` | note |
|---|---|---|---|
| optimiser | `AdamW(trainable, lr, weight_decay, betas)` | `Adam(net.parameters(), lr=learning_rate)` (`base.py:1065-1068`): no decay, betas fixed | the reason the plan gives for the explicit loop |
| learning rate | $\eta$ searched, default $10^{-3}$ | $5 \times 10^{-4}$ default; the standalone tuner searches the same $[10^{-4}, 2 \times 10^{-3}]$ | |
| epoch | $n_{\rm step}$ steps, with replacement | one pass: `SubsetRandomSampler` over the training indices, `drop_last=True` (`:519-523, 535`) | convention 1 |
| batch | $B_{\rm sim}$ i.i.d. rows | `min(training_batch_size, n_train)`, 200 default; the standalone `NPEConfig` 512 | |
| validation split | 15 % of donors, grouped, seeded by `--seed` | `validation_fraction = 0.1` of rows, `torch.randperm` (`:509-513`), no grouping | plan S2.4 |
| validation score | $L_{\rm sel}$ over the whole split each epoch | the mean loss over the validation loader each epoch (`_validate_epoch`) | the same quantity up to `drop_last` |
| stopping | eq. (P5.8), $n_{\rm pat} = 99$ (inoperative) | `_converged`: stop when fruitless epochs $>$ `stop_after_epochs - 1`, default 20; `max_num_epochs` $2^{31} - 1$ | the same rule, 2000 / 2000 curves `[RAN]` B2 |
| restore | the best state dict, always | the best state dict, on convergence and at the ceiling (`:1096-1102, 1252`) | agree |
| clip | 5.0 over `model.parameters()` | `clip_max_norm = 5.0` over `net.parameters()` (`:1154`) | agree in value; F-ah on the frozen arms |
| extra terms | $\lambda_{\rm dsn} \ell_{\rm DSN}$, $\lambda_{\rm rep} \ell_{\rm rep}(t)$, the probe | none: one loss, no labels | |
| record | per epoch: the three terms, $\hat T_{gg'}$, $\hat p_{\rm eff}$, invalid count, ramp, `val_npe`, `rho_grad` | `epochs_trained`, `best_validation_loss`, `validation_loss` per epoch | `epochs_trained` has no joint counterpart (F-d's docstring asks for one) |

What the knowledge base records as sbi's defaults agrees with the wheel:
"early stopping if the validation loss did not decrease for 20 epochs,
batch size 200, Adam optimizer with learning rate $5 \times 10^{-4}$"
(`[KB-PDF p.40]`, the practical guide's DDM appendix), and the multilevel
preprint's description of the same criterion with 20 % of the data held out
(`[KB-PDF p.9]`, **PREPRINT, not peer-reviewed** [corrected 2026-10-03: its p.1 carries the NeurIPS 2025 conference line, so it is peer-reviewed as a conference paper; P7 S6]). Goncalves et al. trained
with "ADAM with default settings" (`[KB-PDF p.19]`); in PubMed Central,
Boelts et al. (eLife 2022, [DOI](https://doi.org/10.7554/eLife.77220))
state that their networks were trained "using the maximum likelihood loss
and the Adam optimizer" and leave the hyper-parameters to an appendix the
PMC text does not carry [PubMed full text, PMC9374439; no number used].

### 3.6 Interactions with the other blocks

- **With the encoder (P1).** $n_\varpi$ is $n_\psi + n_\omega$ on the joint
  arms: 359 450 encoder weights at the runner's defaults (P1 eq. (P1.10))
  plus 107 346 flow weights at the same $E = 10$ (P4 Table P4.1), and up
  to 215 million encoder
  weights at the `depth_exponent` corner (F-s) -- one $\eta$, one
  $\mathcal{H}_1$ and one clip for all of them. The encoder is GroupNorm
  throughout (P1), so its gradient scale is held roughly in check by the
  architecture; the flow's plain conditioners are not (P4, F-ac), and a
  norm clip at 5 over both treats them as one vector. `dropout` consumes
  the global generator that `train_joint` seeds (S3.2).
- **With the DSN term (P2).** $\lambda_{\rm dsn}$ enters eq. (P5.1) as a
  weight on $\ell_{\rm DSN}$; by the loss-scale invariance of S3.2 what
  matters is its ratio to the NPE term's scale, not its size. The
  pre-training of `A0`/`A0s` is a second optimiser (F-q): `AdamW` at the
  runner's `--lr` with torch's $\gamma_{\rm wd} = 10^{-2}$ and
  $(\beta_1, \beta_2) = (0.9, 0.999)$ -- the betas coincide with the runner's defaults and
  differ from a searched $\upsilon_1$; it has no clip, no validation, no
  early stopping, 200 steps on batches of $B_{\rm met}$ rows from a
  generator at `seed + 1` (`run_joint_arms.py:150-168`). Its last
  gradients stay on the encoder tensors (F-ah). The separation ramp's
  horizon is `--encoder-steps` on those arms and $n_{\rm plan}$ on the
  joint ones (`:408-409`).
- **With the replicate term (P3).** The ramp of $\lambda_{\rm rep}$ reads
  $t = n_{\rm done}/n_{\rm plan}$: at the runner's `--warmup-frac-rep 0.3`
  it completes at step 75, epoch 3 (P3 S3.2). Every step of an `A5` run
  draws $2 B_{\rm rep} S_{\rm mc}$ posterior samples with gradient through
  $\psi$ -- the dominant cost of the step at $S_{\rm mc} = 128$ (P3), and
  the reason a step's wall time, not only its count, enters the budget.
  $\mathcal{H}_1$ lags the ramp by about ten steps at the default
  $\upsilon_1$.
- **With the flow (P4).** $n_\omega$ grows quadratically in
  `num_transforms` (P4 eq. (P4.9)), and with it the share of
  $\lVert \Gamma_\tau \rVert_2$ the flow contributes; a configuration at
  the flow's upper corner and one at its lower corner are clipped by the
  same $\gamma_{\rm clip}$ [reasoning]. The identity-initialised loss
  $0.56\, d_\theta$ (P4 eq. (P4.10)) is the value every run starts near,
  whatever the optimiser.
- **With the search (P6).** The four axes are free in every campaign and
  never canonicalised; `boundary_axes` reports $\eta$, $\upsilon_1$ and
  $\gamma_{\rm wd}$ at their edges (relative tolerance $10^{-6}$ of the
  wider bound, `joint_space.py:807-813`) and never $B_{\rm sim}$. A
  control run copies every optimiser and schedule value and permutes only
  the pairing (`control_config_from`, `:697-722`), so the floor it
  measures is that recipe's -- at `patience=99` a control trains the full
  250 steps on shuffled pairs and restores its best epoch, which on
  shuffled data can be the first (F-d). $\sigma_{\rm seed}$, measured over
  `--n-seeds` controls, varies the seed and with it the split (S3.2): it is
  the spread over splits and initialisations together, not the
  initialisation alone -- the plan's statement that seeds are not paired
  across arms (S2.4a) holds within an arm's seeds in this further sense. The
  probe shift of F-ak means that an `A1` and an `A2` trial at one seed
  share their first epoch's simulated batches and none after.
- **With the jobs (P7).** Both jobs run on 4 CPUs and 16 GB with no GPU
  (`joint_arms.pbs:3`; `joint_tune.pbs:3`), with walltimes of 6 and 8 hours;
  the practical guide's remark that for moderate dimensions and batch
  sizes below 1000 CPU training "can be as fast as or even faster than on
  a GPU" (`[KB-PDF p.11]`) is the regime of the runner's 128 and not of
  the space's 1024.

### 3.7 Failure modes and the diagnostics that reveal them

| failure | mechanism | what reveals it | owner |
|---|---|---|---|
| a run that never stops early | $n_{\rm pat} = 99 > n_{\rm ep} - 1$: eq. (P5.8) false at every epoch | every record's `history` has exactly $n_{\rm ep}$ entries; no `early stop` line in any log | F-d |
| the restored weights are not the reported `val_npe` | the record stores `history[-1]["val_npe"]`, the last epoch's score, while `load_state_dict(best_state)` restores $\iota_{\rm best}$'s weights | `val_npe` of the record above the minimum of the history's `val_npe` series; `L` scored on the restored weights | F-aj |
| a clipped flow on the frozen arms | the stale encoder gradients of the pre-training inflate $\lVert \cdot \rVert_2$ in eq. (P5.2): a flow gradient of norm 5 with a stale norm of 5 is scaled by 0.71, of norm 1 with a stale 10 by 0.50 `[RAN]` B1 | nothing in the record; needs a smoke test that compares the clip coefficient with and without `backbone.zero_grad(set_to_none=True)` after the pre-training | F-ah |
| a batch-size axis read as a data-budget axis | an epoch is 25 steps at any $B_{\rm sim}$: a trial at 1024 sees eight times the rows of one at 128 (Table P5.1) | the partial dependence on `batch_size_npe` confounded with the amount of data; the runtime per trial | F-ag |
| a weak weight-decay axis | at most a 0.5 % cumulative shrink over 250 steps anywhere in the box, eq. (P5.6) | a flat partial dependence within $\sigma_{\rm seed}$ -- a correct reading of an inert axis, not evidence about regularisation | F-ai |
| a learning rate that diverges or stalls | constant $\eta$, no schedule, no stop | `npe` series against $0.56\, d_\theta$; `L` near or above $L_0$ | S3.3.1 |
| a second moment that remembers the first steps | $\mathcal{H}_2 = 1000 > n_{\rm plan}$; large early gradients (or a few clipped ones) set the denominator of eq. (P5.4) for the run | a step size that shrinks over the run without a schedule [reasoning]; nothing recorded | S3.2 |
| arms compared at one seed on different simulated batches | the probe's `batcher.next()` advances the simulated stream once per epoch when $\lambda_{\rm dsn} > 0$ | the plan's S2.4a already treats seeds as unpaired across arms; visible by regenerating the batches, not from the record | F-ak |
| the search optimises the report split | the tuner's `nll` is the record's `L`, scored on `theta[rp]`; across a campaign of hundreds of trials the report split is scored once per trial, and the plan's "touched once, at the end" (S4.4) no longer holds for it; the selection split, which the plan's S2.4 names for the arm comparison, decides only which epoch's weights are kept | the record's `n_sel`, `n_report` and `L`; `history[*]["val_npe"]` is the only selection-split quantity | F-al |
| the split moves with the seed | `grouped_split(seed)`: $\sigma_{\rm seed}$ includes split resampling; a Stage 4 trial and its controls at other seeds train on other rows | the `split_hash` of each record; the trial id is computed from the driver's `--split-hash` string, not from the runner's hash (P6) | S3.2 |
| the pre-training's decay | torch's $10^{-2}$ at $10^{-3}$ over 200 steps: 0.20 % `[RAN]` B4 | nothing recorded; small, but it is the one place decay acts at the runner's defaults | F-q |

### 3.8 Findings this document owns

The table of record is `00_INDEX.md` S6; the rows below are this document's,
with what P5 adds to each.

- **F-c (owned with P1).** `RANGE_PROVENANCE["weight_decay"]` names
  `SearchConfig.weight_decay_range` $(10^{-4}, 10^{-2})$ while the space
  carries `RegularizationConfig`'s $(10^{-5}, 10^{-2})$ (`joint_space.py:143,
  240`; `dsn/config.py:934, 1184`). What P5 adds: TUNING_1 S3.7 documents
  the same two ranges for the standalone search and says the `search` one
  is inert there, so the joint space copied the *active* range under the
  *inert* range's name -- right value, wrong string, as P1 said of
  `dropout`. Status: report; open (the F-ad kind of repair).
- **F-d (owned with P6).** `patience=99` (`run_joint_arms.py:522`) against
  `epochs=10`: eq. (P5.8) cannot hold before epoch 99 `[RAN]` B2. What P5
  adds: the rule's semantics are sbi's (`[RAN]` B2, 2000 / 2000 curves),
  so the value and not the rule is the problem; the DSN's `TrainConfig`
  warns on exactly this condition (`dsn/config.py:869-875`) and the joint
  `TrainConfig` does not; `control_config_from` plans for epochs-to-stop
  that no record carries (`joint_space.py:713-717`); and the library
  default 5 would fire from epoch 5 on. Status: report; open (a D-038
  candidate: a `--patience` flag with the library's 5, and a
  `TrainConfig` warning when `epochs <= patience`).
- **F-g (owned with P1, P2).** For the four optimiser ranges the DSN's
  dataclass and its 20 JSON configs agree `[REPO]` (S3.1, S3.5), so F-g's
  two-source problem does not arise in this block; the joint `lr` range is
  from a third source, the NPE tuner.
- **F-i (owned with P3).** The optimiser rows of the constants configured
  by code: `grad_clip = 5.0`, `beta2 = 0.999`, the split fractions, and --
  added here -- `patience = 99` in the runner's call, the $10^{-6}$
  improvement margin, `eps = 1e-8` and `amsgrad = False` from torch (S3.4).
  Status: report.
- **F-l (owned).** `batch_size_npe` reaches the runner as `--b-sim`
  (`npe_tune_joint.py:99`); `b_met` and `b_rep` are neither searched nor job
  variables (`joint_arms.pbs:50-62`). What P5 adds: because an epoch is a
  fixed step count, the axis moves the three streams' ratio from 32 : 8 : 1
  to 256 : 8 : 1 across its range (S3.3.4), and the amount of simulated
  data per trial with it (Table P5.1). Status: report.
- **F-m (optimiser rows).** `TrainConfig` (20 epochs, 50 steps, lr
  $5 \times 10^{-4}$, patience 5) against the runner (10, 25, $10^{-3}$,
  99); `BatchSpec` 512 / 64 / 8 against 128 / 32 / 4. What P5 adds: the
  further layers below the library -- sbi's `train()` ($5 \times 10^{-4}$,
  200, 20 fruitless epochs, Adam without decay) and torch's `AdamW`
  ($10^{-3}$, $10^{-2}$, $(0.9, 0.999)$, $10^{-8}$) -- read from the two
  sources; the torch values P0 Table C marked as "not read here" are now
  read (`[REPO torch v2.10.0]` `adamw.py:24-27`).
- **F-q (owned with P2).** `train_encoder_only` builds
  `torch.optim.AdamW(backbone.parameters(), lr=lr)` (`run_joint_arms.py:154`).
  What P5 adds: the defaults it takes are `weight_decay = 1e-2`, `betas =
  (0.9, 0.999)`, `eps = 1e-8` (`adamw.py:24-27`); at the runner's own
  defaults the betas coincide and the decay differs ($10^{-2}$ against
  the joint loop's 0), a 0.20 % shrink of every encoder weight over the 200
  steps `[RAN]` B4 -- the only decay a default Stage 3 campaign applies to
  anything; under a searched `one_minus_beta1` or `weight_decay` the two
  optimisers of an `A0` run differ in both. Status: report; open (D-038
  candidate, as the index's F-q row records).
- **F-r (optimiser rows).** `--b-sim 128` outside $\{256, 512, 1024\}$ and
  `--weight-decay 0.0` below $[10^{-5}, 10^{-2}]$; P5 adds that `--lr 1e-3`
  is inside its range with 77 % of the prior's mass below it and that
  `--one-minus-beta1 0.1` is the closed upper edge of its range, inside it
  but where `boundary_axes` flags a best configuration `[RAN]` B3.
- **F-ag (new).** The code's epoch is $n_{\rm step}$ optimiser steps on
  batches drawn with replacement (`joint_train.py:157`;
  `joint_batches.py:172`), while the plan's S5.1 ("One epoch is one pass
  over the simulated bank; at $B_{\rm sim} = 512$ that is about 58 steps,
  so the real cohort is revisited roughly 58 times per epoch"), the trim
  rule of `default_joint_space` ("a batch that swallows the epoch makes
  early stopping meaningless", `joint_space.py:283-285, 304-306`, copied
  from `npe_tune_search.py:126-129, 137-138`) and sbi's loop all mean a
  pass. Consequences `[RAN]` B2: at the runner's schedule ten epochs are
  1.5 pass-equivalents of the DUP15HD training rows at 128 and 12.3 at
  1024, with 21 % of the rows never drawn at 128; `batch_size_npe` is a
  data-budget axis (Table P5.1); the trim rule's thresholds (5 120 /
  10 240 / 20 480 rows) and its rationale refer to a loop the stack does
  not run, and the jobs never pass `--n-train` anyway; the plan's "58
  revisits of the real cohort per epoch" is, in the code, $n_{\rm step} = 25$
  metric batches per epoch at any $B_{\rm sim}$. `[REPO]` as cited;
  `[KB]` plan S5.1. Status: report; open (state the epoch's definition in
  the plan and the docstring, or make the rule read $n_{\rm step}$).
- **F-ah (new).** `clip_grad_norm_(model.parameters(), grad_clip)`
  (`joint_train.py:208-210`) measures every parameter whose `.grad` is not
  `None` (`clip_grad.py:230`). On `A0` and `A0s` the encoder's tensors
  carry the gradients of the last pre-training step: `train_encoder_only`
  zeroes before each backward and not after the last (`run_joint_arms.py:161-163`),
  then sets `requires_grad_(False)` (`:166-167`), which does not clear
  `.grad` [textbook, from memory: the flag and the stored gradient are
  independent attributes]; the loop's `opt.zero_grad` reaches only the
  optimiser's `trainable` tensors (`joint_train.py:140, 159`), and
  `freeze_encoder` touches the flag and the mode only (`joint_model.py:80-95`).
  So on those two arms the norm the clip measures is the root of the sum
  of the squared flow-gradient norm and the squared stale norm, the latter
  unchanged from step to step until the clip fires: the clip fires earlier
  and harder than on `A1` -- a flow gradient of norm 5 next to a stale norm
  of 5 is scaled by 0.71, of norm 1 next to 10 by 0.50 `[RAN]` B1 -- and
  each firing also rescales the stale tensors in place, so the stale norm
  shrinks with every firing and the effect fades over the steps on which
  clipping is active. `A3` loads the `A0` weights into a fresh backbone and is not
  affected; `A_ref`'s one frozen parameter never has a gradient. The
  `trainable` filter's comment gives a reason torch does not have (S3.4);
  the filter is nevertheless what keeps the stale tensors out of the
  optimiser. Not run (torch absent); the mechanism is read from the three
  sources. Status: report; open (D-038 candidate: `backbone.zero_grad(set_to_none=True)`
  at the end of `train_encoder_only`, or clip `trainable` instead of
  `model.parameters()`, with a smoke test that asserts the two
  coefficients agree).
- **F-ai (new).** Eq. (P5.6) at the corners of the searched box and the
  runner's schedule: the decoupled decay's cumulative factor over
  $n_{\rm plan} = 250$ steps lies between $1 - 2.5 \times 10^{-7}$ and
  $0.995$, the horizon $\mathcal{H}_{\rm wd}$ between $5 \times 10^{4}$ and
  $10^{9}$ steps `[RAN]` B1. The axis's direct effect on any weight is
  under one percent everywhere the search can go, and under 2 % at the
  library's 1000 steps; the DSN searched the same range with up to $10^{4}$
  steps per run. The partial dependence on `weight_decay` will read as
  flat for this reason, whatever the data say [reasoning on eq. (P5.6)].
  Status: report; open (whether to drop the axis, lengthen the schedule,
  or widen the range upward for the joint runs is a decision).
- **F-aj (new).** The runner's record stores `"val_npe":
  history[-1].get("val_npe")` (`run_joint_arms.py:594`), the last epoch's
  validation score, while the weights the record's `L`, $\hat\Delta$, the
  checkpoint and every diagnostic are computed on are the best epoch's
  (`joint_train.py:269-270`). The two coincide only when the last epoch is
  the best; the full series is in `history`, so nothing is lost, but the
  summary field misreports the selected model's validation score. The
  tuner reads `L`, not `val_npe` (`npe_tune_joint.py:517-525`), so the
  search is unaffected. Status: report; open (store
  `min(h["val_npe"] for h in history)` and `best_epoch`, which is also the
  epochs-to-stop the control docstring asks for).
- **F-ak (new).** When $\lambda_{\rm dsn} > 0$ the gradient-cosine probe
  draws one extra `batcher.next()` per epoch (`joint_train.py:236`), which
  advances the simulated, metric and replicate generators by one batch
  each. Two runs at the same `--seed` that differ in whether the DSN term
  is on -- `A1` against `A2`, or an S-A1 trial against an S-A2 trial with
  `dsn_on = 1` -- therefore share the simulated batches of epoch 0 and
  none thereafter (after ten epochs the `A2` stream is 1 280 rows ahead at
  $B_{\rm sim} = 128$ `[RAN]` B2). J9's independence of the streams holds
  within a run; across arms the seed does not fix the data sequence,
  which the plan's S2.4a already assumes when it declines to pair seeds
  across arms. Status: report (an observation on comparability, not a
  defect; drawing the probe batch from a fourth generator would remove
  it).
- **F-al (new).** The runner scores `L` on the report split
  (`per_row_nll(model, theta[rp], x[rp])`, `run_joint_arms.py:536-538,
  587`), and the tuner reads that `L` as the ledger's objective `nll`
  (`npe_tune_joint.py:524-525`); the selection split is read only by
  `evaluate_npe` once per epoch for the stopping rule and the best-state
  restore (`joint_train.py:227-232`). The plan assigns the roles the other
  way: its decision rule compares arms "on the frozen selection split"
  (S2.4) and the report split "is touched once, at the end" (S4.4); E0's
  row for $L_{\rm sel}$ follows the plan and calls it "the search
  objective". In the code the search objective is the report-split score,
  so across a Stage 4 campaign the report split is scored once per trial
  and selected on, and nothing is left that is touched once; the
  selection split's 15 % of donors serve only the per-epoch score, which
  at `patience=99` decides the restored epoch and nothing else (F-d).
  The distinction between "selection" and "report" is one of role, and
  the roles have crossed between the plan and the code; which assignment
  is wanted is a decision (the plan's, with `L` scored on `se` and the
  report split reserved for the finalists and the gates; or the code's,
  with the plan and E0 reworded). `[REPO]` as cited; `[KB]` plan S2.4,
  S4.4; E0 S1. Status: report; open (a decision for the log; E0 v1.5
  annotates its row rather than renaming).
- **A P0 wording note.** P0 Table C describes `grad_clip` as "global-norm
  gradient clipping over all trainable parameters"; the call is over
  `model.parameters()`, which differs from the trainable set on the frozen
  arms (F-ah). Noted for the Stage 6 fidelity review, not edited here (one
  document per turn).

## 4. Summary of results

1. One optimiser step is eq. (P5.1)-(P5.4): the three-stream batch loss,
   its gradient clipped to a 2-norm of $\gamma_{\rm clip} = 5$ over every
   parameter with a gradient, and torch 2.10.0's AdamW on the trainable
   tensors -- decay by $(1 - \eta \gamma_{\rm wd})$, two exponential moving
   averages, bias correction, the per-coordinate ratio times $\eta$ (S3.2).
2. The horizons: $\mathcal{H}_1 = 1/\upsilon_1 \in [10, 100]$ steps across
   the searched range, $\mathcal{H}_2 = 1000$ steps fixed,
   $\mathcal{H}_{\rm wd} \ge 5 \times 10^{4}$ steps anywhere in the box --
   against a planned run of $n_{\rm plan} = 250$ steps (eq. (P5.5)-(P5.6), `[RAN]` B1).
3. The step is invariant to a common rescaling of all loss terms up to
   $\epsilon_{\rm adam}$ and not to a per-step one; the weights
   $\lambda_{\rm dsn}$, $\lambda_{\rm rep}$ act through ratios, the clip
   through the steps on which it fires (S3.2, `[RAN]` B1).
4. An epoch is 25 steps with replacement; a run at the runner's schedule
   is 1.5 pass-equivalents of the DUP15HD training rows at
   $B_{\rm sim} = 128$ and 12.3 at 1024, covering 79 % and 100 % of them (Table P5.1,
   eq. (P5.9), `[RAN]` B2); the plan's, the trim rule's and sbi's epoch is
   a pass (F-ag).
5. Early stopping is sbi's rule with a $10^{-6}$ margin and cannot fire
   below 100 epochs at the runner's `patience=99` (eq. (P5.8), F-d);
   every run trains its full budget and restores its best epoch, while the
   record's `val_npe` is the last epoch's (F-aj).
6. The split is by donor at $(0.7, 0.15, 0.15)$ of the groups and moves
   with the seed; on the bench at seed 0 it is 352 / 80 / 80 rows (S3.2,
   `[RAN]` B2); $\sigma_{\rm seed}$ therefore includes split resampling.
   The selection rows feed the stopping rule alone; the report rows feed
   `L`, the ledger's objective -- the plan's assignment reversed (F-al).
7. The four axes are free in every campaign; `lr`'s range is the NPE
   tuner's (100 times narrower at the top than the DSN's), `one_minus_beta1`'s
   and `weight_decay`'s are the DSN's with the DSN's log-uniform argument,
   `batch_size_npe`'s set is the NPE tuner's; the runner's defaults sit
   outside, below, inside and on the edge of them respectively (S3.1,
   S3.3, S3.5).
8. Fixed by code with no flag: the clip, $\beta_2$, the patience, the
   split, the margin, torch's `eps`; fixed by the job: the schedule (S3.4).
9. The frozen arms' pre-training is a second AdamW on torch's defaults
   (F-q: a 0.20 % decay over 200 steps, `[RAN]` B4) whose last gradients
   stay on the encoder and enter the joint loop's clip (F-ah).
10. The findings: F-c, F-d, F-g, F-i, F-l, F-m, F-q, F-r carried with
    what P5 adds; F-ag to F-al new (S3.8).

## 5. Open points, caveats, assumptions

- **Nothing here is a training result.** No job of `hpc/joint/` has run;
  "the clip fires", "the second moment remembers", "the axis is weak" are
  properties of the equations at the configured values, not observations.
  The one empirical-looking statement, that the decay axis will read as
  flat, is reasoning on eq. (P5.6) and could be contradicted by a run in
  which the indirect effect of a 0.5 % shrink matters -- unlikely, not
  excluded.
- **Torch behaviour taken from memory.** That `requires_grad_(False)`
  leaves a populated `.grad` in place (F-ah) is standard torch behaviour
  but was not run here; the proposed smoke test is how it gets verified.
  Everything else about AdamW and the clip is read from the v2.10.0
  source.
- **The DUP15HD training-row count.** $20\,731$ is $0.7 \times 29\,616$;
  the real split cuts donors, and the joint stack's cohort bank with a
  `donor` field is not built (plan Stage 6), so the row fraction and the
  group count (the bank's 383 topology draws, plan S5.1) are placeholders
  for the arithmetic of Table P5.1. The bench numbers are exact at the Stage 1
  defaults.
- **Not read:** the DSN's `trainer.py` (how `batches_per_epoch` is applied
  and whether the DSN clips) beyond what `config.py` states; TUNING_2's
  S3.6-S3.8; the standalone tuner's ledger code. The D-036 table cites the
  TUNING sections, not the trainer.
- **Open calls this document raises or sharpens** (none decided here):
  F-d's `--patience` flag and warning; F-ah's zeroing of the stale
  gradients; F-ai's fate of the `weight_decay` axis at a 250-step budget;
  F-aj's summary field; F-ag's definition of the epoch in the plan and the
  trim rule; F-al's assignment of the selection and report splits; whether the schedule (`EPOCHS`, `STEPS_PER_EPOCH`) should be
  part of the trial id (P6); whether a learning-rate schedule is wanted at
  all, given that the loop was motivated partly by its reachability.
- **The seed's reach on the flow's initialisation** (S3.2) is read from
  the order of the calls; a test that builds `A0` and `A1` at one seed and
  compares the flow's initial weights would settle it in one line.

## 6. References / further reading

**Project knowledge base `[KB]`.** `JOINT_DSN_NPE_USAGE_v1.md` v1.3 S5 (the
`-v` variables), S9; `HPC_PATHS.md` sec. 7 (the environment's versions);
`claude/joint_docs/00_INDEX.md` S6 (F-c, F-d, F-g, F-i, F-l, F-m, F-q, F-r)
and S8; `claude/SBI_decisions_and_ideas_log.md` v1.18 (D-035 to D-038,
D-052, D-054); `claude/deck_pack/03_SEC_B_joint_objective.md`
B.4 (the explicit loop). `[KB-PDF]`: Deistler M et al. *Simulation-based
inference: a practical guide* (the project PDF): p.11 (tuning the batch
size, the learning rate and the early-stopping schedule; CPU training
competitive below batch sizes of 1000), p.32 (learning rate, batch size
and weight decay as the training hyper-parameters to search; the validation
NLL as the objective for NPE), p.40 (sbi's defaults: 20 fruitless epochs,
batch 200, Adam at $5 \times 10^{-4}$; a 4096 batch with 30 fruitless
epochs for the pyloric model). Goncalves PJ et al. *Training deep neural
density estimators ...* eLife 2020 (the project PDF): p.19 ("ADAM with
default settings"). Radev ST et al. *BayesFlow* (IEEE TNNLS 2022, the
project PDF): p.6 (Adam from $10^{-3}$ with exponential decay 0.95; 50 000
to 100 000 iterations). Gloeckler M et al. *Compositional simulation-based
inference for time series* (ICLR 2025, the project PDF): p.25 (AdamW at
$5 \times 10^{-4}$ with a cosine schedule, batch 1000, early stopping with
at most 5000 epochs; its FNLE and FNRE baselines, built on sbi's reference
implementations, at batch 1000 "until convergence, as determined by the
default early stopping routine"). Wildberger JB et al. *Flow matching for scalable
simulation-based inference* (NeurIPS 2023, the project PDF): p.27, Table 3
(batch 4096, learning rate $5 \times 10^{-4}$). Hermans J et al. *A trust
crisis in simulation-based inference?* (TMLR 2022, the project PDF): p.17,
Tables 2-4 (NPE at batch 128, 64 on the gravitational-wave task, 100
epochs, learning rate $10^{-3}$ on every task). Hikida
Y et al. *Multilevel neural simulation-based inference* (arXiv, the project
PDF; **PREPRINT, not peer-reviewed** [corrected 2026-10-03: the PDF's p.1 carries the line "39th Conference on Neural Information Processing Systems (NeurIPS 2025)", so the project PDF is the NeurIPS 2025 paper, peer-reviewed as a conference paper; P7 S6]): p.9 (sbi's 20-epoch criterion, 20 %
validation), p.27 (a patience parameter), p.30 (400 epochs). O'Callaghan M,
Mandel KS, Gilmore G. *Misspecification-robust amortised simulation-based
inference using variational methods* (the project PDF; the PDF's first page
names no venue, so its review status is not verified here): p.40 (batch
1024 over 500 iterations; Adam at $10^{-3}$ with $\beta_1 = 0.9$, weight
decay $10^{-5}$, gradient clipping at 10, a 1000-step warm-up schedule,
patience 100 iterations, 10 % validation). Each recipe is cited as
what its authors ran, none as a prescription for this bank.

**Repository `[REPO]`** at `834eb41`, read 2026-10-02: the files and lines of
the changelog row; `dsn/Documentation/TUNING_1_searched_axes.md` S3.5
(`lr`), S3.6 (`one_minus_beta1`, `one_minus_beta2`), S3.7 (`weight_decay`,
"a trap"), S3.8 (`dropout`, the staged pipeline); `TUNING_2_fixed_knobs.md`
S3.2 (budget and schedule), S3.4 (selection and early stopping). **The sbi
wheel** `sbi` 0.27.0 (sha256 `e7ef7800...27e0b8`): `inference/trainers/base.py`,
`inference/trainers/npe/npe_base.py` as cited. **torch 2.10.0**, read from
GitHub at tag `v2.10.0`: `torch/optim/adamw.py` (the class, its defaults,
`decoupled_weight_decay=True`, the documented algorithm), `torch/optim/adam.py`
(`_init_group`, `_single_tensor_adam`), `torch/nn/utils/clip_grad.py`
(`_get_total_norm`, `_clip_grads_with_norm_`, `clip_grad_norm_`).

**Peer-reviewed literature.** According to PubMed, read in full text from
PubMed Central: Sclocchi A, Wyart M. *On the different regimes of
stochastic gradient descent.* Proc Natl Acad Sci USA 2024; PMID 38377198,
PMC10907278, [DOI](https://doi.org/10.1073/pnas.2316301121) -- the
temperature (learning rate over batch size) description of small-batch SGD,
its three regimes, and
the authors' statement that momentum, adaptive learning rates and weight
decay are outside their analysis; used in S3.3.4 for that scope statement
and **no number from it is used**. Zeng L, Tang W, Ren Z, Ding Y.
*Mini-batch estimation for deep Cox models: statistical foundations and
practical guidance.* J Am Stat Assoc 2026; PMID 42577746, PMC13455505,
[DOI](https://doi.org/10.1080/01621459.2026.2644611) -- the linear scaling
rule restated for SGD as a reason to fix one of learning rate and batch
size; used in S3.3.4 qualitatively; no number used. Boelts J, Lueckmann
J-M, Gao R, Macke JH. *Flexible and efficient simulation-based inference for
models of decision-making.* eLife 2022; PMID 35894305, PMC9374439,
[DOI](https://doi.org/10.7554/eLife.77220) -- "the maximum likelihood loss
and the Adam optimizer"; no number used. Deistler M, Macke JH, Goncalves PJ.
*Energy-efficient network activity from disparate circuit parameters.* Proc
Natl Acad Sci USA 2022; PMID 36279461, PMC9636970,
[DOI](https://doi.org/10.1073/pnas.2207632119) -- read in full text; its PMC
text carries no training hyper-parameters; nothing used.

Retrieved, **abstract only, therefore not used for any claim**: Zhou P, Xie
X, Lin Z, Yan S. *Towards understanding convergence and generalization of
AdamW.* IEEE Trans Pattern Anal Mach Intell 2024; PMID 38536692,
[DOI](https://doi.org/10.1109/TPAMI.2024.3382294) (no PMC full text). Full
text not accessible for PMID 38536692 -- this looks important for the
weight-decay axis as the one indexed theory of decoupled decay; if you can
obtain the PDF, upload it and S3.3.3 can carry its statement of what AdamW
minimises. Not in PubMed and not in the knowledge base: Kingma DP, Ba J.
*Adam: a method for stochastic optimization* (ICLR 2015) and Loshchilov I,
Hutter F. *Decoupled weight decay regularization* (ICLR 2019), the sources
torch's docstring names; every statement about the update here is read from
torch's implementation, not from those papers.

**Searches run `[RAN]`, 2026-10-02.**

| source | query | result |
|---|---|---|
| PubMed | "decoupled weight decay" OR AdamW optimizer | 63 records; one on the method (PMID 38536692, abstract only, flagged above); the rest applications that use AdamW, none opened |
| PubMed | Adam optimizer learning rate beta1 beta2 exponential decay rates hyperparameter sensitivity neural network training | 0 records |
| PubMed | gradient norm clipping neural network training stability | 0 records |
| PubMed | early stopping patience validation loss overfitting neural network | 1 record (PMID 41601813, a pneumonia CNN; not on topic, not opened) |
| PubMed | (neural posterior estimation OR simulation-based inference) AND (early stopping OR learning rate OR Adam) AND (normalizing flow OR density estimator) | 1 record (PMID 19449095, Bayesian binning of spike trains; not on topic) |
| PubMed | weight decay regularization neural network generalization L2 penalty | 1 record (PMID 30530381, group-lasso MLPs; not on topic) |
| PubMed | batch size learning rate trade-off stochastic gradient descent generalization gap | 0 records |
| PubMed | Kingma Ba Adam stochastic optimization adaptive moment estimation | 0 records |
| PubMed | "gradient clipping" | 30 records; applications and differential-privacy papers; none on the clip as a training device; none opened |
| PubMed | "early stopping" AND "validation" AND (overfitting OR generalization) AND ("deep learning" OR "neural network") AND (review OR tutorial OR guide) | 4 records, all clinical applications; none opened |
| PubMed | Boelts Lueckmann Gao Macke flexible efficient simulation-based inference decision-making | 1 record (PMC9374439, read in full) |
| PubMed | Deistler Macke Goncalves energy-efficient network activity disparate circuit parameters | 1 record (PMC9636970, read in full; nothing usable) |
| PubMed | "learning rate" AND "batch size" AND ("linear scaling" OR "noise scale" OR "temperature") AND stochastic gradient | 2 records (PMC10907278, PMC13455505; both read in full and used qualitatively) |
| PubMed | (sampling "with replacement" OR "without replacement") AND (stochastic gradient OR mini-batch) AND (neural network OR deep learning) AND (epoch OR convergence) | 0 records |
| PubMed | "exponential moving average" AND (Adam OR momentum) AND gradient AND ("bias correction" OR "effective horizon" OR timescale) | 0 records |
| bioRxiv | neuroscience, last 30 days, 30 records (the connector has no keyword search) | none on optimisers, training schedules or NPE; no preprint is cited |
| bioRxiv | bioinformatics, last 30 days, 30 records | none on topic; no preprint is cited |

**Textbook, from memory** (tagged as such where used): that
`requires_grad_(False)` does not clear a tensor's `.grad` (F-ah); the
closed forms of a geometric series behind eq. (P5.5)-(P5.6), whose values
are checked numerically `[RAN]` B1.

---

### Pre-send check (Precision model)

R1 -- every symbol typed in S1; the update of eq. (P5.4) relates vectors in
$\mathbb{R}^{n_\varpi}$ elementwise; horizons are step counts; fractions of
groups and fractions of rows are kept apart. R2 -- "cannot fire" carries
"at `epochs=10`"; "inert" carries "at $n_{\rm plan} = 250$ within the
searched box"; the SGD statements carry "for plain SGD" and the authors'
own scope limit; the coverage numbers carry "with replacement" and the
placeholder row count; the stale-gradient mechanism carries "on `A0`,
`A0s`" and "not run". R3 -- the linear scaling rule and the temperature
regime are *not* transplanted from SGD to AdamW, and the document says why
(the per-coordinate normalisation); sbi's stopping rule is compared with the
joint rule by simulation before being called the same; TUNING_1's
timescale argument is quoted for the object it was made for (the
complement of an Adam decay rate). R4 -- "epoch" and "pass" are two names for two objects; $\tau$,
$n_{\rm done}$ and $t$ are three related but distinct counters; the runner's,
the library's, sbi's, torch's and the DSN's defaults are named per surface;
$\upsilon_1$ is not TUNING_1's $u_1$ in symbol because $u$ is taken. R5 --
$\hat{\mathcal{L}}_{\rm step} : \mathbb{R}^{n_\varpi} \to \mathbb{R}$, the
clip $\mathbb{R}^{n_\varpi} \to \mathbb{R}^{n_\varpi}$ and the split as a map
from donors to the three parts are named with domain and codomain. R6 --
"weight decay" (decoupled, not penalty), "clip" (norm rescale, not clamp),
"epoch", "patience", "horizon" declared in S1.1 or S2. R7 -- the plan's
"58 steps" is quoted with the semantics it assumes; the docstrings' reasons
are quoted with what they omit; the trust-crisis and compositional recipes
are quoted as what their authors ran. R8 -- $\mathcal{L}$ (expectation),
$\hat{\mathcal{L}}_{\rm step}$ (one batch), $L_{\rm sel}$ (one epoch, one
split) and $L$ (the run, the report split) are four levels with the moves
named in convention 5; the moments are computed state and their bars are
torch's normalisation, not the set's hat. Confirmation questions: none in
the brief.
