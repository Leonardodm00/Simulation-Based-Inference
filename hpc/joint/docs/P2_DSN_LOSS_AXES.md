# P2 -- The DSN loss block: the seven searched axes and the fixed knobs of the metric term

**Document P2 of the joint documentation set.** Owner of the `dsn_loss` block
of `JOINT_KNOB_ORDER` (`dsn_on`, `log10_lambda_dsn`, `loss_type`,
`mining_strategy`, `margin`, `angular_alpha_deg`, `lambda_sep`) and of the
knobs of the metric term the joint stack holds fixed or never sets
(`strict_semihard`, `sep_warmup_frac`, `swap`, `use_angular`,
`reduce_nonzero`, `min_per_class`, the distance family, the class count the
separation target is built from, `--b-met`, `--encoder-steps`). Master
notation: E0. The chapter that explains what the metric term is *for* and
why its minimum is a problem: E3. **Date:** 2026-10-01 (v1). **Applies to:**
the repository `Simulation-Based-Inference` at `834eb41`, `hpc/joint/` and
the `hpc/dsn/train.py`, `dsn_joint_loss.py`, `condition_space.py`,
`config.py` it reads (D-037), and `pytorch_metric_learning` 1.6.3, the
version of the cluster's `sbi_env` (`[KB]` migration handoff S3.1), whose
miners and triplet loss the term is built from.

| date | change |
|---|---|
| 2026-10-05 | v1.2. One dated note from E4, nothing else changed: S3.3.2's "What it represents, plainly" now marks "at $10$ it dominates every step" as the weighted-sum reading and gives the AdamW one -- a weight acts on the encoder coordinate by coordinate against the two gradients' sizes and saturates at both ends (E4 eq. (E4.6)), and on the flow only through the clip's step-to-step variation (E4 eq. (E4.5)). Evidence: E4 S3.4; `tools/e4_numbers.py` B2 `[RAN 2026-10-05]`. |
| 2026-10-05 | v1.1. Dated notes from E3, nothing else changed: the glossary's "Collapse", S3.2's closing reading of the constants, the flow bullet of S3.6, and the S3.7 rows "collapse" and "pre-training without balance" now carry the conditions E3 derives -- the expected loss is zero exactly at class collapse only under `joint_sep` (E3 eq. (E3.3)), the margin and the half-angle constrain the residual's size and not its information (E3 eq. (E3.9)), the cap on what the flow can distinguish needs an exact code on the rows the flow is scored on (E3 eq. (E3.6)), `r_eff` is met at $C = 2$ by a constant encoder or a thin line (F-bb), and at $C \ge 3$ the pre-training's two-class batches change the separation target (F-bc). Evidence: E3 S3.4-S3.7; `tools/e3_numbers.py` B2-B4 `[RAN 2026-10-05]`. |
| 2026-10-01 | v1. Written from `hpc/dsn/dsn_joint_loss.py` (read in full), `hpc/dsn/train.py` `build_loss_and_miner` (`:216-351`) and its notation block (`:55-97`), `hpc/dsn/condition_space.py` (read in full), `hpc/dsn/config.py` (`TrainConfig` `:615-724`, `SearchConfig` `:899-930`, `:1002-1004`), `hpc/dsn/hpc/preflight_config.py:140-200`, `hpc/dsn/Documentation/TUNING_1_searched_axes.md` S3.9-3.14, 3.17, `TUNING_2_fixed_knobs.md` S3.3, `THEORY_joint_condition_search.md` S3.3, 3.6, 3.7; `joint/stage2/dsn_loss_adapter.py`, `joint_batches.py`, `joint_train.py` (all read in full), `joint/stage3/run_joint_arms.py` (the loss path), `joint/stage4/joint_space.py` (the loss block, the canonicalisation, the campaigns), `joint/stage4/npe_tune_joint.py:86-227`, `joint/stage3/jobs/joint_arms.pbs:50-115`; the installed library's source, `pytorch_metric_learning` 1.6.3 (`distances/base_distance.py`, `losses/triplet_margin_loss.py`, `miners/triplet_margin_miner.py`, `miners/batch_easy_hard_miner.py`, `utils/loss_and_miner_utils.py`, `reducers/threshold_reducer.py`), read from the wheel. The constants and identities of S3.2 recomputed in the sandbox `[RAN]`. Findings F-t to F-x added; F-f extended. Grounding searches of S6 run and reported. |

**Abstract.** The seven axes of the `dsn_loss` block are the only axes of
the joint search that change *what the encoder is asked to do with the
label*: whether the metric term is on at all and how heavily it weighs
against the likelihood term ($\lambda_{\rm dsn}$), which objective it is
(`loss_type`), which triplets of a batch it is evaluated on
(`mining_strategy`), and the three constants that set how tight a geometry
it demands ($m_{\cos}$, $\alpha$, $\lambda_{\rm sep}$). The question this
document answers is, for each of these seven and for the knobs held fixed
around them: where the value is set and by which surface it reaches the
trainer; what the term computes, written out from the code as one explicit
function of the knobs (S3.2); what each knob changes in that function and
therefore in the joint objective; how the knobs interact, including the
legality projection and the activity mask that make some coordinates
inert; how they fail; and which number in the run record reveals the
failure. **Covered:** the per-parameter fields of plan S2.2 for all seven
axes (S3.3) and for the ten fixed knobs (S3.4); the loss as built, with its
distance conventions, miners, filter, reduction, separation target and
schedule, eq. (P2.1)-(P2.13), every constant recomputed `[RAN]`; the
differences between the joint stack's loss and the standalone DSN's (D-036,
S3.5); the findings this document owns (F-f, F-h, F-q with P5, and the new
F-t to F-x). **Deliberately excluded:** the network the term trains (P1),
the replicate term (P3), the flow (P4), the optimiser (P5), the search
mechanics that canonicalise these axes (P6), and the information argument
for why a label-trained encoder caps the gain at $\log C$ nats (E3, with
`[KB]` deck section C). Nothing here is a measurement of training
behaviour: no job of `hpc/joint/` has run on the cluster (`[KB]` usage v1.3
S9); every number is a property of the loss as built, read from the code or
recomputed from it.

---

## 1. Notation and symbols

A subset of E0's master table, same types and units, plus the symbols this
document adds (declared in E0 under its convention 14).

| Symbol | Name / Meaning | Type & domain | Units | First used in S |
|---|---|---|---|---|
| $x$ | one IFR window | $x \in \mathbb{R}^{W}_{\ge 0}$ | Hz (cohort); counts per bin per unit (bench) | S3.1 |
| $h_\psi$ | the encoder, $h_\psi : \mathbb{R}^{W} \to S^{E-1}$ | map; weights $\psi$ | -- | S3.1 |
| $\psi$ | encoder weights | $\psi \in \mathbb{R}^{n_\psi}$ | -- | S3.1 |
| $z$ | the embedding of a window, $z = h_\psi(x)$; $z_i$ for row $i$ of a batch | $z \in S^{E-1} \subset \mathbb{R}^{E}$ | dimensionless | S3.2 |
| $S^{E-1}$ | the unit sphere in $\mathbb{R}^{E}$ | set | -- | S3.2 |
| $E$ | embedding dimension (`embedding_size`) | $\mathbb{N}$ | -- | S3.2 |
| $i$ | row index of the metric batch; $i'$ a positive row, $i''$ a negative row of the same batch (instances of the one row index) | $i \in \{1, \dots, n\}$ | -- | S3.2 |
| $c$ | class label | $c \in \{0, \dots, C - 1\}$ | -- | S3.2 |
| $C$ | number of classes the loss is configured with (`n_classes`) | $\mathbb{N}$ | -- | S3.2 |
| $y, y_i$ | the labels of the metric batch, and the label of its row $i$ (E0's $y^{\rm real}_{\rm met}$, indexed) | labels in $\{0, \dots, C-1\}$ | -- | S3.2 |
| $X^{\rm real}_{\rm met}, y^{\rm real}_{\rm met}$ | the metric stream's batch and its labels; real rows and labels for arms `A0`, `A2`, `A3`, simulated rows with generator labels for `A0s`, `A2s` | batch of rows; labels | -- | S3.2 |
| $B_{\rm met}$ | rows requested per step in the metric stream (`--b-met`) | $\mathbb{N}$ | rows | S3.2 |
| $\mathcal{P}_i$ | the positives of row $i$: same label, other row | index set | -- | S3.2 |
| $\mathcal{O}_i$ | the negatives of row $i$: other label | index set | -- | S3.2 |
| $d_{\cos}$ | cosine distance, $d_{\cos}(z, z') = 1 - z^\top z'$ for unit rows | $[0, 2]$ | dimensionless | S3.2 |
| $Q$ | squared Euclidean distance between two rows of a batch, $Q(i, i') = \lVert z_i - z_{i'} \rVert_2^2$ | $[0, 4]$ | dimensionless | S3.2 |
| $Q_{\rm mid}$ | squared Euclidean distance from the negative to the anchor-positive midpoint, $Q_{\rm mid}(i, i', i'') = \lVert z_{i''} - \frac{1}{2}(z_i + z_{i'}) \rVert_2^2$ | $[0, 4]$ | dimensionless | S3.2 |
| $m_{\cos}$ | triplet margin in cosine-distance units (`margin`) | $(0, 1)$; searched in $[0.1, 1.0]$ | dimensionless | S3.2 |
| $m_{\rm sq}$ | the same margin in squared-Euclidean units, $m_{\rm sq} = 2 m_{\cos}$ | $(0, 2)$ | dimensionless | S3.2 |
| $\alpha$ | angular half-angle of the angular hinge (`angular_alpha_deg`) | $(0^\circ, 90^\circ)$; searched in $[2^\circ, 20^\circ]$ | degrees | S3.2 |
| $\mathcal{T}_{\rm mined}$ | the triplets the miner returns for a batch | set of index triples $(i, i', i'')$ | -- | S3.2 |
| $\mathcal{T}_{\rm strict}$ | the mined triplets that survive the strict semi-hard filter (all of them when the filter is off) | set of index triples | -- | S3.2 |
| $\ell_{\rm trip}, \ell_{\rm ang}$ | per-triplet margin hinge and angular hinge | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\ell_{\rm pml}$ | the library's per-triplet triplet-margin loss under `loss_type = triplet`, in cosine units | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $n_{\rm mined}, n_{\rm strict}, n_{\rm act}$ | number of mined, strict-filtered and active (positive-loss) triplets of a batch; the loss's `n_mined`, `n_strict`, `n_active` | $\mathbb{N}_0$ | -- | S3.2 |
| $\mathcal{L}_{\rm joint}$ | the margin-plus-angular part of the loss of one batch, eq. (P2.6) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $n_c$ | rows of class $c$ in the metric batch | $\mathbb{N}_0$ | rows | S3.2 |
| $\bar z^{(B)}_c$ | class-$c$ mean embedding over the rows of one metric batch (computed level; E0's $\bar z_c$ is its population counterpart, analytic level) | $\mathbb{R}^{E}$ | dimensionless | S3.2 |
| $v_c$ | unit direction of the batch class mean, $v_c = \bar z^{(B)}_c / \lVert \bar z^{(B)}_c \rVert_2$ | $S^{E-1}$ | dimensionless | S3.2 |
| $K$ | classes present in the batch with at least `min_per_class` rows (computed level; $K \le C$) | $\mathbb{N}_0$ | -- | S3.2 |
| $\rho_{\rm ETF}$ | the simplex-ETF target cosine, $\rho_{\rm ETF} = -1/(K-1)$ | $[-1, 0)$ | dimensionless | S3.2 |
| $\mathcal{L}_{\rm sep}$ | the centroid-separation penalty of one batch, eq. (P2.8) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\lambda_{\rm sep}$ | asymptotic weight of $\mathcal{L}_{\rm sep}$ (`lambda_sep`); scheduled as $\lambda_{\rm sep}(t)$ | $\mathbb{R}_{\ge 0}$; searched in $[10^{-3}, 1]$ | dimensionless | S3.2 |
| $\tau_{\rm sep}$ | warm-up fraction of the separation weight (`sep_warmup_frac`); fixed at 0 in the joint space | $[0, 1]$ | dimensionless | S3.2 |
| $t$ | training progress as a fraction of the planned optimiser steps, $t = n_{\rm done} / n_{\rm plan}$ | $t \in [0, 1]$ | -- | S3.2 |
| $n_{\rm done}$ | optimiser steps completed so far, as the ramp counts them (one per evaluation of $\ell_{\rm DSN}$) | $\mathbb{N}_0$ | steps | S3.2 |
| $n_{\rm plan}$ | the planned step budget: $n_{\rm ep} n_{\rm step}$ in the joint loop, `--encoder-steps` in the encoder-only pre-training | $\mathbb{N}$ | steps | S3.2 |
| $n_{\rm ep}, n_{\rm step}$ | epochs, and optimiser steps per epoch (`epochs`, `steps_per_epoch`) | $\mathbb{N}$ | -- | S3.2 |
| $\gamma_{\rm sep}$ | the ramp factor of the separation weight, $\gamma_{\rm sep}(t) \in [0, 1]$ | $[0, 1]$ | dimensionless | S3.2 |
| $\ell_{\rm DSN}$ | the metric loss of one batch under the configured `loss_type`, eq. (P2.10) (computed level) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\mathcal{L}^{\rm real}_{\rm DSN}$ | its expectation over the metric stream's distribution (analytic level), plan eq. (1a) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\lambda_{\rm dsn}$ | weight of the DSN term in $\mathcal{L}$, plan eq. (1); searched through $\log_{10} \lambda_{\rm dsn} \in [-3, 1]$ behind the switch `dsn_on` | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\log_{10}\lambda_{\rm dsn}$ | the searched coordinate of $\lambda_{\rm dsn}$ (`log10_lambda_dsn`) | $[-3, 1]$ | -- | S3.3 |
| $\mathcal{L}$ | the joint objective of plan eq. (1), $\mathcal{L}(\psi, \omega)$ | $\mathbb{R}$ | nats/row plus dimensionless terms | S3.2 |
| $\mathcal{L}^{\rm sim}_{\rm NPE}$ | the NPE term of the objective | $\mathbb{R}$ | nats/row | S3.2 |
| $\hat{\mathcal{L}}_{\rm step}$ | the per-step estimate of $\mathcal{L}$, plan eq. (2) | $\mathbb{R}$ | nats/row plus dimensionless terms | S3.2 |
| $\mathcal{M}_{\rm trip}, \mathcal{M}_{\rm joint}, \mathcal{M}_{\rm jsep}$ | the activity masks: the loss hyper-parameters each `loss_type` reads (`condition_space.active_loss_hps`) | sets of knobs | -- | S3.3 |
| $\Pi$ | the legality projection of a (`mining_strategy`, `loss_type`, `strict_semihard`) triple onto a legal one (`condition_space.project_condition`) | map on triples | -- | S3.3 |
| $S_{\rm sil}$ | the cosine silhouette of an embedding cloud against its labels; derivation-only in this stack except as the record's `cluster_silhouette` on the simulated report split | $[-1, 1]$ | dimensionless | S3.2 |
| $\rho_{\rm grad}$ | cosine between the NPE and DSN gradients with respect to $\psi$ | $[-1, 1]$ | -- | S3.6 |
| $r_{\rm eff}$ | participation-ratio effective rank of an embedding cloud | $[1, E]$ | -- | S3.7 |
| $\hat\Delta$ | information gain of an arm, prior floor minus held-out NLL | $\mathbb{R}$ | nats/row | S3.7 |
| $\bar z_c$ | population mean embedding of class $c$ (E0; analytic level) | $\mathbb{R}^{E}$ | dimensionless | S3.2 |
| $\xi$ | within-class residual of an embedding, $\xi = z - \bar z_c$ | $\mathbb{R}^{E}$ | dimensionless | S3.7 |
| $q_\omega$ | the flow, $q_\omega(\theta \mid z)$ | conditional density | -- | S3.6 |
| $\theta$ | the inference parameters | $\theta \in \Theta$ | mixed | S3.6 |
| $\Theta$ | the prior box | set | -- | S3.6 |
| $\gamma_{\rm wd}$ | the AdamW weight-decay coefficient | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.6 |
| $p$ | latent dimension of the bank (the search's `--p` anchor) | $\mathbb{N}$ | -- | S3.6 |
| $d_\theta$ | parameter-space dimension | $\mathbb{N}$ | -- | S3.6 |

### 1.1 Conventions

- Code names in backticks are the knobs as the code spells them; the symbol
  of the same quantity is used only where a formula needs it. The letters
  TUNING_1 uses for the categorical axes (`m`, `ell`, `s` for the mining
  strategy, the loss type and the filter flag) are **not** used here, because
  E0 reserves $m_g$, $\ell_i$ and $s$ for other objects; the three
  categoricals are always written in code font.
- **Two distance units, never mixed silently.** $m_{\cos}$ is what the
  configuration states and what the miner uses; $m_{\rm sq} = 2 m_{\cos}$ is
  what the composite loss uses (S3.2, eq. (P2.2)). Every inequality of S3.2
  says which unit it is in. $Q$ is not the flow $q_\omega$.
- **Levels** (E0 convention 4): $\ell_{\rm DSN}$ is one batch's number
  (computed); $\mathcal{L}^{\rm real}_{\rm DSN}$ is its expectation
  (analytic); $\bar z^{(B)}_c$ and $K$ are batch quantities, $\bar z_c$ and
  $C$ their population and configured counterparts. $t$ is a fraction, as in
  E0; the DSN's own documents write the schedule in step counts (their
  `t` is this document's $n_{\rm done}$), and eq. (P2.9) is the same schedule
  in the fraction.
- **Status** (the Provenance model): every knob of this document is
  `configured`; $\mathcal{T}_{\rm mined}$, $\mathcal{T}_{\rm strict}$,
  $n_{\rm act}$, $\bar z^{(B)}_c$, $K$, $\rho_{\rm ETF}$, $\gamma_{\rm sep}$,
  $\ell_{\rm DSN}$ are `computed` from the batch and the configuration;
  $\mathcal{L}^{\rm real}_{\rm DSN}$ is `derivation-only` (never evaluated;
  its per-step estimate is what trains); $S_{\rm sil}$ is `derivation-only`
  inside the loss (the silhouette gate that read it is no longer built) and
  `computed` once per run as `cluster_silhouette` on the simulated report
  split (S3.7).
- **Default** is per surface, as P0 S1.1: the runner's flag default is what a
  Stage 3 or Stage 4 run trains with when the flag is not passed;
  `DSNLossConfig`'s default is what a direct caller of the adapter gets;
  `TrainConfig`'s default is the standalone DSN's and is never read by the
  joint stack (F-h).
- Equations are numbered (P2.n); plan equations are cited as "plan eq. (n)";
  TUNING_1's as "TUNING_1 eq. (n)".
- "Active" has two senses here and both are used: an *active axis* is one
  the configured `loss_type` reads (the masks of S3.3); an *active triplet*
  is one with a positive hinge value ($n_{\rm act}$, S3.2). "Hard" likewise:
  a *hard negative* is the closest negative of an anchor (S3.2), while the
  `hard` mining strategy is a named level that returns every margin-violating
  triplet with the negative at least as close as the positive (S3.3.4).

## 2. Glossary

Ordered by first appearance.

- **Metric stream** -- the second of the three batch streams of plan eq. (2):
  class-balanced labelled rows pushed through $h_\psi$ and scored by
  $\ell_{\rm DSN}$; real rows for `A0`/`A2`/`A3`, simulated rows with the
  generator's labels for `A0s`/`A2s`. S3.1.
- **Triplet** -- an ordered triple of batch rows (anchor $i$, positive $i'$
  with $y_{i'} = y_i$, negative $i''$ with $y_{i''} \ne y_i$). S3.2.
- **Miner** -- the rule that selects which triplets of a batch the loss is
  evaluated on; here one of three `pytorch_metric_learning` miners under the
  cosine distance. *Everyday meaning differs*: nothing is dug up; the miner
  is a filter over the batch's own index pairs. S3.2.
- **Margin hinge (triplet loss)** -- the penalty $[\,Q(i,i') - Q(i,i'') +
  m_{\rm sq}\,]_+$ that is positive until the negative is farther from the
  anchor than the positive by at least the margin. S3.2.
- **Swap** -- using the smaller of the anchor-negative and positive-negative
  distances in the hinge, so that the hinge is symmetric in anchor and
  positive. S3.2.
- **Angular hinge** -- the penalty $[\,Q(i,i') - 4\tan^2\!\alpha\;
  Q_{\rm mid}(i,i',i'')\,]_+$, zero when the negative sits outside a cone of
  half-angle $\alpha$ around the anchor-positive pair. S3.2.
- **Strict semi-hard filter** -- keeping a triplet only when the negative
  lies inside the margin band beyond the positive, as seen from the anchor
  *and* as seen from the positive. S3.2.
- **Hard / semi-hard / easy negative** -- the closest negative of an anchor;
  a negative farther than the positive but still a margin violator (the
  library's `semihard` level keeps the first condition and drops the margin
  bound, S3.2); a negative outside the margin. S3.2.
- **Easy positive** -- the closest same-class row of an anchor. S3.2.
- **Centroid separation (simplex ETF) penalty** -- the term that pushes the
  unit directions of the batch class means toward pairwise cosine
  $-1/(K-1)$, the maximally spread configuration of $K$ unit vectors. S3.2.
- **Simplex equiangular tight frame (ETF)** -- $K$ unit vectors with equal
  pairwise inner products $-1/(K-1)$ (antipodal at $K = 2$, $120^\circ$ apart
  at $K = 3$); the geometry neural-collapse theory finds class means
  converging to. S3.2.
- **Warm-up (ramp)** -- the deterministic linear schedule that scales the
  separation weight from 0 to $\lambda_{\rm sep}$ over the first
  $\tau_{\rm sep}$ of the planned steps; off ($\gamma_{\rm sep} \equiv 1$)
  in the joint space. S3.2.
- **Activity mask** -- the set of loss hyper-parameters a `loss_type`
  reads; the others are pinned to canonical values so that two
  configurations differing only in an unread coordinate are the same
  configuration. S3.3.
- **Legality projection** -- the map that sets `strict_semihard` to 0 where
  it is unread (`triplet`) or provably empty (`hard` mining). S3.3.
- **Campaign switch** -- a binary axis (`dsn_on`) that gates a whole block:
  off, the block's axes are pinned and the term's weight is 0. S3.3.
- **Collapse** -- the state in which every window of a class maps to one
  point; the global minimum of the composite loss and the reason the metric
  term caps the information an encoder can carry (E3). S3.7. [corrected
  2026-10-05, E3 S3.4, S3.6: a global minimiser -- the zero of the expected
  loss -- and the only one only under `joint_sep` with
  $\lambda_{\rm sep} > 0$, E3 eq. (E3.3); under `triplet` and `joint`
  every geometry with separated classes is a zero; and the cap on
  information needs the collapse to hold on the rows the gain is scored on,
  E3 eq. (E3.6).]

## 3. Main body

### 3.1 Where the loss knobs live, and what reaches the trainer

This section establishes the chain by which each knob of the block reaches
the loss object, and which surfaces can set it. Source tag `[REPO]`.

The chain has five surfaces. (i) The **space** (`joint_space.py`): seven
searched axes in `JOINT_KNOB_ORDER` positions 7-13 (`:92-105`, `BLOCKS`
`:107-117`), with `strict_semihard` and `sep_warmup_frac` carried in
`spec.fixed` (`:322-324`). (ii) The **tuner** (`npe_tune_joint.py`):
`AXIS_TO_FLAG` maps five axes to runner flags (`:101-105`), `DERIVED` maps
`dsn_on` and `log10_lambda_dsn` to `--lambda-dsn` (`:112-117`, `:213-217`),
and `build_argv` passes `--strict-semihard` from `spec.fixed` (`:206-211`)
but **not** `--sep-warmup-frac` (F-w). (iii) The **runner**
(`run_joint_arms.py`): the flags (`:222`, `:226`, `:249-257`), the legality
projection applied to the received triple (`:415-428`), the
`DSNLossConfig` (`:429-436`) and the call that builds the loss (`:437-439`);
the arm decides whether the term is on and on which domain (`arm_config`
`:175-202`). (iv) The **adapter** (`stage2/dsn_loss_adapter.py`):
`DSNLossConfig` holds exactly the fields `build_loss_and_miner` reads
(`:48-73`), with its own defaults (`:56-60`); `build_dsn_loss` refuses a
non-zero warm-up fraction without a horizon (`:169-173`) and wraps the loss
and the miner into one callable $(z, c) \mapsto \ell_{\rm DSN}$ (`:89-107`).
(v) The **DSN** (`hpc/dsn/train.py:216-351`): `build_loss_and_miner`
dispatches on `loss_type`, converts the margin once, and builds the miner
under `CosineSimilarity`; the loss objects are in `dsn_joint_loss.py`.

| knob | surface where it is set | reaches the loss as | runner default | job variable (`joint_arms.pbs`) | searched |
|---|---|---|---|---|---|
| `dsn_on` | space (campaign switch) | `--lambda-dsn 0` when 0 | -- (the arm decides: `arm_config`) | -- | S-A2, S-A25 |
| `log10_lambda_dsn` | space | `--lambda-dsn` $= 10^{\log_{10}\lambda_{\rm dsn}}$ when `dsn_on` $= 1$ | `0.1` | `LAMBDA_DSN` (`:56`, `:111`) | S-A2, S-A25 |
| `loss_type` | space | `--loss-type` $\to$ `DSNLossConfig.loss_type` | `joint_sep` | none | S-A2, S-A25 |
| `mining_strategy` | space | `--mining-strategy` | `easy_pos_semihard_neg` | none | S-A2, S-A25 |
| `margin` | space | `--margin` $\to$ $m_{\cos}$ | `0.2` | none | S-A2, S-A25 |
| `angular_alpha_deg` | space | `--angular-alpha-deg` $\to$ $\alpha$ | `18.0` | none | S-A2, S-A25 |
| `lambda_sep` | space | `--lambda-sep` $\to$ $\lambda_{\rm sep}$ | `0.1` | none | S-A2, S-A25 |
| `strict_semihard` | `spec.fixed` (1); `--strict-semihard` | projected, then `DSNLossConfig.strict_semihard` | `1` | none | fixed |
| `sep_warmup_frac` | `spec.fixed` (0.0); `--sep-warmup-frac` | $\tau_{\rm sep}$ (not passed by the tuner, F-w) | `0.0` | none | fixed |
| `swap` | `DSNLossConfig` default only | `swap=True` | no flag | none | never |
| `use_angular`, `reduce_nonzero`, `min_per_class` | `build_loss_and_miner` / `CompositeDSNLoss` defaults | `True`, `True`, `2` | no flag | none | never |
| distance family | `build_loss_and_miner` | `CosineSimilarity()` for loss and miner | no flag | none | never |
| `n_classes` ($C$) | runner, from the simulated bank's `cls` column (`:402`) | `CompositeDSNLoss(n_classes=C)` | -- | none | never (F-t) |
| `--b-met` ($B_{\rm met}$) | runner | the metric batch size (`:506`) | `32` | none (F-l) | never |
| `--encoder-steps` | runner | the pre-training length and ramp horizon of `A0`/`A0s` (`:408`, `:458`) | `200` | `ENCODER_STEPS` (`:59`, `:114`) | never |

Three consequences of the table. A **Stage 3** arm trains the loss at the
runner's defaults for every knob except $\lambda_{\rm dsn}$ and the
pre-training length, because only those two are job variables: the arms'
`joint_sep` / `easy_pos_semihard_neg` / $0.2$ / $18^\circ$ / $0.1$ / strict
filter on are not choices the job script exposes. A **Stage 4** trial
receives the five loss flags and `--strict-semihard` from the space and the
weight from the switch; `sep_warmup_frac` reaches it only because the runner's
default equals the space's fixed value (F-w). And the term is **built
whenever the arm has a metric domain** (`dsn_domain` not `None`:
`A0`, `A0s`, `A2`, `A2s`, `A3`) but **used in the joint loop only when
$\lambda_{\rm dsn} > 0$** (`run_joint_arms.py:529-530`,
`joint_train.py:170-176`); for `A0`/`A0s` it trains the encoder alone first
(`train_encoder_only`, `:150-168`) and never enters the joint loop
(`lambda_dsn` stays 0 in `arm_config`).

### 3.2 The loss as a function of the knobs

This section establishes, from the code, what $\ell_{\rm DSN}$ computes for
one metric batch, in the order the data flow: batch, distances, miner,
filter, hinges, reduction, separation term, schedule, composite, and its
place in the objective. Source tags `[REPO]` for `hpc/joint` and `hpc/dsn`,
`[CODE pml 1.6.3]` for the library, `[RAN]` for the constants (recomputed by `tools/p2_numbers.py`).

**The batch.** `ThreeStreamBatcher.met_batch` (`joint_batches.py:176-188`)
draws, from each class present in the metric source, $\lfloor B_{\rm met}/C
\rfloor$ rows with replacement (`torch.randint`), so

$$
|X^{\rm real}_{\rm met}| \;=\; C \,\Big\lfloor \frac{B_{\rm met}}{C} \Big\rfloor
\qquad (= 32 \text{ at } C = 2,\ 30 \text{ at } C = 3,\ B_{\rm met} = 32),
\tag{P2.1}
$$

each class with $n_c = \lfloor B_{\rm met}/C \rfloor$ rows (`[RAN]`). The
batch is class-balanced by construction, which is what makes every class
present in every batch and $K = C$ the normal case of eq. (P2.8). The
encoder-only pre-training of `A0`/`A0s` does **not** use this batcher: it
draws $B_{\rm met}$ rows uniformly at random from the whole source
(`train_encoder_only`, `:158`), so there the class counts fluctuate and a
batch can lack a class.

**Distances and the margin unit.** The embeddings are unit vectors (P1:
the head's L2 normalisation, and `CosineSimilarity` normalises again
`[CODE pml 1.6.3]` `distances/cosine_similarity.py`), so for any two rows

$$
d_{\cos}(z_i, z_{i'}) = 1 - z_i^\top z_{i'}, \qquad
Q(i, i') = \lVert z_i - z_{i'} \rVert_2^2 = 2\, d_{\cos}(z_i, z_{i'}), \qquad
m_{\rm sq} = 2\, m_{\cos},
\tag{P2.2}
$$

the identity holding exactly on $S^{E-1}$ (`[RAN]`: residual $10^{-15}$).
The **miner** scores in cosine distance with the margin $m_{\cos}$
(`train.py:317`); the composite **loss** scores in squared Euclidean with
$m_{\rm sq}$ (`train.py:267`, `dsn_joint_loss.py:97-106`). The conversion is
done once, in `build_loss_and_miner`, and the adapter inherits it rather than
redoing it (`dsn_loss_adapter.py:11-15`). A reader who sees `margin=0.4`
inside a `JointTripletLoss` and `margin=0.2` inside the miner is looking at
one number in two units.

**The miner.** Write $\mathcal{P}_i = \{i' \ne i : y_{i'} = y_i\}$ and
$\mathcal{O}_i = \{i'' : y_{i''} \ne y_i\}$. The three strategies return
`[CODE pml 1.6.3]`:

$$
\begin{aligned}
\text{hard}: \quad & \mathcal{T}_{\rm mined} = \{(i, i', i'') : i' \in \mathcal{P}_i,\ i'' \in \mathcal{O}_i,\ Q(i, i'') \le Q(i, i')\}, \\
\text{easy\_positive}: \quad & \mathcal{T}_{\rm mined} = \{(i, i'(i), i''(i))\},\ \ i'(i) = \mathrm{argmin}_{i' \in \mathcal{P}_i} Q(i, i'),\ \ i''(i) = \mathrm{argmin}_{i'' \in \mathcal{O}_i} Q(i, i''), \\
\text{easy\_pos\_semihard\_neg}: \quad & \mathcal{T}_{\rm mined} = \{(i, i'(i), i''(i))\},\ \ i''(i) = \mathrm{argmin}\{Q(i, i'') : i'' \in \mathcal{O}_i,\ Q(i, i'') > Q(i, i'(i))\},
\end{aligned}
\tag{P2.3}
$$

one triplet per anchor for the two easy-positive strategies (anchors with
no admissible negative are dropped), and every violating triplet for
`hard`. In words: `hard` keeps **all** triplets whose negative is at least as
close to the anchor as the positive (`TripletMarginMiner`,
`type_of_triplets="hard"`: `triplet_margin <= 0`, which implies the
`<= margin` condition since $m_{\cos} > 0$); `easy_positive` pairs each
anchor with its **closest** same-class row and its **closest** other-class
row (`BatchEasyHardMiner(pos="easy", neg="hard")`); `easy_pos_semihard_neg`
pairs each anchor with its closest same-class row and the closest other-class
row that is **strictly farther than that positive**
(`BatchEasyHardMiner(pos="easy", neg="semihard")`: the mask
`mat >= positive_dists` excludes every negative at least as similar as the
easy positive, then the maximum similarity is taken). The library's
"semihard" here has **no margin bound**: the runner's comment that the
negatives are "still inside the margin" (`train.py:332-333`) describes the
strict filter below, not the miner. The miner is unaffected by `loss_type`
(`train.py:224-228`); the 4-tuple the easy-positive miners return is joined
on the anchor by `lmu.convert_to_triplets` (`utils/loss_and_miner_utils.py`),
the same rule `TripletMarginLoss` applies internally. At $B_{\rm met} = 32$
and $C = 2$ the `hard` pool is at most $32 \cdot 15 \cdot 16 = 7680$ triplets
and the easy-positive strategies return at most 32 (`[RAN]`, eq. (P2.1)).

**The hinges.** For a triplet $(i, i', i'')$ `JointTripletLoss.forward`
(`dsn_joint_loss.py:291-328`) computes

$$
\ell_{\rm trip}(i, i', i'') = \big[\, Q(i, i') - \min\big(Q(i, i''),\, Q(i', i'')\big) + m_{\rm sq} \,\big]_+
\qquad (\text{swap} = \text{True}),
\tag{P2.4}
$$

$$
\ell_{\rm ang}(i, i', i'') = \big[\, Q(i, i') - 4\tan^2\!\alpha \;\, Q_{\rm mid}(i, i', i'') \,\big]_+ ,
\qquad Q_{\rm mid}(i, i', i'') = \lVert z_{i''} - \frac{1}{2}(z_i + z_{i'}) \rVert_2^2 ,
\tag{P2.5}
$$

with $4\tan^2\!\alpha = 0.00488$, $0.4223$, $0.5299$ at $\alpha = 2^\circ$,
$18^\circ$, $20^\circ$ (`[RAN]`). The angular hinge is symmetric in $i$ and
$i'$ by construction (it uses $Q(i, i')$ and the midpoint), which is why the
DSN offers no "switching" counterpart for it, and the margin hinge gets its
symmetry from the swap (`:246-249`). Apollonius' identity
$Q_{\rm mid} = (2 Q(i, i'') + 2 Q(i', i'') - Q(i, i'))/4$ holds for arbitrary
vectors (`[RAN]`: residual $10^{-15}$; the DSN's own smoke test checks the
same), so the angular hinge is a constraint on the three side lengths of the
triplet's triangle, not on anything outside it.

**The strict filter and the reduction.** With `strict_semihard=True`

$$
\mathcal{T}_{\rm strict} = \big\{ (i, i', i'') \in \mathcal{T}_{\rm mined} :
Q(i, i') < Q(i, i'') < Q(i, i') + m_{\rm sq} \ \text{ and } \
Q(i, i') < Q(i', i'') < Q(i, i') + m_{\rm sq} \big\},
\tag{P2.6a}
$$

and $\mathcal{T}_{\rm strict} = \mathcal{T}_{\rm mined}$ with the filter off
(`:309-313`); the filter is applied as a 0/1 weight, not by indexing
(`:315-318`). The batch value is the mean over the **active** strict triplets,

$$
\mathcal{L}_{\rm joint} = \frac{1}{\max(1, n_{\rm act})}
\sum_{(i, i', i'') \in \mathcal{T}_{\rm strict}} \big[ \ell_{\rm trip}(i, i', i'') + \ell_{\rm ang}(i, i', i'') \big],
\qquad
n_{\rm act} = \big|\{ (i, i', i'') \in \mathcal{T}_{\rm strict} : \ell_{\rm trip} + \ell_{\rm ang} > 0 \}\big|,
\tag{P2.6}
$$

(`reduce_nonzero=True`, `:320-323`), which matches the library's
`AvgNonZeroReducer` (mean over strictly positive losses, zero when none:
`reducers/threshold_reducer.py`). Two consequences `[reasoning]`, both
checkable against (P2.4)-(P2.6a): under the strict filter every kept triplet
is a margin violator, since $\min(Q(i,i''), Q(i',i'')) < Q(i,i') + m_{\rm sq}$
gives $\ell_{\rm trip} > 0$, so $n_{\rm act} = n_{\rm strict}$ and
`reduce_nonzero` has no effect there; and with `hard` mining the strict set
is empty by construction, since `hard` requires $Q(i, i'') \le Q(i, i')$ and
the filter requires the opposite (the DSN measured 16814 mined, 0 surviving,
`condition_space.py:37-44` `[REPO]`) -- which is why the projection $\Pi$
turns the filter off under `hard` (S3.3.3). A batch with $n_{\rm act} = 0$
contributes exactly 0 and raises nothing.

**The plain triplet loss (`loss_type = triplet`).** The library's
`TripletMarginLoss(margin=m_cos, swap=True, distance=CosineSimilarity(),
reducer=AvgNonZeroReducer())` (`train.py:258-264`) scores the same mined
triplets in cosine units,

$$
\ell_{\rm DSN} = \mathrm{mean}\big\{\, \ell_{\rm pml} : \ell_{\rm pml} > 0 \,\big\}, \qquad
\ell_{\rm pml}(i, i', i'') = \big[\, d_{\cos}(z_i, z_{i'}) - \min\big(d_{\cos}(z_i, z_{i''}),\, d_{\cos}(z_{i'}, z_{i''})\big) + m_{\cos} \,\big]_+ ,
\tag{P2.7}
$$

over $(i, i', i'') \in \mathcal{T}_{\rm mined}$: no angular term, no strict
filter, and by (P2.2) each term is exactly half of (P2.4) on the same triplet
(the DSN's smoke test asserts "margin term == 2 x PML TripletMarginLoss"
`[REPO]` `smoke_test_dsn_joint_loss.py:150-153`). The sign convention is the
library's: `CosineSimilarity` is an inverted distance, and
`BaseDistance.margin(x, y)` returns `y - x` for inverted distances
(`distances/base_distance.py`), so the hinge reads $d_{\cos}$ of the positive
minus $d_{\cos}$ of the negative plus the margin, as written.

**The separation term (`loss_type = joint_sep`).** `CentroidSeparationLoss`
(`:340-458`) forms, from the rows of the batch whose label is in
$\{0, \dots, C-1\}$ (`class_onehot`, `:176-184`: any other label contributes
to no class), the per-class means and their unit directions,

$$
\bar z^{(B)}_c = \frac{1}{n_c} \sum_{i : y_i = c} z_i, \qquad
v_c = \frac{\bar z^{(B)}_c}{\lVert \bar z^{(B)}_c \rVert_2}, \qquad
K = \big|\{ c : n_c \ge 2 \}\big|, \qquad
\rho_{\rm ETF} = -\frac{1}{K - 1},
\tag{P2.8a}
$$

$$
\mathcal{L}_{\rm sep} = \frac{1}{K (K - 1)} \sum_{c \ne c' : n_c \ge 2, n_{c'} \ge 2}
\big( v_c^\top v_{c'} - \rho_{\rm ETF} \big)^2 \quad \text{if } K \ge 2, \qquad
\mathcal{L}_{\rm sep} = 0 \quad \text{if } K < 2,
\tag{P2.8}
$$

the sum over ordered pairs of distinct valid classes (`:436-444`;
`min_per_class=2`). The target is a geometric constant of $K$, never a
hyper-parameter: $-1$ (antipodal) at $K = 2$, $-0.5$ ($120^\circ$) at
$K = 3$, $-1/3$ at $K = 4$ (`[RAN]`). The class means are **raw**, never
centred: the DSN removed the centred form because centring then normalising
is invariant to scale and so scores a collapsed cap of class means as
perfect (`:355-388` `[REPO]`, with the DSN's own measurement quoted there).
For unit vectors whose pairwise inner products all equal
$\rho_{\rm ETF}$, $\lVert \sum_c v_c \rVert_2^2 = K + K(K-1)\rho_{\rm ETF} = 0$
`[reasoning]`, so equiangularity at the target and "the class directions
balance about the origin" are one condition, not two.

**The schedule.** `SepWarmup` (`:598-672`) multiplies $\lambda_{\rm sep}$ by
a ramp evaluated at the number of steps completed, in the fraction

$$
\gamma_{\rm sep}(t) = \min\Big(1, \frac{t}{\tau_{\rm sep}}\Big) \ \ (\tau_{\rm sep} > 0), \qquad
\gamma_{\rm sep}(t) \equiv 1 \ \ (\tau_{\rm sep} = 0), \qquad
t = \frac{n_{\rm done}}{n_{\rm plan}}, \qquad
\lambda_{\rm sep}(t) = \lambda_{\rm sep}\, \gamma_{\rm sep}(t),
\tag{P2.9}
$$

with the first batch at $n_{\rm done} = 0$ (weight exactly 0 under a ramp)
and full weight from $n_{\rm done} = \tau_{\rm sep} n_{\rm plan}$ on
(`sep_warmup_scale`, `:556-595`). **In the joint space $\tau_{\rm sep} = 0$
is fixed** (`joint_space.py:322-324`), so $\gamma_{\rm sep} \equiv 1$ and
$\lambda_{\rm sep}(t) = \lambda_{\rm sep}$ at every step: the schedule
machinery is built and logged (`history[].sep_warmup` = (step, scale),
`joint_train.py:250-252`) but inert. The counter advances once per
**evaluation** of the loss, not per optimiser step; the per-epoch
$\rho_{\rm grad}$ probe evaluates it once more and rewinds the counter
(`dsn_loss_adapter.py:113-147`, `joint_train.py:240-245`), so
$n_{\rm done}$ equals the optimiser step count in the joint loop. The
horizon is $n_{\rm plan} = n_{\rm ep} n_{\rm step}$ for the joint arms (250
at the runner's defaults) and `--encoder-steps` for `A0`/`A0s` (200), the
latter a correction recorded in the runner (`:403-409`).

**The composite.** `build_loss_and_miner` dispatches (`train.py:257-313`):

$$
\ell_{\rm DSN} =
\begin{cases}
\text{eq. (P2.7)} & \text{loss\_type} = \text{triplet}, \\
\mathcal{L}_{\rm joint} & \text{loss\_type} = \text{joint}, \\
\mathcal{L}_{\rm joint} + \lambda_{\rm sep}(t)\, \mathcal{L}_{\rm sep} & \text{loss\_type} = \text{joint\_sep},
\end{cases}
\tag{P2.10}
$$

(`CompositeDSNLoss.forward`, `:723-727`). Under `joint` and `joint_sep` the
margin is **read** (eq. (P2.4), (P2.6a)) even though it is not searched
there (S3.3), and under `triplet` the angular hinge, the filter and the
separation term do not exist. $\mathcal{L}_{\rm sep}$ is evaluated on every
batch and multiplied by the ramp rather than skipped, which costs one
$C \times C$ Gram matrix per batch.

**In the objective.** Plan eq. (2)'s second term is
$\lambda_{\rm dsn}\, \ell_{\rm DSN}\big(h_\psi(X^{\rm real}_{\rm met}),
y^{\rm real}_{\rm met}\big)$, added to the NPE batch mean before one backward
pass (`joint_train.py:170-176`), with

$$
\lambda_{\rm dsn} = 10^{\log_{10}\lambda_{\rm dsn}} \ \ \text{when dsn\_on} = 1, \qquad
\lambda_{\rm dsn} = 0 \ \ \text{when dsn\_on} = 0,
\tag{P2.11}
$$

(`npe_tune_joint.py:213-217`); the weight is applied to the batch value, so
the per-epoch `history[].dsn` records the **unweighted** mean of
$\ell_{\rm DSN}$ over the epoch's steps (`:176`, `:218`). The term reaches
$\psi$ only: nothing in (P2.4)-(P2.10) depends on $\omega$ (`[KB]` deck B.3,
test J8). Its expectation over the metric stream's distribution is
$\mathcal{L}^{\rm real}_{\rm DSN}$ of plan eq. (1a), never computed; what
trains is the batch value. For `A0`/`A0s` the same $\ell_{\rm DSN}$ is the
**whole** objective of the pre-training, minimised with its own AdamW on
torch's defaults for `--encoder-steps` steps (`:150-168`; F-q), after which
$\psi$ is frozen and the joint loop sees $\lambda_{\rm dsn} = 0$.

**What the constants assert about the geometry.** Two readings the DSN's
own documents attach to $m_{\cos}$ and $\alpha$, stated here with their
hypotheses. (a) If every triplet of a cloud satisfied the angular hinge with
equality in the isosceles configuration $Q(i, i'') = Q(i', i'')$, the
within/between ratio would be $Q(i, i')/Q(i, i'') = 4\sin^2\!\alpha$
(`[RAN]`: the hinge is zero exactly there, as in the DSN's own check
`smoke_test_dsn_joint_loss.py:176-206`), and the DSN reads this as a floor on
the cosine silhouette,

$$
S_{\rm sil} \;\ge\; 1 - 4\sin^2\!\alpha \qquad (0.995 \text{ at } 2^\circ,\ 0.618 \text{ at } 18^\circ,\ 0.532 \text{ at } 20^\circ,\ 0 \text{ at } 30^\circ) \quad \text{[RAN]},
\tag{P2.12}
$$

(`config.py:650-659`, `preflight_config.py:163-175` `[REPO]`). The floor
is a statement about one isosceles triplet, transplanted to a cloud's
silhouette -- a per-point average of mean distances -- without a proof in
the code or its documents; it is a useful ruler for the *direction* of
$\alpha$ (smaller $\alpha$, tighter within-class geometry; above $30^\circ$
the hinge asks nothing of unit vectors), not a theorem about $S_{\rm sil}$.
(b) The preflight's margin rule, $S_{\rm sil} \ge m_{\cos}(C-1)/C$
(`preflight_config.py:140-141` `[REPO]`; its derivation is not in the tree
and was not reconstructed here), gives $0.1$ to $0.5$ over the searched
$[0.1, 1.0]$ at $C = 2$ `[RAN]`. Both readings say the same thing in E3's
language: $m_{\cos}$ and $\alpha$ set the width of the channel the
within-class residual $\xi$ must pass through, and the composite loss's
minimum is the collapse of each class to one point on a simplex ETF
(`[KB]` deck C.2-C.3; S3.7). [corrected 2026-10-05, E3 S3.4, S3.6:
$m_{\cos}$ and $\alpha$ constrain the residual's size, which bounds its
information only through a noise or resolution scale, E3 eq. (E3.9); under
the strict filter and the easy-positive miners they constrain only how
close the nearest negative may come; and the expected loss's zero is the
collapse only under `joint_sep`, E3 eq. (E3.3).]

Table P2.1 fixes the constants over the searched ranges and at the pins.

| knob | value | $m_{\rm sq}$ or $4\tan^2\!\alpha$ | reading (P2.12) | source |
|---|---|---|---|---|
| `margin` | 0.1 (range low) | 0.2 | band of 0.1 in $d_{\cos}$; floor 0.05 at $C = 2$ | `[RAN]` |
| `margin` | 0.2 (runner default and pin) | 0.4 | band 0.2; floor 0.10 | `[RAN]` |
| `margin` | 0.3 (DSN `TrainConfig`; F-h) | 0.6 | band 0.3; floor 0.15 | `[RAN]` |
| `margin` | 1.0 (range high) | 2.0 | band 1.0 (half the diameter of $S^{E-1}$ in $d_{\cos}$); floor 0.5 | `[RAN]` |
| `angular_alpha_deg` | 2 (range low) | 0.00488 | floor 0.995: near-total collapse demanded | `[RAN]` |
| `angular_alpha_deg` | 18 (default and pin) | 0.4223 | floor 0.618 | `[RAN]` |
| `angular_alpha_deg` | 20 (range high) | 0.5299 | floor 0.532 | `[RAN]` |
| `angular_alpha_deg` | 30 (outside the range) | 1.333 | floor 0: the hinge is vacuous on unit vectors | `[RAN]` |
| `lambda_sep` | $[10^{-3}, 1]$, pin 0.1 | -- | constant in $t$ (eq. (P2.9), $\tau_{\rm sep} = 0$) | `[REPO]` |
| $K$ | 2 / 3 / 4 | $\rho_{\rm ETF}$ $-1$ / $-0.5$ / $-0.333$ | antipodal / $120^\circ$ / $109.5^\circ$ | `[RAN]` |

### 3.3 The seven searched axes

This section establishes, for each axis, the fields of plan S2.2. The source
tag of every row is `[REPO]` unless written otherwise; the analytic effect is
eq. (P2.1)-(P2.11) instantiated. The activity masks and the projection
(`condition_space.py:120-124`, `:222-239`) are, in this document's symbols,

$$
\mathcal{M}_{\rm trip} = \{ m_{\cos} \}, \qquad
\mathcal{M}_{\rm joint} = \{ \alpha \}, \qquad
\mathcal{M}_{\rm jsep} = \{ \alpha, \lambda_{\rm sep}, \tau_{\rm sep} \},
\tag{P2.13}
$$

and $\Pi$ sets `strict_semihard` to 0 when `loss_type` is `triplet` (the
flag is never read there) or when `mining_strategy` is `hard` (the strict
set would be empty, eq. (P2.6a)), leaving the other two coordinates
unchanged. `canonicalise_config` (`joint_space.py:558-572`) applies $\Pi$
and then pins every axis of $\{m_{\cos}, \alpha, \lambda_{\rm sep}\}$ outside
the mask to `INACTIVE_CANONICAL` (`:161-185`): $0.2$, $18.0$, $0.1$. The
pins are the adapter's defaults, not the DSN `TrainConfig`'s, and the
docstrings that still quote $0.3$ are F-f.

#### 3.3.1 `dsn_on`

- **Where it lives.** Axis 7 of `JOINT_KNOB_ORDER`; a campaign switch
  (`RANGE_PROVENANCE`: "campaign switch (plan S5.2)"); `DERIVED`, reaches
  the runner as `--lambda-dsn 0` or $10^{\log_{10}\lambda_{\rm dsn}}$.
  Pinned at 0 in S-A1 and S-A5 (`_DSN_AXES_OFF`, `:358-366`), free in S-A2
  and S-A25 (`:375-398`).
- **Type and domain.** Categorical $\{0, 1\}$.
- **Defaults.** None of its own: a Stage 3 arm's weight comes from
  `arm_config` (0 for `A1`, `A5`, `A_ref`, `shuffled`; `--lambda-dsn` for
  `A2`, `A2s`, `A3`; irrelevant to `A0`/`A0s`, whose term trains the encoder
  before the loop).
- **Status.** Configured.
- **What it represents, plainly.** Whether the label enters the joint
  training at all. Off, the run is arm `A1`'s objective; on, arm `A2`'s
  (S-A2 nests S-A1 at `dsn_on` = 0, which is the whole design of the
  campaign).
- **What it changes, analytically.** Eq. (P2.11): the second term of plan
  eq. (2) is present or absent. Off, the six other axes of the block are
  pinned by clamp (a) of `canonicalise_config` (`:558-561`), so two trials
  differing only there are one trial.
- **Legality and interactions.** Always active. Off, `boundary_axes` skips
  the block (`:786-790`), so a pinned margin can never trigger the
  range-widening rule. The metric stream is still drawn when the arm has a
  metric domain, but the loss is not evaluated at $\lambda_{\rm dsn} = 0$
  (`joint_train.py:170`), so an `A2` trial with `dsn_on` = 0 costs the batch
  draw and nothing else.
- **Failure modes and the diagnostic.** None of its own. The diagnostic that
  it did what the ledger says is `arm_config.lambda_dsn` in the run record
  (`run_joint_arms.py:599`), the weight the loop applied: 0 when off.
  `config.lambda_dsn` (`vars(args)`, `:598`) is the flag as received, which
  for a Stage 3 `A1` is still the default 0.1 although the arm applies 0.

#### 3.3.2 `log10_lambda_dsn` ($\log_{10}\lambda_{\rm dsn}$)

- **Where it lives.** Axis 8; `DERIVED` to `--lambda-dsn`
  (`npe_tune_joint.py:115`, `:213-217`); runner `--lambda-dsn`
  (`run_joint_arms.py:222`); job variable `LAMBDA_DSN`
  (`joint_arms.pbs:56`); `TrainConfig.lambda_dsn` of the joint loop
  (`joint_train.py:42`, `:50`).
- **Type and domain.** Real, searched in $[-3, 1]$, uniform prior (P0 Table
  A); $\lambda_{\rm dsn} \in [10^{-3}, 10]$.
- **Defaults.** Runner `0.1` ($= 10^{-1}$, inside the range); inactive pin
  $0.0$ (`INACTIVE_CANONICAL`), which is never passed as a weight because
  `dsn_on` = 0 sends `--lambda-dsn 0` -- the pin is a coordinate for
  deduplication, not a value that trains.
- **Range provenance.** "plan S5.1, [-3, 1]" (`RANGE_PROVENANCE`): the plan's
  choice, not a DSN range.
- **Status.** Configured.
- **What it represents, plainly.** How hard the label pulls on the encoder,
  relative to the likelihood: at $10^{-3}$ the metric term is a tie-breaker,
  at $10$ it dominates every step. The scale is relative because
  $\mathcal{L}^{\rm sim}_{\rm NPE}$ is in nats per row and $\ell_{\rm DSN}$
  is dimensionless and of order the margin (eq. (P2.4): a fully violated
  triplet costs about $m_{\rm sq}$), so the same $\lambda_{\rm dsn}$ means
  different things on different banks; that is why it is searched over four
  decades. [corrected 2026-10-05, E4 S3.4: "dominates every step" is the
  weighted-sum reading; under AdamW a weight moves the encoder's update only
  against the two terms' gradient sizes, coordinate by coordinate, and
  saturates at both ends, E4 eq. (E4.6), so whether $10$ dominates depends on
  those sizes, which no record keeps (F-bd); on the flow's weights, for a
  given encoder trajectory, it acts only through the clip's variation from
  step to step, E4 eq. (E4.5).]
- **What it changes, analytically.** Eq. (P2.11) and plan eq. (2): the
  gradient on $\psi$ is the NPE gradient plus $\lambda_{\rm dsn}$ times the
  metric gradient. The sign of their cosine, $\rho_{\rm grad}$, is prediction
  P4 (`[KB]` deck C.6); large $\lambda_{\rm dsn}$ is prediction P3's regime,
  $r_{\rm eff} \to C - 1$.
- **Legality and interactions.** Active iff `dsn_on` = 1. Interacts with
  $\lambda_{\rm rep}$ (P3) in the same sum and with the learning rate (P5)
  multiplicatively: AdamW normalises the update per coordinate, so what
  $\lambda_{\rm dsn}$ changes is the *direction* of the summed gradient more
  than its size `[reasoning]`.
- **Failure modes and the diagnostic.** Too large: `history[].val_npe` rises
  while `history[].dsn` falls, and `delta_hat` approaches the $\log C$
  ceiling with `r_eff` near $C - 1$ (S3.7). Too small: indistinguishable from
  `dsn_on` = 0 in the record except by `arm_config.lambda_dsn`.

#### 3.3.3 `loss_type`

- **Where it lives.** Axis 9; `--loss-type` (`:249-250`);
  `DSNLossConfig.loss_type`; the dispatch of `build_loss_and_miner`
  (`train.py:257-313`); levels from `condition_space.LOSS_TYPES`
  (`:108`, `RANGE_PROVENANCE`).
- **Type and domain.** Categorical, $\{$`triplet`, `joint`, `joint_sep`$\}$.
- **Defaults.** Runner and adapter `joint_sep`; inactive pin `triplet`
  (`INACTIVE_CANONICAL`); DSN `TrainConfig` `triplet` (F-h).
- **Status.** Configured.
- **What it represents, plainly.** Which of three objectives the label
  supervises: the plain margin hinge; the margin hinge plus the angular cone
  on strictly semi-hard triplets; or that plus the push of the class
  directions to a simplex ETF.
- **What it changes, analytically.** Eq. (P2.10). It is the axis that
  decides every other loss axis's activity through (P2.13): under `triplet`
  only $m_{\cos}$ is read; under `joint` $m_{\cos}$ is read but fixed and
  $\alpha$ is searched; under `joint_sep` $\lambda_{\rm sep}$ joins. Under
  `triplet` the strict filter does not exist and `swap`, `use_angular`,
  `reduce_nonzero` are the library's.
- **Legality and interactions.** Always active. $\Pi$ zeroes the filter
  under `triplet`. With `strict_semihard` fixed at 1 in the space, the two
  easy-positive strategies under `joint`/`joint_sep` always run with the
  filter on: the four "filter off" cells of the DSN's 13 legal conditions are
  unreachable from the joint space (F-u).
- **Failure modes and the diagnostic.** `joint`/`joint_sep` with an empty
  strict set every step (S3.7): `history[].dsn` exactly 0.0 and
  `history[].rho_grad` NaN. `joint_sep` with the ETF target built from the
  wrong class count (F-t). A categorical has no boundary (`_NO_BOUNDARY`,
  `joint_space.py:764-765`): the best trial choosing `joint_sep` is the
  answer, not evidence that the range was narrow.

#### 3.3.4 `mining_strategy`

- **Where it lives.** Axis 10; `--mining-strategy` (`:251-252`);
  `DSNLossConfig.mining_strategy`; the miner branch of
  `build_loss_and_miner` (`train.py:315-350`); levels from
  `condition_space.MINING_STRATEGIES` (`:107`).
- **Type and domain.** Categorical, $\{$`hard`, `easy_positive`,
  `easy_pos_semihard_neg`$\}$.
- **Defaults.** Runner and adapter `easy_pos_semihard_neg`; inactive pin
  `hard`; DSN `TrainConfig` `hard` (F-h).
- **Status.** Configured.
- **What it represents, plainly.** Which triplets of a batch the hinges see,
  eq. (P2.3): every violating one (`hard`), or one per anchor built from its
  nearest same-class row and its nearest (`easy_positive`) or nearest-beyond-
  the-positive (`easy_pos_semihard_neg`) other-class row. The DSN's reading
  `[REPO]` `config.py:633-646`: `hard` is the collapse-seeking choice (every
  same-class pair is pulled together), the two easy-positive strategies are
  anti-collapse by design (only the closest same-class window must be near,
  so a class may occupy a manifold). According to PubMed, Kertesz (Sensors
  2022, full text read, DOI 10.3390/s22197579) defines the same three
  families -- random-hard, semi-hard (a negative with non-zero loss but
  within the margin) and hardest (the closest negative) -- attributes
  semi-hard negative sampling to FaceNet, and reports on synthetic clusters
  that hardest-negative sampling struggled in the initial phase of training
  while semi-hard sampling was the most stable of the methods compared
  (qualitative; the paper's numbers are not used here). Chung and Lee (Sci
  Rep 2023, full text read, DOI 10.1038/s41598-023-45467-8), the source of
  the DSN's strict filter, state that hard samples can lead to bad local
  optima and select semi-hard samples only.
- **What it changes, analytically.** The set $\mathcal{T}_{\rm mined}$ of
  (P2.3), hence $n_{\rm mined}$ (thousands under `hard`, at most the batch
  size otherwise), the gradient's support, and through (P2.6a) whether
  anything survives the filter. The library's semi-hard negative carries no
  margin bound (S3.2), so under `easy_pos_semihard_neg` the margin acts only
  through the hinge and the filter, not through the miner.
- **Legality and interactions.** Always active. `hard` forces the filter off
  through $\Pi$; `hard` with a small $\alpha$ is the DSN's recommended
  collapse-forcing pairing, an easy-positive strategy with a small $\alpha$ an
  internally opposed one (`preflight_config.py:176-185`; THEORY S3.7.5
  `[REPO]` doc). The joint stack's default pairs `easy_pos_semihard_neg` with
  $\alpha = 18^\circ$ and the ETF term, i.e. an anti-collapse miner with a
  collapse-seeking separation term; whether that is a tension or a balance
  is for the search to measure, and nothing in the stack resolves it.
- **Failure modes and the diagnostic.** Under `hard` the strict filter is
  off, every mined triplet has $\ell_{\rm trip} \ge m_{\rm sq} > 0$, and
  `history[].dsn` cannot be 0 unless the miner returns nothing; under the
  easy-positive strategies an empty $\mathcal{T}_{\rm strict}$ is silent
  (S3.7). No count of mined or strict triplets reaches the record (F-v).

#### 3.3.5 `margin` ($m_{\cos}$)

- **Where it lives.** Axis 11; `--margin` (`:253`); `DSNLossConfig.margin`;
  `train.py:260` (miner and plain loss, in $d_{\cos}$) and `:267`
  ($m_{\rm sq} = 2 m_{\cos}$ for the composite losses); range from
  `SearchConfig.margin_range` (`config.py:903`).
- **Type and domain.** Real, $m_{\cos} > 0$; searched in $[0.1, 1.0]$,
  uniform prior. Validated positive by `JointTripletLoss` (`:273-275`).
- **Defaults.** Runner and adapter `0.2`; inactive pin `0.2`; DSN
  `TrainConfig` `0.3` (F-h); the DSN JSON searches the same $[0.1, 1.0]$
  (TUNING_1 S3.9).
- **Range provenance.** `DSN config.SearchConfig.margin_range` -- correct.
- **Status.** Configured.
- **What it represents, plainly.** How much farther the negative must be
  than the positive before a triplet stops costing anything; in cosine
  distance, on a sphere whose diameter in that unit is 2.
- **What it changes, analytically.** Eq. (P2.4) and (P2.7) through
  $m_{\rm sq}$ and $m_{\cos}$; the band width of the strict filter (P2.6a);
  the miner's threshold under `hard` (where $\le 0$ makes the margin
  irrelevant to selection) -- so under the joint stack's default miner the
  margin never touches which triplets are mined. The preflight's floor
  reading (P2.12b): $m_{\cos} = 1$ asserts $S_{\rm sil} \ge 0.5$ at $C = 2$.
- **Legality and interactions.** Active (searched) only under `triplet`
  ($\mathcal{M}_{\rm trip}$); **read but fixed at the pin 0.2** under `joint`
  and `joint_sep`, where it still sets the hinge and the filter band. Never
  searched together with $\alpha$: both bind on the within/between ratio and
  the pair traces a ridge (THEORY S3.3.3 `[REPO]` doc, the argument the mask
  encodes).
- **Failure modes and the diagnostic.** Near the top of the range the hinge
  may be unreachable and stays active on every triplet all run
  (`history[].dsn` flat, high); with the filter on, a large $m_{\rm sq}$
  widens the band and admits more triplets, a small one admits fewer and
  can empty it. The diagnostic for an unreachable margin is the un-decreasing
  `history[].dsn`; for an empty band, `dsn` = 0.0 (S3.7).

#### 3.3.6 `angular_alpha_deg` ($\alpha$)

- **Where it lives.** Axis 12; `--angular-alpha-deg` (`:254`);
  `DSNLossConfig.angular_alpha_deg`; `JointTripletLoss(alpha_deg=...)`
  (`train.py:271`, `:301`); `four_tan2` (`:281`); range from
  `SearchConfig.angular_alpha_deg_range` (`config.py:912`).
- **Type and domain.** Real, degrees, $(0, 90)$ validated (`:276-278`);
  searched in $[2, 20]$, uniform prior.
- **Defaults.** Runner, adapter and pin `18.0`; DSN `TrainConfig` `18.0`;
  the DSN JSON searches the same $[2, 20]$ (TUNING_1 S3.10).
- **Range provenance.** `DSN config.SearchConfig.angular_alpha_deg_range`
  -- correct.
- **Status.** Configured.
- **What it represents, plainly.** The half-angle of the cone around the
  anchor-positive pair that the negative must stay out of; **smaller is
  stricter** (eq. (P2.5): the tolerated $Q(i,i')$ shrinks as $\tan^2\!\alpha$
  does), the inverse of the intuition that a larger number is a stronger
  setting. The angular loss itself is Wang et al.'s (ICCV 2017), a computer-
  vision venue PubMed does not index (search reported in S6); its use with a
  strict semi-hard filter is Chung and Lee's (full text read, S6), who chose
  the angle on a validation set and whose stated angle values the retrieved
  text does not carry -- the DSN's remark that their angles are
  $\{30, 45, 60, 75\}$ (`config.py:652`) is therefore `[REPO]`, not verified
  here.
- **What it changes, analytically.** The constant $4\tan^2\!\alpha$ of
  (P2.5): 0.00488 to 0.5299 over the range (Table P2.1), a factor of 109. The
  implied floor (P2.12) goes from 0.995 to 0.532; the uniform prior puts
  equal mass on $[2, 4]$ and $[18, 20]$, which are very different regimes,
  and $\tan^2$ varies fastest at the bottom.
- **Legality and interactions.** Active under `joint` and `joint_sep`
  ($\mathcal{M}_{\rm joint}$, $\mathcal{M}_{\rm jsep}$); pinned at 18.0 under
  `triplet`, where it is not read. Small $\alpha$ with an easy-positive miner
  is the internally opposed pairing of S3.3.4.
- **Failure modes and the diagnostic.** Above $30^\circ$ the hinge is
  vacuous on unit vectors (floor $\le 0$); the range stops at $20^\circ$, so
  the search cannot reach it. At $2^\circ$ the hinge asks for near-total
  collapse: `r_eff` falls toward $C - 1$ and `delta_hat` toward the ceiling
  (S3.7), the signature of prediction P3 reached through $\alpha$ instead of
  $\lambda_{\rm dsn}$.

#### 3.3.7 `lambda_sep` ($\lambda_{\rm sep}$)

- **Where it lives.** Axis 13; `--lambda-sep` (`:255`);
  `DSNLossConfig.lambda_sep`; `CompositeDSNLoss(lambda_sep=...)`
  (`train.py:302`, `dsn_joint_loss.py:710-713`); range from
  `SearchConfig.lambda_sep_range` (`config.py:917`).
- **Type and domain.** Real, $\ge 0$ validated; searched in $[10^{-3}, 1]$,
  **log-uniform** prior (`space_dimensions`, `joint_space.py:465-469`).
- **Defaults.** Runner, adapter and pin `0.1`; DSN `TrainConfig` `0.1`. The
  DSN JSON searches $[10^{-2}, 20]$ (TUNING_1 S3.11, "three and a third
  decades", with 20 meaning "the separation term dominates"); the joint space
  copies the dataclass $(10^{-3}, 1.0)$ (F-g), whose top is twenty times
  lower.
- **Range provenance.** `DSN config.SearchConfig.lambda_sep_range` --
  correct as a dataclass reference; the configured range differs (F-g).
- **Status.** Configured.
- **What it represents, plainly.** The weight of the push that drives the
  class directions apart, relative to the two hinges; constant through
  training in the joint space (eq. (P2.9) at $\tau_{\rm sep} = 0$).
- **What it changes, analytically.** Eq. (P2.10), the coefficient of
  $\mathcal{L}_{\rm sep}$. $\mathcal{L}_{\rm sep}$ is of order 1 for a
  collapsed cap of means at $K = 2$ (two unit directions with cosine near
  $+1$ against a target of $-1$ score $(2)^2 = 4$) and 0 at the ETF, so at
  $\lambda_{\rm sep} = 1$ the term is comparable to a violated hinge and at
  $10^{-3}$ negligible `[reasoning]`. The DSN `TrainConfig` comment that the
  term's scale differs "either side of $C = 3$" (`config.py:673-677`)
  describes the removed centred form and is stale with respect to
  `dsn_joint_loss.py` (S5).
- **Legality and interactions.** Active only under `joint_sep`
  ($\mathcal{M}_{\rm jsep}$); pinned at 0.1 otherwise. Its partner
  $\tau_{\rm sep}$, searched by the standalone DSN as the 18th axis (TUNING_1
  S3.17), is fixed at 0 here, so the dose and the terminal weight of the
  standalone argument coincide: there is one degree of freedom, not two.
  Interacts with the class count $K$ of the batch, which sets the target.
- **Failure modes and the diagnostic.** The term can be minimised by moving
  the class **means** without touching within-class spread (it reads $v_c$
  only), so a run can score $\mathcal{L}_{\rm sep} \approx 0$ with poor
  hinges; and it is zero whenever a batch holds fewer than two valid classes,
  which the class-balanced batcher prevents in the joint loop but not in the
  pre-training of `A0`/`A0s` (S3.2). `sep_mean_cos` and `sep_n_classes` are
  exposed by the loss and not recorded (F-v).

### 3.4 The fixed knobs

This section establishes the knobs of the term that the search does not
move and which surface fixes them. *Fixed by plan S5.1* means declared in
`spec.fixed`; *fixed by default* means a code default no flag reaches.

| knob | fixed by | value | read by | what it does |
|---|---|---|---|---|
| `strict_semihard` | plan S5.1 (`spec.fixed`), passed as `--strict-semihard` and projected | 1 (then $\Pi$) | `JointTripletLoss` (`:309-313`) | eq. (P2.6a); inert under `triplet`, forced 0 under `hard`. The DSN searches it (TUNING_1 S3.14); the joint space does not, and `default_joint_space`'s docstring records that S5.1 does not state the value (`:287-291`). Consequence F-u. |
| `sep_warmup_frac` ($\tau_{\rm sep}$) | plan S5.1 (`spec.fixed`); not passed by `build_argv` (F-w) | 0.0 | `SepWarmup` | eq. (P2.9): no ramp; the DSN searches it under a derived cap, the patience divided by the epoch ceiling (TUNING_1 S3.17, `condition_space.sep_warmup_frac_cap`); the joint space's replicate ramp `warmup_frac_rep` reuses its range (P3) |
| `swap` | `DSNLossConfig` default | `True` | both losses | the $\min$ in (P2.4) and (P2.7); the DSN `TrainConfig` default is also `True` |
| `use_angular` | `build_loss_and_miner` | `True` | `JointTripletLoss` | the angular hinge is always on under `joint`/`joint_sep` |
| `reduce_nonzero` | `build_loss_and_miner` | `True` | `JointTripletLoss` | the denominator of (P2.6); no effect under the strict filter (S3.2) |
| `min_per_class` | `CompositeDSNLoss` default | 2 | `CentroidSeparationLoss` | the validity threshold of $K$ in (P2.8a) |
| distance family | `build_loss_and_miner` | `CosineSimilarity()` | miner and plain loss | the miner's default would be `LpDistance`; passing cosine to both is what keeps mining and scoring in one geometry (`train.py:86-97`) |
| `n_classes` ($C$) | runner (`:402`) | `len(unique(sim["cls"]))` | `CentroidSeparationLoss`, `class_onehot` | the ETF target's $K \le C$ and the label mask; taken from the **simulated** bank even when the metric stream is real (F-t) |
| `--b-met` ($B_{\rm met}$) | runner default | 32 | `BatchSpec` | eq. (P2.1); not a job variable (F-l) |
| `--encoder-steps` | runner default; job variable `ENCODER_STEPS` | 200 | `train_encoder_only`, the ramp horizon of `A0`/`A0s` | the pre-training length; its optimiser is AdamW on torch defaults (F-q) |
| `sep_centre_means`, `sep_gate_*` | `DSNLossConfig` (`None`) | inert | `build_loss_and_miner` warns if set | the retired centred form and the retired latching gate; never set by the joint stack |

### 3.5 The differences table (D-036)

This section establishes, for each inherited axis and for the construction
around it, the joint stack's value against the standalone DSN's three
surfaces, so that a reader of TUNING_1 can translate. Columns as P1 S3.5;
the last three rows are not axes but the things that differ *around* the
axes, which matter more than any range.

| axis / construction | joint range or value | joint runner default | DSN configured range (TUNING_1) | `SearchConfig` default | `TrainConfig` default | TUNING section |
|---|---|---|---|---|---|---|
| `loss_type` | $\{$`triplet`, `joint`, `joint_sep`$\}$ | `joint_sep` | same three | same | `triplet` | 1: S3.13 |
| `mining_strategy` | three levels | `easy_pos_semihard_neg` | same three | same | `hard` | 1: S3.12 |
| `margin` | $[0.1, 1.0]$ | 0.2 | $[0.1, 1.0]$ | $(0.1, 1.0)$ | 0.3 | 1: S3.9; 2: S3.3 |
| `angular_alpha_deg` | $[2, 20]$ | 18.0 | $[2, 20]$ | $(2.0, 20.0)$ | 18.0 | 1: S3.10 |
| `lambda_sep` | $[10^{-3}, 1]$, log-uniform | 0.1 | $[10^{-2}, 20]$, log-uniform | $(10^{-3}, 1.0)$ | 0.1 | 1: S3.11; 2: S3.3 |
| `strict_semihard` | fixed 1, projected | 1 | searched $\{0, 1\}$ | -- | `False` | 1: S3.14 |
| `sep_warmup_frac` | fixed 0.0 | 0.0 | $[0, 0.5]$ requested, capped at the derived `sep_warmup_frac_cap` | $(0.0, 0.5)$ | 0.0 | 1: S3.17 |
| `lambda_dsn` | $[10^{-3}, 10]$ behind `dsn_on` | 0.1 | does not exist (the DSN loss is the whole objective) | -- | -- | -- |
| inactive-axis pin | adapter defaults (0.2, 18, 0.1) | -- | the base JSON's values (0.3, 18, 0.1) | -- | -- | 2: S3.3 |
| batch construction | `ThreeStreamBatcher.met_batch`: $\lfloor B_{\rm met}/C\rfloor$ rows per class with replacement; no surrogates; no cross-culture positive rule | $B_{\rm met} = 32$ | `ConditionBalancedBatchSampler` + `TripletCollator` (`config.py:726-729`, `:488`; the sampler and collator themselves not read): `windows_per_condition` per class, augmentation surrogates with labels $\ge 10^6$ masked out of class statistics, `positives_mode` (`augmentation` or `cross_culture`) | -- | -- | 2: S3.5 |
| selection | the NPE validation loss (`val_npe`), never the metric term | -- | ARI and silhouette on the validation split (`train.py:99-114`) | -- | -- | 2: S3.4 |
| horizon of the ramp | $n_{\rm ep} n_{\rm step}$ (joint arms); `--encoder-steps` (`A0`/`A0s`) | 250; 200 | the planned budget (max epochs times batches per epoch), with early stopping at the configured patience | -- | -- | 1: S3.17 |

Two of these rows are findings. **F-h** (row "inactive-axis pin" and the
`TrainConfig` column): the base loss the joint stack builds at its defaults
-- `joint_sep`, `easy_pos_semihard_neg`, $m_{\cos} = 0.2$, filter on -- is
not the standalone DSN's base (`triplet`, `hard`, 0.3, filter off); a
TUNING_1 statement about "the base configuration" does not transfer. The
**batch-construction** row is the difference D-036 exists for: the standalone
DSN's loss sees augmented surrogate rows and, under `cross_culture`, refuses
same-culture positives, so its within-class geometry is shaped by a
nuisance-invariance device the joint stack does not have; the joint stack's
metric batch is a plain class-balanced draw. A result from the 52-cell
screening or the 300-trial search of TUNING_1 is therefore evidence about a
different estimator of $\mathcal{L}^{\rm real}_{\rm DSN}$, not about this
one.

### 3.6 Interactions with the other blocks

This section establishes what the loss axes change in the blocks the other
P documents own.

- **With the encoder (P1).** Every quantity of S3.2 is a function of
  $z = h_\psi(x)$ and nothing else; the term reaches $\psi$ only. The
  embedding dimension $E$ enters only through the geometry of $S^{E-1}$: at
  $E = 8$ to $16$ a simplex ETF of $K \le 4$ classes always fits
  ($K - 1 \le E$). The GroupNorm backbone makes the metric batch's rows
  independent of the simulated batch's in the forward pass (`[KB]` deck B.3),
  so the two terms meet only in the gradient on $\psi$.
- **With the NPE term and the flow (P4).** The sum of plan eq. (2) is where
  the two objectives compete; $\rho_{\rm grad}$ (`gradient_cosine`,
  `joint_train.py:77-101`, one probe batch per epoch, NaN when either
  gradient is zero) is the per-epoch measurement of that competition, and
  P4 predicts it goes negative on a non-trivial fraction of `A2` steps
  (`[KB]` deck C.6). The flow $q_\omega$ is untouched by the term, but what
  it conditions on is: at the collapse point $z$ is a relabelling of $c$ and
  $q_\omega(\theta \mid z)$ can distinguish at most $C$ things (E3).
  [2026-10-05, E3 eqs. (E3.6)-(E3.7): a relabelling of $\hat c$, the class
  read off $z$, which is $c$ only where the code makes no error; and at
  most $C$ things only where the collapse is exact on the rows the flow is
  trained and scored on, which for `A0` are simulated windows the metric
  term never sees.]
- **With the replicate term (P3).** Both terms act on $\psi$ through real
  rows; the replicate term never sees a label and the metric term never sees
  a pair. S-A25 is the only campaign with both free.
- **With the optimiser (P5).** $\lambda_{\rm dsn}$ scales a gradient AdamW
  then normalises per coordinate; the learning rate and $\gamma_{\rm wd}$ of
  P5 apply to the joint loop's optimiser only. The pre-training optimiser of
  `A0`/`A0s` is a second AdamW at `--lr` and torch's own `weight_decay` and
  `betas` (F-q, owned with P5): an `A0` run and an `A2` run at the same flags
  train the metric term under two different optimiser settings.
- **With the search mechanics (P6).** The block is the one with a
  three-valued categorical pair (`loss_type`, `mining_strategy`), a
  projection and a mask; `canonicalise_config` is where they act, and
  `config_key` is what they protect. The control recipe copies every loss
  axis unchanged and permutes only the $(\theta, z)$ pairing of the
  simulated stream (`control_config_from`, `:697-722`): a control trains the
  **same** metric term on the **same** real labels, so its floor measures
  the NPE term's "learned nothing" level under an unchanged label pull.
- **With the bench and the bank (P7).** On the bench both arms carry the
  generator's `cls`, so $C$ from the simulated bank equals the metric
  stream's class count whatever the arm (F-t is harmless there); on a cohort
  bank the metric stream's labels are the cohort's and the simulated `cls`
  is whatever the bank's provider wrote, which nothing checks.

### 3.7 Failure modes and the diagnostics that reveal them

This section establishes what breaks under misapplication and which number
in the run record shows it. The record is `<arm>_seed<k>.json`
(`run_joint_arms.py:578-612`): `history[]` per epoch with `dsn`, `val_npe`,
`rho_grad`, `sep_warmup`; `delta_hat`, `r_eff`, `cluster_ari`,
`cluster_silhouette` (on the **simulated** report split against the
simulated labels, `:607-608`), `config`.

| failure | mechanism | first visible number | source |
|---|---|---|---|
| dead metric term | $\mathcal{T}_{\rm strict}$ empty every step: no admissible semi-hard negative, or a band too narrow for the batch's geometry (eq. (P2.6a)); the loss returns 0 and raises nothing | `history[].dsn` exactly `0.0` at every epoch and `history[].rho_grad` `NaN` (zero DSN gradient); `n_strict` would say it directly but is not logged (F-v) | `[REPO]` `dsn_joint_loss.py:320-323`; `joint_train.py:96-99` |
| collapse | the minimum of (P2.10): within-class residual $\xi \to 0$, class directions at the ETF; reached fastest at large $\lambda_{\rm dsn}$, small $\alpha$, `hard` mining [2026-10-05, E3: the zero of the expected loss under `joint_sep` only, E3 eq. (E3.3); `r_eff` $\to C - 1$ is also met at $C = 2$ by a constant encoder or a thin line (F-bb), so read it with the cluster scores; `delta_hat` at or below $\log C$ needs an exact code on the scored rows and E2's first case, E3 eq. (E3.6)] | `r_eff` $\to C - 1$ (prediction P3: 1 at $C = 2$), `cluster_silhouette` $\to 1$, `delta_hat` at or below $\log C$ (0.693 nats at $C = 2$) | `[KB]` deck C.2-C.3, C.6 |
| the label ceiling mistaken for a good fit | a high `cluster_ari` and silhouette with a `delta_hat` that cannot exceed $\log C$: the encoder did what it was asked | `delta_hat` against $\log C$; per-axis `contraction` flat across the free axes | `[KB]` deck C.2 (P8, P9) |
| unreachable margin | $m_{\cos}$ near 1 asserts a geometry the cloud cannot reach; the hinge is active on every triplet all run | `history[].dsn` high and flat, staying at the margin's scale instead of falling toward 0 | `[REPO]` TUNING_1 S3.9 (the standalone's reading) |
| vacuous angle | $\alpha \ge 30^\circ$: floor $\le 0$, the hinge contributes nothing | not reachable: the range ends at $20^\circ$ | `[RAN]` Table P2.1 |
| opposed pairing | an easy-positive miner with a small $\alpha$ or a large $\lambda_{\rm sep}$: the miner resists what the hinge and the ETF term demand | no single number; read the winning condition beside the score (THEORY S3.7.5's advice) | `[REPO]` `preflight_config.py:176-185` |
| wrong class count in the ETF target | $C$ from the simulated bank's `cls` while the metric stream is the real cohort: real labels $\ge C$ are silently dropped from the class statistics, or $K < C$ every batch | nothing in the record; `sep_n_classes` is exposed by the loss and not logged | F-t, F-v |
| ramp without horizon | $\tau_{\rm sep} > 0$ with `total_steps` `None`: the adapter raises (`dsn_loss_adapter.py:169-173`); inside the DSN module the same case would reach a `warnings.warn` with `warnings` not imported (F-x) | a `ValueError` at build time (joint path); unreachable from the joint path otherwise | `[REPO]` |
| pre-training without balance | `A0`/`A0s` draw uniform batches: a batch can lack a class, making $K < 2$ and $\mathcal{L}_{\rm sep} = 0$, or hold no positive for an anchor [2026-10-05, E3 S3.4: at $C \ge 3$ a batch with two valid classes sets that pair's separation target to $-1$, so the expected loss has no exact zero; F-bc] | nothing per step; the encoder-only log prints `l_DSN` four times (`:164-165`) | `[REPO]` `run_joint_arms.py:158` |
| ledger says one filter, runner trains another | would occur if `spec.fixed["strict_semihard"]` and `--strict-semihard` disagreed; `build_argv` passes it, so they do not (unlike F-a, F-b) | `config.strict_semihard` in the record equals the space's value; the **projected** value is not recorded separately | `[REPO]` `npe_tune_joint.py:206-211` |

### 3.8 Findings this document owns

Stated next to the parameter above; collected here with their resolution.
The table of record is `00_INDEX.md` S6.

| id | finding | where stated | resolution |
|---|---|---|---|
| F-f | `canonicalise_config`'s docstring pins the inactive margin at 0.3, and `boundary_axes`'s docstring says the same (`joint_space.py:536-538`, `:780-782`); `INACTIVE_CANONICAL` pins 0.2 after its `[CORRECTION]` | S3.3 | report; open (docstrings; D-038 stream candidate) |
| F-h | the joint stack's base loss (`DSNLossConfig`: `joint_sep`, `easy_pos_semihard_neg`, 0.2, filter on) is not the standalone DSN's (`TrainConfig`: `triplet`, `hard`, 0.3, filter off) | S3.5 | report |
| F-q | the `A0`/`A0s` pre-training optimiser is AdamW on torch's defaults, not the runner's `--weight-decay` / `--one-minus-beta1` | S3.2, S3.6 | report; open (with P5) |
| F-t | the class count of the separation target and of the class mask is `len(unique(sim["cls"]))` (`run_joint_arms.py:402`), the **simulated** bank's, even when the metric stream is the real cohort (`A0`, `A2`, `A3`); equal on the bench by construction, unchecked on a cohort bank, where a real label $\ge C$ contributes to no class statistic and no error is raised (`dsn_joint_loss.py:112-123`) | S3.2, S3.4, S3.7 | report; open (take $C$ from the metric source's labels) |
| F-u | with `strict_semihard` fixed at 1 and projected, 9 of the DSN's 13 legal (mining, loss, filter) conditions are reachable from the joint space: the four cells of the easy-positive miners under `joint`/`joint_sep` with the filter **off** are not `[RAN]` | S3.3.3 | report (a scope statement of plan S5.1, not a bug) |
| F-v | the loss's per-batch counts `n_mined`, `n_strict`, `n_active` and the separation statistics `sep_mean_cos`, `sep_n_classes` are exposed by `DSNLossAdapter.stats()` but never written to the history; a dead term shows only as `dsn = 0.0` and `rho_grad = NaN` | S3.7 | report; open (log `stats()` per epoch; D-038 stream candidate) |
| F-w | `build_argv` passes `--strict-semihard` from `spec.fixed` but not `--sep-warmup-frac`; the space's fixed 0.0 and the runner's default 0.0 agree, so parity holds today by coincidence of defaults, the F-a pattern one default change away | S3.1, S3.4 | report; open (pass it, as `strict_semihard` is passed) |
| F-x | `dsn_joint_loss.py` calls `warnings.warn` (`:638`) without importing `warnings`: a `SepWarmup` with a positive fraction and no horizon raises `NameError` instead of warning; unreachable from the joint path, whose adapter refuses that case first (`dsn_loss_adapter.py:169-173`); a DSN-tree defect, out of scope for fixes (D-037) | S3.7 | report (DSN tree) |

## 4. Summary of results

- The metric term is one explicit function of the knobs, eq. (P2.1)-(P2.11):
  a class-balanced batch (P2.1), unit embeddings with the margin converted
  once, $m_{\rm sq} = 2 m_{\cos}$ (P2.2), one of three miners (P2.3), two
  hinges (P2.4)-(P2.5), a strict band filter and a mean over active triplets
  (P2.6), the plain library loss under `triplet` (P2.7), a raw-mean ETF
  penalty with target $-1/(K-1)$ (P2.8), a ramp that is identically 1 in the
  joint space (P2.9), the dispatch (P2.10), and the weight (P2.11).
- The constants: $4\tan^2\!\alpha$ from 0.00488 to 0.5299 over the range,
  the DSN's silhouette reading from 0.995 to 0.532 (P2.12), the margin band
  from 0.1 to 1.0 in $d_{\cos}$, the ETF target $-1$, $-0.5$, $-1/3$ at
  $K = 2, 3, 4$ (Table P2.1, `[RAN]`).
- Activity and legality (P2.13): `triplet` reads $m_{\cos}$; `joint` reads
  $\alpha$ and the fixed margin; `joint_sep` adds $\lambda_{\rm sep}$; the
  filter is zeroed under `triplet` and `hard`; with the filter fixed at 1,
  nine of thirteen cells are reachable (F-u).
- What differs from the standalone DSN (S3.5): the base loss (F-h), the
  `lambda_sep` top (1 against 20, F-g), the fixed filter and ramp, the batch
  construction without surrogates or cross-culture positives, the selection
  on `val_npe`, and the existence of $\lambda_{\rm dsn}$ at all.
- The record carries `dsn`, `rho_grad`, `sep_warmup`, `r_eff`,
  `cluster_silhouette`, `delta_hat` and the flags; it does not carry the
  triplet counts (F-v), and a dead term reads `dsn = 0.0`, `rho_grad = NaN`
  (S3.7).

## 5. Open points, caveats, assumptions

- **No run.** Every statement about behaviour (collapse, the dead-term
  signature, the opposed pairing) is a property of the code or a prediction
  of the plan; none is measured in `hpc/joint/`.
- **The silhouette readings (P2.12) are the DSN's rulers**, verified here
  only for the isosceles triplet; the step from one triplet to a cloud's
  silhouette is not proved anywhere read.
- **The library version.** The miner and loss semantics are those of
  `pytorch_metric_learning` 1.6.3, read from the wheel; the cluster's
  `sbi_env` records 1.6.3 (`[KB]` migration handoff S3.1), the other
  environment 2.9.0. A later version may change `BatchEasyHardMiner`'s
  semi-hard rule; the DSN's own smoke test, which constructs the miner and
  asserts a non-empty set, is the check.
- **Stale comments in the DSN tree** (out of scope for edits, D-037): the
  `TrainConfig.lambda_sep` scale note about centred means at $C \ge 3$
  (`config.py:673-677`); `train.py:332-333` on semi-hard negatives "inside
  the margin"; the missing `warnings` import (F-x).
- **The `[30, 45, 60, 75]` attribution** to Chung and Lee is carried from the
  DSN's comment, not from the retrieved full text, which states that the
  angle was a validation-set hyper-parameter without listing the values in
  the extracted text.
- **The angular loss's origin** (Wang et al., ICCV 2017) and FaceNet's
  semi-hard mining (Schroff et al., CVPR 2015) are not in PubMed (searches
  reported in S6); they are cited here through the two full texts that cite
  them and through the DSN's comments, and from memory as to venue and year
  (`[textbook, from memory]`-class statements, flagged).
- **Decisions this document does not make:** whether `strict_semihard`
  should be searched (F-u), whether $C$ should come from the metric
  source (F-t), whether the loss statistics are logged (F-v), whether
  `--sep-warmup-frac` is passed (F-w); all are reported to the Open calls
  through the index.

## 6. References / further reading

Source classes as E0 S1.1: `[REPO]`, `[KB]`, `[KB-PDF]`, `[RAN]`,
`[reasoning]`, `[PubMed full text]`, `[PubMed abstract]`, `[CODE pml 1.6.3]`.

**Project knowledge base.** `claude/deck_pack/03_SEC_B_joint_objective.md`
(the objective, the streams, gradient reach, test J8), `04_SEC_C_information_argument.md`
(the label ceiling, $r_{\rm eff} = C - 1$, predictions P3, P4, P7-P9),
`09_NOTATION_AND_GLOSSARY.md`; `JOINT_DSN_NPE_USAGE_v1.md` v1.3;
`claude/HANDOFF_2026-09-19_migration.md` S3.1 (library versions).

**Repository documents** (`[REPO]` doc, read): `hpc/dsn/Documentation/`
`TUNING_1_searched_axes.md` S3.9-3.14, S3.17; `TUNING_2_fixed_knobs.md`
S3.3; `THEORY_joint_condition_search.md` S3.3 (mask, clamp, the
$m_{\cos}$-$\alpha$ ridge), S3.6 (schedule), S3.7 (the separation term and
its history).

**Peer-reviewed, full text read through the PubMed connector.**

- Chung Y, Lee H. *Joint triplet loss with semi-hard constraint for data
  augmentation and disease prediction using gene expression data.* Sci Rep
  2023; 13:18178. [DOI](https://doi.org/10.1038/s41598-023-45467-8)
  (PMC10598120). The source of the DSN's `JointTripletLoss`: the strict
  semi-hard condition under anchor/positive switching, the margin plus
  angular joint loss, online semi-hard mining, the statement that hard
  samples can lead to bad local optima. **Deviations here:** the DSN scores
  in cosine geometry on unit vectors (the paper used Euclidean distance and
  reports degradation with cosine similarity in its ablation); the DSN has no
  separate switching-loss term, taking the symmetry from `swap` and from the
  angular hinge's construction; the paper's angle values are not in the
  retrieved text.
- Kertesz G. *Deep metric learning using negative sampling probability
  annealing.* Sensors 2022; 22:7579. [DOI](https://doi.org/10.3390/s22197579)
  (PMC9572431). Definitions of random-hard, semi-hard and hardest negative
  mining; semi-hard negative sampling attributed to Schroff et al. (FaceNet);
  the qualitative finding that hardest-negative sampling struggled early in
  training on its synthetic clusters. Its reading of Xuan et al. (easy
  positives; hard negatives at early stages lead to local minima) is a
  secondary citation here.
- Papyan V, Han XY, Donoho DL. *Prevalence of neural collapse during the
  terminal phase of deep learning training.* Proc Natl Acad Sci USA 2020;
  117:24652-24663. [DOI](https://doi.org/10.1073/pnas.2015509117)
  (PMC7547234). NC1 (within-class variability collapses to the class means)
  and NC2 (globally centred class means converge to a simplex ETF), measured
  during the terminal phase of training with cross-entropy, over 300 to 350
  epochs; the paper's class means are the globally centred ones, the DSN's
  raw ones (S3.2), a deliberate deviation recorded in the DSN's code.
- Fang C, He H, Long Q, Su WJ. *Exploring deep neural networks via
  layer-peeled model: minority collapse in imbalanced training.* Proc Natl
  Acad Sci USA 2021; 118:e2103091118.
  [DOI](https://doi.org/10.1073/pnas.2103091118) (PMC8639364). In the
  layer-peeled surrogate, every minimiser on class-balanced data exhibits
  NC1-NC3 (cross-entropy and, by its Theorem 3, the supervised contrastive
  loss); under imbalance the minority classifiers become indistinguishable
  above a ratio threshold ("minority collapse"), mitigated by oversampling --
  the reason a class-balanced metric batch (P2.1) is the right construction
  for an ETF target, and a caveat for cohorts with unequal class sizes.

**Searches run and their returns** (PubMed, 2026-10-01): `"triplet loss" AND
("semi-hard" OR "semihard") AND mining` -- 4 hits, Kertesz 2022 (PMC, read),
Dlamini and van Zyl 2021 (Sensors, PMC, not read: wildlife
re-identification), two 2026 applications without PMC full text (Arin and
Yetik, Forensic Sci Int, DOI 10.1016/j.forsciint.2026.113061; Jeong,
Bioinformatics, DOI 10.1093/bioinformatics/btag729), abstract-only, not
used. `"angular loss" AND (triplet OR "metric learning")` -- 1 hit, Chung
and Lee 2023 (read). `Papyan Han Donoho neural collapse ...` -- 1 hit
(read). `"equiangular tight frame" AND (loss OR "metric learning" OR
"neural collapse")` -- 8 hits: Papyan 2020 and Fang 2021 (read); Cao et al.
2025 PNAS (simplex compression under adversarial training, PMC, not read);
Jeong and Heo 2024 Sensors (ETF transformer for segmentation, PMC, not
read); three without PMC full text (Li et al. 2026 TPAMI, Lai et al. 2026
Neural Netw, Wang et al. 2026 TNNLS), abstract-only, not used; Du et al.
2023 PRL (quantum classifiers), not relevant. `("easy positive" OR "easy
positives") AND triplet AND (mining OR embedding)` -- 0 hits. `(FaceNet OR
"deep metric learning with angular loss") AND (Schroff OR "Wang J")` -- 0
hits: the two originating papers are computer-vision conference papers
PubMed does not index. **bioRxiv** (no keyword search exists in the
connector): the last 30 days of `bioinformatics` inspected by title (30
preprints returned), nothing on metric learning, triplet mining or neural
collapse; nothing cited. No data-repository query: no claim of this document
concerns a dataset.

**Code read.** `hpc/dsn/dsn_joint_loss.py`, `train.py:55-114, 216-351`,
`condition_space.py`, `config.py:615-724, 899-930, 1002-1004`,
`hpc/preflight_config.py:140-200`, `smoke_test_dsn_joint_loss.py:150-206`;
`hpc/joint/stage2/dsn_loss_adapter.py`, `joint_batches.py`,
`joint_train.py`; `stage3/run_joint_arms.py:150-202, 205-271, 334-534,
578-612`; `stage3/jobs/joint_arms.pbs:50-115`; `stage4/joint_space.py`;
`stage4/npe_tune_joint.py:86-227`; `pytorch_metric_learning` 1.6.3 as listed
in the changelog `[CODE pml 1.6.3]`.

---

### Pre-send check (Precision model)

R1: every object carries its type in S1; distances carry their unit
($d_{\cos}$ or $Q$) and the margin its unit at every use; the masks are
sets of knobs; $\Pi$ a map on triples. R2: "the strict set is empty" carries
"under `hard`"; "$\gamma_{\rm sep} \equiv 1$" carries "in the joint space,
$\tau_{\rm sep} = 0$"; "collapse is the minimum" carries "of the composite
loss"; the silhouette floors carry "the DSN's reading, isosceles triplet";
"equal on the bench" carries "both arms from one generator". R3: the ETF
geometry is cited for centred class means (Papyan) and applied to raw ones
only with the DSN's stated reason; the balanced-case theorem (Fang) is
cited for the layer-peeled surrogate and not transplanted to this loss;
Kertesz's and Chung and Lee's findings are reported for their settings, not
asserted for this one. R4: $Q$ versus $d_{\cos}$, $m_{\rm sq}$ versus
$m_{\cos}$, $K$ versus $C$, $\bar z^{(B)}_c$ versus $\bar z_c$ each named
once and kept apart; the DSN's $t$ in steps is renamed $n_{\rm done}$.
R5: $h_\psi : \mathbb{R}^{W} \to S^{E-1}$; every distance is on $S^{E-1}$;
the loss maps a batch of embeddings and labels to a scalar. R6: "active"
and "hard" declared in S1.1 with both senses; "fixed" by plan versus by
default in S3.4. R7: the DSN's comments are quoted with "the DSN's reading";
Kertesz's attribution of semi-hard mining to FaceNet is marked as his.
R8: the loss (computed, one batch) against its expectation (analytic); the
batch class mean against the population one; $K$ against $C$; the moves
named where they occur (S1.1, S3.2). Result: no fix needed.
