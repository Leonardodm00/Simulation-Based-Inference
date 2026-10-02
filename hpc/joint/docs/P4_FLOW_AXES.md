# P4 -- The flow block: `hidden_features`, `num_transforms`, the fixed `num_bins` and the two standardisation choices

**Document P4 of the joint documentation set.** Owner of the `flow` block
of `JOINT_KNOB_ORDER` (`hidden_features`, `num_transforms`) and of the flow
knobs the joint stack holds fixed or sets by code: `num_bins` (fixed at 10
by plan S5.1, trained at 8: F-a), `z_score_theta =
"transform_to_unconstrained"` with the prior passed as `x_dist`,
`z_score_x = "none"` (plan D6), the width rule that resolves the
`hidden_features` range from the bank's shape, and the constants the two
libraries fix below any flag (the spline bound and slope floor, the fully
autoregressive conditioning, the alternating orders, the ReLU conditioner,
the standard-normal base). Master notation: E0. The chapter that explains
what a flow is *for* and builds neural posterior estimation up from the
change of variables: E2. **Date:** 2026-10-02 (v1). **Applies to:** the
repository `Simulation-Based-Inference` at `834eb41`, `hpc/joint/` (D-037):
`stage2/joint_model.py`, `stage3/run_joint_arms.py`,
`stage3/jobs/joint_arms.pbs`, `stage4/joint_space.py`,
`stage4/npe_tune_joint.py`, `stage3b/run_stage3b.py`,
`stage3c/run_stage3c.py`, and the objects the joint stack reads its ranges
and defaults from, `npe_tune_search.py` and `npe_model.py`; the plan
`JOINT_DSN_NPE_PLAN_v0_6.md` at its repository version **v0.6.5**; and the
two installed libraries that build the flow, read from their wheels:
`sbi` 0.27.0 (sha256 `e7ef7800...27e0b8`) and `zuko` 1.6.0 (sha256
`5c073b61...48a4a`), the versions `sbi_env` carries (`[KB]` `HPC_PATHS.md`
sec. 7, package records read 2026-10-01). The one torch object the flow
depends on, the bijection torch registers for an interval support, was read
from the `torch` 2.10.0 source on GitHub (tag `v2.10.0`,
`torch/distributions/constraint_registry.py`); `sbi_env` imports torch
2.10.0 (`[KB]` `HPC_PATHS.md` sec. 7, 2026-10-01).

| date | change |
|---|---|
| 2026-10-02 | v1. Written from `stage2/joint_model.py` (read in full), `stage3/run_joint_arms.py:205-240, 282-325, 375-400, 440-480, 595-630`, `stage3/jobs/joint_arms.pbs:90-117`, `stage4/joint_space.py:1-50, 92-145, 212-331, 441-472, 760-811`, `stage4/npe_tune_joint.py:80-125, 180-235, 470-540`, `stage3b/run_stage3b.py:66-90`, `stage3c/run_stage3c.py:336-373`, `npe_tune_search.py:1-70, 95-170, 295-330`, `npe_model.py:1-140`; the `sbi` 0.27.0 wheel (`neural_nets/net_builders/flow.py:30-34, 574-636, 1078-1169, 1234-1317, 1391-1412`; `neural_nets/factory.py:56, 240-330`; `neural_nets/net_builders/estimator_configs.py:38-125`; `neural_nets/estimators/zuko_flow.py:17-175`; `utils/sbiutils.py:154-190, 323-371, 866-983`; `utils/nn_utils.py:17-47`; `utils/torchutils.py:525-600`); the `zuko` 1.6.0 wheel (`flows/spline.py:21-61`, `flows/autoregressive.py:25-316`, `nn.py:200-293`, `transforms.py:449-567, 966-1007`, `distributions.py:39-127`, `lazy.py:156-172, 290-330`); torch 2.10.0 `constraint_registry.py` (the interval and independent registrations); the plan S2.1, S2.2, S5.1, S8 (D6, D7), Stage 0; `[KB]` `JOINT_DSN_NPE_USAGE_v1.md` S3, S5, S7; `[KB-PDF]` the flows review and the practical guide. Every number of S3.1-S3.4 recomputed by the new torch-free `tools/p4_numbers.py` `[RAN]`, run twice with identical output. Findings F-ac to F-af added; F-a, F-j, F-m, F-r owned or co-owned. Grounding searches of S6 run and reported. |

**Abstract.** Two axes of the joint search, `hidden_features` and
`num_transforms`, and one fixed knob, `num_bins`, decide how sharp and how
shaped a posterior $q_\omega(\theta \mid z)$ the stack can represent for a
given conditioner $z$; two further choices made in code, the
box-to-unconstrained transform of $\theta$ and the absence of any
standardisation of $x$, decide the coordinates the flow works in and the
object the encoder hands it. The question this document answers is, for each
of these and for the constants the libraries fix below them: where the value
is set and by which surface it reaches the trainer (S3.1); what the flow
computes, written out from the two libraries' source as one explicit
function of the knobs, from the box map through the stacked autoregressive
spline transforms to the per-row loss and the sampler, eq. (P4.1)-(P4.12),
with the weight count in closed form and checked against the layer shapes
(S3.2); what each axis changes in that function (S3.3), including one fact
that neither the plan nor the code's docstrings state: `sbi` 0.27.0 passes
`hidden_features` to `zuko` as a list of length `num_transforms`, so
`num_transforms` sets the *depth of every conditioner* as well as the number
of transforms (F-ac, `[RAN]`); what the fixed knobs commit the flow to
(S3.4); how the block relates to the standalone NPE tuner it was copied from
(S3.5); how it interacts with the other blocks (S3.6); how it fails and which
number reveals each failure (S3.7); and the findings this document owns
(S3.8): F-a (owned with P6; the one flow finding with a patch), F-j, F-m,
F-r, and four new ones -- the depth coupling F-ac, the width-rule string
the ledger records describes a different function (F-ad), the Stage 3b/3c
rebuilds restore the flow's three knobs from the checkpoint but not the
encoder's five (F-ae), and plan D6 is listed open while the code has taken
its recommended option (F-af). **Deliberately excluded:** why a normalising
flow is a density estimator and why maximum likelihood on simulated pairs
targets the posterior (E2, with `[KB-PDF]` the flows review); the encoder
whose output is the conditioner (P1); the two loss terms that share a step
with the NPE term (P2, P3), except for the samplers the replicate term uses;
the optimiser and schedule (P5); the search mechanics (P6); the Stage 3b/3c
diagnostics beyond their rebuild of the model (P7, E7). Nothing here is a
measurement of training behaviour: no job of `hpc/joint/` has run on the
cluster (`[KB]` usage v1.3 S9); every number is a property of the flow as
built, read from the code and the two wheels, recomputed from them, or a
closed form marked so.

---

## 1. Notation and symbols

A subset of E0's master table, same types and units, plus the symbols this
document adds (declared in E0 under its convention 14, group "Flow (P4)";
the flow's base variable and transform were reserved there for the first
chapter to need them, which is this one).

| Symbol | Name / Meaning | Type & domain | Units | First used in S |
|---|---|---|---|---|
| $x$ | one IFR window | $x \in \mathbb{R}^{W}_{\ge 0}$ | Hz (cohort); counts per bin per unit (bench) | S3.1 |
| $W$ | window length in samples | $\mathbb{N}$ | samples | S3.4 |
| $h_\psi$ | the encoder, $h_\psi : \mathbb{R}^{W} \to S^{E-1}$; sbi's `embedding_net` | map; weights $\psi$ | -- | S3.2 |
| $\psi$ | encoder weights | $\psi \in \mathbb{R}^{n_\psi}$ | -- | S3.2 |
| $z$ | the embedding of a window, $z = h_\psi(x)$; the flow's conditioner (zuko's "context") | $z \in S^{E-1} \subset \mathbb{R}^{E}$ | dimensionless | S3.2 |
| $S^{E-1}$ | the unit sphere in $\mathbb{R}^{E}$ | set | -- | S3.4 |
| $E$ | embedding dimension (`embedding_size`); the width of the conditioner's context input | $\mathbb{N}$ | -- | S3.1 |
| $\theta$ | the inference parameters of one row, in box (inference) coordinates; $\theta^{(k)}$ its $k$-th component | $\theta \in \Theta \subset \mathbb{R}^{d_\theta}$ | mixed | S3.2 |
| $\Theta$ | the prior box; the unit cube $(0, 1)^{d_\theta}$ on every bank built so far | set | -- | S3.2 |
| $d_\theta$ | parameter-space dimension; 26 on the DUP15HD banks, 10 on the bench | $\mathbb{N}$ | -- | S3.1 |
| $a_k, b_k$ | the box bounds on axis $k$; $0$ and $1$ on the unit cube | reals, $a_k < b_k$ | as $\theta^{(k)}$ | S3.2 |
| $k$ | axis index | $k \in \{1, \dots, d_\theta\}$ | -- | S3.2 |
| $p$ | latent dimension of the bank (the width rule's `--p` anchor); $p = d_\theta$ on every bank built so far | $\mathbb{N}$ | -- | S3.1 |
| $p_\Theta$ | the prior density, uniform on $\Theta$; $p_\Theta(\theta) = 1$ on the unit cube | density on $\Theta$ | (param units)$^{-d_\theta}$ | S3.2 |
| $q_\omega$ | the conditional flow, $q_\omega(\theta \mid z)$: a density on $\Theta$ for each fixed $z$ | conditional density; weights $\omega$ | (param units)$^{-d_\theta}$ | S3.2 |
| $\omega$ | flow weights: the conditioners' weights and biases; the base and the box map carry none | $\omega \in \mathbb{R}^{n_\omega}$ | -- | S3.2 |
| $n_\omega$ | number of flow weights as `parameters()` counts them, masked entries included (nominal) | $\mathbb{N}$ | -- | S3.2 |
| $n^{\rm live}_\omega$ | the flow weights the autoregressive masks leave live; $n^{\rm live}_\omega < n_\omega$ | $\mathbb{N}$ | -- | S3.2 |
| $n_{\rm hid}$ | hidden features (`hidden_features`): the width of every hidden layer of every conditioner | $\mathbb{N}$; searched log-uniform in $[52, 256]$ at $(p, E) = (26, 12)$ | -- | S3.1 |
| $n_{\rm tf}$ | transforms stacked in the flow (`num_transforms`); also, through sbi, the number of hidden layers of each conditioner (F-ac) | $\mathbb{N}$; searched uniform in $[4, 12]$ | -- | S3.1 |
| $K_{\rm bins}$ | bins of each rational-quadratic spline (`num_bins`); fixed at 10 by the space, 8 by the runner | $\mathbb{N}$ | -- | S3.1 |
| $r$ | transform (stage) index along the chain | $r \in \{1, \dots, n_{\rm tf}\}$ | -- | S3.2 |
| $\mathcal{F}_{\rm box}$ | the box-to-unconstrained map, $\mathcal{F}_{\rm box} : \Theta \to \mathbb{R}^{d_\theta}$ (sbi's `transform_to_unconstrained`); no weights | map | -- | S3.2 |
| $\vartheta$ | the unconstrained parameter, $\vartheta = \mathcal{F}_{\rm box}(\theta)$; $\vartheta^{(k)} = \mathrm{logit}\,\theta^{(k)}$ on the unit cube | $\vartheta \in \mathbb{R}^{d_\theta}$ | dimensionless | S3.2 |
| $\mathcal{F}_\omega$ | the flow's whole transform, data to base, $\mathcal{F}_\omega(\cdot \mid z) : \Theta \to \mathbb{R}^{d_\theta}$ for each fixed $z$ | map; weights $\omega$ | -- | S3.2 |
| $\mathcal{F}^{(r)}_\omega$ | the $r$-th stacked autoregressive transform, $\mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta}$ for each fixed $z$ | map | -- | S3.2 |
| $v$ | the input vector of one stacked transform (the output of the previous one, $\vartheta$ for $r = 1$); $v^{(k)}$ its $k$-th component | $v \in \mathbb{R}^{d_\theta}$ | dimensionless | S3.2 |
| $n_\varphi$ | spline parameters per axis per transform, $n_\varphi = 3 K_{\rm bins} - 1$ (29 at 10 bins, 23 at 8) | $\mathbb{N}$ | -- | S3.2 |
| $\mathcal{C}^{(r)}_\omega$ | the conditioner of transform $r$ (zuko's hyper-network, a `MaskedMLP`): $\mathbb{R}^{d_\theta + E} \to \mathbb{R}^{d_\theta \times n_\varphi}$ | masked MLP | -- | S3.2 |
| $\varphi$ | the spline parameters of one axis of one transform (zuko's `phi`); $\varphi^{(r)}_k$ those of axis $k$ in transform $r$ | $\varphi \in \mathbb{R}^{n_\varphi}$ | dimensionless | S3.2 |
| $S_{\rm rqs}$ | the monotonic rational-quadratic spline on $[-B_{\rm rqs}, B_{\rm rqs}]$ with $K_{\rm bins}$ bins, the identity outside; written $S_{\rm rqs}(\cdot\,; \varphi)$ | map $\mathbb{R} \to \mathbb{R}$ | -- | S3.2 |
| $B_{\rm rqs}$ | the spline bound (zuko `bound`), 5 | $\mathbb{R}_{>0}$ | dimensionless | S3.2 |
| $\delta_{\rm rqs}$ | the spline's slope floor (zuko `slope`), $10^{-3}$ | $\mathbb{R}_{>0}$ | dimensionless | S3.2 |
| $\zeta$ | the base variable, $\zeta = \mathcal{F}_\omega(\theta \mid z)$; standard normal under the flow | $\zeta \in \mathbb{R}^{d_\theta}$ | dimensionless | S3.2 |
| $p_\zeta$ | the base density: the standard normal on $\mathbb{R}^{d_\theta}$ (zuko `DiagNormal(0, I)`) | density | -- | S3.2 |
| $i$ | row index | $i \in \{1, \dots, n\}$ for the batch in play | -- | S3.2 |
| $n$ | a generic count, qualified in prose | $\mathbb{N}$ | -- | S3.2 |
| $\ell_i$ | the per-row NPE loss $-\log q_\omega(\theta_i \mid h_\psi(x_i))$ | $\mathbb{R}$ | nats | S3.2 |
| $B_{\rm sim}$ | simulated rows per optimiser step | $\mathbb{N}$ | rows | S3.2 |
| $\mathcal{B}_{\rm sim}$ | one simulated minibatch | index set of size $B_{\rm sim}$ | -- | S3.2 |
| $\mathcal{L}^{\rm sim}_{\rm NPE}$ | expected NLL under $p_{\rm sim}$, plan eq. (1a) (analytic level); estimated by $L$ | $\mathbb{R}$ | nats/row | S3.2 |
| $L$ | held-out NLL (computed level) | $\mathbb{R}$ | nats/row | S3.2 |
| $L_0$ | the prior floor, $-\mathbb{E}[\log p_\Theta(\theta)]$ | $\mathbb{R}$ | nats/row | S3.2 |
| $\hat\Delta$ | information gain, $L_0 - L$ | $\mathbb{R}$ | nats/row | S3.7 |
| $H[\cdot]$ | entropy, in nats | $\mathbb{R}_{\ge 0}$ | nats | S3.2 |
| $g$ | a row (well) of the real bank | index | -- | S3.2 |
| $s$ | posterior-draw index | $s \in \{1, \dots, S_{\rm mc}\}$ | -- | S3.2 |
| $S_{\rm mc}$ | posterior draws per well (`n_posterior_draws`, P3) | $\mathbb{N}$ | draws | S3.2 |
| $\theta^{(s)}_g$ | the $s$-th posterior draw for well $g$, i.i.d. from $q_\omega(\cdot \mid z_g)$ for each fixed $z_g$ | $\Theta$ | mixed | S3.2 |
| $B_{\rm rep}$ | replicate pairs per step (P3) | $\mathbb{N}$ | pairs | S3.6 |
| $\lambda_{\rm rep}$ | weight of the replicate term (P3) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.6 |
| $\bar C$ | the symmetrised posterior covariance the replicate term computes (P3) | PSD $d_\theta \times d_\theta$ | (param units)$^2$ | S3.6 |
| $\eta$, $\gamma_{\rm wd}$ | the learning rate and the weight-decay coefficient (P5) | $\mathbb{R}_{>0}$, $\mathbb{R}_{\ge 0}$ | dimensionless | S3.6 |
| $n_{\rm ep}, n_{\rm step}$ | epochs, and optimiser steps per epoch (P5) | $\mathbb{N}$ | -- | S3.6 |
| $r_{\rm eff}$ | effective rank of an embedding cloud (E3); not the index $r$ | $[1, E]$ | -- | S3.6 |
| $d_{\rm exp}$ | the encoder's depth exponent (P1) | $\mathbb{N}$ | -- | S3.6 |
| $\mathcal{G}$ | a realised connectivity graph (a latent of the simulator) | adjacency matrix | -- | S3.7 |

### 1.1 Conventions

- Code names in backticks are the knobs as the code spells them; the symbol
  of the same quantity is used only where a formula needs it.
- **Direction.** $\mathcal{F}_\omega$ maps data to base, as zuko writes its
  `transform`: density evaluation applies $\mathcal{F}_\omega$, sampling
  applies $\mathcal{F}_\omega^{-1}$. The flows review `[KB-PDF p.3]` writes
  the opposite direction (its transformation maps base to data); every
  statement here follows zuko's,
  so that "the first transform" means the first applied to $\theta$, which
  is $\mathcal{F}_{\rm box}$.
- **"Transform" has three senses here, declared where they meet (R6):** a
  *stacked transform* $\mathcal{F}^{(r)}_\omega$ (`num_transforms` counts
  these); the *box transform* $\mathcal{F}_{\rm box}$, which sbi calls a
  "z-scoring option" (`z_score_theta`) although it standardises nothing;
  and sbi's *standardising transform* (`z_score_x = "independent"`), an
  affine map fitted to a batch, which the joint stack does not use.
- **"Hidden features" is the conditioner's width**, $n_{\rm hid}$: the width
  of each hidden layer of each conditioner. It is not the embedding
  dimension $E$ (the conditioner's *input* width is $d_\theta + E$), not the
  DSN backbone's widths $w_b$ (P1), and -- the point of F-ac -- not a single
  hidden layer: there are $n_{\rm tf}$ of them per conditioner.
- **Levels** (E0 convention 4): $q_\omega(\theta \mid z)$, $\mathcal{F}_\omega$
  and $n_\omega$ are properties of the built network at fixed $\omega$;
  $\ell_i$ is the number one row puts in memory, $L$ the mean over a split
  (computed level), $\mathcal{L}^{\rm sim}_{\rm NPE}$ its expectation
  (analytic level). The identity-initialised loss of S3.2 is an analytic
  value for an idealised initialisation, not a logged number.
- **Status** (the Provenance model): `hidden_features`, `num_transforms` are
  `configured` (searched); `num_bins`, `z_score_theta`, `z_score_x` and the
  prior passed as `x_dist` are `configured` by code; $B_{\rm rqs}$,
  $\delta_{\rm rqs}$, the autoregressive passes, the orders, the activation,
  the base and the initialisation are `configured` by the libraries, below
  any flag; $n_\omega$, $n^{\rm live}_\omega$ and the hidden-layer count are
  `computed` from the configuration and **recorded nowhere** (S3.2);
  $\vartheta$, $\zeta$, $\varphi$ are `computed` per row inside the forward
  pass; $L_0 = 0$ on the unit cube is `analytic`.
- **Default** is per surface, as P0 S1.1: the runner's flag default is what a
  Stage 3 arm trains with, and what a Stage 4 trial trains with for any flag
  `build_argv` does not pass; the library defaults (`build_joint_model`;
  sbi's `posterior_nn` and `build_zuko_nsf`; zuko's own) are what a direct
  caller gets and are never what a cluster job runs (F-m).
- Equations are numbered (P4.n); plan equations are cited as "plan eq. (n)";
  the flows review as `[KB-PDF p.n]` (Papamakarios et al., JMLR 2021, the
  project PDF `Normalizing_Flows_for_Probabilistic_Modeling_and_Inference`).
- Line references into the two wheels are written `sbi flow.py:1143` and
  `zuko nn.py:271`, meaning the file inside the installed package at the
  stated version.
- Logarithms are natural; "nats/row" is per training row (one window).

## 2. Glossary

Ordered by first appearance.

- **Density estimator (sbi)** -- the object `posterior_nn(...)(theta, x)`
  returns: the flow, the embedding net and the shapes, with `.loss`,
  `.log_prob`, `.sample`. The joint model wraps one. S3.1.
- **Embedding net** -- sbi's name for the network applied to the condition
  before the flow reads it; here the DSN backbone $h_\psi$. S3.1.
- **Width rule** -- the function of $(p, E)$ that sets the `hidden_features`
  range; copied from the standalone NPE tuner. S3.1.
- **Change of variables** -- the identity that turns a bijection and a base
  density into a density on the data side, eq. (P4.1); `[KB-PDF p.3]`.
  S3.2.
- **Box transform (unconstrained transform)** -- the coordinatewise logit
  that maps the prior box onto $\mathbb{R}^{d_\theta}$, prepended to the flow;
  built by torch's `biject_to` from the prior's support. S3.2.
- **Autoregressive transform** -- a map whose $k$-th output depends on the
  $k$-th input and on the inputs before it in a fixed order, so that its
  Jacobian is triangular; `[KB-PDF p.12-13]`. S3.2.
- **Conditioner (hyper-network)** -- the network that turns the inputs
  before axis $k$, and the context $z$, into the parameters of axis $k$'s
  spline; the only part of the flow that carries weights. S3.2.
- **Masked MLP** -- a multilayer perceptron whose weight matrices are
  multiplied by fixed binary masks so that the autoregressive dependence
  holds for every axis at once; `[KB-PDF p.18]`. S3.2.
- **Rational-quadratic spline (RQS)** -- a monotone piecewise function made
  of $K_{\rm bins}$ rational-quadratic segments between knots, with an
  analytic inverse; `[KB-PDF p.16-17]`. S3.2.
- **Knot** -- a boundary between two bins; the spline's parameters are the
  knots' positions on both axes and the slopes at the interior knots. S3.2.
- **Tails** -- the two half-lines outside $[-B_{\rm rqs}, B_{\rm rqs}]$,
  where the spline is the identity. S3.2.
- **Base distribution** -- the fixed distribution the flow maps data to; the
  standard normal here. S3.2.
- **Order** -- the permutation of axes a stacked transform conditions along;
  ascending and descending alternate. S3.2.
- **Pass** -- one evaluation of a conditioner on a full vector; sampling a
  fully autoregressive transform takes $d_\theta$ of them. S3.2.
- **Nominal and live weights** -- every element of the weight tensors
  against the elements the masks do not zero. S3.2.
- **Log-uniform prior (search)** -- the sampling density the optimiser puts
  on an axis: uniform in the logarithm, so that each decade gets equal
  weight. S3.3.
- **Standardising transform (sbi)** -- the per-feature affine map
  `Standardize(mean, std)` fitted to the example batch, inserted when
  `z_score_x` is `"independent"` or `"structured"`; not used here. S3.4.

## 3. Main body

### 3.1 Where the flow knobs live, and what reaches the trainer

*Establishes the surfaces a flow knob can be set on and the values each
surface holds, so that every later statement "the default is ..." has a
surface attached.*

The block has two searched axes, one knob the space declares fixed, two
choices made by code, one object passed by code (the prior), and six
constants fixed inside the libraries. They are set on eight surfaces. A
Stage 4 trial is built by `npe_tune_joint.build_argv` from a configuration
of the space; a Stage 3 arm is built from the runner's flags; the cluster
job sets the runner's flags from its `-v` variables; a direct caller of the
library gets the library's defaults, and the library's own defaults sit on
two further layers inside sbi and one inside zuko.

| knob | `JointSpaceSpec` / `default_joint_space` | tuner -> runner flag (`AXIS_TO_FLAG`) | runner flag default (`run_joint_arms.py`) | PBS job (`joint_arms.pbs`) | `build_joint_model` default | sbi `posterior_nn` / `build_zuko_nsf` default | zuko default |
|---|---|---|---|---|---|---|---|
| `hidden_features` $n_{\rm hid}$ | width rule: $[52, 256]$ at $(26, 12)$, $[32, 256]$ on the bench; log-uniform integer | `--hidden-features` | `48` | **not passed** | `64` | `50` / `50` | `(64, 64)`: two hidden layers, whatever the transform count |
| `num_transforms` $n_{\rm tf}$ | $[4, 12]$, uniform integer | `--num-transforms` | `3` | **not passed** | `5` | `5` / `5` | `3` |
| `num_bins` $K_{\rm bins}$ | **fixed** at `10` (`spec.fixed`) | **not passed** (F-a) | `8` | **not passed** | `10` | `10` / `8` | `8` |
| `z_score_theta` | not an axis | -- | no flag | -- | `"transform_to_unconstrained"` | `"independent"` | -- |
| `z_score_x` | not an axis | -- | no flag | -- | `"none"` | `"independent"` | -- |
| the prior (`x_dist`) | not an axis | -- | `BoxUniform(0, 1)` built in `main` | -- | required argument | optional; required by `transform_to_unconstrained` | -- |
| `bound` $B_{\rm rqs}$, `slope` $\delta_{\rm rqs}$ | -- | -- | -- | -- | not reachable (sbi strips `tail_bound`) | not reachable | `5.0`, `1e-3` |
| `passes`, `order`, activation, `residual` | -- | -- | -- | -- | not reachable | not reachable | fully autoregressive; alternating; `ReLU`; `False` |

`[REPO]` `stage4/joint_space.py:139-140, 235-236, 299-302, 319, 322-324,
460-464`; `stage4/npe_tune_joint.py:94-95, 206-211`;
`stage3/run_joint_arms.py:227-229, 385-388, 463-466`;
`stage3/jobs/joint_arms.pbs:102-115`; `stage2/joint_model.py:195-198,
209-219`; `[REPO sbi wheel]` `factory.py:248-250`, `flow.py:583-586,
1140-1143`; `[REPO zuko wheel]` `spline.py:52-53`, `autoregressive.py:
108-124, 294-307`, `nn.py:255-258`, `transforms.py:474-475`.

Four readings of the table that the rest of the document uses.

1. **The job passes no flow flag.** `joint_arms.pbs` forwards `EPOCHS`,
   `STEPS_PER_EPOCH`, `B_SIM`, `LAMBDA_DSN`, `LAMBDA_REP`, `N_DRAWS` and
   `ENCODER_STEPS` and nothing else (`:102-115`; `[KB]` usage v1.3 S5.2
   lists the recognised variables), so every Stage 3 arm on the cluster
   trains the runner's flow, 48 hidden features, 3 transforms, 8 bins,
   unless the runner is invoked by hand.
2. **The tuner passes two of the three.** `AXIS_TO_FLAG` maps
   `hidden_features` and `num_transforms` to their flags; `num_bins` is in
   `spec.fixed`, which `build_argv` consults only for `strict_semihard`
   (`npe_tune_joint.py:206-211`). A Stage 4 trial therefore trains an 8-bin
   flow while the space, `provenance_report` and P0 Table A say 10 (F-a).
   The trace is in three places: the trial's recorded `argv` carries no
   `--num-bins`, and the runner's record `config.num_bins` and the
   checkpoint's `meta["flow"]["num_bins"]` both say 8
   (`run_joint_arms.py:603`, `joint_model.py:224-226`).
3. **The checkpoint is the record of truth for the flow.** `build_joint_model`
   writes the three knobs and the two standardisation choices into `meta`
   (`:221-226`), and `stage3c._rebuild_model` rebuilds the flow from exactly
   those entries, refusing a checkpoint without them (`run_stage3c.py:
   359-370`). The same function rebuilds the encoder from `W` and `E` and
   the runner's default encoder axes, not from the checkpoint's
   `backbone_repr` (F-ae, S3.8).
4. **Two Stage 3 defaults are outside the Stage 4 space** at the DUP15HD
   shapes: 48 below the lower bound 52 and 3 below 4 (F-r); on the bench
   only the transform count is outside ($48 \in [32, 256]$). A Stage 3 arm
   is therefore not a point of the space it is meant to be the baseline of,
   in two of the three flow coordinates.

*Plain.* Three numbers set the flow's size and one more its shape, and
they can be set in eight places that disagree. The job sets none, the
search sets two, and only the checkpoint says what was actually built.

### 3.2 The flow as built

*Establishes, as one explicit function of the knobs, what the flow is: the
density it evaluates, eq. (P4.1)-(P4.7), the loss it returns, eq. (P4.8),
the sampler it exposes, eq. (P4.11), and the weight count, eq. (P4.9) and
Table P4.1. Every constant is read from the two wheels; every derived number
is recomputed `[RAN]` (`tools/p4_numbers.py`, blocks named).*

**What `posterior_nn` builds.** `build_joint_model` calls
`posterior_nn(model="zuko_nsf", embedding_net=backbone,
z_score_theta=..., z_score_x=..., hidden_features=..., num_transforms=...,
num_bins=..., x_dist=prior)` and applies the returned builder to the first
64 rows of the training split (`joint_model.py:209-219`;
`run_joint_arms.py:463`). Inside sbi the two names swap roles --
`z_score_theta` becomes the builder's `z_score_x` and `z_score_x` its
`z_score_y`, because the low-level builders call the estimated variable
`x` and the condition `y` (`factory.py:290-292`; the module docstring of
`joint_model.py` records the trap). `build_zuko_nsf` renames `num_bins` to
zuko's `bins` and calls `build_zuko_flow` (`flow.py:622-634`), which

- reads $d_\theta$ from the example $\theta$ and $E$ by running the backbone
  on one example window (`flow.py:1136-1137`; `nn_utils.py:40`) -- the 64
  rows serve shapes only, because neither standardisation choice of this
  stack estimates anything from them (S3.4);
- turns the integer `hidden_features` into the list
  `[hidden_features] * num_transforms` (`flow.py:1142-1143`);
- builds `zuko.flows.NSF(features=d_theta, context=E,
  hidden_features=<that list>, transforms=num_transforms, bins=num_bins)`
  and takes its base and its tuple of transforms (`flow.py:1265-1275`);
- prepends the box transform (`_prepare_x_transforms`, `flow.py:1298-1312`)
  and composes `zuko.flows.Flow(transforms, base)` (`:1154-1160`);
- leaves the backbone as the embedding net, since `z_score_y = "none"`
  inserts no standardising net (`_prepare_y_embedding`, `:1409-1412`).

**The density.** For each fixed $z$ the flow is the normalising flow of
zuko's `NormalizingFlow` (`distributions.py:114-119`): the change of
variables through a bijection $\mathcal{F}_\omega(\cdot \mid z)$ from
$\Theta$ to $\mathbb{R}^{d_\theta}$ into the base density,

$$q_\omega(\theta \mid z) = p_\zeta\big(\mathcal{F}_\omega(\theta \mid z)\big)\,\Big\lvert \det \frac{\partial \mathcal{F}_\omega(\theta \mid z)}{\partial \theta} \Big\rvert, \qquad \zeta = \mathcal{F}_\omega(\theta \mid z) \in \mathbb{R}^{d_\theta}, \tag{P4.1}$$

the review's eq. (3) in zuko's direction `[KB-PDF p.3]`, with $p_\zeta$ the
standard normal density on $\mathbb{R}^{d_\theta}$ (`DiagNormal` with zero
mean and unit scale, registered as buffers, `autoregressive.py:309-313`).
The bijection is the chain

$$\mathcal{F}_\omega(\cdot \mid z) = \mathcal{F}^{(n_{\rm tf})}_\omega(\cdot \mid z) \circ \cdots \circ \mathcal{F}^{(1)}_\omega(\cdot \mid z) \circ \mathcal{F}_{\rm box}, \tag{P4.2}$$

applied right to left; the log-determinant of eq. (P4.1) is the sum of the
stages' log-determinants (`transforms.py:59-162`, the composed transform's
`call_and_ladj`; the review's eq. (6) `[KB-PDF p.3]`).

**The box transform.** `mcmc_transform(prior)` reads the prior's support,
finds it bounded, calls torch's `biject_to(prior.support)`, wraps the result
as an `IndependentTransform` over the last dimension and returns its
inverse (`sbiutils.py:935-983`); `biject_transform_zuko` turns that inverse
into a parameter-free zuko transform (`:356-371`, buffer, no weights). The
prior is `BoxUniform(low=0, high=1)` in $d_\theta$ dimensions
(`run_joint_arms.py:385-388`), an `Independent(Uniform(low, high), 1)`
(`torchutils.py:525-600`), whose support is the interval constraint with
tensor bounds; for that constraint torch 2.10.0 registers
`ComposeTransform([SigmoidTransform(), AffineTransform(loc=low,
scale=high - low)])` (the unit-interval shortcut applies only to Python
numbers, not to the tensors `BoxUniform` holds), so the inverse is, axis by
axis and for every $k$,

$$\vartheta^{(k)} = \mathcal{F}_{\rm box}(\theta)^{(k)} = \mathrm{logit}\Big(\frac{\theta^{(k)} - a_k}{b_k - a_k}\Big) = \mathrm{logit}\,\theta^{(k)} \ \text{on the unit cube}, \qquad \log \Big\lvert \det \frac{\partial \mathcal{F}_{\rm box}}{\partial \theta} \Big\rvert = -\sum_{k=1}^{d_\theta} \log\big(\theta^{(k)} (1 - \theta^{(k)})\big) \ \ge\ d_\theta \log 4. \tag{P4.3}$$

The map is fixed by the prior alone: it is the same for every $z$, carries
no weight, and is rebuilt from the prior at load time rather than stored
(`CallableTransform`, `sbiutils.py:323-328`: "transform tensors are
intentionally absent from `state_dict`"). It is a bijection from the open
cube onto $\mathbb{R}^{d_\theta}$, so a sample of the flow is never outside
the box and never on its boundary (S3.4). Under the uniform prior the
unconstrained coordinate $\vartheta^{(k)}$ follows the standard logistic law
on every axis, whose density at $\mathrm{logit}\,\theta^{(k)}$ is exactly
$\theta^{(k)}(1 - \theta^{(k)})$ `[RAN]` B5: the box Jacobian of eq. (P4.3)
is the uniform prior's own density change, which is what makes the anchor
of eq. (P4.10) exact.

**One stacked transform.** Each $\mathcal{F}^{(r)}_\omega$ is zuko's
`MaskedAutoregressiveTransform` with the rational-quadratic spline as its
univariate map (`spline.py:49-61`). On its input $v \in \mathbb{R}^{d_\theta}$
(the output of stage $r - 1$; $v = \vartheta$ at $r = 1$) and the context
$z$, it computes all spline parameters in one pass of its conditioner and
transforms every axis at once:

$$\varphi^{(r)} = \mathcal{C}^{(r)}_\omega(v, z) \in \mathbb{R}^{d_\theta \times n_\varphi}, \qquad \mathcal{F}^{(r)}_\omega(v \mid z)^{(k)} = S_{\rm rqs}\big(v^{(k)};\, \varphi^{(r)}_k\big) \quad \text{for each axis } k, \tag{P4.4}$$

where $\varphi^{(r)}_k$ depends on $v^{(k')}$ only for the axes $k'$ that
precede $k$ in the transform's order, and on all of $z$
(`autoregressive.py:207-218`: the context is concatenated to $v$ as the
conditioner's input, `:208-209`; the order is ascending for odd $r$ and
descending for even $r$, `:294-307`, `randperm=False` since sbi passes no
such argument). The Jacobian of each stage is therefore triangular in the
stage's order and its log-determinant is the sum over axes of the spline's
log-slope at $v^{(k)}$ (`transforms.py:1002-1007`; the review's eq. (31)-(32)
`[KB-PDF p.13]`). With $d_\theta = 26$ the two orders are the only ones
used: axis 1 is transformed as a function of nothing but $z$ in the odd
stages and of everything in the even ones, and the reverse for axis 26.

**The conditioner.** $\mathcal{C}^{(r)}_\omega$ is a `MaskedMLP`
(`nn.py:218-293`): plain linear layers with ReLU between them, no
normalisation and no residual connection (`residual=False`, `activation=None`
giving `nn.ReLU`, `:255-258`), each weight matrix multiplied in the forward
pass by a fixed binary mask (`MaskedLinear`, `:200-218`). Its input width is
$d_\theta + E$, its output width $d_\theta n_\varphi$ with
$n_\varphi = 3 K_{\rm bins} - 1$ spline parameters per axis
(`autoregressive.py:147-152`: the adjacency of the $d_\theta$ axes, with
a column of ones for every context feature, repeated $n_\varphi$ times
along the output axis), and -- the fact of F-ac -- its hidden layers
are the list sbi passed, so

$$\mathcal{C}^{(r)}_\omega : \mathbb{R}^{d_\theta + E} \to \mathbb{R}^{n_{\rm hid}} \to \cdots \to \mathbb{R}^{n_{\rm hid}} \to \mathbb{R}^{d_\theta \times n_\varphi}, \qquad n_{\rm tf} \text{ hidden layers of width } n_{\rm hid}, \tag{P4.5}$$

a masked linear map into each space and a ReLU after every one but the
last.

At the runner's defaults the four `MaskedLinear` shapes (out, in) are
$(48, 38)$, $(48, 48)$, $(48, 48)$, $(598, 48)$ at
$(d_\theta, E) = (26, 12)$ `[RAN]` B3: three hidden layers because `num_transforms` is 3. The
masks are built once from the order (`nn.py:271-293`): outputs with the same
dependency set are merged, hidden units are assigned cyclically to the
$d_\theta$ dependency classes (`:290`), and a unit may read a unit of an
earlier or equal class; the masked entries stay in the weight tensors,
are counted by `parameters()`, receive AdamW's weight decay, and never
reach the forward pass.

**The spline.** For one axis, $\varphi \in \mathbb{R}^{n_\varphi}$
holds $K_{\rm bins}$ width logits, $K_{\rm bins}$ height logits and
$K_{\rm bins} - 1$ interior log-slopes (`spline.py:60`). zuko 1.6.0 first
soft-clips each group,

$$\varphi \mapsto \frac{\varphi}{1 + \lvert 2 \varphi / \ln \delta_{\rm rqs} \rvert} \ \text{(widths, heights)}, \qquad \varphi \mapsto \frac{\varphi}{1 + \lvert \varphi / \ln \delta_{\rm rqs} \rvert} \ \text{(slopes)}, \tag{P4.6}$$

then takes a softmax of the widths and of the heights, pads a leading zero,
cumulates, and places the knots at $B_{\rm rqs}(2 \cdot \text{cumsum} - 1)$
on both axes; the slopes are exponentiated with a zero padded at each end
(`transforms.py:480-490`). Consequences, all `[RAN]` B2: with
$\delta_{\rm rqs} = 10^{-3}$ the width and height logits are confined to
$\pm 3.454$, so no two bins of one spline differ in width by more than a
factor $1000$ and the narrowest bin a conditioner can ask for is $0.00111$
of the domain's length $10$ at $K_{\rm bins} = 10$ ($0.00143$ at 8); the
interior slopes are confined to $[10^{-3}, 10^{3}]$ and the two boundary
slopes are exactly 1; outside $[-B_{\rm rqs}, B_{\rm rqs}]$ the transform is
the identity (`:532`, `torch.where(mask, y, x)`), so the tails are linear
with slope 1 and the spline is continuously differentiable across the
boundary. At $\varphi = 0$ -- $K_{\rm bins}$ equal bins, unit slopes -- the
spline is exactly the identity on the whole line (checked on a grid to
$10^{-16}$ `[RAN]` B2); the review describes the family and its segment search, logarithmic in
the bin count `[KB-PDF p.16-17]`, and the rational-quadratic form itself is
Durkan et al.'s (2019), not in the knowledge base (S6). The domain is set in
the coordinate the stage acts on: for the first stage that is
$\vartheta = \mathrm{logit}\,\theta$, and $[-5, 5]$ there is
$\theta^{(k)} \in [0.00669, 0.99331]$ `[RAN]` B2 -- a point to which S3.4
returns.

**The per-row loss.** `JointDSNNPE.npe_loss(theta, x)` is
`estimator.loss(theta, condition=x)` (`joint_model.py:130-132`), which sbi
evaluates as minus the log-probability (`zuko_flow.py:127-139`): the
condition goes through the backbone, the flow is instantiated at the
embedding, and eq. (P4.1) is evaluated at $\theta_i$. Written out along the
chain,

$$\ell_i = -\log q_\omega\big(\theta_i \mid h_\psi(x_i)\big) = -\log p_\zeta(\zeta_i) - \sum_{r=1}^{n_{\rm tf}} \log \Big\lvert \det \frac{\partial \mathcal{F}^{(r)}_\omega}{\partial v} \Big\rvert + \sum_{k=1}^{d_\theta} \log\big(\theta_i^{(k)} (1 - \theta_i^{(k)})\big), \tag{P4.7}$$

each stage's determinant evaluated at that stage's input for row $i$; the
last sum is the box term of eq. (P4.3) with its sign reversed. The step's
NPE term is the batch mean, plan eq. (2):

$$\frac{1}{B_{\rm sim}} \sum_{i \in \mathcal{B}_{\rm sim}} \ell_i, \tag{P4.8}$$

and its expectation under $p_{\rm sim}$ is $\mathcal{L}^{\rm sim}_{\rm NPE}$,
plan eq. (1a), the forward Kullback-Leibler objective of the review's eq.
(13)-(14) `[KB-PDF p.6]`. $\ell_i$ is a density in **box coordinates**:
the box term is part of the chain, so $L$, $L_0$ and $\hat\Delta$ (E7) are
all in nats per row on $\Theta$, and the held-out $L$ of a flow that
matched the uniform prior exactly would be $L_0 = 0$ (eq. (P4.10)).

**The weight count.** From eq. (P4.5), every stage has one input layer,
$n_{\rm tf} - 1$ hidden-to-hidden layers and one output layer, each with a
bias, and there are $n_{\rm tf}$ stages; the base and the box map add
nothing, so

$$n_\omega = n_{\rm tf} \Big[ (d_\theta + E + 1)\, n_{\rm hid} + (n_{\rm tf} - 1)(n_{\rm hid} + 1)\, n_{\rm hid} + (n_{\rm hid} + 1)\, d_\theta n_\varphi \Big], \tag{P4.9}$$

checked against the layer shapes for every row of Table P4.1 `[RAN]` B3.
The live count $n^{\rm live}_\omega$ is the same sum with each weight matrix
replaced by the number of ones in its mask, computed by replicating
`MaskedMLP`'s mask construction; the masks leave 52-59% of the weights live
across the configurations below.

**Table P4.1 -- the flow's size across the surfaces of S3.1** `[RAN]` B3.
"Out" is the width of each conditioner's output, $d_\theta n_\varphi$.

| configuration ($n_{\rm hid}$ / $n_{\rm tf}$ / $K_{\rm bins}$) | $(d_\theta, E)$ | out per stage | $n_\omega$ per stage | live per stage | $n_\omega$ | $n^{\rm live}_\omega$ | live share |
|---|---|---|---|---|---|---|---|
| runner defaults 48 / 3 / 8 | (26, 12) | 598 | 35878 | 20186 | 107634 | 60558 | 0.56 |
| runner defaults at the runner's $E = 10$ | (26, 10) | 598 | 35782 | 20090 | 107346 | 60270 | 0.56 |
| `build_joint_model` 64 / 5 / 10 | (26, 12) | 754 | 68146 | 38570 | 340730 | 192850 | 0.57 |
| standalone `NPEConfig` 128 / 8 / 10 | (26, 12) | 754 | 217842 | 115254 | 1742736 | 922032 | 0.53 |
| space lower corner 52 / 4 / 10 | (26, 12) | 754 | 50258 | 26806 | 201032 | 107224 | 0.53 |
| the same corner as trained, 52 / 4 / 8 (F-a) | (26, 12) | 598 | 41990 | 22438 | 167960 | 89752 | 0.53 |
| space geometric middle 115 / 8 / 10 | (26, 12) | 754 | 185329 | 99914 | 1482632 | 799312 | 0.54 |
| space upper corner 256 / 12 / 10 | (26, 12) | 754 | 927474 | 485884 | 11129688 | 5830608 | 0.52 |
| bench, runner defaults 48 / 3 / 8 | (10, 10) | 230 | 16982 | 9854 | 50946 | 29562 | 0.58 |
| bench, space lower corner 32 / 4 / 10 | (10, 10) | 290 | 13410 | 7902 | 53640 | 31608 | 0.59 |
| bench, space upper corner 256 / 12 / 10 | (10, 10) | 290 | 803618 | 444748 | 9643416 | 5336976 | 0.55 |

For scale: the encoder at the runner's defaults has 359450 weights (P1
eq. (P1.10), `[RAN]`), so the Stage 3 flow is 0.3 of the encoder, the space's
lower corner 0.56 of it and its upper corner 31 times it; across the space
the flow's size spans a factor 55 `[RAN]` B6, and no surface records
$n_\omega$ (the runner records no parameter count for either network, as
F-s notes for the encoder).

**The anchor.** Two closed forms fix where $L$ starts and where it cannot
go below, `[RAN]` B5:

$$L_0 = -\mathbb{E}_{p_\Theta}[\log p_\Theta(\theta)] = 0 \ \text{on the unit cube}; \qquad \text{the identity-initialised flow } (\varphi \equiv 0): \ \mathbb{E}_{p_\Theta}[\ell_i] = d_\theta \cdot \mathrm{KL}(\text{standard logistic} \,\|\, \text{standard normal}) = 0.5639\, d_\theta \ \text{nats/row}, \tag{P4.10}$$

14.66 nats/row at $d_\theta = 26$ and 5.64 on the bench (quadrature and a
200000-row Monte Carlo agree to the Monte Carlo error, 14.63 +- 0.02). The
second statement holds for a flow whose stages are all the identity, which
is the case at $\varphi = 0$ exactly and only approximately at the random
initialisation torch gives the conditioners' last layers [reasoning: the
output logits are of order one at initialisation, so the first splines are
perturbed identities]. It says that on a prior-faithful batch an untrained
flow starts about $0.56\,d_\theta$ nats above the floor -- the cost of
putting a normal where a logistic belongs -- and that `delta_hat` on a
bench bank, whose prior is not the uniform (plan eq. (7); `mc_prior_floor`,
`joint_diagnostics.py:63-70`), is measured against a Monte Carlo floor
rather than this exact zero. The box term of eq. (P4.7) is at most
$-1.386\, d_\theta$ and averages $-2\, d_\theta$ under the uniform prior
($-52$ at $d_\theta = 26$), which is why a loss printed in unconstrained
coordinates and one printed in box coordinates differ by tens of nats for
the same flow: $L$ here is the second.

**The sampler.** `sample_posterior` calls the estimator's `sample`, which
draws from `NormalizingFlow.sample` -- `rsample` under `no_grad` -- and
`rsample_posterior` reaches the same distribution object through
`estimator._embedding_net` and `estimator.net` to call `rsample` with the
graph kept (`joint_model.py:134-166`; `zuko_flow.py:141-156`;
`distributions.py:121-127`). Both evaluate

$$\theta^{(s)}_g = \mathcal{F}_\omega^{-1}\big(\zeta^{(s)} \mid z_g\big), \qquad \zeta^{(s)} \sim p_\zeta \ \text{i.i.d.}, \tag{P4.11}$$

the stages inverted in reverse order and the box map inverted last, so the
draws are in box coordinates and inside the open cube (the fact P3 carries).
Inverting one fully autoregressive stage needs $d_\theta$ sequential passes
of its conditioner (`transforms.py:995-998`; the review's eq. (40) and
"about $D$ times more expensive" `[KB-PDF p.18-19]`), while evaluating
eq. (P4.7) needs one pass per stage: at the runner's defaults a batch of
log-probabilities costs 3 conditioner passes and a batch of draws 78, at the
space's upper bound 12 against 312 `[RAN]` B4. The passes are batched over
rows and draws, so the replicate term's $2 B_{\rm rep} S_{\rm mc} = 1024$
draws per step (P3 S3.2) are one batch through the same 78 passes.

```
theta in (0,1)^26 --F_box: logit--> R^26 --stage 1 (asc.)--> ... --stage n_tf--> zeta ~ N(0, I)
                        no weights        each stage: MaskedMLP(26+E -> n_hid x n_tf layers -> 26*(3K-1))
                                                      then 26 rational-quadratic splines on [-5, 5]
log_prob: left to right, one conditioner pass per stage.   sample: right to left, 26 passes per stage.
```

*Plain.* The flow takes a point of the box, stretches the box out to the
whole space with a logit, bends each coordinate in turn with a smooth
monotone curve whose shape a small network reads off the embedding and the
coordinates already bent, and asks how close the result is to Gaussian
noise. Training makes those curves put the simulated parameters where the
noise is dense; sampling runs the same curves backwards, one coordinate at a
time.

### 3.3 The two searched axes

*Establishes, per axis, the fields of plan S2.2: where it is set, what it
changes in eq. (P4.1)-(P4.11), what the range covers and how it is sampled,
how it fails and what reveals the failure.*

#### 3.3.1 `hidden_features` ($n_{\rm hid}$)

- **Set on:** the space, resolved by the width rule from the shape anchors
  `--p` and `--embedding-dim` (`joint_space.py:299-302`, copied from
  `npe_tune_search.py:131-134`),

  $$n_{\rm hid} \in \big[\min(\max(32,\, 2 \max(p, E)),\, 64),\ \max(8 \max(p, E),\, 256)\big], \tag{P4.12}$$

  $[52, 256]$ at $(26, 12)$ and $[32, 256]$ on the bench `[RAN]` B1; sampled
  log-uniform as an integer (`space_dimensions`, `:460-462`); reaches the
  runner as `--hidden-features`; runner default 48; library default 64;
  not passed by the job (S3.1).
- **Changes:** the width of every hidden layer of every conditioner,
  eq. (P4.5), hence the second-order term of eq. (P4.9): across the range
  the per-stage count goes from 50258 to 927474 at $n_{\rm tf} = 4$ against
  12 (Table P4.1), and the hidden-to-hidden layers dominate from the middle
  of the range up (at 256 and 12 stages they are 724 thousand of the 927
  thousand weights of a stage). It does not change the output width, the
  spline family, or the number of passes.
- **Range and prior:** the rule's lower branch is $2 \max(p, E)$ clamped to
  $[32, 64]$ and its upper branch $\max(8 \max(p, E), 256)$, so the range
  always contains $[64, 256]$ -- the standalone tuner's docstring says
  "widened so that it always CONTAINS" -- and is wider below 64 whenever
  $\max(p, E) < 32$. The string both spaces record in their ledgers,
  `clamp([2,8] * max(p,E), [64,256])`, describes a clamp *into*
  $[64, 256]$, a different function that would give $[64, 208]$ at
  $\max(p, E) = 26$ and agrees with the code at one value of $\max(p, E)$
  in 128 (F-ad, `[RAN]` B1). Under the log-uniform prior 13% of the
  proposals' mass lies below 64 and 57% below 128 on $[52, 256]$ (33% and
  67% on the bench's $[32, 256]$) `[RAN]` B1; the runner's 48 lies below
  the range at the DUP15HD shapes (F-r).
- **Interacts with:** $n_{\rm tf}$ multiplicatively in the cost and
  quadratically in the count (F-ac: a wider conditioner is also a deeper
  one when `num_transforms` is high); with $E$ through the input width
  $d_\theta + E$, a small term; with the optimiser (P5) through the number
  of weights AdamW holds state for and decays; with the search's boundary
  rule (P6): a best configuration at 52 or 256 is reported as an edge
  (`boundary_axes`, `:768-811`), and at 52 the next step down is forbidden
  by the clamp unless `--p` or `--embedding-dim` is lowered.
- **Fails as:** too narrow, the conditioner cannot carry the dependence of
  the spline parameters on $z$ and the earlier axes and the posterior stays
  prior-like along the directions it cannot resolve; too wide, more weights
  than the simulated rows constrain, with early stopping on `val_npe` the
  only guard (P5; `patience=99` makes it inoperative at the runner's
  epochs, F-d). According to PubMed, in an NPE study with a masked
  autoregressive flow the posterior-bias summaries changed little across
  the flow sizes tried while the training time did, which the authors read
  as diminishing returns once the training set is adequate and the flow
  sufficiently expressive, the residual error being set by the simulated
  data rather than by the flow (Fan & White 2026, Stat Comput,
  [DOI](https://doi.org/10.1007/s11222-026-10896-8)) [PubMed full text,
  PMC13180768] -- a statement about their ERGM problem, not about this
  bank, and no number from it is used here.
- **Revealed by:** `L` against `L0` and the gap between `val_npe` and the
  training `npe` in the history; the ledger's partial dependence on the
  axis (P6); the per-axis contraction and `p_eff` of the record, which say
  along how many directions the posterior is narrower than the prior (E7).

#### 3.3.2 `num_transforms` ($n_{\rm tf}$)

- **Set on:** the space, $[4, 12]$ (`JointSpaceSpec`, `:236`; the standalone
  tuner's constant range, `npe_tune_search.py:142`), sampled uniform as an
  integer (`:463-464`); reaches the runner as `--num-transforms`; runner
  default 3; library default 5; sbi's and zuko's defaults 5 and 3; not
  passed by the job.
- **Changes two things at once.** The number of stages in eq. (P4.2), each
  a full autoregressive pass over the $d_\theta$ axes in alternating order
  -- the usual meaning, and the one the plan and every docstring intend --
  and, because sbi hands zuko the list `[hidden_features] * num_transforms`
  (`flow.py:1142-1143`), **the number of hidden layers of every
  conditioner**, eq. (P4.5): at 4 stages each conditioner is 4 hidden layers
  deep, at 12 stages 12. The chain's depth in linear layers is
  $n_{\rm tf}(n_{\rm tf} + 1)$: 20 at the lower bound, 156 at the upper
  `[RAN]` B3, every one of them a plain ReLU layer without normalisation or
  shortcut. This is F-ac. It is not a defect of the joint stack and it is
  consistent with sbi's docstring, which calls `hidden_features` "the number
  of hidden features in the flow" and says nothing about depth; it is a
  property of the library the space searches through, and it means the
  axis is not the "more stages, same conditioner" knob that the review's
  account of composition describes `[KB-PDF p.21]`, nor a depth axis in
  the sense the encoder's `depth_exponent` is one (P1).
- **Range:** 4 to 12. At $d_\theta = 26$ with two alternating orders, 4
  stages give each axis two "ascending" and two "descending" contexts; the
  review's remark that composing coupling layers needs permutations so
  that every dimension gets transformed and interacts `[KB-PDF p.21]`
  applies with less force to fully autoregressive stages, where every axis
  is conditioned on every earlier one in each stage, and the alternation
  is what lets the last axis of one order be the first of the next. The
  upper bound buys $12 \times 26 = 312$ conditioner passes per sample
  `[RAN]` B4 and, through F-ac, conditioners twelve layers deep.
- **Interacts with:** $n_{\rm hid}$ (above); $S_{\rm mc}$ of P3 and the
  diagnostics' draw counts, linearly in the sampling cost; the optimiser
  (P5): depth without normalisation is the regime in which plain networks
  are hard to optimise [reasoning; the standard remedy is the residual
  connection, which `MaskedMLP` offers (`residual=True`, `nn.py:257`) and
  sbi does not expose]; the ledger (P6): two trials that differ only in
  `num_transforms` differ in conditioner depth too, so the partial
  dependence on this axis conflates the two.
- **Fails as:** too few stages, a posterior whose shape needs more than the
  composition of a few monotone warps per axis (strong curvature or
  multimodality in $\theta$) is smoothed; too many, a deep plain network
  per stage that trains slowly or not at all within the runner's 250 steps
  (P5), which the search would read as "deep flows are worse" when the
  cause is optimisation.
- **Revealed by:** the history's `npe` curve against $0.56\, d_\theta$
  (eq. (P4.10)): a run that stays near the identity-initialised value is
  not learning; the ledger's partial dependence (P6); `L` at equal
  $n_{\rm hid}$ across $n_{\rm tf}$.

### 3.4 The fixed knobs

*Establishes what the joint stack holds fixed around the two axes, where,
and what each fixed value commits the flow to.*

| knob | value | where | what it commits to | note |
|---|---|---|---|---|
| `num_bins` $K_{\rm bins}$ | 10 declared, **8 trained** | `spec.fixed` (`joint_space.py:324`); runner `--num-bins 8` (`:229`); `build_joint_model` 10 | the number of segments of every spline and the output width $d_\theta n_\varphi$: 598 against 754 per stage; 167960 against 201032 weights at the space's lower corner `[RAN]` B6 | F-a (patch, D-038); the standalone tuner searches it in $[6, 16]$ (S3.5); plan S5.1 fixes it without stating the value |
| `z_score_theta` | `"transform_to_unconstrained"` | `build_joint_model` default, no flag (`joint_model.py:197`); the standalone `NPEConfig` makes the same choice | eq. (P4.3): the flow works on $\mathrm{logit}\,\theta$, draws stay inside the open box, the density and $L$ are in box coordinates, and the prior must be passed (`x_dist`) or the build raises (`:213-217`; `flow.py:1298-1302`) | the choice the `npe_model.py` docstring argues for ("the flow cannot place posterior mass outside the physical bounds; plain z-scoring leaves the tails free to leak out of the box", `:16-20`); `[KB-PDF]` the same device -- log or logit transforms of bounded parameters, back-transformed in closed form -- in Goncalves et al. 2020, p.21 of the PDF |
| the prior as `x_dist` | `BoxUniform(0, 1)^{d_\theta}` | `run_joint_arms.py:385-388`; `_rebuild_model` builds the same (`run_stage3c.py:347`) | the bounds $a_k, b_k$ of eq. (P4.3): every bank built so far stores $\theta$ on the unit cube (E0 convention 3), so the map is the plain logit; a bank with other bounds would get the affine-then-logit form and no code change | `analytic`; not stored in the checkpoint (rebuilt from the prior at load: `CallableTransform`) |
| `z_score_x` | `"none"` | `build_joint_model` default (`:198`); plan D6 recommended "none" against sbi's `"structured"`; the plan's S8 still lists D6 as **open** (F-af) | no standardising net before the backbone: the window reaches $h_\psi$ as the bank stores it, the backbone's stem and GroupNorm handle scale (P1), and `estimator._embedding_net` **is** the backbone -- which is what lets `JointDSNNPE.encoder`, `freeze_encoder`, `checkpoint()["encoder_state"]` and the Stage 3b `rebuild_encoder` address the backbone alone (`joint_model.py:111-117, 170-181`; `run_stage3b.py:66-90`) | under `"independent"` sbi would wrap the backbone in `nn.Sequential(Standardize, backbone)` (`flow.py:1409-1411`) with constants estimated from the 64 example rows, the `encoder_state` keys would shift, and the encoder-alone loaders would break [reasoning from the code]; the standalone stack uses `"independent"` on its 12-dimensional embedding input, a different object (S3.5) |
| the 64 example rows | `theta_tr[:64], x_tr[:64]` | `run_joint_arms.py:463`; `run_stage3c.py:348-349` | shapes only: $d_\theta$ from $\theta$, $E$ by one backbone pass on $x$ (`flow.py:1136-1137`) | inert under this stack's two standardisation choices; a switch of `z_score_x` would make them the standardisation sample |
| `bound` $B_{\rm rqs}$ | 5 | zuko default (`transforms.py:474`); sbi strips `tail_bound` before calling zuko (`flow.py:34, 1140`) and passes no `bound` | the spline acts on $[-5, 5]$ of each stage's input and is the identity outside; for the first stage that is $\theta^{(k)} \in [0.00669, 0.99331]$; under the uniform prior 1.3% of each axis's mass starts in the tails and 30% of rows have at least one of 26 axes there (13% of 10) `[RAN]` B2 | unreachable from any surface of this stack; other implementations choose it per problem (an arXiv preprint in the knowledge base uses spans of 7 and 3 for two tasks, `[KB-PDF]` Hikida et al., p.26, PREPRINT, not peer-reviewed) |
| `slope` $\delta_{\rm rqs}$ | $10^{-3}$ | zuko NSF default (`spline.py:53`) | eq. (P4.6): interior slopes in $[10^{-3}, 10^{3}]$, bin ratios at most 1000, narrowest bin 0.0011 of the domain at 10 bins `[RAN]` B2: the sharpest feature one stage can carve | unreachable |
| `passes` | fully autoregressive ($d_\theta$) | zuko default `passes=None` (`autoregressive.py:110-111`); sbi passes nothing | every axis conditioned on all earlier ones per stage; $d_\theta$ passes per stage to sample, eq. (P4.11) | coupling (`passes=2`) would make sampling as cheap as density evaluation at the cost of expressiveness per stage `[KB-PDF p.19-21]`; unreachable |
| the orders | ascending, descending, alternating | `autoregressive.py:294-307`, `randperm=False` | two fixed orders; which axis is "first" is the bank's column order, i.e. the label-axes contract | unreachable; `randperm` is a sbi `ConditionalFlowConfig` field (`estimator_configs.py:121`) that `build_zuko_nsf` would forward, and `build_joint_model` does not set it |
| activation, `residual` | `ReLU`, `False` | zuko `MaskedMLP` defaults (`nn.py:255-258`) | plain conditioners; the depth of F-ac is unmitigated | unreachable from sbi's `posterior_nn` |
| the base $p_\zeta$ | standard normal, $d_\theta$ i.i.d. | `autoregressive.py:309-313`, buffers | eq. (P4.1), (P4.10) | no weights; the identity-initialised loss of eq. (P4.10) is the price of a normal base behind a logistic prior |
| the width rule's constants | 32, 64, 2, 8, 256 | `joint_space.py:300-302` | eq. (P4.12) | copied, not derived; the recorded string misdescribes them (F-ad) |

### 3.5 What is inherited from the standalone NPE tuner, and what differs

*Establishes that the flow block, unlike the encoder and loss blocks, is
copied from the other stack of the repository -- the standalone NPE tuner
and its model builder -- and sets the two side by side so a reader of one
can translate to the other.* The standalone stack is out of scope (D-037)
and documented by `hpc/docs/npe_tuning_usage.md` and the docstrings of
`npe_tune_search.py` and `npe_model.py`; what follows is read from those
files at `834eb41`.

| knob | joint space / runner | standalone tuner (`npe_tune_search.default_space`, `KNOB_ORDER`) | standalone model (`npe_model.NPEConfig`) |
|---|---|---|---|
| `hidden_features` | width rule, eq. (P4.12); log-uniform integer; runner 48 | the same rule, the same prior (`:131-134, 157-158`); the provenance string `RANGE_PROVENANCE` names it (`joint_space.py:139`) | 128 |
| `num_transforms` | $[4, 12]$ uniform; runner 3 | $[4, 12]$ uniform (`:142, 159-160`) | 8 |
| `num_bins` | **fixed** 10 (built at 8, F-a) | **searched**, $[6, 16]$ uniform integer (`:143, 161`) | 10 |
| `learning_rate` / `lr` | $[10^{-4}, 2 \cdot 10^{-3}]$ log-uniform (P5) | the same (`:144, 162-163`) | $5 \cdot 10^{-4}$ |
| `batch_size_npe` / `training_batch_size` | $\{256, 512, 1024\}$, trimmed by `n_train` (P5) | the same (`:136-138, 164`) | 512 |
| `weight_decay`, `one_minus_beta1` | searched (P5) | **absent**: sbi's preconfigured loop does not expose them (`npe_tune_search.py:33-36`) | -- |
| `z_score_theta` | `transform_to_unconstrained` | -- | `transform_to_unconstrained` |
| `z_score_x` | `"none"` (the condition is a raw window; the backbone is inside the estimator) | -- | `"independent"` (the condition is a frozen embedding, $E = 10$ for the r2 export (`[KB]` plan S2.1), standardised per component, `npe_model.py:22-25, 83`) |
| the training loop | the joint loop of P5 (`joint_train.py`) | sbi's own, through `NPEConfig` (`stop_after_epochs=20`, `max_num_epochs=500`, `clip_max_norm=5.0`) | the same |
| the condition | $x$, one window, through $h_\psi$ trained jointly | $z$, a frozen embedding from the DSN export | the same |

Where the two differ in kind rather than in value:

- **`num_bins` is a searched axis there and a fixed one here.** Plan S5.1
  fixed it without a value; `default_joint_space` chose 10 (the standalone
  model's default) and the runner 8 (sbi's low-level default). The
  standalone search's range $[6, 16]$ shows the authors of that stack
  regarded it as a knob worth searching; the joint space inherits the
  width and depth ranges but not this one.
- **The condition is a window here and an embedding there.** The joint flow's
  conditioner reads $z = h_\psi(x)$ computed inside the same forward pass,
  with gradients flowing back into $\psi$ (plan S2.2: the NPE term reaches
  both $\psi$ and $\omega$); the standalone flow's conditioner reads a
  stored, frozen $z$. That is why `z_score_x` differs: standardising a
  stored embedding is cheap and harmless, standardising a raw window in
  front of a backbone with its own normalisation would put a batch-fitted
  affine map before the stem the DSN was trained with [reasoning; plan D6
  states the options without a rationale, and S3.4 gives the checkpoint
  argument].
- **The optimiser axes exist only here**, because only the joint stack
  owns its loop (P5).

Pointers: `hpc/docs/npe_tuning_usage.md` for operating the standalone
tuner; `npe_tune_search.py:1-40` for its design decisions; `npe_model.py:
1-45` for its flow choices; usage v1.3 S7 for the joint tuner's
subcommands.

### 3.6 Interactions with the other blocks

*Establishes, per block, which knob of this document changes what the other
block's knobs do.*

- **The encoder (P1).** $E$ is the conditioner's context width; the
  embedding is a unit vector ($S^{E-1}$), so its components are bounded
  and of order $E^{-1/2}$ each, which is the scale the conditioner's first
  layer sees without any standardisation. The NPE gradient reaches $\psi$
  only through $z$'s entry into every stage's conditioner (eq. (P4.4)); a
  collapsed embedding cloud ($r_{\rm eff} \to 1$, E3) gives every row the
  same conditioner and the flow can only learn the prior. The encoder axes
  change nothing in the flow's shapes except $E$; the Stage 3b/3c rebuilds
  restore $E$ and $W$ from the checkpoint but not the other encoder axes
  (F-ae).
- **The DSN term (P2).** No direct interaction: the metric loss never
  evaluates the flow. Indirectly, through $\psi$: an embedding shaped for
  class separation is the conditioner the flow must read.
- **The replicate term (P3).** Its draws are eq. (P4.11) through
  `rsample_posterior`: $n_{\rm tf} d_\theta$ conditioner passes per batch of
  $2 B_{\rm rep} S_{\rm mc}$ draws, with the gradient kept through the
  inverse splines and the box map's inverse (the sigmoid) into $z$ and
  hence $\psi$; the stop-gradient on $\omega$ (P3 S3.2) leaves the flow's
  weights out of that term. $\bar C$ and the target of P3 are moments of
  the flow's posterior in box coordinates; a flow that cannot sharpen
  beyond its spline resolution (eq. (P4.6)) sets a floor on how small
  $\bar C$ can be.
- **The optimiser (P5).** AdamW holds state for all $n_\omega$ flow weights
  and applies $\gamma_{\rm wd}$ to all of them, masked entries included
  (they shrink toward zero and never act); `grad_clip` at 5 is a norm over
  encoder and flow gradients together; `lr` is shared. The identity
  initialisation of eq. (P4.10) means the first steps' loss is about
  $0.56\, d_\theta$ above the floor whatever the optimiser does.
- **The search (P6).** Both axes are always active (no switch pins them),
  so canonicalisation never touches them; `hidden_features` is the one
  integer axis with a log-uniform prior; `boundary_axes` reports either at
  its edge; the width rule makes the lower edge move with `--p` and
  `--embedding-dim`, which must match the bank (`[KB]` usage v1.3 S7).
  F-a makes the ledger's `num_bins` differ from the trained one for every
  trial of every campaign.
- **The diagnostics (E7, P7).** The information spectrum and the
  contraction use `n_posterior_draws` draws of eq. (P4.11) on up to 256
  report rows (`run_joint_arms.py:561`); their cost scales with
  $n_{\rm tf} d_\theta$. Stage 3c's `_rebuild_model` is the one consumer
  that rebuilds the flow from a checkpoint (`run_stage3c.py:364-371`).
- **The bench (E6).** On the bench $d_\theta = 10$, $p = 10$, $E = 10$: the
  width rule gives $[32, 256]$, the runner's 48 is inside, the flow at the
  runner's defaults is 50946 weights, and 13% of prior rows start with an
  axis in the spline's tails `[RAN]`.

### 3.7 Failure modes and the diagnostics that reveal them

*Establishes, in one table, how the flow block fails, which knob is
implicated, and which number in the run record shows it.*

| failure | mechanism | knob(s) | what blocks it, as built | revealed by |
|---|---|---|---|---|
| prior-like posterior | the conditioner cannot carry $z$'s information into the spline parameters, or the embedding carries none | $n_{\rm hid}$, $n_{\rm tf}$; the encoder | nothing in the flow; early stopping cannot fire at the runner's epochs (F-d) | `delta_hat` near 0; `p_eff` near 0; contraction near 1 on every axis; the history's `npe` near $0.56\, d_\theta$ |
| **F-ac, depth coupling** | raising `num_transforms` deepens every conditioner; at 12 the chain is 156 plain ReLU layers deep | $n_{\rm tf}$ | nothing (`residual` not exposed) | `npe` not falling within the budget at high `num_transforms`; the partial dependence on the axis turning down at its upper end |
| **F-a, ledger and trained flow disagree** | `build_argv` passes no `--num-bins`; the runner builds 8 bins | $K_{\rm bins}$ | nothing | `argv` in the trial record without `--num-bins`; `config.num_bins` 8 in the runner's record; `meta["flow"]` in the checkpoint |
| tails at the box edges | for $\theta^{(k)}$ outside $[0.0067, 0.9933]$ the first stage is the identity, and the later stages see whatever the earlier ones produced: the flow's resolution near the prior's edges is the base's, not the spline's | $B_{\rm rqs}$ (unreachable) | nothing; 1.3% of each axis's prior mass per axis at initialisation `[RAN]` | per-axis coverage at the edges (gate G3, E7); not visible in `L` |
| resolution floor | no stage can carve a feature narrower than 0.0011 of its domain or steeper than $10^{3}$; a posterior sharper than that in $\vartheta$ needs the composition to supply it | $K_{\rm bins}$, $\delta_{\rm rqs}$ | nothing | `L` saturating while `p_eff` is high; the replicate term's $\bar C$ not shrinking |
| over-dispersed posteriors | the objective is the forward KL (mass-covering): the fit is penalised more for missing mass than for spreading it | all of the block | nothing | According to PubMed, a well-centred but over-dispersed single-round NPE posterior, read by the authors as the mass-covering behaviour of the objective and improvable by more expressive flows or sequential rounds (Fan & White 2026, [DOI](https://doi.org/10.1007/s11222-026-10896-8)) [PubMed full text, PMC13180768]; here: coverage above nominal at G3, `L` above a sharper competitor's |
| **F-ad, misrecorded width rule** | the ledger's `anchored_to.width_rule` string describes a clamp into $[64, 256]$; the code computes eq. (P4.12) | -- | nothing | reading the string instead of the range: a reader expects a lower bound of 64 where the space searched from 52 (or 32 on the bench) |
| **F-ae, rebuild from a checkpoint** | `_rebuild_model` and `rebuild_encoder` restore the flow's three knobs and the encoder's $W$, $E$ from `meta` but build the backbone with the runner's default encoder axes | the encoder axes (P1), `backbone_repr` unread | the size mismatch raises on load for a depth, width, family or head change; a `dropout` change loads silently (no weights) and is inert in `eval()` | a `RuntimeError` from `load_state_dict` on a Stage 3b/3c run against a Stage 4 finalist trained off the defaults |
| **F-af, D6 open on paper** | the plan's S8 lists D6 as open; the code took the recommended option and the decisions log has no entry | `z_score_x` | -- | a reader of the plan expects a decision pending; a switch to `"structured"` would break the checkpoint's encoder contract (S3.4) |
| the wrong shapes for the bank | `--p`, `--embedding-dim`, `--d-theta` not matching the bank | the search anchors (P6) | nothing at propose time | a `hidden_features` range and a draws floor for the wrong problem (`[KB]` usage v1.3 S7) |

### 3.8 Findings this document owns

The table of record is `00_INDEX.md` S6; the rows below are this document's,
with what P4 adds to each.

- **F-a (owned with P6; patch, D-038).** The runner's `--num-bins` default
  is 8 (`run_joint_arms.py:229`), `build_joint_model`'s 10 (`joint_model.py:
  196`), the space declares 10 fixed (`joint_space.py:324`), and
  `build_argv` passes `spec.fixed` values for `strict_semihard` only
  (`npe_tune_joint.py:206-211`). What P4 adds: the size of the difference
  (598 against 754 outputs per stage; 167960 against 201032 weights at the
  space's lower corner `[RAN]` B6), the three places the trained value is
  recorded (S3.1 reading 2), and the observation that the standalone tuner
  searches this knob (S3.5). The patch the decision names -- pass
  `--num-bins` from `spec.fixed` as `--strict-semihard` is passed -- is
  the same shape as F-w's.
- **F-j (owned with P6).** The resolved ranges: `hidden_features` $[52, 256]$
  at $(26, 12, 26)$ and $[32, 256]$ on the bench, from eq. (P4.12) `[RAN]`
  B1.
- **F-m (flow rows).** 64/5/10 (library) against 48/3/8 (runner): with
  Table P4.1, 340730 against 107634 weights. P4 adds two further default
  layers below the library's, sbi's 50/5/10 (`posterior_nn`) and 50/5/8
  (`build_zuko_nsf`), and zuko's own 3 transforms with two hidden layers of
  64 (`[RAN]` B6) -- five defaults for three knobs before the space's.
- **F-r (flow rows).** 48 below 52 and 3 below 4 at the DUP15HD shapes; on
  the bench the width is inside $[32, 256]$ and the transform count still
  outside (S3.1 reading 4).
- **F-ac (new).** `sbi` 0.27.0 passes `hidden_features` to zuko as
  `[hidden_features] * num_transforms` (`flow.py:1142-1143`), and zuko's
  `MaskedMLP` reads that list as its hidden-layer widths (`nn.py:255,
  276-293`), so `num_transforms` sets both the number of stacked transforms
  and the number of hidden layers of every conditioner: at the space's
  upper bound each conditioner is 12 plain ReLU layers deep and the chain
  156 linear layers, with no residual connection or normalisation
  (`residual=False`). Verified by replicating the mask construction and the
  shapes `[RAN]` B3 (four `MaskedLinear` layers at the runner's 3
  transforms). Consequences: the weight count is quadratic in
  `num_transforms` at fixed width (eq. (P4.9)); the search's partial
  dependence on `num_transforms` conflates stages and depth; the plan's
  axis inventory (S5.1) and `JointSpaceSpec`'s docstring describe the
  axis as a transform count only. `[REPO sbi wheel]`, `[REPO zuko wheel]`
  as cited; the optimisation consequence is [reasoning]. Status: report;
  open (whether to pass an explicit `hidden_features` list of fixed depth
  -- sbi accepts a sequence -- is a D-038-style change to
  `build_joint_model`).
- **F-ad (new).** Both `default_joint_space` and `default_space` record
  `"width_rule": "clamp([2,8] * max(p,E), [64,256])"` in `anchored_to`
  (`joint_space.py:328`; `npe_tune_search.py:148`), which as a function
  disagrees with the code's eq. (P4.12) at 127 of the 128 values
  $\max(p, E) = 1..128$ `[RAN]` B1 -- at the DUP15HD shapes the string
  reads $[64, 208]$ and the code gives $[52, 256]$. P0 S3.2 already states
  the coded rule; the ledger's string does not. The resolved bounds are
  recorded beside it (`hidden_features`), so nothing trains wrong; a reader
  of `anchored_to` alone is misled. `[REPO]` as cited. Status: report; open
  (fix the string, or drop it in favour of the recorded bounds; the F-c
  kind of repair).
- **F-ae (new; owner P7 with P1).** `stage3c._rebuild_model`
  (`run_stage3c.py:336-373`) and `stage3b.rebuild_encoder`
  (`run_stage3b.py:66-90`) rebuild the backbone with `make_backbone(W, E,
  dsn_main_dir, seed)` and no `encoder` dict, so the five encoder axes take
  the runner's defaults (3, 2.0, 0, 0, 0.0; `run_joint_arms.py:301-306`)
  whatever the checkpoint was trained at; the flow's three knobs are
  restored from `meta["flow"]` with a refusal when absent (`:359-363`), and
  the encoder's configuration, present in `meta["backbone_repr"]`
  (`joint_model.py:227`), is read by nothing. A checkpoint of a Stage 4
  trial off the encoder defaults fails to load with a size mismatch for
  `depth_exponent`, `width_multiplier`, `block_family` or `head_fusion`
  (loud); one off only in `dropout` loads silently and is inert in
  `eval()`. Found while reading the flow's rebuild; the encoder side is
  P1's object and the probes are P7's. `[REPO]` as cited; the failure
  modes are [reasoning from the shapes of P1]. Status: report; open (D-038
  candidate: store the encoder dict in `meta` and read it back, as the
  flow's is).
- **F-af (new).** Plan v0.6.5 S8 lists D6 ("Conditioner normalisation.
  `z_score_x = "none"` (recommended) or sbi's `"structured"`?") among the
  open decisions, and D7 with it; `build_joint_model` has taken the
  recommended option as its default since Stage 2 (`joint_model.py:198`),
  the Stage 0 contract names it (`posterior_nn(..., z_score_x="none")`,
  plan Stage 0), and the decisions log carries no D-number for it. S3.4
  shows the choice is load-bearing for the checkpoint's encoder contract,
  not a cosmetic default. `[REPO]` plan S8 `:1571-1574`, Stage 0 `:1146-1147`;
  `[KB]` the log v1.16. Status: report; open (a status line in the plan's
  S8, or a D-number in the log).
- **The P0 line reference.** P0 S3.2 cites the width rule at
  `joint_space.py:297-300`; the four lines are `:299-302` at `834eb41`.
  Noted for the Stage 6 fidelity review, not edited here (one document per
  turn).

## 4. Summary of results

1. **Eight surfaces, two axes, one fixed knob, two choices by code** (S3.1):
   the job passes no flow flag; the tuner passes two of three; the
   checkpoint's `meta["flow"]` is the record of what was built; the Stage 3
   defaults sit outside the Stage 4 space in width and transform count.
2. **The flow as built**, eq. (P4.1)-(P4.8): a logit box map without weights,
   $n_{\rm tf}$ fully autoregressive stages in alternating order, each a
   masked ReLU MLP of $n_{\rm tf}$ hidden layers of $n_{\rm hid}$ producing
   $3 K_{\rm bins} - 1$ soft-clipped spline parameters per axis, a standard
   normal base, and a loss in box-coordinate nats (S3.2).
3. **The weight count**, eq. (P4.9): 107634 at the runner's defaults, 201032
   to 11129688 across the space at $(26, 12, 26)$, 52-59% of it live; no
   surface records it (S3.2).
4. **The anchor**, eq. (P4.10): $L_0 = 0$ on the unit cube; an
   identity-initialised flow starts at $0.564\, d_\theta$ nats/row (14.7 at
   26, 5.6 on the bench) (S3.2).
5. **Sampling costs $d_\theta$ passes per stage** against one for the
   density: 78 against 3 at the runner's defaults, 312 against 12 at the
   space's upper bound (S3.2).
6. **`num_transforms` is two knobs**, F-ac: stages and conditioner depth,
   through sbi's `[hidden_features] * num_transforms` (S3.3.2).
7. **The width rule is eq. (P4.12)**, not the clamp the ledger string
   describes (F-ad); 13% of the log-uniform mass lies below 64 at the DUP15HD
   shapes (S3.3.1).
8. **The fixed knobs** commit the flow to 8 bins in training against 10 on
   paper (F-a), to box coordinates with the prior rebuilt at load, to a raw
   window at the backbone (D6, open on paper: F-af), and to a $[-5, 5]$
   spline domain that leaves 1.3% of each axis's prior mass in the identity
   tails at initialisation (S3.4).
9. **The block is copied from the standalone NPE tuner** with one axis
   dropped (`num_bins`), two added (the optimiser's), and `z_score_x` changed
   because the condition changed from a stored embedding to a raw window
   (S3.5).
10. **Rebuilding from a checkpoint restores the flow and not the encoder's
    axes** (F-ae) (S3.8).

## 5. Open points, caveats, assumptions

**Assumed without proof.**

- The identity-initialised loss of eq. (P4.10) is exact for $\varphi \equiv 0$;
  that the random initialisation is close to it is reasoning from the
  initialisation bounds, and no run has logged a first-step loss.
- The 1.3% tail mass per axis (S3.4) is the uniform prior's at the first
  stage; after training the later stages move mass, and the effective
  tail fraction of a trained flow is not computed here.
- That depth without normalisation or shortcuts is hard to optimise (F-ac)
  is the standard account `[textbook, from memory]`; whether it binds within
  the runner's 250 steps at 12 transforms is unmeasured.
- The live-weight share assumes `torch.unique` sorts rows as `numpy.unique`
  does, which the replication relies on for the cyclic assignment of hidden
  units to dependency classes; the nominal count does not depend on it.

**Approximations and their regime.**

- The log-uniform masses of S3.3.1 treat skopt's log-uniform integer as
  continuous in the logarithm; the rounding at the two ends shifts them by
  less than one integer's share.
- Table P4.1 counts weights, not operations; the masked matrix products are
  computed in full, so the cost scales with the nominal count.

**Left unresolved.**

- F-a's patch (D-038) and its open part, whether the runner's own default
  moves from 8 to 10.
- F-ac: whether `build_joint_model` should pass an explicit list of hidden
  widths so that depth and stage count are two knobs; whether `residual`
  should be exposed.
- F-ad's string; F-ae's rebuild; F-af's status line or D-number.
- Plan D6 and D7 (ensemble semantics) remain open in the plan's S8; D7 has
  no code yet in `hpc/joint/`.
- Whether `num_bins` should be searched, as the standalone tuner searches
  it; plan S5.1 fixed it without stating why.
- Whether the spline bound should move with the prior's edges for banks
  whose posteriors concentrate near a bound (the DUP15HD axes on the
  coordinate threshold, D-004, are the candidates); unreachable through sbi
  0.27.0's `posterior_nn` without a custom builder.
- The flows review's statement that the composition of coupling layers
  needs permutations `[KB-PDF p.21]` was read for the fully autoregressive
  case here by reasoning; no source on the two-order alternation was found
  (S6).

## 6. References / further reading

**Project knowledge base `[KB]`.** `JOINT_DSN_NPE_USAGE_v1.md` v1.3 S3, S5,
S7, S9; `HPC_PATHS.md` sec. 7 (the environment's versions);
`claude/joint_docs/00_INDEX.md` S6 (F-a, F-j, F-m, F-r) and S8;
`claude/SBI_decisions_and_ideas_log.md` v1.16 (D-035 to D-038, D-052, D-054;
no entry for plan D6). `[KB-PDF]`: Papamakarios G, Nalisnick E, Rezende DJ,
Mohamed S, Lakshminarayanan B. *Normalizing flows for probabilistic modeling
and inference.* JMLR 2021 (the project PDF): p.3 (eq. (1)-(6), change of
variables and composition), p.6 (eq. (13)-(14), forward KL and maximum
likelihood), p.12-13 (eq. (29)-(32), autoregressive flows and their
triangular Jacobian), p.16-17 (spline transformers), p.18-19 (masked
conditioners, eq. (40), the $D$-fold inversion cost), p.19-21 (coupling
layers and permutations). Goncalves PJ et al. *Training deep neural density
estimators to identify mechanistic models of neural dynamics.* eLife 2020
(the project PDF): p.20 (five stacked MAF bijections with two to three
hidden layers of 50-100 units "enough to approximate even complex
posterior distributions" in their applications), p.21 (log and logit
transforms of bounded parameters, back-transformed in closed form).
Deistler M et al. *Simulation-based inference: a practical guide* (the
project PDF): p.17 (a neural spline flow with a 1D convolutional embedding
network reducing 2 x 8192 samples to 32 features), p.20 (a spline flow with
an ensemble of five for a many-parameter simulator). Radev ST et al.
*BayesFlow* (IEEE TNNLS 2022, the project PDF): p.5-6 (Proposition 2 and the
joint optimisation of summary and invertible networks). Hikida Y, Bharti A,
Jeffrey N, Briol F-X. *Multilevel neural simulation-based inference*
(arXiv 2506.06087v4, the project PDF; **PREPRINT, not peer-reviewed**):
p.26, spline spans of 7 and 3 chosen per task -- cited only to show the
bound is a per-problem choice elsewhere.

**Repository `[REPO]`** at `834eb41`, read 2026-10-02: the files and lines of
the changelog row. **The two wheels, read from PyPI** (`pip download
--no-deps`): `sbi` 0.27.0 and `zuko` 1.6.0, files and lines as cited in
S3.2; **torch 2.10.0** `torch/distributions/constraint_registry.py`
(`_transform_to_interval`, `_biject_to_independent`), read from GitHub at
tag `v2.10.0`.

**Peer-reviewed literature.** According to PubMed, read in full text from
PubMed Central: Fan Y, White SR. *Neural posterior estimation on exponential
random graph models: evaluating bias and implementation challenges.* Stat
Comput 2026; PMID 42158214, PMC13180768,
[DOI](https://doi.org/10.1007/s11222-026-10896-8). Used in S3.3.1 and S3.7
for two qualitative statements only: that in their cost study the
posterior-bias summaries changed little across flow sizes while training
time did, read by them as diminishing returns once the data are adequate and
the flow expressive enough; and that their single-round NPE posterior was
well centred but over-dispersed, which they attribute to the mass-covering
objective. **No numeric value from that paper is used**; their flow is a
masked autoregressive flow at sbi's defaults on a 3-parameter network
model, not a spline flow on this bank. Jang G, Candan KS, Chowell G. *A
comparative study of simulation-based inference methods for epidemic models
with identifiability considerations.* PLoS Comput Biol 2026; PMID 42228739,
PMC13252848, [DOI](https://doi.org/10.1371/journal.pcbi.1014364): read in
full text; records that on their smooth, unimodal posteriors a neural spline
flow performed comparably to or worse than a masked autoregressive flow, and
that their embedding network and flow were optimised jointly -- the regime
of arm `A1`; no number used. Wang Z, Hasenauer J, Schaelte Y. *Missing data
in amortized simulation-based neural posterior estimation.* PLoS Comput Biol
2024; PMID 38885265, PMC11213359,
[DOI](https://doi.org/10.1371/journal.pcbi.1012184): read in full text;
states the BayesFlow construction in which the summary network's parameters
are trained jointly with the invertible network's; no number used.

Retrieved, **abstract only, therefore not used for any claim**: Mahmoud AH
et al., J Chem Inf Model 2022, PMID 35352898,
[DOI](https://doi.org/10.1021/acs.jcim.1c01438) (rational-quadratic neural
splines for conformational sampling; no PMC full text); John R, Herron L,
Tiwary P, J Chem Phys 2025, PMID 40116311,
[DOI](https://doi.org/10.1063/5.0249683) (a comparison of neural spline
flows, flow matching and diffusion models on molecular data; no PMC full
text). Full text not accessible for PMID 40116311 -- this looks relevant to
the choice of the flow family for low-dimensional, mode-asymmetric targets;
if you can obtain the PDF, upload it and it can be folded into E2. Not
in the knowledge base and not in PubMed Central: Durkan C, Bekasov A, Murray
I, Papamakarios G. *Neural spline flows.* NeurIPS 2019 -- the source of the
rational-quadratic spline; every statement about the spline here is read
from zuko's implementation, not from that paper.

**Searches run `[RAN]`, 2026-10-02.**

| source | query | result |
|---|---|---|
| PubMed | neural spline flow | 24 records; 2 on topic, both abstract-only (PMIDs 35352898, 40116311), flagged above; the rest are unrelated uses of "spline" and "flow" |
| PubMed | normalizing flow neural posterior estimation simulation-based inference | 0 records |
| PubMed | rational quadratic spline normalizing flow | 1 record (PMID 35352898, abstract only) |
| PubMed | masked autoregressive flow density estimation | 0 records |
| PubMed | normalizing flows posterior Bayesian inference neural density estimator simulator | 2 records (PMIDs 42656156, 40929189; conformal calibration of SBI credible sets and variational inference with flows; neither in PMC, neither used) |
| PubMed | "normalizing flow" embedding network summary statistics jointly trained posterior | 0 records |
| PubMed | logit transform bounded parameters unconstrained space neural density estimation posterior | 0 records |
| PubMed | deep neural network depth degradation optimization difficulty residual connections plain network | 0 records |
| PubMed | vanishing gradient deep feedforward network depth residual learning | 0 records |
| PubMed | "neural posterior estimation" | 21 records; full texts read: PMC13252848, PMC13180768, PMC11213359 (used, non-numerically); the others are applications without flow-architecture content in their abstracts and were not opened |
| PubMed | normalizing flow number of transforms hidden units expressiveness posterior approximation hyperparameters | 0 records |
| PubMed | autoregressive flow sampling sequential inverse cost conditioner | 0 records |
| PubMed | input standardization z-scoring neural network training conditioning features normalization layer embedding | 0 records |
| PubMed | BayesFlow invertible neural network summary network amortized Bayesian inference | 1 record (PMID 33338021, the BayesFlow paper; cited from the project PDF instead) |
| bioRxiv | neuroscience, last 30 days, 30 records (the connector has no keyword search) | none on flows or NPE; no preprint is cited |
| bioRxiv | systems biology, last 30 days, 30 records | one neural likelihood-ratio preprint (10.64898/2026.05.06.722433), not on flows; not cited |

**Textbook, from memory** (tagged as such where used): that plain deep
feedforward networks without normalisation or shortcuts are harder to
optimise than residual ones (S3.3.2, S5); the entropy of the standard
logistic law and the standard normal's log-density, both inside the closed
form of eq. (P4.10), whose value is checked by quadrature `[RAN]`.

---

### Pre-send check (Precision model)

R1 -- every symbol typed in S1; relations connect compatible types (the
chain composes maps on $\mathbb{R}^{d_\theta}$ after $\mathcal{F}_{\rm box}$;
the spline acts on scalars; $n^{\rm live}_\omega < n_\omega$ are counts).
R2 -- eq. (P4.3)'s unit-cube form carries "on the unit cube"; eq. (P4.10)
carries "on the unit cube" and "$\varphi \equiv 0$"; the tail fractions
carry "under the uniform prior, at the first stage"; the PubMed statements
carry their domains and "no number used". R3 -- the review's inversion-cost
statement is applied to a masked autoregressive flow, the object it was
made for; the permutation remark is marked as applied by reasoning to a
different conditioner; the standalone tuner's `z_score_x` is not
transplanted to this stack's condition (S3.5). R4 -- "transform" has three
declared senses; $n_\omega$ and $n^{\rm live}_\omega$ are two counts; the
runner's, the space's and the libraries' defaults are named per surface;
$r$ the index and $r_{\rm eff}$ the rank are distinct symbols. R5 -- the
maps $\mathcal{F}_{\rm box} : \Theta \to \mathbb{R}^{d_\theta}$,
$\mathcal{F}_\omega : \Theta \to \mathbb{R}^{d_\theta}$,
$\mathcal{C}^{(r)}_\omega : \mathbb{R}^{d_\theta + E} \to \mathbb{R}^{d_\theta \times n_\varphi}$ and
$h_\psi : \mathbb{R}^{W} \to S^{E-1}$ are named with domain and codomain;
box and unconstrained coordinates are never mixed (the loss is stated in
box coordinates, with the conversion term shown). R6 -- "hidden features",
"transform", "bins", "order", "pass" declared in S1.1 or S2. R7 -- the sbi
docstring's "number of hidden features in the flow" is quoted with what it
omits; the npe_model docstring's argument for the logit map is quoted as
its authors' reasoning. R8 -- $\ell_i$ (one row), $L$ (a split) and
$\mathcal{L}^{\rm sim}_{\rm NPE}$ (the expectation) are three levels with
the moves named; the identity-initialised loss is marked analytic.
Confirmation questions: none in the brief.
