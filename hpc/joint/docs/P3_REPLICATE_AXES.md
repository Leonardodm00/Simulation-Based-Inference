# P3 -- The replicate block: the four searched axes, the loss constants and the draw floor

**Document P3 of the joint documentation set.** Owner of the `replicate`
block of `JOINT_KNOB_ORDER` (`rep_on`, `log10_lambda_rep`,
`warmup_frac_rep`, `n_posterior_draws`) and of the knobs of the replicate
term the joint stack holds fixed or never sets (`ReplicateConsistencyLoss`'s
`jitter`, `t_floor`, `p_eff_min`, `correct_mc`; `replicate_statistic`'s
`detach_metric`; the stop-gradient set; `Sigma0` from the unit box;
`--b-rep`; the pair enumeration; the diagnostic's pair and row caps). Master
notation: E0. The chapter that explains what the term is *for* and derives
its target: E5. **Date:** 2026-10-02 (v1). **Applies to:** the repository
`Simulation-Based-Inference` at `834eb41`, `hpc/joint/` (D-037):
`stage2/joint_losses.py`, `replicate_statistic.py`, `stage2/joint_batches.py`,
`stage2/joint_train.py`, `stage2/joint_model.py`, `stage3/run_joint_arms.py`,
`stage3/joint_diagnostics.py`, `stage3/jobs/joint_arms.pbs`,
`stage4/joint_space.py`, `stage4/npe_tune_joint.py`,
`stage1/build_latent_bank.py`; the plan `JOINT_DSN_NPE_PLAN_v0_6.md` at its
repository version **v0.6.5**; and, as the derivations of record for the
finite-draw correction and the $\kappa_S$ bias, the project documents
`claude/METRIC_REPLICATE_v1_4.md` and `claude/FINITE_DRAW_CORRECTION_v1.md`
`[KB]`, whose documentation patch (plan v0.6.6, three docstrings) is **not**
applied at `834eb41` (F-e). The flow's samplers were checked against the
`sbi` 0.27.0 wheel, the version `stage2/joint_model.py:157` names.

| date | change |
|---|---|
| 2026-10-02 | v1. Written from `stage2/joint_losses.py`, `replicate_statistic.py`, `stage2/joint_batches.py`, `stage2/joint_train.py` (all read in full), `stage2/joint_model.py:109-166, 196-216`, `stage3/run_joint_arms.py:170-262, 336-400, 455-533, 540-625`, `stage3/joint_diagnostics.py:237-272`, `stage3/jobs/joint_arms.pbs:49-117`, `stage4/joint_space.py:1-420, 440-560, 760-860`, `stage4/npe_tune_joint.py:80-262, 343-345`, `stage1/build_latent_bank.py:15-45, 160-262`, `stage1/jobs/build_latent_bank.pbs:44-46`, `stage2/smoke_test_joint_losses.py` (J10-J16), `smoke_test_replicate_statistic.py` (T1-T8), `stage2/smoke_test_joint.py:452-480` (J14r), `stage4/smoke_test_joint_space.py` (J24, J27); the installed library's source, `sbi` 0.27.0 (`neural_nets/net_builders/flow.py:1078-1171, 1278-1317`, `neural_nets/estimators/zuko_flow.py:141-156`), read from the wheel; the plan S2.5, S5.1, S8 (D12, D13, D15-D18), S9; `[KB]` `METRIC_REPLICATE_v1_4.md` S3.13-S3.15, S3.19, S5 and `FINITE_DRAW_CORRECTION_v1.md` S3.1-S3.11. Every number of S3.2-S3.4 recomputed by `tools/p3_numbers.py` `[RAN]`. Findings F-y to F-ab added; F-e, F-i, F-o, F-p owned. Grounding searches of S6 run and reported. |

**Abstract.** The four axes of the `replicate` block are the only axes of
the joint search that change *what real data teaches the encoder without a
label*: whether the replicate term is on at all and how heavily it weighs
against the likelihood term ($\lambda_{\rm rep}$), over what fraction of
training it is ramped in ($t_{\rm warm}$), and how many posterior draws per
well the statistic is computed from ($S_{\rm mc}$). The question this
document answers is, for each of the four and for the constants held fixed
around them: where the value is set and by which surface it reaches the
trainer (S3.1); what the term computes, written out from the code as one
explicit function of the knobs, from the reparameterised draws to the
ramped batch mean, eq. (P3.1)-(P3.12), with the two clamps and the three
gradient decisions made explicit (S3.2); what each axis changes in that
function and therefore in the joint objective (S3.3), including the three
readings of the $4 d_\theta$ draw floor -- the rank reading the code states,
which is false, the $\kappa_S$ reading the knowledge base established, and
the compute reading -- and the bias the subtraction of plan eq. (3h) leaves
behind, recomputed here `[RAN]`; how the knobs interact with the other
blocks (S3.6); how the term fails and which number in the run record reveals
each failure (S3.7); and the findings this document owns (S3.8): F-e, F-i,
F-o, F-p, and four new ones -- the clamp's gradient dead zone coincides with
the collapse signature (F-y), an undefined target becomes a pull toward
agreement (F-z), a real bank with no same-donor pair turns arm `A5` into
`A1` silently (F-aa), and the pair enumeration treats rows as wells, so that
on the bench 47% of the "replicate pairs" are two windows of one well
(F-ab). **Deliberately excluded:** the derivation of why the target is
$p_{\rm eff}$ and not zero (E5, with `[KB]` METRIC S3.7-S3.9), the
per-direction decomposition beyond what the loss returns (E5; `[KB]`
FINITE_DRAW S3.9.8), the DSN term the replicate term shares a step with
(P2), the flow it draws from (P4), the optimiser (P5), the search mechanics
that canonicalise these axes (P6). Nothing here is a measurement of training
behaviour: no job of `hpc/joint/` has run on the cluster (`[KB]` usage v1.3
S9); every number is a property of the loss as built, read from the code,
recomputed from it, or taken from the knowledge base's conjugate-anchor
simulations and marked so.

---

## 1. Notation and symbols

A subset of E0's master table, same types and units, plus the symbols this
document adds (declared in E0 under its convention 14, group "Replicate
term (P3)").

| Symbol | Name / Meaning | Type & domain | Units | First used in S |
|---|---|---|---|---|
| $x$ | one IFR window | $x \in \mathbb{R}^{W}_{\ge 0}$ | Hz (cohort); counts per bin per unit (bench) | S3.1 |
| $h_\psi$ | the encoder, $h_\psi : \mathbb{R}^{W} \to S^{E-1}$ | map; weights $\psi$ | -- | S3.2 |
| $\psi$ | encoder weights; the only weights the replicate term updates | $\psi \in \mathbb{R}^{n_\psi}$ | -- | S3.2 |
| $z$ | the embedding of a window, $z = h_\psi(x)$; $z_g$ for well $g$ | $z \in S^{E-1}$ | dimensionless | S3.2 |
| $q_\omega$ | the conditional flow, $q_\omega(\theta \mid z)$ for each fixed $z$ | conditional density on $\Theta$; weights $\omega$ | (param units)$^{-d_\theta}$ | S3.2 |
| $\omega$ | flow weights; frozen inside the replicate forward pass | $\omega \in \mathbb{R}^{n_\omega}$ | -- | S3.2 |
| $\theta$ | the inference parameters of one row | $\theta \in \Theta \subset \mathbb{R}^{d_\theta}$ | mixed | S3.2 |
| $\Theta$ | the prior box; the unit cube on every bank built so far | set | -- | S3.2 |
| $d_\theta$ | parameter-space dimension; 26 on the DUP15HD banks, 10 on the bench | $\mathbb{N}$ | -- | S3.1 |
| $p$ | latent dimension of the bank (the search's `--p` anchor) | $\mathbb{N}$ | -- | S3.1 |
| $g, g'$ | two rows of the real bank with the same donor; the code's "wells" (S3.2, F-ab) | indices | -- | S3.2 |
| $P$ | a donor | index | -- | S3.2 |
| $s$ | posterior-draw index | $s \in \{1, \dots, S_{\rm mc}\}$ | -- | S3.2 |
| $j$ | direction index | $j \in \{1, \dots, d_\theta\}$ | -- | S3.3 |
| $S_{\rm mc}$ | posterior draws per well (`n_posterior_draws`) | $\mathbb{N}$; searched in $[4 d_\theta, 400]$ | draws | S3.2 |
| $\theta^{(s)}_g$ | the $s$-th posterior draw for well $g$, i.i.d. from $q_\omega(\cdot \mid z_g)$ for each fixed $z_g$ | $\Theta$ | mixed | S3.2 |
| $m_g$ | exact posterior mean $\mathbb{E}_{q_\omega}[\theta \mid z_g]$ (analytic level) | $\mathbb{R}^{d_\theta}$ | param units | S3.2 |
| $\hat m_g$ | the $S_{\rm mc}$-draw sample mean (computed level) | $\mathbb{R}^{d_\theta}$ | param units | S3.2 |
| $e_g$ | the Monte Carlo error of the mean, $e_g = \hat m_g - m_g$; derivation-only | random vector in $\mathbb{R}^{d_\theta}$ | param units | S3.3 |
| $C_g$ | exact posterior covariance $\mathrm{Cov}_{q_\omega}(\theta \mid z_g)$ (analytic level) | PSD $d_\theta \times d_\theta$ | (param units)$^2$ | S3.2 |
| $\hat C_g$ | the $S_{\rm mc}$-draw sample covariance, `ddof=1` (computed level) | PSD; PD almost surely for $S_{\rm mc} > d_\theta$ | (param units)$^2$ | S3.2 |
| $\bar C$ | the symmetrised covariance the code builds, $\bar C = (\hat C_g + \hat C_{g'})/2$ (computed level) | PSD $d_\theta \times d_\theta$; PD almost surely for $S_{\rm mc} \ge d_\theta/2 + 1$ | (param units)$^2$ | S3.2 |
| $\bar C_{\rm true}$ | the exact symmetrised covariance $(C_g + C_{g'})/2$ that $\bar C$ estimates; derivation-only | PD $d_\theta \times d_\theta$ | (param units)$^2$ | S3.3 |
| $\Delta_{gg'}$ | exact disagreement $m_g - m_{g'}$ (analytic level) | $\mathbb{R}^{d_\theta}$ | param units | S3.2 |
| $\hat\Delta_{gg'}$ | the disagreement computed, $\hat m_g - \hat m_{g'}$ (computed level) | $\mathbb{R}^{d_\theta}$ | param units | S3.2 |
| $M$ | the metric of the statistic, $M = (2 \bar C)^{-1}$ in the code | SPD $d_\theta \times d_\theta$ | (param units)$^{-2}$ | S3.2 |
| $M_{\rm fix}$ | any symmetric matrix not built from the draws; the device of the correction's derivation | symmetric $d_\theta \times d_\theta$ | (param units)$^{-2}$ | S3.3 |
| $u$ | the auxiliary vector of the Cholesky solve $(2 \bar C)\, u = \hat\Delta_{gg'}$ | $\mathbb{R}^{d_\theta}$ | (param units)$^{-1}$ | S3.2 |
| $I_{d_\theta}$ | the $d_\theta \times d_\theta$ identity matrix; not the mutual information $I(\cdot\,;\cdot)$ | matrix | -- | S3.2 |
| $\epsilon_{\rm jit}$ | relative Cholesky jitter (`jitter`, `DEFAULT_JITTER`) | $\mathbb{R}_{>0}$ | dimensionless | S3.2 |
| $T_{gg'}$ | the exact statistic $\Delta_{gg'}^\top (2 \bar C_{\rm true})^{-1} \Delta_{gg'}$ (analytic level) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.3 |
| $\hat T^{\rm raw}_{gg'}$ | the statistic before the subtraction, $\hat\Delta_{gg'}^\top (2 \bar C)^{-1} \hat\Delta_{gg'}$ (computed level) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\hat T_{gg'}$ | the corrected statistic the code computes, $\hat T^{\rm raw}_{gg'} - d_\theta / S_{\rm mc}$ (computed level) | $\mathbb{R}$; negative is the collapse signature | dimensionless | S3.2 |
| $T_{\rm floor}$ | the clamp on $\hat T_{gg'}$ before the logarithm (`t_floor`, `DEFAULT_T_FLOOR`); not $T_{gg'}$ | $\mathbb{R}_{>0}$ | dimensionless | S3.2 |
| $\Sigma_0$ | prior covariance, the box's second moment $\mathrm{diag}((b_k - a_k)^2 / 12)$; $I_{d_\theta}/12$ on the unit cube | PD diagonal $d_\theta \times d_\theta$ | (param units)$^2$ | S3.2 |
| $a_k, b_k$ | prior box bounds on axis $k$ in inference coordinates | reals, $a_k < b_k$ | as $\theta$ | S3.2 |
| $k$ | axis index | $k \in \{1, \dots, d_\theta\}$ | -- | S3.2 |
| $p_{\rm eff}$ | effective number of data-constrained directions, $d_\theta - \mathrm{tr}(\Sigma_0^{-1} \bar C_{\rm true})$ (analytic level) | $[0, d_\theta]$ in the conjugate anchor | dimensionless | S3.2 |
| $\hat p_{\rm eff}$ | its estimate $d_\theta - \mathrm{tr}(\Sigma_0^{-1} \bar C)$, affine in $\bar C$ (computed level); the code's `p_eff_raw` | $\mathbb{R}$ | dimensionless | S3.2 |
| $p_{\rm min}$ | the clamp on $\hat p_{\rm eff}$ (`p_eff_min`, `DEFAULT_P_EFF_MIN`); not a density | $\mathbb{R}_{>0}$ | dimensionless | S3.2 |
| $\ell_{\rm rep}$ | the per-pair replicate loss as the code evaluates it, with both clamps, eq. (P3.7) (computed level) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\mathcal{L}^{\rm real}_{\rm rep}$ | its expectation over same-donor pairs (analytic level), plan eq. (3c) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\lambda_{\rm rep}$ | weight of the replicate term in $\mathcal{L}$, plan eq. (1); searched through $\log_{10} \lambda_{\rm rep} \in [-3, 1]$ behind the switch `rep_on` | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $t$ | training progress as a fraction of the planned optimiser steps, $t = n_{\rm done} / n_{\rm plan}$ | $[0, 1]$ | -- | S3.2 |
| $n_{\rm done}, n_{\rm plan}$ | the loop's optimiser-step counter, and the planned budget $n_{\rm ep} n_{\rm step}$ | $\mathbb{N}_0$, $\mathbb{N}$ | steps | S3.2 |
| $n_{\rm ep}, n_{\rm step}$ | epochs, and optimiser steps per epoch (`epochs`, `steps_per_epoch`) | $\mathbb{N}$ | -- | S3.2 |
| $t_{\rm warm}$ | the warm-up fraction (`warmup_frac_rep`) | $[0, 1)$; searched in $[0, 0.5]$ | -- | S3.2 |
| $r_{\rm rep}$ | the ramp of the replicate term, $r_{\rm rep}(t) = \min(1, t / t_{\rm warm})$ for $t_{\rm warm} > 0$, $1$ for $t_{\rm warm} = 0$ | $[0, 1]$ | dimensionless | S3.2 |
| $B_{\rm sim}, B_{\rm met}, B_{\rm rep}$ | rows (pairs for $B_{\rm rep}$) requested per step in the three streams | $\mathbb{N}$ | rows; pairs | S3.2 |
| $\mathcal{B}_{\rm sim}$ | one simulated minibatch | index set of size $B_{\rm sim}$ | -- | S3.2 |
| $\ell_i$ | the per-row NPE loss $-\log q_\omega(\theta_i \mid h_\psi(x_i))$ | $\mathbb{R}$ | nats | S3.2 |
| $\ell_{\rm DSN}$ | the metric loss of one batch (P2) | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\lambda_{\rm dsn}$ | weight of the DSN term | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\hat{\mathcal{L}}_{\rm step}$ | the per-step estimate of $\mathcal{L}$, plan eq. (2) | $\mathbb{R}$ | nats/row plus dimensionless terms | S3.2 |
| $\mathcal{L}$ | the joint objective, plan eq. (1) | $\mathbb{R}$ | nats/row plus dimensionless terms | S3.2 |
| $\mathcal{U}$ | the set of same-donor row pairs of the real bank after surrogate rows are masked, $\lvert \mathcal{U} \rvert = N_{\rm pair}$ | set of index pairs | -- | S3.2 |
| $\mathcal{B}_{\rm rep}$ | the pairs drawn at one step, $\lvert \mathcal{B}_{\rm rep} \rvert = \min(B_{\rm rep}, N_{\rm pair})$ | subset of $\mathcal{U}$ | -- | S3.2 |
| $N_{\rm pair}$ | same-donor pairs available, as the code counts them | $\mathbb{N}_0$ (E0 writes $\mathbb{N}$; the code's count can be 0, F-aa) | pairs | S3.2 |
| $n_{\rm wd}$ | rows per donor in the real bank (wells per donor when one row is one well; windows per donor on the banks built so far, F-ab) | $\mathbb{N}$ | -- | S3.2 |
| $G_{\rm don}$ | number of distinct donors | $\mathbb{N}$ | -- | S3.2 |
| $n_{\rm inv}$ | pairs of one replicate batch with $\hat p_{\rm eff} \le 0$ (`n_p_eff_invalid`), summed per epoch | $\mathbb{N}_0$ | pairs | S3.2 |
| $n_{\rm neg}$ | pairs of the diagnostic set with $\hat T_{gg'} \le 0$ (`n_T_nonpositive`) | $\mathbb{N}_0$ | pairs | S3.4 |
| $H_0$ | the composite null of plan eq. (3e): shared $\theta^*$ and a calibrated posterior | a hypothesis | -- | S3.3 |
| $\theta^*$ | the true parameter of one well; derivation-only on real data | $\Theta$ | mixed | S3.3 |
| $V$ | $\mathrm{Var}(\Delta_{gg'} \mid \theta^*)$ over replicate recordings; derivation-only | PSD $d_\theta \times d_\theta$ | (param units)$^2$ | S3.3 |
| $F$ | Fisher information of one well's data at $\theta^*$; derivation-only | PSD $d_\theta \times d_\theta$ | (param units)$^{-2}$ | S3.3 |
| $\lambda_j$, $c_j$ | the data-to-prior precision ratio along direction $j$, and the prior's share $1 / (1 + \lambda_j)$ | $\mathbb{R}_{\ge 0}$; $(0, 1]$ | dimensionless | S3.3 |
| $n_{\rm W}$ | degrees of freedom of $\bar C$, $n_{\rm W} = 2 (S_{\rm mc} - 1)$ | $\mathbb{N}$ | -- | S3.3 |
| $\mathcal{W}_d$, $\Sigma$, $d$ | the Wishart law $\mathcal{W}_d(n, \Sigma)$ of a $d \times d$ scatter matrix; its generic scale; its dimension | distribution; PSD; $\mathbb{N}$ | -- | S3.3 |
| $n$ | a generic count, qualified in prose | $\mathbb{N}$ | -- | S3.3 |
| $\kappa_S$ | the inflation of $\hat T^{\rm raw}_{gg'}$ from inverting the estimated $\bar C$, $\kappa_S = n_{\rm W} / (n_{\rm W} - d_\theta - 1)$ for Gaussian draws; not implemented | $\mathbb{R}_{>1}$ | dimensionless | S3.3 |
| $\mathrm{res}_S$ | the bias the subtraction leaves, $(\kappa_S - 1)(p_{\rm eff} + d_\theta / S_{\rm mc})$ | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.3 |
| $p_\times$ | the $p_{\rm eff}$ above which $\mathrm{res}_S$ exceeds $d_\theta / S_{\rm mc}$, written $p_\times(S_{\rm mc})$ | $\mathbb{R}$ | dimensionless | S3.3 |
| $\chi^2_d$ | the chi-square law with $d$ degrees of freedom | distribution | -- | S3.3 |
| $\tau_j$, $\hat\tau_j$ | the normalised per-direction term along $j$ (analytic; mean 1 under $H_0$), and its finite-draw counterpart | $\mathbb{R}_{\ge 0}$; $\mathbb{R}$ | dimensionless | S3.4 |
| $\mathcal{G}$ | a realised connectivity graph; a latent of the simulator | adjacency matrix | -- | S3.7 |
| $\nu$ | the nuisance latent | vector | dimensionless | S3.7 |
| $\rho_{\rm grad}$ | cosine between the NPE and DSN gradients with respect to $\psi$ | $[-1, 1]$ | -- | S3.6 |
| $\gamma_{\rm sep}$, $\tau_{\rm sep}$ | the DSN separation ramp and its warm-up fraction (P2) | $[0, 1]$ | dimensionless | S3.5 |
| $\eta$, $\gamma_{\rm wd}$ | the learning rate and the weight-decay coefficient (P5) | $\mathbb{R}_{>0}$, $\mathbb{R}_{\ge 0}$ | dimensionless | S3.6 |
| $\Sigma_{\rm rep}$ | covariance of $m_g - m_{g'}$ over same-donor pairs: the empirical nuisance floor (E7) | PSD $d_\theta \times d_\theta$ | (param units)$^2$ | S3.7 |

### 1.1 Conventions

- Code names in backticks are the knobs as the code spells them; the symbol
  of the same quantity is used only where a formula needs it.
- **Levels** (E0 convention 4): the plain symbol is the property of the flow's
  posterior at fixed conditioners ($m_g$, $C_g$, $\Delta_{gg'}$, $T_{gg'}$,
  $p_{\rm eff}$); the hat is the number one batch of draws puts in memory
  ($\hat m_g$, $\hat C_g$, $\hat\Delta_{gg'}$, $\hat T^{\rm raw}_{gg'}$,
  $\hat T_{gg'}$, $\hat p_{\rm eff}$); $\bar C$ is the computed matrix and
  $\bar C_{\rm true}$ the exact one, as the metric document writes them.
  Every statement about an expectation over repeated batches says so; the
  code holds one realisation.
- **Two statistics, two names.** $\hat T^{\rm raw}_{gg'}$ is the quadratic
  form; $\hat T_{gg'}$ is the quadratic form minus $d_\theta/S_{\rm mc}$. The
  plan writes `T_hat` for the second and the metric document's eq. (76)
  introduces the first; both are kept here because the clamp of eq. (P3.7)
  acts on the second and the collapse signature is its sign.
- **Status** (the Provenance model): every knob of this document is
  `configured`; $\hat m_g$, $\hat C_g$, $\bar C$, $\hat\Delta_{gg'}$,
  $\hat T^{\rm raw}_{gg'}$, $\hat T_{gg'}$, $\hat p_{\rm eff}$,
  $\ell_{\rm rep}$, $r_{\rm rep}$, $n_{\rm inv}$, $n_{\rm neg}$ are
  `computed` from one forward pass and the configuration; $\Sigma_0$ is
  `analytic`; $\kappa_S$, $\mathrm{res}_S$, $p_\times$ are `analytic` and
  **not implemented**; $m_g$, $C_g$, $\bar C_{\rm true}$, $\Delta_{gg'}$,
  $T_{gg'}$, $p_{\rm eff}$, $e_g$, $V$, $F$, $\theta^*$, $\lambda_j$, $c_j$
  are `derivation-only` on real data (the bench holds $\theta^*$).
- **Default** is per surface, as P0 S1.1: the runner's flag default is what a
  Stage 3 or Stage 4 run trains with when the flag is not passed; the
  library default (`ReplicateConsistencyLoss`, `TrainConfig`, `BatchSpec`,
  `train_joint`) is what a direct caller gets and is never what a cluster job
  runs (F-m).
- Equations are numbered (P3.n); plan equations are cited as "plan eq. (n)";
  the knowledge base's as "METRIC eq. (n)" and "FINITE_DRAW eq. (n)".
- "Well" and "row". The plan's pair $(g, g')$ is two *wells* of one donor.
  The code pairs *rows* of the real bank with the same `donor` value, and a
  row is a window (S3.2, F-ab). $g, g'$ index rows throughout; where the
  distinction matters the sentence says "window".
- "Collapse" has one sense here: $\hat T_{gg'} \to 0$, the wells agreeing
  more than the posterior permits (plan S2.5e). It is not the DSN's class
  collapse of P2.

## 2. Glossary

Ordered by first appearance.

- **Replicate stream** -- the third of the three batch streams of plan
  eq. (2): $B_{\rm rep}$ same-donor row pairs of the real bank per step,
  drawn from their own generator. S3.1.
- **Campaign switch** -- a binary axis (`rep_on`) that gates a block: off,
  the block's axes are pinned to canonical values and the term's weight is
  0. S3.1.
- **Reparameterised draw** -- a flow sample written as a deterministic
  function of base noise and the conditioner, so that its gradient with
  respect to the conditioner exists; `rsample`, as against `sample`. S3.2.
- **Symmetrised covariance** -- the average of the two wells' sample
  covariances, then symmetrised in the matrix sense; it makes the statistic
  exchangeable in $(g, g')$. S3.2.
- **Finite-draw correction** -- the additive term $d_\theta/S_{\rm mc}$
  subtracted from the quadratic form; exact for a metric built from the
  exact covariance. *Everyday meaning differs*: it corrects a mean over
  repeated batches, not any one value. S3.2.
- **Inverse-covariance inflation $\kappa_S$** -- the factor by which the
  inverse of an estimated covariance exceeds, on average, the inverse of
  the exact one; multiplicative, left in place by the correction, not
  implemented. S3.3.
- **Target** -- the value $\hat p_{\rm eff}$ the loss drives $\hat T_{gg'}$
  toward; not zero. S3.2.
- **Clamp** -- `torch.clamp(., min=.)`: a floor on a value whose gradient is
  zero wherever the floor binds. S3.2.
- **Dead zone** -- the set of pairs on which the clamp binds, so that the
  loss is constant and its gradient zero. S3.2, F-y.
- **Warm-up (ramp)** -- the linear schedule multiplying the replicate term
  by $r_{\rm rep}(t)$ over the first $t_{\rm warm}$ of the planned steps.
  S3.2.
- **Stop-gradient** -- computing a quantity in the forward pass and blocking
  backpropagation through it; applied to $\omega$ and to $\bar C$. S3.2.
- **Draw floor** -- the hard lower bound $4 d_\theta$ on $S_{\rm mc}$ in the
  search space. S3.3.
- **Conjugate anchor** -- the Gaussian-prior, Gaussian-likelihood reference
  model in which every object has a closed form; the regime of every
  $\kappa_S$ number quoted here. S3.3.
- **Collapse signature** -- $\hat T_{gg'} \le 0$: the quadratic form fell
  below its own Monte Carlo excess. Counted, never clamped, in the
  diagnostic. S3.4.
- **Singleton donor** -- a donor with one row in the real bank, contributing
  no pair; the count plan D12 turns on. S3.2.

## 3. Main body

### 3.1 Where the replicate knobs live, and what reaches the trainer

*Establishes the surfaces a replicate knob can be set on and the values each
surface holds, so that every later statement "the default is ..." has a
surface attached.*

The block has four searched axes and nine fixed knobs, and they are set on
six surfaces. A Stage 4 trial is built by `npe_tune_joint.build_argv` from a
configuration of the space; a Stage 3 arm is built from the runner's flags;
a direct caller of the library gets the library's defaults. The same knob
can carry three different values across those routes, and F-m, F-n and F-p
are all of that kind.

| knob | `JointSpaceSpec` / `default_joint_space` (searched range) | inactive pin (`INACTIVE_CANONICAL`, under `rep_on = 0`) | tuner -> runner flag (`AXIS_TO_FLAG` / `DERIVED`) | runner flag default (`run_joint_arms.py`) | PBS job variable (`joint_arms.pbs`) | library default |
|---|---|---|---|---|---|---|
| `rep_on` | $\{0, 1\}$, campaign switch | -- | gates `--lambda-rep`: $0$ when off, $10^{x}$ when on | no flag; the arm decides (`A5` only) | arm index | -- |
| `log10_lambda_rep` | $[-3, 1]$ | $0.0$ | `--lambda-rep` $= 10^{x}$ when on | `--lambda-rep 0.05` | `LAMBDA_REP=0.05` | `TrainConfig.lambda_rep = 0.0` |
| `warmup_frac_rep` | $[0.0, 0.5]$ | $0.0$ | `--warmup-frac-rep` | `0.3` | **not passed** (runner default applies) | `ReplicateConsistencyLoss.warmup = 0.0` |
| `n_posterior_draws` | $[4 d_\theta, 400]$ resolved by `default_joint_space`; the dataclass default `(100, 400)` (F-o) | the lower bound, resolved from the spec | `--n-posterior-draws` | `128` | `N_DRAWS=128` | `train_joint(n_posterior_draws=256)` |
| `--b-rep` | not an axis (F-l) | -- | not passed | `4` | **not passed** | `BatchSpec.b_rep = 8` |
| `jitter`, `t_floor`, `p_eff_min`, `correct_mc` | not axes | -- | -- | no flags; `ReplicateConsistencyLoss` defaults $10^{-6}$, $10^{-8}$, $10^{-3}$, `True` (F-i) | -- | the same |
| `detach_metric` | not an axis | -- | -- | `True`, no flag | -- | `True` |
| `Sigma0` | not an axis | -- | -- | `box_prior_covariance(0, 1)` $= I_{d_\theta}/12$ | -- | argument |
| diagnostic caps | not axes | -- | -- | 64 pairs, 256 report rows (F-i) | -- | -- |

`[REPO]` `stage4/joint_space.py:100, 136-138, 177-185, 231-233, 308-320`;
`stage4/npe_tune_joint.py:86-117, 213-217`; `stage3/run_joint_arms.py:194-195,
219, 223-225, 386-388, 507, 511-517, 533, 561, 614-618`;
`stage3/jobs/joint_arms.pbs:57-58, 112-113`; `stage2/joint_losses.py:79-83,
264-266`; `stage2/joint_train.py:41-43, 124-126`; `stage2/joint_batches.py:64`.

Three readings of the table that the rest of the document uses.

1. **The weight is derived, not mapped.** `rep_on` and `log10_lambda_rep`
   reach the runner only through `--lambda-rep`
   (`stage4/npe_tune_joint.py:213-217`): $\lambda_{\rm rep} = 10^{x}$ for a
   configuration with `rep_on = 1`, exactly $0$ otherwise. The runner then
   applies that weight only for arm `A5` (`arm_config`,
   `run_joint_arms.py:194-195`), and the tuner picks the arm from the two
   switches (`arm_for_config`, `:238-253`): `A5` when `rep_on = 1` and
   `dsn_on = 0`; both switches on has no arm and raises, so campaign `S-A25`
   cannot be evaluated today (P6).
2. **The two Stage 4 pins differ from the Stage 3 defaults.** A Stage 4 trial
   with `rep_on = 0` records $t_{\rm warm} = 0$ and $S_{\rm mc} = 4 d_\theta$
   (the pins; `:177-185`); a Stage 3 arm `A1` records $t_{\rm warm} = 0.3$
   and $S_{\rm mc} = 128$ (the runner's defaults). The term is off in both,
   so neither value trains anything, but $S_{\rm mc}$ still sets the
   *diagnostic* of every arm (S3.4): the `replicate` block of an `S-A1`
   trial's record is at 104 draws and a Stage 3 `A1` arm's at 128 (S3.6).
3. **The job passes two of the four.** `joint_arms.pbs` passes
   `--lambda-rep` and `--n-posterior-draws` and leaves `--warmup-frac-rep`
   and `--b-rep` at the runner's defaults (`:104-115`); a cluster `A5` run
   therefore ramps over 30% of training with 4 pairs per step unless the
   runner is invoked by hand.

*Plain.* Four numbers decide the replicate term: whether it is on, how
heavy, how slowly it is switched on, and how many samples it is computed
from. Each can be set in several places, and the places do not agree with
each other; the table says which value wins on which route.

### 3.2 The term as built

*Establishes, as one explicit function of the knobs, what one optimiser
step computes for the replicate stream: eq. (P3.1)-(P3.12). Every constant
in Table P3.1 is read from the code; every derived number is recomputed
`[RAN]`.*

**The pairs.** The real bank is loaded as rows with a `donor` field. The
batcher masks surrogate rows (none on the banks built so far: the runner
passes `surrogate=None`, `run_joint_arms.py:504`) and enumerates every
unordered pair of surviving rows with the same donor value
(`enumerate_donor_pairs`, `stage2/joint_batches.py:37-58, 160-164`):

$$\mathcal{U} = \big\{ (g, g') : g < g',\ \text{same donor} \big\}, \qquad N_{\rm pair} = \lvert \mathcal{U} \rvert = \sum_{P} \frac{n_{\rm wd}(n_{\rm wd} - 1)}{2}, \tag{P3.10}$$

the sum over donors $P$ with $n_{\rm wd}$ rows each; a donor with one row
contributes nothing and is counted (`n_singleton_donors`, the D12 number).
Each step draws
$\lvert \mathcal{B}_{\rm rep} \rvert = \min(B_{\rm rep}, N_{\rm pair})$ pairs
without replacement from a permutation of $\mathcal{U}$ under the stream's
own generator (seed $+3$; `:190-199`), so pairs repeat
across steps but not within one. **A row is a window, not a well**: the bench
bank writes one `donor` value per donor and $J$ windows per trace
(`stage1/build_latent_bank.py:224-252`), so $\mathcal{U}$ contains pairs of
two windows of one well as well as pairs across wells; the count and its
consequence are F-ab (S3.8). When $N_{\rm pair} = 0$ the stream returns
`None` every step and the loop skips the term without an error (F-aa).

**The draws, eq. (P3.1).** For each pair, the two conditioners
$z_g = h_\psi(x_g)$ and $z_{g'} = h_\psi(x_{g'})$ are formed and $S_{\rm mc}$
reparameterised draws are taken from each,

$$\theta^{(s)}_g \sim q_\omega(\cdot \mid z_g), \quad s = 1, \dots, S_{\rm mc}, \qquad \text{and likewise for } g', \tag{P3.1}$$

by `rsample_posterior` (`stage2/joint_model.py:143-166`), which reaches past
`sbi`'s public sampler to the zuko distribution's `rsample`. The draws are
in the box coordinates of $\Theta$: in `sbi` 0.27.0 the
`transform_to_unconstrained` map is the first element of the zuko `Flow`'s
transform chain (`neural_nets/net_builders/flow.py:1296-1312`), and
`ZukoFlow.sample` draws from the same distribution object
(`neural_nets/estimators/zuko_flow.py:141-156`) `[REPO]` (the wheel), so the
training-time `rsample` and the diagnostic's `sample` (S3.4) differ only in
whether the draw carries a gradient. The draws are i.i.d. given $z_g$ (base
noise through one fixed map) and independent across the two wells (two
calls), which is all the correction of eq. (P3.5) assumes (`[KB]`
FINITE_DRAW S3.3-S3.4, hypotheses (H1)-(H3)).

**The moments, eq. (P3.2).** `posterior_moments` (`joint_losses.py:125-154`):

$$\hat m_g = \frac{1}{S_{\rm mc}} \sum_{s=1}^{S_{\rm mc}} \theta^{(s)}_g, \qquad \hat C_g = \frac{1}{S_{\rm mc} - 1} \sum_{s=1}^{S_{\rm mc}} \big(\theta^{(s)}_g - \hat m_g\big)\big(\theta^{(s)}_g - \hat m_g\big)^\top, \tag{P3.2}$$

and the same for $g'$; the loss raises if the two wells were drawn with
different $S_{\rm mc}$ (`:304-305`) and if $S_{\rm mc} < 2$ (`:147-148`).

**The metric, eq. (P3.3).** `symmetrised_covariance` and `_chol_of_twice`
(`:157-165, 184-190`):

$$\bar C = \frac{1}{2}\big(\hat C_g + \hat C_{g'}\big), \quad \bar C \leftarrow \frac{1}{2}\big(\bar C + \bar C^\top\big), \qquad 2\Big(\bar C + \epsilon_{\rm jit}\, \frac{\mathrm{tr}(\bar C)}{d_\theta}\, I_{d_\theta}\Big) \text{ is Cholesky-factorised.} \tag{P3.3}$$

The matrix is detached before the factorisation (`replicate_statistic`,
`detach_metric=True`, `:193-207`): the metric carries no gradient. The jitter
is relative to the mean diagonal, so it does not change the scale; it is
dropped from the notation after this equation, and J16e checks that its
perturbation of the statistic is negligible `[REPO]`.

**The statistic, eqs. (P3.4)-(P3.5).** A Cholesky solve, never an inverse
(`:208-216`):

$$(2 \bar C)\, u = \hat\Delta_{gg'}, \qquad \hat\Delta_{gg'} = \hat m_g - \hat m_{g'}, \qquad \hat T^{\rm raw}_{gg'} = \hat\Delta_{gg'}^\top u, \tag{P3.4}$$

$$\hat T_{gg'} = \hat T^{\rm raw}_{gg'} - \frac{d_\theta}{S_{\rm mc}} \qquad (\text{`correct\_mc = True`, the only value any surface sets}). \tag{P3.5}$$

The subtraction is the finite-draw correction of plan eq. (3h): with the
*exact* metric $(2 \bar C_{\rm true})^{-1}$, the sample means' Monte Carlo
errors add $\mathrm{tr}\big((2\bar C_{\rm true})^{-1} \cdot 2\bar C_{\rm true}/S_{\rm mc}\big) = d_\theta/S_{\rm mc}$
to the expectation of the quadratic form, for every pair of conditioners and
every posterior shape with finite second moments (`[KB]` METRIC eq. (16),
FINITE_DRAW eqs. (69)-(71), (79)). Subtracting it makes $\hat T_{gg'}$
unbiased for $\mathbb{E}[T_{gg'} \mid \theta^*]$ -- **under the exact
metric**. The code's metric is $(2\bar C)^{-1}$, built from the same draws;
what that leaves is S3.3.4. Since the subtraction is a constant and the
quadratic form is non-negative, $\hat T_{gg'} < 0$ whenever the two sample
means nearly coincide: that is the collapse signature, and the loss's clamp
(below) is where it is handled.

**The target, eq. (P3.6).** `p_eff_from_trace` on the detached $\bar C$
(`:219-229, 309-310`):

$$\hat p_{\rm eff} = d_\theta - \mathrm{tr}\big(\Sigma_0^{-1} \bar C\big), \qquad \Sigma_0 = \mathrm{diag}\Big(\frac{(b_k - a_k)^2}{12}\Big)_{k=1}^{d_\theta} = \frac{1}{12} I_{d_\theta} \text{ on the unit cube.} \tag{P3.6}$$

$\Sigma_0$ is the box's second moment used as a Gaussian covariance (plan
S9); on every bank built so far the runner's prior is `BoxUniform(0, 1)` and
$\theta$ is stored in the unit cube (`run_joint_arms.py:386-388`; E0
convention 3), so $\Sigma_0^{-1} = 12\, I_{d_\theta}$ and
$\hat p_{\rm eff} = d_\theta - 12\, \mathrm{tr}(\bar C)$: the target is
$d_\theta$ minus twelve times the total posterior variance in cube units. It
is affine in $\bar C$, hence unbiased for $p_{\rm eff}$ at every $S_{\rm mc}$
and for any flow (`[KB]` METRIC S3.13.1). It is also detached, so within an
epoch the target is a constant of the step and the self-limiting effect of
plan S2.5e (a wider $\bar C$ raises the target as it lowers the statistic)
acts only across steps, when $\bar C$ is recomputed -- the code's docstring
says "within an epoch", but $\bar C$ is recomputed at every step from fresh
draws; the detachment is what removes the gradient route, not the epoch
boundary `[REPO]` `:27-32, 307-310`.

**The per-pair loss, eq. (P3.7).** `forward` clamps the target, then
`replicate_loss` clamps both arguments and squares the log-ratio
(`:232-236, 320-325`):

$$\ell_{\rm rep} = \Big( \log \max\big(\hat T_{gg'},\, T_{\rm floor}\big) - \log \max\big(\hat p_{\rm eff},\, p_{\rm min}\big) \Big)^2, \qquad T_{\rm floor} = 10^{-8}, \quad p_{\rm min} = 10^{-3}. \tag{P3.7}$$

(`replicate_loss` also clamps the target at $T_{\rm floor}$; since
$p_{\rm min} > T_{\rm floor}$ that second clamp never binds.) The loss is
two-sided about the target: zero at $\hat T_{gg'} = \hat p_{\rm eff}$,
growing on both sides, and -- as the plan's "divergence" of the constant map
-- growing without bound as $\hat T_{gg'} \to 0^+$ *down to the floor*. Its
derivative with respect to the statistic is

$$\frac{\partial \ell_{\rm rep}}{\partial \hat T_{gg'}} = \begin{cases} \frac{2}{\hat T_{gg'}} \Big( \log \hat T_{gg'} - \log \max(\hat p_{\rm eff}, p_{\rm min}) \Big), & \hat T_{gg'} > T_{\rm floor}, \\ 0, & \hat T_{gg'} \le T_{\rm floor}, \end{cases} \tag{P3.11}$$

because `torch.clamp` has zero gradient where its floor binds. Two
consequences are findings: the floor is reached by every pair with
$\hat T^{\rm raw}_{gg'} \le d_\theta/S_{\rm mc} + T_{\rm floor}$, i.e. by the
collapse signature itself (F-y, S3.7); and a non-positive target is replaced
by $p_{\rm min}$, which makes the loss pull $\hat T_{gg'}$ toward $10^{-3}$
rather than switching the pair off (F-z). The code counts the second case,
$n_{\rm inv}$ per batch, summed per epoch into the history
(`joint_train.py:201, 223`), and counts nothing for the first.

**The ramp, eq. (P3.8).** `set_progress` is called before every replicate
evaluation with $t = n_{\rm done}/n_{\rm plan}$, the loop's step counter over
the planned budget $n_{\rm ep} n_{\rm step}$ (`joint_train.py:148, 181`), and
the `ramp` property (`joint_losses.py:284-290`) is

$$r_{\rm rep}(t) = \begin{cases} \min\big(1,\ t / t_{\rm warm}\big), & t_{\rm warm} > 0, \\ 1, & t_{\rm warm} = 0, \end{cases} \tag{P3.8}$$

so the term enters linearly from 0 and is at full weight from step
$\lceil t_{\rm warm} n_{\rm plan} \rceil$ on. At the runner's defaults
($n_{\rm ep} = 10$, $n_{\rm step} = 25$, $t_{\rm warm} = 0.3$) the ramp is
complete at step 75, the first step of epoch 3; the logged `ramp` is the
value at the *last* step of each epoch (`:224`), 0.320, 0.653, 0.987 and
then 1 `[RAN]`. If `set_progress` is never called the property returns 1
(its default progress is 1.0, `:278`) -- which is what a run with
$N_{\rm pair} = 0$ logs from epoch 0 (F-aa). The fraction is of the *planned*
budget; a run stopped early by the selection score (patience 5 in
`TrainConfig`, 99 in the runner, F-m) can end before the ramp does.

**The step, eq. (P3.9).** With $\mathcal{B}_{\rm rep}$ the pairs drawn, the
loop's total for one step (`joint_train.py:164-202`) is

$$\hat{\mathcal{L}}_{\rm step} = \frac{1}{B_{\rm sim}} \sum_{i \in \mathcal{B}_{\rm sim}} \ell_i \;+\; \lambda_{\rm dsn}\, \ell_{\rm DSN} \;+\; \lambda_{\rm rep}\, r_{\rm rep}(t)\, \frac{1}{\lvert \mathcal{B}_{\rm rep} \rvert} \sum_{(g, g') \in \mathcal{B}_{\rm rep}} \ell_{\rm rep}, \tag{P3.9}$$

the DSN term present only under `A2`/`A2s`/`A3`, the replicate term only
under `A5`, and the gradient of the whole clipped at norm 5 (`:208-210`,
P5). The replicate term is computed under `stop_grad_params(model.flow_parameters())`
(`:185-188`): $\omega$ has `requires_grad = False` for the forward pass, so
the gradient of eq. (P3.9)'s last term reaches $\psi$ through
$z_g, z_{g'}$ and nothing else; the loop raises if the term arrives with no
gradient at all (`:189-195`), the symptom of freezing the whole estimator
(plan S2.5e; `[KB]` METRIC S3.14.1). J14r checks the three routes on the
real model `[REPO]` `stage2/smoke_test_joint.py:452-480`.

**What is logged.** Per epoch: the ramped batch loss averaged over the
epoch's steps (`rep`), the mean of
the batch means of $\hat T_{gg'}$ (`T`) and of the clamped target
(`p_eff`), $n_{\rm inv}$ summed, and the ramp at the epoch's last step
(`joint_train.py:198-201, 215-225`); nothing per pair, and nothing about the
clamp of $\hat T_{gg'}$. The `info` dict's `per_direction` is computed under
`no_grad` from the uncorrected $\hat\Delta_{gg'}$ and the realised $\bar C$
(`joint_losses.py:338-372`) and is consumed only by the diagnostic (S3.4).

**Table P3.1 -- the constants of the term, read from the code `[REPO]`, with
what each sets.**

| constant | value | where | what it sets | status |
|---|---|---|---|---|
| $\epsilon_{\rm jit}$ | $10^{-6}$ | `DEFAULT_JITTER`, `:79` | relative diagonal added before the two Cholesky factorisations (eq. (P3.3); `per_direction`, `:357`) | configured by code |
| $T_{\rm floor}$ | $10^{-8}$ | `DEFAULT_T_FLOOR`, `:80` | floor of $\hat T_{gg'}$ and of the target in the logarithm; the dead zone's edge (F-y) | configured by code |
| $p_{\rm min}$ | $10^{-3}$ | `DEFAULT_P_EFF_MIN`, `:83` | the target substituted for a non-positive $\hat p_{\rm eff}$ (F-z) | configured by code |
| `correct_mc` | `True` | `:264` | the subtraction of eq. (P3.5) | configured by code |
| `detach_metric` | `True` | `:194` | no gradient through $\bar C$ | configured by code |
| `ddof` | 1 | `:151` | unbiased $\hat C_g$, hence unbiased $\hat p_{\rm eff}$ | configured by code |
| stop-gradient set | `model.flow_parameters()` | `joint_train.py:185` | $\omega$ frozen in the replicate forward pass | configured by code |
| $\Sigma_0$ | $I_{d_\theta}/12$ | `run_joint_arms.py:386-388, 514-515` | the target's scale, eq. (P3.6); a registered buffer saved in the checkpoint (`:271`) | analytic |
| $B_{\rm rep}$ | 4 (runner), 8 (`BatchSpec`) | `run_joint_arms.py:219`; `joint_batches.py:64` | pairs per step; not searched, not a job variable (F-l, F-m) | configured |
| draws per step | $2 B_{\rm rep} S_{\rm mc} = 1024$ at the defaults | eq. (P3.1) | the cost of the term: 1024 draws of $d_\theta$ reals and 8 covariances per step `[RAN]` | computed |
| diagnostic caps | 64 pairs; 256 report rows | `run_joint_arms.py:561, 614` | the pairs and rows the end-of-run `replicate` block and the information spectrum are computed on (F-i) | configured by code |

*Plain.* For each pair of same-donor rows the model is asked for a cloud of
guesses about the parameters of each row; the two clouds' centres are
compared in units of the clouds' own width, a known Monte Carlo excess is
subtracted, the result is compared on a log scale with the number of
directions the data actually constrain, and the squared gap -- faded in over
the first part of training -- is added to the step's loss. Only the encoder
is allowed to learn from it.

### 3.3 The four searched axes

*Establishes, per axis, the five fields of plan S2.2: where it is set, what
it changes in eq. (P3.1)-(P3.9), what the range covers, how it fails and
what reveals the failure.*

#### 3.3.1 `rep_on`

- **Set on:** the campaign (`S-A5` and `S-A25` free; `S-A1` and `S-A2` pin
  it to 0); never a runner flag (`joint_space.py:368-398`).
- **Changes:** eq. (P3.9) by $\lambda_{\rm rep} = 0$ against
  $10^{\log_{10}\lambda_{\rm rep}}$, and the arm (`A1` against `A5`). Under
  `rep_on = 0` the canonicalisation pins the other three axes
  (`canonicalise_config` clause (b), `:574-576`), so two configurations that
  differ only in an inactive replicate coordinate build the same experiment
  and the same ledger key.
- **Range:** $\{0, 1\}$; no boundary (a categorical axis has no edge to
  widen, `_NO_BOUNDARY`, `:764`).
- **Interacts with:** `dsn_on` -- both on has no Stage 3 arm and
  `arm_for_config` raises (S3.1, P6).
- **Fails as:** the switch is on but the term is absent, which happens
  without an error when $N_{\rm pair} = 0$ (F-aa) or when the gradient is
  disconnected (the loop raises for the latter, `joint_train.py:189-195`).
- **Revealed by:** the batcher's report line `replicate pairs : 0`
  (`joint_batches.py:213`), a `ramp` of 1.0 logged at epoch 0 with
  $t_{\rm warm} > 0$, `T` equal to `nan` in every epoch, and a record without
  a `replicate` block (`run_joint_arms.py:613`).

#### 3.3.2 `log10_lambda_rep` ($\log_{10} \lambda_{\rm rep}$)

- **Set on:** the space, $[-3, 1]$, linear in the exponent (`:231`); reaches
  the runner as `--lambda-rep` $= 10^{x}$ (S3.1); Stage 3 default 0.05.
- **Changes:** the weight of the last term of eq. (P3.9). Because
  $\ell_{\rm rep}$ is a squared log-ratio -- dimensionless and of order 1
  near its target, of order $(\log(\hat T_{gg'}/\hat p_{\rm eff}))^2$ away
  from it -- $\lambda_{\rm rep}$ carries only the ratio of two loss scales
  and the row-count asymmetry of the streams, not a units conversion (plan
  S5.1). The DSN's analogous weight has the same range and the same
  argument (P2 S3.3.2).
- **Range:** $\lambda_{\rm rep} \in [10^{-3}, 10]$ `[RAN]`. The plan's
  reason for searching rather than setting it is the constant-map argument
  (plan S2.5, second caution): every pure invariance objective has the
  constant encoder as a global minimum, and although the two-sided target
  turns that minimum into a divergence, the weight at which the NPE term
  still dominates the early, collapse-side regime is unknown and is what
  the search measures.
- **Interacts with:** $S_{\rm mc}$, because the expected loss is minimised
  not at the true disagreement $p_{\rm eff}$ but at
  $p_{\rm eff}/\kappa_S - (1 - 1/\kappa_S)\, d_\theta/S_{\rm mc}$ (eq.
  (P3.12), S3.3.4): a smaller $S_{\rm mc}$ asks for more agreement, and a
  larger $\lambda_{\rm rep}$ enforces it harder, so the search's partial
  dependence on the two is confounded (`[KB]` METRIC S3.13.1 item 3,
  FINITE_DRAW S3.9.8 item 2). With $t_{\rm warm}$, through the product
  $\lambda_{\rm rep} r_{\rm rep}(t)$. With the gradient clip: near the floor
  the per-pair gradient weight of eq. (P3.11) is
  $2\lvert \log(\hat T_{gg'}/\hat p_{\rm eff}) \rvert / \hat T_{gg'}$,
  68 at $\hat T_{gg'} = 0.1$ and 1141 at $0.01$ against $p_{\rm eff} = 3$
  `[RAN]`, so at any
  $\lambda_{\rm rep}$ of order 1 the clip at norm 5 (P5) is what bounds the
  step early in training [reasoning].
- **Fails as:** too large, the collapse-side pairs dominate the clipped
  gradient from the first ramped steps and the NPE validation score stalls;
  too small, `A5` is `A1` with extra compute.
- **Revealed by:** `rep` against `npe` and `val_npe` in the history; the
  partial dependence of the ledger on this axis (P6).

#### 3.3.3 `warmup_frac_rep` ($t_{\rm warm}$)

- **Set on:** the space, $[0.0, 0.5]$, a range copied from the DSN's
  `SearchConfig.sep_warmup_frac_range` (`RANGE_PROVENANCE`, `:137`;
  `hpc/dsn/config.py:930`); inactive pin $0.0$; runner default $0.3$; library
  default $0.0$ (F-p); not passed by the job.
- **Changes:** $r_{\rm rep}(t)$ of eq. (P3.8): the fraction of the planned
  steps over which the term's weight rises linearly from 0 to
  $\lambda_{\rm rep}$. At $t_{\rm warm} = 0$ the term is at full weight from
  step 0.
- **Range:** at the runner's budget (250 steps) the ramp spans 0 to 125
  steps across the range `[RAN]`. The plan asked for the warm-up to end on
  a *measured* criterion -- the validation NLL plateauing, or gate G1
  passing -- rather than an epoch count (plan S2.5e); the implementation uses
  a fraction of the planned budget, which the metric document records as a
  deviation (`[KB]` METRIC S3.14.4, S5). Searching the fraction is the
  stack's substitute for measuring the criterion.
- **Why it exists** (plan S2.5e, two independent reasons): before the flow is
  informative $\bar C$ is not a meaningful ruler; and at initialisation an
  untrained flow barely depends on $z$, so $\hat m_g \approx \hat m_{g'}$,
  $\hat T^{\rm raw}_{gg'}$ is near its Monte Carlo excess alone, and the term
  starts on the collapse side of its target, where eq. (P3.11) gives its
  largest gradients -- against the NPE term, over the interval where that
  term does its only important work. According to PubMed, the same device
  under the name "KL cost annealing" -- a regularising term whose weight is
  raised gradually from 0 to 1 over the first epochs because at full weight
  from the start it drives the latent variable to a degenerate (collapsed)
  solution -- is the standard mitigation of posterior collapse in
  variational autoencoders, and the optimal annealing schedule is reported
  to vary with the dataset and the architecture, which is the case for
  searching rather than fixing the fraction (Song et al. 2025,
  [DOI](https://doi.org/10.3390/e27040423)) [PubMed full text, PMC12026048].
  That source concerns text-modelling VAEs; the analogy -- a term that would
  dominate early is faded in -- is drawn here, and no number from it is
  used.
- **Interacts with:** `epochs` and `steps_per_epoch` (P5), which fix
  $n_{\rm plan}$ and therefore how many optimiser steps the fraction is; the
  early-stopping patience, which can end a run before $t = t_{\rm warm}$;
  the DSN's own ramp $\gamma_{\rm sep}$, a different schedule on a different
  quantity (S3.5).
- **Fails as:** too short, the term fights the likelihood from the first
  steps and the invalid-target count stays non-zero into the ramped phase;
  too long, `A5` spends most of its budget as `A1` and the term's effect is
  under-measured.
- **Revealed by:** `ramp` and `n_p_eff_invalid` per epoch; a non-zero
  $n_{\rm inv}$ after the ramp completes is the finding the code's own
  comment names (`joint_losses.py:312-319`).

#### 3.3.4 `n_posterior_draws` ($S_{\rm mc}$)

- **Set on:** the space, $[4 d_\theta, 400]$ as resolved by
  `default_joint_space` (`:308-320`): 104 to 400 at $d_\theta = 26$, 40 to
  400 on the bench `[RAN]`; the dataclass default `(100, 400)` is below the
  floor at $d_\theta = 26$ and only the resolver repairs it (F-o); inactive
  pin: the floor; runner default 128; job variable `N_DRAWS=128`; library
  default 256 (F-m).
- **Changes:** the number of draws of eq. (P3.1), hence the Monte Carlo error
  of every moment of eq. (P3.2), the subtraction of eq. (P3.5), the
  realisation of $\bar C$ that the metric and the target are built from, the
  cost of the step (Table P3.1), and -- for every arm, on or off -- the
  draws the end-of-run diagnostics use (S3.4).
- **The floor, three readings.** The code and the plan justify
  $4 d_\theta$ by rank: "below it the per-culture covariance estimate
  $\hat C_g$ is rank-deficient and the Cholesky solve fails outright; the
  axis trades compute against the $d_\theta/S_{\rm mc}$ variance inflation,
  and the correction is applied regardless, so the axis controls variance
  and never bias" `[REPO]` `stage4/joint_space.py:276-281`, repeated in the
  error message (`:312-316`) and in plan S5.1 (`:1082-1087`, v0.6.5). Both
  halves of that sentence are superseded by the knowledge base (F-e), and
  the repository patch that corrects them (plan v0.6.6, the two docstrings,
  one error message; `[KB]` METRIC S5) is not applied at `834eb41`:
  1. *Rank.* $\hat C_g$ has rank $\min(S_{\rm mc} - 1, d_\theta)$ almost
     surely, so it is full rank from $S_{\rm mc} = 27$ at $d_\theta = 26$,
     and $\bar C$ -- the matrix actually factorised -- has rank
     $\min(n_{\rm W}, d_\theta)$, full from $S_{\rm mc} = 14$ `[RAN]`; the
     knowledge base ran the repository's own `replicate_statistic` through
     200 of 200 pairs at every
     $S_{\rm mc} \in \{14, 15, 20, 26, 27, 50, 104\}$ and saw it fail only at 13 (`[KB]` METRIC S3.13.1, item 2). The
     floor is 7.4 times the rank bound of $\bar C$.
  2. *Bias.* With the exact metric the subtraction removes the whole
     finite-draw bias; with the realised metric it does not, and what is
     left is the multiplicative $\kappa_S$ of eq. (P3.12) below. The axis
     controls a bias as well as a variance, and the floor is where that
     bias is $\kappa_S = (8 d_\theta - 2)/(7 d_\theta - 3)$: 1.151 at
     $d_\theta = 26$, 1.164 at $d_\theta = 10$ (`[RAN]`; `[KB]` METRIC
     S3.13.1).
  3. *Compute.* Draws per step scale linearly in $S_{\rm mc}$ (Table P3.1):
     the top of the range costs 3.8 times the floor.
- **The bias the subtraction leaves, eq. (P3.12).** Under three assumptions
  -- (a) Gaussian draws, (b) $C_g = C_{g'}$, as in the conjugate anchor,
  (c) $H_0$ in that anchor -- the two wells' scatter matrices add to a
  Wishart matrix,
  $n_{\rm W} \bar C \sim \mathcal{W}_{d_\theta}(n_{\rm W}, \bar C_{\rm true})$
  with $n_{\rm W} = 2(S_{\rm mc} - 1)$, whose inverse has mean
  $\bar C_{\rm true}^{-1} \cdot n_{\rm W}/(n_{\rm W} - d_\theta - 1)$
  [textbook, from memory, as the knowledge base tags it], and the
  expectation of the code's statistic is (`[KB]` METRIC eq. (72),
  FINITE_DRAW eq. (72), (88)-(89))

  $$\mathbb{E}\big[\hat T_{gg'} \,\big|\, \theta^*\big] = \kappa_S \Big( p_{\rm eff} + \frac{d_\theta}{S_{\rm mc}} \Big) - \frac{d_\theta}{S_{\rm mc}}, \qquad \kappa_S = \frac{n_{\rm W}}{n_{\rm W} - d_\theta - 1}, \qquad \mathrm{res}_S = (\kappa_S - 1)\Big( p_{\rm eff} + \frac{d_\theta}{S_{\rm mc}} \Big), \tag{P3.12}$$

  valid for $n_{\rm W} > d_\theta + 1$, i.e. $S_{\rm mc} \ge 15$ at
  $d_\theta = 26$ `[RAN]`. The knowledge base verified eq. (P3.12) on a
  conjugate bench from $S_{\rm mc} = 20$ to 1000 to within two standard
  errors in every row (`[KB]` METRIC S3.13.1, the six-decade table). The
  numbers across the range, at $p_{\rm eff} = 3$, recomputed `[RAN]`
  (`tools/p3_numbers.py`, block B2), agree with `[KB]` FINITE_DRAW S3.9.5:

  | $S_{\rm mc}$ | $d_\theta/S_{\rm mc}$ | $\kappa_S$ | $\mathrm{res}_S$ at $p_{\rm eff} = 3$ | $\mathrm{res}_S / (d_\theta/S_{\rm mc})$ | $p_\times(S_{\rm mc})$ | loss minimised at |
  |---|---|---|---|---|---|---|
  | 104 (the floor) | 0.250 | 1.151 | 0.490 | 1.96 | 1.41 | 2.57 |
  | 128 (Stage 3, the job) | 0.203 | 1.119 | 0.381 | 1.88 | 1.51 | 2.66 |
  | 256 (`train_joint` default) | 0.102 | 1.056 | 0.173 | 1.71 | 1.72 | 2.84 |
  | 285 ($\kappa_S \le 1.05$) | 0.091 | 1.050 | 0.154 | 1.69 | 1.74 | 2.85 |
  | 400 (the ceiling) | 0.065 | 1.035 | 0.107 | 1.65 | 1.79 | 2.90 |
  | 1000 | 0.026 | 1.014 | 0.041 | 1.59 | 1.87 | 2.96 |

  The last column is where the expected loss is smallest under (a)-(c) and
  with the logarithm's own Jensen terms ignored: the true disagreement
  $p_{\rm eff}/\kappa_S - (1 - 1/\kappa_S)\, d_\theta/S_{\rm mc}$, below the
  target at every $S_{\rm mc}$ (`[KB]` FINITE_DRAW S3.9.8 item 2). Read
  across the range: at every point of it the bias the realised metric adds
  exceeds the bias the subtraction removes whenever
  $p_{\rm eff} > p_\times(S_{\rm mc})$, between 1.41 and 1.79 -- the data need constrain
  only about two directions for the uncorrected effect to be the larger
  one; raising $S_{\rm mc}$ shrinks both without changing which dominates;
  and the search's ceiling of 400 still leaves 3.5% inflation. Outside
  (a)-(c) the size changes and the direction is established only under
  independence of $\hat\Delta_{gg'}$ and $\bar C$ (`[KB]` FINITE_DRAW
  S3.9.2, S3.9.6-S3.9.7): for a trained flow, whose draws are not Gaussian
  and whose $C_g \ne C_{g'}$, eq. (P3.12) is the anchor's value, not a
  measurement, and the knowledge base lists measuring it on the bench as an
  open point.
- **Range:** the floor is the one hard bound of the space; the ceiling 400
  is `n_posterior_draws_max`, an argument of `default_joint_space`
  (`:267, 309`), raised when a best configuration sits on it
  (`boundary_axes`, P6). What the range trades is $\kappa_S$ and the Monte
  Carlo variance against compute, and -- through eq. (P3.12) -- the point
  the loss trains toward, which is the confound with $\lambda_{\rm rep}$ of
  S3.3.2.
- **Fails as:** below 15, an infinite mean of the inverse (not reachable
  through the space); at the floor, a 15% inflation that the loss reads as
  a 15% overconfidence and corrects by over-agreement; at any
  $S_{\rm mc}$, the dead zone of F-y, whose size the Monte Carlo of block B6
  gives: a collapsed pair ($\Delta_{gg'} = 0$, Gaussian draws, realised
  metric) has $\hat T_{gg'} \le 0$ with probability 0.36 at 104 draws, 0.39
  at 128 and 0.48 at 256, approaching the exact-metric value
  $\Pr(\chi^2_{26} \le 26) = 0.54$ `[RAN]`.
- **Revealed by:** nothing in the training history separates $\kappa_S$
  from a real disagreement; the diagnostic's `p_eff_gap` (S3.4) compares
  the posterior's target with the spectrum's but not the statistic with the
  exact-metric one. The bench can measure $\kappa_S$ for the real flow by
  recomputing the statistic with the exact-covariance metric at fixed
  conditioners (`[KB]` FINITE_DRAW S5); nothing in the repository does.

### 3.4 The fixed knobs

*Establishes what the joint stack holds fixed around the four axes, where,
and what each fixed value commits the term to.*

| knob | value | where | what it commits to | note |
|---|---|---|---|---|
| `correct_mc` | `True` | `ReplicateConsistencyLoss`, no flag | eq. (P3.5): the subtraction, hence $\hat T_{gg'} \in \mathbb{R}$ and the dead zone of F-y; the diagnostic passes `True` as well (`joint_diagnostics.py:237`) | the numpy specification's default is `False` (`replicate_statistic.py:66`): a direct caller of the specification gets the uncorrected statistic (F-i) |
| `jitter` $\epsilon_{\rm jit}$ | $10^{-6}$ | no flag | eq. (P3.3); keeps a collapsing arm scoring badly instead of crashing | relative, so scale-free; its perturbation of $\hat T_{gg'}$ is checked negligible by J16e, not bounded analytically (`[KB]` METRIC S3.14.2) |
| `t_floor` $T_{\rm floor}$ | $10^{-8}$ | no flag | eq. (P3.7); the loss plateau for a dead-zone pair is $(\log 10^{-8} - \log \hat p_{\rm eff})^2$: 381 at $\hat p_{\rm eff} = 3$, 440 at 13 `[RAN]` | J11 tests the loss's growth with a floor of $10^{-30}$ and values, not gradients (`smoke_test_joint_losses.py:82-106`) |
| `p_eff_min` $p_{\rm min}$ | $10^{-3}$ | no flag | eq. (P3.7); the target of an invalid pair (F-z) | the clamp is counted ($n_{\rm inv}$) and the raw value kept in `info`, never logged per pair |
| `detach_metric` | `True` | `replicate_statistic`, no flag | no gradient through $\bar C$: the metric cannot be inflated to lower the loss; that route is blocked by the NPE term instead (plan S2.5e) | the target is detached by the same `.detach()` (`:309`) |
| stop-gradient set | `flow_parameters()` | `joint_train.py:185` | only $\psi$ moves under the term | the whole-estimator variant freezes $\psi$ too and was the Stage 3 trap (`[KB]` METRIC S3.14.1) |
| `Sigma0` | $I_{d_\theta}/12$ | `run_joint_arms.py:514` | eq. (P3.6) over all $d_\theta$ axes, kernel axes included: D17 closed as option (c), no axis-subset parameter exists (`joint_losses.py:37-45`) | the target is overstated on the three kernel axes by an unmeasured amount (plan S8 D17; S3.7) |
| `--b-rep` $B_{\rm rep}$ | 4 (runner); 8 (`BatchSpec`) | not searched, not a job variable | pairs per step; the batch mean of eq. (P3.9) is over 4 pairs, so the per-step replicate gradient is a 4-pair average of a heavy-tailed quantity (eq. (P3.11)) | F-l, F-m |
| `n_posterior_draws` in `train_joint` | 256 | library default only | never reached by a cluster run; the runner passes 128 | F-m; `stage3c` has its own pair at 128/64 (F-n, P7) |
| the samplers | `rsample_posterior` (training), `sample_posterior` (diagnostic) | `joint_model.py:134-166` | same distribution, same coordinates (S3.2); only the gradient differs | a coupling to `sbi` 0.27 internals, flagged in the code |
| diagnostic pairs | the first 64 of $\mathcal{U}$ in enumeration order | `run_joint_arms.py:614` | `replicate_report`: $\hat T_{gg'}$, $\hat p_{\rm eff}$, $n_{\rm neg}$, the log-ratio mean, $n_{\rm inv}$, the unramped loss, the per-direction mean by index | the first 64 pairs are the first donors' pairs, not a sample of $\mathcal{U}$ (F-i); `p_eff_gap` is the posterior target minus the spectrum's $p_{\rm eff}$ on at most 256 report rows (`:561, 572, 620`) |
| per-direction output | `per_direction_mean` | `joint_diagnostics.py:271` | the P16 ranking of `report_joint_arms.py` | as implemented it is dominated by a clamp artefact of order $10^{10}$ in every pair, because the share floor $10^{-12}$ binds wherever the realised whitened eigenvalue exceeds 1 (`[KB]` FINITE_DRAW S3.9.8 item 5; METRIC S3.19, added 2026-10-01); the loss does not use it |

### 3.5 What is inherited from the DSN, and what differs (D-036)

*Establishes that the replicate block inherits one range and no mechanism
from the standalone DSN, and sets the two ramps side by side so they are
not confused.*

Unlike the encoder and loss blocks (P1, P2), the replicate block has no
counterpart in the standalone DSN: the term, its statistic and its target
are the joint stack's own (plan S2.5). The one inheritance is the *range* of
`warmup_frac_rep`, copied from `SearchConfig.sep_warmup_frac_range`
(`[REPO]` `joint_space.py:137`; `dsn/config.py:930`). The ramp it
parameterises differs from the DSN's in every other respect, and P2
eq. (P2.9) is the other side of this table:

| | the replicate ramp $r_{\rm rep}(t)$, eq. (P3.8) | the DSN separation ramp $\gamma_{\rm sep}(t)$ (P2 eq. (P2.9)) |
|---|---|---|
| what is ramped | the whole replicate term, $\lambda_{\rm rep} r_{\rm rep}(t)$ | the separation weight only, $\lambda_{\rm sep} \gamma_{\rm sep}(t)$; the hinges are at full weight from step 0 |
| the fraction | $t_{\rm warm}$, **searched** in $[0, 0.5]$ (`S-A5`) | $\tau_{\rm sep}$, **fixed at 0** in the joint space (plan S5.1): the DSN ramp is off |
| the progress counter | the loop's optimiser step over the planned budget, driven by `set_progress` before every replicate evaluation | the adapter's own evaluation count, frozen during the $\rho_{\rm grad}$ probe |
| the default a Stage 3 arm runs | 0.3 (runner), not passed by the job | 0.0 |
| what the plan asked | a measured end criterion (plan S2.5e) | the DSN's convention, carried over |
| what it protects | the NPE term from a collapse-side replicate gradient at initialisation | the hinges from a premature centroid pull (TUNING_1, P2) |

Two things to carry. First, "warm-up" in a joint run means two different
schedules, one on and one off, and the run record's `ramp` is the replicate
one (`sep_warmup` is the DSN adapter's state, logged separately,
`joint_train.py:250-252`). Second, the inherited range `[0, 0.5]` was chosen
for the DSN's separation weight; nothing in the repository argues it is the
right range for a term that starts on the collapse side of a divergent
loss, and the plan's measured criterion would make the question moot.

### 3.6 Interactions with the other blocks

*Establishes, per block, which knob of this document changes what the other
block's knobs do.*

- **The flow (P4).** The draws of eq. (P3.1) are the flow's; `hidden_features`
  and `num_transforms` set how sharp $q_\omega(\cdot \mid z)$ can be, hence
  $\bar C$, hence both the metric and the target. The stop-gradient leaves
  $\omega$ untouched by the term, so the flow learns only from the NPE term;
  the replicate term can make $z_g \approx z_{g'}$ but not make the flow
  ignore $z$ (the desensitisation route of plan S2.5e). The draws are in box
  coordinates for either sampler (S3.2).
- **The encoder (P1).** The gradient reaches $\psi$ through $z_g$ and
  $z_{g'}$ only; the encoder is asked to map same-donor rows to conditioners
  whose posterior means disagree by $p_{\rm eff}$ in posterior units -- not
  to identical $z$ (plan S2.5, first caution). Under `A5` the encoder is
  trained from step 0 by the NPE term and from step
  $\lceil t_{\rm warm} n_{\rm plan} \rceil$ onward at full replicate weight; `A5` never
  pre-trains the encoder (`run_joint_arms.py:400, 458` runs the encoder-only
  pre-training for the DSN arms only).
- **The DSN term (P2).** The two terms never share a step today (S3.3.1).
  Were `S-A25` to run, the metric batch and the replicate batch would be
  drawn from the real bank by two independent generators (seeds $+2$ and
  $+3$), and the DSN's "warm-up" and this one would be two schedules
  (S3.5).
- **The optimiser (P5).** `epochs` and `steps_per_epoch` set $n_{\rm plan}$
  and therefore how many steps the fraction $t_{\rm warm}$ is; the runner's
  patience of 99 makes the planned and the actual budget equal, the library
  default of 5 does not (F-m). `grad_clip` at 5 bounds the step when the
  per-pair gradient weight of eq. (P3.11) is large (S3.3.2). `lr` scales
  both terms alike. `weight_decay` acts on $\psi$ whatever the term does.
  `batch_size_npe` sets $B_{\rm sim}$ and leaves $B_{\rm rep}$ at 4: the
  replicate gradient is averaged over 4 pairs while the NPE gradient is
  averaged over 128 to 1024 rows, a noise asymmetry no knob controls
  [reasoning].
- **The search (P6).** Under `rep_on = 0` the pins make the three
  coordinates inert for deduplication; `boundary_axes` skips them and uses
  an absolute tolerance near 0 for `warmup_frac_rep` because its range
  starts at 0 (`:808-811`). `S-A5` has 16 free axes (F-j). The floor is
  resolved by `default_joint_space` from `--d-theta` (`npe_tune_joint.py:343-345`);
  a `JointSpaceSpec()` built by hand carries `(100, 400)` (F-o).
- **The diagnostics (every arm).** `n_posterior_draws` is passed to every
  arm and sets the draws of the information spectrum (256 report rows), the
  per-axis contraction and the `replicate` block (64 pairs) whether or not
  the term trained (`run_joint_arms.py:561, 614-618`). Two records at the
  same posterior but different $S_{\rm mc}$ differ in `T_mean` by the ratio
  of their $\kappa_S$: 1.151/1.119, about 3%, between the Stage 4 pin (104)
  and the Stage 3 default (128) `[RAN]`, and in `n_T_nonpositive` by the
  dead-zone fractions of S3.3.4. A comparison of `T_mean` across surfaces
  must carry $S_{\rm mc}$.

### 3.7 Failure modes and the diagnostics that reveal them

*Establishes, in one table, how the term fails, which knob is implicated,
and which number in the run record shows it. The three escape routes of plan
S2.5e head the table; the four findings of this document follow.*

| failure | mechanism | knob(s) | what blocks it, as built | revealed by |
|---|---|---|---|---|
| collapse ($\hat T_{gg'} \to 0$) | the encoder makes same-donor rows indistinguishable to the flow | $\lambda_{\rm rep}$, $t_{\rm warm}$ | the two-sided target: $\ell_{\rm rep} \to (\log T_{\rm floor} - \log \hat p_{\rm eff})^2$ -- but the gradient is zero below the floor (F-y) | `T` falling toward 0 while `rep` rises; $n_{\rm neg}$ in the diagnostic; `rep` near the plateau values of S3.4 |
| inflation ($\bar C$ widened to shrink the metric) | the flow widens its posterior | -- | the NPE term: a wide posterior scores badly on held-out rows; the metric and target are detached, so the term itself has no route to this | `val_npe` worsening; `p_eff` falling toward 0, then $n_{\rm inv}$ rising |
| desensitisation ($q_\omega$ ignores $z$) | the flow stops reading its conditioner | -- | the stop-gradient on $\omega$ | the Stage 3 trap: `A5` equal to `A1` to four decimals when the gradient is cut; the loop now raises |
| disconnected term | `stop_grad_params` over the whole estimator, or `sample` instead of `rsample` | -- | the `requires_grad` check, `joint_train.py:189-195`; J14r | a `RuntimeError` at the first replicate step |
| **F-y, dead zone** | every pair with $\hat T^{\rm raw}_{gg'} \le d_\theta/S_{\rm mc}$ contributes a constant loss and no gradient; at initialisation that is 36-48% of pairs at the searched $S_{\rm mc}$ `[RAN]` | $S_{\rm mc}$, $T_{\rm floor}$, `correct_mc` | nothing; the restoring force exists only above the floor | a `rep` of order 100 with a `T` near 0; not counted in training ($n_{\rm neg}$ exists in the diagnostic only) |
| **F-z, invalid target** | $\hat p_{\rm eff} \le 0$ is replaced by $p_{\rm min} = 10^{-3}$, so the loss pulls $\hat T_{gg'}$ toward $10^{-3}$: toward agreement, with slope sign $+$ for every $\hat T_{gg'} > 10^{-3}$ `[RAN]` | $p_{\rm min}$, $t_{\rm warm}$ | the ramp, if the invalid phase ends before it does | $n_{\rm inv}$ non-zero after the ramp completes |
| **F-aa, no pairs** | $N_{\rm pair} = 0$: the stream returns `None`, the loop skips the term, `A5` trains as `A1` | `rep_on`, the real bank's `donor` field | nothing | `replicate pairs : 0` in the batcher report; `ramp` 1.0 at epoch 0; `T` equal to `nan`; no `replicate` block in the record |
| **F-ab, window pairs** | rows are windows; pairs of two windows of one well are enumerated as replicate pairs, 47% of $\mathcal{U}$ at the Stage 1 defaults `[RAN]`; such a pair shares its realisation $\mathcal{G}$ and nuisance $\nu$, so its disagreement carries only window noise and sits below the target derived for two independent recordings [reasoning] | the real bank's `donor` field; `--wells-per-donor`, `--n-windows` | nothing | not separable in the record: `T_mean` mixes the two kinds of pair; the per-pair $\hat T_{gg'}$ list of the diagnostic (`T`) can be split by well after the fact |
| overstated target on the kernel axes (D17) | one realisation per kernel value in the bank overstates $F$, hence $p_{\rm eff}$; the two-sided loss then trains the encoder to make same-donor rows disagree *more* on those axes | -- (all axes, by decision) | nothing (option (c): accept and record) | `d17_realisation_audit` reports the count informationally (plan S8 D17); the realisation floor of Stage 3c (E7) |
| $\kappa_S$ pushes to the collapse side | eq. (P3.12): the loss is minimised at a true disagreement below $p_{\rm eff}$ | $S_{\rm mc}$ (with $\lambda_{\rm rep}$) | nothing (no deflation implemented; `[KB]` METRIC S5, open) | not visible in the record; measurable on the bench with the exact metric |
| multimodal posterior | the posterior mean sits between modes; two rows on different modes give a large $\hat\Delta_{gg'}$ that is degeneracy, not miscalibration | -- | nothing; the Bayes-factor form of plan eq. (3d) is not implemented (D16, open) | nothing counts modes |
| dependent wells (shared plate, batch) | $\mathrm{Cov}(m_g, m_{g'} \mid \theta^*) \ne 0$: the factor 2 of $V = 2 \Sigma_{\rm post} F \Sigma_{\rm post}$ no longer holds | -- | nothing | $\Sigma_{\rm rep}$ per axis against the nuisance floor (E7) |
| run ends before the ramp | early stopping on `val_npe` before $t = t_{\rm warm}$ | $t_{\rm warm}$, `patience` | nothing | the last logged `ramp` below 1 |

### 3.8 Findings this document owns

The table of record is `00_INDEX.md` S6; the rows below are this document's,
with what P3 adds to each.

- **F-e (owned).** The floor's stated rationale in `joint_space.py:276-281`,
  `:312-316` and plan S5.1 (v0.6.5) -- rank deficiency, "variance and never
  bias" -- is superseded by `[KB]` METRIC S3.13.1 and FINITE_DRAW S3.9: the
  rank bound of $\bar C$ is $S_{\rm mc} = 14$, and the floor's content is
  $\kappa_S \le 1.151$ (S3.3.4). The documentation patch of 2026-09-28 (plan
  v0.6.6, `replicate_statistic.py` and `joint_losses.py` docstrings, the
  error message) is pending; whether to correct for $\kappa_S$ at all is an
  open call of the decisions log (raised 2026-09-28) and a Stage 8
  candidate (D-038).
- **F-i (owned, with P5).** The four loss constants, `detach_metric`, the
  stop-gradient set, the diagnostic caps and the enumeration order of the
  diagnostic's pairs are configured by code and reachable by no flag
  (S3.1, S3.4).
- **F-o (owned, with P6).** `JointSpaceSpec.n_posterior_draws = (100, 400)`
  is below the floor at $d_\theta = 26$; only `default_joint_space` resolves
  it (S3.3.4).
- **F-p (owned).** Three defaults for one fraction: 0.3 (runner), 0.0 (pin),
  0.0 (library); the job passes none, so a cluster `A5` run ramps over 30%
  (S3.1, S3.3.3).
- **F-y (new).** The clamp of eq. (P3.7) zeroes the gradient of every pair
  with $\hat T_{gg'} \le T_{\rm floor}$, which under `correct_mc = True` is
  every pair with $\hat T^{\rm raw}_{gg'} \le d_\theta/S_{\rm mc} + 10^{-8}$:
  the collapse signature itself. Such a pair contributes a constant
  of 381 (at $\hat p_{\rm eff} = 3$) to the logged loss and nothing to the
  gradient. At a collapsed pair the fraction is 0.36-0.48 across the
  searched $S_{\rm mc}$ `[RAN]`, so the restoring force the plan credits to
  the two-sided target acts through the other half of the batch, whose
  gradient weights of eq. (P3.11) are then large. J11 tests loss values at
  $T_{\rm floor} = 10^{-30}$ and the uncorrected statistic; no test covers
  the gradient at the default floor with the correction. `[REPO]`
  `joint_losses.py:232-236, 324`; `[RAN]` blocks B5-B6; the consequence is
  [reasoning]. Status: report; open (D-038 candidate: mask dead-zone pairs
  out of the mean and count them, as $n_{\rm inv}$ is counted).
- **F-z (new).** A non-positive $\hat p_{\rm eff}$ is replaced by
  $p_{\rm min} = 10^{-3}$ and the pair stays in the mean, so the loss's slope in
  $\hat T_{gg'}$ is positive for every $\hat T_{gg'} > 10^{-3}$: an undefined
  target becomes a pull toward agreement, weighted by $r_{\rm rep}(t)$. The
  code's comment calls the target "undefined" and counts the case; the
  count is the only trace. `[REPO]` `joint_losses.py:312-325`; `[RAN]` block
  B5. Status: report; open (D-038 candidate: zero the per-pair loss when
  $\hat p_{\rm eff} \le 0$).
- **F-aa (new).** No surface refuses arm `A5` on a real bank without a
  same-donor pair: `needs_real` checks only that real shards exist
  (`run_joint_arms.py:342-344`), `rep_batch` returns `None` when
  $N_{\rm pair} = 0$ (`joint_batches.py:192-193`) and the loop skips the
  term (`joint_train.py:178-180`). The run completes as `A1`, logs `rep`
  0.0, `T` `nan` and `ramp` 1.0 from epoch 0, and writes a record without a
  `replicate` block. On the bench $N_{\rm pair} > 0$ by construction
  (`--wells-per-donor 2`); on the cohort it is D12. Status: report; open
  (D-038 candidate: raise when `lambda_rep > 0` and $N_{\rm pair} = 0$).
- **F-ab (new).** `enumerate_donor_pairs` pairs rows by `donor` only, and a
  row of every bank built so far is a window: $J$ windows per trace, one
  `donor` per donor, `--wells-per-donor` traces per donor
  (`build_latent_bank.py:216, 224-252`). At the Stage 1 job defaults (64
  traces, 2 wells per donor, 8 windows: `build_latent_bank.pbs:44-46`) a
  donor has 16 rows and 120 pairs, of which 56 (47%) are two windows of one
  well and 64 cross wells `[RAN]`; $N_{\rm pair} = 3840$ against 32 pairs of
  wells. The `well` field the bank carries (`:250`) is not read. The plan's
  pair is two wells (plan eq. (3); E0 convention 6), and the null
  expectation $p_{\rm eff}$ is derived for two recordings that are
  conditionally independent given $\theta^*$ (plan S2.5b; `[KB]` METRIC
  S3.7.3); two windows of one well share the realisation $\mathcal{G}$ and
  the nuisance draw $\nu$, so by the law of total variance their
  disagreement's covariance is at most the cross-well one less the
  realisation and nuisance parts -- the quantities the Stage 3c floors
  measure (E7) -- and their expected statistic sits below the target
  [reasoning]. For those pairs the two-sided loss pulls toward *more*
  disagreement; for the cross-well pairs it does what the plan intends; the
  batch mean mixes the two. On the cohort the fraction depends on how the
  real bank's `donor` field is populated (Stage 6, not written): if it is
  the culture, every pair is within-culture. `[REPO]` as cited; `[RAN]`
  the count (`tools/p3_numbers.py` B7); the consequence is [reasoning]. Status: report; open (D-038
  candidate: enumerate pairs by `donor` across distinct `well` values, or
  record the within-well fraction per run).
- **F-l, F-m, F-n (shared).** $B_{\rm rep}$ is not searched and not a job
  variable; the library and runner defaults for $B_{\rm rep}$ (8/4) and
  $S_{\rm mc}$ (256/128) differ; `stage3c` carries its own pair (128/64).

## 4. Summary of results

1. **Six surfaces, four axes, nine fixed knobs** (S3.1): the weight is
   derived from two axes through one flag; the warm-up fraction and the
   pair count are not passed by the job; the two Stage 4 pins differ from
   the Stage 3 defaults.
2. **The term as built**, eq. (P3.1)-(P3.9): reparameterised draws in box
   coordinates, `ddof=1` moments, a detached symmetrised and jittered
   metric solved by Cholesky, the $d_\theta/S_{\rm mc}$ subtraction, the
   affine target with $\Sigma_0 = I_{d_\theta}/12$, two clamps, a linear
   ramp on the planned budget, a 4-pair batch mean weighted by
   $\lambda_{\rm rep} r_{\rm rep}(t)$, and a gradient that reaches $\psi$
   only (S3.2).
3. **The gradient of the per-pair loss**, eq. (P3.11), is
   $2(\log \hat T_{gg'} - \log \max(\hat p_{\rm eff}, p_{\rm min}))/\hat T_{gg'}$
   above the floor and zero below it (S3.2).
4. **The ramp at the runner's defaults** completes at step 75 of 250; the
   logged value is the epoch's last (S3.2).
5. **The floor is not a rank bound**: $\bar C$ is full rank from
   $S_{\rm mc} = 14$; the floor fixes $\kappa_S = 1.151$ (S3.3.4).
6. **The subtraction leaves $\kappa_S$**, eq. (P3.12): 0.49 at the floor and
   0.38 at 128 draws against a target of 3, twice the term removed, and the
   expected loss is minimised at a disagreement below the target at every
   $S_{\rm mc}$ of the range; valid in the conjugate anchor with Gaussian
   draws (S3.3.4).
7. **The dead zone**: 36-48% of collapsed pairs give no gradient at the
   searched $S_{\rm mc}$ (F-y); **the invalid target** pulls toward
   agreement (F-z); **no pairs** means `A1` under the name `A5` (F-aa);
   **47% of bench pairs are within-well window pairs** (F-ab) (S3.7-S3.8).
8. **One inheritance from the DSN**: the warm-up range; the two ramps differ
   in what, how and whether they ramp (S3.5).
9. **Across surfaces, `T_mean` carries $S_{\rm mc}$**: 3% between 104 and
   128 draws at equal posterior (S3.6).

## 5. Open points, caveats, assumptions

**Assumed without proof.**

- Every $\kappa_S$ number is the conjugate anchor's value under Gaussian
  draws, equal covariances and $H_0$; for the trained flow the size is
  unmeasured and the direction is established only given independence of
  $\hat\Delta_{gg'}$ and $\bar C$ (`[KB]` FINITE_DRAW S3.9.6-S3.9.7). The
  dead-zone fractions of F-y were computed with Gaussian draws.
- The consequence of F-ab -- a within-well pair's expected statistic sits
  below the target -- is reasoning from the law of total variance; its size
  is set by the realisation and nuisance floors in posterior units and has
  not been computed.
- That $\hat T_{gg'}$ at initialisation is near its Monte Carlo excess
  (S3.3.3) assumes an untrained flow's mean barely depends on $z$, the
  plan's own premise for the ramp; no run has measured it.
- The textbook facts behind eq. (P3.12) are as the knowledge base tags
  them: not re-checked against a source here.

**Approximations and their regime.**

- Eq. (P3.12) and the loss-minimum column of S3.3.4 ignore the Jensen terms
  of the logarithm on both arguments (`[KB]` METRIC S3.13.1, S5).
- $\Sigma_0$ is the box's second moment used as a Gaussian covariance (plan
  S9); $\hat p_{\rm eff}$ is the target the code computes, not a measured
  count.

**Left unresolved.**

- Whether and how to correct for $\kappa_S$ (deflate before subtracting;
  a second batch for $\bar C$; raise the floor to 285): open call of the
  decisions log, D-038 candidate.
- The repository text F-e names is still the v0.6.5 text at `834eb41`; the
  v0.6.6 documentation patch is pending with the user.
- The repairs F-y, F-z, F-aa and F-ab suggest are D-038 candidates; none
  is decided.
- D12 (the cohort's design table), D13 (loss or diagnostic), D15 (which
  metric), D16 (means or the Bayes-factor form) remain open in the plan;
  D17 is closed as option (c) and its reopen condition stands.
- The inherited warm-up range $[0, 0.5]$ has no argument of its own for the
  replicate term (S3.5).
- Nothing in the record separates a real disagreement from $\kappa_S$, a
  dead-zone pair from a live one, or a within-well pair from a cross-well
  one; the per-pair `T` list of the diagnostic is the only handle.

## 6. References / further reading

**Project knowledge base `[KB]`.** `claude/METRIC_REPLICATE_v1_4.md` (S3.7.3,
S3.8-S3.9, S3.13-S3.15, S3.19, S5, S6); `claude/FINITE_DRAW_CORRECTION_v1.md`
(S3.1-S3.11; eqs. (69)-(74), (76), (79)-(91)); `JOINT_DSN_NPE_USAGE_v1.md`
S5.3 (repeats the floor claims, corrected alongside METRIC v1.4);
`claude/joint_docs/00_INDEX.md` S6 (F-e, F-i, F-l, F-m, F-n, F-o, F-p);
`claude/SBI_decisions_and_ideas_log.md` (D-035 to D-038, D-052; the open
call on $\kappa_S$).

**Repository `[REPO]`** at `834eb41`, read 2026-10-02: the files and lines of
the changelog row. The `sbi` 0.27.0 wheel, read from PyPI:
`sbi/neural_nets/net_builders/flow.py:1078-1171, 1278-1317`;
`sbi/neural_nets/estimators/zuko_flow.py:141-156`.

**Peer-reviewed literature.** According to PubMed, read in full text from
PubMed Central: Song T, Huang Z, Liu X, Sun J. *Preventing posterior collapse
with DVAE for text modeling.* Entropy 2025; PMID 40282658, PMC12026048,
[DOI](https://doi.org/10.3390/e27040423). Used in S3.3.3 for one qualitative
claim only: that ramping a regularising term's weight from 0 ("KL cost
annealing") is the standard mitigation of posterior collapse in variational
autoencoders and that the optimal schedule is reported to vary with dataset
and architecture. **No numeric value from that paper is used**, and the
paper concerns text-modelling VAEs, not SBI; the analogy is drawn here.

Retrieved, **abstract only, therefore not used for any claim**: Kenward MG,
Roger JH. *Small sample inference for fixed effects from restricted maximum
likelihood.* Biometrics 1997; PMID 9333350 (no DOI returned by the
connector). It addresses small-sample bias from an estimated covariance
inside a Wald-type quadratic form -- the problem of S3.3.4 -- in mixed
models; full text is not in PubMed Central. Full text not accessible for
PMID 9333350 -- this looks important for the $\kappa_S$ repair question; if
you can obtain the PDF, upload it and it can be folded in. Also returned and
not used: Kim J, Hwang Y, Sensors 2026, PMID 41977960, PMC13075281
([DOI](https://doi.org/10.3390/s26072176)), in PMC but about vibration
anomaly detection; its abstract mentions KL annealing for training stability
only.

**Searches run `[RAN]`, 2026-10-02.**

| source | query | result |
|---|---|---|
| PubMed | Hotelling T-squared small sample bias estimated covariance degrees of freedom correction | 0 records |
| PubMed | Hotelling T2 small sample covariance bias | 1 record (PMID 9333350; abstract only, flagged above) |
| PubMed | Mahalanobis distance plug-in sample covariance inverse bias finite sample Wishart | 0 records |
| PubMed | inverse Wishart expectation precision matrix bias correction | 0 records |
| PubMed | Monte Carlo error posterior mean number of draws | 1 record (PMID 25911600, expected value of sample information; off-topic, not used) |
| PubMed | Monte Carlo standard error Bayesian posterior summaries effective sample size | 1 record (PMID 40819155, adaptive trial design; off-topic, not used) |
| PubMed | test-retest reliability Bayesian posterior parameter estimates replicate recordings calibration | 0 records |
| PubMed | simulation-based inference replicate consistency posterior | 0 records |
| PubMed | biological replicates shared parameters Bayesian inference consistency mechanistic model posterior | 0 records |
| PubMed | auxiliary loss warm-up schedule weighting multi-task neural network training | 0 records |
| PubMed | loss weight ramp schedule neural network training stability | 0 records |
| PubMed | KL annealing warm-up variational autoencoder posterior collapse | 0 records |
| PubMed | KL annealing variational autoencoder | 2 records (PMIDs 40282658 full text read and used; 41977960 not used) |
| bioRxiv | neuroscience, last 30 days, 25 records (the connector has no keyword search) | none relevant; no preprint is cited |

Earlier searches on the same topics are in `[KB]` METRIC S6.4 and
FINITE_DRAW S6.5; they returned nothing further.

**Textbook, from memory** (not re-checked, as the knowledge base tags them):
the covariance of a sample mean of i.i.d. draws; the Wishart law of a
Gaussian scatter matrix and the mean of its inverse; the law of total
variance (F-ab).

---

### Pre-send check (Precision model)

R1 -- every symbol typed in S1; relations connect compatible types (the
clamps compare reals, $\mathcal{B}_{\rm rep} \subset \mathcal{U}$, the metric
is SPD). R2 -- eq. (P3.12) carries (a)-(c) and $n_{\rm W} > d_\theta + 1$;
the correction's exactness carries "exact metric"; the dead-zone fractions
carry "Gaussian draws, $\Delta_{gg'} = 0$"; the F-ab consequence carries
"by the law of total variance" and [reasoning]. R3 -- the transplant the
knowledge base identified (eq. (16) applied to the realised metric) is named
as such in S3.2 and S3.3.4; the VAE analogy is marked an analogy. R4 --
$\hat T^{\rm raw}_{gg'}$ and $\hat T_{gg'}$ are two objects with two
symbols; $r_{\rm rep}$ and $\gamma_{\rm sep}$ are two ramps; rows and wells
are named at every place the difference matters. R5 -- the map
$h_\psi: \mathbb{R}^{W} \to S^{E-1}$ and the draws' space $\Theta$ are named;
the box-coordinate check of the two samplers is in S3.2. R6 -- "collapse",
"warm-up", "well", "correction" each declared in S1.1 or S2. R7 -- the
plan's and the code's "variance, never bias" is quoted with its source and
its correction; the PubMed claim carries its domain. R8 -- each quantity
with two levels appears with both symbols and the move named (realisation
of the draws; estimation of $\bar C_{\rm true}$ by $\bar C$; plug-in of the
realised metric). Confirmation questions: none in the brief.
