# The Metric and the Null in the Replicate-Consistency Statistic

**What `T_gg'` is measuring, what it should equal when nothing is wrong, and
why that value is a count rather than zero**

**Date:** 2026-09-02. Version 1.1, superseding `METRIC_REPLICATE_v1.md`
(v1.0, 2026-08-30). Companion to `JOINT_DSN_NPE_PLAN_v0_5.md` (S2.5, arm A5),
`INFO_LOSS_THEORY_v1.md`, and `GATES_v1.md`. Append changelog entries rather
than editing in place.

---

**Abstract.** Two wells from one donor were the same biological preparation,
so whatever their mechanistic parameters are, they are the same parameters.
That is a constraint on an unknown quantity, and constraints on unknowns are
usable without ground truth and without a diagnostic label -- which matters
because every calibration gate in the pipeline needs `theta*` and therefore
cannot run on the real cohort at all. Turning the constraint into a number
requires two decisions that are usually made silently: how to collapse a
23-dimensional disagreement vector into a scalar (the **metric**), and what
value that scalar should take when nothing is wrong (the **null**). This
document argues that both decisions carry the entire scientific content. The
central results are (i) replicate disagreement is *not* largest along
poorly-constrained directions -- it vanishes at both extremes and peaks where
the data happens to be exactly as informative as the prior, a regime of no
scientific interest; (ii) dividing by the posterior's own covariance converts
that non-monotone raw disagreement into a per-direction *share* between 0 and
1, monotone in how strongly the data constrains that direction; and (iii) the
sum of those shares is `p_eff`, the effective number of constrained
directions, which is therefore the value two honest replicates should produce
-- not zero. Version 1.1 rewrites v1.0 so that every section states what the
object is *for* before stating what it *is*, adds the null hypothesis as an
explicit composite statement, separates quantities that are actually
**evaluated** in the pipeline from quantities that appear only in the
**derivation**, and adds a validated reference implementation. Deliberately
excluded: the choice between parameter-space and embedding-space constraints
(settled in the plan), the warm-up schedule (S3.11 only), the estimation of
the posterior itself, and the design of the encoder.

Claims are tagged **[REPO]** (read from source), **[KB]** (project knowledge
base), or **[reasoning]** (mine, no source). Corrections to earlier statements
-- including three inherited from v1.0 and two new to v1.1 -- are marked
**[CORRECTION]**.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| `theta` | parameter vector in inference coordinates | `theta` in `Theta` subset `R^{d}` | mixed; see Conventions | S3.2 |
| `theta^glob` | the globally-shared axes (neuron/synapse block) | subvector, `R^{23}` | mixed | S3.1 |
| `theta^topo` | the connectivity-kernel axes | subvector, `R^{3}` | mixed | S3.1 |
| `d` | dimension of the vector the statistic acts on; `d = 23` here | `N` | -- | S3.1 |
| `theta*` | the ground-truth parameter vector that generated a dataset; known only on the bench | `Theta` | mixed | S3.2 |
| `P` | a donor (patient); `g, g'` two wells belonging to `P` | index | -- | S3.1 |
| `x_g` | the recording of well `g` | `R^{W}` per window | Hz | S3.1 |
| `z_g` | the embedding (summary) of well `g` produced by the encoder | `R^{n_z}` | -- | S3.11 |
| `q_phi(theta \| x)` | the approximate posterior; `phi` its parameters | conditional density on `Theta` for each fixed `x` | -- | S3.2 |
| `q_omega` | the flow (density-estimator) component of `q_phi`; `omega` its parameters | -- | -- | S3.11 |
| `m_g` | posterior mean, `m_g = E_{q_phi}[theta^glob \| x_g]`, for each fixed `x_g` | `R^{23}` | mixed | S3.2 |
| `m_hat_g` | the `S`-draw Monte Carlo estimate of `m_g` | `R^{23}` | mixed | S3.9 |
| `S` | number of posterior draws used to form one `m_hat_g` | `N` | -- | S3.9 |
| `Delta_gg'` | disagreement vector, `Delta_gg' = m_g - m_{g'}` | `R^{23}` | mixed | S3.3 |
| `M` | a metric: symmetric positive-definite matrix | `23 x 23` | inverse of `theta^2` | S3.3 |
| `T_gg'` | the scalar statistic, `T_gg' = Delta_gg'^T M Delta_gg'` | `>= 0` | dimensionless if `M` chosen well | S3.3 |
| `H_0` | the null hypothesis, eq. (N1); a conjunction of two claims | -- | -- | S3.5 |
| `Sigma_0` | prior covariance of `theta` | PSD `23 x 23` | `theta^2` | S3.2 |
| `C` | posterior covariance, `C = Cov_{q_phi}(theta \| x)`, for one fixed `x` | PSD | `theta^2` | S3.2 |
| `C_g`, `C_g'` | `C` evaluated at `x_g` and at `x_g'` respectively | PSD | `theta^2` | S3.9 |
| `C_bar` | symmetrised posterior covariance, `C_bar = (C_g + C_g')/2` | PSD | `theta^2` | S3.9 |
| `F` | Fisher information of one well's data about `theta`, at `theta*` | PSD | `theta^{-2}` | S3.4 |
| `V` | covariance of `Delta_gg'` across replicate datasets at fixed `theta*` | PSD | `theta^2` | S3.4 |
| `theta_hat` | the maximum-likelihood estimate from one dataset | `R^{23}` | mixed | S3.4 |
| `lambda_j` | `j`-th generalised eigenvalue of `(F, Sigma_0^{-1})`; the data-to-prior precision ratio along direction `j` | `>= 0` | dimensionless | S3.4 |
| `v_j` | the corresponding generalised eigenvector (a *direction*, not an axis) | `R^{23}` | -- | S3.4 |
| `c_j` | prior's share of the posterior precision along direction `j`, `c_j = 1/(1 + lambda_j)` | `(0, 1]` | dimensionless | S3.4 |
| `1 - c_j` | the data's share along direction `j`, `= lambda_j/(1 + lambda_j)` | `[0, 1)` | dimensionless | S3.7 |
| `p_eff` | effective number of constrained directions, `p_eff = sum_j lambda_j/(1+lambda_j)` | `[0, d]` | dimensionless | S3.7 |
| `u` | prior-whitened coordinate, `u = Sigma_0^{-1/2} theta` | `R^{23}` | dimensionless | S3.6 |
| `zeta` | sampling-whitened disagreement, `zeta = V^{-1/2} Delta_gg'` | `R^{23}` | dimensionless | S3.8 |
| `A` | an invertible linear reparameterisation, `theta' = A theta` | `23 x 23`, `det A != 0` | -- | S3.3.3 |
| `I_d` | the `d x d` identity matrix | -- | -- | S3.8 |
| `L_rep` | the replicate loss, eq. (12) | `>= 0` | dimensionless | S3.10 |
| `chi^2_d` | chi-square distribution with `d` degrees of freedom | -- | -- | S3.8 |
| `Sigma_rep` | empirical covariance of `Delta_gg'` over donors; the nuisance floor | PSD | `theta^2` | S3.1 |
| `W_shr` | shrinkage weight in a regularised covariance estimate | `[0, 1]` | -- | S3.12 |
| `N_pair` | number of same-donor well pairs available | `N` | pairs | S3.12 |
| `LOG_PARAMS` | the index set of axes carried as natural logarithms | subset of `{0..29}` | -- | S3.3 |
| `a_k, b_k` | prior box bounds on axis `k`, in inference coordinates | reals | as `theta_k` | S3.3 |

### 1.1 Conventions

- **ASCII mathematical notation, inherited.** This document keeps v1.0's
  backtick ASCII style (`Sigma_0`, `tr(FC)`, `A^T`) rather than LaTeX, for
  consistency with the other project documents and so the file survives any
  transfer path. This is a deliberate deviation from the project's default
  document convention and is flagged here once rather than silently.
- **Inference coordinates.** `theta` always means the coordinate the SBI
  pipeline reasons in: natural log on `LOG_PARAMS` axes, natural units on the
  rest [REPO]. The simulator is always fed natural units via
  `theta_to_natural`; it never sees log coordinates.
- **"Axis" versus "direction".** An *axis* is a coordinate of `theta`. A
  *direction* is an arbitrary unit vector in `R^{23}`, in general not aligned
  with any axis. The distinction is load-bearing from S3.6 onward.
- **Every covariance is labelled by the randomness it averages over.** This is
  the single largest source of confusion in this material, and S3.2 is a table
  that does nothing else.
- **Evaluated versus derivation-only.** New in v1.1. Every quantity introduced
  in the main body is marked **[EVALUATED]** if the pipeline actually computes
  it, or **[DERIVATION ONLY]** if it appears solely in the reasoning that
  justifies a formula. `V`, `F` and `theta*` are all derivation-only on real
  data; conflating the two categories is what makes the construction look
  impossible when it is not. See S3.9.1.
- **Conditioning is carried.** `Var(m \| theta*)`, `C = Cov(theta \| x)`,
  `q_phi(theta \| x)`: the conditioning bar is never dropped, because the whole
  document turns on which variable is held fixed.
- `d = 23` throughout, the globally-shared block. The full parameter vector is
  30-dimensional in the simulator, of which 26 are active in the bank
  (23 neuron/synapse + 3 connectivity kernel) [REPO, KB].
- **Bold-free notation.** Matrices are capital Roman, vectors lower-case;
  `A^T` is transpose, `A^{-1}` inverse, `tr` trace, `A >= 0` positive
  semi-definite, `A > 0` positive definite.
- Quantifiers are stated explicitly: "for each fixed `theta*`", "for all
  invertible `A`", "in the limit `lambda_j -> 0`".

---

## 2. Glossary

Ordered alphabetically. Terms whose everyday meaning differs from the
technical one are flagged **[false friend]**.

**Composite null.** A null hypothesis that is a conjunction of two or more
claims, so that rejecting it does not say which conjunct failed. `H_0` here is
composite: shared parameters *and* a calibrated posterior. S3.5.

**Conjugate Gaussian case.** Prior Gaussian, likelihood Gaussian in the
parameters, so the posterior is Gaussian with closed-form mean and covariance.
Used here as an *anchor*: the only regime where every quantity can be written
down, and where the intuition is calibrated before being stressed against the
real case in S3.13.

**Curse of dimensionality.** A family of phenomena in high dimension. The one
that actually bites here is not distance concentration but the difficulty of
*estimating* a `23 x 23` covariance from few samples. S3.3.2, S3.12.

**Effective degrees of freedom / effective number of parameters.** The
standard statistical notion that a prior- or penalty-regularised fit consumes
a fractional number of degrees of freedom, each direction counted by how
strongly the data rather than the penalty determines it. `p_eff` is an
instance. Stated from memory; no source checked this session. S3.7.

**Effective number of constrained directions (`p_eff`).** How many independent
combinations of parameters the data actually pins down, counting each between
0 and 1 by how strongly it is constrained *relative to the prior*. S3.7.

**Fisher information (`F`).** The curvature of the *expected* log-likelihood
with respect to the parameters at `theta*`: how sharply one dataset
discriminates between nearby parameter values. A property of the *likelihood*,
not of the posterior. **[false friend]** It is the expectation over data of
the negative log-likelihood Hessian, not the Hessian at one observed dataset;
the two coincide only in expectation, and exactly (independently of `x`) in
the conjugate Gaussian model. S3.6.1.

**Mahalanobis distance.** A distance measured in units of a covariance:
`Delta^T Sigma^{-1} Delta`. Dimensionless, and invariant under invertible
linear reparameterisation. S3.3.3.

**Metric.** Here, the symmetric positive-definite matrix `M` used to turn a
vector into a scalar. **[false friend]** Not a metric in the metric-space
sense; the usage is that of a quadratic form.

**Null distribution.** The full law of the statistic under `H_0`, not merely
its mean. A mean gives a *target*; a law gives a *p-value*. Only `M = V^{-1}`
supplies the latter. S3.8.

**Null expectation.** The value `E[T_gg' | theta*, H_0]`. What "consistent"
means numerically. S3.8.

**Posterior covariance (`C`).** Uncertainty about `theta` given **one** fixed
dataset. S3.2.

**Posterior-mean sampling covariance (`Var(m | theta*)`).** Variability of the
*point estimate* across **repeated datasets** generated from the same fixed
`theta*`. Different object from `C`, and the confusion between them produced
an error corrected in S3.6.4. S3.2.

**Shrinkage estimator.** A covariance estimate that mixes the sample
covariance with a structured target (diagonal, or scaled identity) to make it
invertible and better-conditioned when samples are few. S3.12.

**Sloppy / stiff direction.** A direction in parameter space along which the
data is respectively weakly or strongly informative -- formally, small or
large generalised eigenvalue `lambda_j` of the Fisher information against the
prior precision. **[false friend]** "Sloppy" does *not* mean "noisy" or
"variable": along a sloppy direction the posterior mean is essentially the
prior mean, a constant, and replicates agree almost exactly. S3.6.4.

**Stop-gradient.** Blocking backpropagation through a sub-expression so the
optimiser cannot reduce a loss by manipulating that sub-expression. Used on
the flow parameters `omega` inside the replicate term. S3.11.

**Whitening.** Applying `Sigma^{-1/2}` so that a covariance becomes the
identity: decorrelation **plus** rescaling to unit variance, so each direction
is measured in units of its own standard deviation. Decorrelation alone
(rotation to the eigenbasis) would leave a diagonal, not identity, covariance.
Two different whitenings appear here -- prior (S3.6) and sampling (S3.8) --
and they must not be confused. S3.6.

---

## 3. Main body

### 3.1 Three aims, one vector

*Establishes that the same object serves three different purposes with three
different correct configurations, and that conflating them is the source of
most of the difficulty.*

**The problem this section solves.** Asking "what is the right metric for
`Delta_gg'`?" has no answer until you say what the number is for. The same
vector is used for three incompatible purposes.

The vector is always the same:

```
Delta_gg' = m_g - m_{g'} ,   for two wells g, g' of one donor P.           (1)
```

| # | aim | what it is | metric wanted | target |
|---|---|---|---|---|
| A | **Loss** (arm A5) | a training signal on real data | dimensionally coherent, stable during training, computable from a single well | `p_eff`, not zero (S3.7) |
| B | **Diagnostic** | a calibration check on real recordings, where no gate can run | a known null *distribution* | `chi^2_d` reference (S3.8) |
| C | **Measurement** | `Sigma_rep`, the empirical nuisance floor for stratification | **none** -- the raw covariance is the object | not applicable |

Aim C is separated out immediately because it explains a possible misreading:
in the plan's S2.7, `Sigma_rep = Cov(Delta_gg')` is used *as a whole matrix*,
not collapsed to a scalar, because the stratification solves a generalised
eigenproblem against it. No metric is chosen there. The metric question arises
only for aims A and B.

**Aim B is the one that justifies the whole construction.** Every gate -- G1,
G2, G3 -- requires `theta*`, so **no gate can run on the real cohort** [REPO,
`GATES_v1.md` S3.2]. The same is true of every standard calibration
diagnostic: SBC, expected coverage, TARP, and local classifier two-sample
testing all need `(theta*, x)` pairs [KB, project bibliography]. Replicate
consistency needs no `theta*`. If two wells of one donor disagree by far more
than their posteriors' stated uncertainty permits, the estimator is
overconfident on real recordings, and that is a statement nothing else in the
pipeline can currently make.

*Plain.* The same disagreement number can be a thing you train on, a thing you
check, or a thing you measure. They need different settings, and asking "what
is the right metric" without saying which of the three is meant has no answer.

### 3.2 Which randomness is which

*Establishes the distinctions that everything after this depends on. Read this
table before S3.6.*

**The problem this section solves.** Six different `23 x 23` covariance
matrices appear in this material, all in the same units, all written with the
same kind of symbol. They are routinely confused because the notation does not
say what is being averaged over. Here it does.

| object | averaged over | held fixed | meaning in words | status |
|---|---|---|---|---|
| `Sigma_0` | `theta ~ p(theta)` | nothing | how much parameters vary across the prior | **[EVALUATED]**, analytic |
| `C = Cov(theta \| x)` | `theta ~ q_phi(theta \| x)` | **one dataset** `x` | how uncertain we are about `theta` after seeing one recording | **[EVALUATED]**, one forward pass |
| `Var(m \| theta*)` | `x ~ p(x \| theta*)` | **one parameter** `theta*` | how much the point estimate moves if we re-run the experiment | **[DERIVATION ONLY]** |
| `V = Var(Delta_gg' \| theta*)` | two independent `x, x'` from `theta*` | `theta*` | how much two replicates should disagree | **[DERIVATION ONLY]** on real data; estimable on the bench |
| `Cov_i(m_i)` | `(theta, x) ~ p(theta, x)` | nothing | how much posterior means vary across the whole evaluation set | **[EVALUATED]** on the bench |
| `Cov(theta*)` | the ground truths in the evaluation set | nothing | the empirical prior covariance; estimates `Sigma_0` | **[EVALUATED]** on the bench |

Three consequences worth stating separately.

- `C` is the uncertainty about the parameters given **one observation** -- one
  well's recording. It is not a property of the cohort and not a property of
  the prior.
- `Cov(theta*)` in `information_spectrum` is the empirical covariance of the
  ground-truth parameter vectors over the evaluation set [REPO]. Since those
  `theta*` are prior draws, it estimates `Sigma_0`. It is *not* a posterior
  quantity, and it is *not* `V`.
- `V` is the covariance of the disagreement between two replicates, taken over
  repeated generation of the two datasets from a *fixed* `theta*`. It is what
  "how much should two honest replicates disagree" means quantitatively. It is
  not `C`; the relation between them is derived in S3.6.3.

### 3.3 From vector to scalar: the collapse is a modelling choice

*Establishes that the metric is not a formality.*

**The problem this section solves.** A loss and a test statistic both need a
number, so the 23-vector must be collapsed. There is no neutral way to do it.

```
T_gg' = Delta_gg'^T M Delta_gg' ,   M symmetric positive-definite.         (2)
```

Different `M` produce different scalars from the same `Delta_gg'`, hence
different gradients, hence a different trained model.

#### 3.3.1 Dimensional coherence, and what the log transform does and does not fix

The parameters enter logarithmically, and the source confirms it precisely
[REPO, `HPC_main_sweep.py`]. An axis is flagged logarithmic by a mechanical
rule -- both bounds strictly positive **and** spanning at least one decade:

```
LOG_PARAMS = { k : PARAM_BOUNDS[k,0] > 0 and log10(PARAM_BOUNDS[k,1]/PARAM_BOUNDS[k,0]) >= 1 }
```

with `LOG_BASE = 'natural'`, `theta = ln(value)` on those axes,
`theta_to_natural` inverting it before the simulator is called, and
`PARAM_BOUNDS_THETA` the box the SBI prior is defined over. The stated purpose
is a well-conditioned, order-one-scaled inference space rather than one mixing
`1e-4` with `5e2` [REPO].

**What this fixes.** A great deal. Without it, an unweighted Euclidean
distance would be dominated by whichever axis has the largest natural units,
and the loss would effectively be one-dimensional.

**What it does not fix.** Two things.

1. *The coordinates are still mixed.* Under the one-decade rule some axes are
   log and some linear -- the source itself notes that `U_0_sr`, `U_max` and
   `alpha_syn` (spanning exactly one decade) are log while `U_A` (0.95 decade)
   is linear [REPO]. A squared difference on a log axis is a squared
   log-ratio, dimensionless; on a linear axis it carries that parameter's
   units. Summing them adds unlike quantities, and the sum changes if an axis
   is reclassified -- a change that alters nothing physical.
2. *Equal scaling is not equal information.* Even if every axis were
   logarithmic with an identical prior box, the axes would still differ
   enormously in how well the data constrains them. That is the subtler
   failure, and it is S3.6.4.

**One practical consequence.** Because the prior is a *box* in `theta`
coordinates, `Sigma_0` is diagonal and analytic:
`Sigma_0[k,k] = (b_k - a_k)^2 / 12`. So the prior-whitened metric
`M = Sigma_0^{-1}` costs nothing to compute, needs no estimation, and removes
failure (1) completely. It is the correct *cheap* fallback [reasoning].
**[CORRECTION, new in v1.1]** v1.0 used this box moment matrix freely as
`Sigma_0` without flagging that the derivations in S3.6 treat the prior as
*Gaussian* with covariance `Sigma_0`. Substituting a uniform box's second
moment there is a separate approximation from the conjugacy one, and it should
be counted as such in S5.

#### 3.3.2 High dimension -- the real problem is not the one usually named

The classical curse-of-dimensionality result about Euclidean distance is
*distance concentration*: for random points in high dimension the ratio
`(d_max - d_min)/d_min` tends to zero, so nearest-neighbour contrast
disappears. That result is about **retrieval** -- ranking points by proximity
-- and it does not damage a quadratic penalty. The gradient of eq. (2) is
`2 M Delta_gg'`, well-behaved in any dimension.

Two genuine high-dimensional problems do bite.

**(i) Estimating `M` is the hard part, not applying it.** A `23 x 23`
covariance has 276 free parameters. If `V` is estimated from `N_pair`
same-donor well pairs and `N_pair` is of order ten, the sample covariance is
**singular** -- rank at most `N_pair - 1 < 23` -- and cannot be inverted at
all. Even when nominally invertible, inverting a poorly estimated covariance
amplifies noise along its smallest eigenvalues, exactly where the estimate is
least reliable. S3.12.

**(ii) The aggregate statistic cannot localise.** `T_gg'` is a sum of 23
terms. Under `H_0` with `M = V^{-1}` it is `chi^2_23`, whose relative
fluctuation is `sqrt(2/23) = 0.29`. A large deviation confined to one
direction contributes one term out of 23 and is diluted. Concentration is good
for *detecting* a diffuse failure and bad for *attributing* a focused one --
the same trade-off as expected coverage versus marginal SBC in `GATES_v1.md`
S3.6. Remedy: report the per-direction decomposition alongside the scalar,
since with `M = V^{-1}` the terms in the whitened basis are individually
`chi^2_1`.

#### 3.3.3 Invariance -- the property that makes the question well posed

*Establishes the criterion that decides among candidate metrics.*

Let `A` be any invertible linear reparameterisation, `theta' = A theta`. Then
`Delta_gg'' = A Delta_gg'` and any covariance transforms as `V' = A V A^T`.
Substituting into eq. (2) with `M = V^{-1}`:

```
T' = (A Delta_gg')^T (A V A^T)^{-1} (A Delta_gg')
   = Delta_gg'^T A^T A^{-T} V^{-1} A^{-1} A Delta_gg'
   = Delta_gg'^T V^{-1} Delta_gg'  =  T_gg' ,   for all invertible A.      (3)
```

**The Mahalanobis statistic is invariant under linear reparameterisation.**
The Euclidean one is not: `||A Delta||^2 != ||Delta||^2` in general. This
settles three things at once.

- It is the formal version of the dimensional argument in S3.3.1: a statistic
  that changes when you rescale an axis is measuring the coordinate system,
  not the disagreement.
- It dissolves the axis-versus-direction worry. Under a Mahalanobis metric the
  distinction stops mattering for the *value* of `T_gg'`: the statistic is the
  same computed in the original coordinates or in any rotated basis, including
  the basis in which sloppy directions are axis-aligned. Coordinates matter
  for *interpretation*, never for the number.
- It tells you which candidate metrics are admissible: only those that
  transform as an inverse covariance.

*Plain.* Measuring in units of the expected spread is what makes the answer
independent of the units you happened to store the parameters in.

### 3.4 What the statistic is being asked, in words, before any formalism

*Establishes the informal target that S3.5 to S3.8 make precise. This section
is new in v1.1.*

Two wells of one donor are the same biological preparation. The recording
process is noisy, so `x_g != x_{g'}` always, and therefore
`m_g != m_{g'}` always. The question is never "do they differ?" -- they do --
but **"do they differ by more than the model's own uncertainty says they
should?"**

That reframing has three immediate consequences, each of which becomes a
section below.

1. It requires a **yardstick**, and the only yardstick available on real data
   is the posterior's own covariance. Hence `M = (2C)^{-1}`: measure the gap
   in units of the posterior's stated width. S3.6, S3.9.
2. It requires a **reference value**, since "how much they should differ" is a
   number, and it is emphatically not zero. S3.7, S3.8.
3. It makes the whole construction **self-referential**: the model supplies
   both the disagreement and the standard it is judged against. That is not a
   flaw to be argued away but a structural property to be defended against,
   and S3.11 lists the three separate mechanisms that do so.

*Plain.* Two labs measure the same sample and report slightly different
numbers. To say whether they disagree you cannot just subtract: you have to
ask by how much, relative to the error bars they themselves quoted. A
one-millimetre gap between two metre rules is nothing; the same gap between
two interferometers is a scandal. The metric is the error bars.

### 3.5 The null hypothesis, stated explicitly

*Establishes what "nothing is wrong" means, and that it means two things at
once. This section is new in v1.1; v1.0 used the null without ever writing it
down.*

**The problem this section solves.** A number on its own says nothing. If a
donor's two wells give `T_gg' = 4.7`, is that good or bad? Unanswerable
without the value `T_gg'` would take if nothing were wrong. That value is the
null expectation, and it exists only relative to a stated hypothesis.

```
H_0 :   (i)  the two wells share one parameter vector,
             theta*_g = theta*_{g'} = theta* ,   and
        (ii) the posterior is calibrated at the relevant x,
             q_phi(theta | x) = p(theta | x) .                             (N1)
```

Under (i) and (ii), `Delta_gg' | theta*` has **mean zero** -- not because the
two means coincide, but because they have the same sampling distribution:

```
E[Delta_gg' | theta*] = E[m_g | theta*] - E[m_{g'} | theta*] = 0 .         (N2)
```

**[CORRECTION, new in v1.1]** It is easy to read "the wells share `theta*`" as
"the wells share a posterior mean". They do not. `m_g = m_g(x_g)` is a
function of that well's data, and `x_g != x_{g'}`. Shared `theta*` centres the
difference at zero; it does not annihilate it. If it did, there would be
nothing to test.

Two structural features of `H_0` matter downstream.

**It is composite.** `H_0` is a conjunction, so `T_gg' >> p_eff` is consistent
with a miscalibrated posterior *and* with a well-calibrated posterior on two
wells that legitimately differ in `theta^glob`. Rejection does not attribute
[reasoning]. The mitigation is already in the plan: estimate `Sigma_rep` per
axis under arm A1 first, see which axes actually replicate, and constrain only
those. Which axes replicate is a biological assumption *and it is measurable*.

**Conjunct (ii) is assumed on real data, not tested.** All the diagnostics
that could test it need `theta*` (S3.1). So calibration is verified on the
bench and then *assumed to transfer* to real `x`. Model misspecification is
precisely the failure of that transfer, and no amount of bench validation
detects it [KB, misspecification literature in the project bibliography]. This
is the load-bearing assumption of the whole construction and it is restated in
S5.

### 3.6 The conjugate Gaussian anchor

*Establishes closed forms for every object, in the one regime where they
exist, so that the null expectation can be computed at all.*

Everything in this section is **[DERIVATION ONLY]**: it justifies the formula
used in S3.9, and none of it is evaluated on real data.

#### 3.6.1 Information adds: `C^{-1} = Sigma_0^{-1} + F`

Prior precision plus data precision equals posterior precision. `Sigma_0^{-1}`
is the prior's information about location; `F` is one well's. **[CORRECTION,
inherited from v1.0]** An earlier informal explanation called `M` "the
information metric", implying `F`. What is available and what the construction
needs is the posterior precision `C^{-1}`. The two coincide only when
`F >> Sigma_0^{-1}` -- the diffuse-prior limit -- which on sloppy directions is
exactly false.

#### 3.6.2 The posterior mean, and what it is a mean over

`m_g = E[theta^glob | x_g]` is a mean of the **posterior distribution** at one
fixed `x_g` -- an average over `theta`, with `x_g` held fixed. In the conjugate
case,

```
m = C F theta_hat  =  (I_d - C Sigma_0^{-1}) theta_hat .                    (4)
```

so the posterior mean is the maximum-likelihood estimate **shrunk toward the
prior mean**, by an amount set by how much the prior contributes to the total
precision.

`Var(m | theta*)` is the mirror image: fix `theta*`, simulate a dataset,
compute `m`, repeat. The spread of those `m` values is the **sampling
distribution of the point estimate** -- an average over `x`, with `theta*`
held fixed.

#### 3.6.3 Deriving `V = 2 C F C`

From eq. (4), `m` is a fixed linear map applied to `theta_hat`, and
`Var(theta_hat | theta*) = F^{-1}`, so for each fixed `theta*`:

```
Var(m | theta*) = (C F) F^{-1} (C F)^T = C F C .                           (5)
```

Two wells of one donor are **conditionally independent given `theta*`**:
`p(x_g, x_{g'} | theta*) = p(x_g | theta*) p(x_{g'} | theta*)`. Hence the
covariance of the difference is twice that of one:

```
V = Var(m_g - m_{g'} | theta*) = 2 C F C .                                 (6)
```

The factor 2 **is** the conditional-independence assumption; it is not a
convention. Note that conditional independence is not marginal independence:
with `theta*` unknown, `x_g` and `x_{g'}` are strongly dependent, precisely
because they share it. Shared plate, medium batch or session induce residual
dependence at fixed `theta*`; then `Cov(m_g, m_{g'} | theta*) > 0`, the factor
2 over-states `V`, and `T_gg'` is inflated.

An equivalent form, substituting `F = C^{-1} - Sigma_0^{-1}`:

```
C F C = C (C^{-1} - Sigma_0^{-1}) C = C - C Sigma_0^{-1} C .               (7)
```

Eq. (7) is the useful one: `Var(m | theta*) <= C`, with the gap being exactly
the prior's contribution. **The point estimate moves less than the posterior
is wide**, because part of the posterior width comes from the prior, which
does not vary between replicates.

#### 3.6.4 The inverted U -- raw disagreement is largest in the middle

*This is the counter-intuitive result and the one most often got backwards.*

Prior-whiten: set `u = Sigma_0^{-1/2} theta`, so the prior precision is `I_d`,
and rotate to the generalised eigenbasis of `(F, Sigma_0^{-1})`, so the Fisher
information is `diag(lambda_1, ..., lambda_23)` and `C = diag(c_j)` with
`c_j = 1/(1 + lambda_j)`. The problem decouples into 23 independent scalar
problems. In these coordinates, from eqs. (5)-(7), for each direction `j`:

```
Var(m | theta*)_j = c_j (1 - c_j) = lambda_j / (1 + lambda_j)^2 .          (8)
```

This is an **inverted U**: maximal at `lambda_j = 1` (value 1/4) and vanishing
at **both** extremes.

- **Sloppy limit, `lambda_j -> 0`.** The posterior along `j` is the prior, so
  the posterior *mean* along `j` is the prior mean -- a constant, independent
  of `x_g` entirely. Both wells return the same number and the replicates
  agree **exactly**. Disagreement is zero, not large.
- **Stiff limit, `lambda_j -> infinity`.** The direction is pinned down
  precisely by both wells, so they agree closely again. Disagreement is small.
- **Middle, `lambda_j = 1`.** Data and prior are equally informative, and this
  is where raw disagreement peaks -- a regime of no scientific interest.

**[CORRECTION, inherited from v1.0]** An earlier explanation said the metric
makes divergence "along stiff directions expensive and along sloppy directions
free", justified by the claim that a sloppy-direction posterior mean wanders
over the prior range. That is **backwards**: if the posterior equals the prior,
the posterior *mean* equals the prior mean, a constant, so replicates agree
exactly. Eq. (8) is the correct statement.

**Why this matters operationally.** An unweighted (Euclidean) loss spends most
of its gradient at `lambda_j` near 1 -- on directions that are accidentally
balanced between prior and data. That is the real argument against `M = I`,
independent of the dimensional one in S3.3.1.

### 3.7 What the metric does: from wobble to a share, and why the total is a count

*Establishes the mechanism behind the null expectation. This section is
substantially expanded in v1.1; it is the explanation that was missing.*

**The problem this section solves.** `T_gg'` is a squared distance, with units
of (parameter units)^2. `p_eff` is a pure count. That a squared distance
should have a *counting number* as its expectation is not obvious, and the
reason is the whole content of the construction.

Continue in the whitened, diagonalised coordinates of S3.6.4. Direction `j`
contributes to `T_gg'` a weight `M_jj = (2 c_j)^{-1}` and carries a wobble
`V_jj = 2 c_j (1 - c_j)`. Their product is:

```
E[T_gg' | theta*]_j = M_jj V_jj = 2 c_j (1 - c_j) / (2 c_j)
                    = 1 - c_j = lambda_j / (1 + lambda_j) .               (11)
```

The `c_j` cancels. **That cancellation is the answer.** Reading eq. (11)
directly:

- **Data-dominated (stiff), `lambda_j -> infinity`:** `c_j -> 0`, contribution
  `-> 1`. The prior contributes nothing to the posterior width here, so *all*
  of the posterior's stated uncertainty is data uncertainty -- and data
  uncertainty is exactly what makes two independent wells disagree. Stated
  width and realised disagreement match: the direction is fully counted.
- **Prior-dominated (sloppy), `lambda_j -> 0`:** `c_j -> 1`, contribution
  `-> 0`. Zero absolute disagreement (S3.6.4) against a wide posterior. The
  direction contributes nothing **because there is nothing there to count**,
  not because disagreement is generously excused there. The metric is not a
  tolerance schedule.
- **Balanced, `lambda_j = 1`:** contribution `= 1/2`. The largest absolute
  disagreement in the whole spectrum, but only half the posterior width.

So `1 - c_j` is *the fraction of the posterior precision along direction `j`
that the data paid for*, and

```
p_eff = sum_{j=1}^{23} (1 - c_j) = sum_j lambda_j / (1 + lambda_j)         (10)
```

adds those fractions. It is a **soft count**: 23 directions, each counted with
weight equal to how much the data owns it, rather than 23 binary in-or-out
decisions. Non-integer values are normal.

**A second thing the metric does, not noted in v1.0** [reasoning]. The
numerator, eq. (8), is non-monotone in `lambda_j`; the normalised
contribution, eq. (11), is **monotone**, rising from 0 to 1. So dividing by
`2C` both makes the statistic reparameterisation-invariant *and* converts a
physically meaningless weighting into one where "more data-constrained" always
means "contributes more". Both properties are needed and only the first was
argued in v1.0.

**Why `p_eff` deserves the name "effective degrees of freedom".** Under
`M = V^{-1}` every direction is counted with weight exactly 1, giving
`tr(I_23) = 23` -- the full degrees of freedom of a whitened Gaussian.
Switching to `M = (2C)^{-1}` multiplies direction `j`'s weight by
`(2c_j)^{-1} / (2 c_j (1 - c_j))^{-1} = 1 - c_j`. So `p_eff` is literally
"23 degrees of freedom, each discounted by the data's share", which makes the
two admissible rows of the table in S3.8 one statement rather than two
[reasoning]. The functional form `sum_j lambda_j/(1 + lambda_j)` is the
standard effective-number-of-parameters expression familiar from ridge
regression and from DIC; stated from memory, no source checked this session.

**`p_eff` is prior-relative.** It counts directions where the data adds
information *beyond what the prior already supplied*. The same likelihood
against a tighter prior gives a smaller `p_eff`, because `lambda_j` is a
ratio. There is no absolute "number of constrained axes", and `p_eff` is not
the rank of `F`: a full-rank `F` with every `lambda_j = 0.01` gives
`p_eff = 0.23`.

**[CORRECTION, new in v1.1]** The contribution `1 - c_j` is an *expectation*,
not a score. A given realised `Delta_j` contributes `Delta_j^2 / (2 c_j)`,
which can take any non-negative value. Saying "a stiff direction contributes
1" is shorthand for "`E[M_jj V_jj] = 1 - c_j -> 1`" -- the product of a large
weight with the small wobble that actually occurs there. A single donor can
perfectly well produce `T_gg' = 40` against `p_eff = 3`; that is the
overconfidence signal, not a contradiction.

### 3.8 The candidate metrics and their nulls

*Establishes which `M` are admissible, and what each one buys.*

For `Delta_gg'` with mean zero and covariance `V`, and for any **fixed**
symmetric `M`,

```
E[Delta_gg'^T M Delta_gg' | theta*] = tr(M V) + mu^T M mu ,   mu = E[Delta_gg' | theta*] .   (14)
```

Proof in one line: `Delta^T M Delta = tr(M Delta Delta^T)`, and trace commutes
with expectation. Under `H_0`, `mu = 0`.

**[CORRECTION, new in v1.1]** v1.0 stated this result "for `Delta ~ N(0, V)`".
Gaussianity is **not** required for eq. (14): mean zero and finite second
moments suffice. Gaussianity is needed only for the `chi^2` law below. This
matters because it means the `p_eff` target survives the non-Gaussian case
more robustly than v1.0's phrasing implied -- what fails outside the conjugate
model is the closed form for `V`, not the trace identity.

| `M` | admissible? | `E[T_gg' | theta*]` under `H_0` | full null law | use |
|---|---|---|---|---|
| `I_d` (Euclidean) | **no** -- not invariant (S3.3.3) | `tr(V)`, no clean form | none | -- |
| `Sigma_0^{-1}` | yes; diagonal and analytic for a box | `tr(Sigma_0^{-1} V)`, no clean form | none | cheap fallback, no target |
| `(2C)^{-1}` | yes | `tr(F C) = p_eff` | approximate | **practical default** |
| `V^{-1}` | yes | exactly `d = 23` | `chi^2_23` exactly | exact, needs `V` |

**The `(2C)^{-1}` row, derived.** This is the row used in practice, and the
derivation is where the uncomputable objects disappear:

```
E[T_gg' | theta*] = tr( (2C)^{-1} . 2 C F C )        [substitute eq. (6)]
                  = tr(F C)                          [the 2 and one C cancel]
                  = tr( (C^{-1} - Sigma_0^{-1}) C )  [substitute F]
                  = tr(I_d) - tr(Sigma_0^{-1} C)
                  = 23 - tr(Sigma_0^{-1} C) = p_eff .                      (9)
```

`tr(I_d) = d = 23` is simply the trace of the identity: `C^{-1} C = I_d`.
Neither `V` nor `F` nor `theta*` survives to the right-hand side, which
contains only the known prior `Sigma_0` and the posterior covariance `C`.
**That is the entire reason `(2C)^{-1}` is the practical default.**

**The `V^{-1}` row.** Whiten by the sampling covariance:
`zeta = V^{-1/2} Delta_gg' ~ N(0, I_23)` under `H_0`, so
`T_gg' = zeta^T zeta ~ chi^2_23`, with mean 23 and variance 46. This is
strictly stronger -- a known *distribution*, hence a p-value per donor, which
is what aim B wants. Note this whitening is by `V`, not by `Sigma_0`; it is a
different transformation from the one in S3.6.4 and the two must not be
confused. The cost is that `V` must be inverted to form the statistic at all,
which S3.12 shows is usually impossible at the available `N_pair`.

**Why the factor 2 and why `C` rather than `C F C`.** Two separate reasons,
both worth seeing:

- The **2** is because `Delta_gg'` is a difference of two conditionally
  independent estimates, eq. (6).
- The **`C`** is an *approximation* to `Var(m | theta*) = C F C`, valid when
  `F >> Sigma_0^{-1}`, i.e. on well-constrained directions. It over-states the
  tolerable disagreement on sloppy directions by the factor
  `c_j / (c_j (1 - c_j)) = 1 / (1 - c_j)`, which diverges as `c_j -> 1`.
  Conservative -- it makes the statistic *less* likely to flag a problem --
  but it is a stated approximation, not an accident.

### 3.9 `p_eff` and the diagnostic already computed

*Establishes that the target is a number the pipeline already produces.*

The plan reports `information_spectrum` [REPO], the generalised eigenvalues of
`Cov_i(m_i)` against `Cov(theta*)`. By the law of total variance,
`Cov(theta) = E[Cov(theta | z)] + Cov_i(m_i)`, so `Cov_i(m_i) = Sigma_0 -
E[C]` and those generalised eigenvalues are

```
1 - c_j = lambda_j / (1 + lambda_j) ,                                     (11')
```

one per direction -- **exactly the per-direction terms of eq. (10)**. Their sum
is `p_eff`. So the target value for the replicate statistic is a number the
pipeline already measures, under a different name, for a different purpose.
Verified numerically in the non-diagonal correlated case [REPO, v0.5 R8].

Note carefully that `Cov_i(m_i)` averages over the *joint* (both `theta` and
`x` vary) while `Var(m | theta*)` fixes `theta*` -- the fifth and third rows of
the S3.2 table. They are different objects that happen to share the same
eigen-directions; the correspondence is not an identity of the matrices.

### 3.9.1 What is actually evaluated, and what is only reasoned through

*New in v1.1. This distinction resolves the objection that the construction
depends on a quantity that cannot be computed.*

| quantity | role | status on real data |
|---|---|---|
| `Sigma_0` | fixes the prior scale in eq. (9) | **[EVALUATED]** -- analytic from the prior box |
| `C_g, C_g', C_bar` | the metric, and the target via eq. (9) | **[EVALUATED]** -- one forward pass of the flow per well |
| `m_g, m_{g'}` | the disagreement vector | **[EVALUATED]** -- sample mean of `S` draws |
| `F` | eliminated between eqs. (6) and (9) | **[DERIVATION ONLY]** |
| `V` | justifies *why* the target is `p_eff` | **[DERIVATION ONLY]** on real data; estimable on the bench |
| `theta*` | defines the averaging in `V`; indexes `H_0` | **[DERIVATION ONLY]** on real data; chosen on the bench |
| `lambda_j, c_j` | interpretation of `p_eff` per direction | **[DERIVATION ONLY]**; the sum is evaluated via eq. (9) |

`V` is doing epistemic work, not numerical work. Without knowing that
`Var(Delta_gg' | theta*) = 2 C F C`, there would be no reason to believe
`E[T_gg']` equals anything in particular. The derivation goes through `V`; the
implementation never touches it.

### 3.10 The practical calculation

*Establishes exactly what to compute, per donor, per training step.*

**Recipe.**

1. **Draw posterior samples.** `{theta^(s)_g}_{s=1..S} ~ q_phi(theta^glob |
   x_g)`, and likewise for `g'`. One forward pass of the flow each; no
   `theta*`, no replicate ensemble.
2. **Moments.** `m_hat_g = S^{-1} sum_s theta^(s)_g`, and
   `C_hat_g = sample covariance of the draws`; same for `g'`.
3. **Symmetrise.** `C_bar = (C_hat_g + C_hat_g')/2`.               (17)
   **[CORRECTION, new in v1.1]** v1.0 and the plan both write a single `C`
   without saying at which well it is evaluated. Since `C = C(x_g)` for the
   trained flow, this is a genuine ambiguity. Symmetrising makes `T_gg'`
   exchangeable in `(g, g')`, which the unsymmetrised form is not
   [reasoning].
4. **Statistic.** `Delta_gg' = m_hat_g - m_hat_g'`; solve
   `(2 C_bar) z = Delta_gg'` by Cholesky and set `T_gg' = Delta_gg'^T z`.
   Never form `C_bar^{-1}` explicitly.
5. **Finite-`S` correction.** Each mean is estimated from `S` draws, so
   `m_hat = m + e` with `Var(e) = C/S`, and `Delta_hat` carries an extra
   covariance `2C/S`. Hence

   ```
   E[T_hat_gg'] = E[T_gg'] + tr( (2C)^{-1} . 2C/S ) = E[T_gg'] + d/S .    (16)
   ```

   Subtract `d/S = 23/S`. With `S = 1000` this is 0.023, negligible against
   `p_eff ~ 3`; with `S = 100` it is 0.23, roughly an 8% bias [reasoning,
   verified numerically -- test T5].
6. **Target.** `p_eff = 23 - tr(Sigma_0^{-1} C_bar)`, with
   `Sigma_0 = diag((b_k - a_k)^2 / 12)` for the box prior. Cross-check against
   `information_spectrum`.
7. **Loss / diagnostic.** eq. (12) below, with stop-gradient on the flow
   parameters and a warm-up ramp (S3.11).

Nothing in steps 1-7 touches `V`, `F`, or `theta*`.

**The loss.** Because the target is not zero, the penalty must be two-sided
about it:

```
L_rep = ( log T_gg' - log p_eff )^2 ,                                     (12)
```

or any smooth penalty with a minimum at `p_eff`. `T_gg' >> p_eff` means the
replicates disagree more than the posterior admits: overconfidence on real
data. `T_gg' << p_eff` means they agree more than the estimator's own
uncertainty allows: the estimator has stopped responding to the data. With 23
axes and `p_eff` measured at, say, 3, two honest replicates should produce
`T_gg' ~ 3` -- not 0, and not 23.

**The target moves during training.** Both `T_gg'` and `p_eff` are built from
the same forward pass, so `p_eff` is not a fixed constant: it tracks `C_bar`
as training changes the posterior. This is a feature for aim A and a
complication for aim B, where a stable reference is wanted.

### 3.11 Self-reference, and the three mechanisms that contain it

*Establishes why the construction does not trivially collapse. Substantially
corrected in v1.1.*

**The problem this section solves.** The target is built from the posterior's
own stated uncertainty, so in principle the model can satisfy the loss by
manipulating the yardstick rather than by improving the estimate. Three
distinct escape routes exist, and **three distinct mechanisms** close them.
They are commonly conflated -- assigning all the work to one of them is a
mistake [KB, plan v0.5].

| escape route | what the model would do | what blocks it |
|---|---|---|
| **Collapse** | drive `F -> 0`, so `V -> 0` and `T_gg' -> 0` under any fixed metric | the **non-zero target** `p_eff` in eq. (12): `T_gg' << p_eff` is penalised |
| **Inflation** | widen `C` so the metric `(2C)^{-1}` shrinks and `T_gg'` falls | the **NPE likelihood term**: an inflated `C` scores badly on held-out `(theta, x)` |
| **Desensitisation** | make `q_omega` less sensitive to its conditioner `z`, so `m_g ~ m_{g'}` whatever the input | the **stop-gradient on `omega`** in the replicate term, leaving only the wanted route `z_g ~ z_{g'}` via the encoder |

**[CORRECTION, inherited from v1.0]** v1.0 implied that the metric protects
against the degenerate solution. It does not: as `F -> 0` both `V -> 0` and
`T_gg' -> 0` under any fixed metric. Only the non-zero target and the NPE term
prevent degeneracy.

**[CORRECTION, new in v1.1]** The stop-gradient is *not* the mechanism that
prevents covariance inflation -- that is the NPE term. The stop-gradient
closes the third, separate route: reducing `T_gg'` by making the flow ignore
its conditioner, which is a G1 failure. Attributing the anti-inflation job to
the stop-gradient leaves the actual inflation route unguarded in one's mental
model.

A weak self-limiting effect is worth noting [reasoning]: widening `C_bar`
raises `p_eff` as well as lowering `T_gg'`, since both come from the same
matrix. That is a brake, not a guarantee; the NPE term does the real work.

**Warm-up.** Two independent reasons for it. First, `C` is estimated from the
posterior, so before the flow is meaningful the metric is undefined -- there is
nothing to normalise by. Second, at initialisation an untrained flow barely
depends on its conditioner, so `m_g ~ m_{g'}` already and `T_gg'` starts near
zero -- on the **collapse** side of the target. An early replicate term
therefore actively opposes learning.

**Staleness.** `C` moves during training. Recompute per epoch rather than per
step, and stop-gradient through the metric.

### 3.12 Estimating the metric with the data actually available

*Establishes the practical constraint, which is severe.*

`V` is `23 x 23`, so 276 free parameters. The sample covariance from `N_pair`
same-donor pairs has rank at most `N_pair - 1`, so with fewer than 24 pairs it
is **singular and cannot be inverted at all**. Whether the cohort supplies even
that many pairs is decision D12 in the plan, still open.

Three options, in order of preference given small `N_pair`:

1. **Estimate `V` on the bench, apply it to the real cohort.** Replicate wells
   with a known shared `theta*` can be simulated in any number. Transfers only
   under the assumption that the bench's information geometry resembles the
   cohort's -- which the misspecification gate says is questionable, so this
   must be stated rather than assumed.
2. **Use `(2C)^{-1}` from the posterior.** Needs no replicate pairs at all.
   Approximate (S3.8) and conservative. **This is the practical default.**
3. **Shrinkage.** If a sample `V` is used, regularise toward a diagonal
   target, `V_shr = (1 - W_shr) V_sample + W_shr diag(V_sample)`, with `W_shr`
   chosen by Ledoit-Wolf. Standard and well-tested in `scikit-learn`.

A **diagonal** metric is also worth considering. It sacrifices correlations
but is estimable from very few pairs and remains dimensionally coherent and
invariant under *axis-wise* rescaling, which is the transformation that
actually occurs (log versus linear). Given `N_pair` of order ten, a diagonal
metric estimated reliably may beat a full metric estimated badly [reasoning].

### 3.13 What survives outside the Gaussian case

*Answers the most serious objection to the whole construction.*

Everything in S3.6 assumed a Gaussian prior and a Gaussian likelihood. The
real posteriors are neither. Three consequences, in increasing severity.

**Correlations: no problem.** Nothing assumed `F` or `Sigma_0` diagonal. The
generalised eigenbasis handles arbitrary correlation automatically, and by
eq. (3) the statistic's value does not depend on the basis at all.

**Non-Gaussian but unimodal: mild.** The closed forms (4)-(11) fail, but the
*definitions* do not. `V = Var(Delta_gg' | theta*)` is a second moment and
needs no distributional assumption, and by the corrected eq. (14) the trace
identity survives without Gaussianity. `V` is directly estimable on the bench,
where replicate wells can be simulated from a known shared `theta*`. The
`chi^2_d` null becomes approximate, justified by a central-limit argument over
23 directions rather than by exact normality.

**Multimodality: this genuinely breaks the construction.** If
`p(theta | x_g)` is bimodal, the posterior *mean* sits between the modes, in a
region of low posterior mass -- it is not a summary of anything. Two replicates
may land on different modes and produce an enormous `Delta_gg'` that reflects
degeneracy, not miscalibration; or land on the same mode and agree spuriously.
The failure is in the choice of **summary**, not of metric. The principled
repair is to stop comparing point estimates and ask directly whether the two
posteriors are consistent with a common `theta`, using the pooled posterior

```
p(theta | x_g, x_g') proportional to p(theta | x_g) p(theta | x_g') / p(theta) , (13)
```

and a Bayes factor for "same theta" against "different theta". That statistic
handles multimodality correctly. A cheaper intermediate: draw
`theta ~ p(theta | x_g)` and evaluate its log-density under
`p(theta | x_g')`, symmetrised over the two orderings.

### 3.14 What the bench check actually tests

*New in v1.1. Clarifies a point that is easy to get exactly backwards.*

The plan requires validating `E[T_gg'] = p_eff` on simulated pairs at known
`theta*`. It is tempting to describe this as "checking the conjugate Gaussian
assumption". It is not, and it could not be: the real model is *known* not to
be conjugate Gaussian, so testing that assumption would be pointless.

What is being tested is whether the **conclusion** derived under that
assumption survives anyway. Eq. (14) needs only mean-zero and finite second
moments, so the only thing Gaussianity buys is the closed form `V = 2 C F C`.
The bench measures whether that closed form is close enough for
`tr((2C)^{-1} V) = p_eff` to hold in practice. If the empirical mean lands far
from `p_eff`, the target is wrong *for this simulator* and must be replaced by
the measured value -- not by a better distributional assumption [reasoning].

Order matters: verify calibration first (G1-G3, SBC, coverage, TARP, L-C2ST),
*then* check `E[T_gg'] = p_eff`. On a miscalibrated arm a discrepancy is
unattributable between "wrong target" and "bad posterior".

The bench also permits the stronger test that the real cohort cannot support:
with arbitrarily many simulated pairs, `V` itself can be estimated as a sample
covariance and the full `chi^2_23` null under `M = V^{-1}` checked by a
goodness-of-fit test -- turning "does the mean match?" into "does the whole
distribution match?".

### 3.15 Reference implementation and smoke test

*Establishes that the formulas above have been checked numerically.*

Two files accompany this document, pure ASCII and LF-only:

- `replicate_statistic.py` -- estimator logic only, with no I/O, no flow
  evaluation and no plotting. Provides `posterior_moments`,
  `replicate_statistic` (Cholesky solve, optional `d/S` correction),
  `p_eff_from_trace`, `p_eff_from_spectrum`, `box_prior_covariance`,
  `replicate_loss`, and bench-only helpers (`conjugate_posterior_covariance`,
  `simulate_replicate_pairs`, `sampling_covariance_V`).
- `smoke_test_replicate_statistic.py` -- 14 checks, all passing, exiting
  non-zero on failure so it drops into the cluster-side verification block.

| test | what it checks | section |
|---|---|---|
| T1 | `23 - tr(Sigma_0^{-1} C)` equals `sum_j lambda_j/(1+lambda_j)`, non-diagonal correlated case | S3.8 eq. (9), (10) |
| T2 | `E[T_gg'] = p_eff` under `M = (2C)^{-1}` -- **the bench check the plan requires** | S3.14 |
| T3 | `E[T] = 23`, `Var[T] = 46`, and `T ~ chi^2_23` by Kolmogorov-Smirnov, under `M = V^{-1}` | S3.8 |
| T4 | invariance under `theta -> A theta`, and that Euclidean `M = I_d` fails it | S3.3.3 eq. (3) |
| T5 | the `d/S` inflation and that subtracting it recovers `p_eff` | S3.10 eq. (16) |
| T6 | sloppy limit (`p_eff -> 0`, `Delta -> 0`) and stiff limit (`p_eff -> d`) | S3.6.4, S3.7 |
| T7, T8 | two-sided loss shape; end-to-end run from posterior samples | S3.10 eq. (12) |

T6a is the numerical confirmation of the counter-intuitive claim: with
`F -> 0`, `max_j |Delta_j|` is at machine level, i.e. sloppy directions
produce *no* replicate disagreement.

### 3.16 Failure modes

**Euclidean metric.** Dimensionally incoherent (S3.3.1); not invariant
(S3.3.3); gradient concentrated on accidentally-intermediate directions
(S3.6.4).

**Right metric, wrong target.** Driving `T_gg' -> 0` retains a degenerate
solution (S3.11).

**Composite null.** Rejection does not attribute between miscalibration and
genuinely different `theta^glob` across wells (S3.5).

**Data-dependent metric.** `M = (2 C_bar)^{-1}` with `C_bar = C_bar(x_g,
x_g')` is not a *fixed* metric, which eq. (14) assumes. In the conjugate
Gaussian model `C` is data-independent and the issue does not arise; for a
trained flow, `M` and `Delta_gg'` are functions of the same data and are
dependent, so `E[T_gg'] = tr(MV)` holds only to the extent `C(x_g)`
concentrates. Unquantified [reasoning]; not raised in v1.0.

**Stale metric.** `C` moves during training (S3.11).

**Singular or ill-conditioned metric.** Inverting a badly estimated `V`
amplifies noise along its least-reliable eigenvalues (S3.12).

**Mean-based summary under multimodality.** S3.13.

**Prior misspecified as Gaussian.** Eq. (9) treats the prior as Gaussian with
covariance `Sigma_0`; the actual prior is a uniform box (S3.3.1).

### 3.17 Commonly confused

| this | not this | why it matters |
|---|---|---|
| `C`, posterior covariance | `V`, replicate-disagreement covariance | `V = 2CFC <= 2C`; using `C` over-states tolerable disagreement on sloppy directions |
| `F`, Fisher information | `C^{-1}`, posterior precision | `C^{-1} = Sigma_0^{-1} + F`; equal only in the diffuse-prior limit |
| `F`, expected information | the log-likelihood Hessian at one `x` | `F` is an expectation over data at `theta*` |
| posterior *width* | posterior *mean's* variability across datasets | they move in **opposite** directions with `lambda_j` (S3.6.4) |
| null expectation | null distribution | a mean gives a target; only `V^{-1}` gives a p-value |
| null value | zero | the null is `p_eff`; too much agreement is a symptom |
| shared `theta*` | shared posterior mean | shared `theta*` centres `Delta_gg'` at zero, it does not annihilate it |
| conditional independence given `theta*` | marginal independence | `x_g` and `x_g'` are strongly dependent when `theta*` is unknown |
| whitening | decorrelation | whitening also rescales to unit variance |
| prior whitening (`Sigma_0^{-1/2}`) | sampling whitening (`V^{-1/2}`) | different transformations, different sections |
| axis (a coordinate) | direction (a generalised eigenvector) | sloppy directions are not axis-aligned; the statistic is basis-free, the interpretation is not |
| `p_eff` | rank of `F`, or number of identifiable parameters | `p_eff` is a soft, prior-relative count |
| `Cov(theta*)`, empirical prior covariance | any posterior covariance | it is the denominator of `information_spectrum`, estimating `Sigma_0` |
| metric on parameters | metric on embeddings | the whole construction lives in `Theta` |

### 3.18 Plain restatement

Two labs measure the same sample and report slightly different numbers. To say
whether they disagree you cannot just subtract: you have to ask by how much,
relative to the error bars they themselves quoted. A one-millimetre gap
between two metre rules is nothing; the same gap between two interferometers
is a scandal. The metric is the error bars.

Two further points that are less obvious. First, on a quantity neither
instrument can measure at all, both will report the same default value, so
they will agree perfectly while knowing nothing -- agreement is not evidence of
competence. Second, two *honest* instruments should not agree perfectly; they
should agree about as well as their error bars say they can. So the target is
not zero disagreement but the right amount of disagreement, and being too
consistent is as much a symptom as being too inconsistent.

And the reason that "right amount" turns out to be a count: each of the 23
directions in parameter space is owned partly by the prior and partly by the
data. Where the data owns a direction outright, two wells disagree along it by
exactly as much as the posterior's error bar says -- because that error bar
*is* data noise. Where the prior owns it, both wells return the prior and
agree perfectly. Measuring in units of the posterior's own error bar turns
each direction's contribution into an ownership share between 0 and 1. Add up
23 shares and you get how many directions' worth of information the data
actually supplied. That sum is `p_eff`.

---

## 4. Summary of results

1. **Three aims, three configurations** (S3.1): the same `Delta_gg'` serves as
   a training loss, a real-data diagnostic, and the raw `Sigma_rep` used for
   stratification. Only the first two need a metric.
2. **No calibration diagnostic runs on the real cohort** (S3.1): SBC,
   coverage, TARP, L-C2ST and gates G1-G3 all require `theta*`. Replicate
   consistency does not. That is the justification for the whole construction.
3. **Every covariance must be labelled by the randomness it averages over**
   (S3.2).
4. **`T_gg' = Delta_gg'^T M Delta_gg'`**, eq. (2), and `M` is a modelling
   choice with real consequences (S3.3).
5. **Mahalanobis is invariant under linear reparameterisation**, eq. (3),
   which is the criterion that selects it (S3.3.3).
6. **The null is composite**, eq. (N1): shared parameters *and* calibration.
   Rejection does not attribute (S3.5).
7. **Shared `theta*` gives `E[Delta_gg' | theta*] = 0`**, eq. (N2), not
   `Delta_gg' = 0` (S3.5).
8. **Information adds**, `C^{-1} = Sigma_0^{-1} + F`; **the posterior mean is
   the shrunk MLE**, eq. (4); **`V = 2 C F C = 2(C - C Sigma_0^{-1} C)`**,
   eqs. (5)-(7) (S3.6).
9. **Raw replicate disagreement is an inverted U in `lambda_j`**, eq. (8),
   maximal at `lambda_j = 1` and vanishing at both extremes. Sloppy directions
   produce *no* disagreement (S3.6.4).
10. **Dividing by `2C` yields a per-direction share**, eq. (11),
    `1 - c_j = lambda_j/(1 + lambda_j)`, which is monotone in `lambda_j`
    unlike the numerator (S3.7).
11. **The null expectation is `p_eff`** for `M = (2C)^{-1}`, eq. (9), and
    exactly `chi^2_23` for `M = V^{-1}` (S3.8).
12. **`E[Delta^T M Delta] = tr(MV)` needs only mean zero and finite second
    moments**, eq. (14) -- not Gaussianity (S3.8).
13. **`p_eff` is the sum of the generalised eigenvalues that
    `information_spectrum` already reports**, eqs. (10), (11') (S3.9).
14. **`V`, `F` and `theta*` are derivation-only**; the evaluated formula is
    `p_eff = 23 - tr(Sigma_0^{-1} C_bar)` (S3.9.1).
15. **The target is not zero**, eq. (12). Too much agreement is the collapse
    signature; too little is overconfidence on real data (S3.10).
16. **Finite posterior sampling inflates `T_gg'` by `d/S`**, eq. (16)
    (S3.10).
17. **Three escape routes, three distinct mechanisms**: non-zero target
    against collapse, NPE term against inflation, stop-gradient against
    desensitisation (S3.11).
18. **Multimodality breaks the mean-based statistic**, and the repair is to
    compare posteriors rather than point estimates, eq. (13) (S3.13).
19. **The bench check tests the conclusion, not the assumption** (S3.14).

---

## 5. Open points, caveats, and assumptions

**Assumed without proof.**

- That two wells of one donor share `theta^glob` exactly -- conjunct (i) of
  eq. (N1). This is an *assumption about biology*. Which axes actually
  replicate is measurable: it is what `Sigma_rep` estimates per axis, so the
  correct order is to measure first under arm A1 and constrain only the axes
  that replicate. Assuming the partition in advance risks penalising correct
  behaviour on axes that legitimately differ.
- That calibration verified on the bench transfers to real `x` -- conjunct
  (ii). This is the load-bearing assumption of the whole construction, and
  model misspecification is exactly its failure. No bench validation detects
  it.
- That `theta^topo` should be exempt. This conflates the connectivity
  *distribution* (the Weibull kernel parameters, which are `theta`) with its
  *realisation* (which is not a parameter). Two wells of one donor plausibly
  share the distribution. The exemption should be decided by measurement.
- That the wells are conditionally independent given `theta*`, used in eq.
  (13) and in the factor 2 of eq. (6). Shared plate, medium batch and session
  correlate them; the factor is then an under-estimate and `T_gg'` is
  inflated.

**Approximations and their regime of validity.**

- Eqs. (4)-(11) are exact only in the conjugate Gaussian model; elsewhere they
  are qualitative and the quantitative targets must be validated on the bench
  (S3.14).
- `(2C)^{-1}` approximates `V^{-1}` well when `F >> Sigma_0^{-1}` and is
  conservative by a factor `1/(1 - c_j)` per direction otherwise.
- The `chi^2_d` null additionally assumes `Delta_gg'` is Gaussian, justified
  by a central-limit argument over directions rather than by exact normality.
- `Sigma_0` is used as a Gaussian prior covariance although the prior is a
  uniform box; the box's second moment is substituted for it (S3.3.1).
- `M = (2 C_bar)^{-1}` is data-dependent, violating the fixed-`M` assumption
  of eq. (14). Unquantified (S3.16).

**Left unresolved.**

- Whether `N_pair` is large enough for any full-covariance metric (D12).
- Whether to use the bench-estimated `V` on real data, given that the
  misspecification gate says the two distributions differ decisively.
- The size of the bias introduced by the data-dependence of `C_bar`. A bench
  experiment with a trained flow rather than the analytic conjugate model
  would settle it.
- Whether `p_eff` computed from a box prior treated as Gaussian is close
  enough to the quantity eq. (9) intends. Measurable on the bench.
- Whether posterior multimodality actually occurs at the operating point. A
  mode count per posterior would settle whether eq. (13) is necessary or a
  precaution.
- No power analysis: the probability that `T_gg'` detects a given degree of
  overconfidence at the available `N_pair` is unknown.
- Whether aim B (diagnostic) should be adopted regardless of aim A (loss). My
  view is yes -- it is the only calibration statistic available on real
  recordings -- but that is a recommendation, not a result.

---

## 6. References and provenance

**Read from source [REPO].**
`Astro-Neuron-Network@main`,
`hpc/Phenomenological_finalv1/Burst Tests/HPC_main_sweep.py`: the `LOG_PARAMS`
rule (both bounds positive and span at least one decade), `LOG_BASE =
'natural'`, `PARAM_BOUNDS_THETA`, `theta_to_natural` / `natural_to_theta`,
`sample_theta`, and the rationale for log parametrisation including the
`U_0_sr` / `U_max` / `alpha_syn` versus `U_A` asymmetry.
`Simulation-Based-Inference@feat/misspec-gate`, `hpc/npe_diagnostics.py`:
`information_spectrum` (generalised eigenvalues of `Cov_i(E[theta|z_i])`
against `Cov(theta*)`), `posterior_contraction`.
`hpc/npe_tune_gates.py`: that G1, G2 and G3 all require `theta*`.

**Knowledge base [KB].** `SBI_PIPELINE.md` -- the 26 active axes (17 log, 9
linear), the prior box, the misspecification verdict. `JOINT_DSN_NPE_PLAN_v0_5.md`
-- eqs. (3a)-(3c), the `p_eff` target, the stop-gradient and warm-up, result R8.
`GATES_v1.md` S3.2, S3.6. `INFO_LOSS_THEORY_v1.md` S3.5.4. The project
bibliography for calibration diagnostics requiring `theta*` (SBC, expected
coverage, TARP, local/global coverage tests, L-C2ST) and for amortised-
inference misspecification; cited for orientation, with no numerical claim
taken from any of them.

**Stated from memory, not verified against a source.** That the Fisher
information of a Gaussian location family is the inverse covariance; that
`E[Delta^T M Delta] = tr(MV) + mu^T M mu` for `Delta` with mean `mu` and
covariance `V`; that `sum_j lambda_j/(1 + lambda_j)` is the standard
effective-degrees-of-freedom expression, as in ridge regression and DIC; the
Ledoit-Wolf shrinkage estimator and its availability in `scikit-learn`; the
law of total variance in the form
`Cov(theta) = E[Cov(theta|z)] + Cov(E[theta|z])`. All are textbook, and the
derivations that depend on them are given above, so the attributions can be
checked independently of the mathematics.

**Literature searches run for v1.1.** PubMed, three queries: Mahalanobis-form
null distributions for replicate/reproducibility statistics; calibration
diagnostics with a null hypothesis for neural posterior estimators; effective
degrees of freedom / effective number of parameters. **All three returned zero
records.** bioRxiv/medRxiv: the available connector supports only category and
date filtering, with no keyword search, so no targeted query was possible;
this is an observed limitation of the tool, not a prediction about coverage.
**No claim in this document is sourced from the peer-reviewed or preprint
literature.** The multimodality repair of eq. (13) is a standard Bayesian
construction, but whether it has been applied to replicate consistency in an
SBI setting has not been checked and it should not be presented as novel
without doing so.

**Code.** `replicate_statistic.py` and `smoke_test_replicate_statistic.py`,
written and executed this session; 14/14 tests pass; both files verified pure
ASCII, LF-only, and `py_compile`-clean.

---

## Changelog

- **2026-09-02 v1.1** -- rewrite for explanatory order: every section now
  states what the object is *for* before what it *is*. Substantive additions:
  the null hypothesis written out explicitly as a composite statement (S3.5);
  the informal statement of what is being asked (S3.4); the wobble-to-share
  mechanism and the effective-degrees-of-freedom reading of `p_eff` (S3.7);
  the evaluated-versus-derivation-only table (S3.9.1); the practical recipe
  and the finite-`S` correction (S3.10); the three-mechanism table for
  self-reference (S3.11); what the bench check actually tests (S3.14); the
  reference implementation and its smoke test (S3.15). New corrections:
  eq. (14) does not require Gaussianity; shared `theta*` does not mean shared
  posterior mean; the stop-gradient does not guard against covariance
  inflation (the NPE term does); `C` is data-dependent so `M` is not a fixed
  metric; the box prior is treated as Gaussian without that having been
  flagged in v1.0; `C` must be symmetrised across the two wells. Literature
  search provenance added to S6.
- **2026-08-30 v1.0** -- first version. Contains three corrections to the
  earlier informal explanation: the direction of the sloppy-axis argument, the
  conflation of Fisher information with posterior precision, and the
  implication that the metric protects against collapse. Adds the invariance
  criterion, the `p_eff` target and its identification with the information
  spectrum, the multimodality limitation and its repair, and the estimation
  constraints at the available sample size.
