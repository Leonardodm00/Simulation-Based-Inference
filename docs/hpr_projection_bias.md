# Why per-axis bounds read off posterior draws are biased inward at $p = 27$

**Date:** 17 August 2026

**Abstract.** We ask what a per-axis interval computed from posterior draws
actually estimates, and whether it estimates the quantity we intend. The
intended quantity is the projection onto each coordinate axis of the joint
highest-posterior-density region $M_{\epsilon}$ at level $1 - \epsilon$. The
computed quantity — the range of the retained draws along that axis — is an
extreme-order statistic, and it converges to the projection only at rate
$\sqrt{2 \log N}$ in the number of draws $N$. At $p = 27$ and $\epsilon =
10^{-3}$ the two differ by a factor of about two, and closing the gap would
require $N \approx 10^{12}$ draws. We then show that the intended quantity is
itself the wrong target: the box circumscribing the Gaussian
$1 - \epsilon$ ellipsoid at $p = 27$ has $6.0 \times 10^{11}$ times its volume
and retains a posterior mass numerically indistinguishable from one, so it
imposes no useful constraint. We conclude that neither the estimator nor its
estimand should be used, and that per-axis bounds should instead be calibrated
against a directly measured retained-mass target. **Covered:** the exact
projection for a Gaussian posterior, the extreme-value behaviour of the sample
range, the numerical gap at $p = 27$, the volume and mass accounting that
reframes the target, and the recommended alternative. **Deliberately
excluded:** the construction of the posterior $q_{\phi}$ itself, the choice of
$\epsilon$ within a sequential scheme such as TSNPE, sampling from a truncated
proposal, and any statement about the biophysical parameters of the simulator.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $p$ | Number of inferred parameters (axes) | $p \in \mathbb{N}$; here $p = 27$ | dimensionless | §3.1 |
| $j$ | Coordinate (axis) index | $j \in \{1, \dots, p\}$ | dimensionless | §3.1 |
| $\theta$ | Parameter vector | $\theta \in \Theta \subseteq \mathbb{R}^{p}$ | mixed (see Conventions) | §3.1 |
| $\theta_{j}$ | $j$-th coordinate of $\theta$ | $\theta_{j} \in \mathbb{R}$ | units of axis $j$ | §3.1 |
| $z$ | An observation (embedded trace) | $z \in \mathbb{R}^{E}$ | dimensionless | §3.1 |
| $E$ | Embedding dimension | $E \in \mathbb{N}$ | dimensionless | §3.1 |
| $q_{\phi}(\theta \mid z)$ | Approximate posterior density, for each fixed $z$ | $q_{\phi}(\cdot \mid z) : \mathbb{R}^{p} \to \mathbb{R}_{\geq 0}$, $\int q_{\phi}(\theta \mid z)\,d\theta = 1$ | density in $\theta$-units$^{-1}$ | §3.1 |
| $\phi$ | Trained parameters of the density estimator | $\phi \in \Phi$ | dimensionless | §3.1 |
| $\epsilon$ | Excluded posterior mass defining the HPR | $\epsilon \in (0,1)$ | dimensionless | §3.1 |
| $\tau_{\epsilon}(z)$ | Density threshold defining the HPR, for each fixed $z$ | $\tau_{\epsilon}(z) \in \mathbb{R}_{> 0}$ | same as $q_{\phi}$ | §3.1 |
| $M_{\epsilon}(z)$ | Highest-posterior-density region at level $1 - \epsilon$, for each fixed $z$ | $M_{\epsilon}(z) \subseteq \mathbb{R}^{p}$ | — | §3.1 |
| $\Pi_{j}(\cdot)$ | Orthogonal projection of a set onto axis $j$ | $\Pi_{j} : 2^{\mathbb{R}^{p}} \to 2^{\mathbb{R}}$ | — | §3.1 |
| $\mu$ | Mean of a Gaussian posterior, for each fixed $z$ | $\mu \in \mathbb{R}^{p}$ | units of $\theta$ | §3.2 |
| $\mu_{j}$ | $j$-th coordinate of $\mu$ | $\mu_{j} \in \mathbb{R}$ | units of axis $j$ | §3.2 |
| $\Sigma$ | Covariance of a Gaussian posterior, for each fixed $z$ | $\Sigma \in \mathbb{R}^{p \times p}$, symmetric positive definite | units of $\theta$ squared | §3.2 |
| $\Sigma_{jj}$ | $j$-th diagonal entry of $\Sigma$ | $\Sigma_{jj} \in \mathbb{R}_{>0}$ | units of axis $j$ squared | §3.2 |
| $\sigma_{j}$ | Marginal standard deviation on axis $j$; $\sigma_{j} = \sqrt{\Sigma_{jj}}$ | $\sigma_{j} \in \mathbb{R}_{>0}$ | units of axis $j$ | §3.2 |
| $\Sigma^{1/2}$ | Symmetric positive-definite square root of $\Sigma$ | $\Sigma^{1/2} \in \mathbb{R}^{p \times p}$ | units of $\theta$ | §3.2 |
| $e_{j}$ | $j$-th standard basis vector of $\mathbb{R}^{p}$ | $e_{j} \in \{0,1\}^{p}$ | dimensionless | §3.2 |
| $u$ | Whitened coordinate, $u = \Sigma^{-1/2}(\theta - \mu)$ | $u \in \mathbb{R}^{p}$ | dimensionless | §3.2 |
| $\chi^{2}_{p}$ | Chi-squared distribution with $p$ degrees of freedom | law on $\mathbb{R}_{\geq 0}$ | dimensionless | §3.2 |
| $F^{-1}_{\chi^{2}_{p}}(\cdot)$ | Quantile function of $\chi^{2}_{p}$ | $F^{-1}_{\chi^{2}_{p}} : (0,1) \to \mathbb{R}_{\geq 0}$ | dimensionless | §3.2 |
| $R_{\epsilon}$ | Mahalanobis radius of the $1-\epsilon$ ellipsoid; $R_{\epsilon} = \sqrt{F^{-1}_{\chi^{2}_{p}}(1-\epsilon)}$ | $R_{\epsilon} \in \mathbb{R}_{>0}$ | dimensionless | §3.2 |
| $N$ | Number of posterior draws available, for each fixed $z$ | $N \in \mathbb{N}$ | dimensionless | §3.3 |
| $n$ | Draw index | $n \in \{1, \dots, N\}$ | dimensionless | §3.3 |
| $\theta^{(n)}$ | $n$-th posterior draw | $\theta^{(n)} \in \mathbb{R}^{p}$ | units of $\theta$ | §3.3 |
| $\widehat{M}_{\epsilon}$ | Retained draws: those with density above $\tau_{\epsilon}$ | finite subset of $\mathbb{R}^{p}$ | — | §3.3 |
| $a_{N}$ | Expected maximum of $N$ i.i.d. standard normals | $a_{N} \in \mathbb{R}_{>0}$ | dimensionless | §3.3 |
| $\Phi$ | Standard normal cumulative distribution function | $\Phi : \mathbb{R} \to (0,1)$ | dimensionless | §3.3 |
| $\Phi^{-1}$ | Standard normal quantile function | $\Phi^{-1} : (0,1) \to \mathbb{R}$ | dimensionless | §3.7 |
| $\rho_{N,\epsilon}$ | Contraction ratio $a_{N} / R_{\epsilon}$ | $\rho_{N,\epsilon} \in (0,1)$ in the regime of interest | dimensionless | §3.4 |
| $\mathrm{Vol}(\cdot)$ | Lebesgue measure on $\mathbb{R}^{p}$ | $\mathrm{Vol} : 2^{\mathbb{R}^{p}} \to \mathbb{R}_{\geq 0}$ | units of $\theta$ to the power $p$ | §3.6 |
| $\Gamma(\cdot)$ | Gamma function | $\Gamma : \mathbb{R}_{>0} \to \mathbb{R}_{>0}$ | dimensionless | §3.6 |
| $a$ | Per-axis half-width in units of $\sigma_{j}$ | $a \in \mathbb{R}_{>0}$ | dimensionless | §3.6 |
| $\mathcal{B}(a)$ | Axis-aligned box of half-width $a\sigma_{j}$ on each axis $j$, centred at $\mu$ | $\mathcal{B}(a) \subset \mathbb{R}^{p}$ | — | §3.6 |
| $m$ | Retained posterior mass of a box | $m \in (0,1)$ | dimensionless | §3.6 |
| $\widehat{m}$ | Empirical retained mass (Monte Carlo estimate of $m$) | $\widehat{m} \in [0,1]$ | dimensionless | §3.7 |
| $l_{j}, u_{j}$ | Lower and upper bound emitted for axis $j$ | $l_{j}, u_{j} \in \mathbb{R}$, $l_{j} < u_{j}$ | units of axis $j$ | §3.7 |
| $r$ | Recording (culture) index | $r \in \{1, \dots, R\}$ | dimensionless | §3.7 |
| $R$ | Number of recordings | $R \in \mathbb{N}$ | dimensionless | §3.7 |
| $\mathbb{1}\{\cdot\}$ | Indicator function | $\mathbb{1} : \{\text{true},\text{false}\} \to \{0,1\}$ | dimensionless | §3.7 |
| $\mathbb{E}[\cdot]$ | Expectation | operator | — | §3.3 |
| $\lVert \cdot \rVert$ | Euclidean norm on $\mathbb{R}^{p}$ | $\lVert \cdot \rVert : \mathbb{R}^{p} \to \mathbb{R}_{\geq 0}$ | units of argument | §3.2 |

### 1.1 Conventions

- **Symbol collision, flagged.** The letter $R$ carries two meanings and they are
  never used in the same expression: $R_{\epsilon}$ (always subscripted by
  $\epsilon$) is the Mahalanobis radius of §3.2, a dimensionless real number;
  bare $R$ in §3.7 is the number of recordings, a positive integer. Where both
  appear in one section the subscript is retained on every occurrence.
- **Vectors are columns.** $\theta \in \mathbb{R}^{p}$ is a column vector;
  $e_{j}^{\top}\theta = \theta_{j}$.
- **Logarithms are natural.** $\log \equiv \log_{e}$ throughout.
- **Index ranges.** $j$ runs over $\{1,\dots,p\}$, $n$ over $\{1,\dots,N\}$, $r$
  over $\{1,\dots,R\}$, in every case unless explicitly restricted.
- **Units of $\theta$.** The 27 axes are physically heterogeneous and some are
  stored in a logarithmic coordinate. Every statement below is made in the
  **stored** coordinate, whatever that is per axis; "units of axis $j$" means
  the units of the stored coordinate. Nothing in this note is invariant to
  changing that coordinate, which is itself one of the caveats (§5).
- **Conditioning is never dropped.** $q_{\phi}(\theta \mid z)$,
  $\tau_{\epsilon}(z)$ and $M_{\epsilon}(z)$ are written with their conditioning
  argument at every occurrence. Where a single fixed $z$ is in force for a whole
  derivation this is stated at the start of the section rather than assumed.
- **Numerical values.** All numbers in §3.4, §3.6 and §4 were computed for
  $p = 27$ with `scipy.stats`; the expected maxima $a_{N}$ are Monte Carlo
  averages over independent replicates and are quoted to three significant
  figures.

---

## 2. Glossary

Ordered by first appearance, because the concepts build on each other.

**Highest-posterior-density region (HPR).** For a fixed observation $z$ and a
fixed level $1-\epsilon$, the smallest-volume set containing $1-\epsilon$ of the
posterior mass. Equivalently a super-level set of the density: all $\theta$
whose posterior density exceeds a threshold. Becomes operative in §3.1.

**Credible interval versus HPR projection (everyday meaning differs).** A
per-axis credible interval contains $1-\epsilon$ of the *marginal* mass on that
axis. The projection of the joint HPR is a different and generally much wider
object: it contains every value of $\theta_{j}$ attained by *any* point of the
joint region. Conflating the two is the most common error in this area. §3.1.

**Projection $\Pi_{j}$.** The shadow a set casts on axis $j$: the set of values
$\theta_{j}$ takes as $\theta$ ranges over the set. For a convex set it is an
interval. §3.1.

**Support function.** For a convex set $M$ and direction $e$, the quantity
$\sup_{\theta \in M} e^{\top}\theta$. Computing it in direction $e_{j}$ gives the
upper endpoint of $\Pi_{j}(M)$; this is the tool that makes §3.2 a two-line
derivation rather than a numerical exercise. §3.2.

**Mahalanobis radius.** The distance from the mean measured in units of the
posterior's own covariance, $\lVert \Sigma^{-1/2}(\theta - \mu)\rVert$. Level
sets of a Gaussian density are spheres in this metric. §3.2.

**Extreme-order statistic.** The maximum (or minimum) of a finite sample, as
opposed to a quantile. The distinction is the whole content of this note: the
sample maximum estimates the *support* of a distribution, not any fixed
quantile of it. §3.3.

**Gumbel / extreme-value limit.** The statement that the maximum of $N$ i.i.d.
Gaussian draws, suitably centred and scaled, converges in law as $N \to \infty$;
the centring grows like $\sqrt{2\log N}$, which is the logarithmically slow rate
responsible for the bias. Named limit law. §3.3.

**Concentration of measure (in the sense used here).** The fact that in high
dimension a ball occupies a vanishing fraction of the box that circumscribes it.
At $p = 27$ the fraction is $1.66 \times 10^{-12}$. §3.6.

**Retained mass.** The posterior probability that a proposed box actually
contains, $\int_{\mathcal{B}} q_{\phi}(\theta \mid z)\,d\theta$. The quantity we
argue should be targeted directly rather than inferred from a threshold. §3.7.

**Shrinkage.** The fraction of an axis's original prior range that survives
truncation. A value near one means the data did not constrain that axis. §3.7.

---

## 3. Main body

### 3.1 What we want, and what we compute

*This section establishes the two objects and shows they are not the same
object.*

Fix an observation $z$. Introduce the threshold $\tau_{\epsilon}(z) \in
\mathbb{R}_{>0}$, defined for each fixed $z$ as the largest value such that the
super-level set

$$M_{\epsilon}(z) \;=\; \bigl\{\, \theta \in \mathbb{R}^{p} \;:\; q_{\phi}(\theta \mid z) > \tau_{\epsilon}(z) \,\bigr\} \tag{1}$$

satisfies $\int_{M_{\epsilon}(z)} q_{\phi}(\theta \mid z)\,d\theta \geq 1 -
\epsilon$. This is the highest-posterior-density region. In practice
$\tau_{\epsilon}(z)$ is estimated as the $\epsilon$-quantile of the values
$q_{\phi}(\theta^{(n)} \mid z)$ over draws $\theta^{(n)} \sim q_{\phi}(\theta
\mid z)$, which is the rule used by TSNPE.

The object we *want*, for each fixed $z$ and each fixed $j$, is the projection

$$\Pi_{j}\bigl(M_{\epsilon}(z)\bigr) \;=\; \bigl\{\, \theta_{j} \;:\; \theta \in M_{\epsilon}(z) \,\bigr\} \;=\; \Bigl[\, \inf_{\theta \in M_{\epsilon}(z)} e_{j}^{\top}\theta, \;\; \sup_{\theta \in M_{\epsilon}(z)} e_{j}^{\top}\theta \,\Bigr]. \tag{2}$$

The object we *compute*, given $N$ draws $\theta^{(1)}, \dots, \theta^{(N)} \sim
q_{\phi}(\theta \mid z)$ and the retained subset

$$\widehat{M}_{\epsilon} \;=\; \bigl\{\, \theta^{(n)} \;:\; q_{\phi}(\theta^{(n)} \mid z) > \tau_{\epsilon}(z) \,\bigr\}, \tag{3}$$

is the sample range

$$\widehat{\Pi}_{j} \;=\; \Bigl[\, \min_{\theta^{(n)} \in \widehat{M}_{\epsilon}} \theta^{(n)}_{j}, \;\; \max_{\theta^{(n)} \in \widehat{M}_{\epsilon}} \theta^{(n)}_{j} \,\Bigr]. \tag{4}$$

Equation (4) is a *finite-sample extreme*, equation (2) a *population supremum*.
Since $\widehat{M}_{\epsilon} \subset M_{\epsilon}(z)$ almost surely, we have
$\widehat{\Pi}_{j} \subseteq \Pi_{j}(M_{\epsilon}(z))$ for every $j$ and every
$N$ — the estimator is **biased inward by construction, never outward**. The
rest of §3 quantifies by how much.

### 3.2 The exact projection, for a Gaussian posterior

*This section establishes that the true projection is $\pm R_{\epsilon}\sigma_{j}$
and derives $R_{\epsilon}$.*

Assume for this section that, for the fixed observation $z$,
$q_{\phi}(\theta \mid z) = \mathcal{N}(\theta \mid \mu, \Sigma)$ with $\mu \in
\mathbb{R}^{p}$ and $\Sigma \in \mathbb{R}^{p\times p}$ symmetric positive
definite. The Gaussian case is not the truth for a normalizing flow, but it is
the case in which every quantity is available in closed form, and §3.8 argues
the conclusion is not an artefact of it.

Because the Gaussian density is a strictly decreasing function of the
Mahalanobis distance, the super-level set (1) is the ellipsoid

$$M_{\epsilon}(z) \;=\; \bigl\{\, \theta \;:\; (\theta - \mu)^{\top}\Sigma^{-1}(\theta - \mu) \;\leq\; R_{\epsilon}^{2} \,\bigr\}, \qquad R_{\epsilon} \;=\; \sqrt{F^{-1}_{\chi^{2}_{p}}(1 - \epsilon)}, \tag{5}$$

the radius following from the fact that $(\theta - \mu)^{\top}\Sigma^{-1}(\theta
- \mu) \sim \chi^{2}_{p}$ when $\theta \sim \mathcal{N}(\mu, \Sigma)$.

Now compute the support function in direction $e_{j}$. Substitute the whitened
coordinate $u = \Sigma^{-1/2}(\theta - \mu)$, so that $\theta = \mu +
\Sigma^{1/2}u$ and $M_{\epsilon}(z)$ becomes the ball $\lVert u \rVert \leq
R_{\epsilon}$. Then, for each fixed $j$,

$$\sup_{\theta \in M_{\epsilon}(z)} e_{j}^{\top}\theta \;=\; \mu_{j} \;+\; \sup_{\lVert u \rVert \leq R_{\epsilon}} \bigl(\Sigma^{1/2}e_{j}\bigr)^{\top} u \;=\; \mu_{j} \;+\; R_{\epsilon}\,\bigl\lVert \Sigma^{1/2}e_{j} \bigr\rVert \;=\; \mu_{j} \;+\; R_{\epsilon}\,\sqrt{e_{j}^{\top}\Sigma e_{j}}, \tag{6}$$

by Cauchy-Schwarz, with equality attained at $u = R_{\epsilon}\,\Sigma^{1/2}e_{j}
/ \lVert \Sigma^{1/2}e_{j}\rVert$. Since $e_{j}^{\top}\Sigma e_{j} = \Sigma_{jj}
= \sigma_{j}^{2}$, and by symmetry for the infimum,

$$\boxed{\;\Pi_{j}\bigl(M_{\epsilon}(z)\bigr) \;=\; \bigl[\, \mu_{j} - R_{\epsilon}\sigma_{j}, \;\; \mu_{j} + R_{\epsilon}\sigma_{j} \,\bigr] \quad \text{for each fixed } j \in \{1,\dots,p\}.\;} \tag{7}$$

Note what (7) does **not** depend on: the off-diagonal structure of $\Sigma$
enters only through $\sigma_{j}$. The projection is blind to correlation, which
is the separate and well-known reason a box is a weak summary of a joint region;
that point is quantified in §3.6 but is not the subject of §3.3–3.5.

At $p = 27$: $R_{10^{-3}} = 7.448$, $R_{10^{-4}} = 7.948$, $R_{10^{-5}} =
8.387$. The true projection of the $99.9\%$ joint HPR is therefore about
$\pm 7.45\sigma_{j}$ on every axis.

### 3.3 What the sample range estimates instead

*This section establishes that the computed interval grows like
$\sqrt{2\log N}$, not like $R_{\epsilon}$.*

Under the same Gaussian assumption, the marginal law of coordinate $j$ is
$\theta_{j} \sim \mathcal{N}(\mu_{j}, \sigma_{j}^{2})$, so the $N$ draws
$\theta^{(1)}_{j}, \dots, \theta^{(N)}_{j}$ are i.i.d. Gaussian and the upper
endpoint of (4) is, up to the second-order correction discussed below,

$$\max_{n \in \{1,\dots,N\}} \theta^{(n)}_{j} \;\approx\; \mu_{j} \;+\; a_{N}\,\sigma_{j}, \qquad a_{N} \;=\; \mathbb{E}\Bigl[\max_{n} \xi_{n}\Bigr], \quad \xi_{1},\dots,\xi_{N} \;\text{i.i.d.}\; \mathcal{N}(0,1). \tag{8}$$

The classical extreme-value expansion gives, as $N \to \infty$,

$$a_{N} \;=\; \sqrt{2\log N} \;-\; \frac{\log\log N + \log 4\pi}{2\sqrt{2\log N}} \;+\; o\!\left(\frac{1}{\sqrt{\log N}}\right). \tag{9}$$

The leading term $\sqrt{2\log N}$ is the entire problem. It is not that the
estimator is noisy; it is that its **centre** sits at $\sqrt{2\log N}$ standard
deviations while the target sits at $R_{\epsilon}$, and these are different
numbers that happen to be of similar magnitude for small $p$ and modest $N$.

Two second-order effects, stated for completeness and both small relative to
(9). First, the threshold step in (3) removes the $\epsilon$-fraction of draws
with lowest density, i.e. those at largest Mahalanobis radius — which are
disproportionately the draws attaining extreme $\theta_{j}$. This makes the
computed interval slightly *narrower* still, reinforcing the bias rather than
offsetting it. Second, for $\epsilon N \lesssim 1$ no draw is removed at all and
$\widehat{\Pi}_{j}$ is exactly the full-sample range; at $\epsilon = 10^{-3}$
and $N = 10^{4}$ only about ten draws are discarded.

### 3.4 The gap, numerically, at $p = 27$

*This section establishes the size of the discrepancy.*

Define the contraction ratio $\rho_{N,\epsilon} = a_{N} / R_{\epsilon}$, the
factor by which the computed interval falls short of the true projection. For
$p = 27$ and $\epsilon = 10^{-3}$ (so $R_{\epsilon} = 7.448$):

| $N$ | $a_{N}$ (Monte Carlo) | $\sqrt{2\log N}$ | $\rho_{N,10^{-3}}$ | interval too narrow by |
|---|---|---|---|---|
| $10^{3}$ | 3.239 | 3.717 | 0.435 | 57% |
| $10^{4}$ | 3.853 | 4.292 | 0.517 | 48% |
| $10^{5}$ | 4.376 | 4.799 | 0.588 | 41% |
| $10^{6}$ | 4.853 | 5.257 | 0.652 | 35% |

At the draw counts one actually uses, **the per-axis interval read off the
retained draws is roughly half the width of the object it is supposed to
estimate.**

Equally important, and easier to act on: $a_{N}$ *moves with $N$*. Going from
$10^{4}$ to $10^{6}$ draws widens every bound by 26% for the same posterior,
the same $\epsilon$ and the same data. The estimator silently makes the number
of posterior draws a tuning parameter of the scientific result. Two runs of the
same pipeline with different draw budgets would emit different parameter
regions and neither would be flagged as wrong.

### 3.5 Why more draws cannot fix it

*This section establishes that the bias is not a budget problem.*

Setting the leading term of (9) equal to $R_{\epsilon}$ and solving for $N$
gives the draw count at which the sample range would, in expectation, reach the
true projection:

$$N_{\text{req}}(\epsilon) \;\approx\; \exp\!\left(\frac{R_{\epsilon}^{2}}{2}\right) \;=\; \exp\!\left(\frac{F^{-1}_{\chi^{2}_{p}}(1-\epsilon)}{2}\right). \tag{10}$$

At $p = 27$: $N_{\text{req}}(10^{-3}) = 1.11 \times 10^{12}$,
$N_{\text{req}}(10^{-4}) = 5.20 \times 10^{13}$, $N_{\text{req}}(10^{-5}) =
1.89 \times 10^{15}$.

The dependence in (10) is exponential in $R_{\epsilon}^{2}$, which is itself
roughly linear in $p$ (since $\mathbb{E}[\chi^{2}_{p}] = p$). The required draw
count is therefore **exponential in the number of parameters**. At $p = 27$ it
is already beyond any feasible budget, and no improvement in sampling
throughput changes that conclusion.

### 3.6 The target is wrong anyway

*This section establishes that even a perfect estimate of (7) would be useless,
and reframes what should be targeted.*

Suppose we could evaluate (7) exactly. Consider the resulting box
$\mathcal{B}(a) = \prod_{j=1}^{p} [\mu_{j} - a\sigma_{j}, \mu_{j} + a\sigma_{j}]$
at $a = R_{\epsilon}$. Two accountings show it constrains nothing.

**Volume.** For the isotropic case ($\Sigma = I$, without loss of generality
after whitening), the ratio of the ball of radius $R_{\epsilon}$ to the box that
circumscribes it is

$$\frac{\mathrm{Vol}\bigl(\{\lVert u \rVert \leq R_{\epsilon}\}\bigr)}{\mathrm{Vol}\bigl(\mathcal{B}(R_{\epsilon})\bigr)} \;=\; \frac{\pi^{p/2}}{\Gamma\!\left(\tfrac{p}{2}+1\right) 2^{p}}, \tag{11}$$

which is independent of $R_{\epsilon}$ and equals $1.661 \times 10^{-12}$ at
$p = 27$. **The box circumscribing the HPR has $6.0 \times 10^{11}$ times its
volume.** This is concentration of measure, and it is why "the projection of the
joint HPR" sounds like a tight object and is not one.

**Mass.** For independent coordinates the box $\mathcal{B}(a)$ retains

$$m(a) \;=\; \bigl(2\Phi(a) - 1\bigr)^{p} \qquad \text{for each fixed } a > 0. \tag{12}$$

At $a = R_{10^{-3}} = 7.448$ and $p = 27$, $m$ is numerically
indistinguishable from $1$ in double precision. So the exact projection of the
$99.9\%$ HPR is a box retaining essentially **all** posterior mass — it excludes
nothing and would truncate nothing.

Inverting (12) is far more informative. For a *target* retained mass $m$, the
required half-width is

$$a(m) \;=\; \Phi^{-1}\!\left(\frac{1 + m^{1/p}}{2}\right) \qquad \text{for each fixed } m \in (0,1). \tag{13}$$

At $p = 27$: $a(0.95) = 3.106$, $a(0.99) = 3.559$, $a(0.999) = 4.125$.

Compare these with the $a_{N}$ column of §3.4. The sample-range box at $N =
10^{4}$ sits at $a = 3.853$, i.e. between the $99\%$ and $99.9\%$ mass targets;
at $N = 10^{6}$ it sits at $a = 4.853$, beyond $99.9\%$. So as a *mass-retaining
box* the sample range is not absurd — it is simply **uncalibrated and
$N$-dependent**. Its failure is not that it is too narrow in absolute terms; it
is that nobody can say what it retains without measuring, and what it retains
changes with the draw budget.

### 3.7 What to compute instead

*This section states the recommended estimator.*

Abandon (2) as the target. Target retained mass directly, and measure it.

Let $r \in \{1,\dots,R\}$ index recordings and let $\theta^{(n)}$, $n \in
\{1,\dots,N\}$, denote the pooled posterior draws across all recordings and all
ensemble members (pooling across an ensemble yields a mixture, which is broader
than any component — the direction that is safe for a region one must not
wrongly exclude from). For a candidate box $\prod_{j}[l_{j}, u_{j}]$ define the
empirical retained mass

$$\widehat{m}\bigl(\{l_{j}, u_{j}\}_{j=1}^{p}\bigr) \;=\; \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\Bigl\{\, \theta^{(n)} \in \textstyle\prod_{j=1}^{p} [\,l_{j}, u_{j}\,] \,\Bigr\}. \tag{14}$$

Then:

1. Apply the joint density threshold $\tau_{\epsilon}(z_{r})$ per recording, as
   in (3). This step is retained and does real work: it removes normalizing-flow
   leakage and low-density tails that no quantile of a marginal would catch.
2. Set $[l_{j}, u_{j}]$ from **per-axis quantiles** of the retained pooled
   draws, at a common level.
3. Tune that single level by one-dimensional search until $\widehat{m}$ equals
   the chosen target (for instance $0.99$ or $0.999$).
4. Union across recordings per axis — projection commutes with union,
   $\Pi_{j}\bigl(\bigcup_{r} M_{\epsilon}(z_{r})\bigr) = \bigcup_{r}
   \Pi_{j}\bigl(M_{\epsilon}(z_{r})\bigr)$, so this step is exact, not an
   approximation.
5. Clip to the original prior box and report **shrinkage** per axis, the
   fraction of the original range retained.

What this buys: the emitted region carries a measured guarantee ("this box
contains $99.9\%$ of pooled posterior mass") instead of an unverifiable
construction; it does not move when the draw budget moves; and the guarantee is
directly falsifiable on a benchmark where the true posterior is known.

Two properties to keep in view. First, a union of intervals is not an interval:
if two recordings give disjoint support on some axis, the convex hull silently
includes a gap that no recording supports, so the disjoint segments should be
emitted alongside the hull. Second, (14) is a Monte Carlo estimate with standard
error $\sqrt{m(1-m)/N}$, which at $m = 0.999$ and $N = 10^{4}$ is $3\times
10^{-4}$ — adequate, but not so accurate that a target of $0.9999$ would be
meaningful at that budget.

### 3.8 Does the Gaussian assumption drive this?

*This section establishes that the conclusion survives the assumption being
dropped.*

The specific constants $R_{\epsilon}$ and $a_{N}$ are Gaussian. The structure is
not, for three reasons.

The inclusion $\widehat{\Pi}_{j} \subseteq \Pi_{j}(M_{\epsilon}(z))$ of §3.1 is
distribution-free: it follows from $\widehat{M}_{\epsilon} \subset
M_{\epsilon}(z)$ alone. The bias has a sign regardless of the posterior's shape.

The $\sqrt{2\log N}$ rate is the *fastest* of the extreme-value regimes. For a
distribution in the Gumbel domain with heavier-than-Gaussian tails the sample
maximum grows faster in $N$ but the target $\Pi_{j}$ recedes faster still; for
one with bounded support the sample maximum converges to the support endpoint at
a polynomial rate, but then $\Pi_{j}(M_{\epsilon}(z))$ is close to that endpoint
too and the whole construction degenerates differently. Neither case restores
agreement between (2) and (4) at feasible $N$.

The volume argument (11) is purely geometric and involves no distributional
assumption beyond convexity of the region — and non-convex regions make the
box-versus-region gap larger, not smaller.

---

## 4. Summary of results

- **(S1)** The computed per-axis interval (4) is contained in the intended
  projection (2) almost surely, for every $N$ and every $j$ — biased inward by
  construction, never outward. Derived in §3.1.
- **(S2)** For a Gaussian posterior the exact projection is $\mu_{j} \pm
  R_{\epsilon}\sigma_{j}$ with $R_{\epsilon} = \sqrt{F^{-1}_{\chi^{2}_{p}}(1 -
  \epsilon)}$, independent of the off-diagonal structure of $\Sigma$. Equation
  (7), derived in §3.2. At $p = 27$, $R_{10^{-3}} = 7.448$.
- **(S3)** The computed interval instead has half-width $a_{N}\sigma_{j}$ with
  $a_{N} \sim \sqrt{2\log N}$. Equations (8)–(9), §3.3.
- **(S4)** At $p = 27$, $\epsilon = 10^{-3}$: the computed interval is 48% too
  narrow at $N = 10^{4}$ and 35% too narrow at $N = 10^{6}$. Table in §3.4.
- **(S5)** The bounds move with the draw budget — 26% wider from $N = 10^{4}$ to
  $N = 10^{6}$ for the same posterior — so $N$ becomes an undeclared tuning
  parameter of the result. §3.4.
- **(S6)** Matching the true projection would need $N_{\text{req}} \approx
  \exp(R_{\epsilon}^{2}/2) = 1.11\times10^{12}$ draws at $\epsilon = 10^{-3}$,
  exponential in $p$. Equation (10), §3.5.
- **(S7)** The intended target is itself useless: the box circumscribing the HPR
  has $6.0\times10^{11}$ times its volume at $p = 27$ (equation (11)) and
  retains posterior mass indistinguishable from one (equation (12)). §3.6.
- **(S8)** Calibrating instead to a target retained mass requires half-widths
  $a(0.99) = 3.559$, $a(0.999) = 4.125$ at $p = 27$ (equation (13)), yielding a
  measured guarantee that is stable in $N$. §3.6–3.7.
- **(S9)** Per-axis projection discards correlation entirely (equation (7)
  depends on $\Sigma$ only through $\sigma_{j}$), which is a separate and
  additive weakness of any box-shaped region. §3.2, §3.6.

---

## 5. Open points, caveats, and assumptions

**Assumed without proof.**

- §3.2–3.6 assume $q_{\phi}(\theta \mid z)$ is Gaussian. A normalizing flow is
  not, and a multimodal posterior would break (5) outright — the HPR would not
  be connected and $\Pi_{j}$ would be a union of intervals rather than one.
  §3.8 argues the qualitative conclusions survive; that argument is heuristic
  and is not a proof.
- Equation (8) treats the retained draws as an unmodified i.i.d. sample. The
  thresholding in (3) makes them dependent and slightly truncated. The
  correction is second-order and, as noted in §3.3, has the same sign as the
  main effect.
- Equation (12) assumes independent coordinates. Under correlation the retained
  mass of a given box is generally *higher*, so (13) is conservative as a
  half-width prescription — but this has not been quantified for the actual
  posterior.

**Regime of validity.**

- The expansion (9) is asymptotic. At $N = 10^{3}$ it disagrees with the Monte
  Carlo value by about 4% (3.116 versus 3.239), which is why the table in §3.4
  quotes both. Conclusions are drawn from the Monte Carlo column.
- The numbers throughout are for $p = 27$. Everything scales, but the specific
  claim "roughly half the width" does not transfer to other dimensions.

**Unresolved.**

- **Coordinate dependence.** Nothing here is invariant to the stored coordinate.
  An axis stored as $\log \theta_{j}$ produces a different box than the same
  axis stored linearly, and the two are not related by exponentiation because
  the projection of a region does not commute with a nonlinear per-axis map.
  Which coordinate is the right one to truncate in is a modelling decision, not
  a statistical one, and is not settled here.
- **Choice of the retained-mass target.** §3.7 recommends measuring $m$ rather
  than assuming it, but offers no principle for choosing between $0.99$ and
  $0.999$. That choice trades simulation budget against the risk of excluding
  true parameter regions and should be made against a stated cost.
- **Interaction with misspecification.** Everything above concerns the geometry
  of a posterior. If the posterior is itself untrustworthy — because the real
  observations lie outside the simulated distribution — then no calibration of
  the box repairs it, and a narrower region is actively harmful. The
  misspecification gate is logically prior to all of this.
- **Disjoint unions.** §3.7 step 4 is exact for the union, but the emitted hull
  may contain gaps supported by no recording. How a downstream sampler should
  treat those gaps is left open.

---

## 6. References and further reading

- **Deistler, Goncalves, Macke**, *Truncated proposals for scalable and
  hassle-free simulation-based inference*. Read in full from the project
  knowledge base. Source of the HPR threshold rule ($\tau$ as the
  $\epsilon$-quantile of the approximate posterior density at its own samples),
  of the values $\epsilon \in \{10^{-3}, 10^{-4}, 10^{-5}\}$ used throughout,
  and of the reported contrast between marginal-based truncation (20% of prior
  samples rejected) and joint-based truncation (99.94%) on the pyloric network.
- **Extreme-value theory for the Gaussian maximum**, equation (9). This is the
  classical Fisher-Tippett-Gnedenko expansion, **stated here from memory** and
  not checked against a source in preparing this note; the Monte Carlo column of
  §3.4 was computed independently and is what the conclusions rest on.
- **Concentration of measure / ball-in-box volume ratio**, equation (11).
  Elementary and computed directly from the standard volume formula; **stated
  from memory** as to attribution.
- No PubMed or bioRxiv source underlies this note. Targeted queries on
  simulation-based inference and misspecification returned nothing (PubMed does
  not index the relevant venues; the bioRxiv connector offers no keyword
  search). Every numerical value in §3.4, §3.6 and §4 was computed directly with
  `scipy.stats` and `numpy` for this document rather than taken from any source.
- Sections 3.1, 3.3–3.8 are the author's own derivation and reasoning, not a
  reproduction of any published argument.
