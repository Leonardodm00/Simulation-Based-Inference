# Reading the witness-function maps: an interpretation guide

**Date:** 2026-08-24
**Applies to:** `hpc/npe_misspec.py` (witness section), `hpc/witness_run.py`,
and the figures they write.

## Abstract

The group-aware misspecification gate reduces the comparison between the
simulated and the real embedding cloud to one scalar per space, a
permutation $p$-value on $\widehat{\mathrm{MMD}}^2$. That scalar answers
*whether* the two distributions differ and is silent on *where*, which is
the question that determines what to do next: a simulator that misses a
region of state space needs a different simulator, whereas a handful of
real recordings outside the simulated support needs a different cohort
or an explicit outlier model. This document explains how to read the
witness-function figures that localise the disagreement, what each of
the three field constructions (`lift`, `proj`, `nw`) does and does not
license, how to decide whether a two-dimensional picture of a
$E$-dimensional object may be trusted, and how to compare the PCA,
contrast and t-SNE views against each other.

**Deliberately excluded.** This document does not re-derive the gate, the
group-aware permutation scheme, or the minimum-detectable-shift
calculation; it does not treat the choice of embedding; and it makes no
claim about which biophysical parameters produce an observed
discrepancy -- the witness map points at regions of embedding space, and
mapping those back to mechanism is a separate step. Nothing here is a
hypothesis test: the inferential statement remains the gate's $p$-value.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in Sec. |
|---|---|---|---|---|
| $E$ | embedding dimension | $E \in \mathbb{N}$ | dimensionless | Sec. 3.1 |
| $S^{E-1}$ | unit sphere in $\mathbb{R}^{E}$, where the L2-normalised embeddings live | $\{z \in \mathbb{R}^{E} : \lVert z \rVert_2 = 1\}$ | dimensionless | Sec. 3.1 |
| $z$ | a generic point of embedding space | $z \in \mathbb{R}^{E}$; $z \in S^{E-1}$ in the `z` space | dimensionless | Sec. 3.1 |
| $z_i^{\mathrm{sim}}$ | $i$-th simulated embedding | $z_i^{\mathrm{sim}} \in \mathbb{R}^{E}$, $i = 1,\dots,n_{\mathrm{sim}}$ | dimensionless | Sec. 3.1 |
| $z_j^{\mathrm{real}}$ | $j$-th real embedding (one analysis window) | $z_j^{\mathrm{real}} \in \mathbb{R}^{E}$, $j = 1,\dots,n_{\mathrm{real}}$ | dimensionless | Sec. 3.1 |
| $n_{\mathrm{sim}}, n_{\mathrm{real}}$ | sample sizes of the two arms | $\in \mathbb{N}$ | counts | Sec. 3.1 |
| $n_{\mathrm{fit}}, n_{\mathrm{eval}}$ | sizes of the simulated subsets used to *estimate* and to *score* $\hat\mu_{\mathrm{sim}}$ | $\in \mathbb{N}$, $n_{\mathrm{fit}} + n_{\mathrm{eval}} \le n_{\mathrm{sim}}$ | counts | Sec. 3.4 |
| $\sigma_s$ | $s$-th kernel bandwidth | $\sigma_s \in \mathbb{R}_{>0}$, $s = 1,\dots,S$ | same as $\lVert z - z' \rVert_2$ | Sec. 3.2 |
| $S$ | number of bandwidths in the multi-scale grid | $S \in \mathbb{N}$ | count | Sec. 3.2 |
| $k_{\sigma_s}(\cdot,\cdot)$ | Gaussian RBF kernel at bandwidth $\sigma_s$ | $k_{\sigma_s} : \mathbb{R}^{E} \times \mathbb{R}^{E} \to (0,1]$ | dimensionless | Sec. 3.2 |
| $k_{\Sigma}(\cdot,\cdot)$ | summed multi-scale kernel, $k_{\Sigma} = \sum_{s=1}^{S} k_{\sigma_s}$ | $k_{\Sigma} : \mathbb{R}^{E} \times \mathbb{R}^{E} \to (0,S]$ | dimensionless | Sec. 3.2 |
| $\hat\mu_{\mathrm{sim}}(z), \hat\mu_{\mathrm{real}}(z)$ | empirical kernel mean embeddings, evaluated at $z$ | $\mathbb{R}^{E} \to \mathbb{R}_{>0}$, for each fixed $z$ | dimensionless | Sec. 3.3 |
| $g_{\sigma_s}(z)$ | witness function at bandwidth $\sigma_s$, evaluated at $z$ | $\mathbb{R}^{E} \to \mathbb{R}$, for each fixed $z$ | dimensionless | Sec. 3.3 |
| $g_{\Sigma}(z)$ | summed witness, $g_{\Sigma}(z) = \sum_{s=1}^{S} g_{\sigma_s}(z)$ | $\mathbb{R}^{E} \to \mathbb{R}$, for each fixed $z$ | dimensionless | Sec. 3.3 |
| $u_i^{(s)}, v_j^{(s)}$ | witness at bandwidth $\sigma_s$ evaluated at simulated point $i$ / real point $j$ | $\in \mathbb{R}$ | dimensionless | Sec. 3.4 |
| $u_i, v_j$ | the same, summed over bandwidths: $u_i = \sum_s u_i^{(s)}$ | $\in \mathbb{R}$ | dimensionless | Sec. 3.4 |
| $\bar{u}, \bar{v}$ | sample means of $\{u_i\}_{i=1}^{n_{\mathrm{eval}}}$ and $\{v_j\}_{j=1}^{n_{\mathrm{real}}}$ | $\in \mathbb{R}$ | dimensionless | Sec. 3.4 |
| $\widehat{\mathrm{MMD}}^2$ | biased empirical squared MMD, the gate's statistic | $\in \mathbb{R}$ | dimensionless | Sec. 3.4 |
| $P$ | pooled *plotted* cloud: evaluated simulated rows stacked on all real rows | $P \in \mathbb{R}^{n_P \times E}$ | dimensionless | Sec. 4.1 |
| $n_P$ | number of rows of $P$, $n_P = n_{\mathrm{eval}}' + n_{\mathrm{real}}$ | $\in \mathbb{N}$ | count | Sec. 4.1 |
| $\bar{P}$ | column mean of $P$ | $\bar{P} \in \mathbb{R}^{1 \times E}$ | dimensionless | Sec. 4.1 |
| $B$ | orthonormal basis of the plotted 2-plane, rows $a_1, a_2$ | $B \in \mathbb{R}^{2 \times E}$, $B B^{\top} = I_2$ | dimensionless | Sec. 4.1 |
| $a_1, a_2$ | the two plotted axes | $a_1, a_2 \in \mathbb{R}^{E}$, $\lVert a_1 \rVert = \lVert a_2 \rVert = 1$, $a_1^{\top} a_2 = 0$ | dimensionless | Sec. 4.1 |
| $y$ | a point of the 2-D layout | $y \in \mathbb{R}^{2}$ | dimensionless | Sec. 4.1 |
| $y_i$ | layout coordinate of pooled row $i$, $y_i = (P_i - \bar{P}) B^{\top}$ | $y_i \in \mathbb{R}^{2}$ | dimensionless | Sec. 4.1 |
| $L(y)$ | the lift, $L(y) = \bar{P} + y B$ | $L : \mathbb{R}^{2} \to \mathbb{R}^{E}$ | dimensionless | Sec. 4.2 |
| $r_i$ | off-plane residual norm of pooled row $i$, $r_i = \lVert P_i - L(y_i) \rVert_2$ | $r_i \in \mathbb{R}_{\ge 0}$ | dimensionless | Sec. 5.2 |
| $\mathrm{var\_frac}$ | share of the total variance of $P$ carried by the two plotted axes | $\in [0,1]$ | dimensionless | Sec. 5.1 |
| $\rho^{(s)}, \rho_{\Sigma}$ | slice fidelity: correlation across pooled rows between $g$ at the lifted shadow and $g$ at the row itself, per bandwidth and summed | $\in [-1,1]$ | dimensionless | Sec. 5.3 |
| $\Delta$ | difference of arm means, $\Delta = \bar{z}^{\mathrm{sim}} - \bar{z}^{\mathrm{real}}$ | $\Delta \in \mathbb{R}^{E}$ | dimensionless | Sec. 4.3 |
| $w$ | slice direction, orthogonal to the plotted plane | $w \in \mathbb{R}^{E}$, $\lVert w \rVert_2 = 1$, $Bw = 0$ | dimensionless | Sec. 6.1 |
| $t$ | signed offset along $w$ | $t \in \mathbb{R}$ | dimensionless | Sec. 6.1 |
| $t_i$ | off-plane coordinate of pooled row $i$ along $w$, $t_i = (P_i - L(y_i))^{\top} w$ | $t_i \in \mathbb{R}$ | dimensionless | Sec. 6.1 |
| $q$ | quantile level placing a slice | $q \in (0,1)$ | dimensionless | Sec. 6.1 |
| $c_q$ | correlation between the field on slice $q$ and on the reference slice | $\in [-1,1]$ | dimensionless | Sec. 6.2 |
| $d_q$ | fraction of grid cells whose field sign differs from the reference slice | $\in [0,1]$ | dimensionless | Sec. 6.2 |
| $h$ | Nadaraya-Watson smoothing bandwidth, in layout units | $h \in \mathbb{R}_{>0}$ | layout units | Sec. 4.5 |
| $\alpha$ | test level of the gate | $\alpha \in (0,1)$ | dimensionless | Sec. 7 |
| $p_{\mathrm{group}}$ | group-aware permutation $p$-value; the gate's verdict | $\in [0,1]$ | dimensionless | Sec. 7 |

### 1.1 Conventions

- Indices: $i$ ranges over simulated rows, $j$ over real rows, $s$ over
 bandwidths ($s = 1,\dots,S$), $q$ over slice quantiles. No summation
 convention is assumed; every sum is written explicitly.
- Vectors in $\mathbb{R}^{E}$ are **row** vectors when they index rows of
 a data matrix ($P_i$, $\bar{P}$) and **column** vectors in quadratic
 forms; $B \in \mathbb{R}^{2 \times E}$ has orthonormal *rows*, so
 $y = (z - \bar{P})B^{\top}$ and $L(y) = \bar{P} + yB$.
- $\lVert \cdot \rVert$ without subscript means $\lVert \cdot \rVert_2$.
- Embeddings are dimensionless: `z` is L2-normalised onto $S^{E-1}$,
 `zraw` is the same embedding before normalisation and therefore retains
 amplitude information. Bandwidths $\sigma_s$ carry the units of
 $\lVert z - z' \rVert_2$, hence are dimensionless too, and are **not**
 comparable between the `z` and `zraw` spaces.
- The witness $g$ is *unnormalised*: it is not divided by the RKHS norm
 of the mean-difference, so its numerical scale has no absolute meaning
 and only sign, relative magnitude and spatial pattern are interpretable.
- "The plane" always means the affine 2-plane $\{L(y) : y \in
 \mathbb{R}^{2}\}$, never a linear subspace: it passes through $\bar{P}$,
 not through the origin.

---

## 2. Glossary

Ordered by first appearance, because the concepts build on one another.

**Misspecification gate.** The permutation test in `npe_misspec.py`
asking whether real embeddings lie inside the simulated distribution. It
tests the *simulator* against reality, not the inference against the
simulator. Operative from Sec. 3.4.

**Maximum mean discrepancy (MMD).** A distance between two distributions
equal to the RKHS norm of the difference of their kernel mean
embeddings. Its empirical version is the gate's statistic. Operative in
Sec. 3.4.

**Kernel mean embedding.** The function $z \mapsto \frac{1}{n}\sum_i
k(z_i, z)$: a kernel-smoothed local density of the sample. Operative in
Sec. 3.3.

**Witness function.** The difference of the two kernel mean embeddings.
Its name is standard: it *witnesses* the discrepancy by taking large
values exactly where the two distributions disagree most. Operative from
Sec. 3.3.

**Bandwidth ($\sigma$).** The length scale of the Gaussian kernel. Note
the everyday-versus-technical clash: this is **not** a frequency-domain
bandwidth. It sets the spatial resolution at which the witness can
distinguish structure. Operative in Sec. 3.2.

**Median heuristic.** The convention of setting $\sigma$ to the median
pairwise distance of the pooled sample. Convenient, not optimal, and
specifically fragile under multimodality, where it is inflated by
between-mode distances. Operative in Sec. 3.2, and the origin of the
sensitivity this whole analysis was built to diagnose.

**Mode 1 / Mode 2 (simulation gap / rare-but-valid).** Two distinct
failure modes that can produce the *same* MMD. Mode 1: the entire real
cohort is displaced from the simulated bulk -- the simulator misses a
region. Mode 2: the bulks overlap and a few real recordings sit outside
the simulated support. Operative in Sec. 3.5. (Terminology follows the
model-misspecification literature in the project knowledge base; see Sec. 9.)

**Lift.** The exact right inverse of a *linear* projection, restricted to
the plotted plane: $L(y) = \bar{P} + yB$. Only linear layouts have one.
Operative from Sec. 4.2.

**Slice.** The set $\{L(y) + tw\}$ for fixed $t$ -- a flat 2-dimensional
cut through $\mathbb{R}^{E}$. A slice is **not** a marginal: the $E-2$
discarded coordinates are *fixed*, not integrated out. This distinction
is the single most common way to misread these figures. Operative in
Sec. 4.2 and Sec. 6.

**Pushforward.** The distribution induced on $\mathbb{R}^{2}$ by
projecting the cloud. `field="proj"` builds the witness of the two
pushforwards, which *is* a marginal-style object, and is therefore weaker
but self-consistent. Operative in Sec. 4.4.

**Nadaraya-Watson smoothing.** Kernel-weighted local averaging of scattered
values onto a grid. Used for t-SNE layouts, where no lift exists.
Interpolates the witness; does not evaluate it. Operative in Sec. 4.5.

**Slice fidelity ($\rho$).** Correlation, over the plotted rows, between
$g$ evaluated at a row's lifted shadow and $g$ evaluated at the row
itself. The diagnostic that decides whether a flat picture may be
believed. Operative in Sec. 5.3.

**Effective rank.** A continuous measure of how many embedding directions
carry appreciable variance, reported by `gate_run.py`. A collapsed
embedding makes a gate PASS weak evidence while leaving a rejection
decisive. Operative in Sec. 8.

---

## 3. What the witness is, and why the gate needs it

### 3.1 The setting

Two clouds in the same space: $\{z_i^{\mathrm{sim}}\}_{i=1}^{n_{\mathrm{sim}}}$
and $\{z_j^{\mathrm{real}}\}_{j=1}^{n_{\mathrm{real}}}$, both in
$\mathbb{R}^{E}$, and in the `z` space both on $S^{E-1}$. Real rows are
*clustered*: each recording contributes several disjoint windows sharing
one culture, which is why the gate's verdict is the group-aware
$p_{\mathrm{group}}$ and not $p_{\mathrm{iid}}$.

### 3.2 The kernel

For each fixed pair $(z, z') \in \mathbb{R}^{E} \times \mathbb{R}^{E}$ and
each fixed $s \in \{1,\dots,S\}$,

$$k_{\sigma_s}(z, z') = \exp\!\left(-\frac{\lVert z - z' \rVert_2^{2}}{2\sigma_s^{2}}\right), \qquad k_{\Sigma}(z, z') = \sum_{s=1}^{S} k_{\sigma_s}(z, z'). \tag{1}$$

The multi-scale sum is used because no single $\sigma$ is right for a
multimodal cloud. This matters for reading the figures: **every witness
figure has one panel per $\sigma_s$ plus a summed panel**, and only the
summed panel corresponds to the statistic the gate thresholds.

### 3.3 The witness function

For each fixed $z \in \mathbb{R}^{E}$ and each fixed $s$,

$$g_{\sigma_s}(z) = \hat\mu^{(s)}_{\mathrm{sim}}(z) - \hat\mu^{(s)}_{\mathrm{real}}(z) = \frac{1}{n_{\mathrm{fit}}}\sum_{i=1}^{n_{\mathrm{fit}}} k_{\sigma_s}(z_i^{\mathrm{sim}}, z) \;-\; \frac{1}{n_{\mathrm{real}}}\sum_{j=1}^{n_{\mathrm{real}}} k_{\sigma_s}(z_j^{\mathrm{real}}, z). \tag{2}$$

Because $\hat\mu$ is **linear in the kernel**, summing over bandwidths
commutes with forming the difference:

$$g_{\Sigma}(z) = \sum_{s=1}^{S} g_{\sigma_s}(z) \qquad \text{for each fixed } z \in \mathbb{R}^{E}. \tag{3}$$

**Reading the sign.** $g_{\Sigma}(z) > 0$ where simulations concentrate
and real data do not; $g_{\Sigma}(z) < 0$ where real recordings live and
the simulator does not go. In every figure red is positive
(simulation-dense) and blue negative (real-dense), on a diverging scale
centred at zero, and the black contour is the level set $g_{\Sigma} = 0$.

*Plainly:* red is territory the simulator visits and reality does not;
blue is territory reality occupies and the simulator misses. The black
line is the border.

### 3.4 The identity that ties the map to the verdict

Evaluate $g$ on the two samples themselves, writing $u_i^{(s)} =
g_{\sigma_s}(z_i^{\mathrm{sim}})$ and $v_j^{(s)} =
g_{\sigma_s}(z_j^{\mathrm{real}})$, and $u_i = \sum_s u_i^{(s)}$,
$v_j = \sum_s v_j^{(s)}$. When $\hat\mu_{\mathrm{sim}}$ is estimated on
the *same* simulated rows it is evaluated on (the `--no_split` regime,
$n_{\mathrm{fit}} = n_{\mathrm{eval}} = n_{\mathrm{sim}}$), expanding the
two sample means gives, exactly,

$$\bar{u} - \bar{v} = \frac{1}{n_{\mathrm{sim}}^{2}}\sum_{i,i'} k_{\Sigma}(z_i^{\mathrm{sim}}, z_{i'}^{\mathrm{sim}}) + \frac{1}{n_{\mathrm{real}}^{2}}\sum_{j,j'} k_{\Sigma}(z_j^{\mathrm{real}}, z_{j'}^{\mathrm{real}}) - \frac{2}{n_{\mathrm{sim}} n_{\mathrm{real}}}\sum_{i,j} k_{\Sigma}(z_i^{\mathrm{sim}}, z_j^{\mathrm{real}}) = \widehat{\mathrm{MMD}}^2. \tag{4}$$

So the gate's scalar **is** the difference between the means of two
histograms, and those histograms are what the witness figures show. This
is verified to machine precision in `smoke_witness.py` and, through the
driver, in `smoke_witness_run.py`.

By default `witness_run.py` runs with `split=True`, which estimates
$\hat\mu_{\mathrm{sim}}$ on one simulated half and scores the other, so
that any "these recordings look anomalous" reading is made on held-out
data. In that regime (4) holds *in expectation*, not exactly; the JSON
field `witness_gap_is_exact_mmd2` records which regime produced the
numbers.

### 3.5 Why the histograms carry what the scalar cannot

A difference of means cannot distinguish "everything moved a little" from
"almost nothing moved and a few things moved a lot". Those are Mode 1 and
Mode 2, and they can give identical $\bar{u} - \bar{v}$.

| Figure `witness_hist_<space>.png` | Signature | Reading |
|---|---|---|
| $\{v_j\}$ bulk displaced from $\{u_i\}$, shapes similar, little overlap | location shift | **Mode 1**, simulation gap |
| $\{v_j\}$ bulk superimposed on $\{u_i\}$, a few $v_j$ stranded far negative | tail excess | **Mode 2**, rare-but-valid recordings |

The most negative $v_j$ are literally the real windows sitting deepest in
the simulation gap; `witness_run.py` names the five most negative by
group label under `most_negative_real` in the summary JSON.

---

## 4. The five views, and what each licenses

### 4.1 The layout

All views project the pooled plotted cloud $P$ (evaluated simulated rows,
thinned to `--max_points`, stacked on **all** real rows) to two
dimensions. Note that $P$ is not the full simulated set: `var_frac` and
$r_i$ are therefore not comparable across different `--max_points` or
`split` settings.

### 4.2 `field="lift"` -- exact, on a flat cut (PCA and contrast)

The grid is lifted back into $\mathbb{R}^{E}$ by $L(y) = \bar{P} + yB$ and
$g$ is **evaluated** there, with real kernel evaluations against the same
fit set the scatter uses. No interpolation anywhere.

What it shows: $g_{\Sigma}$ restricted to a 2-plane slice through
$\bar{P}$, with the $E-2$ discarded coordinates held at $\bar{P}$'s
values. **A slice, not a marginal.** For the `z` space the lifted grid is
re-normalised onto $S^{E-1}$, because an un-normalised lifted plane sits
strictly inside the sphere and would be further from every datum than the
data are from each other, compressing the field toward zero.

*Licensed:* statements about the sign and pattern of $g$ on that cut.
*Not licensed:* statements about the whole space, unless Sec. 5 says the cut
is faithful.

### 4.3 The two linear planes

- **PCA** maximises $\mathrm{var\_frac}$: the best 2-plane for
 *reconstruction*. That is a statement about where the points are, not
 about where $g$ varies. With a multimodal simulator the leading
 principal direction is the between-mode axis -- the nuisance structure
 the gate over-fires on.
- **contrast** sets $a_1 = \Delta / \lVert \Delta \rVert$ with $\Delta =
 \bar{z}^{\mathrm{sim}} - \bar{z}^{\mathrm{real}}$, so any mean
 displacement between the arms is visible **by construction**, and takes
 $a_2$ as the leading direction of $P$ orthogonal to $a_1$. Expect
 $\mathrm{var\_frac}$ slightly *below* PCA's; that is not a defect, since
 variance is not the objective.

Read them together. If they agree, the discrepancy is aligned with the
dominant variance and either plane tells the story. If they disagree, the
contrast plane is the one showing the sim/real difference and the PCA
plane is showing the simulator's internal structure.

*Limitation, stated rather than buried:* the contrast axis is a **mean**
difference. It guarantees visibility of a location shift and guarantees
nothing for a pure *shape* discrepancy (equal means, different
dispersion). For that case use Sec. 4.4 and the per-bandwidth panels.

### 4.4 `field="proj"` -- the witness of the pushforwards

Project both clouds, then build $g$ in 2-D from scratch with its own
median-heuristic bandwidths. Self-consistent -- the background and the
point positions agree by construction -- and it integrates over the
discarded directions rather than fixing them. It answers a **strictly
weaker** question: two clouds can differ in $E$ dimensions and coincide
after projection. Its value is *not* the gate's statistic, and its
bandwidths are 2-D and not comparable with $\{\sigma_s\}$.

### 4.5 `field="nw"` -- smoothing, the only option for t-SNE

t-SNE has no inverse map, so the field is Nadaraya-Watson smoothing in
the plane of the witness values already attached to the plotted points,
pooling $\{u_i\}$ and $\{v_j\}$ -- correct, because both are values of the
*same* function $g$ at different locations. Cells further than
`mask_radius` $\times\, h$ from every sample are left blank rather than
extrapolated; the blank margin is a feature.

*Licensed:* cluster structure, and the sign of $g$ near actual data.
*Not licensed:* any quantitative claim about the field **between** points,
and any reading of distance -- t-SNE distances between islands are not
meaningful, and neither are island sizes. Slice fidelity is recorded as
`NaN` for `nw`, not as 1.0, because the smoother is exact at the samples
by construction and a correlation there would be a tautology.

---

## 5. Deciding whether the flat picture may be believed

This section establishes the decision rule that `witness_run.py` applies
mechanically and prints as `VERDICT`.

### 5.1 Variance explained is *not* the criterion

$\mathrm{var\_frac} = \sum_{k=1}^{2}s_k^{2} / \sum_{k=1}^{E}s_k^{2}$,
with $s_k$ the singular values of the centred $P$, appears in the panel
axis labels and the figure caption. It answers "where are the points",
which is a different question from "does this picture represent $g$".
A high $\mathrm{var\_frac}$ neither implies nor is implied by a faithful
witness picture. Report it; do not decide on it.

### 5.2 Off-plane residual, in units of the bandwidth

$r_i = \lVert P_i - L(y_i) \rVert_2$ satisfies the Pythagorean
decomposition $\lVert P_i - \bar{P} \rVert^{2} = \lVert y_i \rVert^{2} +
r_i^{2}$ for each $i$ (verified numerically in `smoke_witness_heat.py`).
The reported quantity is $\mathrm{median}_i(r_i)/\sigma_s$ for each fixed
$s$. Since $g$ resolves structure only at scale $\sigma_s$, a ratio above
1 means that panel is evaluating $g$ in a region no datum occupies. A
datum one bandwidth off-plane attenuates its own kernel contribution by
$\exp(-1/2) \approx 0.61$.

Typically the ratio exceeds 1 only for the narrowest bandwidths: those
panels should be discounted while the wider ones remain informative.

### 5.3 Slice fidelity $\rho$ -- the criterion with teeth

$$\rho_{\Sigma} = \mathrm{corr}_{i=1,\dots,n_P}\Big(g_{\Sigma}\big(L(y_i)\big),\; g_{\Sigma}(P_i)\Big), \tag{5}$$

with the analogous $\rho^{(s)}$ per bandwidth. $\rho_{\Sigma} \to 1$: the
cut tracks the landscape at the data and may be read. $\rho_{\Sigma} \to
0$: the cut is decorative and the verdict must come from `proj`/`nw` or a
different plane. Values below $0.5$ raise an explicit WARNING.

### 5.4 The rule

$$\text{faithful} \iff \big(\rho_{\Sigma} \ge 0.9\big) \wedge \big(\exists\, s : \mathrm{median}_i(r_i)/\sigma_s \le 1\big) \tag{6}$$

$$\text{stable} \iff \big(\min_q c_q \ge 0.8\big) \wedge \big(\max_q d_q \le 0.1\big) \tag{7}$$

- **faithful and stable** -> *read the plane*. The scope clause states at
 how many of the $S$ bandwidths the plane resolves the data.
- **faithful, not stable** -> the cut is fine but structure exists in the
 discarded directions; Sec. 6 is where it lives.
- **not faithful, stable** -> cross-check against `proj` and t-SNE.
- **neither** -> do not read the plane at all.

$\rho$ decides and the residual reports *scope*: an earlier version of
this rule vetoed on the residual at the smallest bandwidth alone, which
turned $\rho = 0.995$ into a spurious caution. If **no** scale resolves
the data, the plane is downgraded regardless of $\rho$.

---

## 6. The slice stack: does one cut suffice?

### 6.1 Construction

$$z(y, t) = \bar{P} + yB + t\,w, \qquad y \in \mathbb{R}^{2},\; t \in \mathbb{R},\; \lVert w \rVert_2 = 1,\; Bw = 0, \tag{8}$$

with $w$ the leading principal direction of the off-plane residuals.
Offsets are placed at **quantiles of the data's own** $t_i = (P_i -
L(y_i))^{\top} w$, so every slice is one the data populate -- an offset
chosen arbitrarily would show empty space. By construction
$\frac{1}{n_P}\sum_i t_i = 0$, so the median slice is near $t = 0$ and is
used as the reference.

The colour scale is **shared across slices**, unlike
`witness_heatmaps` where each panel is a different bandwidth and hence a
different magnitude. Here every panel is the same statistic at a
different depth, and a shared scale is what makes stability visible
instead of normalising it away.

### 6.2 Reading

For each slice $q$: $c_q$ is the correlation of its field with the
reference slice's, $d_q$ the fraction of grid cells whose sign differs.

- $c_q \approx 1$, $d_q \approx 0$ **across all depths** -> the 2-plane
 picture generalises; one cut was enough.
- $c_q$ decaying with $|t|$, or $d_q$ appreciable -> the discrepancy has
 structure in the discarded directions. The zero contour reorganising
 between slices means the sim-dense and real-dense regions **swap** at
 depth, and no single flat cut can represent that.

*Scope limit:* this slices along **one** direction, the leading one. A
stable verdict proves stability along the direction most likely to
matter, not global stability.

---

## 7. How to compare PCA, contrast and t-SNE

Run all three and treat their agreement as the evidence.

| Observation | Reading |
|---|---|
| All three show the same regions blue, and Sec. 5 says faithful | The discrepancy is robust and localised. Proceed to interpret those regions. |
| PCA and contrast agree, t-SNE shows extra structure | The discrepancy has cluster structure a linear plane cannot resolve. Trust t-SNE for *which points group together*, not for the field between them. |
| Contrast shows a clean split, PCA does not | The sim/real difference lies off the dominant variance directions. Read the contrast plane; PCA is showing the simulator's internal (e.g. multimodal) structure. |
| PCA and contrast disagree with t-SNE about *which* real recordings are extreme | Suspect the layout, not the witness: $u_i$ and $v_j$ are computed in $\mathbb{R}^{E}$ and are identical across all three views. Only the *positions* differ. Compare the histograms, which are layout-free. |
| Any view looks compelling but Sec. 5 says not faithful | Do not use it. Read `proj` and the histograms. |

The single most important consistency check: $\{u_i\}$ and $\{v_j\}$ do
**not** depend on the projection. Any apparent disagreement between views
is a disagreement about layout, never about the witness values
themselves.

### 7.1 What is *not* a conclusion

- A red region does **not** mean "the simulator is wrong there" -- it means
 the simulator puts mass there and the cohort does not. With
 $n_{\mathrm{real}}$ of order tens, absence of real data in a region can
 be sampling.
- The witness is estimated from the data it is evaluated on (up to the
 split), so naming specific anomalous recordings is **exploratory**. The
 inferential statement is $p_{\mathrm{group}}$ against $\alpha$.
- The `zraw` space carries amplitude; a discrepancy visible in `zraw` and
 absent in `z` is a scale/amplitude mismatch, not a shape mismatch.

---

## 8. Summary of results

1. **Witness definition and sign convention** -- (2), Sec. 3.3. Red is
 simulation-dense, blue real-dense, black is $g_{\Sigma} = 0$.
2. **Multi-bandwidth decomposition** $g_{\Sigma} = \sum_s g_{\sigma_s}$ --
 (3), Sec. 3.3. Exact, by linearity of $\hat\mu$ in the kernel.
3. **The identity** $\bar{u} - \bar{v} = \widehat{\mathrm{MMD}}^2$ --
 (4), Sec. 3.4. Exact under `--no_split`; in expectation otherwise.
4. **Mode 1 vs Mode 2 readout** from the histograms -- Sec. 3.5.
5. **Faithfulness criterion** -- (6), Sec. 5.4. $\rho$ decides, residual
 reports scope.
6. **Stability criterion** -- (7), Sec. 5.4 and Sec. 6.2.
7. **Slice construction** -- (8), Sec. 6.1, with offsets at data quantiles.

---

## 9. Open points, caveats, assumptions

- **Assumed without proof here:** that $k_{\Sigma}$ is characteristic, so
 that $\mathrm{MMD} = 0$ implies equality of distributions. Sums of
 Gaussian kernels with distinct widths are standard for this purpose;
 the property is not re-derived in this document.
- **Approximation:** identity (4) holds exactly only in the unsplit
 regime. The default split regime trades exactness for held-out scores.
 Regime of validity: both are correct, they answer slightly different
 questions, and the JSON records which was used.
- **Asymmetry inherited from the gate:** simulated rows sharing a topology
 are not independent, and the gate treats $z^{\mathrm{sim}}$ as i.i.d.
 The witness map inherits this and does not correct it.
- **Conditioning:** if `gate_run.py` applied an activity filter, every
 figure here describes $P_{\mathrm{sim}}(\cdot \mid \text{rate in band})$
 versus $P_{\mathrm{real}}$, a weaker claim than the unconditioned check.
 State the band alongside any conclusion.
- **Unresolved:** no construction here guarantees visibility of a pure
 *shape* discrepancy (equal means, different higher moments). PCA,
 contrast and `proj` each address it partially; none is a guarantee.
- **Unresolved:** the slice stack scans one discarded direction. Scanning
 several principal pairs, or a user-chosen $w$, is a natural extension
 not implemented.
- **Sampling:** the interpretation of blue regions is limited by
 $n_{\mathrm{real}}$, which is tens of windows over a handful of
 cultures. Power is set by the number of *recordings*, not windows -- see
 the gate's minimum-detectable-shift output.
- **Not verified against external literature.** The Mode 1 / Mode 2
 vocabulary and the multi-bandwidth kernel convention follow sources in
 the project knowledge base; no PubMed or bioRxiv search was run for
 this document, and no numeric result from any paper is quoted here.

---

## 10. References and provenance

- **Project knowledge base:** the model-misspecification and
 amortised-inference PDFs attached to this project supply the Mode 1 /
 Mode 2 framing and the sum-of-Gaussian-kernels convention. Cited from
 the project files, not re-verified against the published versions here.
- **Code, stated from the source itself** (not from memory):
 `hpc/npe_misspec.py` -- `witness_function`, `witness_maps`,
 `witness_heatmaps`, `witness_slices`, `_project_2d`;
 `hpc/witness_run.py` -- the driver and the verdict rule;
 `hpc/gate_run.py` -- the arrays consumed here.
- **Verification:** every equation flagged "exact" in this document is
 asserted numerically in `smoke_witness.py`, `smoke_witness_heat.py`,
 `smoke_witness_slices.py` or `smoke_witness_run.py`. Where a document
 claim has a corresponding automated check, the check is authoritative.
- **Stated from general reasoning, unverified:** the guidance in Sec. 7 on
 how to weigh agreement between views is methodological judgement, not a
 result from any source.
