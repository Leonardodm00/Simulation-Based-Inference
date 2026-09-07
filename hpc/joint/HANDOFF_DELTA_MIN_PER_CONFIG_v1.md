# Handoff: per-finalist `delta_min` instead of one pre-search floor

Target: the implementation thread for `npe_tune_gates.py` / `npe_tune_joint.py`.
Changes G1's threshold from a single bank-level constant to a per-configuration
one measured at the finalist stage. Origin: user proposal, 2026-09-04.

| Date | Change |
|---|---|
| 2026-09-04 | v1.0. Two-tier `delta_min`; Holm across finalists; rank/gate split separation; recipe-identity requirement. |

---

## 1. Verdict

Adopt, with three amendments. The proposal is right that `delta_min` is
config-dependent and that computing it after ranking is both cheaper and more
defensible. It is wrong only in the stopping rule (S4.1) and silent on two
things that decide whether the resulting number means anything (S4.2, S4.3).

| Item | Now | Change |
|---|---|---|
| `delta_min` during search | single value from pre-search `baseline` | unchanged -- keep as a **provisional screen** |
| `delta_min` at finalists | same single value | **per-finalist**, measured from that config's own shuffled control |
| Which config ships | best-ranked | best-ranked **among those passing G1 after Holm** |
| Selection statistic | held-out NLL for both ranking and G1 | rank on validation, gate on the third split |
| `baseline` subcommand | measures $\delta_{\min}$, $\sigma_{\rm seed}$ | keeps $\sigma_{\rm seed}$; its $\delta_{\min}$ demoted to provisional |

---

## 2. Why per-config is the right object

`delta_min` estimates the distribution of $\hat\Delta = L_0 - L$ achievable by
an estimator that provably has no signal. That distribution is **not** a
property of the bank alone. It depends on capacity, on the training recipe, and
on the early-stopping rule, because those determine how much of a finite
shuffled sample the network can fit and how much of that survives to the
held-out split. A 4-block flow and a 16-block flow with different learning-rate
schedules have different noise-fitting behaviour and therefore different floors.

A single floor measured from one reference architecture is therefore
mis-calibrated for every config far from that reference -- too lenient for some,
too strict for others -- in a direction nobody can sign without measuring. That
is the whole argument, and it is sufficient. [reasoning]

The plan already separates ranking from gating: *"Gates gate, NLL ranks"*, and
G2/G3 already run on finalists only while G1 runs free at every GP evaluation
[KB, plan S5.1 and `GATES_v1.md` S3.2]. Moving the authoritative `delta_min` to
the finalist tier puts G1 on the same footing as G2 and G3 rather than
introducing a new pattern.

---

## 3. The procedure

Two tiers. Both are needed; the first is not redundant.

**Tier 1 -- provisional screen, during search.** Keep the existing pre-search
`baseline` control and its single `delta_min`. G1 stays free at every GP
evaluation, as now. Its role changes from verdict to screen: it flags configs
that are obviously uninformative without costing anything. **Label its verdict
`provisional` in the ledger** so no downstream reader mistakes it for the
gated result.

**Tier 2 -- authoritative, at finalists.** For each of the $K$ finalists
(current default $K = 3$ via `finalists --top-k 3` [KB, plan Stage 4]; the
proposal suggested 5, either is fine, fix it before running):

1. Take finalist $j$'s **exact** config -- every hyperparameter, the training
   recipe, the epoch ceiling, the early-stopping rule (S4.3).
2. Build $n_{\rm ctrl}$ shuffled banks by permuting the $(\theta, z)$ pairing
   within the training split, keeping both marginals, one permutation per
   control seed.
3. Train that config on each shuffled bank; record
   $\hat\Delta^{\rm ctrl}_{j,1},\dots,\hat\Delta^{\rm ctrl}_{j,n_{\rm ctrl}}$
   on the **gate split** (S4.2).
4. Form the per-finalist $p$-value (S4.1), not a $3\sigma$ threshold.
5. Holm-correct across the $K$ finalists; the shipped config is the
   **best-ranked among those rejected at $\alpha$**.

---

## 4. The three amendments

### 4.1 Replace "first that passes" with "test all $K$, Holm, take best-ranked survivor"

Walking down a ranked list and stopping at the first pass is a sequential
multiple-comparison procedure. Testing up to $K$ candidates each at a nominal
per-test level inflates the family-wise error roughly $K$-fold: with five
candidates, the chance that *some* config clears a signal-blind floor by
fluctuation is about five times the intended rate. A valid early-stopping
version needs an alpha-spending rule, which is more machinery than $K=3$ is
worth.

Instead: run controls for all $K$ finalists, compute $K$ $p$-values, apply
Holm-Bonferroni, and take the highest-ranked config among those rejected. In
the common case where the top-ranked config passes comfortably this returns the
same answer as the user's procedure; it differs only where it matters. The
project already uses Holm for G2's per-axis SBC [KB, `GATES_v1.md` S3.4.3], so
the machinery and the precedent both exist.

**The $p$-value, and why not $\mathrm{mean} + 3\,\mathrm{sd}$.** With
$n_{\rm ctrl}$ small, $s_j$ is a poor estimate of $\sigma_j$ and
"$\mathrm{mean} + 3s$" is nowhere near a $10^{-3}$ tail. Use a one-sided $t$:
for each fixed finalist $j$, with $\hat\Delta_j$ the candidate's own gate-split
score averaged over its $n_{\rm s}$ training seeds,

$$t_j \;=\; \frac{\hat\Delta_j \;-\; \overline{\hat\Delta^{\rm ctrl}_j}}{s_j\,\sqrt{\tfrac{1}{n_{\rm s}} + \tfrac{1}{n_{\rm ctrl}}}}, \qquad p_j \;=\; \Pr\big[\,T_{\nu} > t_j\,\big], \quad \nu = n_{\rm ctrl} - 1,$$

where $\overline{\hat\Delta^{\rm ctrl}_j}$ and $s_j$ are the mean and sample SD
of that finalist's own control runs. If $\hat\Delta_j$ is a single seed rather
than a mean, set $n_{\rm s} = 1$; the $\sqrt{1/n_{\rm s} + 1/n_{\rm ctrl}}$ term
is what makes this a prediction interval rather than a confidence interval on
the control mean, which is the correct object because $\hat\Delta_j$ is one
realisation and not a population parameter.

Keep the existing `floor` as a hard minimum regardless of the $p$-value: a
config can be statistically distinguishable from its own noise floor while
still being useless in nats/row.

### 4.2 Rank and gate on different splits, or the test is anti-conservative

This is the load-bearing amendment and the proposal does not mention it.

The finalists are the argmax of held-out NLL over the whole search -- hundreds
of configs, not $K$. The top-ranked config's $\hat\Delta$ is therefore a maximum
over a large family and is biased upward by selection (winner's curse).
Comparing that maximum against a null calibrated for a *single* config
under-states the false-pass rate, and the Holm correction of S4.1 does not fix
it: Holm corrects for the $K$ tests you performed, not for the $N_{\rm search}$
selections that produced the candidates.

The fix is available and costs nothing extra: the tuning stack already has a
**frozen grouped 3-way split with hash** [KB, plan S1, `npe_tune_data`]. Use
the validation split to rank and the third split to compute both
$\hat\Delta_j$ and $\hat\Delta^{\rm ctrl}_{j,\cdot}$ at the finalist stage.
Then $\hat\Delta_j$ is not the quantity that selected finalist $j$, and the
$p$-value means what it claims.

**Verify before implementing:** which of the three splits `npe_tune_score.py`
currently evaluates $L$ on, and whether the third split is already consumed by
anything else. Not established from the documents I have; the repo was not
inspected in this thread.

### 4.3 The control must use the identical recipe, and this must be enforced

A per-config floor is only meaningful if the control differs from the candidate
in *exactly one* respect: the shuffled pairing. Same architecture, same
optimiser, same learning-rate schedule, same batch size, same epoch ceiling,
same early-stopping rule, same split hash, same seeds policy. If the candidate
early-stops on validation loss and the control does not, the two are not
comparable and the floor is meaningless.

Implementation: build the control config by taking the finalist's config object
and setting only the shuffle flag and permutation seed. Assert byte-identity of
the remaining fields -- the joint-space code already has the pattern of building
byte-identical configs from points differing only in inactive coordinates [KB,
plan S5.1], so reuse that comparison rather than writing a new one.

**Expect early stopping to trigger almost immediately on shuffled data.** That
is correct behaviour, not a bug: it reflects what the recipe does when there is
no signal, which is exactly the quantity being measured. Record epochs-to-stop
for every control run so this is visible rather than inferred.

---

## 5. Cost

$n_{\rm ctrl}$ trainings per finalist, $K$ finalists, so
$K \times n_{\rm ctrl}$ control trainings total, all at the finalist tier where
G2 and G3 already impose per-member posterior sampling. Against the current
scheme this is $K$-fold more control trainings than one pre-search `baseline`
run -- but the pre-search run is not deleted (S3, Tier 1), so the marginal cost
is $K \times n_{\rm ctrl}$ trainings added once per campaign, not per GP
evaluation. At $K = 3$ and $n_{\rm ctrl} = 5$ that is 15 trainings, comparable
to one finalist ensemble at $M_{\rm ens} = 5$.

The user's original motivation -- that per-config floors during the search would
be prohibitive -- is correct and is why Tier 1 stays a cheap screen. Nothing
per-config runs inside the GP loop.

---

## 6. What `baseline` keeps doing

Do not delete the `baseline` subcommand. It also measures $\sigma_{\rm seed}$,
which the search uses for the escalation verdict and which S2.4's "A0 degrades"
decision rule depends on ($D - 2\sigma_{\rm seed} > 0$) [KB, plan S2.4, S5.1].
That is a separate quantity with a separate purpose and is unaffected by this
change.

Its `delta_min` output stays as the Tier-1 screen and as a cross-check: if a
finalist's own floor differs from the bank-level floor by a large factor, that
is informative about how much the config's capacity is driving the floor, and
should be reported rather than silently overridden.

---

## 7. Reporting requirements

Non-negotiable, because this procedure has a garden-of-forking-paths failure
mode if reported selectively:

- Report **all $K$ finalists' results**, passes and failures, with raw $p_j$,
  Holm-adjusted $p_j$, $\hat\Delta_j$, and the per-finalist control mean and SD.
  Never report only the shipped config.
- State $K$, $n_{\rm ctrl}$, $n_{\rm s}$ and $\alpha$, fixed **before** the
  finalists are known.
- State which split ranked and which split gated.
- If no finalist passes, that is the result. Do not extend $K$ after seeing the
  outcome; extending $K$ post hoc reintroduces exactly the multiplicity Holm
  was applied to remove.

---

## 8. Smoke tests

Naming: J1-J18 taken (J17/J18 reserved by `HANDOFF_ALIASING_OPERATOR_v1.md`).
Use J19, J20.

**J19 -- the per-finalist $p$-value is calibrated under the null.** Pure numpy,
no training. Simulate $\hat\Delta_j$ and $\hat\Delta^{\rm ctrl}_{j,\cdot}$ from
a common Gaussian (i.e. the candidate genuinely learned nothing), across many
replicates and across $n_{\rm ctrl} \in \{3, 5, 10\}$. Assert: the $p_j$ of
S4.1 is uniform on $[0,1]$; Holm across $K$ controls the family-wise error at
$\alpha$; and -- the point of the test -- the naive
$\mathrm{mean} + 3\,\mathrm{sd}$ rule does **not** achieve its nominal rate at
small $n_{\rm ctrl}$. That last assertion is the justification for the change
of statistic and should fail loudly if someone reverts it.

**J20 -- recipe identity.** Construct a finalist config and its control config
through the production code path; assert every field except the shuffle flag
and permutation seed is byte-identical, and that the shuffle genuinely destroys
the pairing while preserving both marginals (compare sorted $\theta$ rows and
sorted $z$ rows before and after).

---

## 9. Open items

- **Which split does `npe_tune_score.py` score on.** S4.2 depends on it.
  Unverified -- repo not inspected.
- **How many control seeds `baseline` currently runs by default.** Not stated
  in `GATES_v1.md` or the plan; sets the precedent for $n_{\rm ctrl}$.
- **Which estimator the current pre-search control uses.** Also not stated in
  either document -- this was the gap that prompted the proposal, and it becomes
  moot for Tier 2 (the control inherits the finalist's config) but still applies
  to Tier 1.
- **Does $\hat\Delta^{\rm ctrl}$ have a mean near zero, below it, or above it?**
  On held-out rows a shuffled-trained network should score *worse* than the
  exact prior, so the control mean may be negative while its SD carries the
  information. If so, `max(floor, ...)` will bind often and the $p$-value route
  of S4.1 matters more than the threshold route. Worth checking on the first
  run before tuning $n_{\rm ctrl}$.
- **Interaction with G1 conditions (b) and (c).** The existing gate tests three
  conditions -- point estimate, bootstrap lower bound, across-seed margin
  [KB, `GATES_v1.md` S3.3.1]. This handoff replaces the *threshold* they are
  compared against; whether all three should be re-expressed as $p$-values or
  only condition (a) is undecided. Recommend: keep (b) and (c) as-is against
  the per-finalist floor, and apply Holm only to (a)'s $p$-value, since (b) and
  (c) are robustness conditions rather than independent tests. Not settled.

---

## 10. Provenance

**From the user, this conversation:** the core proposal -- postpone `delta_min`,
compute it per ranked configuration, walk down the finalists. Adopted; S4.1 is
the one substantive change to it.

**Verified against project knowledge (full text read):** the two-tier gate cost
structure and G1's "free at every GP evaluation" status; `delta_min_from_control`
and its `max(floor, mean + 3 sd)` form; Holm-Bonferroni already in use for G2;
"gates gate, NLL ranks"; the frozen grouped 3-way split; `baseline` measuring
both $\delta_{\min}$ and $\sigma_{\rm seed}$; `finalists --top-k 3`. Sources:
`GATES_v1.md` S3.2, S3.3.1, S3.3.2, S3.4.3; `JOINT_DSN_NPE_PLAN_v0_6.md` S1,
S2.4, S5.1, Stage 4.

**Reasoning, not from any source:** that the control's $\hat\Delta$ distribution
is capacity- and recipe-dependent (S2); the selection-bias argument of S4.2;
the $t$-based $p$-value of S4.1 and the objection to $\mathrm{mean} + 3s$ at
small $n_{\rm ctrl}$; the reporting requirements of S7.

**Not inspected:** the repository. Every module and function name here comes
from the two documents, not from code.
