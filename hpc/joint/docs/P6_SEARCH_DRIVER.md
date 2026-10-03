# P6 -- The search driver: campaigns, the shape anchors, canonicalisation, the ledger, proposals, finalists and the per-finalist controls

**Document P6 of the joint documentation set.** Owner of the knobs that
configure the *search* rather than one training (P0 Table D): the campaign
(`--campaign`), the shape anchors (`--p`, `--embedding-dim`, `--d-theta`,
`--n-train`), the identity inputs (`--split-hash`, `--contract-digest`,
`--seed`), the proposal knobs (`--n-points`, `--n-initial-points`,
`--sigma-seed`), the ranking and gating knobs (`--top`, `--top-k`,
`--rank-split`, `--gate-split`), the control knobs (`--n-control`,
`--n-seeds`, `--alpha`, `--floor`, `--delta-min-provisional`, `--tag`), and
the library constants behind them (`INACTIVE_CANONICAL`, the fixed
`strict_semihard` and `n_posterior_draws_max` of `default_joint_space`, the
boundary tolerance `rel_tol`). Master notation: E0. The chapter that explains
Bayesian optimisation itself -- the surrogate, the acquisition function,
nested campaigns -- is E8. **Date:** 2026-10-03 (v1.1). **Applies to:** the
repository `Simulation-Based-Inference` at `834eb41`, `hpc/joint/` (D-037):
`stage4/joint_space.py`, `stage4/npe_tune_joint.py`,
`stage4/jobs/joint_tune.pbs`, `stage4/jobs/launch_joint_tune.sh`,
`stage4/smoke_test_joint_space.py`, `stage4/smoke_test_joint_tune.py`; the
shared search mechanics the driver reuses rather than copies,
`hpc/npe_tune_search.py`, `hpc/npe_tune_ledger.py`, `hpc/npe_tune_gates.py`,
`hpc/npe_diagnostics.py`; the DSN's `dsn/condition_space.py` (the legality
projection and the activity mask the canonicalisation imports); the runner
the driver drives, `stage3/run_joint_arms.py`; for the D-037 mapping,
`dsn/search.py`, the DSN JSON configs under `dsn/hpc/Config/` and
`hpc/npe_tune.py`; the plan `JOINT_DSN_NPE_PLAN_v0_6.md` at its repository
version **v0.6.5** (S5.1, S5.2, Stage 4, S8 D4) and
`hpc/joint/HANDOFF_DELTA_MIN_PER_CONFIG_v1.md`; **scikit-optimize 0.10.2**,
the version `sbi_env` imports (`[KB]` `HPC_PATHS.md` sec. 7, `[CLUSTER
09-20]`), read from the GitHub source at tag `v0.10.2` (`optimizer.py`,
`utils.py`, `learning/gaussian_process/gpr.py`, `acquisition.py`,
`space/space.py`) **and installed in the sandbox for this document**
(`pip install scikit-optimize==0.10.2`, which brought scikit-learn 1.9.1,
scipy 1.18.1 and numpy 2.5.3), so that every number of S3.2-S3.7 is the
library's own output through the driver's own functions,
`tools/p6_numbers.py` `[RAN]`; scikit-learn's Gaussian-process regressor read
at 1.6.1 (GitHub) and 1.9.1 (installed), the two agreeing on the one point
that matters here (S3.2); **the cluster's scikit-learn version is not
recorded** in `HPC_PATHS.md` and is flagged where it matters. The sandbox
still has no torch, so the runner was never invoked: the one piece of it the
driver's behaviour depends on, `arm_config`, is extracted from the runner's
source by `ast` and executed on its own `[RAN]` B4.

| date | change |
|---|---|
| 2026-10-03 | v1.1. One correction, nothing else changed: the multilevel SBI paper (Hikida et al.) was flagged PREPRINT, not peer-reviewed, in S6; the project PDF's p.1 carries the NeurIPS 2025 conference line, so the flag is marked [corrected 2026-10-03]. Evidence: `[KB-PDF p.1]`, read in the P7 turn (P7 S6). |
| 2026-10-02 | v1. Written from `stage4/joint_space.py` and `stage4/npe_tune_joint.py` (both read in full), `stage4/jobs/joint_tune.pbs` and `stage4/jobs/launch_joint_tune.sh` (read in full), `stage4/smoke_test_joint_space.py:1-60, 547-680`, `stage4/smoke_test_joint_tune.py:1-60, 403-500, 600-615`, `npe_tune_search.py:60-110, 195-300, 340-436`, `npe_tune_ledger.py:95-145`, `npe_tune_gates.py:160-305`, `npe_diagnostics.py:408-422`, `dsn/condition_space.py:105-130, 192-260`, `stage3/run_joint_arms.py:174-202, 372-382, 519-540`, `dsn/search.py:206, 806-830, 880-975`, `dsn/hpc/Config/config_l3c_joint_search.json:217-218`, `config_mea_joint_full.davinci.json:226-227`, `npe_tune.py:434-470, 969-1077`; the plan S5.1, S5.2, Stage 4, S8 (D4), S2.4, S4.4; `HANDOFF_DELTA_MIN_PER_CONFIG_v1.md` (read in full); scikit-optimize 0.10.2 (`optimizer.py:241-242, 269-275, 298-325, 355-383, 384-468, 470-500, 532-610, 640`, `utils.py:364-392, 411-430`, `gpr.py:189-245`, `acquisition.py:40-41, 247-320`), scikit-learn's `_gpr.py` `fit` (1.6.1 `:269-274`; 1.9.1 the same lines of logic, read from the installed file); `[KB]` `JOINT_DSN_NPE_USAGE_v1.md` v1.3 S7, S9, S10, `claude/SBI_decisions_and_ideas_log.md` v1.21 (D-037, D-038, D-039, D-052, D-054), P0 Table D, P4 S3.3.1, P5 S3.3.4 and S3.8; `[KB-PDF]` the practical guide p.32, BayesFlow p.13, the frontier review p.4, Goncalves et al. p.3, p.16, the multilevel preprint p.3, pages as cited in S6. Every number of S3.1-S3.7 computed by the new `tools/p6_numbers.py` `[RAN]`, run twice with identical output. Findings F-am to F-as added; F-a, F-b, F-d, F-f, F-j, F-o, F-r, F-s, F-u, F-w, F-ad, F-al carried or extended. Grounding searches of S6 run and reported. |

**Abstract.** Stage 4 turns the parameter documents P1-P5 into an experiment:
a Gaussian-process search over a campaign's free axes proposes
configurations in batches, a PBS array evaluates each through the Stage 3
runner, a ledger of JSON records is the only state, and at the end a few
finalists are tested against their own shuffled-pairing controls. One would
expect the driver to be a thin wrapper that ships the configuration the
plan's protocol names; what the code at `834eb41` does is close to that for
the search and far from it for the gate, and the distance is this document's
subject. The question answered, for each knob of P0 Table D and for the
constants beneath them: where it is set and which surface it reaches (S3.1);
what the search loop computes, written out from `joint_space.py`,
`npe_tune_joint.py`, `npe_tune_search.py` and scikit-optimize's own source as
one explicit chain -- the space and its campaigns, the point-to-configuration
maps with the three canonicalisation clamps and the legality projection, the
trial identity, the surrogate as scikit-optimize builds it (a Matern kernel
on normalised coordinates, standardised targets, expected improvement
against the lowest observed value, the constant-liar batch), the escalation
verdict, the finalists and the per-finalist one-sided $t$ test with the Holm
step-down, eq. (P6.1)-(P6.11) (S3.2); what each knob changes in that chain
(S3.3); what the fixed and library constants commit a campaign to (S3.4);
how the driver maps onto the two stacks it borrows from, the standalone DSN
search and the standalone NPE tuner (D-037, S3.5); how it interacts with the
five parameter blocks (S3.6); how it fails and which file or line reveals
each failure (S3.7); and the findings this document owns (S3.8): F-a, F-b,
F-d, F-f, F-j, F-o, F-r, F-s, F-u, F-w, F-ad and F-al carried, and seven new
ones -- `--rank-split` and `--gate-split` are labels, no fourth split exists,
and finalists are ranked and gated on one split, the case the code's own
warning describes (F-am); the control's permutation seed never reaches the
runner, so the $n_{\rm ctrl}$ controls of a finalist are the same run and the
test returns an undefined p-value or a spurious pass (F-an); a control of a
finalist with either loss term on trains the NPE-only recipe, and the control
specs are written where `evaluate --pending-id` cannot find them (F-ao);
the plan's `baseline`, `partial-dependence`, $M_{\rm ens} = 5$ and the gates
G2/G3 have no counterpart in the driver, so $\sigma_{\rm seed}$ and
$\delta_{\min}$ are typed by hand and the escalation threshold defaults to
zero (F-ap); `--sigma-seed` enters the surrogate as a fixed noise level on
*standardised* targets, so the noise the GP assumes in nats per row is
$\sigma_{\rm seed}$ times the ledger's standard deviation, not
$\sigma_{\rm seed}$ (F-aq); the shape anchors must be passed identically to
`propose` and to the array's `evaluate`, and the launcher forwards none of
them (F-ar); and one `--seed` plays three roles, with a design phase that
replays the same random points round after round (F-as). **Deliberately
excluded:** why a Gaussian process with expected improvement is a reasonable
optimiser and how it compares with alternatives (E8); the objective itself
and the diagnostics the ledger copies, $L$, $L_0$, $\hat\Delta$, $r_{\rm eff}$
(E7, P5); the meaning of any searched axis (P1-P5); the job scripts'
resources beyond what the driver needs (P7); any claim about which
configuration wins or how fast a campaign converges -- no job of
`hpc/joint/` has run on the cluster (`[KB]` usage v1.3 S9), and every number
here is a property of the driver as written, exercised in the sandbox on a
synthetic ledger or a closed form marked so.

---

## 1. Notation and symbols

A subset of E0's master table (same symbol, same type, same units) plus the
forty-eight symbols this document adds, which E0 v1.6 declares in its
"Search driver (P6)" group under convention 14 (the GP surrogate's mean and
variance and the acquisition function were reserved for E8 and are needed
first here, as the flow's symbols were by P4). Status in the Provenance
model's sense is given per knob in S3.1 and S3.4.

| Symbol | Name / Meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| $\varkappa$ | search-axis index: the position of an axis in `JOINT_KNOB_ORDER`; not $\kappa_k$, not $\kappa_S$ | $\varkappa \in \{1, \dots, 23\}$ | -- | S3.2 |
| $\mathcal{X}_\varkappa$, $\mathcal{X}$ | the range of search axis $\varkappa$ (an interval, an integer interval or a finite set), and the joint search space, their product (plan S5.1's $\mathcal{X}$) | set; $\mathcal{X} = \prod_{\varkappa} \mathcal{X}_\varkappa$ | mixed | S3.2 |
| $\mathrm{lo}_\varkappa, \mathrm{hi}_\varkappa$ | the two ends of a numeric axis's range; not the prior box's $a_k, b_k$ | reals or integers, $\mathrm{lo}_\varkappa < \mathrm{hi}_\varkappa$ | as the axis | S3.2 |
| $\mathcal{V}$ | the set of FREE axes of the campaign in play; its complement is pinned | $\mathcal{V} \subseteq \{1, \dots, 23\}$ | -- | S3.2 |
| $n_{\rm free}$ | the number of free axes, $\lvert \mathcal{V} \rvert$: 12, 19, 16, 23 for `S-A1`, `S-A2`, `S-A5`, `S-A25` | $\mathbb{N}$ | -- | S3.1 |
| $n_{\rm dim}$ | the dimension of the surrogate's input after scikit-optimize's transformation (one-hot categoricals, log-scaled and unit-normalised numerics): 14, 25, 18, 29 | $\mathbb{N}$ | -- | S3.2 |
| $\mathcal{X}_{\rm free}$ | the campaign's search space as the optimiser sees it, $\prod_{\varkappa \in \mathcal{V}} \mathcal{X}_\varkappa$ | set | mixed | S3.2 |
| $\mathsf{c}$ | a configuration: one value per axis of `JOINT_KNOB_ORDER`, $\mathsf{c}_\varkappa \in \mathcal{X}_\varkappa$, plus bookkeeping fields the code attaches (`_campaign`, `_strict_semihard_projected`, the two shuffle fields); not the class $c$ | $\mathsf{c} \in \mathcal{X}$ | mixed | S3.2 |
| $\mathsf{c}_{\rm dsn}$, $\mathsf{c}_{\rm rep}$ | the two switch coordinates of a configuration (`dsn_on`, `rep_on`) | $\{0, 1\}$ | -- | S3.2 |
| $\mathsf{c}^{\rm can}_\varkappa$ | the canonical value of axis $\varkappa$: the `INACTIVE_CANONICAL` entry, with `n_posterior_draws` resolved to its lower bound $4 d_\theta$ | $\mathcal{X}_\varkappa$ | as the axis | S3.2 |
| $\mathsf{x}$ | a search point: the free coordinates of a configuration in `JOINT_KNOB_ORDER`, the positional list scikit-optimize proposes and is told; not the window $x$ | $\mathsf{x} \in \mathcal{X}_{\rm free}$ | mixed | S3.2 |
| $\mathcal{K}$ | canonicalisation, `canonicalise_config`: pins every coordinate the configuration does not read and projects the loss condition; idempotent | $\mathcal{K} : \mathcal{X} \to \mathcal{X}$ | -- | S3.2 |
| $\Pi_{\rm leg}$ | the DSN's legality projection, `condition_space.project_condition`: moves only `strict_semihard`; not $\Pi_\theta$ | map on (mining, loss, filter) triples | -- | S3.2 |
| $\imath$ | ledger-observation index: the $\imath$-th completed, untagged, finite trial of a campaign in file order; not the row index $i$ | $\imath \in \{1, \dots, n_{\rm obs}\}$ | -- | S3.2 |
| $n_{\rm obs}$ | observations the surrogate is told: completed, untagged trials with a finite `nll` | $\mathbb{N}_0$ | -- | S3.2 |
| $n_{\rm init}$ | random points before the surrogate is consulted (`--n-initial-points`); scikit-optimize's `n_initial_points` | $\mathbb{N}$ | -- | S3.1 |
| $n_{\rm batch}$ | configurations proposed per round (`--n-points`) | $\mathbb{N}$ | -- | S3.1 |
| $L(\mathsf{c})$, $L_\imath$ | the objective at configuration $\mathsf{c}$ -- the ledger's `nll`, which is the runner's $L$ on the report split (F-al) -- and the value of observation $\imath$ (computed level: one run, one seed) | $\mathbb{R}$ | nats/row | S3.2 |
| $\bar L_{\rm obs}$, $s_{\rm obs}$ | mean and standard deviation of $L_1, \dots, L_{n_{\rm obs}}$, the standardisation scikit-learn applies before fitting the kernel | $\mathbb{R}$, $\mathbb{R}_{\ge 0}$ | nats/row | S3.2 |
| $\tilde L_\imath$ | the standardised target, $(L_\imath - \bar L_{\rm obs}) / s_{\rm obs}$, the number the kernel is fitted to | $\mathbb{R}$ | dimensionless | S3.2 |
| $k_{\rm GP}$ | the surrogate's covariance function on the transformed coordinates, eq. (P6.4) | $\mathcal{X}_{\rm free} \times \mathcal{X}_{\rm free} \to \mathbb{R}$ | dimensionless | S3.2 |
| $k_{\rm M}$ | the Matern kernel of smoothness $5/2$ with one length scale per transformed coordinate (anisotropic), unit amplitude | kernel | dimensionless | S3.2 |
| $a_{\rm GP}$ | the kernel's amplitude (`ConstantKernel`), fitted in $[0.01, 1000]$ | $\mathbb{R}_{>0}$ | dimensionless | S3.2 |
| $\sigma_{\rm noise}$ | the surrogate's observation-noise standard deviation in *standardised* units; $\sigma^2_{\rm noise}$ is the `WhiteKernel` level: fixed at $\sigma_{\rm seed}^2$ when `--sigma-seed` is given, fitted otherwise | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\mu_{\rm GP}(\mathsf{x})$, $\sigma_{\rm GP}(\mathsf{x})$ | the surrogate's posterior predictive mean and standard deviation at point $\mathsf{x}$ given the $n_{\rm obs}$ observations, un-standardised back to nats/row (analytic level with respect to the GP's own model; computed from the ledger) | $\mathbb{R}$, $\mathbb{R}_{\ge 0}$ | nats/row | S3.2 |
| $L_{\rm inc}$ | the incumbent: the smallest observed value, $\min_\imath L_\imath$ (lies included while a batch is being built); the constant-liar value | $\mathbb{R}$ | nats/row | S3.2 |
| $L_{\rm lie}$ | the constant-liar value told to the optimiser's copy for each point of a batch: $L_{\rm inc}$, or $0.0$ on an empty ledger | $\mathbb{R}$ | nats/row | S3.2 |
| $\xi_{\rm EI}$ | the expected-improvement margin (scikit-optimize `xi`), 0.01; not the residual $\xi$ | $\mathbb{R}_{\ge 0}$ | nats/row | S3.2 |
| $\Phi$ | the standard normal distribution function; $\Phi'$ its density | $\mathbb{R} \to (0, 1)$ | -- | S3.2 |
| $\tau_{\rm stop}$ | the escalation threshold: the improvement of the best-so-far over the last $w_{\rm esc}$ observations that counts as "still descending" (`--sigma-seed`, or 0.0 without it); not the step index $\tau$ | $\mathbb{R}_{\ge 0}$ | nats/row | S3.2 |
| $w_{\rm esc}$ | the escalation window, $\min(\max(5, \lfloor n_{\rm obs}/3 \rfloor), n_{\rm obs} - 1)$ | $\mathbb{N}_0$ | observations | S3.2 |
| $\delta_{\rm esc}$ | the recent improvement: the best-so-far before the window minus the best-so-far now | $\mathbb{R}_{\ge 0}$ | nats/row | S3.2 |
| $\epsilon_{\rm edge}$ | the boundary band of a free numeric axis, $10^{-6} \max(1, \lvert \mathrm{lo}_\varkappa \rvert, \lvert \mathrm{hi}_\varkappa \rvert)$ (`rel_tol`) | $\mathbb{R}_{>0}$ | as the axis | S3.2 |
| $\jmath$ | finalist index, by rank on the objective; not the direction index $j$ | $\jmath \in \{1, \dots, K_{\rm fin}\}$ | -- | S3.2 |
| $K_{\rm fin}$ | the number of finalists (`--top-k`); the handoff's $K$, a letter E0 gives to the classes of a metric batch | $\mathbb{N}$ | -- | S3.1 |
| $n_{\rm ctrl}$ | shuffled-control runs per finalist (`--n-control`) | $\mathbb{N}$ | -- | S3.1 |
| $\hat\Delta^{\rm ctrl}_\jmath$ | the gain $L_0 - L$ of one control run of finalist $\jmath$ (the ledger's `delta` of a record tagged `control`; computed level) | $\mathbb{R}$ | nats/row | S3.2 |
| $\bar\Delta^{\rm ctrl}_\jmath$, $s^{\rm ctrl}_\jmath$ | mean and sample standard deviation (`ddof=1`) of finalist $\jmath$'s $n_{\rm ctrl}$ control gains | $\mathbb{R}$, $\mathbb{R}_{\ge 0}$ | nats/row | S3.2 |
| $t^{\rm ctrl}_\jmath$ | the per-finalist one-sided statistic, eq. (P6.10) | $\mathbb{R}$ | dimensionless | S3.2 |
| $\mathrm{p}_\jmath$, $\tilde{\mathrm{p}}_\jmath$ | its p-value, and the Holm-adjusted p-value, eq. (P6.11); upright like $\mathrm{p}_{\rm grp}$, never the latent dimension $p$ | $(0, 1]$ | -- | S3.2 |
| $\delta_{\rm floor}$ | the hard minimum a finalist's gain must exceed whatever its p-value (`--floor`); not $\delta_{\min}$ | $\mathbb{R}$ | nats/row | S3.2 |
| $s_{\rm perm}$ | the permutation seed written into a control's configuration (`shuffle_seed`, $1000 + $ the control's index); never read by the runner (F-an) | $\mathbb{N}$ | -- | S3.2 |
| $s_{\rm seed}$ | the run seed (`--seed`): the runner's seed, the trial id's `seeds` entry and the surrogate's `random_state`, one number in three roles (F-as) | $\mathbb{N}_0$ | -- | S3.1 |
| $\sigma_{\rm seed}$, $n_{\rm seed}$ | across-seed standard deviation of an arm's held-out NLL; seeds behind one finalist's gain (`--n-seeds`, the handoff's `n_s`) | $\mathbb{R}_{\ge 0}$, $\mathbb{N}$ | nats/row; -- | S3.1 |
| $\delta_{\min}$ | the "learned nothing" floor of gate G1 (`--delta-min-provisional` carries a provisional value of it into the control table) | $\mathbb{R}$ | nats/row | S3.3 |
| $\alpha_{\rm H}$ | the family-wise level of the Holm step-down (`--alpha`) | $(0, 1)$ | -- | S3.2 |
| $M_{\rm ens}$ | ensemble members per finalist in the plan's Stage 4 (5); the driver has none (`n_members = 1`, F-ap) | $\mathbb{N}$ | -- | S3.5 |
| $L$, $L_0$, $\hat\Delta$ | held-out NLL on the report split, the prior floor, the gain $L_0 - L$ | $\mathbb{R}$ | nats/row | S3.2 |
| $L_{\rm sel}, L_{\rm gate}$ | $L$ on the selection split and on a gate split (the plan's roles; neither is what the ledger holds, F-al, F-am) | $\mathbb{R}$ | nats/row | S3.2 |
| $d_\theta$, $E$, $p$ | parameter dimension, embedding dimension, latent dimension of a bank: the shape anchors `--d-theta`, `--embedding-dim`, `--p` | $\mathbb{N}$ | -- | S3.1 |
| $N_{\rm train}$ | rows of the training split (the `--n-train` anchor) | $\mathbb{N}$ | rows | S3.3 |
| $S_{\rm mc}$ | posterior draws per well in the replicate term (`n_posterior_draws`), floor $4 d_\theta$ | $\mathbb{N}$ | draws | S3.2 |
| $n_{\rm hid}$ | hidden features of the flow (`hidden_features`), range $[52, 256]$ at $(26, 12)$ | $\mathbb{N}$ | -- | S3.3 |
| $B_{\rm sim}$ | the simulated batch size (`batch_size_npe`), trimmed by `--n-train` | $\mathbb{N}$ | rows | S3.3 |
| $\eta$, $\upsilon_1$, $\gamma_{\rm wd}$ | the three optimiser axes (P5) | $\mathbb{R}_{>0}$, $(0, 1)$, $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $\lambda_{\rm dsn}$, $\lambda_{\rm rep}$ | the two loss weights the runner takes, $10^{\mathsf{c}_\varkappa}$ of the searched `log10_lambda_*` when the switch is on and 0 when off | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.1 |
| $n_{\rm ep}, n_{\rm step}$ | epochs and steps per epoch of every trial (the job's `EPOCHS`, `STEPS_PER_EPOCH`; not in the trial id) | $\mathbb{N}$ | -- | S3.1 |
| $\theta$, $z$ | parameters and embedding of one row; the control permutes the $(\theta, z)$ pairing of the training split | $\Theta$, $S^{E-1}$ | -- | S3.2 |
| $n$ | a generic count, always qualified in prose | $\mathbb{N}$ | -- | S3.2 |

### 1.1 Conventions

1. **"Split" names three different things here, kept apart.** The runner's
   grouped split has three parts -- train, selection, report (P5) -- and
   nothing else exists at `834eb41`. The handoff's "rank on validation, gate
   on the third split" and the plan's S2.4/S4.4 name roles for those parts.
   The driver's `--rank-split` and `--gate-split` are *strings written into
   `finalists.json`*; they select nothing (F-am). The document writes
   "split" for a part of the data, "role" for what the plan assigns to it,
   and "label" for the two flags.
2. **"Campaign" and "arm" are two objects.** A campaign (`S-A1`, `S-A2`,
   `S-A5`, `S-A25`) is a partition of the 23 axes into free and pinned
   (`CAMPAIGNS`, `joint_space.py:375-398`); an arm (`A1`, `A2`, `A5`,
   `shuffled`, ...) is a named experiment of the runner (P5, E4). A
   configuration maps to an arm through its two switches alone
   (`arm_for_config`, `npe_tune_joint.py:230-251`); a campaign that frees
   `dsn_on` evaluates `A1` and `A2` configurations side by side.
3. **Two senses of "seed".** $s_{\rm seed}$ is the driver's `--seed`, one
   integer with three roles (F-as): the runner's `--seed`, the `seeds` list
   hashed into the trial id, and the surrogate's `random_state`.
   $s_{\rm perm}$ is the control's `shuffle_seed`, a field of the control's
   configuration that nothing downstream reads (F-an). The runner's own
   permutation seed is $s_{\rm seed} + 777$ (`run_joint_arms.py:381`).
4. **Two senses of "noise".** $\sigma_{\rm seed}$ is the across-seed
   standard deviation of the objective, a quantity of the training
   (measured by the standalone tuner's `baseline`, typed by hand here);
   $\sigma_{\rm noise}$ is the white-kernel level the surrogate assumes, in
   the units of the standardised target $\tilde L_\imath$. The driver sets
   $\sigma^2_{\rm noise} = \sigma^2_{\rm seed}$ as if the two were in the
   same units; they are not (F-aq, eq. (P6.5)).
5. **Two senses of "Bayesian optimisation" in this literature (R6).** Here
   it is a search over hyper-parameters with a GP surrogate. In the SBI
   literature the project's PDFs cite, "Bayesian optimisation for
   likelihood-free inference" (BOLFI) and GP-surrogate ABC use the same
   machinery as the *inference engine*, modelling a discrepancy over
   $\theta$ (`[KB-PDF]` the frontier review p.4; Goncalves et al. p.30;
   the multilevel preprint p.3, references only). Nothing of that second
   sense is used in Stage 4.
6. **Levels (R8).** $L(\mathsf{c})$ as a function on $\mathcal{X}$ is the
   object the search minimises; the ledger holds one realisation per
   configuration, $L_\imath$, at one seed. $\mu_{\rm GP}$ and
   $\sigma_{\rm GP}$ are the surrogate's posterior given those realisations
   (analytic with respect to the GP model, computed from the ledger). The
   control gains $\hat\Delta^{\rm ctrl}_\jmath$ are realisations; their mean
   and standard deviation are computed from the $n_{\rm ctrl}$ runs; the
   p-value treats them as draws from one law, which F-an shows they are not
   at `834eb41`.
7. **The objective's split.** The ledger's `nll` is the runner's `L`, scored
   on the *report* split (F-al, P5 S3.8); every "rank on the selection
   split" of the plan is, in the code, a rank on the report split. The
   document says "the objective" or "`nll`" for the number the ledger holds
   and names the split only where the distinction acts.
8. **Status, per surface** as in P5: configured (a flag, a job variable or
   the space), configured by code (a constant below any flag), computed
   (derived from the ledger), analytic (a closed form), derivation-only
   (named in the justification, never evaluated).
9. **Tags.** As the set uses them: `[REPO file:line]` at `834eb41`;
   `[REPO skopt v0.10.2]` and `[REPO sklearn]` for the two libraries'
   source; `[RAN]` with the block of `tools/p6_numbers.py` that prints the
   number; `[KB]`, `[KB-PDF p.n]`, `[PubMed full text]`,
   `[PubMed abstract only]`, `[textbook, from memory]`, `[reasoning]`.

## 2. Glossary

Ordered by first appearance in S3, because the concepts build on each
other; the pointer names where the term becomes operative. *Everyday
meaning differs* is flagged where it does.

- **Campaign** -- a named choice of which of the 23 axes are searched and
  which are pinned, with the pinned values resolved against the shape
  anchors; `S-A1` (NPE only, 12 free), `S-A2` (the DSN term free, 19),
  `S-A5` (the replicate term free, 16), `S-A25` (everything free, 23; no
  arm can run it). S3.1.
- **Shape anchors** -- the three integers `--p`, `--embedding-dim`,
  `--d-theta` (and the optional `--n-train`) that resolve the two
  shape-dependent ranges, `hidden_features` and `n_posterior_draws`, and
  trim `batch_size_npe`; they must match the bank and must be passed
  identically to every subcommand (F-ar). S3.1, S3.3.
- **Inactive axis** -- an axis a configuration does not read: the six
  DSN-loss axes when `dsn_on = 0`, the three replicate axes when
  `rep_on = 0`, and the loss hyper-parameters outside the active set of the
  chosen loss type. *Everyday meaning differs*: "inactive" is not "unused"
  -- under `joint` and `joint_sep` the margin is still read as $2 m_{\cos}$
  (P2), which is why an inactive axis is pinned to the base config's value
  and never to zero. S3.2.
- **Canonicalisation** ($\mathcal{K}$) -- the map that pins every inactive
  axis to its canonical value and replaces the (mining, loss, filter)
  triple by its legal projection, so that two points differing only in
  inactive coordinates are the same configuration, the same key and the
  same trial id. S3.2.
- **Legality projection** ($\Pi_{\rm leg}$) -- the DSN's rule that zeroes
  `strict_semihard` under `triplet` (no such filter) and under `hard`
  mining (the provably empty cell); mining and loss type never move (P2).
  S3.2.
- **Config key** -- the JSON of the 23 axis values alone, sorted; the
  identity used for deduplication and for matching a record to its gain.
  Bookkeeping and shuffle fields are outside it. S3.2.
- **Trial id** -- twelve hex characters of a SHA-256 over the *whole*
  configuration dict (bookkeeping fields included), the split hash, the
  contract digest, `n_members = 1`, the seed list and the tag; the file
  name of a pending spec, a ledger record and a control spec. S3.2.
- **Ledger** -- the directory `<results>/<campaign>/trials/`, one JSON per
  evaluated trial, written atomically; the only state of the search. The
  **pending set** is `<results>/<campaign>/pending/`, the proposals not yet
  evaluated; its sorted listing is the PBS array's index map. S3.1.
- **Surrogate** -- the Gaussian-process regression of the objective over the
  free axes, rebuilt from the ledger at every `propose` ("stateless"). S3.2.
- **Initial design** -- the first $n_{\rm init}$ points, drawn uniformly
  from the (transformed) space one at a time; positional: the first
  $n_{\rm init}$ entries of the ledger count as the design whatever proposed
  them. S3.2.
- **Expected improvement (EI)** -- the acquisition function: the expected
  amount by which a point's objective would undercut the incumbent minus a
  margin, under the surrogate's predictive law. S3.2.
- **Incumbent** -- the smallest value observed so far, $L_{\rm inc}$; here
  the smallest *noisy* value, lies included while a batch is being built.
  S3.2.
- **Constant liar** -- the batch heuristic: ask one point, pretend it
  returned the incumbent, refit, ask the next; `cl_min` is the variant whose
  pretended value is the minimum. S3.2.
- **Escalation verdict** -- the three-condition rule that says whether a
  campaign needs more budget or a wider range: still descending beyond
  $\tau_{\rm stop}$, best configuration on a boundary, or fewer than 12
  observations. S3.2.
- **Boundary axis** -- a free numeric axis on which the best configuration
  sits within $\epsilon_{\rm edge}$ of an end of its range; categoricals and
  switches have no edge. S3.2.
- **Finalists** -- the $K_{\rm fin}$ configurations with the smallest
  objective in the ledger, in rank order. S3.2.
- **Control** (per-finalist, shuffled-pairs) -- the finalist's exact
  recipe trained on a bank whose $(\theta, z)$ pairing is permuted within
  the training split, keeping both marginals; its gain is the "learned
  nothing" floor of that recipe. *Everyday meaning differs*: not a
  negative-control arm chosen once per bank (that is the plan's
  `baseline`, Tier 1 of the handoff), but one per finalist. S3.2.
- **Winner's curse / selection bias** -- the finalists' objective values are
  minima over the whole search, so each is biased downward (optimistic) as
  an estimate of that configuration's expected objective; a test calibrated
  for one pre-chosen configuration understates the false-pass rate when
  applied to a selected one. S3.2, F-am.
- **Family-wise error rate; Holm step-down** -- the probability of at least
  one false rejection among $K_{\rm fin}$ tests, and the step-down procedure
  that controls it: p-values sorted, the smallest compared with
  $\alpha_{\rm H}/K_{\rm fin}$, the next with $\alpha_{\rm H}/(K_{\rm fin} -
  1)$, and so on, stopping at the first failure. S3.2.
- **Prediction interval, not confidence interval** -- the $\sqrt{1/n_{\rm seed}
  + 1/n_{\rm ctrl}}$ factor of the control test accounts for the finalist's
  own realisation noise, not only the control mean's. S3.2.
- **Partial dependence** -- the average of a fitted model's prediction over
  all inputs but one, as a function of that one; the plan's read-out of
  "does this term help" on `dsn_on`, `rep_on`. S3.5, F-ap.
- **Index map** -- the sorted listing of the pending directory that maps a
  PBS array index to a trial; running `propose` mid-array renumbers it.
  S3.1.
- **Standardised target** -- scikit-learn's `normalize_y`: the kernel is
  fitted to $(L_\imath - \bar L_{\rm obs})/s_{\rm obs}$ and predictions are
  mapped back; a fixed noise level lives in the standardised units. S3.2,
  F-aq.

---

## 3. Main body

### 3.1 Where the search knobs live, and what reaches the trainer

*Establishes the surfaces of every knob of P0 Table D, the files the driver
reads and writes, which campaign evaluates which arm, and the exact command
line a configuration becomes -- the one integration surface between the
search and the training.*

One would expect a search driver to own its knobs. Here the knobs live on
four surfaces that never see each other: the parser of `npe_tune_joint.py`
(every flag), the module constants of `joint_space.py` (the pins, the
tolerance, the campaign table), the job script's `-v` variables (which
re-default the anchors and the seed for every array element), and
scikit-optimize's own constructor defaults (the kernel, the acquisition
margin, the optimiser of the acquisition), which nothing in the repository
sets. A configuration proposed under one surface and evaluated under
another is a different configuration (F-ar); a knob typed on one surface and
assumed on another is a different experiment (F-aq).

**Table P6.1 -- the subcommands and their knobs** (`build_parser`,
`npe_tune_joint.py:743-812`; `common` flags on every subcommand,
`common[need_shards]` on `argv` and `evaluate` only). Status: configured
unless said otherwise.

| subcommand | what it does (`:350-738`) | knobs beyond `common` | reads | writes |
|---|---|---|---|---|
| `space` | prints the campaign's ranges with their provenance and the derived flags; exit 1 if a FREE axis is unreachable (`UNREACHABLE = {}`, so never at `834eb41`) | -- | nothing | stdout |
| `propose` | rebuilds the surrogate from the ledger, excludes the pending set, asks for a batch, drops duplicates by trial id, writes the rest as pending specs | `--n-points` 8, `--n-initial-points` 12, `--sigma-seed` None | `trials/*.json`, `pending/*.json` | `pending/<id>.json` |
| `argv` | prints the runner command line of one configuration (`--config-json` or `--pending-id`) | the shards, `--out-dir`, `--epochs` 10, `--steps-per-epoch` 25, `--runner`, `--dsn-main-dir`, `--sbi-hpc-dir`, `--dry-run` | the spec file | stdout |
| `evaluate` | runs the runner as a subprocess on one configuration, reads its record back, writes a ledger record, removes the pending spec of the same id | as `argv`, plus `--tag` '' | the spec file; the runner's `*_seed<s>.json` | `trials/<id>.json`; the runner's run directory `<out-dir>/<id>/` |
| `status` | ranks the ledger, prints the top rows and the escalation verdict | `--top` 10, `--sigma-seed` None | `trials/*.json` | stdout |
| `finalists` | takes the $K_{\rm fin}$ smallest objectives, attaches each one's gain, writes `finalists.json` with the two labels | `--top-k` 3, `--rank-split` 'sel', `--gate-split` 'gate' | `trials/*.json` | `<campaign>/finalists.json` |
| `controls --plan` | for every finalist, $n_{\rm ctrl}$ control configurations with `shuffle_pairs = 1`, `shuffle_seed = 1000 + s`, identity asserted; written as specs | `--n-control` 5 | `finalists.json` | `<campaign>/control_specs/<id>.json` |
| `controls --score` | collects each finalist's control records, runs the one-sided $t$ test and the Holm step-down, names the best-ranked survivor | `--n-seeds` 1, `--alpha` 0.05, `--floor` 0.0, `--delta-min-provisional` nan | `finalists.json`, `control_specs/*.json`, `trials/*.json` tagged `control` | `<campaign>/controls.json` |
| `report` | prints all $K_{\rm fin}$ verdicts; refuses when `controls.json` is absent | -- | `controls.json` | stdout |

The `common` flags (`:749-759`): `--campaign` (default `S-A1`),
`--results-dir` (`tune_joint`), the three shape anchors `--p` 26,
`--embedding-dim` 12, `--d-theta` 26, `--n-train` None, `--split-hash` '',
`--contract-digest` '', `--seed` 0. Every one of them is parsed by every
subcommand, and the anchors are *used* by every subcommand that builds a
spec -- every one but `controls` and `report` (`_spec_from_args`, `:342-347`;
`controls` reads `finalists.json` and the ledger as written).

**The campaigns and the arms** (`CAMPAIGNS`, `joint_space.py:375-398`;
`arm_for_config`, `npe_tune_joint.py:230-251`; `[RAN]` B1, B4):

| campaign | pinned | free ($n_{\rm free}$) | surrogate dims ($n_{\rm dim}$) | Categorical / Integer / Real | arm(s) evaluated |
|---|---|---|---|---|---|
| `S-A1` | the 6 DSN-loss axes at their canonical values and `dsn_on = 0`; the 3 replicate axes and `rep_on = 0` (11) | 12 | 14 | 3 / 4 / 5 | `A1` |
| `S-A2` | the replicate block (4) | 19 | 25 | 6 / 4 / 9 | `A1` where `dsn_on = 0`, `A2` where 1 |
| `S-A5` | the DSN block (7) | 16 | 18 | 4 / 5 / 7 | `A1` where `rep_on = 0`, `A5` where 1 |
| `S-A25` | nothing | 23 | 29 | 7 / 5 / 11 | `A1`, `A2`, `A5`; `dsn_on = rep_on = 1` raises `ValueError` -- "no Stage 3 arm trains both" |

The resolved pins at $(p, E, d_\theta) = (26, 12, 26)$: `log10_lambda_dsn`
0.0, `loss_type` `triplet`, `mining_strategy` `hard`, `margin` 0.2,
`angular_alpha_deg` 18.0, `lambda_sep` 0.1, `log10_lambda_rep` 0.0,
`warmup_frac_rep` 0.0, `n_posterior_draws` 104 (`[RAN]` B1); every one lies
inside its axis's range, the invariant J27 asserts and the reason a pinned
configuration can be replayed into the optimiser at all
(`joint_space.py:152-156`). The campaign table's note on `S-A25` -- "Not
one of the four campaigns of S5.2 ... Budget it separately" (`:391-397`) --
is the code's; the plan's S5.2 lists `S-A0` (the DSN's own campaign followed
by the standalone NPE tuner on frozen $z$), `S-A1`, `S-A2`, `S-A5` at equal
budget, and `S-A0` has no entry in the driver: it is run with the other two
stacks (S3.5).

**The results directory** (`:66-69, 258-263, 627-628`): `<results>/<campaign>/`
holds `pending/` (proposals), `trials/` (the ledger), `control_specs/` (the
planned controls), `finalists.json` and `controls.json`. A control is
*evaluated* into `trials/` like any trial, distinguished by its `tag`; it is
never in `pending/` (F-ao). The runner's own records go to
`<out-dir>/<trial id>/`, one directory per trial, and `evaluate` reads back
whichever `*_seed<s>.json` it finds there (`:508-509`).

**What reaches the trainer.** `build_argv` (`:135-227`) turns a
configuration into the runner's command line: the arm, the shards, the run
directory, the seed and the schedule first (`:187-195`); then nineteen axes
by flag (`AXIS_TO_FLAG`, `:86-106`), strings as they are, integers as `int`,
floats at `repr` precision (`:197-204`); then `--strict-semihard` from the
space's fixed table (`:206-211`); then the two derived weights,
$\lambda_{\rm dsn} = 10^{\mathsf{c}_\varkappa}$ of `log10_lambda_dsn` when
`dsn_on = 1` and `0.0` otherwise, likewise $\lambda_{\rm rep}$ (`:213-217`).
For an `S-A1` configuration that is 58 tokens and 28 flags `[RAN]` B4:

```
python run_joint_arms.py --arm A1 --sim-shards S/*.npz --out-dir runs/x --seed 0
  --epochs 10 --steps-per-epoch 25 --depth-exponent 4 --width-multiplier 2.0
  --block-family 1 --embedding-size 12 --head-fusion 0 --dropout 0.1
  --warmup-frac-rep 0.0 --n-posterior-draws 104 --hidden-features 128
  --num-transforms 8 --lr 0.0005 --weight-decay 0.0001 --b-sim 512
  --one-minus-beta1 0.05 --loss-type triplet --mining-strategy hard --margin 0.2
  --angular-alpha-deg 18.0 --lambda-sep 0.1 --strict-semihard 1
  --lambda-dsn 0.0 --lambda-rep 0.0
```

Four readings of this line carry through the document.

1. **Every pinned axis is passed too.** The inactive DSN-loss axes travel
   at their canonical values (`--margin 0.2 --angular-alpha-deg 18.0
   --lambda-sep 0.1 --loss-type triplet --mining-strategy hard`) and the
   replicate axes at theirs; on `A1` the runner builds no DSN loss and no
   replicate criterion (`lambda_dsn > 0`, `lambda_rep > 0` gate them,
   `run_joint_arms.py:529-532`), so these values are inert there, but they
   are in the record, which is what lets a ledger config be replayed.
2. **Four of the space's fixed knobs do not travel, and one of them
   matters.** `num_bins` (space 10, runner default 8: the trained flow has
   8 bins, F-a), `head_pool_ops` (space code 1, built code 0, F-b),
   `sep_warmup_frac` (space 0.0, runner default 0.0: agreement by
   coincidence, F-w) and `one_minus_beta2` (space $10^{-3}$, library
   $\beta_2 = 0.999$: agreement by two independent defaults, P5 F-i). Only
   `strict_semihard` is passed. `UNREACHABLE` is empty, so `space` reports
   "every free axis of campaign ... reaches the trainer" for every campaign
   -- true of the *free* axes, not of the fixed table.
3. **No shuffle flag exists.** `shuffle_pairs` and `shuffle_seed` are
   fields of a control's configuration (`control_config_from`,
   `joint_space.py:697-722`) and `arm_for_config` reads the first to name
   the arm `shuffled` (`:239-240`); neither is in `AXIS_TO_FLAG`, so the
   permutation seed never leaves the driver (F-an).
4. **The schedule travels from the job, not from the configuration.**
   `--epochs` and `--steps-per-epoch` are arguments of `evaluate`
   (defaults 10 and 25; `EPOCHS`, `STEPS_PER_EPOCH` of `joint_tune.pbs`),
   and the trial id does not hash them (S3.2): two evaluations of one
   configuration at different schedules collide on the same ledger file.

**The job and the launcher** (`joint_tune.pbs`; `launch_joint_tune.sh`; P7
for resources). One array element evaluates one pending spec: the element
lists `pending/*.json` once, sorted, and takes entry `PBS_ARRAY_INDEX`
(`:112-129`); an index beyond the listing exits 0 with "stale array range"
(`:124-128`). The element calls `evaluate --pending-id <id>` with
`--seed $SEED` (default 0), `--epochs`, `--steps-per-epoch`, `--p`,
`--embedding-dim`, `--d-theta` from its own variables (`:141-152`; defaults
26 / 12 / 26), and `--split-hash`, `--contract-digest`, `--tag` only when
set (`:153-156`). The launcher forwards `CAMPAIGN`, `RESULTS_DIR`,
`SIM_SHARDS`, `OUT_DIR` and, when exported, `REAL_SHARDS`, `SPLIT_HASH`,
`CONTRACT_DIGEST`, `EPOCHS`, `STEPS_PER_EPOCH`, `SEED`, `SBI_HPC_DIR`
(`:135-143`); it forwards neither `P`, `EMBEDDING_DIM`, `D_THETA` nor
`TAG` (F-ar). Its guards: CR bytes in the job script (`:77-83`), the
`space` subcommand's `BLOCKING` line (`:92-105`), `SIM_SHARDS` and
`OUT_DIR` exported (`:115-120`); then `qsub -J 0-$LAST` with `LAST` the
pending count minus one, or `--max` minus one (`:129-144`); a dry run by
default (`:147-158`).

### 3.2 The search loop as run

*Establishes, as one explicit chain from `joint_space.py`,
`npe_tune_joint.py`, `npe_tune_search.py` and scikit-optimize 0.10.2, what a
round of the search computes: the space and the campaign (P6.1), the point
and the configuration with the three clamps (P6.2), the trial identity, the
surrogate (P6.3)-(P6.5), the acquisition and the batch (P6.6)-(P6.7), the
escalation verdict (P6.8)-(P6.9), the finalists and the control test
(P6.10)-(P6.11). Each step cites the line it is read from; the numbers are
`tools/p6_numbers.py`'s.*

Plan S5.1 promises "one space, one objective, one ledger", with the
mechanics of `npe_tune_search.py` -- a stateless `skopt.Optimizer` rebuilt
from the ledger each round, constant-liar batching, an escalation verdict
against the measured $\sigma_{\rm seed}$ -- and a canonicalisation "so that
points differing only in inactive coordinates build byte-identical configs".
The code keeps every one of those promises for the search. What the chain
below adds is what the promises leave unsaid: which coordinates the
surrogate actually sees, what "the observation noise" is in the surrogate's
own units, which seed drives what, and what the gate at the end measures.

**(a) The space and the campaign.** The 23 axes of `JOINT_KNOB_ORDER`
(`joint_space.py:92-105`) in five blocks (encoder 6, DSN loss 7, replicate
4, flow 2, optimiser 4) carry the ranges of `JointSpaceSpec` (`:212-241`),
two of them resolved against the anchors by `default_joint_space`
(`:262-330`): `hidden_features` by the width rule (P4 eq. (P4.12);
$[52, 256]$ at $(26, 12)$), `n_posterior_draws` by the floor $4 d_\theta$
($[104, 400]$ at $d_\theta = 26$), and `batch_size_npe` trimmed by
`--n-train` (P5 S3.3.4). The search space is

$$\mathcal{X} = \prod_{\varkappa = 1}^{23} \mathcal{X}_\varkappa, \qquad
\mathcal{X}_{\rm free} = \prod_{\varkappa \in \mathcal{V}} \mathcal{X}_\varkappa,
\tag{P6.1}$$

with $\mathcal{V}$ the campaign's free axes in `JOINT_KNOB_ORDER` and the
pinned axes fixed at the values of `resolved_pins` (`:421-434`): the
`INACTIVE_CANONICAL` table (`:161-185`) with `n_posterior_draws` resolved to
its lower bound. The optimiser never sees $\mathcal{X}$; it sees
$\mathcal{X}_{\rm free}$ through `space_dimensions` (`:441-472`): the two
switches, `block_family`, `head_fusion`, `batch_size_npe`, `loss_type` and
`mining_strategy` as `Categorical`; `depth_exponent`, `embedding_size`,
`num_transforms`, `n_posterior_draws` as `Integer`; `hidden_features` as a
log-uniform `Integer`; `lr`, `lambda_sep`, `one_minus_beta1`, `weight_decay`
as log-uniform `Real`; the rest as uniform `Real`. Because the base
estimator is a GP, scikit-optimize normalises the dimensions
(`optimizer.py:312-313`): a numeric axis becomes a coordinate in $[0, 1]$
(on the log scale where the prior is log-uniform) and a categorical with
$n$ levels becomes $n$ one-hot coordinates, except that a two-level
categorical becomes one -- hence $n_{\rm dim} = 14, 25, 18, 29$ against
$n_{\rm free} = 12, 19, 16, 23$ `[RAN]` B1. The kernel's length scales live
on these coordinates, one per transformed dimension (S3.4).

**(b) The point and the configuration.** A proposal is a point
$\mathsf{x} \in \mathcal{X}_{\rm free}$; `config_from_point` (`:582-603`)
fills the pinned axes from the campaign, attaches `_campaign`, and
canonicalises. Canonicalisation (`canonicalise_config`, `:520-579`) is the
map $\mathcal{K} : \mathcal{X} \to \mathcal{X}$ that applies, in this order,

$$\begin{aligned}
\mathsf{c}_{\rm dsn} = 0 &:\quad \mathsf{c}_\varkappa \leftarrow \mathsf{c}^{\rm can}_\varkappa \ \text{ for the six DSN-loss axes;}\\
\mathsf{c}_{\rm dsn} = 1 &:\quad (\text{mining}, \text{loss}, \text{filter}) \leftarrow \Pi_{\rm leg}(\text{mining}, \text{loss}, \text{the fixed filter}),
\ \text{ and } \mathsf{c}_\varkappa \leftarrow \mathsf{c}^{\rm can}_\varkappa \ \text{ for the loss hyper-parameters outside the active set;}\\
\mathsf{c}_{\rm rep} = 0 &:\quad \mathsf{c}_\varkappa \leftarrow \mathsf{c}^{\rm can}_\varkappa \ \text{ for the three replicate axes,}
\end{aligned}
\tag{P6.2}$$

where $\mathsf{c}_{\rm dsn}$ and $\mathsf{c}_{\rm rep}$ are the two switch
coordinates (`dsn_on`, `rep_on`), $\mathsf{c}^{\rm can}_\varkappa$ is the
canonical value of axis $\varkappa$ (`INACTIVE_CANONICAL` with the floor
resolved; the docstring labels the three clauses (a), (b), (c), `:526-530`,
and the code applies them in the order (a), (c), (b), `:558-577`), the
active set is the DSN's `active_loss_hps` --
`triplet` reads `margin`; `joint` reads `angular_alpha_deg`; `joint_sep`
reads `angular_alpha_deg`, `lambda_sep` and the fixed `sep_warmup_frac`
(`condition_space.py:120-124`) -- and $\Pi_{\rm leg}$ zeroes the filter
under `triplet` and under `hard` mining and never moves the other two
(`:222-239`). The filter the projection starts from is the space's fixed
`strict_semihard` (1 by default, `joint_space.py:322`), read from the spec
passed in; the result is stored as a bookkeeping field
`_strict_semihard_projected` and nothing else carries it (`:563-568`).
Checked `[RAN]` B1b: `joint_sep` with `easy_pos_semihard_neg` keeps the
filter at 1 and pins `margin` 0.7 to 0.2; `triplet` with `hard` keeps
`margin` and pins the other two, filter 0; `dsn_on = 0` pins all six and
attaches no filter field; $\mathcal{K} \circ \mathcal{K} = \mathcal{K}$ on
the example; the round trip `point_from_config` then `config_from_point` is
the identity on a canonical configuration (J21); `canonicalise_config`
without a spec raises for any configuration with `rep_on = 0`, because the
floor pin needs $d_\theta$ (`:188-198`), and for `rep_on = 1` it reads the
filter as 0 -- the bookkeeping field then differs while the key does not,
since the key is over the 23 axes alone (`config_key`, `:626-633`: 466
characters, 23 fields, blind to `_campaign`, a trial id and the shuffle
fields `[RAN]` B1b). The production paths always pass the spec
(`_load_config`, `npe_tune_joint.py:458-470`; `config_from_point` through
the adapter).

**(c) The trial identity.** `npe_tune_ledger.trial_id` (`:98-121`) hashes,
as sorted JSON, the *whole* configuration dict (bookkeeping fields and
shuffle fields included), the strings `--split-hash` and
`--contract-digest`, `n_members = 1`, the list `[--seed]` and the tag, and
keeps the first twelve hex characters of the SHA-256. Consequences `[RAN]`
B4: the same configuration under `--split-hash abc123`, under `--seed 1`
or under `--tag control` has a different id; the same configuration
without its `_campaign` field has a different id while its key is
unchanged; the schedule (`EPOCHS`, `STEPS_PER_EPOCH`) and the anchors are
not hashed except through the pins they resolve. The id is computed twice,
by `propose` with the driver's `--seed` and no tag (`:414-415`) and by
`evaluate` with the job's `--seed` and `--tag` (`:477-478`); the pending
spec is removed only when the two agree (`:530-533`). The record stores
the driver's `--split-hash` and then overwrites it with the runner's
`split_hash` when the runner's record carries one (`:497, 517-521`), so the
id's input and the record's field can differ.

**(d) The surrogate as built.** `build_optimizer`
(`npe_tune_search.py:199-244`) constructs
`skopt.Optimizer(dimensions, base_estimator="GP", n_initial_points, acq_func="EI",
acq_optimizer="auto", random_state=seed)` and, when `--sigma-seed` is
given, sets `opt.base_estimator_.noise = sigma_seed ** 2`
(`npe_tune_joint.py:395-396`; `npe_tune_search.py:229-237`). Inside
scikit-optimize `[REPO skopt v0.10.2]`: `cook_estimator("GP")` builds a
`GaussianProcessRegressor(normalize_y=True, noise="gaussian",
n_restarts_optimizer=2)` with kernel `ConstantKernel(1.0, (0.01, 1000))
* Matern(length_scale=1 per dimension, bounds (0.01, 100), nu=2.5)`
(`utils.py:364-392`); `acq_optimizer="auto"` resolves to `"lbfgs"` for a
GP (`optimizer.py:269-275`), with 10 000 random candidates and 5 restarts
(`:298-299`); the acquisition margin is `xi = 0.01` and `kappa = 1.96`
(`acquisition.py:40-41`; `kappa` is for LCB and unused here). At every
`tell` the estimator is cloned and refitted on all observations
(`optimizer.py:609-640`): with the float noise, `fit` appends
`WhiteKernel(noise_level=sigma_seed ** 2, noise_level_bounds="fixed")` to
the kernel (`gpr.py:196-202`), and after the fit it zeroes the white term
in `kernel_` and keeps the level in `noise_` so that predictions at a new
point carry no nugget (`:206-226`). The fitted kernel on twelve synthetic
observations reads `0.966**2 * Matern(length_scale=[0.01, 100, ...], nu=2.5)
+ WhiteKernel(noise_level=0)` with `noise_ = 0.0004` at `--sigma-seed
0.02` `[RAN]` B2 -- several length scales at their bounds 0.01 and 100 on
twelve points, which is the surrogate with too little data, not a fault.

The targets are standardised before the kernel is fitted
(`[REPO sklearn]` `_gpr.py` `fit`: `_y_train_mean`, `_y_train_std`, then
`y = (y - mean) / std`; the same lines at 1.6.1 and 1.9.1; scikit-optimize
copies them to `y_train_mean_`, `y_train_std_` for any scikit-learn $\ge$
0.23, `gpr.py:233-241`):

$$\tilde L_\imath = \frac{L_\imath - \bar L_{\rm obs}}{s_{\rm obs}}, \qquad
\imath = 1, \dots, n_{\rm obs},
\tag{P6.3}$$

and the model on the transformed coordinates is a zero-mean Gaussian
process on the standardised targets $\tilde L_\imath$ with covariance

$$k_{\rm GP}(\mathsf{x}, \mathsf{x}') = a_{\rm GP}\, k_{\rm M}(\mathsf{x}, \mathsf{x}') + \sigma^2_{\rm noise}\, \mathbb{1}[\mathsf{x} = \mathsf{x}'],
\tag{P6.4}$$

whose amplitude and length scales are fitted by marginal likelihood with
two restarts and whose noise level is fixed when `--sigma-seed` is given.
Because the white term is added in the standardised units of eq. (P6.3),
the noise variance it asserts in nats per row is

$$\sigma^2_{\rm noise}\, s^2_{\rm obs} = \sigma^2_{\rm seed}\, s^2_{\rm obs},
\qquad \text{i.e. an assumed seed spread of } \sigma_{\rm seed}\, s_{\rm obs}
\text{ nats/row, not } \sigma_{\rm seed}.
\tag{P6.5}$$

On the synthetic ledger of B2 ($s_{\rm obs} = 0.2276$ nats/row) a typed
`--sigma-seed 0.02` becomes an assumed spread of 0.0046 nats/row; it would
be 0.020 only on a ledger whose spread is 1 nats/row `[RAN]` B2 (F-aq). The
posterior predictive mean and standard deviation at a point,
$\mu_{\rm GP}(\mathsf{x})$ and $\sigma_{\rm GP}(\mathsf{x})$, are the
standard GP-regression formulas (`[textbook, from memory]`: Rasmussen and
Williams, ch. 2), evaluated by scikit-learn in the standardised units and
mapped back by $\bar L_{\rm obs}$ and $s_{\rm obs}$; scikit-optimize's
`predict` uses the nugget-free `kernel_` for the cross-covariance and the
training covariance with the nugget through `alpha_`, as its comment says.

**(e) The acquisition and the batch.** With the surrogate fitted,
scikit-optimize maximises the expected improvement over the incumbent
$L_{\rm inc} = \min_\imath L_\imath$ (`optimizer.py:640`, `y_opt =
np.min(self.yi)`):

$$\mathrm{EI}(\mathsf{x}) = \big(L_{\rm inc} - \xi_{\rm EI} - \mu_{\rm GP}(\mathsf{x})\big)\,
\Phi\!\left(\frac{L_{\rm inc} - \xi_{\rm EI} - \mu_{\rm GP}(\mathsf{x})}{\sigma_{\rm GP}(\mathsf{x})}\right)
+ \sigma_{\rm GP}(\mathsf{x})\,
\Phi'\!\left(\frac{L_{\rm inc} - \xi_{\rm EI} - \mu_{\rm GP}(\mathsf{x})}{\sigma_{\rm GP}(\mathsf{x})}\right)
\tag{P6.6}$$

for $\sigma_{\rm GP}(\mathsf{x}) > 0$ and $0$ otherwise
(`acquisition.py:309-317`), by L-BFGS from the best of 10 000 random
candidates with 5 restarts, in the transformed coordinates, the result
inverse-transformed and rounded onto the integer and categorical axes. The
incumbent is the smallest *observed* value, one noisy realisation: with a
noisy objective it is biased low, and the literature names the
alternatives for exactly this reason -- the best value observed "in the
deterministic case", and in the noisy case the best *mean estimate* over the
sampled designs or over the whole space, or an integration over the noise
that loses the closed form (`[PubMed full text]` Binois et al. 2025, S6).
Nothing in the driver or in scikit-optimize's `"EI"` uses them.

A batch of $n_{\rm batch}$ points is asked with `strategy="cl_min"`
(`npe_tune_search.py:275`; `optimizer.py:384-468`): a copy of the optimiser
is asked for one point, told that point with the lie

$$L_{\rm lie} = \min_\imath L_\imath = L_{\rm inc} \qquad (0.0 \text{ when the ledger is empty}),
\tag{P6.7}$$

refitted, asked again, $n_{\rm batch}$ times; the copy is discarded and the
real optimiser keeps only the points. This is the constant-liar heuristic
of batch EI, "replacing unknown values at selected points by pseudo-values"
(`[PubMed full text]` Binois et al. 2025). While the optimiser still has
initial points to draw ($n_{\rm obs} < n_{\rm init}$, counted positionally,
`optimizer.py:477-485, 584-597`), the copy's asks are uniform random draws
from each axis's prior, uniform in the transformed coordinates
(`space.rvs(random_state=rng)`; the default
`"random"` generator pre-generates nothing, `utils.py:411-430` `[RAN]` B2),
one draw per ask, each lie decrementing the count; the surrogate is
consulted from the first ask at which the count reaches zero. At the
defaults $n_{\rm init} = 12$, $n_{\rm batch} = 8$ `[RAN]` B3: round 1 is 8
random points; round 2 asks 4 random draws and then 4 points from a GP
fitted on 8 observations and 4 lies; round 3 is all GP. The design is
positional: the first 12 entries of the ledger are "the design" whatever
produced them.

`propose` (`npe_tune_search.py:247-294`; `npe_tune_joint.py:376-441`) then
drops any proposal whose key is already observed or pending, re-asks up to
20 times for the shortfall, and -- when a whole ask came back duplicated --
tells the real optimiser the worst observed value at the duplicate as a
nudge (`:287-293`); a second pass computes each survivor's trial id and
drops the ones already on disk (`npe_tune_joint.py:405-419`), printing
"change --seed for the next round" when fewer than $n_{\rm batch}$ were
written (`:436-440`). Two properties of this loop matter in practice
`[RAN]` B3. First, replay: a fresh optimiser at the same `--seed` on the
same ledger proposes the same points, which is what makes `propose`
reproducible (J31) -- and, in the design phase, what makes it repeat
itself: at 8 observations and the same seed, the 4 remaining random draws
of round 2 are round 1's first 4 points again (same generator, same path),
`propose` drops them as duplicates and re-asks, so round 2 delivers 4 GP
points and then 4 fresh random ones in that order; with nothing evaluated
and the 8 proposals of round 1 pending, the same seed yields 0 new points
(the ask is cached until a `tell`, and there is nothing to tell), and
`--seed 1` yields 8. Second, the surrogate round: at 16 observations the
proposals are all the GP's, they depend on the observed values (0 of 8
shared with a run on the same configurations and different values), and
the pending set is escaped through the nudge (8 new points with the 8
pending excluded).

**(f) The escalation verdict** (`escalation_verdict`,
`npe_tune_search.py:372-436`; `cmd_status`, `npe_tune_joint.py:538-562`).
On the finite, untagged, completed observations, with the window

$$w_{\rm esc} = \min\!\big(\max(5, \lfloor n_{\rm obs}/3 \rfloor),\, n_{\rm obs} - 1\big),
\qquad
\delta_{\rm esc} = \min_{\imath \le n_{\rm obs} - w_{\rm esc}} L_\imath \;-\; \min_{\imath \le n_{\rm obs}} L_\imath,
\tag{P6.8}$$

the verdict is ESCALATE if any of: $\delta_{\rm esc} > \tau_{\rm stop}$
("still descending"); the best configuration has a boundary axis
("widen the range, not just the budget"); $n_{\rm obs} < 12$ ("too few for
the surrogate to be informative"). $\tau_{\rm stop}$ is `--sigma-seed`
when given and `0.0` otherwise (`:558`), so without it the first condition
fires on any descent, however small (a 0.0081 nats/row improvement over
six observations escalates at 0.0 and stops at 0.02 `[RAN]` B6). The
window is 5 up to $n_{\rm obs} = 15$, then $\lfloor n_{\rm obs}/3 \rfloor$
(6 at 20, 8 at 24, 20 at 60) `[RAN]` B6. A boundary axis is a FREE numeric
axis with

$$\mathsf{c}_\varkappa \le \mathrm{lo}_\varkappa + \epsilon_{\rm edge}
\ \text{ or }\ \mathsf{c}_\varkappa \ge \mathrm{hi}_\varkappa - \epsilon_{\rm edge},
\qquad \epsilon_{\rm edge} = 10^{-6} \max(1, \lvert \mathrm{lo}_\varkappa \rvert, \lvert \mathrm{hi}_\varkappa \rvert),
\tag{P6.9}$$

for a real axis, and exact equality with an end for an integer axis
(`boundary_axes`, `joint_space.py:768-814`); categoricals, switches and
`batch_size_npe` have no edge (`_NO_BOUNDARY`, `:764-765`), pinned and
inactive axes are skipped. Since every real axis's range has
$\lvert \mathrm{hi}_\varkappa \rvert \le 20$, $\epsilon_{\rm edge}$ lies between
$10^{-6}$ and $2 \times 10^{-5}$ in absolute terms: 10 % of `weight_decay`'s lower end (0.041
decades), 1 % of `lr`'s (0.0043 decades), 0.1 % of `lambda_sep`'s, and
absolute at the zero-based `dropout`, `warmup_frac_rep` and the two
`log10_lambda_*` `[RAN]` B6. The verdict's third condition (the plan's
"surrogate predicts improvement above $\tau_{\rm stop}$ anywhere") is
deliberately not implemented from the surrogate's extrapolation
(`npe_tune_search.py:389-394`).

**(g) The finalists** (`cmd_finalists`, `npe_tune_joint.py:565-603`): the
$K_{\rm fin}$ smallest objectives, each with its config, its trial id
recomputed from the driver's `--seed`, and its gain `delta` looked up by
key in the same ledger (`_deltas_by_key`, `:299-316`); the file records
`rank_split` and `gate_split` as the two strings given. The warning "ranking
and gating on the SAME split ... Re-freeze a four-way split (S4.2)" prints
when and only when the two strings are equal (`:593-598`); at the defaults
(`sel`, `gate`) it does not print `[RAN]` B4b, and no four-way split
exists anywhere in the stack (F-am).

**(h) The controls** (`cmd_controls`, `:606-712`; `control_config_from`,
`assert_control_identity`, `joint_space.py:697-750`; `control_pvalue`,
`per_finalist_control_verdicts`, `npe_tune_gates.py:201-300`;
`holm_bonferroni`, `npe_diagnostics.py:408-422`). `--plan` copies each
finalist's configuration, sets `shuffle_pairs = 1` and `shuffle_seed =
1000 + s` for $s = 0, \dots, n_{\rm ctrl} - 1$, asserts that the control
differs from its candidate in the two shuffle fields and nothing else
(J20), and writes it as a spec whose id carries the tag `control`.
`--score` collects, per finalist, the gains of the records tagged
`control` whose ids the plan wrote, takes the finalist's own gain
(`delta_gate_split` if the entry has one -- nothing writes it -- else
`delta`, `:670-676`), and forms, for each finalist $\jmath$,

$$t^{\rm ctrl}_\jmath = \frac{\hat\Delta_\jmath - \bar\Delta^{\rm ctrl}_\jmath}
{s^{\rm ctrl}_\jmath \sqrt{1/n_{\rm seed} + 1/n_{\rm ctrl}}},
\qquad
\mathrm{p}_\jmath = \Pr\big[\text{Student's } t \text{ with } n_{\rm ctrl} - 1 \text{ degrees of freedom} > t^{\rm ctrl}_\jmath\big],
\tag{P6.10}$$

the handoff's eq. (S4.1) exactly (`npe_tune_gates.py:240-242`), returning
`(nan, nan)` when fewer than two controls are finite or
$s^{\rm ctrl}_\jmath = 0$ (`:235-239`). Then the Holm step-down over the
testable finalists (the untestable ones are left out of the family,
`:291-299`; below, $K_{\rm fin}$ stands for the family's size, which is
$K_{\rm fin}$ itself when every finalist is testable): with the p-values
sorted ascending, the one of ascending rank $\mathrm{rank}(\mathrm{p}_\jmath)$
is multiplied by $K_{\rm fin} - \mathrm{rank}(\mathrm{p}_\jmath) + 1$, a
running maximum enforces monotonicity, and

$$\tilde{\mathrm{p}}_\jmath = \min\Big\{1,\ \max \big(K_{\rm fin} - \mathrm{rank}(\mathrm{p}_{\jmath'}) + 1\big)\, \mathrm{p}_{\jmath'}\Big\}
\ \text{ over the finalists } \jmath' \text{ with } \mathrm{p}_{\jmath'} \le \mathrm{p}_\jmath,
\qquad \text{rejected} \iff \tilde{\mathrm{p}}_\jmath < \alpha_{\rm H} \ \text{and}\ \hat\Delta_\jmath > \delta_{\rm floor},
\tag{P6.11}$$

(`npe_diagnostics.py:414-422`; `npe_tune_gates.py:296-299`). The shipped configuration is "the BEST-RANKED survivor, not the
smallest p" (`:708-710`), and with no survivor the message is to not extend
$K_{\rm fin}$ after the fact (`:703-707`). At $K_{\rm fin} = 3$ and
$\alpha_{\rm H} = 0.05$ the step-down thresholds are 0.0167, 0.025, 0.05;
with $n_{\rm ctrl} = 5$ the one-sided critical value with four degrees of
freedom is 3.186 at 0.0167 and 2.132 at 0.05, so the smallest detectable
gap over the control mean is $3.49\, s^{\rm ctrl}_\jmath$ for a finalist
that must pass the first threshold and $2.34\, s^{\rm ctrl}_\jmath$ for
one that need only pass the last; the prediction-interval factor
$\sqrt{1 + 1/5} = 1.095$ is 2.45 times the confidence-interval factor
$\sqrt{1/5} = 0.447$ `[RAN]` B5. The procedure is Holm's step-down as the
peer-reviewed literature states it -- the p-values "ranked from largest to
smallest, and then each is compared to a significance level that is equal
to 0.05 divided by rank", a step-wise procedure developed "to maintain
greater statistical power while controlling the Type I error rate"
(`[PubMed full text]` Kirkham and Weaver 2015; the abstract of Ludbrook
1998, `[PubMed abstract only]`, commends Holm's step-down for the same
reasons, S6) -- and the reason
it is applied to all $K_{\rm fin}$ at once rather than down the ranked list
is the handoff's S4.1, quoted in the docstring (`npe_tune_gates.py:255-263`).

What the chain does *not* contain is the subject of S3.8: a gate split
(F-am), a permutation that reaches the runner (F-an), a control that trains
the finalist's recipe when a loss term is on (F-ao), a `baseline` or a
partial dependence (F-ap).

### 3.3 The knobs, one by one

*Establishes, for each knob of Table P6.1 and of P0 Table D's library rows,
where it lives, its type and every default it has, what it changes in the
chain of S3.2, what it interacts with, and how a wrong value shows. The
order is the order in which a campaign meets them.*

#### 3.3.1 `--campaign`

- **Lives:** `common`, default `S-A1` (`npe_tune_joint.py:750`); the table
  `CAMPAIGNS` (`joint_space.py:375-398`); the job's `CAMPAIGN=S-A1`
  (`joint_tune.pbs:60`); the launcher's first positional argument. A name
  outside the table exits with "unknown campaign" (`:817-822`). Status:
  configured; the choice is the plan's decision D4 (S8), still open.
- **Type:** one of four strings. **Changes:** $\mathcal{V}$ and the pinned
  values of eq. (P6.1), hence $n_{\rm free}$ and $n_{\rm dim}$ (Table in
  S3.1), the ledger directory, and the arms a campaign can evaluate (`S-A2`
  mixes `A1` and `A2`; `S-A25` cannot run its own `dsn_on = rep_on = 1`
  corner). It does not change the objective, the schedule or the anchors.
- **Interactions:** `S-A2` and `S-A5` nest `S-A1` at `dsn_on = 0` and
  `rep_on = 0` (J22), so an `S-A1` point is a legal `S-A2` point with the
  replicate pins and the switch added -- but the two campaigns keep separate
  ledgers and surrogates, and nothing pools them; the plan's "where the
  winners sit in the weight axes and the GP surrogate's partial dependence
  on `dsn_on` / `rep_on`" (S5.2) is a read-out of one campaign's surrogate
  that the driver does not produce (F-ap). With `strict_semihard` fixed at
  1, 9 of the 13 legal loss conditions are reachable in `S-A2` and `S-A25`
  (F-u, P2 S3.3).
- **Fails as:** the wrong campaign name against an existing results
  directory starts an empty ledger beside the real one; a campaign changed
  between `propose` and `evaluate` makes the element look for pending
  specs in the other directory and exit "holds no pending specs"
  (`joint_tune.pbs:120-123`).

#### 3.3.2 The shape anchors: `--p`, `--embedding-dim`, `--d-theta`, `--n-train`

- **Lives:** `common`, defaults 26 / 12 / 26 / None (`:753-756`); the job's
  `P=26`, `EMBEDDING_DIM=12`, `D_THETA=26` (`joint_tune.pbs:71-73`), no
  `N_TRAIN`; `default_joint_space(p, embedding_dim, d_theta, n_train)`
  (`joint_space.py:262-330`), recorded in the spec's `anchored_to`
  (`:325-329`) and nowhere in a trial record except through the resolved
  pins. Status: configured; they must equal the bank's $p$, $E$ and
  $d_\theta$ (`[KB]` usage v1.3 S7: "on a bench bank you must pass
  `--d-theta 10 --p 10`").
- **Type:** three integers and an optional one. **Changes:** `hidden_features`
  by the width rule, $[52, 256]$ at $\max(p, E) = 26$ and $[32, 256]$ at 10
  (P4 eq. (P4.12); the recorded string misdescribes it, F-ad);
  `n_posterior_draws` by the floor $4 d_\theta$, 104 and 40 (P3 S3.3.4; the
  dataclass default $(100, 400)$ below the floor is F-o); `batch_size_npe`
  by the trim $N_{\rm train} \ge 20 B_{\rm sim}$ with the smallest kept
  regardless -- all three sizes at $N_{\rm train} = 20\,731$, `[256]` at
  352, thresholds 5 120 / 10 240 / 20 480 `[RAN]` B7 (P5 S3.3.4; the rule
  counts a pass the loop never runs, F-ag; the jobs pass no `--n-train`).
  $E$ enters only through $\max(p, E)$; $p$ enters nowhere else in the
  driver (it is the bank's latent dimension, E0 convention 3).
- **Interactions:** the anchors decide the pins, the pins are in the trial
  id. `propose` under one set of anchors and `evaluate` under another
  re-pins `n_posterior_draws` for every `rep_on = 0` configuration (40 to
  104 on a bench campaign evaluated by the array at the job's defaults),
  changes the id, leaves the pending spec in place, and writes a record
  whose configuration the proposal did not make; replaying that record
  into an optimiser built at the proposal's anchors raises "Not all points
  are within the bounds of the space" `[RAN]` B4 (F-ar). The launcher
  cannot forward the anchors (S3.1), so a bench campaign needs them in the
  array's environment by another route (`--qsub-args`, or an edited job).
- **Fails as:** a floor below $4 d_\theta$ is refused by
  `default_joint_space` only when the upper bound is below it
  (`:310-316`); a wrong $d_\theta$ larger than the bank's makes every
  `rep_on = 1` trial draw more posterior samples than needed (harmless) and
  a smaller one lets `n_posterior_draws` fall under the floor the bank
  needs (the Cholesky failure P3 describes); a wrong $(p, E)$ moves the
  flow's width range (P4).

#### 3.3.3 `--split-hash`, `--contract-digest`

- **Lives:** `common`, both default `''` (`:757-758`); the job's
  `SPLIT_HASH`, `CONTRACT_DIGEST`, forwarded only when set; hashed into
  every trial id (S3.2 (c)); copied into the record (`:497-498`), the
  first then overwritten by the runner's own `split_hash` (`:517-521`).
  Status: configured.
- **Type:** two strings; the ledger module expects the frozen split's
  hash and the data contract's digest (`npe_tune_ledger.py:104-110`), but
  the driver validates neither and the runner computes its split from the
  bank and `--seed` at run time (P5 S3.2), so the string typed here is a
  label the id depends on, not a check. **Changes:** the trial id only.
- **Interactions:** the same configuration under two strings is two trials
  (`[RAN]` B4); a string typed at `propose` and omitted at `evaluate` (the
  job's variables unset) recomputes a different id, so the pending spec is
  orphaned as under F-ar. The runner's hash in the record is the one that
  describes the data; the id's string is whatever was typed.
- **Fails as:** silently: nothing compares the typed hash with the
  runner's.

#### 3.3.4 `--seed` ($s_{\rm seed}$)

- **Lives:** `common`, default 0 (`:759`); the job's `SEED=0`, one value for
  every array element (`joint_tune.pbs:67, 147`); the launcher forwards
  `SEED` only when exported. Status: configured.
- **Type:** integer. **Changes:** three things at once (F-as): the runner's
  `--seed` (the split, the generators, the initialisation, P5 S3.2); the
  `seeds` entry of the trial id; and the surrogate's `random_state`, which
  drives the initial design and the acquisition optimiser's random
  candidates. The plan's "stateless optimizer rebuilt from the ledger each
  round" is exact: the optimiser's state is the ledger plus this seed.
- **Interactions:** the same seed for `propose` across rounds replays the
  design's random draws (S3.2 (e)): harmless once observations exist (the
  duplicates are dropped and re-asked), fatal while everything is pending
  and nothing evaluated (0 proposals, `[RAN]` B3), which is the case the
  "change --seed" message was written for. A different `--seed` at
  `evaluate` than at `propose` recomputes the id and orphans the pending
  spec. One seed per campaign also means one training seed per
  configuration: the ledger holds no replicate of any trial, so
  $\sigma_{\rm seed}$ cannot be measured from it (F-ap).
- **Fails as:** `propose` writes fewer than $n_{\rm batch}$ specs and says
  so; an orphaned pending spec is visible as a `pending/` entry that
  survives its array element.

#### 3.3.5 `--n-points` ($n_{\rm batch}$) and `--n-initial-points` ($n_{\rm init}$)

- **Lives:** `propose`, defaults 8 and 12 (`:777-778`); the job knows
  neither (the array's length is the pending count). Status: configured;
  the plan's Stage 4 says "batches of 8" and names no design size; the
  standalone NPE tuner's defaults are the provenance (S3.5).
- **Type:** two positive integers. **Changes:** $n_{\rm batch}$ is the
  array's size per round and the number of constant-liar steps per ask;
  $n_{\rm init}$ is the positional count of random points before the first
  GP ask (S3.2 (e)): with 8 and 12 the surrogate is first consulted in
  round 2 with 8 observations and 4 lies, and after round 2 the ledger
  holds 12 random and 4 GP points `[RAN]` B3. Twelve points on 12 to 23
  free axes (14 to 29 transformed dimensions) is 1.0 to 0.52 points per
  axis; the standalone DSN search spends 100 or 150 of 300 on its design
  (S3.5). The literature describes the exploratory design as the phase that
  estimates the surrogate's hyper-parameters, raises how large it should be
  as a question, and in one setting finds the phase dispensable while
  leaving its transfer to Bayesian optimisation open (`[PubMed full text]`
  Sinsbeck et al. 2020, S6); nothing here chooses its size for this bank.
- **Interactions:** the escalation verdict's third condition keys on 12
  observations too (`n < 12`, `npe_tune_search.py:432`), a second copy of
  the same number; a smaller $n_{\rm init}$ hands the GP fewer points than
  the verdict trusts, a larger one delays the first GP ask by whole rounds.
  Lies are told at $L_{\rm inc}$, so within a batch every later point is
  pushed away from the earlier ones under a belief that they returned the
  best value so far; the belief is discarded with the copy.
- **Fails as:** a batch larger than the pending directory can hold is not
  a failure; a batch asked while the previous round's specs are pending
  excludes them and asks for more, which with no observations and the same
  seed yields nothing (S3.3.4).

#### 3.3.6 `--sigma-seed` ($\sigma_{\rm seed}$, as typed)

- **Lives:** `propose` and `status`, default None (`:779-781, 794`); in
  `propose` its square is the surrogate's noise level
  (`:395-396`); in `status` it is $\tau_{\rm stop}$ (`:558`). Nothing
  measures it: the standalone tuner's `baseline` does (S3.5), the plan's
  Stage 4 orders a `baseline` first, and the driver has no such
  subcommand (F-ap). Status: configured by hand.
- **Type:** a non-negative float in nats per row. **Changes:** (i) the
  white-kernel level of eq. (P6.4), in *standardised* units, so the
  assumed spread in nats per row is $\sigma_{\rm seed} s_{\rm obs}$,
  eq. (P6.5): on a ledger with $s_{\rm obs} = 0.23$ a typed 0.02 asserts
  0.0046 `[RAN]` B2 (F-aq); without the flag the level is fitted
  (`noise="gaussian"`, a `WhiteKernel` with free bounds), which on twelve
  points is the surrogate's guess. (ii) The escalation threshold: without
  the flag $\tau_{\rm stop} = 0$ and any descent escalates `[RAN]` B6.
- **Interactions:** the noise level and the length scales are fitted
  together; a fixed level too small lets the kernel interpolate seed luck
  (the docstring's own warning, `npe_tune_search.py:209-214`), too large
  flattens the posterior mean. The right object to type is the across-seed
  standard deviation of the *objective* at a fixed configuration, divided
  by the ledger's spread -- a quantity that changes as the ledger grows,
  which is one more reason the plan measured it once per bank rather than
  typing it per round `[reasoning]`; the batch-BO literature's own
  position is that "repeating experiments is the best option to separate
  signal from noise" (`[PubMed full text]` Binois et al. 2025), and the
  ledger holds no repeat of any configuration.
- **Fails as:** an escalation verdict that never says STOP (threshold 0);
  a surrogate that re-proposes near the lucky minimum.

#### 3.3.7 `--tag`

- **Lives:** `evaluate`, default `''` (`:790`); the job's `TAG`, forwarded
  only when set; hashed into the id; `load_observations` drops every tagged
  record (`:285-286`), `_deltas_by_key` too (`:309-310`); `controls
  --score` reads only records tagged `control` (`:659`). Status: configured.
- **Type:** string. **Changes:** whether a trial enters the surrogate and
  the ranking (untagged) or only the control test (`control`); any other
  string parks a trial in the ledger for nothing to read -- the standalone
  tuner's learning-curve points live this way (S3.5), and the joint driver
  has no reader for them.
- **Interactions:** the tag changes the id, so a control evaluated without
  `--tag control` is written as an untagged trial, *enters the surrogate*
  with its shuffled gain, and is never matched to its finalist; the plan
  message says `--pending-id <id> --tag control` and the control specs are
  not pending (F-ao).
- **Fails as:** a control record without the tag shows as an outlier
  objective in `status`; a mis-typed tag shows as "no control records" in
  `--score` (an empty `control_deltas`, p-value nan).

#### 3.3.8 `--top`, `--top-k` ($K_{\rm fin}$), `--rank-split`, `--gate-split`

- **Lives:** `status --top` 10 (`:793`); `finalists --top-k` 3,
  `--rank-split 'sel'`, `--gate-split 'gate'` (`:797-799`). Status:
  `--top` and `--top-k` configured; the two split flags configured as
  *labels* (F-am).
- **Type:** two integers; two strings. **Changes:** `--top` the rows
  printed; `--top-k` the family size of the control test and the number of
  control runs ($K_{\rm fin} n_{\rm ctrl}$); the two strings the fields
  `rank_split`, `gate_split` of `finalists.json` and `controls.json`, and
  whether the "SAME split" warning prints (`:593-598`). No subcommand
  selects a split by them: the objective is the runner's $L$ on the report
  split (F-al), and the gain used by the control test is `delta` from the
  same record (`:670-676`); `delta_gate_split` is never produced.
- **Interactions:** with $K_{\rm fin} = 3$ the Holm thresholds are
  0.0167 / 0.025 / 0.05 (`[RAN]` B5); fixing $K_{\rm fin}$ before the
  finalists are known is one of the handoff's reporting requirements, and
  extending it after a null result "reintroduces exactly the multiplicity
  Holm was applied to remove" (`:703-707`). The finalists are the argmin of
  a noisy objective over the whole ledger, so each finalist's own objective
  is biased low as an estimate of its expected value -- the selection
  bias of choosing the best of many configurations on the same held-out
  estimates, which the peer-reviewed literature treats as the multiple-
  comparisons problem of model selection and corrects with a separate
  estimation split or a bias-corrected resampling (`[PubMed full text]`
  Tsamardinos et al. 2018, S6); the handoff's S4.2 asks for exactly that
  separate split, and the labels record the intention without the data.
- **Fails as:** the warning is the only diagnostic and it is silent at the
  defaults; `report` prints the two labels as if they were splits
  (`:724-726`).

#### 3.3.9 `--n-control` ($n_{\rm ctrl}$), `--n-seeds` ($n_{\rm seed}$), `--alpha` ($\alpha_{\rm H}$), `--floor` ($\delta_{\rm floor}$), `--delta-min-provisional`

- **Lives:** `controls`, defaults 5 / 1 / 0.05 / 0.0 / nan (`:805-809`);
  `--n-control` acts in `--plan` (how many specs), the other four in
  `--score`. Status: configured; the handoff's values ($n_{\rm ctrl} = 5$,
  `n_s = 1` for a single-seed finalist, `alpha = 0.05`).
- **Type:** two integers, three floats. **Changes:** $n_{\rm ctrl}$ the
  degrees of freedom $n_{\rm ctrl} - 1$ and the cost $K_{\rm fin} n_{\rm
  ctrl}$ (15 at the defaults, "comparable to one finalist ensemble at
  $M_{\rm ens} = 5$" in the handoff's S5); $n_{\rm seed}$ the
  prediction-interval factor of eq. (P6.10) (1.095 at $1$ and $5$ against
  0.447 for the confidence-interval factor it replaces `[RAN]` B5), and
  it is typed, not read from the ledger, which holds one seed per
  configuration; $\alpha_{\rm H}$ the step-down's level;
  $\delta_{\rm floor}$ a hard minimum on $\hat\Delta_\jmath$ applied with
  the rejection (`npe_tune_gates.py:298-299`), 0 by default so that any
  positive gain passes it; `--delta-min-provisional` is carried into every
  verdict row for comparison and decides nothing (`:288`).
- **Interactions:** the permutation seeds $1000 + s$ exist only in the
  spec (F-an); with identical control runs $s^{\rm ctrl}_\jmath = 0$ and
  every p-value is nan, so the family is empty and "NO finalist rejected
  its own noise floor" is printed for the wrong reason; with round-off
  differences of order $10^{-9}$ the p-values are of order $10^{-35}$ and
  every finalist is shipped `[RAN]` B4b, B5. The suite's J33 writes control
  gains that differ by `0.002 * (control_seed % 5)` by hand
  (`smoke_test_joint_tune.py:453`), which the pipeline cannot produce.
- **Fails as:** `control_sd` 0 and `raw p` nan in `controls.json`; or
  p-values far below any sensible level with a control spread far below
  $\sigma_{\rm seed}$.

#### 3.3.10 The library knobs: `strict_semihard`, `n_posterior_draws_max`, `INACTIVE_CANONICAL`, `rel_tol`

- **Lives:** `default_joint_space(strict_semihard=1,
  n_posterior_draws_max=400)` (`joint_space.py:262-267`): the first goes
  into `spec.fixed` and from there into `--strict-semihard` and the
  projection's starting filter (S3.2 (b)); the second is the upper end of
  $S_{\rm mc}$'s range. `INACTIVE_CANONICAL` (`:161-185`) holds the pins
  (margin 0.2 after the `[CORRECTION]`, `:165-176`; two docstrings still
  say 0.3, F-f); `boundary_axes(rel_tol=1e-6)` (`:771`). The driver
  exposes none of them as flags (`_spec_from_args`, `:342-347`, passes the
  anchors only). Status: configured by code.
- **Changes:** `strict_semihard` decides which 9 of 13 loss conditions
  are reachable (F-u) and travels to the runner; `n_posterior_draws_max`
  the top of the $S_{\rm mc}$ axis (400: $\kappa_S = 1.035$ there against
  1.151 at the floor, P3 S3.3.4); the pins decide which coordinates a
  canonical configuration carries where its switch is off, hence the key
  and the id -- a pin outside its range would make every ledger built on it
  unreplayable (J27, `:152-156`); `rel_tol` the band of eq. (P6.9).
- **Interactions:** `strict_semihard = 1` is the `DSNLossConfig` default,
  not the DSN `TrainConfig`'s 0 (F-h, P2), so the joint space's "fixed by
  S5.1" value is the stack's own and not the standalone study's; the
  margin pin 0.2 is read under `joint` and `joint_sep` as $2 m_{\cos}$
  (P2) and is therefore an experimental value, not a placeholder.
- **Fails as:** a changed `DSNLossConfig` default without a changed pin is
  caught by J35; a changed `rel_tol` changes which best configurations
  escalate.

### 3.4 The fixed and library constants

*Establishes what a campaign commits to below any flag: scikit-optimize's
constructor defaults, the driver's own literals, and the file layout.*

| constant | value | where | what it decides | status |
|---|---|---|---|---|
| kernel | `ConstantKernel(1.0, (0.01, 1000)) * Matern(nu=2.5, length-scale bounds (0.01, 100), one scale per transformed dimension)` | `[REPO skopt v0.10.2]` `utils.py:364-392` | eq. (P6.4); a `HammingKernel` only if every dimension were categorical, which no campaign is | configured by the library |
| `normalize_y`, `alpha` | `True`, $10^{-10}$ (scikit-learn's default) | `utils.py:384-389`; `[RAN]` B2 | the standardisation of eq. (P6.3); a jitter on the diagonal | configured by the library |
| `noise` | `"gaussian"` (fitted) by default; a float with `"fixed"` bounds when `--sigma-seed` is given | `gpr.py:196-202`; `npe_tune_search.py:235` | whether $\sigma_{\rm noise}$ is fitted or asserted | configured (flag) / library |
| `n_restarts_optimizer` of the GP | 2 | `utils.py:388` | restarts of the marginal-likelihood fit | configured by the library |
| acquisition | `EI`, `xi = 0.01`, `kappa = 1.96` (unused) | `acquisition.py:40-41` | eq. (P6.6) | configured by the library |
| acquisition optimiser | `lbfgs` from 10 000 random candidates, 5 restarts | `optimizer.py:269-275, 298-299` | how the argmax of EI is found | configured by the library |
| initial point generator | `"random"`: one uniform draw per ask, nothing pre-generated | `utils.py:411-430`; `[RAN]` B2 | the design's replay behaviour (F-as) | configured by the library |
| `strategy` | `cl_min` | `npe_tune_search.py:275` | eq. (P6.7) | configured by code |
| `max_resample` | 20 | `npe_tune_search.py:254` | re-asks before `propose` gives up | configured by code |
| the nudge | `tell(duplicate, max(y))` | `npe_tune_search.py:290-293` | how a confident surrogate is moved off a pending point | configured by code |
| escalation window; minimum observations | $\max(5, \lfloor n_{\rm obs}/3 \rfloor)$ capped at $n_{\rm obs} - 1$; 12 | `npe_tune_search.py:412-413, 432` | eq. (P6.8) | configured by code |
| `rel_tol` | $10^{-6}$ | `joint_space.py:771` | eq. (P6.9) | configured by code |
| permutation seeds | $1000 + s$ | `npe_tune_joint.py:620` | the control spec's `shuffle_seed` (never read, F-an) | configured by code |
| the runner's permutation seed | $s_{\rm seed} + 777$ | `run_joint_arms.py:381` | the one permutation every control of a job gets | configured by code |
| `n_members` | 1 | `npe_tune_joint.py:334` | the trial id's input; no ensemble (F-ap) | configured by code |
| id length | 12 hex characters | `npe_tune_ledger.py:121` | file names | configured by code |
| file names | `pending`, `trials`, `control_specs`, `finalists.json`, `controls.json`; the runner's record `*_seed<s>.json` | `npe_tune_joint.py:66-69, 508-509, 627-628` | the layout of S3.1 | configured by code |
| `DEFAULT_RUNNER` | `../stage3/run_joint_arms.py` relative to the driver | `:73-74` | which script evaluates | configured by code |
| the job's element | 4 CPUs, 16 GB, 8 h, one pending spec per index | `joint_tune.pbs:3-4, 104-129` | the cost of one trial's wall time | configured (P7) |

The surrogate's constants are not the project's: no file of `hpc/joint/`
sets a kernel, a margin or a candidate count, and the plan's S5.1 names
only the mechanics ("stateless `skopt.Optimizer`", "constant-liar
batching"). A change of scikit-optimize's defaults between versions would
change the search with no diff in the repository; `HPC_PATHS.md` records
0.10.2 on the cluster and `environment.yml` pins `>=0.9,<0.11` (`[KB]`), and
the version of scikit-learn beneath it is not recorded.

### 3.5 What is inherited, and what differs: the standalone DSN search and the standalone NPE tuner (D-037)

*Establishes how the joint driver maps onto the two stacks it borrows from,
so that a reader of `TUNING_1`/`TUNING_2` or of `npe_tuning_usage.md` can
place it, and what the plan's Stage 4 ordered that neither stack supplies
here.*

| aspect | standalone DSN search (`dsn/search.py`) | standalone NPE tuner (`npe_tune.py`) | joint driver (`npe_tune_joint.py`) |
|---|---|---|---|
| optimiser | `skopt.gp_minimize` with EI, one process, resumed by `x0`/`y0` from its own ledger (`:880-975`) | `npe_tune_search.propose`: stateless `Optimizer`, `cl_min` batches | the same `propose` through `joint_space.adapter` (`joint_space.py:817-848`): nothing copied |
| design and budget | `n_calls_joint` 300 with `n_initial_points_joint` 100 (`config_l3c_joint_search.json:217-218`) or 150 (`config_mea_joint_full.davinci.json:226-227`); the legacy rule `min(10, max(1, n_calls // 2))` (`:806-830`) | `--n-initial-points`, `--n-points` (the joint defaults are its) | 12 and 8: 4 % of the DSN's budget on 1.0 to 0.52 points per axis `[RAN]` B7 |
| failed trial | scored `FAILED_OBJECTIVE = 1.0` and told to the GP, also when any one of a trial's seeds fails (`:206, 754-773`) | dropped | dropped (`load_observations`, `:287-291`): the surrogate never learns where training fails |
| space | `SearchConfig` ranges, condition legality by `condition_space` | `SpaceSpec`: 5 flow knobs incl. `num_bins` | 23 axes; `num_bins` fixed (F-a) |
| noise and seeds | every trial averaged over `n_seeds` seeds with disjoint seed blocks; the across-seed standard deviation logged as "the honest GP noise level", not optimised (`:70-80, 775-785`); the GP's level fitted (no `noise` argument, `:973-981`) | `baseline` measures $\sigma_{\rm seed}$ with `--n-seed-reps 2` and `--n-control 2`, then `--sigma-seed` | one seed per configuration; `--sigma-seed` typed; no `baseline` (F-ap) |
| objective | the negated selection metric (`selection_primary`: ARI, or silhouette), averaged over seeds (`:70, 628, 778`) | `npe_tune_score` on the frozen 3-way split, hashed | the runner's $L$ on the report split (F-al); the labels `sel` / `gate` (F-am) |
| escalation | none (fixed `n_calls`) | `status --tau-stop 0.01 --window --k-sigma 3.0`; `learning-curve` | `status`: $\tau_{\rm stop}$ from `--sigma-seed` or 0; no learning curve |
| finalists and gates | refit of the winner (`run_refit.pbs`) | `finalists`, `report` with G1-G3 | `finalists`, `controls`, `report`: G1 per finalist; no G2/G3, no $M_{\rm ens}$ (F-ap) |
| control | none in the search | one shuffled baseline per bank (`shuffle_seed` through `np.random.default_rng`, `npe_tune.py:292-340`) | $n_{\rm ctrl}$ per finalist, whose seed does not travel (F-an) |
| ledger | `trials.jsonl` plus `search_state.json`, resumed as `x0`/`y0` (`:824-825, 960-961`) | `npe_tune_ledger.TrialRecord`, one JSON per trial | the same `TrialRecord` fields, one JSON per trial, written atomically by `write_record` (`:319-327`) |

The plan's Stage 4 reads: "Deliverables `joint_space.py`,
`npe_tune_joint.py` (subcommands mirroring `npe_tune.py`, plus
`partial-dependence`) ... Order: `baseline` (measures $\delta_{\min}$,
$\sigma_{\rm seed}$), then S-A1 / S-A2 / S-A5 in batches of 8, `status`
escalation, `finalists --top-k 3` at $M_{\rm ens} = 5$, gates,
`partial-dependence`, `report` once" (`:1365-1370`). Of these, `baseline`,
`partial-dependence`, the ensemble and the gates beyond G1 are not in the
driver (F-ap): the eight subcommands are S3.1's. The partial dependence
the plan wants -- "the GP surrogate's partial dependence on `dsn_on` /
`rep_on`, which is the cleanest statement of 'does this term help at the
optimum'" (S5.2) -- is, in the interpretable-ML literature, the average of
the fitted model's prediction over the other inputs with the one of
interest held at each of its values, estimated by averaging over the
observed inputs (`[PubMed full text]` Langbein et al. 2025, S6: its
partial-dependence section read); scikit-optimize ships
`skopt.plots.partial_dependence` (and `plot_objective` on top of it),
which evaluates this on the fitted surrogate `[RAN]` (importable in the
sandbox), and the driver calls nothing of the kind. For `S-A0` -- the DSN's own campaign
followed by the standalone NPE tuner on frozen $z$ (plan S5.2) -- the two
left-hand columns are the whole story, and this document stops at the
boundary D-037 draws: `TUNING_1`/`TUNING_2`/`CONFIG_REFERENCE.md` and
`hpc/docs/npe_tuning_usage.md` are their references.

For the Giulia project the user has decided that the joint stack is
trained "on the arm trained only on simulated data", which is arm `A1`,
and that "optimization" is this driver's `S-A1` campaign on the Giulia
simulated bank, with every configuration choice deferred ("afterwards",
D-039 `[STATED]`; D-040 to D-046 for the cohort). Nothing of that
campaign's shapes is fixed yet; the anchors of S3.3.2 will be the Giulia
bank's, not 26 / 12 / 26.

### 3.6 Interactions with the parameter blocks

*Establishes what the driver does to each block's axes beyond proposing
values for them: which axes it can flag on a boundary, which it pins, which
it cannot reach, and how a block's own facts bend the search.*

- **Encoder (P1).** `depth_exponent`, `width_multiplier`, `embedding_size`,
  `dropout` are boundary-checked; `block_family` and `head_fusion` are not.
  The upper end `depth_exponent = 6` builds 49 M to 215 M-parameter
  encoders with no budget guard in the driver (F-s): a best configuration
  there escalates with "widen the range", the one direction `TUNING_1`
  forbids without a budget gate. `head_pool_ops` is fixed at code 1 in the
  spec and built at code 0 (F-b): the ledger's `anchored_to`/`fixed` block
  records a head the runs do not have.
- **DSN loss (P2).** The six axes are free only when `dsn_on` is free
  (`S-A2`, `S-A25`); clause (c) of eq. (P6.2) pins `margin` under `joint`
  and `joint_sep` and pins `angular_alpha_deg`, `lambda_sep` under
  `triplet`, so the surrogate sees, for every configuration, the active
  hyper-parameters at their proposed values and the inactive ones at the
  pins -- two proposals that differ only in an inactive coordinate are one
  configuration (J23) and a duplicate for `propose`. With the filter fixed
  at 1, 9 of 13 conditions are reachable (F-u). `sep_warmup_frac` does not
  travel (F-w). On the `shuffled` arm the term is off whatever the
  configuration says (F-ao).
- **Replicate (P3).** The three axes are free only in `S-A5` and `S-A25`;
  `n_posterior_draws` is pinned to the floor $4 d_\theta$ when `rep_on =
  0`, which is why the pin depends on `--d-theta` and the anchors enter
  the id (F-ar). The dataclass default $(100, 400)$ below the floor is
  reachable only by a caller that bypasses `default_joint_space` (F-o);
  the driver never does. `warmup_frac_rep` travels (`--warmup-frac-rep`),
  at its pin 0.0 when the term is off, which is not the runner's own
  default 0.3 (P3 F-p): a Stage 4 `A1` trial and a Stage 3 `A1` arm differ
  in this inert value and nowhere else that this block reaches.
- **Flow (P4).** `hidden_features` (log-uniform integer) and
  `num_transforms` are boundary-checked; the lower end 52 at $(26, 12)$ is
  the width rule's, not the recorded string's (F-ad); `num_bins` does not
  travel and the trained flow has 8 bins against the recorded 10 (F-a); the
  runner's `--hidden-features 48` and `--num-transforms 3` lie below the
  ranges (F-r), so the Stage 3 default flow is not a point the search can
  propose.
- **Optimiser and schedule (P5).** `lr`, `one_minus_beta1`,
  `weight_decay` are boundary-checked in log coordinates with the band of
  eq. (P6.9) -- 10 % of `weight_decay`'s lower end, where the decay is
  inert at the runner's schedule in any case (F-ai), so a best
  configuration pinned at $10^{-5}$ would escalate the campaign to widen
  an axis that does nothing at 250 steps `[reasoning on F-ai]`;
  `batch_size_npe` is categorical and never on an edge, trimmed by
  `--n-train` under a pass-based rule the loop does not run (F-ag). The
  schedule (`EPOCHS`, `STEPS_PER_EPOCH`) and the patience 99 (F-d) are the
  job's and the runner's, outside the id; the control's docstring expects
  early stopping to fire on shuffled data and nothing can fire it under
  99 (F-d), so a control trains the full 250 steps on a permuted pairing
  and its gain is the gain of a run that never stopped. The objective is
  $L$ on the report split (F-al): the selection split decides only which
  epoch's weights are scored.

### 3.7 Failure modes and the diagnostics that reveal them

| failure | what happens | what reveals it |
|---|---|---|
| anchors differ between `propose` and `evaluate` (F-ar) | the pending spec is re-canonicalised with another floor, gets a new id, is evaluated and recorded under it; the old spec stays pending and is excluded forever; a `propose` at the job's anchors raises when it replays a record proposed at the bench anchors | a `pending/` entry older than the array; `ValueError: Not all points are within the bounds of the space` from `propose` |
| the same `--seed` while everything is pending and nothing evaluated | 0 proposals | "asked for 8, wrote 0 ... change --seed" |
| `--sigma-seed` omitted | the noise level is fitted on few points; $\tau_{\rm stop} = 0$ and the verdict escalates on any descent | `[escalation] ESCALATE ... more than tau_stop=0.0000` |
| `--sigma-seed` typed in nats/row (F-aq) | the surrogate assumes a spread $s_{\rm obs}$ times smaller | `noise_` of the fitted model equals the typed square; nothing in the driver prints it |
| a control evaluated with `--pending-id` (F-ao) | `FileNotFoundError` on `pending/<id>.json` | the element's traceback |
| a control evaluated without `--tag control` | a new id; the shuffled run enters the surrogate as a trial | an outlier `nll` in `status`; `controls --score` finds no control for the finalist |
| the controls of one finalist run as planned (F-an) | identical runs; `control_sd` 0; p-values nan; no survivor | `controls.json`: `control_sd` 0.0, `pvalue` nan |
| ... with round-off differences | p-values of order $10^{-35}$; every finalist survives | `control_sd` of order $10^{-9}$ beside a `delta` of order $10^{-1}$ |
| a finalist with `dsn_on = 1` or `rep_on = 1` (F-ao) | its controls train `A1` with both weights zero; the floor is the NPE-only recipe's | `arm` `shuffled` in the control record with `--lambda-dsn` non-zero in its `argv` |
| the "SAME split" warning (F-am) | prints only when the two labels coincide; at the defaults rank and gate use one split silently | nothing, at the defaults |
| a trial fails in the runner | `status: failed`, dropped from the surrogate (the DSN scores +1.0 instead) | `[evaluate] FAILED`; the record's `error` |
| `propose` run mid-array | the index map is renumbered; elements evaluate other trials than planned | the job header's own warning (`joint_tune.pbs:36-38`) |
| a best configuration on `depth_exponent = 6` (F-s) | ESCALATE "widen the range" toward encoders of up to 215 M parameters | `status`'s boundary line |
| the schedule changed between campaigns | records of one configuration at two schedules share an id | nothing: the second overwrites the first |

### 3.8 Findings this document owns

Carried from the index (S6) with what this document adds; the new ones
follow. Each is a property of the code at `834eb41`, none a result.

- **F-a, F-b (with P4, P1).** The driver's side of the two D-038 parity
  findings is `build_argv`'s omission of `--num-bins` and of any
  `head_pool_ops` flag (`npe_tune_joint.py:197-211`): the ledger's `fixed`
  block says 10 and code 1, the runs train 8 bins and code 0. Stage 8's
  patch (D-038) passes `spec.fixed` through `build_argv`; the design of the
  legality projection shows the pattern -- `--strict-semihard` already
  travels from the same table (`:206-211`).
- **F-d (with P5).** `control_config_from`'s docstring expects early
  stopping "to fire almost immediately on shuffled data" and asks for
  epochs-to-stop per control (`joint_space.py:712-717`); at the runner's
  patience 99 nothing fires and no such field exists, so every control is
  a full-length run.
- **F-f.** The margin docstrings (`:536-538`, `:780-782`) say 0.3; the pin
  is 0.2. Confirmed inert: the pin is what travels (`[RAN]` B4's argv).
- **F-j (with P4).** The resolved ranges and free counts, now with the
  transformed dimensions 14 / 25 / 18 / 29 `[RAN]` B1.
- **F-o (with P3).** The dataclass default $(100, 400)$ is unreachable
  through the driver; stated in S3.6.
- **F-r (with P0, P4, P5).** The runner's defaults are not a point of the
  space; the driver cannot propose the Stage 3 default configuration, so
  "does the search beat the default" has no row in the ledger unless the
  default is evaluated by hand with `--config-json` -- and its
  `batch_size_npe = 128` would still fall outside the categorical.
- **F-s (with P1).** No budget guard in the driver; the escalation rule
  points the other way (S3.6).
- **F-u (with P2).** 9 of 13 conditions reachable; the filter's fixed value
  is `default_joint_space`'s argument, not the plan's.
- **F-w (with P2).** `--sep-warmup-frac` not passed; agreement by two
  defaults.
- **F-ad (with P4).** The recorded width rule misdescribes the code; the
  resolved bounds beside it are right; the driver's `space` prints the
  string.
- **F-al (with P5, E8).** The ledger's `nll` is $L$ on the report split
  (`:524-525`); every ranking and every gain the driver computes is on
  that split.
- **F-am (new) -- the split labels.** `--rank-split` and `--gate-split`
  are strings written into `finalists.json` and `controls.json`
  (`npe_tune_joint.py:584-586, 696-697`); no code path reads a split by
  them, the runner has three parts and scores `L` on the report split
  (F-al), `delta_gate_split` is never produced (`:670-676`), and the
  warning that names the danger ("the finalists are the argmax of that
  split over the whole search, so a null calibrated for a single
  configuration understates the false-pass rate and Holm does not repair
  it", `:593-598`) prints only when the two strings are equal -- which
  they are not at the defaults `[RAN]` B4b. The condition the warning
  describes is the one the code is in: finalists ranked on `nll` and
  gated on `delta = L0 - L` from the same record on the same split. The
  handoff's S4.2 asked to "verify before implementing" which split the
  objective uses; this is the answer. The repair is the handoff's own: a
  gate split (a fourth part of the grouped split, or the selection split
  re-purposed once early stopping is made operative) on which both
  $\hat\Delta_\jmath$ and the control gains are scored; a decision for the
  log.
- **F-an (new) -- the permutation seed never travels.** `controls --plan`
  writes `shuffle_seed = 1000 + s` into each control's configuration
  (`:620`), `assert_control_identity` checks it is the only difference
  (`joint_space.py:725-750`), and `build_argv` has no flag for it
  (`AXIS_TO_FLAG`, `npe_tune_joint.py:86-106`); the runner permutes with
  `torch.Generator().manual_seed(args.seed + 777)`
  (`run_joint_arms.py:380-382`) and the job passes one `SEED` to every
  element. So the $n_{\rm ctrl}$ controls of one finalist have distinct
  ids and one argv (`[RAN]` B4: 5 ids, 1 argv) and are the same run up to
  floating-point non-determinism: $s^{\rm ctrl}_\jmath = 0$, every
  p-value nan, no survivor, "That is the result" printed for the wrong
  reason; or, with round-off jitter, p-values of order $10^{-35}$ and
  every finalist shipped `[RAN]` B4b, B5. `apply_control_shuffle`
  (`joint_space.py:658-694`), which draws the permutation from
  `shuffle_seed` with `np.random.default_rng` and rejects the identity, is
  called by the smoke tests only; the suite's J33 varies the control gains
  by hand (`smoke_test_joint_tune.py:453`). The handoff's "one permutation
  per control seed" (S4.3) is not what runs. Repair (D-038 stream): a
  `--shuffle-seed` flag on the runner, passed by `build_argv` from
  `shuffle_seed`, and a per-control `--seed` as an alternative that also
  changes the initialisation.
- **F-ao (new) -- a control of a finalist with a loss term on is not the
  finalist's recipe, and the control specs are not where the plan message
  points.** `arm_for_config` names a control `shuffled` (`:239-240`), and
  the runner's `arm_config("shuffled")` sets `lambda_dsn = 0`, `lambda_rep
  = 0`, `dsn_domain = None` whatever `--lambda-dsn` and `--lambda-rep` the
  argv carries (`run_joint_arms.py:178-199`; `TrainConfig` takes the arm's
  values, `:519-522`; `[RAN]` B4: the argv passes `--lambda-dsn 0.1`,
  the arm config reads 0.0). The identity the plan step asserts holds at
  the configuration level and breaks at the arm level: an `A2` or `A5`
  finalist's floor is measured by an `A1` run on permuted pairs. In
  `S-A1` every finalist is `A1` and the arm is right. Separately, the plan
  message instructs `evaluate --pending-id <id> --tag control` (`:641-642`)
  while the specs are written to `control_specs/`, not `pending/`
  (`:627-628`), so that command fails with `FileNotFoundError` and the
  working form is `--config-json control_specs/<id>.json --tag control`
  `[RAN]` B4b. Repair: a `shuffled` variant per arm (or a `--shuffle-pairs`
  flag orthogonal to `--arm`), and the message corrected.
- **F-ap (new) -- the plan's Stage 4 order is not the driver's.**
  `baseline` (which measures $\delta_{\min}$ and $\sigma_{\rm seed}$),
  `partial-dependence`, the $M_{\rm ens} = 5$ ensemble at the finalists
  and the gates G2/G3 (plan Stage 4, `:1365-1370`; S5.2's read-out) have
  no subcommand: $\sigma_{\rm seed}$ is typed (`--sigma-seed`), defaults
  to None and makes $\tau_{\rm stop} = 0$; $\delta_{\min}$ is typed as
  `--delta-min-provisional` and decides nothing; `n_members = 1` is
  hard-coded into the id (`:334`); no record holds a second seed of any
  configuration. The standalone tuner has the first two (S3.5). The
  driver's docstring says "subcommands mirroring `npe_tune.py`" of the
  plan only by listing its own eight.
- **F-aq (new) -- `--sigma-seed` is asserted in the wrong units.** The
  noise variance enters as a fixed `WhiteKernel` level on targets that
  scikit-learn has standardised by their mean and standard deviation
  (`[REPO skopt v0.10.2]` `gpr.py:196-202`; `[REPO sklearn]` `_gpr.py`
  `fit`, 1.6.1 and 1.9.1 `[RAN]` B2), so the spread the surrogate assumes
  in nats per row is $\sigma_{\rm seed} s_{\rm obs}$, eq. (P6.5) -- 0.0046
  for a typed 0.02 on a ledger of spread 0.23 `[RAN]` B2 -- and it changes
  as the ledger's spread changes. The docstring ("the GP's assumed
  observation noise VARIANCE. Pass the square of the measured run-to-run
  spread", `npe_tune_search.py:209-214`) describes raw units. The same
  applies to the standalone tuner, which shares the function (out of
  scope, D-037; noted). Repair: divide the typed value by the ledger's
  standard deviation at each `propose`, or let the level be fitted and
  report it. The cluster's scikit-learn version is not recorded; the
  standardisation has been scikit-learn's behaviour under `normalize_y`
  since 0.23 (`gpr.py:233-241` branches on it), and scikit-optimize 0.10.2
  requires it.
- **F-ar (new) -- the anchors are an input of the trial id and the launcher
  cannot forward them.** `evaluate` re-canonicalises the pending
  configuration under its own anchors (`_load_config`, `:465-469`), so a
  campaign proposed at $(10, 10, 10)$ and evaluated by the array at the
  job's defaults $(26, 12, 26)$ re-pins `n_posterior_draws` from 40 to 104
  for every `rep_on = 0` trial, computes a new id, records the trial under
  it, leaves the pending spec in place (removal is by the recomputed id,
  `:530-533`), and a `propose` at the job's anchors raises when it replays
  a record proposed at the bench anchors (`[RAN]` B4: ids `1fd6f1707d57`
  against `2e2b1f9e2001`; `ValueError: Not all points are within the
  bounds`, `hidden_features` 40 below 52).
  `launch_joint_tune.sh` forwards neither `P`, `EMBEDDING_DIM`, `D_THETA`
  nor `TAG` (`:135-143`). The usage note "on a bench bank you must pass
  `--d-theta 10 --p 10`" (`[KB]` usage v1.3 S7) is right for the driver's
  commands and silent on the array. Repair: forward the three variables in
  the launcher and store the anchors in the pending spec so that `evaluate`
  can refuse a mismatch.
- **F-as (new) -- one seed, three roles, and a design that replays.** The
  driver's `--seed` is the runner's seed, the id's `seeds` list and the
  surrogate's `random_state` (S3.3.4). With the default `"random"`
  generator the design is drawn one point per ask from the optimiser's own
  generator, so the same seed across rounds re-draws the same points
  (`[RAN]` B3: round 2's four remaining random draws are round 1's first
  four); `propose` recovers by dropping and re-asking once observations
  exist, and cannot while everything is pending (0 proposals at the same
  seed, 8 at `--seed 1`). The plan's "stateless optimizer rebuilt from the
  ledger each round" is exact and is the cause. A seed per round (the
  message's advice) changes the runner's seed of that round's trials too,
  which is harmless for the search and worth knowing for anyone comparing
  trials across rounds.

Three statements of P0 Table D are corrected by this document, in P0 v1.2:
`--seed` is not only "the seed handed to the runner" (three roles, F-as);
the `--rank-split` / `--gate-split` cell's parenthesis describes the
intention and not the code (F-am); the `--n-train` cell's "at least 20
steps per epoch" is the trim rule's pass-based reading (P5 F-ag).

## 4. Summary of results

| statement | where |
|---|---|
| The search space is 23 axes in 5 blocks; a campaign is a free/pinned partition, 12 / 19 / 16 / 23 free axes and 14 / 25 / 18 / 29 surrogate dimensions for `S-A1` / `S-A2` / `S-A5` / `S-A25`; `S-A25`'s own corner has no arm `[RAN]` B1, B4 | S3.1, eq. (P6.1) |
| Canonicalisation pins the inactive axes to the base config's values (margin 0.2, 18 deg, 0.1; $4 d_\theta$ draws) and projects the loss condition from the fixed filter 1; it is idempotent and key-preserving; the key is the 23 axes, the id is the whole dict plus split hash, digest, seed, tag `[RAN]` B1b, B4 | S3.2 (b), (c), eq. (P6.2) |
| The surrogate is scikit-optimize's GP: Matern 5/2 on normalised coordinates, standardised targets, EI against the smallest observed value with margin 0.01, L-BFGS over 10 000 candidates, `cl_min` lies; `--sigma-seed`$^2$ is a fixed white-kernel level in standardised units, an assumed spread of $\sigma_{\rm seed} s_{\rm obs}$ nats/row `[RAN]` B2 | S3.2 (d), (e), eq. (P6.3)-(P6.7) |
| The design is positional and drawn per ask: at 12 / 8 the GP is first consulted in round 2 after 8 observations with 4 lies; the same seed replays the design, which `propose` survives only once observations exist `[RAN]` B3 | S3.2 (e), F-as |
| Escalation: window $\max(5, \lfloor n_{\rm obs}/3 \rfloor)$ capped at $n_{\rm obs} - 1$; threshold `--sigma-seed` or 0; boundary band $10^{-6} \max(1, \lvert \mathrm{lo}_\varkappa \rvert, \lvert \mathrm{hi}_\varkappa \rvert)$, 10 % of `weight_decay`'s lower end; fewer than 12 observations always escalates `[RAN]` B6 | S3.2 (f), eq. (P6.8)-(P6.9) |
| The control test is the handoff's one-sided $t$ with the prediction-interval factor and Holm across all finalists; thresholds 0.0167 / 0.025 / 0.05 at $K_{\rm fin} = 3$; minimal detectable gap $3.49\, s^{\rm ctrl}_\jmath$ (first threshold) or $2.34\, s^{\rm ctrl}_\jmath$ at $n_{\rm ctrl} = 5$; identical controls give nan and no survivor, $10^{-9}$ jitter gives $10^{-35}$ and three survivors `[RAN]` B4b, B5 | S3.2 (h), eq. (P6.10)-(P6.11) |
| Rank and gate are on one split; the labels select nothing; the warning is silent at the defaults | F-am |
| The permutation seed does not reach the runner; the $n_{\rm ctrl}$ controls of a finalist are one run | F-an |
| A control of an `A2`/`A5` finalist trains `A1`; control specs are not pending | F-ao |
| No `baseline`, `partial-dependence`, ensemble or G2/G3 in the driver; $\sigma_{\rm seed}$, $\delta_{\min}$ typed; $\tau_{\rm stop}$ defaults to 0 | F-ap |
| The anchors enter the id through the pins; the launcher forwards none of them; a mismatch orphans the pending spec and breaks replay | F-ar |
| The DSN search spends 100 or 150 of 300 evaluations on its design and scores failures +1.0; the joint driver 12 of its budget and drops them; the standalone tuner measures what the joint driver asks to be typed | S3.5 |

## 5. Open points, caveats, assumptions

- **No campaign has run.** Every number is the driver's behaviour on a
  synthetic ledger (B4b) or a closed form; the kernel's fitted values on
  twelve synthetic points are illustrative of the mechanism, not of any
  bank.
- **The cluster's scikit-learn version is not recorded** (`HPC_PATHS.md`
  sec. 7 lists scikit-optimize 0.10.2 and not scikit-learn). F-aq's
  mechanism is scikit-learn's `normalize_y` as implemented from 0.23 on;
  if `sbi_env` carried an older scikit-learn, scikit-optimize 0.10.2 would
  not import against it `[reasoning on gpr.py:233-241]`. To be read off
  the cluster.
- **The runner was not invoked.** `arm_config` was extracted and executed
  alone; that `TrainConfig` takes the arm's weights and not the flags is
  read from `run_joint_arms.py:519-522`, not run.
- **The GP posterior formulas** are cited from memory (Rasmussen and
  Williams); nothing numeric depends on them beyond scikit-learn's own
  outputs, which were run.
- **Decisions this document needs and does not make:** the gate split
  (F-am), the control's seed path and arm (F-an, F-ao), whether `baseline`
  and `partial-dependence` are added or the plan's Stage 4 is rewritten to
  what exists (F-ap), the units of `--sigma-seed` (F-aq), the launcher's
  variables (F-ar), the plan's D4 (four campaigns at equal budget, or
  `S-A2`/`S-A5` with partial-dependence read-outs), the Giulia campaign's
  configuration (D-039, "afterwards"). Each is listed for the log's Open
  calls; none is appended as a decision here.
- **The design-size question** -- how many random points before the GP --
  is open in the literature consulted (S6) and open here; 12 is inherited
  from the standalone tuner without a reason recorded for this space.
- **`JOINT_DSN_NPE_STAGE4_OPS_v1.md`**, named by
  `HANDOFF_hpc_implementation_v1_1.md` as the Stage 4 operating sheet, is
  not in the repository at `834eb41` nor in the knowledge base (`find`
  over the clone, 2026-10-02); nothing is cited from it.

## 6. References / further reading

**Project knowledge base `[KB]`.** `JOINT_DSN_NPE_USAGE_v1.md` v1.3 S7 (the
Stage 4 command sequence and the anchors note), S9, S10; `HPC_PATHS.md`
sec. 7 (scikit-optimize 0.10.2 in `sbi_env`); `claude/joint_docs/00_INDEX.md`
S6 and S8; `claude/SBI_decisions_and_ideas_log.md` v1.21 (D-037, D-038,
D-039, D-052, D-054); `hpc/joint/HANDOFF_DELTA_MIN_PER_CONFIG_v1.md` (the
two tiers, S4.1-S4.3, S5, the reporting requirements); P0 Table D; P4
S3.3.1; P5 S3.3.4, S3.8; P2 S3.3; P3 S3.3.4. `[KB-PDF]`: Deistler M et al.
*Simulation-based inference: a practical guide* (the project PDF), p.32
("Common approaches include random search or Bayesian optimization over
architecture choices ... and training hyperparameters"; "For NPE,
minimizing validation loss ... provides a principled objective, as this
loss upper bounds the Kullback-Leibler divergence between the true and
approximate posterior"). Radev ST et al. *BayesFlow* (IEEE TNNLS 2022, the
project PDF), p.13 ("Future research should investigate the impact of
modern hyperparameter optimization methods, such as Bayesian
optimization"). Cranmer K, Brehmer J, Louppe G. *The frontier of
simulation-based inference* (PNAS 2020, the project PDF), p.4 and its
references 27-28, 31 (Gaussian-process surrogates and Bayesian optimisation
as inference engines for likelihood-free inference -- the other sense of
convention 5); Goncalves PJ et al. (eLife 2020, the project PDF), p.3, p.16
(grid search as the alternative SNPE replaces), p.30 (the same references);
Hikida Y et al. *Multilevel neural simulation-based inference* (arXiv, the
project PDF; **PREPRINT, not peer-reviewed** [corrected 2026-10-03: the PDF's p.1 carries the line "39th Conference on Neural Information Processing Systems (NeurIPS 2025)", so the project PDF is the NeurIPS 2025 paper, peer-reviewed as a conference paper; P7 S6]), p.3 (GP surrogates among the
methods it positions against).

**Repository `[REPO]`** at `834eb41`, read 2026-10-02: the files and lines of
the changelog row. **scikit-optimize 0.10.2** `[REPO skopt v0.10.2]`, read
from GitHub at tag `v0.10.2` and run installed: `optimizer.py` (the
constructor's normalisation of dimensions and the acquisition optimiser's
resolution, `copy`, `ask` with `n_points` and the `cl_min` lie, `_ask`'s
random phase, `_tell`'s counter and refit, `y_opt`), `utils.py`
(`cook_estimator`, `cook_initial_point_generator`),
`learning/gaussian_process/gpr.py` (`fit`: the white kernel, its zeroing,
`noise_`, the scikit-learn version branches), `acquisition.py`
(`gaussian_ei`, the defaults), `space/space.py` (`rvs`, the transformers).
**scikit-learn** `[REPO sklearn]`: `gaussian_process/_gpr.py`,
`GaussianProcessRegressor.fit` at 1.6.1 (`:269-274`, GitHub) and 1.9.1 (the
installed file; the same four statements).

**Peer-reviewed literature.** According to PubMed, read in full text from
PubMed Central: Kirkham EM, Weaver EM. *A review of multiple hypothesis
testing in otolaryngology literature.* Laryngoscope 2015; PMID 25111574,
PMC5935793, [DOI](https://doi.org/10.1002/lary.24857) -- the family-wise
error rate for independent tests at a common level (its formula is stated
there), the Bonferroni-Holm method stated as the p-values "ranked from
largest to smallest, and then each is compared to a significance level
that is equal to 0.05 divided by rank", and step-wise procedures as the
class "developed in order to maintain greater statistical power while
controlling the Type I error rate"; used in S3.2 (h) and the glossary; no
number of its survey is used. Tsamardinos I, Greasidou E,
Borboudakis G. *Bootstrapping the out-of-sample predictions for efficient
and accurate cross-validation.* Mach Learn 2018; PMID 30393425, PMC6191021,
[DOI](https://doi.org/10.1007/s10994-018-5714-4) -- "the cross-validated
performance of the best configuration is optimistically biased", the
multiple-comparisons framing of selecting the best of many configurations
on the same tuning estimates, the bias's dependence on the number of
configurations, their correlation, the sample size and the gap to the
true best, and the train / tuning / estimation partition and nested
cross-validation as the remedies; used in S3.3.8 and the glossary for
F-am; no number used. Sinsbeck M, Hoege M, Nowak W. *Exploratory-phase-free
estimation of GP hyperparameters in sequential design methods -- at the
example of Bayesian inverse problems.* Front Artif Intell 2020; 3:52; PMID
33733169, PMC7861299, [DOI](https://doi.org/10.3389/frai.2020.00052) -- the
two-phase structure of sequential design (a space-filling exploratory
design to estimate the GP's hyper-parameters, then points added one by
one), the question of how many evaluations the first phase should get as
one the literature raises, the authors' finding, for their Bayesian
inverse problems, that the hyper-parameters can be estimated without an
exploratory phase, and their caution that transfer to Bayesian
optimisation "remains an open question"; used in S3.3.5 for the
design-size point only; no number used. Binois M, Collier N, Ozik J. *A
portfolio approach to massively parallel Bayesian optimization.* J Artif
Intell Res 2025; 82:137-167; PMID 41000331, PMC12459664,
[DOI](https://doi.org/10.1613/jair.1.16868) -- batch EI
heuristics that "select batch points iteratively, replacing unknown values
at selected points by pseudo-values" ("hallucination"), the incumbent as
"the best value observed so far in the deterministic case" with the noisy
alternatives "the best mean estimation over sampled designs or the entire
space", "noise variance, if present, is seldom known and must be estimated.
With replication, stochastic kriging relies on empirical noise variance
estimates", and "repeating experiments is the best option to separate
signal from noise"; used in S3.2 (e) and S3.3.6; no number used.
Langbein SH, Krzyzinski M, Spytek M, Baniecki H, Biecek P, Wright MN.
*Interpretable machine learning for survival analysis.* Biom J 2025; PMID
41168965, PMC12576049, [DOI](https://doi.org/10.1002/bimj.70089) -- its
section 2.2.1 on partial dependence (the average prediction with the
feature set of interest fixed and the remaining features varying over
their marginal distribution, estimated by averaging over the observed
training values), read; the rest of the review not read; used in S3.5;
no number used.

Retrieved, **abstract only, therefore not used for any claim beyond the
flagged one**: Ludbrook J. *Multiple comparison procedures updated.* Clin
Exp Pharmacol Physiol 1998; PMID 9888002,
[DOI](https://doi.org/10.1111/j.1440-1681.1998.tb02179.x) -- its abstract
commends Holm's step-down procedures for "accuracy, power and
versatility" (abstract-only, flagged; no PMC full text). Ludbrook J.
*Multiple inferences using confidence intervals.* Clin Exp Pharmacol
Physiol 2000; PMID 10744350,
[DOI](https://doi.org/10.1046/j.1440-1681.2000.03223.x) -- abstract only,
not used. White GM, Siegel AP, Tovar A. *Optimizing thermoplastic starch
film with heteroscedastic Gaussian processes in Bayesian experimental
design framework.* Materials 2024; PMID 39517615, PMC11547296,
[DOI](https://doi.org/10.3390/ma17215345) -- an application; not opened,
not used. Full text not accessible for the original statements of Holm
(Scand J Stat 1979) and of the constant-liar heuristic (Ginsbourger,
Le Riche, Carraro 2010) -- neither is indexed in PubMed; the procedures
are read from the project's code and from scikit-optimize's source, and
the peer-reviewed statements above stand in for them. Not in PubMed:
Rasmussen CE, Williams CKI, *Gaussian Processes for Machine Learning*
(2006), `[textbook, from memory]` for the posterior formulas named in
S3.2 (d).

**Searches run `[RAN]`, 2026-10-02.**

| source | query | result |
|---|---|---|
| PubMed | Gaussian process + expected improvement + hyperparameter optimisation (the string not preserved across the chat's compaction; the record list was) | 6 records; PMC7861299 read in full and used; PMC7513107 (K-optimality) and PMC13168308 (an application) not opened |
| PubMed | batch Bayesian optimisation (as above) | 14 records; PMC12459664 read in full and used; three applications not opened |
| PubMed | ("Holm-Bonferroni" OR "Holm procedure" OR "Holm's procedure") AND ("multiple comparisons" OR "multiple testing") AND step-down | 0 records |
| PubMed | Holm procedure multiple comparisons familywise error rate review Bonferroni step-down | 2 records (PMID 10744350, 9888002; abstracts only, flagged above) |
| PubMed | Holm Bonferroni multiple testing correction tutorial | 0 records |
| PubMed | "multiple comparisons" AND Holm AND Bonferroni AND (statistics OR biostatistics) AND review[Publication Type] | 5 records; PMC5935793 read in full and used; PMID 32106113 (a meta-analysis), 39584590 (a reporting review), 9888002 (abstract only) not used beyond the flag |
| PubMed | "nested cross-validation" AND (hyperparameter OR "model selection") AND (optimistic OR "optimistic bias" OR "selection bias") | 7 records; PMC6191021 read in full and used; the others applications (PMID 42342721, 40989918, 34553171, 25092249, 16844704) and one arXiv record (41675351), not opened |
| PubMed | "Gaussian process" AND ("Bayesian optimization" OR "Bayesian optimisation") AND ("noisy" OR "observation noise" OR "nugget") AND (replicat* OR "repeated evaluations") | 0 records |
| PubMed | "Bayesian optimization" AND "Gaussian process" AND ("heteroscedastic" OR "noise variance" OR "nugget" OR "stochastic simulator") | 1 record (PMID 39517615, an application; not opened) |
| PubMed | "partial dependence" AND (hyperparameter OR "Bayesian optimization" OR "surrogate model") AND ("Gaussian process" OR "machine learning") | 60 records, applications of partial-dependence plots; none opened |
| PubMed | "partial dependence plot*" AND (interpretab* OR explainab*) AND (tutorial OR review[Publication Type] OR "accumulated local effects") | 13 records; PMC12576049 (its partial-dependence section) read and used; PMC10442734 returned an empty full text; the rest applications and reviews not opened |
| bioRxiv | bioinformatics, last 30 days, first page of 30 records (the connector has no keyword search) | none on hyper-parameter search, Gaussian-process surrogates or SBI tuning; no preprint is cited, so the published-version lookup had nothing to check |
| bioRxiv | neuroscience, last 30 days, first page of 30 records | none on topic; no preprint is cited |
| KB PDFs | `Bayesian optimi`, `hyperparameter (search / optimi / tuning)`, `grid search`, `random search`, `Gaussian process` over the seven extracted texts | the practical guide p.31-32, BayesFlow p.13-14, the frontier review p.4, p.7, Goncalves et al. p.3, p.16, p.30-33, the multilevel preprint p.3, p.12, p.14, p.16, as cited |

**Textbook, from memory** (tagged as such where used): the GP posterior
predictive mean and variance (S3.2 (d)); that a two-level categorical
one-hot encodes in one coordinate is read from the library's output, not
from memory `[RAN]` B1.

---

### Pre-send check (Precision model)

R1 -- every symbol typed in S1; $\mathcal{K}$ maps $\mathcal{X}$ to itself,
$k_{\rm GP}$ takes two transformed points, $\mathrm{EI}$ takes a point and
returns nats per row; the test statistic is dimensionless, its inputs nats
per row. R2 -- "replays the design" carries "with the default `random`
generator and the same seed"; "identical controls" carries "up to
floating-point non-determinism" and "at `834eb41`"; "no four-way split"
carries "at `834eb41`"; the EI incumbent's bias carries "with a noisy
objective"; the standardisation claim carries the two scikit-learn
versions read and the unrecorded cluster version. R3 -- the Holm procedure
is stated for the family the code builds (testable finalists), not for
$K_{\rm fin}$ when some are untestable; the selection-bias argument is
transplanted from cross-validated configuration selection to a
GP-selected ledger only as a mechanism (a minimum over noisy estimates),
with the source's conditions named and no number carried; the DSN
search's budget is compared with the joint driver's as counts, not as
equivalent designs. R4 -- $\sigma_{\rm seed}$ and $\sigma_{\rm noise}$ are
two objects in two units; $s_{\rm seed}$ and $s_{\rm perm}$ two seeds;
"split", "role" and "label" three names; $K_{\rm fin}$ is not E0's $K$;
$\jmath$ and $\imath$ are not $j$ and $i$; $\mathsf{x}$ and $\mathsf{c}$
are not $x$ and $c$. R5 -- the maps named with domain and codomain:
$\mathcal{K}$, $\Pi_{\rm leg}$, the point-to-configuration map through the
pins, the standardisation and its inverse, the transformation to the
surrogate's coordinates (and its dimension change 23 to 29). R6 --
"split", "seed", "noise", "Bayesian optimisation", "control", "inactive",
"design" declared in S1.1 or S2. R7 -- the plan's Stage 4 sentence and the
handoff's eq. (S4.1) are quoted with what the code omits; the warning text
is quoted with the condition under which it prints; the docstring's noise
sentence with its units. R8 -- $L(\mathsf{c})$, $L_\imath$, $\mu_{\rm GP}$
and $\hat\Delta^{\rm ctrl}_\jmath$ carry their levels in S1 and convention
6; the move from one realisation to a p-value is named as the step F-an
breaks. Confirmation questions: none in the brief.
