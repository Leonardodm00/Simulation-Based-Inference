# HANDOFF -- joint DSN+NPE implementation session (2026-09-02 -> 09-03)

**Purpose.** Everything built, corrected and left open in the session that
IMPLEMENTED Stages 1, 2, 3, 3b and 3c of `JOINT_DSN_NPE_PLAN_v0_6.md`. Written
to be pasted into a new chat. Append to the changelog rather than editing
history. The design-session handoff (`HANDOFF_joint_dsn_npe.md`, 2026-08-28 ->
09-02) is still current for everything upstream of the code and is NOT
superseded by this one; read it first if you need the argument rather than the
artefact.

**Confidence legend** (same as the design handoff):

- **[KB]** -- read from the project knowledge base this session.
- **[REPO]** -- read from repository source this session (tarball of the named
  branch, not the cluster checkout).
- **[RAN]** -- a command was run in the analysis sandbox and its output seen.
  *Not* the cluster: **nothing in this session touched the HPC.**
- **[DERIVED]** -- my reasoning, not read anywhere. Check before relying on it.
- **[STATED]** -- asserted by the user, not independently verified.
- **[OPEN]** -- not established; decision or measurement outstanding.

---

## Changelog

| date | change |
|---|---|
| 2026-09-03 | Stage 3c implemented (floors, aliasing, stratification, P10). Three bugs found in it, all fixed. Second full audit of Stages 1-3 completed: seven further bugs. |
| 2026-09-02 | Stages 1, 2, 3, 3b implemented. `JOINT_DSN_NPE_PLAN_v0_6.md` and `METRIC_REPLICATE_v1_1.md` written. First audit: five bugs plus one plan deviation. |

---

## 1. What exists now

All under `/mnt/user-data/outputs/`, ASCII/LF-clean, `py_compile`-clean,
`bash -n`-clean [RAN].

| file | role | status |
|---|---|---|
| `JOINT_DSN_NPE_PLAN_v0_6.md` | the staged plan | **current**; supersedes v0.5 |
| `METRIC_REPLICATE_v1_1.md` | the metric/null derivation behind S2.5 | **current**; supersedes v1.0 |
| `HANDOFF_joint_dsn_npe.md` | design-session handoff | still current for the argument |
| this file | implementation-session handoff | **current** |

**Code, 9,245 lines across 36 files [RAN].** Layout is `stage<N>/` at the repo
root with `stage<N>/jobs/*.pbs`, plus `replicate_statistic.py` at the top
level because Stage 2 imports it as its specification.

| stage | modules | job | tests |
|---|---|---|---|
| (shared) | `replicate_statistic.py` | -- | 14/14 |
| 1 | `latent_sbi_simulator.py`, `latent_nuisance.py`, `latent_realisation.py`, `latent_gap.py`, `latent_bank.py`, `build_latent_bank.py` | `build_latent_bank.pbs` | **47/47** |
| 2 | `joint_model.py`, `joint_batches.py`, `joint_losses.py`, `joint_train.py`, `dsn_loss_adapter.py` | -- | **40/41** + **27/27** |
| 3 | `run_joint_arms.py`, `joint_diagnostics.py`, `report_joint_arms.py` | `joint_arms.pbs` | **28/28** |
| 3b | `domain_objective.py`, `encoder_probes.py`, `run_stage3b.py` | `stage3b.pbs` | **23/23** |
| 3c | `floor_core.py`, `nuisance_floor.py`, `realisation_floor.py`, `aliasing.py`, `stratify.py`, `window_aggregation.py`, `run_stage3c.py` | `stage3c.pbs` | **24/24** |

The one skip is **J5** (one host sync per epoch), which needs a CUDA device;
assert it on the cluster with `torch.cuda.set_sync_debug_mode`.

**Nothing has run on the HPC.** Every number above is from the sandbox. The
whole point of the next session is the cluster dry-runs.

---

## 2. Environment the code was verified against [RAN]

```
torch 2.13.0+cu130      sbi 0.27.0        zuko 1.6.0
numpy 2.4.4             scipy 1.17.1      pytorch_metric_learning 2.9.0
sklearn 1.8.0 (optional: cluster_scores degrades to None without it)
```

The cluster's `sbi_env` is NOT this. **Version parity is unverified and is the
single most likely cause of a cluster-side failure** [OPEN]. Two couplings
that will break first on a different sbi:

- `joint_model.rsample_posterior` reaches into `estimator._embedding_net` and
  `estimator.net` (S4 below);
- `build_joint_model` passes the prior as `x_dist` (S4 below).

Repository sources were read from GitHub tarballs, not the cluster checkout
[REPO]:
`Deep-Summary-Network@main`, `Sbi-extractor@feat/real-arm-parity`,
`Simulation-Based-Inference@main` and `@feat/misspec-gate`.

Environment variables the code reads: `DSN_MAIN_DIR` (the DSN repo's `Main/`
directory -- use the `dsn_main` symlink, the real path contains a space),
`SBIX_DIR` (Sbi-extractor checkout, only for smoke test S4e), `SBI_HPC_DIR`
(the SBI repo's `hpc/`, for `npe_diagnostics` and `bootstrap_paired`).

---

## 3. Repository findings -- act on these before writing code

**`bootstrap_paired.py` is in NO repository** [REPO, checked `main`,
`feat/misspec-gate`, and both other repos]. Plan v0.6 Stage 1 records it as
"delivered and verified"; that is true of the artefact handed over in the
design session and false of the repository. Every paired interval in
`report_joint_arms.py` and `run_stage3b.py` therefore degrades to a point
estimate marked *incomplete*. **It must be committed before Stage 3's exit
criterion can be met.** Both call sites are written to the plan's S3 names and
wrapped in try/except, so an API mismatch produces a note rather than a lost
table -- reconcile the signatures on first real use.

**`feat/misspec-gate` is merged into `main`, and `main` is ahead** by
`npe_regions.py`, `prior_truncate.py` and their smoke tests [REPO]. Plan v0.6
Stage 6 says to branch `feat/joint-dsn-npe` off `feat/misspec-gate`; that would
branch off a stale point. **Branch off `main`.**

**`feat/joint-dsn-npe` does not exist on origin** [REPO, 404]. It has to be
created.

---

## 4. Two couplings to sbi 0.27 internals, both load-bearing

**(a) `estimator.sample` is not differentiable.** It calls the zuko
distribution's `.sample()`, which detaches. Any replicate loss built on it has
NO gradient path to psi at all. `JointDSNNPE.rsample_posterior` reaches past
the public API to `_embedding_net` and `net` and calls `.rsample()`.
`joint_train` raises if the resulting loss arrives without `requires_grad`.
Smoke tests J14r-c and J14r-d assert both failure modes.

**(b) `z_score_theta="transform_to_unconstrained"` needs the prior.** Outside
sbi's `NPE` class it raises

```
ValueError: Transformation to unconstrained space requires a distribution
            provided through `x_dist`.
```

sbi names the flow's own variable `x` internally, so `x_dist` is the prior over
**theta**; the error message does not say so. `build_joint_model` requires the
prior and passes it as `x_dist`, and raises rather than falling back to
`"none"` -- a silent fallback would change the parameter space the flow works
in and make the joint arms incomparable with A0. **This belongs in the Stage 0
contract checks, which do not yet exist as an artefact** [OPEN].

---

## 5. Bugs found and fixed, by stage

Sixteen inline `[CORRECTION]` markers are in the shipped source [RAN,
grepped]. Each has a regression test. Grouped by what they would have cost.

### 5.1 Would have produced wrong science silently

| what | where | consequence if unfixed |
|---|---|---|
| The replicate term was **disconnected**, two independent causes: `stop_grad_params(model.estimator)` froze psi as well as omega (sbi keeps the encoder inside the estimator), and `estimator.sample` detaches | `joint_losses`, `joint_train`, `joint_model` | arm A5 scored **identically to A1 to four decimals** -- reads as "the term does little", not "the term does nothing" |
| **A2s was identical to A2**: the metric stream always read `real_x` | `joint_batches`, `run_joint_arms` | eq. (9)'s domain/objective split computed from two copies of one run |
| The runner scored the **simulated** split and called it `L` | `run_joint_arms` | D10 makes the pseudo-real NLL primary; P6 says the two rankings need not agree, so P6 would have been untestable |
| Gap (a) shifted **phi** instead of the axis RANGES | `latent_gap`, `latent_sbi_simulator` | DSN generator refuses out-of-box phi; recorded theta would leave the prior box, making `prior_log_prob` -inf and every pseudo-real gain undefined |
| `nu = 0` was **not the identity**: dropout logit 0 means 50% of electrodes lost | `latent_nuisance` | mean cohort electrode loss 47.8%; the nuisance floor measured a fixed 4/9 attenuation as well as the variation |
| The realisation nesting was **inverted**: subregions of a well shared one graph | `latent_realisation` | plan S2.6 Lever 2 reversed; the nesting that breaks the nuisance/mechanism confound absent while the config read as correct |
| A "frozen" encoder was **stochastic during training**: `evaluate_npe` calls `model.train()`, re-enabling dropout | `joint_model` | A0's flow trained on a noisy z and scored on a deterministic one |
| Probe 1 attributed dead units to the **wrong layer**: `nn.ReLU(inplace=True)` rewrote the tensor the hook held a reference to | `encoder_probes` | `norm1` reported 39% dead; every "dead conv/norm" reading was the following ReLU's |

### 5.2 Would have crashed or refused, loudly

| what | where |
|---|---|
| Checkpoints could not be reloaded: rebuild used default flow hyper-parameters (64/5/10) against a checkpoint trained at 48/3/8. **Would have broken every Stage 4 trial**, since the search varies all three | `run_stage3c` |
| `FixedStatsSummary` had no parameters; sbi's `check_net_device` does `next(net.parameters())` and raised `StopIteration` | `run_joint_arms` |
| `p_eff` returned 1e300 from a near-singular `Cov(theta*)` at 14 report rows, d = 6 | `joint_diagnostics` |
| Loss accumulation started on a CPU zero tensor -- fine in the sandbox, broken on a GPU | `joint_train` |

### 5.3 Bad estimators and bad gates

| what | where | why it mattered |
|---|---|---|
| The Stage 3c validation gate required `n_above_null == n_expected` **exactly** | `stratify` | mu_j is noisy; null directions fluctuate around 1/(2k) and about half exceed it. A correct run (mu = [8.30, 0.29, 0.22, 0.15], null 0.25) FAILED. Replaced with separation ratio + identification |
| Concentration threshold "twice uniform" is **unreachable** for k/d >= 0.5 | `realisation_floor` | a floor with 100% of its variance on the kernel axes was called "not concentrated". Survived because the real bank is 3/26 where the rule happens to work |
| The report paired **seed 0 only** | `report_joint_arms` | sigma_seed never formed, so the second clause of the S2.4 rule could not be applied at all |
| An **empty selection split** scored 0.0, so early stopping locked epoch 0 as "best" | `joint_train`, `run_joint_arms` | the run reported an untrained model, silently |
| Weight decay applied to **frozen** encoders | `joint_train` | AdamW decays a frozen tensor toward zero every step even with `grad = None`; arms A0/A0s/A_ref would drift while "held fixed" |
| Surrogate rows could form replicate pairs: relabelling donors `-1, -2, ...` **truncates on a narrow string dtype** (`'-10'` -> `'-1'`) | `joint_batches` | two surrogates fuse into one donor |
| The gap "slow drift" restarted at a **random phase every window** | `latent_gap` | a within-window artefact at the drift period, not a slow drift |
| `--n-per-theta` **enforced nothing** | `build_latent_bank` | a bank built with `--n-per-theta 2 --wells-per-donor 1` looked configured for the D17 check while containing the exact confound it detects |
| Two **vacuous assertions** (`... or True`) in J3b and J8a; J8a also used `(z**2).sum()`, which is constant under L2 normalisation and has gradient exactly zero | `smoke_test_joint` | tests passing while measuring nothing. Zero remain [RAN, grepped] |

---

## 6. Deviations from plan v0.6, and why

| plan says | code does | reason |
|---|---|---|
| per-AXIS information gain (P1, P2, Stage 3b "the same split per axis") | per-axis **contraction** | a flow has no closed-form marginals; per-axis KDE across 26 axes and every held-out row would dominate Stage 3's cost and carry unquantified bias. Labelled in every output |
| mu_j >> 1 marks heterogeneity (S2.7) | null is **1/(2k)**, reported alongside | a donor mean from k wells varies as Sigma_rep/(2k) under the null. Comparing against 1 is conservative by 2k -- safe direction, but the threshold should be stated. Derivation in `stratify.py` |
| gap perturbation (b), heavy-tailed burst durations | **raises** | its override keys are not bound to `BurstParams` fields. Fails loudly rather than acting as a no-op |
| Stage 1 deliverables named `nuisance_floor.py`, `realisation_floor.py` | plus a shared `floor_core.py` | the two floors differ only in which argument is resampled; one driver stops them drifting apart in how they simulate, encode and summarise |
| D17 open | **option (b) is the default**: all 26 axes in the diagnostic, the 3 kernel axes maskable from the loss | an overstated `p_eff` on those axes is not conservative in a loss -- it pushes `T_gg'` UPWARD, training the encoder to make same-donor wells disagree more |

---

## 7. Numbers of record from this session [RAN]

All from the sandbox on small benches; none is a scientific result.

- Stage 3 arms run end to end on a DSN-generated bench: A1, A0, A0s, A2, A2s,
  A3, A5, A_ref, shuffled. A2 vs A2s now diverge (0.5749 vs 0.5765
  pseudo-real NLL) where they were bit-identical before the fix.
- eq. (9) identity residual: **1e-17 to 1e-19** on every seed.
- `nu = 0` gives `A = 1.000000000000`, mean electrode loss 1.8% (was 47.8%).
- Stratification on a planted case: mu = [8.30, 0.29, 0.22, 0.15], null 0.25,
  separation ratio 28.4; null cohort 1.19-1.32 and correctly fails.
- Aliasing on a constructed linear map: a_m = 1.0000000000 inside the
  parameter span, 1.29e-16 orthogonal to it, `delta_theta` recovers the
  planted coefficients to 1.8e-15.
- `T` vs `p_eff` on a 2-epoch flow: 0.012 vs 2.355, i.e. far on the COLLAPSE
  side of the target -- exactly the warm-up rationale of S2.5e, observed.
- `rho_grad = -0.376` in a 3-term integration run: P4's signature, seen once.

---

## 8. Design decisions made in code, not in the plan [DERIVED]

Each is defensible and each is a place a reviewer may disagree.

1. **`C_bar` is symmetrised** across the two wells, `(C_g + C_g')/2`. The plan
   writes a single `C` without saying at which well. Symmetrising makes
   `T_gg'` exchangeable in `(g, g')`, which the unsymmetrised form is not.
2. **The finite-sample correction `d/S_mc`** is applied by default (eq. 3h).
   At `S_mc = 100` the inflation is ~10% of a `p_eff` of 3.
3. **`p_eff <= 0` is clamped and COUNTED**, not raised. It means the posterior
   is at least as wide as the prior -- a G1 failure -- and it is expected
   before the warm-up ends. A non-zero `n_p_eff_invalid` after the ramp
   completes is a finding.
4. **Negative corrected `T` is counted, not clamped away** (`n_T_nonpositive`).
   It is the collapse signature, which `replicate_loss`'s clamp would hide.
5. **`Sigma_rep` is taken about ZERO**, not about the sample mean: subtracting
   a sample mean would remove a real common shift and shrink the floor.
6. **Common random numbers everywhere in the floors and the Jacobians.**
   Without them a central difference estimates the derivative plus a noise
   difference whose variance grows as 1/(2h)^2 -- worst at exactly the small
   steps chosen for accuracy.
7. **Importance sampling for the P10 normalising constant**, with the ESS
   reported per point and untrusted points excluded from the slope fit: a
   curve that bends because the estimator broke looks exactly like one that
   bends because information saturated.
8. **A quantised nuisance component reports `a_m = None`, not 0.** Electrode
   dropout rounds to a whole number of electrodes, so the map is piecewise
   constant; `a_m = 0` would read as "perfectly separable", the opposite of
   the truth.

---

## 9. What is NOT done

- **Stage 0** exists only implicitly. No `check_env.py` extension file. The
  two sbi couplings of S4 above belong in it, plus the `d_theta = 26`
  assertion against `artifacts/label_axes.json`.
- **Stage 3's exit criterion**: a written verdict on P1-P4, P6, P9, P13, P15,
  P16 and the degradation rule, with numbers, appended to the changelog. Needs
  a real campaign (`n_seed >= 5`, real bench sizes) on the cluster.
- **Stage 4** entirely: `joint_space.py`, `npe_tune_joint.py`, the tuning PBS
  jobs, `smoke_test_joint_tune.py`.
- **Stages 5-8** entirely.
- **Every cluster dry-run.** Agreed plan [STATED]: finish Stage 4, then run
  all smoke tests and all dry-runs on the HPC in one pass.

---

## 10. Open decisions carried forward

Unchanged from plan v0.6 S8 except where noted.

- **D1-D4, D12-D14**: still open. The code assumes D1's recommended option
  (three-stream batching) and exposes `--wells-per-donor`, which is where D12
  bites.
- **D15** (which metric for eq. 3c): `(2 C_bar)^{-1}` implemented as default;
  `V^{-1}` needs `N_pair >= 27` at d = 26.
- **D16** (point estimates or full posteriors): unchanged; nothing yet checks
  whether the posteriors are multimodal.
- **D17** (realisation marginalisation on the kernel axes): default is option
  (b). `latent_realisation.distinct_realisations_per_theta` is the audit --
  run it on the REAL bank keyed on the three Weibull axes; expected answer is
  1, which is the problem.
- **D18** closed in v0.6 (all 26 axes).

---

## 11. Next actions, cheapest first

1. **Commit `bootstrap_paired.py`** to `main`. Everything paired is blocked on
   it. [~minutes]
2. **Create `feat/joint-dsn-npe` off `main`**, not off `feat/misspec-gate`.
3. **Write Stage 0's `check_env.py` additions**, including the two sbi
   couplings of S4 and the `d_theta = 26` assertion. [~1 session]
4. **Stage 4**: the search campaigns. [agreed next]
5. **Then the cluster pass**: every smoke test and every dry-run, in order --
   `stage1/smoke_test_latent_sbi.py`, `stage2/smoke_test_joint.py`,
   `stage2/smoke_test_joint_losses.py`, `stage3/smoke_test_joint_arms.py`,
   `stage3b/smoke_test_stage3b.py`, `stage3c/smoke_test_stage3c.py`, then
   `DRYRUN=1` on each of the four PBS jobs.
6. **P8 and P15 are still the cheapest real measurements** and need no new
   simulation: P8 is one `evaluate` run on the existing bank; P15 is one
   `information_spectrum` call on both index sets.

---

## 12. Conventions established this session

- **Stage layout**: `stage<N>/` with `stage<N>/jobs/*.pbs`; smoke tests live
  beside the modules they test and exit non-zero on failure so they drop into
  the cluster-side verification block unchanged.
- **Every PBS script self-locates** from `BASH_SOURCE`/`PBS_O_WORKDIR` by
  looking for a known file, and fails loudly naming both candidates.
- **`DRYRUN=1` on every job** prints the resolved plan and the expected output
  without allocating anything.
- **Pure ASCII, LF-only, byte-scanned** before every hand-over.
- **A test that cannot fail is not a test.** Two vacuous assertions were found
  and removed; the suites are grepped for `or True` before every delivery.
- **Numpy modules are the specification** for their torch counterparts
  (`replicate_statistic.py` for `joint_losses.py`), checked by a parity test
  (J16) to 1e-9. Where they disagree the numpy one is right.
- **Estimators are tested against planted truths**, never against a second
  implementation of the same estimator.

---

## 13. Questions asked and not answered

- Which sbi version is in the cluster's `sbi_env`? [OPEN] Determines whether
  the two couplings in S4 hold.
- How many distinct donors are behind the 35 cultures, and does any donor
  appear in more than one well? [OPEN, = D12] Gates arm A5 and all of S2.7.
- Are there paired pre/post-compound recordings? [OPEN, = D14] Gates Stage 7.

---

## 14. What this session did not do

- Did not touch the cluster.
- Did not run any arm to convergence: every run was 2-4 epochs on a small
  bench, sufficient to exercise the plumbing and nothing more.
- Did not commit anything to any repository.
- Did not resolve the two knowledge-base numeric inconsistencies recorded in
  the design handoff (29,616 vs 29,416 rows; `d_theta` fixed at 26 vs
  determined by `label_axes.json`).
- Did not write the Stage 3 verdict, which is Stage 3's actual exit criterion.
