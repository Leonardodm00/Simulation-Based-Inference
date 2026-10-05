# P7 -- Upstream and jobs: the Stage 1 bank, its identity fields, the job scripts, and the Stage 3b/3c flags

**Document P7 of the joint documentation set.** Owner of the knobs that fix
the problem before any training starts (P0 S3.8) and of the job layer around
them (P0 Table E): the Stage 1 builder's flags `--provider`, `--n-traces`,
`--wells-per-donor`, `--donors-per-batch`, `--n-windows`, `--T-win`, `--fs`,
`--n-neurons`, `--n-latent`, `--n-label-axes`, `--tau-ov`, `--pi`,
`--gap-modes`, `--n-per-theta`; the four spec classes behind them,
`LatentSBISpec`, `NuisanceSpec`, `GapSpec`, `RealisationSpec`, whose other
fields no flag reaches; the module constants `BENCH_AXES`,
`BENCH_LABEL_IDX`, `BENCH_FREE_IDX`, `PAD_BINS`, `GAP_MODES`,
`NU_COMPONENTS`, `N_COMPONENTS`, `NU_LEVELS`; the demonstration scripts'
flags; the job variables of `build_latent_bank.pbs`, `stage3b.pbs` and
`stage3c.pbs` and the array plumbing of `joint_arms.pbs` and
`joint_tune.pbs`; the resources of all eight job files; and, in Appendix A,
the flags of Stages 3b and 3c, which act after training
(`--max-probe-rows`, `--n-floor-draws`, `--n-post-draws`, `--kernel-axes`,
`--spread-axes`, `--fd-step`, `--fd-seeds`), with the probe's `--window`.
The training variables the jobs carry (`EPOCHS`, `B_SIM`, `LAMBDA_DSN`,
...) are P1-P5's, the search driver's P6's. Master notation: E0. The
chapter that explains the bench itself -- the generator, the two arms, the
gap, the nuisance and realisation latents -- is E6. **Date:** 2026-10-04
(v1.1). **Applies to:** the repository `Simulation-Based-Inference` at
`834eb41` (`origin/main` is at `2c0a06d`, and `git diff --stat 834eb41
origin/main` is empty over `hpc/joint` outside `docs/` and over the DSN
modules the providers import `[RAN]`), `hpc/joint/` (D-037): `stage1/` (the
builder, the five library modules, the bench generator and provider, the
two demonstration scripts, and the two Stage 1 smoke suites it cites),
`stage1/jobs/build_latent_bank.pbs`, `stage3/jobs/` (the arms job and the
two probes), `stage3b/run_stage3b.py` and its job, `stage3c/run_stage3c.py`
with the five modules it calls for the floors, the aliasing and the
stratification, and its job, `stage4/jobs/` (the tuning job and the
launcher), and the parts of `stage2/joint_batches.py` and
`stage3/run_joint_arms.py` that read a bank's identity fields; the DSN
generator modules the providers import, `hpc/dsn/latent_burst_generator.py`
and `hpc/dsn/generate_burst_data.py` (imported, never edited); the plan
`JOINT_DSN_NPE_PLAN_v0_6.md` at its repository version v0.6.5 (S2.6, S4,
Stage 1, Stage 3b, Stage 3c, S8). Sandbox: numpy 2.5.3, scipy 1.18.1,
matplotlib 3.11.2, no torch. Every number of S3 and of Appendix A is the
output of `tools/p7_numbers.py` `[RAN]`, which imports the Stage 1 modules
themselves, builds small banks through the real command line in a temporary
directory, and extracts the two functions it needs from torch-importing
modules (`enumerate_donor_pairs`, `grouped_split`) with `ast`; run twice,
identical output.

| date | change |
|---|---|
| 2026-10-05 | v1.2. One dated note from E4, nothing else changed: convention 5 ("Held-out") now says for which arms arm R's rows are held out in its first sense -- those that draw no real stream (`A1`, `A0s`, `A2s`, `A_ref`, `shuffled`) -- and that for `A0`, `A2`, `A3` and `A5` the pseudo-real endpoint is held out in $\theta$ alone. Evidence: E4 S3.7; `run_joint_arms.py:452-459, 491-508` at `834eb41` `[REPO]`. |
| 2026-10-04 | v1.1. One correction and one formatting repair, nothing else changed: S3.2 (b) repeated `simplex_centres`'s docstring in calling the bench's class centres the vertices of a regular simplex; for $C \ge 3$ they are not (finding F-ba, owner E6), and the sentence is marked [corrected 2026-10-04]; four inline formulas broken across two lines (that one, one more in S3.2 (c) and two in Appendix A.2) are rejoined onto one line each, so that the notation checker reads them; no symbol changed. Evidence: `latent_sbi_simulator.py:128-155` read, and the centres' geometry computed against the DSN's `_class_center_vectors` (`hpc/dsn/latent_burst_generator.py:460-527`) by `tools/e2_numbers.py` B4 `[RAN]` (E2 S5). |
| 2026-10-03 | v1. Written from `stage1/build_latent_bank.py`, `latent_bank.py`, `latent_sbi_simulator.py`, `latent_nuisance.py`, `latent_gap.py`, `latent_realisation.py`, `bench_burst_provider.py`, `bench_burst_generator.py`, `demo_classes_generate.py`, `demo_classes_plot.py`, every job file under `stage1/jobs/`, `stage3/jobs/`, `stage3b/jobs/`, `stage3c/jobs/`, `stage4/jobs/` (all read in full), `stage3b/run_stage3b.py`, `stage3c/run_stage3c.py`, `stage3c/floor_core.py`, `nuisance_floor.py`, `realisation_floor.py` (read in full), `stage3c/aliasing.py:54-154`, `stage3c/stratify.py:136-196`, `stage3b/domain_objective.py:62-150`, `stage3b/encoder_probes.py:76-125`, `stage3/probe_dsn_runtime.py:1-40, 315-360`, `stage3/run_joint_arms.py:110-200, 262-470, 470-659`, `stage2/joint_batches.py:1-216`, `stage1/smoke_test_latent_sbi.py:20-125, 542-600`, `stage1/smoke_test_bench_provider.py:88-118`, `hpc/dsn/latent_burst_generator.py:193-206, 252-305`; the plan S2.6, S4.0-S4.5, Stage 1, Stage 3b, Stage 3c, S8 (D10, D11); `[KB]` `JOINT_DSN_NPE_USAGE_v1.md` v1.3 S1-S10, `claude/PLAN_2026-10-01_giulia_hpc_stages.md` v1.7 S3 (B5, B6), S5 (G0, G4, G5), `claude/deck_pack/07_SEC_F_bench_and_stages.md`, `claude/REPLICATE_LOSS_GUARDS_v1.md` S3.7, `HPC_PATHS.md` sec. 5-7, P0, P3 S3.8, P5 S3.3, P6 S3.8; `[KB-PDF]` Schmitt et al. p.5-6, p.11-12, the multilevel paper (NeurIPS 2025) p.1. Every number computed by the new `tools/p7_numbers.py` `[RAN]`. Findings F-at to F-az added; F-n, F-t, F-aa, F-ab, F-ae and F-ar carried on their bank or job side. Grounding searches of S6 run and reported. |

**Abstract.** One would expect a simulation bank's knobs to be data
preparation -- how many traces, how long a window, which seed -- that
changes how much a network sees and not what a run means. In the joint stack
the bank fixes the problem itself: the provider sets the latent dimension
$p$ and the scale of $x$, the window grid sets $W$, `--n-classes` sets $C$,
the donors set the split and every replicate pair, and the nuisance scales
are written in units of $x$. And four records of a bank -- its identity
fields, its provider, its bench arm's draws and its gap mode -- do not mean,
downstream, what the downstream code takes them to mean; that distance is
this document's subject. The questions answered, for each knob of P0 S3.8
and of P0 Table E that P7 owns: where it is set and which surface reaches it
(S3.1); what one shard is, written out from the builder as one chain -- the
layout and the seeds, the prior, the provider, the gap, the nuisance, the
sidecar and its digest, eq. (P7.1)-(P7.9) (S3.2); what each knob changes
(S3.3); what the eight job files request, forward and assume (S3.4); what
the upstream knobs fix for the training knobs of P1-P6 (S3.5); how the bank
and the jobs fail and which line reveals each failure (S3.6); the findings
(S3.7); and, in Appendix A, the post-hoc flags of Stages 3b and 3c with the
floor covariance written out, eq. (P7.10)-(P7.11). Carried: F-n, and the
bank or job side of F-t, F-aa, F-ab, F-ae and F-ar. New: the identity fields
`donor`, `well` and `batch` restart at 0 in every shard and `concat_shards`
does not offset them, so on a bank of $S_{\rm sh}$ shards the replicate
pairs, the replicate diagnostic, the split and Stage 3c's culture means group
unrelated donors and wells -- at the usage guide's 32 shards, 97.1 % of the enumerated
pairs join two different $\theta$ (F-at); Stage 3c simulates its floors and
Jacobians through the DSN provider whatever built the bank, so it raises on a
bench bank and silently swaps the simulator on a reference bank (F-au); the
gap (a) is a different perturbation on each provider and, on the bench
provider, raises as soon as one donor's `fragment_duty` coordinate exceeds
$1 - \pi \delta_{\rm shift}$, which at $\pi = 0.1$ happens in 68 % of default
shards (F-av); bench arm R built with bench arm S's `SEED` repeats its
draws -- $\theta$, class, $\nu$, realisation seeds, and at $\pi = 0$ every
byte of $x$ -- so at $\pi = 0$ the pseudo-real endpoint, which scores every
arm-R row, scores copies of training rows in 68.75 % of them (F-aw); the demonstration's
`--nuisance` raises at the first trace (F-ax); three job scripts expand a
possibly empty array under `set -u`, which bash 4.3 reports as an unbound
variable (F-ay); and the dry runs of two jobs read every shard before they
print the plan (F-az). **Deliberately excluded:** the burst generator's
physics and why each perturbation is a misspecification (E6); the replicate
statistic (P3, E5); the arms and the objective (E4); the decision rule and
what the diagnostics mean (E7); the search (P6, E8); any claim about
outcomes -- no job of `hpc/joint/` has run on the cluster (`[KB]` usage v1.3
S9), and every number here is a property of the code as written, exercised
on banks built in the sandbox or derived in closed form and marked so.

---

## 1. Notation and symbols

A subset of E0's master table -- the same symbols, with the type and units
specialised to the bench where E0 states the cohort's ($x$, $\theta$,
$\theta^*$, $\nu$) -- plus the forty-three symbols this document adds, which
E0 v1.7 declares in its group "Upstream: the bank and the post-hoc stages
(P7)" (30 rows). Rows are in order of first use.

| Symbol | Name / Meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| $p$ | latent dimension of a bank, as its sidecar records it (`n_latent`) | $\mathbb{N}$ | -- | S3.1 |
| $x$ | one window as a provider returns it: IFR samples of one trace | $\mathbb{R}^{W}_{\ge 0}$ | counts per bin (`sum_over_units`) or counts per bin per neuron (`per_unit_mean`) | S3.1 |
| $x^{\rm obs}$ | the window after the nuisance map: the array a bench shard stores under `x` | $\mathbb{R}^{W}$, can be negative | as $x$ | S3.2 |
| $W$ | window length in samples, $W = \mathrm{round}(T_{\rm win} f_s)$ | $\mathbb{N}$ | samples | S3.1 |
| $T_{\rm win}$ | window duration (`--T-win`) | $\mathbb{R}_{>0}$ | s | S3.1 |
| $f_s$ | sampling rate of the IFR trace (`--fs`), $f_s = 1/\Delta t$ | $\mathbb{R}_{>0}$ | Hz | S3.1 |
| $\Delta t$ | IFR bin width | $\mathbb{R}_{>0}$ | s | S3.3.4 |
| $C$ | number of classes (`--n-classes`) | $\mathbb{N}$ | -- | S3.1 |
| $c$ | class label of a donor | $\{0, \dots, C - 1\}$ | -- | S3.2 |
| $\theta$ | the parameters of one row; on the bench $\theta := \phi$ | $(0, 1)^{d_\theta}$ on the bench | dimensionless | S3.2 |
| $d_\theta$ | parameter dimension; $d_\theta = p$ on every bank built so far | $\mathbb{N}$ | -- | S3.5 |
| $\phi$ | the bench generator's latent vector; $\phi^{(k)}$ its $k$-th component | $(0, 1)^{p}$ | dimensionless | S3.2 |
| $\phi^{\rm nat}_k$ | physical image of bench axis $k$ | real | as axis $k$ | S3.2 |
| $a^{\rm nat}_k, b^{\rm nat}_k$ | the natural-unit bounds of axis $k$ (`BENCH_AXES`'s `L_k`, `U_k`) | reals, $a^{\rm nat}_k < b^{\rm nat}_k$ | as axis $k$ | S3.2 |
| $k$ | axis index, counted from 1 as in E0: axis $k$ is the code's index $k - 1$ (`BENCH_LABEL_IDX`, `BENCH_FREE_IDX` and the DSN's `label_axes` count from 0) | $k \in \{1, \dots, p\}$ | -- | S3.2 |
| $\mathcal{A}_{\rm lab}, \mathcal{A}_{\rm free}$ | label-carrying and label-irrelevant axes of the bench latent | disjoint index sets, $\mathcal{A}_{\rm lab} \cup \mathcal{A}_{\rm free} = \{1, \dots, p\}$; on the `bench` provider $\{1, \dots, 7\}$ and $\{8, 9, 10\}$ | -- | S3.2 |
| $\tau_{\rm ov}$ | within-class spread of the label axes (`--tau-ov`) | $\mathbb{R}_{>0}$ | dimensionless | S3.2 |
| $m_{c,k}$ | class centre of class $c$ on label axis $k$ | $(0, 1)$ | dimensionless | S3.2 |
| $\mathcal{TN}$ | the truncated normal law of plan eq. (7) | distribution on $(0, 1)$ | -- | S3.2 |
| $N_{\rm tr}$ | traces (wells) in ONE shard (`--n-traces`) | $\mathbb{N}$ | -- | S3.1 |
| $J$ | windows per trace (`--n-windows`) | $\mathbb{N}$ | -- | S3.1 |
| $N_{\rm neu}$ | neurons per trace (`--n-neurons`) | $\mathbb{N}$ | -- | S3.1 |
| $n_{\rm wd}$ | wells per donor (`--wells-per-donor`) | $\mathbb{N}$ | -- | S3.1 |
| $n_{\rm rl}$ | realisations per $\theta$ the builder requires (`--n-per-theta`) | $\mathbb{N}$ | -- | S3.1 |
| $N_{\rm don}$ | donors in one shard, $N_{\rm tr} / n_{\rm wd}$ | $\mathbb{N}$ | -- | S3.2 |
| $n_{\rm db}$ | donors per nuisance batch (`--donors-per-batch`) | $\mathbb{N}$ | -- | S3.1 |
| $S_{\rm sh}$ | shards in one arm of a bank | $\mathbb{N}$ | -- | S3.1 |
| $\mathsf{k}$ | shard index (`--shard-index`, `PBS_ARRAY_INDEX`); not the axis index $k$ | $\{0, \dots, S_{\rm sh} - 1\}$ | -- | S3.2 |
| $N^{\rm bank}_{\rm tr}$ | traces in one arm of a bank, $S_{\rm sh} N_{\rm tr}$ | $\mathbb{N}$ | -- | S3.3.3 |
| $s_{\rm seed}$ | the seed a command line is given (`--seed`, `SEED`) | $\mathbb{N}_0$ | -- | S3.2 |
| $s_{\rm base}$ | base seed of one shard, eq. (P7.2) | $\mathbb{N}_0$ | -- | S3.2 |
| $i$ | row (window) index: within a shard counted from 0 as the code does (S3.2); over the $J$ windows of one Stage 3c floor draw counted from 1, E0's convention (A.2) | $\{0, \dots, N_{\rm tr} J - 1\}$ in S3.2; $\{1, \dots, J\}$ in A.2 | -- | S3.2 |
| $\nu$ | one well's nuisance vector, $\nu^{(m)}$ its $m$-th component | $\mathbb{R}^{d_\nu}$ | mixed | S3.2 |
| $d_\nu$ | nuisance dimension, 5 (`N_COMPONENTS`) | $\mathbb{N}$ | -- | S3.2 |
| $m$ | nuisance-component index | $\{1, \dots, d_\nu\}$ | -- | S3.2 |
| $\nu_{\rm gain}, \nu_{\rm base}, \nu_{\rm thr}, \nu_{\rm drop}, \nu_{\rm drift}$ | the five components of $\nu$, in `NU_COMPONENTS` order | $\mathbb{R}$ each | dimensionless; units of $x$ for $\nu_{\rm base}$, $\nu_{\rm drift}$ | S3.2 |
| $\nu^{\rm bat}, \nu^{\rm don}, \nu^{\rm wel}$ | the batch, donor and well contributions, $\nu = \nu^{\rm bat} + \nu^{\rm don} + \nu^{\rm wel}$ | $\mathbb{R}^{d_\nu}$ each | as $\nu$ | S3.2 |
| $\sigma^{\rm bat}_m, \sigma^{\rm don}_m, \sigma^{\rm wel}_m$ | standard deviations of component $m$ of the three contributions | $\mathbb{R}_{\ge 0}$ | as component $m$ | S3.2 |
| $\sigma^{\rm tot}_m$ | total standard deviation of component $m$, eq. (P7.4) | $\mathbb{R}_{\ge 0}$ | as component $m$ | S3.2 |
| $\nu^{0}_{\rm drop}$ | the dropout logit at $\nu = 0$ (`dropout_logit0`) | $\mathbb{R}$ | -- | S3.2 |
| $\kappa_{\rm thr}$ | sensitivity of the detected rate to the threshold shift (`kappa`) | $\mathbb{R}_{\ge 0}$ | -- | S3.2 |
| $n_e$ | electrodes pooled per subregion; on the bench the notional count `n_electrodes` | $\mathbb{N}$ | -- | S3.2 |
| $n_{\rm alive}$ | electrodes left after the dropout component acts | $\{1, \dots, n_e\}$ | -- | S3.2 |
| $\mathcal{T}_\nu$ | the nuisance transform of plan eq. (8) for one well, eq. (P7.3) | a map on traces: for each window of the trace an invertible affine map $\mathbb{R}^{W} \to \mathbb{R}^{W}$ | -- | S3.2 |
| $\gamma_\nu, \beta_\nu$ | multiplicative and additive parts of the nuisance map | $\gamma_\nu \in \mathbb{R}_{>0}$; $\beta_\nu : [0, J T_{\rm win}) \to \mathbb{R}$ | --; units of $x$ | S3.2 |
| $t_{\rm abs}$ | time since the start of a trace | $[0, J T_{\rm win})$ | s | S3.2 |
| $T_{\rm drift}, T_{\rm gap}$ | periods of the nuisance drift and of the gap (c)'s drift | $\mathbb{R}_{>0}$ | s | S3.2 |
| $\pi$ | gap severity of bench arm R (`--pi`) | $[0, 1]$ | -- | S3.1 |
| $\mathcal{S}, \mathcal{R}$ | the bench's simulated and pseudo-real arms (`--arm S`, `--arm R`) | arms | -- | S3.3.2 |
| $p_0, p_\pi$ | the base and the perturbed bench generators | generative laws | -- | S3.2 |
| $\delta_{\rm shift}$ | displacement of each free axis's range under the gap (a) at $\pi = 1$, as a fraction of its width | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $A_{\rm gap}$ | the gap (c)'s drift amplitude at $\pi = 1$, relative to the trace's mean | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $f_{\rm con}$ | fraction of windows the gap (d) replaces at $\pi = 1$ | $[0, 1]$ | -- | S3.2 |
| $\mathcal{G}$ | a realisation: the per-well draw inside the likelihood, outside $\theta$ | a latent | -- | S3.2 |
| $\mathcal{U}$ | the row pairs `enumerate_donor_pairs` returns for the real arm | set of index pairs | -- | S3.3.3 |
| $N_{\rm pair}$ | its size, $\lvert \mathcal{U} \rvert$ | $\mathbb{N}$ | pairs | S3.3.3 |
| $r_{\rm same}, r_{\rm well}$ | fractions of $\mathcal{U}$ joining two rows of one shard's donor, and two wells of one shard's donor, eq. (P7.9) | $[0, 1]$ | -- | S3.3.3 |
| $\varsigma_{\rm tr}, \varsigma_{\rm sel}, \varsigma_{\rm rep}$ | the split's fractions $(0.7, 0.15, 0.15)$, applied to donor ids | $[0, 1]$ | -- | S3.3.2 |
| $L$ | held-out NLL of an arm on a split (computed level) | $\mathbb{R}$ | nats/row | S3.3.2 |
| $\ell_i$ | per-row held-out NLL | $\mathbb{R}$ | nats | S3.3.2 |
| $q_\omega$ | the conditional flow, $q_\omega(\theta \mid z)$ for each fixed $z$ | conditional density | (param units)$^{-d_\theta}$ | S3.3.2 |
| $h_\psi$ | the encoder, $h_\psi : \mathbb{R}^{W} \to S^{E-1}$ | map | -- | S3.5 |
| $z$ | the embedding of a window, $z = h_\psi(x)$ | $S^{E-1}$ | dimensionless | S3.5 |
| $E$ | embedding dimension | $\mathbb{N}$ | -- | S3.5 |
| $S^{E-1}$ | the unit sphere in $\mathbb{R}^{E}$ | set | -- | S3.5 |
| $S_{\rm mc}$ | posterior draws per well in the replicate term | $\mathbb{N}$ | draws | S3.5 |
| $\theta^{\rm fl}$ | Stage 3c's evaluation point: the bank's mean $\theta$ clipped to $[0.05, 0.95]$ | $(0, 1)^{d_\theta}$ | dimensionless | A.2 |
| $\theta^*$ | the true parameter of one well | $(0, 1)^{d_\theta}$ on the bench | dimensionless | A.2 |
| $n_{\rm fl}$ | draws of each floor (`--n-floor-draws`) | $\mathbb{N}$ | -- | A.2 |
| $S_{\rm post}$ | posterior draws per window in Stage 3c (`--n-post-draws`) | $\mathbb{N}$ | draws | A.2 |
| $m^{\rm fl}, \hat m^{\rm fl}$ | one floor draw's $J$-window mean of the flow's posterior means: exact (analytic level) and from $S_{\rm post}$ draws per window (computed level) | $\mathbb{R}^{d_\theta}$ | param units | A.2 |
| $C_i$ | the flow's posterior covariance $\mathrm{Cov}_{q_\omega}(\theta \mid x_i)$ for one fixed window $x_i$ (analytic level) | PSD $d_\theta \times d_\theta$ | (param units)$^2$ | A.2 |
| $\Sigma_{\rm fl}$ | the floor covariance the code computes (`ddof=1`, over the $n_{\rm fl}$ draws) | PSD $d_\theta \times d_\theta$ | (param units)$^2$ | A.2 |
| $J_\theta, J_\nu$ | Jacobians $\partial z / \partial \theta$ and $\partial z / \partial \nu$ by finite differences | $E \times d_\theta$, $E \times d_\nu$ | -- | A.2 |
| $h_{\rm fd}, n_{\rm fd}$ | the finite-difference step of $J_\theta$ (`--fd-step`) and the seeds both Jacobians are averaged over (`--fd-seeds`) | $\mathbb{R}_{>0}$; $\mathbb{N}$ | dimensionless; -- | A.2 |
| $a_m$ | aliasing coefficient of nuisance direction $m$, plan eq. (4) | $[0, 1]$ | -- | A.2 |
| $\mu_j$ | generalised eigenvalue of the stratification, plan eq. (5) | $\mathbb{R}_{\ge 0}$ | -- | A.2 |
| $j$ | direction index of that eigenproblem | $\{1, \dots, d_\theta\}$ | -- | A.2 |
| $n_{\rm ker}$ | kernel axes named by `--kernel-axes` | $\mathbb{N}_0$ | -- | A.2 |
| $n_{\rm probe}$ | rows of each bank the Stage 3b probes read (`--max-probe-rows`) | $\mathbb{N}$ | rows | A.1 |

### 1.1 Conventions

1. **Rows and indices.** E0's row index $i$ counts from 1. P7 counts a
   shard's rows from 0, as the code does (eq. (P7.1), S3.2), and a Stage 3c
   floor draw's $J$ windows from 1, as E0 does (eq. (P7.10)); the range is
   stated at each use [the shift by one is flagged here, not hidden]. A
   shard's traces are its wells and carry no index symbol: S3.2 names a
   trace by its stored `well` value. A component of a vector carries its index in parentheses,
   $\phi^{(k)}$, $\nu^{(m)}$ (E0 convention 9); a subscript $k$ on
   $\phi^{\rm nat}_k$, $a^{\rm nat}_k$, $b^{\rm nat}_k$ names the axis of a
   different object, the physical image.
2. **Shard-local and bank-global.** A field of a shard is *shard-local*
   when its values are meaningful only inside that shard -- two shards may
   use the same value for different units -- and *bank-global* when a value
   names one unit across every shard of a bank. At `834eb41` `donor`,
   `well`, `batch`, `subregion` and `window_idx` are shard-local and
   `realisation_id` is bank-global in practice (it hashes $s_{\rm base}$,
   S3.2 (a)); every consumer reads the identity fields as bank-global (F-at).
3. **Two senses of "arm"** (R6). The *bench arm* $\mathcal{S}$ or
   $\mathcal{R}$ is which generator a bank comes from and which label it
   withholds (`--arm`); the *training arm* (`A1`, `A0`, ..., `shuffled`) is
   which objective a Stage 3 run trains. P7 writes "bench arm S" and "arm
   A1" and never "the arm" bare where both are live.
4. **Two drifts** (R4). The nuisance drift (both bench arms, period
   $T_{\rm drift}$, amplitude $\nu_{\rm drift}$ in units of $x$) and the
   gap (c)'s drift (bench arm R only, period $T_{\rm gap}$, amplitude
   relative to the trace's mean) are two objects whose dataclass fields share
   the name `drift_period_s`; P0 Table F lists them as one knob with two
   defaults, which they are not.
5. **"Held-out"** (R6) means a row no training loss of the run read. A row of
   bench arm R is held out from the NPE loss by construction (its $\theta$ is
   withheld); F-aw shows rows that are held out in that sense and are
   nevertheless copies of training rows. [2026-10-05, E4 S3.7: in the first
   sense the rows of arm R are held out only for the arms that draw no real
   stream -- `A1`, `A0s`, `A2s`, `A_ref`, `shuffled`; `A0`, `A2` and `A3` read
   their $x$ and class in the metric loss and `A5` their $x$ in the replicate
   loss, so for these the pseudo-real endpoint is held out in $\theta$ alone.]
6. **Levels** (R8, E0 convention 4). The pair fractions $r_{\rm same}$,
   $r_{\rm well}$ of eq. (P7.9) are properties of the layout (analytic) and
   are also counted on a built bank (computed); the two agree exactly
   because the layout is deterministic, and the agreement is checked `[RAN]`.
   The probability that a bench arm-R shard raises, eq. (P7.8), is a
   probability over the prior's free-axis draw (analytic level); one build
   that raises is one realisation of it. The per-well probability of losing
   an electrode is a Gaussian tail (analytic); the count in a built bank is a
   realisation. $\mathbb{E}[\Sigma_{\rm fl}]$ in eq. (P7.10) is analytic,
   $\Sigma_{\rm fl}$ computed.
7. **Units of $x$** depend on the provider (S3.3.1); the nuisance's additive
   parts are in units of $x$ whatever the provider (eq. (P7.3)).
8. **Numbers.** Defaults are quoted with `[REPO file:line]`; everything a
   sandbox run produced is `[RAN]` and comes from the block of
   `tools/p7_numbers.py` named beside it (B1-B6).

## 2. Glossary

Ordered by first appearance in S3, since the terms build on each other.

- **Bank, shard, sidecar.** A bank is the training data of one bench arm as
  files: `shard_%04d.npz` arrays, one row per window, each with a JSON
  sidecar describing the generator and the specs (S3.2). S3.1.
- **Provider** (`--provider`). The function that turns $\phi$ and a seed
  into $J$ windows: `reference` (a deterministic test fixture from the
  Stage 1 smoke suite, not a simulator), `dsn` (the DSN's burst generator
  with its 6-axis `LatentSpec`), `bench` (the 10-axis bench burst generator
  with the DSN's IFR step). S3.3.1.
- **Contract digest.** A sha256 over 16 named sidecar fields; shards
  concatenate only when their digests agree. Provenance -- the shard index,
  the base seed, the command line, the build time -- is deliberately
  outside it. S3.2 (e).
- **Base seed.** The integer from which every draw of a shard is derived,
  eq. (P7.2). S3.2 (a).
- **Identity fields.** The integer arrays `donor`, `well`, `batch`,
  `subregion`, `window_idx` and `realisation_id` that say which unit a row
  belongs to; the split, the replicate pairs and Stage 3c's culture means
  read them. S3.2 (a), S3.3.3.
- **Truncated normal** `[textbook, from memory]`. The normal law restricted
  to an interval and renormalised; here to $(0, 1)$, so no mass sits on the
  boundary (plan eq. (7)). S3.2 (b).
- **Simplex centres.** Class centres placed at the vertices of a regular
  simplex, scaled to radius 0.30 about 0.5 on the label axes. S3.2 (b).
- **Scale convention.** What one sample of $x$ counts: the IFR summed over
  the simulated units (`sum_over_units`) or divided by their number
  (`per_unit_mean`). S3.3.1.
- **Gap, gap severity.** The deliberate misspecification of bench arm R:
  modes (a) range shift, (c) drift, (d) contamination, each scaled by $\pi$.
  The severity-scalar design is borrowed from Schmitt et al. (S6). S3.2 (d).
- **Range shift** (gap (a)). The free axes' physical ranges are displaced so
  that the same $\phi$ maps to physics the unshifted box cannot reach; the
  recorded $\theta$ stays in the box. S3.2 (d), S3.3.6.
- **Nuisance** (observation level). An affine, invertible map applied to
  every window after the dynamics, eq. (P7.3), with batch, donor and well
  contributions; "nuisance" here is the plan's technical sense (outside the
  likelihood), not "anything unwanted". S3.2 (c).
- **Realisation.** The per-well draw inside the likelihood and outside
  $\theta$, $\mathcal{G}$ of plan eq. (8), on the cohort a connectivity
  graph; on the three Stage 1 providers its stand-in is the stochastic pattern a
  seed produces at fixed $\phi$ (S3.3.7). S3.2 (a).
- **Nesting.** batch, then donor, then well, then subregion, then window
  (plan S2.6, Lever 2). S3.2 (a).
- **Pseudo-real endpoint.** The held-out NLL scored on bench arm R, whose
  $\theta$ is recorded and withheld from every loss (plan D10). S3.3.2.
- **PBS array, `-J`, `PBS_ARRAY_INDEX`, `-v`.** One job script submitted
  as many elements; each element receives its index and the `-v`
  variables. S3.4.
- **`set -u` (nounset).** A shell option that turns the expansion of an
  unset variable into an error that ends a non-interactive shell. S3.4.
- **Dry run.** A job element run with `DRYRUN=1` on a login node: prints
  the plan, trains nothing. S3.4.
- **Floor** (nuisance, realisation). The covariance of posterior means
  over draws of one factor at one fixed $\theta$ -- how much apparent
  mechanistic spread that factor alone manufactures (plan S2.6, Lever 3).
  A.2.
- **Common random numbers.** Holding every random input but one fixed across
  a floor's draws, so the floor measures that one input. A.2.
- **Aliasing coefficient, stratification, P10.** Stage 3c's other outputs:
  how much a nuisance direction is mistaken for $\theta$ (plan eq. (4)),
  the generalised eigenproblem of plan eq. (5), and the window-aggregation
  curve. A.2.
- **Kernel axes.** The axes of $\theta$ that parameterise the connectivity
  kernel: on the campaign bank three, `p0_conn`, `d0_conn`, `beta_conn`
  (`[KB]` usage v1.3 S6.2); none of the three Stage 1 providers has one. A.2.

## 3. Main body

### 3.1 Where the upstream knobs live, and which surface reaches them

This section establishes the surfaces a bank knob can be set on and which
of them the job script reaches, so that the later sections can say of each
knob whether an array can change it. The builder `build_latent_bank.py`
exposes 22 flags (`build_latent_bank.py:49-87`); its job script forwards 15 of them and gives
the other 7 -- `--donors-per-batch`, `--dsn-main-dir`, `--n-classes`,
`--n-label-axes`, `--n-latent`, `--n-neurons`, `--tau-ov` -- no variable
(`build_latent_bank.pbs:91-106` against the inventory's 22 CLI rows `[RAN]`
B2). An array therefore always builds at those seven defaults unless the
script is edited. Below the flags sit the spec classes the builder
instantiates with no flag at all -- every field of `NuisanceSpec`
(`build_latent_bank.py:186`), `GapSpec`'s four magnitudes (`:188-189` pass
only `modes` and `pi`), `RealisationSpec.share_within` (`:187`) -- and the
module constants.

**Table P7.1 -- the knobs P7 owns, every surface they appear on.** Status in
the Provenance model's sense: *configured* by a flag or a variable;
*configured by code* when only an edit changes it; *computed* when it
follows from other knobs. In the "where" column a bare line number is in
`stage1/build_latent_bank.py`, `.pbs:` is `stage1/jobs/build_latent_bank.pbs`,
"spec" is `stage1/latent_sbi_simulator.py` and "demo" is
`stage1/demo_classes_generate.py`; any other file is named.

| knob | where | type & domain | default(s) | reaches | Status | what it fixes downstream |
|---|---|---|---|---|---|---|
| `--provider` / `PROVIDER` | `build_latent_bank.py:76-77`; `.pbs:52` | `reference`, `dsn`, `bench` | `reference` on both | `make_provider_and_spec` `:90-154` | configured | $p$, $\mathcal{A}_{\rm lab}$, the generator, the scale of $x$ (S3.3.1) |
| `--arm` / `ARM` | `:54-55`; `.pbs:42` | `S`, `R` | `S` | the sidecar's `arm`, `theta_withheld` | configured | which bench arm; R withholds $\theta$ in the sidecar only (S3.3.2) |
| `--seed` / `SEED` | `:69`; `.pbs:53` | int $\ge 0$ | 0 | $s_{\rm base}$, `LatentSBISpec.seed` | configured | every draw of the shard, identically in both bench arms (F-aw) |
| `--shard-index` / `PBS_ARRAY_INDEX` | `:56`; `.pbs:63` | int $\ge 0$ | 0 | $s_{\rm base}$, the file name | configured | the shard's draws; never its identity fields (F-at) |
| `--n-traces` / `N_TRACES` | `:57-58`; `.pbs:44` | int, a multiple of $n_{\rm wd}$ | 64 | $N_{\rm tr}$ | configured | rows per shard $N_{\rm tr} J$; $N_{\rm don}$ |
| `--wells-per-donor` / `WELLS_PER_DONOR` | `:59`; `.pbs:45` | int $\ge n_{\rm rl}$ | 2 | $n_{\rm wd}$ | configured | wells sharing a $\theta$; the replicate pairs (P3) and the split's groups |
| `--donors-per-batch` | `:60` | int $\ge 1$ | 8 | $n_{\rm db}$ | configured; no job variable | which wells share the batch part of $\nu$ |
| `--n-windows` / `N_WINDOWS` | `:61`; `.pbs:46`; `LatentSBISpec.n_windows_per_trace` (`latent_sbi_simulator.py:76`); demo `:133` | int $\ge 1$ | 8; demo 4 | $J$ | configured | rows per trace; the within-well pairs (F-ab) |
| `--T-win` / `T_WIN` | `:62`; `.pbs:47`; spec `:76`; demo `:134` | float $> 0$ | 60.0 s; demo 15.0 s | $T_{\rm win}$ | configured | $W$, the encoder's input length (P1) |
| `--fs` / `FS` | `:63`; `.pbs:48`; spec `:76`; demo `:135` | float $> 0$ | 50.0 Hz | $f_s$ | configured; fixed at 50 on `dsn` (`:137-141`) | $W$; the IFR bin |
| `--n-neurons` | `:64`; spec `:77`; demo `:136` | int $\ge 1$ | 100 | $N_{\rm neu}$ | configured; no job variable | the population each trace pools; on `bench` the divisor of $x$ |
| `--n-latent` | `:65`; spec `:75` | int | 6 | $p$ on `reference` only | configured; no job variable | ignored by `dsn` (6) and `bench` (10) |
| `--n-label-axes` | `:66`; spec `label_idx` (0, 1, 2) `:75` | int | 3 | $\mathcal{A}_{\rm lab}$ on `reference` only | configured; no job variable | ignored by `dsn` (2) and `bench` (7) |
| `--n-classes` | `:67` (owner P2) | int $\ge 2$ | 3 | $C$, the class centres, `cls` | configured; no job variable | the DSN loss's class count (P2, F-t) |
| `--tau-ov` | `:68`; spec `:76`; demo `:137` | float $> 0$ | 0.10 | $\tau_{\rm ov}$ | configured; no job variable | the within-class spread of the label axes |
| `--pi` / `PI` | `:70-71`; `.pbs:49`; `GapSpec.pi` (`latent_gap.py:62`) | $[0, 1]$; 0 on bench arm S | 0.0 | $\pi$ | configured | bench arm R's distance from $p_0$ |
| `--gap-modes` / `GAP_MODES` | `:72-73`; `.pbs:50`; `GapSpec.modes` | subset of `GAP_MODES` | `range_shift` | which perturbations act | configured | provider-dependent (F-av) |
| `--n-per-theta` / `N_PER_THETA` | `:74-75`; `.pbs:51`; `RealisationSpec.n_per_theta` (`latent_realisation.py:59`) | int, $1 \le n_{\rm rl} \le n_{\rm wd}$ | 2 | a refusal, a post-build check, the digest (`realisation_spec`) | configured | refuses a bank with fewer realisations per $\theta$ |
| `--max-records` / `MAX_RECORDS` | `:83-84`; `.pbs:54` | int $\ge 0$ | 0 (no cap) | caps $N_{\rm tr}$ to a multiple of $n_{\rm wd}$ | configured | a probe run; outside the digest |
| `NuisanceSpec.scales` | `latent_nuisance.py:77-82` | $3 \times 5$, non-negative | Table P7.3 | $\sigma^{\rm bat}_m, \sigma^{\rm don}_m, \sigma^{\rm wel}_m$ | configured by code | the nuisance's strength, in units of $x$ |
| `NuisanceSpec.kappa`, `n_electrodes`, `drift_period_s`, `dropout_logit0` | `latent_nuisance.py:75-76` | floats; int | 0.5; 9; 600.0 s; -4.0 | $\kappa_{\rm thr}$, $n_e$, $T_{\rm drift}$, $\nu^{0}_{\rm drop}$ | configured by code | eq. (P7.3) |
| `GapSpec.shift_max`, `drift_amp_max`, `drift_period_s`, `contam_frac_max` | `latent_gap.py:62-64` | floats | 0.35; 0.5; 120.0 s; 0.20 | $\delta_{\rm shift}$, $A_{\rm gap}$, $T_{\rm gap}$, $f_{\rm con}$ | configured by code | the gap's magnitudes at $\pi = 1$ |
| `RealisationSpec.share_within` | `latent_realisation.py:59` | tuple of level names | `("subregion",)` | the realisation key | configured by code | one realisation per (donor, well, subregion) |
| `LatentSBISpec.class_centres` | `latent_sbi_simulator.py:87-88, 128-151` | $C \times \lvert \mathcal{A}_{\rm lab} \rvert$ in $(0, 1)$ | `None`: simplex, radius 0.30 | $m_{c,k}$ | computed from $C$ and $\mathcal{A}_{\rm lab}$ | where the classes sit |
| `BENCH_AXES` | `bench_burst_provider.py:82-94` | 10 (name, $a^{\rm nat}_k$, $b^{\rm nat}_k$) | Table P7.2 | the bench's physical images | configured by code | what $\theta$ means on the bench |
| `BENCH_LABEL_IDX`, `BENCH_FREE_IDX` | `bench_burst_provider.py:96-97` | index tuples (from 0) | (0..6), (7, 8, 9) | $\mathcal{A}_{\rm lab}$, $\mathcal{A}_{\rm free}$ on `bench` | configured by code | which axes carry the class; which the gap (a) moves |
| `BenchBurstProvider.PAD_BINS` | `bench_burst_provider.py:165` | int | 2 | extra IFR bins simulated | configured by code | removes a one-sample shortfall (S3.3.4) |
| `BenchBurstProvider(gaussian_window, overlap)` | `bench_burst_provider.py:167, 221-231` | float; `merge` or `clamp` | 0.04 s; `merge` | the IFR smoothing and the envelope policy | configured by code | not reachable from the builder |
| `GAP_MODES` | `latent_gap.py:39` | names | `range_shift`, `drift`, `contamination` | the admissible modes | configured by code | mode (b) removed (`latent_gap.py:24-30`) |
| `NU_COMPONENTS`, `N_COMPONENTS`, `NU_LEVELS` | `latent_nuisance.py:43-54` | names; 5; names | Table P7.3 | the layout of the `nu` array | configured by code | in the digest through `nuisance_spec` |

P0's Table F lists four of these knobs with two defaults each: `T_win`
(60.0 against the demonstration's 15.0), `n_windows` (8 against 4),
`gap_modes` (one value, two spellings) and `drift_period_s`, whose two
"defaults" are two different objects, $T_{\rm gap}$ = 120 s and
$T_{\rm drift}$ = 600 s (convention 4). The fifth P7 row there,
`n_post_draws`, is F-n (Appendix A.2).

**Table P7.2 -- `BENCH_AXES`** (`bench_burst_provider.py:82-97` `[REPO]`).
Units as the code's comments state them; the specification the table cites,
`STAGE_A_BENCH_GENERATOR_SPEC_v1` S4, is in neither the repository nor the
knowledge base (`find` over the clone, 2026-10-03; the KB's document list),
so the ranges' provenance cannot be checked here.

| code index ($k - 1$) | name | $a^{\rm nat}_k$ | $b^{\rm nat}_k$ | unit | set |
|---|---|---|---|---|---|
| 0 | `burst_rate` | 0.10 | 0.40 | bursts/s | $\mathcal{A}_{\rm lab}$ |
| 1 | `irregularity` | 0.30 | 0.90 | dimensionless | $\mathcal{A}_{\rm lab}$ |
| 2 | `ibi_cv` | 0.25 | 1.00 | dimensionless | $\mathcal{A}_{\rm lab}$ |
| 3 | `n_fragments` | 1.0 | 4.0 | count | $\mathcal{A}_{\rm lab}$ |
| 4 | `intraburst_rate` | 60.0 | 140.0 | spikes/s | $\mathcal{A}_{\rm lab}$ |
| 5 | `participation_mean` | 0.45 | 0.80 | probability | $\mathcal{A}_{\rm lab}$ |
| 6 | `burst_duration` | 0.15 | 0.35 | s | $\mathcal{A}_{\rm lab}$ |
| 7 | `fragment_duty` | 0.35 | 1.00 | dimensionless | $\mathcal{A}_{\rm free}$ |
| 8 | `participation_kappa` | 3.0 | 15.0 | dimensionless | $\mathcal{A}_{\rm free}$ |
| 9 | `background` | 0.01 | 0.06 | spikes/s/neuron | $\mathcal{A}_{\rm free}$ |

### 3.2 What one shard is: the build written out

Section 3.1 located the knobs; this section follows one call of the builder
from its flags to the bytes it writes, because every downstream property of
a bank -- what the split groups, what a replicate pair is, what the
pseudo-real endpoint scores -- is decided here, at write time, and is read
back without being re-derived.

**(a) The layout and the seeds.** The builder draws one $\theta$ and one
class per DONOR and one realisation per WELL (`build_latent_bank.py:211-255`):
the donors' $(c, \phi)$ come from one generator seeded with $s_{\rm base}$
(`:211-214`); the shard's traces are its wells, numbered from 0 in the
order the loop at `:238` visits them, and the trace numbered $\text{well}$
belongs to donor $\lfloor \text{well} / n_{\rm wd} \rfloor$ (`:216-218`);
its $J$ windows are consecutive rows. Writing the stored fields of row $i$
of shard $\mathsf{k}$ (counted from 0, convention 1),

$$\text{well} = \lfloor i / J \rfloor, \qquad \text{donor} = \lfloor i / (n_{\rm wd} J) \rfloor, \qquad \text{batch} = \lfloor i / (n_{\rm db}\, n_{\rm wd} J) \rfloor, \qquad \text{subregion} = 0, \tag{P7.1}$$

with `window_idx` the position of row $i$ within its trace -- none of them
depends on $\mathsf{k}$ `[REPO]` `:216-218, 234, 249-254`. The shard index
enters only through the base seed,

$$s_{\rm base} = 1000003\, s_{\rm seed} + \mathsf{k} \tag{P7.2}$$

(`:192`), from which three kinds of stream are derived: the donors' prior
draw (`default_rng(s_base)`, `:211`); the nuisance, one stream per (level,
key) hashed from $s_{\rm base}$ with the keys `b%d`, `d%d`, `w%d` built from
the shard-local integers of eq. (P7.1) (`:220-222`;
`latent_nuisance.py:119-131, 134-172`); and the realisation seed, the first
8 bytes of sha256 over $s_{\rm base}$ and the key
`donor=..|well=..|subregion=..` (`latent_realisation.py:72-96`), stored as
`realisation_id` (`latent_sbi_simulator.py:475-476, 501`). Each trace also
gets its own generator for the gap's draws, `default_rng` of
$s_{\rm base} + 7919\,(\text{well} + 1)$ (`build_latent_bank.py:244`). On
two default shards built `[RAN]` B3: 1024 rows, 64 distinct $\theta$ rows, 128 distinct
`realisation_id` -- and 32 distinct `donor`, 64 distinct `well`, 4 distinct
`batch` values, each `donor` and each `well` value naming two different
$\theta$. Plainly: inside one shard the identity fields are exact; across
shards the integers repeat while the draws behind them differ, because the
seed moves with $\mathsf{k}$ and the labels do not (S3.3.3).

**(b) The prior and the classes.** Plan eq. (7) as coded
(`latent_sbi_simulator.py:158-195`): the class is uniform on
$\{0, \dots, C-1\}$; on each label axis $k \in \mathcal{A}_{\rm lab}$,
$\phi^{(k)}$ follows $\mathcal{TN}$ on $(0, 1)$ about $m_{c,k}$ with scale
$\tau_{\rm ov}$ (scipy's `truncnorm`, standardised bounds
$(0 - m_{c,k})/\tau_{\rm ov}$ and $(1 - m_{c,k})/\tau_{\rm ov}$, `:158-162`);
on each free axis $\phi^{(k)}$ is uniform on $(0, 1)$; the result is clipped
one float64 machine epsilon inside each end of $(0, 1)$ and asserted strictly
inside the box (`:190-194`). The class centres are `simplex_centres`: the vertices
of a regular $(C-1)$-simplex placed on the first
$\min(\lvert \mathcal{A}_{\rm lab} \rvert, C - 1)$ label axes, the remaining label axes
cycling those coordinates so that no label axis is constant across classes,
scaled to radius 0.30 about 0.5 (`:128-151`); [corrected 2026-10-04, E2: the
vertices are not those of a regular simplex for $C \ge 3$. The construction
starts from the $C - 1$ unit coordinate vectors of $\mathbb{R}^{C-1}$ and
the origin, whose mutual distances are $\sqrt{2}$ and 1, and only
translates and rescales them; at $C = 3$ on the seven label axes of the
`bench` provider the centred centres' pairwise cosines are $-0.803$,
$-0.434$ and $-0.189$ instead of $-0.5$ each, while the DSN's
`_class_center_vectors`, used by the `dsn` provider, gives $-0.5$ each
`[RAN]` (E2 S5, `tools/e2_numbers.py` B4; finding F-ba)] on the `dsn`
provider the DSN's own `_class_center_vectors`, clipped to
$[10^{-6}, 1 - 10^{-6}]$
(`:385-412`). The classes, the centres and the prior's shape are E6's
subject; here they matter because `--n-classes`, `--n-label-axes` and
`--tau-ov` set them, and because the free axes' uniform draw is what
eq. (P7.8) below integrates over.

**(c) The observation chain.** `simulate_windows`
(`latent_sbi_simulator.py:440-503`) applies, in this order: the gap (a) as
axis-range overrides handed to the provider; the provider, which returns
$J$ non-negative windows from $\phi$ and the realisation seed (the
recorded $\theta$ is $\phi$ itself, never shifted, `:467-470`); the gap's
trace-layer modes (c) and (d), on the whole trace at once on the absolute
time grid (`:487-494`); and last the nuisance, window by window on the same
grid (`:496-499`). The order is the plan's: the nuisance is an observation
map and is applied after everything that acts on the dynamics (`:445-455`).
For one well with nuisance vector $\nu$, and for each fixed window $x$ of
its trace, the nuisance transform $\mathcal{T}_\nu$ of plan eq. (8) acts on
that window as an affine map $\mathbb{R}^{W} \to \mathbb{R}^{W}$, set by the
well's $\nu$ and by the window's place on the time grid:

$$x^{\rm obs} = \mathcal{T}_\nu[x] = \gamma_\nu\, x + \beta_\nu(t_{\rm abs}), \qquad \gamma_\nu = \exp(\nu_{\rm gain} - \kappa_{\rm thr}\, \nu_{\rm thr})\, \frac{n_{\rm alive}}{n_e}, \qquad n_{\rm alive} = \max\Big(1, \mathrm{round}\Big(\frac{n_e}{1 + \exp(\nu^{0}_{\rm drop} + \nu_{\rm drop})}\Big)\Big), \tag{P7.3}$$

with $\beta_\nu(t_{\rm abs})$ equal to $\nu_{\rm base}$ plus a sinusoid of
amplitude $\nu_{\rm drift}$ and period $T_{\rm drift}$ in $t_{\rm abs}$
(`latent_nuisance.py:175-206`; the fraction of electrodes kept, one minus
the logistic function the code evaluates, is written as
$1/(1 + \exp(\cdot))$ above). The
well's $\nu$ is the sum of three independent contributions, batch, donor
and well, each component a zero-mean normal draw with the standard deviation
of Table P7.3, shared exactly by every well of one batch (resp. one donor)
(`:134-172`); by the variance of a sum of independent terms
`[textbook, from memory]`, for each component $m$,

$$(\sigma^{\rm tot}_m)^2 = \mathrm{Var}(\nu^{(m)}) = (\sigma^{\rm bat}_m)^2 + (\sigma^{\rm don}_m)^2 + (\sigma^{\rm wel}_m)^2. \tag{P7.4}$$

**Table P7.3 -- `NuisanceSpec()`** (`latent_nuisance.py:43-54, 75-105`
`[REPO]`; the derived columns `[RAN]` B5).

| $m$ | `NU_COMPONENTS` | $\sigma^{\rm bat}_m$ | $\sigma^{\rm don}_m$ | $\sigma^{\rm wel}_m$ | $\sigma^{\rm tot}_m$ | batch share of the variance |
|---|---|---|---|---|---|---|
| 1 | `log_gain` ($\nu_{\rm gain}$) | 0.10 | 0.05 | 0.05 | 0.1225 | 2/3 |
| 2 | `baseline` ($\nu_{\rm base}$, units of $x$) | 0.02 | 0.01 | 0.01 | 0.0245 | 2/3 |
| 3 | `dthr` ($\nu_{\rm thr}$) | 0.10 | 0.05 | 0.05 | 0.1225 | 2/3 |
| 4 | `dropout_logit` ($\nu_{\rm drop}$) | 0.30 | 0.15 | 0.15 | 0.3674 | 2/3 |
| 5 | `drift_amp` ($\nu_{\rm drift}$, units of $x$) | 0.02 | 0.01 | 0.01 | 0.0245 | 2/3 |

with $\kappa_{\rm thr} = 0.5$, $n_e = 9$, $T_{\rm drift} = 600$ s and
$\nu^{0}_{\rm drop} = -4.0$. At $\nu = 0$ the map is the identity, gain
1.000000 and additive part 0 `[RAN]` B5 -- the property the
`[CORRECTION]` of `latent_nuisance.py:97-104` restored. The dropout
component is a step function of $\nu_{\rm drop}$: from eq. (P7.3), for each
fixed $\nu^{0}_{\rm drop}$ and $n_e \ge 1$,

$$n_{\rm alive} = n_e \ \text{ for } \ \nu_{\rm drop} < \ln \frac{1}{2 n_e - 1} - \nu^{0}_{\rm drop}, \qquad n_{\rm alive} < n_e \ \text{ for } \ \nu_{\rm drop} > \ln \frac{1}{2 n_e - 1} - \nu^{0}_{\rm drop} \tag{P7.5}$$

(at the threshold itself the rounding decides: Python rounds a half to the
even integer, so at the defaults 8.5 electrodes round to 8), which at the
defaults is a threshold of $1.1668$, i.e.
$3.176\, \sigma^{\rm tot}_4$: under the Gaussian draw of eq. (P7.4) a well loses an
electrode with probability $7.5 \times 10^{-4}$, a second one beyond
$6.51\, \sigma^{\rm tot}_4$ with probability $3.9 \times 10^{-11}$ (analytic
level), and the gain steps from 1 to $8/9$ at the threshold `[RAN]` B5. In
the two-shard bank of B3, none of the 128 wells lost an electrode `[RAN]`
(one realisation of that probability). So at the default scales the
dropout component, which the plan lists among the cohort's nuisances,
almost never acts on a bench bank. The drift period, 600 s, is longer than
a trace, $J T_{\rm win} = 480$ s at the defaults `[RAN]` B1 (B5), so a
trace sees less than one period of the nuisance drift.

**(d) The gap.** `GAP_MODES` holds three modes (`latent_gap.py:39`); mode
(b), heavy-tailed burst durations, was removed -- "REMOVED 2026-09-09" in the
module's docstring, commit `f541b29` of 2026-09-11 -- because its override keys
were never bound to a generator field (`:24-30`), and `GapSpec` refuses any
name outside `GAP_MODES` and any $\pi$ outside $[0, 1]$ (`:65-71`). Every
mode is the identity at $\pi = 0$ (`GapSpec.active`, `:77-80`), which is why
the builder refuses $\pi \ne 0$ on bench arm S (`build_latent_bank.py:160-161`).

- *(a) Range shift* acts at the axis layer: the provider receives
  `free_axis_range_shift` $= \pi\, \delta_{\rm shift}$ (`latent_gap.py:97-123`)
  and displaces the physical range of each free axis by that fraction of the
  range's width, so that for each $k \in \mathcal{A}_{\rm free}$ the bench
  provider maps

  $$\phi^{\rm nat}_k = a^{\rm nat}_k + (\phi^{(k)} + \pi\, \delta_{\rm shift})\,(b^{\rm nat}_k - a^{\rm nat}_k) \tag{P7.6}$$

  (`bench_burst_provider.py:100-131`, in the convex-combination form, which
  equals eq. (P7.6) up to rounding); the `dsn` provider does the same through
  the DSN's own axis ranges (`latent_sbi_simulator.py:314-343`); the
  `reference` fixture adds $\pi \delta_{\rm shift}$ to $\phi$ on EVERY axis it
  reads, label axes included (`smoke_test_latent_sbi.py:78-81`; bitwise equal
  to the unshifted fixture at $\phi + 0.2$, `[RAN]` B4). One flag, three
  perturbations (F-av).
- *(c) Drift* acts at the trace layer: one random phase per trace, a
  sinusoid of period $T_{\rm gap} = 120$ s on the absolute time grid,
  amplitude $\pi A_{\rm gap}$ times the trace's mean, added before the
  nuisance (`latent_gap.py:162-170`). Unlike the nuisance drift (convention
  4) it scales with the trace, so its relative size does not depend on the
  provider.
- *(d) Contamination* replaces $\lfloor \pi f_{\rm con} J \rfloor$ windows of
  each trace, plus one more with probability equal to the remainder: each
  chosen window is multiplied by a factor drawn uniformly on $(2, 5)$, given
  Gaussian noise with the window's own standard deviation, floored at 0, and
  flagged in `contaminated` (`latent_gap.py:172-189`). At $\pi = 0.5$ the expected count is 0.8 windows per trace,
  51.2 per default shard; one bench shard had 53 `[RAN]` B4.

**(e) The sidecar and its digest.** The sidecar carries 16 contract fields
(`latent_bank.py:42-59`): `schema_version`, `param_names`, `bounds_theta`,
`coord`, `fs`, `w_size`, `T_win`, `W`, `scale_convention`, `latent_spec`,
`nuisance_spec`, `realisation_spec`, `gap_spec`, `generator_sha256`,
`theta_withheld`, `arm` `[RAN]` B3; `contract_digest` is a sha256 over them
(`:66-78`), and `provenance` -- shard index, $s_{\rm base}$, build time,
command line -- is deliberately outside it (`build_latent_bank.py:270-274`).
`generator_sha256` is the provider's tag (`reference-fixture`, `dsn`,
`bench`) and a sha256 of its class's source (`:268`;
`latent_sbi_simulator.py:415-426`) -- the class's own source only, not the
modules it calls (`hpc/dsn/generate_burst_data.py` for `dsn` and `bench`,
`bench_burst_generator.py` for `bench`), so an edit there leaves the digest
unchanged `[REPO]`. Two consequences, both `[RAN]` B3: a shard built with `--max-records 2` and one built with
`--wells-per-donor 4` have the digest of a default shard, so the digest
certifies the generator and the specs and says nothing about a shard's
size or nesting -- `concat_shards` (`latent_bank.py:184-204`) will merge
shards of different layouts; and bench arms S and R never share a digest
(`arm`, `theta_withheld` are contract fields), so they never concatenate.

**(f) What the builder checks, and what it does not.** It refuses bench arm
S with $\pi \ne 0$ (`build_latent_bank.py:160-161`), a trace count that is
not a multiple of $n_{\rm wd}$ (`build_latent_bank.py:162-163`), and
$n_{\rm wd} < n_{\rm rl}$ (`build_latent_bank.py:171-176`); after the build
it counts distinct realisations per $\theta$ on the bytes it is about to
write and refuses fewer than $n_{\rm rl}$ (`build_latent_bank.py:276-284`); writes are
atomic, a temporary file then a rename (`latent_bank.py:145-168`). Nothing
checks that identity fields are distinct across shards (F-at), that the two
bench arms draw differently (F-aw), or that the bench provider's parameters
stay in their domain under the gap (F-av; the provider raises, S3.3.6).

### 3.3 The knobs, one by one

Section 3.2 wrote the build as one chain; this section takes the knobs of
Table P7.1 in the order a bank's design meets them and says, for each, what
it changes in that chain and what a run downstream inherits from it.

#### 3.3.1 `--provider` and the scale of $x$

Regardless of every other flag, if the provider changes then the meaning of
$\theta$, its dimension and the unit of $x$ change with it.

**Table P7.4 -- the three providers** (`build_latent_bank.py:90-154`
`[REPO]`; the last three columns `[RAN]` B4 at $\phi = 0.5$ on every axis,
$J = 8$, $W = 3000$, $N_{\rm neu} = 100$, realisation seed 12345).

| `--provider` | $p$ | $\mathcal{A}_{\rm lab}$ | scale of $x$ | what the realisation seed draws | mean of $x$ | sd of $x$ | $\sigma^{\rm tot}_2$ / mean |
|---|---|---|---|---|---|---|---|
| `reference` (`smoke_test_latent_sbi.py:65-106`) | `--n-latent` (6) | the first `--n-label-axes`: code indices 0, 1, 2, i.e. $k = 1, 2, 3$ | `sum_over_units` | the per-neuron weights (its stand-in for $\mathcal{G}$), the burst counts and times, the noise (`:91-104`) | 9.0801 | 5.0974 | 0.27 % |
| `dsn` (`latent_sbi_simulator.py:276-373`) | 6, the DSN `LatentSpec` axes | code indices (0, 1), i.e. $k = 1, 2$, the DSN default (`dsn/latent_burst_generator.py:253`) | `sum_over_units` | the spike times, bursts included (`latent_sbi_simulator.py:358-359`) | 8.4138 | 30.2672 | 0.29 % |
| `bench` (`bench_burst_provider.py:134-218`) | 10, `BENCH_AXES` | code indices 0-6, i.e. $k = 1, \dots, 7$ | `per_unit_mean` | the whole spike pattern at fixed $\phi$ (`bench_burst_provider.py:137-143`) | 0.06742 | 0.22485 | 36.33 % |

The bench row's mean agrees with the generator's own expected rate,
`expected_mfr` at $\phi = 0.5$ = 3.5309 spikes/s/neuron, i.e. 0.07062 counts
per bin per neuron at $f_s = 50$ Hz `[RAN]` B4. The `reference` provider is
the job's default (`build_latent_bank.pbs:52`), as the usage guide warns
(`[KB]` usage v1.3 S3.2): an array submitted without `PROVIDER` builds a
fixture bank. The last column is the usage guide's open item (S3.5 there)
measured: the nuisance's additive parts, 0.0245 in units of $x$ whatever
the provider (Table P7.3), are a quarter of a percent of a sum-scale window
and a third of a per-unit-mean one. The guide's own figures -- "~0.02-0.06
counts/bin at mid-box rates", "~0.6 %" on the sum scale -- are not
reproduced at $\phi = 0.5$ exactly (0.067 and 0.27-0.29 % here); the
conclusion they support -- effectively off on the sum scale, order one on
the per-unit scale -- holds at this point too (R7: the borrowed range
carries "at mid-box rates", and one point of the box is not a range). The provider's tag is in the
sidecar's `generator_sha256` and nothing downstream reads it (F-au).

#### 3.3.2 `--arm` and `--seed`: the two bench arms and their draws (F-aw)

The plan's bench is two arms from one generator: $\mathcal{S}$ from $p_0$
with $\theta$ available, $\mathcal{R}$ from $p_\pi$ with $\theta$ recorded
and withheld, and the pseudo-real held-out NLL on $\mathcal{R}$ as the
primary endpoint (plan S4.0, D10). One would read "two arms from one
generator" as two independent samples of one law. In the code, `--arm`
sets two sidecar fields and the gap; it does not enter the base seed,
eq. (P7.2). Regardless of the arm, if two shards share $s_{\rm seed}$ and
$\mathsf{k}$, then they share every draw: built with the job's defaults
(`SEED` 0 for both arrays), bench arm R's `theta`, `cls`, `nu`,
`realisation_id`, `donor` and `well` equal bench arm S's field for field,
and at $\pi = 0$ so does every byte of `x` `[RAN]` B3. At $\pi > 0$ the same
$\theta$, class, nuisance and realisation seed are pushed through the
perturbed generator, so each arm-R trace is a perturbed copy of an arm-S
trace.

The consequence reaches the primary endpoint. The runner scores the
pseudo-real held-out NLL $L$ -- the mean of
$\ell_i = -\log q_\omega(\theta_i \mid z_i)$ over the rows scored, with
$\theta_i$ the recorded $\theta$ of row $i$ and $z_i$ the embedding $h_\psi$
gives the row's stored window, an $x^{\rm obs}$ -- on EVERY row of the real bank
(`run_joint_arms.py:551-558`), while the flow is trained on
the training split of the simulated bank, grouped by `donor`
(`run_joint_arms.py:369`). With identical layouts and seeds, the arm-R rows whose
donor falls in the training split are copies of training rows: at $\pi = 0$ and
seed 0, the pseudo-real endpoint scores copies of training rows in 68.75 %
of its rows, of selection rows (the early-stopping split) in 15.62 %, and of
report rows in 15.62 % `[RAN]` B3 -- the shares of
$(\varsigma_{\rm tr}, \varsigma_{\rm sel}, \varsigma_{\rm rep})$ after the
split's rounding (22, 5 and 5 of 32 donor ids). The smoke test S7, which
checks that the two arms are indistinguishable at $\pi = 0$, draws them with
different seeds, 100 and 200 (`smoke_test_latent_sbi.py:549-550`), and so
never meets this. Until it is changed, the operational rule is to build
bench arm R with a `SEED` different from bench arm S's; `--seed` also
enters `LatentSBISpec.seed` and so the digest, which the two arms never
share anyway (S3.2 (e)). Which repair is wanted -- the arm in the seed, or a
refusal in the runner when the two banks' `provenance.base_seed` coincide --
is not decided here.

#### 3.3.3 `--n-traces`, `--wells-per-donor`, `--donors-per-batch`, `--shard-index`: the bank's size and its identity fields (F-at)

At the job's defaults one shard holds $N_{\rm tr} = 64$ traces of
$N_{\rm don} = 32$ donors with $n_{\rm wd} = 2$ wells each, 4 nuisance batches
of $n_{\rm db} = 8$ donors, and $N_{\rm tr} J = 512$ rows, 16 per donor
`[RAN]` B1; `x` takes 6.144 MB as float32, so the usage guide's array of 32
shards (`qsub -J 0-31`, `[KB]` usage v1.3 S3.2) holds
$N^{\rm bank}_{\rm tr} = 2048$ traces and 196.6 MB of windows, and the plan's
iteration bank of 32 000 windows 384.0 MB `[RAN]` B1. The plan's
$N^{\rm bank}_{\rm tr} = 4000$ (S4.4) is 62.5 shards of 64; 40 shards of 100,
50 of 80 or 125 of 32 reach it with an even trace count `[RAN]` B1.

Regardless of how many shards a bank has, the identity fields of
eq. (P7.1) take the same 32 donor, 64 well and 4 batch values, because they
are computed from the row position inside the shard; `concat_shards`
concatenates the arrays and offsets nothing (`latent_bank.py:184-204`). Every
consumer reads them as bank-global:

- **The replicate stream** pairs real-arm rows by `donor` alone
  (`joint_batches.py:37-58, 150-164`). With $S_{\rm sh}$ shards of one layout,
  one `donor` value gathers $S_{\rm sh}$ donors with $S_{\rm sh}$ different
  $\theta$, and the fractions of the enumerated pairs $\mathcal{U}$ that
  share $\theta$, and that join two wells of one donor -- the plan's
  replicate pair -- are, for each fixed layout,

  $$r_{\rm same} = \frac{n_{\rm wd} J - 1}{S_{\rm sh}\, n_{\rm wd} J - 1}, \qquad r_{\rm well} = \frac{(n_{\rm wd} - 1)\, J}{S_{\rm sh}\, n_{\rm wd} J - 1}. \tag{P7.9}$$

  At the defaults: $S_{\rm sh} = 1$, 1 and 0.5333 (P3's F-ab, 3840 pairs);
  $S_{\rm sh} = 2$, 0.4839 and 0.2581 of 15 872 pairs -- counted on a built
  two-shard bank through the code's own `enumerate_donor_pairs`, equal to the
  closed form `[RAN]` B1, B3; $S_{\rm sh} = 32$, 0.0294 and 0.0157 of
  4 186 112 pairs, so 97.1 % of the pairs the replicate loss can draw join
  two different $\theta$ `[RAN]` B1, B3b. The loss's target is derived for two
  recordings of one $\theta^*$ (P3 S3.2; plan S2.5b); for a pair of different
  $\theta$ the two-sided loss pulls the two posteriors toward each other, i.e.
  toward erasing the difference the flow should report [reasoning]. Arm A5
  trains on this stream; campaigns `S-A5` and `S-A25` search over it.
- **The replicate diagnostic** of every Stage 3 run with a real bank takes
  the first 64 enumerated pairs (`run_joint_arms.py:610-617`), all of them in
  `donor` 0: on two shards 32 share $\theta$ (16 of them within one well) and
  32 do not; on 32 shards 15 share $\theta$ (7 within a well, 8 across) and
  49 do not `[RAN]` B3, B3b. Arm A1, which never trains on pairs, still
  prints and records this diagnostic.
- **The split** groups the simulated bank by `donor` (`:369`, `grouped_split`
  `:118-143`): 32 groups whatever $S_{\rm sh}$ -- 22, 5 and 5 donor ids, i.e.
  the report split is 5 groups of $S_{\rm sh}$ donors each, where bank-global
  ids would give 717, 154 and 153 donors at $S_{\rm sh} = 32$ `[RAN]` B3b.
  Nothing leaks, since a `donor` value's rows stay together; what changes is
  the granularity of the split and of the cluster bootstrap that groups by
  the same field (Stage 3b, `domain_objective.py:122, 144`). And the split's
  hash, an input of P6's trial identity, is the same for 1, 2 and 32 shards,
  `930462872c62db6a...` `[RAN]` B3, B3b: it cannot tell banks of different
  sizes apart, and neither can the digest (S3.2 (e)).
- **Stage 3c's culture means** group the simulated bank by `well`
  (`run_stage3c.py:376-388`): 64 groups for the 128 wells of two shards,
  each averaging two unrelated wells, 46 of the 64 mixing two classes while
  carrying the first row's label `[RAN]` B3 -- the input of the
  stratification gate (Appendix A.2).

Plainly: the bank's identity fields are labels of positions inside a file,
and every reader takes them for names of donors and wells. Two repairs are
available and neither is chosen here: make the builder write bank-global
integers (for instance $\mathsf{k} N_{\rm don}$ plus the shard-local donor),
or make `concat_shards` offset each shard's fields; the second also covers
banks whose shards differ in layout, which the digest admits (S3.2 (e)).
Until one is chosen, a bank of ONE shard per bench arm -- every trace in one
`--n-traces`, one job rather than an array -- keeps the fields bank-global
by construction; at the plan's 4000 traces that shard holds 32 000 rows and
384.0 MB of windows `[RAN]` B1, within the job's 8 GB [reasoning; no shard
of that size was built here]. The
Giulia plan's window exporter B5, not written, will write `donor`, `well`
and `batch` "from campaign / topology / iteration" into one shard per
(campaign, task) (`[KB]` Giulia plan v1.7 S3, B5): it inherits the same
requirement, bank-global values, because the same consumers will read
them.

`--donors-per-batch` (no job variable) sets which wells share
$\nu^{\rm bat}$; with 32 donors per shard and 8 per batch, every shard has 4
batches of 16 wells. The batch keys are shard-local too, but the nuisance
streams are seeded with $s_{\rm base}$, so batch 0 of two shards are two
independent draws under one label; no consumer reads `batch` at `834eb41`
(a search of `stage2`-`stage4`, S3.6).

#### 3.3.4 `--n-windows`, `--T-win`, `--fs`: the window grid

$W = \mathrm{round}(T_{\rm win} f_s)$ (`latent_sbi_simulator.py:109-111`);
3000 samples at the defaults, and one trace simulates its $J$ windows back
to back, $J W = 24\,000$ samples, 480.0 s `[RAN]` B1. The bench provider
simulates $J W + 2$ bins, 480.04 s, and keeps the first $J W$, because the
IFR step's bin count can fall one short of the target in floating point
(`PAD_BINS`, `bench_burst_provider.py:160-165, 186-218`); the `dsn` provider
has no pad and raises on a short trace instead (`latent_sbi_simulator.py:365-370`),
which a default `dsn` shard does not trigger: 512 rows of $W = 3000$
`[RAN]` B4. On the `dsn` provider `--fs`
must equal the DSN's $1/\Delta t$ = 50 Hz or the builder exits
(`build_latent_bank.py:137-141`); `--T-win` is free on all three. $W$ is
the encoder's input length (P1), the length `probe_dsn_runtime.py --window`
must be given (default 3000, the bench's $W$; the Giulia windows are
$W = 18\,000$, `[KB]` Giulia plan v1.7 D-041), and $J$ sets the within-well
share of the replicate pairs (F-ab; eq. (P7.9) at $S_{\rm sh} = 1$).

#### 3.3.5 `--n-latent`, `--n-label-axes`, `--n-classes`, `--tau-ov`: the prior

`--n-latent` and `--n-label-axes` act only on the `reference` provider; the
`dsn` branch takes six axes and two label axes from the DSN `LatentSpec`
and the `bench` branch ten and seven from `BENCH_AXES` (`build_latent_bank.py:112-143`;
usage v1.3 S3.1). `--n-classes` sets $C$ on all three and `--tau-ov`
$\tau_{\rm ov}$ (on `dsn` as the DSN's `class_overlap`); $\tau_{\rm ov}$ must
be positive (`latent_sbi_simulator.py:96-98`) so that within-class spread
exists by construction (plan S4.0). The draws are clipped to one machine
epsilon inside the box (S3.2 (b)), because `transform_to_unconstrained`
maps a boundary point to infinity (`:6-9`). None of the four is a job
variable, so every array bank has $C = 3$, $\tau_{\rm ov} = 0.10$ and the
provider's own axis set (B2). The bank's class count is what the runner
uses for the DSN loss (`run_joint_arms.py:402`), from the SIMULATED bank's
`cls` (F-t, P2): on the bench every provider writes the prior's class
$0..C-1$ in both arms, so the counts agree by construction; the Giulia
simulated bank is planned with `cls = -1` (`[KB]` Giulia plan B5), where the
count from the simulated bank would be 1 and an arm with a DSN term on the
real labels would see three classes; arm A1 has no DSN term.

#### 3.3.6 `--pi` and `--gap-modes`: the gap on each provider (F-av)

Regardless of the provider, if $\pi > 0$ and the mode is `range_shift`, then
the provider is asked to move the free axes' physical ranges; what follows
depends on which provider receives the request:

- **`bench`** applies eq. (P7.6) to its three free axes. `fragment_duty`'s
  range ends exactly at 1 ($b^{\rm nat}_8 = 1.00$, axis $k = 8$, code index 7, Table P7.2) and
  `BenchBurstParams` refuses a duty above 1 (`bench_burst_generator.py:112-113`),
  so for each fixed $\pi$

  $$\phi^{\rm nat}_8 > 1 \iff \phi^{(8)} > 1 - \pi\, \delta_{\rm shift}, \tag{P7.7}$$

  checked on both sides of the boundary for $\pi$ = 0.05, 0.1, 0.25, 0.5, 1
  `[RAN]` B4. The free axes are uniform on $(0, 1)$ and drawn once per donor
  (S3.2 (b)), so a shard of $N_{\rm don}$ donors raises with probability, over
  the prior,

  $$1 - (1 - \pi\, \delta_{\rm shift})^{N_{\rm don}} \tag{P7.8}$$

  -- 0.43, 0.68, 0.95, 0.998 and 1.0000 at those five severities and
  $N_{\rm don} = 32$ `[RAN]` B4. The raise is an exception inside the
  provider: the build stops and writes no shard. `build_latent_bank.py
  --provider bench --arm R --pi 0.1` at the defaults exits with `ValueError:
  duty must be in (0, 1], got 1.01886740128649` `[RAN]` B4. The smoke test
  of the shift (E2) uses $\phi = 0.25$ everywhere and never reaches the edge
  (`smoke_test_bench_provider.py:95-115`). The bench provider with the
  trace-layer modes alone builds: `--pi 0.5 --gap-modes drift,contamination`,
  512 rows, 53 contaminated windows `[RAN]` B4.
- **`dsn`** shifts the free axes through the DSN's `resolve_axes`, whose
  validation runs on the new ranges (`latent_sbi_simulator.py:314-343`); at
  $\pi = 1$ with $\phi = 0.999$ on every axis it accepted the call, and a
  bench arm-R shard built with this provider at $\pi = 0.5$, the defaults
  otherwise, has 512 rows `[RAN]` B4.
- **`reference`** adds $\pi \delta_{\rm shift}$ to $\phi$ itself on every
  axis it reads (S3.2 (d)), label axes included, through logistic maps: a
  different perturbation from the other two, under the same flag.

So the plan's minimum viable bench design -- $\pi = 0$ plus one moderate
level of gap (a) (plan S8, D11) -- cannot be built on the bench provider
at `834eb41`, and on the default provider it is not the documented
perturbation. Repairs (clipping or reflecting a shifted image, shifting
duty's range inward, excluding axes whose range touches a hard limit) change
what the gap means and are E6's and the user's call.

#### 3.3.7 `--n-per-theta` and `RealisationSpec`: the realisation count

The realisation key is (donor, well) plus the subregion when
`share_within` names it (`latent_realisation.py:72-83`); every bench well has
one subregion (`subregion` = 0, `build_latent_bank.py:234, 242`), so the
realisation is per well and two wells of one donor draw two
(`[KB]` usage v1.3 S3.3; smoke test S10). With $\theta$ drawn per donor, the
number of realisations per $\theta$ is $n_{\rm wd}$ exactly, which is why
the builder refuses $n_{\rm wd} < n_{\rm rl}$ before building and counts on
the written bytes after (`build_latent_bank.py:171-176, 276-284`); $n_{\rm rl} = 1$ reproduces the
one-realisation-per-$\theta$ structure that D17 is about (`latent_realisation.py:53-56`).
What stands in for the realised graph $\mathcal{G}$ differs by provider
(Table P7.4): the `dsn` and `bench` providers have no connectivity graph, so
a realisation there is the stochastic pattern a seed produces at fixed
$\phi$ (`bench_burst_provider.py:140-143`); the fixture draws per-neuron
weights.
D17 was closed as option (c), accept and record, so Stage 3c's audit of the
count is informational (`run_stage3c.py:87-130`).

#### 3.3.8 `NuisanceSpec`: five components at three levels (no flag)

Table P7.3 and eqs. (P7.3)-(P7.5) carry the content. Three properties
matter downstream. First, the additive parts are absolute (units of $x$), so
the same spec is a quarter-percent perturbation on a sum-scale bank and a
third of the signal on a per-unit bank (Table P7.4); the usage guide records
this as open, not retuned (`[KB]` usage v1.3 S3.5), and `bench_burst_provider.py:43-53`
says the same. Second, the additive part makes $x^{\rm obs}$ negative where the
window is near zero; the map must stay invertible, so this is by design
(`:52-53` there; E0's row for $x$ is annotated). In a default shard 37.85 %
of the stored samples are negative on the `bench` provider and 19.84 % on the
`dsn` provider, the smallest $-0.0586$ `[RAN]` B4, so the usage guide's
check "`x >= 0`" on a read-back bench shard (`[KB]` usage v1.3 S3.3) holds for
a provider's output, not for what a shard stores. Third, Stage 3c builds its own
`NuisanceSpec()` and `RealisationSpec()` instead of reading the sidecar's
(`run_stage3c.py:180-181`); with no flag able to change either at build time,
the two agree at `834eb41`, and would silently disagree after any edit of
the spec.

#### 3.3.9 Plumbing: `--max-records`, `--dry-run`, `--dsn-main-dir`

`--max-records` caps the traces at the largest multiple of $n_{\rm wd}$ not
above it, for a probe run (`build_latent_bank.py:178-183`); it is outside
the digest (S3.2 (e)). `--dry-run` (the job's `DRYRUN=1`) prints the plan --
arm, shard, traces, donors, rows, $W$, provider, gap, whether $\theta$ is
withheld, the output path -- after building the provider and the specs, and
writes nothing (`build_latent_bank.py:194-209`); the job still creates
`OUT_DIR` (`build_latent_bank.pbs:89`). `--dsn-main-dir` points the `dsn` and `bench` providers at
another DSN tree; the in-repo `hpc/dsn` otherwise (`dsn_locate.py`).

#### 3.3.10 The demonstration scripts' flags

`demo_classes_generate.py` always uses the bench provider and is not a bank
(no shard, no sidecar, no digest; `:9-15`): `--n-per-class` 30,
`--n-classes` 3 (owner P2), `--n-windows` 4, `--T-win` 15.0, `--fs` 50.0,
`--n-neurons` 100, `--tau-ov` 0.10, `--seed` 0 (owner P5),
`--n-background` 60 uniform-box traces labelled $-1$, and `--nuisance`
(`:127-147`). Each trace is its own donor and well with
`RealisationSpec(n_per_theta=1)` (`:109, 116-123`). `--nuisance` raises at
the first trace: `sample_nuisance` returns a pair `(nu, parts)` and the
script keeps the pair as `nu`, so `nu[0]` is the whole array of every
trace's $\nu$, one row of five per trace, and `nuisance_affine` refuses it, `ValueError: nu_row must have 5 entries`
`[RAN]` B6 (`demo_classes_generate.py:112-113, 120`; F-ax). The usage guide
recommends exactly that flag as the fastest view of the scale question
(`[KB]` usage v1.3 S4). Its other two flags are `--out`
(`demo_classes.npz`) and `--dsn-main-dir`. `demo_classes_plot.py` reads the
file and draws: `--demo` (`demo_classes.npz`), `--out-dir` (`.`),
`--n-show` 3, `--zoom-s` 6.0, `--zoom-start` 0.0, `--embed` `all`,
`--perplexity` 25.0 (capped at a third of the embedded points less one,
never below 2, `demo_classes_plot.py:198`), `--pca-dim` 50, `--seed` 0
(`demo_classes_plot.py:254-270`).

### 3.4 The job scripts: resources, variables, forwarding, arrays

Sections 3.1-3.3 described what a bank knob does when it reaches the
builder; this section turns to the eight job files that decide whether it
reaches anything, and with what resources.

**Table P7.5 -- the job files** (resources `[RAN]` B2 from the `#PBS`
lines; the rest `[REPO]`; a bare line number in a row is in that row's
file).

| job | resources | variables (default) | forwards | conda | `DRYRUN=1` |
|---|---|---|---|---|---|
| `stage1/jobs/build_latent_bank.pbs` | 2 CPUs, 8 GB, 2 h | `ARM` (S), `OUT_DIR` (required), `N_TRACES` (64), `WELLS_PER_DONOR` (2), `N_WINDOWS` (8), `T_WIN` (60.0), `FS` (50.0), `PI` (0.0), `GAP_MODES` (`range_shift`), `N_PER_THETA` (2), `PROVIDER` (`reference`), `SEED` (0), `MAX_RECORDS` (0), `DRYRUN` (0), `ENV_NAME` (`sbi_env`) (`:42-56`) | 15 of 22 flags; the 7 of S3.1 have no variable | `set +u`, hook, activate, `set -u` (`:66-77`) | skips activation; `--dry-run`; reads no shard |
| `stage3/jobs/joint_arms.pbs` | 4 CPUs, 16 GB, 6 h | P0 Table E (`:50-62`) | 14 of 38 flags (B2) | as above (`:78-89`) | skips activation; loads every shard first (F-az) |
| `stage3/jobs/probe_conda_activation.pbs` | 1 CPU, 1 GB, 5 min | none; env `sbi_env` hard-coded (`:38-41`) | -- | the activation under test | -- |
| `stage3/jobs/probe_dsn_runtime.pbs` | 4 CPUs, 8 GB, 10 min | `PYBIN` (`/davinci-1/home/ldellamea/.conda/envs/sbi_env/bin/python`), `STAGE4_E` (12), `WINDOW` (3000) (`:42-44`) | `--embedding-size`, `--window` | none: the interpreter by absolute path (`:25-38`) | -- |
| `stage3b/jobs/stage3b.pbs` | 2 CPUs, 8 GB, 30 min | `RUNS_DIR` (required), `SIM_SHARDS`, `REAL_SHARDS`, `PROBE_CKPT`, `BOOTSTRAP_DIR`, `OUT` (`stage3b_report.md`), `OUT_JSON`, `MAX_PROBE_ROWS` (512), `ENV_NAME`, `DRYRUN` (`:40-49`) | 9 of 10 flags (all but `--dsn-main-dir`) | as above (`:56-67`) | skips activation; reads no shard |
| `stage3c/jobs/stage3c.pbs` | 4 CPUs, 16 GB, 4 h | `CKPT`, `SIM_SHARDS`, `OUT_DIR` (required), `KERNEL_AXES`, `SPREAD_AXES` (empty), `N_FLOOR_DRAWS` (48), `N_POST_DRAWS` (128), `FD_SEEDS` (3), `FD_STEP` (0.02), `SEED` (0), `ALLOW_REAL` (0), `VALIDATION`, `ENV_NAME`, `DRYRUN` (`:43-56`) | 13 of 14 flags | as above (`:63-74`) | skips activation; loads every shard first (F-az) |
| `stage4/jobs/joint_tune.pbs` | 4 CPUs, 16 GB, 8 h | P0 Table E; P6 S3.1 (`:60-76`) | `evaluate`'s flags (`:141-158`) | as above, unless `SKIP_CONDA=1` (`:86-91`) | activates anyway; `evaluate --dry-run` |
| `stage4/jobs/launch_joint_tune.sh` | -- (submits) | positional campaign and results dir; `PYBIN`; `--submit`, `--max`, `--qsub-args` (`:23-38, 48`) | 4 always, 7 when set (`:135-142`) | -- | dry run unless `--submit` (`:151-159`) |

The array index maps: the bank job's element is the shard index
(`PBS_ARRAY_INDEX` to `--shard-index`, `build_latent_bank.pbs:63, 94`); the
arms job's index `IDX` runs arm `ARMS[IDX % 9]` with seed `IDX / 9` (integer
division), with `ARMS = (A1 A0 A0s A2 A2s A3 A5 A_ref shuffled)`
(`joint_arms.pbs:69-73`), so arm A1 is
indices 0, 9, 18, 27, 36 and `A_ref`, `shuffled` are 7 and 8 for seed 0
`[RAN]` B2 -- the Giulia plan's G5 indices (`[KB]` v1.7 S5); the tuning job's
index is a position in the sorted pending listing (P6). The documented
arrays request at most 128 core-hours (bank, 32 elements), 1080 (arms, 45)
and 256 (tuning, 8) `[RAN]` B2.

**Activation.** Every job that activates conda turns `set -u` off across
both the hook and `conda activate` and back on after, the fix for the
`_CONDA_SET_GEOTIFF_CSV: unbound variable` failure (`[KB]` usage v1.3 S8;
`HANDOFF_hpc_implementation_v1_1.md` S3.7) that killed job 1575002
(`probe_conda_activation.pbs:20`); `probe_conda_activation.pbs`
exists to test that guard in a fresh batch shell, and
`probe_dsn_runtime.pbs` avoids activation altogether by calling the
environment's interpreter by absolute path, which skips the `activate.d`
hook that sets davinci's `LD_LIBRARY_PATH` (`[KB]` `HPC_PATHS.md` sec. 5e
and 7.1, the `GLIBCXX_3.4.26` caveat: the probe passed that way, and
anything that imports scipy would not). The two probes also hard-code the user's home in
`#PBS -o` and in `PYBIN` (`probe_conda_activation.pbs:6`,
`probe_dsn_runtime.pbs:6, 42`): the right paths for the user, wrong for any
other account. The probe's docstring cites the trainer's line numbers 334,
347, 396, 442 for the bank load, the dry-run return, the DSN block and the
backbone (`probe_dsn_runtime.py:12-14`); at `834eb41` they are 338, 364, 400
and 444 (`run_joint_arms.py`), a drift the docstring itself asks to be
re-checked.

**Dry runs.** `DRYRUN=1` skips the activation in four jobs and runs
`python3` from the calling shell, so a dry run needs the shell the usage
guide sets up, `sbi_env` activated (`[KB]` usage v1.3 S1); the Stage 3, 3b
and 3c entry points import torch at the top (`run_joint_arms.py:44`,
`run_stage3b.py:29`, `run_stage3c.py:26`). Two of them read the whole bank
before they print: `run_joint_arms.py` loads every simulated and real shard
(`run_joint_arms.py:338-341`) before its dry-run return
(`run_joint_arms.py:351-364`), `run_stage3c.py` every simulated shard
(`run_stage3c.py:152-157`) before its own (`run_stage3c.py:159-170`). At 32 default shards that is
196.6 MB of windows on a login node `[RAN]` B1; the Giulia plan estimates its
simulated bank at 70.6 GB before any filter and schedules a `DRYRUN=1` of
the arms job on it (`[KB]` Giulia plan v1.7 G4, G5, the size marked
[reasoning] there). F-az.

**Empty arrays under `set -u`.** `joint_arms.pbs`, `stage3b.pbs` and
`stage3c.pbs` collect optional flags in `EXTRA=()` and pass `"${EXTRA[@]}"`
(`joint_arms.pbs:94-115`; `stage3b.pbs:74-86`; `stage3c.pbs:80-99`). Outside
a dry run the array is empty for an arms element without `REAL_SHARDS`
(unless arm A3 is given `WARM_START_CKPT`), for a 3b run that sets none of
`SIM_SHARDS`, `REAL_SHARDS`, `PROBE_CKPT`, `BOOTSTRAP_DIR`, `OUT_JSON`, and
for a 3c run at its defaults (no `KERNEL_AXES`, `SPREAD_AXES`,
`VALIDATION`, `ALLOW_REAL`). Bash 4.4 and later expand an empty array to nothing under
`set -u` -- bash 5.2 in the sandbox, `argc=0`, rc 0 `[RAN]` B2 -- while
"An empty array becomes an error (in bash 4.3, but not in bash 4.4 ...)"
(`[WEB]` Greg's Wiki, BashFAQ/112, read 2026-10-03); under `set -u` an
unbound-variable error ends a non-interactive shell before the Python runs.
davinci's bash version is not recorded in the knowledge base. F-ay.

**The launcher** (F-ar, launcher side) forwards `CAMPAIGN`, `RESULTS_DIR`,
`SIM_SHARDS`, `OUT_DIR` always and `REAL_SHARDS`, `SPLIT_HASH`,
`CONTRACT_DIGEST`, `EPOCHS`, `STEPS_PER_EPOCH`, `SEED`, `SBI_HPC_DIR` when set
in its environment (`launch_joint_tune.sh:135-142`); it forwards neither the
shape anchors `P`, `EMBEDDING_DIM`, `D_THETA` nor `TAG`, `ENV_NAME`,
`SKIP_CONDA`, `PYBIN` or `DRYRUN`, so the array evaluates at the job's
anchors (26, 12, 26) whatever `propose` was given (P6 S3.3.2, `[RAN]` there)
-- on a bench bank, at $p = d_\theta = 10$, that re-pins every trial.

### 3.5 What the upstream knobs fix for the training knobs

Section 3.4 closed the job layer; this section lists, for the documents that
come before P7, which of their quantities a bank decides before any of their
knobs is read.

| upstream knob | fixes | for |
|---|---|---|
| `--T-win`, `--fs` | $W$: the backbone's input length, its output length and the receptive field's coverage | P1 |
| `--provider` (and `--n-latent`) | $p = d_\theta$: the search anchors `--p`, `--d-theta` | P6 |
| the same | the $4 d_\theta$ floor of $S_{\rm mc}$ (40 on the bench, 104 at $d_\theta = 26$, `[KB]` usage v1.3 S5.3) | P3 |
| the same | the `hidden_features` range through `max(p, E)` | P4 |
| `--n-classes` | $C$, the DSN loss's class count and its separation target $-1/(C-1)$ | P2 |
| `--wells-per-donor`, `--n-windows`, the shard count | $N_{\rm pair}$, its composition, eq. (P7.9) | P3 |
| `--n-traces`, the shard count | the number of donor ids, hence the split's groups and $N_{\rm train}$ | P5 |
| `--provider` | the scale of $x$, which the flow conditions on through $h_\psi$ unscaled (`z_score_x` `none`) | P4 |
| `NuisanceSpec`, `--pi`, `--gap-modes` | what the encoder must be invariant to, and how far bench arm R is from $p_0$ | E6, E7 |
| `--seed` of the two bench arms | whether the primary endpoint is held out (F-aw) | E7 |

### 3.6 Failure modes and the diagnostics that reveal them

Section 3.5 listed what a bank fixes when it is right; this section lists how
it is wrong and where the evidence appears.

| failure | cause | where it shows | diagnostic |
|---|---|---|---|
| a fixture bank trained as if simulated | `PROVIDER` omitted (`build_latent_bank.pbs:52`) | nowhere in a run's output | the sidecar's `generator_sha256` starts `reference-fixture:` |
| replicate pairs across unrelated donors | multi-shard bank, shard-local ids (F-at) | arm A5's loss; every run's `T vs p_eff` line | count distinct $\theta$ per `donor` value of the concatenated bank; $r_{\rm same}$ of eq. (P7.9) |
| a held-out endpoint that is not held out | both bench arms at one `SEED` (F-aw) | `L pseudo-real` close to the training NLL | compare `provenance.base_seed` of the two banks' sidecars |
| no shard at all for bench arm R | bench provider, `range_shift`, $\pi > 0$ (F-av) | `ValueError: duty must be in (0, 1]` in the element's log | eq. (P7.8) |
| Stage 3c dies at its first floor, or measures another simulator | DSN provider hard-wired (F-au) | `ValueError: phi has length 10, expected n = 6` on a bench bank; nothing on a reference bank | the bank's `generator_sha256` against `run_stage3c.py:179` |
| a 3b/3c run on a Stage 4 finalist fails to load | encoder rebuilt at runner defaults (F-ae) | a size mismatch at `load_state_dict` | the checkpoint's `meta` against the defaults |
| a job dies before Python | empty `EXTRA` under `set -u` on bash $\le$ 4.3 (F-ay) | `EXTRA[@]: unbound variable` | `bash --version` on a compute node |
| a login-node dry run runs out of memory | dry runs read every shard (F-az) | the dry run itself | -- |
| a demonstration with nuisance fails | the unpacked pair (F-ax) | `ValueError: nu_row must have 5 entries` | -- |
| the dropout nuisance never acts | default scales (Table P7.3) | none | eq. (P7.5) |

The search of `stage2`-`stage4` for readers of the identity fields
(`grep` for `["donor"]`, `["well"]`, `["batch"]`, `["realisation_id"]`,
`["subregion"]`, `["window_idx"]`, smoke tests included) returns `run_joint_arms.py:369, 494, 628, 632` and
`run_stage3c.py:122, 304, 379, 386`, nothing else `[REPO]`.

### 3.7 Findings this document owns

Section 3.6 tabulated the failures; this section states each finding once,
with its evidence and its resolution status, in the form the index's S6
carries.

- **F-n (carried).** `stage3c.pbs` defaults `N_POST_DRAWS` to 128
  (`stage3c.pbs:49`) and `run_stage3c.py --n-post-draws` to 64
  (`run_stage3c.py:66`). The two entry
  points run different floors: the Monte Carlo part of $\Sigma_{\rm fl}$
  scales as $1/S_{\rm post}$ (eq. (P7.10)), so the job's floor carries half
  the Monte Carlo variance of the direct call's. Report; open.
- **F-t (carried, bank side).** The runner's class count comes from the
  simulated bank; on the bench both bench arms carry the prior's $0..C-1$,
  so it agrees by construction; the Giulia simulated bank is planned with
  `cls = -1` (S3.3.5). `--n-classes` is not a job variable.
- **F-aa (carried, bank side).** On the bench, $N_{\rm pair} = 0$ needs
  $n_{\rm wd} = 1$ and $J = 1$; any $J > 1$ gives within-well pairs.
- **F-ab (carried, bank side).** The within-well share of $\mathcal{U}$ is
  set by $n_{\rm wd}$ and $J$ (eq. (P7.9) at $S_{\rm sh} = 1$); on several
  shards it combines with F-at.
- **F-ae (carried, job side).** Neither `stage3b.pbs` nor `stage3c.pbs` has
  a variable for an encoder axis, so a checkpoint off the runner's encoder
  defaults (any Stage 4 trial that moved one) cannot be probed or floored
  through the jobs either.
- **F-ar (carried, launcher side).** S3.4.
- **F-at (new).** The identity fields are shard-local and every consumer
  reads them as bank-global (S3.3.3): on $S_{\rm sh}$ shards of one layout
  the replicate stream draws pairs of which only the fraction $r_{\rm same}$
  of eq. (P7.9) share $\theta$ (0.0294 at the usage guide's 32 shards
  `[RAN]`), the Stage 3 replicate diagnostic mixes the two kinds (15 of its
  64 pairs share $\theta$ at 32 shards `[RAN]`), the split has 32 groups
  whatever the bank's size and a hash blind to it, and Stage 3c's culture
  means average unrelated wells under one label. Evidence:
  `build_latent_bank.py:216-218, 249-254`; `latent_bank.py:184-204`;
  `joint_batches.py:37-58, 160-164`; `run_joint_arms.py:369, 610-617`;
  `run_stage3c.py:376-388`; `[RAN]` B1, B3, B3b. Report; open (D-038
  candidate: bank-global ids in the builder, or offsets in `concat_shards`;
  the window exporter B5 inherits the requirement); operating rule
  meanwhile: one shard per bench arm.
- **F-au (new).** Stage 3c builds `load_dsn_provider()` with the DSN's
  default `LatentSpec` for its floors and Jacobians (`run_stage3c.py:178-179`;
  `latent_sbi_simulator.py:376-382`) and never reads the bank's provider
  tag: on a bench bank ($p = 10$) the first simulation raises `ValueError:
  phi has length 10, expected n = 6` `[RAN]` B4; on a reference bank it
  simulates the DSN generator at the fixture's $\theta$, whose windows have
  a similar mean (8.41 against 9.08 at $\phi = 0.5$, `[RAN]` B4) and so
  raise no suspicion; on a `dsn` bank it is consistent, because the DSN's
  parameter map reads the axes and the smoothing width of the spec and the
  duration, bin width and `n_neurons` each call sets, none of which
  `--n-classes` or `--tau-ov` changes (`latent_sbi_simulator.py:326-341`;
  `dsn/latent_burst_generator.py:566-601`). Report; open (build the provider
  from the sidecar's tag and spec).
- **F-av (new).** The gap (a) is three perturbations under one flag, and on
  the bench provider it raises (S3.3.6, eq. (P7.7)-(P7.8)): 68 % of default
  shards at $\pi = 0.1$ `[RAN]`; plan D11's minimum viable design cannot be
  built on the bench provider. Evidence: `bench_burst_provider.py:82-131`;
  `bench_burst_generator.py:112-113`; `smoke_test_latent_sbi.py:78-81`;
  `latent_sbi_simulator.py:314-343`; `[RAN]` B4. Report; open (a decision on
  the gap's semantics, E6).
- **F-aw (new).** Bench arm R repeats bench arm S's draws when built at one
  `SEED` (S3.3.2): every field equal, and at $\pi = 0$ every byte of `x`;
  the pseudo-real endpoint then scores copies of training rows in 68.75 % of
  its rows `[RAN]` B3. Evidence: `build_latent_bank.py:192, 211-222, 244`;
  `run_joint_arms.py:551-558, 369`; `smoke_test_latent_sbi.py:549-550`.
  Report; open (D-038 candidate: the arm in the base seed, or a refusal when
  the banks' base seeds coincide); operating rule meanwhile: a different
  `SEED` for bench arm R.
- **F-ax (new).** `demo_classes_generate.py --nuisance` raises at the first
  trace (`demo_classes_generate.py:112-113, 120`; `[RAN]` B6), while the usage guide recommends the
  flag (`[KB]` usage v1.3 S4). Report; open (unpack the pair).
- **F-ay (new).** `"${EXTRA[@]}"` of a possibly empty array under `set -u` in
  `joint_arms.pbs:115`, `stage3b.pbs:86`, `stage3c.pbs:99`: an error on bash
  4.3 (`[WEB]` BashFAQ/112), not on 4.4+ (`[RAN]` 5.2); davinci's version
  unrecorded. Report; open (`bash --version` on a compute node; the
  `${EXTRA[@]+"${EXTRA[@]}"}` idiom is the usual guard [reasoning]).
- **F-az (new).** The dry runs of `joint_arms.pbs` and `stage3c.pbs` load
  every shard before printing (`run_joint_arms.py:338-341` before `:351`;
  `run_stage3c.py:152-157` before `:159`). Report; open (read the sidecars
  only in a dry run); relevant to the Giulia plan's G5 dry run.

Observations recorded, not findings: the digest certifies specs and not
size or nesting, and hashes each provider class's own source but not the
generator modules it calls (S3.2 (e)); the dropout component almost never acts at the
default scales (S3.2 (c)); Stage 3c uses default spec objects rather than the
sidecar's (S3.3.8); P0 Table F's two `drift_period_s` rows are two objects
(convention 4); the probe's docstring line numbers have drifted (S3.4); the
usage guide's kernel indices 23-25 belong to the campaign bank, not to the
`dsn` provider, and none of the three Stage 1 providers has a kernel axis
(A.2).

## Appendix A -- the Stage 3b and 3c flags (post-hoc, not training)

The two stages read a finished Stage 3 run and a bank; their flags change
nothing a model learned, only what is measured about it. They are P7's
because they are job variables and because their inputs are the bank's
fields.

### A.1 Stage 3b: `run_stage3b.py` and `stage3b.pbs`

Stage 3b splits A0's deficit into a domain and an objective term (plan
eq. (9)) from the per-row arrays the runs persisted, and probes the encoder
directly (`run_stage3b.py:1-20`).

- `--runs-dir` (`RUNS_DIR`, required): the Stage 3 output directory; the
  decomposition needs arms A0, A0s and A1 (`:124`); the dry run reports
  what is computable and reads no shard (`:129-142`).
- `--sim-shards`, `--real-shards`: needed by the probes only (`:208-209`).
- `--probe-ckpt` (`PROBE_CKPT`): default `<runs-dir>/A0_seed0_ckpt.pt`
  (`:126-127`); the encoder is rebuilt at the runner's encoder defaults
  (`:66-90`; F-ae).
- `--max-probe-rows` (`MAX_PROBE_ROWS`, 512; $n_{\rm probe}$): each probe
  feeds the first 512 rows of each concatenated bank to the encoder
  (`run_stage3b.py:220-221, 250-252`; `encoder_probes.py:113-116`) -- the
  FIRST rows in sorted-shard order, not a sample. At the
  defaults one shard is exactly 512 rows (B6), so the probes see shard 0 of
  each bank: 32 donors, 64 wells `[RAN]` B3b.
- `--bootstrap-dir`, `SBI_HPC_DIR`: where `bootstrap_paired` is imported
  from (`:55-63`); without it the intervals are absent and only the
  $\sigma_{\rm seed}$ clause is applied (`:172-175`).
- `--out` (`OUT`, default `stage3b_report.md` in the job, stdout in the
  CLI), `--out-json`: the job's default is a relative path and the job `cd`s
  into `stage3b/` (`stage3b.pbs:38, 45`), so the report lands inside the
  checkout unless `OUT` is absolute.
- `--dsn-main-dir`: no job variable.

### A.2 Stage 3c: `run_stage3c.py` and `stage3c.pbs`

Stage 3c runs the floors, the aliasing test, the stratification and the
P10 curve on the bench and gates on three of them (plan Stage 3c;
`run_stage3c.py:1-16`). Its evaluation point is $\theta^{\rm fl}$, the
bank's mean $\theta$ clipped to $[0.05, 0.95]$ (`:182-183`), a fixed point
standing for "one culture", not a well's $\theta^*$.

**The floors** (`--n-floor-draws`, `N_FLOOR_DRAWS` 48, $n_{\rm fl}$;
`--n-post-draws`, CLI 64, job 128, $S_{\rm post}$). Each floor simulates
$n_{\rm fl}$ traces of $J$ windows at $\theta^{\rm fl}$ varying one factor --
the nuisance $\nu$ with the realisation held, or the realisation with
$\nu = 0$ -- encodes them, draws $S_{\rm post}$ posterior samples per window
and keeps the mean over windows of the per-window means, $\hat m^{\rm fl}$
(`floor_core.py:32-77`; `nuisance_floor.py:31-64`; `realisation_floor.py:28-47`);
$\Sigma_{\rm fl}$ is their sample covariance (`floor_core.py:80-119`). With the
posterior draws independent across windows and draws, and the floor's draws
independent of each other, the law of total covariance
`[textbook, from memory]` gives, for each fixed trained flow,

$$\mathbb{E}[\Sigma_{\rm fl}] = \mathrm{Cov}_\nu(m^{\rm fl}) + \frac{1}{J^2 S_{\rm post}} \sum_{i=1}^{J} \mathbb{E}_\nu[C_i] \tag{P7.10}$$

for the nuisance floor (for the realisation floor, the same with the
realisation $\mathcal{G}$ in place of $\nu$), where $C_i$ is the flow's posterior
covariance for the floor draw's window $i$ [reasoning: the per-window Monte
Carlo means have covariance $C_i / S_{\rm post}$ given the window, and the
`ddof=1` sample covariance is unbiased across independent draws]. The
second term is Monte Carlo noise in the floor, not nuisance; it halves from
the CLI's 64 to the job's 128 (F-n). Two properties of the draws: the
nuisance floor gives every draw its own donor and well key and ONE batch key,
`"B"` (`nuisance_floor.py:44-47`), so it varies $\nu^{\rm don} + \nu^{\rm wel}$
and holds $\nu^{\rm bat}$ at one draw -- one third of each component's
variance at the default scales, which equals the variance of a same-donor
difference, $2 (\sigma^{\rm wel}_m)^2$, because
$\sigma^{\rm don}_m = \sigma^{\rm wel}_m$ there `[RAN]` B5; and the simulations run through the DSN
provider (F-au).

**The concentration gate** (`--kernel-axes`, `KERNEL_AXES`, empty; $n_{\rm ker}$
axes). The realisation floor passes when the share of its total variance
that sits on the named axes exceeds the midpoint between the uniform share
and 1,

$$\frac{1}{2}\Big(1 + \frac{n_{\rm ker}}{d_\theta}\Big) \tag{P7.11}$$

(`realisation_floor.py:50-80`): 0.5577 at $n_{\rm ker} = 3$, $d_\theta = 26$;
0.75 at 3 of 6; 0.65 at 3 of 10 `[RAN]` B6. Empty, the gate and the D17
audit are skipped (`run_stage3c.py:204, 296`); which bench axes play the
kernel role is open (`[KB]` usage v1.3 S6.2). None of the three providers
has a connectivity-kernel axis to name: the `dsn` provider's six axes are
the DSN's burst axes, irregularity to background
(`dsn/latent_burst_generator.py:193-200`), the `bench` provider's ten are
`BENCH_AXES` (Table P7.2), the fixture's are unnamed. The kernel indices
23-25 the usage guide records for "the `dsn` provider" are positions in the
campaign bank's 26-entry `param_names`, read from the r2 export's sidecars
(`[KB]` usage v1.3 S6.2, its own table), and the `dsn` provider's bank has
six axes, not 26 [a misattribution in the guide, corrected in its v1.4]. So on the bench, as
built, the concentration gate can only be run on axes chosen to stand in
for the kernel, which is the open decision.

**The aliasing test** (`--fd-step`, `FD_STEP` 0.02, $h_{\rm fd}$; `--fd-seeds`,
`FD_SEEDS` 3, $n_{\rm fd}$). $J_\theta$ by central differences of step
$h_{\rm fd}$ in the unit box, one-sided at the box's edge
(`aliasing.py:72-102`); $J_\nu$ by central differences of half each
component's total standard deviation, with an 8-fold step to tell a
quantised component from an inert one (`aliasing.py:105-154`); both averaged
over $n_{\rm fd}$ seeds (`run_stage3c.py:225-233`). At the defaults the
dropout component is the quantised one: a step of
$0.5 \times 0.3674 = 0.1837$ leaves the gain at 1 on both sides, the 8-fold step 1.4697 moves it
to $8/9$ on one side `[RAN]` B5 (eq. (P7.5)), so its column of $J_\nu$ is the
small-step difference, exactly zero, with the quantised flag set
(`aliasing.py:140-153`). From the two Jacobians the stage reads the
aliasing coefficient $a_m$ of each nuisance direction $m$ (plan eq. (4);
`run_stage3c.py:239-240`), and the gate compares the nuisance floor's
leading direction with the aliasing prediction, cosine above 0.7
(`run_stage3c.py:255-266`).

**The stratification gate** (`--spread-axes`, `SPREAD_AXES`, empty).
`stratify` solves plan eq. (5) on one posterior mean per WELL value
(`run_stage3c.py:268-293, 376-388`; `stratify.py:136-180`) and, given the
spread axes, gates on whether the leading $\mu_j$ separate and load on them
(`stratify.py:183-196`). On a multi-shard bank the wells are F-at's merged
groups. Empty, the gate is skipped.

**P10** reads the first 16 rows with the first row's `donor` value
(`run_stage3c.py:304-309`): on a sorted bank they are rows 0-15 of shard 0,
one donor, one $\theta$ `[RAN]` B3 -- unaffected by F-at.

**The rest.** `--allow-real` with `--validation` (`ALLOW_REAL`,
`VALIDATION`): refused unless a bench validation file shows no failed gate
(`run_stage3c.py:138-150`); `--seed` (owner P5): the floors' and Jacobians'
base seed; the exit code is non-zero when any gate is `False`
(`run_stage3c.py:328-333`), and the job
passes it through (`stage3c.pbs:89-109`).

## 4. Summary of results

| statement | where |
|---|---|
| A shard is $N_{\rm tr} J$ rows laid out by eq. (P7.1); every draw derives from $s_{\rm base} = 1000003\, s_{\rm seed} + \mathsf{k}$, eq. (P7.2); at the defaults 512 rows, 32 donors, 64 wells, 4 batches, $W = 3000$, 6.144 MB of windows `[RAN]` B1 | S3.2 (a), S3.3.3 |
| The provider fixes $p$ (6, 6, 10), the label axes and the scale of $x$ (mean 9.08, 8.41, 0.0674 at $\phi = 0.5$); the job defaults to the fixture `[RAN]` B4 | S3.3.1, Table P7.4 |
| The nuisance map is eq. (P7.3)-(P7.4); identity at $\nu = 0$; batch share 2/3 of every component; the dropout acts beyond eq. (P7.5), $3.176\, \sigma^{\rm tot}_4$, probability $7.5 \times 10^{-4}$ per well; the additive parts are 0.27-0.29 % of a sum-scale window and 36 % of a per-unit one; a default shard stores 37.85 % (bench) and 19.84 % (`dsn`) negative samples `[RAN]` B4, B5 | S3.2 (c), S3.3.8 |
| The digest covers 16 fields and not a shard's size or nesting; the two bench arms never share one `[RAN]` B3 | S3.2 (e) |
| Identity fields are shard-local; $r_{\rm same}$, $r_{\rm well}$ of eq. (P7.9) fall roughly as $1/S_{\rm sh}$: 2.94 % and 1.57 % at 32 shards; the split has 32 groups and one hash at any size `[RAN]` B1, B3, B3b | F-at, S3.3.3 |
| Bench arm R at bench arm S's seed repeats its draws; the pseudo-real endpoint scores 68.75 % training copies at $\pi = 0$ `[RAN]` B3 | F-aw, S3.3.2 |
| Gap (a): eq. (P7.6) on `bench` and `dsn`, every axis on the fixture; on `bench` a shard raises with probability eq. (P7.8), 0.68 at $\pi = 0.1$ `[RAN]` B4 | F-av, S3.3.6 |
| Stage 3c simulates through the DSN provider: raises on bench banks, swaps the simulator on fixture banks `[RAN]` B4 | F-au |
| Jobs: resources of Table P7.5; 7 builder flags without a variable; empty `EXTRA` under `set -u` on bash 4.3; dry runs that load every shard | S3.4, F-ay, F-az |
| Stage 3c's floor covariance is eq. (P7.10); its Monte Carlo term halves from 64 to 128 draws (F-n); its nuisance floor holds the batch level; the concentration threshold is eq. (P7.11) `[RAN]` B5, B6 | A.2 |
| None of the three Stage 1 providers has a connectivity-kernel axis; the kernel indices 23-25 of the usage guide are positions in the campaign bank's 26 axes `[REPO]`, `[KB]` | A.2 |

## 5. Open points, caveats, assumptions

- **No bank has been built by array on the cluster, and no job of
  `hpc/joint/` has run** (`[KB]` usage v1.3 S9). Every number is the code's
  behaviour on banks built in the sandbox, at the defaults unless stated.
  The providers' means of Table P7.4 are one realisation at $\phi = 0.5$,
  illustrative of scale, not a calibration.
- **The bench generator's axis ranges** (`BENCH_AXES`) cite a specification
  that is in neither the repository nor the knowledge base; their
  provenance is unchecked.
- **davinci's bash version** is not recorded (F-ay); the claim about bash
  4.3 rests on one web source, BashFAQ/112, not on a run of that version.
- **The pairs' effect on training** (F-at) is argued from the loss's form
  [reasoning]; how much it degrades an A5 run is not measured.
- **The pseudo-real endpoint's optimism** under F-aw is stated as a share
  of copied rows; how much it lowers `L pseudo-real` is not measured.
- **Decisions this document needs and does not make** (for the log's Open
  calls): the identity-field repair and B5's contract (F-at); Stage 3c's
  provider (F-au); the gap (a)'s semantics on the bench provider and plan
  D11's design (F-av); the seed separation of the bench arms (F-aw); which
  bench axes are kernel axes (usage v1.3 S6.2, carried); the nuisance scales
  against the per-unit convention (usage v1.3 S3.5, carried, now measured).
- **The DSN provider under the gap** was tried at one corner ($\pi = 1$,
  $\phi = 0.999$) and one default shard at $\pi = 0.5$; that it never raises
  is not shown.
- **Stage 3b's and 3c's numerical behaviour** with a trained model is not
  exercised (no torch in the sandbox); eq. (P7.10) is derived, not run.

## 6. References / further reading

**Project knowledge base `[KB]`.** `JOINT_DSN_NPE_USAGE_v1.md` v1.3: S1
(activation, one environment), S3.1-S3.5 (providers, the array, the shard's
three lines, the scale convention and its open item), S4 (the demonstration
and its `--nuisance`), S5.2 (the arms' index map), S6 (Stages 3b and 3c, the
kernel axes), S8 (the conda trap), S9 (nothing has run), S10.
`claude/PLAN_2026-10-01_giulia_hpc_stages.md` v1.7: S1 (D-041, $W = 18\,000$),
S3 (B5, B6), S5 (G0, G4 with its 70.6 GB estimate marked [reasoning] there,
G5). `claude/deck_pack/07_SEC_F_bench_and_stages.md` (F.1-F.3: the arms,
the severity design, the sizes). `claude/REPLICATE_LOSS_GUARDS_v1.md` S3.7
(the `p_eff_min` wording of P0 Table C, corrected in P0 v1.3 this turn).
`HPC_PATHS.md` sec. 5-7 (the probe's absolute-path caveat, `$HOME`, the
conda trap). `HANDOFF_hpc_implementation_v1_1.md` S3.7 (job 1575002). P0
S3.6-S3.8; P3 S3.2, S3.8 (F-ab); P5 S3.3 (the split); P6 S3.3.2, S3.8
(F-ar). The plan `JOINT_DSN_NPE_PLAN_v0_6.md` (repository, v0.6.5): S2.6
(Lever 3), S4.0-S4.5, Stage 1, Stage 3b, Stage 3c, S8 (D10, D11).

`[KB-PDF]`: Schmitt M, Burkner P-C, Kothe U, Radev ST. *Detecting model
misspecification in amortized Bayesian inference with neural networks: an
extended investigation* (arXiv 2406.03154v2, the project PDF): p.5, the
generative model $x = g(\theta, \xi)$ with "$\xi$ takes care of nuisance
effects that we only treat statistically" and the likelihood as the
marginal over $\xi$, eq. (2) -- the frame in which the bench's realisation
sits inside the likelihood; p.6, the simulation gap, "when the assumed
training model M deviates critically from the unknown true generative model
M*", and "a misspecified prior distribution worsens posterior inference just
like a misspecified likelihood function"; p.11, the necrosis knob, "A
Bernoulli distribution with parameter $\pi$ controls whether a cell is
affected by necrosis or not. Consequently, $\pi = 0$ implies no necrosis
(and thus no simulation gap), and $\pi = 1$ entails that all cells are
affected" -- the severity scalar the bench borrows (the bench's own
perturbations are not Bernoulli; deck 07 F.1 says the same); p.12, a
contaminated fraction of the reaction times, the family of the bench's
mode (d). No number of the paper is used. Hikida Y, Bharti A, Jeffrey N,
Briol F-X. *Multilevel neural simulation-based inference*, NeurIPS 2025,
read in its arXiv copy 2506.06087v4 (the project PDF), whose p.1 carries
the conference line "39th Conference on Neural Information Processing
Systems (NeurIPS 2025)": a peer-reviewed conference paper [corrected
2026-10-03: P4, P5 and P6 flagged it PREPRINT, not peer-reviewed; the three
are annotated in this turn]. P.1: "the performance of neural SBI can suffer
when simulators are computationally expensive" -- the regime the bench is
not in: a default bench shard of 512 windows builds in seconds of wall time
in the sandbox `[SANDBOX 2026-10-03]` (a timing is machine-dependent, so
`p7_numbers.py` prints none; the usage guide's own measure is 3 s for 600
windows of 750 samples, S3.4 there), so the bank's size is bounded by
training and storage, not by simulation.

**Repository `[REPO]`** at `834eb41`: the files and lines of the changelog
row; `hpc/dsn/latent_burst_generator.py:193-200` (the six default axes),
`:252-260` (the `LatentSpec` defaults: label axes (0, 1), three classes,
overlap 0.10, $\Delta t$ = 0.02 s, 100 neurons).

**Peer-reviewed literature.** According to PubMed, read in full text from
PubMed Central: Bernaerts Y, Deistler M, Goncalves PJ, et al. *Combined
statistical-biophysical modeling links ion channel genes to physiology of
cortical neuron types.* Patterns 2025; PMID 41142913, PMC12546760,
[DOI](https://doi.org/10.1016/j.patter.2025.101323) -- a neuroscience
application in which neural posterior estimation "failed for our biophysical
model and dataset due to a small but systematic mismatch between the data
and the model", posterior samples that "did not produce simulations that
came close to the experimental data", and a remedy that adds noise to the
training summaries; used in S3.3.2 and S3.3.6 only as the reason a
controlled, measurable gap (bench arm R) is worth its cost; no number used.

**Web `[WEB]`.** Greg's Wiki, *BashFAQ/112* (mywiki.wooledge.org, read
2026-10-03): "An empty array becomes an error (in bash 4.3, but not in bash
4.4, where there is no error even without assignment array=())", quoted for
F-ay. Not peer-reviewed; the bash 5.2 side was run `[RAN]`.

**Searches run `[RAN]`, 2026-10-03.**

| source | query | result |
|---|---|---|
| PubMed | network bursts cultured neurons multielectrode array Poisson model simulation | 1 record (PMID 23091458, PMC3476068: functional feedforward networks and avalanches; not on burst-generator design; not used) |
| PubMed | network burst detection cultured neuronal networks gamma distributed interburst intervals | 0 records |
| PubMed | network burst model spike train generator renewal process cultured neurons | 0 records |
| PubMed | truncated normal distribution prior bounded parameter Bayesian | 0 records |
| PubMed | truncated normal distribution | 2196 records; none opened (textbook concept, tagged as such) |
| PubMed | nested random effects batch donor well technical replicates variance components hierarchical model | 0 records |
| PubMed | nested design variance components replicate wells batch effects cell culture | 0 records |
| PubMed | nested random effects | 4507 records; none opened (eq. (P7.4) is the textbook variance of a sum) |
| PubMed | multielectrode array replicate wells hierarchical statistical analysis culture batch variability | 0 records |
| PubMed | simulation-based inference model misspecification simulation gap neural posterior estimation | 0 records |
| PubMed | model misspecification simulation-based inference | 6 records; PMC12546760 read in full and used; PMC13298659 (Levy-flight vs diffusion model, BayesFlow) not opened; four outside the field |
| PubMed | simulation-based inference multielectrode array neuronal cultures | 0 records |
| PubMed | bash shell script nounset unbound variable array | 0 records |
| bioRxiv | neuroscience, 2026-09-15 to 2026-10-02, first page of 30 records (the connector has no keyword search) | none on burst generators, MEA banks or SBI; no preprint is cited, so the published-version lookup had nothing to check |
| bioRxiv | bioinformatics, 2026-09-01 to 2026-10-02, first page of 30 records | none on topic (one on measuring a pipeline's cores and wall time, abstract only, not used) |
| data repositories | -- | no claim of P7 concerns a dataset; none queried |
| KB PDFs | `nuisance`, `severity`, `contaminat`, `simulation gap`, `misspecif` in the Schmitt et al. text; `expensive`, `fidelity`, `cost` in the multilevel text | Schmitt et al. p.5, p.6, p.11, p.12; the multilevel paper p.1-2, as cited (its p.1 carries the NeurIPS 2025 line) |

**Textbook, from memory** (tagged where used): the truncated normal law;
the variance of a sum of independent terms (eq. (P7.4)); the law of total
covariance (eq. (P7.10)).

---

### Pre-send check (Precision model)

R1 -- every symbol typed in S1; $x$ and $x^{\rm obs}$ are windows in
$\mathbb{R}^{W}$, the second possibly negative; $\mathcal{T}_\nu$ acts on
each window as an affine map of $\mathbb{R}^{W}$, $\gamma_\nu$ a positive scalar and
$\beta_\nu$ a function of $t_{\rm abs}$; $r_{\rm same}$, $r_{\rm well}$ are
fractions of a finite set; eq. (P7.8) is a probability over the prior's
free-axis draw; $\Sigma_{\rm fl}$ is a matrix in squared parameter units.
R2 -- "every byte of `x` equal" carries "at $\pi = 0$, one `SEED`, one
shard index"; "97.1 %" carries "at 32 shards of one layout at the
defaults"; "68 %" carries "per default shard, $\pi = 0.1$, over the prior";
"almost never acts" carries "at the default scales"; the bash claim carries
"on 4.3" and "davinci unrecorded"; eq. (P7.5) is stated on each side of its
threshold, not as an equivalence, because the rounding decides at the
threshold itself; eq. (P7.10) carries its independence hypotheses; "on a
`dsn` bank it is consistent" carries the fields the DSN map reads. R3 --
P3's target is said to be derived for one $\theta^*$, not transplanted to
pairs of different $\theta$; the severity design of Schmitt et al. is
borrowed as a scalar, not as a mechanism; the Bernaerts et al. remedy is
cited as an application, not as the bench's method; the bash 4.3 statement
is not extended to 4.2. R4 -- $T_{\rm drift}$ and $T_{\rm gap}$ are two
objects with one field name; $s_{\rm seed}$ and $s_{\rm base}$ two seeds;
$\mathsf{k}$ is not $k$; $\theta^{\rm fl}$ is not $\theta^*$; $S_{\rm post}$
is not $S_{\rm mc}$; $N_{\rm tr}$ is per shard and $N^{\rm bank}_{\rm tr}$ per
bank arm (E0's row annotated); the row index $i$ counts from 0 in a shard
and from 1 over a floor draw's windows, said at each use (convention 1);
the axis index $k$ counts from 1 and the code's indices from 0, said where
both appear; the endpoint scores stored windows, $x^{\rm obs}$, not
provider outputs. R5 --
the maps named with domain and codomain: the provider ($\phi$ and a seed to
$J$ windows), eq. (P7.6) (a free axis's coordinate to its physical image),
$\mathcal{T}_\nu$ ($\mathbb{R}^{W}$ to $\mathbb{R}^{W}$), the encoder
$h_\psi : \mathbb{R}^{W} \to S^{E-1}$. R6 -- "arm", "held out", "nuisance",
"realisation", "drift" declared in S1.1 or S2; "trace" is an IFR trace
throughout, and the floor's variance share is written as a share, not a
"trace share"; "bench provider" is only `--provider bench`, and the three
providers together are "the three Stage 1 providers". R7 -- the usage guide's "mid-box" figures are quoted with
the point they were measured at named; BashFAQ's sentence with its
versions; the plan's "two arms from one generator" with what the code
shares beyond the generator; the multilevel paper's quoted sentence with
its venue corrected. R8 -- the probability of eq. (P7.8) and one raising
build, the dropout probability and the count in a bank, $r_{\rm same}$ in
closed form and counted, $\mathbb{E}[\Sigma_{\rm fl}]$ and
$\Sigma_{\rm fl}$, $m^{\rm fl}$ and $\hat m^{\rm fl}$, $L$ and the
per-row $\ell_i$ carry their levels in convention 6 and at first use.
Result: fixes applied before sending -- a machine-epsilon symbol removed
from the prose; eq. (P7.5) restated one-sided; the trace index $t$
(E0's training progress) replaced by the stored `well` value; the axis
index made one-based in eq. (P7.7) and Table P7.2; a contamination factor
$s$ (E0's draw index) written in words; Schmitt et al.'s contamination
symbol written in words; every bare line citation whose file was not the one last named
resolved to its file, and the tables given a citation convention; the multilevel
paper's review status corrected here and in P4-P6; "no bench provider has a
kernel axis" (read as `--provider bench` only) restated for all three
providers, and a `dsn` arm-R build no longer called "a bench arm-R shard"
without its provider.
