# SPEC -- Simulation-Based-Inference: the joint-docs number scripts (Stage 5)

Single source of truth for what the code must do. The code follows this file;
when the two disagree, fix whichever is wrong and record why in the decision
log. The integration tester reads only the repository, so every constraint the
code must satisfy is stated here, including those that come from decisions.

Notation rules: every symbol is defined at first use; a conditional quantity
keeps its conditioning every time it appears (write F(y | x), never F(y));
quantifiers ("for each fixed x", "for all alpha") are stated explicitly.

This spec covers existing code. It was drafted on 2026-10-06 from the
scripts' docstrings and from the documents they serve (E0-E4, P2, P5 under
`hpc/joint/docs/`), not from a session with the author. Every block entry says
`Confirmed: inferred` until the author has checked it; anything else stated
outside the block entries that is still unconfirmed is marked `(inferred)`.

The intent of these scripts is unusual: each one exists to recompute the
numbers a document quotes. The documents are the author's and are the
description of intent; this file says which document governs which script and
what "correct" means for a number-recomputing script.

Last reviewed at commit: 9fcbc88

## 0. Coverage

Paths this spec covers (files, directories ending in `/`, or globs):

- `hpc/joint/docs/tools/e1_numbers.py`
- `hpc/joint/docs/tools/e2_numbers.py`
- `hpc/joint/docs/tools/e3_numbers.py`
- `hpc/joint/docs/tools/e4_numbers.py`
- `hpc/joint/docs/tools/p5_numbers.py`
- `hpc/joint/docs/tools/check_notation.py`
- `hpc/joint/docs/tools/smoke_test_notation.py`

Everything else in the repository is not specified yet and is out of scope for
the tester: the rest of hpc/ (the DSN, the joint stack, the jobs), the other
number scripts (p0, p2-p4, p6, p7), the inventory tools and the top-level
scripts. The covered scripts read some of that code at the freeze commit;
that code is their input, not something under test here.

## 1. Scientific goal

The joint DSN-NPE documentation series (E0-E9 explanations, P0-P7 parameter
chapters, `hpc/joint/docs/00_INDEX.md`) states facts about the joint
summary-network / neural-posterior-estimation stack frozen at commit
`834eb41`. By convention 12 of E0, no number appears in a document without a
claim tag; a number tagged `[RAN]` was computed by a script, and the tag names
the script block that computes it (for example "`[RAN]` B4" in E1 means block
B4 of `e1_numbers.py`). The deliverable of each covered script is its printed
output: every `[RAN]` number of its document, in the order the document
quotes them. The notation checker enforces E0 convention 13 and the master
symbol table: every symbol a document uses in math is declared in E0 S1.

Consumers: the author and the readers of the documents, who rely on every
`[RAN]` number being reproducible from the repository by one command.

## 2. Mathematical formulation

Each script implements the derivations its document states. The equations
are in the documents and are not copied here; the tester reads them there.

- E1 (`E1_THE_PROBLEM.md`): arithmetic on the window grid (f_s = 1 / Delta t,
  the window length W in samples, the smoothing width in bins, windows per
  subregion, per culture and in the cohort), the parameter census (23 + 3,
  17 + 9), the log-axis rule applied to the bounds E1 quotes (E0 convention
  2; E1 eq. (E1.1)), the activity floor's kept share, and the permutation
  count B implied by a p-value floor 1 / (B + 1).
- E2 (`E2_NPE_AND_FLOWS.md`): the decomposition of the expected negative
  log-likelihood, eq. (E2.2), and the gain identity, eq. (E2.5), checked on
  two toys with every term in closed form, then evaluated on the project's
  objects (the r2 bank's bound -ln Pr(pass); the bench prior's differential
  entropy; the anchor of P4 eq. (P4.10)).
- E3 (`E3_THE_SUMMARY_NETWORK.md`): the simplex equiangular tight frame for
  C classes; the participation-ratio rank r_eff = (tr Sigma_z)^2 /
  tr(Sigma_z^2) of an embedding cloud with covariance Sigma_z, for a non-zero
  covariance; the zero sets of the composite metric loss as transcribed in P2
  eqs. (P2.3)-(P2.10); Fano-type label-ceiling numbers.
- E4 (`E4_JOINT_OBJECTIVE_AND_LOOP.md`): the sampling laws of the loop's
  three streams; what a loss weight reaches under AdamW and the gradient
  clip; exact binomial intervals for the fraction of negative gradient-cosine
  probes and the chance of observing none; the fixed-statistics summary's
  eight features on a default bench shard; the arms as configured.
- P5 (`P5_OPTIMISER_AND_SCHEDULE.md`): the AdamW update of torch 2.10.0
  (`_single_tensor_adam`, decoupled weight decay) and `clip_grad_norm_`,
  replicated in numpy; E4 imports these.

### Notation

Symbols are those of the master table, E0 S1 ("Notation and symbols"); the
covered scripts introduce none of their own beyond local variables.

| Symbol | Meaning | Type / shape | Units |
|---|---|---|---|
| `[RAN]`, `[REPO]`, `[KB]` | claim tags of E0 convention 12 | text | -- |
| Bk | block k of a number script, as named in its docstring | label | -- |
| FREEZE | the frozen commit `834eb41` the documents describe | git hash | -- |

## 3. Data contracts

| Item | Source / format | Shape and axis order | dtype | Units | Conventions |
|---|---|---|---|---|---|
| Script output | stdout, plain ASCII text, one section per block in the block order of the docstring | -- | text | as the document quotes them | numbers printed at least to the precision the document quotes |
| `[REPO]` inputs | read with `git show FREEZE:<path>`, or imported from the working tree only after `git diff --quiet FREEZE -- <path>` succeeds | -- | -- | -- | never read from the working tree without the identity check |
| `[KB]` inputs | constants in the script, each with the document and section it is quoted from | -- | -- | as cited | not re-derived |
| Exit code | 0 when every internal check passes; non-zero when a check fails or the freeze check fails (inferred) | -- | int | -- | -- |
| Checker input | Markdown files; master table = E0 "## 1. Notation and symbols" | -- | text | -- | fenced code and inline code are not scanned |
| Checker exit code | 0 when no residual symbol; 1 when any document has one | -- | int | -- | residuals printed with the line of first use |

## 4. Global conventions

- Units: those the documents state (seconds, Hz, samples, nats).
- Floating-point precision: float64 throughout, except where a script
  transcribes code that runs in float32 and says so.
- Randomness: every Monte Carlo or sampling step uses a
  `numpy.random.default_rng` with a fixed seed in the script; two runs print
  identical output.
- Dependencies: standard library, numpy, scipy, scikit-learn (E3). Torch-free
  by design: the scripts must run in a sandbox without torch. Torch code of
  the repository is read with `git show` and executed alone (AST extraction)
  or transcribed in numpy.
- Files are pure ASCII with LF line endings (E0 convention 13).
- Binding decisions:
  - E0 convention 12: no number without a claim tag; a `[RAN]` number is
    printed by the block its tag names.
  - E0 convention 2 (as annotated in E0 v1.8): the log rule does not apply to
    the three Weibull kernel axes, which are linear whatever their span.
  - The documents describe the code at FREEZE `834eb41`; a script that reads
    working-tree code must first establish it is identical to FREEZE.

## 5. Blocks

### Block 1: E1 numbers

- Module: `hpc/joint/docs/tools/e1_numbers.py`
- Public API: script, `python e1_numbers.py` from `hpc/joint/docs/tools`
- Inputs: the DSN refit config of the r2 encoder at FREEZE
  (`hpc/dsn/hpc/Config/refit_mea_joint_full_r2_l0_t82.json`, read with
  `git show`); `[KB]` constants with their sources
- Outputs: the `[RAN]` numbers of E1, blocks B1-B5
- Library calls relied on: `json`, `math`, `subprocess` (git)
- Custom code: arithmetic only
- Test oracles:
  - every number E1 tags `[RAN]` Bk equals the value block Bk prints, to the
    precision E1 quotes (exact for integers);
  - each such number recomputed independently from the derivation E1 states
    and from the FREEZE config (for example f_s = 1 / Delta t; W = window
    length times f_s; B = 1 / floor - 1 for the gate's floor 0.001996, up to
    the rounding E1 states);
  - every `[REPO]` value it prints equals the FREEZE file's field;
  - two runs give identical output.
- Data flow: none (reads FREEZE)
- Confirmed: inferred (drafted from the docstring and E1, not yet checked by
  the author)
- Status: runs, exit 0, under 1 s (2026-10-06, sandbox)

### Block 2: E2 numbers

- Module: `hpc/joint/docs/tools/e2_numbers.py`
- Public API: script, `python e2_numbers.py`
- Inputs: the r2 bank's row counts (`[KB]`); the bench prior built by the
  repository's `hpc/joint/stage1` code (imported only after the identity
  check against FREEZE); the DSN class-centre function read at FREEZE
- Outputs: the `[RAN]` numbers of E2, blocks B1-B5
- Library calls relied on: numpy, scipy (closed-form entropies,
  distributions)
- Custom code: the two toys
- Test oracles:
  - eq. (E2.2) and eq. (E2.5) hold on the B1 linear-Gaussian toy, closed
    form against Monte Carlo within the Monte Carlo standard error the script
    uses (state it);
  - B2 unit-interval toy, theta ~ U(0, 1), z | theta ~ Bernoulli(theta):
    the closed-form values, for example E_U[ln(theta (1 - theta))] = -2 per
    axis exactly;
  - every number E2 tags `[RAN]` Bk equals the value printed by Bk;
  - the B5 anchor agrees with P4 eq. (P4.10);
  - the script refuses to import `hpc/joint/stage1` when it differs from
    FREEZE;
  - two runs give identical output (fixed seeds).
- Data flow: none
- Confirmed: inferred
- Status: runs, exit 0, about 20 s (2026-10-06, sandbox)

### Block 3: E3 numbers

- Module: `hpc/joint/docs/tools/e3_numbers.py`
- Public API: script, `python e3_numbers.py`
- Inputs: `effective_rank` (three implementations: `hpc/dsn/metrics.py`,
  `hpc/joint/stage3/joint_diagnostics.py`, `hpc/joint/stage3b/encoder_probes.py`),
  `cluster_scores`, `_class_center_vectors`, all read at FREEZE and executed
  alone; the bench prior (as Block 2); a numpy transcription of the composite
  metric loss, P2 eqs. (P2.3)-(P2.10)
- Outputs: the `[RAN]` numbers of E3, blocks B1-B6
- Library calls relied on: numpy, scipy, scikit-learn
- Custom code: the loss transcription (the repository's loss is torch)
- Test oracles:
  - simplex ETF, for each C in {2, 3, 4}: unit vectors with pairwise cosine
    -1 / (C - 1), summing to zero;
  - r_eff = (tr Sigma_z)^2 / tr(Sigma_z^2): equals 1 for a rank-one cloud,
    equals d for an isotropic cloud in d dimensions, is invariant to scaling;
    on a constant cloud (zero covariance) each implementation's behaviour is
    as E3 reports it (finding F-bb);
  - the loss transcription agrees with P2's equations term by term on small
    hand-checkable inputs;
  - every number E3 tags `[RAN]` Bk equals the value printed by Bk;
  - two runs give identical output.
- Data flow: none
- Confirmed: inferred
- Status: runs, exit 0, about 2 min (2026-10-06, sandbox)

### Block 4: E4 numbers

- Module: `hpc/joint/docs/tools/e4_numbers.py`
- Public API: script, `python e4_numbers.py`
- Inputs: `run_joint_arms.grouped_split` and `run_joint_arms.arm_config`
  read at FREEZE; the runner's argument-parser defaults read with `ast`; one
  default bench shard built in a temporary directory with
  `build_latent_bank.py --provider bench` and read back with
  `latent_bank.concat_shards`; the AdamW step and clip coefficient imported
  from `p5_numbers.py` (Block 5)
- Outputs: the `[RAN]` numbers of E4, blocks B0-B5
- Library calls relied on: numpy, scipy (`scipy.stats` binomial / beta for
  the exact intervals)
- Custom code: a float64 numpy transcription of `FixedStatsSummary.forward`
  (`run_joint_arms.py:96-111`) with torch's semantics: `std` with Bessel's
  correction, `quantile` with linear interpolation
- Test oracles:
  - B3: exact (Clopper-Pearson) binomial intervals agree with
    `scipy.stats.binomtest(...).proportion_ci(method="exact")`; the chance of
    seeing no negative probe in n probes is (1 - f_-)^n;
  - B4: the eight features agree with numpy (`std(ddof=1)`,
    `quantile(method="linear")`) on the shard, and with torch if torch can
    be installed in the sandbox;
  - B2: open-loop toy -- the reached weight follows from the Block 5 AdamW
    step applied to the stated gradient sequences;
  - the temporary shard is created outside the repository and removed;
  - every number E4 tags `[RAN]` Bk equals the value printed by Bk;
  - two runs give identical output.
- Data flow: consumes Block 5
- Confirmed: inferred
- Status: runs, exit 0, about 15 s (2026-10-06, sandbox)

### Block 5: AdamW and clip replica (P5 numbers)

- Module: `hpc/joint/docs/tools/p5_numbers.py`
- Public API: `adamw_step(param, grad, m, v, t, lr, beta1, beta2, eps,
  weight_decay)`, `clip_coef(total_norm: float, max_norm: float = 5.0) ->
  float`, `coverage(n_rows, n_draws)`, `grouped_split_counts(n_groups,
  fracs=(0.7, 0.15, 0.15), seed=0)`; script `python p5_numbers.py`
- Inputs: numpy arrays (AdamW state), scalars
- Outputs: the updated parameter and moments; the clip coefficient; the
  `[RAN]` numbers of P5
- Library calls relied on: numpy
- Custom code: the replica of torch 2.10.0 `_single_tensor_adam`
  (decoupled weight decay branch) and `clip_grad_norm_` (torch is not a
  dependency of the docs tools)
- Test oracles:
  - `adamw_step` equals `torch.optim.AdamW` (torch 2.10.x, `foreach=False`)
    step by step on random small tensors in float64 for several steps,
    including bias correction at t = 1 and weight decay > 0;
  - `clip_coef` equals torch's coefficient min(1, max_norm / (total_norm +
    1e-6));
  - every number P5 tags `[RAN]` Bk equals the value printed by Bk.
- Data flow: feeds Block 4
- Confirmed: inferred
- Status: runs (2026-10-06, sandbox)

### Block 6: notation checker

- Module: `hpc/joint/docs/tools/check_notation.py`, smoke test
  `hpc/joint/docs/tools/smoke_test_notation.py`
- Public API: CLI `python3 check_notation.py --master <E0> --docs <md> ...`
  and `--master <E0> --self --list`
- Inputs: E0's master table (its rows' first cells hold one or more `$...$`
  spans, each declaring a bare symbol); documents in Markdown
- Outputs: per document an OK line or the residual symbols with the line of
  first use; exit 0 / 1
- Library calls relied on: standard library (`re`, `argparse`)
- Custom code: a TeX-span tokenizer and a canonical-form matcher (no library
  fits)
- Test oracles:
  - the matching rules of the module docstring: canonical form (whitespace
    removed, subscript before superscript, accent braces removed); declared
    symbols with substituted index-like scripts (`\chi^2_{26}` for a
    declared `\chi^2_d`); operators, relations, numbers and type tokens
    accepted, with their scripts still checked;
  - fenced code blocks and inline code are not scanned;
  - a document with an undeclared symbol exits 1 and names it; E1-E4 and
    P0-P7 at the tested commit exit 0;
  - `smoke_test_notation.py --docs-dir ..` passes (43 PASS on 2026-10-06).
- Data flow: none
- Confirmed: inferred
- Status: smoke-tested 2026-10-06 (43 pass)

## 6. Pipeline entry points

All commands from `hpc/joint/docs/tools`, in a clone with full history (the
scripts run `git show 834eb41:...`; a shallow clone will fail).

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python e1_numbers.py` | E1's numbers | as is | < 1 s |
| `python e2_numbers.py` | E2's numbers | as is | about 20 s |
| `python e3_numbers.py` | E3's numbers | as is | about 2 min |
| `python e4_numbers.py` | E4's numbers | as is | about 15 s |
| `python p5_numbers.py` | P5's numbers | as is | seconds |
| `python3 smoke_test_notation.py --docs-dir ..` | checker smoke test | as is | < 1 s |
| `python3 check_notation.py --master ../E0_READERS_GUIDE_NOTATION.md --docs ../E1_THE_PROBLEM.md ../E2_NPE_AND_FLOWS.md ../E3_THE_SUMMARY_NETWORK.md ../E4_JOINT_OBJECTIVE_AND_LOOP.md` | notation coverage | as is | < 1 s |

Setup: `pip install numpy scipy scikit-learn` (`--break-system-packages` if
needed). Torch only for the optional cross-checks of Blocks 4 and 5:
`pip install torch` from PyPI (the CPU index URL may be blocked).

## 7. Not executable in the sandbox

Nothing in the covered scripts needs the cluster. The `[KB]` sources that live
outside the repository (deck pack, usage notes, the plan's PDF) cannot be
checked; a `[KB]` constant is checked only when its cited source is in the
repository.

## 8. Open questions

- Should `[KB]` constants be checked against their sources when those sources
  are in the repository, or are they out of contract? (raised 2026-10-06)
- Exit codes on a failed internal check are inferred from the code, not
  stated in the documents. (raised 2026-10-06)
