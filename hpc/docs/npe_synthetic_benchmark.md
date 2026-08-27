# Stage 1 -- the synthetic full-pipeline benchmark

Companion to `NPE_TUNING_PROTOCOL_v1.md` (S3.8, Stage 1) and to
`npe_tuning_usage.md`. This stage runs the **whole tuner** against a problem
whose posterior is known **exactly**, before the real campaign spends any
budget. It needs no bank, no export, no encoder and no real data: it builds
its own ground truth from `gmm_benchmark.py`. If it fails, the fault is in
the tuner, not in the data -- which is the only circumstance under which
that can be said cheaply.

## Run it

```bash
# 1. plumbing only -- minutes. Proves every code path runs and wires up.
python3 run_synthetic_benchmark.py --quick --out-dir /tmp/bench_q

# 2. the real thing -- hours; this is a job, not a login-node command.
qsub -v OUT_DIR=synth_bench_01 jobs/npe_synth_bench.pbs

# 3. one shape only, or with the optional stricter checks
qsub -v OUT_DIR=sb,SHAPES="shape_A_repo_reference" jobs/npe_synth_bench.pbs
qsub -v OUT_DIR=sb,STRICT=1 jobs/npe_synth_bench.pbs
```

`--quick` is deliberately too small to converge. It gates the exit code on
the **structural** assertions only (ledger, warm start, gates-can-fail) and
reports the recovery assertions without enforcing them. Use it to check
plumbing after a code change; never as evidence the tuner is correct.

## What it asserts

| # | assertion | what it would catch |
|---|---|---|
| 1 | end to end: every evaluation lands in the ledger; the ledger round-trips; a reloaded warm start proposes something new | a search that silently loses or repeats work |
| 2 | the selected config beats a deliberately bad one on C2ST vs the exact posterior | a search that is not actually optimising anything |
| 3 | `Delta_hat` separates trained configs from the shuffled-pairs control | an information gain that does not measure information |
| 4 | **G1 rejects a prior-equal posterior; G2 passes it** | a battery that cannot fail anything, or two gates that are redundant |
| 5 | no mode dropped; mode weights recovered | a flow that collapses a multi-modal posterior to one mode |
| 6 | the same code runs at two different `(n_dim, n_obs)` with no edit | a shape assumption baked into the tuner |

**Assertion 4 is the one that matters most.** A gate that has never failed
anything is a gate nobody has tested. The GMM fixture's prior is a valid
*wrong* posterior, so applying the battery to it must produce exactly one
pattern: G1 fails, G2 passes. That both gates fire in *different*
directions is what proves they are not redundant, and it is what licenses
believing a G1 pass on the real bank later. The PBS job's header states
these two lines explicitly as the ones to look for.

## Budget

Defaults are anchored to the budget the repository's **own** Gaussian
recovery suite (`smoke_test_gmm.py`, `_train`) already demonstrates is
sufficient on this exact problem: 12000 training rows, 80 epochs, patience
12. `--n-rows 15000` leaves ~12000 training rows after the 80/10/10 split.

This was learned the hard way and is worth not relearning: at 960 training
rows and 15 epochs, **even the shuffled control had not converged** to its
own asymptotic optimum (the prior). Trained configurations and the null one
then scored almost identically, and the recovery assertions failed for want
of training rather than for want of correctness. Anything materially below
the reference budget above measures the epoch cap, not the tuner.

## Reading a failure

- **Assertion 4 fails (G1 passes the prior-equal posterior)** -- serious.
  The informativeness gate is not discriminating; do not run the real
  campaign, and do not trust any earlier G1 pass.
- **Assertion 4 fails the other way (G2 rejects it)** -- also informative:
  it means the SBC path is finding structure in exchangeable draws, which
  points at a bug in the ranking code rather than at the estimator.
- **Assertion 2 or 5 fails, everything else passes** -- usually budget.
  Check the printed epoch counts and per-member gains first; a wide
  per-member spread means undertrained or unstable members, which is a
  different diagnosis from a uniformly uninformative estimator.
- **Assertion 3 fails** -- the control and the trained configs are not
  separable. If the control's gain is clearly *positive*, that is the same
  split-leak signal the real pipeline's `status` command warns about.

## Two deliberate scoping decisions

**Mode centres are informational by default.** The protocol's assertion 5
is that multimodality is not *silently lost* -- the failure guarded against
is a flow reporting a single mode. Mode dropping and mode weight are
therefore the hard checks. Centre precision at a tight tolerance is a
different property (recovery quality) that the repository's own G5 already
tests at a budget chosen for it; enforcing it here too would duplicate that
test while making this one fail for reasons unrelated to what it exists to
detect. `--strict-recovery` opts in.

**G2/G3 are informational on the finalist by default.** A small-budget
finalist need not be fully calibrated for the *tuner* to be correct, which
is what this stage tests. `--strict-calibration` opts in.

## Why the benchmark needs its own prior and floor

`GMMBenchmark`'s prior is an unbounded Gaussian **mixture**, not the uniform
box the real pipeline assumes:

- `npe_tune_score.prior_floor` (eq. 5, `sum_k log(U_k - L_k)`) is exact only
  for a box. `npe_tune_benchmark.benchmark_prior_floor` estimates
  `L_0 = -E[log p(theta)]` by Monte Carlo instead, and **reports its
  standard error** rather than assuming it away.
- `z_score_theta="transform_to_unconstrained"` builds a bijection from the
  prior's *support*, i.e. from the box; it is the wrong choice for an
  already-unconstrained prior. The benchmark uses `"independent"`, matching
  the repository's own `smoke_test_gmm.py` and for the same stated reason.

Everything else -- splits, training, ensemble extension, the GP search, the
ledger, the gates -- is the **same code the real campaign runs, unmodified**.
That is the point: a benchmark that exercised a parallel implementation
would prove nothing about the tuner.

One implementation note worth knowing, because it is invisible until it
bites: `GMMBenchmark.torch_prior()` declares its wrapper class *inside* the
method body, so the class has `<locals>` in its `__qualname__` and cannot be
pickled -- and `torch.save()` on a trained posterior pickles its `.prior`.
Training succeeds and only saving fails. `npe_tune_benchmark` therefore
supplies a module-level, picklable replica, verified against the original at
`max|difference| = 0.0` (test S17).
