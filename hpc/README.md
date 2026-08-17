# SBI / NPE stage

Amortized Neural Posterior Estimation over the parameters of the
phenomenological neuron/synapse network model, conditioned on Deep Summary
Network embeddings of simulated MEA activity, and queried on embeddings of
real in-vitro recordings.

This directory contains the **inference** stage only. The simulator lives in
`Astro-Neuron-Network`; the encoder lives in `Deep-Summary-Network`. Nothing
here re-runs simulations or re-trains the encoder: the input is a table of
`(embedding, parameters)` rows and the output is a posterior.

---

## Layout

| File | What it does |
|---|---|
| `npe_contract.py` | Shard loading, the parameter coordinate system, the box prior, and the validation assertions. No networks, no training, no plotting. |
| `npe_model.py` | Flow construction, training, ensembling, persistence. No file IO, no diagnostics. |
| `gmm_benchmark.py` | A K-component n-dimensional Gaussian benchmark with an **exact** analytic posterior, for validating the estimator against a known answer. |
| `smoke_test_npe.py` | 9 tests over the contract and the estimator, including an analytic correctness anchor. |
| `smoke_test_gmm.py` | 6 tests: does the NPE recover three known Gaussians, with a negative control. |
| `recover_npz.py` | Rebuild the diagnostics arrays for a run made before they were saved. |
| `replot.py` | Regenerate figures/diagnostics from a finished run, in seconds. |
| `npe_plots.py` | Figures for every diagnostic layer; drawing only, computes nothing. |
| `npe_diagnostics.py` | Post-training diagnostics: embedding overlap (MMD), SBC, expected coverage, data-dependent SBC, TARP, contraction, information spectrum. |
| `smoke_test_diagnostics.py` | 8 tests validating the diagnostics against exact analytic posteriors. |
| `jobs/smoke_test.pbs` | PBS job running all three test suites plus an encoding guard. |
| `run_synthetic_trial.py` | Full end-to-end dress rehearsal on synthetic data at production shapes. |
| `jobs/replot.pbs` | PBS job for replotting, with optional recovery. |
| `jobs/synthetic_trial.pbs` | PBS job for the trial; every parameter overridable with `qsub -v`. |

The modules are deliberately flat and mutually independent so that swapping
the flow family, the ensemble size, or the data source touches one file.

---

## Install

```bash
conda create -n sbi_env python=3.11 -y
conda activate sbi_env
pip install -r requirements.txt
```

`sbi` and `zuko` are pinned. The code depends on API details that have moved
between releases -- in particular `posterior_nn(..., x_dist=prior)` and the
`"zuko_nsf"` model string. Verify with:

```bash
python3 -c "import sbi, zuko, torch; print(sbi.__version__, zuko.__version__, torch.__version__)"
```

---

## Run the tests

```bash
python3 smoke_test_npe.py            # full, ~10 min on one CPU core
python3 smoke_test_npe.py --fast     # plumbing only, ~5 s
python3 smoke_test_npe.py -k T6      # one test

python3 smoke_test_gmm.py            # full
python3 smoke_test_gmm.py --fast     # ground-truth checks only, ~30 s
python3 smoke_test_gmm.py --demo     # worked example with a results table
```

On the cluster:

```bash
qsub jobs/smoke_test.pbs
```

**Run this once by hand before the first `qsub`.** It is the check that
catches a transfer that mangled the source, which otherwise fails at job
time with a `SyntaxError` and costs a whole submission cycle:

```bash
python3 - *.py << 'EOF'
import sys
for p in sys.argv[1:]:
    d = open(p, 'rb').read()
    bad = [(i+1, hex(b)) for i, b in enumerate(d) if b > 127]
    print(p, 'ASCII OK' if not bad else 'NON-ASCII %s' % bad[:5], 'CRLF=%d' % d.count(b'\r\n'))
EOF
```

---

## Expected test output

Both suites pass in full. Reference numbers from a verified run:

```
T6_correctness_anchor    TV<=0.103 vs exact grid posterior
T7_ensemble_is_mixture   matches logsumexp - log n (dev 2.4e-07)
T8_end_to_end_shapes     p=27, E=12: 1.0000 of samples in-box
T9_sbc_ranks             KS p-values vs uniform: 0.296, 0.102

G1_analytic_is_correct   analytic == importance sampling
G2_nullspace_properties  posterior weights == prior weights to 2.2e-16
G4_c2st_recovery         C2ST 0.542-0.562 (0.5 is ideal)
G5_mode_recovery         all 3 modes found; max weight error 0.026
G6_negative_control      prior correctly rejected at C2ST 0.906

D5_blind_spot            SBC blind 12/12, coverage blind 12/12,
                         bilinear caught 12/12, FP 0/12
D8_information_spectrum  rank 3 of 6 params (observation dim 3)
```

---

## Data format

One Parquet shard plus a JSON sidecar per campaign. Column order comes from
the **sidecar**, never from the file, so a reordered export cannot silently
permute the parameter axes.

```
sbi_<campaign_id>_<shard>.parquet
    campaign_id   str
    topo_idx      int32
    iter_idx      int32
    seed_run      int64
    z_000 ... z_{E-1}                   float32   L2-normalised embedding
    th_<NAME> ... (p columns)           float64   inference coordinates
```

```jsonc
{
  "param_names":  [ /* p names, matching the th_* column order */ ],
  "coord":        [ /* "ln" or "linear" per axis */ ],
  "bounds_theta": [ /* p pairs [lo, hi] */ ],
  "embedding": { "embedding_dim": E, "dsn_checkpoint_sha256": "..." }
}
```

Real recordings use the same `z_*` columns with no `th_*` block.

`make_synthetic_shard()` in `npe_contract.py` writes a schema-valid shard
with a known forward map, so everything downstream can be developed and
tested before the real export exists.

---

## Design decisions, and why

**Parameter standardisation is `transform_to_unconstrained`, not z-scoring.**
With a box prior this maps the bounded support onto all of R^p, so the flow
cannot place mass outside the physical bounds. Test T8 measures 100% of
posterior samples in-box, which removes the sample-and-reject step entirely.
Plain z-scoring leaves the tails free to leak out.

**The prior reaches the flow builder as `x_dist`.** Inside sbi's low-level
builders the *estimated* variable is called `x` and the *conditioner* is
called `y`. For NPE the estimated variable is theta, so the prior over theta
is passed as `x_dist`. Putting it anywhere else silently leaves the
unconstrained transform unconfigured.

**Ensembles combine as the arithmetic mixture, never the geometric mean.**

```
q_bar(theta | z) = (1/n) sum_j q_j(theta | z)
                 = exp( logsumexp_j log q_j(theta | z) - log n )
```

The geometric mean `(1/n) sum_j log q_j` is a product of experts: it is
*sharper* than any member, so it would make overconfidence worse rather than
better, and it admits no closed-form sampler. sbi's `EnsemblePosterior`
implements the mixture; test T7 asserts this numerically rather than
trusting it, and confirms the two rules are distinguishable on the test.

**A "constant column" is judged against the prior width, not against zero.**
NumPy's `std` on a genuinely constant column returns ~1e-16, not 0.0, so an
exact-zero test never fires. This matters because a frozen axis leaking into
the label would otherwise surface as a divergent loss rather than an error.

---

## Validation strategy

The suites are built so that a pass means something:

- **An analytic anchor.** T6 trains on a linear-Gaussian problem whose
  posterior is computable exactly on a grid, and scores the NPE against it.
  This is the only test that validates the estimator rather than the
  plumbing.
- **The ground truth is itself verified.** The conjugate update in
  `gmm_benchmark.py` was derived, not copied, so G1 re-checks it numerically
  by importance sampling -- a different route to the same quantity. If G1
  fails, every other GMM test is scoring against wrong algebra.
- **A negative control.** G6 applies the G4/G5 metrics to a deliberately
  wrong posterior (the prior) and requires it to fail. Without this, a
  threshold loose enough for training noise might also pass an estimator
  that learned nothing.
- **Discrimination checks.** G3 confirms the modes are far enough apart that
  a unimodal fit cannot pass by accident, and that a generic-direction
  control breaks the null-space identity G2 relies on.

---

## Why the Gaussian-mixture benchmark is this particular problem

Three n-dimensional Gaussians are observed through a d-dimensional linear
map with d < n, and their means are placed **inside null(A)** -- the
subspace the observation cannot see. Two exact consequences follow:

- posterior weights equal prior weights exactly (measured to 2.2e-16);
- mode separation survives the update undiminished.

So the posterior is permanently 3-modal: no amount of data in the observed
directions collapses it. That is the controlled analogue of having fewer
informative embedding directions than parameters. An estimator that quietly
reports a single mode is wrong in a way that **calibration checks cannot
detect** -- SBC and coverage both pass on a mode-dropping estimator.

---

## The diagnostic blind spot

Modrak et al. (doi:10.1214/23-ba1404) prove that SBC with parameter-only test
quantities cannot detect a posterior that ignores the data, including one
exactly equal to the prior. Expected coverage does not rescue it either: if
q(theta | z) = p(theta) then log q(theta | z) = log p(theta) carries no
z-dependence, the credible regions are the prior's, and coverage is exactly
nominal.

Measured here over 12 seeds on a posterior set equal to the prior:

| check | detects it |
|---|---|
| marginal SBC | 0/12 -- blind |
| expected coverage | 0/12 -- blind |
| data-dependent SBC, f = theta^T W z | 12/12, median adjusted p 1.8e-06 |
| posterior contraction | ~0 by construction |

This matters directly: an insufficient summary makes the posterior partially
ignore the data, and that is precisely the failure the standard battery
cannot see. Always run `data_dependent_sbc` and `posterior_contraction`
alongside SBC, never SBC alone.

Note also that per-axis testing needs multiple-testing control: at 27 axes
and alpha=0.005 per axis, a perfectly calibrated posterior trips something
about 13% of the time. `family_verdict()` applies Holm-Bonferroni.

## Not built yet

- Posterior predictive checks (need the simulator callable).
- **The sim-vs-real embedding overlap test on YOUR data.** `embedding_overlap()`
  is implemented and tested; it just needs the real z vectors. This is the next thing to write
  and it is a hard go/no-go gate. The encoder was trained on real recordings,
  so it is the *simulations* that are out-of-distribution for it, not the
  reverse. An NPE trained on simulated embeddings will be perfectly
  calibrated on simulations and will report no warning about this. If the
  real embeddings fall off the simulated manifold on the sphere, the flow
  extrapolates and the posterior is arbitrary despite every diagnostic
  passing. The test needs only the query embeddings -- no training.
- Sequential (TSNPE) rounds. These need the simulator callable again;
  amortized NPE is self-contained on a fixed export.


## TARP reference points

`tarp()` accepts `Z` and `reference_mode`. This is not cosmetic. The theorem
behind TARP requires credible regions positioned at `theta_r(x)`, a FUNCTION
of the observation. Reference points drawn at random independently of x do
not satisfy that hypothesis, and the cost is total:

| reference points | detections on a data-ignoring posterior |
|---|---|
| random, x-independent | 0/12 (chance) |
| x-dependent readout of z | 12/12, median p = 3.8e-10, 0 false positives |

Always pass `Z=` when it is available. With `reference_mode="auto"` (the
default) the x-dependent version is used whenever `Z` is supplied, and the
result records which mode was actually used.

## Reading the information spectrum correctly

`information_spectrum()` reports the LINEAR rank of the covariance of the
posterior means. It is tempting to read this as a test of "an embedding of
dimension E constrains at most E-1 parameter directions". That reading is
wrong for a nonlinear estimator.

The map z -> E[theta | z] has an image of intrinsic dimension at most E-1,
because z carries only that many degrees of freedom. But a curved
d-dimensional manifold spans more than d linear dimensions. Measured
directly: for z on S^11, a linear f(z) gives covariance rank exactly 11; a
nonlinear f(z) on the same z gives 12 or more. A normalizing flow is
nonlinear, so an effective rank above E-1 is expected, not a defect.

What survives is information-theoretic: I(theta; z) <= I(theta; x) and z
carries at most E-1 real numbers, so the total information is capped however
it is spread. A nonlinear map can spread that budget thinly across all p
axes instead of concentrating it in E-1 of them.

Judge informativeness from per-axis `posterior_contraction`, and treat the
rank as descriptive. The bound is assertable in `smoke_test_diagnostics.py`
D8 only because that benchmark is conjugate linear-Gaussian, so its
posterior mean is affine and linear rank equals intrinsic dimension.


## Figures

`run_synthetic_trial.py` writes 12 PNGs to `<out-dir>/figures/`, numbered so
`ls` puts them in reading order, which is also decreasing consequence: if 01
is bad, nothing after it matters.

| file | what it answers |
|---|---|
| `01_embedding_overlap` | Is the query data inside the simulated cloud? THE gate. |
| `02_sbc_ecdf` | Are the marginals calibrated? |
| `03_sbc_histograms` | If not, which way -- over, under, or biased? |
| `04_coverage_pp` | Is the JOINT posterior calibrated (correlations included)? |
| `05_data_dependent_sbc` | Is the posterior using the observation at all? |
| `06_tarp` | Coverage without density evaluation. |
| `07_contraction` | Which parameters does the data actually inform? |
| `08_information_spectrum` | How fast does constrained variance decay? |
| `09_recovery` | True vs inferred per axis, with 68% intervals. |
| `10_posterior_pairs` | Ridges and multimodality, invisible in 1D. |
| `11_ensemble_spread` | How much do ensemble members disagree? |

### Regenerating figures without retraining

Every run writes `diagnostics_data.npz` alongside the summary. Posterior
sampling is the expensive stage; the rank statistics, coverage, TARP and
contraction that follow are cheap numpy. So:

```bash
python replot.py synthetic_trial_20260811_150000
python replot.py <run-dir> --n-forms 8 --out-dir /tmp/figs
```

runs in seconds and needs no model reload, no GPU, no re-sampling. Use it to
recover figures from a `--no-plots` run, to redo them after editing
`npe_plots.py`, or to re-read an old run's verdict without opening a job log.

If a run predates `diagnostics_data.npz`, recover it rather than repeating
the run:

```bash
python recover_npz.py <run-dir>     # needs ensemble/, i.e. --save-model
python replot.py <run-dir>
```

`recover_npz.py` reloads the shards, reproduces the exact calibration split
from the stored seed (verified bit-identical), and re-samples from the saved
ensemble -- one sampling pass, no training. Note that `--save-model` is the
default in the PBS job but NOT when running the script by hand, so a manual
run without it cannot be recovered.

One figure is not reproducible this way: 04 (expected coverage) needs
`log q(theta | z)` from the model itself, which is deliberately not stored --
persisting it would mean carrying the ensemble around. Replot writes 11 of
the 12 figures and says so.

Two design choices in figure 01 worth knowing, both made after the first
version was misleading:

- Panel 3 uses the whitened difference-of-means direction, NOT plain PCA.
  PCA picks directions of greatest SIMULATED variance, which need not be
  where the real data differs -- a query set the gate correctly rejected
  looked perfectly well mixed in the top two PCs. Note the asymmetry this
  creates: overlap in this view is strong evidence of no gap, separation is
  weak evidence, since a discriminative axis can overfit at small n.
- Panel 2 draws ECDFs, not histograms. If any query point coincides with a
  simulated one the distance is exactly zero, and a density histogram turns
  that into a spike that flattens everything else into the axis.
