# hpc/dsn -- the Deep Summary Network, mirrored into this repository

Added 2026-09-19 (migration step 1). This directory is a byte-identical
mirror of `Main/` from `Leonardodm00/Deep-Summary-Network` at commit
`7afe1b885b486b60215aeaf5143b637705c5d69a` (tag `dsn-final-20260919`),
minus the exclusions listed below. `ORIGIN_MANIFEST.tsv` names, for every
file here, its sha256 and its source path in the DSN repository; that file
is the provenance record and is how a future reader proves nothing was
altered in transit:

    git -C <DSN clone> show dsn-final-20260919:<source_path> | sha256sum

The DSN repository is retired from this commit on. New work on the encoder,
its training loop, the burst generators and the cohort configuration lands
HERE. The real-data extractor (`Main/hpc/MultiChannel/` in the DSN repo) is
NOT here: it moves to `Sbi-extractor` in migration step 3.

## Why a mirror and not a curated subset

Two reasons, both checked before choosing this shape.

1. The standalone DSN training path (`run_optimization.py`, `search.py`,
   `search_persistence.py`, the Optuna search, `evaluate.py`,
   `data_splits.py`) is kept, not retired (decision 2026-09-19). Its import
   closure covers nearly all of `Main/`, so a subset would have been the
   whole tree minus a handful of files, with the risk of dropping one.
2. Every entry point self-locates. The job scripts accept being run from a
   directory that contains `config.py` (`run_refit.pbs:26-31`,
   `run_mea_joint_search.pbs:32-37`); the smoke-test runner sets
   `PYTHONPATH` to the absolute path of the directory containing
   `config.py` (`hpc/run_all_smoke_tests.sh:41-65`); modules import each
   other by bare name (`from backbone import ...`). None of that depends on
   the directory being called `Main`. So `hpc/dsn/` is `Main/` under a new
   name and everything runs unchanged from `cd hpc/dsn`.

Nothing in the mirrored files was edited (the only additions since step 1
are `hpc/run_smoke_all.pbs`, this README and `ORIGIN_MANIFEST.tsv`). The
joint stack (`hpc/joint/stage*/`) reaches this directory through
`hpc/joint/dsn_locate.py` since migration step 2; `DSN_MAIN_DIR` is no
longer read anywhere and is reported as ignored if set.

## Layout

Identical to the DSN `Main/`:

| here | what |
|---|---|
| `*.py` at this level | encoder (`backbone.py`), checkpoint I/O, config dataclasses (`config.py`, incl. `CohortConfig` at line 1252), data pipeline, losses, training, search, generators (`generate_burst_data.py`, `latent_burst_generator.py`), `make_mea_specs.py` |
| `Smoke_Tests/` | 37 smoke tests + `run_all_smoke_tests.py` (5 more `smoke_test_*.py` sit at this level) |
| `hpc/` | PBS job scripts, configs, `Config/`, `run_all_smoke_tests.sh`, env setup |
| `analysis/` | post-hoc analysis and figures |
| `Documentation/` | technical documents carried over verbatim |
| `runs_synthetic/` | the DSN's `Runs Synthetic/`, renamed because paths with spaces are forbidden in this repo (the two `.pbs` inside are identical to the `hpc/` copies; the three configs and the README differ and were kept for that reason) |
| `README_DSN_MAIN.md` | the DSN's own `Main/README.md`, renamed so this file could exist |

## Excluded, and why

| DSN path | reason |
|---|---|
| `Main/Colab_zips/` | two binary zips (bundles of files that are all here as source) |
| `Main/hpc/MultiChannel/` | the real-data extractor; migrates to `Sbi-extractor` in step 3 |
| `Main/hpc/Config/factorial_configs.tar.gz` | binary; regenerable by `hpc/make_factorial_configs.py` |
| `Main/Runs Synthetic/` | renamed, not dropped -- see `runs_synthetic/` above |

Nothing else was left behind: 195 files here, 195 rows in
`ORIGIN_MANIFEST.tsv`, 0 blob mismatches at build time.

## Machine state this tree expects (gitignored)

Same as in the DSN repo, now covered by `hpc/.gitignore`:

- `hpc/dsn/out/` -- training runs. The r2 run behind every current result
  (`refit_mea_joint_full_r2_l0_t82`) stays where it is on the cluster,
  inside `~/"Deep Summary Network"/Deep_bio/Main_RETIRED_20260914/out/`;
  it is data, not code, and is never moved.
- `hpc/dsn/hpc/Config/npz_specs_mea.json` -- 315 machine-specific
  absolute paths; regenerate with
  `python make_mea_specs.py --config hpc/Config/config_mea_joint_full.davinci.json --dry-run`
  from inside `hpc/dsn/`.
- `config_*_lane[0-9]*.json`, `cache_*/`, `checkpoints/`, `figures/`.

## Environment

`train.py` imports `pytorch_metric_learning`, which `sbi_env` does not list.
Until the environment is consolidated (migration step 5), run this tree under
`meacnn_cpu` exactly as before, from `cd hpc/dsn`.

## Verification of this step

Sandbox (2026-09-19): 195/195 files byte-identical to the frozen DSN blobs;
`py_compile` on all 96 `.py`; `bash -n` on all 31 `.sh`/`.pbs`; zero bytes
above 0x7F in any `.py`/`.sh`/`.pbs`/`.json`/`.yml` (four `Documentation/*.md`
carry mathematical symbols, verbatim from the DSN, and are not executed);
zero CR bytes anywhere. Of the 42 smoke tests, the 9 that need neither torch
nor `pytorch_metric_learning` nor `skopt` pass in the sandbox
(`batch_geometry`, `generate_mc`, `inspect_latent`, `metrics`,
`objective_wiring`, `removed_modules`, `silhouette_floor`,
`latent_and_objective`, `search_persistence`); the other 33 need the
cluster environment.

Cluster [CLUSTER 2026-09-19]: the DSN smoke suite was run as a batch job
(`hpc/run_smoke_all.pbs`, `meacnn_cpu`, dvnode001) from this directory and
from the old tree `~/dsn_git/Main` in the same window, jobs 1722691 and
1722692: **30/30 suites passed in both**, 261.5 s vs 261.9 s, identical
environment (torch 2.13.0+cu130, pml 2.9.0, skopt 0.10.2), `runner_exit=0`.
The old tree stood at DSN `1e369e8`, one commit behind the mirror source
`7afe1b8`; the only file differing between those two commits is
`Main/hpc/MultiChannel/run_extractor_array_mea.pbs`, excluded from the
mirror and not exercised by the suite, so for every file the suite touches
the two trees were byte-identical. Equal counts close step 1.

To repeat the check after any change here:

    cd ~/SBI/hpc/dsn && mkdir -p out && qsub hpc/run_smoke_all.pbs

then read `out/dsn_smoke.log`: the `[job] torch ...` line must appear (the
prerequisites were real) and the last lines must read `30/30 suites passed`
and `runner_exit=0`. Never run `hpc/run_all_smoke_tests.sh` on the login
node: it is a 2 h allocation's worth of work and produces nothing comparable.
