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

Eleven mirrored files have been edited since. `ORIGIN_MANIFEST.tsv` marks
each of their rows EDITED, with the current sha in the row and the source sha
in the note. [Corrected 2026-09-28: this paragraph said two; the seven
step-5b rows had kept the source sha, and nothing checked them.]

- Seven job scripts in `hpc/`, on 2026-09-20 (migration step 5b, `fe9cb20`):
  one line each, the `DSN_CONDA_ENV` default `meacnn_cpu` -> `sbi_env` (see
  Environment below).
- `config.py` and `make_mea_specs.py`, on 2026-09-21 (Stage D), by MOVING
  text, not rewriting it: `CohortConfig` and the extraction-output helpers
  left them, verbatim, for the new torch-free `cohort.py` (a NEW row); each
  original re-imports what it lost, so there is still one definition of
  every name.
- `Smoke_Tests/run_all_smoke_tests.py` (a suite registered, a warning added)
  and `Smoke_Tests/smoke_test_mea_specs.py` (check A2 added), on 2026-09-21
  (Stage D).

Every other mirrored file is still byte-identical to its source blob. The
other additions since step 1 are `hpc/run_smoke_all.pbs`, this README,
`ORIGIN_MANIFEST.tsv`, and (2026-09-28) `verify_origin_manifest.py` with its
test `smoke_test_verify_origin_manifest.py`. The verifier checks every row
against the file it names -- the current sha, not the source blob -- and
runs in `hpc/run_smoke_all.pbs` before the suite:

    cd ~/SBI/hpc/dsn && python3 verify_origin_manifest.py
    cd ~/SBI/hpc/dsn && python3 smoke_test_verify_origin_manifest.py   # its own test, 10 checks

It prints `[manifest] PASS 196 row(s), ...` or one line per changed or
missing file, and exits 1 on any. Editing a mirrored file therefore means
refreshing its row and adding an EDITED note in the same commit. The
joint stack (`hpc/joint/stage*/`) reaches this directory through
`hpc/joint/dsn_locate.py` since migration step 2; `DSN_MAIN_DIR` is no
longer read anywhere and is reported as ignored if set.

## Layout

Identical to the DSN `Main/`:

| here | what |
|---|---|
| `*.py` at this level | encoder (`backbone.py`), checkpoint I/O, config dataclasses (`config.py`; `CohortConfig` is defined in `cohort.py` since Stage D and re-imported at `config.py:1253` [corrected 2026-09-28: this said "at line 1252" of `config.py`]), data pipeline, losses, training, search, generators (`generate_burst_data.py`, `latent_burst_generator.py`), `make_mea_specs.py` |
| `Smoke_Tests/` | 37 smoke tests + `run_all_smoke_tests.py` (5 more mirrored `smoke_test_*.py` sit at this level, and since 2026-09-28 a sixth that is this repo's own, `smoke_test_verify_origin_manifest.py`) |
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

Two environments can run this tree: `meacnn_cpu`, which produced every
result so far, and `sbi_env`, the one the rest of this repo runs in.
Migration step 5 (2026-09-19) consolidates on `sbi_env`: `train.py` needs
`pytorch_metric_learning` and `search.py` needs `scikit-optimize`, both
already installed in the cluster's `sbi_env` and, since step 5, listed in
`hpc/environment.yml` / `hpc/requirements.txt` so a rebuild keeps them.

The consolidation is decided by one run, the same 30-suite job under the
other environment:

    cd ~/SBI/hpc/dsn && mkdir -p out && qsub -v ENV_NAME=sbi_env hpc/run_smoke_all.pbs

`[job] env=sbi_env` and `30/30 suites passed` close it, and the job scripts'
`DSN_CONDA_ENV` default flips from `meacnn_cpu` to `sbi_env` (step 5b);
anything less keeps `meacnn_cpu` as the default and the difference is
diagnosed before anything is flipped. Result: `env=sbi_env`, **30/30 in
267.6 s**, `runner_exit=0`, job 1722779 [CLUSTER 2026-09-20; filled in
2026-09-28 from the migration handoff]; step 5b applies the flip: the seven
`DSN_CONDA_ENV` defaults in `hpc/` and `run_smoke_all.pbs` now say
`sbi_env`. The 11 synthetic experiment scripts that hard-code `conda
activate meacnn_cpu` (7 `hpc/dsn_4c_*.pbs`, 2 `hpc/dsn_latent_*.pbs`, 2
`runs_synthetic/*.pbs`) [corrected 2026-09-28: not nine] are 2026-08
history and were left as they are. The env name is a literal there, so `-v`
cannot change it [corrected 2026-09-28: this said "pass `-v` or edit"]; edit
the line, and refresh that file's manifest row, before reusing one.

One thing the two environments share: on davinci the system
`/lib64/libstdc++.so.6` lacks `GLIBCXX_3.4.26`, and both environments rely
on their `activate.d` hook putting `$CONDA_PREFIX/lib` first on
`LD_LIBRARY_PATH` (`hpc/setup_env_davinci.sh:151-166`). Calling an
environment's python by absolute path skips the hook and scipy fails to
import; always `conda activate`, or submit a job that does.

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
prerequisites were real), `31/31 suites passed` must appear near the end
(30 until Stage D registered `smoke_test_mea_specs.py`; the runner's note on
the suites that are in no `ORDER` follows it), and the last line must read
`[job] runner_exit=0 manifest=PASS ...`. Never run `hpc/run_all_smoke_tests.sh` on
the login node: it is a 2 h allocation's worth of work and produces nothing
comparable.
