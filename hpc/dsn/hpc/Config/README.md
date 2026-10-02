Config files for search

## config_giulia_cohort.davinci.json (2026-10-01)

The Giulia project's cohort config: the real recordings under
`/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/Bio_Data/`,
three classes by astrocyte:neuron plating ratio (`100N0A_wo` = 0:100 without
AraC, Batch1 + Batch2; `50N50A` and `70N30A` under `Batch3/AraC/`), 18 wells
at DIV 35. Read by the Stage D chain in Sbi-extractor (`list_extraction_jobs.py`,
`launch_stage_d.sh` with `COHORT_TAG=giulia`, `cohort_manifest.py`) through
its `cohort` block only.

What the cohort block states, and on what evidence (decisions D-040..D-051 of
the project's decisions log):

| field | value | why |
|---|---|---|
| `fs_raw` | 10000.0 | the device's rate, exact (D-048) |
| `grid_width`, `index_base` | 10, 0 | the files' suffix `_012`..`_087` is an MCS row/column code 10*row+col, rows and columns 1..8, corners absent; `row = k // 10`, `col = k % 10` decodes it exactly (D-048). Not a claim that the chip has 100 electrodes |
| `n_subsets`, `electrodes_per_subset` | 9, 1 | nine single-electrode subregions per well: the nine most active electrodes at or above `mfr_threshold` (D-043) |
| `w_size`, `gaussian_window`, `mfr_threshold` | 0.01 s, 0.02 s, 0.1 Hz | "the default", read as the DUP15HD values (D-044) |
| `ptrain_name_pattern` | `^ptrain_\d+_DIV\d+_\w+_nbasal_\d{4}_(\d{3})\.mat$` | the file names as listed (D-047); the one capture group is the electrode code |
| `ptrain_format`, `ptrain_varname` | `sparse_peaks`, `peak_train` | **a hypothesis until the probe confirms it**: the file sizes exclude dense rasters (D-047), and the naming is the SpyCode convention whose trains are sparse arrays under `peak_train` (Claude's recollection, unverified). Run `extractor/probe_ptrain_tree.py` first; set these two fields to what it reports |
| `exclude_wells` | `[]` | to be filled from the probe's census: wells with fewer than 9 electrodes at or above 0.1 Hz would make the extractor refuse the well and the manifest refuse the cohort |
| `extract_root` | `.../Giulia_Astro/extracted_giulia` | D-046 naming; the launcher accepts a first extraction into a declared root that does not exist yet |

Every block other than `cohort` (`data`, `search`, `train`, `backbone`,
`regularization`, `eval`, `runtime`) is COPIED from
`config_mea_joint_full.davinci.json`, the DUP15HD standalone-DSN search
config, with only `data.npz_specs`, `runtime.cache_dir` and
`runtime.experiment_name` renamed. Those blocks are NOT decided for the Giulia
project (D-039: the joint DSN+NPE stack is what is trained, and its
configuration is discussed afterwards); they are here so the file is a
complete `ExperimentConfig` for `make_mea_specs.py`, nothing more. Do not
read `search`/`train` here as the Giulia training plan.
