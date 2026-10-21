# Stage C -- the manifest-aware exports. Handoff for a fresh chat.

| date | change |
|---|---|
| 2026-09-21 | v1. Written the day Stage D finished. Everything below is either measured on davinci that day, read from the code that day, or flagged. |
| 2026-09-24 | v2. **The cohort manifest is READ** (C0, values). **C5 is SETTLED** by the user (decisions SC-D1..D5) and C4's order gains the sim re-extraction as branch A (C4, C8). **C2a is FIXED** and `LABEL_AXES` is **required** in code (commits C2a, C2b, built and sandbox-verified, NOT YET PUSHED). C6 corrected (the `n_e` heterogeneity trap belongs to the Giulia project's MEA root, not to DUP15HD) and extended (the manifest's path has a space; most likely nothing refuses the pre-Stage-D archives a stale specs file would feed in; the MEA launcher's env is chosen by the launching shell). C0's commands corrected (`make_mea_specs.py --config` is required; the config is edited on the laptop, not on davinci). Corrected the same day after an independent review: the `LegacyArchive` protection that v2's first draft relied on most likely does not cover `extracted/`, the manifest's recorded digests are provenance and not gates, `read_manifest` skips a missing sidecar, and v9's raw output is not on record. `HPC_PATHS.md` v2 and `JOINT_DSN_NPE_USAGE_v1.md` v1.2 rewritten for the post-migration layout. The stale laptop clone is checked clean; see C7 for its rename. |
| 2026-09-28 | v3. **The 09-24 open items, worked; nothing of it pushed yet** -- one bundle, `stagec_bundle_20260928.zip`, supersedes the 09-24 one. **C2c** (new, C2): the archive check refuses version-2 archives -- half of C1's proposed guard, BUILT; still no export calls the check (C6). **ANN launcher env fix** (C6, C8 step 2): built. **SBI manifest**: the 7 stale rows refreshed and a verifier in the DSN smoke job (C7). **Laptop** (C7): the stale Sbi-extractor clone renamed `[DEVICE 09-28]`; the other two stale clones left as they are (the user). **C8**: what the root-level record must also carry (the EAP library's sha256, the env's versions), and the open items routed to a read-only probe, `probe_davinci_20260928.sh`, whose output is not in yet. The pitch question stays with the user (C8). |
| 2026-09-29 | v3.1. The user's decisions now have a register of record, `claude/SBI_decisions_and_ideas_log.md`: SC-D1..SC-D7 below are mirrored there as D-011..D-017, and new decisions go there, not into this handoff. Nothing else changed. |
| 2026-09-30 | v3.2. C8's pitch question is settled: the recording device's pitch is fixed at 60 um `[STATED 2026-09-30]`, equal to the virtual probe's default and the extractor's `PITCH_UM`, so `--pitch` stays (log D-021). Still asked: whether the device's electrodes match `--edge` 25 um. The env choice is deferred by the user until the probe output is in. Nothing else changed. |
| 2026-09-30 | v3.3. **The 09-28 bundle is pushed** (15:33): Sbi-extractor `feat/real-arm-parity` `8c70181` C2a, `becedff` C2b, `63d2574` C2c; SBI `main` `834eb41`; ANN `main` `37bf9f8`. Each branch's tree equals the tested one `[GITHUB 2026-09-30, fetched and compared]`. The davinci pulls are not confirmed yet (the tree checks were not pasted). No ANN git clone exists at `~/ANN`, `~/repos/Astro-Neuron-Network` or `~/Astro-Neuron-Network` `[CLUSTER 2026-09-30]`; C8 therefore gains a proposal (run from a fresh clone). Marked `[pushed 2026-09-30]` where a line said "not yet pushed". |
| 2026-10-01 | v3.11. **The user's answers of 14:22** (log D-033, D-034). "Exporting the new simulations" meant the re-extraction alone -- "it is just re-extraction given the new environment" -- so **Stage C ends at C8's record**, `Outputs_v2/REEXTRACTION_RECORD.json`: status row J (the provisional sim-arm export on r2) is dropped and A3 moves with the later export; the "Stage C ends here" row, the deferred row, C4's diagram note and C8's step 4 are corrected (v3.9's reading of D-025 and v3.10's of D-032 had taken "export" as the embedding export). The C8 commits are pushed first and the raster commit lands on top of them (D-034); the push commands, the bundle's identity, the tree the raster commit is to be cut against (`fa4e4eb5a52ea040dee5d1f71143137bf6d6486f`) and the davinci sequence are in the new `claude/HANDOFF_2026-10-01_c8_push_and_run.md` (C8, its "Push order" paragraph). Nothing has run on davinci yet. |
| 2026-10-01 | v3.10. **C8 is BUILT** (12:34-13:05, sandbox): Sbi-extractor `sim_reextract/` -- plan, launcher, gate + record, gate job, a 23-check smoke test through the real `process_campaign.py`; ANN `submit_mea_array.sh` finds davinci's conda and writes a per-task `mea_env.json` (14-check test). Two commits in `c8_bundle_20261001.zip` (push scripts, the three ANN files to upload, the run sequence); nothing of it on the cluster yet. **The user's calls of 12:34** (log D-030..D-032): v10-v12 share the contract (the plan still checks it); the DSN retrain is deferred, later with optimization; Stage C ends with the re-extraction and the export of the new simulations -- no joint DSN+NPE training. C4's status list cut accordingly; what comes next is the Giulia project (log I-001, `claude/HANDOFF_2026-10-01_giulia.md`). Open: whether the export before the retrain, on r2, is wanted. **[corrected 2026-10-01 14:22 (v3.11): "export" meant the re-extraction alone, so Stage C ends at the record, and the question left open here is answered: no export before the retrain (D-033).]** |
| 2026-10-01 | v3.9. **The user's C8 calls** (12:19, log D-023..D-025): the jobs run in `sbi_export` (Claude's `base` proposal withdrawn), `--edge` stays 25 um, the campaign set is v1-v5, v7-v12 -- v10-v12 in. C8 updated accordingly (its last paragraph): the build adds davinci's conda to the job's lookup so `sbi_export` is really activated, the gate checks pitch and edge, 253 tasks, walltime sized per task. C3: `label_axes.json` is re-made over the new set. C4: a dated "what is left" list. State table: the DSN suite job, 31/31 with `manifest=PASS` `[CLUSTER 2026-10-01]`; `sbi_env` imports torch 2.10.0 (its `2.13.0+cpu` package record is a leftover, `HPC_PATHS.md` sec. 7). |
| 2026-10-01 | v3.8. **The probe and the gate ran** on `dvlogin02` `[CLUSTER 2026-10-01]`. State table: davinci holds Sbi-extractor `63d2574` and SBI `834eb41`, no tracked edits; the gate passes 12 of its 14 steps and fails none, a1/a2 skipped and covered by hand (P2 matches all sixteen tool files present to `37bf9f8`). C8's open items answered (its last paragraph): tools folder and version, one EAP library, the old jobs' env (`base` by `PATH`), v9 (raw output for 8 of 54 tasks; MEA-processed on 08-21, so SC-D3's "never MEA-processed" is corrected), one task without `manifest.json`; **Claude's env proposal**, `base` by `--conda-prefix`, the user's call. **New**: campaigns v10-v12 (39 tasks) exist, outside SC-D3's reading. C0 / C6: the one `extracted/` archive read is not version 2 -- it lacks six of the eight fields and is refused as `LegacyArchive`; the export path still calls no check. C3: what `label_axes.json` records. Still open: `--edge`, the campaign list, the env, `sbi_env`'s torch reading (`HPC_PATHS.md` sec. 7), the DSN smoke job. |
| 2026-09-30 | v3.7. **The ANN launcher env fix is on davinci** (17:15): the five `37bf9f8` files in `ANN/MEA_analysis/` hash to the tested ones and `smoke_test_mea_env.py` passes 12/12 under `sbi_export` `[CLUSTER 2026-09-30]` -- the gate's a1/a2, done by hand. They came as files delivered in the chat and uploaded by the user, not by v3.6's GitHub fetch; v3.6's "cannot be sent directly" was an over-read (corrected in C8). Nothing else changed. |
| 2026-09-30 | v3.6. C8: the five files do not come from the laptop -- files cannot be sent from the laptop to davinci directly `[STATED 2026-09-30]`. They are fetched on davinci from GitHub into `~/SBI`'s object store, extracted with `git cat-file`, checked against their full sha256 and only then copied in. Nothing else changed. |
| 2026-09-30 | v3.5. C8: the tools' cluster folder is `ANN/MEA_analysis/` `[CLUSTER 2026-09-30]` (`HPC_PATHS.md` sec. 4); its two launchers are one byte shorter than at `73c2fcd`, most likely the final newline, so the pre-copy check accepts either hash. Nothing else changed. |
| 2026-09-30 | v3.4. **C8: the fresh-clone proposal of v3.3 is withdrawn** -- the user wants no new folder on davinci (log D-022). The tools stay in the existing, non-git ANN folder; the five files `37bf9f8` changes are copied into it and checked by sha256, and step 3's record carries the tools files' sha256 instead of a git commit `[reasoning]`. Nothing else changed. |

Read `claude/HANDOFF_2026-09-19_migration.md` for how the repos got here and for
the scheduler facts (its S3.3). This document is only Stage C.

## Project naming -- DUP15HD

`DUP15HD` is the user's name for the project this pipeline serves: the real MEA
cohort (`DATA_C` control / `DATA_P` pathological) **and** the simulated
campaigns are both DUP15HD. Everything in the migration handoff, in
`EXTRACTOR_USAGE.md` and in `SBI_PIPELINE.md` is DUP15HD unless it says
otherwise. `[STATED 2026-09-21]`; the acronym is not expanded anywhere and is
not expanded here.

A second project, the **Giulia project**, exists and is out of scope for Stage
C `[STATED]`. It already has code in this repo -- `preflight_giulia_export.py`,
`smoke_test_preflight_giulia_export.py` -- and, by my inference from "Giulia's
simulations" (`HPC_PATHS.md` v1 sec. 4e) rather than a statement, its own
simulator tree and MEA roots on davinci (`HPC_PATHS.md` 4e-4f), which is why
the distinction is worth carrying: a file or a root named for Giulia (or
`hhgap`) is not DUP15HD.

## Decisions of 2026-09-24 (the user's)

**Since 2026-09-29 the register of record is
`claude/SBI_decisions_and_ideas_log.md`**, where these are D-011..D-017
(SC-D1 = D-011, ..., SC-D7 = D-017). The table is kept as written on
2026-09-24; later decisions are only in the log.

| id | decision | statement | status |
|---|---|---|---|
| SC-D1 | C5 (a): what is re-run | re-run the virtual-MEA detection, `process_campaign.py`, over the **existing raw simulations** -- no new simulator sweeps, so theta and the raw spike times do not change -- with the electrode count taken from the real arm | active |
| SC-D2 | C5 (b): why | "it might not change anything", but everything is re-run so that every file of the new bank was produced by the new pipeline and says so | active |
| SC-D3 | C5 (c): which campaigns | **all**. Read as: every DUP15HD campaign under `SIM_MAIN_DIR`, `campaign_cadex_rho1300v{1,2,3,4,5,7,8,9}` -- v9 included although it was never MEA-processed; whether v9 has raw `topo_*/iter_*.npz` at all is **not on record** `[TO VERIFY]` (`HPC_PATHS.md` 4a) **[corrected 2026-10-01: it has, for 8 of its 54 tasks, and v9 WAS MEA-processed, on 2026-08-21 -- 8 sweep folders in `Outputs/` (probe P5/P6); `campaign_cadex_rho1300v10`-`v12` also exist (P5), outside this list -- the log's Open calls]** **[2026-10-01: decided -- v1-v5, v7-v12, v10-v12 in (log D-025)]**; the Giulia trees excluded | active; the reading is mine, correct it if wrong |
| SC-D4 | C5 (d): where | left to me: **`/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2`** -- beside the root of record `Outputs/` (never overwritten), `_v2` as in `extracted_v2`, no whitespace so `qsub -v` can carry it | active; not created yet |
| SC-D5 | `n_e` of the sim arm | set from the real-arm manifest at re-extraction: `n_side = sqrt(electrodes_per_subset)` = 3 (manifest value 9). No profiling of the OLD outputs, since nothing is exported from them any more; `assert_sim_geometry` stays as the export-time gate | active. Changes C1's "n_e can only be CHECKED": for the new outputs it is CONFIGURED from the manifest, then checked |
| SC-D6 | `LABEL_AXES` | passed explicitly for every campaign set | active; **enforced in code by commit C2b** (no default anywhere) |
| SC-D7 | C2a | make `dt`, `sigma_sm` required in `iter_campaign_records` ("shouldn't make much of a difference" -- correct: the one caller passes both) | done in commit C2a |

## State at the start of Stage C

| | |
|---|---|
| `Simulation-Based-Inference` `main` | `49da750`, the Stage D cohort lift `[GITHUB 09-24]`; **+ the hpc/dsn manifest commit** built 09-28, in the bundle, not yet pushed (C7) **[pushed 2026-09-30 as `834eb41`, tree the tested one `[GITHUB 2026-09-30]`]** **[2026-10-01: on davinci -- `~/SBI` HEAD `834eb41`, no tracked edits (probe P1); the gate's s1/s2 PASS `[CLUSTER]`]** |
| `Sbi-extractor` `feat/real-arm-parity` | `442eaf4` on GitHub `[GITHUB 09-24]`; **+ C2a, C2b** (built 09-24) **+ C2c** (built 09-28), waiting for `push_sbix.sh` of the 09-28 bundle, which also accepts a clone where the 09-24 C2a/C2b were already applied **[pushed 2026-09-30: `8c70181` C2a, `becedff` C2b, `63d2574` C2c; tree the tested one `[GITHUB 2026-09-30]`]** **[2026-10-01: on davinci -- HEAD `63d2574`, no tracked edits, the C2c refusal present (probe P1); the gate's x1-x10 PASS `[CLUSTER]`]** |
| `Astro-Neuron-Network` `main` | `73c2fcd` `[GITHUB 09-24]`; **+ the virtual-MEA launcher env fix** built 09-28, in the bundle, not yet pushed (C6, C8) **[pushed 2026-09-30 as `37bf9f8`, tree the tested one `[GITHUB 2026-09-30]`; on davinci no git clone of this repo at `~/ANN`, `~/repos/Astro-Neuron-Network` or `~/Astro-Neuron-Network` `[CLUSTER 2026-09-30]`; none is to be made: the fix is copied into the existing ANN folder (log D-022)]** **[2026-09-30 17:15: on davinci in `ANN/MEA_analysis/` -- the five files hash to `37bf9f8`'s, `smoke_test_mea_env.py` 12/12 `[CLUSTER]`]** **[2026-10-01: probe P2 -- all sixteen tool files present match `37bf9f8` (the five byte for byte, the other eleven identical at both commits); `THEORY.md` absent `[CLUSTER]`]** |
| DSN suite | **31/31**, `runner_exit=0`, `env=sbi_env`, `manifest_rows=196`, job 1723902 `[CLUSTER 09-21]`. `manifest_rows` is a count, not a check: 7 of those rows are stale since step 5b (`HPC_PATHS.md` sec. 3); refreshed, and checked by the job, from the 09-28 SBI commit on. **[2026-10-01: the verifier passes on the login node -- the gate's s1 (its test) and s2 (`PASS 196 row(s)`) `[CLUSTER]`; the suite job itself not re-run yet]** **[2026-10-01, 12:11-12:18, `dvnode002`: the suite job re-run -- `git=834eb41`, `[manifest] PASS 196 row(s), every file present and unchanged`, 31/31 in 354.6 s, `runner_exit=0 manifest=PASS`, torch 2.10.0 / pml 1.6.3 / skopt 0.10.2; the six unregistered suites still named `[CLUSTER]`]** |
| extractor suite | 8/8, and it now runs `verify_origin_manifest.py` first `[CLUSTER 09-21]` |
| export suite | `PASSED 17 FAILED 0 SKIPPED 0` against the real registry `[CLUSTER 09-21]`; **18** tests after C2a (T8c) **[2026-10-01: PASS on davinci, T8c included (the gate's x5) `[CLUSTER]`]** |
| `smoke_test_cohort_manifest.py` | **15/15** `[CLUSTER 09-21]`; **17** after C2c (M16, M17) `[SANDBOX 09-28]` **[2026-10-01: PASS on davinci (the gate's x9) `[CLUSTER]`]** |
| the cohort manifest | `extracted_v2/cohort_manifest.json`, sha256 `c925c5106137fe82`, `cohort_manifest_exit=0` at 16:36:26 `[CLUSTER 09-21]`; **contents read 09-24** (C0) |
| the DUP15HD real cohort | 35 `ptrain_*` wells, `n_subsets = 9`, `electrodes_per_subset = 9`, `w_size = 0.01`, `gaussian_window = 0.02`, `mfr_threshold = 0.1`, `fs_raw = 10110.09` -- in the config AND, measured, in the manifest |

## C0. Flip the cohort of record

**The manifest, read 2026-09-24** (data inspected -- the user pasted it; after
CRLF -> LF it hashes to `c925c5106137fe82`, the digest written on 09-21, so it
is the file of record byte for byte):

| field | value |
|---|---|
| `manifest_version`, `extractor_version`, `extractor_commit` | 1, `run_channel_subset_extraction/3`, `442eaf4` |
| `preprocessing` | `w_size 0.01`, `gaussian_window 0.02`, `electrodes_per_subset 9`, `n_subsets 9`, `mfr_threshold 0.1`, `fs_raw 10110.09`, `index_base 1`, `grid_width 48` -- all eight equal the config |
| `derived` | `fs_ifr 100.0`, `sigma_sm_bins 2.0` |
| `n_wells`, `n_subsets_per_well`, `n_units` | 35, 9, 315 (17 control + 18 pathological wells, 9 archives each) |
| per well | `T_rec 1200.0`, `n_samples_raw 12132108` (= 10110.09 x 1200) in all 35; `n_present` 1724-2299 of the 2304-electrode grid (electrodes with a raster file; recorded, not gated) |
| `sha256` | `config_file 5fa69c26...`, `cohort_block 411e5f63...`, `extraction_flags_sh 8d4654ec...` -- `config_file` and `extraction_flags_sh` equal the sha256 of the committed files (`49da750` config, `442eaf4` flags); `cohort_block` equals `sha256(json.dumps(cohort, sort_keys=True))` recomputed from that config -- a digest of a serialisation, not of a file `[SANDBOX 09-24]` |
| `extract_root` | `/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/extracted_v2` -- note the space (C6) |

Full table: `HPC_PATHS.md` sec. 3b.

`extracted/` is still what the config declares; nothing downstream looks at
`extracted_v2/` until this is done. **Expect two recorded digests to go stale
here, by design:** `config_file` and `cohort_block` were taken from the config
with `extract_root = extracted/`, so after the edit they describe the
pre-flip config. They are provenance only -- `build_manifest` records them,
and nothing compares them then or reads them later (`HPC_PATHS.md` sec. 3b).
Do not "repair" them.

The edit is a commit, so it is made on the laptop, in the working SBI clone,
and pushed; the cluster only pulls:

    # laptop: hpc/dsn/hpc/Config/config_mea_joint_full.davinci.json
    #   "extract_root": "/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/extracted"
    #                -> "/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/extracted_v2"
    # commit that one file, push

    # davinci, login node:
    cd ~/SBI && git pull --ff-only
    cd ~/repos/Sbi-extractor/extractor && conda activate sbi_export && source ../env.sh
    python3 list_extraction_jobs.py --config "$SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json" --out-manifest /tmp/m.tsv --out-flags /tmp/f.sh && cmp /tmp/f.sh extraction_flags.sh && echo "flags unchanged"
    cd ~/SBI/hpc/dsn && conda activate sbi_env && python3 make_mea_specs.py --config hpc/Config/config_mea_joint_full.davinci.json --dry-run
    cd ~/SBI/hpc/dsn && python3 make_mea_specs.py --config hpc/Config/config_mea_joint_full.davinci.json

(v1 of this handoff ran `make_mea_specs.py` with no arguments; `--config` is
required `[CODE SBI@49da750]` -- that line would have stopped at argparse.)

`extraction_flags.sh` must come out **byte-identical** (the `cmp` above, the
form of the migration handoff S8b, which writes nothing into the tree) --
`extract_root` is not one of the eight `PREPROCESSING_FIELDS` and does not
reach the flags. If it changes, something else was edited too. Only the
config is committed; `extraction_flags.sh` is GENERATED and is never
hand-edited.

`npz_specs_mea.json` is gitignored machine state, regenerated not committed.
The real-arm export reads a specs file (`real_source.py --specs`, required);
`artifacts/specs_real.json` is a copy of the OLD `npz_specs_mea.json` and
lists `extracted/` archives. **Most likely nothing would refuse them**: by
the project record they are extractor-v2 archives, which carry all eight
preprocessing fields, and `assert_archive_matches_manifest` checks exactly
those eight (`HPC_PATHS.md` sec. 3b; probable from the code and the project
record, not verified on the files). **[corrected 2026-10-01: the one `extracted/`
archive read (`control/DATA_C_Batch3/ptrain_A1/trace_subregion_00.npz`, probe P8)
carries only `index_base` and `grid_width` of the eight -- no `fs_raw`,
`n_subsets`, `electrodes_per_subset`, `mfr_threshold`, `w_size`,
`gaussian_window`, no `manifest_version` -- so it is not a version-2 archive, and
the check refuses it as `LegacyArchive` (as does the gate's x10)
`[CLUSTER 2026-10-01]`. Nothing in the export path calls that check yet, so the
replacement below stays necessary.]** Replace the copy with the regenerated file before
the C4 real-arm export, and check it (both numbers must be as shown):

    cd ~/repos/Sbi-extractor
    grep -o '"path": *"[^"]*"' artifacts/specs_real.json | wc -l                    # 315
    grep -o '"path": *"[^"]*"' artifacts/specs_real.json | grep -vc '/extracted_v2/'  # 0

## C1. The manifest API Stage C must call

In `cohort_manifest.py`, verified by reading it 2026-09-21 and again 09-24.
Nothing here needs writing; Stage C is the wiring.

| call | arm | contract |
|---|---|---|
| `read_manifest(path)` | both | returns the manifest dict; verifies `<path>.sha256` **only if that file exists** (a copy without its sidecar loads unchecked) and checks `manifest_version`; raises on a digest mismatch |
| `assert_archive_matches_manifest(archive, manifest, what="archive")` | real | `archive` is a `trace_subregion_XX.npz` path, an open npz, or a dict of its keys. Every `PREPROCESSING_FIELD` must equal the manifest's; raises `ManifestError` naming every disagreeing field, `LegacyArchive` if any of the eight is missing. Returns the archive's preprocessing dict. At `442eaf4` it compares those eight fields only -- not `manifest_version`, `extractor_version` or the path -- so a pre-Stage-D (v2) archive at the same values passes. **From C2c** it also requires the archive's `manifest_version` to equal the manifest's: missing -> `LegacyArchive` (v1 and v2 archives), different or not an integer -> `ManifestError`. Still not the path |
| `sim_preprocessing_from_manifest(manifest)` | sim | `-> (dt, sigma_sm, n_e_expected, recorded)`. `dt`/`sigma_sm` are what the sim export must CONFORM to; `recorded` is the `RECORDED_SIM` dict for the sidecar |
| `assert_sim_geometry(manifest, n_e, what="sim campaign")` | sim | raises `ManifestError` when the campaign's `n_e` (`electrode_centers`) differs from `electrodes_per_subset`. Since SC-D5 the new detections are CONFIGURED to it at re-extraction (C8); this call stays the gate at export |

Field groups, from the same file (and written into the manifest's `gating`):

| group | fields |
|---|---|
| `GATING_REAL` | all eight `PREPROCESSING_FIELDS` + `n_units` + `manifest_version` |
| `CONFORMED_SIM` | `w_size`, `gaussian_window` |
| `ASSERTED_SIM` | `electrodes_per_subset` |
| `RECORDED_SIM` | `mfr_threshold`, `n_subsets`, `fs_raw`, `index_base`, `grid_width` |

`mfr_threshold` and `n_subsets` are RECORDED on the sim arm and never applied to
it -- the decision of 2026-09-21. `mfr_threshold` is an electrode-level filter
with no simulated analogue; a simulated recording is a single subregion.

**Two guards the wiring should add** [proposed 09-24; the per-archive
`manifest_version` half BUILT 09-28 as C2c, inside the check itself so every
caller gets it; the rest not built]: the
consumers refuse a manifest whose `<path>.sha256` is missing, instead of
loading it unchecked; and the real arm requires each archive's
`manifest_version` to equal the manifest's (a version-2 archive has none)
and each specs path to lie under `manifest["extract_root"]`. `GATING_REAL`
already names `manifest_version`; the per-archive check does not use it.

## C2. Where each arm reads its preprocessing TODAY, and what changes

Read from the code 2026-09-21.

| | today | after Stage C |
|---|---|---|
| sim arm | `example_export.py` `main()` passes `dt=dsn.w_size`, `sigma_sm=dsn.gaussian_window` -- **from the frozen checkpoint** | from `sim_preprocessing_from_manifest(manifest)`; `assert_sim_geometry` on the campaign's `n_e`; `recorded` into the sidecar |
| real arm | `real_source.py:243,313` validates the npz's `fs_ifr` against "the checkpoint's `1 / cohort.w_size`" and refuses on mismatch | `assert_archive_matches_manifest(npz, manifest)` per archive |
| DSN training loader | the config's `cohort` block | the manifest; a config still carrying the block must AGREE, else raise |
| checkpoint | stores the cohort block | also stores the manifest sha256 |
| shard sidecars | contract block | also the manifest sha256; export asserts ckpt digest == manifest digest |

### C2a. FIXED 2026-09-24 (commit C2a, not yet pushed) [pushed 2026-09-30 as `8c70181`]

`iter_campaign_records(..., n_electrodes=None, dt=0.02, sigma_sm=0.04, ...)`
defaulted to the extractor library's own `DEFAULT_W_SIZE` /
`DEFAULT_GAUSSIAN_WINDOW` -- the pair D-3 removed from `dsn_frozen.py`, twice
the DUP15HD cohort's `0.01 / 0.02`. Now `dt` and `sigma_sm` are required
keyword-only arguments; the function is a generator, so a missing one raises
`TypeError` at the call. Pinned by `smoke_test_sbi_export.py` **T8c** (static
signature check + a call in a child interpreter). `[SANDBOX 09-24]` both halves
of T8c fail on `442eaf4` and pass after; no other test's outcome changed.

### C2b. `LABEL_AXES` required -- SC-D6 in code (commit C2b, not yet pushed) [pushed 2026-09-30 as `becedff`]

`launch_sweep_exports.sh` refuses an unset or empty `LABEL_AXES` (exit 9)
before enumerating a task; `submit_sbi_export.sh` refuses (exit 9) instead of
defaulting to `<artifacts>/label_axes.json` (the r2 weibull freeze);
`example_export.py --mode campaign` refuses without `--label_axes` (exit 2).
`none` still selects the legacy 4-axis block, by name only. New suite
`smoke_test_label_axes_required.py` (12; 7 fail on `442eaf4`).
`EXTRACTOR_USAGE.md` goes v8.3 -> v8.6: the C2a commit also carries in the
09-21 v8.4 row that only the project-knowledge copy had, then v8.5 (C2a) and
v8.6 (C2b).

### C2c. The archive check refuses version-2 archives (commit C2c, 09-28, not yet pushed) [pushed 2026-09-30 as `63d2574`]

`assert_archive_matches_manifest` compared the eight `PREPROCESSING_FIELDS`
and nothing else, so an archive written by extractor version 2 -- the schema
of `extracted/` by the project record -- passed whenever its values equalled
the manifest's (C6). It now also requires the archive's `manifest_version`, a
`GATING_REAL` field the per-archive check had not used, to equal the
manifest's: missing -> `LegacyArchive`, different -> `ManifestError`, and
so is a value that is not a scalar integer (1.5, `True`, `"1"`: `int()`
would have read each as 1; an array of shape `(1,)`; added after the
independent reviews). New checks M16 (a version-2 archive at the manifest's
values, written as `.npz` and read by path, is refused) and M17 (another, or
a non-scalar-integer, `manifest_version` is refused), both failing on
`442eaf4`; `smoke_test_cohort_manifest.py` is 17
checks. `EXTRACTOR_USAGE.md` v8.7; the same commit corrects two stale rows
of `extractor/README.md` (the aggregation job is ordered by `afterok`, not
gated by it; the suite has 17 checks, not 14) and refreshes its
`extractor/ORIGIN_MANIFEST.tsv` row. `[SANDBOX 09-28]`. **Not wired**: no code
outside `cohort_manifest.py` and its suite calls the check (`git grep`,
09-28); the real arm calls it from the C2 table's "after Stage C" on.

**Delivery**: `stagec_bundle_20260928.zip`, which **supersedes**
`stagec_c2_bundle_20260924.zip` (its two commits are in it, unchanged). On
the laptop, one script per repository -- `sbix/push_sbix.sh` in
`/c/Users/leona/Sbi-extractor` (C2a + C2b + C2c from `442eaf4`, or C2c alone
where the 09-24 commits were already applied, pushed or not), `sbi/push_sbi.sh`
in `/c/Users/leona/Simulation-Based-Inference`, `ann/push_ann.sh` in
`/c/Users/leona/Astro-Neuron-Network` -- each checking the resulting TREE id
against the tested one before `git push` is run by hand. Then on the davinci
login node `davinci/hpc_gate_20260928.sh`: it pulls the three clones
(fast-forward only) and must end `VERDICT: PASS` (exit 0) over 14 steps --
the eight of the 09-24 gate (`x5_export` = `PASSED 18   FAILED 0   SKIPPED 0`,
the rho1300v3 dry run listing exactly 54 tasks), `x9` cohort manifest 17/17,
`x10` C2c on real archives (one `extracted/` archive refused, as version 2
or version 1; one `extracted_v2/` archive accepted), `s1`/`s2` the SBI
verifier (its test 10/10, the manifest `PASS 196 row(s)`), `a1`/`a2` the ANN
launcher test (12/12) and the five changed files' bytes. A skipped step (no
ANN clone found; no `artifacts/label_axes.json`; no archive under
`extracted/`) gives `VERDICT: INCOMPLETE` (exit 2), never PASS; any failure
`VERDICT: FAIL` (exit 1). A push script that stops before committing
anything says `REFUSED`; after a commit, `STOPPED`, with the command that
finishes or undoes it; a rerun that finds the commit but stale working
files says `STALE WORKING FILES` and exits 1 (committing over them would
revert the fix).
Rehearsed in the sandbox end to end (push scripts on fresh clones with
default, `core.autocrlf=true` and `core.fileMode=false`; the gate on
davinci-like clones at the old commits, 13/14 there, `x5` failing only for
torch and the simulator repo, which the sandbox lacks). An independent
review the same day found the README's unzip path broken and a handful of
overstatements and weak matches, and a second review confirmed the fixes and
found three more places where a message or a test could mislead; all fixed
and re-rehearsed before delivery. Details in the bundle's `README.txt`.

## C3. The coordinate gate (S6b of the migration handoff)

Rule (1) infers a modelling decision -- which coordinate inference happens in --
from a prior specification. Four ACTIVE axes sit exactly on its threshold
(`U_max`, `U_0_sr`, `alpha_syn`, `x0`, all at margin `+0.000000`; measured
`[CLUSTER 09-21]`, unchanged from 09-20).

Decided, not built: `artifacts/label_axes.json` **declares** the bank's theta
column contract; each campaign's `manifest.json` still **supplies** the
coordinate via `registry_from_manifest`, because that is what wrote the data;
at export the manifest-derived coordinate is **asserted against** the
declaration and a mismatch **raises**, naming the axes.

A frozen `coord` can only CHECK, never impose: `assemble_theta_A` slices
`theta` out of each npz and never re-transforms it, so the coordinate of stored
data was fixed by the simulator at generation time.

Scope, read 2026-09-21: `preflight_label_axes.py` writes only `topology_axes`
and `excluded_axes` (line 225-226) and imports neither the registry nor torch --
the `coord` field has to be ADDED. Then `example_export.py::_load_label_axes`,
`build_label_spec`, the sidecar, and a regeneration of `label_axes.json` on the
cluster.

`label_axes.json` is built from the RAW simulations (`preflight_label_axes.py
--sim_main <SIM_MAIN_DIR>`), which SC-D1 leaves unchanged, so it depends on
the campaign SET, not on the re-extraction's output. The r2 freeze was run
without `--campaigns` (`HPC_PATHS.md` v1 sec. 2), i.e. over the default glob
`campaign_*`, so it took in every rho1300 campaign that had raw output when
it ran (on or after 2026-08-21, when `preflight_label_axes.py` was added in
`d9839aa`) -- v9 included only if v9 had raw output then, which is not on
record. SC-D3 therefore need not change the freeze's input, but that is not
verified. **[2026-10-01, probe P9: the file records `campaign_glob`
`campaign_cadex_rho1300v*`, `created_utc` 2026-08-21T08:13:22Z and 1,513
topologies; v1-v9 hold 1,550 topology folders and v10-v12 another 528 (P5), so
v9 most likely entered the freeze and v10-v12 did not `[inference from the
counts]`. If the campaign set takes in v10-v12, the freeze's input changes.]**
**[2026-10-01: it does (log D-025), so `label_axes.json` is re-made over
v1-v5, v7-v12 once C3 is built -- after v10-v12's contract is read (C4 status,
A0).]**
It can be regenerated as soon as C3 is built.

**Explicitly not the fix:** editing `PARAM_BOUNDS` to move an axis off the
threshold, or moving rule (1)'s threshold.

## C4. The order of work, with the sim re-extraction in it

Nothing downstream of the export touches the checkpoint
`[KB -- EXTRACTOR_USAGE.md sec. 1]`, so changing the encoder means re-exporting
**both** arms, never patching a later stage. With SC-D1..D5 the work runs in
two branches that join at the sim-arm export:

```
A (sim detections):  C8 re-extraction -> Outputs_v2 + its completion gate
B (code + encoder):  C0 flip -> C1-C3 wiring (+ C2a/C2b/C2c pushed)
                     -> 1 retrain the DSN (run_refit.pbs, sbi_env, extracted_v2) -> new frozen checkpoint
                     -> 2 real arm: 315 archives -> sbi_real_cohort.parquet (specs file for extracted_v2, C0)
joined:              -> 3 sim arm: launch_sweep_exports.sh over Outputs_v2, LABEL_AXES explicit
                     -> 4 gate_run.py -> witness_run.py -> NPE
```

**[2026-10-01 14:22: the diagram is the whole order of work. Stage C is cut to branch A (D-032, D-033); branch B and the joined steps 3-4 are the deferred work, after the retrain's "second time" (D-031).]**

Step 3 is the first step that reads the campaigns' detections
(`mea_iter_*.npz`), so it needs all of A plus step 1's checkpoint; steps 1-2
never touch them, so A runs in parallel with B. (v1 of this list, and the deck
pack's slide F.7, left the re-extraction out or put it after step 4; either
would build the first sim export, the gates and the NPE on detections that
are then replaced.)

The r2 export of record was 54 shards / 86,251 rows and the real arm 1890
windows over 35 cultures `[KB -- SBI_PIPELINE.md sec. 5]`. No checkpoint trained
under `meacnn_cpu` is treated as comparable to one trained under `sbi_env`
(torch 2.13.0+cu130 / pml 2.9.0, measured 09-19, vs torch 2.10.0 / pml 1.6.3,
measured 09-20); the DSN suite passing 30/30 under both licenses the flip, it
does not certify numerical identity, and no comparison has been run.

**Status 2026-10-01 -- what is left**, after the user's calls of 12:34
and 14:22 (log D-030..D-032, and D-033, D-034; `[Claude]` = built and tested in the sandbox, then
delivered; `[davinci]` = run by the user):

| step | what | state |
|---|---|---|
| A0 | v10-v12's label contract against v1-v9's | **dropped as a step** -- the same, the user is sure (D-030); the C8 plan refuses the run if the files say otherwise |
| A1 | Build C8: plan, run, gate + record (C8) | **BUILT 2026-10-01** `[SANDBOX]`: `sim_reextract/` in Sbi-extractor, the ANN job's conda lookup and `mea_env.json`; `c8_bundle_20261001.zip` |
| A2 | Run C8: the two pushes and the pull, the three ANN files uploaded, both smoke tests, the plan, one test task, the array, the gate (the bundle's README; the commands are also in `claude/HANDOFF_2026-10-01_c8_push_and_run.md`) | `[davinci]`, next. **Stage C ends at this row's record (D-033).** The C8 commits are pushed before the raster commit (D-034) |
| A3 | `label_axes.json` re-made over v1-v5, v7-v12 (`preflight_label_axes.py --sim_main <SIM_MAIN_DIR>`); C3's coordinate declaration is not built, so this is the current tool's output, as the r2 freeze was | `[davinci]`, after A2's record. **[2026-10-01 14:22: outside Stage C by D-033's reading -- C8 does not read it, and it feeds the embedding export -- so it moves with that export. Claude's reading, not asked (log, Open calls).]** |
| J | **[corrected 2026-10-01 14:22: DROPPED from Stage C -- "exporting the new simulations" meant the re-extraction alone, D-033. No provisional export on r2 is made; this export waits for the retrain and the retrained encoder (deferred row below).]** Sim-arm export over `Outputs_v2`: `launch_sweep_exports.sh` with `MEA_ROOT=Outputs_v2`, `LABEL_AXES` the re-made file, `--ckpt` the r2 encoder -- **provisional** (D-003: re-done after the retrain), since no other frozen encoder exists before D-031's retrain; today's export takes `dt` / `sigma_sm` from the checkpoint (0.01 / 0.02, the manifest's values) and checks `n_e` only through the shard's `record_resolved_n_e`, the manifest wiring (C1-C3) being unbuilt | `[davinci]`; **whether this provisional export is wanted at all is asked** (log, Open calls) **[answered 2026-10-01 14:22: no, D-033]** |
| -- | Stage C ends here (D-032) **[corrected 2026-10-01 14:22: at A2's record, `Outputs_v2/REEXTRACTION_RECORD.json`, not after J -- D-033]**. Then I-001, the Giulia project (`claude/HANDOFF_2026-10-01_giulia.md`) | |
| deferred | B1 C0 flip, B2 C1-C3 wiring, B3 the DSN retrain **with optimization** (D-031), B4 the real-arm export, the joint DSN+NPE training **[2026-10-01 14:22, D-033: and A3 and J, the sim-arm export over `Outputs_v2` with the retrained encoder]** | after Stage C, in the retrain's "second time" |

Done: C2a-C2c on davinci, the gate's 12 runnable steps PASS (2026-10-01); the
ANN launcher fix (2026-09-30); the DSN suite with the manifest verifier, 31/31,
`manifest=PASS` (2026-10-01). Not blocking: the six unregistered DSN suites,
Sbi-extractor's local-only commit `4507c74` on davinci (`HPC_PATHS.md` sec. 1),
and the log's older open calls.

## C5. The simulated arm will be re-extracted -- SETTLED 2026-09-24

`[STATED 2026-09-21, emphatic]` the DUP15HD simulated campaigns WILL be
re-extracted; `[STATED 2026-09-24]` its scope is SC-D1..D5 above. What the
09-21 analysis established is still true as far as it goes:
`process_campaign.py` performs spike detection only, with no downsampling or
Gaussian-smoothing parameter; `dt` and `sigma_sm` enter at export. So on the
sim arm "extraction" and "IFR" are two stages: the re-extraction rewrites the
detections (`det_ch`, `det_t`, `electrode_centers`, ...), and the IFRs are
rebuilt from them at export with the manifest's `dt`, `sigma_sm`.

| question (v1) | answer |
|---|---|
| which stage is re-run | `process_campaign.py` over the existing raw simulations (SC-D1) |
| what changes | `n_e` is now taken from the manifest (SC-D5) -- 9, which is also `process_campaign.py`'s default (`--n_side 3`); detection parameters stay at their values. Possibly nothing numerically; provenance in any case (SC-D2) |
| which campaigns | all DUP15HD rho1300 campaigns, read as including v9 (SC-D3; v9's raw output not on record -- the probe's P5 counts it) **[2026-10-01: P5 -- v9 has raw output for 8 of its 54 tasks; v10-v12 also exist; the list is the user's to confirm]** **[decided 2026-10-01: v1-v5, v7-v12 (log D-025)]** |
| where | `ANN/MEA_analysis/Outputs_v2` (SC-D4) |

## C6. Traps carried into Stage C

| trap | basis |
|---|---|
| **CORRECTED 09-24.** v1 listed "`n_e` heterogeneity: at least one 1-electrode unit sits under `mea_out` under a 4-electrode root; profile the campaign set for `n_e` before launching 54 export jobs". `mea_out` is the **Giulia project's** hhgap MEA root, not DUP15HD's (`MEA_analysis/Outputs/`), so that unit says nothing about the rho1300 set -- the 54 jobs are rho1300v3's, whose `n_e = 9` was measured from `electrode_centers` (plan 6A-1; the r2 sidecars record `null`). And under SC-D5 the DUP15HD detections are re-made with `n_e` from the manifest, so there is nothing old to profile. What remains: the re-extraction's completion gate (C8) checks `electrode_centers` on every unit, and `assert_sim_geometry` checks again at export | `[KB -- EXTRACTOR_USAGE.md sec. 11: "1-electrode output under the 4-electrode root" of mea_out; HPC_PATHS.md 4f]` |
| **NEW 09-24. The manifest's path has a space** (`.../Deep Summary Network/.../extracted_v2/cohort_manifest.json`), and `launch_sweep_exports.sh`'s guard 0 refuses whitespace in its five named variables (`CKPT`, `SBI_HPC_DIR`, `SIM_MAIN_DIR`, `OUT_ROOT`, `LABEL_AXES`) because `qsub -v` cannot carry it; a manifest variable would have to join that list. The sim-arm export array cannot be handed that path. Remedy **[reasoning, not built]**: freeze a copy WITH its `.sha256` into `artifacts/cohort_manifest/` (space-free), as the checkpoint is frozen into `artifacts/frozen_dsn/`, and have the consumer refuse a copy without its sidecar (`read_manifest` would load it unchecked) | `[CODE Sbi-extractor@442eaf4, launch_sweep_exports.sh:108-120; cohort_manifest.py:308-326]` |
| **NEW 09-24, corrected the same day. Most likely nothing refuses the pre-Stage-D archives.** `artifacts/specs_real.json` (and `npz_specs_mea.json`, until C0 regenerates it) lists `extracted/`, whose archives are extractor v2 by the project record and so carry the eight preprocessing fields; `assert_archive_matches_manifest` checks only those eight, so at the cohort's values they would pass. The first draft of this row, and of C0, said they would be refused as `LegacyArchive` -- that holds only for version-1 archives. Regenerate after C0, check every path is under `extracted_v2/` (C0), and add the guard proposed in C1. **Update 09-28**: C2c makes the check itself refuse them; the trap stays armed for the export path until an export calls the check (C2c). The probe's P8 lists one `extracted/` archive's keys and runs the check on it; the gate's `x10` requires the refusal after the pull. **[corrected 2026-10-01: the archive P8 read is not version 2 -- it lacks six of the eight fields -- and is refused as `LegacyArchive`, which `442eaf4` would also have done; x10 PASS `[CLUSTER]`. The trap stays armed through the export path, which calls no check; whether every `extracted/` archive is like that one is not shown]** | `[CODE cohort_manifest.py@442eaf4:347-392; Sbi-extractor@1bf7f0f run_channel_subset_extraction.py:46-74, 127-129, 139, 152-160]`, `[KB -- SBI_PIPELINE.md sec. 5]`; the files' key lists **not verified** (one-line check in `HPC_PATHS.md` sec. 3b) |
| `LABEL_AXES` must be passed explicitly for every campaign set (SC-D6). The old default was the r2 **weibull** file, which trips the NaN guard in `assemble_theta_A` on every job of a **flat** set. **Enforced by C2b** once pushed: no default at any entry point | `[KB -- EXTRACTOR_USAGE.md sec. 7]` |
| `conn_prob` is causally LIVE for the flat (hhgap) campaigns and inert for the weibull (rho1300) ones, so the included-axis set differs between them | `[KB -- EXTRACTOR_USAGE.md sec. 5.1]` |
| `launch_sweep_exports.sh` refuses whitespace in the five variables its guard 0 names. The DSN path is space-free by construction; `CKPT`, `SIM_MAIN_DIR` and a future manifest variable are the ones to watch. `CAMPAIGN` and `MEA_OUT`, derived from `SIM_ROOT` / `MEA_ROOT`, are forwarded unchecked | `[CODE launch_sweep_exports.sh@442eaf4:108-120]` |
| `submit_sbi_export.sh` builds its command line from a hardcoded `EXTRA=""`, so only `LABEL_AXES`, `MAX_RECORDS`, `SIMTIME`, `TRIM_HEAD_S` reach the exporter. Wiring the manifest into the sim export therefore needs a new variable there (e.g. `COHORT_MANIFEST`) | `[KB -- EXTRACTOR_USAGE.md]` |
| `afterok` on a PBS array is ORDERING, not success-gating, on davinci. Any Stage C job chained behind an array inherits this -- including the C8 re-extraction array | measured, migration handoff S3.3b |
| **NEW 09-24. The MEA re-extraction launcher takes its environment from the launching shell.** `launch_mea_array.sh` forwards any active env's `$CONDA_PREFIX` (base included), which then wins over `--conda-env` and is used with no activation (no `activate.d`, scipy may fail the job's preflight); only with no env active does the job fall back to `brian_env` through a conda hook evaluated under `set -euo pipefail`, output discarded. Neither route is known good for C8 as committed. **Fixed 09-28** (ANN commit, bundle, not yet pushed; **pushed 2026-09-30 as `37bf9f8`**): an explicit `--conda-env` / `--conda-prefix` wins, both together are refused, the ambient env is forwarded only when neither is given and the launcher says so; the job lifts `-u` around activation and activates a prefix through conda when it can find conda and the activation succeeds (else `PATH` first, with a WARNING). C8 must still NAME the env | `[CODE ANN@73c2fcd, hpc/MEA Traces/launch_mea_array.sh:67-68, 134-139; submit_mea_array.sh:34, 75, 86, 136, 154]`; `HPC_PATHS.md` 7.2 |
| Same class as C2a, not fixed: `example_export.py` falls back to `T_sim = 180.0` when `job_args.json` is missing and `--simtime` is not given, and to `conn_prob 0.1/0.6`. All 215 DUP15HD `job_args.json` declare `simtime = 200.0`, so it does not fire on this set; the launcher prints `(no job_args.json)` where it would | `[CODE example_export.py main(), 442eaf4]`; `HPC_PATHS.md` 4a |

## C7. Not checked, and out of scope

- The Giulia project's code paths. `preflight_giulia_export.py` and its suite
  exist and were NOT inspected during Stage D.
- Six DSN suites exist in `hpc/dsn/Smoke_Tests/` and are in no runner:
  `culture_grouping`, `joint_condition_space`, `joint_conditions_wiring`,
  `loss_type_wiring`, `search_dry_run`, `sep_warmup`. The runner names them at
  the end of every run since Stage D. Register them or move them to
  `run_tests.sh`; until then they are coverage that is believed and not held.
- **DONE 09-24**: `HPC_PATHS.md` (v2) and `JOINT_DSN_NPE_USAGE_v1.md` (v1.2)
  rewritten for the post-migration layout and written to project knowledge
  after the review. `EXTRACTOR_USAGE.md`: v8.3 on GitHub; the
  project-knowledge copy is v8.7, the C2c state (v8.6 on 09-24) -- ahead of
  the cluster until the 09-28 bundle is pushed. **[pushed 2026-09-30]**
  GitHub is at v8.7 (`63d2574`).
- `Leonardodm00/Deep-Summary-Network` still needs archiving on GitHub.
- **DONE 09-28 -- the stale laptop clone**: renamed to
  `/c/Users/leona/repos/Sbi-extractor_RETIRED_20260924` `[DEVICE 09-28 15:21]`
  after a re-check (`HEAD` `8ca6605` == `origin/main`, no local edits, no stash,
  no unpushed commits; 09-24: all 14 tracked files byte-identical to that
  commit, `COMMIT_EDITMSG` = `9b36df2`, on GitHub). The other two stale clones
  in `repos/` are left as they are (the user, 09-28): `Simulation-Based-Inference`
  (`feat/regions` at `8d9389c`, status, stash and unpushed commits empty
  `[DEVICE 09-28]`) and `sbi-export` (local-only, no remote). Two leftovers of
  the rename sit in `repos/` and are deletable: `retire_stale_clone_20260928.cmd`,
  `retire_report_20260928.txt`. `HPC_PATHS.md` sec. L.
- Three DSN job scripts keep `set -u` armed across `conda activate`:
  `hpc/dsn/hpc/run_search_9ch.pbs`, `run_aug_diag.pbs`, `gate_joint_search.pbs`
  (`HPC_PATHS.md` 7.2); the first two also hard-code a placeholder env name,
  and the third defaults to `brian_env`. `run_refit.pbs`, the one C4 step 1
  uses, is safe.
- **BUILT 09-28 (SBI commit, bundle, not yet pushed) [pushed 2026-09-30 as
  `834eb41`]**: the 7
  `hpc/dsn/ORIGIN_MANIFEST.tsv` rows stale since step 5b carry `# EDITED step
  5b` notes and their current sha; `hpc/dsn/verify_origin_manifest.py` checks
  every row and `run_smoke_all.pbs` runs it before the suite, ending
  `manifest=PASS|FAIL` (not fatal), as D-4 did for Sbi-extractor
  (`HPC_PATHS.md` sec. 3).
- L7/L8 local-calibration failures in `run_tests.sh`, pre-existing and unchased.
- No Stage 1 latent bank exists.

## C8. The sim re-extraction -- what the run needs (designed 2026-09-24; BUILT 2026-10-01)

**Built 2026-10-01** `[SANDBOX]`, as designed below with the day's decisions
in: Sbi-extractor `sim_reextract/` -- `sim_reextract_plan.py` (step 1),
`launch_sim_reextract.sh` (step 2, through ANN's `submit_mea_array.sh`),
`sim_reextract_gate.py` + `run_sim_reextract_gate.pbs` (step 3, the record
`REEXTRACTION_RECORD.json`), `smoke_test_sim_reextract.py` (23 checks, the
real `process_campaign.py` on a synthetic tree, a fake `qsub`), `README.md`
(the run order); `EXTRACTOR_USAGE.md` v8.8 sec. 14. ANN `submit_mea_array.sh`:
the conda lookup knows davinci's base and `module load miniconda3`, and every
task writes `mea_env.json` (versions, env, library sha256, tool hashes) before
it runs; `smoke_test_mea_env.py` 14 checks. Delivered as `c8_bundle_20261001.zip`
(two push scripts, the three ANN files for `ANN/MEA_analysis/`, the run
sequence). Deviations from the design: no `N_SIDE` variable -- the 102 August
jobs carried `--n_side 2 --pitch 200 ...` through `qsub -v` intact (probe P7),
so `EXTRA_ARGS` does, and it carries `--n_side 3 --pitch 60.0 --edge 25.0 --fs
10110.09` (D-015, D-021, D-024, the manifest's `fs_raw`); the record names the
tools' state by a fingerprint table (37bf9f8, the C8 state) and the tree id of
`hpc/MEA Traces`, not a commit, since the folder is no clone (D-022); the
walltime is one value per array, chosen from the test task's time per
iteration (there is no per-task walltime in a PBS array); a `_failures.log`
left by an earlier attempt is a warning in the record when every iteration is
done. Not run on davinci yet.

**Push order, D-034 (2026-10-01 14:22).** The C8 commits go first, unchanged: Sbi-extractor `feat/real-arm-parity` (`63d2574` plus one commit, tree `fa4e4eb5a52ea040dee5d1f71143137bf6d6486f`) and ANN `main` (`37bf9f8` plus one commit, tree `186ce230c675aab7206ef3d8d5b82ed4fed1b6df`); the raster commit lands afterwards, on top, and is to be cut against the Sbi-extractor tree above -- not against `63d2574`, and keyed on the tree, since the commit the laptop makes has its own id, not the sandbox's `87a225f`. `claude/HANDOFF_2026-10-01_c8_push_and_run.md` has the commands, the bundle's identity, the files a raster commit that also edits `EXTRACTOR_USAGE.md` or `.gitignore` must merge by hand, and the davinci sequence. The record that ends Stage C is `Outputs_v2/REEXTRACTION_RECORD.json` (D-033).


Everything here is **[reasoning]** from the code read on 2026-09-24
(`Astro-Neuron-Network@73c2fcd`, `hpc/MEA Traces/`) unless tagged. It is the
sim-arm analogue of Stage D: plan, trace, gate.

1. **Plan (preflight, login node).** `read_manifest` on the frozen copy of the
   cohort manifest (C6); `E = electrodes_per_subset`; `n_side = isqrt(E)`,
   refusing unless `n_side^2 == E` (the virtual probe is an
   `n_side x n_side` grid, `n_e = n_side^2` `[CODE mea_probe.py:14,48,67]`).
   Enumerate the raw sweep tasks of `campaign_cadex_rho1300v*` under
   `SIM_MAIN_DIR`; name every task without `manifest.json` (two of the 215 --
   `HPC_PATHS.md` v1 sec. 4a, counts undated, from its initial version; never
   identified `[TO VERIFY]`) and every campaign with no raw
   `topo_*/iter_*.npz` (v9 is the one not on record) instead of skipping them
   silently; refuse if `Outputs_v2` already holds output. **[2026-10-01, probe
   P5: one task has `job_args.json` and no `manifest.json`,
   `campaign_cadex_rho1300v7/sweep_cpu_task0009`; v9 has raw output for the 8 of
   its 54 sweep folders that carry both files, the other 46 carry neither; and
   the glob also matches `v10`-`v12` (11 + 11 + 17 tasks), outside SC-D3's
   reading -- the user's call]**
2. **Run.** `process_campaign.py` per task through the existing array
   launchers (`launch_mea_array.sh` / `submit_mea_array.sh`), `--n_side` from
   step 1, every other flag at its current value. `EXTRA_ARGS` is forwarded
   verbatim `[CODE]`, but it would carry `--n_side 3` -- a value with a space --
   through `qsub -v`; **[NOT VERIFIED]** that davinci's PBS Pro carries it
   intact. A dedicated `N_SIDE` variable in `submit_mea_array.sh` avoids the
   question. **The job environment must be made explicit first** (C6,
   `HPC_PATHS.md` 7.2): as committed, an active env in the launching shell is
   used without activation, and the by-name fallback runs a conda hook under
   `set -euo pipefail`. Proposed: `set +u` before the hook in
   `activate_env_by_name`, a launch from a shell with no env active and an
   explicit `--conda-env`, and the first task's `[mea-array]` preflight lines
   read before the array is trusted. **Built 09-28** (ANN commit, bundle):
   the `-u` lift around activation, and an explicit `--conda-env` now wins
   over an active env, so the launch needs the explicit flag, not a special
   shell. Reading the first task's preflight lines is still a step of the run,
   and it matters: the job finds conda on the node's `PATH`, through `module
   load anaconda3|conda|python`, or at a few standard install paths;
   davinci's own conda base (`/archive/apps/miniconda/miniconda3/py312_2`, named in `hpc/dsn/hpc/run_smoke_all.pbs`) and a `miniconda3` module are not among the places the job looks (pre-existing search, not changed on 09-28). If
   the probe's P7 shows the old jobs never activating, add those two before C8.
   **[2026-10-01, P7: the 110 logged jobs never activated -- at `73c2fcd` the
   prefix route only puts the forwarded env's `bin/` first on `PATH`
   (`submit_mea_array.sh:135-136`) -- and ran `base`'s python 3.10.14, numpy
   1.26.4, scipy 1.13.0 that way, which `base` survives. The two additions are
   therefore needed only for an env that must be activated (`sbi_env`,
   `sbi_export`: `HPC_PATHS.md` 7.1); with `base` by `--conda-prefix`, Claude's
   proposal (last paragraph), neither is.]**
3. **Trace + gate (aggregation job, after the array).** Because `afterok`
   does not gate on this scheduler, a separate check is the only success
   signal: every task's `mea_manifest.json` has `total_done == total_iters`;
   every `mea_iter_*.npz` has `electrode_centers.shape[0] == E` and a
   `meta_json` with `n_side == isqrt(E)`; then it writes a root-level record
   for `Outputs_v2` carrying the cohort manifest's sha256, the ANN commit and
   the task list -- and writes nothing, naming every bad task, if any check
   fails (the M15 rule of `cohort_manifest.py`). **Added 09-28**: the record
   must also carry the sha256 of the EAP template library the array used and
   the env's python / numpy / scipy versions, because nothing else will:
   `meta_json` is `PipelineConfig.to_dict()`, the pipeline knobs only -- no
   library, no env, no commit `[CODE ANN@73c2fcd, process_campaign.py:95-147,
   243]` -- and `launch_mea_array.sh` builds `./eap_library.npz` in whatever
   directory it is launched from when none is there
   (`launch_mea_array.sh:59, 106-112` `[CODE ANN@73c2fcd]`; lines 60 and
   118-125 after the 09-28 fix), so two launches can use two libraries
   without a trace.
4. **Then** C4 step 3 exports from `Outputs_v2` (as `MEA_ROOT`, with
   `SIM_ROOT` unchanged), `LABEL_AXES` explicit, `dt`/`sigma_sm` conformed to
   the cohort manifest. **[2026-10-01 14:22: not in Stage C -- that export waits for the retrain and the retrained encoder (D-033); Stage C ends at step 3's record.]**

Open before building (09-28: all but the last two are questions the
read-only probe `probe_davinci_20260928.sh` answers; its output is not in
yet): the cluster path of the ANN MEA tools and which version they are
(`HPC_PATHS.md` sec. 4 `[TO VERIFY]`; probe P2) -- **[2026-09-30]** not a
git clone at `~/ANN`, `~/repos/Astro-Neuron-Network` or
`~/Astro-Neuron-Network` `[CLUSTER]`. Because step 3's record must carry the
ANN commit, C8 has to run from a git clone; proposed, a fresh clone at
`~/repos/Astro-Neuron-Network` (where the gate looks), unless P2 finds a
clone elsewhere `[reasoning]` **[corrected 2026-09-30: withdrawn -- no new
folder on davinci, the user's call (log D-022). The tools stay in the
existing ANN folder, which is not a git clone and holds the results. The
five files `37bf9f8` changes (`launch_mea_array.sh`, `submit_mea_array.sh`,
`launch_mea_per_campaign.sh`, `README.md`, new `smoke_test_mea_env.py`) are
copied into the folder that holds `process_campaign.py`, from the laptop
clone, after a tar backup of the old ones, with CR line ends stripped and
each file's sha256 checked before (the `73c2fcd` values) and after (the
`37bf9f8` values); the other twelve tool files are identical at the two
commits. Step 3's record therefore carries the sha256 of the tools files and
the commit they match, not `git rev-parse` `[reasoning]`. The gate
`hpc_gate_20260928.sh` finds no clone and SKIPs a1/a2 (verdict INCOMPLETE);
the same two checks -- the sha256 and `smoke_test_mea_env.py`, ALL 12 CHECKS
PASSED -- are run by hand in that folder. Which folder it is stays open
(probe P2).]** **[corrected 2026-09-30: the folder is
`/davinci-1/home/ldellamea/ANN/MEA_analysis/`, beside `Outputs/`
`[CLUSTER, ls -l]`; its `launch_mea_array.sh` and `submit_mea_array.sh` are
one byte shorter than at `73c2fcd`, most likely the final newline, so the
pre-copy check accepts either hash (`HPC_PATHS.md` sec. 4).]** **[corrected
2026-09-30: not from the laptop clone -- files cannot be sent from the laptop
to davinci directly `[STATED]`. On davinci, `~/SBI` fetches ANN `main`
(= `37bf9f8`) from GitHub, at the URL of its own `origin` with the repository
name swapped; only `FETCH_HEAD` and objects under `~/SBI/.git` change, so no
folder is created and SBI's branches and files stay as they are. `git
cat-file blob` writes the five files to `~/ann37_<name>`, `sha256sum -c`
checks them against the full `37bf9f8` hashes, and only then are they copied
into `ANN/MEA_analysis/`; no CR stripping is needed. The fetch and the
extraction were rehearsed on a scratch repository against the local ANN
clone `[SANDBOX 2026-09-30]`.]** **[corrected 2026-09-30, 17:15: done, by
another route. The user uploaded the five files delivered in the chat; the
GitHub fetch was not used, and "files cannot be sent from the laptop
directly" over-read the user's words (`HPC_PATHS.md` sec. L). In
`ANN/MEA_analysis/`, after CR stripping and `chmod +x`: `launch_mea_array.sh` `41be351b08f3a6e9`, `submit_mea_array.sh` `b64aec71faf07f9d`, `launch_mea_per_campaign.sh` `fe046e1c12f9b7f1`, `smoke_test_mea_env.py` `832d8f785a700acd`, `README.md` `88d0f8de5986fd44` --
`37bf9f8`'s bytes -- and `smoke_test_mea_env.py` ALL 12 CHECKS PASSED under
`sbi_export` on `dvlogin04` `[CLUSTER]`: the gate's a1/a2, by hand. Not
shown: the old launchers' hashes before the copy, and the backup tar. The
other twelve tool files were not hashed, so the folder as a whole is not yet
shown to be `37bf9f8`; the probe's P2 fingerprints all seventeen.]**; which environment
`process_campaign.py` needs and which one produced `Outputs/` (probe P4, and
P7 from the old jobs' preflight lines -- the choice for C8 is then the
user's); which EAP library produced `Outputs/` (probe P3, P7); whether v9 has
raw simulations and which two tasks lack `manifest.json` (probe P5); and
whether the real MEA's electrode pitch should constrain `--pitch` (60 um
today) -- the user's question, which probe P11 can only inform by listing
the recordings' file types. The manifest records the electrode COUNT per
subregion, not the pitch, so SC-D5 fixes `n_e` and leaves the geometry's scale
where it is. **[settled 2026-09-30, log D-021]** The recording device's pitch
is fixed at 60 um `[STATED 2026-09-30]` -- the virtual probe's default and the
real extractor's `PITCH_UM` (`extractor/channel_subset_extraction.py:392`) --
so `--pitch` stays and C8 passes no pitch flag. Still asked: whether the
device's electrodes are 25 um wide, `--edge`'s default (the probe geometry
was the user's specification, ANN `THEORY.md`). **[2026-09-30]** The env
choice is deferred by the user until the probe output is in; Claude proposes
it then, from P4 and P7. **[2026-10-01] The probe's answers**
`[CLUSTER 2026-10-01, dvlogin02]`: the tools are `ANN/MEA_analysis/`, not a
clone, and all sixteen tool files present match `37bf9f8` (P2), so step 3's
record can name `37bf9f8` beside the files' sha256. One EAP library,
`ANN/MEA_analysis/eap_library.npz` (2026-07-24, sha256 `b50814ff27d3c496...`,
full value in `HPC_PATHS.md` sec. 4), named by every logged job (P3, P7). The
env: see step 2's note. **Claude's proposal: `base`, named with `--conda-prefix
/archive/apps/miniconda/miniconda3/py312_2`** -- the env and the route of all
110 logged jobs, no code change, its versions into step 3's record; `sbi_export`
would need a real activation and so one test task first, possibly the two
additions; never the job's default `brian_env`, which P4 shows damaged. The call
is the user's (log, Open calls). Which env wrote `Outputs/` v1-v8 (2026-07-24/25)
is not on record (no July job logs); v9's (08-21) was `base` `[inference from the
dates]`. v9 has raw output for 8 of its 54 tasks and was MEA-processed on 08-21
(P5, P6); one task lacks `manifest.json` (step 1). P11: the recording folders hold
only `ptrain_*.mat` files, so nothing on davinci records the electrode size;
`--edge` stays the user's question, and the existing `Outputs/` used 25 um (P6).
New: campaigns `v10`-`v12` (step 1). **[2026-10-01, the user's calls, log
D-023..D-025]** The env is `sbi_export` (D-023); Claude's `base` proposal is
withdrawn. `sbi_export` needs a real activation (`HPC_PATHS.md` 7.1), so the
build adds davinci's conda base
(`/archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh`) and
`module load miniconda3` to the job's conda lookup, as SBI's and
Sbi-extractor's job scripts have [proposed, not built]. `--edge` stays 25 um
(D-024), so step 3 checks `pitch == 60` and `edge == 25` in every `meta_json`.
The campaigns are v1-v5, v7-v12 (D-025): 253 tasks with `manifest.json`, the
exceptions named (step 1); v10-v12's label contract is read first (C4 status,
A0). Walltime: topology folders per task are about 7 in v1-v8 and v11, 11 in
v10, about 19 in v9 and v12 (P5); 7 of the 110 logged jobs hit the 6 h limit
(P7), and 2 of v9's 8 MEA sweep folders lack `mea_manifest.json` -- plausibly
those two `[inference]`. So the walltime is sized per task from its iteration
count, not set once `[reasoning]`.
