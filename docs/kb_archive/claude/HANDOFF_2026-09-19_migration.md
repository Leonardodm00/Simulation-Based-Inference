# DUP15HD -- handoff 2026-09-19/21: the DSN migration (steps 0-9) and Stage D

For a fresh chat. Supersedes `HANDOFF_2026-09-16_stageC.md` for everything it
covers: that document's S6 (Stage C) and S7 (Stage D) are **redesigned** here
(S6 below), its S1-S2 (repositories, cluster layout) are **obsolete** (S1-S2
below), its S3-S5 measured facts still stand, its S8-S10 still stand.

**Status as of 2026-09-21 16:50 CEST: the migration is COMPLETE and STAGE D
IS DONE -- code pushed, every gate green, and the re-extraction RUN.**

    extracted_v2/cohort_manifest.json   sha256 c925c5106137fe82
    [job] cohort_manifest_exit=0        finished 2026-09-21T16:36:26+02:00

SBI `main` = `49da750` (the cohort lift), Sbi-extractor
`feat/real-arm-parity` = `442eaf4` (D-1..D-6). Cluster gates, all
**[CLUSTER 2026-09-21]**: DSN suite **31/31** under `sbi_env`
(`runner_exit=0`, job 1723902); extractor suite 8/8; export suite
**17/17** with T9's census unchanged from 09-20; `smoke_test_cohort_manifest.py`
**15/15**; `verify_origin_manifest.py` 23/23; the extractor loads the cohort
block with `torch imported: False`; the flags regenerate byte-identical at 35
wells. Decisions settled today: `mfr_threshold` record-only on the sim arm;
`afterok` on an array is ORDERING, not gating (S3.3), so `cohort_manifest.py`
is the gate.

**NEXT: Stage C. Its own handoff is `claude/HANDOFF_2026-09-21_stageC.md`** --
start there, not here; this document is the history and the measured facts it
rests on. Still open and unrelated: archive
`Leonardodm00/Deep-Summary-Network` on GitHub (a UI click), and the six DSN
suites that exist on disk but are in no runner (S7).

**Update 2026-09-24.** The manifest's CONTENTS are now read (S3.1, last row;
full table in `HPC_PATHS.md` v2 sec. 3b). The sim re-extraction's scope is
settled (Stage C handoff, decisions SC-D1..D5). `HPC_PATHS.md` and
`JOINT_DSN_NPE_USAGE_v1.md` are rewritten for this document's S1-S2 layout,
so S7's first item is closed. Lines of S1, S2, S5, S6, S6b, S7 and S10 that
had gone stale since 09-21 -- or were already wrong then -- are corrected in
place, each marked `[corrected 09-24]` (S10's new bundle row `[added 09-24]`);
S3.1 and S4.2 gain one row each.

**Update 2026-09-28.** Built, sandbox-verified and **not yet pushed**, in
one bundle (`stagec_bundle_20260928.zip`, S10): the 7 `ORIGIN_MANIFEST.tsv`
rows S1 calls stale are refreshed and checked by a new SBI-side verifier;
Sbi-extractor C2c; the ANN launcher env fix (the last two are Stage C, see
its handoff). The stale laptop clone of S7 is renamed. Marked `[09-28]`
where they change a line here.

**Update 2026-09-30.** That bundle is **pushed** (15:33): Sbi-extractor
`feat/real-arm-parity` `63d2574` (C2a `8c70181`, C2b `becedff`, C2c
`63d2574`), SBI `main` `834eb41`, ANN `main` `37bf9f8`; each branch's tree
equals the tested one `[GITHUB 2026-09-30]`. The davinci pulls are not
confirmed yet. Marked `[09-30]` where they change a line here.

**PROJECT NAME.** Everything in this document -- the real MEA cohort
(`DATA_C` / `DATA_P`, 35 `ptrain_*` wells) AND the simulated campaigns -- is
the **DUP15HD** project `[STATED 2026-09-21]`. A second project, the **Giulia
project**, exists, is out of scope, and already has code in the extractor repo
(`preflight_giulia_export.py`, `smoke_test_preflight_giulia_export.py`), which
was not inspected during the migration or Stage D.

Tags: **[PUSHED]** on GitHub, commit named; **[CLUSTER]** run on davinci,
output seen; **[SANDBOX]** verified in Claude's sandbox against the real
repos; **[OPEN]**; **[NOT RUN]**.

---

## 1. Repositories -- what is on each branch

| repo | branch | on GitHub now | outstanding |
|---|---|---|---|
| `Deep-Summary-Network` | `main` | **`750d5bb`** retirement README; tag **`dsn-final-20260919`** -> `7afe1b8` **[PUSHED]** | archive it on GitHub (step 6) |
| `Simulation-Based-Inference` | `main` | **`49da750`** = the Stage D cohort lift, on `d460c78` = step 7 (`bc7d578` step 5, `fe9cb20` step 5b) **[PUSHED]**; **[09-30]** now `834eb41` (the `hpc/dsn` manifest refresh) on top | nothing |
| `Sbi-extractor` | `feat/real-arm-parity` | **`442eaf4`** = Stage D-6, on `f90fd29` = step 9 (`28b0625` 5b, `75d455f` 7, `be2b6e9` 8a, `64b01a4` 8b, `67c216e` 8c, `f2e158c` 8d) **[PUSHED]**; **[09-30]** now `63d2574` (Stage C C2a, C2b, C2c) on top | nothing |

Every patch was reproduced on a fresh clone of the pushed HEAD and compared
`diff -r` against the tree the gates ran on (a patch that applies is not a
patch that produces what was tested).

**The migration's shape.** `Deep-Summary-Network/Main/` was mirrored byte for
byte into `Simulation-Based-Inference/hpc/dsn/` (195 files; each row of
`hpc/dsn/ORIGIN_MANIFEST.tsv` is a path, its sha256 and its source path; 0
blob mismatches; the DSN smoke job re-counts them and printed
`manifest_rows=195` **[CLUSTER 2026-09-20]**; **[corrected 09-24]** true of the
mirror step, not of `49da750`: the 7 job scripts step 5b (`fe9cb20`) edited no
longer match their rows, and `manifest_rows` is a count, not a check --
`HPC_PATHS.md` sec. 3; **[09-28]** refreshed, and verified row by row by the
DSN smoke job, from the SBI commit of the 09-28 bundle on). Its `Main/hpc/MultiChannel/`
-- the real-data extractor -- moved to `Sbi-extractor/extractor/` (12 of 17
verbatim; `extractor/ORIGIN_MANIFEST.tsv` says which; the three edits are
spelled out in `extractor/README.md`). The joint stack reaches the DSN
through `hpc/joint/dsn_locate.py`; Sbi-extractor reaches it through
`dsn_tree.py` as `$SBI_HPC_DIR/dsn`. Dependency direction is **Sbi-extractor
-> Simulation-Based-Inference only**. `DSN_MAIN_DIR` resolves nothing any more
**[corrected 09-24: four places still read it, only to report it]**; where it
is still exported it is reported once as IGNORED.

## 2. Cluster layout

```
~/SBI/                          Simulation-Based-Inference@main. THE code tree of record.
~/SBI/hpc/dsn/                  the DSN (mirror of the retired repo's Main/). Standalone DSN
                                training (run_optimization.py, search.py, run_refit.pbs ...)
                                is KEPT here (decision 2026-09-19).
~/SBI/hpc/dsn/hpc/Config/npz_specs_mea.json   gitignored, regenerated (S3 of the 09-16 handoff)
~/repos/Sbi-extractor/          feat/real-arm-parity. extractor/ holds the real-data extractor.
~/repos/Sbi-extractor/artifacts/sbi_hpc -> ~/SBI/hpc     [CLUSTER] the one pointer
~/repos/Sbi-extractor/artifacts/dsn_main                  REMOVED by relocate_artifacts.sh [CLUSTER 09-20]
$HOME/dsn_main                                            ALSO REMOVED by the same script [CLUSTER 09-20]
~/.bashrc                       the DSN_MAIN_DIR default + export (lines 24-25) REMOVED
                                [CLUSTER 09-21]; backup at ~/.bashrc.bak_20260920. A shell
                                opened BEFORE that edit still carries the variable; dsn_locate.py
                                reports it IGNORED and resolves correctly regardless.
~/dsn_git_RETIRED_20260920/     the old DSN clone at 1e369e8, RENAMED [CLUSTER 09-21].
                                Nothing in it is data; deletable at will.
~/"Deep Summary Network"/Deep_bio/
    Main_RETIRED_20260914/out/  refit_mea_joint_full_r2_l0_t82 -- the r2 encoder. DATA. NEVER DELETE.
    extracted/                  315 (+35 combined) cohort archives. DATA. [corrected 09-24:
                                extractor v2 by the project record; keys never listed]
    extracted_v2/               the Stage D re-extraction + cohort_manifest.json (+ .sha256).
                                DATA.  [corrected 09-24: missing from this layout]
    DATA_C/, DATA_P/            raw recordings. DATA.
```

Environments after step 5b, now **CONFIRMED**: **`sbi_env`** for everything
in `~/SBI` including `hpc/dsn`; **`sbi_export`** for everything in
`~/repos/Sbi-extractor` including `extractor/`. `meacnn_cpu` is retired.
Both environments depend on their `activate.d` hook for `LD_LIBRARY_PATH`
(davinci's system libstdc++ lacks `GLIBCXX_3.4.26`); an interpreter called
by absolute path skips it and scipy fails to import **[CLUSTER 2026-09-19]**.
Always `conda activate`, or submit a job that does.

## 3. Measured facts of record

### 3.1 The migration runs

| quantity | value | source |
|---|---|---|
| DSN smoke suite, mirror vs old tree | **30/30 in both**, 261.5 s vs 261.9 s, same node, same env (torch 2.13.0+cu130, pml 2.9.0, skopt 0.10.2) | **[CLUSTER 09-19]** jobs 1722691 / 1722692 under `meacnn_cpu` |
| **DSN smoke suite under the NEW default `sbi_env`** | **30/30 in 267.6 s**, `runner_exit=0`, Exit_status 0, walltime 00:04:44, `git=d460c78`, `manifest_rows=195`; **torch 2.10.0, pml 1.6.3, skopt 0.10.2** | **[CLUSTER 09-20]** job **1722779**, `~/SBI/hpc/dsn/out/dsn_smoke.log` |
| **extractor suite under the NEW default `sbi_export`** | **8/8**, Exit_status 0, walltime 00:00:15, `git=f2e158c`; numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.1 | **[CLUSTER 09-20]** job **1722780**, `extractor/out/extractor_smoke.log` |
| `extraction_flags.sh` regenerated under `sbi_export` | byte-identical to the tracked file, `wrote 35 well(s)`, 35 manifest rows | **[CLUSTER 09-20]** |
| export suite after steps 4a/4b/8a/8b/9 | **`PASSED 16   FAILED 0   SKIPPED 0`** (T1, T2, T2b, T2c, T3-T8, T9, T9b, T9c, T10-T12) | **[CLUSTER 09-20]** `--sim_dir .../Phenomenological/Main` under `sbi_export` |
| `launch_sweep_exports.sh` DRYRUN | `campaign_cadex_rho1300v3`: **54 sweep tasks**, sims 913-1873 per task, `job_args.simtime=200.0` throughout, every `qsub` line carrying `SBI_HPC_DIR=` and `ENV_NAME=sbi_export`; nothing submitted | **[CLUSTER 09-20]** |
| `run_tests.sh` full sweep after step 2 | **12/13 stages**, only local calibration failing (L7, L8 -- pre-existing); J23 PASS, J29 PASS; `probe_dsn_runtime` `VERDICT: PASS (7/7)` naming `hpc/dsn` | **[CLUSTER 09-19]** |
| sim-arm IFR vs the extractor's recipe, pre-4b | un-normalised bit-identical (T2); **normalised differed at the last float32 bit on 50/50 random trials**, worst 3e-8; a spike at exactly t = T discarded here, counted there | **[SANDBOX]** T2b/T2c fail on the old code, pass on the new |
| **DSN smoke suite after the cohort lift** | **31/31 in 270.4 s**, `runner_exit=0`, `env=sbi_env`, `git=49da750`, `manifest_rows=196` (195 + `cohort.py`), torch 2.10.0 / pml 1.6.3 / skopt 0.10.2; the runner also NAMES six suites on disk that are in no `ORDER` | **[CLUSTER 09-21]** job **1723902** |
| **Stage D re-extraction** | launcher listed **35 wells** with flags byte-identical to the tracked file, `extract root = extracted_v2` (config still declares `extracted`); array **1723886[]**, aggregation **1723887**; `wrote .../extracted_v2/cohort_manifest.json sha256 c925c5106137fe82`, `cohort_manifest_exit=0` at 16:36:26 | **[CLUSTER 09-21]** |
| Stage D login-node gates | `torch imported: False` for `cohort_config`; flags regenerate byte-identical, 35 wells; `smoke_test_cohort_manifest.py` **15/15**; export suite **`PASSED 17 FAILED 0 SKIPPED 0`**; `verify_origin_manifest.py` PASS 23 rows | **[CLUSTER 09-21]** |
| `L8_lc2st_local_failure` symptom | back to "0% of corrupted observations" | **[CLUSTER]** it oscillates; still open, still not chased |
| **cohort manifest CONTENTS** (read 2026-09-24) | `manifest_version 1`; `extractor_version run_channel_subset_extraction/3`, `extractor_commit 442eaf4`; `preprocessing`: `w_size 0.01`, `gaussian_window 0.02`, `electrodes_per_subset 9`, `n_subsets 9`, `mfr_threshold 0.1`, `fs_raw 10110.09`, `index_base 1`, `grid_width 48` (all equal the config); `derived`: `fs_ifr 100.0`, `sigma_sm_bins 2.0`; `n_wells 35` (17 control, 18 pathological), 9 archives each, `n_units 315`; every well `T_rec 1200.0`, `n_samples_raw 12132108`; `n_present` 1724-2299 of 2304; `sha256`: `config_file 5fa69c26...`, `cohort_block 411e5f63...`, `extraction_flags_sh 8d4654ec...` -- `config_file` / `extraction_flags_sh` equal the committed files' sha256 (`49da750` / `442eaf4`), `cohort_block` equals `sha256(json.dumps(cohort, sort_keys=True))` recomputed from that config; `extract_root` `.../Deep Summary Network/Deep_bio/extracted_v2` | the user's paste; after CRLF -> LF it hashes to **`c925c5106137fe82`**, the digest of record, so it is the file byte for byte (data inspected); digests re-computed **[SANDBOX 09-24]** |

**The two environments carry materially different torch/pml versions** --
`meacnn_cpu` torch 2.13.0+cu130 / pml 2.9.0, `sbi_env` torch 2.10.0 / pml
1.6.3 -- and the DSN suite gives 30/30 under both. That licenses the flip,
which is what the suite is for; it does **not** certify numerical identity
across the two. The DSN will therefore be retrained under `sbi_env` /
torch 2.10.0, and no checkpoint trained under `meacnn_cpu` is comparable to
one trained after. Consistent with the standing decision to re-run
everything under the new framework.

### 3.2 The label registry, measured 2026-09-20 [CLUSTER]

T9's census: `n=36  |L|=26  active=23 (18 ln + 5 linear: Cm, I_inj, Sigma,
VA, VR)  p=27=23+4`. T9b: all 23 active axes have prior box == the
coordinate rule applied to the natural box.

T9c's margin table. The margin is `m_k = log10(hi_k/lo_k) - 1`, so `k` is in
`L` iff `m_k >= 0` and `|m_k|` is how far the bounds must move for that
axis's coordinate to flip:

| axis | decades | coord | in A | margin |
|---|---|---|---|---|
| `U_max` | 1.000000 | ln | **ACTIVE** | **+0.000000** |
| `U_0_sr` | 1.000000 | ln | **ACTIVE** | **+0.000000** |
| `alpha_syn` | 1.000000 | ln | **ACTIVE** | **+0.000000** |
| `x0` | 1.000000 | ln | **ACTIVE** | **+0.000000** |
| `I_Theta` | 1.000000 | ln | - | +0.000000 |
| `U_A` | 0.954243 | linear | - | -0.045757 |
| `O_3K` | 1.176091 | ln | - | +0.176091 |
| `Sigma` | 0.698970 | linear | ACTIVE | -0.301030 |

**Four ACTIVE axes sit exactly on rule (1)'s threshold.** They are in `L`
only because (1) is written `>=` and not `>`; any narrowing at all flips
`theta_k = ln(vartheta_k)` to `theta_k = vartheta_k` and changes the meaning
of an exported column. `I_Theta` is on the same edge but inactive, so a flip
there moves a registry entry and not a theta column.

This is the modal case, not an accident: a factor-of-ten range is the most
natural way a person writes a parameter interval, and (1)'s threshold sits
exactly there. **[SANDBOX, measured]** over 200000 boxes built as
`[lo, 10*lo]` with `lo` log-uniform on `[1e-6, 1e4]`, **4093 (about 2%) give
`log10(hi/lo) < 1`** and fall out of `L` by floating-point representation
alone, the smallest being `0.99999999999999989`. "One decade wide" does not
by itself determine which side of (1) an axis lands on.

### 3.3 The scheduler, measured 2026-09-21 [CLUSTER]

Two facts about davinci's PBS Pro that Stage D turns on, both established by
`extractor/probe_array_depend.sh` and neither taken from documentation.

**(a) `afterokarray` does not exist here.** It is a TORQUE dependency type
(`afterstartarray` / `afterokarray` / `afternotokarray` / `afteranyarray`).
PBS Pro has no `*array` variants; an array dependency uses the ORDINARY
keyword with an array job id carrying the brackets. Swept against a held
array `1723609[]`:

| `-W depend=` | result |
|---|---|
| `afterok:1723609[].pbsserver01` | ACCEPTED (`1723611`) |
| `afterany:1723609[].pbsserver01` | ACCEPTED (`1723612`) |
| `afterokarray:1723609[].pbsserver01` | `qsub: illegal -W value` |
| `afterok:1723609[]` | ACCEPTED (`1723613`); the server suffix is optional |

**(b) `afterok` on an array is ORDERING, not success-gating.** Run
`20260921_151333`, mode `fail`:

| object | `job_state` | `Exit_status` | |
|---|---|---|---|
| array `1723781[]` | F | **0** | |
| subjob 0 | F | 0 | `.ok` |
| subjob 1 | F | **1** | `.failed` |
| subjob 2 | F | 0 | `.ok` |
| dependent `1723782` | F | 0 | **RAN**, observed `n_ok=2 n_failed=1` |

PBS Pro does not propagate a subjob's exit status into the array job's, so
`afterok` sees a clean array and releases the dependent. The control run
(mode `pass`, `20260921_151237`) gave all three subjobs `Exit_status = 0` and
the dependent running with `n_ok=3`.

**Consequence.** For each fixed Stage D launch, nothing may be inferred from
the extraction array completing. `cohort_manifest.py` is the gate: it asserts
a fragment per well and `n_units == n_wells * n_subsets`, names EVERY missing
or unusable well in one error, and writes nothing when any is (M9, M14, M15).
The PASS line in `out/cohort_manifest.log` is the only success signal.

`-v` propagation was checked in the same runs and WORKS for array subjobs
(`Submit_arguments = -J 0-2 -v FAIL_TASK=1`, and the task echoed it), which
matters because `launch_stage_d.sh` hands `EXTRACT_ROOT` to the aggregation
job the same way.

## 4. Claims that were WRONG -- do not re-import

### 4.1 From the 2026-09-16 handoff

| claim | is |
|---|---|
| S1 SHAs | superseded twice; see S1 above |
| S6: `export_windows.py` + `assert_preprocessing_parity()` + `--assume-preprocessing` | **superseded** by the manifest design (S6 below); no such file will be written |
| S6 Q3: `FrozenDSN.observable_block()` unread | read: returns `ifr_dt_s`, `ifr_smooth_sigma_s`, `ifr_smooth_sigma_bins`, `fs_ifr_hz`, `window_s`, `window_length_samples`; no electrode geometry. Also: `load_frozen_dsn` (`dsn_frozen.py:429-438`) silently defaults `w_size=0.02`, `gaussian_window=0.04` when the checkpoint config lacks `cohort.*` -- neither is the cohort's value; the manifest work removes that fallback |
| `CohortConfig` described as if a file | a dataclass at `config.py:1252` (now `hpc/dsn/config.py`) |

### 4.2 Errors made IN THIS SESSION, corrected

| what was said | what is true |
|---|---|
| "an ACTIVE axis (`Sigma`) has flipped ln -> linear; banks either side of the edit disagree" | **FALSE.** `registry_from_manifest` on `campaign_cadex_rho1300v3/sweep_cpu_task0000` returned `changed: []`, `FLIPPED: []`; `Sigma = [3, 15]` in BOTH the live source and the campaign manifest. No bounds edit separates any campaign on record from the simulator. The claim came from reading `sbi_labels.py`'s illustrative *"Sigma at [1.0, 10.0]"* as a record of the current bounds; it is a hypothetical. The bounds check should have preceded the diagnosis |
| "widen the axis to a comfortable margin, or narrow it decisively" | **WITHDRAWN.** Applied to the four active knife-edge axes it would mean editing four prior intervals to work around a code design choice. `PARAM_BOUNDS` is a scientific statement about where a parameter lives; rule (1) is a heuristic about coordinates. **Do not edit those boxes.** The fix is S6b |
| the `|L| = 27 / 19 active ln` literals in T9 | a snapshot of some earlier simulator state, never contemporaneous with any campaign on record. Removed in step 8a |
| S8's step-6 command starting `rm -v ~/dsn_main` inside an `&&` chain | **BROKEN.** `relocate_artifacts.sh` (step 4a) already removes both `dsn_main` symlinks, so that `rm` now exits non-zero and the chain aborts before the `mv`, the `.bashrc` edit and the grep -- silently looking like it ran. Use `rm -vf`, one command per line. `grep -c` has the mirror problem: it exits 1 when the count is 0, the PASSING case |
| `depend=afterokarray:<array id>` in `launch_stage_d.sh` and the probe | **A TORQUE KEYWORD ON A PBS PRO CLUSTER.** `qsub: illegal -W value`; the aggregation job could never have been submitted. Carried across two schedulers because both are called "PBS" -- the transplant check (R3) applied to a scheduler rather than a formula. Fixed in D-4; the correct form is `afterok:<array id>`, measured (S3.3a) |
| `run_cohort_manifest.pbs`: "if any task fails, PBS never runs it. That is what afterok on an ARRAY job id means in PBS Pro" | **FALSE**, written from documentation and never true here. Measured false in S3.3b. Corrected in D-6, with the numbers rather than a softening |
| the first `probe_afterokarray*` | **destroyed its own evidence twice.** One `#PBS -o` shared by three concurrent subjobs (106 surviving bytes held task 0's lines plus the TAIL of task 1's failure message), and state cleared at SUBMIT time so a still-running previous array wrote `.ok` files into the next run's evidence. The result LOOKED like "the deliberate failure did not fire" -- the opposite of the truth. Rewritten in D-5 |
| "the 30-suite batch run is the gate for `CohortConfig`" | **FALSE, and it was never true.** `smoke_test_mea_specs.py` -- the only direct suite for `CohortConfig` and `make_mea_specs.py` -- exists on disk and was never in `run_all_smoke_tests.py`'s `ORDER`, so `qsub hpc/run_smoke_all.pbs` never ran it. That is how a 30/30 run stayed green over a broken `CohortConfig`. Registered in the D commit; the run is now **31** suites, and the runner now also NAMES suites present on disk but absent from `ORDER`. Six others are so named and are NOT registered: `culture_grouping`, `joint_condition_space`, `joint_conditions_wiring`, `loss_type_wiring`, `search_dry_run`, `sep_warmup` -- an open decision, S7 |
| Stage D's first cut of `hpc/dsn/cohort.py` ("moved VERBATIM") | **lacked `import warnings`**: the moved block's `warnings.warn(...)` for sub-bin smoothing raised `NameError` at construction for any cohort with `gaussian_window / w_size < 1`. The 30-suite run did not see it because the davinci config has exactly 2 bins. Caught by pyflakes in the gate, fixed before the patch was cut, and pinned by the new check A2 in `smoke_test_mea_specs.py` (which fails with the import removed -- negative control run). Lesson in S9 |
| S6's "Manifest fields" paragraph (as written 09-21): gating = `dt`, `sigma_sm`, `w_size`, `gaussian_window`, `electrodes_per_subset`, `mfr_threshold`, `n_units`, `manifest_version` | **[corrected 09-24] not what was coded.** `dt` IS `w_size` and `sigma_sm` IS `gaussian_window` -- one object under the two repos' two names, not four fields -- and the paragraph omitted `fs_raw`, `index_base`, `grid_width`, `n_subsets`, which ARE gating. As coded and as written into the manifest's `gating.real`: the eight `PREPROCESSING_FIELDS` (`fs_raw`, `index_base`, `grid_width`, `n_subsets`, `electrodes_per_subset`, `mfr_threshold`, `w_size`, `gaussian_window`) + `n_units` + `manifest_version`. The same paragraph's "Gating at write / reported at read: sha256 of `extraction_flags.sh` and of the cohort config" is not what was coded either: the write-time gate on the flags is a byte comparison with the regenerated flags (`build_manifest` check 5); the three sha256 values (`config_file`, `cohort_block`, `extraction_flags_sh`) are only recorded -- nothing compares them at write and nothing reads them afterwards; `read_manifest` checks only the manifest's own sidecar, when that file exists, and the version |
| S8's launcher line `DRYRUN=1 ./launch_sweep_exports.sh ... \| grep -E "..."` | **BROKEN TWICE.** (a) no tracked `.sh` in Sbi-extractor was ever executable, so `./` gave `Permission denied` -- a *shell* message matching none of the grep patterns, so the command printed **nothing** and exited 0. Fixed by step 8c. (b) the line omitted `source env.sh` and `export SIM_MAIN_DIR`, both required (guard 1 exits 6; `env.sh` does NOT set `SIM_MAIN_DIR`) |

## 5. The steps -- status and the run behind each

| step | what | status |
|---|---|---|
| 0 | tag `dsn-final-20260919` at `7afe1b8`; retirement README | **[PUSHED]** `750d5bb` |
| 1 | mirror `Main/` -> `hpc/dsn/` (195 files) + `ORIGIN_MANIFEST.tsv` | **[PUSHED]** `5135a4f`, **[CLUSTER]** 30/30 from both trees |
| 1b | commit `hpc/dsn/hpc/run_smoke_all.pbs` | **[PUSHED]** `6b9eda2` |
| 2 | `hpc/joint/dsn_locate.py`; 6 consumers + 6 suites + 3 job scripts rewired | **[PUSHED]** `99dc3f0`, **[CLUSTER]** probe 7/7 |
| 3 | extractor -> `Sbi-extractor/extractor/`; `dsn_tree.py`; `artifacts/sbi_hpc` | **[PUSHED]** `1bf7f0f`, **[CLUSTER]** 8/8 |
| 4a | everything resolves the DSN through `dsn_tree.py`; the two parity files retired | **[PUSHED]** `fdc23c8`, **[CLUSTER]** |
| 4b | `build_pooled_ifr` calls the extractor's `compute_ifr_trace`; T2b/T2c | **[PUSHED]** `2ff4a4b`, **[CLUSTER]** |
| 5 | `sbi_env` spec lists pml + skopt; consolidation rule in `hpc/dsn/README.md` | **[PUSHED]** `bc7d578` |
| 5b | defaults flip: DSN jobs -> `sbi_env`, extractor jobs -> `sbi_export` | **[PUSHED]** `fe9cb20` / `28b0625`; **CONFIRMED [CLUSTER 09-20]** jobs 1722779 (30/30) and 1722780 (8/8) + flags `cmp`. **No revert.** |
| 7 | plan v0.6.5; `EXTRACTOR_USAGE.md` v8; this handoff | **[PUSHED]** `d460c78` / `75d455f` |
| 8a | T9/T11/T12 assert relations not counts; new T9b; new `smoke_test_registry_flexibility.py` (14 checks) | **[PUSHED]** `be2b6e9`, **[CLUSTER]** |
| 8b | `theta_36`/`params_36` -> `theta_registry`/`params_registry`; sidecar `param_names_36` -> `param_names`, `+n_registry_axes` | **[PUSHED]** `64b01a4` |
| 8c | `chmod +x` the five entry-point `.sh` scripts | **[PUSHED]** `67c216e` |
| 8d | `EXTRACTOR_USAGE.md` v8.1 | **[PUSHED]** `f2e158c` |
| 9 | `coordinate_margins` / `at_risk_axes` / `format_margins`; T9c; F13/F14 | **[PUSHED]** `f90fd29`, **[CLUSTER]** `PASSED 16 FAILED 0` |
| D (SBI) | `hpc/dsn/cohort.py` (torch-free `CohortConfig` + well helpers, one definition, re-exported by `config.py` / `make_mea_specs.py`); check A2; `smoke_test_mea_specs.py` REGISTERED in the batch runner (**30 -> 31 suites**) + the mirror "present on disk, not in ORDER" warning; the three `Smoke_Tests/` output dirs gitignored; "195 files" literals dropped | **[PUSHED]** `49da750`, **[CLUSTER 09-21]** DSN suite **31/31 in 270.4 s**, `runner_exit=0`, `manifest_rows=196` (job 1723902). Sandbox: 31/31 in 318.5 s; import guard forbids torch/backbone/config |
| D-1 (SBIX) | `cohort_config.py`; `list_extraction_jobs.py` torch-free, `--extract-root` | **[PUSHED]**, **[CLUSTER 09-21]** `torch imported: False`; flags regenerate byte-identical, 35 wells |
| D-2 (SBIX) | extractor v3 fragments; `cohort_manifest.py`; `run_cohort_manifest.pbs`; `launch_stage_d.sh`; `probe_afterokarray*`; `smoke_test_cohort_manifest.py` (14) | **[SANDBOX 09-21]** 14/14; three launcher refusals exercised; probe plumbing on a fake `qsub`. **[corrected 09-24]** PUSHED as `7b66f1f`; the `afterokarray` form it shipped was replaced by D-4 (PBS Pro has no such keyword, S3.3a) |
| D-3 (SBIX) | `dsn_frozen.py` raises instead of `0.02/0.04`; T8b; T12/T10 literal-free; `EXTRACTOR_USAGE.md` v8.2 | **[PUSHED]**, **[CLUSTER 09-21]** export suite **`PASSED 17 FAILED 0 SKIPPED 0`** against the real registry |
| D-4 (SBIX) | `afterokarray` -> `afterok`; `probe_afterokarray*` -> `probe_array_depend*`; `verify_origin_manifest.py`, wired into `run_extractor_smoke.pbs`; two ORIGIN_MANIFEST shas stale since step 5b, refreshed | **[PUSHED]** `4ffd6f1`, **[CLUSTER 09-21]** `verify_origin_manifest.py` PASS 23 rows |
| D-5 (SBIX) | the probe rewritten: per-run dirs, per-subjob logs, in-flight refusal, computed VERDICT | **[PUSHED]**, **[CLUSTER 09-21]** ran three modes cleanly and produced the S3.3 verdict |
| D-6 (SBIX) | `build_manifest` names EVERY unusable well in one error (M15); the false "PBS never runs it" comment corrected; `launch_stage_d.sh` prints the measurement; `EXTRACTOR_USAGE.md` v8.3 | **[PUSHED]** `442eaf4`, **[CLUSTER 09-21]** `smoke_test_cohort_manifest.py` **ALL 15 CHECKS PASSED** |
| 6 | remove `$HOME/dsn_main`; rename `~/dsn_git`; drop the `.bashrc` export; fresh-clone discriminator | **[CLUSTER 09-21] DONE.** `dsn_main` already gone (4a); `~/dsn_git` -> `~/dsn_git_RETIRED_20260920`; `.bashrc` lines 24-25 removed, `grep -c` = 0 in `.bashrc` and `.bash_profile`, no `.profile`; discriminator: two fresh clones at `f90fd29` / `d460c78`, both resolvers `resolves : yes`, `ALL 14 CHECKS PASSED`. **Remaining: archive the GitHub repo (UI click).** |

Decisions taken (all the user's): standalone DSN training kept; the extractor
emits the manifest; raise on any mismatch; the DSN reads the manifest for
preprocessing and the config for everything else; `mfr_threshold` is gating;
`manifest_version` is present; everything downstream is re-run under the new
framework and the frozen r2 encoder is not carried forward; the inference
coordinate is frozen in the label-axes contract (S6b). **09-21:**
`mfr_threshold` and `n_subsets` are RECORDED on the sim arm and never applied
to it; `depend=afterokarray` is VERIFIED by a probe before the real launch,
not assumed; no `cmp` of old vs new archives (everything is re-extracted);
`CohortConfig` lives in the SBI repo (`hpc/dsn/cohort.py`), Sbi-extractor
reads it through `cohort_config.py` -- dependency direction unchanged.

## 6. Stage C/D as redesigned -- the cohort manifest [D BUILT and RUN 2026-09-21; C in progress -- corrected 09-24]

```
CohortConfig (hpc/dsn/cohort.py, torch-free)     [configured] -- the intended preprocessing   BUILT
  -> cohort_config.py -> extractor/list_extraction_jobs.py --extract-root extracted_v2
     -> extraction_flags.sh (tracked) + extraction_manifest.tsv                              BUILT
  -> run_extractor_array_mea.pbs: each task writes its archives + traces_meta.json (v3)      BUILT
  -> run_cohort_manifest.pbs (depend=afterok:<array id> -- ORDERING only, S3.3b) [corrected 09-24]:
     cohort_manifest.py reads ALL fragments,
     asserts constant / measured == configured / n_units / tracked flags,
     writes extracted_v2/cohort_manifest.json + .sha256                      [measured]      BUILT, RUN 09-21 [corrected 09-24]
  -> DSN training reads its preprocessing FROM the manifest; the config loses that block
     (a config still carrying it must agree, else raise); the checkpoint stores the
     manifest sha256
  -> real-arm export: each archive asserted against the manifest
  -> sim-arm export: conform dt, sigma_sm to the manifest; assert n_e against it
     (sim n_e is campaign geometry, electrode_centers -- [corrected 09-24] since SC-D5
     CONFIGURED from the manifest at the sim re-extraction, then checked)
  -> shards + sidecars carrying the manifest sha256; export asserts ckpt digest == manifest digest
  -> bank -> NPE
```

Rules: mismatch **raises**; there is no `--assume-preprocessing`; legacy
archives (no recorded keys) are not exported at all; never compare a read
value against a literal 9 -- and note there are TWO nines, `electrodes_per_subset`
= 9 electrodes per subregion and 9 subregions per culture (315 = 35 x 9);
they are not the same nine. **[corrected 09-24]** "legacy archives (no recorded
keys)" does not describe `extracted/`: by the project record those archives
were written by extractor v2, which records the eight preprocessing fields, so
the `LegacyArchive` check would most likely not catch them (not verified on
the files; Stage C handoff C1, C6). **[09-28]** Sbi-extractor C2c (in the
09-28 bundle, not yet pushed; **[09-30]** pushed as `63d2574`) makes the
check refuse them; no export calls
the check yet (Stage C handoff C2c).

**[corrected 09-24: not what was coded -- see S4.2]** Manifest fields. Gating (raise on mismatch): `dt`, `sigma_sm`, `w_size`,
`gaussian_window`, `electrodes_per_subset`, `mfr_threshold`, `n_units` (315),
`manifest_version`. Gating at write / reported at read: sha256 of
`extraction_flags.sh` and of the cohort config. Recorded only: extractor
commit (an `Sbi-extractor` commit now), timestamp.

**What differs between the arms, read from `channel_subset_extraction.py`
Stage 3 (lines 541-741).** Both arms share the map `E_pi` -- one
`compute_ifr_trace` since step 4b -- and inherit `dt`, `sigma_sm` from the
manifest. What differs is how the map's DOMAIN is built:

| symbol | meaning | real | sim |
|---|---|---|---|
| `E` = `electrodes_per_subset` | electrodes pooled per subregion | 9 | 9 (`electrode_centers`, n_side = 3) |
| `C` = `n_subsets` | subregions drawn per recording | 9 | 1 |
| `theta` = `mfr_threshold` | electrode valid iff MFR_e >= theta | applied, electrode-level, before the partition | **no electrode-level operation exists** |

Stage 3 is a greedy DISJOINT partition: valid set `V = {e : MFR_e >= theta}`,
ranked by MFR descending, each subregion = hottest unassigned valid electrode
plus its `E-1` nearest unassigned valid neighbours, no radius cap, and
`InsufficientElectrodesError` (DISCARD the recording) if `C` disjoint
subregions of size `E` cannot be formed. So the 9 real subregions are the 9
hottest disjoint clusters, not a tiling, and their geometry is a compact 3x3
only when the neighbourhood is fully valid. `C` and `n_units = 35 x C` are
real-arm multiplicities: **record them, never gate the sim on them**, exactly
as for `n_e`. **DECIDED 09-21 (the user's): `mfr_threshold` is record-only on
the sim arm** -- option (i); the real rule is applied on the real arm only.
In `cohort_manifest.py`: `GATING_REAL` = all eight fields;
`CONFORMED_SIM = (w_size, gaussian_window)`; `ASSERTED_SIM =
(electrodes_per_subset,)`; `RECORDED_SIM = (mfr_threshold, n_subsets, fs_raw,
index_base, grid_width)`.

Done in D: `CohortConfig` lifted (torch-free); `load_frozen_dsn`'s `0.02 /
0.04` fallback removed (raises); `assert_archive_matches_manifest()` written,
owned by Sbi-extractor. Still C: wiring both exports and the DSN training
loader to CALL it, the checkpoint storing the manifest sha256, the sidecars
carrying it, and flipping `cohort.extract_root` to `extracted_v2/` (first act
of C, after the manifest exists).

Family **[reasoning]**: the prospective/retrospective provenance split of
workflow-centric provenance models -- Ott et al. 2026,
`10.1515/jib-2025-0050` (PubMed, full text read): `CohortConfig` is the plan,
the manifest is the trace.

Order of work: (D) re-extract into `extracted_v2/` with the manifest -- **DONE
2026-09-21**; (C) the manifest-aware exports, `claude/HANDOFF_2026-09-21_stageC.md`;
then retrain the DSN under the manifest; then re-export both arms; then Stage 6.

**DECISION 2026-09-21 `[STATED, emphatic]`: the DUP15HD SIMULATED campaigns
WILL BE RE-EXTRACTED.** This overrides the analysis given the same day, which
concluded they needed only re-exporting because `process_campaign.py` performs
spike detection only and has no downsampling or smoothing parameter
`[KB -- HPC_PATHS.md sec. 4b]`, so no manifest field forces it. That analysis
is not withdrawn -- nothing in the manifest compels the re-extraction -- but
the decision stands and is not to be re-argued. **Its SCOPE is OPEN**: which
stage is re-run (`process_campaign.py` over existing raw simulations, or the
simulator sweeps themselves), what changes as a result (a detection parameter,
the geometry, the campaign set, or provenance only), whether it covers all 207
sweep tasks, and which NEW root it writes to. Settled in Stage C, section C5.
**[corrected 09-24] SETTLED**: `process_campaign.py` over the existing raw
simulations, `n_e` from the manifest, every DUP15HD campaign -- read as
including v9, Claude's reading, and v9's raw output is not on record -- into
`ANN/MEA_analysis/Outputs_v2`. Stage C handoff SC-D1..D5, C8.

## 6b. The inference coordinate -- decided 2026-09-20, NOT built

**The problem.** Rule (1), `L = {k : lo_k > 0, hi_k > 0, log10(hi_k/lo_k) >= 1}`,
INFERS a modelling decision -- which coordinate to do inference in -- from a
prior specification, the bounds. They are different objects, and the
inference is discontinuous at exactly the most common human choice. Four
active axes now sit on that discontinuity (S3.2).

**The decision (the user's).** Freeze the coordinate in the label-axes
contract. **But a frozen `coord` can only CHECK, never impose** --
`assemble_theta_A` slices `theta` out of each npz and never re-transforms it,
so the coordinate of stored data was fixed by the simulator at generation
time; a frozen file that disagreed would mislabel real data. So:

> `artifacts/label_axes.json` **declares** the bank's theta column contract.
> Each campaign's `manifest.json` still **supplies** the coordinate, via
> `registry_from_manifest`, because that is what wrote the data. At export
> the manifest-derived coordinate is **asserted against** the declaration and
> a mismatch **raises**, naming the axes.

One bank, one coordinate contract; a campaign of another vintage is refused
rather than silently pooled. Same shape as the cohort manifest: plan, trace,
gate.

**Scope of the change.** `preflight_label_axes.py` (which today imports only
numpy and json -- it has **no registry dependency at all** and writes
`topology_axes` / `excluded_axes`, nothing about coordinates),
`example_export.py::_load_label_axes`, `build_label_spec`, the sidecar; plus
regenerating `label_axes.json` on the cluster. Belongs in Stage D, where
`build_label_spec` and the manifest are being opened anyway. **[corrected
09-24]** it was not built in D; it is Stage C item C3.

**What already protects, measured:** `registry_from_manifest` pins each
campaign at export and prints `COORDINATE FLIP vs the live source on: ...`
(`example_export.py:416-429`); `dataset_profile.py:106` lists
`("labels.coord", "hard", "ln vs linear per axis")`, so `compare_profiles`
already refuses a cross-vintage pool; and step 9's margin instrument names
the at-risk axes before anything happens. Detection is covered three ways.
Only PREVENTION is missing, and that is what this gate adds.

**Explicitly NOT the fix:** editing `PARAM_BOUNDS` to move an axis off the
threshold (changes the prior -- the tail wagging the dog), and moving rule
(1)'s threshold (reclassifies four active axes now, committing the exact harm
being prevented, and relocates the knife edge rather than removing it).

## 7. Standing [OPEN] items

- **CLOSED 09-24 [corrected 09-24]** -- `HPC_PATHS.md` (v2) and
  `JOINT_DSN_NPE_USAGE_v1.md` (v1.2) are rewritten for S1-S2's layout and
  written to project knowledge. `EXTRACTOR_USAGE.md`: v8.3 (D-6) is current on
  GitHub; the Stage C commits C2a / C2b bring the repo to v8.5 / v8.6,
  carrying in the 09-21 v8.4 row that only the project-knowledge copy had;
  the project-knowledge copy is v8.6. **[09-28]** C2c brings both to v8.7.
- **Archive `Leonardodm00/Deep-Summary-Network` on GitHub** -- Settings ->
  Danger zone -> Archive. Reversible. The only step-6 action left.
- **Six DSN suites on disk but not in `run_all_smoke_tests.py`'s `ORDER`**:
  `culture_grouping`, `joint_condition_space`, `joint_conditions_wiring`,
  `loss_type_wiring`, `search_dry_run`, `sep_warmup`. The runner names them
  at the end of every run since Stage D. Each either belongs in that runner
  or in `run_tests.sh`; until that is decided they are coverage that is
  believed and not held.
- **CLOSED 09-24 [corrected 09-24]** -- `extracted_v2/cohort_manifest.json`
  is READ; its contents are the last row of S3.1.
- **CLOSED 09-28 -- the stale second clone on the laptop**,
  `/c/Users/leona/repos/Sbi-extractor` (`main` at `8ca6605`, none of the
  branch work in it; its path matches davinci's layout, so muscle memory
  reached for it): renamed `Sbi-extractor_RETIRED_20260924`
  `[DEVICE 09-28 15:21]` after a re-check (`HEAD == origin/main`, no local
  edits, no stash, no unpushed commits; on 09-24 all 14 files identical to
  `8ca6605`). The working clones are `/c/Users/leona/Simulation-Based-Inference`
  **[corrected 09-28: spelled with capital B and I, read in Explorer; this
  line wrote `Simulation-based-inference`]** and `/c/Users/leona/Sbi-extractor`.
  The two other stale clones in `repos/` stay (the user, 09-28);
  `HPC_PATHS.md` sec. L.
- **The coordinate gate**, S6b. Not built; now belongs to Stage C.
- **`probe_conda_activation.pbs`** still untracked on the cluster
  (`~/SBI/hpc/jobs/`). The two `.RETIRED_20260916` strays and
  `hpc/joint/stage1/figs/` are also untracked there; harmless.
  `~/SBI/hpc/dsn/out/1722691.pbsserver01.OU` is the artefact of the old
  `#PBS -o out/` bug; deletable.
- **The 11 synthetic DSN experiment scripts [corrected 09-24: not 9]** (7
  `hpc/dsn/hpc/dsn_4c_*.pbs`, 2 `hpc/dsn/hpc/dsn_latent_*.pbs`, 2
  `hpc/dsn/runs_synthetic/*.pbs`) hard-code `conda activate meacnn_cpu`; left
  as history. `hpc/dsn/hpc/run_factorial52_screen.pbs` and
  `run_example_3class_simplex.sh` still say `brian_env`.
- From the 2026-09-16 handoff, unchanged: L7/L8 (the symptom oscillates),
  D3/D6 p-value jitter, `STAGE6_PLAN_v0_3.md` lost,
  `JOINT_DSN_NPE_DEVIATIONS_v1.md` S1.3 title, the dead documentation paths
  (now also `extractor/README_HPC_DSN.md`, kept verbatim with OLD paths),
  `run_tests.sh` registering no probe.
- **No Stage 1 latent bank exists.**

## 8. Step 6 -- RUN 2026-09-21. Kept for the record and for re-running.

These were run in this order, one line at a time, after jobs 1722779 and
1722780 passed. Each is safe to re-run.

    rm -vf ~/dsn_main
    if [ -e ~/dsn_git ]; then mv -v ~/dsn_git ~/dsn_git_RETIRED_20260920; else echo "already renamed"; fi
    grep -n DSN_MAIN_DIR ~/.bashrc
    cp ~/.bashrc ~/.bashrc.bak_20260920 && sed -i '/DSN_MAIN_DIR/d' ~/.bashrc
    grep -c DSN_MAIN_DIR ~/.bashrc ~/.bash_profile ~/.profile 2>/dev/null; echo "--- 0 or 'No such file' on every line is the pass ---"

Measured: the `rm` was silent (4a's `relocate_artifacts.sh` had already taken
it, which is why `-f` and not `-v` alone -- inside an `&&` chain a bare `rm`
would abort the rest); the rename printed; `.bashrc` lines 24-25 were
`: "${DSN_MAIN_DIR:=...}"` and `export DSN_MAIN_DIR`, both removed; the final
grep gave `0` for `.bashrc` and `.bash_profile`, and `.profile` does not
exist.

Then the discriminator -- two fresh clones needing nothing outside themselves:

    cd /tmp && rm -rf mig6 && mkdir mig6 && cd mig6 && git clone -q https://github.com/Leonardodm00/Simulation-Based-Inference.git && git clone -q -b feat/real-arm-parity https://github.com/Leonardodm00/Sbi-extractor.git && mkdir -p Sbi-extractor/artifacts && ln -s "$PWD/Simulation-Based-Inference/hpc" Sbi-extractor/artifacts/sbi_hpc && cd Sbi-extractor && python3 dsn_tree.py
    cd /tmp/mig6/Simulation-Based-Inference/hpc && python3 joint/dsn_locate.py
    cd /tmp/mig6/Sbi-extractor && python3 smoke_test_registry_flexibility.py 2>&1 | tail -2

**[CLUSTER 09-21] PASSED**: `f90fd29` and `d460c78`, `resolves : yes` from
both resolvers, `ALL 14 CHECKS PASSED`. The third line is the one that
matters: the resolvers only say a path exists, while the suite proves a bare
clone runs real code with `~/dsn_git` gone and `DSN_MAIN_DIR` out of
`.bashrc`.

`dsn_locate.py` printed its `DSN_MAIN_DIR ... is set but IGNORED` line,
because the shell predated the `.bashrc` edit and still carried the variable
-- pointing at a path 4a had already deleted. It resolved correctly anyway.
That is the step-2 guard working against a dead pointer; a fresh login
clears it.

Remaining: archive `Leonardodm00/Deep-Summary-Network` on GitHub (Settings
-> Danger zone -> Archive; reversible). Then `rm -rf /tmp/mig6` and, at
will, `rm -rf ~/dsn_git_RETIRED_20260920`.

### 8b. The checks, for re-running any of them

    cd ~/repos/Sbi-extractor && conda activate sbi_export && source env.sh && python3 dsn_tree.py
    cd ~/repos/Sbi-extractor && python3 smoke_test_registry_flexibility.py 2>&1 | tail -3
    cd ~/repos/Sbi-extractor && python3 smoke_test_sbi_export.py --sim_dir /davinci-1/home/ldellamea/ANN/Phenomenological/Main 2>&1 | tail -15
    cd ~/repos/Sbi-extractor && export SIM_MAIN_DIR=/davinci-1/home/ldellamea/ANN/Phenomenological/Main && DRYRUN=1 ./launch_sweep_exports.sh artifacts/frozen_dsn/dsn_r2_20260824.pt /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs "$SIM_MAIN_DIR" /tmp/sbi_dryrun 'campaign_cadex_rho1300v3' 2>&1 | head -40
    cd ~/repos/Sbi-extractor/extractor && python3 list_extraction_jobs.py --config "$SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json" --out-manifest /tmp/m.tsv --out-flags /tmp/f.sh && cmp /tmp/f.sh extraction_flags.sh && wc -l /tmp/m.tsv
    cd ~/SBI/hpc/dsn && mkdir -p out && qsub hpc/run_smoke_all.pbs
    cd ~/repos/Sbi-extractor/extractor && mkdir -p out && qsub run_extractor_smoke.pbs

The launcher line needs `source env.sh` (for `SBI_HPC_DIR`) AND
`export SIM_MAIN_DIR` (guard 1 exits 6 without it; `env.sh` does not set it).

## 9. Working conventions

- **A move is a byte-identity claim.** Every migrated file carries its sha256
  and source path in an `ORIGIN_MANIFEST.tsv`; "verbatim" is checked with
  `git show <tag>:<path> | sha256sum`. Nothing is "basically the same".
- **One variable at a time.** The extractor moved under `meacnn_cpu` and only
  then switched to `sbi_export`, each with its own check. The IFR unification
  (4b) is its own commit for the same reason.
- **A one-line `python -c` prerequisite check at the top of a batch job**,
  printing versions, turns "the suite failed 30 times" into "the environment
  lacks X" before the suite starts; and a job's `#PBS -o` names a FILE --
  PBS Pro took `out/` as a filename and delivered `<jobid>.pbsserver01.OU`.
- **NEW 09-20. A narrow `grep` on a command's own success strings turns a
  hard failure into a silent one.** It cannot distinguish "succeeded but
  printed nothing matching" from "died before printing anything". Use
  `| head -40` and read it. This cost two rounds.
- **NEW 09-20. Check the value before diagnosing from the docstring.** An
  illustrative example in a comment is not a record of the current state
  (S4.2, `Sigma`).
- **NEW 09-20. `git apply` cannot make a file executable on Windows.**
  `core.fileMode` is false on NTFS, so a mode-only patch stages nothing and
  the commit fails with "nothing to commit". Use
  `git update-index --chmod=+x <files>` -- it writes the mode into the index
  and ignores `core.fileMode`. The commit then records `100755` and a Linux
  clone checks the file out executable, which is the only place it matters.
- **NEW 09-21. A scheduler is not "PBS".** Torque and PBS Pro share a command
  set and diverge on the details that matter -- dependency keywords, the array
  index variable, how an array's exit status relates to its subjobs'. Before
  using any of them, sweep the forms against the actual server and keep the
  sweep in the file. `afterokarray` cost a round; the array exit status cost
  three.
- **NEW 09-21. An experiment that writes its evidence must isolate its runs
  and never share an output path.** Concurrent subjobs with one `#PBS -o`
  overwrite each other from byte zero; state cleared at submit time gets
  written into by the previous run's jobs still in flight. Both faults produce
  evidence that reads as the OPPOSITE of what happened, which is worse than no
  evidence. Per-run directories, per-subjob logs, and a refusal to start while
  the previous run is queued.
- **NEW 09-21. Simulate the mechanism both ways before shipping the probe.**
  Driving the rewritten probe against a fake `qsub`/`qstat` that gated, and one
  that did not, found two more bugs before the cluster did: `ls <glob> | wc -l`
  under `set -eo pipefail` (kills the job when the glob matches nothing -- the
  normal case) and `$USER` under `set -u` (unset inside a PBS job).
- **NEW 09-21. Before trusting a suite as a gate, check the suite is RUN.**
  A runner with an explicit `ORDER` list silently omits any suite added to
  the directory afterwards. `smoke_test_mea_specs.py` sat unrun and was the
  only direct test of the object being edited. `ls smoke_test_*` against the
  runner's list takes one command and is now automatic (the runner prints
  the difference at the end of every run).
- **NEW 09-21. A suite that writes into the tree needs a gitignore entry,
  or `git add -A` commits its output.** The DSN suite wrote ten PNGs and a
  JSON into `Smoke_Tests/`; nothing ignored them. Both push scripts now
  stage with `git apply --index` -- only the patch's own paths -- rather
  than `git add -A .`.
- **NEW 09-21. A verbatim move is not verified by a green suite; it is
  verified by pyflakes AND a green suite.** A block moved between files
  carries its calls but not its imports; a suite run at one configuration
  (2 smoothing bins) never enters the branch (< 1 bin) that needed the
  missing import. Run `python3 -m pyflakes` on every moved or edited file
  before cutting a patch, and add a check that enters the branch the
  production config does not (A2).
- **NEW 09-20. A test must not assert a property of the data it reads.**
  T9 pinned `|L| = 27` and failed when the simulator's bounds moved,
  reporting a registry change as a defect in the export. Assert RELATIONS,
  REPORT counts. `smoke_test_registry_flexibility.py` exists to catch a count
  creeping back in, with no simulator repo and no DSN tree needed.

## 10. Documents

| file | what |
|---|---|
| `claude/HANDOFF_2026-09-21_stageC.md` | **the entry point for Stage C.** Start there |
| this file | the history and the measured facts; replaces `HANDOFF_2026-09-16_stageC.md` |
| `EXTRACTOR_USAGE.md` | v8.3 in the repo and (with an extra 09-21 v8.4 row) in project knowledge as of 09-21; **[corrected 09-24]** the Stage C commits bring the repo to v8.6 and the project-knowledge copy is synced to it; **[09-28]** v8.7 with C2c, both |
| `stage_d_bundle_20260921.zip` | 4 patches + 4 messages + `push_sbi.sh` / `push_sbix.sh` (laptop) + `hpc_stage_d_gate.sh` (davinci) + `README.txt`; S11 |
| `JOINT_DSN_NPE_PLAN_v0_6.md` v0.6.5 | in the repo only (`hpc/joint/`) |
| `HPC_PATHS.md` v2, `JOINT_DSN_NPE_USAGE_v1.md` v1.2 | rewritten 09-24 for S1-S2 **[corrected 09-24]** |
| `stagec_c2_bundle_20260924.zip` **[added 09-24]** | the Stage C commits C2a, C2b + `push_sbix_c2.sh` (laptop) + `hpc_c2_gate.sh` (davinci); Stage C handoff C2a/C2b. **Superseded [09-28]** by the next row |
| `stagec_bundle_20260928.zip` **[added 09-28]** | three repositories: Sbi-extractor C2a + C2b + C2c, SBI the `hpc/dsn` manifest refresh + verifier, ANN the launcher env fix; `sbix/push_sbix.sh`, `sbi/push_sbi.sh`, `ann/push_ann.sh` (laptop; each checks the tested TREE id), `davinci/hpc_gate_20260928.sh` (14 checks, submits nothing) and `davinci/probe_davinci_20260928.sh` (read-only facts); `README.txt`. **[09-30] Pushed** (`63d2574`, `834eb41`, `37bf9f8`; trees the tested ones). Stage C handoff C2 |

## 11. Stage D -- what is built, how to push it, how to run it

Everything below is **[SANDBOX 2026-09-21]** unless tagged otherwise. The
bundle is `stage_d_bundle_20260921.zip`; every patch was applied to a fresh
clone of its base, committed with its message, and the resulting tree
byte-compared against the tree the suites ran on; `push_sbix.sh` was also
run with `core.fileMode=false` (the Windows case) and the two new shell
entry points came out `100755` in the index. Both scripts stage with
`git apply --index` -- only the patch's own paths -- so untracked files
already sitting in the clone are never swept into the commit; both were
re-tested with junk planted in the tree.

### 11.1 The four commits

| commit | base | files |
|---|---|---|
| SBI "dsn: lift CohortConfig into torch-free cohort.py (Stage D)" | `d460c78` | 10 files: `hpc/dsn/cohort.py` NEW; `config.py`, `make_mea_specs.py` re-import; `Smoke_Tests/smoke_test_mea_specs.py` (A2); `Smoke_Tests/run_all_smoke_tests.py` (registers it, 30 -> 31; mirror warning); `hpc/.gitignore`; `ORIGIN_MANIFEST.tsv`; `README.md`; `hpc/joint/dsn_locate.py`; `hpc/run_tests.sh` |
| SBIX D-1 "read the cohort block torch-free via cohort_config.py" | `f90fd29` | `cohort_config.py` NEW; `extractor/list_extraction_jobs.py` |
| SBIX D-2 "version-3 fragments, cohort_manifest.py, aggregation job, launcher" | D-1 | `cohort_manifest.py`, `smoke_test_cohort_manifest.py` NEW; `extractor/run_channel_subset_extraction.py`; `extractor/{run_cohort_manifest.pbs, launch_stage_d.sh, probe_afterokarray.sh, probe_afterokarray_task.pbs, probe_afterokarray_agg.pbs}` NEW; `extractor/README.md`; `extractor/ORIGIN_MANIFEST.tsv` |
| SBIX D-3 "no silent preprocessing fallback; tests derive names from the spec" | D-2 | `dsn_frozen.py`; `smoke_test_sbi_export.py` (T8b, T12); `smoke_test_registry_flexibility.py` (stub gains `natural_to_theta`/`theta_to_natural`, so T10 runs); `EXTRACTOR_USAGE.md` v8.2 |
| SBIX D-4 "afterokarray is Torque; PBS Pro wants afterok on the array id" | D-3 | `launch_stage_d.sh`, `run_cohort_manifest.pbs`; `probe_afterokarray*` renamed `probe_array_depend*`; NEW `verify_origin_manifest.py`, called from `run_extractor_smoke.pbs`; `ORIGIN_MANIFEST.tsv` (two shas stale since step 5b) |
| SBIX D-5 "rewrite the array-dependency probe; it corrupted its own evidence" | D-4 | the three `probe_array_depend*` files; `ORIGIN_MANIFEST.tsv`; docs |
| SBIX D-6 "name EVERY unusable well; afterok does not gate" | D-5 | `cohort_manifest.py`; `smoke_test_cohort_manifest.py` (M15, suite 15); `run_cohort_manifest.pbs`; `launch_stage_d.sh`; docs |

SBIX depends on the SBI commit (`cohort_config.py` imports
`$SBI_HPC_DIR/dsn/cohort.py`); push SBI first. Against an older SBI tree
the import fails at its first line naming `cohort.py`; nothing else is read.

### 11.2 The manifest contract as coded (`cohort_manifest.py`)

`build_manifest(config_json, manifest_tsv, flags_path, extract_root=None,
extractor_commit=None)` asserts, in order, and raises `ManifestError` on the
first failure: (1) every well in the tsv has `traces_meta.json` with every
field; (2) every field constant across wells, including `fs_ifr`,
`extractor_version`, `manifest_version`; (3) measured == configured for the
eight `PREPROCESSING_FIELDS`; (4) `n_units == n_wells * n_subsets`; (5) the
tracked `extraction_flags.sh` == `build_extra_flags(plan)` byte for byte.
`write_manifest` adds `cohort_manifest.json.sha256`; `read_manifest` verifies
it. A version-1 archive raises `LegacyArchive`. Consumers:
`assert_archive_matches_manifest(archive, manifest, what)` (real arm),
`sim_preprocessing_from_manifest(manifest) -> (dt, sigma_sm, n_e_expected,
recorded)` and `assert_sim_geometry(manifest, n_e)` (sim arm).

### 11.3 Sequence (one command per line; each line's expected output beside it)

Laptop (Git Bash, files in `~/Downloads`):

    cd ~/Downloads && unzip -o stage_d_bundle_20260921.zip -d stage_d_bundle_20260921
    cd <Simulation-Based-Inference clone> && bash ~/Downloads/stage_d_bundle_20260921/push_sbi.sh     # 1 commit, then: git push
    cd <Sbi-extractor clone> && bash ~/Downloads/stage_d_bundle_20260921/push_sbix.sh                 # 3 commits, then: git push

davinci, login node (pull, torch-free check, 5 suites, launcher DRYRUN + refusal; nothing submitted):

    scp ~/Downloads/stage_d_bundle_20260921/hpc_stage_d_gate.sh <your davinci ssh target>:~/     # from the laptop, or paste it
    bash ~/hpc_stage_d_gate.sh                                                            # on davinci; expect PASS on every line

davinci, batch, in this order, waiting for each:

    A  DSN suite            DONE 09-21, job 1723902: 31/31, runner_exit=0
    B  extractor suite      DONE 09-21: 8/8, and it now also prints [manifest] PASS
    C  dependency probe     DONE 09-21: VERDICT NOT GATED (S3.3b); leftovers cleaned
    D  the launch           DONE 09-21: array 1723886[], agg 1723887,
                            manifest sha256 c925c5106137fe82, cohort_manifest_exit=0

To re-run D for another root, the sequence is the same, dry run first:

    cd ~/repos/Sbi-extractor/extractor && DRYRUN=1 bash launch_stage_d.sh "<NEW ROOT>"
    cd ~/repos/Sbi-extractor/extractor && bash launch_stage_d.sh "<NEW ROOT>"

PASS is the `wrote ... sha256 ...` line in `out/cohort_manifest.log`. A partial
cohort ends instead as a REFUSAL naming every well to re-extract. Never infer
success from the extraction array finishing (S3.3b). The launcher refuses a
root that already holds a manifest, so re-running into `extracted_v2` requires
moving the existing one aside deliberately.

C answered NOT GATED, so the aggregation WILL run over a partial extraction
and `cohort_manifest.py` is what refuses it. That is by design and tested
(M9, M14, M15), but it means the only success signal is the PASS line in
`out/cohort_manifest.log`. If A is short of 31/31, `git revert` the SBI Stage
D commit (it is one commit) and report the failing suite.

Stage D writes ONLY under `extracted_v2/`; the launcher refuses the config's
own `extract_root`, an existing manifest, and drifted flags. `extracted/`
stays untouched until Stage C flips `cohort.extract_root`.
