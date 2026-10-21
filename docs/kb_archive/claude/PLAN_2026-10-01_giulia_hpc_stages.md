# The Giulia project on davinci -- staged plan (extraction, joint DSN+NPE training, inference)

| date | change |
|---|---|
| 2026-10-13 | v1.19. G7's `T nan vs p_eff nan` explained -- by construction in arm A1, which has no replicate term `[CODE SBI@9fcbc88; SANDBOX 2026-10-13]`; G5's configuration note annotated: a search trial's batch is the searched `batch_size_npe`, not `B_SIM` (the search keys read and checked, the pipeline plan v1.1, sec. 4a). Sec. 5 G0 and G5; sec. 10. |
| 2026-10-13 | v1.18. **G0 is done; the bank runs as one chained line; the pipeline driver decided** (the user's paste of 14:23 and the cards after it; log v1.57). B2 printed the test task's qsub line, nothing submitted; G8 found the 8 proposals of an earlier run pending (`8 already pending, 0 proposed, 0 NEW`), so every G0 line has run. The user's requirement, one bash command per data set (D-081..D-084), has its work plan, `claude/PLAN_2026-10-13_giulia_pipeline.md`; today's bank is built by the chain line BC (test task, array and gate, real bank, in one `nohup` process; rehearsed `[SANDBOX 2026-10-13]`), the driver's bank stage run by hand. Sec. 5 G0 and G4 annotated; sec. 8; sec. 10. |
| 2026-10-12 | v1.17. **G0's arm A1 trained once on davinci; B5.1 done** (the user's paste of 18:59). G6, the arms' dry run, and G7, one A1 element (job `1829506`, `cpu`, 4 cores, 16 GB, 06:00), passed: the first Stage 3 job on the cluster. LP2 pushed the fix (`e9aeb55`, tree `dc10065fc25a`), D2b passed both suites on davinci, and B1 wrote the plan of record (`bank_plan.json` sha256 `c8303506131a4632`; the B5 plan v1.7, sec. 6). G0 is done but for G8 (and the unpasted G2, G4). Next: B2-B6. Sec. 5 G0 and G4; sec. 8; sec. 10. |
| 2026-10-12 | v1.16. **G3 and G5 passed; B1 refused one task, fixed** (the user's paste of 17:43). G3r: `~/probe_dsn.log` dated 2026-10-12 15:42, `VERDICT: PASS (7/7 checks)`. G5r: both bench banks written (`rows written : 8192` each), sidecar base seeds 0 and 1000003 with G5's argv, `GO`. B1: `REFUSED` on `campaign_cadex_hhgap_v1/sweep_cfd_task0000` -- G2's record counts a topology folder with no simulation, the plan counted only folders with detections; fixed in Sbi-extractor `8d28ae6` (tree `dc10065fc25a`), bundle `b51fix_20261012` (LP2, D2b, B1 again), rehearsed `[SANDBOX 2026-10-12]`. Next: G6, G7, G7r; the outputs of G2, G4 and G8 were not pasted. Sec. 5 G0 and G4: status notes; sec. 8; sec. 10. |
| 2026-10-12 | v1.15. **G1b passed** (the user's paste of 15:17): a job script on `dvnode004` starts with base active (`CONDA_SHLVL` 1, `CONDA_DEFAULT_ENV` base), and the hook block took it to `sbi_env` with `-e` and `-u` restored, torch 2.10.0 imported, `VERDICT: PASS` -- so G5 and G7 run as shipped. Since the login setup runs in job scripts, a dotfile can still override a `qsub -v` value (`HPC_PATHS.md` sec. 6b): G5r now checks, from both sidecars, that G5's values reached the builder; G3r prints the log's date (the 2026-09-19 log is still there). Sec. 5 G0: a status block; sec. 8; sec. 10. |
| 2026-10-12 | v1.14. **G0 started on davinci** (the user's paste of G1r). The laptop push is on GitHub (`feat/real-arm-parity` = `9506d23`, tree `f681a9ece8fc`, the tested one `[GITHUB 2026-10-12]`). G1 recorded bash 4.4.20 on a compute node, which closes F-ay there, and found no `conda` -- in an executable-mode job, which the job scripts are not; G1's design is corrected and G1b, a job script with the scripts' own environment block, decides G5. Sec. 5 G0: the v1.13 paragraph corrected, a status block; sec. 8, a v1.14 paragraph; sec. 10. |
| 2026-10-10 | v1.13. **B5 built; G0 and B5 have a rehearsed run sheet** (the B5 chat; the user, 2026-10-09: "Now the codes to settle both B5 and G0"). Sec. 3's B5 row, G0 and G4 annotated; sec. 8 a v1.13 paragraph. The code: Sbi-extractor `export_windows/` (`specs/SPEC.md` Blocks 6-8; `EXTRACTOR_USAGE.md` sec. 16); the work plan's state: `claude/PLAN_2026-10-08_giulia_B5.md` v1.3. Nothing ran on davinci. |
| 2026-10-09 | v1.12. GC2 is done (102 folders deleted, 0 failed; log v1.53, D-077): sec. 1's D-077 row annotated; sec. 8, sec. 10. The next chat continues from `claude/HANDOFF_2026-10-09_giulia_B5.md`. |
| 2026-10-08 | v1.11. D-078 (the user, 17:10; log v1.52): the Giulia sim bank's `O_N` prior is the declared box [0.01, 3.0], no mixture correction -- sec. 1 row added, the D-074 row and the B6 note annotated. GC1 ran: 102 folders listed (`mea_out_1electrode` 51, `mea_out` 51; two folders under `mea_out/v1/` kept), CODE `e7cf50c2d9d5`; GC2 given, the user's to run (the G2 plan v2.9, sec. 4h). Sec. 8, sec. 10. |
| 2026-10-08 | v1.10. **G3 is done** `[CLUSTER 2026-10-08, the user's paste of 16:16: G3]`: the Giulia label freeze of record, `artifacts/label_axes_giulia_hhgap.json`, sha256 `29e38c89b08a2dd8`, the same as September's (`1ef04131175605a0`) in every field but `created_utc`. The bank's design decided on a card of about 16:30 (log v1.51): D-074 (all 69 tasks, `O_N` declared in [0.01, 3.0]), D-075 (a 0.1 Hz window floor), D-076 (donor = one network); D-077 (the deletion rule). B5 has its work plan, `claude/PLAN_2026-10-08_giulia_B5.md`, written against the joint stack's code: the runner's prior is the unit box and its prior floor is the bench's (`run_joint_arms.py:385-396, 539, 557` `[CODE SBI@9fcbc88]`), so B6 gains a campaign prior. Changed: sec. 1 (rows D-074..D-077), sec. 3 (B5, B6 annotated), sec. 4 (the label-freeze row), sec. 5 (G3's heading and status; G4's heading and a note), sec. 8, sec. 10. |
| 2026-10-08 | v1.9. **N1 ran; G3's write is ready** `[CLUSTER 2026-10-08, the user's paste of 15:20: N1]`: G3's report-only pass printed the September freeze of record's figures (4,467 topologies sampled, `conn_rule` flat, `conn_prob` 2,612 distinct in [0.0503601, 0.399682], the three kernel axes excluded, the topology block `['conn_prob']`, 0 unreadable files); the 18 cross-contract pairs differ in `O_N` alone, one uniform draw in two boxes; `mea_out_1electrode` is covered by the record task folder by task folder (log v1.50). G3's write is the line `G3` of `claude/PLAN_2026-10-06_giulia_G2.md` v2.7, sec. 4g, rehearsed `[SANDBOX 2026-10-08]`. Changed: sec. 4 (the sim-detections and label-freeze rows annotated), sec. 5 (G3's heading and a status block), sec. 8 (where this stands), sec. 10. |
| 2026-10-08 | v1.8. **G2 is done** `[CLUSTER 2026-10-08, the user's paste of 14:21: G2h]`: the Giulia record, `ANN/MEA_analysis/mea_out_giulia_v2/REEXTRACTION_RECORD.json`, sha256 `7af3759a0f3c1350`, written 2026-10-07 at the end of the gate job (17:36-17:57): 69 tasks, 1,260,207 iterations, 27 replays excluded, 0 warnings (log v1.49). Since v1.7: C8's record closed Stage C (2026-10-07 13:11, D-033); G2 ran from its own work plan, `claude/PLAN_2026-10-06_giulia_G2.md` (v2.6), under D-066 and D-069..D-072. Changed: sec. 1 (rows D-066, D-069..D-072), sec. 2 (a note), sec. 3 (B3, B4 annotated), sec. 4 (three rows annotated), sec. 5 (G2's heading and a status block; G3's command corrected against `preflight_label_axes.py`'s interface; G4's sizes corrected), sec. 6 (not taken, D-070), sec. 7 (O3), sec. 8 (where this stands), sec. 10. Evidence: the user's pastes of 2026-10-07 and 2026-10-08 as logged (log v1.43-v1.49); `[CODE Sbi-extractor@4e4dff0]` `preflight_label_axes.py:140-150`; the sizes `[reasoning; the arithmetic run]`. |
| 2026-10-02 | v1.7. **C8 runs now; G2 waits for its record** (the user, 16:01: "C8's record do this now"; log v1.21, D-057), so O8 is settled and the order after G1 is again Stage C first, then the Giulia project. The run sheet for C8 is re-issued in `claude/HANDOFF_2026-10-01_c8_push_and_run.md` v1.2, P10 (gated steps S1-S6, rehearsed 21/21 twice). The same message asked for a command that deletes the two refused G1 runs' roots and logs: given, guarded on the cohort manifest's digest (addendum v1.6). Changed: sec. 1 (D-057 row), sec. 4 (the extract root row), sec. 5 (G2's heading), sec. 7 (O8 settled), sec. 8, sec. 10. |
| 2026-10-02 | v1.6. **G1 is done** (the user's paste of 15:31; log v1.20). The third run -- the gated commands C1-C6 of v1.5 -- printed every expected line: `GO 1 ... (2c0a06d / 0a560ca)`, `PREFLIGHT PASS (7/7)`, `PROBE OK; CONFIG MATCHES (16 wells to extract)`, the dry run's 16 wells, the launch (array `1777495[]`, aggregation `1777496`), and the aggregation's PASS: `extracted_giulia/cohort_manifest.json`, sha256 `21b7ef03bb7f776b`, 16 wells x 9 subregions = 144 units, at `0a560ca`, `cohort_manifest_exit=0`; the manifest's summary as expected `[CLUSTER 2026-10-02; SANDBOX 2026-10-02: the paste checked line by line against the rehearsal, 12/12]`. Found while reading G1's "Then" against the code: the frozen manifest copy is made by the sim-arm launcher, not by hand, and C8's launcher freezes into one fixed folder, `artifacts/cohort_manifest/`, and reuses whatever copy is there -- a trap B3 must close. Changed: sec. 3 (B3 annotated), sec. 4 (the extract root and frozen-copy rows), sec. 5 G1 (heading; a v1.6 status block; the "Then" annotated), sec. 7 (O8 added), sec. 8, sec. 9 (one trap added), sec. 10. |
| 2026-10-02 | v1.5. **The second G1 run, and gated commands for the third** (the user's paste and call of 15:09; log v1.19, D-056). The G1-fix commits are on GitHub with the tested trees (SBI `2c0a06d`, Sbi-extractor `0a560ca`, 15:03-15:04), but davinci pulled before they landed, so the second run (15:07) used the old code and config: the probe said `CONFIG DIFFERS on: exclude_wells`, the launch went ahead, the aggregation refused the same two wells. The reply of 13:50 had put the move, the dry run, the launch and the `cat` in one code block, with nothing checking the probe's line; the third run's six commands (C1-C6) are each gated on the one before, rehearsed at the real paths, 16/16 including the negative paths `[SANDBOX 2026-10-02]`. D-056: real-arm windows of 180 s, back to back, from t = 0 (O6 settled). Changed: sec. 1 (D-041 annotated; D-056 row), sec. 5 G1 (a v1.5 status block with C1-C6), sec. 7 (O6), sec. 8, sec. 9 (two traps added, one annotated), sec. 10. |
| 2026-10-02 | v1.4. **The re-run of G1 is ready** (the user's call of 13:12, "It's ok if some wells are excluded because there is no activity", with the three outputs asked for; log v1.18, D-055): the two refused wells are the probe's `too_few_active` wells -- `ptrain_44456_DIV35_100N0A_nbasal_0001` (V of 1 electrode, task 3) and `ptrain_45198_DIV35_100N0A_nbasal_0001` (V of 5, task 4), both Batch1 0:100 -- and are excluded by name; the probe's `T_rec by well` gives 300 s for all ten 0:100 wells and 600 s for all eight AraC wells. Built and rehearsed `[SANDBOX 2026-10-02]`: `g1fix_bundle_20261002.tar.gz` -- SBI (the config's `exclude_wells`, its README) and Sbi-extractor v8.10 (the launcher's job line and last advice, the aggregation's exit line on a refusal, refusals that name the listing, each well's array index and what happened; `EXTRACTOR_USAGE.md` v8.9 + v8.10, sec. 15) -- on the real heads (SBI `ab7ea82`, Sbi-extractor `481bd48`), 17/17, including the whole re-run on a replica of the census (probe `CONFIG MATCHES (16 wells to extract)`, 16 x 9 = 144 units, `T_rec_range` [300, 600]). Changed: sec. 1 (D-040, D-041 annotated; D-055 row), sec. 3 (B1, B2c annotated), sec. 5 G1 (a v1.4 status block; the run sheet for the re-run is the new bundle's README; two "must appear" lines corrected), sec. 7 (O2, O5 settled; O6, O7 added), sec. 8, sec. 9 (two traps updated), sec. 10. |
| 2026-10-02 | v1.3. **G1's first run** (the user's terminal log, pasted 10:32; log v1.17): the Giulia commits are on GitHub with the tested trees (SBI `745edce`, Sbi-extractor `481bd48`); `PREFLIGHT PASS (7/7)` on davinci; the probe `PROBE OK; CONFIG DIFFERS on: exclude_wells`, every other checked field as configured, so the `sparse_peaks` / `peak_train` format is confirmed; the single well 39489 gave `X.shape=(9, 60000)`, i.e. T_rec = 600 s, not 900; the array `1776556[]` ran anyway and the aggregation `1776557` REFUSED 2 of 18 wells (no manifest). Changed: sec. 1 (D-040, D-041 rows annotated), sec. 5 G1 (a status block at its top; run-sheet steps 5 and 7 corrected), sec. 7 (O1 settled, O2 live, O5 added), sec. 8 (where it stands), sec. 9 (four traps), sec. 10. |
| 2026-10-01 | v1.2. **G1 runs now** (the user, 17:27: "Now let's push and pull and extract the biological data"; log D-053), ahead of the C8 run. Since v1.1: the bundle is re-issued as `giulia_stage_d_bundle_20261001_v2.tar.gz` (59,804 B, sha256 `03a10063d431db49a6cc16c2a6bcaf6c904a4527f951806de438732cbf7e3fc4`; v1 superseded, never applied): `extraction_flags_giulia.sh` ships in the patch (no first-run stop), `probe_ptrain_tree.py --config` ends on `CONFIG MATCHES` or names what differs, `preflight_cohort.sh` replaces `cluster_check.sh` (7 steps, last line `PREFLIGHT PASS (7/7)`), `smoke_test_stage_d_jobs.py` runs the launcher and both PBS jobs end to end through a fake `qsub` (8 checks), `smoke_test_ptrain_formats.py` is 73 checks. GitHub moved at 17:38: C8 (Sbi-extractor `b20dd88`, ANN `04977e0`) and the raster commit (`5e800b6`) are on origin, each C8 commit with its tested tree (D-034 done); the Giulia patch applies on `5e800b6` unchanged, rehearsed on the real heads (15/15: both push scripts PASS with the trees `79a353e5...` SBI and `5ec712b0...` Sbi-extractor; preflight 7/7; the C8 suite 23/23 and the raster suite 12/12 on the combined tree) `[SANDBOX 2026-10-01]`. Changed: sec. 2 (a note under the map), sec. 3 (B1, B2, B2a, B3 annotated; B2c new), sec. 4 (the flags row corrected; two rows), sec. 5 (G1: the run sheet at its top, v1.1's steps kept and marked superseded; G2's heading), sec. 8 (where it stands), sec. 9 (two traps), sec. 10. |
| 2026-10-01 | v1.1. The user's answers of about 16:10 to P1-P10 (log v1.12, D-047..D-051) with the `ls -lhR` listing of `Bio_Data/`, and the builds of the same turn: the recordings are 18 wells at DIV 35 under four condition folders, in files `ptrain_<well>_DIV35_<cond>_nbasal_0001_<rc>.mat` that are NOT the extractor's format (sec. 3 B2b, sec. 9); B1 (`config_giulia_cohort.davinci.json`), B2 (`launch_stage_d.sh` with `COHORT_TAG`, both `.pbs` files taking `-v CHSUB_MANIFEST/CHSUB_FLAGS`), B2a (`extractor/probe_ptrain_tree.py`, read-only) and B2b (the loader's declared `ptrain_format` / `ptrain_varname` / `ptrain_name_pattern`, `exclude_wells`, the manifest's `source_format`) are built, sandbox-verified (67 + 17 + 9 + 4 checks on clean clones with the patches applied) and delivered as `giulia_stage_d_bundle_20261001.tar.gz` (sha256 `48cb2068...`); nothing has run on davinci. Sec. 1 (D-047..D-051), 2, 3, 4, 5 (G1 is no longer blocked: probe, then the two config fields, then the array; G2's flags are fixed: `--edge 26.59 --fs 10000`), 7 (answers), 8 (order: G1 first, D-051), 9, 10 rewritten accordingly. The joint-stack stages G0, G3-G6 and B3-B8 are unchanged. |
| 2026-10-01 | v1. Written the turn the user answered the Giulia handoff's eleven questions (15:35 Europe/Rome; log D-039..D-046) and asked for a staged plan to run on the HPC while the biological recordings are uploaded. Built from: the decisions log v1.11; `claude/HANDOFF_2026-10-01_giulia.md` v1.1; `JOINT_DSN_NPE_USAGE_v1.md` v1.3; `claude/HANDOFF_2026-10-01_c8_push_and_run.md`; `EXTRACTOR_USAGE.md` secs. 12-14; `HPC_PATHS.md`; and the code at SBI `834eb41` (`hpc/joint/stage1/{latent_bank,build_latent_bank}.py`, `JOINT_DSN_NPE_PLAN_v0_6.md` Stage 6, the four joint job scripts), Sbi-extractor `63d2574` (`example_export.py`, `real_source.py`, `extractor/launch_stage_d.sh`, `dataset_profile.py`, `preflight_giulia_export.py`), ANN `37bf9f8` (`mea_probe.py`, `process_campaign.py`). Nothing here has run; nothing is built yet. The C8 commits (`sim_reextract/`) are not on GitHub as of this write, so the Giulia adaptation of C8 is specified against `EXTRACTOR_USAGE.md` sec. 14 and the C8 handoff, not against their code. |

Read first: `claude/SBI_decisions_and_ideas_log.md` (D-039..D-051 are this
plan's decisions; D-002, D-005, D-006, D-010..D-012, D-015 carry over);
`claude/HANDOFF_2026-10-01_giulia.md` (v1.3: what is on record about the
project); `JOINT_DSN_NPE_USAGE_v1.md` (how the joint stack runs);
`claude/HANDOFF_2026-10-01_c8_push_and_run.md` (the C8 run this plan's sim
arm copies). Tags as in `HPC_PATHS.md`: `[STATED]` the user; `[KB]` a project
document; `[CODE <repo>@<commit>]` read from source; `[SANDBOX]` run here;
`[reasoning]` Claude's; `[TO VERIFY]` not yet checked; `[proposed]` Claude's
choice, not decided.

## 0. In one paragraph

Three real culture conditions -- astrocyte:neuron 0:100 (untreated), 50:50
and 30:70 (the ratio kept during maturation with AraC) -- 18 wells at DIV 35
recorded on a 60-electrode MEA (8 x 8 minus the corners, 200 um pitch, 30 um
electrodes, 10 kHz) `[STATED, D-040, D-042, D-047; LISTING]`, and the
hhgap simulations of the same astrocyte-neuron model (300 um cultures,
`Nn = Na = 108`, 22 swept axes + `conn_prob`, 180 s each) `[KB, HPC_PATHS.md
4e]`. The plan: (1) extract the real recordings into nine single-electrode
IFR traces per well (D-043, D-044) with their cohort manifest; (2) re-detect
every deduplicated simulation with one virtual electrode at the device's
geometry and rate (D-045, D-042); (3) turn both into Stage 1 banks of the
joint stack (a builder that does not exist yet); (4) train the joint DSN+NPE
on the simulated bank alone, arm `A1`, then run the Stage 4 search on it
(D-039; configuration to be discussed); (5) freeze the trained encoder,
export both arms' embeddings, run the misspecification gate, draw the
posteriors of every real well and compare the conditions. Plain version:
teach the network on simulations only, then ask it what parameters would
produce each real culture, and see how the answer moves with the astrocyte
ratio.

## 1. What is decided, and what each decision fixes

| decision | fixes | source |
|---|---|---|
| D-039 joint DSN+NPE, arm trained only on simulated data | the stack is `~/SBI/hpc/joint`; the arm is `A1` `[Claude's reading]`; "optimization" is the Stage 4 campaign `S-A1`; configuration "afterwards" | `[STATED]` |
| D-040 three conditions, AraC-kept ratios; recordings to be extracted | K = 3; 15-min recordings for the AraC conditions, variable for 0:100; paths and counts: D-047 **[2026-10-02: one AraC well measured at 600 s; O5]** **[v1.4: every well measured by the probe -- 300 s for the ten 0:100 wells, 600 s for the eight AraC wells (O5 settled)]** | `[STATED]` |
| D-041 windows 180 s; simulations untrimmed | W_s = 180 s, W = 18000 at 100 Hz; one window per simulation; 5 windows per 15-min real trace at stride 180 s **[3 per trace if the AraC wells are 600 s, O5]** **[v1.4: 3 per AraC trace (600 s) and 1 per 0:100 trace (300 s) at stride 180 s; the real-arm stride is O6]** **[v1.5: the stride is decided -- D-056]** | `[STATED]`, `[reasoning]` for the counts |
| D-042 10 kHz, 30 um, 200 um, positions from 1; sim polarity negative | `--pitch 200`; the rate, the grid decoding and `--edge` are D-048 / D-049; its "`ptrain_<k>.mat` rasters" statement is corrected by D-047 (the files are another format) | `[STATED]` |
| D-043 E = 1, C = 9 | `electrodes_per_subset 1`, `n_subsets 9`, `n_side 1`, `n_e = 1` both arms; the nine most active electrodes per well (the extractor's rule, confirmed by P6 "for now") | `[STATED]` |
| D-044 preprocessing "the default" | `w_size 0.01`, `gaussian_window 0.02`, `mfr_threshold 0.1` (confirmed by P7) -- stated explicitly in the config, never left to the code's 0.02 / 0.04 | `[STATED]` |
| D-045 all hhgap campaigns, new extractor, replicated seeds out | v1, v2, v4, v5 re-detected into a new root; one task per `_resolved_seed_master`; `mea_out*` untouched | `[STATED]` |
| D-046 names carry "giulia" | the names of sec. 4 | `[STATED]` |
| D-047 the recordings: `Giulia_Astro/Bio_Data/`, four condition folders, 18 wells at DIV 35; `100N0A_wo` is the untreated 0:100; the files are not the extractor's format | `class_roots` = `Batch1/100N0A_wo`, `Batch2/100N0A_wo` (class 0), `Batch3/AraC/50N50A` (1), `Batch3/AraC/70N30A` (2); `class_names = ["100N0A_wo", "50N50A", "70N30A"]`; no age in the culture id; the loader reads a DECLARED format (B2b) set from the probe (B2a) | `[STATED; LISTING]` |
| D-048 `fs_raw` 10 000 Hz exactly; positions numbered from 1; the MCS code decoded with `grid_width 10`, `index_base 0` | `fs_raw 10000.0`, `grid_width 10`, `index_base 0` in the config (`row = k // 10`, `col = k % 10`; `12 -> (1, 2)`, `87 -> (8, 7)`); `--fs 10000` on the sim arm | `[STATED; LISTING; SANDBOX]` |
| D-049 `--edge 26.59` | the sim-arm `EXTRA_ARGS` are now fixed: `--n_side 1 --pitch 200 --edge 26.59 --fs 10000` | `[STATED]` |
| D-050 the five `q/` units are not used | listed under "outside the glob" in the plan step; not frozen, not in the bank | `[STATED]` |
| D-051 the stages run in order, G1 first | G2 takes `n_side` and `fs` from the Giulia cohort manifest (D-015), not from the config ahead of it | `[STATED]` |
| D-056 (v1.5) real-arm windows: 180 s, back to back, from t = 0 | stride 180 s: windows [k x 180, k x 180 + 180) s; one per 0:100 trace (300 s; 120 s unused), three per AraC trace (600 s; 60 s unused); 72 windows for class 0, 108 per AraC class with the 16 wells | `[STATED]`; the counts `[reasoning]` |
| D-055 (v1.4) the two wells below nine active electrodes are excluded by name | `exclude_wells` = `ptrain_44456_DIV35_100N0A_nbasal_0001`, `ptrain_45198_DIV35_100N0A_nbasal_0001` (V of 1 and 5 electrodes at 0.1 Hz; both Batch1 0:100); the cohort is 16 wells, 8 / 4 / 4, 144 units; `mfr_threshold`, C and E unchanged | `[STATED]` |
| D-057 (v1.7) C8 runs now; G2 waits for C8's record | G2 (sec. 5) starts only once `ANN/MEA_analysis/Outputs_v2/REEXTRACTION_RECORD.json` exists; C8 runs now from the C8 handoff's P10; when B3 is built is not decided (Claude's offer: while C8 runs) | `[STATED]` |
| D-066 (v1.8) the Giulia `mea_out_1electrode` is deleted only after G2's record | listed task folder by task folder against the record (N1), then deleted by the user | `[STATED]` |
| D-069 (v1.8) "re-extract everything" = the simulations (G2), every hhgap campaign; G1's 16 wells kept | G2's scope; done 2026-10-07 (the record, 69 tasks) | `[STATED]` |
| D-070 (v1.8) G2's replays by the C8 replay gate as built (D-063, D-064) | one task per seed within a label contract, the longest copy kept; 27 replays dropped; the 18 same-seed pairs across the two contracts kept (D5: different simulations); sec. 6's rule not taken | `[STATED]` |
| D-071 (v1.8) the MEA jobs use every core of their node | one `process_campaign.py` worker per core, from `NCPUS` (48 on the `cpu` nodes) | `[STATED]` |
| D-072 (v1.8) each simulation's recording noise seeded from its own `seed_run` | the noise scheme `sim` from G2 on; C8's record keeps the old scheme | `[STATED]` |
| D-074 (v1.10) the simulated bank takes all 69 record tasks, `O_N` declared in [0.01, 3.0] | one contract for every sim shard; the `O_N` training prior a known mixture of the two boxes (the 20 [0.03, 3.0] tasks inside it), corrected wherever the prior is used (B6, G5, G6) **[v1.11: not corrected -- D-078]** | `[STATED]` |
| D-075 (v1.10) a 0.1 Hz floor on each simulated window's mean rate | a row enters the bank only if mean_t x(t) >= 0.1 Hz; p(theta | x) unchanged for the kept windows, the bank's theta-marginal tilted | `[STATED]` |
| D-076 (v1.10) donor = one network | the split's groups: a topology's iterations, twin topologies merged by seed | `[STATED]` |
| D-077 (v1.10) delete the old extractions of everything newly re-extracted | `mea_out_1electrode` and `mea_out`'s covered task folders: GC1 lists, GC2 deletes (the user); `SBI_export_hhgap_1e_r2` not covered, as read **[v1.12: done 2026-10-09 -- 102 folders deleted]** | `[STATED]` |
| D-078 (v1.11) the sim bank's `O_N` prior is the declared box [0.01, 3.0]; no mixture correction | B6, G5 and G6 take the unit box as the prior; on `O_N` below 0.03, where the data do not constrain it, posteriors under-weight it by up to the mixture's density ratio (the log, D-078's Implies) | `[STATED]` |

## 2. The pipeline, as a map

```
REAL ARM (ready: probe -> config -> array)            SIMULATED ARM (after G1's manifest, D-051; needs the C8 push)
Giulia_Astro/Bio_Data/Batch{1,2}/100N0A_wo,            Giulia_Astro/campaign_cadex_hhgap_v{1,2,4,5}
  Batch3/AraC/{50N50A,70N30A}/ptrain_<id>_DIV35_...       job_args.json (_resolved_seed_master), manifest.json v3, topo_*/iter_*.npz
  ptrain_..._<rc>.mat per electrode (declared format)
      |  G1  extractor (Stage D chain), E=1, C=9              |  G2  seed dedup + re-detection, n_side 1, pitch 200, edge 26.59, fs 10000
      v                                                        v
Giulia_Astro/extracted_giulia/<class>/<root>/<well>/trace_subregion_0[0-8].npz   mea_out_giulia_v2/<campaign>/<sweep>/topo_*/mea_iter_*.npz
  + cohort_manifest.json (+ .sha256; source_format, excluded_wells)                             + mea_env.json per task, REEXTRACTION_RECORD.json (+ .sha256)
      |                                                        |  G3  label freeze: label_axes_giulia_hhgap.json (p = 23)
      |  G4  real-window bank (theta_present: false)           |  G4  sim-window bank: x = IFR window (W = 18000), theta (23)
      v                                                        v
~/banks/giulia_real/shard_*.npz + .json                   ~/banks/giulia_sim/shard_*.npz + .json      <- latent_bank contract
                                                               |  G5  joint training, arm A1 (+ A_ref, shuffled [proposed]); Stage 4 search S-A1
                                                               v
                                                      ~/runs/giulia_arms/, ~/tune_joint_giulia/ ; the trained encoder psi* and flow
      |                                                        |
      +-------------------------- G6 ----------------------------+
      freeze psi* -> artifacts/frozen_dsn/dsn_giulia_A1_<date>.pt (+ .sha256)
      export both arms' embeddings -> ANN/SBI_export_giulia/ ; gate_run.py -> witness_run.py
      posterior q(theta | z_well) per real well; per-condition comparison; replicate statistic across same-condition wells
```

**[v1.2, 2026-10-01]** The C8 push the simulated arm needed has happened (Sbi-extractor `b20dd88`, 17:38): that arm now waits for G1's manifest (D-051) and for B3 alone. G1 runs now, ahead of the C8 run (D-053).

**[v1.8, 2026-10-08]** Both arms' first stages are done: G1 (2026-10-02, the cohort manifest `21b7ef03bb7f776b`) and G2 (2026-10-07, the record `mea_out_giulia_v2/REEXTRACTION_RECORD.json`, sha256 `7af3759a0f3c1350`). G3's freeze is next; B5 and the banks after the design questions (sec. 8).

The two domains and the map between them, so no object is in two places:
parameter space Theta, a box in R^23 (22 swept axes in their declared
coordinates + `conn_prob`, linear); data space, IFR windows x in R^18000
(one electrode, 100 Hz, 180 s); the simulator-plus-virtual-MEA map
Theta -> R^18000 on the sim arm; the real recordings land in R^18000 through
the extractor with no theta attached (`theta_present: false`, joint plan
D6-1). The encoder psi: R^18000 -> R^E and the flow q(theta | z) are what G5
trains.

## 3. What exists and what must be built

| id | piece | state at SBI `834eb41` / Sbi-extractor `63d2574` + C8 bundle | needed change | smoke test |
|---|---|---|---|---|
| B1 | Giulia cohort config `hpc/dsn/hpc/Config/config_giulia_cohort.davinci.json` | **built `[SANDBOX 2026-10-01]`**, in the bundle's SBI patch, with its README beside it: the `cohort` block of D-043, D-044, D-046..D-048 (`class_roots` the four folders, `fs_raw 10000.0`, `grid_width 10`, `index_base 0`, `n_subsets 9`, `electrodes_per_subset 1`, 0.01 / 0.02 / 0.1, `ptrain_name_pattern` from the listing, `extract_root .../Giulia_Astro/extracted_giulia`); the other blocks copied from the DUP15HD config and NOT the Giulia training plan | two fields wait for the probe: `ptrain_format` / `ptrain_varname` (set to the `sparse_peaks` / `peak_train` hypothesis) and `exclude_wells` (empty) **[v1.2: the probe checks the config itself (`--config`); `CONFIG MATCHES` means the hypothesis held and nothing is patched]** **[v1.4: the format held (2026-10-02); `exclude_wells` = the two wells of D-055, in the G1-fix SBI commit, pending push]** | `cohort_config.py <config>` prints the block and the flags line (cluster check step 5); `Smoke_Tests/smoke_test_cohort_fields.py` (4, torch-free); `smoke_test_mea_specs.py` check G (cluster, `sbi_env`) |
| B2 | real-arm extraction for a second cohort | **built `[SANDBOX 2026-10-01]`**: `launch_stage_d.sh` takes `COHORT_TAG` (per-cohort manifest `out/extraction_manifest_<tag>.tsv`, flags `extraction_flags_<tag>.sh` written on the first run and then required COMMITTED, job names and logs), passes `-v CHSUB_MANIFEST=...,CHSUB_FLAGS=...` to `run_extractor_array_mea.pbs` and `run_cohort_manifest.pbs` (which default to the DUP15HD names when unset); a first extraction into a declared `extract_root` that does not exist yet is allowed, an existing non-empty one still refused; untagged runs are unchanged **[v1.2: `extraction_flags_giulia.sh` ships in the Sbi-extractor commit, generated by `build_extra_flags` from the config; the launcher compares it with the config on every run, so the first run no longer stops]**. `PITCH_UM` is left alone: it enters nothing in the extraction (`electrode_coords` is not called) | none | `smoke_test_ptrain_formats.py` F14 (the launcher in DRYRUN: new-cohort stop, both qsub lines, the refusals); the untouched Stage D suites still pass |
| B2a | the file probe, `extractor/probe_ptrain_tree.py` | **built `[SANDBOX 2026-10-01]`**, read-only, `scipy.io.loadmat` only: per well the name parse, the index decoding (the MCS reading vs row-major 8/1), what every `.mat` holds (storage class, shape, dtype, nonzeros, range), `n_samples` / `T_rec` against `--expect-t-rec`, spike counts, MFRs, the active count against C x E, a verdict and the nine the extractor's rule would pick; a SUGGESTED cohort fragment; exit 1 when undetermined **[v1.2: `--config` reads the cohort block and takes `fs_raw`, `mfr_threshold`, `n_subsets`, E and `well_glob` from it (a flag overrides it), then compares: the probe's too-few wells must all be in `exclude_wells`, the declared format and variable must be what the files hold, the name pattern must match every file; last line `PROBE OK; CONFIG MATCHES (<n> wells to extract)` (exit 0) or `PROBE OK; CONFIG DIFFERS on: <fields>` (exit 3)]** | **the first thing to run on davinci** (sec. 5, G1); its `SUGGESTED` block fills B1's two fields | F13 (a Giulia-like tree, a 3Brain-like tree, a mixed tree) |
| B2b | the loader reads a DECLARED format | `channel_subset_extraction.py` read ONE format: `ptrain_<k>.mat`, a dense binary raster `ptrain` (`:118-119, 211-282`); the Giulia names do not match and the sizes exclude dense rasters (D-047) | **built `[SANDBOX 2026-10-01]`**: `ptrain_format` (`raster` = the old behaviour and default; `sparse_peaks` = every stored nonzero of a scipy.sparse column is a spike), `ptrain_varname`, `ptrain_name_pattern` (one capture group), validated in `hpc/dsn/cohort.py` (`validate_ptrain_fields`, shell-safe), threaded through `load_ptrain_folder`, `extract_channel_subsets`, the CLI (`--ptrain-format/--ptrain-varname/--ptrain-name-pattern`), `build_extra_flags` (appended only when non-default: the DUP15HD flags stay byte-identical), the fragment (extractor version 4) and the manifest's non-gating `source_format`; a file of the other kind refused naming the flag. `exclude_wells` in the cohort block, applied by `find_wells(exclude=...)` in `list_extraction_jobs.py` (an unmatched name aborts) and `make_mea_specs.py` (`--strict` fails), recorded as `excluded_wells` | if the probe shows a third format (a dense vector of spike times, say), one more branch | `smoke_test_ptrain_formats.py` F1-F12 (67 checks in all, through the real CLI, the manifest and the launcher); `smoke_test_cohort_manifest.py` 17 at version 4; `smoke_test_extraction_metadata.py` 9 **[v1.2: `smoke_test_ptrain_formats.py` is 73 checks, F13h-F13l for `--config`]** |
| B2c | **v1.2.** login-node preflight; the job layer tested end to end | the job scripts had been tested only through the launcher in DRYRUN (F14) | **built `[SANDBOX 2026-10-01]`**: `extractor/preflight_cohort.sh CONFIG TAG` (read-only: no CR in any `.sh`/`.pbs`; every `.py` ASCII; compile; the DSN tree carries the source-format fields; the config loads and the tracked flags file equals `build_extra_flags(config)`; the DUP15HD `extraction_flags.sh` byte-identical; five suites) -- last line `PREFLIGHT PASS (7/7)`; `extractor/smoke_test_stage_d_jobs.py`, a fake `qsub` that runs each job inline from a spool copy with `PBS_ARRAY_INDEX`, `PBS_O_WORKDIR` and the `-v` variables, a stub conda, a synthetic SpyCode-like cohort: J0 one well with the tracked flags and its figures ... J7 a too-few well left out of `exclude_wells` makes its task fail and the aggregation REFUSE naming it | itself: 8 checks, about 40 s; run by the preflight **[v1.4: 11 checks -- J1's job line and last advice, J7b (the refusal's listing, array index, wording and exit line), J8/J8b (a re-launch over a refused root is refused; allowed once the root is moved aside); `smoke_test_cohort_manifest.py` 18 (M18: the three ways a fragment can be absent)]** |
| B3 | sim-arm re-detection for the Giulia set | C8's `sim_reextract/` (not on GitHub yet **[v1.2: on GitHub since 2026-10-01 17:38, `b20dd88`; B3 is cut against its code]**; `EXTRACTOR_USAGE.md` sec. 14): glob `campaign_cadex_rho1300v*`, `n_side` from the frozen cohort manifest, `EXTRA_ARGS --n_side 3 --pitch 60.0 --edge 25.0 --fs 10110.09`, one-contract check, exclusions named, tool fingerprints, gate + record | parameters: glob `campaign_cadex_hhgap_v*`, SIM root `Giulia_Astro`, out root `mea_out_giulia_v2`, `EXTRA_ARGS --n_side 1 --pitch 200 --edge 26.59 --fs 10000` (D-048, D-049); `n_side` and `fs` from the Giulia cohort manifest, as C8 does (D-051: no config shortcut); the `q/` units listed as outside the glob (D-050); `manifest_version 3` accepted; a seed-dedup step (sec. 5, G2) writing the kept and dropped tasks into `plan.json` and the record; `--allow-mixed-contract` scoped to v1's `O_N` bound (both groups `ln`, no coordinate flip) **[v1.6: and the freeze. C8's launcher copies `COHORT_MANIFEST` and its `.sha256` (checked against each other) into the fixed `$ARTIFACTS_DIR/cohort_manifest/`, and once a copy is there it uses that copy without comparing it with `COHORT_MANIFEST` (`launch_sim_reextract.sh:91-108` `[CODE Sbi-extractor@0a560ca]`). B3 freezes the Giulia manifest into `artifacts/cohort_manifest_giulia/` (D-046) and refuses a frozen copy that differs from the manifest it was given; C8's own launch stays as it is (sec. 9)]** **[v1.8: built -- Sbi-extractor `4e4dff0`, `COHORT_TAG=giulia` with `profiles/giulia.sh` (`EXTRACTOR_USAGE.md` v8.14, sec. 14); the replays by the C8 gate as built (D-070), not by sec. 6's rule; run as G2 (2026-10-07)]** | the 23-check suite re-pointed at a synthetic hhgap tree, plus checks for the dedup (a replayed pair, a same-seed pair with different theta -> refusal) |
| B4 | label freeze | `preflight_label_axes.py` with `--exclude` for the three NaN kernel axes (trap 6.5); C3's `coord` declaration is designed, not built | run over the kept tasks of G2; `artifacts/label_axes_giulia_hhgap.json`; the `coord` field if C3 lands first, else the manifest-derived coordinates at export (D-004 as it stands) **[v1.8: G2 is done; the call is G3's, corrected in sec. 5]** | the existing `smoke_test_label_axes_required.py`, `smoke_test_registry_from_manifest.py` |
| B5 | **window exporter -> Stage 1 bank** | **does not exist.** The joint stack's Stage 1 builds bench banks from three synthetic providers (`build_latent_bank.py --provider reference|dsn|bench`); its arms read only `shard_%04d.npz` + sidecar in the `latent_bank` contract (`CONTRACT_FIELDS`, `ROW_ARRAYS`: `x (n, W)`, `theta (n, p)`, `cls`, `nu`, `realisation_id`, `donor`, `well`, `batch`, `subregion`, `window_idx`, `contaminated`) `[CODE SBI@834eb41, latent_bank.py:27-59]`. The joint plan's Stage 6 names `Sbi-extractor/export_windows.py` for this and it was never written (usage v1.3 sec. 10). What exists: `example_export.iter_campaign_records` already yields the IFR window x and `theta_A` per simulation (`example_export.py:155-288`), and `real_source.py` yields the archives' `ifr_trace` with culture / condition ids | `Sbi-extractor/export_windows.py`: sim mode over `mea_out_giulia_v2` + `Giulia_Astro` with `LABEL_AXES`, `dt`/`sigma_sm` from the manifest (D-005), `n_e` asserted = 1, writing shards through `latent_bank.build_sidecar` / `write_shard` (`param_names`, `bounds_theta`, `coord`, `fs 100`, `w_size 0.01`, `T_win 180`, `W 18000`, `scale_convention per_electrode_mean`, `theta_withheld false`, `arm S`; `cls = -1`; `donor`/`well`/`batch` from campaign / topology / iteration; `realisation_id` from `seed_run`); real mode over the Giulia archives via the specs, windows of 180 s at stride 180 s, `theta` NaN with `theta_present: false` (joint plan D6-1), `cls` = the condition index, `donor` = culture; refuses on any preprocessing field disagreeing with the manifest (D-002). Contract questions to settle at build: `coord` is one string in the sidecar while the axes are 20 `ln` + 3 linear; `latent_spec` / `nuisance_spec` / `realisation_spec` / `gap_spec` for a campaign bank; `generator_sha256` = the record's digest | a suite that writes a shard from a synthetic campaign and reads it back with `latent_bank.read_shard`, checks `x.shape[1] == 18000`, the digest, the refusal on a mismatched `w_size` **[v1.10: its work plan is `claude/PLAN_2026-10-08_giulia_B5.md` (D-074..D-076): the contract as read, the design table, five blocks]** **[v1.13, 2026-10-10: built, as the folder `export_windows/` rather than one script -- plan step, one array element per task, a gate writing `BANK_RECORD.json`, the real mode, a launcher and two job scripts; 28 + 14 smoke checks and 62 mutations caught `[SANDBOX 2026-10-10]`; the real bank's theta has zero columns (`theta_withheld` false) rather than NaN; the B5 plan v1.3 sec. 5]** |
| B6 | joint training on the Giulia bank | `stage3/run_joint_arms.py` + `jobs/joint_arms.pbs` (arms by index; 4 cpus, 16 GB, 6 h; `EPOCHS 10`, `STEPS_PER_EPOCH 25`, `B_SIM 128`, `N_DRAWS 128`, bench-sized) and `stage4/npe_tune_joint.py` + `jobs/joint_tune.pbs` (`--p`, `--d-theta`, `--embedding-dim` default 26/26/12) `[CODE SBI@834eb41; KB usage v1.3 secs. 5, 7]`; **nothing in `hpc/joint` has run on the cluster** (usage sec. 9) | a Giulia run config: `--d-theta 23 --p 23`, the embedding dimension E, epochs, steps, batch, `N_DRAWS` >= 4 x 23 = 92 (the floor; 128 clears it), the grouped split, resources sized to the bank -- all "configuration, afterwards" (D-039); the D-038 fix patches (`num_bins`, `head_pool_ops`) land first if the documentation chat delivers them | the existing `smoke_test_joint_arms.py` (33), `smoke_test_joint_tune.py`; one bench-bank element on davinci before any Giulia element (G0) **[v1.10: and, before any Giulia element, a campaign prior in `run_joint_arms.py`: the runner builds the bench's `LatentSBISpec` from the sidecar's `latent_spec` and uses it only for the prior floor (`:389-396, 539, 557`), and its prior is `BoxUniform(0, 1)` (`:385-388`) `[CODE SBI@9fcbc88]`; the campaign bank's unit box, with D-074's `O_N` mixture, replaces it (the B5 plan, sec. 1)]** **[v1.11: the unit box alone -- D-078]** |
| B7 | freeze the trained encoder for the export path | `dsn_frozen.py` loads a standalone DSN checkpoint (`best.pt` with its config); the joint model's encoder is a `joint_model.py` object around the DSN backbone `[TO VERIFY: the joint checkpoint's layout and whether `dsn_frozen.py` can load it]` | either save the trained backbone in the standalone checkpoint format at the end of the A1 run, or teach `dsn_frozen.py` the joint checkpoint; the frozen copy + `.sha256` into `artifacts/frozen_dsn/` | `smoke_test_sbi_export.py` against the frozen file |
| B8 | real-well posteriors and the condition comparison | `gate_run.py`, `witness_run.py` exist; the amortised posterior draw on real windows is the joint stack's pseudo-real endpoint (Stage 3's report split; Stage 7 of the plan "stratification on the cohort") `[KB plan]` | a small driver: z for every real window through psi*, q(theta | z) draws per well, pooled per condition; the replicate statistic between same-condition wells (`replicate_statistic.py`); figures | fixture-level |

## 4. Names and places (D-046)

| object | name | note |
|---|---|---|
| real recordings | `/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/Bio_Data/` -- `Batch1/100N0A_wo/`, `Batch2/100N0A_wo/`, `Batch3/AraC/50N50A/`, `Batch3/AraC/70N30A/`, each holding `ptrain_<id>_DIV35_<cond>_nbasal_0001/` well folders | D-047; the user's upload, never written to; no space anywhere in the path (`qsub -v`, `HPC_PATHS.md` 3b) |
| extract root | `/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia/` | the config's `extract_root`; written by G1 (a first extraction into a declared root is allowed); holds `cohort_manifest.json` + `.sha256` **[v1.6: written 2026-10-02 15:31 -- the cohort of record, manifest sha256 `21b7ef03bb7f776b`; the two refused runs' roots beside it, `extracted_giulia_partial_20261002` and `extracted_giulia_partial2_20261002`, are the user's to delete]** **[v1.7: at the user's request a guarded one-line command was given (2026-10-02, about 16:30) that deletes both, and the two runs' logs, only after the cohort manifest verifies against its `.sha256`; once run, they are gone (addendum v1.6)]** |
| probe output | `~/repos/Sbi-extractor/extractor/out/probe_giulia.json` (+ the pasted `.log`) | gitignored (`out/`) |
| preflight log | `~/repos/Sbi-extractor/extractor/out/preflight_giulia.log` | v1.2; gitignored; last line `PREFLIGHT PASS (7/7)` |
| single-well test | `~/repos/Sbi-extractor/extractor/out/giulia_test_well/` (`traces.npz`, `subregion_map.png`, `subregion_ifrs.png`) | v1.2; gitignored; the hand run of one well (G1 run sheet, step 5), not part of the cohort |
| cohort config | `~/SBI/hpc/dsn/hpc/Config/config_giulia_cohort.davinci.json` | tracked; its sha256 goes into the manifest |
| frozen manifest copy | `~/repos/Sbi-extractor/artifacts/cohort_manifest_giulia/` | with its `.sha256`, as C8 does for DUP15HD **[v1.6: made by B3's launcher at its first `plan` run, from `extracted_giulia/cohort_manifest.json` and its sidecar -- not by hand (B3; sec. 9)]** **[v1.8: made by G2d, 2026-10-07: `21b7ef03bb7f776b`]** |
| flags file | `~/repos/Sbi-extractor/extractor/extraction_flags_giulia.sh` | **[corrected 2026-10-01 v1.2: shipped in the Sbi-extractor Giulia commit (bundle v2); v1.1 read "written by the launcher's first run, then COMMITTED", the v1 bundle's route]** the array sources the tracked file; regenerated from the config and compared on every run (the launcher; the preflight's step 5) |
| extraction manifest | `~/repos/Sbi-extractor/extractor/out/extraction_manifest_giulia.tsv` | gitignored; one line per listed well |
| logs | `~/repos/Sbi-extractor/extractor/out/chsub_mea_array_giulia_<i>.log`, `out/cohort_manifest_giulia.log` | the PASS line lives in the second |
| sim detections | `/davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out_giulia_v2/` | beside `mea_out/`, `mea_out_1electrode/`, which stay untouched (D-045) **[v1.8: written by G2 -- 69 task folders and `REEXTRACTION_RECORD.json` (sha256 `7af3759a0f3c1350`); `mea_out_1electrode/` is deleted after N1's list (D-066 (1)); `mea_out/` is the user's call]** **[v1.9: N1 listed `mea_out_1electrode/` against the record -- 51 task folders, 980,959 files, about 10.3 GB, all re-detected -- so D-066 (1) may be carried out by the user; `mea_out/` (4 entries) and `SBI_export_hhgap_1e_r2/` (102 entries) exist, the user's call]** |
| sim plan / tasks | `~/repos/Sbi-extractor/sim_reextract/giulia/plan.json`, `tasks.tsv` | gitignored like C8's **[v1.8: exist since G2d, 2026-10-07]** |
| label freeze | `~/repos/Sbi-extractor/artifacts/label_axes_giulia_hhgap.json` | p = 23; replaces `label_axes_hhgap.json` (2026-09-10, v1/v2/v5 only) for this project **[v1.9: G3 writes it with its `.sha256` (the G2 plan v2.7, sec. 4g); `artifacts/` is gitignored, so the file lives on davinci only, as September's does]** **[v1.10: written 2026-10-08 by G3, sha256 `29e38c89b08a2dd8`; every field but `created_utc` September's]** |
| banks | `/davinci-1/home/ldellamea/banks/giulia_sim/`, `.../banks/giulia_real/` | shared filesystem, never `/tmp` |
| joint runs | `/davinci-1/home/ldellamea/runs/giulia_arms/`, `/davinci-1/home/ldellamea/tune_joint_giulia/` | |
| frozen encoder | `~/repos/Sbi-extractor/artifacts/frozen_dsn/dsn_giulia_A1_<date>.pt` + `.sha256` | a copy, never a symlink |
| exports | `/davinci-1/home/ldellamea/ANN/SBI_export_giulia/` | both arms; gate stems `giulia_v1_*` |
| environments | `sbi_export` for G1-G4 and G6's export; `sbi_env` for G5 and the gate | D-023; `HPC_PATHS.md` sec. 7 |

## 5. The stages

Each stage: what goes in, what runs, what comes out, the line that must
appear before the next stage starts, who does what. Stages G1 and G2 are
independent; G3 needs G2; G4 needs G1 (real) and G2+G3 (sim); G5 needs G4's
sim bank; G6 needs G4's real bank and G5's encoder.

### G0 -- Exercise the joint stack once on davinci (before anything Giulia)

- **Why:** no Stage 3 / 4 job has run anywhere (usage sec. 9); the first
  execution of a job script must never be the real one (delivery skill).
- **Run** `[user, sbi_env]`: `probe_dsn_runtime.pbs` (must print `VERDICT:
  PASS (7/7 checks)`); a small bench bank by array with `PROVIDER=bench`
  (`build_latent_bank.pbs`); one `joint_arms.pbs` element, arm `A1`, with
  the defaults (`DRYRUN=1` first); `npe_tune_joint.py space --campaign S-A1
  --d-theta 10 --p 10` and one `propose`.
- **Must appear:** `arm : A1 (seed 0)` then `L`, `L0`, `delta_hat`, `r_eff`,
  `p_eff` and the written paths (usage sec. 5.2).
- **Output:** the environment problems found here, not in G5.

**[v1.13, 2026-10-10] The run sheet** is lines G1-G8 of the bundle `b5_g0_20261010` (its `README.md`;
the B5 plan v1.3 sec. 6), after D1 (`~/SBI` to `9fcbc88`) `[SANDBOX 2026-10-10: rehearsed with a qsub that
runs each job at once and a stand-in conda; every pass line printed]`. Two changes from the bullets above, from
the usage guide's findings: the bench bank is one shard per arm, not an array (F-at: the identity fields are
shard-local), built twice, S at `SEED=0` and R at `SEED=1` (F-aw), 1,024 traces each; and the element is given
`REAL_SHARDS` (F-ay: an empty `EXTRA` under `set -u` on bash 4.3). A first line (G1) probes a compute node for
what the job scripts assume: the bash version, `conda` in a batch shell, `sbi_env`'s activation and versions. [corrected 2026-10-12: G1 ran as an executable-mode job (`qsub -- /bin/bash -c`), not as the job scripts run; the status block below, and G1b]
Folders: `~/banks/giulia_g0_bench_{S,R}`, `~/runs/giulia_g0_arms`, `~/tune_joint_giulia_g0`, logs `~/g0_*.log`
(D-046's `giulia` kept, though the bench is not Giulia data).

**Status 2026-10-12 (v1.14)** `[GITHUB 2026-10-12; CLUSTER 2026-10-12, the user's paste of G1r; SANDBOX 2026-10-12]`.
LP is done: GitHub's `feat/real-arm-parity` is `9506d23` (committed 14:37:17 +0200) on `f885254` on `4e4dff0`, tree
`f681a9ece8fcbb367db1bd4d51192670211edfca`, the tested one. G1 printed `GNU bash, version 4.4.20(1)-release
(x86_64-redhat-linux-gnu)`, `type: conda: not found` and twice `conda: command not found` (the hook's command
substitution, then `conda activate`, where `set -e` ended the job). Read:

- the bash version closes F-ay on that node: from bash 4.4 an empty `"${EXTRA[@]}"` is not an unbound variable under
  `set -u` (bash's CHANGES file, bash-4.4-rc2, "New Features in Bash" item a `[WEB]`); one image for every compute node
  is assumed `[reasoning]`. `REAL_SHARDS` stays in G7 for the pseudo-real endpoint;
- the `conda` lines answer a question G1 should not have asked. [corrected 2026-10-12: the v1.13 paragraph's
  "`conda` in a batch shell" -- G1 was an executable-mode job (`qsub -- /bin/bash -c '...'`), while every job of G0
  and B5 is a job script (`qsub <file>`). On davinci the two do not start alike: job 1575002, a job script of
  2026-09-07, found `conda` and died inside base's `geotiff-deactivate.sh` (`HANDOFF_hpc_implementation_v1_1.md`
  S3.7, S4.1). G1 shows that an executable-mode job has no `conda`, not whether G5's and G7's scripts can activate.]
  SBI's `build_latent_bank.pbs` and `joint_arms.pbs` activate by the conda hook alone (`[CODE SBI@9fcbc88]` `:73-76`,
  `:85-88`); the bank's job scripts, like the extractor and C8 jobs that ran G1 and G2, fall back to `module load
  miniconda3` and then `/archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh` (`[CODE Sbi-extractor@9506d23]`
  `export_windows/run_bank_task.pbs:25-49`), so the run sheet's B lines do not depend on the answer; G3's probe calls
  `PYBIN` by absolute path;
- **G1b**, the corrected probe: SBI's `hpc/joint/stage3/jobs/probe_conda_activation.pbs` (blob `d6857079`, unchanged
  since `a40a489`, 2026-09-11) submitted as a job script -- the mode and the block of G5 and G7 (`set -euo pipefail`;
  `set +u` before the hook; `conda activate sbi_env`; `set -u`) -- by a line that submits only the committed probe,
  unchanged; read with `ls -l --time-style=long-iso` and `cat` of `~/probe_conda_activation.log`. Pass: `[probe]
  REACHED, flags=...` with `e` and `u`, the `sbi_env` python, `[probe] torch 2.10.0`, `[probe] VERDICT: PASS`; then
  G5. Its "at start" lines are information: `1` / `base` there means the login setup activated base, the state job
  1575002 died in. Two `conda: command not found` lines and no `REACHED`: job scripts lack `conda` as well, and SBI's
  two scripts take the bank's fallback block before G5 (an SBI fix). Rehearsed on the stand-in: with `conda` on the
  node's `PATH` it ends `VERDICT: PASS`; without, it stops at `conda activate` (exit 127) with no `REACHED`; a local
  edit, another probe at `HEAD` and no probe each print `STOP` and submit nothing `[SANDBOX 2026-10-12]`. G2, G3, G4,
  G8 and the run sheet's lines B1-B6 do not wait for it; G5 does, and G6 (its dry run reads G5's shards) and G7 after
  it.

**Status 2026-10-12 15:15 (v1.15), G1b PASSED** `[CLUSTER 2026-10-12, the user's paste of 15:17; SANDBOX 2026-10-12]`.
`~/probe_conda_activation.log`, dated 2026-10-12 15:15, from `dvnode004`: `CONDA_SHLVL` at start `1`,
`CONDA_DEFAULT_ENV` at start `base`; `[probe] REACHED, flags=ehuB`; `/davinci-1/home/ldellamea/.conda/envs/sbi_env/bin/python`;
`[probe] torch 2.10.0`; `[probe] VERDICT: PASS`. Read:

- a job script runs the login setup, which activates base: the job-script half of G1's question is answered; the
  executable-mode half stays G1's (no `conda`);
- the hook block survived deactivating base -- the state job 1575002 died in on 2026-09-07, before the guard -- and
  restored `-e` and `-u`: `build_latent_bank.pbs` and `joint_arms.pbs` (the same block, submitted the same way) run as
  shipped, G5 and G7 included; no SBI fix;
- because the login setup runs, a dotfile assignment reaches every job script (`HPC_PATHS.md` sec. 6b). Two read lines
  are revised, each rehearsed with its failure paths `[SANDBOX 2026-10-12]`: **G3r** prints `ls -l --time-style=long-iso`
  of `~/probe_dsn.log` before its `VERDICT` (the 2026-09-19 log is still there, so a read before the job ends would show
  the old one); **G5r** prints the two logs' last three lines and, from each bank's sidecar (`provenance`), `base_seed`
  and the builder's argv, then `GO: both banks carry the values G5 passed` only if `--arm`, `--provider bench`,
  `--n-traces 1024`, `--out-dir` and the seeds 0 and 1 are what G5 passed and the two base seeds differ (F-aw), and
  `STOP` otherwise -- a same-seed, a wrong-provider and a wrong-folder bank each printed `STOP`, a missing bank ends in
  `FileNotFoundError`. G6 and G7 follow G5r's `GO`.

**Status 2026-10-12 17:43 (v1.16), G3 and G5 PASSED** `[CLUSTER 2026-10-12, the user's paste of 17:43]`. G3r:
`~/probe_dsn.log` dated 2026-10-12 15:42 -- today's job, not the 2026-09-19 log -- and `[probe] VERDICT: PASS (7/7
checks)`. G5r: `rows written : 8192` in both logs; the S bank's sidecar `base_seed` 0 and the R bank's 1000003, each
with the argv G5 passed; last line `GO: both banks carry the values G5 passed`. So `~/banks/giulia_g0_bench_S/` and
`~/banks/giulia_g0_bench_R/` each hold `shard_0000.npz` + `.json` with distinct noise streams (F-aw avoided). Next,
in order: G6 (the arms' dry run on the two shards, login node; pass: `sim shards : 1, rows 8192, W 3000, d_theta 10`,
`real rows : 8192`, `DONE A1 seed 0`), G7 (one A1 element, a job), G7r (`tail -n 12 ~/g0_A1.log` once the job has
ended). The outputs of G2 (`dsn_locate`), G4 (the builder's dry run, superseded by G5's banks for G6's purpose) and
G8 (the tuner's `space` and `propose`) were not pasted `[NOT VERIFIED]`.

**Status 2026-10-12 18:59 (v1.17), G6 and G7 PASSED** `[CLUSTER 2026-10-12, the user's paste of 18:59]`. G6: `DRY RUN --
nothing trained`, `arm : A1 (seed 0)`, `sim shards : 1, rows 8192, W 3000, d_theta 10`, `real rows : 8192`, `embedding
size : 10`, `DONE A1 seed 0`. G7: job `1829506.pbsserver01` (queue `cpu`, 1 node, 4 cores, 16 GB, 06:00 requested) ran
and left `~/g0_A1.log`, whose last lines are the A1 report: `L pseudo-real : -4.3349 nats/row <- PRIMARY (D10)`, `delta_hat
(p-r) : -0.9020`, `L simulated : -4.2512`, `L0 (MC floor) : -5.2490`, `delta_hat (sim) : -0.9978`, `r_eff : 1.2123 of E
= 10`, `p_eff (spectrum) : 1.2992 [local-fallback; 1 of 10 eigenvalues exceeded 1 and were clipped]`, `T vs p_eff :
0.034 vs 5.659 (log ratio -14.226)`, `written : .../A1_seed0.json, _perrow.npz, _ckpt.pt`, `DONE A1 seed 0`. A plumbing
check on a 1,024-trace bench bank, not a result. Two things to read before G5 `[NOT VERIFIED why]`: the spectrum came
from the runner's local fallback, so the repo's `hpc/npe_diagnostics.information_spectrum` was not used in the job:
the runner looks for it under `--sbi-hpc-dir` or else `SBI_HPC_DIR` (`run_joint_arms.py:269, 565-571`,
`joint_diagnostics.load_repo_diagnostics` `[CODE SBI@9fcbc88]`), and `joint_arms.pbs` passes neither, so most likely
`SBI_HPC_DIR` was unset in the job `[reasoning]`; and the epoch lines print `T nan vs p_eff nan` (the last, `epoch 9 | npe -5.4377 | dsn
0.0000 | rep 0.0000 | T nan vs p_eff nan | ramp None | val -4.2034`). **[2026-10-13: explained, not a fault -- the `T` and `p_eff` of an epoch line are the replicate term's per-step statistics, collected only when `lambda_rep > 0` (`stage2/joint_train.py:178-199, 220` `[CODE SBI@9fcbc88]`); arm A1 has no replicate term, so they print `nan`; the same lines on a bench bank in the sandbox `[SANDBOX 2026-10-13]`]** The printed log ratio is the mean of per-replicate
log ratios (`log_ratio_mean`), not the log of the ratio of the two means, so it need not equal ln(0.034 / 5.659) `[CODE
SBI@9fcbc88 run_joint_arms.py:650-653]`. G0 is done but for G8 (and the outputs of G2 and G4, never pasted); no
environment problem was found after G1b's. **[2026-10-13 14:23 `[CLUSTER]`: G8 ran -- `every free axis of campaign S-A1 reaches the trainer.` and `[propose] campaign S-A1: 0 observation(s), 8 already pending, 0 proposed, 0 NEW`: the 8 pending specs are an earlier G8 run's (not pasted), so `propose` wrote none; the tuner's space and propose work on davinci. G0 is done; only G2's and G4's printouts were never pasted.]**

### G1 -- Real-arm extraction (ready: probe, then the two config fields, then the array) [done 2026-10-02 15:31]

**Status 2026-10-02 15:31, G1 DONE (v1.6)** `[CLUSTER 2026-10-02, the user's paste of 15:31; SANDBOX 2026-10-02]`.
The third run, C1-C6 as given, printed every line the v1.5 table below
expects -- checked line by line against the rehearsal, 12/12, the manifest's
preprocessing line byte for byte the rehearsal's:

| # | printed on davinci |
|---|---|
| C1 | the Sbi-extractor pull, `7 files changed, 440 insertions(+), 36 deletions(-)` (= `git diff --shortstat 481bd48 0a560ca`); `GO 1: both fixes are on davinci (2c0a06d / 0a560ca)` |
| C2 | `PREFLIGHT PASS (7/7)`; `PROBE OK; CONFIG MATCHES (16 wells to extract)` |
| C3 | the second run's root moved to `extracted_giulia_partial2_20261002`, its logs to `out/run2_20261002/`; the dry run's `NOTE: ... a first extraction.`, `# wells        : 16   -> array 0-15`, `(DRYRUN -- nothing was submitted. ...)` |
| C4 | `excluded (cohort.exclude_wells)` for 44456 and 45198; `wrote 16 well(s)`; array `1777495[]` (`qsub -J 0-15`), aggregation `1777496` `(runs once the array has ended, whatever its tasks' exit codes)` |
| C5 | `[job] host=dvnode005 started=2026-10-02T15:31:07+02:00`, `git=0a560ca`, `numpy 2.4.6`; `cohort manifest: 16 wells x 9 subregions = 144 units; preprocessing {...}; extractor run_channel_subset_extraction/4 @ 0a560ca`; `wrote /davinci-1/.../Giulia_Astro/extracted_giulia/cohort_manifest.json  sha256 21b7ef03bb7f776b`; `[job] cohort_manifest_exit=0` |
| C6 | `{"n_wells": 16, "n_units": 144, "classes": ["100N0A_wo", "50N50A", "70N30A"], "excluded_wells": [the two of D-055], "T_rec_range": [300.0, 600.0]}` |

The manifest's `preprocessing` is the config's eight fields as decided:
`fs_raw` 10000.0, `grid_width` 10, `index_base` 0 (D-048); `n_subsets` 9,
`electrodes_per_subset` 1 (D-043); `w_size` 0.01, `gaussian_window` 0.02,
`mfr_threshold` 0.1 (D-044). `21b7ef03bb7f776b` is the first 16 hex digits of
the manifest's sha256, as the PASS line prints it; the whole digest is in
`cohort_manifest.json.sha256` beside it. The cohort of record is
`/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia/`.
The two refused runs' roots (`extracted_giulia_partial_20261002`,
`extracted_giulia_partial2_20261002`) and their logs (`out/run1_20261002/`,
`out/run2_20261002/`) are history: nothing reads them, and keeping or deleting
them is the user's call.

G1's "Then" (below), read against the code this turn: the frozen manifest copy
is not a hand step -- the sim-arm launcher makes it from the manifest and its
sidecar, in C8's form into one fixed folder (B3; sec. 9); `make_mea_specs.py
--strict` imports torch (`config.py:48` -> `backbone.py:41`) and writes
`data.npz_specs`, `hpc/Config/npz_specs_giulia.json`, resolved against the
working directory (`os.path.abspath`) `[CODE SBI@2c0a06d,
make_mea_specs.py:74, 273-372]`; the specs are first needed by B5's real mode
(G4).

**Status 2026-10-02 15:09, the second run and the third run's commands (v1.5)** `[CLUSTER 2026-10-02, the user's paste of 15:09; GITHUB 2026-10-02, read 15:1x; SANDBOX 2026-10-02]`.
The G1-fix commits are on GitHub with the tested trees (SBI `2c0a06d`,
committed 15:03:57; Sbi-extractor `0a560ca`, 15:04:11), but davinci's pull ran
before the push landed: "Already up to date", and the aggregation's log says
`git=481bd48`. So the second run used the old code and the old config. The
probe said `CONFIG DIFFERS on: exclude_wells` -- the STOP line -- the first
run's root and logs were moved aside, the dry run listed 18 wells, the launch
followed (array `1777366[]`, aggregation `1777367`, with the old "held until
every array task exits 0" line), and the aggregation refused the same two
wells in the old wording; no manifest. Its output is a second partial root at
`extracted_giulia` (16 wells, two empty folders) with its logs in `out/`. Part
of the cause is the reply of 13:50: it gave the move, the dry run, the launch
and the `cat` in one code block, so pasted together they ran back to back, and
nothing in them checked the probe's line.

The third run (the reply that followed, about 15:30) is six one-line commands, each gated on the
one before, so that pasting all of them at once cannot launch the wrong cohort:

| # | does | prints when it may go on |
|---|---|---|
| C1 | `module load proxy`; pulls SBI and Sbi-extractor; `source env.sh`, `conda activate sbi_export`, `cd extractor`, sets `CFG` and `GIU`; checks the content of both fixes (`ptrain_44456` in the config, the new launcher line) | `GO 1: both fixes are on davinci (2c0a06d / 0a560ca)` |
| C2 | preflight and probe, each into its log; prints the last line of each | `PREFLIGHT PASS (7/7)`; `PROBE OK; CONFIG MATCHES (16 wells to extract)` |
| C3 | only if both C2 lines are exactly those and the target names are free: moves the second run's root to `extracted_giulia_partial2_20261002` and its logs to `out/run2_20261002/` (`mkdir` without `-p`, `mv -n`: nothing overwritten), then the dry run into `out/dryrun_giulia.log` | `NOTE: ... a first extraction.`, `# wells        : 16   -> array 0-15`, `(DRYRUN -- nothing was submitted. ...)`; otherwise `STOP: no dry run was made ...` |
| C4 | only if the dry run listed 16 wells: the launch | `agg   : <id>   (runs once the array has ended, whatever its tasks' exit codes)`; otherwise `STOP: not launched ...` |
| C5 | `cat out/cohort_manifest_giulia.log`, once `qstat -u $USER` no longer lists the two Giulia jobs | `git=0a560ca`; `cohort manifest: 16 wells x 9 subregions = 144 units; ...`; `wrote .../extracted_giulia/cohort_manifest.json  sha256 <16 hex>`; last `[job] cohort_manifest_exit=0` |
| C6 | the manifest's summary | `{"n_wells": 16, "n_units": 144, ..., "T_rec_range": [300.0, 600.0]}` |

Rehearsed at the real paths -- a `/davinci-1/home/ldellamea` tree in the
sandbox, clones at `745edce` / `481bd48` whose remotes hold the real heads,
the replica census, and the two refused runs reproduced first: C1-C6 print
exactly those lines, run 1's logs and root are untouched byte for byte, and
the negative paths hold -- C1 with nothing new to pull prints no GO line, C3
with the probe at `CONFIG DIFFERS` moves nothing, C4 after an 18-well dry run
submits nothing `[SANDBOX 2026-10-02, 16/16]`.

**Status 2026-10-02 13:12, the re-run is ready (v1.4)** `[CLUSTER 2026-10-02, the user's three outputs; SANDBOX 2026-10-02]`.
The logs confirm the reading below: the aggregation refused exactly the
probe's two `too_few_active` wells, tasks 3 and 4 of the array, each ending in
`InsufficientElectrodesError` -- `formed only 1 of 9 requested subregions`
for `ptrain_44456_DIV35_100N0A_nbasal_0001` and `formed only 5 of 9` for
`ptrain_45198_DIV35_100N0A_nbasal_0001`, both in `Batch1/100N0A_wo`. The
probe's `T_rec by well`: 300.0 s for each of the ten 0:100 wells, 600.0 s for
each of the eight AraC wells (O5). The user's call (D-055): exclude them by
name. The census per well is in `claude/HPC_PATHS_GIULIA_ADDENDUM_2026-10-01.md`
sec. A.

The re-run's run sheet is `g1fix_bundle_20261002/README.txt` (laptop A0-A4,
davinci B1-B12): pull both commits; preflight; probe -> GO is `PROBE OK;
CONFIG MATCHES (16 wells to extract)`; move the first run's root and logs
aside (`extracted_giulia` -> `extracted_giulia_partial_20261002`, logs ->
`out/run1_20261002/`; nothing deleted); DRYRUN (`# wells        : 16   ->
array 0-15`); launch; `cat out/cohort_manifest_giulia.log`. PASS: `cohort
manifest: 16 wells x 9 subregions = 144 units; ...`, `wrote
.../extracted_giulia/cohort_manifest.json  sha256 <16 hex>`, last line
`[job] cohort_manifest_exit=0`; the manifest's `excluded_wells` the two names,
`T_rec_range` [300.0, 600.0]. Every one of those lines was produced by the
patched code on a replica of the census (18 wells with the real names, file
counts, active counts and lengths, SpyCode format) through a fake `qsub`
`[SANDBOX 2026-10-02]`. Why all 16 wells are re-extracted rather than the
first run's 16 fragments aggregated: the manifest records as
`extractor_commit` the commit the aggregation runs at, and the builder only
checks that the fragments agree with one another; fragments written at
`481bd48` aggregated after the v8.10 pull would carry a commit that did not
write them (`EXTRACTOR_USAGE.md` sec. 15.8).

**Status 2026-10-02, the first run (v1.3)** `[CLUSTER 2026-10-02, the user's terminal log]`.
Preflight `PREFLIGHT PASS (7/7)`. Probe `PROBE OK; CONFIG DIFFERS on:
exclude_wells`: of the fields the probe checks (`ptrain_format`,
`ptrain_varname`, `ptrain_name_pattern`, `grid_width`, `index_base`, `fs_raw`,
coverage) only `exclude_wells` differed, so the declared format is confirmed
on all 18 wells (O1 settled; D-047's Status) and the probe found wells with
fewer than 9 x E electrodes at or above 0.1 Hz that the config does not
exclude -- the STOP of step 4. One well by hand (39489, 50N50A): `X.shape=(9,
60000)`, `present=35 discarded(<theta)=0`, `centres=[34, 61, 12, 45, 68, 54,
35, 26, 31]` -- K = floor(T_rec / w_size) = 60 000, so T_rec = 600 s, not
900 s (O5). The array was launched anyway: `1776556[]`, aggregation
`1776557`. Result: `REFUSED: 2 of 18 well(s) have no usable
traces_meta.json`, no `cohort_manifest.json`. The two wells are named on the
indented lines under that headline, which the step-7 grep does not print; the
reading of record is that they are the probe's too-few wells
(`InsufficientElectrodesError`), to be confirmed from the logs.

What exists: `/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia/<class_name>/<root_name>/<well>/` for
the 16 wells with a usable fragment (`traces.npz`, `trace_subregion_00..08.npz`,
`traces_meta.json`, `subregion_map.png`, `subregion_ifrs.png`); the two failed
wells' folders exist and are empty (the job creates the folder before the
extractor runs). Without the manifest nothing downstream may use them
(`make_mea_specs.py --strict`, the frozen copy, G2's `n_side` / `fs`, D-051).

To read (login node, `extractor/`): `sed -n '/REFUSED/,$p'
out/cohort_manifest_giulia.log` (the wells and why); `for f in $(grep -L '] done
->' out/chsub_mea_array_giulia_*.log); do echo "== $f"; grep -m1 'culture ='
"$f"; tail -n 2 "$f"; done` (each failed task's well and error; `-j oe`
puts the traceback in the log); `cat out/probe_giulia.log` (the census:
`active` per well, `T_rec by well`, the `DIFF exclude_wells` line).

Next **[done v1.4: O2 decided, D-055; the bundle is built; see the v1.4 status above]**: the user's call on O2 (exclude the two wells by name; a lower
`mfr_threshold` for the whole cohort; or fewer subregions per well), then one
SBI config commit (`exclude_wells` is not in the flags file, so
`extraction_flags_giulia.sh` is untouched unless the threshold or C changes),
pulled; move the partial root aside (`mv /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia
/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia_partial_20261002` `[proposed name]`) because the
launcher refuses its own declared root once non-empty; re-run steps 3-7.

**Run sheet (v1.2; the bundle's `README.txt`, B1-B11)** `[user, login node, sbi_export]`. **[v1.4: for the re-run, superseded by `g1fix_bundle_20261002/README.txt` B1-B12; kept as the record of the first run]**
The laptop part (A0-A4) commits and pushes SBI first, then Sbi-extractor;
C8 is already on GitHub, so its script is not run again.

1. `module load proxy`; `cd ~/SBI && git pull --ff-only`;
   `cd ~/repos/Sbi-extractor && git pull --ff-only`. The trees, when nothing
   else was pushed since: SBI `79a353e5d73873798e679d6ae8e24bba5891c36c`,
   Sbi-extractor `5ec712b034e9c2c90e9b88aa46b3059cd4cdb93d`
   (`git rev-parse HEAD^{tree}`).
2. `source env.sh && conda activate sbi_export && cd extractor &&
   CFG="$SBI_HPC_DIR/dsn/hpc/Config/config_giulia_cohort.davinci.json"`.
3. Preflight: `bash preflight_cohort.sh "$CFG" giulia 2>&1 | tee
   out/preflight_giulia.log | tail -1` -> `PREFLIGHT PASS (7/7)`.
4. Probe: `python3 probe_ptrain_tree.py --root /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/Bio_Data --config
   "$CFG" --out-json out/probe_giulia.json 2>&1 | tee out/probe_giulia.log |
   tail -1`. GO: `PROBE OK; CONFIG MATCHES (18 wells to extract)`. STOP:
   `PROBE OK; CONFIG DIFFERS on: ...` or `PROBE FAILED ...` -- paste the log;
   Claude re-cuts the config (SBI) and the flags file (Sbi-extractor)
   together; an exclusion is the user's call (sec. 7, O2).
5. One well by hand, the code the array runs: `source
   extraction_flags_giulia.sh && python3 run_channel_subset_extraction.py
   /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/Bio_Data/Batch3/AraC/50N50A/ptrain_39489_DIV35_50N50A_nbasal_0001
   --out-dir out/giulia_test_well --mode per_region_single $EXTRA_FLAGS` ->
   `X.shape=(9, K)` with K = 100 x T_rec (90 000 for 900 s) **[2026-10-02: 60 000 on 39489, i.e. 600 s]**, `wrote
   out/giulia_test_well/traces.npz`, and the two figures.
6. `CONFIG="$CFG" COHORT_TAG=giulia DRYRUN=1 bash launch_stage_d.sh
   /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia` -> `NOTE: ... a first extraction.`, `# wells        : 18   -> array 0-17`,
   the two `qsub` lines, `(DRYRUN -- nothing was submitted. ...)`; then the same line without `DRYRUN=1`.
7. The gate (D-006): `grep -h 'wrote\|REFUSED\|cohort_manifest_exit'
   out/cohort_manifest_giulia.log` -> `wrote /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia/cohort_manifest.json
   sha256 <16 hex>` and `cohort_manifest_exit=0`. **[corrected 2026-10-02: this
   grep prints only the `REFUSED:` headline, not the wells named on the
   indented lines below it; read the block with `sed -n '/REFUSED/,$p'
   out/cohort_manifest_giulia.log`. On a refusal there is no
   `cohort_manifest_exit` line: the job runs under `set -e` and stops at the
   failed check.]**

Paste back: the tail of 3, the whole probe log, the last two lines of 5, the
output of 7.

**[v1.1's steps below are superseded by the run sheet above (v1.2): the
flags file ships, so step 3's first-run stop is gone; the probe runs with
`--config` instead of the explicit flags, and its `SUGGESTED` block is
needed only on `CONFIG DIFFERS`; `cluster_check.sh` is replaced by
`preflight_cohort.sh`. What each step establishes is unchanged.]**

- **In:** the 18 wells under `Bio_Data/` (D-047); the bundle applied and
  pulled (SBI first, then Sbi-extractor); `cluster_check.sh` green.
- **Step 1, the probe** `[user, login node, sbi_export; seconds]`:
  `python3 probe_ptrain_tree.py --root .../Giulia_Astro/Bio_Data --fs-raw
  10000 --mfr-threshold 0.1 --n-subsets 9 --electrodes-per-subset 1
  --expect-t-rec 900 --out-json out/probe_giulia.json`. Must end with
  `PROBE OK` and a `SUGGESTED` block; paste the log. What it settles:
  `ptrain_format` and `ptrain_varname` (the config carries the
  `sparse_peaks` / `peak_train` hypothesis), `grid_width 10 / index_base 0`
  (expected `positional_fits: yes` on every well), `T_rec` per well (900 s
  expected on the eight AraC wells; variable on the ten `100N0A_wo`), and
  the wells with fewer than nine electrodes at or above 0.1 Hz -- the AraC
  wells have 17 to 35 files, so some may fail; those go into
  `exclude_wells` (a decision: Open calls (b)), because a well the extractor
  refuses makes the manifest refuse the whole cohort (D-002).
- **Step 2, the config** `[user; Claude prepares the patch from the log]`:
  the two fields and `exclude_wells` set from the probe; committed.
- **Step 3, the dry runs** `[user, sbi_export]`: `CONFIG=$SBI_HPC_DIR/dsn/hpc/Config/config_giulia_cohort.davinci.json
  COHORT_TAG=giulia DRYRUN=1 bash launch_stage_d.sh
  /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia`.
  The first run stops with `NEW COHORT: wrote extraction_flags_giulia.sh`
  (exit 3): commit that file and re-run the same line; the second prints
  `qsub -J 0-<n-1> -N chsub_mea_array_giulia ... -v
  ENV_NAME=sbi_export,CHSUB_MANIFEST=out/extraction_manifest_giulia.tsv,CHSUB_FLAGS=extraction_flags_giulia.sh`
  and the aggregation line; n = 18 minus the exclusions.
- **Step 4, the array** `[user]`: the same line without `DRYRUN=1`; one
  element per well (`run_extractor_array_mea.pbs`), then the aggregation
  (`run_cohort_manifest.pbs`), which is the gate (D-006).
- **Per well:** V = electrodes with MFR >= 0.1 Hz over the whole recording;
  the nine hottest become `trace_subregion_00..08.npz` (one electrode each,
  D-043); the (row, col) decode of the MCS code enters only `validate_grid`
  and the map figure.
- **Must appear:** in `out/cohort_manifest_giulia.log`, `wrote
  .../extracted_giulia/cohort_manifest.json  sha256 ...` and
  `cohort_manifest_exit=0`; in the manifest, `preprocessing` equal to the
  config's eight fields, `source_format` = the three declared values,
  `excluded_wells` = the probe's list, `n_units = 9 x n_wells_kept`,
  `classes = ["100N0A_wo", "50N50A", "70N30A"]`, `T_rec_range` with 900 s
  on the AraC wells. **[corrected 2026-10-02: `T_rec_range` [300.0, 600.0]
  -- the 0:100 wells are 300 s, the AraC wells 600 s; `n_units` = 9 x 16 =
  144]**
- **Then** `[user, sbi_env]`: `make_mea_specs.py --config
  config_giulia_cohort.davinci.json --strict` -> the Giulia specs (9 records
  per well, the exclusions reported); the frozen manifest copy into
  `artifacts/cohort_manifest_giulia/` (+ `.sha256`). **[v1.6: the frozen copy
  is made by B3's launcher, not by hand; the specs are first needed at G4 --
  the v1.6 status block above]**
- **Reports to paste:** the probe's log, `out/cohort_manifest_giulia.log`,
  the manifest. From them: the kept wells per condition and the windows per
  trace (5 per 15-min well; per well for `100N0A_wo`). **[corrected
  2026-10-02: 8 / 4 / 4 wells; at stride 180 s, 3 windows per AraC trace
  and 1 per 0:100 trace (O6)]**

### G2 -- Simulated-arm re-detection (after G1's manifest, D-051; needs the C8 push and B3) [C8 pushed 2026-10-01 17:38] [v1.7: and after C8's record, D-057] [done 2026-10-07 17:57]

**Status 2026-10-08, G2 DONE (v1.8)** `[CLUSTER 2026-10-08, the user's paste of 14:21: G2h]`. G2 ran from its own work plan, `claude/PLAN_2026-10-06_giulia_G2.md` (B3 at Sbi-extractor `4e4dff0`; the ANN tools of D-071 and D-072 at `d0ced88`, subtree `028d786a`): the plan with `COHORT_TAG=giulia` and `--allow-mixed-contract`, one test task, the array (68 tasks, `-J 0-67%20`, WALLTIME 02:00:00) and the gate. The record: `/davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out_giulia_v2/REEXTRACTION_RECORD.json` (+ `.sha256`), sha256 `7af3759a0f3c1350` -- 69 tasks (v1 25, v2 24, v5 20), 1,260,207 iterations, 27 replays excluded (v4's 7, v5's 20 `sweep_intel`), 0 warnings, 12 hosts, one environment (`sbi_export`), `--n_side 1 --pitch 200.0 --edge 26.59 --fs 10000.0`, noise `sim`, the frozen manifest `21b7ef03bb7f776b`. The "Expect" below was September's accounting (51 tasks, about 981 k simulations): G2 keeps 69 because the dedup works within a label contract (D-070), and the 18 v2 `sweep_intel` tasks whose seeds v1's share across the contract split -- different simulations by D5, 279,248 iterations -- are kept (the G2 plan, sec. 4f). The "Must appear" lines were read on the test task's first file (the G2 plan, sec. 4d); the record's geometry is the plan's.

- **In:** `Giulia_Astro/campaign_cadex_hhgap_v{1,2,4,5}`; the ANN tools at
  the C8 state in `ANN/MEA_analysis/` (the C8 upload of
  `claude/HANDOFF_2026-10-01_c8_push_and_run.md` P5 must have happened: the
  job finds davinci's conda and writes `mea_env.json`); B3.
- **Plan step** `[user, login node, sbi_export]`: `DRYRUN=1 bash
  launch_sim_reextract.sh plan` with the Giulia parameters. The plan reads
  every task's `job_args.json` and `manifest.json`, names every exclusion
  (no `manifest.json`, empty folders), dedupes on `_resolved_seed_master`
  (sec. 6), checks one contract per field and names v1's `O_N` split
  (`[0.03, 3.0]` vs `[0.01, 3.0]`, both `ln`) as the one allowed
  difference, fingerprints the tools and the template library, prints the
  per-task sizes and the largest task. `EXTRA_ARGS` are fixed: `--n_side 1
  --pitch 200 --edge 26.59 --fs 10000` (D-048, D-049), `n_side` and `fs`
  read from `cohort_manifest_giulia/cohort_manifest.json` (D-051).
- **Expect** `[KB sec. 12.3, reasoning]`: 51 kept tasks from v1 (7 `cfd` +
  18 `intel`), v2 (6), v5 (20) and whatever v4 adds after dedup (its 7
  overlapping seeds with v5 are replays; the rest of v4, if any, is new);
  about 981 k simulations if the counts of `mea_out_1electrode` carry over
  (980,959 `[KB 4f]`), fewer by the dropped replays. The `q/` units are
  outside the glob and are listed as such, not used (D-050).
- **Test step:** `bash launch_sim_reextract.sh test` -- one task as a plain
  job; its log must show `env activated: .../envs/sbi_export/bin/python3`
  and `[mea] N topos, a/b iters processed in X s`. The August runs at
  `n_side 1|2` took up to the 6 h walltime on 7 of 110 jobs
  (`HPC_PATHS.md` 7.2): size `WALLTIME` from X s / b x the largest task
  x 2, as the C8 handoff does.
- **Array + gate:** `WALLTIME=hh:mm:ss RESUME=1 bash launch_sim_reextract.sh
  array`; the gate job writes `mea_out_giulia_v2/REEXTRACTION_RECORD.json`
  + `.sha256` or REFUSES naming every bad task. The gate's log is the success
  signal (D-006), not the array's exit codes.
- **Must appear:** `n_side 1`, `pitch 200.0`, `edge 26.59`, `fs 10000.0`,
  `polarity neg` in every `mea_iter_*.npz`'s `meta_json`; one environment
  across the run; the kept / dropped task lists in the record.

### G3 -- Label freeze over the kept tasks [v1.9: report-only pass 2026-10-08; the write ready] [done 2026-10-08, pasted 16:16]

**Status 2026-10-08 16:16, G3 DONE (v1.10)** `[CLUSTER 2026-10-08, the user's paste of 16:16: G3]`. The write (the G2 plan v2.8, sec. 4h) printed the scan of the report line for line, `wrote artifacts/label_axes_giulia_hhgap.json` and `sha256 29e38c89b08a2dd8`, the file matching its `.sha256`, `topology_axes ['conn_prob']`, the three kernel axes excluded, `conn_rule_observed ['flat']`, `topologies_sampled 4467`, `conn_prob 2612 distinct in [0.050360135633, 0.399682467427]`, and against September's `artifacts/label_axes_hhgap.json` (sha256 `1ef04131175605a0`, the digest of record, on davinci): `the same in every field but created_utc (8 fields compared)`; last `G3 DONE`, no NOTE. This is the Giulia freeze of record (D-046); it lives on davinci only (`artifacts/` is gitignored).

**Status 2026-10-08, the report-only pass (v1.9)** `[CLUSTER 2026-10-08, the user's paste of 15:20: N1]`. N1 ran the corrected call below without `--out`, into `sim_reextract/out/g3_report.txt`: `topologies sampled : 4467`; `conn_rule observed : ['flat']`; `conn_prob` 2612 distinct, in theta, [0.0503601, 0.399682]; `p0_conn`, `d0_conn`, `beta_conn` EXCLUDED; a NOTE naming `seed_run` and `topo_idx` as other numeric scalars of the iteration files (bookkeeping, not swept axes `[reasoning]`); `theta topology block: ['conn_prob']  (p_topology = 1)`; 0 unreadable files. These are the September freeze of record's figures (`EXTRACTOR_USAGE.md` sec. 5.5). The write is the line `G3` of the G2 plan v2.7, sec. 4g: the same call with `--out`, run only if the file does not exist yet and the report is the one checked, then compared with the report and with September's `artifacts/label_axes_hhgap.json` when that file is on davinci. "Over the kept tasks" in the heading does not hold as written: the tool scans every `topo_*` folder under the glob, and the raw tree still holds v5's 20 replay tasks. The axis decision is the same either way, since a replay repeats a kept task's values; the count fields are not: `topologies_sampled` = `n_independent_topology_draws` = 4,467 counts the replays' topologies too (the record holds 3,791), and the record's topologies hold only 2,612 distinct `conn_prob` values, so neither count is a sample size for G4 (the G2 plan v2.7, sec. 4g) `[reasoning]`.

- **Run** `[user, sbi_export]`: `preflight_label_axes.py --sim_main
  .../Giulia_Astro --campaigns <the kept tasks' campaigns> --exclude p0_conn
  d0_conn beta_conn` -> `artifacts/label_axes_giulia_hhgap.json`. **[corrected 2026-10-08: this call does not match the tool's interface (`preflight_label_axes.py:140-150` `[CODE Sbi-extractor@4e4dff0]`): `--campaigns` takes one glob, `--exclude` takes `NAME=REASON` once per axis, and nothing is written without `--out`. The call, as the freeze of record's (`EXTRACTOR_USAGE.md` sec. 5.5), from the Sbi-extractor root: `python3 preflight_label_axes.py --sim_main /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro --campaigns 'campaign_cadex_hhgap_v[125]' --require-conn-rule flat --exclude 'p0_conn=<reason>' --exclude 'd0_conn=<reason>' --exclude 'beta_conn=<reason>' --out artifacts/label_axes_giulia_hhgap.json`, with the reasons of sec. 5.5. v4 is outside the glob (its seven tasks are all replays). The report-only pass, without `--out`, runs first, in N1 (the G2 plan, sec. 4f)]**
- **Must appear:** `conn_rule_observed flat`, `topology_axes = [conn_prob]`,
  p = 23; the registry sourced per campaign (`registry_from_manifest`), the
  `Sigma` box `[1, 10]` and the `O_N` box read from each task's manifest. **[v1.8: the freeze decides the topology block only -- it prints `conn_rule observed : ['flat']`, the per-axis table and `theta topology block: ['conn_prob']  (p_topology = 1)`, and writes `topology_axes`, `excluded_axes`, `conn_rule_observed`, `topologies_sampled` (`EXTRACTOR_USAGE.md` sec. 5.5); p = 23 is the export's, the 22 swept axes from each campaign's manifest plus that block. `topologies_sampled` counts the replays' topologies too, since the raw tree still holds them]**
- **Trap:** without `--exclude` the three all-NaN kernel axes count as swept
  (trap 6.5, not fixed).

### G4 -- The two Stage 1 banks (needs B5) [v1.10: designed -- D-074..D-076]

**v1.10 (2026-10-08):** the sim bank takes all 69 record tasks with `O_N` declared in [0.01, 3.0] (D-074), keeps a window only if its mean rate is at least 0.1 Hz (D-075), and groups its split by network (D-076); its work plan is `claude/PLAN_2026-10-08_giulia_B5.md`, which supersedes the sizes and levers below where they differ.

- **Sim bank** `[user, sbi_export, PBS array]`: `export_windows.py --mode
  campaign` over `mea_out_giulia_v2` + `Giulia_Astro`, `LABEL_AXES` the G3
  file, the manifest copy for `dt`, `sigma_sm`, `n_e` (asserted = 1 from
  `electrode_centers`), one shard per (campaign, task) into
  `~/banks/giulia_sim/`. Size: 981 k rows x 18000 float32 = **70.6 GB**
  before any filter (`npz` compression helps; the figure is the in-memory
  size a training loader faces) `[reasoning]`. Two levers, both
  configuration (D-039 "afterwards"): an activity floor on the sim rows (the
  DUP15HD gate kept 34.3 % of rows at 0.1 Hz/electrode, `HPC_PATHS.md` 4d;
  on the 1-electrode hhgap root the sampled rate was 3.25 Hz per electrode
  and 32 of 51 units reported simulations going quiet early,
  `EXTRACTOR_USAGE.md` 12.2), and a cap on rows per task. Check the quota on
  davinci before the array (`[TO VERIFY]`; `df -h $HOME`). **[corrected 2026-10-08: the size above is September's count. G2's record holds 1,260,207 rows: 1,260,207 x 18,000 x 4 B = 90.7 GB (84.5 GiB) before any filter, of which the 279,248 rows of the 18 cross-contract v2 tasks are 20.1 GB; how the bank treats those pairs, and the two `O_N` prior boxes it would mix, are design questions beside the two levers (the G2 plan v2.6, sec. 4f; the log's Open calls) `[reasoning]`]**
- **Real bank** `[user, sbi_export]`: `export_windows.py --mode real
  --specs <the Giulia specs>`: 5 windows x 9 subregions per 15-min well,
  `theta` NaN, `theta_present: false`, `cls` the condition, `donor` the
  culture, into `~/banks/giulia_real/`. **[corrected 2026-10-08: by D-056 and the measured lengths, 1 window per 0:100 trace and 3 per AraC trace: 72 + 108 + 108 = 288 windows over the 16 wells' 144 units]**
- **Must appear** (per shard, from `latent_bank.shape_report`): `W 18000`,
  `theta (n, 23)`, `scale_convention per_electrode_mean`, the digest; the
  same digest across every sim shard (`concat_shards` refuses otherwise).
  **[v1.13, 2026-10-10: as built, the pass lines are the plan's `BANK PLAN OK`, each task's `TASK <i> OK`,
  the gate's `wrote .../giulia_sim/BANK_RECORD.json  sha256 <16 hex>` (the gate checks the one digest, the
  files and every shard's id arrays) and the real mode's `REAL BANK OK`; lines B1-B6 of the bundle
  `b5_g0_20261010`]** **[2026-10-12 17:43: B1 refused `campaign_cadex_hhgap_v1/sweep_cfd_task0000` -- "69 topologies and 66240
  detection files ..., the record says 70 and 66240" `[CLUSTER]`: G2's record counts every `topo_*` folder of the
  raw task, empty ones included, the plan counted only folders holding a detection file. Fixed: an empty folder in
  both trees is a topology without simulations, counted, named, no row and no donor (`specs/SPEC.md` sec. 2.5
  E(t)); bundle `b51fix_20261012` (LP2, D2b, then B1 again); the B5 plan v1.6, sec. 6]** **[2026-10-12 18:59: B1 passed -- `bank_plan.json` sha256 `c8303506131a4632`: 2,612 donors, 155 shards, 8
  topologies without simulations, kept share 0.5002 (95% CI 0.4793-0.5215), about 630,404 rows and 5.2 GB after the
  floor, no disk quota; the B5 plan v1.7, sec. 6]** **[2026-10-13: B2 printed the test task's qsub line (`-q cpu`, 1 core, 8 GB,
  02:00:00, `TASK_INDEX=0`) and `(DRYRUN -- nothing was submitted.)` `[CLUSTER]`. B3-B5 are run as ONE chained line, BC
  (the pipeline plan sec. 5; D-081): the test task, the array with `WALLTIME=auto` and its gate, the real bank, each
  waited for and checked, `BANK CHAIN DONE` last; `tail -n 3 export_windows/giulia/logs/chain.log` is its status]**

### G5 -- Joint training, arm A1, then the Stage 4 search (needs G4 sim; configuration to be discussed)

- **Run** `[user, sbi_env]`: `DRYRUN=1 SIM_SHARDS='~/banks/giulia_sim/*.npz'
  REAL_SHARDS='~/banks/giulia_real/*.npz' OUT_DIR=~/runs/giulia_arms
  PBS_ARRAY_INDEX=0 bash jobs/joint_arms.pbs`, then the real element for
  `A1` (index 0 = `A1`, seed 0; `ARMS=(A1 A0 A0s A2 A2s A3 A5 A_ref
  shuffled)`, usage sec. 5.2), then seeds 1..4 (indices 9, 18, 27, 36) and,
  `[proposed]`, `A_ref` (7) and `shuffled` (8) as the reference and the
  negative control. `REAL_SHARDS` feeds only the report split and the
  pseudo-real endpoint here: with `A1` no real window enters the loss.
- **Then** the campaign: `npe_tune_joint.py space --campaign S-A1
  --results-dir ~/tune_joint_giulia --p 23 --d-theta 23 --embedding-dim
  <E>`; `propose`; `bash jobs/launch_joint_tune.sh S-A1 ~/tune_joint_giulia`
  (dry run), then `--submit --max <k>`; `status`, `finalists`.
- **Configuration to settle before the first real element** (D-039,
  "afterwards"): epochs and steps per epoch against ~10^5-10^6 rows (the
  job's `EPOCHS 10 x STEPS_PER_EPOCH 25 x B_SIM 128` sees 32 k rows **[2026-10-13: that is the Stage 3 job; a search trial's batch is its searched `batch_size_npe`, 256-1024, so 10 x 25 steps draw 64-256 k rows -- the pipeline plan v1.1, sec. 4a]**), the
  embedding dimension, `N_DRAWS` (floor 92 at d_theta = 23; 128 default),
  the grouped split (0.7 / 0.15 / 0.15 by default), the activity floor of
  G4, the job's cpus / memory / walltime (4 / 16 GB / 6 h as committed are
  bench sizes), and the D-038 fix patches (`num_bins`, `head_pool_ops`) if
  they land first.
- **Must appear:** the arm line and metrics of usage sec. 5.2; `p_eff
  (spectrum)` a number, not `n/a` (widen the split otherwise).

### G6 -- Freeze, export, gate, posteriors, conditions (needs G5 and G4 real)

- Freeze the `A1` encoder (B7) into `artifacts/frozen_dsn/dsn_giulia_A1_<date>.pt`
  + `.sha256`.
- `[user, sbi_export]` `launch_sweep_exports.sh` over `mea_out_giulia_v2`
  with `LABEL_AXES=artifacts/label_axes_giulia_hhgap.json`, `CKPT` the
  frozen file, `OUT_ROOT ANN/SBI_export_giulia`; `real_source.py --specs
  <the Giulia specs>` for the real arm. Every sidecar: one checkpoint
  digest, `n_electrodes 1`, p = 23.
- `[user, sbi_env]` `gate_run.py` to a new stem `giulia_v1`, `--no_witness`,
  then `witness_run.py`; the gate's verdict is encoder-conditional
  (`HPC_PATHS.md` 5b).
- The posteriors: z for every real window through psi*, q(theta | z) draws
  per well, pooled per condition (B8); the replicate statistic between wells
  of one condition (`replicate_statistic.py`; S_mc >= 92).
- **Must appear:** the gate's per-space verdicts and the MDE; per condition,
  the marginal posteriors of the astrocyte axes -- nine of the 22 by the
  2026-09-10 audit (`EXTRACTOR_USAGE.md` 12.3); which nine is read from
  `Giulia_Astro/campaign_axis_audit/campaign_axis_audit.json`, not inferred
  from the names `[TO VERIFY]`.

## 6. The seed dedup rule for G2 (Claude's, for the plan step)

**[v1.8: not taken. D-070 (2026-10-06) chose the C8 replay gate as built (D-063, D-064): replays dropped within a label contract, the longest copy kept, `theta` and `params` compared at up to 10 shared files; a same-seed pair with different theta is kept and reported, not refused. G2's plan found 18 such pairs across the two contracts (D5: different simulations) and kept them. The rule below is kept for the record.]**

Read `_resolved_seed_master` from every task's `job_args.json` across v1,
v2, v4, v5. Group tasks by seed. For a group of size > 1: open the first
`iter_*.npz` of the first topology of each task and compare `theta`; if all
equal, keep one task -- the one already among the September 51 if any
(`mea_out_1electrode`'s task list), else the first in (campaign, task-name)
order -- and list the others as `dropped_replay` with their seed; if they
differ, REFUSE naming the group (same seed, different draws means the seed
is not what identifies the simulation, and the rule must be re-thought
before a bank is built on it). Both lists go into `plan.json` and the
record. `[KB 12.3]`: the overlaps live in the `intel_task` swaths and v4;
the September 51 had 51 distinct seeds.

## 7. P1-P10, answered (about 16:10; log D-047..D-051), and what is still open

| id | answer | where it landed |
|---|---|---|
| P1 | `Giulia_Astro/Bio_Data/`, four condition folders, 18 wells (10 / 4 / 4), all at DIV 35 -- the listing attached | D-047; sec. 4; the config's `class_roots` |
| P2 | `100N0A_wo` is the 0:100, not treated with AraC | D-047; `class_names` |
| P3 / P4 | `fs_raw` exactly 10 kHz; grid positions start from 1; the electrode count is unknown and immaterial with one electrode per subregion. The listing shows the MCS row/column code, decoded by `grid_width 10 / index_base 0` | D-048; the config |
| P5 | `--edge 26.59` is OK | D-049; G2's `EXTRA_ARGS` |
| P6 | the deterministic rule (the nine most active) is OK "for now" | D-043's Status |
| P7 | "the default" = 0.01 s / 0.02 s / 0.1 Hz | D-044's Status; the config |
| P8 | the five `q/` units: unknown, "do not use them" | D-050; B3's plan step |
| P9 | the data are uploaded: no shortcut, the stages run in order | D-051; G2 reads the manifest |
| P10 | "Joint configuration trained on the simulations only" | D-039 restated; G5 unchanged |

Still open (Open calls of the log):

| id | question | proposal / default |
|---|---|---|
| O1 | **[settled 2026-10-02 by the probe: sparse peak trains under `peak_train`, as configured]** the files' format: variable name, storage class, length -- the probe reports it | the config carries `sparse_peaks` / `peak_train` as the hypothesis; set from the probe's `SUGGESTED` block, never from the hypothesis |
| O2 | **[settled 2026-10-02 13:12 by D-055: the two wells excluded by name]** wells with fewer than nine electrodes at or above 0.1 Hz: exclude them (and record it), or lower the bar? | **exclude**, by name, in `exclude_wells`, from the probe's census; the manifest records `excluded_wells` **[live 2026-10-02: two wells, names and counts in the logs (G1 status); the alternatives are a lower `mfr_threshold` for the whole cohort -- wells that already have nine electrodes at or above 0.1 Hz keep the same nine, since the rule takes the nine of highest MFR -- or fewer subregions per well (C < 9) for every well; excluding the least active wells selects on activity, which is what the conditions are compared on `[reasoning]`]** |
| O3 | the joint configuration (G5) and the sim-bank levers (G4) | discussed when G4's sizes are in hand; nothing submitted before **[v1.8: G4's sizes are in hand -- 1,260,207 sim rows, 90.7 GB before any filter; the questions are listed in the G2 plan v2.6, sec. 4f, and the log's Open calls]** |
| O4 | whether four AraC wells per condition (fewer after exclusions) suffice for the per-condition statistics of G6 | to discuss with O3; the replicate statistic needs same-condition pairs |
| O5 | **[settled 2026-10-02 by the probe: 600 s for every AraC well, 300 s for every 0:100 well]** how long the AraC recordings are: 15 min stated (D-040), 600 s measured on 39489 | the probe's `T_rec by well`; at 600 s, 3 windows of 180 s per trace (D-041), not 5 |
| O6 | **[settled 2026-10-02 15:09 by D-056: 180 s, back to back, from t = 0]** **[v1.4]** the real arm's window stride (D-041 leaves it open; 180 s is Claude's default): a 0:100 trace (300 s) holds one 180 s window, an AraC trace (600 s) three, so class 0 gives 8 x 9 = 72 windows and each AraC class 4 x 9 x 3 = 108 | decide with O3 at G4; a smaller stride gives more, overlapping and correlated, windows |
| O7 | **[v1.4]** class 0's two batches differ in activity: the three kept Batch1 wells have 9, 11, 9 electrodes at or above 0.1 Hz, the five Batch2 wells 32 to 59 (the probe's census); with E = 1 a Batch1 well's nine electrodes are nearly all of its V | decide with O4 at G6: whether, and how, the per-condition statistics account for batch |
| O8 | **[settled 2026-10-02 16:01 by D-057: G2 waits for C8's record; C8 runs now]** **[v1.6]** whether G2 -- B3 and the re-detection of the hhgap campaigns -- waits for the DUP15HD C8 run and its record: D-053's "Does not decide", live now that G1 is done. Both runs need the ANN tools upload of the C8 handoff's P5; G2 reads none of C8's outputs `[reasoning]` | Claude's proposal: build B3 now, and launch G2 once it is pushed and the tools upload is checked, without waiting for C8's record; the C8 run itself can go at any time (log Open calls; asked 2026-10-02) |

## 8. Order, and what Claude builds next

```
now (user)   apply the bundle: SBI commit, Sbi-extractor commit, pull both on davinci, cluster_check.sh
             G1 step 1: the probe  -> paste out/probe_giulia.log
Claude       G1 step 2: the config patch (two fields + exclude_wells) from the log
user         G1 steps 3-4: dry runs (commit extraction_flags_giulia.sh between them), the array, the manifest
G0 (user, any time, sbi_env)                 |  after the C8 push (D-034): Claude adapts B3 -> G2 plan -> test -> array -> record
G3 after G2's record
B5 window exporter (needs G2's record format and G1's manifest to test against real files)
G4 sim bank, G4 real bank
G5 (configuration discussion first) -> G6
```

**[v1.2, 2026-10-01 18:15]** Where this stands: the C8 push is done
(17:38); the bundle is v2 and the user is applying it (D-053) -- SBI commit,
Sbi-extractor commit, pull, preflight, probe. On `CONFIG MATCHES` the
single well and the array follow with no patch in between; on `CONFIG
DIFFERS` Claude's config + flags patch comes first. B3 can start now that
`sim_reextract/` is on GitHub; the `EXTRACTOR_USAGE.md` docs commit (v8.9,
sec. 15) is cut once the Giulia commit is visible on GitHub.

**[v1.3, 2026-10-02 10:45]** Where this stands: both Giulia commits are on
GitHub (SBI `745edce`, Sbi-extractor `481bd48`, the tested trees); G1 ran once
and was refused at the gate for two wells (sec. 5, G1 status). Waiting on the
three logs and the O2 call; then one SBI config commit, the partial root moved
aside, steps 3-7 again. The `EXTRACTOR_USAGE.md` docs commit (v8.9, sec. 15)
and three small Sbi-extractor fixes (the launcher's false "held until every
array task exits 0" line, the step-7 grep in the launcher's footer, the
missing exit line on a refusal) go in the same push, so the documentation
describes the run as it happened.

**[v1.4, 2026-10-02 13:50]** Where this stands: D-055 decided; the G1-fix
bundle is built and rehearsed on the real heads and is with the user
(laptop: SBI then Sbi-extractor; davinci: pull, preflight, probe -> `CONFIG
MATCHES (16 wells to extract)`, move aside, DRYRUN, launch, `cat` the log).
After its PASS: `make_mea_specs.py --strict` and the frozen manifest copy
(G1's "Then"), then G2 needs B3 (C8 is on GitHub). O6 and O7 join O3/O4 for
the configuration discussion.

**[v1.5, 2026-10-02 15:30]** Where this stands: the G1-fix commits are on
GitHub (`2c0a06d`, `0a560ca`); the second run (15:07) predates their pull and
was refused like the first. The third run is C1-C6 (sec. 5, G1). D-056
settles O6; O7 and O3 / O4 stay for the configuration discussion.

**[v1.6, 2026-10-02 15:42]** Where this stands: G1 is done -- the Giulia
cohort manifest exists (sec. 5, G1 status). Next, in order of dependence:
B3 (Claude: C8's plan / launch / gate re-parameterised for the hhgap
campaigns, with the seed dedup of sec. 6 and a freeze of its own, sec. 9)
-> the ANN tools upload of the C8 handoff's P5, if not done -> G2 plan,
test task, array, record -> G3. G0 can run at any time. O8 (whether G2 waits
for C8's record) is asked; O3 / O4 / O7 wait for the configuration
discussion, before G4.

**[v1.7, 2026-10-02 16:25]** Where this stands: D-057 -- C8 runs now and G2
waits for its record. The C8 run sheet is the C8 handoff's P10 (S1 status;
S2 the ANN upload as one archive, if S1 asks for it; S3 the two suites; S4
the plan; S5 the test task; S6 the test log), then the array and the gate.
When B3 is built is open: Claude offered to build it while C8 runs, so that
G2 can start when the record exists. O3 / O4 / O7 as before.

**[v1.8, 2026-10-08 15:30]** Where this stands: Stage C is closed (C8's record, 2026-10-07 13:11) and G2 is done (the Giulia record, 2026-10-07 17:57; sec. 5). Next: N1 -- read-only: the record, D5's per-pair lines, `mea_out_1electrode` against the record -- with G3's report-only pass (the G2 plan v2.6, sec. 4f); then G3's write, D-066 (1) (the user deletes `mea_out_1electrode` once N1 has listed it), and the B5/G4 design questions before B5 is built: the 18 cross-contract pairs, the bank's 90.7 GB, the two `O_N` boxes, an activity floor, a cap per task, the split groups. G0 can run at any time; G5's configuration is still "afterwards" (D-039).

**[v1.9, 2026-10-08 15:45]** Where this stands: N1 is in (2026-10-08 15:20): the Giulia record checks against its `.sha256`; the 18 cross-contract pairs are two simulations of one parameter draw, `O_N` alone rescaled (networks and noise not verified); `mea_out_1electrode` is covered task folder by task folder; G3's report-only pass matches September's freeze of record. Next: G3's write (the G2 plan v2.7, sec. 4g); the user's word on deleting `mea_out_1electrode` (D-066 (1)) and on `mea_out/` and `SBI_export_hhgap_1e_r2/`; then the B5/G4 design questions, before B5 is built -- the pairs and the split groups, between 2,612 and 3,791 independent topology draws, the two `O_N` boxes, the bank's 90.7 GB, an activity floor, a cap per task. G0 can run at any time; G5's configuration is still "afterwards" (D-039).

**[v1.10, 2026-10-08 16:45]** Where this stands: G3 is done -- the Giulia freeze of record (`29e38c89b08a2dd8`). The bank is designed (D-074..D-076) and B5 has its plan; B6 must give the joint runner a campaign prior before any Giulia element runs. Next: GC1, the list step of D-077's deletion (the G2 plan v2.8, sec. 4h), then GC2 (the user's); B5.1, the bank's plan step (read-only); G0 at any time; G5's configuration still "afterwards" (D-039).

**[v1.11, 2026-10-08 17:20]** Where this stands: GC1 ran -- 102 folders listed (`mea_out_1electrode` 51, `mea_out` 51; two folders under `mea_out/v1/` kept), CODE `e7cf50c2d9d5`; GC2, the deletion, given and the user's to run (the G2 plan v2.9, sec. 4h). D-078: the prior is the declared box. Next: GC2's paste; B5.1, the bank's plan step (read-only); G0 at any time; G5's configuration still "afterwards" (D-039).

**[v1.12, 2026-10-09 14:55]** Where this stands: GC2 is done -- `mea_out_1electrode` and `mea_out`'s 51 covered task folders are deleted (102 folders, 0 failed); `mea_out` keeps only the two stray test units under `v1/`. Next, in a new chat (`claude/HANDOFF_2026-10-09_giulia_B5.md`): B5.1, the bank's plan step (read-only); G0 at any time; G5's configuration still "afterwards" (D-039).

**[v1.13, 2026-10-10]** Where this stands: B5 is built and verified in the sandbox (`export_windows/`,
Blocks 6-8 of `specs/SPEC.md`), on the proposals Q6-Q8 and Q10-Q14, which the user has not answered; the bundle
`b5_g0_20261010` carries the two Sbi-extractor commits (the specs, then the banks) and the run sheet. Next (user):
LP on the laptop, D1-D2 on davinci, then G0 (G1-G8) and B5 (B1-B6) in parallel -- B1's printout (the disk
estimate) before B2. Then B6 (Claude, in SBI): the campaign prior in the runner, which stops at
`run_joint_arms.py:390` on these banks; and G5's sizing (the runner loads the whole bank: 90.7 GB raw before the
floor).

**[v1.14, 2026-10-12]** Where this stands: LP is on GitHub; G0 has started -- G1 ran (bash 4.4.20; no `conda` in its
executable-mode job), and G1b, the job-script probe, decides whether G5 runs as shipped or after an SBI fix. In
parallel: G2, G3, G4, G8 and the run sheet's lines B1-B6 in order, B1's printout before B2. Then, as in v1.13, the
build item B6 (Claude, in SBI: the campaign prior) and G5's sizing.

**[v1.15, 2026-10-12 15:20]** Where this stands: G1b passed, so G0 goes on as shipped -- G2, G3, G4, G5 and G8 now,
G6 and G7 after G5r's `GO` -- and B5 starts with B1 (its printout before B2).

**[v1.16, 2026-10-12 17:50]** Where this stands: G3 and G5 passed (G5r `GO`), so G0 goes on with G6, G7 and G7r. B1
refused one task for a counting rule of the plan's own (a topology folder with no simulation); the fix is in the
bundle `b51fix_20261012` -- LP2 (laptop), D2b (davinci: pull, both suites), then B1 again, its printout (the empty
topologies it names, the disk estimate) before B2. The two lines of work are independent. Then, as before, B6
(Claude, in SBI: the campaign prior) and G5's sizing.

**[v1.17, 2026-10-12 19:10]** Where this stands: G0 has exercised the joint stack on davinci end to end -- the probe,
two bench banks, the arms' dry run and one A1 element -- and only G8 (the tuner's `space` and `propose`) is left of it.
B5.1 is done: the plan of record, `bank_plan.json` `c8303506131a4632`. Next (user): B2 (the test task's dry run), B3
and B3r (the test task, a job), B4 (the array and the gate, `WALLTIME=auto`), B5 (the real bank), B6 (the runner's dry
run on one Giulia shard); and G8. Next (Claude): build item B6 in SBI -- the campaign prior, without which no Giulia
element trains (`run_joint_arms.py:390`) -- and G5's sizing: about 630,404 kept rows, 45.4 GB of float32 in memory
`[reasoning]`, against the runner's whole-bank load.

**[v1.18, 2026-10-13 15:30]** Where this stands: G0 is done (G8 ran). The bank (B3-B5) runs as the chain line BC, given
today; its status line is BCr. The rest of the Giulia work changes shape: the user requires one bash command per data set
(D-081), from the spike trains to the search and the best fit, with every optimization and training parameter in a
configuration file (D-082), the posteriors as a separate command (D-083), the runner reading the bank through a
streaming loader (D-084). Its work plan is `claude/PLAN_2026-10-13_giulia_pipeline.md`: eight blocks, of which the SBI
runner's (the campaign prior B6, the streaming loader, checkpoint and resume) come before any Giulia training element.
G5's configuration (D-039, "afterwards") becomes the `search` and `best_fit` blocks of that configuration file.

Critical path: G2 (the array over ~10^6 simulations, which waits for the C8
push) and B5 (new code). G1 is a day's work once the probe is read: one
array element per well, 15-min recordings. Claude's next builds, in order:
the config patch from the probe's log (minutes), B3 as soon as `sim_reextract/`
is on GitHub, then B5 while G2 runs.

## 9. Traps carried over

- `~/.bashrc` overrides `qsub -v` (`HPC_PATHS.md` 6b); `/tmp` carries nothing
  between nodes (6a); the conda `set -u` trap (7.1); `afterok` is ordering
  only (D-006); `smoke_test_mea_presence_check.py` hangs on davinci (trap
  6.8); `preflight_label_axes.py` counts NaN axes as swept (6.5); the
  registry drift on `Sigma` / `O_N` is handled per campaign by
  `registry_from_manifest` (6.6).
- The C8 tools state: the Giulia plan must refuse the 2026-09-30
  `submit_mea_array.sh` as C8's does (it cannot activate `sbi_export` on a
  compute node).
- `PROVIDER` is irrelevant to G4 (the banks come from the exporter, not from
  `build_latent_bank.py`), but G0's bench bank must pass `PROVIDER=bench`
  explicitly or it builds a fixture.
- `--sim-shards` takes Stage 1 shards only; the Parquet embedding exports of
  G6 are a different artefact class (usage sec. 2).
- **The file format is declared, never guessed** (new, 2026-10-01): a file
  of the other kind is refused naming the flag; the probe decides the
  declaration. The config's `sparse_peaks` / `peak_train` is a hypothesis
  from the file sizes and the SpyCode naming (Claude's recollection; the
  2010 SPYCODE paper is abstract-only on PubMed) and is confirmed or
  replaced by the probe before any array runs.
- **A refused well refuses the cohort** (D-002, M9/M15): with 17-35 files
  per AraC well, `InsufficientElectrodesError` is a live outcome; the probe's
  census and `exclude_wells` keep it out of the array, and the manifest
  records the exclusion.
- **Two senses of "index base"**: the user's "positions start from 1" is
  the numbering of rows and columns; the config's `index_base` is the
  loader's offset, 0 for the MCS code (D-048). Do not "fix" the config to 1.
- The Sbi-extractor patch avoids `.gitignore`, `EXTRACTOR_USAGE.md` and
  `sim_reextract/`, so it applies before or after the C8 commit (D-034); the
  v8.9 documentation goes in with `apply_usage_v8_9.py` AFTER C8's v8.8 row
  exists. **[v1.2: it applies on `5e800b6` (C8 + raster) unchanged; the
  raster commit took no `EXTRACTOR_USAGE.md` number, so v8.9 / sec. 15 are
  still free -- re-checked when the docs commit is cut]**
- **The MCS code's digit order** (`10*row + col` against `10*col + row`)
  is Claude's recollection, not checked against the device's documentation
  (D-048 decodes positions only). With E = 1 (D-043) the position enters
  only `validate_grid` and the electrode-map figure, so a wrong order
  transposes that figure and changes no trace (v1.2).
- **Read the refusal, not its headline** (v1.3): `cohort_manifest.py` names
  every bad well on indented lines under `REFUSED:`; `grep` for the headline
  hides them, and on a refusal the aggregation log has no
  `cohort_manifest_exit` line (`set -e`). Use `sed -n '/REFUSED/,$p'`.
  **[v1.4: from Sbi-extractor v8.10 the launcher says `cat` the whole log,
  the exit line is printed on a refusal too, and each refused well carries
  its array index and what happened]**
- **The launcher's "held until every array task exits 0" is false** (v1.3):
  davinci's `afterok` on an array is ordering only (D-006); the 2026-10-02 run
  showed it again. The line predates the Giulia work (`63d2574`).
- **A refused first extraction leaves a non-empty declared root** (v1.3):
  the launcher then refuses it as "the archives of record"; move it aside
  before re-running, do not delete it unasked. **[v1.4: from v8.10 the
  refusal says the root holds no manifest and names both possibilities; the
  move is step B7 of the G1-fix run sheet]**
- **The probe's STOP is real** (v1.3): `CONFIG DIFFERS on: exclude_wells`
  predicted exactly the refusal the array then hit; the array costs a queue
  wait, the probe seconds. **[v1.5: passed a second time at 15:07; from the
  third run on, C3 moves and dry-runs nothing unless the probe's last line is
  `CONFIG MATCHES`]**
- **A pull right after a push can come too early** (v1.5): davinci's "Already
  up to date" a few minutes after the laptop's `git push` meant the push had
  not landed (the second run). C1 checks the fixes' content, not the pull's
  message.
- **Never paste the dry run and the launch in one block** (v1.5): a pasted
  block runs every line whatever the one before printed. The commands are now
  gated: each runs only if the previous one printed its GO line.
- **One frozen manifest per cohort** (v1.6): C8's `launch_sim_reextract.sh`
  freezes into the fixed `artifacts/cohort_manifest/` and, once a copy is
  there, uses it whatever `COHORT_MANIFEST` says (lines 91-108 at `0a560ca`).
  Launched unchanged for Giulia, it would plan the hhgap campaigns with
  DUP15HD's manifest (`n_side` 3) if C8 had frozen first, or leave the Giulia
  manifest where C8 would later pick it up if Giulia went first: one cohort
  re-detected at the other's geometry, either way `[CODE; reasoning]`. B3
  freezes into `cohort_manifest_giulia/` and refuses a frozen copy that
  differs from the manifest it was given.
- **The flags file is tracked from the start** (v1.2): a config change
  without the matching `extraction_flags_giulia.sh` is refused by the
  launcher and by the preflight's step 5; the two are re-cut together.

## 10. Sources

`[KB]` the documents named in the header; `[CODE]` SBI `834eb41`:
`hpc/joint/stage1/latent_bank.py:27-59, 81-109`,
`stage1/build_latent_bank.py:52-90`, `stage3/jobs/joint_arms.pbs:2-5, 50-62`,
`stage4/jobs/joint_tune.pbs:2-5`, `JOINT_DSN_NPE_PLAN_v0_6.md:1382-1517`;
Sbi-extractor `63d2574`: `example_export.py:155-288`, `real_source.py:1-40`,
`extractor/launch_stage_d.sh:11-61, 83-89`, `dataset_profile.py:499-504`,
`preflight_giulia_export.py:443-450, 530-535`; ANN `37bf9f8`:
`hpc/MEA Traces/mea_probe.py:62-97`, `process_campaign.py:118, 532-575`,
`mea_detection.py:35`. `[SANDBOX 2026-10-01]` the geometry check of the
Giulia handoff v1.1 (E = 1 / 4 / 9 on an 8 x 8 grid; the probe at pitch
200 um on a 300 um square). The device reading is grounded in the handoff
v1.1 (Genocchi et al. 2024, PubMed full text).

v1.1 adds: `[LISTING 2026-10-01]` the user's `ls -lhR` of `Bio_Data/` (18
wells, 731 files, 60 codes, sizes 242 B to 94 KB); `[SANDBOX 2026-10-01]` the
`savemat` size measurements (a 9 000 000-sample zero raster compresses to
8 951 B; an empty sparse column is 224 B; 7 800 sparse spikes 93 824 B), the
MCS decoding check, the 67-check `smoke_test_ptrain_formats.py`, the 17 /
9 / 4 checks of the other suites, the launcher rehearsal in DRYRUN, gates
1-6 of the delivery skill on clean clones (`giulia_stage_d_bundle_20261001.tar.gz`,
47 166 B, sha256 `48cb20680cba51de46751ccf0446d4c8ff3c81c801a00b3208337319bd073347`);
`[PubMed, abstract only]` Bologna LL et al., Neural Netw 2010, DOI
10.1016/j.neunet.2010.05.002 (the SPYCODE paper; no PMC full text; the
abstract says nothing about file formats, so the `peak_train` convention
rests on Claude's recollection and is settled by the probe).

v1.2 adds: `[GITHUB 2026-10-01, read before 18:12]` Sbi-extractor
`feat/real-arm-parity` = `5e800b6` (raster) on `b20dd88` (C8, tree
`fa4e4eb5a52ea040dee5d1f71143137bf6d6486f`) on `63d2574`; ANN `main` =
`04977e0` (tree `186ce230c675aab7206ef3d8d5b82ed4fed1b6df`); SBI `main` =
`834eb41`. `[SANDBOX 2026-10-01]` the v2 rehearsal on those heads, 15/15
(`rehearse_push_real.sh`: fake remotes holding the real commits, a laptop
clone with `core.autocrlf=true`, the two push scripts, fresh davinci-side
clones, the 22 touched files compared byte for byte with the tested ones,
`apply_usage_v8_9.py` on the real post-C8 `EXTRACTOR_USAGE.md`, the
preflight, the C8 suite with the ANN tools at `04977e0` and the raster
suite, each against its baseline on `5e800b6`); the bundle
`giulia_stage_d_bundle_20261001_v2.tar.gz`, 59,804 B, sha256
`03a10063d431db49a6cc16c2a6bcaf6c904a4527f951806de438732cbf7e3fc4`.

v1.3 adds: `[GITHUB 2026-10-02, read 10:3x]` SBI `main` = `745edce`
(tree `79a353e5d73873798e679d6ae8e24bba5891c36c`), Sbi-extractor
`feat/real-arm-parity` = `481bd48` (tree
`5ec712b034e9c2c90e9b88aa46b3059cd4cdb93d`), both the tested trees.
`[CLUSTER 2026-10-02]` the user's terminal log of the first run (preflight,
probe, single well, DRYRUN, launch, the aggregation's REFUSED line).
`[CODE Sbi-extractor@481bd48]` `cohort_manifest.py:166-191` (the refusal and
its indented detail), `extractor/run_cohort_manifest.pbs:41, 90-98` (`set -e`
before the exit line), `extractor/run_extractor_array_mea.pbs:54, 135-154`
(`-j oe`; the folder made before the extractor runs), `extractor/launch_stage_d.sh:75-95,
157, 161`, `extractor/channel_subset_extraction.py:922-968` (K = floor(T_rec /
w_size)), `extractor/probe_ptrain_tree.py:449-490` (the fields the probe
compares).

v1.4 adds: `[CLUSTER 2026-10-02]` the user's paste of 13:12 (the
aggregation's REFUSED block, the two failed tasks' logs, the whole
`out/probe_giulia.log`: per-well files, active electrodes, `T_rec`, the
variables seen, the suggested fields, the CONFIG CHECK). `[GITHUB 2026-10-02,
read 13:15]` SBI `main` = `ab7ea82` (the documentation chat's seven commits,
`hpc/joint/docs/` only, on `745edce`); Sbi-extractor unchanged at `481bd48`.
`[SANDBOX 2026-10-02]` `g1fix_bundle_20261002.tar.gz` (sha256
`bd5e7b1dc4855eb79bd0e4d6df6e0d40bf27dd6421f0049eb9ad292df2e6c1c1`): every
suite on the patched trees (the manifest suite 18/18, the job suite 11/11,
the format suite 73/73, the metadata suite 9/9, C8 23/23, raster 12/12,
SBI cohort fields 4/4, preflight 7/7); M18, J1, J7b and J8 fail on
`481bd48`; the rehearsal on bare copies of the real heads with
`core.autocrlf=true` laptop clones, 17/17 (both push scripts PASS with
trees `9128cae8d5af0de0f2c0c61d49a4706da7bd9f3d` and
`d865823f96914b620242f00ce96acbad80175129`; the davinci-side files byte
for byte the tested ones; the preflight, C8 and raster suites; the whole
re-run on a replica of the census, 8/8).

v1.5 adds: `[CLUSTER 2026-10-02]` the user's paste of 15:09 (the second run:
the pull's "Already up to date", the preflight and probe lines, the move, the
dry run and the launch, the aggregation's log with `git=481bd48`, `qstat`).
`[GITHUB 2026-10-02, read 15:1x]` SBI `main` = `2c0a06d` on `ab7ea82`, tree
`9128cae8d5af0de0f2c0c61d49a4706da7bd9f3d`; Sbi-extractor
`feat/real-arm-parity` = `0a560ca` on `481bd48`, tree
`d865823f96914b620242f00ce96acbad80175129`; its seven files byte for byte the
tested ones. `[SANDBOX 2026-10-02]` `rehearse_rerun2.sh`: the commands C1-C6
run verbatim at the real paths from the reproduced post-second-run state,
16/16, run twice from scratch (the first pass's one failure was the rehearsal's
own check on C6, a regex where a fixed string was needed; C6's output was
right).

v1.6 adds: `[CLUSTER 2026-10-02]` the user's paste of 15:31 (C1-C4 in an
attachment, C5 and C6 in the message). `[SANDBOX 2026-10-02]` that paste
checked line by line against the rehearsal's expected lines, 12/12, after
stripping the attachment's Windows line endings; `git diff --shortstat
481bd48 0a560ca` = the pull's `7 files changed, 440 insertions(+), 36
deletions(-)`. `[GITHUB 2026-10-02, read 15:3x]` SBI `main` = `2c0a06d`,
Sbi-extractor `feat/real-arm-parity` = `0a560ca`, unchanged since the push.
`[CODE Sbi-extractor@0a560ca]` `sim_reextract/launch_sim_reextract.sh:66,
91-108` (the default `COHORT_MANIFEST`, the fixed freeze folder, the reuse of
an existing copy; `sim_reextract/` is the same at `481bd48`).
`[CODE SBI@2c0a06d]` `hpc/dsn/make_mea_specs.py:74, 273-372` (the config
import; `--out` defaults to `data.npz_specs`, resolved with
`os.path.abspath`), `hpc/dsn/config.py:48`, `hpc/dsn/backbone.py:41`
(torch), `hpc/dsn/hpc/Config/config_giulia_cohort.davinci.json`
(`data.npz_specs` = `hpc/Config/npz_specs_giulia.json`).

v1.7 adds: `[user]` the call of 16:01 (D-057). `[SANDBOX 2026-10-02]` the C8
run sheet and the deletion line rehearsed at the real paths, 21/21, twice
(`claude/HANDOFF_2026-10-01_c8_push_and_run.md` v1.2, P10).

v1.8 adds: `[CLUSTER 2026-10-08, the user's paste of 14:21: G2h]` (G2h); `[KB]` `claude/PLAN_2026-10-06_giulia_G2.md` v2.6 and the log v1.49; `[CODE Sbi-extractor@4e4dff0]` `preflight_label_axes.py:140-150` (the interface) and `sim_reextract/profiles/giulia.sh`; `[reasoning]` the sizes, run as arithmetic in the sandbox.

v1.9 adds: `[CLUSTER 2026-10-08, the user's paste of 15:20: N1]`; `[KB]` `claude/PLAN_2026-10-06_giulia_G2.md` v2.7, the log v1.50 and `EXTRACTOR_USAGE.md` sec. 5.5; `[CODE Sbi-extractor@4e4dff0]` `preflight_label_axes.py:86-135, 219-246` (the scan and the fields it writes).

v1.10 adds: `[CLUSTER 2026-10-08, the user's paste of 16:16: G3]`; `[user]` the card of about 16:30 (D-074..D-077); `[KB]` `claude/PLAN_2026-10-08_giulia_B5.md` v1, the G2 plan v2.8, the log v1.51; `[CODE SBI@9fcbc88]` `hpc/joint/stage1/latent_bank.py:27-60, 112-142, 184-204`, `hpc/joint/stage3/run_joint_arms.py:118-140, 369, 385-396, 539, 552-557`.

v1.11 adds: `[user]` this chat, 17:10 (D-078); `[CLUSTER 2026-10-08, the user's paste of 17:10: GC1]`; `[KB]` the G2 plan v2.9, the log v1.52, the B5 plan v1.1.

v1.12 adds: `[CLUSTER 2026-10-09, the user's paste of 14:45: GC2's status line]`; `[KB]` `claude/HANDOFF_2026-10-09_giulia_B5.md`.

v1.14 adds: `[GITHUB 2026-10-12]` Sbi-extractor `feat/real-arm-parity` = `9506d23`, SBI `main` = `9fcbc88`; `[CLUSTER 2026-10-12, the user's paste of G1r]`; `[KB]` `HANDOFF_hpc_implementation_v1_1.md` S3.7, S4.1; `[CODE SBI@9fcbc88]` `hpc/joint/stage1/jobs/build_latent_bank.pbs:66-77`, `stage3/jobs/joint_arms.pbs:78-89`, `stage3/jobs/probe_conda_activation.pbs` (unchanged from `a40a489`); `[CODE Sbi-extractor@9506d23]` `export_windows/run_bank_task.pbs:25-49`; `[WEB]` bash's CHANGES file (https://tiswww.case.edu/php/chet/bash/CHANGES), bash-4.4-rc2, sec. 3 item a; `[SANDBOX 2026-10-12]` the G1b rehearsal (`RUNSHEET_G1b.txt`, sha256 `aa0b98c43ec9571c`).

v1.15 adds: `[CLUSTER 2026-10-12, the user's paste of 15:17]` (G1br: G1b's log); `[SANDBOX 2026-10-12]` the revised G3r and G5r (`RUNSHEET_G0r.txt`, sha256 `5606da8d209b3053`), G5r's `STOP` on a same-seed, a wrong-provider and a wrong-folder bank.

v1.16 adds: `[CLUSTER 2026-10-12, the user's paste of 17:43]` (B1's REFUSED line, G3r, G5r); `[CODE Sbi-extractor@9506d23]` `sim_reextract/sim_reextract_plan.py:276-295, 331` (a task's `n_topos` is its count of raw `topo_*` folders), `sim_reextract/sim_reextract_gate.py:153-171, 314-316`, `export_windows/bank_plan_load.py` (the count it compared); `[SANDBOX 2026-10-12]` the fix, Sbi-extractor `8d28ae6` (tree `dc10065fc25a35c7ac0a84d237530cbb5c68e537`): suites 32 + 2 skipped and 15 + 1 skipped, 14 mutations caught, the bundle `b51fix_20261012` rehearsed on the stand-ins (the old code refusing as davinci did, then LP2, D2b and B1 passing).

v1.17 adds: `[CLUSTER 2026-10-12, the user's paste of 18:59]` (D2b, B1's printout, G6, G7's `qstat` and G7r); `[GITHUB 2026-10-12]` Sbi-extractor `feat/real-arm-parity` = `e9aeb55`, tree `dc10065fc25a`; `[CODE SBI@9fcbc88]` `hpc/joint/stage3/run_joint_arms.py:556-576, 636-656` (the spectrum's source, the report lines); `[KB]` `claude/HANDOFF_2026-10-01_c8_push_and_run.md` v1.23 (the other jobs in `qstat`).

v1.18 adds: `[CLUSTER 2026-10-13, the user's paste of 14:23]` (B2, G8); `[user]` 2026-10-13 14:23 and the two cards (log v1.57, D-081..D-084); `[KB]` `claude/PLAN_2026-10-13_giulia_pipeline.md` v1; `[SANDBOX 2026-10-13]` the chain line BC rehearsed (`RUNSHEET_CHAIN.txt`) on the stand-in with an asynchronous fake PBS: the full chain, the re-run, a refused test task, a refused gate.

v1.19 adds: `[CODE SBI@9fcbc88]` `hpc/joint/stage2/joint_train.py:124-271`, `stage4/npe_tune_joint.py:86-106`; `[SANDBOX 2026-10-13]` arm A1 trained twice on a bench bank at 40 and 400 posterior draws (the bundle `search_keys_checks_20261013`); `[KB]` `claude/PLAN_2026-10-13_giulia_pipeline.md` v1.1.
