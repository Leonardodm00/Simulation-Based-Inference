# C8 -- push the bundle, run the re-extraction. Handoff for a fresh chat (and for the raster chat).

| date | change |
|---|---|
| 2026-10-07 | v1.26. **C2 done; the C8 stage is finished on davinci.** `[CLUSTER 2026-10-07, the user's paste of 14:56: C2's log tail, G2a-G2d]`: C2 finished: `[clean] DONE: 376 deleted, 0 failed`. G2a-G2d ran and passed (the Giulia work continues in `claude/PLAN_2026-10-06_giulia_G2.md` v2.2). New in P10: the v1.26 block, closing this handoff's run. Log v1.45. |
| 2026-10-07 | v1.25. C1b ran `[CLUSTER 2026-10-07, the user's paste of 14:11: C1b]`: C1b listed 376 folders, nothing deleted, CODE `681c30382046`: A, the raw simulations of the record's 160 replays, 160 folders, 438,064 files, 136.6 GB; B, the old `Outputs/` detections of the tasks the record covers, 214 folders, 340,580 files, 50.6 GB (kept under `Outputs/`, not covered: `v7/sweep_cpu_task0009`); C, the moved replay detections (the 10-03 test task), 1 folder, 1,336 files, 0.2 GB; E, `Deep_bio/extracted`, 1 folder, 420 files, 0.3 GB, all 35 wells re-extracted in `extracted_v2`; no D line (the two Giulia partial roots); the NOTE on `v2/sweep_intel_task0008` (1 iteration file lost, D-064) and the C0 NOTE (the SBI config and `artifacts/specs_real.json` still name `extracted/`). New in P10: the v1.25 block -- that list read, and C2 with its CODE. Log v1.44. |
| 2026-10-07 | v1.24. **The C8 record exists; Stage C is closed.** `[CLUSTER 2026-10-07, the user's paste of 13:13: the record, cat whole]`: `Outputs_v2/REEXTRACTION_RECORD.json` written 2026-10-07 13:11:04 +02:00, sha256 `7a55c96cbda6a1a6`, from the plan of 13:09:09: 94 tasks, 285,178 iterations, 285,178 files read, 208 excluded -- 160 replays (1 file lost), 46 empty sweep folders, `v7/sweep_cpu_task0009` (no `manifest.json`), and `v12/sweep_cfd_task0011` left out by name (D-073); `--n_side 3 --pitch 60.0 --edge 25.0 --fs 10110.09` (`n_e` 9); the cohort manifest `c925c5106137fe82`; the tools "ANN main C8 (2026-10-01, ...; tree 6e01860c)"; the library `b50814ff27d3c496`; one environment, `sbi_export`, python 3.11.15, numpy 2.4.6, scipy 1.17.1; the contract mixed on `simtime`, `log_params`, `param_bounds`, `manifest_version` (D-059); 254 tasks compared for replays, 94 distinct seeds, 47 groups; 7 warnings, the `_failures.log` files of v1.22. Checked by Claude against the v1.23 expectations, every value as expected. New in P10: the v1.24 block -- the record, and the order from here: C1b, C2, then G2a-G2d. Log v1.43. |
| 2026-10-07 | v1.23. **D3 read: `v12/sweep_cfd_task0011` is still simulating; D-073 -- it is left out of this record.** `[CLUSTER 2026-10-07, the user's paste of 12:44: D3]`: its sweep job `1562195[11]` `R` on `dvnode208` (192 cores), 40:43:30 of 70:00:00 used, 80 topologies asked, 5,760 iterations written in 15 topologies, the newest 9 min old; 1 of 302 raw folders differs from the plan. The user, on a question card: "leave it out for now" (log D-073). New in P10: the v1.23 block -- M3 (moves its partial output out of `Outputs_v2`), G7 (the plan with the task left out and the gate's line, dry), G8 (the gate alone), then S9y; rehearsed 14/14 twice on the C8 suite's fixture `[SANDBOX 2026-10-07]`. Corrections: v1.22's header time (12:30, not 12:45); the account of what the gate catches made precise (each topology is listed when the MEA job reaches it), in v1.22's paragraph and in the v1.8 bracket. Log v1.42. |
| 2026-10-07 | v1.22. **The re-run's gate REFUSED one task of 95, `v12/sweep_cfd_task0011`: its simulation was still being written at the plan.** `[CLUSTER 2026-10-07, the user's paste of 12:08: S9y, then the gate log whole]`: gate `1807427` on `dvnode003`, 2026-10-06 21:41:29-21:42:49, `git=1b23adc`; 95 tasks, 288,250 iterations planned, 288,442 `mea_iter_*.npz` read; the one FAIL: `total_iters 3264, planned 3072` and `topo_00008: 192 mea_iter_*.npz, raw has 0 iter_*.npz`; seven WARN lines (`_failures.log` of the first attempt, every iteration done) in six tasks; nothing written, exit 1. The "raw has 0" is the plan's count: the plan saw `topo_00008` created and empty -- the sweep makes a topology's folder before its iterations -- and the task's MEA job found 192 there. The gate compares each output with the plan's counts and does not re-count the raw folders `[CODE Sbi-extractor@1b23adc sim_reextract_gate.py:112, 146-163]`: v1.8's sentence that it does is corrected (marked there). New in P10: the v1.22 block -- the reading, the warnings, D3 (read-only: the task's simulation job and counts, every raw folder of the plan against its counts, what `--resume` would do; one verdict line; 34/34 checks on a mock, six mutations caught, the line rehearsed at the real paths), and what follows each verdict. No cleanup and no G2 line until a record exists (v1.20 block). Log v1.41. |
| 2026-10-07 | v1.21. The reminder of 09:00 fired. `[GITHUB 2026-10-07, read by Claude about 07:05 UTC: git ls-remote and a shallow fetch]`: L5a and L5b ran: both commits are on GitHub with the tested trees, committed on 2026-10-06 at 18:13 +0200 (Sbi-extractor `4e4dff0`, tree `460d7494...`; ANN `d0ced88`, tree `8aa8cc11...`). Nothing on davinci is known to have changed since v1.20. New in P10: the v1.21 block -- today's order and what each step should print. Log v1.40. |
| 2026-10-06 | v1.20. No change to the C8 lines. The Giulia G2 lines are ready (`claude/PLAN_2026-10-06_giulia_G2.md` v2, sec. 4: laptop L5a and L5b, davinci G2a-G2d, rehearsed `[SANDBOX 2026-10-06]`); on davinci they come after C2, and not at all before a C8 record: a C8 re-run after G2a would mix two tool states and two noise schemes in one record (D-072), which the gate refuses. New in P10: the v1.20 block. Log v1.39. |
| 2026-10-06 | v1.19. **Until S9y prints the record, nothing in `ANN/MEA_analysis/` changes and no launcher mode runs in `sim_reextract/`** -- the gate re-reads the tools, the library, the cohort manifest and `plan.json` when it starts (`sim_reextract_gate.py:209-228` `[CODE Sbi-extractor@1b23adc]`); the Giulia G2 work (B3, D-069..D-071) is built meanwhile in the sandbox only. A reminder is scheduled into this chat for 2026-10-07 09:00 Europe/Rome: S9y, then the cleanup. New in P10: the v1.19 block -- C1b, C1 with its grep widened to group E. Log v1.38. |
| 2026-10-06 | v1.18. **The re-run is submitted.** `[CLUSTER 2026-10-06, the user's paste of 15:42]`: S7z -- 19 tasks to run (array 0-18), 63,063 iterations, largest 11,728 over 26 topologies; 76 kept as done (--resume); 255 tasks compared, 95 distinct seeds, 47 same-seed groups, 160 replays dropped (1 file lost); EXCLUDED 207, none left out by name; WALLTIME 15:10:00; `v9/sweep_intel_task0002` planned at 6129 (M2's move took effect; M2's own output not pasted). S8y: the same counts, array `1807426[]`, gate `1807427`, no ABORT or REFUSED. The grep the launcher suggests, run right after, printed this morning's refusal: the gate log is the 07:16 one until `1807427` ends. New in P10: the v1.18 block -- that reading, PROG (95 expected), a node check, S9y. Log v1.37. |
| 2026-10-06 | v1.17. **D2 read; the re-run is set up.** `[CLUSTER 2026-10-06, the user's paste of 15:10]`: `dvnode012` `state = job-busy` (in service, no comment), `dvnode022` free; the raw file `v9/sweep_intel_task0002/topo_00025/iter_00100.npz` is 608,526 bytes and NOT A ZIP, its neighbours valid, no other copy in any campaign; only `v12/sweep_cfd_task0011` written in the last 6 h (960 files, newest 189 min old). L4 and P3 passed (`[GITHUB 2026-10-06]` head `1b23adc`, tree `4ee52e57...`). The user's answers on a question card after D2: D-067 (the damaged raw file is moved aside, into `Main_quarantine/`, not deleted), D-068 (`v12/sweep_cfd_task0010` goes into the re-run; `v12/sweep_cfd_task0011` is finished and goes in too). New in P10: the v1.17 block -- M2 (the guarded move), S7z (S7y with `XT=` empty), S8y and S9y unchanged, a note for the HPC support. Log v1.36. |
| 2026-10-06 | v1.16. **D1 read on davinci: one node killed the 16 tasks; one raw simulation file is unreadable.** `[CLUSTER 2026-10-06, the user's paste of 14:54, D1 through c8_diag.py at 459aa72]`: 16 of the 17 bad tasks ran on `dvnode012` (17 tasks went there, one passed), each dying 15-57 min after it started, one after another from 16:35 to 02:28, with `OSError [Errno 5] Input/output error`, `[Errno 19] No such device`, `[Errno 108] Cannot send after transport endpoint shutdown` on `Outputs_v2` and `Bus error (core dumped)` (exit 135 = signal 7; 1; 120); `/davinci-1/home` is a Lustre mount (`df`), and -108 is the Lustre client being evicted; the 17th, `v9/sweep_intel_task0002` on `dvnode022`, finished 6129/6130: its RAW file `topo_00025/iter_00100.npz` raised `BadZipFile`. All 17 are re-run by RESUME (none kept by a manifest). The 76 good tasks: 2.30 s per iteration (median; 0.97-3.09), runs up to 6.7 h, `workers` 1 in all (`PBS_NCPUS` unset). The user, 14:54: D-066 (the Giulia `mea_out_1electrode` waits for G2; the DUP15HD pre-Stage-D archives go if re-extracted). New in P10: the v1.16 block -- the reading, D2 (the nodes, every copy of the unreadable raw file, what is still being written), the second patch `c8_cleanup_E.patch` (group E: `Deep_bio/extracted/`), L4 and P3. Corrections to v1.14: Lustre, not GPFS; one node through the night, not one event. Log v1.35. |
| 2026-10-06 | v1.15. **D1 goes through git, with a cleanup tool for after the record** (the user, 12:49: "the push and pull please"; 12:51 and the question card: D-065). Sbi-extractor patch `c8_tools.patch` on `a8e9a43` (tree `67133a86...`): `sim_reextract/c8_diag.py` (D1 as a module, the same output), `sim_reextract/c8_cleanup.py` (after the record: lists, then deletes with a CODE, the raw simulations of the replays, the old `Outputs/` detections of every task the record covers, the moved test task, the Giulia real-arm partial roots), `sim_reextract/smoke_test_c8_tools.py` (13 checks), a README section; tree after it `404496d5ed0fe8bc9e32911016a070feb9feadb6`. `[SANDBOX 2026-10-06]`: 13/13 twice; seven mutations each caught; the 26-check C8 suite 26/26 on the new tree; L3 and P2 rehearsed with their STOP paths. New in P10: the v1.15 block -- L3 (laptop push), P2 (davinci pull, the 13 checks, D1), C1 and C2 (the cleanup, after the record). The v1.14 block's heredoc is superseded by P2. Log v1.34. |
| 2026-10-06 | v1.14. **The array ran; the gate REFUSED: 17 of 93 tasks bad, nothing written; Stage C stays open.** `[CLUSTER 2026-10-06 07:16:41]` (S9y, the user's paste of 12:28): `[gate] read 239732 mea_iter_*.npz file(s) across 93 task(s); 17 task(s) bad`, `[job] sim_reextract_gate_exit=1`; 276,347 iterations were planned, so 36,615 outputs are missing or unreadable, all in the 17. Two signatures in the 60-line tail: tasks that stopped part-way (no `mea_manifest.json`, one topology part done, later ones without a folder, the newest files `EOFError('No data left in file')`), and `v9/sweep_intel_task0002` finished at 6129 of 6130 with a `_failures.log` in `topo_00025`. Claude's reading `[reasoning, from the code; NOT VERIFIED on davinci]`: the EOFError is a zero-byte file `[verified by running]`, and `process_campaign.py` writes by temp file + rename, so a zero-byte file under the final name is data lost after the rename (a node or filesystem event, or space/quota reported late), not a walltime or code kill. New in P10: the v1.14 block -- the refusal, that reading, D1 (a read-only diagnostic: per bad task the disk state, `mea_env.json` host/workers/start, whether RESUME would re-run it; PBS exit status; the job logs' error lines; storage), tested on a mock with the real gate. Log v1.33. |
| 2026-10-05 | v1.13. **The v2 gate is on GitHub; the plan ran on the real data; the array is NOT submitted yet.** `[GITHUB 2026-10-05 about 15:25]` (read by Claude: `git ls-remote` and a shallow fetch) `feat/real-arm-parity` is at `a8e9a43` (2026-10-05 14:38:12 +0200), tree `67133a86a70943bc8083629ba18fb186c1333e58`, parent `6173e92`, five files +508 -16 -- the v2 tree of v1.12; its trailer names Claude Sonnet 5.5, so the L2 that was run is the one sent in the chat (the v1.12 block's L2 line differs from it in that one word: Opus 5.5). `[CLUSTER 2026-10-05 15:00-15:10]` (the user's pastes): the pasted line of P shows the v12 task `sweep_cfd_task0010` still being written (214 files, newest 0 minutes old); the first S7y REFUSED on the test task `v1/sweep_cpu_task0000` alone, a replay of `v9/sweep_cpu_task0000`; M1 moved its output to `Outputs_v2_replays_moved/campaign_cadex_rho1300v1/sweep_cpu_task0000`; the second S7y: 93 tasks to run (array 0-92), 276,347 iterations, 160 replays dropped (one of them with 1 file lost), 208 tasks excluded, WALLTIME 15:10:00, the v12 task left out by name. New in P10: the v1.13 block -- that record, the walltime check, G0 (new: a read-only geometry check of the 10-03 test task's output, tested in the sandbox on real tool output), P again, S8y, a progress line (new), S9y, and what the lines do not cover. P0 items 2-4 carry corrections (C8's two commits and the raster commit are on GitHub since 2026-10-01 17:38). No code or patch change. Log v1.32. |
| 2026-10-05 | v1.12. **The census C1 ran on davinci; the gate is re-cut as v2, one task per seed** (log v1.31, D-064). Census `[CLUSTER 2026-10-05 14:00]`: all 255 tasks with `job_args.json` carry `_resolved_seed_master` and the offsets; 47 seeds are shared by 208 tasks of v1-v9, no shared-seed pair with different theta; the test task `v1/sweep_cpu_task0000` is a subset of `v9/sweep_cpu_task0000`; two pairs only overlap. The user (D-064): an overlapping replay is dropped too, with the files only it holds (counted, `files_lost`). v2 also moves a replay's detections OUT of `Outputs_v2` in its refusal advice. Patch `c8_replay_gate_v2.patch` on `6173e92`, tree after it `67133a86...` -- the v1.11 patch (tree `39643844`) is superseded, not to be applied. New in P10: the v1.12 block -- L2 and S3y for v2, and M1, which moves the test task's output out of `Outputs_v2` once S7y names it as a replay (the census says it will); S7y, S8y, S9y unchanged. Suite 26/26 twice, six mutations each failed P7, run sheet v2 16/16 twice `[SANDBOX 2026-10-05]`. |
| 2026-10-05 | v1.11. **The replay gate is built** (log D-061, D-063: the user chose, on a question card, to keep the copy with the most iterations and to compare 10 shared files per pair). The C8 plan now drops a task that recorded the same seed values (`_resolved_seed_master` and the offsets; the key read from ANN's sweep code, which writes it since 2026-06-16) and the same contract as a task kept before it, when its iteration files are the same as or a subset of that task's and theta and params are byte-identical at the sampled files; the rest of a same-seed group is kept and printed. Sbi-extractor patch `c8_replay_gate.patch` on `6173e92` (tree `baeaa95f`, on GitHub), tree after it `39643844...`; suite 26/26 twice, five mutations of the gate each failed P7 alone `[SANDBOX 2026-10-05]`. New in P10: the v1.11 block -- L2 (laptop push), S3y (pull + suite), P (as before), S7y (the plan with the gate + the array's dry run; `XT=` empty if P shows the v12 task finished), S8y (submit), S9y (the record) -- rehearsed 13/13 twice on a replica `[SANDBOX 2026-10-05]`. C1 is no longer needed: S7y prints every same-seed group before anything is submitted. The v1.10 block's S7x/S8x/S9x are superseded (marked). Not on GitHub yet: this session cannot push to Sbi-extractor (log D-062). |
| 2026-10-03 | v1.10. S7x ran on davinci and passed `[CLUSTER 2026-10-03 16:08]`: 252 tasks to run (array 0-251), 491,893 iterations, largest 11,728 over 26 topologies, 1 kept as done, 48 excluded with `v12/sweep_cfd_task0010` left out by name, WALLTIME 15:10:00 (the `cpu` queue allows 300:00:00); so L and S3x had passed. **S8x is on hold**: the user's call of 16:08 (log D-061) -- tasks that replay another task's seed are not re-detected, and the plan has no such gate. New in P10: the v1.10 block -- C1, a read-only census of the seeds over the plan's tasks, tested on a fixture with every case `[SANDBOX 2026-10-03]`. |
| 2026-10-03 | v1.9. The user asked (15:57) whether the left-out task can be added to the same folder once it has finished. It can, to `Outputs_v2` and to one record covering every task, with today's code -- v1.8's "the task cannot be added to it later" was wrong and is marked [corrected]. The route: rename the record and its `.sha256` aside, re-run the same plan/array lines with `RESUME=1` and without `--exclude-task`; the plan keeps every recorded task as done and runs only the finished one, and the gate re-checks all of them and writes a new record. Rehearsed on the v1.8 replica: 4 more checks, 22/22 `[SANDBOX 2026-10-03]`. Conditions and the provenance gap (the new record does not name the first) in P10's new paragraph "Adding the left-out task later". |
| 2026-10-03 | v1.8. S6 PASSED on davinci: job `1797833` ran in `sbi_export` (`env activated: .../envs/sbi_export/bin/python3`, python 3.11.15, numpy 2.4.6, scipy 1.17.1, no WARNING), `[mea] 6 topos, 1334/1334 iters processed in 3102.1 s` `[CLUSTER 2026-10-03 15:25]`; WALLTIME 15:10:00 from it. The user: a v12 `cfd` task is still simulating, "for now let's just use the campaigns available" (log D-060, I-002). The plan cannot tell a simulation still being written from a finished one, so the task is left out by name: a new plan option `--exclude-task` (Sbi-extractor patch `c8_exclude_task.patch`, tree `baeaa95f...`, suite 25 checks; this session cannot push, so it goes through the laptop). New in P10: the v1.8 block -- P (what is being written now), L (laptop push), S3x (pull + suite), S7x (plan with the exclusion, WALLTIME, array dry run), S8x (submit), S9x (the record) -- rehearsed 18/18 on a replica with a task being written `[SANDBOX 2026-10-03]`; and a hazard found while building it: the gate job reads `plan.json` when it starts, and every launcher mode rewrites it, so no launcher mode runs between the array's submission and the gate's log. P10's "After S6" paragraph is superseded by that block (marked). |
| 2026-10-03 | v1.7. S4m ran on davinci and made the plan over the mixed contract: 254 tasks (array 0-253), 495,531 iterations, largest task 11,728 iterations / 26 topologies, 47 excluded; mixed `simtime` 200 x214 / 180 x40, `log_params` and `param_bounds` x243 / x11, `manifest_version` 3 x214 / 4 x40 (reported only); library sha256 `b50814ff27d3c496` `[CLUSTER 2026-10-03 14:20]`. S5 then REFUSED: every launcher mode re-runs the plan with `$PLAN_ARGS` (`launch_sim_reextract.sh:114-117`), and S5 did not set it. [corrected 2026-10-03: v1.6's "S5 and S6 are unchanged" was wrong -- S5 and the array need `PLAN_ARGS=--allow-mixed-contract` too.] P10 gains S5m. Nothing was submitted. |
| 2026-10-03 | v1.6. The suite fix is on davinci and passes: `GO: the suite fix is on davinci (5682a6c)` (the laptop's commit, with the tested tree `c4b9c58c...`) and `ALL 23 CHECKS PASSED` `[CLUSTER 2026-10-03]`. The user decided the mixed contract: "They can definitely go together" (log v1.24, D-059) -- C8 runs over all 253 tasks with `PLAN_ARGS=--allow-mixed-contract`. P10 gains S4m, the S4 line with that variable; the frozen manifest copy exists since S4b, so S4m prints `[launch] frozen manifest: ...` instead of `[launch] froze ...`, and the mixed fields under `[plan] contract fields with more than one value:`. |
| 2026-10-02 | v1.5. Two findings from davinci (the user's paste of 17:18). (1) S4b confirmed the diagnosis of v1.4 (`notfound` / `found`, `E1 E3`, both in `base`) and the suite is now fixed: its fixture links `~/.conda/envs/sbi_export` to the env the suite runs in (`sim_reextract/smoke_test_sim_reextract.py`, +15 -1; tree `c4b9c58c0e2d610f16a3bd799b203e7c4d3d10ed` on `0a560ca`). The failure was reproduced in the sandbox with a real conda (Miniforge, conda 26.7.2, a user-level env without conda in it, base with another numpy): `FAILED 2 of 23`, E1 and E3; with the fix 23/23 twice, and 23/23 under the system python `[SANDBOX 2026-10-02]`. This session cannot push to Sbi-extractor (read-only GitHub access), so the commit goes through the laptop as a patch (P10, S3c). (2) The plan REFUSED the campaign set on its label contract: v10-v12 at 180 s against 200 s, v10's own `log_params` and `param_bounds` -- D-030 does not hold on disk (log v1.22, Open calls). |
| 2026-10-02 | v1.4. S2 by git PASSED on davinci (`GO 2`, ANN `04977e0`) and S3 gave `ALL 14 CHECKS PASSED` / `FAILED 2 of 23` `[CLUSTER 2026-10-02 17:01]`. The two failures, E1 and E3, are the suite's own environment, not C8: the suite runs its fake job with `HOME` set to its temporary fixture and every `CONDA*` variable removed (`smoke_test_sim_reextract.py:301-308`), so the job cannot find `sbi_export` (a user env, resolved through `~/.conda/envs` and `~/.condarc`) and falls back to `base` (python 3.10.14, numpy 1.26.4); E1 and E3 compare the job's numpy with the suite's and fail on that alone -- E3's record shows the right counts, geometry and tools label `[CODE Sbi-extractor@0a560ca; ANN@04977e0 submit_mea_array.sh:85-175]`. The sandbox rehearsal missed it: no conda there, so job and suite shared one interpreter. New in P10: S4b, which checks that diagnosis on davinci and makes the plan only if it holds; rehearsed with a stub conda, positive and three negatives `[SANDBOX 2026-10-02]`. Not fixed: the suite itself (a Sbi-extractor change, later). |
| 2026-10-02 | v1.3. The user could not upload the archive (16:53: "I canno load the tar do a push and pull please"). Nothing needs pushing: the five files are on GitHub at ANN `main` `04977e0` since 2026-10-01 17:38, byte-identical to the archive's `[GITHUB 2026-10-02]`. New in P10: S2 by git -- one line that makes a throwaway shallow clone of ANN in `~/ann_c8_tmp`, copies the five files into `ANN/MEA_analysis/`, deletes the clone and checks the same five hashes; rehearsed on a copy of the folder at its `37bf9f8` state against the real GitHub, twice, plus a failed clone (STOP, files untouched); the ANN environment test then passed 14/14 `[SANDBOX 2026-10-02]`. |
| 2026-10-02 | v1.2. **C8 runs now** (the user, 16:01 in the Giulia chat: "C8's record do this now"; log D-057, which also makes the Giulia project's G2 wait for this record). Since v1.1: both C8 commits were pushed on 2026-10-01 at 17:38 with the tested trees (log D-034); the raster commit and the two Giulia commits followed on `feat/real-arm-parity` (head `0a560ca`), none of them touching `sim_reextract/`; and G1 ran ahead of C8 (D-053) and is done. New P10: the run sheet re-issued against today's heads as one-line gated steps S1-S6 -- status, the ANN upload as one archive, the two suites, the plan, the test task, the test log -- rehearsed at the real paths, 21/21, twice `[SANDBOX 2026-10-02]`. Corrections in P5, marked: the test job's WARNING goes to `~/c8_test.e<id>`, not to the `.o` file (`#PBS -k eo` without `-j oe`); the gate log is read whole, since a refusal's reasons are on indented lines under each `FAIL`; the ANN environment test also needs `launch_mea_array.sh` and `launch_mea_per_campaign.sh` beside it. P0 item 6, P6 and P9 annotated. |
| 2026-10-01 | v1.1. The Giulia handoff's question count corrected, in P0 item 6 and in P8: eleven questions, Q1-Q11, as in v1.1 of `claude/HANDOFF_2026-10-01_giulia.md` (revised by the other chat while this document was being built); v1 of this document had carried the seven, Q1-Q7, of that handoff's first version. Found when the project copy was read back after the write. Also v1's "Written at 14:22" made exact: the turn began at 14:22, the write came later. Nothing else changed. |
| 2026-10-01 | v1. Written in the turn that began at 14:22 (Europe/Rome), when the user answered the two questions left open by the C8 build (log D-033, D-034): "exporting the new simulations" meant the re-extraction alone, so Stage C ends at C8's record; the C8 commits are pushed first and the raster commit lands afterwards, on top. Holds the commands that push `c8_bundle_20261001.zip` (laptop) and run C8 (davinci), the bundle's identity, the tree id a raster commit is cut against, what a refusal means, and what was and was not checked. The run commands are copied from the bundle's own `README.txt` and push scripts, read again the same turn; the pre-checks (P3) and the post-push checks (P4) are Claude's additions. Nothing here has run on the laptop's clones or on davinci. |

Read first: `claude/SBI_decisions_and_ideas_log.md` (D-033 and D-034 are this
document's decisions; D-015, D-021, D-023, D-024, D-025 and D-030 fix what C8
runs), `claude/HANDOFF_2026-09-21_stageC.md` (C8, and the "what is left" table
as corrected at 14:22), `HPC_PATHS.md` (secs. 4, 4g, 7.2, L). Tags as in
`HPC_PATHS.md`: `[user]` said by the user in a chat; `[SANDBOX 2026-10-01]` run
in Claude's sandbox that day; `[CLUSTER ...]` seen on davinci; `[reasoning]`
Claude's own reading; `[NOT VERIFIED]` not read or not run.

## P0. The six facts a fresh chat must have

1. **Stage C is the simulated-arm re-extraction alone, and it ends at its
   record**, `ANN/MEA_analysis/Outputs_v2/REEXTRACTION_RECORD.json` (with its
   `.sha256`) `[user, 2026-10-01 14:22, D-033]`: "It is just re-extraction
   given the new environment." No embedding export is made before the DSN
   retrain, so there is no provisional export on the r2 encoder. The
   embedding export of the new simulations waits for the retrain (D-031) and
   the encoder it produces.
2. **C8 is built and rehearsed, and has not run.** `c8_bundle_20261001.zip`
   (identity in P2) holds two commits -- one for Sbi-extractor, one for
   Astro-Neuron-Network (ANN) -- their two push scripts, three ANN files to
   upload to davinci, and the run sequence. `[SANDBOX 2026-10-01]`
   **[corrected 2026-10-05: still true of the array, which is not submitted (P10, v1.13). The machinery has run: the plan on the
   real tasks (S7y, twice) and, on 2026-10-03, the test task `v1/sweep_cpu_task0000` (1,334 iterations).]**
3. **Push order (D-034, `[user]`: "It will land afterwards.").** The two C8
   commits are pushed first and unchanged. The raster commit -- the user's
   other chat, typical-well rasters of the real cohort, D-026..D-029 -- lands
   afterwards, on top of C8, on the same Sbi-extractor branch,
   `feat/real-arm-parity` (that is what Claude's reply of 2026-10-01 13:10
   said and the user's answer did not correct; the raster commit's base and
   files were not read here `[NOT VERIFIED]`).
   **[corrected 2026-10-05 `[GITHUB 2026-10-05 about 15:25]`: both are on `feat/real-arm-parity` -- C8's Sbi-extractor commit `b20dd88`
   (2026-10-01 17:38:27) and, 16 seconds later, the raster commit `5e800b6` on top of it; since then `481bd48` and `0a560ca` (10-02),
   `5682a6c` (10-02), `6173e92` (10-03) and `a8e9a43`, the replay gate v2 (10-05 14:38, tree
   `67133a86a70943bc8083629ba18fb186c1333e58`).]**
4. **The tree a raster commit is cut against** is
   `fa4e4eb5a52ea040dee5d1f71143137bf6d6486f` -- the tree of
   `feat/real-arm-parity` once C8's Sbi-extractor commit is on it. It is a
   TREE id on purpose: the commit the laptop makes from the patch has its own
   commit id, not the sandbox's `87a225f`, so a base check keyed on a commit
   id would fail. P4 has the commands.
   **[corrected 2026-10-05: historical. `fa4e4eb5...` is the tree of `b20dd88`, the raster commit `5e800b6` sits on it, and the branch
   head is now `a8e9a43` (see item 3).]**
5. **One call D-033 leaves open:** `label_axes.json` over v1-v5, v7-v12
   (Stage C status row A3). C8 does not read it; it feeds the embedding
   export, so Claude's reading is that it moves with that export. Not asked.
6. **Next, after the record exists:** the Giulia project (log I-001,
   `claude/HANDOFF_2026-10-01_giulia.md` v1.1; its eleven questions, Q1-Q11,
   are not yet answered). Not before.
   **[2026-10-02: G1, the Giulia real-arm extraction, ran ahead of C8 by the
   user's call (D-053) and is done; everything after it -- G2 onward --
   waits for this record (D-057).]**

## P1. What C8 is, in plain words

The "simulated arm" is the set of raw DUP15HD network simulations
(`campaign_cadex_rho1300v1-5,7-12`: spike times of the simulated cultures,
`theta` and spike times untouched by C8). The virtual MEA -- ANN's
`process_campaign.py` -- turns each simulation into the detections a
multi-electrode array would have produced. C8 re-runs it over every existing
raw simulation into a new root, `Outputs_v2`, so that the whole simulated
bank is made by the new pipeline, and with the electrode patch the real arm's
cohort manifest defines: `n_side = isqrt(electrodes_per_subset) = 3`, so
`n_e = 9` electrodes per subregion (D-015), pitch 60 um (D-021), electrode
side 25 um (D-024), `fs` 10110.09 Hz (the manifest's `fs_raw`), in the
`sbi_export` environment (D-023). It runs as: plan (login node) -> one test
task -> a PBS array (253 tasks, 20 at a time, 48 cores each) -> a gate job.
The gate checks every planned task's output against the plan and writes the
record, or REFUSES naming every bad task and writes nothing. Plain version:
detect spikes again on every simulated culture with the same virtual electrode
patch as the real recordings, so that real and simulated data come out of one
pipeline; the record is the receipt, and the export over `Outputs_v2` will
cite it.

## P2. The bundle: identity

| | value | tag |
|---|---|---|
| file | `c8_bundle_20261001.zip`, 73,017 bytes, sha256 `d7121dd9d3749b1780925f8cd28115a1ce76cb38ee39f74d58f9827d79859a04` | `[SANDBOX 2026-10-01]` |
| folder inside | `README.txt`; `sbix/` (patch, `0001.msg`, `push_sbix_c8.sh`); `ann/` (patch, `0001.msg`, `push_ann_c8.sh`); `davinci_upload/` (`submit_mea_array.sh`, `smoke_test_mea_env.py`, `README.md`, `SHA256SUMS`) | same |

| | Sbi-extractor | Astro-Neuron-Network |
|---|---|---|
| laptop clone | `C:\Users\leona\Sbi-extractor` | `C:\Users\leona\Astro-Neuron-Network` |
| branch | `feat/real-arm-parity` | `main` |
| cut against (base commit) | `63d2574dd4d0da976dbc65413fc889f05692dbbd` | `37bf9f83a3c011887f21b205a32c6095de9c7887` |
| base tree | `8e28e2e19dbcf5bb174ae3b2ed66a3c8e524f886` | `28a4b0edfcf3f18c95e26fef1148cd838ae4671f` |
| **tree after the commit** | **`fa4e4eb5a52ea040dee5d1f71143137bf6d6486f`** | **`186ce230c675aab7206ef3d8d5b82ed4fed1b6df`** |
| commit subject | `sim_reextract: plan, run and gate of the simulated-arm re-extraction (Stage C, C8)` | `MEA array job: find the cluster's conda; write mea_env.json per work unit` |
| sandbox commit id (information only) | `87a225f2f65ef746f51c16b4e42ece2764d5eb29` | `1a98a6aea920a803acd58c71e6c5622eef53a116` |
| touches | 8 files, +1,944: `.gitignore` +5 (a blank line, a comment, `sim_reextract/plan.json`, `tasks.tsv`, `tasks_test.tsv`, appended at the end); `EXTRACTOR_USAGE.md` +23 (the v8.8 changelog row directly above the v8.7 row, and a new sec. 14 appended at the end); six new files in `sim_reextract/` | 3 files, +147 -7, all in `hpc/MEA Traces/`: `README.md`, `smoke_test_mea_env.py`, `submit_mea_array.sh` |

Do not confuse two ANN tree ids. The table's `186ce230...` is the tree of the
whole repository at the commit. The label the plan and the record give the ANN
tools, "ANN main C8 (...; tree 6e01860c)", is the tree of the folder
`hpc/MEA Traces` alone, `6e01860cdc6a113ca467d59327653f9278742681` (the
2026-09-30 state, `37bf9f8`, is `0969d9f3`; `[SANDBOX 2026-10-01]`,
`git rev-parse <commit>:"hpc/MEA Traces"`). The second is a subtree of the
first: two different objects, so two names.

The three `davinci_upload/` files are the ones the ANN commit changes. Their
sha256 (first 16 hex; the second hash of each is the same file without its
final newline, which an upload can drop): `submit_mea_array.sh`
`1575bb80f87c0806` / `48c4b6633f15efb6`; `smoke_test_mea_env.py`
`949fc0d98b181900` / `03c04866e862d77e`; `README.md` `10ed10ed0cfbabcb` /
`d1783c959c9cff3a`. `[SANDBOX 2026-10-01]`

## P3. Push the two commits (laptop, Git Bash)

`<bundle>` below is the folder the zip was extracted to -- the one holding
`README.txt`, `sbix/`, `ann/` -- written Git-Bash style
(`/c/Users/...`, quoted if the path has a space).

Before anything, two checks (each prints nothing alarming if it is right):

```
cd <bundle>/davinci_upload && sha256sum -c SHA256SUMS      # three lines ending OK
cd /c/Users/leona/Sbi-extractor && git fetch origin && git status -sb && git log --oneline -1
    # want: "## feat/real-arm-parity...origin/feat/real-arm-parity" with no ahead/behind,
    #       and "63d2574 cohort_manifest: refuse version-2 archives in the archive check (Stage C, C2c)"
```

Then the pushes -- one script per repository, each followed by its own
`git push`:

```
cd /c/Users/leona/Sbi-extractor && bash <bundle>/sbix/push_sbix_c8.sh
git push
cd /c/Users/leona/Astro-Neuron-Network && bash <bundle>/ann/push_ann_c8.sh
git push
```

What each script does: refuses unless the clone is a clone of its repository
(not under `repos/`, origin URL matching), on the right branch, with no edits
to tracked files, and at the base commit above == `origin/<branch>`; applies
the patch to the INDEX only (`git apply --cached`), so what is committed
depends on the stored blobs alone, never on CRLF or LF in the working files;
commits with the message in `0001.msg`; rewrites the working files in scope
from the new HEAD; compares the new tree id with the tested one. It never
pushes. A success prints `same   tree of HEAD (fa4e4eb5a52e)` (ANN:
`186ce230c675`) and ends with `PASS: HEAD reproduces the tested tree. Now:  git
push`. It is one-shot: a second run says `ALREADY APPLIED`.

If `git push` complains that there is no upstream branch:
`git push -u origin feat/real-arm-parity` (ANN: `git push -u origin main`).

**For D-034, in the Sbi-extractor clone, before the script runs:** the script
refuses on any tracked edit, and a staged new file counts as one. If the
raster work is in that clone, plain untracked files (a new `raster/` folder
that has not been `git add`ed) do not matter; staged or modified tracked
files do -- `git stash` them, run the script, push, then `git stash pop`
(`.gitignore` and `EXTRACTOR_USAGE.md` are the two files that can then need a
hand merge, see P4). A raster commit already made on `feat/real-arm-parity`
puts HEAD off `63d2574` and the script refuses, changing nothing: that is the
"raster first" case of P7.

## P4. After the pushes -- for the raster chat

In the Sbi-extractor clone, right after the push (the second line is the
check; the third holds only until the raster commit lands):

```
git log --oneline -2                                   # the laptop's C8 commit id on top of 63d2574
git rev-parse 'HEAD^{tree}'                            # must print fa4e4eb5a52ea040dee5d1f71143137bf6d6486f
git rev-parse 'origin/feat/real-arm-parity^{tree}'     # the same, until the raster commit lands
```

- **The raster commit is cut against that tree, not against `63d2574`.** A
  script that guards its base should compare `git rev-parse 'HEAD^{tree}'` with
  `fa4e4eb5a52ea040dee5d1f71143137bf6d6486f` (the way `push_sbix_c8.sh` does
  with `TREE_BASE`), not a commit id: the laptop's C8 commit id is not
  `87a225f`.
- **No overlap with a commit that only adds `raster/`.** C8 touches exactly
  `.gitignore`, `EXTRACTOR_USAGE.md` and the new `sim_reextract/`.
- **A raster commit that also edits `.gitignore` or `EXTRACTOR_USAGE.md`**
  must start from the post-C8 files: `.gitignore` gains 5 lines at the end,
  so another appended block there is a trivial hand merge; `EXTRACTOR_USAGE.md`
  has C8's row **v8.8** at the top of its changelog and C8's **sec. 14** at
  the end, so the next free numbers are **v8.9** and **sec. 15**. The project's
  copy of `EXTRACTOR_USAGE.md` already carries C8's v8.8 and sec. 14.
- **ANN is not involved** in the raster commit.

## P5. davinci: pull, upload, test, plan, run, gate

**[2026-10-02: for the run, superseded by P10 -- the same steps, re-checked
against today's heads (Sbi-extractor `0a560ca`, ANN `04977e0`) and given as
one-line gated commands. What follows stays as the record of v1.]**

Login node, after the pushes. `[SANDBOX 2026-10-01]` for every command and
expected line below (the README's, re-read this turn); nothing has run on
davinci.

```
module load proxy
cd ~/repos/Sbi-extractor && git pull --ff-only && git log --oneline -1
    # the C8 commit on top of 63d2574 (a later commit on top of it, if the raster commit has landed; ls sim_reextract/ then holds six files)
```

Upload the three files of `<bundle>/davinci_upload/` into
`~/ANN/MEA_analysis/` (they replace the 2026-09-30 ones; that folder is not a
git clone, D-022), then:

```
cd ~/ANN/MEA_analysis
sed -i 's/\r$//' submit_mea_array.sh smoke_test_mea_env.py README.md
sed -i -e '$a\' submit_mea_array.sh smoke_test_mea_env.py README.md     # restores a final newline an upload may drop
chmod +x submit_mea_array.sh
sha256sum submit_mea_array.sh smoke_test_mea_env.py README.md | cut -c1-16,65-
    # 1575bb80f87c0806  submit_mea_array.sh   (48c4b6633f15efb6 = final newline missing: rerun the sed -e '$a\' line)
    # 949fc0d98b181900  smoke_test_mea_env.py (03c04866e862d77e = same)
    # 10ed10ed0cfbabcb  README.md             (d1783c959c9cff3a = same)
conda activate sbi_export && python3 smoke_test_mea_env.py | tail -1
    # ALL 14 CHECKS PASSED
```

**[2026-10-02: `smoke_test_mea_env.py` also copies `launch_mea_array.sh` and
`launch_mea_per_campaign.sh` from beside it (`:121-123` `[CODE ANN@04977e0]`);
both came with the 2026-09-30 copy (D-022) and are unchanged at `04977e0`.
P10's archive carries them as well, so the test finds its neighbours.]**

The plan refuses the 2026-09-30 `submit_mea_array.sh` by its hash: that job
cannot activate `sbi_export` on davinci (no conda on a compute node's `PATH`,
no module) and writes no `mea_env.json`.

```
cd ~/repos/Sbi-extractor/sim_reextract && conda activate sbi_export
ANN_TOOLS=~/ANN/MEA_analysis python3 smoke_test_sim_reextract.py
    # ALL 23 CHECKS PASSED  (2-3 minutes; submits nothing -- fake qsub)

DRYRUN=1 bash launch_sim_reextract.sh plan
    # freezes the cohort manifest WITH its .sha256 into ../artifacts/cohort_manifest/, writes plan.json + tasks.tsv here.
    # Expect: n_side 3; 253 tasks to run; EXCLUDED: v7/sweep_cpu_task0009 (no manifest.json) and v9's 46 empty sweep folders; one contract.
    # A contract refusal names the field; if the value is expected: PLAN_ARGS=--allow-mixed-contract.

bash launch_sim_reextract.sh test
    # ONE task (line 0 of tasks.tsv; TEST_INDEX=k for another) as a plain job, 6 h walltime.
    # ~/c8_test.o<jobid> must show "[mea-array] env activated: .../envs/sbi_export/bin/python3",
    # "[mea] N topos, a/b iters processed in X s", and NOT the WARNING "putting its bin/ first on PATH".
    # WALLTIME for the array = X s / b iterations x the plan's largest task (printed by the plan) x 2, as hh:mm:ss.

WALLTIME=hh:mm:ss RESUME=1 bash launch_sim_reextract.sh array
    # RESUME=1 keeps the test task's complete output. The array (~/c8_mea.o<jobid>.<index>) and, behind it, the gate job.

grep -h 'wrote\|REFUSED\|FAIL\|WARN' out/sim_reextract_gate.log
    # PASS: "wrote .../Outputs_v2/REEXTRACTION_RECORD.json  sha256 <16 hex>"
    # REFUSED: every bad task is named. Walltime kills leave no mea_manifest.json: run RESUME=1 again (only those tasks run), then
    #   bash launch_sim_reextract.sh gate
```

**[corrected 2026-10-02: the test job writes its WARNING lines to stderr,
which under `#PBS -k eo` without `-j oe` is `~/c8_test.e<jobid>`, not the
`.o` file -- read both (P10, S6) `[CODE ANN@04977e0, submit_mea_array.sh:4,
170-182]`. And read the gate's log whole, `cat out/sim_reextract_gate.log`: a
refusal's reasons are on the indented `[gate]      - ...` lines under each
`FAIL`, which the grep above does not print; the log ends with
`[job] sim_reextract_gate_exit=<rc>` on both outcomes `[CODE
Sbi-extractor@0a560ca, sim_reextract/run_sim_reextract_gate.pbs:71-74,
sim_reextract_gate.py:266-281]`.]**

`afterok` is ordering only on davinci (D-006): the gate job runs when the
array has ended, however it ended, so **the gate's log is the success
signal**, not the array's exit codes.

## P6. What to paste back into the chat

The plan's output (`DRYRUN=1 ... plan`), the test task's log
(`~/c8_test.o<jobid>`), and the gate's log
(`out/sim_reextract_gate.log`). From them Claude checks: `n_side 3` and 253
tasks, the exclusions named and nothing else dropped, one contract, the
environment activated (not a `PATH` prepend), the walltime arithmetic, and
the record's sha256. The record is the end of Stage C; the next steps are in
P8.
**[2026-10-02: in P10's terms, the output of S1, S4 and S6, then the whole
gate log.]**

## P7. When something is not as planned

| what you see | what it means | what to do |
|---|---|---|
| `REFUSED: HEAD is <id>; this patch is cut against 63d2574` (or `origin/... is <id>, not 63d2574`) | a commit is on the branch, locally or on origin, that the patch was not cut on: the "raster first" case | nothing was changed; send `git log --oneline -5` and the list of files that commit touches -- the C8 patch is re-cut on top of it (Claude's offer of 2026-10-01) |
| `REFUSED: tracked files have local edits` | uncommitted edits, or staged new files, in the clone | `git stash`, rerun, push, `git stash pop` |
| `REFUSED: ... not a clone of ...` / `... is under repos/` / `origin is ...` | wrong folder (the stale clones under `repos/` are never the target, D-018, D-019) | `cd` into `C:\Users\leona\Sbi-extractor` or `...\Astro-Neuron-Network` |
| `DIFF tree of HEAD: got ..., want ...` and "The commit does NOT reproduce the tested tree" | the commit is not byte-identical to the tested one; the lines above name which file differs | **do not push**; undo with `git reset --hard 63d2574` (ANN: `37bf9f8`) and send the output |
| `ALREADY APPLIED ... It is on origin too; nothing to do.` | this tree is already pushed | nothing |
| `ALREADY APPLIED ... Not on origin yet: run git push` | the commit is made and not pushed -- or origin is ahead of this clone (the raster commit landed from elsewhere), in which case the script's advice ("run git push") is wrong: the next step is `git pull --ff-only`, and `git push` would be rejected | compare `git status -sb`; push or pull accordingly |
| `ALREADY APPLIED, but STALE WORKING FILES` | HEAD is right, the working files in scope are not | the line it prints: `git checkout HEAD -- <scope>`, then rerun |
| on davinci, the plan stops with "the tools in ... are no known state of Astro-Neuron-Network's hpc/MEA Traces", or "... older than this plan needs" | the plan fingerprints the seven ANN tool files by whole-file sha256: `submit_mea_array.sh` is still the 2026-09-30 one, or the upload changed a file (CRLF, a lost final newline) | redo the upload and the two `sed` lines of P5, and compare the hashes with P2 |
| the test log shows the `PATH` WARNING and no `env activated` | the job did not activate `sbi_export` | do not start the array; paste the log |

## P8. Not in C8, and what comes next

Not in this bundle (D-032, D-033): the export over `Outputs_v2`
(`launch_sweep_exports.sh`, `MEA_ROOT=Outputs_v2`) and the `label_axes.json`
it needs (status row A3: the one open call, P0 item 5), the manifest wiring of
the exports (C1-C3), the C0 flip, the DSN retrain (D-031: later, with
optimization), the real-arm export, and the joint DSN+NPE training. After the
record: the Giulia project (I-001), which waits for the answers to its eleven
questions (Q1-Q11 of `claude/HANDOFF_2026-10-01_giulia.md` v1.1).

## P9. Checked, and not checked

Checked `[SANDBOX 2026-10-01]`:

- both push scripts on fresh clones against bare "origin" repositories, with a
  default checkout and with `core.autocrlf=true`: PASS, the new tree ids equal
  the tested ones (P2), a second run says `ALREADY APPLIED`, and after `git
  push` the origin's trees equal the tested ones;
- "a commit lands on origin first": the Sbi-extractor script REFUSES and
  changes nothing;
- "a commit lands after C8": a fast-forward push on top, with no force; a clone
  that already has the later HEAD is refused by the script and left untouched;
- the zip's sha256 and `davinci_upload/SHA256SUMS`, and the trees and diffstat
  of P2, re-read this turn from the sandbox repositories.

Not checked:

- nothing ran on the real laptop clones or on davinci. The sandbox is Linux:
  Git for Windows' own behaviour (its version, Git Bash path handling, its
  line-ending configuration beyond the `core.autocrlf=true` rehearsal above)
  was not exercised `[NOT VERIFIED]`. That a job on a compute node activates
  `sbi_export` is what the `test` step is for;
- the raster commit: its base commit and files were not read, and its branch is
  Claude's statement of 2026-10-01 13:10, not something read from the clone;
- `~/repos/Sbi-extractor` on davinci is read as clean at `63d2574`
  `[CLUSTER 2026-10-01]`; whether anything changed since is not known.
  **[2026-10-02: at `0a560ca` -- the G1 run's pull printed `GO 1: both fixes
  are on davinci (2c0a06d / 0a560ca)` `[CLUSTER 2026-10-02]`.]**

## P10. The run, re-issued 2026-10-02 (D-057): one-line gated steps

**What changed since P5.** The C8 commits are on GitHub (2026-10-01 17:38,
D-034); the raster commit and the two Giulia commits followed on
`feat/real-arm-parity`, head `0a560ca`, and none touches `sim_reextract/`,
which is byte-identical to C8's `[GITHUB 2026-10-02; git diff 481bd48 0a560ca
-- sim_reextract/ is empty, and the Giulia commit touched none of C8's files,
D-053]`. ANN `main` is still `04977e0`. The steps are P5's, in the form G1
used after its second run went past a STOP line (the Giulia plan, sec. 5, G1):
one line each; each line does nothing unless the one before printed its GO
line; every path absolute; each line activates its own environment, so a new
shell between steps is fine.

| step | does | prints when it may go on |
|---|---|---|
| S1 | `module load proxy`; pulls Sbi-extractor; then only reports: the head; the seven fingerprinted ANN tools (`sim_reextract_plan.fingerprint_tools`) and the three other files the environment test needs; the library; the DUP15HD manifest's digest (checked against its `.sha256` by `read_manifest`); the frozen copy; `Outputs_v2` | `S1: the ANN tools are at the C8 state -- skip S2, go to S3`, or `S1: UPLOAD NEEDED -- do S2 first`; `S1: STOP ...` when the seven files are in no known state |
| S2 | only after S1's UPLOAD NEEDED, with the archive uploaded to `~`: extracts it into `ANN/MEA_analysis/` and checks the five hashes | `GO 2: the ANN files are the C8 state` |
| S3 | the ANN environment test and the C8 suite, each into its log under `sim_reextract/out/` | `ALL 14 CHECKS PASSED`; `ALL 23 CHECKS PASSED` |
| S4 | only if both suites passed: `DRYRUN=1 bash launch_sim_reextract.sh plan` into `out/plan_c8.log`, shown without the per-task exclusion lines | `[launch] froze ... (sha256 c925c5106137fe82)`; `[plan] EXTRA_ARGS      : --n_side 3 --pitch 60.0 --edge 25.0 --fs 10110.09`; the C8 tools label; the per-campaign table; `[plan] tasks to run : <n> ...`; `[plan] EXCLUDED <k> task(s)`; `[launch] plan only: <n> task(s) ...` |
| S5 | only if the plan log names the manifest `c925c5106137fe82`, the `n_side 3` arguments, the C8 tools and at least one task: `bash launch_sim_reextract.sh test` | `[launch] test task: line 0 of tasks.tsv -> ...`, `  job   : <id>`; otherwise `STOP: not submitted ...` |
| S6 | once `qstat -u $USER` no longer lists the test job: the lines that matter from `~/c8_test.o<id>` AND `~/c8_test.e<id>`, and the plan's counts | `[mea-array] env activated: .../envs/sbi_export/bin/python3` and no WARNING; `[mea] N topos, a/b iters processed in X s`; `[plan] tasks to run ... largest task M iterations` |

**The archive** `c8_ann_upload_20261002.tar.gz` (sent in the Giulia chat,
2026-10-02): 26,084 bytes, sha256
`b8adf5966ce81747d313788d92045c66d5cf13d1e9d0bb01e8d4a96fcfa8b886`, built from
the ANN `04977e0` blobs of `hpc/MEA Traces/`; files and sha256 (first 16 hex):
`submit_mea_array.sh` `1575bb80f87c0806`, `smoke_test_mea_env.py`
`949fc0d98b181900`, `README.md` `10ed10ed0cfbabcb` (the three of P2), and
`launch_mea_array.sh` `41be351b08f3a6e9`, `launch_mea_per_campaign.sh`
`fe046e1c12f9b7f1` (unchanged since `37bf9f8`) `[SANDBOX 2026-10-02]`. It
replaces P5's upload of three loose files: one binary transfer, so no CR to
strip and no final newline to restore.

**[v1.3, 2026-10-02 16:53: S2 by git, for when the archive cannot be uploaded.]**
The same five files come from GitHub (ANN `main` `04977e0`). The clone is
throwaway -- made in `~/ann_c8_tmp` and deleted by the same line -- so the tools
still live and run in the non-git `ANN/MEA_analysis/` (D-022). It prints the
commit it copied from, the five hashes, and `GO 2` or `STOP`; a failed clone
copies nothing:

```
S2g: module load proxy; A=/davinci-1/home/ldellamea/ANN/MEA_analysis; T=/davinci-1/home/ldellamea/ann_c8_tmp; rm -rf "$T"; if git -c core.autocrlf=false clone -q --depth 1 https://github.com/Leonardodm00/Astro-Neuron-Network.git "$T"; then echo "ANN on GitHub: $(git -C "$T" log -1 --format='%h %s')"; for f in submit_mea_array.sh smoke_test_mea_env.py README.md launch_mea_array.sh launch_mea_per_campaign.sh; do cp "$T/hpc/MEA Traces/$f" "$A/$f"; done; chmod +x "$A/submit_mea_array.sh" "$A/launch_mea_array.sh" "$A/launch_mea_per_campaign.sh"; fi; rm -rf "$T"; cd "$A" && sha256sum submit_mea_array.sh smoke_test_mea_env.py README.md launch_mea_array.sh launch_mea_per_campaign.sh | cut -c1-16,65- && [ "$(sha256sum submit_mea_array.sh smoke_test_mea_env.py README.md launch_mea_array.sh launch_mea_per_campaign.sh | cut -c1-16 | tr '\n' ' ')" = "1575bb80f87c0806 949fc0d98b181900 10ed10ed0cfbabcb 41be351b08f3a6e9 fe046e1c12f9b7f1 " ] && echo "GO 2: the ANN files are the C8 state" || echo "STOP: S2 did not complete -- the clone failed, or the hashes above are not the C8 ones"
```

**[v1.4, 2026-10-02 17:10: S3 on davinci gave `FAILED 2 of 23` -- E1 and E3, a test-only
failure (changelog v1.4).]** S4b replaces S4 for this case. It prints whether `sbi_export`
resolves with a temporary `HOME` and with the real one, which checks failed, and how many of
them ran in `base`; it makes the plan only for `notfound` / `found` / `E1 E3` / `2` with the
environment suite passed, else `STOP`:

```
S4b: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && rm -f out/plan_c8.log && T=$(mktemp -d) && r1=$(HOME="$T" bash --noprofile --norc -c 'eval "$("$0" shell.bash hook 2>/dev/null)" >/dev/null 2>&1; conda activate sbi_export >/dev/null 2>&1 && echo found || echo notfound' "$CONDA_EXE"); r2=$(bash --noprofile --norc -c 'eval "$("$0" shell.bash hook 2>/dev/null)" >/dev/null 2>&1; conda activate sbi_export >/dev/null 2>&1 && echo found || echo notfound' "$CONDA_EXE"); rmdir "$T"; f=$(grep -oE '^ *\[FAIL\] +[A-Z][0-9]+' out/smoke_c8.log | awk '{print $2}' | tr '\n' ' '); nb=$(grep -cE "^ *\[FAIL\] +E[13] .*'env_name': 'base'" out/smoke_c8.log); echo "sbi_export with a temporary HOME: $r1; with your HOME: $r2; failed checks: ${f:-none}; of them in env base: $nb"; if [ "$r1" = notfound ] && [ "$r2" = found ] && [ "$f" = "E1 E3 " ] && [ "$nb" = 2 ] && tail -1 out/smoke_mea_env.log | grep -qx 'ALL 14 CHECKS PASSED' && tail -1 out/smoke_c8.log | grep -qx 'FAILED 2 of 23'; then DRYRUN=1 bash launch_sim_reextract.sh plan > out/plan_c8.log 2>&1; grep -v '^\[plan\]   [^ ]*/' out/plan_c8.log; else echo "STOP: not the known test-only failure -- no plan made; paste this output"; fi
```

**[v1.5, 2026-10-02 17:2x: S4b confirmed the diagnosis; the suite is fixed (changelog v1.5).]**
Laptop (Git Bash), with `c8_suite_env_fix.patch` (2,438 bytes, sha256
`b715df6600a2c33ca2b7d30e5fd06c7feda0bda4fdd7339d962e4e701cb7a8ca`) in Downloads: pull, check
HEAD is `0a560ca`, apply to the index only, commit, refresh the working file, check the tree is
`c4b9c58c...`, push. davinci: S3c pulls, checks the same tree, re-runs the C8 suite -> `ALL 23
CHECKS PASSED`; then S4 as given (or with `PLAN_ARGS=--allow-mixed-contract`, if the open call on
the mixed contract is settled that way -- log Open calls). The plan's refusal, as printed:
`simtime: 200.0 (214 tasks) | 180.0 (39 tasks, e.g. campaign_cadex_rho1300v10/sweep_cfd_task0000)`;
`log_params` and `param_bounds` differ on 11 tasks (v10).

**[v1.6, 2026-10-03 14:04: suite 23/23 on davinci; D-059 -- the plan runs over the mixed contract.]**
S4m replaces S4; S5 and S6 are unchanged: **[corrected 2026-10-03 14:20: S5 needs the override too -- S5m, below]**

```
S4m: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && rm -f out/plan_c8.log && { tail -1 out/smoke_mea_env.log | grep -qx 'ALL 14 CHECKS PASSED' && tail -1 out/smoke_c8.log | grep -qx 'ALL 23 CHECKS PASSED' && DRYRUN=1 PLAN_ARGS=--allow-mixed-contract bash launch_sim_reextract.sh plan > out/plan_c8.log 2>&1; grep -v '^\[plan\]   [^ ]*/' out/plan_c8.log 2>/dev/null || echo "STOP: no plan was made -- the two suites of S3 did not both pass (out/smoke_mea_env.log, out/smoke_c8.log)"; }
```

**[v1.7, 2026-10-03 14:20: every launcher mode re-plans, so the override goes on S5 and on the array as well.]**
S5m replaces S5 (S6 is unchanged); the array line will carry `PLAN_ARGS=--allow-mixed-contract` too:

```
S5m: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && rm -f out/test_c8_submit.log && { grep -q '^\[plan\] cohort manifest : .*  sha256 c925c5106137fe82$' out/plan_c8.log && grep -qx '\[plan\] EXTRA_ARGS      : --n_side 3 --pitch 60.0 --edge 25.0 --fs 10110.09' out/plan_c8.log && grep -q '^\[plan\]  *ANN main C8 (2026-10-01' out/plan_c8.log && grep -q '^\[launch\] plan only: [1-9][0-9]* task(s) in tasks.tsv; nothing submitted$' out/plan_c8.log && PLAN_ARGS=--allow-mixed-contract bash launch_sim_reextract.sh test > out/test_c8_submit.log 2>&1; grep -E '^\[launch\] test task|^  job|^ABORT|^REFUSED|^watch' out/test_c8_submit.log 2>/dev/null || echo "STOP: not submitted -- the plan is not the expected one (out/plan_c8.log)"; }
```

**[v1.8, 2026-10-03 15:25: S6 passed; a v12 task is still simulating and is left out by name (D-060).]**
S6 `[CLUSTER 2026-10-03]`: job `1797833` -- `[mea-array] env activated: /davinci-1/home/ldellamea/.conda/envs/sbi_export/bin/python3`,
python 3.11.15, numpy 2.4.6, scipy 1.17.1, no WARNING (D-023 checked on a compute node); `[mea] 6 topos, 1334/1334 iters
processed in 3102.1 s`; the plan's 254 tasks, largest 11,728 iterations over 26 topologies, 47 excluded. So the test task ran at
3102.1 / 1334 = 2.3254 s per iteration -- its own mean; that the other tasks run at that rate is a transplant from one task, which the
factor 2 covers only in part `[reasoning]`. WALLTIME = 11,728 x 2.3254 x 2 = 54,545 s, rounded up to 10 min: **15:10:00**; S7x
computes it from the plan it makes, so it follows the largest task left after the exclusion. At 20 tasks at a time the array is at most
494,197 iterations (495,531 less the test task's 1,334, less the left-out task's) x 2.3254 s / 20 = at most about 16 h of wall
time, if that rate holds and 20 tasks run at once throughout `[reasoning]`.

**Why a running simulation must be left out, not run.** The plan counts a task's `iter_*.npz` at plan time
(`sim_reextract_plan.py:203-222`), so a simulation still being written is indistinguishable from a finished one with fewer
iterations. The gate re-counts the raw iterations when it runs and refuses a task whose count grew
(`sim_reextract_gate.py:21-34`), but only if the simulation wrote a new file between the array's plan and the gate; one that wrote
nothing in that window would enter the record as complete, and the record is the receipt that ends Stage C (D-033). An
`iter_*.npz` read while it is being written fails that iteration (`_failures.log`), which the gate also refuses. And once the
record exists the plan refuses `Outputs_v2` as finished (`:424`), so the task cannot be added to it later
`[CODE Sbi-extractor@0a560ca]`. **[corrected 2026-10-03 15:57: it can be added later -- the refusal is of a root that
holds `REEXTRACTION_RECORD.json`; with the record renamed aside, `RESUME=1` adds the task and the gate writes a
new record over all tasks ("Adding the left-out task later", below).]** Hence `--exclude-task CAMPAIGN/SWEEP`: the task is listed under `excluded` with the reason
`left out on the command line (--exclude-task)`, in `plan.json` (also as `excluded_by_name`) and in the record; a name matching no
task, or not of that form, is refused, so a typo cannot exclude nothing. **[corrected 2026-10-07: "The gate re-counts the raw iterations when it runs and refuses a task whose count grew ... but only if the simulation wrote a new file between the array's plan and the gate", above, is wrong. The gate never re-counts the raw folders: it compares each task's MEA output with the plan's counts (`sim_reextract_gate.py:112, 146-163` `[CODE Sbi-extractor@1b23adc]`), so it refuses a task whose raw folder grew between the plan and the start of that task's MEA job -- the job processes what it finds, more than planned -- and it does not see growth after that start. The re-run of 2026-10-06 shows the first case (v1.22 block).]** **[corrected 2026-10-07, about 13:10: "the start of that task's MEA job" is too coarse -- it is the moment the job lists each topology (v1.22 block, its corrected paragraph).]**

**The patch** `c8_exclude_task.patch` (sent in the Giulia chat, 2026-10-03): 28,182 bytes, sha256
`2475e0ed63d72e637432e5b6768f730ce2262bc67f7ced5817a8523b21384ad8`; applies to the tree `c4b9c58c0e2d610f16a3bd799b203e7c4d3d10ed`
(the laptop's `5682a6c`, davinci's head); tree after it `baeaa95fb977d53879b30a71df69547a9f4f3273` (sandbox commit `c060dd4`,
information only). Four files: `sim_reextract/sim_reextract_plan.py` (+57: the option, `exclude_named`, `excluded_by_name`),
`sim_reextract/smoke_test_sim_reextract.py` (P6: the option and its two refusals; L4: `PLAN_ARGS` carrying two flags through the
launcher; 25 checks), `sim_reextract/README.md`, `EXTRACTOR_USAGE.md` (v8.12, sec. 14; it also carries the project-only v8.11 row
and sec. 15.9 into the repo). Checked `[SANDBOX 2026-10-03]`: compiles, pure ASCII, LF only; the suite 25/25 six times, on
patches with the same code, the last on the final tree (a real conda, a user-level `sbi_export` without conda in it); two mutations -- the option made a no-op, the typo
refusal removed -- each failed the suite (P6 and L4; P6).

| step | where | does | prints when it may go on |
|---|---|---|---|
| P | davinci, any time, read-only | `qstat -u $USER`; every task with an `iter_*.npz` written in the last 6 h, with the count and the minutes since the newest | one line per task being written; the one to leave out is `campaign_cadex_rho1300v12/sweep_cfd_task<nnnn>` |
| L | laptop, Git Bash, the patch in Downloads | pull; check the tree is `c4b9c58c...`; apply to the index; commit; refresh the four working files; check the tree is `baeaa95f...`; push | `PUSHED: the tested tree baeaa95f` |
| S3x | davinci | pull; check the tree; the C8 suite | `GO: --exclude-task is on davinci (<id>)`, `ALL 25 CHECKS PASSED` |
| S7x | davinci, with `XT=<the task>` | only after S3x's 25/25: the plan with `RESUME=1` and both flags; only if the task is left out: WALLTIME from the plan, the array's dry run, the queue's walltime cap | `[plan] tasks to run : 252 (array 0-251) ...`, `[plan] tasks complete : 1 kept as done (--resume)`, `[plan] EXCLUDED 48 task(s)`, the task's `left out` line, the array line `-J 0-251%20 ... walltime=15:10:00` (if the largest task is still 11,728), the gate line, `WALLTIME=...`, `resources_max.walltime` |
| S8x | davinci, the same `XT` | only after S7x's dry run, for the same task: the array and the gate, for real | the `left out` line, `  job   : <id>[]...`, `  gate  : <id>...` |
| S9x | davinci, once `qstat` no longer lists the gate | the record's PASS line, the exit line, the record's counts and the task left out | `wrote .../Outputs_v2/REEXTRACTION_RECORD.json  sha256 <16 hex>`, `[job] sim_reextract_gate_exit=0`, `record: 253 tasks, ... 48 excluded; left out by name: <the task>` |

The counts in S7x and S9x are expected from the plan of 14:20 with one task left out `[reasoning]`; if P names more than one task, the
lines take one `XT` only -- paste P's output first.

**Between S8x and S9x, run no launcher mode.** The gate job is submitted with `PLAN=<sim_reextract>/plan.json`
(`launch_sim_reextract.sh:166-172`) and reads it when it starts (`run_sim_reextract_gate.pbs:61-72`), hours later; every mode of
the launcher rewrites that file (it re-plans), so a dry run in between would hand the gate another plan `[CODE
Sbi-extractor@0a560ca]`. A walltime kill of a member is handled as before: once the gate has refused, `RESUME=1` with the same
`PLAN_ARGS` and WALLTIME re-runs only the incomplete tasks, then `bash launch_sim_reextract.sh gate`.

```
P:   cd /davinci-1/home/ldellamea/ANN/Phenomenological/Main && qstat -u $USER; echo "--- tasks with an iter_*.npz written in the last 6 h  (files | newest, minutes ago):"; find campaign_cadex_rho1300v* -name 'iter_*.npz' -mmin -360 -printf '%T@ %h\n' 2>/dev/null | awk -v now=$(date +%s) '{split($2,a,"/"); k=a[1]"/"a[2]; n[k]++; if (!(k in m) || $1 > m[k]) m[k] = $1} END {for (k in n) printf "%-58s %6d | %5.0f\n", k, n[k], (now - m[k]) / 60}' | sort; echo "--- probe done"
L:   cd /c/Users/leona/Sbi-extractor && git checkout feat/real-arm-parity && git pull --ff-only && [ "$(git rev-parse 'HEAD^{tree}')" = c4b9c58c0e2d610f16a3bd799b203e7c4d3d10ed ] && git apply --cached ~/Downloads/c8_exclude_task.patch && git commit -q -m "sim_reextract plan: --exclude-task CAMPAIGN/SWEEP leaves a named task out (C8, a simulation still being written)" -m "$(printf 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_015awffdnTEKqUyY5D7NVVRX')" && git checkout -- EXTRACTOR_USAGE.md sim_reextract/README.md sim_reextract/sim_reextract_plan.py sim_reextract/smoke_test_sim_reextract.py && [ "$(git rev-parse 'HEAD^{tree}')" = baeaa95fb977d53879b30a71df69547a9f4f3273 ] && git push && echo "PUSHED: the tested tree baeaa95f" || echo "STOP: nothing pushed, or not the tested tree -- paste this output"
S3x: module load proxy; cd /davinci-1/home/ldellamea/repos/Sbi-extractor && git pull --ff-only && if [ "$(git rev-parse 'HEAD^{tree}')" = baeaa95fb977d53879b30a71df69547a9f4f3273 ]; then echo "GO: --exclude-task is on davinci ($(git log -1 --format=%h))"; source env.sh && conda activate sbi_export && mkdir -p sim_reextract/out && rm -f sim_reextract/out/smoke_c8.log && (cd sim_reextract && ANN_TOOLS=/davinci-1/home/ldellamea/ANN/MEA_analysis python3 smoke_test_sim_reextract.py) > sim_reextract/out/smoke_c8.log 2>&1; tail -1 sim_reextract/out/smoke_c8.log; else echo "STOP: davinci is not at the tested tree (HEAD $(git log -1 --format=%h)) -- was the laptop push done?"; fi
S7x: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && XT=PUT_THE_TASK_HERE && rm -f out/plan_c8x.log out/array_c8x_dry.log out/walltime_c8x.txt && if tail -1 out/smoke_c8.log | grep -qx 'ALL 25 CHECKS PASSED' && RESUME=1 DRYRUN=1 PLAN_ARGS="--allow-mixed-contract --exclude-task $XT" bash launch_sim_reextract.sh plan > out/plan_c8x.log 2>&1 && grep -qF "[plan]   $XT: left out on the command line (--exclude-task)" out/plan_c8x.log; then W=$(python3 -c "import json, math; n = json.load(open('plan.json'))['counts']['max_iters_task']; s = math.ceil(n * 3102.1 / 1334 * 2 / 600) * 600; print('%02d:%02d:00' % (s // 3600, s % 3600 // 60))") && echo "$W" > out/walltime_c8x.txt && RESUME=1 DRYRUN=1 WALLTIME=$W PLAN_ARGS="--allow-mixed-contract --exclude-task $XT" bash launch_sim_reextract.sh array > out/array_c8x_dry.log 2>&1; grep -E '^\[plan\] (tasks|EXCLUDED)|: left out on the command line|^# (tasks|walltime)|qsub|ABORT|REFUSED|DRYRUN' out/array_c8x_dry.log; echo "WALLTIME=$W"; qstat -Qf cpu | grep -i walltime; else echo "STOP: no dry run -- the suite is not 25/25, the plan refused, or $XT is not left out (out/plan_c8x.log):"; grep -E 'REFUSED|ABORT|names no task|not CAMPAIGN|^\[plan\] tasks' out/plan_c8x.log; fi
S8x: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && XT=PUT_THE_TASK_HERE && if grep -q '^(DRYRUN' out/array_c8x_dry.log 2>/dev/null && grep -qF "[plan]   $XT: left out on the command line (--exclude-task)" out/array_c8x_dry.log && W=$(cat out/walltime_c8x.txt) && [ -n "$W" ]; then RESUME=1 WALLTIME=$W PLAN_ARGS="--allow-mixed-contract --exclude-task $XT" bash launch_sim_reextract.sh array > out/array_c8x_submit.log 2>&1; grep -E '^\[plan\] tasks|: left out on the command line|^# walltime|^  (job|gate) *:|ABORT|REFUSED' out/array_c8x_submit.log; else echo "STOP: nothing submitted -- run S7x first, with the same XT"; fi
S9x: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && if grep -q '^wrote .*/Outputs_v2/REEXTRACTION_RECORD.json  sha256 [0-9a-f]\{16\}$' out/sim_reextract_gate.log 2>/dev/null; then grep -E '^wrote|sim_reextract_gate_exit' out/sim_reextract_gate.log; python3 -c "import json; r = json.load(open('/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2/REEXTRACTION_RECORD.json')); c = r['counts']; print('record: %d tasks, %d iterations, %d excluded; left out by name: %s' % (c['tasks'], c['iterations'], c['tasks_excluded'], ', '.join(e['campaign'] + '/' + e['sweep'] for e in r['excluded'] if 'exclude-task' in e['reason'])))"; else echo "NO RECORD (yet?) -- check qstat -u \$USER; the gate log so far:"; tail -60 out/sim_reextract_gate.log 2>/dev/null; fi
```

(`PUT_THE_TASK_HERE` in S7x and S8x is replaced by the name P prints, e.g. `campaign_cadex_rho1300v12/sweep_cfd_task0003`.)

**Rehearsed** `[SANDBOX 2026-10-03]`, 18/18, from scratch each time: a bare origin at the tree `c4b9c58c`; a laptop clone with
`core.autocrlf=true` (the repo's `.gitattributes` keeps `*.py`/`*.md` LF); a davinci clone with the untracked `plan.json` and
`tasks*.tsv` of S4m; the suite's synthetic campaign tree plus a v12 with a finished `sweep_cfd_task0000` and a
`sweep_cfd_task0001` whose newest files are 4 min old (everything else 2 days); the test task run first, as on davinci; a fake
`qsub` that runs the array members, stub `module`/`conda`/`qstat`. Held: L pushes the tested tree with the two trailers and leaves
the clone clean; L run again, and L on another base, STOP with nothing pushed or committed; S3x 25/25 on the pulled tree; P names
the running task only; S7x leaves it out (4 to run, 1 done, WALLTIME 00:10:00) and submits nothing; S7x with a wrong name STOPs
showing the refusal; S8x without a good dry run STOPs; S8x submits the array and the gate, and no output is made for the task left
out; the gate (run as its job would) PASSES; S9x prints the record line and names the task; S9x before the gate prints `NO
RECORD`. Not checked: the name of the running task and whether it is the only one (P); the `cpu` queue's walltime cap (S7x prints
it); whether the gate job's 4 h and 8 cores suffice for about half a million `mea_iter_*.npz` `[NOT VERIFIED]`.

**Adding the left-out task later (v1.9).** Once P shows nothing being written for that task (and `qstat` no
longer lists its simulation), it joins the same `Outputs_v2` and one record covering every task, with no code
change: (1) rename `Outputs_v2/REEXTRACTION_RECORD.json` and its `.sha256` aside (e.g. `..._run1.json`), nothing
deleted -- the plan refuses a root while that file is there (`sim_reextract_plan.py:305-308, 424`) and so does the
gate (`sim_reextract_gate.py:204-207`); (2) the S7x/S8x lines with `RESUME=1` and `PLAN_ARGS=--allow-mixed-contract`
alone: `--resume` keeps as done every task whose `mea_manifest.json` reports every iteration done at the raw count
it has now (`:327-339`), so only the finished task runs (one task: a plain job), with any new campaign folder that
matches `campaign_cadex_rho1300v*` under `Main/` (I-002); (3) the gate re-checks every task, the recorded ones too
(about half a million files again), and writes a new record over all of them. Conditions: the same ANN tools, the
same template library and the same frozen cohort manifest (the plan fingerprints them, the gate re-reads them), and
the same `sbi_export` -- the gate requires python, numpy, scipy, env prefix, interpreter and tool hashes to agree
across every task's `mea_env.json` (`:251-262`), so an update of the environment or of the tools in between makes
it refuse, and then the additions need a root of their own. Not recorded by this route: that the record was made in
two passes, and the first record's sha256 (that file stays beside the new one). A small `--extend` option that
writes the previous record's digest into the new one would close that; not built. Rehearsed `[SANDBOX 2026-10-03]`
on the v1.8 replica after S9x, 4 checks: with the record in place the plan refuses; with it renamed, 5 tasks kept,
only the finished one run, as a plain job; the gate passes and the new record holds 6 tasks and no `--exclude-task`
entry, the first record beside it; with one task's `mea_env.json` naming another numpy, the gate refuses and writes
nothing. The exact lines are given when the task has finished.

**[v1.10, 2026-10-03 16:08: S7x passed; S8x held for the replayed seeds (D-061).]** **[2026-10-05: superseded by the v1.11 block below -- the gate is built; S7y, S8y, S9y replace S7x, S8x, S9x, and C1 is optional.]**
S7x `[CLUSTER 2026-10-03 16:08]`: `[plan] tasks to run : 252 (array 0-251), 491893 iterations; largest task 11728
iterations, 26 topologies`; `tasks complete : 1 kept as done (--resume)`; `EXCLUDED 48`, with
`campaign_cadex_rho1300v12/sweep_cfd_task0010: left out on the command line (--exclude-task)` (the user's name for the
running task, 16:06; P's output was not pasted); `# walltime : 15:10:00 ncpus 48 queue cpu concurrency 20`; the array
line `-J 0-251%20` and the gate line; `resources_max.walltime = 300:00:00`, `resources_default.walltime = 24:00:00`. The
left-out task held 495,531 - 1,334 - 491,893 = 2,304 iterations at plan time. **Do not run S8x yet**: the user (16:08)
does not want tasks that replay another task's seed re-detected (D-061), and the plan reads no seed. Next: C1, then a
replay gate in the plan (default on, each dropped task named with the task it replays), then S7x and S8x again.

C1 -- the seed census, read-only, on the `plan.json` S7x wrote (run it before any other launcher mode, which would
rewrite that file). It reports every `job_args.json` key whose name contains "seed" over the plan's tasks and
exclusions, groups the tasks by those values, and for each group prints its first task (in plan order) and, for each
other member, one of: `REPLAY: same theta, same <n> iteration files`; `REPLAY, shorter` (its files are a subset of the
first's); `REPLAY, longer` (it holds all of the first's); `DIFFERENT theta at <file>` (same seed, not the same
simulations); `NO common iteration file`; `UNREADABLE` (a file being written, say). Theta is compared at the first and
the last iteration file the two tasks share. The script is passed base64-encoded so it survives the paste as one line;
its source is `census.py` (sha256 `2b9a9feb37aff098...`, kept in the session's scratchpad), output to
`out/seed_census.log`:

```
C1:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && python3 -c 'import base64; exec(base64.b64decode("aW1wb3J0IGpzb24sIG9zLCBnbG9iLCBjb2xsZWN0aW9ucywgbnVtcHkgYXMgbnAKUCA9IGpzb24ubG9hZChvcGVuKCJwbGFuLmpzb24iKSkKVCA9IGxpc3QoUFsidGFza3MiXSkgKyBsaXN0KFBbImV4Y2x1ZGVkIl0pCnJvd3MsIG5qYSwga2V5cyA9IFtdLCAwLCBjb2xsZWN0aW9ucy5Db3VudGVyKCkKZm9yIHQgaW4gVDoKICAgIGYgPSBvcy5wYXRoLmpvaW4odFsiY2FtcGFpZ25fZGlyIl0sICJqb2JfYXJncy5qc29uIikKICAgIGlmIG5vdCBvcy5wYXRoLmlzZmlsZShmKToKICAgICAgICBuamEgKz0gMQogICAgICAgIGNvbnRpbnVlCiAgICBqID0ganNvbi5sb2FkKG9wZW4oZikpCiAgICBzID0ge2s6IGpba10gZm9yIGsgaW4gc29ydGVkKGopIGlmICJzZWVkIiBpbiBrLmxvd2VyKCl9CiAgICBrZXlzLnVwZGF0ZShzLmtleXMoKSkKICAgIHJvd3MuYXBwZW5kKCh0WyJjYW1wYWlnbiJdICsgIi8iICsgdFsic3dlZXAiXSwgdFsiY2FtcGFpZ25fZGlyIl0sIHQuZ2V0KCJzdGF0dXMiLCAiZXhjbHVkZWQiKSwganNvbi5kdW1wcyhzLCBzb3J0X2tleXM9VHJ1ZSkpKQpHID0gY29sbGVjdGlvbnMuZGVmYXVsdGRpY3QobGlzdCkKZm9yIHIgaW4gcm93czoKICAgIEdbclszXV0uYXBwZW5kKHIpCmR1cCA9IHtrOiB2IGZvciBrLCB2IGluIEcuaXRlbXMoKSBpZiBsZW4odikgPiAxIGFuZCBrICE9ICJ7fSJ9CnByaW50KCJbc2VlZF0gdGFza3Mgd2l0aCBqb2JfYXJncy5qc29uOiAlZCAod2l0aG91dDogJWQpOyBzZWVkIGtleXM6ICVzIiAlIChsZW4ocm93cyksIG5qYSwgZGljdChrZXlzKSkpCnByaW50KCJbc2VlZF0gdGFza3Mgd2l0aCBubyBzZWVkIGtleTogJWQ7IGRpc3RpbmN0IHNlZWRzOiAlZDsgc2VlZHMgc2hhcmVkIGJ5IG1vcmUgdGhhbiBvbmUgdGFzazogJWQsIGNvdmVyaW5nICVkIHRhc2tzIiAlIChsZW4oRy5nZXQoInt9IiwgW10pKSwgbGVuKFtrIGZvciBrIGluIEcgaWYgayAhPSAie30iXSksIGxlbihkdXApLCBzdW0obGVuKHYpIGZvciB2IGluIGR1cC52YWx1ZXMoKSkpKQpDID0ge30KZGVmIGZpbGVzKGQpOgogICAgaWYgZCBub3QgaW4gQzoKICAgICAgICBDW2RdID0ge29zLnBhdGguYmFzZW5hbWUodGQpOiBzb3J0ZWQob3MucGF0aC5iYXNlbmFtZSh4KSBmb3IgeCBpbiBnbG9iLmdsb2Iob3MucGF0aC5qb2luKHRkLCAiaXRlcl8qLm5weiIpKSkgZm9yIHRkIGluIHNvcnRlZChnbG9iLmdsb2Iob3MucGF0aC5qb2luKGQsICJ0b3BvXyoiKSkpfQogICAgcmV0dXJuIENbZF0KZGVmIHRoKGQsIHQsIGkpOgogICAgcmV0dXJuIG5wLmxvYWQob3MucGF0aC5qb2luKGQsIHQsIGkpKVsidGhldGEiXQpkZWYgcmVsKGEsIGIpOgogICAgZmEsIGZiID0gZmlsZXMoYSksIGZpbGVzKGIpCiAgICBuYSwgbmIgPSBzdW0obWFwKGxlbiwgZmEudmFsdWVzKCkpKSwgc3VtKG1hcChsZW4sIGZiLnZhbHVlcygpKSkKICAgIGNvbW1vbiA9IFsodCwgaSkgZm9yIHQgaW4gZmEgaWYgdCBpbiBmYiBmb3IgaSBpbiBmYVt0XSBpZiBpIGluIHNldChmYlt0XSldCiAgICBpZiBub3QgY29tbW9uOgogICAgICAgIHJldHVybiAiTk8gY29tbW9uIGl0ZXJhdGlvbiBmaWxlICglZCB2cyAlZCkiICUgKG5iLCBuYSkKICAgIHRyeToKICAgICAgICBmb3IgKHQsIGkpIGluIChjb21tb25bMF0sIGNvbW1vblstMV0pOgogICAgICAgICAgICBpZiBub3QgbnAuYXJyYXlfZXF1YWwodGgoYSwgdCwgaSksIHRoKGIsIHQsIGkpKToKICAgICAgICAgICAgICAgIHJldHVybiAiRElGRkVSRU5UIHRoZXRhIGF0ICVzLyVzICglZCB2cyAlZCBpdGVyYXRpb25zKSIgJSAodCwgaSwgbmIsIG5hKQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgIHJldHVybiAiVU5SRUFEQUJMRSBhdCAlcy8lczogJXMiICUgKHQsIGksIHR5cGUoZSkuX19uYW1lX18pCiAgICBhYiA9IGFsbCh0IGluIGZiIGFuZCBzZXQoZmFbdF0pIDw9IHNldChmYlt0XSkgZm9yIHQgaW4gZmEpCiAgICBiYSA9IGFsbCh0IGluIGZhIGFuZCBzZXQoZmJbdF0pIDw9IHNldChmYVt0XSkgZm9yIHQgaW4gZmIpCiAgICBpZiBhYiBhbmQgYmE6CiAgICAgICAgcmV0dXJuICJSRVBMQVk6IHNhbWUgdGhldGEsIHNhbWUgJWQgaXRlcmF0aW9uIGZpbGVzIiAlIG5hCiAgICBpZiBiYToKICAgICAgICByZXR1cm4gIlJFUExBWSwgc2hvcnRlcjogc2FtZSB0aGV0YSwgaXRzICVkIGZpbGVzIGFyZSBhIHN1YnNldCBvZiB0aGUgZmlyc3QncyAlZCIgJSAobmIsIG5hKQogICAgaWYgYWI6CiAgICAgICAgcmV0dXJuICJSRVBMQVksIGxvbmdlcjogc2FtZSB0aGV0YSwgaG9sZHMgYWxsICVkIG9mIHRoZSBmaXJzdCdzIGZpbGVzLCAlZCBpbiBhbGwiICUgKG5hLCBuYikKICAgIHJldHVybiAic2FtZSB0aGV0YSBhdCB0aGUgZmlyc3QgYW5kIGxhc3QgY29tbW9uIGZpbGUsIGZpbGUgc2V0cyBkaWZmZXIgKCVkIHZzICVkKSIgJSAobmIsIG5hKQpmb3IgaywgdiBpbiBzb3J0ZWQoZHVwLml0ZW1zKCksIGtleT1sYW1iZGEga3Y6IGt2WzFdWzBdWzBdKToKICAgIHByaW50KCJbc2VlZF0gJXMiICUgaykKICAgIHByaW50KCJbc2VlZF0gICAgJS01MHMgJS04cyAlZCBpdGVyYXRpb25zICh0aGUgZ3JvdXAncyBmaXJzdCkiICUgKHZbMF1bMF0sIHZbMF1bMl0sIHN1bShtYXAobGVuLCBmaWxlcyh2WzBdWzFdKS52YWx1ZXMoKSkpKSkKICAgIGZvciByIGluIHZbMTpdOgogICAgICAgIHByaW50KCJbc2VlZF0gICAgJS01MHMgJS04cyAlcyIgJSAoclswXSwgclsyXSwgcmVsKHZbMF1bMV0sIHJbMV0pKSkKcHJpbnQoIltzZWVkXSBjZW5zdXMgZG9uZSIpCg==").decode())' 2>&1 | tee out/seed_census.log
```

Tested `[SANDBOX 2026-10-03]` on a fixture with one case each -- a replay with the same files, a shorter replay across
two campaigns, a same-seed pair with different theta, a unique seed, a task without a seed key, a truncated file in an
excluded task, an exclusion without `job_args.json` -- each classified as above; the one-line form's output is identical
to the script's. Not known until it runs: whether the rho1300 `job_args.json` carry a seed key at all, and under which
name (the hhgap ones carry `_resolved_seed_master`).

**[v1.11, 2026-10-05: the replay gate is built (D-061, D-063); S7y, S8y, S9y replace S7x, S8x, S9x.]** **[2026-10-05 about 14:10: the patch and the L2 and S3y lines of this block are superseded by the v1.12 block below (D-064); do not apply `c8_replay_gate.patch`.]**
What the gate does, in one paragraph (the reference is `EXTRACTOR_USAGE.md` v8.13, sec. 14, and
`sim_reextract/README.md`): the plan groups the runnable tasks by the seed values in their
`job_args.json` -- every key containing `seed`; ANN's sweep writes `_resolved_seed_master` and the
offsets `seed_device`, `seed_neuron`, `seed_synapse`, `seed_astro` since its first version in the repo,
2026-06-16 `[CODE ANN@04977e0 hpc/Phenomenological_finalv1/HPC_main_sweep.py:1419-1428, 1505-1506]`; the
CLI `seed_master`, `None` when the seed came from the job id, is ignored beside `_resolved_seed_master`
-- and by contract signature. Within a group the tasks are taken in the keeper order (most iterations,
then the lower campaign version, then the lower task index; D-063) and a task is DROPPED as a replay of
an earlier-kept one when its iteration files are the same as, or a subset of, that task's and the
`theta` and `params` members are byte-identical at up to 10 shared files from the first to the last
(D-063). A replay is `excluded` with the reason `replay (D-061) of CAMPAIGN/SWEEP: seed <s>; <relation>`;
the plan prints every group, `plan.json` and the record carry the report (`replays`). Kept and printed:
same seed with different theta, a task holding files the kept one lacks, a seed shared across two
contracts. REFUSED: a replay whose output folder already holds detections (the message names the
folder; move it aside, nothing is deleted), and an unreadable iteration file in a compared pair.
`--keep-replays` reports and drops nothing. The plan stays stdlib-only (`zipfile`).

**The patch** `c8_replay_gate.patch` (sent in this chat, 2026-10-05): 47,284 bytes, sha256
`8466616e118860c7dcb5c610c46a25c595ce796449bdbcf14b3a82c692340f06`; applies to the tree
`baeaa95fb977d53879b30a71df69547a9f4f3273` (`6173e92`, on GitHub since 2026-10-03 16:01); tree after it
`39643844715788ec5920ca0388ddc60ae9a7fe52` (sandbox commit `ba62f2f`, information only). Five files,
+477 -16: `sim_reextract/sim_reextract_plan.py` (the gate, `--keep-replays`), `sim_reextract_gate.py`
(the record carries `replays`), `smoke_test_sim_reextract.py` (P7; the fixture's tasks get one seed
each; 26 checks), `sim_reextract/README.md`, `EXTRACTOR_USAGE.md` (v8.13). Checked `[SANDBOX
2026-10-05]`: compiles, pure ASCII, LF only; the suite 26/26 twice (system python, no conda, the ANN
tools at `04977e0`, SBI at `2c0a06d`); five mutations of the gate -- no drop, no theta comparison, the
keeper order reversed, no subset test, the CLI seed kept in the key -- each failed P7 and nothing else.

| step | where | does | prints when it may go on |
|---|---|---|---|
| L2 | laptop, Git Bash, the patch in Downloads | pull; check the tree is `baeaa95f...`; apply to the index; commit; refresh the five working files; check the tree is `39643844...`; push | `PUSHED: the tested tree 39643844` |
| S3y | davinci | pull; check the tree; the C8 suite | `GO: the replay gate is on davinci (<id>)`, `ALL 26 CHECKS PASSED` |
| P | davinci, read-only | as in the v1.8 block: what is being written now | the left-out `v12/sweep_cfd_task0010` listed, or nothing |
| S7y | davinci, `XT=` the task to leave out, or empty | only after S3y's 26/26: the plan with `RESUME=1`, the gate and the flags; the replay lines; WALLTIME from the plan; the array's dry run; the queue's walltime cap; saves WALLTIME and `PLAN_ARGS` for S8y | `[plan] replays (D-061) : ...`, every `DROPPED` / `kept` line, `[plan] replays dropped : <n>`, `[plan] tasks to run ...`, `-J 0-<n-1>%20`, `WALLTIME=... PLAN_ARGS=...` |
| S8y | davinci | only after S7y's dry run: the array and the gate, with S7y's saved WALLTIME and `PLAN_ARGS` | `  job   : <id>[]...`, `  gate  : <id>...` |
| S9y | davinci, once `qstat` no longer lists the gate | the record's PASS line, the exit line, the counts, the replays and the task left out | `wrote .../Outputs_v2/REEXTRACTION_RECORD.json  sha256 <16 hex>`, `[job] sim_reextract_gate_exit=0`, `record: <n> tasks, ... excluded (<r> replays); left out by name: ...` |

The user pastes S7y's output before S8y: the replay lines are the first look at the real seed
structure. If the test task (v1's line 0 of the 10-03 plan) turns out to be a shorter copy of
another task, S7y stops on the refusal and names its folder in `Outputs_v2`: move it aside (e.g.
`mv <folder> <folder>_replay_moved`) and run S7y again. Between S8y and S9y run no launcher mode.

```
L2: cd /c/Users/leona/Sbi-extractor && git checkout feat/real-arm-parity && git pull --ff-only && [ "$(git rev-parse 'HEAD^{tree}')" = baeaa95fb977d53879b30a71df69547a9f4f3273 ] && git apply --cached ~/Downloads/c8_replay_gate.patch && git commit -q -m "sim_reextract plan: a replay gate -- a task that replays another task's seed and simulations is not re-detected (C8, D-061)" -m "$(printf 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01RtCUZWW1sF31UDbqxk9r8Z')" && git checkout -- EXTRACTOR_USAGE.md sim_reextract/README.md sim_reextract/sim_reextract_gate.py sim_reextract/sim_reextract_plan.py sim_reextract/smoke_test_sim_reextract.py && [ "$(git rev-parse 'HEAD^{tree}')" = 39643844715788ec5920ca0388ddc60ae9a7fe52 ] && git push && echo "PUSHED: the tested tree 39643844" || echo "STOP: nothing pushed, or not the tested tree -- paste this output"
S3y: module load proxy; cd /davinci-1/home/ldellamea/repos/Sbi-extractor && git pull --ff-only && if [ "$(git rev-parse 'HEAD^{tree}')" = 39643844715788ec5920ca0388ddc60ae9a7fe52 ]; then echo "GO: the replay gate is on davinci ($(git log -1 --format=%h))"; source env.sh && conda activate sbi_export && mkdir -p sim_reextract/out && rm -f sim_reextract/out/smoke_c8.log && (cd sim_reextract && ANN_TOOLS=/davinci-1/home/ldellamea/ANN/MEA_analysis python3 smoke_test_sim_reextract.py) > sim_reextract/out/smoke_c8.log 2>&1; tail -1 sim_reextract/out/smoke_c8.log; else echo "STOP: davinci is not at the tested tree (HEAD $(git log -1 --format=%h)) -- was the laptop push done?"; fi
S7y: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && XT=campaign_cadex_rho1300v12/sweep_cfd_task0010 && PA="--allow-mixed-contract${XT:+ --exclude-task $XT}" && rm -f out/plan_c8y.log out/array_c8y_dry.log out/walltime_c8y.txt out/plan_args_c8y.txt && if tail -1 out/smoke_c8.log | grep -qx 'ALL 26 CHECKS PASSED' && RESUME=1 DRYRUN=1 PLAN_ARGS="$PA" bash launch_sim_reextract.sh plan > out/plan_c8y.log 2>&1 && { [ -z "$XT" ] || grep -qF "[plan]   $XT: left out on the command line (--exclude-task)" out/plan_c8y.log; }; then grep -E '^\[plan\] (replays|tasks|EXCLUDED)|^\[plan\]   seed |^\[plan\]     (DROPPED|kept) ' out/plan_c8y.log; W=$(python3 -c "import json, math; n = json.load(open('plan.json'))['counts']['max_iters_task']; s = math.ceil(n * 3102.1 / 1334 * 2 / 600) * 600; print('%02d:%02d:00' % (s // 3600, s % 3600 // 60))") && echo "$W" > out/walltime_c8y.txt && echo "$PA" > out/plan_args_c8y.txt && RESUME=1 DRYRUN=1 WALLTIME=$W PLAN_ARGS="$PA" bash launch_sim_reextract.sh array > out/array_c8y_dry.log 2>&1; grep -E '^# (tasks|walltime)|qsub|ABORT|REFUSED|DRYRUN' out/array_c8y_dry.log; echo "WALLTIME=$W  PLAN_ARGS=$PA"; qstat -Qf cpu | grep -i walltime; else echo "STOP: no dry run -- the suite is not 26/26, the plan refused, or the task is not left out (out/plan_c8y.log):"; grep -E 'REFUSED|ABORT|names no task|not CAMPAIGN|already have detections|cannot read|^\[plan\] tasks' out/plan_c8y.log; fi
S8y: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && if grep -q '^(DRYRUN' out/array_c8y_dry.log 2>/dev/null && W=$(cat out/walltime_c8y.txt) && PA=$(cat out/plan_args_c8y.txt) && [ -n "$W" ] && [ -n "$PA" ]; then RESUME=1 WALLTIME=$W PLAN_ARGS="$PA" bash launch_sim_reextract.sh array > out/array_c8y_submit.log 2>&1; grep -E '^\[plan\] (tasks|replays dropped)|: left out on the command line|^# walltime|^  (job|gate) *:|ABORT|REFUSED' out/array_c8y_submit.log; else echo "STOP: nothing submitted -- run S7y first"; fi
S9y: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && if grep -q '^wrote .*/Outputs_v2/REEXTRACTION_RECORD.json  sha256 [0-9a-f]\{16\}$' out/sim_reextract_gate.log 2>/dev/null; then grep -E '^wrote|sim_reextract_gate_exit' out/sim_reextract_gate.log; python3 -c "import json; r = json.load(open('/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2/REEXTRACTION_RECORD.json')); c = r['counts']; ex = r['excluded']; print('record: %d tasks, %d iterations, %d excluded (%d replays); left out by name: %s' % (c['tasks'], c['iterations'], c['tasks_excluded'], sum(1 for e in ex if e.get('replay_of')), ', '.join(e['campaign'] + '/' + e['sweep'] for e in ex if 'exclude-task' in e['reason']) or 'none'))"; else echo "NO RECORD (yet?) -- check qstat -u \$USER; the gate log so far:"; tail -60 out/sim_reextract_gate.log 2>/dev/null; fi
```

(P is the v1.8 block's line, unchanged. `XT=campaign_cadex_rho1300v12/sweep_cfd_task0010` in S7y is
the task left out on 10-03; set `XT=` (empty) if P shows nothing being written, and the task joins
the run.)

**Rehearsed** `[SANDBOX 2026-10-05]`, 13/13, twice from scratch: a bare origin at `6173e92`; a laptop
clone at `/c/Users/leona/Sbi-extractor` with `core.autocrlf=true` and the patch in `~/Downloads`; a
davinci clone at `/davinci-1/home/ldellamea/repos/Sbi-extractor` with `artifacts/sbi_hpc`; the ANN
tools at `04977e0` with a template library; the cohort manifest at its spaced path; a synthetic
`Main/` -- v1 with three tasks (line 0 the test task), v2 with a replay of a v1 task (a subset) and a
same-seed task with other theta, v3 with a replay of the test task (the same files), v7 a task without
`manifest.json`, v12 at `simtime` 180 with a task still being written; the test task run first at
the old code, as on 10-03; stub `conda` / `module` / `qstat`, the suite's fake `qsub`. Held: L2 pushes
the tested tree with the two trailers and leaves the clone clean, and run again STOPs with nothing
pushed; S3y GO and 26/26; P names the running task only; S8y before S7y STOPs; S7y with the test task
made a replay of a longer copy STOPs, refusing and naming its folder; S7y with `XT=` empty takes the
task in (5 to run); S7y drops the two replays, keeps the other-theta task, 4 to run, 1 done,
`EXCLUDED 4`, `-J 0-3%20`, WALLTIME 00:10:00; S8y submits the array and the gate; the gate, run as its
job would, exits 0; S9y prints the record line, `5 tasks`, `4 excluded (2 replays)` and the task left
out; the record's `replays` block names the two. Not checked: the real seed key on the data (the code
says `_resolved_seed_master`; S7y's `seed keys ...` line shows it per task count, and many tasks with
`no seed recorded` would mean stop and paste); the number of replays and whether the test task is one;
the real array's walltime; whether the gate job's 4 h and 8 cores suffice `[NOT VERIFIED]`.

**[v1.12, 2026-10-05: the census ran; the gate is v2, one task per seed (D-064).]**
The census C1 `[CLUSTER 2026-10-05 14:00, out/seed_census.log]`: 255 tasks with `job_args.json` (46
without: v9's empty folders), every one with `_resolved_seed_master`, `seed_master`, `seed_device`,
`seed_neuron`, `seed_synapse`, `seed_astro`; 94 distinct seeds; 47 shared by 208 tasks -- the
`_resolved_seed_master` series 3000000-3000019 (`sweep_intel_task*`) and 4000000-4000026
(`sweep_cpu_task*`) across v1-v9 ("especially within the first ones", D-061); no shared-seed pair with
different theta at the first and last common file. Relative to each group's first task: 85 shorter
copies, 54 longer, 19 with the same files, 2 that only overlap (`v2/sweep_cpu_task0007`, 1,037 vs
1,048; `v2/sweep_intel_task0008`, 1,444 vs 1,444), and `v7/sweep_cpu_task0009` (excluded, no files).
The test task `v1/sweep_cpu_task0000` (1,334 files, done on 10-03) is a subset of
`v9/sweep_cpu_task0000` (3,504), the copy the gate keeps (D-063), so the plan will refuse until its
output is moved out of `Outputs_v2` -- M1 below. Claude's estimate from the census `[reasoning]`:
one task per seed drops about 160 tasks; about 93 tasks run (94 with the v12 task, if P shows it
finished), about 276,000 iterations against the 491,893 of the 10-03 plan; at the test task's
2.3254 s per iteration and 20 tasks at a time, about 9 h of wall time if that rate holds. S7y prints
the real numbers.

What v2 changes (D-064): a same-seed, same-contract task whose shared iteration files carry the same
theta is a replay whether its files are the same as, a subset of, or only overlap the kept task's;
the files only it holds are dropped with it, counted as `files_lost` (per task in `excluded`, the
total in `counts.replays_files_lost`) and printed (`[plan] replays dropped : <n> (named under
EXCLUDED); iteration files only a dropped replay held: <m>`). And the refusal for a replay that already
has detections now says `mkdir -p '<root>_replays_moved/<campaign>' && mv '<folder>' ...`: out of the
root, since a moved copy left beside the others would sit under the record unnamed.

**The patch** `c8_replay_gate_v2.patch` (sent in this chat, 2026-10-05, about 14:20): 50,047 bytes,
sha256 `40bf4fa86cb2c53d6812b8372372cd805f6e2c12092fccab04080197a422a64f`; applies to the tree
`baeaa95fb977d53879b30a71df69547a9f4f3273` (`6173e92`); tree after it
`67133a86a70943bc8083629ba18fb186c1333e58` (sandbox commit `044fafc`, information only); the same five
files, +508 -16. Checked `[SANDBOX 2026-10-05]`: compiles, ASCII, LF; the suite 26/26 twice; six
mutations -- no drop, no theta comparison, the keeper order reversed, an overlapping replay kept, the
CLI seed kept in the key, the lost files not counted -- each failed P7 and nothing else.

Order: L2, S3y, P, S7y -- which stops on the test task -- M1, S7y again (paste its output), S8y, S9y.

```
L2: cd /c/Users/leona/Sbi-extractor && git checkout feat/real-arm-parity && git pull --ff-only && [ "$(git rev-parse 'HEAD^{tree}')" = baeaa95fb977d53879b30a71df69547a9f4f3273 ] && git apply --cached ~/Downloads/c8_replay_gate_v2.patch && git commit -q -m "sim_reextract plan: a replay gate -- one task per seed; a task that replays another task's seed and simulations is not re-detected (C8, D-061, D-063, D-064)" -m "$(printf 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01RtCUZWW1sF31UDbqxk9r8Z')" && git checkout -- EXTRACTOR_USAGE.md sim_reextract/README.md sim_reextract/sim_reextract_gate.py sim_reextract/sim_reextract_plan.py sim_reextract/smoke_test_sim_reextract.py && [ "$(git rev-parse 'HEAD^{tree}')" = 67133a86a70943bc8083629ba18fb186c1333e58 ] && git push && echo "PUSHED: the tested tree 67133a86" || echo "STOP: nothing pushed, or not the tested tree -- paste this output"
S3y: module load proxy; cd /davinci-1/home/ldellamea/repos/Sbi-extractor && git pull --ff-only && if [ "$(git rev-parse 'HEAD^{tree}')" = 67133a86a70943bc8083629ba18fb186c1333e58 ]; then echo "GO: the replay gate is on davinci ($(git log -1 --format=%h))"; source env.sh && conda activate sbi_export && mkdir -p sim_reextract/out && rm -f sim_reextract/out/smoke_c8.log && (cd sim_reextract && ANN_TOOLS=/davinci-1/home/ldellamea/ANN/MEA_analysis python3 smoke_test_sim_reextract.py) > sim_reextract/out/smoke_c8.log 2>&1; tail -1 sim_reextract/out/smoke_c8.log; else echo "STOP: davinci is not at the tested tree (HEAD $(git log -1 --format=%h)) -- was the laptop push done?"; fi
M1:  O=/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2; D=/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2_replays_moved; T=campaign_cadex_rho1300v1/sweep_cpu_task0000; if grep -qF "already have detections" /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract/out/plan_c8y.log && grep -qF "$T ($O/$T)" /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract/out/plan_c8y.log && [ -d "$O/$T" ] && [ ! -e "$D/$T" ]; then mkdir -p "$D/campaign_cadex_rho1300v1" && mv "$O/$T" "$D/$T" && echo "MOVED: $O/$T -> $D/$T"; else echo "STOP: S7y did not refuse on $T, or it is already moved -- nothing moved"; fi
```

(S7y, S8y, S9y are the v1.11 block's lines, unchanged; P is the v1.8 block's.)

**Rehearsed** `[SANDBOX 2026-10-05]`, 16/16, twice from scratch, on the v1.11 replica with the test
task made a replay: v1 `sweep_cpu_task0000` (line 0, run first at the old code) with a longer copy in
v3; a subset replay; a same-seed task with other theta; an overlapping pair (2 topologies x 1 against
1 x 2); v7 without `manifest.json`; v12 at 180 s with a task being written. Held: L2 PUSHED 67133a86,
again STOP; S3y GO and 26/26; P names the running task; M1 before any S7y STOP; S8y before S7y STOP;
S7y STOPs on the test task, naming its folder and the `mkdir`/`mv` line; M1 MOVED it to
`Outputs_v2_replays_moved/`, and again STOP; S7y with `XT=` empty 7 to run; S7y with the task left out:
3 replays DROPPED (subset, same files, overlap with 1 file lost), the other-theta task kept, 6 to run,
`EXCLUDED 5`, `-J 0-5%20`, WALLTIME 00:10:00; S8y submits; the gate job exits 0; S9y `record: 6
tasks, ... 5 excluded (3 replays)`, the task left out named; the record's `replays` names the three and
no replay's output is under `Outputs_v2`.

**[v1.13, 2026-10-05 about 15:30: the v2 gate is on GitHub; the plan ran on the real data; the array is NOT submitted.]** **[2026-10-06: the array was submitted after this block was written (S8y; the time is in `out/submissions.txt`), ran, and the gate refused -- the v1.14 block.]**
The user asked (14:28) for the run lines in the chat and not only here: they were sent there, and this
block keeps them as the record. Claude has submitted nothing and changed nothing on davinci.
Where the run stands -- done by the user and pasted: L2, S7y (REFUSED on the test task), M1, S7y (the
plan). Not done: G0, P again, S8y (the submission), the array, the gate job, S9y.

**L2** `[GITHUB 2026-10-05 about 15:25, read by Claude]`: `feat/real-arm-parity` is at `a8e9a43`
(2026-10-05 14:38:12 +0200, Leonardo D.M.), tree `67133a86a70943bc8083629ba18fb186c1333e58`, parent
`6173e92` (tree `baeaa95fb977d53879b30a71df69547a9f4f3273`); five files, +508 -16
(`EXTRACTOR_USAGE.md`, `sim_reextract/README.md`, `sim_reextract_gate.py`, `sim_reextract_plan.py`,
`smoke_test_sim_reextract.py`) -- the tested tree of the v1.12 block. Its trailer names Claude Sonnet 5.5:
the L2 that was run is the one sent in the chat, which differs from the v1.12 block's L2 line in that one
word (Opus 5.5 there). S3y's output was not pasted; the suite's 26/26 is inferred, because S7y runs the
plan only when `tail -1 out/smoke_c8.log` is `ALL 26 CHECKS PASSED`, and it did run it `[reasoning]`.

**P** `[CLUSTER 2026-10-05 about 15:00]`: the pasted line is `campaign_cadex_rho1300v12/sweep_cfd_task0010
214 | 0` (files written in the last 6 h | minutes since the newest): the v12 task is still being written,
so it stays left out by name (`XT=` in S7y; the plan printed it as left out, which S7y's guard checks).

**S7y, first run: STOP** `[CLUSTER 2026-10-05 15:01]`. REFUSED on `campaign_cadex_rho1300v1/sweep_cpu_task0000`
alone: the 10-03 test task is a replay of `v9/sweep_cpu_task0000` (3,504 iterations, the copy the gate
keeps; census, v1.12 block) and already had detections in `Outputs_v2`. **M1** `[CLUSTER 2026-10-05 15:10]`:
`MOVED: .../Outputs_v2/campaign_cadex_rho1300v1/sweep_cpu_task0000 ->
.../Outputs_v2_replays_moved/campaign_cadex_rho1300v1/sweep_cpu_task0000`. Nothing was deleted; the folder is
outside the output root and so outside the record; it is the user's to delete when it is no longer wanted.

**S7y, second run: the plan** `[CLUSTER 2026-10-05 15:10]`, the lines of the user's paste that carry the totals:

```
[plan] tasks to run    : 93 (array 0-92), 276347 iterations; largest task 11728 iterations, 26 topologies
[plan] replays (D-061) : seed keys _resolved_seed_master x253, seed_astro x253, seed_device x253, seed_neuron x253, seed_synapse x253; 253 tasks compared, 93 distinct seeds, 47 same-seed group(s), 0 task(s) with no seed recorded
[plan] replays dropped : 160 (named under EXCLUDED); iteration files only a dropped replay held: 1
[plan] EXCLUDED 208 task(s), named, not run:
# tasks       : 93   out root: /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2
# walltime    : 15:10:00   ncpus 48   queue cpu   concurrency 20
WALLTIME=15:10:00  PLAN_ARGS=--allow-mixed-contract --exclude-task campaign_cadex_rho1300v12/sweep_cfd_task0010
    resources_max.walltime = 300:00:00
```

Read with the census `[computed by Claude from the pasted lines, checked against the census, 0 mismatches]`:
47 groups, all of one contract signature (`1649221c57628597`); the keeper of each, by campaign: v1 7, v2 4,
v3 8, v5 20, v9 8 -- 112,856 iterations; 46 more tasks with a seed of their own -- 163,491 iterations;
together the 93 tasks and 276,347 iterations (491,893 in the plan of 10-03). The 160 dropped replays, by
relation to the kept copy: 3 with the same files, 156 subsets, 1 overlapping -- `v2/sweep_intel_task0008`
(1,444 iterations) against the kept `v1/sweep_intel_task0008` (1,444; a tie, so the lower campaign version is
kept), the same theta at 10 of 1,443 shared files and 1 file only the dropped copy holds, which is lost
(D-064). By campaign the dropped are v1 25, v2 28, v3 39, v4 12, v5 7, v7 9, v8 40, holding 216,880
iterations. The 208 excluded are the 160 replays and 48 others -- by subtraction `[reasoning]`: the 46 tasks
without `job_args.json` and the empty `v7/sweep_cpu_task0009` (census) and the v12 task left out by name; the
plan's EXCLUDED list itself was not in the paste.

**Walltime** `[reasoning]`: `ceil(11,728 x 3102.1 / 1334 x 2 / 600) x 600` s = 15:10:00 for the largest task.
At the 10-03 test task's 2.3254 s per iteration (3,102.1 s for 1,334 iterations; measured on that task alone,
`[NOT VERIFIED]` for any other) the largest task takes 7.58 h by itself, so the walltime is 2.0 times that;
all 276,347 iterations are 178.5 node-hours (a node is 48 cores), at least 8.9 h over 20 concurrent tasks and
at most about 16.5 h (the total over 20 plus the longest task) if the scheduler gives 20 slots throughout.
A task killed at the walltime leaves no `mea_manifest.json` and the gate names it; the way back is the one
of the v1.10 block: once the gate has refused, `RESUME=1` with the same `PLAN_ARGS` and WALLTIME re-runs only
the incomplete tasks, then `bash launch_sim_reextract.sh gate` -- Claude gives the exact lines then.

**G0, new, read-only** `[SANDBOX 2026-10-05]`. The geometry of a C8 output (`n_side 3`, pitch 60 um, edge
25 um, `n_sub`, `fs` 10110.09; D-015, D-021, D-024) has been read so far only from the qsub line the plan prints,
never from an output file; and `qsub -v` carries `EXTRA_ARGS` with its spaces (the August jobs did, probe P7, and
the 10-03 test job used the same route) `[KB]` -- but whether it reached the job is a thing the job recorded.
G0 reads the one real output there is, the test task's, now in `Outputs_v2_replays_moved/`: `mea_env.json`
`extra_args` (as the job saw it) against `plan.json` `extra_args`; `mea_manifest.json` `total_done` against
`total_iters`; the first `mea_iter_*.npz`: `electrode_centers` shape against `(n_e, 2)` and its `meta_json`
`n_side`, `pitch`, `edge`, `n_sub`, `fs` against `plan.json` `geometry` (tolerance 1e-9 relative). It prints
four lines of what it read and then `GEO OK: ...` or `GEO STOP: <reasons>`. Run on real `process_campaign.py`
output (the suite's Fixture), twice from scratch: `GEO OK`, and `GEO STOP` for each of an altered `extra_args`,
an incomplete run, a wrong `electrode_centers` shape and a wrong `meta_json` field. On davinci the first line should
read `1334 of 1334 iterations`, `extra_args '--n_side 3 --pitch 60.0 --edge 25.0 --fs 10110.09'`, then
`electrode_centers (9, 2)`; `n_sub` is whatever the plan holds, and the two must agree. A traceback means G0 could not
read something: paste it, do not run S8y.

```
G0:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && python3 -c "import json, glob, numpy as np; D = '/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2_replays_moved/campaign_cadex_rho1300v1/sweep_cpu_task0000'; p = json.load(open('plan.json')); g = p['geometry']; e = json.load(open(D + '/mea_env.json')); m = json.load(open(D + '/mea_manifest.json')); f = sorted(glob.glob(D + '/topo_*/mea_iter_*.npz')); z = np.load(f[0], allow_pickle=False); k = json.loads(str(z['meta_json'])); ec = tuple(z['electrode_centers'].shape); print('test task :', m.get('total_done'), 'of', m.get('total_iters'), 'iterations,', len(f), 'mea_iter files | extra_args', repr(e.get('extra_args'))); print('its job   : python', e.get('python'), 'numpy', e.get('numpy'), 'scipy', e.get('scipy'), '|', e.get('executable')); print('one file  :', f[0].split('/')[-1], 'electrode_centers', ec, {a: k.get(a) for a in ('n_side', 'pitch', 'edge', 'n_sub', 'fs')}); print('plan      : extra_args', repr(p['extra_args']), {a: g.get(a) for a in ('n_side', 'n_e', 'pitch_um', 'edge_um', 'n_sub', 'fs')}); bad = (['extra_args differ'] if not (e.get('extra_args') == p['extra_args']) else []) + (['electrode_centers %r' % (ec,)] if not (ec == (g['n_e'], 2)) else []) + (['total_done %r of %r' % (m.get('total_done'), m.get('total_iters'))] if not (m.get('total_done') == m.get('total_iters')) else []) + ['meta_json %s = %r, want %r' % (a, k.get(a), g[b]) for a, b in (('n_side', 'n_side'), ('pitch', 'pitch_um'), ('edge', 'edge_um'), ('n_sub', 'n_sub'), ('fs', 'fs')) if not (abs(float(k[a]) - float(g[b])) <= 1e-9 * max(1.0, abs(float(g[b]))))]; print('GEO OK: the test task output has the planned geometry' if not bad else 'GEO STOP: ' + '; '.join(bad))"
```

**The order** (S7y and M1 are done): G0 -> P -> S8y; PROG at any time while the array runs; S9y once `qstat -u $USER`
is empty. G0 must print `GEO OK`; P must list no task other than the one left out by name that has been written in the
last minutes -- if one has, it is still being simulated: stop and tell Claude; S8y must print `[plan] tasks to run : 93
(array 0-92), 276347 iterations`, `replays dropped : 160`, the v12 task as left out on the command line, `# walltime :
15:10:00`, and the two job ids, and no `ABORT` or `REFUSED`. P is the v1.8 block's line, S8y and S9y are the v1.11 block's,
unchanged (S9y prints the record's digest line, `record: <n> tasks, <it> iterations, <x> excluded (<r> replays); left out
by name: <task or none>`, which here should read `93 tasks, 276347 iterations, 208 excluded (160 replays)` `[reasoning]`).

```
PROG: find /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2 -mindepth 3 -maxdepth 3 -name mea_manifest.json | wc -l
```

PROG counts the `mea_manifest.json` files under `Outputs_v2`, one per finished task, out of 93 (a task writes it last). **No
launcher mode between S8y and S9y** (`launch_sim_reextract.sh plan`, `array` or `gate`, with or without `DRYRUN=1`, and so
not S7y either): each re-plans and rewrites `plan.json`, which the gate job reads when it starts.

**Not covered by these lines** `[NOT VERIFIED]`: the gate job's limits (8 cores, 16 GB, 4 h) against about 276,000
`mea_iter` opens and the library check -- Claude's estimate is minutes to under an hour, not timed; the 2.3254 s per
iteration for tasks other than the test task; whether the queue grants 20 concurrent 48-core nodes; whether
`-J 0-92%20` is accepted by this PBS Pro (if `qsub` refuses, nothing is submitted and S8y prints the ABORT); whether
any task other than the v12 one is being written (P, right before S8y). The suite 26/26 and S3y were not pasted.

**[v1.14, 2026-10-06 about 12:45: the array ran; the gate REFUSED, 17 of 93 tasks bad; nothing written.]** **[2026-10-06 about 13:40: D1 is now delivered through git as `sim_reextract/c8_diag.py` (P2 of the v1.15 block runs it); the heredoc below is not to be pasted.]** **[corrected 2026-10-06 about 15:20, from D1's output (v1.16 block): the filesystem is Lustre (`df` shows an `@o2ib2:/home01` mount), not GPFS; and it was not one event hitting the ~20 running tasks -- one node, `dvnode012`, failed 16 tasks one after another from 16:35 to 02:28, each 15-57 min after it started; the 17th task's one failure is an unreadable RAW file. The empty-files reading (data lost after the rename) stands; on Lustre, a client eviction.]**
What happened `[CLUSTER 2026-10-06 07:16:41, S9y, the user's paste of 12:28]`: the gate job ended with
`[gate] read 239732 mea_iter_*.npz file(s) across 93 task(s); 17 task(s) bad`, `[gate] REFUSED -- 17 bad
task(s), nothing written`, `[job] sim_reextract_gate_exit=1`. No `REEXTRACTION_RECORD.json`: Stage C is not
closed (D-033). The 76 tasks that passed are complete on disk and stay; 276,347 iterations were planned (S7y),
so 36,615 outputs (13%) are missing or unreadable, all inside the 17. G0, P, S8y and the progress line were
run without their outputs being pasted; when S8y ran is in `out/submissions.txt`.

The pasted tail (60 lines) shows two signatures. (1) Stopped part-way -- `v5/sweep_cpu_task0023` (2,350
iterations planned), `v9/sweep_cpu_task0003` (4,320) and one task whose block is cut off: no
`mea_manifest.json` (written last); one topology part done (`topo_00001` 93 and 92 of 240 files); every later
topology `no output directory`; the newest files of the part-done topology `unreadable: EOFError('No data left
in file')` (83-91 of 92 in one, 86, 87, 89, 90, 92 of 93 in the other). (2) Finished with one failure --
`v9/sweep_intel_task0002`: `total_done 6129 of total_iters 6130`, `topo_00025: _failures.log present`.

**Claude's reading.** `EOFError('No data left in file')` is what `np.load` raises on a ZERO-BYTE file; a
half-written zip raises `BadZipFile` instead `[verified by running, numpy 2.4.4]`, and the real gate, run here on a
mock with emptied files, prints that exact line `[SANDBOX 2026-10-06]`. `process_campaign.py` never writes in
place: `_atomic_savez` writes a temp file and renames it to `mea_iter_*.npz` once the write has returned
`[CODE ANN@04977e0 hpc/MEA Traces/process_campaign.py:155-160]`. So a zero-byte file under the final name is
data lost AFTER the rename: the node losing its unwritten buffers (crash, reboot, dropped from GPFS) or the
filesystem refusing the data late (quota or space at write-back); a PBS kill (walltime, memory, qdel) leaves
at most a `tmp*.npz` `[reasoning; NOT VERIFIED on davinci]`. Nor is it the walltime by arithmetic: those tasks
had done at most about 330 iterations, 13 to 80 minutes at the 10-03 test task's pace (6 topologies, 1,334
iterations in 3,102.1 s: 2.3 s per iteration one topology at a time, about 14 s if all six ran at once) against
15:10:00 `[reasoning]` -- unless they ran 10-60 times slower, which D1's start and last-write times show. 17 is
close to the 20 that run at once (`%20`), what one event during the night would give `[reasoning, a guess]`.

**One worker per task** `[reasoning from the code; NOT VERIFIED]`: `process_campaign.py` gives each topology its
own worker and creates its folder first (`process_topo`, `walk_campaign`), so with 48 workers every topology of a
task has a folder within seconds; later topologies with no folder mean one worker. `submit_mea_array.sh:216`
takes the count from `NCPUS="${PBS_NCPUS:-1}"`; if davinci's PBS does not set `PBS_NCPUS`, every task used 1 of
its 48 cores. The detections do not depend on it (noise and template seeds are per topology and iteration,
`process_campaign.py:219, 288-289`); it is a speed question for after the record, not a change to make now (the
tools are fingerprinted into the record).

**The way back** is the built one: once the cause is known, the plan with `RESUME=1` keeps the 76 complete tasks
and runs the 17, then the gate. From the code `[CODE Sbi-extractor@a8e9a43]`: the re-run re-processes every
iteration of a task and overwrites the empty files by rename; a stale `_failures.log` in a task that then
completes is only a warning (`sim_reextract_gate.py:151-160`); `tmp*` leftovers are not counted (only
`mea_iter_*.npz`). Three things to settle first, which is what D1 is for: (a) the cause -- if quota or space,
the re-run fails the same way; (b) RESUME decides "complete" from `mea_manifest.json` alone
(`task_is_complete`), so a bad task with a complete manifest but empty files would be kept as done and the gate
would refuse again -- such a folder is moved out of `Outputs_v2` first, as M1 did; the two visible ones have no
manifest; (c) the iteration that raised in `v9/sweep_intel_task0002` -- transient (re-run fixes it) or the raw
file itself (it fails again: a decision for the user). Still open, the user's call when the re-plan is made:
whether the v12 task left out by name (D-060) goes in, if P shows it finished.

**D1 -- read-only diagnostic** (it writes only its own output, `out/c8_diag_d1.txt`). Run it BEFORE any
launcher mode: it reads the `plan.json` the gate used, and every mode rewrites that file. It prints the gate's
summary; per bad task: array index, planned iterations/topologies, mea_iter files with data and EMPTY, `tmp`
leftovers, the manifest, `mea_env.json` host, workers and start time, last write, seconds per iteration, whether
RESUME would re-run it or keep it, the gate's lines; the same timing over the 76 that passed; the
`_failures.log` entries; `qstat -xft` of the array (Exit_status counts, the bad subjobs' exit, walltime used,
host, start, comment); the error lines of every `.e` log of the array; `df` and `mmlsquota`. Tested
`[SANDBOX 2026-10-06]` on a mock (the fixture's tasks, real `process_campaign.py` outputs, one task stopped with two
emptied files and a tmp leftover, one with a `_failures.log` and one file short, the real gate, a fake `qstat` and
fake job logs), as a file and pasted into an interactive bash: both gave the same output; a task with a complete
manifest and one emptied file is shown as `KEPT as done by RESUME`; with no `qstat` and no `sim_reextract_plan`
importable it reports both and goes on. Not tested: davinci's `qstat -xft` text and the names of the array's
`.o`/`.e` files (it prints how many it found and one name).

```
cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && python3 - <<'EOF' 2>&1 | tee out/c8_diag_d1.txt
import collections, glob, json, os, re, statistics, subprocess, sys, time
HERE = os.environ.get('C8_HERE', '/davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract')
LOGS = os.path.expanduser(os.environ.get('C8_LOGS', '~'))
os.chdir(HERE); sys.path.insert(0, HERE)
try:
    import sim_reextract_plan as PLAN
    complete = PLAN.task_is_complete
except Exception as exc:
    print('(sim_reextract_plan not importable here: %r; the RESUME column is blank)' % (exc,)); complete = None
G = open('out/sim_reextract_gate.log').read().splitlines()
plan = json.load(open('plan.json'))
T = {t['campaign'] + '/' + t['sweep']: t for t in plan['tasks']}
hm = lambda x: time.strftime('%m-%d %H:%M', time.localtime(x)) if x else '?'
print('== D1, read-only. plan.json last written %s, %d task(s) in it' % (hm(os.path.getmtime('plan.json')), len(T)))
print('\n'.join(l for l in G if l.startswith(('[gate] tasks', '[gate] read', '[gate] REFUSED', '[job]'))))
bad, cur = collections.OrderedDict(), None
for l in G:
    m = re.match(r'\[gate\] FAIL (\S+)$', l)
    if m and m.group(1) in T:
        cur = m.group(1); bad[cur] = []
    elif cur and l.startswith('[gate]      - '):
        bad[cur].append(l[14:])
    else:
        cur = None

def facts(name):
    t = T[name]; d = t['out_dir']; r = {'idx': t.get('array_index'), 'plan': '%d/%d' % (t['n_iters'], t['n_topos'])}
    fs = glob.glob(os.path.join(d, 'topo_*', 'mea_iter_*.npz'))
    st = [(os.path.getmtime(f), os.path.getsize(f)) for f in fs]
    r['full'] = sum(1 for m, s in st if s > 0); r['empty'] = sum(1 for m, s in st if s == 0)
    r['tmp'] = len(glob.glob(os.path.join(d, 'topo_*', 'tmp*')))
    r['flog'] = sorted(glob.glob(os.path.join(d, 'topo_*', '_failures.log')))
    try:
        e = json.load(open(os.path.join(d, 'mea_env.json')))
    except Exception:
        e = {}
    r['host'], r['workers'] = e.get('host', '?'), e.get('workers', '?')
    w = e.get('written')
    r['start'] = time.mktime(time.strptime(w[:19], '%Y-%m-%dT%H:%M:%S')) if w else None
    mp = os.path.join(d, 'mea_manifest.json')
    if os.path.isfile(mp):
        m = json.load(open(mp)); r['man'] = 'manifest %s/%s' % (m.get('total_done'), m.get('total_iters')); r['end'] = os.path.getmtime(mp); n = m.get('total_done') or 0
    else:
        r['man'] = 'NO manifest'; r['end'] = max([m for m, s in st] or [0]) or None; n = r['full']
    r['resume'] = '?' if complete is None else ('KEPT as done by RESUME' if complete(t) else 're-run by RESUME')
    r['h'] = (r['end'] - r['start']) / 3600.0 if r['start'] and r['end'] else None
    r['rate'] = r['h'] * 3600.0 / n if r['h'] and n else None
    return r

print('--- the %d bad task(s): array index | task | planned iters/topos | on disk: mea_iter files with data + EMPTY, tmp leftovers | manifest | host, workers | started -> last write (h) | s/iter | RESUME | gate lines' % len(bad))
F = {}
for name, probs in bad.items():
    r = F[name] = facts(name)
    un = sum('unreadable' in p for p in probs); nd = sum('no output directory' in p for p in probs)
    oth = [p for p in probs if 'unreadable' not in p and 'no output directory' not in p and 'mea_iter_*.npz, raw has' not in p]
    print('%3s | %-45s | %11s | %5d + %d EMPTY, %d tmp | %-17s | %s, w=%s | %s -> %s (%s h) | %s | %s | unreadable %d, no dir %d%s' % (
        r['idx'], name, r['plan'], r['full'], r['empty'], r['tmp'], r['man'], r['host'], r['workers'], hm(r['start']), hm(r['end']),
        '%.1f' % r['h'] if r['h'] is not None else '?', '%.2f' % r['rate'] if r['rate'] else '?', r['resume'], un, nd, ('; ' + '; '.join(oth))[:160] if oth else ''))

good = [n for n in T if n not in bad]
R = {n: facts(n) for n in good}
rates = sorted(r['rate'] for r in R.values() if r['rate'])
hours = sorted(r['h'] for r in R.values() if r['h'])
print('--- the %d task(s) that passed: s/iter min %s median %s max %s; run time (h) median %s max %s; workers %s' % (
    len(good), *(('%.2f' % x) if rates else '?' for x in ((rates or [0])[0], statistics.median(rates or [0]), (rates or [0])[-1])),
    *(('%.1f' % x) if hours else '?' for x in (statistics.median(hours or [0]), (hours or [0])[-1])),
    dict(collections.Counter(r['workers'] for r in R.values()))))
starts = sorted(r['start'] for r in list(R.values()) + list(F.values()) if r['start'])
ends = sorted(r['end'] for r in R.values() if r['end'])
print('    first task started %s, last started %s; last good task finished %s' % (hm(starts[0] if starts else None), hm(starts[-1] if starts else None), hm(ends[-1] if ends else None)))
hosts = collections.Counter(r['host'] for r in F.values()); hall = collections.Counter(r['host'] for r in list(R.values()) + list(F.values()))
print('    hosts of the bad tasks: %s (all tasks on those hosts: %s)' % (dict(hosts), {h: hall[h] for h in hosts}))
print('    last writes of the bad tasks, sorted: %s' % ', '.join(hm(r['end']) for r in sorted(F.values(), key=lambda r: r['end'] or 0)))

print('--- _failures.log files in the bad tasks (path | entries | first entry):')
for name, r in F.items():
    for fl in r['flog']:
        txt = open(fl, errors='replace').read()
        ent = re.findall(r'^\S*iter_\d+\.npz: .*$', txt, flags=re.M)
        print('%s/%s | %d | %s' % (name, fl.split('/')[-2], len(ent), (ent[0] if ent else txt[:200])[:400]))
        last = [l for l in txt.splitlines() if l.strip()][-1:] or ['']
        print('    last line: %s' % last[0][:300])

sub = [l.split() for l in open('out/submissions.txt').read().splitlines() if l.strip()]
print('--- out/submissions.txt, last 3:'); print('\n'.join(' '.join(s) for s in sub[-3:]))
arr = [s[2] for s in sub if len(s) > 2 and s[1] == 'array']
if arr:
    A = arr[-1]; seq = A.split('[')[0].split('.')[0]
    try:
        txt = subprocess.run(['qstat', '-xft', A], capture_output=True, text=True, timeout=300).stdout
    except Exception as exc:
        txt = ''; print('qstat -xft %s failed: %r' % (A, exc))
    recs, rec, key = {}, None, None
    for l in txt.split('\n'):
        if l.startswith('Job Id:'):
            jid = l.split(':', 1)[1].strip(); m = re.search(r'\[(\d+)\]', jid); rec = {}; key = None
            if m: recs[int(m.group(1))] = rec
        elif l.startswith('\t') and rec is not None and key:
            rec[key] += l.strip()
        elif ' = ' in l and rec is not None:
            key, v = l.strip().split(' = ', 1); rec[key] = v
    print('--- PBS, array %s: %d subjob record(s); Exit_status counts: %s' % (A, len(recs), dict(collections.Counter(r.get('Exit_status', '?') for r in recs.values()))))
    bidx = {F[n]['idx'] for n in F}
    for i in sorted(recs):
        r = recs[i]
        if i in bidx or r.get('Exit_status') != '0':
            print('%3d | exit %s | walltime used %s | %s | start %s | %s' % (i, r.get('Exit_status'), r.get('resources_used.walltime'), r.get('exec_host', '?').split('/')[0], r.get('stime'), r.get('comment', '')[:150]))
    lo = sorted(glob.glob(os.path.join(LOGS, 'c8_mea.[oe]%s*' % seq)))
    print('--- job logs: %d file(s) %s (e.g. %s)' % (len(lo), os.path.join(LOGS, 'c8_mea.[oe]%s*' % seq), os.path.basename(lo[0]) if lo else '-'))
    for name, r in F.items():
        i = r['idx']; out = []
        for k, pat in (('o', r'workers|dispatching|iters processed|done\.'), ('e', r'PBS|[Kk]ill|Error|Terminat|Traceback|signal|Bus|memory')):
            p = os.path.join(LOGS, 'c8_mea.%s%s.%s' % (k, seq, i))
            if not os.path.isfile(p):
                out.append('.%s missing' % k); continue
            L = [l.strip() for l in open(p, errors='replace').read().splitlines() if l.strip()]
            out.append('.%s: %s || last: %s' % (k, ' / '.join([l for l in L if re.search(pat, l)][:4])[:300], (L[-1] if L else '<empty>')[:200]))
        print('%3s %s\n      %s' % (i, name, '\n      '.join(out)))
    allE = sorted(glob.glob(os.path.join(LOGS, 'c8_mea.e%s*' % seq)))
    C = collections.Counter()
    for p in allE:
        for l in open(p, errors='replace').read().splitlines():
            if re.search(r'Errno|Error|PBS|[Kk]illed|Terminat|signal', l):
                C[re.sub(r'\d+', 'N', re.sub(r'/\S+', '<path>', l.strip()))[:160]] += 1
    print('--- error lines over all %d .e logs of the array (count | line, numbers and paths masked):' % len(allE))
    print('\n'.join('%5d | %s' % (n, l) for l, n in C.most_common(10)) or '  (none)')
print('--- storage now:')
for cmd in (['df', '-h', os.path.dirname(plan['out_root'])], ['mmlsquota', '--block-size', 'auto']):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print('$ ' + ' '.join(cmd) + '\n' + (r.stdout + r.stderr).strip()[:1200])
    except Exception as exc:
        print('$ %s: %r' % (' '.join(cmd), exc))
print('== D1 done')
EOF
```

**[v1.15, 2026-10-06 about 13:40: D1 through git; a cleanup tool for after the record (D-065).]**
The user asked (12:49) for "the push and pull", i.e. D1 delivered through git, not pasted; then (12:51)
for a way to remove "all the old extracted files and the seed duplicates", settled on a question card as
D-065: everything being re-extracted now, and the Giulia recordings' old extractions; the duplicates' raw
simulations; after the C8 record; list, then delete.

**The patch** `c8_tools.patch` (sent in this chat, 2026-10-06): 43,484 bytes, sha256
`a53eacb4ec6f2451b0e7b1e3087cef20eada3b3f91153d15c501e3bb610106e5`; applies to the tree
`67133a86a70943bc8083629ba18fb186c1333e58` (`a8e9a43`, on GitHub); tree after it
`404496d5ed0fe8bc9e32911016a070feb9feadb6` (sandbox commit `ec5d1fa`, information only); four files, +847:
`sim_reextract/c8_diag.py` (D1 of the v1.14 block as a module; on the mock it prints the same lines),
`sim_reextract/c8_cleanup.py`, `sim_reextract/smoke_test_c8_tools.py` (13 checks), `sim_reextract/README.md`
(three table rows; steps 4-5 of the davinci order). Checked `[SANDBOX 2026-10-06]`: ASCII, LF, no tabs;
`py_compile`; every `args.*` declared; 13/13 twice; seven mutations (no seed comparison, no sidecar check, no
completeness check, no list-equality check, the whole old root listed, the CODE not checked, RESUME always
"re-run") each failed at least one check; the C8 suite `smoke_test_sim_reextract.py` 26/26 on the new tree;
the patch applied with `git apply --cached` on a fresh clone of `a8e9a43` gives the tree above, the four
files byte-identical to the tested ones.

**c8_cleanup.py** (read from its header; D-065): LIST (default) writes `out/c8_cleanup_list.txt` (group,
path, files, bytes, note) and prints a CODE (12 hex of the sha256 of the listed paths); `--delete CODE`
recomputes the list, refuses unless the same paths and the same CODE, deletes each path (`shutil.rmtree`),
logs to `out/c8_cleanup_deleted.txt`, and removes campaign folders left empty. Groups: A the raw folders
`<sim_main>/<campaign>/<sweep>` of the record's replays -- only if the kept copy is a record task with its raw
folder present and the two `job_args.json` carry the same seed values (`sim_reextract_plan.task_seed`); a
replay with `files_lost` > 0 is listed with a NOTE (`v2/sweep_intel_task0008`, 1 file, on the 10-05 plan); B
`ANN/MEA_analysis/Outputs/<campaign>/<sweep>` for every task the record covers (re-extracted, or a replay);
everything else under `Outputs/` is printed as kept; C `Outputs_v2_replays_moved/<campaign>/<sweep>` for the
record's replays (the 10-03 test task); D `Giulia_Astro/extracted_giulia_partial_20261002` and
`..._partial2_20261002`, only if `extracted_giulia/cohort_manifest.json` matches its `.sha256`. Refused
outright: no record, a record not matching its `.sha256`, a record task not complete on disk, a listed path
that is a symlink, lies inside `Outputs_v2`, or contains a kept task's raw or output folder. NOT touched:
`Outputs_v2`, the kept tasks' raw folders, `Giulia_Astro/extracted_giulia/` and `Bio_Data/`, the
September Giulia sim detections `mea_out_1electrode` (G2 has not re-extracted them), the SBI exports, the
DUP15HD real-arm archives -- none is in D-065 as read.

**The lines.** L3 on the laptop (Git Bash), then P2 on davinci; C1 and C2 only once S9y prints a record.
Rehearsed `[SANDBOX 2026-10-06]` on a local origin at `a8e9a43`, a laptop clone with `core.autocrlf=true`
and a davinci clone at `a8e9a43` with the mock's gate log and plan: L3 `PUSHED: the tested tree 404496d5`,
again `STOP` (already applied), the laptop's working copy LF; P2 `GO`, `ALL 13 CHECKS PASSED`,
`D1: == D1 done, 38 lines`; P2 with the push not done: `STOP: davinci is not at the tested tree`, nothing
run.

```
L3:  cd /c/Users/leona/Sbi-extractor && git checkout feat/real-arm-parity && git pull --ff-only && [ "$(git rev-parse 'HEAD^{tree}')" = 67133a86a70943bc8083629ba18fb186c1333e58 ] && git apply --cached ~/Downloads/c8_tools.patch && git commit -q -m "sim_reextract: c8_diag.py reads a refused gate; c8_cleanup.py lists, then deletes, what the record makes redundant (C8, D-065)" -m "$(printf 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01RtCUZWW1sF31UDbqxk9r8Z')" && git checkout -- sim_reextract/README.md sim_reextract/c8_diag.py sim_reextract/c8_cleanup.py sim_reextract/smoke_test_c8_tools.py && [ "$(git rev-parse 'HEAD^{tree}')" = 404496d5ed0fe8bc9e32911016a070feb9feadb6 ] && git push && echo "PUSHED: the tested tree 404496d5" || echo "STOP: nothing pushed, or not the tested tree -- paste this output"
P2:  module load proxy; cd /davinci-1/home/ldellamea/repos/Sbi-extractor && git pull --ff-only && if [ "$(git rev-parse 'HEAD^{tree}')" = 404496d5ed0fe8bc9e32911016a070feb9feadb6 ]; then echo "GO: the C8 tools are on davinci ($(git log -1 --format=%h))"; source env.sh && conda activate sbi_export && cd sim_reextract && python3 smoke_test_c8_tools.py > out/smoke_c8_tools.log 2>&1; tail -1 out/smoke_c8_tools.log; if tail -1 out/smoke_c8_tools.log | grep -qx 'ALL 13 CHECKS PASSED'; then python3 c8_diag.py > out/c8_diag_d1.txt 2>&1; echo "D1: $(tail -1 out/c8_diag_d1.txt), $(wc -l < out/c8_diag_d1.txt) lines in sim_reextract/out/c8_diag_d1.txt"; fi; else echo "STOP: davinci is not at the tested tree (HEAD $(git log -1 --format=%h)) -- was the laptop push done?"; fi
C1:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor && source env.sh && conda activate sbi_export && cd sim_reextract && python3 c8_cleanup.py 2>&1 | tee out/c8_cleanup_list.log | grep -E '^\[clean\] (record|[A-D] |NOTE|NOT LISTED|LISTED|REFUSED|kept)|--delete'
C2:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor && source env.sh && conda activate sbi_export && cd sim_reextract && nohup python3 c8_cleanup.py --delete PUT_THE_CODE_HERE > out/c8_cleanup_delete.log 2>&1 &
```

P2's expected output: `GO: the C8 tools are on davinci (<hash>)`, `ALL 13 CHECKS PASSED`, `D1: == D1 done, <n>
lines in sim_reextract/out/c8_diag_d1.txt`; the user sends that file. C1 (after the record) prints the record
line (`93 task(s), 160 replay(s) dropped` expected, if the re-run keeps the 10-05 plan), the four group totals,
the NOTE for the replay with a file only it holds, what it keeps under `Outputs/`, `LISTED <n> folder(s)` and
the `--delete <CODE>` line; C2 runs in the background (`nohup`): `tail -3 out/c8_cleanup_delete.log` ends with
`[clean] DONE: <n> deleted, 0 failed`. Not timed: the deletion of a few hundred thousand files on GPFS.

**[v1.16, 2026-10-06 about 15:20: D1 read -- a bad node, and one unreadable raw file; D-066; the second patch.]**
D1 ran on davinci through `c8_diag.py` (the c8_tools commit `459aa72`, pushed 14:19, tree `404496d5`),
`[CLUSTER 2026-10-06, the user's paste of 14:54]`. The array `1802376[]`, submitted 2026-10-05 15:47:34:
`Exit_status` 0 for 78 subjobs, 135 for 8, 1 for 6, 120 for 1.

**What happened.** 16 of the 17 bad tasks ran on `dvnode012` (17 tasks were placed there; one passed). Each
died 15 to 57 minutes after it started, and they followed one another from 16:35 to 02:28 -- a node that
kept accepting tasks and killing them. Their `.e` logs: `OSError: [Errno 5] Input/output error` writing a
`tmp*.npz` under `Outputs_v2` (12 times over the array), `[Errno 19] No such device` (2),
`BrokenPipeError: [Errno 108] Cannot send after transport endpoint shutdown` (2), then `Bus error (core
dumped)` from the shell (exit 135 = 128 + 7, SIGBUS), or a Python traceback (exit 1), or exit 120 (Python
could not flush its output at exit). `/davinci-1/home` is a Lustre mount (`df`: `...@o2ib2:/home01/home`,
1.1P, 38% used), and "Cannot send after transport endpoint shutdown (-108)" is the error a Lustre client
gives once the servers have evicted it `[lustre-discuss list, 2008, read]`; the empty files under their
final names are what such a client leaves of data it had not written back `[reasoning]`. Several of them
failed again while writing their own `_failures.log` (`Errno 5`, `19` or `108` on the log itself).
The 17th, `v9/sweep_intel_task0002`, ran on `dvnode022` for 4.5 h and finished 6129 of 6130: the one
failure is `BadZipFile('File is not a zip file')` reading the RAW simulation file
`Main/campaign_cadex_rho1300v9/sweep_intel_task0002/topo_00025/iter_00100.npz` -- if that file is damaged on
disk, a re-run fails the same way. Every one of the 17 is "re-run by RESUME" (no complete manifest among
them), `tmp` leftovers 0. Corrections to the v1.14 block are on its header line.

**The 76 that passed** `[CLUSTER]`: 2.30 s per iteration (median; 0.97 to 3.09) -- the 10-03 test task's
2.3254 holds; runs up to 6.7 h against 15:10:00; `workers` 1 in every task (`[mea] ... dispatching over 1
worker(s) (1 topo/worker; BLAS pinned: OMP=48 OPENBLAS=1 MKL=1)`): `PBS_NCPUS` is not set on davinci, so each
task used one core of its 48, as read from the code in v1.14. First start 10-05 15:47, last good finish
10-06 07:10; the gate ran on `dvnode003` with 8 workers, 07:15-07:16.

**Before the re-run** (S7y with `RESUME=1`, then S8y -- the v1.11 block's lines): D2, read-only, settles
three things. (1) Whether `dvnode012` (and `dvnode022`) are still in service (`pbsnodes`): PBS Pro has no
user-side way to exclude a node from `select` -- only an administrator's node resource or an explicit list of
hosts `[OpenPBS community forum, read]` -- so if `dvnode012` is still `free`, the HPC support is asked to
offline it before S8y; with 17 tasks starting together at most one would land there. (2) Every copy of the
unreadable raw file across the campaigns, with its neighbours (size, sha256, seed, whether it is a valid
zip): a damaged v9 file with an intact same-seed copy in a replay campaign (the seed-3000002 group: v1, v2,
v3, v4, v8 -- dropped replays, to be deleted after the record by D-065) is a repair for the user to decide; no
intact copy means that iteration cannot be re-detected and the gate will refuse that task again. (3) Whether
the v12 task left out by name (`sweep_cfd_task0010`) is still being written -- if not, the re-plan can take it
in (`XT=` empty); the user's call (D-060).

```
D2:  cd /davinci-1/home/ldellamea/ANN/Phenomenological/Main && echo "--- the two nodes:"; for n in dvnode012 dvnode022; do pbsnodes $n 2>&1 | grep -E "^$n|state =|comment =" | cut -c1-200; done; echo "--- every copy of the raw file the v9 task could not read, with its neighbours (sweep_intel_task0002/topo_00025):"; python3 -c "import glob,hashlib,json,os,zipfile; [print('%-68s %9d bytes  sha %s  seed %s  %s' % (p, os.path.getsize(p), hashlib.sha256(open(p,'rb').read()).hexdigest()[:16], json.load(open(p.split('/topo_')[0] + '/job_args.json')).get('_resolved_seed_master'), ('zip, first bad member: %s' % zipfile.ZipFile(p).testzip()) if zipfile.is_zipfile(p) else 'NOT A ZIP')) for p in sorted(glob.glob('campaign_cadex_rho1300v*/sweep_intel_task0002/topo_00025/iter_00099.npz') + glob.glob('campaign_cadex_rho1300v*/sweep_intel_task0002/topo_00025/iter_0010[01].npz'))]"; echo "--- tasks with an iter_*.npz written in the last 6 h  (files | newest, minutes ago):"; find campaign_cadex_rho1300v* -name 'iter_*.npz' -mmin -360 -printf '%T@ %h\n' 2>/dev/null | awk -v now=$(date +%s) '{split($2,a,"/"); k=a[1]"/"a[2]; n[k]++; if (!(k in m) || $1 > m[k]) m[k] = $1} END {for (k in n) printf "%-58s %6d | %5.0f\n", k, n[k], (now - m[k]) / 60}' | sort; echo "--- D2 done"
```

**D-066** (the user, 14:54): `mea_out_1electrode` (the Giulia simulations' September detections) is deleted
only after G2 has re-extracted them -- not part of `c8_cleanup.py`; the DUP15HD real-arm archives of before
Stage D are deleted if re-extracted -- group E. **The second patch** `c8_cleanup_E.patch` (sent in this chat):
15,751 bytes, sha256 `d9a84f54936c2deeb15d435c964705b13e6f67d685aa3e6d40088a7a664df5f8`; applies to the tree
`404496d5ed0fe8bc9e32911016a070feb9feadb6` (`459aa72`, on GitHub); tree after it
`4ee52e578581555d83f5f4def1c897d21965afc8`; three files, +120 -12. Group E lists
`/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/extracted` only if `extracted_v2/cohort_manifest.json`
matches its `.sha256` and every folder of `extracted/` holding an archive has its counterpart with
`trace_subregion_*.npz` in `extracted_v2/`; if the SBI config's `cohort.extract_root`, `npz_specs_mea.json` or
`artifacts/specs_real.json` still name `extracted/` -- the C0 flip not done; on GitHub the SBI config at `main`
`9fcbc88` still does `[GITHUB 2026-10-06]` -- it is listed with a NOTE naming them: after the deletion they
point at deleted archives until C0 regenerates them. Checked `[SANDBOX 2026-10-06]`: 16/16 twice; four
mutations of E (no coverage check, no C0 note, the note always, E never listed) each caught; the patch on a
fresh clone of `459aa72` gives the tree above, the files byte-identical; L4 and P3 rehearsed with their STOP
paths (P3 before the push: STOP; L4: PUSHED, again STOP; P3 after: GO, `ALL 16 CHECKS PASSED`).

```
L4:  cd /c/Users/leona/Sbi-extractor && git checkout feat/real-arm-parity && git pull --ff-only && [ "$(git rev-parse 'HEAD^{tree}')" = 404496d5ed0fe8bc9e32911016a070feb9feadb6 ] && git apply --cached ~/Downloads/c8_cleanup_E.patch && git commit -q -m "sim_reextract: c8_cleanup.py also lists the DUP15HD pre-Stage-D archives once every well is in extracted_v2 (D-066)" -m "$(printf 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01RtCUZWW1sF31UDbqxk9r8Z')" && git checkout -- sim_reextract/README.md sim_reextract/c8_cleanup.py sim_reextract/smoke_test_c8_tools.py && [ "$(git rev-parse 'HEAD^{tree}')" = 4ee52e578581555d83f5f4def1c897d21965afc8 ] && git push && echo "PUSHED: the tested tree 4ee52e57" || echo "STOP: nothing pushed, or not the tested tree -- paste this output"
P3:  module load proxy; cd /davinci-1/home/ldellamea/repos/Sbi-extractor && git pull --ff-only && if [ "$(git rev-parse 'HEAD^{tree}')" = 4ee52e578581555d83f5f4def1c897d21965afc8 ]; then echo "GO: the cleanup tool with E is on davinci ($(git log -1 --format=%h))"; source env.sh && conda activate sbi_export && cd sim_reextract && mkdir -p out && python3 smoke_test_c8_tools.py > out/smoke_c8_tools.log 2>&1; tail -1 out/smoke_c8_tools.log; else echo "STOP: davinci is not at the tested tree (HEAD $(git log -1 --format=%h)) -- was the laptop push done?"; fi
```

**[v1.17, 2026-10-06 about 15:25: D2 read; D-067, D-068; the re-run.]**
D2 `[CLUSTER 2026-10-06, the user's paste of 15:10]`: `dvnode012` `state = job-busy` (in service, running a job,
no comment from the administrators), `dvnode022` `free`. Of
`campaign_cadex_rho1300v*/sweep_intel_task0002/topo_00025/iter_0009[9]|0010[01].npz` only v9's exist:
`iter_00099.npz` 734,882 bytes and `iter_00101.npz` 12,368 bytes are valid zips; `iter_00100.npz` is 608,526
bytes, sha256 `0fd6c0c125af5930...`, NOT A ZIP -- damaged on disk, not a read glitch (read twice, by the C8 run
and by D2), and no other campaign holds any of the three files. Written in the last 6 h: only `v12/sweep_cfd_task0011`, 960 files, the newest 189 min old.
L4 and P3 printed their PASS lines (the user: "Everything passed"; `[GITHUB 2026-10-06]` `feat/real-arm-parity`
at `1b23adc`, tree `4ee52e578581555d83f5f4def1c897d21965afc8`).

The user's answers (question card, after 15:10): **D-067** the damaged file is moved aside, not deleted, and
the loss of that one iteration (1 of the 276,347 planned on 10-05) is recorded here and in the log, not in the
record; **D-068** `v12/sweep_cfd_task0010` (left out by name since D-060) goes into the re-run, and
`v12/sweep_cfd_task0011` is finished and goes in with every iteration it holds now. `[reasoning, from the code
as read on 10-05 and the 10-05 counts; S7z shows the real ones]` `RESUME=1` keeps a task only if its manifest
matches the new plan's count (`task_is_complete`), so a task whose raw folder grew since 10-05 is run again,
and `v9/sweep_intel_task0002` (finished at 6129 of 6130 on 10-05, now 6129 planned) is run again, about 4.5 h as on
10-05 (`dvnode022`).

**M2** moves the file only if it is still the one D2 saw (sha256 prefix and not a zip) and the target is
free, writes a NOTE beside it, and prints the remaining count; `Main_quarantine/` is a sibling of `Main/`, so
no campaign glob sees it. Tested `[SANDBOX 2026-10-06]` on a fake tree, the path and the sha prefix substituted: a
half-written zip is moved, noted and counted (2 files left in the fake topology); a second run STOPs; a valid zip
under the guarded sha STOPs and creates nothing. The gap at `iter_00100` shifts nothing: `process_campaign.py`
takes its files by `glob('iter_*.npz')` and each file's index, hence its noise seed, from the file's name
(lines 297, 214-219, read in the ANN copy here at `04977e0`), and the plan counts files (`scan_task`). **S7z** is S7y with
`XT=` empty (no task left out), the form the v1.11 block foresaw for a finished v12 task, character for
character. Then S8y and S9y of the v1.11 block, unchanged.

```
M2:  cd /davinci-1/home/ldellamea/ANN/Phenomenological && F=Main/campaign_cadex_rho1300v9/sweep_intel_task0002/topo_00025/iter_00100.npz && Q=Main_quarantine/campaign_cadex_rho1300v9/sweep_intel_task0002/topo_00025 && if [ -f "$F" ] && [ "$(sha256sum "$F" | cut -c1-16)" = 0fd6c0c125af5930 ] && ! python3 -c "import sys, zipfile; sys.exit(0 if zipfile.is_zipfile('$F') else 1)" && [ ! -e "$Q/iter_00100.npz" ]; then mkdir -p "$Q" && mv -n "$F" "$Q/" && echo "moved here $(date -Is) from $PWD/$F (decisions log D-067): not a valid zip (BadZipFile in the C8 run of 2026-10-05/06), $(stat -c %s "$Q/iter_00100.npz") bytes, sha256 $(sha256sum "$Q/iter_00100.npz" | cut -d' ' -f1); no other copy exists" > "$Q/iter_00100.npz.NOTE.txt" && ls -l "$Q" && echo "MOVED: the damaged raw file is in $PWD/$Q; the task now holds $(ls Main/campaign_cadex_rho1300v9/sweep_intel_task0002/topo_00025/iter_*.npz | wc -l) iteration files in topo_00025"; else echo "STOP: nothing moved -- the file is not the damaged one D2 saw, or it is already moved"; fi
S7z: cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && XT= && PA="--allow-mixed-contract${XT:+ --exclude-task $XT}" && rm -f out/plan_c8y.log out/array_c8y_dry.log out/walltime_c8y.txt out/plan_args_c8y.txt && if tail -1 out/smoke_c8.log | grep -qx 'ALL 26 CHECKS PASSED' && RESUME=1 DRYRUN=1 PLAN_ARGS="$PA" bash launch_sim_reextract.sh plan > out/plan_c8y.log 2>&1 && { [ -z "$XT" ] || grep -qF "[plan]   $XT: left out on the command line (--exclude-task)" out/plan_c8y.log; }; then grep -E '^\[plan\] (replays|tasks|EXCLUDED)|^\[plan\]   seed |^\[plan\]     (DROPPED|kept) ' out/plan_c8y.log; W=$(python3 -c "import json, math; n = json.load(open('plan.json'))['counts']['max_iters_task']; s = math.ceil(n * 3102.1 / 1334 * 2 / 600) * 600; print('%02d:%02d:00' % (s // 3600, s % 3600 // 60))") && echo "$W" > out/walltime_c8y.txt && echo "$PA" > out/plan_args_c8y.txt && RESUME=1 DRYRUN=1 WALLTIME=$W PLAN_ARGS="$PA" bash launch_sim_reextract.sh array > out/array_c8y_dry.log 2>&1; grep -E '^# (tasks|walltime)|qsub|ABORT|REFUSED|DRYRUN' out/array_c8y_dry.log; echo "WALLTIME=$W  PLAN_ARGS=$PA"; qstat -Qf cpu | grep -i walltime; else echo "STOP: no dry run -- the suite is not 26/26, the plan refused, or the task is not left out (out/plan_c8y.log):"; grep -E 'REFUSED|ABORT|names no task|not CAMPAIGN|already have detections|cannot read|^\[plan\] tasks' out/plan_c8y.log; fi
```

S7z's expected lines `[reasoning, from the 10-05 counts]`: `[plan] tasks complete : <about 75> kept as done
(--resume)`, `[plan] tasks to run : <about 19> ...`, 160 replays dropped, `[plan] EXCLUDED <about 207> task(s)`
with no `left out on the command line`, then `# tasks`, `# walltime`, the dry-run `qsub` lines, the `(DRYRUN`
line, `WALLTIME=... PLAN_ARGS=--allow-mixed-contract` and the queue's walltime cap. `dvnode012` is still in
service: PBS Pro gives a user no way to exclude it (v1.16), so the HPC support is asked to check it. `[reasoning]`
Every task asks for a whole node (`select=1:ncpus=48`, `launch_sim_reextract.sh`); if all of them start
together (about 19 against the `%20` throttle, nodes free), at most
one lands there and nothing is queued behind it; a failure there costs one more RESUME round of one task.

**[v1.18, 2026-10-06 about 15:50: the re-run is submitted; the FAIL list after it is this morning's.]**
S7z and S8y `[CLUSTER 2026-10-06, the user's paste of 15:42]`: `[plan] tasks to run : 19 (array 0-18), 63063
iterations; largest task 11728 iterations, 26 topologies`; `tasks complete : 76 kept as done (--resume)`; 255 tasks
compared, 95 distinct seeds, 47 same-seed groups of the one contract `1649221c57628597`, 0 without a seed; `replays
dropped : 160`, 1 file lost; `EXCLUDED 207`; no `left out on the command line`; `# walltime : 15:10:00`, `-J 0-18%20`;
`WALLTIME=15:10:00  PLAN_ARGS=--allow-mixed-contract`; `resources_max.walltime = 300:00:00`. The seed-3000002 group
keeps `v9/sweep_intel_task0002` at 6129 iterations (6130 on 10-05), so the file M2 moves is out of `Main/`. S8y:
the same three count lines, `job : 1807426[].pbsserver01`, `gate : 1807427.pbsserver01`.

Read with the 10-05 counts `[reasoning, from the counts]`: 76 + 17 = 93, the tasks of 10-05, and 19 = 17 + 2, the two
v12 tasks; so `v12/sweep_cfd_task0011` was not among the 93 tasks of 10-05 (D-068's NOT VERIFIED). 93 + 208 = 301
tasks were enumerated on 10-05 and 19 + 76 + 207 = 302 now: one task folder more. 253 compared then, 255 now; 93
seeds then, 95 now: the two v12 tasks have seeds of their own.

**The FAIL list printed after S8y is the old log.** The launcher's `then:` line was run at once; it printed 17 FAIL
names and the REFUSED line of this morning's gate (07:16): the same 17 tasks as in the v1.14 block, and neither v12
task, which the new plan holds. The gate job writes `out/sim_reextract_gate.log` through `#PBS -j oe` and `#PBS -o`
(`run_sim_reextract_gate.pbs:27-28`), so the file is replaced when `1807427` ends `[reasoning: PBS writes the -o
file at the job's end]`, and `1807427` starts only once `1807426[]` has ended.

While it runs, read-only: PROG counts the finished tasks (a task writes `mea_manifest.json` last) -- 76 now, 95 at
the end; the node check shows where each subjob runs (`dvnode012` in its node column is the one to watch)
`[NOT VERIFIED: davinci's qstat column layout]`. Then S9y of the v1.11 block once `qstat -u $USER` lists neither
job. Expected `[reasoning]`: the largest task, 11,728 iterations at the 76 tasks' median 2.30 s per iteration, about
7.5 h (3.2 to 10.1 h at their 0.97 to 3.09 s), so the array ends about 23:10 at the median rate and by about 01:50
at the slowest, if all 19 started at about 15:40; the gate took a minute this morning for 93 tasks. S9y's PASS: `wrote .../Outputs_v2/REEXTRACTION_RECORD.json
sha256 <16 hex>`, `[job] sim_reextract_gate_exit=0`, `record: 95 tasks, <n> iterations, 207 excluded (160
replays); left out by name: none`. A refusal again: paste the gate log whole; a task that failed on `dvnode012`
gets one more round of S7z and S8y.

```
PROG: find /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2 -mindepth 3 -maxdepth 3 -name mea_manifest.json | wc -l
NODE: qstat -t -n -1 '1807426[]'
```

**[v1.19, 2026-10-06 about 16:30: hold the tools still until the record; tomorrow's order.]**
The gate job `1807427` starts after the array and, before reading any output, re-fingerprints the seven
tool files in `ANN/MEA_analysis/`, re-hashes `eap_library.npz` and re-reads the DUP15HD cohort manifest against
the digests in `plan.json`, refusing on any change (`sim_reextract_gate.py:209-228` `[CODE Sbi-extractor@1b23adc]`).
So, until S9y shows the record: no upload into `ANN/MEA_analysis/` (the Giulia worker change of D-071 waits), and
no `plan`, `array`, `gate` or `sim_reextract_plan.py` run in `sim_reextract/` (they rewrite `plan.json`). Reading
is fine (PROG, the node check, `qstat`).

Tomorrow, in order (a reminder is scheduled into this chat for 2026-10-07 09:00): S9y; on its PASS, C1b (below:
C1 of the v1.15 block with its grep widened from `[A-D] ` to `[A-E] `, so that group E's total line shows), read
the LISTED count and the NOTE lines (the one file only `v2/sweep_intel_task0008` holds; the C0 note for
`Deep_bio/extracted`), then C2 with the printed CODE. Then the Giulia G2 (`claude/PLAN_2026-10-06_giulia_G2.md`).
On a refusal: paste the gate log whole.

```
C1b: cd /davinci-1/home/ldellamea/repos/Sbi-extractor && source env.sh && conda activate sbi_export && cd sim_reextract && python3 c8_cleanup.py 2>&1 | tee out/c8_cleanup_list.log | grep -E '^\[clean\] (record|[A-E] |NOTE|NOT LISTED|LISTED|REFUSED|kept)|--delete'
```

**[v1.20, 2026-10-06 about 18:15: after C2, the Giulia G2 -- never before a C8 record.]**
The G2 lines are in `claude/PLAN_2026-10-06_giulia_G2.md` v2, sec. 4. L5a and L5b (laptop) push to GitHub only, so they
may run at any time, tonight included: nothing reaches davinci until a davinci line pulls it. On davinci, tomorrow's
order: S9y; on its PASS, C1b, then C2 (in the background); then G2a (the ANN files into `ANN/MEA_analysis/`, the C8 ones
backed up into `~/ann_c8_tools_20261006/`), G2b (the three ANN tests), G2c (the Sbi-extractor pull and its two suites),
G2d (the Giulia dry-run plan: paste it). G2a-G2c touch nothing C2 deletes, and G2d only lists `Giulia_Astro`'s folders
(C2's group D deletes two of them), so they can run while C2 deletes `[CODE Sbi-extractor@1b23adc c8_cleanup.py:20-45;
reasoning]`.

If S9y REFUSES again: no G2 line. The refused tasks are re-run with the C8 tools and code, as before (S7z, S8y). After
G2a the C8 plan would require the G2 tools and plan `noise_seed_scheme sim`, while the 76 kept tasks carry the C8 tools'
hashes and the `topo_iter` noise, and the gate refuses one record over both (one environment across the run; the noise
check, D-072) `[CODE Sbi-extractor sandbox d1525f7]`. Paste the gate log whole first.

**[v1.21, 2026-10-07 about 09:10: today's order; the laptop pushes are done.]**
GitHub `[GITHUB 2026-10-07, read by Claude about 07:05 UTC: git ls-remote and a shallow fetch]`: Sbi-extractor `feat/real-arm-parity` is at `4e4dff0` (2026-10-06 18:13:29 +0200,
parent `1b23adc`), tree `460d749428c2e7b8b22685614af13fc2952e199c`; ANN `main` at `d0ced88` (18:13:41, parent
`04977e0`), tree `8aa8cc112e140375cb0f09c1052077e672560374`, subtree `hpc/MEA Traces`
`028d786ac55a409efcff897e54884184a6c2db15` -- L5a and L5b ran and pushed the tested trees. Nothing reaches davinci
before a davinci line pulls it, so the C8 gate is not affected. Today, in order: S9y (v1.11 block); on its PASS, C1b
(v1.19 block), then C2 (v1.15 block, with the CODE C1b prints, in the background); then G2a, G2b, G2c, G2d
(`claude/PLAN_2026-10-06_giulia_G2.md` v2.1, sec. 4), which may run while C2 deletes (v1.20 block). On a refusal: no
other line (v1.20 block); paste the gate log whole.

What each should print `[reasoning, from the v1.18 counts and c8_cleanup.py, CODE Sbi-extractor@1b23adc]`: S9y
`wrote .../Outputs_v2/REEXTRACTION_RECORD.json  sha256 <16 hex>`, `[job] sim_reextract_gate_exit=0`, `record: 95
tasks, <n> iterations, 207 excluded (160 replays); left out by name: none`. C1b: the record line (95 tasks, 160
replays dropped); the `kept under .../Outputs` line; one total line per group -- A, the raw folders of the record's
replays (160 if each is still there and passes its checks), B, the old `Outputs/` detections of every task the
record covers (its tasks and its replays) that has one, C, the 10-03 test task under `Outputs_v2_replays_moved/`,
D, the two Giulia partial roots (`extracted_giulia_partial_20261002`, `extracted_giulia_partial2_20261002`), E,
`Deep_bio/extracted`; the NOTE on the replay `v2/sweep_intel_task0008`, which holds 1 iteration file no kept copy
has (D-064), and the C0 NOTE on E; then `LISTED <n> folder(s) ...` and the `--delete <CODE>` line. A `NOT LISTED` line names a folder kept
back and why. C2 runs in the background; `tail -3 out/c8_cleanup_delete.log` ends with `[clean] DONE: <n> deleted,
0 failed; log ...`.

**[v1.22, 2026-10-07 about 12:45: the re-run's gate refused one task -- `v12/sweep_cfd_task0011`, still simulating at the plan; D3.]** **[corrected 2026-10-07: written about 12:30 Europe/Rome, not 12:45.]**
The gate `[CLUSTER 2026-10-07, the user's paste of 12:08: S9y, then the gate log whole]`: job `1807427` on `dvnode003`, 2026-10-06 21:41:29 to 21:42:49, `git=1b23adc`,
`sbi_export`, numpy 2.4.6, 8 workers; the plan `plan.json` of 2026-10-06T15:40:03+02:00; 95 tasks, 288,250 iterations
planned; `read 288442 mea_iter_*.npz file(s) across 95 task(s); 1 task(s) bad`; `FAIL
campaign_cadex_rho1300v12/sweep_cfd_task0011` with `total_iters 3264, planned 3072` and `topo_00008: 192 mea_iter_*.npz,
raw has 0 iter_*.npz`; seven `WARN ... _failures.log from an earlier attempt (every iteration is done in this one; delete
the log if that is right)` -- `v12/sweep_cfd_task0007` (`topo_00001`, `topo_00002`), `v3/sweep_cpu_task0024` (`topo_00002`),
`v3/sweep_cpu_task0026` (`topo_00000`), `v3/sweep_intel_task0021` (`topo_00001`), `v5/sweep_cpu_task0017` (`topo_00000`),
`v9/sweep_intel_task0002` (`topo_00025`); `REFUSED -- 1 bad task(s), nothing written`; `[job] sim_reextract_gate_exit=1`.
No record: Stage C stays open, and no cleanup line (C1b, C2) and no G2 line runs (v1.20 block).

**The reading** `[CODE Sbi-extractor@1b23adc; CODE ANN@04977e0; reasoning]`. "raw has 0" is the plan's count, not one the
gate made: the gate loops over the plan's `iters_per_topo`, compares each topology's `mea_iter_*.npz` with it, and the task's
`total_iters` (what its MEA job found) with the plan's `n_iters` (`sim_reextract_gate.py:112, 146-163`). The plan's `scan_task`
counts every `topo_*` folder, empty ones too (`sim_reextract_plan.py:245-264`). So at the plan (10-06 15:40) `topo_00008`
existed with no iteration file in it, and when the task's MEA job listed the raw folder it held 192; 288,442 - 288,250 = 192,
so every other task matched its planned count exactly. The ANN sweep makes each topology's folder before its pool of workers
writes that topology's iterations, rebuilds `manifest.json` after each topology, and runs until its walltime or its
`--n_topologies` (default 10,000) (`hpc/Phenomenological_finalv1/HPC_main_sweep.py:5-37, 1300, 1557-1558, 1749`; that the
sweep on davinci runs this code is `[NOT VERIFIED]`). So an empty topology folder is a sweep in progress, and a sweep is
finished when its PBS job has ended: D2's 189 minutes without a new file (10-06 15:10) were most likely topology 8 being
built and its first iterations running `[reasoning]`.
D-068's "finished" did not hold at 15:40.

**The seven warnings** are `_failures.log` files of the 10-05/06 attempt (the `dvnode012` failures; for
`v9/sweep_intel_task0002`, the damaged raw file of D-067) in topologies done in full now; the gate puts them in the record
as warnings and refuses nothing on them (`sim_reextract_gate.py:151-160`). Claude's default: leave them, so the record says
those tasks failed once and were re-run.

**What the gate does not see** (the v1.8 sentence, corrected there): it never re-counts the raw folders, so a task whose raw
folder grows after its MEA job has listed it passes with the plan's count, and the record would claim fewer iterations than
the raw holds. D3 makes that check by hand for this run. Proposed, not built: the gate re-counts each raw folder and refuses a
count other than the plan's -- directory listings only, and it would close the gap for every later pass, G2's included. **[corrected 2026-10-07, about 13:10: "after its MEA job has listed it" is per topology. The MEA job lists the `topo_*` folders when it starts and each topology's `iter_*.npz` when its worker reaches that topology (`process_campaign.py:297` `[CODE ANN@04977e0]`); with one worker the topologies run in turn, so growth in a topology the job has not reached yet is caught -- task0011's `topo_00008`, listed about 2.5 h after the job started (v1.23 block) -- and growth after that is not.]**

**D3 -- read-only** (it writes only `out/c8_d3.txt`, through `tee`). Run it BEFORE any launcher mode: it reads the
`plan.json` the gate used. It prints the task's simulation job (the `_pbs_jobid` of its `job_args.json`, `qstat -x -f`:
state, host, start, end, exit, walltime, comment); per topology the raw count now, the plan's, the MEA output's and the
newest raw file; the sweep's `manifest.json`; the task's MEA manifest, `mea_env.json` and its job log's `[mea]` lines
(`~/c8_mea.o<array>.<index>`); then every task of the plan, and every excluded one with counts, whose raw folder now holds a
number of iteration files other than the plan counted (only folders changed since 60 min before the plan are re-counted),
with its simulation job's state; what `--resume` would do with each (`sim_reextract_plan.task_is_complete`, or the same rule
by hand if that module does not import); `qstat -u $USER`; and one verdict line:

| last line | means | next |
|---|---|---|
| `D3: FINISHED -- ... would RE-RUN ...; next: S7z` | every simulation job concerned has ended; their raw now differs from their output | S7z (v1.17 block, `XT=` empty), pasted; then S8y (one task: a plain job, and the gate) and S9y |
| `D3: FINISHED -- ... would KEEP every task as done ...; next: the gate alone` | the raw now matches each output | a gate-only line, given after D3 |
| `D3: STILL RUNNING -- ...` | a simulation job has not ended, or its state is unknown and a raw file is under 30 min old | the user's call: wait for it to end, or leave the task out of this record (`--exclude-task`, its output moved out of `Outputs_v2` first) |
| `D3: UNCLEAR -- ...` or `D3: FINISHED, BUT ...` | a job state unknown with no recent file; an output holding more files than its raw | paste D3 |

Tested `[SANDBOX 2026-10-07]`: `d3.py` (13,150 bytes, sha256 `9e4fef2783b67f762cb3fae20c58e856d484e923857c7fe19b4eff90294b3443`)
on a mock of the 10-06 state -- the bad task (nine topologies, the last empty at the plan and filled after it), a done task, a
run task, an excluded replay, the plan, the MEA job's log, a fake `qstat` that wraps long values -- 34 of 34 checks over ten
cases (finished, re-run; finished, kept; running; the job unknown with a 70-min-old and with a 5-min-old file; another task
grown after the plan; the plan module not importable; an output larger than its raw; a folder touched before the plan with its
count unchanged; a job of the user's in the queue), and the one-line form prints the same lines as the file; six mutations
(the pre-filter too strict, a finished subjob read as running, `qstat -f`'s wrapping not undone, keep and re-run swapped, the
larger-output check removed, the recency window ignored) each failed at least one check. The line below ran as written at
davinci's paths, on a clone at `1b23adc` with its `env.sh` and a stub `conda`: the verdict, the plan module imported, the job
log found under `~`, `tee` writing the same text, exit 0. The line carries the whole source (zlib, base64), so the source is
recoverable from it. Not tested: davinci's `qstat -x -f` text for a finished subjob, and whether the sweep's job is still in
PBS's history.

```
D3:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && python3 -c 'import base64, zlib; exec(zlib.decompress(base64.b64decode("eNrdW+tz27aW/66/Asuup0QjyZLTyWaU+nbSRmk9N3Ezdrr37ii6HFqELMbiIwRly7fN/dv3dw4AvkQ77e5+Ws/EIQng4LxfgL8S0dNxfi9GI/Hjc1GoUbFLhyJcl6oQ5UaJ67BUIluLk8nJs9F0Mpo8EyfT2bcnM8wNo1GWbu/Hg6+EP5U8PUuVuAojUYb6ZibiUgsdJ7ttWMZZKj5mV0P+VoR3Ikwj8Xb+UqyyXYpPOe2X5dk2u75/QQBPpFC3qrhnUIQBgc+3YSp8WmrG1H613UUq4n3v4nJjwcmZiDKl3WaARz/rbBthmzS7Exs8ilBE8XqtCpWWIt0lVxjDPjFIN/iu4y1glBvs6Ta3kHgXFX0v3jsWrbIkDwtMV+FqI7Jdme9KgxEtfffm5fnX2tFK+KdEgIUGrtsRmkvMMZjqodAZvsUgxIytNmp1AwwNdWlWiiS8Ucyvp1LcbcISggQWuwTsyHagMcoMFowWs7KkWSsQda0iXvmtZPYBnyhelWIbpwobl2FRxum18F49nQmPZPy3AqzhXTcYeGEQCrdbkpxSBkFLeJyWGT0fr54H0K9yXxKAd7tCiZeXP56dDcWb14J0R/ibfDXK78tNlo6Yh6UcD+IkzwoQCcaWMUgJtQii0n3+qLPUPWfaPRXKPendVV5kK6WrMX1fPRK8weDn+cVcnGL1WKW3cZGl42tV+iA1oBFvKLzjKLyN01U8mh5vskQdQx7bbZio8LhQeaaPL6/ikdqXRbgqs+IYSh4Uyr57cvDml58uDXwQtBmrfQ6Z77Qq/J4taTJt+S9PysH5y7cPYPb+5eVfadoqBJfi6zRYhZHaB8Ummz6dTG6nJ8f6Tqk8WK2jgAQ9mUynQOVi/uP8/H3w9uwcYJ9OWOG+guqzmkHBjX4VagUrGNYaC1slHYBm79IbWAycgtNeLa4UacYd9KFU6eDty4ufzs7tDs/cDlaFnaaJXQ77NpttM6y+UuusULVV3wGuhShgSJVRYGmZZYP5+av5K8D3vdfEhL97kjZ598OleFdkM/EatKSx3mA2e5m/1+9QB3x6YT0GlBf6aiiLjQmpFB5kcPZ+fgH4hRqTGoIxfuH9g3xB4H+InsgP4zT/57+Dn+SzutOgFkHP1PNf/oappHFj+uXLwWAQqTXsVKX+Xs4GxKl4zUjszZtxB+WuSIU38gbN1yMt/KNIJDH4c51JTxwJn2HrslgzfO8oGWHG0c+zo7dgEg9uM5goj+6lHFZ7PPQDw/ULsD3yfcJ+JPZSHEOo44mUDv1tFkY+6bWloCzua+RZgbIcBPIM0pb1Ztba1xJEZjxmWOuN5Anw5iovhf/L5bwosmIo/jPc7hQ/MyCMH3DpNy/YpRSKwqut8uCpjgrmDOYO5WeLMWKav0qiPnwLSKh2GGM7E8oe5oCvAuPQTt8XOzjFEhbuHsFSjJ1OTyayi5NfQCYRRkVWCA+a+sR+UkVhP/XT20Dksnp8nH5vxW6eVAi4iyM9E5YDHrz2xyw2lA9psRMg21UAB+A7lnie9xtF3yCKi5nw28Ew+IYUeoiQdac0Ig6RLj/TKIUN8iMcV4zBjwGKQRL5p+K3z00ld/4w1tin2rxBDRYNai3SY72C3+SpRH9c1guArSbRwaeryFfjFL4ZKBQCZp1ShMSWChsRRb7kkGsmjTmyadrB95hmcrsmPUAgi2g5Aa+3gutLiJbJUMAOOppe4+iIY55Dg8pIHmDttrFIzg7s0WFNntmiTY5pnITlamPJlLNeM07Fk1Mx7R1i9JNw7ydw8cyB0pf4LzCyrNaA/YsyWpKbZaLloCObpv7g/QH9qdWncoysQpXO2Czh/7fa/E/1oxYC0jh/2otdQz0Qj9ra8YDQEAMDDns+LJbIq0XnU7CPgRKFUxMased5Rnkgfo0oIyRkkrC4ly4VpzWcSm5DOIW7Iiv5WxAW13rMvt2J9GMIWtjRt1hg0UAwby1zbP0YR8SCsvA/hiYHCvIrHWBuHMGlNtyo1RIsOFCMj0SVIcRLM1FDIH62toXnJK9Z7dUIKsNqJ+SxFPsRIxbeJ2IUjY32/HuN39huKcdIELfhCuH4Q/qhNKs5I+IVYoT4X4TQQtYP+FTm7hPx/uUPRp1q9SfJ8yjpFlJonW/jkj5ov2EDIN/DGo9m0WBbm26G4pbYjwFKFOKcDJ/g+LxoKKay7Z0WN6R6t46zXqU3HnMZm6wfZbOlck+EcjRivvo8pQpJFTW+lIvZyWSyNFjcqHvNSV69K1apvVoFm0wzMzW5LHqAGEv3PN/HJc/faa+d5HgXSme7YqWCN7Eux3eoV9yiwo7oAJl51BpCbpcgH/ZatsQkrBcN3JaY+sLRRAnaqSP3ZsicXMyQHywlS/KGeMcEgq38sq4DcpJvVamCq/sA6XLkk2es7bNVXQSULY9pQgBH4FYOgSTjFAmfqDHFVbw2Jppk0W5LxVpKIryCQnExpCJZGWmCFLwuWIyQsMfCgwchM2VayZ1jYrzGXtZYTSKT3v6xxZjYXGct168dPDs2bGF8W+c7VstD5/863GrFHyPnZwhAT6pnszPKcSNj5mVWhtsgguZC6qMpHPLpqTgYpwimacJJPcHQl9qxpTyIu4R/A1IaUNDQnW0cFDO2lO288P19rg4z4QcYYDSJIkCQravqoi4ejq1m7hdVBUliwSuXjUSCAZGEEJ9dDgGsNhS8qCw2yKGWtkJJUc+WPrKiepArOSsEjxW1LW2PCNaVJ6EZNTl5QSzhfsOv5z++mb+8oK4UMK8AuYrNuebKuxAGQwZ36L6lPEjQp4fK4ToFB6ZGIZraN9XM2uQ22R2IpcEee/wjVus1xT3n/6jvZFN9ihlp9imciR/ezFHNEzeo+UCGa7ovmrKXgkzb9IewNIdXhflf3QtyJA8i3fU4wLcN0AIQ/iEZVGHIusjiPVZgNzmf04YQ7DevxxQta0+prTN2bZ7xusiSWGdwlkhofLtccu0M35YgdP0p86g2ochkcCDtszgurC4uBw3dg1m+ejqsW5tDKNhY1PrnGhQc2EyX09fStOUM113PdEjjrj3pNbxDT9X+X6Nm4Y5azZI+FFtlPKmW5rlmbgV6KBbwG1bHIU3OaJFCNDyB2HMA2nMeQdCq7Jq7TdYY7dov26OzwrBqzdaWDvIIZo/FDUxbyDWKXZPYyAUI2+0XtNzIBAni0NYDZe20XDwpG9HlC2nmQylmipE0z7GsSvsoJ9jeqihIg7usuCHfDta74TTIwyJMdJCjpDHjVrupfX0KgOIbhgjWxhoOsgxTJIK0D1hpYlrze57bAQUPLrzvPZf5Uv91yCk3IdfM3GVTXRfR0yXrYlgUIcJ9Gqk93l/UXTXXQiNlNd31WGnWzarNrTvaSQIAew3FDDhgwMSIshGs7JsLgIeIiU7/f8Y1w5G2zWXCSgvOvF8IA9Wgxx9IzSoMW6cDwseYFY4gcmWHAKocqDz43qslF9gtA+TsDYFyVklq15ZyjQt9x+ZDpyu9ZJogRNKy+eIKeot/+Yo9rWu2kPSGjeIZ/2TFaGYj6xVtDsx/+9y71Wg6BYOe23/04vY3ZSlxiJPb8I4OOuiRFCFlT+FR89IU3/RmGzo0tS7SvSoJgccH/otlp8q1tbMG1sVKit8FPWXVU75q5miud1KsmMwSmu2bQmHSCMt/jMZOikWOlOLWyOM8krwSQTa2lBL3qy0xR5JIOh+4D5uUzQQBkBrTJlL8RfgpadOk03Vh9owRbhXSdUr96URJk2HZ5gZnrNqk/vgKJnsO5WxFBT70yQB+TM6R+TedRk7ESEg9dg3+7WKyZPC3JBZw+JaiIGpD2TZNMztrjNstdfKIx+zk+X1IcsZAyePXWrSmz8Qup6hO5sYGD/407NslH2ymOZ0zhVueg8raTI5ivQqLw8ipE2MtNXiSo/vaNNwqx4qCrAjsJt3JlNzTnl63M96dtA7j7Q6xIagRc0xMEqi46mUkNKG/aKJo/sjsukp6iOu1Hc9Eo0ZhXtc1Db9af81OlfyxPdhp+V/muXGp/oE7TZLDWggEtD9zCdX46qIEfVLmk82dmp+4pu/yvprv4q/TVvWplcv1nzq0WEoZAvcMdldILDWFEjqL9GTvuQRmkb8zrRLTnGi1YNYbcg/NTsqyWo84SWv14sSYpGZPCYC0Amhp8iQnnADoxXRJ9Z+JrV4NwxCIr4vRdOmaNAtPwszd25jfCChtyJ6uYohNi+1hQiONC7V2+R5tQTj0RXeX1hHAevU271b2dFxJLZLnAXR1nCFBtvEHwNnxNKEuO761XdBv845XreW4RWqkiBB96tmGmtcrNfq5zspKcoWVTY/oCghOhcVq4xfehwWw/7D8nf8fMcoflsK3Svc7dTRQ78S3XIJAhwhSQ95GXlvsSntbIeNRiu9OxXMjGbwuZs+W4olYeOPx2KMn+jY6mbUBHSZNZN9kqPBllN8Re8Grg2MERx4wOWRKG6rLEmhJs+PGmgNsZ38OI9aUNR0UOtxcfl/d26jvMRyko9XtDHMPg+8tdOd8b+pKaKvm/qUubSVE/Gz07B8qiUyvvC+3amm6kcSKqwzfVowj0TjS/sacfnIvzQyzcN1JkDvebiVKXGkx6jVbacq+W8bUtWrTm7EXtCc2dIBAxzV5fV7DW+S0xSJi7eo5WWjka9i0w4DlsveUqGWf1PTJG8b72NnsIeZ0dvzgz1ei7s5QBnBNDseIHSrRdBeJ+I5E04a/ytIyTnfqEfhpJlQKloowikyWUagku6XLADEKr+7dgzpbze6Cds4e1Rygu0IkZkr2jqLRX46iKqHrYXEjhxwauHUCPOHsdzHp6Rr2ptm83OXXPXtJ2XeO+NCm4t9Ov4DwsikDIrsjAKPyLvn190OeNGR9XdxSeKuyUoOESzyXJIoFTGfpmNKT4TTuP8EzZLC1+kaWURK6wGVubiEcFFlSCfJr05bJ1vTbNmU62Qz5aUuA7amwnbokw16CCorsTnds2lFpqiaLPxu6AVcz6SPm6r2p3/etAv7AAdSSM6cX5CSM07KnGeakq7rn5lO5S8VmWfh2IkyJckU+Z5KL2beT5UN1lRaLI72cmYLb1H58A6i3tqr7R0OL25COi+xhB7GCghvfbfAEhTcXBWmIsp1ntqtRVVpmS/lg3Sd67gu2jpD2VVnPnLVhrAbXFF5TO0kWLalJ2dcAWziSi4U7synoexPscvb4bqXp3PRYQvGAGRSEjQueX7jEBz1PbFufr2neoPZiLR12NPWA4i4drePDrsZZfkC1dqxXJrn/016YQqXTW3MflK83qhFHUbrftUVqGq9jTgUaSpFdcQez6u/NyF81j5JcaTvrqYPbHkfybFOLzFg9jQJ8rrbzmYlxfQjnMwK2J8eMls0637edFe6hpe3GmvS+eMGq76djbbx/q36vvrTPiZrmU2kMoUZeYmZ152I+uvj1nG3efPjrfP6OEmlbIXb84ya7ax5qGvrZ7uy9T685TLxrjzaU+Vsp7rNdQWZM+Rbgcumxqa6a8sRPLnNvZbTdM/YdfnfvRP56Ob8wbm8pW2fj/WVaGjAizWsVbrdPtjow9ygK7x8foic253cweljeIk749uB7JykGkcMyG8qDo/xPqO9QAswedNMuTzdLuQwhg1i0fXJl4rLP6oOuxQ8OEgRf75uZsClO9876+cIly9bOq+bY0EGP5o6g/fCdaNw1pbTZMt5eID2g4H+BPdeyHazoIPnPopbSrRW6b4CleOTsAUKjch1JBAlY82XU2NwA/7RTO7awF5wsp8pehkn4YIjcGN17DoWL8TaPAW3hPRK6jnuoNISzNaOdNmS68GRk33skc/n+7M0bAdM+Pzv/yRzMzNzdbLppa67yi3tVkjrVJtu+HUG9n/qqL/eL2okBOycQwEZuJenZgH4njfBoHJ+ISwZh10LeggS75kvHStxLLO0fG/BLOw+gnI5Pnhw86muU5vYyEzh8iMBKKyxJ0tHUpYEJsOCbFHCrtxf/12fnZ5c/z18NxQ+/vm/efzfYmpZwYkqNMHV/jUAJDZHcJsCeu9SC4m1lExF2xo8iUnGyxTxgEsKD8D3nF93cwsWH6jTT5Yf7Ekr2H//sTQ4pkJrQcBgshmxVFd5a/V9jzAGs8bchLpZ18fR89+cU5J/Zt7f/PkM6Mqs/dAm3AGS0nolo3qGYDAYDCCHgY8og4BZeENBtiSDwZtXdCLWPS9/coZCD/wboqk2y")).decode())' 2>&1 | tee out/c8_d3.txt
```

**[v1.23, 2026-10-07 about 13:10: D3 -- task0011 is still simulating; D-073 -- it is left out of this record; M3, G7, G8.]**
D3 `[CLUSTER 2026-10-07 12:43, the user's paste of 12:44]`, its verdict `D3: STILL RUNNING -- campaign_cadex_rho1300v12/sweep_cfd_task0011,
job state R, newest file 10-07 12:34 (9 min ago): nothing to re-run yet`. The task's sweep job is `1562195[11].pbsserver01`
(`job_state = R`, `exec_host = dvnode208/0*192`, `stime` Mon Oct 5 19:57:07, `Resource_List.walltime = 70:00:00`,
`resources_used.walltime = 40:43:30`); `n_topologies` 80, 384 iterations per topology (192 workers x 2). Raw now, per topology:
`topo_00000`-`00014` 384 each (5,760), `topo_00015` created and empty; the plan's 3,072 in topologies 0-7 and 0 in `topo_00008`;
the MEA output 3,264 (192 in `topo_00008`). The sweep's `manifest.json`: updated 2026-10-07 12:34:57, 15 topologies completed or
partial, 5,760 runs. The task's MEA job: started 2026-10-06 15:40:32 on `dvnode022`, 1 worker, `[mea] 9 topos, 3264/3264 iters
processed in 9421.6 s` (2.89 s per iteration). Of the 302 raw folders the plan enumerated, 1 differs from the plan: this one.
`qstat -u` lists three of the user's jobs: `1544754[]` and `1544755[]` (`meama_cam*`, `intel` and `cpu`, 48 cores, 24:00, `B`;
the user: unrelated, ignore) and `1562195[]` (`asn_campa*`, `cfd`, 192 cores, 70:00, `B`) -- the sweep array; task0011 is its
subjob `[11]`, which `qstat -u` does not list (an array is one row; `qstat -t '1562195[]'` lists the subjobs).

**The reading** `[CLUSTER; reasoning]`. Each topology is two waves of 192 simulations, and a wave's files appear when it ends:
`topo_00007`'s newest file is 10-06 11:59, so at D2 (15:10) the sweep was 191 min into `topo_00008`'s first wave -- the quiet
interval D2 saw. The MEA job listed `topo_00008` after about 8,870 s of the 3,072 iterations before it (at 2.89 s), about
18:08, and found the first wave's 192; the second wave's files end at 20:25. At about 2.9 h per topology the job reaches
about 25 of its 80 topologies by the end of its walltime, about 2026-10-08 17:57 `[reasoning]`.

**D-073** (the user, about 12:50, on a question card; "However leave it out for now"): the record is made now over the 94 tasks
that passed; task0011 is left out by name and goes to a later pass. Can it join THIS record later? Only by the route of the
v1.9 paragraph (record renamed aside, `RESUME=1` without the exclusion) and only while `ANN/MEA_analysis/` holds the C8 tools,
i.e. before G2a; after G2a it needs a root and record of its own (log D-073).

**The lines.** M3 moves the task's partial C8 output (3,264 files) out of `Outputs_v2` into `Outputs_v2_left_out/`, beside the
root -- moved, not deleted, and out of the record's root, so the record does not sit over detections it does not name; it runs
only if D3's last line names the task as still running, the folder is in `Outputs_v2` and the target is free. G7 is the dry run:
only once the folder is gone from `Outputs_v2` and the suite log says 26/26, the plan with `RESUME=1` and `--allow-mixed-contract
--exclude-task <the task>` through the launcher's `gate` mode with `DRYRUN=1`; it goes on only if the task is printed as left out
AND nothing is left to run (the gate alone cannot run a task: a task left to run would be refused); it saves `PLAN_ARGS` for G8.
G8 submits the gate alone with them (no array, no dependency). Then S9y (v1.11 block), once `qstat -u $USER` no longer lists the
gate. **No launcher mode between G8 and S9y** (the gate job reads `plan.json` when it starts).

```
M3:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && XT=campaign_cadex_rho1300v12/sweep_cfd_task0011 && O=/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2 && D=/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2_left_out && if tail -1 out/c8_d3.txt | grep -qF "D3: STILL RUNNING -- $XT," && [ -d "$O/$XT" ] && [ ! -e "$D/$XT" ]; then mkdir -p "$D/${XT%%/*}" && mv "$O/$XT" "$D/$XT" && echo "MOVED: $O/$XT -> $D/$XT ($(find "$D/$XT" -name 'mea_iter_*.npz' | wc -l) mea_iter files)"; else echo "STOP: nothing moved -- D3's last line does not name $XT as still running, its folder is not in Outputs_v2, or it is already moved"; fi
G7:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && XT=campaign_cadex_rho1300v12/sweep_cfd_task0011 && PA="--allow-mixed-contract --exclude-task $XT" && rm -f out/gate_c8g_dry.log out/plan_args_c8g.txt && if [ ! -e "/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2/$XT" ] && tail -1 out/smoke_c8.log | grep -qx 'ALL 26 CHECKS PASSED' && RESUME=1 DRYRUN=1 PLAN_ARGS="$PA" bash launch_sim_reextract.sh gate > out/gate_c8g_dry.log 2>&1 && grep -qF "[plan]   $XT: left out on the command line (--exclude-task)" out/gate_c8g_dry.log && grep -q '^\[plan\] tasks to run    : 0 ' out/gate_c8g_dry.log; then echo "$PA" > out/plan_args_c8g.txt; grep -E '^\[plan\] (tasks|replays dropped|EXCLUDED)|: left out on the command line|qsub|DRYRUN' out/gate_c8g_dry.log; else echo "STOP: no dry run -- its output is still in Outputs_v2 (run M3 first), the suite is not 26/26, the plan refused, the task is not left out, or some task would still run (out/gate_c8g_dry.log):"; grep -E 'REFUSED|ABORT|names no task|^\[plan\] tasks' out/gate_c8g_dry.log; fi
G8:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && if grep -q '^(DRYRUN' out/gate_c8g_dry.log 2>/dev/null && PA=$(cat out/plan_args_c8g.txt) && [ -n "$PA" ]; then RESUME=1 PLAN_ARGS="$PA" bash launch_sim_reextract.sh gate > out/gate_c8g_submit.log 2>&1; grep -E '^\[plan\] (tasks|replays dropped)|: left out on the command line|^  gate *:|ABORT|REFUSED' out/gate_c8g_submit.log; else echo "STOP: nothing submitted -- run G7 first"; fi
```

Expected on davinci `[reasoning, from the 10-06 plan and D3]`: M3 `MOVED: .../Outputs_v2/campaign_cadex_rho1300v12/sweep_cfd_task0011
-> .../Outputs_v2_left_out/campaign_cadex_rho1300v12/sweep_cfd_task0011 (3264 mea_iter files)`. G7 `[plan] tasks to run    : 0
(array 0-0), 0 iterations; largest task 11728 iterations, 26 topologies`, `tasks complete  : 94 kept as done (--resume)`,
`replays dropped : 160 ...`, `EXCLUDED 208 task(s)`, the task's `left out on the command line (--exclude-task)` line, the gate's
`qsub -v PLAN=...,ENV_NAME=sbi_export run_sim_reextract_gate.pbs` line and `(DRYRUN -- nothing was submitted.)`. G8: the same
plan lines and `  gate  : <id>.pbsserver01` (the launcher's `then:` hint after it is not needed: S9y reads the log). S9y: `wrote
.../Outputs_v2/REEXTRACTION_RECORD.json  sha256 <16 hex>`, `[job] sim_reextract_gate_exit=0`, `record: 94 tasks, 285178 iterations,
208 excluded (160 replays); left out by name: campaign_cadex_rho1300v12/sweep_cfd_task0011`. The gate log will also carry the
seven WARN lines of v1.22. Then the order of v1.21: C1b, C2, G2a-G2d.

**Rehearsed** `[SANDBOX 2026-10-07]`, 14/14, twice from scratch, on the C8 suite's fixture (Sbi-extractor `1b23adc`, the ANN tools at
`04977e0`, subtree `6e01860c`; the real plan, launcher and gate; the suite's fake `qsub`; a stub `conda`), the three lines run as
written with the `sim_reextract` path, the `Outputs_v2` path and the task name substituted: the array run over every task, then one
task's raw folder grown by a topology and an empty one, as a sweep in progress, and D3's verdict naming it. Held: M3 STOPs on a D3
verdict that is not STILL RUNNING for the task; G8 before G7 STOPs, nothing submitted; G7 before M3 STOPs; M3 moves the folder
(4 files) out of the root, and again STOPs; G7 STOPs while another task's raw has grown (`tasks to run : 1`); G7 prints 0 to run, 3
kept, the task left out, the gate line, `(DRYRUN`, and submits nothing; G8 submits one gate job, no array, no dependency; the gate
passes; the record holds 3 tasks and names the task as left out by name; S9y prints `record: 3 tasks, 7 iterations, 5 excluded (0
replays); left out by name: ...`. Not checked: davinci itself; the suite log's 26/26 was written by the rehearsal, not run.

**[v1.24, 2026-10-07 about 13:15: the C8 record exists -- Stage C is closed.]**
`[CLUSTER 2026-10-07, the user's paste of 13:13: the record, cat whole]`: `Outputs_v2/REEXTRACTION_RECORD.json` written 2026-10-07 13:11:04 +02:00, sha256 `7a55c96cbda6a1a6`, from the plan of 13:09:09: 94 tasks, 285,178 iterations, 285,178 files read, 208 excluded -- 160 replays (1 file lost), 46 empty sweep folders, `v7/sweep_cpu_task0009` (no `manifest.json`), and `v12/sweep_cfd_task0011` left out by name (D-073); `--n_side 3 --pitch 60.0 --edge 25.0 --fs 10110.09` (`n_e` 9); the cohort manifest `c925c5106137fe82`; the tools "ANN main C8 (2026-10-01, ...; tree 6e01860c)"; the library `b50814ff27d3c496`; one environment, `sbi_export`, python 3.11.15, numpy 2.4.6, scipy 1.17.1; the contract mixed on `simtime`, `log_params`, `param_bounds`, `manifest_version` (D-059); 254 tasks compared for replays, 94 distinct seeds, 47 groups; 7 warnings, the `_failures.log` files of v1.22. Every value is the one the v1.23 block expected; `hosts` lists the 19 nodes the tasks' outputs ran on, `dvnode012` among them (the task that passed there in the first attempt). The left-out task's partial output is in `ANN/MEA_analysis/Outputs_v2_left_out/campaign_cadex_rho1300v12/sweep_cfd_task0011`, outside the record (D-073).

**From here** (the v1.21 order): C1b (v1.19 block) lists what the record makes redundant and prints a CODE -- the record line (94 tasks, 160 replays dropped), one total per group A-E, the NOTE on `v2/sweep_intel_task0008` (1 file only it held, D-064) and the C0 NOTE on E, `LISTED <n> folder(s)` and the `--delete <CODE>` line; paste it before C2. Then C2 (v1.15 block) with that CODE, in the background; G2a-G2d (`claude/PLAN_2026-10-06_giulia_G2.md` v2.1, sec. 4) may run while C2 deletes (v1.20 block). The hold-still rule of v1.19 ends with the record.

**[v1.25, 2026-10-07 about 14:15: C1b read; C2.]**
`[CLUSTER 2026-10-07, the user's paste of 14:11: C1b]`: C1b listed 376 folders, nothing deleted, CODE `681c30382046`: A, the raw simulations of the record's 160 replays, 160 folders, 438,064 files, 136.6 GB; B, the old `Outputs/` detections of the tasks the record covers, 214 folders, 340,580 files, 50.6 GB (kept under `Outputs/`, not covered: `v7/sweep_cpu_task0009`); C, the moved replay detections (the 10-03 test task), 1 folder, 1,336 files, 0.2 GB; E, `Deep_bio/extracted`, 1 folder, 420 files, 0.3 GB, all 35 wells re-extracted in `extracted_v2`; no D line (the two Giulia partial roots); the NOTE on `v2/sweep_intel_task0008` (1 iteration file lost, D-064) and the C0 NOTE (the SBI config and `artifacts/specs_real.json` still name `extracted/`). Read against the v1.21 expectations `[reasoning, from the record and c8_cleanup.py]`: A is the record's 160 replays; B is the 215 sweep folders of `Outputs/` (207 of v1-v8, 8 of v9) less `v7/sweep_cpu_task0009`, which the record does not cover; C is the 10-03 test task; the left-out `Outputs_v2_left_out/` is in no group; the missing D line most likely means the two partial Giulia roots are already gone `[NOT VERIFIED]`. After C2 the r2-era bank `SBI_export_r2` can no longer be rebuilt from detections (D-065), and until C0 the SBI config and `specs_real.json` name deleted archives (D-066). C2, the v1.15 line with the CODE (runs in the background; `tail -3 sim_reextract/out/c8_cleanup_delete.log` ends with `[clean] DONE: 376 deleted, 0 failed; log ...`):

```
C2:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor && source env.sh && conda activate sbi_export && cd sim_reextract && nohup python3 c8_cleanup.py --delete 681c30382046 > out/c8_cleanup_delete.log 2>&1 &
```

G2a-G2d may run while it deletes (v1.20 block).

**[v1.26, 2026-10-07 about 15:10: C2 done -- the C8 run is complete.]**
`[CLUSTER 2026-10-07, the user's paste of 14:56: C2's log tail, G2a-G2d]`: C2 finished: `[clean] DONE: 376 deleted, 0 failed` (log `sim_reextract/out/c8_cleanup_deleted.txt`). What C8 leaves on davinci: `Outputs_v2/` with the record (v1.24), `Outputs_v2_left_out/` (task0011, D-073), `Outputs/` holding only `v7/sweep_cpu_task0009`; `Outputs_v2_replays_moved/` and `Deep_bio/extracted/` are gone; the replays' raw folders are gone; the C8 tools are in `~/ann_c8_tools_20261006/` since G2a, which put the G2 tools in `ANN/MEA_analysis/`. Still open from C8: task0011's later pass (D-073), the C0 flip (D-066's NOTE). The Giulia run continues in `claude/PLAN_2026-10-06_giulia_G2.md`.

What S4b cannot cover, S6 does: the gate checks one environment across the run, not that it
is `sbi_export`, so the test job's `env activated: .../envs/sbi_export/bin/python3` line is
the check of D-023 on a compute node.

**The lines, as given** (the exact text the rehearsal ran):

```
S1:  module load proxy; cd /davinci-1/home/ldellamea/repos/Sbi-extractor && git pull --ff-only && source env.sh && conda activate sbi_export && cd sim_reextract && echo "HEAD $(git log -1 --format=%h)" && python3 -c "import os, hashlib, sim_reextract_plan as P, cohort_manifest as CM; T = '/davinci-1/home/ldellamea/ANN/MEA_analysis'; f, lab = P.fingerprint_tools(T); print('tools      :', lab or 'NO KNOWN STATE'); [print('   %-26s %s' % (k, f[k][:16])) for k in sorted(f)]; want = {'smoke_test_mea_env.py': '949fc0d98b181900', 'launch_mea_array.sh': '41be351b08f3a6e9', 'launch_mea_per_campaign.sh': 'fe046e1c12f9b7f1'}; h = {n: (hashlib.sha256(open(os.path.join(T, n), 'rb').read()).hexdigest()[:16] if os.path.isfile(os.path.join(T, n)) else 'missing') for n in want}; [print('   %-26s %s' % (n, h[n])) for n in sorted(h)]; print('library    :', 'present' if os.path.isfile(os.path.join(T, 'eap_library.npz')) else 'MISSING'); m = CM.read_manifest('/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/extracted_v2/cohort_manifest.json'); print('manifest   : sha256', m['_digest'][:16]); fz = os.path.join(os.environ['ARTIFACTS_DIR'], 'cohort_manifest', 'cohort_manifest.json'); print('frozen     :', CM.read_manifest(fz)['_digest'][:16] if os.path.isfile(fz) else 'not yet'); print('Outputs_v2 :', P.existing_output('/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2') or 'empty or absent'); print('S1: the ANN tools are at the C8 state -- skip S2, go to S3' if (lab == P.REQUIRED_TOOLS_LABEL and h == want) else ('S1: UPLOAD NEEDED -- do S2 first' if lab else 'S1: STOP -- the tools are in no known state; paste this output before anything else'))"
S2:  cd /davinci-1/home/ldellamea/ANN/MEA_analysis && tar -xzf /davinci-1/home/ldellamea/c8_ann_upload_20261002.tar.gz && chmod +x submit_mea_array.sh launch_mea_array.sh launch_mea_per_campaign.sh && sha256sum submit_mea_array.sh smoke_test_mea_env.py README.md launch_mea_array.sh launch_mea_per_campaign.sh | cut -c1-16,65- && [ "$(sha256sum submit_mea_array.sh smoke_test_mea_env.py README.md launch_mea_array.sh launch_mea_per_campaign.sh | cut -c1-16 | tr '\n' ' ')" = "1575bb80f87c0806 949fc0d98b181900 10ed10ed0cfbabcb 41be351b08f3a6e9 fe046e1c12f9b7f1 " ] && echo "GO 2: the ANN files are the C8 state" || echo "STOP: S2 did not complete -- the upload is missing, or the hashes above are not the C8 ones"
S3:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor && source env.sh && conda activate sbi_export && mkdir -p sim_reextract/out && rm -f sim_reextract/out/smoke_mea_env.log sim_reextract/out/smoke_c8.log && { (cd /davinci-1/home/ldellamea/ANN/MEA_analysis && python3 smoke_test_mea_env.py) > sim_reextract/out/smoke_mea_env.log 2>&1; tail -1 sim_reextract/out/smoke_mea_env.log; (cd sim_reextract && ANN_TOOLS=/davinci-1/home/ldellamea/ANN/MEA_analysis python3 smoke_test_sim_reextract.py) > sim_reextract/out/smoke_c8.log 2>&1; tail -1 sim_reextract/out/smoke_c8.log; }
S4:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && rm -f out/plan_c8.log && { tail -1 out/smoke_mea_env.log | grep -qx 'ALL 14 CHECKS PASSED' && tail -1 out/smoke_c8.log | grep -qx 'ALL 23 CHECKS PASSED' && DRYRUN=1 bash launch_sim_reextract.sh plan > out/plan_c8.log 2>&1; grep -v '^\[plan\]   [^ ]*/' out/plan_c8.log 2>/dev/null || echo "STOP: no plan was made -- the two suites of S3 did not both pass (out/smoke_mea_env.log, out/smoke_c8.log)"; }
S5:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && source ../env.sh && conda activate sbi_export && rm -f out/test_c8_submit.log && { grep -q '^\[plan\] cohort manifest : .*  sha256 c925c5106137fe82$' out/plan_c8.log && grep -qx '\[plan\] EXTRA_ARGS      : --n_side 3 --pitch 60.0 --edge 25.0 --fs 10110.09' out/plan_c8.log && grep -q '^\[plan\]  *ANN main C8 (2026-10-01' out/plan_c8.log && grep -q '^\[launch\] plan only: [1-9][0-9]* task(s) in tasks.tsv; nothing submitted$' out/plan_c8.log && bash launch_sim_reextract.sh test > out/test_c8_submit.log 2>&1; grep -E '^\[launch\] test task|^  job|^ABORT|^REFUSED|^watch' out/test_c8_submit.log 2>/dev/null || echo "STOP: not submitted -- the plan is not the expected one (out/plan_c8.log)"; }
S6:  cd /davinci-1/home/ldellamea/repos/Sbi-extractor/sim_reextract && id=$(sed -n 's/^  job   : \([0-9][0-9]*\).*/\1/p' out/test_c8_submit.log 2>/dev/null | tail -1) && if [ -n "$id" ] && [ -f /davinci-1/home/ldellamea/c8_test.o$id ]; then grep -HE '\[mea-array\] (env activated|WARNING|python|numpy|scipy)|\[mea\] .*processed in|Traceback|Error' /davinci-1/home/ldellamea/c8_test.[oe]$id; grep -E 'tasks to run|EXCLUDED' out/plan_c8.log; else echo "STOP: no log of the test job yet (job ${id:-not found}) -- check qstat -u \$USER"; fi
```

(The `Sx:` prefixes are labels here, not part of the commands.)

**After S6.** **[superseded 2026-10-03 by the v1.8 block above: S7x and S8x carry `PLAN_ARGS` (both flags), `RESUME=1` and the WALLTIME computed from the plan; S9x reads the record.]** WALLTIME for the array = X s / b iterations x M (the plan's
largest task) x 2, rounded up, as `hh:mm:ss`; then, in `sim_reextract/` with
`sbi_export` active, `DRYRUN=1 WALLTIME=<hh:mm:ss> RESUME=1 bash
launch_sim_reextract.sh array` (the array line `-J 0-<n-1>%20` and the gate's
line), then the same without `DRYRUN=1`; once `qstat` is empty, `cat
out/sim_reextract_gate.log` -- PASS is `wrote .../Outputs_v2/REEXTRACTION_RECORD.json
sha256 <16 hex>` and a last line `[job] sim_reextract_gate_exit=0`. Claude gives
the exact array line once S6's output is in. A walltime kill leaves a task
without `mea_manifest.json`; `RESUME=1` re-runs only those, then
`bash launch_sim_reextract.sh gate`.

**Rehearsed** `[SANDBOX 2026-10-02]`: a `/davinci-1/home/ldellamea` tree with
Sbi-extractor at `0a560ca` (its remote at the same head, so the pull is a
no-op), SBI at `2c0a06d`, the ANN folder at `37bf9f8` (the whole
`hpc/MEA Traces`, as before the C8 upload) with a template library, the
DUP15HD manifest with its sidecar at its spaced path, a synthetic campaign
tree (v1 two tasks; v7 one task and one without `manifest.json`; v9 one task
and three empty sweep folders), a PBS-like fake `qsub` (stdout to
`~/<name>.o<n>`, stderr to `~/<name>.e<n>`), each line in a fresh shell: 21/21
checks, run twice from scratch. Negatives that held: S1 with one tool file
edited -> STOP; S2 with the archive missing, or with one file in it altered
-> STOP and no GO line; S4 after a failed suite -> no plan, nothing frozen; S5
with davinci's digest against the replica's manifest, or with a plan at
`n_side 1` -> nothing submitted; S6 before any submission -> STOP. The
rehearsal's first run found that the environment test needs the two
launchers beside it; they are in the archive since.

**Not checked:** the real campaign tree's counts (253 tasks and 47
exclusions are the 2026-10-01 probe's, P5) and the time per iteration, which
S4 and S6 measure on davinci; whether a compute node activates `sbi_export`
(S6 shows it); the state of davinci's ANN folder (S1 shows it).

## Terms used here

| term | meaning |
|---|---|
| tree id vs commit id | a tree id hashes the whole directory snapshot of a commit (`git rev-parse HEAD^{tree}`): equal tree ids mean every file and mode is byte-identical, whatever the parents, dates or authors. A commit id also hashes parent, author and time, so the laptop's commit of the same patch has a different id from the sandbox's. Everything here that must be compared is compared by tree id |
| `git apply --cached` | applies a patch to the index (the staging area) only, not to the working files |
| fast-forward | a push or pull where the remote branch is a strict ancestor of the new tip: no merge commit, no force |
| completion gate | the job behind the array that checks every planned task's output against the plan and writes the record, or REFUSES naming each bad task; the gate's log is the success signal (D-006) |
| `REEXTRACTION_RECORD.json` | the record (+ `.sha256`) that ends Stage C: it carries the frozen cohort manifest, the geometry and `EXTRA_ARGS`, the campaigns and tasks with the named exclusions, the fingerprints of the ANN tools and the template library, the one environment across the run, the label contract, the counts and any warnings `[CODE Sbi-extractor, sandbox 87a225f]`; the export over `Outputs_v2` will cite it |
| `Outputs_v2` | `ANN/MEA_analysis/Outputs_v2`, the new root of the re-extracted detections (D-014) |
| `sbi_export` | the conda environment the C8 jobs run in (D-023) |
| `n_side`, `n_e` | `n_e = n_side^2` electrodes in a subregion's grid; C8 uses `n_side = 3`, `n_e = 9`, read from the real arm's cohort manifest (D-015) |
| raster commit | the user's other chat's commit adding `raster/` (typical-well raster figures of the real cohort, D-026..D-029) to Sbi-extractor `feat/real-arm-parity` |
| REFUSED | a push script's own refusal: it found something not as expected, and changed nothing |
