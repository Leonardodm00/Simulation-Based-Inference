# HANDOFF 2026-11-17 -- LindDUP on `lcdup15_rho1300_neur_n01`: the configuration, three arms, the launch

| Date | Change |
|---|---|
| 2026-11-17 | v1. Written by the results-commentary chat (the chat that read synth v4's evaluation, D-136's Status) for the LindDUP pipeline chat, at the user's word of 09:29 Europe/Rome. State at writing: four decisions logged (D-138..D-141, log v1.140 on `main` `8d70dd8`); `linddup.json` still v10 (MEAMA, A1 only) on `sci/pipeline` `ee8b382`; nothing built, nothing run. |
| 2026-11-17 | v2. The outputs of sec. 4's three checks, pasted by the user at 09:55 Europe/Rome from `dvlogin02` `[CLUSTER, the user's paste]`: recorded in sec. 4, with what they change in sec. 5. |
| 2026-11-17 | v3. The `intel` nodes' sizes, pasted by the user at 10:18 `[CLUSTER, the user's paste]`: 82 nodes of 48 cores, 81 with 1007 GB and 1 with 881 GB; a 48-core trial or member fits one whole node (sec. 4). The user's word of 10:17 that the simulation arrays on `cfd` and `cpu` will be removed: the snapshot is to be fixed after that (sec. 7). |
| 2026-11-17 | v4. The Neuro-astro network project's reply to the SBI request (a file the user passed on at 10:46; its own tags `[REPO]`, `[RUN]`, `[OWN]`, at ANN `main` 1789e40 and f216b00): new sec. 3b. It corrects the brief and this handoff: 192 or 48 simulations share each topology, not one. The well-to-donor map is not known to that project. |

Tags: `[LOG D-nnn]` the decision log on `main`; `[CODE SBI@ee8b382 file]` read in the code on `sci/pipeline`; `[BRIEF]` the user's `LindaColloDUP15_SBI_brief.md` (written 2026-11-16 from the ANN repository; attached by the user to the results-commentary chat, not in this repository: ask the user to attach it); `[BENCH]` computed by the results-commentary chat on the synth v4 bundle (`synth_v4_results.tar.gz`), scripts delivered to the user; `[user]`; `[estimate]`; `[reasoning]` not checked.
| 2026-11-17 | v5. The user's message of 10:57 with an `ls -R` of `Deep_bio/` `[CLUSTER, the user's paste]`: the recordings move to `LindaColloDUP/recordings/` (D-142; commands and the 35-well map in `docs/project/LINDDUP_RECORDINGS_MAP.md`, the move not yet run); partial topologies are dropped whole (D-143, the user: "That's correct"); on the split by topology and on `conn_prob`'s few thousand distinct values the user: "do not worry" (no action asked; `linddup.json`'s split is already by donor = one network, D-076). The tree names no donor (map sec. 4), and `DATA_P/Batch4` shares its file names with `DATA_P/Batch3/SubBatch1` in five wells (map sec. 5): sec. 7 updated. |
| 2026-11-17 | v6. The user's message of 11:14: the donor of a real well is its condition (D-144: two donors, `C` and `P`; A5 is no longer blocked on the map); the `cmp` check `[CLUSTER, the user's paste]`: five wells of `DATA_P/Batch4` (A2, A3, B1, B2, B3) are byte-for-byte copies of `DATA_P/Batch3/SubBatch1`'s, so 30 distinct recordings, 13 DUP. Sec. 7 updated; details and consequences in `docs/project/LINDDUP_RECORDINGS_MAP.md` v2, secs. 4-5. |
| 2026-11-17 | v7. The user's message of 11:51: the recordings' move ran (D-142 done; raw data at `LindaColloDUP/recordings/DATA_{C,P}`, symlinks at the old paths); the five copied wells are left out (D-145: 30 wells, 17 control, 13 DUP; how to exclude them is this chat's to build). Open: the A5 pair -- the user leans to pairs within a well across its subregions; Claude recommends within-well pairs (the same culture, so the same parameters); today's runner refuses within-well pairs for A5 (D-118 Status), so this needs a rule change and a decision. |
| 2026-11-17 | v8. The user's messages of 12:11 and 12:17: the A5 pair is settled (D-146) -- two subregion windows of one batch, the batch being `Batch3` / `Batch4` within a condition (four groups: C-Batch3 5 wells, C-Batch4 12, P-Batch3 12, P-Batch4 1); within-well pairs are included, so the runner's refusal of them (D-118 Status) has to go. To build here: the pairing level as a configuration value (well, plate folder, batch), set to batch; and a check on the A1 fit before A5's search -- the summaries' distance over within-well pairs against across-well pairs of one batch (map sec. 4b). v7's open point on the pair is closed. |
| 2026-11-17 | v9. The user's message of 12:24 (D-147): v8's check on the A1 fit before A5's search is dropped -- the three arms are searched and fitted side by side, A5 not waiting for A1; an A5 pair is two distinct windows drawn uniformly at random within its group (condition x batch; any well, subregion and 180 s). With 54 windows per well (9 subregions x 6 windows, the 6 assumed), about 8 % of the pairs fall inside one well in the 12-well groups, 20 % in C-Batch3, all in P-Batch4 (map sec. 4b). Sec. 5 and 7 updated. |
| 2026-11-17 | v10. The build of D-148 is done, checked and pushed. SBI `sci/pipeline` `7ba708f` and Sbi-extractor `feat/real-arm-parity` `0ff531d` `[GITHUB 2026-11-17, ls-remote]`. Wave W3 (SBI blocks 29-32, Sbi-extractor Blocks 9-10) and the final checks' fix round are in new sec. 9. The davinci run sheet is SBI `hpc/pipeline/README.md`, "LindDUP: lcdup15 and its three arms -- the first submission" (N1-N9; at `54edb32` its N3 also runs the reset and identity suites, a README-only commit). Sec. 5's open items are marked settled where D-148..D-150 settled them. Nothing has run on davinci yet. |
| 2026-11-17 | v11. The user's paste of 18:52 Europe/Rome from `dvlogin04` `[CLUSTER, the user's paste]`. It ran the listing line sent mid-build, the earlier form of N4. The campaign's arrays `1942752[]` (cfd, 192 cores) and `1942753[]` (cpu, 48 cores) were still queued, state B, named `lcdup_cam*`. There were 5,673 topology folders and 0 without `topology_meta.json`. Sec. 9 is updated. The lines are to run on 2026-11-18; a reminder is set for 09:00. |
| 2026-11-18 | v12. The user's message of 12:26 Europe/Rome: D-155 -- S2's quiet window (D-112) is 1 h, not 6 h. Built and pushed: SBI `sci/pipeline` `23c17ad` (`linddup.json` `sim.plan_args` `--quiet-hours 1`, sha256 `970b21288b930a12`; the run sheet's N5 and N6, N6's inputs check with `--quiet-hours 1`; the inputs-check suite). Sec. 7's "6 h after the kill" marked corrected. Log v1.151 on `main`. |
| 2026-11-18 | v13. The user's paste of 13:21 Europe/Rome from `dvlogin02` `[CLUSTER, the user's paste]`: the campaign's arrays deleted (`qdel 1942752[] 1942753[]`, then `ARRAYS GONE`); SBI pulled to `23c17ad` and the inputs-check suite 8 of 9; N4: 5,984 topology folders, none without `topology_meta.json`, 476,027 simulation files, 22 without their json. New subsection "The campaign at 13:21" in sec. 7. |
| 2026-11-18 | v14. The user's paste of 13:32 `[CLUSTER, the user's paste]`: N5 moved the 22 simulation files without their json (and no topology folder) to `campaign_lcdup15_rho1300_neur_n01_set_aside`, the folders' times kept. The user asked to run the rest now: N6-N9 handed over with the timing note of sec. 7. |
| 2026-11-18 | v15. The user's paste of 13:44 `[CLUSTER, the user's paste]`: N7's dry run clean (exit 0; every line the run sheet names, no REFUSE, STOP or FAILED), then N8 submitted the sequencer, job `2099686.pbsserver01` (`pl_linddup`), log `/davinci-1/home/ldellamea/pipeline/linddup/driver/driver.1.log`. N6 was not run. Submitted at 13:44, before the 1 h window had passed for the tasks the kill cut (sec. 7). |
| 2026-11-18 | v16. The user's paste of 14:19 `[CLUSTER, the user's paste]`: N6 run after the submission, `INPUTS CHECK: 0 FAIL, 1 WARN` (the WARN: S2's output root holds the running test task's files and no record yet); the `[to verify]` values read on the data; the pace measured; S2's first plan left out 5 tasks. Sec. 7 updated. |
| 2026-11-19 | v17. The user's paste of 12:27 `[CLUSTER, the user's paste]`: S2's array plan (2026-11-18 14:25:56) planned the 188 other tasks with the test task's one kept as done and none left out, so all 189 are in; S2, the label freeze, the bank plan, the simulated bank (verified) and the real bank done; the three searches running since about 20:55 on 2026-11-18, 9 trials of 48 cores at once (432 cores). Sec. 7 updated. |
| 2026-11-28 | v18. The LindDUP chat: D-171 (the user, 2026-11-27 15:59) -- the searches A2 and A5 are stopped because the recordings' control wells are bad; the user ran `stop-search --search A2|A5 --kill-running` at 15:09:09Z; both ended `user_stop`; their best fits run and are to be discarded; A1 goes on (it never reads the recordings). The run on 2026-11-28 at 13:09 in the new subsection "The run on 2026-11-28" `[CLUSTER, the user's paste]`. |

## 1. The task

- Make LindDUP launchable on the new simulated campaign: rewrite `hpc/pipeline/projects/linddup/linddup.json`, check that the pipeline's and the extractor's tools read the campaign's files, set up three arms (A1, A2, A5), and give the user the davinci launch lines.
- The coding directives bind (SPEC first, blocks, the fan-out, the final check). What follows is what the user decided and what the bench showed; the design of secs. 5-6 is open and is this chat's to settle with the user.
- The results-commentary chat changed no code and no configuration.

## 2. What the user decided (2026-11-16)

| ID | Decision | Supersedes |
|---|---|---|
| D-138 | The simulated arm of LindDUP is `lcdup15_rho1300_neur_n01`; it replaces MEAMA completely for this project | D-104's simulated arm; D-107 and D-112 for LindDUP |
| D-139 | Train on n01 as it is, seed defect included | -- |
| D-140 | Arms A2 and A5 as well as A1; an A5 replicate pair is two wells of one donor | D-106 ("A1 only") |
| D-141 | Giulia's optimisation and training values per arm; the user watches interim results and may end a search early and fit the best configurations reached; jobs on `intel`; 148 cores per arm (444 in all) | D-109 (200 cores on `cpu`) |

Also agreed in conversation, not logged as decisions: the flow's conditioner depth stays fixed at 2 (the user's "OK agreed", 2026-11-16 18:03, after Claude's reasons); a promotion stage was proposed by Claude and neither accepted nor refused (sec. 6).

## 3. The new campaign against `linddup.json` v10

Campaign facts are `[BRIEF]`, not checked on the files by any chat. Folder: `~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP/campaign_lcdup15_rho1300_neur_n01/`.

| Item | v10 (MEAMA) `[CODE SBI@ee8b382 linddup.json]` | lcdup15 n01 `[BRIEF]` |
|---|---|---|
| Network | tripartite, 1600 cells/mm^2 | neurons only, 159 excitatory CAdEx neurons, 1300/mm^2, 350 um square |
| theta | 32 active axes + `conn_prob` + `stoa_gate_p`, p = 34 | 13 active axes of group `neuron_synapse` + `conn_prob`, p = 14 |
| Stored vectors | 37-D registry, 32 active | 37-D `params` and `theta`; frozen axes are constant columns; `conn_prob` stored beside `theta`, not in it |
| Coordinates | -- | log axes in natural-log coordinates, linear axes as values; each axis uniform in its coordinate |
| Simulated time | 200 s, W 20,000 | 180 s (dt 0.05 ms), so W 18,000 at 0.01 s |
| Wiring | flat, contact gate per topology | flat, `conn_prob` ~ U[0.01, 0.4] per topology; no gate |
| Simulations per topology | one | [corrected 2026-11-17: 192 (cfd tasks) or 48 (cpu tasks), not one -- sec. 3b] |
| Files | per-topology folders | `sweep_<queue>_task<NNNN>/topo_<NNNNN>/iter_<NNNNN>.npz` (+ `.json`), `topology.npz`, `job_args.json` (`_active_param_names`, `_axis_declaration`), `manifest.json`, `_failures.jsonl` |
| Count | -- | about 310,000 on 2026-11-16, still running, target 1e6 `[user]` |

The 13 active axes and their boxes (index in the 37-D vector): 0 Sigma [3, 15] mV lin; 1 gbarA [0.01, 10] nS log; 4 tauA [10, 2000] ms log; 5 U_0_ar [1e-4, 0.05] log; 6 U_max [0.1, 1] 1/ms log; 7 U_0_sr [0.1, 1] log as launched (the registry box is [0.01, 0.7] since 2026-11-03); 8 Omega_f_sr, 9 Omega_f_ar, 10 Omega_d [0.1, 4.5] 1/s log; 14 g_ampa, 15 g_nmda [0.01, 10] nS log (folder's box patch; registry [0.01, 5]); 16 delta_gA [0.005, 10] nS log; 17 x0 [0.05, 0.5] log `[BRIEF sec. 3]`.

Known properties to carry into the build `[BRIEF]`:
- Seed defect: every simulation's in-simulation draws start from one seed, and V(0) and the bias pattern are the same in every network; `seed_run` and `noise_mode` 'fresh' are not true. Accepted by D-139; a caveat on every result.
- The box leans active: in the range check, interior runs silent 1.0 %, bursting 44.8 %, above 20 Hz 49.2 %; saturated runs near 1/t_ref and silent runs are to be screened before training.
- Six axes (x0, U_0_sr, U_0_ar, Omega_f_ar, Omega_f_sr, U_max) barely move rate and burst share; expect them near the prior unless the encoder sees release dynamics.
- Never pool n01 with a later n02 or with GiuliaGate's simulations.

Sections of `linddup.json` that are MEAMA's and must be rewritten: `sim` (root, campaign, `t_sim_s`), `freeze` (`campaigns`, `require_conn_rule`), `bank` (`on_axis` `O_N`, `on_declared_box`, `topology_axes` with `stoa_gate_p`, `expected_p` 34, `t_win_s` 200), `search.shape_anchors` (p 34, d_theta 34, W 20,000), `resources` (queue, cap), `search.max_total_walltime_h`. The real arm (`real`, the DUP15HD cohort of record, 35 wells: 17 control, 18 pathological) is unchanged `[LOG D-104]`.

Not checked, to check first: whether `sim_reextract/` (the re-detection profile `linddup.sh`), the label freeze and the bank tool read this campaign (the constant columns, `conn_prob` beside `theta`, no `O_N` axis, no astrocyte spikes); which commit's code ran and the U_0_sr box as launched (the campaign's `PROVENANCE_*.txt` and `job_args.json`).

## 3b. The ANN project's reply of 2026-11-17 (`[ANN-REPLY]`; with the user, not in this repository)

| Point | What it says | Consequence for the build |
|---|---|---|
| Topologies are shared | each topology (one `conn_prob`, one wiring, one set of positions, the build seed 50, so the same V(0) and bias pattern) carries 192 simulations on a cfd task and 48 on a cpu task, each with its own independent theta draw; 32 topologies per task | the simulated "donor" is the topology folder `sweep_*/topo_*`: split train / selection / report by topology, never by simulation; donors have unequal sizes (192 or 48 rows), so E3's F1 (Dbar with donors of unequal rows) now arises; `conn_prob` has one distinct value per topology, a few thousand in all `[reasoning]`, not one per simulation |
| Usable files | a simulation is usable iff its `iter_*.json` exists; a topology iff its `topology_meta.json` exists; a topology cut by a kill keeps only its quickest runs, which lean quiet | drop partial topologies whole (the reply's audit line writes `artifacts/lcdup15_n01_partial_topologies.txt`) |
| Columns | the 13 free axes are columns 0, 1, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17 of the 37-D `theta`; read `_active_indices` from `job_args.json` and check it is the same in every task; the other 24 columns are constant; log axes are ln(value / unit), Sigma linear | theta boxes in coordinates are in the reply's table 4.1; do not test rows against the registry's 37-D box (I_inj is frozen outside it) |
| `conn_prob` | npz key `conn_prob` (float64), also in the sidecar, `topology_meta.json` and `summary.csv`; uniform on [0.01, 0.4], linear | read it beside `theta` |
| Time base | `spk_N_t` float32 seconds, `spk_N_i` 0-158; 180 s; the range check drops the first 2 s; the ANN side prescribes no rate bin | the bin is the recordings' |
| Geometry | none stored with a simulation; `topology.npz` holds `N_pos` (159 x 2, um, in [0, 350]^2). The ANN repository's `hpc/MEA Traces/process_campaign.py` already re-detects at the DUP15HD preset (`--n_side 3 --pitch 60 --edge 25 --fs 10110.09`), writing `mea_iter_*.npz`; its `simtime` key is not the duration (take 180 s from `topology_meta.json`); a topology folder without a readable `topology.npz` stops that task's run | decide between that tool and Sbi-extractor's `sim_reextract/` (D-015, D-021, D-024 fix the same probe); 55-67 of 159 neurons are within detection range in the reply's sandbox run |
| Screening | screen on the observation only (a statistic of x computed the same way on simulations and recordings), never on theta or on the full-raster rate; apply the same screen to the held-out simulations and report the share removed | for each x kept, the screened training set targets the unscreened posterior (the reply's eq. (2), its own derivation, not re-derived here) |
| Code that ran | the PROVENANCE file holds sha256 values, no commit; the reply gives the f216b00 reference hashes and a compare line | to run on davinci |
| Silent / saturated shares | not known for the campaign; a line on `summary.csv` computes them after the index is written | sizing only, not a screen |
| The well-to-donor map | NOT KNOWN to the ANN project (repository, project knowledge, past chats and PubMed searched) | it has to come from the experimental side's plating records: per well, donor, line or clone, plating date, plate and DIV at recording; A5 stays blocked until then |

The reply's order on davinci after the arrays are removed: no lcdup job left; the count; the audit; provenance; the index and its completeness check; the shares.

## 4. Read-only checks for davinci: run 2026-11-17 09:55, results below

The first two are the brief's own lines; none was tested by the results-commentary chat.

```bash
cd ~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP && conda activate brian_final && python aggregate_sweep.py campaign_lcdup15_rho1300_neur_n01 --target 1000000 | tail -3; qstat -u $USER | grep lcdup
```

```bash
cd ~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP && python -c "import json,glob; f=sorted(glob.glob('campaign_lcdup15_rho1300_neur_n01/sweep_*/job_args.json'))[0]; d=json.load(open(f)); a=d['_axis_declaration']['swept_axes']; print(len(d['_active_param_names']), {k: (v['low'], v['high'], v['log']) for k, v in a.items()})"
```

```bash
qstat -Qf intel | grep -E "resources_max|resources_default|max_run|enabled|started"
```

Results `[CLUSTER, the user's paste of 2026-11-17 09:55, dvlogin02]`:

| Check | Output | Reading |
|---|---|---|
| Count | `progress: 442,473 / 1,000,000 (44.2%)   discard rate: 0.28%   total disk: 1.200 TiB`; arrays `1942752[]` (cfd, 192 cores) and `1942753[]` (cpu, 48 cores) in state `B` | still running; the bank is a snapshot, so the count to train on has to be frozen when the bank is built |
| Axes | 13 active names; 14 swept axes: Sigma (3, 15, lin), gbarA (0.01, 10, log), tauA (10, 2000, log), U_0_ar (1e-4, 0.05, log), U_max (0.1, 1, log), U_0_sr (0.1, 1, log), Omega_f_sr, Omega_f_ar, Omega_d (0.1, 4.5, log), g_ampa, g_nmda (0.01, 10, log), delta_gA (0.005, 10, log), x0 (0.05, 0.5, log), conn_prob (0.01, 0.4, lin) | as the brief says, the launched U_0_sr box [0.1, 1] included; p = 14 confirmed on one task's `job_args.json` (the first sweep folder; the others not read) |
| `intel` | `resources_max.walltime = 240:00:00`; `max_run_res.ncpus = [u:ldellamea=960]`; enabled and started; no per-job ncpus or mem line | a job may ask at most 240 h: the best fit's 300 h walltime must come down (members resume, block 6); 444 cores fit under the user's 960; node sizes, from `pbsnodes -a` filtered on `resources_available.tag = intel` (10:18): 82 nodes of 48 cores, 81 with 1007 GB and 1 with 881 GB, so a 48-core job of 160-200 GB fills one node's cores and fits its memory; three arms at three jobs each take 9 whole nodes, and each arm's 4-core sequencer needs a share of another node |

At 442,473 simulations: n_train about 376,100; a 6-pass trial is 8,820 / 4,420 / 2,220 steps at batch 256 / 512 / 1024; a member pass 1,470 steps at batch 256; trial and arm times of sec. 5 scale by about 1.43 (typical trial about 7 h, one arm about 850 h) `[estimate]`.

Expected for the second: 13 active names, U_0_sr (0.1, 1.0, True), conn_prob (0.01, 0.4, False) `[BRIEF sec. 7]`. The third answers whether an `intel` node holds a 48-core, 160 GB trial; if not, the trial's size and the arithmetic of sec. 5 change.

## 5. Budget and resources as decided, and what they imply

Per arm `[LOG D-141; D-088, D-089, D-114]`: 50 random trials then 20 GP rounds of 10 (250 trials), a trial 6 passes in 20 epochs, no patience, best epoch kept; best fit 10 members, up to 40 passes in 40 epochs, patience 10; trial 48 cores, 160 GB; member 48 cores, 200 GB.

| Quantity | Value | Basis |
|---|---|---|
| Slots per arm at 148 cores | 3 jobs of 48 + the sequencer's 4 | arithmetic |
| n_train at 310,000 simulations | about 263,500 windows (0.85, one window per simulation) | `[estimate]` |
| Steps per 6-pass trial | 6,180 / 3,100 / 1,560 at batch 256 / 512 / 1024 | S = ceil(P / N_ep x n_train / B), SPEC sec. 2.3 |
| Steps per member pass | 1,030 at batch 256 | same |
| Typical trial | about 5 h | `[estimate]` from Giulia's about 10 h per trial at about 536,000 training rows (her build handoff v1.41); the budget gate measures it |
| One arm, search + best fit, 3 slots | about 600 h; the three arms run side by side | `[estimate]`; a GP round waits for its slowest trial |
| Edge rows at p = 14 | 17.2 % if theta is uniform on its box | 1 - (1 - 2 x 0.0067)^14 `[reasoning]` |

Open, to settle before launch:
- [settled 2026-11-17: D-150, `max_total_walltime_h` null; a search ends after its 250 trials or by hand, block 30] `max_total_walltime_h` 400 ends a search at the cap, it does not pause it `[CODE SBI@ee8b382, SPEC block 7]`; the user expects to stop by hand, so the cap has to be raised well beyond the expected stop.
- [settled 2026-11-17: block 30, `pipeline.py stop-search --search <A1|A2|A5> [--kill-running]`; the best fit then starts on the best trial so far] How the user ends a search early and starts its best fit on the trials finished so far: not read in the code.
- [settled 2026-11-17: D-148 (1), one recorded project with a `searches` list, each search with its own `max_cores` (block 29); the banks are built once] Three arms under separate per-arm caps: one project with three searches (the synthetic kind's `searches`, with a shared cap) or three projects sharing one bank (each pipeline counts only its own jobs; three sequencers must not build the same bank at once). Not designed. [2026-11-17 12:24, D-147: no check gates A5; the three searches start together.]
- The flow's edge-support call is still open in the log; no ruling is recorded for LindDUP's best fit (the 2026-10-29 ruling was Giulia's).
- The LindDUP roadmap is still unconfirmed (the log's open call).

## 6. What the bench (synth v4) says for this design

Numbers are `[BENCH]`; the full reading is D-136's Status and the user's document `synth_v4_evaluation_results.md` (delivered as files, not in the repository).

| Finding | Number | Consequence here |
|---|---|---|
| Term searches ranked on the simulated L chose near-off weights | log10 lambda -2.15, -2.81, -2.42, -1.90 on [-3, 1]; Spearman(log10 lambda, L) +0.32 to +0.70 | A2's and A5's searches ranked on L will again push the term off; the real cohort has no theta* to rank on. How the term arms are ranked, or lambda fixed or floored, is open and is the user's call |
| No comparison resolved | 0 of 15; the seed clause failed in all | do not promise an A2/A5 verdict from one search per arm |
| Arm confounded with architecture | A1's finalist flow 512 wide, the term arms' 75-147 | a shared architecture across arms, or a read of each arm at its own tuning: to decide |
| 6-pass trials favoured fast learners | all ten best trials of every search at batch 256; value at 6 passes against the best reached, Spearman -0.80 over the five finalist configurations | at 263,500 windows a trial has 7.7 times the bench's steps, so the bias should be weaker `[reasoning]`; check the finalists for pinned bounds (batch 256, lr 0.002, depth 3, dropout 0) and the share of trials whose best epoch is the last |
| A curve's slope at 6 passes does not predict its end | slope alone +0.10; value + h x slope -0.60 to +0.20 | no tilt in the objective; Claude proposed a promotion stage instead (top 5-10 configurations retrained at full budget, one seed, finalist picked there): not decided |
| Members overfit, the mixture repairs | members' C(0.95) 0.63-0.85, mixture 0.957-0.980 off the box edges; mixture gain 0.7-1.1 nats | keep the ensemble; report members and mixture apart |
| Small-flow members hit the 40-epoch cap | 4 of 5 members in two searches | after the best fit, check epochs run against the cap |
| Edge rows carry the coverage deficit | 10.8 % of rows at p = 10, +7.5 to +8.6 nats, none inside its 95 % region | sec. 5's 17.2 % at p = 14; report L with and without edge rows, or take the open call |
| Contaminated-looking windows are found without theta* | 18 windows of R3: L-C2ST statistic at its maximum 0.25 and observed SD above all 20 predictive windows | usable as an outlier screen on the DUP15HD recordings `[reasoning]` |

## 7. Still needed from the user

- The well-to-donor map of the 35 DUP15HD wells (A5's pairs, D-140). Asked 2026-11-16 and 2026-11-17; the ANN project does not have it (sec. 3b); it has to come from the plating records.
  - [2026-11-17 10:57] The user pointed to the folder tree of `Deep_bio/` as the map. Written out as `docs/project/LINDDUP_RECORDINGS_MAP.md` + `linddup_recordings_map.csv` (35 wells: condition, batch, sub-batch, well). The tree does not name a donor; three readings are listed there (plate folder, condition x batch, condition), the user's call. A5 stays blocked on that one answer. [corrected 2026-11-17 12:24: unblocked -- the donor is the condition (D-144), the pair group is condition x batch (D-146), the pair is drawn at random within it and no check gates A5 (D-147).]
- [settled 2026-11-17 11:14] The donor: the condition (D-144). To design with it: A5's pairs coincide with A2's same-class pairs; a donor bootstrap on the real arm has 2 clusters; pairs across or within batches not decided (map sec. 4). [settled 12:17 and 12:24: D-146, D-147 -- map sec. 4b.]
- [confirmed 2026-11-17 11:14] `DATA_P/Batch4` wells A2-B3 are copies of `DATA_P/Batch3/SubBatch1`'s (map sec. 5). [decided 2026-11-17 11:51, D-145: left out.]
- After the move (D-142): the raw recordings are at `LindaColloDUP/recordings/DATA_{C,P}`, with symlinks at the old paths; `real.extract_root` (`extracted_v2`) is unchanged.
- The outputs of sec. 4: given 2026-11-17 (recorded there).
- The final simulation count: the user said on 2026-11-17 10:17 that the campaign's arrays on `cfd` and `cpu` will be removed; after that, rebuild the campaign index (a job killed mid-write may leave a truncated file `[reasoning]`) and record the count as the snapshot LindDUP trains on. Not yet done. [2026-11-17, v10: the run sheet's N4 counts the topology folders and simulation files, and N5 sets aside the folders without `topology_meta.json` and the `iter_*.npz` without their json -- unusable by the ANN reply's rule, possibly truncated by the kill, and a truncated one fails S2's re-detection of its whole task -- keeping the folders' times (D-112's 6 h). The bank plan then drops the partial topologies whole and records them (D-143); the count trained on is the plan's.]
- The calls of sec. 5 (the clock, the three-arm layout, the edge support) and of sec. 6 (the term arms' ranking, the promotion stage).
- The brief itself, attached to the pipeline chat.

## 9. The build and the launch (2026-11-17, v10)

### What was built (D-148; `[CODE]` at the hashes of the changelog row)

| Repository | Block | What | Status |
|---|---|---|---|
| SBI | 29 | several searches in one recorded project (`searches`: name, campaign, max_cores), each with its own core cap; the run views; `search_report.py --search` | smoke-tested; final check below |
| SBI | 30 (+30b) | `pipeline.py stop-search --search <q> [--kill-running] [--cancel]`: no new trial, then the best fit on the best trial so far; the report and the dry run follow a stop request | smoke-tested |
| SBI | 31 | the replicate pair rule `replicate.pairs` within_group / cross_well, `replicate.group_key`; LindDUP within_group on `rep_group` (condition x batch) | smoke-tested |
| SBI | 32 | the inputs check on config v11: cohort map, topology metadata, partial topologies, searches | smoke-tested |
| SBI | 2 (amendment 4) | the bank's test task is the plan's first task with simulations | smoke-tested |
| SBI | config v11 | `linddup.json` (lcdup15, three arms of 148 cores, within_group pairs, the cohort map, partial topologies dropped), `linddup_cohort_map.csv` (35 rows, 5 excluded); `giulia.json` and `synth.json` moved to v11 | in the contract |
| Sbi-extractor | 9 | the cohort map of the real bank (donor, rep_group, wells left out; REAL_VERSION 2) | tested |
| Sbi-extractor | 10 | partial topologies dropped whole (K(t) from job_args, f(t, k) = json files + failure lines) | tested |
| Sbi-extractor | 7.4, 8 (amendments 2, 3) | WALLTIME=auto refuses a test task of 0 simulations; too-short traces counted (provenance `too_short`); an out_dir without `<class>/<root>/` refused; `cohort_map` must be absolute | smoke-tested |
| Sbi-extractor | profile | `sim_reextract/profiles/linddup.sh` defaults on lcdup15; its suite's Q7-Q9 with them | 40 of 40 |

### The final checks (independent Opus agents; reports in each repository's `specs/test_reports/`)

- Sbi-extractor: 2026-11-17_1500 at db4b418, PASS WITH ISSUES: F1-F5 (a relative cohort_map, the re-detection suite's Q7-Q9, a test task emptied by the drop, too-short traces, an out_dir without a root), fixed by amendments 2-3 and f24f4cc; the second check 2026-11-17_1605 at 5b61d6c, PASS WITH ISSUES: three NOTEs, recorded.
- SBI: 2026-11-17_1530 at a915950, PASS WITH ISSUES: F1 MAJOR (a reset left a stop request behind, so the reset search ended at once), F2-F3 (two suites broken by config v11 and block 31's parser), F4 (the README at v10), F5-F7 NOTEs, S1-S9; fixed by amendment 5 (blocks 30c, 31, the identity suite, config.py, the README); the second check 2026-11-17_1730 at 6c6c297, PASS WITH ISSUES: two NOTEs (value formats with a trailing newline, recorded; two run-sheet pass lines, corrected).
- Each finding's decision is in its report's triage table. The fixes were made by implementer agents and gated before they were merged.
- The changes to the science, all logged as `[orchestrator]` decisions: D-152, D-153 and D-154.
- Everything else was plumbing.

### On davinci

Run the run sheet N1-N9 in SBI `hpc/pipeline/README.md`, one line at a time:
- N1-N3: the code and its checks, in both repositories.
- N4: the campaign check (read-only).
- N5: the set-aside. This step is the user's call.
- N6: the inputs check.
- N7: the dry run.
- N8: the submission.
- N9: status, the per-arm reports and `stop-search`.

The order of the stages, and what to expect at each:
1. S1 is skipped.
2. S2 re-detects the campaign: 815 tasks, 9 at once, 12 h of walltime each (an estimate).
3. S3: the freeze, p = 14.
4. S4-S6: the banks.
5. The three searches start together, with 3 jobs of 48 cores each, and no clock cap.

### The campaign at 18:52 (`[CLUSTER, the user's paste]`)

- The campaign's arrays were still in the queue, in state B (some subjobs running):
  - `1942752[]`, cfd, `lcdup_cam*`, 192 cores;
  - `1942753[]`, cpu, `lcdup_cam*`, 48 cores.
  So the campaign is still being written. The user's plan of 10:17 was to remove them (v3).
- There were 5,673 topology folders, and none without `topology_meta.json`.
- The npz files without their json were not counted: that is N4's check, which the earlier line lacked.
- If the arrays are deleted (`qdel 1942752[] 1942753[]`), the kill can leave:
  - partial topologies, which the bank drops (D-143);
  - simulation files without their json, possibly truncated. N4 lists them and N5 sets them aside.
- Running the lines 6 h or more after the kill puts every task past S2's quiet rule (D-112), so its first round plans them all `[reasoning]`. [corrected 2026-11-18: 1 h or more, since D-155 set the window to 1 h; pull SBI `sci/pipeline` `23c17ad` before N1.]

### The campaign at 13:21 on 2026-11-18 (`[CLUSTER, the user's paste]`)

- The arrays `1942752[]` and `1942753[]` were deleted just before 13:21 (`qdel`, then no qstat line: `ARRAYS GONE`). The campaign's last writes are therefore about 13:20, and S2's 1 h window (D-155) covers every task from about 14:21 `[reasoning]`.
- SBI was pulled to `sci/pipeline` `23c17ad`; `test_smoke_inputs_check.py`: 8 of 9 checks pass. N1-N3 were run before (their output not pasted).
- N4: no LindDUP job in the queue; topology folders 5,984 (5,673 at 18:52 on 2026-11-17), without `topology_meta.json` 0; simulation files 476,027, without their json 22.
- N5, 13:32: `MOVED 0 topology folder(s) and 22 simulation file(s) to .../campaign_lcdup15_rho1300_neur_n01_set_aside (the folders' times kept)` `[CLUSTER, the user's paste]`.
- 13:43-13:44: N7's dry run clean; N8 submitted the sequencer `2099686.pbsserver01` at 13:44:03 `[CLUSTER, the user's paste]`; N6 not run. S2's test-task plan, made before about 14:21, leaves out the tasks the kill cut; whether its array's plan does too depends on when the test task ends: the log line `sim_redetection: [plan] being written : <n> task(s) ...` at the array's launch says how many tasks this bank misses (0 if the array plans after about 14:21) `[reasoning; CODE SBI@23c17ad stages/inputs.py, Sbi-extractor@0ff531d sim_reextract_plan.py]`.
- 14:18, N6 `[CLUSTER, the user's paste]`: `INPUTS CHECK: 0 FAIL, 1 WARN`. 189 tasks, 5,984 topologies, 476,005 iteration files; the campaign's last change 13:14 (`sweep_cpu_task0143`), so `being written` OK: S2 plans all 189; pace: 95 % of the gaps between two iteration files of one task under 0.00 h, the longest 0.43 h, from 20,000 files, so the 1 h window (D-155) is longer than every gap read; 22 partial topologies in 16 tasks (566 simulations), dropped whole by the bank (D-143); simtime 180 s = `t_win_s`; W 18,000 from the manifest's `w_size` 0.01 s; `n_e` 9 = the manifest's `electrodes_per_subset`; the cohort map as decided (35 wells, 5 excluded, 2 donors, 4 pairing groups, 17 control and 13 DUP included); p = 13 active axes + `conn_prob` = 14; `gbarA`'s box [0.01, 10] in every task, the declared one; `conn_prob` 200 distinct sampled values in [0.0117, 0.398]; 476,005 rows before the floor, 34.3 GB. The one WARN: `mea_out_linddup_lcdup15` holds files but no record, the test task's own output while S2 runs. qstat: the sequencer `2099686` running (33 min) and its successor `2099687` held, S2's test task `2099688` (`ld_test`) running (32 min); two arrays of another campaign (`gg_campai*`, `2084488[]` intel, `2084489[]` cpu) in state B.
- S2's first plan (13:45:13): 184 tasks to run (468,613 iterations), 5 left out as changed within 1 h (`sweep_cfd_task0041`, `sweep_cpu_task0143`-`0146`). By 14:18 every task is past the window, so the array's plan, made when the test task ends, plans them too: expected `tasks to run : 188`, `tasks complete : 1 kept as done (--resume)`, `being written : 0 task(s)` `[reasoning on N6's being-written row]`.
- 2026-11-19 12:27 `[CLUSTER, the user's paste]`: S2's array plan at 14:25:56 on 2026-11-18: `tasks to run : 188`, `tasks complete : 1 kept as done (--resume)`, `being written : 0 task(s)`, so the 5 tasks the first plan left out are in. Done: `sim_redetection` (its record), `label_freeze`, `bank_plan`, `sim_bank` (verified), `real_bank`. Running: `search_A1` (trials ok 5, running 3, waiting 42), `search_A2` (ok 3, running 3, waiting 44), `search_A5` (its budget gate's three probes exit 0; ok 2, running 3), all in round 0 of 20, 432 cores in 9 jobs; trial NLLs so far from -3.35 to +2.56. The first trials started about 20:55 on 2026-11-18; finished trials took about 3-5 h, two have run 15.5 h. Each search is configured for 250 trials (50 initial points, 20 rounds of 10) at 3 at a time, so at this pace round 0 alone takes days and the whole search weeks `[reasoning on the paste; not measured]`. The sequencer `2099686` at 22:40 of its 24 h, its successor `2099687` held.
- (superseded) Next: N6 and N8 not before about 14:25, so that S2's first round plans every task, the ones the kill cut included; a run started earlier leaves those tasks out of the bank unless a later round plans them (D-112's Implies).

### The run on 2026-11-28 (`[CLUSTER, the user's paste of 13:09]`)

- **A2 and A5 stopped (D-171):** the stop was requested on 2026-11-27 at 15:09:09Z with `--kill-running`; both searches ended `user_stop` (A2 at round 1 with 57 trials ok, A5 at round 1 with 53, on the status of 2026-11-27 11:48). Their best fits started on the best trial so far, as the stop command does (D-148 (2)): members 1-3 of each running, about 21 h at 12:09Z, 288 cores; they train with the recordings, whose control wells are bad, so their output is to be discarded. Do not delete their jobs to free the cores: a deleted member is resubmitted up to `best_fit.max_resubmits` (2), then the stage fails, and a failed stage ends the whole sequencer, A1 included `[CODE: stages/best_fit.py, pipeline.py]`.
- **A1 (unaffected: for a recorded data set the runner gets no real bank for S-A1, `[CODE: stages/common.py shard_globs]`):** round 5 of 20, 99 trials ok, none failed or killed; its guided trials took 3.0 to 3.6 h each and scored an NLL of -4.88 to -4.58 (the six results of round 5 in the paste); one trial running while round 5 waits for its last one. About 13.5 h per round over rounds 4 and 5, so the remaining 15 rounds take about 8 days at that pace: the search ends around 6-7 December, then its best fit.
- **Cores:** 336 of 444 in flight (A1 48, the two best fits 288); A1's own cap of 148 is not reduced by the best fits.

### Still open

- The user's confirmations: Sbi-extractor Block 8's Q13 (the real ids, now with the cohort map) and Blocks 6.2-7.4's open questions (Q6-Q8, Q10-Q12, Q14). None blocks the run.
- The flow's edge-support call, sec. 6's promotion stage, and a shared architecture across arms: as in secs. 5-6.
- `HPC_PATHS.md` has no copy in the repository (the log's open call of 2026-11-13). The LindDUP paths are in this handoff's sec. 9, in `LINDDUP_RECORDINGS_MAP.md` and in the run sheet.
- The LindDUP roadmap is unconfirmed.

## 8. Sources

| What | Where |
|---|---|
| The decisions D-104..D-115 (LindDUP as built), D-136 (Status: synth v4's numbers), D-138..D-141 | `docs/project/SBI_decisions_and_ideas_log.md`, SBI `main` |
| The configuration as it stands (v10) | `hpc/pipeline/projects/linddup/linddup.json`, SBI `sci/pipeline` `ee8b382` [v11 since 2026-11-17: `7ba708f`] |
| Passes, epochs, steps; the search's rounds and clock | `specs/SPEC.md` sec. 2.3, blocks 6-7, SBI `sci/pipeline` |
| The campaign | the user's `LindaColloDUP15_SBI_brief.md`; the ANN repository's `APPLY_LCDUP.txt` and SPEC blocks 4 and 11 (named by the brief, not read here) |
| The bench scripts | `params_relations.py`, `tilt_check.py`, and the document's `scripts/` (with the user) |
| Giulia's trial times | `docs/kb_archive/` and the knowledge-base snapshot `claude/HANDOFF_2026-10-15_giulia_pipeline_build.md` (v1.41), `claude/PLAN_2026-10-13_giulia_pipeline.md` sec. 4b |
