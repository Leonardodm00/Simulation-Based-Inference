# HANDOFF 2026-11-17 -- LindDUP on `lcdup15_rho1300_neur_n01`: the configuration, three arms, the launch

| Date | Change |
|---|---|
| 2026-11-17 | v1. Written by the results-commentary chat (the chat that read synth v4's evaluation, D-136's Status) for the LindDUP pipeline chat, at the user's word of 09:29 Europe/Rome. State at writing: four decisions logged (D-138..D-141, log v1.140 on `main` `8d70dd8`); `linddup.json` still v10 (MEAMA, A1 only) on `sci/pipeline` `ee8b382`; nothing built, nothing run. |
| 2026-11-17 | v2. The outputs of sec. 4's three checks, pasted by the user at 09:55 Europe/Rome from `dvlogin02` `[CLUSTER, the user's paste]`: recorded in sec. 4, with what they change in sec. 5. |
| 2026-11-17 | v3. The `intel` nodes' sizes, pasted by the user at 10:18 `[CLUSTER, the user's paste]`: 82 nodes of 48 cores, 81 with 1007 GB and 1 with 881 GB; a 48-core trial or member fits one whole node (sec. 4). The user's word of 10:17 that the simulation arrays on `cfd` and `cpu` will be removed: the snapshot is to be fixed after that (sec. 7). |

Tags: `[LOG D-nnn]` the decision log on `main`; `[CODE SBI@ee8b382 file]` read in the code on `sci/pipeline`; `[BRIEF]` the user's `LindaColloDUP15_SBI_brief.md` (written 2026-11-16 from the ANN repository; attached by the user to the results-commentary chat, not in this repository: ask the user to attach it); `[BENCH]` computed by the results-commentary chat on the synth v4 bundle (`synth_v4_results.tar.gz`), scripts delivered to the user; `[user]`; `[estimate]`; `[reasoning]` not checked.

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
| One simulation per topology | yes | yes |
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
- `max_total_walltime_h` 400 ends a search at the cap, it does not pause it `[CODE SBI@ee8b382, SPEC block 7]`; the user expects to stop by hand, so the cap has to be raised well beyond the expected stop.
- How the user ends a search early and starts its best fit on the trials finished so far: not read in the code.
- Three arms under separate per-arm caps: one project with three searches (the synthetic kind's `searches`, with a shared cap) or three projects sharing one bank (each pipeline counts only its own jobs; three sequencers must not build the same bank at once). Not designed.
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

- The well-to-donor map of the 35 DUP15HD wells (A5's pairs, D-140). Asked 2026-11-16 and 2026-11-17, not given.
- The outputs of sec. 4: given 2026-11-17 (recorded there).
- The final simulation count: the user said on 2026-11-17 10:17 that the campaign's arrays on `cfd` and `cpu` will be removed; after that, rebuild the campaign index (a job killed mid-write may leave a truncated file `[reasoning]`) and record the count as the snapshot LindDUP trains on. Not yet done.
- The calls of sec. 5 (the clock, the three-arm layout, the edge support) and of sec. 6 (the term arms' ranking, the promotion stage).
- The brief itself, attached to the pipeline chat.

## 8. Sources

| What | Where |
|---|---|
| The decisions D-104..D-115 (LindDUP as built), D-136 (Status: synth v4's numbers), D-138..D-141 | `docs/project/SBI_decisions_and_ideas_log.md`, SBI `main` |
| The configuration as it stands (v10) | `hpc/pipeline/projects/linddup/linddup.json`, SBI `sci/pipeline` `ee8b382` |
| Passes, epochs, steps; the search's rounds and clock | `specs/SPEC.md` sec. 2.3, blocks 6-7, SBI `sci/pipeline` |
| The campaign | the user's `LindaColloDUP15_SBI_brief.md`; the ANN repository's `APPLY_LCDUP.txt` and SPEC blocks 4 and 11 (named by the brief, not read here) |
| The bench scripts | `params_relations.py`, `tilt_check.py`, and the document's `scripts/` (with the user) |
| Giulia's trial times | `docs/kb_archive/` and the knowledge-base snapshot `claude/HANDOFF_2026-10-15_giulia_pipeline_build.md` (v1.41), `claude/PLAN_2026-10-13_giulia_pipeline.md` sec. 4b |
