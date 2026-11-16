# HANDOFF 2026-11-16 -- commenting on the evaluation of the synthetic relaunch (synth v4)

| Date | Change |
|---|---|
| 2026-11-16 | v1. Written by the LindDUP chat (the pipeline chat that built SBI blocks 24-28) for a new chat that comments on the results of the relaunch's evaluation (D-132, D-136). State at writing: the five samplings running on davinci's `intel` queue, no report yet (sec. 2). |
| 2026-11-16 | v2. By the results-commentary chat, on the user's uploads of 14:45 Europe/Rome: the samplings and the report ran complete (`EVAL REPORT OK`, finished 13:31; every check of sec. 7 holds), so the "running", "not run" and "not recorded" of secs. 2 and 11 are marked corrected; the numbers and the reading are in D-136's Status (log v1.138). The rest is unchanged. |

Tags: `[LOG D-nnn]` the decision log on `main`; `[SPEC ...]` `specs/SPEC.md` on SBI `sci/pipeline` at `2e9577b`; `[CODE SBI@2e9577b file:lines]` read in the code; `[SANDBOX 2026-11-16]` run in the author's sandbox; `[CLUSTER, the user's paste]`; `[KB]` project knowledge (frozen copies); `[reasoning]` the author's inference, not checked.

## 1. The task

- Comment, for the user, on the results of the relaunch's evaluation: what each comparison says about arms A2 and A5 against A1, and A2 against A5, on R0 (no gap) and R3 (gap pi = 0.3); whether each ensemble is calibrated (D_sim) and how well the model fits each cohort (PPC, L-C2ST, embeddings).
- Do not change code or configurations, and submit nothing on davinci. The results reach the chat as files the user uploads or tables the user pastes (sec. 6). Standalone analysis scripts on the uploaded files are fine. A suspected bug in the evaluation goes back to the pipeline chat with the evidence: the code is SBI `sci/pipeline`, `hpc/pipeline/evaluation/`, SPEC blocks 19-23, 27 and 28.
- The project instructions bind here as everywhere: the explanation modes, the source rules (KB, then PubMed, bioRxiv and the data repositories; full text for any number), the decision log (read before acting on what a decision constrains; append when the user decides), the roadmap rules.
- Roadmap `synth` (SBI `sci/pipeline`, `specs/roadmap/synth.yaml`): this is step S6. The concepts are the theory topics T6.1 "Reading the synthetic run's results" and T6.2 "Comparing two tuned arms", both open; their status changes only on the user's word. T6.2 carries a revise flag from I27 (D-136): its summary describes v3's within-search pair, and the comparisons are now across searches (sec. 4).

## 2. State at handoff (2026-11-16, about 14:30 Europe/Rome)

| What | State | Basis |
|---|---|---|
| Banks, the five searches and their best fits (`synth.json` v4) | finished. Their numbers (trials ok or failed, finalists, L, L_pseudo_real) were not pasted to any chat, so none are recorded; read them first (sec. 5, "Per search") [corrected 2026-11-16: recorded in D-136's Status, log v1.138] | [user] 09:39: "The synthetic optimization and training has finished"; [LOG D-136] |
| Calibration bank of L-C2ST | done (5 shards) | [CLUSTER, the user's paste of 11:57] |
| Samplings `ev_s0b`..`ev_s4b`, jobs `2097359`..`2097363` | running on `intel` (8 cores, 64gb, walltime 24 h) at 12:29, after a `qmove` from `cpu` [corrected 2026-11-16: finished, `EVAL SAMPLE OK` and `EVAL JOB EXIT: 0` for all five -- changelog v2] | [CLUSTER, the user's paste of 12:29]; [LOG D-137] |
| The report (metrics, then figures) | not run. [corrected 2026-11-16: it ran, `EVAL REPORT OK` at 13:31 -- changelog v2] At 12:29 the user was given the commands to pull to `2e9577b` and to submit the report now, held by PBS with `afterok` on the five samplings. Whether they ran is not known | this chat |

Check first, on davinci (read-only):

```
qstat -u ldellamea | grep ev_ ; grep -h "EVAL SAMPLE OK\|REFUSE\|EVAL JOB EXIT" /davinci-1/home/ldellamea/runs/synth/evaluation/jobs/logs/*.log
```

A sampling passes with `EVAL JOB EXIT: 0` and `EVAL SAMPLE OK` for 5 files (S-A1_A1: sim, R0, R3, R3dag, cal), 3 (S-A2-R0_A2 and S-A5-R0_A5: sim, R0, cal) or 4 (S-A2-R3_A2 and S-A5-R3_A5: sim, R3, R3dag, cal). If the report was not submitted: the README's run sheet "synth: the evaluation of the relaunch", V1 then V5-V7 (SBI `sci/pipeline`, `hpc/pipeline/README.md`). If a sampling fails, PBS drops the held report; its log goes to the pipeline chat.

## 3. The experiment, v4

| Item | v4 | Basis |
|---|---|---|
| Generator | the bench provider: theta in the open box (0, 1)^10, each axis mapped affinely onto its physical range; 100 units; per-unit-mean IFR, 180-s windows at 100 Hz (W = 18,000); observation nuisance nested batch : donor : well | [SPEC 2.4]; `synth.json` `builder` |
| Class-bearing axes K_L | 0-7: `burst_rate`, `irregularity`, `ibi_cv`, `n_fragments`, `intraburst_rate`, `participation_mean`, `burst_duration`, `fragment_duty` | [LOG D-132 (1)]; `label_axes` |
| Free axes K_F | 8 `participation_kappa`, 9 `background` | [LOG D-132 (1)] |
| Arm S, the simulated bank | 20,000 theta i.i.d. uniform on the box; 2 wells x 1 window each: 40,000 windows; seed 1000; no conditions | `synth.json` `sim`; [SPEC 2.4] |
| Arm R, the cohorts | R0 (pi = 0, seed 2000) and R3 (pi = 0.3, seed 3000), independent; each 3 conditions x 9 donors x 2 wells x 6 windows: 27 donors, 324 windows; theta* recorded and withheld from every training loss | `synth.json` `real`; [LOG D-121 (2)] |
| Conditions | theta*_k ~ normal truncated to (0, 1) about mu_{c,k} with SD tau_ov = 0.10 on K_L, uniform on K_F; mu_c = 0.5 + s (m_c - 0.5), m_c the DSN generator's regular simplex for 3 classes on the 8 label axes, s = 0.562743: centres 0.441934 apart, as in v3 | [SPEC 2.4]; [LOG D-132 (2)] |
| R3's gap | (a) range shift: axes 8 and 9 displaced by delta = 0.35 pi = 0.105 of their width; (c) drift: per well x(t) + 0.5 pi xbar sin(2 pi t / 120 s + phi); (d) contamination: per well one window with probability 0.36 becomes max(u x + e, 0), u ~ U(2, 5), so Binomial(54, 0.36) windows, 19.4 of 324 in expectation | [SPEC 2.4]; [LOG D-132 (3)]; [LOG D-125 Status] |
| Arms | A1: NPE on arm S alone. A2: + lambda_dsn x the DSN metric loss on the cohort windows' condition labels. A5: + lambda_rep x the replicate-consistency term on pairs of windows from the two wells of one donor. Both extra terms reach the encoder alone: the DSN loss is a function of the embeddings, the replicate term runs under a stop-gradient on the flow's weights | [LOG D-118, D-121 (3), D-122] |
| Searches | five, 60 trials each (20 initial points, 4 GP rounds of 10), trials of 6 passes in 20 epochs, ranked on L, the simulated report split's NLL: S-A1 (A1; R0 as its real stream, read by no loss), S-A2-R0, S-A5-R0, S-A2-R3, S-A5-R3 (term always on, campaigns `S-A2-only` and `S-A5-only`: no A1 trial inside) | [LOG D-132 (4)]; `synth.json` `search`, `searches` |
| Ensembles | each search's best trial on L refitted as 5 members (member seeds 1-5, split seed 1, 40 passes in 40 epochs, patience 10); stems `S-A1_A1`, `S-A2-R0_A2`, `S-A5-R0_A5`, `S-A2-R3_A2`, `S-A5-R3_A5`; the density is the members' equal-weight mixture (E1) | `synth.json` `best_fit`; [SPEC 2.5] |
| Split | donors 0.85 / 0.10 / 0.05 (train / selection / report), split seed 1 for every member, arm and search | `synth.json` `split`; [SPEC 2.5] |
| Data sets | `sim` (D_sim: arm S's report split, about 1,000 donors x 2 windows; NLL over both windows, the calibration tests over one window per donor); `R0`, `R3` (324 windows, theta*); `R3dag` (R3's windows at theta-dagger = theta* + 0.105 on axes 8 and 9, pseudo-true for the range shift alone) | [SPEC 2.5]; [LOG D-127 Q-E2, D-132 (3)] |
| Ensemble x data set | S-A1_A1: sim, R0, R3, R3dag. S-A2-R0_A2, S-A5-R0_A5: sim, R0. S-A2-R3_A2, S-A5-R3_A5: sim, R3, R3dag | README V4, V7 |
| Evaluation settings | 1,000 draws per row (200 per member); PPC 20 simulated windows per row (fresh nuisance, no gap) on the eight window statistics, 324 rows of D_sim as the reference; L-C2ST calibrated on 10,000 fresh arm-S windows, 100 permutations; bootstraps of 10,000 resamples; level 0.95; base seed 4242 | `synth_eval.json`; [LOG D-127] |

Changed from v3, the first run (3-13 Nov): `fragment_duty` was free (K_L = 0-6, s = 0.6 on 7 axes); the range shift moved axes 7-9, the duty clipped at 1; S-A2 and S-A5 nested A1 trials, and each had a second ensemble of the other arm (`alt_best_fit`, block 18), compared within the search. [LOG D-132, "Supersedes"]

## 4. The comparisons and the verdict

| Name | Kind | E | E_ref | Data sets |
|---|---|---|---|---|
| S-A2-R0 | vs_A1 | S-A2-R0:A2 | S-A1:A1 | sim, R0 |
| S-A5-R0 | vs_A1 | S-A5-R0:A5 | S-A1:A1 | sim, R0 |
| S-A2-R3 | vs_A1 | S-A2-R3:A2 | S-A1:A1 | sim, R3, R3dag |
| S-A5-R3 | vs_A1 | S-A5-R3:A5 | S-A1:A1 | sim, R3, R3dag |
| S-A2-R0_vs_S-A5-R0 | A2_vs_A5 | S-A2-R0:A2 | S-A5-R0:A5 | sim, R0 |
| S-A2-R3_vs_S-A5-R3 | A2_vs_A5 | S-A2-R3:A2 | S-A5-R3:A5 | sim, R3, R3dag |

Six comparisons, 15 (comparison, data set) pairs [SPEC 2.5 "The comparisons"; LOG D-136]. For each pair:

- l_i(E) = -log q_E(theta^ref_i | x_i), the ensemble's NLL at row i in nats; d_i = l_i(E) - l_i(E_ref) on the rows where theta^ref_i is inside the box and both densities are finite.
- Dbar = the mean of d_i over those rows. It equals E3's mean over donors of Delta_d = lbar_d(E) - lbar_d(E_ref) when every donor has the same number of rows, as on every synth data set (2 rows per donor on sim, 12 on a cohort, all or none on R3dag) [SPEC 2.5, F1 note].
- **Dbar < 0 favours E**: the term arm against A1, A2 against A5.
- The interval: a donor-cluster bootstrap, one copy of each donor (`bootstrap_paired.paired_bootstrap(..., within="all")`), 95 %, 10,000 resamples.
- sigma_seed = the larger of the two ensembles' SDs (ddof 1) over their 5 members of the member's mean NLL on that data set (`sd_members_E`, `sd_members_ref`).
- Verdict, the deck pack's two-clause rule G.2 [SPEC 2.5; KB `claude/deck_pack/08_SEC_G_decision_and_diagnostics.md`]: **resolved** iff |Dbar| > 2 sigma_seed and the interval excludes 0; then `lower_nll_ensemble` names the better ensemble, otherwise it is null.
- No correction for the 15 verdicts' multiplicity [LOG D-136; open since D-126].

## 5. The other quantities

Key names read from a real metrics tree of the wave's tests [SANDBOX 2026-11-16; the fixture's sizes are reduced: M 2, n_post 40].

Per ensemble and data set, `metrics/<stem>_<D>.json`:

| Keys | What | How to read |
|---|---|---|
| `nll.mean`, `nll.members`, `nll.members_sd`, `nll.n_out_of_box`, `n_in_box`, `n_rows`, `n_donors` | mean NLL over the in-box rows; each member's; their SD | nats per window, lower is better; the uniform box prior scores 0, so a negative value is more density at the truth than the prior gives [SPEC 2.4; LOG D-125 Status]; the mixture's NLL is at most the members' mean (Jensen) |
| `coverage.at` (0.50, 0.68, 0.90, 0.95), `.signed_area`, `.max_abs_dev`, `.ks_p`, `.verdict`, `.members` | expected HPD coverage C(beta) (E2) | signed area < 0 overconfident, > 0 underconfident; on sim it tests calibration against p_S (Hermans et al.); on a cohort it reads calibration under that cohort's own law [SPEC 2.5] |
| `sbc.n_rejected`, `.holm_adjusted_p`, `.ks_p`, `.verdict` | marginal SBC ranks per axis, KS tests, Holm over the 10 axes | uniform ranks expected on sim only |
| `tarp.x`, `tarp.random` (`ks_p`, `max_dev`) | TARP with references read from the windows' statistics (primary) and uniform ones | the uniform form cannot see a posterior that ignores the data [SPEC block 20] |
| `recovery` (`axes`, `contraction`, `interval90_coverage`, `rmse`, `zeta_mean`, `zeta_sd`, `sigma_mean`, `sigma0`) | per axis: 1 - sigma^2 / (1/12), the 90 % interval's coverage, RMSE, z-scores | contraction 0 = the prior's width; `zeta_mean` far from 0 = a biased posterior on that axis |
| `ppc.stats`, `ppc.frac_outside_90` | per window statistic (`mean`, `sd`, `max`, `q90`, `burst_frac`, `acf1`, `skew`, `roughness`): the fraction of rows whose observed statistic lies outside the 5-95 % range of 20 posterior-predictive simulations | D_sim's value is the reference for a well-specified model; a cohort well above it is misfit, as R3's drift and contamination should produce [reasoning] |
| `lc2st` (`stat_mean`, `stat_q95`, `frac_p_below_alpha`, `alpha`) | local C2ST at each cohort window | against D_sim's `stat_q95` |
| `embedding.members[*]` (`mmd`, `mmd2`, `p`) | MMD^2 between each member's embeddings of D_sim and of the cohort | the sim-to-cohort distance in the summary space, which the extra terms act on |

`R3dag` files have no `ppc`, `lc2st` or `embedding` (the same windows as R3) [SANDBOX fixture].

Per comparison and data set, `metrics/paired_<name>_<D>.json`: `Dbar`, `ci`, `se`, `level`, `n_boot`, `scheme`, `n_rows`, `n_donors`, `sigma_seed`, `two_sigma_seed`, `sd_members_E`, `sd_members_ref`, `beyond_2_sigma_seed`, `ci_excludes_0`, `resolved`, `lower_nll_ensemble`, `comparison`, `kind`, `E`, `E_ref`, `search`, `search_ref`, `arm`, `arm_ref`, `bootstrap_seed`, `dataset`; the `.npz` holds `donor_ids`, `delta_d`, `se_d` [SPEC block 27; SANDBOX fixture].

Per search, `metrics/search_<q>.json`: `trials` in proposal order (`trial_id`, `round`, `arm`, `status`, `nll` = L, `L_pseudo_real` on the search's own cohort, `running_min_nll`), `finalist`, `ensembles.<E>.members` (each member's `L`, `L_pseudo_real`, learning curves), `example_donors` (the corner plots' donors: per condition, the donor of median lbar_d under S-A1's ensemble, so the choice is not tuned to an arm). A trial's full record, hyperparameters included (`config`: `dsn_on`, `rep_on`, `log10_lambda_dsn`, `log10_lambda_rep`, `lr`, `embedding_size`, ...; also `cluster_ari`, whose definition was not checked here), is `tune_joint_synth/<q>/<campaign>/trials/<trial_id>.json` [SANDBOX fixture].

## 6. Where the results are, and how they reach the chat

On davinci, `out_root` = `/davinci-1/home/ldellamea/runs/synth/evaluation` (`synth_eval.json`):

| Path under out_root | Content |
|---|---|
| `REPORT_COMPLETE.json` | the report's inputs, written last |
| `metrics/METRICS.json` | the index: `ensembles`, `files`, `left_out`, `comparisons` (each with its `paired` files), `comparisons_left_out`, `uncompared`, `report_inputs` |
| `metrics/<stem>_<D>.json` and `.npz` | 14 files of each kind (sec. 5) |
| `metrics/paired_<name>_<D>.json` and `.npz` | 15 of each (sec. 4) |
| `metrics/search_<q>.json` | 5 |
| `figures/FIGURES.md` | one caption per figure; its header names anything left out |
| `figures/ensembles/<stem>/` | `cal_coverage`, `cal_tarp`, `cal_sbc_ecdf_<D>`, `cal_sbc_hist_<D>`, `acc_recovery_<D>`, `acc_zscore_<D>`, `acc_sigma`, `acc_nll`, `post_ppc_<D>`, `post_lc2st`, `post_corner_<cohort>_c<k>`, `emb_pca_<cohort>` |
| `figures/comparisons/<name>/` | `cmp_coverage`, `cmp_nll_<D>` (not sim), `cmp_ppc_<D>` and `cmp_lc2st_<D>` (R0, R3), `cmp_forest_<D>` (donor-paired Delta_d with the verdict) |
| `figures/searches/<q>/` | `search_trials`, `search_nll_vs_lpr`, `search_learning` |
| `figures/across/` | `dbar` (every pair's Dbar, interval and +-2 sigma_seed; left panel against A1, right A2 against A5), `lpr` |
| `jobs/logs/` | every job's log; `report.log` the report's |
| `data/` | the data overview (block 23), only if `--what data` was run: not run for v4 as far as recorded |
| `samples/` | the posterior draws: large, not needed |

Best fits: `/davinci-1/home/ldellamea/runs/synth/<q>/synth_best/` (`finalist_config.json`, `BEST_FIT_COMPLETE.json`, `ensemble_nll.json`, `member_seed1`..`member_seed5`). Searches: `/davinci-1/home/ldellamea/tune_joint_synth/<q>/<campaign>/` (`trials/`, `pipeline_rounds.json`, `finalists.json`, `SEARCH_COMPLETE.json`). [CODE SBI@2e9577b `hpc/pipeline/stages/synthetic.py:69-79`, `stages/best_fit.py:76-87`, `stages/search.py:145-156`]

Colours in every figure: A1 #2a78d6 (blue), A2 #eb6834 (orange), A5 #1baf7a (green); conditions 0, 1, 2 a pink circle, a purple square, a green triangle [SPEC block 21].

### Route 1, upload (preferred): two bundles

Run on davinci from any folder; both commands were run as written on a mock of this tree, exit 0 [SANDBOX 2026-11-16]:

```
cd ~ && tar -czf ~/synth_v4_results.tar.gz runs/synth/evaluation/REPORT_COMPLETE.json runs/synth/evaluation/metrics runs/synth/evaluation/jobs/logs runs/synth/*/synth_best/finalist_config.json runs/synth/*/synth_best/BEST_FIT_COMPLETE.json runs/synth/*/synth_best/ensemble_nll.json tune_joint_synth/*/*/SEARCH_COMPLETE.json tune_joint_synth/*/*/pipeline_rounds.json tune_joint_synth/*/*/finalists.json tune_joint_synth/*/*/trials; echo "exit $?"; ls -lh ~/synth_v4_results.tar.gz
cd ~/runs/synth/evaluation && tar -czf ~/synth_v4_figures_png.tar.gz --exclude='*.pdf' figures; echo "exit $?"; ls -lh ~/synth_v4_figures_png.tar.gz
```

Each must print `exit 0`; a `Cannot stat` line names a file that does not exist yet (then exit 2, and the rest is packed). The user downloads both with MobaXterm's file panel and uploads them; extract each into its own empty folder before reading.

### Route 2, paste: three tables, read-only

Run on davinci (the activation only provides a `python3`; the lines use `json` and `glob` alone); tested on the fixture's real metrics files [SANDBOX 2026-11-16]. The comparisons' verdicts:

```
cd ~/runs/synth/evaluation/metrics && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 -c "import json,glob;g=lambda v:float('nan') if v is None else v;m=json.load(open('METRICS.json'));[print('%-20s %-6s Dbar %+.3f  CI [%+.3f, %+.3f]  2sigma_seed %.3f  donors %s  -> %s' % (c['name'],D,g(p['Dbar']),g(p['ci'][0]),g(p['ci'][1]),g(p['two_sigma_seed']),p['n_donors'],('resolved, %s lower' % p['lower_nll_ensemble']) if p['resolved'] else 'not resolved')) for c in m['comparisons'] for D,f in c['paired'].items() for p in [json.load(open(f))]]"
```

Each ensemble on each data set:

```
cd ~/runs/synth/evaluation/metrics && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 -c "import json,glob;g=lambda v:float('nan') if v is None else v;[print('%-24s nll %+.3f (members sd %.3f)  in box %s of %s rows  coverage area %+.3f  SBC axes rejected %s  TARP(x) KS p %.3g' % (e['stem']+'_'+e['dataset'],g(e['nll']['mean']),g(e['nll']['members_sd']),e['n_in_box'],e['n_rows'],g(e['coverage']['signed_area']),e['sbc']['n_rejected'],g(e['tarp']['x']['ks_p']))) for f in sorted(glob.glob('*.json')) if not f.startswith(('paired_','search_','METRICS')) for e in [json.load(open(f))]]"
```

Each search, its trials and its ensemble's members:

```
cd ~/runs/synth/evaluation/metrics && eval "$(conda shell.bash hook)" && conda activate sbi_export && python3 -c "import json,glob;g=lambda v:float('nan') if v is None else v;[print('%-8s trials ok %d of %d, finalist %s; %s' % (s['search'],sum(t.get('status')=='ok' for t in s['trials']),len(s['trials']),s['finalist'],'; '.join('%s members L %s, L_pseudo_real %s' % (k,' '.join('%.3f' % g(mm.get('L')) for mm in v['members']),' '.join('%.3f' % g(mm.get('L_pseudo_real')) for mm in v['members'])) for k,v in s['ensembles'].items()))) for f in sorted(glob.glob('search_*.json')) for s in [json.load(open(f))]]"
```

## 7. Sanity checks before reading a number

- `report.log` (README V7): `comparisons: 6 made, 0 left out, 0 search(es) in none`; `EVAL METRICS OK: 5 ensemble(s), 14 data-set file(s), 15 paired comparison(s), 5 search(es), 0 left out`; `EVAL FIGURES OK: 175 figure(s)` (by the figure rules: 116 of the ensembles, 42 of the comparisons, 15 of the searches, 2 across; not yet seen on davinci); `EVAL REPORT OK`; `EVAL JOB EXIT: 0`.
- `METRICS.json`: `left_out`, `comparisons_left_out` and `uncompared` empty; `FIGURES.md`'s header "Comparisons left out: none".
- Ensemble files: `M` 5, `n_post` 1000, `n_rows` 324 on R0 and R3.
- Paired files: `n_donors` 27 on R0 and R3; on R3dag only the donors whose theta-dagger is inside the box, 0.80 of 27 in expectation (1 - (1 - 0.105)^2 = 0.20 leave it) [SPEC 2.5]; on sim about 1,000.
- Read the files `METRICS.json` and `FIGURES.md` list, not a folder listing: an earlier report's files would stay beside a new one's, unlisted (the final check's F7, open) [SPEC block 27].

## 8. Reading caveats

1. Each arm at its own best tuning, not an ablation. A term arm's ensemble is the best trial of its own 60-trial search, S-A1's of another, so the comparison measures each arm at its own best hyperparameters, not the term switched on and off at fixed ones [roadmap T6.2's summary; LOG D-132 (4)].
2. The cohorts' windows are seen in training. A2's and A5's extra losses read every window of their search's cohort, and the pseudo-real NLL scores those same windows; theta* is withheld from every loss [CODE SBI@2e9577b `hpc/joint/stage3/run_joint_arms.py:855-885` (the real stream, all windows), `:958-975` (the endpoint on `real["x"]`)]. A2's labels carry information about theta*, since a condition fixes the centre of the 8 label axes; the flow itself trains on arm S alone, so that information reaches the posterior only through the encoder's geometry [reasoning].
3. Ranking on L, endpoint on the cohort. The searches rank on the simulated L; the evaluation reads the cohort; the two can disagree. In v3, at R3 the fully trained models had better L and worse L_pseudo_real than the 6-pass trials [LOG D-125 Status].
4. R3's gap in v4. The range shift now moves axes 8 and 9 only, which the joint docs found move none of the eight window statistics beyond their noise at the class-2 centre (F-bk) [KB deck pack 07, frozen; LOG D-132 (3)], so R3's misfit should come mostly from the drift (every window) and the contamination (about 6 % of the windows) [reasoning, from D-132]. R3dag corrects the range shift only: drift and contamination have no coordinate in the box.
5. R0 and R3 are independent cohorts on different seeds: a difference between them carries the cohort-to-cohort spread of the separability as well as the gap [LOG D-121 (2)].
6. sigma_seed is an SD over 5 members (ddof 1), itself noisy, and the larger of two; read `sd_members_E` and `sd_members_ref` with it. In v3 the members' SD of L_pseudo_real was 0.47-0.73 nats on R0 and 0.44-0.47 on R3 [LOG D-125 Status].
7. 15 verdicts and no multiplicity correction [LOG D-136].
8. On sim, the donors are arm S's report donors and Dbar is the simulated endpoint: what a term costs or gains on simulations, not on the cohort.
9. Coverage on a cohort is calibration under that cohort's law (three clusters on the label axes), not under p_S; only sim tests the trained posterior against the prior it was trained under [SPEC 2.5].
10. On R3dag the donors whose theta-dagger leaves the box are out of the NLL and the coverage, and in the ranks, the recovery and the z-scores [LOG D-127 Q-E3; SPEC 2.5].
11. Open limits of the code from the final check of wave W2, none expected on these runs: F4 (fewer than two donors on a paired data set), F5 (an unreadable sample provenance), F6 (a one-member ensemble's sigma_seed of 0), F7 (above, sec. 7), F8 (the launcher's exit code) [SPEC block 27, "Open"].

## 9. Earlier readings to build on (v3, the first run)

- D-125's Status in the log: the first run's search results as the user pasted them on 2026-11-11, Claude's reading and two corrections. At R0 every fully trained model placed its density on the truth and the arms were not told apart by those numbers; at R3 better L went with worse L_pseudo_real, consistent with the range shift (then on axes 7-9) offsetting theta* by 0.105, so a posterior sharper than the offset scores worse. Single-seed trial contrasts of L_pseudo_real: A2 - A1 +1.77 (R0), +2.20 (R3); A5 - A1 -0.16, -0.76; not on the members' noise scale.
- The results chat "DSN-NPE results explanation" (https://claude.ai/chat/c0bf9135-a0b0-4a35-8906-dae67de5272d): the reading's eqs. (1)-(6), `fig2_lpr_by_search.png`, `fig3_offset_vs_width.png`, `synth_results_reading_v2.py` (checks C0-C5); T6.1's material.
- v3's evaluation: archived by the relaunch's purge at `~/synth_eval_2026-11-13.tar.gz` (the README's run sheet P3), holding three of v3's four within-search comparisons [LOG D-132 "Implies"]. That its report completed, and what it showed, is recorded nowhere [not verified]. Its design differs (sec. 3), so its numbers do not compare one to one.

## 10. Sources

| What | Where |
|---|---|
| The decisions: D-118-D-122 (the experiment), D-125 (Status: the first run's results), D-126, D-127 (the evaluation), D-132 (the relaunch), D-136 (the comparisons), D-137 (the queue) | `docs/project/SBI_decisions_and_ideas_log.md`, SBI `main` |
| The mathematics: sec. 1.1, 2.4, 2.5 (E1-E3, the comparisons, notation); blocks 19-23, 27, 28 | `specs/SPEC.md`, SBI `sci/pipeline` |
| The run sheets: "synth: the evaluation" (E10-E11, the data overview), "synth: the evaluation of the relaunch" (V1-V7) | `hpc/pipeline/README.md`, SBI `sci/pipeline` |
| The two-clause rule (G.0-G.4); the arms table (B.5, per D-118); the replicate term | KB, frozen copies: `claude/deck_pack/08_SEC_G_decision_and_diagnostics.md`, `claude/deck_pack/03_SEC_B_joint_objective.md`, `claude/deck_pack/05_SEC_D_replicate_consistency.md`; `JOINT_DSN_NPE_USAGE_v1.md` sec. 5.1 |
| Coverage, SBC, TARP, local tests, misspecification | KB PDFs: "A Trust Crisis In Simulation Based Inference ..." (Hermans et al.), "Simulation Based Inference A Practical Guide" (pp. 32, 38), "Sampling-Based Accuracy Testing of Posterior Estimators ..." (TARP), "Detecting Model Misspecification in Amortized Bayesian Inference ..." (Schmitt et al.). L-C2ST's own paper is not in the KB: search before citing |
| The roadmap | `specs/roadmap/synth.yaml` and `.md`, SBI `sci/pipeline`; the page "Subproject Roadmaps" |

## 11. Open at handoff

- The report: not run (sec. 2). [corrected 2026-11-16: it ran -- changelog v2]
- The relaunch's search and best-fit numbers: not recorded anywhere; read first. [corrected 2026-11-16: recorded in D-136's Status, log v1.138]
- A multiplicity correction across the 15 verdicts: open (D-126, D-136).
- Dbar's estimator for donors of unequal rows (F1): open; does not arise on synth.
- T6.2's revise flag (D-136).
- The data overview of v4 (block 23): not run as far as recorded; it shows the cohorts' windows and eight statistics against the simulated bank, and theta* and theta-dagger, context for R3's PPC (README E10-E11).
