# Deck design: the synthetic runs (synth v4), joint DSN + NPE, for the supervisor

| Date | Change |
|---|---|
| 2026-11-18 | v1. Design of a 15-20 min PowerPoint on synth v4: the synthetic traces, the joint DSN + NPE model, the optimisation and training, the three arms and the results. 19 main slides, 8 backup slides, 18 figures (PNG at 200 dpi and SVG) with the scripts that make them. Numbers taken from the synth v4 evaluation files and checked against the results document of 2026-11-16. |

This is the brief for whoever builds the deck: you, a design session, or the Slides tool. Per slide it gives the title (the message), what goes on the slide, which figure and where, the equation if there is one, and the speaker notes. The slides use plain words and only five equations. The technical detail is in the notes and the backup slides.

## 1. How to use this pack

| Item | Where |
|---|---|
| This document | `DECK_DESIGN_synth_v4.md` |
| Figures, one PNG (2666 x 1320 px at 200 dpi; F16 2666 x 1440) and one SVG each | `figures/F01_idea` ... `figures/F18_take_home` |
| Scripts that make the figures | `scripts/` (section 8) |
| Contact sheet of all figures | `figures/CONTACT_SHEET.png` |
| Data the figures read | the synth v4 evaluation bundle `synth_v4_results.tar.gz`, extracted to `data/` beside `scripts/` (ignored by git); `data/traces.npz` and `data/raster.npz` (simulated here, committed) |

Rules carried over from the project's first deck pack (style guide 11, 2026-09-22):

- **The slide title is the message.** Keep it to two lines at 28-32 pt. If a title is too long, cut a subordinate clause, never the main clause.
- **Claim tags go in the speaker notes only.** Every number on a slide has a tag in its notes. The tags are explained below.
- **Prefer the SVG.** Fall back to the PNG only where SVG import fails. Leave the provenance footer of each figure visible.
- **Do not add a number** that is not in this pack.

Tags used in the notes:

| Tag | Meaning |
|---|---|
| `[DATA: f]` | read from the synth v4 evaluation file `f` (inspected for this pack) |
| `[RUN: s]` | computed here by script `s` from those files; every printed value was compared with the results document |
| `[RES §n]` | the verified results document `synth_v4_evaluation_results.md` (2026-11-16), section n |
| `[CODE: f]` | read in the SBI code, configuration or design notes (branch `sci/pipeline`, 54edb32) |
| `[LOG D-n]` | the decision log, `docs/project/SBI_decisions_and_ideas_log.md` |
| `[KB: title]` | a full-text PDF in project knowledge |
| `[OWN]` | a reading or inference, not a measurement |

## 2. Storyline and timing

The story has four acts. Act 1 sets up the question, act 2 the synthetic world, act 3 the method, act 4 the results. Each results slide answers one question about the run.

| # | Title (the message) | Figure | min |
|---|---|---|---|
| 1 | Can the cohort's own data improve the posterior? A synthetic test of joint DSN + NPE training | none | 0.3 |
| 2 | We test whether reading the cohort during training helps, in a world where the truth is known | F01 | 1.2 |
| 3 | The bench simulator turns ten numbers into the bursting of 100 neurons | F02 | 1.0 |
| 4 | Ten parameters: eight carry the condition, two are free | F03 | 0.8 |
| 5 | The three conditions burst differently | F04 | 0.5 |
| 6 | Training sees the whole box; the cohorts are three clusters with a hidden truth | F05 | 1.0 |
| 7 | R3 adds a gap: shifted ranges, a slow drift and contaminated windows | F06 | 1.0 |
| 8 | One network, three losses: NPE on simulations plus two terms that read the cohort | F07 | 1.5 |
| 9 | Three arms: A1 learns from simulations only; A2 and A5 add one cohort term each | F08 | 1.5 |
| 10 | Each arm is tuned by a 60-trial search, retrained five times and averaged | F09 | 1.0 |
| 11 | We score by the density at the truth, and a difference counts only if it beats training noise | F10 | 1.0 |
| 12 | Every search went from worse than the prior to 3-4 nats better, and was still improving | F11 | 0.9 |
| 13 | The members overfit; averaging five of them gains 0.7-1.1 nats | F12 | 0.9 |
| 14 | Six parameters are learned well, two weakly, the two free ones not at all, in every arm | F13 | 0.8 |
| 15 | Inside the box the ensembles are calibrated or conservative; the failures are truths at the box edges | F14 | 1.1 |
| 16 | R0 is scored well on every window; R3's penalty sits in 18 windows | F15 | 1.1 |
| 17 | None of the 15 comparisons passes the rule: training noise is larger than every difference | F16 | 1.2 |
| 18 | On R3 the tie hides a trade: A5 is better on ordinary windows, much worse on contaminated ones | F17 | 0.9 |
| 19 | The pipeline works end to end; this run cannot yet tell the arms apart | F18 | 1.0 |
| | **Total** | | **18.7** |

**The 15-minute cut.** Drop slide 5 (its figure goes to backup) and slide 14 (backup B3 keeps it). Merge slides 17 and 18: put F17 on the right half of slide 17 and keep only its first bullet. Say slides 3-4 in 1.2 min together. That saves about 3.2 min and lands at about 15.5 min.

## 3. The slides

Figure placement: **full** means the figure fills the content area under the title (13.33 x 6.6 in). **Right half** means it sits beside a text column of at most four short lines. The figures were drawn at full-slide size; on a half slide, scale the SVG down and keep the footer.

### Slide 1: title

- **Title:** Can the cohort's own data improve the posterior? A synthetic test of joint DSN + NPE training
- **Subtitle:** synth v4: synthetic traces, the joint model, training, three arms, results; [name], [date]
- **Notes:** "Today: one synthetic experiment, end to end. What it was for, how the data were made, what we trained and how, and what came out. The short version: the pipeline works and gives usable posteriors, but this run cannot yet say whether the two extra terms help."

### Slide 2: the question, and why a synthetic world

- **Title:** We test whether reading the cohort during training helps, in a world where the truth is known
- **Figure:** F01, full.
- **On the slide** (one line under the figure, or none): NPE learns from simulations. A2 and A5 also read the real cohort, without needing its parameters.
- **Notes:** "Neural posterior estimation learns, from simulations, a map from a recording to a posterior over the simulator's parameters. Real recordings differ from any simulator. So we add terms that read the real cohort during training. They use only what real data has: which condition a recording belongs to, and the fact that two wells of one donor are replicates. On real data we cannot tell whether that helps, because nobody knows the true parameters. So we built a synthetic cohort whose true parameters are recorded and hidden from training, and scored every arm against them. One cohort, R0, comes straight from the simulator. A second, R3, has a deliberate gap between simulator and data." Tags: the design [LOG D-121, D-132]; [CODE: handoff 2026-11-16 sec. 3].

### Slide 3: the simulator

- **Title:** The bench simulator turns ten numbers into the bursting of 100 neurons
- **Figure:** F02, full. If the slide needs text, use the figure at about 70 % width with this column on the left:
  - bursts arrive at random times: **burst rate**, **IBI CV**
  - each lasts a variable time (**burst duration**, **irregularity**) and splits into pieces (**n fragments**, **fragment duty**)
  - each neuron joins a piece with some probability (**participation mean**, **kappa**), then fires fast (**intraburst rate**)
  - between bursts, slow **background** firing
- **Notes:** "This is the generator we use for all synthetic data. Ten numbers set a burst process. Burst onsets follow a renewal process: the rate sets how often bursts come, the IBI CV how regular they are. Each burst lasts a log-normally distributed time and is cut into fragments. Every neuron decides, fragment by fragment, whether to join, with a probability drawn per neuron from a Beta distribution. Kappa sets how different the neurons are. Neurons that join fire at the intraburst rate; all fire slowly in between. The network never sees the spikes, only x: the population's firing per unit, smoothed and sampled at 100 Hz. One window is 180 s, so 18,000 numbers. The right panel zooms on one burst that came out as two fragments." Tags: model structure [CODE: `hpc/joint/stage1/bench_burst_generator.py` docstring and `BenchBurstParams`]; window size, 100 units, 100 Hz [CODE: handoff 2026-11-16 sec. 3; SPEC 2.4]; the raster is a fresh simulation at condition 3's centre [RUN: `gen_raster.py`].

### Slide 4: the ten parameters

- **Title:** Ten parameters: eight carry the condition, two are free
- **Figure:** F03, full.
- **On the slide:** nothing beyond the figure, or: *R3 moves the free axes' ranges up by 10.5 %.*
- **Notes:** "Each parameter is a coordinate between 0 and 1, mapped linearly onto a physical range, shown on the right. The first eight are the label axes: a condition fixes their centre, the dots. A donor's true parameters scatter around that centre with standard deviation 0.10. The last two, participation kappa and background, are free: uniform in every data set, whatever the condition. The centres are the corners of a regular triangle in the eight label dimensions, each 0.44 from the others. In R3 the free axes' ranges move up by 10.5 % of their width; that is the red line." Tags: ranges and units [CODE: `bench_burst_provider.py`, BENCH_AXES]; 8 label and 2 free axes, s = 0.562743, centres 0.441934 apart, SD 0.10 [LOG D-132; CODE: handoff sec. 3]; 10.5 % = 0.3 x 0.35 [CODE: `latent_gap.py`, `free_axis_range_shift`]; centre coordinates [RUN: `centres.py`, sec. 9.3].

### Slide 5: what the conditions look like (optional; cut first)

- **Title:** The three conditions burst differently
- **Figure:** F04, full.
- **Notes:** "One simulated 180-s window at each condition's centre. Condition 3 bursts most often: 0.29 bursts per second against 0.21 and 0.25. Its intervals are also the least regular. Condition 2's bursts are the weakest: fewer neurons join (participation 0.59 against 0.64-0.65) and they fire more slowly (93 against 102-105 spikes per second). The differences are real but modest. Donors scatter around these centres, so two donors of different conditions can look alike." Tags: centre values mapped to physical units [RUN: `centres.py` with BENCH_AXES]; the windows [RUN: `gen_traces.py`].

### Slide 6: the two data sets

- **Title:** Training sees the whole box; the cohorts are three clusters with a hidden truth
- **Figure:** F05, full.
- **Notes:** "Left: the simulated bank, arm S. 20,000 parameter vectors drawn uniformly over the whole box, each simulated into two windows: 40,000 windows. Donors are split 85 / 10 / 5 % into training, selection for early stopping, and a report split for scoring. Middle: cohort R0, arm R. 27 donors, nine per condition, seen in the plane through the three condition centres. They sit well inside the cloud the bank covers. Right: how a cohort is built. Three conditions, nine donors each, two wells per donor with the same parameters but their own randomness, six 180-s windows per well: 324 windows. R0 and R3 are built the same way with different seeds. Their true parameters are recorded and never enter a training loss." Tags: 20,000 x 2 = 40,000; 27 donors, 324 windows; split 0.85 / 0.10 / 0.05 [CODE: handoff sec. 3, `synth.json`]; the plotted parameters [DATA: `S-A1_A1_sim.npz`, `S-A1_A1_R0.npz`, theta_ref]; 1,000 report-split vectors shown [RUN: `figs_setup.py`].

### Slide 7: R3's gap

- **Title:** R3 adds a gap: shifted ranges, a slow drift and contaminated windows
- **Figure:** F06, full.
- **On the slide:** severity pi = 0.3. Three perturbations, lettered as in the code:
  - (a) the two free axes' ranges move up by 10.5 %
  - (c) a slow wave on every well: 15 % of its mean, period 120 s
  - (d) per well, with probability 0.36, one window amplified 2-5 times plus noise
- **Notes:** "R0 is what the simulator makes. R3 adds three ways real recordings can differ from it, all at severity 0.3. First, the free parameters' ranges are shifted, so R3's data come partly from parameters outside the box the network was trained on. Second, a slow sinusoidal drift is added to each well; panel (c) shows it as R3 minus the clean trace. Third, contamination: in a well, with probability 0.36, one window is replaced by an amplified, noisy copy. The simulator cannot produce such a window at any parameter. We expect about 19 contaminated windows among R3's 324. Perturbation (b) was removed from the code earlier, so the letters skip it." Tags: pi, shift_max 0.35, drift amplitude 0.5 pi of the mean, period 120 s, contamination probability 0.36, factor 2-5 [CODE: `latent_gap.py`; handoff sec. 3; LOG D-132 (3)]; 54 x 0.36 = 19.4 expected [RES §3.7]; the example well [RUN: `gen_traces.py`].

### Slide 8: the joint model and its objective

- **Title:** One network, three losses: NPE on simulations plus two terms that read the cohort
- **Figure:** F07, full, with equation (1) under it. Or F07 at about 80 % height with the equation as a band at the bottom.
- **Equation (1)** (section 4 has the full form and its reading):

$$\mathcal{L}(\psi,\omega)\;=\;\mathcal{L}^{\rm sim}_{\rm NPE}(\psi,\omega)\;+\;\lambda_{\rm dsn}\,\mathcal{L}^{\rm real}_{\rm DSN}(\psi)\;+\;\lambda_{\rm rep}\,\mathcal{L}^{\rm real}_{\rm rep}(\psi,\omega)$$

- **Notes:** "The network has two parts. A summary network h-psi compresses the 18,000-sample window into a short vector z. A normalizing flow q-omega turns z into a full posterior over the ten parameters. The first loss is ordinary NPE: on simulated pairs it makes the posterior put high density on the true parameters, and it trains both parts. The other two losses read the cohort. The DSN loss looks at the cohort windows' summaries with their condition labels. The replicate loss compares the posteriors of two windows from the two wells of one donor. Both are allowed to change only the summary network. The replicate term passes through the flow, but a stop-gradient keeps it from changing the flow's weights. The two lambdas set how much each extra term counts." Tags: eq. (1), streams, encoder-only routing, stop-gradient [CODE: `hpc/joint/docs/E4_JOINT_OBJECTIVE_AND_LOOP.md` eq. (plan 1), (E4.3); LOG D-118, D-121 (3), D-122].

### Slide 9: the three arms

- **Title:** Three arms: A1 learns from simulations only; A2 and A5 add one cohort term each
- **Figure:** F08, full, with equation (2) as a small band under the A5 card. Or move eq. (2) to the notes and keep the slide clean.
- **Equation (2)**, replicate consistency, as the slide shows it (full form in section 4):

$$\hat T=\hat\Delta^{\top}(2\bar C)^{-1}\hat\Delta-\frac{d_\theta}{S_{\rm mc}},\qquad \ell_{\rm rep}=\big(\log\hat T-\log\hat p_{\rm eff}\big)^2,\qquad \hat p_{\rm eff}=d_\theta-\mathrm{tr}\big(\Sigma_0^{-1}\bar C\big)$$

- **Notes:** "A1 is plain NPE: the two lambdas are zero, so it is the reference. A2 adds the DSN loss. Windows of the same condition should have summaries close together, different conditions far apart. A second part spreads the three condition means 120 degrees apart on the sphere where the summaries live. A5 adds the replicate term. Two wells of one donor share the same true parameters, so their posteriors should differ only by chance. T-hat measures the gap between the two posterior means in units of the posteriors' own spread. If the posteriors are honest, T should be about p-eff, the number of directions the data actually pin down. The loss pulls log T towards log p-eff from both sides. A too-large T means overconfident posteriors that disagree. A too-small T means a summary that ignores the data. Neither term needs the true parameters." Tags: DSN triplets and separation target -1/(K-1), 120 degrees at K = 3 [CODE: `P2_DSN_LOSS_AXES.md` eqs. (P2.3), (P2.8)]; T, p_eff, the loss and its clamps (floor 1e-8, p_min 1e-3), the d_theta/S_mc correction [CODE: `P3_REPLICATE_AXES.md` eqs. (P3.4)-(P3.7)]; arms [LOG D-118, D-121, D-122].

### Slide 10: optimisation and training

- **Title:** Each arm is tuned by a 60-trial search, retrained five times and averaged
- **Figure:** F09, full, with equation (3) in a corner of stage 2 or in the notes.
- **Equation (3)**, the ensemble:

$$q_E(\theta\mid x)\;=\;\frac{1}{5}\sum_{m=1}^{5}q_{E,m}(\theta\mid x)$$

- **Notes:** "Each arm gets its own hyperparameter search: 20 random configurations, then four rounds of ten proposed by a Gaussian-process model of the score. Each trial trains one network for six passes over the bank. It is ranked by L, the NLL at the truth on simulated windows it never trained on, never by its score on the cohort. The best trial is retrained five times with seeds 1 to 5, up to 40 epochs, stopping after 10 epochs without improvement on the selection split. The five members are averaged as densities: the ensemble is their equal mixture. There are five searches: A1, plus A2 and A5 each run once with R0 and once with R3 as the cohort their term reads. All 300 trials finished." Tags: 60 = 20 + 4 x 10, 6 passes, ranked on L, five searches [CODE: handoff sec. 3, `synth.json` search; LOG D-132 (4)]; seeds 1-5, 40 epochs, patience 10, split 85 / 10 / 5, mixture (E1) [CODE: handoff sec. 3; SPEC 2.5]; all 300 ok [RES §3.3].

### Slide 11: how a posterior is scored

- **Title:** We score by the density at the truth, and a difference counts only if it beats training noise
- **Figure:** F10, full, with equations (4) and (5) under the two panels.
- **Equation (4)**, NLL at the truth: $\ell_i(E)=-\log q_E\big(\theta^{\rm ref}_i\mid x_i\big)$, and $L=\frac{1}{n}\sum_{i=1}^{n}\ell_i(E)$.
- **Equation (5)**, rule G.2: resolved $\iff |\bar D(E,E')|>2\,\sigma_{\rm seed}$ **and** the 95 % donor interval of $\bar D$ excludes 0.
- **Notes:** "The main score is simple. Take the posterior for a window, read its density at the true parameters, take minus the log. Lower is better. The uniform prior has density 1 everywhere, so it scores exactly 0. Negative means more density on the truth than the prior gives. Minus 7 nats means about a thousand times the prior's density. A posterior that is sharp and wrong scores worse than the prior. To compare two arms we take their difference window by window, average per donor, then over donors: that is D-bar. A difference counts only if two things hold. Its 95 % interval, from resampling donors, must exclude zero. And it must be larger than twice the spread between single retrainings, sigma-seed. The rule keeps the two clauses separate on purpose: the first is about which donors we happened to have, the second about which seeds we happened to draw." Tags: eq. (4), the prior's 0 [RES §3.3, eqs. (1)-(2); CODE: SPEC 2.5]; D-bar, sigma_seed, rule G.2 [RES §3.8 eqs. (10)-(11); KB: deck pack 08, G.1-G.2]; e^7 = 1,097 [OWN, arithmetic]; the curves of F10 are an illustration, not data.

### Slide 12: result 1, the searches

- **Title:** Every search went from worse than the prior to 3-4 nats better, and was still improving
- **Figure:** F11, full.
- **On the slide** (right of the figure, or as a footer line): finalists' L from -3.12 to -4.17; 4 of 5 found in the last round.
- **Notes:** "Each curve is the best L found so far in one search. The faint dots are single trials and the rings the finalists. At the start, a short untuned training is worse than the prior: the median of the random round is +1.5 to +1.75 nats. By the end every search is 3 to 4 nats better than the prior. Four of the five finalists come from the last round, and three searches had their best median in the last round. More rounds would probably have found better configurations. Two things to remember for later. First, ranking on simulations did not mislead on the cohort: within a search, the simulated and the cohort scores order trials almost identically. Second, every term search picked a small weight, lambda between 0.0015 and 0.013. A larger weight went with a worse simulated score, so the criterion pushed the terms towards off." Tags: medians +1.49 to +1.75, finalists -3.122 to -4.173, rounds, Spearman 0.85-0.98, lambda, Spearman(log lambda, L) +0.32 to +0.70 [DATA: `search_<q>.json`, `finalist_config.json`; RES §3.3]; "more rounds would help" [OWN, Likely].

### Slide 13: result 2, members and the mixture

- **Title:** The members overfit; averaging five of them gains 0.7-1.1 nats
- **Figure:** F12, full.
- **Notes:** "Left: the five members of the A1 ensemble. The training loss keeps falling to between -7 and -11 nats. The held-out loss stops improving around epoch 10 to 25: they overfit, and early stopping keeps the best epoch, the circles. Right: for each ensemble, the five members, their mean, and the mixture. The mixture is 0.68 to 1.09 nats better than its average member, because averaging densities rescues the windows where one member is right and the others are not. The gain is largest for A1, whose flow is the widest and the most overfit. One warning: the six-pass ranking did not survive full training. The search with the best trial, S-A2-R0, ended with the worst ensemble." Tags: training -7 to -11, epochs 10-25, best epochs 19, 14, 23, 24, 11 [DATA: `search_S-A1.json` npe, val_npe; RES §3.4]; gains 0.676-1.085, ensemble L -5.858 to -6.838 [DATA: `<stem>_sim.json`; RUN: `figs_results.py`]; the reading of the gain [RES §3.4 eq. (4)].

### Slide 14: result 3, what the posterior learns per parameter

- **Title:** Six parameters are learned well, two weakly, the two free ones not at all, in every arm
- **Figure:** F13, full.
- **On the slide:** contraction = 1 - (posterior variance) / (prior variance): 0 means as wide as the prior, 1 means a point.
- **Notes:** "Each cell says how much narrower the posterior is than the prior on one parameter, averaged over simulated test windows. Six parameters are pinned down well, 0.74 to 0.97, nearly the same in every arm (within 0.05). Intraburst rate and participation mean are learned weakly. The two free parameters stay near the prior's width. That is the right answer: they barely change the window statistics, so the data cannot identify them. So the extra terms did not change what a window identifies. Where the ensembles differ, on the weak axes, the pattern follows the search, not the arm." Tags: contraction values [DATA: `<stem>_sim.json`, recovery.contraction; RES §3.5]; the free axes do not move the window statistics [LOG D-132 (3), finding F-bk, as cited in RES §3.5].

### Slide 15: result 4, calibration and the box edges

- **Title:** Inside the box the ensembles are calibrated or conservative; the failures are truths at the box edges
- **Figure:** F14, full.
- **Notes:** "Left: coverage. For each credibility level we ask how often the true parameters fall inside the region of that credibility. The diagonal is perfect calibration. Overall, the 95 % regions hold the truth only 85 to 87 % of the time. The curve follows the diagonal and then stops short, which points to a fixed group of windows that no region reaches. Right: those windows. In 10.8 % of simulated test windows a true coordinate lies within 0.0067 of a box edge. The flow puts almost no mass there by construction, so those windows score +7.5 to +8.6 nats, worse than the prior. Without them, the 95 % regions hold the truth 96 to 98 % of the time: calibrated or conservative, the safe direction. Single members are overconfident; the mixture repairs that. This is an open point in the decision log, common to all arms. The cohorts are barely affected: only one R3 donor has a truth at an edge." Tags: C(0.95) 0.854-0.874 overall and 0.957-0.980 interior, 10.8 % edge rows, NLL +7.5 to +8.6 [RUN: `figs_results.py`; RES §3.6, check 1]; epsilon = sigmoid(-5) = 0.0067 and the flow's spline bound [LOG, open item "The flow's edge support before S8"; RES §3.6]; members overconfident [DATA: coverage_members; RES §3.6]; conservative is the safe direction [KB: "A Trust Crisis In Simulation Based Inference ...", p. 3].

### Slide 16: result 5, R0 against R3

- **Title:** R0 is scored well on every window; R3's penalty sits in 18 windows
- **Figure:** F15, full.
- **Notes:** "One dot per window, three ensembles per cohort. On R0 every window is better than the prior, and the means are -7.1 to -7.7 nats: roughly 1,300 to 2,300 times the prior's density at the truth. R3's means look 1.2 to 1.9 nats worse. That penalty is not spread out. Eighteen windows, the black diamonds, score +20 to +25 nats on average. A classifier test (L-C2ST) is certain about exactly these 18, under all three ensembles. They fall at most one per well and are about as many as the contamination should produce: most likely the contaminated windows, though the bank's flags have not been checked yet. The open squares are one donor whose true background sits at a box edge. On the other 295 windows every arm scores -7.8 to -8.1, as on R0. The drift and the range shift cost nothing visible." Tags: R0 means -7.142 to -7.729, max window -0.25 [DATA: `<stem>_R0.npz`; RES §3.7]; R3 means -5.877 to -5.942, flagged 18 at +19.6 to +25.2, the 295 others -7.761 to -8.127, donor 0 at theta*_9 = 0.004 [RUN: `figs_results.py`; RES §3.7, check 2]; "contaminated" [OWN, Likely; open in RES §3.7]; e^7.14 = 1,264, e^7.73 = 2,273 [OWN, arithmetic].

### Slide 17: result 6, the comparisons

- **Title:** None of the 15 comparisons passes the rule: training noise is larger than every difference
- **Figure:** F16, full (it is 7.2 in tall; give it the whole slide with a one-line title).
- **Notes:** "Six comparisons on two or three data sets each: fifteen pairs. Each dot is D-bar with its 95 % donor interval. The grey band is plus or minus twice sigma-seed. No dot leaves its band: the bands are 0.64 to 1.23 nats wide on each side, and no difference exceeds 0.69. Ten of the fifteen intervals exclude zero. So the donors are not what limits us; the spread between retrainings is. Two readings. On R0, A1's ensemble is ahead by about 0.57 nats, but its members are not better than the others'. The lead is its larger mixture gain, and its flow is also several times wider, so term and architecture are mixed. On simulations, the two searches of the same arm differ by 0.7 to 0.8 nats, as much as any arm differs from A1. The search itself is a source of spread that sigma-seed does not count." Tags: 15 pairs, D-bar, intervals, 2 sigma_seed 0.642-1.231, max |D-bar| 0.688, 10 of 15 [DATA: `paired_<name>_<D>.json`; RES §3.8, check 3]; members' means and gains on R0 [RES §3.8, T8]; readings [OWN, Likely, as in RES §3.8].

### Slide 18: result 6b, the trade under R3's tie

- **Title:** On R3 the tie hides a trade: A5 is better on ordinary windows, much worse on contaminated ones
- **Figure:** F17, full.
- **Notes:** "This splits R3's mean difference into its two parts. On the 306 ordinary windows A5 gives the truth more density than A1, 0.36 nats per window. On the 18 flagged windows it is 5.6 nats per window worse. Weighted by how many windows each group has, the two cancel almost exactly: -0.34 plus +0.31 gives the tie of -0.03. A2 shows the same sign pattern, smaller. Treat this as exploratory. The split was chosen after seeing the data, though by a criterion that uses no true parameters. It is one cohort and one search. What it shows is that a mean NLL can be decided by 6 % of the windows." Tags: -0.362, +5.64, -0.112, +0.73, all -0.028, -0.065, contributions [RUN: `figs_results.py`; RES §3.8]; intervals [-0.512, -0.216] and [+3.93, +7.31] if asked [RES §3.8, check 2]; exploratory status [RES §3.8].

### Slide 19: take-home

- **Title:** The pipeline works end to end; this run cannot yet tell the arms apart
- **Figure:** F18, full.
- **Notes:** "Four messages. The optimisation worked but had not finished. Training gives overfit members, and the mixture repairs them. Calibration is good inside the box and fails at its edges, for a reason we understand. The comparison has no verdict: training noise is larger than every difference, and the searches chose term weights so small that the terms were nearly off. Before any conclusion on A2 or A5, the open questions are whether to run longer searches, what to do about the edge support, and confirming the contaminated windows. Two things to discuss: a ranking criterion that does not push the terms towards off, and a second ensemble per arm, so we measure the seed noise of an ensemble rather than of a single member." Tags: all from slides 12-18; the open points [RES §3.3, §3.6, §3.7 (Open)]; the two discussion items [OWN, from RES §3.3 and §3.8 readings].

## 4. Backup slides

| # | Title | Content | Source |
|---|---|---|---|
| B1 | The DSN loss in one picture | Triplets (anchor, same-condition positive, other-condition negative) mined in each class-balanced batch on unit-norm summaries. Plus the separation term: pairwise cosines of the condition-mean directions pulled to -1/(K-1), i.e. -0.5 at K = 3. Reuse the A2 card of F08, enlarged. | [CODE: P2 eqs. (P2.1)-(P2.8)] |
| B2 | Why T should equal p_eff | For honest posteriors and true replicates, E[T] = p_eff = d_theta - tr(Sigma_0^-1 C): the number of directions the data constrain (P3 sec. 3.3). The MC correction d_theta/S_mc and the clamps. | [CODE: P3 sec. 3.2-3.3] |
| B3 | What the finalists look like | Table: log10 lambda (-2.15, -2.81, -2.42, -1.90), flow hidden features (512, 147, 75, 88, 92), transforms (12, 12, 9, 7, 8), embedding (64, 37, 64, 52, 16), weight decay. Point: A1's flow is 3.5-7 times wider. | [DATA: `finalist_config.json`; RES §3.3 T1b] |
| B4 | The 15 comparisons as a table | The table of RES §3.8 (D-bar, interval, 2 sigma_seed, ratio, verdict). | [RES §3.8] |
| B5 | Scoring R3 at the shifted truth | R3dag scores R3's windows at theta-dagger = theta* + 0.105 on axes 8-9, the parameters at which the unshifted simulator reproduces the range shift. 9 of 27 donors leave the box and are dropped. Against theta-dagger, A1 leads A2 by 0.43 nats in 16 of 18 donors (ratio 0.68), the closest a cohort comparison comes to a verdict. | [RES §3.8, check 4] |
| B6 | The other diagnostics | PPC: without the 18 flagged windows, R3's outside rates equal R0's, and both are below nominal. The drift does not move the eight window statistics. L-C2ST: the flagged windows saturate at 0.25. MMD between summaries of simulations and cohort: the terms did not lower it; it measures the change of parameter law, not the gap. | [RES §3.7, result 7 of §3.8] |
| B7 | What changed from v3 | `fragment_duty` became a label axis (8 instead of 7). The range shift moved to axes 8-9 only. The term searches no longer nest A1 trials. | [CODE: handoff sec. 3, "Changed from v3"; LOG D-132] |
| B8 | Pipeline figures, if asked | `figures/ensembles/S-A1_A1/acc_recovery_sim.png` (posterior mean and 90 % interval against truth per axis); `figures/ensembles/S-A1_A1/cal_coverage.png` (members against mixture); `figures/across/dbar.png` (the pipeline's own D-bar plot) | the pipeline's figure bundle `synth_v4_figures_png.tar.gz` |

## 5. The five equations

Notation: $\theta\in(0,1)^{10}$ the parameters in unit-box coordinates; $x\in\mathbb{R}^{18000}$ one window; $h_\psi$ the summary network with weights $\psi$; $q_\omega(\theta\mid z)$ the flow's density over $\theta$ for each fixed summary $z$, weights $\omega$; $p_{\rm sim}(\theta,x)$ the bank's law (uniform $\theta$, simulated $x$); $p_{\rm real}$ the cohort's law; $E$ an ensemble and $q_{E,m}$ its member $m$.

**(1) The joint objective** [CODE: E4 eq. (plan 1), (plan 1a)]

$$\mathcal{L}(\psi,\omega)=\mathcal{L}^{\rm sim}_{\rm NPE}(\psi,\omega)+\lambda_{\rm dsn}\,\mathcal{L}^{\rm real}_{\rm DSN}(\psi)+\lambda_{\rm rep}\,\mathcal{L}^{\rm real}_{\rm rep}(\psi,\omega),\qquad \mathcal{L}^{\rm sim}_{\rm NPE}(\psi,\omega)=\mathbb{E}_{(\theta,x)\sim p_{\rm sim}}\big[-\log q_\omega(\theta\mid h_\psi(x))\big]$$

Reading: the total loss is the expected NLL on simulated pairs, plus two weighted terms computed on cohort windows. The DSN term depends on $\psi$ alone. The replicate term is evaluated through the flow, but its gradient is applied to $\psi$ only: in the update, $\omega$ moves with the NPE term alone [CODE: E4 eq. (E4.3)]. Arms: A1 has $\lambda_{\rm dsn}=\lambda_{\rm rep}=0$; A2 has $\lambda_{\rm rep}=0$; A5 has $\lambda_{\rm dsn}=0$.

Unicode fallback: `ℒ(ψ, ω) = ℒ_NPE^sim(ψ, ω) + λ_dsn ℒ_DSN^real(ψ) + λ_rep ℒ_rep^real(ψ, ω)`

**(2) Replicate consistency** for windows $g$ and $g'$ from the two wells of one donor [CODE: P3 eqs. (P3.2)-(P3.7)]

$$\hat T_{gg'}=\hat\Delta_{gg'}^{\top}(2\bar C)^{-1}\hat\Delta_{gg'}-\frac{d_\theta}{S_{\rm mc}},\quad \hat\Delta_{gg'}=\hat m_g-\hat m_{g'},\quad \bar C=\tfrac12(\hat C_g+\hat C_{g'}),\quad \hat p_{\rm eff}=d_\theta-\mathrm{tr}\big(\Sigma_0^{-1}\bar C\big),$$

$$\ell_{\rm rep}=\Big(\log\max(\hat T_{gg'},10^{-8})-\log\max(\hat p_{\rm eff},10^{-3})\Big)^2 .$$

Reading: $\hat m_g$ and $\hat C_g$ are the mean and covariance of $S_{\rm mc}$ posterior draws for window $g$, so $\hat T$ is the squared gap between the two posterior means, measured in the posteriors' own average spread. The term $d_\theta/S_{\rm mc}$ removes the part the Monte Carlo error of the two means adds. $\hat p_{\rm eff}$ counts how many of the $d_\theta=10$ directions the data narrow down: $\Sigma_0=\tfrac1{12}I$ is the prior's covariance on the unit box, so $\hat p_{\rm eff}=10-12\,\mathrm{tr}(\bar C)$. The loss is zero when $\hat T=\hat p_{\rm eff}$ and grows on both sides. The two clamps keep the logarithms finite.

Unicode fallback: `T̂ = Δ̂ᵀ (2C̄)⁻¹ Δ̂ − d_θ/S_mc ;  p̂_eff = d_θ − tr(Σ₀⁻¹ C̄) ;  ℓ_rep = (log T̂ − log p̂_eff)²`

**(3) The ensemble** [CODE: SPEC 2.5 (E1); RES eq. (3)], for each fixed window $x$ and each $\theta\in(0,1)^{10}$:

$$q_E(\theta\mid x)=\frac{1}{5}\sum_{m=1}^{5}q_{E,m}(\theta\mid x)$$

Reading: the five members' densities are averaged, not their parameters or their summaries. The mixture puts mass wherever any member does. Its NLL at a window is at most the members' average NLL, and at most $\log 5=1.61$ nats worse than the best member [RES eq. (4)].

Unicode fallback: `q_E(θ | x) = (1/5) Σ_{m=1..5} q_{E,m}(θ | x)`

**(4) NLL at the truth** [RES eqs. (1)-(2); CODE: SPEC 2.5], for a data set of rows $(x_i,\theta^{\rm ref}_i)$, $i=1..n$:

$$\ell_i(E)=-\log q_E\big(\theta^{\rm ref}_i\mid x_i\big),\qquad L=\frac{1}{n}\sum_{i=1}^{n}\ell_i(E)$$

Reading: one number per window in nats, lower is better. $\theta^{\rm ref}_i$ is the row's own $\theta$ on simulations and the donor's hidden $\theta^*$ on a cohort. The uniform prior scores exactly 0. On simulations $L$ is the search criterion; on a cohort it is the endpoint.

Unicode fallback: `ℓ_i(E) = −log q_E(θ_i^ref | x_i),  L = (1/n) Σ_i ℓ_i(E)`

**(5) The decision rule G.2** [RES eqs. (10)-(11); KB: deck pack 08]

$$\bar D(E,E')=\frac1N\sum_{d=1}^{N}\frac{1}{n_d}\sum_{i:\,d(i)=d}\big(\ell_i(E)-\ell_i(E')\big),\qquad \text{resolved}\iff|\bar D(E,E')|>2\sigma_{\rm seed}\ \text{and}\ 0\notin\mathrm{CI}_{95}\big(\bar D(E,E')\big)$$

with $\sigma_{\rm seed}=\max\big(\mathrm{SD}_m[\bar\ell(E,m)],\,\mathrm{SD}_m[\bar\ell(E',m)]\big)$, the larger spread of single members' mean NLLs. The interval comes from 10,000 donor resamples. $\bar D<0$ favours $E$. Keep the two clauses separate on the slide; the first deck's style guide forbids merging them.

Unicode fallback: `resolved ⇔ |D̄(E, E′)| > 2σ_seed  and  0 ∉ CI₉₅(D̄)`

## 6. Figure manifest

All figures are 13.33 x 6.6 in (F16: 13.33 x 7.2) in the deck palette. **Schematic** means drawn by hand: its numbers are configured values. **Plotted** means drawn from the evaluation files or from a fresh simulation.

| File | Slide | Kind | What it shows | Data | Script |
|---|---|---|---|---|---|
| F01_idea | 2 | schematic | arm S and arm R feed one network; theta* reaches only the score | sizes from synth.json | figs_schematics.py |
| F02_simulator | 3 | plotted | raster of 100 neurons, 40 s, at condition 3's centre; x below; zoom on one burst in two fragments | fresh simulation | gen_raster.py, figs_schematics.py |
| F03_parameters | 4 | plotted | the 10 axes, physical ranges, the 3 condition centres, uniform free axes, R3's shift | BENCH_AXES, centres | centres.py, figs_setup.py |
| F04_condition_traces | 5 | plotted | one 180-s window per condition centre, with a 20-s zoom | fresh simulation | gen_traces.py, figs_setup.py |
| F05_data_sets | 6 | plotted | bank against R0 in the plane of the centres; how a cohort is built | theta_ref of S-A1_A1_sim/R0.npz | figs_setup.py |
| F06_gap | 7 | plotted | one well with and without the gap; drift; contamination zoom; range shift | fresh simulation | gen_traces.py, figs_setup.py |
| F07_model | 8 | schematic | summary network, flow, the three losses and what each trains, the stop-gradient | E4, P2, P3 | figs_schematics.py |
| F08_arms | 9 | schematic | A1, A2 (clusters, means 120 degrees apart), A5 (two wells' posteriors, gap Delta) | P2, P3 | figs_schematics.py |
| F09_procedure | 10 | schematic | search (60 trials), 5 members, mixture, scoring; the 5 searches; the split | synth.json | figs_schematics.py |
| F10_scoring | 11 | schematic | NLL at the truth (sharp-right, prior, sharp-wrong); the two-clause rule | illustrative curves | figs_schematics.py |
| F11_searches | 12 | plotted | best-so-far L per search, trials, finalists, rounds | search_<q>.json | figs_results.py |
| F12_members_mixture | 13 | plotted | S-A1 members' learning curves; members, mean and mixture per ensemble | search_<q>.json, <stem>_sim.json | figs_results.py |
| F13_contraction | 14 | plotted | contraction per axis and ensemble, grouped | <stem>_sim.json | figs_results.py |
| F14_calibration_edges | 15 | plotted | coverage curves all against interior; NLL against distance to the edge | <stem>_sim.npz | figs_results.py |
| F15_r0_r3_windows | 16 | plotted | per-window NLL on R0 and R3, flagged windows, donor 0 | <stem>_R0/R3.npz | figs_results.py |
| F16_forest | 17 | plotted | D-bar, intervals and the 2 sigma_seed bands for the 15 pairs | paired_<name>_<D>.json | figs_results.py |
| F17_r3_trade | 18 | plotted | contributions of ordinary and flagged windows to R3's tie | <stem>_R3.npz | figs_results.py |
| F18_take_home | 19 | schematic | four verdicts and the open points | the results document | figs_schematics.py |

Captions for the notes (one line each, the claim the figure supports):

- **F11:** the best L found kept falling until the last or second-to-last round in every search [DATA].
- **F12:** the training loss of every S-A1 member keeps falling while the held-out NLL stalls; every mixture is 0.68-1.09 nats better than its members' mean [DATA].
- **F13:** 0.74-0.97 on six axes, 0.10-0.29 on two, 0.02-0.14 on the free axes, in all five ensembles [DATA].
- **F14:** the coverage deficit disappears without the 10.8 % edge rows; edge rows score above 0, interior rows about -8 [RUN].
- **F15:** on R3, 18 windows flagged by L-C2ST under all three ensembles carry the penalty; R0 has no window above 0 [RUN].
- **F16:** no D-bar leaves its +/- 2 sigma_seed band; 10 of 15 intervals exclude 0 [DATA].
- **F17:** the R3 tie is -0.34 (306 ordinary windows) + 0.31 (18 flagged) for A5 - A1, and -0.11 + 0.04 for A2 - A1 [RUN].

## 7. Style

| Role | Colour | Used for |
|---|---|---|
| A1, simulations, the NPE term | `#2a78d6` blue | arm S, the bank, the NPE loss, A1 |
| A2, the cohort's labels | `#eb6834` orange | arm R, the DSN loss, A2 |
| A5, replicates | `#1baf7a` green | the replicate loss, A5 |
| conditions 1, 2, 3 | `#4a3aa7` violet, `#e87ba4` magenta, `#eda100` ochre | the three conditions and their windows |
| problems | `#e34948` red | R3's gap, edge rows, the stop-gradient; **never beside the orange** |
| text, grid, boxes | `#0b0b0b`, `#52514e`, `#e6e5e1`, `#f0efec` | |

These roles agree with the first deck pack's style guide: blue for the simulated stream, orange for the real stream, green for the replicate stream, red for blocked or failing. One change: violet, magenta and ochre are here the three conditions. In the first pack they meant the target, the realisation and the nuisance. So draw reference lines (the diagonal, p_eff) in ink, not violet. The arm colours were checked with the dataviz palette validator. The green passes with a contrast warning, so the arms always carry a text label in the figures; keep them when re-drawing.

Typography: one sans for everything. The figures use Inter, and the SVGs keep their text as text, so they take the deck's font. Titles 28-32 pt, at most two lines; body 18-22 pt; at most four bullets of one line each; equations in the deck tool's equation editor from the LaTeX above, otherwise the Unicode fallback. Slide chrome neutral (white), the figures carry the colour. No stock images, no gradients.

## 8. Rebuilding the figures

From the folder that holds `scripts/`, with the evaluation bundle extracted to `data/` (so that `data/runs/synth/evaluation/metrics/` exists) and an SBI checkout for the two simulation scripts:

```bash
python3 scripts/centres.py <SBI checkout> data/runs/synth/evaluation/metrics   # the centre check of sec. 9.3
python3 scripts/gen_traces.py <SBI checkout> data/traces.npz                    # F04, F06 windows
python3 scripts/gen_raster.py <SBI checkout> data/raster.npz                     # F02 raster
python3 scripts/figs_setup.py .        # F03-F06
python3 scripts/figs_results.py .      # F11-F17; prints every number it draws
python3 scripts/figs_schematics.py .   # F01, F02, F07-F10, F18
```

Python 3 with NumPy, SciPy and Matplotlib; the Inter font, or the deck's sans (change `font.family` in `scripts/deckstyle.py`). The simulation scripts import the pipeline's own generator, nuisance and gap code from `hpc/joint/stage1` and `hpc/dsn`, so the windows are made exactly as the bank's are. These scripts are illustrations, not pipeline code.

## 9. Sources and checks

### 9.1 What was read

- The synth v4 evaluation bundle (`synth_v4_results.tar.gz`, report of 2026-11-16 13:31): `metrics/` (search, per-ensemble and paired files) and `runs/synth/<q>/synth_best/`. [DATA]
- The results document `synth_v4_evaluation_results.md` (2026-11-16), whose numbers were checked against the same files when it was written. [RES]
- The SBI code and design notes on `sci/pipeline` 54edb32: `bench_burst_provider.py`, `bench_burst_generator.py`, `latent_gap.py`, `latent_sbi_simulator.py`, `hpc/dsn/latent_burst_generator.py`, E4 (joint objective), P2 (DSN loss), P3 (replicate statistic); the handoff `HANDOFF_2026-11-16_synth_v4_results_commentary.md` (sec. 3). [CODE]
- Project knowledge: the first deck pack's style guide (11) and its decision chapter (08, rule G.2); the PDFs cited in the results document: Hermans et al., *A Trust Crisis In Simulation-Based Inference?* (TMLR 2022), the *Simulation-Based Inference: A Practical Guide*, and Lemos et al., *Sampling-Based Accuracy Testing of Posterior Estimators* (ICML 2023). [KB]

### 9.2 Checks run for this pack

- Every number drawn in F11-F17 is printed by `figs_results.py` and equals the results document's tables: finalist L, round-0 medians, members' means, mixtures and gains, contractions, C(0.95) all and interior, the edge share, R0 and R3 means, the 18 flagged windows with their means, the 15 D-bars and the trade decomposition (A5 - A1: -0.362 x 306/324 + 5.64 x 18/324 = -0.028). The F17 bootstrap intervals use another seed than the results document and differ by at most 0.08 nats. The figure does not draw them, and the notes quote the document's. [RUN]
- The members' best epochs (argmin of the held-out NLL) are 19, 14, 23, 24, 11 for S-A1, as in the results document. [RUN]
- The R3 range shift moves each free axis's physical range up by 0.105 of its width before the affine map. It does not move theta itself. So `participation_kappa` 3-15 becomes 4.26-16.26 and `background` 0.01-0.06 becomes 0.01525-0.06525. [CODE: `phi_to_bench_params`]
- The drift's amplitude in the example well is 0.006 on a mean of 0.042, i.e. 0.15 of the mean, period 120 s, as configured. [RUN: `gen_traces.py` well, R3 minus clean]

### 9.3 A finding while drawing F03-F05: the cluster's condition centres are mirrored

The condition centres come from the DSN generator's `_class_center_vectors(3, 8, "simplex")`. It takes the simplex basis from an SVD of the centring projector $I-\mathbf{1}\mathbf{1}^\top/3$, whose singular value 1 is repeated, so that basis is fixed only up to a rotation or a sign. Computed in this sandbox, the centres do not fit the cohorts' recorded truths. The same construction with the second basis vector's sign flipped does:

| Basis | SSE of 54 donors' label coordinates about their centres | Residual SD |
|---|---|---|
| this sandbox's SVD | 7.757 | 0.134 |
| second vector's sign flipped (used in the figures) | 4.440 | 0.101 |
| expected if the centres are right (tau = 0.10) | 4.32 +/- 0.29 | 0.100 |

The distances between centres, 0.441934, are the same in both bases. Only the orientation differs. The results are unaffected, because every score uses the recorded theta*. Two things follow beyond this deck, neither acted on. First, a cohort rebuilt on another machine may place the conditions differently, so its theta* would not be reproduced. Second, the centres are not written down anywhere a reader can check them. Both are for the pipeline, not the deck. [RUN: `centres.py`; DATA: theta_ref of R0 and R3]

### 9.4 Literature searches

- PubMed, "simulation-based inference neural posterior estimation calibration": 11 records. They are applications of NPE in epidemiology, cell migration, diffusion MRI, acoustics and liquid biopsy, plus one calibration method, CP4SBI (Cabezas et al., Phil. Trans. A 2026, doi:10.1098/rsta.2025.0069). Its abstract (abstract only, not read in full, no number used) says that SBI posteriors are often miscalibrated with undercovering credible regions. Nothing in this deck depends on it.
- PubMed, "neural posterior estimation summary network misspecification simulation-based inference": 0 records.
- bioRxiv, the bioinformatics category over the last 30 days (30 records): nothing on SBI calibration or ensembles. The connector has no keyword search.
- The method statements of the deck rest on the code and design notes and on the knowledge-base PDFs above, read in full when the results document was written.

### 9.5 Open, as the results document leaves them

Whether to run the searches longer; the flow's edge support (bound 5 or 10); confirming the 18 flagged windows against the bank's contamination flags (the data overview, block 23, has not been run); why the L-C2ST rejection rate is higher on R0 than on simulations at equal mean statistic; a multiplicity correction over the 15 verdicts. [RES §3.3, §3.6, §3.7, §3.8]

## 10. Questions the supervisor may ask

| Question | Short answer | Where |
|---|---|---|
| Why is A1 better on R0? | Its ensemble is; its members are not. The lead is its larger mixture gain, and its flow is 3.5-7 times wider, so arm and architecture are mixed. | slide 17, B3 |
| The intervals exclude zero; why "not resolved"? | Rule G.2 also asks the difference to beat twice the spread between single retrainings. None does: the ratios are at most 0.82. | slides 11, 17 |
| Is sigma_seed fair? | It is strict. It is the spread of single members, while D-bar compares mixtures of five, which vary less. A second ensemble per arm would measure the right spread. | slide 19 |
| Did the terms do anything? | At the weights the searches chose (lambda 0.0015-0.013) they were nearly off. "Not resolved" is not a measured zero. | slides 12, 19 |
| Are the posteriors miscalibrated? | Only at the box edges, a known property of the flow shared by all arms. Inside, they are calibrated or conservative. Single members are overconfident. | slide 15 |
| What does -7.7 nats mean? | On average, about 2,300 times the prior's density at the true parameters. | slides 11, 16 |
| Are the 18 windows really the contaminated ones? | Very likely: same 18 under three ensembles, at most one per well, about the expected count. Not yet confirmed against the bank's flags. | slide 16 |
| Why do the free axes stay at the prior? | They barely change the window statistics, so the data cannot identify them. That is the correct posterior. | slide 14 |
| What next? | Decide on longer searches and a criterion the terms can win on, add a second ensemble per arm, and settle the edge support. Then re-run. | slide 19 |
