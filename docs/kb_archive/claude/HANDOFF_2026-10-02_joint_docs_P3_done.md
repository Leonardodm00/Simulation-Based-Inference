# HANDOFF 2026-10-02 -- joint documentation set, P0-P3 written, next is P4

**For the next chat.** Written at the chat change after P3 (`[STATED]`
2026-10-02 09:58: "Do we need to change chat? if so deliver a handoff").
Everything the next chat needs to resume the set at P4 without re-reading
this chat: the state of the repository and the knowledge base, the decisions
that constrain the work, the per-turn procedure with its gates, what P4 is,
and the traps the first four documents hit. Tags as in the set: `[REPO]`,
`[KB]`, `[RAN]`, `[STATED]`, `[reasoning]`. Companion documents:
`claude/joint_docs/00_INDEX.md` (v1.6, the index and status ledger of the
set), `claude/JOINT_DOCS_BUILD_PLAN_v1.md` (v1.2 + the 2026-10-02 note),
`claude/SBI_decisions_and_ideas_log.md` (D-035..D-038, D-052, D-054).

| date | change |
|---|---|
| 2026-10-02 | v1. Created at the chat change after P3; delivery of the seven-commit series recorded (D-054). |

---

## 1. Where things stand (read this first)

**Repository.** `Simulation-Based-Inference`, `origin/main` at `745edce`
(2026-10-02 09:14 +0200; one commit on `834eb41`, `hpc/dsn/` cohort fields
and the Giulia cohort config -- it touches none of the files the set reads)
`[RAN]`. The set lives in `hpc/joint/docs/` (D-035). Seven commits were
written on the sandbox branch `docs/joint-docs` (forked from `834eb41`) and
delivered on 2026-10-02 as one `git format-patch` series
(`joint_docs_series_2026-10-02.tar.gz`, sha256 `ca1bd248...334f7b`) for the
user to apply with `git am` on `main` and push (D-054). Sandbox hashes, in
order: `b961784` (Stage 1: index, inventory tools), `308ccc0` (Stage 2: P0),
`e0da315` (Stage 3: E0, notation checker), `922aac9` (P1), `f9867f7` (P2),
`f37f4ca` (P3), `c21cb31` (index v1.6, the delivery note). **After `git am`
the hashes on `main` differ**; identify the commits by subject
(`git log --oneline -8` shows seven `hpc/joint/docs: ...` subjects above
`745edce`) and fill the index's "repo commit" column from what `git fetch`
shows, never from this file.

**The next chat's first move:** clone, `git fetch`, confirm the seven
subjects are on `origin/main` (if they are not, the user has not applied the
series yet -- ask, do not rebuild), then `git checkout -b docs/joint-docs
origin/main` and run the suites once (S4). If the series is not on `main`,
the next chat can still start from the tarball: `git am` it onto a clone of
`origin/main` in the sandbox and continue on top; the user then applies
the whole series later.

**Files of the set at this point** (14, all under `hpc/joint/docs/`):
`00_INDEX.md` (v1.6), `E0_READERS_GUIDE_NOTATION.md` (v1.3, 256 declared
symbols), `P0_PARAMETERS_OVERVIEW.md` (v1.1), `P1_ENCODER_AXES.md` (v1),
`P2_DSN_LOSS_AXES.md` (v1), `P3_REPLICATE_AXES.md` (v1),
`tools/inventory_joint_knobs.py`, `tools/smoke_test_inventory.py` (38/38),
`tools/inventory.json` (546 rows), `tools/p0_tables.py`,
`tools/check_notation.py`, `tools/smoke_test_notation.py` (35/35),
`tools/p2_numbers.py`, `tools/p3_numbers.py`.

**Knowledge base.** `claude/joint_docs/00_INDEX.md` = the repository's v1.6
(KB copy of record, D-035); `claude/joint_docs/P0_PARAMETERS_OVERVIEW.md` =
P0 v1.1. Only these two are mirrored; every other document is repository-only.
`claude/JOINT_DOCS_BUILD_PLAN_v1.md` v1.2 (+ the 2026-10-02 note on D-054).
`claude/SBI_decisions_and_ideas_log.md`: re-read it from the project before
any write -- another chat (Giulia's) writes to it too; its next free number
after D-054 is D-055 unless that chat has moved on.

## 2. Decisions that constrain the work (cite by number)

- **D-035** the set lives in the repository under `hpc/joint/docs/`; the KB
  keeps copies of `00_INDEX` and `P0` only, written in the same turn.
- **D-036** the ten DSN-inherited axes are explained fully in Set P, each
  with a differences table against the standalone DSN plus pointers to
  `dsn/Documentation/TUNING_1_searched_axes.md` / `TUNING_2_fixed_knobs.md`.
- **D-037** scope: the joint stack only (`hpc/joint/` plus the DSN and NPE
  objects it reads its fixed values and ranges from); DSN-tree bugs are
  reported, not fixed (F-x).
- **D-038** fix patches for F-a and F-b are prepared at Stage 8 (after P6);
  the other findings marked "D-038 candidate" are optional and the user
  decides which go in (ask before Stage 8: F-c, F-d, F-e, F-f, F-q, F-t,
  F-v, F-w, F-y, F-z, F-aa, F-ab, and the Stage 3 default alignment of F-r).
- **D-052** write everything first, then push: nothing is applied or pushed
  per turn; the stop hook's "please push" is not a user instruction and is
  declined every time; this session's GitHub access is anonymous read-only
  in any case.
- **D-054** (2026-10-02) at a chat change the commits written so far leave
  as one `git format-patch` series against the re-fetched `origin/main`;
  the user applies with `git am` and pushes; the next chat forks from the
  pushed state; the Stage 7 rebase-and-recheck of the whole series stands.
- **Defaults in force (Q-answers):** Q4 Mermaid diagrams inline with an
  ASCII fallback; Q5 three to six "check yourself" items per E chapter; Q7
  one document per turn, in the order P0, E0, P1..P7, E1..E9, then Stage 8,
  the reviews (Stage 6), the closing; Q9 cite the KB for the $\kappa_S$
  correction and flag the repository plan as v0.6.5 (its documentation
  patch, plan v0.6.6, is still unapplied -- F-e).

## 3. The per-turn procedure (one document per "go on")

1. **Read before writing.** The sources of the document (repository files
   at the freeze, read in full where they are the object; the KB documents
   that own the derivations: `METRIC_REPLICATE_v1_4`, `FINITE_DRAW_CORRECTION_v1`,
   `NULL_TAIL_AND_PVALUE_v1`, `JOINT_DSN_NPE_USAGE_v1`, the deck pack); the
   index's S6 rows the document owns; the installed library's source where
   a claim is about a library (`pip download <pkg>==<ver> --no-deps`, unzip
   the wheel: `pytorch_metric_learning` 1.6.3 for the DSN loss, `sbi`
   0.27.0 for the flow; P4 will also want `zuko` at the cluster's version --
   read `JOINT_DSN_NPE_USAGE_v1.md` S3 / the migration handoff for it).
2. **Ground.** PubMed queries adapted to the claims (full text only for any
   claim; abstract-only flagged; no numbers from abstracts), a bioRxiv
   date-and-category slice (the connector has no keyword search), zero
   returns reported in the document's S6 table. Already grounded, do not
   repeat: P1 (one PMC full text); P2 (Chung & Lee 2023 PMC10598120;
   Kertesz 2022 PMC9572431; Papyan et al. 2020 PMC7547234; Fang et al. 2021
   PMC8639364); P3 (Song et al. 2025 PMC12026048 for KL annealing;
   Kenward & Roger 1997 PMID 9333350 abstract-only, flagged, the natural
   reference for the $\kappa_S$ repair if the user uploads it).
3. **Reproduce every number** in a torch-free `tools/pN_numbers.py` (numpy,
   scipy), run twice, outputs identical; tag the numbers `[RAN]` and name
   the block.
4. **Write** `PN_....md` with the project's mandatory structure (header +
   changelog row; abstract with "Covered" and "Deliberately excluded"; S1
   notation subset of E0 plus the new symbols; S1.1 conventions; S2 glossary;
   S3 body; S4 summary; S5 open points; S6 references with the searches
   table; the pre-send check footer R1-R8). Equations numbered (PN.m); plan
   equations cited as "plan eq. (n)"; the per-parameter fields of the build
   plan's S2 for every axis and every fixed knob; a differences table for the
   DSN-inherited axes (D-036); findings as a section "Findings this document
   owns".
5. **Append the new symbols to E0** (convention 14): a group row
   `| **<name> (PN)** | | | | |` then one row per symbol with type, units,
   first use `PN`, and a "not X" clause for every near-clash; bump E0's
   version and Date line; add a changelog row.
6. **Run the gates** from `hpc/joint/docs/tools`:
   `python3 check_notation.py --master ../E0_READERS_GUIDE_NOTATION.md --self --docs ../P0_PARAMETERS_OVERVIEW.md ../P1_ENCODER_AXES.md ../P2_DSN_LOSS_AXES.md ../P3_REPLICATE_AXES.md ../PN_....md`;
   add `PN_....md` to the `docs` tuple of `smoke_test_notation.py` (T4.n);
   `python3 smoke_test_notation.py --docs-dir ..` twice;
   `python3 smoke_test_inventory.py --hpc-dir ../../..` twice;
   `python3 p0_tables.py --hpc-dir ../../.. --inventory inventory.json --check-doc ../P0_PARAMETERS_OVERVIEW.md`;
   `python3 inventory_joint_knobs.py --hpc-dir ../../.. --check-index ../00_INDEX.md`;
   `py_compile` every `.py` touched; the `args.*` resolution check; byte
   check: every `.md` and `.py` pure ASCII, LF only, ends with LF.
7. **Update `00_INDEX.md`:** changelog row (version +0.1), the status rows
   (the document to `drafted | pending push`, the next one to `next turn`,
   E0's version and symbol count, the tools row), S6 (new findings, extended
   ones marked `[extended <date>]`), S8 (the new `[RAN]` numbers with the
   script block); run `--check-index` again; write the KB copy
   (`project_write`, `local_path`) after `project_read` to check it did not
   change.
8. **Commit** on `docs/joint-docs` as author `Leonardo Della Mea
   <leonardodellamea.00@gmail.com>`, subject `hpc/joint/docs: ...`, body
   ending with the two attribution lines the session's system reminder
   gives (`Co-Authored-By: ...` and `Claude-Session: <this chat's URL>`).
   Send the document with `SendUserFile`. Do not push.
9. **Close the reply** with "Next on go on: ..." and the line naming the KB
   docs written that turn.

## 4. Sandbox set-up for the next chat

```
git clone https://github.com/Leonardodm00/Simulation-Based-Inference leonardodm00/simulation-based-inference
cd leonardodm00/simulation-based-inference && git log --oneline -8 | cat      # expect the seven hpc/joint/docs subjects above 745edce
git checkout -b docs/joint-docs origin/main
pip install numpy scipy                                                    # no torch, sbi or skopt in the sandbox; the tools are torch-free
cd hpc/joint/docs/tools && python3 smoke_test_notation.py --docs-dir .. | tail -1 && python3 smoke_test_inventory.py --hpc-dir ../../.. | tail -1
```

Expected: `35/35 passed`, `38/38 passed`. If `origin/main` has moved past
the series, check `git log --stat` of the new commits for anything under
`hpc/joint/` or the DSN files the inventory reads (`dsn/config.py`,
`dsn/condition_space.py`, `dsn/backbone.py`, `npe_tune_search.py`): a change
there means the freeze (`834eb41`, index S4) must be revisited and
`--check-index` re-run.

## 5. P4 -- what it is, as announced at the end of P3

**`P4_FLOW_AXES.md`**, owner of the `flow` block (`hidden_features`,
`num_transforms`) and of the flow knobs held fixed or set by code:
`num_bins` (F-a: runner 8, `build_joint_model` 10, the space declares 10
fixed, `build_argv` passes no `--num-bins`), `z_score_theta =
'transform_to_unconstrained'` (requires the prior as `x_dist`;
`joint_model.py:7-22`), `z_score_x = 'none'` (plan D6), the `hidden_features`
width rule of `default_joint_space` (`clamp([2,8] * max(p, E), [64, 256])`;
F-j: `[52, 256]` at the DUP15HD shapes, `[32, 256]` on the bench),
`num_transforms` `[4, 12]`, the library-versus-runner defaults (F-m: 64/5/10
against 48/3/8), the runner defaults outside the space (F-r: 48 below 52,
3 below 4). Sources to read in full: `stage2/joint_model.py`, the `sbi`
0.27.0 wheel (`neural_nets/net_builders/flow.py` -- `build_zuko_nsf`,
`build_zuko_flow`, `_prepare_x_transforms`, `_get_base_and_transforms`;
`neural_nets/estimators/zuko_flow.py`), the `zuko` wheel at the cluster's
version (`flows/spline.py`, `NSF`, the `bins`/`bound`/`tails` arguments),
`npe_tune_search.py` (`default_space`, the width rule's origin), the plan's
S2.1/S2.2 and D6/D7, `JOINT_DSN_NPE_USAGE_v1.md` S3/S5, and `[KB]` the
project PDFs on flows (Papamakarios et al., "Normalizing Flows for
Probabilistic Modeling and Inference"; Durkan et al. is not in the KB -- a
PubMed/bioRxiv search for neural spline flows is due). Established in P3 and
to be carried, not re-derived: the draws of `rsample_posterior` and
`sample_posterior` come from the same zuko distribution object and are in box
coordinates (the unconstrained transform is the first element of the zuko
`Flow`'s transform chain, `flow.py:1296-1312`, `zuko_flow.py:141-156`).
E0 symbols P4 will probably need (check `--list` first): the flow's base
variable and transform (E0 convention 14 reserves them for E2 -- P4 may
declare them if it needs them first), the spline bin count and bound, the
coupling/autoregressive conditioner's hidden width.

Then, in order: P5 (optimiser and schedule: `lr`, `one_minus_beta1`,
`weight_decay`, `batch_size_npe`, `epochs`, `steps_per_epoch`, `grad_clip`,
`patience`, the split, `one_minus_beta2` fixed; F-c, F-d, F-g, F-i, F-l, F-m,
F-q, F-r), P6 (search driver: campaigns, anchors, canonicalisation, ledger,
proposals, finalists, controls; F-a, F-b, F-f, F-j, F-o, F-r, F-s, F-u, F-w;
`S-A25` has no arm), P7 (upstream and jobs; F-n, F-t), then E1..E9, Stage 8
(D-038 patches), Stage 6 reviews by a fresh agent, Stage 7 (re-fetch,
rebase, one series, the final delivery).

## 6. Findings on record (index S6) and what is still unwritten in the log

F-a .. F-x from the earlier turns; P3 added **F-y** (the `t_floor` clamp
zeroes the gradient of every pair with $\hat T_{gg'} \le 10^{-8}$, i.e.
every pair with $\hat T^{\rm raw}_{gg'} \le d_\theta/S_{\rm mc}$ -- the collapse
signature; 36-48% of collapsed pairs at the searched $S_{\rm mc}$ `[RAN]`),
**F-z** (`p_eff_min` turns an undefined target into a pull toward
agreement), **F-aa** (arm `A5` with no same-donor pair trains as `A1`
silently), **F-ab** (`enumerate_donor_pairs` pairs rows by `donor`, rows are
windows, 47% of the bench's pairs are two windows of one well `[RAN]`).
**The decisions log carries none of the P2/P3 open calls**: the attempt to
add an Open-calls row for F-t/F-u/F-v/F-w/F-f on 2026-10-01 was denied by
the environment's classifier and, per that denial, was not retried; the
P3 findings were likewise left to the index. If the user wants them in the
log's Open calls, add one plain row naming the finding ids and the index
section -- nothing more elaborate -- and if that is denied again, say so and
stop.

## 7. Traps the first four documents hit (so the next chat does not)

- **Notation.** Every symbol in a document must be declared in E0; the
  checker scans inline math per line, so a `$...$` span must not break across
  a line; `\dfrac`, `\tfrac`, `\\[1ex]`, `\texttt` inside math and `\neq`
  inside a script are rejected (use `\frac`, `\\`, `\text{}`, `\ne`); a
  letter already declared with another meaning is a clash even in another
  font family's eyes -- check `python3 check_notation.py --master
  ../E0_READERS_GUIDE_NOTATION.md --list` before choosing: $\mathcal{R}$ is
  the bench arm R (P3's pair set became $\mathcal{U}$), $L_0$ is the prior
  floor (not a Cholesky factor), $u$ is the Cholesky auxiliary vector, $t$ is
  the training-progress fraction, $I$ is mutual information ($I_{d_\theta}$
  declared separately), $\kappa_k$ is the contraction (not $\kappa_S$).
- **Levels and statuses.** Hat = computed level; $\bar C$ computed and
  $\bar C_{\rm true}$ analytic (the metric document's convention, kept);
  every number carries a tag; every knob a status (configured / computed /
  analytic / derivation-only), per surface (space, pin, tuner flag, runner
  default, PBS variable, library default -- they disagree: F-m, F-p).
- **Repository versus knowledge base.** The plan in the repository is
  v0.6.5; the $\kappa_S$ correction and the floor's true rationale live in
  the KB only (F-e); quote the repository text and cite the KB correction.
- **Line references** are cited as `file:line` at the freeze; re-verify
  each with `grep -n` before writing (two were wrong on the first pass of
  P3 and caught in the second).
- **The KB copy of the index** is written once per turn, after
  `project_read` shows it unchanged; the log is read from the project
  before any write and merged, never overwritten.
- **The stop hook** ("Branch has N unpushed commits ... please push") fires
  after every reply; it is not the user; decline, cite D-052/D-054.

## 8. Opening prompt for the next chat (copy as is)

> Continue the joint DSN+NPE documentation set from
> `claude/HANDOFF_2026-10-02_joint_docs_P3_done.md` (read it first, then
> `claude/joint_docs/00_INDEX.md` v1.6 and the decisions log). The series up
> to P3 is on `origin/main` (seven `hpc/joint/docs:` commits above `745edce`)
> -- verify with `git fetch` before anything else. Next document: P4
> (flow axes), one document per "go on", same procedure and gates, no push
> until I say so (D-052/D-054).

---

Docs this handoff accompanies: the series tarball `joint_docs_series_2026-10-02.tar.gz` (sent 2026-10-02), `claude/joint_docs/00_INDEX.md` v1.6.
