# LindDUP: the DUP15HD recordings -- where they are and the well map

| Date | Change |
|---|---|
| 2026-11-17 | v1. Written from the user's `ls -R` of `Deep_bio/` (pasted 10:57 Europe/Rome) `[CLUSTER, the user's paste]`. The move of sec. 2 is handed to the user as commands; NOT yet run when this was written. The donor column is empty: the folder tree does not name a donor (sec. 4). |
| 2026-11-17 | v2. The user's message of 11:14: the donor is the condition (D-144; sec. 4, the donor column filled: `C`, `P`); the `cmp` check of sec. 5 run on davinci `[CLUSTER, the user's paste]`: `DATA_P/Batch4` wells A2, A3, B1, B2, B3 are byte-for-byte copies of `DATA_P/Batch3/SubBatch1`'s, A1 shares no file with it. The move of sec. 2 not reported run. |
| 2026-11-17 | v3. The user's message of 11:51: the move of sec. 2 ran ("This worked"; output not pasted), so the recordings are at `LindaColloDUP/recordings/`; the five copies are left out of the real data (D-145): 30 wells, 17 control, 13 DUP. |
| 2026-11-17 | v4. The user's messages of 12:11 and 12:17: an A5 replicate pair is two subregions of one batch, the batch being `Batch3` / `Batch4` within a condition (D-146; new sec. 4b; the CSV gains `pair_group` and `in_real_data`). |
| 2026-11-17 | v5. The user's message of 12:24: the check before A5's search is dropped, the three arms run side by side; an A5 pair is drawn at random within its group (D-147). Sec. 4b: the check marked corrected; the draw and the share of within-well pairs it gives added (computed, formula given). |

Machine-readable copy of sec. 3: `docs/project/linddup_recordings_map.csv` (same rows, same columns, plus the paths).

## 1. What the listing shows

Source: `ls -R` in `/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/`. Only names were read: no file was opened, no size or date is known.

- `DATA_C` (controls) and `DATA_P` (DUP patients) `[user, 2026-11-17]`: 35 well folders `ptrain_<A1..B3>`, 17 control and 18 DUP, in 6 folders of at most 6 wells (below called a plate folder; that a folder is one 6-well plate is `[reasoning, Likely]`, from the well names A1-A3, B1-B3).
- A well folder holds one `ptrain_<k>.mat` per electrode, k within 2..2304, 78,452 files in all; 15 well folders also hold an empty `MATLAB_parallel/`.
- The 35 wells match `extracted_v2/` one to one (the cohort of record, manifest `c925c5106137fe82`, D-104).
- Also in `Deep_bio/`, not touched: `extracted_v2/`, `Main_RETIRED_20260914/`, `Main.zip`, `dsn_mea_lane{0..3}.out`.

## 2. The move (D-142)

From `Deep_bio/DATA_C`, `Deep_bio/DATA_P` to `/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP/recordings/DATA_C`, `.../recordings/DATA_P`. The subfolder name `recordings/` is `[Claude's default]`. Two symlinks stay at the old place, so the old paths keep resolving (the retired DSN config `hpc/dsn/hpc/Config/config_mea_joint_full.davinci.json`, the DUP15HD cohort block S1 would read). `extracted_v2/` stays where it is: `linddup.json` `real.extract_root` points at it. The new folder sits inside the ANN repository's working tree, so it is added to that clone's `.git/info/exclude`.

Each line is one command, tested on a copy of the layout `[SANDBOX 2026-11-17]`; run on a login node, in order.

Check, read-only (expected: the three folders listed, the same first number on both `stat` lines, `78452`, and no `recordings` line):

```bash
S="/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio"; D="/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP"; ls -d "$S/DATA_C" "$S/DATA_P" "$D"; stat -c '%d %n' "$S" "$D"; find "$S/DATA_C" "$S/DATA_P" -name 'ptrain_*.mat' | wc -l; ls -d "$D/recordings" 2>/dev/null
```

Move (prints `MOVED`; refuses if `recordings/` already exists):

```bash
S="/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio"; D="/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP"; mkdir "$D/recordings" && mv "$S/DATA_C" "$S/DATA_P" "$D/recordings/" && ln -s "$D/recordings/DATA_C" "$S/DATA_C" && ln -s "$D/recordings/DATA_P" "$S/DATA_P" && echo MOVED
```

Keep git from seeing the files (prints `EXCLUDED`):

```bash
D="/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP"; echo 'hpc/Phenomenological_finalv1/LindaColloDUP/recordings/' >> "$(git -C "$D" rev-parse --absolute-git-dir)/info/exclude" && echo EXCLUDED
```

Verify (expected: `78452` twice, two `->` lines, then `0`):

```bash
S="/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio"; D="/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP"; find "$D/recordings" -name 'ptrain_*.mat' | wc -l; find -L "$S/DATA_C" "$S/DATA_P" -name 'ptrain_*.mat' | wc -l; ls -l "$S" | grep DATA_; git -C "$D" status --short recordings | wc -l
```

If the two `stat` numbers differ, the folders are on different filesystems and `mv` copies the files instead of renaming; it still works, more slowly.

## 3. The well map

`n files` = electrode files `ptrain_<k>.mat` in the well folder. Paths: raw `recordings/<DATA_x>/<batch>[/<sub-batch>]/ptrain_<well>`; extracted `extracted_v2/<class>/<root_name>/ptrain_<well>` (both in the CSV).

| well_id | condition | batch | sub-batch | plate folder | well | n files | note | donor |
|---|---|---|---|---|---|---|---|---|
| C-B3-A1 | control | Batch3 | - | C-B3 | A1 | 2292 |  | C |
| C-B3-A2 | control | Batch3 | - | C-B3 | A2 | 1760 |  | C |
| C-B3-A3 | control | Batch3 | - | C-B3 | A3 | 2109 |  | C |
| C-B3-B1 | control | Batch3 | - | C-B3 | B1 | 2294 |  | C |
| C-B3-B2 | control | Batch3 | - | C-B3 | B2 | 2299 |  | C |
| C-B4-S1-A1 | control | Batch4 | SubBatch1 | C-B4-S1 | A1 | 2272 |  | C |
| C-B4-S1-A2 | control | Batch4 | SubBatch1 | C-B4-S1 | A2 | 2296 |  | C |
| C-B4-S1-A3 | control | Batch4 | SubBatch1 | C-B4-S1 | A3 | 2299 |  | C |
| C-B4-S1-B1 | control | Batch4 | SubBatch1 | C-B4-S1 | B1 | 2291 |  | C |
| C-B4-S1-B2 | control | Batch4 | SubBatch1 | C-B4-S1 | B2 | 2296 |  | C |
| C-B4-S1-B3 | control | Batch4 | SubBatch1 | C-B4-S1 | B3 | 2275 |  | C |
| C-B4-S2-A1 | control | Batch4 | SubBatch2 | C-B4-S2 | A1 | 2292 |  | C |
| C-B4-S2-A2 | control | Batch4 | SubBatch2 | C-B4-S2 | A2 | 2275 |  | C |
| C-B4-S2-A3 | control | Batch4 | SubBatch2 | C-B4-S2 | A3 | 1724 |  | C |
| C-B4-S2-B1 | control | Batch4 | SubBatch2 | C-B4-S2 | B1 | 2285 |  | C |
| C-B4-S2-B2 | control | Batch4 | SubBatch2 | C-B4-S2 | B2 | 2292 |  | C |
| C-B4-S2-B3 | control | Batch4 | SubBatch2 | C-B4-S2 | B3 | 2050 |  | C |
| P-B3-S1-A1 | DUP | Batch3 | SubBatch1 | P-B3-S1 | A1 | 2287 |  | P |
| P-B3-S1-A2 | DUP | Batch3 | SubBatch1 | P-B3-S1 | A2 | 2281 |  | P |
| P-B3-S1-A3 | DUP | Batch3 | SubBatch1 | P-B3-S1 | A3 | 2285 |  | P |
| P-B3-S1-B1 | DUP | Batch3 | SubBatch1 | P-B3-S1 | B1 | 2286 |  | P |
| P-B3-S1-B2 | DUP | Batch3 | SubBatch1 | P-B3-S1 | B2 | 2297 |  | P |
| P-B3-S1-B3 | DUP | Batch3 | SubBatch1 | P-B3-S1 | B3 | 2273 |  | P |
| P-B3-S2-A1 | DUP | Batch3 | SubBatch2 | P-B3-S2 | A1 | 2283 |  | P |
| P-B3-S2-A2 | DUP | Batch3 | SubBatch2 | P-B3-S2 | A2 | 2263 |  | P |
| P-B3-S2-A3 | DUP | Batch3 | SubBatch2 | P-B3-S2 | A3 | 2286 |  | P |
| P-B3-S2-B1 | DUP | Batch3 | SubBatch2 | P-B3-S2 | B1 | 2255 |  | P |
| P-B3-S2-B2 | DUP | Batch3 | SubBatch2 | P-B3-S2 | B2 | 2273 |  | P |
| P-B3-S2-B3 | DUP | Batch3 | SubBatch2 | P-B3-S2 | B3 | 2274 |  | P |
| P-B4-A1 | DUP | Batch4 | - | P-B4 | A1 | 2286 | no file identical to P-B3-S1-A1 | P |
| P-B4-A2 | DUP | Batch4 | - | P-B4 | A2 | 2281 | copy of (all files identical) P-B3-S1-A2 | P |
| P-B4-A3 | DUP | Batch4 | - | P-B4 | A3 | 2285 | copy of (all files identical) P-B3-S1-A3 | P |
| P-B4-B1 | DUP | Batch4 | - | P-B4 | B1 | 2286 | copy of (all files identical) P-B3-S1-B1 | P |
| P-B4-B2 | DUP | Batch4 | - | P-B4 | B2 | 2297 | copy of (all files identical) P-B3-S1-B2 | P |
| P-B4-B3 | DUP | Batch4 | - | P-B4 | B3 | 2273 | copy of (all files identical) P-B3-S1-B3 | P |

## 4. The donor (D-144)

The tree gives condition, batch, sub-batch and well position; it names no donor, line, plating date or age at recording. The user, 2026-11-17 11:14: "We assume as donor the conditions which is then subdivided in batches and wells which should differ by the biological variability" (spelling corrected). So there are two donors, `C` (17 wells) and `P` (18 wells), and batches, sub-batches and wells are replicates within a donor. An A5 pair (D-140) is any two wells of one condition.

Consequences `[reasoning]`, for the LindDUP pipeline chat:

- A5's pairs are then exactly the same-class pairs of A2's labels: the replicate term pulls together what the DSN term's positives pull together. The two arms differ on the real windows only in A2's push between classes.
- Any statistic that resamples real donors (a donor-cluster bootstrap) has 2 clusters and cannot give an interval; on the real arm the resampling unit has to be the well or the plate folder.
- Wells of one plate folder are likely closer than wells of different batches; whether pairs are drawn across batches or within is not decided.

## 4b. The A5 pair group (D-146)

An A5 replicate pair is two subregion windows of the same batch, the batch being `Batch3` or `Batch4` within a condition `[user, 2026-11-17 12:11, 12:17]`: pairs within a well, across wells of a plate folder, and across the sub-batches of a batch. This narrows D-144's "any two wells of one condition". With the five copies out (D-145):

| pair group | wells | plate folders |
|---|---|---|
| C-Batch3 | 5 | C-B3 |
| C-Batch4 | 12 | C-B4-S1, C-B4-S2 |
| P-Batch3 | 12 | P-B3-S1, P-B3-S2 |
| P-Batch4 | 1 | P-B4 (well A1 only: its pairs are within the well) |

The level is to be a configuration value (well, plate folder, batch), set to batch. Before A5's search, on the A1 fit (no replicate term): compare the summaries' distance over pairs of subregions of one well with that over pairs from different wells of one batch; if the second is clearly larger, wells of a batch are not exchangeable and the level goes back to the user `[Claude's proposal, accepted: "Go on then"]`. [corrected 2026-11-17 12:24: the check is dropped -- the user: "drop this just run all the arms" (D-147). No comparison gates A5; the three arms run side by side.]

**The draw (D-147).** An A5 pair is two distinct windows drawn uniformly at random among all the real windows of one group: any well, any subregion, any 180 s. Every extracted well holds 9 subregions (`trace_subregion_00..08`, all 35 folders `[CLUSTER, the listing]`); assuming 6 non-overlapping windows of 180 s per subregion (1200 s; the bank's stride not read), a well gives n_w = 54 windows and a group of w wells W = 54 w. The share of pairs inside one well is (n_w - 1) / (W - 1); inside one well and the same 180 s (simultaneous subregions), (9 - 1) / (W - 1):

| group | wells w | windows W | pairs inside one well | of which in the same 180 s |
|---|---|---|---|---|
| C-Batch3 | 5 | 270 | 0.197 | 0.030 |
| C-Batch4 | 12 | 648 | 0.082 | 0.012 |
| P-Batch3 | 12 | 648 | 0.082 | 0.012 |
| P-Batch4 | 1 | 54 | 1.000 | 0.151 |

`[verified by running it, 2026-11-17: exact count against 200,000 Monte Carlo draws, seed 0, agreement within 0.005; the script pair_mix.py was delivered to the user, not committed]`. So in the two large groups about 92 % of A5's pairs join different wells. Weighting the draw towards within-well pairs was not asked.

## 5. `DATA_P/Batch4` against `DATA_P/Batch3/SubBatch1`: five copies (run 2026-11-17)

Output of the `cmp` loop on davinci `[CLUSTER, the user's paste of 11:14]`:

| well | files in `Batch4` | identical in content to `Batch3/SubBatch1` |
|---|---|---|
| A1 | 2286 | 0 |
| A2 | 2281 | 2281 |
| A3 | 2285 | 2285 |
| B1 | 2286 | 2286 |
| B2 | 2297 | 2297 |
| B3 | 2273 | 2273 |

So P-B4-A2, -A3, -B1, -B2, -B3 are the same recordings as P-B3-S1-A2, -A3, -B1, -B2, -B3; P-B4-A1 is a different recording from P-B3-S1-A1. The DUP class holds 13 distinct recordings (18 - 5), the cohort 30 (35 - 5). `extracted_v2` and its manifest (`c925c5106137fe82`, 35 wells) hold the five twice. Decided 2026-11-17 11:51 (D-145): the five copies are left out of the real data. Not decided: where the real `Batch4` wells A2-B3 are, if they exist.

The command, for the record:

```bash
cd "/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/DATA_P" && for w in A1 A2 A3 B1 B2 B3; do n=0; t=0; for f in Batch4/ptrain_$w/ptrain_*.mat; do t=$((t+1)); cmp -s "$f" "Batch3/SubBatch1/ptrain_$w/${f##*/}" && n=$((n+1)); done; echo "$w files=$t identical=$n"; done
```
