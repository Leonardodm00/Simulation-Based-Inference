# LindDUP: the DUP15HD recordings -- where they are and the well map

| Date | Change |
|---|---|
| 2026-11-17 | v1. Written from the user's `ls -R` of `Deep_bio/` (pasted 10:57 Europe/Rome) `[CLUSTER, the user's paste]`. The move of sec. 2 is handed to the user as commands; NOT yet run when this was written. The donor column is empty: the folder tree does not name a donor (sec. 4). |

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
| C-B3-A1 | control | Batch3 | - | C-B3 | A1 | 2292 |  | not given |
| C-B3-A2 | control | Batch3 | - | C-B3 | A2 | 1760 |  | not given |
| C-B3-A3 | control | Batch3 | - | C-B3 | A3 | 2109 |  | not given |
| C-B3-B1 | control | Batch3 | - | C-B3 | B1 | 2294 |  | not given |
| C-B3-B2 | control | Batch3 | - | C-B3 | B2 | 2299 |  | not given |
| C-B4-S1-A1 | control | Batch4 | SubBatch1 | C-B4-S1 | A1 | 2272 |  | not given |
| C-B4-S1-A2 | control | Batch4 | SubBatch1 | C-B4-S1 | A2 | 2296 |  | not given |
| C-B4-S1-A3 | control | Batch4 | SubBatch1 | C-B4-S1 | A3 | 2299 |  | not given |
| C-B4-S1-B1 | control | Batch4 | SubBatch1 | C-B4-S1 | B1 | 2291 |  | not given |
| C-B4-S1-B2 | control | Batch4 | SubBatch1 | C-B4-S1 | B2 | 2296 |  | not given |
| C-B4-S1-B3 | control | Batch4 | SubBatch1 | C-B4-S1 | B3 | 2275 |  | not given |
| C-B4-S2-A1 | control | Batch4 | SubBatch2 | C-B4-S2 | A1 | 2292 |  | not given |
| C-B4-S2-A2 | control | Batch4 | SubBatch2 | C-B4-S2 | A2 | 2275 |  | not given |
| C-B4-S2-A3 | control | Batch4 | SubBatch2 | C-B4-S2 | A3 | 1724 |  | not given |
| C-B4-S2-B1 | control | Batch4 | SubBatch2 | C-B4-S2 | B1 | 2285 |  | not given |
| C-B4-S2-B2 | control | Batch4 | SubBatch2 | C-B4-S2 | B2 | 2292 |  | not given |
| C-B4-S2-B3 | control | Batch4 | SubBatch2 | C-B4-S2 | B3 | 2050 |  | not given |
| P-B3-S1-A1 | DUP | Batch3 | SubBatch1 | P-B3-S1 | A1 | 2287 |  | not given |
| P-B3-S1-A2 | DUP | Batch3 | SubBatch1 | P-B3-S1 | A2 | 2281 |  | not given |
| P-B3-S1-A3 | DUP | Batch3 | SubBatch1 | P-B3-S1 | A3 | 2285 |  | not given |
| P-B3-S1-B1 | DUP | Batch3 | SubBatch1 | P-B3-S1 | B1 | 2286 |  | not given |
| P-B3-S1-B2 | DUP | Batch3 | SubBatch1 | P-B3-S1 | B2 | 2297 |  | not given |
| P-B3-S1-B3 | DUP | Batch3 | SubBatch1 | P-B3-S1 | B3 | 2273 |  | not given |
| P-B3-S2-A1 | DUP | Batch3 | SubBatch2 | P-B3-S2 | A1 | 2283 |  | not given |
| P-B3-S2-A2 | DUP | Batch3 | SubBatch2 | P-B3-S2 | A2 | 2263 |  | not given |
| P-B3-S2-A3 | DUP | Batch3 | SubBatch2 | P-B3-S2 | A3 | 2286 |  | not given |
| P-B3-S2-B1 | DUP | Batch3 | SubBatch2 | P-B3-S2 | B1 | 2255 |  | not given |
| P-B3-S2-B2 | DUP | Batch3 | SubBatch2 | P-B3-S2 | B2 | 2273 |  | not given |
| P-B3-S2-B3 | DUP | Batch3 | SubBatch2 | P-B3-S2 | B3 | 2274 |  | not given |
| P-B4-A1 | DUP | Batch4 | - | P-B4 | A1 | 2286 | 33 names differ from P-B3-S1-A1 | not given |
| P-B4-A2 | DUP | Batch4 | - | P-B4 | A2 | 2281 | names identical to P-B3-S1-A2 | not given |
| P-B4-A3 | DUP | Batch4 | - | P-B4 | A3 | 2285 | names identical to P-B3-S1-A3 | not given |
| P-B4-B1 | DUP | Batch4 | - | P-B4 | B1 | 2286 | names identical to P-B3-S1-B1 | not given |
| P-B4-B2 | DUP | Batch4 | - | P-B4 | B2 | 2297 | names identical to P-B3-S1-B2 | not given |
| P-B4-B3 | DUP | Batch4 | - | P-B4 | B3 | 2273 | names identical to P-B3-S1-B3 | not given |

## 4. What is not in the tree: the donor

The tree gives condition, batch, sub-batch and well position. It does not name a donor, a line or clone, a plating date or the age at recording. The pipeline's real bank today takes the well as the donor, and arm A5 refuses such a bank (log, D-118 Status). A5 (D-140: a pair is two wells of one donor) therefore needs the `donor` column filled. Three readings the tree allows, none confirmed:

| reading | donors | same-donor pairs it gives |
|---|---|---|
| donor = plate folder | 6 | wells of one plate |
| donor = condition x batch | 4 | wells of one batch, across its sub-batches |
| donor = condition | 2 | every pair of wells of one condition |

Open, the user's call.

## 5. Open: `DATA_P/Batch4` against `DATA_P/Batch3/SubBatch1`

For wells A2, A3, B1, B2, B3 the two folders hold exactly the same set of electrode file names (0 names differ); for A1, 33 names differ. Among all other pairs of the 35 wells the smallest difference is 8 names and the median 41 `[computed here from the listing]`. So `DATA_P/Batch4` is probably a copy of `DATA_P/Batch3/SubBatch1` in five wells `[reasoning, Likely; names only, no content read]`. If so, the DUP class has 13 distinct recordings, not 18, and five would count twice in the class statistics and in any replicate pair.

Check, read-only, before or after the move (per well: files compared, files with identical content):

```bash
cd "/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/DATA_P" && for w in A1 A2 A3 B1 B2 B3; do n=0; t=0; for f in Batch4/ptrain_$w/ptrain_*.mat; do t=$((t+1)); cmp -s "$f" "Batch3/SubBatch1/ptrain_$w/${f##*/}" && n=$((n+1)); done; echo "$w files=$t identical=$n"; done
```

`identical` equal to `files` means a copy; `identical=0` means different recordings that happen to share their file names.
