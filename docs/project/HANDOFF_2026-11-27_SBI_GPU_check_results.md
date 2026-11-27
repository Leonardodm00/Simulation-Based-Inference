# HANDOFF 2026-11-27 -- SBI: results of the GPU check on davinci (joint DSN + NPE training)

Answer to `HANDOFF_2026-11-26_SBI_GPU_check_davinci.md` (v2, written in the HPC-access chat;
its M0-M8 and its sections 6, 7 and 9 are used as defined there). Written in the SBI chat
that ran the check.

| Date | Change |
|---|---|
| 2026-11-27 | v1.1. New section "Training-set files and I/O", for the portal's I/O field: the shard format and how a run reads it, from the code (SBI `sci/device` 8607e5c and Sbi-extractor `feat/real-arm-parity` 0ff531d, whose bank tools call SBI's `write_shard`); the files on davinci not yet inspected. |
| 2026-11-27 | v1. M0, M1, M3 and M4 (largest configuration), M5 (partial) and M6 measured; M2 and the median configuration of M3 not yet run. |

**Tags.** `[CLUSTER]` read from the user's pastes of davinci output (dates below).
`[CODE]` read in the repository, `sci/device` 8607e5c. `[calc]` arithmetic run by script in
the SBI chat from the `[CLUSTER]` numbers. `[extrapolated]` a `[calc]` beyond the measured
range. `[inference]` Claude's reading, not measured. `[handoff]` taken from the v2 handoff.

## What ran

| Item | Value | Basis |
|---|---|---|
| Code | `run_joint_arms.py --device cuda` (SPEC block 39, decision D-169), branch `sci/device` | `[CODE]` |
| Environment | `sbi_gpu`: torch 2.10.0+cu128, every other package at `sbi_env`'s version (sbi 0.27.0); `sbi_env` (torch 2.10.0, CPU build) untouched | `[CLUSTER 2026-11-26]` |
| Job shape | queue `gpu`, `select=1:ncpus=8:ngpus=1:mem=120gb`, `--threads 8` | `[CODE]` the submission command |
| Measurement | the runner's probe: 1 warm-up step, then the timed steps (20, as the CPU budget gate used; one run with 200), then 5,000 selection rows scored; the NPE term alone unless `--probe-full-loss` | `[CODE]` `run_probe`; the LindDUP gate's settings `probe_steps 20`, `probe_val_rows 5000` |
| Configuration | the LindDUP search's costliest configuration (`A1/S-A1/jobs/gate_b<B>.argv`), the "largest" of M3; A2 and A5 from their own gate files | `[CODE]` |
| CPU reference | the budget gate's own probes, 48 threads, 2026-11-18, same configurations, same probe settings | `[CLUSTER 2026-11-26 17:34]` |
| Validity | every GPU job started on an idle GPU (no process, 0 % busy) and ended with no other process on it | `[CLUSTER 2026-11-27 14:53]` |

## Results (the table of the v2 handoff, section 9)

`S` below is per training step. `S, trial` includes the per-epoch scoring of the selection
split (next section), which the GPU speeds up only about 2x.

| Configuration | Batch | s/step, 48 cores | s/step, 1 GPU | S per step | S, trial | Peak GPU GiB (reserved) | GPU util % |
|---|---|---|---|---|---|---|---|
| largest, A1, NPE term | 256 | 3.04 | 0.336 | 9.0 | 4.7 | 13.7 (15.3) | 8 * |
| largest, A1, NPE term | 512 | 5.81 | 0.623 | 9.3 | 4.8 | 27.1 (28.7) | 16 * |
| largest, A1, NPE term, 200 steps | 512 | 5.81 (20-step CPU probe) | 0.804 | 7.2 | 4.2 | 27.1 (28.7) | 59 |
| largest, A1, NPE term | 1,024 | 10.83 | out of GPU memory (40 GB card) | -- | -- | > 38.6 at the failure | -- |
| largest, A2, full objective | 512 | not measured | 0.693 | -- | -- | 27.9 (30.6) | 18 * |
| largest, A5, full objective | 512 | not measured | 1.975 | -- | -- | 34.2 (36.4) | 32 * |

`[CLUSTER 2026-11-27]` GPU columns; `[CLUSTER 2026-11-26]` CPU column; `[calc]` both S columns.
`*` mean over a job of about one minute in which start-up and scoring dominate: not a
utilisation of training (section "Utilisation").

## Whole trial, validation included

| Quantity | CPU, 48 cores | 1 GPU | Ratio | Basis |
|---|---|---|---|---|
| seconds per training step, b512 | 5.81 | 0.623 (20 steps) / 0.804 (200 steps) | 9.3 / 7.2 | `[CLUSTER]`, `[calc]` |
| seconds per selection row scored | 0.0094-0.0095 | 0.0047-0.0049 | 1.9-2.0 | `[CLUSTER]`, `[calc]` |
| projected trial, b512 (20 epochs x 234 steps, 6 passes) | 10.2 h | 2.1 h (20 steps) / 2.4 h (200 steps) | 4.8 / 4.2 | `[calc]` |
| projected trial, b256 (20 x 468 steps) | 10.5 h | 2.2 h | 4.7 | `[calc]` |
| share of a projected GPU trial spent scoring the selection split | | 56-62 % | | `[calc]` |

Projection: `T = E * (K * t_step + n_sel * t_row)`, the probe's own formula `[CODE]`, with
`E` = 20 epochs, `K` = steps per epoch, `n_sel` about 49,900 selection rows. `n_sel` was
recovered from the printed projections (+-2.3 %); it matches a 10 % split of the bank
(398,714 training rows / 8 = 49,839) `[calc]`. The exact value is the field `n_sel` of each
`*_probe.json`. The trial projection leaves out start-up and checkpoint writes `[CODE]`.

## Memory fit

| Quantity | Value | Basis |
|---|---|---|
| davinci GPU | A100-SXM4-40GB, 39.49 GiB usable | `[CLUSTER]` the OOM message |
| A1 GPU memory against batch | 0.0521 GiB per window + 0.40 GiB | `[calc]` from b256 and b512 |
| A1 at batch 1,024 | about 53.8 GiB allocated: does not fit 40 GB (measured: out of memory) | `[extrapolated]`; `[CLUSTER]` |
| A2 / A5 at batch 1,024 | about 55.4 / 67.8 GiB, if their extra memory scales with the batch as A1's does | `[extrapolated]` |
| Leonardo Booster | A100 with 64 GB `[handoff]`: A1 and A2 at 1,024 would fit with a few GiB to spare; A5 at 1,024 would not | `[extrapolated]` |
| host RAM of a GPU run | 3.2-3.6 GB at b256-b512 (a CPU run: 15.5 / 29.0 / 55.8 GB at 256 / 512 / 1,024) | `[CLUSTER]` |

The CPU run's host memory per window (0.0527 GB) equals the GPU's (0.0521 GiB) `[calc]`:
the activations move to the card. A cap on the batch size for GPU runs is the user's
decision `[handoff]`, not taken here.

## M6: the A2 and A5 terms on the GPU

| Arm, b512 | s/step | Against A1 | Peak GPU GiB | Basis |
|---|---|---|---|---|
| A1, NPE term | 0.623 | 1 | 27.1 | `[CLUSTER]` |
| A2, NPE + DSN term | 0.693 | 1.11x (+0.070 s) | 27.9 (+0.8) | `[CLUSTER]`, `[calc]` |
| A5, NPE + replicate term | 1.975 | 3.17x (+1.352 s) | 34.2 (+7.1) | `[CLUSTER]`, `[calc]` |

Both run on the GPU. Their CPU cost is not on record: the budget gate's CPU probes time the
NPE term alone for every arm `[CODE]` (finding of 2026-11-26), so `S` for the full objectives
of A2 and A5 is not measured.

## Utilisation (M5) and the bottlenecks

- The 200-step job: the chosen GPU 59 % busy on average over 43 samples (5 s apart),
  start-up and scoring included `[CLUSTER]`, `[calc]`. The 20-step jobs (8-32 %) last about a
  minute and mostly measure start-up.
- Scoring the selection split: the scorer reads the rows from the bank's shard files as it
  goes, in batches of 512 with at most two shards in memory `[CODE]`
  `stage2/joint_train.py` `evaluate_npe_shards`. On the GPU it is 2x faster than on the CPU,
  against 9x for a training step, and takes 56-62 % of a projected GPU trial `[calc]`. That
  the reading of the shards, not the GPU, sets its pace is `[inference]`.
- A step is 29 % slower over 200 steps than over 20 (0.804 against 0.623 s): consistent with
  the training stream's buffer refills entering the longer window `[inference, not verified]`.
- The development work these point to (proposals, none built): score the selection split
  from memory or with read-ahead; overlap shard reading with GPU compute (a background
  prefetch, pinned memory); mixed precision to cut memory and time (M7, not measured); for
  batch 1,024 on A5, memory work or a cap. Gradient accumulation may not reproduce A5's
  replicate term if that term is estimated over the whole batch (to be checked in
  `joint_losses.py`).

## GPU-hours per arm (v2 handoff, section 7)

`H = N_pass * t_CPU_mean / S`, `N_pass` = 1,900, `t_CPU_mean` = 1.683 h `[handoff]`:

| S used | GPU-hours per arm | Arms in 18,000 GPU-hours |
|---|---|---|
| 4.25 (trial, 200-step GPU timing; the conservative one) | 753 | 23.9 |
| 4.82 (trial, 20-step timing) | 664 | 27.1 |
| 7.2-9.3 (per step: if scoring were made as fast as training) | 443-343 | 40.7-52.5 |

`[calc]`. Limits: one configuration (the largest) measured; `t_CPU_mean` comes from real
trials, whose A2 and A5 runs include their own terms on the CPU, while the measured `S` is
for the NPE term; DCGP's 448,000 core-hours hold 2.92 arms `[handoff]`.

## The five lines of the v2 handoff, section 9

1. NVIDIA A100-SXM4-40GB (39.49 GiB usable), 4 per node, queue `gpu` (nodes dvgpu015,
   dvgpu031 and the probes' node). Cores per node not read. davinci's PBS sets no
   `CUDA_VISIBLE_DEVICES` and names no GPU in `exec_vnode`: the job script picks the idle GPU
   (a busy GPU 0 holding 37.9 GB of another job was seen on dvgpu015) `[CLUSTER]`.
2. Driver 580.82.07 (CUDA 13.0); torch 2.10.0+cu128 (CUDA 12.8) in `sbi_gpu`, sbi 0.27.0
   `[CLUSTER]`.
3. M2 not run yet on the real bank. On the device suite's toy bank, a GPU model and a CPU
   model with the same weights agree on the loss within 1.5e-5 (check V6) `[CLUSTER 2026-11-26]`.
4. A2 and A5 run on the GPU: per step 1.11x and 3.17x A1; A5 peaks at 34.2 GiB at batch 512.
5. Code: SPEC block 39 on `sci/device` -- the `--device` option (the model built on the
   device, batches moved in the forward paths, checkpoints in CPU tensors, `cuda` refused
   without a GPU): 4 files, 201 lines added, 28 removed (`run_joint_arms.py`,
   `joint_model.py`, `joint_train.py`, `joint_losses.py`); plus the check's tooling, 3 new
   files, 872 lines (`smoke_test_device.py`, `device_probe_argv.py`,
   `jobs/device_check.pbs`) `[CODE]`. The CPU path is bit for bit the old one (arms A0, A1,
   A2, A5, shuffled) `[sandbox, 2026-11-26]`.

## Against the decision rule (v2 handoff, section 6)

`[inference]`, for the user: S is well above 1 (4.2-4.8 per trial, 7-9 per step); M6 runs;
batches 256 and 512 fit even on 40 GB; batch 1,024 does not fit davinci's card and, by
extrapolation, fits Leonardo's 64 GB for A1 and A2 but not for A5. The first row of the
rule ("SBI training moves to Booster") holds once M2 agrees and the batch-size question is
settled (a cap for A5, or memory work).

## Not measured yet

| Item | What it needs |
|---|---|
| M2 on the real bank | one short run of the same configuration and seed on the CPU (queue `cpu`, 48 cores) and on the GPU; compare the training and selection losses |
| M3, the median configuration | a finished trial of median cost, probed on the CPU (queue `cpu`) and on the GPU |
| S for the full objectives of A2 and A5 | their probes with `--probe-full-loss` on the CPU (queue `cpu`) |
| M7, M8 | optional in the v2 handoff |

## Training-set files and I/O (the portal's I/O field)

| Item | Value | Basis |
|---|---|---|
| Shard file | one `.npz` per shard: a ZIP archive of NumPy arrays, deflate-compressed (`np.savez_compressed`) | `[CODE]` `hpc/joint/stage1/latent_bank.py`, `write_shard` |
| Beside each shard | a JSON sidecar `<shard>.json`: the bank's contract (schema `latent_bank/1`, parameter names and bounds, window length `W`, sampling rate) with its sha256 digest | `[CODE]` `build_sidecar`, `read_shard` |
| Arrays in a shard | `x` (rows x `W` windows, float), `theta` (rows x parameters, float), per-row labels (`cls`, `nu`, `realisation_id`, `donor`, `well`, `batch`, `subregion`, `window_idx`, `contaminated`; `rep_group` in a real bank) | `[CODE]` `ROW_ARRAYS`, `OPTIONAL_ROW_ARRAYS` |
| Written by | Sbi-extractor's bank tools, which import SBI's `latent_bank` and call its `write_shard` (`export_windows/bank_export.py` for the simulated bank, `bank_real.py` for the real one): one file per write, written to a temporary name then renamed | `[CODE]` `write_shard`; Sbi-extractor `feat/real-arm-parity` 0ff531d, which davinci's checkout is assumed to match (the command below checks the files themselves) |
| Read by | one Python process per training job, plain POSIX file reads through `np.load`: every shard's small arrays once at the start, then the windows streamed shard by shard, `buffer_shards` shards resident while training and at most two while scoring the selection split | `[CODE]` `ShardIndex`, `iter_batches`; `stage2/joint_batches.py` |
| Not used | HDF5, netCDF, zarr, MPI I/O, memory mapping | `[CODE]` a search of `hpc/joint` and `hpc/pipeline` for their calls |
| Other files of a run | the model checkpoint (PyTorch `.pt`), the record (JSON), per-row results (`.npz`) | `[CODE]` `run_joint_arms.py` |
| On davinci | not yet inspected; the command below prints the bank's file types, a sidecar, each array's dtype, shape and compression, and the shard count and total size | -- |

Read-only, run on a login node; it takes the bank's path from the gate's own command file:

```bash
bash -c 'A=$1; G=$(grep -A1 -x -- "--sim-shards" $A | tail -1); B=$(dirname "$G"); S=$(ls $G | head -1); echo "bank folder: $B"; echo "--- file types in it:"; ls "$B" | sed "s/.*\.//" | sort | uniq -c; echo "--- sidecar $(basename ${S%.npz}.json), first 600 bytes:"; head -c 600 "${S%.npz}.json"; echo; echo "--- arrays inside $(basename $S):"; python3 -c "import sys,zipfile,ast,glob,os; p=sys.argv[1]; z=zipfile.ZipFile(p); [print(\"%-16s %-6s %-14s %9.2f MB raw %9.2f MB on disk  %s\" % (i.filename, d[\"descr\"], str(d[\"shape\"]), i.file_size/1e6, i.compress_size/1e6, {0:\"stored\",8:\"deflate\"}.get(i.compress_type,i.compress_type))) for i in z.infolist() for h in [z.open(i).read(512)] for d in [ast.literal_eval(h[h.index(b\"{\"):h.index(b\"}\")+1].decode())]]; fs=sorted(glob.glob(sys.argv[2])); t=sum(map(os.path.getsize,fs)); print(\"%d shards, %.2f GB on disk, %.1f MB per shard on average\" % (len(fs), t/1e9, t/1e6/max(len(fs),1)))" "$S" "$G"' _ ~/tune_joint_linddup/A1/S-A1/jobs/gate_b512.argv
```

## On davinci

| Path | What | Basis |
|---|---|---|
| `~/SBI_device` | a clone of `sci/device` (run from it; `~/SBI` and the searches untouched) | `[CLUSTER]` |
| `~/gpu_check/` | `info.log`, `suite.log`, the `sbi_env` package list, `probes/` | `[CLUSTER]` |
| `~/gpu_check/probes/<name>.{argv,log,gpu_util.csv}`, `<name>/<arm>_seed0_probe.json` | each probe's command, log, utilisation samples and record | `[CLUSTER]` |
| `~/.conda/envs/sbi_gpu` | the GPU environment | `[CLUSTER]` |

The results summary, run in `~/gpu_check/probes`:

```bash
cd ~/gpu_check/probes && bash -c 'for f in *.log; do echo "== $f"; grep -E "gpu choice|gpu at the end|PROBE OK|PROBE GPU|Error|DEVICE_CHECK EXIT" $f; i=$(sed -n "s/^gpu choice *: index \([0-9]*\) .*/\1/p" $f); [ -n "$i" ] && [ -f ${f%.log}.gpu_util.csv ] && awk -F", *" -v i="$i" "NR>1 && \$2==i {s+=\$3; c++} END {if (c) printf \"GPU %s mean utilisation %.0f %% over %d samples\n\", i, s/c, c}" ${f%.log}.gpu_util.csv; done; python3 -c "import json,glob; [print(p, \"s/step\", round(r[\"step_seconds\"],4), \"| GPU peak GiB\", r[\"gpu_peak_allocated_gb\"], \"| host GB\", round(r[\"maxrss_gb\"],1), \"| full loss\", r[\"probe_full_loss\"]) for p in sorted(glob.glob(\"*/*_probe.json\")) for r in [json.load(open(p))]]"'
```

A probe is not resubmitted over an existing `<name>.argv`: `device_probe_argv.py` refuses
it (exit 2). Remove the probe's files first, or give it a new name.

## Sources

- The user's davinci pastes: the CPU gate probes (2026-11-26 17:34), `info.log` (17:40),
  `suite.log` (18:38), the probes' summary (2026-11-27 14:53).
- `sci/device` 8607e5c: `hpc/joint/stage3/run_joint_arms.py` (`run_probe`),
  `hpc/joint/stage2/joint_train.py` (`evaluate_npe_shards`), SPEC block 39; decision log
  D-169, D-170.
- `HANDOFF_2026-11-26_SBI_GPU_check_davinci.md` v2 (the HPC-access chat; attached by the
  user, not in this repository).
