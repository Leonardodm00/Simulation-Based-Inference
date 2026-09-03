# NPE hyperparameter tuning -- usage

Companion to `NPE_TUNING_PROTOCOL_v1.md`, which explains *why*. This file is
*how*. Every command prints a shape report first; if `p`, `E`, `n` or the
split hash is not what you expected, nothing after it matters.

## Files

| file | role |
|---|---|
| `npe_tune.py` | the driver: one subcommand per campaign stage |
| `npe_tune_data.py` | bank loading, topology grouping, the frozen split |
| `npe_tune_score.py` | the prior floor, held-out NLL, information gain, Jensen check. Pure numpy |
| `npe_tune_ledger.py` | per-trial result files assembled into a ledger |
| `npe_tune_search.py` | the search space and the GP loop; warm start, batching, escalation |
| `npe_tune_gates.py` | G1 informativeness, G2 marginal calibration, G3 joint calibration |
| `npe_tune_train.py` | independent member training, convergence capture, ensemble extension |
| `smoke_test_tune.py` | 13 tests; the fast tier needs no torch or sbi |
| `jobs/npe_tune.pbs` | one job = one evaluation |
| `jobs/launch_npe_tune.sh` | one job per pending spec; dry run by default |

The results directory IS the campaign state. The search warm-starts by
replaying it, so a crashed coordinator loses nothing and raising the budget
never repeats work.

## Before the first submission

Run these on the login node, in this order. The first one catches the bug
that would otherwise cost a whole submission cycle.

```bash
# 1. line endings: CRLF in a shebang'd file makes qsub fail with "bad
#    interpreter", and a non-ASCII scan does NOT catch it (\r is ASCII).
python3 -c "
import sys
bad=[(p,open(p,'rb').read().count(b'\r')) for p in sys.argv[1:]]
[print('CRLF',p,n) for p,n in bad if n]
print('OK: LF-only' if not any(n for _,n in bad) else 'FAIL')
" jobs/npe_tune.pbs jobs/launch_npe_tune.sh

# 2. encoding
python3 -c "
import sys
for p in sys.argv[1:]:
    b=[hex(c) for c in open(p,'rb').read() if c>127]
    print(p,'ASCII OK' if not b else ('NON-ASCII %s'%b[:5]))
" npe_tune*.py smoke_test_tune.py

# 3. syntax and imports
python3 -m py_compile npe_tune*.py smoke_test_tune.py
python3 -c "import npe_tune" && echo "imports resolve"

# 4. which environment is this, actually?  HPC_PATHS.md S7 flags an
#    unresolved sbi_export vs sbi_env conflict -- settle it here.
python3 check_env.py
python3 -c "import sbi,zuko,torch,skopt;print(sbi.__version__,zuko.__version__,torch.__version__,skopt.__version__)"

# 5. the tests
python3 smoke_test_tune.py --fast      # no torch needed, ~5 s
python3 smoke_test_tune.py             # adds S11/S12/S13, needs sbi
```

`scikit-optimize` is a NEW dependency, needed only by `propose`. Install it
deliberately and pinned: `pip install scikit-optimize==0.10.2`. Verified
working against numpy 2.4.x and scikit-learn 1.8.x.

## The campaign

```bash
OUT=run_r2_01

# Stage 0: freeze the split. Records the loader arguments every later
# command reuses, so the bank can never silently change under the campaign.
python3 npe_tune.py freeze-split \
    --sim '/path/to/SBI_export_r2/*.parquet' \
    --activity /path/to/activity_v3.npz --min-rate 0.1 \
    --out-dir $OUT --m-probe 3 --seeds 0 1 2 3 4 5 6 7 8 9

# Stage 2: the baseline and the two floors. Do this BEFORE any search: it
# measures delta_min (the G1 threshold) and sigma_seed (tau_stop and the
# one-SE scale). Both are measured, never guessed.
python3 npe_tune.py baseline --out-dir $OUT --n-seed-reps 2 --n-control 2
bash jobs/launch_npe_tune.sh $OUT                 # DRY RUN: check the count
bash jobs/launch_npe_tune.sh $OUT --submit --max 1   # one job first
# read the whole .o file, then:
bash jobs/launch_npe_tune.sh $OUT --submit

# Stage 3: the search, in batches. Repeat propose -> launch -> status.
python3 npe_tune.py propose --out-dir $OUT --n 8 --n-initial 20 \
    --noise-sd <measured run-to-run spread>
bash jobs/launch_npe_tune.sh $OUT --submit
python3 npe_tune.py status --out-dir $OUT

# Stage 4: the escalation decision is printed by `status`. Escalating just
# means running more propose/launch rounds -- the warm start makes that free.

# Stage 5: promote the top-K to M=10 and run the gate battery.
python3 npe_tune.py finalists --out-dir $OUT --top-k 3 --n-calib 500

# Stage 6: the learning curve at the chosen configuration.
python3 npe_tune.py learning-curve --out-dir $OUT
bash jobs/launch_npe_tune.sh $OUT --submit

# Stage 7: the report split, touched exactly once.
python3 npe_tune.py report --out-dir $OUT --real /path/to/sbi_real_cohort.parquet
```

## What correct output looks like

**`freeze-split`** -- the shape report with your `p`, `E`, `n` and topology
group count; `[ok] A-T1: one row per distinct theta`; a printed floor
`L_0`; a split hash. Record that hash: it identifies the experiment.

**one `evaluate` job** -- the shape report, one `[trial] id=...` line,
`M_probe` `[train] member` lines each ending in an epoch count, one
`[score]` line, one `[ledger] wrote ...` line. The score's NLL should sit
*below* the floor (a positive gain), or close to it. A gain that is large
and negative means the fit diverged.

**`status`** -- the ledger table sorted by NLL, the measured `delta_min`
and `sigma_seed`, the best-so-far trace, and one escalation verdict.

Two lines in `status` are stop-the-campaign signals:

- a shuffled control with a clearly positive gain. The control destroys the
  theta-z association, so its optimum is the prior. A positive gain there
  means the split leaks and nothing downstream is interpretable.
- `[warn] the mixture scored WORSE than the mean of its members`. Eq. (4)
  forbids this for an arithmetic mixture; it means the ensemble is being
  combined some other way.

**`finalists`** -- per candidate: the full-M score, then the gate battery.
G1 rejecting everything is the information-limited branch of the identity
`Delta = I(theta; z) - E[KL]`, not a bug. Do not relax a gate; the failure
routing is printed.

## Things worth knowing

- **Groups come from theta, not from `topo_idx`.** `gate_data.load_sim`
  does not return the row masks it applied, so an externally read
  `topo_idx` cannot be realigned to its output. Rows from one topology
  draw share the connectivity-kernel axes exactly, so grouping on those
  columns reproduces the partition from data that survives every filter.
- **Probe scores and full-M scores are different axes.** By eq. (4) the
  mixture score improves with M, so an `M_probe=3` score is not comparable
  to an `M=10` score. The ledger records `M` on every row for this reason.
- **Promotion is cheap because members are independent.** Extending an
  ensemble from 3 to 10 trains 7 members, not 10. This only works if
  `--save-model` was on (it is by default).
- **`report` may be run once.** Running it twice does not produce two
  independent numbers.
