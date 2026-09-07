#!/usr/bin/env bash
#
# launch_joint_tune.sh -- submit one job per pending joint-tuning spec.
#
#   bash jobs/launch_joint_tune.sh S-A1 /path/tune       # DRY RUN (default)
#   bash jobs/launch_joint_tune.sh S-A1 /path/tune --submit
#   bash jobs/launch_joint_tune.sh S-A1 /path/tune --submit --max 4
#
# DRY RUN IS THE DEFAULT ON PURPOSE. The first execution of a changed job
# script must never be a real submission, and never a full array. Check that
# the printed element count equals the number of pending specs and that every
# -v variable appears, then re-run with --submit.
#
# Invoked as `bash jobs/launch_joint_tune.sh` rather than `./...` on purpose:
# that route ignores the shebang, so it still runs if a Windows transfer left
# CRLF line endings on this file.
#
# Guards run ONCE here, before submitting. Checking once beats discovering the
# same error separately in each of N elements, each after its own queue wait.

set -uo pipefail

CAMPAIGN="${1:-}"
RESULTS_DIR="${2:-}"
shift 2 2>/dev/null || true

SUBMIT=0
MAX=0
QSUB_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --submit)    SUBMIT=1 ;;
        --max)       shift; MAX="${1:-0}" ;;
        --qsub-args) shift; QSUB_ARGS="${1:-}" ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
    shift || true
done

if [ -z "$CAMPAIGN" ] || [ -z "$RESULTS_DIR" ]; then
    echo "usage: bash jobs/launch_joint_tune.sh <campaign> <results-dir>" \
         "[--submit] [--max N] [--qsub-args '...']"
    exit 2
fi

STAGE4="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
JOB="$STAGE4/jobs/joint_tune.pbs"
PYBIN="${PYBIN:-python}"

# ---- guards, once ----------------------------------------------------------
FAIL=0
note() { echo "  $*"; }

echo "=========================================================="
echo "campaign    : $CAMPAIGN"
echo "results dir : $RESULTS_DIR"
echo "job script  : $JOB"
echo "=========================================================="

[ -f "$JOB" ] || { note "MISSING job script: $JOB"; FAIL=1; }
[ -f "$STAGE4/npe_tune_joint.py" ] || {
    note "MISSING $STAGE4/npe_tune_joint.py"; FAIL=1; }

PENDING_DIR="${RESULTS_DIR}/${CAMPAIGN}/pending"
if [ ! -d "$PENDING_DIR" ]; then
    note "no pending directory at $PENDING_DIR"
    note "run: $PYBIN npe_tune_joint.py propose --campaign $CAMPAIGN \\"
    note "         --results-dir $RESULTS_DIR --n-points N"
    FAIL=1
    N=0
else
    N="$(find "$PENDING_DIR" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')"
    note "pending specs: $N"
    [ "$N" -gt 0 ] || { note "nothing to submit"; FAIL=1; }
fi

# CR bytes in the job script mean qsub will fail with a bad-interpreter error
# that names /bin/bash and says nothing about line endings. Catch it here.
if [ -f "$JOB" ]; then
    CR="$(tr -cd '\r' < "$JOB" | wc -c | tr -d ' ')"
    if [ "$CR" != "0" ]; then
        note "$JOB has $CR CR byte(s): a Windows transfer corrupted it."
        note "fix with: sed -i 's/\\r\$//' $JOB"
        FAIL=1
    fi
fi

# Every free axis must reach the trainer, or the array spends its budget
# moving axes that never touch training.
if [ -f "$STAGE4/npe_tune_joint.py" ]; then
    CHECK="$(mktemp)"
    if ! (cd "$STAGE4" && "$PYBIN" npe_tune_joint.py space \
            --campaign "$CAMPAIGN" > "$CHECK" 2>&1); then
        # `space` exits non-zero for two different reasons and they need
        # different fixes: an unknown campaign is a typo, unreachable axes
        # are a missing flag. Reporting the first as the second sends the
        # reader to edit run_joint_arms.py over a misspelling.
        if grep -q "BLOCKING" "$CHECK"; then
            note "campaign $CAMPAIGN has FREE axes that do not reach the"
            note "trainer, so the array would spend its budget moving axes"
            note "that never touch training:"
            grep -A 20 "BLOCKING" "$CHECK" | head -12 | \
                while IFS= read -r l; do note "  $l"; done
        else
            note "npe_tune_joint.py space failed for campaign $CAMPAIGN:"
            tail -5 "$CHECK" | while IFS= read -r l; do note "  $l"; done
        fi
        FAIL=1
    else
        note "every free axis of $CAMPAIGN reaches the trainer"
    fi
    rm -f "$CHECK"
fi

for v in SIM_SHARDS OUT_DIR; do
    eval "val=\${$v:-}"
    if [ -z "$val" ]; then
        note "environment variable $v is unset; export it before submitting"
        FAIL=1
    fi
done

if [ "$FAIL" != "0" ]; then
    echo ""
    echo "GUARDS FAILED -- nothing submitted."
    exit 1
fi

LAST=$((N - 1))
if [ "$MAX" != "0" ] && [ "$MAX" -lt "$N" ]; then
    LAST=$((MAX - 1))
    note "limiting to the first $MAX element(s)"
fi

VARS="CAMPAIGN=$CAMPAIGN,RESULTS_DIR=$RESULTS_DIR,SIM_SHARDS=$SIM_SHARDS,OUT_DIR=$OUT_DIR"
[ -n "${REAL_SHARDS:-}" ]     && VARS="$VARS,REAL_SHARDS=$REAL_SHARDS"
[ -n "${SPLIT_HASH:-}" ]      && VARS="$VARS,SPLIT_HASH=$SPLIT_HASH"
[ -n "${CONTRACT_DIGEST:-}" ] && VARS="$VARS,CONTRACT_DIGEST=$CONTRACT_DIGEST"
[ -n "${EPOCHS:-}" ]          && VARS="$VARS,EPOCHS=$EPOCHS"
[ -n "${STEPS_PER_EPOCH:-}" ] && VARS="$VARS,STEPS_PER_EPOCH=$STEPS_PER_EPOCH"
[ -n "${SEED:-}" ]            && VARS="$VARS,SEED=$SEED"
[ -n "${DSN_MAIN_DIR:-}" ]    && VARS="$VARS,DSN_MAIN_DIR=$DSN_MAIN_DIR"
[ -n "${SBI_HPC_DIR:-}" ]     && VARS="$VARS,SBI_HPC_DIR=$SBI_HPC_DIR"

CMD="qsub -J 0-$LAST -v $VARS $QSUB_ARGS $JOB"

echo ""
echo "would submit ONE array of $((LAST + 1)) element(s):"
echo "  $CMD"
echo ""

if [ "$SUBMIT" = "0" ]; then
    echo "DRY RUN -- nothing submitted. Re-run with --submit once the above"
    echo "is what you expect. Before the first real submission, run one"
    echo "element locally:"
    echo ""
    echo "  DRYRUN=1 CAMPAIGN=$CAMPAIGN RESULTS_DIR=$RESULTS_DIR \\"
    echo "  SIM_SHARDS='$SIM_SHARDS' OUT_DIR='$OUT_DIR' \\"
    echo "  PBS_ARRAY_INDEX=0 bash $JOB"
    exit 0
fi

echo "submitting..."
eval "$CMD"
