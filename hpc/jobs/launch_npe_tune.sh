#!/usr/bin/env bash
#
# launch_npe_tune.sh -- submit one job per pending hyperparameter evaluation.
#
#   bash jobs/launch_npe_tune.sh run1                 # DRY RUN (default)
#   bash jobs/launch_npe_tune.sh run1 --submit        # actually submit
#   bash jobs/launch_npe_tune.sh run1 --submit --max 4
#
# DRY RUN IS THE DEFAULT ON PURPOSE. The first execution of a changed job
# script must never be a real submission, and never a full batch. Check that
# the printed job count equals the number of pending specs and that every
# -v variable appears, then re-run with --submit.
#
# Invoked as `bash jobs/launch_npe_tune.sh` rather than `./...` on purpose:
# that route ignores the shebang, so it still runs if a Windows transfer left
# CRLF line endings on this file.
#
# Guards run ONCE here, before the loop. Checking once beats discovering the
# same error separately in each of N jobs.

set -uo pipefail

OUT_DIR="${1:-}"
shift || true

SUBMIT=0
MAX=0
QSUB_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --submit)   SUBMIT=1 ;;
        --max)      shift; MAX="${1:-0}" ;;
        --qsub-args) shift; QSUB_ARGS="${1:-}" ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
    shift || true
done

if [ -z "${OUT_DIR}" ]; then
    echo "usage: bash jobs/launch_npe_tune.sh <out-dir> [--submit] [--max N]"
    exit 2
fi

# ---- guards, once ----------------------------------------------------------
FAIL=0
if [ ! -f "${OUT_DIR}/campaign.json" ]; then
    echo "GUARD FAIL: ${OUT_DIR}/campaign.json missing -- run freeze-split first"
    FAIL=1
fi
if [ ! -f "${OUT_DIR}/split_manifest.json" ]; then
    echo "GUARD FAIL: ${OUT_DIR}/split_manifest.json missing"
    FAIL=1
fi
if [ ! -f "jobs/npe_tune.pbs" ]; then
    echo "GUARD FAIL: jobs/npe_tune.pbs not found -- run this from hpc/"
    FAIL=1
fi
if [ "${SUBMIT}" -eq 1 ] && ! command -v qsub >/dev/null 2>&1; then
    echo "GUARD FAIL: --submit given but qsub is not on PATH"
    FAIL=1
fi
CR=$(tr -cd '\r' < jobs/npe_tune.pbs | wc -c | tr -d ' ')
if [ "${CR}" != "0" ]; then
    echo "GUARD FAIL: jobs/npe_tune.pbs contains ${CR} CR byte(s) -- a CRLF"
    echo "            transfer will make qsub fail with 'bad interpreter'."
    echo "            Fix with: sed -i 's/\r$//' jobs/npe_tune.pbs"
    FAIL=1
fi
if [ "${FAIL}" -ne 0 ]; then
    exit 2
fi

SPECS=$(ls -1 "${OUT_DIR}"/pending/trial_*.json 2>/dev/null)
if [ -z "${SPECS}" ]; then
    echo "nothing pending in ${OUT_DIR}/pending -- run 'npe_tune.py propose' first"
    exit 0
fi

N=$(echo "${SPECS}" | wc -l | tr -d ' ')
echo "pending specs : ${N}"
echo "mode          : $([ "${SUBMIT}" -eq 1 ] && echo SUBMIT || echo 'DRY RUN (no jobs submitted)')"
[ "${MAX}" -gt 0 ] && echo "cap           : ${MAX}"
echo "----------------------------------------------------------------"

COUNT=0
for SPEC in ${SPECS}; do
    if [ "${MAX}" -gt 0 ] && [ "${COUNT}" -ge "${MAX}" ]; then
        echo "(cap reached; $((N - COUNT)) spec(s) left for the next batch)"
        break
    fi
    CMD="qsub -v OUT_DIR=${OUT_DIR},SPEC=${SPEC} ${QSUB_ARGS} jobs/npe_tune.pbs"
    if [ "${SUBMIT}" -eq 1 ]; then
        echo "${CMD}"
        eval "${CMD}"
    else
        echo "${CMD}"
    fi
    COUNT=$((COUNT + 1))
done

echo "----------------------------------------------------------------"
echo "$([ "${SUBMIT}" -eq 1 ] && echo submitted || echo 'would submit') ${COUNT} job(s)"
if [ "${SUBMIT}" -eq 0 ]; then
    echo "Re-run with --submit once the count and the -v variables look right."
fi
