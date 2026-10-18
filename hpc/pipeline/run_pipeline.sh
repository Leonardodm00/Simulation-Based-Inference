#!/bin/bash
# =============================================================================
# run_pipeline.sh -- the one command per data set (D-081): from the spike
# trains to the search and the best fit, never pausing, resumable.
#
# Every mode works from ANY login node and leaves nothing running there
# (D-098, 2026-10-18): the sequencer runs as a chain of PBS jobs on compute
# nodes (pl_<tag>, the configuration's `driver` block), each submitting its
# successor before it starts ticking.
#
#   bash run_pipeline.sh --project <tag> --dry-run        # the stage table and every qsub line; submits nothing
#   bash run_pipeline.sh --project <tag>                  # submits the sequencer's first job and returns
#   bash run_pipeline.sh --project <tag> --status         # the stages, their jobs, the sequencer's jobs
#   bash run_pipeline.sh --project <tag> --stop [--wait S]  # the sequencer stops at its next tick; the jobs keep running
#   bash run_pipeline.sh --project <tag> --retire-login-sequencer [--wait S]
#        # once, 2026-10-18: stops a sequencer started on a login node by the code before D-098
#   bash run_pipeline.sh --project <tag> --foreground     # the sequencer in this shell: the tests only, never on a cluster
#
# --project <tag> names hpc/pipeline/projects/<tag>/<tag>.json; --config <path>
# names any configuration file. Anything else on the line goes to the
# submission (--retry STAGE, --accept-config-change, --break-lock: they reach
# the first sequencer job only).
#
# The short commands run under the project's extraction environment (the
# configuration's envs.extract) and need only Python's standard library.
# SKIP_CONDA=1 skips the activation (the tests; a shell already in the right
# environment) and is passed on as PIPELINE_SKIP_CONDA to the jobs.
#
# ORDER on a cluster: --dry-run first and read it; then the plain call; then
# --status whenever you like. The pass lines: the dry run ends without a
# REFUSE line; the plain call prints `SUBMITTED the sequencer: job <id>`; the
# log's last line at the end is `PIPELINE DONE: every stage done or skipped`,
# or `STOP: stage <name> failed: <reason>` -- fix, then submit again (with
# --retry <name> when the stage is recorded as failed).
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

CONFIG=""
MODE="start"
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --project)    shift; CONFIG="$HERE/projects/${1:-}/${1:-}.json" ;;
        --config)     shift; CONFIG="${1:-}" ;;
        --dry-run)    MODE="dry-run" ;;
        --status)     MODE="status" ;;
        --stop)       MODE="stop" ;;
        --retire-login-sequencer) MODE="retire" ;;
        --foreground) MODE="foreground" ;;
        -h|--help)    sed -n '2,36p' "$HERE/run_pipeline.sh"; exit 0 ;;
        *)            PASS+=("$1") ;;
    esac
    shift || true
done
[ -n "$CONFIG" ] || { echo "ABORT: give --project <tag> or --config <path>"; exit 2; }
[ -f "$CONFIG" ] || { echo "ABORT: no configuration at $CONFIG"; exit 2; }

command -v python3 >/dev/null 2>&1 || { echo "ABORT: no python3 on PATH (activate a conda environment first)"; exit 2; }

if [ "${SKIP_CONDA:-0}" != "1" ]; then
    ENV_NAME="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["envs"]["extract"])' "$CONFIG")" \
        || { echo "ABORT: cannot read envs.extract from $CONFIG"; exit 2; }
    # set -u MUST be off across the hook and the activate: conda's
    # deactivate.d/activate.d scripts are third-party and not -u clean.
    set +u
    eval "$(conda shell.bash hook)" || { echo "ABORT: conda shell.bash hook failed"; exit 2; }
    conda activate "$ENV_NAME" || { echo "ABORT: conda activate $ENV_NAME failed"; exit 2; }
    set -u
fi
export PIPELINE_SKIP_CONDA="${SKIP_CONDA:-0}"
echo "[pipeline] python3 : $(command -v python3)  (env ${CONDA_DEFAULT_ENV:-none})"
echo "[pipeline] config  : $CONFIG"

PATHS="$(python3 "$HERE/pipeline.py" paths --config "$CONFIG")" || { echo "$PATHS"; echo "ABORT: the configuration was refused"; exit 2; }
WORK_ROOT="$(sed -n 1p <<< "$PATHS")"
LOG="$(sed -n 2p <<< "$PATHS")"
STATE="$(sed -n 3p <<< "$PATHS")"

case "$MODE" in
    dry-run)
        exec python3 "$HERE/pipeline.py" run --config "$CONFIG" --dry-run ${PASS[@]+"${PASS[@]}"} ;;
    status)
        exec python3 "$HERE/pipeline.py" status --config "$CONFIG" ;;
    stop)
        exec python3 "$HERE/pipeline.py" stop --config "$CONFIG" ${PASS[@]+"${PASS[@]}"} ;;
    retire)
        exec python3 "$HERE/pipeline.py" retire --config "$CONFIG" ${PASS[@]+"${PASS[@]}"} ;;
    foreground)
        exec python3 "$HERE/pipeline.py" run --config "$CONFIG" ${PASS[@]+"${PASS[@]}"} ;;
    start)
        # D-098: submit the sequencer's first job; nothing stays on this node
        exec python3 "$HERE/pipeline.py" submit --config "$CONFIG" ${PASS[@]+"${PASS[@]}"} ;;
esac
