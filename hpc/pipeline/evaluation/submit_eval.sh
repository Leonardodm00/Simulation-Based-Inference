#!/bin/bash
# hpc/pipeline/evaluation/submit_eval.sh -- the evaluation's one command on
# davinci's login node (specs/SPEC.md block 22): the calibration bank, one
# sampling job per complete ensemble, the report, each after what it needs.
#
#   bash hpc/pipeline/evaluation/submit_eval.sh --project synth --dry-run
#   bash hpc/pipeline/evaluation/submit_eval.sh --project synth
#   bash hpc/pipeline/evaluation/submit_eval.sh --project synth --what calbank,sample
#   bash hpc/pipeline/evaluation/submit_eval.sh --project synth --resample S-A1:A1
#
# --project <tag> reads hpc/pipeline/projects/<tag>/<tag>.json and
# <tag>_eval.json; --config <path> and --eval-config <path> name them
# instead. Everything else goes to ev_submit.py (its docstring: --what,
# --resample, --dry-run). Run it again whenever a best fit completes: what is
# done or queued is not submitted twice.
#
# The launcher needs only python's standard library: it runs in the
# configuration's envs.extract, as run_pipeline.sh does (set -u off across
# the conda hook and the activate). SKIP_CONDA=1 runs it under the shell's
# python3 (the tests).
#
# Exit: ev_submit.py's (0, 2, 3, 4); 2 when a file is missing or the
# environment cannot be activated.
#
# Pure ASCII, LF only.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PIPE="$(cd "$HERE/.." && pwd -P)"

CONFIG=""
EVAL=""
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --project)     shift; CONFIG="$PIPE/projects/${1:-}/${1:-}.json"
                       EVAL="$PIPE/projects/${1:-}/${1:-}_eval.json" ;;
        --config)      shift; CONFIG="${1:-}" ;;
        --eval-config) shift; EVAL="${1:-}" ;;
        -h|--help)     sed -n '2,25p' "$HERE/submit_eval.sh"; exit 0 ;;
        *)             PASS+=("$1") ;;
    esac
    shift || true
done
[ -n "$CONFIG" ] && [ -n "$EVAL" ] || { echo "ABORT: give --project <tag>, or --config and --eval-config"; exit 2; }
[ -f "$CONFIG" ] || { echo "ABORT: no configuration at $CONFIG"; exit 2; }
[ -f "$EVAL" ] || { echo "ABORT: no evaluation configuration at $EVAL"; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "ABORT: no python3 on PATH"; exit 2; }

if [ "${SKIP_CONDA:-0}" != "1" ]; then
    ENV_NAME="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["envs"]["extract"])' "$CONFIG")" \
        || { echo "ABORT: cannot read envs.extract from $CONFIG"; exit 2; }
    set +u
    eval "$(conda shell.bash hook)" || { echo "ABORT: conda shell.bash hook failed"; exit 2; }
    conda activate "$ENV_NAME" || { echo "ABORT: conda activate $ENV_NAME failed"; exit 2; }
    set -u
fi
export PIPELINE_SKIP_CONDA="${SKIP_CONDA:-0}"
echo "[evaluation] python3 : $(command -v python3)  (env ${CONDA_DEFAULT_ENV:-none})"

python3 "$HERE/ev_submit.py" --config "$CONFIG" --eval-config "$EVAL" ${PASS[@]+"${PASS[@]}"}
