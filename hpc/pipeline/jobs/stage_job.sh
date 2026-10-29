# hpc/pipeline/jobs/stage_job.sh -- the pipeline's job protocol for the job
# scripts of other repositories that the pipeline submits through their own
# launchers: Sbi-extractor's (S1, S2's gate, the bank's tasks and gate) and
# Astro-Neuron-Network's MEA job through Sbi-extractor's run_mea_task.pbs
# (D-113: D-111 and D-102 for every job of the pipeline, whatever the data
# set; specs/SPEC.md block 3, amended 2026-10-29 (b)). Sourced by absolute
# path when the pipeline passed PIPELINE_JOBS_DIR (this folder) through the
# launcher's qsub -v:
#
#   source "$PIPELINE_JOBS_DIR/stage_job.sh" || exit 2
#   pipeline_stage_job_start <env> || exit $?      # 0, 94 or 96
#   trap 'pipeline_stage_job_end' EXIT
#
# pipeline_stage_job_start prints `RUN_ARGV: host <h>, job <id>, env <env>`
# (the stages read the host there), then: on a host of the avoid list
# (AVOID_FILE, D-102) `RUN_ARGV AVOID: ...`, the slot held AVOID_HOLD_S
# seconds, the watchdog, status 94; else the activation of activate_env.sh
# (conda, else the environment's own folder); when neither starts,
# `RUN_ARGV ENV FAULT: ...` (activate_env.sh), the slot held, the watchdog,
# status 96. The stages submit such a job again, not counted as a
# resubmission (D-113). pipeline_stage_job_end runs the chain's watchdog
# (chain.sh: a chain gone without ending on purpose is revived); it never
# changes the job's exit status. Variables (qsub -v): WORK_ROOT, CHAIN_NAME,
# AVOID_FILE, AVOID_HOLD_S (default 600); SKIP_CONDA=1 skips the activation
# (the tests).
#
# Pure ASCII, LF only; bash 4.

_SJ_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=activate_env.sh
. "$_SJ_DIR/activate_env.sh" || return 2
# shellcheck source=chain.sh
. "$_SJ_DIR/chain.sh" || return 2

_sj_listed() {           # <host>: 0 when the avoid list names it
    [ -n "${AVOID_FILE:-}" ] && [ -f "$AVOID_FILE" ] || return 1
    grep -v '^[[:space:]]*#' "$AVOID_FILE" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
        | grep -qxF -- "$1"
}

pipeline_stage_job_start() {   # <env>: 0 (the environment is active), 94 or 96
    local env="${1:-}" host rc hold="${AVOID_HOLD_S:-600}"
    case "$hold" in ''|*[!0-9]*) hold=600 ;; esac
    host=$(hostname 2>/dev/null) || host="?"
    echo "RUN_ARGV: host ${host:-?}, job ${PBS_JOBID:-<none>}, env ${env:-<none>} (the pipeline's job protocol, D-113)"
    if _sj_listed "$host"; then
        echo "RUN_ARGV AVOID: host $host is on the avoid list $AVOID_FILE (D-102); holding this slot for $hold s, then exiting 94"
        sleep "$hold"
        pipeline_watchdog
        return 94
    fi
    pipeline_activate "$env"
    rc=$?
    if [ "$rc" != "0" ]; then
        echo "RUN_ARGV: holding this slot for $hold s (D-113), then exiting $rc"
        sleep "$hold"
        pipeline_watchdog
        return "$rc"
    fi
    return 0
}

pipeline_stage_job_end() {     # the job's end, whatever it was: the chain's watchdog
    pipeline_watchdog || true
    return 0
}
