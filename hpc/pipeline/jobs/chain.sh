# hpc/pipeline/jobs/chain.sh -- the sequencer's chain kept going from the
# jobs' side (D-111; specs/SPEC.md block 1, amended 2026-10-29 (g)). Sourced
# by run_argv.pbs, which sets WORK_ROOT, CHAIN_NAME, ARGV_FILE and PBS_JOBID:
#
#   pipeline_chain_replace <why>   a driver job that could not start its
#                                  command: unless another job of the chain
#                                  is queued, held or running, a replacement
#                                  from the template, delayed with -a
#   pipeline_watchdog              a stage job's end: a chain that is gone
#                                  without ending on purpose is revived
#   pipeline_chain_end <why>       a driver job's command refused (exit 2):
#                                  the chain ended on purpose
#
# The template, <WORK_ROOT>/driver/chain.qsub, is written by the sequencer
# at every driver submission: the qsub arguments of a driver job, one per
# line, with @ARGV@ for its argv file and @LOG@ for its log. The deliberate
# end is <WORK_ROOT>/driver/CHAIN_END; a requested stop <WORK_ROOT>/STOP_REQUEST.
#
# Pure ASCII, LF only; bash 4.

_ch_user() {
    local u="${USER:-${LOGNAME:-}}"
    [ -n "$u" ] || u=$(id -un 2>/dev/null)
    printf '%s' "$u"
}

_ch_dir() {
    printf '%s' "${WORK_ROOT:-}/driver"
}

_ch_others() {           # the chain's other live jobs, one per line; status 1 when qselect fails
    local out me id
    out=$(qselect -u "$(_ch_user)" -N "${CHAIN_NAME:-}" -s HQRSTUW 2>/dev/null) || return 1
    me="${PBS_JOBID:-}"
    me="${me%%.*}"
    for id in $out; do
        [ -n "$id" ] || continue
        [ "${id%%.*}" = "$me" ] && continue
        printf '%s\n' "$id"
    done
    return 0
}

_ch_log() {              # one line in the pipeline's log, as the sequencer writes them (best effort)
    [ -n "${WORK_ROOT:-}" ] && [ -d "${WORK_ROOT}" ] || return 0
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${WORK_ROOT}/pipeline.log" 2>/dev/null
    return 0
}

_ch_submit() {           # <argv file> <log> <delay s>: qsub from the template; prints the job id
    local tpl argv_file="$1" log="$2" delay="$3" line n i out when
    local -a args=()
    tpl="$(_ch_dir)/chain.qsub"
    [ -f "$tpl" ] || return 2
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line//@ARGV@/$argv_file}"
        line="${line//@LOG@/$log}"
        args+=("$line")
    done < "$tpl"
    n=${#args[@]}
    [ "$n" -ge 2 ] || return 2
    if [ "$delay" -gt 0 ]; then
        when=$(date -d "@$(( $(date +%s) + delay ))" '+%Y%m%d%H%M.%S') || return 3
        args=("${args[@]:0:$((n - 1))}" -a "$when" "${args[$((n - 1))]}")
    fi
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if out=$("${args[@]}" 2>&1); then
            printf '%s' "${out%%[[:space:]]*}"
            return 0
        fi
        [ "$i" -lt 10 ] && sleep "${PIPELINE_CHAIN_QSUB_WAIT_S:-30}"
    done
    printf '%s\n' "$out" >&2
    return 1
}

pipeline_chain_end() {   # <why>: the chain ended on purpose (written once)
    local f
    [ -n "${WORK_ROOT:-}" ] && [ -d "$(_ch_dir)" ] || return 0
    f="$(_ch_dir)/CHAIN_END"
    [ -f "$f" ] && return 0
    printf '%s %s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${PBS_JOBID:-?}" "$1" > "$f" 2>/dev/null
    echo "RUN_ARGV CHAIN: the chain ended on purpose: $1"
    return 0
}

pipeline_chain_replace() {   # <why>: a driver job that could not start its command
    local why="$1" others cnt n i delay base max id log host
    host=$(hostname 2>/dev/null) || host="?"
    if [ -z "${WORK_ROOT:-}" ] || [ -z "${CHAIN_NAME:-}" ]; then
        echo "RUN_ARGV CHAIN: no WORK_ROOT or CHAIN_NAME: the chain ends here -- submit again"
        return 0
    fi
    if [ -f "${WORK_ROOT}/STOP_REQUEST" ] || [ -f "$(_ch_dir)/CHAIN_END" ]; then
        echo "RUN_ARGV CHAIN: a stop was requested or the chain ended on purpose: no replacement"
        return 0
    fi
    if others=$(_ch_others) && [ -n "$others" ]; then
        echo "RUN_ARGV CHAIN: the sequencer's chain goes on with" $others
        return 0
    fi
    if [ ! -f "$(_ch_dir)/chain.qsub" ]; then
        echo "RUN_ARGV CHAIN: no chain template $(_ch_dir)/chain.qsub: the chain ends here -- submit again"
        return 0
    fi
    cnt="$(_ch_dir)/env_replacements"
    n=0
    if [ -f "$cnt" ]; then
        IFS= read -r n < "$cnt" 2>/dev/null || n=0
    fi
    case "$n" in ''|*[!0-9]*) n=0 ;; esac
    n=$((n + 1))
    base="${PIPELINE_REPLACE_BASE_S:-300}"
    max="${PIPELINE_REPLACE_MAX_S:-3600}"
    delay="$base"
    i=1
    while [ "$i" -lt "$n" ] && [ "$delay" -lt "$max" ]; do
        delay=$((delay * 2))
        i=$((i + 1))
    done
    [ "$delay" -gt "$max" ] && delay="$max"
    printf '%d\n' "$n" > "$cnt" 2>/dev/null
    log="$(_ch_dir)/replace.$(date -u '+%Y%m%dT%H%M%SZ').$$.log"
    if id=$(_ch_submit "$ARGV_FILE" "$log" "$delay"); then
        echo "RUN_ARGV CHAIN: submitted the replacement sequencer job $id, to start in $delay s (replacement $n since the last full tick), log $log"
        _ch_log "driver ${PBS_JOBID:-?}: could not start on $host ($why); its replacement $id starts in $delay s (replacement $n since the last full tick; D-111)"
    else
        echo "RUN_ARGV CHAIN: the replacement could not be submitted: the chain ends here unless a stage job's watchdog revives it"
        _ch_log "driver ${PBS_JOBID:-?}: could not start on $host ($why), and its replacement could not be submitted (D-111)"
    fi
    return 0
}

pipeline_watchdog() {    # a stage job's end: revive a chain that is gone without ending on purpose
    local d others lock age id log host
    [ -n "${WORK_ROOT:-}" ] && [ -n "${CHAIN_NAME:-}" ] || return 0
    d="$(_ch_dir)"
    [ -d "$d" ] || return 0
    [ -f "${WORK_ROOT}/STOP_REQUEST" ] && return 0
    [ -f "$d/CHAIN_END" ] && return 0
    [ -f "$d/chain.qsub" ] && [ -f "$d/revive.argv" ] || return 0
    others=$(_ch_others) || return 0
    [ -z "$others" ] || return 0
    lock="$d/watchdog.lock"
    if ! mkdir "$lock" 2>/dev/null; then
        age=$(( $(date +%s) - $(stat -c %Y "$lock" 2>/dev/null || date +%s) ))
        [ "$age" -gt "${PIPELINE_WATCHDOG_STALE_S:-600}" ] || return 0
        rmdir "$lock" 2>/dev/null
        mkdir "$lock" 2>/dev/null || return 0
    fi
    host=$(hostname 2>/dev/null) || host="?"
    if [ ! -f "${WORK_ROOT}/STOP_REQUEST" ] && [ ! -f "$d/CHAIN_END" ] \
            && others=$(_ch_others) && [ -z "$others" ]; then
        log="$d/revive.$(date -u '+%Y%m%dT%H%M%SZ').$$.log"
        if id=$(_ch_submit "$d/revive.argv" "$log" 0); then
            echo "RUN_ARGV CHAIN: no sequencer job was queued, held or running and the chain had not ended on purpose; submitted $id to revive it, log $log"
            _ch_log "watchdog: job ${PBS_JOBID:-?} on $host: no sequencer job was queued, held or running and the chain had not ended on purpose; submitted $id to revive it (D-111)"
        else
            echo "RUN_ARGV CHAIN: the chain is gone and a sequencer job could not be submitted"
        fi
    fi
    rmdir "$lock" 2>/dev/null
    return 0
}
