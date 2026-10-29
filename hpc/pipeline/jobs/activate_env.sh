# hpc/pipeline/jobs/activate_env.sh -- the environment's activation for the
# pipeline's jobs and in-environment commands (D-111; specs/SPEC.md block 7,
# "Environment faults and reruns" (a)). Sourced, not run:
#
#   . "$JOBS_DIR/activate_env.sh" && pipeline_activate <env>     # 0, or 96
#
# First conda, as the job scripts always did: `set +u` across the hook AND
# the activation (third-party activate.d / deactivate.d scripts are not -u
# clean), the caller's -u restored after; a success records the prefix in
# $WORK_ROOT/envs/<env>.prefix. When conda is absent from the node or fails
# there, the environment's own folder: the recorded prefix,
# $HOME/.conda/envs/<env>, then <d>/<env> for each d of CONDA_ENVS_PATH --
# the first whose python starts on this node -- activated as conda activates
# one environment (CONDA_PREFIX, CONDA_DEFAULT_ENV, CONDA_SHLVL, PATH, its
# activate.d scripts). Printed: `RUN_ARGV ENV: host <h>: conda unavailable
# (...); <env> activated from its folder <p>`. When nothing works:
# `RUN_ARGV ENV FAULT: host <h>: the environment <env> cannot start (...)`
# and status 96. With SKIP_CONDA=1 nothing is done (the tests).
#
# Pure ASCII, LF only; bash 4.

PIPELINE_ENV_FAULT_EXIT=96

_pa_host() {
    local h
    h=$(hostname 2>/dev/null) || h=""
    printf '%s' "${h:-?}"
}

_pa_first_line() {       # <file>: its first non-blank line, at most 200 characters
    local line
    if [ -f "$1" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            case "$line" in
                *[![:space:]]*) printf '%s' "${line:0:200}"; return 0 ;;
            esac
        done < "$1"
    fi
    return 0
}

_pa_record() {           # <env> <prefix>: remember the prefix conda gave (best effort)
    local env="$1" p="$2" d f cur
    [ -n "${WORK_ROOT:-}" ] && [ -n "$p" ] || return 0
    d="${WORK_ROOT}/envs"
    f="$d/$env.prefix"
    cur=""
    if [ -f "$f" ]; then
        IFS= read -r cur < "$f" 2>/dev/null || cur=""
    fi
    [ "$cur" = "$p" ] && return 0
    mkdir -p "$d" 2>/dev/null || return 0
    if printf '%s\n' "$p" > "$f.tmp.$$" 2>/dev/null; then
        mv -f "$f.tmp.$$" "$f" 2>/dev/null
    fi
    rm -f "$f.tmp.$$" 2>/dev/null
    return 0
}

_pa_usable() {           # <prefix>: 0 when its python starts on this node, else prints why
    local p="$1"
    if [ -z "$p" ]; then echo "empty"; return 1; fi
    if [ ! -d "$p/conda-meta" ]; then echo "no conda-meta"; return 1; fi
    if [ ! -x "$p/bin/python" ]; then echo "no bin/python"; return 1; fi
    if command -v timeout >/dev/null 2>&1; then
        timeout 120 "$p/bin/python" -c "import sys" >/dev/null 2>&1 \
            || { echo "its python does not start on this node"; return 1; }
    else
        "$p/bin/python" -c "import sys" >/dev/null 2>&1 \
            || { echo "its python does not start on this node"; return 1; }
    fi
    return 0
}

pipeline_activate() {    # <env>: 0 (conda, or the environment's folder) or 96
    local env="${1:-}" had_u=0 errf hook reason p why tried f rest d
    local -a cands=()
    local -A seen=()
    if [ "${SKIP_CONDA:-0}" = "1" ]; then
        return 0
    fi
    case $- in *u*) had_u=1 ;; esac
    set +u
    errf=$(mktemp 2>/dev/null) || errf="${TMPDIR:-/tmp}/pipeline_activate.$$"
    : > "$errf" 2>/dev/null
    # 1. conda, the way the job scripts always activated
    if hook=$(conda shell.bash hook 2>"$errf"); then
        if eval "$hook" 2>>"$errf" && conda activate "$env" 2>>"$errf"; then
            _pa_record "$env" "${CONDA_PREFIX:-}"
            rm -f "$errf" 2>/dev/null
            [ "$had_u" = "1" ] && set -u
            return 0
        fi
    fi
    reason=$(_pa_first_line "$errf")
    rm -f "$errf" 2>/dev/null
    [ -n "$reason" ] || reason="the hook or the activation failed"
    # 2. the environment's own folder
    if [ -n "${WORK_ROOT:-}" ] && [ -f "${WORK_ROOT}/envs/$env.prefix" ]; then
        p=""
        IFS= read -r p < "${WORK_ROOT}/envs/$env.prefix" 2>/dev/null || true
        [ -n "$p" ] && cands+=("$p")
    fi
    [ -n "${HOME:-}" ] && cands+=("${HOME}/.conda/envs/$env")
    rest="${CONDA_ENVS_PATH:-}"
    while [ -n "$rest" ]; do
        d="${rest%%:*}"
        if [ "$d" = "$rest" ]; then rest=""; else rest="${rest#*:}"; fi
        [ -n "$d" ] && cands+=("$d/$env")
    done
    tried=""
    for p in "${cands[@]}"; do
        [ -n "${seen[$p]:-}" ] && continue
        seen[$p]=1
        if why=$(_pa_usable "$p"); then
            export CONDA_PREFIX="$p" CONDA_DEFAULT_ENV="$env" CONDA_SHLVL=1
            export PATH="$p/bin:$PATH"
            if [ -d "$p/etc/conda/activate.d" ]; then
                for f in "$p"/etc/conda/activate.d/*.sh; do
                    [ -f "$f" ] && . "$f"
                done
            fi
            echo "RUN_ARGV ENV: host $(_pa_host): conda unavailable ($reason); $env activated from its folder $p"
            [ "$had_u" = "1" ] && set -u
            return 0
        fi
        tried="${tried:+$tried, }$p ($why)"
    done
    echo "RUN_ARGV ENV FAULT: host $(_pa_host): the environment $env cannot start (conda: $reason; folders: ${tried:-none found})"
    [ "$had_u" = "1" ] && set -u
    return "$PIPELINE_ENV_FAULT_EXIT"
}
