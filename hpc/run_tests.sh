#!/usr/bin/env bash
#
# run_tests.sh -- run every smoke test suite and report one verdict.
#
# Usage, from inside the hpc/ directory:
#
#     bash run_tests.sh               # everything (~12 min on one core)
#     bash run_tests.sh --fast        # skip the network-training tests (~40 s)
#     bash run_tests.sh -k T6         # only tests whose id contains T6
#     bash run_tests.sh --no-env      # skip conda activation (already active)
#     ENV_NAME=my_env bash run_tests.sh
#
# Invoked with `bash run_tests.sh` rather than `./run_tests.sh` on purpose:
# that route ignores the shebang, so it still runs if a Windows transfer left
# CRLF line endings on this file.
#
# Exit code is 0 only if EVERY stage passed, so this is safe to use as the
# last command of a PBS job.

set -uo pipefail

FAST=""
SELECTOR=""
SKIP_ENV=""
ENV_NAME="${ENV_NAME:-sbi_env}"

while [ $# -gt 0 ]; do
    case "$1" in
        --fast)    FAST="--fast"; shift ;;
        --no-env)  SKIP_ENV="1"; shift ;;
        -k)        SELECTOR="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

cd "$(dirname "$0")" || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="logs/tests_${STAMP}.log"

# Everything from here is written to the terminal AND to the log.
exec > >(tee -a "$LOG") 2>&1

STAGES=()
CODES=()
TIMES=()

hdr()  { printf '\n%s\n=== %s\n%s\n' "======================================================================" "$1" "======================================================================"; }
note() { printf '  %s\n' "$1"; }

stage() {
    # stage <name> <command...>
    local name="$1"; shift
    hdr "$name"
    local t0 t1
    t0=$(date +%s)
    "$@"
    local code=$?
    t1=$(date +%s)
    STAGES+=("$name")
    CODES+=("$code")
    TIMES+=("$((t1 - t0))")
    if [ "$code" -eq 0 ]; then note "-> PASS"; else note "-> FAIL (exit $code)"; fi
    return 0   # never abort; collect every result
}

# ---------------------------------------------------------------------------
hdr "context"
note "host    : $(hostname)"
note "workdir : $(pwd)"
note "started : $(date -Is)"
note "log     : $LOG"
note "mode    : ${FAST:-full}${SELECTOR:+ (filter: $SELECTOR)}"

# ---------------------------------------------------------------------------
if [ -z "$SKIP_ENV" ]; then
    hdr "activate environment"
    if command -v conda >/dev/null 2>&1; then
        # Non-interactive shells do not read .bashrc, so `conda activate` is
        # undefined unless this profile script is sourced explicitly.
        #
        # `set +u` around it is REQUIRED, not cosmetic. conda.sh references
        # variables that are unset in a non-interactive shell (PS1 among
        # others). Under `set -u` that is a fatal error and bash exits on the
        # spot -- silently, because the message goes into the tee pipe. The
        # symptom is the script dying right after this header with no output.
        # It only shows up when launched from an env whose CONDA_* variables
        # are not already exported, which is why it can pass one day and fail
        # the next.
        set +u
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME" 2>/dev/null
        _act=$?
        set -u
        if [ "$_act" -eq 0 ]; then
            note "activated: $ENV_NAME"
        else
            note "could not activate '$ENV_NAME'; using the current interpreter"
        fi
    else
        note "conda not on PATH; using the current interpreter"
    fi
fi

PY="$(command -v python || command -v python3)"
[ -n "$PY" ] || { echo "FATAL: no python interpreter found"; exit 1; }
note "python  : $PY"
"$PY" -c "import sys; print('  version : ' + sys.version.split()[0])"

# ---------------------------------------------------------------------------
# Encoding guard. This runs FIRST because a mangled byte or a stray carriage
# return raises at import time, and the resulting error points at the wrong
# thing. \r is itself ASCII, so the byte scan alone does not catch CRLF --
# both checks are needed.
encoding_guard() {
    "$PY" - npe_contract.py npe_model.py gmm_benchmark.py \
              smoke_test_npe.py smoke_test_gmm.py npe_diagnostics.py \
              smoke_test_diagnostics.py -- check_env.py << 'PYEOF'
import sys

# Arguments before "--" are required; after it, optional. A missing optional
# file is a warning, a missing required one is fatal. Reporting a missing
# file with the remedy for a CORRUPTED file (run sed) sends the reader after
# a fix that cannot possibly work, so the two are kept distinct.
argv = sys.argv[1:]
split = argv.index('--') if '--' in argv else len(argv)
required, optional = argv[:split], argv[split + 1:]

missing_req, missing_opt, corrupted = [], [], []
for path in required + optional:
    try:
        data = open(path, 'rb').read()
    except OSError:
        (missing_req if path in required else missing_opt).append(path)
        print('  %-22s %s' % (path, 'MISSING (required)' if path in required
                              else 'missing (optional, skipped)'))
        continue
    bad = [(i + 1, hex(b)) for i, b in enumerate(data) if b > 127]
    crlf = data.count(b'\r\n')
    status = 'ASCII OK' if not bad else 'NON-ASCII %s' % bad[:4]
    print('  %-22s %-26s CRLF=%d' % (path, status, crlf))
    if bad or crlf:
        corrupted.append(path)

if corrupted:
    print('\n  CORRUPTED: %s' % ' '.join(corrupted))
    print("  Fix with:  sed -i 's/\\r$//' *.py *.sh jobs/*.pbs")
    print('  If non-ASCII bytes remain afterwards, re-transfer as a binary')
    print('  tar.gz -- never copy-paste or drag single text files.')
if missing_req:
    print('\n  MISSING (required): %s' % ' '.join(missing_req))
    print('  Restore with:  tar -xzf ~/sbi_hpc.tar.gz --strip-components=1 \\')
    print('                     %s' % ' '.join('hpc/' + m for m in missing_req))
if missing_opt:
    print('\n  Note: %s absent; the stage that uses it will be skipped.'
          % ' '.join(missing_opt))

sys.exit(1 if (corrupted or missing_req) else 0)
PYEOF
}
stage "encoding guard" encoding_guard

compile_check() {
    local files="npe_contract.py npe_model.py gmm_benchmark.py npe_diagnostics.py"
    files="$files smoke_test_npe.py smoke_test_gmm.py smoke_test_diagnostics.py"
    [ -f check_env.py ] && files="$files check_env.py"
    # shellcheck disable=SC2086
    "$PY" -m py_compile $files && note "all present modules compile"
}
stage "compile check" compile_check

# ---------------------------------------------------------------------------
if [ -f check_env.py ]; then
    stage "environment check" "$PY" check_env.py
else
    note ""
    note "SKIP: check_env.py not present, environment not verified"
fi

# ---------------------------------------------------------------------------
NPE_ARGS=()
[ -n "$FAST" ] && NPE_ARGS+=("$FAST")
[ -n "$SELECTOR" ] && NPE_ARGS+=(-k "$SELECTOR")

stage "suite: NPE pipeline (T1-T9)"        "$PY" smoke_test_npe.py "${NPE_ARGS[@]}"
stage "suite: Gaussian recovery (G1-G6)"   "$PY" smoke_test_gmm.py "${NPE_ARGS[@]}"

DIAG_ARGS=()
[ -n "$SELECTOR" ] && DIAG_ARGS+=(-k "$SELECTOR")
# The diagnostics suite scores against exact analytic posteriors, so it
# trains nothing and runs in seconds -- no --fast variant needed.
stage "suite: diagnostics (D1-D8)"        "$PY" smoke_test_diagnostics.py "${DIAG_ARGS[@]}"

LOCAL_ARGS=()
[ -n "$SELECTOR" ] && LOCAL_ARGS+=(-k "$SELECTOR")
# Unlike the diagnostics suite this one DOES have a --fast variant: L4 and
# L5 are rate tests over many seeds and take a few minutes, while L0-L3 run
# in well under a minute. L6-L8 need sbi and SKIP without it -- a skip is
# reported as a skip and does not affect this stage's exit code.
[ -n "$FAST" ] && LOCAL_ARGS+=(--fast)
stage "suite: local calibration (L0-L8)" "$PY" smoke_test_local.py "${LOCAL_ARGS[@]}"

MIS_ARGS=()
[ -n "$SELECTOR" ] && MIS_ARGS+=(-k "$SELECTOR")
# G4 is a rate test over many seeds; the rest run in about a minute.
[ -n "$FAST" ] && MIS_ARGS+=(--fast)
stage "suite: misspecification (G0-G7)" "$PY" smoke_test_misspec.py "${MIS_ARGS[@]}"

# ---------------------------------------------------------------------------
hdr "summary"
n_fail=0
printf '  %-34s %-8s %s\n' "STAGE" "RESULT" "SECONDS"
printf '  %-34s %-8s %s\n' "----------------------------------" "--------" "-------"
i=0
while [ "$i" -lt "${#STAGES[@]}" ]; do
    if [ "${CODES[$i]}" -eq 0 ]; then r="PASS"; else r="FAIL"; n_fail=$((n_fail + 1)); fi
    printf '  %-34s %-8s %s\n' "${STAGES[$i]}" "$r" "${TIMES[$i]}"
    i=$((i + 1))
done

echo
if [ "$n_fail" -eq 0 ]; then
    note "ALL ${#STAGES[@]} STAGES PASSED"
    note "finished: $(date -Is)"
    note "log kept at: $LOG"
    exit 0
fi
note "$n_fail OF ${#STAGES[@]} STAGES FAILED -- see $LOG"
note "finished: $(date -Is)"
exit 1
