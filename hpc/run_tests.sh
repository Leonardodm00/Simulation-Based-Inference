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
              smoke_test_diagnostics.py -- check_env.py \
              bootstrap_paired.py smoke_test_bootstrap_paired.py \
              npe_tune.py npe_tune_data.py npe_tune_gates.py \
              smoke_test_tune.py joint/stage4/joint_space.py \
              joint/stage4/smoke_test_joint_space.py \
              joint/stage4/npe_tune_joint.py \
              joint/stage4/smoke_test_joint_tune.py << 'PYEOF'
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
    [ -f bootstrap_paired.py ] && files="$files bootstrap_paired.py smoke_test_bootstrap_paired.py"
    [ -f npe_tune.py ] && files="$files npe_tune.py npe_tune_data.py npe_tune_gates.py smoke_test_tune.py"
    [ -f joint/stage4/joint_space.py ] && files="$files joint/stage4/joint_space.py joint/stage4/smoke_test_joint_space.py joint/stage4/npe_tune_joint.py joint/stage4/smoke_test_joint_tune.py"
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

REG_ARGS=()
[ -n "$SELECTOR" ] && REG_ARGS+=(-k "$SELECTOR")
# No --fast variant: R0-R8 score against exact analytic posteriors and run
# in well under a minute, same as the diagnostics suite.
stage "suite: region extraction (R0-R8)" "$PY" smoke_test_regions.py "${REG_ARGS[@]}"

if [ -f smoke_test_tune.py ]; then
    TUNE_ARGS=()
    [ -n "$SELECTOR" ] && TUNE_ARGS+=(-k "$SELECTOR")
    # S11/S12/S13/S17 need torch, sbi or npe_model and SKIP without them; the
    # fast tier is pure numpy/scipy and includes the split tests (S18/S19) and
    # the control-statistic calibration tests (S20/S21).
    [ -n "$FAST" ] && TUNE_ARGS+=(--fast)
    stage "suite: tuning stack (S1-S21)" "$PY" smoke_test_tune.py "${TUNE_ARGS[@]}"
else
    note ""
    note "SKIP: smoke_test_tune.py not present"
fi

if [ -f joint/stage4/smoke_test_joint_space.py ]; then
    JS_ARGS=()
    [ -n "$SELECTOR" ] && JS_ARGS+=(-k "$SELECTOR")
    # Pure bookkeeping: no torch, no training, seconds. J23, J35 and parts
    # of J20/J21/J22/J26/J29 need the DSN's condition_space and SKIP or
    # narrow unless DSN_MAIN_DIR is set; J25/J26/J31 need skopt. Set
    # DSN_MAIN_DIR before this runs or the clause that matters most
    # (inactive-coordinate canonicalisation) is not exercised.
    # The suite reaches the DSN through DSN_MAIN_DIR and this directory
    # through SBI_HPC_DIR. Default SBI_HPC_DIR to where we already are, so
    # J26 does not skip merely because the caller did not know to set it.
    export SBI_HPC_DIR="${SBI_HPC_DIR:-$(pwd)}"
    # Three distinct failures, three different fixes. The cluster run of
    # 2026-09-07 hit the middle one -- the $HOME/dsn_main symlink was gone --
    # and the suites reported it as "unset", which sent the diagnosis the
    # wrong way for a cycle.
    if [ -z "${DSN_MAIN_DIR:-}" ]; then
        note "  NOTE: DSN_MAIN_DIR unset -- J23/J35 SKIP; J20-J22/J26/J29 narrow."
        note "        Those are the inactive-coordinate clauses; set it."
    elif [ ! -d "${DSN_MAIN_DIR}" ]; then
        note "  NOTE: DSN_MAIN_DIR=${DSN_MAIN_DIR} DOES NOT EXIST."
        note "        A deleted symlink looks exactly like this. Recreate"
        note "        it with ln -s pointing at the DSN repo Main directory"
        note "        (the real path contains a space, which is why the"
        note "        symlink exists at all: qsub -v cannot carry it)."
    elif [ ! -f "${DSN_MAIN_DIR}/condition_space.py" ]; then
        note "  NOTE: ${DSN_MAIN_DIR} has no condition_space.py -- that DSN"
        note "        checkout predates it. git pull the DSN repo."
    fi
    stage "suite: joint space (J20-J35)" "$PY" \
          joint/stage4/smoke_test_joint_space.py "${JS_ARGS[@]}"
    if [ -f joint/stage4/smoke_test_joint_tune.py ]; then
        # The tuner suite drives run_joint_arms only through its argv, which
        # it round-trips against that script's OWN parser (extracted by AST),
        # so it needs no torch. J29 needs the DSN for an S-A2 config; J31
        # needs skopt.
        stage "suite: joint tuner (J28-J36)" "$PY" \
              joint/stage4/smoke_test_joint_tune.py "${JS_ARGS[@]}"
    fi
else
    note ""
    note "SKIP: joint/stage4/smoke_test_joint_space.py not present"
fi

if [ -f smoke_test_bootstrap_paired.py ]; then
    BP_ARGS=()
    [ -n "$SELECTOR" ] && BP_ARGS+=(-k "$SELECTOR")
    # Pure numpy, no training: B1-B12 run in a few seconds. B13 is a coverage
    # rate test over 300 replicate datasets (the 57%-vs-93% result of plan
    # S2.4a) and is skipped under --fast.
    [ -z "$FAST" ] && BP_ARGS+=(--full)
    stage "suite: paired bootstrap (B1-B13)" "$PY" smoke_test_bootstrap_paired.py "${BP_ARGS[@]}"
else
    note ""
    note "SKIP: smoke_test_bootstrap_paired.py not present"
fi

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
