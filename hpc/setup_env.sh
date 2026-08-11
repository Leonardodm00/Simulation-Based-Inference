#!/usr/bin/env bash
#
# setup_env.sh -- create and populate the conda environment for the SBI stage.
#
# Usage, from inside the hpc/ directory on the cluster:
#
#     bash setup_env.sh                 # CPU build of torch (default)
#     CUDA=cu124 bash setup_env.sh      # CUDA 12.4 build
#     CUDA=cu121 bash setup_env.sh      # CUDA 12.1 build
#     ENV_NAME=my_env bash setup_env.sh # different environment name
#     ENV_PREFIX=/scratch/me/envs/sbi_env bash setup_env.sh   # explicit path
#
# Invoked with `bash setup_env.sh` rather than `./setup_env.sh` on purpose:
# that route ignores the shebang, so the script still runs even if a Windows
# transfer left CRLF line endings on it. The script repairs its own line
# endings and those of its siblings below.
#
# Safe to re-run. An existing environment is reused, not rebuilt.
#
# Deliberately does NOT install into an existing brian_env: Brian2 needs its
# own compiler toolchain and codegen cache, and the simulator environment
# should stay reproducible independently of the inference one.

set -uo pipefail

ENV_NAME="${ENV_NAME:-sbi_env}"
CUDA="${CUDA:-cpu}"
PY_VER="${PY_VER:-3.11}"

say()  { printf '\n=== %s ===\n' "$1"; }
fail() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "0. normalise line endings (harmless if already clean)"
# \r is itself ASCII, so a byte scan does NOT catch CRLF. A shebang line
# ending in \r produces "bad interpreter: no such file or directory", which
# is an unhelpful error to meet at job submission time.
#
# The repair is applied UNCONDITIONALLY rather than gated behind a detection
# step. `grep -q $'\r'` is not reliable across greps -- it fails to match a
# genuine CR on some builds -- and `grep -P` is not always available. sed is
# idempotent on a file that is already clean, so running it always is both
# simpler and safer than deciding whether to run it. Detection below is used
# only to report, never to decide.
n_files=0
for f in ./*.py ./*.sh ./jobs/*.pbs; do
    [ -f "$f" ] || continue
    sed -i 's/\r$//' "$f" 2>/dev/null && n_files=$((n_files + 1))
done
echo "  normalised $n_files file(s)"

# Verify with awk, which does match a CR reliably where grep may not.
leftover=""
for f in ./*.py ./*.sh ./jobs/*.pbs; do
    [ -f "$f" ] || continue
    if awk '/\r/{found=1} END{exit !found}' "$f" 2>/dev/null; then
        leftover="$leftover $f"
    fi
done
if [ -n "$leftover" ]; then
    fail "carriage returns remain in:$leftover
 Re-transfer these as a binary archive (tar.gz) rather than by copy-paste."
fi
echo "  verified: no carriage returns remain"

# ---------------------------------------------------------------------------
say "1. locate conda"
if ! command -v conda >/dev/null 2>&1; then
    echo "  conda not on PATH; trying module system"
    for m in anaconda3 anaconda miniconda3 miniconda conda python/anaconda3; do
        module load "$m" >/dev/null 2>&1 && echo "  module load $m" && break
    done
fi
command -v conda >/dev/null 2>&1 || fail \
"conda not found. Run 'module avail 2>&1 | grep -i conda' to see what this
 cluster provides, then 'module load <name>' and re-run this script."

CONDA_BASE="$(conda info --base)" || fail "conda info --base failed"
echo "  conda base : $CONDA_BASE"
echo "  conda ver  : $(conda --version)"

# Non-interactive shells do not read .bashrc, so `conda activate` is not
# defined unless this profile script is sourced explicitly. Same reason the
# PBS job script does it.
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh" || fail "could not source conda.sh"

# ---------------------------------------------------------------------------
say "2. create the environment"
if [ -n "${ENV_PREFIX:-}" ]; then
    TARGET_ARGS=(-p "$ENV_PREFIX")
    TARGET_DESC="$ENV_PREFIX"
    EXISTS=$([ -d "$ENV_PREFIX" ] && echo yes || echo no)
else
    TARGET_ARGS=(-n "$ENV_NAME")
    TARGET_DESC="$ENV_NAME"
    EXISTS=$(conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" && echo yes || echo no)
fi

if [ "$EXISTS" = "yes" ]; then
    echo "  environment '$TARGET_DESC' already exists, reusing it"
else
    echo "  creating '$TARGET_DESC' with python $PY_VER"
    conda create "${TARGET_ARGS[@]}" "python=$PY_VER" pip -y \
        || fail "conda create failed. If this is a disk-quota error, retry with
 ENV_PREFIX=/path/to/scratch/envs/sbi_env bash setup_env.sh"
fi

conda activate "$TARGET_DESC" || fail "could not activate $TARGET_DESC"
echo "  active python: $(which python)"
python -c "import sys; print('  version      :', sys.version.split()[0])"

# ---------------------------------------------------------------------------
say "3. install torch ($CUDA)"
# --no-cache-dir matters on clusters: pip's cache lives under \$HOME and a
# torch download is large enough to trip a home quota on its own.
if [ "$CUDA" = "cpu" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
    TORCH_INDEX="https://download.pytorch.org/whl/$CUDA"
fi
echo "  index: $TORCH_INDEX"
python -m pip install --no-cache-dir --upgrade pip >/dev/null
python -m pip install --no-cache-dir torch --index-url "$TORCH_INDEX" \
    || fail "torch install failed. Check the CUDA tag against 'nvidia-smi' on a
 GPU node, then retry with CUDA=cuXXX bash setup_env.sh"

# ---------------------------------------------------------------------------
say "4. install the SBI stack"
# sbi and zuko are pinned: this code depends on API details that have moved
# between their releases.
python -m pip install --no-cache-dir \
    "sbi==0.27.0" "zuko==1.6.0" \
    "numpy>=1.24" "scipy>=1.10" "pandas>=2.0" "pyarrow>=14.0" "scikit-learn>=1.3" \
    || fail "package install failed"

# ---------------------------------------------------------------------------
say "5. verify"
if [ -f check_env.py ]; then
    python check_env.py
    STATUS=$?
else
    echo "  check_env.py not found in $(pwd); skipping"
    STATUS=0
fi

say "done"
cat <<EOF
Activate this environment in future sessions with:

    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate $TARGET_DESC

Next steps:
    python check_env.py             # if it did not run above
    python smoke_test_npe.py --fast # ~5 seconds
    python smoke_test_npe.py        # full, ~10 min on one core

Set CONDA_ENV=$TARGET_DESC in jobs/smoke_test.pbs before submitting.
EOF
exit "$STATUS"
