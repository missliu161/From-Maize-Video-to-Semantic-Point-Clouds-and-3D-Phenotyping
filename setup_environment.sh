#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${GSID_ENV_NAME:-gsid-seg}"
CUDA_VERSION_REQUIRED="11.8"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v conda >/dev/null 2>&1 || fail "conda was not found in PATH. Install Miniconda/Anaconda and initialize the shell first."
[[ "$(uname -s)" == Linux ]] || fail "This setup script currently supports Linux only."
[[ "$(uname -m)" == x86_64 ]] || fail "This setup script currently supports Linux x86_64 only."

command -v nvcc >/dev/null 2>&1 || fail "nvcc was not found. Install CUDA Toolkit ${CUDA_VERSION_REQUIRED} and make its bin directory available in PATH."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi was not found. Install a working NVIDIA driver before setting up the GPU workflow."
nvidia-smi -L >/dev/null 2>&1 || fail "No accessible NVIDIA GPU was found. The 2DGS workflow requires a working NVIDIA GPU/driver."

CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
CC="${CC:-$(command -v gcc || true)}"
CXX="${CXX:-$(command -v g++ || true)}"
[[ -x "$CC" ]] || fail "C compiler '$CC' is unavailable. Install gcc or set CC to a compatible compiler executable."
[[ -x "$CXX" ]] || fail "C++ compiler '$CXX' is unavailable. Install g++ or set CXX to a compatible compiler executable."

NVCC_VERSION="$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | awk '/release/ { for (i=1; i<=NF; i++) if ($i == "release") { gsub(",", "", $(i+1)); print $(i+1); exit } }')"
[[ "$NVCC_VERSION" == "$CUDA_VERSION_REQUIRED" ]] || fail "Found nvcc ${NVCC_VERSION:-unknown}; CUDA Toolkit ${CUDA_VERSION_REQUIRED} is required by the pinned PyTorch build."

CC_MAJOR="$("$CC" -dumpversion | cut -d. -f1)"
[[ "$CC_MAJOR" =~ ^[0-9]+$ ]] || fail "Could not determine the C compiler version from $CC."
(( CC_MAJOR <= 11 )) || fail "Found compiler version ${CC_MAJOR}; CUDA Toolkit ${CUDA_VERSION_REQUIRED} supports gcc versions up to 11. Select a compatible compiler with CC/CXX."
CXX_MAJOR="$("$CXX" -dumpversion | cut -d. -f1)"
[[ "$CXX_MAJOR" =~ ^[0-9]+$ ]] || fail "Could not determine the C++ compiler version from $CXX."
(( CXX_MAJOR <= 11 )) || fail "Found C++ compiler version ${CXX_MAJOR}; CUDA Toolkit ${CUDA_VERSION_REQUIRED} supports gcc versions up to 11. Select a compatible compiler with CC/CXX."

[[ -f "$SCRIPT_DIR/2dgs/environment.yml" ]] || fail "Missing 2dgs/environment.yml beside this script."
[[ -f "$SCRIPT_DIR/2dgs/requirements.txt" ]] || fail "Missing 2dgs/requirements.txt beside this script."

# Prevent user-site packages and unrelated PYTHONPATH entries from leaking into the new environment.
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

printf 'Creating a new Conda environment: %s\n' "$ENV_NAME"
printf 'Existing environments will not be updated or removed. If this name already exists, choose another with GSID_ENV_NAME.\n'
CONDA_NO_PLUGINS=true CONDA_SOLVER=classic conda env create \
  --name "$ENV_NAME" \
  --file "$SCRIPT_DIR/2dgs/environment.yml"

ENV_PREFIX="$(CONDA_NO_PLUGINS=true CONDA_SOLVER=classic conda run --no-capture-output -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
export CUDA_HOME CC CXX
export MAX_JOBS="${MAX_JOBS:-2}"

printf 'Installing pinned Python dependencies into %s\n' "$ENV_PREFIX"
CONDA_NO_PLUGINS=true CONDA_SOLVER=classic conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install --isolated -r "$SCRIPT_DIR/2dgs/requirements.txt"

printf 'Building diff-surfel-rasterization with CUDA_HOME=%s, CC=%s, CXX=%s, MAX_JOBS=%s\n' \
  "$CUDA_HOME" "$CC" "$CXX" "$MAX_JOBS"
CONDA_NO_PLUGINS=true CONDA_SOLVER=classic conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip --isolated install --no-build-isolation "$SCRIPT_DIR/2dgs/submodules/diff-surfel-rasterization"

printf 'Building simple-knn\n'
CONDA_NO_PLUGINS=true CONDA_SOLVER=classic conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip --isolated install --no-build-isolation "$SCRIPT_DIR/2dgs/submodules/simple-knn"

printf 'Checking the completed environment\n'
CONDA_NO_PLUGINS=true CONDA_SOLVER=classic conda run --no-capture-output -n "$ENV_NAME" \
  python "$SCRIPT_DIR/check_environment.py"

printf '\nSetup complete. Activate it with:\n  conda activate %s\n' "$ENV_NAME"
