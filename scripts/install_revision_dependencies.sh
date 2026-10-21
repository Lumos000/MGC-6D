#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
ENV_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export CUDA_HOME
export PATH="$ENV_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export CMAKE_PREFIX_PATH="$ENV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae"
PYTORCH3D_COMMIT="33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba" # v0.7.9

"$PYTHON_BIN" -m pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast@${NVDIFFRAST_COMMIT}"
"$PYTHON_BIN" -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_COMMIT}"

"$PYTHON_BIN" -m pip install --no-build-isolation -e sam2
"$PYTHON_BIN" -m pip install --use-pep517 --no-build-isolation -e bop_toolkit

PYBIND11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)"
cmake -S foundationpose/mycpp -B foundationpose/mycpp/build \
  -DPython_EXECUTABLE="$PYTHON_BIN" \
  -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build foundationpose/mycpp/build --parallel "${BUILD_JOBS:-$(nproc)}"

"$PYTHON_BIN" scripts/smoke_imports.py --full
