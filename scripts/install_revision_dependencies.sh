#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae"
PYTORCH3D_TAG="v0.7.8"

"$PYTHON_BIN" -m pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast@${NVDIFFRAST_COMMIT}"
"$PYTHON_BIN" -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_TAG}"

"$PYTHON_BIN" -m pip install -e sam2
"$PYTHON_BIN" -m pip install -e bop_toolkit

PYBIND11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)"
cmake -S foundationpose/mycpp -B foundationpose/mycpp/build \
  -DPYTHON_EXECUTABLE="$(command -v "$PYTHON_BIN")" \
  -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build foundationpose/mycpp/build --parallel "${BUILD_JOBS:-$(nproc)}"

"$PYTHON_BIN" scripts/smoke_imports.py --full
