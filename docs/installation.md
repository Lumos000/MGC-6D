# Installation

MGC-6D was validated on Python 3.11.14, CUDA 12.1, PyTorch 2.5.1+cu121,
TorchVision 0.20.1+cu121, and NVIDIA driver 565.57.01. The development
server used four RTX 3090 24GB GPUs.

```bash
git clone <your-mgc-6d-repo-url>
cd MGC-6D

conda env create -f environment.yml
conda activate mgc6d
```

If you install with pip instead, install the CUDA-enabled PyTorch wheel first,
then install `requirements.txt`.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

FoundationPose contains C++/CUDA extensions. The most important one for
query-time pose refinement is `foundationpose/mycpp`; it provides
`cluster_poses`, which is called when the Any6D estimator initializes its
rotation grid. Build it after activating the environment:

```bash
# pybind11 must be visible to CMake.
python -m pip install pybind11

cd foundationpose/mycpp
rm -rf build && mkdir -p build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python) \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
make -j$(nproc)
cd ../../..
```

Verify the extension:

```bash
python - <<'PY'
from foundationpose.Utils import mycpp
print(mycpp.__file__)
print('cluster_poses:', hasattr(mycpp, 'cluster_poses'))
PY
```

The path should point to `foundationpose/mycpp/build/mycpp*.so`, and
`cluster_poses` should be `True`. If it is `False` or `mycpp` is `None`, query
scripts will fail with an error similar to:

```text
AttributeError: 'NoneType' object has no attribute 'cluster_poses'
```

The bundled FoundationPose helper under `foundationpose/bundlesdf/mycuda` can
then be installed with:

```bash
cd foundationpose/bundlesdf/mycuda
python -m pip install -e .
cd ../../..
```

Install SAM2 in editable mode:

```bash
cd sam2
pip install -e .
cd ..
```

RaySt3R is intentionally kept as an external dependency:

```bash
git clone https://github.com/Duisterhof/rayst3r.git ../rayst3r
export RAYST3R_ROOT=$(realpath ../rayst3r)
```

If your RaySt3R checkout does not propagate the selected device into
`eval_model`, apply `third_party/rayst3r_patches/device_eval.patch`.

## Troubleshooting

### CMake Cannot Find `pybind11`

If `cmake ..` in `foundationpose/mycpp/build` fails with:

```text
Could not find a package configuration file provided by "pybind11"
```

install pybind11 in the active environment and pass its CMake directory
explicitly:

```bash
python -m pip install pybind11
cmake .. -DPYTHON_EXECUTABLE=$(which python) \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
```

### PyOpenGL / VisPy Import Stalls

On some shared servers, importing PyOpenGL or VisPy from a conda environment
may stall while Python reads cached bytecode from the environment
`__pycache__` directories. If `scripts/smoke_imports.py` or
`query_paper.py --help` hangs before printing any MGC-6D output, redirect
Python bytecode caches to a local temporary directory:

```bash
export PYTHONPYCACHEPREFIX=/tmp/$USER/mgc6d_pycache
mkdir -p "$PYTHONPYCACHEPREFIX"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
python scripts/smoke_imports.py
```

This does not change model weights, datasets, CUDA kernels, or experiment
results. It only changes where Python stores and reads `.pyc` bytecode cache
files. The first run may regenerate cache files under `/tmp`; if `/tmp` is
cleaned, Python will regenerate them on the next run.
