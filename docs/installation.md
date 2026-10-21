# Installation and model assets

MGC-6D was validated on Python 3.11.17, CUDA 12.1, PyTorch 2.5.1+cu121,
TorchVision 0.20.1+cu121, NVIDIA driver 565.57.01, and RTX 3090 24 GB GPUs.
Anchor construction uses two GPUs by default; query evaluation uses one.

## Environment

The Conda environment supplies CMake, Make, Boost, Eigen, Ninja, and pybind11.
The host still needs a C/C++ compiler compatible with CUDA 12.1.

```bash
git clone https://github.com/Lumos000/MGC-6D.git
cd MGC-6D
conda env create -f environment.yml
conda activate mgc6d
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash scripts/install_revision_dependencies.sh
```

For a pip installation, install the CUDA-enabled PyTorch wheel first:

```bash
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The installer builds pinned nvdiffrast commit
`253ac4fcea7de5f396371124af597e6cc957bfae`, PyTorch3D `v0.7.9`, SAM2, BOP
Toolkit, and the FoundationPose `mycpp` extension. Model weights are downloaded
separately as described below.

## RaySt3R

RaySt3R remains external. Pin the tested commit and apply the device patch:

```bash
git clone https://github.com/Duisterhof/rayst3r.git ../rayst3r
cd ../rayst3r
git checkout 2c93f81b696909695c6dd5367b81753225b7254e
git apply ../MGC-6D/third_party/rayst3r_patches/device_eval.patch
cd ../MGC-6D
export RAYST3R_ROOT="$(realpath ../rayst3r)"
```

Download `rayst3r.pth` from the public Hugging Face repository and retain the
returned cache path:

```bash
export RAYST3R_CKPT="$(python -c \
  'from huggingface_hub import hf_hub_download; print(hf_hub_download("bartduis/rayst3r", "rayst3r.pth"))')"
```

## DINOv2 required by RaySt3R

The RaySt3R wrapper intentionally loads DINOv2 from a local checkout and a
local checkpoint, avoiding network access during experiments.

```bash
git clone https://github.com/facebookresearch/dinov2.git ../dinov2
export MGC6D_DINOV2_REPO="$(realpath ../dinov2)"

export MGC6D_TORCH_HUB="$(python -c 'import torch; print(torch.hub.get_dir())')"
mkdir -p "$MGC6D_TORCH_HUB/checkpoints"
wget -O "$MGC6D_TORCH_HUB/checkpoints/dinov2_vitl14_reg4_pretrain.pth" \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
```

The required architecture is `dinov2_vitl14_reg`; the expected checkpoint
filename is fixed by `sam2_rayst3r.py`.

## InstantMesh and Zero123++

The released configuration is `instantmesh/configs/instant-mesh-large.yaml`
and requires two files from `TencentARC/InstantMesh`:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in (
    "diffusion_pytorch_model.bin",
    "instant_mesh_large.ckpt",
):
    print(hf_hub_download(
        "TencentARC/InstantMesh",
        filename,
        local_dir="instantmesh/ckpts",
    ))
PY
```

The multiview diffusion pipeline also loads `sudo-ai/zero123plus-v1.2` through
Diffusers. It is cached automatically on first use. To prepare an offline
machine, populate the Hugging Face cache before disconnecting it.

## FoundationPose

Download the official FoundationPose network weights from:

<https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing>

Only the following released directories are required:

```text
foundationpose/weights/
  2023-10-28-18-33-37/       # refiner
  2024-01-11-20-02-45/       # scorer
```

The query-time extension must expose `cluster_poses`. The installer builds it,
but the equivalent manual commands are:

```bash
python -m pip install pybind11
cd foundationpose/mycpp
mkdir -p build
cd build
cmake .. -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
make -j"$(nproc)"
cd ../../..

python - <<'PY'
from foundationpose.Utils import mycpp
print(mycpp.__file__)
print("cluster_poses:", hasattr(mycpp, "cluster_poses"))
PY
```

The bundled BundleSDF CUDA helper can be installed with:

```bash
cd foundationpose/bundlesdf/mycuda
python -m pip install -e .
cd ../../..
```

## nvdiffrast

nvdiffrast is installed from the exact source commit used by the release. It
must be built after PyTorch is installed and with CUDA available:

```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
python -m pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast@253ac4fcea7de5f396371124af597e6cc957bfae"
```

Pip 25.2 and newer ignore the `PIP_NO_BUILD_ISOLATION` environment variable;
use the explicit CLI option above.

## Verify the complete installation

After all code and checkpoints are present:

```bash
python scripts/smoke_imports.py --full
```

This checks dependency imports, FoundationPose extensions, and the released
CLI entry points. To execute small PyTorch3D and nvdiffrast CUDA kernel checks
on a machine with a visible GPU, run `python scripts/smoke_imports.py --full
--cuda`. Neither command downloads model weights or datasets.

## Optional checkpoint fingerprints

Record local hashes in the experiment archive without adding checkpoint files
to Git:

```bash
sha256sum \
  "$RAYST3R_CKPT" \
  "$MGC6D_TORCH_HUB/checkpoints/dinov2_vitl14_reg4_pretrain.pth" \
  instantmesh/ckpts/diffusion_pytorch_model.bin \
  instantmesh/ckpts/instant_mesh_large.ckpt
```

## Troubleshooting

### CMake cannot find pybind11

```bash
python -m pip install pybind11
cmake .. -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
```

### Missing local DINOv2 files

If RaySt3R raises `Local DINOv2 repository/weights are required`, confirm that
`MGC6D_DINOV2_REPO` points to a checkout containing `hubconf.py` and that
`dinov2_vitl14_reg4_pretrain.pth` is under the Torch Hub `checkpoints/`
directory printed by:

```bash
python -c 'import torch; print(torch.hub.get_dir())'
```

### PyOpenGL or VisPy import stalls

On some shared filesystems, redirect Python bytecode caches to local temporary
storage:

```bash
export PYTHONPYCACHEPREFIX=/tmp/$USER/mgc6d_pycache
mkdir -p "$PYTHONPYCACHEPREFIX"
PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" python scripts/smoke_imports.py
```

This changes only the bytecode-cache location, not datasets, weights, kernels,
or experiment outputs.
