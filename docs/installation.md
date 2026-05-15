# Installation


> Compatibility note: older internal checkouts may still use the former project name, and the existing conda environment may still be named `geoanchor`. These names do not change the public project name, **MGC-6D**.

MGC-6D was validated on Python 3.11.14, CUDA 12.1, PyTorch 2.5.1+cu121,
TorchVision 0.20.1+cu121, and NVIDIA driver 565.57.01. The development
server used four RTX 3090 24GB GPUs.

```bash
git clone <your-mgc-6d-repo-url>
cd MGC-6D

conda env create -f environment.yml
conda activate geoanchor
```

If you install with pip instead, install the CUDA-enabled PyTorch wheel first,
then install `requirements.txt`.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

FoundationPose contains CUDA/C++ extensions. Build them after activating the
environment:

```bash
cd foundationpose
bash build_all_conda.sh
cd ..
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

### PyOpenGL / VisPy Import Stalls

On some shared servers, importing PyOpenGL or VisPy from a conda environment
may stall while Python reads cached bytecode from the environment
`__pycache__` directories. If `scripts/smoke_imports.py` or
`query_paper.py --help` hangs before printing any MGC-6D output, redirect
Python bytecode caches to a local temporary directory:

```bash
export PYTHONPYCACHEPREFIX=/tmp/$USER/geoanchor_pycache
mkdir -p "$PYTHONPYCACHEPREFIX"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
python scripts/smoke_imports.py
```

This does not change model weights, datasets, CUDA kernels, or experiment
results. It only changes where Python stores and reads `.pyc` bytecode cache
files. The first run may regenerate cache files under `/tmp`; if `/tmp` is
cleaned, Python will regenerate them on the next run.
