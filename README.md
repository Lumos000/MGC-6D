# MGC-6D

Official implementation of **Multi-Candidate Geometry Collaboration for
Single-Anchor 6-DoF Pose Estimation**.

Given one RGB-D reference observation, MGC-6D constructs RaySt3R,
InstantMesh, and observation-constrained fused geometry candidates. Each
candidate is registered independently with FoundationPose, and Eq. (9)
selects the final relative pose using anchor and query-view evidence.

## Evaluation protocols

| Dataset | Main protocol | Evaluation size |
|---|---|---:|
| HO3D | Fixed anchor per object; stride-10 queries | 2,008 valid queries |
| TOYL | Official Oryon ordered-pair list | 2,000 pairs / 890 references |
| REAL275 | Official Oryon ordered-pair list | 2,000 pairs / 1,447 references |

Every TOYL or REAL275 reference constructs its own candidate pool. Failed
pairs remain in the 2,000-pair denominator with zero metrics. The separate
fixed-reference TOYL and REAL275 runs reproduce only the candidate-frequency
and runtime analyses in the paper; they are not used for the main accuracy
comparisons. See [docs/experiments.md](docs/experiments.md).

## Installation

The validated environment uses Python 3.11.17, PyTorch 2.5.1+cu121,
TorchVision 0.20.1+cu121, CUDA 12.1, and NVIDIA RTX 3090 GPUs.

```bash
git clone https://github.com/Lumos000/MGC-6D.git
cd MGC-6D
conda env create -f environment.yml
conda activate mgc6d
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash scripts/install_revision_dependencies.sh
```

RaySt3R is installed separately at the validated commit:

```bash
git clone https://github.com/Duisterhof/rayst3r.git ../rayst3r
cd ../rayst3r
git checkout 2c93f81b696909695c6dd5367b81753225b7254e
git apply ../MGC-6D/third_party/rayst3r_patches/device_eval.patch
cd ../MGC-6D
```

Required model assets:

| Component | File or location |
|---|---|
| RaySt3R | `RAYST3R_CKPT=/path/to/rayst3r.pth` |
| DINOv2 | Local DINOv2 repository and `dinov2_vitl14_reg4_pretrain.pth` |
| InstantMesh | `instantmesh/ckpts/diffusion_pytorch_model.bin` and `instantmesh/ckpts/instant_mesh_large.ckpt` |
| Zero123++ | Hugging Face model `sudo-ai/zero123plus-v1.2` |
| FoundationPose | `foundationpose/weights/2023-10-28-18-33-37/` and `foundationpose/weights/2024-01-11-20-02-45/` |

Download instructions and dependency details are in
[docs/installation.md](docs/installation.md). Dataset layouts, the seven HO3D
objects, prepared anchors, and checkpoint locations are documented in
[docs/datasets.md](docs/datasets.md).

## Reproducing the main results

Set the common paths and devices first. Anchor construction requires two
visible GPUs; query evaluation uses one GPU.

```bash
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r.pth
export MGC6D_DINOV2_REPO=/path/to/dinov2
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
```

### HO3D

```bash
export ANCHOR_SOURCE=/path/to/prepared/ho3d_anchor_observations
export ANCHOR_OUT=/path/to/new/ho3d_anchor_results
export HO3D_ROOT=/path/to/Any6D/data/ho3d
export YCB_MODEL_PATH="$HO3D_ROOT/YCB_Video_Models"
export MODELS_INFO="$PWD/models_info.json"
export QUERY_NAME=ho3d_mgc6d_seed0
bash scripts/run_ho3d_full.sh
```

### TOYL Oryon 2K

```bash
export TOYL_ROOT=/path/to/TOYL
export TOYL_MODELS="$TOYL_ROOT/models"
export RUN_TAG=toyl_oryon_2k_seed0
bash scripts/run_toyl_oryon_2k.sh
```

### REAL275 Oryon 2K

```bash
export REAL_ROOT=/path/to/REAL275
export MODELS_INFO="$PWD/real275_models_info.json"
export RUN_TAG=real275_oryon_2k_seed0
bash scripts/run_real275_oryon_2k.sh
```

The 2K launchers prepare the official Oryon pairs, construct one candidate
pool per unique reference, and evaluate all ordered pairs. Details and
query-only commands are in [docs/oryon_2k.md](docs/oryon_2k.md).

## Expected results

All values are percentages. ADD-S and ADD(-S) are distinct metrics; a dash
indicates that the paper does not report that metric for the benchmark.

| Dataset | ADD-S | ADD | ADD(-S) | AR | MSSD | MSPD | VSD |
|---|---:|---:|---:|---:|---:|---:|---:|
| HO3D | 98.1 | 50.7 | -- | 50.1 | 57.9 | 41.8 | 50.7 |
| TOYL Oryon 2K | -- | -- | 64.8 | 60.6 | 60.9 | 62.0 | 58.9 |
| REAL275 Oryon 2K | -- | -- | 82.6 | 73.9 | 78.3 | 84.8 | 58.4 |

## License

MGC-6D-specific code is released under the [MIT license](LICENSE). Vendored
third-party components retain their original licenses; see
[third_party/README.md](third_party/README.md).
