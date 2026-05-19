# MGC-6D: Multi-Candidate Geometry Collaboration for Single-Anchor 6-DoF Pose Estimation

MGC-6D is the reference code for **Multi-Candidate Geometry Collaboration
for Single-anchor 6-DoF Pose Estimation**. The method reconstructs multiple
geometry candidates from a single anchor observation, calibrates their anchor
reliability, and selects or reuses the best candidate during query-frame pose
estimation.

> Project rename note: earlier internal handoff notes and server paths may still use `GeoAnchor`. The public project name is **MGC-6D**; the server checkout path and existing conda environment can remain unchanged for compatibility.

The cleaned first release focuses on the HO3D/YCB workflow. REAL275 and
Toyota-Light scripts are included as advanced experimental entry points, but
HO3D is the maintained one-command reproduction path.

## Repository Layout

```text
anchor_paper.py                         # anchor reconstruction and candidate registry generation
query_paper.py                          # HO3D query evaluation with multi-candidate selection
construction.py                         # RaySt3R / InstantMesh / fusion reconstruction utilities
sam2_rayst3r.py                         # SAM2 + RaySt3R wrappers
sam2_instantmesh.py                     # SAM2 + InstantMesh wrappers
foundationpose/                         # FoundationPose dependency code
instantmesh/                            # InstantMesh dependency code
sam2/                                   # SAM2 dependency code
bop_toolkit/                            # BOP evaluation utilities
docs/                                   # install, data, and experiment notes
third_party/rayst3r_patches/            # small RaySt3R compatibility patch
```

## Installation

The development environment used Python 3.11.14, PyTorch 2.5.1+cu121,
TorchVision 0.20.1+cu121, CUDA 12.1, and driver 565.57.01 on RTX 3090 GPUs.

```bash
conda env create -f environment.yml
conda activate geoanchor

# FoundationPose pose refinement requires a local C++ extension.
# If pybind11 is not visible to CMake, install it first:
python -m pip install pybind11

cd foundationpose/mycpp
rm -rf build && mkdir -p build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python) \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
make -j$(nproc)
cd ../../..

# Optional: build the FoundationPose CUDA helper used by BundleSDF paths.
cd foundationpose/bundlesdf/mycuda
python -m pip install -e .
cd ../../..

cd sam2
pip install -e .
cd ..
```

RaySt3R is kept external:

```bash
git clone https://github.com/Duisterhof/rayst3r.git ../rayst3r
export RAYST3R_ROOT=$(realpath ../rayst3r)
```

If your RaySt3R checkout ignores the selected CUDA device, apply the patch
intent in `third_party/rayst3r_patches/device_eval.patch`.

### Verify FoundationPose `mycpp`

`query_paper.py` and the dataset-specific query scripts call
`foundationpose.mycpp.cluster_poses` during Any6D initialization. If this
extension is missing, the run fails early with
`AttributeError: 'NoneType' object has no attribute 'cluster_poses'`.
Verify the build before running experiments:

```bash
python - <<'PY'
from foundationpose.Utils import mycpp
print(mycpp.__file__)
print('cluster_poses:', hasattr(mycpp, 'cluster_poses'))
PY
```

The expected output points to
`foundationpose/mycpp/build/mycpp*.so` and prints `cluster_poses: True`.

## Data And Checkpoints

Keep datasets, generated results, and model weights outside git. Set paths at
runtime:

```bash
export HO3D_ROOT=/path/to/ho3d
export YCB_MODEL_PATH=/path/to/ho3d/YCB_Video_Models
export SAM2_CKPT=/path/to/sam2.1_hiera_large.pt
```

RaySt3R weights are downloaded from Hugging Face by default with
`hf_hub_download("bartduis/rayst3r", "rayst3r.pth")`. You can also pass
`--rayst3r_checkpoint /path/to/rayst3r.pth`.

See `docs/datasets.md` for the expected HO3D layout and checkpoint folders.

If PyOpenGL or VisPy import stalls before any MGC-6D output appears on a
shared server, redirect Python bytecode caches to a local temporary directory:

```bash
export PYTHONPYCACHEPREFIX=/tmp/$USER/geoanchor_pycache
mkdir -p "$PYTHONPYCACHEPREFIX"
```

This only changes where Python stores and reads `.pyc` cache files. It does
not change model weights, datasets, CUDA kernels, or experiment results. See
`docs/installation.md` for details.

## Anchor Reconstruction

This step writes `candidate_registry.json`, which the query scripts consume.

```bash
python anchor_paper.py \
  --anchor_folder /path/to/anchor_results/dexycb_reference_view_ours \
  --ycb_model_path /path/to/ho3d/YCB_Video_Models \
  --depth_preprocess \
  --depth_unit_try_both \
  --refine_mask \
  --align_use_guess_translation \
  --align_bidirectional_icp \
  --rayst3r_set_conf 2.5 \
  --rayst3r_n_pred_views 5 \
  --rayst3r_filter_all_masks \
  --rayst3r_device cuda:0 \
  --instantmesh_device cuda:1 \
  --any6d_iter 5 \
  --any6d_refine 1 \
  --score_alpha 0.3 \
  --score_beta 0.7 \
  --seed 0
```

## Query Evaluation

```bash
python query_paper.py \
  --name ho3d_mgc6d_run1 \
  --anchor_path /path/to/paper_anchor_results/dexycb_reference_view_ours \
  --metric_anchor_path /path/to/metric_anchor_results/dexycb_reference_view_ours \
  --hot3d_data_root /path/to/ho3d \
  --ycb_model_path /path/to/ho3d/YCB_Video_Models \
  --ycbv_modesl_info_path ./models_info.json \
  --running_stride 10 \
  --register_iteration 5
```

## Outputs

- Anchor reconstruction outputs candidate meshes, poses, per-object metadata,
  and `candidate_registry.json` under the anchor result folder.
- Query evaluation writes metrics and frame-level outputs under
  `results/ho3d_results/<name>/`.

## License

MGC-6D-specific code and documentation in the repository root are released
under the MIT License. Vendored third-party subdirectories retain their own
licenses; see `third_party/README.md` and the license files inside each
third-party directory.
