# MGC-6D

Official implementation of **Multi-Candidate Geometry Collaboration for
Single-Anchor 6-DoF Pose Estimation**.

This release freezes the reproducible HO3D path. From one RGB-D anchor
observation, it constructs RaySt3R, InstantMesh, and visible-patch fused
geometries; estimates one anchor pose for each candidate; and selects a
candidate independently in every query frame with the weighted score in the
paper's Eq. (22).

## Released HO3D protocol

The main protocol is deliberately explicit:

- anchor masks: dataset GT masks;
- query masks: the precomputed XMem masks distributed with the HO3D setup;
- geometry pool: `rayst3r,instantmesh,fused` only;
- alignment: 24 deterministic inverse-paired initializations in each ICP
  direction, followed by best-of-two direction selection;
- fusion: topology-preserving visible-patch update in the RaySt3R anchor frame;
- query selection: direct Eq. (22) weighted argmax, with
  `alpha=0.4`, `beta=0.6`, `(w_depth,w_mask,w_geom)=(0.5,0.25,0.25)`, and
  `w_pose=0`;
- appearance: `prior_transfer`;
- inference-time GT symmetry adjustment and temporal smoothing: disabled;
- external `obs` and legacy fallback candidates: disabled.

The reciprocal cycle error is a reported diagnostic. It is not used to choose
the ICP direction, admit a fused candidate, or score a query candidate.
`strict-both` is available only as the reviewer-requested ablation.

## Reference validation

The frozen protocol was checked on all 13 HO3D evaluation sequences with
`running_stride=10`: 2,046 frames were sampled, 38 frames whose HO3D metadata
has no object GT pose were explicitly skipped, and 2,008 valid-GT frames were
evaluated:

| ADD-S | ADD | AR | MSSD | MSPD | VSD |
|---:|---:|---:|---:|---:|---:|
| 98.7 | 45.1 | 47.7 | 53.1 | 37.4 | 52.5 |

The selected candidates were InstantMesh 1,366 times, fused 566 times, and
RaySt3R 76 times. The release verifier reproduced Eq. (22) from the saved score
components with a maximum absolute residual below `2e-16` and found no winner
mismatch. These values are a regression reference, not a promise of bitwise
identity across different CUDA, driver, and dependency builds.

## Installation

The validated environment used Python 3.11.14, PyTorch 2.5.1+cu121,
TorchVision 0.20.1+cu121, CUDA 12.1, and RTX 3090 GPUs.

```bash
conda env create -f environment.yml
conda activate mgc6d
```

`nvdiffrast` must be built from source after PyTorch is installed:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast@253ac4fcea7de5f396371124af597e6cc957bfae"
```

Build the FoundationPose extension used by query registration:

```bash
python -m pip install pybind11
cd foundationpose/mycpp
mkdir -p build
cd build
cmake .. -DPYTHON_EXECUTABLE="$(which python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
make -j"$(nproc)"
cd ../../..
```

Install SAM2 and keep RaySt3R as a sibling checkout:

```bash
cd sam2
pip install -e .
cd ..

git clone https://github.com/Duisterhof/rayst3r.git ../rayst3r
export RAYST3R_ROOT="$(realpath ../rayst3r)"
```

The validated RaySt3R checkout was commit
`2c93f81b696909695c6dd5367b81753225b7254e`, with the device-propagation
compatibility change documented in
`third_party/rayst3r_patches/device_eval.patch`. Apply it from the RaySt3R
checkout with:

```bash
git checkout 2c93f81b696909695c6dd5367b81753225b7254e
git apply /path/to/MGC-6D/third_party/rayst3r_patches/device_eval.patch
```

The other local RaySt3R training/data-loader experiments are not required by
MGC-6D inference. See
[`docs/installation.md`](docs/installation.md) for build checks and common
failures.

## Data preparation

Keep datasets and checkpoints outside git. Required paths are supplied as
environment variables:

```bash
export ANCHOR_SOURCE=/path/to/prepared/ho3d_anchor_observations
export ANCHOR_OUT=/path/to/new/ho3d_anchor_results
export HO3D_ROOT=/path/to/ho3d
export YCB_MODEL_PATH="$HO3D_ROOT/YCB_Video_Models"
export MODELS_INFO="$PWD/models_info.json"
export RAYST3R_CKPT=/path/to/rayst3r.pth
export PYTHON_BIN=/path/to/python
```

Each object under `ANCHOR_SOURCE` must contain `color.png`, `depth.png`,
`mask.png`, `K.txt`, and its `<object>_gt_pose.txt`. The GT pose is copied for
final pose evaluation only; reconstruction, anchor reliability, query scoring,
and candidate selection do not read it. See
[`docs/datasets.md`](docs/datasets.md) for the complete layout.

## Run anchor and query

Use two visible GPUs for anchor reconstruction and one for query evaluation.
Inside the anchor process the two visible devices are addressed as `cuda:0`
and `cuda:1`:

```bash
export ANCHOR_GPUS=0,2
export QUERY_GPU=0
export QUERY_NAME=ho3d_open_release_eq22_seed0
bash scripts/run_ho3d_full.sh
```

The script refuses to overwrite a non-empty `ANCHOR_OUT`. To evaluate an
existing registry without rebuilding anchors:

```bash
export ANCHOR_RUN=/path/to/ho3d_anchor_results
export QUERY_GPU=0
export QUERY_NAME=ho3d_open_release_eq22_seed0
bash scripts/run_ho3d_query.sh
```

Outputs are written below `results/ho3d_results/$QUERY_NAME/`. Every run saves
the CLI configuration, per-candidate Eq. (22) components, selected candidate,
aggregate metrics, and evaluation coverage. No `Audit_*` or per-candidate GT
metric is computed before selection.

Validate a completed result folder with:

```bash
python scripts/verify_ho3d_result.py \
  results/ho3d_results/ho3d_open_release_eq22_seed0/<timestamp>
```

## Reviewer ablations

The main command uses best-of-two and only reports reciprocal and independent
cycle diagnostics. To construct the strict two-direction ablation, rerun the
same anchor command from `scripts/run_ho3d_full.sh` in a new output directory
and add `--align_strict_both`. Do not combine strict-both anchors with the main
result name.

The optional flags `--refine_mask`, `--obs_anchor_path`, nonzero `--w_pose`,
`--symmetry_consistency_enable`, and `--anchor_symmetry_align` are ablations and
are intentionally absent from the released main command.

## Scope

HO3D is the maintained reproduction path in this release. Dataset-specific
REAL275 and Toyota-Light files remain experimental and are not invoked by the
HO3D scripts. Generated results, checkpoints, datasets, and compiled extensions
are excluded from version control.
