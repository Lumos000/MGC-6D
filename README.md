# MGC-6D

Official inference code for **Multi-Candidate Geometry Collaboration for
Single-Anchor 6-DoF Pose Estimation**.

Given one RGB-D anchor observation, MGC-6D builds three geometry candidates
with RaySt3R, InstantMesh, and visible-patch fusion. FoundationPose estimates a
pose for each candidate, and the direct weighted score from Eq. (22) selects
one candidate independently for every query frame. This repository contains
the released HO3D, TOYL, and REAL275 evaluation pipelines only; it contains no
MGC-6D training, fine-tuning, learned reranking, or scorer checkpoint.

## Released protocol

- candidates: `rayst3r,instantmesh,fused`;
- alignment: deterministic 24-hypothesis ICP in both directions, with the
  better valid direction retained;
- appearance: `anchor_prior_hybrid`;
- query score: `alpha=0.4`, `beta=0.6`, and
  `(w_depth,w_mask,w_geom)=(0.5,0.25,0.25)`;
- registration: the standard FoundationPose hypothesis generator and scorer
  top-1, one run per candidate;
- HO3D masks: dataset GT anchor mask and precomputed XMem query masks;
- TOYL and REAL275 masks: dataset GT instance masks for anchor and query;
- temporal smoothing, GT-assisted symmetry correction, fallback candidates,
  and reference-conditioned selection: disabled.

The anchor registry records the Fitness/RMSE values from both independently
optimized ICP directions, their pass/fail decisions, reciprocal cycle error,
construction time, candidate paths, and candidate scores. Query results record
the complete command/provenance, evaluated coverage, runtime, every Eq. (22)
component, the selected candidate, and per-dataset selection frequencies.

## Installation

The validated setup uses Python 3.11, PyTorch 2.5.1+cu121, TorchVision
0.20.1+cu121, CUDA 12.1, and RTX 3090 GPUs.

```bash
git clone <repository-url> MGC-6D
cd MGC-6D
conda env create -f environment.yml
conda activate mgc6d
export CUDA_HOME=/usr/local/cuda-12.1
bash scripts/install_revision_dependencies.sh
```

RaySt3R is an external dependency. Pin the validated commit and apply the
included inference-only device patch:

```bash
git clone https://github.com/Duisterhof/rayst3r.git ../rayst3r
cd ../rayst3r
git checkout 2c93f81b696909695c6dd5367b81753225b7254e
git apply ../MGC-6D/third_party/rayst3r_patches/device_eval.patch
cd ../MGC-6D
export RAYST3R_ROOT="$(realpath ../rayst3r)"
```

The installer obtains nvdiffrast from commit
`253ac4fcea7de5f396371124af597e6cc957bfae` rather than PyPI, pins PyTorch3D
to `v0.7.8`, builds the query-time FoundationPose extension, and finishes with an
import/build smoke test. See [docs/installation.md](docs/installation.md) for
manual installation and troubleshooting.

## Data and checkpoints

Datasets, checkpoints, prepared anchor observations, and generated results are
kept outside Git. Each prepared anchor object directory contains `color.png`,
`depth.png`, `mask.png`, `K.txt`, and `<object>_gt_pose.txt`. The GT anchor pose
is copied for final pose evaluation only; reconstruction, alignment, candidate
scoring, and candidate selection do not use it. See
[docs/datasets.md](docs/datasets.md) for layouts and mask provenance.

FoundationPose weights belong under `foundationpose/weights/`, InstantMesh
weights under `instantmesh/ckpts/`, and `RAYST3R_CKPT` points to the local
RaySt3R checkpoint.

## Run the three datasets

All launchers resolve repository files relative to their own location and
refuse to overwrite an existing anchor output. `ANCHOR_GPUS` must expose two
GPUs: they become `cuda:0` for RaySt3R and `cuda:1` for InstantMesh inside the
anchor process. Query evaluation starts only after the registry validator has
confirmed all required anchor artifacts.

### HO3D

```bash
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r.pth
export ANCHOR_SOURCE=/path/to/prepared/ho3d_anchor_observations
export ANCHOR_OUT=/path/to/new/ho3d_anchor_results
export HO3D_ROOT=/path/to/HO3D
export YCB_MODEL_PATH="$HO3D_ROOT/YCB_Video_Models"
export MODELS_INFO="$PWD/models_info.json"
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
export QUERY_NAME=ho3d_mgc6d_seed0
bash scripts/run_ho3d_full.sh
```

For query-only reuse of a validated registry:

```bash
export ANCHOR_RUN=/path/to/existing/ho3d_anchor_results
bash scripts/run_ho3d_query.sh
```

### TOYL

```bash
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r.pth
export ANCHOR_SOURCE=/path/to/prepared/toyl_anchor_observations
export ANCHOR_OUT=/path/to/new/toyl_anchor_results
export TOYL_ROOT=/path/to/TOYL
export TOYL_MODELS="$TOYL_ROOT/models"
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
export QUERY_NAME=toyl_mgc6d_seed0
bash scripts/run_toyl_full.sh
```

### REAL275

```bash
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r.pth
export ANCHOR_SOURCE=/path/to/prepared/real275_anchor_observations
export ANCHOR_OUT=/path/to/new/real275_anchor_results
export REAL_ROOT=/path/to/REAL275
export MODELS_INFO="$PWD/real275_models_info.json"
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
export QUERY_NAME=real275_mgc6d_seed0
bash scripts/run_real275_full.sh
```

## Official Oryon 2K pair protocol

For comparisons with Oryon, Any6D, CoordAR, and One2Any, use the official
Oryon pair lists rather than the fixed-anchor commands above. The launchers
download only the pair-list member from Oryon's data archive, verify its
SHA-256, prepare one anchor observation per unique reference, construct that
reference's own candidate pool, evaluate all 2,000 ordered pairs, and keep
failed pairs in the denominator with zero metrics.

- REAL275: 2,000 cross-scene pairs and 1,447 unique references;
- TOYL: 2,000 different-frame pairs and 890 unique references. TOYL's released
  rows use the same object scene on both sides despite the archive directory
  being named `cross_scene_test`.

Run REAL275 2K:

```bash
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r.pth
export REAL_ROOT=/path/to/REAL275
export MODELS_INFO="$PWD/real275_models_info.json"
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
export RUN_TAG=real275_oryon_2k_seed0
bash scripts/run_real275_oryon_2k.sh
```

Run TOYL 2K:

```bash
export PYTHON_BIN=/path/to/mgc6d/python
export RAYST3R_ROOT=/path/to/rayst3r
export RAYST3R_CKPT=/path/to/rayst3r.pth
export TOYL_ROOT=/path/to/TOYL
export TOYL_MODELS="$TOYL_ROOT/models"
export ANCHOR_GPUS=0,1
export QUERY_GPU=1
export SEED=0
export RUN_TAG=toyl_oryon_2k_seed0
bash scripts/run_toyl_oryon_2k.sh
```

Set `RUN_FETCH=0`, `RUN_PREPARE=0`, or `RUN_ANCHOR=0` to reuse the
corresponding completed stage. REAL275 anchor construction supports atomic
per-reference resume after a native crash. TOYL reconstructs the 890 unique
references in 21 resumable object batches. `SAVE_CANDIDATE_METRICS=1` adds
evaluation-only metrics for every candidate; it does not affect Eq. (22)
selection. See [docs/oryon_2k.md](docs/oryon_2k.md) for exact output contracts,
query-only examples, and validation commands.

TOYL uses query stride 1; HO3D and REAL275 use stride 10. The shared optional
environment variables `CANDIDATE_LABELS`, `CANDIDATE_COLOR_MODE`, `W_DEPTH`,
`W_MASK`, `W_GEOM`, `SCORE_ALPHA`, and `SCORE_BETA` map directly to recorded
CLI arguments. The commands above use the released values.

## Validate outputs

Query outputs are saved under `results/{ho3d,toyl,real275}_results/QUERY_NAME/`.
The validators independently replay Eq. (22), verify one argmax winner per
evaluated frame, check coverage and candidate labels, and report metrics,
candidate frequencies, and runtime. They do not impose a hidden performance
threshold.

```bash
python scripts/verify_ho3d_result.py RESULT_DIR
python scripts/verify_paper_result.py --dataset toyl RESULT_DIR
python scripts/verify_paper_result.py --dataset real275 RESULT_DIR
```

Run the repository checks before an experiment:

```bash
python scripts/smoke_imports.py
pytest -q tests/test_paper_protocol.py tests/test_revision_protocol.py
pytest -q tests/test_anchor_resume.py tests/test_oryon_real275_protocol.py \
  tests/test_oryon_toyl_protocol.py
```

Generated results, datasets, checkpoints, compiled extensions, and local
experiment files are excluded from version control.
