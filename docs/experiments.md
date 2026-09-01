# HO3D experiments

The maintained paper path is encoded in two launchers so that the command in
the documentation cannot drift away from the implementation.

## Full anchor-to-query run

```bash
cd /path/to/MGC-6D
export RAYST3R_ROOT=/path/to/rayst3r
export ANCHOR_SOURCE=/path/to/prepared/ho3d_anchor_observations
export ANCHOR_OUT=/path/to/new/ho3d_anchor_results
export HO3D_ROOT=/path/to/ho3d
export YCB_MODEL_PATH="$HO3D_ROOT/YCB_Video_Models"
export MODELS_INFO="$PWD/models_info.json"
export RAYST3R_CKPT=/path/to/rayst3r.pth
export PYTHON_BIN=/path/to/python
export ANCHOR_GPUS=0,2
export QUERY_GPU=0
export QUERY_NAME=ho3d_open_release_eq22_seed0
bash scripts/run_ho3d_full.sh
```

`ANCHOR_GPUS` must expose two devices. They are remapped to `cuda:0` for
RaySt3R/depth processing and `cuda:1` for InstantMesh. The script reconstructs
all sorted object folders with object-specific deterministic seeds, writes a
candidate registry, releases the anchor process, and starts query evaluation.

The main anchor protocol uses dataset GT masks, a single Any6D anchor-pose run
per geometry label, ICP24 best-of-two alignment, topology-preserving
visible-patch fusion, and bounded RGB-D anchor reliability. It does not enable
SAM2 mask refinement, strict-both gating, relaxed ICP rescue, GT symmetry pose
adjustment, or cycle-based candidate scoring.

## Query-only run

```bash
export ANCHOR_RUN=/path/to/existing/ho3d_anchor_results
export HO3D_ROOT=/path/to/ho3d
export YCB_MODEL_PATH="$HO3D_ROOT/YCB_Video_Models"
export MODELS_INFO=/path/to/MGC-6D/models_info.json
export PYTHON_BIN=/path/to/python
export QUERY_GPU=0
export QUERY_NAME=ho3d_open_release_eq22_seed0
bash scripts/run_ho3d_query.sh
```

The query launcher uses only `rayst3r`, `instantmesh`, and `fused`, transfers
the InstantMesh prior appearance to all candidates, and chooses the direct
weighted-score argmax independently in each query frame. It never loads an
external `obs` candidate or legacy fallback candidate.

## Saved evidence

Anchor output contains `anchor_run_meta.json`, `candidate_registry.json`, and
per-object construction metadata. Query output contains
`query_run_manifest.json`, `evaluation_coverage.json`, aggregate/per-frame
metrics, and `0_all_candidate_scores.xlsx`. The latter records only quantities
available to the selector; no candidate-wise GT audit metric is produced.
For the 13-sequence stride-10 protocol, the coverage file should account for
2,046 sampled frames: 2,008 evaluated frames and 38 frames explicitly skipped
because the HO3D metadata contains no object GT pose.

```bash
python scripts/verify_ho3d_result.py \
  results/ho3d_results/ho3d_open_release_eq22_seed0/<timestamp>
```

The verifier checks the disclosed flags, replays Eq. (22), checks the argmax,
rejects `Audit_*` columns, and reports metrics and candidate frequencies.

## Bidirectional validation

The main registry records both independently optimized directional Fitness and
RMSE values, whether both pass the shared thresholds, independent-direction
cycle error, and reciprocal round-trip cycle error. The reciprocal diagnostic
starts from the inverse of the best-of-two transform and refines it in the
reverse direction. Cycle values do not affect the main candidate pool or score.

For the strict-both ablation, copy the anchor invocation in
`scripts/run_ho3d_full.sh`, add `--align_strict_both`, and use a distinct
`ANCHOR_OUT` and query name. Compare pose metrics together with fused-generation
rate and candidate availability; do not relabel strict-both as the main method.
