#!/usr/bin/env bash
set -euo pipefail

# Run the disclosed Eq. (22) HO3D query protocol with an existing registry.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${ANCHOR_RUN:?Set ANCHOR_RUN to an anchor folder containing candidate_registry.json}"
: "${HO3D_ROOT:?Set HO3D_ROOT to the HO3D dataset root}"
: "${YCB_MODEL_PATH:?Set YCB_MODEL_PATH to the YCB model root}"
: "${MODELS_INFO:?Set MODELS_INFO to models_info.json}"

PYTHON_BIN="${PYTHON_BIN:-python}"
QUERY_GPU="${QUERY_GPU:-0}"
QUERY_NAME="${QUERY_NAME:-ho3d_open_release_eq22_seed0}"
RUNNING_STRIDE="${RUNNING_STRIDE:-10}"

if [[ ! -f "$ANCHOR_RUN/candidate_registry.json" ]]; then
  echo "Missing candidate registry: $ANCHOR_RUN/candidate_registry.json" >&2
  exit 2
fi

mkdir -p results/logs

CUDA_VISIBLE_DEVICES="$QUERY_GPU" "$PYTHON_BIN" -u query_paper.py \
  --name "$QUERY_NAME" \
  --seed 0 \
  --anchor_path "$ANCHOR_RUN" \
  --hot3d_data_root "$HO3D_ROOT" \
  --ycb_model_path "$YCB_MODEL_PATH" \
  --ycbv_modesl_info_path "$MODELS_INFO" \
  --running_stride "$RUNNING_STRIDE" \
  --register_iteration 5 \
  --per_frame_selection \
  --selection_policy weighted \
  --candidate_labels rayst3r,instantmesh,fused \
  --candidate_color_mode prior_transfer \
  --w_depth 0.5 \
  --w_mask 0.25 \
  --w_geom 0.25 \
  --score_alpha 0.4 \
  --score_beta 0.6 \
  --query_loss_normalization fixed_scale \
  --query_depth_scale 0.10 \
  --query_mask_scale 1.00 \
  --query_geom_scale 0.10 \
  --anchor_score_mode observation_calibrated \
  --anchor_depth_scale 0.05 \
  --anchor_mask_scale 0.50 \
  --anchor_geom_scale 0.05 \
  --anchor_calibration_weight 0.15 \
  --no-include_fallback_candidate \
  --no-symmetry_consistency_enable \
  --no-anchor_symmetry_align \
  2>&1 | tee "results/logs/query_${QUERY_NAME}.log"
