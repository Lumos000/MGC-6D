#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
NAME="${2:-ho3d_mgc6d_full_$(date +%Y%m%d_%H%M%S)}"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$GPU"
: "${ANCHOR_RUN:?Set ANCHOR_RUN to the anchor result folder containing candidate_registry.json}"
: "${HO3D_ROOT:?Set HO3D_ROOT to the HO3D dataset root}"
: "${YCB_MODEL_PATH:?Set YCB_MODEL_PATH to the YCB model root}"
: "${MODELS_INFO:?Set MODELS_INFO to models_info.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" -u query_paper.py \
  --name "$NAME" \
  --anchor_path "$ANCHOR_RUN" \
  --hot3d_data_root "$HO3D_ROOT" \
  --ycb_model_path "$YCB_MODEL_PATH" \
  --ycbv_modesl_info_path "$MODELS_INFO" \
  --running_stride 10 \
  --register_iteration 5 \
  --per_frame_selection \
  --selection_policy weighted \
  --candidate_labels rayst3r,instantmesh,fused \
  --candidate_color_mode prior_transfer \
  --anchor_score_mode observation_calibrated \
  --anchor_depth_scale 0.05 \
  --anchor_mask_scale 0.50 \
  --anchor_geom_scale 0.05 \
  --anchor_calibration_weight 0.15 \
  --query_loss_normalization fixed_scale \
  --query_depth_scale 0.10 \
  --query_mask_scale 1.00 \
  --query_geom_scale 0.10 \
  --w_depth 0.5 \
  --w_mask 0.25 \
  --w_geom 0.25 \
  --w_pose 0.0 \
  --score_alpha 0.4 \
  --score_beta 0.6 \
  --seed 0 \
  --no-include_fallback_candidate
