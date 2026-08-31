#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
NAME="${2:-real275_mgc6d_full_$(date +%Y%m%d_%H%M%S)}"
ANCHOR_PATH="${ANCHOR_PATH:-/data/gst/MGC-6D/results/anchor_results/real275_phi_v2_bounded_seed0}"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec /data/gst/envs/rayst3r/bin/python -u run_real275_query.py \
  --name "$NAME" \
  --anchor_path "$ANCHOR_PATH" \
  --real275_data_root /data/gst/data/REAL275 \
  --models_info_path /data/gst/MGC-6D/real275_models_info.json \
  --running_stride 10 \
  --register_iteration 5 \
  --no-use_tracking_init \
  --candidate_labels rayst3r,instantmesh,fused \
  --candidate_color_mode prior_transfer \
  --w_depth 0.5 \
  --w_mask 0.25 \
  --w_geom 0.25 \
  --w_pose 0 \
  --query_loss_normalization fixed_scale \
  --query_depth_scale 0.10 \
  --query_mask_scale 1.00 \
  --query_geom_scale 0.10 \
  --score_alpha 0.4 \
  --score_beta 0.6 \
  --eval_scale_mode baseline_fair \
  --seed 0
