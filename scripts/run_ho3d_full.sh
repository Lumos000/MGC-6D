#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
NAME="${2:-ho3d_mgc6d_full_$(date +%Y%m%d_%H%M%S)}"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$GPU"
export HO3D_ROOT=/data/gst/any6d/Any6D/data/ho3d
export YCB_MODEL_PATH=/data/gst/any6d/Any6D/data/ho3d/YCB_Video_Models
exec /data/gst/envs/rayst3r/bin/python -u query_paper.py \
  --name "$NAME" \
  --anchor_path /data/gst/Any6D/Any6D/results/anchor_results/paper/dexycb_reference_view_ours \
  --obs_anchor_path /data/gst/any6d/Any6D/results/anchor_results/dexycb_reference_view_ours \
  --hot3d_data_root /data/gst/any6d/Any6D/data/ho3d \
  --ycb_model_path /data/gst/any6d/Any6D/data/ho3d/YCB_Video_Models \
  --ycbv_modesl_info_path ./models_info.json \
  --running_stride 10 \
  --register_iteration 5
