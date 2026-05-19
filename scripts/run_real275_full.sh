#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
NAME="${2:-real275_mgc6d_full_$(date +%Y%m%d_%H%M%S)}"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
exec /data/gst/envs/rayst3r/bin/python -u run_real275_query.py \
  --name "$NAME" \
  --anchor_path /data/gst/Any6D/Any6D/results/anchor_results/real275_reference_view \
  --real275_data_root /data/gst/data/REAL275 \
  --running_stride 10 \
  --register_iteration 5
