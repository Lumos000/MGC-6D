#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
NAME="${2:-toyl_mgc6d_full_$(date +%Y%m%d_%H%M%S)}"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$GPU"
exec /data/gst/envs/rayst3r/bin/python -u query_paper_toyl.py \
  --name "$NAME" \
  --running_stride 1 \
  --query_multitry 1 \
  --register_iteration 5 \
  --mask_type mask_visib \
  --stable_query_mode register_simple \
  --include_legacy_candidate \
  --per_candidate_estimator \
  --prefer_legacy_candidate \
  --legacy_guard_margin 0.08 \
  --candidate_error_log_limit 30
