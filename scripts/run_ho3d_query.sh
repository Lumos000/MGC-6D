#!/usr/bin/env bash
set -euo pipefail

# Run the disclosed Eq. (9) HO3D query protocol with an existing registry.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mgc6d_ho3d_matplotlib}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/mgc6d_ho3d_numba_cache}"
export MPLCONFIGDIR NUMBA_CACHE_DIR

: "${ANCHOR_RUN:?Set ANCHOR_RUN to an anchor folder containing candidate_registry.json}"
: "${HO3D_ROOT:?Set HO3D_ROOT to the HO3D dataset root}"
: "${YCB_MODEL_PATH:?Set YCB_MODEL_PATH to the YCB model root}"
: "${MODELS_INFO:?Set MODELS_INFO to models_info.json}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! RESOLVED_PYTHON_BIN="$(command -v "$PYTHON_BIN" 2>/dev/null)"; then
  RESOLVED_PYTHON_BIN="$(command -v python 2>/dev/null || true)"
  if [[ -z "$RESOLVED_PYTHON_BIN" ]]; then
    echo "Cannot find PYTHON_BIN=$PYTHON_BIN or a python executable in PATH." >&2
    exit 2
  fi
  echo "WARNING: PYTHON_BIN=$PYTHON_BIN is unavailable; using $RESOLVED_PYTHON_BIN" >&2
fi
PYTHON_BIN="$RESOLVED_PYTHON_BIN"
PYTHON_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SEED="${SEED:-0}"
QUERY_GPU="${QUERY_GPU:-0}"
QUERY_NAME="${QUERY_NAME:-ho3d_open_release_eq9_seed0}"
RUNNING_STRIDE="${RUNNING_STRIDE:-10}"
CANDIDATE_LABELS="${CANDIDATE_LABELS:-rayst3r,instantmesh,fused}"
CANDIDATE_COLOR_MODE="${CANDIDATE_COLOR_MODE:-anchor_prior_hybrid}"
W_DEPTH="${W_DEPTH:-0.5}"
W_MASK="${W_MASK:-0.25}"
W_GEOM="${W_GEOM:-0.25}"
SCORE_ALPHA="${SCORE_ALPHA:-0.4}"
SCORE_BETA="${SCORE_BETA:-0.6}"
OBJ_FOLDERS="${OBJ_FOLDERS:-}"
OBJECT_ARGS=()
if [[ -n "$OBJ_FOLDERS" ]]; then
  IFS=',' read -r -a OBJECT_LIST <<< "$OBJ_FOLDERS"
  OBJECT_ARGS=(--obj_folders "${OBJECT_LIST[@]}")
fi
if [[ ! -f "$ANCHOR_RUN/candidate_registry.json" ]]; then
  echo "Missing candidate registry: $ANCHOR_RUN/candidate_registry.json" >&2
  exit 2
fi

mkdir -p results/logs "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

CUDA_VISIBLE_DEVICES="$QUERY_GPU" "$PYTHON_BIN" -u query_paper.py \
  --name "$QUERY_NAME" \
  --seed "$SEED" \
  --anchor_path "$ANCHOR_RUN" \
  --hot3d_data_root "$HO3D_ROOT" \
  --ycb_model_path "$YCB_MODEL_PATH" \
  --ycbv_modesl_info_path "$MODELS_INFO" \
  --running_stride "$RUNNING_STRIDE" \
  --register_iteration 5 \
  --per_frame_selection \
  --selection_policy weighted \
  --candidate_labels "$CANDIDATE_LABELS" \
  --candidate_color_mode "$CANDIDATE_COLOR_MODE" \
  --w_depth "$W_DEPTH" \
  --w_mask "$W_MASK" \
  --w_geom "$W_GEOM" \
  --score_alpha "$SCORE_ALPHA" \
  --score_beta "$SCORE_BETA" \
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
  "${OBJECT_ARGS[@]}" \
  2>&1 | tee "results/logs/query_${QUERY_NAME}.log"
