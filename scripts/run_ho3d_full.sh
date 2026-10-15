#!/usr/bin/env bash
set -euo pipefail

# Reconstruct all HO3D anchors and then run the paper-aligned query protocol.
# Physical GPUs listed in ANCHOR_GPUS become cuda:0 and cuda:1 inside the
# anchor process. The query uses the single physical GPU in QUERY_GPU.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mgc6d_ho3d_matplotlib}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/mgc6d_ho3d_numba_cache}"
export MPLCONFIGDIR NUMBA_CACHE_DIR

: "${ANCHOR_OUT:?Set ANCHOR_OUT to a new anchor-result folder}"
: "${YCB_MODEL_PATH:?Set YCB_MODEL_PATH to the YCB model root}"

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
SEED="${SEED:-0}"
ANCHOR_GPUS="${ANCHOR_GPUS:-0,1}"
QUERY_GPU="${QUERY_GPU:-0}"
QUERY_NAME="${QUERY_NAME:-ho3d_open_release_eq22_seed0}"
RUNNING_STRIDE="${RUNNING_STRIDE:-10}"
RUN_ANCHOR="${RUN_ANCHOR:-1}"
RUN_QUERY="${RUN_QUERY:-1}"
ALIGN_STRICT_BOTH="${ALIGN_STRICT_BOTH:-0}"
ICP_FITNESS_THRESHOLD="${ICP_FITNESS_THRESHOLD:-0.05}"
ICP_RMSE_THRESHOLD="${ICP_RMSE_THRESHOLD:-0.02}"
FUSION_ADMISSION_POLICY="${FUSION_ADMISSION_POLICY:-ho3d}"
CANDIDATE_LABELS="${CANDIDATE_LABELS:-rayst3r,instantmesh,fused}"
CANDIDATE_COLOR_MODE="${CANDIDATE_COLOR_MODE:-anchor_prior_hybrid}"
W_DEPTH="${W_DEPTH:-0.5}"
W_MASK="${W_MASK:-0.25}"
W_GEOM="${W_GEOM:-0.25}"
SCORE_ALPHA="${SCORE_ALPHA:-0.4}"
SCORE_BETA="${SCORE_BETA:-0.6}"
STRICT_ARGS=()
STRICT_VERIFY_ARGS=()
if [[ "$ALIGN_STRICT_BOTH" == "1" ]]; then
  STRICT_ARGS=(--align_strict_both)
  STRICT_VERIFY_ARGS=(
    --require-strict-both
    --strict-tau-f "$ICP_FITNESS_THRESHOLD"
    --strict-tau-r "$ICP_RMSE_THRESHOLD"
  )
fi

if [[ "$RUN_ANCHOR" != "0" ]]; then
  : "${ANCHOR_SOURCE:?Set ANCHOR_SOURCE to the prepared HO3D anchor-observation folder}"
  : "${RAYST3R_CKPT:?Set RAYST3R_CKPT to the local rayst3r.pth checkpoint}"
fi
if [[ "$RUN_QUERY" != "0" ]]; then
  : "${HO3D_ROOT:?Set HO3D_ROOT to the HO3D dataset root}"
  : "${MODELS_INFO:?Set MODELS_INFO to models_info.json}"
fi

mkdir -p results/logs "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

if [[ "$RUN_ANCHOR" != "0" ]]; then
  if [[ -e "$ANCHOR_OUT" ]]; then
    echo "Refusing to overwrite existing anchor output: $ANCHOR_OUT" >&2
    echo "Set ANCHOR_OUT to a fresh directory." >&2
    exit 2
  fi
  set +e
  CUDA_VISIBLE_DEVICES="$ANCHOR_GPUS" "$PYTHON_BIN" -u anchor_paper.py \
  --anchor_folder "$ANCHOR_SOURCE" \
  --output_root "$ANCHOR_OUT" \
  --ycb_model_path "$YCB_MODEL_PATH" \
  --depth_unit auto \
  --depth_preprocess \
  --depth_unit_try_both \
  --depth_filter_device cuda:0 \
  --rayst3r_device cuda:0 \
  --rayst3r_checkpoint "$RAYST3R_CKPT" \
  --instantmesh_device cuda:1 \
  --instantmesh_flip \
  --align_use_guess_translation \
  --align_bidirectional_icp \
  "${STRICT_ARGS[@]}" \
  --no-align_relaxed_rescue \
  --icp_max_iter 50 \
  --icp_fitness_threshold "$ICP_FITNESS_THRESHOLD" \
  --icp_rmse_threshold "$ICP_RMSE_THRESHOLD" \
  --icp_multi_hypo 24 \
  --icp_reciprocal_pairing \
  --icp_init_sigma 0.01 \
  --icp_max_corr_ratio 0.05 \
  --icp_voxel_size 0.005 \
  --sample_points 50000 \
  --rayst3r_set_conf 2.5 \
  --rayst3r_n_pred_views 5 \
  --rayst3r_filter_all_masks \
  --rayst3r_voxel_size 0.002 \
  --rayst3r_std_ratio 2.5 \
  --rayst3r_poisson_depth 9 \
  --rayst3r_poisson_scale 1.0 \
  --rayst3r_density_quantile 0.02 \
  --rayst3r_crop_margin_ratio 0.05 \
  --rayst3r_min_component_area_ratio 0.005 \
  --rayst3r_target_faces 100000 \
  --fusion_mode visible_patch \
  --fusion_voxel_size 0.003 \
  --fusion_sample_points 15000 \
  --fusion_poisson_depth 8 \
  --fusion_target_faces 120000 \
  --fusion_density_quantile 0.02 \
  --fusion_min_component_area_ratio 0.005 \
  --fusion_visible_depth_tolerance 0.015 \
  --fusion_observed_support_ratio 0.08 \
  --fusion_agreement_ratio 0.025 \
  --fusion_min_prior_coverage 0.35 \
  --fusion_min_observed_coverage 0.50 \
  --fusion_min_extent_ratio 0.65 \
  --fusion_max_extent_ratio 1.35 \
  --fusion_max_center_offset_ratio 0.15 \
  --no-fusion_cycle_in_reliability \
  --select_depth_weight 0.5 \
  --select_mask_weight 0.25 \
  --select_geom_weight 0.25 \
  --anchor_c_base 1.0 \
  --anchor_w_inlier 1.0 \
  --anchor_w_rmse 1.0 \
  --anchor_tau_inlier "$ICP_FITNESS_THRESHOLD" \
  --anchor_tau_rmse "$ICP_RMSE_THRESHOLD" \
  --fusion_admission_policy "$FUSION_ADMISSION_POLICY" \
  --anchor_depth_scale 0.05 \
  --anchor_mask_scale 0.50 \
  --anchor_geom_scale 0.05 \
  --anchor_calibration_weight 0.15 \
  --any6d_iter 5 \
  --any6d_refine 1 \
  --no-anchor_gt_symmetry_ablation \
  --score_alpha "$SCORE_ALPHA" \
  --score_beta "$SCORE_BETA" \
  --seed "$SEED" \
  2>&1 | tee "results/logs/anchor_${QUERY_NAME}.log"
  ANCHOR_PIPE_STATUS=("${PIPESTATUS[@]}")
  set -e
  ANCHOR_STATUS="${ANCHOR_PIPE_STATUS[0]}"
  TEE_STATUS="${ANCHOR_PIPE_STATUS[1]}"
  if [[ "$TEE_STATUS" != "0" ]]; then
    echo "Anchor log writer failed with status $TEE_STATUS" >&2
    exit "$TEE_STATUS"
  fi
  if [[ "$ANCHOR_STATUS" != "0" && "$ANCHOR_STATUS" != "139" ]]; then
    echo "Anchor process failed with status $ANCHOR_STATUS" >&2
    exit "$ANCHOR_STATUS"
  fi
  "$PYTHON_BIN" scripts/verify_anchor_registry.py \
    --anchor-output "$ANCHOR_OUT" --source-root "$ANCHOR_SOURCE" \
    "${STRICT_VERIFY_ARGS[@]}"
  if [[ "$ANCHOR_STATUS" == "139" ]]; then
    echo "WARNING: anchor process received SIGSEGV during native-library teardown; validated complete outputs and continuing." >&2
  fi
fi

if [[ "$RUN_QUERY" == "0" ]]; then
  exit 0
fi
if [[ ! -f "$ANCHOR_OUT/candidate_registry.json" ]]; then
  echo "Missing candidate registry: $ANCHOR_OUT/candidate_registry.json" >&2
  exit 2
fi

ANCHOR_RUN="$ANCHOR_OUT" \
SEED="$SEED" \
QUERY_GPU="$QUERY_GPU" \
QUERY_NAME="$QUERY_NAME" \
RUNNING_STRIDE="$RUNNING_STRIDE" \
CANDIDATE_LABELS="$CANDIDATE_LABELS" \
CANDIDATE_COLOR_MODE="$CANDIDATE_COLOR_MODE" \
W_DEPTH="$W_DEPTH" \
W_MASK="$W_MASK" \
W_GEOM="$W_GEOM" \
SCORE_ALPHA="$SCORE_ALPHA" \
SCORE_BETA="$SCORE_BETA" \
PYTHON_BIN="$PYTHON_BIN" \
HO3D_ROOT="$HO3D_ROOT" \
YCB_MODEL_PATH="$YCB_MODEL_PATH" \
MODELS_INFO="$MODELS_INFO" \
  bash scripts/run_ho3d_query.sh
