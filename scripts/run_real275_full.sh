#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mgc6d_real275_matplotlib}"
export MPLCONFIGDIR
mkdir -p results/logs "$MPLCONFIGDIR"

SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ANCHOR_OUT="${ANCHOR_OUT:-$REPO_ROOT/results/anchor_results/real275_paper_seed${SEED}_${RUN_TAG}}"
MODELS_INFO="${MODELS_INFO:-$REPO_ROOT/real275_models_info.json}"
QUERY_NAME="${QUERY_NAME:-real275_paper_eq9_seed${SEED}_${RUN_TAG}}"
ANCHOR_GPUS="${ANCHOR_GPUS:-0,1}"
QUERY_GPU="${QUERY_GPU:-1}"
RUNNING_STRIDE="${RUNNING_STRIDE:-10}"
RUN_ANCHOR="${RUN_ANCHOR:-1}"
RUN_QUERY="${RUN_QUERY:-1}"
OBJ_CLASSES="${OBJ_CLASSES:-}"
SCENES="${SCENES:-}"
CANDIDATE_LABELS="${CANDIDATE_LABELS:-rayst3r,instantmesh,fused}"
CANDIDATE_COLOR_MODE="${CANDIDATE_COLOR_MODE:-anchor_prior_hybrid}"
W_DEPTH="${W_DEPTH:-0.5}"
W_MASK="${W_MASK:-0.25}"
W_GEOM="${W_GEOM:-0.25}"
SCORE_ALPHA="${SCORE_ALPHA:-0.4}"
SCORE_BETA="${SCORE_BETA:-0.6}"
ALIGN_STRICT_BOTH="${ALIGN_STRICT_BOTH:-0}"

if [[ "$RUN_ANCHOR" != "0" ]]; then
  : "${ANCHOR_SOURCE:?Set ANCHOR_SOURCE to prepared REAL275 anchor observations}"
  : "${RAYST3R_CKPT:?Set RAYST3R_CKPT to rayst3r.pth}"
fi
if [[ "$RUN_QUERY" != "0" ]]; then
  : "${REAL_ROOT:?Set REAL_ROOT to the REAL275 dataset root}"
fi

ANCHOR_OBJECT_ARGS=()
QUERY_OBJECT_ARGS=()
QUERY_SCENE_ARGS=()
STRICT_ARGS=()
if [[ -n "$OBJ_CLASSES" ]]; then
  ANCHOR_OBJECT_ARGS=(--objects "$OBJ_CLASSES")
  QUERY_OBJECT_ARGS=(--obj_classes "$OBJ_CLASSES")
fi
if [[ -n "$SCENES" ]]; then
  QUERY_SCENE_ARGS=(--scenes "$SCENES")
fi
if [[ "$ALIGN_STRICT_BOTH" == "1" ]]; then
  STRICT_ARGS=(--align_strict_both)
fi

if [[ "$RUN_ANCHOR" != "0" ]]; then
  if [[ -e "$ANCHOR_OUT" ]]; then
    echo "Refusing to overwrite existing anchor output: $ANCHOR_OUT" >&2
    echo "Set ANCHOR_OUT to a fresh directory." >&2
    exit 2
  fi
  set +e
  CUDA_VISIBLE_DEVICES="$ANCHOR_GPUS" "$PYTHON_BIN" -u anchor_paper_real275.py \
  --anchor_folder "$ANCHOR_SOURCE" \
  --output_root "$ANCHOR_OUT" \
  --depth_unit auto \
  --depth_unit_try_both \
  --depth_preprocess \
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
  --icp_fitness_threshold 0.05 \
  --icp_multi_hypo 24 \
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
  --fusion_target_faces 120000 \
  --fusion_min_component_area_ratio 0.001 \
  --fusion_visible_depth_tolerance 0.015 \
  --fusion_observed_support_ratio 0.08 \
  --fusion_agreement_ratio 0.025 \
  --fusion_min_prior_coverage 0.35 \
  --fusion_min_observed_coverage 0.50 \
  --fusion_min_extent_ratio 0.65 \
  --fusion_max_extent_ratio 1.35 \
  --fusion_max_center_offset_ratio 0.15 \
  --fusion_max_update_ratio 0.04 \
  --fusion_min_normal_cosine 0.15 \
  --fusion_smooth_iterations 3 \
  --fusion_global_scale_min 0.88 \
  --fusion_global_scale_max 1.12 \
  --no-fusion_cycle_in_reliability \
  --rgbd_scale_recovery \
  --rgbd_scale_grid_steps 9 \
  --rgbd_scale_multiplier_min 0.65 \
  --rgbd_scale_multiplier_max 1.35 \
  --rgbd_scale_z_offset_steps 5 \
  --rgbd_scale_max_points 15000 \
  --rgbd_scale_depth_tolerance 0.03 \
  --rgbd_scale_support_ratio 0.08 \
  --select_depth_weight 0.5 \
  --select_mask_weight 0.25 \
  --select_geom_weight 0.25 \
  --anchor_c_base 1.0 \
  --anchor_w_inlier 1.0 \
  --anchor_w_rmse 1.0 \
  --anchor_tau_inlier 0.05 \
  --anchor_tau_rmse 0.02 \
  --anchor_depth_scale 0.05 \
  --anchor_mask_scale 0.50 \
  --anchor_geom_scale 0.05 \
  --anchor_calibration_weight 0.15 \
  --register_iteration 5 \
  --score_alpha "$SCORE_ALPHA" \
  --score_beta "$SCORE_BETA" \
  "${ANCHOR_OBJECT_ARGS[@]}" \
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
    --objects "$OBJ_CLASSES"
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

CUDA_VISIBLE_DEVICES="$QUERY_GPU" "$PYTHON_BIN" -u run_real275_query.py \
  --name "$QUERY_NAME" \
  --seed "$SEED" \
  --anchor_path "$ANCHOR_OUT" \
  --real275_data_root "$REAL_ROOT" \
  --models_info_path "$MODELS_INFO" \
  --running_stride "$RUNNING_STRIDE" \
  --register_iteration 5 \
  --mesh_diameter_mode bbox_diagonal \
  --candidate_labels "$CANDIDATE_LABELS" \
  --candidate_color_mode "$CANDIDATE_COLOR_MODE" \
  --w_depth "$W_DEPTH" \
  --w_mask "$W_MASK" \
  --w_geom "$W_GEOM" \
  --score_alpha "$SCORE_ALPHA" \
  --score_beta "$SCORE_BETA" \
  "${QUERY_OBJECT_ARGS[@]}" \
  "${QUERY_SCENE_ARGS[@]}" \
  2>&1 | tee "results/logs/query_${QUERY_NAME}.log"
