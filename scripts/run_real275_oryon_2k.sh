#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mgc6d_real275_oryon_matplotlib}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/mgc6d_real275_oryon_numba}"
export MPLCONFIGDIR
export NUMBA_CACHE_DIR
mkdir -p results/logs "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_FETCH="${RUN_FETCH:-1}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_ANCHOR="${RUN_ANCHOR:-1}"
RUN_QUERY="${RUN_QUERY:-1}"
RESUME_ANCHOR="${RESUME_ANCHOR:-1}"
ANCHOR_MAX_RESTARTS="${ANCHOR_MAX_RESTARTS:-12}"

: "${REAL_ROOT:?Set REAL_ROOT to the REAL275 root}"
if [[ "$RUN_ANCHOR" != "0" ]]; then
  : "${RAYST3R_CKPT:?Set RAYST3R_CKPT to rayst3r.pth}"
fi

PAIRS_FILE="${PAIRS_FILE:-$REPO_ROOT/results/protocols/oryon/real275_cross_scene_test_instance_list.txt}"
ANCHOR_INPUT_ROOT="${ANCHOR_INPUT_ROOT:-$REPO_ROOT/results/oryon_real275/anchor_inputs}"
PAIR_MANIFEST="${PAIR_MANIFEST:-$ANCHOR_INPUT_ROOT/oryon_real275_manifest.json}"
ANCHOR_OUT="${ANCHOR_OUT:-$REPO_ROOT/results/anchor_results/real275_oryon_2k_seed${SEED}}"
QUERY_OUT="${QUERY_OUT:-$REPO_ROOT/results/real275_oryon_results/real275_oryon_2k_seed${SEED}_${RUN_TAG}}"
MODELS_INFO="${MODELS_INFO:-$REPO_ROOT/real275_models_info.json}"
ANCHOR_GPUS="${ANCHOR_GPUS:-0,1}"
QUERY_GPU="${QUERY_GPU:-0}"
CANDIDATE_LABELS="${CANDIDATE_LABELS:-rayst3r,instantmesh,fused}"
CANDIDATE_COLOR_MODE="${CANDIDATE_COLOR_MODE:-anchor_prior_hybrid}"
W_DEPTH="${W_DEPTH:-0.5}"
W_MASK="${W_MASK:-0.25}"
W_GEOM="${W_GEOM:-0.25}"
SCORE_ALPHA="${SCORE_ALPHA:-0.4}"
SCORE_BETA="${SCORE_BETA:-0.6}"
SAVE_CANDIDATE_METRICS="${SAVE_CANDIDATE_METRICS:-0}"

if [[ "$RUN_FETCH" != "0" && ! -f "$PAIRS_FILE" ]]; then
  "$PYTHON_BIN" scripts/fetch_oryon_real275_pairs.py --output "$PAIRS_FILE"
fi
if [[ ! -f "$PAIRS_FILE" ]]; then
  echo "Missing official Oryon pair file: $PAIRS_FILE" >&2
  exit 2
fi

if [[ "$RUN_PREPARE" != "0" ]]; then
  "$PYTHON_BIN" scripts/prepare_oryon_real275.py \
    --real275-data-root "$REAL_ROOT" \
    --pairs-file "$PAIRS_FILE" \
    --output-root "$ANCHOR_INPUT_ROOT" \
    --resume
fi
if [[ ! -f "$PAIR_MANIFEST" ]]; then
  echo "Missing prepared pair manifest: $PAIR_MANIFEST" >&2
  exit 2
fi

if [[ "$RUN_ANCHOR" != "0" ]]; then
  if [[ -e "$ANCHOR_OUT" ]]; then
    if [[ ! -d "$ANCHOR_OUT" ]]; then
      echo "Anchor output exists but is not a directory: $ANCHOR_OUT" >&2
      exit 2
    fi
    if [[ "$RESUME_ANCHOR" == "0" ]]; then
      echo "Refusing to overwrite existing anchor output: $ANCHOR_OUT" >&2
      echo "Set RESUME_ANCHOR=1 to resume it, RUN_ANCHOR=0 to reuse a complete run, or choose a fresh ANCHOR_OUT." >&2
      exit 2
    fi
  fi
  RESUME_ARGS=()
  if [[ "$RESUME_ANCHOR" != "0" ]]; then
    RESUME_ARGS=(--resume_existing)
  fi
  if [[ ! "$ANCHOR_MAX_RESTARTS" =~ ^[0-9]+$ || "$ANCHOR_MAX_RESTARTS" -lt 1 ]]; then
    echo "ANCHOR_MAX_RESTARTS must be a positive integer" >&2
    exit 2
  fi
  ANCHOR_ATTEMPT=0
  while true; do
    ANCHOR_ATTEMPT=$((ANCHOR_ATTEMPT + 1))
    echo "Anchor attempt ${ANCHOR_ATTEMPT}/${ANCHOR_MAX_RESTARTS} (resume=${RESUME_ANCHOR})"
    set +e
    CUDA_VISIBLE_DEVICES="$ANCHOR_GPUS" "$PYTHON_BIN" -u anchor_paper_real275.py \
    --anchor_folder "$ANCHOR_INPUT_ROOT" \
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
    --no-align_relaxed_rescue \
    --icp_max_iter 50 \
    --icp_fitness_threshold 0.05 \
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
    --seed "$SEED" \
      "${RESUME_ARGS[@]}" \
      2>&1 | tee -a "results/logs/anchor_real275_oryon_2k_seed${SEED}_${RUN_TAG}.log"
    ANCHOR_PIPE_STATUS=("${PIPESTATUS[@]}")
    set -e
    ANCHOR_STATUS="${ANCHOR_PIPE_STATUS[0]}"
    TEE_STATUS="${ANCHOR_PIPE_STATUS[1]}"
    if [[ "$TEE_STATUS" != "0" ]]; then
      exit "$TEE_STATUS"
    fi
    if [[ "$ANCHOR_STATUS" == "0" ]]; then
      break
    fi
    if [[ "$ANCHOR_STATUS" != "139" ]]; then
      exit "$ANCHOR_STATUS"
    fi
    REQUIRED_FINAL=(
      "$ANCHOR_OUT/candidate_registry.json"
      "$ANCHOR_OUT/anchor_run_meta.json"
      "$ANCHOR_OUT/chamfer_distances.xlsx"
      "$ANCHOR_OUT/pose_metrics.xlsx"
    )
    FINAL_OUTPUT_COMPLETE=1
    for REQUIRED_PATH in "${REQUIRED_FINAL[@]}"; do
      if [[ ! -s "$REQUIRED_PATH" ]]; then
        FINAL_OUTPUT_COMPLETE=0
        break
      fi
    done
    if [[ "$FINAL_OUTPUT_COMPLETE" == "1" ]]; then
      break
    fi
    echo "Anchor process crashed before completion (SIGSEGV/139)." >&2
    echo "Missing final artifact: $REQUIRED_PATH" >&2
    if [[ "$RESUME_ANCHOR" == "0" || "$ANCHOR_ATTEMPT" -ge "$ANCHOR_MAX_RESTARTS" ]]; then
      echo "Automatic resume is disabled or exhausted. Rerun with RESUME_ANCHOR=1 after inspecting the last log entry." >&2
      exit 139
    fi
    echo "Restarting from atomic per-reference checkpoints..." >&2
  done
  "$PYTHON_BIN" scripts/verify_oryon_anchor_registry.py \
    --anchor-output "$ANCHOR_OUT" \
    --pair-manifest "$PAIR_MANIFEST"
  if [[ "$ANCHOR_STATUS" == "139" ]]; then
    echo "WARNING: native teardown SIGSEGV occurred after complete outputs; continuing." >&2
  fi
fi

if [[ "$RUN_QUERY" == "0" ]]; then
  exit 0
fi
if [[ ! -f "$ANCHOR_OUT/candidate_registry.json" ]]; then
  echo "Missing anchor registry: $ANCHOR_OUT/candidate_registry.json" >&2
  exit 2
fi
if [[ -e "$QUERY_OUT" ]]; then
  echo "Refusing to overwrite query output: $QUERY_OUT" >&2
  exit 2
fi

METRIC_ARGS=(--no-save_candidate_metrics)
if [[ "$SAVE_CANDIDATE_METRICS" == "1" ]]; then
  METRIC_ARGS=(--save_candidate_metrics)
fi
CUDA_VISIBLE_DEVICES="$QUERY_GPU" "$PYTHON_BIN" -u run_real275_oryon_query.py \
  --name "real275_oryon_2k_seed${SEED}" \
  --seed "$SEED" \
  --anchor_path "$ANCHOR_OUT" \
  --anchor_input_root "$ANCHOR_INPUT_ROOT" \
  --pair_manifest "$PAIR_MANIFEST" \
  --real275_data_root "$REAL_ROOT" \
  --models_info_path "$MODELS_INFO" \
  --output_root "$QUERY_OUT" \
  --register_iteration 5 \
  --mesh_diameter_mode bbox_diagonal \
  --candidate_labels "$CANDIDATE_LABELS" \
  --candidate_color_mode "$CANDIDATE_COLOR_MODE" \
  --w_depth "$W_DEPTH" \
  --w_mask "$W_MASK" \
  --w_geom "$W_GEOM" \
  --score_alpha "$SCORE_ALPHA" \
  --score_beta "$SCORE_BETA" \
  "${METRIC_ARGS[@]}" \
  2>&1 | tee "results/logs/query_real275_oryon_2k_seed${SEED}_${RUN_TAG}.log"

"$PYTHON_BIN" scripts/verify_oryon_real275_result.py "$QUERY_OUT"
