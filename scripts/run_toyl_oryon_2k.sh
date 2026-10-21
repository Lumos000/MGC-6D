#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mgc6d_toyl_oryon_matplotlib}"
export MPLCONFIGDIR
mkdir -p results/logs results/protocols/oryon "$MPLCONFIGDIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! PYTHON_BIN="$(command -v "$PYTHON_BIN" 2>/dev/null)"; then
  echo "Cannot find PYTHON_BIN in PATH." >&2
  exit 2
fi
PYTHON_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_FETCH="${RUN_FETCH:-1}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_ANCHOR="${RUN_ANCHOR:-1}"
RUN_QUERY="${RUN_QUERY:-1}"
ANCHOR_GPUS="${ANCHOR_GPUS:-0,1}"
QUERY_GPU="${QUERY_GPU:-0}"
PAIRS_FILE="${PAIRS_FILE:-$REPO_ROOT/results/protocols/oryon/toyl_cross_scene_test_instance_list.txt}"
ANCHOR_INPUT_ROOT="${ANCHOR_INPUT_ROOT:-$REPO_ROOT/results/oryon_toyl/anchor_inputs}"
PAIR_MANIFEST="${PAIR_MANIFEST:-$ANCHOR_INPUT_ROOT/oryon_toyl_manifest.json}"
ANCHOR_OUT="${ANCHOR_OUT:-$REPO_ROOT/results/anchor_results/toyl_oryon_2k_seed${SEED}}"
ANCHOR_PARTS_ROOT="${ANCHOR_PARTS_ROOT:-${ANCHOR_OUT}.parts}"
QUERY_NAME="${QUERY_NAME:-toyl_oryon_2k_seed${SEED}_${RUN_TAG}}"
QUERY_OUT="${QUERY_OUT:-$REPO_ROOT/results/toyl_oryon_results/$QUERY_NAME}"
CANDIDATE_LABELS="${CANDIDATE_LABELS:-rayst3r,instantmesh,fused}"
CANDIDATE_COLOR_MODE="${CANDIDATE_COLOR_MODE:-anchor_prior_hybrid}"
W_DEPTH="${W_DEPTH:-0.5}"
W_MASK="${W_MASK:-0.25}"
W_GEOM="${W_GEOM:-0.25}"
SCORE_ALPHA="${SCORE_ALPHA:-0.4}"
SCORE_BETA="${SCORE_BETA:-0.6}"
SAVE_CANDIDATE_METRICS="${SAVE_CANDIDATE_METRICS:-0}"
ALIGN_STRICT_BOTH="${ALIGN_STRICT_BOTH:-0}"

: "${TOYL_ROOT:?Set TOYL_ROOT to the TOYL dataset root}"
TOYL_MODELS="${TOYL_MODELS:-$TOYL_ROOT/models}"

if [[ "$RUN_FETCH" != "0" ]]; then
  if [[ -f "$PAIRS_FILE" ]]; then
    echo "Using existing pair list: $PAIRS_FILE"
  else
    "$PYTHON_BIN" scripts/fetch_oryon_toyl_pairs.py --output "$PAIRS_FILE"
  fi
fi

if [[ "$RUN_PREPARE" != "0" ]]; then
  if [[ ! -f "$PAIRS_FILE" ]]; then
    echo "Missing Oryon TOYL pair list: $PAIRS_FILE" >&2
    exit 2
  fi
  "$PYTHON_BIN" scripts/prepare_oryon_toyl.py \
    --toyl-root "$TOYL_ROOT" \
    --pairs-file "$PAIRS_FILE" \
    --output-root "$ANCHOR_INPUT_ROOT" \
    --resume
fi

if [[ ! -f "$PAIR_MANIFEST" ]]; then
  echo "Missing prepared pair manifest: $PAIR_MANIFEST" >&2
  exit 2
fi

STRICT_ARGS=()
if [[ "$ALIGN_STRICT_BOTH" == "1" ]]; then
  STRICT_ARGS=(--align_strict_both)
fi

if [[ "$RUN_ANCHOR" != "0" ]]; then
  : "${RAYST3R_CKPT:?Set RAYST3R_CKPT to rayst3r.pth}"
  if [[ -e "$ANCHOR_OUT" ]]; then
    echo "Refusing to overwrite existing anchor output: $ANCHOR_OUT" >&2
    echo "Set ANCHOR_OUT to a fresh directory." >&2
    exit 2
  fi
  mkdir -p "$ANCHOR_PARTS_ROOT"
  for OBJ_ID in $(seq 1 21); do
    OBJ_LABEL="$(printf 'obj_%02d' "$OBJ_ID")"
    PART_OUT="$ANCHOR_PARTS_ROOT/$OBJ_LABEL"
    if [[ -f "$PART_OUT/candidate_registry.json" ]]; then
      echo "Reusing completed anchor batch: $PART_OUT"
      continue
    fi
    set +e
    CUDA_VISIBLE_DEVICES="$ANCHOR_GPUS" "$PYTHON_BIN" -u anchor_paper_toyl_oryon.py \
    --anchor_folder "$ANCHOR_INPUT_ROOT" \
    --output_root "$PART_OUT" \
    --depth_unit auto \
    --depth_unit_try_both \
    --depth_preprocess \
    --depth_filter_radius 2 \
    --depth_filter_device cuda:0 \
    --rayst3r_device cuda:0 \
    --rayst3r_checkpoint "$RAYST3R_CKPT" \
    --instantmesh_device cuda:1 \
    --instantmesh_flip \
    --align_use_guess_translation \
    --align_bidirectional_icp \
    "${STRICT_ARGS[@]}" \
    --no-align_relaxed_rescue \
    --icp_reciprocal_pairing \
    --icp_max_iter 50 \
    --icp_fitness_threshold 0.05 \
    --icp_multi_hypo 24 \
    --icp_init_sigma 0.01 \
    --icp_max_corr_ratio 0.05 \
    --icp_voxel_size 0.005 \
    --sample_points 50000 \
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
    --rgbd_scale_visible_sim3 \
    --rgbd_scale_grid_steps 9 \
    --rgbd_scale_multiplier_min 0.65 \
    --rgbd_scale_multiplier_max 1.35 \
    --rgbd_scale_min_margin 0.002 \
    --rgbd_scale_sim3_iterations 5 \
    --rgbd_scale_max_points 5000 \
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
    --candidate_canonicalization_scale 1.0 \
    --score_alpha "$SCORE_ALPHA" \
    --score_beta "$SCORE_BETA" \
    --any6d_iter 5 \
    --any6d_refine 1 \
    --obj_ids "$OBJ_ID" \
    --seed "$SEED" \
    2>&1 | tee "results/logs/anchor_${QUERY_NAME}_${OBJ_LABEL}.log"
    ANCHOR_PIPE_STATUS=("${PIPESTATUS[@]}")
    set -e
    ANCHOR_STATUS="${ANCHOR_PIPE_STATUS[0]}"
    TEE_STATUS="${ANCHOR_PIPE_STATUS[1]}"
    if [[ "$TEE_STATUS" != "0" ]]; then
      exit "$TEE_STATUS"
    fi
    if [[ "$ANCHOR_STATUS" != "0" && "$ANCHOR_STATUS" != "139" ]]; then
      exit "$ANCHOR_STATUS"
    fi
    if [[ ! -f "$PART_OUT/candidate_registry.json" ]]; then
      echo "Anchor batch did not produce a registry: $PART_OUT" >&2
      exit 2
    fi
    if [[ "$ANCHOR_STATUS" == "139" ]]; then
      echo "WARNING: $OBJ_LABEL received SIGSEGV during native teardown; its registry exists and will be validated after merge." >&2
    fi
  done
  "$PYTHON_BIN" scripts/merge_oryon_toyl_anchor_parts.py \
    --parts-root "$ANCHOR_PARTS_ROOT" \
    --pair-manifest "$PAIR_MANIFEST" \
    --output-root "$ANCHOR_OUT"
  "$PYTHON_BIN" scripts/verify_oryon_toyl_anchor_registry.py \
    --anchor-output "$ANCHOR_OUT" \
    --pair-manifest "$PAIR_MANIFEST"
fi

if [[ "$RUN_QUERY" == "0" ]]; then
  exit 0
fi
if [[ ! -f "$ANCHOR_OUT/candidate_registry.json" ]]; then
  echo "Missing candidate registry: $ANCHOR_OUT/candidate_registry.json" >&2
  exit 2
fi
if [[ -e "$QUERY_OUT" ]]; then
  echo "Refusing to overwrite existing query output: $QUERY_OUT" >&2
  exit 2
fi

METRIC_ARGS=(--no-save_candidate_metrics)
if [[ "$SAVE_CANDIDATE_METRICS" == "1" ]]; then
  METRIC_ARGS=(--save_candidate_metrics)
fi
CUDA_VISIBLE_DEVICES="$QUERY_GPU" "$PYTHON_BIN" -u run_toyl_oryon_query.py \
  --name "$QUERY_NAME" \
  --anchor_path "$ANCHOR_OUT" \
  --anchor_input_root "$ANCHOR_INPUT_ROOT" \
  --pair_manifest "$PAIR_MANIFEST" \
  --toyl_root "$TOYL_ROOT" \
  --toyl_model_path "$TOYL_MODELS" \
  --output_root "$QUERY_OUT" \
  --device cuda:0 \
  --register_iteration 5 \
  --mask_type mask_visib \
  --anchor_color_depth_tolerance 0.03 \
  --candidate_labels "$CANDIDATE_LABELS" \
  --candidate_color_mode "$CANDIDATE_COLOR_MODE" \
  --w_depth "$W_DEPTH" \
  --w_mask "$W_MASK" \
  --w_geom "$W_GEOM" \
  --score_alpha "$SCORE_ALPHA" \
  --score_beta "$SCORE_BETA" \
  "${METRIC_ARGS[@]}" \
  --seed "$SEED" \
  2>&1 | tee "results/logs/query_${QUERY_NAME}.log"

"$PYTHON_BIN" scripts/verify_oryon_toyl_result.py "$QUERY_OUT"
