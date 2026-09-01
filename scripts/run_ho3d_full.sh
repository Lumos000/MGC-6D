#!/usr/bin/env bash
set -euo pipefail

# Reconstruct all HO3D anchors and then run the paper-aligned query protocol.
# Physical GPUs listed in ANCHOR_GPUS become cuda:0 and cuda:1 inside the
# anchor process. The query uses the single physical GPU in QUERY_GPU.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${ANCHOR_SOURCE:?Set ANCHOR_SOURCE to the prepared HO3D anchor-observation folder}"
: "${ANCHOR_OUT:?Set ANCHOR_OUT to a new anchor-result folder}"
: "${HO3D_ROOT:?Set HO3D_ROOT to the HO3D dataset root}"
: "${YCB_MODEL_PATH:?Set YCB_MODEL_PATH to the YCB model root}"
: "${MODELS_INFO:?Set MODELS_INFO to models_info.json}"
: "${RAYST3R_CKPT:?Set RAYST3R_CKPT to the local rayst3r.pth checkpoint}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ANCHOR_GPUS="${ANCHOR_GPUS:-0,1}"
QUERY_GPU="${QUERY_GPU:-0}"
QUERY_NAME="${QUERY_NAME:-ho3d_open_release_eq22_seed0}"
RUNNING_STRIDE="${RUNNING_STRIDE:-10}"

if [[ -e "$ANCHOR_OUT" ]] && [[ -n "$(find "$ANCHOR_OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ANCHOR_OUT already exists and is not empty: $ANCHOR_OUT" >&2
  echo "Choose a new folder, or use scripts/run_ho3d_query.sh with the existing registry." >&2
  exit 2
fi

mkdir -p "$ANCHOR_OUT" results/logs

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
  --anchor_tau_inlier 0.05 \
  --anchor_tau_rmse 0.02 \
  --anchor_depth_scale 0.05 \
  --anchor_mask_scale 0.50 \
  --anchor_geom_scale 0.05 \
  --anchor_calibration_weight 0.15 \
  --any6d_iter 5 \
  --any6d_refine 1 \
  --no-anchor_gt_symmetry_ablation \
  --score_alpha 0.4 \
  --score_beta 0.6 \
  --seed 0 \
  2>&1 | tee "results/logs/anchor_${QUERY_NAME}.log"

ANCHOR_RUN="$ANCHOR_OUT" \
QUERY_GPU="$QUERY_GPU" \
QUERY_NAME="$QUERY_NAME" \
RUNNING_STRIDE="$RUNNING_STRIDE" \
PYTHON_BIN="$PYTHON_BIN" \
HO3D_ROOT="$HO3D_ROOT" \
YCB_MODEL_PATH="$YCB_MODEL_PATH" \
MODELS_INFO="$MODELS_INFO" \
  bash scripts/run_ho3d_query.sh
