# Experiments

## Anchor Reconstruction

This step reconstructs candidate geometries from the anchor observations,
calibrates candidate scores, and writes `candidate_registry.json` into the
anchor result folder.

```bash
export RAYST3R_ROOT=/path/to/rayst3r
export SAM2_CKPT=/path/to/sam2.1_hiera_large.pt

python anchor_paper.py \
  --anchor_folder /path/to/anchor_results/dexycb_reference_view_ours \
  --ycb_model_path /path/to/ho3d/YCB_Video_Models \
  --depth_preprocess \
  --depth_unit_try_both \
  --refine_mask \
  --align_use_guess_translation \
  --align_bidirectional_icp \
  --rayst3r_set_conf 2.5 \
  --rayst3r_n_pred_views 5 \
  --rayst3r_filter_all_masks \
  --rayst3r_device cuda:0 \
  --instantmesh_device cuda:1 \
  --any6d_iter 5 \
  --any6d_refine 1 \
  --score_alpha 0.3 \
  --score_beta 0.7 \
  --seed 0
```

## Query Evaluation

```bash
python query_paper.py \
  --name ho3d_qeff_run1 \
  --anchor_path /path/to/anchor_results/dexycb_reference_view_ours \
  --hot3d_data_root /path/to/ho3d \
  --ycb_model_path /path/to/ho3d/YCB_Video_Models \
  --ycbv_modesl_info_path ./models_info.json \
  --running_stride 10 \
  --register_iteration 5 \
  --score_alpha 0.3 \
  --score_beta 0.7 \
  --per_frame_selection \
  --use_bbox_diameter \
  --lazy_tensors
```

