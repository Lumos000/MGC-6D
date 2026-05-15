
from project_paths import setup_project_paths
setup_project_paths()
import argparse
import glob
import json
import os
import random
import warnings

import cv2
import nvdiffrast.torch as dr
import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

from construction import ReconstructionConfig, run_construction
from estimater import Any6D
from foundationpose.Utils import (
    visualize_frame_results,
    visualize_frame_results_gt,
    align_mesh_to_coordinate,
)
from sam2_rayst3r import running_sam_box


OBJ_NUM_MAP = {
    "003_cracker_box": 2,
    "006_mustard_bottle": 5,
    "021_bleach_cleanser": 12,
    "019_pitcher_base": 11,
    "004_sugar_box": 3,
    "005_tomato_soup_can": 4,
    "010_potted_meat_can": 9,
}


def _find_file(search_dirs, preferred_names, suffix=None):
    for d in search_dirs:
        for name in preferred_names:
            candidate = os.path.join(d, name)
            if os.path.exists(candidate):
                return candidate
        if suffix:
            matches = glob.glob(os.path.join(d, f"*{suffix}"))
            if matches:
                return sorted(matches)[0]
    return None


def _load_intrinsics(search_dirs):
    intrinsics_pt = _find_file(search_dirs, ["intrinsics.pt"])
    if intrinsics_pt:
        try:
            intrinsic = torch.load(intrinsics_pt, map_location="cpu", weights_only=True)
        except TypeError:
            intrinsic = torch.load(intrinsics_pt, map_location="cpu")
        if isinstance(intrinsic, torch.Tensor):
            return intrinsic.numpy()
        return np.array(intrinsic)
    intrinsics_txt = _find_file(search_dirs, ["K.txt"])
    if intrinsics_txt:
        return np.loadtxt(intrinsics_txt)
    raise FileNotFoundError(f"Intrinsics file not found in {search_dirs}")


def _detect_depth_unit(depth_uint16):
    max_val = float(np.nanmax(depth_uint16))
    if max_val > 20000:
        return "rayst3r_uint16"
    if max_val > 200:
        return "mm"
    return "meters"


def _convert_depth(depth_raw, depth_unit):
    if depth_unit == "mm":
        return depth_raw / 1000.0
    if depth_unit == "rayst3r_uint16":
        return depth_raw / 65535.0 * 10.0
    if depth_unit == "meters":
        return depth_raw.astype(np.float32)
    raise ValueError(f"Unsupported depth_unit: {depth_unit}")


def _load_eval_inputs(obj_dir, depth_unit):
    search_dirs = [obj_dir, os.path.join(obj_dir, "anchor_init")]
    color_path = _find_file(search_dirs, ["color.png"], suffix="_color.png")
    depth_path = _find_file(search_dirs, ["depth.png"], suffix="_depth.png")
    mask_path = _find_file(search_dirs, ["mask.png"], suffix="_mask.png")

    if not color_path or not depth_path:
        raise FileNotFoundError(f"Missing color/depth in {obj_dir}")

    color = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if depth_unit == "auto":
        depth_unit = _detect_depth_unit(depth_raw)
    depth = _convert_depth(depth_raw, depth_unit)

    if mask_path and os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask_img > 0).astype(np.bool_)
    else:
        h, w = color.shape[:2]
        box = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
        mask = running_sam_box(color, box)

    intrinsic = _load_intrinsics(search_dirs)
    return color, depth, mask, intrinsic, depth_unit


def _load_model_info(ycb_model_path: str) -> dict:
    candidates = [
        os.path.join(ycb_model_path, "models_info.json"),
        os.path.join(ycb_model_path, "models", "models_info.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _build_symmetry_tfs_for_object(obj_name: str, model_info: dict) -> np.ndarray | None:
    obj_id = OBJ_NUM_MAP.get(obj_name, None)
    if obj_id is None:
        return None
    info = model_info.get(str(obj_id), {})
    syms = info.get("symmetries_discrete", [])
    if not syms:
        return None
    all_tfs = [np.eye(4, dtype=np.float32)]
    for sym in syms:
        all_tfs.append(np.reshape(np.asarray(sym, dtype=np.float32), (4, 4)))
    return np.stack(all_tfs, axis=0)


def _backproject_depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    valid_mask: np.ndarray,
    max_points: int = 5000,
) -> np.ndarray:
    ys, xs = np.where(valid_mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float32) - cx) * z / max(fx, 1e-8)
    y = (ys.astype(np.float32) - cy) * z / max(fy, 1e-8)
    pts = np.stack([x, y, z], axis=1)

    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def _chamfer_distance_pointcloud(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return 1e3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.as_tensor(pts_a, dtype=torch.float32, device=device)
    b = torch.as_tensor(pts_b, dtype=torch.float32, device=device)
    dists = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2.0).squeeze(0)
    ch = dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
    return float(ch.detach().cpu().item())


def _compute_self_supervised_losses(
    est,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    pred_pose: np.ndarray,
    glctx,
) -> tuple[float, float, float]:
    """Observation consistency scoring (Eq. 14).

    Returns (L_depth, L_mask, L_geom) measuring how well the candidate
    at its predicted pose explains the observation.
    """
    from foundationpose.Utils import nvdiffrast_render

    h, w = depth.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_tensor = torch.as_tensor(pred_pose, device=device, dtype=torch.float32).unsqueeze(0)
    tf = est.get_tf_to_centered_mesh()
    if torch.is_tensor(tf):
        tf = tf.to(device=pose_tensor.device, dtype=torch.float32).unsqueeze(0)
    else:
        tf = torch.as_tensor(tf, device=pose_tensor.device, dtype=torch.float32).unsqueeze(0)
    ob_in_cams = pose_tensor @ torch.linalg.inv(tf)

    rendered = nvdiffrast_render(
        mesh=est.mesh,
        mesh_tensors=est.mesh_tensors,
        # Utils.nvdiffrast_render expects torch tensor for matmul with glcam tf.
        ob_in_cams=ob_in_cams,
        K=K,
        H=h,
        W=w,
        glctx=glctx,
    )
    if rendered is None:
        return 1.0, 1.0, 1e3

    rendered_depth = None
    if isinstance(rendered, tuple):
        # foundationpose.Utils.nvdiffrast_render returns (color, depth, normal_map)
        if len(rendered) >= 2:
            rendered_depth = rendered[1]
    elif hasattr(rendered, "shape") and rendered.shape[-1] > 3:
        # Backward-compatible path for packed outputs.
        rendered_depth = rendered[..., 3]
    if rendered_depth is None:
        return 1.0, 1.0, 1e3
    rd = rendered_depth[0] if getattr(rendered_depth, "ndim", 0) > 2 else rendered_depth
    if torch.is_tensor(rd):
        rd = rd.detach().cpu().numpy()
    rd = rd.astype(np.float32)

    obs_mask = (mask > 0) & (depth > 1e-4)
    ren_mask = rd > 1e-4

    overlap = obs_mask & ren_mask
    if overlap.sum() > 10:
        depth_scale = max(float(np.mean(depth[overlap])), 1e-4)
        l_depth = float(np.mean(np.abs(depth[overlap] - rd[overlap])) / depth_scale)
    else:
        l_depth = 1.0

    union = obs_mask | ren_mask
    if union.sum() > 0:
        iou = float((obs_mask & ren_mask).sum()) / float(union.sum())
        l_mask = float(1.0 - iou)
    else:
        l_mask = 1.0

    obs_pts = _backproject_depth_to_points(depth, K, obs_mask)
    ren_pts = _backproject_depth_to_points(rd, K, ren_mask)
    l_geom = _chamfer_distance_pointcloud(ren_pts, obs_pts)
    return l_depth, l_mask, l_geom


def _compute_anchor_calibration_score(
    meta: dict,
    label: str,
    c_base: float,
    w_inlier: float,
    w_rmse: float,
    tau_inlier: float,
    tau_rmse: float,
) -> float:
    """Compute anchor-phase geometric credibility score s_anchor (Eq. 9, 10).

    For base candidates (obs/prior): s_anchor = c_base.
    For fused candidate satisfying consistency: s_anchor = c_base + w_inlier*fitness + w_rmse*(1-rmse).
    """
    if label in ("rayst3r", "instantmesh"):
        return c_base

    fitness = meta.get("align_fitness", -1.0)
    rmse = meta.get("align_rmse", float("inf"))
    consistency_ok = (fitness >= tau_inlier) and (rmse <= tau_rmse)

    if label == "fused" and consistency_ok:
        return c_base + w_inlier * fitness + w_rmse * max(0.0, 1.0 - rmse)
    return c_base


def main():
    parser = argparse.ArgumentParser(description="Anchor reconstruction + Any6D evaluation with fused meshes")
    parser.add_argument(
        "--anchor_folder",
        type=str,
        required=True,
        help="Path to anchor results folder",
    )
    parser.add_argument(
        "--ycb_model_path",
        type=str,
        required=True,
        help="Path to YCB Video Models",
    )
    parser.add_argument(
        "--depth_unit",
        type=str,
        default="auto",
        choices=["auto", "mm", "rayst3r_uint16", "meters"],
        help="Depth unit conversion for evaluation",
    )
    parser.add_argument(
        "--depth_preprocess",
        action="store_true",
        help="Apply depth preprocessing (erode + bilateral)",
    )
    parser.add_argument(
        "--depth_unit_try_both",
        action="store_true",
        help="Try mm and rayst3r_uint16 and pick best",
    )
    parser.add_argument(
        "--depth_filter_radius",
        type=int,
        default=2,
        help="Radius for depth preprocessing filters",
    )
    parser.add_argument(
        "--depth_filter_device",
        type=str,
        default="cuda",
        help="Device for depth preprocessing filters (warp)",
    )
    parser.add_argument(
        "--valid_ratio_threshold",
        type=float,
        default=0.01,
        help="Valid depth ratio threshold for skipping ICP/fusion",
    )
    parser.add_argument(
        "--depth_range_min_m",
        type=float,
        default=0.2,
        help="Physical depth range min (meters) for unit selection",
    )
    parser.add_argument(
        "--depth_range_max_m",
        type=float,
        default=3.0,
        help="Physical depth range max (meters) for unit selection",
    )
    parser.add_argument(
        "--depth_range_min_ratio",
        type=float,
        default=0.3,
        help="Min in-range ratio to accept a depth unit candidate",
    )
    parser.add_argument(
        "--depth_range_soft_penalty",
        action="store_true",
        help="Penalize out-of-range candidates instead of rejecting",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional output root for reconstruction (defaults to anchor_folder)",
    )
    parser.add_argument(
        "--rayst3r_device",
        type=str,
        default="cuda:0",
        help="Device for RaySt3R (e.g., cuda:0)",
    )
    parser.add_argument(
        "--rayst3r_checkpoint",
        type=str,
        default=None,
        help="Local RaySt3R checkpoint path (.pth). If set, skip HuggingFace download.",
    )
    parser.add_argument(
        "--instantmesh_device",
        type=str,
        default="cuda:1",
        help="Device for InstantMesh (e.g., cuda:1)",
    )
    parser.add_argument(
        "--align_use_guess_translation",
        action="store_true",
        help="Use depth-based translation guess before ICP",
    )
    parser.add_argument(
        "--align_bidirectional_icp",
        action="store_true",
        help="Run bidirectional ICP and choose best alignment",
    )
    parser.add_argument(
        "--icp_max_iter",
        type=int,
        default=50,
        help="Max iterations for ICP",
    )
    parser.add_argument(
        "--icp_fitness_threshold",
        type=float,
        default=0.05,
        help="Minimum ICP fitness to accept alignment",
    )
    parser.add_argument(
        "--icp_multi_hypo",
        type=int,
        default=1,
        help="Number of ICP initial hypotheses",
    )
    parser.add_argument(
        "--icp_init_sigma",
        type=float,
        default=0.01,
        help="Translation sigma for ICP hypotheses (meters)",
    )
    parser.add_argument(
        "--mask_stability_enabled",
        action="store_true",
        help="Enable mask stability filtering",
    )
    parser.add_argument(
        "--mask_stability_threshold",
        type=float,
        default=0.6,
        help="Mask stability threshold",
    )
    parser.add_argument(
        "--mask_iou_threshold",
        type=float,
        default=0.5,
        help="Mask IoU threshold with original mask",
    )
    parser.add_argument(
        "--refine_mask",
        action="store_true",
        help="Refine mask with SAM2 before reconstruction",
    )
    parser.add_argument(
        "--rayst3r_set_conf",
        type=float,
        default=2.5,
        help="RaySt3R confidence threshold",
    )
    parser.add_argument(
        "--rayst3r_n_pred_views",
        type=int,
        default=5,
        help="RaySt3R predicted views",
    )
    parser.add_argument(
        "--rayst3r_filter_all_masks",
        action="store_true",
        help="Filter all masks in RaySt3R",
    )
    parser.add_argument(
        "--rayst3r_tsdf",
        action="store_true",
        help="Use TSDF fusion in RaySt3R",
    )
    parser.add_argument(
        "--rayst3r_voxel_size",
        type=float,
        default=0.002,
        help="Voxel size for RaySt3R mesh",
    )
    parser.add_argument(
        "--rayst3r_std_ratio",
        type=float,
        default=2.5,
        help="Std ratio for outlier removal",
    )
    parser.add_argument(
        "--rayst3r_poisson_depth",
        type=int,
        default=9,
        help="Poisson depth for RaySt3R mesh",
    )
    parser.add_argument(
        "--rayst3r_poisson_scale",
        type=float,
        default=1.0,
        help="Poisson scale for RaySt3R mesh",
    )
    parser.add_argument(
        "--rayst3r_density_quantile",
        type=float,
        default=0.02,
        help="Low-density removal quantile",
    )
    parser.add_argument(
        "--instantmesh_flip",
        action="store_true",
        help="Flip input image horizontally for InstantMesh",
    )
    parser.add_argument(
        "--instantmesh_no_remove_bg",
        action="store_true",
        help="Disable background removal for InstantMesh",
    )
    parser.add_argument(
        "--icp_max_corr_ratio",
        type=float,
        default=0.05,
        help="ICP max correspondence ratio of RaySt3R bbox diag",
    )
    parser.add_argument(
        "--icp_voxel_size",
        type=float,
        default=0.005,
        help="Voxel size for ICP downsampling",
    )
    parser.add_argument(
        "--sample_points",
        type=int,
        default=50000,
        help="Number of points to sample from InstantMesh for ICP",
    )
    # --- Observation consistency weights (Eq. 14): s_query ---
    parser.add_argument(
        "--select_depth_weight",
        type=float,
        default=0.5,
        help="w1: weight for L_depth in observation consistency (Eq. 14)",
    )
    parser.add_argument(
        "--select_mask_weight",
        type=float,
        default=0.25,
        help="w2: weight for L_mask in observation consistency (Eq. 14)",
    )
    parser.add_argument(
        "--select_geom_weight",
        type=float,
        default=0.25,
        help="w3: weight for L_geom (Chamfer) in observation consistency (Eq. 14)",
    )
    parser.add_argument(
        "--select_chamfer_force_ratio",
        type=float,
        default=0.2,
        help="If selected L_geom > best*(1+ratio), force best-L_geom candidate",
    )
    # --- Anchor calibration score parameters (Eq. 9, 10) ---
    parser.add_argument(
        "--anchor_c_base",
        type=float,
        default=1.0,
        help="c_base: constant baseline credibility for base candidates (Eq. 9)",
    )
    parser.add_argument(
        "--anchor_w_inlier",
        type=float,
        default=1.0,
        help="w_inlier: weight for ICP inlier ratio in fused candidate score (Eq. 10)",
    )
    parser.add_argument(
        "--anchor_w_rmse",
        type=float,
        default=1.0,
        help="w_rmse: weight for (1-RMSE) in fused candidate score (Eq. 10)",
    )
    parser.add_argument(
        "--anchor_tau_inlier",
        type=float,
        default=0.05,
        help="tau_inlier: minimum ICP coverage ratio for consistency (Eq. 7)",
    )
    parser.add_argument(
        "--anchor_tau_rmse",
        type=float,
        default=0.02,
        help="tau_rmse: maximum allowed RMSE for consistency (Eq. 7)",
    )
    # --- Total score balance (Eq. 2): S(k) = alpha*s_anchor + beta*s_query ---
    parser.add_argument(
        "--score_alpha",
        type=float,
        default=0.3,
        help="alpha: weight for anchor calibration score in total score (Eq. 2)",
    )
    parser.add_argument(
        "--score_beta",
        type=float,
        default=0.7,
        help="beta: weight for query observation consistency in total score (Eq. 2)",
    )
    # Legacy selection weights kept for CLI backward compatibility.
    parser.add_argument("--select_chamfer_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--select_adds_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--select_add_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--select_r_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--select_t_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--any6d_iter",
        type=int,
        default=5,
        help="Any6D refinement iterations (register_any6d)",
    )
    parser.add_argument(
        "--any6d_refine",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable Any6D refinement stage (1=on, 0=off)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed for reproducible anchor reconstruction/evaluation",
    )
    args = parser.parse_args()
    if any(
        v is not None
        for v in [
            args.select_chamfer_weight,
            args.select_adds_weight,
            args.select_add_weight,
            args.select_r_weight,
            args.select_t_weight,
        ]
    ):
        if args.select_chamfer_weight is not None:
            args.select_geom_weight = float(args.select_chamfer_weight)
        if args.select_adds_weight is not None:
            args.select_mask_weight = float(args.select_adds_weight)
        legacy_depth = [
            v for v in [args.select_add_weight, args.select_r_weight, args.select_t_weight] if v is not None
        ]
        if legacy_depth:
            args.select_depth_weight = float(sum(legacy_depth))
        print(
            "Warning: legacy selection weights detected. "
            "Mapped to self-supervised weights (geom<-chamfer, mask<-adds, depth<-add+r+t)."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = ReconstructionConfig(
        anchor_root=args.anchor_folder,
        output_root=args.output_root,
        depth_unit=args.depth_unit,
        depth_unit_try_both=args.depth_unit_try_both,
        depth_preprocess=args.depth_preprocess,
        depth_filter_radius=args.depth_filter_radius,
        depth_filter_device=args.depth_filter_device,
        valid_ratio_threshold=args.valid_ratio_threshold,
        depth_range_min_m=args.depth_range_min_m,
        depth_range_max_m=args.depth_range_max_m,
        depth_range_min_ratio=args.depth_range_min_ratio,
        depth_range_soft_penalty=args.depth_range_soft_penalty,
        rayst3r_device=args.rayst3r_device,
        rayst3r_checkpoint=args.rayst3r_checkpoint,
        refine_mask=args.refine_mask,
        mask_stability_enabled=args.mask_stability_enabled,
        mask_stability_threshold=args.mask_stability_threshold,
        mask_iou_threshold=args.mask_iou_threshold,
        rayst3r_set_conf=args.rayst3r_set_conf,
        rayst3r_n_pred_views=args.rayst3r_n_pred_views,
        rayst3r_filter_all_masks=args.rayst3r_filter_all_masks,
        rayst3r_tsdf=args.rayst3r_tsdf,
        rayst3r_voxel_size=args.rayst3r_voxel_size,
        rayst3r_std_ratio=args.rayst3r_std_ratio,
        rayst3r_poisson_depth=args.rayst3r_poisson_depth,
        rayst3r_poisson_scale=args.rayst3r_poisson_scale,
        rayst3r_density_quantile=args.rayst3r_density_quantile,
        instantmesh_device=args.instantmesh_device,
        instantmesh_flip=args.instantmesh_flip,
        instantmesh_remove_bg=not args.instantmesh_no_remove_bg,
        align_use_guess_translation=args.align_use_guess_translation,
        align_bidirectional_icp=args.align_bidirectional_icp,
        icp_max_iter=args.icp_max_iter,
        icp_fitness_threshold=args.icp_fitness_threshold,
        icp_multi_hypo=args.icp_multi_hypo,
        icp_init_sigma=args.icp_init_sigma,
        icp_max_corr_ratio=args.icp_max_corr_ratio,
        icp_voxel_size=args.icp_voxel_size,
        sample_points=args.sample_points,
    )

    # ====================================================================
    # Step 1: Anchor-phase candidate geometry construction (Section 3.2)
    # ====================================================================
    recon_results = run_construction(cfg)
    model_info = _load_model_info(args.ycb_model_path)

    run_meta = {
        "anchor_folder": args.anchor_folder,
        "output_root": args.output_root,
        "ycb_model_path": args.ycb_model_path,
        "selection": {
            "select_depth_weight": args.select_depth_weight,
            "select_mask_weight": args.select_mask_weight,
            "select_geom_weight": args.select_geom_weight,
            "select_chamfer_force_ratio": args.select_chamfer_force_ratio,
        },
        "calibration": {
            "anchor_c_base": args.anchor_c_base,
            "anchor_w_inlier": args.anchor_w_inlier,
            "anchor_w_rmse": args.anchor_w_rmse,
            "anchor_tau_inlier": args.anchor_tau_inlier,
            "anchor_tau_rmse": args.anchor_tau_rmse,
            "score_alpha": args.score_alpha,
            "score_beta": args.score_beta,
        },
        "args": vars(args),
    }
    with open(os.path.join(args.anchor_folder, "anchor_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    results = []
    pose_rows = []
    pose_rows_all = []
    candidate_registry = {}
    obj_list = [d for d in os.listdir(args.anchor_folder) if os.path.isdir(os.path.join(args.anchor_folder, d))]

    glctx = dr.RasterizeCudaContext()

    for obj in tqdm(obj_list, desc="Object"):
        obj_num = OBJ_NUM_MAP.get(obj, -1)
        symmetry_tfs = _build_symmetry_tfs_for_object(obj, model_info)
        obj_dir = os.path.join(args.anchor_folder, obj)

        if obj not in recon_results:
            raise RuntimeError(f"Missing reconstruction outputs for {obj}")

        # ====================================================================
        # Step 2: Bidirectional geometric calibration (Section 3.3.1)
        # Load ICP meta to compute s_anchor for each candidate (Eq. 5-10)
        # ====================================================================
        meta_path = recon_results[obj].get("meta")
        recon_meta = {}
        align_direction = None
        chosen_depth_unit = args.depth_unit
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                recon_meta = json.load(f)
            align_direction = recon_meta.get("align_direction")
            chosen_depth_unit = recon_meta.get("depth_unit", args.depth_unit)

        color, depth, mask, intrinsic, depth_unit = _load_eval_inputs(obj_dir, chosen_depth_unit)

        ray_path = recon_results[obj]["mesh_rayst3r"]
        inst_path = recon_results[obj]["mesh_instant"]
        fused_path = recon_results[obj]["mesh_fused"]

        candidates = {}
        if ray_path and os.path.exists(ray_path):
            candidates["rayst3r"] = ray_path
        if inst_path and os.path.exists(inst_path):
            candidates["instantmesh"] = inst_path
        if fused_path and os.path.exists(fused_path):
            fused_consistency_ok = (
                recon_meta.get("align_fitness", -1.0) >= args.anchor_tau_inlier
                and recon_meta.get("align_rmse", float("inf")) <= args.anchor_tau_rmse
            )
            if fused_consistency_ok:
                candidates["fused"] = fused_path

        s_anchor_scores = {}
        for label in candidates:
            s_anchor_scores[label] = _compute_anchor_calibration_score(
                meta=recon_meta,
                label=label,
                c_base=args.anchor_c_base,
                w_inlier=args.anchor_w_inlier,
                w_rmse=args.anchor_w_rmse,
                tau_inlier=args.anchor_tau_inlier,
                tau_rmse=args.anchor_tau_rmse,
            )

        # ====================================================================
        # Step 3: For each candidate, run Any6D pose registration on anchor
        # image and compute observation consistency score (Eq. 14)
        # ====================================================================
        def _evaluate_mesh_pose(mesh_path: str, label: str, visualize: bool = False):
            mesh_local = trimesh.load(mesh_path)
            mesh_local = align_mesh_to_coordinate(mesh_local)
            mesh_local.export(os.path.join(obj_dir, f"center_mesh_{obj}_{label}.obj"))
            est_local = Any6D(symmetry_tfs=symmetry_tfs, mesh=mesh_local, debug_dir=obj_dir, debug=0)
            pred_pose_local = est_local.register_any6d(
                K=intrinsic,
                rgb=color,
                depth=depth,
                ob_mask=mask,
                iteration=args.any6d_iter,
                refinement=bool(args.any6d_refine),
                name="demo",
            )
            l_depth, l_mask, l_geom = _compute_self_supervised_losses(
                est=est_local,
                depth=depth,
                mask=mask,
                K=intrinsic,
                pred_pose=pred_pose_local,
                glctx=glctx,
            )
            np.savetxt(os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt"), pred_pose_local)
            est_local.mesh.export(os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj"))
            np.savetxt(os.path.join(obj_dir, f"{obj}_ldepth_{label}.txt"), [l_depth])
            np.savetxt(os.path.join(obj_dir, f"{obj}_lmask_{label}.txt"), [l_mask])
            np.savetxt(os.path.join(obj_dir, f"{obj}_cd_{label}.txt"), [l_geom])
            if visualize:
                try:
                    gt_pose = np.loadtxt(os.path.join(obj_dir, f"{obj}_gt_pose.txt"))
                    gt_mesh = trimesh.load(f"{args.ycb_model_path}/models/{obj}/textured_simple.obj")
                    visualize_frame_results(
                        color=color,
                        gt_mesh=gt_mesh,
                        est=est_local,
                        K=intrinsic,
                        gt_pose=gt_pose,
                        pred_pose=pred_pose_local,
                        metric=None,
                        obj_f=obj,
                        frame_idx=0,
                        save_path=obj_dir,
                        glctx=glctx,
                        name=f"demo_data_{label}",
                        mesh_index=0,
                        init=False,
                        save_on_folder=True,
                    )
                    visualize_frame_results_gt(
                        color=color,
                        gt_mesh=gt_mesh,
                        K=intrinsic,
                        gt_pose=gt_pose,
                        pred_pose=pred_pose_local,
                        obj_f=obj,
                        frame_idx=0,
                        save_path=obj_dir,
                        glctx=glctx,
                        name=f"anchor_con_{label}",
                        est_mesh=est_local.mesh,
                        save_on_folder=False,
                    )
                except Exception:
                    pass
            return {
                "mesh": mesh_local,
                "pred_pose": pred_pose_local,
                "l_depth": float(l_depth),
                "l_mask": float(l_mask),
                "l_geom": float(l_geom),
            }

        results_pose = {}
        for label, path in candidates.items():
            results_pose[label] = _evaluate_mesh_pose(path, label, visualize=False)
            res = results_pose[label]
            pose_rows_all.append(
                {
                    "Object": obj,
                    "Object_Number": obj_num,
                    "Candidate": label,
                    "L_depth": float(res["l_depth"]),
                    "L_mask": float(res["l_mask"]),
                    "L_geom": float(res["l_geom"]),
                    "S_anchor": float(s_anchor_scores[label]),
                }
            )

        if not results_pose:
            raise RuntimeError(f"No valid meshes for pose evaluation: {obj}")

        # ====================================================================
        # Step 4: Total scoring and candidate selection (Eq. 2, 3, 15)
        # S(k) = alpha * s_anchor(k) + beta * s_query(k)
        # s_query is normalized by min values across candidates
        # ====================================================================
        def _norm(v: float, min_v: float) -> float:
            return float(v) / float(min_v + 1e-8)

        min_l_depth = min(res["l_depth"] for res in results_pose.values())
        min_l_mask = min(res["l_mask"] for res in results_pose.values())
        min_l_geom = min(res["l_geom"] for res in results_pose.values())

        w_depth = args.select_depth_weight
        w_mask = args.select_mask_weight
        w_geom = args.select_geom_weight
        alpha = args.score_alpha
        beta = args.score_beta

        max_s_anchor = max(s_anchor_scores.values()) if s_anchor_scores else 1.0

        scored = []
        score_breakdown = {}
        for label, res in results_pose.items():
            depth_term = w_depth * _norm(res["l_depth"], min_l_depth)
            mask_term = w_mask * _norm(res["l_mask"], min_l_mask)
            geom_term = w_geom * _norm(res["l_geom"], min_l_geom)
            s_query = depth_term + mask_term + geom_term

            s_anchor_norm = s_anchor_scores[label] / max(max_s_anchor, 1e-8)
            total_score = -alpha * s_anchor_norm + beta * s_query

            scored.append((label, res, float(total_score)))
            score_breakdown[label] = {
                "Score_L_depth": float(depth_term),
                "Score_L_mask": float(mask_term),
                "Score_L_geom": float(geom_term),
                "S_query": float(s_query),
                "S_anchor": float(s_anchor_scores[label]),
                "S_anchor_norm": float(s_anchor_norm),
                "Total_Score": float(total_score),
                "Weight_L_depth": float(w_depth),
                "Weight_L_mask": float(w_mask),
                "Weight_L_geom": float(w_geom),
                "Alpha": float(alpha),
                "Beta": float(beta),
            }
        scored.sort(key=lambda x: x[2])
        selection_scores = {label: score for label, _, score in scored}

        chosen_label, best, _ = scored[0]
        forced_by_chamfer = False
        best_geom_label, best_geom = min(
            results_pose.items(), key=lambda x: x[1]["l_geom"]
        )
        if best["l_geom"] > best_geom["l_geom"] * (1.0 + args.select_chamfer_force_ratio):
            chosen_label, best = best_geom_label, best_geom
            forced_by_chamfer = True

        for i, row in enumerate(pose_rows_all):
            if row["Object"] == obj and row["Candidate"] in selection_scores:
                pose_rows_all[i]["Selection_Score"] = float(selection_scores[row["Candidate"]])
                breakdown = score_breakdown[row["Candidate"]]
                pose_rows_all[i].update(breakdown)

        print(
            f"{obj}: pose-best={chosen_label} "
            f"L_depth={best['l_depth']:.6f} L_mask={best['l_mask']:.6f} L_geom={best['l_geom']:.6f} "
            f"S_anchor={s_anchor_scores[chosen_label]:.4f}"
        )
        _ = _evaluate_mesh_pose(candidates[chosen_label], f"chosen_{chosen_label}", visualize=True)
        chamfer_dis = best["l_geom"]

        chosen_mesh_path = candidates[chosen_label]
        final_mesh = trimesh.load(chosen_mesh_path)
        final_mesh = align_mesh_to_coordinate(final_mesh)
        final_mesh.export(os.path.join(obj_dir, f"center_mesh_{obj}.obj"))
        est_final = Any6D(symmetry_tfs=symmetry_tfs, mesh=final_mesh, debug_dir=obj_dir, debug=0)
        pred_pose_final = est_final.register_any6d(
            K=intrinsic,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=args.any6d_iter,
            refinement=bool(args.any6d_refine),
            name="demo",
        )
        np.savetxt(os.path.join(obj_dir, f"{obj}_initial_pose.txt"), pred_pose_final)
        est_final.mesh.export(os.path.join(obj_dir, f"final_mesh_{obj}.obj"))
        np.savetxt(os.path.join(obj_dir, f"{obj}_cd.txt"), [chamfer_dis])

        results.append(
            {"Object": obj, "Object_Number": obj_num, "Chamfer_Distance": float(chamfer_dis)}
        )
        pose_rows.append(
            {
                "Object": obj,
                "Object_Number": obj_num,
                "Chosen_Mesh": chosen_label,
                "L_depth": float(best["l_depth"]),
                "L_mask": float(best["l_mask"]),
                "L_geom": float(chamfer_dis),
                "S_anchor": float(s_anchor_scores[chosen_label]),
                "Selection_Score": float(selection_scores.get(chosen_label, float("inf"))),
                "Forced_By_Chamfer": bool(forced_by_chamfer),
            }
        )

        # ====================================================================
        # Save per-object candidate registry for query phase (Section 3.3.2)
        # ====================================================================
        obj_candidates_info = {}
        for label, path in candidates.items():
            final_mesh_p = os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj")
            obj_candidates_info[label] = {
                "mesh_path": os.path.abspath(final_mesh_p),
                "raw_mesh_path": os.path.abspath(path),
                "pose_path": os.path.abspath(os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt")),
                "s_anchor": float(s_anchor_scores[label]),
                "l_depth": float(results_pose[label]["l_depth"]),
                "l_mask": float(results_pose[label]["l_mask"]),
                "l_geom": float(results_pose[label]["l_geom"]),
            }
        candidate_registry[obj] = {
            "chosen": chosen_label,
            "candidates": obj_candidates_info,
            "final_mesh_path": os.path.abspath(os.path.join(obj_dir, f"final_mesh_{obj}.obj")),
            "final_pose_path": os.path.abspath(os.path.join(obj_dir, f"{obj}_initial_pose.txt")),
            "K_path": os.path.abspath(os.path.join(obj_dir, "K.txt")),
        }

    # ====================================================================
    # Save candidate registry JSON for query phase
    # ====================================================================
    registry_path = os.path.join(args.anchor_folder, "candidate_registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(candidate_registry, f, indent=2)
    print(f"\nCandidate registry saved to: {registry_path}")

    df = pd.DataFrame(results).sort_values("Object")
    excel_path = os.path.join(args.anchor_folder, "chamfer_distances.xlsx")
    df.to_excel(excel_path, index=False)

    pose_df = pd.DataFrame(pose_rows).sort_values("Object")
    pose_all_df = pd.DataFrame(pose_rows_all).sort_values(["Object", "Candidate"])
    pose_excel_path = os.path.join(args.anchor_folder, "pose_metrics.xlsx")
    with pd.ExcelWriter(pose_excel_path) as writer:
        pose_df.to_excel(writer, index=False, sheet_name="chosen")
        pose_all_df.to_excel(writer, index=False, sheet_name="candidates")

    print("\nChamfer Distance Summary Statistics:")
    print(df["Chamfer_Distance"].describe())
    print(f"\nResults saved to: {excel_path}")
    print(f"Pose metrics saved to: {pose_excel_path}")


if __name__ == "__main__":
    main()
