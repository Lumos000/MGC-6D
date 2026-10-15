
from project_paths import setup_project_paths
setup_project_paths()
import argparse
import gc
import glob
import hashlib
import json
import os
import random
import shutil
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
from candidate_admission import is_genuine_fusion, is_instantmesh_only
from candidate_selection import derive_anchor_reliability_scores, score_candidates
from revision_protocol import runtime_provenance


OBJ_NUM_MAP = {
    "003_cracker_box": 2,
    "006_mustard_bottle": 5,
    "021_bleach_cleanser": 12,
    "019_pitcher_base": 11,
    "004_sugar_box": 3,
    "005_tomato_soup_can": 4,
    "010_potted_meat_can": 9,
}


def _release_cuda_memory():
    """Release cyclic model references and unused CUDA allocations between stages."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        with torch.cuda.device(device_idx):
            torch.cuda.empty_cache()


def _stable_object_seed(global_seed: int, obj: str, label: str) -> int:
    """Return the one deterministic registration seed for an object candidate."""

    key = f"{obj}|{label}|single_any6d_coarse_aa".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return int((int(global_seed) + offset) % (2**31 - 1))


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
    failed_fused_uses_base: bool = False,
) -> float:
    """Compute a bounded geometric-compatibility prior in ``[0, 1]``.

    Base candidates receive the neutral bounded prior ``c_base``.  A fused
    candidate receives a weighted mean of ICP coverage and an exponential RMSE
    quality.  Fusion is never rewarded merely for existing.
    """
    if label in ("rayst3r", "instantmesh"):
        return float(np.clip(c_base, 0.0, 1.0))

    fitness = meta.get("align_fitness", -1.0)
    rmse = meta.get("align_rmse", float("inf"))
    fitness_ok = fitness >= tau_inlier

    if label != "fused":
        return 0.0
    if not fitness_ok:
        return (
            float(np.clip(c_base, 0.0, 1.0))
            if failed_fused_uses_base
            else 0.0
        )
    weights = np.asarray(
        [max(float(w_inlier), 0.0), max(float(w_rmse), 0.0)], dtype=np.float64
    )
    if weights.sum() <= 0.0:
        return 0.0
    inlier_quality = float(np.clip(fitness, 0.0, 1.0))
    rmse_quality = float(
        np.exp(-max(float(rmse), 0.0) / max(float(tau_rmse), 1e-8))
    )
    alignment_quality = float(
        np.clip(
            np.dot(weights, [inlier_quality, rmse_quality]) / weights.sum(),
            0.0,
            1.0,
        )
    )
    # construction.py already folds bidirectional cycle confidence into this
    # Phi reliability.  Keeping it bounded prevents a fused candidate from
    # winning merely because one ICP direction happens to pass.
    fusion_quality = float(
        meta.get("fusion_diagnostics", {}).get("reliability", alignment_quality)
    )
    return float(np.clip(0.5 * alignment_quality + 0.5 * fusion_quality, 0.0, 1.0))


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
        "--align_strict_both",
        action="store_true",
        help="Ablation: require both ICP directions to pass Fitness/RMSE thresholds",
    )
    parser.add_argument(
        "--align_relaxed_rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional rescue ablation; disabled for the paper main method.",
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
        "--icp_rmse_threshold",
        type=float,
        default=0.02,
        help=(
            "Maximum ICP inlier RMSE (meters) accepted by each direction of "
            "the strict-both gate"
        ),
    )
    parser.add_argument(
        "--icp_multi_hypo",
        type=int,
        default=24,
        help=(
            "Number of deterministic ICP initial hypotheses. Values up to 24 "
            "enumerate proper axis rotations about the object centroid."
        ),
    )
    parser.add_argument(
        "--icp_reciprocal_pairing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use inverse-paired deterministic initial hypotheses in the two "
            "directional ICP searches, then report a reciprocal round trip by "
            "refining the inverse of the selected transform. Direction selection "
            "remains best-of-two; cycle values are diagnostic only."
        ),
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip the anchor crop horizontally for InstantMesh (default: enabled)",
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
    parser.add_argument("--rayst3r_crop_margin_ratio", type=float, default=0.05)
    parser.add_argument("--rayst3r_min_component_area_ratio", type=float, default=0.005)
    parser.add_argument("--rayst3r_target_faces", type=int, default=100000)
    parser.add_argument(
        "--fusion_mode",
        choices=["poisson", "visible_patch"],
        default="visible_patch",
        help=(
            "HO3D main method: preserve the InstantMesh topology and update "
            "only its RGB-D-supported visible surface from RaySt3R."
        ),
    )
    parser.add_argument("--fusion_voxel_size", type=float, default=0.003)
    parser.add_argument("--fusion_sample_points", type=int, default=15000)
    parser.add_argument("--fusion_poisson_depth", type=int, default=8)
    parser.add_argument(
        "--fusion_target_faces",
        type=int,
        default=120000,
        help="Safety cap metadata; visible_patch keeps the InstantMesh face connectivity.",
    )
    parser.add_argument("--fusion_density_quantile", type=float, default=0.02)
    parser.add_argument("--fusion_min_component_area_ratio", type=float, default=0.005)
    parser.add_argument("--fusion_visible_depth_tolerance", type=float, default=0.015)
    parser.add_argument("--fusion_observed_support_ratio", type=float, default=0.08)
    parser.add_argument("--fusion_agreement_ratio", type=float, default=0.025)
    parser.add_argument("--fusion_min_prior_coverage", type=float, default=0.35)
    parser.add_argument("--fusion_min_observed_coverage", type=float, default=0.50)
    parser.add_argument("--fusion_min_extent_ratio", type=float, default=0.65)
    parser.add_argument("--fusion_max_extent_ratio", type=float, default=1.35)
    parser.add_argument("--fusion_max_center_offset_ratio", type=float, default=0.15)
    parser.add_argument(
        "--fusion_cycle_in_reliability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablation only: use bidirectional cycle confidence in fused reliability. "
            "The paper main result reports cycle error but does not gate or score with it."
        ),
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
    parser.add_argument(
        "--fusion_admission_policy",
        choices=("ho3d", "toyl_real275"),
        default="ho3d",
        help=(
            "ho3d keeps the HO3D Fitness admission and zeroes failed fused "
            "calibration; toyl_real275 admits genuine fusion outputs and uses "
            "the neutral base calibration when Fitness/RMSE consistency fails"
        ),
    )
    parser.add_argument("--anchor_depth_scale", type=float, default=0.05)
    parser.add_argument("--anchor_mask_scale", type=float, default=0.50)
    parser.add_argument("--anchor_geom_scale", type=float, default=0.05)
    parser.add_argument(
        "--anchor_calibration_weight",
        type=float,
        default=0.15,
        help="Bounded contribution of ICP compatibility to effective anchor reliability",
    )
    # --- Total score balance (Eq. 2): S(k) = alpha*s_anchor + beta*s_query ---
    parser.add_argument(
        "--score_alpha",
        type=float,
        default=0.4,
        help="alpha: weight for anchor calibration score in total score (Eq. 2)",
    )
    parser.add_argument(
        "--score_beta",
        type=float,
        default=0.6,
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
    parser.add_argument(
        "--objects",
        type=str,
        default="",
        help="Comma-separated HO3D object names to process; empty processes all.",
    )
    parser.add_argument(
        "--anchor_gt_symmetry_ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablation only: pass dataset-provided YCB discrete symmetries to "
            "anchor Any6D registration. Disabled for the paper main protocol, "
            "which does not use GT object symmetry during estimation."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.anchor_calibration_weight <= 1.0:
        raise ValueError("--anchor_calibration_weight must lie in [0, 1]")
    for name in ("anchor_depth_scale", "anchor_mask_scale", "anchor_geom_scale"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"--{name} must be finite and positive")
    if args.icp_max_corr_ratio <= 0.0:
        raise ValueError("--icp_max_corr_ratio must be positive")
    if not np.isfinite(args.icp_fitness_threshold) or args.icp_fitness_threshold < 0.0:
        raise ValueError("--icp_fitness_threshold must be finite and non-negative")
    if not np.isfinite(args.icp_rmse_threshold) or args.icp_rmse_threshold < 0.0:
        raise ValueError("--icp_rmse_threshold must be finite and non-negative")
    if args.align_strict_both and not args.align_bidirectional_icp:
        raise ValueError("--align_strict_both requires --align_bidirectional_icp")
    if args.rayst3r_crop_margin_ratio < 0.0:
        raise ValueError("--rayst3r_crop_margin_ratio must be non-negative")
    if not 0.0 <= args.rayst3r_min_component_area_ratio < 1.0:
        raise ValueError("--rayst3r_min_component_area_ratio must lie in [0, 1)")
    for name in (
        "fusion_voxel_size",
        "fusion_visible_depth_tolerance",
        "fusion_observed_support_ratio",
        "fusion_agreement_ratio",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"--{name} must be finite and positive")
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
    input_root = os.path.realpath(args.anchor_folder)
    result_root = os.path.realpath(args.output_root or args.anchor_folder)
    os.makedirs(result_root, exist_ok=True)
    obj_list = sorted(
        d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))
    )
    if args.objects.strip():
        requested = {name.strip() for name in args.objects.split(",") if name.strip()}
        obj_list = [obj for obj in obj_list if obj in requested]
    if not obj_list:
        raise RuntimeError(f"No requested object folders found in {input_root}")
    cfg = ReconstructionConfig(
        anchor_root=input_root,
        output_root=result_root,
        object_names=tuple(obj_list),
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
        align_strict_both=args.align_strict_both,
        align_relaxed_rescue=args.align_relaxed_rescue,
        icp_max_iter=args.icp_max_iter,
        icp_fitness_threshold=args.icp_fitness_threshold,
        align_rmse_penalty_threshold=args.icp_rmse_threshold,
        icp_multi_hypo=args.icp_multi_hypo,
        icp_reciprocal_pairing=args.icp_reciprocal_pairing,
        icp_init_sigma=args.icp_init_sigma,
        icp_max_corr_ratio=args.icp_max_corr_ratio,
        icp_voxel_size=args.icp_voxel_size,
        sample_points=args.sample_points,
        rayst3r_crop_margin_ratio=args.rayst3r_crop_margin_ratio,
        rayst3r_min_component_area_ratio=args.rayst3r_min_component_area_ratio,
        rayst3r_target_faces=args.rayst3r_target_faces,
        fusion_mode=args.fusion_mode,
        fusion_voxel_size=args.fusion_voxel_size,
        fusion_sample_points=args.fusion_sample_points,
        fusion_poisson_depth=args.fusion_poisson_depth,
        fusion_target_faces=args.fusion_target_faces,
        fusion_density_quantile=args.fusion_density_quantile,
        fusion_min_component_area_ratio=args.fusion_min_component_area_ratio,
        fusion_visible_depth_tolerance=args.fusion_visible_depth_tolerance,
        fusion_observed_support_ratio=args.fusion_observed_support_ratio,
        fusion_agreement_ratio=args.fusion_agreement_ratio,
        fusion_min_prior_coverage=args.fusion_min_prior_coverage,
        fusion_min_observed_coverage=args.fusion_min_observed_coverage,
        fusion_min_extent_ratio=args.fusion_min_extent_ratio,
        fusion_max_extent_ratio=args.fusion_max_extent_ratio,
        fusion_max_center_offset_ratio=args.fusion_max_center_offset_ratio,
        fusion_cycle_in_reliability=args.fusion_cycle_in_reliability,
        seed=args.seed,
    )

    # ====================================================================
    # Step 1: Anchor-phase candidate geometry construction (Section 3.2)
    # ====================================================================
    recon_results = run_construction(cfg)
    _release_cuda_memory()
    model_info = (
        _load_model_info(args.ycb_model_path)
        if args.anchor_gt_symmetry_ablation
        else {}
    )

    run_meta = {
        "anchor_folder": input_root,
        "output_root": result_root,
        "ycb_model_path": args.ycb_model_path,
        "mask_source": "dataset_gt" if not args.refine_mask else "sam2_refined",
        "geometry_units": "meters_after_construction",
        "candidate_canonicalization_scale": 1.0,
        "fusion_admission_policy": args.fusion_admission_policy,
        "reconstruction_method": (
            "current_metric_scale_topology_preserving_visible_surface_phi_v5_ray_frame"
            if args.fusion_mode == "visible_patch"
            else "current_metric_scale_visible_prior_poisson_phi_v3_ray_frame"
        ),
        "anchor_pose_method": "single_any6d_coarse_axis_aligned",
        "estimation_postprocessing": {
            "gt_symmetry_adjustment": bool(args.anchor_gt_symmetry_ablation),
            "paper_main_protocol": bool(not args.anchor_gt_symmetry_ablation),
        },
        "bidirectional_alignment": {
            "selection_rule": "best_of_two",
            "strict_both_gate": bool(args.align_strict_both),
            "strict_both_gate_source": "independent_directional_optima",
            "strict_both_gate_definition": (
                "both independent ICP directions satisfy Fitness>=tau_f and RMSE<=tau_r"
            ),
            "strict_tau_f": float(args.icp_fitness_threshold),
            "strict_tau_r": float(args.icp_rmse_threshold),
            "cycle_role": (
                "bounded_reliability_ablation"
                if args.fusion_cycle_in_reliability
                else "diagnostic_only"
            ),
            "icp_hypotheses_per_direction": int(args.icp_multi_hypo),
            "reciprocal_pairing": bool(args.icp_reciprocal_pairing),
            "independent_cycle_definition": (
                "composition_of_independently_optimized_directional_results"
            ),
            "reciprocal_cycle_definition": (
                "selected_transform_inverse_refinement_round_trip"
                if args.icp_reciprocal_pairing
                else "not_computed"
            ),
            "translation_cycle_definition": (
                "mean_bidirectional_surface_center_round_trip_displacement_over_diameter"
            ),
        },
        "selection": {
            "select_depth_weight": args.select_depth_weight,
            "select_mask_weight": args.select_mask_weight,
            "select_geom_weight": args.select_geom_weight,
        },
        "calibration": {
            "anchor_c_base": args.anchor_c_base,
            "anchor_w_inlier": args.anchor_w_inlier,
            "anchor_w_rmse": args.anchor_w_rmse,
            "anchor_tau_inlier": args.anchor_tau_inlier,
            "anchor_tau_rmse": args.anchor_tau_rmse,
            "anchor_depth_scale": args.anchor_depth_scale,
            "anchor_mask_scale": args.anchor_mask_scale,
            "anchor_geom_scale": args.anchor_geom_scale,
            "anchor_calibration_weight": args.anchor_calibration_weight,
            "score_alpha": args.score_alpha,
            "score_beta": args.score_beta,
        },
        "provenance": runtime_provenance(os.path.dirname(__file__)),
        "args": vars(args),
    }
    with open(os.path.join(result_root, "anchor_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    results = []
    pose_rows = []
    pose_rows_all = []
    candidate_registry = {}
    glctx = dr.RasterizeCudaContext()

    for obj in tqdm(obj_list, desc="Object"):
        obj_num = OBJ_NUM_MAP.get(obj, -1)
        symmetry_tfs = (
            _build_symmetry_tfs_for_object(obj, model_info)
            if args.anchor_gt_symmetry_ablation
            else None
        )
        input_obj_dir = os.path.join(input_root, obj)
        obj_dir = os.path.join(result_root, obj)
        os.makedirs(obj_dir, exist_ok=True)
        for input_name in (
            "color.png",
            "depth.png",
            "mask.png",
            "K.txt",
            f"{obj}_gt_pose.txt",
            "source_frame.txt",
        ):
            src = os.path.join(input_obj_dir, input_name)
            dst = os.path.join(obj_dir, input_name)
            if os.path.isfile(src) and os.path.realpath(src) != os.path.realpath(dst):
                shutil.copy2(src, dst)

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
        align_valid = (
            align_direction is not None
            and not str(align_direction).startswith("none")
            and align_direction != "instantmesh_only_fallback"
        )
        if align_valid and fused_path and os.path.exists(fused_path):
            fusion_diagnostics = recon_meta.get("fusion_diagnostics", {})
            if args.fusion_admission_policy == "toyl_real275":
                fused_ok = is_genuine_fusion(
                    recon_meta,
                    instantmesh_only=is_instantmesh_only(recon_meta),
                )
            else:
                fused_ok = (
                    recon_meta.get("align_fitness", -1.0) >= args.anchor_tau_inlier
                    and not bool(fusion_diagnostics.get("fallback_used", False))
                    and bool(fusion_diagnostics.get("output_valid", False))
                )
            if fused_ok:
                candidates["fused"] = fused_path

        calibration_scores = {}
        for label in candidates:
            calibration_scores[label] = _compute_anchor_calibration_score(
                meta=recon_meta,
                label=label,
                c_base=args.anchor_c_base,
                w_inlier=args.anchor_w_inlier,
                w_rmse=args.anchor_w_rmse,
                tau_inlier=args.anchor_tau_inlier,
                tau_rmse=args.anchor_tau_rmse,
                failed_fused_uses_base=(
                    args.fusion_admission_policy == "toyl_real275"
                ),
            )

        # ====================================================================
        # Step 3: For each candidate, run Any6D pose registration on anchor
        # image and compute observation consistency score (Eq. 14)
        # ====================================================================
        def _evaluate_mesh_pose(mesh_path: str, label: str):
            mesh_local = trimesh.load(mesh_path)
            if isinstance(mesh_local, trimesh.Scene):
                mesh_local = trimesh.util.concatenate(
                    [
                        geometry
                        for geometry in mesh_local.geometry.values()
                        if isinstance(geometry, trimesh.Trimesh)
                    ]
                )
            # Construction outputs are already metric.  Applying the former
            # x0.1 here would silently undo the recovered scale.
            mesh_local = align_mesh_to_coordinate(mesh_local, scale=1.0)
            mesh_local.export(os.path.join(obj_dir, f"center_mesh_{obj}_{label}.obj"))

            registration_seed = _stable_object_seed(args.seed, obj, label)
            random.seed(registration_seed)
            np.random.seed(registration_seed)
            torch.manual_seed(registration_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(registration_seed)
            est_local = Any6D(
                symmetry_tfs=symmetry_tfs,
                mesh=mesh_local.copy(),
                debug_dir=obj_dir,
                debug=0,
            )
            try:
                pred_pose_local = est_local.register_any6d(
                    K=intrinsic,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.any6d_iter,
                    refinement=bool(args.any6d_refine),
                    axis_align=True,
                    coarse_est=True,
                    random_seed=registration_seed,
                    name=f"anchor_{label}_single_aa",
                )
                l_depth, l_mask, l_geom = _compute_self_supervised_losses(
                    est=est_local,
                    depth=depth,
                    mask=mask,
                    K=intrinsic,
                    pred_pose=pred_pose_local,
                    glctx=glctx,
                )
                registered_mesh = est_local.mesh.copy()
            except Exception:
                del est_local
                _release_cuda_memory()
                raise
            del est_local
            _release_cuda_memory()

            np.savetxt(
                os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt"),
                pred_pose_local,
            )
            registered_mesh.export(os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj"))
            np.savetxt(
                os.path.join(obj_dir, f"{obj}_ldepth_{label}.txt"),
                [l_depth],
            )
            np.savetxt(
                os.path.join(obj_dir, f"{obj}_lmask_{label}.txt"),
                [l_mask],
            )
            np.savetxt(
                os.path.join(obj_dir, f"{obj}_cd_{label}.txt"),
                [l_geom],
            )
            return {
                "mesh": registered_mesh,
                "pred_pose": pred_pose_local,
                "l_depth": float(l_depth),
                "l_mask": float(l_mask),
                "l_geom": float(l_geom),
                "anchor_pose_method": "single_any6d_coarse_axis_aligned",
                "anchor_pose_random_seed": int(registration_seed),
            }

        results_pose = {}
        for label, path in candidates.items():
            results_pose[label] = _evaluate_mesh_pose(path, label)

        if not results_pose:
            raise RuntimeError(f"No valid meshes for pose evaluation: {obj}")

        reliability_rows = [
            {
                "label": label,
                "s_anchor": float(calibration_scores[label]),
                "l_depth": float(res["l_depth"]),
                "l_mask": float(res["l_mask"]),
                "l_geom": float(res["l_geom"]),
            }
            for label, res in results_pose.items()
        ]
        derive_anchor_reliability_scores(
            reliability_rows,
            w_depth=args.select_depth_weight,
            w_mask=args.select_mask_weight,
            w_geom=args.select_geom_weight,
            depth_scale=args.anchor_depth_scale,
            mask_scale=args.anchor_mask_scale,
            geom_scale=args.anchor_geom_scale,
            calibration_weight=args.anchor_calibration_weight,
        )
        reliability_by_label = {row["label"]: row for row in reliability_rows}
        s_anchor_scores = {
            label: float(row["s_anchor"]) for label, row in reliability_by_label.items()
        }
        for label, res in results_pose.items():
            reliability = reliability_by_label[label]
            pose_rows_all.append(
                {
                    "Object": obj,
                    "Object_Number": obj_num,
                    "Candidate": label,
                    "L_depth": float(res["l_depth"]),
                    "L_mask": float(res["l_mask"]),
                    "L_geom": float(res["l_geom"]),
                    "S_anchor": float(reliability["s_anchor"]),
                    "S_anchor_raw": float(reliability["s_anchor_raw"]),
                    "Anchor_Observation_Quality": float(
                        reliability["anchor_observation_quality"]
                    ),
                    "Anchor_Calibration_Norm": float(
                        reliability["anchor_calibration_norm"]
                    ),
                }
            )

        # ====================================================================
        # Step 4: Total scoring and candidate selection (Eq. 2, 3, 15)
        # S(k) = alpha * s_anchor(k) + beta * s_query(k)
        # Anchor losses use disclosed fixed scales, matching the revised query
        # scorer and avoiding candidate-set-dependent L/min(L) behavior.
        # ====================================================================
        w_depth = args.select_depth_weight
        w_mask = args.select_mask_weight
        w_geom = args.select_geom_weight
        alpha = args.score_alpha
        beta = args.score_beta

        anchor_rows = []
        for label, res in results_pose.items():
            anchor_rows.append({
                "label": label,
                "s_anchor": float(s_anchor_scores[label]),
                "anchor_score_version": "bounded_observation_calibrated_v1",
                "l_depth": float(res["l_depth"]),
                "l_mask": float(res["l_mask"]),
                "l_geom": float(res["l_geom"]),
            })
        score_candidates(
            anchor_rows,
            w_depth=w_depth,
            w_mask=w_mask,
            w_geom=w_geom,
            score_alpha=alpha,
            score_beta=beta,
            loss_normalization="fixed_scale",
            depth_scale=args.anchor_depth_scale,
            mask_scale=args.anchor_mask_scale,
            geom_scale=args.anchor_geom_scale,
        )
        scored_rows = sorted(anchor_rows, key=lambda row: row["total_score"], reverse=True)
        chosen_label = scored_rows[0]["label"]
        best = results_pose[chosen_label]
        selection_scores = {row["label"]: row["total_score"] for row in scored_rows}
        score_breakdown = {
            row["label"]: {
                "Score_L_depth": float(w_depth * row["l_depth_norm"]),
                "Score_L_mask": float(w_mask * row["l_mask_norm"]),
                "Score_L_geom": float(w_geom * row["l_geom_norm"]),
                "S_query": float(row["s_query"]),
                "S_anchor": float(row["s_anchor"]),
                "S_anchor_norm": float(row["s_anchor_norm"]),
                "Total_Score": float(row["total_score"]),
                "Weight_L_depth": float(w_depth),
                "Weight_L_mask": float(w_mask),
                "Weight_L_geom": float(w_geom),
                "Alpha": float(alpha),
                "Beta": float(beta),
            }
            for row in scored_rows
        }

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
        chamfer_dis = best["l_geom"]
        # Reuse the single registered pose/mesh exactly; do not run an outer
        # axis-align/no-axis-align selection after candidate scoring.
        best["mesh"].export(os.path.join(obj_dir, f"center_mesh_{obj}.obj"))
        best["mesh"].export(os.path.join(obj_dir, f"final_mesh_{obj}.obj"))
        np.savetxt(
            os.path.join(obj_dir, f"{obj}_initial_pose.txt"), best["pred_pose"]
        )
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
                "Anchor_Pose_Method": best["anchor_pose_method"],
                "Anchor_Pose_Random_Seed": best["anchor_pose_random_seed"],
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
                "anchor_score_version": "bounded_observation_calibrated_v1",
                "s_anchor_raw": float(reliability_by_label[label]["s_anchor_raw"]),
                "anchor_observation_quality": float(
                    reliability_by_label[label]["anchor_observation_quality"]
                ),
                "anchor_calibration_norm": float(
                    reliability_by_label[label]["anchor_calibration_norm"]
                ),
                "l_depth": float(results_pose[label]["l_depth"]),
                "l_mask": float(results_pose[label]["l_mask"]),
                "l_geom": float(results_pose[label]["l_geom"]),
                "anchor_pose_method": results_pose[label]["anchor_pose_method"],
                "anchor_pose_random_seed": results_pose[label]["anchor_pose_random_seed"],
            }
        candidate_registry[obj] = {
            "chosen": chosen_label,
            "anchor_score_version": "bounded_observation_calibrated_v1",
            "mask_source": "dataset_gt" if not args.refine_mask else "sam2_refined",
            "geometry_units": "meters_after_construction",
            "scale_metadata": {
                "depth_unit": recon_meta.get("depth_unit", chosen_depth_unit),
                "instantmesh_metric_scale": recon_meta.get(
                    "instantmesh_metric_scale", 1.0
                ),
                "construction_transform": recon_meta.get("transform"),
                "mesh_statistics": recon_meta.get("mesh_statistics", {}),
            },
            "fusion_metadata": recon_meta.get("fusion_diagnostics", {}),
            "bidirectional_metadata": recon_meta.get("bidirectional", {}),
            "alignment_metadata": {
                "selection_rule": "best_of_two",
                "strict_both_gate": bool(args.align_strict_both),
                "strict_both_gate_source": "independent_directional_optima",
                "strict_tau_f": float(args.icp_fitness_threshold),
                "strict_tau_r": float(args.icp_rmse_threshold),
                "reciprocal_pairing": bool(args.icp_reciprocal_pairing),
                "cycle_role": (
                    "bounded_reliability_ablation"
                    if args.fusion_cycle_in_reliability
                    else "diagnostic_only"
                ),
                "direction": recon_meta.get("align_direction"),
                "fitness": recon_meta.get("align_fitness"),
                "rmse": recon_meta.get("align_rmse"),
                "decision_score": recon_meta.get("align_decision_score"),
            },
            "reconstruction_meta_path": (
                os.path.abspath(meta_path) if meta_path else None
            ),
            "candidates": obj_candidates_info,
            "final_mesh_path": os.path.abspath(os.path.join(obj_dir, f"final_mesh_{obj}.obj")),
            "final_pose_path": os.path.abspath(os.path.join(obj_dir, f"{obj}_initial_pose.txt")),
            "K_path": os.path.abspath(os.path.join(obj_dir, "K.txt")),
        }

        del results_pose
        _release_cuda_memory()

    # ====================================================================
    # Save candidate registry JSON for query phase
    # ====================================================================
    registry_path = os.path.join(result_root, "candidate_registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(candidate_registry, f, indent=2)
    print(f"\nCandidate registry saved to: {registry_path}")

    df = pd.DataFrame(results).sort_values("Object")
    excel_path = os.path.join(result_root, "chamfer_distances.xlsx")
    df.to_excel(excel_path, index=False)

    pose_df = pd.DataFrame(pose_rows).sort_values("Object")
    pose_all_df = pd.DataFrame(pose_rows_all).sort_values(["Object", "Candidate"])
    pose_excel_path = os.path.join(result_root, "pose_metrics.xlsx")
    with pd.ExcelWriter(pose_excel_path) as writer:
        pose_df.to_excel(writer, index=False, sheet_name="chosen")
        pose_all_df.to_excel(writer, index=False, sheet_name="candidates")

    print("\nChamfer Distance Summary Statistics:")
    print(df["Chamfer_Distance"].describe())
    print(f"\nResults saved to: {excel_path}")
    print(f"Pose metrics saved to: {pose_excel_path}")


if __name__ == "__main__":
    main()
