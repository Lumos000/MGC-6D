from project_paths import setup_project_paths
setup_project_paths()

import argparse
import gc
import hashlib
import json
import os
import random
import shutil

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
    nvdiffrast_render,
)
from candidate_selection import (
    derive_anchor_reliability_scores,
    score_candidates,
)
from candidate_admission import is_genuine_fusion, is_instantmesh_only
from revision_protocol import runtime_provenance


def _release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _parse_object_number_from_source_frame(obj_dir: str) -> int:
    src = os.path.join(obj_dir, "source_frame.txt")
    if not os.path.exists(src):
        return -1
    try:
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("class_id="):
                    return int(line.split("=", 1)[1].strip())
    except Exception:
        pass
    return -1


def _as_trimesh(mesh_obj) -> trimesh.Trimesh:
    if isinstance(mesh_obj, trimesh.Scene):
        meshes = [
            geometry
            for geometry in mesh_obj.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError("Loaded scene contains no triangle mesh")
        return trimesh.util.concatenate(meshes)
    if not isinstance(mesh_obj, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh or Scene, got {type(mesh_obj).__name__}")
    return mesh_obj


def _backproject_depth_to_points(
    depth: np.ndarray,
    k: np.ndarray,
    valid_mask: np.ndarray,
    max_points: int = 5000,
) -> np.ndarray:
    ys, xs = np.where(valid_mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float32)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    x = (xs.astype(np.float32) - cx) * z / max(fx, 1e-8)
    y = (ys.astype(np.float32) - cy) * z / max(fy, 1e-8)
    points = np.stack([x, y, z], axis=1)
    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[indices]
    return points


def _chamfer_distance_pointcloud(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return 1e3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_a = torch.as_tensor(points_a, dtype=torch.float32, device=device)
    tensor_b = torch.as_tensor(points_b, dtype=torch.float32, device=device)
    distances = torch.cdist(tensor_a.unsqueeze(0), tensor_b.unsqueeze(0), p=2.0).squeeze(0)
    chamfer = distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()
    return float(chamfer.detach().cpu().item())


def _compute_self_supervised_losses(
    est,
    depth: np.ndarray,
    mask: np.ndarray,
    k: np.ndarray,
    pred_pose: np.ndarray,
    glctx,
) -> tuple[float, float, float]:
    height, width = depth.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_tensor = torch.as_tensor(
        pred_pose, device=device, dtype=torch.float32
    ).unsqueeze(0)
    tf_to_centered = est.get_tf_to_centered_mesh()
    if torch.is_tensor(tf_to_centered):
        tf_to_centered = tf_to_centered.to(
            device=pose_tensor.device, dtype=torch.float32
        ).unsqueeze(0)
    else:
        tf_to_centered = torch.as_tensor(
            tf_to_centered, device=pose_tensor.device, dtype=torch.float32
        ).unsqueeze(0)
    ob_in_cams = pose_tensor @ torch.linalg.inv(tf_to_centered)
    rendered = nvdiffrast_render(
        mesh=est.mesh,
        mesh_tensors=est.mesh_tensors,
        ob_in_cams=ob_in_cams,
        K=k,
        H=height,
        W=width,
        glctx=glctx,
    )
    if rendered is None:
        return 1.0, 1.0, 1e3

    rendered_depth = None
    if isinstance(rendered, tuple) and len(rendered) >= 2:
        rendered_depth = rendered[1]
    elif hasattr(rendered, "shape") and rendered.shape[-1] > 3:
        rendered_depth = rendered[..., 3]
    if rendered_depth is None:
        return 1.0, 1.0, 1e3
    rendered_depth = (
        rendered_depth[0]
        if getattr(rendered_depth, "ndim", 0) > 2
        else rendered_depth
    )
    if torch.is_tensor(rendered_depth):
        rendered_depth = rendered_depth.detach().cpu().numpy()
    rendered_depth = rendered_depth.astype(np.float32)

    observed_mask = (mask > 0) & (depth > 1e-4)
    rendered_mask = rendered_depth > 1e-4
    overlap = observed_mask & rendered_mask
    if overlap.sum() > 10:
        depth_scale = max(float(np.mean(depth[overlap])), 1e-4)
        depth_loss = float(
            np.mean(np.abs(depth[overlap] - rendered_depth[overlap])) / depth_scale
        )
    else:
        depth_loss = 1.0

    union = observed_mask | rendered_mask
    if union.sum() > 0:
        iou = float((observed_mask & rendered_mask).sum()) / float(union.sum())
        mask_loss = float(1.0 - iou)
    else:
        mask_loss = 1.0

    observed_points = _backproject_depth_to_points(depth, k, observed_mask)
    rendered_points = _backproject_depth_to_points(
        rendered_depth, k, rendered_mask
    )
    geometry_loss = _chamfer_distance_pointcloud(rendered_points, observed_points)
    return depth_loss, mask_loss, geometry_loss


def _exported_mesh_pose(
    est: Any6D, returned_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map the exact centered mesh exported by Any6D into the camera frame."""

    tf_to_centered = est.get_tf_to_centered_mesh()
    if torch.is_tensor(tf_to_centered):
        tf_to_centered = tf_to_centered.detach().cpu().numpy()
    tf_to_centered = np.asarray(tf_to_centered, dtype=np.float64).reshape(4, 4)
    returned_pose = np.asarray(returned_pose, dtype=np.float64).reshape(4, 4)
    centered_pose = returned_pose @ np.linalg.inv(tf_to_centered)
    return centered_pose, tf_to_centered


def _compute_anchor_calibration_score(
    meta: dict,
    label: str,
    c_base: float,
    w_inlier: float,
    w_rmse: float,
    tau_inlier: float,
    tau_rmse: float,
) -> float:
    base_score = float(np.clip(c_base, 0.0, 1.0))
    scale_meta = meta.get("scale_recovery", {})
    if label == "fused":
        scale_entries = [
            scale_meta.get("rayst3r", {}),
            scale_meta.get("instantmesh", {}),
        ]
        scale_values = [
            float(entry.get("reliability", 0.0))
            if bool(entry.get("valid", False))
            else 0.0
            for entry in scale_entries
            if bool(entry.get("enabled", False))
        ]
        scale_reliability = min(scale_values) if scale_values else 1.0
    else:
        scale_entry = scale_meta.get(label, {})
        scale_reliability = (
            float(scale_entry.get("reliability", 0.0))
            if bool(scale_entry.get("enabled", False))
            and bool(scale_entry.get("valid", False))
            else (0.0 if bool(scale_entry.get("enabled", False)) else 1.0)
        )
    scale_factor = float(np.clip(0.8 + 0.2 * scale_reliability, 0.8, 1.0))
    if label in ("rayst3r", "instantmesh"):
        return base_score * scale_factor

    fitness = meta.get("align_fitness", -1.0)
    rmse = meta.get("align_rmse", float("inf"))
    if fitness >= tau_inlier and rmse <= tau_rmse:
        weight_sum = max(float(w_inlier) + float(w_rmse), 1e-8)
        rmse_quality = float(
            np.exp(-max(float(rmse), 0.0) / max(float(tau_rmse), 1e-8))
        )
        alignment_quality = (
            float(w_inlier) * float(np.clip(fitness, 0.0, 1.0))
            + float(w_rmse) * rmse_quality
        ) / weight_sum
        fusion_quality = float(
            meta.get("fusion_diagnostics", {}).get(
                "reliability", alignment_quality
            )
        )
        return base_score * scale_factor * float(
            np.clip(0.5 * alignment_quality + 0.5 * fusion_quality, 0.0, 1.0)
        )
    return base_score


def _stable_trial_seed(global_seed: int, obj: str, label: str, strategy: str, seed_i: int) -> int:
    key = f"{obj}|{label}|{strategy}|{seed_i}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return int((int(global_seed) + offset) % (2**31 - 1))


def _detect_depth_unit(depth_uint16: np.ndarray) -> str:
    max_val = float(np.nanmax(depth_uint16))
    if max_val > 20000:
        return "rayst3r_uint16"
    if max_val > 200:
        return "mm"
    return "meters"


def _convert_depth(depth_raw: np.ndarray, depth_unit: str) -> np.ndarray:
    if depth_unit == "mm":
        return depth_raw / 1000.0
    if depth_unit == "rayst3r_uint16":
        return depth_raw / 65535.0 * 10.0
    if depth_unit == "meters":
        return depth_raw.astype(np.float32)
    raise ValueError(f"Unsupported depth_unit: {depth_unit}")


def _run_single_trial(
    mesh_template: trimesh.Trimesh,
    save_path: str,
    glctx,
    args,
    intrinsic: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    name: str,
    random_seed: int = 0,
):
    """Run the same FoundationPose registration path used by REAL275 query."""

    random.seed(int(random_seed))
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_seed))
    mesh_trial = mesh_template.copy()
    est = Any6D(
        symmetry_tfs=None,
        mesh=mesh_trial,
        debug_dir=save_path,
        debug=0,
        glctx=glctx,
        grid_n_views=40,
        grid_inplane_step=60,
    )
    pred_pose = est.register(
        K=intrinsic,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=args.register_iteration,
        glctx=glctx,
        name=name,
    )
    return pred_pose, est


def main():
    parser = argparse.ArgumentParser(description="REAL275 anchor phase (strict paper flow with RaySt3R/InstantMesh/Fused)")
    parser.add_argument(
        "--anchor_folder",
        type=str,
        required=True,
    )
    parser.add_argument("--depth_unit", type=str, default="auto", choices=["auto", "mm", "rayst3r_uint16", "meters"])
    parser.add_argument("--depth_unit_try_both", action="store_true")
    parser.add_argument(
        "--depth_preprocess", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--depth_filter_radius", type=int, default=2)
    parser.add_argument("--depth_filter_device", type=str, default="cuda")
    parser.add_argument("--valid_ratio_threshold", type=float, default=0.01)
    parser.add_argument("--depth_range_min_m", type=float, default=0.2)
    parser.add_argument("--depth_range_max_m", type=float, default=3.0)
    parser.add_argument("--depth_range_min_ratio", type=float, default=0.3)
    parser.add_argument("--depth_range_soft_penalty", action="store_true")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--rayst3r_device", type=str, default="cuda:0")
    parser.add_argument("--rayst3r_checkpoint", type=str, default=None)
    parser.add_argument("--instantmesh_device", type=str, default="cuda:1")
    parser.add_argument(
        "--align_use_guess_translation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--align_bidirectional_icp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--align_strict_both",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ablation: require both ICP directions to pass Fitness/RMSE thresholds",
    )
    parser.add_argument(
        "--align_relaxed_rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional rescue ablation; disabled in the paper main method.",
    )
    parser.add_argument("--icp_max_iter", type=int, default=50)
    parser.add_argument("--icp_fitness_threshold", type=float, default=0.05)
    parser.add_argument(
        "--icp_multi_hypo",
        type=int,
        default=24,
        help=(
            "Number of deterministic ICP initial hypotheses; values up to 24 "
            "enumerate proper axis rotations about the object centroid."
        ),
    )
    parser.add_argument(
        "--icp_reciprocal_pairing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Report cycle consistency by refining the inverse of the selected "
            "best-of-two transform in the reverse direction. This diagnostic "
            "does not change either main directional ICP search."
        ),
    )
    parser.add_argument("--icp_init_sigma", type=float, default=0.01)
    parser.add_argument("--mask_stability_enabled", action="store_true")
    parser.add_argument("--mask_stability_threshold", type=float, default=0.6)
    parser.add_argument("--mask_iou_threshold", type=float, default=0.5)
    parser.add_argument("--refine_mask", action="store_true")
    parser.add_argument("--rayst3r_set_conf", type=float, default=2.5)
    parser.add_argument("--rayst3r_n_pred_views", type=int, default=5)
    parser.add_argument(
        "--rayst3r_filter_all_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rayst3r_tsdf", action="store_true")
    parser.add_argument("--rayst3r_voxel_size", type=float, default=0.002)
    parser.add_argument("--rayst3r_std_ratio", type=float, default=2.5)
    parser.add_argument("--rayst3r_poisson_depth", type=int, default=9)
    parser.add_argument("--rayst3r_poisson_scale", type=float, default=1.0)
    parser.add_argument("--rayst3r_density_quantile", type=float, default=0.02)
    parser.add_argument(
        "--instantmesh_flip", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--instantmesh_no_remove_bg", action="store_true")
    parser.add_argument("--icp_max_corr_ratio", type=float, default=0.05)
    parser.add_argument("--icp_voxel_size", type=float, default=0.005)
    parser.add_argument("--sample_points", type=int, default=50000)
    parser.add_argument("--rayst3r_crop_margin_ratio", type=float, default=0.05)
    parser.add_argument("--rayst3r_min_component_area_ratio", type=float, default=0.005)
    parser.add_argument("--rayst3r_target_faces", type=int, default=100000)
    parser.add_argument(
        "--fusion_mode",
        choices=["poisson", "visible_patch"],
        default="visible_patch",
        help=(
            "REAL275 main method keeps the InstantMesh topology as a prior "
            "backbone and replaces only the RGB-D-supported visible patch."
        ),
    )
    parser.add_argument("--fusion_voxel_size", type=float, default=0.003)
    parser.add_argument("--fusion_sample_points", type=int, default=15000)
    parser.add_argument("--fusion_poisson_depth", type=int, default=8)
    parser.add_argument(
        "--fusion_target_faces",
        type=int,
        default=120000,
        help=(
            "Approximate fused face cap. In visible_patch mode only the "
            "RaySt3R patch is simplified; InstantMesh backbone topology is retained."
        ),
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
        "--fusion_max_update_ratio",
        type=float,
        default=0.04,
        help="Maximum visible-surface vertex displacement as an object-diameter ratio.",
    )
    parser.add_argument(
        "--fusion_min_normal_cosine",
        type=float,
        default=0.15,
        help="Minimum absolute prior/RaySt3R normal cosine for a local update.",
    )
    parser.add_argument(
        "--fusion_smooth_iterations",
        type=int,
        default=3,
        help="Topology-neighbour smoothing iterations for bounded Phi displacement.",
    )
    parser.add_argument(
        "--fusion_min_update_fraction",
        type=float,
        default=0.01,
        help=(
            "Minimum prior-vertex fraction changed by visible-patch Phi before "
            "the fused mesh is admitted as a distinct candidate."
        ),
    )
    parser.add_argument(
        "--fusion_min_visible_depth_gain",
        type=float,
        default=0.0,
        help=(
            "Minimum anchor visible-depth residual improvement in metres for "
            "fused-candidate admission; this uses anchor RGB-D only."
        ),
    )
    parser.add_argument(
        "--fusion_global_scale_min",
        type=float,
        default=0.88,
        help="Lower bound for the no-GT Ray/prior envelope scale correction.",
    )
    parser.add_argument(
        "--fusion_global_scale_max",
        type=float,
        default=1.12,
        help="Upper bound for the no-GT Ray/prior envelope scale correction.",
    )
    parser.add_argument(
        "--fusion_cycle_in_reliability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablation only: include cycle confidence in fused anchor reliability. "
            "The REAL275 main method reports cycle error without using it."
        ),
    )
    parser.add_argument(
        "--rgbd_scale_recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Independently recover RaySt3R and InstantMesh metric scale and "
            "translation from anchor mask, depth and intrinsics."
        ),
    )
    parser.add_argument("--rgbd_scale_grid_steps", type=int, default=9)
    parser.add_argument("--rgbd_scale_multiplier_min", type=float, default=0.65)
    parser.add_argument("--rgbd_scale_multiplier_max", type=float, default=1.35)
    parser.add_argument("--rgbd_scale_z_offset_steps", type=int, default=5)
    parser.add_argument("--rgbd_scale_max_points", type=int, default=15000)
    parser.add_argument("--rgbd_scale_depth_tolerance", type=float, default=0.03)
    parser.add_argument("--rgbd_scale_support_ratio", type=float, default=0.08)
    parser.add_argument("--objects", type=str, default="", help="Comma-separated object names (empty=all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--register_iteration",
        type=int,
        default=5,
        help="FoundationPose register iterations used identically at anchor/query.",
    )
    parser.add_argument("--any6d_iter", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--any6d_refine", type=int, default=1, choices=[0, 1])

    parser.add_argument("--select_depth_weight", type=float, default=0.5)
    parser.add_argument("--select_mask_weight", type=float, default=0.25)
    parser.add_argument("--select_geom_weight", type=float, default=0.25)
    parser.add_argument("--anchor_c_base", type=float, default=1.0)
    parser.add_argument("--anchor_w_inlier", type=float, default=1.0)
    parser.add_argument("--anchor_w_rmse", type=float, default=1.0)
    parser.add_argument("--anchor_tau_inlier", type=float, default=0.05)
    parser.add_argument("--anchor_tau_rmse", type=float, default=0.02)
    parser.add_argument("--anchor_depth_scale", type=float, default=0.05)
    parser.add_argument("--anchor_mask_scale", type=float, default=0.50)
    parser.add_argument("--anchor_geom_scale", type=float, default=0.05)
    parser.add_argument("--anchor_calibration_weight", type=float, default=0.15)
    parser.add_argument("--score_alpha", type=float, default=0.4)
    parser.add_argument("--score_beta", type=float, default=0.6)
    parser.add_argument("--summary_filename", type=str, default="chamfer_distances.xlsx")
    args = parser.parse_args()

    anchor_scales = (
        args.anchor_depth_scale,
        args.anchor_mask_scale,
        args.anchor_geom_scale,
    )
    anchor_weights = (
        args.select_depth_weight,
        args.select_mask_weight,
        args.select_geom_weight,
    )
    if min(anchor_scales) <= 0:
        parser.error("anchor fixed scales must be positive")
    if min(anchor_weights) < 0 or sum(anchor_weights) <= 0:
        parser.error("anchor score weights must be non-negative and not all zero")
    if not 0.0 <= args.anchor_calibration_weight <= 1.0:
        parser.error("--anchor_calibration_weight must lie in [0, 1]")
    if args.register_iteration < 1:
        parser.error("--register_iteration must be positive")
    if args.rgbd_scale_grid_steps < 1 or args.rgbd_scale_z_offset_steps < 1:
        parser.error("RGB-D scale grid sizes must be positive")
    if args.rgbd_scale_multiplier_min <= 0.0:
        parser.error("--rgbd_scale_multiplier_min must be positive")
    if args.rgbd_scale_multiplier_min > args.rgbd_scale_multiplier_max:
        parser.error("RGB-D scale multiplier bounds are reversed")
    if not 0.0 < args.fusion_max_update_ratio <= 0.25:
        parser.error("--fusion_max_update_ratio must lie in (0, 0.25]")
    if not 0.0 <= args.fusion_min_normal_cosine < 1.0:
        parser.error("--fusion_min_normal_cosine must lie in [0, 1)")
    if args.fusion_smooth_iterations < 0:
        parser.error("--fusion_smooth_iterations must be non-negative")
    if not 0.0 <= args.fusion_min_update_fraction <= 1.0:
        parser.error("--fusion_min_update_fraction must lie in [0, 1]")
    if args.fusion_global_scale_min <= 0.0:
        parser.error("--fusion_global_scale_min must be positive")
    if args.fusion_global_scale_min > 1.0:
        parser.error("--fusion_global_scale_min must not exceed 1")
    if args.fusion_global_scale_max < 1.0:
        parser.error("--fusion_global_scale_max must be at least 1")
    if args.fusion_global_scale_min > args.fusion_global_scale_max:
        parser.error("fusion global scale bounds are reversed")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(torch.device(args.rayst3r_device))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    obj_list = sorted(d for d in os.listdir(args.anchor_folder) if os.path.isdir(os.path.join(args.anchor_folder, d)))
    if args.objects.strip():
        requested = {x.strip() for x in args.objects.split(",") if x.strip()}
        obj_list = [o for o in obj_list if o in requested]
    if not obj_list:
        raise RuntimeError(f"No object folders found in {args.anchor_folder}")
    result_root = os.path.realpath(args.output_root or args.anchor_folder)
    os.makedirs(result_root, exist_ok=True)

    cfg = ReconstructionConfig(
        anchor_root=args.anchor_folder,
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
        fusion_max_update_ratio=args.fusion_max_update_ratio,
        fusion_min_normal_cosine=args.fusion_min_normal_cosine,
        fusion_smooth_iterations=args.fusion_smooth_iterations,
        fusion_global_scale_min=args.fusion_global_scale_min,
        fusion_global_scale_max=args.fusion_global_scale_max,
        fusion_cycle_in_reliability=args.fusion_cycle_in_reliability,
        rgbd_scale_recovery=args.rgbd_scale_recovery,
        rgbd_scale_visible_sim3=False,
        rgbd_scale_min_margin=0.002,
        rgbd_rotation_min_margin=0.002,
        rgbd_scale_min_reciprocal_fraction=0.0,
        rgbd_scale_silhouette_weight=0.0,
        rgbd_scale_min_silhouette_quality=0.0,
        rgbd_scale_ambiguity_fallback=False,
        rgbd_scale_sim3_iterations=5,
        rgbd_scale_grid_steps=args.rgbd_scale_grid_steps,
        rgbd_scale_multiplier_min=args.rgbd_scale_multiplier_min,
        rgbd_scale_multiplier_max=args.rgbd_scale_multiplier_max,
        rgbd_scale_z_offset_steps=args.rgbd_scale_z_offset_steps,
        rgbd_scale_max_points=args.rgbd_scale_max_points,
        rgbd_scale_depth_tolerance=args.rgbd_scale_depth_tolerance,
        rgbd_scale_support_ratio=args.rgbd_scale_support_ratio,
        seed=args.seed,
    )
    recon_results = run_construction(cfg)
    _release_cuda_memory()
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.rayst3r_device))

    results = []
    pose_rows = []
    pose_rows_all = []
    candidate_registry = {}
    glctx = dr.RasterizeCudaContext(device=args.rayst3r_device)
    device_mismatch_fail_count = 0

    for obj in tqdm(obj_list, desc="Object"):
        if obj not in recon_results:
            print(f"Skip {obj}: no reconstruction output")
            continue
        input_obj_dir = os.path.join(args.anchor_folder, obj)
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
        color_bgr = cv2.imread(os.path.join(obj_dir, "color.png"))
        depth_raw = cv2.imread(os.path.join(obj_dir, "depth.png"), cv2.IMREAD_ANYDEPTH)
        mask_img = cv2.imread(os.path.join(obj_dir, "mask.png"), cv2.IMREAD_GRAYSCALE)
        k_path = os.path.join(obj_dir, "K.txt")
        gt_pose_path = os.path.join(obj_dir, f"{obj}_gt_pose.txt")
        if any(not os.path.exists(p) for p in [k_path, gt_pose_path]):
            print(f"Skip {obj}: missing required files")
            continue
        if color_bgr is None or depth_raw is None or mask_img is None:
            print(f"Skip {obj}: missing color/depth/mask image")
            continue

        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        reconstruction = recon_results[obj]
        meta_path = reconstruction.get("meta")
        recon_meta = {}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                recon_meta = json.load(handle)

        depth_unit_candidates = recon_meta.get("depth_unit_candidates", {})
        if isinstance(depth_unit_candidates, dict):
            mismatch_messages = [
                str(value.get("error", ""))
                for value in depth_unit_candidates.values()
                if isinstance(value, dict)
                and "Expected all tensors to be on the same device"
                in str(value.get("error", ""))
            ]
            if mismatch_messages:
                device_mismatch_fail_count += 1

        mesh_paths = {}
        instant_path = reconstruction.get("mesh_instant")
        ray_path = reconstruction.get("mesh_rayst3r")
        fused_path = reconstruction.get("mesh_fused")
        instantmesh_only = is_instantmesh_only(recon_meta)
        if instant_path and os.path.exists(instant_path):
            mesh_paths["instantmesh"] = instant_path
        if not instantmesh_only and ray_path and os.path.exists(ray_path):
            mesh_paths["rayst3r"] = ray_path

        fusion_admission = {
            "admitted": False,
            "uses_query": False,
            "uses_gt_pose_or_cad": False,
            "mask_source": "dataset_gt",
            "min_update_fraction": float(args.fusion_min_update_fraction),
            "min_visible_depth_gain": float(args.fusion_min_visible_depth_gain),
        }
        if not instantmesh_only and fused_path and os.path.exists(fused_path):
            fusion_diagnostics = recon_meta.get("fusion_diagnostics", {})
            update_fraction = float(
                fusion_diagnostics.get("prior_vertex_update_ratio", 0.0)
            )
            depth_gain = float(
                fusion_diagnostics.get("visible_depth_gain", float("nan"))
            )
            align_direction = str(recon_meta.get("align_direction", ""))
            align_pass = bool(
                not align_direction.startswith("none")
                and recon_meta.get("align_fitness", -1.0)
                >= args.anchor_tau_inlier
                and recon_meta.get("align_rmse", float("inf"))
                <= args.anchor_tau_rmse
            )
            quality_pass = bool(
                align_pass
                and update_fraction >= args.fusion_min_update_fraction
                and np.isfinite(depth_gain)
                and depth_gain >= args.fusion_min_visible_depth_gain
            )
            fused_ok = is_genuine_fusion(
                recon_meta, instantmesh_only=instantmesh_only
            )
            fusion_admission.update(
                {
                    "admitted": fused_ok,
                    "quality_pass": quality_pass,
                    "align_pass": align_pass,
                    "construction_output_valid": bool(
                        fusion_diagnostics.get("output_valid", False)
                    ),
                    "fallback_used": bool(
                        fusion_diagnostics.get("fallback_used", False)
                    ),
                    "prior_vertex_update_ratio": update_fraction,
                    "visible_depth_gain": (
                        depth_gain if np.isfinite(depth_gain) else None
                    ),
                }
            )
            if fused_ok:
                mesh_paths["fused"] = fused_path

        if not mesh_paths:
            print(f"Skip {obj}: no reconstruction candidate")
            continue
        candidate_meshes = {
            label: _as_trimesh(trimesh.load(path, process=False))
            for label, path in mesh_paths.items()
        }
        chosen_depth_unit = recon_meta.get("depth_unit", args.depth_unit)
        if chosen_depth_unit == "auto":
            chosen_depth_unit = _detect_depth_unit(depth_raw.astype(np.float32))
        depth = _convert_depth(depth_raw.astype(np.float32), chosen_depth_unit)
        mask = (mask_img > 0).astype(np.bool_)
        intrinsic = np.loadtxt(k_path)
        candidates = {}
        for label in ("rayst3r", "instantmesh", "fused"):
            if label not in candidate_meshes:
                continue
            registration_name = "foundationpose_register"
            trial_seed = _stable_trial_seed(
                args.seed, obj, label, registration_name, 0
            )
            estimator = None
            try:
                pred_pose_returned, estimator = _run_single_trial(
                    mesh_template=candidate_meshes[label],
                    save_path=obj_dir,
                    glctx=glctx,
                    args=args,
                    intrinsic=intrinsic,
                    color=color,
                    depth=depth,
                    mask=mask,
                    name=f"{obj}_{label}_{registration_name}",
                    random_seed=trial_seed,
                )
                l_depth, l_mask, l_geom = _compute_self_supervised_losses(
                    est=estimator,
                    depth=depth,
                    mask=mask,
                    k=intrinsic,
                    pred_pose=pred_pose_returned,
                    glctx=glctx,
                )
                pred_pose_exported, tf_to_centered = _exported_mesh_pose(
                    estimator, pred_pose_returned
                )
                est_mesh = estimator.mesh.copy()
                precenter_offset = np.asarray(
                    estimator.model_center, dtype=np.float64
                ).reshape(3)
            except Exception as exc:
                print(f"{obj} [{label}] anchor pose failed: {exc}")
                continue
            finally:
                if estimator is not None:
                    del estimator
                _release_cuda_memory()

            s_anchor_raw = _compute_anchor_calibration_score(
                meta=recon_meta,
                label=label,
                c_base=args.anchor_c_base,
                w_inlier=args.anchor_w_inlier,
                w_rmse=args.anchor_w_rmse,
                tau_inlier=args.anchor_tau_inlier,
                tau_rmse=args.anchor_tau_rmse,
            )
            mesh_out = os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj")
            pose_out = os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt")
            est_mesh.export(mesh_out)
            np.savetxt(pose_out, pred_pose_exported)
            np.savetxt(
                os.path.join(obj_dir, f"{obj}_ldepth_{label}.txt"), [l_depth]
            )
            np.savetxt(
                os.path.join(obj_dir, f"{obj}_lmask_{label}.txt"), [l_mask]
            )
            np.savetxt(
                os.path.join(obj_dir, f"{obj}_cd_{label}.txt"), [l_geom]
            )
            candidates[label] = {
                "base_geometry": label,
                "pred_pose": pred_pose_exported,
                "est_mesh": est_mesh,
                "mesh_path": mesh_out,
                "raw_mesh_path": os.path.abspath(mesh_paths[label]),
                "pose_path": pose_out,
                "l_depth": float(l_depth),
                "l_mask": float(l_mask),
                "l_geom": float(l_geom),
                "s_anchor": float(s_anchor_raw),
                "model_center": [0.0, 0.0, 0.0],
                "any6d_precenter_offset": precenter_offset.tolist(),
                "tf_to_centered": tf_to_centered.tolist(),
                "mesh_frame": "any6d_internal_centered",
                "pose_frame": "any6d_internal_centered_to_anchor_camera",
                "anchor_pose_strategy": registration_name,
                "anchor_pose_trial_count": 1,
                "anchor_pose_seed": int(trial_seed),
                "reconstruction_meta": recon_meta,
                "reconstruction_meta_path": meta_path,
                "fusion_admission": fusion_admission,
            }

        if not candidates:
            print(f"Skip {obj}: no valid candidate")
            continue

        anchor_rows = []
        for label, info in candidates.items():
            anchor_rows.append({
                "label": label,
                "s_anchor": float(info["s_anchor"]),
                "l_depth": float(info["l_depth"]),
                "l_mask": float(info["l_mask"]),
                "l_geom": float(info["l_geom"]),
            })
        derive_anchor_reliability_scores(
            anchor_rows,
            w_depth=args.select_depth_weight,
            w_mask=args.select_mask_weight,
            w_geom=args.select_geom_weight,
            depth_scale=args.anchor_depth_scale,
            mask_scale=args.anchor_mask_scale,
            geom_scale=args.anchor_geom_scale,
            calibration_weight=args.anchor_calibration_weight,
        )
        score_candidates(
            anchor_rows,
            w_depth=args.select_depth_weight,
            w_mask=args.select_mask_weight,
            w_geom=args.select_geom_weight,
            score_alpha=args.score_alpha,
            score_beta=args.score_beta,
            loss_normalization="fixed_scale",
            depth_scale=args.anchor_depth_scale,
            mask_scale=args.anchor_mask_scale,
            geom_scale=args.anchor_geom_scale,
        )
        scored_rows = sorted(anchor_rows, key=lambda row: row["total_score"], reverse=True)
        for row in scored_rows:
            info = candidates[row["label"]]
            info["s_anchor_raw"] = float(row["s_anchor_raw"])
            info["s_anchor"] = float(row["s_anchor"])
            info["anchor_observation_quality"] = float(row["anchor_observation_quality"])
            info["anchor_calibration_norm"] = float(row["anchor_calibration_norm"])
            info["selection_score"] = float(row["total_score"])
            info["s_query"] = float(row["s_query"])
            pose_rows_all.append(
                {
                    "Object": obj,
                    "Object_Number": _parse_object_number_from_source_frame(obj_dir),
                    "Candidate": row["label"],
                    "L_depth": info["l_depth"],
                    "L_mask": info["l_mask"],
                    "L_geom": info["l_geom"],
                    "S_anchor": info["s_anchor"],
                    "S_anchor_raw": info["s_anchor_raw"],
                    "Anchor_Observation_Quality": info["anchor_observation_quality"],
                    "Anchor_Calibration_Norm": info["anchor_calibration_norm"],
                    "L_depth_norm": row["l_depth_norm"],
                    "L_mask_norm": row["l_mask_norm"],
                    "L_geom_norm": row["l_geom_norm"],
                    "Selection_Score": float(row["total_score"]),
                }
            )
        chosen_label = scored_rows[0]["label"]

        chosen = candidates[chosen_label]
        recon_meta = chosen["reconstruction_meta"]
        meta_path = chosen["reconstruction_meta_path"]
        fusion_admission = chosen["fusion_admission"]
        np.savetxt(os.path.join(obj_dir, f"{obj}_initial_pose.txt"), chosen["pred_pose"])
        chosen["est_mesh"].export(os.path.join(obj_dir, f"final_mesh_{obj}.obj"))
        np.savetxt(os.path.join(obj_dir, f"{obj}_cd.txt"), [chosen["l_geom"]])
        np.savetxt(os.path.join(obj_dir, f"{obj}_model_center.txt"), np.asarray(chosen["model_center"], dtype=np.float64))

        results.append(
            {
                "Object": obj,
                "Object_Number": _parse_object_number_from_source_frame(obj_dir),
                "Chamfer_Distance": float(chosen["l_geom"]),
                "Chosen_Mesh": chosen_label,
            }
        )
        pose_rows.append(
            {
                "Object": obj,
                "Object_Number": _parse_object_number_from_source_frame(obj_dir),
                "Chosen_Mesh": chosen_label,
                "L_depth": chosen["l_depth"],
                "L_mask": chosen["l_mask"],
                "L_geom": chosen["l_geom"],
                "S_anchor": chosen["s_anchor"],
                "Selection_Score": chosen["selection_score"],
            }
        )

        obj_candidates_info = {}
        for label, info in candidates.items():
            base_geometry = info.get("base_geometry", label)
            info_scale_recovery = info.get("reconstruction_meta", {}).get(
                "scale_recovery", {}
            )
            if base_geometry == "fused":
                candidate_scale_meta = {
                    "rayst3r": info_scale_recovery.get("rayst3r", {}),
                    "instantmesh": info_scale_recovery.get("instantmesh", {}),
                }
            else:
                candidate_scale_meta = info_scale_recovery.get(base_geometry, {})
            obj_candidates_info[label] = {
                "base_geometry": base_geometry,
                "mesh_path": os.path.abspath(info["mesh_path"]),
                "raw_mesh_path": info["raw_mesh_path"],
                "pose_path": os.path.abspath(info["pose_path"]),
                "source_provenance": f"reconstruction:{base_geometry}",
                "is_fallback_alias": False,
                "s_anchor": float(info["s_anchor"]),
                "anchor_score_version": "bounded_observation_calibrated_v1",
                "s_anchor_raw": float(info["s_anchor_raw"]),
                "anchor_observation_quality": float(info["anchor_observation_quality"]),
                "anchor_calibration_norm": float(info["anchor_calibration_norm"]),
                "l_depth": float(info["l_depth"]),
                "l_mask": float(info["l_mask"]),
                "l_geom": float(info["l_geom"]),
                "selection_score": float(info["selection_score"]),
                "model_center": info["model_center"],
                "scale_recovery": candidate_scale_meta,
                "anchor_pose_strategy": info.get("anchor_pose_strategy", "recovered"),
                "anchor_pose_trial_count": int(info.get("anchor_pose_trial_count", 1)),
                "anchor_pose_seed": info.get("anchor_pose_seed"),
                "mesh_frame": info.get("mesh_frame", "legacy_unknown"),
                "pose_frame": info.get("pose_frame", "legacy_unknown"),
                "tf_to_centered": info.get("tf_to_centered"),
                "candidate_local_symmetry_tfs": [np.eye(4).tolist()],
                "candidate_local_symmetry_source": (
                    "identity_no_estimation_adjustment"
                ),
                "any6d_precenter_offset": info.get(
                    "any6d_precenter_offset", [0.0, 0.0, 0.0]
                ),
                "geometry_units": "meters",
                "mesh_vertices": int(len(info["est_mesh"].vertices)),
                "mesh_faces": int(len(info["est_mesh"].faces)),
                "mesh_extents": np.asarray(info["est_mesh"].extents, dtype=np.float64).tolist(),
            }
        candidate_registry[obj] = {
            "chosen": chosen_label,
            "anchor_score_version": "bounded_observation_calibrated_v1",
            "mask_source": "dataset_gt",
            "geometry_units": "meters_after_rgbd_self_supervised_scale_recovery",
            "candidate_mesh_frame": "any6d_internal_centered",
            "scale_metadata": {
                "mode": "rgbd_extent_depth_single_scale",
                "depth_unit": recon_meta.get("depth_unit", chosen_depth_unit),
                "rayst3r_metric_scale": recon_meta.get("rayst3r_metric_scale", 1.0),
                "instantmesh_metric_scale": recon_meta.get(
                    "instantmesh_metric_scale", 1.0
                ),
                "rgbd_self_supervised": recon_meta.get("scale_recovery", {}),
                "construction_transform": recon_meta.get("transform"),
                "mesh_statistics": recon_meta.get("mesh_statistics", {}),
            },
            "fusion_diagnostics": recon_meta.get("fusion_diagnostics", {}),
            "fusion_metadata": recon_meta.get("fusion_diagnostics", {}),
            "fusion_admission": fusion_admission,
            "bidirectional": recon_meta.get("bidirectional", {}),
            "bidirectional_metadata": recon_meta.get("bidirectional", {}),
            "alignment_metadata": {
                "selection_rule": "best_of_two",
                "strict_both_gate": bool(args.align_strict_both),
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
            "instantmesh_metric_scale": recon_meta.get("instantmesh_metric_scale", 1.0),
            "reconstruction_meta_path": (
                os.path.abspath(meta_path) if meta_path else None
            ),
            "candidates": obj_candidates_info,
            "final_mesh_path": os.path.abspath(os.path.join(obj_dir, f"final_mesh_{obj}.obj")),
            "final_pose_path": os.path.abspath(os.path.join(obj_dir, f"{obj}_initial_pose.txt")),
            "K_path": os.path.abspath(k_path),
        }
        _release_cuda_memory()

    if not results:
        print("No objects processed.")
        return

    registry_path = os.path.join(result_root, "candidate_registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(candidate_registry, f, indent=2)
    print(f"Candidate registry saved to: {registry_path}")

    run_meta = {
        "anchor_folder": os.path.realpath(args.anchor_folder),
        "output_root": result_root,
        "mask_source": "dataset_gt",
        "geometry_units": "meters_after_rgbd_self_supervised_scale_recovery",
        "appearance_contract": {
            "anchor": "native_reconstruction_colors",
            "query": "anchor_prior_hybrid",
            "geometry_changed": False,
        },
        "reconstruction_method": "rgbd_extent_depth_single_scale",
        "frame_contract": {
            "mesh_frame": "any6d_internal_centered",
            "pose_frame": "any6d_internal_centered_to_anchor_camera",
            "registration": "FoundationPose.register shared by anchor and query",
            "outer_axis_alignment_selection": False,
        },
        "gt_usage": {
            "anchor_candidate_construction": False,
            "anchor_candidate_scoring": False,
            "gt_cad_pose_audit_enabled": False,
            "gt_anchor_pose_copied_for_query_evaluation_only": True,
        },
        "bidirectional_alignment": {
            "selection_rule": "best_of_two",
            "strict_both_gate": bool(args.align_strict_both),
            "cycle_role": (
                "bounded_reliability_ablation"
                if args.fusion_cycle_in_reliability
                else "diagnostic_only"
            ),
            "icp_hypotheses_per_direction": int(args.icp_multi_hypo),
            "reciprocal_pairing": bool(args.icp_reciprocal_pairing),
            "cycle_definition": (
                "selected_transform_inverse_refinement_round_trip"
                if args.icp_reciprocal_pairing
                else "independent_directional_optima"
            ),
        },
        "selection": {
            "select_depth_weight": args.select_depth_weight,
            "select_mask_weight": args.select_mask_weight,
            "select_geom_weight": args.select_geom_weight,
            "anchor_c_base": args.anchor_c_base,
            "anchor_w_inlier": args.anchor_w_inlier,
            "anchor_w_rmse": args.anchor_w_rmse,
            "anchor_tau_inlier": args.anchor_tau_inlier,
            "anchor_tau_rmse": args.anchor_tau_rmse,
            "anchor_depth_scale": args.anchor_depth_scale,
            "anchor_mask_scale": args.anchor_mask_scale,
            "anchor_geom_scale": args.anchor_geom_scale,
            "anchor_calibration_weight": args.anchor_calibration_weight,
            "anchor_score_version": "bounded_observation_calibrated_v1",
            "loss_normalization": "fixed_scale",
            "score_alpha": args.score_alpha,
            "score_beta": args.score_beta,
        },
        "provenance": runtime_provenance(os.path.dirname(__file__)),
        "args": vars(args),
    }
    with open(os.path.join(result_root, "anchor_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    df = pd.DataFrame(results).sort_values("Object")
    excel_path = os.path.join(result_root, args.summary_filename)
    df.to_excel(excel_path, index=False)

    pose_df = pd.DataFrame(pose_rows).sort_values("Object")
    pose_all_df = pd.DataFrame(pose_rows_all).sort_values(["Object", "Candidate"])
    pose_excel_path = os.path.join(result_root, "pose_metrics.xlsx")
    with pd.ExcelWriter(pose_excel_path) as writer:
        pose_df.to_excel(writer, index=False, sheet_name="chosen")
        pose_all_df.to_excel(writer, index=False, sheet_name="candidates")

    print(df["Chamfer_Distance"].describe())
    print(f"Results saved to: {excel_path}")
    print(f"Pose metrics saved to: {pose_excel_path}")
    if device_mismatch_fail_count > 0:
        print(
            "[WARN] Detected RaySt3R device-mismatch failures in reconstruction meta. "
            "If you use CUDA_VISIBLE_DEVICES, prefer --rayst3r_device cuda:0 (mapped first visible GPU), "
            "or set both --rayst3r_device and --instantmesh_device to the same visible device."
        )


if __name__ == "__main__":
    main()
