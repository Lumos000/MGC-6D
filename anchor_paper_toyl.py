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
import re

import cv2
import nvdiffrast.torch as dr
import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

from construction import ReconstructionConfig, run_construction
from estimater import Any6D
from foundationpose.Utils import align_mesh_to_coordinate
from candidate_selection import (
    derive_anchor_reliability_scores,
    score_candidates,
)
from candidate_admission import is_genuine_fusion, is_instantmesh_only
from revision_protocol import runtime_provenance


MM_TO_M = 0.001


def _release_cuda_memory():
    """Release cyclic Any6D references and unused CUDA allocations between trials."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        with torch.cuda.device(device_idx):
            torch.cuda.empty_cache()


def _parse_obj_id(obj_name: str) -> int | None:
    m = re.match(r"^obj_(\d+)$", obj_name)
    if m:
        return int(m.group(1))
    return None


def _stable_anchor_seed(global_seed: int, obj: str, label: str) -> int:
    """One deterministic pose-registration seed per object/candidate."""

    key = f"{obj}|{label}|single_anchor_registration".encode("utf-8")
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
        mask = np.ones(depth.shape, dtype=np.bool_)

    intrinsic = _load_intrinsics(search_dirs)
    return color, depth, mask, intrinsic, depth_unit


def _backproject_depth_to_points(depth: np.ndarray, K: np.ndarray, valid_mask: np.ndarray, max_points: int = 5000) -> np.ndarray:
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


def _compute_self_supervised_losses(est, depth: np.ndarray, mask: np.ndarray, K: np.ndarray, pred_pose: np.ndarray, glctx) -> tuple[float, float, float]:
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
        if len(rendered) >= 2:
            rendered_depth = rendered[1]
    elif hasattr(rendered, "shape") and rendered.shape[-1] > 3:
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


def _load_trimesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [
            geometry
            for geometry in mesh.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No triangle mesh in {path}")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Invalid or empty mesh: {path}")
    return mesh


def _compute_anchor_calibration_score(meta: dict, label: str, c_base: float, w_inlier: float, w_rmse: float, tau_inlier: float, tau_rmse: float) -> float:
    scale_meta = meta.get("scale_recovery", {})
    if label == "fused":
        scale_entries = [scale_meta.get("rayst3r", {}), scale_meta.get("instantmesh", {})]
        scale_values = [
            (
                float(entry.get("reliability", 0.0))
                if bool(entry.get("valid", False))
                else 0.0
            )
            for entry in scale_entries
            if bool(entry.get("enabled", False))
        ]
        scale_reliability = min(scale_values) if scale_values else 1.0
    else:
        scale_entry = scale_meta.get(label, {})
        scale_reliability = (
            (
                float(scale_entry.get("reliability", 0.0))
                if bool(scale_entry.get("valid", False))
                else 0.0
            )
            if bool(scale_entry.get("enabled", False))
            else 1.0
        )
    # Scale recovery is self-supervised and contributes continuously.  It is
    # deliberately bounded to 20% so it cannot become an undisclosed gate.
    scale_factor = float(np.clip(0.8 + 0.2 * scale_reliability, 0.8, 1.0))
    if label in ("rayst3r", "instantmesh"):
        return float(c_base) * scale_factor
    fitness = meta.get("align_fitness", -1.0)
    rmse = meta.get("align_rmse", float("inf"))
    fitness_ok = fitness >= tau_inlier
    if label == "fused" and fitness_ok:
        weight_sum = max(float(w_inlier) + float(w_rmse), 1e-8)
        rmse_quality = float(np.exp(-max(float(rmse), 0.0) / max(float(tau_rmse), 1e-8)))
        alignment_quality = (
            float(w_inlier) * float(np.clip(fitness, 0.0, 1.0))
            + float(w_rmse) * rmse_quality
        ) / weight_sum
        fusion_quality = float(
            meta.get("fusion_diagnostics", {}).get("reliability", alignment_quality)
        )
        return float(c_base) * scale_factor * float(
            np.clip(0.5 * alignment_quality + 0.5 * fusion_quality, 0.0, 1.0)
        )
    return c_base


def main():
    parser = argparse.ArgumentParser(description="TOYL anchor phase with paper multi-candidate logic")
    parser.add_argument("--anchor_folder", type=str, required=True)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--obj_ids", type=str, default="", help="Comma-separated ids, empty=all")
    parser.add_argument("--depth_unit", type=str, default="auto", choices=["auto", "mm", "rayst3r_uint16", "meters"])
    parser.add_argument("--depth_unit_try_both", action="store_true")
    parser.add_argument("--depth_preprocess", action="store_true")
    parser.add_argument("--depth_filter_radius", type=int, default=2)
    parser.add_argument("--depth_filter_device", type=str, default="cuda")
    parser.add_argument("--rayst3r_device", type=str, default="cuda:0")
    parser.add_argument("--rayst3r_checkpoint", type=str, default=None)
    parser.add_argument("--instantmesh_device", type=str, default="cuda:1")
    parser.add_argument(
        "--instantmesh_flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--align_use_guess_translation", action="store_true")
    parser.add_argument("--align_bidirectional_icp", action="store_true")
    parser.add_argument(
        "--align_strict_both",
        action="store_true",
        help="Ablation: require both ICP directions to pass Fitness/RMSE thresholds",
    )
    parser.add_argument("--icp_max_iter", type=int, default=50)
    parser.add_argument("--icp_fitness_threshold", type=float, default=0.05)
    parser.add_argument(
        "--icp_multi_hypo",
        type=int,
        default=1,
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
    )
    parser.add_argument("--fusion_voxel_size", type=float, default=0.003)
    parser.add_argument("--fusion_sample_points", type=int, default=15000)
    parser.add_argument("--fusion_poisson_depth", type=int, default=8)
    parser.add_argument("--fusion_target_faces", type=int, default=20000)
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
    parser.add_argument("--fusion_max_update_ratio", type=float, default=0.04)
    parser.add_argument("--fusion_min_normal_cosine", type=float, default=0.15)
    parser.add_argument("--fusion_smooth_iterations", type=int, default=3)
    parser.add_argument("--fusion_global_scale_min", type=float, default=0.88)
    parser.add_argument("--fusion_global_scale_max", type=float, default=1.12)
    parser.add_argument(
        "--fusion_min_update_fraction",
        type=float,
        default=0.01,
        help=(
            "Admit Phi only when at least this fraction of prior vertices was "
            "actually updated by the visible RGB-D patch."
        ),
    )
    parser.add_argument(
        "--fusion_min_visible_depth_gain",
        type=float,
        default=0.0,
        help=(
            "Admit Phi only when its anchor-visible depth loss improves by at "
            "least this amount over the unmodified prior."
        ),
    )
    parser.add_argument(
        "--fusion_cycle_in_reliability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ablation only; the main run reports cycle error without scoring it.",
    )
    parser.add_argument(
        "--rgbd_scale_recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Independently recover RaySt3R/InstantMesh scale from anchor mask+depth.",
    )
    parser.add_argument("--rgbd_scale_grid_steps", type=int, default=9)
    parser.add_argument("--rgbd_scale_multiplier_min", type=float, default=0.65)
    parser.add_argument("--rgbd_scale_multiplier_max", type=float, default=1.35)
    parser.add_argument("--rgbd_scale_z_offset_steps", type=int, default=5)
    parser.add_argument("--rgbd_scale_max_points", type=int, default=15000)
    parser.add_argument("--rgbd_scale_depth_tolerance", type=float, default=0.03)
    parser.add_argument("--rgbd_scale_support_ratio", type=float, default=0.08)
    parser.add_argument(
        "--rgbd_scale_visible_sim3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use robust reciprocal visible-surface Sim(3) scale recovery.",
    )
    parser.add_argument("--rgbd_scale_min_margin", type=float, default=0.002)
    parser.add_argument("--rgbd_rotation_min_margin", type=float, default=0.002)
    parser.add_argument(
        "--rgbd_scale_min_reciprocal_fraction", type=float, default=0.0
    )
    parser.add_argument("--rgbd_scale_silhouette_weight", type=float, default=0.0)
    parser.add_argument(
        "--rgbd_scale_min_silhouette_quality", type=float, default=0.0
    )
    parser.add_argument(
        "--rgbd_scale_ambiguity_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reject ambiguous visible Sim(3) estimates: preserve native metric "
            "RaySt3R or use observed-depth extent for InstantMesh."
        ),
    )
    parser.add_argument("--rgbd_scale_sim3_iterations", type=int, default=5)
    parser.add_argument(
        "--align_relaxed_rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional disclosed rescue ablation; disabled in the formal method.",
    )
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
    parser.add_argument("--candidate_canonicalization_scale", type=float, default=1.0)
    parser.add_argument("--score_alpha", type=float, default=0.4)
    parser.add_argument("--score_beta", type=float, default=0.6)
    parser.add_argument("--any6d_iter", type=int, default=5)
    parser.add_argument("--any6d_refine", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0.0 <= args.anchor_calibration_weight <= 1.0:
        raise ValueError("--anchor_calibration_weight must lie in [0, 1]")
    if args.fusion_min_extent_ratio > args.fusion_max_extent_ratio:
        raise ValueError("fusion extent ratio bounds are reversed")
    if args.rgbd_scale_grid_steps < 1 or args.rgbd_scale_z_offset_steps < 1:
        raise ValueError("RGB-D scale grid sizes must be positive")
    if args.rgbd_scale_multiplier_min <= 0.0:
        raise ValueError("--rgbd_scale_multiplier_min must be positive")
    if args.rgbd_scale_multiplier_min > args.rgbd_scale_multiplier_max:
        raise ValueError("RGB-D scale multiplier bounds are reversed")
    if args.rgbd_scale_min_margin < 0.0:
        raise ValueError("--rgbd_scale_min_margin must be non-negative")
    if args.rgbd_rotation_min_margin < 0.0:
        raise ValueError("--rgbd_rotation_min_margin must be non-negative")
    if not 0.0 <= args.rgbd_scale_min_reciprocal_fraction <= 1.0:
        raise ValueError("--rgbd_scale_min_reciprocal_fraction must lie in [0, 1]")
    if not 0.0 <= args.rgbd_scale_silhouette_weight <= 0.5:
        raise ValueError("--rgbd_scale_silhouette_weight must lie in [0, 0.5]")
    if not 0.0 <= args.rgbd_scale_min_silhouette_quality <= 1.0:
        raise ValueError("--rgbd_scale_min_silhouette_quality must lie in [0, 1]")
    if args.rgbd_scale_sim3_iterations < 1:
        raise ValueError("--rgbd_scale_sim3_iterations must be positive")
    if not np.isfinite(args.candidate_canonicalization_scale) or args.candidate_canonicalization_scale <= 0.0:
        raise ValueError("--candidate_canonicalization_scale must be finite and positive")
    if not 0.0 < args.fusion_max_update_ratio <= 0.25:
        raise ValueError("--fusion_max_update_ratio must lie in (0, 0.25]")
    if not 0.0 <= args.fusion_min_normal_cosine < 1.0:
        raise ValueError("--fusion_min_normal_cosine must lie in [0, 1)")
    if args.fusion_smooth_iterations < 0:
        raise ValueError("--fusion_smooth_iterations must be non-negative")
    if not 0.0 <= args.fusion_min_update_fraction <= 1.0:
        raise ValueError("--fusion_min_update_fraction must lie in [0, 1]")
    if not np.isfinite(args.fusion_min_visible_depth_gain):
        raise ValueError("--fusion_min_visible_depth_gain must be finite")
    for name in (
        "anchor_depth_scale",
        "anchor_mask_scale",
        "anchor_geom_scale",
        "fusion_voxel_size",
        "fusion_visible_depth_tolerance",
        "fusion_observed_support_ratio",
        "fusion_agreement_ratio",
        "rgbd_scale_depth_tolerance",
        "rgbd_scale_support_ratio",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"--{name} must be finite and positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    requested_ids = None
    requested_object_names = None
    if args.obj_ids.strip():
        requested_ids = {int(x.strip()) for x in args.obj_ids.split(",") if x.strip()}
        requested_object_names = tuple(
            sorted(
                name
                for name in os.listdir(args.anchor_folder)
                if os.path.isdir(os.path.join(args.anchor_folder, name))
                and _parse_obj_id(name) in requested_ids
            )
        )
        if not requested_object_names:
            raise ValueError(
                f"--obj_ids={args.obj_ids!r} did not match any object directory "
                f"under {args.anchor_folder}"
            )

    cfg = ReconstructionConfig(
        anchor_root=args.anchor_folder,
        output_root=args.output_root,
        object_names=requested_object_names,
        depth_unit=args.depth_unit,
        depth_unit_try_both=args.depth_unit_try_both,
        depth_preprocess=args.depth_preprocess,
        depth_filter_radius=args.depth_filter_radius,
        depth_filter_device=args.depth_filter_device,
        rayst3r_device=args.rayst3r_device,
        rayst3r_checkpoint=args.rayst3r_checkpoint,
        instantmesh_device=args.instantmesh_device,
        instantmesh_flip=args.instantmesh_flip,
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
        rgbd_scale_visible_sim3=args.rgbd_scale_visible_sim3,
        rgbd_scale_min_margin=args.rgbd_scale_min_margin,
        rgbd_rotation_min_margin=args.rgbd_rotation_min_margin,
        rgbd_scale_min_reciprocal_fraction=(
            args.rgbd_scale_min_reciprocal_fraction
        ),
        rgbd_scale_silhouette_weight=args.rgbd_scale_silhouette_weight,
        rgbd_scale_min_silhouette_quality=(
            args.rgbd_scale_min_silhouette_quality
        ),
        rgbd_scale_ambiguity_fallback=args.rgbd_scale_ambiguity_fallback,
        rgbd_scale_sim3_iterations=args.rgbd_scale_sim3_iterations,
        rgbd_scale_grid_steps=args.rgbd_scale_grid_steps,
        rgbd_scale_multiplier_min=args.rgbd_scale_multiplier_min,
        rgbd_scale_multiplier_max=args.rgbd_scale_multiplier_max,
        rgbd_scale_z_offset_steps=args.rgbd_scale_z_offset_steps,
        rgbd_scale_max_points=args.rgbd_scale_max_points,
        rgbd_scale_depth_tolerance=args.rgbd_scale_depth_tolerance,
        rgbd_scale_support_ratio=args.rgbd_scale_support_ratio,
        seed=args.seed,
    )

    result_root = os.path.realpath(args.output_root or args.anchor_folder)
    os.makedirs(result_root, exist_ok=True)
    recon_results = run_construction(cfg)
    _release_cuda_memory()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.rayst3r_device)

    candidate_frame = "any6d_internal_centered"
    anchor_pose_method = "single_any6d_coarse_axis_aligned"

    run_meta = {
        "anchor_folder": args.anchor_folder,
        "output_root": args.output_root,
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
        "geometry_units": "meters_after_construction",
        "candidate_canonicalization_scale": args.candidate_canonicalization_scale,
        "candidate_mesh_frame": candidate_frame,
        "reconstruction_method": "anchor_visible_robust_sim3_topology_phi_v7_ray_frame",
        "anchor_pose_method": anchor_pose_method,
        "provenance": runtime_provenance(os.path.dirname(__file__)),
        "args": vars(args),
    }
    with open(os.path.join(result_root, "anchor_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    results = []
    pose_rows = []
    pose_rows_all = []
    candidate_registry = {}

    obj_list = sorted([d for d in os.listdir(args.anchor_folder) if os.path.isdir(os.path.join(args.anchor_folder, d))])
    glctx = dr.RasterizeCudaContext(device=args.rayst3r_device)

    for obj in tqdm(obj_list, desc="Object"):
        obj_id = _parse_obj_id(obj)
        if obj_id is None:
            continue
        if requested_ids is not None and obj_id not in requested_ids:
            continue
        if obj not in recon_results:
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
        meta_path = recon_results[obj].get("meta")
        recon_meta = {}
        chosen_depth_unit = args.depth_unit
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                recon_meta = json.load(f)
            chosen_depth_unit = recon_meta.get("depth_unit", args.depth_unit)

        color, depth, mask, intrinsic, _ = _load_eval_inputs(
            input_obj_dir, chosen_depth_unit
        )

        ray_path = recon_results[obj].get("mesh_rayst3r")
        inst_path = recon_results[obj].get("mesh_instant")
        fused_path = recon_results[obj].get("mesh_fused")
        candidates = {}

        align_direction = recon_meta.get("align_direction", None)
        instantmesh_only = is_instantmesh_only(recon_meta)
        align_valid = (
            align_direction is not None
            and not str(align_direction).startswith("none")
            and not instantmesh_only
        )
        # Never expose fallback copies as independent RaySt3R/Phi families.
        if not instantmesh_only and ray_path and os.path.exists(ray_path):
            candidates["rayst3r"] = ray_path
        if inst_path and os.path.exists(inst_path):
            candidates["instantmesh"] = inst_path
        fusion_admission = {
            "admitted": False,
            "uses_query": False,
            "uses_gt_pose_or_cad": False,
            "mask_source": "dataset_gt",
            "min_update_fraction": float(args.fusion_min_update_fraction),
            "min_visible_depth_gain": float(args.fusion_min_visible_depth_gain),
        }
        if align_valid and fused_path and os.path.exists(fused_path):
            fusion_diag = recon_meta.get("fusion_diagnostics", {})
            fusion_update_fraction = float(
                fusion_diag.get("prior_vertex_update_ratio", 0.0)
            )
            fusion_depth_gain = float(
                fusion_diag.get("visible_depth_gain", float("nan"))
            )
            quality_pass = bool(
                recon_meta.get("align_fitness", -1.0) >= args.anchor_tau_inlier
                and recon_meta.get("align_rmse", float("inf")) <= args.anchor_tau_rmse
                and fusion_update_fraction >= args.fusion_min_update_fraction
                and np.isfinite(fusion_depth_gain)
                and fusion_depth_gain >= args.fusion_min_visible_depth_gain
            )
            # Quality terms are reliability evidence, not an admission gate.
            # Only invalid construction outputs and fallback clones are absent.
            fused_ok = is_genuine_fusion(
                recon_meta, instantmesh_only=instantmesh_only
            )
            fusion_admission.update(
                {
                    "admitted": bool(fused_ok),
                    "quality_pass": quality_pass,
                    "align_pass": bool(
                        recon_meta.get("align_fitness", -1.0)
                        >= args.anchor_tau_inlier
                        and recon_meta.get("align_rmse", float("inf"))
                        <= args.anchor_tau_rmse
                    ),
                    "construction_output_valid": bool(
                        fusion_diag.get("output_valid", False)
                    ),
                    "fallback_used": bool(fusion_diag.get("fallback_used", False)),
                    "prior_vertex_update_ratio": fusion_update_fraction,
                    "visible_depth_gain": (
                        fusion_depth_gain if np.isfinite(fusion_depth_gain) else None
                    ),
                }
            )
            if fused_ok:
                candidates["fused"] = fused_path

        if not candidates:
            continue

        candidate_meshes = {}
        for label, path in candidates.items():
            mesh = align_mesh_to_coordinate(_load_trimesh(path))
            if args.candidate_canonicalization_scale != 1.0:
                mesh.apply_scale(args.candidate_canonicalization_scale)
            candidate_meshes[label] = mesh

        s_anchor_scores = {
            label: _compute_anchor_calibration_score(
                meta=recon_meta,
                label=label,
                c_base=args.anchor_c_base,
                w_inlier=args.anchor_w_inlier,
                w_rmse=args.anchor_w_rmse,
                tau_inlier=args.anchor_tau_inlier,
                tau_rmse=args.anchor_tau_rmse,
            )
            for label in candidates
        }
        def _evaluate_mesh_pose(mesh_local_raw: trimesh.Trimesh, label: str):
            mesh_local_raw = mesh_local_raw.copy()
            mesh_local_raw.export(
                os.path.join(obj_dir, f"center_mesh_{obj}_{label}.obj")
            )
            registration_seed = _stable_anchor_seed(args.seed, obj, label)
            random.seed(registration_seed)
            np.random.seed(registration_seed)
            torch.manual_seed(registration_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(registration_seed)

            est_local = None
            try:
                est_local = Any6D(
                    symmetry_tfs=None,
                    mesh=mesh_local_raw,
                    debug_dir=obj_dir,
                    debug=0,
                )
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
                best_mesh = est_local.mesh.copy()
                tf_to_centered = est_local.get_tf_to_centered_mesh()
                if torch.is_tensor(tf_to_centered):
                    tf_to_centered = tf_to_centered.detach().cpu().numpy()
                tf_to_centered = np.asarray(
                    tf_to_centered, dtype=np.float64
                ).reshape(4, 4)
            finally:
                if est_local is not None:
                    del est_local
                _release_cuda_memory()

            pred_pose_local = np.asarray(pred_pose_local, dtype=np.float64).reshape(4, 4)
            np.savetxt(os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt"), pred_pose_local)
            best_mesh.export(os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj"))
            np.savetxt(os.path.join(obj_dir, f"{obj}_ldepth_{label}.txt"), [l_depth])
            np.savetxt(os.path.join(obj_dir, f"{obj}_lmask_{label}.txt"), [l_mask])
            np.savetxt(os.path.join(obj_dir, f"{obj}_cd_{label}.txt"), [l_geom])
            return {
                "mesh": best_mesh,
                "pred_pose": pred_pose_local,
                "l_depth": float(l_depth),
                "l_mask": float(l_mask),
                "l_geom": float(l_geom),
                "anchor_pose_strategy": anchor_pose_method,
                "anchor_pose_random_seed": int(registration_seed),
                "anchor_pose_trial_count": 1,
                "mesh_frame": candidate_frame,
                "pose_frame": candidate_frame,
                "tf_to_centered": tf_to_centered.tolist(),
            }

        results_pose = {}
        for label in candidates:
            results_pose[label] = _evaluate_mesh_pose(
                candidate_meshes[label], label
            )
            res = results_pose[label]
            pose_rows_all.append(
                {
                    "Object": obj,
                    "Object_Number": obj_id,
                    "Candidate": label,
                    "L_depth": res["l_depth"],
                    "L_mask": res["l_mask"],
                    "L_geom": res["l_geom"],
                    "S_anchor": float(s_anchor_scores[label]),
                }
            )
        if not results_pose:
            continue

        anchor_rows = []
        for label, res in results_pose.items():
            anchor_rows.append({
                "label": label,
                "s_anchor": float(s_anchor_scores[label]),
                "l_depth": float(res["l_depth"]),
                "l_mask": float(res["l_mask"]),
                "l_geom": float(res["l_geom"]),
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
        anchor_row_by_label = {row["label"]: row for row in anchor_rows}
        s_anchor_scores = {
            row["label"]: float(row["s_anchor"]) for row in anchor_rows
        }
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
        chosen_label = scored_rows[0]["label"]
        best = results_pose[chosen_label]
        selection_scores = {row["label"]: row["total_score"] for row in scored_rows}
        score_breakdown = {
            row["label"]: {
                "Score_L_depth": args.select_depth_weight * row["l_depth_norm"],
                "Score_L_mask": args.select_mask_weight * row["l_mask_norm"],
                "Score_L_geom": args.select_geom_weight * row["l_geom_norm"],
                "S_query": row["s_query"],
                "S_anchor": row["s_anchor"],
                "S_anchor_norm": row["s_anchor_norm"],
                "Total_Score": row["total_score"],
            }
            for row in scored_rows
        }

        for i, row in enumerate(pose_rows_all):
            if row["Object"] == obj and row["Candidate"] in selection_scores:
                pose_rows_all[i]["Selection_Score"] = float(selection_scores[row["Candidate"]])
                pose_rows_all[i].update(score_breakdown[row["Candidate"]])
                bounded = anchor_row_by_label[row["Candidate"]]
                pose_rows_all[i]["S_anchor_raw"] = bounded["s_anchor_raw"]
                pose_rows_all[i]["Anchor_Observation_Quality"] = bounded[
                    "anchor_observation_quality"
                ]
                pose_rows_all[i]["Anchor_Calibration_Norm"] = bounded[
                    "anchor_calibration_norm"
                ]

        chosen_pose_path = os.path.join(obj_dir, f"{obj}_initial_pose_{chosen_label}.txt")
        chosen_mesh_path = os.path.join(obj_dir, f"final_mesh_{obj}_{chosen_label}.obj")
        pred_pose_final = np.loadtxt(chosen_pose_path)
        final_mesh = trimesh.load(chosen_mesh_path)

        np.savetxt(os.path.join(obj_dir, f"{obj}_initial_pose.txt"), pred_pose_final)
        final_mesh.export(os.path.join(obj_dir, f"final_mesh_{obj}.obj"))
        np.savetxt(os.path.join(obj_dir, f"{obj}_cd.txt"), [best["l_geom"]])

        results.append({"Object": obj, "Object_Number": obj_id, "Chamfer_Distance": float(best["l_geom"])})
        pose_rows.append(
            {
                "Object": obj,
                "Object_Number": obj_id,
                "Chosen_Mesh": chosen_label,
                "L_depth": float(best["l_depth"]),
                "L_mask": float(best["l_mask"]),
                "L_geom": float(best["l_geom"]),
                "S_anchor": float(s_anchor_scores[chosen_label]),
                "Selection_Score": float(selection_scores.get(chosen_label, float("inf"))),
            }
        )

        obj_candidates_info = {}
        scale_recovery_meta = recon_meta.get("scale_recovery", {})
        for label, path in candidates.items():
            bounded = anchor_row_by_label[label]
            if label == "fused":
                candidate_scale_meta = {
                    "rayst3r": scale_recovery_meta.get("rayst3r", {}),
                    "instantmesh": scale_recovery_meta.get("instantmesh", {}),
                }
            else:
                candidate_scale_meta = scale_recovery_meta.get(label, {})
            exported_mesh_path = os.path.abspath(
                os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj")
            )
            exported_pose_path = os.path.abspath(
                os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt")
            )
            obj_candidates_info[label] = {
                "base_geometry": label,
                "mesh_path": exported_mesh_path,
                "raw_mesh_path": os.path.abspath(path),
                "pose_path": exported_pose_path,
                "source_provenance": (
                    "instantmesh_only_fallback"
                    if instantmesh_only and label == "instantmesh"
                    else f"reconstruction:{label}"
                ),
                "is_fallback_alias": False,
                "s_anchor": float(s_anchor_scores[label]),
                "anchor_score_version": "bounded_observation_calibrated_v1",
                "s_anchor_raw": float(bounded["s_anchor_raw"]),
                "anchor_observation_quality": float(
                    bounded["anchor_observation_quality"]
                ),
                "anchor_calibration_norm": float(bounded["anchor_calibration_norm"]),
                "l_depth": float(results_pose[label]["l_depth"]),
                "l_mask": float(results_pose[label]["l_mask"]),
                "l_geom": float(results_pose[label]["l_geom"]),
                "scale_recovery": candidate_scale_meta,
                "mesh_frame": results_pose[label]["mesh_frame"],
                "pose_frame": results_pose[label]["pose_frame"],
                "construction_frame": "rayst3r_anchor_camera",
                "tf_to_centered": results_pose[label]["tf_to_centered"],
                "anchor_pose_strategy": results_pose[label]["anchor_pose_strategy"],
                "anchor_pose_random_seed": results_pose[label]["anchor_pose_random_seed"],
                "anchor_pose_trial_count": results_pose[label]["anchor_pose_trial_count"],
            }
        candidate_registry[obj] = {
            "chosen": chosen_label,
            "anchor_score_version": "bounded_observation_calibrated_v1",
            "mask_source": "dataset_gt",
            "geometry_units": "meters_after_construction",
            "candidate_mesh_frame": candidate_frame,
            "scale_metadata": {
                "depth_unit": recon_meta.get("depth_unit", chosen_depth_unit),
                "rayst3r_metric_scale": recon_meta.get(
                    "rayst3r_metric_scale", 1.0
                ),
                "instantmesh_metric_scale": recon_meta.get(
                    "instantmesh_metric_scale", 1.0
                ),
                "rgbd_self_supervised": recon_meta.get("scale_recovery", {}),
                "construction_transform": recon_meta.get("transform"),
                "mesh_statistics": recon_meta.get("mesh_statistics", {}),
            },
            "fusion_metadata": recon_meta.get("fusion_diagnostics", {}),
            "fusion_admission": fusion_admission,
            "bidirectional_metadata": recon_meta.get("bidirectional", {}),
            "alignment_metadata": {
                "selection_rule": "best_of_two",
                "strict_both_gate": bool(args.align_strict_both),
                "reciprocal_initialization_pairing": bool(
                    args.icp_reciprocal_pairing
                ),
                "hypotheses_per_direction": int(args.icp_multi_hypo),
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
        print(
            f"{obj}: chosen={chosen_label} "
            f"L_depth={best['l_depth']:.6f} L_mask={best['l_mask']:.6f} L_geom={best['l_geom']:.6f}"
        )
        _release_cuda_memory()

    registry_path = os.path.join(result_root, "candidate_registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(candidate_registry, f, indent=2)
    print(f"\nCandidate registry saved to: {registry_path}")

    if results:
        df = pd.DataFrame(results).sort_values("Object")
        excel_path = os.path.join(result_root, "chamfer_distances.xlsx")
        df.to_excel(excel_path, index=False)

        pose_df = pd.DataFrame(pose_rows).sort_values("Object")
        pose_all_df = pd.DataFrame(pose_rows_all).sort_values(["Object", "Candidate"])
        pose_excel_path = os.path.join(result_root, "pose_metrics.xlsx")
        with pd.ExcelWriter(pose_excel_path) as writer:
            pose_df.to_excel(writer, index=False, sheet_name="chosen")
            pose_all_df.to_excel(writer, index=False, sheet_name="candidates")
        print(f"Results saved to: {excel_path}")
        print(f"Pose metrics saved to: {pose_excel_path}")
    else:
        print("No objects were processed.")


if __name__ == "__main__":
    main()
