from project_paths import setup_project_paths
setup_project_paths()

import argparse
import gc
import json
import os
import random
import glob

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
    calculate_chamfer_distance_gt_mesh,
    nvdiffrast_render,
    visualize_frame_results,
)


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


def _extract_pure_rotation_pose(nocs_gt_pose: np.ndarray) -> np.ndarray:
    s_r = nocs_gt_pose[:3, :3]
    u, _s, vt = np.linalg.svd(s_r)
    r_pure = u @ vt
    if np.linalg.det(r_pure) < 0:
        r_pure = -r_pure
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = r_pure
    pose[:3, 3] = nocs_gt_pose[:3, 3]
    return pose


def _nocs_scale_from_pose(nocs_gt_pose: np.ndarray) -> float:
    _, s, _ = np.linalg.svd(nocs_gt_pose[:3, :3])
    return float(s.mean())


def _as_trimesh(mesh_obj) -> trimesh.Trimesh:
    if isinstance(mesh_obj, trimesh.Scene):
        return trimesh.util.concatenate(
            [g for g in mesh_obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    return mesh_obj


def _backproject_depth_to_points(depth: np.ndarray, k: np.ndarray, valid_mask: np.ndarray, max_points: int = 5000):
    ys, xs = np.where(valid_mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float32)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
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


def _compute_self_supervised_losses(est, depth: np.ndarray, mask: np.ndarray, k: np.ndarray, pred_pose: np.ndarray, glctx):
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
        K=k,
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

    obs_pts = _backproject_depth_to_points(depth, k, obs_mask)
    ren_pts = _backproject_depth_to_points(rd, k, ren_mask)
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
    if label in ("rayst3r", "instantmesh"):
        return c_base

    fitness = meta.get("align_fitness", -1.0)
    rmse = meta.get("align_rmse", float("inf"))
    consistency_ok = (fitness >= tau_inlier) and (rmse <= tau_rmse)
    if label == "fused" and consistency_ok:
        return c_base + w_inlier * fitness + w_rmse * max(0.0, 1.0 - rmse)
    return c_base


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


def _recover_existing_candidates_from_disk(obj_dir: str, obj: str, c_base: float) -> dict:
    recovered = {}
    labels = ["rayst3r", "instantmesh", "fused"]
    for label in labels:
        mesh_out = os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj")
        pose_out = os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt")
        if not (os.path.exists(mesh_out) and os.path.exists(pose_out)):
            continue
        try:
            mesh_local = _as_trimesh(trimesh.load(mesh_out, process=False))
            pred_pose = np.loadtxt(pose_out)
        except Exception:
            continue
        l_depth_path = os.path.join(obj_dir, f"{obj}_ldepth_{label}.txt")
        l_mask_path = os.path.join(obj_dir, f"{obj}_lmask_{label}.txt")
        l_geom_path = os.path.join(obj_dir, f"{obj}_cd_{label}.txt")
        l_depth = float(np.loadtxt(l_depth_path)) if os.path.exists(l_depth_path) else 1.0
        l_mask = float(np.loadtxt(l_mask_path)) if os.path.exists(l_mask_path) else 1.0
        l_geom = float(np.loadtxt(l_geom_path)) if os.path.exists(l_geom_path) else 1.0
        recovered[label] = {
            "pred_pose": pred_pose,
            "est_mesh": mesh_local.copy(),
            "mesh_path": mesh_out,
            "raw_mesh_path": os.path.abspath(mesh_out),
            "pose_path": pose_out,
            "l_depth": l_depth,
            "l_mask": l_mask,
            "l_geom": l_geom,
            "s_anchor": float(c_base),
            "chamfer_gt": float(l_geom),
            "model_center": np.asarray(mesh_local.vertices.mean(axis=0)).reshape(-1).tolist(),
        }
    return recovered


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
    use_any6d: bool = True,
    axis_align: bool = True,
    coarse_est: bool = True,
    refinement: bool = False,
):
    mesh_trial = mesh_template.copy()
    est = Any6D(
        symmetry_tfs=None,
        mesh=mesh_trial,
        debug_dir=save_path,
        debug=0,
        glctx=glctx,
        grid_n_views=args.strategy_grid_n_views,
        grid_inplane_step=args.strategy_grid_inplane_step,
    )
    if use_any6d:
        pred_pose = est.register_any6d(
            K=intrinsic,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=args.any6d_iter,
            refinement=bool(args.any6d_refine) or refinement,
            axis_align=axis_align,
            coarse_est=coarse_est,
            name=name,
        )
    else:
        pred_pose = est.register(
            K=intrinsic,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=args.any6d_iter,
            glctx=glctx,
            name=name,
        )
    return pred_pose, est


def main():
    parser = argparse.ArgumentParser(description="REAL275 anchor phase (strict paper flow with RaySt3R/InstantMesh/Fused)")
    parser.add_argument(
        "--anchor_folder",
        type=str,
        default="/data/gst/MGC-6D/results/anchor_results/real275_paper/real275_reference_view",
    )
    parser.add_argument(
        "--real_model_path",
        type=str,
        default="/data/gst/data/REAL275/obj_models/real_test",
    )
    parser.add_argument("--depth_unit", type=str, default="auto", choices=["auto", "mm", "rayst3r_uint16", "meters"])
    parser.add_argument("--depth_unit_try_both", action="store_true")
    parser.add_argument("--depth_preprocess", action="store_true")
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
    parser.add_argument("--align_use_guess_translation", action="store_true")
    parser.add_argument("--align_bidirectional_icp", action="store_true")
    parser.add_argument("--icp_max_iter", type=int, default=50)
    parser.add_argument("--icp_fitness_threshold", type=float, default=0.05)
    parser.add_argument("--icp_multi_hypo", type=int, default=1)
    parser.add_argument("--icp_init_sigma", type=float, default=0.01)
    parser.add_argument("--mask_stability_enabled", action="store_true")
    parser.add_argument("--mask_stability_threshold", type=float, default=0.6)
    parser.add_argument("--mask_iou_threshold", type=float, default=0.5)
    parser.add_argument("--refine_mask", action="store_true")
    parser.add_argument("--rayst3r_set_conf", type=float, default=2.5)
    parser.add_argument("--rayst3r_n_pred_views", type=int, default=5)
    parser.add_argument("--rayst3r_filter_all_masks", action="store_true")
    parser.add_argument("--rayst3r_tsdf", action="store_true")
    parser.add_argument("--rayst3r_voxel_size", type=float, default=0.002)
    parser.add_argument("--rayst3r_std_ratio", type=float, default=2.5)
    parser.add_argument("--rayst3r_poisson_depth", type=int, default=9)
    parser.add_argument("--rayst3r_poisson_scale", type=float, default=1.0)
    parser.add_argument("--rayst3r_density_quantile", type=float, default=0.02)
    parser.add_argument("--instantmesh_flip", action="store_true")
    parser.add_argument("--instantmesh_no_remove_bg", action="store_true")
    parser.add_argument("--icp_max_corr_ratio", type=float, default=0.05)
    parser.add_argument("--icp_voxel_size", type=float, default=0.005)
    parser.add_argument("--sample_points", type=int, default=50000)
    parser.add_argument(
        "--include_legacy_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include legacy Any6D-generated meshes (e.g., final_mesh_any6d_*.obj) as extra candidates.",
    )
    parser.add_argument(
        "--legacy_candidate_patterns",
        type=str,
        default="final_mesh_any6d_nocoarse_refine_s*.obj,final_mesh_any6d_nocoarse_s*.obj,final_mesh_any6d_coarse_noaa_s*.obj,final_mesh_any6d_coarse_aa_s*.obj",
        help="Comma-separated glob patterns under object folder for extra candidate meshes.",
    )
    parser.add_argument("--objects", type=str, default="", help="Comma-separated object names (empty=all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--any6d_iter", type=int, default=5)
    parser.add_argument("--any6d_refine", type=int, default=1, choices=[0, 1])

    parser.add_argument("--select_depth_weight", type=float, default=0.5)
    parser.add_argument("--select_mask_weight", type=float, default=0.25)
    parser.add_argument("--select_geom_weight", type=float, default=0.25)
    parser.add_argument("--select_chamfer_force_ratio", type=float, default=0.2)
    parser.add_argument("--anchor_c_base", type=float, default=1.0)
    parser.add_argument("--anchor_w_inlier", type=float, default=1.0)
    parser.add_argument("--anchor_w_rmse", type=float, default=1.0)
    parser.add_argument("--anchor_tau_inlier", type=float, default=0.05)
    parser.add_argument("--anchor_tau_rmse", type=float, default=0.02)
    parser.add_argument("--score_alpha", type=float, default=0.3)
    parser.add_argument("--score_beta", type=float, default=0.7)
    parser.add_argument("--summary_filename", type=str, default="chamfer_distances.xlsx")
    parser.add_argument(
        "--strategy_trials_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Borrow old single-anchor multi-strategy multi-seed trials as extra candidates.",
    )
    parser.add_argument(
        "--strategy_n_seeds",
        type=int,
        default=2,
        help="Number of random seeds for each strategy trial.",
    )
    parser.add_argument(
        "--strategy_grid_n_views",
        type=int,
        default=120,
        help="grid_n_views for Any6D strategy trials.",
    )
    parser.add_argument(
        "--strategy_grid_inplane_step",
        type=int,
        default=20,
        help="grid_inplane_step for Any6D strategy trials.",
    )
    args = parser.parse_args()

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
        obj_dir = os.path.join(args.anchor_folder, obj)
        color_bgr = cv2.imread(os.path.join(obj_dir, "color.png"))
        depth_raw = cv2.imread(os.path.join(obj_dir, "depth.png"), cv2.IMREAD_ANYDEPTH)
        mask_img = cv2.imread(os.path.join(obj_dir, "mask.png"), cv2.IMREAD_GRAYSCALE)
        k_path = os.path.join(obj_dir, "K.txt")
        gt_pose_path = os.path.join(obj_dir, f"{obj}_gt_pose.txt")
        gt_mesh_path = os.path.join(args.real_model_path, f"{obj}.obj")
        if any(not os.path.exists(p) for p in [k_path, gt_pose_path, gt_mesh_path]):
            print(f"Skip {obj}: missing required files")
            continue
        if color_bgr is None or depth_raw is None or mask_img is None:
            print(f"Skip {obj}: missing color/depth/mask image")
            continue

        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        recon_meta = {}
        meta_path = recon_results[obj].get("meta")
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                recon_meta = json.load(f)
        chosen_depth_unit = recon_meta.get("depth_unit", args.depth_unit)
        if chosen_depth_unit == "auto":
            chosen_depth_unit = _detect_depth_unit(depth_raw.astype(np.float32))
        depth = _convert_depth(depth_raw.astype(np.float32), chosen_depth_unit)
        mask = (mask_img > 0).astype(np.bool_)
        intrinsic = np.loadtxt(k_path)
        gt_pose_nocs = np.loadtxt(gt_pose_path)
        gt_pose = _extract_pure_rotation_pose(gt_pose_nocs)
        _ = _nocs_scale_from_pose(gt_pose_nocs)

        gt_mesh = _as_trimesh(trimesh.load(gt_mesh_path, process=False))
        ray_path = recon_results[obj].get("mesh_rayst3r")
        inst_path = recon_results[obj].get("mesh_instant")
        fused_path = recon_results[obj].get("mesh_fused")
        align_direction = recon_meta.get("align_direction", "")
        depth_unit_candidates = recon_meta.get("depth_unit_candidates", {})
        ray_failed_all = False
        if isinstance(depth_unit_candidates, dict) and depth_unit_candidates:
            aligns = [str(v.get("align", "")) for v in depth_unit_candidates.values() if isinstance(v, dict)]
            ray_failed_all = bool(aligns) and all(a in {"rayst3r_failed", "rayst3r_empty"} for a in aligns)
            mismatch_msgs = [
                str(v.get("error", ""))
                for v in depth_unit_candidates.values()
                if isinstance(v, dict) and "Expected all tensors to be on the same device" in str(v.get("error", ""))
            ]
            if mismatch_msgs:
                device_mismatch_fail_count += 1
        candidate_mesh_paths = {}
        instantmesh_only = (align_direction == "instantmesh_only_fallback") or ray_failed_all
        if inst_path and os.path.exists(inst_path):
            candidate_mesh_paths["instantmesh"] = inst_path
        if not instantmesh_only:
            if ray_path and os.path.exists(ray_path):
                candidate_mesh_paths["rayst3r"] = ray_path
            if fused_path and os.path.exists(fused_path):
                fused_ok = (
                    recon_meta.get("align_fitness", -1.0) >= args.anchor_tau_inlier
                    and recon_meta.get("align_rmse", float("inf")) <= args.anchor_tau_rmse
                )
                if fused_ok:
                    candidate_mesh_paths["fused"] = fused_path
        elif "instantmesh" in candidate_mesh_paths:
            print(f"[INFO] {obj}: RaySt3R reconstruction invalid; keep instantmesh-only candidate for registry.")

        if args.include_legacy_candidates:
            patterns = [p.strip() for p in args.legacy_candidate_patterns.split(",") if p.strip()]
            for pattern in patterns:
                for p in sorted(glob.glob(os.path.join(obj_dir, pattern))):
                    label = f"legacy_{os.path.splitext(os.path.basename(p))[0]}"
                    if label in candidate_mesh_paths:
                        continue
                    candidate_mesh_paths[label] = p

        candidates = {}
        for label, mesh_path in candidate_mesh_paths.items():
            mesh_local = _as_trimesh(trimesh.load(mesh_path, process=False))
            est = Any6D(symmetry_tfs=None, mesh=mesh_local, debug_dir=obj_dir, debug=0, glctx=glctx)
            try:
                pred_pose = est.register_any6d(
                    K=intrinsic,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.any6d_iter,
                    refinement=bool(args.any6d_refine),
                    name=f"{obj}_{label}",
                )
            except Exception as exc:
                print(f"{obj} [{label}] register_any6d failed: {exc}")
                continue

            l_depth, l_mask, l_geom = _compute_self_supervised_losses(
                est=est,
                depth=depth,
                mask=mask,
                k=intrinsic,
                pred_pose=pred_pose,
                glctx=glctx,
            )
            chamfer_gt = float(calculate_chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, est.mesh))
            mesh_out = os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj")
            pose_out = os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt")
            est.mesh.export(mesh_out)
            np.savetxt(pose_out, pred_pose)
            np.savetxt(os.path.join(obj_dir, f"{obj}_ldepth_{label}.txt"), [l_depth])
            np.savetxt(os.path.join(obj_dir, f"{obj}_lmask_{label}.txt"), [l_mask])
            np.savetxt(os.path.join(obj_dir, f"{obj}_cd_{label}.txt"), [l_geom])
            if label in {"rayst3r", "instantmesh", "fused"}:
                s_anchor = _compute_anchor_calibration_score(
                    meta=recon_meta,
                    label=label,
                    c_base=args.anchor_c_base,
                    w_inlier=args.anchor_w_inlier,
                    w_rmse=args.anchor_w_rmse,
                    tau_inlier=args.anchor_tau_inlier,
                    tau_rmse=args.anchor_tau_rmse,
                )
            else:
                # Legacy candidates do not have reconstruction ICP meta; use base anchor credibility.
                s_anchor = float(args.anchor_c_base)
            candidates[label] = {
                "pred_pose": pred_pose,
                "est_mesh": est.mesh.copy(),
                "mesh_path": mesh_out,
                "raw_mesh_path": os.path.abspath(mesh_path),
                "pose_path": pose_out,
                "l_depth": float(l_depth),
                "l_mask": float(l_mask),
                "l_geom": float(l_geom),
                "s_anchor": float(s_anchor),
                "chamfer_gt": chamfer_gt,
                "model_center": np.asarray(est.model_center).reshape(-1).tolist(),
            }

            if args.strategy_trials_enable:
                strategies = [
                    {"name": "any6d_coarse_aa", "use_any6d": True, "axis_align": True, "coarse_est": True, "refinement": False},
                    {"name": "any6d_coarse_noaa", "use_any6d": True, "axis_align": False, "coarse_est": True, "refinement": False},
                    {"name": "any6d_nocoarse", "use_any6d": True, "axis_align": False, "coarse_est": False, "refinement": False},
                    {"name": "any6d_nocoarse_refine", "use_any6d": True, "axis_align": False, "coarse_est": False, "refinement": True},
                    {"name": "register_simple", "use_any6d": False, "axis_align": False, "coarse_est": False, "refinement": False},
                ]
                mesh_template = mesh_local.copy()
                for strat in strategies:
                    for seed_i in range(max(1, int(args.strategy_n_seeds))):
                        trial_seed = args.seed + seed_i * 7919
                        random.seed(trial_seed)
                        np.random.seed(trial_seed)
                        torch.manual_seed(trial_seed)
                        trial_label = f"{label}_trial_{strat['name']}_s{seed_i}"
                        if trial_label in candidates:
                            continue
                        try:
                            pred_pose_t, est_t = _run_single_trial(
                                mesh_template=mesh_template,
                                save_path=obj_dir,
                                glctx=glctx,
                                args=args,
                                intrinsic=intrinsic,
                                color=color,
                                depth=depth,
                                mask=mask,
                                name=f"{obj}_{trial_label}",
                                use_any6d=strat["use_any6d"],
                                axis_align=strat["axis_align"],
                                coarse_est=strat["coarse_est"],
                                refinement=strat["refinement"],
                            )
                        except Exception:
                            continue
                        l_depth_t, l_mask_t, l_geom_t = _compute_self_supervised_losses(
                            est=est_t,
                            depth=depth,
                            mask=mask,
                            k=intrinsic,
                            pred_pose=pred_pose_t,
                            glctx=glctx,
                        )
                        chamfer_gt_t = float(calculate_chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose_t, est_t.mesh))
                        mesh_out_t = os.path.join(obj_dir, f"final_mesh_{obj}_{trial_label}.obj")
                        pose_out_t = os.path.join(obj_dir, f"{obj}_initial_pose_{trial_label}.txt")
                        est_t.mesh.export(mesh_out_t)
                        np.savetxt(pose_out_t, pred_pose_t)
                        np.savetxt(os.path.join(obj_dir, f"{obj}_ldepth_{trial_label}.txt"), [l_depth_t])
                        np.savetxt(os.path.join(obj_dir, f"{obj}_lmask_{trial_label}.txt"), [l_mask_t])
                        np.savetxt(os.path.join(obj_dir, f"{obj}_cd_{trial_label}.txt"), [l_geom_t])
                        candidates[trial_label] = {
                            "pred_pose": pred_pose_t,
                            "est_mesh": est_t.mesh.copy(),
                            "mesh_path": mesh_out_t,
                            "raw_mesh_path": os.path.abspath(mesh_path),
                            "pose_path": pose_out_t,
                            "l_depth": float(l_depth_t),
                            "l_mask": float(l_mask_t),
                            "l_geom": float(l_geom_t),
                            "s_anchor": float(s_anchor),
                            "chamfer_gt": chamfer_gt_t,
                            "model_center": np.asarray(est_t.model_center).reshape(-1).tolist(),
                        }

        if not candidates:
            recovered = _recover_existing_candidates_from_disk(obj_dir, obj, args.anchor_c_base)
            if recovered:
                print(f"[WARN] {obj}: live candidate eval failed, recovered {len(recovered)} candidate(s) from disk artifacts.")
                candidates = recovered
            else:
                print(f"Skip {obj}: no valid candidate")
                continue

        min_l_depth = min(v["l_depth"] for v in candidates.values())
        min_l_mask = min(v["l_mask"] for v in candidates.values())
        min_l_geom = min(v["l_geom"] for v in candidates.values())
        max_s_anchor = max(v["s_anchor"] for v in candidates.values())

        scored = []
        for label, info in candidates.items():
            s_query = (
                args.select_depth_weight * (info["l_depth"] / max(min_l_depth, 1e-8))
                + args.select_mask_weight * (info["l_mask"] / max(min_l_mask, 1e-8))
                + args.select_geom_weight * (info["l_geom"] / max(min_l_geom, 1e-8))
            )
            s_anchor_norm = info["s_anchor"] / max(max_s_anchor, 1e-8)
            total_score = -args.score_alpha * s_anchor_norm + args.score_beta * s_query
            info["selection_score"] = float(total_score)
            info["s_query"] = float(s_query)
            scored.append((label, float(total_score)))
            pose_rows_all.append(
                {
                    "Object": obj,
                    "Object_Number": _parse_object_number_from_source_frame(obj_dir),
                    "Candidate": label,
                    "L_depth": info["l_depth"],
                    "L_mask": info["l_mask"],
                    "L_geom": info["l_geom"],
                    "S_anchor": info["s_anchor"],
                    "Selection_Score": float(total_score),
                }
            )
        scored.sort(key=lambda x: x[1])

        chosen_label = scored[0][0]
        best_geom_label = min(candidates.items(), key=lambda kv: kv[1]["l_geom"])[0]
        forced_by_chamfer = False
        if candidates[chosen_label]["l_geom"] > candidates[best_geom_label]["l_geom"] * (1.0 + args.select_chamfer_force_ratio):
            chosen_label = best_geom_label
            forced_by_chamfer = True

        chosen = candidates[chosen_label]
        np.savetxt(os.path.join(obj_dir, f"{obj}_initial_pose.txt"), chosen["pred_pose"])
        chosen["est_mesh"].export(os.path.join(obj_dir, f"final_mesh_{obj}.obj"))
        np.savetxt(os.path.join(obj_dir, f"{obj}_cd.txt"), [chosen["l_geom"]])
        np.savetxt(os.path.join(obj_dir, f"{obj}_model_center.txt"), np.asarray(chosen["model_center"], dtype=np.float64))

        try:
            est_vis = Any6D(symmetry_tfs=None, mesh=chosen["est_mesh"].copy(), debug_dir=obj_dir, debug=0, glctx=glctx)
            visualize_frame_results(
                color=color,
                gt_mesh=gt_mesh,
                est=est_vis,
                K=intrinsic,
                gt_pose=gt_pose,
                pred_pose=chosen["pred_pose"],
                metric=None,
                obj_f=obj,
                frame_idx=0,
                save_path=obj_dir,
                glctx=glctx,
                name="demo_data",
                mesh_index=0,
                init=False,
                save_on_folder=True,
            )
        except Exception:
            pass

        results.append(
            {
                "Object": obj,
                "Object_Number": _parse_object_number_from_source_frame(obj_dir),
                "Chamfer_Distance": float(chosen["l_geom"]),
                "Chamfer_GT": float(chosen["chamfer_gt"]),
                "Chosen_Mesh": chosen_label,
                "Forced_By_Chamfer": bool(forced_by_chamfer),
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
                "Forced_By_Chamfer": bool(forced_by_chamfer),
            }
        )

        obj_candidates_info = {}
        for label, info in candidates.items():
            obj_candidates_info[label] = {
                "mesh_path": os.path.abspath(info["mesh_path"]),
                "raw_mesh_path": info["raw_mesh_path"],
                "pose_path": os.path.abspath(info["pose_path"]),
                "s_anchor": float(info["s_anchor"]),
                "l_depth": float(info["l_depth"]),
                "l_mask": float(info["l_mask"]),
                "l_geom": float(info["l_geom"]),
                "selection_score": float(info["selection_score"]),
                "model_center": info["model_center"],
            }
        candidate_registry[obj] = {
            "chosen": chosen_label,
            "candidates": obj_candidates_info,
            "final_mesh_path": os.path.abspath(os.path.join(obj_dir, f"final_mesh_{obj}.obj")),
            "final_pose_path": os.path.abspath(os.path.join(obj_dir, f"{obj}_initial_pose.txt")),
            "K_path": os.path.abspath(k_path),
        }
        _release_cuda_memory()

    if not results:
        print("No objects processed.")
        return

    registry_path = os.path.join(args.anchor_folder, "candidate_registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(candidate_registry, f, indent=2)
    print(f"Candidate registry saved to: {registry_path}")

    run_meta = {
        "anchor_folder": args.anchor_folder,
        "real_model_path": args.real_model_path,
        "selection": {
            "select_depth_weight": args.select_depth_weight,
            "select_mask_weight": args.select_mask_weight,
            "select_geom_weight": args.select_geom_weight,
            "select_chamfer_force_ratio": args.select_chamfer_force_ratio,
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

    df = pd.DataFrame(results).sort_values("Object")
    excel_path = os.path.join(args.anchor_folder, args.summary_filename)
    df.to_excel(excel_path, index=False)

    pose_df = pd.DataFrame(pose_rows).sort_values("Object")
    pose_all_df = pd.DataFrame(pose_rows_all).sort_values(["Object", "Candidate"])
    pose_excel_path = os.path.join(args.anchor_folder, "pose_metrics.xlsx")
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
