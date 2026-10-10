"""TOYL query phase with paper-style multi-candidate selection."""

from project_paths import setup_project_paths
setup_project_paths()

import argparse
import copy
import gc
import json
import os
import time
import warnings
from datetime import datetime

import cv2
import nvdiffrast.torch as dr
import numpy as np
import pandas as pd
import torch
import trimesh
from pytorch_lightning import seed_everything
from tqdm import tqdm

from bop_toolkit_lib.pose_error_custom import mspd, mssd, vsd
from bop_toolkit_lib.misc import get_symmetry_transformations
from estimater import Any6D, PoseRefinePredictor, ScorePredictor
from foundationpose.Utils import nvdiffrast_render, visualize_frame_results_gt
from metrics import compute_RT_distances, compute_add, compute_adds
from candidate_selection import (
    PAPER_CANDIDATE_LABELS,
    apply_candidate_appearance,
    compute_anchor_reliability_components,
    derive_anchor_reliability_scores,
    paper_registry_candidates,
    score_candidates,
    select_scored_candidate,
)
from revision_protocol import (
    add_query_protocol_arguments,
    runtime_provenance,
    validate_query_protocol,
)

TOYL_OBJ_IDS = list(range(1, 22))
MM_TO_M = 0.001


class CandidateCache:
    __slots__ = ("label", "mesh", "s_anchor", "geometry", "scale", "appearance")

    def __init__(self, label: str, mesh: trimesh.Trimesh, s_anchor: float = 1.0, use_bbox_diameter: bool = False, lazy_tensors: bool = False, *, geometry: str | None = None, scale: str = "registry", appearance: str = "configured"):
        self.label = label
        self.mesh = mesh
        self.s_anchor = float(s_anchor)
        self.geometry = geometry or label
        self.scale = scale
        self.appearance = appearance


def swap_candidate(est: Any6D, cache: CandidateCache):
    # For TOYL, use the same robust code path as single-candidate baseline
    # to avoid mesh-coordinate mismatch from fast geometry-state swapping.
    est.reset_object(mesh=cache.mesh, symmetry_tfs=None)


def make_candidate_estimator(cache: CandidateCache, save_root: str, glctx):
    est = Any6D(
        mesh=cache.mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=save_root,
        debug=0,
        glctx=glctx,
    )
    if hasattr(est, "to_device"):
        est.to_device("cuda" if torch.cuda.is_available() else "cpu")
    return est


def _cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


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


def _chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return 1e3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.as_tensor(pts_a, dtype=torch.float32, device=device)
    b = torch.as_tensor(pts_b, dtype=torch.float32, device=device)
    dists = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2.0).squeeze(0)
    return float((dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()).cpu().item())


def compute_observation_consistency(
    est: Any6D,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    pred_pose: np.ndarray,
    glctx,
) -> tuple[float, float, float, float, float]:
    h, w = depth.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_t = torch.as_tensor(pred_pose, device=device, dtype=torch.float32).unsqueeze(0)
    tf = est.get_tf_to_centered_mesh()
    if torch.is_tensor(tf):
        tf = tf.to(device=device, dtype=torch.float32).unsqueeze(0)
    else:
        tf = torch.as_tensor(tf, device=device, dtype=torch.float32).unsqueeze(0)
    ob_in_cams = pose_t @ torch.linalg.inv(tf)
    rendered = nvdiffrast_render(mesh=est.mesh, mesh_tensors=est.mesh_tensors, ob_in_cams=ob_in_cams, K=K, H=h, W=w, glctx=glctx)
    if rendered is None:
        return 1.0, 1.0, 1e3, 1.0, 0.0
    rendered_color = rendered[0] if isinstance(rendered, tuple) and len(rendered) >= 1 else None
    rendered_depth = rendered[1] if isinstance(rendered, tuple) and len(rendered) >= 2 else None
    if rendered_depth is None:
        return 1.0, 1.0, 1e3, 1.0, 0.0
    rd = rendered_depth[0] if getattr(rendered_depth, "ndim", 0) > 2 else rendered_depth
    if torch.is_tensor(rd):
        rd = rd.detach().cpu().numpy()
    rd = rd.astype(np.float32)
    obs_mask = (mask > 0) & (depth > 1e-4)
    ren_mask = rd > 1e-4
    overlap = obs_mask & ren_mask
    coverage_ratio = float(overlap.sum()) / max(float(obs_mask.sum()), 1.0)
    if overlap.sum() > 10:
        depth_scale = max(float(np.mean(depth[overlap])), 1e-4)
        l_depth = float(np.mean(np.abs(depth[overlap] - rd[overlap])) / depth_scale)
    else:
        l_depth = 1.0
    union = obs_mask | ren_mask
    l_mask = float(1.0 - (obs_mask & ren_mask).sum() / max(float(union.sum()), 1.0))
    l_geom = _chamfer_distance(_backproject_depth_to_points(rd, K, ren_mask), _backproject_depth_to_points(depth, K, obs_mask))
    l_appearance = 1.0
    if rendered_color is not None and overlap.sum() > 10:
        rc = rendered_color[0] if getattr(rendered_color, "ndim", 0) == 4 else rendered_color
        if torch.is_tensor(rc):
            rc = rc.detach().cpu().numpy()
        rc = np.asarray(rc, dtype=np.float32)[..., :3]
        observed_rgb = np.asarray(rgb, dtype=np.float32)[..., :3]
        if observed_rgb.size and float(np.max(observed_rgb)) > 1.5:
            observed_rgb = observed_rgb / 255.0
        color_mask = overlap & (np.abs(depth - rd) <= 0.03)
        if color_mask.sum() > 10:
            rendered_pixels = np.clip(rc[color_mask], 0.0, 1.0)
            observed_pixels = np.clip(observed_rgb[color_mask], 0.0, 1.0)
            rendered_unit = rendered_pixels / np.maximum(
                np.linalg.norm(rendered_pixels, axis=1, keepdims=True), 1e-6
            )
            observed_unit = observed_pixels / np.maximum(
                np.linalg.norm(observed_pixels, axis=1, keepdims=True), 1e-6
            )
            chromatic_loss = np.mean(
                1.0 - np.clip(np.sum(rendered_unit * observed_unit, axis=1), -1.0, 1.0)
            )
            intensity_loss = np.mean(
                np.abs(rendered_pixels.mean(axis=1) - observed_pixels.mean(axis=1))
            )
            l_appearance = float(
                np.clip(0.75 * chromatic_loss + 0.25 * intensity_loss, 0.0, 1.0)
            )
    return l_depth, l_mask, l_geom, l_appearance, coverage_ratio


def _load_scene_json(scene_dir: str, filename: str) -> dict:
    with open(os.path.join(scene_dir, filename), "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_bop_pose(gt_entry: dict) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(gt_entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    pose[:3, 3:4] = np.array(gt_entry["cam_t_m2c"], dtype=np.float64).reshape(3, 1)
    return pose


def _parse_bop_K(cam_entry: dict) -> np.ndarray:
    return np.array(cam_entry["cam_K"], dtype=np.float64).reshape(3, 3)


def _gt_pose_to_meters(pose_mm: np.ndarray) -> np.ndarray:
    pose_m = pose_mm.copy()
    pose_m[:3, 3] *= MM_TO_M
    return pose_m


def _obj_name(obj_id: int) -> str:
    return f"obj_{obj_id:02d}"


def _build_ho3d_style_trans_disc(model_info_entry: dict) -> list[dict]:
    """Return the official BOP19 discrete + discretized continuous symmetries."""

    transforms = get_symmetry_transformations(
        model_info_entry, max_sym_disc_step=0.01
    )
    return [
        {
            "R": np.asarray(tf["R"], dtype=np.float64).reshape(3, 3),
            "t": np.asarray(tf["t"], dtype=np.float64).reshape(3, 1) * MM_TO_M,
        }
        for tf in transforms
    ]


def _parse_id_list(raw: str) -> list[int] | None:
    if not raw.strip():
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _list_scene_dirs(test_dir: str, scene_ids: list[int] | None) -> list[tuple[int, str]]:
    if scene_ids is not None:
        candidates = [(sid, os.path.join(test_dir, f"{sid:06d}")) for sid in scene_ids]
    else:
        candidates = [(int(name), os.path.join(test_dir, name)) for name in sorted(os.listdir(test_dir)) if name.isdigit() and os.path.isdir(os.path.join(test_dir, name))]
    return [(sid, sdir) for sid, sdir in candidates if os.path.exists(os.path.join(sdir, "scene_gt.json"))]


def _find_obj_annos(gt_list: list[dict], obj_id: int) -> list[tuple[int, dict]]:
    return [(anno_idx, entry) for anno_idx, entry in enumerate(gt_list) if int(entry.get("obj_id", -1)) == int(obj_id)]


def _read_mask(scene_dir: str, mask_type: str, im_id: int, anno_idx: int) -> np.ndarray | None:
    mask_types = [mask_type, "mask" if mask_type == "mask_visib" else "mask_visib"]
    for cur_mask_type in mask_types:
        inst = cv2.imread(os.path.join(scene_dir, cur_mask_type, f"{im_id:06d}_{anno_idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if inst is not None:
            return inst > 0
        merged = cv2.imread(os.path.join(scene_dir, cur_mask_type, f"{im_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if merged is not None:
            return merged == (anno_idx + 1) if int(merged.max()) > 1 else merged > 0
    return None


def _is_valid_pose(pose: np.ndarray) -> bool:
    if pose is None or pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False
    det = np.linalg.det(pose[:3, :3])
    return np.isfinite(det) and abs(det - 1.0) < 0.05


def _append_zero_frame(object_metrics: dict, all_frame_data: dict, obj_name: str, obj_id: int, im_id: int, frame_idx: int, scene_id: int | None = None, anno_idx: int | None = None):
    scene_id = obj_id if scene_id is None else scene_id
    object_metrics[obj_name]["ADD"].append(0.0)
    object_metrics[obj_name]["ADD-S"].append(0.0)
    object_metrics[obj_name]["ADD(-S)"].append(0.0)
    object_metrics[obj_name]["AR"].append(0.0)
    object_metrics[obj_name]["VSD"].append(0.0)
    object_metrics[obj_name]["MSSD"].append(0.0)
    object_metrics[obj_name]["MSPD"].append(0.0)
    object_metrics[obj_name]["R error"].append(float("nan"))
    object_metrics[obj_name]["T error"].append(float("nan"))
    object_metrics[obj_name]["chosen_candidate"].append("failed")
    object_metrics[obj_name]["cls_id"].append(obj_name)
    object_metrics[obj_name]["instance_id"].append(frame_idx)
    object_metrics[obj_name]["scene_id"].append(scene_id)
    object_metrics[obj_name]["image_id"].append(im_id)
    object_metrics[obj_name]["anno_idx"].append(anno_idx)
    all_frame_data["Frame_ID"].append(frame_idx)
    all_frame_data["Scene"].append(f"{scene_id:06d}")
    all_frame_data["Image_ID"].append(im_id)
    all_frame_data["Anno_ID"].append(anno_idx)
    all_frame_data["Class"].append(obj_name)
    all_frame_data["Chosen_Candidate"].append("failed")
    all_frame_data["ADD-S"].append(0.0)
    all_frame_data["ADD"].append(0.0)
    all_frame_data["ADD(-S)"].append(0.0)
    all_frame_data["AR"].append(0.0)
    all_frame_data["MSSD"].append(0.0)
    all_frame_data["MSPD"].append(0.0)
    all_frame_data["VSD"].append(0.0)
    all_frame_data["R_error"].append(float("nan"))
    all_frame_data["T_error"].append(float("nan"))
    all_frame_data["Selection_Time_s"].append(float("nan"))


def _record_failed_frame(object_metrics: dict, all_frame_data: dict, obj_name: str, obj_id: int, im_id: int, obj_count: int, frame_count_this_obj: int, count_failures: bool, scene_id: int | None = None, anno_idx: int | None = None) -> tuple[int, int]:
    if count_failures:
        _append_zero_frame(object_metrics, all_frame_data, obj_name, obj_id, im_id, obj_count, scene_id=scene_id, anno_idx=anno_idx)
        obj_count += 1
        frame_count_this_obj += 1
    return obj_count, frame_count_this_obj


def _nanmean_or_nan(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float("nan") if arr.size == 0 or np.isnan(arr).all() else float(np.nanmean(arr))


def _evaluate_candidate_pose_metrics(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
    gt_vertices_m: np.ndarray,
    gt_diameter_m: float,
    gt_diameter_mm: float,
    is_symmetric: bool,
    symmetry_transforms: list,
    intrinsic: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    renderer,
    obj_id: int,
) -> dict[str, float]:
    """Compute evaluation-only metrics for one geometry candidate pose."""

    pred_pose = np.asarray(pred_pose, dtype=np.float64).reshape(4, 4)
    gt_pose = np.asarray(gt_pose, dtype=np.float64).reshape(4, 4)
    err_r, err_t = compute_RT_distances(pred_pose, gt_pose)
    add_value = compute_add(gt_vertices_m, pred_pose, gt_pose)
    adds_value = compute_adds(gt_vertices_m, pred_pose, gt_pose)
    add_recall = float(add_value <= float(gt_diameter_m) * 0.1)
    adds_recall = float(adds_value <= float(gt_diameter_m) * 0.1)
    add_mix = adds_recall if is_symmetric else add_recall

    mssd_error = (
        mssd(
            pose_est=pred_pose,
            pose_gt=gt_pose,
            pts=gt_vertices_m,
            syms=symmetry_transforms,
        )
        * 1e3
    )
    mspd_error = mspd(
        pose_est=pred_pose,
        pose_gt=gt_pose,
        pts=gt_vertices_m,
        K=intrinsic,
        syms=symmetry_transforms,
    )
    recall_grid = np.array(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        dtype=np.float64,
    )
    mean_mssd = float((mssd_error < recall_grid * float(gt_diameter_mm)).mean())
    mean_mspd = float(
        (mspd_error < np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])).mean()
    )
    pred_r = pred_pose[:3, :3]
    pred_t = np.expand_dims(pred_pose[:3, 3], axis=1) * 1e3
    gt_r = gt_pose[:3, :3]
    gt_t = np.expand_dims(gt_pose[:3, 3], axis=1) * 1e3
    try:
        vsd_errors = np.asarray(
            vsd(
                pred_r,
                pred_t,
                gt_r,
                gt_t,
                depth_raw.astype(np.float32) * depth_scale,
                intrinsic.reshape(3, 3),
                15.0,
                recall_grid.tolist(),
                True,
                gt_diameter_mm,
                renderer,
                obj_id,
            )
        )
        mean_vsd = float(
            np.stack([vsd_errors < threshold for threshold in recall_grid], axis=1).mean()
        )
    except Exception:
        mean_vsd = 0.0
    mean_ar = float((mean_mssd + mean_mspd + mean_vsd) / 3.0)
    return {
        "ADD": add_recall,
        "ADD-S": adds_recall,
        "ADD(-S)": add_mix,
        "AR": mean_ar,
        "MSSD": mean_mssd,
        "MSPD": mean_mspd,
        "VSD": mean_vsd,
        "R_error": float(np.asarray(err_r).reshape(-1)[0]),
        "T_error": float(np.asarray(err_t).reshape(-1)[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="TOYL query paper-style multi-candidate evaluation")
    parser.add_argument("--name", type=str, default="any6d_toyl_paper")
    parser.add_argument(
        "--anchor_path",
        type=str,
        required=True,
        help="Anchor result folder containing candidate_registry.json",
    )
    parser.add_argument("--toyl_root", type=str, required=True)
    parser.add_argument("--toyl_model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--running_stride", type=int, default=1)
    parser.add_argument("--register_iteration", type=int, default=5)
    parser.add_argument("--mask_type", type=str, default="mask_visib", choices=["mask_visib", "mask"])
    parser.add_argument("--obj_ids", type=str, default="")
    parser.add_argument("--scene_ids", type=str, default="")
    parser.add_argument(
        "--candidate_error_log_limit",
        type=int,
        default=8,
        help="Maximum per-run candidate registration exceptions printed to stderr",
    )
    parser.add_argument("--use_bbox_diameter", action="store_true", default=False)
    parser.add_argument("--lazy_tensors", action="store_true", default=False)
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--count_failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--targets_path",
        type=str,
        default="",
        help="BOP target JSON; empty uses <toyl_root>/tyol/test_targets_bop19.json.",
    )
    parser.add_argument(
        "--anchor_color_depth_tolerance",
        type=float,
        default=0.03,
        help=(
            "Depth-consistency tolerance in meters for projecting anchor RGB "
            "onto candidate meshes. The formal/v7 protocol uses 0.03."
        ),
    )
    add_query_protocol_arguments(parser)
    args = parser.parse_args()
    args.w_appearance = 0.0
    args.query_loss_normalization = "fixed_scale"
    args.query_depth_scale = 0.10
    args.query_mask_scale = 1.00
    args.query_geom_scale = 0.10
    args.query_appearance_scale = 1.00
    args.anchor_depth_scale = 0.05
    args.anchor_mask_scale = 0.50
    args.anchor_geom_scale = 0.05
    args.anchor_calibration_weight = 0.15
    score_weights = np.asarray(
        [
            args.w_depth,
            args.w_mask,
            args.w_geom,
            args.w_appearance,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(score_weights).all() or (score_weights < 0.0).any():
        raise ValueError("All candidate-score weights must be finite and non-negative")
    if (
        args.w_depth + args.w_mask + args.w_geom + args.w_appearance
    ) <= 0.0:
        raise ValueError("At least one Eq. (22) observation weight must be positive")
    if args.score_alpha < 0.0 or args.score_beta < 0.0:
        raise ValueError("--score_alpha and --score_beta must be non-negative")
    if args.score_alpha + args.score_beta <= 0.0:
        raise ValueError("--score_alpha and --score_beta cannot both be zero")
    if not np.isfinite(args.anchor_color_depth_tolerance) or args.anchor_color_depth_tolerance < 0.0:
        raise ValueError("--anchor_color_depth_tolerance must be finite and non-negative")
    seed_everything(int(args.seed))
    selected_labels = validate_query_protocol(args)

    eval_obj_ids = _parse_id_list(args.obj_ids) or list(TOYL_OBJ_IDS)
    eval_scene_ids = _parse_id_list(args.scene_ids)
    test_dir = os.path.join(args.toyl_root, "test")
    scene_dirs = _list_scene_dirs(test_dir, eval_scene_ids)
    if not scene_dirs:
        raise RuntimeError(f"No BOP scenes found under {test_dir}")

    targets_path = args.targets_path or os.path.join(
        args.toyl_root, "tyol", "test_targets_bop19.json"
    )
    official_targets = None
    if os.path.exists(targets_path):
        with open(targets_path, "r", encoding="utf-8") as f:
            target_rows = json.load(f)
        official_targets = {
            (int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"]))
            for row in target_rows
        }
        print(f"Loaded {len(official_targets)} official BOP19 TOYL targets")
    else:
        warnings.warn(f"Official TOYL target list not found: {targets_path}")

    registry_path = os.path.join(args.anchor_path, "candidate_registry.json")
    candidate_registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            candidate_registry = json.load(f)
        print(f"Loaded candidate registry with {len(candidate_registry)} objects")
    else:
        raise FileNotFoundError(
            f"Formal multi-candidate evaluation requires {registry_path}"
        )

    date_str = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    save_root = f"./results/toyl_results/{args.name}/{date_str}"
    os.makedirs(save_root, exist_ok=True)
    run_manifest = {
        "dataset": "TOYL",
        "anchor_path": os.path.abspath(args.anchor_path),
        "targets_path": os.path.abspath(targets_path),
        "mask_type": args.mask_type,
        "mask_sources": {
            "anchor": "dataset_ground_truth_instance_mask",
            "query": "dataset_ground_truth_instance_mask",
        },
        "seed": int(args.seed),
        "candidate_labels": sorted(selected_labels or []),
        "candidate_color_mode": args.candidate_color_mode,
        "candidate_appearance": {
            "mode": args.candidate_color_mode,
            "depth_parameter_m": float(args.anchor_color_depth_tolerance),
            "visibility_rule": "shared_candidate_selection_appearance_rule",
            "uses_query": False,
            "uses_gt_pose_or_cad": False,
            "mask_source": "dataset_gt",
        },
        "selection": {
            "rule": "direct weighted argmax; no default candidate or hard replacement gate",
            "score_version": "paper_eq22_v1",
            "formula": "alpha*S_anchor_norm+beta*(-wd*Ld-wm*Lm-wg*Lg)",
            "eq22_main_score": True,
            "appearance_role": "candidate coloring only; excluded from Eq. (22)",
            "loss_normalization": args.query_loss_normalization,
            "w_depth": args.w_depth,
            "w_mask": args.w_mask,
            "w_geom": args.w_geom,
            "w_appearance": args.w_appearance,
            "score_alpha": args.score_alpha,
            "score_beta": args.score_beta,
            "fixed_scales": {
                "depth": args.query_depth_scale,
                "mask": args.query_mask_scale,
                "geom": args.query_geom_scale,
                "appearance": args.query_appearance_scale,
            },
        },
        "provenance": runtime_provenance(os.path.dirname(__file__)),
        "args": vars(args),
    }
    with open(os.path.join(save_root, "query_run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    with open(os.path.join(args.toyl_model_path, "models_info.json"), "r", encoding="utf-8") as f:
        model_info = json.load(f)

    device = args.device
    if torch.cuda.is_available():
        if device.startswith("cuda:") and int(device.split(":")[1]) >= torch.cuda.device_count():
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    glctx = dr.RasterizeCudaContext(device)
    mesh_tmp = copy.deepcopy(trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4)))
    mesh_placeholder = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(), faces=mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh_placeholder, scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir=save_root, debug=0, glctx=glctx)
    if hasattr(est, "to_device"):
        est.to_device(device)
    shared_scorer = getattr(est, "scorer", None)
    shared_refiner = getattr(est, "refiner", None)
    # Defer EGL initialization until after argument parsing.
    from bop_toolkit_lib.renderer_vispy import RendererVispy

    renderer = RendererVispy(640, 480, mode="depth")
    candidate_error_count = 0

    obj_names = [_obj_name(oid) for oid in eval_obj_ids]
    object_metrics = {
        obj: {
            "ADD": [],
            "ADD-S": [],
            "ADD(-S)": [],
            "AR": [],
            "VSD": [],
            "MSSD": [],
            "MSPD": [],
            "R error": [],
            "T error": [],
            "chosen_candidate": [],
            "cls_id": [],
            "instance_id": [],
            "scene_id": [],
            "image_id": [],
            "anno_idx": [],
        }
        for obj in obj_names
    }
    all_frame_data = {
        "Frame_ID": [],
        "Scene": [],
        "Image_ID": [],
        "Anno_ID": [],
        "Class": [],
        "Chosen_Candidate": [],
        "ADD-S": [],
        "ADD": [],
        "ADD(-S)": [],
        "AR": [],
        "MSSD": [],
        "MSPD": [],
        "VSD": [],
        "R_error": [],
        "T_error": [],
        "Selection_Time_s": [],
    }

    obj_count = 0
    summary_data = []
    candidate_score_rows = []
    selection_times = []
    sampled_frame_instances = 0
    registration_failure_frames = 0

    for obj_id in tqdm(eval_obj_ids, desc="Evaluating Object"):
        obj_name = _obj_name(obj_id)
        obj_anchor_dir = os.path.join(args.anchor_path, obj_name)
        anchor_gt_pose_path = os.path.join(obj_anchor_dir, f"{obj_name}_gt_pose.txt")
        anchor_K_path = os.path.join(obj_anchor_dir, "K.txt")
        if not (os.path.exists(anchor_gt_pose_path) and os.path.exists(anchor_K_path)):
            warnings.warn(f"Skip {obj_name}: missing anchor GT pose/K")
            continue
        anchor_k = np.loadtxt(anchor_K_path)
        anchor_color_bgr = cv2.imread(os.path.join(obj_anchor_dir, "color.png"))
        anchor_mask_raw = cv2.imread(
            os.path.join(obj_anchor_dir, "mask.png"), cv2.IMREAD_UNCHANGED
        )
        anchor_depth_raw = cv2.imread(
            os.path.join(obj_anchor_dir, "depth.png"), cv2.IMREAD_UNCHANGED
        )
        if anchor_color_bgr is None or anchor_mask_raw is None:
            warnings.warn(f"Skip {obj_name}: missing saved anchor RGB/mask")
            continue
        anchor_rgb = cv2.cvtColor(anchor_color_bgr, cv2.COLOR_BGR2RGB)
        anchor_mask = (
            np.any(anchor_mask_raw > 0, axis=2)
            if anchor_mask_raw.ndim == 3 else anchor_mask_raw > 0
        )
        anchor_depth = (
            anchor_depth_raw.astype(np.float32) * MM_TO_M
            if anchor_depth_raw is not None else None
        )

        obj_registry = candidate_registry.get(obj_name, {})
        anchor_appearance_mode = obj_registry.get("appearance_metadata", {}).get(
            "mode"
        )
        if (
            args.candidate_color_mode == "anchor_prior_hybrid"
            and anchor_appearance_mode is not None
            and anchor_appearance_mode != args.candidate_color_mode
        ):
            raise RuntimeError(
                f"{obj_name}: anchor appearance={anchor_appearance_mode!r} but "
                f"query requested {args.candidate_color_mode!r}; use one contract"
            )
        obj_candidates = paper_registry_candidates(
            obj_registry.get("candidates", {})
        )
        obj_candidates = {
            label: info
            for label, info in obj_candidates.items()
            if label in selected_labels
        }
        cand_caches = []
        cand_anchor_poses = {}
        cand_info = {}
        if obj_candidates:
            for label, info in obj_candidates.items():
                mesh_path = info.get("mesh_path", "")
                if not os.path.exists(mesh_path):
                    mesh_path = os.path.join(obj_anchor_dir, f"final_mesh_{obj_name}_{label}.obj")
                if not os.path.exists(mesh_path):
                    continue
                mesh = trimesh.load(mesh_path)
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
                cache = CandidateCache(label=label, mesh=mesh, s_anchor=float(info.get("s_anchor", 1.0)), use_bbox_diameter=args.use_bbox_diameter, lazy_tensors=args.lazy_tensors)
                cand_caches.append(cache)
                cand_info[label] = info
                cand_info[label]["shared_anchor_reliability"] = (
                    compute_anchor_reliability_components(info, obj_registry)
                )
                mesh_frame = info.get("mesh_frame")
                pose_frame = info.get("pose_frame")
                if (
                    mesh_frame is not None
                    and pose_frame is not None
                    and mesh_frame != pose_frame
                ):
                    raise RuntimeError(
                        f"Registry mesh/pose frame mismatch for {obj_name}/{label}: "
                        f"mesh_frame={mesh_frame}, pose_frame={pose_frame}"
                    )
                pose_path = info.get("pose_path", "")
                if os.path.exists(pose_path):
                    cand_anchor_poses[label] = np.loadtxt(pose_path)

        if not cand_caches:
            warnings.warn(
                f"Skip {obj_name}: none of the requested formal registry candidates exist"
            )
            continue

        anchor_rows = []
        registry_is_bounded = True
        for cache in cand_caches:
            info = cand_info[cache.label]
            registry_is_bounded &= (
                info.get("anchor_score_version")
                == "bounded_observation_calibrated_v1"
            )
            anchor_rows.append(
                {
                    "label": cache.label,
                    "s_anchor": float(info.get("s_anchor", 1.0)),
                    "l_depth": float(info.get("l_depth", args.anchor_depth_scale)),
                    "l_mask": float(info.get("l_mask", args.anchor_mask_scale)),
                    "l_geom": float(info.get("l_geom", args.anchor_geom_scale)),
                }
            )
        if not registry_is_bounded:
            derive_anchor_reliability_scores(
                anchor_rows,
                w_depth=args.w_depth,
                w_mask=args.w_mask,
                w_geom=args.w_geom,
                depth_scale=args.anchor_depth_scale,
                mask_scale=args.anchor_mask_scale,
                geom_scale=args.anchor_geom_scale,
                calibration_weight=args.anchor_calibration_weight,
            )
        bounded_anchor = {row["label"]: row for row in anchor_rows}
        for cache in cand_caches:
            cache.s_anchor = float(bounded_anchor[cache.label]["s_anchor"])

        appearance_poses = cand_anchor_poses
        appearance_meshes, appearance_summary = apply_candidate_appearance(
            {cache.label: cache.mesh for cache in cand_caches},
            appearance_poses,
            args.candidate_color_mode,
            anchor_rgb=anchor_rgb,
            anchor_mask=anchor_mask,
            anchor_k=anchor_k,
            anchor_depth=anchor_depth,
            anchor_depth_tolerance_m=args.anchor_color_depth_tolerance,
        )
        for cache in cand_caches:
            cache.mesh = appearance_meshes[cache.label]
        print(
            f"{obj_name}: {len(cand_caches)} candidate(s), appearance={appearance_summary.mode}, "
            f"color_source={appearance_summary.source_label}, "
            f"transferred={list(appearance_summary.transferred_labels)}, "
            f"projected={dict(appearance_summary.projected_vertex_counts)}"
        )

        gt_ply_path = os.path.join(args.toyl_model_path, f"obj_{obj_id:06d}.ply")
        gt_mesh_mm = None
        gt_mesh_m = None
        gt_diameter_mm = None
        gt_diameter_m = None
        trans_disc = None
        is_symmetric = None

        def load_object_evaluation_resources():
            nonlocal gt_mesh_mm, gt_mesh_m, gt_diameter_mm, gt_diameter_m
            nonlocal trans_disc, is_symmetric
            if gt_mesh_m is not None:
                return
            gt_mesh_mm = trimesh.load(gt_ply_path)
            if isinstance(gt_mesh_mm, trimesh.Scene):
                gt_mesh_mm = trimesh.util.concatenate(
                    [g for g in gt_mesh_mm.geometry.values() if isinstance(g, trimesh.Trimesh)]
                )
            gt_mesh_m = gt_mesh_mm.copy()
            gt_mesh_m.vertices = gt_mesh_mm.vertices * MM_TO_M
            gt_diameter_mm = float(model_info[str(obj_id)]["diameter"])
            gt_diameter_m = gt_diameter_mm * MM_TO_M
            gt_mesh_dict = {
                "pts": np.asarray(gt_mesh_mm.vertices, dtype=np.float64),
                "normals": np.asarray(gt_mesh_mm.face_normals, dtype=np.float64),
                "faces": np.asarray(gt_mesh_mm.faces, dtype=np.int32),
            }
            renderer.my_add_object(gt_mesh_dict, obj_id)
            model_info_entry = model_info.get(str(obj_id), {})
            trans_disc = _build_ho3d_style_trans_disc(model_info_entry)
            is_symmetric = bool(
                model_info_entry.get("symmetries_discrete")
                or model_info_entry.get("symmetries_continuous")
            )

        query_instances = []
        for scene_id, scene_dir in scene_dirs:
            scene_gt = _load_scene_json(scene_dir, "scene_gt.json")
            scene_camera = _load_scene_json(scene_dir, "scene_camera.json")
            frame_keys = sorted(scene_gt.keys(), key=lambda x: int(x))[:: args.running_stride]
            for frame_id_str in frame_keys:
                if official_targets is not None and (
                    scene_id,
                    int(frame_id_str),
                    obj_id,
                ) not in official_targets:
                    continue
                gt_list = scene_gt.get(frame_id_str, [])
                for anno_idx, gt_entry in _find_obj_annos(gt_list, obj_id):
                    query_instances.append(
                        {
                            "scene_id": scene_id,
                            "scene_dir": scene_dir,
                            "im_id": int(frame_id_str),
                            "anno_idx": anno_idx,
                            "cam_entry": scene_camera.get(frame_id_str),
                        }
                    )

        frame_count_this_obj = 0
        sampled_frame_instances += len(query_instances)
        for query in tqdm(query_instances, desc=f"{obj_name} instances", leave=False):
            scene_id = query["scene_id"]
            scene_dir = query["scene_dir"]
            im_id = query["im_id"]
            anno_idx = query["anno_idx"]
            cam_entry = query["cam_entry"]
            if cam_entry is None:
                obj_count, frame_count_this_obj = _record_failed_frame(object_metrics, all_frame_data, obj_name, obj_id, im_id, obj_count, frame_count_this_obj, args.count_failures, scene_id=scene_id, anno_idx=anno_idx)
                continue

            color_bgr = cv2.imread(os.path.join(scene_dir, "rgb", f"{im_id:06d}.png"))
            depth_raw = cv2.imread(os.path.join(scene_dir, "depth", f"{im_id:06d}.png"), cv2.IMREAD_ANYDEPTH)
            if color_bgr is None or depth_raw is None:
                obj_count, frame_count_this_obj = _record_failed_frame(object_metrics, all_frame_data, obj_name, obj_id, im_id, obj_count, frame_count_this_obj, args.count_failures, scene_id=scene_id, anno_idx=anno_idx)
                continue
            color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            depth_scale = float(cam_entry.get("depth_scale", 1.0))
            depth_m = depth_raw.astype(np.float32) * depth_scale * MM_TO_M
            mask = _read_mask(scene_dir, str(args.mask_type), im_id, anno_idx)
            if mask is None:
                obj_count, frame_count_this_obj = _record_failed_frame(object_metrics, all_frame_data, obj_name, obj_id, im_id, obj_count, frame_count_this_obj, args.count_failures, scene_id=scene_id, anno_idx=anno_idx)
                continue
            K = _parse_bop_K(cam_entry)
            selection_started = time.perf_counter()
            eval_caches = cand_caches
            candidate_results = []
            for cache in eval_caches:
                cur_est = None
                try:
                    cur_est = Any6D(
                        mesh=cache.mesh,
                        scorer=shared_scorer or ScorePredictor(),
                        refiner=shared_refiner or PoseRefinePredictor(),
                        debug_dir=save_root,
                        debug=0,
                        glctx=glctx,
                    )
                    if hasattr(cur_est, "to_device"):
                        cur_est.to_device(device)
                except Exception as exc:
                    if candidate_error_count < int(args.candidate_error_log_limit):
                        warnings.warn(f"Candidate estimator init failed for {obj_name}/{cache.label}: {type(exc).__name__}: {exc}")
                    candidate_error_count += 1
                    _cuda_cleanup()
                    continue
                pred_pose_a = cand_anchor_poses.get(cache.label, np.eye(4))
                try:
                    pred_pose_q = cur_est.register(
                        K=K,
                        rgb=color,
                        depth=depth_m,
                        ob_mask=mask,
                        iteration=int(args.register_iteration),
                        name=obj_name,
                    )
                    if not _is_valid_pose(pred_pose_q):
                        continue
                    l_depth, l_mask, l_geom, _rendered_appearance, rendered_coverage = compute_observation_consistency(
                        est=cur_est,
                        rgb=color,
                        depth=depth_m,
                        mask=mask,
                        K=K,
                        pred_pose=pred_pose_q,
                        glctx=glctx,
                    )
                    anchor_info = cand_info.get(cache.label, {})
                    candidate_results.append(
                        {
                            "label": cache.label,
                            "pred_pose_q": pred_pose_q,
                            "pred_pose_a": pred_pose_a,
                            "s_anchor": float(cache.s_anchor),
                            "s_anchor_raw": float(
                                anchor_info.get("s_anchor_raw", cache.s_anchor)
                            ),
                            "anchor_observation_quality": float(
                                anchor_info.get(
                                    "anchor_observation_quality", cache.s_anchor
                                )
                            ),
                            "anchor_calibration_norm": float(
                                anchor_info.get("anchor_calibration_norm", 1.0)
                            ),
                            "anchor_reliability": float(
                                anchor_info.get("shared_anchor_reliability", {}).get(
                                    "score", cache.s_anchor
                                )
                            ),
                            "anchor_low_reliability": bool(
                                anchor_info.get("shared_anchor_reliability", {}).get(
                                    "low_reliability", False
                                )
                            ),
                            "l_depth": float(l_depth),
                            "l_mask": float(l_mask),
                            "l_geom": float(l_geom),
                            "l_appearance": 1.0,
                            "coverage": float(np.clip(rendered_coverage, 0.0, 1.0)),
                            "total_score": 0.0,
                        }
                    )
                except Exception as exc:
                    if candidate_error_count < int(args.candidate_error_log_limit):
                        warnings.warn(
                            f"Candidate register failed for {obj_name}/{cache.label} "
                            f"scene={scene_id} im={im_id} anno={anno_idx}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    candidate_error_count += 1
                    _cuda_cleanup()
                finally:
                    if cur_est is not None:
                        del cur_est
                        _cuda_cleanup()

            if not candidate_results:
                registration_failure_frames += 1
                obj_count, frame_count_this_obj = _record_failed_frame(object_metrics, all_frame_data, obj_name, obj_id, im_id, obj_count, frame_count_this_obj, args.count_failures, scene_id=scene_id, anno_idx=anno_idx)
                continue

            score_candidates(
                candidate_results,
                w_depth=args.w_depth,
                w_mask=args.w_mask,
                w_geom=args.w_geom,
                w_appearance=args.w_appearance,
                score_alpha=args.score_alpha,
                score_beta=args.score_beta,
                loss_normalization=args.query_loss_normalization,
                depth_scale=args.query_depth_scale,
                mask_scale=args.query_mask_scale,
                geom_scale=args.query_geom_scale,
                appearance_scale=args.query_appearance_scale,
            )

            candidate_results.sort(key=lambda x: x["total_score"], reverse=True)
            winner = select_scored_candidate(candidate_results)
            best_label = winner["label"]
            pred_pose_q = winner["pred_pose_q"]
            selection_time_s = float(time.perf_counter() - selection_started)
            selection_times.append(selection_time_s)
            # Pose/CAD evaluation begins only after the winner is immutable.
            load_object_evaluation_resources()
            gt_pose_a = _gt_pose_to_meters(np.loadtxt(anchor_gt_pose_path))
            scene_gt_after_selection = _load_scene_json(scene_dir, "scene_gt.json")
            gt_entries_after_selection = scene_gt_after_selection.get(str(im_id), [])
            if anno_idx >= len(gt_entries_after_selection):
                obj_count, frame_count_this_obj = _record_failed_frame(
                    object_metrics, all_frame_data, obj_name, obj_id, im_id,
                    obj_count, frame_count_this_obj, args.count_failures,
                    scene_id=scene_id, anno_idx=anno_idx,
                )
                continue
            gt_pose_q = _gt_pose_to_meters(
                _parse_bop_pose(gt_entries_after_selection[anno_idx])
            )
            for result in candidate_results:
                result["pred_q"] = (
                    result["pred_pose_q"]
                    @ np.linalg.inv(result.get("pred_pose_a", cand_anchor_poses.get(
                        result.get("candidate_label", result["label"]), np.eye(4)
                    )))
                    @ gt_pose_a
                )
            gt_eval = gt_pose_q.astype(np.float64)

            metrics_by_label = {}
            for result in [winner]:
                metrics_by_label[result["label"]] = _evaluate_candidate_pose_metrics(
                    pred_pose=result["pred_q"],
                    gt_pose=gt_eval,
                    gt_vertices_m=gt_mesh_m.vertices,
                    gt_diameter_m=gt_diameter_m,
                    gt_diameter_mm=gt_diameter_mm,
                    is_symmetric=is_symmetric,
                    symmetry_transforms=trans_disc,
                    intrinsic=K,
                    depth_raw=depth_raw,
                    depth_scale=depth_scale,
                    renderer=renderer,
                    obj_id=obj_id,
                )

            for result in candidate_results:
                score_row = {
                        "Scene": f"{scene_id:06d}",
                        "Image_ID": im_id,
                        "Anno_ID": anno_idx,
                        "Class": obj_name,
                        "Candidate": result["label"],
                        "Selected": result is winner,
                        "S_anchor": result["s_anchor"],
                        "S_anchor_norm": result.get("s_anchor_norm", 1.0),
                        "S_anchor_raw": result.get("s_anchor_raw", result["s_anchor"]),
                        "Anchor_Observation_Quality": result.get(
                            "anchor_observation_quality", result["s_anchor"]
                        ),
                        "Anchor_Calibration_Norm": result.get(
                            "anchor_calibration_norm", 1.0
                        ),
                        "Anchor_Reliability": result.get("anchor_reliability", result["s_anchor"]),
                        "Anchor_Low_Reliability": result.get("anchor_low_reliability", False),
                        "L_depth": result["l_depth"],
                        "L_mask": result["l_mask"],
                        "L_geom": result["l_geom"],
                        "L_appearance": result.get("l_appearance", 1.0),
                        "Coverage": result.get("coverage", 0.0),
                        "L_depth_norm": result.get("l_depth_norm", 0.0),
                        "L_mask_norm": result.get("l_mask_norm", 0.0),
                        "L_geom_norm": result.get("l_geom_norm", 0.0),
                        "L_appearance_norm": result.get("l_appearance_norm", 0.0),
                        "Appearance_Mode": args.candidate_color_mode,
                        "S_query": result.get("s_query", 0.0),
                        "Observation_Quality": result.get(
                            "observation_quality", result.get("s_query", 0.0)
                        ),
                        "Total_Score": result["total_score"],
                        "Selection_Time_s": selection_time_s,
                    }
                candidate_score_rows.append(score_row)

            winner_metrics = metrics_by_label[winner["label"]]
            add_thres = winner_metrics["ADD"]
            adds_thres = winner_metrics["ADD-S"]
            add_mix_thres = winner_metrics["ADD(-S)"]
            mean_ar = winner_metrics["AR"]
            mean_mssd = winner_metrics["MSSD"]
            mean_mspd = winner_metrics["MSPD"]
            mean_vsd = winner_metrics["VSD"]
            err_R = winner_metrics["R_error"]
            err_T = winner_metrics["T_error"]

            object_metrics[obj_name]["ADD"].append(add_thres)
            object_metrics[obj_name]["ADD-S"].append(adds_thres)
            object_metrics[obj_name]["ADD(-S)"].append(add_mix_thres)
            object_metrics[obj_name]["AR"].append(mean_ar)
            object_metrics[obj_name]["VSD"].append(mean_vsd)
            object_metrics[obj_name]["MSSD"].append(mean_mssd)
            object_metrics[obj_name]["MSPD"].append(mean_mspd)
            object_metrics[obj_name]["R error"].append(float(err_R))
            object_metrics[obj_name]["T error"].append(float(err_T))
            object_metrics[obj_name]["chosen_candidate"].append(best_label)
            object_metrics[obj_name]["cls_id"].append(obj_name)
            object_metrics[obj_name]["instance_id"].append(obj_count)
            object_metrics[obj_name]["scene_id"].append(scene_id)
            object_metrics[obj_name]["image_id"].append(im_id)
            object_metrics[obj_name]["anno_idx"].append(anno_idx)

            all_frame_data["Frame_ID"].append(obj_count)
            all_frame_data["Scene"].append(f"{scene_id:06d}")
            all_frame_data["Image_ID"].append(im_id)
            all_frame_data["Anno_ID"].append(anno_idx)
            all_frame_data["Class"].append(obj_name)
            all_frame_data["Chosen_Candidate"].append(best_label)
            all_frame_data["ADD-S"].append(adds_thres)
            all_frame_data["ADD"].append(add_thres)
            all_frame_data["ADD(-S)"].append(add_mix_thres)
            all_frame_data["AR"].append(mean_ar)
            all_frame_data["MSSD"].append(mean_mssd)
            all_frame_data["MSPD"].append(mean_mspd)
            all_frame_data["VSD"].append(mean_vsd)
            all_frame_data["R_error"].append(float(err_R))
            all_frame_data["T_error"].append(float(err_T))
            all_frame_data["Selection_Time_s"].append(selection_time_s)

            if args.save_visualizations:
                winning_candidate_label = winner.get("candidate_label", best_label)
                winning_cache = next(
                    (c for c in cand_caches if c.label == winning_candidate_label), None
                )
                vis_est = est
                if winning_cache is not None:
                    swap_candidate(vis_est, winning_cache)
                try:
                    visualize_frame_results_gt(
                        color=color,
                        gt_mesh=gt_mesh_m,
                        K=K,
                        gt_pose=gt_eval,
                        pred_pose=pred_pose_q,
                        metric=object_metrics[obj_name],
                        obj_f=obj_name,
                        frame_idx=im_id,
                        save_path=save_root,
                        glctx=glctx,
                        name=f"toyl_{args.name}",
                        nocs_metric=True,
                        est_mesh=vis_est.mesh,
                    )
                except Exception:
                    pass

            obj_count += 1
            frame_count_this_obj += 1

        _cuda_cleanup()
        n_frames = len(object_metrics[obj_name]["instance_id"])
        if n_frames == 0:
            continue
        class_means = {
            "ADD-S": np.mean(object_metrics[obj_name]["ADD-S"]) * 100,
            "ADD": np.mean(object_metrics[obj_name]["ADD"]) * 100,
            "ADD(-S)": np.mean(object_metrics[obj_name]["ADD(-S)"]) * 100,
            "AR": np.mean(object_metrics[obj_name]["AR"]) * 100,
            "MSSD": np.mean(object_metrics[obj_name]["MSSD"]) * 100,
            "MSPD": np.mean(object_metrics[obj_name]["MSPD"]) * 100,
            "VSD": np.mean(object_metrics[obj_name]["VSD"]) * 100,
            "R_error": _nanmean_or_nan(object_metrics[obj_name]["R error"]),
            "T_error": _nanmean_or_nan(object_metrics[obj_name]["T error"]),
        }
        summary_data.append(
            {
                "Class_ID": obj_name,
                "Obj_ID": obj_id,
                "Frames": n_frames,
                "ADD-S": f"{class_means['ADD-S']:.1f}",
                "ADD": f"{class_means['ADD']:.1f}",
                "ADD(-S)": f"{class_means['ADD(-S)']:.1f}",
                "AR": f"{class_means['AR']:.1f}",
                "MSSD": f"{class_means['MSSD']:.1f}",
                "MSPD": f"{class_means['MSPD']:.1f}",
                "VSD": f"{class_means['VSD']:.1f}",
                "R_error": f"{class_means['R_error']:.1f}",
                "T_error": f"{class_means['T_error']:.1f}",
            }
        )

    evaluated_objs = [obj for obj in obj_names if len(object_metrics[obj]["instance_id"]) > 0]
    if not evaluated_objs:
        print("\nNo objects were evaluated.")
        return
    overall_means = {
        "ADD-S": np.mean([np.mean(object_metrics[o]["ADD-S"]) for o in evaluated_objs]) * 100,
        "ADD": np.mean([np.mean(object_metrics[o]["ADD"]) for o in evaluated_objs]) * 100,
        "ADD(-S)": np.mean([np.mean(object_metrics[o]["ADD(-S)"]) for o in evaluated_objs]) * 100,
        "AR": np.mean([np.mean(object_metrics[o]["AR"]) for o in evaluated_objs]) * 100,
        "MSSD": np.mean([np.mean(object_metrics[o]["MSSD"]) for o in evaluated_objs]) * 100,
        "MSPD": np.mean([np.mean(object_metrics[o]["MSPD"]) for o in evaluated_objs]) * 100,
        "VSD": np.mean([np.mean(object_metrics[o]["VSD"]) for o in evaluated_objs]) * 100,
    }
    summary_data.append(
        {
            "Class_ID": "MEAN",
            "Obj_ID": "",
            "Frames": "",
            "ADD-S": f"{overall_means['ADD-S']:.1f}",
            "ADD": f"{overall_means['ADD']:.1f}",
            "ADD(-S)": f"{overall_means['ADD(-S)']:.1f}",
            "AR": f"{overall_means['AR']:.1f}",
            "MSSD": f"{overall_means['MSSD']:.1f}",
            "MSPD": f"{overall_means['MSPD']:.1f}",
            "VSD": f"{overall_means['VSD']:.1f}",
            "R_error": "",
            "T_error": "",
        }
    )
    pd.DataFrame(summary_data).to_excel(f"{save_root}/0_mean_all_metrics_classes_results.xlsx", index=False)
    df_all = pd.DataFrame(all_frame_data)
    # Counts come from the actual Eq. (22) winner on every evaluated frame.
    requested_candidate_order = list(PAPER_CANDIDATE_LABELS)
    selected_series = df_all["Chosen_Candidate"].astype(str)
    overall_counts = selected_series.value_counts()
    selection_rows = []
    count_labels = list(dict.fromkeys(requested_candidate_order + overall_counts.index.tolist()))
    total_selected = max(int(len(df_all)), 1)
    for label in count_labels:
        count = int(overall_counts.get(label, 0))
        selection_rows.append(
            {
                "Candidate": label,
                "Selected_Count": count,
                "Selected_Fraction": float(count / total_selected),
                "Total_Frames": int(len(df_all)),
            }
        )
    selection_overall_df = pd.DataFrame(selection_rows)
    selection_by_object_df = (
        df_all.groupby(["Class", "Chosen_Candidate"], as_index=False)
        .size()
        .rename(columns={"size": "Selected_Count"})
    )
    object_totals = df_all.groupby("Class").size().rename("Total_Frames")
    selection_by_object_df = selection_by_object_df.join(
        object_totals, on="Class"
    )
    selection_by_object_df["Selected_Fraction"] = (
        selection_by_object_df["Selected_Count"]
        / selection_by_object_df["Total_Frames"].clip(lower=1)
    )
    selection_counts_path = f"{save_root}/0_candidate_selection_counts.xlsx"
    with pd.ExcelWriter(selection_counts_path) as writer:
        selection_overall_df.to_excel(writer, index=False, sheet_name="overall")
        selection_by_object_df.to_excel(
            writer, index=False, sheet_name="by_object"
        )
    with open(
        f"{save_root}/0_candidate_selection_counts.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "total_frames": int(len(df_all)),
                "overall": selection_rows,
            },
            f,
            indent=2,
        )
    print("\nCandidate selection counts (Eq. 22):")
    for row in selection_rows:
        print(
            f"  {row['Candidate']}: {row['Selected_Count']}/"
            f"{row['Total_Frames']} ({100.0 * row['Selected_Fraction']:.2f}%)"
        )
    means_all_row = {
        "Frame_ID": "MEAN",
        "Scene": "ALL",
        "Image_ID": "",
        "Anno_ID": "",
        "Class": "ALL",
        "Chosen_Candidate": "",
        "ADD-S": f"{df_all['ADD-S'].mean() * 100:.1f}",
        "ADD": f"{df_all['ADD'].mean() * 100:.1f}",
        "ADD(-S)": f"{df_all['ADD(-S)'].mean() * 100:.1f}",
        "AR": f"{df_all['AR'].mean() * 100:.1f}",
        "MSSD": f"{df_all['MSSD'].mean() * 100:.1f}",
        "MSPD": f"{df_all['MSPD'].mean() * 100:.1f}",
        "VSD": f"{df_all['VSD'].mean() * 100:.1f}",
        "R_error": f"{df_all['R_error'].mean(skipna=True):.1f}",
        "T_error": f"{df_all['T_error'].mean(skipna=True):.1f}",
        "Selection_Time_s": f"{df_all['Selection_Time_s'].mean(skipna=True):.3f}",
    }
    df_all = pd.concat([df_all, pd.DataFrame([means_all_row])], ignore_index=True)
    all_frames_path = f"{save_root}/0_all_frames_metrics_results.xlsx"
    df_all.to_excel(all_frames_path, index=False)
    candidate_scores_path = f"{save_root}/0_all_candidate_scores.xlsx"
    candidate_scores_df = pd.DataFrame(candidate_score_rows)
    candidate_scores_df.to_excel(candidate_scores_path, index=False)
    run_manifest["completed"] = True
    run_manifest["evaluated_frames"] = int(obj_count)
    run_manifest["evaluated_objects"] = list(evaluated_objs)
    run_manifest["coverage"] = {
        "sampled_frame_instances": int(sampled_frame_instances),
        "evaluated_frame_instances": int(obj_count),
        "skipped_frame_instances": int(max(sampled_frame_instances - obj_count, 0)),
        "registration_failure_frames": int(registration_failure_frames),
    }
    run_manifest["runtime"] = {
        "measurement": "registration through Eq.22 winner selection",
        "frames": int(len(selection_times)),
        "median_seconds": (
            float(np.median(selection_times)) if selection_times else None
        ),
        "p95_seconds": (
            float(np.percentile(selection_times, 95)) if selection_times else None
        ),
    }
    with open(
        os.path.join(save_root, "evaluation_coverage.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_manifest["coverage"], handle, indent=2)
    with open(
        os.path.join(save_root, "query_run_manifest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(run_manifest, f, indent=2)
    print(f"All frames metrics saved to: {all_frames_path}")
    print(f"Per-candidate weighted scores saved to: {candidate_scores_path}")
    print(f"Candidate selection counts saved to: {selection_counts_path}")
    print(f"Total evaluated: {obj_count} frame-object pairs across {len(evaluated_objs)} objects")
    print(f"Results directory: {save_root}")


if __name__ == "__main__":
    main()
