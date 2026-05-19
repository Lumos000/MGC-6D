"""TOYL query phase with paper-style multi-candidate selection."""

from project_paths import setup_project_paths
setup_project_paths()

import argparse
import copy
import gc
import json
import os
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
from bop_toolkit_lib.renderer_vispy import RendererVispy
from estimater import Any6D, PoseRefinePredictor, ScorePredictor
from foundationpose.Utils import nvdiffrast_render, visualize_frame_results_gt
from metrics import compute_RT_distances, compute_add, compute_adds

try:
    from bop_toolkit_lib.misc import format_sym_set, get_symmetry_transformations
except Exception:
    format_sym_set = None
    get_symmetry_transformations = None

TOYL_OBJ_IDS = list(range(1, 22))
MM_TO_M = 0.001


class CandidateCache:
    __slots__ = ("label", "mesh", "s_anchor")

    def __init__(self, label: str, mesh: trimesh.Trimesh, s_anchor: float = 1.0, use_bbox_diameter: bool = False, lazy_tensors: bool = False):
        self.label = label
        self.mesh = mesh
        self.s_anchor = float(s_anchor)


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


def compute_observation_consistency(est: Any6D, depth: np.ndarray, mask: np.ndarray, K: np.ndarray, pred_pose: np.ndarray, glctx) -> tuple[float, float, float]:
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
        return 1.0, 1.0, 1e3
    rendered_depth = rendered[1] if isinstance(rendered, tuple) and len(rendered) >= 2 else None
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
    l_mask = float(1.0 - (obs_mask & ren_mask).sum() / max(float(union.sum()), 1.0))
    l_geom = _chamfer_distance(_backproject_depth_to_points(rd, K, ren_mask), _backproject_depth_to_points(depth, K, obs_mask))
    return l_depth, l_mask, l_geom


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
    trans_disc = [{"R": np.eye(3), "t": np.array([[0.0, 0.0, 0.0]]).T}]
    for sym in model_info_entry.get("symmetries_discrete", []):
        sym_4x4 = np.reshape(sym, (4, 4))
        trans_disc.append({"R": sym_4x4[:3, :3].astype(np.float64), "t": (sym_4x4[:3, 3].reshape((3, 1)) * MM_TO_M).astype(np.float64)})
    return trans_disc


def _build_oryon_style_symmetry(model_info_entry: dict) -> np.ndarray:
    if get_symmetry_transformations is not None and format_sym_set is not None:
        sym_set = get_symmetry_transformations(model_info_entry, max_sym_disc_step=0.05)
        return format_sym_set(sym_set)
    has_sym = bool(model_info_entry.get("symmetries_discrete")) or bool(model_info_entry.get("symmetries_continuous"))
    if has_sym:
        return np.stack([np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)])
    return np.eye(4, dtype=np.float64)[None]


def _get_oryon_add_diameter_m(obj_pts_mm: np.ndarray, model_info_entry: dict) -> float:
    try:
        from utils.pcd import get_diameter as oryon_get_diameter

        return float(oryon_get_diameter(obj_pts_mm)) * MM_TO_M
    except Exception:
        return float(model_info_entry["diameter"]) * MM_TO_M


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
    object_metrics[obj_name]["Oryon ADD(S)-0.1d"].append(0.0)
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
    all_frame_data["Oryon ADD(S)-0.1d"].append(0.0)
    all_frame_data["AR"].append(0.0)
    all_frame_data["MSSD"].append(0.0)
    all_frame_data["MSPD"].append(0.0)
    all_frame_data["VSD"].append(0.0)
    all_frame_data["R_error"].append(float("nan"))
    all_frame_data["T_error"].append(float("nan"))


def _record_failed_frame(object_metrics: dict, all_frame_data: dict, obj_name: str, obj_id: int, im_id: int, obj_count: int, frame_count_this_obj: int, count_failures: bool, scene_id: int | None = None, anno_idx: int | None = None) -> tuple[int, int]:
    if count_failures:
        _append_zero_frame(object_metrics, all_frame_data, obj_name, obj_id, im_id, obj_count, scene_id=scene_id, anno_idx=anno_idx)
        obj_count += 1
        frame_count_this_obj += 1
    return obj_count, frame_count_this_obj


def _nanmean_or_nan(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float("nan") if arr.size == 0 or np.isnan(arr).all() else float(np.nanmean(arr))


def main():
    seed_everything(0)
    parser = argparse.ArgumentParser(description="TOYL query paper-style multi-candidate evaluation")
    parser.add_argument("--name", type=str, default="any6d_toyl_paper")
    parser.add_argument("--anchor_path", type=str, default="/data/gst/Any6D/Any6D/results/anchor_results/toyl_paper/toyl")
    parser.add_argument("--legacy_anchor_path", type=str, default="/data/gst/Any6D/Any6D/results/anchor_results/toyl")
    parser.add_argument("--toyl_root", type=str, default="/data/gst/data/TOYL")
    parser.add_argument("--toyl_model_path", type=str, default="/data/gst/data/TOYL/models")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--running_stride", type=int, default=1)
    parser.add_argument("--register_iteration", type=int, default=5)
    parser.add_argument("--mask_type", type=str, default="mask_visib", choices=["mask_visib", "mask"])
    parser.add_argument("--obj_ids", type=str, default="")
    parser.add_argument("--scene_ids", type=str, default="")
    parser.add_argument("--w_depth", type=float, default=0.5)
    parser.add_argument("--w_mask", type=float, default=0.25)
    parser.add_argument("--w_geom", type=float, default=0.25)
    parser.add_argument("--score_alpha", type=float, default=0.3)
    parser.add_argument("--score_beta", type=float, default=0.7)
    parser.add_argument(
        "--selection_hysteresis",
        type=float,
        default=0.02,
        help="Per-frame winner switch margin; keep previous winner unless new score improves by this value",
    )
    parser.add_argument(
        "--prefer_stable_over_fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a small score penalty to fused candidate in per-frame selection",
    )
    parser.add_argument(
        "--fused_penalty",
        type=float,
        default=0.08,
        help="Extra score added to fused candidate (lower means weaker penalty)",
    )
    parser.add_argument(
        "--query_multitry",
        type=int,
        default=2,
        help="Per-candidate register retries; keep best observation-consistency trial",
    )
    parser.add_argument(
        "--query_try_include_nocoarse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--query_try_include_register",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stable_query_mode",
        type=str,
        default="register_simple",
        choices=["default", "register_simple", "any6d_nocoarse"],
        help="Restrict TOYL query trials to a stable strategy; any6d_nocoarse disables coarse/refinement mesh resets",
    )
    parser.add_argument(
        "--include_legacy_candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the old stable single-anchor TOYL mesh/pose as a candidate safety baseline",
    )
    parser.add_argument(
        "--legacy_candidate_label",
        type=str,
        default="legacy_single",
    )
    parser.add_argument(
        "--legacy_guard_margin",
        type=float,
        default=0.08,
        help="Keep legacy candidate unless the best non-legacy score improves by this margin",
    )
    parser.add_argument(
        "--prefer_legacy_candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--per_candidate_estimator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build an isolated Any6D estimator for each candidate attempt to avoid native reset_object state churn",
    )
    parser.add_argument(
        "--candidate_error_log_limit",
        type=int,
        default=8,
        help="Maximum per-run candidate registration exceptions printed to stderr",
    )
    parser.add_argument("--query_trial_depth_weight", type=float, default=0.25)
    parser.add_argument("--query_trial_mask_weight", type=float, default=0.15)
    parser.add_argument("--query_trial_geom_weight", type=float, default=0.60)
    parser.add_argument("--per_frame_selection", dest="per_frame_selection", action="store_true")
    parser.add_argument("--per_object_selection", dest="per_frame_selection", action="store_false")
    parser.set_defaults(per_frame_selection=True)
    parser.add_argument("--use_bbox_diameter", action="store_true", default=False)
    parser.add_argument("--lazy_tensors", action="store_true", default=False)
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--count_failures", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    eval_obj_ids = _parse_id_list(args.obj_ids) or list(TOYL_OBJ_IDS)
    eval_scene_ids = _parse_id_list(args.scene_ids)
    test_dir = os.path.join(args.toyl_root, "test")
    scene_dirs = _list_scene_dirs(test_dir, eval_scene_ids)
    if not scene_dirs:
        raise RuntimeError(f"No BOP scenes found under {test_dir}")

    registry_path = os.path.join(args.anchor_path, "candidate_registry.json")
    candidate_registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            candidate_registry = json.load(f)
        print(f"Loaded candidate registry with {len(candidate_registry)} objects")
    else:
        print(f"WARNING: candidate_registry.json not found at {registry_path}")

    date_str = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    save_root = f"./results/toyl_results/{args.name}/{date_str}"
    os.makedirs(save_root, exist_ok=True)

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
    renderer = RendererVispy(640, 480, mode="depth")
    candidate_error_count = 0

    obj_names = [_obj_name(oid) for oid in eval_obj_ids]
    object_metrics = {
        obj: {
            "ADD": [],
            "ADD-S": [],
            "ADD(-S)": [],
            "Oryon ADD(S)-0.1d": [],
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
        "Oryon ADD(S)-0.1d": [],
        "AR": [],
        "MSSD": [],
        "MSPD": [],
        "VSD": [],
        "R_error": [],
        "T_error": [],
    }

    obj_count = 0
    summary_data = []

    for obj_id in tqdm(eval_obj_ids, desc="Evaluating Object"):
        obj_name = _obj_name(obj_id)
        obj_anchor_dir = os.path.join(args.anchor_path, obj_name)
        anchor_gt_pose_path = os.path.join(obj_anchor_dir, f"{obj_name}_gt_pose.txt")
        anchor_K_path = os.path.join(obj_anchor_dir, "K.txt")
        if not (os.path.exists(anchor_gt_pose_path) and os.path.exists(anchor_K_path)):
            warnings.warn(f"Skip {obj_name}: missing anchor GT pose/K")
            continue
        gt_pose_a = _gt_pose_to_meters(np.loadtxt(anchor_gt_pose_path))

        obj_registry = candidate_registry.get(obj_name, {})
        obj_candidates = obj_registry.get("candidates", {})
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
                pose_path = info.get("pose_path", "")
                if os.path.exists(pose_path):
                    cand_anchor_poses[label] = np.loadtxt(pose_path)

        if args.include_legacy_candidate:
            legacy_dir = os.path.join(args.legacy_anchor_path, obj_name)
            legacy_mesh = os.path.join(legacy_dir, f"final_mesh_{obj_name}.obj")
            legacy_pose = os.path.join(legacy_dir, f"{obj_name}_initial_pose.txt")
            if os.path.exists(legacy_mesh) and os.path.exists(legacy_pose):
                mesh = trimesh.load(legacy_mesh)
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
                label = str(args.legacy_candidate_label)
                cand_caches.append(CandidateCache(label=label, mesh=mesh, s_anchor=1.0, use_bbox_diameter=args.use_bbox_diameter, lazy_tensors=args.lazy_tensors))
                cand_anchor_poses[label] = np.loadtxt(legacy_pose)
                cand_info[label] = {"l_depth": 1.0, "l_mask": 1.0, "l_geom": 1.0, "s_anchor": 1.0, "legacy": True}
            else:
                warnings.warn(f"Legacy candidate missing for {obj_name}: {legacy_mesh} / {legacy_pose}")

        if not cand_caches:
            fallback_mesh = os.path.join(obj_anchor_dir, f"final_mesh_{obj_name}.obj")
            fallback_pose = os.path.join(obj_anchor_dir, f"{obj_name}_initial_pose.txt")
            if not (os.path.exists(fallback_mesh) and os.path.exists(fallback_pose)):
                warnings.warn(f"Skip {obj_name}: no candidate meshes found")
                continue
            mesh = trimesh.load(fallback_mesh)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
            cand_caches = [CandidateCache(label="fallback", mesh=mesh, s_anchor=1.0, use_bbox_diameter=args.use_bbox_diameter, lazy_tensors=False)]
            cand_anchor_poses["fallback"] = np.loadtxt(fallback_pose)
            cand_info["fallback"] = {"l_depth": 1.0, "l_mask": 1.0, "l_geom": 1.0, "s_anchor": 1.0}

        gt_ply_path = os.path.join(args.toyl_model_path, f"obj_{obj_id:06d}.ply")
        gt_mesh_mm = trimesh.load(gt_ply_path)
        if isinstance(gt_mesh_mm, trimesh.Scene):
            gt_mesh_mm = trimesh.util.concatenate([g for g in gt_mesh_mm.geometry.values() if isinstance(g, trimesh.Trimesh)])
        gt_mesh_m = gt_mesh_mm.copy()
        gt_mesh_m.vertices = gt_mesh_mm.vertices * MM_TO_M
        gt_diameter_mm = float(model_info[str(obj_id)]["diameter"])
        gt_diameter_m = gt_diameter_mm * MM_TO_M
        gt_mesh_dict = {"pts": np.asarray(gt_mesh_mm.vertices, dtype=np.float64), "normals": np.asarray(gt_mesh_mm.face_normals, dtype=np.float64), "faces": np.asarray(gt_mesh_mm.faces, dtype=np.int32)}
        renderer.my_add_object(gt_mesh_dict, obj_id)
        model_info_entry = model_info.get(str(obj_id), {})
        trans_disc = _build_ho3d_style_trans_disc(model_info_entry)
        is_symmetric = bool(model_info_entry.get("symmetries_discrete"))
        oryon_obj_sym = _build_oryon_style_symmetry(model_info_entry)
        oryon_add_diameter_m = _get_oryon_add_diameter_m(gt_mesh_dict["pts"], model_info_entry)

        static_winner = None
        prev_winner_label = None
        if len(cand_caches) > 1 and not args.per_frame_selection:
            valid_l_depth = [float(cand_info[c.label].get("l_depth", 1.0)) for c in cand_caches if c.label in cand_info]
            valid_l_mask = [float(cand_info[c.label].get("l_mask", 1.0)) for c in cand_caches if c.label in cand_info]
            valid_l_geom = [float(cand_info[c.label].get("l_geom", 1.0)) for c in cand_caches if c.label in cand_info]
            min_l_depth = min(valid_l_depth) if valid_l_depth else 1.0
            min_l_mask = min(valid_l_mask) if valid_l_mask else 1.0
            min_l_geom = min(valid_l_geom) if valid_l_geom else 1.0
            max_s_anchor = max(float(c.s_anchor) for c in cand_caches)
            scores = {}
            for c in cand_caches:
                info = cand_info.get(c.label, {})
                s_query = (
                    args.w_depth * (float(info.get("l_depth", min_l_depth)) / max(min_l_depth, 1e-8))
                    + args.w_mask * (float(info.get("l_mask", min_l_mask)) / max(min_l_mask, 1e-8))
                    + args.w_geom * (float(info.get("l_geom", min_l_geom)) / max(min_l_geom, 1e-8))
                )
                s_anchor_norm = float(c.s_anchor) / max(max_s_anchor, 1e-8)
                score = -args.score_alpha * s_anchor_norm + args.score_beta * s_query
                scores[c.label] = score
            static_winner = min(scores.items(), key=lambda kv: kv[1])[0]

        query_instances = []
        for scene_id, scene_dir in scene_dirs:
            scene_gt = _load_scene_json(scene_dir, "scene_gt.json")
            scene_camera = _load_scene_json(scene_dir, "scene_camera.json")
            frame_keys = sorted(scene_gt.keys(), key=lambda x: int(x))[:: args.running_stride]
            for frame_id_str in frame_keys:
                gt_list = scene_gt.get(frame_id_str, [])
                for anno_idx, gt_entry in _find_obj_annos(gt_list, obj_id):
                    query_instances.append(
                        {
                            "scene_id": scene_id,
                            "scene_dir": scene_dir,
                            "im_id": int(frame_id_str),
                            "anno_idx": anno_idx,
                            "gt_entry": gt_entry,
                            "cam_entry": scene_camera.get(frame_id_str),
                        }
                    )

        frame_count_this_obj = 0
        for query in tqdm(query_instances, desc=f"{obj_name} instances", leave=False):
            scene_id = query["scene_id"]
            scene_dir = query["scene_dir"]
            im_id = query["im_id"]
            anno_idx = query["anno_idx"]
            gt_entry = query["gt_entry"]
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
            gt_pose_q = _gt_pose_to_meters(_parse_bop_pose(gt_entry))

            eval_caches = cand_caches if static_winner is None else [c for c in cand_caches if c.label == static_winner]
            candidate_results = []
            for cache in eval_caches:
                cur_est = est
                if args.per_candidate_estimator:
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
                else:
                    swap_candidate(cur_est, cache)
                pred_pose_a = cand_anchor_poses.get(cache.label, np.eye(4))
                if args.stable_query_mode == "register_simple":
                    trial_strategies = [
                        {"name": "register_simple", "use_any6d": False, "axis_align": False, "coarse_est": False, "refinement": False},
                    ]
                elif args.stable_query_mode == "any6d_nocoarse":
                    trial_strategies = [
                        {"name": "any6d_nocoarse_stable", "use_any6d": True, "axis_align": False, "coarse_est": False, "refinement": False},
                    ]
                else:
                    trial_strategies = [
                        {"name": "any6d_coarse_aa", "use_any6d": True, "axis_align": True, "coarse_est": True, "refinement": True},
                        {"name": "any6d_coarse_noaa", "use_any6d": True, "axis_align": False, "coarse_est": True, "refinement": True},
                    ]
                    if args.query_try_include_nocoarse:
                        trial_strategies.append({"name": "any6d_nocoarse", "use_any6d": True, "axis_align": False, "coarse_est": False, "refinement": True})
                    if args.query_try_include_register:
                        trial_strategies.append({"name": "register_simple", "use_any6d": False, "axis_align": False, "coarse_est": False, "refinement": False})

                try:
                    best_trial = None
                    for strat_idx, strat in enumerate(trial_strategies):
                        for try_i in range(max(1, int(args.query_multitry))):
                            trial_seed = int(obj_id * 100003 + im_id * 313 + anno_idx * 17 + strat_idx * 53 + try_i * 7919)
                            np.random.seed(trial_seed % (2**32 - 1))
                            torch.manual_seed(trial_seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(trial_seed)
                            try:
                                if strat["use_any6d"]:
                                    pred_pose_q = cur_est.register_any6d(
                                        K=K,
                                        rgb=color,
                                        depth=depth_m,
                                        ob_mask=mask,
                                        iteration=int(args.register_iteration),
                                        refinement=bool(strat.get("refinement", True)),
                                        axis_align=bool(strat["axis_align"]),
                                        coarse_est=bool(strat["coarse_est"]),
                                        name=obj_name,
                                    )
                                else:
                                    pred_pose_q = cur_est.register(
                                        K=K, rgb=color, depth=depth_m, ob_mask=mask,
                                        iteration=int(args.register_iteration), name=obj_name
                                    )
                            except Exception as exc:
                                if candidate_error_count < int(args.candidate_error_log_limit):
                                    warnings.warn(
                                        f"Candidate register failed for {obj_name}/{cache.label} "
                                        f"scene={scene_id} im={im_id} anno={anno_idx} "
                                        f"strategy={strat['name']}: {type(exc).__name__}: {exc}"
                                    )
                                candidate_error_count += 1
                                _cuda_cleanup()
                                continue
                            if not _is_valid_pose(pred_pose_q):
                                continue
                            try:
                                pred_q = (pred_pose_q @ np.linalg.inv(pred_pose_a)) @ gt_pose_a
                            except np.linalg.LinAlgError:
                                continue
                            if not _is_valid_pose(pred_q):
                                continue
                            if len(eval_caches) > 1 and args.per_frame_selection:
                                l_depth, l_mask, l_geom = compute_observation_consistency(
                                    est=cur_est, depth=depth_m, mask=mask, K=K, pred_pose=pred_pose_q, glctx=glctx
                                )
                            else:
                                l_depth, l_mask, l_geom = 1.0, 1.0, 1.0
                            trial_score = (
                                float(args.query_trial_depth_weight) * float(l_depth)
                                + float(args.query_trial_mask_weight) * float(l_mask)
                                + float(args.query_trial_geom_weight) * float(l_geom)
                            )
                            if best_trial is None or trial_score < best_trial["trial_score"]:
                                best_trial = {
                                    "pred_pose_q": pred_pose_q,
                                    "pred_q": pred_q,
                                    "l_depth": float(l_depth),
                                    "l_mask": float(l_mask),
                                    "l_geom": float(l_geom),
                                    "trial_score": float(trial_score),
                                }
                    if best_trial is not None:
                        candidate_results.append(
                            {
                                "label": cache.label,
                                "pred_pose_q": best_trial["pred_pose_q"],
                                "pred_q": best_trial["pred_q"],
                                "s_anchor": float(cache.s_anchor),
                                "l_depth": float(best_trial["l_depth"]),
                                "l_mask": float(best_trial["l_mask"]),
                                "l_geom": float(best_trial["l_geom"]),
                                "total_score": 0.0,
                            }
                        )
                finally:
                    if args.per_candidate_estimator:
                        del cur_est
                        _cuda_cleanup()

            if not candidate_results:
                obj_count, frame_count_this_obj = _record_failed_frame(object_metrics, all_frame_data, obj_name, obj_id, im_id, obj_count, frame_count_this_obj, args.count_failures, scene_id=scene_id, anno_idx=anno_idx)
                continue

            if len(candidate_results) > 1 and args.per_frame_selection:
                min_l_depth = min(r["l_depth"] for r in candidate_results)
                min_l_mask = min(r["l_mask"] for r in candidate_results)
                min_l_geom = min(r["l_geom"] for r in candidate_results)
                max_s_anchor = max(float(r["s_anchor"]) for r in candidate_results)
                for r in candidate_results:
                    s_query = (
                        args.w_depth * (r["l_depth"] / max(min_l_depth, 1e-8))
                        + args.w_mask * (r["l_mask"] / max(min_l_mask, 1e-8))
                        + args.w_geom * (r["l_geom"] / max(min_l_geom, 1e-8))
                    )
                    s_anchor_norm = float(r["s_anchor"]) / max(max_s_anchor, 1e-8)
                    score = float(-args.score_alpha * s_anchor_norm + args.score_beta * s_query)
                    if bool(args.prefer_stable_over_fused) and r["label"] == "fused":
                        score += float(args.fused_penalty)
                    r["total_score"] = score
            elif len(candidate_results) == 1:
                candidate_results[0]["total_score"] = 0.0

            candidate_results.sort(key=lambda x: x["total_score"])
            winner = candidate_results[0]
            if args.prefer_legacy_candidate and len(candidate_results) > 1:
                legacy_item = next((r for r in candidate_results if r["label"] == str(args.legacy_candidate_label)), None)
                if legacy_item is not None and winner["label"] != str(args.legacy_candidate_label):
                    if (legacy_item["total_score"] - winner["total_score"]) < float(args.legacy_guard_margin):
                        winner = legacy_item
            if (
                args.per_frame_selection
                and prev_winner_label is not None
                and len(candidate_results) > 1
            ):
                prev_item = next((r for r in candidate_results if r["label"] == prev_winner_label), None)
                if prev_item is not None:
                    if (prev_item["total_score"] - winner["total_score"]) < float(args.selection_hysteresis):
                        winner = prev_item
            best_label = winner["label"]
            prev_winner_label = best_label
            pred_q = winner["pred_q"].astype(np.float64)
            pred_pose_q = winner["pred_pose_q"]
            gt_eval = gt_pose_q.astype(np.float64)

            err_R, err_T = compute_RT_distances(pred_q, gt_eval)
            add_val = compute_add(gt_mesh_m.vertices, pred_q, gt_eval)
            adds_val = compute_adds(gt_mesh_m.vertices, pred_q, gt_eval)
            add_thres = float(add_val <= gt_diameter_m * 0.1)
            adds_thres = float(adds_val <= gt_diameter_m * 0.1)
            add_mix_thres = adds_thres if is_symmetric else add_thres
            if np.asarray(oryon_obj_sym).shape[0] > 1:
                oryon_add_val = compute_adds(gt_mesh_dict["pts"] * MM_TO_M, pred_q, gt_eval)
            else:
                oryon_add_val = compute_add(gt_mesh_dict["pts"] * MM_TO_M, pred_q, gt_eval)
            oryon_adds_thres = float(oryon_add_val <= oryon_add_diameter_m * 0.1)
            mssd_err = mssd(pose_est=pred_q, pose_gt=gt_eval, pts=gt_mesh_m.vertices, syms=trans_disc) * 1e3
            mspd_err = mspd(pose_est=pred_q, pose_gt=gt_eval, pts=gt_mesh_m.vertices, K=K, syms=trans_disc)
            pred_r = pred_q[:3, :3]
            pred_t = np.expand_dims(pred_q[:3, 3], axis=1) * 1e3
            gt_r = gt_eval[:3, :3]
            gt_t = np.expand_dims(gt_eval[:3, 3], axis=1) * 1e3
            try:
                vsd_errs = np.asarray(vsd(pred_r, pred_t, gt_r, gt_t, depth_raw.astype(np.float32) * depth_scale, K.reshape(3, 3), 15.0, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], True, gt_diameter_mm, renderer, obj_id))
                mean_vsd = float(np.stack([vsd_errs < rec for rec in np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])], axis=1).mean())
            except Exception:
                mean_vsd = 0.0
            mssd_cur_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]) * gt_diameter_mm
            mean_mssd = float((mssd_err < mssd_cur_rec).mean())
            mean_mspd = float((mspd_err < np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])).mean())
            mean_ar = (mean_mssd + mean_mspd + mean_vsd) / 3.0

            object_metrics[obj_name]["ADD"].append(add_thres)
            object_metrics[obj_name]["ADD-S"].append(adds_thres)
            object_metrics[obj_name]["ADD(-S)"].append(add_mix_thres)
            object_metrics[obj_name]["Oryon ADD(S)-0.1d"].append(oryon_adds_thres)
            object_metrics[obj_name]["AR"].append(mean_ar)
            object_metrics[obj_name]["VSD"].append(mean_vsd)
            object_metrics[obj_name]["MSSD"].append(mean_mssd)
            object_metrics[obj_name]["MSPD"].append(mean_mspd)
            object_metrics[obj_name]["R error"].append(float(np.asarray(err_R).reshape(-1)[0]))
            object_metrics[obj_name]["T error"].append(float(np.asarray(err_T).reshape(-1)[0]))
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
            all_frame_data["Oryon ADD(S)-0.1d"].append(oryon_adds_thres)
            all_frame_data["AR"].append(mean_ar)
            all_frame_data["MSSD"].append(mean_mssd)
            all_frame_data["MSPD"].append(mean_mspd)
            all_frame_data["VSD"].append(mean_vsd)
            all_frame_data["R_error"].append(float(np.asarray(err_R).reshape(-1)[0]))
            all_frame_data["T_error"].append(float(np.asarray(err_T).reshape(-1)[0]))

            if args.save_visualizations:
                winning_cache = next((c for c in cand_caches if c.label == best_label), None)
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
            "Oryon ADD(S)-0.1d": np.mean(object_metrics[obj_name]["Oryon ADD(S)-0.1d"]) * 100,
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
                "Oryon ADD(S)-0.1d": f"{class_means['Oryon ADD(S)-0.1d']:.1f}",
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
        "Oryon ADD(S)-0.1d": np.mean([np.mean(object_metrics[o]["Oryon ADD(S)-0.1d"]) for o in evaluated_objs]) * 100,
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
            "Oryon ADD(S)-0.1d": f"{overall_means['Oryon ADD(S)-0.1d']:.1f}",
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
        "Oryon ADD(S)-0.1d": f"{df_all['Oryon ADD(S)-0.1d'].mean() * 100:.1f}",
        "AR": f"{df_all['AR'].mean() * 100:.1f}",
        "MSSD": f"{df_all['MSSD'].mean() * 100:.1f}",
        "MSPD": f"{df_all['MSPD'].mean() * 100:.1f}",
        "VSD": f"{df_all['VSD'].mean() * 100:.1f}",
        "R_error": f"{df_all['R_error'].mean(skipna=True):.1f}",
        "T_error": f"{df_all['T_error'].mean(skipna=True):.1f}",
    }
    df_all = pd.concat([df_all, pd.DataFrame([means_all_row])], ignore_index=True)
    all_frames_path = f"{save_root}/0_all_frames_metrics_results.xlsx"
    df_all.to_excel(all_frames_path, index=False)
    print(f"All frames metrics saved to: {all_frames_path}")
    print(f"Total evaluated: {obj_count} frame-object pairs across {len(evaluated_objs)} objects")
    print(f"Results directory: {save_root}")


if __name__ == "__main__":
    main()
