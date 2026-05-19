"""
REAL275 query evaluation — v3 (baseline-fair).

Evaluation modes:
  - 'baseline_fair': Matches HO3D baseline protocol exactly.
      * Model vertices scaled by NOCS scale (gt_vertices_eval = base * s).
      * Poses decomposed to pure SE(3) via decompose_nocs_pose.
      * Diameter computed on scaled vertices.
      * ADD/ADD-S threshold = diameter * 0.1 (consistent scale).
      * BOP symmetry from models_info.json discrete transforms (like HO3D).
      * No z-correction, no extra tricks.
  - 'nocs_scaled': Legacy mode (direct_nocs / decomposed_se3 sub-modes).
  - 'physical': Legacy mode (physical metric scale).

The 'baseline_fair' mode is the recommended default for fair comparison.
"""

from project_paths import setup_project_paths
setup_project_paths()

import copy
import os
import argparse
import pickle
import warnings

import cv2
import numpy as np
import pandas as pd
import trimesh
import torch
import nvdiffrast.torch as dr
from tqdm import tqdm

from estimater import *
from metrics import *
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd
from bop_toolkit_lib.renderer_vispy import RendererVispy
from foundationpose.Utils import calc_pts_diameter, make_mesh_tensors, nvdiffrast_render
from pytorch_lightning import seed_everything
from datetime import datetime
import json


# ---------------------------------------------------------------------------
# Symmetry helpers for pose consistency (ported from query_method_ablation.py)
# ---------------------------------------------------------------------------

def _resolve_pose_to_anchor_symmetry(
    pose_q: np.ndarray,
    pose_a: np.ndarray,
    sym_tfs: list,
) -> np.ndarray:
    """Pick the symmetry branch of pose_q closest to pose_a."""
    if len(sym_tfs) <= 1:
        return pose_q
    inv_a = np.linalg.inv(pose_a.astype(np.float64))
    best_pose = pose_q
    best_angle = float("inf")
    for sym_tf in sym_tfs:
        pose_try = pose_q @ sym_tf
        rel = pose_try.astype(np.float64) @ inv_a
        trace = np.clip((np.trace(rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(trace)))
        if angle < best_angle:
            best_angle = angle
            best_pose = pose_try
    return best_pose


def _pose_jump_score(pose_curr: np.ndarray, pose_prev: np.ndarray) -> float:
    rel = pose_curr @ np.linalg.inv(pose_prev)
    r = rel[:3, :3]
    trace_val = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(trace_val)))
    trans_cm = float(np.linalg.norm(rel[:3, 3]) * 100.0)
    return rot_deg + 2.0 * trans_cm


def _resolve_pose_with_symmetry(
    pose_curr: np.ndarray,
    sym_tfs: list,
    pose_prev: np.ndarray | None,
) -> np.ndarray:
    """Pick the symmetry branch of pose_curr closest to pose_prev (temporal)."""
    if pose_prev is None or len(sym_tfs) <= 1:
        return pose_curr
    best_pose = pose_curr
    best_score = float("inf")
    for sym_tf in sym_tfs:
        pose_try = pose_curr @ sym_tf
        score_try = _pose_jump_score(pose_try, pose_prev)
        if score_try < best_score:
            best_score = float(score_try)
            best_pose = pose_try
    return best_pose


def _to_symmetry_4x4_list(trans_disc: list) -> list:
    sym_tfs = []
    for tf in trans_disc:
        r = np.asarray(tf.get("R", np.eye(3)), dtype=np.float64).reshape(3, 3)
        t = np.asarray(tf.get("t", np.zeros((3, 1))), dtype=np.float64).reshape(3)
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = r
        m[:3, 3] = t
        sym_tfs.append(m)
    if not sym_tfs:
        sym_tfs = [np.eye(4, dtype=np.float64)]
    return sym_tfs


# ---------------------------------------------------------------------------
# Observation consistency scoring (Eq. 14 from paper)
# ---------------------------------------------------------------------------

def _backproject_depth_to_points(depth, K, valid_mask, max_points=5000):
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


def _chamfer_distance(pts_a, pts_b):
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return 1e3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.as_tensor(pts_a, dtype=torch.float32, device=device)
    b = torch.as_tensor(pts_b, dtype=torch.float32, device=device)
    dists = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2.0).squeeze(0)
    ch = dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
    return float(ch.cpu().item())


def compute_observation_consistency(est, depth, mask, K, pred_pose, glctx):
    h, w = depth.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_t = torch.as_tensor(pred_pose, device=device, dtype=torch.float32).unsqueeze(0)
    tf = est.get_tf_to_centered_mesh()
    if torch.is_tensor(tf):
        tf = tf.to(device=device, dtype=torch.float32).unsqueeze(0)
    else:
        tf = torch.as_tensor(tf, device=device, dtype=torch.float32).unsqueeze(0)
    ob_in_cams = pose_t @ torch.linalg.inv(tf)
    rendered = nvdiffrast_render(
        mesh=est.mesh, mesh_tensors=est.mesh_tensors,
        ob_in_cams=ob_in_cams, K=K, H=h, W=w, glctx=glctx,
    )
    if rendered is None:
        return 1.0, 1.0, 1e3, 0.0
    rendered_depth = None
    if isinstance(rendered, tuple):
        if len(rendered) >= 2:
            rendered_depth = rendered[1]
    elif hasattr(rendered, "shape") and rendered.shape[-1] > 3:
        rendered_depth = rendered[..., 3]
    if rendered_depth is None:
        return 1.0, 1.0, 1e3, 0.0
    rd = rendered_depth[0] if getattr(rendered_depth, "ndim", 0) > 2 else rendered_depth
    if torch.is_tensor(rd):
        rd = rd.detach().cpu().numpy()
    rd = rd.astype(np.float32)
    obs_mask = (mask > 0) & (depth > 1e-4)
    ren_mask = rd > 1e-4
    obs_count = float(obs_mask.sum())
    overlap = obs_mask & ren_mask
    coverage_ratio = float(overlap.sum()) / max(obs_count, 1.0)
    if overlap.sum() > 10:
        depth_scale = max(float(np.mean(depth[overlap])), 1e-4)
        l_depth = float(np.mean(np.abs(depth[overlap] - rd[overlap])) / depth_scale)
    else:
        l_depth = 1.0
    union = obs_mask | ren_mask
    l_mask = float(1.0 - (obs_mask & ren_mask).sum() / max(float(union.sum()), 1.0))
    obs_pts = _backproject_depth_to_points(depth, K, obs_mask)
    ren_pts = _backproject_depth_to_points(rd, K, ren_mask)
    l_geom = _chamfer_distance(ren_pts, obs_pts)
    return l_depth, l_mask, l_geom, coverage_ratio


REAL275_K = np.array([
    [591.0125, 0.0, 322.525],
    [0.0, 590.16775, 244.11084],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

REAL275_SCENES = [
    'scene_1', 'scene_2', 'scene_3', 'scene_4', 'scene_5', 'scene_6',
]

REAL275_OBJECTS = [
    'bottle_red_stanford_norm',
    'bottle_shampoo_norm',
    'bottle_shengjun_norm',
    'bowl_blue_white_chinese_norm',
    'bowl_shengjun_norm',
    'bowl_white_small_norm',
    'camera_canon_len_norm',
    'camera_canon_wo_len_norm',
    'camera_shengjun_norm',
    'can_arizona_tea_norm',
    'can_green_norm',
    'can_lotte_milk_norm',
    'laptop_air_xin_norm',
    'laptop_alienware_norm',
    'laptop_mac_pro_norm',
    'mug_anastasia_norm',
    'mug_brown_starbucks_norm',
    'mug_daniel_norm',
]

REVOLUTION_SYMMETRIC_OBJECTS = {
    'bottle_red_stanford_norm',
    'bottle_shampoo_norm',
    'bottle_shengjun_norm',
    'bowl_blue_white_chinese_norm',
    'bowl_shengjun_norm',
    'bowl_white_small_norm',
    'can_arizona_tea_norm',
    'can_green_norm',
    'can_lotte_milk_norm',
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def parse_meta(meta_path):
    entries = []
    with open(meta_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                entries.append((int(parts[0]), int(parts[1]), parts[2]))
    return entries


def build_frame_index(data_root):
    object_index: dict[str, list] = {}
    for scene in REAL275_SCENES:
        scene_dir = os.path.join(data_root, 'real_test', scene)
        if not os.path.isdir(scene_dir):
            continue
        for fname in sorted(os.listdir(scene_dir)):
            if not fname.endswith('_meta.txt'):
                continue
            frame_id = int(fname.split('_')[0])
            entries = parse_meta(os.path.join(scene_dir, fname))
            for inst_id, _cls_id, cls_name in entries:
                object_index.setdefault(cls_name, []).append(
                    (scene, frame_id, inst_id)
                )
    for cls_name in object_index:
        object_index[cls_name].sort()
    return object_index


def decompose_nocs_pose(nocs_pose):
    sR = nocs_pose[:3, :3].astype(np.float64)
    t = nocs_pose[:3, 3].astype(np.float64)
    s = float(np.linalg.norm(sR, axis=0).mean())
    if s < 1e-8:
        return s, np.eye(4, dtype=np.float64)
    R = sR / s
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return s, pose


def parse_source_frame(anchor_path, obj_name):
    sf_path = os.path.join(anchor_path, obj_name, 'source_frame.txt')
    if not os.path.exists(sf_path):
        return None, None, None
    info = {}
    with open(sf_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                info[k.strip()] = v.strip()
    return (
        info.get('scene'),
        int(info['frame']) if 'frame' in info else None,
        int(info['instance_id']) if 'instance_id' in info else None,
    )


def load_gt_poses(gt_dir, scene, frame_id):
    path = os.path.join(gt_dir, f"results_real_test_{scene}_{frame_id:04d}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return np.array(pickle.load(f)['gt_RTs'])


# ---------------------------------------------------------------------------
# Symmetry helpers
# ---------------------------------------------------------------------------

def _axis_rotation_matrix(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _build_continuous_symmetry_disc(n_steps=12, axis='z'):
    """Discrete approximation of continuous revolution symmetry around axis."""
    trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]
    for i in range(1, n_steps):
        angle = 2 * np.pi * i / n_steps
        R = _axis_rotation_matrix(axis, angle)
        trans_disc.append({"R": R, "t": np.array([[0, 0, 0]]).T})
    return trans_disc


def _infer_revolution_axis(obj_name, verts):
    """Infer likely revolution axis for REAL275 symmetric objects."""
    ext = verts.max(axis=0) - verts.min(axis=0)
    # REAL275 convention by category:
    # - bottle/can: revolution axis aligns with the longest extent (height)
    # - bowl: revolution axis aligns with the shortest extent (height)
    if obj_name.startswith('bowl_'):
        axis_idx = int(np.argmin(ext))
    elif obj_name.startswith('bottle_') or obj_name.startswith('can_'):
        axis_idx = int(np.argmax(ext))
    else:
        axis_idx = 2
    return ['x', 'y', 'z'][axis_idx]


# ---------------------------------------------------------------------------
# Lightweight z-only depth correction (final prediction, optional)
# ---------------------------------------------------------------------------

def _depth_z_correction(pred_q, depth, mask, K, diameter_m,
                        z_weight=0.3, min_pts=50):
    """Correct ONLY the z-component of the final prediction using depth obs.

    Unlike the full _depth_correct_final_pose (which was harmful for REAL275),
    this only touches z and uses a conservative weight.  This addresses the
    systematic depth bias from relative-pose-chain translation transfer
    without perturbing xy (which the chain estimates well).
    """
    valid = (depth > 1e-6) & np.isfinite(depth) & (mask > 0)
    if int(valid.sum()) < min_pts:
        return pred_q

    z_vals = depth[valid].astype(np.float64)
    lo, hi = np.percentile(z_vals, [15, 85])
    trimmed = z_vals[(z_vals >= lo) & (z_vals <= hi)]
    if len(trimmed) < 16:
        z_obs = float(np.median(z_vals))
    else:
        z_obs = float(np.mean(trimmed))

    pred_z = float(pred_q[2, 3])
    dev = abs(pred_z - z_obs)
    if dev < diameter_m * 0.05 or dev > diameter_m * 5.0:
        return pred_q

    corrected = pred_q.copy()
    corrected[2, 3] = (1.0 - z_weight) * pred_z + z_weight * z_obs
    return corrected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    seed_everything(0)

    parser = argparse.ArgumentParser(description="REAL275 Query Evaluation (v2-fixed)")
    parser.add_argument("--name", type=str, default="any6d_real275")
    parser.add_argument(
        "--anchor_path", type=str,
        default="/data/gst/Any6D/Any6D/results/anchor_results/real275_reference_view",
    )
    parser.add_argument(
        "--real275_data_root", type=str,
        default="/data/gst/data/REAL275",
    )
    parser.add_argument("--running_stride", type=int, default=10)
    parser.add_argument("--register_iteration", type=int, default=5)
    parser.add_argument(
        "--reinit_interval",
        type=int,
        default=25,
        help=(
            "Periodically re-initialize tracking every N successful frames "
            "within a scene (<=0 disables periodic reinit)."
        ),
    )
    parser.add_argument(
        "--oom_retry_iteration",
        type=int,
        default=2,
        help="Fallback register iterations used after CUDA OOM (set <=0 to disable retry)",
    )
    parser.add_argument(
        "--obj_classes", type=str, default="",
        help="Comma-separated object classes to evaluate (empty = all 18)",
    )
    parser.add_argument(
        "--scenes", type=str, default="",
        help="Comma-separated scenes to evaluate (empty = all 6)",
    )
    parser.add_argument("--min_mask_pixels", type=int, default=64)
    parser.add_argument("--symmetry_n_steps", type=int, default=12,
                        help="Discrete steps for revolution-symmetric BOP metrics")
    parser.add_argument("--z_correct_enable", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Enable lightweight z-only depth correction on final pose")
    parser.add_argument("--z_correct_weight", type=float, default=0.3,
                        help="Blend weight for z correction (0=no change, 1=full depth)")
    parser.add_argument(
        "--eval_scale_mode",
        type=str,
        default="baseline_fair",
        choices=["baseline_fair", "physical", "nocs_scaled"],
        help=(
            "Evaluation protocol selector. "
            "'baseline_fair': match HO3D baseline exactly — scale model by "
            "NOCS scale, decompose to SE(3), use discrete symmetry from "
            "models_info.json, no z-correction. "
            "'physical': use REAL275 *_norm.obj in native metric scale and "
            "evaluate on decomposed SE(3) poses. "
            "'nocs_scaled': legacy Any6D-style REAL275 evaluation."
        ),
    )
    parser.add_argument(
        "--models_info_path",
        type=str,
        default="./real275_models_info.json",
        help=(
            "Path to REAL275 models_info.json (BOP format) for discrete "
            "symmetry transforms. Used only when eval_scale_mode=baseline_fair."
        ),
    )
    parser.add_argument(
        "--nocs_scaled_disable_symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When eval_scale_mode=nocs_scaled, disable REAL275-specific BOP "
            "symmetry handling (MSSD/MSPD/VSD). This does NOT affect the "
            "dataset primary ADD(-S) computation, which still uses ADD-S for "
            "known symmetric classes."
        ),
    )
    parser.add_argument(
        "--nocs_scaled_pose_mode",
        type=str,
        default="decomposed_se3",
        choices=["decomposed_se3", "direct_nocs"],
        help=(
            "How to evaluate poses when eval_scale_mode=nocs_scaled. "
            "'decomposed_se3' scales the model by NOCS scale and evaluates on "
            "pure SE(3) poses (recommended; closer to HO3D/Any6D style). "
            "'direct_nocs' keeps the model unscaled and evaluates directly on "
            "NOCS poses carrying scale."
        ),
    )
    parser.add_argument(
        "--nocs_scaled_disable_z_correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When eval_scale_mode=nocs_scaled, force-disable z-only depth "
            "correction to keep the protocol aligned with original Any6D."
        ),
    )
    parser.add_argument(
        "--nocs_scaled_report_add_as_primary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When eval_scale_mode=nocs_scaled, use ADD as the primary summary "
            "metric instead of the dataset primary ADD(-S) summary."
        ),
    )
    # --- Symmetry consistency (ported from ablation) ---
    parser.add_argument(
        "--symmetry_consistency_enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve pose branch by discrete symmetry to reduce frame-to-frame jumps.",
    )
    parser.add_argument(
        "--anchor_symmetry_align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align pred_pose_q to anchor symmetry state before pose transfer.",
    )
    # --- Multi-candidate mesh support ---
    parser.add_argument(
        "--multi_candidate_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable multi-candidate mesh registration with observation consistency selection.",
    )
    parser.add_argument(
        "--candidate_mesh_patterns",
        type=str,
        default="final_mesh_any6d_nocoarse_refine_s0.obj,final_mesh_any6d_nocoarse_refine_s1.obj,final_mesh_any6d_nocoarse_refine_s2.obj",
        help="Comma-separated mesh filenames to use as additional candidates.",
    )
    parser.add_argument("--w_depth", type=float, default=0.4,
                        help="Weight for L_depth in observation consistency")
    parser.add_argument("--w_mask", type=float, default=0.35,
                        help="Weight for L_mask in observation consistency")
    parser.add_argument("--w_geom", type=float, default=0.25,
                        help="Weight for L_geom (Chamfer) in observation consistency")
    parser.add_argument("--w_fp_score", type=float, default=0.15,
                        help="Weight for FoundationPose internal scorer")
    parser.add_argument("--min_coverage", type=float, default=0.3,
                        help="Minimum mask coverage ratio for candidate acceptance")
    args = parser.parse_args()

    anchor_path = args.anchor_path
    data_root = args.real275_data_root
    gt_dir = os.path.join(data_root, 'gts', 'real_test')
    model_dir = os.path.join(data_root, 'obj_models', 'real_test')

    date_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S}'
    save_root = f"./results/real275_results/{args.name}/{date_str}"
    os.makedirs(save_root, exist_ok=True)

    obj_list = list(REAL275_OBJECTS)
    if args.obj_classes.strip():
        requested = {x.strip() for x in args.obj_classes.split(',') if x.strip()}
        obj_list = [o for o in obj_list if o in requested]

    scenes_filter = None
    if args.scenes.strip():
        scenes_filter = {x.strip() for x in args.scenes.split(',') if x.strip()}

    print(f"Building frame index from {data_root} ...")
    object_index = build_frame_index(data_root)
    total_entries = sum(len(v) for v in object_index.values())
    print(f"Indexed {total_entries} object occurrences across "
          f"{len(object_index)} classes")

    K = REAL275_K.copy()

    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(
        trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4))
    )
    mesh_init = trimesh.Trimesh(
        vertices=mesh_tmp.vertices.copy(), faces=mesh_tmp.faces.copy()
    )
    est = Any6D(
        mesh=mesh_init,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=save_root,
        debug=0,
        glctx=glctx,
    )

    renderer = RendererVispy(640, 480, mode='depth')

    object_metrics = {
        obj: {
            'ADD': [], 'ADD-S': [], 'ADD(-S)': [], 'AR': [], 'VSD': [],
            'MSSD': [], 'MSPD': [], 'R error': [], 'T error': [],
            'cls_id': [], 'instance_id': [],
            'scene': [], 'frame_num': [],
        }
        for obj in obj_list
    }
    all_frame_data = {
        'Frame_ID': [], 'Scene': [], 'Frame_Num': [], 'Class': [],
        'ADD-S': [], 'ADD': [], 'ADD(-S)': [], 'AR': [],
        'MSSD': [], 'MSPD': [], 'VSD': [],
        'R_error': [], 'T_error': [],
    }

    data = []
    obj_count = 0
    evaluated_objects = []

    for obj_name in tqdm(obj_list, desc='Evaluating Objects'):
        # ============================================================
        #  1. Load anchor data
        # ============================================================
        obj_anchor_dir = os.path.join(anchor_path, obj_name)
        anchor_pose_path = os.path.join(
            obj_anchor_dir, f"{obj_name}_initial_pose.txt"
        )
        anchor_gt_path = os.path.join(
            obj_anchor_dir, f"{obj_name}_gt_pose.txt"
        )
        anchor_mesh_path = os.path.join(
            obj_anchor_dir, f"final_mesh_{obj_name}.obj"
        )
        anchor_k_path = os.path.join(obj_anchor_dir, "K.txt")

        required = [anchor_pose_path, anchor_gt_path,
                     anchor_mesh_path, anchor_k_path]
        if not all(os.path.exists(p) for p in required):
            print(f"Skip {obj_name}: missing anchor files")
            continue

        pred_pose_a = np.loadtxt(anchor_pose_path)       # SE(3)
        gt_pose_a = np.loadtxt(anchor_gt_path)            # NOCS [sR|t]
        anchor_mesh = trimesh.load(anchor_mesh_path)

        mc_path = os.path.join(obj_anchor_dir, f"{obj_name}_model_center.txt")
        if os.path.exists(mc_path):
            model_center = np.loadtxt(mc_path).ravel()
        else:
            fm_verts = np.asarray(anchor_mesh.vertices, dtype=np.float64)
            model_center = -fm_verts.mean(axis=0)

        offset_tf = np.eye(4, dtype=np.float64)
        offset_tf[:3, 3] = model_center
        pred_pose_a = pred_pose_a @ offset_tf

        # ============================================================
        #  2. Load normalised GT model & compute real-world scale
        # ============================================================
        gt_model_path = os.path.join(model_dir, f"{obj_name}.obj")
        if not os.path.exists(gt_model_path):
            print(f"Skip {obj_name}: GT model not found at {gt_model_path}")
            continue
        gt_mesh_nocs = trimesh.load(gt_model_path)

        nocs_scale, _ = decompose_nocs_pose(gt_pose_a)
        gt_vertices_base = np.asarray(gt_mesh_nocs.vertices, dtype=np.float64)
        use_any6d_nocs_protocol = (args.eval_scale_mode == "nocs_scaled")
        use_baseline_fair = (args.eval_scale_mode == "baseline_fair")

        if use_baseline_fair:
            # baseline_fair: REAL275 _norm.obj vertices are already in physical
            # metric scale (meters). Do NOT multiply by nocs_scale — that is
            # the scale factor inside the NOCS pose [sR|t], not a model scaler.
            # This matches HO3D where gt_mesh.vertices are also metric-scale.
            gt_vertices_eval = gt_vertices_base
        elif use_any6d_nocs_protocol:
            if args.nocs_scaled_pose_mode == "decomposed_se3":
                gt_vertices_eval = gt_vertices_base * nocs_scale
            else:
                gt_vertices_eval = gt_vertices_base
        else:
            gt_vertices_eval = gt_vertices_base
        gt_diameter = calc_pts_diameter(gt_vertices_eval)

        ob_id = REAL275_OBJECTS.index(obj_name) + 1
        renderer.my_add_object(
            {
                'pts': gt_vertices_eval * 1e3,
                'normals': np.asarray(gt_mesh_nocs.face_normals),
                'faces': np.asarray(gt_mesh_nocs.faces),
            },
            ob_id,
        )

        is_symmetric = obj_name in REVOLUTION_SYMMETRIC_OBJECTS

        if use_baseline_fair:
            # baseline_fair: use discrete symmetry from models_info.json,
            # exactly like HO3D baseline does.
            trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]
            sym_axis = 'discrete'
            if os.path.exists(args.models_info_path):
                with open(args.models_info_path, 'r') as f:
                    models_info = json.load(f)
                key = str(ob_id)
                if key in models_info and "symmetries_discrete" in models_info[key]:
                    for sym in models_info[key]["symmetries_discrete"]:
                        sym_4x4 = np.reshape(sym, (4, 4))
                        R = sym_4x4[:3, :3]
                        t = sym_4x4[:3, 3].reshape((3, 1))
                        trans_disc.append({"R": R, "t": t})
            use_symmetry_metrics = len(trans_disc) > 1
        else:
            use_symmetry_metrics = is_symmetric
            if use_any6d_nocs_protocol and args.nocs_scaled_disable_symmetry:
                use_symmetry_metrics = False

            if use_symmetry_metrics:
                sym_axis = _infer_revolution_axis(obj_name, gt_vertices_eval)
                trans_disc = _build_continuous_symmetry_disc(
                    args.symmetry_n_steps, axis=sym_axis
                )
            else:
                sym_axis = 'none'
                trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]

        # ============================================================
        #  3. Collect query frames (exclude anchor, apply stride)
        # ============================================================
        anchor_scene, anchor_frame, anchor_inst = parse_source_frame(
            anchor_path, obj_name
        )

        if obj_name not in object_index:
            print(f"Skip {obj_name}: not found in any frame")
            continue

        all_occ = object_index[obj_name]
        if scenes_filter:
            all_occ = [(s, f, i) for s, f, i in all_occ if s in scenes_filter]

        if anchor_scene is not None:
            all_occ = [
                (s, f, i) for s, f, i in all_occ
                if not (s == anchor_scene and f == anchor_frame
                        and i == anchor_inst)
            ]

        query_frames = []
        for scene in sorted({s for s, _, _ in all_occ}):
            scene_occ = sorted(
                [(s, f, i) for s, f, i in all_occ if s == scene],
                key=lambda x: x[1],
            )
            query_frames.extend(scene_occ[::args.running_stride])

        if not query_frames:
            print(f"Skip {obj_name}: no query frames after filtering/stride")
            continue

        # ============================================================
        #  4. Reset estimator for this object + build symmetry 4x4 list
        # ============================================================
        sym_tfs_list = _to_symmetry_4x4_list(trans_disc)

        # Load multi-candidate meshes if enabled
        candidate_meshes = [("primary", anchor_mesh, pred_pose_a)]
        if args.multi_candidate_enable:
            for pattern in args.candidate_mesh_patterns.split(","):
                pattern = pattern.strip()
                if not pattern:
                    continue
                cand_path = os.path.join(obj_anchor_dir, pattern)
                if os.path.exists(cand_path):
                    cand_mesh = trimesh.load(cand_path)
                    candidate_meshes.append((pattern, cand_mesh, pred_pose_a))

        est.reset_object(mesh=anchor_mesh, symmetry_tfs=None)
        print(f"\n{obj_name}: {len(query_frames)} query frames "
              f"(stride={args.running_stride}, scale={nocs_scale:.4f}, "
              f"diameter={gt_diameter:.4f}m, eval_scale_mode={args.eval_scale_mode}, "
              f"sym={is_symmetric}, sym_eval={use_symmetry_metrics}, "
              f"sym_axis={sym_axis}, candidates={len(candidate_meshes)})")

        # ============================================================
        #  5. Per-frame evaluation loop (SIMPLE — no depth correction)
        # ============================================================
        prev_scene = None
        prev_pose_q = None
        scene_success_count = 0

        for scene, frame_id, inst_id in tqdm(
            query_frames, desc=f"{obj_name}"
        ):
            scene_dir = os.path.join(data_root, 'real_test', scene)

            if scene != prev_scene:
                prev_pose_q = None
                prev_scene = scene
                scene_success_count = 0

            color_path = os.path.join(scene_dir, f"{frame_id:04d}_color.png")
            if not os.path.exists(color_path):
                continue
            color = cv2.cvtColor(
                cv2.imread(color_path), cv2.COLOR_BGR2RGB
            )

            depth_path = os.path.join(scene_dir, f"{frame_id:04d}_depth.png")
            depth = cv2.imread(
                depth_path, cv2.IMREAD_ANYDEPTH
            ).astype(np.float32) / 1e3

            mask_path = os.path.join(scene_dir, f"{frame_id:04d}_mask.png")
            mask_full = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = (mask_full == inst_id).astype(np.bool_)
            if mask.sum() < args.min_mask_pixels:
                continue

            gt_rts = load_gt_poses(gt_dir, scene, frame_id)
            if gt_rts is None or inst_id - 1 >= gt_rts.shape[0]:
                continue
            gt_pose_q = gt_rts[inst_id - 1]
            if not np.isfinite(gt_pose_q).all():
                continue
            q_scale = np.linalg.norm(gt_pose_q[:3, :3], axis=0).mean()
            if q_scale < 1e-4:
                continue

            # --- Registration with multi-candidate + symmetry ---
            use_reinit = (
                args.reinit_interval > 0
                and prev_pose_q is not None
                and scene_success_count > 0
                and (scene_success_count % args.reinit_interval == 0)
            )
            init_pose_cur = None if use_reinit else prev_pose_q

            if args.multi_candidate_enable and len(candidate_meshes) > 1:
                best_cand_score = float("inf")
                best_pred_pose_q = None
                best_cand_label = "primary"
                for cand_label, cand_mesh, cand_pose_a in candidate_meshes:
                    est.reset_object(mesh=cand_mesh, symmetry_tfs=None)
                    try:
                        cand_pred = est.register(
                            K=K, rgb=color, depth=depth, ob_mask=mask,
                            iteration=args.register_iteration, name=obj_name,
                            init_pose=init_pose_cur,
                        )
                    except Exception:
                        continue
                    fp_score = 0.0
                    if hasattr(est, "scores") and est.scores is not None and len(est.scores) > 0:
                        fp_score = float(est.scores[0].cpu())
                    try:
                        l_d, l_m, l_g, cov = compute_observation_consistency(
                            est, depth, mask, K, cand_pred, glctx,
                        )
                    except Exception:
                        l_d, l_m, l_g, cov = 1.0, 1.0, 1e3, 0.0
                    s_query = (args.w_depth * l_d + args.w_mask * l_m + args.w_geom * l_g)
                    if cov < args.min_coverage:
                        s_query += (args.min_coverage - cov) / args.min_coverage * 2.0
                    if fp_score > 1e-6 and args.w_fp_score > 0:
                        s_query -= args.w_fp_score * fp_score
                    if s_query < best_cand_score:
                        best_cand_score = s_query
                        best_pred_pose_q = cand_pred
                        best_cand_label = cand_label
                if best_pred_pose_q is None:
                    continue
                pred_pose_q = best_pred_pose_q
                # Restore primary mesh for visualization
                est.reset_object(mesh=anchor_mesh, symmetry_tfs=None)
            else:
                try:
                    pred_pose_q = est.register(
                        K=K, rgb=color, depth=depth, ob_mask=mask,
                        iteration=args.register_iteration, name=obj_name,
                        init_pose=init_pose_cur,
                    )
                except Exception as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg and args.oom_retry_iteration > 0:
                        warnings.warn(
                            f"OOM on {obj_name} {scene}/{frame_id}, "
                            f"retry with iteration={args.oom_retry_iteration}"
                        )
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            pred_pose_q = est.register(
                                K=K, rgb=color, depth=depth, ob_mask=mask,
                                iteration=args.oom_retry_iteration, name=obj_name,
                                init_pose=None,
                            )
                        except Exception as exc2:
                            warnings.warn(
                                f"Register retry failed: {obj_name} {scene}/{frame_id}: {exc2}"
                            )
                            continue
                    else:
                        warnings.warn(
                            f"Register failed: {obj_name} {scene}/{frame_id}: {exc}"
                        )
                        continue

            # --- Symmetry consistency resolution ---
            if args.anchor_symmetry_align and len(sym_tfs_list) > 1:
                pred_pose_q = _resolve_pose_to_anchor_symmetry(
                    pred_pose_q, pred_pose_a, sym_tfs_list
                )
            if args.symmetry_consistency_enable and len(sym_tfs_list) > 1:
                pred_pose_q = _resolve_pose_with_symmetry(
                    pred_pose_q, sym_tfs_list, prev_pose_q
                )

            prev_pose_q = pred_pose_q
            scene_success_count += 1

            # --- Relative pose transfer ---
            try:
                pose_aq = pred_pose_q @ np.linalg.inv(pred_pose_a)
            except np.linalg.LinAlgError:
                continue
            pred_q_world = pose_aq @ gt_pose_a

            # --- Choose evaluation pose protocol ---
            if use_baseline_fair:
                # baseline_fair: always decompose to pure SE(3), like HO3D
                _, pred_eval = decompose_nocs_pose(pred_q_world)
                _, gt_eval = decompose_nocs_pose(gt_pose_q)
            elif use_any6d_nocs_protocol:
                if args.nocs_scaled_pose_mode == "direct_nocs":
                    pred_eval = pred_q_world.astype(np.float64)
                    gt_eval = gt_pose_q.astype(np.float64)
                else:
                    _, pred_eval = decompose_nocs_pose(pred_q_world)
                    _, gt_eval = decompose_nocs_pose(gt_pose_q)
            else:
                _, pred_eval = decompose_nocs_pose(pred_q_world)
                _, gt_eval = decompose_nocs_pose(gt_pose_q)

            if (not np.isfinite(pred_eval).all()
                    or not np.isfinite(gt_eval).all()):
                continue

            # --- Optional: lightweight z-only correction ---
            # baseline_fair: never use z-correction (match HO3D baseline)
            use_z_correction = args.z_correct_enable
            if use_baseline_fair:
                use_z_correction = False
            elif use_any6d_nocs_protocol and args.nocs_scaled_disable_z_correction:
                use_z_correction = False
            if use_z_correction:
                pred_eval = _depth_z_correction(
                    pred_eval, depth, mask, K, gt_diameter,
                    z_weight=args.z_correct_weight,
                )

            # --- Rotation / translation error ---
            err_R, err_T = compute_RT_distances(pred_eval, gt_eval)
            if err_R is None or (np.isscalar(err_R) and err_R == -1):
                continue

            # --- ADD / ADD-S ---
            add = compute_add(gt_vertices_eval, pred_eval, gt_eval)
            adds = compute_adds(gt_vertices_eval, pred_eval, gt_eval)
            if not np.isfinite(add) or not np.isfinite(adds):
                continue

            add_thres = float(add <= gt_diameter * 0.1)
            adds_thres = float(adds <= gt_diameter * 0.1)
            # REAL275 primary metric is ADD(-S): symmetric classes always use
            # ADD-S, regardless of whether BOP symmetry-aware evaluation is
            # enabled for MSSD/MSPD/VSD.
            add_combo = adds_thres if is_symmetric else add_thres

            # --- BOP metrics ---
            # baseline_fair: use float64 to match HO3D baseline precision.
            pred_eval_bop = pred_eval.astype(np.float64)
            gt_eval_bop = gt_eval.astype(np.float64)

            pred_r = pred_eval_bop[:3, :3]
            pred_t = np.expand_dims(pred_eval_bop[:3, 3], axis=1) * 1e3
            gt_r = gt_eval_bop[:3, :3]
            gt_t = np.expand_dims(gt_eval_bop[:3, 3], axis=1) * 1e3

            mssd_err = mssd(
                pose_est=pred_eval_bop, pose_gt=gt_eval_bop,
                pts=gt_vertices_eval, syms=trans_disc,
            ) * 1e3
            mspd_err = mspd(
                pose_est=pred_eval_bop, pose_gt=gt_eval_bop,
                pts=gt_vertices_eval, K=K, syms=trans_disc,
            )

            mssd_rec = np.array(
                [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            )
            mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

            vsd_delta = 15.0
            vsd_taus = list(mssd_rec)
            vsd_rec = np.array(list(mssd_rec))

            try:
                vsd_errs = vsd(
                    pred_r, pred_t, gt_r, gt_t,
                    depth * 1e3, K.reshape(3, 3),
                    vsd_delta, vsd_taus, True,
                    gt_diameter * 1e3, renderer, ob_id,
                )
                vsd_errs = np.asarray(vsd_errs)
                all_vsd_recs = np.stack(
                    [vsd_errs < rec_i for rec_i in vsd_rec], axis=1
                )
                mean_vsd = all_vsd_recs.mean()
            except Exception:
                mean_vsd = 0.0

            mssd_cur_rec = mssd_rec * (gt_diameter * 1e3)
            mean_mssd = (mssd_err < mssd_cur_rec).mean()
            mean_mspd = (mspd_err < mspd_rec).mean()
            mean_ar = (mean_mssd + mean_mspd + mean_vsd) / 3.0

            # --- Store results ---
            err_R_val = (
                err_R.tolist()[0] if hasattr(err_R, 'tolist') else float(err_R)
            )
            err_T_val = (
                err_T.tolist()[0] if hasattr(err_T, 'tolist') else float(err_T)
            )

            m = object_metrics[obj_name]
            m['ADD'].append(add_thres)
            m['ADD-S'].append(adds_thres)
            m['ADD(-S)'].append(add_combo)
            m['AR'].append(mean_ar)
            m['VSD'].append(mean_vsd)
            m['MSSD'].append(mean_mssd)
            m['MSPD'].append(mean_mspd)
            m['R error'].append(err_R_val)
            m['T error'].append(err_T_val)
            m['cls_id'].append(obj_name)
            m['instance_id'].append(obj_count)
            m['scene'].append(scene)
            m['frame_num'].append(frame_id)

            try:
                visualize_frame_results_gt(
                    color=color, gt_mesh=gt_mesh_nocs, K=K,
                    gt_pose=gt_pose_q, pred_pose=pred_pose_q,
                    metric=m, obj_f=obj_name,
                    frame_idx=frame_id, save_path=save_root, glctx=glctx,
                    name=f"{len(query_frames)}_{args.name}",
                    nocs_metric=True, est_mesh=est.mesh,
                )
            except Exception:
                pass
            obj_count += 1

        # ============================================================
        #  6. Per-object summary
        # ============================================================
        m = object_metrics[obj_name]
        if len(m['ADD']) == 0:
            print(f"Skip {obj_name}: no valid evaluation frames")
            continue

        evaluated_objects.append(obj_name)

        df_obj = pd.DataFrame({
            'Frame_ID': m['instance_id'],
            'Scene': m['scene'],
            'Frame_Num': m['frame_num'],
            'Class': m['cls_id'],
            'ADD-S': m['ADD-S'],
            'ADD': m['ADD'],
            'ADD(-S)': m['ADD(-S)'],
            'AR': m['AR'],
            'MSSD': m['MSSD'],
            'MSPD': m['MSPD'],
            'VSD': m['VSD'],
            'R_error': m['R error'],
            'T_error': m['T error'],
        })

        means_obj = {
            'ADD-S': np.mean(m['ADD-S']) * 100,
            'ADD':   np.mean(m['ADD']) * 100,
            'ADD(-S)': np.mean(m['ADD(-S)']) * 100,
            'AR':    np.mean(m['AR']) * 100,
            'MSSD':  np.mean(m['MSSD']) * 100,
            'MSPD':  np.mean(m['MSPD']) * 100,
            'VSD':   np.mean(m['VSD']) * 100,
            'R_error': np.mean(m['R error']),
            'T_error': np.mean(m['T error']),
        }

        mean_row = pd.DataFrame({
            'Frame_ID': ['MEAN'], 'Scene': [''], 'Frame_Num': [''],
            'Class': [obj_name],
            'ADD-S': [f"{means_obj['ADD-S']:.1f}"],
            'ADD':   [f"{means_obj['ADD']:.1f}"],
            'ADD(-S)': [f"{means_obj['ADD(-S)']:.1f}"],
            'AR':    [f"{means_obj['AR']:.1f}"],
            'MSSD':  [f"{means_obj['MSSD']:.1f}"],
            'MSPD':  [f"{means_obj['MSPD']:.1f}"],
            'VSD':   [f"{means_obj['VSD']:.1f}"],
            'R_error': [f"{means_obj['R_error']:.1f}"],
            'T_error': [f"{means_obj['T_error']:.1f}"],
        })
        df_obj = pd.concat([df_obj, mean_row], ignore_index=True)

        data.append({
            'Class_ID': obj_name,
            'ADD-S': f"{means_obj['ADD-S']:.1f}",
            'ADD':   f"{means_obj['ADD']:.1f}",
            'ADD(-S)': f"{means_obj['ADD(-S)']:.1f}",
            'AR':    f"{means_obj['AR']:.1f}",
            'MSSD':  f"{means_obj['MSSD']:.1f}",
            'MSPD':  f"{means_obj['MSPD']:.1f}",
            'VSD':   f"{means_obj['VSD']:.1f}",
        })

        df_obj.to_excel(
            f'{save_root}/{obj_name}_metrics_results.xlsx', index=False
        )

        all_frame_data['Frame_ID'].extend(m['instance_id'])
        all_frame_data['Scene'].extend(m['scene'])
        all_frame_data['Frame_Num'].extend(m['frame_num'])
        all_frame_data['Class'].extend(m['cls_id'])
        all_frame_data['ADD-S'].extend(m['ADD-S'])
        all_frame_data['ADD'].extend(m['ADD'])
        all_frame_data['ADD(-S)'].extend(m['ADD(-S)'])
        all_frame_data['AR'].extend(m['AR'])
        all_frame_data['MSSD'].extend(m['MSSD'])
        all_frame_data['MSPD'].extend(m['MSPD'])
        all_frame_data['VSD'].extend(m['VSD'])
        all_frame_data['R_error'].extend(m['R error'])
        all_frame_data['T_error'].extend(m['T error'])

    # ================================================================
    #  7. Aggregate results across all objects
    # ================================================================
    if evaluated_objects:
        overall_means = {
            k: np.mean([
                np.mean(object_metrics[o][k]) for o in evaluated_objects
            ]) * 100
            for k in ['ADD-S', 'ADD', 'ADD(-S)', 'AR', 'MSSD', 'MSPD', 'VSD']
        }
        data.append({
            'Class_ID': 'MEAN',
            **{k: f"{v:.1f}" for k, v in overall_means.items()},
        })

        primary_add_key = 'ADD(-S)'
        if args.eval_scale_mode == 'nocs_scaled' and args.nocs_scaled_report_add_as_primary:
            primary_add_key = 'ADD'

        latex_str = (
            f"MEAN & {overall_means['AR']:.1f} & {overall_means['VSD']:.1f}"
            f" & {overall_means['MSSD']:.1f} & {overall_means['MSPD']:.1f}"
            f" & {overall_means[primary_add_key]:.1f} & - \\\\"
        )
        print("\n" + latex_str)
    else:
        print("No objects were evaluated.")

    df = pd.DataFrame(data)
    df.to_excel(
        f'{save_root}/0_mean_all_metrics_classes_results.xlsx', index=False
    )

    df_all = pd.DataFrame(all_frame_data)
    if len(df_all) > 0:
        means_all = {
            'Frame_ID': 'MEAN', 'Scene': '', 'Frame_Num': '',
            'Class': 'ALL',
            'ADD-S': f"{df_all['ADD-S'].mean() * 100:.1f}",
            'ADD':   f"{df_all['ADD'].mean() * 100:.1f}",
            'ADD(-S)': f"{df_all['ADD(-S)'].mean() * 100:.1f}",
            'AR':    f"{df_all['AR'].mean() * 100:.1f}",
            'MSSD':  f"{df_all['MSSD'].mean() * 100:.1f}",
            'MSPD':  f"{df_all['MSPD'].mean() * 100:.1f}",
            'VSD':   f"{df_all['VSD'].mean() * 100:.1f}",
            'R_error': f"{df_all['R_error'].mean():.1f}",
            'T_error': f"{df_all['T_error'].mean():.1f}",
        }
        df_all = pd.concat(
            [df_all, pd.DataFrame([means_all])], ignore_index=True
        )

    output_path = f'{save_root}/0_all_frames_metrics_results.xlsx'
    df_all.to_excel(output_path, index=False)
    print(f"\nAll frames metrics saved to {output_path}")
    print(f"\nEvaluated {len(evaluated_objects)}/{len(obj_list)} objects, "
          f"{obj_count} total frames")
    print("\nSaved data preview:")
    print(df_all)
