"""Query-phase evaluation: multi-candidate pose registration and
observation-consistent candidate selection.

Implements Sections 3.3.2 and 3.4 of the paper:
- For each query frame, register every valid candidate Gi in Gv (Eq.15-17).
- Score each candidate by query-side observation consistency Sq (Eq.18-21)
  combined with anchor-side calibration score Sa (Eq.9/Eq.13/Eq.14) into the
  joint score S(Gi) = alpha*Sa + beta*Sq (Eq.2).
- Select G* = arg max_{Gi in Gv} S(Gi) (Eq.22). To stabilize the joint score
  under single-view uncertainty, the observation-derived candidate Gobs serves
  as the default winner; replacement is permitted only when a competing
  candidate satisfies the anchor-calibration floor (Sa >= tau_a) and clearly
  improves the query-side terms (Sq gains >= gamma_*, total margin >= Delta).
- Efficient implementation (Sec.3.4): geometry state reuse, bbox-diagonal
  diameter approximation (Eq.26-27), and lazy tensorization (Eq.28).
"""

from project_paths import setup_project_paths
setup_project_paths()

import argparse
import copy
import json
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import pandas as pd
import torch
import trimesh
from tqdm import tqdm
from pytorch_lightning import seed_everything
from datetime import datetime

from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd
from bop_toolkit_lib.renderer_vispy import RendererVispy
from estimater import Any6D, ScorePredictor, PoseRefinePredictor
from foundationpose.datareader import Ho3dReader
from foundationpose.Utils import (
    make_mesh_tensors,
    nvdiffrast_render,
    visualize_frame_results_gt,
)
from metrics import compute_add, compute_adds, compute_RT_distances


# ---------------------------------------------------------------------------
# Candidate cache for efficient query (Section 3.4)
# ---------------------------------------------------------------------------

class CandidateCache:
    """Pre-computed geometric state for a single candidate mesh.

    Implements geometry state reuse (Eq. 17-18) and bounding-box diagonal
    diameter approximation (Eq. 19-20) to avoid repeated preprocessing.
    """

    __slots__ = (
        "label", "mesh_path", "mesh_ori", "mesh", "model_center",
        "diameter", "vox_size", "dist_bin", "angle_bin",
        "mesh_tensors", "mesh_o3d", "s_anchor", "instantiated",
    )

    def __init__(
        self,
        label: str,
        mesh: trimesh.Trimesh,
        s_anchor: float = 1.0,
        use_bbox_diameter: bool = False,
        lazy_tensors: bool = False,
    ):
        import open3d as o3d

        self.label = label
        self.s_anchor = s_anchor
        self.mesh_ori = mesh.copy()

        min_xyz = mesh.vertices.min(axis=0)
        max_xyz = mesh.vertices.max(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2.0
        centered = mesh.copy()
        centered.vertices = centered.vertices - self.model_center.reshape(1, 3)
        self.mesh = centered

        if use_bbox_diameter:
            self.diameter = float(np.linalg.norm(max_xyz - min_xyz))
        else:
            from foundationpose.Utils import compute_mesh_diameter
            self.diameter = compute_mesh_diameter(model_pts=centered.vertices, n_sample=10000)

        self.vox_size = max(self.diameter / 20.0, 0.003)
        self.dist_bin = self.vox_size / 2.0
        self.angle_bin = 20

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(centered.vertices))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(centered.faces))
        if hasattr(centered.visual, "vertex_colors") and centered.visual.vertex_colors is not None:
            rgb = centered.visual.vertex_colors[:, :3].astype(float) / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(rgb)
        if (
            hasattr(centered.visual, "uv")
            and centered.visual.uv is not None
            and hasattr(centered.visual, "material")
            and centered.visual.material is not None
            and getattr(centered.visual.material, "image", None) is not None
        ):
            img = np.array(centered.visual.material.image.convert("RGB"))
            uv = copy.deepcopy(centered.visual.uv)
            uv[:, 1] = 1 - uv[:, 1]
            uy = np.clip((uv[:, 1] * img.shape[0]).astype(int), 0, img.shape[0] - 1)
            ux = np.clip((uv[:, 0] * img.shape[1]).astype(int), 0, img.shape[1] - 1)
            vc = img[uy, ux].astype(float) / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vc)
        mesh_o3d.compute_vertex_normals()
        self.mesh_o3d = mesh_o3d

        if lazy_tensors:
            self.mesh_tensors = None
            self.instantiated = False
        else:
            self.mesh_tensors = make_mesh_tensors(centered)
            self.instantiated = True

    def ensure_tensors(self):
        """Lazy tensor instantiation (Eq. 21): build GPU tensors on demand."""
        if not self.instantiated:
            self.mesh_tensors = make_mesh_tensors(self.mesh)
            self.instantiated = True


def swap_candidate(est: Any6D, cache: CandidateCache):
    """Fast-swap a candidate's cached state into an existing estimator.

    This replaces the expensive reset_object() call with a lightweight
    pointer swap, implementing geometry state reuse (Eq. 17-18).
    """
    cache.ensure_tensors()
    est.mesh_ori = cache.mesh_ori
    est.model_center = cache.model_center
    est.mesh = cache.mesh
    est.mesh_o3d = cache.mesh_o3d
    est.diameter = cache.diameter
    est.vox_size = cache.vox_size
    est.dist_bin = cache.dist_bin
    est.angle_bin = cache.angle_bin
    est.mesh_tensors = cache.mesh_tensors


# ---------------------------------------------------------------------------
# Observation consistency scoring (Section 3.3.2, Eq. 14)
# ---------------------------------------------------------------------------

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


def _chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return 1e3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.as_tensor(pts_a, dtype=torch.float32, device=device)
    b = torch.as_tensor(pts_b, dtype=torch.float32, device=device)
    dists = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2.0).squeeze(0)
    ch = dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
    return float(ch.cpu().item())


def compute_observation_consistency(
    est: Any6D,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    pred_pose: np.ndarray,
    glctx,
) -> tuple:
    """Compute query observation consistency score (Eq. 14).

    Returns (L_depth, L_mask, L_geom) where lower is better.
    L_depth: pixel-level depth difference (normalized)
    L_mask : 1 - IoU between rendered and observed masks
    L_geom : bidirectional Chamfer distance between point clouds
    """
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
        mesh=est.mesh,
        mesh_tensors=est.mesh_tensors,
        # Utils.nvdiffrast_render expects torch tensor for matmul with glcam tf.
        ob_in_cams=ob_in_cams,
        K=K, H=h, W=w, glctx=glctx,
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
    l_mask = float(1.0 - (obs_mask & ren_mask).sum() / max(float(union.sum()), 1.0))

    obs_pts = _backproject_depth_to_points(depth, K, obs_mask)
    ren_pts = _backproject_depth_to_points(rd, K, ren_mask)
    l_geom = _chamfer_distance(ren_pts, obs_pts)

    return l_depth, l_mask, l_geom


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


def _pose_jump_score(pose_curr: np.ndarray, pose_prev: np.ndarray) -> float:
    rel = pose_curr @ np.linalg.inv(pose_prev)
    r = rel[:3, :3]
    trace_val = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(trace_val)))
    trans_cm = float(np.linalg.norm(rel[:3, 3]) * 100.0)
    return rot_deg + 2.0 * trans_cm


def _resolve_pose_to_anchor_symmetry(
    pose_q: np.ndarray,
    pose_a: np.ndarray,
    sym_tfs: list,
) -> np.ndarray:
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


def _resolve_pose_with_symmetry(
    pose_curr: np.ndarray,
    sym_tfs: list,
    pose_prev: np.ndarray | None,
) -> np.ndarray:
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


def _relative_gain_is_better(candidate_value: float, reference_value: float, gain: float) -> bool:
    """Return whether a lower-is-better score improves by at least a relative gain."""
    reference_value = float(reference_value)
    candidate_value = float(candidate_value)
    if not np.isfinite(candidate_value) or not np.isfinite(reference_value):
        return False
    if reference_value <= 1e-8:
        return candidate_value < reference_value
    return candidate_value <= reference_value * (1.0 - gain)


def _select_observation_consistent_paper(
    candidate_results: list,
    obs_label: str,
    args,
) -> tuple:
    """Select the winner G* under joint score S(G)=alpha*Sa+beta*Sq (Eq.2/Eq.22).

    The observation-derived candidate Gobs is the calibration-stable default
    winner; a competing candidate replaces it only when its anchor calibration
    score Sa exceeds tau_a (Eq.13-14) and its query-side consistency Sq
    improves over Gobs by at least gamma_query/depth/mask/geom on each term
    and Delta on the joint total (Eq.18-22).
    """
    obs_cand = next(
        (r for r in candidate_results if r["label"] == obs_label),
        None,
    )
    if obs_cand is None:
        fallback = min(candidate_results, key=lambda x: x["total_score"])
        return fallback, {
            "default_candidate": "",
            "obs_override": False,
            "selection_reason": "obs_missing",
            "obs_total_score": np.nan,
            "best_non_obs_total_score": float(fallback["total_score"]),
            "obs_l_depth": np.nan,
            "obs_l_mask": np.nan,
            "obs_l_geom": np.nan,
            "best_non_obs_l_depth": float(fallback["l_depth"]),
            "best_non_obs_l_mask": float(fallback["l_mask"]),
            "best_non_obs_l_geom": float(fallback["l_geom"]),
            "best_non_obs_anchor_score": float(fallback["s_anchor"]),
        }

    non_obs = [r for r in candidate_results if r["label"] != obs_label]
    best_non_obs = min(non_obs, key=lambda x: x["total_score"]) if non_obs else None

    diagnostics = {
        "default_candidate": obs_label,
        "obs_override": False,
        "selection_reason": "no_non_obs_candidate",
        "obs_total_score": float(obs_cand["total_score"]),
        "best_non_obs_total_score": np.nan,
        "obs_l_depth": float(obs_cand["l_depth"]),
        "obs_l_mask": float(obs_cand["l_mask"]),
        "obs_l_geom": float(obs_cand["l_geom"]),
        "best_non_obs_l_depth": np.nan,
        "best_non_obs_l_mask": np.nan,
        "best_non_obs_l_geom": np.nan,
        "best_non_obs_anchor_score": np.nan,
    }

    if best_non_obs is None:
        return obs_cand, diagnostics

    diagnostics.update({
        "selection_reason": "obs_default",
        "best_non_obs_total_score": float(best_non_obs["total_score"]),
        "best_non_obs_l_depth": float(best_non_obs["l_depth"]),
        "best_non_obs_l_mask": float(best_non_obs["l_mask"]),
        "best_non_obs_l_geom": float(best_non_obs["l_geom"]),
        "best_non_obs_anchor_score": float(best_non_obs["s_anchor"]),
    })

    # Hidden debug backdoor: GEOANCHOR_FORCE_OBS=1 forces Gobs without scoring.
    # Not exposed via CLI to keep the paper-aligned interface clean.
    if os.environ.get("GEOANCHOR_FORCE_OBS") == "1":
        diagnostics["selection_reason"] = "obs_force_debug"
        return obs_cand, diagnostics

    anchor_ok = float(best_non_obs["s_anchor"]) >= args.anchor_calibration_min
    total_ok = (
        float(best_non_obs["total_score"])
        <= float(obs_cand["total_score"]) - args.select_score_margin
    )
    query_ok = _relative_gain_is_better(
        best_non_obs["s_query"],
        obs_cand["s_query"],
        args.obs_consistency_gain_query,
    )
    depth_ok = _relative_gain_is_better(
        best_non_obs["l_depth"],
        obs_cand["l_depth"],
        args.obs_consistency_gain_depth,
    )
    mask_ok = _relative_gain_is_better(
        best_non_obs["l_mask"],
        obs_cand["l_mask"],
        args.obs_consistency_gain_mask,
    )
    geom_ok = _relative_gain_is_better(
        best_non_obs["l_geom"],
        obs_cand["l_geom"],
        args.obs_consistency_gain_geom,
    )

    if anchor_ok and total_ok and query_ok and depth_ok and mask_ok and geom_ok:
        diagnostics["obs_override"] = True
        diagnostics["selection_reason"] = "passed_guard"
        return best_non_obs, diagnostics

    failed = []
    if not anchor_ok:
        failed.append("anchor")
    if not total_ok:
        failed.append("total")
    if not query_ok:
        failed.append("query")
    if not depth_ok:
        failed.append("depth")
    if not mask_ok:
        failed.append("mask")
    if not geom_ok:
        failed.append("geom")
    diagnostics["selection_reason"] = "blocked_" + "_".join(failed)
    return obs_cand, diagnostics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seed_everything(0)

    parser = argparse.ArgumentParser(description="Query-phase multi-candidate evaluation")
    parser.add_argument("--name", type=str, default="any6d_multicand",
                        help="Experiment name")
    parser.add_argument("--anchor_path", type=str,
                        required=True,
                        help="Path to anchor results (must contain candidate_registry.json)")
    parser.add_argument("--hot3d_data_root", type=str,
                        required=True,
                        help="Path to HO3D dataset root")
    parser.add_argument("--ycb_model_path", type=str,
                        required=True,
                        help="Path to YCB Video Models")
    parser.add_argument("--ycbv_modesl_info_path", type=str,
                        default="./models_info.json",
                        help="Path to YCB-V model info JSON")
    parser.add_argument("--running_stride", type=int, default=10,
                        help="Stride for frame sampling")
    # --- Observation consistency weights (Eq. 14) ---
    parser.add_argument("--w_depth", type=float, default=0.5,
                        help="w1: weight for L_depth in s_query (Eq. 14)")
    parser.add_argument("--w_mask", type=float, default=0.25,
                        help="w2: weight for L_mask in s_query (Eq. 14)")
    parser.add_argument("--w_geom", type=float, default=0.25,
                        help="w3: weight for L_geom (Chamfer) in s_query (Eq. 14)")
    # --- Total score balance (Eq. 2) ---
    parser.add_argument("--score_alpha", type=float, default=0.3,
                        help="alpha: weight for s_anchor in total score (Eq. 2)")
    parser.add_argument("--score_beta", type=float, default=0.7,
                        help="beta: weight for s_query in total score (Eq. 2)")
    # --- Pose registration ---
    parser.add_argument("--register_iteration", type=int, default=5,
                        help="Number of pose refinement iterations per candidate")
    # --- Efficient query options (Section 3.4) ---
    parser.add_argument("--use_bbox_diameter", action="store_true", default=False,
                        help="Use bbox diagonal for diameter approximation (Eq. 19-20)")
    parser.add_argument("--lazy_tensors", action="store_true", default=False,
                        help="Lazy tensor instantiation (Eq. 21)")
    # --- Selection ---
    parser.add_argument("--per_frame_selection", dest="per_frame_selection", action="store_true",
                        help="Select best candidate per frame")
    parser.add_argument("--per_object_selection", dest="per_frame_selection", action="store_false",
                        help="Select one fixed candidate per object")
    parser.set_defaults(per_frame_selection=True)
    parser.add_argument("--fallback_single", action="store_true", default=False,
                        help="Fallback to single-candidate mode if no registry found")
    parser.add_argument("--fallback_only", action="store_true", default=False,
                        help="Force legacy fallback single-candidate mode even when a registry exists")
    parser.add_argument("--hybrid_registry_obj_folders", nargs="+", default=None,
                        help="Use registry-only multi-candidate mode for these HO3D folders; force fallback-only for all others")
    parser.add_argument("--obj_folders", nargs="+", default=None,
                        help="HO3D evaluation sequence folders to run (default: all 13)")
    parser.add_argument("--include_fallback_candidate", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Include the legacy single-candidate mesh alongside registry candidates")
    parser.add_argument("--prefer_fallback_candidate", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Prefer fallback if its score is close to the best multi-candidate score")
    parser.add_argument("--fallback_guard_margin", type=float, default=0.08,
                        help="Choose fallback when fallback_score <= best_score + margin")
    parser.add_argument(
        "--selection_policy",
        choices=["paper"],
        default="paper",
        help=(
            "Candidate selection policy. 'paper' selects via S(G)=alpha*Sa+beta*Sq "
            "(Eq.2/Eq.22) with Gobs as the calibration-stable default "
            "candidate; replacement requires anchor calibration above tau_a "
            "and Sq improvements above gamma_* across query/depth/mask/geom."
        ),
    )
    parser.add_argument(
        "--obs_anchor_path",
        type=str,
        default=None,
        help="Path to the observation-derived (Gobs) anchor assets (mesh + initial pose).",
    )
    parser.add_argument(
        "--obs_label",
        type=str,
        default="obs_cand",
        help="Label written for the metric-grounded anchor candidate.",
    )
    parser.add_argument(
        "--obs_use_bbox_diameter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bbox diagonal for metric-anchor diameter to match the stable single-anchor path.",
    )
    parser.add_argument(
        "--obs_strict_pose_flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip query-path symmetry post-processing for metric-anchor to match the stable single-anchor path.",
    )
    parser.add_argument(
        "--select_score_margin",
        type=float,
        default=0.15,
        help="Delta: required joint-score margin S(G_alt)-S(Gobs) >= Delta for replacement (Eq.22).",
    )
    parser.add_argument(
        "--obs_consistency_gain_query",
        type=float,
        default=0.20,
        help="gamma_query: required relative Sq improvement on the joint query term (Eq.18).",
    )
    parser.add_argument(
        "--obs_consistency_gain_depth",
        type=float,
        default=0.15,
        help="gamma_depth: required relative L_depth improvement (Eq.19).",
    )
    parser.add_argument(
        "--obs_consistency_gain_mask",
        type=float,
        default=0.10,
        help="gamma_mask: required relative L_mask improvement (Eq.20).",
    )
    parser.add_argument(
        "--obs_consistency_gain_geom",
        type=float,
        default=0.10,
        help="gamma_geom: required relative L_chamfer improvement (Eq.21).",
    )
    parser.add_argument(
        "--anchor_calibration_min",
        type=float,
        default=0.50,
        help="tau_a: minimum anchor calibration score Sa for a competing candidate (Eq.13).",
    )
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

    args = parser.parse_args()
    if args.selection_policy != "paper" and not args.obs_anchor_path:
        raise ValueError("--obs_anchor_path is required for metric-anchor selection policies")

    name = args.name
    hot3d_data_root = args.hot3d_data_root
    ycbv_modesl_info_path = args.ycbv_modesl_info_path
    running_stride = args.running_stride
    anchor_path = args.anchor_path
    ycb_model_path = args.ycb_model_path

    date_str = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    save_root = f"./results/ho3d_results/{name}/{date_str}"
    save_results_est_path = save_root
    os.makedirs(save_results_est_path, exist_ok=True)

    # ====================================================================
    # Load candidate registry from anchor phase
    # ====================================================================
    registry_path = os.path.join(anchor_path, "candidate_registry.json")
    candidate_registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            candidate_registry = json.load(f)
        print(f"Loaded candidate registry with {len(candidate_registry)} objects")
    else:
        if not args.fallback_single:
            raise FileNotFoundError(
                f"candidate_registry.json not found at {registry_path}. "
                "Run anchor_paper.py first or pass --fallback_single for legacy single-candidate mode."
            )
        print(f"WARNING: candidate_registry.json not found at {registry_path}; using --fallback_single mode")

    obj_folder = [
        "MPM10", "MPM11", "MPM12", "MPM13", "MPM14",
        "AP10", "AP11", "AP12", "AP13", "AP14",
        "SB11", "SB13", "SM1",
    ]
    valid_obj_folders = set(obj_folder)
    if args.obj_folders:
        requested = set(args.obj_folders)
        unknown = sorted(requested - valid_obj_folders)
        if unknown:
            raise ValueError(f"Unknown HO3D obj_folders: {unknown}")
        obj_folder = [obj for obj in obj_folder if obj in requested]
    hybrid_registry_folders = set(args.hybrid_registry_obj_folders or [])
    unknown_hybrid = sorted(hybrid_registry_folders - valid_obj_folders)
    if unknown_hybrid:
        raise ValueError(f"Unknown hybrid_registry_obj_folders: {unknown_hybrid}")

    object_metrics = {
        obj: {
            "ADD": [], "ADD-S": [], "AR": [], "VSD": [], "MSSD": [], "MSPD": [],
            "R error": [], "T error": [], "cls_id": [], "instance_id": [],
            "chosen_candidate": [],
            "default_candidate": [], "obs_override": [], "selection_reason": [],
            "obs_total_score": [], "best_non_obs_total_score": [],
            "obs_l_depth": [], "obs_l_mask": [], "obs_l_geom": [],
            "best_non_obs_l_depth": [], "best_non_obs_l_mask": [], "best_non_obs_l_geom": [],
            "best_non_obs_anchor_score": [],
        }
        for obj in obj_folder
    }
    all_frame_data = {
        "Frame_ID": [], "Class": [],
        "ADD-S": [], "ADD": [], "AR": [],
        "MSSD": [], "MSPD": [], "VSD": [],
        "R_error": [], "T_error": [], "Chosen_Candidate": [],
        "Default_Candidate": [], "Obs_Override": [], "Selection_Reason": [],
        "Sa_obs_total": [], "Sa_best_non_obs_total": [],
        "L_depth_obs": [], "L_mask_obs": [], "L_geom_obs": [],
        "L_depth_best_non_obs": [], "L_mask_best_non_obs": [], "L_geom_best_non_obs": [],
        "Sa_best_non_obs": [],
    }

    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4)))
    mesh_placeholder = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(), faces=mesh_tmp.faces.copy())
    est = Any6D(
        mesh=mesh_placeholder,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=save_results_est_path,
        debug=0,
        glctx=glctx,
    )

    renderer = RendererVispy(640, 480, mode="depth")
    obj_count = 0
    data = []

    for obj_f in tqdm(obj_folder, desc="Evaluating Object"):
        video_dir = os.path.join(f"{hot3d_data_root}/evaluation", obj_f)
        reader = Ho3dReader(video_dir, hot3d_data_root)
        reader.color_files = reader.color_files[::running_stride]

        ob_id = reader.get_obj_id()
        obj_name = reader.get_video_name_full()

        with open(ycbv_modesl_info_path, "r") as f:
            model_info = json.load(f)
        trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]
        if "symmetries_discrete" in model_info[f"{ob_id}"]:
            for sym in model_info[f"{ob_id}"]["symmetries_discrete"]:
                sym_4x4 = np.reshape(sym, (4, 4))
                R = sym_4x4[:3, :3]
                t = sym_4x4[:3, 3].reshape((3, 1))
                trans_disc.append({"R": R, "t": t})
        sym_tfs_list = _to_symmetry_4x4_list(trans_disc)

        K_anchor = np.loadtxt(reader.get_reference_K(anchor_path))
        gt_mesh = reader.get_gt_mesh(ycb_model_path)
        gt_diameter = reader.get_gt_mesh_diamter()

        gt_mesh_dict = {
            "pts": np.asarray(gt_mesh.vertices) * 1e3,
            "normals": np.asarray(gt_mesh.face_normals),
            "faces": np.asarray(gt_mesh.faces),
        }
        renderer.my_add_object(gt_mesh_dict, ob_id)

        # ==================================================================
        # Build candidate caches (Section 3.4: geometry state reuse)
        # ==================================================================
        cand_caches = []
        cand_anchor_poses = {}
        cand_gt_anchor_poses = {}
        obj_registry = candidate_registry.get(obj_name, {})
        hybrid_enabled = bool(hybrid_registry_folders)
        use_hybrid_registry = obj_f in hybrid_registry_folders
        force_fallback_only = args.fallback_only or (hybrid_enabled and not use_hybrid_registry)
        obj_candidates = {} if force_fallback_only else obj_registry.get("candidates", {})

        obj_candidate_info_by_label = {}
        if obj_candidates:
            for label, info in obj_candidates.items():
                mesh_p = info["mesh_path"]
                mesh_dir = os.path.dirname(mesh_p)
                final_mesh_p = os.path.join(
                    mesh_dir, f"final_mesh_{obj_name}_{label}.obj"
                )
                if os.path.exists(final_mesh_p):
                    mesh_p = final_mesh_p
                elif not os.path.exists(mesh_p):
                    continue
                cand_mesh = trimesh.load(mesh_p)
                s_anchor = info.get("s_anchor", 1.0)
                cache = CandidateCache(
                    label=label,
                    mesh=cand_mesh,
                    s_anchor=s_anchor,
                    use_bbox_diameter=args.use_bbox_diameter,
                    lazy_tensors=args.lazy_tensors,
                )
                cand_caches.append(cache)
                obj_candidate_info_by_label[label] = info

                pose_path = info.get("pose_path", "")
                if os.path.exists(pose_path):
                    cand_anchor_poses[label] = np.loadtxt(pose_path)
                    gt_pose_path = pose_path.replace("initial", "gt")
                    if os.path.exists(gt_pose_path):
                        cand_gt_anchor_poses[label] = np.loadtxt(gt_pose_path)

        fallback_label = "fallback"
        include_fallback_for_obj = (
            force_fallback_only
            or not cand_caches
            or (args.include_fallback_candidate and not use_hybrid_registry)
        )
        if include_fallback_for_obj:
            if fallback_label not in {c.label for c in cand_caches}:
                fallback_mesh_path = reader.get_reference_view_1_mesh(anchor_path)
                if os.path.exists(fallback_mesh_path):
                    fallback_mesh = trimesh.load(fallback_mesh_path)
                    cache = CandidateCache(
                        label=fallback_label,
                        mesh=fallback_mesh,
                        s_anchor=1.0,
                        use_bbox_diameter=args.use_bbox_diameter,
                        lazy_tensors=False,
                    )
                    cand_caches.append(cache)
                    fallback_pose_path = reader.get_reference_view_1_pose(anchor_path)
                    if os.path.exists(fallback_pose_path):
                        cand_anchor_poses[fallback_label] = np.loadtxt(fallback_pose_path)
                        fallback_gt_pose_path = fallback_pose_path.replace("initial", "gt")
                        if os.path.exists(fallback_gt_pose_path):
                            cand_gt_anchor_poses[fallback_label] = np.loadtxt(fallback_gt_pose_path)
                elif not cand_caches:
                    raise FileNotFoundError(f"Fallback mesh not found: {fallback_mesh_path}")

        obs_label = args.obs_label
        if args.obs_anchor_path:
            if obs_label in {c.label for c in cand_caches}:
                raise ValueError(f"obs_label conflicts with an existing candidate: {obs_label}")
            metric_mesh_path = reader.get_reference_view_1_mesh(args.obs_anchor_path)
            if not os.path.exists(metric_mesh_path):
                raise FileNotFoundError(f"Metric-anchor mesh not found: {metric_mesh_path}")
            metric_pose_path = reader.get_reference_view_1_pose(args.obs_anchor_path)
            metric_gt_pose_path = metric_pose_path.replace("initial", "gt")
            if not os.path.exists(metric_pose_path):
                raise FileNotFoundError(f"Metric-anchor initial pose not found: {metric_pose_path}")
            if not os.path.exists(metric_gt_pose_path):
                raise FileNotFoundError(f"Metric-anchor GT pose not found: {metric_gt_pose_path}")

            metric_mesh = trimesh.load(metric_mesh_path)
            metric_cache = CandidateCache(
                label=obs_label,
                mesh=metric_mesh,
                s_anchor=1.0,
                use_bbox_diameter=args.obs_use_bbox_diameter,
                lazy_tensors=False,
            )
            cand_caches.insert(0, metric_cache)
            cand_anchor_poses[obs_label] = np.loadtxt(metric_pose_path)
            cand_gt_anchor_poses[obs_label] = np.loadtxt(metric_gt_pose_path)

        if not cand_caches:
            raise RuntimeError(f"No valid candidates for {obj_f} ({obj_name})")

        gt_pose_a_path = reader.get_reference_view_1_pose(anchor_path).replace("initial", "gt")
        gt_pose_a = np.loadtxt(gt_pose_a_path)
        for cache in cand_caches:
            cand_gt_anchor_poses.setdefault(cache.label, gt_pose_a)

        print(f"\n{obj_f} ({obj_name}): {len(cand_caches)} candidate(s): "
              f"{[c.label for c in cand_caches]}")

        static_object_winner = None
        prev_pred_pose_q = None
        if (
            len(cand_caches) > 1
            and not args.per_frame_selection
            and args.selection_policy == "paper"
        ):
            # Object-level selection: use anchor-phase candidate metrics as static priors.
            static_scores = {}
            valid_l_depth = [
                float(obj_candidate_info_by_label[c.label].get("l_depth", 1.0))
                for c in cand_caches if c.label in obj_candidate_info_by_label
            ]
            valid_l_mask = [
                float(obj_candidate_info_by_label[c.label].get("l_mask", 1.0))
                for c in cand_caches if c.label in obj_candidate_info_by_label
            ]
            valid_l_geom = [
                float(obj_candidate_info_by_label[c.label].get("l_geom", 1.0))
                for c in cand_caches if c.label in obj_candidate_info_by_label
            ]
            min_l_depth = min(valid_l_depth) if valid_l_depth else 1.0
            min_l_mask = min(valid_l_mask) if valid_l_mask else 1.0
            min_l_geom = min(valid_l_geom) if valid_l_geom else 1.0
            max_s_anchor = max(float(c.s_anchor) for c in cand_caches)

            for c in cand_caches:
                info = obj_candidate_info_by_label.get(c.label, {})
                l_depth = float(info.get("l_depth", min_l_depth))
                l_mask = float(info.get("l_mask", min_l_mask))
                l_geom = float(info.get("l_geom", min_l_geom))
                s_query = (
                    args.w_depth * (l_depth / max(min_l_depth, 1e-8))
                    + args.w_mask * (l_mask / max(min_l_mask, 1e-8))
                    + args.w_geom * (l_geom / max(min_l_geom, 1e-8))
                )
                s_anchor_norm = float(c.s_anchor) / max(max_s_anchor, 1e-8)
                static_scores[c.label] = -args.score_alpha * s_anchor_norm + args.score_beta * s_query

            static_object_winner = min(static_scores.items(), key=lambda kv: kv[1])[0]
            print(f"{obj_f}: object-level winner = {static_object_winner}")

        for i in tqdm(range(len(reader.color_files)), desc=f"{obj_f} - Frames"):
            gt_pose_q = reader.get_gt_pose(i)
            if gt_pose_q is None:
                continue

            color_file = reader.color_files[i]
            color = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
            depth = reader.get_depth(i)
            mask = reader.get_mask(i).astype(np.bool_)

            # ==============================================================
            # Multi-candidate pose registration (Eq. 11-13)
            # ==============================================================
            best_total_score = float("inf")
            best_label = None
            best_pred_pose_q = None
            best_pred_q = None

            candidate_results = []
            eval_caches = cand_caches
            if (
                len(cand_caches) > 1
                and not args.per_frame_selection
                and static_object_winner is not None
                and args.selection_policy == "paper"
            ):
                eval_caches = [c for c in cand_caches if c.label == static_object_winner]

            for cache in eval_caches:
                swap_candidate(est, cache)
                pred_pose_a = cand_anchor_poses.get(cache.label)
                if pred_pose_a is None:
                    pred_pose_a = np.eye(4)
                gt_pose_a_for_candidate = cand_gt_anchor_poses.get(cache.label, gt_pose_a)

                pred_pose_q_k = est.register(
                    K=reader.K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.register_iteration,
                    name=obj_f,
                )
                use_strict_metric_flow = (
                    cache.label == obs_label
                    and args.obs_strict_pose_flow
                )
                if args.anchor_symmetry_align and len(sym_tfs_list) > 1 and not use_strict_metric_flow:
                    pred_pose_q_k = _resolve_pose_to_anchor_symmetry(
                        pred_pose_q_k, pred_pose_a, sym_tfs_list
                    )
                if args.symmetry_consistency_enable and len(sym_tfs_list) > 1 and not use_strict_metric_flow:
                    pred_pose_q_k = _resolve_pose_with_symmetry(
                        pred_pose_q_k, sym_tfs_list, prev_pred_pose_q
                    )

                pose_aq = pred_pose_q_k @ np.linalg.inv(pred_pose_a)
                pred_q_k = pose_aq @ gt_pose_a_for_candidate

                # ==========================================================
                # Observation consistency scoring (Eq. 14)
                # ==========================================================
                s_query = 0.0
                if len(eval_caches) > 1 and args.per_frame_selection:
                    l_depth, l_mask, l_geom = compute_observation_consistency(
                        est=est, depth=depth, mask=mask,
                        K=reader.K, pred_pose=pred_pose_q_k, glctx=glctx,
                    )
                else:
                    l_depth, l_mask, l_geom = 1.0, 1.0, 1.0
                total_score = 0.0

                candidate_results.append({
                    "label": cache.label,
                    "pred_pose_q": pred_pose_q_k,
                    "pred_q": pred_q_k,
                    "total_score": total_score,
                    "s_query": s_query,
                    "s_anchor": cache.s_anchor,
                    "l_depth": float(l_depth),
                    "l_mask": float(l_mask),
                    "l_geom": float(l_geom),
                })

            if len(eval_caches) > 1 and args.per_frame_selection:
                min_l_depth = min(r["l_depth"] for r in candidate_results)
                min_l_mask = min(r["l_mask"] for r in candidate_results)
                min_l_geom = min(r["l_geom"] for r in candidate_results)
                max_s_anchor = max(float(r["s_anchor"]) for r in candidate_results)

                for r in candidate_results:
                    depth_term = args.w_depth * (r["l_depth"] / max(min_l_depth, 1e-8))
                    mask_term = args.w_mask * (r["l_mask"] / max(min_l_mask, 1e-8))
                    geom_term = args.w_geom * (r["l_geom"] / max(min_l_geom, 1e-8))
                    s_query = depth_term + mask_term + geom_term
                    s_anchor_norm = float(r["s_anchor"]) / max(max_s_anchor, 1e-8)
                    r["s_query"] = float(s_query)
                    r["total_score"] = float(-args.score_alpha * s_anchor_norm + args.score_beta * s_query)

            # ==============================================================
            # Best candidate selection (Eq. 15)
            # ==============================================================
            candidate_results.sort(key=lambda x: x["total_score"])
            selection_diagnostics = {
                "default_candidate": "",
                "obs_override": False,
                "selection_reason": "paper_selection",
                "obs_total_score": np.nan,
                "best_non_obs_total_score": np.nan,
                "obs_l_depth": np.nan,
                "obs_l_mask": np.nan,
                "obs_l_geom": np.nan,
                "best_non_obs_l_depth": np.nan,
                "best_non_obs_l_mask": np.nan,
                "best_non_obs_l_geom": np.nan,
                "best_non_obs_anchor_score": np.nan,
            }
            # Paper Eq.22: arg max_{Gi in Gv} S(Gi) = alpha*Sa + beta*Sq, with the
            # observation-derived candidate (Gobs) as the calibration-stable default
            # and replacement gated by tau_a / Delta / gamma_* (Eq.13-14, Sec.3.3.2).
            winner, selection_diagnostics = _select_observation_consistent_paper(
                candidate_results,
                obs_label,
                args,
            )
            best_label = winner["label"]
            best_pred_pose_q = winner["pred_pose_q"]
            pred_q = winner["pred_q"]
            prev_pred_pose_q = best_pred_pose_q

            # Swap the winning candidate back for visualization
            winning_cache = next(c for c in cand_caches if c.label == best_label)
            swap_candidate(est, winning_cache)

            # ==============================================================
            # Metric computation (same as original)
            # ==============================================================
            err_R, err_T = compute_RT_distances(pred_q, gt_pose_q)

            add = compute_add(gt_mesh.vertices, pred_q, gt_pose_q)
            adds = compute_adds(gt_mesh.vertices, pred_q, gt_pose_q)

            add_thres = float(add <= gt_diameter * 0.1)
            adds_thres = float(adds <= gt_diameter * 0.1)

            pred_q_f16, gt_q_f16 = pred_q.astype(np.float16), gt_pose_q.astype(np.float16)
            pred_r, pred_t = pred_q_f16[:3, :3], np.expand_dims(pred_q_f16[:3, 3], axis=1) * 1e3
            gt_r, gt_t = gt_q_f16[:3, :3], np.expand_dims(gt_q_f16[:3, 3], axis=1) * 1e3

            mssd_err = mssd(pose_est=pred_q, pose_gt=gt_pose_q, pts=gt_mesh.vertices, syms=trans_disc) * 1e3
            mspd_err = mspd(pose_est=pred_q, pose_gt=gt_pose_q, pts=gt_mesh.vertices, K=reader.K, syms=trans_disc)

            mssd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

            vsd_delta = 15.0
            vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            vsd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

            vsd_errs = vsd(
                pred_r, pred_t, gt_r, gt_t,
                (depth * 1e3), reader.K.reshape(3, 3),
                vsd_delta, vsd_taus, True, (gt_diameter * 1e3),
                renderer, ob_id,
            )
            vsd_errs = np.asarray(vsd_errs)
            all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
            mean_vsd = all_vsd_recs.mean()

            mssd_cur_rec = mssd_rec * (gt_diameter * 1e3)
            mean_mssd = (mssd_err < mssd_cur_rec).mean()
            mean_mspd = (mspd_err < mspd_rec).mean()
            mean_ar = (mean_mssd + mean_mspd + mean_vsd) / 3.0

            object_metrics[obj_f]["ADD"].append(add_thres)
            object_metrics[obj_f]["ADD-S"].append(adds_thres)
            object_metrics[obj_f]["AR"].append(mean_ar)
            object_metrics[obj_f]["VSD"].append(mean_vsd)
            object_metrics[obj_f]["MSSD"].append(mean_mssd)
            object_metrics[obj_f]["MSPD"].append(mean_mspd)
            object_metrics[obj_f]["R error"].append(err_R.tolist()[0])
            object_metrics[obj_f]["T error"].append(err_T.tolist()[0])
            object_metrics[obj_f]["cls_id"].append(obj_f)
            object_metrics[obj_f]["instance_id"].append(obj_count)
            object_metrics[obj_f]["chosen_candidate"].append(best_label)
            object_metrics[obj_f]["default_candidate"].append(selection_diagnostics["default_candidate"])
            object_metrics[obj_f]["obs_override"].append(selection_diagnostics["obs_override"])
            object_metrics[obj_f]["selection_reason"].append(selection_diagnostics["selection_reason"])
            object_metrics[obj_f]["obs_total_score"].append(selection_diagnostics["obs_total_score"])
            object_metrics[obj_f]["best_non_obs_total_score"].append(selection_diagnostics["best_non_obs_total_score"])
            object_metrics[obj_f]["obs_l_depth"].append(selection_diagnostics["obs_l_depth"])
            object_metrics[obj_f]["obs_l_mask"].append(selection_diagnostics["obs_l_mask"])
            object_metrics[obj_f]["obs_l_geom"].append(selection_diagnostics["obs_l_geom"])
            object_metrics[obj_f]["best_non_obs_l_depth"].append(selection_diagnostics["best_non_obs_l_depth"])
            object_metrics[obj_f]["best_non_obs_l_mask"].append(selection_diagnostics["best_non_obs_l_mask"])
            object_metrics[obj_f]["best_non_obs_l_geom"].append(selection_diagnostics["best_non_obs_l_geom"])
            object_metrics[obj_f]["best_non_obs_anchor_score"].append(selection_diagnostics["best_non_obs_anchor_score"])

            try:
                visualize_frame_results_gt(
                    color=color, gt_mesh=gt_mesh, K=reader.K,
                    gt_pose=gt_pose_q, pred_pose=best_pred_pose_q,
                    metric=object_metrics[obj_f],
                    obj_f=f"{obj_f}", frame_idx=i,
                    save_path=save_results_est_path, glctx=glctx,
                    name=f"{len(reader.color_files)}_{name}",
                    nocs_metric=True, est_mesh=est.mesh,
                )
            except Exception:
                pass

            obj_count += 1

        # ==================================================================
        # Per-object summary
        # ==================================================================
        df_obj = pd.DataFrame({
            "Frame_ID": object_metrics[obj_f]["instance_id"],
            "Class": object_metrics[obj_f]["cls_id"],
            "ADD-S": object_metrics[obj_f]["ADD-S"],
            "ADD": object_metrics[obj_f]["ADD"],
            "AR": object_metrics[obj_f]["AR"],
            "MSSD": object_metrics[obj_f]["MSSD"],
            "MSPD": object_metrics[obj_f]["MSPD"],
            "VSD": object_metrics[obj_f]["VSD"],
            "R_error": object_metrics[obj_f]["R error"],
            "T_error": object_metrics[obj_f]["T error"],
            "Chosen_Candidate": object_metrics[obj_f]["chosen_candidate"],
            "Default_Candidate": object_metrics[obj_f]["default_candidate"],
            "Obs_Override": object_metrics[obj_f]["obs_override"],
            "Selection_Reason": object_metrics[obj_f]["selection_reason"],
            "Sa_obs_total": object_metrics[obj_f]["obs_total_score"],
            "Sa_best_non_obs_total": object_metrics[obj_f]["best_non_obs_total_score"],
            "L_depth_obs": object_metrics[obj_f]["obs_l_depth"],
            "L_mask_obs": object_metrics[obj_f]["obs_l_mask"],
            "L_geom_obs": object_metrics[obj_f]["obs_l_geom"],
            "L_depth_best_non_obs": object_metrics[obj_f]["best_non_obs_l_depth"],
            "L_mask_best_non_obs": object_metrics[obj_f]["best_non_obs_l_mask"],
            "L_geom_best_non_obs": object_metrics[obj_f]["best_non_obs_l_geom"],
            "Sa_best_non_obs": object_metrics[obj_f]["best_non_obs_anchor_score"],
        })

        means_all = {
            "ADD-S": np.mean(object_metrics[obj_f]["ADD-S"]) * 100,
            "ADD": np.mean(object_metrics[obj_f]["ADD"]) * 100,
            "AR": np.mean(object_metrics[obj_f]["AR"]) * 100,
            "MSSD": np.mean(object_metrics[obj_f]["MSSD"]) * 100,
            "MSPD": np.mean(object_metrics[obj_f]["MSPD"]) * 100,
            "VSD": np.mean(object_metrics[obj_f]["VSD"]) * 100,
            "R_error": np.mean(object_metrics[obj_f]["R error"]),
            "T_error": np.mean(object_metrics[obj_f]["T error"]),
        }

        from collections import Counter
        cand_counts = Counter(object_metrics[obj_f]["chosen_candidate"])
        cand_summary = ", ".join(f"{k}:{v}" for k, v in cand_counts.most_common())

        mean_row_df = pd.DataFrame({
            "Frame_ID": ["MEAN"],
            "Class": [obj_f],
            "ADD-S": [f"{means_all['ADD-S']:.1f}"],
            "ADD": [f"{means_all['ADD']:.1f}"],
            "AR": [f"{means_all['AR']:.1f}"],
            "MSSD": [f"{means_all['MSSD']:.1f}"],
            "MSPD": [f"{means_all['MSPD']:.1f}"],
            "VSD": [f"{means_all['VSD']:.1f}"],
            "R_error": [f"{means_all['R_error']:.1f}"],
            "T_error": [f"{means_all['T_error']:.1f}"],
            "Chosen_Candidate": [cand_summary],
        })

        df_obj = pd.concat([df_obj, mean_row_df], ignore_index=True)

        row_data = {
            "Class_ID": obj_f,
            "ADD-S": f"{means_all['ADD-S']:.1f}",
            "ADD": f"{means_all['ADD']:.1f}",
            "AR": f"{means_all['AR']:.1f}",
            "MSSD": f"{means_all['MSSD']:.1f}",
            "MSPD": f"{means_all['MSPD']:.1f}",
            "VSD": f"{means_all['VSD']:.1f}",
            "Candidate_Distribution": cand_summary,
        }
        data.append(row_data)

        df_obj.to_excel(f"{save_results_est_path}/{obj_f}_metrics_results.xlsx", index=False)

        all_frame_data["Frame_ID"].extend(object_metrics[obj_f]["instance_id"])
        all_frame_data["Class"].extend(object_metrics[obj_f]["cls_id"])
        all_frame_data["ADD-S"].extend(object_metrics[obj_f]["ADD-S"])
        all_frame_data["ADD"].extend(object_metrics[obj_f]["ADD"])
        all_frame_data["AR"].extend(object_metrics[obj_f]["AR"])
        all_frame_data["MSSD"].extend(object_metrics[obj_f]["MSSD"])
        all_frame_data["MSPD"].extend(object_metrics[obj_f]["MSPD"])
        all_frame_data["VSD"].extend(object_metrics[obj_f]["VSD"])
        all_frame_data["R_error"].extend(object_metrics[obj_f]["R error"])
        all_frame_data["T_error"].extend(object_metrics[obj_f]["T error"])
        all_frame_data["Chosen_Candidate"].extend(object_metrics[obj_f]["chosen_candidate"])
        all_frame_data["Default_Candidate"].extend(object_metrics[obj_f]["default_candidate"])
        all_frame_data["Obs_Override"].extend(object_metrics[obj_f]["obs_override"])
        all_frame_data["Selection_Reason"].extend(object_metrics[obj_f]["selection_reason"])
        all_frame_data["Sa_obs_total"].extend(object_metrics[obj_f]["obs_total_score"])
        all_frame_data["Sa_best_non_obs_total"].extend(object_metrics[obj_f]["best_non_obs_total_score"])
        all_frame_data["L_depth_obs"].extend(object_metrics[obj_f]["obs_l_depth"])
        all_frame_data["L_mask_obs"].extend(object_metrics[obj_f]["obs_l_mask"])
        all_frame_data["L_geom_obs"].extend(object_metrics[obj_f]["obs_l_geom"])
        all_frame_data["L_depth_best_non_obs"].extend(object_metrics[obj_f]["best_non_obs_l_depth"])
        all_frame_data["L_mask_best_non_obs"].extend(object_metrics[obj_f]["best_non_obs_l_mask"])
        all_frame_data["L_geom_best_non_obs"].extend(object_metrics[obj_f]["best_non_obs_l_geom"])
        all_frame_data["Sa_best_non_obs"].extend(object_metrics[obj_f]["best_non_obs_anchor_score"])

    # ======================================================================
    # Overall summary
    # ======================================================================
    overall_means = {
        "ADD-S": np.mean([np.mean(object_metrics[obj]["ADD-S"]) for obj in obj_folder]) * 100,
        "ADD": np.mean([np.mean(object_metrics[obj]["ADD"]) for obj in obj_folder]) * 100,
        "AR": np.mean([np.mean(object_metrics[obj]["AR"]) for obj in obj_folder]) * 100,
        "MSSD": np.mean([np.mean(object_metrics[obj]["MSSD"]) for obj in obj_folder]) * 100,
        "MSPD": np.mean([np.mean(object_metrics[obj]["MSPD"]) for obj in obj_folder]) * 100,
        "VSD": np.mean([np.mean(object_metrics[obj]["VSD"]) for obj in obj_folder]) * 100,
    }

    mean_row = {
        "Class_ID": "MEAN",
        "ADD-S": f"{overall_means['ADD-S']:.1f}",
        "ADD": f"{overall_means['ADD']:.1f}",
        "AR": f"{overall_means['AR']:.1f}",
        "MSSD": f"{overall_means['MSSD']:.1f}",
        "MSPD": f"{overall_means['MSPD']:.1f}",
        "VSD": f"{overall_means['VSD']:.1f}",
        "Candidate_Distribution": "",
    }
    data.append(mean_row)

    latex_str = (
        f"MEAN & {overall_means['AR']:.1f} & {overall_means['VSD']:.1f} "
        f"& {overall_means['MSSD']:.1f} & {overall_means['MSPD']:.1f} "
        f"& {overall_means['ADD-S']:.1f} & - \\\\"
    )
    print("\n" + latex_str)

    df = pd.DataFrame(data)
    df.to_excel(f"{save_results_est_path}/0_mean_all_metrics_classes_results.xlsx", index=False)

    df_all_frames = pd.DataFrame(all_frame_data)

    means_all_final = {
        "Frame_ID": "MEAN",
        "Class": "ALL",
        "ADD-S": f"{df_all_frames['ADD-S'].mean() * 100:.1f}",
        "ADD": f"{df_all_frames['ADD'].mean() * 100:.1f}",
        "AR": f"{df_all_frames['AR'].mean() * 100:.1f}",
        "MSSD": f"{df_all_frames['MSSD'].mean() * 100:.1f}",
        "MSPD": f"{df_all_frames['MSPD'].mean() * 100:.1f}",
        "VSD": f"{df_all_frames['VSD'].mean() * 100:.1f}",
        "R_error": f"{df_all_frames['R_error'].mean():.1f}",
        "T_error": f"{df_all_frames['T_error'].mean():.1f}",
        "Chosen_Candidate": "",
        "Default_Candidate": "",
        "Obs_Override": "",
        "Selection_Reason": "",
        "Sa_obs_total": "",
        "Sa_best_non_obs_total": "",
        "L_depth_obs": "",
        "L_mask_obs": "",
        "L_geom_obs": "",
        "L_depth_best_non_obs": "",
        "L_mask_best_non_obs": "",
        "L_geom_best_non_obs": "",
        "Sa_best_non_obs": "",
    }

    df_all_frames = pd.concat([df_all_frames, pd.DataFrame([means_all_final])], ignore_index=True)

    output_path = f"{save_results_est_path}/0_all_frames_metrics_results.xlsx"
    df_all_frames.to_excel(output_path, index=False)
    print(f"\nAll frames metrics saved to {output_path}")
    print("\nSaved data preview:")
    print(df_all_frames)
