"""HO3D query evaluation with paper-aligned weighted candidate selection.

For every query frame, each valid geometry candidate is independently
registered and scored by

    S(G_i) = alpha * S_a_norm(G_i) + beta * S_q(G_i),
    S_q(G_i) = -(w_d*L_depth_norm + w_m*L_mask_norm + w_c*L_geom_norm).

The final candidate is the direct argmax of this scalar score.  No candidate
is privileged as a default and no undisclosed hard replacement gate is used.
The disclosed fixed-scale normalization is recorded in the run manifest and
in the per-candidate score table so the implementation is reproducible.
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
    compute_mesh_diameter,
    make_mesh_tensors,
    nvdiffrast_render,
    visualize_frame_results_gt,
)
from metrics import compute_add, compute_adds, compute_RT_distances
from candidate_selection import (
    PAPER_CANDIDATE_LABELS,
    apply_candidate_appearance,
    derive_anchor_reliability_scores,
    parse_candidate_labels,
    score_candidates,
)


def _load_anchor_depth_meters(path: str, depth_unit: str) -> np.ndarray:
    """Load the copied anchor depth using the construction registry contract."""

    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(path)
    depth = depth.astype(np.float32)
    unit = str(depth_unit or "auto")
    if unit == "auto":
        max_value = float(np.nanmax(depth))
        unit = "rayst3r_uint16" if max_value > 20000 else (
            "mm" if max_value > 200 else "meters"
        )
    if unit == "mm":
        return depth / 1000.0
    if unit == "rayst3r_uint16":
        return depth / 65535.0 * 10.0
    if unit == "meters":
        return depth
    raise ValueError(f"Unsupported anchor depth unit in registry: {unit}")


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


def _select_weighted_candidate(
    candidate_results: list,
) -> tuple:
    """Return the direct argmax of the paper's weighted score (Eq. 22)."""
    if not candidate_results:
        raise ValueError("candidate_results must not be empty")

    ranked = sorted(
        candidate_results,
        key=lambda item: float(item["selection_score"]),
        reverse=True,
    )
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    diagnostics = {
        "selection_reason": "weighted_argmax",
        "winning_score": float(winner["selection_score"]),
        "runner_up_candidate": runner_up["label"] if runner_up else "",
        "runner_up_score": float(runner_up["selection_score"]) if runner_up else np.nan,
        "score_margin": (
            float(winner["selection_score"] - runner_up["selection_score"])
            if runner_up else np.nan
        ),
    }
    return winner, diagnostics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query-phase multi-candidate evaluation")
    parser.add_argument("--name", type=str, default="any6d_multicand",
                        help="Experiment name")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed recorded with the evaluation run")
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
    parser.add_argument(
        "--w_pose",
        type=float,
        default=0.0,
        help=(
            "Weight for normalized FoundationPose confidence. The revised main "
            "protocol uses 0; non-zero values are pose-confidence ablations."
        ),
    )
    # --- Total score balance (Eq. 2) ---
    parser.add_argument("--score_alpha", type=float, default=0.3,
                        help="alpha: weight for s_anchor in total score (Eq. 2)")
    parser.add_argument("--score_beta", type=float, default=0.7,
                        help="beta: weight for s_query in total score (Eq. 2)")
    parser.add_argument(
        "--query_loss_normalization",
        choices=["fixed_scale", "minmax"],
        default="fixed_scale",
        help="Normalize query losses using disclosed fixed scales or candidate-set min-max.",
    )
    parser.add_argument("--query_depth_scale", type=float, default=0.10)
    parser.add_argument("--query_mask_scale", type=float, default=1.00)
    parser.add_argument("--query_geom_scale", type=float, default=0.10)
    parser.add_argument(
        "--anchor_score_mode",
        choices=["observation_calibrated", "registry_raw"],
        default="observation_calibrated",
        help="Use anchor RGB-D consistency plus bounded ICP calibration, or raw registry Sa.",
    )
    parser.add_argument("--anchor_depth_scale", type=float, default=0.05)
    parser.add_argument("--anchor_mask_scale", type=float, default=0.50)
    parser.add_argument("--anchor_geom_scale", type=float, default=0.05)
    parser.add_argument(
        "--anchor_calibration_weight",
        type=float,
        default=0.15,
        help="Bounded contribution of the registry ICP calibration term to effective Sa.",
    )
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
    parser.add_argument(
        "--candidate_labels",
        type=str,
        default=",".join(PAPER_CANDIDATE_LABELS),
        help="Comma-separated registry labels used in the formal candidate pool.",
    )
    parser.add_argument(
        "--candidate_color_mode",
        choices=[
            "native",
            "prior_transfer",
            "anchor_projected",
            "anchor_prior_hybrid",
            "uniform_gray",
        ],
        default="anchor_prior_hybrid",
        help=(
            "Common disclosed appearance treatment. anchor_prior_hybrid first "
            "transfers the InstantMesh prior appearance and then overwrites only "
            "RGB-D-visible vertices with anchor RGB."
        ),
    )
    parser.add_argument("--include_fallback_candidate", action=argparse.BooleanOptionalAction,
                        default=False,
                        help=(
                            "Include the legacy single-candidate mesh as an explicitly labeled "
                            "ablation candidate (disabled in the paper-aligned candidate pool)"
                        ))
    parser.add_argument(
        "--selection_policy",
        choices=["weighted"],
        default="weighted",
        help=(
            "Directly maximize S(G)=alpha*S_a_norm+beta*S_q (Eq.2/Eq.22); "
            "no default candidate, score margin, or component-wise hard gate"
        ),
    )
    parser.add_argument(
        "--obs_anchor_path",
        type=str,
        default=None,
        help=(
            "Optional external observation-anchor candidate for a separately labeled ablation; "
            "omit it for the paper-aligned RaySt3R/InstantMesh/fused candidate pool"
        ),
    )
    parser.add_argument(
        "--obs_label",
        type=str,
        default="obs",
        help="Label assigned to the optional external observation-anchor candidate.",
    )
    parser.add_argument(
        "--symmetry_consistency_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablation only: use dataset-provided discrete symmetries and the previous "
            "query prediction to reduce frame-to-frame jumps. Disabled for the "
            "paper main protocol because query pairs are evaluated independently."
        ),
    )
    parser.add_argument(
        "--anchor_symmetry_align",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablation only: align the query prediction to a dataset-provided anchor "
            "symmetry branch. Disabled for the paper main protocol; GT symmetry "
            "metadata is otherwise used only by symmetry-aware evaluation metrics."
        ),
    )

    args = parser.parse_args()
    seed_everything(args.seed)
    selected_candidate_labels = parse_candidate_labels(args.candidate_labels)
    score_weights = {
        "w_depth": args.w_depth,
        "w_mask": args.w_mask,
        "w_geom": args.w_geom,
        "w_pose": args.w_pose,
        "score_alpha": args.score_alpha,
        "score_beta": args.score_beta,
    }
    if any((not np.isfinite(value)) or value < 0.0 for value in score_weights.values()):
        raise ValueError(f"All scoring weights must be finite and non-negative: {score_weights}")
    if args.score_alpha + args.score_beta <= 0.0:
        raise ValueError("At least one of --score_alpha and --score_beta must be positive")
    if args.w_depth + args.w_mask + args.w_geom + args.w_pose <= 0.0:
        raise ValueError("At least one query-consistency weight must be positive")
    positive_scales = {
        "query_depth_scale": args.query_depth_scale,
        "query_mask_scale": args.query_mask_scale,
        "query_geom_scale": args.query_geom_scale,
        "anchor_depth_scale": args.anchor_depth_scale,
        "anchor_mask_scale": args.anchor_mask_scale,
        "anchor_geom_scale": args.anchor_geom_scale,
    }
    if any((not np.isfinite(value)) or value <= 0.0 for value in positive_scales.values()):
        raise ValueError(f"All loss scales must be finite and positive: {positive_scales}")
    if not 0.0 <= args.anchor_calibration_weight <= 1.0:
        raise ValueError("--anchor_calibration_weight must lie in [0, 1]")

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

    run_manifest = {
        "dataset": "HO3D",
        "mask_source": "precomputed_XMem_query_masks",
        "selection_policy": args.selection_policy,
        "selection_formula": (
            "argmax_i alpha*S_anchor_norm(i) + beta*S_query(i), where "
            "S_query=-(w_depth*L_depth_norm+w_mask*L_mask_norm+w_geom*L_geom_norm)"
            "+w_pose*S_pose_norm"
        ),
        "pose_confidence_role": (
            "diagnostic_only" if float(args.w_pose) == 0.0 else "selection_ablation"
        ),
        "estimation_postprocessing": {
            "gt_symmetry_adjustment": bool(args.anchor_symmetry_align),
            "cross_frame_symmetry_smoothing": bool(args.symmetry_consistency_enable),
            "paper_main_protocol": bool(
                not args.anchor_symmetry_align
                and not args.symmetry_consistency_enable
            ),
            "gt_symmetry_evaluation_only": bool(
                not args.anchor_symmetry_align
                and not args.symmetry_consistency_enable
            ),
        },
        "normalization": {
            "anchor": "S_anchor / max_j(S_anchor_j) per query frame",
            "query_losses": args.query_loss_normalization,
            "query_fixed_scales": {
                "depth": args.query_depth_scale,
                "mask": args.query_mask_scale,
                "geom": args.query_geom_scale,
            },
            "anchor_score_mode": args.anchor_score_mode,
            "anchor_fixed_scales": {
                "depth": args.anchor_depth_scale,
                "mask": args.anchor_mask_scale,
                "geom": args.anchor_geom_scale,
            },
            "anchor_calibration_weight": args.anchor_calibration_weight,
        },
        "candidate_pool": {
            "registry_candidates": True,
            "registry_labels": sorted(selected_candidate_labels or []),
            "candidate_color_mode": args.candidate_color_mode,
            "external_obs_candidate": bool(args.obs_anchor_path),
            "legacy_fallback_candidate": bool(args.include_fallback_candidate),
        },
        "evaluation_protocol": {
            "reference": "taeyeopl/Any6D@80eb4866a1c96ecb18be18836aba4f4bd6e80e9e",
            "add": "mean corresponding-point distance <= 0.1*object_diameter",
            "adds": "mean nearest-neighbor distance <= 0.1*object_diameter",
            "mssd_thresholds_diameter": [
                0.05, 0.10, 0.15, 0.20, 0.25,
                0.30, 0.35, 0.40, 0.45, 0.50,
            ],
            "mspd_thresholds_px": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "vsd_delta_mm": 15.0,
            "vsd_taus": [
                0.05, 0.10, 0.15, 0.20, 0.25,
                0.30, 0.35, 0.40, 0.45, 0.50,
            ],
            "ar": "mean(MSSD_recall, MSPD_recall, VSD_recall)",
            "primary_aggregation": "macro mean over the 13 HO3D sequences",
            "secondary_aggregation": "frame-weighted mean, saved separately",
            "add_adds_precision": "float64 stability fix; metric definition unchanged",
        },
        "args": vars(args),
    }
    with open(
        os.path.join(save_results_est_path, "query_run_manifest.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_manifest, f, indent=2)

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
            "selection_reason": [], "winning_score": [],
            "runner_up_candidate": [], "runner_up_score": [], "score_margin": [],
        }
        for obj in obj_folder
    }
    candidate_score_rows = []
    all_frame_data = {
        "Frame_ID": [], "Class": [],
        "ADD-S": [], "ADD": [], "AR": [],
        "MSSD": [], "MSPD": [], "VSD": [],
        "R_error": [], "T_error": [], "Chosen_Candidate": [],
        "Selection_Reason": [], "Winning_Score": [],
        "Runner_Up_Candidate": [], "Runner_Up_Score": [], "Score_Margin": [],
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
        gt_diameter = compute_mesh_diameter(
            model_pts=np.asarray(gt_mesh.vertices), n_sample=10000
        )

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
        registry_meshes = {}
        if obj_candidates:
            for label, info in obj_candidates.items():
                if selected_candidate_labels is not None and label not in selected_candidate_labels:
                    continue
                mesh_p = info["mesh_path"]
                mesh_dir = os.path.dirname(mesh_p)
                final_mesh_p = os.path.join(
                    mesh_dir, f"final_mesh_{obj_name}_{label}.obj"
                )
                if os.path.exists(final_mesh_p):
                    mesh_p = final_mesh_p
                elif not os.path.exists(mesh_p):
                    continue
                registry_meshes[label] = trimesh.load(mesh_p)
                obj_candidate_info_by_label[label] = dict(info)

                pose_path = info.get("pose_path", "")
                if os.path.exists(pose_path):
                    cand_anchor_poses[label] = np.loadtxt(pose_path)
                    gt_pose_path = pose_path.replace("initial", "gt")
                    if os.path.exists(gt_pose_path):
                        cand_gt_anchor_poses[label] = np.loadtxt(gt_pose_path)

        anchor_score_rows = []
        registry_has_bounded_anchor_scores = bool(registry_meshes)
        for label in registry_meshes:
            info = obj_candidate_info_by_label[label]
            is_bounded = (
                info.get("anchor_score_version") == "bounded_observation_calibrated_v1"
                and "s_anchor_raw" in info
                and "anchor_observation_quality" in info
                and "anchor_calibration_norm" in info
            )
            registry_has_bounded_anchor_scores &= is_bounded
            anchor_score_rows.append(
                {
                    "label": label,
                    "s_anchor": float(info.get("s_anchor", 1.0)),
                    "s_anchor_raw": float(info.get("s_anchor_raw", info.get("s_anchor", 1.0))),
                    "anchor_observation_quality": float(
                        info.get("anchor_observation_quality", info.get("s_anchor", 1.0))
                    ),
                    "anchor_calibration_norm": float(
                        info.get("anchor_calibration_norm", 1.0)
                    ),
                    "l_depth": float(info.get("l_depth", args.anchor_depth_scale)),
                    "l_mask": float(info.get("l_mask", args.anchor_mask_scale)),
                    "l_geom": float(info.get("l_geom", args.anchor_geom_scale)),
                }
            )
        if (
            args.anchor_score_mode == "observation_calibrated"
            and not registry_has_bounded_anchor_scores
        ):
            derive_anchor_reliability_scores(
                anchor_score_rows,
                w_depth=args.w_depth,
                w_mask=args.w_mask,
                w_geom=args.w_geom,
                depth_scale=args.anchor_depth_scale,
                mask_scale=args.anchor_mask_scale,
                geom_scale=args.anchor_geom_scale,
                calibration_weight=args.anchor_calibration_weight,
            )
        anchor_score_by_label = {row["label"]: row for row in anchor_score_rows}

        if registry_meshes:
            anchor_rgb = None
            anchor_mask = None
            anchor_depth = None
            if args.candidate_color_mode in {"anchor_projected", "anchor_prior_hybrid"}:
                obj_anchor_dir = os.path.join(anchor_path, obj_name)
                anchor_bgr = cv2.imread(os.path.join(obj_anchor_dir, "color.png"))
                anchor_mask = cv2.imread(
                    os.path.join(obj_anchor_dir, "mask.png"), cv2.IMREAD_GRAYSCALE
                )
                if anchor_bgr is None or anchor_mask is None:
                    raise FileNotFoundError(
                        f"{args.candidate_color_mode} requires color.png and mask.png "
                        f"in {obj_anchor_dir}"
                    )
                anchor_rgb = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB)
                depth_path = os.path.join(obj_anchor_dir, "depth.png")
                anchor_depth = _load_anchor_depth_meters(
                    depth_path,
                    obj_registry.get("scale_metadata", {}).get("depth_unit", "auto"),
                )
            registry_meshes, appearance_summary = apply_candidate_appearance(
                registry_meshes,
                cand_anchor_poses,
                args.candidate_color_mode,
                anchor_rgb=anchor_rgb,
                anchor_mask=anchor_mask,
                anchor_k=K_anchor if anchor_rgb is not None else None,
                anchor_depth=anchor_depth,
            )
            print(
                f"{obj_f}: appearance={appearance_summary.mode}, "
                f"source={appearance_summary.source_label}, "
                f"transferred={list(appearance_summary.transferred_labels)}, "
                f"anchor_projected={dict(appearance_summary.projected_vertex_counts)}"
            )
            for label, cand_mesh in registry_meshes.items():
                score_row = anchor_score_by_label[label]
                info = obj_candidate_info_by_label[label]
                info.update({key: value for key, value in score_row.items() if key != "label"})
                cand_caches.append(
                    CandidateCache(
                        label=label,
                        mesh=cand_mesh,
                        s_anchor=float(score_row["s_anchor"]),
                        use_bbox_diameter=args.use_bbox_diameter,
                        lazy_tensors=args.lazy_tensors,
                    )
                )

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
                use_bbox_diameter=args.use_bbox_diameter,
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
        prev_pred_pose_by_label = {}
        if (
            len(cand_caches) > 1
            and not args.per_frame_selection
            and args.selection_policy == "weighted"
        ):
            # Object-level selection: use anchor-phase candidate metrics as static priors.
            static_rows = []
            for c in cand_caches:
                info = obj_candidate_info_by_label.get(c.label, {})
                static_rows.append({
                    "label": c.label,
                    "s_anchor": float(c.s_anchor),
                    "l_depth": float(info.get("l_depth", 1.0)),
                    "l_mask": float(info.get("l_mask", 1.0)),
                    "l_geom": float(info.get("l_geom", 1.0)),
                })
            score_candidates(
                static_rows,
                w_depth=args.w_depth,
                w_mask=args.w_mask,
                w_geom=args.w_geom,
                score_alpha=args.score_alpha,
                score_beta=args.score_beta,
                loss_normalization=args.query_loss_normalization,
                depth_scale=args.query_depth_scale,
                mask_scale=args.query_mask_scale,
                geom_scale=args.query_geom_scale,
            )
            static_object_winner = max(
                static_rows, key=lambda row: row["total_score"]
            )["label"]
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
            best_label = None
            best_pred_pose_q = None
            best_pred_q = None

            candidate_results = []
            eval_caches = cand_caches
            if (
                len(cand_caches) > 1
                and not args.per_frame_selection
                and static_object_winner is not None
                and args.selection_policy == "weighted"
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
                pose_scores = getattr(est, "scores", None)
                if pose_scores is None or len(pose_scores) == 0:
                    pose_score = 0.0
                else:
                    pose_score_value = pose_scores[0]
                    if torch.is_tensor(pose_score_value):
                        pose_score_value = pose_score_value.detach().cpu().item()
                    pose_score = float(pose_score_value)
                if args.anchor_symmetry_align and len(sym_tfs_list) > 1:
                    pred_pose_q_k = _resolve_pose_to_anchor_symmetry(
                        pred_pose_q_k, pred_pose_a, sym_tfs_list
                    )
                if args.symmetry_consistency_enable and len(sym_tfs_list) > 1:
                    pred_pose_q_k = _resolve_pose_with_symmetry(
                        pred_pose_q_k,
                        sym_tfs_list,
                        prev_pred_pose_by_label.get(cache.label),
                    )
                prev_pred_pose_by_label[cache.label] = pred_pose_q_k.copy()

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

                anchor_info = obj_candidate_info_by_label.get(cache.label, {})

                candidate_results.append({
                    "label": cache.label,
                    "pred_pose_q": pred_pose_q_k,
                    "pred_q": pred_q_k,
                    "selection_score": 0.0,
                    "s_query": s_query,
                    "pose_score": pose_score,
                    "s_anchor": cache.s_anchor,
                    "s_anchor_raw": float(
                        anchor_info.get("s_anchor_raw", anchor_info.get("s_anchor", cache.s_anchor))
                    ),
                    "anchor_observation_quality": float(
                        anchor_info.get("anchor_observation_quality", cache.s_anchor)
                    ),
                    "anchor_calibration_norm": float(
                        anchor_info.get("anchor_calibration_norm", 1.0)
                    ),
                    "s_anchor_norm": 1.0,
                    "l_depth": float(l_depth),
                    "l_mask": float(l_mask),
                    "l_geom": float(l_geom),
                    "l_depth_norm": 1.0,
                    "l_mask_norm": 1.0,
                    "l_geom_norm": 1.0,
                })

            if len(eval_caches) > 1 and args.per_frame_selection:
                score_candidates(
                    candidate_results,
                    w_depth=args.w_depth,
                    w_mask=args.w_mask,
                    w_geom=args.w_geom,
                    score_alpha=args.score_alpha,
                    score_beta=args.score_beta,
                    w_pose=args.w_pose,
                    loss_normalization=args.query_loss_normalization,
                    depth_scale=args.query_depth_scale,
                    mask_scale=args.query_mask_scale,
                    geom_scale=args.query_geom_scale,
                )
                for r in candidate_results:
                    r["selection_score"] = r["total_score"]
            else:
                # With a single candidate the score does not affect selection, but
                # recording the same formula keeps result files self-consistent.
                r = candidate_results[0]
                score_candidates(
                    candidate_results,
                    w_depth=args.w_depth,
                    w_mask=args.w_mask,
                    w_geom=args.w_geom,
                    score_alpha=args.score_alpha,
                    score_beta=args.score_beta,
                    w_pose=args.w_pose,
                    loss_normalization=args.query_loss_normalization,
                    depth_scale=args.query_depth_scale,
                    mask_scale=args.query_mask_scale,
                    geom_scale=args.query_geom_scale,
                )
                r["selection_score"] = r["total_score"]

            # ==============================================================
            # Direct weighted candidate selection (Eq. 22)
            # ==============================================================
            candidate_results.sort(
                key=lambda item: float(item["selection_score"]),
                reverse=True,
            )
            winner, selection_diagnostics = _select_weighted_candidate(
                candidate_results,
            )
            best_label = winner["label"]
            best_pred_pose_q = winner["pred_pose_q"]
            pred_q = winner["pred_q"]

            for rank, result in enumerate(candidate_results, start=1):
                candidate_score_rows.append({
                    "Frame_ID": obj_count,
                    "Query_Frame_Index": i,
                    "Class": obj_f,
                    "Candidate": result["label"],
                    "Rank": rank,
                    "Selected": result["label"] == best_label,
                    "S_anchor": float(result["s_anchor"]),
                    "S_anchor_raw": float(result["s_anchor_raw"]),
                    "Anchor_Observation_Quality": float(result["anchor_observation_quality"]),
                    "Anchor_Calibration_Norm": float(result["anchor_calibration_norm"]),
                    "S_anchor_norm": float(result["s_anchor_norm"]),
                    "L_depth": float(result["l_depth"]),
                    "L_mask": float(result["l_mask"]),
                    "L_geom": float(result["l_geom"]),
                    "L_depth_norm": float(result["l_depth_norm"]),
                    "L_mask_norm": float(result["l_mask_norm"]),
                    "L_geom_norm": float(result["l_geom_norm"]),
                    "S_query": float(result["s_query"]),
                    "Pose_Score_Raw": float(result["pose_score"]),
                    "Pose_Score_Norm": float(result["pose_score_norm"]),
                    "Selection_Score": float(result["selection_score"]),
                })

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
            object_metrics[obj_f]["selection_reason"].append(selection_diagnostics["selection_reason"])
            object_metrics[obj_f]["winning_score"].append(selection_diagnostics["winning_score"])
            object_metrics[obj_f]["runner_up_candidate"].append(selection_diagnostics["runner_up_candidate"])
            object_metrics[obj_f]["runner_up_score"].append(selection_diagnostics["runner_up_score"])
            object_metrics[obj_f]["score_margin"].append(selection_diagnostics["score_margin"])

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
            "Selection_Reason": object_metrics[obj_f]["selection_reason"],
            "Winning_Score": object_metrics[obj_f]["winning_score"],
            "Runner_Up_Candidate": object_metrics[obj_f]["runner_up_candidate"],
            "Runner_Up_Score": object_metrics[obj_f]["runner_up_score"],
            "Score_Margin": object_metrics[obj_f]["score_margin"],
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
        all_frame_data["Selection_Reason"].extend(object_metrics[obj_f]["selection_reason"])
        all_frame_data["Winning_Score"].extend(object_metrics[obj_f]["winning_score"])
        all_frame_data["Runner_Up_Candidate"].extend(object_metrics[obj_f]["runner_up_candidate"])
        all_frame_data["Runner_Up_Score"].extend(object_metrics[obj_f]["runner_up_score"])
        all_frame_data["Score_Margin"].extend(object_metrics[obj_f]["score_margin"])

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
        "Selection_Reason": "",
        "Winning_Score": "",
        "Runner_Up_Candidate": "",
        "Runner_Up_Score": "",
        "Score_Margin": "",
    }

    df_all_frames = pd.concat([df_all_frames, pd.DataFrame([means_all_final])], ignore_index=True)

    output_path = f"{save_results_est_path}/0_all_frames_metrics_results.xlsx"
    df_all_frames.to_excel(output_path, index=False)
    candidate_scores_path = f"{save_results_est_path}/0_all_candidate_scores.xlsx"
    pd.DataFrame(candidate_score_rows).to_excel(candidate_scores_path, index=False)
    print(f"\nAll frames metrics saved to {output_path}")
    print(f"Per-candidate weighted scores saved to {candidate_scores_path}")
    print("\nSaved data preview:")
    print(df_all_frames)
