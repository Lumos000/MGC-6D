"""REAL275 query evaluation using the fixed baseline-fair paper protocol."""

from project_paths import setup_project_paths
setup_project_paths()

import copy
import hashlib
import os
import argparse
import pickle
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import trimesh
import torch
import nvdiffrast.torch as dr
from tqdm import tqdm

from estimater import Any6D, PoseRefinePredictor, ScorePredictor
from metrics import compute_RT_distances, compute_add, compute_adds
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd
from foundationpose.Utils import (
    calc_pts_diameter,
    compute_mesh_diameter,
    make_mesh_tensors,
    nvdiffrast_render,
    visualize_frame_results_gt,
)
from pytorch_lightning import seed_everything
from datetime import datetime
import json
from candidate_selection import (
    PAPER_CANDIDATE_LABELS,
    apply_candidate_appearance,
    candidate_mesh_diameter,
    compute_anchor_reliability_components,
    paper_registry_candidates,
    score_candidates,
    select_scored_candidate,
)
from revision_protocol import (
    add_query_protocol_arguments,
    runtime_provenance,
    validate_query_protocol,
)


class CandidateCache:
    """Complete, candidate-specific geometry state used during query registration.

    REAL275 previously called ``Any6D.reset_object`` repeatedly.  A stale-cache
    path could leave ``est.mesh`` pointing at the last candidate while restoring
    metadata for another candidate.  Keeping the same explicit state bundle as
    the HO3D query makes every mesh/pose/renderer tuple auditable.
    """

    __slots__ = (
        "label", "mesh_ori", "mesh", "model_center", "diameter", "vox_size",
        "dist_bin", "angle_bin", "mesh_tensors", "mesh_o3d", "fingerprint",
    )

    def __init__(
        self,
        label: str,
        mesh: trimesh.Trimesh,
        diameter_mode: str = "bbox_diagonal",
    ):
        import open3d as o3d

        self.label = str(label)
        self.mesh_ori = mesh.copy()
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        min_xyz = vertices.min(axis=0)
        max_xyz = vertices.max(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2.0
        centered = mesh.copy()
        centered.vertices = vertices - self.model_center.reshape(1, 3)
        self.mesh = centered
        self.diameter = candidate_mesh_diameter(
            centered.vertices,
            diameter_mode,
            sampled_diameter_fn=compute_mesh_diameter,
        )
        self.vox_size = max(self.diameter / 20.0, 0.003)
        self.dist_bin = self.vox_size / 2.0
        self.angle_bin = 20
        self.mesh_tensors = make_mesh_tensors(centered)

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(centered.vertices))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(centered.faces))
        visual = getattr(centered, "visual", None)
        if visual is not None and getattr(visual, "kind", None) == "texture":
            try:
                visual = visual.to_color()
            except Exception:
                visual = None
        if visual is not None and getattr(visual, "kind", None) == "vertex":
            colors = np.asarray(visual.vertex_colors)
            if colors.ndim == 2 and len(colors) == len(centered.vertices):
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
                    colors[:, :3].astype(np.float64) / 255.0
                )
        mesh_o3d.compute_vertex_normals()
        self.mesh_o3d = mesh_o3d

        digest = hashlib.sha256()
        digest.update(np.asarray(centered.vertices, dtype=np.float32).tobytes())
        digest.update(np.asarray(centered.faces, dtype=np.int32).tobytes())
        self.fingerprint = digest.hexdigest()[:16]


def swap_candidate(est: Any6D, cache: CandidateCache) -> None:
    """Atomically restore all geometry state for one candidate."""

    est.mesh_ori = cache.mesh_ori
    est.model_center = cache.model_center
    est.mesh = cache.mesh
    est.mesh_o3d = cache.mesh_o3d
    est.diameter = cache.diameter
    est.vox_size = cache.vox_size
    est.dist_bin = cache.dist_bin
    est.angle_bin = cache.angle_bin
    est.mesh_tensors = cache.mesh_tensors
    if est.mesh is not cache.mesh or est.mesh_tensors is not cache.mesh_tensors:
        raise RuntimeError(f"Candidate state swap failed for {cache.label}")


# ---------------------------------------------------------------------------
# Symmetry helpers for pose consistency (ported from query_method_ablation.py)
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

def _load_real275_evaluation_resources(
    *, obj_name, model_dir, anchor_gt_path, models_info_path, renderer,
):
    """Load the fixed baseline-fair evaluation resources after selection."""

    anchor_gt_pose = np.loadtxt(anchor_gt_path)
    gt_model_path = os.path.join(model_dir, f"{obj_name}.obj")
    if not os.path.exists(gt_model_path):
        raise FileNotFoundError(gt_model_path)
    gt_mesh_nocs = trimesh.load(gt_model_path)
    gt_vertices_eval = np.asarray(gt_mesh_nocs.vertices, dtype=np.float64)
    gt_diameter = calc_pts_diameter(gt_vertices_eval)
    ob_id = REAL275_OBJECTS.index(obj_name) + 1
    renderer.my_add_object(
        {
            "pts": gt_vertices_eval * 1e3,
            "normals": np.asarray(gt_mesh_nocs.face_normals),
            "faces": np.asarray(gt_mesh_nocs.faces),
        },
        ob_id,
    )
    trans_disc = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
    if os.path.exists(models_info_path):
        with open(models_info_path, "r", encoding="utf-8") as handle:
            models_info = json.load(handle)
        for symmetry in models_info.get(str(ob_id), {}).get("symmetries_discrete", []):
            matrix = np.reshape(symmetry, (4, 4))
            trans_disc.append(
                {"R": matrix[:3, :3], "t": matrix[:3, 3].reshape((3, 1))}
            )
    return {
        "anchor_gt_pose": anchor_gt_pose,
        "gt_mesh_nocs": gt_mesh_nocs,
        "gt_vertices_eval": gt_vertices_eval,
        "gt_diameter": gt_diameter,
        "ob_id": ob_id,
        "is_symmetric": obj_name in REVOLUTION_SYMMETRIC_OBJECTS,
        "use_symmetry_metrics": len(trans_disc) > 1,
        "sym_axis": "discrete",
        "trans_disc": trans_disc,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="REAL275 paper-protocol query evaluation")
    parser.add_argument("--name", type=str, default="any6d_real275")
    parser.add_argument(
        "--anchor_path", type=str,
        required=True,
        help="Anchor result folder containing candidate_registry.json",
    )
    parser.add_argument(
        "--real275_data_root", type=str,
        required=True,
    )
    parser.add_argument("--running_stride", type=int, default=10)
    parser.add_argument("--register_iteration", type=int, default=5)
    parser.add_argument(
        "--obj_classes", type=str, default="",
        help="Comma-separated object classes to evaluate (empty = all 18)",
    )
    parser.add_argument(
        "--scenes", type=str, default="",
        help="Comma-separated scenes to evaluate (empty = all 6)",
    )
    parser.add_argument("--min_mask_pixels", type=int, default=64)
    parser.add_argument(
        "--models_info_path",
        type=str,
        default="./real275_models_info.json",
        help=(
            "Path to REAL275 models_info.json (BOP format) for discrete "
            "symmetry transforms. Used only when eval_scale_mode=baseline_fair."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mesh_diameter_mode",
        choices=["bbox_diagonal", "sampled_surface"],
        default="bbox_diagonal",
        help=(
            "FoundationPose registration diameter. bbox_diagonal restores the "
            "validated v7 protocol; sampled_surface is retained only for A/B."
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

    seed_everything(int(args.seed))
    selected_labels = validate_query_protocol(args)

    anchor_path = args.anchor_path
    data_root = args.real275_data_root
    gt_dir = os.path.join(data_root, 'gts', 'real_test')
    model_dir = os.path.join(data_root, 'obj_models', 'real_test')

    date_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S}'
    save_root = f"./results/real275_results/{args.name}/{date_str}"
    os.makedirs(save_root, exist_ok=True)
    registry_path = os.path.join(anchor_path, "candidate_registry.json")
    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"MGC-6D evaluation requires {registry_path}. Run the REAL275 anchor first."
        )
    with open(registry_path, "r", encoding="utf-8") as f:
        candidate_registry = json.load(f)
    print(f"Loaded candidate registry with {len(candidate_registry)} objects")

    run_manifest = {
        "dataset": "REAL275",
        "anchor_path": os.path.abspath(anchor_path),
        "registry_path": os.path.abspath(registry_path),
        "seed": int(args.seed),
        "mask_source": "ground-truth instance mask",
        "mask_sources": {
            "anchor": "dataset_ground_truth_instance_mask",
            "query": "dataset_ground_truth_instance_mask",
        },
        "query_protocol": "independent anchor-query",
        "candidate_labels": sorted(selected_labels or []),
        "candidate_color_mode": args.candidate_color_mode,
        "mesh_diameter_mode": args.mesh_diameter_mode,
        "symmetry": {
            "gt_evaluation": (
                "dataset/BOP symmetry definitions remain active in ADD(-S), MSSD, MSPD and VSD"
            ),
            "candidate_local_anchor_adjustment": False,
            "candidate_local_temporal_adjustment": False,
            "separated_contract": True,
        },
        "gt_usage": {
            "candidate_registration": False,
            "candidate_scoring": False,
            "candidate_selection": False,
            "anchor_gt_pose": "relative-pose evaluation conversion after winner selection only",
            "query_gt_pose": "loaded only after winner selection for final metrics",
            "gt_cad": "evaluation module only; never passed to registration, scoring, or selection",
        },
        "selection": {
            "policy": "weighted",
            "rule": "direct weighted argmax; no default candidate or hard replacement gate",
            "formula": "alpha*S_anchor_norm+beta*(-wd*Ld-wm*Lm-wg*Lg)",
            "paper_equation_22": True,
            "loss_normalization": args.query_loss_normalization,
            "w_depth": args.w_depth,
            "w_mask": args.w_mask,
            "w_geom": args.w_geom,
            "w_appearance": args.w_appearance,
            "depth_scale": args.query_depth_scale,
            "mask_scale": args.query_mask_scale,
            "geom_scale": args.query_geom_scale,
            "appearance_scale": args.query_appearance_scale,
            "score_alpha": args.score_alpha,
            "score_beta": args.score_beta,
        },
        "provenance": runtime_provenance(os.path.dirname(__file__)),
        "args": vars(args),
    }
    with open(os.path.join(save_root, "query_run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

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
    # Defer EGL initialization until after argument parsing.
    from bop_toolkit_lib.renderer_vispy import RendererVispy

    renderer = RendererVispy(640, 480, mode='depth')

    object_metrics = {
        obj: {
            'ADD': [], 'ADD-S': [], 'ADD(-S)': [], 'AR': [], 'VSD': [],
            'MSSD': [], 'MSPD': [], 'R error': [], 'T error': [],
            'cls_id': [], 'instance_id': [],
            'scene': [], 'frame_num': [], 'chosen_candidate': [],
            'selection_time_s': [],
        }
        for obj in obj_list
    }
    all_frame_data = {
        'Frame_ID': [], 'Scene': [], 'Frame_Num': [], 'Class': [],
        'ADD-S': [], 'ADD': [], 'ADD(-S)': [], 'AR': [],
        'MSSD': [], 'MSPD': [], 'VSD': [],
        'R_error': [], 'T_error': [], 'Chosen_Candidate': [],
        'Selection_Time_s': [],
    }

    data = []
    obj_count = 0
    evaluated_objects = []
    candidate_score_rows = []
    selection_times = []
    registration_failure_frames = 0
    sampled_frame_instances = 0

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
        anchor_mesh = trimesh.load(anchor_mesh_path)
        anchor_k = np.loadtxt(anchor_k_path)
        anchor_color_bgr = cv2.imread(os.path.join(obj_anchor_dir, "color.png"))
        anchor_mask_raw = cv2.imread(
            os.path.join(obj_anchor_dir, "mask.png"), cv2.IMREAD_UNCHANGED
        )
        anchor_depth_raw = cv2.imread(
            os.path.join(obj_anchor_dir, "depth.png"), cv2.IMREAD_UNCHANGED
        )
        if anchor_color_bgr is None or anchor_mask_raw is None:
            print(f"Skip {obj_name}: missing saved anchor RGB/mask")
            continue
        anchor_rgb = cv2.cvtColor(anchor_color_bgr, cv2.COLOR_BGR2RGB)
        anchor_mask = (
            np.any(anchor_mask_raw > 0, axis=2)
            if anchor_mask_raw.ndim == 3 else anchor_mask_raw > 0
        )
        anchor_depth = (
            anchor_depth_raw.astype(np.float32) / 1e3
            if anchor_depth_raw is not None else None
        )

        # The formal candidate registry below owns the mesh/pose contract.
        # Do not apply an object-level centering offset here: doing so a second
        # time was one source of REAL275 anchor/query frame inconsistency.

        evaluation_resources = None
        anchor_gt_pose_evaluation_only = None
        gt_mesh_nocs = None
        gt_vertices_eval = None
        gt_diameter = float("nan")
        ob_id = REAL275_OBJECTS.index(obj_name) + 1
        is_symmetric = False
        use_symmetry_metrics = False
        sym_axis = "deferred"
        trans_disc = None

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
        #  4. Reset estimator and load candidate-local frame contracts
        # ============================================================
        # Load the formal candidate registry.  Each geometry keeps its own
        # anchor pose; reusing the primary pose for every mesh corrupts the
        # relative transform when candidate coordinate frames differ.
        candidate_meshes = []
        obj_registry = candidate_registry.get(obj_name, {})
        if (
            obj_registry.get("anchor_score_version")
            != "bounded_observation_calibrated_v1"
        ):
            raise ValueError(
                f"{obj_name}: legacy or missing anchor score contract; "
                "regenerate the REAL275 anchor with the current code"
            )
        registry_candidates = paper_registry_candidates(
            obj_registry.get("candidates", {})
        )
        registry_candidates = {
            label: info
            for label, info in registry_candidates.items()
            if label in selected_labels
        }
        for label, info in registry_candidates.items():
            if (
                info.get("anchor_score_version")
                != "bounded_observation_calibrated_v1"
            ):
                raise ValueError(
                    f"{obj_name}/{label}: missing bounded anchor score metadata; "
                    "regenerate the REAL275 anchor with the current code"
                )
            mesh_path = info.get("mesh_path", "")
            pose_path = info.get("pose_path", "")
            if not (os.path.exists(mesh_path) and os.path.exists(pose_path)):
                warnings.warn(f"Skip missing REAL275 candidate {obj_name}/{label}")
                continue
            candidate_mesh = trimesh.load(mesh_path, process=False)
            if isinstance(candidate_mesh, trimesh.Scene):
                candidate_mesh = trimesh.util.concatenate(
                    [g for g in candidate_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                )
            candidate_anchor_pose = np.loadtxt(pose_path)
            candidate_model_center = np.asarray(
                info.get("model_center", [0.0, 0.0, 0.0]), dtype=np.float64
            ).reshape(3)
            mesh_frame = str(info.get("mesh_frame", "legacy_unknown"))
            pose_frame = str(info.get("pose_frame", "legacy_unknown"))
            if (
                mesh_frame == "any6d_internal_centered"
                and pose_frame == "any6d_internal_centered_to_anchor_camera"
            ):
                # Exact current contract: saved pose already maps the saved
                # mesh vertices to the anchor camera.  No query correction.
                pass
            elif pose_frame == "legacy_precenter_mesh_to_anchor_camera":
                # Explicitly tagged compatibility path only.  Untagged old
                # registries are not silently transformed.
                center_to_raw = np.eye(4, dtype=np.float64)
                center_to_raw[:3, 3] = candidate_model_center
                candidate_anchor_pose = candidate_anchor_pose @ center_to_raw
            else:
                warnings.warn(
                    f"Skip {obj_name}/{label}: unsupported mesh/pose frame "
                    f"contract ({mesh_frame!r}, {pose_frame!r}); regenerate "
                    "the REAL275 anchor with the current anchor script."
                )
                continue
            candidate_meshes.append(
                {
                    "label": label,
                    "mesh": candidate_mesh,
                    "anchor_pose": candidate_anchor_pose,
                    "registry_model_center": candidate_model_center,
                    "mesh_frame": mesh_frame,
                    "pose_frame": pose_frame,
                    "s_anchor": float(info.get("s_anchor", 1.0)),
                    "anchor_observation_quality": float(
                        info.get("anchor_observation_quality", info.get("s_anchor", 1.0))
                    ),
                    "fusion_admission": info.get("fusion_admission", {}),
                    "registry_info": dict(info),
                    "shared_anchor_reliability": compute_anchor_reliability_components(
                        info, obj_registry
                    ),
                }
            )
        if not candidate_meshes:
            print(f"Skip {obj_name}: no valid registry candidates")
            continue

        base_meshes_for_appearance = {
            item["label"]: item["mesh"].copy() for item in candidate_meshes
        }
        anchor_poses_for_appearance = {
            item["label"]: item["anchor_pose"] for item in candidate_meshes
        }
        appearance_meshes, appearance_summary = apply_candidate_appearance(
            base_meshes_for_appearance,
            anchor_poses_for_appearance,
            args.candidate_color_mode,
            anchor_rgb=anchor_rgb,
            anchor_mask=anchor_mask,
            anchor_k=anchor_k,
            anchor_depth=anchor_depth,
        )
        for item in candidate_meshes:
            item["mesh"] = appearance_meshes[item["label"]]
            item["cache"] = CandidateCache(
                item["label"], item["mesh"], args.mesh_diameter_mode
            )
        anchor_mesh = candidate_meshes[0]["mesh"]
        pred_pose_a = candidate_meshes[0]["anchor_pose"]
        print(
            f"{obj_name}: appearance={appearance_summary.mode}, "
            f"color_source={appearance_summary.source_label}, "
            f"transferred={list(appearance_summary.transferred_labels)}, "
            f"projected={list(appearance_summary.projected_labels)}"
        )
        swap_candidate(est, candidate_meshes[0]["cache"])
        sampled_frame_instances += len(query_frames)
        print(
            f"\n{obj_name}: {len(query_frames)} query frames "
            f"(stride={args.running_stride}, protocol=baseline_fair, "
            f"candidates={len(candidate_meshes)})"
        )

        # ============================================================
        #  5. Per-frame evaluation loop (SIMPLE — no depth correction)
        # ============================================================
        for scene, frame_id, inst_id in tqdm(
            query_frames, desc=f"{obj_name}"
        ):
            scene_dir = os.path.join(data_root, 'real_test', scene)

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

            # Registration and Eq. (22) selection intentionally happen before
            # loading the query GT pose. GT instance masks remain the disclosed
            # input protocol, but pose/CAD annotations cannot affect selection.
            init_pose_cur = None
            selection_started = time.perf_counter()
            candidate_results = []
            eval_candidate_meshes = candidate_meshes
            for candidate in eval_candidate_meshes:
                cand_label = candidate["label"]
                cand_mesh = candidate["mesh"]
                cand_pose_a = candidate["anchor_pose"]
                cand_cache = candidate["cache"]
                swap_candidate(est, cand_cache)
                try:
                    cand_pred = est.register(
                        K=K, rgb=color, depth=depth, ob_mask=mask,
                        iteration=args.register_iteration, name=obj_name,
                        glctx=glctx,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Register failed: {obj_name} {scene}/{frame_id}: {exc}"
                    )
                    continue

                try:
                    l_d, l_m, l_g, coverage = compute_observation_consistency(
                        est, depth, mask, K, cand_pred, glctx,
                    )
                except Exception:
                    l_d, l_m, l_g, coverage = 1.0, 1.0, 1e3, 0.0
                candidate_results.append(
                    {
                        "label": cand_label,
                        "mesh": cand_mesh,
                        "cache": cand_cache,
                        "pred_pose_q": cand_pred,
                        "anchor_pose": cand_pose_a,
                        "s_anchor": candidate["s_anchor"],
                        "anchor_reliability": candidate["shared_anchor_reliability"]["score"],
                        "anchor_low_reliability": candidate["shared_anchor_reliability"]["low_reliability"],
                        "l_depth": float(l_d),
                        "l_mask": float(l_m),
                        "l_geom": float(l_g),
                        "l_appearance": 1.0,
                        "coverage": float(np.clip(coverage, 0.0, 1.0)),
                    }
                )

            if not candidate_results:
                registration_failure_frames += 1
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
            ranked_results = sorted(
                candidate_results,
                key=lambda row: float(row["total_score"]),
                reverse=True,
            )
            winner = select_scored_candidate(candidate_results)
            runner_up = ranked_results[1] if len(ranked_results) > 1 else None
            score_margin = (
                float(winner["total_score"] - runner_up["total_score"])
                if runner_up is not None else float("nan")
            )
            pred_pose_q = winner["pred_pose_q"]
            pred_pose_a = winner["anchor_pose"]
            best_cand_label = winner["label"]
            swap_candidate(est, winner["cache"])
            selection_time_s = float(time.perf_counter() - selection_started)
            selection_times.append(selection_time_s)

            if evaluation_resources is None:
                evaluation_resources = _load_real275_evaluation_resources(
                    obj_name=obj_name,
                    model_dir=model_dir,
                    anchor_gt_path=anchor_gt_path,
                    models_info_path=args.models_info_path,
                    renderer=renderer,
                )
                anchor_gt_pose_evaluation_only = evaluation_resources["anchor_gt_pose"]
                gt_mesh_nocs = evaluation_resources["gt_mesh_nocs"]
                gt_vertices_eval = evaluation_resources["gt_vertices_eval"]
                gt_diameter = evaluation_resources["gt_diameter"]
                ob_id = evaluation_resources["ob_id"]
                is_symmetric = evaluation_resources["is_symmetric"]
                use_symmetry_metrics = evaluation_resources["use_symmetry_metrics"]
                sym_axis = evaluation_resources["sym_axis"]
                trans_disc = evaluation_resources["trans_disc"]

            gt_rts = load_gt_poses(gt_dir, scene, frame_id)
            if gt_rts is None or inst_id - 1 >= gt_rts.shape[0]:
                continue
            gt_pose_q = gt_rts[inst_id - 1]
            if not np.isfinite(gt_pose_q).all():
                continue
            q_scale = np.linalg.norm(gt_pose_q[:3, :3], axis=0).mean()
            if q_scale < 1e-4:
                continue

            rank_by_label = {
                result["label"]: rank
                for rank, result in enumerate(ranked_results, start=1)
            }
            for result in candidate_results:
                audit_row = {
                        "Scene": scene,
                        "Frame_Num": frame_id,
                        "Instance_ID": inst_id,
                        "Class": obj_name,
                        "Candidate": result["label"],
                        "Selected": result is winner,
                        "S_anchor": result["s_anchor"],
                        "S_anchor_norm": result["s_anchor_norm"],
                        "Anchor_Reliability": result.get("anchor_reliability", result["s_anchor"]),
                        "Anchor_Low_Reliability": result.get("anchor_low_reliability", False),
                        "L_depth": result["l_depth"],
                        "L_mask": result["l_mask"],
                        "L_geom": result["l_geom"],
                        "L_appearance": result.get("l_appearance", 1.0),
                        "L_depth_norm": result["l_depth_norm"],
                        "L_mask_norm": result["l_mask_norm"],
                        "L_geom_norm": result["l_geom_norm"],
                        "L_appearance_norm": result.get("l_appearance_norm", 0.0),
                        "Mesh_Fingerprint": result["cache"].fingerprint,
                        "Mesh_Vertices": int(len(result["cache"].mesh.vertices)),
                        "Mesh_Diameter": float(result["cache"].diameter),
                        "Coverage": result["coverage"],
                        "S_query": result["s_query"],
                        "Total_Score": result["total_score"],
                        "Selection_Rank": rank_by_label[result["label"]],
                        "Winning_Score": float(winner["total_score"]),
                        "Runner_Up_Candidate": (
                            runner_up["label"] if runner_up is not None else ""
                        ),
                        "Score_Margin": score_margin,
                        "Selection_Time_s": selection_time_s,
                    }
                candidate_score_rows.append(audit_row)
                result["audit_row"] = audit_row

            # --- Relative pose transfer ---
            try:
                pose_aq = pred_pose_q @ np.linalg.inv(pred_pose_a)
            except np.linalg.LinAlgError:
                continue
            pred_q_world = pose_aq @ anchor_gt_pose_evaluation_only

            # Fixed baseline-fair protocol: pure SE(3), like HO3D.
            _, pred_eval = decompose_nocs_pose(pred_q_world)
            _, gt_eval = decompose_nocs_pose(gt_pose_q)

            if (not np.isfinite(pred_eval).all()
                    or not np.isfinite(gt_eval).all()):
                continue

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
            m['chosen_candidate'].append(best_cand_label)
            m['selection_time_s'].append(selection_time_s)

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
            'Chosen_Candidate': m['chosen_candidate'],
            'Selection_Time_s': m['selection_time_s'],
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
            'Selection_Time_s': np.mean(m['selection_time_s']),
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
            'Chosen_Candidate': [''],
            'Selection_Time_s': [f"{means_obj['Selection_Time_s']:.3f}"],
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
        all_frame_data['Chosen_Candidate'].extend(m['chosen_candidate'])
        all_frame_data['Selection_Time_s'].extend(m['selection_time_s'])

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
            'Chosen_Candidate': '',
            'Selection_Time_s': f"{df_all['Selection_Time_s'].mean():.3f}",
        }
        df_all = pd.concat(
            [df_all, pd.DataFrame([means_all])], ignore_index=True
        )

    output_path = f'{save_root}/0_all_frames_metrics_results.xlsx'
    df_all.to_excel(output_path, index=False)
    score_output_path = f'{save_root}/0_all_candidate_scores.xlsx'
    pd.DataFrame(candidate_score_rows).to_excel(score_output_path, index=False)
    selection_count_path = f'{save_root}/0_candidate_selection_counts.xlsx'
    if len(df_all) > 0:
        selected_frames = df_all[df_all['Frame_ID'] != 'MEAN']
        selection_counts = (
            selected_frames['Chosen_Candidate']
            .value_counts(dropna=False)
            .rename_axis('Candidate')
            .reset_index(name='Selected_Frames')
        )
        selection_counts['Selection_Rate'] = (
            selection_counts['Selected_Frames']
            / max(int(selection_counts['Selected_Frames'].sum()), 1)
        )
    else:
        selection_counts = pd.DataFrame(
            columns=['Candidate', 'Selected_Frames', 'Selection_Rate']
        )
    selection_counts.to_excel(selection_count_path, index=False)
    run_manifest["completed"] = True
    run_manifest["evaluated_frames"] = int(obj_count)
    run_manifest["evaluated_objects"] = list(evaluated_objects)
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
    score_frame = pd.DataFrame(candidate_score_rows)
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
    print(f"\nAll frames metrics saved to {output_path}")
    print(f"Candidate scores saved to {score_output_path}")
    print(f"Candidate selection counts saved to {selection_count_path}")
    print(selection_counts.to_string(index=False))
    print(f"\nEvaluated {len(evaluated_objects)}/{len(obj_list)} objects, "
          f"{obj_count} total frames")
    print("\nSaved data preview:")
    print(df_all)
