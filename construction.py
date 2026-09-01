
from project_paths import setup_project_paths
setup_project_paths()
import json
import os
import glob
import hashlib
import itertools
from dataclasses import dataclass, asdict, replace
from typing import Dict, Optional, Tuple, List, Iterable

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree

from sam2_instantmesh import preprocess_image, diffusion_image_generation, instant_mesh_process
from sam2_rayst3r import (
    Rayst3RRunner,
    get_bounding_box,
    running_sam_box,
    pointcloud_to_mesh,
)
from pose_utils import preprocess_depth, guess_translation


@dataclass
class ReconstructionConfig:
    anchor_root: str
    output_root: Optional[str] = None
    object_names: Optional[Tuple[str, ...]] = None
    depth_unit: str = "auto"  # auto|mm|rayst3r_uint16|meters
    depth_unit_try_both: bool = False
    depth_preprocess: bool = True
    depth_filter_radius: int = 2
    depth_filter_device: str = "cuda"
    valid_ratio_threshold: float = 0.01
    depth_range_min_m: float = 0.2
    depth_range_max_m: float = 3.0
    depth_range_min_ratio: float = 0.3
    depth_range_soft_penalty: bool = False
    refine_mask: bool = False
    mask_stability_enabled: bool = False
    mask_stability_threshold: float = 0.6
    mask_iou_threshold: float = 0.5
    rayst3r_device: str = "cuda:0"
    rayst3r_checkpoint: Optional[str] = None
    rayst3r_set_conf: float = 2.5
    rayst3r_n_pred_views: int = 5
    rayst3r_filter_all_masks: bool = True
    rayst3r_tsdf: bool = False
    rayst3r_voxel_size: float = 0.002
    rayst3r_std_ratio: float = 2.5
    rayst3r_poisson_depth: int = 9
    rayst3r_poisson_scale: float = 1.0
    rayst3r_density_quantile: float = 0.02
    instantmesh_device: str = "cuda:1"
    instantmesh_flip: bool = True
    instantmesh_remove_bg: bool = True
    align_use_guess_translation: bool = True
    align_bidirectional_icp: bool = True
    align_strict_both: bool = False
    align_relaxed_rescue: bool = False
    icp_max_iter: int = 50
    icp_fitness_threshold: float = 0.05
    icp_multi_hypo: int = 1
    icp_reciprocal_pairing: bool = False
    icp_init_sigma: float = 0.01
    icp_max_corr_ratio: float = 0.05  # fraction of rayst3r bbox diag
    icp_voxel_size: float = 0.005
    sample_points: int = 50000
    rayst3r_crop_margin_ratio: float = 0.05
    rayst3r_min_component_area_ratio: float = 0.005
    rayst3r_target_faces: int = 100000
    fusion_mode: str = "poisson"  # poisson|visible_patch
    fusion_voxel_size: float = 0.003
    fusion_sample_points: int = 15000
    fusion_poisson_depth: int = 8
    fusion_target_faces: int = 20000
    fusion_density_quantile: float = 0.02
    fusion_min_component_area_ratio: float = 0.005
    fusion_visible_depth_tolerance: float = 0.015
    fusion_observed_support_ratio: float = 0.08
    fusion_agreement_ratio: float = 0.025
    fusion_min_prior_coverage: float = 0.35
    fusion_min_observed_coverage: float = 0.50
    fusion_min_extent_ratio: float = 0.65
    fusion_max_extent_ratio: float = 1.35
    fusion_max_center_offset_ratio: float = 0.15
    fusion_max_update_ratio: float = 0.04
    fusion_min_normal_cosine: float = 0.15
    fusion_smooth_iterations: int = 3
    fusion_global_scale_min: float = 0.88
    fusion_global_scale_max: float = 1.12
    fusion_cycle_in_reliability: bool = True
    rgbd_scale_recovery: bool = False
    rgbd_scale_visible_sim3: bool = False
    rgbd_scale_min_margin: float = 0.002
    rgbd_scale_min_reciprocal_fraction: float = 0.0
    rgbd_scale_silhouette_weight: float = 0.0
    rgbd_scale_min_silhouette_quality: float = 0.0
    rgbd_scale_ambiguity_fallback: bool = False
    rgbd_scale_sim3_iterations: int = 5
    rgbd_scale_grid_steps: int = 9
    rgbd_scale_multiplier_min: float = 0.65
    rgbd_scale_multiplier_max: float = 1.35
    rgbd_scale_z_offset_steps: int = 5
    rgbd_scale_max_points: int = 15000
    rgbd_scale_depth_tolerance: float = 0.03
    rgbd_scale_support_ratio: float = 0.08
    align_score_rmse_weight: float = 40.0
    align_low_fitness_penalty: float = 0.15
    align_high_rmse_penalty: float = 0.15
    align_rmse_penalty_threshold: float = 0.02
    seed: int = 0


SB_SM_OBJECTS = {
    "006_mustard_bottle",
    "021_bleach_cleanser",
    "005_tomato_soup_can",
    "010_potted_meat_can",
}


def _find_file(search_dirs: List[str], preferred_names: List[str], suffix: Optional[str] = None) -> Optional[str]:
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


def _load_intrinsics(search_dirs: List[str]) -> np.ndarray:
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


def _load_object_inputs_raw(obj_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    search_dirs = [obj_dir, os.path.join(obj_dir, "anchor_init")]

    color_path = _find_file(search_dirs, ["color.png"], suffix="_color.png")
    depth_path = _find_file(search_dirs, ["depth.png"], suffix="_depth.png")
    mask_path = _find_file(search_dirs, ["mask.png"], suffix="_mask.png")

    if not color_path or not depth_path:
        raise FileNotFoundError(f"Missing color/depth in {obj_dir}")

    color = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if mask_path and os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask_img > 0).astype(np.bool_)
    else:
        h, w = color.shape[:2]
        full_box = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
        mask = running_sam_box(color, full_box)

    intrinsic = _load_intrinsics(search_dirs)
    return color, depth_raw, mask, intrinsic


def _load_object_inputs(obj_dir: str, cfg: ReconstructionConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    color, depth_raw, mask, intrinsic = _load_object_inputs_raw(obj_dir)
    if cfg.depth_unit == "auto":
        detected = _detect_depth_unit(depth_raw)
    else:
        detected = cfg.depth_unit
    depth_m = _convert_depth(depth_raw, detected)
    return color, depth_m, mask, intrinsic, detected


def _mask_stability_score(mask: np.ndarray, kernel_size: int = 3) -> float:
    if mask is None or mask.sum() == 0:
        return 0.0
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return float(eroded.sum()) / float(mask_u8.sum() + 1e-6)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union + 1e-6)


def _refine_mask_if_needed(color: np.ndarray, mask: np.ndarray, cfg: ReconstructionConfig) -> np.ndarray:
    if not cfg.refine_mask:
        return mask
    if mask is None or mask.sum() == 0:
        h, w = color.shape[:2]
        box = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
    else:
        cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
        box = np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)
    refined = running_sam_box(color, box)
    if not cfg.mask_stability_enabled:
        return refined
    stability = _mask_stability_score(refined)
    iou = _mask_iou(mask, refined) if mask is not None else 1.0
    if stability < cfg.mask_stability_threshold or iou < cfg.mask_iou_threshold:
        return mask
    return refined


def _valid_ratio(depth_m: np.ndarray, mask: np.ndarray) -> float:
    if depth_m is None or mask is None:
        return 0.0
    object_mask = mask.astype(bool)
    object_pixels = int(object_mask.sum())
    if object_pixels == 0:
        return 0.0
    valid = object_mask & np.isfinite(depth_m) & (depth_m > 0.0)
    # This threshold assesses depth completeness *inside the object mask*.
    # Dividing by the full image area silently rejected small HO3D objects
    # before ICP (e.g. potted-meat had 0.7% image coverage despite essentially
    # complete object depth).
    return float(valid.sum()) / float(object_pixels)


def _depth_in_range_ratio(
    depth_m: np.ndarray, mask: np.ndarray, min_m: float, max_m: float
) -> float:
    if depth_m is None or mask is None:
        return 0.0
    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0.0)
    if valid.sum() == 0:
        return 0.0
    in_range = valid & (depth_m >= min_m) & (depth_m <= max_m)
    return float(in_range.sum()) / float(valid.sum() + 1e-6)


def _points_to_o3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def _mesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    return mesh_o3d


def _stable_object_seed(global_seed: int, obj_name: str) -> int:
    """Return an order-independent seed for one reconstruction object."""

    digest = hashlib.sha256(obj_name.encode("utf-8")).digest()
    object_offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((int(global_seed) + object_offset) % (2**31 - 1))


def _set_object_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _proper_axis_rotations() -> List[np.ndarray]:
    """Return the 24 deterministic proper signed-axis rotations."""

    rotations: List[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3), dtype=np.float64)
            rotation[np.arange(3), permutation] = signs
            if np.linalg.det(rotation) > 0.5:
                rotations.append(rotation)
    rotations.sort(key=lambda rotation: float(np.linalg.norm(rotation - np.eye(3))))
    return rotations


def _build_icp_inits(
    source_center: np.ndarray,
    target_center: np.ndarray,
    num_inits: int,
    sigma: float,
) -> List[np.ndarray]:
    source_center = np.asarray(source_center, dtype=np.float64).reshape(3)
    target_center = np.asarray(target_center, dtype=np.float64).reshape(3)
    base = np.eye(4, dtype=np.float64)
    base[:3, 3] = target_center - source_center
    if num_inits <= 1:
        return [base]

    # InstantMesh and RaySt3R can use different canonical orientations.  The
    # previous "multi hypothesis" implementation perturbed translation only,
    # even though centroids were already aligned.  It therefore could not
    # recover a 90/180-degree orientation mismatch (HO3D potted-meat was the
    # clearest failure).  Use the 24 proper axis rotations deterministically;
    # every transform rotates about the source centroid and maps it to the
    # target centroid.  This is still one ICP alignment stage, not an extra
    # geometry candidate.
    rotations = _proper_axis_rotations()

    rng = np.random.RandomState(0)
    inits = []
    for index in range(max(int(num_inits), 1)):
        rotation = rotations[index % len(rotations)]
        init = np.eye(4, dtype=np.float64)
        init[:3, :3] = rotation
        init[:3, 3] = target_center - rotation @ source_center
        # More than 24 hypotheses add deterministic translation perturbations
        # without replacing the complete orientation set.
        if index >= len(rotations) and sigma > 0:
            init[:3, 3] += rng.normal(scale=sigma, size=(3,))
        inits.append(init)
    return inits


def _icp_registration_hypotheses(
    prior_mesh: trimesh.Trimesh,
    target_points: np.ndarray,
    voxel_size: float,
    max_corr_ratio: float,
    max_iter: int,
    init_transforms: Optional[Iterable[np.ndarray]] = None,
    source_points: Optional[np.ndarray] = None,
) -> List[o3d.pipelines.registration.RegistrationResult]:
    if prior_mesh is None or target_points is None or len(target_points) == 0:
        return []
    target_pcd = _points_to_o3d(target_points)
    if voxel_size and voxel_size > 0:
        target_pcd = target_pcd.voxel_down_sample(voxel_size=voxel_size)

    if source_points is None:
        prior_pcd = _mesh_to_o3d(prior_mesh).sample_points_uniformly(
            number_of_points=min(50000, len(target_pcd.points))
        )
    else:
        prior_pcd = _points_to_o3d(np.asarray(source_points, dtype=np.float64))
    if voxel_size and voxel_size > 0:
        prior_pcd = prior_pcd.voxel_down_sample(voxel_size=voxel_size)

    prior_pcd.estimate_normals()
    target_pcd.estimate_normals()

    target_extent = np.asarray(
        target_pcd.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64
    )
    target_diagonal = max(float(np.linalg.norm(target_extent)), 1e-8)
    max_correspondence_distance = max(
        float(voxel_size) * 2.0,
        float(max_corr_ratio) * target_diagonal,
        1e-6,
    )

    if init_transforms is None:
        init_transforms = [np.eye(4, dtype=np.float64)]

    registrations = []
    for init in init_transforms:
        reg = o3d.pipelines.registration.registration_icp(
            prior_pcd,
            target_pcd,
            max_correspondence_distance=max_correspondence_distance,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
        registrations.append(reg)
    return registrations


def _select_best_icp_registration(
    registrations: Iterable[o3d.pipelines.registration.RegistrationResult],
    fitness_threshold: float,
) -> Tuple[Optional[o3d.pipelines.registration.RegistrationResult], Optional[int]]:
    registrations = list(registrations)
    if not registrations:
        return None, None
    best_index = max(
        range(len(registrations)),
        key=lambda index: (
            float(registrations[index].fitness),
            -float(registrations[index].inlier_rmse),
        ),
    )
    best_reg = registrations[best_index]
    if (not best_reg.fitness) or best_reg.fitness < fitness_threshold:
        return None, best_index
    return best_reg, best_index


def _mesh_with_registration(
    prior_mesh: trimesh.Trimesh,
    registration: Optional[o3d.pipelines.registration.RegistrationResult],
) -> Optional[trimesh.Trimesh]:
    if registration is None:
        return None
    prior_mesh_aligned = prior_mesh.copy()
    prior_mesh_aligned.apply_transform(registration.transformation)
    return prior_mesh_aligned


def _icp_align_prior(
    prior_mesh: trimesh.Trimesh,
    target_points: np.ndarray,
    voxel_size: float,
    max_corr_ratio: float,
    max_iter: int,
    fitness_threshold: float,
    init_transforms: Optional[Iterable[np.ndarray]] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[o3d.pipelines.registration.RegistrationResult]]:
    registrations = _icp_registration_hypotheses(
        prior_mesh,
        target_points,
        voxel_size,
        max_corr_ratio,
        max_iter,
        init_transforms,
    )
    best_reg, _ = _select_best_icp_registration(
        registrations, fitness_threshold
    )

    if best_reg is None:
        return None, None
    return _mesh_with_registration(prior_mesh, best_reg), best_reg


def _rayst3r_reconstruct(
    runner: Rayst3RRunner,
    color: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    cfg: ReconstructionConfig,
    work_dir: str,
) -> Tuple[np.ndarray, Optional[trimesh.Trimesh]]:
    points, _ = runner.reconstruct_from_arrays(
        color,
        depth_m,
        mask,
        intrinsic,
        work_dir=work_dir,
        set_conf=cfg.rayst3r_set_conf,
        n_pred_views=cfg.rayst3r_n_pred_views,
        filter_all_masks=cfg.rayst3r_filter_all_masks,
        tsdf=cfg.rayst3r_tsdf,
        return_conf=False,
    )
    mesh = pointcloud_to_mesh(
        points,
        voxel_size=cfg.rayst3r_voxel_size,
        nb_neighbors=20,
        std_ratio=cfg.rayst3r_std_ratio,
        poisson_depth=cfg.rayst3r_poisson_depth,
        poisson_scale=cfg.rayst3r_poisson_scale,
        density_quantile=cfg.rayst3r_density_quantile,
    )
    mesh = _clean_rayst3r_mesh(mesh, points, cfg)
    return points, mesh


def _clean_rayst3r_mesh(
    mesh: Optional[trimesh.Trimesh],
    points: np.ndarray,
    cfg: ReconstructionConfig,
) -> Optional[trimesh.Trimesh]:
    """Crop Poisson spillover, remove tiny islands, and cap mesh complexity."""

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh
    original = mesh.copy()
    mesh_o3d = _mesh_to_o3d(mesh)

    points_np = np.asarray(points, dtype=np.float64)
    finite_points = points_np[np.isfinite(points_np).all(axis=1)]
    if len(finite_points) >= 4:
        lo = finite_points.min(axis=0)
        hi = finite_points.max(axis=0)
        extent = np.maximum(hi - lo, 1e-8)
        margin = float(cfg.rayst3r_crop_margin_ratio) * extent
        bbox = o3d.geometry.AxisAlignedBoundingBox(lo - margin, hi + margin)
        cropped = mesh_o3d.crop(bbox)
        if len(cropped.vertices) and len(cropped.triangles):
            mesh_o3d = cropped

    if len(mesh_o3d.triangles):
        triangle_clusters, _, cluster_areas = mesh_o3d.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters, dtype=np.int64)
        cluster_areas = np.asarray(cluster_areas, dtype=np.float64)
        if cluster_areas.size:
            min_area = float(cluster_areas.max()) * float(
                cfg.rayst3r_min_component_area_ratio
            )
            remove_mask = cluster_areas[triangle_clusters] < min_area
            if np.any(remove_mask) and not np.all(remove_mask):
                mesh_o3d.remove_triangles_by_mask(remove_mask)
                mesh_o3d.remove_unreferenced_vertices()

    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_unreferenced_vertices()

    target_faces = int(cfg.rayst3r_target_faces)
    if target_faces > 0 and len(mesh_o3d.triangles) > target_faces:
        simplified = mesh_o3d.simplify_quadric_decimation(target_faces)
        if len(simplified.vertices) and len(simplified.triangles):
            mesh_o3d = simplified

    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    if len(vertices) == 0 or len(faces) == 0:
        return original
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _instantmesh_reconstruct(
    color: np.ndarray,
    mask: np.ndarray,
    obj_dir: str,
    obj_name: str,
    cfg: ReconstructionConfig,
    object_seed: int,
) -> trimesh.Trimesh:
    input_image = preprocess_image(
        color,
        mask,
        obj_dir,
        name=obj_name,
        rem_bg=cfg.instantmesh_remove_bg,
        flip=cfg.instantmesh_flip,
    )
    images = diffusion_image_generation(
        obj_dir,
        obj_dir,
        name=obj_name,
        input_image=input_image,
        device=cfg.instantmesh_device,
        seed=object_seed,
    )
    instant_mesh_process(images, obj_dir, name=obj_name, device=cfg.instantmesh_device)
    mesh_path = os.path.join(obj_dir, f"mesh_{obj_name}.obj")
    return trimesh.load(mesh_path)


def _mesh_to_o3d_pcd(mesh: trimesh.Trimesh, n_points: int) -> o3d.geometry.PointCloud:
    if n_points <= 0:
        n_points = 10000
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def _prepare_instantmesh_metric_frame(
    mesh_instant: trimesh.Trimesh,
    ray_points: np.ndarray,
    init_center: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    """Put the scale-ambiguous InstantMesh output in the RaySt3R metric frame.

    This single preparation is shared by both ICP directions.  Applying scale
    before translation avoids scaling the initialized camera-frame centroid.
    """

    if ray_points is None or len(ray_points) == 0:
        return mesh_instant.copy()

    prepared = mesh_instant.copy()
    ray_np = np.asarray(ray_points, dtype=np.float64)
    inst_np = np.asarray(prepared.vertices, dtype=np.float64)
    ray_center = np.median(ray_np, axis=0)
    inst_center = np.median(inst_np, axis=0)
    ray_radius = float(np.quantile(np.linalg.norm(ray_np - ray_center, axis=1), 0.90))
    inst_radius = float(np.quantile(np.linalg.norm(inst_np - inst_center, axis=1), 0.90))
    if np.isfinite(ray_radius) and np.isfinite(inst_radius) and inst_radius > 1e-8:
        scale = float(np.clip(ray_radius / inst_radius, 1e-3, 1e3))
        prepared.apply_scale(scale)
    else:
        scale = 1.0

    target_center = (
        np.asarray(init_center, dtype=np.float64).reshape(3)
        if init_center is not None and np.all(np.isfinite(init_center))
        else np.asarray(ray_points, dtype=np.float64).mean(axis=0)
    )
    source_center = np.asarray(prepared.vertices, dtype=np.float64).mean(axis=0)
    translation = target_center - source_center
    prepared.apply_translation(translation)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= float(scale)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    prepared.metadata = dict(getattr(prepared, "metadata", {}) or {})
    prepared.metadata["instantmesh_metric_scale"] = float(scale)
    prepared.metadata["instantmesh_metric_transform"] = transform.tolist()
    return prepared


def _apply_similarity_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 similarity transform to an Nx3 point array."""

    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    points_h = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    return (transform @ points_h.T).T[:, :3]


def _robust_pixel_bounds(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return np.zeros(2, dtype=np.float64), np.ones(2, dtype=np.float64)
    lo = np.array([np.quantile(xs, 0.01), np.quantile(ys, 0.01)], dtype=np.float64)
    hi = np.array([np.quantile(xs, 0.99), np.quantile(ys, 0.99)], dtype=np.float64)
    return lo, np.maximum(hi, lo + 1.0)


def _projected_silhouette_quality(
    points: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[float, float, float]:
    """Compare a complete hypothesis projection with the observed silhouette.

    Visible-surface distance alone can reward an oversized completion because
    only the subset falling inside the anchor mask participates in the fit.
    This GT-free check uses the full projected hypothesis and therefore exposes
    scale/translation solutions that explain the visible patch while violating
    the observed image extent.
    """

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 20:
        return 0.0, 0.0, float("inf")
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    if int(valid.sum()) < 20:
        return 0.0, 0.0, float("inf")
    projected = (np.asarray(intrinsic, dtype=np.float64).reshape(3, 3) @ points[valid].T).T
    uv = projected[:, :2] / z[valid, None]
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) < 20:
        return 0.0, 0.0, float("inf")

    obs_lo, obs_hi = _robust_pixel_bounds(mask)
    obs_size = np.maximum(obs_hi - obs_lo, 1.0)
    obs_center = 0.5 * (obs_lo + obs_hi)
    pred_lo = np.quantile(uv, 0.01, axis=0)
    pred_hi = np.quantile(uv, 0.99, axis=0)
    pred_size = np.maximum(pred_hi - pred_lo, 1.0)
    pred_center = 0.5 * (pred_lo + pred_hi)
    size_error = float(np.mean(np.abs(np.log(pred_size / obs_size))))
    center_error = float(np.linalg.norm((pred_center - obs_center) / obs_size))
    quality = float(np.exp(-size_error - 0.5 * center_error))

    h, w = np.asarray(mask).shape[:2]
    pixels = np.rint(uv).astype(np.int64)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    supported = np.zeros(len(pixels), dtype=bool)
    if np.any(inside):
        supported[inside] = np.asarray(mask, dtype=bool)[
            pixels[inside, 1], pixels[inside, 0]
        ]
    projected_mask_support = float(np.mean(supported))
    return quality, projected_mask_support, size_error


def _umeyama_similarity(source: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    """Least-squares Sim(3) estimate for paired 3-D points."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 3 or source.shape != target.shape:
        return None
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    if not np.isfinite(variance) or variance <= 1e-12:
        return None
    covariance = target_zero.T @ source_zero / float(len(source))
    try:
        u, singular, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 1e-8:
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_mean - scale * rotation @ source_mean
    return transform


def _similarity_scale(transform: np.ndarray) -> float:
    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    determinant = float(np.linalg.det(linear))
    return float(np.cbrt(max(determinant, 1e-24)))


def _visible_surface_sim3_scale_recovery(
    mesh: trimesh.Trimesh,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    cfg: ReconstructionConfig,
    label: str,
) -> Tuple[trimesh.Trimesh, np.ndarray, Dict[str, float | int | str | bool]]:
    """Robustly align the generated visible surface to anchor RGB-D by Sim(3).

    The fit uses only front-most mesh samples projected inside the GT anchor
    mask. Correspondences are reciprocal whenever enough mutual pairs exist,
    then MAD-filtered before a closed-form Umeyama update. The 24 rotations are
    initialization hypotheses of one reconstruction, not geometry candidates.
    Boundary or scale-ambiguous solutions are re-estimated with wider bounds;
    an unresolved solution is rejected rather than silently accepted.
    """

    identity = np.eye(4, dtype=np.float64)
    diagnostics: Dict[str, float | int | str | bool] = {
        "enabled": True,
        "method": "anchor_visible_surface_robust_sim3_v2",
        "label": str(label),
        "valid": False,
        "transform_applied": False,
    }
    observed = _backproject_anchor_depth(
        depth,
        mask,
        intrinsic,
        max_points=min(max(int(cfg.rgbd_scale_max_points), 1000), 5000),
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(observed) < 80 or len(vertices) < 20 or not np.isfinite(vertices).all():
        diagnostics["reason"] = "insufficient_anchor_depth_or_mesh"
        return mesh.copy(), identity, diagnostics

    sample_count = min(max(int(cfg.rgbd_scale_max_points), 1000), 5000)
    try:
        source, _ = trimesh.sample.sample_surface(mesh, sample_count)
    except Exception:
        source = vertices
    source = np.asarray(source, dtype=np.float64)
    source = source[np.isfinite(source).all(axis=1)]
    if len(source) < 80:
        diagnostics["reason"] = "insufficient_surface_samples"
        return mesh.copy(), identity, diagnostics

    observed_center = np.median(observed, axis=0)
    source_center = np.median(source, axis=0)
    observed_radius = float(
        np.quantile(np.linalg.norm(observed - observed_center, axis=1), 0.90)
    )
    source_radius = float(
        np.quantile(np.linalg.norm(source - source_center, axis=1), 0.90)
    )
    if observed_radius <= 1e-6 or source_radius <= 1e-8:
        diagnostics["reason"] = "degenerate_visible_extent"
        return mesh.copy(), identity, diagnostics
    initial_scale = float(np.clip(observed_radius / source_radius, 1e-4, 10.0))
    observed_tree = cKDTree(observed)
    support_distance = max(
        float(cfg.rgbd_scale_depth_tolerance),
        2.0 * float(cfg.icp_voxel_size),
    )
    image_shape = np.asarray(mask).shape[:2]

    def _clamp_similarity(
        transform: np.ndarray,
        lower: float,
        upper: float,
        paired_source: np.ndarray,
        paired_target: np.ndarray,
    ) -> np.ndarray:
        total_scale = _similarity_scale(transform)
        lower_scale = initial_scale * lower
        upper_scale = initial_scale * upper
        clipped_scale = float(np.clip(total_scale, lower_scale, upper_scale))
        if abs(clipped_scale - total_scale) <= 1e-10:
            return transform
        rotation = transform[:3, :3] / max(total_scale, 1e-12)
        corrected = transform.copy()
        corrected[:3, :3] = clipped_scale * rotation
        corrected[:3, 3] = (
            paired_target.mean(axis=0)
            - clipped_scale * rotation @ paired_source.mean(axis=0)
        )
        return corrected

    def _fit_one(init: np.ndarray, lower: float, upper: float, iterations: int):
        transform = init.copy()
        reciprocal_fraction = 0.0
        paired_count = 0
        for _ in range(max(int(iterations), 1)):
            transformed = _apply_similarity_to_points(source, transform)
            u, v, inside, in_mask, observed_depth = _project_points_to_anchor(
                transformed, depth, mask, intrinsic
            )
            front = _zbuffer_visible_vertices(
                u,
                v,
                transformed[:, 2],
                inside,
                in_mask,
                observed_depth,
                image_shape,
                front_slack=max(float(cfg.icp_voxel_size), 0.002),
            )
            visible_ids = np.flatnonzero(front)
            if len(visible_ids) < 30:
                return None
            visible = transformed[visible_ids]
            distance, target_ids = observed_tree.query(visible, k=1, workers=-1)
            back_ids = cKDTree(visible).query(observed, k=1, workers=-1)[1]
            mutual = back_ids[target_ids] == np.arange(len(visible))
            reciprocal_fraction = float(np.mean(mutual))
            pair_mask = mutual.copy()
            if int(pair_mask.sum()) < 30:
                pair_mask[:] = True
            pair_distance = distance[pair_mask]
            median = float(np.median(pair_distance))
            mad = float(np.median(np.abs(pair_distance - median)))
            cutoff = min(
                2.5 * support_distance,
                max(median + 2.5 * max(1.4826 * mad, 1e-5), 0.5 * support_distance),
            )
            pair_mask &= distance <= cutoff
            pair_local_ids = np.flatnonzero(pair_mask)
            if len(pair_local_ids) < 20:
                return None
            # A final trimmed set makes the Sim(3) update insensitive to
            # completion surfaces that happen to project inside the mask.
            ordered = pair_local_ids[np.argsort(distance[pair_local_ids])]
            keep = max(20, int(0.80 * len(ordered)))
            pair_local_ids = ordered[:keep]
            source_ids = visible_ids[pair_local_ids]
            paired_source = source[source_ids]
            paired_target = observed[target_ids[pair_local_ids]]
            updated = _umeyama_similarity(paired_source, paired_target)
            if updated is None:
                return None
            updated = _clamp_similarity(
                updated, lower, upper, paired_source, paired_target
            )
            paired_count = len(paired_source)
            if np.linalg.norm(updated - transform) < 1e-6:
                transform = updated
                break
            transform = updated

        transformed = _apply_similarity_to_points(source, transform)
        u, v, inside, in_mask, observed_depth = _project_points_to_anchor(
            transformed, depth, mask, intrinsic
        )
        front = _zbuffer_visible_vertices(
            u,
            v,
            transformed[:, 2],
            inside,
            in_mask,
            observed_depth,
            image_shape,
            front_slack=max(float(cfg.icp_voxel_size), 0.002),
        )
        visible = transformed[front]
        if len(visible) < 30:
            return None
        surface_distance = observed_tree.query(visible, k=1, workers=-1)[0]
        visible_tree = cKDTree(visible)
        observed_distance = visible_tree.query(observed, k=1, workers=-1)[0]
        trimmed_surface = np.sort(surface_distance)[: max(1, int(0.80 * len(surface_distance)))]
        trimmed_observed = np.sort(observed_distance)[: max(1, int(0.80 * len(observed_distance)))]
        symmetric_distance = float(0.5 * (trimmed_surface.mean() + trimmed_observed.mean()))
        support = float(np.mean(observed_distance <= support_distance))
        quality = float(np.exp(-symmetric_distance / max(support_distance, 1e-8)))
        silhouette_quality, projected_mask_support, silhouette_size_error = (
            _projected_silhouette_quality(transformed, mask, intrinsic)
        )
        silhouette_weight = float(
            np.clip(cfg.rgbd_scale_silhouette_weight, 0.0, 0.50)
        )
        visible_score = float(0.75 * quality + 0.25 * support)
        score = float(
            (1.0 - silhouette_weight) * visible_score
            + silhouette_weight * silhouette_quality
        )
        total_scale = _similarity_scale(transform)
        multiplier = total_scale / max(initial_scale, 1e-12)
        return {
            "transform": transform,
            "score": score,
            "scale": total_scale,
            "multiplier": multiplier,
            "symmetric_distance": symmetric_distance,
            "support": support,
            "reciprocal_fraction": reciprocal_fraction,
            "paired_points": int(paired_count),
            "silhouette_quality": silhouette_quality,
            "projected_mask_support": projected_mask_support,
            "silhouette_size_error": silhouette_size_error,
        }

    def _search(lower: float, upper: float, iterations: int):
        records = []
        rotations = _proper_axis_rotations()
        for rotation in rotations:
            init = np.eye(4, dtype=np.float64)
            init[:3, :3] = initial_scale * rotation
            init[:3, 3] = observed_center - initial_scale * rotation @ source_center
            record = _fit_one(init, lower, upper, iterations)
            if record is not None:
                records.append(record)
        records.sort(key=lambda record: record["score"], reverse=True)
        return records

    lower = float(cfg.rgbd_scale_multiplier_min)
    upper = float(cfg.rgbd_scale_multiplier_max)
    records = _search(lower, upper, cfg.rgbd_scale_sim3_iterations)
    if not records:
        diagnostics["reason"] = "no_finite_visible_sim3_hypothesis"
        return mesh.copy(), identity, diagnostics

    def _margin(best_record, candidates) -> float:
        distinct = [
            record["score"]
            for record in candidates[1:]
            if abs(np.log(max(record["scale"], 1e-12) / max(best_record["scale"], 1e-12)))
            >= 0.02
        ]
        runner_up = max(distinct) if distinct else 0.0
        return max(float(best_record["score"]) - float(runner_up), 0.0)

    best = records[0]
    margin = _margin(best, records)
    boundary = bool(
        np.isclose(best["multiplier"], lower, atol=0.01)
        or np.isclose(best["multiplier"], upper, atol=0.01)
    )
    reestimated = bool(boundary or margin < float(cfg.rgbd_scale_min_margin))
    if reestimated:
        expanded_lower = max(0.35, 0.80 * lower)
        expanded_upper = min(2.50, 1.20 * upper)
        expanded = _search(
            expanded_lower,
            expanded_upper,
            max(int(cfg.rgbd_scale_sim3_iterations) + 2, 3),
        )
        if expanded and expanded[0]["score"] >= best["score"] - 0.01:
            records = expanded
            best = expanded[0]
            lower, upper = expanded_lower, expanded_upper
            margin = _margin(best, records)
            boundary = bool(
                np.isclose(best["multiplier"], lower, atol=0.01)
                or np.isclose(best["multiplier"], upper, atol=0.01)
            )

    weak_reciprocal = bool(
        float(best["reciprocal_fraction"])
        < float(cfg.rgbd_scale_min_reciprocal_fraction)
    )
    weak_silhouette = bool(
        float(best["silhouette_quality"])
        < float(cfg.rgbd_scale_min_silhouette_quality)
    )
    rejected = bool(
        boundary
        or margin < float(cfg.rgbd_scale_min_margin)
        or weak_reciprocal
        or weak_silhouette
    )
    diagnostics.update(
        {
            "initial_scale": initial_scale,
            "chosen_scale": float(best["scale"]),
            "chosen_multiplier": float(best["multiplier"]),
            "objective": float(best["score"]),
            "objective_margin": float(margin),
            "symmetric_visible_distance": float(best["symmetric_distance"]),
            "observed_support": float(best["support"]),
            "reciprocal_pair_fraction": float(best["reciprocal_fraction"]),
            "paired_points": int(best["paired_points"]),
            "silhouette_quality": float(best["silhouette_quality"]),
            "projected_mask_support": float(best["projected_mask_support"]),
            "silhouette_size_error": float(best["silhouette_size_error"]),
            "silhouette_weight": float(cfg.rgbd_scale_silhouette_weight),
            "min_reciprocal_pair_fraction": float(
                cfg.rgbd_scale_min_reciprocal_fraction
            ),
            "min_silhouette_quality": float(
                cfg.rgbd_scale_min_silhouette_quality
            ),
            "at_search_boundary": boundary,
            "weak_reciprocal_pairing": weak_reciprocal,
            "weak_silhouette": weak_silhouette,
            "reestimated": reestimated,
            "rejected": rejected,
            "hypotheses": int(len(records)),
            "search_multiplier_min": float(lower),
            "search_multiplier_max": float(upper),
        }
    )
    if rejected:
        if boundary:
            diagnostics["reason"] = "boundary_solution"
        elif margin < float(cfg.rgbd_scale_min_margin):
            diagnostics["reason"] = "low_scale_objective_margin"
        elif weak_reciprocal:
            diagnostics["reason"] = "low_reciprocal_pair_fraction"
        else:
            diagnostics["reason"] = "low_projected_silhouette_quality"
        return mesh.copy(), identity, diagnostics

    recovered = mesh.copy()
    recovered.apply_transform(best["transform"])
    reliability = float(
        np.clip(
            best["score"]
            * (0.80 + 0.20 * min(margin / max(float(cfg.rgbd_scale_min_margin) * 5.0, 1e-6), 1.0)),
            0.0,
            1.0,
        )
    )
    diagnostics.update(
        {
            "valid": True,
            "transform_applied": True,
            "reliability": reliability,
        }
    )
    recovered.metadata = dict(getattr(recovered, "metadata", {}) or {})
    recovered.metadata["rgbd_scale_recovery"] = diagnostics
    recovered.metadata[f"{label}_metric_scale"] = float(best["scale"])
    return recovered, np.asarray(best["transform"], dtype=np.float64), diagnostics


def _rgbd_self_supervised_scale_recovery(
    mesh: trimesh.Trimesh,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    cfg: ReconstructionConfig,
    label: str,
) -> Tuple[trimesh.Trimesh, np.ndarray, Dict[str, float | int | str | bool]]:
    """Recover one reconstruction's metric scale from anchor RGB-D only.

    RaySt3R and InstantMesh are calibrated independently.  The initial scale is
    obtained from the robust lateral extent of the masked depth points, then a
    small fixed grid jointly searches uniform scale and camera-z placement.
    Candidates are ranked using only observed-to-surface distance, 3-D support,
    and projected silhouette agreement.  No dataset CAD or query ground truth
    enters this operation.
    """

    if cfg.rgbd_scale_visible_sim3:
        recovered, transform, diagnostics = _visible_surface_sim3_scale_recovery(
            mesh, depth, mask, intrinsic, cfg, label
        )
        if bool(diagnostics.get("valid", False)) or not cfg.rgbd_scale_ambiguity_fallback:
            return recovered, transform, diagnostics

        primary_attempt = dict(diagnostics)
        if label == "instantmesh":
            observed = _backproject_anchor_depth(
                depth,
                mask,
                intrinsic,
                max_points=max(int(cfg.rgbd_scale_max_points), 1000),
            )
            if len(observed) >= 50:
                init_center = guess_translation(depth, mask, intrinsic)
                fallback = _prepare_instantmesh_metric_frame(
                    mesh, observed, init_center=init_center
                )
                fallback_meta = dict(getattr(fallback, "metadata", {}) or {})
                fallback_transform = np.asarray(
                    fallback_meta.get("instantmesh_metric_transform", np.eye(4)),
                    dtype=np.float64,
                ).reshape(4, 4)
                fallback_scale = float(
                    fallback_meta.get("instantmesh_metric_scale", 1.0)
                )
                diagnostics = {
                    "enabled": True,
                    "valid": True,
                    "transform_applied": True,
                    "method": "anchor_depth_extent_fallback_after_ambiguous_sim3_v1",
                    "label": str(label),
                    "fallback_used": True,
                    "fallback_reason": primary_attempt.get("reason", "ambiguous_sim3"),
                    "chosen_scale": fallback_scale,
                    "reliability": 0.50,
                    "primary_attempt": primary_attempt,
                }
                fallback.metadata["rgbd_scale_recovery"] = diagnostics
                return fallback, fallback_transform, diagnostics

        # RaySt3R is already RGB-D conditioned and metric.  If its optional
        # Sim(3) correction is ambiguous, preserving the native metric frame is
        # safer than applying an underconstrained single-view similarity.
        diagnostics = {
            "enabled": True,
            "valid": True,
            "transform_applied": False,
            "method": "native_metric_fallback_after_ambiguous_sim3_v1",
            "label": str(label),
            "fallback_used": True,
            "fallback_reason": primary_attempt.get("reason", "ambiguous_sim3"),
            "chosen_scale": 1.0,
            "reliability": 0.50,
            "primary_attempt": primary_attempt,
        }
        native = mesh.copy()
        native.metadata = dict(getattr(native, "metadata", {}) or {})
        native.metadata["rgbd_scale_recovery"] = diagnostics
        return native, np.eye(4, dtype=np.float64), diagnostics

    identity = np.eye(4, dtype=np.float64)
    diagnostics: Dict[str, float | int | str | bool] = {
        "enabled": bool(cfg.rgbd_scale_recovery),
        "method": "anchor_rgbd_extent_depth_support_v1",
        "label": str(label),
        "valid": False,
    }
    if not cfg.rgbd_scale_recovery:
        return mesh.copy(), identity, diagnostics

    observed = _backproject_anchor_depth(
        depth,
        mask,
        intrinsic,
        max_points=max(int(cfg.rgbd_scale_max_points), 1000),
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(observed) < 50 or len(vertices) < 20 or not np.isfinite(vertices).all():
        diagnostics["reason"] = "insufficient_anchor_depth_or_mesh"
        return mesh.copy(), identity, diagnostics

    sample_count = min(
        max(int(cfg.rgbd_scale_max_points), 1000),
        max(int(len(mesh.faces)), 1000),
    )
    try:
        surface_points, _ = trimesh.sample.sample_surface(mesh, sample_count)
    except Exception:
        surface_points = vertices
    surface_points = np.asarray(surface_points, dtype=np.float64)
    surface_points = surface_points[np.isfinite(surface_points).all(axis=1)]
    if len(surface_points) < 20:
        diagnostics["reason"] = "insufficient_surface_samples"
        return mesh.copy(), identity, diagnostics

    observed_center = np.median(observed, axis=0)
    source_center = np.median(surface_points, axis=0)
    observed_centered = observed - observed_center
    surface_centered = surface_points - source_center

    observed_xy_radius = float(
        np.quantile(np.linalg.norm(observed_centered[:, :2], axis=1), 0.95)
    )
    source_xy_radius = float(
        np.quantile(np.linalg.norm(surface_centered[:, :2], axis=1), 0.95)
    )
    observed_radius = float(
        np.quantile(np.linalg.norm(observed_centered, axis=1), 0.95)
    )
    source_radius = float(
        np.quantile(np.linalg.norm(surface_centered, axis=1), 0.95)
    )
    if observed_xy_radius > 1e-6 and source_xy_radius > 1e-8:
        initial_scale = observed_xy_radius / source_xy_radius
        initial_source = "masked_depth_lateral_extent"
    elif observed_radius > 1e-6 and source_radius > 1e-8:
        initial_scale = observed_radius / source_radius
        initial_source = "masked_depth_radial_extent"
    else:
        diagnostics["reason"] = "degenerate_extent"
        return mesh.copy(), identity, diagnostics
    initial_scale = float(np.clip(initial_scale, 1e-4, 10.0))

    grid_steps = max(int(cfg.rgbd_scale_grid_steps), 1)
    if grid_steps == 1:
        multipliers = np.array([1.0], dtype=np.float64)
    else:
        multipliers = np.linspace(
            float(cfg.rgbd_scale_multiplier_min),
            float(cfg.rgbd_scale_multiplier_max),
            grid_steps,
            dtype=np.float64,
        )
        if not np.any(np.isclose(multipliers, 1.0)):
            multipliers = np.unique(np.concatenate([multipliers, [1.0]]))

    z_steps = max(int(cfg.rgbd_scale_z_offset_steps), 1)
    # The measured depth is the visible front surface, so the completed-model
    # center is searched at or behind it, never with an object-specific offset.
    z_offset_fractions = np.linspace(0.0, 0.40, z_steps, dtype=np.float64)
    source_tree = cKDTree(surface_centered)
    obs_lo, obs_hi = _robust_pixel_bounds(mask)
    obs_size = np.maximum(obs_hi - obs_lo, 1.0)
    obs_pixel_center = 0.5 * (obs_lo + obs_hi)
    observed_diameter = max(2.0 * observed_radius, 1e-4)
    support_distance = max(
        float(cfg.rgbd_scale_depth_tolerance),
        float(cfg.rgbd_scale_support_ratio) * observed_diameter,
    )
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)

    best = None
    all_scores: List[float] = []
    for multiplier in multipliers:
        scale = float(initial_scale * multiplier)
        model_diameter = max(2.0 * source_radius * scale, observed_diameter)
        for z_fraction in z_offset_fractions:
            target_center = observed_center.copy()
            target_center[2] += float(z_fraction) * model_diameter

            observed_local = (observed - target_center) / max(scale, 1e-8)
            nearest = source_tree.query(observed_local, k=1, workers=-1)[0] * scale
            finite_nearest = nearest[np.isfinite(nearest)]
            if len(finite_nearest) == 0:
                continue
            clipped = np.clip(finite_nearest, 0.0, 3.0 * support_distance)
            depth_distance = float(np.mean(np.sort(clipped)[: max(1, int(0.90 * len(clipped)))]))
            depth_quality = float(
                np.exp(-depth_distance / max(float(cfg.rgbd_scale_depth_tolerance), 1e-6))
            )
            support = float(np.mean(finite_nearest <= support_distance))

            transformed_surface = surface_centered * scale + target_center
            z = transformed_surface[:, 2]
            valid_z = z > 1e-6
            if np.count_nonzero(valid_z) < 20:
                silhouette_quality = 0.0
            else:
                projected = (k @ transformed_surface[valid_z].T).T
                uv = projected[:, :2] / z[valid_z, None]
                finite_uv = uv[np.isfinite(uv).all(axis=1)]
                if len(finite_uv) < 20:
                    silhouette_quality = 0.0
                else:
                    pred_lo = np.quantile(finite_uv, 0.01, axis=0)
                    pred_hi = np.quantile(finite_uv, 0.99, axis=0)
                    pred_size = np.maximum(pred_hi - pred_lo, 1.0)
                    pred_center = 0.5 * (pred_lo + pred_hi)
                    size_error = float(
                        np.mean(np.abs(np.log(np.maximum(pred_size, 1.0) / obs_size)))
                    )
                    center_error = float(
                        np.linalg.norm((pred_center - obs_pixel_center) / obs_size)
                    )
                    silhouette_quality = float(
                        np.exp(-size_error - 0.5 * center_error)
                    )

            score = float(
                0.50 * depth_quality + 0.30 * silhouette_quality + 0.20 * support
            )
            all_scores.append(score)
            record = {
                "score": score,
                "scale": scale,
                "multiplier": float(multiplier),
                "z_offset_fraction": float(z_fraction),
                "target_center": target_center,
                "depth_distance": depth_distance,
                "depth_quality": depth_quality,
                "support": support,
                "silhouette_quality": silhouette_quality,
            }
            if best is None or score > best["score"]:
                best = record

    if best is None:
        diagnostics["reason"] = "no_finite_scale_hypothesis"
        return mesh.copy(), identity, diagnostics

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= float(best["scale"])
    transform[:3, 3] = np.asarray(best["target_center"]) - float(best["scale"]) * source_center
    recovered = mesh.copy()
    recovered.apply_transform(transform)

    sorted_scores = np.sort(np.asarray(all_scores, dtype=np.float64))
    runner_up = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
    margin = max(float(best["score"]) - runner_up, 0.0)
    at_multiplier_boundary = bool(
        np.isclose(best["multiplier"], multipliers.min())
        or np.isclose(best["multiplier"], multipliers.max())
    )
    reliability = float(
        np.clip(float(best["score"]) * (0.85 + 0.15 * min(margin / 0.05, 1.0)), 0.0, 1.0)
    )
    if at_multiplier_boundary:
        reliability *= 0.85

    diagnostics.update(
        {
            "valid": True,
            "initial_scale": initial_scale,
            "initial_scale_source": initial_source,
            "chosen_scale": float(best["scale"]),
            "chosen_multiplier": float(best["multiplier"]),
            "chosen_z_offset_fraction": float(best["z_offset_fraction"]),
            "objective": float(best["score"]),
            "objective_margin": margin,
            "depth_distance": float(best["depth_distance"]),
            "depth_quality": float(best["depth_quality"]),
            "observed_support": float(best["support"]),
            "silhouette_quality": float(best["silhouette_quality"]),
            "support_distance": support_distance,
            "reliability": float(reliability),
            "at_search_boundary": at_multiplier_boundary,
            "grid_hypotheses": int(len(multipliers) * len(z_offset_fractions)),
        }
    )
    recovered.metadata = dict(getattr(recovered, "metadata", {}) or {})
    recovered.metadata["rgbd_scale_recovery"] = diagnostics
    recovered.metadata[f"{label}_metric_scale"] = float(best["scale"])
    return recovered, transform, diagnostics


def _align_instantmesh_to_rayst3r(
    mesh_instant_metric: trimesh.Trimesh,
    ray_points: np.ndarray,
    cfg: ReconstructionConfig,
    init_transforms: Optional[Iterable[np.ndarray]] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[o3d.pipelines.registration.RegistrationResult]]:
    if ray_points is None or len(ray_points) == 0:
        return mesh_instant_metric, None

    target_center = ray_points.mean(axis=0)
    source_center = mesh_instant_metric.vertices.mean(axis=0)
    if init_transforms is None:
        init_transforms = _build_icp_inits(
            source_center,
            target_center,
            cfg.icp_multi_hypo,
            cfg.icp_init_sigma,
        )
    aligned, reg = _icp_align_prior(
        mesh_instant_metric,
        ray_points,
        voxel_size=cfg.icp_voxel_size,
        max_corr_ratio=cfg.icp_max_corr_ratio,
        max_iter=cfg.icp_max_iter,
        fitness_threshold=cfg.icp_fitness_threshold,
        init_transforms=init_transforms,
    )
    return aligned, reg


def _align_rayst3r_to_instantmesh(
    mesh_ray: trimesh.Trimesh,
    mesh_instant: trimesh.Trimesh,
    cfg: ReconstructionConfig,
    init_transforms: Optional[Iterable[np.ndarray]] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[o3d.pipelines.registration.RegistrationResult]]:
    target_points, _ = trimesh.sample.sample_surface(mesh_instant, cfg.sample_points)
    target_center = target_points.mean(axis=0)
    source_center = mesh_ray.vertices.mean(axis=0)
    if init_transforms is None:
        init_transforms = _build_icp_inits(
            source_center,
            target_center,
            cfg.icp_multi_hypo,
            cfg.icp_init_sigma,
        )
    aligned, reg = _icp_align_prior(
        mesh_ray,
        target_points,
        voxel_size=cfg.icp_voxel_size,
        max_corr_ratio=cfg.icp_max_corr_ratio,
        max_iter=cfg.icp_max_iter,
        fitness_threshold=cfg.icp_fitness_threshold,
        init_transforms=init_transforms,
    )
    return aligned, reg


def _instantmesh_in_rayst3r_frame(
    mesh_instant: trimesh.Trimesh,
    reg_ray_to_instant: o3d.pipelines.registration.RegistrationResult,
) -> trimesh.Trimesh:
    """Express InstantMesh in the RaySt3R/anchor-camera coordinate frame.

    ``reg_ray_to_instant`` maps RaySt3R to InstantMesh.  Phi consumes anchor
    depth and camera intrinsics, so its inputs must remain in the anchor-camera
    frame.  Moving RaySt3R into the InstantMesh frame and then projecting it
    with the unchanged anchor camera was the cause of the malformed HO3D fused
    surfaces.  Applying the inverse transform to InstantMesh preserves the
    selected best-of-two relative alignment while restoring the correct frame.
    """

    transform = np.asarray(reg_ray_to_instant.transformation, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("ray-to-InstantMesh ICP returned an invalid transform")
    instant_in_ray = mesh_instant.copy()
    instant_in_ray.apply_transform(np.linalg.inv(transform))
    instant_in_ray.metadata = dict(getattr(instant_in_ray, "metadata", {}) or {})
    instant_in_ray.metadata["alignment_frame"] = "rayst3r_anchor_camera"
    instant_in_ray.metadata["alignment_transform_source"] = "inverse_ray_to_inst"
    return instant_in_ray


def _concatenate_meshes(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    """Compatibility fallback used only when surface reconstruction fails."""

    fused = trimesh.util.concatenate([mesh_a, mesh_b])
    try:
        fused.remove_degenerate_faces()
    except Exception:
        pass
    try:
        fused.remove_duplicate_faces()
    except Exception:
        pass
    try:
        fused.merge_vertices()
    except Exception:
        pass
    try:
        fused.remove_unreferenced_vertices()
    except Exception:
        pass
    return fused


def _backproject_anchor_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    max_points: int = 30000,
) -> np.ndarray:
    valid = (
        np.asarray(mask).astype(bool)
        & np.isfinite(depth)
        & (np.asarray(depth, dtype=np.float32) > 1e-5)
    )
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(xs) > max_points:
        # Deterministic uniform subsampling avoids object-order dependence.
        ids = np.linspace(0, len(xs) - 1, max_points, dtype=np.int64)
        ys, xs = ys[ids], xs[ids]
    z = np.asarray(depth, dtype=np.float64)[ys, xs]
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    x = (xs.astype(np.float64) - k[0, 2]) * z / max(float(k[0, 0]), 1e-8)
    y = (ys.astype(np.float64) - k[1, 2]) * z / max(float(k[1, 1]), 1e-8)
    return np.stack([x, y, z], axis=1)


def _sample_mesh_with_normals(
    mesh: trimesh.Trimesh, n_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    count = min(max(int(n_points), 1000), max(len(mesh.faces), 1000))
    points, face_ids = trimesh.sample.sample_surface(mesh, count)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_ids]
    return np.asarray(points, dtype=np.float64), normals


def _visible_depth_conflict_mask(
    points: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Mark samples that occlude the measured anchor-visible surface.

    A completed object surface that lies *behind* the measured front surface is
    valid and must be retained.  The previous absolute-depth test incorrectly
    removed those back-side samples, reducing the fused candidate to a
    single-view shell.  Only samples in front of the observed depth by more
    than ``tolerance`` are geometrically contradictory.
    """

    points = np.asarray(points, dtype=np.float64)
    z = points[:, 2]
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    proj = (k @ points.T).T
    safe_z = np.where(z > 1e-8, z, 1.0)
    u = np.rint(proj[:, 0] / safe_z).astype(np.int64)
    v = np.rint(proj[:, 1] / safe_z).astype(np.int64)
    h, w = np.asarray(mask).shape[:2]
    inside = (z > 1e-8) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ids = np.flatnonzero(inside)
    conflict = np.zeros(len(points), dtype=bool)
    if ids.size == 0:
        return conflict
    visible = np.asarray(mask).astype(bool)[v[ids], u[ids]]
    observed = np.asarray(depth, dtype=np.float64)[v[ids], u[ids]]
    valid_depth = np.isfinite(observed) & (observed > 1e-5)
    bad = visible & valid_depth & (z[ids] < observed - float(tolerance))
    conflict[ids[bad]] = True
    return conflict


def _supported_anchor_observations(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    ray_points: np.ndarray,
    instant_points: np.ndarray,
    diameter: float,
    voxel: float,
    support_distance: float,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | int]]:
    """Return measured depth samples supported by either geometry hypothesis.

    HO3D object masks can contain invalid/background depth at the object
    boundary.  Feeding those points directly to Poisson can stretch a
    decimetre-scale object into a metre-scale surface.  We first reject depth
    outside the robust joint mesh z-range and then require 3-D support from at
    least one aligned candidate.  The returned pixel mask is also used by the
    visibility-conflict test, so unsupported background depth cannot delete
    valid model surfaces.
    """

    ray_points = np.asarray(ray_points, dtype=np.float64)
    instant_points = np.asarray(instant_points, dtype=np.float64)
    joint = np.vstack([ray_points, instant_points])
    joint = joint[np.isfinite(joint).all(axis=1)]
    if len(joint) < 50:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(np.asarray(mask).shape[:2], dtype=bool),
            {"observed_raw": 0, "observed_z_supported": 0, "observed_mesh_supported": 0},
        )

    z_lo, z_hi = np.quantile(joint[:, 2], [0.005, 0.995])
    z_margin = max(0.10 * float(diameter), 3.0 * float(voxel))
    depth_np = np.asarray(depth, dtype=np.float64)
    valid = (
        np.asarray(mask).astype(bool)
        & np.isfinite(depth_np)
        & (depth_np > 1e-5)
    )
    observed_raw = int(valid.sum())
    valid &= (depth_np >= z_lo - z_margin) & (depth_np <= z_hi + z_margin)
    observed_z_supported = int(valid.sum())

    ys, xs = np.where(valid)
    if len(xs) == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(np.asarray(mask).shape[:2], dtype=bool),
            {
                "observed_raw": observed_raw,
                "observed_z_supported": observed_z_supported,
                "observed_mesh_supported": 0,
            },
        )
    if len(xs) > max_points:
        ids = np.linspace(0, len(xs) - 1, max_points, dtype=np.int64)
        ys, xs = ys[ids], xs[ids]

    z = depth_np[ys, xs]
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    x = (xs.astype(np.float64) - k[0, 2]) * z / max(float(k[0, 0]), 1e-8)
    y = (ys.astype(np.float64) - k[1, 2]) * z / max(float(k[1, 1]), 1e-8)
    points = np.stack([x, y, z], axis=1)

    ray_distance = cKDTree(ray_points).query(points, k=1, workers=-1)[0]
    inst_distance = cKDTree(instant_points).query(points, k=1, workers=-1)[0]
    mesh_distance = np.minimum(ray_distance, inst_distance)
    cutoff = max(float(support_distance), 3.0 * float(voxel))
    keep = mesh_distance <= cutoff
    supported_mask = np.zeros(np.asarray(mask).shape[:2], dtype=bool)
    supported_mask[ys[keep], xs[keep]] = True
    diagnostics = {
        "observed_raw": observed_raw,
        "observed_z_supported": observed_z_supported,
        "observed_mesh_supported": int(keep.sum()),
        "observed_support_cutoff": float(cutoff),
    }
    return points[keep], supported_mask, diagnostics


def _orient_mesh_sample_normals(
    points: np.ndarray, normals: np.ndarray, center: np.ndarray
) -> np.ndarray:
    """Normalize sampled face normals and orient gross flips outwards."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64).copy()
    center = np.asarray(center, dtype=np.float64).reshape(1, 3)
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
    normals[~valid] = points[~valid] - center
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-8)
    inward = np.einsum("ij,ij->i", normals, points - center) < 0.0
    normals[inward] *= -1.0
    return normals


def _estimate_observed_normals(points: np.ndarray, voxel: float) -> np.ndarray:
    """Estimate front-surface normals and orient them towards the camera."""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(3.0 * float(voxel), 0.01), max_nn=40
        )
    )
    pcd.orient_normals_towards_camera_location(np.zeros(3, dtype=np.float64))
    return np.asarray(pcd.normals, dtype=np.float64)


def _remove_small_mesh_components(
    mesh: o3d.geometry.TriangleMesh, min_area_ratio: float
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    triangle_clusters, _, cluster_areas = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters, dtype=np.int64)
    cluster_areas = np.asarray(cluster_areas, dtype=np.float64)
    if cluster_areas.size:
        cutoff = float(cluster_areas.max()) * max(float(min_area_ratio), 0.0)
        remove = cluster_areas[triangle_clusters] < cutoff
        if np.any(remove) and not np.all(remove):
            mesh.remove_triangles_by_mask(remove)
            mesh.remove_unreferenced_vertices()
    return mesh


def _project_points_to_anchor(
    points: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points and return anchor support information."""

    pts = np.asarray(points, dtype=np.float64)
    z = pts[:, 2]
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    proj = (k @ pts.T).T
    safe_z = np.where(z > 1e-8, z, 1.0)
    u = np.rint(proj[:, 0] / safe_z).astype(np.int64)
    v = np.rint(proj[:, 1] / safe_z).astype(np.int64)
    h, w = np.asarray(mask).shape[:2]
    inside = (z > 1e-8) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    observed_depth = np.full(len(pts), np.nan, dtype=np.float64)
    observed_mask = np.zeros(len(pts), dtype=bool)
    ids = np.flatnonzero(inside)
    if ids.size:
        observed_depth[ids] = np.asarray(depth, dtype=np.float64)[v[ids], u[ids]]
        observed_mask[ids] = np.asarray(mask).astype(bool)[v[ids], u[ids]]
    valid_depth = np.isfinite(observed_depth) & (observed_depth > 1e-5)
    return u, v, inside, observed_mask, np.where(valid_depth, observed_depth, np.nan)


def _trimesh_to_clean_o3d(
    mesh: trimesh.Trimesh,
    target_faces: int,
    min_component_area_ratio: float,
) -> trimesh.Trimesh:
    """Run topology cleanup without reconstructing or smoothing the surface."""

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices, dtype=np.float64)
    )
    mesh_o3d.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.faces, dtype=np.int32)
    )
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()
    mesh_o3d.remove_unreferenced_vertices()
    mesh_o3d = _remove_small_mesh_components(
        mesh_o3d, min_component_area_ratio
    )
    if target_faces > 0 and len(mesh_o3d.triangles) > target_faces:
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_faces)
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_unreferenced_vertices()
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float64)
    faces = np.asarray(mesh_o3d.triangles, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise RuntimeError("visible-patch Phi cleanup produced an empty mesh")
    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        trimesh.repair.fix_normals(result, multibody=True)
    except Exception:
        pass
    return result


def _zbuffer_visible_vertices(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    inside: np.ndarray,
    in_mask: np.ndarray,
    observed_depth: np.ndarray,
    image_shape: Tuple[int, int],
    front_slack: float,
) -> np.ndarray:
    """Return vertices on the front-most mesh layer at each anchor pixel.

    The earlier visible-patch implementation accepted every vertex close to
    the measured depth.  Thin or concave objects can project front and back
    surfaces to the same pixel, causing both layers to move and producing an
    inflated/doubled fused surface.  This lightweight z-buffer keeps only the
    closest candidate layer while still using RGB-D as the visibility target.
    """

    h, w = int(image_shape[0]), int(image_shape[1])
    valid = (
        np.asarray(inside, dtype=bool)
        & np.asarray(in_mask, dtype=bool)
        & np.isfinite(z)
        & (np.asarray(z) > 1e-8)
        & np.isfinite(observed_depth)
        & (np.asarray(observed_depth) > 1e-8)
    )
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        return np.zeros(len(z), dtype=bool)
    pixels = np.asarray(v[ids], dtype=np.int64) * w + np.asarray(
        u[ids], dtype=np.int64
    )
    z_buffer = np.full(h * w, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, pixels, np.asarray(z, dtype=np.float64)[ids])
    front = np.zeros(len(z), dtype=bool)
    front[ids] = np.asarray(z, dtype=np.float64)[ids] <= (
        z_buffer[pixels] + max(float(front_slack), 1e-6)
    )
    return valid & front


def _smooth_bounded_displacements(
    mesh: trimesh.Trimesh,
    displacements: np.ndarray,
    active: np.ndarray,
    max_distance: float,
    iterations: int,
) -> np.ndarray:
    """Taper visible-surface updates over the prior topology.

    Only active vertices can move.  Inactive neighbours contribute a zero
    displacement, which naturally fades the update at the visible/unobserved
    boundary without changing connectivity or creating a seam.
    """

    disp = np.asarray(displacements, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    disp[~active] = 0.0
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) and int(iterations) > 0:
        for _ in range(int(iterations)):
            sums = np.zeros_like(disp)
            counts = np.zeros(len(disp), dtype=np.float64)
            np.add.at(sums, edges[:, 0], disp[edges[:, 1]])
            np.add.at(sums, edges[:, 1], disp[edges[:, 0]])
            np.add.at(counts, edges[:, 0], 1.0)
            np.add.at(counts, edges[:, 1], 1.0)
            neighbour = sums / np.maximum(counts[:, None], 1.0)
            disp[active] = 0.80 * disp[active] + 0.20 * neighbour[active]
            disp[~active] = 0.0
    lengths = np.linalg.norm(disp, axis=1)
    too_far = lengths > max(float(max_distance), 1e-8)
    if np.any(too_far):
        disp[too_far] *= (
            float(max_distance) / np.maximum(lengths[too_far], 1e-12)
        )[:, None]
    return disp


def _fuse_meshes_visible_patch(
    mesh_ray: trimesh.Trimesh,
    mesh_instant: trimesh.Trimesh,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    cfg: ReconstructionConfig,
) -> trimesh.Trimesh:
    """RGB-D-constrained topology-preserving fusion operator ``Phi``.

    InstantMesh supplies the complete prior topology.  At vertices that are
    visible in the anchor and supported by measured RGB-D, the prior surface is
    smoothly displaced towards the nearest supported RaySt3R surface.  The
    unobserved/back surface and the face connectivity are left unchanged.

    This is intentionally different from both global Poisson reconstruction
    and the old face-delete/mesh-concatenate implementation: the former erased
    thin completion details, while the latter produced duplicate surfaces and
    disconnected seams that were particularly harmful to FoundationPose.
    """

    # A failed fusion is never admitted to the candidate registry.  Returning
    # the prior as the diagnostic fallback avoids recreating the old doubled
    # concat mesh even if a caller accidentally exports the failed output.
    fallback = mesh_instant.copy()
    diagnostics: Dict[str, float | int | str | bool] = {
        "method": "rgbd_scale_corrected_topology_phi_v7_ray_frame",
        "fallback_used": False,
        "coordinate_frame": "rayst3r_anchor_camera",
        "prior_backbone_preserved": True,
        "global_poisson_used": False,
        "surface_update": "bounded_ray_envelope_scale_then_zbuffered_rgbd_patch_deformation",
    }
    try:
        n_samples = max(int(cfg.fusion_sample_points), 1000)
        ray_pts, _ = _sample_mesh_with_normals(mesh_ray, n_samples)
        inst_pts, _ = _sample_mesh_with_normals(mesh_instant, n_samples)

        # InstantMesh supplies the complete topology, but its unseen extent can
        # remain systematically too large or too small even after the two
        # reconstructions have independently recovered metric scale.  Estimate
        # a *single bounded uniform correction* from the robust RaySt3R and
        # InstantMesh envelopes.  This uses only the two anchor reconstructions
        # (no CAD/GT geometry), preserves aspect ratio and connectivity, and is
        # deliberately capped so that RaySt3R cannot erase the completion
        # prior through an unreliable anisotropic warp.
        ray_lo, ray_hi = np.percentile(ray_pts, [2.0, 98.0], axis=0)
        inst_lo, inst_hi = np.percentile(inst_pts, [2.0, 98.0], axis=0)
        ray_extent = np.maximum(ray_hi - ray_lo, 1e-8)
        inst_extent = np.maximum(inst_hi - inst_lo, 1e-8)
        valid_axes = inst_extent > 0.05 * float(np.max(inst_extent))
        axis_ratios = ray_extent[valid_axes] / inst_extent[valid_axes]
        robust_diagonal_ratio = float(
            np.linalg.norm(ray_extent) / max(np.linalg.norm(inst_extent), 1e-8)
        )
        envelope_ratio = float(
            0.5 * np.median(axis_ratios) + 0.5 * robust_diagonal_ratio
        )
        global_scale = float(
            np.clip(
                envelope_ratio,
                float(cfg.fusion_global_scale_min),
                float(cfg.fusion_global_scale_max),
            )
        )
        prior_scale_center = np.median(
            np.asarray(mesh_instant.vertices, dtype=np.float64), axis=0
        )
        mesh_prior = mesh_instant.copy()
        mesh_prior.vertices = prior_scale_center + global_scale * (
            np.asarray(mesh_instant.vertices, dtype=np.float64)
            - prior_scale_center
        )
        inst_pts, _ = _sample_mesh_with_normals(mesh_prior, n_samples)
        diagnostics.update(
            {
                "prior_global_scale_correction": global_scale,
                "prior_global_scale_estimate": envelope_ratio,
                "prior_global_scale_bounds": [
                    float(cfg.fusion_global_scale_min),
                    float(cfg.fusion_global_scale_max),
                ],
                "prior_global_scale_source": "robust_cross_reconstruction_envelope_no_gt",
            }
        )
        combined_extent = np.ptp(np.vstack([ray_pts, inst_pts]), axis=0)
        diameter = max(float(np.linalg.norm(combined_extent)), 1e-6)
        voxel = max(float(cfg.fusion_voxel_size), 1e-4)
        support_distance = max(
            float(cfg.fusion_observed_support_ratio) * diameter, 2.0 * voxel
        )
        agreement_distance = max(
            float(cfg.fusion_agreement_ratio) * diameter, 2.0 * voxel
        )
        obs_pts, supported_depth_mask, observation_diag = (
            _supported_anchor_observations(
                depth,
                mask,
                intrinsic,
                ray_pts,
                inst_pts,
                diameter,
                voxel,
                support_distance,
                max_points=min(n_samples, 30000),
            )
        )
        diagnostics.update(observation_diag)
        if len(obs_pts) < 50:
            raise RuntimeError("too few candidate-supported anchor points for patch Phi")

        ray_vertices = np.asarray(mesh_ray.vertices, dtype=np.float64)
        _, _, ray_inside, ray_in_mask, ray_observed_depth = _project_points_to_anchor(
            ray_vertices, depth, supported_depth_mask, intrinsic
        )
        ray_valid_depth = np.isfinite(ray_observed_depth)
        depth_tolerance = max(
            float(cfg.fusion_visible_depth_tolerance), 2.0 * voxel
        )
        ray_depth_supported = (
            ray_inside
            & ray_in_mask
            & ray_valid_depth
            & (np.abs(ray_vertices[:, 2] - ray_observed_depth) <= depth_tolerance)
        )
        ray_observation_distance = cKDTree(obs_pts).query(
            ray_vertices, k=1, workers=-1
        )[0]
        ray_supported = (
            ray_depth_supported | (ray_observation_distance <= support_distance)
        )
        ray_conflict = _visible_depth_conflict_mask(
            ray_vertices,
            depth,
            supported_depth_mask,
            intrinsic,
            cfg.fusion_visible_depth_tolerance,
        )
        ray_supported &= ~ray_conflict
        supported_ray_vertices = ray_vertices[ray_supported]
        if len(supported_ray_vertices) < 50:
            raise RuntimeError("RaySt3R has too few RGB-D-supported surface vertices")

        prior_vertices = np.asarray(mesh_prior.vertices, dtype=np.float64)
        prior_u, prior_v, prior_inside, prior_in_mask, prior_observed_depth = (
            _project_points_to_anchor(
                prior_vertices, depth, supported_depth_mask, intrinsic
            )
        )
        prior_valid_depth = np.isfinite(prior_observed_depth)
        prior_depth_residual = np.abs(prior_vertices[:, 2] - prior_observed_depth)
        prior_front = _zbuffer_visible_vertices(
            prior_u,
            prior_v,
            prior_vertices[:, 2],
            prior_inside,
            prior_in_mask,
            prior_observed_depth,
            np.asarray(mask).shape[:2],
            front_slack=max(voxel, 0.25 * depth_tolerance),
        )
        prior_visible = (
            prior_front
            & prior_valid_depth
            & (prior_depth_residual <= max(depth_tolerance, agreement_distance))
        )
        ray_tree = cKDTree(supported_ray_vertices)
        prior_to_ray_distance, prior_to_ray_index = ray_tree.query(
            prior_vertices, k=1, workers=-1
        )
        prior_normals = np.asarray(mesh_prior.vertex_normals, dtype=np.float64)
        ray_normals_all = np.asarray(mesh_ray.vertex_normals, dtype=np.float64)
        supported_ray_normals = ray_normals_all[ray_supported]
        normal_cosine = np.abs(
            np.einsum(
                "ij,ij->i",
                prior_normals,
                supported_ray_normals[prior_to_ray_index],
            )
        )
        min_normal_cosine = float(np.clip(cfg.fusion_min_normal_cosine, 0.0, 0.95))
        normal_confidence = np.clip(
            (normal_cosine - min_normal_cosine)
            / max(1.0 - min_normal_cosine, 1e-8),
            0.0,
            1.0,
        )
        update_radius = max(1.5 * agreement_distance, 2.0 * voxel)
        update_mask = (
            prior_visible
            & (prior_to_ray_distance <= update_radius)
            & (normal_cosine >= min_normal_cosine)
        )
        if int(update_mask.sum()) < 25:
            raise RuntimeError("too few InstantMesh vertices have reliable visible support")

        # A bounded confidence blend avoids copying isolated RaySt3R outliers.
        # Confidence decays with both cross-model distance and disagreement
        # with the measured depth.  Connectivity never changes, so no seam or
        # duplicate surface is introduced.
        distance_confidence = np.exp(
            -prior_to_ray_distance / max(agreement_distance, 1e-8)
        )
        depth_confidence = np.exp(
            -np.nan_to_num(prior_depth_residual, nan=update_radius)
            / max(depth_tolerance, 1e-8)
        )
        blend = np.clip(
            distance_confidence * depth_confidence * normal_confidence,
            0.0,
            1.0,
        )
        # Supported vertices must inherit a meaningful portion of the
        # RaySt3R surface; an overly conservative blend made Phi practically
        # identical to InstantMesh and therefore unable to contribute a real
        # third hypothesis.  Z-buffer, normal agreement and the displacement
        # cap still bound every update.
        blend = np.where(update_mask, 0.95 * (0.40 + 0.60 * blend), 0.0)

        # Use the RaySt3R surface for local detail while anchoring every update
        # to the measured RGB-D ray at that pixel.  This prevents an ICP-local
        # optimum from dragging the visible patch away from the observation.
        ray_targets = supported_ray_vertices[prior_to_ray_index]
        observed_targets = ray_targets.copy()
        obs_ids = np.flatnonzero(prior_valid_depth & prior_inside & prior_in_mask)
        if obs_ids.size:
            observed_z = prior_observed_depth[obs_ids]
            fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
            cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
            observed_targets[obs_ids, 0] = (
                (prior_u[obs_ids].astype(np.float64) - cx)
                * observed_z
                / max(fx, 1e-8)
            )
            observed_targets[obs_ids, 1] = (
                (prior_v[obs_ids].astype(np.float64) - cy)
                * observed_z
                / max(fy, 1e-8)
            )
            observed_targets[obs_ids, 2] = observed_z
        target_vertices = 0.75 * ray_targets + 0.25 * observed_targets
        raw_displacements = blend[:, None] * (target_vertices - prior_vertices)
        max_update_distance = max(
            float(cfg.fusion_max_update_ratio) * diameter,
            2.0 * voxel,
        )
        displacements = _smooth_bounded_displacements(
            mesh_prior,
            raw_displacements,
            update_mask,
            max_update_distance,
            cfg.fusion_smooth_iterations,
        )
        fused_vertices = prior_vertices + displacements
        fused = mesh_prior.copy()
        fused.vertices = fused_vertices
        if not isinstance(fused, trimesh.Trimesh) or len(fused.faces) < 50:
            raise RuntimeError("topology-preserving Phi produced an empty mesh")

        fused_count = min(15000, max(len(fused.faces), 2000))
        fused_pts, _ = trimesh.sample.sample_surface(fused, fused_count)
        fused_tree = cKDTree(fused_pts)
        coverage_distance = max(agreement_distance, 2.5 * voxel)
        prior_coverage = float(
            np.mean(
                fused_tree.query(inst_pts, k=1, workers=-1)[0]
                <= coverage_distance
            )
        )
        prior_tree = cKDTree(inst_pts)
        prior_observed_coverage = float(
            np.mean(
                prior_tree.query(obs_pts, k=1, workers=-1)[0]
                <= coverage_distance
            )
        )
        observed_coverage = float(
            np.mean(
                fused_tree.query(obs_pts, k=1, workers=-1)[0]
                <= coverage_distance
            )
        )
        visible_ray_coverage = float(
            np.mean(
                fused_tree.query(
                    supported_ray_vertices,
                    k=1,
                    workers=-1,
                )[0]
                <= coverage_distance
            )
        )
        source_diagonal = max(
            float(np.linalg.norm(np.ptp(ray_pts, axis=0))),
            float(np.linalg.norm(np.ptp(inst_pts, axis=0))),
            1e-8,
        )
        fused_diagonal = float(np.linalg.norm(np.asarray(fused.extents)))
        extent_ratio = fused_diagonal / source_diagonal
        source_center = np.median(inst_pts, axis=0)
        fused_center = 0.5 * (
            np.asarray(fused.bounds[0]) + np.asarray(fused.bounds[1])
        )
        center_offset_ratio = float(
            np.linalg.norm(fused_center - source_center) / source_diagonal
        )
        # Validate relative improvement/non-degradation as well as absolute
        # bounds.  A fixed 50% observation threshold unfairly rejected useful
        # partial fusions when the completion prior itself covered less than
        # 50%; conversely, merely retaining the prior is not sufficient.
        required_observed_coverage = min(
            float(cfg.fusion_min_observed_coverage),
            max(0.20, 0.95 * prior_observed_coverage),
        )
        depth_before = float(np.median(prior_depth_residual[update_mask]))
        fused_u, fused_v, fused_inside, fused_in_mask, fused_observed_depth = (
            _project_points_to_anchor(
                fused_vertices, depth, supported_depth_mask, intrinsic
            )
        )
        fused_depth_valid = (
            fused_inside
            & fused_in_mask
            & np.isfinite(fused_observed_depth)
            & update_mask
        )
        depth_after = (
            float(
                np.median(
                    np.abs(
                        fused_vertices[fused_depth_valid, 2]
                        - fused_observed_depth[fused_depth_valid]
                    )
                )
            )
            if np.any(fused_depth_valid)
            else float("inf")
        )
        rgbd_non_degrading = bool(
            np.isfinite(depth_after)
            and depth_after <= max(1.05 * depth_before, depth_before + voxel)
        )
        output_valid = bool(
            prior_coverage >= float(cfg.fusion_min_prior_coverage)
            and observed_coverage >= required_observed_coverage
            and observed_coverage + 0.02 >= prior_observed_coverage
            and rgbd_non_degrading
            and float(cfg.fusion_min_extent_ratio)
            <= extent_ratio
            <= float(cfg.fusion_max_extent_ratio)
            and center_offset_ratio <= float(cfg.fusion_max_center_offset_ratio)
        )
        if not output_valid:
            raise RuntimeError(
                "patch Phi failed bounded validation: "
                f"prior={prior_coverage:.3f}, observed={observed_coverage:.3f}, "
                f"required_observed={required_observed_coverage:.3f}, "
                f"depth_before={depth_before:.5f}, depth_after={depth_after:.5f}, "
                f"extent={extent_ratio:.3f}, center={center_offset_ratio:.3f}"
            )

        ray_to_inst = cKDTree(inst_pts).query(ray_pts, k=1, workers=-1)[0]
        inst_to_ray = cKDTree(ray_pts).query(inst_pts, k=1, workers=-1)[0]
        symmetric_distance = float(
            0.5 * (ray_to_inst.mean() + inst_to_ray.mean())
        )
        overlap = float(
            0.5
            * (
                np.mean(ray_to_inst <= agreement_distance)
                + np.mean(inst_to_ray <= agreement_distance)
            )
        )
        conflict_ratio = float(np.mean(ray_conflict[ray_inside & ray_in_mask])) if np.any(
            ray_inside & ray_in_mask
        ) else 1.0
        visible_distances = prior_to_ray_distance[prior_visible]
        visible_distance = float(np.median(visible_distances))
        distance_quality = float(
            np.exp(-visible_distance / max(agreement_distance, 1e-8))
        )
        rgbd_quality = float(
            np.exp(-depth_after / max(depth_tolerance, 1e-8))
        )
        rgbd_gain = float(
            np.clip(
                (depth_before - depth_after) / max(depth_tolerance, 1e-8),
                -1.0,
                1.0,
            )
        )
        gain_quality = float(0.5 + 0.5 * np.tanh(2.0 * rgbd_gain))
        coverage_quality = (
            0.35 * prior_coverage
            + 0.40 * observed_coverage
            + 0.25 * visible_ray_coverage
        )
        shape_quality = float(
            np.exp(-abs(np.log(max(extent_ratio, 1e-8))) / 0.35)
            * np.exp(-center_offset_ratio / 0.15)
        )
        reliability = float(
            np.clip(
                (
                    0.25 * distance_quality
                    + 0.35 * rgbd_quality
                    + 0.15 * gain_quality
                    + 0.25 * coverage_quality
                )
                * (1.0 - 0.5 * conflict_ratio)
                * shape_quality,
                0.0,
                1.0,
            )
        )
        diagnostics.update(
            {
                "output_prior_coverage": prior_coverage,
                "input_prior_observed_coverage": prior_observed_coverage,
                "output_observed_coverage": observed_coverage,
                "required_observed_coverage": required_observed_coverage,
                "output_visible_ray_coverage": visible_ray_coverage,
                "output_coverage_distance": float(coverage_distance),
                "output_extent_ratio": float(extent_ratio),
                "output_center_offset_ratio": center_offset_ratio,
                "output_valid": True,
                "ray_samples": int(len(ray_pts)),
                "instantmesh_samples": int(len(inst_pts)),
                "observed_samples": int(len(obs_pts)),
                "supported_ray_vertices": int(len(supported_ray_vertices)),
                "prior_vertices_visible": int(prior_visible.sum()),
                "prior_vertices_updated": int(update_mask.sum()),
                "prior_vertex_update_ratio": float(np.mean(update_mask)),
                "mean_update_distance": float(
                    np.mean(np.linalg.norm(displacements[update_mask], axis=1))
                ),
                "max_update_distance": float(
                    np.max(np.linalg.norm(displacements[update_mask], axis=1))
                ),
                "update_distance_cap": float(max_update_distance),
                "normal_cosine_median": float(np.median(normal_cosine[update_mask])),
                "visible_depth_before": depth_before,
                "visible_depth_after": depth_after,
                "visible_depth_gain": float(depth_before - depth_after),
                "rgbd_non_degrading": rgbd_non_degrading,
                "prior_faces_retained": int(len(mesh_prior.faces)),
                "prior_faces_removed": 0,
                "prior_backbone_topology_decimated": False,
                "topology_preserved": True,
                "output_faces": int(len(fused.faces)),
                "symmetric_surface_distance": symmetric_distance,
                "surface_overlap": overlap,
                "visible_conflict_ratio": conflict_ratio,
                "reliability": reliability,
            }
        )
    except Exception as exc:
        fused = fallback
        diagnostics.update(
            {
                "method": "prior_fallback_not_candidate",
                "fallback_used": True,
                "output_valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "reliability": 0.0,
            }
        )
    fused.metadata = dict(getattr(fused, "metadata", {}) or {})
    fused.metadata["fusion_diagnostics"] = diagnostics
    return fused


def _fuse_meshes(
    mesh_ray: trimesh.Trimesh,
    mesh_instant: trimesh.Trimesh,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    cfg: ReconstructionConfig,
) -> trimesh.Trimesh:
    """Confidence-aware implementation of the paper's fusion operator Phi.

    RaySt3R supplies the anchor-visible RGB-D-conditioned surface. InstantMesh
    supplies completion away from measured depth. Samples that contradict the
    anchor depth are removed, overlapping surfaces are de-duplicated, and a
    single Poisson surface is reconstructed on CPU. Point and face caps keep
    the operation practical on a 24-GB RTX 3090 host.
    """

    if str(cfg.fusion_mode) == "visible_patch":
        return _fuse_meshes_visible_patch(
            mesh_ray, mesh_instant, depth, mask, intrinsic, cfg
        )
    if str(cfg.fusion_mode) != "poisson":
        raise ValueError(f"Unsupported fusion_mode: {cfg.fusion_mode}")

    fallback = _concatenate_meshes(mesh_ray, mesh_instant)
    diagnostics: Dict[str, float | int | str | bool] = {
        "method": "visible_prior_poisson_v3_ray_frame",
        "fallback_used": False,
        "coordinate_frame": "rayst3r_anchor_camera",
    }
    try:
        n_samples = max(int(cfg.fusion_sample_points), 1000)
        ray_pts, ray_normals = _sample_mesh_with_normals(mesh_ray, n_samples)
        inst_pts, inst_normals = _sample_mesh_with_normals(mesh_instant, n_samples)
        combined_extent = np.ptp(np.vstack([ray_pts, inst_pts]), axis=0)
        diameter = max(float(np.linalg.norm(combined_extent)), 1e-6)
        voxel = max(float(cfg.fusion_voxel_size), 1e-4)
        support_distance = max(
            float(cfg.fusion_observed_support_ratio) * diameter, 2.0 * voxel
        )
        agreement_distance = max(
            float(cfg.fusion_agreement_ratio) * diameter, 2.0 * voxel
        )

        obs_pts, supported_depth_mask, observation_diag = _supported_anchor_observations(
            depth,
            mask,
            intrinsic,
            ray_pts,
            inst_pts,
            diameter,
            voxel,
            support_distance,
            max_points=min(n_samples, 30000),
        )
        diagnostics.update(observation_diag)
        if len(obs_pts) < 50:
            raise RuntimeError("too few candidate-supported anchor depth points for Phi")

        ray_center = np.asarray(mesh_ray.vertices, dtype=np.float64).mean(axis=0)
        inst_center = np.asarray(mesh_instant.vertices, dtype=np.float64).mean(axis=0)
        ray_normals = _orient_mesh_sample_normals(ray_pts, ray_normals, ray_center)
        inst_normals = _orient_mesh_sample_normals(inst_pts, inst_normals, inst_center)
        obs_normals = _estimate_observed_normals(obs_pts, voxel)

        ray_obs_distance = cKDTree(obs_pts).query(ray_pts, k=1, workers=-1)[0]
        ray_inst_distance = cKDTree(inst_pts).query(ray_pts, k=1, workers=-1)[0]
        ray_conflict = _visible_depth_conflict_mask(
            ray_pts,
            depth,
            supported_depth_mask,
            intrinsic,
            cfg.fusion_visible_depth_tolerance,
        )
        # Retain RGB-D-supported RaySt3R surfaces and mutually supported
        # completion, while rejecting visible-depth contradictions.
        ray_keep = (
            (ray_obs_distance <= support_distance)
            | (ray_inst_distance <= agreement_distance)
        ) & ~ray_conflict
        if ray_keep.sum() < 100:
            ray_keep = ~ray_conflict

        inst_conflict = _visible_depth_conflict_mask(
            inst_pts,
            depth,
            supported_depth_mask,
            intrinsic,
            cfg.fusion_visible_depth_tolerance,
        )
        inst_keep = ~inst_conflict
        retained_ray = ray_pts[ray_keep]
        retained_ray_visible = ray_pts[
            ray_keep & (ray_obs_distance <= support_distance)
        ]
        if len(retained_ray_visible):
            inst_to_ray = cKDTree(retained_ray_visible).query(
                inst_pts, k=1, workers=-1
            )[0]
            # RaySt3R wins only on the measured visible patch.  The aligned
            # InstantMesh back side remains intact as the completion prior.
            inst_keep &= inst_to_ray > max(1.5 * voxel, 0.5 * agreement_distance)

        points = np.vstack([retained_ray, inst_pts[inst_keep], obs_pts])
        normals = np.vstack(
            [
                ray_normals[ray_keep],
                inst_normals[inst_keep],
                obs_normals,
            ]
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd = pcd.voxel_down_sample(voxel)
        if len(pcd.points) < 200:
            raise RuntimeError("too few retained points after Phi filtering")
        if not pcd.has_normals():
            raise RuntimeError("Phi lost source normals during voxel downsampling")
        pcd.normalize_normals()

        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=max(6, int(cfg.fusion_poisson_depth)),
            scale=1.05,
            linear_fit=True,
        )
        densities = np.asarray(densities, dtype=np.float64)
        if densities.size and 0.0 < cfg.fusion_density_quantile < 1.0:
            cutoff = float(np.quantile(densities, cfg.fusion_density_quantile))
            mesh_o3d.remove_vertices_by_mask(densities < cutoff)

        bbox = pcd.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.04, bbox.get_center())
        mesh_o3d = mesh_o3d.crop(bbox)
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_unreferenced_vertices()
        mesh_o3d = _remove_small_mesh_components(
            mesh_o3d, cfg.fusion_min_component_area_ratio
        )
        target_faces = int(cfg.fusion_target_faces)
        if target_faces > 0 and len(mesh_o3d.triangles) > target_faces:
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_faces)
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_unreferenced_vertices()
        mesh_o3d = _remove_small_mesh_components(
            mesh_o3d, cfg.fusion_min_component_area_ratio
        )

        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError("Poisson Phi produced an empty mesh")
        fused = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Validate the reconstructed surface itself.  ICP compatibility alone
        # cannot detect Poisson spillover or collapse, so require bounded
        # scale/centering and explicit coverage of both the completion prior
        # and the measured visible surface.  Failed outputs are marked as
        # fallback and are excluded by anchor_paper.py from the candidate pool.
        fused_count = min(10000, max(len(fused.faces), 1000))
        fused_pts, _ = trimesh.sample.sample_surface(fused, fused_count)
        fused_tree = cKDTree(fused_pts)
        coverage_distance = max(agreement_distance, 2.5 * voxel)
        prior_coverage = float(
            np.mean(fused_tree.query(inst_pts, k=1, workers=-1)[0] <= coverage_distance)
        )
        observed_coverage = float(
            np.mean(fused_tree.query(obs_pts, k=1, workers=-1)[0] <= coverage_distance)
        )
        visible_ray_points = (
            retained_ray_visible if len(retained_ray_visible) else retained_ray
        )
        visible_ray_coverage = float(
            np.mean(
                fused_tree.query(visible_ray_points, k=1, workers=-1)[0]
                <= coverage_distance
            )
        )
        source_diagonal = max(
            float(np.linalg.norm(np.ptp(ray_pts, axis=0))),
            float(np.linalg.norm(np.ptp(inst_pts, axis=0))),
            1e-8,
        )
        fused_diagonal = float(np.linalg.norm(np.asarray(fused.extents)))
        extent_ratio = fused_diagonal / source_diagonal
        source_center = 0.5 * (
            np.median(ray_pts, axis=0) + np.median(inst_pts, axis=0)
        )
        fused_center = 0.5 * (
            np.asarray(fused.bounds[0]) + np.asarray(fused.bounds[1])
        )
        center_offset_ratio = float(
            np.linalg.norm(fused_center - source_center) / source_diagonal
        )
        output_valid = bool(
            prior_coverage >= float(cfg.fusion_min_prior_coverage)
            and observed_coverage >= float(cfg.fusion_min_observed_coverage)
            and float(cfg.fusion_min_extent_ratio)
            <= extent_ratio
            <= float(cfg.fusion_max_extent_ratio)
            and center_offset_ratio <= float(cfg.fusion_max_center_offset_ratio)
        )
        diagnostics.update(
            {
                "output_prior_coverage": prior_coverage,
                "output_observed_coverage": observed_coverage,
                "output_visible_ray_coverage": visible_ray_coverage,
                "output_coverage_distance": float(coverage_distance),
                "output_extent_ratio": float(extent_ratio),
                "output_center_offset_ratio": center_offset_ratio,
                "output_valid": output_valid,
            }
        )
        if not output_valid:
            raise RuntimeError(
                "Phi output failed bounded geometry validation: "
                f"prior_coverage={prior_coverage:.3f}, "
                f"observed_coverage={observed_coverage:.3f}, "
                f"extent_ratio={extent_ratio:.3f}, "
                f"center_offset_ratio={center_offset_ratio:.3f}"
            )

        # Symmetric post-alignment compatibility is diagnostic and contributes
        # continuously to anchor reliability; it is not a hidden hard gate.
        ray_to_inst = cKDTree(inst_pts).query(ray_pts, k=1, workers=-1)[0]
        inst_to_ray = cKDTree(ray_pts).query(inst_pts, k=1, workers=-1)[0]
        symmetric_distance = float(0.5 * (ray_to_inst.mean() + inst_to_ray.mean()))
        overlap = float(
            0.5
            * (
                np.mean(ray_to_inst <= agreement_distance)
                + np.mean(inst_to_ray <= agreement_distance)
            )
        )
        distance_quality = float(np.exp(-symmetric_distance / agreement_distance))
        conflict_ratio = float(
            0.5 * (np.mean(ray_conflict) + np.mean(inst_conflict))
        )
        source_quality = (0.5 * overlap + 0.5 * distance_quality) * (
            1.0 - conflict_ratio
        )
        coverage_quality = (
            0.45 * prior_coverage
            + 0.35 * observed_coverage
            + 0.20 * visible_ray_coverage
        )
        shape_quality = float(
            np.exp(-abs(np.log(max(extent_ratio, 1e-8))) / 0.35)
            * np.exp(-center_offset_ratio / 0.15)
        )
        reliability = float(
            np.clip(source_quality * coverage_quality * shape_quality, 0.0, 1.0)
        )
        diagnostics.update(
            {
                "ray_samples": int(len(ray_pts)),
                "instantmesh_samples": int(len(inst_pts)),
                "observed_samples": int(len(obs_pts)),
                "ray_retained": int(ray_keep.sum()),
                "instantmesh_retained": int(inst_keep.sum()),
                "visible_ray_retained": int(len(retained_ray_visible)),
                "poisson_points": int(len(pcd.points)),
                "symmetric_surface_distance": symmetric_distance,
                "surface_overlap": overlap,
                "visible_conflict_ratio": conflict_ratio,
                "reliability": reliability,
            }
        )
    except Exception as exc:
        fused = fallback
        diagnostics.update(
            {
                "method": "concat_fallback",
                "fallback_used": True,
                "error": f"{type(exc).__name__}: {exc}",
                "reliability": 0.0,
            }
        )
    fused.metadata = dict(getattr(fused, "metadata", {}) or {})
    fused.metadata["fusion_diagnostics"] = diagnostics
    return fused


def _object_specific_cfg(cfg: ReconstructionConfig, obj_name: str) -> Tuple[ReconstructionConfig, Dict[str, float | bool]]:
    if obj_name not in SB_SM_OBJECTS:
        return cfg, {}

    overrides: Dict[str, float | bool] = {
        # Keep more thin structure for bottle-like objects.
        "rayst3r_density_quantile": min(cfg.rayst3r_density_quantile, 0.01),
        "rayst3r_poisson_depth": max(cfg.rayst3r_poisson_depth, 10),
        "rayst3r_voxel_size": min(cfg.rayst3r_voxel_size, 0.0025),
        "rayst3r_std_ratio": max(cfg.rayst3r_std_ratio, 2.2),
    }
    return replace(cfg, **overrides), overrides


def _align_decision_score(
    reg: Optional[o3d.pipelines.registration.RegistrationResult], cfg: ReconstructionConfig
) -> float:
    if reg is None:
        return -1e9
    fitness = float(reg.fitness)
    rmse = float(reg.inlier_rmse)
    score = fitness - cfg.align_score_rmse_weight * rmse
    if fitness < cfg.icp_fitness_threshold:
        score -= cfg.align_low_fitness_penalty
    if rmse > cfg.align_rmse_penalty_threshold:
        score -= cfg.align_high_rmse_penalty
    return score


def _is_valid_reg(reg: Optional[o3d.pipelines.registration.RegistrationResult]) -> bool:
    if reg is None:
        return False
    try:
        fitness = float(reg.fitness)
        rmse = float(reg.inlier_rmse)
    except Exception:
        return False
    if not np.isfinite(fitness) or not np.isfinite(rmse):
        return False
    if fitness <= 0.0:
        return False
    return True


def _bidirectional_diagnostics(
    reg_inst_to_ray: Optional[o3d.pipelines.registration.RegistrationResult],
    reg_ray_to_inst: Optional[o3d.pipelines.registration.RegistrationResult],
    mesh_ray: trimesh.Trimesh,
    mesh_inst: trimesh.Trimesh,
    cfg: ReconstructionConfig,
) -> Dict[str, float | bool | str]:
    """Report strict-both validity and transform cycle consistency.

    The translation cycle error is normalized by the larger candidate bbox
    diagonal. These diagnostics never affect best-of-two selection unless the
    explicit ``align_strict_both`` ablation is enabled.
    """

    def _values(reg):
        if reg is None:
            return -1.0, float("inf")
        return float(reg.fitness), float(reg.inlier_rmse)

    inst_fitness, inst_rmse = _values(reg_inst_to_ray)
    ray_fitness, ray_rmse = _values(reg_ray_to_inst)
    inst_pass = (
        np.isfinite(inst_fitness)
        and np.isfinite(inst_rmse)
        and inst_fitness >= cfg.icp_fitness_threshold
        and inst_rmse <= cfg.align_rmse_penalty_threshold
    )
    ray_pass = (
        np.isfinite(ray_fitness)
        and np.isfinite(ray_rmse)
        and ray_fitness >= cfg.icp_fitness_threshold
        and ray_rmse <= cfg.align_rmse_penalty_threshold
    )
    rotation_deg = float("nan")
    translation_norm = float("nan")
    raw_translation_norm = float("nan")
    if reg_inst_to_ray is not None and reg_ray_to_inst is not None:
        transform_inst_to_ray = np.asarray(
            reg_inst_to_ray.transformation, dtype=np.float64
        )
        transform_ray_to_inst = np.asarray(
            reg_ray_to_inst.transformation, dtype=np.float64
        )
        cycle_ray = (
            transform_inst_to_ray @ transform_ray_to_inst
        )
        cycle_inst = transform_ray_to_inst @ transform_inst_to_ray
        cos_angle = np.clip(
            (np.trace(cycle_ray[:3, :3]) - 1.0) * 0.5, -1.0, 1.0
        )
        rotation_deg = float(np.degrees(np.arccos(cos_angle)))
        diameter = max(
            float(np.linalg.norm(np.asarray(mesh_ray.extents, dtype=np.float64))),
            float(np.linalg.norm(np.asarray(mesh_inst.extents, dtype=np.float64))),
            1e-8,
        )
        # Both reconstructed surfaces currently live in anchor-camera
        # coordinates, not object-centred coordinates.  The raw translation
        # column of a non-identity cycle therefore depends on the arbitrary
        # camera origin.  Measure the actual round-trip displacement of each
        # surface centre and average the two directions instead.
        ray_center = np.asarray(mesh_ray.vertices, dtype=np.float64).mean(axis=0)
        inst_center = np.asarray(mesh_inst.vertices, dtype=np.float64).mean(axis=0)
        ray_displacement = (
            cycle_ray[:3, :3] @ ray_center + cycle_ray[:3, 3] - ray_center
        )
        inst_displacement = (
            cycle_inst[:3, :3] @ inst_center + cycle_inst[:3, 3] - inst_center
        )
        translation_norm = float(
            0.5
            * (np.linalg.norm(ray_displacement) + np.linalg.norm(inst_displacement))
            / diameter
        )
        raw_translation_norm = float(
            0.5
            * (
                np.linalg.norm(cycle_ray[:3, 3])
                + np.linalg.norm(cycle_inst[:3, 3])
            )
            / diameter
        )
    return {
        "inst_to_ray_fitness": inst_fitness,
        "inst_to_ray_rmse": inst_rmse,
        "inst_to_ray_pass": bool(inst_pass),
        "ray_to_inst_fitness": ray_fitness,
        "ray_to_inst_rmse": ray_rmse,
        "ray_to_inst_pass": bool(ray_pass),
        "strict_both_pass": bool(inst_pass and ray_pass),
        "cycle_rotation_deg": rotation_deg,
        "cycle_translation_normalized": translation_norm,
        "cycle_translation_raw_column_normalized": raw_translation_norm,
        "cycle_translation_definition": (
            "mean_bidirectional_surface_center_round_trip_displacement_over_diameter"
        ),
    }


def _reciprocal_cycle_diagnostics(
    independent_inst_to_ray: Optional[o3d.pipelines.registration.RegistrationResult],
    independent_ray_to_inst: Optional[o3d.pipelines.registration.RegistrationResult],
    paired_inst_to_ray: Optional[o3d.pipelines.registration.RegistrationResult],
    paired_ray_to_inst: Optional[o3d.pipelines.registration.RegistrationResult],
    mesh_ray: trimesh.Trimesh,
    mesh_inst: trimesh.Trimesh,
    cfg: ReconstructionConfig,
    reference_direction: str,
    hypothesis_index: Optional[int],
) -> Dict[str, float | bool | str]:
    """Measure a round trip without changing the main best-of-two search.

    The independently optimized directional results define both the disclosed
    best-of-two decision and the strict-both ablation.  For cycle reporting only,
    the selected transform is inverted and refined once in the reverse direction.
    This produces a true round trip for the selected estimate while preventing a
    reviewer diagnostic from silently changing either directional ICP optimum.
    No cycle term is used to select a direction or admit a fused candidate.
    """

    independent = _bidirectional_diagnostics(
        independent_inst_to_ray,
        independent_ray_to_inst,
        mesh_ray,
        mesh_inst,
        cfg,
    )
    diagnostics: Dict[str, float | bool | str] = dict(independent)
    diagnostics.update(
        {f"independent_{key}": value for key, value in independent.items()}
    )
    diagnostics.update(
        {
            "cycle_pairing_mode": "selected_transform_inverse_refinement",
            "cycle_reference_direction": str(reference_direction),
            "cycle_hypothesis_index": (
                int(hypothesis_index) if hypothesis_index is not None else -1
            ),
            "reciprocal_initialization_pairing": True,
        }
    )
    reciprocal = _bidirectional_diagnostics(
        paired_inst_to_ray, paired_ray_to_inst, mesh_ray, mesh_inst, cfg
    )
    # Keep the main directional pass/fail fields and strict-both result tied to
    # the independently optimized searches.  Only the cycle values come from
    # the selected-transform round trip.
    diagnostics["cycle_rotation_deg"] = reciprocal["cycle_rotation_deg"]
    diagnostics["cycle_translation_normalized"] = reciprocal[
        "cycle_translation_normalized"
    ]
    diagnostics["cycle_translation_raw_column_normalized"] = reciprocal[
        "cycle_translation_raw_column_normalized"
    ]
    diagnostics["cycle_translation_definition"] = reciprocal[
        "cycle_translation_definition"
    ]
    diagnostics.update(
        {f"reciprocal_{key}": value for key, value in reciprocal.items()}
    )
    diagnostics["reciprocal_reverse_valid"] = bool(
        _is_valid_reg(paired_ray_to_inst)
        if reference_direction == "inst_to_ray"
        else _is_valid_reg(paired_inst_to_ray)
    )
    return diagnostics


def run_construction(cfg: ReconstructionConfig) -> Dict[str, Dict[str, str]]:
    anchor_root = os.path.realpath(cfg.anchor_root)
    output_root = os.path.realpath(cfg.output_root or cfg.anchor_root)

    checkpoint_path = cfg.rayst3r_checkpoint
    if checkpoint_path:
        checkpoint_path = os.path.realpath(checkpoint_path)
    runner = Rayst3RRunner(checkpoint_path=checkpoint_path, device=cfg.rayst3r_device)

    results: Dict[str, Dict[str, str]] = {}
    obj_list = sorted(
        d for d in os.listdir(anchor_root) if os.path.isdir(os.path.join(anchor_root, d))
    )
    if cfg.object_names:
        requested = set(cfg.object_names)
        obj_list = [obj_name for obj_name in obj_list if obj_name in requested]

    for obj_name in obj_list:
        object_seed = _stable_object_seed(cfg.seed, obj_name)
        _set_object_seed(object_seed)
        obj_dir = os.path.join(anchor_root, obj_name)
        out_dir = os.path.join(output_root, obj_name)
        os.makedirs(out_dir, exist_ok=True)

        obj_cfg, obj_overrides = _object_specific_cfg(cfg, obj_name)

        color, depth_raw, mask, intrinsic = _load_object_inputs_raw(obj_dir)
        mask = _refine_mask_if_needed(color, mask, obj_cfg)
        mesh_inst_raw = _instantmesh_reconstruct(
            color, mask, out_dir, obj_name, obj_cfg, object_seed
        )

        depth_candidates = [obj_cfg.depth_unit]
        if obj_cfg.depth_unit_try_both or obj_cfg.depth_unit == "auto":
            depth_candidates = ["mm", "rayst3r_uint16"]

        best = None
        best_any = None
        candidate_scores: Dict[str, Dict[str, float | str]] = {}
        valid_ratio_score = None
        mesh_inst_metric_fallback = None
        metric_fallback_points = None
        scale_fallback_diagnostics = None
        for depth_unit in depth_candidates:
            depth_m = _convert_depth(depth_raw, depth_unit)
            if obj_cfg.depth_preprocess:
                depth_m = preprocess_depth(
                    depth_m,
                    mask=mask,
                    device=obj_cfg.depth_filter_device,
                    radius=obj_cfg.depth_filter_radius,
                )

            if valid_ratio_score is None:
                valid_ratio_score = _valid_ratio(depth_m, mask)
            range_ratio = _depth_in_range_ratio(
                depth_m, mask, obj_cfg.depth_range_min_m, obj_cfg.depth_range_max_m
            )
            range_ok = range_ratio >= obj_cfg.depth_range_min_ratio

            init_center = None
            if obj_cfg.align_use_guess_translation:
                init_center = guess_translation(depth_m, mask, intrinsic)

            observed_points = _backproject_anchor_depth(
                depth_m, mask, intrinsic, max_points=min(obj_cfg.sample_points, 30000)
            )
            mesh_inst_rgbd = None
            instant_scale_diagnostics: Dict[str, float | int | str | bool] = {
                "enabled": False,
                "valid": False,
                "method": "rayst3r_extent_legacy",
                "label": "instantmesh",
            }
            if len(observed_points) >= 50 and (
                mesh_inst_metric_fallback is None or range_ok
            ):
                if obj_cfg.rgbd_scale_recovery:
                    (
                        mesh_inst_rgbd,
                        _,
                        instant_scale_diagnostics,
                    ) = _rgbd_self_supervised_scale_recovery(
                        mesh_inst_raw,
                        depth_m,
                        mask,
                        intrinsic,
                        obj_cfg,
                        "instantmesh",
                    )
                    mesh_inst_metric_fallback = mesh_inst_rgbd.copy()
                    scale_fallback_diagnostics = dict(instant_scale_diagnostics)
                else:
                    mesh_inst_metric_fallback = _prepare_instantmesh_metric_frame(
                        mesh_inst_raw, observed_points, init_center=init_center
                    )
                metric_fallback_points = observed_points

            work_dir = os.path.join(out_dir, "rayst3r_input")
            try:
                points, mesh_ray = _rayst3r_reconstruct(
                    runner, color, depth_m, mask, intrinsic, obj_cfg, work_dir
                )
            except Exception as exc:
                candidate_scores[depth_unit] = {
                    "fitness": -1.0,
                    "rmse": float("inf"),
                    "decision_score": -1e9,
                    "fitness_raw": -1.0,
                    "rmse_raw": float("inf"),
                    "decision_score_raw": -1e9,
                    "align": "rayst3r_failed",
                    "range_ratio": float(range_ratio),
                    "range_ok": bool(range_ok),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"[WARN] {obj_name}: RaySt3R failed for depth_unit={depth_unit}: "
                    f"{type(exc).__name__}: {exc}; trying the next depth candidate."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if mesh_ray is None or points is None or len(points) == 0:
                candidate_scores[depth_unit] = {
                    "fitness": -1.0,
                    "rmse": float("inf"),
                    "decision_score": -1e9,
                    "fitness_raw": -1.0,
                    "rmse_raw": float("inf"),
                    "decision_score_raw": -1e9,
                    "align": "rayst3r_empty",
                    "range_ratio": float(range_ratio),
                    "range_ok": bool(range_ok),
                }
                continue

            ray_scale_diagnostics: Dict[str, float | int | str | bool] = {
                "enabled": False,
                "valid": False,
                "method": "rayst3r_native_metric",
                "label": "rayst3r",
            }
            if obj_cfg.rgbd_scale_recovery:
                mesh_ray, ray_similarity, ray_scale_diagnostics = (
                    _rgbd_self_supervised_scale_recovery(
                        mesh_ray,
                        depth_m,
                        mask,
                        intrinsic,
                        obj_cfg,
                        "rayst3r",
                    )
                )
                if bool(ray_scale_diagnostics.get("valid", False)):
                    points = _apply_similarity_to_points(points, ray_similarity)
                if mesh_inst_rgbd is None:
                    (
                        mesh_inst_rgbd,
                        _,
                        instant_scale_diagnostics,
                    ) = _rgbd_self_supervised_scale_recovery(
                        mesh_inst_raw,
                        depth_m,
                        mask,
                        intrinsic,
                        obj_cfg,
                        "instantmesh",
                    )
                mesh_inst = mesh_inst_rgbd
            else:
                mesh_inst = _prepare_instantmesh_metric_frame(
                    mesh_inst_raw, points, init_center=init_center
                )

            reg_inst_initial = None
            reg_ray_initial = None
            bidirectional_diag = _bidirectional_diagnostics(
                None, None, mesh_ray, mesh_inst, obj_cfg
            )
            if (valid_ratio_score is not None) and (valid_ratio_score < obj_cfg.valid_ratio_threshold):
                chosen = "none"
                chosen_reg = None
                chosen_mesh_ray = mesh_ray
                chosen_mesh_inst = mesh_inst
                mesh_fused = mesh_ray
            elif obj_cfg.align_bidirectional_icp:
                # Build paired initializations for the two directional ICP
                # searches. Each direction is still optimized and selected
                # independently; pairing only makes the hypothesis bank
                # deterministic and comparable in both directions.
                reciprocal_inst_inits = None
                reciprocal_ray_inits = None
                if obj_cfg.icp_reciprocal_pairing:
                    reciprocal_inst_inits = _build_icp_inits(
                        np.asarray(mesh_inst.vertices, dtype=np.float64).mean(axis=0),
                        np.asarray(points, dtype=np.float64).mean(axis=0),
                        obj_cfg.icp_multi_hypo,
                        obj_cfg.icp_init_sigma,
                    )
                    reciprocal_ray_inits = [
                        np.linalg.inv(init) for init in reciprocal_inst_inits
                    ]

                if obj_cfg.icp_reciprocal_pairing:
                    inst_registrations = _icp_registration_hypotheses(
                        mesh_inst,
                        points,
                        obj_cfg.icp_voxel_size,
                        obj_cfg.icp_max_corr_ratio,
                        obj_cfg.icp_max_iter,
                        reciprocal_inst_inits,
                    )
                    reg_inst, _ = _select_best_icp_registration(
                        inst_registrations, obj_cfg.icp_fitness_threshold
                    )
                    mesh_inst_aligned = _mesh_with_registration(mesh_inst, reg_inst)

                    instant_target_points, _ = trimesh.sample.sample_surface(
                        mesh_inst, obj_cfg.sample_points
                    )
                    ray_registrations = _icp_registration_hypotheses(
                        mesh_ray,
                        instant_target_points,
                        obj_cfg.icp_voxel_size,
                        obj_cfg.icp_max_corr_ratio,
                        obj_cfg.icp_max_iter,
                        reciprocal_ray_inits,
                    )
                    reg_ray, _ = _select_best_icp_registration(
                        ray_registrations, obj_cfg.icp_fitness_threshold
                    )
                    mesh_ray_aligned = _mesh_with_registration(mesh_ray, reg_ray)
                else:
                    mesh_inst_aligned, reg_inst = _align_instantmesh_to_rayst3r(
                        mesh_inst, points, obj_cfg
                    )
                    mesh_ray_aligned, reg_ray = _align_rayst3r_to_instantmesh(
                        mesh_ray, mesh_inst, obj_cfg
                    )
                reg_inst_initial = reg_inst
                reg_ray_initial = reg_ray
                score_inst = _align_decision_score(reg_inst, obj_cfg)
                score_ray = _align_decision_score(reg_ray, obj_cfg)
                reference_direction = (
                    "inst_to_ray" if score_inst >= score_ray else "ray_to_inst"
                )
                if obj_cfg.icp_reciprocal_pairing:
                    paired_inst = None
                    paired_ray = None
                    if reference_direction == "inst_to_ray":
                        paired_inst = reg_inst_initial
                        if _is_valid_reg(reg_inst_initial):
                            instant_target_points, _ = trimesh.sample.sample_surface(
                                mesh_inst,
                                obj_cfg.sample_points,
                                seed=int(object_seed + 101),
                            )
                            ray_source_points, _ = trimesh.sample.sample_surface(
                                mesh_ray,
                                min(50000, len(instant_target_points)),
                                seed=int(object_seed + 102),
                            )
                            reverse = _icp_registration_hypotheses(
                                mesh_ray,
                                instant_target_points,
                                obj_cfg.icp_voxel_size,
                                obj_cfg.icp_max_corr_ratio,
                                obj_cfg.icp_max_iter,
                                [np.linalg.inv(reg_inst_initial.transformation)],
                                source_points=ray_source_points,
                            )
                            paired_ray = reverse[0] if reverse else None
                    else:
                        paired_ray = reg_ray_initial
                        if _is_valid_reg(reg_ray_initial):
                            inst_source_points, _ = trimesh.sample.sample_surface(
                                mesh_inst,
                                min(50000, len(points)),
                                seed=int(object_seed + 103),
                            )
                            reverse = _icp_registration_hypotheses(
                                mesh_inst,
                                points,
                                obj_cfg.icp_voxel_size,
                                obj_cfg.icp_max_corr_ratio,
                                obj_cfg.icp_max_iter,
                                [np.linalg.inv(reg_ray_initial.transformation)],
                                source_points=inst_source_points,
                            )
                            paired_inst = reverse[0] if reverse else None
                    bidirectional_diag = _reciprocal_cycle_diagnostics(
                        reg_inst_initial,
                        reg_ray_initial,
                        paired_inst,
                        paired_ray,
                        mesh_ray,
                        mesh_inst,
                        obj_cfg,
                        reference_direction,
                        None,
                    )
                else:
                    bidirectional_diag = _bidirectional_diagnostics(
                        reg_inst_initial,
                        reg_ray_initial,
                        mesh_ray,
                        mesh_inst,
                        obj_cfg,
                    )
                    bidirectional_diag["cycle_pairing_mode"] = (
                        "independent_directional_optima"
                    )
                    bidirectional_diag["cycle_reference_direction"] = (
                        reference_direction
                    )
                    bidirectional_diag["reciprocal_initialization_pairing"] = False
                bidirectional_diag["hypotheses_per_direction"] = int(
                    obj_cfg.icp_multi_hypo
                )

                if score_inst >= score_ray and reg_inst is not None:
                    mesh_fused = _fuse_meshes(
                        mesh_ray, mesh_inst_aligned, depth_m, mask, intrinsic, obj_cfg
                    )
                    chosen = "inst_to_ray"
                    chosen_reg = reg_inst
                    chosen_mesh_ray = mesh_ray
                    chosen_mesh_inst = mesh_inst_aligned
                elif reg_ray is not None:
                    # Keep Phi in the RaySt3R/anchor-camera frame.  The ICP
                    # direction is still selected by best-of-two, but its
                    # inverse is applied to the target InstantMesh instead of
                    # moving RaySt3R away from the camera frame.
                    mesh_inst_in_ray = _instantmesh_in_rayst3r_frame(
                        mesh_inst, reg_ray
                    )
                    mesh_fused = _fuse_meshes(
                        mesh_ray, mesh_inst_in_ray, depth_m, mask, intrinsic, obj_cfg
                    )
                    chosen = "ray_to_inst"
                    chosen_reg = reg_ray
                    chosen_mesh_ray = mesh_ray
                    chosen_mesh_inst = mesh_inst_in_ray
                elif obj_cfg.align_relaxed_rescue:
                    # Retry once with relaxed ICP settings for hard failure cases.
                    relaxed_cfg = replace(
                        obj_cfg,
                        icp_fitness_threshold=max(0.01, float(obj_cfg.icp_fitness_threshold) * 0.5),
                        icp_max_iter=max(int(obj_cfg.icp_max_iter), 250),
                        icp_multi_hypo=max(int(obj_cfg.icp_multi_hypo), 9),
                        icp_init_sigma=max(float(obj_cfg.icp_init_sigma), 0.02),
                        icp_max_corr_ratio=max(float(obj_cfg.icp_max_corr_ratio), 0.08),
                        icp_voxel_size=max(float(obj_cfg.icp_voxel_size), 0.0075),
                    )
                    mesh_inst_relaxed, reg_inst_relaxed = _align_instantmesh_to_rayst3r(
                        mesh_inst, points, relaxed_cfg
                    )
                    mesh_ray_relaxed, reg_ray_relaxed = _align_rayst3r_to_instantmesh(
                        mesh_ray, mesh_inst, relaxed_cfg
                    )
                    score_inst_relaxed = _align_decision_score(reg_inst_relaxed, relaxed_cfg)
                    score_ray_relaxed = _align_decision_score(reg_ray_relaxed, relaxed_cfg)
                    if score_inst_relaxed >= score_ray_relaxed and reg_inst_relaxed is not None:
                        mesh_fused = _fuse_meshes(
                            mesh_ray, mesh_inst_relaxed, depth_m, mask, intrinsic, relaxed_cfg
                        )
                        chosen = "inst_to_ray_relaxed"
                        chosen_reg = reg_inst_relaxed
                        chosen_mesh_ray = mesh_ray
                        chosen_mesh_inst = mesh_inst_relaxed
                    elif reg_ray_relaxed is not None:
                        mesh_inst_in_ray = _instantmesh_in_rayst3r_frame(
                            mesh_inst, reg_ray_relaxed
                        )
                        mesh_fused = _fuse_meshes(
                            mesh_ray, mesh_inst_in_ray, depth_m, mask, intrinsic, relaxed_cfg
                        )
                        chosen = "ray_to_inst_relaxed"
                        chosen_reg = reg_ray_relaxed
                        chosen_mesh_ray = mesh_ray
                        chosen_mesh_inst = mesh_inst_in_ray
                    else:
                        # Last-chance rescue for difficult objects: use a very permissive ICP
                        # setting to avoid dropping directly to raw mesh.
                        super_relaxed_cfg = replace(
                            obj_cfg,
                            icp_fitness_threshold=max(0.002, float(obj_cfg.icp_fitness_threshold) * 0.2),
                            icp_max_iter=max(int(obj_cfg.icp_max_iter), 400),
                            icp_multi_hypo=max(int(obj_cfg.icp_multi_hypo), 15),
                            icp_init_sigma=max(float(obj_cfg.icp_init_sigma), 0.06),
                            icp_max_corr_ratio=max(float(obj_cfg.icp_max_corr_ratio), 0.22),
                            icp_voxel_size=max(float(obj_cfg.icp_voxel_size), 0.012),
                        )
                        mesh_inst_super, reg_inst_super = _align_instantmesh_to_rayst3r(
                            mesh_inst, points, super_relaxed_cfg
                        )
                        mesh_ray_super, reg_ray_super = _align_rayst3r_to_instantmesh(
                            mesh_ray, mesh_inst, super_relaxed_cfg
                        )
                        score_inst_super = _align_decision_score(reg_inst_super, super_relaxed_cfg)
                        score_ray_super = _align_decision_score(reg_ray_super, super_relaxed_cfg)
                        if score_inst_super >= score_ray_super and reg_inst_super is not None:
                            mesh_fused = _fuse_meshes(
                                mesh_ray, mesh_inst_super, depth_m, mask, intrinsic, super_relaxed_cfg
                            )
                            chosen = "inst_to_ray_super_relaxed"
                            chosen_reg = reg_inst_super
                            chosen_mesh_ray = mesh_ray
                            chosen_mesh_inst = mesh_inst_super
                        elif reg_ray_super is not None:
                            mesh_inst_in_ray = _instantmesh_in_rayst3r_frame(
                                mesh_inst, reg_ray_super
                            )
                            mesh_fused = _fuse_meshes(
                                mesh_ray, mesh_inst_in_ray, depth_m, mask, intrinsic, super_relaxed_cfg
                            )
                            chosen = "ray_to_inst_super_relaxed"
                            chosen_reg = reg_ray_super
                            chosen_mesh_ray = mesh_ray
                            chosen_mesh_inst = mesh_inst_in_ray
                        else:
                            chosen = "none"
                            chosen_reg = None
                            chosen_mesh_ray = mesh_ray
                            chosen_mesh_inst = mesh_inst
                            mesh_fused = mesh_ray
                else:
                    chosen = "none"
                    chosen_reg = None
                    chosen_mesh_ray = mesh_ray
                    chosen_mesh_inst = mesh_inst
                    mesh_fused = mesh_ray
            else:
                mesh_inst_aligned, chosen_reg = _align_instantmesh_to_rayst3r(
                    mesh_inst, points, obj_cfg
                )
                if chosen_reg is None:
                    chosen = "none"
                    chosen_mesh_ray = mesh_ray
                    chosen_mesh_inst = mesh_inst
                    mesh_fused = mesh_ray
                else:
                    mesh_fused = _fuse_meshes(
                        mesh_ray, mesh_inst_aligned, depth_m, mask, intrinsic, obj_cfg
                    )
                    chosen = "inst_to_ray"
                    chosen_mesh_ray = mesh_ray
                    chosen_mesh_inst = mesh_inst_aligned

            if obj_cfg.align_strict_both and not bidirectional_diag["strict_both_pass"]:
                chosen = "none_strict_both"
                chosen_reg = None
                chosen_mesh_ray = mesh_ray
                chosen_mesh_inst = mesh_inst
                mesh_fused = mesh_ray

            fusion_diagnostics = dict(
                (getattr(mesh_fused, "metadata", {}) or {}).get(
                    "fusion_diagnostics", {}
                )
            )
            fusion_diagnostics["coordinate_frame"] = "rayst3r_anchor_camera"
            fusion_diagnostics["alignment_canonicalization"] = (
                "inverse_ray_to_inst_on_instantmesh"
                if str(chosen).startswith("ray_to_inst")
                else "instantmesh_to_rayst3r"
                if str(chosen).startswith("inst_to_ray")
                else "not_available"
            )
            fusion_reliability = float(fusion_diagnostics.get("reliability", 0.0))
            cycle_rotation = bidirectional_diag.get("cycle_rotation_deg")
            cycle_translation = bidirectional_diag.get("cycle_translation_normalized")
            if np.isfinite(cycle_rotation) and np.isfinite(cycle_translation):
                cycle_confidence = float(
                    np.exp(-float(cycle_rotation) / 90.0 - float(cycle_translation))
                )
                # Cycle consistency is primarily a reviewer-requested
                # diagnostic.  Datasets can opt into using it as a bounded
                # confidence term, but the paper-aligned REAL275 main method
                # records it without changing best-of-two reliability.
                if obj_cfg.fusion_cycle_in_reliability:
                    fusion_reliability *= 0.5 + 0.5 * cycle_confidence
            else:
                cycle_confidence = float("nan")
            fusion_diagnostics["cycle_confidence"] = cycle_confidence
            fusion_diagnostics["cycle_used_in_reliability"] = bool(
                obj_cfg.fusion_cycle_in_reliability
            )
            fusion_diagnostics["reliability"] = float(
                np.clip(fusion_reliability, 0.0, 1.0)
            )

            candidate = {
                "depth_unit": depth_unit,
                "points": points,
                "mesh_ray": chosen_mesh_ray,
                "mesh_inst": chosen_mesh_inst,
                "mesh_fused": mesh_fused,
                "align_direction": chosen,
                "fitness": float(chosen_reg.fitness) if chosen_reg is not None else -1.0,
                "rmse": float(chosen_reg.inlier_rmse) if chosen_reg is not None else float("inf"),
                "transform": chosen_reg.transformation if chosen_reg is not None else None,
                "range_ratio": float(range_ratio),
                "range_ok": bool(range_ok),
                "decision_score": (
                    _align_decision_score(chosen_reg, obj_cfg) if chosen_reg is not None else -1e9
                ),
                "bidirectional": bidirectional_diag,
                "fusion_diagnostics": fusion_diagnostics,
                "scale_recovery": {
                    "rayst3r": ray_scale_diagnostics,
                    "instantmesh": instant_scale_diagnostics,
                },
                "rayst3r_metric_scale": float(
                    ray_scale_diagnostics.get("chosen_scale", 1.0)
                ),
                "instantmesh_metric_scale": float(
                    instant_scale_diagnostics.get(
                        "chosen_scale",
                        (getattr(mesh_inst, "metadata", {}) or {}).get(
                            "instantmesh_metric_scale", 1.0
                        ),
                    )
                ),
            }
            fitness_raw = candidate["fitness"]
            rmse_raw = candidate["rmse"]
            decision_score_raw = candidate["decision_score"]
            if (not range_ok) and (not obj_cfg.depth_range_soft_penalty):
                candidate["fitness"] = -1.0
                candidate["rmse"] = float("inf")
                candidate["decision_score"] = -1e9
            elif obj_cfg.depth_range_soft_penalty:
                candidate["fitness"] = candidate["fitness"] * float(range_ratio)
                candidate["decision_score"] = candidate["decision_score"] * float(range_ratio)
            candidate_scores[depth_unit] = {
                "fitness": candidate["fitness"],
                "rmse": candidate["rmse"],
                "decision_score": candidate["decision_score"],
                "fitness_raw": fitness_raw,
                "rmse_raw": rmse_raw,
                "decision_score_raw": decision_score_raw,
                "align": chosen,
                "range_ratio": float(range_ratio),
                "range_ok": bool(range_ok),
                "fusion_method": fusion_diagnostics.get("method", "none"),
                "fusion_reliability": fusion_diagnostics.get("reliability", 0.0),
                "rayst3r_scale": candidate["rayst3r_metric_scale"],
                "instantmesh_scale": candidate["instantmesh_metric_scale"],
                "rayst3r_scale_reliability": ray_scale_diagnostics.get(
                    "reliability", 0.0
                ),
                "instantmesh_scale_reliability": instant_scale_diagnostics.get(
                    "reliability", 0.0
                ),
                **bidirectional_diag,
            }
            if best_any is None or (decision_score_raw > best_any["decision_score"]) or (
                decision_score_raw == best_any["decision_score"] and rmse_raw < best_any["rmse"]
            ):
                best_any = candidate
            candidate_is_valid = (
                not str(candidate["align_direction"]).startswith("none")
                and _is_valid_reg(chosen_reg)
                and np.isfinite(candidate["rmse"])
                and np.isfinite(candidate["fitness"])
            )
            if candidate_is_valid and (range_ok or obj_cfg.depth_range_soft_penalty):
                if best is None or (candidate["decision_score"] > best["decision_score"]) or (
                    candidate["decision_score"] == best["decision_score"] and candidate["rmse"] < best["rmse"]
                ):
                    best = candidate

        if best is None:
            print(
                f"[WARN] {obj_name}: no valid ICP-aligned candidate found; "
                "fallback to best raw candidate (可能导致mesh与位姿质量显著下降)."
            )
            best = best_any
        if best is None:
            print(
                f"[WARN] {obj_name}: all RaySt3R candidates failed; "
                "using instantmesh_only_fallback."
            )
            mesh_inst_fallback = (
                mesh_inst_metric_fallback.copy()
                if mesh_inst_metric_fallback is not None
                else mesh_inst_raw.copy()
            )
            try:
                fallback_points = (
                    np.asarray(metric_fallback_points, dtype=np.float64)
                    if metric_fallback_points is not None
                    else trimesh.sample.sample_surface(
                        mesh_inst_fallback,
                        min(max(int(cfg.sample_points), 10000), 50000),
                    )[0]
                )
            except Exception:
                fallback_points = np.asarray(mesh_inst_fallback.vertices, dtype=np.float64)
            best = {
                "depth_unit": str(depth_candidates[0]) if depth_candidates else str(obj_cfg.depth_unit),
                "points": np.asarray(fallback_points, dtype=np.float64),
                "mesh_ray": mesh_inst_fallback.copy(),
                "mesh_inst": mesh_inst_fallback.copy(),
                "mesh_fused": mesh_inst_fallback.copy(),
                "align_direction": "instantmesh_only_fallback",
                "fitness": -1.0,
                "rmse": float("inf"),
                "transform": None,
                "range_ratio": 0.0,
                "range_ok": False,
                "decision_score": -1e9,
                "fusion_diagnostics": {
                    "method": "instantmesh_metric_fallback",
                    "fallback_used": True,
                    "reliability": 0.0,
                },
                "scale_recovery": {
                    "rayst3r": {
                        "enabled": bool(cfg.rgbd_scale_recovery),
                        "valid": False,
                        "method": "instantmesh_only_fallback",
                    },
                    "instantmesh": scale_fallback_diagnostics or {},
                },
                "rayst3r_metric_scale": 1.0,
                "instantmesh_metric_scale": float(
                    (getattr(mesh_inst_fallback, "metadata", {}) or {}).get(
                        "instantmesh_metric_scale", 1.0
                    )
                ),
            }

        points = best["points"]
        mesh_ray = best["mesh_ray"]
        mesh_inst_aligned = best["mesh_inst"]
        mesh_fused = best["mesh_fused"]
        depth_unit = best["depth_unit"]
        transform = best["transform"]

        pointcloud_path = os.path.join(out_dir, f"rayst3r_points_{obj_name}.ply")
        mesh_ray_path = os.path.join(out_dir, f"mesh_rayst3r_{obj_name}.obj")
        mesh_inst_path = os.path.join(out_dir, f"mesh_instant_{obj_name}.obj")
        mesh_fused_path = os.path.join(out_dir, f"mesh_fused_{obj_name}.obj")

        trimesh.PointCloud(points).export(pointcloud_path)
        mesh_ray.export(mesh_ray_path)
        mesh_inst_aligned.export(mesh_inst_path)
        mesh_fused.export(mesh_fused_path)

        meta = {
            "object": obj_name,
            "object_seed": object_seed,
            "mask_source": "dataset_gt" if not obj_cfg.refine_mask else "sam2_refined",
            "icp_correspondence_rule": "max(2*voxel_size, max_corr_ratio*target_bbox_diagonal)",
            "depth_unit": depth_unit,
            "depth_unit_candidates": candidate_scores,
            "valid_ratio": valid_ratio_score,
            "valid_ratio_threshold": obj_cfg.valid_ratio_threshold,
            "depth_range_min_m": obj_cfg.depth_range_min_m,
            "depth_range_max_m": obj_cfg.depth_range_max_m,
            "depth_range_min_ratio": obj_cfg.depth_range_min_ratio,
            "depth_range_soft_penalty": obj_cfg.depth_range_soft_penalty,
            "depth_range_ratio": best.get("range_ratio"),
            "depth_range_ok": best.get("range_ok"),
            "align_direction": best["align_direction"],
            "align_fitness": best["fitness"],
            "align_rmse": best["rmse"],
            "align_decision_score": best["decision_score"],
            "rayst3r_metric_scale": best.get("rayst3r_metric_scale", 1.0),
            "instantmesh_metric_scale": best.get("instantmesh_metric_scale", 1.0),
            "scale_recovery": best.get("scale_recovery", {}),
            "fusion_diagnostics": best.get("fusion_diagnostics", {}),
            "bidirectional": best.get(
                "bidirectional",
                _bidirectional_diagnostics(None, None, mesh_ray, mesh_inst_aligned, obj_cfg),
            ),
            "object_overrides": obj_overrides,
            "outputs": {
                "pointcloud": pointcloud_path,
                "mesh_rayst3r": mesh_ray_path,
                "mesh_instant": mesh_inst_path,
                "mesh_fused": mesh_fused_path,
            },
            "mesh_statistics": {
                "rayst3r": {
                    "vertices": int(len(mesh_ray.vertices)),
                    "faces": int(len(mesh_ray.faces)),
                },
                "instantmesh": {
                    "vertices": int(len(mesh_inst_aligned.vertices)),
                    "faces": int(len(mesh_inst_aligned.faces)),
                },
                "fused": {
                    "vertices": int(len(mesh_fused.vertices)),
                    "faces": int(len(mesh_fused.faces)),
                },
            },
            "transform": transform.tolist() if transform is not None else None,
            "config": asdict(obj_cfg),
            "base_config": asdict(cfg),
        }
        meta_path = os.path.join(out_dir, "reconstruction_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"{obj_name}: depth_unit={depth_unit} range_ratio={best.get('range_ratio'):.3f} "
            f"range_ok={best.get('range_ok')} align={best['align_direction']} "
            f"fitness={best['fitness']:.3f} rmse={best['rmse']:.6f}"
        )

        results[obj_name] = {
            "pointcloud": pointcloud_path,
            "mesh_rayst3r": mesh_ray_path,
            "mesh_instant": mesh_inst_path,
            "mesh_fused": mesh_fused_path,
            "meta": meta_path,
        }

    return results


__all__ = ["ReconstructionConfig", "run_construction"]
