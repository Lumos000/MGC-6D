
from project_paths import setup_project_paths
setup_project_paths()
import json
import os
import glob
import hashlib
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
    icp_init_sigma: float = 0.01
    icp_max_corr_ratio: float = 0.05  # fraction of rayst3r bbox diag
    icp_voxel_size: float = 0.005
    sample_points: int = 50000
    rayst3r_crop_margin_ratio: float = 0.05
    rayst3r_min_component_area_ratio: float = 0.005
    rayst3r_target_faces: int = 100000
    fusion_voxel_size: float = 0.003
    fusion_sample_points: int = 15000
    fusion_poisson_depth: int = 6
    fusion_target_faces: int = 20000
    fusion_density_quantile: float = 0.02
    fusion_visible_depth_tolerance: float = 0.015
    fusion_observed_support_ratio: float = 0.08
    fusion_agreement_ratio: float = 0.025
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
    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0.0)
    return float(valid.sum()) / float(mask.size + 1e-6)


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


def _build_icp_inits(
    source_center: np.ndarray,
    target_center: np.ndarray,
    num_inits: int,
    sigma: float,
) -> List[np.ndarray]:
    base = np.eye(4, dtype=np.float64)
    base[:3, 3] = (target_center - source_center).astype(np.float64)
    if num_inits <= 1:
        return [base]
    rng = np.random.RandomState(0)
    inits = [base]
    for _ in range(num_inits - 1):
        t = rng.normal(scale=sigma, size=(3,))
        init = base.copy()
        init[:3, 3] += t
        inits.append(init)
    return inits


def _icp_align_prior(
    prior_mesh: trimesh.Trimesh,
    target_points: np.ndarray,
    voxel_size: float,
    max_corr_ratio: float,
    max_iter: int,
    fitness_threshold: float,
    init_transforms: Optional[Iterable[np.ndarray]] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[o3d.pipelines.registration.RegistrationResult]]:
    if prior_mesh is None or target_points is None or len(target_points) == 0:
        return None, None
    target_pcd = _points_to_o3d(target_points)
    if voxel_size and voxel_size > 0:
        target_pcd = target_pcd.voxel_down_sample(voxel_size=voxel_size)

    prior_pcd = _mesh_to_o3d(prior_mesh).sample_points_uniformly(
        number_of_points=min(50000, len(target_pcd.points))
    )
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

    best_reg = None
    for init in init_transforms:
        reg = o3d.pipelines.registration.registration_icp(
            prior_pcd,
            target_pcd,
            max_correspondence_distance=max_correspondence_distance,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
        if best_reg is None:
            best_reg = reg
        else:
            if (reg.fitness > best_reg.fitness) or (
                reg.fitness == best_reg.fitness and reg.inlier_rmse < best_reg.inlier_rmse
            ):
                best_reg = reg

    if best_reg is None or (not best_reg.fitness) or best_reg.fitness < fitness_threshold:
        return None, None

    prior_mesh_aligned = prior_mesh.copy()
    prior_mesh_aligned.apply_transform(best_reg.transformation)
    return prior_mesh_aligned, best_reg


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
    prepared.apply_translation(target_center - source_center)
    prepared.metadata = dict(getattr(prepared, "metadata", {}) or {})
    prepared.metadata["instantmesh_metric_scale"] = float(scale)
    return prepared


def _align_instantmesh_to_rayst3r(
    mesh_instant_metric: trimesh.Trimesh,
    ray_points: np.ndarray,
    cfg: ReconstructionConfig,
) -> Tuple[Optional[trimesh.Trimesh], Optional[o3d.pipelines.registration.RegistrationResult]]:
    if ray_points is None or len(ray_points) == 0:
        return mesh_instant_metric, None

    target_center = ray_points.mean(axis=0)
    source_center = mesh_instant_metric.vertices.mean(axis=0)
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
) -> Tuple[Optional[trimesh.Trimesh], Optional[o3d.pipelines.registration.RegistrationResult]]:
    target_points, _ = trimesh.sample.sample_surface(mesh_instant, cfg.sample_points)
    target_center = target_points.mean(axis=0)
    source_center = mesh_ray.vertices.mean(axis=0)
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
    """Mark model samples that contradict the measured anchor-visible surface."""

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
    bad = visible & valid_depth & (np.abs(z[ids] - observed) > float(tolerance))
    conflict[ids[bad]] = True
    return conflict


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

    fallback = _concatenate_meshes(mesh_ray, mesh_instant)
    diagnostics: Dict[str, float | int | str | bool] = {
        "method": "visible_prior_poisson_v1",
        "fallback_used": False,
    }
    try:
        n_samples = max(int(cfg.fusion_sample_points), 1000)
        ray_pts, ray_normals = _sample_mesh_with_normals(mesh_ray, n_samples)
        inst_pts, inst_normals = _sample_mesh_with_normals(mesh_instant, n_samples)
        obs_pts = _backproject_anchor_depth(
            depth, mask, intrinsic, max_points=min(n_samples, 30000)
        )
        if len(obs_pts) < 50:
            raise RuntimeError("too few valid anchor depth points for Phi")

        combined_extent = np.ptp(np.vstack([ray_pts, inst_pts]), axis=0)
        diameter = max(float(np.linalg.norm(combined_extent)), 1e-6)
        voxel = max(float(cfg.fusion_voxel_size), 1e-4)
        support_distance = max(
            float(cfg.fusion_observed_support_ratio) * diameter, 2.0 * voxel
        )
        agreement_distance = max(
            float(cfg.fusion_agreement_ratio) * diameter, 2.0 * voxel
        )

        ray_obs_distance = cKDTree(obs_pts).query(ray_pts, k=1, workers=-1)[0]
        ray_inst_distance = cKDTree(inst_pts).query(ray_pts, k=1, workers=-1)[0]
        ray_conflict = _visible_depth_conflict_mask(
            ray_pts,
            depth,
            mask,
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
            mask,
            intrinsic,
            cfg.fusion_visible_depth_tolerance,
        )
        inst_keep = ~inst_conflict
        retained_ray = ray_pts[ray_keep]
        if len(retained_ray):
            inst_to_ray = cKDTree(retained_ray).query(inst_pts, k=1, workers=-1)[0]
            # RaySt3R wins in overlap; InstantMesh remains as completion.
            inst_keep &= inst_to_ray > max(1.5 * voxel, 0.5 * agreement_distance)

        points = np.vstack([retained_ray, inst_pts[inst_keep], obs_pts])
        normals = np.vstack(
            [
                ray_normals[ray_keep],
                inst_normals[inst_keep],
                np.zeros_like(obs_pts),
            ]
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd = pcd.voxel_down_sample(voxel)
        if len(pcd.points) < 200:
            raise RuntimeError("too few retained points after Phi filtering")

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(3.0 * voxel, 0.01), max_nn=40
            )
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(30)
        except Exception:
            pcd.orient_normals_towards_camera_location(np.zeros(3))

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
        target_faces = int(cfg.fusion_target_faces)
        if target_faces > 0 and len(mesh_o3d.triangles) > target_faces:
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_faces)

        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError("Poisson Phi produced an empty mesh")
        fused = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

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
        reliability = float(
            np.clip(
                (0.5 * overlap + 0.5 * distance_quality) * (1.0 - conflict_ratio),
                0.0,
                1.0,
            )
        )
        diagnostics.update(
            {
                "ray_samples": int(len(ray_pts)),
                "instantmesh_samples": int(len(inst_pts)),
                "observed_samples": int(len(obs_pts)),
                "ray_retained": int(ray_keep.sum()),
                "instantmesh_retained": int(inst_keep.sum()),
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
) -> Dict[str, float | bool]:
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
    if reg_inst_to_ray is not None and reg_ray_to_inst is not None:
        cycle = (
            np.asarray(reg_inst_to_ray.transformation, dtype=np.float64)
            @ np.asarray(reg_ray_to_inst.transformation, dtype=np.float64)
        )
        cos_angle = np.clip((np.trace(cycle[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        rotation_deg = float(np.degrees(np.arccos(cos_angle)))
        diameter = max(
            float(np.linalg.norm(np.asarray(mesh_ray.extents, dtype=np.float64))),
            float(np.linalg.norm(np.asarray(mesh_inst.extents, dtype=np.float64))),
            1e-8,
        )
        translation_norm = float(np.linalg.norm(cycle[:3, 3]) / diameter)
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
    }


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
            if len(observed_points) >= 50 and (
                mesh_inst_metric_fallback is None or range_ok
            ):
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
                mesh_inst_aligned, reg_inst = _align_instantmesh_to_rayst3r(
                    mesh_inst, points, obj_cfg
                )
                mesh_ray_aligned, reg_ray = _align_rayst3r_to_instantmesh(mesh_ray, mesh_inst, obj_cfg)
                reg_inst_initial = reg_inst
                reg_ray_initial = reg_ray
                bidirectional_diag = _bidirectional_diagnostics(
                    reg_inst_initial, reg_ray_initial, mesh_ray, mesh_inst, obj_cfg
                )

                score_inst = _align_decision_score(reg_inst, obj_cfg)
                score_ray = _align_decision_score(reg_ray, obj_cfg)

                if score_inst >= score_ray and reg_inst is not None:
                    mesh_fused = _fuse_meshes(
                        mesh_ray, mesh_inst_aligned, depth_m, mask, intrinsic, obj_cfg
                    )
                    chosen = "inst_to_ray"
                    chosen_reg = reg_inst
                    chosen_mesh_ray = mesh_ray
                    chosen_mesh_inst = mesh_inst_aligned
                elif reg_ray is not None:
                    mesh_fused = _fuse_meshes(
                        mesh_ray_aligned, mesh_inst, depth_m, mask, intrinsic, obj_cfg
                    )
                    chosen = "ray_to_inst"
                    chosen_reg = reg_ray
                    chosen_mesh_ray = mesh_ray_aligned
                    chosen_mesh_inst = mesh_inst
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
                        mesh_fused = _fuse_meshes(
                            mesh_ray_relaxed, mesh_inst, depth_m, mask, intrinsic, relaxed_cfg
                        )
                        chosen = "ray_to_inst_relaxed"
                        chosen_reg = reg_ray_relaxed
                        chosen_mesh_ray = mesh_ray_relaxed
                        chosen_mesh_inst = mesh_inst
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
                            mesh_fused = _fuse_meshes(
                                mesh_ray_super, mesh_inst, depth_m, mask, intrinsic, super_relaxed_cfg
                            )
                            chosen = "ray_to_inst_super_relaxed"
                            chosen_reg = reg_ray_super
                            chosen_mesh_ray = mesh_ray_super
                            chosen_mesh_inst = mesh_inst
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
            fusion_reliability = float(fusion_diagnostics.get("reliability", 0.0))
            cycle_rotation = bidirectional_diag.get("cycle_rotation_deg")
            cycle_translation = bidirectional_diag.get("cycle_translation_normalized")
            if np.isfinite(cycle_rotation) and np.isfinite(cycle_translation):
                cycle_confidence = float(
                    np.exp(-float(cycle_rotation) / 90.0 - float(cycle_translation))
                )
                # Cycle agreement is a continuous confidence term, not the
                # strict-both gate requested only for reviewer ablation.
                fusion_reliability *= 0.5 + 0.5 * cycle_confidence
            else:
                cycle_confidence = float("nan")
            fusion_diagnostics["cycle_confidence"] = cycle_confidence
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
                "instantmesh_metric_scale": float(
                    (getattr(mesh_inst, "metadata", {}) or {}).get(
                        "instantmesh_metric_scale", 1.0
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
            "instantmesh_metric_scale": best.get("instantmesh_metric_scale", 1.0),
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
