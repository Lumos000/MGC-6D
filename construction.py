
from project_paths import setup_project_paths
setup_project_paths()
import json
import os
import glob
from dataclasses import dataclass, asdict, replace
from typing import Dict, Optional, Tuple, List, Iterable

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh

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
    icp_max_iter: int = 50
    icp_fitness_threshold: float = 0.05
    icp_multi_hypo: int = 1
    icp_init_sigma: float = 0.01
    icp_max_corr_ratio: float = 0.05  # fraction of rayst3r bbox diag
    icp_voxel_size: float = 0.005
    sample_points: int = 50000
    align_score_rmse_weight: float = 40.0
    align_low_fitness_penalty: float = 0.15
    align_high_rmse_penalty: float = 0.15
    align_rmse_penalty_threshold: float = 0.02


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

    if init_transforms is None:
        init_transforms = [np.eye(4, dtype=np.float64)]

    best_reg = None
    for init in init_transforms:
        reg = o3d.pipelines.registration.registration_icp(
            prior_pcd,
            target_pcd,
            max_correspondence_distance=max(voxel_size * 5.0, 1e-6),
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
    return points, mesh


def _instantmesh_reconstruct(
    color: np.ndarray,
    mask: np.ndarray,
    obj_dir: str,
    obj_name: str,
    cfg: ReconstructionConfig,
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


def _align_instantmesh_to_rayst3r(
    mesh_instant: trimesh.Trimesh,
    ray_points: np.ndarray,
    cfg: ReconstructionConfig,
    init_center: Optional[np.ndarray] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[o3d.pipelines.registration.RegistrationResult]]:
    if ray_points is None or len(ray_points) == 0:
        return mesh_instant, None

    ray_pcd = _points_to_o3d(ray_points)
    ray_bbox = ray_pcd.get_axis_aligned_bounding_box()
    ray_extent = np.array(ray_bbox.get_extent(), dtype=np.float64)
    inst_bbox = mesh_instant.bounding_box.extents

    if init_center is not None and np.any(np.isfinite(init_center)):
        mesh_instant = mesh_instant.copy()
        inst_centroid = mesh_instant.vertices.mean(axis=0)
        mesh_instant.apply_translation(init_center - inst_centroid)

    if np.all(inst_bbox > 0):
        scale = float(np.mean(ray_extent) / np.mean(inst_bbox))
        mesh_instant = mesh_instant.copy()
        mesh_instant.apply_scale(scale)
    else:
        scale = 1.0

    target_center = ray_points.mean(axis=0)
    source_center = mesh_instant.vertices.mean(axis=0)
    init_transforms = _build_icp_inits(
        source_center,
        target_center,
        cfg.icp_multi_hypo,
        cfg.icp_init_sigma,
    )
    aligned, reg = _icp_align_prior(
        mesh_instant,
        ray_points,
        voxel_size=cfg.icp_voxel_size,
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
        max_iter=cfg.icp_max_iter,
        fitness_threshold=cfg.icp_fitness_threshold,
        init_transforms=init_transforms,
    )
    return aligned, reg


def _fuse_meshes(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
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


def _object_specific_cfg(cfg: ReconstructionConfig, obj_name: str) -> Tuple[ReconstructionConfig, Dict[str, float | bool]]:
    if obj_name not in SB_SM_OBJECTS:
        return cfg, {}

    overrides: Dict[str, float | bool] = {
        "refine_mask": True,
        "mask_stability_enabled": True,
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


def run_construction(cfg: ReconstructionConfig) -> Dict[str, Dict[str, str]]:
    anchor_root = os.path.realpath(cfg.anchor_root)
    output_root = os.path.realpath(cfg.output_root or cfg.anchor_root)

    checkpoint_path = cfg.rayst3r_checkpoint
    if checkpoint_path:
        checkpoint_path = os.path.realpath(checkpoint_path)
    runner = Rayst3RRunner(checkpoint_path=checkpoint_path, device=cfg.rayst3r_device)

    results: Dict[str, Dict[str, str]] = {}
    obj_list = [d for d in os.listdir(anchor_root) if os.path.isdir(os.path.join(anchor_root, d))]

    for obj_name in obj_list:
        obj_dir = os.path.join(anchor_root, obj_name)
        out_dir = os.path.join(output_root, obj_name)
        os.makedirs(out_dir, exist_ok=True)

        obj_cfg, obj_overrides = _object_specific_cfg(cfg, obj_name)

        color, depth_raw, mask, intrinsic = _load_object_inputs_raw(obj_dir)
        mask = _refine_mask_if_needed(color, mask, obj_cfg)
        mesh_inst = _instantmesh_reconstruct(color, mask, out_dir, obj_name, obj_cfg)

        depth_candidates = [obj_cfg.depth_unit]
        if obj_cfg.depth_unit_try_both or obj_cfg.depth_unit == "auto":
            depth_candidates = ["mm", "rayst3r_uint16"]

        best = None
        best_any = None
        candidate_scores: Dict[str, Dict[str, float | str]] = {}
        valid_ratio_score = None
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

            work_dir = os.path.join(out_dir, "rayst3r_input")
            points, mesh_ray = _rayst3r_reconstruct(runner, color, depth_m, mask, intrinsic, obj_cfg, work_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if mesh_ray is None:
                continue

            if (valid_ratio_score is not None) and (valid_ratio_score < obj_cfg.valid_ratio_threshold):
                chosen = "none"
                chosen_reg = None
                chosen_mesh_ray = mesh_ray
                chosen_mesh_inst = mesh_inst
                mesh_fused = mesh_ray
            elif obj_cfg.align_bidirectional_icp:
                mesh_inst_aligned, reg_inst = _align_instantmesh_to_rayst3r(
                    mesh_inst, points, obj_cfg, init_center=init_center
                )
                mesh_ray_aligned, reg_ray = _align_rayst3r_to_instantmesh(mesh_ray, mesh_inst, obj_cfg)

                score_inst = _align_decision_score(reg_inst, obj_cfg)
                score_ray = _align_decision_score(reg_ray, obj_cfg)

                if score_inst >= score_ray and reg_inst is not None:
                    mesh_fused = _fuse_meshes(mesh_ray, mesh_inst_aligned)
                    chosen = "inst_to_ray"
                    chosen_reg = reg_inst
                    chosen_mesh_ray = mesh_ray
                    chosen_mesh_inst = mesh_inst_aligned
                elif reg_ray is not None:
                    mesh_fused = _fuse_meshes(mesh_ray_aligned, mesh_inst)
                    chosen = "ray_to_inst"
                    chosen_reg = reg_ray
                    chosen_mesh_ray = mesh_ray_aligned
                    chosen_mesh_inst = mesh_inst
                else:
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
                        mesh_inst, points, relaxed_cfg, init_center=init_center
                    )
                    mesh_ray_relaxed, reg_ray_relaxed = _align_rayst3r_to_instantmesh(
                        mesh_ray, mesh_inst, relaxed_cfg
                    )
                    score_inst_relaxed = _align_decision_score(reg_inst_relaxed, relaxed_cfg)
                    score_ray_relaxed = _align_decision_score(reg_ray_relaxed, relaxed_cfg)
                    if score_inst_relaxed >= score_ray_relaxed and reg_inst_relaxed is not None:
                        mesh_fused = _fuse_meshes(mesh_ray, mesh_inst_relaxed)
                        chosen = "inst_to_ray_relaxed"
                        chosen_reg = reg_inst_relaxed
                        chosen_mesh_ray = mesh_ray
                        chosen_mesh_inst = mesh_inst_relaxed
                    elif reg_ray_relaxed is not None:
                        mesh_fused = _fuse_meshes(mesh_ray_relaxed, mesh_inst)
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
                            mesh_inst, points, super_relaxed_cfg, init_center=init_center
                        )
                        mesh_ray_super, reg_ray_super = _align_rayst3r_to_instantmesh(
                            mesh_ray, mesh_inst, super_relaxed_cfg
                        )
                        score_inst_super = _align_decision_score(reg_inst_super, super_relaxed_cfg)
                        score_ray_super = _align_decision_score(reg_ray_super, super_relaxed_cfg)
                        if score_inst_super >= score_ray_super and reg_inst_super is not None:
                            mesh_fused = _fuse_meshes(mesh_ray, mesh_inst_super)
                            chosen = "inst_to_ray_super_relaxed"
                            chosen_reg = reg_inst_super
                            chosen_mesh_ray = mesh_ray
                            chosen_mesh_inst = mesh_inst_super
                        elif reg_ray_super is not None:
                            mesh_fused = _fuse_meshes(mesh_ray_super, mesh_inst)
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
                mesh_inst_aligned, chosen_reg = _align_instantmesh_to_rayst3r(
                    mesh_inst, points, obj_cfg, init_center=init_center
                )
                if chosen_reg is None:
                    chosen = "none"
                    chosen_mesh_ray = mesh_ray
                    chosen_mesh_inst = mesh_inst
                    mesh_fused = mesh_ray
                else:
                    mesh_fused = _fuse_meshes(mesh_ray, mesh_inst_aligned)
                    chosen = "inst_to_ray"
                    chosen_mesh_ray = mesh_ray
                    chosen_mesh_inst = mesh_inst_aligned

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
            }
            if best_any is None or (decision_score_raw > best_any["decision_score"]) or (
                decision_score_raw == best_any["decision_score"] and rmse_raw < best_any["rmse"]
            ):
                best_any = candidate
            candidate_is_valid = (
                candidate["align_direction"] != "none"
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
            raise RuntimeError(f"Reconstruction failed for {obj_name}")

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
            "object_overrides": obj_overrides,
            "outputs": {
                "pointcloud": pointcloud_path,
                "mesh_rayst3r": mesh_ray_path,
                "mesh_instant": mesh_inst_path,
                "mesh_fused": mesh_fused_path,
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
