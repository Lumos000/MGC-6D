from project_paths import setup_project_paths
setup_project_paths()

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SAM2_ROOT = PROJECT_ROOT / "sam2"
SAM2_PKG_ROOT = SAM2_ROOT / "sam2"
DEFAULT_SAM2_CKPT = str(PROJECT_ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _import_rayst3r():
    try:
        from rayst3r.eval_wrapper.eval import EvalWrapper, eval_scene
        return EvalWrapper, eval_scene
    except ImportError:
        pass
    try:
        from eval_wrapper.eval import EvalWrapper, eval_scene
        return EvalWrapper, eval_scene
    except ImportError as exc:
        ray_root = os.environ.get("RAYST3R_ROOT", str(PROJECT_ROOT.parent / "rayst3r"))
        raise ImportError(
            "RaySt3R is required for anchor reconstruction. Clone RaySt3R and set "
            f"RAYST3R_ROOT to its checkout, for example: export RAYST3R_ROOT={ray_root}"
        ) from exc


EvalWrapper, eval_scene = _import_rayst3r()

from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor


def _resolve_sam2_checkpoint(checkpoint: Optional[str] = None) -> str:
    ckpt = checkpoint or os.environ.get("SAM2_CKPT") or DEFAULT_SAM2_CKPT
    ckpt_path = Path(ckpt).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {ckpt_path}. Download sam2.1_hiera_large.pt "
            "to sam2/checkpoints/ or set SAM2_CKPT."
        )
    return str(ckpt_path)


def _resolve_sam2_config(cfg_path: Optional[str]) -> Optional[str]:
    if cfg_path is None:
        return cfg_path
    cfg_path = os.environ.get("SAM2_CFG", cfg_path)
    cfg_real = Path(cfg_path).expanduser()
    if cfg_real.exists():
        try:
            return str(cfg_real.resolve().relative_to(SAM2_PKG_ROOT.resolve()))
        except ValueError:
            return str(cfg_real)
    return cfg_path


def get_bounding_box(mask: np.ndarray, pad_rel: float = 0.0, return_torch: bool = False):
    """Compute xyxy bounding box for a binary mask."""
    non_zero_indices = np.nonzero(mask)

    if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
        x_min = y_min = x_max = y_max = 0
    else:
        y_min, x_min = np.min(non_zero_indices, axis=1)
        y_max, x_max = np.max(non_zero_indices, axis=1)

    x_pad = pad_rel * (x_max - x_min)
    y_pad = pad_rel * (y_max - y_min)

    height, width = mask.shape
    x1 = max(0, x_min - x_pad)
    y1 = max(0, y_min - y_pad)
    x2 = min(width - 1, x_max + x_pad)
    y2 = min(height - 1, y_max + y_pad)

    bbox = np.array([x1, y1, x2, y2])
    return torch.from_numpy(bbox[None]).cuda() if return_torch else bbox


def running_sam_box(
    color: np.ndarray,
    box: Optional[np.ndarray] = None,
    checkpoint: Optional[str] = None,
    model_cfg: str = DEFAULT_SAM2_CFG,
) -> np.ndarray:
    """Run SAM2 to predict a binary mask given an optional box prompt."""
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    cfg_name = _resolve_sam2_config(model_cfg)
    ckpt_path = _resolve_sam2_checkpoint(checkpoint)
    sam_predictor = SAM2ImagePredictor(build_sam2(cfg_name, ckpt_path))
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam_predictor.set_image(color)
            masks, scores, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            mask = masks[0].astype(np.bool_)
    finally:
        torch.backends.cudnn.enabled = prev_cudnn

    del sam_predictor
    torch.cuda.empty_cache()
    return mask


def depth_to_uint16(depth_m: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
    """Convert depth in meters to uint16 format expected by RaySt3R."""
    depth_clipped = np.clip(depth_m, 0.0, max_depth)
    scale = np.iinfo(np.uint16).max / max_depth
    return (depth_clipped * scale).astype(np.uint16)


def write_rayst3r_inputs(
    color: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    out_dir: str,
) -> str:
    """Persist RaySt3R input files to a folder."""
    os.makedirs(out_dir, exist_ok=True)

    Image.fromarray(color).save(os.path.join(out_dir, "rgb.png"))
    Image.fromarray((mask.astype(np.uint8) * 255)).save(os.path.join(out_dir, "mask.png"))

    depth_uint16 = depth_to_uint16(depth_m)
    Image.fromarray(depth_uint16, mode="I;16").save(os.path.join(out_dir, "depth.png"))

    torch.save(torch.tensor(intrinsic, dtype=torch.float32), os.path.join(out_dir, "intrinsics.pt"))
    torch.save(torch.eye(4, dtype=torch.float32), os.path.join(out_dir, "cam2world.pt"))
    return out_dir


class Rayst3RRunner:
    """Lightweight wrapper around RaySt3R inference."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download("bartduis/rayst3r", "rayst3r.pth")

        self.model = EvalWrapper(checkpoint_path, distributed=False, device=device, dtype=dtype)
        self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        self.dino_model.eval()
        self.dino_model.to(device)

    def reconstruct_from_arrays(
        self,
        color: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        intrinsic: np.ndarray,
        work_dir: Optional[str] = None,
        set_conf: float = 5.0,
        n_pred_views: int = 5,
        filter_all_masks: bool = True,
        tsdf: bool = False,
        return_conf: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run RaySt3R on in-memory arrays and return point cloud and optional confidences."""
        temp_dir = None
        if work_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="rayst3r_input_")
            work_dir = temp_dir

        write_rayst3r_inputs(color, depth_m, mask, intrinsic, work_dir)

        result = eval_scene(
            self.model,
            work_dir,
            set_conf=set_conf,
            n_pred_views=n_pred_views,
            do_filter_all_masks=filter_all_masks,
            dino_model=self.dino_model,
            tsdf=tsdf,
        )

        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if return_conf:
            return result.cpu().numpy(), None
        return result.cpu().numpy(), None


def pointcloud_to_mesh(
    points: np.ndarray,
    voxel_size: float = 0.001,
    nb_neighbors: int = 20,
    std_ratio: float = 3.0,
    poisson_depth: int = 8,
    poisson_scale: float = 1.0,
    density_quantile: float = 0.01,
) -> Optional[trimesh.Trimesh]:
    """Convert a point cloud to a mesh with mild filtering."""
    if points is None or len(points) == 0:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=0, scale=poisson_scale, linear_fit=False
    )
    if densities is not None and density_quantile is not None:
        densities = np.asarray(densities)
        mask = densities < np.quantile(densities, density_quantile)
        mesh_o3d.remove_vertices_by_mask(mask)

    mesh_o3d.remove_unreferenced_vertices()
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


__all__ = [
    "Rayst3RRunner",
    "get_bounding_box",
    "running_sam_box",
    "pointcloud_to_mesh",
    "write_rayst3r_inputs",
]
