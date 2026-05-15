import numpy as np

from foundationpose.Utils import erode_depth, bilateral_filter_depth


def _normalize_device(device: str) -> str:
    if device is None:
        return "cuda"
    if device.startswith("cuda"):
        return "cuda"
    return device


def preprocess_depth(
    depth: np.ndarray,
    mask: np.ndarray | None = None,
    device: str = "cuda",
    radius: int = 2,
) -> np.ndarray:
    if depth is None:
        return depth
    depth = depth.astype(np.float32)
    warp_device = _normalize_device(device)
    depth = erode_depth(depth, radius=radius, device=warp_device)
    depth = bilateral_filter_depth(depth, radius=radius, device=warp_device)
    if mask is not None:
        depth = depth.copy()
        depth[mask == False] = 0
    return depth


def guess_translation(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        return np.zeros((3), dtype=np.float32)
    uc = (us.min() + us.max()) / 2.0
    vc = (vs.min() + vs.max()) / 2.0
    valid = mask.astype(bool) & (depth >= 0.001)
    if not valid.any():
        return np.zeros((3), dtype=np.float32)
    zc = np.median(depth[valid])
    center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc
    return center.reshape(3).astype(np.float32)


__all__ = ["preprocess_depth", "guess_translation"]
