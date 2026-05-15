import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def add_suffix(path: str, suffix: str):
    p = Path(path)
    return str(p.with_name(p.stem + suffix + p.suffix))


def load_intrinsics(k_path: str):
    """Load 3x3 camera intrinsic matrix from K.txt."""
    K = np.loadtxt(k_path).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return fx, fy, cx, cy, K


def load_depth(depth_path: str, depth_scale: float):
    """
    Load depth image and convert it to meters.

    depth_scale:
        uint16 millimeter depth: use 1000.0
        meter-valued npy depth: use 1.0
    """
    depth_path = str(depth_path)

    if depth_path.endswith(".npy"):
        depth_raw = np.load(depth_path)
    else:
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth_raw is None:
        raise FileNotFoundError(f"Cannot read depth image: {depth_path}")

    print("\n[Depth raw]")
    print("  path:", depth_path)
    print("  shape:", depth_raw.shape)
    print("  dtype:", depth_raw.dtype)
    print("  min/max:", depth_raw.min(), depth_raw.max())

    if depth_raw.ndim == 3:
        raise ValueError(
            "Depth image has 3 channels. It is likely a visualized colormap, "
            "not raw metric depth."
        )

    if depth_raw.dtype == np.uint16 and depth_scale == 1:
        print(
            "[WARNING] depth image is uint16 but depth_scale=1. "
            "If this is millimeter depth, use --depth_scale 1000."
        )

    depth = depth_raw.astype(np.float32) / float(depth_scale)
    depth[~np.isfinite(depth)] = 0.0

    nz = depth[depth > 0]
    if nz.size > 0:
        print("[Depth meters]")
        print("  nonzero min/max:", float(nz.min()), float(nz.max()))
        print("  percentiles:", np.percentile(nz, [1, 5, 50, 95, 99]))
    else:
        print("[WARNING] no nonzero depth values.")

    return depth


def load_rgb(rgb_path: str, h: int, w: int):
    if rgb_path is None:
        return None

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Cannot read RGB image: {rgb_path}")

    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    return rgb


def load_mask(mask_path: str, h: int, w: int, threshold: int = 0):
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Cannot read mask image: {mask_path}")

    if mask_raw.shape[:2] != (h, w):
        mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    print("\n[Mask raw]")
    print("  path:", mask_path)
    print("  shape:", mask_raw.shape)
    print("  dtype:", mask_raw.dtype)
    print("  min/max:", mask_raw.min(), mask_raw.max())
    print("  foreground pixels:", int((mask_raw > threshold).sum()))

    return mask_raw > threshold


def save_mask(path: str, mask: np.ndarray):
    ensure_parent(path)
    cv2.imwrite(path, (mask.astype(np.uint8) * 255))


def save_overlay(path: str, rgb_bgr: np.ndarray, mask: np.ndarray):
    if rgb_bgr is None:
        return
    overlay = rgb_bgr.copy()
    color = np.zeros_like(overlay)
    color[:, :, 2] = 255
    alpha = 0.45
    overlay[mask] = cv2.addWeighted(overlay, 1 - alpha, color, alpha, 0)[mask]
    ensure_parent(path)
    cv2.imwrite(path, overlay)


def clean_mask(mask: np.ndarray, open_iter=0, close_iter=1, erode_iter=0):
    """
    Clean object mask.

    For small objects:
        open_iter should usually be 0;
        close_iter can be 1 or 2;
        erode_iter should usually be 0.
    """
    m = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if open_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if erode_iter > 0:
        m = cv2.erode(m, kernel, iterations=erode_iter)

    return m > 0


def expand_mask_depth_aware(
    mask: np.ndarray,
    depth_m: np.ndarray,
    depth_min: float,
    depth_max: float,
    max_expand_px: int = 10,
    kernel_size: int = 3,
    depth_margin: float = 0.05,
    bbox_padding: int = 30,
    close_iter: int = 1,
    min_seed_pixels: int = 10,
):
    """
    Expand an under-sized mask for a small object.

    This is not shape completion. It only accepts additional pixels that:
        1) are near the original mask in 2D,
        2) have valid depth,
        3) have depth compatible with the original masked object region.

    This keeps the result observation-derived and metric.
    """
    mask = mask.astype(bool)
    valid_depth = (
        np.isfinite(depth_m)
        & (depth_m > depth_min)
        & (depth_m < depth_max)
    )

    seed = mask & valid_depth
    seed_count = int(seed.sum())

    if seed_count < min_seed_pixels:
        print(
            f"[WARNING] Too few valid seed pixels in original mask: {seed_count}. "
            "Expansion may be unreliable. Check whether mask is inverted or misaligned."
        )
        return mask

    seed_depth = depth_m[seed]
    d_lo, d_hi = np.percentile(seed_depth, [1, 99])
    d_lo = max(float(d_lo - depth_margin), depth_min)
    d_hi = min(float(d_hi + depth_margin), depth_max)

    ys, xs = np.where(mask)
    h, w = mask.shape
    y1 = max(int(ys.min()) - bbox_padding, 0)
    y2 = min(int(ys.max()) + bbox_padding + 1, h)
    x1 = max(int(xs.min()) - bbox_padding, 0)
    x2 = min(int(xs.max()) + bbox_padding + 1, w)

    bbox_gate = np.zeros_like(mask, dtype=bool)
    bbox_gate[y1:y2, x1:x2] = True

    depth_gate = (
        valid_depth
        & (depth_m >= d_lo)
        & (depth_m <= d_hi)
        & bbox_gate
    )

    m = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    if close_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    expanded = m > 0

    for _ in range(max_expand_px):
        dilated = cv2.dilate(
            expanded.astype(np.uint8),
            kernel,
            iterations=1
        ).astype(bool)

        candidates = dilated & (~expanded)
        accepted = candidates & depth_gate
        expanded = expanded | accepted

    print("\n[Depth-aware mask expansion]")
    print("  original valid seed pixels:", seed_count)
    print("  robust seed depth range:",
          f"{float(seed_depth.min()):.4f} - {float(seed_depth.max()):.4f} m")
    print("  accepted depth gate:",
          f"{d_lo:.4f} - {d_hi:.4f} m")
    print("  mask pixels before:", int(mask.sum()))
    print("  mask pixels after:", int(expanded.sum()))

    return expanded


def bilateral_filter_depth(depth: np.ndarray, mask: np.ndarray,
                           sigma_color=0.03, sigma_space=5):
    """
    Light depth denoising without inferring back-facing or occluded regions.
    """
    valid = (depth > 0) & mask
    depth_filled = depth.copy()
    depth_filled[~valid] = 0.0

    filtered = cv2.bilateralFilter(
        depth_filled.astype(np.float32),
        d=5,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    filtered[~valid] = 0.0
    return filtered


def upsample_visible_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    rgb,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float = 1.0,
):
    """
    Upsample only the visible depth region for figure-quality point cloud.

    This creates interpolated visible-surface points for visualization.
    It does not complete unobserved/back-facing surfaces.
    """
    if scale <= 1.0:
        return depth, mask, rgb, fx, fy, cx, cy

    h, w = depth.shape
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    valid = ((depth > 0) & mask).astype(np.float32)

    numerator = cv2.resize(
        depth * valid,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )
    denominator = cv2.resize(
        valid,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    depth_up = np.zeros((new_h, new_w), dtype=np.float32)
    good = denominator > 1e-6
    depth_up[good] = numerator[good] / denominator[good]

    mask_up_nearest = cv2.resize(
        mask.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Slightly stricter support to avoid interpolation leakage around borders.
    mask_support = denominator > 0.25
    mask_up = mask_up_nearest & mask_support
    depth_up[~mask_up] = 0.0

    rgb_up = None
    if rgb is not None:
        rgb_up = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Intrinsic scaling. This is sufficient for visualization-level upsampling.
    fx_up = fx * scale
    fy_up = fy * scale
    cx_up = cx * scale
    cy_up = cy * scale

    print("\n[Visible depth upsampling]")
    print("  scale:", scale)
    print("  original size:", (h, w))
    print("  upsampled size:", (new_h, new_w))
    print("  original mask pixels:", int(mask.sum()))
    print("  upsampled mask pixels:", int(mask_up.sum()))

    return depth_up, mask_up, rgb_up, fx_up, fy_up, cx_up, cy_up


def depth_to_point_cloud(depth, mask, rgb, fx, fy, cx, cy,
                         depth_min=0.05, depth_max=3.0):
    """
    Back-project masked depth into a metric object point cloud.

    Camera coordinate:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        z = depth
    """
    valid = (
        mask
        & np.isfinite(depth)
        & (depth > depth_min)
        & (depth < depth_max)
    )

    v, u = np.where(valid)

    pcd = o3d.geometry.PointCloud()

    if len(v) == 0:
        return pcd

    z = depth[v, u]
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy

    points = np.stack([x, y, z], axis=1)
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if rgb is not None:
        if rgb.shape[:2] != depth.shape:
            rgb = cv2.resize(
                rgb,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        colors = rgb[v, u, ::-1].astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd


def light_clean_dense_pcd(pcd, nb_neighbors=20, std_ratio=2.8):
    """
    Light outlier removal for dense visualization point cloud.
    For small objects, do not make this too aggressive.
    """
    if len(pcd.points) == 0:
        raise RuntimeError("Empty point cloud after masking and depth filtering.")

    if len(pcd.points) < nb_neighbors + 5:
        print("[WARNING] Too few points for statistical outlier removal. Skipping.")
        return pcd

    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd_clean


def prepare_mesh_point_cloud(
    pcd_dense,
    voxel_size=0.0008,
    use_radius_filter=False,
    radius_nb_points=4,
):
    """
    Prepare a point cloud for BPA meshing.

    For small objects:
        keep voxel_size small;
        disable radius filtering unless there are obvious flying points.
    """
    if len(pcd_dense.points) == 0:
        raise RuntimeError("Empty dense point cloud.")

    pcd = pcd_dense

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if use_radius_filter:
        radius = max(voxel_size * 4.0, 0.002)
        if len(pcd.points) > radius_nb_points * 2:
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=radius_nb_points,
                radius=radius
            )

    return pcd


def estimate_normals(pcd, normal_radius):
    """
    Estimate normals and orient them toward the camera.
    """
    if len(pcd.points) == 0:
        raise RuntimeError("Cannot estimate normals for empty point cloud.")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=40
        )
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.0])
    )
    return pcd


def estimate_ball_radius(pcd, factor=2.8):
    """
    Estimate BPA radius from nearest-neighbor distances.
    """
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0]

    if len(distances) == 0:
        raise RuntimeError("Cannot estimate point spacing.")

    median_dist = np.median(distances)
    radius = median_dist * factor
    return float(radius)


def reconstruct_ball_pivoting(pcd, radius):
    """
    Ball Pivoting visible-surface reconstruction.
    """
    radii = o3d.utility.DoubleVector([
        radius,
        radius * 1.5,
        radius * 2.5,
        radius * 4.0
    ])

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )
    return mesh


def reconstruct_alpha_shape(pcd, alpha):
    """
    Fallback option. Use carefully: a large alpha may over-close the shape.
    """
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )
    return mesh


def cleanup_mesh(mesh,
                 min_cluster_triangles=5,
                 keep_top_k=None,
                 smooth_iter=1):
    """
    Remove tiny triangle islands and lightly smooth the mesh.

    For small objects, min_cluster_triangles should be small, e.g., 3-10.
    """
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) == 0:
        return mesh

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if keep_top_k is not None:
        sorted_ids = np.argsort(-cluster_n_triangles)
        keep_ids = set(sorted_ids[:keep_top_k].tolist())
        remove_mask = np.array([cid not in keep_ids for cid in triangle_clusters])
    else:
        remove_mask = cluster_n_triangles[triangle_clusters] < min_cluster_triangles

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()

    if smooth_iter > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    return mesh


def normalize_color_for_figure(mesh, color=(0.18, 0.42, 0.95)):
    mesh.paint_uniform_color(color)
    return mesh


def print_valid_count(name, depth, mask, depth_min, depth_max):
    valid = (
        mask
        & np.isfinite(depth)
        & (depth > depth_min)
        & (depth < depth_max)
    )
    print(f"[INFO] {name} valid depth pixels:", int(valid.sum()))
    if int(valid.sum()) > 0:
        d = depth[valid]
        print(f"[INFO] {name} depth percentiles:",
              np.percentile(d, [1, 5, 50, 95, 99]))


def main():
    parser = argparse.ArgumentParser(
        description="Observation-derived visible-surface reconstruction for small objects."
    )

    parser.add_argument("--rgb", type=str, default=None, help="RGB image path.")
    parser.add_argument("--depth", type=str, required=True, help="Depth image path.")
    parser.add_argument("--mask", type=str, required=True, help="Object mask path.")
    parser.add_argument("--K", type=str, required=True, help="3x3 camera intrinsic matrix txt.")

    parser.add_argument("--out_mesh", type=str, default="observed_geometry_bpa.ply")
    parser.add_argument("--out_pcd", type=str, default="observed_geometry_points.ply")
    parser.add_argument("--debug_dir", type=str, default="debug_observed_geometry")

    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--depth_min", type=float, default=0.05)
    parser.add_argument("--depth_max", type=float, default=3.0)

    # For small objects, do not erode by default.
    parser.add_argument("--mask_open", type=int, default=0)
    parser.add_argument("--mask_close", type=int, default=1)
    parser.add_argument("--mask_erode", type=int, default=0)

    # Depth-aware mask expansion.
    parser.add_argument("--mask_expand_px", type=int, default=10)
    parser.add_argument("--mask_expand_kernel", type=int, default=3)
    parser.add_argument("--mask_depth_margin", type=float, default=0.05)
    parser.add_argument("--mask_bbox_padding", type=int, default=30)
    parser.add_argument("--min_seed_pixels", type=int, default=10)

    # Visible surface upsampling for figure-quality point cloud.
    parser.add_argument("--upsample_factor", type=float, default=2.0)

    # Dense point cloud cleaning.
    parser.add_argument("--dense_nb_neighbors", type=int, default=20)
    parser.add_argument("--dense_std_ratio", type=float, default=2.8)

    # Meshing parameters.
    parser.add_argument("--voxel_size", type=float, default=0.0008)
    parser.add_argument("--normal_radius", type=float, default=0.008)
    parser.add_argument("--ball_radius_factor", type=float, default=2.8)
    parser.add_argument("--use_radius_filter", action="store_true")
    parser.add_argument("--radius_nb_points", type=int, default=4)

    parser.add_argument("--smooth_iter", type=int, default=1)
    parser.add_argument("--method", type=str, default="bpa", choices=["bpa", "alpha"])
    parser.add_argument("--alpha", type=float, default=0.006)
    parser.add_argument("--min_cluster_triangles", type=int, default=5)

    args = parser.parse_args()

    Path(args.debug_dir).mkdir(parents=True, exist_ok=True)

    fx, fy, cx, cy, _ = load_intrinsics(args.K)

    depth = load_depth(args.depth, args.depth_scale)
    h, w = depth.shape

    rgb = load_rgb(args.rgb, h, w)

    mask_raw = load_mask(args.mask, h, w)
    save_mask(str(Path(args.debug_dir) / "mask_00_raw.png"), mask_raw)
    save_overlay(str(Path(args.debug_dir) / "mask_00_raw_overlay.png"), rgb, mask_raw)

    print_valid_count("raw mask", depth, mask_raw, args.depth_min, args.depth_max)

    mask_clean = clean_mask(
        mask_raw,
        open_iter=args.mask_open,
        close_iter=args.mask_close,
        erode_iter=args.mask_erode
    )
    save_mask(str(Path(args.debug_dir) / "mask_01_clean.png"), mask_clean)
    save_overlay(str(Path(args.debug_dir) / "mask_01_clean_overlay.png"), rgb, mask_clean)

    print_valid_count("clean mask", depth, mask_clean, args.depth_min, args.depth_max)

    if args.mask_expand_px > 0:
        mask_expanded = expand_mask_depth_aware(
            mask=mask_clean,
            depth_m=depth,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            max_expand_px=args.mask_expand_px,
            kernel_size=args.mask_expand_kernel,
            depth_margin=args.mask_depth_margin,
            bbox_padding=args.mask_bbox_padding,
            close_iter=1,
            min_seed_pixels=args.min_seed_pixels
        )
    else:
        mask_expanded = mask_clean

    save_mask(str(Path(args.debug_dir) / "mask_02_expanded.png"), mask_expanded)
    save_overlay(str(Path(args.debug_dir) / "mask_02_expanded_overlay.png"), rgb, mask_expanded)
    print_valid_count("expanded mask", depth, mask_expanded, args.depth_min, args.depth_max)

    # Light filtering only inside the expanded visible region.
    depth_filtered = bilateral_filter_depth(depth, mask_expanded)

    # Direct metric point cloud at original image resolution.
    pcd_direct = depth_to_point_cloud(
        depth_filtered,
        mask_expanded,
        rgb,
        fx, fy, cx, cy,
        depth_min=args.depth_min,
        depth_max=args.depth_max
    )
    print("\n[Point cloud]")
    print("[INFO] direct metric points:", len(pcd_direct.points))

    direct_path = add_suffix(args.out_pcd, "_direct_metric")
    ensure_parent(direct_path)
    o3d.io.write_point_cloud(direct_path, pcd_direct)
    print(f"[DONE] Direct metric point cloud saved to: {direct_path}")

    # Upsample visible surface for Fig. 1 visualization and meshing.
    depth_use, mask_use, rgb_use, fx_use, fy_use, cx_use, cy_use = upsample_visible_depth(
        depth_filtered,
        mask_expanded,
        rgb,
        fx, fy, cx, cy,
        scale=args.upsample_factor
    )

    pcd_raw = depth_to_point_cloud(
        depth_use,
        mask_use,
        rgb_use,
        fx_use, fy_use, cx_use, cy_use,
        depth_min=args.depth_min,
        depth_max=args.depth_max
    )

    raw_vis_path = add_suffix(args.out_pcd, "_vis_raw_dense")
    ensure_parent(raw_vis_path)
    o3d.io.write_point_cloud(raw_vis_path, pcd_raw)
    print(f"[DONE] Visualization raw dense point cloud saved to: {raw_vis_path}")
    print("[INFO] visualization raw dense points:", len(pcd_raw.points))

    pcd_dense = light_clean_dense_pcd(
        pcd_raw,
        nb_neighbors=args.dense_nb_neighbors,
        std_ratio=args.dense_std_ratio
    )

    clean_vis_path = add_suffix(args.out_pcd, "_vis_clean_dense")
    ensure_parent(clean_vis_path)
    o3d.io.write_point_cloud(clean_vis_path, pcd_dense)
    print(f"[DONE] Visualization clean dense point cloud saved to: {clean_vis_path}")
    print("[INFO] visualization clean dense points:", len(pcd_dense.points))

    # Mesh point cloud.
    pcd_mesh = prepare_mesh_point_cloud(
        pcd_dense,
        voxel_size=args.voxel_size,
        use_radius_filter=args.use_radius_filter,
        radius_nb_points=args.radius_nb_points
    )
    print("[INFO] points for meshing:", len(pcd_mesh.points))

    if len(pcd_mesh.points) == 0:
        raise RuntimeError(
            "No points left for meshing. Try smaller --voxel_size, "
            "larger --mask_expand_px, or disable radius filtering."
        )

    pcd_mesh = estimate_normals(pcd_mesh, normal_radius=args.normal_radius)

    ensure_parent(args.out_pcd)
    o3d.io.write_point_cloud(args.out_pcd, pcd_mesh)
    print(f"[DONE] Meshing point cloud saved to: {args.out_pcd}")

    if args.method == "bpa":
        ball_radius = estimate_ball_radius(pcd_mesh, factor=args.ball_radius_factor)
        print(f"[INFO] Estimated BPA radius: {ball_radius:.6f} m")
        mesh = reconstruct_ball_pivoting(pcd_mesh, ball_radius)
    else:
        print(f"[INFO] Alpha value: {args.alpha:.6f} m")
        mesh = reconstruct_alpha_shape(pcd_mesh, alpha=args.alpha)

    mesh = cleanup_mesh(
        mesh,
        min_cluster_triangles=args.min_cluster_triangles,
        keep_top_k=None,
        smooth_iter=args.smooth_iter
    )

    mesh = normalize_color_for_figure(mesh)

    ensure_parent(args.out_mesh)
    o3d.io.write_triangle_mesh(args.out_mesh, mesh)

    print(f"[DONE] Mesh saved to: {args.out_mesh}")
    print("[INFO] final meshing points:", len(pcd_mesh.points))
    print("[INFO] final mesh vertices:", len(mesh.vertices))
    print("[INFO] final mesh triangles:", len(mesh.triangles))

    print("\n[Outputs to check]")
    print("  Direct metric pcd:", direct_path)
    print("  Fig. visualization dense pcd:", clean_vis_path)
    print("  Meshing pcd:", args.out_pcd)
    print("  Mesh:", args.out_mesh)
    print("  Debug masks:", args.debug_dir)


if __name__ == "__main__":
    main()