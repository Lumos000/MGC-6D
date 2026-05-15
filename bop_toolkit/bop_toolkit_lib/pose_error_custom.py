# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Implementation of the pose error functions described in:
Hodan, Michel et al., "BOP: Benchmark for 6D Object Pose Estimation", ECCV'18
Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW'16
"""

import math
import numpy as np
from scipy import spatial

from bop_toolkit_lib import misc
from bop_toolkit_lib import visibility


# Standard Library
from typing import Optional
from dataclasses import dataclass


@dataclass
class POSE_ERROR_VSD_ARGS:
    # all args in pose_error.vsd
    R_e: Optional[np.ndarray] = None
    t_e: Optional[np.ndarray] = None
    R_g: Optional[np.ndarray] = None
    t_g: Optional[np.ndarray] = None
    depth_im: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    vsd_deltas: Optional[float] = None
    vsd_taus: Optional[list] = None
    vsd_normalized_by_diameter: Optional[bool] = None
    diameter: Optional[float] = None
    obj_id: Optional[int] = None
    step: Optional[str] = None

    def from_dict(self, data):
        for key, value in data.items():
            setattr(self, key, value)
        return self

    def to_file(self, path):
        for key, value in self.__dict__.items():
            if value is None:
                raise ValueError("Field {} is None".format(key))
            value = np.array(value)
        np.savez(path, self.__dict__)

    def from_file(path):
        args = POSE_ERROR_VSD_ARGS()
        data = np.load(path, allow_pickle=True)
        data = data["arr_0"].item()
        for key, value in data.items():
            if key == "vsd_taus":
                setattr(args, key, list(value))
            if key == "vsd_normalized_by_diameter":
                setattr(args, key, bool(value))
            if key == "step":
                setattr(args, key, str(value))
            if key == "obj_id":
                setattr(args, key, int(value))
            else:
                setattr(args, key, value)
        return args


def vsd(
    R_est,
    t_est,
    R_gt,
    t_gt,
    depth_test,
    K,
    delta,
    taus,
    normalized_by_diameter,
    diameter,
    renderer,
    obj_id,
    cost_type="step",
):
    """Visible Surface Discrepancy -- by Hodan, Michel et al. (ECCV 2018).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param depth_test: hxw ndarray with the test depth image.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param taus: A list of misalignment tolerance values.
    :param normalized_by_diameter: Whether to normalize the pixel-wise distances
        by the object diameter.
    :param diameter: Object diameter.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :param cost_type: Type of the pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16
        'step' - Used for SIXD Challenge 2017 onwards.
    :return: List of calculated errors (one for each misalignment tolerance).
    """
    # Render depth images of the model in the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth_est = renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)["depth"]
    depth_gt = renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)["depth"]

    # Convert depth images to distance images.
    dist_test = misc.depth_im_to_dist_im_fast(depth_test, K)
    dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im_fast(depth_est, K)

    # Visibility mask of the model in the ground-truth pose.
    visib_gt = visibility.estimate_visib_mask_gt(
        dist_test, dist_gt, delta, visib_mode="bop19"
    )

    # Visibility mask of the model in the estimated pose.
    visib_est = visibility.estimate_visib_mask_est(
        dist_test, dist_est, visib_gt, delta, visib_mode="bop19"
    )

    # Intersection and union of the visibility masks.
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()

    # Pixel-wise distances.
    dists = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])

    # Normalization of pixel-wise distances by object diameter.
    if normalized_by_diameter:
        dists /= diameter

    # Calculate VSD for each provided value of the misalignment tolerance.
    if visib_union_count == 0:
        errors = [1.0] * len(taus)
    else:
        errors = []
        for tau in taus:
            # Pixel-wise matching cost.
            if cost_type == "step":
                costs = dists >= tau
            elif cost_type == "tlinear":  # Truncated linear function.
                costs = dists / tau
                costs[costs > 1.0] = 1.0
            else:
                raise ValueError("Unknown pixel matching cost.")

            e = (np.sum(costs) + visib_comp_count) / float(visib_union_count)
            errors.append(e)

    return errors

def opengl_coord_change(pts3D):
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    pts3D = pts3D.dot(coord_change_mat.T)
    return pts3D


def decompose_pose_matrix(pose):
    """Decompose 4x4 pose matrix into rotation and translation.

    :param pose: 4x4 ndarray with the pose matrix
    :return: tuple (R, t) where R is 3x3 rotation matrix and t is 3x1 translation vector
    """
    if pose.shape == (4, 4):
        R = pose[:3, :3]
        t = pose[:3, 3:4]  # Keep as 2D array with shape (3,1)
        return R, t
    else:
        raise ValueError("Pose matrix must be 4x4")


def mssd(pose_est=None, pose_gt=None, R_est=None, t_est=None, R_gt=None, t_gt=None, pts=None, syms=None):
    """Maximum Symmetry-Aware Surface Distance (MSSD).

    Accepts either 4x4 pose matrices or separate R, t inputs.

    :param pose_est: 4x4 ndarray with the estimated pose matrix (optional)
    :param pose_gt: 4x4 ndarray with the ground-truth pose matrix (optional)
    :param R_est: 3x3 ndarray with the estimated rotation matrix (optional)
    :param t_est: 3x1 ndarray with the estimated translation vector (optional)
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix (optional)
    :param t_gt: 3x1 ndarray with the ground-truth translation vector (optional)
    :param pts: nx3 ndarray with 3D model points
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix
      - 't': 3x1 ndarray with the translation vector
      or
      - 'pose': 4x4 ndarray with the pose matrix
    :return: The calculated error
    """
    # Handle pose matrix inputs
    if pose_est is not None:
        R_est, t_est = decompose_pose_matrix(pose_est)
    if pose_gt is not None:
        R_gt, t_gt = decompose_pose_matrix(pose_gt)

    # Verify we have all needed transforms
    if R_est is None or t_est is None or R_gt is None or t_gt is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")

    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    es = []

    for sym in syms:
        if "pose" in sym:
            R_sym, t_sym = decompose_pose_matrix(sym["pose"])
        else:
            R_sym, t_sym = sym["R"], sym["t"]

        R_gt_sym = R_gt.dot(R_sym)
        t_gt_sym = R_gt.dot(t_sym) + t_gt
        pts_gt_sym = misc.transform_pts_Rt(pts, R_gt_sym, t_gt_sym)
        es.append(np.linalg.norm(pts_est - pts_gt_sym, axis=1).max())

    return min(es)

def mspd(pose_est=None, pose_gt=None, R_est=None, t_est=None, R_gt=None, t_gt=None, K=None, pts=None, syms=None):
    """Maximum Symmetry-Aware Projection Distance (MSPD).

    Accepts either 4x4 pose matrices or separate R, t inputs.

    :param pose_est: 4x4 ndarray with the estimated pose matrix (optional)
    :param pose_gt: 4x4 ndarray with the ground-truth pose matrix (optional)
    :param R_est: 3x3 ndarray with the estimated rotation matrix (optional)
    :param t_est: 3x1 ndarray with the estimated translation vector (optional)
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix (optional)
    :param t_gt: 3x1 ndarray with the ground-truth translation vector (optional)
    :param K: 3x3 ndarray with the intrinsic camera matrix
    :param pts: nx3 ndarray with 3D model points
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix
      - 't': 3x1 ndarray with the translation vector
      or
      - 'pose': 4x4 ndarray with the pose matrix
    :return: The calculated error
    """
    # Handle pose matrix inputs
    if pose_est is not None:
        R_est, t_est = decompose_pose_matrix(pose_est)
    if pose_gt is not None:
        R_gt, t_gt = decompose_pose_matrix(pose_gt)

    # Verify we have all needed transforms
    if R_est is None or t_est is None or R_gt is None or t_gt is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")

    proj_est = misc.project_pts(pts, K, R_est, t_est)
    es = []

    for sym in syms:
        if "pose" in sym:
            R_sym, t_sym = decompose_pose_matrix(sym["pose"])
        else:
            R_sym, t_sym = sym["R"], sym["t"]

        R_gt_sym = R_gt.dot(R_sym)
        t_gt_sym = R_gt.dot(t_sym) + t_gt
        proj_gt_sym = misc.project_pts(pts, K, R_gt_sym, t_gt_sym)
        es.append(np.linalg.norm(proj_est - proj_gt_sym, axis=1).max())

    return min(es)

def mssd_est(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig, syms):
    """Maximum Symmetry-Aware Surface Distance (MSSD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts_est_orig: nx3 ndarray with 3D model points for estimated pose.
    :param pts_gt_orig: mx3 ndarray with 3D model points for ground truth pose.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    pts_est = misc.transform_pts_Rt(pts_est_orig, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        pts_gt_sym = misc.transform_pts_Rt(pts_gt_orig, R_gt_sym, t_gt_sym)
        # Build KD-tree for fast nearest neighbor search
        tree = spatial.cKDTree(pts_gt_sym)

        # Find distances to nearest neighbors
        distances, _ = tree.query(pts_est)

        # Maximum distance for this symmetry
        es.append(np.max(distances))
        # es.append(np.linalg.norm(pts_est - pts_gt_sym, axis=1).max())

    return min(es)


def mspd_est(K, R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig, syms):
    """Maximum Symmetry-Aware Projection Distance (MSPD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param pts_est_orig: nx3 ndarray with 3D model points for estimated pose.
    :param pts_gt_orig: mx3 ndarray with 3D model points for ground truth pose.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    proj_est = misc.project_pts(pts_est_orig, K, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        proj_gt_sym = misc.project_pts(pts_gt_orig, K, R_gt_sym, t_gt_sym)
        # es.append(np.linalg.norm(proj_est - proj_gt_sym, axis=1).max())
        # Build KD-tree for fast nearest neighbor search in 2D
        tree = spatial.cKDTree(proj_gt_sym)

        # Find distances to nearest neighbors
        distances, _ = tree.query(proj_est)

        # Maximum distance for this symmetry
        es.append(np.max(distances))
    return min(es)


def adi(pose_est=None, pose_gt=None, R_est=None, t_est=None, R_gt=None, t_gt=None, pts=None):
    """Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).

    Accepts either 4x4 pose matrices or separate R, t inputs.

    :param pose_est: 4x4 ndarray with the estimated pose matrix (optional)
    :param pose_gt: 4x4 ndarray with the ground-truth pose matrix (optional)
    :param R_est: 3x3 ndarray with the estimated rotation matrix (optional)
    :param t_est: 3x1 ndarray with the estimated translation vector (optional)
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix (optional)
    :param t_gt: 3x1 ndarray with the ground-truth translation vector (optional)
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    # Handle pose matrix inputs
    if pose_est is not None:
        R_est, t_est = decompose_pose_matrix(pose_est)
    if pose_gt is not None:
        R_gt, t_gt = decompose_pose_matrix(pose_gt)

    # Verify we have all needed transforms
    if R_est is None or t_est is None or R_gt is None or t_gt is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")

    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e

def add(pose_est=None, pose_gt=None, R_est=None, t_est=None, R_gt=None, t_gt=None, pts=None):
    """Average Distance of Model Points for objects with no indistinguishable
    views - by Hinterstoisser et al. (ACCV'12).

    Accepts either 4x4 pose matrices or separate R, t inputs.

    :param pose_est: 4x4 ndarray with the estimated pose matrix (optional)
    :param pose_gt: 4x4 ndarray with the ground-truth pose matrix (optional)
    :param R_est: 3x3 ndarray with the estimated rotation matrix (optional)
    :param t_est: 3x1 ndarray with the estimated translation vector (optional)
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix (optional)
    :param t_gt: 3x1 ndarray with the ground-truth translation vector (optional)
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    # Handle pose matrix inputs
    if pose_est is not None:
        R_est, t_est = decompose_pose_matrix(pose_est)
    if pose_gt is not None:
        R_gt, t_gt = decompose_pose_matrix(pose_gt)

    # Verify we have all needed transforms
    if R_est is None or t_est is None or R_gt is None or t_gt is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")

    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

def calculate_3d_iou(pts_est, pts_gt):
    """
    Calculate 3D IOU between two point clouds using their bounding boxes

    Args:
        pts_est: nx3 ndarray of estimated 3D points
        pts_gt: mx3 ndarray of ground truth 3D points

    Returns:
        float: IOU score between 0 and 1
    """
    # Calculate min/max bounds for each point cloud
    min_est = np.min(pts_est, axis=0)
    max_est = np.max(pts_est, axis=0)
    min_gt = np.min(pts_gt, axis=0)
    max_gt = np.max(pts_gt, axis=0)

    # Calculate volumes
    vol_est = np.prod(max_est - min_est)
    vol_gt = np.prod(max_gt - min_gt)

    # Calculate intersection bounds
    min_intersection = np.maximum(min_est, min_gt)
    max_intersection = np.minimum(max_est, max_gt)

    # Check if boxes overlap
    if np.any(max_intersection < min_intersection):
        return 0.0

    # Calculate intersection volume
    vol_intersection = np.prod(np.maximum(0, max_intersection - min_intersection))

    # Calculate union volume
    vol_union = vol_est + vol_gt - vol_intersection

    # Calculate IOU
    iou = vol_intersection / vol_union

    return (iou * 100.0)


def chamfer_distance(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig):
    """Calculate the Chamfer distance between two point clouds.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts_est_orig: nx3 ndarray with 3D model points for estimated pose.
    :param pts_gt_orig: mx3 ndarray with 3D model points for ground truth pose.
    :return: Tuple (chamfer_dist, est_to_gt_mean, gt_to_est_mean)
            - chamfer_dist: The Chamfer distance (mean of both directions)
            - est_to_gt_mean: Mean distance from estimated to ground truth points
            - gt_to_est_mean: Mean distance from ground truth to estimated points
    """
    # Transform points using respective poses
    pts_est = misc.transform_pts_Rt(pts_est_orig, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts_gt_orig, R_gt, t_gt)

    # Create KD-trees for efficient nearest neighbor search
    tree_est = spatial.cKDTree(pts_est)
    tree_gt = spatial.cKDTree(pts_gt)

    # Calculate distances from estimated points to nearest ground truth points
    est_to_gt_dists, _ = tree_gt.query(pts_est, k=1)
    est_to_gt_mean = est_to_gt_dists.mean()

    # Calculate distances from ground truth points to nearest estimated points
    gt_to_est_dists, _ = tree_est.query(pts_gt, k=1)
    gt_to_est_mean = gt_to_est_dists.mean()

    # Chamfer distance is the mean of both directional distances
    chamfer_dist = (est_to_gt_mean + gt_to_est_mean) / 2

    return chamfer_dist


def calculate_3d_iou_with_pose(pose1=None, pose2=None, R1=None, t1=None, R2=None, t2=None, pts=None):
    """Calculate 3D IoU between two transformed point clouds.

    Accepts either 4x4 pose matrices or separate R, t inputs.

    :param pose1: 4x4 ndarray with the first pose matrix (optional)
    :param pose2: 4x4 ndarray with the second pose matrix (optional)
    :param R1: 3x3 ndarray with the first rotation matrix (optional)
    :param t1: 3x1 ndarray with the first translation vector (optional)
    :param R2: 3x3 ndarray with the second rotation matrix (optional)
    :param t2: 3x1 ndarray with the second translation vector (optional)
    :param pts1: nx3 ndarray with first set of original 3D points
    :param pts2: mx3 ndarray with second set of original 3D points
    :return: IoU score between 0 and 1
    """
    # Handle pose matrix inputs
    if pose1 is not None:
        R1, t1 = decompose_pose_matrix(pose1)
    if pose2 is not None:
        R2, t2 = decompose_pose_matrix(pose2)

    # Verify we have all needed transforms
    if R1 is None or t1 is None or R2 is None or t2 is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")

    # Transform points using respective poses
    transformed_pts1 = misc.transform_pts_Rt(pts, R1, t1)
    transformed_pts2 = misc.transform_pts_Rt(pts, R2, t2)

    return calculate_3d_iou(transformed_pts1, transformed_pts2)

# def adi_est(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig, opengl=False):
def adi_est(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig):
    """Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts_est_orig: nx3 ndarray with 3D model points for estimated pose.
    :param pts_gt_orig: mx3 ndarray with 3D model points for ground truth pose.
    :return: The calculated error.
    """
    # Transform points using respective poses
    pts_est = misc.transform_pts_Rt(pts_est_orig, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts_gt_orig, R_gt, t_gt)

    iou = calculate_3d_iou(pts_est, pts_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return iou, e

def re(R_est, R_gt):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg].
    return error


def te(t_est, t_gt):
    """Translational Error.

    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :return: The calculated error.
    """
    assert t_est.size == t_gt.size == 3
    error = np.linalg.norm(t_gt - t_est)
    return error


def proj(R_est, t_est, R_gt, t_gt, K, pts):
    """Average distance of projections of object model vertices [px]
    - by Brachmann et al. (CVPR'16).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    proj_est = misc.project_pts(pts, K, R_est, t_est)
    proj_gt = misc.project_pts(pts, K, R_gt, t_gt)
    e = np.linalg.norm(proj_est - proj_gt, axis=1).mean()
    return e


def cou_mask(mask_est, mask_gt):
    """Complement over Union of 2D binary masks.

    :param mask_est: hxw ndarray with the estimated mask.
    :param mask_gt: hxw ndarray with the ground-truth mask.
    :return: The calculated error.
    """
    mask_est_bool = mask_est.astype(np.bool)
    mask_gt_bool = mask_gt.astype(np.bool)

    inter = np.logical_and(mask_gt_bool, mask_est_bool)
    union = np.logical_or(mask_gt_bool, mask_est_bool)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e


def cus(R_est, t_est, R_gt, t_gt, K, renderer, obj_id):
    """Complement over Union of projected 2D masks.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :return: The calculated error.
    """
    # Render depth images of the model at the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth_est = renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)["depth"]
    depth_gt = renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)["depth"]

    # Masks of the rendered model and their intersection and union.
    mask_est = depth_est > 0
    mask_gt = depth_gt > 0
    inter = np.logical_and(mask_gt, mask_est)
    union = np.logical_or(mask_gt, mask_est)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e


def cou_bb(bb_est, bb_gt):
    """Complement over Union of 2D bounding boxes.

    :param bb_est: The estimated bounding box (x1, y1, w1, h1).
    :param bb_gt: The ground-truth bounding box (x2, y2, w2, h2).
    :return: The calculated error.
    """
    e = 1.0 - misc.iou(bb_est, bb_gt)
    return e


def cou_bb_proj(R_est, t_est, R_gt, t_gt, K, renderer, obj_id):
    """Complement over Union of projected 2D bounding boxes.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :return: The calculated error.
    """
    # Render depth images of the model at the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth_est = renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)["depth"]
    depth_gt = renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)["depth"]

    # Masks of the rendered model and their intersection and union
    mask_est = depth_est > 0
    mask_gt = depth_gt > 0

    ys_est, xs_est = mask_est.nonzero()
    bb_est = misc.calc_2d_bbox(xs_est, ys_est, im_size=None, clip=False)

    ys_gt, xs_gt = mask_gt.nonzero()
    bb_gt = misc.calc_2d_bbox(xs_gt, ys_gt, im_size=None, clip=False)

    e = 1.0 - misc.iou(bb_est, bb_gt)
    return e

def np_transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)

    Returns:
        transformed points of size (N, 3)
    """
    rot = g[:, :3, :3]  # (3, 3)
    trans = g[:, :3, 3]  # (3)
    transformed = pts[:, :3] @ np.swapaxes(rot, -1, -2)
    transformed += trans[:, None, :]
    return transformed

def my_project_pts(pts, K, pose):
    """Projects 3D points.

    :param pts: mxnx3 ndarray with the 3D points.
    :param K: mx3x3 ndarray with an intrinsic camera matrix.
    :param R: mx3x3 ndarray with a rotation matrix.
    :param t: mx3x1 ndarray with a translation vector.
    :return: mxnx2 ndarray with 2D image coordinates of the projections.
    """
    assert (pts.shape[2] == 3)

    rot_pcd = np_transform(pose, pts)
    pts_im = rot_pcd @ np.swapaxes(K, -1, -2)
    pts_im /= pts_im[:, :, 2, None]
    return pts_im[:, :, :2]


def my_mssd(R_est: np.ndarray, t_est: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray, pts: np.ndarray,
            syms: np.ndarray):
    """Maximum Symmetry-Aware Surface Distance (MSSD).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    n_syms = syms.shape[0]
    R_gt, t_gt = np.expand_dims(R_gt, axis=0), np.expand_dims(t_gt, axis=0)
    R_est, t_est = np.expand_dims(R_est, axis=0), np.expand_dims(t_est, axis=0)
    pose_est = np.concatenate((R_est, t_est), axis=-1)
    pts = np.expand_dims(pts, axis=0)

    R_gt, t_gt = np.tile(R_gt, (n_syms, 1, 1)), np.tile(t_gt, (n_syms, 1, 1))

    pts_est = np_transform(pose_est, pts)

    R_gt_sym = R_gt @ syms[:, :3, :3]
    t_gt_sym = (R_gt @ syms[:, :3, 3, None]) + t_gt
    pose_sym = np.concatenate((R_gt_sym, t_gt_sym), axis=2)
    pts_gt_sym = np_transform(pose_sym, pts)

    dist = (np.linalg.norm(pts_est - pts_gt_sym, axis=2)).max(axis=1)

    return dist.min()


def my_mspd(R_est: np.ndarray, t_est: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray, K: np.ndarray, pts: np.ndarray,
            syms: np.ndarray):
    '''
    R_est, R_gt: [3,3]
    t_est, t_gt: [3,1]
    K : [3,3]
    pts : [N,3]
    syms : [M,3,4]
    '''
    n_syms = syms.shape[0]
    R_gt, t_gt = np.expand_dims(R_gt, axis=0), np.expand_dims(t_gt, axis=0)
    R_est, t_est = np.expand_dims(R_est, axis=0), np.expand_dims(t_est, axis=0)
    K, pts = np.expand_dims(K, axis=0), np.expand_dims(pts, axis=0)
    R_gt, t_gt = np.tile(R_gt, (n_syms, 1, 1)), np.tile(t_gt, (n_syms, 1, 1))

    pose_est = np.concatenate((R_est, t_est), axis=-1)
    proj_est = my_project_pts(pts, K, pose_est)

    R_gt_sym = R_gt @ syms[:, :3, :3]  # np.swapaxes(syms[:,:3,:3], -1, -2)
    t_gt_sym = (R_gt @ syms[:, :3, 3, None]) + t_gt
    pose_sym = np.concatenate((R_gt_sym, t_gt_sym), axis=2)
    proj_gt_sym = my_project_pts(pts, K, pose_sym)

    dist = (np.linalg.norm(proj_est - proj_gt_sym, axis=2)).max(axis=1)

    return dist.min()
