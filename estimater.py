# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy

import numpy as np
import nvdiffrast.torch as dr

from foundationpose.Utils import *
from foundationpose.datareader import *
import itertools
from foundationpose.learning.training.predict_score import *
from foundationpose.learning.training.predict_pose_refine import *

class Any6D:
    def __init__(self, symmetry_tfs=None, mesh=None, scorer: ScorePredictor | None = None,
                 refiner: PoseRefinePredictor | None = None, glctx=None, debug=0,
                 debug_dir='./debug/', grid_n_views=40, grid_inplane_step=60):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = debug_dir

        self.refiner_dir = os.path.join(self.debug_dir,"refine")
        self.scorer_dir = os.path.join(self.debug_dir,"score")
        if self.debug != 0:
            os.makedirs(debug_dir, exist_ok=True)
            os.makedirs(self.refiner_dir, exist_ok=True)
            os.makedirs(self.scorer_dir, exist_ok=True)

        self.reset_object(mesh=mesh, symmetry_tfs=symmetry_tfs)
        self.make_rotation_grid(min_n_views=grid_n_views, inplane_step=grid_inplane_step)

        self.glctx = glctx if glctx is not None else dr.RasterizeCudaContext()

        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = ScorePredictor()

        if refiner is not None:
            self.refiner = refiner
        else:
            self.refiner = PoseRefinePredictor()

        self.pose_last = None  # Used for tracking; per the centered mesh
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def _runtime_device_str(self):
        """Return the explicit runtime device string for tensor creation."""
        dev = getattr(self, "device", None)
        if dev is not None:
            if isinstance(dev, torch.device):
                return str(dev)
            return str(dev)
        mesh_tensors = getattr(self, "mesh_tensors", None)
        if isinstance(mesh_tensors, dict):
            for v in mesh_tensors.values():
                if torch.is_tensor(v):
                    return str(v.device)
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    def reset_object(self, mesh=None, symmetry_tfs=None):
        """
        加速版 reset_object()
        功能：重置物体模型并构建必要的几何信息，但避免重复计算和不必要的数据拷贝。
        可直接替换原版函数。
        """

        # ========= 1. 检查缓存（同一个 mesh 对象可直接复用） =========
        mesh_id = id(mesh)
        if hasattr(self, "_reset_cache") and mesh_id in self._reset_cache:
            cache = self._reset_cache[mesh_id]
            self.model_center = cache['model_center']
            self.mesh_o3d = cache['mesh_o3d']
            self.diameter = cache['diameter']
            self.vox_size = cache['vox_size']
            self.mesh_tensors = cache['mesh_tensors']
            self.symmetry_tfs = cache['symmetry_tfs']
            return

        # ========= 2. 初始化缓存容器 =========
        if not hasattr(self, "_reset_cache"):
            self._reset_cache = {}

        # ========= 3. 计算包围盒中心 =========
        vertices = np.asarray(mesh.vertices)
        min_xyz = vertices.min(axis=0)
        max_xyz = vertices.max(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2

        # ========= 4. 顶点平移至原点 =========
        mesh = mesh.copy()
        mesh.vertices = vertices - self.model_center.reshape(1, 3)

        # ========= 5. 使用 Open3D Tensor Mesh（如可用） =========
        try:
            tmesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
                o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.Int32)
            )
            # 计算法线（GPU支持会快很多）
            tmesh.compute_vertex_normals()
            mesh_o3d = tmesh.to_legacy()
        except Exception:
            # 兼容旧版 Open3D
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
            mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
            mesh_o3d.compute_vertex_normals()

        # ========= 6. 如果有顶点颜色或纹理，优先使用缓存或简化处理 =========
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            rgb_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(rgb_colors)

        elif hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and hasattr(mesh.visual, 'material') and mesh.visual.material.image is not None:
            img = np.array(mesh.visual.material.image.convert('RGB'))
            uv = np.copy(mesh.visual.uv)
            uv[:, 1] = 1 - uv[:, 1]
            # 使用矢量化方式采样
            uv_x = np.clip((uv[:, 0] * (img.shape[1] - 1)).astype(int), 0, img.shape[1] - 1)
            uv_y = np.clip((uv[:, 1] * (img.shape[0] - 1)).astype(int), 0, img.shape[0] - 1)
            vertex_colors = img[uv_y, uv_x].astype(np.float32) / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        # ========= 7. 快速近似网格直径计算（替代 compute_mesh_diameter） =========
        bbox_diag = np.linalg.norm(max_xyz - min_xyz)
        self.diameter = float(bbox_diag)
        self.vox_size = max(self.diameter / 20.0, 0.003)
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # 度数

        # ========= 8. 延迟 mesh_tensors 构造（惰性加载） =========
        self.mesh = mesh
        self.mesh_tensors = None  # 延迟构造
        def lazy_make_mesh_tensors():
            if self.mesh_tensors is None:
                self.mesh_tensors = make_mesh_tensors(self.mesh)
            return self.mesh_tensors
        self.make_mesh_tensors_lazy = lazy_make_mesh_tensors

        # ========= 9. 设置对称矩阵 =========
        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4, device='cuda', dtype=torch.float32)[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float32)

        # ========= 10. 缓存结果以加速后续调用 =========
        self._reset_cache[mesh_id] = {
            'model_center': self.model_center,
            'mesh_o3d': mesh_o3d,
            'diameter': self.diameter,
            'vox_size': self.vox_size,
            'mesh_tensors': self.mesh_tensors,
            'symmetry_tfs': self.symmetry_tfs
        }

        self.mesh_o3d = mesh_o3d

    def get_tf_to_centered_mesh(self):
        runtime_device = self._runtime_device_str()
        tf_to_center = torch.eye(4, dtype=torch.float, device=runtime_device)
        tf_to_center[:3, 3] = -torch.as_tensor(self.model_center.copy(), dtype=torch.float, device=runtime_device)
        return tf_to_center

    def to_device(self, s='cuda:0'):
        self.device = str(s)
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k]
            if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
                # logging.info(f"Moving {k} to device {s}")
                self.__dict__[k] = self.__dict__[k].to(s)

        # mesh_tensors is lazily constructed in reset_object(); make sure it exists first.
        if self.mesh_tensors is None and hasattr(self, 'make_mesh_tensors_lazy'):
            self.mesh_tensors = self.make_mesh_tensors_lazy()

        if self.mesh_tensors is not None:
            for k in self.mesh_tensors:
                # logging.info(f"Moving {k} to device {s}")
                self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
        if self.refiner is not None:
            self.refiner.model.to(s)
        if self.scorer is not None:
            self.scorer.model.to(s)
        if self.glctx is not None:
            self.glctx = dr.RasterizeCudaContext(s)

    def make_rotation_grid(self, min_n_views=40, inplane_step=60):
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        # logging.info(f'cam_in_obs:{cam_in_obs.shape}')
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        # logging.info(f"rot_grid:{rot_grid.shape}")
        rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        # logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
        self.rot_grid = torch.as_tensor(rot_grid, device=self._runtime_device_str(), dtype=torch.float)
        # logging.info(f"self.rot_grid: {self.rot_grid.shape}")

    def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None,initial_center=False):
        '''
        @scene_pts: torch tensor (N,3)
        '''
        ob_in_cams = self.rot_grid.clone()
        if initial_center:
            center = self.guess_translation_bounding_box(depth=depth, mask=mask, K=K)
        else:
            center = self.guess_translation(depth=depth, mask=mask, K=K)
        runtime_device = self._runtime_device_str()
        ob_in_cams[:, :3, 3] = torch.tensor(center, device=runtime_device, dtype=torch.float).reshape(1, 3)
        return ob_in_cams

    def guess_translation_bounding_box(self, depth, mask, K):
        xyz_map = depth2xyzmap(depth, K)

        xyz_map[mask == False] = 0
        points = xyz_map[mask].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=int(np.array(points).shape[0] * 0.01), std_ratio=2.0)

        pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=20))
        pcd_clean.orient_normals_consistent_tangent_plane(k=1000)
        obb_pcd_clean = pcd_clean.get_oriented_bounding_box()

        center = obb_pcd_clean.get_center()

        # if self.debug >= 2:
        #     o3d.io.write_point_cloud(f'{self.debug_dir}/points.ply', pcd)
        #     # Save bounding box for visualization
        #     bbox_points = np.asarray(obb.get_box_points())
        #     bbox_pcd = o3d.geometry.PointCloud()
        #     bbox_pcd.points = o3d.utility.Vector3dVector(bbox_points)
        #     o3d.io.write_point_cloud(f'{self.debug_dir}/bbox.ply', bbox_pcd)

        return np.asarray(center)

    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            # logging.info(f'mask is all zero')
            return np.zeros((3))
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            # logging.info(f"valid is empty")
            return np.zeros((3))

        zc = np.median(depth[valid])
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

        # if self.debug >= 2:
        #     pcd = toOpen3dCloud(center.reshape(1, 3))
        #     o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

        return center.reshape(3)

    def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, name=None, no_center=False, initial_center=False, init_pose=None):
        '''Copmute pose from given pts to self.pcd
        @init_pose: optional (4,4) or (N,4,4) numpy array of initial pose hypotheses to prepend
        '''
        set_seed(0)

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()
            else:
                self.glctx = glctx

        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')

        depth[ob_mask==False] = 0

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None, initial_center=initial_center)
        poses = poses.data.cpu().numpy()
        if initial_center:
            center = self.guess_translation_bounding_box(depth=depth, mask=ob_mask, K=K)
        else:
            center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device='cuda')

        if init_pose is not None:
            init_t = torch.as_tensor(init_pose, device='cuda', dtype=torch.float)
            if init_t.ndim == 2:
                init_t = init_t.unsqueeze(0)
            poses = torch.cat([init_t, poses], dim=0)

        add_errs = self.compute_add_err_to_gt_pose(poses)
        # logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

        xyz_map = depth2xyzmap(depth, K)
        poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                          glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                          get_vis=self.debug >= 2)
        if vis is not None:
            imageio.imwrite(f'{self.refiner_dir}/vis_refiner.png', vis)

        scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter,
                                          get_vis=self.debug >= 2)
        if vis is not None:
            imageio.imwrite(f'{self.scorer_dir}/vis_score.png', vis)

        add_errs = self.compute_add_err_to_gt_pose(poses)
        # logging.info(f"final, add_errs min:{add_errs.min()}")

        ids = torch.as_tensor(scores).argsort(descending=True)
        # logging.info(f'sort ids:{ids}')
        scores = scores[ids]
        poses = poses[ids]

        # logging.info(f'sorted scores:{scores}')

        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[0]
        self.best_id = ids[0]

        self.poses = poses
        self.scores = scores
        if no_center:
            return poses[0].data.cpu().numpy()
        else:
            return best_pose.data.cpu().numpy()


    @staticmethod
    def _obb_align_with_flip_disambiguation(pcd_clean, obb_mesh_R, obb_pcd_R, obb_pcd_center,
                                            mesh_extent=None):
        """Try all 8 axis-flip combinations of OBB alignment and pick the one
        whose aligned point cloud OBB axes best match the mesh OBB.
        *mesh_extent*: np array (3,) of mesh OBB extent for extent-order scoring.
        Returns the best-aligned point cloud (in-place rotation applied)."""
        sign_combos = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        ], dtype=np.float64)

        best_score = -np.inf
        best_R_apply = None
        mesh_ext_sorted = np.sort(mesh_extent) if mesh_extent is not None else None

        for signs in sign_combos:
            flip = np.diag(signs)
            R_candidate = (obb_mesh_R @ flip) @ obb_pcd_R.T
            if np.linalg.det(R_candidate) < 0:
                R_candidate = (obb_mesh_R @ (-flip)) @ obb_pcd_R.T

            pcd_test = copy.deepcopy(pcd_clean)
            pcd_test.rotate(R_candidate, center=obb_pcd_center)
            try:
                obb_test = pcd_test.get_oriented_bounding_box(robust=True)
            except Exception:
                continue
            ext_test = np.asarray(obb_test.extent)

            diag_dominance = np.abs(np.diag(np.asarray(obb_test.R))).sum()
            score = diag_dominance

            if mesh_ext_sorted is not None:
                ext_sorted = np.sort(ext_test)
                denom = np.maximum(mesh_ext_sorted, 1e-8)
                ratio_diff = np.abs(ext_sorted / denom - 1.0)
                score -= ratio_diff.sum()

            if score > best_score:
                best_score = score
                best_R_apply = R_candidate

        if best_R_apply is not None:
            pcd_clean.rotate(best_R_apply, center=obb_pcd_center)
        else:
            pcd_clean.rotate(obb_mesh_R @ obb_pcd_R.T, center=obb_pcd_center)
        return pcd_clean

    def register_any6d(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, name=None, refinement=True, axis_align=True, coarse_est=True, init_pose=None,
                       scale_range=(0.85, 1.15), scale_volume_tol=0.3):
        '''Copmute pose from given pts to self.pcd
        @pts: (N,3) np array, downsampled scene points
        '''
        set_seed(0)

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()
            else:
                self.glctx = glctx

        runtime_device = self._runtime_device_str()
        depth = erode_depth(depth, radius=2, device=runtime_device)
        depth = bilateral_filter_depth(depth, radius=2, device=runtime_device)
        xyz_map = depth2xyzmap(depth, K)

        xyz_map[ob_mask == False] = 0
        points = xyz_map[ob_mask].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile([0.529, 0.808, 0.922], (len(pcd.points), 1)))


        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=int(points.shape[0] * 0.01), std_ratio=2.0)
        pcd_clean.translate(-pcd_clean.get_oriented_bounding_box(robust=True).center)

        pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=20))
        pcd_clean.orient_normals_consistent_tangent_plane(k=1000)
        obb_pcd_clean = pcd_clean.get_oriented_bounding_box()
        obb_pcd_clean.color = (0, 0, 0)

        if coarse_est:
            mesh_pcd = copy.deepcopy(self.mesh_o3d)
            pcd_ = mesh_pcd.sample_points_uniformly(number_of_points=100000)
            cl, _ = pcd_.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            obb_mesh = cl.get_oriented_bounding_box()
            obb_mesh.color = (0, 0, 1)

            if axis_align:
                pcd_clean = self._obb_align_with_flip_disambiguation(
                    pcd_clean, np.asarray(obb_mesh.R), np.asarray(obb_pcd_clean.R),
                    np.asarray(obb_pcd_clean.center),
                    mesh_extent=np.asarray(obb_mesh.extent))
            obb_pcd_clean = pcd_clean.get_oriented_bounding_box(robust=True)
            obb_pcd_clean.color = (0, 1, 0)


            extent_pcd_clean = obb_pcd_clean.extent
            extent_mesh = obb_mesh.extent

            ratio, best_perm, best_iou = find_best_ratio_combination(extent_pcd_clean, extent_mesh, obb_pcd_clean, obb_mesh)
            mesh_pcd.scale(ratio[1], center=obb_mesh.center)

            obb_mesh = mesh_pcd.get_oriented_bounding_box(robust=True)
            obb_mesh.color = (1, 0, 0)

            mesh = copy.deepcopy(self.mesh)
            mesh.vertices = np.asarray(mesh_pcd.vertices)
            self.reset_object(mesh=mesh, symmetry_tfs=self.symmetry_tfs)
            self.mesh.export(os.path.join(self.debug_dir, f'refine_init_mesh_{name}.obj'))

        valid_indices = np.argwhere(ob_mask)  # Get (h, w) coordinates where ob_mask is True
        selected_indices = valid_indices[ind]  # Use the indices from the outlier filtering to get (h, w)

        depth_mask = np.zeros((xyz_map.shape[:2]), dtype=bool)
        depth_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
        xyz_map[depth_mask == False] = 0

        depth = xyz_map[..., -1]

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            # logging.info(f'valid too small, return')
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
        poses = poses.data.cpu().numpy()
        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        poses = torch.as_tensor(poses, device=runtime_device, dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device=runtime_device)
        if init_pose is not None:
            init_pose_tensor = torch.as_tensor(init_pose, device=runtime_device, dtype=torch.float)
            if init_pose_tensor.ndim == 2:
                init_pose_tensor = init_pose_tensor.unsqueeze(0)
            poses = torch.cat([init_pose_tensor, poses], dim=0)

        xyz_map = depth2xyzmap(depth, K)

        poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                          glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                          get_vis=self.debug >= 2)

        if vis is not None:
            imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_1_consider_pose_{name}.png', vis)

        scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter,
                                          get_vis=self.debug >= 2)
        if vis is not None:
            imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_1_consider_pose_{name}.png', vis)

        ids = torch.as_tensor(scores).argsort(descending=True)
        # logging.info(f'sort ids:{ids}')
        scores = scores[ids]
        poses = poses[ids]
        best_pose = poses[0].data.cpu().numpy()



        if refinement:
            cam_in_ob = np.linalg.inv(best_pose)
            points = xyz_map[ob_mask].reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([1,0,0])
            pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=int(points.shape[0] * 0.01), std_ratio=2.0)
            pcd_clean.transform(cam_in_ob)

            mesh_pcd = (copy.deepcopy(self.mesh_o3d))
            obb_mesh = mesh_pcd.get_oriented_bounding_box(robust=True)
            obb_mesh.color = (0, 1, 0)

            obb_pcd_clean = pcd_clean.get_oriented_bounding_box(robust=True)
            obb_pcd_clean.color = (0, 1, 0)

            if axis_align:
                pcd_clean = self._obb_align_with_flip_disambiguation(
                    pcd_clean, np.asarray(obb_mesh.R), np.asarray(obb_pcd_clean.R),
                    np.asarray(obb_pcd_clean.center),
                    mesh_extent=np.asarray(obb_mesh.extent))
            obb_pcd_clean = pcd_clean.get_oriented_bounding_box(robust=True)
            obb_pcd_clean.color = (1, 0, 0)

            # o3d.visualization.draw_geometries([obb_mesh, pcd_clean, mesh_pcd,coordinate_frame, obb_pcd_clean])

            extent_pcd_clean = obb_pcd_clean.extent
            extent_mesh = obb_mesh.extent
            ratio = extent_pcd_clean / extent_mesh
            mesh_pcd.translate(obb_pcd_clean.center - obb_mesh.center)
            mesh_pcd.vertices = o3d.utility.Vector3dVector(np.array(mesh_pcd.vertices) * ratio[None])
            mesh_pcd.translate(obb_mesh.center - obb_pcd_clean.center)

            obb_mesh = mesh_pcd.get_oriented_bounding_box(robust=True)
            obb_mesh.color = (0, 1, 0)
            # o3d.visualization.draw_geometries([obb_mesh, pcd_clean, mesh_pcd,coordinate_frame, obb_pcd_clean])

            mesh = copy.deepcopy(self.mesh)
            mesh.vertices = np.asarray(mesh_pcd.vertices)
            self.reset_object(mesh=mesh, symmetry_tfs=self.symmetry_tfs)

            poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
            poses = poses.data.cpu().numpy()
            runtime_device = self._runtime_device_str()
            poses = torch.as_tensor(poses, device=runtime_device, dtype=torch.float)
            poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device=runtime_device)

            if init_pose is not None:
                carry = torch.as_tensor(
                    best_pose if not isinstance(best_pose, torch.Tensor) else best_pose.data.cpu().numpy(),
                    device=runtime_device, dtype=torch.float,
                )
                if carry.ndim == 2:
                    carry = carry.unsqueeze(0)
                poses = torch.cat([carry, poses], dim=0)

            poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                              glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                              get_vis=self.debug >= 2)

            if vis is not None:
                imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_2_consider_pose_{name}.png', vis)

            scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                              mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter,
                                              get_vis=self.debug >= 2)
            if vis is not None:
                imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_2_consider_pose_{name}.png', vis)

            ids = torch.as_tensor(scores).argsort(descending=True)
            scores = scores[ids]
            poses = poses[ids]



        if refinement:
            best_pose = poses[0].data.cpu().numpy()
            top3_poses_stage2 = poses[:min(3, len(poses))].data.cpu().numpy()
            if True:
                num_target = 252
                ratio_i, ratio_l = float(scale_range[0]), float(scale_range[1])

                num_oversample = num_target * 4
                sx = np.random.uniform(ratio_i, ratio_l, num_oversample)
                sy = np.random.uniform(ratio_i, ratio_l, num_oversample)
                sz = np.random.uniform(ratio_i, ratio_l, num_oversample)

                if scale_volume_tol is not None and scale_volume_tol > 0:
                    vol = sx * sy * sz
                    valid = np.abs(vol - 1.0) <= float(scale_volume_tol)
                    valid_idx = np.where(valid)[0]
                    if len(valid_idx) >= num_target:
                        chosen = valid_idx[:num_target]
                    else:
                        rest = np.where(~valid)[0]
                        need = num_target - len(valid_idx)
                        extra = rest[np.argsort(np.abs(vol[rest] - 1.0))[:need]]
                        chosen = np.concatenate([valid_idx, extra])
                    sx, sy, sz = sx[chosen], sy[chosen], sz[chosen]

                num_samples = min(num_target, len(sx))
                sx, sy, sz = sx[:num_samples], sy[:num_samples], sz[:num_samples]

                scaling_matrices = np.array([np.diag([sx[i], sy[i], sz[i], 1]) for i in range(num_samples)])
                final_transforms = np.einsum('ij,njk->nik', best_pose, scaling_matrices)

                rescale_poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb,
                                                          depth=depth,
                                                          K=K,
                                                          ob_in_cams=final_transforms, normal_map=normal_map,
                                                          xyz_map=xyz_map,
                                                          glctx=self.glctx, mesh_diameter=self.diameter,
                                                          iteration=iteration,
                                                          get_vis=self.debug >= 2)
                if vis is not None:
                    imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_3_consider_size.png', vis)

                rescale_scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                                          ob_in_cams=rescale_poses.data.cpu().numpy(),
                                                          normal_map=normal_map,
                                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx,
                                                          mesh_diameter=self.diameter,
                                                          get_vis=self.debug >= 2)
                if vis is not None:
                    imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_3_consider_size.png', vis)

                rescale_ids = torch.as_tensor(rescale_scores).argsort(descending=True)
                scaling_matrices = scaling_matrices[rescale_ids.detach().cpu().numpy()]

                scale = np.array([scaling_matrices[0][0, 0], scaling_matrices[0][1, 1], scaling_matrices[0][2, 2]])

                aspect_change = max(scale) / max(min(scale), 1e-6)
                if aspect_change > 1.5:
                    scale = np.clip(scale, 0.92, 1.08)

                self.mesh.vertices = self.mesh.vertices * scale


            self.reset_object(mesh=self.mesh, symmetry_tfs=self.symmetry_tfs)
            self.mesh.export(os.path.join(self.debug_dir, f'final_mesh_{name}.obj'))

            stage4_init = poses.data.cpu().numpy()
            for tp in top3_poses_stage2:
                tp_tensor = np.expand_dims(tp, 0)
                stage4_init = np.concatenate([stage4_init, tp_tensor], axis=0)

            poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=stage4_init, normal_map=normal_map,
                                              xyz_map=xyz_map,
                                              glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                              get_vis=self.debug >= 2)
            if vis is not None:
                imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_4_rerun_pose.png', vis)

            scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                              mesh_tensors=self.mesh_tensors, glctx=self.glctx,
                                              mesh_diameter=self.diameter,
                                              get_vis=self.debug >= 2)
            if vis is not None:
                imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_4_rerun_pose.png', vis)

            ids = torch.as_tensor(scores).argsort(descending=True)
            scores = scores[ids]
            poses = poses[ids]

        best_pose = poses[0] @ self.get_tf_to_centered_mesh()

        best_pose_np = best_pose.data.cpu().numpy()
        R33 = best_pose_np[:3, :3].astype(np.float64)
        U, _, Vt = np.linalg.svd(R33)
        R_so3 = U @ Vt
        if np.linalg.det(R_so3) < 0:
            U[:, -1] *= -1.0
            R_so3 = U @ Vt
        best_pose_np[:3, :3] = R_so3

        self.pose_last = poses[0]
        self.best_id = ids[0]

        self.poses = poses
        self.scores = scores
        return best_pose_np


    def compute_add_err_to_gt_pose(self, poses):
        '''
        @poses: wrt. the centered mesh
        '''
        return -torch.ones(len(poses), device='cuda', dtype=torch.float)

    def track_one(self, rgb, depth, K, iteration, extra={},no_center=False):
        if self.pose_last is None:
            # logging.info("Please init pose by register first")
            raise RuntimeError
        # logging.info("Welcome")

        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        # logging.info("depth processing done")

        xyz_map = \
        depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

        pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                         ob_in_cams=self.pose_last.reshape(1, 4, 4).data.cpu().numpy(), normal_map=None,
                                         xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx,
                                         iteration=iteration, get_vis=self.debug >= 2)
        # logging.info("pose done")
        if self.debug >= 2:
            extra['vis'] = vis
        self.pose_last = pose
        if no_center:
            return pose[0].data.cpu().numpy()
        else:
            return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)

    def track_one_any6d(self, rgb, depth, K, iteration, extra={}):
        if self.pose_last is None:
            # logging.info("Please init pose by register first")
            raise RuntimeError
        # logging.info("Welcome")

        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        # logging.info("depth processing done")

        xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

        pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                         ob_in_cams=self.pose_last.reshape(1, 4, 4).data.cpu().numpy(), normal_map=None,
                                         xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx,
                                         iteration=iteration, get_vis=self.debug >= 2)
        # logging.info("pose done")
        if self.debug >= 2:
            extra['vis'] = vis
        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)
