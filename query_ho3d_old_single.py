"""Minimal HO3D query using the old stable single-candidate path.

This script intentionally avoids the multi-candidate query path in
query_paper.py. It is used as a Python 3.11 compatibility and baseline
validation probe before running full HO3D experiments.
"""

from project_paths import setup_project_paths
setup_project_paths()

import argparse
import copy
import json
import os
from datetime import datetime as dt_datetime

import cv2
import nvdiffrast.torch as dr
import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

from foundationpose.datareader import Ho3dReader
from foundationpose.Utils import calc_pts_diameter
from estimater import *
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd

from metrics import *
from bop_toolkit_lib.renderer_vispy import RendererVispy
from pytorch_lightning import seed_everything

if __name__ == '__main__':

    seed_everything(0)

    parser = argparse.ArgumentParser(description="Set experiment name and paths")

    parser.add_argument("--name", type=str, default="any6d", help="Experiment name")
    parser.add_argument("--anchor_path", type=str, required=True, help="Path to the HO3D anchor results")
    parser.add_argument("--hot3d_data_root", type=str, required=True, help="Path to the HO3D dataset root")
    parser.add_argument("--ycb_model_path", type=str, required=True, help="Path to the YCB Video Models")
    parser.add_argument("--ycbv_modesl_info_path", type=str, default="./models_info.json", help="Path to the YCB-V model info JSON")
    parser.add_argument("--running_stride", type=int, default=10, help="Running stride")
    parser.add_argument("--register_iteration", type=int, default=5, help="FoundationPose registration iterations")
    parser.add_argument("--obj_folders", nargs="+", default=None, help="HO3D evaluation sequence folders to run")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (e.g., 'cuda:0', 'cuda:1')")
    parser.add_argument("--visualize", action="store_true", help="Save query overlay visualization (pink contour + axes)")

    args = parser.parse_args()

    name = args.name
    hot3d_data_root = args.hot3d_data_root
    ycbv_modesl_info_path = args.ycbv_modesl_info_path
    running_stride = args.running_stride
    anchor_path = args.anchor_path
    ycb_model_path = args.ycb_model_path

    date_str = f'{dt_datetime.now():%Y-%m-%d_%H-%M-%S}'
    save_root = f"./results/ho3d_results/{name}/{date_str}"
    save_results_est_path = f'{save_root}'

    os.makedirs(save_results_est_path, exist_ok=True)

    obj_folder =[
        'MPM10',
        'MPM11',
        'MPM12',
        'MPM13',
        'MPM14',
        'AP10',
        'AP11',
        'AP12',
        'AP13',
        'AP14',
        'SB11',
        'SB13',
        'SM1',
        ]

    if args.obj_folders:
        unknown = sorted(set(args.obj_folders) - set(obj_folder))
        if unknown:
            raise ValueError(f"Unknown HO3D folders: {unknown}. Valid folders: {obj_folder}")
        obj_folder = list(args.obj_folders)

    object_metrics = {obj: {
        'ADD': [], 'ADD-S': [], 'AR': [], 'VSD': [], 'MSSD': [], 'MSPD': [],
        'R error': [], 'T error': [], 'cls_id': [], 'instance_id': []
        } for obj in obj_folder}
    all_frame_data = {
        'Frame_ID': [],
        'Class': [],
        'ADD-S': [],
        'ADD': [],
        'AR': [],
        'MSSD': [],
        'MSPD': [],
        'VSD': [],
        'R_error': [],
        'T_error': [],
        }

    excel_files = []
    # 设置 PyTorch 默认设备
    device = args.device
    if torch.cuda.is_available():
        if device.startswith('cuda:'):
            device_id = int(device.split(':')[1])
            if device_id >= torch.cuda.device_count():
                print(f"⚠️  警告: GPU {device_id} 不存在，使用 cuda:0")
                device = "cuda:0"
        torch.cuda.set_device(device)
        print(f"✅ 使用设备: {device}")
    else:
        print("⚠️  CUDA 不可用，将使用 CPU")
        device = "cpu"
    
    glctx = dr.RasterizeCudaContext(device)
    mesh_tmp = copy.deepcopy(trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)))
    mesh = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(), faces= mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir=save_results_est_path, debug=0, glctx=glctx)
    
    # 将模型移动到指定设备
    if hasattr(est, 'to_device'):
        est.to_device(device)

    renderer = RendererVispy(640, 480, mode='depth')
    obj_count = 0

    data = []

    for obj_f in tqdm(obj_folder, desc='Evaluating Object'):

        video_dir = os.path.join(f"{hot3d_data_root}/evaluation", obj_f)
        reader = Ho3dReader(video_dir, hot3d_data_root)
        reader.color_files = reader.color_files[::running_stride]

        ob_id = reader.get_obj_id()

        # get bop information
        with open(ycbv_modesl_info_path, 'r') as f:
            model_info = json.load(f)
        trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
        if "symmetries_discrete" in model_info[f"{ob_id}"]:
            for sym in model_info[f"{ob_id}"]["symmetries_discrete"]:
                sym_4x4 = np.reshape(sym, (4, 4))
                R = sym_4x4[:3, :3]
                t = sym_4x4[:3, 3].reshape((3, 1))
                trans_disc.append({"R": R, "t": t})

        K_anchor = np.loadtxt(reader.get_reference_K(anchor_path))


        gt_mesh = reader.get_gt_mesh(ycb_model_path)
        gt_diameter = calc_pts_diameter(np.asarray(gt_mesh.vertices))
        mesh = trimesh.load(reader.get_reference_view_1_mesh(anchor_path))

        gt_mesh_dict = {
            'pts': np.asarray(gt_mesh.vertices) * 1e3,
            'normals': np.asarray(gt_mesh.face_normals),
            'faces': np.asarray(gt_mesh.faces),
            }
        renderer.my_add_object(gt_mesh_dict, ob_id)

        pred_pose_a = np.loadtxt(reader.get_reference_view_1_pose(anchor_path))
        gt_pose_a = np.loadtxt(reader.get_reference_view_1_pose(anchor_path).replace('initial','gt'))

        est.reset_object(mesh=mesh, symmetry_tfs=None)

        for i in tqdm(range(0, len(reader.color_files), 1), desc=f"{obj_f} - Frames"):
            gt_pose_q = reader.get_gt_pose(i)

            if gt_pose_q is None:
                continue


            color_file = reader.color_files[i]
            color = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
            H, W = color.shape[:2]
            depth = reader.get_depth(i)
            mask = reader.get_mask(i).astype(np.bool_)
            pred_pose_q = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.register_iteration, name=obj_f)

            pose_aq = pred_pose_q @ np.linalg.inv(pred_pose_a)  # obtained pose A->Q
            pred_q = pose_aq @ gt_pose_a


            err_R, err_T = compute_RT_distances(pred_q, gt_pose_q)

            pose_recall_th = [(5, 5), (5, 10), (10, 10)]

            for r_th, t_th in pose_recall_th:
                succ_r, succ_t = err_R <= r_th, err_T <= t_th
                succ_pose = np.logical_and(succ_r, succ_t).astype(float)

            add = compute_add(gt_mesh.vertices, pred_q, gt_pose_q)
            adds = compute_adds(gt_mesh.vertices, pred_q, gt_pose_q)


            add_thres = float(add <= gt_diameter * 0.1)
            adds_thres = float(adds <= gt_diameter * 0.1)


            pred_r, pred_t = pred_q[:3, :3], np.expand_dims(pred_q[:3, 3], axis=1) * 1e3
            gt_r, gt_t = gt_pose_q[:3, :3], np.expand_dims(gt_pose_q[:3, 3], axis=1) * 1e3

            mssd_err = mssd(pose_est=pred_q, pose_gt=gt_pose_q, pts=gt_mesh.vertices, syms=trans_disc) * 1e3
            mspd_err = mspd(pose_est=pred_q, pose_gt=gt_pose_q, pts=gt_mesh.vertices, K=reader.K, syms=trans_disc)

            mssd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

            vsd_delta = 15.0
            vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            vsd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

            vsd_errs = vsd(pred_r, pred_t, gt_r, gt_t, (depth *1e3), reader.K.reshape(3, 3), vsd_delta, vsd_taus, True, (gt_diameter*1e3), renderer, ob_id)
            vsd_errs = np.asarray(vsd_errs)
            all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
            mean_vsd = all_vsd_recs.mean()

            mssd_cur_rec = mssd_rec * (gt_diameter * 1e3)
            mean_mssd = (mssd_err < mssd_cur_rec).mean()
            mean_mspd = (mspd_err < mspd_rec).mean()

            mean_ar = (mean_mssd + mean_mspd + mean_vsd) / 3.


            object_metrics[obj_f]['ADD'].append(add_thres)
            object_metrics[obj_f]['ADD-S'].append(adds_thres)
            object_metrics[obj_f]['AR'].append(mean_ar)
            object_metrics[obj_f]['VSD'].append(mean_vsd)
            object_metrics[obj_f]['MSSD'].append(mean_mssd)
            object_metrics[obj_f]['MSPD'].append(mean_mspd)
            object_metrics[obj_f]['R error'].append(err_R.tolist()[0])
            object_metrics[obj_f]['T error'].append(err_T.tolist()[0])
            object_metrics[obj_f]['cls_id'].append(obj_f)
            object_metrics[obj_f]['instance_id'].append(obj_count)

            try:
                visualize_frame_results_gt(color=color, gt_mesh=gt_mesh, K=reader.K, gt_pose=gt_pose_q, pred_pose=pred_q, metric=object_metrics[obj_f], obj_f=f"{obj_f}", frame_idx=i, save_path=save_results_est_path, glctx=glctx, name=f"{len(reader.color_files)}_{name}",nocs_metric=True, est_mesh=gt_mesh)
            except:
                pass

            if args.visualize:
                try:
                    vis_dir = os.path.join(save_results_est_path, f"{obj_f}_query_vis")
                    os.makedirs(vis_dir, exist_ok=True)
                    _, rd_depth, _ = nvdiffrast_render(
                        K=reader.K, H=H, W=W, mesh=gt_mesh,
                        ob_in_cams=torch.tensor(pred_q[None]).to(device).float(),
                        context='cuda', use_light=False, glctx=glctx, extra={})
                    rd_mask = (rd_depth[0].detach().cpu().numpy() > 0).astype(np.bool_)
                    vis = vis_mask_contours(color.copy(), rd_mask, color=(255, 40, 150), thickness=3)
                    vis = draw_xyz_axis(vis, ob_in_cam=pred_q, scale=gt_diameter * 0.5, K=reader.K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    cv2.imwrite(os.path.join(vis_dir, f"{obj_f}_query_{i:05d}.png"),
                               cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"[vis] {obj_f} frame {i} failed: {e}")

            obj_count+=1


        df_obj = pd.DataFrame({
            'Frame_ID': object_metrics[obj_f]['instance_id'],
            'Class': object_metrics[obj_f]['cls_id'],
            'ADD-S': object_metrics[obj_f]['ADD-S'],
            'ADD': object_metrics[obj_f]['ADD'],
            'AR': object_metrics[obj_f]['AR'],
            'MSSD': object_metrics[obj_f]['MSSD'],
            'MSPD': object_metrics[obj_f]['MSPD'],
            'VSD': object_metrics[obj_f]['VSD'],
            'R_error': object_metrics[obj_f]['R error'],
            'T_error': object_metrics[obj_f]['T error'],
            })

        means_all = {
            'ADD-S': np.mean(object_metrics[obj_f]['ADD-S']) * 100,
            'ADD': np.mean(object_metrics[obj_f]['ADD']) * 100,
            'AR': np.mean(object_metrics[obj_f]['AR']) * 100,
            'MSSD': np.mean(object_metrics[obj_f]['MSSD']) * 100,
            'MSPD': np.mean(object_metrics[obj_f]['MSPD']) * 100,
            'VSD': np.mean(object_metrics[obj_f]['VSD']) * 100,
            'R_error': np.mean(object_metrics[obj_f]['R error']),
            'T_error': np.mean(object_metrics[obj_f]['T error'])
            }

        mean_row_df = pd.DataFrame({
            'Frame_ID': ['MEAN'],
            'Class': [obj_f],
            'ADD-S': [f"{means_all['ADD-S']:.1f}"],
            'ADD': [f"{means_all['ADD']:.1f}"],
            'AR': [f"{means_all['AR']:.1f}"],
            'MSSD': [f"{means_all['MSSD']:.1f}"],
            'MSPD': [f"{means_all['MSPD']:.1f}"],
            'VSD': [f"{means_all['VSD']:.1f}"],
            'R_error': [f"{means_all['R_error']:.1f}"],
            'T_error': [f"{means_all['T_error']:.1f}"]
            })

        df_obj = pd.concat([df_obj, mean_row_df], ignore_index=True)

        row_data = {
            'Class_ID': obj_f,
            'ADD-S': f"{means_all['ADD-S']:.1f}",
            'ADD': f"{means_all['ADD']:.1f}",
            'AR': f"{means_all['AR']:.1f}",
            'MSSD': f"{means_all['MSSD']:.1f}",
            'MSPD': f"{means_all['MSPD']:.1f}",
            'VSD': f"{means_all['VSD']:.1f}",
            }

        data.append(row_data)

        df_obj.to_excel(f'{save_results_est_path}/{obj_f}_metrics_results.xlsx', index=False)
        all_frame_data['Frame_ID'].extend(object_metrics[obj_f]['instance_id'])
        all_frame_data['Class'].extend(object_metrics[obj_f]['cls_id'])
        all_frame_data['ADD-S'].extend(object_metrics[obj_f]['ADD-S'])
        all_frame_data['ADD'].extend(object_metrics[obj_f]['ADD'])
        all_frame_data['AR'].extend(object_metrics[obj_f]['AR'])
        all_frame_data['MSSD'].extend(object_metrics[obj_f]['MSSD'])
        all_frame_data['MSPD'].extend(object_metrics[obj_f]['MSPD'])
        all_frame_data['VSD'].extend(object_metrics[obj_f]['VSD'])
        all_frame_data['R_error'].extend(object_metrics[obj_f]['R error'])
        all_frame_data['T_error'].extend(object_metrics[obj_f]['T error'])

    overall_means = {
        'ADD-S': np.mean([np.mean(object_metrics[obj]['ADD-S']) for obj in obj_folder]) * 100,
        'ADD': np.mean([np.mean(object_metrics[obj]['ADD']) for obj in obj_folder]) * 100,
        'AR': np.mean([np.mean(object_metrics[obj]['AR']) for obj in obj_folder]) * 100,
        'MSSD': np.mean([np.mean(object_metrics[obj]['MSSD']) for obj in obj_folder]) * 100,
        'MSPD': np.mean([np.mean(object_metrics[obj]['MSPD']) for obj in obj_folder]) * 100,
        'VSD': np.mean([np.mean(object_metrics[obj]['VSD']) for obj in obj_folder]) * 100,
        }

    mean_row = {
        'Class_ID': 'MEAN',
        'ADD-S': f"{overall_means['ADD-S']:.1f}",
        'ADD': f"{overall_means['ADD']:.1f}",
        'AR': f"{overall_means['AR']:.1f}",
        'MSSD': f"{overall_means['MSSD']:.1f}",
        'MSPD': f"{overall_means['MSPD']:.1f}",
        'VSD': f"{overall_means['VSD']:.1f}",
        }
    data.append(mean_row)

    latex_str = f"MEAN & {means_all['AR']:.1f} & {means_all['VSD']:.1f} & {means_all['MSSD']:.1f} & {means_all['MSPD']:.1f} & {means_all['ADD-S']:.1f} & - \\\\"
    print("\n" + latex_str)

    df = pd.DataFrame(data)
    df.to_excel(f'{save_results_est_path}/0_mean_all_metrics_classes_results.xlsx', index=False)

    df_all_frames = pd.DataFrame(all_frame_data)

    means_all = {
        'Frame_ID': 'MEAN',
        'Class': 'ALL',
        'ADD-S': f"{df_all_frames['ADD-S'].mean() * 100:.1f}",
        'ADD': f"{df_all_frames['ADD'].mean() * 100:.1f}",
        'AR': f"{df_all_frames['AR'].mean() * 100:.1f}",
        'MSSD': f"{df_all_frames['MSSD'].mean() * 100:.1f}",
        'MSPD': f"{df_all_frames['MSPD'].mean() * 100:.1f}",
        'VSD': f"{df_all_frames['VSD'].mean() * 100:.1f}",
        'R_error': f"{df_all_frames['R_error'].mean():.1f}",
        'T_error': f"{df_all_frames['T_error'].mean():.1f}",
        }

    df_all_frames = pd.concat([df_all_frames, pd.DataFrame([means_all])], ignore_index=True)

    output_path = f'{save_results_est_path}/0_all_frames_metrics_results.xlsx'
    df_all_frames.to_excel(output_path, index=False)
    print(f"\nAll frames metrics saved to {output_path}")

    print("\nSaved data preview:")
    print(df_all_frames)