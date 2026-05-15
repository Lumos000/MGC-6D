# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import functools
import os,sys,kornia
import time
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../')
from learning.datasets.h5_dataset import *
from learning.models.score_network import *
from learning.datasets.pose_dataset import *
from Utils import *
from datareader import *

def _resolve_default_device(device=None):
  if device is not None:
    return torch.device(device)
  if torch.cuda.is_available():
    return torch.device(f"cuda:{torch.cuda.current_device()}")
  return torch.device("cpu")


def vis_batch_data_scores(pose_data, ids, scores, pad_margin=5):
  assert len(scores)==len(ids)
  canvas = []
  for id in ids:
    rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
    rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
    H,W = rgbA_vis.shape[:2]
    zmin = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).min()
    zmax = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).max()
    depthA_vis = depth_to_vis(pose_data.depthAs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    depthB_vis = depth_to_vis(pose_data.depthBs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    if pose_data.normalAs is not None:
      pass
    pad = np.ones((rgbA_vis.shape[0],pad_margin,3))*255
    if pose_data.normalAs is not None:
      pass
    else:
      row = np.concatenate([rgbA_vis, pad, depthA_vis, pad, rgbB_vis, pad, depthB_vis], axis=1)
    s = 100/row.shape[0]
    row = cv2.resize(row, fx=s, fy=s, dsize=None)
    row = cv_draw_text(row, text=f'id:{id}, score:{scores[id]:.3f}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
    canvas.append(row)
    pad = np.ones((pad_margin, row.shape[1], 3))*255
    canvas.append(pad)
  canvas = np.concatenate(canvas, axis=0).astype(np.uint8)
  return canvas



@torch.no_grad()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, normal_map=None, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:TripletH5Dataset=None, cfg=None, work_device=None):
  # logging.info("Welcome make_crop_data_batch")
  H,W = depth.shape[:2]

  args = []
  method = 'box_3d'
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
  # logging.info("make tf_to_crops done")

  if work_device is None:
    if isinstance(mesh_tensors, dict) and ('pos' in mesh_tensors) and torch.is_tensor(mesh_tensors['pos']):
      work_device = mesh_tensors['pos'].device
    else:
      work_device = torch.device('cuda')
  B = len(ob_in_cams)
  poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device=work_device)

  bs = 512
  rgb_rs = []
  depth_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device=work_device, dtype=torch.float)
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:,None]).reshape(-1,4)

  for b in range(0,len(ob_in_cams),bs):
    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseAs[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra)
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    xyz_map_rs.append(extra['xyz_map'])

  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  # logging.info("render done")

  rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device=work_device).permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth, dtype=torch.float, device=work_device)[None,None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    depthAs = kornia.geometry.transform.warp_perspective(depth_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    rgbAs = rgb_rs
    depthAs = depth_rs

  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs

  normalAs = None
  normalBs = None

  Ks = torch.as_tensor(K, dtype=torch.float).reshape(1,3,3).expand(len(rgbAs),3,3)
  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device=work_device)*mesh_diameter

  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=normalAs, normalBs=normalBs, poseA=poseAs, xyz_mapAs=xyz_mapAs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data = dataset.transform_batch(pose_data, H_ori=H, W_ori=W, bound=1)

  # logging.info("pose batch data done")

  return pose_data


class ScorePredictor:
  def __init__(self, amp=True, device=None):
    self.amp = amp
    self.device = _resolve_default_device(device)
    if self.device.type == "cuda" and self.device.index is not None:
      torch.cuda.set_device(self.device.index)
    self.run_name = "2024-01-11-20-02-45"

    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')

    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = np.inf
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 4
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2

    # logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode='test', h5_file=None, max_num_key=1)
    self.model = ScoreNetMultiPair(cfg=self.cfg, c_in=self.cfg['c_in']).to(self.device)

    # logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir, map_location=self.device)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    self.model.load_state_dict(ckpt)

    self.model.to(self.device).eval()
    # logging.info("init done")


  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None):
    '''
    @rgb: np array (H,W,3)
    '''
    # logging.info(f"ob_in_cams:{ob_in_cams.shape}")
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    work_device = self.device
    if work_device.type == "cuda" and work_device.index is not None:
      torch.cuda.set_device(work_device.index)
    ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device=work_device)

    # logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    # logging.info("making cropped data")

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh)
    if isinstance(mesh_tensors, dict):
      for k, v in mesh_tensors.items():
        if torch.is_tensor(v) and v.device != work_device:
          mesh_tensors[k] = v.to(work_device)

    rgb = torch.as_tensor(rgb, device=work_device, dtype=torch.float)
    depth = torch.as_tensor(depth, device=work_device, dtype=torch.float)

    pose_data = make_crop_data_batch(self.cfg.input_resize, ob_in_cams, mesh, rgb, depth, K, crop_ratio=self.cfg['crop_ratio'], glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, cfg=self.cfg, mesh_diameter=mesh_diameter, work_device=work_device)

    def find_best_among_pairs(pose_data:BatchPoseData):
      # logging.info(f'pose_data.rgbAs.shape[0]: {pose_data.rgbAs.shape[0]}')
      ids = []
      scores = []
      bs = pose_data.rgbAs.shape[0]
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].to(work_device), pose_data.xyz_mapAs[b:b+bs].to(work_device)], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].to(work_device), pose_data.xyz_mapBs[b:b+bs].to(work_device)], dim=1).float()
        if pose_data.normalAs is not None:
          A = torch.cat([A, pose_data.normalAs.to(work_device).float()], dim=1)
          B = torch.cat([B, pose_data.normalBs.to(work_device).float()], dim=1)
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.default_stream(device=A.device)):
          with torch.amp.autocast(device_type="cuda", enabled=self.amp):
            output = self.model(A, B, L=len(A))
        scores_cur = output["score_logit"].float().reshape(-1)
        ids.append(scores_cur.argmax()+b)
        scores.append(scores_cur)
      ids = torch.stack(ids, dim=0).reshape(-1)
      scores = torch.cat(scores, dim=0).reshape(-1)
      return ids, scores

    pose_data_iter = pose_data
    global_ids = torch.arange(len(ob_in_cams), device=work_device, dtype=torch.long)
    scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device=work_device)

    while 1:
      ids, scores = find_best_among_pairs(pose_data_iter)
      if len(ids)==1:
        scores_global[global_ids] = scores + 100
        break
      global_ids = global_ids[ids]
      pose_data_iter = pose_data.select_by_indices(global_ids)

    scores = scores_global

    # logging.info(f'forward done')
    torch.cuda.empty_cache()

    if get_vis:
      # logging.info("get_vis...")
      canvas = []
      ids = scores.argsort(descending=True)
      canvas = vis_batch_data_scores(pose_data, ids=ids, scores=scores)
      torch.backends.cudnn.enabled = prev_cudnn
      return scores, canvas

    torch.backends.cudnn.enabled = prev_cudnn
    return scores, None

