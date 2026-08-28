from project_paths import setup_project_paths
setup_project_paths()

import argparse
import gc
import glob
import json
import os
import random
import re

import cv2
import nvdiffrast.torch as dr
import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

from construction import ReconstructionConfig, run_construction
from estimater import Any6D
from foundationpose.Utils import align_mesh_to_coordinate


MM_TO_M = 0.001


def _release_cuda_memory():
    """Release cyclic Any6D references and unused CUDA allocations between trials."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        with torch.cuda.device(device_idx):
            torch.cuda.empty_cache()


def _parse_obj_id(obj_name: str) -> int | None:
    m = re.match(r"^obj_(\d+)$", obj_name)
    if m:
        return int(m.group(1))
    return None


def _find_file(search_dirs, preferred_names, suffix=None):
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


def _load_intrinsics(search_dirs):
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


def _detect_depth_unit(depth_uint16):
    max_val = float(np.nanmax(depth_uint16))
    if max_val > 20000:
        return "rayst3r_uint16"
    if max_val > 200:
        return "mm"
    return "meters"


def _convert_depth(depth_raw, depth_unit):
    if depth_unit == "mm":
        return depth_raw / 1000.0
    if depth_unit == "rayst3r_uint16":
        return depth_raw / 65535.0 * 10.0
    if depth_unit == "meters":
        return depth_raw.astype(np.float32)
    raise ValueError(f"Unsupported depth_unit: {depth_unit}")


def _load_eval_inputs(obj_dir, depth_unit):
    search_dirs = [obj_dir, os.path.join(obj_dir, "anchor_init")]
    color_path = _find_file(search_dirs, ["color.png"], suffix="_color.png")
    depth_path = _find_file(search_dirs, ["depth.png"], suffix="_depth.png")
    mask_path = _find_file(search_dirs, ["mask.png"], suffix="_mask.png")

    if not color_path or not depth_path:
        raise FileNotFoundError(f"Missing color/depth in {obj_dir}")

    color = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if depth_unit == "auto":
        depth_unit = _detect_depth_unit(depth_raw)
    depth = _convert_depth(depth_raw, depth_unit)

    if mask_path and os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask_img > 0).astype(np.bool_)
    else:
        mask = np.ones(depth.shape, dtype=np.bool_)

    intrinsic = _load_intrinsics(search_dirs)
    return color, depth, mask, intrinsic, depth_unit


def _load_model_info(model_path: str) -> dict:
    path = os.path.join(model_path, "models_info.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _build_symmetry_tfs_for_object(obj_id: int, model_info: dict) -> np.ndarray | None:
    info = model_info.get(str(obj_id), {})
    syms = info.get("symmetries_discrete", [])
    if not syms:
        return None
    all_tfs = [np.eye(4, dtype=np.float32)]
    for sym in syms:
        all_tfs.append(np.reshape(np.asarray(sym, dtype=np.float32), (4, 4)))
    return np.stack(all_tfs, axis=0)


def _backproject_depth_to_points(depth: np.ndarray, K: np.ndarray, valid_mask: np.ndarray, max_points: int = 5000) -> np.ndarray:
    ys, xs = np.where(valid_mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float32) - cx) * z / max(fx, 1e-8)
    y = (ys.astype(np.float32) - cy) * z / max(fy, 1e-8)
    pts = np.stack([x, y, z], axis=1)
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def _chamfer_distance_pointcloud(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return 1e3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.as_tensor(pts_a, dtype=torch.float32, device=device)
    b = torch.as_tensor(pts_b, dtype=torch.float32, device=device)
    dists = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2.0).squeeze(0)
    ch = dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
    return float(ch.detach().cpu().item())


def _compute_self_supervised_losses(est, depth: np.ndarray, mask: np.ndarray, K: np.ndarray, pred_pose: np.ndarray, glctx) -> tuple[float, float, float]:
    from foundationpose.Utils import nvdiffrast_render

    h, w = depth.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_tensor = torch.as_tensor(pred_pose, device=device, dtype=torch.float32).unsqueeze(0)
    tf = est.get_tf_to_centered_mesh()
    if torch.is_tensor(tf):
        tf = tf.to(device=pose_tensor.device, dtype=torch.float32).unsqueeze(0)
    else:
        tf = torch.as_tensor(tf, device=pose_tensor.device, dtype=torch.float32).unsqueeze(0)
    ob_in_cams = pose_tensor @ torch.linalg.inv(tf)

    rendered = nvdiffrast_render(
        mesh=est.mesh,
        mesh_tensors=est.mesh_tensors,
        ob_in_cams=ob_in_cams,
        K=K,
        H=h,
        W=w,
        glctx=glctx,
    )
    if rendered is None:
        return 1.0, 1.0, 1e3

    rendered_depth = None
    if isinstance(rendered, tuple):
        if len(rendered) >= 2:
            rendered_depth = rendered[1]
    elif hasattr(rendered, "shape") and rendered.shape[-1] > 3:
        rendered_depth = rendered[..., 3]
    if rendered_depth is None:
        return 1.0, 1.0, 1e3
    rd = rendered_depth[0] if getattr(rendered_depth, "ndim", 0) > 2 else rendered_depth
    if torch.is_tensor(rd):
        rd = rd.detach().cpu().numpy()
    rd = rd.astype(np.float32)

    obs_mask = (mask > 0) & (depth > 1e-4)
    ren_mask = rd > 1e-4

    overlap = obs_mask & ren_mask
    if overlap.sum() > 10:
        depth_scale = max(float(np.mean(depth[overlap])), 1e-4)
        l_depth = float(np.mean(np.abs(depth[overlap] - rd[overlap])) / depth_scale)
    else:
        l_depth = 1.0

    union = obs_mask | ren_mask
    if union.sum() > 0:
        iou = float((obs_mask & ren_mask).sum()) / float(union.sum())
        l_mask = float(1.0 - iou)
    else:
        l_mask = 1.0

    obs_pts = _backproject_depth_to_points(depth, K, obs_mask)
    ren_pts = _backproject_depth_to_points(rd, K, ren_mask)
    l_geom = _chamfer_distance_pointcloud(ren_pts, obs_pts)
    return l_depth, l_mask, l_geom


def _compute_anchor_calibration_score(meta: dict, label: str, c_base: float, w_inlier: float, w_rmse: float, tau_inlier: float, tau_rmse: float) -> float:
    if label in ("rayst3r", "instantmesh"):
        return c_base
    fitness = meta.get("align_fitness", -1.0)
    rmse = meta.get("align_rmse", float("inf"))
    consistency_ok = (fitness >= tau_inlier) and (rmse <= tau_rmse)
    if label == "fused" and consistency_ok:
        return c_base + w_inlier * fitness + w_rmse * max(0.0, 1.0 - rmse)
    return c_base


def main():
    parser = argparse.ArgumentParser(description="TOYL anchor phase with paper multi-candidate logic")
    parser.add_argument("--anchor_folder", type=str, default="/data/gst/MGC-6D/results/anchor_results/toyl_paper/toyl")
    parser.add_argument("--toyl_model_path", type=str, default="/data/gst/data/TOYL/models")
    parser.add_argument(
        "--legacy_anchor_path",
        type=str,
        default="",
        help="Optional legacy single-candidate anchor root (disabled by default)",
    )
    parser.add_argument(
        "--include_legacy_candidate",
        action="store_true",
        help="Include legacy single-candidate anchor mesh as an extra candidate",
    )
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--obj_ids", type=str, default="", help="Comma-separated ids, empty=all")
    parser.add_argument("--depth_unit", type=str, default="auto", choices=["auto", "mm", "rayst3r_uint16", "meters"])
    parser.add_argument("--depth_unit_try_both", action="store_true")
    parser.add_argument("--depth_preprocess", action="store_true")
    parser.add_argument("--rayst3r_device", type=str, default="cuda:0")
    parser.add_argument("--rayst3r_checkpoint", type=str, default=None)
    parser.add_argument("--instantmesh_device", type=str, default="cuda:1")
    parser.add_argument("--align_use_guess_translation", action="store_true")
    parser.add_argument("--align_bidirectional_icp", action="store_true")
    parser.add_argument("--icp_max_iter", type=int, default=50)
    parser.add_argument("--icp_fitness_threshold", type=float, default=0.05)
    parser.add_argument("--icp_multi_hypo", type=int, default=1)
    parser.add_argument("--icp_init_sigma", type=float, default=0.01)
    parser.add_argument("--icp_max_corr_ratio", type=float, default=0.05)
    parser.add_argument("--icp_voxel_size", type=float, default=0.005)
    parser.add_argument("--sample_points", type=int, default=50000)
    parser.add_argument("--select_depth_weight", type=float, default=0.5)
    parser.add_argument("--select_mask_weight", type=float, default=0.25)
    parser.add_argument("--select_geom_weight", type=float, default=0.25)
    parser.add_argument("--select_chamfer_force_ratio", type=float, default=0.2)
    parser.add_argument("--anchor_c_base", type=float, default=1.0)
    parser.add_argument("--anchor_w_inlier", type=float, default=1.0)
    parser.add_argument("--anchor_w_rmse", type=float, default=1.0)
    parser.add_argument("--anchor_tau_inlier", type=float, default=0.05)
    parser.add_argument("--anchor_tau_rmse", type=float, default=0.02)
    parser.add_argument("--score_alpha", type=float, default=0.3)
    parser.add_argument("--score_beta", type=float, default=0.7)
    parser.add_argument("--any6d_iter", type=int, default=5)
    parser.add_argument("--any6d_refine", type=int, default=1, choices=[0, 1])
    parser.add_argument("--anchor_trials_per_candidate", type=int, default=2)
    parser.add_argument(
        "--anchor_trial_include_register",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--anchor_trial_include_nocoarse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trial_depth_weight", type=float, default=0.25)
    parser.add_argument("--trial_mask_weight", type=float, default=0.15)
    parser.add_argument("--trial_geom_weight", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = ReconstructionConfig(
        anchor_root=args.anchor_folder,
        output_root=args.output_root,
        depth_unit=args.depth_unit,
        depth_unit_try_both=args.depth_unit_try_both,
        depth_preprocess=args.depth_preprocess,
        rayst3r_device=args.rayst3r_device,
        rayst3r_checkpoint=args.rayst3r_checkpoint,
        instantmesh_device=args.instantmesh_device,
        align_use_guess_translation=args.align_use_guess_translation,
        align_bidirectional_icp=args.align_bidirectional_icp,
        icp_max_iter=args.icp_max_iter,
        icp_fitness_threshold=args.icp_fitness_threshold,
        icp_multi_hypo=args.icp_multi_hypo,
        icp_init_sigma=args.icp_init_sigma,
        icp_max_corr_ratio=args.icp_max_corr_ratio,
        icp_voxel_size=args.icp_voxel_size,
        sample_points=args.sample_points,
    )

    recon_results = run_construction(cfg)
    _release_cuda_memory()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.rayst3r_device)
    model_info = _load_model_info(args.toyl_model_path)

    requested_ids = None
    if args.obj_ids.strip():
        requested_ids = {int(x.strip()) for x in args.obj_ids.split(",") if x.strip()}

    run_meta = {
        "anchor_folder": args.anchor_folder,
        "output_root": args.output_root,
        "toyl_model_path": args.toyl_model_path,
        "legacy_anchor_path": args.legacy_anchor_path,
        "include_legacy_candidate": bool(args.include_legacy_candidate),
        "selection": {
            "select_depth_weight": args.select_depth_weight,
            "select_mask_weight": args.select_mask_weight,
            "select_geom_weight": args.select_geom_weight,
            "select_chamfer_force_ratio": args.select_chamfer_force_ratio,
        },
        "calibration": {
            "anchor_c_base": args.anchor_c_base,
            "anchor_w_inlier": args.anchor_w_inlier,
            "anchor_w_rmse": args.anchor_w_rmse,
            "anchor_tau_inlier": args.anchor_tau_inlier,
            "anchor_tau_rmse": args.anchor_tau_rmse,
            "score_alpha": args.score_alpha,
            "score_beta": args.score_beta,
        },
        "args": vars(args),
    }
    with open(os.path.join(args.anchor_folder, "anchor_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    results = []
    pose_rows = []
    pose_rows_all = []
    candidate_registry = {}

    obj_list = sorted([d for d in os.listdir(args.anchor_folder) if os.path.isdir(os.path.join(args.anchor_folder, d))])
    glctx = dr.RasterizeCudaContext(device=args.rayst3r_device)

    for obj in tqdm(obj_list, desc="Object"):
        obj_id = _parse_obj_id(obj)
        if obj_id is None:
            continue
        if requested_ids is not None and obj_id not in requested_ids:
            continue
        if obj not in recon_results:
            continue

        obj_dir = os.path.join(args.anchor_folder, obj)
        symmetry_tfs = _build_symmetry_tfs_for_object(obj_id, model_info)

        meta_path = recon_results[obj].get("meta")
        recon_meta = {}
        chosen_depth_unit = args.depth_unit
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                recon_meta = json.load(f)
            chosen_depth_unit = recon_meta.get("depth_unit", args.depth_unit)

        color, depth, mask, intrinsic, _ = _load_eval_inputs(obj_dir, chosen_depth_unit)

        ray_path = recon_results[obj].get("mesh_rayst3r")
        inst_path = recon_results[obj].get("mesh_instant")
        fused_path = recon_results[obj].get("mesh_fused")
        candidates = {}

        align_direction = recon_meta.get("align_direction", None)
        align_valid = align_direction not in (None, "none")

        # Conservative fallback: if reconstruction alignment itself is invalid,
        # avoid aggressive multi-candidate competition and keep the most stable
        # prior candidate to reduce downstream query instability.
        if not align_valid:
            if inst_path and os.path.exists(inst_path):
                candidates["instantmesh"] = inst_path
            elif ray_path and os.path.exists(ray_path):
                candidates["rayst3r"] = ray_path
        else:
            if ray_path and os.path.exists(ray_path):
                candidates["rayst3r"] = ray_path
            if inst_path and os.path.exists(inst_path):
                candidates["instantmesh"] = inst_path
            if fused_path and os.path.exists(fused_path):
                fused_ok = (
                    recon_meta.get("align_fitness", -1.0) >= args.anchor_tau_inlier
                    and recon_meta.get("align_rmse", float("inf")) <= args.anchor_tau_rmse
                )
                if fused_ok:
                    candidates["fused"] = fused_path

        # Optional safety candidate from previous single-candidate anchor.
        if args.include_legacy_candidate and args.legacy_anchor_path:
            legacy_obj_dir = os.path.join(args.legacy_anchor_path, obj)
            legacy_mesh = os.path.join(legacy_obj_dir, f"final_mesh_{obj}.obj")
            legacy_pose = os.path.join(legacy_obj_dir, f"{obj}_initial_pose.txt")
            if os.path.exists(legacy_mesh) and os.path.exists(legacy_pose):
                candidates["legacy_single"] = legacy_mesh
        if not candidates:
            continue

        s_anchor_scores = {
            label: _compute_anchor_calibration_score(
                meta=recon_meta,
                label=label,
                c_base=args.anchor_c_base,
                w_inlier=args.anchor_w_inlier,
                w_rmse=args.anchor_w_rmse,
                tau_inlier=args.anchor_tau_inlier,
                tau_rmse=args.anchor_tau_rmse,
            )
            for label in candidates
        }
        if "legacy_single" in candidates:
            # Favor legacy anchor slightly so multi-candidate won't regress badly
            # when reconstruction candidates are unstable.
            s_anchor_scores["legacy_single"] = max(
                s_anchor_scores.get("legacy_single", args.anchor_c_base),
                float(args.anchor_c_base) + 0.15,
            )

        trial_strategies = [
            {"name": "any6d_coarse_aa", "use_any6d": True, "axis_align": True, "coarse_est": True, "refinement": bool(args.any6d_refine)},
            {"name": "any6d_coarse_noaa", "use_any6d": True, "axis_align": False, "coarse_est": True, "refinement": bool(args.any6d_refine)},
        ]
        if args.anchor_trial_include_nocoarse:
            trial_strategies.append(
                {"name": "any6d_nocoarse", "use_any6d": True, "axis_align": False, "coarse_est": False, "refinement": bool(args.any6d_refine)}
            )
        if args.anchor_trial_include_register:
            trial_strategies.append(
                {"name": "register_simple", "use_any6d": False, "axis_align": False, "coarse_est": False, "refinement": False}
            )

        def _evaluate_mesh_pose(mesh_path: str, label: str):
            mesh_local_raw = trimesh.load(mesh_path)
            if isinstance(mesh_local_raw, trimesh.Scene):
                mesh_local_raw = trimesh.util.concatenate(
                    [g for g in mesh_local_raw.geometry.values() if isinstance(g, trimesh.Trimesh)]
                )
            mesh_local_raw = align_mesh_to_coordinate(mesh_local_raw)
            mesh_local_raw.export(os.path.join(obj_dir, f"center_mesh_{obj}_{label}.obj"))

            best_trial = None
            trial_records = []
            for strat_idx, strat in enumerate(trial_strategies):
                for seed_i in range(max(1, int(args.anchor_trials_per_candidate))):
                    trial_seed = int(args.seed + obj_id * 10007 + strat_idx * 101 + seed_i * 7919)
                    random.seed(trial_seed)
                    np.random.seed(trial_seed)
                    torch.manual_seed(trial_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(trial_seed)

                    mesh_local = mesh_local_raw.copy()
                    est_local = Any6D(symmetry_tfs=symmetry_tfs, mesh=mesh_local, debug_dir=obj_dir, debug=0)
                    try:
                        if strat["use_any6d"]:
                            pred_pose_local = est_local.register_any6d(
                                K=intrinsic,
                                rgb=color,
                                depth=depth,
                                ob_mask=mask,
                                iteration=args.any6d_iter,
                                refinement=bool(strat["refinement"]),
                                axis_align=bool(strat["axis_align"]),
                                coarse_est=bool(strat["coarse_est"]),
                                name=f"anchor_{label}_{strat['name']}_s{seed_i}",
                            )
                        else:
                            pred_pose_local = est_local.register(
                                K=intrinsic,
                                rgb=color,
                                depth=depth,
                                ob_mask=mask,
                                iteration=args.any6d_iter,
                                name=f"anchor_{label}_{strat['name']}_s{seed_i}",
                            )
                    except Exception:
                        del est_local
                        _release_cuda_memory()
                        continue

                    l_depth, l_mask, l_geom = _compute_self_supervised_losses(
                        est=est_local,
                        depth=depth,
                        mask=mask,
                        K=intrinsic,
                        pred_pose=pred_pose_local,
                        glctx=glctx,
                    )
                    trial_score = (
                        float(args.trial_depth_weight) * float(l_depth)
                        + float(args.trial_mask_weight) * float(l_mask)
                        + float(args.trial_geom_weight) * float(l_geom)
                    )
                    rec = {
                        "strategy": strat["name"],
                        "seed": int(seed_i),
                        "score": float(trial_score),
                        "l_depth": float(l_depth),
                        "l_mask": float(l_mask),
                        "l_geom": float(l_geom),
                        "pred_pose": pred_pose_local,
                        "mesh": est_local.mesh.copy(),
                    }
                    trial_records.append(rec)
                    if best_trial is None or rec["score"] < best_trial["score"]:
                        best_trial = rec
                    del est_local
                    _release_cuda_memory()

            if best_trial is None:
                raise RuntimeError(f"No valid anchor trials for candidate {label}")

            with open(os.path.join(obj_dir, f"{obj}_anchor_trials_{label}.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {k: v for k, v in r.items() if k not in ("pred_pose", "mesh")}
                        for r in trial_records
                    ],
                    f,
                    indent=2,
                )

            pred_pose_local = np.array(best_trial["pred_pose"], copy=True)
            best_mesh = best_trial["mesh"]
            np.savetxt(os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt"), pred_pose_local)
            best_mesh.export(os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj"))
            np.savetxt(os.path.join(obj_dir, f"{obj}_ldepth_{label}.txt"), [best_trial["l_depth"]])
            np.savetxt(os.path.join(obj_dir, f"{obj}_lmask_{label}.txt"), [best_trial["l_mask"]])
            np.savetxt(os.path.join(obj_dir, f"{obj}_cd_{label}.txt"), [best_trial["l_geom"]])
            return {
                "pred_pose": pred_pose_local,
                "l_depth": float(best_trial["l_depth"]),
                "l_mask": float(best_trial["l_mask"]),
                "l_geom": float(best_trial["l_geom"]),
            }

        results_pose = {}
        for label, path in candidates.items():
            results_pose[label] = _evaluate_mesh_pose(path, label)
            res = results_pose[label]
            pose_rows_all.append(
                {
                    "Object": obj,
                    "Object_Number": obj_id,
                    "Candidate": label,
                    "L_depth": res["l_depth"],
                    "L_mask": res["l_mask"],
                    "L_geom": res["l_geom"],
                    "S_anchor": float(s_anchor_scores[label]),
                }
            )
        if not results_pose:
            continue

        min_l_depth = min(res["l_depth"] for res in results_pose.values())
        min_l_mask = min(res["l_mask"] for res in results_pose.values())
        min_l_geom = min(res["l_geom"] for res in results_pose.values())
        max_s_anchor = max(s_anchor_scores.values())

        scored = []
        score_breakdown = {}
        for label, res in results_pose.items():
            depth_term = args.select_depth_weight * (res["l_depth"] / max(min_l_depth, 1e-8))
            mask_term = args.select_mask_weight * (res["l_mask"] / max(min_l_mask, 1e-8))
            geom_term = args.select_geom_weight * (res["l_geom"] / max(min_l_geom, 1e-8))
            s_query = depth_term + mask_term + geom_term
            s_anchor_norm = s_anchor_scores[label] / max(max_s_anchor, 1e-8)
            total_score = -args.score_alpha * s_anchor_norm + args.score_beta * s_query
            scored.append((label, res, float(total_score)))
            score_breakdown[label] = {
                "Score_L_depth": float(depth_term),
                "Score_L_mask": float(mask_term),
                "Score_L_geom": float(geom_term),
                "S_query": float(s_query),
                "S_anchor": float(s_anchor_scores[label]),
                "S_anchor_norm": float(s_anchor_norm),
                "Total_Score": float(total_score),
            }
        scored.sort(key=lambda x: x[2])
        selection_scores = {label: score for label, _, score in scored}

        chosen_label, best, _ = scored[0]
        forced_by_chamfer = False
        best_geom_label, best_geom = min(results_pose.items(), key=lambda x: x[1]["l_geom"])
        if best["l_geom"] > best_geom["l_geom"] * (1.0 + args.select_chamfer_force_ratio):
            chosen_label, best = best_geom_label, best_geom
            forced_by_chamfer = True

        for i, row in enumerate(pose_rows_all):
            if row["Object"] == obj and row["Candidate"] in selection_scores:
                pose_rows_all[i]["Selection_Score"] = float(selection_scores[row["Candidate"]])
                pose_rows_all[i].update(score_breakdown[row["Candidate"]])

        chosen_pose_path = os.path.join(obj_dir, f"{obj}_initial_pose_{chosen_label}.txt")
        chosen_mesh_path = os.path.join(obj_dir, f"final_mesh_{obj}_{chosen_label}.obj")
        pred_pose_final = np.loadtxt(chosen_pose_path)
        final_mesh = trimesh.load(chosen_mesh_path)

        np.savetxt(os.path.join(obj_dir, f"{obj}_initial_pose.txt"), pred_pose_final)
        final_mesh.export(os.path.join(obj_dir, f"final_mesh_{obj}.obj"))
        np.savetxt(os.path.join(obj_dir, f"{obj}_cd.txt"), [best["l_geom"]])

        results.append({"Object": obj, "Object_Number": obj_id, "Chamfer_Distance": float(best["l_geom"])})
        pose_rows.append(
            {
                "Object": obj,
                "Object_Number": obj_id,
                "Chosen_Mesh": chosen_label,
                "L_depth": float(best["l_depth"]),
                "L_mask": float(best["l_mask"]),
                "L_geom": float(best["l_geom"]),
                "S_anchor": float(s_anchor_scores[chosen_label]),
                "Selection_Score": float(selection_scores.get(chosen_label, float("inf"))),
                "Forced_By_Chamfer": bool(forced_by_chamfer),
            }
        )

        obj_candidates_info = {}
        for label, path in candidates.items():
            obj_candidates_info[label] = {
                "mesh_path": os.path.abspath(os.path.join(obj_dir, f"final_mesh_{obj}_{label}.obj")),
                "raw_mesh_path": os.path.abspath(path),
                "pose_path": os.path.abspath(os.path.join(obj_dir, f"{obj}_initial_pose_{label}.txt")),
                "s_anchor": float(s_anchor_scores[label]),
                "l_depth": float(results_pose[label]["l_depth"]),
                "l_mask": float(results_pose[label]["l_mask"]),
                "l_geom": float(results_pose[label]["l_geom"]),
            }
        candidate_registry[obj] = {
            "chosen": chosen_label,
            "candidates": obj_candidates_info,
            "final_mesh_path": os.path.abspath(os.path.join(obj_dir, f"final_mesh_{obj}.obj")),
            "final_pose_path": os.path.abspath(os.path.join(obj_dir, f"{obj}_initial_pose.txt")),
            "K_path": os.path.abspath(os.path.join(obj_dir, "K.txt")),
        }
        print(
            f"{obj}: chosen={chosen_label} "
            f"L_depth={best['l_depth']:.6f} L_mask={best['l_mask']:.6f} L_geom={best['l_geom']:.6f}"
        )
        _release_cuda_memory()

    registry_path = os.path.join(args.anchor_folder, "candidate_registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(candidate_registry, f, indent=2)
    print(f"\nCandidate registry saved to: {registry_path}")

    if results:
        df = pd.DataFrame(results).sort_values("Object")
        excel_path = os.path.join(args.anchor_folder, "chamfer_distances.xlsx")
        df.to_excel(excel_path, index=False)

        pose_df = pd.DataFrame(pose_rows).sort_values("Object")
        pose_all_df = pd.DataFrame(pose_rows_all).sort_values(["Object", "Candidate"])
        pose_excel_path = os.path.join(args.anchor_folder, "pose_metrics.xlsx")
        with pd.ExcelWriter(pose_excel_path) as writer:
            pose_df.to_excel(writer, index=False, sheet_name="chosen")
            pose_all_df.to_excel(writer, index=False, sheet_name="candidates")
        print(f"Results saved to: {excel_path}")
        print(f"Pose metrics saved to: {pose_excel_path}")
    else:
        print("No objects were processed.")


if __name__ == "__main__":
    main()
