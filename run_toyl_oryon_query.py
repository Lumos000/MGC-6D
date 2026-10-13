"""Evaluate MGC-6D on the official Oryon TOYL 2,000 view pairs."""

from __future__ import annotations

from project_paths import setup_project_paths
setup_project_paths()

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import warnings

import cv2
import nvdiffrast.torch as dr
import numpy as np
import pandas as pd
import torch
import trimesh
from pytorch_lightning import seed_everything
from tqdm import tqdm

from bop_toolkit_lib.renderer_vispy import RendererVispy
from candidate_selection import (
    apply_candidate_appearance,
    compute_anchor_reliability_components,
    derive_anchor_reliability_scores,
    paper_registry_candidates,
    score_candidates,
    select_scored_candidate,
)
from estimater import Any6D, PoseRefinePredictor, ScorePredictor
from oryon_toyl_protocol import ORYON_TOYL_PROTOCOL, load_manifest
from query_paper_toyl import (
    CandidateCache,
    MM_TO_M,
    _build_ho3d_style_trans_disc,
    _cuda_cleanup,
    _evaluate_candidate_pose_metrics,
    _gt_pose_to_meters,
    _is_valid_pose,
    _parse_bop_K,
    _parse_bop_pose,
    _read_mask,
    compute_observation_consistency,
)
from revision_protocol import (
    add_query_protocol_arguments,
    runtime_provenance,
    validate_query_protocol,
    write_json_atomic,
)


METRICS = ("ADD", "ADD-S", "ADD(-S)", "AR", "MSSD", "MSPD", "VSD")


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [
            geometry
            for geometry in mesh.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not parts:
            raise ValueError(f"No triangle geometry in {path}")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices):
        raise ValueError(f"Invalid mesh: {path}")
    return mesh


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_identity(pair: dict[str, object]) -> dict[str, object]:
    return {
        "Pair_ID": int(pair["pair_id"]),
        "Anchor_Key": str(pair["anchor_key"]),
        "Reference_Scene": int(pair["reference_scene"]),
        "Reference_Image_ID": int(pair["reference_frame"]),
        "Reference_Anno_ID": int(pair["reference_anno_idx"]),
        "Query_Scene": int(pair["query_scene"]),
        "Query_Image_ID": int(pair["query_frame"]),
        "Query_Anno_ID": int(pair["query_anno_idx"]),
        "Object_ID": int(pair["object_id"]),
        "Class": str(pair["object_name"]),
    }


def _zero_row(
    pair: dict[str, object], status: str, error: str
) -> dict[str, object]:
    return {
        **_pair_identity(pair),
        "Status": status,
        "Error": str(error),
        "Chosen_Candidate": "failed",
        "Selection_Time_s": float("nan"),
        "R_error": float("nan"),
        "T_error": float("nan"),
        **{metric: 0.0 for metric in METRICS},
    }


def _anchor_depth_m(
    raw: np.ndarray, *, toyl_root: Path, reference_scene: int, reference_frame: int
) -> np.ndarray:
    camera = _load_json(
        toyl_root / "test" / f"{reference_scene:06d}" / "scene_camera.json"
    )
    entry = camera.get(str(reference_frame))
    if not isinstance(entry, dict):
        raise ValueError("Reference frame camera entry is absent")
    depth_scale = float(entry.get("depth_scale", 1.0))
    return raw.astype(np.float32) * depth_scale * MM_TO_M


def _load_anchor_candidates(
    *,
    pair: dict[str, object],
    anchor_path: Path,
    anchor_input_root: Path,
    toyl_root: Path,
    registry: dict[str, object],
    selected_labels: set[str],
    candidate_color_mode: str,
    anchor_color_depth_tolerance: float,
    w_depth: float,
    w_mask: float,
    w_geom: float,
) -> tuple[list[dict[str, object]], Path]:
    anchor_key = str(pair["anchor_key"])
    registry_entry = registry.get(anchor_key)
    if not isinstance(registry_entry, dict):
        raise KeyError(f"Missing candidate registry entry for {anchor_key}")
    input_dir = anchor_input_root / anchor_key
    result_dir = anchor_path / anchor_key
    anchor_gt_path = input_dir / f"{anchor_key}_gt_pose.txt"
    required_inputs = (
        input_dir / "color.png",
        input_dir / "depth.png",
        input_dir / "mask.png",
        input_dir / "K.txt",
        anchor_gt_path,
    )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prepared anchor inputs: {missing}")

    color_bgr = cv2.imread(str(input_dir / "color.png"), cv2.IMREAD_COLOR)
    mask_raw = cv2.imread(str(input_dir / "mask.png"), cv2.IMREAD_UNCHANGED)
    depth_raw = cv2.imread(str(input_dir / "depth.png"), cv2.IMREAD_ANYDEPTH)
    if color_bgr is None or mask_raw is None or depth_raw is None:
        raise ValueError(f"Cannot load prepared anchor observation for {anchor_key}")
    anchor_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    anchor_mask = (
        np.any(mask_raw > 0, axis=2) if mask_raw.ndim == 3 else mask_raw > 0
    )
    anchor_k = np.loadtxt(input_dir / "K.txt")
    anchor_depth = _anchor_depth_m(
        depth_raw,
        toyl_root=toyl_root,
        reference_scene=int(pair["reference_scene"]),
        reference_frame=int(pair["reference_frame"]),
    )

    registered = paper_registry_candidates(registry_entry.get("candidates", {}))
    caches: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    poses: dict[str, np.ndarray] = {}
    info_by_label: dict[str, dict[str, object]] = {}
    registry_is_bounded = True
    for label, raw_info in registered.items():
        if label not in selected_labels:
            continue
        info = dict(raw_info)
        mesh_path = Path(str(info.get("mesh_path", "")))
        if not mesh_path.is_file():
            fallback = result_dir / f"final_mesh_{anchor_key}_{label}.obj"
            mesh_path = fallback if fallback.is_file() else mesh_path
        pose_path = Path(str(info.get("pose_path", "")))
        if not mesh_path.is_file() or not pose_path.is_file():
            warnings.warn(f"{anchor_key}/{label}: candidate artifacts are absent")
            continue
        mesh_frame = info.get("mesh_frame")
        pose_frame = info.get("pose_frame")
        if mesh_frame is not None and pose_frame is not None and mesh_frame != pose_frame:
            raise ValueError(
                f"{anchor_key}/{label}: mesh/pose frame mismatch "
                f"({mesh_frame!r}, {pose_frame!r})"
            )
        pose = np.asarray(np.loadtxt(pose_path), dtype=np.float64).reshape(4, 4)
        if not _is_valid_pose(pose):
            raise ValueError(f"{anchor_key}/{label}: invalid anchor pose")
        mesh = _as_trimesh(mesh_path)
        cache = CandidateCache(
            label=label,
            mesh=mesh,
            s_anchor=float(info.get("s_anchor", 1.0)),
        )
        reliability = compute_anchor_reliability_components(info, registry_entry)
        info["shared_anchor_reliability"] = reliability
        caches.append(
            {
                "label": label,
                "cache": cache,
                "mesh_path": mesh_path,
                "mesh_fingerprint": _file_fingerprint(mesh_path),
                "anchor_pose": pose,
                "info": info,
            }
        )
        poses[label] = pose
        info_by_label[label] = info
        registry_is_bounded &= (
            info.get("anchor_score_version") == "bounded_observation_calibrated_v1"
        )
        anchor_rows.append(
            {
                "label": label,
                "s_anchor": float(info.get("s_anchor", 1.0)),
                "l_depth": float(info.get("l_depth", 0.05)),
                "l_mask": float(info.get("l_mask", 0.50)),
                "l_geom": float(info.get("l_geom", 0.05)),
            }
        )
    if not caches:
        raise ValueError(f"{anchor_key}: no requested candidates are usable")
    if not registry_is_bounded:
        derive_anchor_reliability_scores(
            anchor_rows,
            w_depth=w_depth,
            w_mask=w_mask,
            w_geom=w_geom,
            depth_scale=0.05,
            mask_scale=0.50,
            geom_scale=0.05,
            calibration_weight=0.15,
        )
    bounded = {row["label"]: row for row in anchor_rows}
    for item in caches:
        item["cache"].s_anchor = float(bounded[str(item["label"])]["s_anchor"])

    colored, appearance = apply_candidate_appearance(
        {str(item["label"]): item["cache"].mesh for item in caches},
        poses,
        candidate_color_mode,
        anchor_rgb=anchor_rgb,
        anchor_mask=anchor_mask,
        anchor_k=anchor_k,
        anchor_depth=anchor_depth,
        anchor_depth_tolerance_m=anchor_color_depth_tolerance,
    )
    for item in caches:
        item["cache"].mesh = colored[str(item["label"])]
        item["appearance"] = {
            "mode": appearance.mode,
            "source_label": appearance.source_label,
            "transferred_labels": list(appearance.transferred_labels),
            "projected_vertex_counts": dict(appearance.projected_vertex_counts),
        }
    return caches, anchor_gt_path


def _load_query_observation(
    pair: dict[str, object], *, toyl_root: Path, mask_type: str, min_mask_pixels: int
) -> dict[str, object]:
    scene = int(pair["query_scene"])
    frame = int(pair["query_frame"])
    anno_idx = int(pair["query_anno_idx"])
    scene_dir = toyl_root / "test" / f"{scene:06d}"
    camera = _load_json(scene_dir / "scene_camera.json")
    camera_entry = camera.get(str(frame))
    if not isinstance(camera_entry, dict):
        raise ValueError("Query camera entry is absent")
    color_bgr = cv2.imread(str(scene_dir / "rgb" / f"{frame:06d}.png"))
    depth_raw = cv2.imread(
        str(scene_dir / "depth" / f"{frame:06d}.png"), cv2.IMREAD_ANYDEPTH
    )
    mask = _read_mask(str(scene_dir), mask_type, frame, anno_idx)
    if color_bgr is None or depth_raw is None or mask is None:
        raise FileNotFoundError("Query RGB/depth/mask is incomplete")
    pixels = int(np.count_nonzero(mask))
    if pixels < min_mask_pixels:
        raise ValueError(f"Query mask has only {pixels} pixels")
    depth_scale = float(camera_entry.get("depth_scale", 1.0))
    return {
        "scene_dir": scene_dir,
        "rgb": cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB),
        "depth_raw": depth_raw,
        "depth_scale": depth_scale,
        "depth_m": depth_raw.astype(np.float32) * depth_scale * MM_TO_M,
        "mask": mask,
        "K": _parse_bop_K(camera_entry),
    }


def _load_object_evaluation_resources(
    *,
    obj_id: int,
    model_root: Path,
    model_info: dict,
    renderer: RendererVispy,
) -> dict[str, object]:
    mesh_mm = _as_trimesh(model_root / f"obj_{obj_id:06d}.ply")
    mesh_m = mesh_mm.copy()
    mesh_m.vertices = np.asarray(mesh_mm.vertices, dtype=np.float64) * MM_TO_M
    entry = model_info[str(obj_id)]
    diameter_mm = float(entry["diameter"])
    renderer.my_add_object(
        {
            "pts": np.asarray(mesh_mm.vertices, dtype=np.float64),
            "normals": np.asarray(mesh_mm.face_normals, dtype=np.float64),
            "faces": np.asarray(mesh_mm.faces, dtype=np.int32),
        },
        obj_id,
    )
    return {
        "vertices_m": np.asarray(mesh_m.vertices, dtype=np.float64),
        "diameter_mm": diameter_mm,
        "diameter_m": diameter_mm * MM_TO_M,
        "symmetries": _build_ho3d_style_trans_disc(entry),
        "is_symmetric": bool(
            entry.get("symmetries_discrete") or entry.get("symmetries_continuous")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MGC-6D on the official Oryon TOYL 2K pair split"
    )
    parser.add_argument("--name", default="toyl_oryon_2k")
    parser.add_argument("--anchor_path", type=Path, required=True)
    parser.add_argument("--anchor_input_root", type=Path, required=True)
    parser.add_argument("--pair_manifest", type=Path, required=True)
    parser.add_argument("--toyl_root", type=Path, required=True)
    parser.add_argument("--toyl_model_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--register_iteration", type=int, default=5)
    parser.add_argument("--mask_type", choices=("mask_visib", "mask"), default="mask_visib")
    parser.add_argument("--min_mask_pixels", type=int, default=64)
    parser.add_argument("--candidate_error_log_limit", type=int, default=8)
    parser.add_argument("--anchor_color_depth_tolerance", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_nonofficial_split", action="store_true")
    add_query_protocol_arguments(parser)
    args = parser.parse_args()
    if args.register_iteration < 1:
        parser.error("--register_iteration must be positive")
    if args.min_mask_pixels < 1:
        parser.error("--min_mask_pixels must be positive")
    if args.anchor_color_depth_tolerance < 0:
        parser.error("--anchor_color_depth_tolerance must be non-negative")
    selected_labels = validate_query_protocol(args)
    seed_everything(int(args.seed))

    manifest, pairs = load_manifest(
        args.pair_manifest,
        require_official=not args.allow_nonofficial_split,
    )
    anchor_path = args.anchor_path.resolve()
    anchor_input_root = args.anchor_input_root.resolve()
    toyl_root = args.toyl_root.resolve()
    model_root = args.toyl_model_path.resolve()
    registry_path = anchor_path / "candidate_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    registry = _load_json(registry_path)
    if args.output_root is None:
        stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        output = Path("results/toyl_oryon_results") / args.name / stamp
    else:
        output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)

    run_manifest: dict[str, object] = {
        "dataset": "TOYL",
        "protocol": ORYON_TOYL_PROTOCOL,
        "pair_protocol": ORYON_TOYL_PROTOCOL,
        "method_variant": "pair_specific_geometry_reconstruction",
        "official_pair_file_sha256": manifest.get("pair_file_sha256"),
        "pair_manifest": str(args.pair_manifest.resolve()),
        "pair_count": len(pairs),
        "unique_reference_count": len({str(row["anchor_key"]) for row in pairs}),
        "failure_policy": "all requested pairs remain in denominator with zero metrics",
        "mask_sources": {"reference": "dataset_gt", "query": "dataset_gt"},
        "candidate_labels": sorted(selected_labels),
        "candidate_color_mode": args.candidate_color_mode,
        "selection": {
            "policy": "direct_eq22_argmax",
            "loss_normalization": "fixed_scale",
            "w_depth": args.w_depth,
            "w_mask": args.w_mask,
            "w_geom": args.w_geom,
            "score_alpha": args.score_alpha,
            "score_beta": args.score_beta,
            "depth_scale": 0.10,
            "mask_scale": 1.00,
            "geom_scale": 0.10,
        },
        "gt_usage": {
            "reference_mask": "protocol input",
            "query_mask": "protocol input",
            "registration": False,
            "scoring": False,
            "selection": False,
            "pose_and_cad": "metrics only after online winner is fixed",
        },
        "anchor_path": str(anchor_path),
        "anchor_input_root": str(anchor_input_root),
        "seed": int(args.seed),
        "provenance": runtime_provenance(Path(__file__).parent),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "completed": False,
    }
    write_json_atomic(output / "query_run_manifest.json", run_manifest)

    device = args.device
    if torch.cuda.is_available():
        if device.startswith("cuda:") and int(device.split(":", 1)[1]) >= torch.cuda.device_count():
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    glctx = dr.RasterizeCudaContext(device=device)
    placeholder = copy.deepcopy(trimesh.primitives.Box(extents=np.ones(3)))
    bootstrap = Any6D(
        mesh=trimesh.Trimesh(
            vertices=placeholder.vertices.copy(), faces=placeholder.faces.copy()
        ),
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=str(output),
        debug=0,
        glctx=glctx,
    )
    if hasattr(bootstrap, "to_device"):
        bootstrap.to_device(device)
    shared_scorer = bootstrap.scorer
    shared_refiner = bootstrap.refiner
    renderer = RendererVispy(640, 480, mode="depth")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["anchor_key"])].append(pair)
    frame_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    evaluation_cache: dict[int, dict[str, object]] = {}
    model_info: dict | None = None
    selection_times: list[float] = []
    anchor_failure_pairs = 0
    query_input_failure_pairs = 0
    registration_failure_pairs = 0
    evaluation_failure_pairs = 0
    candidate_error_count = 0

    for anchor_key in tqdm(sorted(grouped), desc="Oryon TOYL references"):
        anchor_pairs = grouped[anchor_key]
        try:
            candidates, anchor_gt_path = _load_anchor_candidates(
                pair=anchor_pairs[0],
                anchor_path=anchor_path,
                anchor_input_root=anchor_input_root,
                toyl_root=toyl_root,
                registry=registry,
                selected_labels=selected_labels,
                candidate_color_mode=args.candidate_color_mode,
                anchor_color_depth_tolerance=args.anchor_color_depth_tolerance,
                w_depth=args.w_depth,
                w_mask=args.w_mask,
                w_geom=args.w_geom,
            )
        except Exception as exc:
            anchor_failure_pairs += len(anchor_pairs)
            warnings.warn(f"{anchor_key}: anchor load failed: {exc}")
            frame_rows.extend(_zero_row(pair, "anchor_failure", str(exc)) for pair in anchor_pairs)
            continue

        for pair in anchor_pairs:
            try:
                observation = _load_query_observation(
                    pair,
                    toyl_root=toyl_root,
                    mask_type=args.mask_type,
                    min_mask_pixels=args.min_mask_pixels,
                )
            except Exception as exc:
                query_input_failure_pairs += 1
                frame_rows.append(_zero_row(pair, "query_input_failure", str(exc)))
                continue

            started = time.perf_counter()
            results: list[dict[str, object]] = []
            for candidate in candidates:
                cache = candidate["cache"]
                estimator = None
                try:
                    estimator = Any6D(
                        mesh=cache.mesh,
                        scorer=shared_scorer,
                        refiner=shared_refiner,
                        debug_dir=str(output),
                        debug=0,
                        glctx=glctx,
                    )
                    if hasattr(estimator, "to_device"):
                        estimator.to_device(device)
                    predicted = estimator.register(
                        K=observation["K"],
                        rgb=observation["rgb"],
                        depth=observation["depth_m"],
                        ob_mask=observation["mask"],
                        iteration=int(args.register_iteration),
                        name=f"pair_{int(pair['pair_id']):04d}_{cache.label}",
                    )
                    if not _is_valid_pose(predicted):
                        raise ValueError("FoundationPose returned an invalid pose")
                    losses = compute_observation_consistency(
                        est=estimator,
                        rgb=observation["rgb"],
                        depth=observation["depth_m"],
                        mask=observation["mask"],
                        K=observation["K"],
                        pred_pose=predicted,
                        glctx=glctx,
                    )
                    info = candidate["info"]
                    reliability = info["shared_anchor_reliability"]
                    results.append(
                        {
                            "label": cache.label,
                            "pred_pose_q": np.asarray(predicted, dtype=np.float64),
                            "pred_pose_a": candidate["anchor_pose"],
                            "s_anchor": float(cache.s_anchor),
                            "s_anchor_raw": float(info.get("s_anchor_raw", cache.s_anchor)),
                            "anchor_observation_quality": float(
                                info.get("anchor_observation_quality", cache.s_anchor)
                            ),
                            "anchor_calibration_norm": float(
                                info.get("anchor_calibration_norm", 1.0)
                            ),
                            "anchor_reliability": float(reliability["score"]),
                            "anchor_low_reliability": bool(reliability["low_reliability"]),
                            "l_depth": float(losses[0]),
                            "l_mask": float(losses[1]),
                            "l_geom": float(losses[2]),
                            "l_appearance": 1.0,
                            "coverage": float(np.clip(losses[4], 0.0, 1.0)),
                            "mesh_fingerprint": candidate["mesh_fingerprint"],
                        }
                    )
                except Exception as exc:
                    if candidate_error_count < args.candidate_error_log_limit:
                        warnings.warn(
                            f"Pair {pair['pair_id']} candidate {cache.label} failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    candidate_error_count += 1
                finally:
                    if estimator is not None:
                        del estimator
                    _cuda_cleanup()
            if not results:
                registration_failure_pairs += 1
                frame_rows.append(
                    _zero_row(pair, "registration_failure", "all candidates failed")
                )
                continue

            score_candidates(
                results,
                w_depth=args.w_depth,
                w_mask=args.w_mask,
                w_geom=args.w_geom,
                w_appearance=0.0,
                score_alpha=args.score_alpha,
                score_beta=args.score_beta,
                loss_normalization="fixed_scale",
                depth_scale=0.10,
                mask_scale=1.00,
                geom_scale=0.10,
                appearance_scale=1.00,
            )
            ranked = sorted(results, key=lambda row: float(row["total_score"]), reverse=True)
            winner = select_scored_candidate(results)
            runner_up = ranked[1] if len(ranked) > 1 else None
            margin = (
                float(winner["total_score"] - runner_up["total_score"])
                if runner_up is not None
                else float("nan")
            )
            elapsed = float(time.perf_counter() - started)
            selection_times.append(elapsed)
            rank_by_label = {
                str(result["label"]): rank for rank, result in enumerate(ranked, start=1)
            }
            audits: list[tuple[dict[str, object], dict[str, object]]] = []
            for result in results:
                audit = {
                    **_pair_identity(pair),
                    "Candidate": result["label"],
                    "Selected": result is winner,
                    "S_anchor": result["s_anchor"],
                    "S_anchor_norm": result["s_anchor_norm"],
                    "S_anchor_raw": result["s_anchor_raw"],
                    "Anchor_Observation_Quality": result["anchor_observation_quality"],
                    "Anchor_Calibration_Norm": result["anchor_calibration_norm"],
                    "Anchor_Reliability": result["anchor_reliability"],
                    "Anchor_Low_Reliability": result["anchor_low_reliability"],
                    "L_depth": result["l_depth"],
                    "L_mask": result["l_mask"],
                    "L_geom": result["l_geom"],
                    "L_depth_norm": result["l_depth_norm"],
                    "L_mask_norm": result["l_mask_norm"],
                    "L_geom_norm": result["l_geom_norm"],
                    "Coverage": result["coverage"],
                    "S_query": result["s_query"],
                    "Total_Score": result["total_score"],
                    "Selection_Rank": rank_by_label[str(result["label"])],
                    "Winning_Score": winner["total_score"],
                    "Runner_Up_Candidate": runner_up["label"] if runner_up else "",
                    "Score_Margin": margin,
                    "Mesh_Fingerprint": result["mesh_fingerprint"],
                    "Selection_Time_s": elapsed,
                }
                audits.append((result, audit))

            # Evaluation-only inputs begin here, after the Eq. (22) winner is fixed.
            try:
                anchor_gt = _gt_pose_to_meters(np.loadtxt(anchor_gt_path))
                scene_gt = _load_json(Path(observation["scene_dir"]) / "scene_gt.json")
                query_entries = scene_gt.get(str(int(pair["query_frame"])), [])
                anno_idx = int(pair["query_anno_idx"])
                if anno_idx >= len(query_entries):
                    raise ValueError("Query GT annotation is absent")
                query_gt = _gt_pose_to_meters(_parse_bop_pose(query_entries[anno_idx]))
                obj_id = int(pair["object_id"])
                if obj_id not in evaluation_cache:
                    if model_info is None:
                        model_info = _load_json(model_root / "models_info.json")
                    evaluation_cache[obj_id] = _load_object_evaluation_resources(
                        obj_id=obj_id,
                        model_root=model_root,
                        model_info=model_info,
                        renderer=renderer,
                    )
                resources = evaluation_cache[obj_id]
                prediction = (
                    winner["pred_pose_q"]
                    @ np.linalg.inv(winner["pred_pose_a"])
                    @ anchor_gt
                )
                metrics = _evaluate_candidate_pose_metrics(
                    pred_pose=prediction,
                    gt_pose=query_gt,
                    gt_vertices_m=resources["vertices_m"],
                    gt_diameter_m=resources["diameter_m"],
                    gt_diameter_mm=resources["diameter_mm"],
                    is_symmetric=resources["is_symmetric"],
                    symmetry_transforms=resources["symmetries"],
                    intrinsic=observation["K"],
                    depth_raw=observation["depth_raw"],
                    depth_scale=observation["depth_scale"],
                    renderer=renderer,
                    obj_id=obj_id,
                )
            except Exception as exc:
                evaluation_failure_pairs += 1
                row = _zero_row(pair, "evaluation_failure", str(exc))
                row["Chosen_Candidate"] = str(winner["label"])
                row["Selection_Time_s"] = elapsed
                frame_rows.append(row)
                score_rows.extend(audit for _, audit in audits)
                continue

            if args.save_candidate_metrics:
                for result, audit in audits:
                    candidate_prediction = (
                        result["pred_pose_q"]
                        @ np.linalg.inv(result["pred_pose_a"])
                        @ anchor_gt
                    )
                    candidate_metrics = _evaluate_candidate_pose_metrics(
                        pred_pose=candidate_prediction,
                        gt_pose=query_gt,
                        gt_vertices_m=resources["vertices_m"],
                        gt_diameter_m=resources["diameter_m"],
                        gt_diameter_mm=resources["diameter_mm"],
                        is_symmetric=resources["is_symmetric"],
                        symmetry_transforms=resources["symmetries"],
                        intrinsic=observation["K"],
                        depth_raw=observation["depth_raw"],
                        depth_scale=observation["depth_scale"],
                        renderer=renderer,
                        obj_id=obj_id,
                    )
                    audit.update(
                        {f"Metric_{key}": float(value) for key, value in candidate_metrics.items()}
                    )
            score_rows.extend(audit for _, audit in audits)
            frame_rows.append(
                {
                    **_pair_identity(pair),
                    "Status": "ok",
                    "Error": "",
                    "Chosen_Candidate": str(winner["label"]),
                    "Selection_Time_s": elapsed,
                    "R_error": float(metrics["R_error"]),
                    "T_error": float(metrics["T_error"]),
                    **{metric: float(metrics[metric]) for metric in METRICS},
                }
            )

        del candidates
        _cuda_cleanup()

    frames = pd.DataFrame(frame_rows).sort_values("Pair_ID").reset_index(drop=True)
    if len(frames) != len(pairs) or frames["Pair_ID"].tolist() != list(range(len(pairs))):
        raise RuntimeError("Output rows do not cover every requested pair exactly once")
    scores = pd.DataFrame(score_rows)
    frames.to_csv(output / "0_all_pairs_metrics.csv", index=False)
    frames.to_excel(output / "0_all_pairs_metrics.xlsx", index=False)
    scores.to_csv(output / "0_all_candidate_scores.csv", index=False)
    per_object = []
    for obj_name, group in frames.groupby("Class", sort=True):
        per_object.append(
            {
                "Class": obj_name,
                "Pairs": int(len(group)),
                **{metric: float(group[metric].mean() * 100.0) for metric in METRICS},
            }
        )
    per_object_frame = pd.DataFrame(per_object)
    per_object_frame.to_csv(output / "0_per_object_metrics.csv", index=False)
    metric_summary = {
        metric: float(frames[metric].mean() * 100.0) for metric in METRICS
    }
    selection_counts = Counter(
        frames.loc[frames["Chosen_Candidate"] != "failed", "Chosen_Candidate"].astype(str)
    )
    summary = {
        "dataset": "TOYL",
        "protocol": ORYON_TOYL_PROTOCOL,
        "pair_protocol": ORYON_TOYL_PROTOCOL,
        "method_variant": run_manifest["method_variant"],
        "denominator_pairs": len(frames),
        "successful_pairs": int((frames["Status"] == "ok").sum()),
        "failed_pairs": int((frames["Status"] != "ok").sum()),
        "anchor_failure_pairs": anchor_failure_pairs,
        "query_input_failure_pairs": query_input_failure_pairs,
        "registration_failure_pairs": registration_failure_pairs,
        "evaluation_failure_pairs": evaluation_failure_pairs,
        "metrics_percent": metric_summary,
        "macro_object_metrics_percent": {
            metric: float(per_object_frame[metric].mean()) for metric in METRICS
        },
        "candidate_selection": {
            label: {
                "count": count,
                "percentage_of_all_pairs": 100.0 * count / len(frames),
            }
            for label, count in sorted(selection_counts.items())
        },
        "runtime_seconds": {
            "median": float(np.median(selection_times)) if selection_times else None,
            "p95": float(np.percentile(selection_times, 95)) if selection_times else None,
        },
    }
    write_json_atomic(output / "summary.json", summary)
    run_manifest["completed"] = True
    run_manifest["summary"] = summary
    write_json_atomic(output / "query_run_manifest.json", run_manifest)
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
