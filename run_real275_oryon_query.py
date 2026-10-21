"""Evaluate MGC-6D on the official Oryon REAL275 2,000 view pairs."""

from project_paths import setup_project_paths
setup_project_paths()

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import time
import warnings

import cv2
import nvdiffrast.torch as dr
import numpy as np
import pandas as pd
import trimesh
from pytorch_lightning import seed_everything
from tqdm import tqdm

from candidate_selection import (
    apply_candidate_appearance,
    compute_anchor_reliability_components,
    paper_registry_candidates,
    score_candidates,
    select_scored_candidate,
)
from estimater import Any6D, PoseRefinePredictor, ScorePredictor
from oryon_real275_protocol import load_manifest
from revision_protocol import (
    add_query_protocol_arguments,
    runtime_provenance,
    validate_query_protocol,
)
from run_real275_query import (
    CandidateCache,
    REAL275_K,
    _evaluate_real275_pose_metrics,
    _load_real275_evaluation_resources,
    compute_observation_consistency,
    decompose_nocs_pose,
    load_gt_poses,
    swap_candidate,
)


METRICS = ("ADD", "ADD-S", "ADD(-S)", "AR", "MSSD", "MSPD", "VSD")


def _as_trimesh(path: str) -> trimesh.Trimesh:
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
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh in {path}")
    return mesh


def _load_anchor_candidates(
    *,
    anchor_key: str,
    anchor_input_root: Path,
    candidate_registry: dict[str, object],
    selected_labels: set[str],
    candidate_color_mode: str,
    mesh_diameter_mode: str,
) -> list[dict[str, object]]:
    input_dir = anchor_input_root / anchor_key
    entry = candidate_registry.get(anchor_key)
    if not isinstance(entry, dict):
        raise KeyError(f"Missing registry entry for {anchor_key}")
    if entry.get("anchor_score_version") != "bounded_observation_calibrated_v1":
        raise ValueError(f"{anchor_key}: invalid anchor score contract")

    raw_candidates = paper_registry_candidates(entry.get("candidates", {}))
    candidates: list[dict[str, object]] = []
    for label, info in raw_candidates.items():
        if label not in selected_labels:
            continue
        if info.get("anchor_score_version") != "bounded_observation_calibrated_v1":
            raise ValueError(f"{anchor_key}/{label}: invalid anchor score contract")
        mesh_path = str(info.get("mesh_path", ""))
        pose_path = str(info.get("pose_path", ""))
        if not Path(mesh_path).is_file() or not Path(pose_path).is_file():
            raise FileNotFoundError(
                f"Missing candidate artifacts for {anchor_key}/{label}"
            )
        mesh = _as_trimesh(mesh_path)
        pose = np.loadtxt(pose_path)
        mesh_frame = str(info.get("mesh_frame", "legacy_unknown"))
        pose_frame = str(info.get("pose_frame", "legacy_unknown"))
        if not (
            mesh_frame == "any6d_internal_centered"
            and pose_frame == "any6d_internal_centered_to_anchor_camera"
        ):
            raise ValueError(
                f"{anchor_key}/{label}: unsupported frame contract "
                f"{mesh_frame!r}/{pose_frame!r}"
            )
        candidates.append(
            {
                "label": label,
                "mesh": mesh,
                "anchor_pose": np.asarray(pose, dtype=np.float64),
                "s_anchor": float(info.get("s_anchor", 1.0)),
                "anchor_reliability": compute_anchor_reliability_components(
                    info, entry
                ),
            }
        )
    if not candidates:
        raise ValueError(f"{anchor_key}: no requested candidates are available")

    color_bgr = cv2.imread(str(input_dir / "color.png"))
    mask_raw = cv2.imread(str(input_dir / "mask.png"), cv2.IMREAD_UNCHANGED)
    depth_raw = cv2.imread(str(input_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    intrinsic = np.loadtxt(input_dir / "K.txt")
    if color_bgr is None or mask_raw is None or depth_raw is None:
        raise FileNotFoundError(f"Incomplete prepared anchor input for {anchor_key}")
    anchor_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    anchor_mask = (
        np.any(mask_raw > 0, axis=2) if mask_raw.ndim == 3 else mask_raw > 0
    )
    anchor_depth = depth_raw.astype(np.float32) / 1e3
    meshes = {str(row["label"]): row["mesh"].copy() for row in candidates}
    poses = {str(row["label"]): row["anchor_pose"] for row in candidates}
    meshes, summary = apply_candidate_appearance(
        meshes,
        poses,
        candidate_color_mode,
        anchor_rgb=anchor_rgb,
        anchor_mask=anchor_mask,
        anchor_k=intrinsic,
        anchor_depth=anchor_depth,
    )
    for row in candidates:
        label = str(row["label"])
        row["mesh"] = meshes[label]
        row["cache"] = CandidateCache(label, meshes[label], mesh_diameter_mode)
    print(
        f"{anchor_key}: appearance={summary.mode}, candidates="
        f"{','.join(str(row['label']) for row in candidates)}"
    )
    return candidates


def _zero_row(pair: dict[str, object], status: str, error: str) -> dict[str, object]:
    row: dict[str, object] = {
        "Pair_ID": int(pair["pair_id"]),
        "Anchor_Key": str(pair["anchor_key"]),
        "Reference_Scene": f"scene_{int(pair['reference_scene'])}",
        "Reference_Frame": int(pair["reference_frame"]),
        "Reference_Instance_ID": int(pair["reference_instance_id"]),
        "Query_Scene": f"scene_{int(pair['query_scene'])}",
        "Query_Frame": int(pair["query_frame"]),
        "Query_Instance_ID": int(pair["query_instance_id"]),
        "Class": str(pair["object_name"]),
        "Status": status,
        "Error": error,
        "Chosen_Candidate": "FAILURE",
        "Selection_Time_s": 0.0,
        "R_error": float("nan"),
        "T_error": float("nan"),
    }
    row.update({metric: 0.0 for metric in METRICS})
    return row


def _pair_identity(pair: dict[str, object]) -> dict[str, object]:
    return {
        "Pair_ID": int(pair["pair_id"]),
        "Anchor_Key": str(pair["anchor_key"]),
        "Reference_Scene": f"scene_{int(pair['reference_scene'])}",
        "Reference_Frame": int(pair["reference_frame"]),
        "Reference_Instance_ID": int(pair["reference_instance_id"]),
        "Query_Scene": f"scene_{int(pair['query_scene'])}",
        "Query_Frame": int(pair["query_frame"]),
        "Query_Instance_ID": int(pair["query_instance_id"]),
        "Class": str(pair["object_name"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MGC-6D on the official Oryon REAL275 2K pair split"
    )
    parser.add_argument("--name", default="real275_oryon_2k")
    parser.add_argument("--anchor_path", type=Path, required=True)
    parser.add_argument("--anchor_input_root", type=Path, required=True)
    parser.add_argument("--pair_manifest", type=Path, required=True)
    parser.add_argument("--real275_data_root", type=Path, required=True)
    parser.add_argument("--models_info_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--register_iteration", type=int, default=5)
    parser.add_argument("--min_mask_pixels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mesh_diameter_mode",
        choices=("bbox_diagonal", "sampled_surface"),
        default="bbox_diagonal",
    )
    parser.add_argument(
        "--allow_nonofficial_split",
        action="store_true",
        help="Testing only: allow a non-official manifest.",
    )
    add_query_protocol_arguments(parser)
    args = parser.parse_args()
    args.w_appearance = 0.0
    args.query_loss_normalization = "fixed_scale"
    args.query_depth_scale = 0.10
    args.query_mask_scale = 1.00
    args.query_geom_scale = 0.10
    args.query_appearance_scale = 1.00
    if args.register_iteration < 1:
        parser.error("--register_iteration must be positive")

    seed_everything(int(args.seed))
    selected_labels = validate_query_protocol(args)
    manifest, pairs = load_manifest(
        args.pair_manifest,
        require_official=not args.allow_nonofficial_split,
    )
    anchor_path = args.anchor_path.resolve()
    anchor_input_root = args.anchor_input_root.resolve()
    data_root = args.real275_data_root.resolve()
    registry_path = anchor_path / "candidate_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    candidate_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    expected_keys = {str(row["anchor_key"]) for row in pairs}
    missing_keys = expected_keys - set(candidate_registry)
    if missing_keys:
        preview = sorted(missing_keys)[:5]
        warnings.warn(
            f"Anchor registry misses {len(missing_keys)} Oryon references; "
            f"their pairs will receive zero metrics. Examples: {preview}"
        )

    if args.output_root is None:
        stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        output_root = Path("results/real275_oryon_results") / args.name / stamp
    else:
        output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_root}")
    output_root.mkdir(parents=True)

    run_manifest: dict[str, object] = {
        "dataset": "REAL275",
        "protocol": "oryon_cross_scene_2k",
        "official_pair_file_sha256": manifest.get("pair_file_sha256"),
        "pair_count": len(pairs),
        "unique_reference_count": len(expected_keys),
        "failure_policy": "all requested pairs remain in denominator with zero metrics",
        "mask_sources": {"reference": "dataset_gt", "query": "dataset_gt"},
        "anchor_path": str(anchor_path),
        "anchor_input_root": str(anchor_input_root),
        "pair_manifest": str(args.pair_manifest.resolve()),
        "candidate_labels": sorted(selected_labels),
        "candidate_color_mode": args.candidate_color_mode,
        "selection": {
            "policy": "direct_eq9_argmax",
            "w_depth": args.w_depth,
            "w_mask": args.w_mask,
            "w_geom": args.w_geom,
            "score_alpha": args.score_alpha,
            "score_beta": args.score_beta,
            "depth_scale": args.query_depth_scale,
            "mask_scale": args.query_mask_scale,
            "geom_scale": args.query_geom_scale,
        },
        "gt_usage": {
            "registration": False,
            "scoring": False,
            "selection": False,
            "reference_and_query_pose": "metrics only after online winner is fixed",
            "gt_cad": "metrics only after online winner is fixed",
        },
        "seed": int(args.seed),
        "provenance": runtime_provenance(Path(__file__).parent),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "completed": False,
    }
    (output_root / "query_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(
        trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4))
    )
    estimator = Any6D(
        mesh=trimesh.Trimesh(
            vertices=mesh_tmp.vertices.copy(), faces=mesh_tmp.faces.copy()
        ),
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=str(output_root),
        debug=0,
        glctx=glctx,
    )
    from bop_toolkit_lib.renderer_vispy import RendererVispy

    renderer = RendererVispy(640, 480, mode="depth")
    evaluation_cache: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["anchor_key"])].append(pair)

    frame_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    registration_failures = 0
    anchor_failures = 0
    gt_failures = 0
    selection_times: list[float] = []

    for anchor_key in tqdm(sorted(grouped), desc="Oryon unique references"):
        anchor_pairs = grouped[anchor_key]
        object_name = str(anchor_pairs[0]["object_name"])
        try:
            candidates = _load_anchor_candidates(
                anchor_key=anchor_key,
                anchor_input_root=anchor_input_root,
                candidate_registry=candidate_registry,
                selected_labels=selected_labels,
                candidate_color_mode=args.candidate_color_mode,
                mesh_diameter_mode=args.mesh_diameter_mode,
            )
            anchor_gt_path = anchor_input_root / anchor_key / f"{anchor_key}_gt_pose.txt"
        except Exception as exc:
            anchor_failures += len(anchor_pairs)
            for pair in anchor_pairs:
                frame_rows.append(_zero_row(pair, "anchor_failure", str(exc)))
            warnings.warn(f"{anchor_key}: anchor load failed: {exc}")
            continue

        for pair in anchor_pairs:
            identity = _pair_identity(pair)
            query_scene = f"scene_{int(pair['query_scene'])}"
            query_frame = int(pair["query_frame"])
            query_instance = int(pair["query_instance_id"])
            scene_dir = data_root / "real_test" / query_scene
            try:
                color_bgr = cv2.imread(str(scene_dir / f"{query_frame:04d}_color.png"))
                depth_raw = cv2.imread(
                    str(scene_dir / f"{query_frame:04d}_depth.png"),
                    cv2.IMREAD_ANYDEPTH,
                )
                mask_full = cv2.imread(
                    str(scene_dir / f"{query_frame:04d}_mask.png"),
                    cv2.IMREAD_UNCHANGED,
                )
                if color_bgr is None or depth_raw is None or mask_full is None:
                    raise FileNotFoundError("query RGB/depth/mask is incomplete")
                color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                depth = depth_raw.astype(np.float32) / 1e3
                mask = mask_full == query_instance
                if int(mask.sum()) < args.min_mask_pixels:
                    raise ValueError(f"query mask has only {int(mask.sum())} pixels")
            except Exception as exc:
                registration_failures += 1
                frame_rows.append(_zero_row(pair, "query_input_failure", str(exc)))
                continue

            started = time.perf_counter()
            candidate_results: list[dict[str, object]] = []
            for candidate in candidates:
                label = str(candidate["label"])
                swap_candidate(estimator, candidate["cache"])
                try:
                    predicted = estimator.register(
                        K=REAL275_K,
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        iteration=args.register_iteration,
                        name=f"pair_{int(pair['pair_id']):04d}_{label}",
                        glctx=glctx,
                    )
                    l_depth, l_mask, l_geom, coverage = compute_observation_consistency(
                        estimator, depth, mask, REAL275_K, predicted, glctx
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Pair {pair['pair_id']} candidate {label} failed: {exc}"
                    )
                    continue
                reliability = candidate["anchor_reliability"]
                candidate_results.append(
                    {
                        "label": label,
                        "cache": candidate["cache"],
                        "pred_pose_q": np.asarray(predicted, dtype=np.float64),
                        "anchor_pose": candidate["anchor_pose"],
                        "s_anchor": candidate["s_anchor"],
                        "anchor_reliability": reliability["score"],
                        "anchor_low_reliability": reliability["low_reliability"],
                        "l_depth": float(l_depth),
                        "l_mask": float(l_mask),
                        "l_geom": float(l_geom),
                        "l_appearance": 1.0,
                        "coverage": float(np.clip(coverage, 0.0, 1.0)),
                    }
                )
            if not candidate_results:
                registration_failures += 1
                frame_rows.append(
                    _zero_row(pair, "registration_failure", "all candidates failed")
                )
                continue

            score_candidates(
                candidate_results,
                w_depth=args.w_depth,
                w_mask=args.w_mask,
                w_geom=args.w_geom,
                w_appearance=args.w_appearance,
                score_alpha=args.score_alpha,
                score_beta=args.score_beta,
                loss_normalization=args.query_loss_normalization,
                depth_scale=args.query_depth_scale,
                mask_scale=args.query_mask_scale,
                geom_scale=args.query_geom_scale,
                appearance_scale=args.query_appearance_scale,
            )
            ranked = sorted(
                candidate_results,
                key=lambda row: float(row["total_score"]),
                reverse=True,
            )
            winner = select_scored_candidate(candidate_results)
            runner_up = ranked[1] if len(ranked) > 1 else None
            score_margin = (
                float(winner["total_score"] - runner_up["total_score"])
                if runner_up is not None
                else float("nan")
            )
            elapsed = float(time.perf_counter() - started)
            selection_times.append(elapsed)
            rank_by_label = {
                str(row["label"]): rank for rank, row in enumerate(ranked, start=1)
            }
            candidate_audits: list[tuple[dict[str, object], dict[str, object]]] = []
            for result in candidate_results:
                audit = {
                    **identity,
                    "Candidate": result["label"],
                    "Selected": result is winner,
                    "S_anchor": result["s_anchor"],
                    "S_anchor_norm": result["s_anchor_norm"],
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
                    "Winning_Score": float(winner["total_score"]),
                    "Runner_Up_Candidate": (
                        str(runner_up["label"]) if runner_up is not None else ""
                    ),
                    "Score_Margin": score_margin,
                    "Mesh_Fingerprint": result["cache"].fingerprint,
                    "Selection_Time_s": elapsed,
                }
                candidate_audits.append((result, audit))

            # GT pose and CAD resources are deliberately touched only after Eq. (9).
            try:
                if object_name not in evaluation_cache:
                    evaluation_cache[object_name] = _load_real275_evaluation_resources(
                        obj_name=object_name,
                        model_dir=str(data_root / "obj_models" / "real_test"),
                        anchor_gt_path=str(anchor_gt_path),
                        models_info_path=str(args.models_info_path),
                        renderer=renderer,
                    )
                resources = evaluation_cache[object_name]
                gt_poses = load_gt_poses(
                    str(data_root / "gts" / "real_test"), query_scene, query_frame
                )
                if gt_poses is None or query_instance - 1 >= len(gt_poses):
                    raise ValueError("query GT pose is absent")
                query_gt_raw = gt_poses[query_instance - 1]
                _, query_gt = decompose_nocs_pose(query_gt_raw)
                relative = winner["pred_pose_q"] @ np.linalg.inv(winner["anchor_pose"])
                _, prediction = decompose_nocs_pose(
                    relative @ resources["anchor_gt_pose"]
                )
                metrics = _evaluate_real275_pose_metrics(
                    pred_eval=prediction,
                    gt_eval=query_gt,
                    gt_vertices_eval=resources["gt_vertices_eval"],
                    gt_diameter=resources["gt_diameter"],
                    is_symmetric=resources["is_symmetric"],
                    trans_disc=resources["trans_disc"],
                    intrinsic=REAL275_K,
                    depth_m=depth,
                    renderer=renderer,
                    object_id=resources["ob_id"],
                )
                if metrics is None:
                    raise ValueError("metric computation rejected the predicted pose")
            except Exception as exc:
                gt_failures += 1
                failed = _zero_row(pair, "evaluation_failure", str(exc))
                failed["Chosen_Candidate"] = str(winner["label"])
                failed["Selection_Time_s"] = elapsed
                frame_rows.append(failed)
                score_rows.extend(audit for _, audit in candidate_audits)
                continue

            row = {
                **identity,
                "Status": "ok",
                "Error": "",
                "Chosen_Candidate": str(winner["label"]),
                "Selection_Time_s": elapsed,
                "R_error": metrics["R_error"],
                "T_error": metrics["T_error"],
            }
            row.update({metric: float(metrics[metric]) for metric in METRICS})
            frame_rows.append(row)

            if args.save_candidate_metrics:
                for result, audit in candidate_audits:
                    try:
                        relative = result["pred_pose_q"] @ np.linalg.inv(
                            result["anchor_pose"]
                        )
                        _, candidate_prediction = decompose_nocs_pose(
                            relative @ resources["anchor_gt_pose"]
                        )
                        candidate_metrics = _evaluate_real275_pose_metrics(
                            pred_eval=candidate_prediction,
                            gt_eval=query_gt,
                            gt_vertices_eval=resources["gt_vertices_eval"],
                            gt_diameter=resources["gt_diameter"],
                            is_symmetric=resources["is_symmetric"],
                            trans_disc=resources["trans_disc"],
                            intrinsic=REAL275_K,
                            depth_m=depth,
                            renderer=renderer,
                            object_id=resources["ob_id"],
                        )
                        if candidate_metrics is not None:
                            audit.update(
                                {
                                    f"Metric_{key}": float(value)
                                    for key, value in candidate_metrics.items()
                                }
                            )
                    except Exception as exc:
                        audit["Metric_Error"] = str(exc)
            score_rows.extend(audit for _, audit in candidate_audits)

    frames = pd.DataFrame(frame_rows).sort_values("Pair_ID")
    if len(frames) != len(pairs) or frames["Pair_ID"].tolist() != list(range(len(pairs))):
        raise RuntimeError("Output rows do not cover every official pair exactly once")
    scores = pd.DataFrame(score_rows)
    frames.to_csv(output_root / "0_all_pairs_metrics.csv", index=False)
    frames.to_excel(output_root / "0_all_pairs_metrics.xlsx", index=False)
    scores.to_csv(output_root / "0_all_candidate_scores.csv", index=False)

    metric_summary = {
        metric: float(frames[metric].mean() * 100.0) for metric in METRICS
    }
    per_object_rows = []
    for object_name, group in frames.groupby("Class", sort=True):
        per_object_rows.append(
            {
                "Class": object_name,
                "Pairs": int(len(group)),
                **{metric: float(group[metric].mean() * 100.0) for metric in METRICS},
            }
        )
    pd.DataFrame(per_object_rows).to_csv(
        output_root / "0_per_object_metrics.csv", index=False
    )
    selection_counts = Counter(frames["Chosen_Candidate"].astype(str))
    selection_payload = {
        label: {
            "count": count,
            "percentage": 100.0 * count / len(frames),
        }
        for label, count in sorted(selection_counts.items())
    }
    summary = {
        "dataset": "REAL275",
        "protocol": "oryon_cross_scene_2k",
        "denominator_pairs": len(frames),
        "successful_pairs": int((frames["Status"] == "ok").sum()),
        "failed_pairs": int((frames["Status"] != "ok").sum()),
        "anchor_failure_pairs": anchor_failures,
        "registration_failure_pairs": registration_failures,
        "evaluation_failure_pairs": gt_failures,
        "metrics_percent": metric_summary,
        "candidate_selection": selection_payload,
        "runtime_seconds": {
            "median": float(np.median(selection_times)) if selection_times else None,
            "p95": float(np.percentile(selection_times, 95)) if selection_times else None,
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    run_manifest["completed"] = True
    run_manifest["summary"] = summary
    (output_root / "query_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {output_root}")


if __name__ == "__main__":
    main()
