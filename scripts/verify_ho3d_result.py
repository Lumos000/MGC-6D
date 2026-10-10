#!/usr/bin/env python3
"""Audit a completed HO3D run without rerunning pose estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PROTOCOL = {
    "selection_policy": "weighted",
    "candidate_color_mode": "anchor_prior_hybrid",
    "w_depth": 0.5,
    "w_mask": 0.25,
    "w_geom": 0.25,
    "w_appearance": 0.0,
    "score_alpha": 0.4,
    "score_beta": 0.6,
    "query_loss_normalization": "fixed_scale",
    "include_fallback_candidate": False,
    "symmetry_consistency_enable": False,
    "anchor_symmetry_align": False,
}


def _read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_excel(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--score-tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    root = args.result_dir.resolve()
    manifest_path = root / "query_run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_args = manifest.get("args", {})
    if not manifest.get("completed", False):
        raise RuntimeError("HO3D run manifest is not marked completed")

    mismatches = {
        key: {"expected": expected, "actual": run_args.get(key)}
        for key, expected in REQUIRED_PROTOCOL.items()
        if run_args.get(key) != expected
    }
    if run_args.get("candidate_labels") != "rayst3r,instantmesh,fused":
        mismatches["candidate_labels"] = {
            "expected": "rayst3r,instantmesh,fused",
            "actual": run_args.get("candidate_labels"),
        }
    if manifest.get("candidate_pool", {}).get("external_obs_candidate"):
        mismatches["external_obs_candidate"] = {
            "expected": False,
            "actual": True,
        }
    if mismatches:
        raise RuntimeError(
            "Result does not match the released HO3D protocol:\n"
            + json.dumps(mismatches, indent=2)
        )

    scores = _read_required(root / "0_all_candidate_scores.xlsx")
    frames = _read_required(root / "0_all_frames_metrics_results.xlsx")
    macro = _read_required(root / "0_mean_all_metrics_classes_results.xlsx")
    audit_columns = [column for column in scores.columns if str(column).startswith("Audit_")]
    if audit_columns:
        raise RuntimeError(f"GT candidate-audit columns are present: {audit_columns}")
    expected_score = (
        0.4 * scores["S_anchor_norm"].astype(float)
        + 0.6
        * (
            -0.5 * scores["L_depth_norm"].astype(float)
            -0.25 * scores["L_mask_norm"].astype(float)
            -0.25 * scores["L_geom_norm"].astype(float)
        )
    )
    residual = np.abs(expected_score - scores["Selection_Score"].astype(float))
    max_residual = float(residual.max()) if len(residual) else 0.0
    if max_residual > args.score_tolerance:
        raise RuntimeError(
            f"Eq. (22) score mismatch: max residual {max_residual:.3e}"
        )

    winner_mismatches = 0
    for _, group in scores.groupby(["Class", "Query_Frame_Index"], sort=False):
        selected = group[group["Selected"].astype(bool)]
        if len(selected) != 1:
            winner_mismatches += 1
            continue
        best_score = float(group["Selection_Score"].max())
        if not np.isclose(
            float(selected.iloc[0]["Selection_Score"]),
            best_score,
            atol=args.score_tolerance,
            rtol=0.0,
        ):
            winner_mismatches += 1
    if winner_mismatches:
        raise RuntimeError(f"Weighted-argmax winner mismatches: {winner_mismatches}")

    evaluated = frames[frames["Frame_ID"].astype(str) != "MEAN"].copy()
    query_keys = ["Class", "Query_Frame_Index"]
    missing_frame_keys = [key for key in query_keys if key not in evaluated.columns]
    if missing_frame_keys:
        raise RuntimeError(
            "Frame table is missing stable query keys: "
            + ", ".join(missing_frame_keys)
        )
    if evaluated.duplicated(query_keys).any():
        raise RuntimeError("Duplicate evaluated HO3D query keys in frame table")
    if scores.duplicated(query_keys + ["Candidate"]).any():
        raise RuntimeError("Duplicate candidate/query rows in HO3D score table")

    frame_keys = evaluated[query_keys].drop_duplicates()
    score_keys = scores[query_keys].drop_duplicates()
    key_check = frame_keys.merge(
        score_keys, on=query_keys, how="outer", indicator=True
    )
    if not (key_check["_merge"] == "both").all():
        counts = key_check["_merge"].value_counts().to_dict()
        raise RuntimeError(
            "Candidate-score queries differ from evaluated queries: "
            + json.dumps(counts, sort_keys=True)
        )

    selected_per_query = (
        scores.assign(_selected=scores["Selected"].astype(bool))
        .groupby(query_keys, dropna=False)["_selected"]
        .sum()
    )
    if not (selected_per_query == 1).all():
        raise RuntimeError(
            "Each evaluated HO3D query must have exactly one selected candidate"
        )
    distribution = (
        evaluated["Chosen_Candidate"].value_counts().sort_index().to_dict()
    )
    macro_row = macro[macro["Class_ID"].astype(str) == "MEAN"]
    if len(macro_row) != 1:
        raise RuntimeError("Missing unique MEAN row in macro result file")
    metric_names = ["ADD-S", "ADD", "AR", "MSSD", "MSPD", "VSD"]
    metrics = {
        name: float(macro_row.iloc[0][name]) for name in metric_names
    }

    coverage_path = root / "evaluation_coverage.json"
    coverage = None
    if coverage_path.is_file():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        if not coverage.get("complete", False):
            raise RuntimeError(f"Incomplete evaluation coverage: {coverage}")
        if int(coverage.get("evaluated_query_frames", -1)) != len(evaluated):
            raise RuntimeError("Coverage count differs from the saved frame table")
        if len(score_keys) != len(evaluated):
            raise RuntimeError(
                "Candidate-score query count differs from evaluated coverage"
            )

    runtime = manifest.get("runtime", {})
    expected_measurement = "candidate registration through Eq.22 winner selection"
    if runtime.get("measurement") != expected_measurement:
        raise RuntimeError(
            "Incorrect runtime measurement description: "
            f"{runtime.get('measurement')!r}"
        )
    if int(runtime.get("frames", -1)) != len(evaluated):
        raise RuntimeError("Runtime frame count differs from evaluated coverage")

    count_json_path = root / "0_candidate_selection_counts.json"
    count_xlsx_path = root / "0_candidate_selection_counts.xlsx"
    if not count_json_path.is_file() or not count_xlsx_path.is_file():
        raise RuntimeError("Missing HO3D candidate-selection count outputs")
    count_payload = json.loads(count_json_path.read_text(encoding="utf-8"))
    saved_distribution = {
        str(row["Candidate"]): int(row["Selected_Count"])
        for row in count_payload.get("overall", [])
    }
    if int(count_payload.get("total_frames", -1)) != len(evaluated):
        raise RuntimeError("Saved candidate-selection total differs from coverage")
    if saved_distribution != distribution:
        raise RuntimeError(
            "Saved candidate-selection distribution differs from frame table"
        )

    report = {
        "result_dir": str(root),
        "evaluated_frames": int(len(evaluated)),
        "metrics_macro_percent": metrics,
        "candidate_distribution": distribution,
        "eq22_max_abs_residual": max_residual,
        "weighted_argmax_mismatches": 0,
        "audit_columns": [],
        "coverage": coverage,
        "runtime": runtime,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
