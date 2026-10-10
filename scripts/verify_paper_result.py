#!/usr/bin/env python3
"""Verify a completed TOYL or REAL275 paper-protocol result directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOLS = {
    "toyl": {
        "dataset": "TOYL",
        "frames": 1669,
        "frame_keys": ["Scene", "Image_ID", "Anno_ID", "Class"],
    },
    "real275": {
        "dataset": "REAL275",
        "frames": 1627,
        "frame_keys": ["Scene", "Frame_Num", "Class"],
    },
}
PAPER_LABELS = {"rayst3r", "instantmesh", "fused"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_mean_metrics(result_dir: Path) -> dict[str, float]:
    path = result_dir / "0_mean_all_metrics_classes_results.xlsx"
    _require(path.is_file(), f"Missing metric summary: {path}")
    frame = pd.read_excel(path)
    mean_rows = frame[frame["Class_ID"].astype(str) == "MEAN"]
    _require(len(mean_rows) == 1, "Metric summary must contain exactly one MEAN row")
    row = mean_rows.iloc[0]
    return {
        metric: float(row[metric])
        for metric in ("ADD-S", "ADD", "ADD(-S)", "AR", "MSSD", "MSPD", "VSD")
    }


def verify(dataset: str, result_dir: Path) -> dict:
    protocol = PROTOCOLS[dataset]
    manifest_path = result_dir / "query_run_manifest.json"
    score_path = result_dir / "0_all_candidate_scores.xlsx"
    frame_path = result_dir / "0_all_frames_metrics_results.xlsx"
    coverage_path = result_dir / "evaluation_coverage.json"
    for path in (manifest_path, score_path, frame_path, coverage_path):
        _require(path.is_file(), f"Missing required output: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    _require(manifest.get("dataset") == protocol["dataset"], "Dataset mismatch")
    _require(bool(manifest.get("completed")), "Run manifest is not marked completed")

    frames = pd.read_excel(frame_path)
    frames = frames[frames["Frame_ID"].astype(str) != "MEAN"]
    evaluated_frames = int(len(frames))
    _require(
        evaluated_frames == protocol["frames"],
        f"Expected {protocol['frames']} evaluated frames, got {evaluated_frames}",
    )
    _require(
        int(coverage.get("evaluated_frame_instances", -1)) == evaluated_frames,
        "Coverage/evaluated-frame mismatch",
    )

    scores = pd.read_excel(score_path)
    _require(len(scores) > 0, "Candidate score table is empty")
    forbidden = [
        column
        for column in scores.columns
        if str(column).startswith("Audit_")
        or any(token in str(column).lower() for token in ("hypothesis", "cluster", "oracle"))
    ]
    _require(not forbidden, f"Forbidden diagnostic columns: {forbidden}")
    labels = set(scores["Candidate"].dropna().astype(str))
    _require(labels <= PAPER_LABELS, f"Unexpected candidate labels: {sorted(labels - PAPER_LABELS)}")

    selection = manifest.get("selection", {})
    alpha = float(selection.get("score_alpha", 0.4))
    beta = float(selection.get("score_beta", 0.6))
    replay = alpha * scores["S_anchor_norm"].astype(float) + beta * (
        -float(selection.get("w_depth", 0.5)) * scores["L_depth_norm"].astype(float)
        -float(selection.get("w_mask", 0.25)) * scores["L_mask_norm"].astype(float)
        -float(selection.get("w_geom", 0.25)) * scores["L_geom_norm"].astype(float)
    )
    residual = np.abs(replay.to_numpy() - scores["Total_Score"].astype(float).to_numpy())
    max_residual = float(residual.max(initial=0.0))
    _require(max_residual <= 1e-9, f"Eq.22 replay residual is {max_residual:.3e}")

    winner_mismatches = 0
    selected_total = 0
    for _, group in scores.groupby(protocol["frame_keys"], dropna=False, sort=False):
        selected = group[group["Selected"].astype(bool)]
        selected_total += len(selected)
        if len(selected) != 1:
            winner_mismatches += 1
            continue
        if float(selected.iloc[0]["Total_Score"]) < float(group["Total_Score"].max()) - 1e-12:
            winner_mismatches += 1
    _require(winner_mismatches == 0, f"Eq.22 winner mismatches: {winner_mismatches}")
    _require(selected_total == evaluated_frames, "Selected-candidate/frame count mismatch")

    metrics = _read_mean_metrics(result_dir)
    runtime = manifest.get("runtime", {})
    median_seconds = runtime.get("median_seconds")
    _require(median_seconds is not None, "Manifest does not contain runtime median")

    forbidden_outputs = [
        path.name
        for path in result_dir.iterdir()
        if any(token in path.name.lower() for token in ("pose_pool", "hypothesis", "cluster", "oracle"))
    ]
    _require(not forbidden_outputs, f"Forbidden outputs: {forbidden_outputs}")
    return {
        "dataset": protocol["dataset"],
        "frames": evaluated_frames,
        "metrics": metrics,
        "max_eq22_residual": max_residual,
        "winner_mismatches": winner_mismatches,
        "median_seconds": float(median_seconds),
        "candidate_counts": frames["Chosen_Candidate"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(PROTOCOLS))
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    summary = verify(args.dataset, args.result_dir.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
