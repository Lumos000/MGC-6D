#!/usr/bin/env python3
"""Independently verify an Oryon TOYL result and Eq. (9) selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oryon_toyl_protocol import (
    ORYON_TOYL_PAIR_COUNT,
    ORYON_TOYL_PROTOCOL,
    ORYON_TOYL_SHA256,
    load_manifest,
)


METRICS = ("ADD", "ADD-S", "ADD(-S)", "AR", "MSSD", "MSPD", "VSD")
SCORED_STATUSES = {"ok", "evaluation_failure"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--allow-nonofficial-split", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. The result directory is read-only by default.",
    )
    args = parser.parse_args()
    root = args.result_dir.resolve()
    run = json.loads((root / "query_run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    _, protocol_pairs = load_manifest(
        run["pair_manifest"],
        require_official=not args.allow_nonofficial_split,
    )
    if run.get("completed") is not True:
        raise ValueError("Query manifest is not marked complete")
    if run.get("protocol") != ORYON_TOYL_PROTOCOL:
        raise ValueError("Query manifest has the wrong protocol")
    if run.get("pair_protocol", ORYON_TOYL_PROTOCOL) != ORYON_TOYL_PROTOCOL:
        raise ValueError("Query manifest has the wrong pair protocol")
    if not args.allow_nonofficial_split:
        if run.get("official_pair_file_sha256") != ORYON_TOYL_SHA256:
            raise ValueError("Query manifest does not identify the official pair file")
        if len(protocol_pairs) != ORYON_TOYL_PAIR_COUNT:
            raise ValueError("Result does not use exactly 2,000 official pairs")

    frames = pd.read_csv(root / "0_all_pairs_metrics.csv")
    scores = pd.read_csv(root / "0_all_candidate_scores.csv")
    if len(frames) != len(protocol_pairs):
        raise ValueError("Result does not contain every requested protocol pair")
    expected_ids = np.arange(len(protocol_pairs), dtype=np.int64)
    if not np.array_equal(frames["Pair_ID"].to_numpy(dtype=np.int64), expected_ids):
        raise ValueError("Pair_ID coverage is incomplete or reordered")
    if not set(scores["Pair_ID"].astype(int)).issubset(set(expected_ids)):
        raise ValueError("Candidate score table contains an unknown pair")

    selection = run["selection"]
    replay = (
        float(selection["score_alpha"]) * scores["S_anchor_norm"]
        + float(selection["score_beta"])
        * (
            -float(selection["w_depth"]) * scores["L_depth_norm"]
            -float(selection["w_mask"]) * scores["L_mask_norm"]
            -float(selection["w_geom"]) * scores["L_geom_norm"]
        )
    )
    if not np.allclose(
        replay.to_numpy(dtype=np.float64),
        scores["Total_Score"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=args.atol,
    ):
        raise ValueError("Saved Eq. (9) scores do not match independent replay")

    grouped = {int(pair_id): group for pair_id, group in scores.groupby("Pair_ID")}
    for _, frame in frames.iterrows():
        pair_id = int(frame["Pair_ID"])
        status = str(frame["Status"])
        group = grouped.get(pair_id)
        if status not in SCORED_STATUSES:
            if group is not None:
                raise ValueError(f"Failed-before-selection pair {pair_id} has scores")
            continue
        if group is None:
            raise ValueError(f"Scored pair {pair_id} has no candidate rows")
        selected = group[group["Selected"].astype(str).str.lower() == "true"]
        if len(selected) != 1:
            raise ValueError(f"Pair {pair_id} does not have exactly one selected row")
        if not np.isclose(
            float(selected.iloc[0]["Total_Score"]),
            float(group["Total_Score"].max()),
            rtol=0.0,
            atol=args.atol,
        ):
            raise ValueError(f"Pair {pair_id} selected a non-argmax candidate")
        if str(frame["Chosen_Candidate"]) != str(selected.iloc[0]["Candidate"]):
            raise ValueError(f"Pair {pair_id} winner differs between output tables")

    replay_metrics = {
        metric: float(frames[metric].mean() * 100.0) for metric in METRICS
    }
    for metric, value in replay_metrics.items():
        if not np.isclose(
            value,
            float(summary["metrics_percent"][metric]),
            rtol=0.0,
            atol=args.atol,
        ):
            raise ValueError(f"Summary metric mismatch for {metric}")
    if int(summary["denominator_pairs"]) != len(protocol_pairs):
        raise ValueError("Summary denominator does not match the protocol")
    if int(summary["successful_pairs"]) + int(summary["failed_pairs"]) != len(protocol_pairs):
        raise ValueError("Success/failure coverage does not match the protocol")

    result = {
        "status": "valid",
        "pairs": len(frames),
        "candidate_rows": len(scores),
        "score_max_abs_error": float(
            np.max(np.abs(replay.to_numpy() - scores["Total_Score"].to_numpy()))
        ) if len(scores) else 0.0,
        "metrics_percent": replay_metrics,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
