#!/usr/bin/env python3
"""Validate completed Oryon anchor output while preserving failed references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oryon_real275_protocol import load_manifest
from scripts.verify_anchor_registry import SUPPORTED_LABELS, _require_file, _validate_pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-output", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        "--coverage-output",
        dest="output",
        type=Path,
        help="Optional JSON report path. Anchor artifacts are read-only by default.",
    )
    args = parser.parse_args()

    _, pairs = load_manifest(args.pair_manifest, require_official=True)
    expected = {str(row["anchor_key"]) for row in pairs}
    pair_counts: dict[str, int] = {}
    for row in pairs:
        key = str(row["anchor_key"])
        pair_counts[key] = pair_counts.get(key, 0) + 1

    registry_path = args.anchor_output / "candidate_registry.json"
    if not registry_path.is_file():
        raise SystemExit(
            f"Anchor run is incomplete: missing {registry_path}. "
            "Resume the anchor run before starting query evaluation."
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not registry:
        raise ValueError(f"Registry is empty: {registry_path}")
    extra = set(registry) - expected
    if extra:
        raise ValueError(f"Registry contains references outside the Oryon split: {sorted(extra)[:5]}")

    candidate_count = 0
    for anchor_key, entry in registry.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid registry entry for {anchor_key}")
        candidates = entry.get("candidates")
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError(f"No candidates for {anchor_key}")
        if set(candidates) - SUPPORTED_LABELS:
            raise ValueError(f"Unsupported candidate labels for {anchor_key}")
        if entry.get("chosen") not in candidates:
            raise ValueError(f"Chosen candidate is absent for {anchor_key}")
        _require_file(entry.get("K_path"), f"{anchor_key} intrinsics")
        final_pose = _require_file(entry.get("final_pose_path"), f"{anchor_key} final pose")
        _validate_pose(final_pose, f"{anchor_key} final pose")
        _require_file(entry.get("final_mesh_path"), f"{anchor_key} final mesh")
        for label, candidate in candidates.items():
            _require_file(candidate.get("mesh_path"), f"{anchor_key}/{label} mesh")
            pose = _require_file(candidate.get("pose_path"), f"{anchor_key}/{label} pose")
            _validate_pose(pose, f"{anchor_key}/{label} pose")
            candidate_count += 1

    missing = expected - set(registry)
    coverage = {
        "status": "valid",
        "official_pairs": len(pairs),
        "expected_unique_references": len(expected),
        "successful_unique_references": len(registry),
        "failed_unique_references": len(missing),
        "pairs_with_anchor_candidates": sum(pair_counts[key] for key in registry),
        "pairs_with_anchor_failure": sum(pair_counts[key] for key in missing),
        "candidate_artifacts": candidate_count,
        "missing_reference_examples": sorted(missing)[:20],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(coverage, sort_keys=True))


if __name__ == "__main__":
    main()
