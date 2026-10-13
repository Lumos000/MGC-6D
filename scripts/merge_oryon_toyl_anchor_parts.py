#!/usr/bin/env python3
"""Merge resumable per-object TOYL Oryon anchor registries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oryon_toyl_protocol import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-nonofficial-split", action="store_true")
    args = parser.parse_args()

    _, pairs = load_manifest(
        args.pair_manifest,
        require_official=not args.allow_nonofficial_split,
    )
    expected_by_object: dict[int, set[str]] = {}
    for row in pairs:
        expected_by_object.setdefault(int(row["object_id"]), set()).add(
            str(row["anchor_key"])
        )

    merged: dict[str, object] = {}
    part_summary: dict[str, object] = {}
    for obj_id in sorted(expected_by_object):
        part = args.parts_root / f"obj_{obj_id:02d}"
        registry_path = part / "candidate_registry.json"
        if not registry_path.is_file():
            raise FileNotFoundError(f"Missing completed anchor part: {registry_path}")
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(registry, dict):
            raise ValueError(f"Invalid part registry: {registry_path}")
        unexpected = set(registry) - expected_by_object[obj_id]
        if unexpected:
            raise ValueError(
                f"Part obj_{obj_id:02d} contains anchors for another object: "
                f"{sorted(unexpected)[:5]}"
            )
        duplicate = set(registry) & set(merged)
        if duplicate:
            raise ValueError(f"Duplicate anchor keys across parts: {sorted(duplicate)[:5]}")
        merged.update(registry)
        part_summary[f"obj_{obj_id:02d}"] = {
            "expected_references": len(expected_by_object[obj_id]),
            "successful_references": len(registry),
            "registry": str(registry_path.resolve()),
        }
    if not merged:
        raise ValueError("No successful anchor candidates were produced")

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merged anchor output: {output}")
    output.mkdir(parents=True)
    registry_path = output / "candidate_registry.json"
    temporary = registry_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, registry_path)
    merge_manifest = {
        "dataset": "TOYL",
        "protocol": "oryon_toyl_2k",
        "pair_manifest": str(args.pair_manifest.resolve()),
        "parts_root": str(args.parts_root.resolve()),
        "expected_references": sum(map(len, expected_by_object.values())),
        "successful_references": len(merged),
        "parts": part_summary,
    }
    (output / "anchor_batch_manifest.json").write_text(
        json.dumps(merge_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(merge_manifest, sort_keys=True))


if __name__ == "__main__":
    main()
