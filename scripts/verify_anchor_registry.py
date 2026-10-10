#!/usr/bin/env python3
"""Validate that an anchor phase produced a complete, query-ready registry.

The reconstruction process loads several native CUDA/graphics libraries.  On
some driver/library combinations those libraries can raise SIGSEGV during
interpreter teardown, after every output has already been flushed.  Launchers
may recover from exit status 139 only when this independent, stdlib-only
validator proves that the requested anchor set is complete.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SUPPORTED_LABELS = {"rayst3r", "instantmesh", "fused"}


def _csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _object_id(name: str) -> int | None:
    digits = "".join(character for character in name if character.isdigit())
    return int(digits) if digits else None


def _expected_objects(source_root: Path, objects: str, object_ids: str) -> set[str]:
    available = {
        path.name for path in source_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    }
    requested = _csv(objects)
    if requested:
        unknown = requested - available
        if unknown:
            raise ValueError(f"Requested objects absent from source root: {sorted(unknown)}")
        return requested
    requested_ids = {int(value) for value in _csv(object_ids)}
    if requested_ids:
        selected = {name for name in available if _object_id(name) in requested_ids}
        found_ids = {_object_id(name) for name in selected}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            raise ValueError(f"Requested object ids absent from source root: {sorted(missing_ids)}")
        return selected
    return available


def _require_file(value: object, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {description} path")
    path = Path(value)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty {description}: {path}")
    return path


def _validate_pose(path: Path, description: str) -> None:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append([float(value) for value in line.split()])
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"{description} is not a 4x4 matrix: {path}")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{description} contains a non-finite value: {path}")


def validate_registry(
    anchor_output: Path,
    source_root: Path,
    *,
    objects: str = "",
    object_ids: str = "",
    require_strict_both: bool = False,
    strict_tau_f: float | None = None,
    strict_tau_r: float | None = None,
) -> dict[str, object]:
    registry_path = anchor_output / "candidate_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not registry:
        raise ValueError(f"Registry is empty or not an object: {registry_path}")

    expected = _expected_objects(source_root, objects, object_ids)
    actual = set(registry)
    if actual != expected:
        raise ValueError(
            "Registry object set differs from requested anchor set: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    candidate_count = 0
    for object_name in sorted(expected):
        entry = registry[object_name]
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid registry entry for {object_name}")
        candidates = entry.get("candidates")
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError(f"No candidates recorded for {object_name}")
        unknown = set(candidates) - SUPPORTED_LABELS
        if unknown:
            raise ValueError(f"Unsupported candidates for {object_name}: {sorted(unknown)}")
        chosen = entry.get("chosen")
        if chosen not in candidates:
            raise ValueError(f"Chosen candidate is absent for {object_name}: {chosen!r}")

        if require_strict_both:
            alignment = entry.get("alignment_metadata", {})
            bidirectional = entry.get("bidirectional_metadata", {})
            if alignment.get("strict_both_gate") is not True:
                raise ValueError(f"Strict-both gate is not enabled for {object_name}")
            if bidirectional.get("strict_both_gate_source") != "independent_directional_optima":
                raise ValueError(
                    f"Strict-both does not use independent directional optima for {object_name}"
                )
            strict_pass = bidirectional.get("strict_both_pass")
            if not isinstance(strict_pass, bool):
                raise ValueError(f"Missing strict-both result for {object_name}")
            if "fused" in candidates and not strict_pass:
                raise ValueError(
                    f"Fused candidate survived a failed strict-both gate for {object_name}"
                )
            for key, expected_threshold in (
                ("strict_tau_f", strict_tau_f),
                ("strict_tau_r", strict_tau_r),
            ):
                value = bidirectional.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"Missing finite {key} for {object_name}: {value!r}")
                if expected_threshold is not None and not math.isclose(
                    float(value), float(expected_threshold), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"Unexpected {key} for {object_name}: "
                        f"{value!r} != {expected_threshold!r}"
                    )

        _require_file(entry.get("K_path"), f"{object_name} intrinsics")
        final_pose = _require_file(entry.get("final_pose_path"), f"{object_name} final pose")
        _validate_pose(final_pose, f"{object_name} final pose")
        _require_file(entry.get("final_mesh_path"), f"{object_name} final mesh")

        for label, candidate in candidates.items():
            if not isinstance(candidate, dict):
                raise ValueError(f"Invalid {object_name}/{label} candidate entry")
            _require_file(candidate.get("mesh_path"), f"{object_name}/{label} mesh")
            pose = _require_file(candidate.get("pose_path"), f"{object_name}/{label} pose")
            _validate_pose(pose, f"{object_name}/{label} pose")
            score = candidate.get("s_anchor")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ValueError(f"Invalid s_anchor for {object_name}/{label}: {score!r}")
            candidate_count += 1

    for filename in ("chamfer_distances.xlsx", "pose_metrics.xlsx"):
        _require_file(str(anchor_output / filename), filename)

    return {
        "registry": str(registry_path.resolve()),
        "objects": len(expected),
        "candidates": candidate_count,
        "strict_both": bool(require_strict_both),
        "status": "valid",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-output", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--objects", default="")
    parser.add_argument("--object-ids", default="")
    parser.add_argument("--require-strict-both", action="store_true")
    parser.add_argument("--strict-tau-f", type=float)
    parser.add_argument("--strict-tau-r", type=float)
    args = parser.parse_args()
    result = validate_registry(
        args.anchor_output,
        args.source_root,
        objects=args.objects,
        object_ids=args.object_ids,
        require_strict_both=args.require_strict_both,
        strict_tau_f=args.strict_tau_f,
        strict_tau_r=args.strict_tau_r,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
