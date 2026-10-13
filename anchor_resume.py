"""Crash-safe helpers shared by long-running anchor pipelines.

The Oryon 2K protocol contains more than one thousand distinct references.
Native CUDA/Open3D failures cannot be caught by Python, so a completed object
must be recognizable and reusable after the process is restarted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


_RUNTIME_CONFIG_KEYS = {"output_root", "object_names", "resume_existing"}


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Write JSON atomically so a killed process cannot leave valid-looking data."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _normalized(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    return value


def completed_reconstruction(
    output_dir: str | os.PathLike[str],
    object_name: str,
    current_config: Mapping[str, Any],
) -> dict[str, str] | None:
    """Return reusable reconstruction paths, or ``None`` when it is incomplete.

    A present metadata file is the commit marker because construction writes it
    only after all exported geometries.  Configuration disagreement is an
    error, not a cache miss: silently mixing experiment settings would make the
    result irreproducible.
    """

    output_dir = Path(output_dir)
    metadata_path = output_dir / "reconstruction_meta.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict) or metadata.get("object") != object_name:
        return None

    recorded_config = metadata.get("base_config")
    if not isinstance(recorded_config, dict):
        raise ValueError(
            f"Cannot safely resume {object_name}: reconstruction metadata has no base_config"
        )
    mismatches: list[str] = []
    for key in sorted(set(recorded_config) & set(current_config)):
        if key in _RUNTIME_CONFIG_KEYS:
            continue
        if _normalized(recorded_config[key]) != _normalized(current_config[key]):
            mismatches.append(key)
    if mismatches:
        preview = ", ".join(mismatches[:8])
        raise ValueError(
            f"Cannot resume {object_name}: reconstruction settings differ: {preview}. "
            "Choose a fresh ANCHOR_OUT or use the original settings."
        )

    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict):
        return None
    required = {"pointcloud": "pointcloud", "mesh_instant": "mesh_instant"}
    reusable: dict[str, str] = {"meta": str(metadata_path.resolve())}
    for metadata_key, result_key in required.items():
        value = outputs.get(metadata_key)
        if not _nonempty_file(value):
            return None
        reusable[result_key] = str(Path(value).resolve())
    for metadata_key, result_key in (
        ("mesh_rayst3r", "mesh_rayst3r"),
        ("mesh_fused", "mesh_fused"),
    ):
        value = outputs.get(metadata_key)
        if value is not None:
            if not _nonempty_file(value):
                return None
            reusable[result_key] = str(Path(value).resolve())
    return reusable


def _nonempty_file(path: Any) -> bool:
    if not isinstance(path, str) or not path:
        return False
    candidate = Path(path)
    return candidate.is_file() and candidate.stat().st_size > 0


def registry_entry_complete(entry: Any) -> bool:
    """Check whether one anchor registry entry and all its artifacts are usable."""

    if not isinstance(entry, dict):
        return False
    candidates = entry.get("candidates")
    chosen = entry.get("chosen")
    if not isinstance(candidates, dict) or not candidates or chosen not in candidates:
        return False
    for key in ("K_path", "final_pose_path", "final_mesh_path"):
        if not _nonempty_file(entry.get(key)):
            return False
    for candidate in candidates.values():
        if not isinstance(candidate, dict):
            return False
        if not _nonempty_file(candidate.get("mesh_path")):
            return False
        if not _nonempty_file(candidate.get("pose_path")):
            return False
    return True


def load_registry_checkpoints(
    registry_path: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
) -> dict[str, dict[str, Any]]:
    """Load valid entries from a final registry and per-reference checkpoints."""

    entries: dict[str, dict[str, Any]] = {}
    registry_path = Path(registry_path)
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            registry = {}
        if isinstance(registry, dict):
            entries.update(
                (str(key), value)
                for key, value in registry.items()
                if registry_entry_complete(value)
            )
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            object_name = record.get("object")
            entry = record.get("entry")
            if isinstance(object_name, str) and registry_entry_complete(entry):
                entries[object_name] = entry
    return entries
