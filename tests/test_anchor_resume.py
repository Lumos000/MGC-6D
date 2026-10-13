from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor_resume import (
    atomic_write_json,
    completed_reconstruction,
    load_registry_checkpoints,
    registry_entry_complete,
)


def _write(path: Path, contents: str = "x") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return str(path.resolve())


def test_completed_reconstruction_reuses_only_committed_outputs(tmp_path: Path) -> None:
    output = tmp_path / "reference"
    paths = {
        "pointcloud": _write(output / "points.ply"),
        "mesh_rayst3r": _write(output / "ray.obj"),
        "mesh_instant": _write(output / "instant.obj"),
        "mesh_fused": None,
    }
    metadata = {
        "object": "reference",
        "outputs": paths,
        "base_config": {"seed": 2, "object_names": ["reference"]},
    }
    atomic_write_json(output / "reconstruction_meta.json", metadata)

    resumed = completed_reconstruction(
        output,
        "reference",
        {"seed": 2, "object_names": ("different-runtime-list",), "resume_existing": True},
    )
    assert resumed is not None
    assert resumed["mesh_rayst3r"] == paths["mesh_rayst3r"]
    assert "mesh_fused" not in resumed

    Path(paths["pointcloud"]).unlink()
    assert completed_reconstruction(output, "reference", {"seed": 2}) is None


def test_completed_reconstruction_rejects_configuration_mixing(tmp_path: Path) -> None:
    output = tmp_path / "reference"
    metadata = {
        "object": "reference",
        "outputs": {
            "pointcloud": _write(output / "points.ply"),
            "mesh_instant": _write(output / "instant.obj"),
            "mesh_rayst3r": None,
            "mesh_fused": None,
        },
        "base_config": {"seed": 2, "fusion_mode": "visible_patch"},
    }
    atomic_write_json(output / "reconstruction_meta.json", metadata)
    with pytest.raises(ValueError, match="fusion_mode"):
        completed_reconstruction(
            output, "reference", {"seed": 2, "fusion_mode": "poisson"}
        )


def test_registry_checkpoint_survives_missing_final_registry(tmp_path: Path) -> None:
    artifact = _write(tmp_path / "artifact.dat")
    entry = {
        "chosen": "rayst3r",
        "K_path": artifact,
        "final_pose_path": artifact,
        "final_mesh_path": artifact,
        "candidates": {
            "rayst3r": {"mesh_path": artifact, "pose_path": artifact}
        },
    }
    assert registry_entry_complete(entry)
    checkpoints = tmp_path / "entries"
    checkpoints.mkdir()
    (checkpoints / "reference.json").write_text(
        json.dumps({"object": "reference", "entry": entry}), encoding="utf-8"
    )

    loaded = load_registry_checkpoints(tmp_path / "absent.json", checkpoints)
    assert loaded == {"reference": entry}


def test_atomic_json_never_leaves_temporary_file(tmp_path: Path) -> None:
    destination = tmp_path / "value.json"
    atomic_write_json(destination, {"value": 1})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"value": 1}
    assert not list(tmp_path.glob(".value.json.tmp-*"))
