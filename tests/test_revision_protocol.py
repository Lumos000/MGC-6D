from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from revision_protocol import (
    add_query_protocol_arguments,
    icp_direction_passes,
    runtime_provenance,
    strict_both_icp_passes,
    validate_query_protocol,
)
from scripts.verify_anchor_registry import validate_registry


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    "run_ho3d_full.sh",
    "run_ho3d_query.sh",
    "run_toyl_full.sh",
    "run_real275_full.sh",
)


def test_common_query_arguments_and_validation() -> None:
    parser = argparse.ArgumentParser()
    add_query_protocol_arguments(parser)
    args = parser.parse_args([])
    assert validate_query_protocol(args) == {
        "rayst3r",
        "instantmesh",
        "fused",
    }
    assert args.candidate_color_mode == "anchor_prior_hybrid"
    assert args.save_candidate_metrics is False
    assert parser.parse_args(["--save_candidate_metrics"]).save_candidate_metrics is True
    assert (
        parser.parse_args(["--no-save_candidate_metrics"]).save_candidate_metrics
        is False
    )


def test_invalid_candidate_and_weights_are_rejected() -> None:
    parser = argparse.ArgumentParser()
    add_query_protocol_arguments(parser)
    with pytest.raises(ValueError, match="Unknown candidate"):
        validate_query_protocol(
            parser.parse_args(["--candidate_labels", "rayst3r,bad"])
        )
    with pytest.raises(ValueError, match="query-loss"):
        validate_query_protocol(
            parser.parse_args(
                ["--w_depth", "0", "--w_mask", "0", "--w_geom", "0"]
            )
        )


def test_runtime_provenance_marks_untracked_files_dirty(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("released\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MGC-6D test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "clean",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert runtime_provenance(tmp_path)["git_dirty"] is False

    (tmp_path / "untracked.txt").write_text("local\n", encoding="utf-8")
    dirty = runtime_provenance(tmp_path)
    assert dirty["git_dirty"] is True
    assert dirty["git_status_sha256"] is not None


def test_two_direction_criterion_uses_independent_optima() -> None:
    threshold = {"tau_f": 0.05, "tau_r": 0.02}
    assert icp_direction_passes(0.05, 0.02, **threshold)
    assert strict_both_icp_passes(0.06, 0.01, 0.05, 0.02, **threshold)
    assert not strict_both_icp_passes(0.049, 0.01, 0.08, 0.01, **threshold)
    assert not strict_both_icp_passes(0.08, 0.021, 0.08, 0.01, **threshold)
    assert not strict_both_icp_passes(
        float("nan"), 0.01, 0.08, 0.01, **threshold
    )


def test_release_launchers_are_portable_and_executable() -> None:
    for name in LAUNCHERS:
        path = ROOT / "scripts" / name
        source = path.read_text(encoding="utf-8")
        assert path.stat().st_mode & 0o111
        assert "/data/gst/" not in source
        assert 'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"' in source


def test_anchor_launchers_validate_outputs_before_accepting_teardown_sigsegv() -> None:
    for name in ("run_ho3d_full.sh", "run_toyl_full.sh", "run_real275_full.sh"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        validation = source.index("scripts/verify_anchor_registry.py")
        warning = source.index("validated complete outputs and continuing")
        assert '"$ANCHOR_STATUS" != "139"' in source
        assert validation < warning


def test_anchor_registry_validator_requires_query_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "object_a").mkdir(parents=True)
    artifact = output / "object_a"
    artifact.mkdir(parents=True)
    pose = "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    for name, content in {
        "K.txt": "1 0 0\n0 1 0\n0 0 1\n",
        "final_pose.txt": pose,
        "candidate_pose.txt": pose,
        "final.obj": "v 0 0 0\n",
        "candidate.obj": "v 0 0 0\n",
    }.items():
        (artifact / name).write_text(content, encoding="utf-8")
    for name in ("chamfer_distances.xlsx", "pose_metrics.xlsx"):
        (output / name).write_bytes(b"complete")
    registry = {
        "object_a": {
            "chosen": "rayst3r",
            "K_path": str(artifact / "K.txt"),
            "final_pose_path": str(artifact / "final_pose.txt"),
            "final_mesh_path": str(artifact / "final.obj"),
            "candidates": {
                "rayst3r": {
                    "mesh_path": str(artifact / "candidate.obj"),
                    "pose_path": str(artifact / "candidate_pose.txt"),
                    "s_anchor": 0.5,
                }
            },
        }
    }
    registry_path = output / "candidate_registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    assert validate_registry(output, source)["status"] == "valid"
    (artifact / "candidate_pose.txt").unlink()
    with pytest.raises(ValueError, match="Missing or empty"):
        validate_registry(output, source)


def test_ho3d_missing_gt_rows_are_committed_only_after_gt_checks() -> None:
    source = (ROOT / "query_paper.py").read_text(encoding="utf-8")
    winner = source.index("winner, selection_diagnostics = _select_weighted_candidate")
    pending = source.index("frame_candidate_score_rows = []", winner)
    query_gt = source.index("gt_pose_q = reader.get_gt_pose(i)", pending)
    anchor_gt = source.index("if not os.path.exists(gt_anchor_path):", query_gt)
    commit = source.index(
        "candidate_score_rows.extend(frame_candidate_score_rows)", anchor_gt
    )
    assert winner < pending < query_gt < anchor_gt < commit
    assert "DINO query feature extraction through Eq.22 winner selection" not in source
    assert "candidate registration through Eq.22 winner selection" in source


def test_released_query_paths_have_no_learned_reranker() -> None:
    forbidden = ("reranker_checkpoint", "scorer_checkpoint", "frozen_pose_pool")
    for name in ("query_paper.py", "query_paper_toyl.py", "run_real275_query.py"):
        source = (ROOT / name).read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in source
