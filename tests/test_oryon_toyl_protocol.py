from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pytest

from oryon_toyl_protocol import (
    ORYON_TOYL_MEMBER,
    anchor_key,
    parse_instance_list_text,
    unique_reference_rows,
)
from scripts.fetch_oryon_real275_pairs import extract_local_member


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pose(obj_id: int, z_mm: float) -> dict:
    return {
        "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
        "cam_t_m2c": [0.0, 0.0, z_mm],
        "obj_id": obj_id,
    }


def _make_scene(root: Path, obj_id: int = 1) -> None:
    scene = root / "test" / f"{obj_id:06d}"
    for child in ("rgb", "depth", "mask_visib"):
        (scene / child).mkdir(parents=True, exist_ok=True)
    gt = {"0": [_pose(obj_id, 500.0)], "1": [_pose(obj_id, 510.0)], "2": [_pose(obj_id, 520.0)]}
    camera = {
        str(frame): {
            "cam_K": [500.0, 0.0, 2.0, 0.0, 500.0, 2.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        for frame in range(3)
    }
    (scene / "scene_gt.json").write_text(json.dumps(gt), encoding="utf-8")
    (scene / "scene_camera.json").write_text(json.dumps(camera), encoding="utf-8")
    for frame in range(3):
        rgb = np.full((4, 5, 3), 30 + frame, dtype=np.uint8)
        depth = np.full((4, 5), 500 + frame, dtype=np.uint16)
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 1:4] = 255
        assert cv2.imwrite(str(scene / "rgb" / f"{frame:06d}.png"), rgb)
        assert cv2.imwrite(str(scene / "depth" / f"{frame:06d}.png"), depth)
        assert cv2.imwrite(
            str(scene / "mask_visib" / f"{frame:06d}_000000.png"), mask
        )


def test_parse_toyl_pairs_and_unique_references() -> None:
    text = "test, 1 0, 1 1, 1\ntest, 1 0, 1 2, 1\ntest, 1 1, 1 2, 1\n"
    pairs = parse_instance_list_text(text)
    assert len(pairs) == 3
    rows = [
        {
            "anchor_key": anchor_key(pair.reference_scene, pair.reference_frame, pair.object_id),
            "reference_scene": pair.reference_scene,
            "reference_frame": pair.reference_frame,
            "reference_anno_idx": 0,
            "object_id": pair.object_id,
        }
        for pair in pairs
    ]
    assert len(unique_reference_rows(rows)) == 2


@pytest.mark.parametrize(
    "row",
    (
        "test, 1 0, 2 1, 1",
        "test, 1 0, 1 0, 1",
        "train, 1 0, 1 1, 1",
        "test, 1 0, 1 1, 22",
    ),
)
def test_parser_rejects_invalid_toyl_contract(row: str) -> None:
    with pytest.raises(ValueError):
        parse_instance_list_text(row + "\n")


def test_local_archive_member_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "oryon.zip"
    expected = b"test, 1 0, 1 1, 1\n"
    with ZipFile(archive, "w") as handle:
        handle.writestr(ORYON_TOYL_MEMBER, expected)
    assert extract_local_member(archive, ORYON_TOYL_MEMBER) == expected


def test_prepare_deduplicates_reference_and_preserves_bop_pose(tmp_path: Path) -> None:
    dataset = tmp_path / "toyl"
    _make_scene(dataset)
    pair_file = tmp_path / "pairs.txt"
    pair_file.write_text(
        "test, 1 0, 1 1, 1\ntest, 1 0, 1 2, 1\n", encoding="utf-8"
    )
    output = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "prepare_oryon_toyl.py"),
            "--toyl-root",
            str(dataset),
            "--pairs-file",
            str(pair_file),
            "--output-root",
            str(output),
            "--allow-nonofficial-split",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    key = anchor_key(1, 0, 1)
    anchor = output / key
    assert (anchor / "color.png").is_symlink()
    assert (anchor / "depth.png").is_symlink()
    assert np.count_nonzero(cv2.imread(str(anchor / "mask.png"), 0)) == 6
    pose = np.loadtxt(anchor / f"{key}_gt_pose.txt")
    assert pose[2, 3] == 500.0
    manifest = json.loads((output / "oryon_toyl_manifest.json").read_text())
    assert manifest["pair_count"] == 2
    assert manifest["unique_reference_count"] == 1
    assert len(list(path for path in output.iterdir() if path.is_dir())) == 1


def test_query_source_selects_before_loading_pose_gt() -> None:
    source = (REPO_ROOT / "run_toyl_oryon_query.py").read_text(encoding="utf-8")
    selection = source.index("winner = select_scored_candidate(results)")
    evaluation_marker = source.index("# Evaluation-only inputs begin here")
    anchor_gt = source.index("anchor_gt = _gt_pose_to_meters", evaluation_marker)
    scene_gt = source.index('"scene_gt.json"', evaluation_marker)
    assert selection < evaluation_marker < anchor_gt
    assert selection < scene_gt


def test_independent_launcher_has_separate_anchor_and_query_stages() -> None:
    launcher = (REPO_ROOT / "scripts" / "run_toyl_oryon_2k.sh").read_text()
    assert "anchor_paper_toyl_oryon.py" in launcher
    assert "merge_oryon_toyl_anchor_parts.py" in launcher
    assert "run_toyl_oryon_query.py" in launcher
    assert 'RUN_ANCHOR="${RUN_ANCHOR:-1}"' in launcher
    assert 'RUN_QUERY="${RUN_QUERY:-1}"' in launcher
    wrapper = (REPO_ROOT / "anchor_paper_toyl_oryon.py").read_text()
    assert 're.search(r"_obj_(\\d+)$"' in wrapper
    query_source = (REPO_ROOT / "run_toyl_oryon_query.py").read_text()
    assert "--shared_geometry_registry" not in query_source
    assert '"method_variant": "pair_specific_geometry_reconstruction"' in query_source


def test_result_verifier_replays_eq22_and_pair_coverage(tmp_path: Path) -> None:
    result = tmp_path / "result"
    result.mkdir()
    rows = []
    for pair in parse_instance_list_text(
        "test, 1 0, 1 1, 1\ntest, 1 1, 1 2, 1\n"
    ):
        rows.append(
            {
                "pair_id": pair.pair_id,
                "anchor_key": anchor_key(
                    pair.reference_scene, pair.reference_frame, pair.object_id
                ),
                "object_id": pair.object_id,
                "object_name": "obj_01",
                "reference_scene": pair.reference_scene,
                "reference_frame": pair.reference_frame,
                "reference_anno_idx": 0,
                "query_scene": pair.query_scene,
                "query_frame": pair.query_frame,
                "query_anno_idx": 0,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "TOYL",
                "protocol": "oryon_toyl_2k",
                "pair_file_sha256": "synthetic",
                "unique_reference_count": 2,
                "pairs": rows,
            }
        ),
        encoding="utf-8",
    )
    (result / "query_run_manifest.json").write_text(
        json.dumps(
            {
                "completed": True,
                "protocol": "oryon_toyl_2k",
                "pair_manifest": str(manifest_path),
                "selection": {
                    "score_alpha": 0.4,
                    "score_beta": 0.6,
                    "w_depth": 0.5,
                    "w_mask": 0.25,
                    "w_geom": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )
    frames = pd.DataFrame(
        [
            {
                "Pair_ID": pair_id,
                "Status": "ok",
                "Chosen_Candidate": "rayst3r",
                **{metric: 1.0 for metric in ("ADD", "ADD-S", "ADD(-S)", "AR", "MSSD", "MSPD", "VSD")},
            }
            for pair_id in range(2)
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "Pair_ID": pair_id,
                "Candidate": "rayst3r",
                "Selected": True,
                "S_anchor_norm": 1.0,
                "L_depth_norm": 0.1,
                "L_mask_norm": 0.2,
                "L_geom_norm": 0.3,
                "Total_Score": 0.4 + 0.6 * -(0.5 * 0.1 + 0.25 * 0.2 + 0.25 * 0.3),
            }
            for pair_id in range(2)
        ]
    )
    frames.to_csv(result / "0_all_pairs_metrics.csv", index=False)
    scores.to_csv(result / "0_all_candidate_scores.csv", index=False)
    (result / "summary.json").write_text(
        json.dumps(
            {
                "denominator_pairs": 2,
                "successful_pairs": 2,
                "failed_pairs": 0,
                "metrics_percent": {
                    metric: 100.0
                    for metric in ("ADD", "ADD-S", "ADD(-S)", "AR", "MSSD", "MSPD", "VSD")
                },
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_oryon_toyl_result.py"),
            str(result),
            "--allow-nonofficial-split",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert json.loads((result / "verification.json").read_text())["status"] == "valid"
