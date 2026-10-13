from __future__ import annotations

import json
from pathlib import Path
import pickle
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from oryon_real275_protocol import (
    ORYON_REAL275_MEMBER,
    anchor_key,
    load_manifest,
    manifest_pair_dict,
    parse_instance_list_text,
    resolve_instance,
    unique_reference_rows,
)
from scripts.fetch_oryon_real275_pairs import extract_local_member
from scripts import prepare_oryon_real275


ROOT = Path(__file__).resolve().parents[1]


def test_pair_parser_and_unique_reference_cache_contract() -> None:
    pairs = parse_instance_list_text(
        "\n".join(
            [
                "real_test, 1 20, 4 39, 3 camera_canon_len_norm",
                "real_test, 1 20, 5 40, 3 camera_canon_len_norm",
            ]
        )
    )
    assert [pair.pair_id for pair in pairs] == [0, 1]
    assert pairs[0].reference_identity == pairs[1].reference_identity
    rows = [
        manifest_pair_dict(pair, reference_instance_id=4, query_instance_id=2)
        for pair in pairs
    ]
    assert len(unique_reference_rows(rows)) == 1
    assert rows[0]["anchor_key"] == anchor_key(
        1, 20, 4, "camera_canon_len_norm"
    )


def test_cross_scene_parser_rejects_same_scene() -> None:
    with pytest.raises(ValueError, match="same scene"):
        parse_instance_list_text(
            "real_test, 1 20, 1 39, 3 camera_canon_len_norm\n"
        )


def test_local_official_archive_member_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "oryon_data.zip"
    expected = b"real_test, 1 20, 4 39, 3 camera_canon_len_norm\n"
    with ZipFile(archive, "w") as handle:
        handle.writestr(ORYON_REAL275_MEMBER, expected)
    assert extract_local_member(archive, ORYON_REAL275_MEMBER) == expected


def _write_frame(
    root: Path,
    *,
    scene: int,
    frame: int,
    instance_id: int,
    category_id: int,
    object_name: str,
) -> None:
    scene_root = root / "real_test" / f"scene_{scene}"
    scene_root.mkdir(parents=True, exist_ok=True)
    color = np.zeros((8, 10, 3), dtype=np.uint8)
    depth = np.full((8, 10), 700, dtype=np.uint16)
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:7, 3:9] = instance_id
    assert cv2.imwrite(str(scene_root / f"{frame:04d}_color.png"), color)
    assert cv2.imwrite(str(scene_root / f"{frame:04d}_depth.png"), depth)
    assert cv2.imwrite(str(scene_root / f"{frame:04d}_mask.png"), mask)
    (scene_root / f"{frame:04d}_meta.txt").write_text(
        f"{instance_id} {category_id} {object_name}\n", encoding="utf-8"
    )
    gt_root = root / "gts" / "real_test"
    gt_root.mkdir(parents=True, exist_ok=True)
    poses = np.repeat(np.eye(4)[None], instance_id, axis=0)
    with (gt_root / f"results_real_test_scene_{scene}_{frame:04d}.pkl").open("wb") as handle:
        pickle.dump({"gt_RTs": poses}, handle)


def test_prepare_builds_one_anchor_for_repeated_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "REAL275"
    object_name = "camera_canon_len_norm"
    _write_frame(
        data_root,
        scene=1,
        frame=20,
        instance_id=4,
        category_id=3,
        object_name=object_name,
    )
    _write_frame(
        data_root,
        scene=4,
        frame=39,
        instance_id=2,
        category_id=3,
        object_name=object_name,
    )
    _write_frame(
        data_root,
        scene=5,
        frame=40,
        instance_id=3,
        category_id=3,
        object_name=object_name,
    )
    pairs = tmp_path / "pairs.txt"
    pairs.write_text(
        "\n".join(
            [
                f"real_test, 1 20, 4 39, 3 {object_name}",
                f"real_test, 1 20, 5 40, 3 {object_name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "anchors"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_oryon_real275.py",
            "--real275-data-root",
            str(data_root),
            "--pairs-file",
            str(pairs),
            "--output-root",
            str(output),
            "--allow-nonofficial-split",
        ],
    )
    prepare_oryon_real275.main()

    payload, manifest_pairs = load_manifest(
        output / "oryon_real275_manifest.json", require_official=False
    )
    assert payload["pair_count"] == 2
    assert payload["unique_reference_count"] == 1
    assert len(manifest_pairs) == 2
    reference_dirs = [path for path in output.iterdir() if path.is_dir()]
    assert len(reference_dirs) == 1
    mask = cv2.imread(str(reference_dirs[0] / "mask.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(mask)) == {0, 255}
    assert int((mask > 0).sum()) == 30
    assert (reference_dirs[0] / "color.png").is_symlink()


def test_query_runner_preserves_2k_denominator_and_defers_gt() -> None:
    source = (ROOT / "run_real275_oryon_query.py").read_text(encoding="utf-8")
    selection = source.index("winner = select_scored_candidate(candidate_results)")
    gt_load = source.index("gt_poses = load_gt_poses(", selection)
    assert selection < gt_load
    assert "Output rows do not cover every official pair exactly once" in source
    assert '"failure_policy": "all requested pairs remain in denominator with zero metrics"' in source
