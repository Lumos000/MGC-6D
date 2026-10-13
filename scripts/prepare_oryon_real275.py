#!/usr/bin/env python3
"""Prepare one MGC-6D anchor input per unique Oryon REAL275 reference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oryon_real275_protocol import (
    ORYON_REAL275_MEMBER,
    load_instance_list,
    manifest_pair_dict,
    resolve_instance,
    unique_reference_rows,
)


REAL275_K = np.array(
    [
        [591.0125, 0.0, 322.525],
        [0.0, 590.16775, 244.11084],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _frame_paths(real_test_root: Path, scene: int, frame: int) -> dict[str, Path]:
    root = real_test_root / f"scene_{scene}"
    return {
        "color": root / f"{frame:04d}_color.png",
        "depth": root / f"{frame:04d}_depth.png",
        "mask": root / f"{frame:04d}_mask.png",
        "meta": root / f"{frame:04d}_meta.txt",
    }


def _replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.resolve())


def _load_gt_pose(gt_root: Path, scene: int, frame: int, instance_id: int) -> np.ndarray:
    path = gt_root / f"results_real_test_scene_{scene}_{frame:04d}.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    poses = np.asarray(payload["gt_RTs"])
    index = instance_id - 1
    if index < 0 or index >= len(poses):
        raise IndexError(f"Instance {instance_id} is absent from {path}")
    pose = np.asarray(poses[index], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid GT pose in {path} for instance {instance_id}")
    return pose


def _prepare_anchor(
    row: dict[str, object],
    *,
    real_test_root: Path,
    gt_root: Path,
    output_root: Path,
    resume: bool,
) -> None:
    key = str(row["anchor_key"])
    scene = int(row["reference_scene"])
    frame = int(row["reference_frame"])
    instance_id = int(row["reference_instance_id"])
    object_name = str(row["object_name"])
    category_id = int(row["category_id"])
    paths = _frame_paths(real_test_root, scene, frame)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    target = output_root / key
    if target.exists() and not resume:
        raise FileExistsError(
            f"Anchor input already exists: {target}; pass --resume to validate/update"
        )
    target.mkdir(parents=True, exist_ok=True)
    _replace_symlink(target / "color.png", paths["color"])
    _replace_symlink(target / "depth.png", paths["depth"])

    full_mask = cv2.imread(str(paths["mask"]), cv2.IMREAD_UNCHANGED)
    if full_mask is None:
        raise ValueError(f"Cannot read {paths['mask']}")
    binary_mask = np.where(full_mask == instance_id, 255, 0).astype(np.uint8)
    mask_pixels = int(np.count_nonzero(binary_mask))
    if mask_pixels == 0:
        raise ValueError(f"Empty anchor mask for {key}")
    if not cv2.imwrite(str(target / "mask.png"), binary_mask):
        raise OSError(f"Failed to write {target / 'mask.png'}")

    np.savetxt(target / "K.txt", REAL275_K)
    pose_path = target / f"{key}_gt_pose.txt"
    np.savetxt(
        pose_path,
        _load_gt_pose(gt_root, scene, frame, instance_id),
    )
    source = "\n".join(
        [
            f"scene=scene_{scene}",
            f"frame={frame:04d}",
            f"instance_id={instance_id}",
            f"class_id={category_id}",
            f"object_name={object_name}",
            f"mask_pixels={mask_pixels}",
            "protocol=oryon_cross_scene_2k",
        ]
    )
    (target / "source_frame.txt").write_text(source + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real275-data-root", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-nonofficial-split",
        action="store_true",
        help="Testing only: disable the official SHA-256 and 2,000-pair gate.",
    )
    args = parser.parse_args()

    real_root = args.real275_data_root.resolve()
    real_test_root = real_root / "real_test"
    gt_root = real_root / "gts" / "real_test"
    if not real_test_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError(
            "REAL275 root must contain real_test/ and gts/real_test/"
        )
    pairs, digest = load_instance_list(
        args.pairs_file,
        require_official=not args.allow_nonofficial_split,
    )

    manifest_rows: list[dict[str, object]] = []
    for pair in pairs:
        reference_instance = resolve_instance(
            real_test_root,
            scene=pair.reference_scene,
            frame=pair.reference_frame,
            object_name=pair.object_name,
            category_id=pair.category_id,
        )
        query_instance = resolve_instance(
            real_test_root,
            scene=pair.query_scene,
            frame=pair.query_frame,
            object_name=pair.object_name,
            category_id=pair.category_id,
        )
        manifest_rows.append(
            manifest_pair_dict(
                pair,
                reference_instance_id=reference_instance,
                query_instance_id=query_instance,
            )
        )

    unique_rows = unique_reference_rows(manifest_rows)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(unique_rows, start=1):
        _prepare_anchor(
            row,
            real_test_root=real_test_root,
            gt_root=gt_root,
            output_root=output_root,
            resume=args.resume,
        )
        if index % 100 == 0 or index == len(unique_rows):
            print(f"Prepared {index}/{len(unique_rows)} unique references")

    manifest = {
        "dataset": "REAL275",
        "protocol": "oryon_cross_scene_2k",
        "official_archive_member": ORYON_REAL275_MEMBER,
        "pair_file": str(args.pairs_file.resolve()),
        "pair_file_sha256": digest,
        "pair_count": len(manifest_rows),
        "unique_reference_count": len(unique_rows),
        "mask_sources": {"reference": "dataset_gt", "query": "dataset_gt"},
        "reference_cache_key": "scene,frame,instance_id,object_name",
        "pairs": manifest_rows,
    }
    manifest_path = output_root / "oryon_real275_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(
        f"Prepared {len(unique_rows)} unique references for {len(manifest_rows)} pairs; "
        f"manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
