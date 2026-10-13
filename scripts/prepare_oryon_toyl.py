#!/usr/bin/env python3
"""Prepare one MGC-6D anchor input per unique Oryon TOYL reference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oryon_toyl_protocol import (
    ORYON_TOYL_MEMBER,
    ORYON_TOYL_PROTOCOL,
    load_instance_list,
    manifest_pair_dict,
    unique_reference_rows,
)


def _scene_dir(toyl_root: Path, scene: int) -> Path:
    return toyl_root / "test" / f"{scene:06d}"


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_annotation(scene_gt: dict, frame: int, object_id: int) -> int:
    entries = scene_gt.get(str(frame), [])
    matches = [
        index
        for index, entry in enumerate(entries)
        if int(entry.get("obj_id", -1)) == object_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one obj_{object_id:02d} annotation in frame {frame}; "
            f"found {matches}"
        )
    return matches[0]


def _replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.resolve())


def _read_instance_mask(scene_dir: Path, frame: int, anno_idx: int) -> np.ndarray:
    candidates = (
        scene_dir / "mask_visib" / f"{frame:06d}_{anno_idx:06d}.png",
        scene_dir / "mask" / f"{frame:06d}_{anno_idx:06d}.png",
    )
    for path in candidates:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            binary = np.where(mask > 0, 255, 0).astype(np.uint8)
            if np.count_nonzero(binary):
                return binary
    raise FileNotFoundError(f"No non-empty instance mask among {candidates}")


def _bop_pose(entry: dict) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    # Keep raw BOP millimetres: query evaluation performs the same conversion
    # as the fixed-anchor TOYL path only after candidate selection.
    pose[:3, 3] = np.asarray(entry["cam_t_m2c"], dtype=np.float64).reshape(3)
    return pose


def _prepare_anchor(
    row: dict[str, object], *, toyl_root: Path, output_root: Path, resume: bool
) -> None:
    key = str(row["anchor_key"])
    scene = int(row["reference_scene"])
    frame = int(row["reference_frame"])
    anno_idx = int(row["reference_anno_idx"])
    obj_id = int(row["object_id"])
    scene_dir = _scene_dir(toyl_root, scene)
    rgb_path = scene_dir / "rgb" / f"{frame:06d}.png"
    depth_path = scene_dir / "depth" / f"{frame:06d}.png"
    gt = _load_json(scene_dir / "scene_gt.json")
    camera = _load_json(scene_dir / "scene_camera.json")
    if not rgb_path.is_file() or not depth_path.is_file():
        raise FileNotFoundError(f"Missing RGB/depth for {key}")
    entries = gt.get(str(frame), [])
    if anno_idx >= len(entries) or int(entries[anno_idx].get("obj_id", -1)) != obj_id:
        raise ValueError(f"Manifest annotation no longer matches BOP GT for {key}")
    cam_entry = camera.get(str(frame))
    if not isinstance(cam_entry, dict) or "cam_K" not in cam_entry:
        raise ValueError(f"Missing camera intrinsics for {key}")

    target = output_root / key
    if target.exists() and not resume:
        raise FileExistsError(
            f"Anchor input already exists: {target}; pass --resume to validate/update"
        )
    target.mkdir(parents=True, exist_ok=True)
    _replace_symlink(target / "color.png", rgb_path)
    _replace_symlink(target / "depth.png", depth_path)
    mask = _read_instance_mask(scene_dir, frame, anno_idx)
    if not cv2.imwrite(str(target / "mask.png"), mask):
        raise OSError(f"Failed to write {target / 'mask.png'}")
    np.savetxt(
        target / "K.txt",
        np.asarray(cam_entry["cam_K"], dtype=np.float64).reshape(3, 3),
    )
    np.savetxt(target / f"{key}_gt_pose.txt", _bop_pose(entries[anno_idx]))
    source = "\n".join(
        [
            f"scene={scene:06d}",
            f"frame={frame:06d}",
            f"annotation_index={anno_idx}",
            f"object_id={obj_id}",
            f"mask_pixels={int(np.count_nonzero(mask))}",
            f"protocol={ORYON_TOYL_PROTOCOL}",
        ]
    )
    (target / "source_frame.txt").write_text(source + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyl-root", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonofficial-split", action="store_true")
    args = parser.parse_args()

    root = args.toyl_root.resolve()
    if not (root / "test").is_dir():
        raise FileNotFoundError(f"TOYL root must contain test/: {root}")
    pairs, digest = load_instance_list(
        args.pairs_file, require_official=not args.allow_nonofficial_split
    )
    scene_cache: dict[int, dict] = {}
    rows: list[dict[str, object]] = []
    for pair in pairs:
        if pair.object_id not in scene_cache:
            scene_cache[pair.object_id] = _load_json(
                _scene_dir(root, pair.object_id) / "scene_gt.json"
            )
        scene_gt = scene_cache[pair.object_id]
        rows.append(
            manifest_pair_dict(
                pair,
                reference_anno_idx=_resolve_annotation(
                    scene_gt, pair.reference_frame, pair.object_id
                ),
                query_anno_idx=_resolve_annotation(
                    scene_gt, pair.query_frame, pair.object_id
                ),
            )
        )
    unique_rows = unique_reference_rows(rows)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(unique_rows, start=1):
        _prepare_anchor(row, toyl_root=root, output_root=output, resume=args.resume)
        if index % 100 == 0 or index == len(unique_rows):
            print(f"Prepared {index}/{len(unique_rows)} unique references")

    manifest = {
        "dataset": "TOYL",
        "protocol": ORYON_TOYL_PROTOCOL,
        "official_archive_member": ORYON_TOYL_MEMBER,
        "pair_file": str(args.pairs_file.resolve()),
        "pair_file_sha256": digest,
        "pair_count": len(rows),
        "unique_reference_count": len(unique_rows),
        "mask_sources": {"reference": "dataset_gt", "query": "dataset_gt"},
        "reference_cache_key": "scene,frame,object_id",
        "pairs": rows,
    }
    destination = output / "oryon_toyl_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(
        f"Prepared {len(unique_rows)} unique references for {len(rows)} pairs; "
        f"manifest: {destination}"
    )


if __name__ == "__main__":
    main()
