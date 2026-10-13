"""Pure helpers for the official Oryon TOYL 2,000 view-pair protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


ORYON_ARCHIVE_URL = (
    "https://github.com/jcorsetti/oryon/releases/download/"
    "v1.0.0-data/oryon_data.zip"
)
ORYON_TOYL_MEMBER = "datasets/toyl/fixed_split/cross_scene_test/instance_list.txt"
ORYON_TOYL_SHA256 = (
    "99eb3a67edb126bb6b3787482b3092acb684debac828c7ee476c7e24e56a5dea"
)
ORYON_TOYL_PAIR_COUNT = 2000
ORYON_TOYL_UNIQUE_REFERENCE_COUNT = 890
ORYON_TOYL_PROTOCOL = "oryon_toyl_2k"


@dataclass(frozen=True)
class OryonToylPair:
    pair_id: int
    split: str
    reference_scene: int
    reference_frame: int
    query_scene: int
    query_frame: int
    object_id: int

    @property
    def reference_identity(self) -> tuple[int, int, int]:
        return self.reference_scene, self.reference_frame, self.object_id


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_instance_list_text(text: str) -> list[OryonToylPair]:
    """Parse Oryon's TOYL list and enforce its BOP scene/object contract.

    Oryon stores this file below a directory named ``cross_scene_test``, but
    the released TOYL rows use two different frames in the same object scene.
    We preserve that fact instead of incorrectly describing them as
    cross-scene pairs.
    """

    pairs: list[OryonToylPair] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(
                f"Line {line_number}: expected four comma-separated fields"
            )
        try:
            split = fields[0]
            reference_scene, reference_frame = map(int, fields[1].split())
            query_scene, query_frame = map(int, fields[2].split())
            object_id = int(fields[3])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: malformed Oryon pair: {line}") from exc
        if split != "test":
            raise ValueError(f"Line {line_number}: unexpected split {split!r}")
        if not 1 <= object_id <= 21:
            raise ValueError(f"Line {line_number}: TOYL object id is outside [1, 21]")
        if reference_scene != object_id or query_scene != object_id:
            raise ValueError(
                f"Line {line_number}: TOYL scene ids must equal object id "
                f"({reference_scene}, {query_scene}, {object_id})"
            )
        if min(reference_frame, query_frame) < 0:
            raise ValueError(f"Line {line_number}: frame ids must be non-negative")
        if reference_frame == query_frame:
            raise ValueError(f"Line {line_number}: reference and query frames are equal")
        pairs.append(
            OryonToylPair(
                pair_id=len(pairs),
                split=split,
                reference_scene=reference_scene,
                reference_frame=reference_frame,
                query_scene=query_scene,
                query_frame=query_frame,
                object_id=object_id,
            )
        )
    if not pairs:
        raise ValueError("Oryon TOYL pair list is empty")
    return pairs


def load_instance_list(
    path: str | Path, *, require_official: bool = True
) -> tuple[list[OryonToylPair], str]:
    data = Path(path).read_bytes()
    digest = sha256_bytes(data)
    pairs = parse_instance_list_text(data.decode("utf-8"))
    if require_official:
        if digest != ORYON_TOYL_SHA256:
            raise ValueError(
                "The pair file is not the official Oryon TOYL split: "
                f"sha256={digest}, expected={ORYON_TOYL_SHA256}"
            )
        if len(pairs) != ORYON_TOYL_PAIR_COUNT:
            raise ValueError(
                f"Official split must contain {ORYON_TOYL_PAIR_COUNT} pairs, "
                f"found {len(pairs)}"
            )
        if len({pair.reference_identity for pair in pairs}) != ORYON_TOYL_UNIQUE_REFERENCE_COUNT:
            raise ValueError("Official split does not contain the expected 890 references")
    return pairs, digest


def anchor_key(scene: int, frame: int, object_id: int) -> str:
    # The suffix is intentional: anchor_paper_toyl_oryon.py extracts obj_id
    # without changing the fixed-anchor implementation.
    return f"ref_s{scene:06d}_f{frame:06d}_obj_{object_id:02d}"


def manifest_pair_dict(
    pair: OryonToylPair,
    *,
    reference_anno_idx: int,
    query_anno_idx: int,
) -> dict[str, object]:
    payload = asdict(pair)
    payload.update(
        {
            "anchor_key": anchor_key(
                pair.reference_scene, pair.reference_frame, pair.object_id
            ),
            "reference_anno_idx": int(reference_anno_idx),
            "query_anno_idx": int(query_anno_idx),
            "object_name": f"obj_{pair.object_id:02d}",
        }
    )
    return payload


def validate_manifest(
    payload: dict[str, object], *, require_official: bool = True
) -> list[dict[str, object]]:
    if payload.get("dataset") != "TOYL":
        raise ValueError("Pair manifest dataset must be TOYL")
    if payload.get("protocol") != ORYON_TOYL_PROTOCOL:
        raise ValueError(f"Pair manifest protocol must be {ORYON_TOYL_PROTOCOL}")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Pair manifest has no pairs")
    if require_official:
        if payload.get("pair_file_sha256") != ORYON_TOYL_SHA256:
            raise ValueError("Pair manifest does not identify the official Oryon split")
        if len(pairs) != ORYON_TOYL_PAIR_COUNT:
            raise ValueError("Pair manifest does not contain 2,000 official pairs")
    actual_ids = [int(row.get("pair_id", -1)) for row in pairs if isinstance(row, dict)]
    if actual_ids != list(range(len(pairs))):
        raise ValueError("Pair manifest pair_id values are not contiguous and ordered")
    required = (
        "anchor_key",
        "object_id",
        "object_name",
        "reference_scene",
        "reference_frame",
        "reference_anno_idx",
        "query_scene",
        "query_frame",
        "query_anno_idx",
    )
    for row in pairs:
        if not isinstance(row, dict):
            raise ValueError("Pair manifest rows must be JSON objects")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Pair manifest row is missing {missing}")
        obj_id = int(row["object_id"])
        if int(row["reference_scene"]) != obj_id or int(row["query_scene"]) != obj_id:
            raise ValueError("TOYL manifest scene/object identity is inconsistent")
        expected_key = anchor_key(
            int(row["reference_scene"]), int(row["reference_frame"]), obj_id
        )
        if str(row["anchor_key"]) != expected_key:
            raise ValueError("TOYL manifest anchor key is inconsistent")
    unique_keys = {str(row["anchor_key"]) for row in pairs}
    if int(payload.get("unique_reference_count", len(unique_keys))) != len(unique_keys):
        raise ValueError("Pair manifest unique_reference_count is inconsistent")
    if require_official and len(unique_keys) != ORYON_TOYL_UNIQUE_REFERENCE_COUNT:
        raise ValueError("Official manifest must contain 890 unique references")
    return pairs


def load_manifest(
    path: str | Path, *, require_official: bool = True
) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pair manifest root must be a JSON object")
    return payload, validate_manifest(payload, require_official=require_official)


def unique_reference_rows(
    pairs: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    unique: dict[str, dict[str, object]] = {}
    for row in pairs:
        key = str(row["anchor_key"])
        previous = unique.setdefault(key, row)
        for field in (
            "reference_scene",
            "reference_frame",
            "reference_anno_idx",
            "object_id",
        ):
            if previous[field] != row[field]:
                raise ValueError(f"Anchor key collision for {key}")
    return [unique[key] for key in sorted(unique)]
