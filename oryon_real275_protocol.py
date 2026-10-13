"""Pure helpers for the official Oryon REAL275 2K view-pair protocol."""

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
ORYON_REAL275_MEMBER = (
    "datasets/nocs/fixed_split/cross_scene_test/instance_list.txt"
)
ORYON_REAL275_SHA256 = (
    "1553e4ecfe1a3a3be1d55e6c5da157e7aa199dd80246d7b0501da4fefdbf7424"
)
ORYON_REAL275_PAIR_COUNT = 2000


@dataclass(frozen=True)
class OryonPair:
    pair_id: int
    split: str
    reference_scene: int
    reference_frame: int
    query_scene: int
    query_frame: int
    category_id: int
    object_name: str

    @property
    def reference_scene_name(self) -> str:
        return f"scene_{self.reference_scene}"

    @property
    def query_scene_name(self) -> str:
        return f"scene_{self.query_scene}"

    @property
    def reference_identity(self) -> tuple[int, int, str]:
        return self.reference_scene, self.reference_frame, self.object_name


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_instance_list_text(text: str) -> list[OryonPair]:
    pairs: list[OryonPair] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(
                f"Line {line_number}: expected four comma-separated fields"
            )
        split = fields[0]
        try:
            reference_scene, reference_frame = map(int, fields[1].split())
            query_scene, query_frame = map(int, fields[2].split())
            category_token, object_name = fields[3].split(maxsplit=1)
            category_id = int(category_token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: malformed Oryon pair: {line}") from exc
        if split != "real_test":
            raise ValueError(f"Line {line_number}: unexpected split {split!r}")
        if reference_scene == query_scene:
            raise ValueError(
                f"Line {line_number}: Oryon cross-scene pair uses the same scene"
            )
        if min(reference_scene, query_scene) < 1:
            raise ValueError(f"Line {line_number}: scene ids must be positive")
        if min(reference_frame, query_frame) < 0:
            raise ValueError(f"Line {line_number}: frame ids must be non-negative")
        pairs.append(
            OryonPair(
                pair_id=len(pairs),
                split=split,
                reference_scene=reference_scene,
                reference_frame=reference_frame,
                query_scene=query_scene,
                query_frame=query_frame,
                category_id=category_id,
                object_name=object_name,
            )
        )
    if not pairs:
        raise ValueError("Oryon pair list is empty")
    return pairs


def load_instance_list(
    path: str | Path,
    *,
    require_official: bool = True,
) -> tuple[list[OryonPair], str]:
    pair_path = Path(path)
    data = pair_path.read_bytes()
    digest = sha256_bytes(data)
    pairs = parse_instance_list_text(data.decode("utf-8"))
    if require_official:
        if digest != ORYON_REAL275_SHA256:
            raise ValueError(
                "The pair file is not the official Oryon REAL275 split: "
                f"sha256={digest}, expected={ORYON_REAL275_SHA256}"
            )
        if len(pairs) != ORYON_REAL275_PAIR_COUNT:
            raise ValueError(
                f"Official split must contain {ORYON_REAL275_PAIR_COUNT} pairs, "
                f"found {len(pairs)}"
            )
    return pairs, digest


def anchor_key(
    scene: int,
    frame: int,
    instance_id: int,
    object_name: str,
) -> str:
    return f"ref_s{scene:02d}_f{frame:04d}_i{instance_id:02d}_{object_name}"


def parse_nocs_meta(path: str | Path) -> list[tuple[int, int, str]]:
    entries: list[tuple[int, int, str]] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) < 3:
            raise ValueError(f"{path}:{line_number}: malformed REAL275 meta row")
        entries.append((int(fields[0]), int(fields[1]), fields[2]))
    return entries


def resolve_instance(
    real_test_root: str | Path,
    *,
    scene: int,
    frame: int,
    object_name: str,
    category_id: int,
) -> int:
    meta_path = (
        Path(real_test_root)
        / f"scene_{scene}"
        / f"{frame:04d}_meta.txt"
    )
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    matches = [
        instance_id
        for instance_id, class_id, name in parse_nocs_meta(meta_path)
        if name == object_name and class_id == category_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {object_name} (class {category_id}) in "
            f"scene_{scene}/{frame:04d}; found {matches}"
        )
    return matches[0]


def manifest_pair_dict(
    pair: OryonPair,
    *,
    reference_instance_id: int,
    query_instance_id: int,
) -> dict[str, object]:
    key = anchor_key(
        pair.reference_scene,
        pair.reference_frame,
        reference_instance_id,
        pair.object_name,
    )
    payload = asdict(pair)
    payload.update(
        {
            "anchor_key": key,
            "reference_instance_id": int(reference_instance_id),
            "query_instance_id": int(query_instance_id),
        }
    )
    return payload


def validate_manifest(
    payload: dict[str, object],
    *,
    require_official: bool = True,
) -> list[dict[str, object]]:
    if payload.get("dataset") != "REAL275":
        raise ValueError("Pair manifest dataset must be REAL275")
    if payload.get("protocol") != "oryon_cross_scene_2k":
        raise ValueError("Pair manifest protocol must be oryon_cross_scene_2k")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Pair manifest has no pairs")
    if require_official:
        if payload.get("pair_file_sha256") != ORYON_REAL275_SHA256:
            raise ValueError("Pair manifest does not identify the official Oryon split")
        if len(pairs) != ORYON_REAL275_PAIR_COUNT:
            raise ValueError(
                f"Pair manifest has {len(pairs)} rather than 2000 pairs"
            )
    expected_ids = list(range(len(pairs)))
    actual_ids = [int(row.get("pair_id", -1)) for row in pairs if isinstance(row, dict)]
    if actual_ids != expected_ids:
        raise ValueError("Pair manifest pair_id values are not contiguous and ordered")
    for row in pairs:
        if not isinstance(row, dict):
            raise ValueError("Pair manifest rows must be JSON objects")
        for key in (
            "anchor_key",
            "object_name",
            "reference_scene",
            "reference_frame",
            "reference_instance_id",
            "query_scene",
            "query_frame",
            "query_instance_id",
        ):
            if key not in row:
                raise ValueError(f"Pair manifest row is missing {key}")
        if int(row["reference_scene"]) == int(row["query_scene"]):
            raise ValueError("Pair manifest contains a non-cross-scene pair")
    unique_keys = {str(row["anchor_key"]) for row in pairs}
    recorded_unique = payload.get("unique_reference_count")
    if recorded_unique is not None and int(recorded_unique) != len(unique_keys):
        raise ValueError("Pair manifest unique_reference_count is inconsistent")
    return pairs


def load_manifest(
    path: str | Path,
    *,
    require_official: bool = True,
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
        identity_fields = (
            "reference_scene",
            "reference_frame",
            "reference_instance_id",
            "object_name",
        )
        if any(previous[field] != row[field] for field in identity_fields):
            raise ValueError(f"Anchor key collision for {key}")
    return [unique[key] for key in sorted(unique)]
