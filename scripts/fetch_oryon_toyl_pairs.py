#!/usr/bin/env python3
"""Extract the official Oryon TOYL 2K pair list with ZIP range requests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oryon_toyl_protocol import (
    ORYON_ARCHIVE_URL,
    ORYON_TOYL_MEMBER,
    ORYON_TOYL_PAIR_COUNT,
    ORYON_TOYL_SHA256,
    parse_instance_list_text,
    sha256_bytes,
)
from scripts.fetch_oryon_real275_pairs import extract_local_member, extract_remote_member


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--url", default=ORYON_ARCHIVE_URL)
    args = parser.parse_args()

    data = (
        extract_local_member(args.archive, ORYON_TOYL_MEMBER)
        if args.archive is not None
        else extract_remote_member(args.url, ORYON_TOYL_MEMBER)
    )
    digest = sha256_bytes(data)
    if digest != ORYON_TOYL_SHA256:
        raise ValueError(f"Official split SHA-256 mismatch: {digest}")
    pairs = parse_instance_list_text(data.decode("utf-8"))
    if len(pairs) != ORYON_TOYL_PAIR_COUNT:
        raise ValueError(f"Expected 2000 pairs, found {len(pairs)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(args.output)
    print(
        f"Saved {len(pairs)} official Oryon TOYL pairs to {args.output} "
        f"(sha256={digest})"
    )


if __name__ == "__main__":
    main()
