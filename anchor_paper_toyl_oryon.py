"""TOYL Oryon-anchor entry point without altering the fixed-anchor mainline."""

from __future__ import annotations

import re

import anchor_paper_toyl as implementation


def parse_oryon_obj_id(anchor_name: str) -> int | None:
    match = re.search(r"_obj_(\d+)$", anchor_name)
    return int(match.group(1)) if match else None


def main() -> None:
    implementation._parse_obj_id = parse_oryon_obj_id
    implementation.main()


if __name__ == "__main__":
    main()
