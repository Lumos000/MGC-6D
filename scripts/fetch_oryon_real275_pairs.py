#!/usr/bin/env python3
"""Extract the official REAL275 2K pair list without downloading 804 MB."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
import sys
from urllib.request import Request, urlopen
import zlib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oryon_real275_protocol import (
    ORYON_ARCHIVE_URL,
    ORYON_REAL275_MEMBER,
    ORYON_REAL275_PAIR_COUNT,
    ORYON_REAL275_SHA256,
    parse_instance_list_text,
    sha256_bytes,
)


def _http_size(url: str) -> int:
    request = Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
    with urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
        if value is None:
            raise RuntimeError("Oryon archive server did not provide Content-Length")
        return int(value)


def _http_range(url: str, start: int, end: int) -> bytes:
    request = Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        },
    )
    with urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(
                "Oryon archive server ignored the byte-range request; refusing "
                "to download the entire archive"
            )
        data = response.read(end - start + 1)
    if len(data) != end - start + 1:
        raise RuntimeError("Short byte-range response from Oryon archive")
    return data


def _central_directory_from_remote(url: str) -> bytes:
    size = _http_size(url)
    tail_start = max(0, size - (65535 + 22))
    tail = _http_range(url, tail_start, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("ZIP end-of-central-directory record not found")
    _, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, _ = struct.unpack_from(
        "<4s4H2LH", tail, eocd
    )
    if disk != 0 or cd_disk != 0 or disk_entries != entries:
        raise ValueError("Multi-disk ZIP archives are unsupported")
    return _http_range(url, cd_offset, cd_offset + cd_size - 1)


def _find_member(central: bytes, member: str) -> tuple[int, int, int, int]:
    offset = 0
    while offset + 46 <= len(central):
        fields = struct.unpack_from("<4s6H3L5H2L", central, offset)
        if fields[0] != b"PK\x01\x02":
            break
        method = fields[4]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        name_length, extra_length, comment_length = fields[10:13]
        local_offset = fields[-1]
        name = central[
            offset + 46 : offset + 46 + name_length
        ].decode("utf-8")
        if name == member:
            return local_offset, method, compressed_size, uncompressed_size
        offset += 46 + name_length + extra_length + comment_length
    raise KeyError(f"ZIP member not found: {member}")


def extract_remote_member(url: str, member: str) -> bytes:
    central = _central_directory_from_remote(url)
    local_offset, expected_method, compressed_size, uncompressed_size = _find_member(
        central, member
    )
    header = _http_range(url, local_offset, local_offset + 29)
    fields = struct.unpack_from("<4s5H3L2H", header, 0)
    if fields[0] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local-file header")
    method, name_length, extra_length = fields[3], fields[9], fields[10]
    if method != expected_method:
        raise ValueError("ZIP method differs between directory and local header")
    data_start = local_offset + 30 + name_length + extra_length
    compressed = _http_range(
        url, data_start, data_start + compressed_size - 1
    )
    if method == 0:
        data = compressed
    elif method == 8:
        data = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"Unsupported ZIP compression method: {method}")
    if len(data) != uncompressed_size:
        raise ValueError("Uncompressed ZIP member has an unexpected size")
    return data


def extract_local_member(archive: Path, member: str) -> bytes:
    from zipfile import ZipFile

    with ZipFile(archive) as handle:
        return handle.read(member)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--url", default=ORYON_ARCHIVE_URL)
    args = parser.parse_args()

    data = (
        extract_local_member(args.archive, ORYON_REAL275_MEMBER)
        if args.archive is not None
        else extract_remote_member(args.url, ORYON_REAL275_MEMBER)
    )
    digest = sha256_bytes(data)
    if digest != ORYON_REAL275_SHA256:
        raise ValueError(f"Official split SHA-256 mismatch: {digest}")
    pairs = parse_instance_list_text(data.decode("utf-8"))
    if len(pairs) != ORYON_REAL275_PAIR_COUNT:
        raise ValueError(f"Expected 2000 pairs, found {len(pairs)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(args.output)
    print(
        f"Saved {len(pairs)} official Oryon REAL275 pairs to {args.output} "
        f"(sha256={digest})"
    )


if __name__ == "__main__":
    main()
