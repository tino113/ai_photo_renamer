from __future__ import annotations

from pathlib import Path

import xxhash


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = xxhash.xxh64()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
