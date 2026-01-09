from __future__ import annotations

from pathlib import Path

from media_annotator.constants import SUPPORTED_EXTENSIONS


def is_supported_media(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter:03d}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
