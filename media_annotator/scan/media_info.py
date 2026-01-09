from __future__ import annotations

from pathlib import Path

from media_annotator.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported media type: {path}")
