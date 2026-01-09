from __future__ import annotations

from pathlib import Path
from typing import Iterable

from media_annotator.utils.paths import is_supported_media


def discover_media(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if is_supported_media(path):
            yield path
