from __future__ import annotations

import re

from media_annotator.constants import FORBIDDEN_FILENAME_CHARS


def sanitize_filename(name: str) -> str:
    for ch in FORBIDDEN_FILENAME_CHARS:
        name = name.replace(ch, "")
    name = re.sub(r"\s+", "_", name.strip())
    name = name.strip(". ")
    return name
