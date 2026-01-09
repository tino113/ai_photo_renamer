from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from media_annotator.utils.subprocess import run_command


def extract_ffprobe(path: Path) -> Dict[str, Any]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)
