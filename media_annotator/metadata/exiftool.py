from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from media_annotator.utils.subprocess import run_command


def extract_exif(path: Path) -> Dict[str, Any]:
    result = run_command(["exiftool", "-j", "-n", str(path)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    data = json.loads(result.stdout)
    return data[0] if data else {}
