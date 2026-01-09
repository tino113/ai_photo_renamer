from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def write_atomic(path: Path, content: str) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
    logger.debug("Wrote sidecar {}", path)


def write_text_sidecar(media_path: Path, text: str) -> Path:
    sidecar = media_path.with_suffix(".txt")
    write_atomic(sidecar, text)
    return sidecar


def write_json_sidecar(media_path: Path, data: dict[str, Any]) -> Path:
    sidecar = media_path.with_suffix(".json")
    write_atomic(sidecar, json.dumps(data, ensure_ascii=False, indent=2))
    return sidecar
