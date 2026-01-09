from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from media_annotator.config import AppConfig
from media_annotator.utils.paths import ensure_unique_path
from media_annotator.utils.slugify import sanitize_filename


def build_plan_entry(
    media_path: Path,
    suggested_base: str,
    capture_date: str | None,
    people: list[str],
    max_length: int,
    output_root: Path | None,
    media_hash: str,
    input_root: Path | None,
    mirror_structure: bool,
) -> dict[str, Any]:
    base = sanitize_filename(suggested_base)
    if capture_date:
        date_prefix = capture_date.split("T")[0]
        base = f"{date_prefix}_{base}"
    if people:
        base = f"{base}_{'_'.join(people[:3])}"
    if len(base) > max_length:
        base = base[:max_length]
    new_name = f"{base}{media_path.suffix}"
    if output_root and mirror_structure and input_root:
        target_dir = output_root / media_path.parent.relative_to(input_root)
    elif output_root:
        target_dir = output_root
    else:
        target_dir = media_path.parent
    target = ensure_unique_path(target_dir / new_name)
    return {
        "media_hash": media_hash,
        "old_path": str(media_path),
        "new_path": str(target),
        "sidecars_old": [str(media_path.with_suffix(".txt")), str(media_path.with_suffix(".json"))],
        "sidecars_new": [str(target.with_suffix(".txt")), str(target.with_suffix(".json"))],
        "conflicts_resolved": target != (target_dir / new_name),
        "resolution_strategy": "suffix" if target != (target_dir / new_name) else "none",
    }


def generate_plan(
    config: AppConfig,
    items: list[dict[str, Any]],
    output_root: Path | None,
    input_root: Path | None = None,
) -> dict[str, Any]:
    operations = []
    for item in items:
        meta = json.loads(item["meta_json"]) if item.get("meta_json") else {}
        suggested_base = meta.get("suggested_filename_base", Path(item["path"]).stem)
        capture_datetime = meta.get("capture_datetime")
        people = [p["name"] for p in meta.get("detected_persons", []) if not p["name"].startswith("unknown")]
        operations.append(
            build_plan_entry(
                Path(item["path"]),
                suggested_base,
                capture_datetime,
                people,
                config.pipeline.max_filename_length,
                output_root,
                item.get("hash", ""),
                input_root,
                config.pipeline.copy_mirror_structure,
            )
        )
    plan = {
        "created_at": datetime.utcnow().isoformat(),
        "tool_version": config.pipeline.pipeline_version,
        "mode": "copy" if config.pipeline.copy_mode else "rename",
        "input_root": str(input_root) if input_root else None,
        "output_root": str(output_root) if output_root else None,
        "operations": operations,
    }
    logger.info("Generated plan with {} operations", len(operations))
    return plan
