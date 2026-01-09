from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from media_annotator.db import dao
from media_annotator.utils.paths import ensure_unique_path


def apply_plan(
    session,
    plan: dict[str, Any],
    mode: str,
    output_dir: Path | None,
    dry_run: bool,
    undo_file: Path | None = None,
) -> None:
    operations = plan["operations"]
    temp_map = {}
    if mode == "rename" and not dry_run:
        for op in operations:
            old_path = Path(op["old_path"])
            new_path = Path(op["new_path"])
            if old_path == new_path:
                continue
            if new_path.exists():
                temp_path = old_path.with_suffix(old_path.suffix + ".tmp_rename")
                counter = 1
                while temp_path.exists():
                    temp_path = old_path.with_suffix(f"{old_path.suffix}.tmp_{counter}")
                    counter += 1
                old_path.rename(temp_path)
                for sidecar_suffix in [".txt", ".json"]:
                    sidecar = old_path.with_suffix(sidecar_suffix)
                    if sidecar.exists():
                        sidecar.rename(temp_path.with_suffix(sidecar_suffix))
                temp_map[str(old_path)] = temp_path

    for op in operations:
        old_path = Path(op["old_path"])
        if str(old_path) in temp_map:
            old_path = temp_map[str(old_path)]
        new_path = Path(op["new_path"])
        if mode == "copy":
            if output_dir:
                target = ensure_unique_path(output_dir / new_path.name)
            else:
                target = ensure_unique_path(new_path)
        else:
            target = new_path
        sidecars_old = [old_path.with_suffix(".txt"), old_path.with_suffix(".json")]
        sidecars_old = [p for p in sidecars_old if p.exists()]
        sidecars_new = [target.with_suffix(".txt"), target.with_suffix(".json")]
        if dry_run:
            logger.info("Dry run: {} -> {}", old_path, target)
            continue
        if mode == "copy":
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, target)
            for old_sidecar, new_sidecar in zip(sidecars_old, sidecars_new):
                shutil.copy2(old_sidecar, new_sidecar)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(target)
            for old_sidecar, new_sidecar in zip(sidecars_old, sidecars_new):
                if old_sidecar.exists():
                    old_sidecar.rename(new_sidecar)
        dao.record_rename_history(
            session,
            media_hash=op.get("media_hash", ""),
            old_path=str(old_path),
            new_path=str(target),
            sidecars_old=[str(p) for p in sidecars_old],
            sidecars_new=[str(p) for p in sidecars_new],
            mode=mode,
        )
        session.commit()
        logger.info("Applied change {} -> {}", old_path, target)
    undo = {
        "created_at": plan.get("created_at"),
        "operations": [
            {
                "old_path": op["new_path"],
                "new_path": op["old_path"],
                "sidecars_old": op.get("sidecars_new", []),
                "sidecars_new": op.get("sidecars_old", []),
            }
            for op in plan["operations"]
        ],
    }
    if undo_file:
        with undo_file.open("w", encoding="utf-8") as handle:
            json.dump(undo, handle, indent=2)
