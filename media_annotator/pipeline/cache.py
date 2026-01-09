from __future__ import annotations

from media_annotator.db.models import MediaItem


def should_process(item: MediaItem, pipeline_version: str, force: bool, required_status: str) -> bool:
    if force:
        return True
    if item.pipeline_version != pipeline_version:
        return True
    status_order = ["discovered", "faces_done", "llm_done", "renamed"]
    try:
        current_index = status_order.index(item.status)
        required_index = status_order.index(required_status)
    except ValueError:
        return True
    return current_index < required_index
