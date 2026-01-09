from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger
from sqlalchemy import select

from media_annotator.config import AppConfig
from media_annotator.db import dao
from media_annotator.db.models import MediaFace, MediaItem, Person
from media_annotator.llm.base import LLMBackend
from media_annotator.metadata.exiftool import extract_exif
from media_annotator.faces.video_sampling import build_sampling_plan
from media_annotator.metadata.ffprobe import extract_ffprobe
from media_annotator.metadata.location import format_location
from media_annotator.sidecar.writer import write_json_sidecar, write_text_sidecar
from media_annotator.utils.subprocess import run_command
from media_annotator.utils.time import parse_datetime


def _select_backend(config: AppConfig) -> LLMBackend:
    if config.llm.backend == "ollama":
        from media_annotator.llm.ollama_backend import OllamaBackend

        return OllamaBackend(config.llm.model, config.llm.base_url, config.llm.timeout_s)
    if config.llm.backend == "lmstudio":
        if not config.llm.base_url:
            raise ValueError("LM Studio backend requires base_url")
        from media_annotator.llm.lmstudio_backend import LMStudioBackend

        return LMStudioBackend(config.llm.model, config.llm.base_url, config.llm.timeout_s)
    if config.llm.backend == "local":
        from media_annotator.llm.local_safetensors_backend import LocalSafetensorsBackend

        return LocalSafetensorsBackend(config.llm.model)
    raise ValueError(f"Unsupported LLM backend: {config.llm.backend}")


def _capture_datetime_from_exif(exif: dict) -> Optional[str]:
    for key in ["DateTimeOriginal", "CreateDate", "FileModifyDate"]:
        value = exif.get(key)
        parsed = parse_datetime(value)
        if parsed:
            return parsed.isoformat()
    return None


def _capture_datetime_from_ffprobe(meta: dict) -> Optional[str]:
    tags = meta.get("format", {}).get("tags", {})
    for key in ["creation_time", "com.apple.quicktime.creationdate"]:
        parsed = parse_datetime(tags.get(key))
        if parsed:
            return parsed.isoformat()
    return None


def _frame_paths_for_video(path: Path, cache_dir: Path, sample_plan: list[int]) -> list[Path]:
    output_dir = cache_dir / f"llm_{path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for idx, time_ms in enumerate(sample_plan):
        out_path = output_dir / f"frame_{idx:03d}.jpg"
        if out_path.exists():
            frame_paths.append(out_path)
            continue
        time_s = time_ms / 1000
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(time_s),
            "-i",
            str(path),
            "-frames:v",
            "1",
            str(out_path),
        ]
        result = run_command(cmd)
        if result.returncode == 0 and out_path.exists():
            frame_paths.append(out_path)
    return frame_paths


def describe_media(config: AppConfig, session, item: MediaItem, write_sidecars: bool = True) -> None:
    backend = _select_backend(config)
    people_counts = (
        session.execute(
            select(MediaFace.person_id, MediaFace.count)
            .where(MediaFace.media_id == item.media_id)
            .order_by(MediaFace.count.desc())
        )
        .all()
    )
    people_payload = []
    for person_id, count in people_counts:
        person = session.get(Person, person_id)
        name = person.display_name or f"unknown_{person.person_id:06d}"
        people_payload.append({"name": name, "count": count, "notes": person.notes})

    metadata = {}
    location_text = "Location unknown"
    capture_datetime = None
    if item.type == "image":
        exif = extract_exif(Path(item.path))
        metadata = exif
        capture_datetime = _capture_datetime_from_exif(exif)
        lat = exif.get("GPSLatitude")
        lon = exif.get("GPSLongitude")
        location_text = format_location(lat, lon, reverse_geocode=False)
    else:
        meta = extract_ffprobe(Path(item.path))
        metadata = meta
        capture_datetime = _capture_datetime_from_ffprobe(meta)
        duration = float(meta.get("format", {}).get("duration", 0))
        plan = build_sampling_plan(duration, 0.5, 5, 12)
        images = [str(p) for p in _frame_paths_for_video(Path(item.path), config.cache_dir, plan.times_ms)]
        if not images:
            images = [item.path]

    if item.type == "image":
        images = [item.path]
    result = backend.describe(
        images=images,
        people=people_payload,
        location_text=location_text,
        capture_datetime=capture_datetime,
        media_type=item.type,
        metadata=metadata,
    )

    sidecar_json = {
        "original_path": item.path,
        "hash": item.hash,
        "capture_datetime": capture_datetime,
        "location_text": location_text,
        "detected_persons": people_payload,
        "llm_backend": config.llm.backend,
        "llm_model": config.llm.model,
        "summary": result.summary,
        "description": result.description,
        "tags": result.tags,
        "suggested_filename_base": result.suggested_filename_base,
        "pipeline_version": config.pipeline.pipeline_version,
    }

    if write_sidecars:
        write_text_sidecar(Path(item.path), result.description)
        write_json_sidecar(Path(item.path), sidecar_json)

    item.meta_json = json.dumps(sidecar_json)
    dao.mark_media_status(session, item, "llm_done")
    session.commit()
    logger.info("Description generated for {}", item.path)
