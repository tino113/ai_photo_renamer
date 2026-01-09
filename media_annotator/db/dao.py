from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from media_annotator.db.models import FaceEmbedding, MediaFace, MediaItem, Person, RenameHistory


def get_or_create_media_item(
    session: Session,
    path: str,
    hash_value: str,
    media_type: str,
    pipeline_version: str,
) -> MediaItem:
    existing = session.execute(select(MediaItem).where(MediaItem.path == path)).scalars().first()
    if existing:
        existing.hash = hash_value
        existing.type = media_type
        existing.pipeline_version = pipeline_version
        return existing
    item = MediaItem(
        path=path,
        hash=hash_value,
        type=media_type,
        pipeline_version=pipeline_version,
        status="discovered",
    )
    session.add(item)
    return item


def mark_media_status(session: Session, item: MediaItem, status: str, error: Optional[str] = None) -> None:
    item.status = status
    item.error_message = error
    item.last_processed_at = datetime.utcnow()


def add_face_embedding(
    session: Session,
    person_id: int,
    media_path: str,
    media_hash: str,
    embedding: bytes,
    bbox: Optional[str],
    frame_time_ms: Optional[int],
    quality_score: Optional[float],
) -> None:
    session.add(
        FaceEmbedding(
            person_id=person_id,
            media_path=media_path,
            media_hash=media_hash,
            frame_time_ms=frame_time_ms,
            bbox=bbox,
            embedding=embedding,
            quality_score=quality_score,
        )
    )


def update_media_face_summary(
    session: Session,
    media_id: int,
    person_id: int,
    frame_time_ms: Optional[int] = None,
) -> None:
    existing = (
        session.execute(
            select(MediaFace).where(
                MediaFace.media_id == media_id, MediaFace.person_id == person_id
            )
        )
        .scalars()
        .first()
    )
    if not existing:
        existing = MediaFace(
            media_id=media_id,
            person_id=person_id,
            count=1,
            first_seen_frame_ms=frame_time_ms,
            last_seen_frame_ms=frame_time_ms,
        )
        session.add(existing)
        return
    existing.count += 1
    if frame_time_ms is not None:
        if existing.first_seen_frame_ms is None or frame_time_ms < existing.first_seen_frame_ms:
            existing.first_seen_frame_ms = frame_time_ms
        if existing.last_seen_frame_ms is None or frame_time_ms > existing.last_seen_frame_ms:
            existing.last_seen_frame_ms = frame_time_ms


def get_unknown_people(session: Session) -> Iterable[Person]:
    return session.execute(select(Person).where(Person.is_known.is_(False))).scalars().all()


def upsert_person(session: Session, display_name: Optional[str], is_known: bool) -> Person:
    person = Person(display_name=display_name, is_known=is_known)
    session.add(person)
    return person


def record_rename_history(
    session: Session,
    media_hash: str,
    old_path: str,
    new_path: str,
    sidecars_old: list[str],
    sidecars_new: list[str],
    mode: str,
) -> None:
    session.add(
        RenameHistory(
            media_hash=media_hash,
            old_path=old_path,
            new_path=new_path,
            sidecars_old=json.dumps(sidecars_old),
            sidecars_new=json.dumps(sidecars_new),
            mode=mode,
        )
    )
