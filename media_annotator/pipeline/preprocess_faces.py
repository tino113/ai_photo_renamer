from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

from media_annotator.config import AppConfig
from media_annotator.db import dao
from media_annotator.db.models import MediaItem, Person
from media_annotator.faces.clustering import search_embeddings
from media_annotator.faces.insightface_backend import InsightFaceBackend, load_image
from media_annotator.faces.video_sampling import build_sampling_plan
from media_annotator.metadata.ffprobe import extract_ffprobe
from media_annotator.utils.subprocess import run_command


def _match_person(
    embedding: np.ndarray,
    known_embeddings: np.ndarray,
    known_ids: List[int],
    unknown_embeddings: np.ndarray,
    unknown_ids: List[int],
    known_threshold: float,
    unknown_threshold: float,
    use_faiss: bool,
) -> int | None:
    if known_embeddings.size:
        result = search_embeddings(embedding, known_embeddings, known_ids, known_threshold, use_faiss)
        if result.person_id is not None:
            return result.person_id
    result = search_embeddings(embedding, unknown_embeddings, unknown_ids, unknown_threshold, use_faiss)
    if result.person_id is not None:
        return result.person_id
    return None


def _frame_paths_for_video(path: Path, cache_dir: Path, sample_plan: List[int]) -> List[Path]:
    output_dir = cache_dir / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for idx, time_ms in enumerate(sample_plan):
        out_path = output_dir / f"frame_{idx:04d}.jpg"
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


def preprocess_faces(
    config: AppConfig,
    session,
    item: MediaItem,
) -> None:
    backend = InsightFaceBackend()
    known_people = session.query(Person).filter(Person.is_known.is_(True)).all()
    unknown_people = session.query(Person).filter(Person.is_known.is_(False)).all()
    known_embeddings = []
    known_ids = []
    for person in known_people:
        for emb in person.embeddings:
            known_embeddings.append(np.frombuffer(emb.embedding, dtype=np.float32))
            known_ids.append(person.person_id)
    unknown_embeddings = []
    unknown_ids = []
    for person in unknown_people:
        for emb in person.embeddings:
            unknown_embeddings.append(np.frombuffer(emb.embedding, dtype=np.float32))
            unknown_ids.append(person.person_id)
    known_embeddings = np.vstack(known_embeddings) if known_embeddings else np.array([], dtype=np.float32).reshape(0, 512)
    unknown_embeddings = (
        np.vstack(unknown_embeddings) if unknown_embeddings else np.array([], dtype=np.float32).reshape(0, 512)
    )

    def handle_embedding(face, frame_time_ms=None):
        nonlocal known_embeddings, unknown_embeddings, known_ids, unknown_ids
        match = _match_person(
            face.embedding,
            known_embeddings,
            known_ids,
            unknown_embeddings,
            unknown_ids,
            config.faces.known_match_threshold,
            config.faces.unknown_match_threshold,
            config.faces.use_faiss,
        )
        if match is None:
            display_name = f"unknown_{len(unknown_people) + 1:06d}"
            person = dao.upsert_person(session, display_name=display_name, is_known=False)
            session.flush()
            unknown_people.append(person)
            match = person.person_id
            unknown_embeddings = np.vstack([unknown_embeddings, face.embedding]) if unknown_embeddings.size else face.embedding[np.newaxis, :]
            unknown_ids.append(match)
        else:
            if match in known_ids:
                known_embeddings = np.vstack([known_embeddings, face.embedding]) if known_embeddings.size else face.embedding[np.newaxis, :]
                known_ids.append(match)
            else:
                unknown_embeddings = np.vstack([unknown_embeddings, face.embedding]) if unknown_embeddings.size else face.embedding[np.newaxis, :]
                unknown_ids.append(match)
        dao.add_face_embedding(
            session,
            person_id=match,
            media_path=item.path,
            media_hash=item.hash,
            embedding=face.embedding.astype(np.float32).tobytes(),
            bbox=json.dumps(face.bbox),
            frame_time_ms=frame_time_ms,
            quality_score=face.quality,
        )
        dao.update_media_face_summary(session, item.media_id, match, frame_time_ms)

    if item.type == "image":
        image = load_image(item.path)
        faces = backend.detect(image)
        for face in faces:
            handle_embedding(face)
    else:
        metadata = extract_ffprobe(Path(item.path))
        duration = float(metadata.get("format", {}).get("duration", 0))
        plan = build_sampling_plan(
            duration,
            config.faces.video_sample_rate,
            config.faces.video_min_frames,
            config.faces.video_max_frames,
        )
        frames = _frame_paths_for_video(Path(item.path), config.cache_dir, plan.times_ms)
        for frame_path, time_ms in zip(frames, plan.times_ms):
            image = load_image(str(frame_path))
            faces = backend.detect(image)
            for face in faces:
                handle_embedding(face, frame_time_ms=time_ms)
    dao.mark_media_status(session, item, "faces_done")
    session.commit()
    logger.info("Faces processed for {}", item.path)
