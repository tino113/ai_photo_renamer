from __future__ import annotations

from pathlib import Path
from loguru import logger

from media_annotator.config import AppConfig
from media_annotator.db import dao
from media_annotator.db.migrations import run_migrations
from media_annotator.db.session import create_session
from media_annotator.db.models import MediaItem
from media_annotator.pipeline.cache import should_process
from media_annotator.pipeline.describe_media import describe_media
from media_annotator.scan.discover import discover_media
from media_annotator.scan.media_info import media_type_for
from media_annotator.utils.hashing import hash_file


def scan_media(config: AppConfig, input_dir: Path) -> None:
    config.ensure_dirs()
    session_factory = create_session(str(config.db_path))
    with session_factory() as session:
        run_migrations(session)
        for media_path in discover_media(input_dir):
            hash_value = hash_file(media_path)
            media_type = media_type_for(media_path)
            item = dao.get_or_create_media_item(
                session,
                path=str(media_path),
                hash_value=hash_value,
                media_type=media_type,
                pipeline_version=config.pipeline.pipeline_version,
            )
            session.add(item)
        session.commit()
    logger.info("Scan complete for {}", input_dir)


def run_faces(config: AppConfig, input_dir: Path, progress_callback=None) -> None:
    config.ensure_dirs()
    session_factory = create_session(str(config.db_path))
    with session_factory() as session:
        run_migrations(session)
        for item in session.query(MediaItem).filter(MediaItem.path.like(f"{input_dir}%")).all():
            if not Path(item.path).exists():
                continue
            if not should_process(item, config.pipeline.pipeline_version, config.pipeline.force, "faces_done"):
                continue
            try:
                from media_annotator.pipeline.preprocess_faces import preprocess_faces

                preprocess_faces(config, session, item)
                if progress_callback:
                    progress_callback(item.path, "faces_done")
            except Exception as exc:
                dao.mark_media_status(session, item, "error", str(exc))
                session.commit()
                logger.error("Failed faces for {}: {}", item.path, exc)


def run_describe(config: AppConfig, input_dir: Path, progress_callback=None) -> None:
    config.ensure_dirs()
    session_factory = create_session(str(config.db_path))
    with session_factory() as session:
        run_migrations(session)
        for item in session.query(MediaItem).filter(MediaItem.path.like(f\"{input_dir}%\")).all():
            if not Path(item.path).exists():
                continue
            if not should_process(item, config.pipeline.pipeline_version, config.pipeline.force, "llm_done"):
                continue
            try:
                describe_media(config, session, item, write_sidecars=True)
                if progress_callback:
                    progress_callback(item.path, "llm_done")
            except Exception as exc:
                dao.mark_media_status(session, item, "error", str(exc))
                session.commit()
                logger.error("Failed describe for {}: {}", item.path, exc)
