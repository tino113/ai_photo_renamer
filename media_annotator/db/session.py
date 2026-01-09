from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from media_annotator.db.models import get_engine


def create_session(db_path: str):
    engine = get_engine(db_path)
    return sessionmaker(bind=engine, future=True)
