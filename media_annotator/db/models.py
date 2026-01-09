from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class SchemaMeta(Base):
    __tablename__ = "schema_meta"
    id = Column(Integer, primary_key=True)
    schema_version = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Person(Base):
    __tablename__ = "persons"
    person_id = Column(Integer, primary_key=True)
    display_name = Column(String, nullable=True)
    is_known = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
    embeddings = relationship("FaceEmbedding", back_populates="person")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    embedding_id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.person_id"))
    media_path = Column(Text, nullable=False)
    media_hash = Column(String, nullable=False)
    frame_time_ms = Column(Integer, nullable=True)
    bbox = Column(Text, nullable=True)
    embedding = Column(LargeBinary, nullable=False)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    person = relationship("Person", back_populates="embeddings")


class MediaItem(Base):
    __tablename__ = "media_items"
    media_id = Column(Integer, primary_key=True)
    path = Column(Text, unique=True, nullable=False)
    hash = Column(String, nullable=False)
    type = Column(String, nullable=False)
    exif_json = Column(Text, nullable=True)
    meta_json = Column(Text, nullable=True)
    last_processed_at = Column(DateTime, nullable=True)
    pipeline_version = Column(String, nullable=False)
    status = Column(String, nullable=False)
    error_message = Column(Text, nullable=True)


class MediaFace(Base):
    __tablename__ = "media_faces"
    id = Column(Integer, primary_key=True)
    media_id = Column(Integer, ForeignKey("media_items.media_id"))
    person_id = Column(Integer, ForeignKey("persons.person_id"))
    count = Column(Integer, default=0)
    first_seen_frame_ms = Column(Integer, nullable=True)
    last_seen_frame_ms = Column(Integer, nullable=True)


class RenameHistory(Base):
    __tablename__ = "rename_history"
    id = Column(Integer, primary_key=True)
    media_hash = Column(String, nullable=False)
    old_path = Column(Text, nullable=False)
    new_path = Column(Text, nullable=False)
    sidecars_old = Column(Text, nullable=True)
    sidecars_new = Column(Text, nullable=True)
    applied_at = Column(DateTime, default=datetime.utcnow)
    mode = Column(String, nullable=False)


def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", future=True)
