from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from media_annotator.db.models import Base, SchemaMeta

SCHEMA_VERSION = 1


def run_migrations(session: Session) -> None:
    Base.metadata.create_all(session.bind)
    existing = session.execute(select(SchemaMeta)).scalars().first()
    if not existing:
        session.add(SchemaMeta(id=1, schema_version=SCHEMA_VERSION))
        session.commit()
    elif existing.schema_version != SCHEMA_VERSION:
        existing.schema_version = SCHEMA_VERSION
        session.commit()
