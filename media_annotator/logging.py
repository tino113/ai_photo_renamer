from __future__ import annotations

from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        log_dir / "media_annotator.log",
        rotation="10 MB",
        retention="10 days",
        level="INFO",
    )
    logger.add(lambda msg: print(msg, end=""))
