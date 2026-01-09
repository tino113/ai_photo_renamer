from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    backend: str = Field(default="ollama")
    model: str = Field(default="llava")
    base_url: Optional[str] = None
    temperature: float = 0.2
    timeout_s: int = 120


class FaceConfig(BaseModel):
    known_match_threshold: float = 0.65
    unknown_match_threshold: float = 0.7
    video_sample_rate: float = 0.5
    video_min_frames: int = 10
    video_max_frames: int = 300
    use_faiss: bool = True


class PipelineConfig(BaseModel):
    pipeline_version: str = "1.0"
    force: bool = False
    dry_run: bool = True
    copy_mode: bool = False
    copy_mirror_structure: bool = False
    max_filename_length: int = 120


class AppConfig(BaseModel):
    db_path: Path = Field(default_factory=lambda: Path.home() / ".media_annotator" / "media_annotator.db")
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".media_annotator" / "cache")
    log_dir: Path = Field(default_factory=lambda: Path.home() / ".media_annotator" / "logs")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    faces: FaceConfig = Field(default_factory=FaceConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
