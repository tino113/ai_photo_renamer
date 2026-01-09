from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from media_annotator.config import AppConfig
from media_annotator.pipeline.runner import run_describe, run_faces, scan_media


class PipelineWorker(QThread):
    progress = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, input_dir: Path, enable_faces: bool, enable_llm: bool) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.enable_faces = enable_faces
        self.enable_llm = enable_llm

    def run(self) -> None:
        config = AppConfig()
        try:
            self.progress.emit("Scanning")
            scan_media(config, self.input_dir)
            if self.isInterruptionRequested():
                self.finished.emit()
                return
            if self.enable_faces:
                self.progress.emit("Processing faces")
                run_faces(config, self.input_dir)
                if self.isInterruptionRequested():
                    self.finished.emit()
                    return
            if self.enable_llm:
                self.progress.emit("Generating descriptions")
                run_describe(config, self.input_dir)
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
