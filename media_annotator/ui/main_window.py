from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

from media_annotator.config import AppConfig
from media_annotator.db import dao
from media_annotator.db.migrations import run_migrations
from media_annotator.db.models import FaceEmbedding, MediaItem, MediaFace, Person
from media_annotator.db.session import create_session
from media_annotator.pipeline.rename_plan import generate_plan
from media_annotator.ui.workers import PipelineWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Media Annotator")
        self.config = AppConfig()
        self.worker: PipelineWorker | None = None
        self.settings = QSettings("media-annotator", "media-annotator")

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.pipeline_tab = QWidget()
        self.unknowns_tab = QWidget()
        self.rename_tab = QWidget()
        self.settings_tab = QWidget()

        self.tabs.addTab(self.pipeline_tab, "Pipeline")
        self.tabs.addTab(self.unknowns_tab, "Unknown People")
        self.tabs.addTab(self.rename_tab, "Rename Preview")
        self.tabs.addTab(self.settings_tab, "Settings")

        self._build_pipeline_tab()
        self._build_unknowns_tab()
        self._build_rename_tab()
        self._build_settings_tab()
        self._load_settings()

    def _build_pipeline_tab(self) -> None:
        layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_input)
        input_layout.addWidget(QLabel("Input Folder"))
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(browse_btn)
        layout.addLayout(input_layout)

        self.enable_faces = QPushButton("Faces: ON")
        self.enable_faces.setCheckable(True)
        self.enable_faces.setChecked(True)
        self.enable_faces.clicked.connect(self._toggle_button)

        self.enable_llm = QPushButton("LLM: ON")
        self.enable_llm.setCheckable(True)
        self.enable_llm.setChecked(True)
        self.enable_llm.clicked.connect(self._toggle_button)

        layout.addWidget(self.enable_faces)
        layout.addWidget(self.enable_llm)

        settings_btn = QPushButton("Open Settings")
        settings_btn.clicked.connect(self._open_settings_tab)
        layout.addWidget(settings_btn)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start_pipeline)
        layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_pipeline)
        layout.addWidget(self.cancel_btn)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        layout.addWidget(self.progress_bar)

        self.pipeline_tab.setLayout(layout)

    def _build_unknowns_tab(self) -> None:
        layout = QVBoxLayout()
        self.unknown_list = QListWidget()
        layout.addWidget(self.unknown_list)
        self.rename_input = QLineEdit()
        self.rename_btn = QPushButton("Save Name")
        self.rename_btn.clicked.connect(self._save_unknown_name)
        layout.addWidget(self.rename_input)
        layout.addWidget(self.rename_btn)
        layout.addWidget(QLabel("Detected face crops"))
        self.example_list = QListWidget()
        self.example_list.setViewMode(QListWidget.IconMode)
        self.example_list.setIconSize(QPixmap(96, 96).size())
        self.example_list.setResizeMode(QListWidget.Adjust)
        self.example_list.setSpacing(8)
        layout.addWidget(self.example_list)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_unknowns)
        layout.addWidget(refresh_btn)
        self.unknowns_tab.setLayout(layout)
        self.unknown_list.currentItemChanged.connect(self._show_examples)

    def _build_rename_tab(self) -> None:
        layout = QVBoxLayout()
        self.rename_preview = QListWidget()
        layout.addWidget(self.rename_preview)
        generate_btn = QPushButton("Generate Preview")
        generate_btn.clicked.connect(self._generate_preview)
        layout.addWidget(generate_btn)
        self.rename_tab.setLayout(layout)

    def _build_settings_tab(self) -> None:
        layout = QVBoxLayout()
        form_group = QGroupBox("LLM Settings")
        form = QFormLayout(form_group)

        self.llm_backend = QComboBox()
        self.llm_backend.addItems(["ollama", "lmstudio", "local"])
        form.addRow("LLM Backend", self.llm_backend)

        self.llm_model = QLineEdit()
        form.addRow("LLM Model", self.llm_model)

        self.llm_base_url = QLineEdit()
        form.addRow("Base URL", self.llm_base_url)

        self.llm_temperature = QDoubleSpinBox()
        self.llm_temperature.setRange(0.0, 1.0)
        self.llm_temperature.setSingleStep(0.05)
        form.addRow("Temperature", self.llm_temperature)

        self.llm_timeout = QSpinBox()
        self.llm_timeout.setRange(1, 600)
        form.addRow("Timeout (s)", self.llm_timeout)

        layout.addWidget(form_group)
        apply_btn = QPushButton("Apply Settings")
        apply_btn.clicked.connect(self._apply_settings)
        layout.addWidget(apply_btn)
        layout.addStretch()
        self.settings_tab.setLayout(layout)

    def _browse_input(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if directory:
            self.input_path.setText(directory)
            self.settings.setValue("input_path", directory)

    def _toggle_button(self) -> None:
        button = self.sender()
        if isinstance(button, QPushButton):
            label = button.text().split(":")[0]
            button.setText(f"{label}: {'ON' if button.isChecked() else 'OFF'}")

    def _open_settings_tab(self) -> None:
        self.tabs.setCurrentWidget(self.settings_tab)

    def _apply_settings(self) -> None:
        self._save_settings()
        self._apply_settings_to_config()

    def _start_pipeline(self) -> None:
        input_dir = Path(self.input_path.text())
        if not input_dir.exists():
            QMessageBox.warning(self, "Error", "Input directory does not exist")
            return
        self._save_settings()
        self._apply_settings_to_config()
        self.worker = PipelineWorker(
            input_dir=input_dir,
            enable_faces=self.enable_faces.isChecked(),
            enable_llm=self.enable_llm.isChecked(),
            config=self.config,
        )
        self.worker.progress.connect(self._append_log)
        self.worker.error.connect(self._append_log)
        self.worker.finished.connect(self._pipeline_finished)
        self.progress_bar.setRange(0, 0)
        self.worker.start()

    def _cancel_pipeline(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self._append_log("Cancellation requested")
            self.progress_bar.setRange(0, 1)

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    def _save_settings(self) -> None:
        self.settings.setValue("input_path", self.input_path.text())
        self.settings.setValue("faces_enabled", self.enable_faces.isChecked())
        self.settings.setValue("llm_enabled", self.enable_llm.isChecked())
        self.settings.setValue("llm_backend", self.llm_backend.currentText())
        self.settings.setValue("llm_model", self.llm_model.text())
        self.settings.setValue("llm_base_url", self.llm_base_url.text())
        self.settings.setValue("llm_temperature", self.llm_temperature.value())
        self.settings.setValue("llm_timeout_s", self.llm_timeout.value())

    def _load_settings(self) -> None:
        self.input_path.setText(self.settings.value("input_path", ""))
        self.enable_faces.setChecked(self.settings.value("faces_enabled", True, type=bool))
        self.enable_llm.setChecked(self.settings.value("llm_enabled", True, type=bool))
        self.enable_faces.setText(f"Faces: {'ON' if self.enable_faces.isChecked() else 'OFF'}")
        self.enable_llm.setText(f"LLM: {'ON' if self.enable_llm.isChecked() else 'OFF'}")
        self.llm_backend.setCurrentText(self.settings.value("llm_backend", self.config.llm.backend))
        self.llm_model.setText(self.settings.value("llm_model", self.config.llm.model))
        self.llm_base_url.setText(self.settings.value("llm_base_url", self.config.llm.base_url or ""))
        self.llm_temperature.setValue(
            self.settings.value("llm_temperature", self.config.llm.temperature, type=float)
        )
        self.llm_timeout.setValue(self.settings.value("llm_timeout_s", self.config.llm.timeout_s, type=int))

    def _apply_settings_to_config(self) -> None:
        self.config.llm.backend = self.llm_backend.currentText()
        self.config.llm.model = self.llm_model.text().strip() or self.config.llm.model
        base_url = self.llm_base_url.text().strip()
        self.config.llm.base_url = base_url or None
        self.config.llm.temperature = float(self.llm_temperature.value())
        self.config.llm.timeout_s = int(self.llm_timeout.value())

    def _append_log(self, message: str) -> None:
        self.log_output.append(message)
        if message.lower().startswith("error"):
            self.progress_bar.setRange(0, 1)

    def _pipeline_finished(self) -> None:
        self._append_log("Pipeline finished")
        self.progress_bar.setRange(0, 1)

    def refresh_unknowns(self) -> None:
        self.unknown_list.clear()
        session_factory = create_session(str(self.config.db_path))
        with session_factory() as session:
            run_migrations(session)
            unknowns = dao.get_unknown_people(session)
            for person in unknowns:
                count = (
                    session.query(MediaFace)
                    .filter(MediaFace.person_id == person.person_id)
                    .with_entities(MediaFace.count)
                    .all()
                )
                total = sum(c[0] for c in count)
                item = QListWidgetItem(f"{person.person_id}: {person.display_name} ({total})")
                item.setData(Qt.UserRole, person.person_id)
                self.unknown_list.addItem(item)
        self.example_list.clear()

    def _show_examples(self) -> None:
        self.example_list.clear()
        selected = self.unknown_list.currentItem()
        if not selected:
            return
        person_id = selected.data(Qt.UserRole)
        session_factory = create_session(str(self.config.db_path))
        with session_factory() as session:
            examples = (
                session.query(FaceEmbedding)
                .filter(FaceEmbedding.person_id == person_id)
                .order_by(FaceEmbedding.quality_score.desc().nullslast(), FaceEmbedding.embedding_id.desc())
                .all()
            )
        for example in examples:
            item = QListWidgetItem()
            item.setToolTip(Path(example.media_path).name)
            try:
                bbox = json.loads(example.bbox or "[]")
                if bbox and Path(example.media_path).suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".tiff", ".heic", ".heif"]:
                    image = Image.open(example.media_path).convert("RGB")
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = image.crop((x1, y1, x2, y2)).resize((96, 96))
                    qimage = QImage(
                        crop.tobytes(),
                        crop.width,
                        crop.height,
                        crop.width * 3,
                        QImage.Format_RGB888,
                    ).copy()
                    item.setIcon(QPixmap.fromImage(qimage))
                    item.setText("")
            except Exception:
                pass
            self.example_list.addItem(item)

    def _save_unknown_name(self) -> None:
        selected = self.unknown_list.currentItem()
        if not selected:
            return
        person_id = selected.data(Qt.UserRole)
        new_name = self.rename_input.text().strip()
        if not new_name:
            return
        session_factory = create_session(str(self.config.db_path))
        with session_factory() as session:
            person = session.get(Person, person_id)
            person.display_name = new_name
            person.is_known = True
            session.commit()
        self.refresh_unknowns()

    def _generate_preview(self) -> None:
        input_dir = Path(self.input_path.text())
        if not input_dir.exists():
            return
        session_factory = create_session(str(self.config.db_path))
        with session_factory() as session:
            run_migrations(session)
            items = [
                {"path": item.path, "meta_json": item.meta_json, "hash": item.hash}
                for item in session.query(MediaItem).filter(MediaItem.path.like(f"{input_dir}%")).all()
            ]
        plan = generate_plan(self.config, items, output_root=None, input_root=input_dir)
        self.rename_preview.clear()
        for op in plan["operations"]:
            self.rename_preview.addItem(f"{op['old_path']} -> {op['new_path']}")
