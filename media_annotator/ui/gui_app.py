from __future__ import annotations

import sys
import subprocess

from PySide6.QtWidgets import QApplication, QMessageBox

from media_annotator.ui.main_window import MainWindow


def run_gui() -> None:
    app = QApplication(sys.argv)
    missing = []
    for binary, args in {"exiftool": ["-ver"], "ffprobe": ["-version"]}.items():
        result = subprocess.run([binary] + args, capture_output=True, text=True)
        if result.returncode != 0:
            missing.append(binary)
    if missing:
        QMessageBox.warning(None, "Missing dependencies", f"Missing binaries: {', '.join(missing)}")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
