from __future__ import annotations

from PySide6.QtGui import QStandardItemModel


class RenamePreviewModel(QStandardItemModel):
    def __init__(self) -> None:
        super().__init__()
        self.setHorizontalHeaderLabels(["Old Path", "New Path"])
