"""Small typed input widgets mirroring the CLI's numeric/path prompts (_prompt_int/_prompt_float and
the experimental-data file prompts)."""
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget


class FloatField(QLineEdit):
    def __init__(self, default: float = 0.0, parent=None):
        super().__init__(str(default), parent)
        self.setValidator(QDoubleValidator())

    def value(self) -> float:
        try:
            return float(self.text())
        except ValueError:
            return 0.0


class IntField(QLineEdit):
    def __init__(self, default: int = 0, parent=None):
        super().__init__(str(default), parent)
        self.setValidator(QIntValidator())

    def value(self) -> int:
        try:
            return int(self.text())
        except ValueError:
            return 0


class PathField(QWidget):
    def __init__(self, file_filter: str = "Data (*.csv *.npy);;All files (*)", parent=None):
        super().__init__(parent)
        self.edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", self._filter)
        if path:
            self.edit.setText(path)

    def value(self) -> str:
        return self.edit.text().strip()
