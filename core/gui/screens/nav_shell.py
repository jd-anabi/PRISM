"""The MAPPI navigation shell: a persistent top-left title + back arrow over a stack of screens.

Navigation is two levels deep -- a Home/splash screen (index 0) and the section screens -- so the back
arrow always returns Home. The "MAPPI" title stays in the top-left AT ALL TIMES; the back arrow sits
just below it and is hidden on Home.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStackedWidget, QToolButton, QVBoxLayout, QWidget


class NavShell(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.title = QLabel("MAPPI")
        self.title.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.btn_back = QToolButton()
        self.btn_back.setText("←")
        self.btn_back.setToolTip("Back to the home screen")
        self.btn_back.setAutoRaise(True)
        self.btn_back.setStyleSheet("font-size: 18px;")
        self.btn_back.clicked.connect(self.go_home)
        self.btn_back.setVisible(False)                  # hidden on Home; shown on any section

        header = QVBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(2)
        header.addWidget(self.title)
        header.addWidget(self.btn_back, alignment=Qt.AlignLeft)

        header_row = QHBoxLayout()
        header_row.addLayout(header)
        header_row.addStretch(1)

        self.stack = QStackedWidget()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.addLayout(header_row)
        outer.addWidget(self.stack, 1)

    def add_screen(self, widget) -> int:
        """Append a screen; the first one added (index 0) is Home."""
        return self.stack.addWidget(widget)

    def go_home(self) -> None:
        self.stack.setCurrentIndex(0)
        self._sync_back()

    def go_to(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self._sync_back()

    def _sync_back(self) -> None:
        self.btn_back.setVisible(self.stack.currentIndex() != 0)
