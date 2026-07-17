"""The PRISM navigation shell: a persistent top-left title + back arrow over a stack of screens.

Navigation is two levels deep -- a Home/splash screen (index 0) and the section screens -- so the back
arrow always returns Home. The "PRISM" title stays in the top-left AT ALL TIMES; the back arrow sits
just below it and is hidden on Home.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from ..widgets.anim import slide_screens, snapshot


class NavShell(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.title = QLabel("PRISM")
        self.title.setProperty("type", "subtitle")     # Fluent type ramp (global QSS)

        self.btn_back = QToolButton()
        self.btn_back.setObjectName("navBack")          # -> QToolButton#navBack in the global QSS
        self.btn_back.setText("←")
        self.btn_back.setToolTip("Back to the home screen")
        self.btn_back.setAutoRaise(True)
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

        # A persistent settings gear in the bottom-left seam (shown on every screen). Its menu is
        # attached by MainWindow (which knows the appearance controller + the Settings screen index).
        self.btn_settings = QToolButton()
        self.btn_settings.setObjectName("navSettings")   # -> QToolButton#navSettings in the global QSS
        self.btn_settings.setText("⚙")
        self.btn_settings.setToolTip("Appearance & settings")
        self.btn_settings.setAutoRaise(True)
        self.btn_settings.setPopupMode(QToolButton.InstantPopup)

        settings_row = QHBoxLayout()
        settings_row.addWidget(self.btn_settings)
        settings_row.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.addLayout(header_row)
        outer.addWidget(self.stack, 1)
        outer.addLayout(settings_row)

    def add_screen(self, widget) -> int:
        """Append a screen; the first one added (index 0) is Home."""
        return self.stack.addWidget(widget)

    def go_home(self) -> None:
        self._navigate(0)

    def go_to(self, index: int) -> None:
        self._navigate(index)

    def _navigate(self, index: int) -> None:
        prev = self.stack.currentIndex()
        # Snapshot the OUTGOING screen before the switch, flip logical state SYNCHRONOUSLY (tests read the
        # index / back-arrow right after with no pump), then slide the incoming screen in over the snapshot.
        old_pm = snapshot(self.stack.currentWidget()) if index != prev else None
        self.stack.setCurrentIndex(index)
        self._sync_back()
        if index != prev:
            slide_screens(self.stack, old_pm, forward=index > prev)   # no-op offscreen / not-yet-visible

    def _sync_back(self) -> None:
        self.btn_back.setVisible(self.stack.currentIndex() != 0)
