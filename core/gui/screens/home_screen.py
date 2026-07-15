"""The MAPPI home / splash screen: a time-of-day greeting and one button per section.

`greeting(hour)` is a pure function (easy to unit-test); the screen calls it with the local clock's
hour on every show, so a session left open across the day updates rather than freezing on "morning".
"""
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

# The four sections, in display order. "Simulate" has no screen yet (deferred), so it is a no-op button.
SECTIONS = ("Reduction Map", "FDT Analysis", "Parameter Inference", "Simulate")


def greeting(hour: int) -> str:
    """A greeting for a 24-hour clock hour: morning 5-11, afternoon 12-16, otherwise evening."""
    if 5 <= hour <= 11:
        return "Good morning"
    if 12 <= hour <= 16:
        return "Good afternoon"
    return "Good evening"


class HomeScreen(QWidget):
    """Emits ``navigate(section_name)`` when a live section button is clicked. The owner (MainWindow)
    maps the name to a stack index. `live_sections` is the set of names that actually have a screen --
    "Simulate" is omitted, so its button is a no-op with a "Coming soon" tooltip."""

    navigate = Signal(str)

    def __init__(self, live_sections, parent=None):
        super().__init__(parent)
        live = set(live_sections)

        self.greeting_label = QLabel()
        self.greeting_label.setAlignment(Qt.AlignCenter)
        self.greeting_label.setStyleSheet("font-size: 26px; font-weight: bold;")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)
        layout.addStretch(1)
        layout.addWidget(self.greeting_label)
        layout.addSpacing(10)

        for name in SECTIONS:
            btn = QPushButton(name)
            btn.setMinimumWidth(260)
            btn.setMinimumHeight(44)
            if name in live:
                btn.clicked.connect(lambda _=False, n=name: self.navigate.emit(n))
            else:
                btn.setToolTip("Coming soon")           # Simulate is deferred: clickable, no target
            layout.addWidget(btn, alignment=Qt.AlignCenter)

        layout.addStretch(2)
        self._refresh_greeting()

    def _refresh_greeting(self):
        self.greeting_label.setText(greeting(datetime.now().hour))

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_greeting()
