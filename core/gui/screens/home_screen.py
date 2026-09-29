"""The PRISM home / splash screen: a time-of-day greeting and one button per section.

`greeting(hour)` is a pure function (easy to unit-test); the screen calls it with the local clock's
hour on every show, so a session left open across the day updates rather than freezing on "morning".
"""
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from .nav_shell import RUNNING_MARK

# The five sections, in display order. All five are live; MainWindow passes them all in
# live_sections. "Artifacts" is the artifact browser -- a peer of the four stage sections, and a
# plain screen rather than a panel.
SECTIONS = ("Reduction Map", "FDT Analysis", "Parameter Inference", "Simulate", "Artifacts")


def greeting(hour: int) -> str:
    """A greeting for a 24-hour clock hour: morning 5-11, afternoon 12-16, otherwise evening."""
    if 5 <= hour <= 11:
        return "Good morning"
    if 12 <= hour <= 16:
        return "Good afternoon"
    return "Good evening"


class HomeScreen(QWidget):
    """Emits ``navigate(section_name)`` when a live section button is clicked. The owner (MainWindow)
    maps the name to a stack index. `live_sections` is the set of names that actually have a screen;
    any name NOT in it becomes a no-op button with a "Coming soon" tooltip (all five are live today)."""

    navigate = Signal(str)

    def __init__(self, live_sections, parent=None):
        super().__init__(parent)
        live = set(live_sections)

        self.greeting_label = QLabel()
        self.greeting_label.setAlignment(Qt.AlignCenter)
        self.greeting_label.setProperty("type", "title")     # Fluent type ramp (global QSS)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)
        layout.addStretch(1)
        layout.addWidget(self.greeting_label)
        layout.addSpacing(10)

        # Kept on self: the running section's tile takes a marker, so the live run is shown
        # app-wide, and a tile that was only ever added to a layout cannot be found again.
        self.tiles: dict = {}
        for name in SECTIONS:
            btn = QPushButton(name)
            btn.setMinimumWidth(260)
            btn.setMinimumHeight(44)
            if name in live:
                btn.setProperty("accent", True)         # live section tiles are primary (Fluent accent)
                btn.clicked.connect(lambda _=False, n=name: self.navigate.emit(n))
            else:
                btn.setToolTip("Coming soon")           # a name with no screen: clickable, no target
            self.tiles[name] = btn
            layout.addWidget(btn, alignment=Qt.AlignCenter)

        layout.addStretch(2)
        self._refresh_greeting()

    def _refresh_greeting(self):
        self.greeting_label.setText(greeting(datetime.now().hour))

    def set_running_section(self, name: "str | None") -> None:
        """Mark one section's tile as the one with a live run; None (or an unknown name) clears every
        mark. A SUFFIX on the tile's own label, rewritten from the section name rather than edited in
        place, so marking is idempotent and a cleared tile reads exactly as it was built. Nothing is
        disabled -- every tile stays clickable while a run is live."""
        running = name if name in self.tiles else None
        for section, btn in self.tiles.items():
            text = f"{section}  {RUNNING_MARK}" if section == running else section
            if btn.text() != text:
                btn.setText(text)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_greeting()
