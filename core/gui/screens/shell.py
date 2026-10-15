"""The window's two regions: the rail down its left edge and, beside it, the page of the part in sight.

The rail is the way between the parts; the page is the part's own top widget, one of a stack. The shell holds
the parts by their keys and shows one at a time, at once, with no animation. When the window is lower than the
look's threshold it tells the rail and every part that offers ``set_small`` to take the small layout, and tells
them again when the window grows past it. Which part to show, and when, is the main window's to decide.
"""
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QWidget

from .. import design
from ..reported import reported
from ..widgets.rail import Rail


def clock_words(seconds: int) -> str:
    """How long a run has gone on, as the foot of the rail writes it: minutes and seconds, "4:07", and from an
    hour the hours as well, "1:02:05". A time below nothing, as a jump of the computer's clock can give, reads
    as nothing."""
    hours, rest = divmod(max(0, int(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class Shell(QWidget):
    """The rail and, beside it, a stack of the parts, with no margins and no space between them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parts: dict = {}
        self._small = False
        self.rail = Rail()
        self.stack = QStackedWidget()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.rail)
        row.addWidget(self.stack, 1)

    def add_part(self, name: str, widget: QWidget) -> None:
        """Hold ``widget`` as the part ``name``, unseen until it is shown. A part added while the window is low
        is told so at once. A name held already raises ValueError."""
        if name in self._parts:
            raise ValueError(f"a part of that name is held already: {name!r}")
        self._parts[name] = widget
        self.stack.addWidget(widget)
        told = getattr(widget, "set_small", None)
        if self._small and callable(told):
            told(True)

    def show_part(self, name: str) -> None:
        """Show the part ``name`` in place of the one in sight; KeyError for a part not held."""
        self.stack.setCurrentWidget(self._parts[name])

    def has_part(self, name: str) -> bool:
        return name in self._parts

    def part_widget(self, name: str) -> QWidget | None:
        """The part held as ``name``, or None."""
        return self._parts.get(name)

    def set_small(self, on: bool) -> None:
        """The small layout, or the usual one again: told to the rail and to every part that offers it."""
        self._small = bool(on)
        self.rail.set_small(self._small)
        for widget in self._parts.values():
            told = getattr(widget, "set_small", None)
            if callable(told):
                told(self._small)

    @reported(QWidget.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        small = self.height() < design.small_below()
        if small != self._small:
            self.set_small(small)
