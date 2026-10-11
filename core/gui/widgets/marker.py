"""The state marker as a widget, and the same drawing as an icon.

Both draw through the one function that paints the markers and the glyphs, and neither keeps a colour: each
reads the look's table when it paints, so a change of theme takes a repaint and nothing more. An icon made
this way is sharp at every display scaling and needs no image file. A kind is checked when a marker or an
icon is built and when a marker is given another, never when it paints.
"""
from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QIcon, QIconEngine, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from .. import design, paint
from ..reported import reported

_TWINS: list = []           # every engine that clone() made, kept alive: see PaintedIcon.clone


class Marker(QWidget):
    """One marker or glyph as a widget: a square of the look's marker size, or of ``size``, either one a
    number of the look's table that is grown with the text here. Greyed, it draws itself disabled. It has
    no accessible name: a marker always stands beside the word for its state."""

    def __init__(self, kind: str, size: int | None = None):
        if kind not in paint.MARKERS + paint.GLYPHS:
            raise ValueError(f"not a marker or a glyph: {kind!r}")
        super().__init__()
        self._kind = kind
        side = design.scaled(design.SIZE["marker"] if size is None else size)
        self.setFixedSize(side, side)

    @property
    def kind(self) -> str:
        """The marker or glyph shown; changed through ``set_kind``."""
        return self._kind

    def set_kind(self, kind: str) -> None:
        """Show another marker or glyph, from the next repaint, which this asks for."""
        if kind not in paint.MARKERS + paint.GLYPHS:
            raise ValueError(f"not a marker or a glyph: {kind!r}")
        self._kind = kind
        self.update()

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the widget
            paint.paint_marker(painter, self._kind, QRectF(self.rect()), disabled=not self.isEnabled())
        finally:
            painter.end()


def _clear(_engine, size, *_told) -> QPixmap:
    """What the engine answers when its picture cannot be made: a clear one of the size asked."""
    blank = QPixmap(size)
    blank.fill(Qt.transparent)
    return blank


class PaintedIcon(QIconEngine):
    """An icon that is one marker or glyph, drawn whenever the toolkit asks for it: at the display scaling
    asked, in the colours of the theme in force, greyed where the toolkit draws the icon disabled."""

    def __init__(self, kind: str):
        if kind not in paint.MARKERS + paint.GLYPHS:
            raise ValueError(f"not a marker or a glyph: {kind!r}")
        super().__init__()
        self._kind = kind

    @reported()
    def paint(self, painter, rect, mode, state):
        side = min(rect.width(), rect.height())      # the glyph keeps to the left of a wider icon, clear of the words
        box = QRectF(rect.left(), rect.top() + (rect.height() - side) / 2, side, side)
        paint.paint_marker(painter, self._kind, box, disabled=(mode == QIcon.Disabled))

    @reported(_clear)
    def pixmap(self, size, mode, state):
        picture = QPixmap(size)
        picture.fill(Qt.transparent)
        painter = QPainter(picture)
        try:                                 # a picture let go while its painter still paints ends the process
            self.paint(painter, QRect(QPoint(0, 0), size), mode, state)
        finally:
            painter.end()
        return picture

    def clone(self):
        """A twin for a copy of the icon that is being changed. The twin is put in ``_TWINS`` before
        it is handed over: a new engine that no name holds is gone before the toolkit takes it, and
        the process ends without a word."""
        twin = PaintedIcon(self._kind)
        _TWINS.append(twin)
        return twin


def painted_icon(kind: str) -> QIcon:
    """An icon that draws one marker or glyph; a kind that is neither raises ValueError."""
    return QIcon(PaintedIcon(kind))
