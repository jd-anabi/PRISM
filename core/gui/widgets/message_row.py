"""A message row: a marker and one sentence in the book face, its opening words bold."""
from datetime import datetime

from PySide6.QtCore import QRectF
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .. import design, paint
from ..reported import reported
from .marker import Marker
from .page import Sentence


def opening(lead: str | None, when: datetime | None, now: datetime | None = None) -> str:
    """The bold opening of a message: its lead word, its date when it has one, and a full stop."""
    word = (lead or "").strip().removesuffix(".").rstrip()   # the one full stop that closes the opening is set below
    said = [word] if word else []
    if when is not None:
        said.append(design.date_text(when, now))
    return ", ".join(said) + "." if said else ""


class MessageRow(QWidget):
    """A marker and one sentence in the book face whose opening words are bold; a row with no marker keeps the
    marker's room. An error has an edge at its left."""

    def __init__(self, kind: str, sentence: str, *, lead: str | None = None,
                 when: datetime | None = None, now: datetime | None = None) -> None:
        if kind and kind not in paint.MARKERS:       # refused before anything is built; a control's glyph is no marker
            raise ValueError(f"{kind!r} is not a marker")
        super().__init__()
        self.kind = kind
        row = QHBoxLayout(self)
        row.setContentsMargins(*(design.scaled(n) for n in design.GAP["message"]))
        row.setSpacing(design.scaled(design.GAP["message_marker"]))
        self.marker = Marker(kind) if kind else None
        if self.marker is not None:
            column = QVBoxLayout()                   # a column of its own: the marker stays beside the first line
            column.setContentsMargins(0, design.scaled(design.GAP["message_marker_drop"]), 0, 0)
            column.addWidget(self.marker)
            column.addStretch(1)
            row.addLayout(column)
        else:
            # the marker's room and the gap after it, so that the words start where a marked row's do; a layout
            # puts none of its spacing beside a space, so the space holds the gap as well
            row.addSpacing(design.scaled(design.SIZE["marker"]) + design.scaled(design.GAP["message_marker"]))
        self.sentence = Sentence()
        self.sentence.set_text(sentence, lead=opening(lead, when, now) or None)
        row.addWidget(self.sentence, 1)

    @reported()
    def paintEvent(self, _event):
        if self.kind != "err":
            return                           # a row has no ground of its own: nothing else is painted
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the widget
            paint.fill_px(painter, QRectF(self.rect()), design.colour("err"), left=design.SIZE["rule"])
        finally:
            painter.end()
