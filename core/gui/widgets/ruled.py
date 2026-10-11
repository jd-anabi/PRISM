"""A block with a ruled line along its top edge, its bottom edge or both, and the section heading built on it.

The block paints its own rules, from its own edges and on whole device pixels: a rule then has one weight at
every display scaling and at every position, and nothing the block holds covers part of it. A rule that is a
widget one or two pixels high has neither. The style sheet gives the block no border. The block keeps the
name of its rules' colour and reads the colour from the look's table when it paints, so it follows a change
of theme with the repaint alone.
"""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .. import design, fonts, paint, style
from ..reported import reported


class Ruled(QWidget):
    """A block with a rule ``top`` logical pixels thick along its top edge and one ``bottom`` thick along its
    bottom edge (0: none), in the colour of the look that ``colour`` names. What the block holds goes into
    ``box``, which keeps clear of each rule and of ``gap_top`` and ``gap_bottom`` beside it. A rule's weight
    does not grow with the text; a gap is added as it is handed in, so a caller scales it."""

    def __init__(self, top: int = 0, bottom: int = 0, colour: str = "rule", *, gap_top: int = 0,
                 gap_bottom: int = 0):
        super().__init__()
        if colour not in design.COLOURS:     # named when the block is built, not found out when it paints
            raise ValueError(f"not a colour of the look: {colour!r}")
        self._top, self._bottom, self._colour = top, bottom, colour
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, top + gap_top, 0, bottom + gap_bottom)
        self.box.setSpacing(0)

    @reported()
    def paintEvent(self, _event):
        if not (self._top or self._bottom):
            return
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the block
            paint.fill_px(painter, QRectF(self.rect()), design.colour(self._colour), top=self._top, bottom=self._bottom)
        finally:
            painter.end()


class SectionHead(Ruled):
    """A section's heading: its words at the heading's size and weight over one heavy rule. ``label`` holds
    the words; the gap between them and the rule, and the height of their line, grow with the text."""

    def __init__(self, text: str):
        super().__init__(bottom=design.SIZE["rule"], gap_bottom=design.scaled(design.GAP["section_rule"]))
        self.label = QLabel(text)
        self.label.setTextFormat(Qt.PlainText)               # a name may hold "<" or "&"
        style.set_class(self.label, cls="head")              # what the sheet's rule for a heading matches
        # The font is given here, when the label is built, and nowhere later: where no sheet is applied it
        # gives the size and the weight the sheet's rule would, and a font handed to a widget after a sheet
        # has dressed it is dropped at the next change of sheet.
        self.label.setFont(fonts.inter("head"))
        self.label.setFixedHeight(design.scaled(design.TYPE["head"][1]))
        self._row = QHBoxLayout()
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.addWidget(self.label)
        self._row.addStretch(1)
        self.box.addLayout(self._row)

    def set_text(self, text: str) -> None:
        """Change the heading's words."""
        self.label.setText(text)

    def add_right(self, widget: QWidget) -> None:
        """Put ``widget`` at the right end of the heading's line, standing on the line's foot."""
        self._row.addWidget(widget, 0, Qt.AlignBottom)
