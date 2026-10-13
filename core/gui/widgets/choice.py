"""The drop-down of the look: a stock combo box that paints its own chevron."""
from PySide6.QtCore import QRectF
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QComboBox

from .. import design
from ..paint import paint_marker
from ..reported import reported


class Choice(QComboBox):
    """A stock drop-down that paints its own chevron. With a border of the sheet's own the style is no longer
    asked for an arrow, and a sheet could only name an image file for one."""

    @reported()
    def paintEvent(self, event):
        super().paintEvent(event)                        # the box itself, as the toolkit draws it
        m = design.scaled(design.SIZE["marker"])
        inset = design.scaled(design.SIZE["combo_pad"][1])
        box = QRectF(self.width() - 1 - inset - m, (self.height() - m) / 2, m, m)   # 1: the outline, never scaled
        painter = QPainter(self)
        try:                                             # a raise below must not leave a live painter on the widget
            paint_marker(painter, "chev_down", box, not self.isEnabled())
        finally:
            painter.end()
