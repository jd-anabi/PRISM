"""Painters of the look's small marks: pure functions of a painter. Each reads the look's table of colours
when it paints and keeps nothing; a style sheet could only name an image file for these."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen

from . import design


def paint_indicator(painter, kind: str, box: QRectF, *, on: bool, enabled: bool) -> None:
    """The tick box (``kind`` "check") or the radio dot ("radio") inside ``box``, drawn in an 18-unit square
    that is scaled to the box's width. Empty, it is the field's ground inside the outline of a control; set, the
    accent with its mark; disabled, it keeps its mark and loses the accent. Any other kind paints nothing."""
    if kind not in ("check", "radio"):
        return
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    unit = box.width() / 18.0
    line = design.colour("field_line" if enabled else "disabled_line")
    ground = design.colour("field" if enabled else "disabled_fill")
    mark = design.colour("on_accent" if enabled else "disabled_ink")
    filled = on and enabled
    painter.setPen(Qt.NoPen if filled else QPen(line, 1.0))
    painter.setBrush(design.colour("accent") if filled else ground)
    body = box if filled else box.adjusted(0.5, 0.5, -0.5, -0.5)     # an outline lies whole inside the box
    if kind == "radio":
        painter.drawEllipse(body)
        if on:
            painter.setPen(Qt.NoPen)
            painter.setBrush(mark)
            painter.drawEllipse(box.center(), 4 * unit, 4 * unit)
    else:
        corner = (3.0 if filled else 2.5) * unit
        painter.drawRoundedRect(body, corner, corner)
        if on:
            tick = QPainterPath(box.topLeft() + QPointF(4.975 * unit, 9.35 * unit))
            tick.lineTo(box.topLeft() + QPointF(7.6 * unit, 11.975 * unit))
            tick.lineTo(box.topLeft() + QPointF(13.025 * unit, 6.025 * unit))
            painter.setPen(QPen(mark, 1.75 * unit, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(tick)
    painter.restore()
