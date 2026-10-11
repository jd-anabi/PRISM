"""Painters of the look's small marks: pure functions of a painter, which keep nothing. The tick box, the
radio dot, the markers and the glyphs read the look's table of colours when they paint; a rule and the
hatching are handed their colour. A style sheet could only name an image file for these."""
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen

from . import design


def paint_indicator(painter, kind: str, box: QRectF, *, on: bool, enabled: bool) -> None:
    """The tick box (``kind`` "check") or the radio dot ("radio") inside ``box``, drawn in an 18-unit square
    that is scaled to the box's width. Empty, it is the field's ground inside the outline of a control; set, the
    accent with its mark; disabled, it keeps its mark and loses the accent. Any other kind paints nothing."""
    if kind not in ("check", "radio"):
        return
    painter.save()
    try:                                     # a raise below must not leave the caller's painter one state deeper
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
    finally:
        painter.restore()


MARKERS = ("done", "now", "later", "lock", "half", "none", "fine", "info", "warn", "err")    # one shape to a state
GLYPHS = ("chev_right", "chev_down", "chev_up", "help", "stop")                              # the glyphs of controls
_ON_A_SHAPE = ("paper", "on_err", "warn_mark")      # the marks that stand on a filled shape


def _shade(name: str, disabled: bool) -> QColor:
    """A colour of the look for one part of a marker. Drawn disabled, a shape takes the colour of
    disabled text and a mark that stands on it stays the page's colour, so a tick, a cross or a half
    never vanishes into its shape."""
    if not disabled:
        return design.colour(name)
    return design.colour("paper" if name in _ON_A_SHAPE else "disabled_ink")


def paint_marker(painter, kind: str, box: QRectF, disabled: bool = False) -> None:
    """One of the ten state markers or the five glyphs of controls, drawn in a 16-unit square that is scaled
    to ``box``: one shape to a state, so that a state is told without its colour. Every colour is read from
    the look's table when the function paints. Drawn disabled, a shape is the colour of disabled text and
    the mark that stands on it keeps the page's colour. A kind that is neither a marker nor a glyph paints
    nothing and raises nothing: a kind is checked where a device is built, not at every repaint. The painter
    is left as it was found, also when a colour fails part-way."""
    painter.save()
    try:                                     # a raise below must not leave the caller's painter moved and scaled
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.translate(box.topLeft())
        painter.scale(box.width() / 16.0, box.height() / 16.0)

        def pen(name, width=2.0, cap=Qt.RoundCap):
            made = QPen(_shade(name, disabled), width)
            made.setCapStyle(cap)
            made.setJoinStyle(Qt.RoundJoin)
            return made

        def line(*points):
            path = QPainterPath(QPointF(*points[0]))
            for point in points[1:]:
                path.lineTo(QPointF(*point))
            return path

        def stroke(path, name, width=2.0, cap=Qt.RoundCap):
            painter.setPen(pen(name, width, cap))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

        def disc(cx, cy, r, fill=None, edge=None, width=2.0):
            painter.setPen(pen(edge, width) if edge else Qt.NoPen)
            painter.setBrush(_shade(fill, disabled) if fill else Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r, r)

        if kind == "done":                          # a filled disc with a tick on it
            disc(8, 8, 8, "ink")
            stroke(line((4.5, 8.4), (6.9, 10.8), (11.6, 5.6)), "paper")
        elif kind == "now":                         # a dot in the accent
            disc(8, 8, 6, "accent")
        elif kind == "later":                       # an open, broken circle
            dashed = pen("ink_3", 2.0, Qt.FlatCap)
            dashed.setDashPattern([1.04, 1.0])      # a dash and its gap, in widths of the pen
            painter.setPen(dashed)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(8, 8), 6.5, 6.5)
        elif kind == "half":                        # not finished: a disc filled on its left half
            disc(8, 8, 7, "paper", "ink")
            half = QPainterPath(QPointF(8, 1.5))
            half.arcTo(QRectF(1.5, 1.5, 13, 13), 90, 180)
            half.closeSubpath()
            painter.setPen(Qt.NoPen)
            painter.setBrush(_shade("ink", disabled))
            painter.drawPath(half)
        elif kind == "none":                        # nothing running: a dash
            stroke(line((4, 8), (12, 8)), "ink_3")
        elif kind == "fine":                        # a ring with a tick in it
            disc(8, 8, 7, None, "ok")
            stroke(line((4.9, 8.3), (7.1, 10.5), (11.1, 5.9)), "ok")
        elif kind == "info":                        # a ring with the letter i in it
            disc(8, 8, 7, None, "ink_2")
            disc(8, 4.9, 1.1, "ink_2")
            stroke(line((8, 7.5), (8, 11.7)), "ink_2")
        elif kind == "warn":                        # a triangle: on white its edge gives the shape, not its amber
            triangle = line((8, 1.6), (15, 14.3), (1, 14.3))
            triangle.closeSubpath()
            painter.setPen(pen("warn_edge", 1.5))
            painter.setBrush(_shade("warn", disabled))
            painter.drawPath(triangle)
            stroke(line((8, 6.2), (8, 10.0)), "warn_mark")
            disc(8, 12.1, 1.0, "warn_mark")
        elif kind == "err":                         # a filled square with a cross on it
            painter.setPen(Qt.NoPen)
            painter.setBrush(_shade("err", disabled))
            painter.drawRoundedRect(QRectF(0, 0, 16, 16), 2, 2)
            stroke(line((5, 5), (11, 11)), "on_err")
            stroke(line((11, 5), (5, 11)), "on_err")
        elif kind == "lock":                        # blocked: a padlock
            painter.setPen(Qt.NoPen)
            painter.setBrush(_shade("ink_2", disabled))
            painter.drawRoundedRect(QRectF(3, 7, 10, 8), 1.5, 1.5)
            bow = QPainterPath(QPointF(5, 7.5))
            bow.lineTo(5, 5)
            bow.arcTo(QRectF(5, 2, 6, 6), 180, -180)
            bow.lineTo(11, 7.5)
            stroke(bow, "ink_2", 2.0, Qt.FlatCap)
        elif kind == "chev_right":                  # from here on the glyphs of controls
            stroke(line((6, 3.5), (10.5, 8), (6, 12.5)), "ink")
        elif kind == "chev_down":
            stroke(line((3.5, 6), (8, 10.5), (12.5, 6)), "ink")
        elif kind == "chev_up":
            stroke(line((3.5, 10), (8, 5.5), (12.5, 10)), "ink")
        elif kind == "help":                        # a ring with a question mark in it
            disc(8, 8, 7, None, "ink")
            hook = QPainterPath(QPointF(6, 6.3))
            hook.cubicTo(6.0, 4.2, 9.6, 3.9, 9.9, 6.2)
            hook.cubicTo(10.0, 7.6, 8.0, 7.7, 8.0, 9.5)
            stroke(hook, "ink")
            disc(8, 12, 1.0, "ink")
        elif kind == "stop":                        # the small square on an action that loses work
            painter.setPen(Qt.NoPen)
            painter.setBrush(_shade("err", disabled))
            painter.drawRect(QRectF(3, 3, 10, 10))
    finally:
        painter.restore()


def fill_px(painter, rect: QRectF, colour: QColor, *, top: int = 0, bottom: int = 0,
            left: int = 0, right: int = 0) -> None:
    """Rule the named edges of ``rect``: ``top=2`` is a rule 2 logical pixels thick along its top edge.
    A rule's thickness is a whole number of device pixels, worked out once, rounded down as a browser
    rounds a border and never under one; it is laid from the edge it is named for. So it has one weight
    wherever it falls, stays inside its rectangle, and no widget beside it covers part of it. Left to
    the painter, a 1 px line at 150 % is 1 or 2 pixels by position. No edge named, nothing painted. The
    fill is made with the painter's smoothing off, and the smoothing is left as it was found, also when the
    fill fails: filled under a painter that smooths, a rule can hold pixels one level off its colour."""
    to_device = painter.deviceTransform()
    to_logical, invertible = to_device.inverted()
    if not invertible or rect.width() <= 0 or rect.height() <= 0:
        return
    near, far = to_device.map(rect.topLeft()), to_device.map(rect.bottomRight())
    x0, x1 = math.floor(near.x() + 0.5), math.floor(far.x() + 0.5)
    y0, y1 = math.floor(near.y() + 0.5), math.floor(far.y() + 0.5)

    def whole(pixels, per_pixel):
        return max(1, math.floor(pixels * abs(per_pixel) + 1e-6))

    bands = []
    if top:
        bands.append((x0, y0, x1, y0 + whole(top, to_device.m22())))
    if bottom:
        bands.append((x0, y1 - whole(bottom, to_device.m22()), x1, y1))
    if left:
        bands.append((x0, y0, x0 + whole(left, to_device.m11()), y1))
    if right:
        bands.append((x1 - whole(right, to_device.m11()), y0, x1, y1))
    smoothing = painter.testRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.Antialiasing, False)
    try:                                     # a raise below must not leave the painter's smoothing off
        for a, b, c, d in bands:
            painter.fillRect(QRectF(to_logical.map(QPointF(a, b)), to_logical.map(QPointF(c, d))), colour)
    finally:
        painter.setRenderHint(QPainter.Antialiasing, smoothing)


def paint_hatch(painter, rect: QRectF, colour: QColor) -> None:
    """Slanting strokes 1 px thick inside ``rect`` and nowhere else, in the colour handed in: one at every
    hatch step of the look's table, measured across the strokes. A rectangle with no width or no height is
    left alone. The painter is left as it was found, also when the drawing fails part-way."""
    if rect.width() <= 0 or rect.height() <= 0:
        return
    painter.save()
    try:                                     # a raise below must not leave the caller's painter clipped
        painter.setClipRect(rect)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(colour, 1.0))
        step, h = design.SIZE["hatch_step"] * math.sqrt(2.0), rect.height()
        # The strokes are counted from the painter's origin, not from the rectangle: they stay where they are
        # when the rectangle moves, and those of two rectangles of one top and one height fall on each other.
        x = rect.left() - h - (rect.left() % step)
        while x < rect.right() + h:
            painter.drawLine(QPointF(x, rect.bottom()), QPointF(x + h, rect.top()))
            x += step
    finally:
        painter.restore()
