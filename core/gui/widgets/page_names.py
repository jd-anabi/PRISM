"""The row of a part's page names, above its page: one name for each page, the current one semibold in the
first ink with the accent's line under it, the others in the second ink, and a hairline under the row.

The row is one stop of the Tab key, held by the current name; Left and Right choose the name before or after it.
A name is chosen by a click too. Every name is painted here and measured semibold, the weight it takes when it is
current, so the names keep their places when another is chosen.
"""
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFontMetricsF, QPainter
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QWidget

from .. import design, fonts, keys, paint
from ..reported import reported


def _px(name: str) -> int:
    """A length of the look's size table at the text size in force."""
    return design.scaled(design.SIZE[name])


class _Name(QAbstractButton):
    """One page's name in the row: its words, and under the current one the accent's line, as wide as they."""

    def __init__(self, place: keys.Place):
        super().__init__()
        self.place = place
        self.current = False
        self.setText(place.name)
        self.setAccessibleName(place.name)
        self.setFocusPolicy(Qt.NoFocus)

    def _line(self) -> int:
        return design.scaled(design.TYPE["strong"][1])

    @reported(QAbstractButton.sizeHint)
    def sizeHint(self):
        wide = QFontMetricsF(fonts.inter("strong")).horizontalAdvance(self.place.name)
        return QSize(int(wide + 0.999), self._line() + design.scaled(design.GAP["head_lead"]) + design.SIZE["rule"])

    def keyPressEvent(self, event):
        row = self.parentWidget()
        if isinstance(row, PageNames) and row._key(self, event):
            return
        super().keyPressEvent(event)

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the name
            font = fonts.inter("strong" if self.current else "body")
            painter.setFont(font)
            painter.setPen(design.colour("ink" if self.current else "ink_2"))
            line, width = self._line(), self.width()
            painter.drawText(QRectF(0, 0, width, line), Qt.AlignLeft | Qt.AlignVCenter,
                             QFontMetricsF(font).elidedText(self.place.name, Qt.ElideRight, width))
            if self.current:
                weight = design.SIZE["rule"]
                under = QRectF(0, self.height() - weight, QFontMetricsF(font).horizontalAdvance(self.place.name),
                               weight)
                paint.fill_px(painter, under, design.colour("accent"), bottom=weight)
        finally:
            painter.end()


class PageNames(QWidget):
    """The names of a part's pages in a row, in the order handed in; the first is current at the start.
    ``chosen`` is emitted with a page's key when a person chooses it, by a click or with Left or Right;
    ``set_current`` marks one from the program and emits nothing."""
    chosen = Signal(str)

    def __init__(self, pages: tuple[keys.Place, ...]):
        super().__init__()
        if not pages:
            raise ValueError("a row of page names needs at least one page")
        self._names = []
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, design.scaled(design.SIZE["row_gap"]) + design.SIZE["hair"])
        row.setSpacing(design.scaled(design.GAP["switch"]))
        for place in pages:
            name = _Name(place)
            name.clicked.connect(self._pressed)
            row.addWidget(name)
            self._names.append(name)
        row.addStretch(1)
        self._current = None
        self.set_current(pages[0].key)

    def current(self) -> str:
        """The key of the current page."""
        return self._current.place.key

    def set_current(self, name: str) -> None:
        """Mark the page whose key is ``name`` as the current one; KeyError for a key the row does not hold."""
        for one in self._names:
            if one.place.key == name:
                break
        else:
            raise KeyError(name)
        for other in self._names:
            other.current = other is one
            other.setFocusPolicy(Qt.TabFocus if other is one else Qt.NoFocus)
            other.update()
        self._current = one

    def _choose(self, one) -> None:
        if one is not self._current:
            self.set_current(one.place.key)
            self.chosen.emit(one.place.key)

    def _pressed(self) -> None:
        self._choose(self.sender())

    def _key(self, one, event) -> bool:
        """A key pressed on one of the names: True when the row took it."""
        if event.key() not in (Qt.Key_Left, Qt.Key_Right):
            return False
        at = self._names.index(self._current)
        step = -1 if event.key() == Qt.Key_Left else 1
        target = self._names[max(0, min(len(self._names) - 1, at + step))]
        self._choose(target)
        target.setFocus(Qt.TabFocusReason)
        return True

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the row
            paint.fill_px(painter, QRectF(self.rect()), design.colour("rule_light"), bottom=design.SIZE["hair"])
        finally:
            painter.end()
