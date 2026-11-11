"""A list and the detail of one of its items: the list alone while no item is open; the item's detail beside the list
while the page is wide enough for both, the list then told to show less of itself; and on a narrower page the detail in
the list's place, with a plain link above it that goes back to the whole list.

The switch reads the page's width, as the page's body is told it, and nothing else: the low window is a rule on the
window's height and says nothing of the room beside a list. Beside, neither part is ever narrower than it needs to be:
the detail stands beside the list only where the width holds the list's least width, the gap and the detail's least
width, and from the width the caller names, which it scales.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .. import design, style

BACK_WORDS = "All checks"


def _least(part: QWidget) -> int:
    """The least width ``part`` stands at: what its layout asks, or what it was given, whichever is wider."""
    return max(part.minimumSizeHint().width(), part.minimumWidth(), 0)


def _wanted(part: QWidget) -> int:
    """The width ``part`` asks to stand at, never less than its least."""
    return max(part.sizeHint().width(), _least(part))


def _stops_of(part: QWidget) -> list:
    """What the Tab key stops at in ``part``, in its own order where it names one."""
    own = getattr(part, "stops", None)
    found = own() if callable(own) else ([part] if part.focusPolicy() & Qt.TabFocus else part.findChildren(QWidget))
    return [stop for stop in found if stop.focusPolicy() & Qt.TabFocus]


class ListAndDetail(QWidget):
    """``listing`` and the detail ``show_detail`` hands it. The listing answers ``set_narrow(on)``: told True while a detail
    is open, it shows less of itself. ``back_pressed`` is said when the link above a detail in the list's place is
    pressed, once the whole list stands again."""

    back_pressed = Signal()

    def __init__(self, listing: QWidget, *, beside_width: int, back_words: str = BACK_WORDS) -> None:
        super().__init__()
        self.listing = listing
        self.beside_width = beside_width
        self.detail: "QWidget | None" = None
        self._width: "int | None" = None     # the width the page last told it; none before it first does
        self._beside = False
        self._gap = design.scaled(design.SIZE["col_gap"])
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(design.scaled(design.GAP["head_table"]))
        self.back = style.set_class(QPushButton(back_words), cls="link")
        self.back.setCursor(Qt.PointingHandCursor)
        self.back.setFixedHeight(design.scaled(design.SIZE["line_h"]))
        self.back.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.back.setVisible(False)
        self.back.clicked.connect(self._back_pressed)
        box.addWidget(self.back, 0, Qt.AlignLeft)
        self._row = QHBoxLayout()
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(self._gap)
        self._row.addWidget(listing, 1)
        box.addLayout(self._row)
        listing.set_narrow(False)

    def show_detail(self, detail: "QWidget | None") -> None:
        """Open ``detail``, the one in its place before going, or None for the whole list again. The detail stays the
        caller's: one put away is hidden, never deleted."""
        if detail is not self.detail:
            if self.detail is not None:
                self._row.removeWidget(self.detail)
                self.detail.hide()
            self.detail = detail
            if detail is not None:
                self._row.addWidget(detail, 1)
        self.listing.set_narrow(detail is not None)
        self._arrange()

    def fit(self, width: int) -> None:
        """The page's body is ``width`` wide: the detail beside the list, or in its place, as that width holds."""
        self._width = width
        self._arrange()

    def beside(self) -> bool:
        """Whether a detail stands beside the list."""
        return self.detail is not None and self._beside

    def _arrange(self) -> None:
        """Beside where the width holds both, else in the list's place; the whole list with no detail. Beside, the list
        stands at the width it asks for, told narrow, and the detail takes the rest, which it is told, as it is told the
        whole width in the list's place."""
        detail = self.detail
        if detail is None:
            self._beside = False
            self.back.setVisible(False)
            self.listing.setVisible(True)
            self._row.setStretchFactor(self.listing, 1)
            return
        listing = _wanted(self.listing)
        need = listing + self._gap + _least(detail)
        self._beside = self._width is not None and self._width >= max(self.beside_width, need)
        self.back.setVisible(not self._beside)
        self.listing.setVisible(self._beside)
        self._row.setStretchFactor(self.listing, 0 if self._beside else 1)
        detail.setVisible(True)
        told = getattr(detail, "fit", None)
        if callable(told) and self._width is not None:
            told(self._width - listing - self._gap if self._beside else self._width)

    def _back_pressed(self) -> None:
        self.show_detail(None)
        self.back_pressed.emit()

    def stops(self) -> list:
        """What the Tab key stops at, in the order it is read: the link and the detail in the list's place; else the
        list, then the detail beside it."""
        if self.detail is None:
            return _stops_of(self.listing)
        if not self._beside:
            return [self.back, *_stops_of(self.detail)]
        return [*_stops_of(self.listing), *_stops_of(self.detail)]
