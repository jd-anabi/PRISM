"""A list and the detail of one of its items: the list alone while no item is open; the item's detail beside the list
while the page is wide enough for both, the list then told to show less of itself; and on a narrower page the detail in
the list's place, with a plain link above it that goes back to the whole list.

The switch reads the page's width, as the page's body is told it, and nothing else: the low window is a rule on the
window's height and says nothing of the room beside a list. Beside, neither part is ever narrower than it needs to be:
the detail stands beside the list only where the width holds the width the list asks for, the gap and the detail's least
width as it stands in the room left, and from the width the caller names, which it scales. Beside, each stands at the
top of its column, as tall as it is, whatever the other holds.
"""
from PySide6.QtCore import QCoreApplication, QEvent, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .. import design, style

BACK_WORDS = "All checks"


def _least(part: QWidget) -> int:
    """The least width ``part`` stands at: what its layout asks, or what it was given, whichever is wider."""
    return max(part.minimumSizeHint().width(), part.minimumWidth(), 0)


def _least_as_told(part: QWidget) -> int:
    """The least width ``part`` stands at as it was last told its width. A part told a width lays itself out again only
    when the layouts it asked for come round, and until then each layout above reads its parts' least widths as they
    were: the layouts inside ``part`` are laid out first, the deepest first, each telling the one above it."""
    for widget in [*reversed(part.findChildren(QWidget)), part]:
        QCoreApplication.sendPostedEvents(widget, QEvent.LayoutRequest)
    return _least(part)


def _wanted(part: QWidget) -> int:
    """The width ``part`` asks to stand at, never less than its least."""
    return max(part.sizeHint().width(), _least(part))


def _stops_of(part: QWidget) -> list:
    """What the Tab key stops at in ``part``: where it names its stops, those, in its own order, a stop that is one only
    at times among them, which has its place all the same; else the part, or what it holds, that is a stop now."""
    own = getattr(part, "stops", None)
    if callable(own):
        return list(own())
    found = [part] if part.focusPolicy() & Qt.TabFocus else part.findChildren(QWidget)
    return [stop for stop in found if stop.focusPolicy() & Qt.TabFocus]


class ListAndDetail(QWidget):
    """``listing`` and the detail ``show_detail`` hands it. The listing answers ``set_narrow(on)``: told True while a detail
    is open, it shows less of itself; and it may answer ``take_the_keyboard(detail)``, putting the keyboard on the row
    of ``detail`` and saying True, which "All checks" asks when it held the keyboard, else the list's first stop takes
    it. ``back_pressed`` is said when the link above a detail in the list's place is pressed, once the whole list
    stands again. ``switched`` is said when the detail moves between beside the list and its place, which changes what
    the Tab key stops at."""

    back_pressed = Signal()
    switched = Signal()

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
        self._row.addWidget(listing, 1, Qt.AlignTop)
        box.addLayout(self._row)
        self._listing_widths = (listing.minimumWidth(), listing.maximumWidth())   # its own, put back in place
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
                self._row.addWidget(detail, 1, Qt.AlignTop)
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
        is held at the width it asks for, told narrow, and the detail takes the rest, which it is told, as it is told the
        whole width in the list's place. The detail's least width is read as it would stand beside, told that room
        first, so that one width gives one arrangement, however the window came to it and whatever the detail was told
        before."""
        was = self.beside()
        self._arrange_now()
        if self.detail is not None and self.beside() != was:
            self.switched.emit()

    def _arrange_now(self) -> None:
        detail = self.detail
        self._hold_the_list(None)
        if detail is None:
            self._beside = False
            self.back.setVisible(False)
            self.listing.setVisible(True)
            return
        listing = _wanted(self.listing)
        told = getattr(detail, "fit", None)
        room = None if self._width is None else self._width - listing - self._gap
        detail.setVisible(True)
        if callable(told) and room is not None:
            told(room)
        self._beside = room is not None and self._width >= self.beside_width and room >= _least_as_told(detail)
        self.back.setVisible(not self._beside)
        self._hold_the_list(listing if self._beside else None)
        self.listing.setVisible(self._beside)
        if callable(told) and room is not None and not self._beside:
            told(self._width)

    def _hold_the_list(self, width: "int | None") -> None:
        """Hold the list at ``width``, or let it have its own least and most widths again with None."""
        if width is None:
            self.listing.setMinimumWidth(self._listing_widths[0])
            self.listing.setMaximumWidth(self._listing_widths[1])
        else:
            self.listing.setFixedWidth(width)

    def _back_pressed(self) -> None:
        """"All checks": the whole list again, and the keyboard, when the link held it, back on the list: on the row of
        the detail just left where the list names one, else at its first stop."""
        held = self.window().focusWidget() is self.back
        left = self.detail
        self.show_detail(None)
        self.back_pressed.emit()
        if not held:
            return
        take = getattr(self.listing, "take_the_keyboard", None)
        if callable(take) and take(left):
            return
        stops = _stops_of(self.listing)
        if stops:
            stops[0].setFocus(Qt.TabFocusReason)

    def stops(self) -> list:
        """What the Tab key stops at, in the order it is read: the link and the detail in the list's place; else the
        list, then the detail beside it."""
        if self.detail is None:
            return _stops_of(self.listing)
        if not self._beside:
            return [self.back, *_stops_of(self.detail)]
        return [*_stops_of(self.listing), *_stops_of(self.detail)]
