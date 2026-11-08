"""The drop-down of the look: a stock combo box that paints its own chevron."""
import math

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QFontMetricsF, QPainter, QPalette
from PySide6.QtWidgets import QComboBox, QStyle, QStyleOptionComboBox, QStylePainter

from .. import design
from ..paint import paint_marker
from ..reported import reported


class Choice(QComboBox):
    """A stock drop-down that paints its own chevron. With a border of the sheet's own the style is no longer
    asked for an arrow, and a sheet could only name an image file for one. It is as wide as its longest item, whenever
    its items change: left to the toolkit that is worked out once, as it is first shown, and an item added later is
    cut mid-letter where the chevron's box begins.

    ``most`` caps that width, in the look's pixels at the ordinary text size, for a list whose items a person names, as a
    file's or a record's name: the item shown is cut short in its middle, with the whole of it in the tip, and the list
    opens as wide as its longest item. A page never scrolls sideways for a long name. A form that has not the room for
    a list's longest item beside its label holds it to the room it has (``hold_to``), cut short the same way."""

    def __init__(self, parent=None, *, most: "int | None" = None) -> None:
        super().__init__(parent)
        self._most = most
        self._room: "int | None" = None          # the room a form holds the box to, in pixels; None for no hold
        self._widest = None                      # the widest advance of the items and the hint, with what it was for
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.currentIndexChanged.connect(self._said_whole)
        held = self.model()                      # an item added, taken away or given other words: measured again
        for said in (held.rowsInserted, held.rowsRemoved, held.dataChanged, held.modelReset, held.layoutChanged):
            said.connect(self._items_changed)

    def hold_to(self, room: "int | None") -> None:
        """Hold the box to ``room`` pixels, the item shown cut short in its middle where it needs more, the whole of it
        in the tip and in the open list; None lets it go, and with it the tip of a box no longer cut, which would name
        the item shown at the last choice made while it was held."""
        if room == self._room:
            return
        self._room = room
        if self._cut():
            self.say_whole()
        else:
            self.setToolTip("")
        self.updateGeometry()
        self.update()

    def _cut(self) -> bool:
        """Whether the box may be narrower than its longest item: capped, or held by its form."""
        return self._most is not None or self._room is not None

    def _said_whole(self, _index: int = 0) -> None:
        if self._cut():
            self.say_whole()

    def _whole_width(self) -> int:
        """As wide as the longest item, or the hint, with the chevron's room: what the toolkit gives with no cap, and
        wider by what the toolkit's measure of an item falls short of the room it is drawn in. On Windows, at some text
        sizes, the toolkit measures an item a pixel narrower than it draws it, and a box of that width cut "No plan" to
        "No…an" at one and a half times the ordinary text."""
        given, chrome = self._chrome()
        return max(given, chrome + self._widest_advance())

    def _widest_advance(self) -> int:
        """The widest advance of the items and the hint, kept until the items, the hint or the font change: a list of
        hundreds of records is measured once, not at every pass of a layout."""
        key = (self.font().key(), self.placeholderText())
        if self._widest is None or self._widest[0] != key:
            texts = [self.itemText(at) for at in range(self.count())] + [self.placeholderText()]
            self._widest = (key, max((self._advance(text) for text in texts if text), default=0))
        return self._widest[1]

    def _items_changed(self, *_args) -> None:
        self._widest = None

    def shown_width(self) -> int:
        """As wide as the box must be to show the item chosen, or the hint while none is, whole: the item as it is
        drawn, which a list that speaks of its items by other names than it shows does not give as its current text."""
        shown = self.placeholderText() if self._hinting() else self.itemText(self.currentIndex())
        return self._chrome()[1] + self._advance(shown)

    def _chrome(self) -> tuple[int, int]:
        """The toolkit's own width of the box, and the room the box takes beside its words: its padding and the
        chevron's room."""
        given = super().sizeHint().width()
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.rect = QRect(0, 0, given, max(self.height(), 1))
        room = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                           QStyle.SubControl.SC_ComboBoxEditField, self).width()
        return given, given - room

    def _advance(self, text: str) -> int:
        """How wide ``text`` is drawn, to the pixel above."""
        exact = math.ceil(QFontMetricsF(self.font()).horizontalAdvance(text))
        return max(self.fontMetrics().horizontalAdvance(text), exact)

    @reported(QComboBox.sizeHint)
    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(self._whole_width())
        if self._most is not None:
            hint.setWidth(min(hint.width(), design.scaled(self._most)))
        if self._room is not None:
            hint.setWidth(min(hint.width(), self._room))
        return hint

    @reported(QComboBox.minimumSizeHint)
    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(self.sizeHint().width())               # the toolkit's own holds no item
        return hint

    def say_whole(self, _index: int = 0) -> None:
        """The item shown, whole, in the tip: the box may show it cut short."""
        self.setToolTip(self.currentText())

    def showPopup(self):
        if self._cut():
            self.view().setMinimumWidth(self._whole_width())  # the open list shows every name whole
        super().showPopup()

    def _hinting(self) -> bool:
        """Whether the box shows its hint: no item chosen, and a hint to show."""
        return self.currentIndex() < 0 and bool(self.placeholderText())

    @reported()
    def paintEvent(self, event):
        if not self._cut() and not self._hinting():
            super().paintEvent(event)                    # the box itself, as the toolkit draws it
        else:
            self._paint_cut_short()
        m = design.scaled(design.SIZE["marker"])
        inset = design.scaled(design.SIZE["combo_pad"][1])
        box = QRectF(self.width() - 1 - inset - m, (self.height() - m) / 2, m, m)   # 1: the outline, never scaled
        painter = QPainter(self)
        try:                                             # a raise below must not leave a live painter on the widget
            paint_marker(painter, "chev_down", box, not self.isEnabled())
        finally:
            painter.end()

    def shown_text(self) -> str:
        """The item shown, or the hint while none is, as the box draws it: cut short in its middle to the room the box
        gives it."""
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        text = self.placeholderText() if self.currentIndex() < 0 and self.placeholderText() else option.currentText
        room = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                           QStyle.SubControl.SC_ComboBoxEditField, self)
        return self.fontMetrics().elidedText(text, Qt.ElideMiddle, room.width())

    def _paint_cut_short(self) -> None:
        """The box as the toolkit draws it, the item shown cut short in its middle to the room it has. The hint is drawn
        in the hint's colour of the palette, greyed with the box: the style sheet's colour of a box's text would win
        over the palette the toolkit's own label is drawn with, and the hint would read as a name chosen."""
        painter = QStylePainter(self)
        try:
            option = QStyleOptionComboBox()
            self.initStyleOption(option)
            painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
            option.currentText = self.shown_text()
            if not self._hinting():
                painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
                return
            room = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                               QStyle.SubControl.SC_ComboBoxEditField, self)
            group = QPalette.ColorGroup.Active if self.isEnabled() else QPalette.ColorGroup.Disabled
            painter.setPen(self.palette().color(group, QPalette.ColorRole.PlaceholderText))
            painter.drawText(room, int(Qt.AlignLeft | Qt.AlignVCenter), option.currentText)
        finally:
            painter.end()
