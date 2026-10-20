"""The window's two regions: the rail down its left edge and, beside it, the page of the part in sight; and the page
a part shows in its place when it could not be built.

The rail is the way between the parts; the page is the part's own top widget, one of a stack. The shell holds
the parts by their keys and shows one at a time, at once, with no animation. When the window is lower than the
look's threshold it tells the rail and every part that offers ``set_small`` to take the small layout, and tells
them again when the window grows past it. Which part to show, and when, is the main window's to decide.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QPlainTextEdit, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout,
                               QWidget)

from .. import design, fonts, style
from ..reported import reported
from ..widgets.ask import DETAIL_LINES, FULL_ERROR
from ..widgets.message_row import MessageRow
from ..widgets.page import Page, PageHead
from ..widgets.rail import Rail


def clock_words(seconds: int) -> str:
    """How long a run has gone on, as the foot of the rail writes it: minutes and seconds, "4:07", and from an
    hour the hours as well, "1:02:05". A time below nothing, as a jump of the computer's clock can give, reads
    as nothing."""
    hours, rest = divmod(max(0, int(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _as_sentence(message: str) -> str:
    """An error's message as a sentence of the page: its first letter a capital when it is a small one, and a full
    stop at its end when it ends without a mark that ends a sentence. An empty message stays empty."""
    text = message.rstrip()
    if not text:
        return ""
    if text[0].islower():
        text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


class Shell(QWidget):
    """The rail and, beside it, a stack of the parts, with no margins and no space between them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parts: dict = {}
        self._small = False
        self.rail = Rail()
        self.stack = QStackedWidget()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.rail)
        row.addWidget(self.stack, 1)

    def add_part(self, name: str, widget: QWidget) -> None:
        """Hold ``widget`` as the part ``name``, unseen until it is shown. A part added while the window is low
        is told so at once. A name held already raises ValueError."""
        if name in self._parts:
            raise ValueError(f"a part of that name is held already: {name!r}")
        self._parts[name] = widget
        self.stack.addWidget(widget)
        self._tell(widget)

    def replace_part(self, name: str, widget: QWidget) -> QWidget:
        """Hold ``widget`` as the part ``name`` in place of the one held, at its place in the stack and in sight
        when that one was, told so at once when the window is low; the one held is handed back, out of the stack,
        for the caller to let go. KeyError for a part not held."""
        held = self._parts[name]
        shown = self.stack.currentWidget() is held
        self.stack.insertWidget(self.stack.indexOf(held), widget)
        self._parts[name] = widget
        if shown:
            self.stack.setCurrentWidget(widget)
        self.stack.removeWidget(held)
        self._tell(widget)
        return held

    def _tell(self, widget: QWidget) -> None:
        """Tell a part just handed over the small layout, when the window is low."""
        told = getattr(widget, "set_small", None)
        if self._small and callable(told):
            told(True)

    def show_part(self, name: str) -> None:
        """Show the part ``name`` in place of the one in sight; KeyError for a part not held."""
        self.stack.setCurrentWidget(self._parts[name])

    def has_part(self, name: str) -> bool:
        return name in self._parts

    def part_widget(self, name: str) -> QWidget | None:
        """The part held as ``name``, or None."""
        return self._parts.get(name)

    def set_small(self, on: bool) -> None:
        """The small layout, or the usual one again: told to the rail and to every part that offers it."""
        self._small = bool(on)
        self.rail.set_small(self._small)
        for widget in self._parts.values():
            told = getattr(widget, "set_small", None)
            if callable(told):
                told(self._small)

    @reported(QWidget.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        small = self.height() < design.small_below()
        if small != self._small:
            self.set_small(small)


class PartError(QWidget):
    """What stands in a part's place when the part could not be built, in the look of a page: the part's name, the
    Error marker beside "<Part> could not be opened." and the error's message, "The rest of PRISM still works.",
    and two buttons. "Try again" asks for the part to be built again; "Show the full error" opens, under them, the
    error's kind and its trace in a read-only box in the fixed-width face, the one place the error's kind is shown.
    The page keeps the error's words, never the error itself, whose trace would hold what was half built."""

    try_again = Signal()

    def __init__(self, part_name: str, error: BaseException, trace: str, parent=None) -> None:
        super().__init__(parent)
        self.page = Page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.page)
        self.page.add_fixed(PageHead(part_name))
        self.page.add_gap(design.scaled(design.GAP["head_to_body"]), design.scaled(design.GAP["head_to_body_small"]))

        self.message = MessageRow("err", _as_sentence(str(error)), lead=f"{part_name} could not be opened")
        self.rest = MessageRow("", "The rest of PRISM still works.")
        self.again = style.set_class(QPushButton("Try again"), cls="primary")
        self.again.clicked.connect(self.try_again)
        self.link = style.set_class(QPushButton(FULL_ERROR), cls="link")
        self.link.setCursor(Qt.PointingHandCursor)
        self.link.setFixedHeight(design.scaled(design.SIZE["line_h"]))
        self.link.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.link.clicked.connect(self.show_full_error)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(design.scaled(design.SIZE["table_col_gap"]))
        buttons.addWidget(self.again)
        buttons.addWidget(self.link, 0, Qt.AlignVCenter)
        buttons.addStretch(1)

        self.full_error = QPlainTextEdit()
        self.full_error.setReadOnly(True)
        self.full_error.setLineWrapMode(QPlainTextEdit.NoWrap)
        # The font is given here, at build, and never when the link is pressed: a font handed to a widget after a
        # sheet has dressed it is dropped at the next change of sheet.
        self.full_error.setFont(fonts.fixed())
        self.full_error.setPlainText(trace)
        # In a plain holder of the fixed height: the sheet gives a text box a least height of its own, which
        # replaces the one a fixed height sets.
        self._detail = QWidget()
        self._detail.setFixedHeight(DETAIL_LINES * design.scaled(design.TYPE["body"][1]))
        inside = QVBoxLayout(self._detail)
        inside.setContentsMargins(0, 0, 0, 0)
        inside.addWidget(self.full_error)
        self._detail.setVisible(False)

        body = QWidget()
        lines = QVBoxLayout(body)
        lines.setContentsMargins(0, 0, 0, 0)
        lines.setSpacing(design.scaled(design.SIZE["row_gap"]))
        lines.addWidget(self.message)
        lines.addWidget(self.rest)
        lines.addLayout(buttons)
        lines.addWidget(self._detail)
        lines.addStretch(1)
        self.page.add_scrolling(body)

    def show_full_error(self) -> None:
        """Open the full error under the buttons, once and for good, with the keyboard in it."""
        self._detail.setVisible(True)
        self.full_error.setFocus(Qt.OtherFocusReason)        # before the link goes, or the focus would go on
        self.link.setVisible(False)

    def first_focus(self) -> QWidget:
        """The control the keyboard starts at: "Try again"."""
        return self.again

    def set_small(self, on: bool) -> None:
        """The low window, or the ordinary one again."""
        self.page.set_small(on)
