"""The one dialog the window asks and tells through.

A question, a notice, a refusal and a failure are the same dialog: a marker, one sentence in the book face,
and the answers at opposite ends, the safe one at the right. The safe answer is the default and stays the
default: Enter, Escape and the frame's close button all mean it wherever the keyboard focus is, and the
other answer is pressed with the mouse or the space bar. One further form asks for a line of text; there
Enter means the other answer once the text holds and nothing before.

``ask`` and ``ask_text`` are the two ways in. Each builds the dialog, blocks in its ``exec`` until it is
answered, hands back the answer and lets go of the dialog. Importing this module builds no widget and no font.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QLayout, QLineEdit, QPlainTextEdit,
                               QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from .. import design, fonts
from ..style import set_class
from .marker import Marker, painted_icon
from .page import Sentence

ASK_MARKERS = ("warn", "err", "info")   # the three a question may carry; not the ten states of the paint module
ITEMS_SHOWN = 8                    # a list longer than this scrolls inside the question
DETAIL_LINES = 12                  # the height of the full error, in lines of console text
CANCEL = "Cancel"                  # the safe answer of the form that asks for a line of text
FULL_ERROR = "Show the full error"


@dataclass(frozen=True)
class Wording:
    """What a question says, as plain words: readable after the dialog itself is gone."""
    title: str
    sentence: str
    safe: str
    other: str | None              # None: the question only tells, and has one button
    loses_work: bool
    marker: str
    items: tuple[str, ...]
    beneath: str                   # "" for no closing lines
    detail: str                    # "" for no error
    label: str | None              # None, except in the form that asks for a line of text


class Ask(QDialog):
    """A question, a notice, a refusal or a failure, or the form that asks for one line of text. A caller
    opens one through ``ask`` or ``ask_text``. A part that was not asked for is not built, and its name holds
    None. ``wording`` and ``answer_text`` are plain data, so both can be read after the dialog has closed."""

    def __init__(self, parent, title: str, sentence: str, safe: str, other: str | None = None, *,
                 loses_work: bool = False, marker: str = "warn", items: Sequence[str] | None = None,
                 beneath: str | None = None, detail: str | None = None, label: str | None = None,
                 text: str = "", check: Callable[[str], str | None] | None = None) -> None:
        # Both refusals come before the base class is built: a call refused later would leave a dialog,
        # half built, under its parent.
        if marker not in ASK_MARKERS:
            raise ValueError(f"a question carries one of the markers {ASK_MARKERS}, not {marker!r}")
        if label is not None and other is None:
            raise ValueError("a question that asks for a line of text needs the answer that takes the line")
        super().__init__(parent)
        self.wording = Wording(title=title, sentence=sentence, safe=safe, other=other, loses_work=bool(loses_work),
                               marker=marker, items=tuple(str(item) for item in (items or ())),
                               beneath=beneath or "", detail=detail or "", label=label)
        self.answer_text = None                  # the line of text, kept when the other answer is taken
        self._check = check
        scaled = design.scaled
        set_class(self, cls="ask")
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.WindowModal if parent is not None
                               else Qt.WindowModality.ApplicationModal)
        left, top, right, bottom = (scaled(n) for n in design.DIALOG["pad"])
        mark_gap = scaled(design.DIALOG["marker_gap"])
        # The text column is worked out from the scaled parts, never scaled as one number: the question
        # is then exactly its scaled width at any text size.
        column = scaled(design.DIALOG["w"]) - left - right - scaled(design.DIALOG["marker"]) - mark_gap
        outer = QVBoxLayout(self)
        outer.setContentsMargins(left, top, right, bottom)
        outer.setSpacing(scaled(design.DIALOG["gap"]))
        outer.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)     # as tall as its words, and no taller

        # Each row is filled in reading order and joins the dialog only when it is full: the Tab key then
        # walks the controls in the order they are read, with no tab order set.
        body = QHBoxLayout()
        body.setSpacing(mark_gap)
        beside = QVBoxLayout()
        beside.setContentsMargins(0, scaled(design.DIALOG["marker_drop"]), 0, 0)
        self.marker = Marker(marker, design.DIALOG["marker"])        # handed over as it stands: the marker scales it
        beside.addWidget(self.marker)
        beside.addStretch(1)
        body.addLayout(beside)
        words = QVBoxLayout()
        words.setSpacing(scaled(design.SIZE["row_gap"]))
        self.sentence = Sentence(sentence)
        self.sentence.setFixedWidth(column)
        self.sentence.setMinimumHeight(self.sentence.heightForWidth(column))     # a wider question must not cut it
        words.addWidget(self.sentence)
        self.items_area = self._list(column) if self.wording.items else None
        if self.items_area is not None:
            words.addWidget(self.items_area)
        self.beneath = None
        if self.wording.beneath:
            self.beneath = Sentence(beneath, quiet=True)
            self.beneath.setFixedWidth(column)
            self.beneath.setMinimumHeight(self.beneath.heightForWidth(column))
            words.addWidget(self.beneath)
        self.field = self.field_message = self._wrong = None
        if label is not None:
            words.addLayout(self._line(label, text))
        self.detail_link = self.detail_box = None
        if self.wording.detail:
            self.detail_link = set_class(QPushButton(FULL_ERROR), cls="link")
            self.detail_link.setAutoDefault(False)           # a link never takes the default from an answer
            self.detail_link.setCursor(Qt.CursorShape.PointingHandCursor)
            self.detail_link.setFixedHeight(scaled(design.SIZE["line_h"]))
            self.detail_link.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            self.detail_link.clicked.connect(self._show_detail)
            words.addWidget(self.detail_link, 0, Qt.AlignmentFlag.AlignLeft)
        words.addStretch(1)
        body.addLayout(words)
        outer.addLayout(body)

        answers = QHBoxLayout()
        answers.setSpacing(scaled(design.GAP["action"]))
        # A button reads a single "&" in its words as the mark of an access key, leaves it out and takes
        # the key. Doubled, it is shown and no key is taken; the wording keeps the words as they were written.
        self.safe_button = QPushButton(safe.replace("&", "&&"))
        self.safe_button.clicked.connect(self.reject)
        self.other_button = None
        if other is not None:
            self.other_button = QPushButton(other.replace("&", "&&"))
            self.other_button.clicked.connect(self._take_other)
            if loses_work:
                set_class(self.other_button, cls="danger")
                self.other_button.setIcon(painted_icon("stop"))
                # the glyph and the space after it, each scaled by itself and then added
                self.other_button.setIconSize(QSize(scaled(design.SIZE["marker"]) + scaled(design.SIZE["icon_gap"]),
                                                    scaled(design.SIZE["marker"])))
            answers.addWidget(self.other_button)
        answers.addStretch(1)
        answers.addWidget(self.safe_button)
        outer.addLayout(answers)

        if self.wording.detail:
            self.detail_box = QPlainTextEdit()
            self.detail_box.setReadOnly(True)
            self.detail_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            # The font is given here, at build, and never when the link is pressed: a font handed to a
            # widget after a sheet has dressed it is dropped at the next change of sheet.
            self.detail_box.setFont(fonts.fixed())
            self.detail_box.setPlainText(detail)
            # In a plain holder of the fixed height: the sheet gives a text box a least height of its own,
            # which replaces the one a fixed height sets, and the box would take the toolkit's own height.
            self._detail = QWidget()
            self._detail.setFixedHeight(DETAIL_LINES * scaled(design.TYPE["body"][1]))
            inside = QVBoxLayout(self._detail)
            inside.setContentsMargins(0, 0, 0, 0)
            inside.addWidget(self.detail_box)
            self._detail.setVisible(False)
            outer.addWidget(self._detail)

        # The default is named last, when every button is the dialog's own: the dialog then gives the
        # default back to this button whenever the focus leaves one that took it with the focus.
        if label is None:
            set_class(self.safe_button, cls="primary")
            if self.other_button is not None:
                self.other_button.setAutoDefault(False)      # or Enter on it would mean the other answer
            self.safe_button.setDefault(True)
            self.safe_button.setFocus(Qt.FocusReason.TabFocusReason)
        else:
            # Here the other answer is the main button. The safe one is left as the toolkit makes it: it
            # takes the default while it has the focus, so Enter on it means the safe answer.
            set_class(self.other_button, cls="primary")
            self.other_button.setDefault(True)
            self.field.textChanged.connect(self._judge)      # not textEdited: a text set from code is judged too
            self._judge(self.field.text())
            self.field.setFocus(Qt.FocusReason.TabFocusReason)       # which selects the text handed in

    def _list(self, column: int) -> QScrollArea:
        """The items, one to a line and none cut: as tall as the first eight of them, scrolling beyond that."""
        scrolls = len(self.wording.items) > ITEMS_SHOWN
        area = QScrollArea()
        # The frame is switched off here, at build, before anything else: the sheet gives a line to a
        # scrolling area that still has the frame it is born with, and reads that when the area is dressed.
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The bar is always there beyond eight items and never there up to eight, so the width an item
        # has is known before the question is shown.
        area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn if scrolls
                                        else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gap = design.SIZE["scroll_gap"] if scrolls else 0    # a line weight: it does not grow with the text
        # the whole column while the list does not scroll; less the bar and the gap beside it while it does
        width = column - (area.verticalScrollBar().sizeHint().width() + gap if scrolls else 0)
        holder = QWidget()
        rows = QVBoxLayout(holder)
        rows.setContentsMargins(0, 0, gap, 0)
        rows.setSpacing(0)
        heights = []
        for item in self.wording.items:
            row = Sentence(item)
            row.setFixedWidth(width)             # or one item that cannot break widens every item
            heights.append(row.heightForWidth(width))
            rows.addWidget(row)
        area.setWidget(holder)
        # Handing the holder over switches on its ground, and the view fills itself with the window's
        # colour: both are switched off again, so that the question's own ground shows through.
        holder.setAutoFillBackground(False)
        area.viewport().setAutoFillBackground(False)
        area.setFixedSize(column, sum(heights[:ITEMS_SHOWN]))
        # a stop for the Tab key only while it scrolls, its focus ring then drawn just inside its edge
        area.setFocusPolicy(Qt.FocusPolicy.TabFocus if scrolls else Qt.FocusPolicy.NoFocus)
        if scrolls:
            area.setProperty("ring", "inset")
        return area

    def _line(self, label: str, text: str) -> QVBoxLayout:
        """The one labelled field and, under it, the place for what is wrong with the line."""
        scaled = design.scaled
        block = QVBoxLayout()
        block.setSpacing(scaled(design.GAP["field_message"]))
        row = QHBoxLayout()
        row.setSpacing(scaled(design.SIZE["row_gap"]))
        named = QLabel()
        named.setTextFormat(Qt.TextFormat.PlainText)
        named.setText(label.replace("&", "&&"))              # a label that names a field reads "&" as a button does
        self.field = QLineEdit(text)
        named.setBuddy(self.field)
        row.addWidget(named)
        row.addWidget(self.field, 1)
        block.addLayout(row)
        self._wrong = QWidget()                              # what is wrong with the line; hidden while nothing is
        wrong = QHBoxLayout(self._wrong)
        wrong.setContentsMargins(0, 0, 0, 0)                 # a widget's own layout is born with the style's margins
        wrong.setSpacing(scaled(design.GAP["marker_text"]))
        beside = QVBoxLayout()
        beside.setContentsMargins(0, scaled(design.GAP["marker_drop"]), 0, 0)
        beside.addWidget(Marker("err"))
        beside.addStretch(1)
        wrong.addLayout(beside)
        self.field_message = QLabel()
        self.field_message.setTextFormat(Qt.TextFormat.PlainText)
        self.field_message.setWordWrap(True)
        wrong.addWidget(self.field_message, 1)
        self._wrong.setVisible(False)
        block.addWidget(self._wrong)
        return block

    def _judge(self, text: str) -> None:
        """Hears every change of the line and says whether it holds. A blank line never holds, shows nothing
        and is not handed to the caller's rule; any other line holds unless that rule answers with the
        sentence to show. What the rule raises is not caught."""
        self.other_button.setEnabled(False)          # first: a rule that raises leaves the answer greyed
        text = text.strip()
        wrong = (self._check(text) if text and self._check is not None else None) or ""
        self.field_message.setText(wrong)
        self._wrong.setVisible(bool(wrong))
        set_class(self.field, state="error" if wrong else "")
        self.other_button.setEnabled(bool(text) and not wrong)

    def _take_other(self) -> None:
        """Hears the other answer. In the form that asks for a line of text the line is kept first."""
        if self.field is not None:
            self.answer_text = self.field.text().strip()
        self.accept()

    def _show_detail(self) -> None:
        """Hears the link: the full error opens under the answers, once and for good."""
        self._detail.setVisible(True)
        self.detail_box.setFocus(Qt.FocusReason.TabFocusReason)      # or the focus would go on to an answer
        self.detail_link.setVisible(False)


def ask(parent, title: str, sentence: str, safe: str, other: str | None = None, *, loses_work: bool = False,
        marker: str = "warn", items: Sequence[str] | None = None, beneath: str | None = None,
        detail: str | None = None) -> bool:
    """Ask, or with no other answer tell. True for the other answer, False for the safe one."""
    dialog = Ask(parent, title, sentence, safe, other, loses_work=loses_work, marker=marker, items=items,
                 beneath=beneath, detail=detail)
    try:
        return dialog.exec() == QDialog.DialogCode.Accepted
    finally:
        dialog.deleteLater()


def ask_text(parent, title: str, sentence: str, label: str, other: str, *, text: str = "",
             check: Callable[[str], str | None] | None = None, items=None, beneath=None) -> str | None:
    """Ask for one line of text. The text for the other answer, None for "Cancel"."""
    dialog = Ask(parent, title, sentence, CANCEL, other, marker="info", items=items, beneath=beneath,
                 label=label, text=text, check=check)
    try:
        dialog.exec()
        return dialog.answer_text
    finally:
        dialog.deleteLater()
