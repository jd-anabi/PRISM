"""What a page says after its run, under what the step holds: a verdict, sentences, a ruled table, pictures with their
captions, lines a person copies into a file of their own with the button that copies them, and up to two plain
buttons for the next steps, each of which carries what the step it opens is to take.
A verdict is a check's: one message row with the marker of its state and no lead word, its sentence beginning with
the check's name.

The region reads nothing but the pictures a page hands it by their paths, which ``set_pictures`` reads: a page calls
it once its run has ended, never while a run is live. The words a page hands it it shows. It is hidden while it holds
nothing, so that a page with no result has no gap where one would stand. A table shows every row, its
numbers at the right, its headings across its width, its caption above it and its footnote below it in the small type.
Its first column, which names its rows, keeps its widest word: where the region is narrower than the table, the table
scrolls sideways inside a block of its own rather than crush that column.
A picture is the figure the engine drew, shown as it is, light in both themes, and no wider than the page: it is made
smaller, never larger.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from .. import design, fonts, style
from ..reported import reported
from ..step_facts import LEADS, Fact, TableWords
from .figure_stack import _FitLabel
from .message_row import MessageRow
from .rule_table import Column, RuleTable, TextCell, ValueCell

#: The most next steps a result offers: one where a person goes on, one where they look again.
MOST_NEXT_STEPS = 2
#: The words of the button beside lines to copy.
COPY_WORDS = "Copy"


@dataclass(frozen=True)
class NextStep:
    """A plain button under a result: its ``words``, named for where it goes, the ``step`` it opens (a key of the six
    steps) and what it ``carries`` there, by name, which the step opened takes."""
    words: str
    step: str
    carries: Mapping = field(default_factory=dict)


class _Sideways(QScrollArea):
    """A result's ``table`` in a block of its own, made inside ``parent``: as wide as the region, and where that is
    narrower than the least width the table keeps, its first column's words whole among it, the table scrolls sideways
    inside the block, its bar under it. The block asks for no more width than the table's first column, so that a page
    beside a list stays beside it; it is as tall as the table and, while it scrolls, its bar. While it scrolls it is a
    stop of the Tab key, so that the keyboard can scroll it, as a page's own middle is; the region lists it among its
    stops all the same, so that it has its place in the page's order before it first scrolls."""

    def __init__(self, table: RuleTable, parent=None) -> None:
        super().__init__(parent)
        self.table = table
        # The frame is switched off at build: the sheet gives a line to a scrolling area that still has the frame it is
        # born with.
        self.setFrameShape(QFrame.NoFrame)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setWidget(table)
        self.viewport().setAutoFillBackground(False)         # the table stands on the page's own ground
        table.installEventFilter(self)
        # A bound method, never a function that holds the block: connected to its own bar, such a function would keep
        # a block with no parent alive after its last name is gone.
        self.horizontalScrollBar().rangeChanged.connect(self._range_changed)
        self._fit_height()

    def scrolls(self) -> bool:
        """Whether the table is wider than the block, so that it scrolls sideways."""
        bar = self.horizontalScrollBar()
        return bar.maximum() > bar.minimum()

    def _height(self) -> int:
        bar = self.horizontalScrollBar()
        return self.table.sizeHint().height() + (bar.sizeHint().height() if self.scrolls() else 0)

    def _fit_height(self) -> None:
        if self.height() != self._height() or self.minimumHeight() != self._height():
            self.setFixedHeight(self._height())

    @reported()
    def _range_changed(self, _low, _high):
        """Hears the bar's range: while there is something to scroll, room for the bar under the table, and the block a
        stop of the Tab key."""
        self.setFocusPolicy(Qt.TabFocus if self.scrolls() else Qt.NoFocus)
        self._fit_height()

    @reported(False)
    def eventFilter(self, watched, event):
        """The table laid out again, its rows of another height: the block follows its height."""
        if watched is self.table and event.type() in (QEvent.LayoutRequest, QEvent.Resize):
            self._fit_height()
        return super().eventFilter(watched, event)

    @reported(QScrollArea.sizeHint)
    def sizeHint(self):
        return QSize(self.table.sizeHint().width(), self._height())

    @reported(QScrollArea.minimumSizeHint)
    def minimumSizeHint(self):
        return QSize(self.table.view.least_width(0), self._height())


class ResultRegion(QWidget):
    """A page's result after its run, made inside ``parent``. ``next_pressed`` carries the ``NextStep`` whose button was
    pressed; ``joined`` says a stop of the Tab key joined the region or left it (a table's block, "Copy", a next step's
    button, a button of its own), so that the page that holds it orders its keys again. A region with no parent is a
    window of its own: the first words it is handed would show it as one."""

    next_pressed = Signal(object)
    joined = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._facts: tuple = ()
        self.verdict: "MessageRow | None" = None   # a check's verdict, first in the region
        self._rows: list = []               # the message row of each fact said
        self._table: "TableWords | None" = None
        self._steps: tuple = ()
        self._actions: list = []            # buttons of the result's own, after everything it says
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(design.scaled(design.GAP["head_table"]))
        self._verdict_box = QVBoxLayout()
        self._verdict_box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(self._verdict_box)
        self._sentences = QVBoxLayout()
        self._sentences.setContentsMargins(0, 0, 0, 0)
        self._sentences.setSpacing(design.scaled(design.GAP["gate_rows"]))
        box.addLayout(self._sentences)
        self.caption = self._small()
        box.addWidget(self.caption)
        self._table_box = QVBoxLayout()
        self._table_box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(self._table_box)
        self.table: "RuleTable | None" = None
        self.table_block: "_Sideways | None" = None     # the block the table scrolls sideways in
        self.footnote = self._small()
        box.addWidget(self.footnote)
        self._pictures = QVBoxLayout()
        self._pictures.setContentsMargins(0, 0, 0, 0)
        self._pictures.setSpacing(design.scaled(design.GAP["head_table"]))
        box.addLayout(self._pictures)
        self.pictures: list = []             # (its caption, the picture) for each picture shown
        # Lines a person copies into a file of their own: "Copy" on its own line, so that no width cuts it, then the lines
        # in the fixed face, selectable, a line too long for the column wrapped on the page and copied whole.
        self.copy_button = QPushButton(COPY_WORDS)
        self.copy_button.setVisible(False)
        self.copy_button.clicked.connect(self._copy)
        copying = QHBoxLayout()
        copying.setContentsMargins(0, 0, 0, 0)
        copying.addWidget(self.copy_button)
        copying.addStretch(1)
        box.addLayout(copying)
        self.lines = QLabel()
        self.lines.setTextFormat(Qt.PlainText)
        self.lines.setFont(fonts.fixed())
        self.lines.setWordWrap(True)
        self.lines.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.lines.setVisible(False)
        box.addWidget(self.lines)
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(design.scaled(design.GAP["action"]))
        self.next_buttons: list = []
        for _ in range(MOST_NEXT_STEPS):
            button = QPushButton()
            button.setVisible(False)
            button.clicked.connect(self._pressed)
            line.addWidget(button)
            self.next_buttons.append(button)
        line.addStretch(1)
        box.addLayout(line)
        self.setVisible(False)

    @staticmethod
    def _small() -> QLabel:
        label = style.set_class(QLabel(), cls="small")
        label.setTextFormat(Qt.PlainText)
        label.setWordWrap(True)
        label.setVisible(False)
        return label

    def set_sentences(self, facts: Sequence[Fact]) -> None:
        """Say ``facts``, each a message row with its marker and its lead word."""
        for row in self._rows:
            self._sentences.removeWidget(row)
            row.hide()
            row.deleteLater()
        self._facts = tuple(facts)
        self._rows = [MessageRow(fact.kind, fact.sentence, lead=LEADS[fact.kind]) for fact in self._facts]
        for row in self._rows:
            self._sentences.addWidget(row)
        self._judge()

    def set_verdict(self, marker: str, sentence: str) -> None:
        """Say a check's verdict first in the region: a message row with the marker ``marker`` of its state and no lead
        word; an empty sentence says none."""
        if self.verdict is not None:
            self._verdict_box.removeWidget(self.verdict)
            self.verdict.hide()
            self.verdict.deleteLater()
            self.verdict = None
        if sentence:
            self.verdict = MessageRow(marker, sentence)
            self._verdict_box.addWidget(self.verdict)
        self._judge()

    def sentences(self) -> tuple:
        """The facts said, in their order."""
        return self._facts

    def set_table(self, table: "TableWords | None") -> None:
        """Show ``table`` as a ruled table, or none: every row, the columns of numbers at the right, the headings across
        the table's width, the caption above and the footnote below; the first column at least as wide as its widest
        word, the table scrolling sideways in its block where the region is narrower than it."""
        if self.table_block is not None:
            self._table_box.removeWidget(self.table_block)
            self.table_block.hide()
            self.table_block.deleteLater()             # the table with it
            self.table = self.table_block = None
        self._table = table
        if table is not None:
            columns = [Column(title, width=None if at == 0 else "fit", right=at in table.right, wrap=at != 0)
                       for at, title in enumerate(table.columns)]
            rows = [[TextCell(words, wrap=True) if at == 0 or row in table.headings or at not in table.right
                     else ValueCell(words)
                     for at, words in enumerate(cells)] for row, cells in enumerate(table.rows)]
            self.table = RuleTable(columns, rows, selectable=False, keeps_words=True)
            for row in table.headings:
                self.table.view.setSpan(row, 0, 1, len(table.columns))
            self.table_block = _Sideways(self.table, self)
            self._table_box.addWidget(self.table_block)
        self.caption.setText(table.caption if table is not None else "")
        self.caption.setVisible(table is not None and bool(table.caption))
        self.footnote.setText(table.footnote if table is not None else "")
        self.footnote.setVisible(table is not None and bool(table.footnote))
        self._judge()
        self.joined.emit()

    def table_words(self) -> "TableWords | None":
        return self._table

    def set_pictures(self, pictures: Sequence[tuple]) -> None:
        """Show each ``(caption, path)`` of ``pictures``: its caption in the small type over the picture, made no wider
        than the page. Each file is read here, so a page calls this once its run has ended, never while a run is live.
        A file that holds no picture is left out."""
        for caption, picture in self.pictures:
            for part in (caption, picture):
                self._pictures.removeWidget(part)
                part.hide()
                part.deleteLater()
        self.pictures = []
        for words, path in pictures:
            pixmap = QPixmap(str(Path(path)))
            if pixmap.isNull():
                continue
            caption = self._small()
            caption.setText(words)
            picture = _FitLabel()
            picture.set_source(pixmap)
            self._pictures.addWidget(caption)
            self._pictures.addWidget(picture)
            caption.setVisible(bool(words))    # in the region first: a caption with no parent made visible is a window
            self.pictures.append((caption, picture))
        self._judge()

    def set_next_steps(self, steps: Sequence[NextStep]) -> None:
        """Offer ``steps``, each a plain button named for where it goes; ValueError for more than two, before anything
        changes."""
        steps = tuple(steps)
        if len(steps) > MOST_NEXT_STEPS:
            raise ValueError(f"a result offers at most {MOST_NEXT_STEPS} next steps, not {len(steps)}")
        self._steps = steps
        for at, button in enumerate(self.next_buttons):
            button.setText(steps[at].words if at < len(steps) else "")
            button.setVisible(at < len(steps))
        self._judge()
        self.joined.emit()

    def next_steps(self) -> tuple:
        return self._steps

    def _pressed(self) -> None:
        """A next step's button was pressed: said with what it carries."""
        at = self.next_buttons.index(self.sender())
        if at < len(self._steps):
            self.next_pressed.emit(self._steps[at])

    def set_lines(self, text: str) -> None:
        """Lines to copy, ``text``, in the fixed face under "Copy", which puts them on the clipboard as they stand, a line
        the column cannot hold wrapped on the page only; an empty text shows neither."""
        self.lines.setText(text)
        for part in (self.lines, self.copy_button):
            part.setVisible(bool(text))
        self._judge()
        self.joined.emit()

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.lines.text())

    def clear(self) -> None:
        """Nothing said, shown or offered: the region hidden."""
        self.set_verdict("", "")
        self.set_sentences(())
        self.set_table(None)
        self.set_pictures(())
        self.set_lines("")
        self.set_next_steps(())

    def add_action(self, button: QPushButton, beneath: str = "") -> None:
        """A button of the result's own, on a line of its own after everything the region says, and ``beneath`` it,
        small, why it is greyed ("" for nothing): what a check's result offers to do with itself."""
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.addWidget(button)
        line.addStretch(1)
        self.layout().addLayout(line)
        if beneath:
            said = self._small()
            said.setText(beneath)
            self.layout().addWidget(said)
            said.setVisible(True)
        self._actions.append(button)
        self._judge()
        self.joined.emit()

    def actions(self) -> list:
        """The buttons of the result's own, in their order."""
        return list(self._actions)

    def stops(self) -> list:
        """What the Tab key stops at here, in the order it is read: the table's block whenever there is one, a stop
        only while it scrolls sideways but in its place all the same, so that a page orders its keys with it at any
        moment; the lines' "Copy" while it is offered, then the next steps' buttons offered, in their order, then the
        result's own buttons that can be pressed."""
        sideways = [self.table_block] if self.table_block is not None else []
        return [*sideways, *([self.copy_button] if self.lines.text() else []), *self.next_buttons[:len(self._steps)],
                *(button for button in self._actions if button.isEnabled())]

    def is_empty(self) -> bool:
        return not (self.verdict is not None or self._facts or self._table is not None or self.pictures or self._steps
                    or self.lines.text())

    def _judge(self) -> None:
        self.setVisible(not self.is_empty())
