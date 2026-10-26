"""Home: what to do next, where each step of parameter inference stands, what each part of PRISM is for and the
newest records; on a computer where nothing has been made yet, four plain choices in place of the next step.

Home states only what the function it is handed reads, at the moment it reads. It reads once when it is built,
again when it is told to while it is shown, and once when it is shown after it was marked to read again; never
while it is hidden. A table is filled again only when its rows have changed, so a person reading it with the keys
keeps their row. A row that opens something opens on a click, on Enter and on Space, under a pointing hand.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .. import chain, design, fonts, keys, style
from ..reported import reported
from ..widgets.page import Columns, Page, PageHead, Sentence
from ..widgets.rule_table import Column, RuleTable, StateCell, TextCell
from ..widgets.ruled import SectionHead

LATEST = 5                       # how many of the newest records Home lists

_HEAD = ("Home", "What to do next, where your work stands, and what each part is for.")
_FIRST_HEAD = ("What do you want to do?",
               "PRISM simulates models of the hair bundle, the motion sensor of the inner ear's hair cells, and "
               "infers a cell's model parameters from recordings. Nothing has been made yet.")

# The first opening's four choices, in their order: the button's words, the sentence beside it, and the part and
# page the button opens. The first is the main one.
_CHOICES = (
    ("Watch a live simulation", "Needs nothing first. See a cell's bundle move as it is simulated.", "live", None),
    ("Begin parameter inference", "Six steps, each needing the one before. It starts with choosing a model.",
     "inference", "model"),
    ("Measure an effective temperature", "On one simulated cell, or in a sweep study over two parameters.",
     "temperature", None),
    ("Read what each part is for", "A short section per part.", "help", None),
)

# the keys that open the row a table's keyboard is on
_OPENING_KEYS = (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space)


@dataclass(frozen=True)
class RecordRow:
    """One record as Home lists it: the store's word for its kind, its id, its label, the word for one record of
    its kind, and the moment it was made, in local time (None when its manifest gives none)."""
    kind: str
    id: str
    label: str
    kind_word: str
    made: datetime | None


@dataclass(frozen=True)
class HomeFacts:
    """What Home states, read at one moment: the inference session, its six steps' states, the part and page a
    live run is in (None while nothing runs), every record of the store in use, whether the records folder in
    use has a run in its history, and the live run's word and name while it is stopping ("" otherwise)."""
    session: object
    states: tuple
    live_place: tuple[str, str | None] | None
    records: tuple[RecordRow, ...]
    has_runs: bool = False
    stopping: str = ""


def _access_key(words: str) -> str:
    """A main button's words with its access key on their first letter: Alt and that letter press it."""
    return "&" + words.replace("&", "&&")


def _capital(words: str) -> str:
    """Words as a table's cell begins them, with a capital letter."""
    return words[:1].upper() + words[1:]


def _newest(records) -> list:
    """The records newest first; one whose moment is not known comes last."""
    return sorted(records, key=lambda row: (row.made is None, -row.made.timestamp() if row.made else 0.0))


def _made_words(made: datetime | None) -> str:
    """When a record was made, as the window writes a date: with its weekday within seven days of now."""
    return "" if made is None else design.date_text(made, datetime.now(made.tzinfo))


def _heading(words: str, right: QWidget | None = None) -> QHBoxLayout:
    """A block's heading line: its words at the heading's size and weight, and a link at its right when one is
    handed in, standing on the line's foot. The table under it brings its own heavy rule."""
    label = QLabel(words)
    label.setTextFormat(Qt.PlainText)
    style.set_class(label, cls="head")                   # what the sheet's rule for a heading matches
    # The font is given here, when the label is built, and nowhere later: a font handed to a widget after a
    # sheet has dressed it is dropped at the next change of sheet.
    label.setFont(fonts.inter("head"))
    label.setFixedHeight(design.scaled(design.TYPE["head"][1]))
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(label)
    row.addStretch(1)
    if right is not None:
        row.addWidget(right, 0, Qt.AlignBottom)
    return row


def _block(heading: QHBoxLayout, *parts: QWidget) -> QWidget:
    """A block of a column: its heading line, the space the look puts between a heading and its table, and what
    stands under the heading, a table most often."""
    block = QWidget()
    lines = QVBoxLayout(block)
    lines.setContentsMargins(0, 0, 0, 0)
    lines.setSpacing(0)
    lines.addLayout(heading)
    lines.addSpacing(design.scaled(design.GAP["head_table"]))
    for part in parts:
        lines.addWidget(part)
    return block


class _Body(QWidget):
    """The middle of Home: the next step, or the four choices, over two columns. It hands the width the page gives
    it on to the columns, which stand one above the other on a narrow page."""

    def __init__(self, top: QWidget, choices: QWidget, columns: Columns) -> None:
        super().__init__()
        self._columns = columns
        lines = QVBoxLayout(self)
        lines.setContentsMargins(0, 0, 0, 0)
        lines.setSpacing(0)
        lines.addWidget(top)
        lines.addWidget(choices)
        lines.addSpacing(design.scaled(design.GAP["block"]))
        lines.addWidget(columns)

    def fit(self, width: int) -> None:
        """The width the page gives the middle, handed on to the two columns."""
        self._columns.fit(width)


class HomeScreen(QWidget):
    """Home. ``read`` gives what Home states, as a ``HomeFacts``; ``navigate(part, page)`` asks for a place to be
    opened: a part and one of its pages, a step of parameter inference as ("inference", step), or the records of
    one kind as ("records", kind)."""

    navigate = Signal(str, object)

    def __init__(self, read: Callable[[], HomeFacts], parent=None) -> None:
        super().__init__(parent)
        self._read = read
        self._stale = False
        self._first = False
        self._next = None
        self._rows: dict = {}            # each table's view: the rows it holds, to fill it again only on a change
        self._opens: dict = {}           # each table's view: the place each of its rows opens

        self.page = Page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.page)
        self._head = PageHead(*_HEAD)
        self.page.add_fixed(self._head)
        self.page.add_gap(design.scaled(design.GAP["head_to_body"]), design.scaled(design.GAP["head_to_body_small"]))

        # the next step: its heading over a rule, a sentence, and the main button
        self.next_heading = SectionHead("")
        self._next_sentence = Sentence()
        self.next_button = style.set_class(QPushButton(), cls="primary")
        self.next_button.clicked.connect(self._next_pressed)
        self._next_block = QWidget()
        lines = QVBoxLayout(self._next_block)
        lines.setContentsMargins(0, 0, 0, 0)
        lines.setSpacing(design.scaled(design.SIZE["row_gap"]))    # one space between those shown: a next
        lines.addWidget(self.next_heading)                         # step with no sentence keeps no room for one
        lines.addWidget(self._next_sentence)
        lines.addWidget(self.next_button, 0, Qt.AlignLeft)

        # the first opening: four buttons, each with its sentence beside it
        self._choices_block = QWidget()
        grid = QGridLayout(self._choices_block)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(design.scaled(design.SIZE["table_col_gap"]))
        grid.setVerticalSpacing(design.scaled(design.SIZE["row_gap"]))
        grid.setColumnStretch(1, 1)
        self.choices = []
        for row, (words, sentence, part, page) in enumerate(_CHOICES):
            button = QPushButton(_access_key(words) if row == 0 else words)
            if row == 0:
                style.set_class(button, cls="primary")
            button.clicked.connect(self._choice_pressed)
            grid.addWidget(button, row, 0)
            grid.addWidget(Sentence(sentence), row, 1, Qt.AlignVCenter)
            self.choices.append(button)

        # the three tables
        self.steps_table = RuleTable([Column("Step", "fit"), Column("State", "fit"), Column("Holds or waits for")])
        self.parts_table = RuleTable([Column("Part", "fit"), Column("What it is for")])
        self.records_table = RuleTable([Column("Name"), Column("Kind", "fit"), Column("Made", "fit")])
        for table in (self.steps_table, self.parts_table, self.records_table):
            view = table.view
            view.viewport().setCursor(Qt.PointingHandCursor)
            view.clicked.connect(self._row_clicked)
            view.installEventFilter(self)
            self._rows[view], self._opens[view] = None, ()
        self._all_records = style.set_class(QPushButton("All records"), cls="link")
        self._all_records.setCursor(Qt.PointingHandCursor)
        self._all_records.setFixedHeight(design.scaled(design.SIZE["line_h"]))
        self._all_records.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self._all_records.clicked.connect(self._all_records_pressed)
        self._no_records = Sentence("No records yet.", quiet=True)

        # the steps at the left; the parts and the latest records at the right
        self._left = QWidget()
        column = QVBoxLayout(self._left)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(_block(_heading("Parameter inference"), self.steps_table))
        column.addStretch(1)
        right = QWidget()
        blocks = QVBoxLayout(right)
        blocks.setContentsMargins(0, 0, 0, 0)
        blocks.setSpacing(design.scaled(design.GAP["block"]))
        blocks.addWidget(_block(_heading("The parts of PRISM"), self.parts_table))
        blocks.addWidget(_block(_heading("Latest records", self._all_records), self.records_table, self._no_records))
        blocks.addStretch(1)
        self._columns = Columns(self._left, right, stretch=(2, 3))     # the parts' lines need the wider column
        self.page.add_scrolling(_Body(self._next_block, self._choices_block, self._columns))

        self._fill_parts()
        self._fill(read())

    # what Home states
    def refresh(self) -> None:
        """Read what Home states and show it, now while Home is shown; while it is hidden, at its next show."""
        if self.isVisible():
            self._fill(self._read())
        else:
            self._stale = True

    def mark_stale(self) -> None:
        """Have Home read again when it is next shown; nothing is read now."""
        self._stale = True

    def is_first_time(self) -> bool:
        """Whether Home shows its first opening: no record, no run and nothing chosen when it last read."""
        return self._first

    def set_small(self, on: bool) -> None:
        """The low window, or the ordinary one again."""
        self.page.set_small(on)

    def first_focus(self) -> QWidget | None:
        """The control the keyboard starts at: the main button, which opens a page and starts nothing."""
        return self.choices[0] if self._first else self.next_button

    @reported(QWidget.showEvent)
    def showEvent(self, event):
        super().showEvent(event)
        if self._stale:
            self._fill(self._read())

    def _fill(self, facts: HomeFacts) -> None:
        """Show ``facts``: the first opening or the page of a project under way, the steps, the newest records."""
        self._stale = False
        self._first = chain.first_time(has_records=bool(facts.records), has_runs=facts.has_runs,
                                       session=facts.session)
        title, sentence = _FIRST_HEAD if self._first else _HEAD
        self._head.title.setText(title)
        self._head.set_sentence(sentence)
        self._choices_block.setVisible(self._first)
        self._next_block.setVisible(not self._first)
        self._left.setVisible(not self._first)
        self._next = chain.next_step(facts.session, facts.states, live_place=facts.live_place,
                                     stopping=facts.stopping)
        self.next_heading.set_text(self._next.heading)
        self._next_sentence.set_text(self._next.sentence)
        self._next_sentence.setVisible(bool(self._next.sentence))
        self.next_button.setText(_access_key(self._next.button))
        self._set_rows(self.steps_table,
                       [[TextCell(f"{state.number} {keys.step(state.key).name}"), StateCell(state.marker, state.word),
                         TextCell(state.detail, wrap=True)] for state in facts.states],
                       [("inference", state.key) for state in facts.states])
        newest = _newest(facts.records)[:LATEST]
        self._set_rows(self.records_table,
                       [[TextCell(row.label, elide="middle"), TextCell(_capital(row.kind_word), wrap=True),
                         TextCell(_made_words(row.made))] for row in newest],
                       [("records", row.kind) for row in newest])
        self.records_table.setVisible(bool(newest))
        self._all_records.setVisible(bool(newest))
        self._no_records.setVisible(not newest)

    def _fill_parts(self) -> None:
        """The parts of PRISM, each with its line: they do not change while PRISM runs."""
        parts = keys.PARTS[1:]
        self._set_rows(self.parts_table, [[TextCell(place.name), TextCell(place.line, wrap=True)] for place in parts],
                       [(place.key, None) for place in parts])

    def _set_rows(self, table: RuleTable, rows: list, opens: list) -> None:
        """Fill ``table`` with ``rows`` only when they differ from the rows it holds; ``opens`` are the places its
        rows open, in their order."""
        view = table.view
        self._opens[view] = tuple(opens)
        if rows != self._rows[view]:
            table.set_rows(rows)
            self._rows[view] = rows

    # what Home opens
    def _next_pressed(self) -> None:
        if self._next is not None:
            self.navigate.emit(self._next.part, self._next.page)

    def _choice_pressed(self) -> None:
        for button, (_words, _sentence, part, page) in zip(self.choices, _CHOICES):
            if button is self.sender():
                self.navigate.emit(part, page)

    def _all_records_pressed(self) -> None:
        self.navigate.emit("records", None)

    def _row_clicked(self, index) -> None:
        for view in self._opens:
            if index.model() is view.model():
                self._open(view, index.row())

    def _open(self, view, row: int) -> None:
        """Open the place of a table's row; the row is not left selected behind it."""
        opens = self._opens[view]
        if 0 <= row < len(opens):
            view.clearSelection()
            self.navigate.emit(*opens[row])

    @reported(False)
    def eventFilter(self, watched, event):
        """Enter and Space open the row the keyboard is on in one of the three tables."""
        if event.type() == QEvent.KeyPress and event.key() in _OPENING_KEYS:
            for view in self._opens:
                if watched is view:
                    index = view.currentIndex()
                    if index.isValid():
                        self._open(view, index.row())
                    return True
        return False
