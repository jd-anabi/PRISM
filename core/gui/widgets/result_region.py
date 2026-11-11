"""What a page says after its run, under what the step holds: sentences, a ruled table, pictures with their captions,
and up to two plain buttons for the next steps, each of which carries what the step it opens is to take.

The region reads nothing but the pictures a page hands it by their paths, which ``set_pictures`` reads: a page calls
it once its run has ended, never while a run is live. The words a page hands it it shows. It is hidden while it holds
nothing, so that a page with no result has no gap where one would stand. A table shows every row, its
numbers at the right, its headings across its width, its caption above it and its footnote below it in the small type.
A picture is the figure the engine drew, shown as it is, light in both themes, and no wider than the page: it is made
smaller, never larger.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import design, style
from ..step_facts import LEADS, Fact, TableWords
from .figure_stack import _FitLabel
from .message_row import MessageRow
from .rule_table import Column, RuleTable, TextCell, ValueCell

#: The most next steps a result offers: one where a person goes on, one where they look again.
MOST_NEXT_STEPS = 2


@dataclass(frozen=True)
class NextStep:
    """A plain button under a result: its ``words``, named for where it goes, the ``step`` it opens (a key of the six
    steps) and what it ``carries`` there, by name, which the step opened takes."""
    words: str
    step: str
    carries: Mapping = field(default_factory=dict)


class ResultRegion(QWidget):
    """A page's result after its run. ``next_pressed`` carries the ``NextStep`` whose button was pressed."""

    next_pressed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._facts: tuple = ()
        self._rows: list = []               # the message row of each fact said
        self._table: "TableWords | None" = None
        self._steps: tuple = ()
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(design.scaled(design.GAP["head_table"]))
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
        self.footnote = self._small()
        box.addWidget(self.footnote)
        self._pictures = QVBoxLayout()
        self._pictures.setContentsMargins(0, 0, 0, 0)
        self._pictures.setSpacing(design.scaled(design.GAP["head_table"]))
        box.addLayout(self._pictures)
        self.pictures: list = []             # (its caption, the picture) for each picture shown
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

    def sentences(self) -> tuple:
        """The facts said, in their order."""
        return self._facts

    def set_table(self, table: "TableWords | None") -> None:
        """Show ``table`` as a ruled table, or none: every row, the columns of numbers at the right, the headings across
        the table's width, the caption above and the footnote below."""
        if self.table is not None:
            self._table_box.removeWidget(self.table)
            self.table.hide()
            self.table.deleteLater()
            self.table = None
        self._table = table
        if table is not None:
            columns = [Column(title, width=None if at == 0 else "fit", right=at in table.right, wrap=at != 0)
                       for at, title in enumerate(table.columns)]
            rows = [[TextCell(words, wrap=True) if at == 0 or row in table.headings or at not in table.right
                     else ValueCell(words)
                     for at, words in enumerate(cells)] for row, cells in enumerate(table.rows)]
            self.table = RuleTable(columns, rows, selectable=False)
            for row in table.headings:
                self.table.view.setSpan(row, 0, 1, len(table.columns))
            self._table_box.addWidget(self.table)
        self.caption.setText(table.caption if table is not None else "")
        self.caption.setVisible(table is not None and bool(table.caption))
        self.footnote.setText(table.footnote if table is not None else "")
        self.footnote.setVisible(table is not None and bool(table.footnote))
        self._judge()

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
            caption.setVisible(bool(words))
            picture = _FitLabel()
            picture.set_source(pixmap)
            self._pictures.addWidget(caption)
            self._pictures.addWidget(picture)
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

    def next_steps(self) -> tuple:
        return self._steps

    def _pressed(self) -> None:
        """A next step's button was pressed: said with what it carries."""
        at = self.next_buttons.index(self.sender())
        if at < len(self._steps):
            self.next_pressed.emit(self._steps[at])

    def clear(self) -> None:
        """Nothing said, shown or offered: the region hidden."""
        self.set_sentences(())
        self.set_table(None)
        self.set_pictures(())
        self.set_next_steps(())

    def stops(self) -> list:
        """What the Tab key stops at here: the next steps' buttons offered, in their order."""
        return [button for button in self.next_buttons[:len(self._steps)]]

    def is_empty(self) -> bool:
        return not (self._facts or self._table is not None or self.pictures or self._steps)

    def _judge(self) -> None:
        self.setVisible(not self.is_empty())
