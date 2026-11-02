"""The head of a page's scrolling body: what the step holds, one sentence on it, and the line a page says of its own."""
import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .. import design
from ..step_facts import LEADS, Fact
from .message_row import MessageRow
from .page import Sentence

NOTHING_YET = "In use: nothing yet."
IN_USE = "In use: {}"
ONE_PENDING = "1 choice below is changed and not applied."
SEVERAL_PENDING = "{} choices below are changed and not applied."


class InUse(QWidget):
    """"In use: retrain_prior" and what tells it from its neighbours, read from what the step holds and never from a
    control; how many choices below it are changed and not applied; and one line of the page's own."""

    def __init__(self) -> None:
        super().__init__()
        self._said = NOTHING_YET
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(design.scaled(design.GAP["head_table"]))
        self.line = QLabel(NOTHING_YET)
        self.line.setTextFormat(Qt.RichText)
        self.line.setWordWrap(True)
        self.sentence = Sentence()
        self.sentence.setVisible(False)
        self.pending = MessageRow("warn", "", lead=LEADS["warn"])
        self.pending.setVisible(False)
        self.said = MessageRow("info", "", lead=LEADS["info"])
        self.said.setVisible(False)
        for part in (self.line, self.sentence, self.pending, self.said):
            box.addWidget(part)

    def set_held(self, label: str, sentence: str) -> None:
        """Say what the step holds: ``label``, in semibold, and ``sentence``; with no label, that it holds nothing yet,
        and ``sentence`` what it waits for."""
        self._said = IN_USE.format(label) if label else NOTHING_YET
        self.line.setText(IN_USE.format(f"<span style='font-weight:600'>{html.escape(label, quote=False)}</span>") if label
                          else NOTHING_YET)
        self.sentence.set_text(sentence)
        self.sentence.setVisible(bool(sentence))

    def line_text(self) -> str:
        """The first line in plain words."""
        return self._said

    def set_pending(self, count: int) -> None:
        """Say how many choices below are changed and not applied; none says nothing."""
        self.pending.set_sentence(ONE_PENDING if count == 1 else SEVERAL_PENDING.format(count), lead=LEADS["warn"])
        self.pending.setVisible(count > 0)

    def say(self, fact: "Fact | None") -> None:
        """Say a line of the page's own under what is in use; None takes it away."""
        if fact is not None:
            self.said.set_kind(fact.kind)
            self.said.set_sentence(fact.sentence, lead=LEADS[fact.kind])
        self.said.setVisible(fact is not None)
