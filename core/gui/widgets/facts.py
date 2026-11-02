"""The foot of a page: "Before you start" and, under it, the reason, "Check again" and the one main button.

A foot never scrolls, so the main button stays in the window at every size. The facts take what their rows need, up to
a height of the look's, past which they scroll inside their own region, so that they are never drawn over the reason or
the buttons. A reading keeps the rows it can: a row whose fact is unchanged is left alone, and one that changed is told
its new words, so that nothing a person reads or has scrolled moves. In the low window only what blocks or needs
attention stands, with a link that opens the rest in the same region.

While a run is live the two buttons are greyed and the facts stay readable: the foot is no part of what a run locks.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .. import design, style
from ..reported import reported
from ..step_facts import LEADS, Fact
from .message_row import MessageRow
from .page import Scroller, _ring_room
from .ruled import SectionHead

BEFORE_YOU_START = "Before you start"
CHECK_AGAIN = "Check again"
GO_TO_THE_RUN = "Go to the run"
SHOW_ALL = "Show all {}"
SHOW_FEWER = "Show fewer"
#: The most height the facts take, in pixels at the ordinary text size.
FACTS_CAP = 200
#: The markers of the facts a low window keeps in sight.
NEEDS_ATTENTION = ("err", "lock", "warn")


class Facts(QWidget):
    """The rows of "Before you start". ``hidden`` is how many the low window leaves out of sight just now; ``opened``,
    whether it shows them all."""

    def __init__(self) -> None:
        super().__init__()
        self._facts: tuple[Fact, ...] = ()
        self._rows: list[MessageRow] = []
        self._small = self._all = False
        body = QWidget()
        self._box = QVBoxLayout(body)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(design.scaled(design.GAP["gate_rows"]))
        self.scroller = Scroller(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroller)
        self.setVisible(False)

    def rows(self) -> list:
        """The rows, in their order, those out of sight among them."""
        return list(self._rows)

    def set_facts(self, facts) -> None:
        """Show ``facts`` in the rows there are, in their order: a row is told only what changed, one more is added when
        one is needed and the last go when they are not, so that a reading moves nothing it does not change."""
        facts = tuple(facts)
        if facts == self._facts:
            return
        for at, fact in enumerate(facts):
            if at == len(self._rows):
                self._rows.append(MessageRow(fact.kind, fact.sentence, lead=LEADS[fact.kind]))
                self._box.addWidget(self._rows[at])
            elif self._facts[at] != fact:
                self._rows[at].set_kind(fact.kind)
                self._rows[at].set_sentence(fact.sentence, lead=LEADS[fact.kind])
        for row in self._rows[len(facts):]:
            self._box.removeWidget(row)
            row.hide()
            row.deleteLater()
        del self._rows[len(facts):]
        self._facts = facts
        self._arrange()

    def set_small(self, small: bool) -> None:
        """The low window, or the ordinary one again. Told the form it is in already, it changes nothing: the window
        tells it at every move to another screen."""
        if small != self._small:
            self._small, self._all = small, False
            self._arrange()

    def show_all(self, on: bool) -> None:
        self._all = on
        self._arrange()

    def opened(self) -> bool:
        return self._small and self._all

    def hidden(self) -> int:
        """How many rows the low window keeps out of sight, with the link shut."""
        return sum(fact.kind not in NEEDS_ATTENTION for fact in self._facts) if self._small else 0

    def count(self) -> int:
        return len(self._facts)

    def _arrange(self) -> None:
        few = self._small and not self._all
        for row, fact in zip(self._rows, self._facts):
            row.setVisible(not few or fact.kind in NEEDS_ATTENTION)
        self._fit()

    def _fit(self) -> None:
        """As high as the rows stand at this width, up to the look's height, past which they scroll here. Each row is
        asked at the width it has with no bar beside it: with the bar it stands higher still, which the height stands
        under already. A region with nothing in sight takes no room."""
        shown = [row for row in self._rows if not row.isHidden()]
        self.setVisible(bool(shown))
        if not shown:
            return
        room = _ring_room()
        width = max(1, self.width() - 2 * room)
        gap = self._box.spacing()
        need = sum(row.heightForWidth(width) if row.hasHeightForWidth() else row.sizeHint().height() for row in shown)
        need += gap * (len(shown) - 1) + room
        self.scroller.setFixedHeight(min(need, design.scaled(FACTS_CAP)))

    @reported(QWidget.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class Foot(QWidget):
    """Region 4 of a page. ``main_button`` is the page's one main button, ``check_again`` reads the facts again, ``reason``
    says why the button is greyed, ``go_to_run`` is said when "Go to the run" is pressed."""

    go_to_run = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._live = False
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, design.scaled(design.GAP["gate_above"]), 0, 0)
        box.setSpacing(0)
        self.head = SectionHead(BEFORE_YOU_START)
        self.reading = style.set_class(QLabel(), cls="small")
        self.reading.setTextFormat(Qt.PlainText)
        self.more = self._link("")
        self.more.setVisible(False)
        self.more.clicked.connect(self._more_pressed)
        right = QHBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(design.scaled(design.GAP["action"]))
        right.addWidget(self.reading)
        right.addWidget(self.more)
        beside = QWidget()
        beside.setLayout(right)
        self.head.add_right(beside)
        box.addWidget(self.head)
        box.addSpacing(design.scaled(design.GAP["gate_rows"]))
        self.facts = Facts()
        box.addWidget(self.facts)
        box.addSpacing(design.scaled(design.GAP["gate_action"]))
        line = QHBoxLayout()
        line.setSpacing(design.scaled(design.GAP["action"]))
        self.reason = QLabel()
        self.reason.setTextFormat(Qt.PlainText)
        self.reason.setWordWrap(True)
        self.reason.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.run_link = self._link(GO_TO_THE_RUN)
        self.run_link.setVisible(False)
        self.run_link.clicked.connect(self.go_to_run)
        self.check_again = QPushButton(CHECK_AGAIN)
        self.main_button = style.set_class(QPushButton(), cls="primary")
        line.addWidget(self.reason, 1)
        line.addWidget(self.run_link)
        line.addWidget(self.check_again)
        line.addWidget(self.main_button)
        box.addLayout(line)

    @staticmethod
    def _link(words: str) -> QPushButton:
        link = style.set_class(QPushButton(words), cls="link")
        link.setCursor(Qt.PointingHandCursor)
        link.setFixedHeight(design.scaled(design.SIZE["line_h"]))
        link.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        return link

    def stops(self) -> list:
        """What the Tab key stops at in the foot, after the facts, in its order."""
        return [self.more, self.run_link, self.check_again, self.main_button]

    def set_facts(self, facts) -> None:
        """Show ``facts`` under the heading."""
        self.facts.set_facts(facts)
        self._say_link()

    def set_reading(self, words: str) -> None:
        """Say when the saved records were read, at the right of the heading."""
        self.reading.setText(words)

    def set_main_words(self, words: str) -> None:
        """Name the main button for its outcome, with its access key."""
        self.main_button.setText(style.access_key(words))

    def set_main_enabled(self, on: bool, reason: str) -> None:
        """The main button greyed or not, ``reason`` beside it and its accessible description."""
        self.main_button.setEnabled(on and not self._live)
        self.reason.setText(reason)
        self.main_button.setAccessibleDescription(reason)

    def set_live(self, live: bool) -> None:
        """While a run is live the two buttons are greyed, and the way to the run stands beside the reason."""
        self._live = live
        self.check_again.setEnabled(not live)
        if live:
            self.main_button.setEnabled(False)
        self.run_link.setVisible(live)

    def set_small(self, on: bool) -> None:
        """The low window, or the ordinary one again."""
        self.facts.set_small(on)
        self._say_link()

    def _more_pressed(self) -> None:
        self.facts.show_all(not self.facts.opened())
        self._say_link()

    def _say_link(self) -> None:
        """"Show all 5" while the low window leaves something out, "Show fewer" while it shows all, and no link when
        nothing is left out."""
        hidden = self.facts.hidden()
        self.more.setVisible(bool(hidden))
        self.more.setText(SHOW_FEWER if self.facts.opened() else SHOW_ALL.format(self.facts.count()))
