"""The run's page: what a run is doing, read from its model and drawn with the look's devices, in the place of a
page's results while it runs and after it has stopped or failed.

The page shows the run's stages with the chain's markers and words, the one sentence that says where the run
stands, the time axis of the stage running with the line under it and the line that names what follows, the
training's health, the fit's loss by epoch from its first record on, the messages newest first and, under them,
the figures the run made, which the page that started the run holds and lends it while it shows (nothing stands
there when it made none). "Show the full text" opens every line the run wrote, which the run's controller hands over
as it says them; "Open the file" and "Open its folder" open the run's text file and the folder of runs, whose path is
never written on the page.

The stop button asks what the run's stage keeps before it stops the run (a run whose stop loses nothing is stopped
with no question); while the stop is pending, "Don't stop after all" withdraws it, and, when the window waits for
the run before it closes, the page says so. A training saving its posterior is not stopped, and the window's close
waits for the save with no stop: "Keep PRISM open" withdraws that close. Once a stop of a training has landed, the
page asks whether to keep the training it kept, never in a box of its own. A run that failed says so on its page,
never in a box: when and where it failed, what was saved and what was lost, the command line it names on a line of its
own, the error's message where it adds to that, and its kind and traceback when asked for. While the window waits to
try a run again, the page says when, and "Stop trying" stands in the stop button's place. The page of a training that
failed or was stopped offers "Continue from batch N" or "Resume the training" while its run's call is kept for it, and
no longer.

It is laid out to fit the column it stands in at any width: its buttons go to rows of their own under the run's name
where the two do not fit one row, as the full text's tools do under its field, and what a run's end leaves to say (its
failure, the question of keeping a training, what the page offers) stands at the head of what scrolls, so that the
page's height never cuts it short, and is brought into sight when it comes.

It is live while the run is, ended once it has ended, and past when it is drawn from a run's files alone. A run
with a view of its own shows the page in its compact form: the sentence and the stop button. The page lives
inside the page that started the run, never in the window's own stack of places. Nothing here draws with the
figure library: the figures it holds are pictures already, and the loss by epoch is painted by the window itself.
"""
import math
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QDesktopServices, QFontMetrics, QKeySequence, QPainter, QTextCharFormat, QTextLayout,
                           QTextOption)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMenu, QPlainTextEdit,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from .. import design, fonts, style
from ..reported import reported
from ..widgets.ask import ask
from ..widgets.log_pane import LINES_KEPT, LogPane
from ..widgets.marker import Marker, painted_icon
from ..widgets.message_row import MessageRow, opening
from ..widgets.page import Scroller
from ..widgets.rule_table import Column, RuleTable, StateCell, TextCell, ValueCell
from ..widgets.ruled import Ruled, SectionHead
from ..widgets.time_axis import AxisCaption, TimeAxis
from . import ends, questions, retry
from .loss import LossCurve
from .model import LEADS, OUTCOMES, RunModel, RunSpec, Stage

SHOW_TEXT = "Show the full text"
HIDE_TEXT = "Hide the full text"
FIND_TEXT = "Find…"
FIND_HINT = "Find in the full text"
OPEN_FILE = "Open the file"
OPEN_FOLDER = "Open its folder"
BACK_TEXT = "Back to the form"
HOW_TEXT = "How the run went"
EARLIER_TEXT = "Earlier lines are in the file."
#: The past view's sentence for a run whose history file cannot be read, or drawn.
UNREADABLE_TEXT = "This run's history file cannot be read, so its page cannot be drawn."
HEALTH_TEXT = "Training health"
MESSAGES_TEXT = "Messages"
#: The stop button's words for a run whose page names none.
STOP_TEXT = "Stop"
#: The lines of the full text in sight at once, at the ordinary text size.
FULL_TEXT_LINES = 16
#: The lines of a failure's full error in sight at once, at the ordinary text size.
FULL_ERROR_LINES = 12
#: The least height of the figures the page holds, in pixels at the ordinary text size.
FIGURES_LEAST_H = 320
#: The least width of the page while it shows in full, in pixels at the ordinary text size: the room the window's
#: least size gives the column it stands in. Every part of the page is laid out to fit it.
LEAST_W = 326
#: The least room the head keeps for the run's name beside its buttons, in pixels at the ordinary text size: a short
#: name whole. When the name at this room and the buttons do not fit one row, the buttons go to rows of their own.
NAME_LEAST_W = 160
#: The marker beside each state word of the health table.
STATE_MARKERS = {"Fine": "fine", "Look closer": "warn", "Not compared": "warn"}
_STAMP = "%Y-%m-%d %H:%M:%S"


def _local_now() -> datetime:
    """The computer's clock, as local time with its zone."""
    return datetime.now().astimezone()


def _value(now: str) -> ValueCell:
    """A health figure as its cell draws it: a leading number semibold and what follows it plain, or words
    alone plain."""
    number, _space, rest = now.partition(" ")
    if number and number.replace(",", "").replace(".", "", 1).lstrip("-").isdigit():
        return ValueCell(number, rest)
    return ValueCell("", now)


def _cells(row) -> list:
    """One row of the health table: what is measured, its figure now and its state with its marker."""
    marker = STATE_MARKERS.get(row.state)
    return [TextCell(row.measure, wrap=True), _value(row.now),
            StateCell(marker, row.state) if marker else TextCell(row.state)]


def _entries(path: Path) -> list:
    """A run's text file as ``(moment, level, text)`` for each line written, a text's further lines joined back
    under it; the moment as the computer's local time. Empty when the file cannot be read."""
    from .textfile import STAMP_WIDTH
    out = []
    try:
        written = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    for line in written:
        if line.startswith(" " * STAMP_WIDTH) and out:
            moment, level, text = out[-1]
            out[-1] = (moment, level, f"{text}\n{line[STAMP_WIDTH:]}")
            continue
        try:
            moment = datetime.strptime(line[:STAMP_WIDTH - 1], _STAMP).astimezone()
            level, _space, text = line[STAMP_WIDTH:].partition(" ")
        except ValueError:
            continue
        out.append((moment, level, text))
    return out


class _StageLine(QWidget):
    """The run's stages on one line, each a marker, its name and its words; two to a line when one line is too
    narrow for them all, and one to a line when two are too wide for it."""

    def __init__(self) -> None:
        super().__init__()
        self._items: list = []               # (holder, marker, name, words)
        self._said: tuple = ()
        self._columns = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(design.scaled(design.GAP["stages"][0]))
        self._grid.setVerticalSpacing(design.scaled(design.GAP["stages"][1]))
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)          # its width is the page's

    def show_words(self, words) -> None:
        """Show these stages, each a ``StageWord``: its marker, its name and its words."""
        said = tuple((word.name, word.marker, word.word) for word in words)
        if said == self._said:
            return
        if [name for name, _m, _w in said] != [name for name, _m, _w in self._said]:
            for holder, *_parts in self._items:
                self._grid.removeWidget(holder)
                holder.deleteLater()
            self._items, self._columns = [], 0
            for name, marker, _word in said:
                holder = QWidget()
                row = QHBoxLayout(holder)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(design.scaled(design.GAP["stage_parts"]))
                shown = Marker(marker)
                label = QLabel(name)
                label.setTextFormat(Qt.PlainText)
                label.setFont(fonts.inter("strong"))
                words_label = QLabel()
                words_label.setTextFormat(Qt.PlainText)
                # its digits of equal width: a pace or a count that changes in place keeps the words after it still
                words_label.setFont(fonts.inter("body", tabular=True))
                for part in (shown, label, words_label):
                    row.addWidget(part, 0, Qt.AlignVCenter)
                self._items.append((holder, shown, label, words_label))
        for (_holder, shown, _label, words_label), (_name, marker, word) in zip(self._items, said):
            shown.set_kind(marker)
            words_label.setText(word)
            style.set_class(words_label, cls="" if marker == "now" else "quiet")
        self._said = said
        self._fit()

    def words(self) -> tuple:
        """The stages shown, each ``(name, marker, words)``."""
        return tuple((label.text(), shown.kind, words.text()) for _holder, shown, label, words in self._items)

    def _fit(self) -> None:
        """One line when the stages fit the width, else two to a line when two fit it, else one to a line."""
        if not self._items:
            return
        gap = self._grid.horizontalSpacing()
        wide = [holder.sizeHint().width() for holder, *_parts in self._items]
        if sum(wide) + (len(wide) - 1) * gap <= self.width():
            columns = len(wide)
        else:
            columns = 2 if max(wide[0::2]) + gap + max(wide[1::2], default=0) <= self.width() else 1
        if columns == self._columns:
            return
        for holder, *_parts in self._items:
            self._grid.removeWidget(holder)
        for index, (holder, *_parts) in enumerate(self._items):
            self._grid.addWidget(holder, index // columns, index % columns, Qt.AlignLeft)
        for column in range(len(self._items) + 1):
            self._grid.setColumnStretch(column, 1 if column == columns else 0)
        self._columns = columns

    @reported(QWidget.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class _Name(QLabel):
    """The run's name in the page's head, cut at its end with "…" where the room it has is narrower than it; while it
    is cut, its tooltip is the whole name, and while it is whole it has none."""

    def shown(self) -> str:
        """The name as it is drawn in the room it has now: whole, or cut at its end with "…"."""
        return QFontMetrics(self.font()).elidedText(self.text(), Qt.ElideRight, self.contentsRect().width())

    def setText(self, text: str) -> None:
        """Show the name ``text``."""
        super().setText(text)
        self._tell_whole()

    @reported(QLabel.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._tell_whole()

    def _tell_whole(self) -> None:
        """The whole name as the tooltip while it is cut; none while it is whole."""
        self.setToolTip(self.text() if self.shown() != self.text() else "")

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the label
            self.style().drawItemText(painter, self.contentsRect(), int(self.alignment()), self.palette(),
                                      self.isEnabled(), self.shown(), self.foregroundRole())
        finally:
            painter.end()


class _Head(QWidget):
    """The page's head: the run's name at the left and its buttons at the right, each button at its own width. They
    stand on one row while the name, at its least room, and the buttons in sight fit side by side; else the name
    stands alone on its row and the buttons on rows of their own under it, at the right, as many to a row as fit.
    The head never asks for more width than its widest button."""

    def __init__(self, title: QLabel, buttons) -> None:
        super().__init__()
        self._title, self._buttons = title, tuple(buttons)
        self._plan = None
        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(design.scaled(design.GAP["head_table"]))
        # The name's row: the room the buttons leave goes to the name, or, where the name is not shown (the compact
        # form), to the stretch after it: never to a button, which keeps its own width at the right.
        self._first = QHBoxLayout()
        self._first.setSpacing(design.scaled(design.GAP["head_buttons"]))
        self._first.addWidget(title, 1)
        self._first.addStretch(0)
        self._box.addLayout(self._first)
        self._rows = []                      # the rows of buttons under the name's, as many as there are buttons
        for _button in self._buttons:
            row = QHBoxLayout()
            row.setSpacing(design.scaled(design.GAP["head_buttons"]))
            row.addStretch(1)
            self._box.addLayout(row)
            self._rows.append(row)
        for button in self._buttons:
            self._first.addWidget(button, 0, Qt.AlignTop)

    @reported(QWidget.minimumSizeHint)
    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        widest = max((button.sizeHint().width() for button in self._buttons if not button.isHidden()), default=0)
        return QSize(widest, hint.height())

    @reported(QWidget.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def fit(self) -> None:
        """Lay the name and the buttons in sight out for the head's width now."""
        shown = [button for button in self._buttons if not button.isHidden()]
        gap, width = design.scaled(design.GAP["head_buttons"]), self.width()
        wide = [button.sizeHint().width() for button in shown]
        least = min(self._title.sizeHint().width(), design.scaled(NAME_LEAST_W))
        if self._title.isHidden() or least + sum(gap + one for one in wide) <= width:
            plan = (tuple(shown),)
        else:
            plan, row, used = [()], [], 0
            for button, one in zip(shown, wide):
                if row and used + gap + one > width:
                    plan.append(tuple(row))
                    row, used = [], 0
                used += (gap if row else 0) + one
                row.append(button)
            plan = tuple(plan + [tuple(row)])
        if plan == self._plan:
            return
        for button in self._buttons:
            for row in (self._first, *self._rows):
                row.removeWidget(button)
        for row, buttons in zip((self._first, *self._rows), plan):
            for button in buttons:
                row.addWidget(button, 0, Qt.AlignTop)
        self._plan = plan


class _Tools(QWidget):
    """The full text's tools: the field that finds words in it, while it is open, then its buttons. They stand on one
    row while every one in sight fits it at the width it asks, the field taking the room the buttons leave and the
    first button standing beside it, the others at the right; else the field stands alone on a row of its own, as wide
    as the row, and the buttons on rows under it, as many to a row as fit, each at its own width, from the left."""

    def __init__(self, field: QLineEdit, buttons) -> None:
        super().__init__()
        self._field, self._buttons = field, tuple(buttons)
        self._plan = ()                      # None: one row; else the buttons of each row under the field's
        self._gap = design.scaled(design.GAP["action"])
        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(design.scaled(design.GAP["head_table"]))
        # The field's row: the field takes the room the buttons leave; while it is shut, the stretch after the first
        # button takes it, so that the others stand at the right.
        self._first = QHBoxLayout()
        self._first.setSpacing(self._gap)
        self._first.addWidget(field, 1)
        self._first.addStretch(0)
        self._box.addLayout(self._first)
        self._rows = []                      # the rows of buttons under the field's, as many as there are buttons
        for _button in self._buttons:
            row = QHBoxLayout()
            row.setSpacing(self._gap)
            row.addStretch(1)
            self._box.addLayout(row)
            self._rows.append(row)
        self.fit()

    @reported(QWidget.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def fit(self) -> None:
        """Lay the field, while it is open, and the buttons in sight out for the tools' width now."""
        width = self.width()
        shown = [button for button in self._buttons if not button.isHidden()]
        wide = [part.sizeHint().width() for part in ([] if self._field.isHidden() else [self._field]) + shown]
        if sum(wide) + self._gap * (len(wide) - 1) <= width:
            plan = None                      # one row
        else:
            plan, row, used = [], [], 0
            for button in shown:
                one = button.sizeHint().width()
                if row and used + self._gap + one > width:
                    plan.append(tuple(row))
                    row, used = [], 0
                used += (self._gap if row else 0) + one
                row.append(button)
            plan = tuple(plan + [tuple(row)])
        if plan == self._plan:
            return
        for button in self._buttons:
            for row in (self._first, *self._rows):
                row.removeWidget(button)
        if plan is None:
            first, *others = self._buttons
            self._first.insertWidget(1, first)
            for button in others:
                self._first.addWidget(button)
        else:
            for row, buttons in zip(self._rows, plan):
                for button in buttons:
                    row.insertWidget(row.count() - 1, button)
        self._plan = plan


class _Wrapped(QLabel):
    """Words wrapped within the page's width: between words where they can be, and inside a word wider than the line,
    a long name or path most often, where they must, with nothing added to the words; a line break of the words' own
    stays. Shown as written, in the font the label is given; never wider than the page. The words are laid out once
    for one text, font and width, and that layout is kept: a repaint while the page scrolls, or a move of a drag, lays
    nothing out again."""

    def __init__(self) -> None:
        super().__init__()
        self.setTextFormat(Qt.PlainText)
        policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self._kept = None                    # the last layout, with the text, the font and the width it was laid for

    def _laid(self, width: float) -> QTextLayout:
        """The words laid out in lines no wider than ``width``, from the top left: the layout kept, when it was laid
        for the same text, font and width."""
        asked = (self.text(), self.font(), width)
        if self._kept is not None and self._kept[0] == asked:
            return self._kept[1]
        layout = QTextLayout(self.text().replace("\n", chr(0x2028)), self.font())      # one character for one
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        layout.beginLayout()
        top = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(1.0, width))
            line.setPosition(QPointF(0, top))
            top += line.height()
        layout.endLayout()
        self._kept = (asked, layout)
        return layout

    def hasHeightForWidth(self) -> bool:
        return True

    @reported(QLabel.heightForWidth)
    def heightForWidth(self, width):
        if not self.text():
            return 0
        margins = self.contentsMargins()
        layout = self._laid(width - margins.left() - margins.right())
        last = layout.lineAt(layout.lineCount() - 1)
        return math.ceil(last.y() + last.height()) + margins.top() + margins.bottom()

    @reported(QLabel.sizeHint)
    def sizeHint(self):
        widest = max((QFontMetrics(self.font()).horizontalAdvance(line) for line in self.text().split("\n")),
                     default=0)
        return QSize(widest, self.heightForWidth(widest))

    @reported(QLabel.minimumSizeHint)
    def minimumSizeHint(self):
        return QSize(0, self.heightForWidth(self.sizeHint().width()))

    def _marked(self) -> list:
        """What of the words is drawn marked: nothing here."""
        return []

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the label
            room = self.contentsRect()
            painter.setPen(self.palette().color(self.foregroundRole()))
            self._laid(room.width()).draw(painter, QPointF(room.topLeft()), self._marked())
        finally:
            painter.end()


class _Lines(_Wrapped):
    """Wrapped words in the fixed-width face that a person selects with the mouse, a double click selecting them
    whole, and copies as written: the keyboard's copy and select-all and the menu of the right button do that, the
    menu the toolkit's own for a selectable label. A selection is counted as the text layout counts the words, in
    UTF-16 units (two for a character outside the Basic Multilingual Plane), and ends when the keyboard goes to
    another control, but not when the window or a popup takes it. A command so copied is one line, as it was
    written."""

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.ClickFocus)
        self.setCursor(Qt.IBeamCursor)
        self._anchor = self._cursor = 0

    def setText(self, text: str) -> None:
        """Show ``text``; nothing is selected."""
        super().setText(text)
        self._anchor = self._cursor = 0
        self.update()

    def _units(self) -> int:
        """How long the words are, in the units the text layout counts."""
        return len(self.text().encode("utf-16-le")) // 2

    def selectedText(self) -> str:
        """The words selected, as written."""
        start, end = sorted((self._anchor, self._cursor))
        return self.text().encode("utf-16-le")[2 * start:2 * end].decode("utf-16-le")

    def hasSelectedText(self) -> bool:
        return self._anchor != self._cursor

    def setSelection(self, start: int, length: int) -> None:
        """Select ``length`` units of the words from ``start``, counted as the text layout counts them."""
        self._anchor = max(0, min(start, self._units()))
        self._cursor = max(0, min(start + length, self._units()))
        self.update()

    def _marked(self) -> list:
        """The selection, drawn in the colours of a selection."""
        start, end = sorted((self._anchor, self._cursor))
        if end == start:
            return []
        marked = QTextLayout.FormatRange()
        marked.start, marked.length = start, end - start
        shown = QTextCharFormat()
        shown.setBackground(self.palette().highlight())
        shown.setForeground(self.palette().highlightedText())
        marked.format = shown
        return [marked]

    def _place(self, point) -> int:
        """The place in the words nearest ``point``."""
        room = self.contentsRect()
        layout = self._laid(room.width())
        for index in range(layout.lineCount()):
            line = layout.lineAt(index)
            if point.y() - room.top() < line.y() + line.height() or index == layout.lineCount() - 1:
                return line.xToCursor(point.x() - room.left())
        return 0

    @reported(QLabel.mousePressEvent)
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self._anchor = self._cursor = self._place(event.position())
        self.update()

    @reported(QLabel.mouseMoveEvent)
    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.LeftButton:
            super().mouseMoveEvent(event)
            return
        self._cursor = self._place(event.position())
        self.update()

    @reported(QLabel.mouseDoubleClickEvent)
    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        self._select_all()

    @reported(QLabel.keyPressEvent)
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self._copy()
        elif event.matches(QKeySequence.SelectAll):
            self._select_all()
        else:
            super().keyPressEvent(event)

    @reported(QLabel.focusOutEvent)
    def focusOutEvent(self, event):
        if event.reason() not in (Qt.ActiveWindowFocusReason, Qt.PopupFocusReason):
            self.setSelection(0, 0)          # the selection ends with the keyboard, as a selectable label's does
        super().focusOutEvent(event)

    @reported(QLabel.contextMenuEvent)
    def contextMenuEvent(self, event):
        """The toolkit's menu for a selectable label: "Copy", greyed with nothing selected, a line, then "Select All",
        greyed for no words; a popup that is deleted once it closes."""
        menu = QMenu(self)
        menu.setAttribute(Qt.WA_DeleteOnClose)
        copy = menu.addAction("&Copy\t" + QKeySequence(QKeySequence.Copy).toString(QKeySequence.NativeText))
        copy.setEnabled(self.hasSelectedText())
        copy.triggered.connect(self._copy)
        menu.addSeparator()
        every = menu.addAction("Select All\t" + QKeySequence(QKeySequence.SelectAll).toString(QKeySequence.NativeText))
        every.setEnabled(bool(self.text()))
        every.triggered.connect(self._select_all)
        menu.popup(event.globalPos())

    def _select_all(self) -> None:
        self.setSelection(0, self._units())

    def _copy(self) -> None:
        """The selection to the clipboard, as written."""
        if self.hasSelectedText():
            QApplication.clipboard().setText(self.selectedText())


class RunView(QWidget):
    """The page of one run. ``back`` is said when "Back to the form" is pressed. ``how_link``, "How the run went",
    is the page's own but stands above the figures of the page that started the run, which places it there."""

    back = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mode = "live"
        self._compact = False
        self._model: RunModel | None = None
        self._stopping = False
        self._closing = False
        self._close_saves = False                # the close that waits began with no stop asked: it waits for a save
        self._run: str | None = None             # the run's id, as its status names it
        self._answered = False                   # whether Keep or Discard was answered for the training it kept
        self._failed = False                     # whether the page says its run's failure
        self._quoted = False                     # whether the error's own message adds to what its failure's block says
        self._noted = self._error_shown = False
        self._retrying = ""                      # while the run waits to be tried again: what the page says
        self._refused = ""                       # the refusal of a run that repeated an offer of the page
        self._ended_words = ""                   # once its run has ended: how and when, and what it kept
        self._continue_from: int | None = None   # the batch the page offers to continue its run from
        self._resumable = False                  # whether the page offers to resume the training its run kept
        self._clock = _local_now
        self._text: Path | None = None
        self._folder: Path | None = None
        self._held: QWidget | None = None
        self._axis_shown = False
        self._text_shown = False
        self._shown_health = None
        self._fit_shown = False
        self._shown_messages = None
        self._message_rows: list = []

        # the head: the run's name, and its actions at the right
        self.title = _Name()
        self.title.setTextFormat(Qt.PlainText)               # a name may hold "<" or "&"
        style.set_class(self.title, cls="head")
        self.title.setFont(fonts.inter("head"))
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.text_button = QPushButton(SHOW_TEXT)
        self.text_button.clicked.connect(self._toggle_text)
        self.stop_button = style.set_class(QPushButton(STOP_TEXT), cls="danger")
        self.stop_button.setIcon(painted_icon("stop"))
        # the glyph and the space after it, each scaled by itself and then added
        self.stop_button.setIconSize(QSize(design.scaled(design.SIZE["marker"])
                                           + design.scaled(design.SIZE["icon_gap"]),
                                           design.scaled(design.SIZE["marker"])))
        self.stop_button.clicked.connect(self._stop)
        self.withdraw_button = QPushButton(questions.WITHDRAW_TEXT)
        self.withdraw_button.clicked.connect(self._withdraw)
        self.keep_open_button = QPushButton(questions.KEEP_OPEN_TEXT)
        self.keep_open_button.clicked.connect(self._keep_open)
        self.stop_trying_button = QPushButton(retry.STOP_TRYING_TEXT)
        self.stop_trying_button.clicked.connect(self._stop_trying)
        self.back_button = QPushButton(BACK_TEXT)
        self.back_button.clicked.connect(self.back.emit)
        self.head = _Head(self.title, (self.text_button, self.stop_button, self.withdraw_button, self.keep_open_button,
                                       self.stop_trying_button, self.back_button))

        # where the run stands: its stages, its sentence, the time axis with its line, and what follows
        self.status = Ruled(top=design.SIZE["rule"], gap_top=design.scaled(design.GAP["status"][1]))
        self.stage_line = _StageLine()
        self.sentence = _Wrapped()              # a long name in it is wrapped inside, never let widen the page
        # The lettering face's heading size at the ordinary weight, its digits of equal width: the sentence's
        # counts and times change in place, and its words must not move with them. Given here, at build: a
        # font handed to a widget the sheet has already dressed is dropped at the next change of sheet.
        self.sentence.setFont(fonts.inter("head", 400, tabular=True))
        now = _local_now()
        self.axis = TimeAxis(now, now)
        self.axis_caption = AxisCaption(now, now)
        for label in (self.axis_caption.started, self.axis_caption.words):
            label.setFont(fonts.inter("body", tabular=True))
        self.follows = _Wrapped()               # within the page's width, a long name in it wrapped inside
        self.follows.setFont(fonts.inter("body", tabular=True))
        # under the sentence: what the page says of a stop and a close, and the question of keeping a training a
        # stop kept, which the page asks itself once the stop has landed
        self.notice = QLabel()
        self.closing_line = QLabel(questions.CLOSING_TEXT)
        for label in (self.notice, self.closing_line):
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
        self.keep_block = QWidget()
        keep = QVBoxLayout(self.keep_block)
        keep.setContentsMargins(0, 0, 0, 0)
        keep.setSpacing(design.scaled(design.GAP["head_table"]))
        self.keep_words = QLabel()
        self.keep_words.setFont(fonts.inter("strong"))
        self.keep_beneath = style.set_class(QLabel(), cls="quiet")
        for label in (self.keep_words, self.keep_beneath):
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
        self.keep_button = style.set_class(QPushButton(questions.KEEP_TEXT), cls="primary")
        self.keep_button.clicked.connect(self._keep)
        self.discard_button = style.set_class(QPushButton(questions.DISCARD_TEXT), cls="danger")
        self.discard_button.setIcon(painted_icon("stop"))
        self.discard_button.setIconSize(self.stop_button.iconSize())
        self.discard_button.clicked.connect(self._discard)
        answers = QHBoxLayout()
        answers.setSpacing(design.scaled(design.GAP["action"]))
        answers.addWidget(self.keep_button)
        answers.addWidget(self.discard_button)
        answers.addStretch(1)
        keep.addWidget(self.keep_words)
        keep.addWidget(self.keep_beneath)
        keep.addLayout(answers)
        # at a failure: when and where it failed, what was saved and what was lost, beside the error's marker (the
        # page's sentence says the run failed); the error's own message, quoted in the fixed-width face where it adds
        # to that; where the full error is kept, and the full error itself (its kind and its traceback), shown only
        # when asked for
        self.failure = QWidget()
        failure = QVBoxLayout(self.failure)
        failure.setContentsMargins(0, 0, 0, 0)
        failure.setSpacing(design.scaled(design.GAP["head_table"]))
        self.failure_row = MessageRow("err", "")
        # Console text: the fixed-width face, given at build. Its width is the page's: a long word of a message
        # (a path) is wrapped inside it, never let widen the page.
        self.failure_message = _Lines()
        self.failure_message.setFont(fonts.fixed())
        # The command line a page sets off beneath its sentence (a kept training resumed outside the window): on a line
        # of its own, in the fixed-width face given at build, selectable, so that it is copied whole and nothing of
        # the sentence with it; wrapped within the page, inside a long path where it must, and copied as one line.
        self.command_line = _Lines()
        self.command_line.setFont(fonts.fixed())
        self.noted = style.set_class(QLabel(retry.NOTED_TEXT), cls="quiet")
        self.file_line = style.set_class(QLabel(retry.FILE_TEXT), cls="quiet")
        for label in (self.noted, self.file_line):
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
        self.full_error_button = QPushButton(retry.SHOW_ERROR_TEXT)
        self.full_error_button.clicked.connect(self._toggle_error)
        where = QHBoxLayout()
        where.setSpacing(design.scaled(design.GAP["action"]))
        where.addWidget(self.file_line)
        where.addWidget(self.full_error_button)
        where.addStretch(1)
        self.full_error = QPlainTextEdit()
        self.full_error.setReadOnly(True)
        self.full_error.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.full_error.setFont(fonts.fixed())
        # In a plain holder of the fixed height, as the full text is.
        self._error_holder = QWidget()
        self._error_holder.setFixedHeight(FULL_ERROR_LINES * design.scaled(design.TYPE["body"][1]))
        inside = QVBoxLayout(self._error_holder)
        inside.setContentsMargins(0, 0, 0, 0)
        inside.addWidget(self.full_error)
        for part in (self.failure_row, self.command_line, self.failure_message, self.noted):
            failure.addWidget(part)
        failure.addLayout(where)
        failure.addWidget(self._error_holder)
        # what the page of a training that failed or was stopped offers: its own call again, continuing its
        # simulation set or resuming the training it kept
        self.offers = QWidget()
        offers = QHBoxLayout(self.offers)
        offers.setContentsMargins(0, 0, 0, 0)
        offers.setSpacing(design.scaled(design.GAP["action"]))
        self.continue_button = style.set_class(QPushButton(), cls="primary")
        self.continue_button.clicked.connect(self._continue)
        self.resume_button = style.set_class(QPushButton(retry.RESUME_TEXT), cls="primary")
        self.resume_button.clicked.connect(self._resume)
        offers.addWidget(self.continue_button)
        offers.addWidget(self.resume_button)
        offers.addStretch(1)
        self.said = QWidget()
        self._said = QVBoxLayout(self.said)
        self._said.setContentsMargins(0, design.scaled(design.GAP["now"][1]), 0, 0)
        self._said.setSpacing(design.scaled(design.GAP["head_table"]))
        for part in (self.notice, self.closing_line):
            self._said.addWidget(part)
        # What the page says once its run has ended -- its failure, the question of keeping the training its stop kept,
        # and what it offers -- stands at the head of what scrolls, so that no room the page has cuts it short; in the
        # compact form, which shows nothing that scrolls, under the sentence (``_place_the_end``).
        self.end = QWidget()
        end = QVBoxLayout(self.end)
        end.setContentsMargins(0, 0, 0, 0)
        end.setSpacing(design.scaled(design.GAP["head_table"]))
        for part in (self.failure, self.keep_block, self.offers):
            end.addWidget(part)
        self._end_gap = QWidget()                   # the gap after it at the head of what scrolls
        self._end_gap.setFixedHeight(design.scaled(design.GAP["block"]))
        box = self.status.box
        box.addWidget(self.stage_line)
        box.addSpacing(design.scaled(design.GAP["now"][0]))
        box.addWidget(self.sentence)
        box.addWidget(self.said)
        box.addSpacing(design.scaled(design.GAP["now"][1]))
        for part in (self.axis, self.axis_caption, self.follows):
            box.addWidget(part)

        # what scrolls: what the run's end leaves to say, the training's health, the fit's loss by epoch, the messages,
        # the figures
        body = QWidget()
        self._column = column = QVBoxLayout(body)
        column.setContentsMargins(0, design.scaled(design.GAP["status_to_body"]), 0, 0)
        column.setSpacing(0)
        self.health_head = SectionHead(HEALTH_TEXT)
        # a figure too long for its column goes on to a second line, and more, never cut
        self.health = RuleTable([Column("Measure"), Column("Now", "fit", right=True, wrap=True),
                                 Column("State", "fit")], selectable=False)
        # the gaps after the health's head and after its table, which come and go with them, as the curve's does
        self._health_head_gap, self._health_gap = QWidget(), QWidget()
        self._health_head_gap.setFixedHeight(design.scaled(design.GAP["head_table"]))
        self._health_gap.setFixedHeight(design.scaled(design.GAP["block"]))
        self.loss = LossCurve()
        self._loss_gap = QWidget()                  # the gap after the curve, which comes and goes with it
        self._loss_gap.setFixedHeight(design.scaled(design.GAP["block"]))
        self.messages_head = SectionHead(MESSAGES_TEXT)
        self._messages_gap = QWidget()              # the gap after the messages' head, which comes and goes with it
        self._messages_gap.setFixedHeight(design.scaled(design.GAP["head_table"]))
        self.messages = QWidget()
        self._rows = QVBoxLayout(self.messages)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(0)
        self.figures = QWidget()
        self._figures = QVBoxLayout(self.figures)
        self._figures.setContentsMargins(0, design.scaled(design.GAP["block"]), 0, 0)
        self.figures.setMinimumHeight(design.scaled(FIGURES_LEAST_H))
        for part in (self.end, self._end_gap, self.health_head, self._health_head_gap, self.health, self._health_gap,
                     self.loss, self._loss_gap, self.messages_head, self._messages_gap, self.messages, self.figures):
            column.addWidget(part)
        self.body = Scroller(body)

        # the full text, hidden until asked for
        self.text_block = QWidget()
        lines = QVBoxLayout(self.text_block)
        lines.setContentsMargins(0, design.scaled(design.GAP["block"]), 0, 0)
        lines.setSpacing(design.scaled(design.GAP["head_table"]))
        self.find_field = QLineEdit()
        self.find_field.setPlaceholderText(FIND_HINT)
        self.find_field.returnPressed.connect(self._find)
        self.find_field.setVisible(False)
        self.find_button = QPushButton(FIND_TEXT)
        self.find_button.clicked.connect(self._open_find)
        self.file_button = QPushButton(OPEN_FILE)
        self.file_button.clicked.connect(self._open_file)
        self.folder_button = QPushButton(OPEN_FOLDER)
        self.folder_button.clicked.connect(self._open_folder)
        self.tools = _Tools(self.find_field, (self.find_button, self.file_button, self.folder_button))
        lines.addWidget(self.tools)
        self.full_text = LogPane(head=EARLIER_TEXT)
        # Console text: the fixed-width face, given at build.
        self.full_text.setFont(fonts.fixed())
        # In a plain holder of the fixed height: the sheet gives a text box a least height of its own, which
        # replaces the one a fixed height sets.
        holder = QWidget()
        holder.setFixedHeight(FULL_TEXT_LINES * design.scaled(design.TYPE["body"][1]))
        inside = QVBoxLayout(holder)
        inside.setContentsMargins(0, 0, 0, 0)
        inside.addWidget(self.full_text)
        lines.addWidget(holder)
        self.text_block.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.head)
        outer.addSpacing(design.scaled(design.GAP["status"][0]))
        outer.addWidget(self.status)
        outer.addWidget(self.body, 1)
        outer.addWidget(self.text_block)

        # "How the run went": the page's own, placed above its figures by the page that started the run
        self.how_link = style.set_class(QPushButton(HOW_TEXT, self), cls="link")
        self.how_link.setCursor(Qt.PointingHandCursor)
        self.how_link.setFixedHeight(design.scaled(design.SIZE["line_h"]))
        self.how_link.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.how_link.setVisible(False)
        self._arrange()

    # ── what the page is ──
    @property
    def mode(self) -> str:
        """"live" while its run is, "ended" once it has ended, "past" when it is drawn from a run's files."""
        return self._mode

    @property
    def compact(self) -> bool:
        """Whether the page is in its compact form: the sentence and the stop button, for a run with a view of
        its own."""
        return self._compact

    # ── what the run's controller hands it ──
    def begin(self, spec: RunSpec, *, text: Path | None = None, folder: Path | None = None, clock=None) -> None:
        """A new run of this page: its name and its stop button's words, its text file and the folder of runs
        (None for a run that keeps none), and the clock its moments are read from; its full text emptied. The
        page is live, in its compact form for a run with a view of its own."""
        self._mode, self._compact, self._model, self._stopping = "live", spec.own_view, None, False
        self._place_the_end()
        self._closing, self._run, self._answered = False, None, False
        self._failed = self._noted = self._error_shown = self._resumable = self._close_saves = self._quoted = False
        self._retrying = self._refused = self._ended_words = ""
        self._continue_from = None
        self.failure_row.sentence.set_text("")
        self.failure_message.setText("")
        self.command_line.setText("")
        self.full_error.setPlainText("")
        self.full_error_button.setText(retry.SHOW_ERROR_TEXT)
        self.notice.setText("")
        self._axis_shown = self._text_shown = False
        self.text_button.setText(SHOW_TEXT)
        self._text = None if text is None else Path(text)
        self._folder = None if folder is None else Path(folder)
        self._clock = clock or _local_now
        self.file_button.setEnabled(self._text is not None)
        self.folder_button.setEnabled(self._folder is not None)
        self.title.setText(f"{spec.word} {spec.name}".strip())
        self.stop_button.setText(spec.stop_button or STOP_TEXT)
        self.full_text.clear_lines()
        self.find_field.clear()
        self.find_field.setVisible(False)
        self._shown_health = self._shown_messages = None
        self.health.set_rows([])
        self._fit_shown = False
        self.loss.set_epochs((), patience=spec.patience)
        self._set_messages([], self._clock())
        self.stage_line.show_words(())
        self.sentence.setText("")
        self.how_link.setVisible(False)
        self._arrange()

    def take_lines(self, batch) -> None:
        """Lines of the run's full text, each ``(text, level)``, as the run's controller says them."""
        self.full_text.append_lines(batch)

    def show_status(self, status, model: RunModel | None) -> None:
        """Draw the run as ``model`` reads it now, live or ended as ``status`` says. Once it has ended, its sentence
        is how and when it ended and what it kept (``status.sentence``), but for a failure, whose sentence stands
        beside its error in the failure's block. What the page said of an earlier stop goes when a stop is asked for;
        at a stopped end of a training that kept nothing, the page says so; at an end that the close which waited for
        it is not carried out at (a failure), the page says PRISM stayed open: but for the end of the tries the close
        itself ended, at which it is carried out. A close that waited for the posterior's save (one that began with no
        stop asked) did not end the tries that followed, whoever ended them: at the failed end they leave, the page
        says PRISM stayed open."""
        ending = self._mode == "live" and not status.live
        closed = ending and self._closing and (not self._retrying or self._close_saves)
        if (status.stop_requested and not self._stopping) or ending:
            self.notice.setText("")
        self._model = model
        self._ended_words = "" if status.live or status.outcome == "failed" else status.sentence
        self._mode = "live" if status.live else "ended"
        self._stopping = status.stop_requested
        self._closing = status.closing and status.live
        self._close_saves = self._closing and (self._close_saves or not status.stop_requested)
        self._run = status.run
        if status.state != "retrying":
            self._retrying = ""
        if ending and model is not None and self._kept_nothing(model):
            self.notice.setText(questions.NOT_KEPT_TEXT)
        elif closed and model is not None and model.outcome not in questions.CLOSES_AT:
            self.notice.setText(questions.STAYED_OPEN_TEXT)
        self._draw()

    def say(self, text: str) -> None:
        """Say ``text`` on the page, under its sentence, in place of what it said before ("" says nothing)."""
        self.notice.setText(text)
        self._arrange()

    def show_failure(self, sentence: str, message: str, detail: str, noted: bool, kept: bool,
                     command: str = "") -> None:
        """The run failed: under the page's sentence, which says so, beside the error's marker, ``sentence`` (when and
        where it failed, what was saved and what was lost); the ``command`` line it names, on its own beneath it (""
        for none); the error's ``message``, quoted, where it adds to ``sentence``; that a stop had been asked for
        before it, when ``noted``; that the full error is in the run's file, when the run ``kept`` one; and the full
        error, ``detail`` (its kind and traceback), behind "Show the full error". Shown once the run has ended, until
        the page's next run."""
        self._failed, self._noted, self._error_shown = True, noted, False
        self._quoted = retry.adds_to(sentence, message)
        self.failure_row.sentence.set_text(sentence)
        self.command_line.setText(command)
        self.failure_message.setText(message)
        self.file_line.setText(retry.FILE_TEXT if kept else "")
        self.full_error.setPlainText(detail)
        self.full_error_button.setText(retry.SHOW_ERROR_TEXT)
        self._arrange()

    def show_retrying(self, sentence: str) -> None:
        """The run failed and waits to be tried again: ``sentence`` (what failed, when the next try begins, what is
        kept) is the page's sentence until the try begins, and "Stop trying" stands in the stop button's place."""
        self._retrying = sentence
        if self.sentence.text() != sentence:
            self.sentence.setText(sentence)

    def offer(self, continue_from: int | None, resume: bool) -> None:
        """What the page offers once its run has ended: "Continue from batch N" from ``continue_from`` (None: no
        continuing), and "Resume the training" when ``resume``. Until the page's next run."""
        self._continue_from, self._resumable = continue_from, resume
        if continue_from is not None:
            self.continue_button.setText(retry.CONTINUE_TEXT.format(design.number_text(continue_from)))
        self._arrange()

    def refused(self, sentence: str) -> None:
        """The run the page started from one of its offers was refused before anything was spent: ``sentence``, the
        refusal's own, is the page's sentence, with "Back to the form" beside it."""
        self._refused = sentence
        self.sentence.setText(sentence)
        self._arrange()

    def show_past(self, record: dict, text: Path | None, *, clock=None) -> None:
        """Draw a run that is over from its two files alone: ``record``, its history, and ``text``, its text
        file (None when there is none). Its model, of the stages its history names, reads the file's lines again,
        each at the moment written, so that its stage line and its health are drawn as they stood at its end; its
        sentence says how and when it ended and what it kept, as its history says it, dated against the moment
        ``clock`` gives (the computer's own clock when none is handed)."""
        place = record.get("place")
        stages = []
        for stage in record.get("stages") or ():
            if (isinstance(stage, list) and len(stage) == 3 and isinstance(stage[0], str)
                    and isinstance(stage[1], list) and (stage[2] is None or isinstance(stage[2], str))):
                stages.append(Stage(stage[0], tuple(str(begins) for begins in stage[1]), stage[2]))
        spec = RunSpec(str(record.get("word") or ""), tuple(place) if isinstance(place, list) and len(place) == 2
                       else None, name=str(record.get("name") or ""), stages=tuple(stages))
        entries = [] if text is None else _entries(text)
        written = [entries[0][0] if entries else _local_now()]           # the moment of the line read now
        self.begin(spec, text=text, folder=None if text is None else Path(text).parent, clock=clock)
        model = RunModel(spec, clock=lambda: written[0])
        for moment, level, said in entries:
            written[0] = moment
            model.feed([(said, level)])
        outcome = record.get("outcome")
        model.ended(outcome if outcome in OUTCOMES else "not seen to end")
        self.full_text.append_lines([(said, level) for _moment, level, said in entries[-LINES_KEPT:]])
        self._ended_words = ends.words({**record, "outcome": model.outcome}, self._clock())[0]
        self._model, self._mode, self._stopping = model, "past", False
        self._draw()

    def show_unreadable(self) -> None:
        """A run that is over whose history cannot be read, or drawn: the page says so, in the past view, and draws
        nothing else of it."""
        self.begin(RunSpec("", None))
        self._mode = "past"
        self.sentence.setText(UNREADABLE_TEXT)
        self._arrange()

    def set_count(self, done: int, total: int, unit: str) -> None:
        """A run that prints no counted row says how far it is: ``done`` of ``total`` ``unit``."""
        if self._model is not None:
            self._model.set_count(done, total, unit)
            self._draw()

    def set_stage(self, name: str) -> None:
        """A run that says its stages itself begins the one of that name."""
        if self._model is not None:
            self._model.set_stage(name)
            self._draw()

    def hold_figures(self, widget: QWidget) -> None:
        """Hold the figures of the page that started the run, under the messages, while the page shows: in sight while
        any of them is (the run's own, as the page that holds them shows them), and nothing while none is."""
        if widget is not self._held:
            self.release_figures()
            self._held = widget
            self._figures.addWidget(widget)
        self._arrange()

    def release_figures(self) -> QWidget | None:
        """Hand back the figures held, for their page to place again; None when none are held."""
        widget, self._held = self._held, None
        if widget is not None:
            self._figures.removeWidget(widget)
        self._arrange()
        return widget

    # ── drawing ──
    def _draw(self) -> None:
        model = self._model
        if model is None:
            self._arrange()
            return
        now = self._clock()
        said = self._refused or self._retrying or self._ended_words or model.sentence()
        if self.sentence.text() != said:
            self.sentence.setText(said)
        self.stage_line.show_words(model.stage_line())
        axis = model.axis()
        if axis is not None:
            self.axis.set_times(axis.start, axis.now, axis.end_from, axis.end_by, unknown=axis.unknown)
            self.axis_caption.set_times(axis.start, axis.now, axis.end_from, axis.end_by, unknown=axis.unknown)
            self.follows.setText(axis.follows)
        self._axis_shown = axis is not None
        health = model.health()
        if health != self._shown_health:
            self.health.set_rows([_cells(row) for row in health])
            self._shown_health = health
        epochs = model.epochs()
        self.loss.set_epochs(epochs, best=model.best_epoch(), patience=model.spec.patience)
        self._fit_shown = bool(epochs)
        self._set_messages(model.messages(), now)
        if model.outcome == "stopped" and model.kept_fit is not None:
            words, beneath = questions.keep_question(model.kept_fit)
            self.keep_words.setText(words)
            self.keep_beneath.setText(beneath)
        self._arrange()

    def _set_messages(self, messages, now) -> None:
        """The message rows, newest first, built again only when what they say has changed. Each is one paragraph:
        the line breaks and the runs of spaces of its words one space each, as a library's warning, written for a
        console, comes with them. A message that came more than once says how many times in its opening, unless its
        sentence says that count itself."""
        rows = [(message.kind, " ".join(message.sentence.split()), message.when,
                 1 if message.count_said else message.count) for message in messages]
        said = tuple((kind, sentence, opening(LEADS[kind], when, now, times)) for kind, sentence, when, times in rows)
        if said == self._shown_messages:
            return
        for row in self._message_rows:
            self._rows.removeWidget(row)
            row.deleteLater()
        self._message_rows = [MessageRow(kind, sentence, lead=LEADS[kind], when=when, now=now, times=times)
                              for kind, sentence, when, times in rows]
        for row in self._message_rows:
            self._rows.addWidget(row)
        self._shown_messages = said

    def message_rows(self) -> list:
        """The message rows shown, newest first."""
        return list(self._message_rows)

    def _arrange(self) -> None:
        """What is in sight in the page's mode and form. A control of the page that holds the keyboard and goes, or
        is greyed, first hands it on, for no key of its own (its ring shows while the keyboard has the turn): the
        stop button to "Don't stop after all" while a stop can be withdrawn, and back; "Keep PRISM open", once it has
        withdrawn its close, to the sentence, the stop button being greyed while the posterior is saved; the stop
        button to "Stop trying" while the run waits to be tried again, and back; an offer pressed to the stop button of
        the run it starts; at a stopped end to "Keep" while the training the stop kept is asked about, else to the
        sentence; at any other end to "Back to the form". A keyboard on a control that stays (the full text's, "Show
        the full text") stays where it is. When what a run's end leaves to say comes into sight on the full page, at
        the head of what scrolls, the middle is scrolled to its top, so that it is in sight when it comes."""
        live, full = self._mode == "live", not self._compact
        retrying = live and bool(self._retrying)
        axis = self._axis_shown and live and full and not retrying
        stops = live and not retrying and not self._stopping and not self._saving()
        withdraws = (live and not retrying and self._stopping and self._model is not None
                     and self._model.spec.stop_asks)
        keeps = self._keeps()
        closing, notice = self._closing and live, bool(self.notice.text())
        keeps_open = closing and not self._stopping          # a close that waits for a save, no stop asked
        self.closing_line.setText(questions.CLOSING_SAVED_TEXT if keeps_open else questions.CLOSING_TEXT)
        failed = self._failed and self._mode == "ended"
        continues = self._mode == "ended" and full and self._continue_from is not None
        resumes = self._mode == "ended" and full and self._resumable
        main = "" if keeps else "primary"        # while Keep is asked, Keep is the main button and Resume beside it
        if self.resume_button.property("cls") != main:
            style.set_class(self.resume_button, cls=main)
        ended = keeps or failed or continues or resumes
        shown = ((self.said, closing or notice or (ended and self._compact)), (self.notice, notice),
                 (self.closing_line, closing), (self.end, ended), (self._end_gap, ended and not self._compact),
                 (self.failure, failed), (self.failure_message, failed and self._quoted),
                 (self.command_line, failed and bool(self.command_line.text())), (self.noted, failed and self._noted),
                 (self.file_line, failed and bool(self.file_line.text())),
                 (self._error_holder, failed and self._error_shown), (self.keep_block, keeps),
                 (self.offers, continues or resumes), (self.continue_button, continues),
                 (self.resume_button, resumes), (self.stop_button, live and not retrying),
                 (self.withdraw_button, withdraws), (self.keep_open_button, keeps_open),
                 (self.stop_trying_button, retrying), (self.back_button, not live and full))
        comes = ended and full and self.end.isHidden()       # what the run's end leaves to say comes into sight
        for widget, on in shown:                 # what takes the keyboard is in sight before what holds it goes
            if on:
                widget.setVisible(True)
        if comes:
            self.body.verticalScrollBar().setValue(0)
        if stops:
            self.stop_button.setEnabled(True)
        focus = QApplication.focusWidget()
        going = ((self.stop_button, stops), (self.withdraw_button, withdraws), (self.keep_open_button, keeps_open),
                 (self.keep_button, keeps), (self.discard_button, keeps), (self.stop_trying_button, retrying),
                 (self.continue_button, continues), (self.resume_button, resumes))
        if any(focus is widget and not stays for widget, stays in going):
            self._hand_on(stops, withdraws, keeps, retrying)
        for widget, on in shown:
            if not on:
                widget.setVisible(False)
        self.stop_button.setEnabled(stops)
        self.title.setVisible(full)
        self.text_button.setVisible(full)
        self.stage_line.setVisible(full and bool(self.stage_line.words()))
        self.axis.setVisible(axis)
        self.axis_caption.setVisible(axis)
        self.follows.setVisible(axis and bool(self.follows.text()))
        health = bool(self._shown_health) and full
        for part in (self.health_head, self._health_head_gap, self.health, self._health_gap):
            part.setVisible(health)
        self.loss.setVisible(full and self._fit_shown)
        self._loss_gap.setVisible(full and self._fit_shown)
        messages = bool(self._message_rows) and full
        for part in (self.messages_head, self._messages_gap, self.messages):
            part.setVisible(messages)
        count = getattr(self._held, "in_sight", None) or getattr(self._held, "count", None)
        self.figures.setVisible(full and self._held is not None and (count is None or count() > 0))
        self.body.setVisible(full)
        self.text_block.setVisible(full and self._text_shown)
        self.head.fit()
        self.tools.fit()
        self._hold_width()

    @reported(QWidget.showEvent)
    def showEvent(self, event):
        super().showEvent(event)
        self._hold_width()

    @reported(QWidget.hideEvent)
    def hideEvent(self, event):
        super().hideEvent(event)
        self._hold_width()

    def _place_the_end(self) -> None:
        """What the page says once its run has ended at the head of what scrolls; in the compact form, which shows
        nothing that scrolls, under the sentence."""
        if (self._said.indexOf(self.end) >= 0) == self._compact:
            return
        if self._compact:
            self._column.removeWidget(self.end)
            self._said.addWidget(self.end)
        else:
            self._said.removeWidget(self.end)
            self._column.insertWidget(0, self.end)

    def _hold_width(self) -> None:
        """The page asks for its least width while it shows in full, and for nothing more of its own while it is out
        of sight or compact: its least width is then what its parts ask, and no more."""
        least = design.scaled(LEAST_W) if self.isVisible() and not self._compact else 0
        if self.minimumWidth() != least:
            self.setMinimumWidth(least)

    def _hand_on(self, stops: bool, withdraws: bool, keeps: bool, retrying: bool) -> None:
        """The keyboard, held by a control of the page that goes or is greyed, to where the page's state now puts
        it (``_arrange`` says where). A control inside what scrolls is then made visible there, once the page is laid
        out."""
        if self._mode == "live":
            target = (self.withdraw_button if withdraws else self.stop_trying_button if retrying
                      else self.stop_button if stops else self.sentence)
        elif keeps:
            target = self.keep_button
        elif self._compact or (self._model is not None and self._model.outcome == "stopped"):
            target = self.sentence
        else:
            target = self.back_button
        target.setFocus(Qt.OtherFocusReason)
        if self.body.isAncestorOf(target):
            QTimer.singleShot(0, self, lambda: self._into_sight(target))

    def _into_sight(self, target: QWidget) -> None:
        """``target``, inside what scrolls, made visible there while it is in sight on the page."""
        if target.isVisibleTo(self) and self.body.isVisibleTo(self):
            self.body.ensureWidgetVisible(target)

    def _saving(self) -> bool:
        """Whether the run is saving its posterior, its fit ended: the stop button is greyed then."""
        return self._model is not None and self._model.stage_kind() == "saving"

    def _keeps(self) -> bool:
        """Whether the page asks to keep the training its run's stop kept: once the stop has landed with a note of
        one, until it is answered."""
        model = self._model
        return (self._mode == "ended" and not self._compact and not self._answered and model is not None
                and model.outcome == "stopped" and model.kept_fit is not None)

    @staticmethod
    def _kept_nothing(model: RunModel) -> bool:
        """Whether a training stopped before its fit began, so that its stop kept no training."""
        return (model.outcome == "stopped" and model.kept_fit is None and model.spec.word in questions.TRAINING_WORDS
                and not model.epochs() and model.stage_kind() not in ("fit", "saving"))

    # ── the page's own actions ──
    def _stop(self) -> None:
        """The stop button: the run's question first, for the stage it is in (none for a run whose stop loses
        nothing), then, on its other answer, the live run is asked to stop at its next point where a stop is
        taken. A run that ended while the question was open is not stopped: the page says it was too late."""
        if self._mode != "live" or self._model is None or not self.stop_button.isEnabled():
            return
        from .controller import controller
        held = controller()
        question = questions.stop_question(held.status, self._model)
        if question is not None and not ask(self.window(), question.title, question.sentence, question.safe,
                                            question.other, loses_work=question.loses_work,
                                            beneath=question.beneath or None):
            return
        held.request_stop()

    def _withdraw(self) -> None:
        """"Don't stop after all": the stop asked for is withdrawn, and the close that waits for it."""
        from .controller import controller
        controller().withdraw_stop()

    def _keep_open(self) -> None:
        """"Keep PRISM open": the close that waits for the posterior to be saved is withdrawn; the run goes on as it
        would have."""
        from .controller import controller
        controller().withdraw_close()

    def _stop_trying(self) -> None:
        """"Stop trying": the run that waits to be tried again ends failed; nothing is lost by it, so it asks
        nothing."""
        from .controller import controller
        controller().stop_trying()

    def _continue(self) -> None:
        """"Continue from batch N": the run's own call again, continuing its simulation set, as the page's next run;
        while another run is live, nothing starts and the page says so."""
        from .controller import controller
        self._after_press(controller().continue_run(self._run))

    def _resume(self) -> None:
        """"Resume the training": the run's own call again, resuming the training it kept, as the page's next run;
        pressed while the page asks whether to keep that training, it is the answer Keep (the training stays kept,
        and the run resumes it), and it is drawn then as a secondary button beside "Keep", the main one; a press that
        starts nothing leaves the question asked. While another run is live, nothing starts and the page says so."""
        from .controller import controller
        self._after_press(controller().resume_training(self._run))

    def _after_press(self, started: bool) -> None:
        """An offer pressed: nothing more when its run ``started``; else, while another run is live, the page says a
        run is live; and with none live, the call the offer repeats has gone, so the page offers it no more (the
        controller has said why on the page, where it has a sentence for it)."""
        from .controller import LIVE_TEXT, controller
        if started:
            return
        if controller().is_live():
            self.say(LIVE_TEXT)
        else:
            self.offer(None, False)

    def _keep(self) -> None:
        """"Keep": the training the stop kept stays kept, as a question nobody answers leaves it, and the question
        goes."""
        self._answered = True
        self._arrange()

    def _discard(self) -> None:
        """"Discard": the training the stop kept is deleted, or found gone already, and the page says which, asking
        no more, and its sentence says how its run ended again, the training gone, in the words the run's controller
        hands back; or the page says why it was not deleted, and the question stays."""
        from .controller import controller
        answer = controller().discard(self._run)
        if answer is None:
            return
        gone, words, ended = answer
        if gone:
            self._answered, self._resumable = True, False      # nothing kept is left to resume
            if ended:
                self._ended_words = ended
                self._draw()
        self.say(words)

    def _toggle_text(self) -> None:
        """Show the full text, or hide it again."""
        self._text_shown = not self._text_shown
        self.text_button.setText(HIDE_TEXT if self._text_shown else SHOW_TEXT)
        self._arrange()

    def _toggle_error(self) -> None:
        """Show a failure's full error, its kind and traceback, or hide it again."""
        self._error_shown = not self._error_shown
        self.full_error_button.setText(retry.HIDE_ERROR_TEXT if self._error_shown else retry.SHOW_ERROR_TEXT)
        self._arrange()

    def _open_find(self) -> None:
        """Open the field that finds words in the full text, with the keyboard in it."""
        self.find_field.setVisible(True)
        self.tools.fit()
        self.find_field.setFocus(Qt.OtherFocusReason)

    def _find(self) -> None:
        """Select the next place in the full text that holds the words typed."""
        self.full_text.find_next(self.find_field.text())

    def _open_file(self) -> None:
        """Open the run's text file in the program the computer opens such a file with."""
        if self._text is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._text)))

    def _open_folder(self) -> None:
        """Open the folder of runs, where the run's text file and its history are kept."""
        if self._folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._folder)))
