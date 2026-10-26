"""The run's page: what a run is doing, read from its model and drawn with the look's devices, in the place of a
page's results while it runs and after it has stopped or failed.

The page shows the run's stages with the chain's markers and words, the one sentence that says where the run
stands, the time axis of the stage running with the line under it and the line that names what follows, the
training's health, the fit's loss by epoch from its first record on, the messages newest first and, under them,
the figures the page that started the run holds and lends it while it shows. "Show the full text" opens every
line the run wrote, which the run's controller hands over as it says them; "Open the file" and "Open its folder"
open the run's text file and the folder of runs, whose path is never written on the page.

It is live while the run is, ended once it has ended, and past when it is drawn from a run's files alone. A run
with a view of its own shows the page in its compact form: the sentence and the stop button. The page lives
inside the page that started the run, never in the window's own stack of places. Nothing here draws with the
figure library: the figures it holds are pictures already, and the loss by epoch is painted by the window itself.
"""
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from .. import design, fonts, style
from ..reported import reported
from ..widgets.log_pane import LINES_KEPT, LogPane
from ..widgets.marker import Marker, painted_icon
from ..widgets.message_row import MessageRow, opening
from ..widgets.page import Scroller
from ..widgets.rule_table import Column, RuleTable, StateCell, TextCell, ValueCell
from ..widgets.ruled import Ruled, SectionHead
from ..widgets.time_axis import AxisCaption, TimeAxis
from .loss import LossCurve
from .model import LEADS, OUTCOMES, RunModel, RunSpec

SHOW_TEXT = "Show the full text"
HIDE_TEXT = "Hide the full text"
FIND_TEXT = "Find…"
FIND_HINT = "Find in the full text"
OPEN_FILE = "Open the file"
OPEN_FOLDER = "Open its folder"
BACK_TEXT = "Back to the form"
HOW_TEXT = "How the run went"
EARLIER_TEXT = "Earlier lines are in the file."
HEALTH_TEXT = "Training health"
MESSAGES_TEXT = "Messages"
#: The stop button's words for a run whose page names none.
STOP_TEXT = "Stop"
#: The lines of the full text in sight at once, at the ordinary text size.
FULL_TEXT_LINES = 16
#: The least height of the figures the page holds, in pixels at the ordinary text size.
FIGURES_LEAST_H = 320
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
    narrow for them all."""

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
        """One line when the stages fit the width, else two to a line."""
        if not self._items:
            return
        gap = self._grid.horizontalSpacing()
        need = sum(holder.sizeHint().width() for holder, *_parts in self._items) + (len(self._items) - 1) * gap
        columns = len(self._items) if need <= self.width() or len(self._items) <= 2 else 2
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
        self.head = QWidget()
        head = QHBoxLayout(self.head)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(design.scaled(design.GAP["head_buttons"]))
        self.title = QLabel()
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
        self.back_button = QPushButton(BACK_TEXT)
        self.back_button.clicked.connect(self.back.emit)
        head.addWidget(self.title, 1)
        # The room the buttons leave goes to the title, or, in the compact form, where the title is not shown, to
        # this stretch: never to a button, which keeps its own width at the right.
        head.addStretch(0)
        for button in (self.text_button, self.stop_button, self.back_button):
            head.addWidget(button, 0, Qt.AlignTop)

        # where the run stands: its stages, its sentence, the time axis with its line, and what follows
        self.status = Ruled(top=design.SIZE["rule"], gap_top=design.scaled(design.GAP["status"][1]))
        self.stage_line = _StageLine()
        self.sentence = QLabel()
        self.sentence.setTextFormat(Qt.PlainText)
        self.sentence.setWordWrap(True)
        # The lettering face's heading size at the ordinary weight, its digits of equal width: the sentence's
        # counts and times change in place, and its words must not move with them. Given here, at build: a
        # font handed to a widget the sheet has already dressed is dropped at the next change of sheet.
        self.sentence.setFont(fonts.inter("head", 400, tabular=True))
        now = _local_now()
        self.axis = TimeAxis(now, now)
        self.axis_caption = AxisCaption(now, now)
        for label in (self.axis_caption.started, self.axis_caption.words):
            label.setFont(fonts.inter("body", tabular=True))
        self.follows = QLabel()
        self.follows.setTextFormat(Qt.PlainText)
        self.follows.setFont(fonts.inter("body", tabular=True))
        box = self.status.box
        box.addWidget(self.stage_line)
        box.addSpacing(design.scaled(design.GAP["now"][0]))
        box.addWidget(self.sentence)
        box.addSpacing(design.scaled(design.GAP["now"][1]))
        for part in (self.axis, self.axis_caption, self.follows):
            box.addWidget(part)

        # what scrolls: the training's health, the fit's loss by epoch, the messages, the figures
        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, design.scaled(design.GAP["status_to_body"]), 0, 0)
        column.setSpacing(0)
        self.health_head = SectionHead(HEALTH_TEXT)
        self.health = RuleTable([Column("Measure"), Column("Now", "fit", right=True), Column("State", "fit")],
                                selectable=False)
        self.loss = LossCurve()
        self._loss_gap = QWidget()                  # the gap after the curve, which comes and goes with it
        self._loss_gap.setFixedHeight(design.scaled(design.GAP["block"]))
        self.messages_head = SectionHead(MESSAGES_TEXT)
        self.messages = QWidget()
        self._rows = QVBoxLayout(self.messages)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(0)
        self.figures = QWidget()
        self._figures = QVBoxLayout(self.figures)
        self._figures.setContentsMargins(0, design.scaled(design.GAP["block"]), 0, 0)
        self.figures.setMinimumHeight(design.scaled(FIGURES_LEAST_H))
        for part, after in ((self.health_head, "head_table"), (self.health, "block"), (self.loss, None),
                            (self._loss_gap, None), (self.messages_head, "head_table"), (self.messages, None),
                            (self.figures, None)):
            column.addWidget(part)
            if after:
                column.addSpacing(design.scaled(design.GAP[after]))
        self.body = Scroller(body)

        # the full text, hidden until asked for
        self.text_block = QWidget()
        lines = QVBoxLayout(self.text_block)
        lines.setContentsMargins(0, design.scaled(design.GAP["block"]), 0, 0)
        lines.setSpacing(design.scaled(design.GAP["head_table"]))
        tools = QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(design.scaled(design.GAP["action"]))
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
        tools.addWidget(self.find_field, 1)
        tools.addWidget(self.find_button)
        tools.addStretch(0)
        tools.addWidget(self.file_button)
        tools.addWidget(self.folder_button)
        lines.addLayout(tools)
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
        """Draw the run as ``model`` reads it now, live or ended as ``status`` says. A page that holds the keyboard
        when its run ends gives it to "Back to the form", for no key of its own: the stop button that held it is
        gone."""
        focus = QApplication.focusWidget()
        keys = self._mode == "live" and not status.live and focus is not None and self.isAncestorOf(focus)
        self._model = model
        self._mode = "live" if status.live else "ended"
        self._stopping = status.stop_requested
        self._draw()
        if keys and self.back_button.isVisibleTo(self):
            self.back_button.setFocus(Qt.OtherFocusReason)

    def show_past(self, record: dict, text: Path | None) -> None:
        """Draw a run that is over from its two files alone: ``record``, its history, and ``text``, its text
        file (None when there is none). Its model reads the file's lines again, each at the moment written."""
        place = record.get("place")
        spec = RunSpec(str(record.get("word") or ""), tuple(place) if isinstance(place, list) and len(place) == 2
                       else None, name=str(record.get("name") or ""))
        entries = [] if text is None else _entries(text)
        written = [entries[0][0] if entries else _local_now()]           # the moment of the line read now
        self.begin(spec, text=text, folder=None if text is None else Path(text).parent)
        model = RunModel(spec, clock=lambda: written[0])
        for moment, level, said in entries:
            written[0] = moment
            model.feed([(said, level)])
        outcome = record.get("outcome")
        model.ended(outcome if outcome in OUTCOMES else "not seen to end")
        self.full_text.append_lines([(said, level) for _moment, level, said in entries[-LINES_KEPT:]])
        self._model, self._mode, self._stopping = model, "past", False
        self._draw()

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
        """Hold the figures of the page that started the run, under the messages, while the page shows."""
        if widget is self._held:
            return
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
        said = model.sentence()
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
        self._arrange()

    def _set_messages(self, messages, now) -> None:
        """The message rows, newest first, built again only when what they say has changed."""
        said = tuple((message.kind, message.sentence, opening(LEADS[message.kind], message.when, now))
                     for message in messages)
        if said == self._shown_messages:
            return
        for row in self._message_rows:
            self._rows.removeWidget(row)
            row.deleteLater()
        self._message_rows = [MessageRow(message.kind, message.sentence, lead=LEADS[message.kind],
                                         when=message.when, now=now) for message in messages]
        for row in self._message_rows:
            self._rows.addWidget(row)
        self._shown_messages = said

    def message_rows(self) -> list:
        """The message rows shown, newest first."""
        return list(self._message_rows)

    def _arrange(self) -> None:
        """What is in sight in the page's mode and form."""
        live, full = self._mode == "live", not self._compact
        axis = self._axis_shown and live and full
        self.title.setVisible(full)
        self.text_button.setVisible(full)
        self.stop_button.setVisible(live)
        self.stop_button.setEnabled(live and not self._stopping)
        self.back_button.setVisible(not live and full)
        self.stage_line.setVisible(full and bool(self.stage_line.words()))
        self.axis.setVisible(axis)
        self.axis_caption.setVisible(axis)
        self.follows.setVisible(axis and bool(self.follows.text()))
        health = bool(self._shown_health) and full
        self.health_head.setVisible(health)
        self.health.setVisible(health)
        self.loss.setVisible(full and self._fit_shown)
        self._loss_gap.setVisible(full and self._fit_shown)
        messages = bool(self._message_rows) and full
        self.messages_head.setVisible(messages)
        self.messages.setVisible(messages)
        count = getattr(self._held, "count", None)
        self.figures.setVisible(full and self._held is not None and (count is None or count() > 0))
        self.body.setVisible(full)
        self.text_block.setVisible(full and self._text_shown)

    # ── the page's own actions ──
    def _stop(self) -> None:
        """The stop button: the live run is asked to stop at its next point where a stop is taken."""
        if self._mode == "live":
            from .controller import controller
            controller().request_stop()

    def _toggle_text(self) -> None:
        """Show the full text, or hide it again."""
        self._text_shown = not self._text_shown
        self.text_button.setText(HIDE_TEXT if self._text_shown else SHOW_TEXT)
        self._arrange()

    def _open_find(self) -> None:
        """Open the field that finds words in the full text, with the keyboard in it."""
        self.find_field.setVisible(True)
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
