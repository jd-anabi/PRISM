"""The Settings page, the rail's row Settings: two sections, the appearance and what belongs to this computer.

"Appearance" is a chooser (Follow Windows / Light / Dark) that calls back to MainWindow to apply and persist the mode.
"This computer" is the memory limit and the keep-awake switch, each remembered for this computer
(``core/gui/memory_limit.py``). The limit is checked when editing ends and never on a keystroke; a limit that is not
applied says why beside the field and changes nothing. The free memory of the graphics card is read only when its button
is pressed, on a thread of its own: never when the page is built or shown. What the last run on this computer ran with
is read from the run history when the page is shown.

While a run is live the memory limit is locked and neither the card nor the run history is read, which the window
says (``set_locked``); the appearance and the switch stay usable, the switch acting on the run at once.
"""
import logging

from PySide6.QtCore import QRunnable, Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QPushButton, QRadioButton, QVBoxLayout,
                               QWidget)

from core.artifacts import provenance

from .. import design, fonts, memory_limit, texts, this_computer
from ..reported import reported
from ..run import ends, history
from ..run.controller import controller
from ..theming import MODE_LABELS
from ..widgets.labeled_inputs import FloatField
from ..widgets.page import Page, PageHead, Sentence
from ..widgets.ruled import SectionHead

log = logging.getLogger(__name__)

_TEXT_SIZE = "Text size follows the text size set in Windows."
_LIMIT_NAME = "Memory limit"
_LIMIT_SENTENCE = "The most graphics-card memory one batch of simulations may plan to use."
_LIMIT_NOTE = ("A lower limit makes PRISM split a batch into smaller pieces, which takes longer and changes nothing in "
               "what is simulated. A limit above what PRISM would plan anyway changes nothing. With no limit, PRISM "
               "plans from the free-memory figure Windows gives it, which can count memory other programs are using as "
               "free.")
_FREE_NOT_READ = "Free graphics-card memory: not read yet."
_FREE_READ = "Free graphics-card memory: {:.1f} GiB, read {}."
_FREE_UNREADABLE = "Free graphics-card memory: could not be read."
_READ_NOW, _READ_AGAIN = "Read now", "Read again"
_NOT_READ_LIVE = "Not read while a run is live."
_LAST_RUN_LATER = "A run is live. Not read yet; read when it ends."
_AWAKE_BOX = "Keep this computer awake while a run is live"
_AWAKE_NOTE = ("PRISM asks Windows not to go to sleep from the start of a run until it ends. The screen may still turn "
               "off. Shutting down or signing out is separate: Windows then shows PRISM's reason and waits for your "
               "answer, whatever this switch says.")
_AWAKE_SINCE = "Windows has been asked to stay awake since {}."
_AWAKE_REFUSED = "Windows did not accept the request. This computer may go to sleep during the run."
_ONLY_ON_WINDOWS = "Only on Windows."
#: How often, in milliseconds, a page looks for the answer of a reading under way; the program takes about 50 ms here.
READING_LOOK_MS = 50
_LAST_RUN_NOT_READ = "The last run could not be read from the history (%s): Settings says that none is recorded."
_NOT_KEPT = ("What you just set could not be remembered: the settings file cannot be written. It holds until PRISM "
             "closes.")


class _FreeMemoryReading(QRunnable):
    """One reading of the free memory, off the window's thread: the program it starts can take two seconds. It leaves
    its answer in the list it is handed and touches nothing of the toolkit: a signal sent on this thread from an object
    the window's thread is deleting ends the process, and the page is deleted as the window goes, a reading or none."""

    def __init__(self, answer: list) -> None:
        super().__init__()
        self._answer = answer

    def run(self) -> None:
        self._answer.append(provenance._nvidia_smi_free_gib())


class SettingsScreen(QWidget):
    """Settings. Drives the one appearance callback it is constructed with; persistence of the chosen mode is that
    callback's job (app-level QSettings), not this widget's. Building it reads the environment, the limit chosen and
    the settings file, and neither the card nor the run history. The window tells it when a run begins and ends
    (``set_locked``), and holds no run itself."""

    def __init__(self, on_mode, current_mode="system", parent=None):
        """``on_mode(mode)`` is called when the user picks an appearance; ``current_mode`` pre-selects."""
        super().__init__(parent)
        self._on_mode = on_mode
        self._locked = False                 # a run is live
        self._last_run_read = False          # the last run's sentence stands as read
        self._answer = None                  # where a reading under way leaves its answer; None: none is
        self._free_words = _FREE_NOT_READ
        self._looking = QTimer(self)         # started by a press, stopped by the answer: nothing ticks before it
        self._looking.setInterval(READING_LOOK_MS)
        self._looking.timeout.connect(self._free_memory_read)

        self.page = Page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.page)
        self.page.add_fixed(PageHead("Settings"))
        self.page.add_gap(design.scaled(design.GAP["head_to_body"]), design.scaled(design.GAP["head_to_body_small"]))

        self.appearance = QWidget()
        lines = self._lines(self.appearance)
        self._group = QButtonGroup(self)
        self._radios = {}
        for mode, label in MODE_LABELS:
            rb = QRadioButton(label)
            self._group.addButton(rb)
            self._radios[mode] = rb
            lines.addWidget(rb)
        lines.addWidget(Sentence(_TEXT_SIZE, quiet=True))
        # Select the current mode BEFORE connecting, so construction never fires on_mode (which would
        # re-enter MainWindow before it has finished wiring this screen).
        self._radios.get(current_mode, self._radios["system"]).setChecked(True)
        for mode, rb in self._radios.items():
            rb.toggled.connect(lambda checked, m=mode: checked and self._on_mode(m))

        computer = QWidget()
        lines = self._lines(computer)
        name = Sentence()
        name.set_text(_LIMIT_SENTENCE, lead=f"{_LIMIT_NAME}.")
        lines.addWidget(name)
        self.no_limit = QRadioButton("No limit")
        self.limited = QRadioButton("Limit each batch to")
        self._options = QButtonGroup(self)
        self._options.addButton(self.no_limit)
        self._options.addButton(self.limited)
        self.limit_field = FloatField(None)
        self.limit_field.setAlignment(Qt.AlignRight)
        self.limit_field.setFont(fonts.inter("body", tabular=True))
        self.limit_field.setFixedWidth(design.scaled(design.SIZE["number_w"]))
        self.limit_field.setAccessibleName("Limit each batch to, in GiB")
        self.limit_unit = QLabel("GiB")
        lines.addWidget(self.no_limit)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(design.scaled(design.GAP["marker_text"]))
        row.addWidget(self.limited)
        row.addWidget(self.limit_field)
        row.addWidget(self.limit_unit)
        row.addStretch(1)
        lines.addLayout(row)
        self.limit_line = Sentence()
        self.limit_line.setVisible(False)
        lines.addWidget(self.limit_line)
        self.in_force = Sentence()
        lines.addWidget(self.in_force)
        lines.addWidget(Sentence(_LIMIT_NOTE, quiet=True))
        self.free_memory = Sentence(_FREE_NOT_READ)
        self.read_button = QPushButton(_READ_NOW)
        lines.addWidget(self.free_memory)
        lines.addWidget(self.read_button, 0, Qt.AlignLeft)
        self.last_run = Sentence()
        lines.addWidget(self.last_run)
        windows = this_computer.on_windows()
        self.awake_box = QCheckBox(_AWAKE_BOX)
        self.awake_box.setChecked(controller().keep_awake_wish())
        self.awake_box.setEnabled(windows)
        lines.addWidget(self.awake_box)
        lines.addWidget(Sentence(_AWAKE_NOTE, quiet=True))
        self.awake_line = Sentence("" if windows else _ONLY_ON_WINDOWS)
        lines.addWidget(self.awake_line)
        self.awake_line.setVisible(not windows)              # now that it is a child: no window of its own
        self.not_kept = Sentence(_NOT_KEPT)
        lines.addWidget(self.not_kept)
        self.not_kept.setVisible(False)

        chosen = memory_limit.chosen()
        (self.no_limit if chosen is None else self.limited).setChecked(True)
        if chosen is not None:
            self.limit_field.setText(f"{chosen:g}")
        self.no_limit.clicked.connect(self._no_limit_pressed)
        self.limited.clicked.connect(self._limited_pressed)
        self.limit_field.editingFinished.connect(self._judge)
        self.read_button.clicked.connect(self._read_free_memory)
        self.awake_box.toggled.connect(self._awake_changed)
        self._show_limit()

        body = QWidget()
        column = self._lines(body, design.scaled(design.GAP["block"]))
        for part in (SectionHead("Appearance"), self.appearance, SectionHead("This computer"), computer):
            column.addWidget(part)
        self._scroller = self.page.add_scrolling(body)

    @staticmethod
    def _lines(block: QWidget, gap: int | None = None) -> QVBoxLayout:
        """A column of lines for ``block``, at the look's gap between rows unless another is handed in."""
        lines = QVBoxLayout(block)
        lines.setContentsMargins(0, 0, 0, 0)
        lines.setSpacing(design.scaled(design.SIZE["row_gap"]) if gap is None else gap)
        return lines

    def set_mode(self, mode: str) -> None:
        """Reflect a mode change made elsewhere (without re-firing ``on_mode``)."""
        rb = self._radios.get(mode)
        if rb is not None and not rb.isChecked():
            rb.blockSignals(True)
            rb.setChecked(True)
            rb.blockSignals(False)

    def set_small(self, on: bool) -> None:
        """The low window, or the ordinary one again."""
        self.page.set_small(on)

    def set_locked(self, live: bool) -> None:
        """A run is live, or none is: the limit and the reading of the free memory are locked and said to be, or free
        again, and the page's reading of the last run, which waited, is done. The appearance and the switch stay as
        they are."""
        live = bool(live)
        if live == self._locked:
            return
        self._locked = live                  # first: a field greyed with the keyboard on it says its editing ended
        self._show_limit()
        self._show_free_memory()
        self._show_awake()
        if not live:
            self._last_run_read = False
            self._read_last_run()

    @reported(QWidget.showEvent)
    def showEvent(self, event):
        super().showEvent(event)
        self._read_last_run()

    # ── the memory limit ──
    def _show_limit(self) -> None:
        """The line under the options, and what can be changed: nothing while a number in the variable decides."""
        force = memory_limit.memory_limit_in_force()
        gib, read = memory_limit.stored_limit()
        self.in_force.set_text(memory_limit.locked_line(self._limit_of_the_run()) if self._locked
                               else memory_limit.limit_line(force, gib, read))
        free = force.source != "environment" and not self._locked
        self.no_limit.setEnabled(free)
        self.limited.setEnabled(free)
        self.limit_field.setEnabled(free and self.limited.isChecked())
        self.limit_unit.setEnabled(self.limit_field.isEnabled())

    @staticmethod
    def _limit_of_the_run() -> float | None:
        """The limit the live run started with, as its start read it."""
        model = controller().model
        return None if model is None else model.spec.facts.get("memory_limit")

    def _say_why(self, sentence: str) -> None:
        """The line beside the field: why what it holds is not applied; "" takes it back."""
        self.limit_line.set_text(sentence)
        self.limit_line.setVisible(bool(sentence))
        self.limit_field.set_error(sentence)

    def _apply(self, gib: float | None) -> None:
        """``gib`` is accepted: the limit, stored once for each change, and what the page says of it."""
        if gib != memory_limit.chosen() or not self.not_kept.isHidden():
            self._say_kept(memory_limit.set_memory_limit(gib))
        self._say_why("")
        self._show_limit()

    def _say_kept(self, kept: bool) -> None:
        """The last write reached the settings file, or it did not: the line follows the answer. A line that has just come
        is brought into sight, since it stands at the foot of a page that scrolls."""
        coming = not kept and self.not_kept.isHidden()
        self.not_kept.setVisible(not kept)
        if coming:
            QTimer.singleShot(0, self, self._bring_the_line_into_sight)

    def _bring_the_line_into_sight(self) -> None:
        """At the loop's next turn, once the line has a place."""
        if not self.not_kept.isHidden():
            self._scroller.ensureWidgetVisible(self.not_kept)

    def _no_limit_pressed(self) -> None:
        self._apply(None)

    def _limited_pressed(self) -> None:
        """"Limit each batch to": the field is opened, and a number it holds is judged at once."""
        self._show_limit()
        if self.limit_field.text().strip():
            self._judge()
        else:
            self.limit_field.setFocus(Qt.OtherFocusReason)

    def _judge(self) -> None:
        """Editing ended in the field, or its option was pressed: the text is applied, or the line says why not. Nothing
        is judged while a run is live, when a field greyed with the keyboard on it still says its editing ended."""
        if self._locked or not self.limited.isChecked():
            return
        gib, why = memory_limit.judged(self.limit_field.text(), memory_limit.memory_limit_in_force())
        if why:
            self._say_why(why)
        else:
            self._apply(gib)

    # ── the free memory ──
    def _read_free_memory(self) -> None:
        """A press of the button: one reading off the window's thread. A press while one is under way does nothing."""
        if self._answer is not None or self._locked:
            return
        self._answer = []
        QThreadPool.globalInstance().start(_FreeMemoryReading(self._answer))
        self._looking.start()

    def _free_memory_read(self) -> None:
        """The page's timer: the answer, once it is there."""
        if not self._answer:
            return
        self._looking.stop()
        gib, self._answer = self._answer[0], None
        self.read_button.setText(_READ_AGAIN)
        self._free_words = (_FREE_UNREADABLE if gib is None
                            else _FREE_READ.format(gib, texts.date_text(controller().now())))
        self._show_free_memory()

    def _show_free_memory(self) -> None:
        """What was last read, and while a run is live that nothing is read, the button greyed."""
        self.free_memory.set_text(f"{self._free_words} {_NOT_READ_LIVE}" if self._locked else self._free_words)
        self.read_button.setEnabled(not self._locked)

    # ── the last run ──
    def _read_last_run(self) -> None:
        """The last run on this computer, read from the run history, once until a run has ended; while a run is live
        nothing is read, and a page that has read nothing yet says so. Only a page in sight reads. A history no reading
        foresaw is said as none recorded, and one warning in the window's log names what went wrong."""
        if self._last_run_read or not self.isVisible():
            return
        if self._locked:
            self.last_run.set_text(_LAST_RUN_LATER)
            return
        self._last_run_read = True
        held = controller()
        try:
            record = history.last_on(held.folder(), memory_limit.computer())
            measured = history.last_measured_on(held.folder(), memory_limit.computer())
            said = ends.last_run_sentence(record, measured)
        except Exception as error:           # noqa: BLE001 -- a history no reading foresaw opens no window's end
            log.warning(_LAST_RUN_NOT_READ, f"{type(error).__name__}: {error}")
            said = ends.NO_LAST_RUN_TEXT
        self.last_run.set_text(said)

    # ── the switch ──
    def _awake_changed(self, on: bool) -> None:
        self._say_kept(memory_limit.set_keep_awake_wish(on))
        self._show_awake()

    def _show_awake(self) -> None:
        """What Windows answered, while a run is live: since when it was asked, or that it did not accept."""
        if not this_computer.on_windows():
            return
        held, words = controller(), ""
        if held.awake_refused():
            words = _AWAKE_REFUSED
        elif held.awake_since() is not None:
            words = _AWAKE_SINCE.format(texts.date_text(held.awake_since()))
        self.awake_line.set_text(words)
        self.awake_line.setVisible(bool(words))
