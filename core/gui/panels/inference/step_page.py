"""A step of parameter inference as a page of the look: a heading; what the step holds, and its figures, and the form,
in one body that scrolls; and a foot that never does, "Before you start" and the one main button. A run started here
takes the place of everything under the heading until it is over.

What a step holds changes only when a run returns its record. Facts that come from the form and the session follow
every edit. Facts that read the disk or the card are read when the page is shown, 300 ms after the last edit, when a
run ends, when the store changes, on "Check again" and at the press, which is the reading that counts; and never
while a run is live, which the line beside "Before you start" says. A page that is not shown reads nothing: it marks
itself stale, and reads as it is shown. A constructor reads none of this and starts no timer.

A page says what it supplies by answering ``main_words``, ``main_pressed``, ``nothing_to_apply``, ``read_disk``,
``facts_from_form`` and ``in_use_words``; it hands the keyboard its first setting through ``first_focus``; and it says
``reads_the_disk`` when it has nothing saved to read.

A refusal is told in the shared question and stands on the page after it. On a row it marks the field, under which its
sentence stands until the field is edited; with no row it is the first of the facts. Either stops the start until the
facts are next read, since what cures it may stand somewhere else, and the press asks the engine again.

Wherever the keyboard lands in the body, by the Tab key or handed over, the control comes whole into view.
"""
from datetime import datetime

from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from ... import design, fields, style
from ...reported import reported
from ...run.controller import controller
from ...run.view import RunView
from ...step_facts import Fact, ordered, reading_words, reason_beside, refused_facts
from ...widgets.ask import ask
from ...widgets.facts import Foot
from ...widgets.figure_stack import FigureStack
from ...widgets.form import READ_AFTER_MS, Form
from ...widgets.in_use import InUse
from ...widgets.marker import painted_icon
from ...widgets.page import Page, PageHead, Scroller, _ring_room
from ..run_host import RunHost

HELP_TEXT = "Help"
REFUSAL_TITLE = "Check your inputs"
OK_TEXT = "OK"
#: How a page's own line is marked. An "Error." blocks, and a line of the page's own stops nothing.
_LINE_KINDS = {"info": "info", "warning": "warn", "error": "warn"}


class _Body(QWidget):
    """The scrolling body of a page: it hands the width the scroller tells it to the form."""

    form = None

    def fit(self, width: int) -> None:
        if self.form is not None:
            self.form.fit(width)


class _Column(QWidget):
    """What a run's page takes the place of, the form over the foot: it tells the foot the height the two share."""

    def __init__(self, foot: Foot) -> None:
        super().__init__()
        self._foot = foot

    @reported(QWidget.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._foot.hold_to(event.size().height())


class StepPage(RunHost, QWidget):
    """The four regions of a step, on the look's ``Page``, over a stack that holds them and the run's page. ``head``,
    ``in_use``, ``form`` and ``foot`` are the regions; ``controls``, what a run locks, is the form; the stale flag says a
    reading is owed."""

    #: Whether the page has anything saved to read, asked at every drawing: a page answers False, or a property of its own
    #: when what it holds decides that, and its foot leaves out the line of when it read and "Check again".
    reads_the_disk = True

    def __init__(self, screen, title: str, sentence: str, *, step: str, parent=None):
        super().__init__(parent)
        self._enrol()
        self._screen, self.step = screen, step
        self.stale = True
        self._keys_ordered = False
        self._read_at: datetime | None = None
        self._head_fact: Fact | None = None
        self._live = controller().is_live()
        self.names = None                    # the name and note of the record a run makes, when the page holds them
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self.page = Page()
        box.addWidget(self.page)
        self.head = PageHead(title, sentence)
        self.help_button = QPushButton(style.access_key(HELP_TEXT))
        self.help_button.setIcon(painted_icon("help"))
        self.help_button.setIconSize(QSize(design.scaled(design.SIZE["marker"]) + design.scaled(design.SIZE["icon_gap"]),
                                           design.scaled(design.SIZE["marker"])))
        self.help_button.clicked.connect(self._help_pressed)
        self.head.add_action(self.help_button)
        self.page.add_fixed(self.head)
        self.page.add_gap(design.scaled(design.GAP["head_to_body"]), design.scaled(design.GAP["head_to_body_small"]))

        self.in_use = InUse()
        self.figure_stack = FigureStack()
        self.figure_stack.setVisible(False)
        self.figure_stack.tabs_changed.connect(self._fit_figures)
        self.figure_stack.tabs_changed.connect(self._a_stop_joined)
        self.run_view = self._new_run_page()
        self.run_view.setObjectName("run")
        self.run_view.back.connect(self._back_to_the_form)
        self.run_view.how_link.clicked.connect(self.show_run)
        self.next_row = QHBoxLayout()
        self.next_row.setContentsMargins(0, 0, 0, 0)
        self.next_button = QPushButton()
        self.next_button.setVisible(False)
        self.next_button.clicked.connect(self._next_pressed)
        self.next_row.addWidget(self.next_button)
        self.next_row.addStretch(1)
        self._next_step = None
        self.form = self.controls = Form()
        self.form.edited.connect(self._edited)
        self.form.joined.connect(self._a_stop_joined)
        body = _Body()
        body.form = self.form
        self._body = QVBoxLayout(body)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(design.scaled(design.GAP["block"]))
        self._body.addWidget(self.in_use)
        self._body.addWidget(self.run_view.how_link, 0, Qt.AlignLeft)
        self._body.addWidget(self.figure_stack)
        self._body.addLayout(self.next_row)
        self._body.addWidget(self.form)
        self.scroller = Scroller(body)
        self.foot = Foot()
        self.foot.main_button.clicked.connect(self._main_clicked)
        self.foot.check_again.clicked.connect(self.read_facts)
        self.figures = _Column(self.foot)
        self.figures.setObjectName("figures")
        column = QVBoxLayout(self.figures)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.scroller, 1)
        room = _ring_room()
        inset = QHBoxLayout()
        inset.setContentsMargins(room, 0, room, 0)           # lined up with the body, which brings that room itself
        inset.addWidget(self.foot)
        column.addLayout(inset)
        self.results = QStackedWidget()
        self.results.addWidget(self.figures)
        self.results.addWidget(self.run_view)
        self.page.add_stack(self.results)

        self.read_timer = QTimer(self)
        self.read_timer.setSingleShot(True)
        self.read_timer.setInterval(READ_AFTER_MS)
        self.read_timer.timeout.connect(self._timer_fired)
        if screen is not None:
            self.foot.go_to_run.connect(screen.go_to_run)
        # Methods of this page's own, so that a page deleted leaves no connection on the process's controller.
        held = controller()
        held.status_changed.connect(self._status_heard)
        held.run_ended.connect(self._run_ended)
        QApplication.instance().focusChanged.connect(self._focus_moved)
        self.foot.set_live(self._live)

    @property
    def session(self):
        return self._screen.session

    def now(self) -> datetime:
        """The moment now, on the clock the runs are read from."""
        return controller().now()

    def _new_run_page(self) -> RunView:
        """A run's page takes the place of the form, so it brings the room for a focus ring as every part of a page does,
        and stands at the heading's edges."""
        view = super()._new_run_page()
        room = _ring_room()
        view.setContentsMargins(room, 0, room, 0)
        return view

    # ── what a page answers ──
    def main_words(self) -> str:
        """What the main button is called: named for its outcome."""
        raise NotImplementedError(f"{type(self).__name__} does not name its main button")

    def main_pressed(self) -> None:
        """The main button was pressed, with the reading that counts made and nothing blocking."""
        raise NotImplementedError(f"{type(self).__name__} does not say what its main button does")

    def nothing_to_apply(self) -> bool:
        """Whether the step has nothing to run and holds what the form shows."""
        return False

    def read_disk(self) -> None:
        """Read what the page's facts read of the disk and the card: ``read_facts`` calls this, and nothing else may."""

    def facts_from_form(self) -> tuple:
        """Every fact of the page as it stands, in its order: no disk and no card."""
        return ()

    def in_use_words(self) -> tuple:
        """What the step holds: its label and the sentence on it."""
        return "", ""

    def use_names(self, block) -> None:
        """Hold the name and note of the record this page's run makes: in the form's block for the run, read with the
        page's facts, and drawn again as their verdict moves."""
        self.names = block
        self.form.add_name(block)
        block.changed.connect(self.redraw)

    def plan_fields(self) -> dict:
        """The rows a plan may fill, by field key."""
        return {}

    def selection_rows(self) -> tuple:
        """The rows that pick what the step holds."""
        return ()

    # ── the window ──
    def first_focus(self):
        """The page's first setting: none while the form is locked, so that a step first opened during a run leaves the
        keyboard on its row of the rail."""
        return self.form.first_control() if self.form.isEnabled() else None

    def order_the_keys(self) -> None:
        """The Tab key's order, as the page is read: the heading's buttons, the run's page, which stands in place of what
        comes next, what stands over the form, the form, then what scrolls while it scrolls, and the foot. Figures stand on
        the run's page or over the form, so they follow both. Widgets take their turn as they come to the window, and a page adds its
        rows after this page is built, so the order is set here, once they stand, as a page first comes into sight, and
        again as a row joins the form or a figure joins the stack. A part that is a stop only at times has its place all the
        same. A control that makes its own fields again after that, as a grid does, calls this when it has."""
        stops = [*self.head.findChildren(QPushButton), *self.run_view.stops(), self.run_view.how_link,
                 *self.figure_stack.stops(), self.next_button, *self.form.stops(), self.scroller, self.foot.facts.scroller,
                 *self.foot.stops()]
        policies = [(stop, stop.focusPolicy()) for stop in stops]
        for stop, policy in policies:
            stop.setFocusPolicy(policy | Qt.TabFocus)
        for before, after in zip(stops, stops[1:]):
            QWidget.setTabOrder(before, after)
        for stop, policy in policies:
            stop.setFocusPolicy(policy)

    def _a_stop_joined(self) -> None:
        """A row or a figure joined the page: it takes its place in the order, once there is one."""
        if self._keys_ordered:
            self.order_the_keys()

    def set_small(self, on: bool) -> None:
        """The low window, or the ordinary one again: the page, and the foot, which stands inside its stack."""
        self.page.set_small(on)
        self.foot.set_small(on)

    def set_next(self, words: str, step: "str | None") -> None:
        """The plain button under the result that names the next step, or none."""
        self._next_step = step
        self.next_button.setText(words)
        self.next_button.setVisible(step is not None)

    def _help_pressed(self) -> None:
        self._screen.navigate.emit("help", None)

    def _next_pressed(self) -> None:
        self._screen.navigate.emit("inference", self._next_step)

    def save_settings(self, qs) -> None:
        """Nothing: what a step remembers is written as it comes into use."""

    def restore_settings(self, qs) -> None:
        """Put remembered values back as choices, never applied: the page's own to say."""

    # ── the page's facts ──
    def redraw(self) -> None:
        """"Before you start" and the main button as the form and the session stand: reads nothing."""
        live = controller().is_live()
        facts = ordered(((self._head_fact,) if self._head_fact is not None else ()) + refused_facts(self.form.refused())
                        + tuple(self.facts_from_form()))
        nothing = self.nothing_to_apply()
        self.foot.set_reads(self.reads_the_disk)
        self.foot.set_main_words(self.main_words())
        self.foot.set_facts(facts)
        self.foot.set_main_enabled(not nothing and not any(fact.blocks for fact in facts),
                                   reason_beside(facts, live=live, nothing_to_apply=nothing))
        self.foot.set_reading(reading_words(self._read_at, live=live))

    def refresh_local_gates(self) -> None:
        """What the step holds and what follows from it. No reading."""
        self.in_use.set_held(*self.in_use_words())
        self.redraw()

    def read_facts(self) -> None:
        """The one reading of the disk and the card. While a run is live nothing is read and the line says so; its end
        reads."""
        self.read_timer.stop()
        if not controller().is_live():
            self.form.ease()
            self.read_disk()
            if self.names is not None:
                self.names.check_now()
            self._read_at, self._head_fact, self.stale = self.now(), None, False
        self.redraw()

    def read_soon(self) -> None:
        """A reading once the typing has stopped, on a page in sight; one that is not in sight, or waits for a run,
        reads when it can."""
        if controller().is_live() or not self.isVisible():
            self.stale = True
        else:
            self.read_timer.start()

    def _timer_fired(self) -> None:
        if self.isVisible():
            self.read_facts()
        else:
            self.stale = True

    def stale_store(self) -> None:
        """The store changed: read now on a page in sight, else at its next show."""
        if self.isVisible():
            self.read_facts()
        else:
            self.stale = True

    def say_at_the_head(self, fact: Fact) -> None:
        """Say ``fact`` first among the facts, until they are next read."""
        self._head_fact = fact
        self.redraw()

    def _edited(self) -> None:
        self.redraw()
        self.read_soon()

    def _status_heard(self, status) -> None:
        """A run began or ended, or a start was taken back, which says no ``run_ended``. What follows from it is drawn
        again, and on any status of no run live: a start taken back at its first drawing never said it was live, and left
        the page drawn as if it were. A change to no run live reads, once the end of the run has had its say."""
        changed = status.live != self._live
        self._live = status.live
        if changed or not status.live:
            self.redraw()
        if changed and not status.live:
            self.read_soon()

    def _run_ended(self, _status) -> None:
        """A run has ended: what it wrote is read now on a page in sight, else at its next show."""
        if self.isVisible():
            self.read_facts()
        else:
            self.stale = True

    @reported(QWidget.showEvent)
    def showEvent(self, event):
        super().showEvent(event)
        if not self._keys_ordered:
            self._keys_ordered = True
            self.order_the_keys()
        if self.stale:
            self.read_facts()

    def _main_clicked(self) -> None:
        """The press: the reading that counts, then the page's own, unless that reading found something that blocks,
        which greys the button and says so beside it."""
        self.read_facts()
        if not self.foot.main_button.isEnabled():
            return
        self.main_pressed()

    @reported()
    def _focus_moved(self, before, _now) -> None:
        """The keyboard moved, in any window. It is brought into view once what moved it has been laid out; a window that
        only came to the front, whose keyboard comes from nowhere, stays as it was scrolled."""
        if before is not None:
            QTimer.singleShot(0, self, self._keyboard_into_view)

    def _keyboard_into_view(self) -> None:
        """The control that holds the keyboard in this window, whole, in the viewport, on a page in sight, if it stands in
        the body."""
        held = self.window().focusWidget()
        if held is not None and self.isVisible() and self.scroller.body.isAncestorOf(held):
            QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)   # a fold, or the foot, may be growing yet
            self.scroller.bring_into_view(held)

    # ── what a run asks of its host ──
    def own_line(self, text: str, level: str = "info") -> None:
        """A line of the page's own stands under what is in use."""
        self.in_use.say(Fact(_LINE_KINDS.get(level, "info"), text))

    def _figures_leaving(self) -> None:
        """Nothing is kept; the run's page shows or hides the figures as a whole, so they stand in sight for it."""
        self.figure_stack.setVisible(True)

    def _figures_back(self, stack) -> None:
        """The figures back in their place in the body, in sight while they hold anything."""
        self._body.insertWidget(self._body.indexOf(self.run_view.how_link) + 1, stack)
        self._fit_figures()

    def _fit_figures(self) -> None:
        """Figures in their place in the body are in sight while they hold anything."""
        if self._body.indexOf(self.figure_stack) >= 0:
            self.figure_stack.setVisible(self.figure_stack.count() > 0)

    def _run_began(self, spec) -> None:
        """A run of this page's own empties its figures and takes its column, unless it only loads a record."""
        if not spec.loads_only:
            self.figure_stack.clear_all()
            self.in_use.say(None)
        super()._run_began(spec)

    def set_controls_enabled(self, enabled: bool) -> None:
        """A run locks the form and the foot's two buttons; the facts stay readable."""
        self.controls.setEnabled(enabled)
        self.foot.set_live(not enabled)
        if enabled:
            self.refresh_local_gates()

    def _refusal(self, exc) -> None:
        """A refusal is told in the shared question, with where to fix it. When it closes, the row of the refused field
        keeps the sentence and takes the keyboard; a refusal with no row on this page stands first among the facts."""
        fix = fields.fix_sentence(exc.field, here=self.step)
        ask(self, REFUSAL_TITLE, exc.message, OK_TEXT, beneath=fix or None)
        row = self.form.rows.get(exc.field)
        if row is None or row.widget is None or not row.is_shown():
            self.say_at_the_head(refused_facts((f"{exc.message} {fix}".rstrip(),))[0])
            return
        row.mark(exc.message)
        self.redraw()
        target = row.focus_target()
        if target is not None and self.isVisible():
            target.setFocus(Qt.OtherFocusReason)
            QTimer.singleShot(0, self, self._keyboard_into_view)  # it may hold the keyboard already, with no move to hear
