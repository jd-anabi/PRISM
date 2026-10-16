"""The PRISM main window: the rail down its left edge and, beside it, the page of the part in sight -- Home,
Parameter inference (the six steps over one session, each a page of its own), Effective temperature (one cell,
or a sweep study), Live simulation, Reduction map, Records, Your own models (their list and the model builder),
Settings and Help. Every part is built with the window, and the panels inside them are reused unchanged in
behaviour. ``go_to`` is the one way to move between the parts; the run's state stands at the rail's foot."""
import time
from datetime import datetime

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QTabBar, QTextEdit, QToolButton, QWidget)

from core.artifacts import KIND_DIRS, StoreError, default_store
from core.config import VALID_MODELS

from . import chain, design, keys, record_words, settings
from .panels.base_panel import RUN_STATE, BasePanel
from .panels.reduction_panel import ReductionPanel
from .panels.simulate_panel import SimulatePanel
from .screens.artifact_screen import ArtifactScreen
from .screens.help_screen import HelpScreen
from .screens.home_screen import HomeFacts, HomeScreen, RecordRow
from .screens.inference_screen import InferenceScreen
from .screens.models_screen import ModelsScreen
from .screens.settings_screen import SettingsScreen
from .screens.shell import Shell, clock_words
from .screens.temperature_screen import TemperatureScreen
from .widgets.artifact_picker import StorePicker
from .widgets.page_names import PageNames


# ── screen-geometry guards ────────────────────────────────────────────────────
# All of them are safe to call unconditionally: when Qt cannot tell us about the screen, the room is
# as large as a widget can be, every frame counts as visible and nothing is moved.
def _available(widget=None):
    """The usable desktop rect (excludes the taskbar), or None when there is no screen."""
    from PySide6.QtGui import QGuiApplication
    screen = (widget.screen() if widget is not None and widget.screen() else
              QGuiApplication.primaryScreen())
    return screen.availableGeometry() if screen is not None else None


def _room_on_screen() -> tuple[int, int]:
    """What the screen can show of a window, (width, height), with a small margin left free. When Qt
    knows no screen it is Qt's own largest widget size, so nothing is held back."""
    rect = _available()
    if rect is None:
        return 16_777_215, 16_777_215
    return rect.width() - 40, rect.height() - 60


def _on_a_visible_screen(widget) -> bool:
    """True when the window's frame actually intersects some attached screen.

    Geometry saved on an external monitor restores off-screen once that monitor is gone -- the window
    exists, is 'open', and cannot be reached.
    """
    from PySide6.QtGui import QGuiApplication
    frame = widget.frameGeometry()
    screens = QGuiApplication.screens()
    if not screens:
        return True                      # headless: nothing to validate against
    return any(s.availableGeometry().intersects(frame) for s in screens)


def _centre_on_primary(widget) -> None:
    rect = _available(widget)
    if rect is None:
        return
    frame = widget.frameGeometry()
    frame.moveCenter(rect.center())
    widget.move(frame.topLeft())


# the keys of the pages of each part that has several, in their order: a page another key names is refused
_PAGES = {"temperature": tuple(place.key for place in keys.PAGES["temperature"]),
          "inference": tuple(place.key for place in keys.STEPS), "models": ("list", "builder")}

# Every place but Home stands the same distance in from the rail, from the window's right edge, from its top and
# from its bottom, to its first frame or its first line: this many pixels at the ordinary text size, scaled with
# it. Home keeps the look's own page margin.
_PLACE_MARGIN = 12
# how far in a part's own first frame or first line stands already, at every text size: a panel's frame 6, the
# toolkit's own margin round a layout that sets none 9, a frame at the part's very edge 0. The window gives each
# part the rest of the place's margin.
_OWN_INSET = {"inference": 6, "temperature": 6, "live": 6, "reduction": 6, "records": 9, "models": 0, "settings": 0,
              "help": 9}


def _is_setting(widget: QWidget) -> bool:
    """Whether ``widget`` is something a person sets: not a button that does something when pressed, nor a tab
    bar or a row of page names (a row of such buttons), nor an area that only shows or scrolls what stands in it
    -- a scrolling area, a plot's view, a text that can only be read. A list and a text one types in are
    settings."""
    if isinstance(widget, (QPushButton, QToolButton, QTabBar)) or isinstance(widget.parentWidget(), PageNames):
        return False
    if isinstance(widget, QAbstractScrollArea):
        typed_in = isinstance(widget, (QPlainTextEdit, QTextEdit)) and not widget.isReadOnly()
        return typed_in or isinstance(widget, QAbstractItemView)
    return True


def _first_control(part: QWidget) -> QWidget | None:
    """The first setting of ``part`` in the order the Tab key takes, shown and able to take the keyboard: where a
    part names no control to start at, the keyboard starts there. None when the part has none."""
    widget = part.nextInFocusChain()
    for _ in range(10_000):                  # the chain is a ring through the window: it comes back to the part
        if widget is None or widget is part:
            return None
        if (part.isAncestorOf(widget) and widget.isVisibleTo(part) and widget.isEnabled()
                and widget.focusPolicy() & Qt.TabFocus and widget.focusProxy() is None and _is_setting(widget)):
            return widget
        widget = widget.nextInFocusChain()
    return None


def _local(created):
    """A manifest's moment, turned into local time; None when it cannot be read as one."""
    try:
        return datetime.fromisoformat(created).astimezone()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _home_records(store) -> tuple:
    """Every record of ``store`` that has a manifest, of every kind, as Home lists it. A simulation set is called
    by the prior it was simulated for, found among the priors by its id. A kind whose folder cannot be read is
    left out: Home lists what it can read."""
    listed = {}
    for kind in KIND_DIRS:
        try:
            listed[kind] = [summary for summary in store.list(kind) if summary.reason is None]
        except (OSError, StoreError):
            listed[kind] = []
    priors = {summary.id: summary for summary in listed["prior"]}
    rows = []
    for kind, summaries in listed.items():
        for summary in summaries:
            prior = priors.get(summary.parents.get("prior")) if kind == "simulation" else None
            prior_label = (None if prior is None else
                           record_words.record_label("prior", prior.name, _local(prior.created)))
            if prior is not None and not prior.name:        # words inside another record's label: a small letter
                prior_label = prior_label[:1].lower() + prior_label[1:]
            made = _local(summary.created)
            rows.append(RecordRow(kind, summary.id,
                                  record_words.record_label(kind, summary.name, made, prior_label=prior_label,
                                                            study=summary.study),
                                  record_words.kind_word(kind, summary.study), made))
    return tuple(rows)


class MainWindow(QMainWindow):
    """The application's window: the rail and, beside it, the parts (``Shell``).

    Always opens on Home -- the last place is deliberately not restored. ``place_changed`` gives the part
    and the page in sight after every move. Owns the QSettings write that runs at quit (``_save_state``,
    reached from closeEvent): panel selections, and the Artifacts screen's own (called by name, because
    _all_panels() is panel-typed and that screen is not). Two other write sites exist on purpose: a
    panel's splitter position is written 1.5 s after it was last dragged (see base_panel.py), and appearance settings
    are written eagerly, as they are applied immediately.
    """
    place_changed = Signal(str, object)

    def __init__(self, appearance=None):
        super().__init__()
        self.appearance = appearance         # theming.Appearance or None (tests build without one)
        self.setWindowTitle("PRISM")
        # The opening size and the smallest size both come from the look's table, grow with Windows'
        # text size and are then held to what the screen can show: the table's opening height is more
        # than a 1366x768 laptop can show, and more than 1920x1080 at 150% scaling, where the action
        # buttons used to start below the fold. The look's two layout thresholds are held to the same room:
        # at a large text size a window as large as the screen keeps the usual layout.
        room = _room_on_screen()
        design.set_room(room)
        opening, smallest = design.window_sizes(room)
        self.resize(*opening)
        # Without a smallest size Qt lets the user (or a restored geometry) shrink the shell until
        # the controls column is unusable. It is held to the screen too, so that it never asks for
        # more than the screen can show.
        self.setMinimumSize(*smallest)

        self.shell = Shell()
        self.setCentralWidget(self.shell)

        # The parts that hold today's panels, reused verbatim: the Reduction map and Live simulation are their
        # panels themselves, Effective temperature chooses between its two by name, and Parameter inference
        # holds its six steps.
        self.reduction_screen = ReductionPanel()
        self.fdt_screen = TemperatureScreen()
        self.inference_screen = InferenceScreen()
        self.simulate_screen = SimulatePanel()

        # Where each panel lives: its part, and its page or step. The foot names a running panel's place
        # from this and goes there, and Home reads where a run is live. A panel that is not here (the bare
        # run host a test builds) is named by no place and leads nowhere.
        self._places = {self.reduction_screen: ("reduction", None), self.simulate_screen: ("live", None)}
        for part, screen in (("temperature", self.fdt_screen), ("inference", self.inference_screen)):
            for panel, page in zip(screen.panels(), _PAGES[part]):
                self._places[panel] = (part, page)

        # Home reads what it states as it is built, the running panel among it: none, at launch.
        self._running_panel = None
        self.home_screen = HomeScreen(self._home_facts)
        # The artifact browser. A plain QWidget, not a BasePanel -- a run anywhere must not grey out the
        # log you are reading.
        self.artifact_screen = ArtifactScreen()
        current_mode = (appearance.mode() if appearance is not None
                        else settings.get_appearance(settings.settings()))
        self.settings_screen = SettingsScreen(self.set_appearance_mode, current_mode)
        # Your own models: their list and the model builder, which a saved or deleted model has every list of
        # models in the panels judge again.
        self.models_screen = ModelsScreen(self._on_user_models_changed)
        self.model_builder_screen = self.models_screen.builder
        self.help_screen = HelpScreen()
        parts = {"home": self.home_screen, "inference": self.inference_screen, "temperature": self.fdt_screen,
                 "live": self.simulate_screen, "reduction": self.reduction_screen, "records": self.artifact_screen,
                 "models": self.models_screen, "settings": self.settings_screen, "help": self.help_screen}
        for place in keys.PARTS:                                     # in the rail's order
            widget = parts[place.key]
            if place.key in _OWN_INSET:                              # the place's margin, less what it stands in
                margin = design.scaled(_PLACE_MARGIN) - _OWN_INSET[place.key]
                widget.setContentsMargins(margin, margin, margin, margin)
            self.shell.add_part(place.key, widget)

        # Every way to move is go_to: a row of the rail, Home's buttons and rows, the run's foot. Each handler
        # is a bound method of this window, so that nothing holds the window once it is deleted.
        self._inference_opened = False           # Parameter inference opens at the step in hand the first time
        self.home_screen.navigate.connect(self.go_to)
        self.shell.rail.opened.connect(self._rail_opened)
        self.shell.rail.left.connect(self._rail_left)
        self.shell.rail.foot.clicked.connect(self.go_to_run)
        # A page of Effective temperature chosen by its name is said as a move: connected after the part's own
        # handler, which shows the page.
        self.fdt_screen.names.chosen.connect(self._page_chosen)
        # The six steps' states go to the rail and to Home each time they are worked out again; a change of
        # the store works out the gates and the states again, after the pickers have listed it.
        self.inference_screen.states_changed.connect(self._states_changed)
        self.artifact_screen.store_changed.connect(self._store_changed)

        # The run's clock, STOPPED at launch: the timer is constructed here but never started, and an
        # unstarted QTimer registers nothing with the OS, so nothing on the launch path ticks.
        self._run_title = self._run_place_words = ""
        self._run_started = 0.0
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(1000)
        self._run_timer.timeout.connect(self._tick_running)
        RUN_STATE.changed.connect(self._on_run_state)

        # ALWAYS open on Home, its row marked and the steps' states in the rail. Home has read what it states
        # as it was built, and reads nothing more at its first show.
        self.shell.show_part("home")
        self.shell.rail.set_step_states(self.inference_screen.step_states())
        self.shell.rail.set_place("home")

        # Restore geometry only. Each panel restored its own selections in its __init__; we deliberately
        # do NOT restore the last screen -- the app always opens on Home.
        qs = settings.settings()
        geom = qs.value("window/geometry")
        if geom is not None:
            # CHECK the return value, and then check the RESULT. restoreGeometry returns False for an
            # unusable blob, and even on success it will happily restore a window onto a monitor that
            # is no longer attached -- geometry saved on an external display came back off-screen with
            # no way to reach it.
            if not self.restoreGeometry(geom) or not _on_a_visible_screen(self):
                self.resize(*opening)
                _centre_on_primary(self)

    def set_appearance_mode(self, mode):
        """Apply + persist an appearance mode, and show it on the Settings screen when that is built.

        Safe to call from the Settings radio itself: the radio blocks its own signal when it is set, so there
        is no feedback loop.
        """
        qs = settings.settings()
        settings.set_appearance(qs, mode)
        qs.sync()
        if self.appearance is not None:
            self.appearance.set_mode(mode)
        shown = self.shell.part_widget("settings")
        if shown is not None:
            shown.set_mode(mode)

    # ── models of your own ────────────────────────────────────────────────────
    def _on_user_models_changed(self, _name: str = ""):
        """After a model of your own was saved or deleted: re-sync every model combo (the list of Your own models
        has named it already)."""
        self._refresh_model_combos()

    def _refresh_model_combos(self):
        """Reload every panel's model combo from the live VALID_MODELS list, keeping the selection
        (falling back to NADROWSKI if the previous selection was deleted). The panel's model-changed
        hook is re-fired ONLY when the selection actually changed: the hooks refresh cell/bounds
        pickers (which resets them to their first entry), so firing on a mere item-list update would
        silently discard the user's picker selections app-wide on every model save/delete."""
        for panel in self._all_panels():
            combo = getattr(panel, "model_combo", None)
            if combo is None:
                continue
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(VALID_MODELS)
            combo.setCurrentText(current if current in VALID_MODELS else "NADROWSKI")
            combo.blockSignals(False)
            if combo.currentText() != current:
                handler = getattr(panel, "_on_model_changed", None)
                if handler is not None:
                    handler(combo.currentText())

    def _store_changed(self):
        """The records screen changed the store: every picker lists it again, then the gates and the steps'
        states are worked out again, so that the rail and Home follow a record written or deleted."""
        self._refresh_store_pickers()
        self.inference_screen.refresh_gates()

    def _refresh_store_pickers(self):
        """Re-list every artifact picker after the browser changed the store -- the twin of
        _refresh_model_combos, which a saved or deleted user model already drives.

        Found by TYPE rather than by naming the three attributes (prior_picker, post_picker,
        obs_picker), so a fourth picker added to a tab is covered by construction.
        StorePicker.refresh re-reads its kind and re-applies its own current key, so a selection that
        still exists survives and one that was just deleted falls back to whatever is current --
        which is the point: restore_key silently does nothing when the saved id has vanished, and that
        silence becomes a defect the moment a browser can delete.
        """
        for panel in self._all_panels():
            for picker in panel.findChildren(StorePicker):
                picker.refresh()

    def _all_panels(self):
        return ([self.reduction_screen] + self.fdt_screen.panels() + self.inference_screen.panels()
                + [self.simulate_screen])

    # ── Home ──────────────────────────────────────────────────────────────────
    def _home_facts(self) -> HomeFacts:
        """What Home states, read now: the inference screen's session and its six steps as last worked out,
        the part and page a live run is in, and every record of the store in use."""
        return HomeFacts(self.inference_screen.session, self.inference_screen.step_states(),
                         self._run_place(self._running_panel), _home_records(default_store()))

    def _run_place(self, panel):
        """The part and page a panel's run is in: None for no panel, or for one with no place in the window."""
        return self._places.get(panel) if panel is not None else None

    def _states_changed(self, states) -> None:
        """The six steps' states were worked out again: the rail shows them, and Home reads now while it is
        shown, else at its next opening."""
        self.shell.rail.set_step_states(states)
        self.home_screen.refresh()

    # ── moving between the parts ──────────────────────────────────────────────
    def part(self, name: str) -> QWidget:
        """The top widget of the part ``name``; KeyError for a part the window does not have."""
        widget = self.shell.part_widget(name)
        if widget is None:
            raise KeyError(f"not a part of the window: {name!r}")
        return widget

    def current_place(self) -> tuple[str, str | None]:
        """The part in sight, and the page or step it shows (None for a part of one page)."""
        shown = self.shell.stack.currentWidget()
        part = next(place.key for place in keys.PARTS if self.shell.part_widget(place.key) is shown)
        current = getattr(shown, "current_page", None)
        return part, (current() if callable(current) else None)

    def go_to(self, part: str, page: str | None = None) -> None:
        """Show the part ``part`` and, when one is named, its page: a step of Parameter inference, a page of
        Effective temperature, or for Records the kind of record it lists. Then mark the rail, give the page
        the keyboard and say where the window is, all before returning. Parameter inference opened with no
        step shows the step in hand the first time and the step last shown after that; every step opens, one
        that cannot start yet too. Home is read again each time it is opened. A part the window does not have,
        or a page its part does not have, raises KeyError before anything moves."""
        self._move(part, page, keyboard=Qt.OtherFocusReason)

    def go_to_run(self) -> None:
        """Go to the place of the live run. Nothing when no run is live, or its panel has no place."""
        place = self._run_place(self._running_panel)
        if place is not None:
            self.go_to(*place)

    def _move(self, part: str, page, *, keyboard) -> None:
        """The move ``go_to`` makes. ``keyboard`` is the reason the page takes the keyboard for, or None for a move
        that gives the page no keyboard."""
        if not self.shell.has_part(part):
            raise KeyError(f"not a part of the window: {part!r}")
        pages = _PAGES.get(part, ())
        if page is not None and pages and page not in pages:
            raise KeyError(f"not a page of {part!r}: {page!r}")
        widget = self.shell.part_widget(part)
        if part == "inference":
            if page is None and not self._inference_opened:
                page = chain.step_in_hand(self.inference_screen.step_states())
            self._inference_opened = True
        if part == "records" and page is not None:
            found = self.artifact_screen.kind_combo.findData(page)
            if found >= 0:
                self.artifact_screen.kind_combo.setCurrentIndex(found)
        if part == "home":
            if self.shell.stack.currentWidget() is widget:
                self.home_screen.refresh()           # in sight already: read now (nothing while the window is hidden)
            else:
                self.home_screen.mark_stale()        # read as it is shown
        self.shell.show_part(part)
        show = getattr(widget, "show_page", None)
        if page is not None and callable(show):
            show(page)
        now, shown = self.current_place()
        self.shell.rail.set_place(now, shown if now == "inference" else None)
        if keyboard is not None:
            self._give_keyboard(widget, keyboard)
        self.place_changed.emit(now, shown)

    def _give_keyboard(self, widget: QWidget, reason) -> None:
        """Put the keyboard on the control the part names to start at, or else on its first control that is not a
        button."""
        first = getattr(widget, "first_focus", None)
        target = first() if callable(first) else None
        if target is None:
            target = _first_control(widget)
        if target is not None:
            target.setFocus(reason)

    def _rail_opened(self, part: str, step) -> None:
        """A row of the rail was pressed. By Enter or Space in the rail: the page takes the keyboard, as after a
        key. By a click: the page takes none, whatever held the keyboard. A page that held it lets it go to
        nothing; a rail that held it keeps it, on the row the click marked. So no control of the page the click
        opens wears the focus ring, a field one types in among them, and no ring is left on another row."""
        rail = self.shell.rail
        if rail.opened_by_key():
            self._move(part, step, keyboard=Qt.ShortcutFocusReason)
            return
        held = self.focusWidget()
        if held is not None and self.shell.stack.currentWidget().isAncestorOf(held):
            held.clearFocus()
        self._move(part, step, keyboard=None)
        if held is not None and rail.isAncestorOf(held):
            now, shown = self.current_place()
            rail.row(shown if now == "inference" else now).setFocus(Qt.OtherFocusReason)

    def _rail_left(self) -> None:
        """Escape in the rail: the page in sight takes the keyboard."""
        self._give_keyboard(self.shell.stack.currentWidget(), Qt.ShortcutFocusReason)

    def _page_chosen(self, page: str) -> None:
        """A person chose a page of Effective temperature by its name, which only a part in sight offers: the move
        is said, as go_to says one, and the rail keeps the part marked."""
        self.place_changed.emit("temperature", page)

    # ── the live run, at the foot of the rail ─────────────────────────────────
    def _on_run_state(self, panel) -> None:
        """A run started (``panel``) or ended (None): the steps' states are worked out again, which shows the
        live step in the rail and has Home read, and the foot says what is running and for how long. When the
        run ends with the keyboard on the foot, which is then no control, the rail takes the keyboard.

        Connected to the module-level RUN_STATE in base_panel.py, which every window built in the process
        hears until it is deleted: this handler touches only its own window's widgets, and is safe to run on
        a window nobody is looking at. The run's title and the words of its place are worked out ONCE per run,
        not per tick.
        """
        self._running_panel = panel
        self.inference_screen.recompute_states()
        foot = self.shell.rail.foot
        if panel is None:
            self._run_timer.stop()
            held = self.focusWidget() is foot
            foot.set_state("none", "Nothing is running", "Only one run at a time.")
            if held:
                self.shell.rail.take_keyboard()
            return
        place = self._run_place(panel)
        if place is None:
            self._run_title = self._run_place_words = "Running: a task"
        else:
            part, page = place
            self._run_title = f"Running: {keys.step(page).name if part == 'inference' else keys.part(part).name}"
            words = keys.place_words(page if part == "inference" else f"{part}/{page}" if page else part)
            self._run_place_words = f"Running: {words}"
        self._run_started = time.monotonic()
        self._tick_running()                       # shown at once, not a second from now
        self._run_timer.start()

    def _tick_running(self) -> None:
        """Write the elapsed time again: a divmod and the foot's words, once a second."""
        words = f"{clock_words(int(time.monotonic() - self._run_started))} so far. Settings are locked."
        foot = self.shell.rail.foot
        foot.set_state("now", self._run_title, words, button=self._run_place(self._running_panel) is not None)
        whole = f"{self._run_place_words}. {words}"   # the place in full: the title names only its part or step
        foot.setAccessibleName(whole)
        foot.setToolTip(whole)

    def panel(self, cls):
        """The first panel of type ``cls`` across all screens (convenience for callers + tests)."""
        return next((p for p in self._all_panels() if isinstance(p, cls)), None)

    # Written in Python so that the interpreter stays with this thread while the base class handles an
    # event, a repaint among them. The parts of the look written in Python (the small style, the focus
    # follower) are entered dozens to hundreds of times in one repaint, and without this each entry
    # waits for the interpreter while a run's thread is working: a repaint of a page then took 0.2 to
    # 1.5 s beside plain Python, where it takes about 20 ms now. Measured, this costs a run nothing
    # while the window does what a run does to it. What is kept is the interpreter, and only while
    # the handling itself is at work: a page repainted without pause would hold a run up. It is let go
    # whenever the handling waits in a loop of its own, so a box opened from under this call (the
    # question asked at closing, the box of an unhandled error) holds no run up while it waits for
    # its answer. The method stands in the class body because the binding decides at an object's
    # first event whether its handler is written in Python.
    # One thing is lost by it. An error raised under this call is dropped by the binding, without a
    # word, when the toolkit enters another method written in Python before the handling returns: in
    # practice whatever is raised while the window is painted, and a page's show or hide event when
    # the window comes back from minimised or goes there. A slot's error is reported as before; so
    # is one of the window's own handlers (the close handler among them), under a longer first
    # line. That is why the methods this program writes for painting and for the showing of a page
    # carry the guard of core/gui/reported.py; a drawing library's own code inside the window
    # carries none. This method carries none either, and stays this one call: a fallback would
    # handle the event twice.
    def event(self, event):
        return super().event(event)

    def closeEvent(self, event):
        """On close, offer to cancel a running task first; otherwise close normally.

        A cancel is cooperative (it lands at the run's next checkpoint), so even "Cancel & quit" does
        not stop the process instantly -- but it does stop it, instead of leaving it holding the CPU/GPU
        invisibly after the window is gone, which is what "Quit anyway" still does.
        """
        if BasePanel._running:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("A task is still running")
            box.setText("A task is still running.")
            box.setInformativeText(
                "Cancelling stops it at its next checkpoint (up to ~1 min during training). "
                "Quitting anyway closes the window now, but the task keeps running in the background "
                "until it finishes on its own.")
            cancel_quit = box.addButton("Cancel task && quit", QMessageBox.AcceptRole)
            quit_anyway = box.addButton("Quit anyway", QMessageBox.DestructiveRole)
            box.addButton("Don't quit", QMessageBox.RejectRole)
            box.setDefaultButton(cancel_quit)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_quit:
                BasePanel.request_cancel_all()
            elif clicked is not quit_anyway:
                event.ignore()
                return

        self._save_state()
        super().closeEvent(event)

    def _save_state(self):
        """Persist window geometry + each panel's selections. Called only when a close is accepted."""
        qs = settings.settings()
        qs.setValue("window/geometry", self.saveGeometry())
        for panel in self._all_panels():
            panel.save_settings(qs)
        # BY NAME: the browser is a QWidget, not a BasePanel, so it is not in _all_panels() and
        # gets no sweep for free -- which is also what keeps it out of _refresh_model_combos.
        self.artifact_screen.save_settings(qs)
        qs.sync()
