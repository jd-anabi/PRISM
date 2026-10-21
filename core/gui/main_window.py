"""The PRISM main window: the rail down its left edge and, beside it, the page of the part in sight -- Home,
Parameter inference (the six steps over one session, each a page of its own), Effective temperature (one cell,
or a sweep study), Live simulation, Reduction map, Records, Your own models (their list and the model builder),
Settings and Help. The window is built with Home alone; every other part is built the first time it is opened,
and one that cannot be built shows why in its place while the rest of the window works. The panels inside the
parts are reused unchanged in behaviour. ``go_to`` is the one way to move between the parts; the run's state
stands at the rail's foot."""
import logging
import sys
import time
import traceback
from datetime import datetime

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QApplication, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QTabBar, QTextEdit, QToolButton, QWidget)

from core.artifacts import KIND_DIRS, StoreError, default_store
from core.config import VALID_MODELS

from . import chain, design, keys, one_instance, record_words, settings, style
from .panels.base_panel import RUN_STATE, BasePanel
from .panels.crossval_panel import CrossValPanel
from .panels.fdt_panel import FdtPanel
from .panels.inference_tabs import ConfigPanel, InferPanel, PosteriorPanel, PriorPanel, TSNPEPanel, ValidatePanel
from .panels.reduction_panel import ReductionPanel
from .panels.simulate_panel import SimulatePanel
from .screens.artifact_screen import ArtifactScreen
from .screens.help_screen import HelpScreen
from .screens.home_screen import HomeFacts, HomeScreen, RecordRow
from .screens.inference_screen import InferenceScreen
from .screens.models_screen import PAGES as MODEL_PAGES, ModelsScreen
from .screens.settings_screen import SettingsScreen
from .screens.shell import PartError, Shell, clock_words
from .screens.temperature_screen import TemperatureScreen
from .session import SbiSession
from .widgets.artifact_picker import StorePicker
from .widgets.page_names import PageNames

log = logging.getLogger(__name__)


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


# the keys of the window's parts, in the rail's order
_PARTS = tuple(place.key for place in keys.PARTS)

# the keys of the pages of each part that has several, in their order: a page another key names is refused
_PAGES = {"temperature": tuple(place.key for place in keys.PAGES["temperature"]),
          "inference": tuple(place.key for place in keys.STEPS), "models": MODEL_PAGES}

# the part that holds each kind of panel, which ``panel`` builds when it is asked for a panel no part built holds
_PANEL_PARTS = {ReductionPanel: "reduction", FdtPanel: "temperature", CrossValPanel: "temperature",
                ConfigPanel: "inference", PriorPanel: "inference", PosteriorPanel: "inference",
                ValidatePanel: "inference", InferPanel: "inference", TSNPEPanel: "inference", SimulatePanel: "live"}

# Every place but Home stands the same distance in from the rail, from the window's right edge, from its top and
# from its bottom, to its first frame or its first line: this many pixels at the ordinary text size, scaled with
# it. Home keeps the look's own page margin, and so does a part's error page.
_PLACE_MARGIN = 12
# how far in a part's own first frame or first line stands already, at every text size: a panel's frame 6, the
# toolkit's own margin round a layout that sets none 9, a frame at the part's very edge 0. The window gives each
# part the rest of the place's margin.
_OWN_INSET = {"inference": 6, "temperature": 6, "live": 6, "reduction": 6, "records": 9, "models": 0, "settings": 0,
              "help": 9}


def _is_a_button(widget: QWidget) -> bool:
    """Whether ``widget`` does something when pressed: a button, or a tab bar or a row of page names (a row of
    such buttons)."""
    return isinstance(widget, (QPushButton, QToolButton, QTabBar)) or isinstance(widget.parentWidget(), PageNames)


def _is_setting(widget: QWidget) -> bool:
    """Whether ``widget`` is something a person sets: not a button that does something when pressed, nor a tab
    bar or a row of page names (a row of such buttons), nor an area that only shows or scrolls what stands in it
    -- a scrolling area, a plot's view, a text that can only be read. A list and a text one types in are
    settings."""
    if _is_a_button(widget):
        return False
    if isinstance(widget, QAbstractScrollArea):
        typed_in = isinstance(widget, (QPlainTextEdit, QTextEdit)) and not widget.isReadOnly()
        return typed_in or isinstance(widget, QAbstractItemView)
    return True


def _first_control(part: QWidget, takes=_is_setting) -> QWidget | None:
    """The first control of ``part`` that ``takes`` admits, its first setting unless another rule is handed in, in
    the order the Tab key takes, shown and able to take the keyboard: where a part names no control to start at,
    the keyboard starts there. None when the part has none."""
    widget = part.nextInFocusChain()
    for _ in range(10_000):                  # the chain is a ring through the window: it comes back to the part
        if widget is None or widget is part:
            return None
        if (part.isAncestorOf(widget) and widget.isVisibleTo(part) and widget.isEnabled()
                and widget.focusPolicy() & Qt.TabFocus and widget.focusProxy() is None and takes(widget)):
            return widget
        widget = widget.nextInFocusChain()
    return None


def _holds_the_front(window) -> bool:
    """Whether ``window`` is the window the system holds in front, the one that takes the person's keys: on Windows
    the system's own answer, read at once; elsewhere the toolkit's, which it may learn a moment later."""
    if sys.platform == "win32":
        import ctypes
        user32 = ctypes.WinDLL("user32")
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        return user32.GetForegroundWindow() == int(window.winId())
    return window.isActiveWindow()


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

    Always opens on Home -- the last place is deliberately not restored -- and is built with Home alone: every
    other part is built the first time it is opened, inside a guard (``_build``). ``place_changed`` gives the part
    and the page in sight after every move. Owns the QSettings write that runs at quit (``_save_state``, reached
    from closeEvent): the selections of the panels of the parts built in this launch, and the Artifacts screen's
    own when it was built (called by name, because _all_panels() is panel-typed and that screen is not); a part
    never opened writes nothing and keeps what it remembered. Two other write sites exist on purpose: a
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

        # The window is built with Home alone. Every other part is built the first time it is opened (``_build``);
        # until then nothing of it exists, and when it is built its panels read the lists of models and the
        # records as they are then, and restore what they remembered.
        self._built: dict = {}                   # each part built, by its key: its top widget
        # Where each panel of a part built lives: its part, and its page or step. The foot names a running panel's
        # place from this and goes there, and Home reads where a run is live. A panel that is not here (the bare
        # run host a test builds) is named by no place and leads nowhere.
        self._places: dict = {}
        self._inference_opened = False           # Parameter inference opens at the step in hand the first time
        # Home reads what it states as it is built, the running panel among it: none, at launch.
        self._running_panel = None
        self._build("home")

        # Every way to move is go_to: a row of the rail, Home's buttons and rows, the run's foot. Each handler
        # is a bound method of this window, so that nothing holds the window once it is deleted.
        self.shell.rail.opened.connect(self._rail_opened)
        self.shell.rail.left.connect(self._rail_left)
        self.shell.rail.foot.clicked.connect(self.go_to_run)
        # A key for every part, every step and the run, and one between the rail and the page: a shortcut of this
        # window for each row of the key map, heard only while the window is the one in front, so that no key
        # moves anything while a question the window asks is open. A key held down acts once.
        for sequence, does in keys.BINDINGS:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.setProperty("does", does)
            shortcut.activated.connect(self._key_pressed)

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
        self.shell.rail.set_step_states(self._step_states())
        self.shell.rail.set_place("home")

        # Restore geometry only. Each panel restores its own selections as its part is built; we deliberately
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
        """Apply + persist an appearance mode, and show it on the Settings screen when that is built: one built
        later reads the mode in force.

        Safe to call from the Settings radio itself: the radio blocks its own signal when it is set, so there
        is no feedback loop.
        """
        qs = settings.settings()
        settings.set_appearance(qs, mode)
        qs.sync()
        if self.appearance is not None:
            self.appearance.set_mode(mode)
        shown = self._built.get("settings")
        if shown is not None:
            shown.set_mode(mode)

    # ── the parts, each built when first opened ───────────────────────────────
    def _make(self, name: str) -> QWidget:
        """The top widget of the part ``name``, built now: the Reduction map and Live simulation are their panels
        themselves, Effective temperature chooses between its two by name, Parameter inference holds its six
        steps, and Your own models hands a model saved or deleted to the window."""
        if name == "home":
            return HomeScreen(self._home_facts)
        if name == "inference":
            return InferenceScreen()
        if name == "temperature":
            return TemperatureScreen()
        if name == "live":
            return SimulatePanel()
        if name == "reduction":
            return ReductionPanel()
        if name == "records":
            # A plain QWidget, not a BasePanel -- a run anywhere must not grey out the log you are reading.
            return ArtifactScreen()
        if name == "models":
            return ModelsScreen(self._on_user_models_changed)
        if name == "settings":
            current_mode = (self.appearance.mode() if self.appearance is not None
                            else settings.get_appearance(settings.settings()))
            return SettingsScreen(self.set_appearance_mode, current_mode)
        return HelpScreen()

    def _build(self, name: str) -> QWidget:
        """Build the part ``name`` and hold it in its place -- in place of its error page when it shows one -- and
        hand back what now stands there. It is built inside a guard: a part whose construction or wiring raises
        shows its error page in its place, nothing the failed build made is kept (no run locks a panel of it), its
        error and the error's trace go to the log, and the rest of the window goes on. The pointer shows the wait
        cursor while the part is built."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # What stands before the build: any other panel, or widget with no parent, is the build's own.
        enrolled, parentless = set(BasePanel._instances), set(QApplication.topLevelWidgets())
        try:
            widget = self._make(name)
            self._wire(name, widget)
        except Exception as error:               # noqa: BLE001 -- one part that cannot be built stops no other
            # Nothing the failed build made is kept: no run locks a panel of it and no place names one, and every
            # widget it made with no parent goes, with all it holds (its panels, a half-built part's top widget).
            for panel in set(BasePanel._instances) - enrolled:
                BasePanel._instances.discard(panel)
                self._places.pop(panel, None)
            for made in set(QApplication.topLevelWidgets()) - parentless:
                made.deleteLater()
            words, trace = keys.part(name).name, traceback.format_exc()
            # The error's words and its trace, never the error itself: its trace holds what was half built.
            log.error("%s could not be opened. %s\n%s", words, str(error), trace.rstrip())
            widget = PartError(words, error, trace)
            widget.try_again.connect(self._try_again)
        finally:
            QApplication.restoreOverrideCursor()
        held = self.shell.part_widget(name)
        if held is None:
            self.shell.add_part(name, widget)
        else:
            self.shell.replace_part(name, widget)
            held.deleteLater()
        if not isinstance(widget, PartError):
            self._built[name] = widget
        return widget

    def _wire(self, name: str, widget: QWidget) -> None:
        """What the window does for a part it has just built: the place's margin, the part's signals, where each of
        its panels lives, and, while a run is live, its panels locked as those of the parts built already are."""
        if name in _OWN_INSET:                                   # the place's margin, less what it stands in
            margin = design.scaled(_PLACE_MARGIN) - _OWN_INSET[name]
            widget.setContentsMargins(margin, margin, margin, margin)
        if name == "home":
            widget.navigate.connect(self.go_to)
        elif name == "inference":
            # The six steps' states go to the rail and to Home each time they are worked out again.
            widget.states_changed.connect(self._states_changed)
        elif name == "temperature":
            # A page chosen by its name is said as a move: connected after the part's own handler, which shows the
            # page.
            widget.names.chosen.connect(self._page_chosen)
        elif name == "records":
            # A change of the store works out the gates and the states again, after the pickers have listed it.
            widget.store_changed.connect(self._store_changed)
        panels = self._panels_of(name, widget)
        for panel, page in zip(panels, _PAGES.get(name, (None,) * len(panels))):
            self._places[panel] = (name, page)
        if BasePanel._running:
            for panel in panels:
                panel.set_controls_enabled(False)

    @staticmethod
    def _panels_of(name: str, widget: QWidget) -> list:
        """The panels of the part ``name``, whose top widget is ``widget``, in their order; none for a part that
        holds no panel."""
        if name in ("reduction", "live"):
            return [widget]
        if name in ("temperature", "inference"):
            return widget.panels()
        return []

    def _try_again(self) -> None:
        """"Try again" on a part's error page: the part is built again in the page's place and opened as a move
        opens it. One that still cannot be built shows its error page again."""
        name = next((key for key in _PARTS if self.shell.part_widget(key) is self.sender()), None)
        if name is not None:
            self._build(name)
            self._move(name, None, keyboard=Qt.OtherFocusReason)

    def built_parts(self) -> tuple[str, ...]:
        """The keys of the parts built, in the rail's order: a part not yet opened is not built, nor is one whose
        error page stands in its place."""
        return tuple(key for key in _PARTS if key in self._built)

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
        silently discard the user's picker selections app-wide on every model save/delete. A part not
        built yet lists the models as they are when it is built."""
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
        states are worked out again, so that the rail and Home follow a record written or deleted. Before
        Parameter inference is built, Home reads again."""
        self._refresh_store_pickers()
        inference = self._built.get("inference")
        if inference is not None:
            inference.refresh_gates()
        else:
            self._states_changed(self._step_states())

    def _refresh_store_pickers(self):
        """Re-list every artifact picker after the browser changed the store -- the twin of
        _refresh_model_combos, which a saved or deleted user model already drives.

        Found by TYPE rather than by naming the three attributes (prior_picker, post_picker,
        obs_picker), so a fourth picker added to a step is covered by construction.
        StorePicker.refresh re-reads its kind and re-applies its own current key, so a selection that
        still exists survives and one that was just deleted falls back to whatever is current --
        which is the point: restore_key silently does nothing when the saved id has vanished, and that
        silence becomes a defect the moment a browser can delete. A part not built yet lists the
        store as it is when it is built.
        """
        for panel in self._all_panels():
            for picker in panel.findChildren(StorePicker):
                picker.refresh()

    def _all_panels(self):
        """The panels of the parts built, in this order: the Reduction map, Effective temperature's two, the six
        steps, Live simulation. A part not built yet has none."""
        return [panel for name in ("reduction", "temperature", "inference", "live") if name in self._built
                for panel in self._panels_of(name, self._built[name])]

    # ── Home ──────────────────────────────────────────────────────────────────
    def _step_states(self) -> tuple:
        """The six steps' states as Parameter inference last worked them out; before it is built, those of a
        session that holds nothing, which is the session it is built with."""
        inference = self._built.get("inference")
        if inference is not None:
            return inference.step_states()
        return chain.step_states(SbiSession(), label=record_words.label_of)

    def _home_facts(self) -> HomeFacts:
        """What Home states, read now: Parameter inference's session and its six steps as last worked out (before
        it is built, a session that holds nothing), the part and page a live run is in, and every record of the
        store in use."""
        inference = self._built.get("inference")
        session = inference.session if inference is not None else SbiSession()
        return HomeFacts(session, self._step_states(), self._run_place(self._running_panel),
                         _home_records(default_store()))

    def _run_place(self, panel):
        """The part and page a panel's run is in: None for no panel, or for one with no place in the window."""
        return self._places.get(panel) if panel is not None else None

    def _states_changed(self, states) -> None:
        """The six steps' states were worked out again: the rail shows them, and Home reads now while it is
        shown, else at its next opening."""
        self.shell.rail.set_step_states(states)
        home = self._built.get("home")
        if home is not None:
            home.refresh()

    # ── moving between the parts ──────────────────────────────────────────────
    def part(self, name: str) -> QWidget:
        """The top widget of the part ``name``, built now if it is not yet: its error page when it cannot be built.
        KeyError for a part the window does not have."""
        if name not in _PARTS:
            raise KeyError(f"not a part of the window: {name!r}")
        widget = self.shell.part_widget(name)
        return widget if widget is not None else self._build(name)

    def current_place(self) -> tuple[str, str | None]:
        """The part in sight, and the page or step it shows (None for a part of one page)."""
        shown = self.shell.stack.currentWidget()
        part = next(key for key in _PARTS if self.shell.part_widget(key) is shown)
        current = getattr(shown, "current_page", None)
        return part, (current() if callable(current) else None)

    def go_to(self, part: str, page: str | None = None) -> None:
        """Show the part ``part`` and, when one is named, its page: a step of Parameter inference, a page of
        Effective temperature, or for Records the kind of record it lists. A part not yet opened is built first,
        its row of the rail marked before it is. Then mark the rail, give the page the keyboard and say where the
        window is, all before returning. Parameter inference opened with no step shows the step in hand the first
        time and the step last shown after that; every step opens, one that cannot start yet too. Home is read
        again each time it is opened. A part the window does not have, or a page that a part of several pages does
        not have -- Parameter inference, Effective temperature, Your own models -- raises KeyError before anything
        moves or is built; a page named for a part of one page is not read, and a kind of record Records does not
        list leaves Records at the kind it showed."""
        self._move(part, page, keyboard=Qt.OtherFocusReason)

    def come_to_front(self) -> None:
        """Bring the window to the front, as a second start of PRISM asks: out of the minimised state when it is
        there (a window that was maximised comes back maximised), raised over the other windows and given the
        keyboard. While a question is open (a dialog that must be answered first), it is the question that is raised
        and given the keyboard, and whose hold of the front is read. When the system keeps the front for the window a
        person is working in, the window's button on the taskbar flashes instead, until the window is brought to the
        front."""
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
        front = QApplication.activeModalWidget() or self
        front.raise_()
        front.activateWindow()
        if not _holds_the_front(front):
            QApplication.alert(self)

    def go_to_run(self) -> None:
        """Go to the place of the live run. Nothing when no run is live, or its panel has no place."""
        self._to_run(Qt.OtherFocusReason)

    def _to_run(self, keyboard) -> None:
        """The move to the live run's place, the page taking the keyboard for the reason ``keyboard``."""
        place = self._run_place(self._running_panel)
        if place is not None:
            self._move(*place, keyboard=keyboard)

    def _move(self, part: str, page, *, keyboard) -> None:
        """The move ``go_to`` makes. ``keyboard`` is the reason the page takes the keyboard for, or None for a move
        that gives the page no keyboard."""
        if part not in _PARTS:
            raise KeyError(f"not a part of the window: {part!r}")
        pages = _PAGES.get(part, ())
        if page is not None and pages and page not in pages:
            raise KeyError(f"not a page of {part!r}: {page!r}")
        if part == "inference" and page is None and not self._inference_opened:
            page = chain.step_in_hand(self._step_states())
        if not self.shell.has_part(part):
            # Marked, and drawn at once, before the part is built: a first opening that takes a while is seen to
            # be under way, under the wait cursor.
            self.shell.rail.set_place(part, page if part == "inference" else None)
            self.shell.rail.repaint()
            self._build(part)
        widget = self.shell.part_widget(part)
        if part == "inference" and part in self._built:
            self._inference_opened = True
        records = self._built.get("records")
        if part == "records" and page is not None and records is not None:
            found = records.kind_combo.findData(page)
            if found >= 0:
                records.kind_combo.setCurrentIndex(found)
        home = self._built.get("home")
        if part == "home" and home is not None:
            if self.shell.stack.currentWidget() is home:
                home.refresh()                       # in sight already: read now (nothing while the window is hidden)
            else:
                home.mark_stale()                    # read as it is shown
        self.shell.show_part(part)
        show = getattr(widget, "show_page", None)
        if page is not None and callable(show):
            show(page)
        now, shown = self.current_place()
        self.shell.rail.set_place(now, shown if now == "inference" else None)
        if keyboard is not None:
            self._give_keyboard(widget, keyboard)
        self.place_changed.emit(now, shown)

    @staticmethod
    def _start_of(widget: QWidget) -> QWidget | None:
        """The control the part ``widget`` names to start at, or else its first setting; None when it has neither,
        its every setting locked while a run is live."""
        first = getattr(widget, "first_focus", None)
        target = first() if callable(first) else None
        return target if target is not None else _first_control(widget)

    def _give_keyboard(self, widget: QWidget, reason) -> None:
        """Put the keyboard on the control the part names to start at, or else on its first setting. A page with
        neither -- its every setting locked while a run is live -- gives it to its row of the rail, so that the
        keyboard is never left on the page a move left. When a shortcut gives the keyboard to a control that holds
        it already, the ring shows there too: the toolkit tells a control nothing of a focus it holds, so the
        follower that draws the focus ring is told here."""
        target = self._start_of(widget)
        if target is None:
            part, page = self.current_place()
            target = self.shell.rail.row(page if part == "inference" and page else part)
        held = target.hasFocus()
        target.setFocus(reason)
        follower = style.current_follower()
        if held and reason == Qt.ShortcutFocusReason and follower is not None:
            follower.arrived(target, reason)

    def _to_page(self) -> None:
        """The keyboard from the rail to the page in sight, as a key gives it: to the control the page names to
        start at, or else to its first setting; on a page whose every setting a run has locked, to its first
        control that is not a button (a scrolling area, a log or a view, which the keys scroll); on a page with
        none of these it stays where it is."""
        page = self.shell.stack.currentWidget()
        target = self._start_of(page) or _first_control(page, takes=lambda widget: not _is_a_button(widget))
        if target is not None:
            target.setFocus(Qt.ShortcutFocusReason)

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
            # the row the move marked: the step's, or the part's own while Parameter inference shows no step (its
            # error page)
            rail.row(shown if now == "inference" and shown else now).setFocus(Qt.OtherFocusReason)

    def _rail_left(self) -> None:
        """Escape in the rail: the page in sight takes the keyboard."""
        self._to_page()

    def _key_pressed(self) -> None:
        """A key of the map was pressed: what its row says. The key of a part or of a step opens that place as a
        move does, the page taking the keyboard as after a key; the key to the run goes to the place of the live
        run, which the foot is a button for, and does nothing while it is none; the key between the rail and the
        page moves the keyboard from the one to the other."""
        does = self.sender().property("does")
        if does == "rail or page":
            self._rail_or_page()
        elif does == "run":
            self._to_run(Qt.ShortcutFocusReason)
        else:
            kind, name = does.split(" ", 1)
            if kind == "part":
                self._move(name, None, keyboard=Qt.ShortcutFocusReason)
            else:
                self._move("inference", name, keyboard=Qt.ShortcutFocusReason)

    def _rail_or_page(self) -> None:
        """The key between the rail and the page, either way round: from the rail, a row of it or the run's foot,
        the page in sight takes the keyboard; from anywhere else the rail takes it."""
        held = self.focusWidget()
        if held is not None and self.shell.rail.isAncestorOf(held):
            self._to_page()
        else:
            self.shell.rail.take_keyboard()

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
        a window nobody is looking at. It builds no part. The run's title and the words of its place are worked
        out ONCE per run, not per tick.
        """
        self._running_panel = panel
        inference = self._built.get("inference")
        if inference is not None:
            inference.recompute_states()
        else:
            self._states_changed(self._step_states())
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
        """The first panel of type ``cls`` among the parts built (convenience for callers + tests); when none holds
        one, the part that holds that kind of panel is built and asked. None when no part holds such a panel."""
        found = next((p for p in self._all_panels() if isinstance(p, cls)), None)
        for kind, name in _PANEL_PARTS.items():
            if found is None and issubclass(kind, cls) and name not in self._built:
                self.part(name)
                found = next((p for p in self._all_panels() if isinstance(p, cls)), None)
        return found

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
        # The close is accepted: the lock and the listener are let go now, not when the event loop ends, which a
        # figure popped out of the window keeps running, so that the next start of PRISM opens a window of its own.
        one_instance.release()
        super().closeEvent(event)

    def _save_state(self):
        """Persist window geometry + the selections of each panel of the parts built in this launch: a part never
        opened writes nothing and keeps what it remembered. Called only when a close is accepted."""
        qs = settings.settings()
        qs.setValue("window/geometry", self.saveGeometry())
        for panel in self._all_panels():
            panel.save_settings(qs)
        # BY NAME: the browser is a QWidget, not a BasePanel, so it is not in _all_panels() and
        # gets no sweep for free -- which is also what keeps it out of _refresh_model_combos.
        records = self._built.get("records")
        if records is not None:
            records.save_settings(qs)
        qs.sync()
