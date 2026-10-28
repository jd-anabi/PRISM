"""The PRISM main window: the rail down its left edge and, beside it, the page of the part in sight -- Home,
Parameter inference (the six steps over one session, each a page of its own), Effective temperature (one cell,
or a sweep study), Live simulation, Reduction map, Records, Your own models (their list and the model builder),
Settings and Help. The window is built with Home alone; every other part is built the first time it is opened,
and one that cannot be built shows why in its place while the rest of the window works. The panels inside the
parts are reused unchanged in behaviour. ``go_to`` is the one way to move between the parts; the run's state
stands at the rail's foot."""
import logging
import sys
import traceback
from datetime import datetime

from PySide6.QtCore import QEvent, QEventLoop, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QApplication, QMainWindow,
                               QPlainTextEdit, QPushButton, QTabBar, QTextEdit, QToolButton, QWidget)

from core.artifacts import KIND_DIRS, StoreError, default_store
from core.config import VALID_MODELS

from . import chain, design, keys, one_instance, record_words, settings, style, this_computer
from .panels.base_panel import BasePanel
from .panels.crossval_panel import CrossValPanel
from .panels.fdt_panel import FdtPanel
from .panels.inference_tabs import ConfigPanel, InferPanel, PosteriorPanel, PriorPanel, TSNPEPanel, ValidatePanel
from .panels.reduction_panel import ReductionPanel
from .panels.simulate_panel import SimulatePanel
from .reported import reported
from .run import ends, history, questions
from .run.controller import controller
from .screens.artifact_screen import ArtifactScreen
from .screens.help_screen import HelpScreen
from .screens.home_screen import LAST_RUNS, HomeFacts, HomeScreen, RecordRow, RunRow
from .screens.inference_screen import InferenceScreen
from .screens.models_screen import PAGES as MODEL_PAGES, ModelsScreen
from .screens.settings_screen import SettingsScreen
from .screens.shell import PartError, Shell
from .screens.temperature_screen import TemperatureScreen
from .session import SbiSession
from .widgets.artifact_picker import StorePicker
from .widgets.ask import ask
from .widgets.page_names import PageNames

log = logging.getLogger(__name__)

# How often a close that waited for a run, the run having ended, looks again for a box or a question still open.
_ANSWERED_EVERY_MS = 200
#: What the window's log says when Windows asks to end the session, and when that end is cancelled.
SESSION_ASKED_TEXT = "Windows asked to end the session."
SESSION_CANCELLED_TEXT = "Windows' end of the session was cancelled."
#: What the window's log says when a run's page cannot be opened, its place being one this window does not have: the
#: run and its place. Its end is seen all the same.
PAGE_GONE_TEXT = "The page of the run %s no longer exists in this PRISM (%s); its end is taken as seen."


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


def _last_runs(folder, records_root: str) -> tuple:
    """The latest runs of the records folder ``records_root``, newest first, as Home lists them, read from the
    history files in ``folder``: each with its word and name, when it started and ended, in local time, and how it
    ended. None when the folder cannot be read."""
    try:
        records = history.latest(folder, records_root=records_root, n=LAST_RUNS)
    except OSError:
        return ()
    rows = []
    for record in records:
        if isinstance(record.get("id"), str):
            label = f"{record.get('word') or 'Run'} {record.get('name') or ''}".strip()
            rows.append(RunRow(record["id"], label, _local(record.get("started")), _local(record.get("ended")),
                               str(record.get("outcome") or "running")))
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
        # Where each panel of a part built lives: its part, and its page or step.
        self._places: dict = {}
        self._inference_opened = False           # Parameter inference opens at the step in hand the first time
        self._run_live = False                   # whether the run's status said a run is live, as last heard
        # What the steps' states read of the run's status, as they were last worked out from it: worked out again
        # only when it changes.
        self._run_key = None
        # Every record of the store in use and the latest runs of its records folder, as Home last read them: read
        # again only after a run has ended, as Home is opened and when the store changes (None: at Home's next reading).
        self._stored = None
        self._close_at_end = False               # the close was asked for: the window closes once the run has ended
        self._ends_tries = False                 # ... asked while the run waited to be tried again: it ended the tries
        # From Windows' question to end the session until it is cancelled the close asks nothing.
        self._session_ending = False
        # At a launch: a run whose history still says it is running ended unseen when an earlier PRISM closed, and is
        # said so; only by the PRISM that holds the one-PRISM claim, since one started without the claim may stand
        # beside a PRISM whose run is live. Then the newest run of the records folder in use whose end nobody has seen
        # becomes the run's status, which the foot and Home say until it is seen.
        held = controller()
        if one_instance.current() is not None and not held.is_live():
            try:
                history.close_unseen(held.folder())
            except (OSError, ValueError):
                pass                             # a folder of runs that cannot be read or written closes nothing
        held.recall(str(default_store().root))
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

        # The process's one run is heard through a method of this window's own, so that a deleted window leaves no
        # connection behind; and its reason against ending the session goes on this window, the newest built. The
        # foot runs no clock: it says what the run's status says, at each change of it. Every move is looked at for
        # an end whose page has come into sight.
        controller().status_changed.connect(self._on_status)
        controller().use_window(self)
        self.place_changed.connect(self._seen)

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
        # A window built while a run is live shows it, as a part built then comes up locked; and so does one built
        # while an end waits to be seen.
        if controller().status.state != "idle":
            self._on_status(controller().status)

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
            widget.open_run.connect(self._open_run)
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
        if controller().is_live():
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
        states are worked out again, so that the rail and Home follow a record written or deleted, Home reading the
        store again. Before Parameter inference is built, Home reads again."""
        self._refresh_store_pickers()
        self._stored = None
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
        it is built, a session that holds nothing) and the run's status; and, as last read, every record of the store
        in use and the latest runs of its records folder, which are read again only after a run has ended, as Home
        is opened and when the store changes, never at a change of the run's status, which comes at every save and
        every epoch of a training. A run in the history of the records folder in use, or a run live, ends the
        first-time page."""
        inference = self._built.get("inference")
        session = inference.session if inference is not None else SbiSession()
        if self._stored is None:
            store = default_store()
            self._stored = (_home_records(store), _last_runs(controller().folder(), str(store.root)))
        records, runs = self._stored
        status = controller().status
        return HomeFacts(session, self._step_states(), status, records, has_runs=bool(runs) or status.live,
                         runs=runs)

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
            this_computer.flash(self)

    def go_to_run(self) -> None:
        """Go to the run the foot says, as a press of the foot does: the place of the live run; or the page of a run
        whose end nobody has seen, which says how it ended, and its end is then seen (``_open_run``). Nothing while
        nothing runs and no end waits to be seen, nor for a run with no place."""
        self._to_run(Qt.OtherFocusReason)

    def _to_run(self, keyboard) -> None:
        """The move to the run the foot says, the page taking the keyboard for the reason ``keyboard``."""
        status = controller().status
        if status.place is None:
            return
        if status.live:
            self._move(*status.place, keyboard=keyboard)
        elif status.unacknowledged_end and status.run is not None:
            self._open_run(status.run, keyboard)

    def _open_run(self, run: str, keyboard=Qt.OtherFocusReason) -> None:
        """Open the page of the run ``run`` at its place, as the foot, Home's next step and a row of Home's "Last runs"
        do, and see its end once it has ended: a run this window holds shows as its page stands; one of an earlier
        launch, or any run this window no longer holds, is drawn in its page's past view from its two files. A live
        run is not seen: its end is to come. A page whose own run is live shows that run, and sees no other. Nothing
        for a run whose place is not known; a run whose place this window does not have, whose page cannot be opened,
        is seen all the same, so that it is not said again, and the window's log says its page no longer exists."""
        held = controller()
        place = held.where(run)
        if place is None:
            return
        ended = not (held.is_live() and held.status.run == run)
        try:
            if place[0] not in _PARTS:
                raise KeyError(place[0])
            self._move(*place, keyboard=keyboard)
        except KeyError:                         # a place of an earlier PRISM this window no longer has
            log.warning(PAGE_GONE_TEXT, run, "/".join(str(part) for part in place if part))
            if ended:
                held.acknowledge(run)
            return
        panel = next((panel for panel, at in self._places.items() if at == tuple(place)), None)
        shown = panel is None or held.show_past(run, panel)
        if shown and ended:
            held.acknowledge(run)

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
            self._stored = None                      # opened: Home reads the store and the runs again
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
        move does, the page taking the keyboard as after a key; the key to the run goes where the foot goes, and does
        nothing while the foot is no button; the key between the rail and the page moves the keyboard from the one to
        the other."""
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

    # ── the run, at the foot of the rail ──────────────────────────────────────
    def _on_status(self, status) -> None:
        """The run's status changed. The foot says it, from the status alone and with no clock (``ends.foot``): what
        runs and the stage it is in, that it is stopping, waits to be tried again or needs someone, how and when it
        ended until that end has been seen, or that nothing runs; it is a button while a run is live and while an end
        waits to be seen, for a run with a place, and a foot that held the keyboard and is a button no more hands it to
        the rail. When what the steps read of the run changes (it began, ended or was seen to end, or began or ceased
        to stop or to wait to be tried again), the steps' states are worked out again, which has Home read; at any
        other change Home reads again, while it is shown, the run's status and not the store, which it reads again
        only after a run has ended, as it is opened and when the store changes. An end is seen at once when the run's
        page is in sight in the window in front (``_seen``).

        Connected to the process's run controller, which every window built in the process hears until it is
        deleted: this handler touches only its own window's widgets, and is safe to run on a window nobody is
        looking at. It builds no part.

        From the run's start to its end every figure window the window's pages opened shows its figure's picture.
        When the window's close waits for the run, a run that ends stopped or finished closes the window, once all of
        the end has been said (``_close_at_the_end``); one that ends otherwise (a failure, a refusal) leaves the window
        open, its close withdrawn, and its page and its box say how it ended, but for a close that ended the tries of a
        run that waited to be tried again, which closes the window at the failed end it leaves, the failure being the
        one the page said before the close was asked. A close withdrawn with its stop waits no more.
        """
        if self._close_at_end:
            if not status.live:
                self._close_at_end = False
                model = controller().model
                if model is not None and (model.outcome in questions.CLOSES_AT or self._ends_tries):
                    QTimer.singleShot(0, self, self._close_at_the_end)
            elif not status.closing:
                self._close_at_end = False
            self._ends_tries = self._ends_tries and self._close_at_end
        began, ended = status.live and not self._run_live, self._run_live and not status.live
        self._run_live = status.live
        if began or ended:
            for panel in self._all_panels():
                panel.figure_stack.hold(status.live)
        if ended:
            self._stored = None                  # what the run wrote: Home reads the store and the runs again
        key = (status.live, status.state, status.place, status.outcome)
        if key != self._run_key:
            self._run_key = key
            inference = self._built.get("inference")
            if inference is not None:
                inference.recompute_states()
            else:
                self._states_changed(self._step_states())
        else:
            home = self._built.get("home")
            if home is not None:
                home.refresh()
        foot = self.shell.rail.foot
        held = self.focusWidget() is foot
        button = status.place is not None and (status.live or status.unacknowledged_end)
        foot.set_state(*ends.foot(status, controller().now()), button=button)
        if held and not button:
            self.shell.rail.take_keyboard()
        self._seen()

    def _seen(self, *_place) -> None:
        """An end nobody has seen is seen once its run's page is in sight while the window is the one in front: looked
        at as the run ends, at every move and when the window comes to the front; the run's controller then writes
        it into the run's history and is idle again. Nothing is seen from Windows' question to end the session until
        it is cancelled: nobody is looking then, and the next launch says the end."""
        status = controller().status
        if (status.unacknowledged_end and status.run is not None and status.place is not None
                and not self._session_ending
                and self.isActiveWindow() and self.current_place() == tuple(status.place)):
            controller().acknowledge(status.run)

    @reported()
    def changeEvent(self, event):
        """The window came to the front, or went from it: in front, an end whose page is in sight is seen."""
        super().changeEvent(event)
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self._seen()

    def _close_at_the_end(self) -> None:
        """The close that waited for the run, the run having ended: carried out now, or, while a box or a question
        is open (one a person opened meanwhile), once it has been answered, looked at again every
        ``_ANSWERED_EVERY_MS``: a close carried out under an open box would close the box unanswered."""
        if QApplication.activeModalWidget() is not None:
            QTimer.singleShot(_ANSWERED_EVERY_MS, self, self._close_at_the_end)
            return
        self.close()

    # ── Windows' end of a session ─────────────────────────────────────────────
    def session_asked(self, manager) -> None:
        """Windows asks whether the session may end, for a sign-out or a shutdown (the toolkit's session manager,
        inside Windows' message): from now on the window's close asks nothing, and while a run is live Windows is
        answered that PRISM blocks the end, beside the reason the run registered. Said in the window's log."""
        self._session_ending = True
        log.info(SESSION_ASKED_TEXT)
        if controller().is_live():
            manager.cancel()

    def session_ending(self, will_end: bool) -> None:
        """Windows' answer, inside its end message. "Cancel" (``will_end`` False): the window's close asks its
        questions again, which the window's log says. "Shut down anyway": the live run is asked to stop, as a stop
        asked by the person, its history saying that Windows shut down (a training saving its posterior is not
        stopped: the save goes on); then the window's thread waits for the run's end, the event loop turning, with
        no limit of its own: Windows decides how long it lets PRISM live. Heard again, during the wait or after it,
        it stops nothing more: a stop asked again of a stopping run asks nothing."""
        if not will_end:
            self._session_ending = False
            log.info(SESSION_CANCELLED_TEXT)
            return
        self._session_ending = True
        held = controller()
        if held.is_live():
            held.request_stop(reason=questions.SHUT_DOWN_REASON)
        while held.is_live():
            QApplication.processEvents(QEventLoop.AllEvents | QEventLoop.WaitForMoreEvents)

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
        """The window never closes under a live run (starting, running, stopping or waiting to try again): the close
        is refused, then asked about. "Stop the run and close" asks the run to stop, and "Close when it has stopped",
        for a run already stopping, keeps the window waiting too: the window goes to the run's page, which says that
        PRISM closes when the run has stopped and offers to withdraw the stop and the close, and the run's end
        closes it. A training saving its posterior is not stopped: "Close when it is saved" asks no stop, and the window
        goes to the run's page and waits for the save, the page offering to keep PRISM open. "Stop the run and close"
        while the run waits to be tried again ends its tries, and the window closes at the failed end they leave. An
        answer given after the run ended is too late: nothing is stopped, and the window stays. A close while a close
        already waits asks nothing again: the window goes to the run's page, where the close can be withdrawn.

        From Windows' question to end the session until it is cancelled the close asks nothing: the toolkit closes a
        shown window inside Windows' end message, where a question would wait inside that message. With a run live
        the close is refused, which keeps PRISM and its run going until Windows ends them; with none it is accepted.

        A close accepted, with no run live, writes the window's own settings, closes every other window of the
        program (the figure windows the pages opened among them), so that the program's event loop ends, and lets
        go of the one-PRISM claim last, in a ``finally``: after the run's last history write, and also when the
        settings cannot be written, since a window that has closed must not keep the claim."""
        held = controller()
        if held.is_live():
            event.ignore()
            if self._session_ending:
                return
            if self._close_at_end:
                self.go_to_run()
                return
            question = questions.close_question(held.status, held.model)
            if ask(self, question.title, question.sentence, question.safe, question.other,
                   loses_work=question.loses_work, beneath=question.beneath or None):
                tries = held.status.state == "retrying"              # read at the answer: a try may have begun
                held.request_stop(closing=True)
                self._close_at_end = held.is_live() and held.status.closing
                self._ends_tries = self._close_at_end and tries
                if self._close_at_end:
                    self.go_to_run()
            return
        try:
            self._save_state()
        finally:
            try:
                QApplication.closeAllWindows()
            finally:
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
