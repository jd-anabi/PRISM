"""The PRISM main window: a NavShell (persistent "PRISM" title + back arrow) over Home,
four section screens, the Artifacts browser, and two Settings-reached screens (the Settings/Help screen
and the user-defined model builder). Replaces the old flat four-tab layout; the section panels are
reused unchanged in behaviour -- only where they are mounted changes. Cross-validation now lives inside
the FDT Analysis section, and the SBI panel is split into the Parameter Inference section's gated
tabs."""
import time
from datetime import datetime

from PySide6.QtCore import QTimer
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QMainWindow, QMenu, QMessageBox

from core import registry
from core.artifacts import KIND_DIRS, StoreError, default_store
from core.config import VALID_MODELS
from core.Helpers import model_store

from . import chain, design, keys, record_words, settings, theming
from .panels.base_panel import RUN_STATE, BasePanel
from .panels.crossval_panel import CrossValPanel
from .panels.fdt_panel import FdtPanel
from .panels.reduction_panel import ReductionPanel
from .panels.simulate_panel import SimulatePanel
from .screens.artifact_screen import ArtifactScreen
from .screens.home_screen import HomeFacts, HomeScreen, RecordRow
from .screens.inference_screen import InferenceScreen
from .screens.model_builder_screen import ModelBuilderScreen
from .screens.nav_shell import NavShell, running_banner
from .screens.section_screen import SectionScreen
from .screens.settings_screen import SettingsScreen
from .widgets.artifact_picker import StorePicker


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


# Home's places in the stack the window holds today: the stack's name of the screen each part opens. Help's words
# stand on the Settings screen, and a model of your own is made in the model builder; Home is the stack's first.
_HOME_PLACES = {"home": None, "reduction": "Reduction Map", "temperature": "FDT Analysis",
                "inference": "Parameter Inference", "live": "Simulate", "records": "Artifacts",
                "settings": "Settings", "help": "Settings", "models": "Model builder"}
# a part's pages that are tabs of its screen today, in the order of the tabs
_TABBED = {"temperature": tuple(place.key for place in keys.PAGES["temperature"]),
           "inference": tuple(place.key for place in keys.STEPS)}


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
    """The application shell: a NavShell over Home plus the five section screens.

    Always opens on Home -- the last screen is deliberately not restored. Owns the QSettings write
    that runs at quit (``_save_state``, reached from closeEvent): panel selections, and the Artifacts
    screen's own (called by name, because _all_panels() is panel-typed and that screen is not). Two
    other write sites exist on purpose: BasePanel._persist_layout writes a splitter position on a
    1500 ms debounce (save-on-clean-quit lost the drag -- see base_panel.py), and appearance settings
    are written eagerly, as they are applied immediately.
    """
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

        self.nav = NavShell()
        self.setCentralWidget(self.nav)

        # Section screens (built once; their panels are the existing ones, reused verbatim).
        self.reduction_screen = SectionScreen(
            "Reduction Map", [("NWK → Hopf reduction map", ReductionPanel())])
        self.fdt_screen = SectionScreen(
            "FDT Analysis",
            [("FDT analysis", FdtPanel()), ("Sweep study cross-validation", CrossValPanel())])
        self.inference_screen = InferenceScreen("Parameter Inference")
        self.simulate_screen = SectionScreen(
            "Simulate", [("Live simulation", SimulatePanel())])

        # Home reads what it states as it is built, the run slot's panel among it: none, at launch.
        self._running_panel = None
        self.home_screen = HomeScreen(self._home_facts)
        self.nav.add_screen(self.home_screen)                        # index 0 -- Home
        idx_red = self.nav.add_screen(self.reduction_screen)
        idx_fdt = self.nav.add_screen(self.fdt_screen)
        idx_inf = self.nav.add_screen(self.inference_screen)
        idx_sim = self.nav.add_screen(self.simulate_screen)

        # The artifact browser: AFTER the four sections and BEFORE the Settings screen, so the
        # back-arrow slide direction stays monotone with the order of the sections. A plain QWidget, not a
        # BasePanel -- a run anywhere must not grey out the log you are reading.
        self.artifact_screen = ArtifactScreen()
        idx_artifacts = self.nav.add_screen(self.artifact_screen)
        self.artifact_screen.store_changed.connect(self._refresh_store_pickers)

        # The Settings/Help screen: reached from the gear popover's "Full settings…" and from Home's rows
        # for Settings and Help; the back arrow returns Home.
        current_mode = (appearance.mode() if appearance is not None
                        else settings.get_appearance(settings.settings()))
        self.settings_screen = SettingsScreen(self.set_appearance_mode, current_mode,
                                              on_open_builder=self._open_model_builder,
                                              on_edit_model=self._edit_user_model,
                                              on_delete_model=self._delete_user_model)
        idx_settings = self.nav.add_screen(self.settings_screen)

        # The model builder: reached from the Settings "User-defined models" group and from Home's row for
        # your own models; its own Back button returns to Settings, the nav back arrow still returns Home.
        self.model_builder_screen = ModelBuilderScreen(
            on_saved=self._on_user_models_changed,
            on_back=lambda: self.nav.go_to(self._section_index["Settings"]))
        idx_builder = self.nav.add_screen(self.model_builder_screen)

        self._section_index = {"Reduction Map": idx_red, "FDT Analysis": idx_fdt,
                               "Parameter Inference": idx_inf, "Simulate": idx_sim,
                               "Artifacts": idx_artifacts,
                               "Settings": idx_settings, "Model builder": idx_builder}
        # Home asks for a place by its part and page, which today's stack and tabs stand for. Home is read
        # again at its next opening once another screen has been shown.
        self.home_screen.navigate.connect(self._go_from_home)
        self.nav.stack.currentChanged.connect(self._on_screen_changed)

        # Where each panel LIVES. A BasePanel carries no title of its own, so this is what lets the
        # header's run slot say "Posterior" and jump there. Built ONCE, off the screens' own tab
        # widgets -- the same QTabWidgets the screens were built from, and the only thing MainWindow
        # already has that pairs a panel with a label -- so a renamed tab cannot leave the banner
        # naming a tab that is gone. Both screen kinds expose `.tabs` and `panels()` in tab order, so
        # one loop covers the generic sections and the inference screen alike. A panel that is NOT in
        # the map (the model builder, which is not a BasePanel, or a future screen) gets the banner
        # with no destination rather than a KeyError inside a signal handler.
        self._tab_screens = [(name, self.nav.stack.widget(self._section_index[name]))
                             for name in ("Reduction Map", "FDT Analysis", "Parameter Inference",
                                          "Simulate")]
        self._panel_home = {}
        for name, screen in self._tab_screens:
            for tab_index in range(screen.tabs.count()):
                self._panel_home[screen.tabs.widget(tab_index)] = (
                    name, screen.tabs.tabText(tab_index), self._section_index[name], tab_index)

        # The run slot, EMPTY at launch: the timer is constructed here but never started, and an
        # unstarted QTimer registers nothing with the OS, so nothing on the launch path ticks.
        self._run_title = ""
        self._run_started = 0.0
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(1000)
        self._run_timer.timeout.connect(self._tick_running)
        self.nav.btn_running.clicked.connect(self._go_to_running)
        RUN_STATE.changed.connect(self._on_run_state)

        self._build_settings_menu(current_mode)
        self.nav.go_home()                                          # ALWAYS open on Home

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

    def _build_settings_menu(self, current_mode):
        """Attach the gear's appearance popover: the three exclusive choices + 'Full settings…'."""
        menu = QMenu(self)
        group = QActionGroup(self)
        group.setExclusive(True)
        self._appearance_actions = {}
        for mode, label in theming.MODE_LABELS:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(mode == current_mode)         # setChecked emits toggled, not triggered
            group.addAction(act)
            act.triggered.connect(lambda _checked=False, m=mode: self.set_appearance_mode(m))
            self._appearance_actions[mode] = act
        menu.addSeparator()
        full = menu.addAction("Full settings…")
        full.triggered.connect(lambda: self.nav.go_to(self._section_index["Settings"]))
        self.nav.btn_settings.setMenu(menu)

    def set_appearance_mode(self, mode):
        """Apply + persist an appearance mode, and keep the gear menu and Settings screen in sync.

        Safe to call from either entry point (gear action or Settings radio): the reflected updates use
        setChecked (which emits `toggled`, not `triggered`) and the Settings radio blocks its own signal,
        so there is no feedback loop.
        """
        qs = settings.settings()
        settings.set_appearance(qs, mode)
        qs.sync()
        if self.appearance is not None:
            self.appearance.set_mode(mode)
        act = self._appearance_actions.get(mode)
        if act is not None and not act.isChecked():
            act.setChecked(True)
        self.settings_screen.set_mode(mode)

    # ── user-defined models (Settings group + builder screen) ─────────────────
    def _open_model_builder(self):
        self.model_builder_screen.reset()
        self.nav.go_to(self._section_index["Model builder"])

    def _tell(self, title: str, text: str, icon) -> None:
        """One plain box with an Ok button: an INSTANCE dialog shown with .exec().

        Never QMessageBox.warning/information. Those are C++ statics, and the suite's session guard
        patches the instance method only (tests/conftest.py::_no_modal_dialogs): offscreen a static
        spins a nested event loop nothing closes, so a test that reaches one stalls instead of
        failing. Every box in this window now goes through here or is built inline like the
        confirmation below and the one in closeEvent.
        """
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def _edit_user_model(self, name: str):
        try:
            self.model_builder_screen.load_existing(name)
        except Exception as e:                       # noqa: BLE001 -- a corrupt file must not crash
            self._tell("Cannot edit model", f"Could not load '{name}':\n{e}", QMessageBox.Warning)
            return
        self.nav.go_to(self._section_index["Model builder"])

    def _delete_user_model(self, name: str):
        if BasePanel._running:
            self._tell("A task is running",
                       "Wait for the running task to finish before deleting a model.",
                       QMessageBox.Information)
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Delete model")
        box.setText(f"Delete the user-defined model '{name}'?")
        box.setInformativeText("This removes its definition and its generated Bounds/Cells/Units files.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec() != QMessageBox.Yes:
            return
        try:
            model_store.delete_user_model(name)
        except Exception as e:                       # noqa: BLE001
            self._tell("Delete failed", str(e), QMessageBox.Warning)
            return
        registry.unregister(name)
        self._on_user_models_changed()

    def _on_user_models_changed(self, _name: str = ""):
        """After a save/delete: re-sync every model combo + the Settings list."""
        self._refresh_model_combos()
        self.settings_screen.refresh_models()

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
        return (self.reduction_screen.panels() + self.fdt_screen.panels()
                + self.inference_screen.panels() + self.simulate_screen.panels())

    # ── Home ──────────────────────────────────────────────────────────────────
    def _home_facts(self) -> HomeFacts:
        """What Home states, read now: the inference screen's session and its six steps, the part and page a
        live run is in, and every record of the store in use. The narrowing round has an observation to
        choose when its panel's own list holds one."""
        session = self.inference_screen.session
        live_place = self._run_place(self._running_panel)
        live = live_place[1] if live_place is not None and live_place[0] == "inference" else None
        offered = self.inference_screen.tsnpe_panel.obs_picker.has_entries()
        states = chain.step_states(session, live=live, observation_on_offer=offered, label=record_words.label_of)
        return HomeFacts(session, states, live_place, _home_records(default_store()))

    def _run_place(self, panel):
        """The part and page a panel's run is in: None for no panel, or for one with no known home."""
        where = self._panel_home.get(panel) if panel is not None else None
        if where is None:
            return None
        part = next(name for name, screen in _HOME_PLACES.items() if screen == where[0])
        pages = _TABBED.get(part, ())
        return part, (pages[where[3]] if pages else None)

    def _go_from_home(self, part: str, page) -> None:
        """Open the place Home asks for: a part's screen; a page of Effective temperature, or a step, at its
        tab when that tab can be opened (a greyed one leaves the tab shown); the records screen at a kind. A
        place the window does not hold opens nothing."""
        if part not in _HOME_PLACES:
            return
        if part == "records" and page is not None:
            found = self.artifact_screen.kind_combo.findData(page)
            if found >= 0:
                self.artifact_screen.kind_combo.setCurrentIndex(found)
        screen = _HOME_PLACES[part]
        self.nav.go_to(0 if screen is None else self._section_index[screen])
        pages = _TABBED.get(part, ())
        if page in pages:
            tabs = (self.inference_screen if part == "inference" else self.fdt_screen).tabs
            if tabs.isTabEnabled(pages.index(page)):
                tabs.setCurrentIndex(pages.index(page))

    def _on_screen_changed(self, index: int) -> None:
        """A screen other than Home is shown: Home is read again at its next opening, since what it states
        may change while it is away."""
        if self.nav.stack.widget(index) is not self.home_screen:
            self.home_screen.mark_stale()

    # ── the live run, visible app-wide ────────────────────────────────────────
    def _on_run_state(self, panel) -> None:
        """A run started (``panel``) or ended (None): fill or clear the shell's run slot.

        Connected to the module-level RUN_STATE in base_panel.py, which _set_busy publishes on.
        EVERY window constructed in the process receives EVERY run-state signal -- including one a
        test has discarded: such a window is never destroyed (PySide holds the lambda slots that
        capture ``self``, which Python's collector cannot reach), so Qt never drops its connection
        and its handler runs, and its timer ticks, on every later run. This handler must therefore
        touch ONLY its own window's widgets and be safe to run on a window nobody is looking at.
        The title is looked up ONCE per run, not per tick.
        """
        self._running_panel = panel
        self._mark_running_section(panel)
        self.home_screen.refresh()                 # read now while Home is shown, else at its next opening
        if panel is None:
            self._run_timer.stop()
            self.nav.set_running(None)
            return
        where = self._panel_home.get(panel)
        self._run_title = where[1] if where is not None else ""
        self._run_started = time.monotonic()
        self._tick_running()                       # paint immediately, not a second from now
        self.nav.btn_running.setToolTip(
            f"{where[0]} → {where[1]}: go to it" if where is not None else "")
        self._run_timer.start()

    def _tick_running(self) -> None:
        """Re-render the elapsed time. One second's work: a divmod and a setText."""
        self.nav.set_running(
            running_banner(self._run_title, int(time.monotonic() - self._run_started)))

    def _go_to_running(self) -> None:
        """Click the run slot: show the panel that is running -- its screen AND its tab, since a run
        on a section's second tab is not found by arriving at the first. A panel with no known home
        does nothing: the banner still says what is running, it just has nowhere to send you."""
        where = self._panel_home.get(self._running_panel)
        if where is None:
            return
        _name, _label, screen_index, tab_index = where
        self.nav.go_to(screen_index)
        tabs = getattr(self.nav.stack.widget(screen_index), "tabs", None)
        if tabs is not None:
            tabs.setCurrentIndex(tab_index)

    def _mark_running_section(self, panel) -> None:
        """Put the running marker on the running section's tab, and clear every other one. ``panel``
        None -- or a panel with no known home -- clears the lot, so a marker can never outlive the run
        that set it. Home carries no marker.

        Every screen is rewritten on every change rather than remembering what was marked: four
        screens is nothing, and the bookkeeping version is what leaves a stale marker behind when a
        run ends on a window that was navigated in between. Nothing is DISABLED here: the app-wide
        control lock (BasePanel._set_busy) already stands and navigation stays free -- you must be
        able to look at another tab while a twenty-minute train runs.
        """
        where = self._panel_home.get(panel)
        name = where[0] if where is not None else None
        tab_index = where[3] if where is not None else None
        for section, screen in self._tab_screens:
            screen.set_running_tab(tab_index if section == name else None)

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
