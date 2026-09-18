"""The PRISM main window: a NavShell (persistent "PRISM" title + back arrow) over a home/splash screen,
four section screens, the Artifacts browser, and two Settings-reached screens (the Settings/Help screen
and the user-defined model builder). Replaces the old flat four-tab layout; the section panels are
reused unchanged in behaviour -- only where they are mounted changes. Cross-validation now lives inside
the FDT Analysis section, and the SBI panel is split into the Parameter Inference section's gated
tabs."""
import time

from PySide6.QtCore import QTimer
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QMainWindow, QMenu, QMessageBox

from core import registry
from core.config import VALID_MODELS
from core.Helpers import model_store

from . import settings, theming
from .panels.base_panel import RUN_STATE, BasePanel
from .panels.crossval_panel import CrossValPanel
from .panels.fdt_panel import FdtPanel
from .panels.reduction_panel import ReductionPanel
from .panels.simulate_panel import SimulatePanel
from .screens.artifact_screen import ArtifactScreen
from .screens.home_screen import HomeScreen
from .screens.inference_screen import InferenceScreen
from .screens.model_builder_screen import ModelBuilderScreen
from .screens.nav_shell import NavShell, running_banner
from .screens.section_screen import SectionScreen
from .screens.settings_screen import SettingsScreen
from .widgets.artifact_picker import StorePicker


# ── screen-geometry guards ────────────────────────────────────────────────────
# All three degrade to the requested value if Qt cannot tell us about the screen (offscreen tests),
# so they are safe to call unconditionally.
def _available(widget=None):
    """The usable desktop rect (excludes the taskbar), or None when there is no screen."""
    from PySide6.QtGui import QGuiApplication
    screen = (widget.screen() if widget is not None and widget.screen() else
              QGuiApplication.primaryScreen())
    return screen.availableGeometry() if screen is not None else None


def _fits_on_screen(w: int, h: int) -> tuple:
    """Shrink a preferred size to what the screen can actually show, leaving a small margin."""
    rect = _available()
    if rect is None:
        return w, h
    return min(w, rect.width() - 40), min(h, rect.height() - 60)


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


class MainWindow(QMainWindow):
    """The application shell: a NavShell over Home plus the five section screens.

    Always opens on Home -- the last screen is deliberately not restored. Owns the only QSettings
    WRITE site (``_save_state``, reached from closeEvent), so panel selections and layouts persist
    from here; appearance settings are written eagerly instead, as they are applied immediately.
    """
    def __init__(self, appearance=None):
        super().__init__()
        self.appearance = appearance         # theming.Appearance or None (tests build without one)
        self.setWindowTitle("PRISM")
        # 820px exceeded the usable height of a 1366x768 laptop and of 1920x1080 at 150% scaling, so
        # the action buttons started below the fold. Open at the smaller of the preference and what
        # the screen can actually show.
        self.resize(*_fits_on_screen(1300, 820))
        # A window can always be made this small; without it Qt lets the user (or a restored
        # geometry) shrink the shell until the controls column is unusable.
        self.setMinimumSize(900, 600)

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

        self.home_screen = HomeScreen(
            live_sections={"Reduction Map", "FDT Analysis", "Parameter Inference", "Simulate",
                           "Artifacts"})
        self.nav.add_screen(self.home_screen)                        # index 0 -- Home
        idx_red = self.nav.add_screen(self.reduction_screen)
        idx_fdt = self.nav.add_screen(self.fdt_screen)
        idx_inf = self.nav.add_screen(self.inference_screen)
        idx_sim = self.nav.add_screen(self.simulate_screen)

        # The artifact browser: AFTER the four sections and BEFORE the Settings screen, so the
        # back-arrow slide direction stays monotone with the Home tile order. A plain QWidget, not a
        # BasePanel (B1) -- a run anywhere must not grey out the log you are reading.
        self.artifact_screen = ArtifactScreen()
        idx_artifacts = self.nav.add_screen(self.artifact_screen)
        self.artifact_screen.store_changed.connect(self._refresh_store_pickers)

        # The Settings/Help screen: reached only from the gear popover's "Full settings…" (never a Home
        # button), so the back arrow still returns Home.
        current_mode = (appearance.mode() if appearance is not None
                        else settings.get_appearance(settings.settings()))
        qs0 = settings.settings()
        self.settings_screen = SettingsScreen(self.set_appearance_mode, current_mode,
                                              on_open_builder=self._open_model_builder,
                                              on_edit_model=self._edit_user_model,
                                              on_delete_model=self._delete_user_model,
                                              on_system_accent=self._set_system_accent,
                                              system_accent=settings.get_system_accent(qs0),
                                              on_force_inter=self._set_force_inter,
                                              force_inter=settings.get_force_inter(qs0))
        idx_settings = self.nav.add_screen(self.settings_screen)

        # The model builder: reached only from the Settings "User-defined models" group (never a Home
        # tile); its own Back button returns to Settings, the nav back arrow still returns Home.
        self.model_builder_screen = ModelBuilderScreen(
            on_saved=self._on_user_models_changed,
            on_back=lambda: self.nav.go_to(self._section_index["Settings"]))
        idx_builder = self.nav.add_screen(self.model_builder_screen)

        self._section_index = {"Reduction Map": idx_red, "FDT Analysis": idx_fdt,
                               "Parameter Inference": idx_inf, "Simulate": idx_sim,
                               "Artifacts": idx_artifacts,
                               "Settings": idx_settings, "Model builder": idx_builder}
        self.home_screen.navigate.connect(lambda name: self.nav.go_to(self._section_index[name]))

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

        # The run slot, EMPTY at launch (spec §1.2): the timer is constructed here but never started,
        # and an unstarted QTimer registers nothing with the OS, so nothing on the launch path ticks.
        self._running_panel = None
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
                self.resize(*_fits_on_screen(1300, 820))
                _centre_on_primary(self)

    def _build_settings_menu(self, current_mode):
        """Attach the gear's appearance popover: exclusive Light/Dark/Auto/System + 'Full settings…'."""
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

    # ── appearance extras (Settings checkboxes) ───────────────────────────────
    def _set_system_accent(self, enabled: bool):
        """Persist + live-apply the follow-OS-accent toggle (no-op appearance under tests)."""
        qs = settings.settings()
        settings.set_system_accent(qs, enabled)
        qs.sync()
        if self.appearance is not None:
            self.appearance.set_system_accent(enabled)

    def _set_force_inter(self, enabled: bool):
        """Persist + live-apply the Inter-everywhere toggle (custom-painted caches settle on restart)."""
        qs = settings.settings()
        settings.set_force_inter(qs, enabled)
        qs.sync()
        from PySide6.QtWidgets import QApplication
        from . import fonts
        app = QApplication.instance()
        if app is not None:
            fonts.load_app_font(app, prefer_inter=enabled)

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
        """Re-list every artifact picker after the browser changed the store (B8) -- the twin of
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

    # ── the live run, visible app-wide (piece 4, B11) ─────────────────────────
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
        """Put the running marker on the running section's Home tile and its tab, and clear every
        other one. ``panel`` None -- or a panel with no known home -- clears the lot, so a marker can
        never outlive the run that set it.

        Every screen is rewritten on every change rather than remembering what was marked: four
        screens is nothing, and the bookkeeping version is what leaves a stale marker behind when a
        run ends on a window that was navigated in between. Nothing is DISABLED here: the app-wide
        control lock (BasePanel._set_busy) already stands and navigation stays free -- you must be
        able to look at another tab while a twenty-minute train runs (spec §6.1, B11).
        """
        where = self._panel_home.get(panel)
        name = where[0] if where is not None else None
        tab_index = where[3] if where is not None else None
        self.home_screen.set_running_section(name)
        for section, screen in self._tab_screens:
            screen.set_running_tab(tab_index if section == name else None)

    def panel(self, cls):
        """The first panel of type ``cls`` across all screens (convenience for callers + tests)."""
        return next((p for p in self._all_panels() if isinstance(p, cls)), None)

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
        # BY NAME (B1): the browser is a QWidget, not a BasePanel, so it is not in _all_panels() and
        # gets no sweep for free -- which is also what keeps it out of _refresh_model_combos.
        self.artifact_screen.save_settings(qs)
        qs.sync()
