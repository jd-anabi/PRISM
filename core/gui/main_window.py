"""The MAPPI main window: a NavShell (persistent "MAPPI" title + back arrow) over a home/splash screen
and four section screens. Replaces the old flat four-tab layout; the section panels are reused
unchanged in behaviour -- only where they are mounted changes. Cross-validation now lives inside the
FDT Analysis section, and the SBI panel is split into the Parameter Inference section's gated tabs."""
from PySide6.QtWidgets import QMainWindow, QMessageBox

from . import settings
from .panels.base_panel import BasePanel
from .panels.crossval_panel import CrossValPanel
from .panels.fdt_panel import FdtPanel
from .panels.reduction_panel import ReductionPanel
from .screens.home_screen import HomeScreen
from .screens.inference_screen import InferenceScreen
from .screens.nav_shell import NavShell
from .screens.section_screen import SectionScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAPPI — hair-cell parameter inference & FDT analysis")
        self.resize(1300, 820)

        self.nav = NavShell()
        self.setCentralWidget(self.nav)

        # Section screens (built once; their panels are the existing ones, reused verbatim).
        self.reduction_screen = SectionScreen(
            "Reduction Map", [("NWK → Hopf reduction map", ReductionPanel())])
        self.fdt_screen = SectionScreen(
            "FDT Analysis",
            [("FDT analysis", FdtPanel()), ("Sweep study cross-validation", CrossValPanel())])
        self.inference_screen = InferenceScreen("Parameter Inference")

        home = HomeScreen(live_sections={"Reduction Map", "FDT Analysis", "Parameter Inference"})
        self.nav.add_screen(home)                                    # index 0 -- Home
        idx_red = self.nav.add_screen(self.reduction_screen)
        idx_fdt = self.nav.add_screen(self.fdt_screen)
        idx_inf = self.nav.add_screen(self.inference_screen)
        self._section_index = {"Reduction Map": idx_red, "FDT Analysis": idx_fdt,
                               "Parameter Inference": idx_inf}
        home.navigate.connect(lambda name: self.nav.go_to(self._section_index[name]))
        self.nav.go_home()                                          # ALWAYS open on Home

        # Restore geometry only. Each panel restored its own selections in its __init__; we deliberately
        # do NOT restore the last screen -- the app always opens on Home.
        qs = settings.settings()
        geom = qs.value("window/geometry")
        if geom is not None:
            self.restoreGeometry(geom)

    def _all_panels(self):
        return (self.reduction_screen.panels() + self.fdt_screen.panels()
                + self.inference_screen.panels())

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
        qs.sync()
