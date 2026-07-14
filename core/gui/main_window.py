"""The main window: a tab per analysis mode, mirroring the CLI's four modes (core/__main__.py)."""
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget

from . import settings
from .panels.base_panel import BasePanel
from .panels.crossval_panel import CrossValPanel
from .panels.fdt_panel import FdtPanel
from .panels.reduction_panel import ReductionPanel
from .panels.sbi_panel import SbiPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GFDT — hair-cell parameter inference & FDT analysis")
        self.resize(1300, 820)

        self.tabs = QTabWidget()
        self.tabs.addTab(SbiPanel(), "SBI")
        self.tabs.addTab(FdtPanel(), "FDT")
        self.tabs.addTab(ReductionPanel(), "Reduction")
        self.tabs.addTab(CrossValPanel(), "Cross-validation")
        self.setCentralWidget(self.tabs)

        # After resize(), so a saved geometry overrides the default. Each panel restored its own
        # selections in its __init__.
        qs = settings.settings()
        geom = qs.value("window/geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        idx = qs.value("window/tab")
        if idx is not None:
            try:
                self.tabs.setCurrentIndex(int(idx))
            except (ValueError, TypeError):
                pass

    def _panels(self):
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

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
        qs.setValue("window/tab", self.tabs.currentIndex())
        for panel in self._panels():
            panel.save_settings(qs)
        qs.sync()
