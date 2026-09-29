"""A section screen: a heading plus a tab widget of the section's panels.

Used for the simple sections -- Reduction Map (one tab) and FDT Analysis (FDT + Cross-validation). The
Parameter Inference section needs cross-tab gating over a shared session, so it has its own screen
(inference_screen.py) rather than using this generic host.
"""
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ..widgets.anim import crossfade_tab
from .nav_shell import mark_tabs


class SectionScreen(QWidget):
    """Generic section host: a heading over a tab widget of independent panels.

    Drives nothing itself and persists nothing -- the panels it hosts own both.
    """

    def __init__(self, title: str, tabs, parent=None):
        """`tabs` is a list of ``(label, panel)`` pairs, added left to right."""
        super().__init__(parent)
        heading = QLabel(title)
        heading.setProperty("type", "heading")     # Fluent type ramp (global QSS)

        self.tabs = QTabWidget()
        self._tab_labels: list = []        # the labels as built; mark_tabs rewrites from these
        for label, panel in tabs:
            self.tabs.addTab(panel, label)
            self._tab_labels.append(label)
        # Track the outgoing page + connect AFTER the loop (the first addTab already emitted
        # currentChanged(0) mid-construction, and the handler dereferences _prev_tab).
        self._prev_tab = self.tabs.currentWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(heading)
        layout.addWidget(self.tabs, 1)

    def _on_tab_changed(self, _index):
        crossfade_tab(self.tabs, self._prev_tab)          # no-op offscreen / single-tab / not-yet-visible
        self._prev_tab = self.tabs.currentWidget()

    def panels(self):
        """The section's panels (for MainWindow persistence + tests)."""
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def set_running_tab(self, index: "int | None") -> None:
        """Mark the tab at ``index`` as the one with a live run; None clears every mark."""
        mark_tabs(self.tabs, self._tab_labels, index)
