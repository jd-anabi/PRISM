"""A section screen: a heading plus a tab widget of the section's panels.

Used for the simple sections -- Reduction Map (one tab) and FDT Analysis (FDT + Cross-validation). The
Parameter Inference section needs cross-tab gating over a shared session, so it has its own screen
(inference_screen.py) rather than using this generic host.
"""
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget


class SectionScreen(QWidget):
    def __init__(self, title: str, tabs, parent=None):
        """`tabs` is a list of ``(label, panel)`` pairs, added left to right."""
        super().__init__(parent)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.tabs = QTabWidget()
        for label, panel in tabs:
            self.tabs.addTab(panel, label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(heading)
        layout.addWidget(self.tabs, 1)

    def panels(self):
        """The section's panels (for MainWindow persistence + tests)."""
        return [self.tabs.widget(i) for i in range(self.tabs.count())]
