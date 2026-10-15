"""Help: what each part's settings mean, in the help text the Settings screen also shows."""
from PySide6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget

from .settings_screen import help_text


class HelpScreen(QWidget):
    """Help: a heading over the help text, which scrolls itself and can be selected."""

    def __init__(self, parent=None):
        super().__init__(parent)
        heading = QLabel("Help")
        heading.setProperty("type", "heading")
        self.viewer = QTextBrowser()
        self.viewer.setHtml(help_text())
        self.viewer.setOpenExternalLinks(True)
        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self.viewer, 1)

    def first_focus(self) -> QWidget:
        """The control the keyboard starts at: the help text, which the keys scroll."""
        return self.viewer
