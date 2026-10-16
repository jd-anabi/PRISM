"""Effective temperature: the row of its page names over the page shown, One cell or Sweep study.

A name chosen in the row shows its page. The part opens on One cell in every launch and shows the page last shown
after that; the page shown is never stored. Each page is today's panel, unchanged: One cell the measurement on
one cell, Sweep study the sweep study's cross-validation, each with its own box for comparing saved runs.
"""
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .. import keys
from ..panels.crossval_panel import CrossValPanel
from ..panels.fdt_panel import FdtPanel
from ..widgets.adaptive_stack import AdaptiveStack
from ..widgets.page_names import PageNames

# the keys of the two pages, in the row's order
_PAGES = tuple(place.key for place in keys.PAGES["temperature"])


class TemperatureScreen(QWidget):
    """The row of page names over a stack of the two pages, whose height follows the page shown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fdt_panel = FdtPanel()
        self.crossval_panel = CrossValPanel()
        self.names = PageNames(keys.PAGES["temperature"])
        self._pages = AdaptiveStack()
        for panel in self.panels():
            self._pages.addWidget(panel)
        self.names.chosen.connect(self.show_page)
        # The names stand as far in from the page's sides and top as each panel stands its own first frame, so
        # that they line up with the frame under them and the row's line runs across the page's width.
        inset = self.fdt_panel.layout().contentsMargins()
        row = QVBoxLayout()
        row.setContentsMargins(inset.left(), inset.top(), inset.right(), 0)
        row.addWidget(self.names)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(row)
        layout.addWidget(self._pages, 1)

    def show_page(self, name: str) -> None:
        """Show the page whose key is ``name`` and mark its name in the row; KeyError for a key that is no page's,
        before anything moves."""
        self.names.set_current(name)
        self._pages.setCurrentIndex(_PAGES.index(name))

    def current_page(self) -> str:
        """The key of the page shown: One cell's until another is shown."""
        return _PAGES[self._pages.currentIndex()]

    def panels(self) -> list:
        """The two pages' panels, One cell's first."""
        return [self.fdt_panel, self.crossval_panel]
