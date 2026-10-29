"""A tabbed container of figure images. Figures are rendered to PNG on the worker thread (see
BasePanel's fig_sink) and shown here as QPixmaps -- deliberately NOT live FigureCanvasQTAgg widgets:
painting a matplotlib figure that was created on a worker thread deadlocks on matplotlib's global
lock. Static PNGs sidestep that entirely and unify with the FDT/Reduction/CrossVal runners, which
already save PNGs to disk.

Each tab carries a "Pop out" button that re-opens the figure in its own window where it CAN be zoomed,
panned and saved (core/gui/widgets/figure_window.py), a window of its own that no window owns:
  * SBI-panel figures also arrive as a pickled Figure -> a true interactive matplotlib window,
    reconstructed on the GUI thread (never painting the worker's original -- see above).
  * disk-PNG figures (and any figure whose pickle is missing / unloadable) -> a pan/zoom image viewer.
While a run is live every figure pops out as its picture, in the image viewer, and says so on its button, and an
interactive window already open shows its picture until the run has ended (``hold``): the window's thread lays out
no figure while a run builds its own.

While a run's page holds the stack, only the figures that run made are in sight (``only_the_run``): the tabs added
from the run's start (``run_begins``) to its end (``run_ends``). The others stay as they are, out of sight, and come
back when the page's figures are given back; a window popped out from one of them is left as it is. Each tab added or
removed is said (``tabs_changed``), so that the page holding the stack shows its frame only while a tab is in sight.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QScrollArea, QTabWidget, QVBoxLayout,
                               QWidget)

from ..reported import reported

#: The "Pop out" button's tool tip: for a figure that opens as an interactive copy, the same while a run is live, and
#: for a figure that is a picture only.
POP_INTERACTIVE = "Open an interactive, zoomable copy in a new window"
POP_HELD = "Open a zoomable picture in a new window; an interactive copy opens once the run has ended"
POP_PICTURE = "Open a zoomable copy in a new window"


class _FitLabel(QLabel):
    """A QLabel that scales its pixmap DOWN to the available width, keeping aspect ratio.

    Never scales UP past 1:1 -- enlarging a raster figure only blurs it. The full-resolution pixmap
    is retained, so the "Pop out" window still gets the original.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self.setMinimumSize(1, 1)

    def set_source(self, pix: QPixmap) -> None:
        self._source = pix
        self._rescale()

    @reported(QLabel.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        avail = max(1, self.width())
        if self._source.width() <= avail:
            super().setPixmap(self._source)          # already fits: show it 1:1
            return
        super().setPixmap(self._source.scaledToWidth(avail, Qt.SmoothTransformation))


class FigureStack(QTabWidget):
    #: A tab was added, or removed (closed by its own button, or by the saved-run viewer).
    tabs_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabCloseRequested.connect(self._close_tab)
        # Pop-out windows are parentless top-levels; hold them or Python/Qt GCs them the instant show()
        # returns (the lifetime trap BasePanel.dispatch handles for its workers), and so that each is reached while
        # it is open (``hold``). Dropped on close.
        self._windows = set()
        self._added = 0                      # the tabs added so far: each carries its number, in the order it came
        self._run = None                     # the page's last run's tabs by number: [its first, the first after it]
        self._only = False                   # whether only that run's tabs are in sight
        self._before = None                  # the number of the tab in front when they alone came into sight

    def add_figure(self, title: str, png_bytes: bytes, fig_pickle=None):
        """Show a figure rendered to PNG bytes on a worker thread. ``fig_pickle`` (when present) is the
        pickled Figure, so "Pop out" can rebuild an interactive copy on the GUI thread."""
        pix = QPixmap()
        if png_bytes:
            pix.loadFromData(png_bytes, "PNG")
        self._add_tab(title, pix, fig_pickle=fig_pickle, png_path=None)

    def add_png(self, title: str, path):
        """Show a PNG that a runner saved to disk (FDT/Reduction/CrossVal modes). No Figure exists for
        these, so "Pop out" is a pan/zoom image viewer over the file."""
        self._add_tab(title, QPixmap(str(path)), fig_pickle=None, png_path=str(path))

    def _add_tab(self, title: str, pix: QPixmap, *, fig_pickle, png_path):
        # FIT TO WIDTH. Stage figures are rendered at dpi=110 with bbox_inches="tight", routinely
        # 1200-1800px wide, and were shown at full resolution in a scroll area with no scaling -- so
        # an embedded figure was ALWAYS scrolled in both axes, while only the pop-out window had
        # Fit/100% controls. Downscale-only (never enlarge past 1:1, which would just blur it).
        label = _FitLabel()
        label.setAlignment(Qt.AlignCenter)
        if pix.isNull():
            label.setText(f"(could not render figure: {title})")
        else:
            label.set_source(pix)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)

        from ..run.controller import controller
        pop = QPushButton("Pop out")

        container = QWidget()
        # Stash everything the pop-out needs directly on the tab widget, so it travels with the tab
        # (tabs are movable) and is freed when the tab closes.
        container._fig_pickle = fig_pickle
        container._png_path = png_path
        container._pix = pix
        container._title = title
        container._pop = pop
        container._number, self._added = self._added, self._added + 1
        self._tip(container, controller().is_live())
        pop.clicked.connect(lambda: self._pop_out(container))

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addStretch(1)
        top.addWidget(pop)
        layout.addLayout(top)
        layout.addWidget(scroll, 1)

        idx = self.addTab(container, title)
        if self._shows(container):
            self.setCurrentIndex(idx)
        else:
            self.setTabVisible(idx, False)       # a figure of no run the page shows comes into sight with the others
        self.tabs_changed.emit()                 # said once the tab is in sight, or not, as it stays

    @staticmethod
    def _tip(container, live: bool) -> None:
        """Say on ``container``'s "Pop out" what it opens: an interactive copy, or, while a run is live (``live``),
        its picture until the run has ended; a picture alone for a figure that has no interactive copy."""
        if getattr(container, "_fig_pickle", None) is None:
            container._pop.setToolTip(POP_PICTURE)
        else:
            container._pop.setToolTip(POP_HELD if live else POP_INTERACTIVE)

    def _pop_out(self, container):
        """Open the figure of ``container`` in a window of its own, which no window owns: an interactive copy when
        its pickle loads, else its picture in the image viewer. While a run is live it opens as its picture: the
        window's thread lays out no figure while a run builds its own."""
        from ..run.controller import controller
        from .figure_window import ImageZoomWindow, build_interactive_window

        title = getattr(container, "_title", "Figure")
        window = None
        if getattr(container, "_fig_pickle", None) is not None and not controller().is_live():
            try:
                window = build_interactive_window(container._fig_pickle, title, on_close=self._drop_window,
                                                  picture=container._pix)
            except Exception:      # noqa: BLE001 -- a version-skewed / unloadable pickle: fall back
                window = None
        if window is None:
            pix = container._pix
            if container._png_path:                       # prefer the crisp full-res file on disk
                disk = QPixmap(container._png_path)
                if not disk.isNull():
                    pix = disk
            window = ImageZoomWindow(pix, title, on_close=self._drop_window)
        self._windows.add(window)
        window.show()
        window.raise_()

    def hold(self, on: bool) -> None:
        """While a run is live (``on``), every interactive window this stack opened shows its figure's picture and
        lays out nothing of its figure, and every "Pop out" says it opens a picture; off, each is interactive again,
        at the view it had, and says so."""
        for index in range(self.count()):
            if hasattr(self.widget(index), "_pop"):
                self._tip(self.widget(index), on)
        for window in list(self._windows):
            held = getattr(window, "hold", None)
            if callable(held):
                held(on)

    def run_begins(self) -> None:
        """A run of the page has begun: the tabs added from now until it ends are its figures."""
        self._run = [self._added, None]

    def run_ends(self) -> None:
        """The page's run has ended: no tab added from now on is a figure of it."""
        if self._run is not None and self._run[1] is None:
            self._run[1] = self._added

    def only_the_run(self, on: bool) -> None:
        """While the run's page holds the stack (``on``), only the figures its last run made are in sight, the newest
        of them in front unless one of them was in front already; off, every tab is in sight again where it stands, in
        the order the tabs came unless a person moved one, and the tab that was in front before is in front again,
        unless a figure of the run came since or that tab has been closed. That tab is kept by its number, never the
        tab itself: a closed tab's page is deleted, and touching it raises, where a number no open tab carries is
        simply not found. Which tab is in front is read before any tab is hidden, as hiding the one in front moves the
        front to the next in sight."""
        front = self.currentWidget()
        if on and not self._only:
            self._before = getattr(front, "_number", None)
        self._only = on
        for index in range(self.count()):
            self.setTabVisible(index, self._shows(self.widget(index)))
        shown = [index for index in range(self.count()) if self.isTabVisible(index)]
        if on and shown and (front is None or not self._of_the_run(front)):
            self.setCurrentIndex(shown[-1])
        elif not on:
            ran = front is not None and self._of_the_run(front)
            back = None if ran else self._numbered(self._before)
            if back is None:
                back = front
            if back is not None:
                self.setCurrentWidget(back)
            self._before = None

    def _numbered(self, number):
        """The open tab that carries ``number``; None for none (a closed tab's number, or no number)."""
        for index in range(self.count()):
            if number is not None and getattr(self.widget(index), "_number", None) == number:
                return self.widget(index)
        return None

    def in_sight(self) -> int:
        """How many tabs are in sight."""
        return sum(1 for index in range(self.count()) if self.isTabVisible(index))

    def _of_the_run(self, container) -> bool:
        """Whether the tab ``container`` is a figure of the page's last run: added from its start to its end."""
        number, run = getattr(container, "_number", None), self._run
        return (number is not None and run is not None and run[0] <= number
                and (run[1] is None or number < run[1]))

    def _shows(self, container) -> bool:
        """Whether the tab ``container`` is in sight: every tab, or only the run's while its page holds the stack."""
        return not self._only or self._of_the_run(container)

    def _drop_window(self, window):
        self._windows.discard(window)

    def png_path(self, index: int) -> "str | None":
        """The file the tab at ``index`` shows when ``add_png`` opened it; None for a figure that
        arrived as bytes (``add_figure``) and for an index with no tab. A saved-run viewer asks this so
        a PNG the run's own watcher already put up is not opened a second time
        (core/gui/panels/record_view.py)."""
        return getattr(self.widget(index), "_png_path", None)

    def tabRemoved(self, index: int) -> None:
        """A tab has gone, by whatever closed it: said, as a tab added is."""
        super().tabRemoved(index)
        self.tabs_changed.emit()

    def _close_tab(self, index: int):
        widget = self.widget(index)
        self.removeTab(index)
        widget.deleteLater()

    def clear_all(self):
        while self.count():
            self._close_tab(0)
