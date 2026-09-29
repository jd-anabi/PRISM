"""The PRISM navigation shell: a persistent top-left title + back arrow over a stack of screens.

Navigation is two levels deep -- a Home/splash screen (index 0) and the section screens -- so the back
arrow always returns Home. The "PRISM" title stays in the top-left AT ALL TIMES; the back arrow sits
just below it and is hidden on Home.

The header row also carries the app-wide RUN SLOT (``btn_running``): a flat button naming what is
running, where and for how long, hidden whenever nothing is, so the live run is shown app-wide.
MainWindow owns what it says and when -- this module owns only the widget and the wording.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from .. import icons
from ..widgets.anim import slide_screens, snapshot

# The app-wide "a run is live" marker. A PLAIN CHARACTER, not an icons.NAMES entry: the bundled icon
# font defines U+E000..U+E004 only and tests/test_user_models.py pins NAMES against its cmap, so a
# sixth name renders an empty box until the .ttf is regenerated; icons.glyph returns the codepoint
# whenever the FONT loaded, not whenever the glyph exists; and apply_icon sets the WIDGET's font
# family, which a QTabBar cannot do for one tab. LogPane._PREFIX marks its lines the same way.
RUNNING_MARK = "●"


def running_banner(panel_title: str, seconds: int) -> str:
    """"Running: Posterior — 4:07": what is running, and for how long.

    PURE, so a test asserts every shape without waiting for a clock. Under an hour the clock is
    ``m:ss``; from an hour it grows an hours field (``h:mm:ss``) rather than counting to 90 minutes.
    Seconds below zero read as 0 -- a wall-clock jump must not print a negative age -- and a blank
    title reads "a task", which is what a panel MainWindow's destination map does not know shows: the
    banner still says something is running, it just cannot say where.
    """
    secs = max(0, int(seconds))
    hours, rest = divmod(secs, 3600)
    minutes, sec = divmod(rest, 60)
    clock = f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes}:{sec:02d}"
    return f"Running: {panel_title.strip() or 'a task'} — {clock}"


def mark_tabs(tabs, labels, running) -> None:
    """Rewrite every tab's text from ``labels``, prefixing RUNNING_MARK on index ``running``; None
    clears every mark.

    Rewritten from the stored labels rather than edited in place, so marking is IDEMPOTENT and a
    cleared tab is byte-identical to the label it was built with -- the tab titles are read back
    verbatim by tests/test_nav_and_gating.py's control-table pin. Shared by the two screen kinds that
    host tabs (SectionScreen, InferenceScreen), which have no common base but QWidget.
    """
    for index, base in enumerate(labels):
        text = f"{RUNNING_MARK} {base}" if index == running else base
        if tabs.tabText(index) != text:
            tabs.setTabText(index, text)


class NavShell(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.title = QLabel("PRISM")
        self.title.setProperty("type", "subtitle")     # Fluent type ramp (global QSS)

        self.btn_back = QToolButton()
        self.btn_back.setObjectName("navBack")          # -> QToolButton#navBack in the global QSS
        icons.apply_icon(self.btn_back, "back")         # bundled icon font (falls back to "←")
        self.btn_back.setToolTip("Back to the home screen")
        self.btn_back.setAutoRaise(True)
        self.btn_back.clicked.connect(self.go_home)
        self.btn_back.setVisible(False)                  # hidden on Home; shown on any section

        # The app-wide run slot. Created EMPTY and hidden: no icon load, no timer, no store read.
        # The 2026-09-11 taskbar-icon incident was ~150 ms of layout between the native show and the
        # first idle turn, and this button is on that path, so it does nothing at launch.
        # MainWindow fills it from base_panel.RUN_STATE and starts the 1 s clock.
        self.btn_running = QToolButton()
        self.btn_running.setObjectName("navRunning")     # -> QToolButton#navRunning in the global QSS
        self.btn_running.setAutoRaise(True)
        self.btn_running.setVisible(False)

        header = QVBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(2)
        header.addWidget(self.title)
        header.addWidget(self.btn_back, alignment=Qt.AlignLeft)

        header_row = QHBoxLayout()
        header_row.addLayout(header)
        header_row.addWidget(self.btn_running, alignment=Qt.AlignVCenter)
        header_row.addStretch(1)

        self.stack = QStackedWidget()

        # A persistent settings gear in the bottom-left seam (shown on every screen). Its menu is
        # attached by MainWindow (which knows the appearance controller + the Settings screen index).
        self.btn_settings = QToolButton()
        self.btn_settings.setObjectName("navSettings")   # -> QToolButton#navSettings in the global QSS
        icons.apply_icon(self.btn_settings, "settings")  # bundled icon font (falls back to "⚙")
        self.btn_settings.setToolTip("Appearance & settings")
        self.btn_settings.setAutoRaise(True)
        self.btn_settings.setPopupMode(QToolButton.InstantPopup)

        settings_row = QHBoxLayout()
        settings_row.addWidget(self.btn_settings)
        settings_row.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.addLayout(header_row)
        outer.addWidget(self.stack, 1)
        outer.addLayout(settings_row)

    def add_screen(self, widget) -> int:
        """Append a screen; the first one added (index 0) is Home."""
        return self.stack.addWidget(widget)

    def go_home(self) -> None:
        self._navigate(0)

    def go_to(self, index: int) -> None:
        self._navigate(index)

    def _navigate(self, index: int) -> None:
        prev = self.stack.currentIndex()
        # Snapshot the OUTGOING screen before the switch, flip logical state SYNCHRONOUSLY (tests read the
        # index / back-arrow right after with no pump), then slide the incoming screen in over the snapshot.
        old_pm = snapshot(self.stack.currentWidget()) if index != prev else None
        self.stack.setCurrentIndex(index)
        self._sync_back()
        if index != prev:
            slide_screens(self.stack, old_pm, forward=index > prev)   # no-op offscreen / not-yet-visible

    def _sync_back(self) -> None:
        self.btn_back.setVisible(self.stack.currentIndex() != 0)

    def set_running(self, text: "str | None") -> None:
        """Show the run slot with ``text``, or hide and clear it on None. The ONLY writer of the run
        button's text and visibility; MainWindow owns what it says and when (RUN_STATE plus its 1 s
        timer), and sets the tooltip itself right after -- this clears it only on the way out, so a
        hidden slot cannot keep pointing at a run that has finished."""
        if text:
            self.btn_running.setText(text)
            self.btn_running.setVisible(True)
        else:
            self.btn_running.setText("")
            self.btn_running.setToolTip("")
            self.btn_running.setVisible(False)
