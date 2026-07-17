"""App appearance (Follow-system / Light / Dark / Auto) + the bespoke Fluent palette & stylesheet.

Each mode resolves to an effective light/dark, then applies:
  * ``styleHints().setColorScheme`` -- keeps NATIVE chrome correct (Win11 title bar, native QFileDialogs).
  * ``design.build_palette`` + ``design.build_qss`` -- the Fluent look. Because the app is otherwise
    palette-driven, these recolour the WHOLE chrome; custom-painted widgets (progress_pane._Sparkline,
    help_badge) read the palette directly, so the palette -- not just the QSS -- must be correct.

Kept out of the test path on purpose: only ``build_app`` constructs an ``Appearance`` (tests build
``MainWindow`` directly), so none of this fires during the suite.

Two rendering surfaces stay outside the palette pipeline: the pyqtgraph live view re-themes itself via the
``theme_changed`` signal (see live_hair_bundle.py), and matplotlib figures stay white (scientific norm).
"""
from datetime import datetime

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from . import design

# (mode key, human label), in display order -- shared by the gear popover and the Settings screen.
MODE_LABELS = (("system", "Follow system"), ("light", "Light"), ("dark", "Dark"),
               ("auto", "Auto (time of day)"))
MODES = tuple(mode for mode, _ in MODE_LABELS)

# Auto (time-of-day) light/dark boundary -- deliberately SEPARATE from home_screen.greeting()'s bands.
_DAY_START = 7       # 07:00 -> light
_NIGHT_START = 19    # 19:00 -> dark

_ACTIVE: "Appearance | None" = None


def active_appearance():
    """The app's live Appearance (constructed in build_app), or None under tests. Lets non-QSS surfaces
    (e.g. the pyqtgraph live view) read the current theme + subscribe to changes without a hard import."""
    return _ACTIVE


def _scheme_for(mode: str, hour: int):
    if mode == "light":
        return Qt.ColorScheme.Light
    if mode == "dark":
        return Qt.ColorScheme.Dark
    if mode == "auto":
        return Qt.ColorScheme.Light if _DAY_START <= hour < _NIGHT_START else Qt.ColorScheme.Dark
    return Qt.ColorScheme.Unknown        # "system" -> follow the OS


class Appearance(QObject):
    """Owns the current appearance mode + the auto-mode timer, and applies the Fluent palette/QSS."""

    theme_changed = Signal(bool)         # emitted with the resolved `dark` after every apply

    def __init__(self, app):
        super().__init__(app)
        global _ACTIVE
        _ACTIVE = self
        self._app = app
        self._mode = "system"
        self._dark = False
        self._system_accent = False          # follow the OS accent colour (opt-in; Windows only)
        self._applying = False               # re-entrancy guard: setColorScheme can re-fire colorSchemeChanged
        self._timer = QTimer(self)
        self._timer.setInterval(60_000)      # re-check the wall clock each minute while in auto mode
        self._timer.timeout.connect(self._reapply)
        # Re-theme on a live OS light/dark flip -- but only while following the system (guarded so an
        # explicit setColorScheme below can't re-enter). setPalette/setStyleSheet don't emit this signal.
        app.styleHints().colorSchemeChanged.connect(self._on_os_scheme)

    def mode(self) -> str:
        return self._mode

    def is_dark(self) -> bool:
        return self._dark

    def set_mode(self, mode: str) -> None:
        self._mode = mode if mode in MODES else "system"
        self._reapply()
        if self._mode == "auto":
            self._timer.start()
        else:
            self._timer.stop()

    def system_accent_enabled(self) -> bool:
        return self._system_accent

    def set_system_accent(self, enabled: bool) -> None:
        """Follow (or stop following) the OS accent colour. Re-reads the OS colour on every apply, so
        toggling and theme flips stay live; there is no Qt signal for an OS accent CHANGE, so a changed
        accent lands on the next apply (mode flip / auto tick / restart)."""
        self._system_accent = bool(enabled)
        self._reapply()

    def _on_os_scheme(self, _scheme=None) -> None:
        if self._mode == "system":
            self._reapply()

    def _reapply(self) -> None:
        if self._applying:                                       # setColorScheme below can re-fire
            return                                               # colorSchemeChanged -> _on_os_scheme
        self._applying = True
        try:
            hints = self._app.styleHints()
            scheme = _scheme_for(self._mode, datetime.now().hour)
            hints.setColorScheme(scheme)                          # first: regenerates the style palette
            effective = scheme if scheme != Qt.ColorScheme.Unknown else hints.colorScheme()
            self._dark = (effective == Qt.ColorScheme.Dark)      # Unknown -> light fallback
            accent = design.system_accent() if self._system_accent else None
            self._app.setPalette(design.build_palette(self._dark, accent))  # our explicit Fluent values win
            self._app.setStyleSheet(design.build_qss(self._dark, accent))
        finally:
            self._applying = False
        self.theme_changed.emit(self._dark)
