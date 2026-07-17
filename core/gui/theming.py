"""App appearance (Follow-system / Light / Dark / Auto) for the Qt chrome.

The GUI paints only palette-driven colours -- help_badge uses ``palette(mid)``/``palette(highlight)`` and
everything else inherits the style palette -- so switching the Qt colour scheme recolours the WHOLE app
with no per-widget edits. Applied once at startup (``app.build_app``) and again from the settings gear.

Kept out of the test path on purpose: only ``build_app`` constructs an ``Appearance`` (tests build
``MainWindow`` directly), so ``setColorScheme`` / the auto timer never fire during the suite.

TWO rendering surfaces are DELIBERATELY not palette-driven and keep their own (scientifically-tuned)
colours in every theme: the pyqtgraph live Simulate view (its ``inferno`` heatmap is designed for a dark
backdrop) and matplotlib figures/PNGs (white, the scientific-plot norm). This mirrors the usual
convention that data plots keep a stable look regardless of UI chrome.
"""
from datetime import datetime

from PySide6.QtCore import QObject, Qt, QTimer

# (mode key, human label), in display order -- shared by the gear popover and the Settings screen.
MODE_LABELS = (("system", "Follow system"), ("light", "Light"), ("dark", "Dark"),
               ("auto", "Auto (time of day)"))
MODES = tuple(mode for mode, _ in MODE_LABELS)

# Auto (time-of-day) light/dark boundary -- deliberately SEPARATE from home_screen.greeting()'s
# morning/afternoon/evening bands (those are greeting copy, not a light/dark threshold).
_DAY_START = 7       # 07:00 -> light
_NIGHT_START = 19    # 19:00 -> dark


def _scheme_for(mode: str, hour: int):
    if mode == "light":
        return Qt.ColorScheme.Light
    if mode == "dark":
        return Qt.ColorScheme.Dark
    if mode == "auto":
        return Qt.ColorScheme.Light if _DAY_START <= hour < _NIGHT_START else Qt.ColorScheme.Dark
    return Qt.ColorScheme.Unknown        # "system" -> reset to following the OS preference


class Appearance(QObject):
    """Owns the current appearance mode and the auto-mode re-check timer. One instance per app."""

    def __init__(self, app):
        super().__init__(app)
        self._app = app
        self._mode = "system"
        self._timer = QTimer(self)
        self._timer.setInterval(60_000)      # re-check the wall clock each minute while in auto mode
        self._timer.timeout.connect(self._reapply)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        self._mode = mode if mode in MODES else "system"
        self._reapply()
        if self._mode == "auto":
            self._timer.start()
        else:
            self._timer.stop()

    def _reapply(self) -> None:
        self._app.styleHints().setColorScheme(_scheme_for(self._mode, datetime.now().hour))
