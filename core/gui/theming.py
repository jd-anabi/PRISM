"""The window's appearance: three choices (Follow Windows, Light, Dark) and the one place that
applies them.

Each choice resolves to light or dark; then an apply does, in order:
  * ``styleHints().setColorScheme`` -- so the frames Windows draws (the title bar, the file
    choosers) follow;
  * ``design.set_dark`` -- so every device that paints itself reads the colours of the theme in
    force;
  * ``design.build_palette`` + ``design.build_qss`` -- the palette and the one application style
    sheet. A widget that paints itself (progress_pane._Sparkline) reads the palette, so the
    palette -- not only the sheet -- must be right;
  * ``fonts.apply_class_fonts`` (a new sheet puts the system's font back on lists, tables,
    headers, menus and tool tips), ``retheme()`` on each widget that has one, and an update of
    every top-level widget.

An apply whose resolved scheme is the one this Appearance already applied stops after the first
of these: it sets neither the palette nor the sheet and signals nothing. ``should_apply``
decides, and an Appearance's first apply always goes on.

``build_app`` constructs the one Appearance. A test that builds the window directly has none.

Two rendering surfaces stay outside the palette pipeline but still follow the theme via the
``theme_changed`` signal: the pyqtgraph live view re-themes itself (see live_hair_bundle.py), and
matplotlib figures follow the theme through rcParams (see mpl_theme.install, wired in app.build_app).
The one deliberate exception is the Simulate video export, which stays white (simulate_export.py).
"""
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QWidget

from . import design, fonts

# (mode key, human label), in display order -- shared by the gear popover and the Settings screen.
MODE_LABELS = (("system", "Follow Windows"), ("light", "Light"), ("dark", "Dark"))
MODES = tuple(mode for mode, _ in MODE_LABELS)

_ACTIVE: "Appearance | None" = None


def active_appearance():
    """The app's live Appearance (constructed in build_app), or None when nothing has built one (a
    test that builds the window directly). Lets non-QSS surfaces (e.g. the pyqtgraph live view) read
    the current theme + subscribe to changes without a hard import."""
    return _ACTIVE


def _scheme_for(mode: str):
    if mode == "light":
        return Qt.ColorScheme.Light
    if mode == "dark":
        return Qt.ColorScheme.Dark
    return Qt.ColorScheme.Unknown        # "system" -> follow the OS


def should_apply(applied: "tuple | None", wanted: tuple) -> bool:
    """Whether an apply goes on: always the first time (``applied`` is None), and after that only
    when what is wanted differs from what was applied. Pure, so its test needs no application."""
    return applied is None or applied != wanted


class Appearance(QObject):
    """Owns the appearance mode and applies the palette and the style sheet for it."""

    theme_changed = Signal(bool)         # emitted with the resolved `dark` after every apply that went on

    def __init__(self, app):
        super().__init__(app)
        global _ACTIVE
        _ACTIVE = self
        self._app = app
        self._mode = "system"
        self._dark = False
        # What the last apply that finished applied. None until the first apply, while an apply is
        # under way, and after one that did not finish.
        self._applied = None
        self._applying = False               # re-entrancy guard: setColorScheme can re-fire colorSchemeChanged
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

    def _on_os_scheme(self, _scheme=None) -> None:
        if self._mode == "system":
            self._reapply()

    def _reapply(self) -> None:
        if self._applying:                                       # setColorScheme below can re-fire
            return                                               # colorSchemeChanged -> _on_os_scheme
        self._applying = True
        try:
            hints = self._app.styleHints()
            scheme = _scheme_for(self._mode)
            hints.setColorScheme(scheme)                          # always: the frames Windows draws follow
            effective = scheme if scheme != Qt.ColorScheme.Unknown else hints.colorScheme()
            self._dark = (effective == Qt.ColorScheme.Dark)      # Unknown -> light fallback
            wanted = (self._dark,)
            if not should_apply(self._applied, wanted):
                return                                           # nothing changed: set nothing, say nothing
            # While the work is under way nothing is known to be applied, so an apply that fails
            # part-way is never mistaken for one that was made.
            self._applied = None
            design.set_dark(self._dark)                          # first: all that follows reads the theme in force
            self._app.setPalette(design.build_palette(self._dark))
            self._app.setStyleSheet(design.build_qss(self._dark))
            fonts.apply_class_fonts(self._app)                   # after the sheet, and on this application alone
            for top in self._app.topLevelWidgets():              # each widget that keeps colours of its own, once
                for widget in (top, *top.findChildren(QWidget)):
                    if widget.window() is not top:               # in a window of its own that has this one for
                        continue                                 # its parent: reached when that window's turn comes
                    retheme = getattr(widget, "retheme", None)
                    if callable(retheme):
                        retheme()
                top.update()
            self._applied = wanted                               # only once all of it is done
        finally:
            self._applying = False
        self.theme_changed.emit(self._dark)
