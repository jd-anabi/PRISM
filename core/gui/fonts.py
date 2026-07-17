"""App typography: register a bundled cross-platform font (Inter) and set the base app font.

Fluent leans on its typeface. We bundle Inter (SIL OFL 1.1, redistributable) as the cross-platform
guarantee, but PREFER the OS-native Fluent face (Segoe UI) where present -- so Windows 11 gets the
authentic look and macOS/Linux get a consistent Inter. Base size is 14 px (Fluent Body); the heading
ramp is applied via QSS ``QLabel[type=...]`` properties (see design.py), not here.

Non-blocking: if the bundled font is absent/unloadable we fall back to the best available native face
(Segoe UI on Windows, system default elsewhere), so the app never fails to start over a missing asset.
"""
from pathlib import Path

from PySide6.QtGui import QFontDatabase

_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
# Preference order: native Fluent face first (Windows), then the bundled Inter, then whatever's default.
_PREFERRED = ("Segoe UI Variable Text", "Segoe UI", "Inter")


def _register_bundled() -> None:
    if not _FONT_DIR.is_dir():
        return
    for path in sorted(_FONT_DIR.glob("*.ttf")):            # static or variable Inter -> family "Inter"
        QFontDatabase.addApplicationFont(str(path))


def load_app_font(app, size_px: int = 14, prefer_inter: bool = False) -> str:
    """Register the bundled font, choose the best available family, set it as the base app font, and
    return the chosen family name. ``prefer_inter`` puts the bundled Inter FIRST (the "Inter
    everywhere" setting) for a consistent cross-platform look instead of the native Fluent face."""
    _register_bundled()
    available = set(QFontDatabase.families())
    order = ("Inter",) + _PREFERRED if prefer_inter else _PREFERRED
    family = next((fam for fam in order if fam in available), None)
    font = app.font()
    if family is not None:
        font.setFamily(family)
    font.setPixelSize(size_px)
    app.setFont(font)
    return family or font.family()
