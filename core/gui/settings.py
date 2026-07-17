"""One QSettings store for the GUI, plus tiny helpers panels use to persist their selections.

A default QSettings() resolves its location from the application's organisation + application name, so
core.gui.app.build_app() sets both. Tests point the store at a temp file via use_ini_file().
"""
from PySide6.QtCore import QSettings

ORG = "GFDTResearch"
APP = "GFDT-SBI"

_override_path: str | None = None


def use_ini_file(path: str | None) -> None:
    """Redirect the store to an explicit .ini file (tests), or back to the default (None)."""
    global _override_path
    _override_path = path


def settings() -> QSettings:
    if _override_path is not None:
        return QSettings(_override_path, QSettings.IniFormat)
    return QSettings(QSettings.IniFormat, QSettings.UserScope, ORG, APP)


# ── typed get/set that survive the QSettings str round-trip ──────────────────
def set_str(qs: QSettings, key: str, value: str) -> None:
    qs.setValue(key, value)


def get_str(qs: QSettings, key: str, default: str = "") -> str:
    v = qs.value(key, default)
    return default if v is None else str(v)


def set_bool(qs: QSettings, key: str, value: bool) -> None:
    qs.setValue(key, "1" if value else "0")


def get_bool(qs: QSettings, key: str, default: bool) -> bool:
    v = qs.value(key, None)
    if v is None:
        return default
    return str(v) in ("1", "true", "True")


def get_appearance(qs: QSettings) -> str:
    """The persisted appearance mode ('system' | 'light' | 'dark' | 'auto'); default 'system'."""
    return get_str(qs, "appearance/mode", "system")


def set_appearance(qs: QSettings, mode: str) -> None:
    set_str(qs, "appearance/mode", mode)


def get_system_accent(qs: QSettings) -> bool:
    """Follow the Windows accent colour (opt-in; the fixed Fluent blue is the default)."""
    return get_bool(qs, "appearance/system_accent", False)


def set_system_accent(qs: QSettings, enabled: bool) -> None:
    set_bool(qs, "appearance/system_accent", enabled)


def get_force_inter(qs: QSettings) -> bool:
    """Force the bundled Inter font on all platforms (default: prefer the native Fluent face)."""
    return get_bool(qs, "appearance/force_inter", False)


def set_force_inter(qs: QSettings, enabled: bool) -> None:
    set_bool(qs, "appearance/force_inter", enabled)


def restore_field(qs: QSettings, key: str, field) -> None:
    """Restore a FloatField / IntField / PathField / QLineEdit from its saved text, if present."""
    v = qs.value(key, None)
    if v is None:
        return
    edit = getattr(field, "edit", field)   # PathField wraps a QLineEdit; the others ARE QLineEdits
    edit.setText(str(v))


def save_field(qs: QSettings, key: str, field) -> None:
    edit = getattr(field, "edit", field)
    qs.setValue(key, edit.text())
