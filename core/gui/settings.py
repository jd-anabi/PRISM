"""One QSettings store for the GUI, plus tiny helpers panels use to persist their selections.

A default QSettings() resolves its location from the application's organisation + application name, so
core.gui.app.build_app() sets both. Tests point the store at a temp file via use_ini_file().
"""
from PySide6.QtCore import QSettings

ORG = "PRISM"
APP = "PRISM"

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


def get_int(qs: QSettings, key: str, default: int) -> int:
    """QSettings hands back a STRING on Windows (the INI/registry backends are untyped), so an int
    written as 5000 reads back as "5000" and int() has to be explicit. A missing or unparseable key
    falls back to `default` rather than 0 -- 0 is a meaningful value for TRAINING_RUN_SIZE."""
    raw = qs.value(key, None)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def get_bool(qs: QSettings, key: str, default: bool) -> bool:
    """"1"/"true"/"True" is True and "0"/"false"/"False" is False; a missing key or ANY other text --
    a blank, a hand-edited "maybe" -- falls back to `default`, the rule get_int already follows.
    It used to return False for every unrecognised value, so a corrupt `reparam_rotate` restored the
    Fisher rotation OFF instead of at config.REPARAM_ROTATE."""
    v = qs.value(key, None)
    if v is None:
        return default
    text = str(v)
    if text in ("1", "true", "True"):
        return True
    if text in ("0", "false", "False"):
        return False
    return default


def get_appearance(qs: QSettings) -> str:
    """The appearance mode to open with: 'light' or 'dark' when that is what is stored, 'system'
    (Follow Windows) for a missing key and for any other stored word, a mode an older build
    offered among them. Reading writes nothing; the stored word stays until the next choice."""
    mode = get_str(qs, "appearance/mode", "system")
    return mode if mode in ("light", "dark") else "system"


def set_appearance(qs: QSettings, mode: str) -> None:
    set_str(qs, "appearance/mode", mode)


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
