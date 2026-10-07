"""The look's one table (colours, type, sizes) and the few functions every reader of it shares.

Four things read the table and nothing else holds a colour or a size: the application's style sheet, its palette,
the small style over Fusion, and the devices that paint themselves. Pure and import-safe: it needs QtGui only,
for QColor and QPalette, builds no application object, and asks Windows for its text size only when called to.

Still here beside the table, until the palette and the sheet are built from it: the older LIGHT and DARK dicts
with ``tokens``, ``build_palette`` and ``build_qss``. Two palette roles stay load-bearing whatever a sheet says,
because custom painting reads them and a sheet does not feed it: ``QPalette.Mid`` (the sparkline's baseline in
widgets/progress_pane.py) and ``QPalette.Highlight`` (its trace, and the selection colour).
"""
import math
import numbers
import sys
from datetime import datetime, timedelta
from string import Template

from PySide6.QtGui import QColor, QPalette

# ── the look's table: colours, type, sizes ──
#   name              light      dark
COLOURS: dict[str, tuple[str, str]] = {
    "paper":         ("#FFFFFF", "#1C1C1C"),   # the page: every working surface
    "margin":        ("#F2F2F2", "#141414"),   # the rail's grey; the scroll bar's track
    "field":         ("#FFFFFF", "#141414"),   # inside a field
    "raised":        ("#FFFFFF", "#262626"),   # inside a question; a tool tip
    "ink":           ("#1A1A1A", "#EDEDED"),   # text; the "done" marker
    "ink_2":         ("#4D4D4D", "#B8B8B8"),   # secondary text, tick labels, the information marker and the padlock
    "ink_3":         ("#636363", "#9C9C9C"),   # hints, "to come", the idle dash, the scroll handle
    "rule":          ("#1A1A1A", "#C8C8C8"),   # the three rules of a table; an axis
    "rule_light":    ("#D4D4D4", "#3A3A3A"),   # a quiet separator, never the only sign of anything
    "field_line":    ("#6B6B6B", "#8A8A8A"),   # the outline of a control
    "accent":        ("#0072B2", "#56B4E9"),   # what is now, what is selected, the main action
    "on_accent":     ("#FFFFFF", "#0D0D0D"),   # text and marks on the accent
    "accent_tint":   ("#E2EEF6", "#1A3447"),   # the ground of the step in hand and of a selected row
    "accent_hover":  ("#00639B", "#7CC5EE"),   # the main button under the pointer
    "accent_press":  ("#005382", "#3FA3DC"),   # the main button pressed
    "ok":            ("#007A5A", "#2FBF94"),   # the "fine" marker
    "warn":          ("#E69F00", "#E69F00"),   # the fill of the attention triangle
    "warn_edge":     ("#1A1A1A", "#E69F00"),   # the triangle's edge
    "warn_mark":     ("#1A1A1A", "#1A1A1A"),   # the stroke inside the triangle, both themes
    "err":           ("#C24E00", "#F0813A"),   # the error marker; a field in error; an action that loses work
    "on_err":        ("#FFFFFF", "#0D0D0D"),   # the cross inside the error marker
    "focus":         ("#1A1A1A", "#FFFFFF"),   # the focus ring
    "hover_fill":    ("#F2F2F2", "#262626"),   # an ordinary button under the pointer
    "press_fill":    ("#DEDEDE", "#343434"),   # an ordinary button pressed
    "disabled_fill": ("#F2F2F2", "#232323"),   # inside a disabled control
    "disabled_line": ("#C4C4C4", "#4A4A4A"),   # the outline of a disabled control
    "disabled_ink":  ("#636363", "#9C9C9C"),   # disabled text and marks
    "hatch":         ("#4D4D4D", "#B8B8B8"),   # the hatched range on the time axis
    "elapsed":       ("#1A1A1A", "#C8C8C8"),   # the elapsed bar on the time axis
    "series_2":      ("#0072B2", "#56B4E9"),   # the second colour of SERIES
    "series_6":      ("#56B4E9", "#0072B2"),   # the sixth colour of SERIES: a figure's line
    "series_7":      ("#CC79A7", "#CC79A7"),   # the seventh colour of SERIES: a figure's line
}
#   name      (pixel size, line height, weight)      five sizes and nothing between
TYPE = {
    "small":  (12, 16, 400),
    "body":   (14, 20, 400),
    "strong": (14, 20, 600),
    "head":   (16, 22, 600),
    "brand":  (18, 24, 600),
    "title":  (24, 30, 600),
    "book":   (16, 22, 400),      # Cambria: the line under a heading, a caption, a message, a dialog's question
}
LETTERING = "Inter"
BOOK_FACES = ("Cambria", "Georgia")
FIXED_FACES = ("Cascadia Mono", "Consolas", "Courier New")
BRAND_SPACING = 104               # per cent: a style sheet has no letter-spacing
SIZE = {
    # the window, and when a page changes its arrangement
    "window": (1300, 820), "window_min": (900, 600),
    "small_below": 700,           # a window lower than this gets the small layout
    "two_columns_from": 940,      # a page narrower than this sets its two columns one above the other
    "rail": 220, "page_pad": (22, 20, 20, 14), "page_pad_small": (16, 16, 14, 14),     # left, top, right, bottom
    "scroll_bar": 8, "scroll_gap": 2,
    # controls
    "field_h": 32, "button_h": 34, "radius": 3, "indicator": 18, "number_w": 120,
    "field_pad": 8, "button_pad": 16, "primary_pad": 20, "combo_pad": (10, 8), "icon_gap": 4, "line_h": 20,
    "check_gap": 10, "radio_gap": 8, "tip_pad": (4, 8), "scroll_handle": (28, 2, 2),     # least length, inset, corner
    # markers, rules, the ring
    "marker": 16, "rule": 2, "hair": 1, "bar": 4, "focus_ring": 2, "focus_gap": 2, "row_ring_inset": 2,
    "hatch_step": 5, "hatch_key": (24, 10),
    # the navigation column
    "nav_h": 36, "nav_h_small": 32, "nav_pad": 20, "step_h": 46, "step_h_one": 32,
    "brand": (20, 19, 20, 15), "brand_small": (20, 13, 20, 11),             # round the 24 px line of the name
    "run_line": (20, 12, 14, 14), "run_line_gap": 9, "run_marker_drop": 2,
    # forms and tables
    "row_gap": 12, "col_gap": 32, "label_w": (200, 165),
    "table_row": 30, "table_pad": 5, "table_col_gap": 16, "table_lead": 10, "table_fit_most": 180, "state_gap": 6,
    "table_stretch_least": 40, "table_section_least": 20,
}
GAP = {                           # the spaces between the blocks of a page, top to bottom
    "head_lead": 2, "head_buttons": 12, "head_to_body": 22, "head_to_body_small": 14,
    "section_rule": 6, "form_to_fold": 20, "fold": (10, 4, 11), "fold_indent": 22, "fold_icon": (18, 14),
    "setup_columns": (470, 536), "columns_stacked": 20, "field_message": 6, "marker_text": 8, "marker_drop": 2,
    "message": (10, 4, 0, 4), "message_marker": 10, "message_marker_drop": 3,
    "gate_rows": 5, "gate_action": 9, "gate_above": 12, "action": 12, "action_why": 4,
    "status": (12, 10), "stages": (36, 6), "stage_parts": 7, "now": (8, 4), "axis_key": 8, "status_to_body": 12,
    "run_columns": (668, 354, 16), "health_columns": (118, 56), "figure_inset": 28, "caption_inset": 8, "figure_caption": 6, "trace_caption": 4,
    "block": 14, "side_block": 10, "head_table": 6, "switch": 16,
    "rail_rest": 5, "rail_group": 12,
}
STEP = {                          # one step of the chain, as painted
    "marker": (22, 7), "marker_one": (22, 8), "line_x": 30, "line_from": 26, "line_in": 4,
    "text_x": 50, "name_y": 5, "name_y_one": 6, "words_y": 25, "number_gap": 7, "pad_right": 10,
}
AXIS = {                          # the time axis, as painted
    "h": 42, "track_y": 4, "track_h": 14, "mark_w": 4, "mark_halo": 1, "mark_h": 22,
    "tick_y": 18, "tick_h": 5, "label_y": 25, "label_h": 16, "label_gap": 8,
}
DIALOG = {"w": 448, "pad": (20, 16, 20, 18), "gap": 16, "marker": 24, "marker_gap": 14, "marker_drop": 1}
FIGURE = {                        # figures: margins round the axes in pixels; the rest follows the widget's size
    "loss":  {"h": 282, "min_w": 420, "left": 60, "top": 12, "right": 80, "bottom": 42, "x_name_drop": 38, "tick_pad": 5},
    "trace": {"h": 100, "min_w": 240, "left": 40, "top": 18, "right": 12, "bottom": 34, "x_name_drop": 31, "tick_pad": 4,
              "y_name_above": True},
    "tick": 5, "axis_w": 1, "line_w": 2, "y_name_gap": 10, "y_name_sink": 2, "y_name_lift": 8, "line_label_gap": 8,
    "best_mark": 9, "note_x": 12,
}
# the colours of a figure's lines, in this order: seven, chosen so that a colour-blind reader can tell the lines
# apart. The set carries no line style.
SERIES = ("ink", "series_2", "err", "ok", "warn", "series_6", "series_7")
# The tables hold the numbers of Windows' ordinary text size: a reader passes every length it takes from them through
# scaled(), a TYPE row's size and line height among them, and takes the keys named here as they are. These are line
# weights and radii, not lengths that grow with the text: the rules, the accent bar, the focus ring and its gaps, the
# corner radius, the hatch step, the scroll bar with its gap and its handle, the width and the halo of the "now" mark,
# and a figure's axis, line and tick. (A TYPE row's weight, BRAND_SPACING and the truth value in FIGURE are not
# lengths at all.)
UNSCALED = frozenset({
    "SIZE.rule", "SIZE.hair", "SIZE.bar", "SIZE.focus_ring", "SIZE.focus_gap", "SIZE.row_ring_inset", "SIZE.radius",
    "SIZE.hatch_step", "SIZE.scroll_bar", "SIZE.scroll_gap", "SIZE.scroll_handle",
    "AXIS.mark_w", "AXIS.mark_halo", "FIGURE.axis_w", "FIGURE.line_w", "FIGURE.tick"})
# English letters on every computer: a date is written from these, never from Windows' regional setting
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

_dark = False                     # the theme in force: light until a caller says otherwise


def set_dark(dark: bool) -> None:
    """Say which theme is in force; ``colour`` and ``hex_colour`` read it from then on."""
    global _dark
    _dark = bool(dark)


def is_dark() -> bool:
    """Whether the dark theme is in force (light when none was set)."""
    return _dark


def hex_colour(name: str, dark: bool | None = None) -> str:
    """A colour of the table as '#RRGGBB', in the theme asked for; None asks for the theme in force.
    A name the table does not hold raises KeyError."""
    return COLOURS[name][1 if (_dark if dark is None else dark) else 0]


def colour(name: str) -> QColor:
    """A colour of the table in the theme in force. A new QColor at every call, so a caller may change its own."""
    return QColor(hex_colour(name))


_LARGEST_TEXT = 2.25              # where Windows' text-size setting ends
_text_scale = 1.0                 # the text factor in force: the ordinary size until a caller installs another


def text_scale() -> float:
    """The text factor in force: 1.0 at Windows' ordinary text size, 2.25 at most."""
    return _text_scale


def set_text_scale(factor: float) -> None:
    """Install the text factor ``scaled`` reads from then on, held between 1.0 and 2.25. What float() cannot
    take, and what is not finite, counts as the ordinary size."""
    global _text_scale
    try:
        factor = float(factor)
    except (TypeError, ValueError, OverflowError):
        factor = 1.0
    _text_scale = min(_LARGEST_TEXT, max(1.0, factor)) if math.isfinite(factor) else 1.0


def scaled(n: int) -> int:
    """A length of the table, in pixels, at the text size in force. A half rounds up at every factor."""
    return math.floor(n * text_scale() + 0.5)


def window_sizes(screen: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """The size the window opens at and its smallest size, each (width, height), at the text size in force.
    Each grows with the text and is then held to ``screen``, the room the caller has (width, height; nothing is
    subtracted here), and is never below 1."""
    def fitted(size):
        return tuple(max(1, min(scaled(length), room)) for length, room in zip(size, screen))

    return fitted(SIZE["window"]), fitted(SIZE["window_min"])


def system_text_scale() -> float:
    """The factor of Windows' own text-size setting, 1.0 to 2.25; 1.0 off Windows and whenever it cannot be read.

    Windows' documented setting is asked first, and the stored value only when that has not answered: a reader
    that raises, or answers None or anything that is not a finite number, has not answered. This only reads, and
    never raises; when it is called is the caller's affair."""
    if not _on_windows():
        return 1.0
    for read in (_windows_text_scale, _registry_text_scale):
        try:
            answer = float(read())
        except Exception:                                   # noqa: BLE001 -- a reader that fails has not answered
            continue
        if math.isfinite(answer):
            return min(_LARGEST_TEXT, max(1.0, answer))
    return 1.0


def _on_windows() -> bool:
    return sys.platform == "win32"


def _windows_text_scale():
    """What Windows itself reports for its text-size setting (1.0 to 2.25), read from its documented settings
    object through ctypes; None when that object cannot be reached. It reads and writes nothing."""
    import ctypes
    combase = ctypes.WinDLL("combase")

    class _Guid(ctypes.Structure):
        _fields_ = [("a", ctypes.c_ulong), ("b", ctypes.c_ushort), ("c", ctypes.c_ushort), ("d", ctypes.c_ubyte * 8)]

    wanted = _Guid(0xBAD82401, 0x2721, 0x44F9, (ctypes.c_ubyte * 8)(0xBB, 0x91, 0x2B, 0xB2, 0x28, 0xBE, 0x44, 0x2F))
    combase.RoInitialize.argtypes = [ctypes.c_int]
    combase.RoInitialize.restype = ctypes.c_long
    # Refused when the toolkit has already set this thread up: the read works all the same, and then there is
    # nothing to undo. When this call began the set-up, the finally below undoes it, so that the toolkit can still
    # set the thread up in its own way afterwards.
    began = combase.RoInitialize(1) >= 0
    try:
        name = "Windows.UI.ViewManagement.UISettings"
        text = ctypes.c_void_p()
        combase.WindowsCreateString.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        combase.WindowsCreateString.restype = ctypes.c_long
        if combase.WindowsCreateString(name, len(name), ctypes.byref(text)) < 0:
            return None
        first = ctypes.c_void_p()
        combase.RoActivateInstance.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        combase.RoActivateInstance.restype = ctypes.c_long
        status = combase.RoActivateInstance(text, ctypes.byref(first))
        combase.WindowsDeleteString.argtypes = [ctypes.c_void_p]
        combase.WindowsDeleteString(text)
        if status < 0:
            return None

        def method(pointer, slot, restype, *argtypes):
            table = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value
            entry = ctypes.cast(table + slot * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
            return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(entry)

        second = ctypes.c_void_p()           # the interface that carries the text size
        status = method(first, 0, ctypes.c_long, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))(
            first, ctypes.byref(wanted), ctypes.byref(second))
        method(first, 2, ctypes.c_ulong)(first)
        if status < 0:
            return None
        value = ctypes.c_double(-1.0)
        status = method(second, 6, ctypes.c_long, ctypes.POINTER(ctypes.c_double))(second, ctypes.byref(value))
        method(second, 2, ctypes.c_ulong)(second)
        return float(value.value) if status >= 0 else None
    finally:
        if began:
            combase.RoUninitialize()


def _registry_text_scale():
    """The value Windows' Settings page stores for its text size, as a factor. The ordinary size stores no value
    at all, so an absent value is the answer 1.0; that is why this place is asked second: asked first, it would
    always answer."""
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Accessibility") as opened:
            stored, _kind = winreg.QueryValueEx(opened, "TextScaleFactor")
    except FileNotFoundError:
        return 1.0
    return int(stored) / 100.0


def number_text(value) -> str:
    """A quantity as the window writes it on every computer, whatever Windows' regional setting: a comma between
    thousands, a full stop for the decimal mark, the keyboard's hyphen for a minus, so the text reads back in a
    field. A whole number is written whole ("20,480,000"); any other real number in the shortest form that reads
    back the same ("2,048.0", "0.05").

    The unit is the caller's: ``number_text(40.1) + " %"``. An identifier (a seed, a record's id) is not a
    quantity and is never handed to this. A truth value, a text, None or anything else raises TypeError."""
    if isinstance(value, bool):                             # asked first: Python counts True among the whole numbers
        raise TypeError("a truth value is not a quantity")
    if isinstance(value, numbers.Integral):
        return format(int(value), ",")
    if isinstance(value, numbers.Real):
        return format(float(value), ",")
    raise TypeError(f"not a quantity: {type(value).__name__}")


def date_text(when: datetime, now: datetime | None = None) -> str:
    """A date and time as the window writes it on every computer: "3 Oct 2026, 14:02", the hours on the 24-hour
    clock and the seconds dropped. Handed the present moment, a date less than seven days from it, ahead of it or
    behind it, carries its weekday and no year: "Sun 4 Oct, 12:10".

    The letters are English whatever Windows' regional setting. The fields of ``when`` are written as handed in:
    nothing is converted between zones."""
    clock = f"{when.hour:02d}:{when.minute:02d}"
    if now is not None and abs(when - now) < timedelta(days=7):
        return f"{WEEKDAYS[when.weekday()]} {when.day} {MONTHS[when.month - 1]}, {clock}"
    return f"{when.day} {MONTHS[when.month - 1]} {when.year}, {clock}"


# ── the older tokens and sheet: unchanged below, until the palette and the sheet are built from the table ──
RADIUS_SM = 4
RADIUS_MD = 8
CTL_H = 32              # standard Fluent control height
CTL_H_SM = 28
SPACE = (4, 8, 12, 16, 20, 24)

# ── Layout sizing ─────────────────────────────────────────────────────────────────────────────────
# The controls column had a hard 460px MAXIMUM, which no amount of window-widening could escape, so
# any form wider than that got a permanent horizontal scrollbar. The cap is gone; these set the floor
# and the starting split instead, and the user's own drag is remembered from there.
CONTROLS_MIN_W = 360      # narrower than this and the two-field composite rows stop being readable
DEFAULT_SPLIT = (420, 880)  # initial (controls, results) split; overridden by the saved layout
FIELD_MIN_W = 96          # a numeric field must show ~8 characters; "0.033333" used to scroll inside
                          # a 3-6 character box and could not be read back without clicking into it
PATH_FIELD_MIN_W = 220    # path fields hold absolute paths, so they need materially more
DEFAULT_RESULTS_SPLIT = (520, 200)  # initial (figures, log) split in the results column
HELP_VIEWER_MIN_H = 260   # Settings' help viewer: tall enough to be usable, it scrolls itself
LABEL_MAX_W = 190         # ceiling on a form's label column: one long option name (e.g. "F0 (ND
                          # forcing amplitude)") used to narrow every field in the same form

# Type ramp: (pixel size, weight). Weight 600 = SemiBold (Fluent's "Strong"/heading weight).
_OLD_TYPE = {
    "caption":     (12, 400),
    "body":        (14, 400),
    "body_strong": (14, 600),
    "heading":     (16, 600),
    "subtitle":    (20, 600),
    "title":       (28, 600),
}

# Neutral + accent palettes. Tuned to Fluent (Win11) light/dark. Retune here in ONE place.
LIGHT = {
    "window":       "#F3F3F3",
    "base":         "#FFFFFF",   # cards, inputs, popups
    "alt_base":     "#F7F7F7",
    "mid":          "#D1D1D1",   # borders / dividers  (-> QPalette.Mid)
    "mid_strong":   "#C4C4C4",
    "text":         "#1B1B1B",
    "text_2nd":     "#616161",   # secondary / placeholder / captions
    "text_disabled":"#A0A0A0",
    "button":       "#FBFBFB",
    "button_hover": "#F0F0F0",
    "button_press": "#E5E5E5",
    "accent":       "#0F6CBD",   # -> QPalette.Highlight
    "accent_hover": "#115EA3",
    "accent_press": "#0C3B5E",
    "on_accent":    "#FFFFFF",
    "tooltip_bg":   "#FFFFFF",
    "scrollbar":    "#B8B8B8",
    "scrollbar_hover":"#9A9A9A",
}
DARK = {
    "window":       "#202020",
    "base":         "#2B2B2B",
    "alt_base":     "#333333",
    "mid":          "#3D3D3D",
    "mid_strong":   "#4A4A4A",
    "text":         "#F5F5F5",
    "text_2nd":     "#C7C7C7",
    "text_disabled":"#6E6E6E",
    "button":       "#313131",
    "button_hover": "#3A3A3A",
    "button_press": "#454545",
    "accent":       "#4CA0E0",
    "accent_hover": "#62B0EE",
    "accent_press": "#3A88CC",
    "on_accent":    "#1B1B1B",
    "tooltip_bg":   "#2B2B2B",
    "scrollbar":    "#4A4A4A",
    "scrollbar_hover":"#5E5E5E",
}


def _c(hex_str: str) -> QColor:
    return QColor(hex_str)


def system_accent() -> "str | None":
    """The Windows OS accent colour as '#RRGGBB', or None (non-Windows, or any read failure).

    Primary source is the Explorer accent (an ABGR DWORD -- the colour the user picked in Windows
    Settings); fallback is the DWM ColorizationColor (ARGB). Both reads are wrapped: this must never
    raise, it degrades to the fixed Fluent-blue token."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Accent") as key:
            value, _ = winreg.QueryValueEx(key, "AccentColorMenu")
        r, g, b = value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF          # ABGR
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:                                       # noqa: BLE001
        pass
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\DWM") as key:
            value, _ = winreg.QueryValueEx(key, "ColorizationColor")
        r, g, b = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF          # ARGB
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:                                       # noqa: BLE001
        return None


def tokens(dark: bool, accent: "str | None" = None) -> dict:
    """The token dict for a scheme, optionally re-based on an accent override ('#RRGGBB').

    The override recomputes accent/accent_hover/accent_press (hover/press follow the fixed tokens'
    direction: light mode darkens on interaction, dark mode lightens then darkens) and picks on_accent
    black/white by luminance so CTA text stays readable on any OS accent."""
    base = DARK if dark else LIGHT
    if not accent:
        return base
    c = _c(accent)
    if not c.isValid():
        return base
    t = dict(base)
    hover = c.lighter(112) if dark else c.darker(112)
    press = c.darker(120) if dark else c.darker(145)
    luminance = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    t["accent"] = c.name().upper()
    t["accent_hover"] = hover.name().upper()
    t["accent_press"] = press.name().upper()
    t["on_accent"] = "#1B1B1B" if luminance > 160 else "#FFFFFF"
    return t


def build_palette(dark: bool, accent: "str | None" = None) -> QPalette:
    """A fully-populated QPalette from the tokens. Sets every role that matters (including the
    Disabled group -- the whole controls column is disabled per run), because partially-QSS'd complex
    controls fall back to the palette for their un-styled sub-parts."""
    t = tokens(dark, accent)
    p = QPalette()

    def setall(role, color, groups=(QPalette.Active, QPalette.Inactive)):
        for g in groups:
            p.setColor(g, role, _c(color))

    setall(QPalette.Window, t["window"])
    setall(QPalette.WindowText, t["text"])
    setall(QPalette.Base, t["base"])
    setall(QPalette.AlternateBase, t["alt_base"])
    setall(QPalette.Text, t["text"])
    setall(QPalette.Button, t["button"])
    setall(QPalette.ButtonText, t["text"])
    setall(QPalette.BrightText, "#FFFFFF")
    setall(QPalette.ToolTipBase, t["tooltip_bg"])
    setall(QPalette.ToolTipText, t["text"])
    setall(QPalette.PlaceholderText, t["text_2nd"])
    setall(QPalette.Highlight, t["accent"])                 # LOAD-BEARING (custom paint + selection)
    setall(QPalette.HighlightedText, t["on_accent"])
    setall(QPalette.Link, t["accent"])
    setall(QPalette.LinkVisited, t["accent_press"])
    # Bevel/edge roles kept coherent so any Fusion-drawn sub-part we don't fully QSS matches.
    setall(QPalette.Mid, t["mid"])                          # LOAD-BEARING (_Sparkline baseline, badge)
    setall(QPalette.Midlight, t["alt_base"])
    setall(QPalette.Light, t["base"])
    setall(QPalette.Dark, t["mid_strong"])
    setall(QPalette.Shadow, "#000000" if not dark else "#000000")

    # Disabled group: BasePanel disables the entire controls column during a run.
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, _c(t["text_disabled"]))
    p.setColor(QPalette.Disabled, QPalette.Base, _c(t["alt_base"]))
    p.setColor(QPalette.Disabled, QPalette.Button, _c(t["button"]))
    p.setColor(QPalette.Disabled, QPalette.Highlight, _c(t["mid"]))
    p.setColor(QPalette.Disabled, QPalette.HighlightedText, _c(t["text_disabled"]))
    return p


# ── stylesheet ──────────────────────────────────────────────────────────────
# Named-class selectors ONLY -- never `*` or bare `QWidget` (would paint backgrounds behind the
# custom-painted _Sparkline, the pyqtgraph view, matplotlib QLabels, and the anim.py snapshot overlays).
# $-substitution (string.Template) so CSS braces stay literal.
_QSS = Template("""
/* ---- type ramp (replaces the old inline font-size stylesheets) ---- */
QLabel[type="title"]       { font-size: ${fs_title}px; font-weight: 600; }
QLabel[type="subtitle"]    { font-size: ${fs_subtitle}px; font-weight: 600; }
QLabel[type="heading"]     { font-size: ${fs_heading}px; font-weight: 600; }
QLabel[type="caption"]     { font-size: ${fs_caption}px; color: $text_2nd; }

/* ---- push buttons: neutral + accent variant ---- */
QPushButton {
    min-height: ${ctl_h}px; padding: 4px 14px;
    border: 1px solid $mid; border-radius: ${radius_sm}px;
    background: $button; color: $text;
}
QPushButton:hover   { background: $button_hover; }
QPushButton:pressed { background: $button_press; }
QPushButton:focus   { border: 1px solid $accent; }
QPushButton:disabled{ color: $text_disabled; border-color: $mid; background: $button; }
QPushButton[accent="true"] {
    background: $accent; color: $on_accent; border: 1px solid $accent; font-weight: 600;
}
QPushButton[accent="true"]:hover   { background: $accent_hover; border-color: $accent_hover; }
QPushButton[accent="true"]:pressed { background: $accent_press; border-color: $accent_press; }
QPushButton[accent="true"]:focus   { border: 1px solid $text; }
QPushButton[accent="true"]:disabled{ background: $mid; border-color: $mid; color: $text_disabled; }
/* compact square icon buttons (e.g. the picker refresh) -- override the wide default padding so a
   fixed-width glyph button isn't squeezed to a sliver, and match the adjacent input height. */
QPushButton#iconButton { padding: 0px; min-height: ${ctl_h_sm}px; font-size: ${fs_icon_btn}px; }

/* ---- text inputs (covers FloatField/IntField/PathField.edit subclasses of QLineEdit) ---- */
QLineEdit, QPlainTextEdit {
    min-height: ${ctl_h_sm}px; padding: 2px 8px;
    border: 1px solid $mid; border-radius: ${radius_sm}px;
    background: $base; color: $text; selection-background-color: $accent;
    selection-color: $on_accent;
}
QLineEdit:focus, QPlainTextEdit:focus { border: 1px solid $accent; }
QLineEdit:disabled { color: $text_disabled; background: $alt_base; }

/* ---- combo box + its (separate top-level) popup. Do NOT null ::down-arrow (no arrow asset). ---- */
QComboBox {
    min-height: ${ctl_h_sm}px; padding: 2px 8px;
    border: 1px solid $mid; border-radius: ${radius_sm}px; background: $base; color: $text;
}
QComboBox:hover { border-color: $mid_strong; }
QComboBox:focus { border: 1px solid $accent; }
QComboBox::drop-down { width: 22px; border: none; background: transparent; }
QComboBox QAbstractItemView {
    border: 1px solid $mid; border-radius: ${radius_sm}px;
    background: $base; color: $text;
    selection-background-color: $accent; selection-color: $on_accent; outline: 0;
}

/* ---- item views: the artifact browser's table, a QTreeWidget used flat. The app's only item view
   besides the combo popup above, so these are all of it. Selection colour comes from $accent, which
   is also QPalette.Highlight, so a custom-painted row would match. ---- */
QTreeView {
    border: 1px solid $mid; border-radius: ${radius_sm}px;
    background: $base; alternate-background-color: $alt_base; color: $text;
    selection-background-color: $accent; selection-color: $on_accent; outline: 0;
}
QTreeView::item { padding: 4px 6px; border: none; }
QTreeView::item:hover { background: $button_hover; }
QTreeView::item:selected { background: $accent; color: $on_accent; }
QHeaderView { background: transparent; border: none; }
QHeaderView::section {
    background: $alt_base; color: $text_2nd;
    padding: 4px 6px; border: none;
    border-right: 1px solid $mid; border-bottom: 1px solid $mid; font-weight: 600;
}
QHeaderView::section:hover { color: $text; }
QHeaderView::section:last, QHeaderView::section:only-one { border-right: none; }

/* ---- group box as a Fluent card ---- */
QGroupBox {
    background: $base; border: 1px solid $mid; border-radius: ${radius_md}px;
    margin-top: 14px; padding: 10px 12px 12px 12px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 4px; background: $base; color: $text_2nd;
}

/* ---- tabs as a Fluent pivot (accent underline on the selected tab) ---- */
QTabWidget::pane { border: none; }
QTabBar { qproperty-drawBase: 0; }
QTabBar::tab {
    background: transparent; color: $text_2nd; border: none;
    padding: 6px 14px; margin-right: 4px; border-bottom: 2px solid transparent;
}
QTabBar::tab:hover     { color: $text; }
QTabBar::tab:selected  { color: $text; border-bottom: 2px solid $accent; }
QTabBar::tab:disabled  { color: $text_disabled; }

/* ---- progress bars ---- */
QProgressBar {
    border: none; border-radius: 3px; background: $alt_base; text-align: center; color: $text;
}
QProgressBar::chunk { border-radius: 3px; background: $accent; }

/* ---- tool buttons (nav glyphs + help badge) ---- */
QToolButton { border: none; background: transparent; border-radius: ${radius_sm}px; color: $text; }
QToolButton:hover   { background: $button_hover; }
QToolButton:pressed { background: $button_press; }
QToolButton#navBack, QToolButton#navSettings { font-size: ${fs_nav_btn}px; padding: 2px 6px; }
QToolButton#navRunning {
    font-size: ${fs_caption}px; padding: 2px 8px; color: $text_2nd;
    border: 1px solid $mid; border-radius: ${radius_sm}px;
}
QToolButton#navRunning:hover { color: $text; border-color: $accent; }
QToolButton#helpBadge {
    border: 1px solid $mid; border-radius: 8px; color: $text_2nd;
    font-size: ${fs_badge}px; padding: 0px; background: transparent;
}
QToolButton#helpBadge:hover { border-color: $accent; color: $accent; }

/* ---- scroll bars: thin, no arrow buttons ---- */
QScrollBar:vertical   { background: transparent; width: 12px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle:vertical   { background: $scrollbar; min-height: 28px; border-radius: 6px; margin: 2px; }
QScrollBar::handle:horizontal { background: $scrollbar; min-width: 28px; border-radius: 6px; margin: 2px; }
QScrollBar::handle:hover { background: $scrollbar_hover; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; background: none; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* ---- menus (gear popover) ---- */
QMenu { background: $base; border: 1px solid $mid; border-radius: ${radius_md}px; padding: 4px; }
QMenu::item { padding: 6px 24px 6px 12px; border-radius: ${radius_sm}px; color: $text; }
QMenu::item:selected { background: $accent; color: $on_accent; }
QMenu::separator { height: 1px; background: $mid; margin: 4px 8px; }

/* ---- tooltips (help badge UX) ---- */
QToolTip {
    background: $tooltip_bg; color: $text; border: 1px solid $mid;
    border-radius: ${radius_sm}px; padding: 4px 8px;
}
""")


def _qss_vars(dark: bool, accent: "str | None" = None) -> dict:
    t = tokens(dark, accent)
    return {
        **t,
        "radius_sm": RADIUS_SM,
        "radius_md": RADIUS_MD,
        "ctl_h": scaled(CTL_H),
        "ctl_h_sm": scaled(CTL_H_SM),
        # Type ramp, so the QSS sizes track the OS text setting instead of being frozen literals.
        "fs_title": scaled(_OLD_TYPE["title"][0]),
        "fs_subtitle": scaled(_OLD_TYPE["subtitle"][0]),
        "fs_heading": scaled(_OLD_TYPE["heading"][0]),
        "fs_caption": scaled(_OLD_TYPE["caption"][0]),
        "fs_icon_btn": scaled(18),
        "fs_nav_btn": scaled(20),
        "fs_badge": scaled(13),
    }


def build_qss(dark: bool, accent: "str | None" = None) -> str:
    return _QSS.substitute(_qss_vars(dark, accent))
