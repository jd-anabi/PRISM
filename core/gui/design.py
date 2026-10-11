"""The look's one table (colours, type, sizes) and the few functions every reader of it shares.

Four things read the table and nothing else holds a colour or a size: the application's style sheet, its palette,
the small style over Fusion, and the devices that paint themselves. Pure and import-safe: it needs QtGui only,
for QColor and QPalette, builds no application object, and asks Windows for its text size only when called to.

``tokens``, ``build_palette`` and ``build_qss`` give one theme's names, its palette and its style sheet, all
from the table. Two palette roles stay load-bearing whatever a sheet says, because custom painting reads them
and a sheet does not feed it: ``QPalette.Mid`` (the sparkline's baseline in widgets/progress_pane.py) and
``QPalette.Highlight`` (its trace, and the selection colour). The sheet's last rules serve the widgets of the
old layout: ``INTERIM_SELECTORS`` names them, and each leaves with its widget.
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


def tokens(dark: bool) -> dict[str, str]:
    """The table's 32 names for one theme, each as '#RRGGBB'. A new dict at every call, so a caller may change
    its own."""
    return {name: pair[1 if dark else 0] for name, pair in COLOURS.items()}


def build_palette(dark: bool) -> QPalette:
    """The palette of one theme: every role in all three groups, from the table. A sheet does some things badly
    or not at all (link colours, the selection, a disabled control), and a part it does not style, or one that
    paints itself, falls back to the palette. The hint text of a field the sheet styles is not decided here: the
    toolkit derives it from the field's text colour unless the sheet names it. The Disabled group is set in full
    because a whole form is disabled for the length of a run."""
    t = tokens(dark)
    roles = {
        QPalette.Window: "paper", QPalette.WindowText: "ink", QPalette.Base: "field", QPalette.AlternateBase: "margin",
        QPalette.Text: "ink", QPalette.Button: "paper", QPalette.ButtonText: "ink", QPalette.BrightText: "on_accent",
        QPalette.ToolTipBase: "raised", QPalette.ToolTipText: "ink", QPalette.PlaceholderText: "ink_3",
        QPalette.Highlight: "accent",                       # load-bearing: the sparkline's trace, and the selection
        QPalette.HighlightedText: "on_accent", QPalette.Link: "accent", QPalette.LinkVisited: "accent",
        QPalette.Mid: "rule_light",                         # load-bearing: the sparkline's baseline
        QPalette.Midlight: "margin", QPalette.Light: "paper", QPalette.Dark: "field_line", QPalette.Shadow: "ink",
        QPalette.Accent: "accent",
    }
    disabled = {                                            # every other role is the same in a disabled control
        QPalette.WindowText: "disabled_ink", QPalette.Text: "disabled_ink", QPalette.ButtonText: "disabled_ink",
        QPalette.PlaceholderText: "disabled_ink", QPalette.Base: "disabled_fill", QPalette.Button: "disabled_fill",
        QPalette.Highlight: "disabled_line",
    }
    palette = QPalette()
    for role, name in roles.items():
        for group in (QPalette.Active, QPalette.Inactive):
            palette.setColor(group, role, QColor(t[name]))
        palette.setColor(QPalette.Disabled, role, QColor(t[disabled.get(role, name)]))
    return palette


# ── the one style sheet ──
# Selectors name widget classes only, never `*` or a bare `QWidget`: either would paint a ground behind the
# sparkline, the live view, the figure labels and the snapshot overlays. The names are filled in by
# string.Template, so the braces of a rule stay as written. Every size carries the prefix `px_`, because the
# filler lets a size and a colour of one name silently replace each other; the borders are written as numbers,
# since a line's weight does not grow with the text. No comment stands inside a template: it would reach the
# built text.
#
# The rules that stay. A sheet gives colours, borders and padding: a face and its features are set in code,
# and the accent comes from a class, never from the default button, which follows the keyboard focus from
# button to button. No rule marks keyboard focus: the focus ring is its one sign, so a tick box and a radio
# button give up their outline, the style's own mark of it. No rule names their indicator: one would take the
# whole box away from the style. Nothing is said of a drop-down's arrow either. A rule that gives a field its
# text colour names the colour of its hint as well: left unnamed, the toolkit makes the hint of the text colour
# at half strength, in every state of the field.
_QSS = Template("""
QToolTip { background: $raised; color: $ink; border: 1px solid $field_line; padding: ${px_tip_v}px ${px_tip_h}px; }
QDialog[cls="ask"] { background: $raised; }

QLabel[cls="title"]  { font-size: ${px_t_title}px; font-weight: 600; }
QLabel[cls="head"]   { font-size: ${px_t_head}px; font-weight: 600; }
QLabel[cls="now"]    { font-size: ${px_t_head}px; }
QLabel[cls="strong"] { font-weight: 600; }
QLabel[cls="small"]  { font-size: ${px_t_small}px; color: $ink_2; }
QLabel[cls="quiet"]  { color: $ink_2; }
QLabel[cls="book2"]  { color: $ink_2; }

QPushButton { min-height: ${px_btn_in}px; padding: 0px ${px_btn_pad}px; border: 1px solid $field_line;
              border-radius: ${px_radius}px; background: $paper; color: $ink; outline: none; }
QPushButton:hover    { background: $hover_fill; }
QPushButton:pressed  { background: $press_fill; }
QPushButton:disabled { background: $disabled_fill; border-color: $disabled_line; color: $disabled_ink; }
QPushButton[cls="primary"] { font-weight: 600; padding: 0px ${px_primary_pad}px;
                             background: $accent; border-color: $accent; color: $on_accent; }
QPushButton[cls="primary"]:hover    { background: $accent_hover; border-color: $accent_hover; }
QPushButton[cls="primary"]:pressed  { background: $accent_press; border-color: $accent_press; }
QPushButton[cls="primary"]:disabled { background: $disabled_fill; border-color: $disabled_line; color: $disabled_ink; }
QPushButton[cls="danger"] { border-color: $err; font-weight: 600; }
QPushButton[cls="danger"]:disabled { border-color: $disabled_line; }
QDialog[cls="ask"] QPushButton[cls="danger"]         { background: $raised; }
QDialog[cls="ask"] QPushButton[cls="danger"]:hover   { background: $hover_fill; }
QDialog[cls="ask"] QPushButton[cls="danger"]:pressed { background: $press_fill; }
QPushButton[cls="nav"]  { min-height: ${px_nav_h}px; padding: 0px ${px_nav_pad}px; border: none; border-radius: 0px;
                          background: transparent; text-align: left; }
QPushButton[cls="nav"][small="true"]   { min-height: ${px_nav_small}px; }
QPushButton[cls="nav"][current="true"] { font-weight: 600; }
QPushButton[cls="fold"] { min-height: ${px_line_h}px; padding: 0px; border: none; background: transparent;
                          text-align: left; font-weight: 600; }
QPushButton[cls="link"] { min-height: ${px_line_h}px; padding: 0px; border: none; background: transparent;
                          color: $accent; text-decoration: underline; }

QLineEdit, QComboBox {
    min-height: ${px_fld_in}px; padding: 0px ${px_fld_pad}px; border: 1px solid $field_line;
    border-radius: ${px_radius}px; background: $field; color: $ink; placeholder-text-color: $ink_3;
    selection-background-color: $accent; selection-color: $on_accent; }
QLineEdit:disabled, QComboBox:disabled {
    background: $disabled_fill; border-color: $disabled_line; color: $disabled_ink;
    placeholder-text-color: $disabled_ink; }
QLineEdit[state="error"] { min-height: ${px_fld_in_err}px; padding: 0px ${px_fld_pad_err}px; border: 2px solid $err; }
QComboBox { padding: 0px ${px_combo_r}px 0px ${px_combo_l}px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: ${px_marker}px;
                       border: none; }
QComboBox QAbstractItemView { border: 1px solid $field_line; background: $field; color: $ink;
                              selection-background-color: $accent_tint; selection-color: $ink; outline: 0; }
QPlainTextEdit {
    min-height: ${px_fld_in}px; padding: 2px ${px_fld_pad}px; border: 1px solid $field_line;
    border-radius: ${px_radius}px; background: $field; color: $ink; placeholder-text-color: $ink_3;
    selection-background-color: $accent; selection-color: $on_accent; }

QCheckBox, QRadioButton { spacing: ${px_check_gap}px; color: $ink; outline: none; }
QRadioButton { spacing: ${px_radio_gap}px; }
QCheckBox:disabled, QRadioButton:disabled { color: $disabled_ink; }

QTableView { border: none; background: $paper; color: $ink; outline: 0; }
QHeaderView { background: $paper; border: none; }

QScrollBar:vertical { background: $margin; width: ${px_bar}px; margin: 0px; }
QScrollBar::handle:vertical { background: $ink_3; border-radius: ${px_handle_r}px; margin: ${px_handle_in}px;
                              min-height: ${px_handle}px; }
QScrollBar:horizontal { background: $margin; height: ${px_bar}px; margin: 0px; }
QScrollBar::handle:horizontal { background: $ink_3; border-radius: ${px_handle_r}px; margin: ${px_handle_in}px;
                                min-width: ${px_handle}px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0px; width: 0px; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
""")

# The rules of the old layout: each serves a widget that is still built the old way, and leaves with it. A list
# and a text viewer are boxes one picks or reads in, so they are dressed as the log pane is; left to the toolkit
# their frame is the page's colour made darker, which is a pale line on the light page and none on the dark one.
# The rule names the list widget and not every list view: the list a drop-down opens is a list view with a rule
# of its own, and must keep its square corners. Greyed, the two look as a greyed field does: a rule that gives a
# box its ground and its text colour takes the palette's greyed ones away, so they are named again. A bare
# scrolling area takes the quiet separator for its frame, but only where it has the frame an area is born with
# (the styled panel, which the toolkit numbers 6): one whose frame was switched off would be given a line, and
# lose a pixel on each side to it. The square where two scroll bars meet takes the bars' own track. The icon
# button takes no padding, so that a glyph in a fixed width is not squeezed, and the inner height of the field
# beside it. The tree view is the records table, a tree used flat; its selection is the accent, which is also
# QPalette.Highlight, so a row painted by hand would match. A tab that can be opened is written in the full
# text colour and a locked one in the greyed colour: in the secondary text colour an open tab stood too near a
# locked one to be told from it. The word of an open tab then has no stronger colour left to answer the pointer
# with, so the tab takes the ground an ordinary button takes there; the sheet's rule for the pointer is never
# given to a greyed tab, so a locked one keeps its look. The tab in front keeps its ground: it leads nowhere.
# The greyed main button names the ordinary weight, the one rule with a state that names a weight at all: bold
# words on a greyed button read as a button one may press. A state otherwise names no weight because a control
# is measured once, at the weight of its plain rule, and wider words would be cut; here the plain rule is the
# bold one, so the lighter words are narrower than what was measured and cannot be cut.
_QSS_INTERIM = Template("""
QListWidget, QTextEdit { border: 1px solid $field_line; border-radius: ${px_radius}px; background: $field;
                         color: $ink; placeholder-text-color: $ink_3; }
QListWidget:disabled, QTextEdit:disabled { border: 1px solid $disabled_line; background: $disabled_fill;
                                           color: $disabled_ink; placeholder-text-color: $disabled_ink; }
QScrollArea[frameShape="6"] { border: 1px solid $rule_light; }
QAbstractScrollArea::corner { background: $margin; border: none; }

QLabel[type="title"]       { font-size: ${px_t_title}px; font-weight: 600; }
QLabel[type="subtitle"]    { font-size: ${px_t_brand}px; font-weight: 600; }
QLabel[type="heading"]     { font-size: ${px_t_head}px; font-weight: 600; }
QLabel[type="caption"]     { font-size: ${px_t_small}px; color: $ink_2; }

QPushButton[accent="true"] {
    background: $accent; color: $on_accent; border: 1px solid $accent; font-weight: 600;
}
QPushButton[accent="true"]:hover   { background: $accent_hover; border-color: $accent_hover; }
QPushButton[accent="true"]:pressed { background: $accent_press; border-color: $accent_press; }
QPushButton[accent="true"]:disabled{ background: $disabled_fill; border-color: $disabled_line; color: $disabled_ink;
                                     font-weight: 400; }
QPushButton#iconButton { padding: 0px; min-height: ${px_fld_in}px; font-size: ${px_i_icon}px; }

QTreeView {
    border: 1px solid $rule_light; border-radius: ${px_radius}px;
    background: $paper; alternate-background-color: $margin; color: $ink;
    selection-background-color: $accent; selection-color: $on_accent; outline: 0;
}
QTreeView::item { padding: 4px 6px; border: none; }
QTreeView::item:hover { background: $hover_fill; }
QTreeView::item:selected { background: $accent; color: $on_accent; }
QHeaderView::section {
    background: $margin; color: $ink_2;
    padding: 4px 6px; border: none;
    border-right: 1px solid $rule_light; border-bottom: 1px solid $rule_light; font-weight: 600;
}
QHeaderView::section:hover { color: $ink; }
QHeaderView::section:last, QHeaderView::section:only-one { border-right: none; }

QGroupBox {
    background: $paper; border: 1px solid $rule_light; border-radius: ${px_radius}px;
    margin-top: 14px; padding: 10px 12px 12px 12px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 4px; background: $paper; color: $ink_2;
}

QTabWidget::pane { border: none; }
QTabBar { qproperty-drawBase: 0; }
QTabBar::tab {
    background: transparent; color: $ink; border: none;
    padding: 6px 14px; margin-right: 4px; border-bottom: 2px solid transparent;
}
QTabBar::tab:hover     { color: $ink; background: $hover_fill; }
QTabBar::tab:selected  { color: $ink; background: transparent; border-bottom: 2px solid $accent; }
QTabBar::tab:disabled  { color: $disabled_ink; }

QProgressBar {
    border: none; border-radius: 3px; background: $margin; text-align: center; color: $ink;
}
QProgressBar::chunk { border-radius: 3px; background: $accent; }

QToolButton { border: none; background: transparent; border-radius: ${px_radius}px; color: $ink; }
QToolButton:hover   { background: $hover_fill; }
QToolButton:pressed { background: $press_fill; }
QToolButton#navBack, QToolButton#navSettings { font-size: ${px_i_nav}px; padding: 2px 6px; }
QToolButton#navRunning {
    font-size: ${px_t_small}px; padding: 2px 8px; color: $ink_2;
    border: 1px solid $field_line; border-radius: ${px_radius}px;
}
QToolButton#navRunning:hover { color: $ink; border-color: $accent; }
QToolButton#helpBadge {
    border: 1px solid $rule_light; border-radius: ${px_badge_r}px; color: $ink_2;
    font-size: ${px_i_badge}px; padding: 0px; background: transparent;
}
QToolButton#helpBadge:hover { border-color: $accent; color: $accent; }

QMenu { background: $raised; border: 1px solid $field_line; border-radius: ${px_radius}px; padding: 4px; }
QMenu::item { padding: 6px 24px 6px 12px; border-radius: ${px_radius}px; color: $ink; }
QMenu::item:selected { background: $accent; color: $on_accent; }
QMenu::separator { height: 1px; background: $rule_light; margin: 4px 8px; }
""")

# These rules serve the widgets of the old layout, and each leaves with its widget: every single selector of
# the second template, in the sheet's order. They alone may name an object, or a property the rules that stay
# do not use.
INTERIM_SELECTORS = (
    "QListWidget", "QTextEdit", "QListWidget:disabled", "QTextEdit:disabled", 'QScrollArea[frameShape="6"]',
    "QAbstractScrollArea::corner",
    'QLabel[type="title"]', 'QLabel[type="subtitle"]', 'QLabel[type="heading"]', 'QLabel[type="caption"]',
    'QPushButton[accent="true"]', 'QPushButton[accent="true"]:hover', 'QPushButton[accent="true"]:pressed',
    'QPushButton[accent="true"]:disabled', "QPushButton#iconButton",
    "QTreeView", "QTreeView::item", "QTreeView::item:hover", "QTreeView::item:selected",
    "QHeaderView::section", "QHeaderView::section:hover", "QHeaderView::section:last",
    "QHeaderView::section:only-one", "QGroupBox", "QGroupBox::title", "QTabWidget::pane", "QTabBar",
    "QTabBar::tab", "QTabBar::tab:hover", "QTabBar::tab:selected", "QTabBar::tab:disabled", "QProgressBar",
    "QProgressBar::chunk", "QToolButton", "QToolButton:hover", "QToolButton:pressed", "QToolButton#navBack",
    "QToolButton#navSettings", "QToolButton#navRunning", "QToolButton#navRunning:hover", "QToolButton#helpBadge",
    "QToolButton#helpBadge:hover", "QMenu", "QMenu::item", "QMenu::item:selected", "QMenu::separator")

# the icon font's glyph sizes at the ordinary text size; only rules of the old layout read them
_GLYPH_PX = {"px_i_icon": 18, "px_i_nav": 20, "px_i_badge": 13}


def _sheet_sizes() -> dict[str, int]:
    """Every size the sheet writes, in pixels at the text size in force, under the name its placeholder has.
    A height is what stands inside the border: the outer height less the two lines, and a field in error, whose
    border is twice as heavy, gives up as much again in height and in padding, so it keeps its outer size. The
    radius and the scroll bar with its handle do not grow with the text; the help badge's corner does, being
    half the badge's side."""
    size = SIZE
    handle, handle_in, handle_r = size["scroll_handle"]
    return {
        "px_t_title": scaled(TYPE["title"][0]), "px_t_head": scaled(TYPE["head"][0]),
        "px_t_brand": scaled(TYPE["brand"][0]), "px_t_small": scaled(TYPE["small"][0]),
        "px_btn_in": scaled(size["button_h"]) - 2,
        "px_btn_pad": scaled(size["button_pad"]), "px_primary_pad": scaled(size["primary_pad"]),
        "px_fld_in": scaled(size["field_h"]) - 2, "px_fld_in_err": scaled(size["field_h"]) - 4,
        "px_fld_pad": scaled(size["field_pad"]), "px_fld_pad_err": scaled(size["field_pad"]) - 1,
        "px_combo_l": scaled(size["combo_pad"][0]), "px_combo_r": scaled(size["combo_pad"][1]),
        "px_marker": scaled(size["marker"]), "px_line_h": scaled(size["line_h"]),
        "px_nav_h": scaled(size["nav_h"]), "px_nav_small": scaled(size["nav_h_small"]),
        "px_nav_pad": scaled(size["nav_pad"]),
        "px_check_gap": scaled(size["check_gap"]), "px_radio_gap": scaled(size["radio_gap"]),
        "px_tip_v": scaled(size["tip_pad"][0]), "px_tip_h": scaled(size["tip_pad"][1]),
        "px_radius": size["radius"], "px_bar": size["scroll_bar"],
        "px_handle": handle, "px_handle_in": handle_in, "px_handle_r": handle_r,
        **{name: scaled(px) for name, px in _GLYPH_PX.items()},
        # 16 is the side the help badge gives itself (widgets/help_badge.py), scaled as it scales it: half of
        # that, rounded down, keeps the badge round at every text size
        "px_badge_r": scaled(16) // 2,
    }


def build_qss(dark: bool) -> str:
    """The application's one style sheet for one theme, at the text size in force: the rules that stay, then
    the rules of the old layout."""
    names = {**tokens(dark), **_sheet_sizes()}
    return _QSS.substitute(names) + _QSS_INTERIM.substitute(names)
