"""The look: its one table of colours, type and sizes, and the functions every reader of it shares.

Nothing here imports torch, so the look can be tested without the engine;
``test_the_look_and_this_file_load_without_the_engine`` proves it in a fresh interpreter, which runs this whole file:
its module level holds imports, constants and definitions only. One application object serves every suite of a run,
so the theme and the text factor are changed here only inside a helper or a fixture that puts back what it found:
``theme`` and ``text_size`` are the two helpers for it.
"""
import ast
import contextlib
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (QFocusEvent, QFont, QFontDatabase, QFontInfo, QFontMetricsF, QGuiApplication, QImage,
                           QKeyEvent, QMouseEvent, QPainter, QRegion, QWheelEvent)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QLineEdit, QMenu, QPlainTextEdit, QPushButton,
                               QRadioButton, QStyle, QStyleFactory, QStyleOptionButton, QStyleOptionFocusRect,
                               QWidget)

from core.gui import design, fonts

REPO = Path(__file__).resolve().parents[1]
# every module of the look that exists; a new one joins this tuple in the commit that adds it, and the
# fresh-interpreter test is then its proof that it loads no engine
LOOK_MODULES = ("core.gui.design", "core.gui.fonts", "core.gui.theming", "core.gui.mpl_theme", "core.gui.settings",
                "core.gui.file_chooser", "core.gui.widgets.labeled_inputs", "core.gui.paint", "core.gui.style")
# what the look may never load: the engine, and the modules that reach it
FORBIDDEN_MODULES = ("torch", "core.config", "core.registry", "core.gui.fields", "core.Helpers.file_manager", "tests._fixtures")


@contextlib.contextmanager
def theme(dark):
    """The theme in force inside the block; what was in force before is put back."""
    was = design.is_dark()
    design.set_dark(dark)
    try:
        yield
    finally:
        design.set_dark(was)


@contextlib.contextmanager
def text_size(factor):
    """The text factor in force inside the block; what was in force before is put back."""
    was = design.text_scale()
    design.set_text_scale(factor)
    try:
        yield
    finally:
        design.set_text_scale(was)


# name: (light, dark). Typed a second time on purpose, so that a changed colour fails here by name.
EXPECTED_COLOURS = {
    "paper": ("#FFFFFF", "#1C1C1C"), "margin": ("#F2F2F2", "#141414"), "field": ("#FFFFFF", "#141414"), "raised": ("#FFFFFF", "#262626"),
    "ink": ("#1A1A1A", "#EDEDED"), "ink_2": ("#4D4D4D", "#B8B8B8"), "ink_3": ("#636363", "#9C9C9C"), "rule": ("#1A1A1A", "#C8C8C8"),
    "rule_light": ("#D4D4D4", "#3A3A3A"), "field_line": ("#6B6B6B", "#8A8A8A"), "accent": ("#0072B2", "#56B4E9"), "on_accent": ("#FFFFFF", "#0D0D0D"),
    "accent_tint": ("#E2EEF6", "#1A3447"), "accent_hover": ("#00639B", "#7CC5EE"), "accent_press": ("#005382", "#3FA3DC"), "ok": ("#007A5A", "#2FBF94"),
    "warn": ("#E69F00", "#E69F00"), "warn_edge": ("#1A1A1A", "#E69F00"), "warn_mark": ("#1A1A1A", "#1A1A1A"), "err": ("#C24E00", "#F0813A"),
    "on_err": ("#FFFFFF", "#0D0D0D"), "focus": ("#1A1A1A", "#FFFFFF"), "hover_fill": ("#F2F2F2", "#262626"), "press_fill": ("#DEDEDE", "#343434"),
    "disabled_fill": ("#F2F2F2", "#232323"), "disabled_line": ("#C4C4C4", "#4A4A4A"), "disabled_ink": ("#636363", "#9C9C9C"), "hatch": ("#4D4D4D", "#B8B8B8"),
    "elapsed": ("#1A1A1A", "#C8C8C8"), "series_2": ("#0072B2", "#56B4E9"), "series_6": ("#56B4E9", "#0072B2"), "series_7": ("#CC79A7", "#CC79A7"),
}
SIZES_DIGEST = "52347416e5db2d1e1206f3db9d868896c9cba54feea87fc4267038d7472158f7"


def test_the_colour_table_is_the_32_named_pairs_in_order():
    assert list(design.COLOURS) == list(EXPECTED_COLOURS)
    for name, pair in EXPECTED_COLOURS.items():
        assert design.COLOURS[name] == pair, name
    assert design.SERIES == ("ink", "series_2", "err", "ok", "warn", "series_6", "series_7")
    for dark in (0, 1):                                      # seven lines, seven colours, in either theme
        assert len({design.COLOURS[name][dark] for name in design.SERIES}) == 7, dark


def test_type_has_five_sizes_two_weights_and_three_faces():
    assert design.TYPE == {"small": (12, 16, 400), "body": (14, 20, 400), "strong": (14, 20, 600), "head": (16, 22, 600),
                           "brand": (18, 24, 600), "title": (24, 30, 600), "book": (16, 22, 400)}
    assert sorted({size for size, _line, _weight in design.TYPE.values()}) == [12, 14, 16, 18, 24]
    assert {weight for _size, _line, weight in design.TYPE.values()} == {400, 600}
    assert design.LETTERING == "Inter" and design.BOOK_FACES == ("Cambria", "Georgia")
    assert design.FIXED_FACES == ("Cascadia Mono", "Consolas", "Courier New") and design.BRAND_SPACING == 104


def test_the_size_tables_are_the_agreed_numbers():
    size, dialog = design.SIZE, design.DIALOG
    tables = {name: getattr(design, name) for name in ("TYPE", "SIZE", "GAP", "STEP", "AXIS", "DIALOG", "FIGURE")}
    assert [len(tables[name]) for name in ("SIZE", "GAP", "STEP", "AXIS", "DIALOG", "FIGURE")] == [54, 40, 11, 11, 6, 11]
    assert size["window"] == (1300, 820) and size["window_min"] == (900, 600)
    assert (size["small_below"], size["two_columns_from"], size["rail"]) == (700, 940, 220)
    assert size["page_pad"] == (22, 20, 20, 14) and size["page_pad_small"] == (16, 16, 14, 14)
    assert (size["nav_h"], size["nav_h_small"], size["step_h"], size["step_h_one"]) == (36, 32, 46, 32)
    assert (size["field_h"], size["button_h"], size["radius"], size["number_w"], size["indicator"]) == (32, 34, 3, 120, 18)
    assert size["label_w"] == (200, 165) and size["row_gap"] == 12 and size["table_row"] == 30
    assert (size["marker"], dialog["marker"], dialog["w"]) == (16, 24, 448)
    assert (size["rule"], size["hair"], size["bar"], size["focus_ring"], size["focus_gap"]) == (2, 1, 4, 2, 2)
    assert design.TYPE["small"][1] + 2 * size["table_pad"] + size["hair"] == 27          # a table's header
    text = json.dumps(tables, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == SIZES_DIGEST, \
        "a size of the look's table changed: if it is meant, say which one and record the new digest here"


def test_weights_and_radii_are_named_as_not_growing_with_the_text():
    assert design.UNSCALED == frozenset({
        "SIZE.rule", "SIZE.hair", "SIZE.bar", "SIZE.focus_ring", "SIZE.focus_gap", "SIZE.row_ring_inset", "SIZE.radius",
        "SIZE.hatch_step", "SIZE.scroll_bar", "SIZE.scroll_gap", "SIZE.scroll_handle",
        "AXIS.mark_w", "AXIS.mark_halo", "FIGURE.axis_w", "FIGURE.line_w", "FIGURE.tick"})
    for dotted in design.UNSCALED:                           # every entry names a key that exists
        table, name = dotted.split(".")
        assert name in getattr(design, table), dotted
    axis = design.AXIS                                       # the mark's height grows, so it still overhangs the track
    for factor in (1.0, 1.25, 1.5, 2.0, 2.25):
        with text_size(factor):
            assert design.scaled(axis["mark_h"]) > design.scaled(axis["track_y"]) + design.scaled(axis["track_h"]), factor


def test_a_colour_is_read_in_the_theme_in_force():
    with theme(False):
        assert design.is_dark() is False
        assert design.colour("ink").name().upper() == "#1A1A1A" and design.hex_colour("ink") == "#1A1A1A"
        assert design.hex_colour("ink", dark=True) == "#EDEDED"              # asked for by theme, whatever is in force
    with theme(True):
        assert design.is_dark() is True
        assert design.colour("ink").name().upper() == "#EDEDED" and design.hex_colour("accent") == "#56B4E9"
        assert design.hex_colour("accent", dark=False) == "#0072B2"
    assert design.colour("paper") is not design.colour("paper")              # a caller may change its own copy
    for read in (design.colour, design.hex_colour):
        with pytest.raises(KeyError):
            read("no_such_colour")


def _luminance(hex_colour):
    red, green, blue = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (red, green, blue)]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(fore, back, dark):
    """The contrast of two colours of the table in one theme, by the WCAG 2 formula."""
    one, other = (_luminance(design.COLOURS[name][1 if dark else 0]) for name in (fore, back))
    return (max(one, other) + 0.05) / (min(one, other) + 0.05)


def _pairs(*rows):
    return [(fore, back) for fores, backs in rows for fore in fores.split() for back in backs.split()]


LETTERING_PAIRS = _pairs(                                    # floor 4.5
    ("ink", "paper margin field raised accent_tint hover_fill press_fill"), ("ink_2", "paper margin raised accent_tint"),
    ("ink_3", "paper margin field"), ("disabled_ink", "disabled_fill paper margin"),
    ("on_accent", "accent accent_hover accent_press"), ("accent", "paper raised"))
MARK_PAIRS = _pairs(                                         # floor 3.0
    ("field_line", "paper field raised margin"), ("rule hatch elapsed series_2", "paper"),
    ("accent", "paper margin accent_tint"), ("ok err", "paper raised margin accent_tint"), ("on_err", "err"),
    ("warn_edge", "paper raised margin accent_tint"), ("warn_mark", "warn"),
    ("focus", "paper margin raised accent_tint"), ("ink_2 ink_3", "paper margin"))
# fore, back, its floor, light, dark. Each stands BELOW its floor in one theme at least; the words say what carries it.
BELOW_THE_FLOOR = [
    ("warn", "paper", 3.0, 2.25, 7.57),                      # the triangle; and a figure's orange lines, on a light page
    ("warn", "raised", 3.0, 2.25, 6.72),                     # the triangle's edge and its mark carry it on every ground
    ("warn", "margin", 3.0, 2.01, 8.18),
    ("warn", "accent_tint", 3.0, 1.91, 5.73),
    ("rule_light", "paper", 3.0, 1.48, 1.50),                # never the only sign of anything
    ("disabled_line", "disabled_fill", 3.0, 1.56, 1.77),     # a disabled control; its text passes
    ("accent", "elapsed", 3.0, 3.36, 1.38),                  # the "now" mark has paper each side and overhangs the track
    ("focus", "accent", 3.0, 3.36, 2.31),                    # the ring stands clear of the main button
    ("accent_tint", "paper", 3.0, 1.18, 1.32),               # a selected row also has two lines, a bar and a semibold name
    ("accent", "accent_tint", 4.5, 4.39, 5.60),              # as lettering: never allowed, no link stands on a selected row
    ("series_6", "paper", 3.0, 2.31, 3.29),                  # a figure's sky-blue lines only, always on a light page
]


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_lettering_and_marks_keep_their_contrast_floors(dark):
    assert len(LETTERING_PAIRS) == 22 and len(MARK_PAIRS) == 33
    low = {pair: round(contrast(*pair, dark), 2) for pair in LETTERING_PAIRS if contrast(*pair, dark) < 4.5}
    low.update({pair: round(contrast(*pair, dark), 2) for pair in MARK_PAIRS if contrast(*pair, dark) < 3.0})
    assert not low, low
    assert round(min(contrast(*pair, dark) for pair in LETTERING_PAIRS), 2) == (5.72 if dark else 5.19)
    assert round(min(contrast(*pair, dark) for pair in MARK_PAIRS), 2) == (4.38 if dark else 4.06)


def test_each_named_exception_stands_below_its_floor_as_recorded():
    for fore, back, floor, light, dark in BELOW_THE_FLOOR:
        got = (round(contrast(fore, back, False), 2), round(contrast(fore, back, True), 2))
        assert got == (light, dark), (fore, back, got)
        assert min(got) < floor, (fore, back, "no longer an exception: it belongs with the pairs that must pass")
    grounds = ("paper", "raised", "margin", "accent_tint")   # what carries the triangle, on every ground it stands on
    assert round(min(contrast("warn_edge", ground, False) for ground in grounds), 2) == 14.75
    assert round(min(contrast("warn_edge", ground, True) for ground in grounds), 2) == 5.73
    assert round(contrast("warn_edge", "paper", False), 2) == 17.40 and round(contrast("warn_mark", "warn", False), 2) == 7.73
    disabled = [round(contrast("disabled_ink", "disabled_fill", dark), 2) for dark in (False, True)]
    assert disabled == [5.37, 5.72]                          # a form locked for days stays readable


def test_a_figures_seven_line_colours_on_the_page():
    light = [round(contrast(name, "paper", False), 2) for name in design.SERIES]
    dark = [round(contrast(name, "paper", True), 2) for name in design.SERIES]
    assert light == [17.40, 5.19, 4.79, 5.34, 2.25, 2.31, 3.06]              # orange and sky blue: the two exceptions above
    assert dark == [14.56, 7.39, 6.42, 7.30, 7.57, 3.29, 5.57]
    # the floor for marks: on the light page every line colour but those two keeps it; on the dark page all seven do
    assert {name for name, ratio in zip(design.SERIES, light) if ratio < 3.0} == {"warn", "series_6"}
    assert min(dark) >= 3.0


def test_a_length_grows_with_the_text_factor_and_halves_round_up():
    with text_size(1.0):
        assert [design.scaled(n) for n in (0, 1, 2, 14, 16, 34, 900)] == [0, 1, 2, 14, 16, 34, 900]
    with text_size(1.25):
        assert design.text_scale() == 1.25
        assert [design.scaled(n) for n in (2, 10, 12, 14, 18, 22, 34, 900)] == [3, 13, 15, 18, 23, 28, 43, 1125]
        # a quarter rounds down and three quarters up: neither a ceiling nor a floor would do
        assert [design.scaled(n) for n in (1, 3, 5)] == [1, 4, 6]
    with text_size(2.0):
        assert [design.scaled(n) for n in (1, 2, 14, 34, 600)] == [2, 4, 28, 68, 1200]
        assert type(design.scaled(14)) is int


def test_the_text_factor_is_held_between_the_ordinary_size_and_the_most_windows_offers():
    for asked, held in ((0.5, 1.0), (1.0, 1.0), (1.5, 1.5), (2.25, 2.25), (3.0, 2.25), (float("nan"), 1.0)):
        with text_size(asked):
            assert design.text_scale() == held, asked


def _readers(monkeypatch, documented, stored, windows=True):
    """Stand-ins for the two reads of Windows' text size. Each notes that it was asked, and an exception handed in
    is raised. No test in this process asks Windows itself."""
    asked = []

    def stand_in(label, answer):
        def read():
            asked.append(label)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        return read

    monkeypatch.setattr(design, "_on_windows", lambda: windows)
    monkeypatch.setattr(design, "_windows_text_scale", stand_in("documented", documented))
    monkeypatch.setattr(design, "_registry_text_scale", stand_in("stored", stored))
    return asked


def test_the_factor_windows_reports_comes_from_its_documented_setting_first(monkeypatch):
    asked = _readers(monkeypatch, 1.5, 2.0)
    assert design.system_text_scale() == 1.5 and asked == ["documented"]     # the stored value is not consulted
    asked = _readers(monkeypatch, None, 1.75)
    assert design.system_text_scale() == 1.75 and asked == ["documented", "stored"]
    asked = _readers(monkeypatch, OSError("no such library"), 1.25)          # a read that raises has failed
    assert design.system_text_scale() == 1.25 and asked == ["documented", "stored"]
    assert type(design.system_text_scale()) is float


def test_the_factor_windows_reports_can_never_raise_or_leave_its_range(monkeypatch):
    for documented, stored, expected in ((None, None, 1.0), (OSError(), ValueError(), 1.0), (9.0, 1.0, 2.25),
                                         (0.4, 2.0, 1.0), (float("nan"), 1.5, 1.5), ("large", 1.75, 1.75)):
        _readers(monkeypatch, documented, stored)
        assert design.system_text_scale() == expected, (documented, stored)
    asked = _readers(monkeypatch, 1.5, 1.5, windows=False)
    assert design.system_text_scale() == 1.0 and asked == []                 # off Windows nothing is asked


def test_the_stored_text_size_is_a_percentage_and_its_absence_is_the_ordinary_size(monkeypatch):
    """The reader of the stored value, run against a stand-in for the registry's module: on a computer at the
    ordinary text size the value is absent, so nothing else would ever run the lines that read one."""
    asked = []

    class _Opened:
        def __enter__(self):
            return self

        def __exit__(self, *raised):
            return False

    def registry(stored):
        def open_key(hive, path):
            asked.append((hive, path))
            return _Opened()

        def query(opened, name):
            asked.append(name)
            if stored is None:
                raise FileNotFoundError(name)
            return stored, 4
        return types.SimpleNamespace(HKEY_CURRENT_USER="the user's own", OpenKey=open_key, QueryValueEx=query)

    monkeypatch.setitem(sys.modules, "winreg", registry(150))
    assert design._registry_text_scale() == 1.5
    assert asked == [("the user's own", r"Software\Microsoft\Accessibility"), "TextScaleFactor"]
    monkeypatch.setitem(sys.modules, "winreg", registry(None))
    assert design._registry_text_scale() == 1.0                              # no value stored: the ordinary size
    monkeypatch.setitem(sys.modules, "winreg", registry("225"))
    assert design._registry_text_scale() == 2.25


def test_the_window_opens_and_stops_shrinking_at_sizes_that_fit_the_text_and_the_screen():
    with text_size(1.0):
        assert design.window_sizes((3400, 1332)) == ((1300, 820), (900, 600))
        assert design.window_sizes((1326, 668)) == ((1300, 668), (900, 600))     # the room a small laptop offers
        assert design.window_sizes((760, 740)) == ((760, 740), (760, 600))       # less room than the smallest window asks
        assert design.window_sizes((16_777_215, 16_777_215)) == ((1300, 820), (900, 600))
        assert design.window_sizes((-30, -50)) == ((1, 1), (1, 1))               # never nothing
    with text_size(1.25):
        assert design.window_sizes((1880, 980)) == ((1625, 980), (1125, 750))
    with text_size(2.25):
        assert design.window_sizes((3400, 1332)) == ((2925, 1332), (2025, 1332))
        assert design.window_sizes((1366, 768)) == ((1366, 768), (1366, 768))    # the largest text on a small laptop: never past the screen


def test_a_number_is_written_with_commas_between_thousands_and_a_point_for_the_decimal():
    for value, text in ((20480000, "20,480,000"), (1000, "1,000"), (999, "999"), (0, "0"), (-3, "-3"),
                        (-20480000, "-20,480,000"), (40.1, "40.1"), (0.05, "0.05"), (2048.0, "2,048.0"),
                        (1234567.5, "1,234,567.5"), (-1234.5, "-1,234.5")):
        assert design.number_text(value) == text, value
    assert design.number_text(40.1) + " %" == "40.1 %" and design.number_text(6) + " GiB" == "6 GiB"
    for refused in (True, "5000", None):                     # a truth value and a text are not quantities
        with pytest.raises(TypeError):
            design.number_text(refused)


def test_a_date_is_written_one_way_and_carries_its_weekday_within_seven_days_of_now():
    at, week = datetime.datetime, datetime.timedelta(days=7)
    when, now = at(2026, 10, 4, 12, 10, 59), at(2026, 10, 6, 9, 0)
    assert design.date_text(at(2026, 10, 3, 14, 2)) == "3 Oct 2026, 14:02"       # no present moment: the full form
    assert design.date_text(when) == "4 Oct 2026, 12:10"                         # seconds are dropped, not rounded
    assert design.date_text(when, now) == "Sun 4 Oct, 12:10"
    assert design.date_text(at(2026, 10, 7, 19, 0), now) == "Wed 7 Oct, 19:00"   # ahead of now as well as behind it
    assert design.date_text(at(2026, 10, 1, 19, 2), now) == "Thu 1 Oct, 19:02"
    assert design.date_text(now - week, now) == "29 Sep 2026, 09:00"             # exactly seven days is outside
    assert design.date_text(now + week, now) == "13 Oct 2026, 09:00"
    assert design.date_text(now - week + datetime.timedelta(minutes=1), now) == "Tue 29 Sep, 09:01"
    assert design.date_text(at(2027, 1, 2, 0, 5), at(2026, 12, 30, 23, 0)) == "Sat 2 Jan, 00:05"
    assert design.date_text(at(2026, 3, 9, 7, 5)) == "9 Mar 2026, 07:05"
    assert design.MONTHS == ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    assert design.WEEKDAYS == ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    utc = datetime.timezone.utc                              # written as handed in: nothing is converted
    assert design.date_text(when.replace(tzinfo=utc), now.replace(tzinfo=utc)) == "Sun 4 Oct, 12:10"
    with pytest.raises(TypeError):                           # one moment with a zone and one without
        design.date_text(when.replace(tzinfo=utc), now)


def _reaching(loaded, forbidden=FORBIDDEN_MODULES):
    """The names among ``loaded`` that are, or lie under, a module the look may never load."""
    return sorted(name for name in loaded if any(name == root or name.startswith(root + ".") for root in forbidden))


_FRESH = """
import datetime, importlib, locale, runpy, sys
for name in sys.argv[2].split(","):
    importlib.import_module(name)
suite = runpy.run_path(sys.argv[1], run_name="look_suite")
from core.gui import design
assert design.is_dark() is False and design.text_scale() == 1.0, "a fresh process starts light, at the ordinary text size"
factor = design.system_text_scale()
assert type(factor) is float and 1.0 <= factor <= 2.25, factor
if sys.platform == "win32":                                         # each reader once by itself: a slip in either body shows here
    for read in (design._windows_text_scale, design._registry_text_scale):
        answer = read()
        assert answer is None or type(answer) is float, (read.__name__, answer)
for regional in ("de-DE", "de_DE", "German_Germany.1252"):          # a regional setting that writes "So" and "Okt"
    try:
        locale.setlocale(locale.LC_ALL, regional)
        break
    except locale.Error:
        pass
else:
    assert sys.platform != "win32", "no German regional setting could be set"
at = datetime.datetime
assert design.date_text(at(2026, 10, 4, 12, 10), at(2026, 10, 6, 9, 0)) == "Sun 4 Oct, 12:10"
assert design.date_text(at(2026, 10, 3, 14, 2)) == "3 Oct 2026, 14:02" and design.number_text(1234567.5) == "1,234,567.5"
bad = suite["_reaching"](sys.modules)
sys.exit(repr(bad) if bad else 0)
"""


def test_the_look_and_this_file_load_without_the_engine():
    """In a pytest process torch is already loaded by the session's fixtures, so a look at ``sys.modules`` here would
    prove nothing: the proof is a fresh interpreter that imports every module of the look and runs this file. There
    it also asks Windows for its text size (through the function, then each of its two readers once by itself) and
    writes a date under a German regional setting if one can be set."""
    assert _reaching(["os", "torch", "torch.nn", "core.configuration", "core.config", "tests._fixtures", "core.gui.design"]) \
        == ["core.config", "tests._fixtures", "torch", "torch.nn"]
    named = set()
    for node in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            named |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            named |= {node.module or ""} | {f"{node.module}.{alias.name}" for alias in node.names}
    assert not _reaching(named) and not [name for name in named if name.endswith("_fixtures")], sorted(named)
    r = subprocess.run([sys.executable, "-c", _FRESH, __file__, ",".join(LOOK_MODULES)],
                       cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


# ── the appearance: three choices, and an apply that changes nothing does nothing ────────────────
def test_there_are_three_appearance_modes_and_no_others(tmp_path):
    """Follow Windows, Light, Dark, in that order and under those words; and each comes back from a
    settings file as itself, so the list of modes and the settings reader cannot drift apart."""
    from PySide6.QtCore import QSettings
    from core.gui import settings, theming

    assert theming.MODE_LABELS == (("system", "Follow Windows"), ("light", "Light"), ("dark", "Dark"))
    assert theming.MODES == ("system", "light", "dark")
    path = str(tmp_path / "modes.ini")
    for mode in theming.MODES:
        qs = QSettings(path, QSettings.IniFormat)
        settings.set_appearance(qs, mode)
        qs.sync()
        assert settings.get_appearance(QSettings(path, QSettings.IniFormat)) == mode


def test_a_stored_mode_that_is_not_offered_reads_as_follow_windows(tmp_path):
    """A word an older build stored (the time-of-day mode among them), a word in another case, a
    blank and a number all read as the first choice: through the settings object that stored them,
    through a second one on the same file, and from a file written by hand. Only the files written
    by hand are read from disk (a second object on a file is served what the first one holds), so
    the retired word, an unknown word, a blank and the spaced word are written by hand as well.
    Reading rewrites nothing: the old word stays in the file until the next choice is made."""
    from PySide6.QtCore import QSettings
    from core.gui import settings

    none = QSettings(str(tmp_path / "none.ini"), QSettings.IniFormat)
    assert settings.get_appearance(none) == "system"
    none.sync()
    assert not (tmp_path / "none.ini").exists(), "reading a file that holds no mode wrote one into it"
    for n, stored in enumerate(("auto", "Light", "DARK", "", "sepia", 7, " light ")):     # the last: nothing is stripped
        path = tmp_path / f"stored_{n}.ini"
        qs = QSettings(str(path), QSettings.IniFormat)
        qs.setValue("appearance/mode", stored)
        qs.sync()
        on_disk = path.read_bytes()
        assert settings.get_appearance(qs) == "system", stored      # as held: a number is still a number
        qs.sync()
        assert path.read_bytes() == on_disk, f"reading through the object that stored {stored!r} rewrote its file"
        again = QSettings(str(path), QSettings.IniFormat)
        assert settings.get_appearance(again) == "system", stored
        again.sync()
        assert path.read_bytes() == on_disk, f"reading rewrote the file that held {stored!r}"
    # read from disk: the first comes back as a list; the last is the spaced word as the toolkit stores it, in quotes
    for n, line in enumerate(("light, dark", "7", "Dark", "auto", "sepia", "", '" light "')):
        by_hand = tmp_path / f"by_hand_{n}.ini"
        by_hand.write_text(f"[appearance]\nmode={line}\n", encoding="utf-8")
        assert settings.get_appearance(QSettings(str(by_hand), QSettings.IniFormat)) == "system", line


def test_the_two_retired_appearance_keys_stay_in_the_file_and_nothing_reads_them(tmp_path):
    """The accent switch and the font switch are gone: no helper for either, no method on the
    Appearance, and neither key named anywhere in the window's code. A file that still holds them
    opens as if they were not there, and a new choice of mode leaves them where they are."""
    from pathlib import Path
    from PySide6.QtCore import QSettings
    import core.gui
    from core.gui import settings, theming

    for gone in ("get_system_accent", "set_system_accent", "get_force_inter", "set_force_inter"):
        assert not hasattr(settings, gone), f"settings.{gone} still serves a retired key"
    for gone in ("system_accent_enabled", "set_system_accent"):
        assert not hasattr(theming.Appearance, gone), f"Appearance.{gone} is still there"
    keys = ("appearance/system_accent", "appearance/force_inter")
    readers = sorted(p.name for p in Path(core.gui.__file__).resolve().parent.rglob("*.py")
                     if any(key in p.read_text(encoding="utf-8") for key in keys))
    assert readers == [], f"these files still name a retired key: {readers}"
    old = tmp_path / "old.ini"
    old.write_text("[appearance]\nmode=dark\nsystem_accent=1\nforce_inter=1\n", encoding="utf-8")
    assert settings.get_appearance(QSettings(str(old), QSettings.IniFormat)) == "dark"
    qs = QSettings(str(old), QSettings.IniFormat)
    settings.set_appearance(qs, "light")
    qs.sync()
    text = old.read_text(encoding="utf-8")
    assert "mode=light" in text and "system_accent=1" in text and "force_inter=1" in text, text


def test_an_apply_goes_on_only_for_a_scheme_not_yet_applied():
    """The one decision of an apply, as a pure function: the first apply always goes on, and after
    it only a scheme that differs from the one applied."""
    from core.gui.theming import should_apply

    assert should_apply(None, (False,)) is True and should_apply(None, (True,)) is True
    assert should_apply((False,), (False,)) is False and should_apply((True,), (True,)) is False
    assert should_apply((False,), (True,)) is True and should_apply((True,), (False,)) is True


@pytest.fixture
def stand_in_app():
    """An application stand-in that records what an Appearance sets on it, so these tests set
    nothing on the real one. Its style hints answer colorScheme() as Windows does: the scheme last
    asked for, or what Windows itself holds (``windows``) when that was Unknown. Off screen the real
    hints always answer Unknown and never signal, so only a stand-in can play a computer that turns
    dark. The module's active Appearance is put back afterwards. It also records the fonts an apply
    sets for widget classes, keeps the font last set under each name (``given``; None is the
    application's own) and hands out no top-level widget; the look's theme flag is put back
    afterwards."""
    from PySide6.QtCore import QObject, Qt, Signal
    from PySide6.QtWidgets import QApplication
    from core.gui import theming

    _real = QApplication.instance() or QApplication([])    # palettes and signals want one to exist

    class Hints(QObject):                                   # what styleHints() hands out
        colorSchemeChanged = Signal(Qt.ColorScheme)

        def __init__(self):
            super().__init__()
            self.windows, self.asked = Qt.ColorScheme.Light, []

        def setColorScheme(self, scheme):
            self.asked.append(scheme)

        def colorScheme(self):
            last = self.asked[-1] if self.asked else Qt.ColorScheme.Unknown
            return self.windows if last == Qt.ColorScheme.Unknown else last

    class Application(QObject):
        def __init__(self):
            super().__init__()
            self.hints, self.palettes, self.sheets = Hints(), [], []
            self.fonts, self.tops, self.given = [], [], {}

        def styleHints(self):
            return self.hints

        def setPalette(self, palette):
            self.palettes.append(palette)

        def setStyleSheet(self, sheet):
            self.sheets.append(sheet)

        def setFont(self, font, name=None):        # the fonts of widget classes, as an apply sets them
            self.fonts.append(name)
            self.given[name] = font

        def font(self):                            # the application's own font, once one was set
            return self.given[None]

        def topLevelWidgets(self):
            return list(self.tops)

    active = theming._ACTIVE
    was_dark = design.is_dark()
    try:
        yield Application()
    finally:
        theming._ACTIVE = active
        design.set_dark(was_dark)


def test_the_first_apply_goes_on_and_a_repeat_of_the_same_scheme_sets_nothing(stand_in_app):
    """Windows is light and so is an Appearance's starting state, yet its first apply must set the
    palette and the sheet: nothing was applied before it. After that, choosing the scheme already
    shown sets neither and signals nothing, while the frames Windows draws are still told. What
    was applied is each Appearance's own: a second one on the same application starts from
    nothing applied, and applies even the scheme the first one left."""
    from PySide6.QtCore import Qt
    from core.gui import theming

    app = stand_in_app
    appearance = theming.Appearance(app)
    signalled = []
    appearance.theme_changed.connect(signalled.append)
    appearance.set_mode("system")
    assert (len(app.palettes), len(app.sheets), signalled) == (1, 1, [False]), "the first apply"
    appearance.set_mode("light")
    appearance.set_mode("light")
    assert (len(app.palettes), len(app.sheets), signalled) == (1, 1, [False]), "a repeat set something"
    assert app.hints.asked == [Qt.ColorScheme.Unknown, Qt.ColorScheme.Light, Qt.ColorScheme.Light]
    appearance.set_mode("dark")
    assert (len(app.palettes), len(app.sheets), signalled) == (2, 2, [False, True])
    assert appearance.is_dark() is True and appearance.mode() == "dark"
    another = theming.Appearance(app)
    another.set_mode("dark")
    assert (len(app.palettes), len(app.sheets)) == (3, 3), \
        "a new Appearance did not apply: what was applied must be kept by each one, not shared"


def test_follow_windows_follows_a_change_made_while_open_and_a_chosen_scheme_does_not(stand_in_app):
    """While PRISM follows Windows, Windows turning dark turns PRISM dark, once; being told the same
    again sets nothing. Once a scheme is chosen by name, what Windows does is ignored; and choosing
    Follow Windows again takes up what Windows shows by then."""
    from PySide6.QtCore import Qt
    from core.gui import theming

    app = stand_in_app
    appearance = theming.Appearance(app)
    signalled = []
    appearance.theme_changed.connect(signalled.append)
    appearance.set_mode("system")
    app.hints.windows = Qt.ColorScheme.Dark
    app.hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    assert (len(app.sheets), signalled, appearance.is_dark()) == (2, [False, True], True)
    app.hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    assert (len(app.sheets), signalled) == (2, [False, True])
    appearance.set_mode("dark")
    app.hints.windows = Qt.ColorScheme.Light
    app.hints.colorSchemeChanged.emit(Qt.ColorScheme.Light)
    assert (len(app.sheets), signalled, appearance.is_dark()) == (2, [False, True], True)
    appearance.set_mode("system")
    assert (len(app.sheets), signalled, appearance.is_dark()) == (3, [False, True, False], False)


def test_a_retired_mode_handed_to_the_appearance_is_follow_windows_and_no_clock_runs(stand_in_app):
    """The time-of-day mode is gone from the Appearance too: its word is taken as the first choice,
    and nothing re-applies the look on a timer."""
    from PySide6.QtCore import Qt, QTimer
    from core.gui import theming

    app = stand_in_app
    appearance = theming.Appearance(app)
    appearance.set_mode("auto")
    assert appearance.mode() == "system"
    assert app.hints.asked == [Qt.ColorScheme.Unknown]
    assert appearance.findChildren(QTimer) == []


def test_a_listener_of_the_theme_signal_may_choose_a_mode(stand_in_app):
    """A listener that is told the theme changed may choose a mode itself, and its choice is applied.
    The signal is sent once the apply is over: sent from inside it, the listener's choice would meet
    the guard against an apply entered twice and be dropped without a word."""
    from core.gui import theming

    app = stand_in_app
    appearance = theming.Appearance(app)
    heard = []

    def listener(dark):
        heard.append(dark)
        if heard == [True]:                    # told for the first time that the theme went dark
            appearance.set_mode("light")

    appearance.theme_changed.connect(listener)
    appearance.set_mode("dark")
    assert (len(app.palettes), len(app.sheets)) == (2, 2), "the listener's own choice was not applied"
    assert heard == [True, False] and appearance.mode() == "light" and appearance.is_dark() is False


@pytest.mark.parametrize("step", ["setPalette", "setStyleSheet", "setFont", "retheme"])
def test_an_apply_that_fails_part_way_is_made_again_when_the_scheme_before_it_is_chosen(stand_in_app, step):
    """An apply that raises part-way -- at the palette, at the sheet, at the fonts of the widget
    classes or in a widget's own ``retheme()`` -- leaves a look that is half one scheme and half
    the other. The error comes out, nothing is signalled, and the Appearance can be used again.
    Nothing counts as applied while the work is under way, so neither scheme is taken for a repeat
    when it is chosen next: not the one shown before the failure, and not the one that failed.
    Either sets the palette and the sheet again."""
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QWidget
    from core.gui import theming

    class Painted(QWidget):
        def retheme(self): pass

    app = stand_in_app
    # what the failed apply had set when the step refused: palettes, sheets, fonts of widget classes
    set_before_the_step = {"setPalette": (0, 0, 0), "setStyleSheet": (1, 0, 0), "setFont": (1, 1, 0),
                           "retheme": (1, 1, len(CLASSES_WITH_A_FONT))}[step]
    refuser = app                              # whose step refuses: the stand-in, or a widget it lists
    if step == "retheme":
        refuser = Painted()                    # a top-level widget of the stand-in; never shown
        app.tops.append(refuser)

    def chosen(appearance, mode):
        """Choose ``mode``: how often that asked the hints, set a palette and set a sheet."""
        before = (len(app.hints.asked), len(app.palettes), len(app.sheets))
        appearance.set_mode(mode)
        return tuple(now - was for now, was in
                     zip((len(app.hints.asked), len(app.palettes), len(app.sheets)), before))

    def after_a_failed_dark():
        """A new Appearance on the stand-in: light applied, then dark refused at ``step``, which is
        repaired before this returns. What the Appearance signalled comes with it."""
        appearance = theming.Appearance(app)
        signalled = []
        appearance.theme_changed.connect(signalled.append)
        assert chosen(appearance, "light") == (1, 1, 1) and signalled == [False], "the first apply"

        def refuse(*args):
            raise RuntimeError("this step could not be made")

        before = (len(app.palettes), len(app.sheets), len(app.fonts))
        setattr(refuser, step, refuse)
        try:
            with pytest.raises(RuntimeError, match="this step could not be made"):
                appearance.set_mode("dark")
        finally:
            delattr(refuser, step)             # repaired: its own method again
        now = (len(app.palettes), len(app.sheets), len(app.fonts))
        assert tuple(n - was for n, was in zip(now, before)) == set_before_the_step, \
            "the apply did not stop at the step that refused"
        assert signalled == [False], "an apply that failed signalled"
        return appearance, signalled

    try:
        # the scheme shown before the failure, chosen again: it is half gone, so it is made again
        appearance, signalled = after_a_failed_dark()
        assert chosen(appearance, "light") == (1, 1, 1), \
            "the scheme shown before the failure was taken for a repeat (or the Appearance stayed shut)"
        assert signalled == [False, False]
        # the scheme that failed, chosen again once the step works: it was never finished, so it is made
        appearance, signalled = after_a_failed_dark()
        assert chosen(appearance, "dark") == (1, 1, 1), \
            "the scheme that failed was taken for one that was applied, and the half-set look stayed"
        assert signalled == [False, True] and appearance.is_dark() is True
        assert app.sheets[-1] == design.build_qss(True)
        assert app.palettes[-1].color(QPalette.ColorRole.Window).name().upper() == design.tokens(True)["paper"]
    finally:
        del app.tops[:]                        # a widget handed to the stand-in must not wait for it to go


# ── lettering: Inter, its digits of equal width in lists and tables, the book face, the fixed-width face ──
INTER_KINDS = ("small", "body", "strong", "head", "brand", "title")
CLASSES_WITH_A_FONT = ("QAbstractItemView", "QHeaderView", "QMessageBox", "QMenu", "QMenuBar", "QTipLabel",
                       "QStatusBar")
CLASSES_WITH_EQUAL_DIGITS = ("QAbstractItemView", "QHeaderView")      # where numbers stand under one another
FIXED_FACE_CALLERS = {"screens/artifact_screen.py"}       # the record's text; each later caller joins by name


def _tabular():
    return QFont.Tag.fromString("tnum")


@contextlib.contextmanager
def font_and_style_put_back(app):
    """The application's font and style as found, put back when the block ends; the block is handed the
    style's name, to make a fresh style of the same kind from. The font goes back first and the style
    last: a font set with no class name empties the fonts of the widget classes, and on a real display
    it is the change of style that gives each class the system's own font again."""
    assert app.styleSheet() == "", "an earlier test left a style sheet on the application"
    font, style = app.font(), app.style().name()
    assert style.lower() in {key.lower() for key in QStyleFactory.keys()}, f"no style is made from {style!r}"
    try:
        yield style
    finally:
        app.setFont(font)
        app.setStyle(QStyleFactory.create(style))


def test_an_inter_font_has_digits_of_equal_width_only_when_asked_and_a_copy_keeps_them():
    """In the bundled files the feature that gives the digits one width also widens the hyphen, the brackets
    and other marks to a digit's width: switched on everywhere, every compound word reads as if it were spaced.
    So a font carries it only when it is asked for."""
    QApplication.instance() or QApplication([])
    assert design.LETTERING == "Inter" and design.text_scale() == 1.0
    for kind in INTER_KINDS:
        size, _line, weight = design.TYPE[kind]
        font = fonts.inter(kind)
        assert (font.family(), font.pixelSize(), font.weight()) == ("Inter", size, QFont.Weight(weight)), kind
        assert QFontInfo(font).family() == "Inter", "the bundled files are registered on first use"
        assert not font.isFeatureSet(_tabular()), kind
        columns = fonts.inter(kind, tabular=True)
        assert (columns.family(), columns.pixelSize(), columns.weight()) == ("Inter", size, QFont.Weight(weight)), kind
        assert columns.isFeatureSet(_tabular()) and columns.featureValue(_tabular()) == 1, kind
        copy = QFont(columns)
        copy.setPixelSize(size + 3)
        assert copy.isFeatureSet(_tabular()), kind
        assert len({QFontMetricsF(columns).horizontalAdvance(digit) for digit in "0123456789"}) == 1, kind
        assert not fonts.inter(kind).isFeatureSet(_tabular()), "the font with the feature reached the one without"
    assert fonts.inter("small", 600).weight() == QFont.Weight.DemiBold      # a table's header
    assert fonts.inter("head", 400).weight() == QFont.Weight.Normal         # the line that says where a run stands


def test_a_font_follows_the_text_factor_and_a_caller_gets_its_own_copy():
    QApplication.instance() or QApplication([])
    mine = fonts.inter("body")
    mine.setPixelSize(99)
    assert fonts.inter("body").pixelSize() == 14, "a caller's change must not reach the stored font"
    with text_size(1.25):
        assert [fonts.inter(kind).pixelSize() for kind in INTER_KINDS] == [15, 18, 18, 20, 23, 30]
        assert (fonts.fixed().pixelSize(), fonts.book().pixelSize()) == (18, 20)
    with text_size(2.0):
        assert fonts.inter("body").pixelSize() == 28 and fonts.inter("title").pixelSize() == 48
    assert fonts.inter("body").pixelSize() == 14 and fonts.book().pixelSize() == 16


def test_the_book_face_falls_back_to_georgia_and_then_to_inter():
    QApplication.instance() or QApplication([])
    assert design.BOOK_FACES == ("Cambria", "Georgia")
    assert fonts.book_family(["Tahoma", "Georgia", "Inter", "Cambria"]) == "Cambria"
    assert fonts.book_family(["Tahoma", "Inter", "Cambria"]) == "Cambria"       # no Georgia
    assert fonts.book_family(["Tahoma", "Georgia", "Inter"]) == "Georgia"       # no Cambria
    assert fonts.book_family(["Tahoma", "Inter"]) == "Inter"                    # neither
    assert fonts.book_family([]) == "Inter"
    assert fonts.book_family(name for name in ("Georgia",)) == "Georgia"       # any iterable, read once
    font = fonts.book()
    assert font.pixelSize() == design.TYPE["book"][0] == 16, "the book's own size, whichever face is found"
    assert fonts.book(bold=True).bold() and fonts.book(bold=True).pixelSize() == 16
    if not set(design.BOOK_FACES) & set(QFontDatabase.families()):             # off-screen: no book face at all
        assert (font.family(), font.weight(), font.isFeatureSet(_tabular())) == ("Inter", QFont.Weight.Normal, False)
        assert fonts.book(bold=True).weight() == QFont.Weight.DemiBold


def test_console_text_alone_is_set_in_the_fixed_width_face():
    QApplication.instance() or QApplication([])
    gui = REPO / "core" / "gui"
    system_face, callers = [], set()
    for path in sorted(gui.rglob("*.py")):
        rel = path.relative_to(gui).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == "FixedFont" and rel != "fonts.py":
                system_face.append(f"{rel}:{node.lineno}")
            named = (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fixed"
                     and isinstance(node.func.value, ast.Name) and node.func.value.id == "fonts")
            imported = (isinstance(node, ast.ImportFrom) and (node.module or "").endswith("fonts")
                        and any(alias.name == "fixed" for alias in node.names))
            if named or imported:
                callers.add(rel)
    assert system_face == [], f"only fonts.py may ask for the system's fixed-width face: {system_face}"
    assert callers == FIXED_FACE_CALLERS, callers
    font, faces = fonts.fixed(), len(design.FIXED_FACES)
    assert tuple(font.families()[:faces]) == design.FIXED_FACES
    assert font.families()[faces:] == [QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()]
    assert font.styleHint() == QFont.StyleHint.Monospace and font.fixedPitch()
    assert font.pixelSize() == 14


def test_the_class_fonts_are_inter_again_after_a_change_of_style():
    app = QApplication.instance() or QApplication([])
    found = app.font()
    with font_and_style_put_back(app) as style:
        fonts.apply_class_fonts(app)
        for name in CLASSES_WITH_A_FONT:
            font, columns = QApplication.font(name), name in CLASSES_WITH_EQUAL_DIGITS
            assert (font.family(), font.pixelSize(), font.isFeatureSet(_tabular())) == ("Inter", 14, columns), name
        app.setStyle(QStyleFactory.create(style))               # what the first style sheet does too
        assert QApplication.font("QMenu") == app.font(), "a change of style empties the per-class fonts"
        assert not QApplication.font("QAbstractItemView").isFeatureSet(_tabular())
        fonts.apply_class_fonts()                               # no argument: the running application
        menu = QMenu()
        menu.ensurePolished()                                   # a widget takes its class font when polished
        assert (menu.font().family(), menu.font().pixelSize()) == ("Inter", 14)
    assert QApplication.font("QMenu") == app.font() == found


def test_digits_of_equal_width_go_to_lists_tables_and_their_headers_and_nowhere_else(stand_in_app):
    """Numbers stand under one another in lists and tables, so the fonts of those classes and of their headers
    carry the digits of equal width. No other font does, the application's own least of all, which every other
    widget inherits: the same feature widens a hyphen to a digit's width. The stand-in takes the fonts, so
    nothing is set on the application."""
    app = stand_in_app
    assert fonts.load_app_font(app) == "Inter"
    own = app.font()
    assert (own.family(), own.pixelSize(), own.isFeatureSet(_tabular())) == ("Inter", 14, False)
    fonts.apply_class_fonts(app)
    assert set(app.given) == {None, *CLASSES_WITH_A_FONT}
    for name in CLASSES_WITH_A_FONT:
        assert (app.given[name].family(), app.given[name].pixelSize()) == ("Inter", 14), name
    equal = {name for name in CLASSES_WITH_A_FONT if app.given[name].isFeatureSet(_tabular())}
    assert equal == set(CLASSES_WITH_EQUAL_DIGITS), equal


def test_inter_is_handed_to_matplotlib_once():
    from matplotlib import font_manager

    def entries():
        return sum(entry.name == "Inter" for entry in font_manager.fontManager.ttflist)

    assert fonts.register_with_matplotlib() is True
    first = entries()
    assert first >= 2, "the regular and the semibold file"
    assert fonts.register_with_matplotlib() is True and entries() == first, "a second call adds nothing"
    for weight, name in ((400, "Inter-Regular.ttf"), (600, "Inter-SemiBold.ttf")):
        found = font_manager.findfont(font_manager.FontProperties(family="Inter", weight=weight),
                                      fallback_to_default=False)
        assert Path(found).name == name, (weight, found)


def test_the_font_factories_can_be_called_before_the_application_exists():
    code = textwrap.dedent('''
        import sys
        from core.gui import fonts
        early = fonts.inter("body")              # no application yet: no file is registered, nor said to be
        fonts.register_fonts()
        fonts.apply_class_fonts()
        assert (early.pixelSize(), fonts.book().pixelSize(), fonts.fixed().pixelSize()) == (14, 16, 14)
        assert "matplotlib" not in sys.modules, "building a font must not load the figure library"
        from PySide6.QtGui import QFontInfo
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        assert QFontInfo(fonts.inter("body")).family() == "Inter"      # registered now, on first use
        assert QFontInfo(early).family() == "Inter"
    ''')
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          env=dict(os.environ, QT_QPA_PLATFORM="offscreen"), capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.display
def test_the_three_faces_resolve_on_a_real_windows_display():
    """Off-screen there is no system font, so only a real display can say which faces a person sees.
    Builds the application and no window."""
    app = QApplication.instance() or QApplication([])
    if QGuiApplication.platformName() != "windows":
        pytest.skip(f"needs the windows platform, got {QGuiApplication.platformName()!r}")
    for weight in (400, 600):
        assert QFontInfo(fonts.inter("body", weight)).family() == "Inter", weight
    # both files are with the toolkit: with the regular one alone a semibold font still reads "Inter"
    assert {"Regular", "SemiBold"} <= set(QFontDatabase.styles("Inter"))
    # The digits of equal width come with a hyphen as wide as a digit, so a font has them only when asked.
    plain = QFontMetricsF(fonts.inter("body"))
    assert plain.horizontalAdvance("-") < plain.horizontalAdvance("0"), "a hyphen as wide as a digit"
    columns = QFontMetricsF(fonts.inter("body", tabular=True))
    assert columns.horizontalAdvance("-") == columns.horizontalAdvance("0")
    assert len({columns.horizontalAdvance(digit) for digit in "0123456789"}) == 1
    wanted = fonts.book_family(QFontDatabase.families())
    assert QFontInfo(fonts.book()).family() == wanted
    assert fonts.book().pixelSize() == 16 and fonts.book(bold=True).bold()
    console = QFontInfo(fonts.fixed())
    assert console.fixedPitch() and console.pixelSize() == 14
    assert console.family() in (*design.FIXED_FACES,
                                QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family())
    with font_and_style_put_back(app) as style:
        fonts.load_app_font(app)
        assert not app.font().isFeatureSet(_tabular())
        app.setStyle(QStyleFactory.create(style))
        assert QApplication.font("QMenu").family() != "Inter", "Windows' own font is back on a menu"
        fonts.apply_class_fonts(app)
        assert {QApplication.font(name).family() for name in CLASSES_WITH_A_FONT} == {"Inter"}
        assert {name for name in CLASSES_WITH_A_FONT if QApplication.font(name).isFeatureSet(_tabular())} == set(
            CLASSES_WITH_EQUAL_DIGITS)


# ── the palette and the style sheet from the table; the order of an apply ────────────────────────
def _rules(qss):
    """Every rule of a built sheet, in order: (its selectors, its declarations)."""
    rules = []
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", qss):
        declarations = {}
        for part in body.split(";"):
            if ":" in part:
                name, value = part.split(":", 1)
                declarations[name.strip()] = " ".join(value.split())
        rules.append(([" ".join(s.split()) for s in selectors.split(",")], declarations))
    return rules


def _classes(selector):
    """The widget classes a selector names: the word that opens it and each word after a space."""
    return re.findall(r"(?:^|\s)([A-Za-z]+)", selector)


def _keeps_the_selector_rule(selector):
    """A widget class, narrowed only by one of the four properties: no object name, no wildcard."""
    return ("#" not in selector and "*" not in selector and "QWidget" not in _classes(selector)
            and set(re.findall(r"\[(\w+)", selector)) <= {"cls", "state", "current", "small"})


def test_tokens_gives_the_whole_table_for_one_theme():
    import inspect
    from core.gui import design
    light, dark = design.tokens(False), design.tokens(True)
    assert light == {name: pair[0] for name, pair in design.COLOURS.items()}
    assert dark == {name: pair[1] for name, pair in design.COLOURS.items()}
    assert len(light) == 32 and (light["accent"], dark["accent"]) == ("#0072B2", "#56B4E9")
    assert design.tokens(False) is not light, "a caller's edit must not reach the table"
    for function in (design.tokens, design.build_palette, design.build_qss):
        assert list(inspect.signature(function).parameters) == ["dark"], function.__name__
    assert not [name for name in ("LIGHT", "DARK", "system_accent") if hasattr(design, name)]


def test_the_palette_takes_every_role_from_the_table():
    from PySide6.QtGui import QPalette
    from core.gui import design
    Role, Group = QPalette.ColorRole, QPalette.ColorGroup
    roles = {Role.Window: "paper", Role.WindowText: "ink", Role.Base: "field", Role.AlternateBase: "margin",
             Role.Text: "ink", Role.Button: "paper", Role.ButtonText: "ink", Role.BrightText: "on_accent",
             Role.ToolTipBase: "raised", Role.ToolTipText: "ink", Role.PlaceholderText: "ink_3",
             Role.Highlight: "accent", Role.HighlightedText: "on_accent", Role.Link: "accent",
             Role.LinkVisited: "accent", Role.Mid: "rule_light", Role.Midlight: "margin", Role.Light: "paper",
             Role.Dark: "field_line", Role.Shadow: "ink", Role.Accent: "accent"}
    disabled = {Role.WindowText: "disabled_ink", Role.Text: "disabled_ink", Role.ButtonText: "disabled_ink",
                Role.PlaceholderText: "disabled_ink", Role.Base: "disabled_fill", Role.Button: "disabled_fill",
                Role.Highlight: "disabled_line"}
    for dark in (False, True):
        palette, t = design.build_palette(dark), design.tokens(dark)
        for role, name in roles.items():
            for group in (Group.Active, Group.Inactive):
                assert palette.color(group, role).name().upper() == t[name], (dark, group, role, name)
            assert palette.color(Group.Disabled, role).name().upper() == t[disabled.get(role, name)], (dark, role)


def test_the_sheet_keeps_its_rules_and_only_the_named_interim_selectors_break_them():
    from core.gui import design
    for dark in (False, True):
        qss = design.build_qss(dark)
        for banned in ("$", "url(", "font-family", "letter-spacing", ":default", "::indicator", "*"):
            assert banned not in qss, (dark, banned)
        rules = _rules(qss)
        assert len(rules) == 86, len(rules)
        first = next(i for i, (selectors, _) in enumerate(rules) if selectors[0] in design.INTERIM_SELECTORS)
        final, interim = rules[:first], rules[first:]
        assert [s for selectors, _ in interim for s in selectors] == list(design.INTERIM_SELECTORS)
        for selectors, declarations in final:
            for selector in selectors:
                assert _keeps_the_selector_rule(selector), (dark, selector)
                assert selector not in design.INTERIM_SELECTORS, selector
            assert not [name for name in declarations if name.startswith("qproperty-")], selectors
        for selectors, _ in interim:                     # an old widget's rule still names a class of its own
            assert not [s for s in selectors if "QWidget" in _classes(s)], selectors
    assert len(set(design.INTERIM_SELECTORS)) == len(design.INTERIM_SELECTORS) == 44


def test_no_pseudo_state_sets_a_weight_or_a_size_and_words_take_lettering_colours():
    from core.gui import design
    for dark in (False, True):
        t = design.tokens(dark)
        lettering = {t[name] for name in ("ink", "ink_2", "ink_3", "disabled_ink", "on_accent", "accent")}
        for selectors, declarations in _rules(design.build_qss(dark)):
            if any(re.search(r"(?<!:):(?!:)", selector) for selector in selectors):
                assert not {"font", "font-size", "font-weight"} & set(declarations), selectors
            for name in ("color", "selection-color"):
                if name in declarations:
                    assert declarations[name] in lettering, (dark, selectors, name, declarations[name])


def test_the_sheets_lengths_follow_the_text_factor_and_its_line_weights_do_not():
    from core.gui import design
    with text_size(1.0):
        one = {tuple(selectors): d for selectors, d in _rules(design.build_qss(False))}
    with text_size(2.0):
        two = {tuple(selectors): d for selectors, d in _rules(design.build_qss(False))}
        sizes = design._sheet_sizes()
    assert len(sizes) == 31 and all(name.startswith("px_") for name in sizes) and not set(sizes) & set(design.COLOURS)
    button, field, bad = ("QPushButton",), ("QLineEdit", "QComboBox"), ('QLineEdit[state="error"]',)
    title = ('QLabel[type="title"]',)
    assert (one[button]["min-height"], two[button]["min-height"]) == ("32px", "66px")
    assert (one[button]["padding"], two[button]["padding"]) == ("0px 16px", "0px 32px")
    assert (one[field]["min-height"], two[field]["min-height"]) == ("30px", "62px")
    assert (one[bad]["min-height"], one[bad]["padding"]) == ("28px", "0px 7px")
    assert (two[bad]["min-height"], two[bad]["padding"]) == ("60px", "0px 15px")
    assert (one[title]["font-size"], two[title]["font-size"]) == ("24px", "48px")
    for built in (one, two):
        assert built[button]["border"] == "1px solid #6B6B6B" and built[button]["border-radius"] == "3px"
        assert built[bad]["border"] == "2px solid #C24E00"
        assert built[("QScrollBar:vertical",)]["width"] == "8px"
        assert built[("QScrollBar::handle:horizontal",)]["min-width"] == "28px"


def _help_badge_rule(dark):
    """What the built sheet declares for the small round help button."""
    rules = {tuple(selectors): declarations for selectors, declarations in _rules(design.build_qss(dark))}
    return rules[("QToolButton#helpBadge",)]


def test_the_help_badges_ring_is_the_faint_rule_and_not_the_outline_of_a_control():
    """The badge's mark lies on its ring. In the outline colour of a control the ring is nearly the mark's own
    tone and the mark cannot be read; in the faint rule's colour the mark reads on its own."""
    for dark in (False, True):
        rule = _help_badge_rule(dark)
        assert rule["border"] == "1px solid " + design.hex_colour("rule_light", dark), dark
        assert design.hex_colour("field_line", dark) not in rule["border"], dark
        assert rule["color"] == design.hex_colour("ink_2", dark), dark


def test_the_help_badge_is_round_at_every_text_size():
    """The badge is as wide as the text makes it, so its corner is half that width: a corner of a fixed size
    leaves a soft square once the text is larger."""
    for factor, side, corner in ((1.0, 16, "8px"), (1.5, 24, "12px")):
        with text_size(factor):
            assert design.scaled(16) == side, "the side the badge gives itself"
            assert _help_badge_rule(False)["border-radius"] == _help_badge_rule(True)["border-radius"] == corner, factor


def test_a_fields_hint_text_is_the_hint_colour_in_both_themes():
    """A sheet rule that gives a field its text colour makes the toolkit work the field's hint colour out of that
    colour, at half strength; the palette's own hint colour is then never read. So every rule that gives a field a
    colour names the colour of its hint too, and a field's hint is the table's at full strength, disabled or not.
    A drop-down is read through the line edit it has when it can be typed in, which paints its hint from this
    role; one that cannot be typed in paints its hint in the text colour, whatever the role holds. The sheet and
    the palette are set on each field itself, never on the application."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QComboBox, QLineEdit, QPlainTextEdit, QTextEdit, QWidget
    from core.gui import design
    QApplication.instance() or QApplication([])
    fields = ("QLineEdit", "QComboBox", "QPlainTextEdit", "QTextEdit")
    hint, disabled = QPalette.ColorRole.PlaceholderText, QPalette.ColorGroup.Disabled

    def read(own, *group):
        return own.color(*group, hint).name(QColor.NameFormat.HexArgb).upper()

    for dark in (False, True):
        t, qss, palette = design.tokens(dark), design.build_qss(dark), design.build_palette(dark)
        assert t["ink_3"] == design.COLOURS["ink_3"][1 if dark else 0]
        holder = QWidget()
        # the sheet has no disabled rule for a text box, so a disabled one keeps the hint colour of its plain rule
        for make, when_disabled in ((QLineEdit, "disabled_ink"), (QPlainTextEdit, "ink_3"), (QTextEdit, "ink_3"),
                                    (QComboBox, "disabled_ink")):
            box = make(holder)
            if make is QComboBox:
                box.setEditable(True)
            box.setPalette(palette)
            box.setStyleSheet(qss)
            box.ensurePolished()
            field = box.lineEdit() if make is QComboBox else box
            field.setPlaceholderText("a hint")
            field.ensurePolished()
            own = field.palette()
            assert read(own) == "#FF" + t["ink_3"][1:], (dark, make.__name__)
            assert read(own, disabled) == "#FF" + t[when_disabled][1:], (dark, make.__name__, "disabled")
        for selectors, declarations in _rules(qss):
            for selector in selectors:
                # the class a rule colours is the last one its selector names: a drop-down's list is not a field
                if re.match(r"[A-Za-z]+", selector.split()[-1]).group() in fields and "color" in declarations:
                    assert "placeholder-text-color" in declarations, (dark, selector)


def test_a_list_a_text_viewer_and_a_scrolling_area_are_framed_from_the_table():
    """Left to the toolkit, the frame of a list, of a text viewer and of a scrolling area is the page's colour made
    darker: a pale line on the light page and next to none on the dark one. A list and a viewer are boxes one
    picks or reads in, so they take a field's outline and ground; a bare scrolling area takes the quiet
    separator; and the square where two scroll bars meet is the bars' own track. The list a drop-down opens has
    a rule of its own and keeps all of it, its square corners too: the rule for lists names the list widget, so
    it does not reach that one. The palette and the sheet are set on one holder, which is never shown."""
    from PySide6.QtWidgets import QListWidget, QScrollArea, QTextBrowser
    QApplication.instance() or QApplication([])
    width, height = 120, 80

    def painted(widget):
        widget.setFixedSize(width, height)
        widget.ensurePolished()
        picture = _image(width, height, 1.0)
        widget.render(picture)
        return picture

    for dark in (False, True):
        outline, quiet, ground = (design.hex_colour(name, dark) for name in ("field_line", "rule_light", "field"))
        qss = design.build_qss(dark)
        holder = QWidget()
        holder.setPalette(design.build_palette(dark))
        holder.setStyleSheet(qss)
        a_list, viewer, area, box = QListWidget(holder), QTextBrowser(holder), QScrollArea(holder), QComboBox(holder)
        area.setWidget(QWidget())
        for boxed in (a_list, viewer):               # the middle of its top edge, and its middle
            picture = painted(boxed)
            assert (_at(picture, width / 2, 0), _at(picture, width / 2, height / 2)) == (outline, ground), (
                dark, type(boxed).__name__)
        assert _at(painted(area), width / 2, 0) == quiet, dark
        opened = painted(box.view())                 # the drop-down's own list, with no rows in it; never opened
        corners = [_at(opened, x, y) for x in (0, width - 1) for y in (0, height - 1)]
        assert corners == [outline] * 4, (dark, "the list a drop-down opens lost its square corners")
        assert (_at(opened, width / 2, 0), _at(opened, width / 2, height / 2)) == (outline, ground), dark
        rules = {tuple(selectors): declarations for selectors, declarations in _rules(qss)}
        assert rules[("QAbstractScrollArea::corner",)]["background"] == rules[("QScrollBar:vertical",)]["background"]


def test_an_apply_runs_its_steps_in_one_order_and_stops_when_nothing_changed(monkeypatch):
    from PySide6.QtCore import QObject, Qt, Signal
    from PySide6.QtWidgets import QWidget
    from core.gui import design, fonts, style, theming
    QApplication.instance() or QApplication([])
    calls = []

    class Hints(QObject):
        colorSchemeChanged = Signal(object)
        def setColorScheme(self, scheme): calls.append("scheme")
        def colorScheme(self): return Qt.ColorScheme.Unknown

    class Painted(QWidget):
        def retheme(self): calls.append("retheme")
        def update(self): calls.append("update")

    class StandIn(QObject):                     # what an apply asks of the application, and no more
        def __init__(self, tops):
            super().__init__()
            self.hints, self.tops = Hints(), tops
        def styleHints(self): return self.hints
        def setPalette(self, palette): calls.append("palette")
        def setStyleSheet(self, text): calls.append("sheet")
        def topLevelWidgets(self): return list(self.tops)

    top = Painted()
    child = Painted(top)
    stand_in = StandIn([top])
    monkeypatch.setattr(design, "set_dark", lambda dark: calls.append(("set_dark", dark)))
    monkeypatch.setattr(fonts, "apply_class_fonts", lambda app=None: calls.append("class_fonts"))
    monkeypatch.setattr(style, "refresh_focus", lambda: calls.append("ring"))
    monkeypatch.setattr(theming, "_ACTIVE", theming._ACTIVE)     # the constructor rebinds it
    try:
        appearance = theming.Appearance(stand_in)
        appearance.theme_changed.connect(lambda dark: calls.append(("theme_changed", dark)))
        appearance.set_mode("dark")
        assert calls == ["scheme", ("set_dark", True), "palette", "sheet", "class_fonts",
                         "retheme", "retheme", "update", "ring", ("theme_changed", True)]
        del calls[:]
        appearance.set_mode("dark")
        assert calls == ["scheme"], "an apply that changes nothing stops after the colour scheme"
    finally:
        del stand_in.tops[:]     # the stand-in and its Appearance hold each other: the widget must not wait for them


def test_an_apply_rethemes_each_widget_once_also_in_a_window_that_has_a_parent(stand_in_app):
    """A dialog opened over a window is a top-level widget of the application and, having the window for its
    parent, one of the window's children as well. An apply reaches it, and what it holds, once: in its own turn.
    What the dialog holds stands inside a plain container, so a walk over a window's own children alone, which
    would miss it, cannot pass."""
    from PySide6.QtWidgets import QDialog, QWidget
    from core.gui import theming
    counts = {}

    def counted(widget, name):
        widget.setObjectName(name)
        counts[name] = 0
        return widget

    class Painted(QWidget):
        def retheme(self): counts[self.objectName()] += 1

    class Asked(QDialog):
        def retheme(self): counts[self.objectName()] += 1

    window = counted(Painted(), "the window")
    held = counted(Painted(window), "the window's child")
    dialog = counted(Asked(window), "the dialog")
    asked = counted(Painted(QWidget(dialog)), "inside the dialog")       # two levels below its window
    assert asked.parent().parent() is dialog and asked.window() is dialog
    assert dialog.parent() is window and dialog.isWindow() and dialog in window.findChildren(QWidget)
    stand_in_app.tops.extend([window, dialog])       # as the application lists them: both are top-level widgets
    try:
        appearance = theming.Appearance(stand_in_app)
        appearance.set_mode("dark")
        assert len(stand_in_app.sheets) == 1, "the apply did not go on"
        assert counts == {"the window": 1, "the window's child": 1, "the dialog": 1, "inside the dialog": 1}
    finally:
        del stand_in_app.tops[:]                     # the widgets must not wait for the stand-in to go


def test_an_apply_tells_the_looks_table_and_sets_the_class_fonts_on_its_own_application(stand_in_app):
    from PySide6.QtGui import QPalette
    from core.gui import design, theming
    menu_font = QApplication.font("QMenu")
    appearance = theming.Appearance(stand_in_app)
    appearance.set_mode("dark")
    assert design.is_dark() is True, "an apply must tell the look's table which theme is in force"
    assert len(stand_in_app.fonts) == 7 and None not in stand_in_app.fonts, stand_in_app.fonts
    assert QApplication.font("QMenu") == menu_font, "the class fonts were set on the real application"
    assert stand_in_app.sheets == [design.build_qss(True)]
    assert stand_in_app.palettes[0].color(QPalette.ColorRole.Window).name().upper() == design.tokens(True)["paper"]
    appearance.set_mode("light")
    assert design.is_dark() is False and len(stand_in_app.fonts) == 14


def _messages_under(qss):
    """What Qt says while one widget of every kind the sheet names is polished and painted under ``qss``. The
    sheet is set on their holder alone, which is never shown: nothing is set on the application. The holder
    and all it holds are gone when this returns: its last name goes with the call."""
    from PySide6 import QtWidgets as W
    from PySide6.QtCore import qInstallMessageHandler

    QApplication.instance() or QApplication([])
    said = []
    previous = qInstallMessageHandler(lambda kind, context, text: said.append(text))
    try:
        holder = W.QWidget()
        made = []

        def add(make, name="", parent=None, **properties):
            widget = make(parent or holder)
            widget.setObjectName(name)
            for prop, value in properties.items():
                widget.setProperty(prop, value)
            made.append(widget)
            return widget

        for kind in ("title", "head", "now", "strong", "small", "quiet", "book2"):
            add(W.QLabel, cls=kind)
        for kind in ("title", "subtitle", "heading", "caption"):
            add(W.QLabel, type=kind)
        add(W.QPushButton)
        add(W.QPushButton).setEnabled(False)
        for kind in ("primary", "danger", "nav", "fold", "link"):
            add(W.QPushButton, cls=kind)
        add(W.QPushButton, cls="nav", small=True, current=True)
        add(W.QPushButton, accent=True)
        add(W.QPushButton, "iconButton")
        add(W.QLineEdit)
        add(W.QLineEdit, state="error")
        add(W.QLineEdit).setEnabled(False)
        add(W.QComboBox).addItems(["one", "two"])
        for make in (W.QPlainTextEdit, W.QCheckBox, W.QRadioButton, W.QTableView, W.QProgressBar, W.QScrollBar,
                     W.QGroupBox, W.QMenu, W.QListWidget, W.QTextBrowser, W.QScrollArea):
            add(make)
        tree = add(W.QTreeWidget)
        tree.setHeaderLabels(["one", "two"])
        W.QTreeWidgetItem(tree, ["row", "row"])
        tabs = add(W.QTabWidget)
        tabs.addTab(W.QLabel("page"), "one")
        for name in ("", "navBack", "navSettings", "navRunning", "helpBadge"):
            add(W.QToolButton, name)
        add(W.QPushButton, parent=add(W.QDialog, cls="ask"), cls="danger")
        holder.setStyleSheet(qss)
        for widget in (holder, *made):
            widget.ensurePolished()
            widget.grab()
    finally:
        qInstallMessageHandler(previous)
    return said


def test_qt_takes_both_sheets_without_a_warning():
    from core.gui import design
    quiet = set(_messages_under(""))
    for factor in (1.0, 2.25):                   # the ordinary text size, and the largest factor the look takes
        for dark in (False, True):
            with text_size(factor):
                qss = design.build_qss(dark)
            said = [text for text in _messages_under(qss) if text not in quiet]
            assert said == [], (factor, dark, said)
    # the same probe must hear a sheet Qt cannot read and a property it does not know
    assert any("Could not parse" in text for text in _messages_under("QPushButton { color: red"))
    assert any("Unknown property" in text for text in _messages_under("QPushButton { colour: red; }"))


def _colour_strings(tree):
    """The string literals of a parsed module that hold a six-digit colour, as (line, node)."""
    return [(node.lineno, node) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and re.search(r"#[0-9A-Fa-f]{6}\b", node.value)]


def test_colours_are_written_in_the_looks_table_alone_and_a_sheet_is_set_by_the_appearance_alone():
    gui = Path(__file__).resolve().parents[1] / "core" / "gui"
    probe = ast.parse('a = "#12AB9F"\nb = f"1px solid #12ab9f"\nc = "no colour #12AB"\nw.setStyleSheet(a)\n')
    assert [line for line, _ in _colour_strings(probe)] == [1, 2], "the scan must see a colour, and only a colour"
    colours, sheets = [], []
    for path in sorted(gui.rglob("*.py")):
        rel = path.relative_to(gui).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if rel == "design.py":
            table = next(node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                         and "COLOURS" in {getattr(target, "id", "") for target in
                                           (node.targets if isinstance(node, ast.Assign) else [node.target])})
            inside = {id(node) for node in ast.walk(table)}
            colours += [f"{rel}:{line}" for line, node in _colour_strings(tree) if id(node) not in inside]
        else:
            colours += [f"{rel}:{line}" for line, _ in _colour_strings(tree)]
        if rel != "theming.py":
            sheets += [f"{rel}:{node.lineno}" for node in ast.walk(tree)
                       if isinstance(node, ast.Attribute) and node.attr == "setStyleSheet"]
    assert colours == [], f"a colour written outside the look's table: {colours}"
    assert sheets == [], f"a style sheet set outside the appearance: {sheets}"


# ── the small style: the indicators it paints, the focus ring and where the ring shows ───────────
def _image(width, height, ratio):
    """A clear picture of that many logical pixels at a display scaling, its size rounded up."""
    picture = QImage(math.ceil(width * ratio), math.ceil(height * ratio), QImage.Format_ARGB32_Premultiplied)
    picture.setDevicePixelRatio(ratio)
    picture.fill(Qt.transparent)
    return picture


def _at(picture, x, y):
    """The colour, as the look's table writes it, of the device pixel under a logical point."""
    ratio = picture.devicePixelRatio()
    return picture.pixelColor(int(x * ratio), int(y * ratio)).name().upper()


def test_the_tick_box_and_the_radio_dot_are_painted_from_the_table():
    from core.gui import design, paint

    def drawn(kind, on, enabled, ratio):
        picture = _image(18, 18, ratio)
        painter = QPainter(picture)
        try:
            paint.paint_indicator(painter, kind, QRectF(0, 0, 18, 18), on=on, enabled=enabled)
        finally:
            painter.end()                    # a raise under a live painter would end the run, not this test
        return picture

    def edge(picture):                       # the first device pixel of the middle row: the box's outline
        return picture.pixelColor(0, picture.height() // 2).name().upper()

    name = design.hex_colour
    for dark in (False, True):
        with theme(dark):
            for ratio in (1.0, 1.25, 1.5, 2.0):
                where = (dark, ratio)
                empty = drawn("check", False, True, ratio)
                assert (_at(empty, 9, 9), edge(empty)) == (name("field"), name("field_line")), where
                ticked = drawn("check", True, True, ratio)
                assert (_at(ticked, 2.5, 9), _at(ticked, 7.6, 11.5)) == (name("accent"), name("on_accent")), where
                greyed = drawn("check", True, False, ratio)           # it keeps its tick and loses the accent
                assert (_at(greyed, 2.5, 9), _at(greyed, 7.6, 11.5), edge(greyed)) == (
                    name("disabled_fill"), name("disabled_ink"), name("disabled_line")), where
                dot = drawn("radio", True, True, ratio)
                assert (_at(dot, 9, 2.5), _at(dot, 9, 9)) == (name("accent"), name("on_accent")), where
                assert _at(drawn("radio", False, True, ratio), 9, 9) == name("field"), where
                off = drawn("radio", True, False, ratio)
                assert (_at(off, 9, 2.5), _at(off, 9, 9)) == (name("disabled_fill"), name("disabled_ink")), where
                assert drawn("neither", True, True, ratio) == _image(18, 18, ratio), "an unknown kind painted"


def test_the_small_style_sizes_the_indicators_and_paints_them_itself():
    from core.gui import design, style
    QApplication.instance() or QApplication([])
    worn = style.PrismStyle()

    def fill(sheet, cls, part, x, y, enabled=True):    # a pixel inside a set indicator, under one holder's own sheet
        holder = QWidget()
        holder.setStyleSheet(sheet)
        box = cls("words", holder)
        box.setStyle(worn)                   # on this one widget, never on the application
        box.setChecked(True)
        box.setEnabled(enabled)
        box.ensurePolished()
        box.resize(box.sizeHint())
        option = QStyleOptionButton()
        option.initFrom(box)
        where = box.style().subElementRect(part, option, box)
        assert (where.width(), where.height()) == (18, 18)
        picture = _image(box.width(), box.height(), 1.5)
        box.render(picture)
        return _at(picture, where.x() + x, where.y() + y)

    with text_size(1.0):
        for metric in (QStyle.PM_IndicatorWidth, QStyle.PM_IndicatorHeight,
                       QStyle.PM_ExclusiveIndicatorWidth, QStyle.PM_ExclusiveIndicatorHeight):
            assert worn.pixelMetric(metric) == 18
        for cls, part, x, y in ((QCheckBox, QStyle.SE_CheckBoxIndicator, 2.5, 9),
                                (QRadioButton, QStyle.SE_RadioButtonIndicator, 9, 2.5)):
            sheets = ("", design.build_qss(False), design.build_qss(True))     # none, and the whole sheet twice
            assert [fill(s, cls, part, x, y) for s in sheets] == [design.hex_colour("accent")] * 3, cls.__name__
            assert [fill(s, cls, part, x, y, enabled=False) for s in sheets] == [
                design.hex_colour("disabled_fill")] * 3, cls.__name__          # greyed: set, and no accent
            spoiled = cls.__name__ + "::indicator { border: 1px solid red; }"
            assert fill(spoiled, cls, part, x, y) != design.hex_colour("accent"), "the lines above prove nothing"
    with text_size(1.25):
        assert worn.pixelMetric(QStyle.PM_IndicatorWidth) == 23        # the box grows with the lettering
        assert worn.pixelMetric(QStyle.PM_FocusFrameHMargin) == 4      # the ring and its gap do not
        assert worn.pixelMetric(QStyle.PM_FocusFrameVMargin) == 4


def test_the_toolkits_dotted_focus_mark_is_never_drawn():
    from core.gui import style
    QApplication.instance() or QApplication([])
    option = QStyleOptionFocusRect()
    option.rect = QRect(2, 2, 56, 20)
    option.state = QStyle.State_Enabled | QStyle.State_HasFocus | QStyle.State_KeyboardFocusChange

    def drawn(by):
        picture = _image(60, 24, 1.0)
        painter = QPainter(picture)
        try:
            by.drawPrimitive(QStyle.PE_FrameFocusRect, option, painter, None)
        finally:
            painter.end()
        return picture

    assert drawn(QStyleFactory.create("Fusion")) != _image(60, 24, 1.0), "the stock style draws no mark to leave out"
    assert drawn(style.PrismStyle()) == _image(60, 24, 1.0)


def test_the_wheel_over_a_closed_drop_down_leaves_its_choice():
    from core.gui import style
    QApplication.instance() or QApplication([])

    def one_notch(worn):
        box = QComboBox()
        box.addItems(["a", "b", "c"])
        box.setStyle(worn)
        turn = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120), Qt.NoButton,
                           Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(box, turn)
        return box.currentIndex(), turn.isAccepted()

    assert one_notch(QStyleFactory.create("Fusion")) == (1, True)      # the toolkit's own answer: the choice moves
    assert one_notch(style.PrismStyle()) == (0, False)                 # not taken: left for the page behind it


def test_a_class_set_after_the_first_polish_takes_effect():
    from core.gui import style
    QApplication.instance() or QApplication([])
    holder = QWidget()
    holder.setStyleSheet('QPushButton[cls="primary"] { font-weight: 600; }')     # this one widget's own sheet
    button = QPushButton("Start", holder)
    button.ensurePolished()
    assert button.font().weight() == 400
    button.setProperty("cls", "primary")
    assert button.font().weight() == 400, "a property alone took effect: this test proves nothing"
    assert style.set_class(button, cls="primary") is button and button.font().weight() == 600
    assert style.set_class(button, cls="").font().weight() == 400
    assert style.set_class(QPushButton("never polished"), cls="primary").property("cls") == "primary"


@pytest.fixture
def ring_room():
    """A window that is never shown, four controls in it, and one follower whose ring wears its own
    copy of the small style. Never shown, so no real keyboard focus and no activation reach the one
    application the suites share; the follower comes off that application before the window goes."""
    from core.gui import style
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.resize(420, 300)
    button, field = QPushButton("Check again", window), QLineEdit(window)
    row, box = QPushButton("A row", window), QPlainTextEdit(window)
    button.setGeometry(40, 40, 120, 34)
    field.setGeometry(40, 100, 120, 32)
    row.setGeometry(0, 200, 420, 36)
    row.setProperty("ring", "inset")
    box.setGeometry(200, 100, 200, 80)
    follows = style.FocusFollower(app, style.PrismStyle())
    try:
        yield SimpleNamespace(window=window, button=button, field=field, row=row, box=box, follows=follows)
    finally:
        follows.stop()                       # first: the ring holds its control by address


def _moved_on(widget, reason):
    """The focus arrives with no press of its own, as when the toolkit moves it on from a greyed control."""
    QApplication.sendEvent(widget, QFocusEvent(QEvent.FocusIn, reason))


def _arrive(widget, reason):
    """The focus arrives as a person moves it: first the press, which the application alone hears (a key for
    the keyboard's three reasons, a mouse button for a click), then the focus itself."""
    if reason in (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason):
        QApplication.sendEvent(QObject(), QKeyEvent(QEvent.KeyPress, Qt.Key_Tab, Qt.NoModifier))
    elif reason == Qt.MouseFocusReason:
        QApplication.sendEvent(QObject(), QMouseEvent(QEvent.MouseButtonPress, QPointF(1, 1), QPointF(1, 1),
                                                      Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    _moved_on(widget, reason)


def _leave(widget, reason):
    QApplication.sendEvent(widget, QFocusEvent(QEvent.FocusOut, reason))


def _ringed(room, widget):
    ring = room.follows.ring()
    return ring is not None and ring.widget() is widget and ring.isVisibleTo(room.window)


def test_the_ring_shows_after_the_keyboard_and_after_a_click_only_in_a_field_one_types_in(ring_room):
    r = ring_room

    def comes(widget, reason, arrive=_arrive):       # the focus arrives, the ring is looked for, the focus leaves
        arrive(widget, reason)
        shown = _ringed(r, widget)
        _leave(widget, reason)
        return shown and r.follows.target() is None and not _ringed(r, widget)

    assert not comes(r.button, Qt.TabFocusReason, _moved_on)   # nothing pressed yet: no key moved it
    assert not comes(r.button, Qt.MouseFocusReason)
    assert comes(r.field, Qt.MouseFocusReason)
    for reason in (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason):
        assert comes(r.button, reason), reason
    assert comes(r.button, Qt.ActiveWindowFocusReason)         # the window coming back keeps what it had
    assert not comes(r.button, Qt.MouseFocusReason)            # the mouse again: the keyboard's turn is over
    assert not comes(r.button, Qt.ActiveWindowFocusReason)
    # A clicked control that is then greyed or hidden loses the focus to the next one, and the toolkit gives
    # that move the Tab key's reason. No key was pressed, so no ring, except in a field one types in.
    for reason in (Qt.TabFocusReason, Qt.BacktabFocusReason):
        assert not comes(r.button, reason, _moved_on), reason
    assert comes(r.field, Qt.TabFocusReason, _moved_on)
    assert comes(r.button, Qt.ShortcutFocusReason, _moved_on)  # a shortcut is a key, whatever was pressed before
    assert not comes(r.button, Qt.MouseFocusReason)
    assert comes(r.button, Qt.TabFocusReason)                  # and the Tab key itself still shows it


def test_in_a_message_box_the_focus_is_shown_whatever_brought_it(ring_room):
    """A message box of the old layout marks the answer the Enter key gives by nothing but the focus, and it
    opens with the focus already on that answer: no key and no click brought it there. So inside such a box
    the ring shows wherever the focus is. A plain dialog gets no such favour."""
    from PySide6.QtWidgets import QDialog, QMessageBox
    r = ring_room
    box = QMessageBox(r.window)              # never shown and never run; it goes with the window it belongs to
    start, cancel = box.addButton("Start", QMessageBox.AcceptRole), box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(start)
    in_box = SimpleNamespace(follows=r.follows, window=box)
    _moved_on(start, Qt.ActiveWindowFocusReason)             # as the box opens: nothing was pressed
    assert r.follows.target() is start and _ringed(in_box, start)
    _arrive(cancel, Qt.MouseFocusReason)                     # a click on the other answer: the ring goes with it
    assert r.follows.target() is cancel and _ringed(in_box, cancel)
    _leave(cancel, Qt.MouseFocusReason)
    assert r.follows.target() is None and not _ringed(in_box, cancel)
    plain = QDialog(r.window)                                # the same two arrivals in a plain dialog: no ring
    first, second = QPushButton("Start", plain), QPushButton("Cancel", plain)
    _moved_on(first, Qt.ActiveWindowFocusReason)
    assert r.follows.target() is None
    _arrive(second, Qt.MouseFocusReason)
    assert r.follows.target() is None
    # Whose turn it is goes on being kept as everywhere: a click inside a box ends the keyboard's.
    _arrive(r.button, Qt.TabFocusReason)
    assert _ringed(r, r.button)
    _leave(r.button, Qt.TabFocusReason)
    _arrive(cancel, Qt.MouseFocusReason)
    assert _ringed(in_box, cancel)
    _leave(cancel, Qt.MouseFocusReason)
    _moved_on(r.button, Qt.ActiveWindowFocusReason)          # the window comes back: no ring, the mouse was last
    assert r.follows.target() is None and not _ringed(r, r.button)


def test_the_ring_stands_four_pixels_outside_a_control_and_just_inside_one_that_fills_its_place(ring_room):
    r = ring_room
    _arrive(r.button, Qt.TabFocusReason)
    ring = r.follows.ring()
    assert ring.geometry() == r.button.geometry().adjusted(-4, -4, 4, 4)
    assert ring.parentWidget() is r.window and not ring.inset
    _leave(r.button, Qt.TabFocusReason)
    for inside in (r.row, r.box):            # one asks by its property; one is a widget that scrolls
        _arrive(inside, Qt.TabFocusReason)
        assert r.follows.ring().inset and r.follows.ring().geometry() == inside.geometry()
        _leave(inside, Qt.TabFocusReason)
    _arrive(r.button, Qt.TabFocusReason)     # and outside again: the form is decided at every arrival
    assert r.follows.ring().geometry() == r.button.geometry().adjusted(-4, -4, 4, 4)


def test_the_ring_is_put_back_after_a_change_of_style_and_goes_with_its_control(ring_room):
    r = ring_room
    _arrive(r.button, Qt.TabFocusReason)
    QApplication.sendEvent(r.button, QEvent(QEvent.StyleChange))       # what every change of theme sends it
    assert not _ringed(r, r.button), "the toolkit no longer hides its focus frame at a change of style"
    r.follows.refresh()
    assert _ringed(r, r.button)
    r.button.hide()
    assert r.follows.target() is None and not _ringed(r, r.button)


class _OwnSign(QPushButton):
    """A control that shows keyboard focus itself, as a table does for its current row."""
    shown = None

    def set_focus_shown(self, on):
        self.shown = on


def test_the_ring_is_left_off_a_control_that_shows_focus_itself_and_off_an_opened_list(ring_room):
    r = ring_room
    own = _OwnSign("a table", r.window)
    _arrive(own, Qt.TabFocusReason)
    assert own.shown is True and r.follows.target() is own and not _ringed(r, own)
    _leave(own, Qt.TabFocusReason)
    assert own.shown is False and r.follows.target() is None
    opened = QWidget(None, Qt.Popup)         # never shown: an opened list marks its own row
    line = QPushButton("a row of the list", opened)
    _arrive(line, Qt.TabFocusReason)
    assert r.follows.target() is None and not _ringed(r, line)


def test_the_ring_is_two_pixels_of_the_focus_colour_and_clear_inside(ring_room):
    from core.gui import design
    r = ring_room

    def painted(widget, ratio):              # the ring on a control, as it paints itself at a display scaling
        _arrive(widget, Qt.TabFocusReason)
        ring = r.follows.ring()
        picture = _image(ring.width(), ring.height(), ratio)
        ring.render(picture, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
        middle, half = ring.width() / 2, ring.height() / 2
        _leave(widget, Qt.TabFocusReason)
        return picture, middle, half

    def cover(picture, x, y):                # how solid the device pixel under a logical point is, 0 to 255
        ratio = picture.devicePixelRatio()
        return picture.pixelColor(int(x * ratio), int(y * ratio)).alpha()

    band = (0.5, 1.5, 2.5, 3.5)              # logical points across the ring: two inside it, two past it
    r.row.setProperty("ring_left", 4)        # a row with a bar four wide at its left: the ring starts after it
    for dark in (False, True):
        with theme(dark):
            focus = design.hex_colour("focus")
            for ratio in (1.0, 1.25, 1.5, 2.0):
                where = (dark, ratio)
                outside, middle, half = painted(r.button, ratio)
                assert [cover(outside, middle, y) for y in band + (half,)] == [255, 255, 0, 0, 0], where
                assert [_at(outside, middle, y) for y in band[:2]] == [focus] * 2, where
                inside, middle, half = painted(r.row, ratio)
                assert [cover(inside, x, half) for x in band] == [0] * 4, where       # the bar's four, left free
                assert [cover(inside, 4 + x, half) for x in band] == [255, 255, 0, 0], where
                assert [cover(inside, middle, y) for y in band + (half,)] == [255, 255, 0, 0, 0], where
                assert _at(inside, 4.5, half) == focus, where


def test_a_follower_told_to_wait_hears_nothing_until_it_listens():
    from core.gui import style
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    button = QPushButton("Check again", window)
    button.setGeometry(40, 40, 120, 34)
    follows = style.FocusFollower(app, style.PrismStyle(), listen=False)
    try:
        _arrive(button, Qt.TabFocusReason)
        assert not follows.listening() and follows.target() is None
        follows.listen()
        follows.listen()                     # twice is once
        _arrive(button, Qt.TabFocusReason)
        assert follows.listening() and follows.target() is button
    finally:
        follows.stop()
    follows.stop()                           # and stopping twice is once
    assert not follows.listening() and follows.ring() is None and follows.parent() is None


def test_the_sheet_marks_no_focus_of_its_own():
    from core.gui import design
    assert not [selector for selector in design.INTERIM_SELECTORS if ":focus" in selector]
    for dark in (False, True):
        qss = design.build_qss(dark)
        assert ":focus" not in qss, dark
        rules = {tuple(selectors): declarations for selectors, declarations in _rules(qss)}
        assert rules[("QCheckBox", "QRadioButton")]["outline"] == "none"    # the ring is the one sign of focus
