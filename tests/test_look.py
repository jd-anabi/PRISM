"""The look: its one table of colours, type and sizes, and the functions every reader of it shares.

Nothing here imports torch, so the look can be tested without the engine;
``test_the_look_and_this_file_load_without_the_engine`` proves it in a fresh interpreter, which runs this whole file:
its module level holds imports, constants and definitions only. One application object serves every suite of a run,
so the theme and the text factor are changed here only inside a helper or a fixture that puts back what it found:
``theme`` and ``text_size`` are the two helpers for it.
"""
import ast
import builtins
import contextlib
import datetime
import gc
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import (QColor, QFocusEvent, QFont, QFontDatabase, QFontInfo, QFontMetricsF, QGuiApplication,
                           QIcon, QImage, QKeyEvent, QMouseEvent, QPainter, QRegion, QWheelEvent)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QLineEdit, QMenu, QPlainTextEdit, QPushButton,
                               QRadioButton, QStyle, QStyleFactory, QStyleOptionButton, QStyleOptionFocusRect,
                               QWidget)

from core.gui import design, fonts, paint
from core.gui.widgets.marker import Marker, painted_icon
from core.gui.widgets.ruled import Ruled, SectionHead

REPO = Path(__file__).resolve().parents[1]
# every module of the look that exists; a new one joins this tuple in the commit that adds it, and the
# fresh-interpreter test is then its proof that it loads no engine
LOOK_MODULES = ("core.gui.design", "core.gui.fonts", "core.gui.theming", "core.gui.mpl_theme", "core.gui.settings",
                "core.gui.file_chooser", "core.gui.widgets.labeled_inputs", "core.gui.paint", "core.gui.style",
                "core.gui.reported", "core.Helpers.plot_colours", "core.gui.widgets.marker", "core.gui.widgets.ruled")
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


@pytest.mark.parametrize("step", ["setPalette", "setStyleSheet", "setFont", "retheme", "refresh_focus"])
def test_an_apply_that_fails_part_way_is_made_again_when_the_scheme_before_it_is_chosen(stand_in_app, step):
    """An apply that raises part-way -- at the palette, at the sheet, at the fonts of the widget
    classes, in a widget's own ``retheme()`` or, last of all, where the focus ring is put back on
    its control -- leaves a look that is not whole: half one scheme and half the other, or the new
    scheme with its ring not put back. The error comes out, nothing is signalled, and the
    Appearance can be used again. Nothing counts as applied while the work is under way, so
    neither scheme is taken for a repeat when it is chosen next: not the one shown before the
    failure, and not the one that failed. Either sets the palette and the sheet again."""
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QWidget
    from core.gui import style, theming

    class Painted(QWidget):
        def retheme(self): pass

    app = stand_in_app
    # what the failed apply had set when the step refused: palettes, sheets, fonts of widget classes
    all_set = (1, 1, len(CLASSES_WITH_A_FONT))      # the palette, the sheet and every class font
    set_before_the_step = {"setPalette": (0, 0, 0), "setStyleSheet": (1, 0, 0), "setFont": (1, 1, 0),
                           "retheme": all_set, "refresh_focus": all_set}[step]
    refuser = app                              # whose step refuses: the stand-in, a widget it lists,
    if step == "retheme":                      # or the module that puts the ring back
        refuser = Painted()                    # a top-level widget of the stand-in; never shown
        app.tops.append(refuser)
    elif step == "refresh_focus":
        refuser = style
    # A module holds its function itself, and that function is what the repair puts back. The stand-in
    # and the widget take the step from their class: the refusal is set in front of it on the one
    # object, and taking the refusal away is the repair.
    own = vars(refuser).get(step)

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
            if own is None:
                delattr(refuser, step)         # repaired: its own method again
            else:
                setattr(refuser, step, own)    # repaired: the module's own function again
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
    """In the bundled files the feature that gives the digits one width also widens the hyphen and the asterisk
    to a digit's width: switched on everywhere, every compound word reads as if it were spaced. So a font
    carries it only when it is asked for."""
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
        assert len(rules) == 87, len(rules)
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
    assert len(set(design.INTERIM_SELECTORS)) == len(design.INTERIM_SELECTORS) == 46


def test_no_pseudo_state_sets_a_weight_or_a_size_and_words_take_lettering_colours():
    from core.gui import design
    for dark in (False, True):
        t = design.tokens(dark)
        lettering = {t[name] for name in ("ink", "ink_2", "ink_3", "disabled_ink", "on_accent", "accent")}
        for selectors, declarations in _rules(design.build_qss(dark)):
            named = _named_in_a_state(selectors, declarations)      # a face, a size or a weight in a state; one greyed button may name 400
            assert not named, (dark, selectors, named)
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
        for make, when_disabled in ((QLineEdit, "disabled_ink"), (QPlainTextEdit, "ink_3"),
                                    (QTextEdit, "disabled_ink"), (QComboBox, "disabled_ink")):
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
    it does not reach that one. An area whose own frame was switched off is given none: the rule names the
    frame an area has by default, or it would put a line round a page that scrolls and take a pixel from each
    of its sides. And a list or a viewer that is greyed, as the list of a form is for the length of a run, looks
    greyed as a field does. The palette and the sheet are set on one holder, which is never shown."""
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QFrame, QListWidget, QScrollArea, QTextBrowser
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
        bare = QScrollArea(holder)                   # its own frame switched off: the sheet gives it none
        bare.setFrameShape(QFrame.NoFrame)
        bare.setWidget(QWidget())
        unframed = painted(bare)
        assert bare.frameWidth() == 0, (dark, "an area that switched its frame off was given one")
        assert _at(unframed, width / 2, 0) == _at(unframed, width / 2, height / 2) != quiet, dark
        greyed_look = (design.hex_colour("disabled_line", dark), design.hex_colour("disabled_fill", dark))
        for greyed in (QListWidget(holder), QTextBrowser(holder)):
            greyed.setEnabled(False)
            picture = painted(greyed)
            assert (_at(picture, width / 2, 0), _at(picture, width / 2, height / 2)) == greyed_look, (
                dark, type(greyed).__name__, "greyed")
            assert greyed.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).name().upper() \
                == design.hex_colour("disabled_ink", dark), (dark, type(greyed).__name__, "greyed lettering")
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
    # An apply asks the repaint through the widget class, so a method of that name on a widget would not hear
    # it: the class's own is stood in for, for the length of this test.
    monkeypatch.setattr(QWidget, "update", lambda widget, *region: calls.append("update"))
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


def test_an_apply_reaches_a_top_level_list_without_failing(stand_in_app):
    """A list, a tree or a table refuses the plain repaint when it is asked on the widget itself: on those
    classes the name stands for the repaint of one row, which wants the row. An apply that asked it so of a
    list that is a window of its own stopped there, with the new look set and nothing recorded or signalled.
    Asked through the widget class, every widget takes it."""
    from PySide6.QtWidgets import QListWidget
    from core.gui import theming
    app = stand_in_app
    app.tops.append(QListWidget())                   # a window of its own; never shown
    try:
        appearance = theming.Appearance(app)
        signalled = []
        appearance.theme_changed.connect(signalled.append)
        appearance.set_mode("dark")
        assert (len(app.palettes), len(app.sheets), signalled) == (1, 1, [True])
        appearance.set_mode("dark")                  # recorded as applied: the same scheme again sets nothing
        assert (len(app.palettes), len(app.sheets), signalled) == (1, 1, [True])
    finally:
        del app.tops[:]                              # the list must not wait for the stand-in to go


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
        for make in (W.QListWidget, W.QTextBrowser):
            add(make).setEnabled(False)
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
        ring = style.FocusRing()                                       # the frame asks these two for itself
        assert worn.pixelMetric(QStyle.PM_FocusFrameHMargin, None, ring) == 4      # the ring and its gap do not
        assert worn.pixelMetric(QStyle.PM_FocusFrameVMargin, None, ring) == 4


def test_the_focus_frames_margins_are_the_rings_for_the_ring_alone():
    """The two metrics that tell the focus frame how far outside its control it stands are also where lists and
    tables take the margin of their cells from. Answered with the ring's four pixels for every widget, they move
    the text of every cell two pixels to the right and make every column four wider. So the small style gives
    the ring's answer to the ring alone; any other widget, and none, gets the toolkit's own."""
    from PySide6.QtWidgets import QListWidget, QTableView
    from core.gui import design, style
    QApplication.instance() or QApplication([])
    worn, stock = style.PrismStyle(), QStyleFactory.create("Fusion")
    table, rows = QTableView(), QListWidget()
    ring, inset = style.FocusRing(), style.FocusRing()
    inset.inset = True
    for metric in (QStyle.PM_FocusFrameHMargin, QStyle.PM_FocusFrameVMargin):
        assert stock.pixelMetric(metric, None, None) != 4, "the toolkit answers as the ring does: this proves nothing"
        for widget in (table, rows, None):
            assert worn.pixelMetric(metric, None, widget) == stock.pixelMetric(metric, None, widget), (metric, widget)
        assert worn.pixelMetric(metric, None, ring) == design.SIZE["focus_ring"] + design.SIZE["focus_gap"] == 4
        assert worn.pixelMetric(metric, None, inset) == 0

    def column(by):                              # how wide a list makes its one column, under one style
        listed = QListWidget()
        listed.setStyle(by)                      # on this one widget, never on the application
        listed.addItem("a row")
        return listed.sizeHintForColumn(0)

    assert column(worn) == column(stock), "a cell is wider under the small style than the toolkit makes it"


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


# ── the figures the engine draws: light settings, written once ───────────────────────────────────
@contextlib.contextmanager
def _figure_settings_kept():
    """matplotlib's process-wide settings may be written inside the block; what was in force before
    is put back."""
    import matplotlib
    saved = dict(matplotlib.rcParams)
    try:
        yield
    finally:
        matplotlib.rcParams.update(saved)


_LIGHT_FIGURE_SETTINGS = {
    "figure.facecolor": "#FFFFFF", "figure.edgecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF", "savefig.edgecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#1A1A1A",
    "axes.labelcolor": "#1A1A1A", "axes.titlecolor": "#1A1A1A", "text.color": "#1A1A1A",
    "xtick.color": "#1A1A1A", "ytick.color": "#1A1A1A",
    "xtick.labelcolor": "#4D4D4D", "ytick.labelcolor": "#4D4D4D", "grid.color": "#D4D4D4",
    "legend.facecolor": "#FFFFFF", "legend.edgecolor": "#FFFFFF", "legend.framealpha": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.6, "ytick.major.size": 3.6,
    "axes.linewidth": 0.72, "xtick.major.width": 0.72, "ytick.major.width": 0.72,
    "lines.linewidth": 1.44, "lines.solid_joinstyle": "round", "lines.dash_joinstyle": "round",
    "axes.unicode_minus": True, "font.family": ["Inter", "DejaVu Sans"],
}
_LINE_COLOURS = ["#1A1A1A", "#0072B2", "#C24E00", "#007A5A", "#E69F00", "#56B4E9", "#CC79A7"]


def test_the_figure_settings_are_the_light_column_and_nothing_else():
    from core.gui import mpl_theme
    rc = mpl_theme._rc()
    cycle = rc.pop("axes.prop_cycle")
    assert set(rc) == set(_LIGHT_FIGURE_SETTINGS), sorted(set(rc) ^ set(_LIGHT_FIGURE_SETTINGS))
    for name, value in _LIGHT_FIGURE_SETTINGS.items():
        assert rc[name] == (pytest.approx(value) if isinstance(value, float) else value), name
    assert cycle.keys == {"color"}, "the cycle carries more than a colour"
    assert cycle.by_key()["color"] == _LINE_COLOURS
    assert _LINE_COLOURS == [design.hex_colour(name, dark=False) for name in design.SERIES]


def test_the_figure_settings_do_not_follow_the_theme_or_the_text_size():
    """Neither way a theme reaches the window moves a figure setting: not the look's own flag
    (nor the text factor beside it), and not the signal a change of appearance sends."""
    import matplotlib
    from PySide6.QtCore import QObject, Signal
    from core.gui import mpl_theme

    class DarkAppearance(QObject):            # all that install() may ask of an appearance
        theme_changed = Signal(bool)

        def is_dark(self):
            return True

    with theme(False), text_size(1.0):
        light = mpl_theme._rc()
    with theme(True), text_size(2.0), _figure_settings_kept():
        assert mpl_theme._rc() == light
        appearance = DarkAppearance()
        mpl_theme.install(appearance)
        assert {name: matplotlib.rcParams[name] for name in light} == light, \
            "under a dark appearance the settings written are not the light ones"
        matplotlib.rcdefaults()   # matplotlib's own back, so a setting install wrote would show if written again
        matplotlib.rcParams["figure.facecolor"] = "#123456"      # a mark no setting of the look has
        before = dict(matplotlib.rcParams)
        appearance.theme_changed.emit(False)
        appearance.theme_changed.emit(True)
        rewritten = sorted(name for name, value in dict(matplotlib.rcParams).items() if before[name] != value)
        assert not rewritten, f"a change of theme rewrote these figure settings: {rewritten}"


def test_the_figure_settings_name_inter_only_when_both_files_reached_matplotlib(monkeypatch):
    from core.gui import fonts, mpl_theme
    monkeypatch.setattr(fonts, "register_with_matplotlib", lambda: False)
    assert "font.family" not in mpl_theme._rc() and len(mpl_theme._rc()) == 31
    monkeypatch.setattr(fonts, "register_with_matplotlib", lambda: True)
    assert mpl_theme._rc()["font.family"] == ["Inter", "DejaVu Sans"]


def test_under_the_figure_settings_unnamed_lines_take_the_seven_colours_in_order():
    from matplotlib.figure import Figure
    from core.gui import mpl_theme
    with _figure_settings_kept():
        mpl_theme.apply_mpl_theme()
        ax = Figure().add_subplot()
        lines = [ax.plot([0, 1], [i, i + 1])[0] for i in range(8)]
        assert [ln.get_color() for ln in lines] == _LINE_COLOURS + _LINE_COLOURS[:1]
        assert {ln.get_linestyle() for ln in lines} == {"-"}
        assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()


def test_under_the_figure_settings_a_character_the_lettering_lacks_is_drawn_without_a_warning():
    """A record's or a parameter's name can hold a character Inter has no drawing for. The settings
    name a second face for those, so the label is drawn and matplotlib warns of no missing glyph.
    With Inter alone the same label does warn, which is what makes this test able to fail."""
    import io
    import warnings
    import matplotlib
    from matplotlib.figure import Figure
    from core.gui import mpl_theme

    def missing_glyphs():
        fig = Figure()
        ax = fig.add_subplot()
        ax.set_title("run ✖ seven")              # a heavy multiplication sign: not in Inter
        ax.set_xlabel("ordinary words")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.savefig(io.BytesIO(), format="png")
        return [str(w.message) for w in caught if "missing from font" in str(w.message)]

    with _figure_settings_kept():
        mpl_theme.apply_mpl_theme()
        assert matplotlib.rcParams["font.family"][0] == "Inter", "Inter did not reach matplotlib"
        assert missing_glyphs() == []
        matplotlib.rcParams["font.family"] = ["Inter"]
        assert missing_glyphs(), "this label no longer holds a character Inter lacks: choose another"


# ── the plotting code's line colours ────────────────────────────────────────────────────────────
_PLOTTING_FILES = ("core/Helpers/visualizers.py", "core/FDT/plots.py", "core/FDT/compare.py",
                   "core/FDT/cross_validation_plots.py", "core/Reduction/plots.py",
                   "core/diagnostics/identifiability.py", "core/diagnostics/probes.py")
# the names those files once wrote out themselves: each is refused now, as every other string is
_OLD_LINE_COLOURS = frozenset({"steelblue", "tab:blue", "red", "r", "crimson", "tab:red", "green",
                               "tab:green", "darkorange", "orange", "tab:orange", "gray"})
_COLOUR_KEYWORDS = frozenset({"color", "c", "colors", "edgecolor", "edgecolors", "facecolor",
                              "facecolors", "ecolor", "markerfacecolor", "markeredgecolor", "mfc",
                              "mec", "fc", "ec", "foreground", "labelcolor"})
# The only colours a plotting file may write out itself, each because one of them does today, and why
# it may stay. A string that is not a key here is refused wherever a call can take a colour.
_COLOURS_THAT_STAY = {
    "white": "a line or lettering laid over a colour map: its ground is the map, not the page",
    "black": "lettering over the light cells of a colour map",
    "none": "no edge round a marker, and no mesh lines on a surface",
}
_LINE_CALLS = frozenset({"plot", "semilogx", "semilogy", "loglog", "errorbar", "step", "stem"})
# A line call's short format: markers and line styles, and of the colour letters the white one alone,
# for that same white line over a colour map. Any other string handed over that way is refused.
_LINE_FORMAT = re.compile(r"(?:w|--|-\.|[-:.,ov^<>1-48spP*hH+xXDd|_])*")


def _bound_values(tree) -> dict:
    """Name to every expression a parsed file binds it to: by an assignment (pair by pair when a
    written-out tuple is assigned to one), as a parameter's default, and as the variable of a loop
    over a written-out tuple or list. Whole file, whatever the function: a name is looked up by its
    spelling alone."""
    bound = {}

    def bind(target, value):
        if isinstance(target, ast.Name):
            bound.setdefault(target.id, []).append(value)
        elif (isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List))
              and len(target.elts) == len(value.elts)):
            for one, its_value in zip(target.elts, value.elts):
                bind(one, its_value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
            for element in node.iter.elts:
                bind(node.target, element)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            named = node.args.posonlyargs + node.args.args
            pairs = list(zip(named[len(named) - len(node.args.defaults):], node.args.defaults))
            pairs += [pair for pair in zip(node.args.kwonlyargs, node.args.kw_defaults) if pair[1] is not None]
            for parameter, default in pairs:
                bound.setdefault(parameter.arg, []).append(default)
    return bound


def _written_out(value, bound, followed=()) -> list:
    """``(line, text)`` for each colour written out in an expression that a call hands to a colour
    keyword or a colour setter. Written out: a string other than the three of ``_COLOURS_THAT_STAY``;
    a string put together on the spot (an f-string, a concatenation or a ``%`` with a string in it,
    a method called on a string); a tuple or list of numbers alone. A name is followed to every value
    the file binds it to, a conditional to both of its arms, an ``or`` to each side, a subscript to
    what it indexes (a read of matplotlib's settings excepted), any other tuple, list, set or dict to
    its members. Everything else is no literal and passes: a name of the colours module, a read of
    the settings, a call, data, a number by itself."""
    def is_text(node):
        return isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str))

    def is_number(node):
        node = node.operand if isinstance(node, ast.UnaryOp) else node
        return isinstance(node, ast.Constant) and type(node.value) in (int, float)

    def each(nodes, followed=followed):
        return [hit for node in nodes for hit in _written_out(node, bound, followed)]

    if isinstance(value, ast.Constant):
        stays = not isinstance(value.value, str) or value.value in _COLOURS_THAT_STAY
        return [] if stays else [(value.lineno, value.value)]
    if (isinstance(value, ast.JoinedStr)
            or (isinstance(value, ast.BinOp) and any(is_text(part) for part in ast.walk(value)))
            or (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and is_text(value.func.value))
            or (isinstance(value, (ast.Tuple, ast.List)) and value.elts and all(is_number(e) for e in value.elts))):
        return [(value.lineno, ast.unparse(value))]
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return each(value.elts)
    if isinstance(value, ast.Dict):
        return each(value.values)
    if isinstance(value, ast.IfExp):
        return each([value.body, value.orelse])
    if isinstance(value, ast.BoolOp):
        return each(value.values)
    if isinstance(value, ast.Name) and value.id not in followed:
        return each(bound.get(value.id, ()), followed + (value.id,))
    if isinstance(value, ast.Subscript):
        held = value.value                                   # plt.rcParams[...], matplotlib.rcParams[...]
        settings = (isinstance(held, ast.Attribute) and held.attr == "rcParams"
                    and isinstance(held.value, ast.Name) and held.value.id in ("plt", "matplotlib"))
        return [] if settings else each([held])
    return []


def _own_colour_arguments(tree) -> list:
    """``(line, text)`` for every colour a parsed file writes out itself where a call can take one:
    under a colour keyword (of a call, or as a key of a dict written out), as the first argument of a
    colour setter (``set_`` and a colour keyword), and as the short format of a line call. Parsed, so
    a colour name in a comment, a label or a file mode is not an argument."""
    bound, hits = _bound_values(tree), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in _COLOUR_KEYWORDS:
                    hits += _written_out(value, bound)
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg in _COLOUR_KEYWORDS:
                hits += _written_out(kw.value, bound)
        method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if method.startswith("set_") and method[4:] in _COLOUR_KEYWORDS and node.args:
            hits += _written_out(node.args[0], bound)
        if method in _LINE_CALLS:
            for handed in node.args + [kw.value for kw in node.keywords if kw.arg == "fmt"]:
                if isinstance(handed, ast.JoinedStr):
                    hits.append((handed.lineno, ast.unparse(handed)))
                elif (isinstance(handed, ast.Constant) and isinstance(handed.value, str)
                      and not _LINE_FORMAT.fullmatch(handed.value)):
                    hits.append((handed.lineno, handed.value))
    return sorted(set(hits))


def test_the_plotting_colour_scan_catches_each_form_and_spares_what_stays():
    """The scanner is checked on snippets first: with the tree clean, the walk of the seven files
    would go on passing if the scanner stopped matching anything."""
    def found(source):
        return _own_colour_arguments(ast.parse(source))

    def as_written(expression):
        return ast.unparse(ast.parse(expression, mode="eval").body)

    by_number = "C" "3"                      # a colour of the cycle by its number, spelt here in two parts
    others = ["blue", "navy", "grey", "k", "w", "0.5", by_number, "#0072B2", "xkcd:teal"]
    for name in sorted(_OLD_LINE_COLOURS) + others:
        assert found(f"ax.plot(x, y, color={name!r})") == [(1, name)], name
    for source, reported in (
            ("ax.scatter(x, y, c='gray', edgecolors='none')", [(1, "gray")]),
            ("ax.bar(x, y, color=['tab:blue', 'tab:red'])", [(1, "tab:blue"), (1, "tab:red")]),
            ("ax.plot(x, y, color=(0.1, 0.2, 0.3))", [(1, "(0.1, 0.2, 0.3)")]),
            ("patch.set_facecolor([0.5, 0.5, 0.5, 1.0])", [(1, "[0.5, 0.5, 0.5, 1.0]")]),
            ("ax.axvspan(0, 1, facecolor=f\"#{'00'}72B2\")", [(1, as_written("f\"#{'00'}72B2\""))]),
            ("ax.axvline(1.0, color='tab:' + shade)", [(1, "'tab:' + shade")]),
            ("ax.axvline(1.0, color='#{}'.format(code))", [(1, "'#{}'.format(code)")]),
            ("ax.lines[0].set_color('teal')", [(1, "teal")]),
            ("ax.text(0, 0, 'n', bbox={'facecolor': 'yellow'})", [(1, "yellow")]),
            ("ax.text(0, 0, 'n', color='white' if dark else 'navy')", [(1, "navy")]),
            ("shade = 'teal'\nax.axhline(0.0, color=shade)", [(1, "teal")]),
            ("def draw(ax, shade='teal'):\n    ax.axhline(0.0, color=shade)", [(1, "teal")]),
            ("for ax, shade in ((a, 'teal'), (b, 'navy')):\n    ax.axhline(0.0, color=shade)",
             [(1, "navy"), (1, "teal")]),
            ("shades = ['teal', 'navy']\nax.plot(x, y, color=shades[i])", [(1, "navy"), (1, "teal")]),
            ("ax.plot(x, y, 'r--')", [(1, "r--")]),
            ("ax.plot(x, y, 'k-')", [(1, "k-")]),
            (f"ax.plot(x, y, '{by_number}-')", [(1, by_number + "-")]),
            ("ax.plot(x, y, 'green')", [(1, "green")]),
            ("ax.errorbar(x, y, fmt='go')", [(1, "go")])):
        assert found(source) == reported, source
    spared = ("ax.plot(x, y, 'o-', color=plot_colours.BLUE)\n"
              "ax.plot(x, y, 'w-', lw=2)\n"
              "ax.text(0, 0, 'r', color='white' if dark else 'black')\n"
              "ax.axhline(0, color=plt.rcParams['axes.edgecolor'])\n"
              "ax.axvline(0, color=matplotlib.rcParams['text.color'])\n"
              "ax.scatter(x, y, c=plot_colours.ORANGE, edgecolors='none')\n"
              "ink = plt.rcParams['text.color']\n"
              "cell.get_text().set_color(ink)\n"
              "cell.set_edgecolor(plt.rcParams['axes.edgecolor'])\n"
              "box = dict(boxstyle='round', facecolor=plt.rcParams['axes.facecolor'], alpha=0.9)\n"
              "ax.scatter(x, y, c=z_scores, cmap='viridis')\n"
              "ax.plot(x, y, color=cmap(0.5))\n"
              "ax.plot(x, y, color=None)\n"
              "ax.imshow(data, cmap='viridis')\n"
              "ax.plot(x, y, label='red')\n"
              "open(path, 'r')\n")
    assert found(spared) == []


def test_no_plotting_file_names_a_line_colour_of_its_own():
    """The scan reads the seven files of ``_PLOTTING_FILES`` and no other. In them, wherever a call
    can take a colour (a colour keyword, a colour setter, a line call's short format), nothing is
    written out but the exceptions the tree uses today, each named with its reason: the three
    strings of ``_COLOURS_THAT_STAY`` and the white letter of a short format. Every other colour is
    a name of core/Helpers/plot_colours.py or a read of matplotlib's settings, so a figure saved
    from the command line and one saved from the window carry the same named ones; a line that
    names no colour takes matplotlib's cycle, and a colour map is named as before. Refused: any
    other string (a colour's name, a letter, a grey level, a hex value, a colour of the cycle by its
    number), a string put together on the spot, a tuple or list of numbers, directly or through a
    name the file binds to one. Not seen: a colour that reaches the call through another call, an
    attribute of another module or a container filled elsewhere. An exception no file uses any more
    fails here too, so the list cannot outlive its reasons."""
    left, in_use, white_format = [], set(), False
    for rel in _PLOTTING_FILES:
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        left += [f"{rel}:{line} {text!r}" for line, text in _own_colour_arguments(tree)]
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in _COLOUR_KEYWORDS:
                in_use |= {part.value for part in ast.walk(node.value) if isinstance(part, ast.Constant)}
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _LINE_CALLS:
                white_format |= any(isinstance(handed, ast.Constant) and isinstance(handed.value, str)
                                    and "w" in handed.value for handed in node.args)
    assert left == [], (f"{len(left)} colour argument(s) still name a colour of their own; each "
                        f"takes its colour from core/Helpers/plot_colours.py:\n  " + "\n  ".join(left))
    unused = sorted(set(_COLOURS_THAT_STAY) - in_use) + ([] if white_format else ["the white letter of a short format"])
    assert not unused, f"no plotting file writes these out any more, so they stop being exceptions: {unused}"


def test_the_plotting_colours_module_imports_nothing_but_itself():
    """The effective-temperature and reduction plotting files, and a comparison of saved runs, must
    be able to import the colours with neither torch nor matplotlib behind them. RUN in a fresh
    interpreter: this process already holds both, so a look at its own modules would prove
    nothing. The module's own statements are read too: it holds four values and imports nothing."""
    probe = ("import sys\n"
             "before = set(sys.modules)\n"
             "import core.Helpers.plot_colours\n"
             "new = sorted(set(sys.modules) - before)\n"
             "sys.exit(0 if new == ['core', 'core.Helpers', 'core.Helpers.plot_colours'] else repr(new))\n")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO), capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    tree = ast.parse((REPO / "core" / "Helpers" / "plot_colours.py").read_text(encoding="utf-8"))
    imports = [n.lineno for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [], f"core/Helpers/plot_colours.py imports at lines {imports}"


def test_the_plotting_colours_are_the_second_to_fifth_line_colours():
    """The engine holds its own copy of four colours, so that it imports nothing of the window;
    this is what keeps the copy equal to the window's table. The first line colour, the ink, is
    not held (a plotting function reads it from matplotlib's settings), and nothing else is."""
    from core.Helpers import plot_colours
    held = (plot_colours.BLUE, plot_colours.VERMILION, plot_colours.GREEN, plot_colours.ORANGE)
    assert held == ("#0072B2", "#C24E00", "#007A5A", "#E69F00")
    assert design.SERIES[1:5] == ("series_2", "err", "ok", "warn")
    assert held == tuple(design.COLOURS[name][0] for name in design.SERIES[1:5])
    assert sorted(n for n in vars(plot_colours) if not n.startswith("_")) == [
        "BLUE", "GREEN", "ORANGE", "VERMILION"]


# ── drawing code reports its own errors ──────────────────────────────────────────────────────────
def _name_of(method) -> str:
    """The name a guarded method is recorded under: its module's and its own."""
    return f"{method.__module__}.{method.__qualname__}"


def test_a_guarded_method_that_raises_is_reported_once_and_answers_its_fallback(monkeypatch):
    """The guard a method carries when the toolkit calls it while a window is drawn or laid out. Its error
    goes to the hook in force at that moment, once for each method, and the method answers its fallback. A
    keyboard interrupt and an exit pass; nothing else comes out, whether the hook fails or the fallback does."""
    from core.gui import reported

    class Drawn:
        fail = True

        @reported.reported("stock")
        def by_a_value(self, n, twice=False):
            """What the method says of itself."""
            if self.fail:
                raise ValueError(f"no {n}")
            return n * (2 if twice else 1)

        @reported.reported(lambda self, n, twice=False: ("stock", self, n, twice))
        def by_a_call(self, n, twice=False):
            raise ValueError(f"no {n}")

        @reported.reported()
        def stopped(self, by):
            raise by

        @reported.reported(lambda self: 1 // 0)
        def by_a_call_that_fails(self):
            raise ValueError("first")

    def told():                              # what the record lists: the first error under each name, as words
        return [words for _name, words in reported.ERRORS]

    shown = []
    drawn, value, call = Drawn(), _name_of(Drawn.by_a_value), _name_of(Drawn.by_a_call)
    failing = _name_of(Drawn.by_a_call_that_fails)
    assert value.endswith("Drawn.by_a_value") and call.endswith("Drawn.by_a_call")
    # set after the class is made: the hook is looked for when a method fails, not when it is guarded
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: shown.append((kind, error, trace)))
    reported.forget()
    try:
        assert drawn.by_a_value(3) == "stock"
        assert reported.ERRORS == [(value, "ValueError: no 3")] and reported.COUNTS == {value: 1}
        (kind, error, trace), = shown        # the hook is handed the error itself, with its traceback
        assert kind is ValueError and str(error) == "no 3" and trace is error.__traceback__ is not None
        assert drawn.by_a_value(4) == "stock"                # the second time: counted, and no second box
        assert told() == ["ValueError: no 3"] and reported.COUNTS == {value: 2} and len(shown) == 1
        assert drawn.by_a_call(5, twice=True) == ("stock", drawn, 5, True)       # each method has its own first time
        assert [name for name, _words in reported.ERRORS] == [value, call] and len(shown) == 2
        reported.forget()                                    # the record goes, and every count with it
        assert reported.ERRORS == [] and reported.COUNTS == {}
        assert drawn.by_a_value(6) == "stock" and len(shown) == 3

        drawn.fail = False                                   # a method that does not raise is as it was
        assert drawn.by_a_value(6) == 6 and drawn.by_a_value(6, twice=True) == 12
        assert Drawn.by_a_value.__name__ == "by_a_value"
        assert Drawn.by_a_value.__doc__ == "What the method says of itself."
        assert told() == ["ValueError: no 6"] and reported.COUNTS == {value: 1} and len(shown) == 3
        drawn.fail = True

        for stop in (KeyboardInterrupt(), SystemExit(2)):    # neither is held back, neither is counted
            with pytest.raises(type(stop)):
                drawn.stopped(stop)
        assert drawn.stopped(OSError("held")) is None and told() == ["ValueError: no 6", "OSError: held"]

        again = []

        def box(kind, error, trace):         # the box's own loop paints the failing widget once more
            again.append(error)
            drawn.by_a_value(8)

        monkeypatch.setattr(sys, "excepthook", box)
        reported.forget()
        assert drawn.by_a_value(7) == "stock" and len(again) == 1
        assert told() == ["ValueError: no 7"] and reported.COUNTS == {value: 2}

        def broken(kind, error, trace):
            raise RuntimeError("the hook failed")

        monkeypatch.setattr(sys, "excepthook", broken)
        reported.forget()
        assert drawn.by_a_value(9) == "stock"
        assert reported.ERRORS == [(value, "ValueError: no 9"), ("sys.excepthook", "RuntimeError: the hook failed")]
        assert drawn.by_a_call_that_fails() is None          # a fallback that fails too: listed, and no answer
        assert [name for name, _words in reported.ERRORS[2:]] == [failing, failing + " (its fallback)"]
        assert told()[2] == "ValueError: first" and told()[3].startswith("ZeroDivisionError: ")
        assert reported.COUNTS == {value: 1, "sys.excepthook": 2, failing: 1, failing + " (its fallback)": 1}

        # a test that makes a guarded method raise on purpose: counted and listed as always, shown to
        # nobody, and on the way out the record and the counts are as they were found
        del shown[:]
        monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: shown.append(str(error)))
        reported.forget()
        drawn.by_a_value(10)
        with reported.expected() as caught:
            assert drawn.by_a_value(11) == "stock" and drawn.by_a_call(12) == ("stock", drawn, 12, False)
            assert caught == [(call, "ValueError: no 12")]   # the other had failed before the block: a repeat
            assert told() == ["ValueError: no 10", "ValueError: no 12"] and reported.COUNTS == {value: 2, call: 1}
            assert shown == ["no 10"]
        assert told() == ["ValueError: no 10"] and reported.COUNTS == {value: 1} and len(caught) == 1
        drawn.by_a_call(13)                  # not shown inside the block, so this is its first time
        drawn.by_a_value(14)                 # shown before the block: not again
        assert shown == ["no 10", "no 13"]
        with pytest.raises(KeyError):        # put back also when the block is left by an error
            with reported.expected():
                assert drawn.stopped(OSError("inside")) is None and len(reported.ERRORS) == 3
                raise KeyError("the test's own")
        assert told() == ["ValueError: no 10", "ValueError: no 13"] and reported.COUNTS == {value: 2, call: 1}
    finally:
        reported.forget()                    # raised on purpose: the session's check must find nothing


def test_a_method_that_keeps_failing_is_listed_once_and_counted(monkeypatch):
    """A paint that fails at every repaint, for as long as a run of days lasts, must not fill the record: a
    method's first failure is listed and shown, and its later ones are counted and nothing more."""
    from core.gui import reported

    class Drawn:
        @reported.reported("stock")
        def paintEvent(self, n):
            raise ValueError(f"no {n}")

    shown = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: shown.append(str(error)))
    drawn, name = Drawn(), _name_of(Drawn.paintEvent)
    reported.forget()
    try:
        assert [drawn.paintEvent(n) for n in range(5)] == ["stock"] * 5
        assert reported.ERRORS == [(name, "ValueError: no 0")]       # the first, as words; the other four are not
        assert reported.COUNTS == {name: 5} and shown == ["no 0"]
        reported.forget()
        assert reported.ERRORS == [] and reported.COUNTS == {}
        assert drawn.paintEvent(5) == "stock" and shown == ["no 0", "no 5"]      # a first time again
        with reported.expected() as caught:  # a repeat inside the block is counted too, and is not caught
            drawn.paintEvent(6)
            assert caught == [] and reported.COUNTS == {name: 2}
        assert reported.ERRORS == [(name, "ValueError: no 5")] and reported.COUNTS == {name: 1}
        reported.forget()
        with reported.expected() as caught:  # a block that begins with nothing recorded: the first, then counts
            assert [drawn.paintEvent(n) for n in (7, 8, 9)] == ["stock"] * 3
            assert caught == reported.ERRORS == [(name, "ValueError: no 7")] and reported.COUNTS == {name: 3}
        assert reported.ERRORS == [] and reported.COUNTS == {} and shown == ["no 0", "no 5"]
    finally:
        reported.forget()


def test_two_methods_of_one_name_in_two_modules_each_have_their_first_time(monkeypatch):
    """Two modules can each hold a private class of one name. Their methods are two methods: the second
    one's first failure is shown and listed as the first one's was, under the module's name and its own."""
    from core.gui import reported

    def twin(module):                        # the paint event of a class called _Row, as that module would hold it
        def paintEvent(self):
            raise ValueError(f"in {module}")

        paintEvent.__qualname__, paintEvent.__module__ = "_Row.paintEvent", module
        return reported.reported()(paintEvent)

    shown = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: shown.append(str(error)))
    first, second = twin("core.gui.widgets.first"), twin("core.gui.widgets.second")
    reported.forget()
    try:
        assert first(None) is None and second(None) is None and first(None) is None
        assert shown == ["in core.gui.widgets.first", "in core.gui.widgets.second"]
        assert reported.ERRORS == [
            ("core.gui.widgets.first._Row.paintEvent", "ValueError: in core.gui.widgets.first"),
            ("core.gui.widgets.second._Row.paintEvent", "ValueError: in core.gui.widgets.second")]
        assert reported.COUNTS == {"core.gui.widgets.first._Row.paintEvent": 2,
                                   "core.gui.widgets.second._Row.paintEvent": 1}
    finally:
        reported.forget()


def test_a_failed_paint_leaves_no_painter_behind(monkeypatch):
    """An error keeps its traceback, the traceback the frames of the failing call, and a frame its locals: in
    a paint event that is the live painter. So the record keeps an error as words and nothing of the error
    itself: when the guarded call returns, the painter a failed paint had opened is gone, and the widget
    can be painted again. Two of the window's paint events open their painter as this one does, with no
    ``finally``. Checked on the two roads a failing paint takes in the running window: its first failure,
    which is handed to the hook, and a repeat, which is counted and nothing more."""
    from PySide6.QtCore import qInstallMessageHandler
    from core.gui import reported
    QApplication.instance() or QApplication([])

    class Local:
        pass

    class Failing(QWidget):
        fails, local = False, None

        @reported.reported()
        def paintEvent(self, _event):
            painter = QPainter(self)         # a local, ended at the end and by nothing else
            held = Local()
            Failing.local = weakref.ref(held)
            if Failing.fails:
                raise ValueError("the paint fails")
            painter.fillRect(self.rect(), Qt.magenta)        # a colour no ground has
            painter.end()

    shown = []                               # a quiet hook: it keeps the error's words and not the error
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: shown.append(str(error)))
    widget, name = Failing(), _name_of(Failing.paintEvent)
    widget.resize(20, 12)
    picture = _image(20, 12, 1.0)            # every paint goes into this one picture, which outlives them all
    said = []
    previous = qInstallMessageHandler(lambda kind, context, text: said.append(text))
    reported.forget()
    try:
        for failures in (1, 2):              # the failure that is shown, then a repeat
            Failing.fails = True
            widget.render(picture)           # the paint fails under its live painter
            Failing.fails = False
            assert shown == ["the paint fails"], failures                    # handed to the hook the first time alone
            assert reported.ERRORS == [(name, "ValueError: the paint fails")], failures      # and kept as words
            assert reported.COUNTS == {name: failures}
            # at once, with no collection: the next paint may come before any
            assert not picture.paintingActive(), f"failure {failures} left its painter on the picture"
            assert Failing.local() is None, f"a local of the failing call lives on after failure {failures}"
            assert _at(picture, 10, 6) != "#FF00FF"
            widget.render(picture)           # and the next paint is a good one
            assert [text for text in said if "paint" in text.lower()] == [], failures
            assert _at(picture, 10, 6) == "#FF00FF", failures
    finally:
        qInstallMessageHandler(previous)
        reported.forget()                    # failed on purpose, outside any block: the session's check finds nothing


# What the toolkit calls while a window is drawn or laid out. In the main window each is entered from inside
# the window's own handling of an event, where the binding drops whatever it raises.
_CALLED_WHILE_DRAWING = frozenset({
    "paintEvent", "eventFilter", "resizeEvent", "drawPrimitive", "drawControl", "drawComplexControl",
    "pixelMetric", "styleHint", "sizeHint", "minimumSizeHint", "heightForWidth", "event"})
# Nine more that the toolkit calls from inside that handling: a delegate's and an icon engine's ``paint``, a
# header's ``paintSection``, an icon engine's ``pixmap``, what a model answers in every repaint (``data``,
# ``headerData``, ``flags`` and its two counts, ``rowCount`` and ``columnCount``), and the ``showEvent`` of a
# page when the window comes back from minimised. Several of them are ordinary words, so these count only as
# a method of a class that has a class of the toolkit among its bases.
_CALLED_IN_A_TOOLKIT_CLASS = frozenset({"paint", "paintSection", "pixmap", "data", "headerData", "flags", "rowCount",
                                        "columnCount", "showEvent"})
# Ten of them, each with the fallback its guard names as it is written (None: it names none). Typed out so
# that a guard taken off one of them, or a fallback changed, fails here by name; a new one need not join.
_GUARDED_TODAY = {
    "screens/artifact_screen.py: ArtifactScreen.showEvent": "QWidget.showEvent",
    "screens/home_screen.py: HomeScreen.showEvent": "QWidget.showEvent",
    "style.py: PrismStyle.pixelMetric": "QProxyStyle.pixelMetric",
    "style.py: PrismStyle.styleHint": "QProxyStyle.styleHint",
    "style.py: PrismStyle.drawPrimitive": "QProxyStyle.drawPrimitive",
    "style.py: FocusRing.paintEvent": None,
    "style.py: FocusFollower.eventFilter": "False",
    "widgets/anim.py: _TransitionOverlay.paintEvent": None,
    "widgets/progress_pane.py: _Sparkline.paintEvent": None,
    "widgets/figure_stack.py: _FitLabel.resizeEvent": "QLabel.resizeEvent",
}
# The files the scan does not read, each with its reason.
_NOT_GUARDED_FILES = {
    "widgets/figure_window.py": "the popped-out figure is a window of its own, which does not keep the "
                                "interpreter while it handles an event: its canvas is the plotting library's own "
                                "drawing code, which cannot be made to report, so the binding still reports "
                                "what is raised in that window",
}
# NOT in that table, and read like every other file: widgets/live_hair_bundle.py, the live simulation's
# plotting widget, which stands inside the main window. A method this program writes in its class carries
# the guard. What no scan of this program's text can reach is the code the class inherits: the paint and
# resize events of the plotting library's view and its items' painting are that library's own, written in
# Python, and cannot be made to report. An error raised there during a repaint is dropped by the binding
# whenever another method written in Python is entered after it in the same handling, and is still reported
# when it was the last.
# The methods of those names that carry no guard, each with its reason: what must not have one, and any
# helper that only shares its name with a method of the toolkit, which the toolkit therefore never calls.
_NOT_GUARDED = {
    "main_window.py: MainWindow.event": "what keeps the interpreter while the window handles an event: it only "
                                        "calls the base class, and a fallback would handle the event a second time",
}


def _built_on(cls, rel, modules, seen=frozenset()) -> str:
    """What a class is built on, as far as the text says: "toolkit" when a class of the toolkit is among its
    bases, "plain" when none can be, and else "cannot follow <base>" for the first base, as it is written,
    that the scan cannot place. ``modules`` holds every parsed file of the program by its path, ``rel`` is
    the file of ``cls``.

    A base is the toolkit's when its last name is a capital Q and a capital letter with more behind them, or
    when the file imports the name it begins with from the toolkit. It is plain when that name comes from the
    standard library or is a built-in. Any other bare name is taken for a class of the program and looked
    for by that name alone, in the same file first and else in every file; each class found is followed in
    the same way, to any depth, and one of them built on the toolkit is enough. Whatever is left (a name no
    file defines, a class of another library, a call) cannot be followed."""
    placed = {}
    for node in ast.walk(modules[rel]):
        if isinstance(node, ast.Import):
            bound = [(alias.asname or alias.name.partition(".")[0], alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            bound = [(alias.asname or alias.name, node.module or "") for alias in node.names]
        else:
            continue
        for name, module in bound:
            root = module.partition(".")[0]
            if root in ("PySide6", "shiboken6") or root in sys.stdlib_module_names:
                placed[name] = "toolkit" if root in ("PySide6", "shiboken6") else "plain"
    doubt = None
    for base in cls.bases:
        first = base
        while isinstance(first, ast.Attribute):
            first = first.value
        first = getattr(first, "id", "")     # the name the base begins with, as it is written
        last = base.attr if isinstance(base, ast.Attribute) else first
        if re.fullmatch(r"Q[A-Z]\w+", last) or placed.get(first) == "toolkit":
            return "toolkit"
        if placed.get(first) == "plain" or (isinstance(base, ast.Name) and hasattr(builtins, first)):
            continue
        named = [(where, node) for where, tree in modules.items() for node in ast.walk(tree)
                 if isinstance(base, ast.Name) and isinstance(node, ast.ClassDef) and node.name == first]
        named = [one for one in named if one[0] == rel] or named
        kinds = [_built_on(node, where, modules, seen | {(rel, cls.lineno)}) for where, node in named
                 if (where, node.lineno) not in seen]
        if "toolkit" in kinds:
            return "toolkit"
        if set(kinds) != {"plain"}:
            doubt = doubt or next((kind for kind in kinds if kind != "plain"), f"cannot follow {ast.unparse(base)}")
    return doubt or "plain"


def _drawing_methods(tree, rel="", modules=None) -> list:
    """Every place in a parsed module where a name the toolkit calls while a window is drawn or laid out is
    bound, as (line, name, guarded, fallback). A method defined in a class is "Class.method": guarded when it
    stands under ``@reported(...)`` and the module imports that name from the module called ``reported``;
    ``fallback`` is the guard's argument as written, None when it has none. One of the twelve names bound any
    other way (assigned in a class body, or set on a class or an object afterwards) is listed as it is
    written and is never guarded: the binding takes that for an override as well.

    One of the nine further names counts only as a method, and only in a class built on the toolkit
    (``_built_on``, which needs ``modules``, every parsed file of the program by its path, and ``rel``, the
    path of this one; without them the module is read by itself). In a plain class it is left alone. In a
    class whose base cannot be followed it is listed with that base in its name, so that the doubt is seen:
    it passes only under the guard."""
    modules = modules or {rel: tree}
    imported = any(isinstance(node, ast.ImportFrom) and (node.module or "").rpartition(".")[2] == "reported"
                   and any(alias.name == "reported" and alias.asname is None for alias in node.names)
                   for node in ast.walk(tree))
    own_scope = (ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    found = []
    for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        body = list(cls.body)                # the class's own statements, through its ifs and its tries,
        while body:                          # and not into the functions and the classes it defines
            node = body.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                built_on = _built_on(cls, rel, modules) if node.name in _CALLED_IN_A_TOOLKIT_CLASS else "plain"
                if node.name in _CALLED_WHILE_DRAWING or built_on != "plain":
                    guards = [d for d in node.decorator_list if isinstance(d, ast.Call)
                              and isinstance(d.func, ast.Name) and d.func.id == "reported"]
                    handed = guards[0].args + [keyword.value for keyword in guards[0].keywords] if guards else []
                    guarded = imported and bool(guards)
                    doubt = "" if guarded or built_on in ("plain", "toolkit") else f" ({built_on})"
                    found.append((node.lineno, f"{cls.name}.{node.name}{doubt}", guarded,
                                  ast.unparse(handed[0]) if handed else None))
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store) and node.id in _CALLED_WHILE_DRAWING:
                    found.append((node.lineno, f"{cls.name}.{node.id}", False, None))
            elif not isinstance(node, own_scope) and not (isinstance(node, ast.AnnAssign) and node.value is None):
                body.extend(ast.iter_child_nodes(node))
    found += [(node.lineno, ast.unparse(node), False, None) for node in ast.walk(tree)
              if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
              and node.attr in _CALLED_WHILE_DRAWING]
    return sorted(found)


def test_every_method_the_toolkit_calls_while_drawing_reports_its_errors():
    """Every ``.py`` under core/gui is read, but the files of ``_NOT_GUARDED_FILES``: a method of one of the
    twelve names, defined in a class, stands under the guard of core/gui/reported.py, all but the one of
    ``_NOT_GUARDED``; and so does a method of one of the nine further names in a class built on the toolkit.
    Of those nine the scan does not see a method written in a plain class that a class built on the toolkit
    only mixes in, nor one assigned rather than defined. The scanner is checked on snippets first: with the
    tree clean, the walk would go on passing if the scanner stopped matching anything."""
    def scan(source, **beside):              # ``beside``: other files of the program, by name, as their text
        modules = {"scanned.py": ast.parse(source)}
        modules.update({f"{name}.py": ast.parse(text) for name, text in beside.items()})
        return [(name, guarded, fallback) for _line, name, guarded, fallback
                in _drawing_methods(modules["scanned.py"], "scanned.py", modules)]

    head = "from .reported import reported\n"
    for guard, fallback in (("@reported()", None), ("@reported(QLabel.resizeEvent)", "QLabel.resizeEvent"),
                            ("@reported(fallback=False)", "False")):
        for start in (head, "from ..reported import reported\n", "from core.gui.reported import reported\n"):
            source = start + f"class Ring(QWidget):\n    {guard}\n    def paintEvent(self, event): pass\n"
            assert scan(source) == [("Ring.paintEvent", True, fallback)], source
    for name in sorted(_CALLED_WHILE_DRAWING):               # each of the twelve names, with no guard
        source = head + f"class Bare(QWidget):\n    def {name}(self, *told): pass\n"
        assert scan(source) == [(f"Bare.{name}", False, None)], name
    nine = ("paint", "paintSection", "pixmap", "data", "headerData", "flags", "rowCount", "columnCount", "showEvent")
    assert set(nine) == _CALLED_IN_A_TOOLKIT_CLASS and len(nine) == 9        # typed out: none can drop out unseen
    for name in nine:                                        # each of the nine: refused bare, passed guarded,
        bare = head + f"class Bare(QWidget):\n    def {name}(self, *told): pass\n"
        kept = head + f"class Kept(QWidget):\n    @reported(QWidget.{name})\n    def {name}(self, *told): pass\n"
        assert scan(bare) == [(f"Bare.{name}", False, None)], name
        assert scan(kept) == [(f"Kept.{name}", True, f"QWidget.{name}")], name
        assert scan(head + f"class Plain:\n    def {name}(self, *told): pass\n") == [], name     # and left alone
    # which classes those nine count in: one built on the toolkit, as far as the text says
    model = "    def data(self, index, role): pass\n"
    rows = "class Row(QWidget): pass\n"
    for built in ("class Bare(QAbstractTableModel):\n", "class Bare(QtCore.QAbstractTableModel):\n",
                  "from PySide6.QtCore import QAbstractTableModel as Table\nclass Bare(Mixin, Table):\n",
                  "import PySide6.QtCore as kit\nclass Bare(kit.Model):\n",
                  rows + "class Bare(Row):\n",                                  # through a class of the program,
                  rows + "class Wide(Row): pass\nclass Bare(Other, Wide):\n"):   # to any depth
        assert scan(built + model) == [("Bare.data", False, None)], built
    in_another_file = "from .rows import Row\nclass Bare(Row):\n" + model
    assert scan(in_another_file, rows=rows) == [("Bare.data", False, None)]
    assert scan(in_another_file, rows="class Row: pass\n", more=rows) == [("Bare.data", False, None)]    # one is enough
    assert scan("class Row: pass\nclass Bare(Row):\n" + model, rows=rows) == []     # the one in the same file is meant
    for plain in ("class Plain:\n", "class Plain(object):\n", "class Plain(dict):\n",
                  "import logging\nclass Plain(logging.Handler):\n", "from enum import Enum\nclass Plain(Enum):\n",
                  "class Other: pass\nclass Plain(Other):\n", "class Queue: pass\nclass Plain(Queue):\n"):
        assert scan(plain + model) == [], plain
    for unknown, base in (("import pyqtgraph as pg\nclass Bare(pg.GraphicsLayoutWidget):\n", "pg.GraphicsLayoutWidget"),
                          ("class Bare(Elsewhere):\n", "Elsewhere"), ("class Bare(make()):\n", "make()"),
                          ("from .other import Base\nclass Row(Base): pass\nclass Bare(Row):\n", "Base")):
        assert scan(unknown + model) == [(f"Bare.data (cannot follow {base})", False, None)], unknown   # refused
        assert scan(head + unknown + "    @reported(None)\n" + model) == [("Bare.data", True, "None")], unknown
    for bare in ("class Bare(QWidget):\n    @reported\n    def paintEvent(self, event): pass\n",       # not called
                 "class Bare(QWidget):\n    @other.reported()\n    def paintEvent(self, event): pass\n",
                 "class Bare(QWidget):\n    @Slot()\n    def paintEvent(self, event): pass\n",
                 "class Bare(QWidget):\n    if True:\n        def paintEvent(self, event): pass\n",
                 "class Bare(QWidget):\n    try:\n        paintEvent = draw\n    finally:\n        pass\n",
                 "def make():\n    class Bare(QWidget):\n        def paintEvent(self, event): pass\n",
                 "class Outer:\n    class Bare(QWidget):\n        async def paintEvent(self, event): pass\n"):
        assert scan(head + bare) == [("Bare.paintEvent", False, None)], bare
    not_imported = "class Bare(QWidget):\n    @reported()\n    def paintEvent(self, event): pass\n"
    for start in ("", "from .reported import expected\n", "from .other import reported\n",
                  "from .reported import guard as reported\n"):
        assert scan(start + not_imported) == [("Bare.paintEvent", False, None)], start
    assert scan("Window.event = handle\n") == [("Window.event", False, None)]            # set on afterwards
    assert scan("class Box:\n    def build(self):\n        self.sizeHint = lambda: size\n") == [
        ("self.sizeHint", False, None)]
    spared = (head + "class Plain(QWidget):\n    event: object\n    kinds = [event for event in ()]\n"
              "    flags = 0\n    def closeEvent(self, event): pass\n    def keyPressEvent(self, event): pass\n"
              "    def update(self):\n        event = self.event(sent)\n        self.data = event\n"
              "        return event\n"
              "def paintEvent(): pass\ndef paint(): pass\nhandled = window.event(sent)\n")
    assert scan(spared) == []                # of the nine, a method alone counts: not a value of that name

    gui = REPO / "core" / "gui"
    modules = {path.relative_to(gui).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
               for path in sorted(gui.rglob("*.py"))}

    def walk(modules, not_guarded=_NOT_GUARDED):
        """The program's files walked: every method found with its fallback, the ones refused as
        "file:line Class.method", and the entries of ``not_guarded`` that were met."""
        found, bare, excepted = {}, [], set()
        for rel, tree in modules.items():
            if rel in _NOT_GUARDED_FILES:
                continue
            for line, name, guarded, fallback in _drawing_methods(tree, rel, modules):
                key = f"{rel}: {name.partition(' (')[0]}"
                found[key] = fallback
                if not guarded and key in not_guarded:
                    excepted.add(key)
                elif not guarded:
                    bare.append(f"{rel}:{line} {name}")
        return found, bare, excepted

    def refused(modules, not_guarded=_NOT_GUARDED):
        return [entry.partition(" ")[2] for entry in walk(modules, not_guarded)[1]]

    # The program's own files first: a refusal there is told with its file, its line and the two ways out.
    found, bare, excepted = walk(modules)
    listed = f"all {len(found)} found:\n  " + "\n  ".join(f"{name} -> {fallback}" for name, fallback in found.items())
    assert bare == [], (
        f"called by the toolkit while a window is drawn or laid out, and not reporting its own errors: {bare}. "
        f"Two ways out. Give it the guard: put it under @reported(...) of core/gui/reported.py, with what it "
        f"answers when it fails. Or, if the toolkit does not call it (a helper that only shares the name of "
        f"one of the toolkit's methods), name it in _NOT_GUARDED in this file, with the reason. {listed}")
    # The live simulation's plotting widget is read like any other file: what this program writes in that
    # class carries the guard. Shown on the file's own text, with a method written into the class in memory.
    live = "widgets/live_hair_bundle.py"
    text = (gui / live).read_text(encoding="utf-8").splitlines(keepends=True)
    view, = [node for node in ast.walk(modules[live]) if isinstance(node, ast.ClassDef)
             and node.name == "LiveHairBundleView"]

    def with_in_the_view(method, start=""):
        lines = text[:view.end_lineno] + ["\n", method] + text[view.end_lineno:]
        return {**modules, live: ast.parse(start + "".join(lines))}

    paints = "    def paintEvent(self, event):\n        super().paintEvent(event)\n"
    assert refused(with_in_the_view(paints)) == ["LiveHairBundleView.paintEvent"]
    assert refused(with_in_the_view("    @reported()\n" + paints, "from ..reported import reported\n")) == []
    # one of the nine there: the class is built on the plotting library's, which the scan cannot place, so
    # the method is refused with that said until it is guarded
    shows = "    def showEvent(self, event):\n        super().showEvent(event)\n"
    assert refused(with_in_the_view(shows)) == ["LiveHairBundleView.showEvent (cannot follow pg.GraphicsLayoutWidget)"]
    # the other way out of a refusal, for a method of such a name that the toolkit does not call
    table = ast.parse("class Table(QWidget):\n    def data(self): return self.rows\n")
    helper = {**modules, "widgets/helper.py": table}
    assert refused(helper) == ["Table.data"]
    assert refused(helper, {**_NOT_GUARDED, "widgets/helper.py: Table.data": "the widget's own rows"}) == []

    assert {name: found.get(name, "not found") for name in _GUARDED_TODAY} == _GUARDED_TODAY, listed
    missing = [rel for rel in _NOT_GUARDED_FILES if not (gui / rel).is_file()] + sorted(set(_NOT_GUARDED) - excepted)
    assert not missing, f"no longer there, or guarded now, so it stops being an exception: {missing}"


def test_the_looks_overrides_fall_back_to_the_stock_answer_when_they_fail(monkeypatch, ring_room):
    """An override of the small style that fails answers, and draws, what Fusion itself does: a tick box is
    still a tick box. The ring's paint draws nothing and ends its painter; the follower's filter answers
    that the event is not taken. Nothing comes out, and each failure is recorded under its method's name."""
    from PySide6.QtWidgets import QListWidget, QListWidgetItem
    from core.gui import design, paint, reported, style
    r = ring_room
    worn, stock = style.PrismStyle(), QStyleFactory.create("Fusion")
    wheel, width = QStyle.SH_ComboBox_AllowWheelScrolling, QStyle.PM_IndicatorWidth

    def fail(*told, **named):
        raise RuntimeError("made to fail")

    def tick_box():                          # a set tick box as it paints itself, and where its indicator is
        box = QCheckBox("words")
        box.setStyle(worn)                   # on this one widget, never on the application
        box.setChecked(True)
        box.ensurePolished()
        box.resize(box.sizeHint())
        option = QStyleOptionButton()
        option.initFrom(box)
        where = box.style().subElementRect(QStyle.SE_CheckBoxIndicator, option, box)
        picture = _image(box.width(), box.height(), 1.5)
        box.render(picture)
        return picture, where

    def indicator(by):                       # the tick box's indicator alone, as one style draws it
        option = QStyleOptionButton()
        option.rect = QRect(0, 0, 18, 18)
        option.state = QStyle.State_Enabled | QStyle.State_On
        picture = _image(18, 18, 1.0)
        painter = QPainter(picture)
        try:
            by.drawPrimitive(QStyle.PE_IndicatorCheckBox, option, painter, None)
        finally:
            painter.end()
        return picture

    def names(caught):
        return [name for name, _words in caught]

    with text_size(1.0), reported.expected() as caught:
        picture, where = tick_box()
        own, fusion = indicator(worn), indicator(stock)
        assert _at(picture, where.x() + 2.5, where.y() + 9) == design.hex_colour("accent")
        assert own != fusion and worn.styleHint(wheel) == 0 != stock.styleHint(wheel), "the lines below prove nothing"
        assert worn.pixelMetric(width) == 18 != stock.pixelMetric(width), "the lines below prove nothing"
        assert caught == []

        with monkeypatch.context() as patch:                 # the indicator's paint fails
            patch.setattr(paint, "paint_indicator", fail)
            picture, where = tick_box()
            assert names(caught) == ["core.gui.style.PrismStyle.drawPrimitive"] and reported.ERRORS == caught
            middle = picture.pixelColor(int((where.x() + 9) * 1.5), int((where.y() + 9) * 1.5))
            assert middle.alpha() == 255, "nothing was drawn where the box stands"
            assert _at(picture, where.x() + 2.5, where.y() + 9) != design.hex_colour("accent")
            assert indicator(worn) == fusion                 # the stock box, to the pixel
        with monkeypatch.context() as patch:                 # the size of the indicator fails
            patch.setattr(design, "scaled", fail)
            assert worn.pixelMetric(width) == stock.pixelMetric(width)
            assert worn.pixelMetric(width, None, r.button) == stock.pixelMetric(width, None, r.button)
        with monkeypatch.context() as patch:                 # the hints fail at their first comparison
            patch.setattr(style, "QStyle", SimpleNamespace())
            assert worn.styleHint(wheel) == stock.styleHint(wheel)
            assert worn.styleHint(wheel, None, r.box, None) == stock.styleHint(wheel, None, r.box, None)
        # each failed twice: listed the first time, counted both
        small = "core.gui.style.PrismStyle."
        assert names(caught) == [small + "drawPrimitive", small + "pixelMetric", small + "styleHint"]
        assert reported.COUNTS == dict.fromkeys(names(caught), 2)

        rows = QListWidget()                 # two ticked rows and no words: one painter paints them both
        rows.setStyle(worn)                  # on this one widget, never on the application
        for _row in range(2):
            QListWidgetItem(rows).setCheckState(Qt.Checked)
        rows.setFixedSize(120, 80)
        rows.ensurePolished()
        with monkeypatch.context() as patch:                 # a colour fails after the indicator saved the painter
            patch.setattr(design, "colour", fail)
            picture = _image(rows.width(), rows.height(), 1.0)
            rows.render(picture)

        def under(row):                      # the pixels under one row; its place is told inside the list's frame
            box = rows.visualItemRect(rows.item(row)).translated(rows.frameWidth(), rows.frameWidth())
            return [picture.pixel(x, y) for y in range(box.top(), box.bottom() + 1)
                    for x in range(box.left(), box.right() + 1)]

        assert under(1) == under(0) and len(set(under(1))) > 1, "the row after a failed tick box is not drawn as the first"

        _arrive(r.button, Qt.TabFocusReason)
        ring = r.follows.ring()
        with monkeypatch.context() as patch:                 # the ring's colour fails, under its live painter
            patch.setattr(design, "colour", fail)
            picture = _image(ring.width(), ring.height(), 1.0)
            ring.render(picture, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
            assert picture == _image(ring.width(), ring.height(), 1.0)
            assert not ring.paintingActive() and not picture.paintingActive()
        assert names(caught)[3:] == ["core.gui.style.FocusRing.paintEvent"]
        assert reported.COUNTS[names(caught)[3]] == 1
        _leave(r.button, Qt.TabFocusReason)

        with monkeypatch.context() as patch:                 # the follower fails where the focus arrives
            patch.setattr(r.follows, "arrived", fail)
            assert r.follows.eventFilter(r.button, QFocusEvent(QEvent.FocusIn, Qt.TabFocusReason)) is False
            _arrive(r.button, Qt.TabFocusReason)             # and as the toolkit calls it: nothing comes out
        assert names(caught)[4:] == ["core.gui.style.FocusFollower.eventFilter"] and r.follows.target() is None
        assert reported.COUNTS[names(caught)[4]] == 2


def test_a_test_in_which_drawing_code_failed_is_failed(request, monkeypatch):
    """A guarded method keeps its error from the test that made it draw, so that test would pass over it.
    The session's fixture empties the record before every test and reads it after, and fails the test when
    it holds anything, naming each method with its first error and how often it failed. The fixture's own
    function is driven here as the test runner drives it: what stands before its ``yield``, a test's worth
    of drawing, then the rest."""
    from core.gui import reported
    # tests/ is no package, so pytest holds its conftest under a name of its own; imported by name the
    # file would be loaded a second time, beside the one whose fixtures are in force
    here = Path(__file__).resolve().with_name("conftest.py")
    guards = next(plugin for plugin in request.config.pluginmanager.get_plugins()
                  if getattr(plugin, "__file__", None) and Path(plugin.__file__).resolve() == here)
    assert "_drawing_code_raises_nothing" in request.fixturenames, "in force for every test, without being asked for"
    around_a_test = guards._drawing_code_raises_nothing.__wrapped__      # the function the fixture was made from

    class Drawn:
        @reported.reported()
        def paintEvent(self):
            raise ValueError("no colour")

    monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: None)      # quiet
    drawn, name = Drawn(), _name_of(Drawn.paintEvent)
    try:
        drawn.paintEvent()                   # left by whatever ran before the test
        run = around_a_test()
        next(run)                            # before the test: the record is emptied
        assert reported.ERRORS == [] and reported.COUNTS == {}
        assert drawn.paintEvent() is None and drawn.paintEvent() is None         # the test: drawing code fails twice
        with pytest.raises(AssertionError) as failed:        # after the test
            next(run)
        for needle in (f"{name}: ValueError: no colour (2 time(s))", "reported.expected()"):
            assert needle in str(failed.value), str(failed.value)
        assert reported.ERRORS == [] and reported.COUNTS == {}                   # and nothing is left for the next one

        run = around_a_test()                # a test in which drawing code failed only where the test said it would
        next(run)
        with reported.expected():
            drawn.paintEvent()
        with pytest.raises(StopIteration):   # the function ends, and nothing is raised
            next(run)
    finally:
        reported.forget()


# ── the tabs and the main button of the old layout, as the toolkit draws them under the sheet ────
def _dressed(dark):
    """A holder that wears one theme's sheet and is never shown; nothing is set on the application. Whoever
    builds a widget in it gives that widget its own font: a holder with a sheet of its own does not hand its
    font down. Given to the holder after its sheet, a font reaches no child; given before, it reaches the
    children built until the holder is polished, which the first drawing of any child does, and none after."""
    QApplication.instance() or QApplication([])
    holder = QWidget()
    holder.setStyleSheet(design.build_qss(dark))
    return holder


def _over_paper(widget, dark, ratio=4.0):
    """A widget drawn on the page's colour, at a display scaling at which the stroke of a letter covers whole
    device pixels. The widget's own window ground is left out: it would come from a palette the holder does
    not hand down, in a grey of the toolkit's own."""
    from PySide6.QtGui import QColor
    picture = _image(widget.width(), widget.height(), ratio)
    picture.fill(QColor(design.hex_colour("paper", dark)))
    widget.render(picture, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return picture


def _lettering(picture, box):
    """What stands in a box of a picture: its commonest colour (the ground), the colour farthest from that
    (the core of the letters), both as the look's table writes a colour, and how wide the letters stand, in
    logical pixels. A box with nothing in it gives its ground, None and 0.0."""
    import collections
    ratio = picture.devicePixelRatio()
    part = picture.copy(QRect(round(box.x() * ratio), round(box.y() * ratio), round(box.width() * ratio),
                              round(box.height() * ratio))).convertToFormat(QImage.Format_RGB32)
    values, across = memoryview(part.constBits()).cast("I"), part.width()
    seen = collections.Counter(values)

    def apart(one, other):                   # how far two pixel values stand from each other, over the three channels
        return sum(abs((one >> shift & 255) - (other >> shift & 255)) for shift in (16, 8, 0))

    def written(value):
        return f"#{value & 0xFFFFFF:06X}"

    ground = seen.most_common(1)[0][0]
    core = max(seen, key=lambda value: apart(value, ground))
    if core == ground:
        return written(ground), None, 0.0
    inked = {value for value in seen if 2 * apart(value, ground) > apart(core, ground)}
    columns = [index % across for index, value in enumerate(values) if value in inked]
    return written(ground), written(core), (max(columns) - min(columns) + 1) / ratio


def _tab_row(dark):
    """A stock tab bar under one theme's sheet, with its holder, which must outlive it: the first tab is the
    one in front, the second can be opened, the last two are locked."""
    from PySide6.QtWidgets import QTabBar
    holder = _dressed(dark)
    bar = QTabBar(holder)
    bar.setFont(fonts.inter("body"))
    for word in ("Config", "Prior", "Posterior", "Validate"):
        bar.addTab(word)
    for locked in (2, 3):
        bar.setTabEnabled(locked, False)
    bar.ensurePolished()
    bar.resize(bar.sizeHint())
    return holder, bar


def test_an_open_tab_is_lettered_in_the_full_text_colour_and_a_locked_one_in_the_greyed_colour():
    """A tab that can be opened is written in the full text colour, in front or not, and a locked one in the
    greyed colour, which is dark enough to be read: written in the secondary text colour, an open tab stood so
    near a locked one that the two could not be told apart at a glance. The tab in front is then marked by its
    line in the accent alone, and no other tab has that line. Read off a stock tab bar as it paints itself
    under the built sheet, in both themes."""
    for dark in (False, True):
        ink, greyed, accent, paper = (design.hex_colour(name, dark)
                                      for name in ("ink", "disabled_ink", "accent", "paper"))
        holder, bar = _tab_row(dark)
        picture = _over_paper(bar, dark)
        boxes = [bar.tabRect(index) for index in range(bar.count())]
        words = [_lettering(picture, box.adjusted(2, 2, -2, -4))[1] for box in boxes]     # clear of the foot line
        assert words == [ink, ink, greyed, greyed], (dark, words)
        feet = [_at(picture, box.center().x(), box.bottom()) for box in boxes]
        assert feet == [accent, paper, paper, paper], (dark, feet)


def test_the_pointer_gives_an_open_tab_a_ground_and_a_locked_tab_nothing():
    """An open tab's word is in the full text colour already, so its colour cannot answer the pointer: the tab
    takes the ground an ordinary button takes under the pointer. A locked tab keeps its look there, because the
    sheet's rule for the pointer is never given to a greyed tab (the bar does give the pointer's place to it), and the tab
    in front keeps its ground and its line. The pointer here is a hover event handed to a bar that is never shown."""
    from PySide6.QtGui import QHoverEvent

    def pointed(bar, index, dark):           # a tab as the bar draws it with the pointer over its middle
        box = bar.tabRect(index)
        middle = QPointF(box.center())
        QApplication.sendEvent(bar, QHoverEvent(QEvent.HoverMove, middle, middle, QPointF(-1, -1)))
        assert bar.tabAt(middle.toPoint()) == index, (dark, index, "the bar gives that point to another tab")
        picture = _over_paper(bar, dark)
        ground, word, _wide = _lettering(picture, box.adjusted(2, 2, -2, -4))
        return ground, word, _at(picture, box.center().x(), box.bottom())

    for dark in (False, True):
        ink, greyed, accent, paper, under = (design.hex_colour(name, dark) for name in
                                             ("ink", "disabled_ink", "accent", "paper", "hover_fill"))
        assert under != paper, "the pointer's ground is the page's colour: the lines below prove nothing"
        holder, bar = _tab_row(dark)
        assert pointed(bar, 1, dark) == (under, ink, under), (dark, "a tab that can be opened")
        assert pointed(bar, 2, dark) == (paper, greyed, paper), (dark, "a locked tab")
        left = _lettering(_over_paper(bar, dark), bar.tabRect(1).adjusted(2, 2, -2, -4))[0]
        assert left == paper, (dark, "the pointer left the open tab for the locked one")
        assert pointed(bar, 0, dark) == (paper, ink, accent), (dark, "the tab in front")
        rules = {tuple(selectors): declarations for selectors, declarations in _rules(design.build_qss(dark))}
        assert rules[("QTabBar::tab:hover",)] == {"color": ink, "background": under}, dark
        assert "background" not in rules[("QTabBar::tab:disabled",)], dark
        assert pointed(bar, 1, dark)[0] == under, (dark, "the pointer back on the open tab")
        QApplication.sendEvent(bar, QHoverEvent(QEvent.HoverLeave, QPointF(-1, -1), QPointF(-1, -1), QPointF(1, 1)))
        gone = _lettering(_over_paper(bar, dark), bar.tabRect(1).adjusted(2, 2, -2, -4))
        assert gone[:2] == (paper, ink), (dark, "the pointer gone")


# the one rule with a state that may name a weight: the greyed main button of the old layout
_GREYED_MAIN = 'QPushButton[accent="true"]:disabled'


def _named_in_a_state(selectors, declarations):
    """What a rule names of a face, a size or a weight where it may name none: an empty dict for a rule with no
    state, and for the one rule with a state that may name a weight, as long as it names the ordinary weight
    and nothing else. A control is measured once, at the face of its plain rule; a state that names a larger
    size or a heavier weight draws words wider than what was measured, and they are cut. The greyed main button
    of the old layout is the exception because its plain rule is the bold one: its lighter words are narrower
    than what was measured and cannot be cut."""
    if not any(re.search(r"(?<!:):(?!:)", selector) for selector in selectors):
        return {}
    named = {name: declarations[name] for name in ("font", "font-size", "font-weight") if name in declarations}
    return {} if (selectors, named) == ([_GREYED_MAIN], {"font-weight": "400"}) else named


def test_one_greyed_button_alone_may_name_a_weight_in_a_state_and_only_the_ordinary_weight():
    """No rule with a state names a face, a size or a weight: the words would be drawn wider than the control
    was measured for, and cut. One rule may, because it can cut nothing: the greyed main button of the old
    layout names the ordinary weight, 400, where its plain rule is the bold one. The exception is that one
    rule with that one value. It covers no other rule and no other value, it is not kept for a rule that does
    not use it, and it goes when the old main button does."""
    def refused(qss):                        # the rules of a sheet that the check does not let pass
        return [selectors for selectors, declarations in _rules(qss) if _named_in_a_state(selectors, declarations)]

    assert _GREYED_MAIN in design.INTERIM_SELECTORS, "the old main button is gone: the exception made for it goes too"
    for dark in (False, True):
        qss = design.build_qss(dark)
        rules = {tuple(selectors): declarations for selectors, declarations in _rules(qss)}
        weight = rules[(_GREYED_MAIN,)].get("font-weight")
        assert weight == "400", (dark, weight, "the excepted rule does not use its exception")
        assert refused(qss) == [], dark
    # a weight, a size or a face in any state is still refused; so is the excepted rule with any other value,
    # and the excepted value in any other rule: a kept one, or the excepted rule widened by a second selector
    for sample in ("QTabBar::tab:selected { font-weight: 600; }", "QPushButton:hover { font-size: 15px; }",
                   "QLineEdit:disabled { font: 14px Inter; }", _GREYED_MAIN + " { font-weight: 600; }",
                   _GREYED_MAIN + " { font-weight: 400; font-size: 12px; }",
                   'QPushButton[cls="primary"]:disabled { font-weight: 400; }',
                   "QPushButton:disabled, " + _GREYED_MAIN + " { font-weight: 400; }"):
        assert refused(sample) != [], sample
    assert refused(_GREYED_MAIN + " { font-weight: 400; }") == []
    assert refused('QPushButton[cls="primary"] { font-weight: 600; }') == [], "a rule with no state may name a weight"


def test_a_greyed_main_button_of_the_old_layout_is_lettered_in_the_ordinary_weight():
    """Greyed, a main button of the old layout has its words in the ordinary weight and so looks as every
    other greyed button does: bold words on a greyed button read as a button one may press. It is bold again
    the moment it can be pressed, and it keeps one size throughout, because a button is measured once, at the
    weight of its plain rule. That is why this one rule may name a weight for a state: its plain rule is the
    bold one, so the lighter words are narrower than what was measured and cannot be cut. A button of the two
    classes that stay is bold when greyed, as before: those classes are for the pages built the new way, where
    a greyed main button stands beside a sentence that says why it cannot be pressed, which an old form does
    not have. Read off stock buttons as they paint themselves under the built sheet, in both themes: how wide
    the words stand tells the two weights apart."""
    QApplication.instance() or QApplication([])
    words, wide, high = "Build / Load prior", 200, 34
    with text_size(1.0):
        needs = {weight: QFontMetricsF(fonts.inter("body", weight)).horizontalAdvance(words)
                 for weight in (400, 600)}       # the room the words take at each weight
        assert needs[600] - needs[400] > 6, "the two weights stand too near for the lines below to tell them apart"

        def button(holder, usable=False, **properties):    # greyed unless asked, with its own font and its class
            made = QPushButton(words, holder)
            made.setFont(fonts.inter("body"))
            for name, value in properties.items():
                made.setProperty(name, value)
            made.setEnabled(usable)
            made.ensurePolished()
            return made

        def drawn(made, dark):               # at one size for all, so that two pictures can be laid on each other
            made.resize(wide, high)
            picture = _over_paper(made, dark)
            stands = _lettering(picture, QRect(4, 4, wide - 8, high - 8))[2]
            near = [weight for weight, room in needs.items() if abs(room - stands) < 3]
            return picture, (near[0] if len(near) == 1 else None)

        for dark in (False, True):
            holder = _dressed(dark)
            plain, old = button(holder), button(holder, accent=True)
            asked = old.sizeHint()
            assert asked.width() > plain.sizeHint().width(), (dark, "measured for the bold words of its plain rule")
            # a button answers a change of state from the size it stored: only one measured usable from the start can
            # show that both states are measured alike
            assert button(holder, usable=True, accent=True).sizeHint() == asked, (dark, "measured usable")
            ordinary, weight = drawn(plain, dark)
            assert weight == 400, (dark, "a greyed ordinary button", weight)
            for cls in ("primary", "danger"):                # the kept classes are bold when greyed too
                assert drawn(button(holder, cls=cls), dark)[1] == 600, (dark, cls, "greyed")
            greyed, weight = drawn(old, dark)
            assert (weight, greyed == ordinary) == (400, True), (dark, "the old main button, greyed", weight)
            old.setEnabled(True)
            assert drawn(old, dark)[1] == 600 and old.sizeHint() == asked, (dark, "the old main button, usable")
            old.setEnabled(False)
            assert drawn(old, dark)[0] == greyed and old.sizeHint() == asked, (dark, "greyed again")
            assert old.font().weight() == QFont.Weight.DemiBold, "its own font is the bold one in every state"


def test_the_follower_hears_an_event_and_never_takes_it(ring_room):
    """The follower's filter stands before every event of the program, and an event it took would end
    there: a field would not learn that it holds the keyboard, a key would type nothing, a click would
    press nothing, and a control would not learn that it was hidden. So on each road of the filter -- a
    key pressed, the focus arriving, the focus leaving the control that shows it, that control hidden, a
    mouse button pressed -- and for a kind of event it does nothing with, the event reaches what it was
    sent to. Each road is also shown to have been taken, by what the follower did on it."""
    class Control(QPushButton):              # a control that notes the kind of every event that reaches it
        def __init__(self, *args):
            super().__init__(*args)
            self.kinds = []

        def event(self, event):
            self.kinds.append(event.type())
            return super().event(event)

    class Elsewhere(QObject):                # the same for an object that is no control: a press on it moves nothing
        def __init__(self):
            super().__init__()
            self.kinds = []

        def event(self, event):
            self.kinds.append(event.type())
            return super().event(event)

    def reached(what, kind):                 # whether an event of that kind came through to it since last asked
        came = kind in what.kinds
        del what.kinds[:]
        return came

    r = ring_room
    control, elsewhere = Control("Check again", r.window), Elsewhere()
    control.setGeometry(220, 20, 120, 34)
    QApplication.sendEvent(elsewhere, QKeyEvent(QEvent.KeyPress, Qt.Key_Tab, Qt.NoModifier))
    assert reached(elsewhere, QEvent.KeyPress), "the follower took a key press"
    _moved_on(control, Qt.TabFocusReason)
    assert r.follows.target() is control, "the key press was not heard: the focus it moved brought no ring"
    assert reached(control, QEvent.FocusIn), "the follower took the focus on its way to the control"
    _leave(control, Qt.TabFocusReason)
    assert r.follows.target() is None, "the focus leaving the control was not heard"
    assert reached(control, QEvent.FocusOut), "the follower took the focus on its way out of the control"
    _moved_on(control, Qt.TabFocusReason)
    assert r.follows.target() is control, "the focus coming back brought no ring: the next lines prove nothing"
    control.hide()
    assert r.follows.target() is None, "the control being hidden was not heard"
    assert reached(control, QEvent.Hide), "the follower took the word that the control was hidden"
    QApplication.sendEvent(elsewhere, QMouseEvent(QEvent.MouseButtonPress, QPointF(1, 1), QPointF(1, 1),
                                                  Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    assert reached(elsewhere, QEvent.MouseButtonPress), "the follower took a mouse press"
    _moved_on(r.button, Qt.MouseFocusReason)         # a click ends the keyboard's turn; and with a mouse button
    _moved_on(r.button, Qt.TabFocusReason)           # the last thing pressed, no key moved the focus on
    assert r.follows.target() is None, "the mouse press was not heard: the key before it still counted"
    QApplication.sendEvent(elsewhere, QEvent(QEvent.User))
    assert reached(elsewhere, QEvent.User), "the follower took an event it does nothing with"


def test_the_ring_keeps_its_two_pixels_and_its_corner_at_every_text_size(ring_room):
    """The ring is a line, the gap to its control and a corner, and the look's table names all three among
    the lengths that do not grow with the text. Painted at the ordinary size the line is two pixels deep,
    the corner is clear, and the top line is whole from seven pixels in: the control's own corner of
    three, the gap's two and the line's two. At two larger text sizes the line is two pixels deep still,
    and the whole picture is the one painted at the ordinary size."""
    r = ring_room

    def painted():                           # the ring round the button, as it paints itself at one to one
        _arrive(r.button, Qt.TabFocusReason)
        ring = r.follows.ring()
        picture = _image(ring.width(), ring.height(), 1.0)
        ring.render(picture, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
        _leave(r.button, Qt.TabFocusReason)
        return picture

    def cover(picture, x, y):                # how solid the pixel at a place is, 0 to 255
        return picture.pixelColor(x, y).alpha()

    with text_size(1.0):
        ordinary = painted()
    middle = ordinary.width() // 2
    assert [cover(ordinary, middle, y) for y in range(4)] == [255, 255, 0, 0], \
        "the ring's line is not two pixels at the ordinary text size"
    assert [cover(ordinary, at, at) for at in (0, 1)] == [0, 0], "the ring's corner is not clear: it is not round"
    whole = [x for x in range(ordinary.width()) if cover(ordinary, x, 0) == 255]    # the top line where it is whole
    assert (whole[0], whole[-1]) == (7, ordinary.width() - 8), \
        "the ring's corner is not the control's own, seen across the gap"
    for factor in (1.25, 2.25):
        with text_size(factor):
            larger = painted()
        assert [cover(larger, middle, y) for y in range(4)] == [255, 255, 0, 0], \
            f"the ring's line is not two pixels at a text factor of {factor}"
        assert larger == ordinary, f"the ring is not the same picture at a text factor of {factor}"


def test_the_ring_is_seen_over_its_control_and_outside_the_group_that_holds_it(ring_room):
    """The toolkit asks the style whether the focus frame lies above its control. Told no, it makes the
    frame a child of the control's own parent and paints it beneath the control: the ring just inside the
    edge of a text box is then painted over by the box, and the ring round a control at the edge of a
    group is cut off where the group ends. Neither is seen. So the window is painted with the ring on
    such a control, and the pixel where the ring's line must show is the focus colour: outside the group
    for the one, just inside the box's own edge for the other."""
    from core.gui import design
    r = ring_room
    group = QWidget(r.window)                # a group in the window's free corner, and a control flush with its edge
    group.setGeometry(220, 20, 180, 60)
    held = QPushButton("In a group", group)
    held.setGeometry(0, 0, 120, 34)

    def painted_at(x, y):                    # one pixel of the window, as it paints itself with all it holds
        picture = _image(r.window.width(), r.window.height(), 1.0)
        r.window.render(picture)
        return _at(picture, x, y)

    with theme(False):
        focus = design.hex_colour("focus")
        for control, x, y in ((held, group.x() - 3.5, group.y() + 17),      # the ring's left line, outside the group
                              (r.box, r.box.x() + 0.5, r.box.y() + r.box.height() / 2)):      # and just inside the box
            assert painted_at(x, y) != focus, "the ground there is the ring's colour: the next line proves nothing"
            _arrive(control, Qt.TabFocusReason)
            assert painted_at(x, y) == focus, f"the ring is not seen on a {type(control).__name__}"
            _leave(control, Qt.TabFocusReason)


def test_a_stopped_follower_is_off_the_application_and_hears_nothing_more():
    """Stopping takes the follower's filter off the application itself. Its own word that it listens no
    longer is not enough: a filter left on is still called for every event of the program, for as long as
    any name holds the follower, whatever it then does with the event. So the follower here notes every
    call of its filter. Stopped, it shows the focus nowhere; and when the focus then arrives by the
    keyboard, as it did before the stop, its filter is not called at all."""
    from core.gui import style
    calls = []

    class Noting(style.FocusFollower):       # a follower that notes the kind of every event its filter is handed
        def eventFilter(self, obj, event):
            calls.append(event.type())
            return super().eventFilter(obj, event)

    app = QApplication.instance() or QApplication([])
    window = QWidget()
    button = QPushButton("Check again", window)
    button.setGeometry(40, 40, 120, 34)
    follows = Noting(app, style.PrismStyle())
    try:
        _arrive(button, Qt.TabFocusReason)
        assert calls and follows.target() is button, \
            "the follower heard nothing while it listened: the next lines prove nothing"
        follows.stop()
        assert follows.target() is None, "a stopped follower still shows the focus where it was"
        del calls[:]                         # what it was handed up to the stop, and by the stop itself
        _arrive(button, Qt.TabFocusReason)
        assert calls == [], "a stopped follower's filter is still on the application"
    finally:
        follows.stop()


# ── painted devices: markers, glyphs, icons, ruled lines, the hatching ─────────────────────────
_SCALINGS = (1.0, 1.25, 1.5, 2.0)                    # the display scalings a device is drawn at
_BLANK = QColor(255, 0, 255)                         # a ground no colour of the look can be taken for
_BARE = _BLANK.name().upper()                        # that ground, as the colour of a pixel is named
_HEAVY_RULE = {1.0: 2, 1.25: 2, 1.5: 3, 2.0: 4}      # a 2 px rule in device pixels at each scaling
_HAIR_RULE = {1.0: 1, 1.25: 1, 1.5: 1, 2.0: 2}       # a 1 px rule
# Where each shape shows which colour, and where it leaves the ground bare, in the 16-unit box a
# marker is drawn in. Every point lies on the middle of a stroke or well inside a fill.
_MARK_SHOWS = {
    "done": ((8, 2.5, "ink"), (9.25, 8.2, "paper")), "now": ((8, 8, "accent"),),
    "later": ((14.42, 9.04, "ink_3"),), "lock": ((8, 11, "ink_2"), (8, 2, "ink_2")),
    "half": ((5, 8, "ink"), (11, 8, "paper"), (8, 1, "ink")), "none": ((8, 8, "ink_3"),),
    "fine": ((8, 1, "ok"), (9.1, 8.2, "ok")), "info": ((8, 1, "ink_2"), (8, 4.9, "ink_2"), (8, 9.6, "ink_2")),
    "warn": ((5, 12.5, "warn"), (8, 8, "warn_mark"), (8, 12.1, "warn_mark"), (8, 14.3, "warn_edge")),
    "err": ((2, 8, "err"), (8, 8, "on_err"), (5.5, 5.5, "on_err")), "stop": ((8, 8, "err"),),
    "chev_right": ((10.5, 8, "ink"),), "chev_down": ((8, 10.5, "ink"),), "chev_up": ((8, 5.5, "ink"),),
    "help": ((8, 1, "ink"), (8, 12, "ink")),
}
_MARK_BARE = {
    "done": ((0.5, 0.5),), "now": ((1.5, 1.5),), "later": ((8, 8), (13.78, 10.97)), "lock": ((8, 5.5),),
    "half": ((0.5, 0.5),), "none": ((8, 4),), "fine": ((10, 12),), "info": ((5, 8),), "warn": ((2, 3),),
    "err": ((0.1, 0.1),), "chev_right": ((4, 8), (12.5, 8)), "chev_down": ((8, 5.5),),
    "chev_up": ((8, 10.5),), "help": ((3.5, 8),), "stop": ((1.5, 8),),
}
_MARK_INNER = ("paper", "on_err", "warn_mark")       # the marks that stand ON a filled shape


def _screen(width, height, ratio, ground=_BLANK, draw=None):
    """A picture standing in for a screen at one scaling, on a ground of one colour, with
    ``draw(painter)`` painted into it. The painter is ended whatever ``draw`` does: a picture that
    goes while a painter still paints on it ends the whole run, where the test should only fail."""
    image = _image(width, height, ratio)             # its size is rounded UP there: rounded down, a last row is cut
    image.fill(ground)
    if draw is not None:
        painter = QPainter(image)
        try:
            draw(painter)
        finally:
            painter.end()
    return image


def _screen_of(widget, ratio, ground=_BLANK):
    """A widget painted into such a picture, with no ground of its own under it."""
    image = _screen(widget.width(), widget.height(), ratio, ground)
    widget.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image


def _every_pixel(image):
    return [image.pixelColor(x, y) for y in range(image.height()) for x in range(image.width())]


def _run_of(image, colour, *, column=None, row=None):
    """Where one colour's device pixels lie down one column, or along one row."""
    if column is not None:
        return [y for y in range(image.height()) if image.pixelColor(column, y) == colour]
    return [x for x in range(image.width()) if image.pixelColor(x, row) == colour]


def _device_edge(value):
    return int(math.floor(value + 0.5))


def test_the_ten_states_and_the_five_glyphs_are_named_once():
    assert paint.MARKERS == ("done", "now", "later", "lock", "half", "none", "fine", "info", "warn", "err")
    assert paint.GLYPHS == ("chev_right", "chev_down", "chev_up", "help", "stop")
    assert set(_MARK_SHOWS) == set(_MARK_BARE) == set(paint.MARKERS + paint.GLYPHS)


def test_each_marker_and_glyph_shows_its_colours_in_both_themes_at_four_scalings(monkeypatch):
    QApplication.instance() or QApplication([])

    def draw(painter):
        paint.paint_marker(painter, kind, QRectF(4, 4, 64, 64))                  # 4 px to a unit
        assert painter.transform().isIdentity() and not painter.testRenderHint(QPainter.Antialiasing)   # left as found

    for dark, ratio, kind in itertools.product((False, True), _SCALINGS, paint.MARKERS + paint.GLYPHS):
        with theme(dark):
            image = _screen(72, 72, ratio, draw=draw)
            for x, y, name in _MARK_SHOWS[kind]:
                assert _at(image, 4 + 4 * x, 4 + 4 * y) == design.hex_colour(name), (kind, dark, ratio, x, y, name)
            for x, y in _MARK_BARE[kind]:
                assert _at(image, 4 + 4 * x, 4 + 4 * y) == _BARE, (kind, dark, ratio, x, y)

    def fails(painter):                      # no colour to be had: the paint fails once the painter is moved, scaled and smoothing
        with pytest.raises(KeyError):
            paint.paint_marker(painter, "done", QRectF(4, 4, 64, 64))
        assert painter.transform().isIdentity() and not painter.testRenderHint(QPainter.Antialiasing), "a paint that failed did not leave the painter as it found it"

    with monkeypatch.context() as patch:
        patch.setattr(design, "COLOURS", {})
        _screen(72, 72, 1.0, draw=fails)


def test_a_disabled_marker_keeps_its_inner_mark():
    """Drawn disabled, a shape takes the colour of disabled text and the mark that stands on it stays
    the page's colour: a tick, a cross or a half must not vanish into its shape."""
    QApplication.instance() or QApplication([])
    for dark, ratio, kind in itertools.product((False, True), _SCALINGS, paint.MARKERS + paint.GLYPHS):
        with theme(dark):
            image = _screen(72, 72, ratio, draw=lambda p: paint.paint_marker(p, kind, QRectF(4, 4, 64, 64), disabled=True))
            for x, y, name in _MARK_SHOWS[kind]:
                want = "paper" if name in _MARK_INNER else "disabled_ink"
                assert _at(image, 4 + 4 * x, 4 + 4 * y) == design.hex_colour(want), (kind, dark, ratio, x, y, name)
            assert design.hex_colour("paper") != design.hex_colour("disabled_ink")


def test_a_marker_stays_inside_its_box_and_an_unknown_kind_paints_nothing():
    QApplication.instance() or QApplication([])
    for ratio, kind in itertools.product(_SCALINGS, paint.MARKERS + paint.GLYPHS + ("tick",)):
        image = _screen(24, 24, ratio, draw=lambda p: paint.paint_marker(p, kind, QRectF(4, 4, 16, 16)))   # as on a page
        low, high = int(4 * ratio), int(20 * ratio)
        touched = [(x, y) for y in range(image.height()) for x in range(image.width())
                   if image.pixelColor(x, y) != _BLANK]
        assert all(low <= x < high and low <= y < high for x, y in touched), (kind, ratio)
        assert bool(touched) == (kind != "tick"), (kind, ratio)


def test_fill_px_lays_whole_device_pixels_from_the_edge_named_at_every_position(monkeypatch):
    QApplication.instance() or QApplication([])
    ink = QColor(26, 26, 26)
    # the four scalings, and two that lie between and beyond them: a weight is rounded DOWN to whole
    # device pixels there too, and is never none
    heavy, hair = {**_HEAVY_RULE, 1.75: 3, 2.5: 5}, {**_HAIR_RULE, 1.75: 1, 2.5: 2}
    bar = {1.0: 4, 1.25: 5, 1.5: 6, 1.75: 7, 2.0: 8, 2.5: 10}

    def smoothed(painter):                   # a caller's painter may be smoothing: the rule is the same whole pixels
        painter.setRenderHint(QPainter.Antialiasing, True)
        paint.fill_px(painter, rect, ink, **{side: weight})
        assert painter.testRenderHint(QPainter.Antialiasing), "the fill did not leave the painter's smoothing as it found it"

    for ratio, (weight, want), position, side in itertools.product(
            sorted(bar), ((2, heavy), (1, hair), (4, bar)), range(24), ("top", "bottom", "left", "right")):
        across = side in ("top", "bottom")
        rect = QRectF(5, position, 30, 11) if across else QRectF(position, 5, 11, 30)
        image = _screen(48, 48, ratio, draw=lambda p: paint.fill_px(p, rect, ink, **{side: weight}))
        middle = int(20 * ratio)
        run = _run_of(image, ink, column=middle) if across else _run_of(image, ink, row=middle)
        first = _device_edge(position * ratio) if side in ("top", "left") else _device_edge((position + 11) * ratio) - want[ratio]
        assert run == list(range(first, first + want[ratio])), (ratio, weight, side, position, run)
        assert _screen(48, 48, ratio, draw=smoothed) == image, (ratio, weight, side, position)      # pixel for pixel

    def frame(painter):
        painter.setRenderHint(QPainter.Antialiasing, True)      # a caller's painter may be smoothing
        painter.translate(0, 0.3)                               # and may stand between two pixels
        paint.fill_px(painter, QRectF(5, 5, 30, 20), ink, top=1, bottom=1, left=1, right=1)

    for ratio in sorted(bar):
        pixels = _every_pixel(_screen(48, 48, ratio, draw=frame))
        assert {pixel.rgb() for pixel in pixels} == {ink.rgb(), _BLANK.rgb()}, ratio        # no blended pixel
        n, wide, high = hair[ratio], _device_edge(35 * ratio) - _device_edge(5 * ratio), _device_edge(25.3 * ratio) - _device_edge(5.3 * ratio)
        assert pixels.count(ink) == wide * high - (wide - 2 * n) * (high - 2 * n), ratio     # each corner once, the inside bare

    def smaller(painter):
        painter.scale(0.4, 0.4)                                 # a drawing made smaller: 1 px is 0.4 of a device pixel
        paint.fill_px(painter, QRectF(10, 10, 60, 40), ink, top=1, left=1)
        assert not painter.testRenderHint(QPainter.Antialiasing), "the fill switched a plain painter's smoothing on"

    small = _screen(48, 48, 1.0, draw=smaller)
    assert (_run_of(small, ink, column=10), _run_of(small, ink, row=10)) == ([4], [4])   # one device pixel each, never none
    nothing = _screen(48, 48, 1.5, draw=lambda p: paint.fill_px(p, QRectF(5, 5, 30, 20), ink))
    assert all(pixel == _BLANK for pixel in _every_pixel(nothing))                               # no edge named, nothing painted

    def fails(painter):                      # no rectangle to be made: the fill fails once the painter's smoothing is off
        painter.setRenderHint(QPainter.Antialiasing, True)
        with pytest.raises(TypeError):
            paint.fill_px(painter, QRectF(5, 5, 30, 20), ink, top=1)
        assert painter.testRenderHint(QPainter.Antialiasing), "a fill that failed did not leave the painter's smoothing as it found it"

    with monkeypatch.context() as patch:
        patch.setattr(paint, "QRectF", None)
        _screen(48, 48, 1.0, draw=fails)


def test_the_hatching_stays_in_its_rectangle_and_does_not_move_with_it(monkeypatch):
    QApplication.instance() or QApplication([])
    for dark, ratio in itertools.product((False, True), _SCALINGS):
        with theme(dark):
            paper, hatch = design.colour("paper"), design.colour("hatch")

            def hatched(painter):
                paint.paint_hatch(painter, QRectF(10, 4, 212, 12), hatch)
                assert not painter.hasClipping() and not painter.testRenderHint(QPainter.Antialiasing)   # left as found

            image = _screen(240, 20, ratio, paper, hatched)
            left, right, top, bottom = int(10 * ratio), math.ceil(222 * ratio), int(4 * ratio), math.ceil(16 * ratio)
            assert all(image.pixelColor(x, y) == paper for y in range(image.height()) for x in range(image.width())
                       if not (left <= x < right and top <= y < bottom)), (dark, ratio)
            row = [image.pixelColor(x, int(10 * ratio)).red() for x in range(image.width())]
            strokes = sum(1 for i, red in enumerate(row) if red != paper.red() and (i == 0 or row[i - 1] == paper.red()))
            assert strokes in (29, 30, 31), (dark, ratio, strokes)  # one every 5 px ACROSS the stroke: 212 / (5 x root 2) = 29.98
            low, high = sorted((paper.red(), hatch.red()))
            assert all(low - 1 <= red <= high + 1 for red in row)   # the two colours and their mixtures, nothing else
            assert max(abs(red - paper.red()) for red in row) >= 0.9 * (high - low), (dark, ratio)
            # A range is hatched twice on the time axis, ahead of the elapsed bar and through it: the strokes
            # of two rectangles with one top and one height fall on each other wherever both are.
            wide, narrow = (_screen(240, 20, ratio, paper, lambda p: paint.paint_hatch(p, QRectF(x0, 4, 160 - x0, 12), hatch))
                            for x0 in (10, 47))
            worst = max(abs(wide.pixelColor(x, y).red() - narrow.pixelColor(x, y).red())
                        for y in range(int(4 * ratio) + 1, int(16 * ratio) - 1)
                        for x in range(math.ceil(48 * ratio) + 1, int(160 * ratio) - 1))
            assert worst <= 8, (dark, ratio, worst)     # smoothing rounds by a level or two; a moved stroke by over a hundred
    for width in (0, -5):
        empty = _screen(40, 20, 1.0, draw=lambda p: paint.paint_hatch(p, QRectF(10, 4, width, 12), QColor(77, 77, 77)))
        assert all(pixel == _BLANK for pixel in _every_pixel(empty))

    def clipped(painter):                    # a clip the caller set stands: the hatching is cut to both
        painter.setClipRect(QRectF(0, 0, 30, 20))
        paint.paint_hatch(painter, QRectF(10, 4, 100, 12), QColor(77, 77, 77))
        assert painter.hasClipping() and painter.clipBoundingRect() == QRectF(0, 0, 30, 20), "the hatching did not leave its caller's clip as it found it"

    for ratio in _SCALINGS:
        kept, edge = _screen(120, 20, ratio, draw=clipped), math.ceil(30 * ratio)
        columns = [x for y in range(kept.height()) for x in range(kept.width()) if kept.pixelColor(x, y) != _BLANK]
        assert sum(x >= edge for x in columns) == 0, (ratio, "hatched outside the clip its caller set")
        assert columns and min(columns) >= int(10 * ratio), (ratio, "not hatched inside its rectangle and the clip its caller set")

    def fails(painter):                      # no hatch step to be had: the paint fails once the painter is clipped and smoothing
        with pytest.raises(KeyError):
            paint.paint_hatch(painter, QRectF(10, 4, 20, 12), QColor(77, 77, 77))
        assert not painter.hasClipping() and not painter.testRenderHint(QPainter.Antialiasing), "a paint that failed did not leave the painter as it found it"

    with monkeypatch.context() as patch:
        patch.setattr(design, "SIZE", {name: size for name, size in design.SIZE.items() if name != "hatch_step"})
        _screen(40, 20, 1.0, draw=fails)


def test_a_ruled_block_paints_whole_rules_from_its_own_edges_in_both_themes_at_four_scalings():
    QApplication.instance() or QApplication([])
    for dark, ratio in itertools.product((False, True), _SCALINGS):
        with theme(dark):
            block, child = Ruled(top=2, bottom=2), QWidget()        # what a block holds never covers its rules
            child.setAutoFillBackground(True)
            look = child.palette()
            look.setColor(child.backgroundRole(), QColor(0, 255, 0))
            child.setPalette(look)
            block.box.addWidget(child)
            block.resize(60, 31)                                    # 31: at 1.25 and 1.5 its foot falls inside a pixel
            block.layout().activate()
            column, n, foot = int(30 * ratio), _HEAVY_RULE[ratio], _device_edge(31 * ratio)
            assert _run_of(_screen_of(block, ratio), design.colour("rule"), column=column) == [*range(n), *range(foot - n, foot)], (dark, ratio)
            quiet = Ruled(top=1, colour="rule_light")
            quiet.resize(60, 31)
            assert _run_of(_screen_of(quiet, ratio), design.colour("rule_light"), column=column) == [*range(_HAIR_RULE[ratio])], (dark, ratio)
    spaced = Ruled(top=2, bottom=1, gap_top=7, gap_bottom=3)    # held in a name: its layout goes when it does
    room = spaced.box.contentsMargins()
    assert (room.left(), room.top(), room.right(), room.bottom(), spaced.box.spacing()) == (0, 9, 0, 4, 0)
    with pytest.raises(ValueError):                             # a colour is named when the block is built, not when it paints
        Ruled(top=1, colour="grey")
    blocks = [widget for widget in QApplication.topLevelWidgets() if isinstance(widget, Ruled)]
    try:
        Ruled(top=1, colour="grey")
    except ValueError:                       # while the error is held, whatever was built before it was raised is held with it
        assert sum(isinstance(widget, Ruled) for widget in QApplication.topLevelWidgets()) == len(blocks), "a block was built before its colour was refused"
    plain = Ruled()
    plain.resize(20, 20)
    assert all(pixel == _BLANK for pixel in _every_pixel(_screen_of(plain, 1.0)))


def test_a_section_heading_is_sixteen_pixels_semibold_over_a_heavy_rule_six_pixels_under_it():
    QApplication.instance() or QApplication([])
    with text_size(1.0):                                        # the sizes below are those of the ordinary text size
        head = SectionHead("Main settings")
        font = head.label.font()
        assert (head.label.text(), font.pixelSize(), font.weight()) == ("Main settings", 16, QFont.Weight.DemiBold)
        assert head.label.property("cls") == "head" and head.label.textFormat() == Qt.PlainText   # a name may hold "<" or "&"
        assert head.sizeHint().height() == 22 + 6 + 2           # the heading's line, the gap, the rule
        head.resize(600, 30)
        head.layout().activate()
        for dark, ratio in itertools.product((False, True), _SCALINGS):
            with theme(dark):
                image, n, foot = _screen_of(head, ratio), _HEAVY_RULE[ratio], _device_edge(30 * ratio)
                # read at the far right, where no letter stands: in the light theme a letter has the rule's colour
                assert _run_of(image, design.colour("rule"), column=image.width() - 3) == [*range(foot - n, foot)], (dark, ratio)
        aside = QWidget()                                       # of a fixed size: no width of lettering is read
        aside.setFixedSize(40, 12)
        head.add_right(aside)
        head.layout().activate()
        assert (aside.geometry().right(), aside.geometry().bottom()) == (599, 21)     # at the right, on the heading's foot
        head.set_text("This training")
        assert head.label.text() == "This training"
    with text_size(1.25):                                       # lengths grow with the text; the rule does not
        assert SectionHead("Main settings").sizeHint().height() == 28 + 8 + 2


def test_a_marker_widget_is_sixteen_pixels_grows_with_the_text_and_reads_the_theme_when_it_paints():
    QApplication.instance() or QApplication([])
    with text_size(1.0):                                        # the sizes below are those of the ordinary text size
        dot = Marker("now")
        assert (dot.width(), dot.height(), dot.kind) == (16, 16, "now")
        assert Marker("warn", design.DIALOG["marker"]).width() == 24
        assert design.hex_colour("accent", False) != design.hex_colour("accent", True)
        for ratio, dark in itertools.product(_SCALINGS, (False, True)):
            with theme(dark):
                image = _screen_of(dot, ratio)                          # the SAME widget: it keeps no colour
                assert (_at(image, 8, 8), _at(image, 0.3, 0.3)) == (design.hex_colour("accent"), _BARE), (ratio, dark)
        dot.set_kind("stop")
        assert dot.kind == "stop" and _at(_screen_of(dot, 2.0), 8, 8) == design.hex_colour("err")
        dot.setEnabled(False)                                   # greyed with the form it stands in
        assert _at(_screen_of(dot, 1.0), 8, 8) == design.hex_colour("disabled_ink")
        for unknown in ("tick", "", None):
            with pytest.raises(ValueError):
                Marker(unknown)
        with pytest.raises(ValueError):
            dot.set_kind("tick")
    with text_size(1.25):                                       # a length grows with the text
        assert Marker("done").width() == 20 and Marker("warn", design.DIALOG["marker"]).width() == 30


def test_a_painted_icon_is_sharp_at_every_scaling_follows_the_theme_and_greys_when_disabled():
    QApplication.instance() or QApplication([])
    square = {1.0: 100, 1.25: 144, 1.5: 196, 2.0: 400}          # the 10-unit square of "stop" in whole device pixels
    icon = painted_icon("stop")
    gc.collect()                                                # the engine lives on with no name of its own
    assert not icon.isNull()
    for dark, ratio in itertools.product((False, True), _SCALINGS):
        with theme(dark):
            pixmap = icon.pixmap(QSize(16, 16), ratio)
            assert (pixmap.width(), pixmap.devicePixelRatio()) == (math.ceil(16 * ratio), ratio)
            picture = pixmap.toImage()
            assert picture.pixelColor(0, 0).alpha() == 0 and _every_pixel(picture).count(design.colour("err")) == square[ratio]
            for mode, name in ((QIcon.Normal, "err"), (QIcon.Disabled, "disabled_ink")):
                shade = design.hex_colour(name)
                wide = _screen(32, 24, ratio, draw=lambda p: icon.paint(p, QRect(4, 4, 20, 16), Qt.AlignCenter, mode, QIcon.Off))
                tall = _screen(32, 24, ratio, draw=lambda p: icon.paint(p, QRect(4, 2, 16, 20), Qt.AlignCenter, mode, QIcon.Off))
                # a wider icon keeps the glyph at its left, so the room before a button's words stays bare: the square
                # of "stop" runs from 7 to 17 there; centred or at the right, the point at 8 would be bare
                assert [_at(wide, x, 12) for x in (5, 8, 16, 18)] == [_BARE, shade, shade, _BARE], (dark, ratio, name)
                # a taller one centres it down its height: at the head or at the foot one of the two inner points is bare
                assert [_at(tall, 12, y) for y in (6, 8, 16, 18)] == [_BARE, shade, shade, _BARE], (dark, ratio, name)
    with theme(False):
        button = QPushButton()                                  # no words: no pixel of lettering is read
        button.setIcon(icon)
        button.setIconSize(QSize(20, 16))
        button.resize(60, 34)
        assert _every_pixel(_screen_of(button, 1.0, QColor(255, 255, 255))).count(design.colour("err")) == 100
        button.setEnabled(False)
        greyed = _every_pixel(_screen_of(button, 1.0, QColor(255, 255, 255)))
        assert greyed.count(design.colour("err")) == 0 and greyed.count(design.colour("disabled_ink")) >= 100
    with pytest.raises(ValueError):
        painted_icon("tick")


def test_a_marker_and_a_ruled_block_whose_paint_fails_draw_nothing_end_their_painter_and_report_it(monkeypatch):
    """What the two paint events answer when a colour fails under their painter: nothing is drawn, nothing
    comes out of the call that made the widget draw, and the error stands in the record under the paint
    event's own name. Each has ended its painter by then: the words of the error, which are read while the
    failed call still stands, say whether its painter still paints. The next paint is a whole one."""
    from core.gui import reported
    from core.gui.widgets import marker, ruled
    QApplication.instance() or QApplication([])
    with text_size(1.0):                                        # the marker's middle is read at the ordinary text size
        dot, block = Marker("now"), Ruled(top=2)
    block.resize(60, 31)
    opened = []                                                 # each painter a paint event opens, noted and not kept alive

    class Noted(QPainter):
        def __init__(self, device):
            super().__init__(device)
            opened.append(weakref.ref(self))

    class NoColour(Exception):                                  # its words are made when the record reads them
        def __str__(self):
            return f"its painter still paints: {opened[-1]().isActive()}"

    def fail(*told, **named):
        raise NoColour()

    with reported.expected() as caught, monkeypatch.context() as patch:
        patch.setattr(marker, "QPainter", Noted)
        patch.setattr(ruled, "QPainter", Noted)
        patch.setattr(design, "colour", fail)                   # each paint fails under its live painter
        for widget in (dot, block):
            assert all(pixel == _BLANK for pixel in _every_pixel(_screen_of(widget, 1.0))), type(widget).__name__
    assert caught == [("core.gui.widgets.marker.Marker.paintEvent", "NoColour: its painter still paints: False"),
                      ("core.gui.widgets.ruled.Ruled.paintEvent", "NoColour: its painter still paints: False")]
    assert _at(_screen_of(dot, 1.0), 8, 8) == design.hex_colour("accent")       # and each paints whole again
    assert _run_of(_screen_of(block, 1.0), design.colour("rule"), column=30) == [0, 1]


def test_a_painted_icon_never_ends_the_process():
    """In an interpreter of its own, because each fault here ends the whole process, which would be
    the test run itself. When a copy of an icon is changed the toolkit asks the icon's engine for a
    twin, and a twin that nothing keeps alive ends the process. So does a picture that is let go while
    the engine's painter still paints on it, as after an error raised under that painter; the toolkit
    says so before it happens, and whatever it says about painting is printed. No error comes out of the
    engine: each is recorded under its method's name, and the toolkit is answered a clear picture."""
    code = "\n".join((
        "import gc",
        "from PySide6.QtCore import QRect, QSize, qInstallMessageHandler",
        "from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap",
        "from PySide6.QtWidgets import QApplication",
        "app = QApplication([])",
        "qInstallMessageHandler(lambda kind, context, text: print('the toolkit said:', text) if 'paint' in text.lower() else None)",
        "from core.gui import paint, reported",
        "from core.gui.widgets import marker",
        "icon = marker.painted_icon('stop'); copy = QIcon(icon); copy.addPixmap(QPixmap(8, 8)); gc.collect()",
        "image = QImage(32, 32, QImage.Format_ARGB32_Premultiplied); image.fill(0); painter = QPainter(image)",
        "icon.paint(painter, QRect(0, 0, 16, 16)); copy.paint(painter, QRect(16, 16, 16, 16)); painter.end()",
        "print(image.pixelColor(8, 8).name(), image.pixelColor(24, 24).name())",
        "del icon, copy; gc.collect()",
        "def broken(*args, **kwargs): raise KeyError('a colour the table lacks')",
        "with reported.expected() as caught:",
        "    paint.paint_marker = broken",
        "    first = marker.painted_icon('stop').pixmap(QSize(16, 16), 1.0)",
        "    marker.PaintedIcon.paint = broken",
        "    second = marker.painted_icon('stop').pixmap(QSize(16, 16), 1.0)",
        "    gc.collect()",
        "    print(*(name for name, _words in caught))",
        "    print(first.width(), first.toImage().pixelColor(8, 8).alpha(), second.width(), second.toImage().pixelColor(8, 8).alpha())",
        "gc.collect()"))
    done = subprocess.run([sys.executable, "-u", "-c", code], cwd=str(REPO),
                          env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
                          capture_output=True, text=True, timeout=120)     # -u: what was printed before a fault is kept
    vermilion, engine = design.hex_colour("err", False).lower(), "core.gui.widgets.marker.PaintedIcon."
    assert (done.returncode, done.stdout.split()) == (
        0, [vermilion, vermilion, engine + "paint", engine + "pixmap", "16", "0", "16", "0"]), \
        (done.returncode, done.stdout, done.stderr[-2000:])
