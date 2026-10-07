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
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
from PySide6.QtGui import QFont, QFontDatabase, QFontInfo, QFontMetricsF, QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QStyleFactory

from core.gui import design, fonts

REPO = Path(__file__).resolve().parents[1]
# every module of the look that exists; a new one joins this tuple in the commit that adds it, and the
# fresh-interpreter test is then its proof that it loads no engine
LOOK_MODULES = ("core.gui.design", "core.gui.fonts", "core.gui.theming", "core.gui.mpl_theme", "core.gui.settings",
                "core.gui.file_chooser", "core.gui.widgets.labeled_inputs")
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
    dark. The module's active Appearance is put back afterwards."""
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

        def styleHints(self):
            return self.hints

        def setPalette(self, palette):
            self.palettes.append(palette)

        def setStyleSheet(self, sheet):
            self.sheets.append(sheet)

    active = theming._ACTIVE
    try:
        yield Application()
    finally:
        theming._ACTIVE = active


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


# ── lettering: Inter with digits of equal width, the book face, the fixed-width face ─────────────
INTER_KINDS = ("small", "body", "strong", "head", "brand", "title")
CLASSES_WITH_A_FONT = ("QAbstractItemView", "QHeaderView", "QMessageBox", "QMenu", "QMenuBar", "QTipLabel",
                       "QStatusBar")
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


def test_every_inter_font_has_digits_of_equal_width_and_so_does_a_copy():
    QApplication.instance() or QApplication([])
    assert design.LETTERING == "Inter" and design.text_scale() == 1.0
    for kind in INTER_KINDS:
        size, _line, weight = design.TYPE[kind]
        font = fonts.inter(kind)
        assert (font.family(), font.pixelSize(), font.weight()) == ("Inter", size, QFont.Weight(weight)), kind
        assert QFontInfo(font).family() == "Inter", "the bundled files are registered on first use"
        assert font.isFeatureSet(_tabular()) and font.featureValue(_tabular()) == 1, kind
        copy = QFont(font)
        copy.setPixelSize(size + 3)
        assert copy.isFeatureSet(_tabular()), kind
        assert len({QFontMetricsF(font).horizontalAdvance(digit) for digit in "0123456789"}) == 1, kind
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
        assert (font.family(), font.weight(), font.isFeatureSet(_tabular())) == ("Inter", QFont.Weight.Normal, True)
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
            font = QApplication.font(name)
            assert (font.family(), font.pixelSize(), font.isFeatureSet(_tabular())) == ("Inter", 14, True), name
        app.setStyle(QStyleFactory.create(style))               # what the first style sheet does too
        assert QApplication.font("QMenu") == app.font(), "a change of style empties the per-class fonts"
        assert not QApplication.font("QMenu").isFeatureSet(_tabular())
        fonts.apply_class_fonts()                               # no argument: the running application
        menu = QMenu()
        menu.ensurePolished()                                   # a widget takes its class font when polished
        assert (menu.font().family(), menu.font().pixelSize()) == ("Inter", 14)
    assert QApplication.font("QMenu") == app.font() == found


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
    assert len({QFontMetricsF(fonts.inter("body")).horizontalAdvance(digit) for digit in "0123456789"}) == 1
    wanted = fonts.book_family(QFontDatabase.families())
    assert QFontInfo(fonts.book()).family() == wanted
    assert fonts.book().pixelSize() == 16 and fonts.book(bold=True).bold()
    console = QFontInfo(fonts.fixed())
    assert console.fixedPitch() and console.pixelSize() == 14
    assert console.family() in (*design.FIXED_FACES,
                                QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family())
    with font_and_style_put_back(app) as style:
        fonts.load_app_font(app)
        app.setStyle(QStyleFactory.create(style))
        assert QApplication.font("QMenu").family() != "Inter", "Windows' own font is back on a menu"
        fonts.apply_class_fonts(app)
        assert {QApplication.font(name).family() for name in CLASSES_WITH_A_FONT} == {"Inter"}
