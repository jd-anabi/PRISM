"""The window's places -- the rail, ``go_to`` and the run's foot -- and Home.

Home is built as the window builds it, through the builders of tests/_fixtures.py, and handed what it states by a
function each test makes: a session is the window's own session class (tests/_window_helpers.py), and the
steps' states are the rules' own. The window's reading of its session, its run and the record store in use, and
what each of Home's places opens in the window, are tested on a whole window.
"""
import contextlib
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeySequence, QTextLayout
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from core.gui import chain, design, keys, record_words, this_computer
from core.gui.main_window import MainWindow
from core.gui.run.controller import controller
from core.gui.screens.home_screen import HomeFacts, HomeScreen, RecordRow
from core.gui.widgets.page import Columns, PageHead, Sentence
from core.gui.widgets.rule_table import StateCell, TextCell
from tests._fixtures import build_part, build_run_host, build_window, qt_app
from tests._runs import RunText, StandIn, running, start_run, turn_until
from tests._window_helpers import (given as _given, in_front as _in_front, press as _press, session as _session,
                                   shown_off_screen as _shown_off_screen)

HEAD = ("Home", "What to do next, where your work stands, and what each part is for.")
FIRST_HEAD = ("What do you want to do?",
              "PRISM simulates models of the hair bundle, the motion sensor of the inner ear's hair cells, and infers "
              "a cell's model parameters from recordings. Nothing has been made yet.")
CHOICES = [("&Watch a live simulation", "Needs nothing first. See a cell's bundle move as it is simulated.",
            ("live", None)),
           ("Begin parameter inference", "Six steps, each needing the one before. It starts with choosing a model.",
            ("inference", "model")),
           ("Measure an effective temperature", "On one simulated cell, or in a sweep study over two parameters.",
            ("temperature", None)),
           ("Read what each part is for", "A short section per part.", ("help", None))]
PARTS = [(place.name, place.line) for place in keys.PARTS[1:]]
# The window's own way to the front, read as this file is loaded: in a test not marked for the real screen a guard
# stands in for it.
COME_TO_FRONT = MainWindow.come_to_front
# The taskbar's flash as its module defines it, read the same way: the guard of the requests made of Windows stands in
# for it.
FLASH = this_computer.flash


def _under_way():
    """A model chosen and a prior in use: the training is the step in hand."""
    return _session(draft=SimpleNamespace(model="NADROWSKI"), inf_prior=SimpleNamespace(name="retrain_prior"))


def _facts(session=None, records=(), run=None, has_runs=False, runs=()):
    """What Home is handed: the steps' states worked out from the session as the rules work them out, the run's
    status among them."""
    session = _session() if session is None else session
    place = run.place if run is not None and run.live else None
    live = place[1] if place is not None and place[0] == "inference" else None
    return HomeFacts(session, chain.step_states(session, live=live), run, tuple(records), has_runs, tuple(runs))


def _live(word="Training", name="", place=("inference", "training"), **more):
    """The status of a run live at ``place``, as the run's controller publishes it."""
    from core.gui.run.model import RunStatus
    return RunStatus(state="running", word=word, name=name, place=place, **more)


def _records():
    """Seven records of six kinds, made over two years, and one whose moment is not known."""
    at = lambda *when: datetime(*when, tzinfo=timezone.utc)
    return [RecordRow("prior", "p1", "retrain_prior", "prior", at(2025, 3, 1, 15, 29)),
            RecordRow("diagnostic", "d1", "retrain_band", "check", at(2025, 3, 1, 15, 30)),
            RecordRow("simulation", "s1", "Simulation set for retrain_prior", "simulation set", at(2025, 3, 2, 18, 46)),
            RecordRow("posterior", "q1", "an_old_posterior", "posterior", at(2024, 1, 5, 9, 0)),
            RecordRow("fdt", "f1", "sweep-s", "sweep", at(2025, 3, 3, 11, 2)),
            RecordRow("calibration", "c1", "Unnamed calibration, 3 Mar 2025, 14:02", "calibration", None),
            RecordRow("observation", "o1", "cell_7", "observation", at(2025, 2, 20, 8, 5))]


def _home(facts):
    """Home, handed ``facts`` each time it reads."""
    return build_part(HomeScreen, read=lambda: facts)


def _cells(table):
    """A table's rows as its cells hold them: a text's words, a state's marker and word."""
    model = table.view.model()
    return [[(cell.kind, cell.word) if isinstance(cell, StateCell) else cell.text
             for cell in (model.index(row, column).data(Qt.UserRole) for column in range(model.columnCount()))]
            for row in range(model.rowCount())]


def _titles(table):
    model = table.view.model()
    return [model.headerData(column, Qt.Horizontal) for column in range(model.columnCount())]


def _shown(home, widget):
    return widget.isVisibleTo(home)


def _laid_out(home, width=1080, height=820):
    """Home at a size, laid out and drawn once, as a page of the window is."""
    home.resize(width, height)
    home.grab()


def _drawn_lines(monkeypatch, table):
    """What a table draws in lines, read from its painting: for each cell set in lines, the words of each of its
    lines as they were handed in, and the widest line's width."""
    drawn, kept = [], QTextLayout.draw

    def draw(layout, painter, *told):
        laid, found = layout.text(), [layout.lineAt(n) for n in range(layout.lineCount())]
        drawn.append(([laid[line.textStart():line.textStart() + line.textLength()].replace(chr(0x200B), "")
                       for line in found], max(line.naturalTextWidth() for line in found)))
        return kept(layout, painter, *told)

    with monkeypatch.context() as patch:
        patch.setattr(QTextLayout, "draw", draw)
        table.grab()
    return drawn


def _summary(kind, ident, name, created, parents=None):
    """One record as the store's listing gives it, with what the window reads of it."""
    return SimpleNamespace(kind=kind, id=ident, name=name, created=created, reason=None, parents=parents or {},
                           study=None)


def _stand_in_store(listed):
    """A store whose listing gives the records handed in for each kind, and none of any other kind."""
    return SimpleNamespace(list=lambda kind: list(listed.get(kind, ())))


# local time with its zone, as the window's clock gives it
_ZONE = timezone(timedelta(hours=-7))
# a training's lines, as the engine writes them
_BUDGET = "[budget] 10,000 batches x 2,048 rows = 20,480,000 training rows"
_ROTATING = "Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)..."
_ROTATED = "[fisher] eigenvalue spread (best/worst direction): 412"
_LIVE = ("and no other run can start until it ends. You can still move around and read results. Keep PRISM open; "
         "you can minimise it.")


def _at(day, hour, minute, second=0):
    """A moment of October 2026 in the test's local time (the 8th is a Thursday)."""
    return datetime(2026, 10, day, hour, minute, second, tzinfo=_ZONE)


def _on_the_runs_clock(tmp_path, start=None):
    """The process's run controller, its clock (from Thu 8 Oct, 19:02:11, unless ``start`` is handed) and its folder
    of runs this test's own; the end of the test puts both back. The controller and the clock."""
    from tests._runs import Clock
    clock = Clock(start or _at(8, 19, 2, 11))
    held = controller()
    held.use_clock(clock)
    held.use_folder(tmp_path / "runs")
    return held, clock


def _training_spec(**changed):
    """A training as its page hands it to its run: its stages, its patience and its epoch limit."""
    from core.gui.panels.inference.base import TRAINING_STAGES
    from core.gui.run.model import RunSpec
    return RunSpec(**{"word": "Training", "place": ("inference", "training"), "name": "retrain_3",
                      "stop_button": "Stop training…", "stages": TRAINING_STAGES,
                      "kinds": ("posterior", "simulation", "fit"), "simulation_set": True, "patience": 20,
                      "epoch_limit": 300, "facts": {"model": "NADROWSKI", "observation": "chi", "narrowed": False,
                                                    "network": (256, 10)}, **changed})


def _writing(folder) -> str:
    """The engine's line that names the simulation set's folder as the simulation begins."""
    return f"[checkpoint] writing to {folder} every 50 batches (~10.3 GiB total, 120.5 GiB free)"


def _epoch(epoch, since=0, took=3780.0) -> str:
    """An epoch's record, as the training call writes it."""
    return (f"[fit] epoch {epoch}: training loss -3, validation loss -3.1, took {took:.1f} s, epochs since the best "
            f"{since}")


def _foot(window):
    """What the window's foot shows: its marker, its title, its words, and whether it is a button."""
    foot = window.shell.rail.foot
    return foot._marker, foot._title, foot._words, foot.is_button()


def _clocks_started(monkeypatch) -> list:
    """Every timer started from the window's own files -- the window, the shell and the rail -- from now to the
    test's end, each a timer of the toolkit (whatever its parent) or an object whose own timer was started: the start
    is noted where it is made, as the launch test notes the starts it watches."""
    import sys
    from pathlib import Path
    from PySide6.QtCore import QObject, QTimer
    own = ("core/gui/main_window.py", "core/gui/screens/shell.py", "core/gui/widgets/rail.py")
    started, start, start_timer = [], QTimer.start, QObject.startTimer

    def by_the_window() -> bool:
        return Path(sys._getframe(2).f_code.co_filename).as_posix().endswith(own)

    def started_timer(timer, *args):
        if by_the_window():
            started.append(timer)
        return start(timer, *args)

    def started_by(timed, *args, **kwargs):
        if by_the_window():
            started.append(timed)
        return start_timer(timed, *args, **kwargs)

    monkeypatch.setattr(QTimer, "start", started_timer)
    monkeypatch.setattr(QObject, "startTimer", started_by)
    return started


def _repeating(started) -> list:
    """Of the timers ``_clocks_started`` noted, those that repeat: every one but a timer of a single shot."""
    from PySide6.QtCore import QTimer
    return [timer for timer in started if not isinstance(timer, QTimer) or not timer.isSingleShot()]


def _ended_history(folder, run, records_root, **changed):
    """A training's history file in ``folder``, as the window wrote it at the training's end: stopped on Sun 11 Oct
    at 12:31 during its fit, its 20,480,000 simulations kept and its training kept up to epoch 62, its end not yet
    seen; ``changed`` sets any field. The file's path."""
    from core.gui.panels.inference.base import TRAINING_STAGES
    from core.gui.run import history
    ended = _at(11, 12, 31)
    record = {"id": run, "word": "Training", "name": "retrain_3", "place": ["inference", "training"],
              "records_root": str(records_root), "started": _at(8, 19, 2, 11), "ended": ended, "outcome": "stopped",
              "stage": "Fitting", "acknowledged": False,
              "stages": [[stage.name, list(stage.begins), stage.counts] for stage in TRAINING_STAGES],
              "tries": [{"started": _at(8, 19, 2, 11), "ended": ended, "outcome": "stopped", "batches_after": 10_000}],
              "simulation_set": {"folder": "eda5c2a1e468", "batches": 10_000, "at_end": 10_000, "per_batch": 2_048},
              "fit": {"epochs": 62, "best_epoch": 60, "simulations": 20_480_000, "kept_id": "k1", "passes": 62}}
    record.update(changed)
    return history.write(folder, record)


def test_home_with_nothing_on_disk_opens_on_what_do_you_want_to_do():
    """On a computer where nothing has been made yet (no record, no run, no model chosen) Home opens on what a
    person may do first. Its head asks the question and says what PRISM is; four buttons stand in their order,
    each with its sentence beside it, the first the main one; the parts of PRISM are listed, each with its line,
    and the latest records say there are none, with no way to all records. There is no next step and no table of
    the steps."""
    home = _home(_facts())
    for button in home.choices:              # as tall as the look's sheet makes a button, which no suite applies
        button.setMinimumHeight(design.scaled(design.SIZE["button_h"]))
    _laid_out(home)
    head = home.findChild(PageHead)
    assert home.is_first_time()
    assert (head.title.text(), head.sentence.plain_text()) == FIRST_HEAD and _shown(home, head.sentence)
    assert [button.text() for button in home.choices] == [words for words, _sentence, _place in CHOICES]
    assert [_shown(home, button) for button in home.choices] == [True] * 4
    assert [button.property("cls") for button in home.choices] == ["primary", None, None, None]
    tops = [button.mapTo(home, button.rect().topLeft()).y() for button in home.choices]
    assert tops == sorted(tops) and len(set(tops)) == 4, "the four buttons stand one under another, in their order"
    sentences = {label.plain_text(): label for label in home.findChildren(Sentence) if _shown(home, label)}
    for button, (_words, sentence, _place) in zip(home.choices, CHOICES):
        beside = sentences[sentence]
        middle = lambda widget: widget.mapTo(home, widget.rect().center()).y()
        assert abs(middle(beside) - middle(button)) <= 1, sentence
        assert beside.mapTo(home, beside.rect().topLeft()).x() > button.mapTo(home, button.rect().topRight()).x()
    assert _titles(home.parts_table) == ["Part", "What it is for"] and _cells(home.parts_table) == [
        [name, line] for name, line in PARTS]
    assert _shown(home, home.parts_table)
    assert _shown(home, sentences["No records yet."]) and not _shown(home, home.records_table)
    assert [button.text() for button in home.findChildren(QPushButton) if _shown(home, button)
            and button not in home.choices] == [], "a way to all records, or another button, stands on the page"
    assert not _shown(home, home.steps_table) and not _shown(home, home.next_heading)


def test_home_mid_project_states_the_one_next_step_and_where_each_step_stands(monkeypatch):
    """With work under way Home states, under its head, the one next step: its heading, its sentence when it has
    one, and the main button that opens it. The table of the six steps gives each step's number and name, its
    marker and state word, and what it holds or waits for, in full. The parts are listed as on the first opening,
    and the five newest records of every kind, newest first, each with its label, its kind and when it was made;
    one whose moment is not known comes last. All records are a link away. While a run is live the next step is
    to let it end, and its button goes to the place the run is in."""
    home = _home(_facts(_under_way(), _records()))
    _laid_out(home)
    head = home.findChild(PageHead)
    assert not home.is_first_time()
    assert (head.title.text(), head.sentence.plain_text()) == HEAD
    assert not any(_shown(home, button) for button in home.choices)
    assert home.next_heading.label.text() == "Next: start a training"
    sentences = {label.plain_text(): label for label in home.findChildren(Sentence) if _shown(home, label)}
    assert "The Training page says what it will cost before it starts." in sentences
    assert (home.next_button.text(), home.next_button.property("cls")) == ("&Set up training", "primary")
    assert _shown(home, home.next_button) and _shown(home, home.steps_table)
    assert _titles(home.steps_table) == ["Step", "State", "Holds or waits for"]
    assert _cells(home.steps_table) == [
        ["1 Model and options", ("done", "Done"), "Nadrowski"],
        ["2 Prior", ("done", "Done"), "retrain_prior"],
        ["3 Training", ("now", "Now"), "Setting up"],
        ["4 Checks", ("later", "To come"), "Needs a posterior"],
        ["5 Inference", ("later", "To come"), "Needs a posterior"],
        ["6 Narrowing round", ("later", "To come"), "Needs a posterior"]]
    model = home.steps_table.view.model()
    assert all(model.index(row, 2).data(Qt.UserRole).wrap for row in range(6)), "a step's detail is cut"
    assert _cells(home.parts_table) == [[name, line] for name, line in PARTS]
    assert _titles(home.records_table) == ["Name", "Kind", "Made"]
    assert _cells(home.records_table) == [
        ["sweep-s", "Sweep", "3 Mar 2025, 11:02"],
        ["Simulation set for retrain_prior", "Simulation set", "2 Mar 2025, 18:46"],
        ["retrain_band", "Check", "1 Mar 2025, 15:30"],
        ["retrain_prior", "Prior", "1 Mar 2025, 15:29"],
        ["cell_7", "Observation", "20 Feb 2025, 08:05"]]
    assert _shown(home, home.records_table) and "No records yet." not in sentences
    links = [button for button in home.findChildren(QPushButton) if button.text() == "All records"]
    assert len(links) == 1 and _shown(home, links[0]) and links[0].property("cls") == "link"
    recent = datetime.now(timezone.utc) - timedelta(days=1)
    home = _home(_facts(_under_way(), [RecordRow("prior", "p2", "today_prior", "prior", recent)]))
    assert _cells(home.records_table) == [["today_prior", "Prior", design.date_text(recent, datetime.now(timezone.utc))]]
    assert _cells(home.records_table)[0][2].split(" ")[0] in design.WEEKDAYS, "a record of yesterday has its weekday"
    home = _home(_facts(_session(draft=SimpleNamespace(model="HOPF"))))
    assert (home.next_heading.label.text(), home.next_button.text()) == ("Next: build or load a prior", "&Open Prior")
    assert not [label for label in home.findChildren(Sentence) if _shown(home, label)
                and label.plain_text() not in (HEAD[1], "No records yet.")], "a next step with no sentence shows one"
    home = _home(_facts(_under_way(), run=_live("Sweep study", "study", ("temperature", "sweep"))))
    assert home.next_heading.label.text() == "Next: nothing until the sweep study ends"
    sentences = [label.plain_text() for label in home.findChildren(Sentence) if _shown(home, label)]
    assert ("Sweep study study is running, and no other run can start until it ends. You can still move around and "
            "read results. Keep PRISM open; you can minimise it.") in sentences
    assert home.next_button.text() == "&Go to the run"
    long_name = "nadrowski_master_tier1_prior_2026_10_05"           # a record's name: one word, however long
    home = _home(_facts(_session(draft=SimpleNamespace(model="BP"), inf_prior=SimpleNamespace(name=long_name))))
    _laid_out(home)
    view = home.steps_table.view
    (laid, widest), = [(lines, widest) for lines, widest in _drawn_lines(monkeypatch, home.steps_table)
                       if "".join(lines) == long_name]
    assert _cells(home.steps_table)[1][2] == long_name, "a step's detail is not in full"
    assert len(laid) > 1 and widest <= view.columnWidth(2) and view.rowHeight(1) > view.rowHeight(0), (
        "a step's detail is cut where it is drawn", laid, widest, view.columnWidth(2))
    measured = _home(_facts(_under_way(), [RecordRow("fdt", "f2", "cell7_fdt", "effective-temperature measurement",
                                                     datetime(2025, 3, 3, 11, 2, tzinfo=timezone.utc))]))
    _laid_out(measured)
    drawn = _drawn_lines(monkeypatch, measured.records_table)
    assert ([lines for lines, _widest in drawn], measured.records_table.view.rowHeight(0)) == (
        [["Effective-temperature ", "measurement"]], 50), "a record's kind word is cut"


def test_a_row_of_home_opens_its_place_on_a_click_enter_or_space():
    """A row of each of Home's three tables opens its place: a step of parameter inference, a part, or the records
    of a record's kind. It opens on one click, on Enter and on Space, the pointer is a pointing hand over the
    rows and not over their titles, and the row is not left selected behind it. A key with no row under the
    keyboard opens nothing, and the keys that move along a table still move along it. The main button opens the
    next step's place, the link all records, and each of the first opening's buttons its own."""
    home = _home(_facts(_under_way(), _records()))
    _laid_out(home)
    heard = []
    home.navigate.connect(lambda part, page: heard.append((part, page)))
    view = home.steps_table.view
    for key in (Qt.Key_Return, Qt.Key_Space):
        QTest.keyClick(view, key)
    assert heard == [], "a key opened a row though the keyboard is on none"
    for table, row, place in ((home.steps_table, 2, ("inference", "training")),
                              (home.parts_table, 4, ("records", None)),
                              (home.parts_table, 1, ("temperature", None)),
                              (home.records_table, 1, ("records", "simulation"))):
        view = table.view
        assert view.viewport().cursor().shape() == Qt.PointingHandCursor
        QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier,
                         view.visualRect(view.model().index(row, 0)).center())
        assert heard == [place] and table.selected_row() is None, (row, place)
        view.setCurrentIndex(view.model().index(row, 1))
        for key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            QTest.keyClick(view, key)
        assert heard == [place] * 4 and table.selected_row() is None, (row, place)
        heard.clear()
    home.next_button.click()
    link, = [button for button in home.findChildren(QPushButton) if button.text() == "All records"]
    link.click()
    assert heard == [("inference", "training"), ("records", None)]
    assert link.cursor().shape() == Qt.PointingHandCursor
    heard.clear()
    for table in (home.steps_table, home.parts_table, home.records_table):
        view = table.view
        view.setCurrentIndex(view.model().index(0, 0))
        QTest.keyClick(view, Qt.Key_Down)
        QTest.keyClick(view, Qt.Key_Down)
        assert (view.currentIndex().row(), heard) == (2, []), "a key that moves along a table was taken by Home"
        assert view.horizontalHeader().cursor().shape() != Qt.PointingHandCursor, "a hand over the titles"
    first = _home(_facts())
    first.navigate.connect(lambda part, page: heard.append((part, page)))
    for button in first.choices:
        button.click()
    assert heard == [place for _words, _sentence, place in CHOICES]


def test_home_reads_again_when_shown_and_only_then():
    """Home reads what it states once when it is built. Marked to read again, or told to while it is hidden, it
    reads nothing until it is next shown, and then once; shown again with nothing marked, it reads nothing. Told to
    while it is shown, it reads at once. What it shows is what it read last."""
    told = [_facts(), _facts(_under_way())]
    reads = []

    def read():
        reads.append(len(reads))
        return told[min(len(reads), len(told)) - 1]

    home = build_part(HomeScreen, read=read)
    assert len(reads) == 1 and home.is_first_time()
    home.mark_stale()
    home.refresh()
    assert len(reads) == 1 and home.is_first_time(), "Home read while it was hidden"
    home.setAttribute(Qt.WA_DontShowOnScreen)
    try:
        home.show()
        assert len(reads) == 2 and not home.is_first_time()
        home.hide()
        home.show()
        assert len(reads) == 2, "Home read again at a show with nothing marked"
        home.refresh()
        assert len(reads) == 3
        home.mark_stale()
        assert len(reads) == 3, "marked while shown, Home read at once"
        home.hide()
        home.show()
        assert len(reads) == 4
    finally:
        home.hide()


def test_a_refresh_that_changes_nothing_keeps_the_keyboards_row():
    """A table of Home is filled again only when its rows have changed. A person moving down a table with the keys
    keeps their row, and its selection, through every reading that changes nothing in that table; a table whose
    rows change is filled again, and nothing in it is then selected."""
    told = [_facts(_under_way(), _records())]
    home = build_part(HomeScreen, read=lambda: told[0])
    home.setAttribute(Qt.WA_DontShowOnScreen)
    try:
        home.show()
        home.steps_table.select_row(3)
        home.parts_table.select_row(5)
        home.records_table.select_row(2)
        home.refresh()
        assert [table.selected_row() for table in (home.steps_table, home.parts_table, home.records_table)] == [3, 5, 2]
        assert home.steps_table.view.currentIndex().row() == 3
        held = _under_way()
        told[0] = _facts(_session(draft=held.draft, inf_prior=held.inf_prior, posterior=SimpleNamespace(name="q")),
                         _records())
        home.refresh()
        assert home.steps_table.selected_row() is None and home.steps_table.view.currentIndex().row() == -1
        assert [home.parts_table.selected_row(), home.records_table.selected_row()] == [5, 2]
    finally:
        home.hide()


def test_homes_columns_stack_on_a_narrow_page():
    """The table of the steps stands beside the parts and the records while the page has room for two columns,
    and above them when it has not: the middle of the page hands its width on to the two columns. The page's
    padding, the room the middle keeps for a focus ring, its scroll bar and the gap beside it are taken off the
    page's width first."""
    home = _home(_facts(_under_way(), _records()))
    columns = home.findChild(Columns)
    left, _top, right, _bottom = design.SIZE["page_pad"]     # the ring's room comes off the padding and is kept
    edge = design.scaled(left) + design.scaled(right) + design.SIZE["scroll_bar"] + design.SIZE["scroll_gap"]
    _laid_out(home, 940 + edge, 800)
    assert not columns.stacked()
    left = home.steps_table.mapTo(home, home.steps_table.rect().topLeft())
    right = home.parts_table.mapTo(home, home.parts_table.rect().topLeft())
    assert left.x() < right.x() and left.y() == right.y(), "the steps are not beside the parts"
    _laid_out(home, 940 + edge - 1, 800)
    assert columns.stacked()
    left = home.steps_table.mapTo(home, home.steps_table.rect().topLeft())
    right = home.parts_table.mapTo(home, home.parts_table.rect().topLeft())
    assert left.x() == right.x() and left.y() < right.y(), "the steps are not above the parts"


def test_home_takes_the_form_of_a_low_window_and_gives_it_back():
    """Told that the window is low, Home takes the page's small form and its head drops its sentence; told that it
    is not, Home takes its usual form again."""
    home = _home(_facts(_under_way(), _records()))
    head = home.findChild(PageHead)
    home.set_small(True)
    assert home.page.is_small() and head.sentence.isHidden(), "Home kept its usual form in a low window"
    home.set_small(False)
    assert not home.page.is_small() and not head.sentence.isHidden()


def test_home_keeps_one_main_button_and_its_access_key():
    """On each of Home's pages one button is the main one, of the main button's class, and it has an access key on
    its first letter; no other button on the page has one. It is where the keyboard starts on the page, and it
    opens a place and starts nothing."""
    for facts, words, key in ((_facts(), "&Watch a live simulation", "Alt+W"),
                              (_facts(_under_way(), _records()), "&Set up training", "Alt+S"),
                              (_facts(_under_way(), run=_live("Live simulation", place=("live", None))),
                               "&Go to the run", "Alt+G")):
        home = _home(facts)
        shown = [button for button in home.findChildren(QPushButton) if _shown(home, button)]
        main = [button for button in shown if button.property("cls") == "primary"]
        assert [button.text() for button in main] == [words] and home.first_focus() is main[0]
        assert QKeySequence.mnemonic(main[0].text()).toString() == key
        assert [button.text() for button in shown if "&" in button.text()] == [words]


def test_the_window_reads_for_home_its_session_its_run_and_every_record_of_the_store_in_use(tmp_path):
    """What the window hands Home: the inference screen's session and its six steps, once the screen is built, the
    run's status, live with its place while a run is live (the step whose run it is then reads "Now"), and every
    record of the store in use that has a manifest, of every kind, each with its label, the word for its kind and
    the moment it was made, in local time. A simulation set is called by the prior it was simulated for, or said to
    have lost it. A folder with no manifest is no record. A store with no record and nothing chosen opens Home on
    its first opening."""
    from core.artifacts import KIND_DIRS, ArtifactStore, use_store, write_simulation_manifest
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.SBI.training_checkpoint import identity_digest
    from tests._fixtures import build_browse_store
    qt_app()
    with use_store(ArtifactStore(tmp_path / "empty")):
        assert build_window().part("home").is_first_time()
    ids = build_browse_store(tmp_path / "Artifacts")
    store = ArtifactStore(tmp_path / "Artifacts")
    identity = {"format": "training-rows/2", "prior_fingerprint": "e" * 16, "n_runs": 2, "truncation": None}
    held = write_simulation_manifest(store.kind_dir("simulation") / identity_digest(identity), identity,
                                     parents={"prior": ids["prior"]}, batches_done=1)
    with use_store(store):
        window = build_window()
        inference = window.part("inference")
        facts = window._home_facts()
        assert facts.session is inference.session and facts.run == controller().status and not facts.run.live
        assert facts.states == chain.step_states(inference.session) and not facts.has_runs
        rows = {(row.kind, row.id): row for row in facts.records}
        wanted = {(kind, ids[kind]) for kind in KIND_DIRS} | {("simulation", held.id)}
        assert set(rows) == wanted and len(facts.records) == len(wanted)
        for kind in KIND_DIRS:
            summary, = [one for one in store.list(kind) if one.id == ids[kind]]
            made = datetime.fromisoformat(summary.created).astimezone()
            row = rows[(kind, ids[kind])]
            label = ("Simulation set for a prior that is no longer here" if kind == "simulation" else
                     record_words.record_label("fit", "", made) if kind == "fit" else f"browse_{kind}")
            assert (row.label, row.kind_word, row.made, row.made.utcoffset()) == (
                label, record_words.kind_word(kind, summary.study), made, made.utcoffset())
        assert rows[("simulation", held.id)].label == "Simulation set for browse_prior"
        assert rows[("fdt", ids["fdt"])].kind_word == "sweep"
        assert not window.part("home").is_first_time()
        with running(window.panel(CrossValPanel)):
            assert window._home_facts().run.place == ("temperature", "sweep") and window._home_facts().run.live
        with running(window.part("inference").posterior_panel):
            live = window._home_facts()
            assert (live.run.place, live.run.live) == (("inference", "training"), True)
            assert (live.states[2].marker, live.states[2].word, live.states[2].detail) == ("now", "Now", "Running")
        assert not window._home_facts().run.live


def test_a_simulation_set_for_an_unnamed_prior_names_it_in_small_letters():
    """A simulation set is called by the prior it was simulated for. A prior with no name is called by its kind and
    the moment it was made, and inside the set's label those words begin with a small letter; a prior's name is
    kept as it was written."""
    from core.gui import main_window
    priors = [_summary("prior", "p1", "", "2026-10-03T14:02:00+05:45"),
              _summary("prior", "p2", "Retrain_prior", "2026-10-03T14:02:00+05:45")]
    made = datetime.fromisoformat(priors[0].created).astimezone()
    rows = main_window._home_records(_stand_in_store({"prior": priors, "simulation": [
        _summary("simulation", "s1", "", "2026-10-03T15:00:00+05:45", {"prior": "p1"}),
        _summary("simulation", "s2", "", "2026-10-03T15:00:00+05:45", {"prior": "p2"})]}))
    labels = {row.id: row.label for row in rows}
    assert (labels["s1"], labels["s2"]) == (f"Simulation set for unnamed prior, {design.date_text(made)}",
                                            "Simulation set for Retrain_prior")


def test_homes_records_name_an_unnamed_record_in_local_time_and_leave_out_what_cannot_be_read():
    """The window lists for Home what it can read of the store in use, and nothing it cannot read keeps the window
    from opening: an unnamed record is labelled with the moment it was made in local time, as the Made column
    writes it, whatever zone its manifest was written in; a moment that cannot be read leaves the date out; a kind
    whose folder cannot be listed is left out. Home lists a record whose moment is not known last, with no date."""
    from core.artifacts import StoreError
    from core.gui import main_window
    dated = _summary("calibration", "c1", "", "2026-10-03T14:02:00+05:45")
    made = datetime.fromisoformat(dated.created).astimezone()
    named = ("c1", f"Unnamed calibration, {design.date_text(made)}", made)
    rows = main_window._home_records(_stand_in_store({"calibration": [dated]}))
    assert [(row.id, row.label, row.made) for row in rows] == [named], "an unnamed record dated in another zone"

    def listed(kind):
        if kind == "posterior":
            raise PermissionError("Access is denied")
        if kind == "prior":
            raise StoreError("unreadable")
        return {"calibration": [dated, _summary("calibration", "c2", "", "not a date")]}.get(kind, [])

    rows = main_window._home_records(SimpleNamespace(list=listed))
    assert [(row.id, row.label, row.made) for row in rows] == [named, ("c2", "Unnamed calibration", None)]
    home = _home(_facts(_under_way(), [RecordRow("calibration", "c2", "Unnamed calibration", "calibration", None),
                                       _records()[0]]))
    assert _cells(home.records_table) == [["retrain_prior", "Prior", "1 Mar 2025, 15:29"],
                                          ["Unnamed calibration", "Calibration", ""]]


def test_home_opens_each_part_step_and_kind_through_go_to(tmp_path):
    """Each place Home asks for is opened by ``go_to``: a part its own page (Help the help, Your own models their
    list), a page of Effective temperature its page with its name marked, a step its page, one that
    cannot start yet too, and a record's kind the records at that kind. Home reads again each time it is opened,
    and while it is shown when the steps' states are worked out again, when a run starts or ends and when its end is
    seen; never on its own."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.session import SbiSession
    qt_app()
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window = build_window()
        home, stack = window.part("home"), window.shell.stack
        for part in ("reduction", "temperature", "inference", "live", "records", "settings", "help", "models"):
            home.navigate.emit(part, None)
            assert stack.currentWidget() is window.part(part), part
            home.navigate.emit("home", None)
            assert stack.currentWidget() is home
        assert window.part("models").current_page() == "list"
        for page, panel in (("sweep", 1), ("one_cell", 0)):
            home.navigate.emit("temperature", page)
            screen = window.part("temperature")
            assert (window.current_place(), screen.names.current()) == (("temperature", page), page)
            assert screen.panels()[panel].isVisibleTo(screen) and not screen.panels()[1 - panel].isVisibleTo(screen)
        home.navigate.emit("records", "posterior")
        assert (window.current_place(), window.part("records").kind()) == (("records", None), "posterior")
        home.navigate.emit("inference", "infer")
        assert window.current_place() == ("inference", "infer"), "a step that cannot start yet did not open"
        window.part("inference").session = SbiSession(draft=object(), cfg=object(), inf_prior=object(),
                                                     posterior=object())
        window.part("inference").refresh_gates()
        for step in keys.STEPS:
            home.navigate.emit("inference", step.key)
            assert window.current_place() == ("inference", step.key)
        window.part("inference").session = SbiSession()
        window.part("inference").refresh_gates()
        home.navigate.emit("home", None)
        window.setAttribute(Qt.WA_DontShowOnScreen)
        try:
            window.show()
            assert home.is_first_time()
            window.part("inference").session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"))
            assert home.is_first_time(), "Home read while it was shown and nothing told it to"
            window.part("inference").refresh_gates()
            assert not home.is_first_time() and home.next_heading.label.text() == "Next: build or load a prior"
            home.navigate.emit("live", None)
            window.part("inference").session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"),
                                                         inf_prior=SimpleNamespace(name="p"))
            window.part("inference").refresh_gates()
            assert home.next_heading.label.text() == "Next: build or load a prior", "Home read while hidden"
            home.navigate.emit("home", None)
            assert home.next_heading.label.text() == "Next: start a training", (
                "Home stated what it read when it was left")
            with running(window.panel(CrossValPanel)):
                assert home.next_heading.label.text() == "Next: nothing until the training ends"
            assert home.next_heading.label.text() == "Next: see the training", "Home did not read as the run ended"
            controller().acknowledge(controller().status.run)
            assert home.next_heading.label.text() == "Next: start a training", "Home did not read as the end was seen"
        finally:
            window.hide()


def test_in_the_window_home_reads_nothing_while_another_place_is_shown(tmp_path, monkeypatch):
    """In the window Home reads nothing while the window is not shown, nothing as another place is opened, nothing
    while another place is shown and a run starts or ends, and nothing at a move between two other places; opened
    again, it reads once."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui import main_window
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    reads, read = [], main_window._home_records
    monkeypatch.setattr(main_window, "_home_records", lambda store: (reads.append(store), read(store))[1])

    def run_starts_and_ends():
        """The number of readings as a run starts, and when it has ended."""
        with running(panel):
            started = len(reads)
        return started, len(reads)

    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window = build_window()
        home, panel = window.part("home"), window.panel(CrossValPanel)
        assert len(reads) == 1, "Home did not read as it was built"
        assert run_starts_and_ends() == (1, 1), "Home read while the window was not shown"
        window.setAttribute(Qt.WA_DontShowOnScreen)
        try:
            window.show()
            assert len(reads) == 2, "Home marked to read again did not read when it was shown"
            home.navigate.emit("live", None)
            assert len(reads) == 2, "Home read as another place was opened"
            assert run_starts_and_ends() == (2, 2), "Home read while another place was shown"
            home.navigate.emit("reduction", None)
            assert len(reads) == 2, "Home read at a move between two other places"
            home.navigate.emit("home", None)
            assert len(reads) == 3, "Home opened again did not read once"
        finally:
            window.hide()


def test_homes_next_step_follows_the_run():
    """Home's next step follows the run's status, the first that holds: a live run that needs someone (its reason,
    and "See why" goes to it); a live run (nothing until it ends, while it runs, stops or waits to be tried again, and
    "Go to the run" goes to it); an end nobody has seen (how and when it ended and what it kept, and the button opens
    the run's page: "See the result" for a finished run, "Go to the run" for a stopped one, "See what happened" for a
    failure or a run not seen to end); then the ordinary next step, once that end has been seen. Home never offers
    to continue a run from its batch."""
    from dataclasses import replace
    from core.gui.run.model import IDLE, RunStatus
    place, run = ("inference", "training"), "20261008T190211-training"
    live = _live(name="retrain_3")
    said = "Training was stopped on Sun 11 Oct, 12:31, during simulation. Nothing was simulated."
    ended = RunStatus(state="ended", word="Training", name="retrain_3", run=run, place=place, outcome="stopped",
                      sentence=said)
    ordinary = ("Next: start a training", "The Training page says what it will cost before it starts.",
                "&Set up training", ("navigate", place))
    cases = [
        (replace(live, needs_you="Nothing has been heard from the run for 10 minutes."), (
            "Next: look at the training", "Nothing has been heard from the run for 10 minutes.", "&See why",
            ("navigate", place))),
        (live, ("Next: nothing until the training ends", f"Training retrain_3 is running, {_LIVE}", "&Go to the run",
                ("navigate", place))),
        (replace(live, state="stopping", stop_requested=True), (
            "Next: nothing until the training ends", f"Training retrain_3 is stopping, {_LIVE}", "&Go to the run",
            ("navigate", place))),
        (replace(live, state="retrying"), (
            "Next: nothing until the training ends", f"Training retrain_3 is waiting to try again, {_LIVE}",
            "&Go to the run", ("navigate", place))),
        (replace(ended, outcome="finished"), ("Next: see the training", said, "&See the result", ("open_run", run))),
        (ended, ("Next: see what happened to the training", said, "&Go to the run", ("open_run", run))),
        (replace(ended, outcome="failed"), (
            "Next: see what happened to the training", said, "&See what happened", ("open_run", run))),
        (replace(ended, state="not_seen", outcome="not seen to end"), (
            "Next: see what happened to the training", said, "&See what happened", ("open_run", run))),
        (replace(ended, acknowledged=True), ordinary),
        (IDLE, ordinary),
    ]
    for status, (heading, sentence, button, opened) in cases:
        home = _home(_facts(_under_way(), run=status))
        heard = []
        home.navigate.connect(lambda part, page: heard.append(("navigate", (part, page))))
        home.open_run.connect(lambda ran: heard.append(("open_run", ran)))
        sentences = [label.plain_text() for label in home.findChildren(Sentence) if _shown(home, label)]
        assert (home.next_heading.label.text(), sentence in sentences, home.next_button.text()) == (
            heading, True, button), (status, sentences)
        home.next_button.click()
        assert heard == [opened], (status, heard)
        offers = [shown.text() for shown in home.findChildren(QPushButton) if _shown(home, shown)]
        assert not [words for words in offers if "Continue" in words or "Resume" in words], offers


def test_home_lists_the_last_runs_and_opens_one_in_its_past_view(tmp_path):
    """Home lists the three latest runs of the records folder in use, newest first: the run's word and name, when it
    started and when it ended, as a date is written, and how it ended, by its marker and its word. A row opens the
    run's place, its page drawn in its past view from the run's two files alone: its stages as its history names
    them, read again from its text, its sentence how and when it ended and what it kept, and its full text the
    file's."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.run import history
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path, _at(12, 9, 0))
    root, folder = tmp_path / "Artifacts", tmp_path / "runs"
    with use_store(ArtifactStore(root)):
        _ended_history(folder, "20261001T120000-training", root, outcome="finished", acknowledged=True,
                       started=_at(1, 12, 0), ended=_at(2, 12, 0))
        history.write(folder, {"id": "20261002T120000-measurement", "word": "Measurement", "name": "cell_a",
                               "place": ["temperature", "one_cell"], "records_root": str(root),
                               "started": _at(2, 12, 0), "ended": _at(2, 13, 0), "outcome": "stopped",
                               "acknowledged": True})
        _ended_history(folder, "20261003T120000-training", root, outcome="failed", acknowledged=True,
                       started=_at(3, 12, 0), ended=_at(3, 14, 0), failure={"type": "RuntimeError",
                                                                             "message": "the run broke"})
        newest = "20261004T120000-training"
        _ended_history(folder, newest, root, acknowledged=True, started=_at(4, 12, 0), ended=_at(11, 12, 31),
                       stage="Simulation", fit=None, tries=[{"ended": _at(11, 12, 31), "batches_after": 6_150}])
        lines = ["2026-10-04 12:00:00 info " + _BUDGET, "2026-10-04 12:00:01 info " + _ROTATING,
                 "2026-10-04 12:00:02 info " + _ROTATED, "2026-10-04 12:00:03 info " + _writing(folder)]
        (folder / f"{newest}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _ended_history(folder, "20261005T120000-training", tmp_path / "elsewhere", acknowledged=True,
                       started=_at(5, 12, 0))
        window = build_window()
        home = window.part("home")

        def now(moment):
            return design.date_text(moment.astimezone(), datetime.now(timezone.utc).astimezone())

        assert _titles(home.runs_table) == ["Run", "Started", "Ended", "How it ended"]
        assert _cells(home.runs_table) == [
            ["Training retrain_3", now(_at(4, 12, 0)), now(_at(11, 12, 31)), ("half", "Stopped")],
            ["Training retrain_3", now(_at(3, 12, 0)), now(_at(3, 14, 0)), ("err", "Failed")],
            ["Measurement cell_a", now(_at(2, 12, 0)), now(_at(2, 13, 0)), ("half", "Stopped")]]
        view = home.runs_table.view
        QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier,
                         view.visualRect(view.model().index(0, 0)).center())
        host = window.part("inference").posterior_panel
        page = host.past_view
        assert (window.current_place(), host.results.currentWidget() is page, page.mode) == (
            ("inference", "training"), True, "past")
        assert page.sentence.text() == ("Training was stopped on Sun 11 Oct, 12:31, during simulation. The 6,150 "
                                        "batches simulated so far are kept; you can continue from batch 6,150.")
        assert [(name, marker) for name, marker, _words in page.stage_line.words()] == [
            ("Rotation", "done"), ("Simulation", "half"), ("Fitting", "later"), ("Saving", "later")]
        assert page.full_text.lines()[-len(lines):] == [line[20:].partition(" ")[2] for line in lines]


def test_a_live_run_ends_the_first_time_page(tmp_path):
    """On a computer where nothing has been made, Home opens on its first-time page. A run started from the window
    ends that page at once, while Home is in sight, and Home does not go back to it once the run has ended: the run
    is then in the history of the records folder in use."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window = build_window()
        home = window.part("home")
        try:
            _shown_off_screen(window)
            assert home.is_first_time()
            with running(window.panel(CrossValPanel)):
                assert not home.is_first_time(), "a live run left the first-time page"
            assert not home.is_first_time()
            held.acknowledge(held.status.run)
            window.go_to("home")
            assert not home.is_first_time(), "a run in the history left Home on its first-time page"
        finally:
            window.hide()


def test_home_reads_the_store_only_when_a_run_ends_or_home_is_shown(tmp_path, monkeypatch):
    """While Home is shown it states the run's status at every change of it, but reads the store's records and the
    run history only as it is built, when a run has ended, as it is opened and when the store changes: a training's
    status changes at each of its stages, every save of its simulation set and every epoch, and none of those reads
    the store."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui import main_window
    from core.gui.run import history
    from core.gui.run.model import SIMULATION_ROW
    app = qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    reads, records, latest = [], main_window._home_records, history.latest
    monkeypatch.setattr(main_window, "_home_records", lambda store: (reads.append("records"), records(store))[1])
    monkeypatch.setattr(history, "latest", lambda *args, **kwargs: (reads.append("runs"), latest(*args, **kwargs))[1])
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window = build_window()
        home = window.part("home")
        try:
            _shown_off_screen(window)
            assert reads == ["records", "runs"], reads
            del reads[:]
            text, stand_in = RunText(held), StandIn()
            start_run(window.part("inference").posterior_panel, stand_in, spec=_training_spec())
            for line in (_BUDGET, _ROTATING, _ROTATED, _writing(tmp_path / "simulations" / "_unnamed__set")):
                stand_in.say(line)
            for done in (50, 100, 150):
                stand_in.row(SIMULATION_ROW, done, 10_000)
                turn_until(app, lambda: held.model.batches_done() == done)
            assert len(text.statuses) > 4 and home.next_heading.label.text() == (
                "Next: nothing until the training ends"), [status.state for status in text.statuses]
            assert reads == [], f"Home read the store at a change of the run's status: {reads}"
            stand_in.release()
            turn_until(app, lambda: not held.is_live())
            assert (reads, home.next_heading.label.text()) == (["records", "runs"], "Next: see the training"), reads
        finally:
            window.hide()


def _marked(rail):
    """The keys of the rail's rows that carry the mark of the place in sight: a row marked keeps its focus ring
    clear of the bar it then wears."""
    return [key for key in [place.key for place in keys.PARTS] + [step.key for step in keys.STEPS]
            if rail.row(key).property("ring_left")]


def _name(screen, page):
    """The name of the page ``page`` in the row of names over ``screen``'s pages."""
    from PySide6.QtWidgets import QAbstractButton
    return next(name for name in screen.names.findChildren(QAbstractButton) if name.place.key == page)


def _button(widget, words):
    """The push button of ``widget`` that reads ``words``."""
    return next(button for button in widget.findChildren(QPushButton) if button.text() == words)


def _in_view(rail, row):
    """Whether ``row`` stands whole inside the part of the rail's list that is in sight."""
    view = rail._scroll.viewport()
    top = row.mapTo(view, QPoint(0, 0)).y()
    return row.isVisible() and top >= 0 and top + row.height() <= view.height()


def test_the_rails_places_are_the_parts_in_order_read_off_the_built_window():
    """The rail lists the window's nine parts in their order, each by its name; a click on a row shows that part's
    own page, built at its first opening, and ``part`` gives the same page by the same key, so that the parts
    opened in the rail's order stand in the stack in that order. The six steps are rows of Parameter inference, in
    sight only while it is open."""
    from core.gui.widgets.rail import PartRow, StepRow
    qt_app()
    window = build_window()
    rail, stack = window.shell.rail, window.shell.stack
    assert [row.place.key for row in rail.visible_rows() if isinstance(row, PartRow)] == [
        place.key for place in keys.PARTS]
    assert [rail.row(place.key).text() for place in keys.PARTS] == [place.name for place in keys.PARTS]
    for place in keys.PARTS:
        rail.row(place.key).click()
        page = window.part(place.key)
        assert stack.currentWidget() is page and window.current_place()[0] == place.key, place.key
        steps = [row for row in rail.visible_rows() if isinstance(row, StepRow)]
        assert steps == ([rail.row(step.key) for step in keys.STEPS] if place.key == "inference" else []), place.key
    assert [stack.widget(n) for n in range(stack.count())] == [window.part(place.key) for place in keys.PARTS]
    with pytest.raises(KeyError):
        window.part("nowhere")


def test_the_window_opens_on_home_whatever_was_open():
    """The window opens on Home, its row marked in the rail, whatever an older build left in the settings file."""
    from core.gui import settings as st
    qt_app()
    remembered = st.settings()
    remembered.setValue("window/tab", 2)          # a stale key from the old flat-tab layout
    remembered.sync()
    window = build_window()
    assert window.shell.stack.currentWidget() is window.part("home")
    assert window.current_place() == ("home", None) and _marked(window.shell.rail) == ["home"]


def test_go_to_shows_the_part_marks_the_rail_and_says_so_before_it_returns():
    """``go_to`` shows the part and its page, marks the row of what is shown and says where the window is, all
    before it returns, once a call, the same place too. Parameter inference opened with no step shows the step in
    hand the first time and the step last shown after that; a step that cannot start yet opens too, and the rail
    marks it. A part the window does not have, or a page its part does not have, is a mistake of the caller's,
    refused before anything moves. The gates move no step and say nothing. A page of Effective temperature a
    person chooses by its name is said as a move, the part's row marked; a page shown out of sight by anything
    but ``go_to`` is not said. A click on the rail's row of a step that cannot start opens that step too."""
    from core.gui.session import SbiSession
    qt_app()
    window = build_window()
    rail, inference = window.shell.rail, window.part("inference")
    said = []
    window.place_changed.connect(
        lambda part, page: said.append((part, page, window.shell.stack.currentWidget() is window.part(part),
                                        _marked(rail))))
    inference.session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"))
    inference.refresh_gates()
    window.go_to("inference")
    assert said == [("inference", "prior", True, ["prior"])], "the first opening is not at the step in hand"
    window.go_to("inference", "model")
    window.go_to("live")
    window.go_to("inference")
    assert said[1:] == [("inference", "model", True, ["model"]), ("live", None, True, ["live"]),
                        ("inference", "model", True, ["model"])], "a later opening is not at the step last shown"
    window.go_to("inference", "infer")
    assert said[-1] == ("inference", "infer", True, ["infer"]), "a step that cannot start yet did not open"
    window.go_to("temperature", "sweep")
    window.go_to("temperature", "sweep")
    assert said[-2:] == [("temperature", "sweep", True, ["temperature"])] * 2
    assert window.part("temperature").current_page() == "sweep" and len(said) == 7
    with pytest.raises(KeyError):
        window.go_to("nowhere")
    assert (window.current_place(), _marked(rail), len(said)) == (("temperature", "sweep"), ["temperature"], 7), (
        "a part the window does not have moved the window")
    _name(window.part("temperature"), "one_cell").click()         # its own row of names, while it is in sight
    assert said[-1] == ("temperature", "one_cell", True, ["temperature"]) and len(said) == 8
    window.go_to("inference", "prior")
    inference.session = SbiSession()
    inference.refresh_gates()                                     # the step can no longer start: it stays shown
    assert (said[-1], len(said), window.current_place()) == (("inference", "prior", True, ["prior"]), 9,
                                                             ("inference", "prior"))
    window.go_to("records")
    window.part("temperature").show_page("sweep")                 # out of sight: nothing is said or marked
    assert len(said) == 10 and _marked(rail) == ["records"]
    for part in ("inference", "temperature", "models"):
        with pytest.raises(KeyError):
            window.go_to(part, "nowhere")
    assert (window.current_place(), _marked(rail), len(said)) == (("records", None), ["records"], 10), (
        "a page its part does not have moved the window")
    window.go_to("inference", "model")
    rail.row("infer").click()                                     # the session is empty: Inference cannot start
    assert (window.current_place(), _marked(rail), inference.step_states()[4].can_start) == (
        ("inference", "infer"), ["infer"], False), "a click on the row of a step that cannot start did not open it"


def test_go_to_gives_the_page_the_keyboard_and_escape_in_the_rail_gives_it_back():
    """After a move the keyboard is on the page's first setting: the control a page names to start at (Home's
    main button, Help's text, the button of Your own models' list that opens the builder, which has no setting),
    or else its first control that is not a button that does something, nor a tab bar or a row of page names,
    nor an area that only shows or scrolls (each step and each page's first setting, the records' kind, the
    first appearance). Escape in the rail, and Enter on a row of the rail, give the page the keyboard the same
    way."""
    from PySide6.QtWidgets import QComboBox
    qt_app()
    window = build_window()
    rail = window.shell.rail
    first = {"home": window.part("home").first_focus(), "inference": window.part("inference").config_panel.model_combo,
             "temperature": window.part("temperature").panels()[0].model_combo,
             "live": window.part("live").model_combo, "records": window.part("records").kind_combo,
             "models": _button(window.part("models"), "Open model builder"),
             "settings": window.part("settings")._radios["system"], "help": window.part("help").viewer}
    for place in keys.PARTS:
        window.go_to(place.key)
        held = window.focusWidget()
        if place.key == "reduction":
            controls = window.part("reduction").controls
            assert isinstance(held, QComboBox) and controls.isAncestorOf(held), held
        else:
            assert held is first[place.key], (place.key, held)
    window.go_to("records")
    rail.take_keyboard()
    assert rail.isAncestorOf(window.focusWidget())
    QTest.keyClick(window.focusWidget(), Qt.Key_Escape)
    assert window.focusWidget() is first["records"], "Escape in the rail did not give the page the keyboard"
    rail.take_keyboard()
    QTest.keyClick(window.focusWidget(), Qt.Key_Down)                 # to Your own models, the next row
    QTest.keyClick(window.focusWidget(), Qt.Key_Return)
    assert window.current_place() == ("models", "list") and window.focusWidget() is first["models"]


def test_a_mouse_click_on_a_rail_row_puts_no_ring_on_the_page():
    """With the keyboard's turn on a page's first setting, a click on another part's row takes the keyboard off
    the page it leaves and gives it to nothing: no control of the page it opens wears the focus ring, not even the
    field one types in that the page held the keyboard in before. The window is not shown on a screen, so the
    focus that would arrive there is handed to the ring's follower as the screen would hand it."""
    from core.gui import style
    from tests._look_helpers import _arrive, _moved_on
    app = qt_app()
    window = build_window()                             # first: a window that cannot be built leaves no follower
    follower = style.FocusFollower(app)
    try:
        _shown_off_screen(window)
        for (start, page), target in ((("live", None), "models"), (("models", "builder"), "live"),
                                      (("records", None), "settings"), (("models", "builder"), "inference")):
            window.go_to(start, page)
            field = window.focusWidget()
            _arrive(field, Qt.TabFocusReason)                         # a press of Tab, then the focus
            assert follower.target() is field, start
            row = window.shell.rail.row(target)
            QTest.mouseClick(row, Qt.LeftButton, Qt.NoModifier, row.rect().center())
            held = window.focusWidget()
            if held is not None:
                _moved_on(held, Qt.OtherFocusReason)                  # what the screen would deliver
            assert window.current_place()[0] == target, target
            assert follower.target() is None, (start, target, follower.target())
            assert held is None or not window.part(target).isAncestorOf(held), (target, held)
    finally:
        window.hide()
        follower.stop()


def test_a_click_on_a_rail_row_gives_the_page_no_keyboard_whatever_held_it_and_a_key_does(monkeypatch):
    """With the keyboard in the rail -- on its first row, as the window coming to the front leaves it, or brought
    there by a key -- a click on a part's row opens the part and gives its page neither the keyboard nor the ring:
    the keyboard stays in the rail, on the row the click marked, and no ring shows, not even on a field one types
    in. Enter and Space on a row give the page its first setting, ringed as after a key. The window is not shown
    on a screen, so each focus the window gives is handed to the ring's follower as the screen would hand it."""
    from PySide6.QtWidgets import QWidget
    from core.gui import style
    app = qt_app()
    window = build_window()                             # first: a window that cannot be built leaves no follower
    rail = window.shell.rail
    follower = style.FocusFollower(app)
    kept = QWidget.setFocus

    def set_focus(widget, reason=Qt.OtherFocusReason):
        kept(widget, reason)
        follower.arrived(widget, reason)

    monkeypatch.setattr(QWidget, "setFocus", set_focus)

    def click(key):
        """A click on the row of the part ``key``; the row the rail marks then."""
        row = rail.row(key)
        QTest.mouseClick(row, Qt.LeftButton, Qt.NoModifier, row.rect().center())
        part, page = window.current_place()
        assert part == key, key
        return rail.row(page if part == "inference" else part)

    try:
        _shown_off_screen(window)
        rail.row("home").setFocus(Qt.ActiveWindowFocusReason)          # the window comes to the front
        for key in ("models", "inference", "temperature", "live", "reduction", "records", "settings", "help", "home"):
            click(key)
            held = window.focusWidget()
            assert rail.isAncestorOf(held) and follower.target() is None, (key, held, follower.target())
        rail.take_keyboard()                                             # a key brings it there: the ring shows
        assert follower.target() is rail.row("home")
        marked = click("live")
        assert (window.focusWidget(), follower.target()) == (marked, None), "a ring was left in the rail"
        QTest.keyClick(window.focusWidget(), Qt.Key_Return)
        first = window.part("live").model_combo
        assert (window.current_place(), window.focusWidget(), follower.target()) == (("live", None), first, first)
        rail.take_keyboard()
        marked = click("records")
        assert (window.focusWidget(), follower.target()) == (marked, None), "a key's opening outlived its key"
        QTest.keyClick(window.focusWidget(), Qt.Key_Space)
        first = window.part("records").kind_combo
        assert (window.current_place(), window.focusWidget(), follower.target()) == (("records", None), first, first)
    finally:
        window.hide()
        follower.stop()


def test_a_page_named_by_go_to_takes_the_keyboard_on_its_own_first_setting():
    """A move to a page of a part that has several -- Effective temperature's sweep study, a step of Parameter
    inference -- puts the keyboard on a setting of that page, never on one of a page not shown."""
    from core.gui.session import SbiSession
    qt_app()
    window = build_window()
    window.part("inference").session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"))
    window.part("inference").refresh_gates()
    held = {}
    for part, page, panel in (("temperature", "sweep", window.part("temperature").panels()[1]),
                              ("inference", "prior", window.part("inference").prior_panel)):
        window.go_to(part, page)
        held[page] = (window.current_place(), panel.isAncestorOf(window.focusWidget()))
    assert held == {"sweep": (("temperature", "sweep"), True), "prior": (("inference", "prior"), True)}, held


def test_go_to_gives_the_keyboard_as_a_move_does(monkeypatch):
    """``go_to`` gives the page the keyboard as a move, not as a key of the map, so that the ring shows only while
    the keyboard has the turn: a mouse click on Home's main button puts no ring on the page it opens. Escape in
    the rail gives the page the keyboard as a key does."""
    from PySide6.QtWidgets import QWidget
    qt_app()
    window = build_window()
    given, kept = [], QWidget.setFocus

    def set_focus(widget, *reason):
        given.append((widget, reason))
        return kept(widget, *reason)

    monkeypatch.setattr(QWidget, "setFocus", set_focus)
    for place in ("live", "models", "help", "home"):
        window.go_to(place)
    assert [reason for _widget, reason in given] == [(Qt.OtherFocusReason,)] * 4, given
    window.go_to("models", "builder")
    window.shell.rail.take_keyboard()
    given.clear()
    QTest.keyClick(window.focusWidget(), Qt.Key_Escape)
    assert given == [(window.part("models").builder.name_edit, (Qt.ShortcutFocusReason,))], given


def test_a_greyed_control_is_passed_over_for_the_next_setting():
    """A page whose first setting is greyed gives the keyboard to its next setting, not to nothing."""
    qt_app()
    window = build_window()
    panel = window.part("live")
    panel.model_combo.setEnabled(False)
    window.go_to("live")
    held = window.focusWidget()
    assert held is not None and held is not panel.model_combo and panel.isAncestorOf(held) and held.isEnabled(), held


def test_a_part_with_no_setting_keeps_the_keyboard_inside_it():
    """The search for a part's first setting stays inside the part: for a part with no setting of its own it finds
    nothing, never a control outside it."""
    from core.gui import main_window
    qt_app()
    window = build_window()
    try:
        _shown_off_screen(window)             # shown: a hidden window hides the walk's way out of the part
        assert main_window._first_control(window.part("help")) is None
    finally:
        window.hide()


def test_the_chain_follows_the_session_and_the_run(tmp_path):
    """The rail's six step rows read what the rules give for the session in use, with the narrowing round's
    observation on offer when its list holds one and each held record by its label: at the window's start, after
    a config is installed, after a prior with no name is held, after a posterior is held (the narrowing round can
    then start, the store holding an observation), while a step's run is live (that step reads Now) and after it
    ends. ``step_states`` is what was last worked out, and the rail shows it."""
    from core import config
    from core.artifacts import ArtifactStore, use_store
    from core.gui.session import SbiSession
    from tests._fixtures import build_browse_store
    qt_app()
    build_browse_store(tmp_path / "Artifacts")
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window = build_window()
        inference, rail = window.part("inference"), window.shell.rail
        offered = inference.tsnpe_panel.obs_picker.has_entries()
        assert offered, "the store's observation is not on offer"

        def rules(live=None):
            return chain.step_states(inference.session, live=live, observation_on_offer=offered,
                                     label=record_words.label_of)

        def spoken():
            return [rail.row(step.key).accessibleName() for step in keys.STEPS]

        assert inference.step_states() == rules() and spoken() == [state.spoken for state in rules()]
        inference.session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"))
        inference.install_config(SimpleNamespace(model="NADROWSKI", params_dict={}, rescale_params={},
                                                  force_params_dict={}, has_forcing=False, chi_mode=False,
                                                  chi_k_pad=config.CHI_K_PAD, observation_mode="spontaneous"))
        assert inference.step_states() == rules() and spoken() == [state.spoken for state in rules()]
        prior = SimpleNamespace(kind="prior", id="p1", name="",
                                manifest=SimpleNamespace(created="2026-10-03T14:02:00+00:00"))
        inference.session.inf_prior = prior
        inference.refresh_gates()
        made = datetime.fromisoformat(prior.manifest.created).astimezone()
        assert rules()[1].detail == f"Unnamed prior, {design.date_text(made)}"
        assert spoken() == [state.spoken for state in rules()]
        inference.session.posterior = SimpleNamespace(kind="posterior", id="q1", name="retrain", manifest=prior.manifest,
                                                      posterior=SimpleNamespace(truncation=None))
        inference.refresh_gates()
        assert spoken()[5] == "Step 6, Narrowing round. To come" and spoken() == [state.spoken for state in rules()]
        with running(inference.posterior_panel):
            assert inference.step_states() == rules(live="training")
            assert spoken()[2] == "Step 3, Training. Now: running"
        assert inference.step_states() == rules() and spoken() == [state.spoken for state in rules()]


def test_the_narrowing_round_reads_an_observation_written_after_the_window_opened(tmp_path):
    """The steps' states are worked out after the panels' own gates: once an observation is in the store and the
    gates run again (as they do after the inference step records one), the narrowing round no longer waits for an
    observation, in the rail."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.session import SbiSession
    from tests._fixtures import build_browse_store
    qt_app()
    build_browse_store(tmp_path / "full")
    made = SimpleNamespace(created="2026-10-03T14:02:00+00:00")
    session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"), cfg=object(),
                         inf_prior=SimpleNamespace(kind="prior", id="p1", name="p", manifest=made),
                         posterior=SimpleNamespace(kind="posterior", id="q1", name="q", manifest=made,
                                                   posterior=SimpleNamespace(truncation=None)))

    def spoken(on_offer):
        return chain.step_states(session, observation_on_offer=on_offer, label=record_words.label_of)[5].spoken

    with use_store(ArtifactStore(tmp_path / "empty")):
        window = build_window()
        window.part("inference").session = session
        window.part("inference").refresh_gates()
        assert window.shell.rail.row("narrowing").accessibleName() == spoken(False)
    with use_store(ArtifactStore(tmp_path / "full")):
        window.part("inference").refresh_gates()
        assert window.shell.rail.row("narrowing").accessibleName() == spoken(True) != spoken(False)


def test_a_step_names_a_held_record_by_the_name_its_save_box_gave_it(monkeypatch):
    """Naming the held prior, or the held posterior, in its Save box renames the record, and the step's row in
    the rail then names it by its new name."""
    from core.gui.panels.inference import posterior_tab, prior_tab
    from core.gui.session import SbiSession
    qt_app()
    window = build_window()
    inference, rail = window.part("inference"), window.shell.rail
    made = SimpleNamespace(created="2026-10-03T14:02:00+00:00")
    held = {kind: SimpleNamespace(kind=kind, id=f"{kind[0]}1", name="", manifest=made)
            for kind in ("prior", "posterior")}
    inference.session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"), cfg=object(), inf_prior=held["prior"],
                                   posterior=held["posterior"])
    inference.refresh_gates()
    assert rail.row("prior").accessibleName().startswith("Step 2, Prior. Done: Unnamed prior, ")
    for module in (prior_tab, posterior_tab):
        monkeypatch.setattr(module, "default_store", lambda: SimpleNamespace(rename=lambda kind, ref, name: made))
    inference.prior_panel.prior_name.setText("retrain_prior")
    inference.prior_panel._save_prior()
    assert rail.row("prior").accessibleName() == "Step 2, Prior. Done: retrain_prior"
    inference.posterior_panel.post_name.setText("retrain")
    inference.posterior_panel._save_posterior()
    assert rail.row("training").accessibleName() == "Step 3, Training. Done: retrain"


def test_the_foot_reads_nothing_is_running_at_start_and_follows_each_run_to_its_end(tmp_path, monkeypatch):
    """At the window's start the foot says nothing is running and is no control, and the window keeps no clock.
    While a panel's run is live the foot says the run's word, the stage it is in, in a few words, and that settings
    are locked; it is a button then. When the run ends the foot says how and when, and stays a button, until the end
    is seen; then it says nothing is running again, and a foot that held the keyboard hands it to the rail. At no
    moment does the window keep a clock: what the foot says is the run's status alone, and no timer that repeats is
    started by the window, the shell or the rail, whatever its parent."""
    from PySide6.QtCore import QTimer
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    window = build_window()
    idle = ("none", "Nothing is running", "Only one run at a time.", False)
    started = _clocks_started(monkeypatch)

    def foot():
        clocks = [timer for timer in window.findChildren(QTimer, options=Qt.FindDirectChildrenOnly)
                  if timer.isActive()]
        assert clocks == [], f"the window keeps a clock: {clocks}"
        assert _repeating(started) == [], f"the window keeps a clock: {started}"
        return _foot(window)

    assert foot() == idle and not hasattr(window, "_run_timer")
    for panel in (window.panel(CrossValPanel), window.part("inference").posterior_panel):
        with running(panel):
            assert foot() == ("now", "Training is running", "Running. Settings are locked.", True)
        assert foot() == ("done", "Training finished", "Thu 8 Oct, 19:02.", True), "the run's end is not said"
        window.shell.rail.foot.setFocus(Qt.TabFocusReason)
        held.acknowledge(held.status.run)
        assert foot() == idle
        assert window.focusWidget() is window.shell.rail.row("home"), "the end was seen and the keyboard went nowhere"


def test_the_foots_words_are_written_in_digits_of_equal_width(tmp_path):
    """The foot's words are laid out in the small lettering with digits of equal width, so that a count among them
    keeps its width as it moves: the batch a training's simulation is at, counted on, keeps the words' width.
    Without the feature the same font gives two widths."""
    from PySide6.QtGui import QFont, QFontMetricsF
    from core.gui import fonts
    from core.gui.run.model import SIMULATION_ROW
    app = qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    window = build_window()
    text, stand_in = RunText(held), StandIn()
    start_run(window.part("inference").posterior_panel, stand_in, spec=_training_spec())
    for line in (_BUDGET, _ROTATING, _ROTATED, _writing(tmp_path / "simulations" / "_unnamed__20261008T190211")):
        stand_in.say(line)
    laid_out = []
    for done in (6_111, 6_888):
        stand_in.row(SIMULATION_ROW, done, 10_000)
        turn_until(app, lambda: held.model.batches_done() == done)
        laid_out.append(_foot(window)[2])
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    assert laid_out == ["Simulating, batch 6,111 of 10,000. Settings are locked.",
                        "Simulating, batch 6,888 of 10,000. Settings are locked."], laid_out
    font, equal, small = fonts.inter("small", tabular=True), QFont.Tag.fromString("tnum"), fonts.inter("small")
    assert font.isFeatureSet(equal)
    assert (font.family(), font.pixelSize(), font.weight()) == (small.family(), small.pixelSize(), small.weight())
    widths = [QFontMetricsF(font).horizontalAdvance(words) for words in laid_out]
    assert widths[0] == widths[1], widths
    without = QFont(font)
    without.unsetFeature(equal)
    one, other = (QFontMetricsF(without).horizontalAdvance(words) for words in laid_out)
    assert one != other, "the digits have one width without the feature as well: the lines above prove nothing"


def test_clicking_the_foot_opens_the_running_place_and_a_bare_run_host_goes_nowhere(monkeypatch):
    """While a run is live the foot is a button that goes to the run's place, its page or step too, as
    ``go_to_run`` does, a step that cannot start among them, the place taking the keyboard as after a click. Once the
    run has ended the foot is a button still, until its end is seen: a click goes to the run's place and sees it. A
    run of a panel the window has no place for -- the bare run host a test builds -- is named by its word in the
    foot's title and its accessible name; the foot is no button, live or ended, and the window stays where it is."""
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    given = _given(monkeypatch)
    window = build_window()
    foot, held = window.shell.rail.foot, controller()
    for panel, place in ((window.panel(CrossValPanel), ("temperature", "sweep")),
                         (window.part("inference").validate_panel, ("inference", "checks"))):
        window.go_to("home")
        window.part("temperature").show_page("one_cell")
        with running(panel):
            del given[:]
            QTest.mouseClick(foot, Qt.LeftButton, Qt.NoModifier, foot.rect().center())
            assert window.current_place() == place
            assert given[-1:] == [(window.focusWidget(), Qt.OtherFocusReason)], (
                f"{place}: a click on the foot gave the keyboard otherwise than as a click gives it")
        window.go_to("home")
        assert (held.status.unacknowledged_end, foot.is_button()) == (True, True), place
        QTest.mouseClick(foot, Qt.LeftButton, Qt.NoModifier, foot.rect().center())
        assert (window.current_place(), held.status.state, foot.is_button()) == (place, "idle", False), (
            f"{place}: a click on the foot of an ended run did not go to it and see its end")
    orphan = build_run_host()
    window.go_to("live")
    with running(orphan):
        assert (foot._title, foot.is_button()) == ("Training is running", False)
        assert (foot.accessibleName(), foot.toolTip()) == ("Training is running. Running. Settings are locked.", "")
        QTest.mouseClick(foot, Qt.LeftButton, Qt.NoModifier, foot.rect().center())
        window.go_to_run()
        assert window.current_place() == ("live", None), "a panel with no place moved the window"
    assert (foot._title, foot.is_button(), held.status.unacknowledged_end) == ("Training finished", False, True)
    QTest.mouseClick(foot, Qt.LeftButton, Qt.NoModifier, foot.rect().center())
    window.go_to_run()
    assert window.current_place() == ("live", None), "an ended run with no place moved the window"


def test_no_row_carries_a_mark_for_a_live_run_and_nothing_is_disabled():
    """A live run is said at the foot alone: no row of the rail is marked for it or carries a mark of it, no page
    name does, and no row of Home's three tables, which Home reads in sight as the run starts; the page names keep
    their words. Nothing is disabled by it: Home's main button goes to the run, every part and every step still
    opens and no part is disabled as a whole, so that every list and page stays usable, and Effective
    temperature's other page can still be chosen by its name."""
    from PySide6.QtWidgets import QAbstractButton
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    window = build_window()
    rail, home = window.shell.rail, window.part("home")
    names = window.part("temperature").names.findChildren(QAbstractButton)
    page_words = [name.text() for name in names]
    rows = [place.key for place in keys.PARTS] + [step.key for step in keys.STEPS]

    def seen():
        return (_marked(rail), [rail.row(key).text() for key in rows],
                [_cells(table) for table in (home.parts_table, home.records_table)],
                [row[0] for row in _cells(home.steps_table)])

    def during(panel, where):
        """A run of ``panel`` live, started while Home is in sight; ``where`` is the run's place."""
        window.go_to("home")
        with running(panel):
            assert home.next_heading.label.text() == "Next: nothing until the training ends", "Home did not read"
            assert seen() == before, "a row carries a mark of the run"
            assert [name.text() for name in names] == page_words
            home.next_button.click()
            assert window.current_place() == where, (
                f"Home's main button did not go to the run: {window.current_place()}")
            for place in keys.PARTS:
                window.go_to(place.key)
                assert window.current_place()[0] == place.key and rail.row(place.key).isEnabled()
                assert window.part(place.key).isEnabled(), f"{place.key} is disabled by a live run"
            for step in keys.STEPS:
                window.go_to("inference", step.key)
                assert window.current_place() == ("inference", step.key) and rail.row(step.key).isEnabled()
            window.go_to("temperature", "one_cell")
            _name(window.part("temperature"), "sweep").click()
            assert window.current_place() == ("temperature", "sweep") and all(name.isEnabled() for name in names)

    try:
        _shown_off_screen(window)
        before = seen()
        for panel, where in ((window.panel(CrossValPanel), ("temperature", "sweep")),
                             (window.part("inference").posterior_panel, ("inference", "training"))):
            during(panel, where)
    finally:
        window.hide()


def test_the_foot_says_each_state_of_a_run_in_its_words(tmp_path, monkeypatch):
    """The foot says each state of a run from the run's status alone, with no clock: nothing running, and no
    control; a run live, titled by its word, with the stage it is in, in a few words, and that settings are locked;
    a run that needs someone, with why; a run stopping, with when its stop is taken; a run that only loads a record;
    and an end nobody has seen: finished with its date, stopped and failed with the date and what was kept, and a
    run PRISM did not see end, as a failure dated when it was last heard. A date within seven days carries its
    weekday. Every state but the first is a button. No word the window writes there holds a hyphen, which the
    words' digits of equal width would widen. No timer that repeats is started by the window, the shell or the
    rail at any moment, whatever its parent."""
    from PySide6.QtCore import QTimer
    from core.artifacts import default_store
    from core.gui.run.model import SILENCE_NEEDS_YOU_S, SIMULATION_ROW, RunModel, RunSpec
    from core.SBI.fit_state import KeptNote
    from tests._fixtures import choose
    app = qt_app()
    held, clock = _on_the_runs_clock(tmp_path)
    window = build_window()
    host, text = window.part("inference").posterior_panel, RunText(held)
    folder = tmp_path / "simulations" / "_unnamed__20261008T190211"
    written, started = [], _clocks_started(monkeypatch)

    def foot():
        clocks = [timer for timer in window.findChildren(QTimer, options=Qt.FindDirectChildrenOnly)
                  if timer.isActive()]
        assert clocks == [], f"the window keeps a clock: {clocks}"
        assert _repeating(started) == [], f"the window keeps a clock: {started}"
        written.append(_foot(window)[2])
        return _foot(window)

    def said(stand_in, *lines):
        before = text.texts().count(lines[-1])
        for line in lines:
            stand_in.say(line)
        turn_until(app, lambda: text.texts().count(lines[-1]) > before)

    def seen():
        held.acknowledge(held.status.run)
        assert foot() == ("none", "Nothing is running", "Only one run at a time.", False)

    def running_words(words):
        return "now", "Training is running", f"{words}. Settings are locked.", True

    assert foot() == ("none", "Nothing is running", "Only one run at a time.", False)
    choose(monkeypatch, words="Stop training")
    stand_in = StandIn(takes_stops=False)
    stand_in.kept_fit = KeptNote("k1", 3, False, True, 1)
    start_run(host, stand_in, spec=_training_spec())
    assert foot() == running_words("Running")
    said(stand_in, _BUDGET, _ROTATING)
    assert foot() == running_words("Computing the rotation")
    said(stand_in, _ROTATED, _writing(folder))
    stand_in.row(SIMULATION_ROW, 6_150, 10_000)
    turn_until(app, lambda: held.model.batches_done() == 6_150)
    assert foot() == running_words("Simulating, batch 6,150 of 10,000")
    clock.advance(SILENCE_NEEDS_YOU_S + 1)
    held.tick()
    assert held.status.needs_you and foot() == ("warn", "Training needs you", held.status.needs_you, True)
    said(stand_in, f"[checkpoint] complete: 10000 batches in {folder}. Safe to delete once the posterior is saved; "
                   f"keeping it lets you retrain the flow without re-simulating.")
    clock.advance(3780)
    said(stand_in, _epoch(1, 0), _epoch(2, 1), _epoch(3, 2))
    assert foot() == running_words("Fitting, epoch 3")
    estimated = RunModel(_training_spec(), clock=clock, expected=(130, "about 130 epochs"))
    for line in (_BUDGET, _ROTATING, _ROTATED, _writing(folder), _epoch(1, 0)):
        estimated.feed([(line, "info")])
    estimated.feed([(f"[checkpoint] complete: 10000 batches in {folder}. Safe to delete once the posterior is "
                     f"saved; keeping it lets you retrain the flow without re-simulating.", "info"),
                    (_epoch(2, 0), "info")])
    assert estimated.doing() == "Fitting, epoch 2 of about 130", estimated.doing()
    host.run_view.stop_button.click()
    assert foot() == ("now", "Training is stopping", "Stopping: takes effect when this epoch ends, up to about 63 "
                                                     "minutes.", True)
    stand_in.say(_epoch(4, 3))
    turn_until(app, lambda: not held.is_live())
    stopped = ("Thu 8 Oct, 20:15. The 20,480,000 simulations are kept; the training is kept up to epoch 3, so you can "
               "resume it.")
    assert foot() == ("half", "Training stopped", stopped, True)
    seen()
    stand_in = StandIn()
    start_run(host, stand_in, spec=_training_spec())
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    assert foot() == ("done", "Training finished", "Thu 8 Oct, 20:15.", True)
    seen()
    stand_in = StandIn()
    start_run(host, stand_in, spec=_training_spec())
    said(stand_in, _BUDGET, _ROTATING, _ROTATED, _writing(folder))
    stand_in.fail(RuntimeError("the run broke"))
    turn_until(app, lambda: not held.is_live())
    assert foot() == ("err", "Training failed", "Thu 8 Oct, 20:15. No simulation was saved.", True)
    seen()
    loads = StandIn(word="Prior load")
    start_run(host, loads, spec=RunSpec("Prior load", ("inference", "prior"), loads_only=True, keeps_files=False,
                                        stop_asks=False))
    assert foot() == ("now", "Loading…", "", True)
    loads.release()
    turn_until(app, lambda: not held.is_live())
    assert foot() == ("none", "Nothing is running", "Only one run at a time.", False), "a load left an end to see"
    _ended_history(tmp_path / "runs", "20261003T010000-training", default_store().root, outcome="not seen to end",
                   stage="Simulation", ended=None, last_heard=_at(3, 3, 12), tries=[], fit=None,
                   simulation_set={"folder": "eda5c2a1e468", "batches": 10_000, "at_end": 6_150})
    held.recall(str(default_store().root))
    assert foot() == ("err", "Training failed", "Sat 3 Oct, 03:12. What it had saved by then is kept.", True)
    seen()
    assert [words for words in written if "-" in words] == [], "a hyphen in the foot's own words"


def test_an_end_is_acknowledged_once_its_page_has_been_seen_in_the_active_window(tmp_path):
    """An end is seen, and said at the foot no more, once its run's page has been in sight while the window was the
    one in front: looked at as the run ends, at every move and when the window comes to the front. Its history then
    says it was seen. The run's page in sight in a window not in front, or another page in sight in the window in
    front, sees nothing. A run started while an end waits to be seen sets that end aside: the window says the new
    run, and the end is written seen, so that the next launch does not say it."""
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.run import history
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    window = build_window()
    panel, place = window.panel(CrossValPanel), ("temperature", "sweep")

    def end_of_a_run():
        """A run of the sweep study, ended; its id."""
        with running(panel):
            run = held.status.run
        return run

    def seen(run):
        record = history.read(tmp_path / "runs" / f"{run}.json")
        return record["acknowledged"], held.status.unacknowledged_end, window.shell.rail.foot._title

    try:
        _shown_off_screen(window)
        window.go_to(*place)
        run = end_of_a_run()
        assert seen(run) == (False, True, "Training finished"), "seen in a window not in front"
        with _in_front(window):
            assert seen(run) == (True, False, "Nothing is running"), "not seen as the window came to the front"
            window.go_to("home")
            run = end_of_a_run()
            assert seen(run) == (False, True, "Training finished"), "seen with another page in sight"
            window.go_to(*place)
            assert seen(run) == (True, False, "Nothing is running"), "not seen as its page came into sight"
            run = end_of_a_run()
            assert seen(run) == (True, False, "Nothing is running"), "not seen as it ended in sight"
            window.go_to("home")
            run = end_of_a_run()
            assert seen(run) == (False, True, "Training finished")
            later = end_of_a_run()
            assert (seen(run)[0], seen(later)) == (True, (False, True, "Training finished")), (
                "an end set aside by the next run was not written seen")
    finally:
        window.hide()


def test_ctrl_r_opens_the_run_whenever_the_foot_is_a_button(tmp_path):
    """Ctrl+R goes where the foot goes, whenever the foot is a button: to the live run's place; to the page of a run
    whose end nobody has seen, which it then sees, a finished run's figures in sight, its result; and to the page
    of an end an earlier PRISM left unseen, drawn in its past view from the run's files. While nothing runs and no
    end waits to be seen, it does nothing."""
    from core.artifacts import default_store
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    window = build_window()
    foot = window.shell.rail.foot
    try:
        _shown_off_screen(window)
        with _in_front(window):
            _press(window, "Ctrl+R")
            assert (window.current_place(), foot.is_button()) == (("home", None), False)
            with running(window.panel(CrossValPanel)):
                _press(window, "Ctrl+R")
                assert window.current_place() == ("temperature", "sweep"), "Ctrl+R did not go to the live run"
                window.go_to("home")
            assert (held.status.unacknowledged_end, foot.is_button()) == (True, True)
            _press(window, "Ctrl+R")
            sweep = window.panel(CrossValPanel)
            assert (window.current_place(), held.status.state, sweep.results.currentWidget() is sweep.figures) == (
                ("temperature", "sweep"), "idle", True), "Ctrl+R did not go to the finished run's result and see it"
            window.go_to("home")
            _press(window, "Ctrl+R")
            assert window.current_place() == ("home", None), "Ctrl+R moved the window with no run to go to"
            run = "20261003T010000-training"
            _ended_history(tmp_path / "runs", run, default_store().root)
            held.recall(str(default_store().root))
            assert (held.status.run, foot.is_button()) == (run, True)
            _press(window, "Ctrl+R")
            view = window.part("inference").posterior_panel.past_view
            assert (window.current_place(), view.mode, held.status.state) == (("inference", "training"), "past",
                                                                              "idle")
    finally:
        window.hide()


def test_during_a_run_every_setting_is_locked_but_appearance_and_every_page_reads(tmp_path, monkeypatch):
    """From a run's start to its end every setting of every page that starts runs is locked; Records' note, its
    delete and its clearing of leftovers, and Your own models' validate, save and delete, are refused after the
    click, each with its own sentence, as they are today, and change nothing; while the rail, every part and every
    step, Home, Help, Records' lists and the choice of appearance stay usable. At the run's end every setting is
    free again."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.crossval_panel import CrossValPanel
    from tests._fixtures import all_asked, build_browse_store, forget_asked
    qt_app()
    _on_the_runs_clock(tmp_path)
    ids = build_browse_store(tmp_path / "Artifacts")
    store = ArtifactStore(tmp_path / "Artifacts")
    with use_store(store):
        window = build_window()
        for place in keys.PARTS:
            window.part(place.key)
        panels = window._all_panels()
        records, models = window.part("records"), window.part("models")
        records.note_edit.setText("written during a run")
        note = store.get("prior", ids["prior"]).note
        leftover = store.kind_dir("prior") / "_unnamed__20260917T090000"
        leftover.mkdir(parents=True, exist_ok=True)
        assert all(panel.controls.isEnabled() for panel in panels)
        forget_asked()
        with running(window.panel(CrossValPanel)):
            assert [panel for panel in panels if panel.controls.isEnabled()] == [], "a setting is not locked"
            refused = []
            for action, doing in ((records._set_note, "setting a note"), (records._delete, "deleting an artifact"),
                                  (lambda: records._sweep(all_kinds=False), "removing leftovers")):
                records._set_status("")
                action()
                refused.append(records.status.text())
            assert [f"A task is running -- wait for it to finish before {doing}." in said for said, doing in zip(
                refused, ("setting a note", "deleting an artifact", "removing leftovers"))] == [True] * 3, refused
            assert (store.get("prior", ids["prior"]).note, leftover.is_dir()) == (note, True), (
                "a refused action changed the store")
            builder = models.builder
            builder._validate()
            validating = builder.status.text()
            builder._save()
            assert (validating, builder.status.text()) == (
                "⚠ A task is running -- wait for it to finish before validating.",
                "⚠ A task is running -- wait for it to finish before saving.")
            models._delete_user_model("BROKEN")
            assert [(told.title, told.sentence) for told in all_asked()] == [
                ("A task is running", "Wait for the running task to finish before deleting a model.")]
            for place in keys.PARTS:
                window.go_to(place.key)
                assert window.current_place()[0] == place.key and window.shell.rail.row(place.key).isEnabled()
                assert window.part(place.key).isEnabled(), place.key
            for step in keys.STEPS:
                window.go_to("inference", step.key)
                assert window.current_place() == ("inference", step.key)
            home = window.part("home")
            assert all(table.isEnabled() for table in (home.steps_table, home.parts_table, home.records_table))
            assert (records.table.isEnabled(), records.kind_combo.isEnabled()) == (True, True)
            records.kind_combo.setCurrentIndex(records.kind_combo.findData("posterior"))
            assert records.kind() == "posterior", "Records' list could not be changed"
            radios = window.part("settings")._radios
            assert radios and all(radio.isEnabled() for radio in radios.values()), "appearance was locked"
        assert all(panel.controls.isEnabled() for panel in panels), "a setting stayed locked after the run"


def test_the_chain_says_stopping_and_the_batch_a_stopped_training_kept(tmp_path, monkeypatch):
    """The steps read the run's status. A run live on a step other than the step in hand reads "Now: running" on
    that step, and the step in hand keeps its own "Now". A training stopping reads "Now: stopping" on its step; once
    stopped with batches saved in its simulation set, the training's step says the batch it can be continued from,
    in the rail and in Home's table, after its end has been seen too."""
    import torch
    from core.gui.session import SbiSession
    from tests._fixtures import choose
    app = qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    window = build_window()
    inference, rail, home = window.part("inference"), window.shell.rail, window.part("home")
    inference.session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"),
                                   inf_prior=SimpleNamespace(name="retrain_prior"))
    inference.refresh_gates()
    _shown_off_screen(window)                       # shown: Home reads as it is opened

    def spoken(*steps):
        return [rail.row(step).accessibleName() for step in steps]

    assert spoken("training", "checks") == ["Step 3, Training. Now: setting up",
                                                "Step 4, Checks. To come: needs a posterior"]
    with running(inference.validate_panel):
        assert spoken("training", "checks") == ["Step 3, Training. Now: setting up", "Step 4, Checks. Now: running"]
    held.acknowledge(held.status.run)
    folder = tmp_path / "simulations" / "_unnamed__20261003T030211"
    folder.mkdir(parents=True)
    torch.save({"batches_done": 6_150, "complete": False}, folder / "state.pt")
    text, stand_in = RunText(held), StandIn(takes_stops=False)
    start_run(inference.posterior_panel, stand_in, spec=_training_spec())
    for line in (_BUDGET, _ROTATING, _ROTATED, _writing(folder)):
        stand_in.say(line)
    turn_until(app, lambda: _writing(folder) in text.texts())
    assert spoken("training") == ["Step 3, Training. Now: running"]
    choose(monkeypatch, words="Stop and keep the simulations")
    inference.posterior_panel.run_view.stop_button.click()
    assert spoken("training") == ["Step 3, Training. Now: stopping"]
    stand_in.say("a line of the run")
    turn_until(app, lambda: not held.is_live())
    kept = ["Step 3, Training. Unfinished: simulations saved to batch 6,150"]
    assert (held.status.continue_from, spoken("training")) == (6_150, kept)
    held.acknowledge(held.status.run)
    window.go_to("home")
    assert spoken("training") == kept and _cells(home.steps_table)[2] == [
        "3 Training", ("half", "Unfinished"), "Simulations saved to batch 6,150"]
    held.forget_offers()                              # as the end of every test does
    inference.recompute_states()
    assert spoken("training") == ["Step 3, Training. Now: setting up"], "the batch outlived the offers"
    window.hide()


def test_the_keyboard_leaves_a_foot_that_stops_being_a_button(tmp_path):
    """The foot holds the keyboard while it is a button, for a live run and for an end nobody has seen; when it stops
    being one -- the end is seen, or a run that only loads a record ends into nothing -- the rail takes the
    keyboard, on the row of the place in sight."""
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.run.model import RunSpec
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    window = build_window()
    foot, rail = window.shell.rail.foot, window.shell.rail
    with running(window.panel(CrossValPanel)):
        pass
    foot.setFocus(Qt.TabFocusReason)
    assert (foot.is_button(), window.focusWidget() is foot) == (True, True)
    held.acknowledge(held.status.run)
    assert (foot.is_button(), window.focusWidget() is rail.row("home")) == (False, True), (
        "the end was seen and the foot kept the keyboard")
    loads = StandIn(word="Prior load")
    start_run(window.part("inference").prior_panel, loads, spec=RunSpec(
        "Prior load", ("inference", "prior"), loads_only=True, keeps_files=False, stop_asks=False))
    foot.setFocus(Qt.TabFocusReason)
    loads.release()
    turn_until(qt_app(), lambda: not held.is_live())
    assert (foot.is_button(), window.focusWidget() is rail.row("home")) == (False, True), (
        "a load ended and the foot kept the keyboard")


def test_the_foots_tool_tip_follows_its_own_rule(tmp_path):
    """With no clock, the foot's tool tip is the foot's own: none while its words stand whole, and its title and its
    words when they are cut; its accessible name is always its title and its words."""
    from core.artifacts import default_store
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path, _at(12, 9, 0))
    window = build_window()
    foot = window.shell.rail.foot
    whole = "Training is running. Running. Settings are locked."
    with running(window.panel(CrossValPanel)):
        assert (foot.accessibleName(), foot.toolTip()) == (whole, ""), (foot.accessibleName(), foot.toolTip())
    held.acknowledge(held.status.run)
    _ended_history(tmp_path / "runs", "20261003T010000-training", default_store().root)
    held.recall(str(default_store().root))
    said = ("Training stopped. Sun 11 Oct, 12:31. The 20,480,000 simulations are kept; the training is kept up to "
            "epoch 62, so you can resume it.")
    assert (foot.accessibleName(), foot.toolTip()) == (said, said), (foot.accessibleName(), foot.toolTip())


def _open_last_run(window, run):
    """Home opened, and its "Last runs" row of the run ``run`` clicked."""
    window.go_to("home")
    rows = [row.run for row in window._home_facts().runs]
    view = window.part("home").runs_table.view
    QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier,
                     view.visualRect(view.model().index(rows.index(run), 0)).center())


def test_a_row_of_last_runs_for_the_live_run_opens_its_page_and_leaves_its_end_to_come(tmp_path):
    """During a run, Home's "Last runs" lists it, running, and its row opens its page; the run is not seen by it, nor
    by the run's controller asked to see it, since its end is still to come: its history still says that nobody has
    seen its end, so that a PRISM that closed then would say it at the next launch. At its end, its page in sight in
    the window in front sees it."""
    import shutil
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.run import history
    app = qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    folder, root = tmp_path / "runs", tmp_path / "Artifacts"
    with use_store(ArtifactStore(root)):
        window = build_window()
        try:
            _shown_off_screen(window)
            with _in_front(window):
                stand_in = StandIn()
                start_run(window.panel(CrossValPanel), stand_in)
                run = held.status.run
                _open_last_run(window, run)
                during = history.read(folder / f"{run}.json")
                assert (window.current_place(), held.is_live(), during["outcome"], during["acknowledged"]) == (
                    ("temperature", "sweep"), True, "running", False), "a row of Last runs wrote the live run seen"
                held.acknowledge(run)                     # asked of the run's controller itself, the window aside
                assert (held.is_live(), history.read(folder / f"{run}.json")["acknowledged"]) == (True, False), (
                    "the run's controller wrote the live run seen")
                crashed = tmp_path / "crashed"            # the file as a PRISM that closed now would leave it
                crashed.mkdir()
                shutil.copy(folder / f"{run}.json", crashed / f"{run}.json")
                assert (history.close_unseen(crashed),
                        history.newest_unseen(crashed, records_root=str(root))["id"]) == ((run,), run), (
                    "a PRISM that closed during the run would not say it at the next launch")
                stand_in.release()
                turn_until(app, lambda: not held.is_live())
                assert (history.read(folder / f"{run}.json")["acknowledged"], held.status.state) == (True, "idle"), (
                    "the run's end, its page in sight in the window in front, was not seen")
        finally:
            window.hide()


def test_a_past_view_stands_beside_the_pages_own_run_and_holds_none_of_its_figures(tmp_path):
    """A run that is over, opened from Home's "Last runs", is drawn in its page's past view, beside the run page and
    never in its place, and holds no figures. A training that finished with a figure leaves the figure on its page;
    an older training's past view opened then shows none, the figure staying the page's own, and the finished run's
    page is as it was: "Back to the form" gives the figures back, "How the run went" above them opening the finished
    run's page. A training that then stops with batches kept keeps its page: an older run's past opened over it,
    "Back to the form" and "How the run went" bring the stopped run's page back as it was at its end, with its
    "Continue from batch N", and so does the stopped run's row, also once a run of another page has ended since,
    while the stopped run's page still offers to continue it."""
    import torch
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.run.model import SIMULATION_ROW
    app = qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    folder, root = tmp_path / "runs", tmp_path / "Artifacts"
    older = "20261001T120000-training"
    with use_store(ArtifactStore(root)):
        _ended_history(folder, older, root, outcome="finished", acknowledged=True, started=_at(1, 12, 0),
                       ended=_at(2, 12, 0), stage="Saving", fit=None)
        window = build_window()
        host = window.part("inference").posterior_panel
        try:
            _shown_off_screen(window)
            text, finished = RunText(held), StandIn()
            start_run(host, finished, spec=_training_spec(), provide_fig_sink=True)
            finished.figure("the loss by epoch")
            turn_until(app, lambda: host.figure_stack.count() == 1)
            finished.release()
            turn_until(app, lambda: not held.is_live())
            own = host.run_view.sentence.text()
            _open_last_run(window, older)
            assert (host.run_view.mode, host.run_view.sentence.text()) == ("ended", own), (
                "an older run's past was drawn on the finished run's page")
            past = host.past_view
            assert (window.current_place(), host.results.currentWidget() is past, past.mode, past.sentence.text()) == (
                ("inference", "training"), True, "past", "Training finished on Fri 2 Oct, 12:00, after 1 day."), (
                past.sentence.text())
            assert (past.isAncestorOf(host.figure_stack), past.figures.isVisibleTo(past),
                    host.figure_stack.count()) == (False, False, 1), "the past view shows the page's own figures"
            past.back_button.click()
            assert (host.results.currentWidget() is host.figures, host.run_view.how_link.isVisibleTo(host)) == (
                True, True), "Back to the form left the page's own run out of reach"
            host.run_view.how_link.click()
            assert (host.results.currentWidget() is host.run_view, host.run_view.sentence.text()) == (
                True, own), "How the run went did not open the page's own run"

            sets = tmp_path / "simulations" / "_unnamed__20261008T190211"
            sets.mkdir(parents=True)
            torch.save({"batches_done": 6_150, "complete": False}, sets / "state.pt")
            stopped = StandIn()
            start_run(host, stopped, spec=_training_spec())
            for line in (_BUDGET, _ROTATING, _ROTATED, _writing(sets)):
                stopped.say(line)
            stopped.row(SIMULATION_ROW, 6_150, 10_000)
            turn_until(app, lambda: held.model.batches_done() == 6_150)
            held.request_stop()
            turn_until(app, lambda: not held.is_live())
            run, page = held.status.run, host.run_view

            def offered():
                return (host.results.currentWidget() is page, page.mode, page.continue_button.isVisibleTo(page),
                        page.continue_button.text())

            assert offered() == (True, "ended", True, "Continue from batch 6,150"), offered()
            _open_last_run(window, older)
            assert host.results.currentWidget() is host.past_view
            host.past_view.back_button.click()
            assert (host.results.currentWidget() is host.figures, host.run_view.how_link.isVisibleTo(host)) == (
                True, True), "Back to the form left the stopped run's page out of reach"
            host.run_view.how_link.click()
            assert offered() == (True, "ended", True, "Continue from batch 6,150"), (
                "How the run went did not bring back the stopped run's page and its offer", offered())
            _open_last_run(window, older)
            _open_last_run(window, run)
            assert offered() == (True, "ended", True, "Continue from batch 6,150"), (
                "the stopped run's row did not bring back its page and its offer", offered())
            with running(window.panel(CrossValPanel)):
                pass
            _open_last_run(window, run)
            assert offered() == (True, "ended", True, "Continue from batch 6,150"), (
                "once another page's run had ended, the stopped run's row lost its page and its offer", offered())
        finally:
            window.hide()


def test_during_a_training_a_row_of_an_earlier_training_leaves_the_live_page_in_sight(tmp_path):
    """While a training is live, the row of an earlier training in Home's "Last runs" opens the training's page,
    which shows the live run as it was: nothing of the past is drawn over it, and the earlier run is not seen by
    it."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.run import history
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    folder, root = tmp_path / "runs", tmp_path / "Artifacts"
    earlier = "20261001T120000-training"
    with use_store(ArtifactStore(root)):
        window = build_window()
        host = window.part("inference").posterior_panel
        _ended_history(folder, earlier, root, started=_at(1, 12, 0))
        try:
            _shown_off_screen(window)
            with running(host, spec=_training_spec()):
                run = held.status.run
                _open_last_run(window, earlier)
                assert (window.current_place(), host.results.currentWidget() is host.run_view, host.run_view.mode,
                        held.is_live(), held.status.run) == (("inference", "training"), True, "live", True, run), (
                    "the live training's page did not stay in sight")
                assert history.read(folder / f"{earlier}.json")["acknowledged"] is False, (
                    "the earlier run was seen though nothing of it was shown")
        finally:
            window.hide()


def test_an_end_during_windows_end_of_the_session_is_not_seen_until_it_is_cancelled(tmp_path):
    """From Windows' question to end the session until it is cancelled, nothing is seen: a run that "shut down
    anyway" stopped in that time, its page in sight in the window in front, is said at the foot and in its history
    as an end nobody has seen, so that the next launch says it. Once Windows' end is cancelled, a look sees it."""
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.run import history
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    window = build_window()
    try:
        _shown_off_screen(window)
        window.go_to("temperature", "sweep")
        with _in_front(window):
            start_run(window.panel(CrossValPanel), StandIn())
            run = held.status.run
            window.session_asked(SimpleNamespace(cancel=lambda: None))
            window.session_ending(True)           # "shut down anyway": the run is stopped, and the window waits
            record = history.read(tmp_path / "runs" / f"{run}.json")
            assert (held.is_live(), record["outcome"], record["stop_reason"], record["acknowledged"],
                    held.status.unacknowledged_end, window.current_place()) == (
                False, "stopped", "Windows shut down", False, True, ("temperature", "sweep")), (
                "an end during Windows' end of the session was seen")
            window.session_ending(False)          # cancelled after all
            window.go_to("home")
            window.go_to("temperature", "sweep")
            assert (history.read(tmp_path / "runs" / f"{run}.json")["acknowledged"], held.status.state) == (
                True, "idle"), "the end was not seen once Windows' end of the session was cancelled"
    finally:
        window.hide()


def test_an_end_whose_page_this_prism_does_not_have_is_seen_when_its_page_is_asked_for(tmp_path, caplog):
    """An end an earlier PRISM left unseen whose place this window does not have -- a part, or a step of one, that a
    later PRISM names otherwise -- cannot have its page opened: asked for from the foot, it is seen all the same, so
    that the foot moves on, and the window's log says its page no longer exists."""
    import logging
    from core.artifacts import default_store
    from core.gui.run import history
    qt_app()
    held, _clock = _on_the_runs_clock(tmp_path)
    window = build_window()
    foot, folder, root = window.shell.rail.foot, tmp_path / "runs", default_store().root
    for run, place in (("20261003T010000-training", ["fitting room", None]),
                       ("20261004T010000-training", ["inference", "posterior"])):
        _ended_history(folder, run, root, place=place, started=_at(int(run[6:8]), 1, 0))
        held.recall(str(root))
        assert (held.status.run, foot.is_button()) == (run, True), held.status
        with caplog.at_level(logging.WARNING, logger="core.gui.main_window"):
            QTest.mouseClick(foot, Qt.LeftButton, Qt.NoModifier, foot.rect().center())
        assert (history.read(folder / f"{run}.json")["acknowledged"], held.status.state, foot._title,
                foot.is_button(), window.current_place()) == (True, "idle", "Nothing is running", False,
                                                              ("home", None)), place
        said = [record.getMessage() for record in caplog.records if record.name == "core.gui.main_window"]
        where = "/".join(part for part in place if part)
        assert said[-1:] == [f"The page of the run {run} no longer exists in this PRISM ({where}); its end is taken "
                             f"as seen."], said


def test_go_to_records_with_a_kind_opens_records_at_that_kind():
    """A record's kind opens Records listing that kind; Records opened with no kind keeps the kind last chosen."""
    qt_app()
    window = build_window()
    window.go_to("records", "posterior")
    assert window.current_place() == ("records", None) and window.part("records").kind() == "posterior"
    window.go_to("home")
    window.go_to("records")
    assert window.part("records").kind() == "posterior"
    window.go_to("records", "calibration")
    assert window.part("records").kind() == "calibration"


def test_home_is_read_again_when_opened_and_when_the_store_changes(tmp_path, monkeypatch):
    """Home reads once while the window is built and nothing more at its first show. A record deleted in
    Records is listed no more when Home is next opened, and is not read for while another place is shown; while
    Home is shown a change of the store reads it at once; opened while it is shown, it reads again."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui import main_window
    from tests._fixtures import build_browse_store
    qt_app()
    ids = build_browse_store(tmp_path / "Artifacts")
    store = ArtifactStore(tmp_path / "Artifacts")
    reads, read = [], main_window._home_records
    monkeypatch.setattr(main_window, "_home_records", lambda used: (reads.append(used), read(used))[1])
    with use_store(store):
        window = build_window()
        home = window.part("home")
        try:
            _shown_off_screen(window)
            assert len(reads) == 1, "Home read again at its first show"
            first, second = [label for label in (row[0] for row in _cells(home.records_table))
                             if label.startswith("browse_")][:2]
            window.go_to("records")
            store.delete(first[len("browse_"):], ids[first[len("browse_"):]], force=True)
            window.part("records").store_changed.emit()
            assert len(reads) == 1, "Home read while another place was shown"
            window.go_to("home")
            assert len(reads) == 2 and first not in [row[0] for row in _cells(home.records_table)]
            store.delete(second[len("browse_"):], ids[second[len("browse_"):]], force=True)
            window.part("records").store_changed.emit()
            assert len(reads) == 3, "a change of the store did not read Home while it was shown"
            assert second not in [row[0] for row in _cells(home.records_table)]
            window.go_to("home")
            assert len(reads) == 4, "Home opened while it was shown did not read again"
        finally:
            window.hide()


def test_a_maximised_window_at_a_large_text_size_keeps_the_usual_layout(monkeypatch):
    """At 150 % text on a 1920 by 1080 screen (its taskbar below) the small layout begins under the screen's room,
    not under 1.5 times 700, so a window that fills the screen keeps the usual layout; one lower than that takes
    the small layout, and the rail and Home are told. Two columns are held to the screen's width, so Home sets its
    two side by side in a window that fills that screen. On a 1366 by 768 screen a window that fills the screen is
    lower than 700 and takes the small layout, and Home sets its two columns one above the other: beside the rail,
    side by side, they would be too narrow for their words. At the ordinary text size the threshold stays 700."""
    from PySide6.QtWidgets import QApplication
    from core.gui import main_window
    from tests._look_helpers import text_size
    qt_app()
    monkeypatch.setattr(main_window, "_room_on_screen", lambda: (1920 - 40, 1032 - 60))
    with text_size(1.5):
        window = build_window()
        try:
            assert (design.small_below(), design.two_columns_from()) == (972, 1410)
            _shown_off_screen(window, 1920, 1001)
            assert not window.shell._small and not window.part("home").page._small
            assert not window.part("home")._columns.stacked(), "a page as wide as the screen allows set one column"
            window.resize(1920, 971)
            QApplication.processEvents()
            assert window.shell._small and window.part("home").page._small
            assert window.shell.rail.row("home").height() == design.scaled(design.SIZE["nav_h_small"])
        finally:
            window.hide()
    monkeypatch.setattr(main_window, "_room_on_screen", lambda: (1366 - 40, 720 - 60))
    with text_size(1.5):
        window = build_window()
        try:
            assert (design.small_below(), design.two_columns_from()) == (700, 1326)
            _shown_off_screen(window, 1366, 689)
            assert window.shell._small and window.part("home")._columns.stacked()
        finally:
            window.hide()
    monkeypatch.undo()
    with text_size(1.0):
        window = build_window()
        try:
            assert design.small_below() == 700
            _shown_off_screen(window, 760, 700)
            assert not window.shell._small
            window.resize(760, 699)
            QApplication.processEvents()
            assert window.shell._small and window.part("home").page._small
        finally:
            window.hide()


def test_the_current_row_stays_in_view_when_the_rail_has_less_room():
    """At 150 % text the rail's list scrolls, and the current row stays in view when the list has less room: after
    the foot's title takes a second line, and after the window is made lower and takes the small layout."""
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.session import SbiSession
    from tests._look_helpers import text_size
    qt_app()
    with text_size(1.5):
        window = build_window()
        rail, panel = window.shell.rail, window.panel(CrossValPanel)
        try:
            window.part("inference").session = SbiSession(draft=object(), cfg=object(), inf_prior=object(),
                                                         posterior=object())
            window.part("inference").refresh_gates()          # every step can be opened: the last is the lowest row
            window.setMinimumSize(1, 1)
            _shown_off_screen(window, 760, 740)
            window.go_to("inference", "narrowing")
            assert window.current_place() == ("inference", "narrowing")
            assert not window.shell._small and _in_view(rail, rail.row("narrowing"))
            usual = rail.foot.height()
            with running(panel, StandIn(word="Narrowing round")):     # a title of two lines: the foot grows
                QApplication.processEvents()
                assert rail.foot.height() > usual
                assert _in_view(rail, rail.row("narrowing")), "the foot grew over the current row"
            window.go_to("help")
            window.resize(760, 600)                            # a lower window, in the small layout
            QApplication.processEvents()
            assert window.shell._small and _in_view(rail, rail.row("help")), "a lower window hid the current row"
        finally:
            window.hide()


def test_the_current_row_stays_in_view_when_a_step_above_it_takes_a_second_line(tmp_path):
    """In a low window at 150 % text the open step, the last of the chain, stays in view when a step above it takes
    its second line again (the inference step becomes the step in hand once the session holds no observation): the
    list grows while the room it has stays as it was."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.session import SbiSession
    from tests._fixtures import build_browse_store
    from tests._look_helpers import text_size
    qt_app()
    build_browse_store(tmp_path / "Artifacts")
    made = SimpleNamespace(created="2026-10-03T14:02:00+00:00")

    def held(kind, ident):
        return SimpleNamespace(kind=kind, id=ident, name=kind, manifest=made,
                               posterior=SimpleNamespace(truncation=None))

    with use_store(ArtifactStore(tmp_path / "Artifacts")), text_size(1.5):
        window = build_window()
        rail, inference = window.shell.rail, window.part("inference")
        try:
            inference.session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"), cfg=object(),
                                           inf_prior=held("prior", "p1"), posterior=held("posterior", "q1"),
                                           observation=held("observation", "o1"))
            inference.refresh_gates()
            window.setMinimumSize(1, 1)
            _shown_off_screen(window, 760, 600)
            window.go_to("inference", "narrowing")
            one_line = rail.row("infer").height()
            assert window.shell._small and window.current_place() == ("inference", "narrowing") and _in_view(
                rail, rail.row("narrowing"))
            inference.session.observation = None
            inference.refresh_gates()
            QApplication.processEvents()                        # the rows' own layouts settle
            assert rail.row("infer").height() > one_line, "the inference step kept one line"
            assert window.current_place() == ("inference", "narrowing")
            assert _in_view(rail, rail.row("narrowing")), "a step above the open one grew over it"
        finally:
            window.hide()


def test_the_mounted_rail_marks_one_row_for_every_place():
    """In the window, whatever place is open, the accent's bar stands in one row of the rail: the row of the
    step open while Parameter inference is open, or else of the part open, down that whole row but for the open
    part's hairlines, which lie over the row's top or bottom edge where the row stands at the top or the bottom of
    the open part, and nowhere else."""
    from tests._look_helpers import _image
    qt_app()
    window = build_window()
    rail = window.shell.rail
    accent = design.colour("accent").rgb()
    places = [(place.key, None) for place in keys.PARTS] + [("inference", step.key) for step in keys.STEPS]
    try:
        _shown_off_screen(window, 760, 740)
        for part, step in places:
            window.go_to(part, step)
            image = _image(rail.width(), rail.height(), 1.0)
            rail.render(image)
            marked = [y for y in range(image.height()) if image.pixel(0, y) == accent]
            shown, page = window.current_place()
            row = rail.row(page if shown == "inference" else shown)
            top = row.mapTo(rail, QPoint(0, 0)).y()
            block = rail.row(shown).parentWidget()
            above = row.mapTo(block, QPoint(0, 0)).y()
            under = block.height() - above - row.height()
            hair = design.SIZE["hair"]              # the open part's lines lie over the row's top and bottom edge
            first = top + (hair if above == 0 else 0)
            end = top + row.height() - (hair if under == 0 and shown != "help" else 0)
            assert marked == list(range(first, end)), (part, step)
    finally:
        window.hide()


def test_home_is_told_the_small_layout_through_the_shell():
    """The shell tells Home the window's layout: in the small one its head keeps no sentence, in the usual one
    the sentence is back."""
    qt_app()
    window = build_window()
    home = window.part("home")
    window.shell.set_small(True)
    assert home._head.sentence.isHidden()
    window.shell.set_small(False)
    assert not home._head.sentence.isHidden()


def test_a_part_added_to_a_low_window_is_told_so_at_once():
    """A part the shell is handed while the window is low is told the small layout as it is added; in the usual
    layout it is told nothing. A name held already is refused."""
    from PySide6.QtWidgets import QWidget
    from core.gui.screens.shell import Shell
    qt_app()
    shell = build_part(Shell)
    told = {"usual": [], "low": []}
    usual = QWidget()
    usual.set_small = told["usual"].append
    shell.add_part("usual", usual)
    shell.set_small(True)
    low = QWidget()
    low.set_small = told["low"].append
    shell.add_part("low", low)
    assert told == {"usual": [True], "low": [True]}
    with pytest.raises(ValueError):
        shell.add_part("low", QWidget())


def test_a_part_held_in_the_place_of_another_stands_where_it_stood():
    """A part the shell holds in place of another stands where that one stood in the stack, in sight when that one
    was, and is told the small layout at once while the window is low; the part it replaces is handed back, out of
    the stack. A name the shell does not hold is refused."""
    from PySide6.QtWidgets import QWidget
    from core.gui.screens.shell import Shell
    qt_app()
    shell = build_part(Shell)
    first, second, third, told = QWidget(), QWidget(), QWidget(), []
    for name, widget in (("first", first), ("second", second), ("third", third)):
        shell.add_part(name, widget)
    shell.show_part("second")
    shell.set_small(True)
    again = QWidget()
    again.set_small = told.append
    assert shell.replace_part("second", again) is second
    assert [shell.stack.widget(n) for n in range(shell.stack.count())] == [first, again, third]
    assert (shell.part_widget("second"), shell.stack.currentWidget(), told) == (again, again, [True])
    assert shell.replace_part("third", QWidget()) is third and shell.stack.currentWidget() is again
    with pytest.raises(KeyError):
        shell.replace_part("nowhere", QWidget())


def test_help_shows_the_help_text_that_settings_held():
    """Help shows its heading, "Help", over the help text Settings used to hold, built here now: the titles of its
    five sections, and every word of the help text, in a viewer at least as tall as the look asks. The one other
    heading of the page is that of the keys, under the text."""
    from PySide6.QtGui import QTextDocument
    from PySide6.QtWidgets import QLabel
    from core.gui.screens import help_screen, settings_screen
    qt_app()
    window = build_window()
    viewer = window.part("help").viewer
    shown = viewer.toPlainText()
    titles = ["Reduction Map", "FDT Analysis", "Cross-validation (sweep study)", "Parameter Inference", "Simulate"]
    assert [title for _module, title in help_screen._HELP_SECTIONS] == titles, "Help lost or gained a section"
    assert all(title in shown for title in titles), shown[:200]
    written = QTextDocument()
    written.setHtml(help_screen.help_text())
    assert shown == written.toPlainText()
    assert viewer.minimumHeight() == design.HELP_VIEWER_MIN_H and viewer.isReadOnly()
    assert [label.text() for label in window.part("help").findChildren(QLabel)] == ["Help", "Keys"]
    assert not hasattr(settings_screen, "help_text"), "the help text is built in two places"


def test_settings_keeps_the_appearance_choices_alone():
    """Settings holds its heading over the appearance choices, Follow Windows, Light and Dark, and nothing else: no
    help text, no list of models and no button that opens the model builder."""
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel, QRadioButton, QTextBrowser
    qt_app()
    window = build_window()
    settings = window.part("settings")
    assert [box.title() for box in settings.findChildren(QGroupBox)] == ["Appearance"]
    buttons = settings.findChildren(QAbstractButton)
    assert all(isinstance(button, QRadioButton) for button in buttons)
    assert [button.text() for button in buttons] == ["Follow Windows", "Light", "Dark"]
    assert settings.findChildren(QTextBrowser) == []
    assert [label.text() for label in settings.findChildren(QLabel)] == ["Settings"]


def test_your_own_models_lists_the_models_and_opens_the_builder_and_back(monkeypatch):
    """Your own models opens on the list of the models you defined, each with Edit and Delete. "Open model builder"
    opens the builder on a blank form; "Back to your models" returns to the list and keeps what the form holds;
    Edit opens the builder on the model. The keyboard follows each move to the page's first stop, given as a move
    gives it, not as a key. A model saved in the builder, or deleted, is named in the list again and every list of
    models in the panels is judged again; Delete asks first, and a question answered as the guard answers deletes
    nothing. Opened again in the launch the part shows the page last shown, a form left half filled as it was left;
    after the window is closed, the next launch opens on the list."""
    from PySide6.QtWidgets import QGroupBox, QLabel, QWidget
    from core import registry
    from core.Helpers import model_store
    from tests._fixtures import all_asked, choose, forget_asked
    qt_app()
    names = ["SHM", "SHM2"]                                       # the two shipped with PRISM
    monkeypatch.setattr(registry, "user_model_names", lambda: list(names))
    monkeypatch.setattr(registry, "load_errors", [])
    window = build_window()
    models = window.part("models")
    builder = models.builder
    window.go_to("models")
    assert window.current_place() == ("models", "list")
    listed, = [box for box in models.findChildren(QGroupBox) if box.title() == "User-defined models"]

    def rows():
        return [([label.text() for label in row.findChildren(QLabel)],
                 [button.text() for button in row.findChildren(QPushButton)])
                for row in listed.findChildren(QWidget, "", Qt.FindDirectChildrenOnly) if row.findChildren(QPushButton)]

    assert rows() == [([name], ["Edit", "Delete"]) for name in names]
    opener = _button(models, "Open model builder")
    given, kept = [], QWidget.setFocus

    def set_focus(widget, *reason):
        given.append((widget, reason))
        return kept(widget, *reason)

    monkeypatch.setattr(QWidget, "setFocus", set_focus)
    builder.name_edit.setText("DRAFT")
    _button(models, "Open model builder").click()
    assert (window.current_place(), builder.name_edit.text()) == (("models", "builder"), "")
    assert window.focusWidget() is builder.name_edit
    builder.name_edit.setText("DRAFT")
    _button(builder, "Back to your models").click()
    assert (window.current_place(), builder.name_edit.text()) == (("models", "list"), "DRAFT")
    assert window.focusWidget() is opener
    assert given == [(builder.name_edit, (Qt.OtherFocusReason,)), (opener, (Qt.OtherFocusReason,))], (
        "a button of the part gave the keyboard as a key gives it")
    window.go_to("live")
    window.go_to("models")
    assert window.current_place() == ("models", "list"), "a later opening is not at the page last shown"
    _button(listed, "Edit").click()                               # the first row's: the first model's
    assert (window.current_place(), builder.name_edit.text()) == (("models", "builder"), names[0])
    window.go_to("home")
    window.go_to("models")
    assert (window.current_place(), builder.name_edit.text()) == (("models", "builder"), names[0]), (
        "a later opening is not at the page last shown, as it was left")
    judged, deleted = [], []
    monkeypatch.setattr(window, "_refresh_model_combos", lambda: judged.append(len(rows())))
    names.append("DUFFING")
    builder._on_saved("DUFFING")                                  # what the builder's Save calls once it has written
    assert judged == [3] and rows()[-1] == (["DUFFING"], ["Edit", "Delete"]), "a saved model is not named"
    monkeypatch.setattr(model_store, "delete_user_model", deleted.append)
    forget_asked()
    models._delete_user_model("DUFFING")
    assert [box.title for box in all_asked()] == ["Delete model"] and (deleted, judged) == ([], [3])
    choose(monkeypatch, other=True)
    monkeypatch.setattr(registry, "unregister", lambda name: names.remove(name))
    models._delete_user_model("DUFFING")
    assert (deleted, judged) == (["DUFFING"], [3, 2]), "a deleted model is still named"
    window.close()                                                # closing writes what the window keeps
    again = build_window()
    again.go_to("models")
    assert again.current_place() == ("models", "list"), "the page shown was kept for the next launch"


def test_the_builder_sends_a_taken_name_back_to_your_own_models(monkeypatch):
    """A new model saved under the name of one that exists is refused before anything is written, and the sentence
    sends the person to Your own models, where the list of models stands."""
    from core.Helpers import model_store
    from core.gui.screens.model_builder_screen import ModelBuilderScreen
    qt_app()
    saved = []
    monkeypatch.setattr(model_store, "save_user_model", saved.append)
    builder = build_part(ModelBuilderScreen)
    builder.vars_edit.setText("x")
    builder._set_variables()
    builder._var_rows[0].drift.setText("-k*x")
    builder._var_rows[0].noise.setText("d0")
    builder.name_edit.setText("SHM")                              # a model shipped with PRISM
    builder._detect_params()
    builder._save()
    assert saved == [], "a taken name was written over"
    said = builder.status.text()
    assert said.endswith("A model named 'SHM' already exists -- edit it from Your own models or pick another name."), said


def test_a_delete_leaves_the_keyboard_on_the_lists_first_stop(monkeypatch):
    """A model deleted from the list takes its row away, and with it the Delete that was pressed and held the
    keyboard: the keyboard goes to the list's first stop, the button that opens the builder, given as a move of
    the part gives it."""
    from PySide6.QtWidgets import QWidget
    from core import registry
    from core.Helpers import model_store
    from tests._fixtures import choose
    qt_app()
    names = ["SHM", "SHM2"]                                       # the two shipped with PRISM
    monkeypatch.setattr(registry, "user_model_names", lambda: list(names))
    monkeypatch.setattr(registry, "load_errors", [])
    monkeypatch.setattr(model_store, "delete_user_model", lambda name: None)
    monkeypatch.setattr(registry, "unregister", names.remove)
    window = build_window()
    window.go_to("models")
    opener = _button(window.part("models"), "Open model builder")
    pressed = _button(window.part("models"), "Delete")             # the first row's: SHM's
    given, kept = [], QWidget.setFocus

    def set_focus(widget, *reason):
        given.append((widget, reason))
        return kept(widget, *reason)

    monkeypatch.setattr(QWidget, "setFocus", set_focus)
    pressed.setFocus(Qt.OtherFocusReason)                         # a click on it gives it the keyboard
    choose(monkeypatch, other=True)
    pressed.click()
    assert (names, given[-1], window.focusWidget()) == (["SHM2"], (opener, (Qt.OtherFocusReason,)), opener), (
        "the keyboard did not go to the list's first stop after a delete")


def test_a_window_restored_off_every_screen_opens_at_its_opening_size_on_the_primary_screen(monkeypatch):
    """A saved geometry that restores where no screen shows it (saved on a screen no longer attached) is set aside:
    the window opens at its opening size, in the middle of the primary screen."""
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QGuiApplication
    from core.gui import main_window, settings as st
    qt_app()
    placed = build_window()
    placed.move(40, 30)
    remembered = st.settings()
    remembered.setValue("window/geometry", placed.saveGeometry())
    remembered.sync()
    monkeypatch.setattr(main_window, "_on_a_visible_screen", lambda widget: False)
    window = build_window()
    opening, _smallest = design.window_sizes(main_window._room_on_screen())
    assert (window.width(), window.height()) == opening
    centre = QGuiApplication.primaryScreen().availableGeometry().center()
    assert (window.frameGeometry().center() - centre).manhattanLength() <= 1
    assert isinstance(remembered.value("window/geometry"), QByteArray)


def test_effective_temperature_opens_on_one_cell_and_its_names_choose_the_page():
    """Effective temperature opens on One cell, its name marked in the row over the page. A click on Sweep study
    shows that page, Left on the row shows One cell again and Right Sweep study, each said as a move. Opened again
    in the launch, the part shows the page last shown; the page shown is not kept, so the next launch opens on One
    cell again."""
    qt_app()
    window = build_window()
    screen = window.part("temperature")
    said = []
    window.place_changed.connect(lambda part, page: said.append((part, page)))
    window.go_to("temperature")
    assert (window.current_place(), screen.names.current()) == (("temperature", "one_cell"), "one_cell")
    assert screen.panels()[0].isVisibleTo(screen) and not screen.panels()[1].isVisibleTo(screen)
    _name(screen, "sweep").click()
    assert (window.current_place(), screen.names.current(), said[-1]) == (
        ("temperature", "sweep"), "sweep", ("temperature", "sweep"))
    assert screen.panels()[1].isVisibleTo(screen) and not screen.panels()[0].isVisibleTo(screen)
    QTest.keyClick(_name(screen, "sweep"), Qt.Key_Left)
    assert (window.current_place(), said[-1]) == (("temperature", "one_cell"), ("temperature", "one_cell"))
    QTest.keyClick(_name(screen, "one_cell"), Qt.Key_Right)
    assert (window.current_place(), said[-1]) == (("temperature", "sweep"), ("temperature", "sweep"))
    window.go_to("live")
    window.go_to("temperature")
    assert window.current_place() == ("temperature", "sweep"), "a later opening is not at the page last shown"
    window.close()                                                # closing writes what the window keeps
    again = build_window()
    again.go_to("temperature")
    assert again.current_place() == ("temperature", "one_cell"), "the page shown was kept for the next launch"


def _inside(widget, part):
    """What ``widget``'s own layout holds, its items' rectangles taken together, in the part's coordinates."""
    from PySide6.QtCore import QRect
    layout, rect = widget.layout(), QRect()
    for number in range(layout.count()):
        item = layout.itemAt(number)
        if not item.isEmpty():
            rect = rect.united(item.geometry())
    return QRect(widget.mapTo(part, rect.topLeft()), rect.size())


def _margins(window, part):
    """How far in the place ``part`` stands its first frame or first line from the rail, the window's right edge,
    its top and its bottom: a step by the frame of the step shown, Effective temperature by its row of names and
    the frame of One cell, every other place by what its own layout holds."""
    from PySide6.QtCore import QRect
    page = window.part(part)
    if part == "inference":
        found = [_inside(page.pages.currentWidget(), page)]
    elif part == "temperature":
        found = [QRect(page.names.mapTo(page, QPoint(0, 0)), page.names.size()), _inside(page.panels()[0], page)]
    else:
        found = [_inside(page, page)]
    return (min(rect.left() for rect in found), min(rect.top() for rect in found),
            page.width() - max(rect.right() + 1 for rect in found),
            page.height() - max(rect.bottom() + 1 for rect in found))


def test_every_place_but_home_stands_the_same_margin_from_the_rail_and_the_windows_edges():
    """Every place but Home stands its first frame or its first line twelve pixels in from the rail, from the
    window's right edge, from its top and from its bottom, at the ordinary text size, and as much more as a larger
    text size scales it, opened first in the usual layout and in the small one: a step and a panel by the frame
    their own layout holds, Effective temperature's row of page names too, lined up with the frame under it, and
    every other place by what its own layout holds, the list of Your own models by its group too. Home keeps the
    look's page margin, and the window gives it none."""
    from PySide6.QtCore import QMargins
    from PySide6.QtWidgets import QGroupBox
    from tests._look_helpers import text_size
    qt_app()
    for factor in (1.0, 1.5):
        for low in (False, True):
            with text_size(factor):
                window = build_window()
                try:
                    window.setMinimumSize(1, 1)
                    _shown_off_screen(window, 1300, design.small_below() - 1 if low else 820)
                    assert window.shell._small == low, (factor, low)
                    for place in keys.PARTS[1:]:
                        window.go_to(place.key)
                        QApplication.processEvents()
                        assert _margins(window, place.key) == (design.scaled(12),) * 4, (factor, low, place.key)
                    listed = window.part("models")
                    group, = [box for box in listed.findChildren(QGroupBox) if box.title() == "User-defined models"]
                    at = group.mapTo(listed, QPoint(0, 0))
                    assert (at.x(), at.y(), listed.width() - at.x() - group.width()) == (design.scaled(12),) * 3, (
                        factor, low)
                    assert window.part("home").contentsMargins() == QMargins(0, 0, 0, 0)
                finally:
                    window.hide()


def test_a_panels_column_lays_each_item_at_its_height_for_the_columns_width():
    """In the column of a panel's controls -- each of the six steps, Effective temperature's two pages, Live
    simulation and the Reduction map -- no line that wraps is drawn shorter than its words take at its width, on
    an empty session and with a configuration made from the shipped Nadrowski bounds; and in a column where
    nothing grows, each item stands at its own height for the column's width, what the column has to spare left
    under its last item."""
    from PySide6.QtWidgets import QLabel
    from core import config
    qt_app()
    window = build_window()
    inference = window.part("inference")
    panels = [("inference", step.key, panel) for step, panel in zip(keys.STEPS, inference.panels())]
    panels += [("temperature", page.key, panel)
               for page, panel in zip(keys.PAGES["temperature"], window.part("temperature").panels())]
    panels += [("live", None, window.part("live")), ("reduction", None, window.part("reduction"))]

    def named(widget):
        """A group's title, or the kind of a widget that has none."""
        title = getattr(widget, "title", None)
        return title() if callable(title) else type(widget).__name__

    def wrong(panel):
        """The wrapped lines of the column drawn shorter than their words take, and the items of a column where
        nothing grows that stand at another height than their own for the column's width."""
        column = panel.controls_layout
        short = [label.text()[:30] for label in panel.controls.findChildren(QLabel)
                 if label.isVisibleTo(panel) and label.wordWrap() and label.text()
                 and label.height() < label.heightForWidth(label.width())]
        loose = []
        if not column.expandingDirections() & Qt.Vertical:
            for item in (column.itemAt(number) for number in range(column.count())):
                width, height = item.geometry().width(), item.geometry().height()
                own = item.heightForWidth(width) if item.hasHeightForWidth() else item.sizeHint().height()
                if not item.isEmpty() and height != own:
                    loose.append(named(item.widget()))
        return short, loose

    try:
        _shown_off_screen(window, 1300, 820)
        found = {}
        for session in ("empty", "configuration"):
            if session == "configuration":
                inference.config_panel.model_combo.setCurrentText("NADROWSKI")
                inference.config_panel.btn_config.click()          # the first step's own button: a model
                bounds = config.BOUNDS_PATH / "nadrowski" / "master.txt"
                inference.install_config(inference.session.draft.make_config(bounds_path=str(bounds)))
            for part, page, panel in panels:
                window.go_to(part, page)
                QApplication.processEvents()
                short, loose = wrong(panel)
                if short or loose:
                    found[(session, part, page)] = (short, loose)
        assert found == {}, found
    finally:
        window.hide()


class _CannotBeBuilt(Exception):
    """An error of a kind no part of the window raises: its name is looked for in what the window shows."""


def _card_and_programs(monkeypatch) -> list:
    """Every question put to the graphics card and every program started, from now to the test's end, each named in
    the list handed back: a program run (by the program's name), whether there is a card, the card's free memory,
    the memory the card holds for the program and the memory it uses, and the device check. Each of these six
    stand-ins records its call and answers as a card with room to spare would, so that the code behind it goes on
    as it goes on a computer with one, whatever an earlier test woke on the card. Beneath them every other public
    function of the card's package and of its memory module, and the other ways to start a program, are recorded by
    their names, each under the path it was called by, and passed through to the real call: a question put to the
    card some other way is named too."""
    import os
    import subprocess
    import torch
    from torch.cuda import memory as cuda_memory
    from core import config
    calls = []
    card = config.DeviceConfig(device=torch.device("cuda"), dtype=torch.float32, batch_size=2 ** 11)

    def passed_through(name, real):
        def stand_in(*args, **kwargs):
            calls.append(name)
            return real(*args, **kwargs)
        return stand_in

    # A function of the memory module is reachable by two paths, the package's name for it and the module's own, and
    # a call by the second passes through no stand-in set on the first: both are recorded, each by its own path.
    for space, prefix in ((torch.cuda, "torch.cuda"), (cuda_memory, "torch.cuda.memory")):
        for name in dir(space):
            value = getattr(space, name)
            if (not name.startswith("_") and callable(value) and not isinstance(value, type)
                    and str(getattr(value, "__module__", "")).startswith("torch.cuda")):
                monkeypatch.setattr(space, name, passed_through(f"{prefix}.{name}", value))
    monkeypatch.setattr(subprocess, "Popen", passed_through("subprocess.Popen", subprocess.Popen))
    monkeypatch.setattr(os, "system", passed_through("os.system", os.system))

    def recorded(words, answer):
        def stand_in(*args, **kwargs):
            calls.append(words(*args, **kwargs) if callable(words) else words)
            return answer
        return stand_in

    def program(command=(), *_args, **_kwargs):
        """The program a run was asked for, by its name: the first word of its command."""
        words = list(command) if isinstance(command, (list, tuple)) else str(command).split()
        return "the program " + (str(words[0]) if words else "")

    monkeypatch.setattr(subprocess, "run", recorded(program, subprocess.CompletedProcess((), 1, "", "")))
    monkeypatch.setattr(torch.cuda, "is_available", recorded("whether there is a card", True))
    monkeypatch.setattr(torch.cuda, "mem_get_info", recorded("the card's free memory", (8 * 2 ** 30, 16 * 2 ** 30)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", recorded("the card's memory held", 0))
    monkeypatch.setattr(torch.cuda, "memory_allocated", recorded("the card's memory in use", 0))
    monkeypatch.setattr(config, "detect_device", recorded("the device check", card))
    return calls


# What the first opening of a part asks of the card or starts, by name and how often: Parameter inference's six
# steps are built together, four of them read the card as they are built, and their gates read it again. The list
# may only shrink: a page that reads the card reads it at a moment of its own, never as it is built.
_READ_AT_FIRST_OPENING = {"inference": {"the program nvidia-smi": 1, "the device check": 3, "the card's free memory": 4,
                                        "the card's memory held": 4, "the card's memory in use": 4}}


def test_opening_prism_touches_no_card_and_starts_no_program(tmp_path, monkeypatch):
    """From the start of the whole program -- as a start of PRISM builds it, its window among it -- to its first idle
    turn with Home shown, nothing asks the graphics card anything and no other program is started: a program run,
    whether there is a card, its free memory and the device check are each recorded (not refused by raising: the
    callers swallow what is raised, so a stand-in that raised would prove nothing), and none is recorded. Nothing
    ticks then either: no timer of the window or of the application runs, and of every start of a timer from Python,
    a timer with no parent among them, none still runs after the first idle turn, nor is a single shot of a later
    moment waiting. The same holds after the entry point's own lines, which claim, show the window, run the event loop
    and let go of the claim once the program is built: run here with a loop that turns once and ends."""
    from PySide6.QtCore import QObject, QTimer
    from core.gui import __main__ as entry
    from core.gui import app as gui_app
    from tests._fixtures import whole_program
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    calls = _card_and_programs(monkeypatch)
    started, waiting = [], []
    start, single_shot, start_timer = QTimer.start, QTimer.singleShot, QObject.startTimer

    def started_timer(timer, *args):
        started.append(timer)
        return start(timer, *args)

    def waiting_shot(msec, *args, **kwargs):
        if msec > 0:
            waiting.append(msec)
        return single_shot(msec, *args, **kwargs)

    def started_by(timed, *args, **kwargs):
        started.append(timed)
        return start_timer(timed, *args, **kwargs)

    monkeypatch.setattr(QTimer, "start", started_timer)
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(waiting_shot))
    monkeypatch.setattr(QObject, "startTimer", started_by)

    def nothing_asked_and_nothing_ticking(window):
        """Home is in sight, the card was asked nothing, no program was started, and no timer runs or waits."""
        assert window.current_place() == ("home", None)
        assert calls == [], f"opening PRISM asked the card or started a program: {calls}"
        ticking = [timer for timer in app.findChildren(QTimer) + window.findChildren(QTimer) if timer.isActive()]
        ticking += [timer for timer in started if not isinstance(timer, QTimer) or timer.isActive()]
        assert (ticking, waiting) == ([], []), f"a timer runs at launch: {ticking}, {waiting}"

    class Loop:
        """The program's event loop, as far as the start goes: it turns once and ends."""
        def exec(self):
            QApplication.processEvents()
            return 0

    with whole_program() as (_app, window):
        try:
            _shown_off_screen(window)                             # Home shown, then one idle turn
            nothing_asked_and_nothing_ticking(window)
            monkeypatch.setattr(gui_app, "build_app", lambda **handed: (Loop(), window))
            assert _main(entry) == 0                              # the entry point's own lines, the build stood in
            nothing_asked_and_nothing_ticking(window)
        finally:
            window.hide()


def test_only_the_known_pages_read_the_card_when_first_opened(monkeypatch):
    """Each part opened for the first time, in the rail's order, asks the graphics card nothing and starts no
    program, but Parameter inference, whose six steps are built together at its first opening: what it asks is
    what is listed, so many times each, and nothing else. The window is shown, off the screen, and given an idle
    turn after each opening, so that each part's first show runs as it runs in front of a person: a first opening
    is a build and a show."""
    from collections import Counter
    qt_app()
    calls = _card_and_programs(monkeypatch)
    window = build_window()
    found = {}
    try:
        _shown_off_screen(window)
        for place in keys.PARTS:
            calls.clear()
            window.go_to(place.key)
            QApplication.processEvents()
            found[place.key] = dict(Counter(calls))
    finally:
        window.hide()
    listed = {place.key: _READ_AT_FIRST_OPENING.get(place.key, {}) for place in keys.PARTS}
    otherwise = {part: asked for part, asked in found.items() if asked != listed[part]}
    assert otherwise == {}, f"a first opening asked other than is listed: {otherwise}"


def test_every_panel_of_the_built_parts_has_its_place():
    """Every kind of panel any part of the window holds is found by the window's ``panel``, each kind in a window of
    its own that has built nothing but Home: asking for the kind builds the part that holds it, and no part that does
    not, and hands back a panel of that kind standing in that part. The kinds are found by walking each part's
    widgets, not by the window's own lists, so a kind of panel held by a part those lists do not name is seen too. A
    part that cannot be built stands as an error page and hides its panels, so every part is required to be built."""
    from core.gui.panels.base_panel import BasePanel
    qt_app()
    every = build_window()
    held = {}                                          # each kind of panel, with the keys of the parts holding one
    for place in keys.PARTS:
        part = every.part(place.key)
        for panel in ([part] if isinstance(part, BasePanel) else []) + part.findChildren(BasePanel):
            held.setdefault(type(panel), set()).add(place.key)
    assert every.built_parts() == tuple(place.key for place in keys.PARTS), "a part could not be built"
    assert len(held) >= 10, f"the walk found only {sorted(kind.__name__ for kind in held)}"
    wrong = []
    for kind, parts in held.items():
        # a window of its own: in one window shared by every kind, a part built for an earlier kind would hand over
        # every later kind it holds without the window's list of the part each kind stands in being read
        fresh = build_window()
        before = set(fresh.built_parts())
        found = fresh.panel(kind)
        built = set(fresh.built_parts()) - before
        standing = found is not None and any(
            key in fresh.built_parts() and (fresh.part(key) is found or fresh.part(key).isAncestorOf(found))
            for key in parts)
        if not (isinstance(found, kind) and standing and built <= parts):
            wrong.append(f"{kind.__name__}: held in {sorted(parts)}, asking built {sorted(built)} and found {found}")
    assert wrong == [], f"a kind of panel the window does not find in the part that holds it: {wrong}"


def test_a_part_is_built_at_its_first_opening_and_once():
    """The window is built with Home alone, and the stack holds Home alone. Every other part is built the first
    time it is asked for -- opened by ``go_to``, handed by ``part``, or holding the panel ``panel`` is asked for --
    and once: the part handed later is the one built. A part the window does not have, or a page its part does not
    have, is refused before anything is built."""
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.screens.temperature_screen import TemperatureScreen
    qt_app()
    window = build_window()
    stack = window.shell.stack
    assert window.built_parts() == ("home",)
    assert [stack.widget(n) for n in range(stack.count())] == [window.part("home")]
    for part, page in (("nowhere", None), ("inference", "nowhere"), ("temperature", "nowhere"), ("models", "nowhere")):
        with pytest.raises(KeyError):
            window.go_to(part, page)
    assert (window.built_parts(), window.current_place(), stack.count()) == (("home",), ("home", None), 1), (
        "a move refused built a part or moved the window")
    window.go_to("live")
    live = window.part("live")
    window.go_to("home")
    window.go_to("live")
    assert window.part("live") is live and stack.currentWidget() is live
    assert window.built_parts() == ("home", "live")
    help_page = window.part("help")                               # handed, built, not shown
    assert window.built_parts() == ("home", "live", "help") and stack.currentWidget() is live
    fdt = window.panel(FdtPanel)
    assert window.built_parts() == ("home", "temperature", "live", "help")
    temperature = window.part("temperature")
    assert isinstance(temperature, TemperatureScreen) and fdt is temperature.fdt_panel
    assert window.panel(FdtPanel) is fdt and window.panel(CrossValPanel) is temperature.crossval_panel
    assert window.part("help") is help_page and stack.count() == 4, "a part was built twice"


def _shown_words(page):
    """Every text in sight on ``page``: each label's and sentence's words, each button's, and a text box's."""
    from PySide6.QtWidgets import QAbstractButton, QLabel, QPlainTextEdit, QWidget
    from core.gui.widgets.page import Sentence
    words = []
    for widget in page.findChildren(QWidget):
        if not widget.isVisibleTo(page):
            continue
        if isinstance(widget, Sentence):
            words.append(widget.plain_text())
        elif isinstance(widget, (QLabel, QAbstractButton)):
            words.append(widget.text())
        elif isinstance(widget, QPlainTextEdit):
            words.append(widget.toPlainText())
    return words


def test_a_part_that_fails_to_build_shows_its_error_page_and_the_rest_works(monkeypatch, caplog):
    """A part whose construction raises -- here once its two panels were built -- shows its error page in its place,
    in the look of a page, and its row of the rail is marked: the Error marker beside "Effective temperature could
    not be opened." and the error's message, "The rest of PRISM still works.", "Try again" and "Show the full
    error". The error's kind is in no text the page shows until "Show the full error" opens it, with its trace, in
    a read-only box in the fixed-width face, which takes the keyboard; the sentence, the kind and the trace went to
    the log. Opened again, the part shows its error page again, and every other part opens. "Try again", the
    construction mended, builds the part in that place and opens it as a move does, at the place's own margin, and
    the error page goes."""
    import logging
    from core.gui import fonts, main_window
    from core.gui.screens.shell import PartError
    from core.gui.screens.temperature_screen import TemperatureScreen
    qt_app()

    def half_built():
        build_part(TemperatureScreen)                             # its two panels are built, then it fails
        raise _CannotBeBuilt("the cell file could not be read")

    monkeypatch.setattr(main_window, "TemperatureScreen", half_built)
    window = build_window()
    rail, said = window.shell.rail, []
    window.place_changed.connect(lambda part, page: said.append((part, page)))
    try:
        _shown_off_screen(window, 1300, 820)
        with caplog.at_level(logging.ERROR, logger="core.gui.main_window"):
            window.go_to("temperature")
        page = window.part("temperature")
        assert isinstance(page, PartError) and window.shell.stack.currentWidget() is page
        assert window.focusWidget() is page.again, "the error page did not give Try again the keyboard"
        assert (window.current_place(), _marked(rail), window.built_parts()) == (
            ("temperature", None), ["temperature"], ("home",))
        assert page.message.kind == "err" and page.message.marker is not None, "no Error marker"
        words = _shown_words(page)
        assert "Effective temperature could not be opened. The cell file could not be read." in words, words
        assert {"The rest of PRISM still works.", "Try again", "Show the full error"} <= set(words), words
        assert not [text for text in words if "_CannotBeBuilt" in text], "the error's kind is shown before it is asked"
        logged, = [record.getMessage() for record in caplog.records if record.name == "core.gui.main_window"]
        assert logged.startswith("Effective temperature could not be opened. the cell file could not be read\n")
        assert "Traceback" in logged and "_CannotBeBuilt: the cell file could not be read" in logged, logged
        _button(page, "Show the full error").click()
        shown = [text for text in _shown_words(page) if "_CannotBeBuilt" in text]
        assert len(shown) == 1 and shown[0].startswith("Traceback") and page.full_error.isReadOnly(), shown
        assert page.full_error.font().family() == fonts.fixed().family()
        assert window.focusWidget() is page.full_error and "Show the full error" not in _shown_words(page)
        window.go_to("live")
        window.go_to("temperature")
        assert window.shell.stack.currentWidget() is page, "the part was built again without being asked"
        window._store_changed()                                   # an error page is no part to tell
        for place in keys.PARTS:
            if place.key != "temperature":
                window.go_to(place.key)
                assert window.current_place()[0] == place.key and place.key in window.built_parts(), place.key
        monkeypatch.setattr(main_window, "TemperatureScreen", TemperatureScreen)
        window.go_to("temperature")
        _button(page, "Try again").click()
        built = window.part("temperature")
        assert isinstance(built, TemperatureScreen) and window.shell.stack.currentWidget() is built
        assert (window.current_place(), _marked(rail), said[-1]) == (
            ("temperature", "one_cell"), ["temperature"], ("temperature", "one_cell"))
        assert window.focusWidget() is built.fdt_panel.model_combo, "the part built again took no keyboard"
        assert window._places[built.fdt_panel] == ("temperature", "one_cell"), "its panel would have no place"
        assert window.built_parts() == tuple(place.key for place in keys.PARTS)
        assert window.shell.stack.indexOf(page) == -1, "the error page stays in the stack"
        QApplication.processEvents()
        assert _margins(window, "temperature") == (design.scaled(12),) * 4, "a part built again has no margin"
    finally:
        window.hide()


def test_a_click_in_the_rail_over_parameter_inferences_error_page_raises_nothing(monkeypatch):
    """With the keyboard in the rail, as the window coming to the front leaves it, a click on Parameter
    inference's row or on a step's row while the part shows its error page raises nothing: the error page stays
    in sight, the part's row is marked, and the keyboard stays in the rail on that row, the row the click marked,
    since the part shows no step. Once the part is built, a click on a step's row leaves the keyboard on that
    step's row, as before."""
    import sys
    from core.gui import main_window
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.screens.shell import PartError
    qt_app()
    raised = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: raised.append(f"{kind.__name__}: {error}"))

    def broken():
        raise _CannotBeBuilt("the cell file could not be read")

    monkeypatch.setattr(main_window, "InferenceScreen", broken)
    window = build_window()
    rail = window.shell.rail
    try:
        _shown_off_screen(window)
        for row in ("inference", "prior", "inference"):
            rail.take_keyboard()                                  # the keyboard in the rail
            QTest.mouseClick(rail.row(row), Qt.LeftButton)
            assert raised == [], f"a click on the row of {row} raised: {raised}"
            assert isinstance(window.shell.stack.currentWidget(), PartError), row
            assert (window.current_place(), _marked(rail), window.focusWidget()) == (
                ("inference", None), ["inference"], rail.row("inference")), (row, window.focusWidget())
        monkeypatch.setattr(main_window, "InferenceScreen", InferenceScreen)
        _button(window.part("inference"), "Try again").click()
        rail.take_keyboard()
        QTest.mouseClick(rail.row("prior"), Qt.LeftButton)
        assert (raised, window.current_place(), window.focusWidget()) == (
            [], ("inference", "prior"), rail.row("prior")), (raised, window.current_place(), window.focusWidget())
    finally:
        window.hide()


def test_a_part_that_fails_to_build_leaves_no_panel_for_a_run_to_lock_and_the_next_run_goes_through(monkeypatch):
    """However a part fails to be built -- a panel's own construction raising partway (Parameter inference's
    narrowing round, at its first field), the part raising once its two panels are whole (Effective temperature),
    or the window's wiring of the part raising while a run is live (the lock of Effective temperature's first
    panel) -- no panel the failed build made is left among the panels a run locks or in the window's map of
    places, and nothing it made is left alive, a panel or a widget with no parent (a half-built part's own top
    widget among them); the part's error page stands in its place. A run's end then raises nothing: the foot says
    again that nothing is running, and no panel is left greyed. "Try again" builds the part whole."""
    from PySide6.QtCore import QEvent
    from core.gui import main_window
    from core.gui.panels.base_panel import BasePanel
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.inference import tsnpe_tab
    from core.gui.panels.inference_tabs import (ConfigPanel, InferPanel, PosteriorPanel, PriorPanel, TSNPEPanel,
                                                ValidatePanel)
    from core.gui.panels.simulate_panel import SimulatePanel
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.screens.shell import PartError
    from core.gui.screens.temperature_screen import TemperatureScreen
    app = qt_app()

    def raising(*_args, **_kwargs):
        raise _CannotBeBuilt("the cell file could not be read")

    def half_built():
        build_part(TemperatureScreen)                             # its two panels are built, then it fails
        raise _CannotBeBuilt("the cell file could not be read")

    def held(window, kinds, before):
        """By the names of their kinds, what was made since ``before`` and is still held once the deletes asked for
        are done: the panels among the panels a run locks, in the window's map of places, and alive, and the
        widgets with no parent alive."""
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        made = [[thing for thing in found if thing not in before]
                for found in (BasePanel._instances, window._places, QApplication.allWidgets())]
        return tuple(sorted(type(thing).__name__ for thing in found if isinstance(thing, kinds)) for found in made) + (
            sorted(type(alone).__name__ for alone in QApplication.topLevelWidgets() if alone not in before),)

    steps = (ConfigPanel, PriorPanel, PosteriorPanel, ValidatePanel, InferPanel, TSNPEPanel)
    for name, cls, kinds, holder, attribute, failing, during_a_run in (
            ("inference", InferenceScreen, steps, tsnpe_tab, "FloatField", raising, False),
            ("temperature", TemperatureScreen, (FdtPanel, CrossValPanel), main_window, "TemperatureScreen",
             half_built, False),
            ("temperature", TemperatureScreen, (FdtPanel, CrossValPanel), FdtPanel, "set_controls_enabled",
             raising, True)):
        window = build_window()
        runner, foot, raised = window.panel(SimulatePanel), window.shell.rail.foot, []
        before = set(BasePanel._instances) | set(QApplication.allWidgets())
        stand_in = StandIn()
        if during_a_run:
            start_run(runner, stand_in)
        try:
            with monkeypatch.context() as failing_now:
                failing_now.setattr(holder, attribute, failing)
                window.go_to(name)
        finally:
            for busy in ([False] if during_a_run else []) + [True, False]:     # the run's end, and a run after it
                try:
                    if busy:
                        stand_in = StandIn()
                        start_run(runner, stand_in)
                    else:
                        stand_in.release()
                        turn_until(app, lambda: not controller().is_live())
                except Exception as error:                        # noqa: BLE001 -- recorded, then asserted
                    raised.append(repr(error))
        page = window.part(name)
        assert isinstance(page, PartError) and window.shell.stack.currentWidget() is page, attribute
        greyed = sorted(type(panel).__name__ for panel in BasePanel._instances if not panel.controls.isEnabled())
        assert (raised, foot.accessibleName().startswith("Training finished"), greyed) == ([], True, []), (
            "a run's end did not go through", attribute, raised)
        assert held(window, kinds, before) == ([], [], [], []), (
            f"what the failed build of {name} made is still held", attribute, held(window, kinds, before))
        _button(page, "Try again").click()
        built = window.part(name)
        assert isinstance(built, cls) and len(built.panels()) == len(kinds), attribute
        assert set(built.panels()) <= set(BasePanel._instances), attribute


def test_a_part_that_fails_to_build_while_a_run_is_live_leaves_the_runs_log_to_the_run(monkeypatch, capsys):
    """While a run is live its log shows what the run itself says: a part first opened then, whose construction
    raises, shows its error page, and its sentence, the error's kind and its trace go to the console the run's
    log stands in front of, not into the run's log, which still shows the run's own record."""
    from core.gui import main_window
    from core.gui.panels.simulate_panel import SimulatePanel
    from core.gui.screens.shell import PartError
    app = qt_app()
    window = build_window()
    runner, stand_in, text = window.panel(SimulatePanel), StandIn(), RunText(controller())

    def shown():
        return "\n".join(text.texts())

    def broken():
        raise _CannotBeBuilt("the cell file could not be read")

    capsys.readouterr()
    start_run(runner, stand_in)
    try:
        stand_in.say("the run's own record")
        turn_until(app, lambda: "the run's own record" in shown(), 10)
        monkeypatch.setattr(main_window, "TemperatureScreen", broken)
        window.go_to("temperature")
        assert isinstance(window.part("temperature"), PartError)
    finally:
        stand_in.release()
    turn_until(app, lambda: not controller().is_live(), 10)
    console = capsys.readouterr().err
    assert "the run's own record" in shown() and not controller().is_live(), (
        f"the run's own record is lost: {shown()}")
    assert [words for words in ("could not be opened", "Traceback", "_CannotBeBuilt") if words in shown()] == [], (
        f"the run's log holds the failure: {shown()}")
    assert "error: Effective temperature could not be opened. the cell file could not be read" in console, (
        f"the console did not get the failure: {console!r}")
    assert "_CannotBeBuilt: the cell file could not be read" in console, f"the console got no trace: {console!r}"


def test_a_record_the_console_cannot_take_during_a_run_is_reported_and_not_raised(capsys):
    """While a run is live a record logged on another thread -- the window's own word -- goes to the console the
    run's log stands in front of. A console that cannot take it, whose encoding has no letter for a word of it, does
    not raise where the record was logged: the failed write is reported on that console's own error stream, as
    logging reports one, with the record's words escaped, so the console that could not take them can take the
    report, and the program goes on. The report never goes to the process's error stream of the moment, which
    during a run is the run's own: it would land in the run's text from the window's thread."""
    import io
    import logging
    import threading
    from core.gui import streams
    narrow = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    made = []
    run = threading.Thread(target=lambda: made.append(streams._PumpLogHandler(None, None, (narrow, narrow))))
    run.start()                                       # the run's handler is made on the run's own thread
    run.join()
    record = logging.makeLogRecord({"name": "core.gui.main_window", "levelno": logging.ERROR, "levelname": "ERROR",
                                    "msg": "the \u03c7 probe table holds no row"})
    capsys.readouterr()
    made[0].handle(record)
    narrow.flush()
    said = narrow.buffer.getvalue().decode("ascii")
    assert "--- Logging error ---" in said and "UnicodeEncodeError" in said, (
        f"the failed write was not reported on the console: {said!r}")
    assert "Message: 'the \\u03c7 probe table holds no row'" in said, (
        f"the record's words were not written escaped: {said!r}")
    assert capsys.readouterr().err == "", "the failed write was reported on the process's error stream"


def test_the_error_page_writes_the_errors_message_as_a_sentence():
    """The error page gives the error's message as a sentence after its bold opening words: its first letter a
    capital only when its first word -- up to the first space, a comma, a semicolon or a colon at its end set aside
    -- is an ordinary word, letters alone, an apostrophe or a hyphen between two of them counted in, that starts
    with a small one; any other start is kept as it was raised: a name in code, a path, a number, a quotation. A
    message that holds no word outside its quotations, as a missing key's name is raised, follows the general
    sentence and a colon. A full stop stands at its end when it ends without a mark that ends a sentence, and
    nothing else of it changes. A message that is a sentence already is shown as it is, and an empty one leaves the
    opening words alone."""
    from core.gui.screens.shell import PartError
    qt_app()
    for message, shown in (("the cell file could not be read", "The cell file could not be read."),
                           ("The cell file could not be read.", "The cell file could not be read."),
                           ("is the cell file there?", "Is the cell file there?"),
                           ("missing: the bounds file", "Missing: the bounds file."),
                           ("missing, the bounds file", "Missing, the bounds file."),
                           ("missing; the bounds file", "Missing; the bounds file."),
                           ("can't multiply sequence by non-int of type 'float'",
                            "Can't multiply sequence by non-int of type 'float'."),
                           ("non-finite values in the prior", "Non-finite values in the prior."),
                           ("my-cells/x.txt not found", "my-cells/x.txt not found."),
                           ("'model' is missing", "'model' is missing."),
                           ("'model'", "Something went wrong that PRISM did not expect: 'model'."),
                           ("0", "Something went wrong that PRISM did not expect: 0."),
                           ("sbi_posterior.sample() failed", "sbi_posterior.sample() failed."),
                           ("cells/x.txt not found", "cells/x.txt not found."),
                           ("C:\\cells\\x.txt not found", "C:\\cells\\x.txt not found."),
                           ("x2 is out of its box", "x2 is out of its box."),
                           ("[Errno 2] No such file or directory: 'cell.txt'",
                            "[Errno 2] No such file or directory: 'cell.txt'."),
                           ("'NoneType' object has no attribute 'model'",
                            "'NoneType' object has no attribute 'model'."),
                           ("", "")):
        page = build_part(PartError, part_name="Help", error=_CannotBeBuilt(message), trace="")
        sentence = " ".join(words for words in ("Help could not be opened.", shown) if words)
        assert sentence in _shown_words(page), (message, _shown_words(page))


def test_the_rail_marks_a_part_before_it_is_built_and_the_pointer_waits_while_it_is(monkeypatch):
    """While a part is built at its first opening its row of the rail is marked already, and was drawn so, and the
    pointer shows the wait cursor; once the part is built the pointer is the window's own again, and so it is when
    the construction raised, the part's row then marked over its error page. A part handed by ``part`` is built
    under the wait cursor too, the rail left as it was."""
    from core.gui import main_window
    from core.gui.panels.simulate_panel import SimulatePanel
    from core.gui.screens.help_screen import HelpScreen
    from core.gui.screens.shell import PartError
    from core.gui.widgets.rail import PartRow
    app = qt_app()
    painted, seen = [], []
    kept = PartRow.paintEvent
    monkeypatch.setattr(PartRow, "paintEvent",
                        lambda row, event: (painted.append((row.place.key, row._current)), kept(row, event))[1])

    def watched(name, cls=None):
        """A construction of the part ``name`` that notes what the rail and the pointer show while it runs, then
        builds ``cls``, or raises when there is none."""
        def build():
            cursor = app.overrideCursor()
            seen.append((name, _marked(window.shell.rail), (name, True) in painted,
                         None if cursor is None else cursor.shape()))
            if cls is None:
                raise _CannotBeBuilt("the cell file could not be read")
            return build_part(cls)
        return build

    monkeypatch.setattr(main_window, "ReductionPanel", watched("reduction"))
    monkeypatch.setattr(main_window, "SimulatePanel", watched("live", SimulatePanel))
    monkeypatch.setattr(main_window, "HelpScreen", watched("help", HelpScreen))
    window = build_window()
    try:
        _shown_off_screen(window)
        for name, handed, marked in (("reduction", False, True), ("live", False, True), ("help", True, False)):
            painted.clear()
            if handed:
                window.part(name)
            else:
                window.go_to(name)
            assert seen[-1] == (name, [name] if marked else ["live"], marked, Qt.WaitCursor), seen[-1]
            assert app.overrideCursor() is None, f"the wait cursor outlived the build of {name}"
            assert _marked(window.shell.rail) == ([name] if marked else ["live"]), name
        assert isinstance(window.part("reduction"), PartError) and window.built_parts() == ("home", "live", "help")
    finally:
        window.hide()


def test_a_part_first_opened_during_a_run_comes_up_locked_and_is_freed_at_its_end():
    """A part first opened while a run is live comes up locked as the parts built already are: the controls of
    each of its panels are greyed, those of the six steps of Parameter inference among them, while the rail still
    opens every part, and the keyboard, which no locked setting can take, goes to the place's row of the rail
    instead of staying on the page the move left. When the run ends the panels are freed with the rest, and a move
    gives a page's first setting the keyboard again. The run here is the Reduction map's."""
    from core.gui.panels.reduction_panel import ReductionPanel
    qt_app()
    window = build_window()
    rail, runner = window.shell.rail, window.panel(ReductionPanel)
    assert window.built_parts() == ("home", "reduction")
    with running(runner):
        assert window.built_parts() == ("home", "reduction"), "a run's start built a part"
        for name, row in (("inference", "model"), ("temperature", "temperature"), ("live", "live")):
            window.go_to("home")                                  # the keyboard on Home's main button
            window.go_to(name)
            panels = window._panels_of(name, window.part(name))
            assert [panel.controls.isEnabled() for panel in panels] == [False] * len(panels), (
                f"{name}, first opened during a run, is not locked")
            assert (window.current_place()[0], window.focusWidget()) == (name, rail.row(row)), name
        assert len(window._all_panels()) == 10 and not runner.controls.isEnabled()
    assert [panel.controls.isEnabled() for panel in window._all_panels()] == [True] * 10, "the run's end freed none"
    window.go_to("home")
    window.go_to("live")
    assert window.focusWidget() is window.part("live").model_combo


def test_a_part_handed_or_built_again_during_a_run_comes_up_locked_and_is_freed_at_its_end(monkeypatch):
    """A part built while a run is live comes up locked by every route, as one first opened by a move does: a part
    handed by ``part`` (Live simulation), and a part built again by "Try again" on its error page (Effective
    temperature). The run's end frees them with the rest. The run here is the Reduction map's."""
    from core.gui import main_window
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.screens.temperature_screen import TemperatureScreen
    qt_app()

    def broken():
        raise _CannotBeBuilt("the cell file could not be read")

    monkeypatch.setattr(main_window, "TemperatureScreen", broken)
    window = build_window()
    window.go_to("temperature")                                   # its error page
    runner = window.panel(ReductionPanel)
    with running(runner):
        assert not window.part("live").controls.isEnabled(), "a part handed during a run is not locked"
        monkeypatch.setattr(main_window, "TemperatureScreen", TemperatureScreen)
        _button(window.part("temperature"), "Try again").click()
        rebuilt = window.part("temperature")
        assert isinstance(rebuilt, TemperatureScreen)
        assert [panel.controls.isEnabled() for panel in rebuilt.panels()] == [False, False], (
            "a part built again during a run is not locked")
    assert [panel.controls.isEnabled() for panel in window._all_panels()] == [True] * 4, "the run's end freed none"


def test_a_refusal_shown_in_a_part_builds_no_other_part_and_asks_the_card_nothing(monkeypatch):
    """A refusal Live simulation's panel shows in the window builds no other part and asks the graphics card
    nothing."""
    from core.gui.panels.simulate_panel import SimulatePanel
    from core.refusals import Refusal
    qt_app()
    window = build_window()
    window.go_to("live")
    calls = _card_and_programs(monkeypatch)
    window.panel(SimulatePanel)._on_error(Refusal("The seed is not a whole number.", field="seed"), "")
    assert (window.built_parts(), calls) == (("home", "live"), []), (window.built_parts(), calls)


def test_a_part_never_opened_keeps_what_it_remembered_when_the_window_closes(tmp_path):
    """Closing the window writes what the parts opened in that launch remember and nothing for a part never
    opened: every value stored for one is as it was stored, the layout of each splitter among them, though such a
    part, built and closed, writes values of its own in their place. Here Live simulation alone is opened."""
    from PySide6.QtCore import QByteArray
    from core.gui import settings as st
    qt_app()
    every = build_window()                    # what each part writes as the window closes, off one that opened all
    for place in keys.PARTS:
        every.go_to(place.key)
    every.close()
    written = st.settings()
    remembered = {key: "kept from the last launch" for key in written.allKeys()
                  if key.split("/")[0] not in ("window", "simulate")}
    remembered.update({f"layout/{panel.layout_key()}/{splitter}": QByteArray(b"kept from the last launch")
                       for panel in every._all_panels() for splitter in ("splitter", "results")})
    groups = {key.split("/")[0] for key in remembered}
    assert groups == {"artifacts", "crossval", "fdt", "inference_config", "inference_infer", "inference_posterior",
                      "inference_prior", "inference_tsnpe", "layout", "reduction"}, groups
    st.use_ini_file(str(tmp_path / "kept.ini"))
    kept = st.settings()
    for key, value in remembered.items():
        kept.setValue(key, value)
    kept.sync()
    window = build_window()
    window.go_to("live")
    window.close()
    after = st.settings()
    assert {key: after.value(key) for key in remembered} == remembered, "a part never opened wrote at the close"
    assert [key for key in after.allKeys() if key.startswith("simulate/")], "the part opened wrote nothing"


def test_the_memory_limit_in_force_does_not_wait_for_parameter_inference_to_be_opened(monkeypatch):
    """The memory limit a run's planner reads -- the environment's when the variable is set, else the
    configuration's ceiling -- is the same before Parameter inference is first opened as once it has been, with
    the variable set and unset: opening the part, whose field writes the ceiling it shows, changes nothing a run
    started before it would plan with."""
    from core import config
    from core.SBI import pipeline
    qt_app()
    monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", config.SIM_VRAM_CEILING_GIB)    # put back at the end
    for ceiling in (None, "6"):
        if ceiling is None:
            monkeypatch.delenv(pipeline.VRAM_CEILING_ENV, raising=False)
        else:
            monkeypatch.setenv(pipeline.VRAM_CEILING_ENV, ceiling)
        window = build_window()
        before = pipeline.vram_ceiling_gib()
        assert "inference" not in window.built_parts(), ceiling
        window.go_to("inference")
        shown = window.part("inference").config_panel.vram_ceiling.value()
        assert pipeline.vram_ceiling_gib() == before == shown, (ceiling, before, shown)
    assert before == 6.0


def test_a_store_change_reaches_only_the_built_parts_and_a_part_built_later_lists_the_store_as_it_is(
        tmp_path, monkeypatch):
    """A change of the store in Records has every artifact picker of the parts built list the store again, before
    the steps' states are worked out again. The pickers are found by type, so that a picker added to a page is
    covered by construction: what is held is every one of them, in the order the parts' panels are read, the kinds
    of today among them, never a count, which a picker added later would break. A part not built yet has no
    picker to tell; built later, its pickers list the store as it is then."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.widgets.artifact_picker import StorePicker
    from tests._fixtures import build_fdt_record
    qt_app()
    with use_store(ArtifactStore(tmp_path / "Artifacts")) as store:
        window = build_window()
        window.go_to("records")
        window.go_to("inference")
        pickers = [picker for panel in window._all_panels() for picker in panel.findChildren(StorePicker)]
        assert {"prior", "posterior", "observation"} <= {picker.kind for picker in pickers}, pickers
        seen = []
        for picker in pickers:
            monkeypatch.setattr(picker, "refresh",
                                lambda kept=picker.refresh, picker=picker: (seen.append(picker), kept())[1])
        made = build_fdt_record(store, name="written_later")
        window.part("records").store_changed.emit()
        assert seen[:len(pickers)] == pickers, [picker.kind for picker in seen]
        assert "temperature" not in window.built_parts()
        window.go_to("temperature")
        listed = window.part("temperature").fdt_panel.record_picker
        assert made in [listed.combo.itemData(n) for n in range(listed.combo.count())], (
            "a part built later lists the store as it was")


def test_home_reads_an_empty_session_before_parameter_inference_is_built(tmp_path):
    """Until Parameter inference is built Home reads a session that holds nothing, and that session's six steps'
    states as the rules give them, which the rail shows too: the states the part works out when it is built, so
    that its first opening changes nothing Home or the rail say. Home's reading builds no part."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.session import SbiSession
    from tests._fixtures import build_browse_store
    qt_app()
    build_browse_store(tmp_path / "Artifacts")                    # records, so that Home states its next step
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window = build_window()
        rail, home, facts = window.shell.rail, window.part("home"), window._home_facts()
        empty = chain.step_states(SbiSession(), label=record_words.label_of)
        assert window.built_parts() == ("home",)
        assert (facts.session, facts.states) == (SbiSession(), empty)
        assert [rail.row(step.key).accessibleName() for step in keys.STEPS] == [state.spoken for state in empty]
        assert home.next_heading.label.text() == "Next: choose a model" and not home.is_first_time()
        window.go_to("inference")
        assert window.part("inference").step_states() == empty == window._home_facts().states
        assert [rail.row(step.key).accessibleName() for step in keys.STEPS] == [state.spoken for state in empty]


def test_a_part_built_later_reads_the_appearance_and_the_models_in_force():
    """A change of appearance is shown on Settings once it is built, and a model of your own saved or deleted has
    the lists of models of the panels built judged again; a part not built yet is told neither, and reads both as
    they are when it is built: Settings marks the appearance in force, and a panel lists the models in force."""
    from core.config import VALID_MODELS
    qt_app()
    window = build_window()
    window.set_appearance_mode("dark")
    assert window.built_parts() == ("home",)
    window.go_to("settings")
    assert window.part("settings")._radios["dark"].isChecked(), "Settings built later shows another appearance"
    window.go_to("live")
    VALID_MODELS.append("DUFFING")                                 # a model of your own, saved
    try:
        window._on_user_models_changed("DUFFING")
        live = window.part("live").model_combo
        assert "DUFFING" in [live.itemText(n) for n in range(live.count())]
        assert "temperature" not in window.built_parts()
        window.go_to("temperature")
        fdt = window.part("temperature").fdt_panel.model_combo
        assert "DUFFING" in [fdt.itemText(n) for n in range(fdt.count())], "a part built later lists old models"
    finally:
        VALID_MODELS.remove("DUFFING")


# The key map in the window. A key of the map is a shortcut of the window, which only the window in front hears, so
# a test that presses one first brings the window to the front (``_in_front``).
def test_every_key_of_the_map_does_what_its_row_says(monkeypatch):
    """The window makes one shortcut for each row of the key map and no other, each in the window's own context,
    so that only the window in front hears it, and none repeating while its key is held down; each row of the rail
    names in its tool tip the keys that open its place, and Help prints the rows of the map. With the window in
    front, each part's key and each step's key, pressed where a person's key goes, shows that place and gives its
    page the keyboard as a shortcut gives it, so that the focus ring shows there; a part first opened by its key,
    and Parameter inference first opened by a step's key, is built under the wait cursor with its row of the rail
    marked already. The ring shows also where the keyboard already stood, after a click in the help text or a move
    the mouse made, neither of which shows it. The key to the run goes to the running place only while the foot is
    a button, and moves nothing once the run's end has been seen; the key between the rail and the page takes the
    keyboard from the page to the rail and back."""
    from PySide6.QtGui import QShortcut
    from core.gui import main_window, style
    from core.gui.panels.reduction_panel import ReductionPanel
    app = qt_app()
    built, kept = [], main_window.MainWindow._make

    def make(window, name):
        """The construction of a part, noting what the rail marks and what the pointer shows while it runs."""
        cursor = app.overrideCursor()
        built.append((name, _marked(window.shell.rail), None if cursor is None else cursor.shape()))
        return kept(window, name)

    monkeypatch.setattr(main_window.MainWindow, "_make", make)
    given = _given(monkeypatch)
    window, other = build_window(), build_window()      # first: a window that cannot be built leaves no follower
    rail = window.shell.rail
    follower = style.FocusFollower(app)
    monkeypatch.setattr(style, "_FOLLOWER", follower)         # the application's follower, as the window finds it
    try:
        made = sorted(((shortcut.key().toString(QKeySequence.PortableText), str(shortcut.property("does")),
                        shortcut.context(), shortcut.autoRepeat()) for shortcut in window.findChildren(QShortcut)),
                      key=repr)                          # in words, so that a shortcut made twice is named too
        assert made == sorted(((sequence, does, Qt.WindowShortcut, False) for sequence, does in keys.BINDINGS),
                              key=repr), f"the window's shortcuts are not the key map's rows: {made}"
        assert [rail.row(place.key).toolTip().rpartition(" (")[2] for place in keys.PARTS + keys.STEPS] == [
            ", ".join(place.keys) + ")" for place in keys.PARTS + keys.STEPS], "a row's tool tip names no key"
        _shown_off_screen(window)
        _shown_off_screen(other)
        with _in_front(window):
            for sequence, does in keys.BINDINGS:
                kind, _, name = does.partition(" ")
                if kind not in ("part", "step"):
                    continue
                part, page = (name, None) if kind == "part" else ("inference", name)
                first = part not in window.built_parts()
                del built[:], given[:]
                _press(window, sequence)
                shown, held = window.current_place(), window.focusWidget()
                assert shown[0] == part and (page is None or shown[1] == page), f"{sequence} opened {shown}"
                assert held is not None and window.part(part).isAncestorOf(held), f"{sequence}: no keyboard on the page"
                assert given[-1:] == [(held, Qt.ShortcutFocusReason)] and follower.target() is held, (
                    f"{sequence}: the page took the keyboard otherwise than as a shortcut gives it")
                row = shown[1] if part == "inference" else part
                assert built == ([(part, [row], Qt.WaitCursor)] if first else []), f"{sequence} built {built}"
            viewer = window.part("help").viewer
            for sequence in ("F1", "Ctrl+9"):          # Help opened by a click on its row, a click in its text, its key
                QTest.mouseClick(rail.row("help"), Qt.LeftButton)
                QTest.mouseClick(viewer.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(4, 4))
                assert (window.focusWidget(), follower.target()) == (viewer, None), (
                    "a click in the help text showed the ring")
                _press(window, sequence)
                assert follower.target() is viewer, f"{sequence}: no ring on the help text, which held the keyboard"
            for sequence, does in keys.BINDINGS:        # each place's own key, after a move the mouse made there
                kind, _, name = does.partition(" ")
                if kind in ("part", "step"):
                    QTest.mouseClick(window.shell, Qt.LeftButton)       # a press of the mouse ends the keys' turn
                    window.go_to(*((name, None) if kind == "part" else ("inference", name)))
                    held = window.focusWidget()
                    _press(window, sequence)
                    assert (window.focusWidget(), follower.target()) == (held, held), (
                        f"{sequence}: no ring on the control that held the keyboard")
            window.go_to("home")
            _press(window, "Ctrl+R")
            assert (rail.foot.is_button(), window.current_place()) == (False, ("home", None)), (
                "the key to the run moved with nothing running")
            with running(build_run_host()):            # a run of a panel with no place: the foot is no button
                _press(window, "Ctrl+R")
                assert (rail.foot.is_button(), window.current_place()) == (False, ("home", None)), (
                    "the key to the run moved while the foot is no button")
            with running(window.panel(ReductionPanel)):
                del given[:]
                _press(window, "Ctrl+R")
                assert (rail.foot.is_button(), window.current_place()) == (True, ("reduction", None)), (
                    "the key to the run did not go to the run")
                assert given[-1:] == [(rail.row("reduction"), Qt.ShortcutFocusReason)], (     # its settings locked
                    "the running place took the keyboard otherwise than as a shortcut gives it")
            window.go_to("home")
            _press(window, "Ctrl+R")
            assert (rail.foot.is_button(), window.current_place()) == (False, ("home", None)), (
                "the key to the run moved after the run ended")
            window.go_to("live")
            field = window.focusWidget()
            _press(window, "F6")
            row = rail.row("live")
            assert (window.focusWidget(), given[-1:]) == (row, [(row, Qt.ShortcutFocusReason)]), (
                "the key between the rail and the page did not reach the rail")
            _press(window, "F6")
            assert (window.focusWidget(), given[-1:]) == (field, [(field, Qt.ShortcutFocusReason)]), (
                "the key between the rail and the page did not come back to the page")
        with _in_front(other):
            del built[:]
            _press(other, "Alt+3")
            assert (other.current_place(), built) == (("inference", "training"), [
                ("inference", ["training"], Qt.WaitCursor)]), f"a step's key opened {other.current_place()}: {built}"
        assert _cells(window.part("help").keys_table) == [list(row) for row in keys.KEYS], "Help prints another map"
    finally:
        window.hide()
        other.hide()
        follower.stop()


def test_the_key_between_the_rail_and_the_page_moves_the_keyboard_both_ways(monkeypatch):
    """F6 and Shift+F6 each move the keyboard between the rail and the page, either way round, as a shortcut gives
    it, the focus ring showing: from the page's first setting to the row of the place in sight, a step's row on a
    step's page, and from there back to that setting; from the run's foot, while it is a button, to the page; and
    from nowhere, when no control holds the keyboard, to the rail. On a page whose every setting a run has locked,
    the place's row, which the move gave the keyboard, gives it on to the page's first control that is not a button,
    and takes it back; on either page of Effective temperature that control is none of its row of page names."""
    from PySide6.QtWidgets import QPushButton, QToolButton
    from core.gui import style
    from core.gui.panels.reduction_panel import ReductionPanel
    app = qt_app()
    given = _given(monkeypatch)
    window = build_window()                             # first: a window that cannot be built leaves no follower
    rail = window.shell.rail
    follower = style.FocusFollower(app)
    try:
        _shown_off_screen(window)
        with _in_front(window):
            for part, page, row in (("live", None, "live"), ("inference", "training", "training"),
                                    ("temperature", "sweep", "temperature"), ("help", None, "help")):
                window.go_to(part, page)
                first = window.focusWidget()
                for sequence, to in (("F6", rail.row(row)), ("F6", first), ("Shift+F6", rail.row(row)),
                                     ("Shift+F6", first)):
                    _press(window, sequence)
                    assert (window.focusWidget(), given[-1], follower.target()) == (
                        to, (to, Qt.ShortcutFocusReason), to), f"{part}: {sequence} did not move the keyboard as a key"
            window.go_to("home")
            window.focusWidget().clearFocus()
            assert window.focusWidget() is None
            _press(window, "F6")
            assert window.focusWidget() is rail.row("home"), "the key from nowhere did not reach the rail"
            with running(window.panel(ReductionPanel)):
                window.go_to("home")
                _press(window, "F6")
                QTest.keyClick(window.focusWidget(), Qt.Key_Tab)            # on to the foot, a button now
                assert window.focusWidget() is rail.foot, "Tab from the rail did not reach the foot"
                _press(window, "F6")
                assert window.focusWidget() is window.part("home").first_focus(), "the foot kept the keyboard"
                window.go_to("live")                                         # every setting locked by the run
                assert window.focusWidget() is rail.row("live"), "a move to a page a run locks left the rail"
                _press(window, "F6")
                held = window.focusWidget()
                assert window.part("live").isAncestorOf(held) and held.isEnabled(), (
                    "the key did not reach a page a run locks")
                assert not isinstance(held, (QPushButton, QToolButton)) and given[-1] == (
                    held, Qt.ShortcutFocusReason), "the key gave a page a run locks a button, or not as a key"
                _press(window, "Shift+F6")
                assert window.focusWidget() is rail.row("live"), "the key did not take the keyboard back to the rail"
                for page in ("one_cell", "sweep"):                           # a part of pages, every setting locked
                    window.go_to("temperature", page)
                    moved = window.focusWidget()
                    _press(window, "F6")
                    shown, held = window.part("temperature"), window.focusWidget()
                    assert (moved, shown.isAncestorOf(held), shown.names.isAncestorOf(held)) == (
                        rail.row("temperature"), True, False), f"{page}: the key from the rail reached the page names"
    finally:
        window.hide()
        follower.stop()


def test_the_help_key_opens_help(monkeypatch):
    """F1 opens Help from wherever the window is -- a step, a page of Effective temperature, the model builder with
    the keyboard in a field one types in, Help itself -- and so does Ctrl+9: Help is shown, its row of the rail is
    marked, the window says where it is, and the help text takes the keyboard as a shortcut gives it."""
    qt_app()
    given = _given(monkeypatch)
    window = build_window()
    said = []
    window.place_changed.connect(lambda part, page: said.append((part, page)))
    try:
        _shown_off_screen(window)
        with _in_front(window):
            for start in (("inference", "training"), ("temperature", "sweep"), ("models", "builder"), ("help", None)):
                for sequence in ("F1", "Ctrl+9"):
                    window.go_to(*start)
                    del said[:]
                    _press(window, sequence)
                    viewer = window.part("help").viewer
                    assert (window.current_place(), _marked(window.shell.rail), said) == (
                        ("help", None), ["help"], [("help", None)]), f"{sequence} did not open Help from {start}"
                    assert (window.focusWidget(), given[-1]) == (viewer, (viewer, Qt.ShortcutFocusReason)), (
                        f"{sequence} from {start}: the help text did not take the keyboard as a key gives it")
    finally:
        window.hide()


def test_escape_in_the_rail_gives_the_keyboard_back_to_the_page_and_elsewhere_moves_nothing(monkeypatch):
    """Escape in the rail, on the row of the place in sight or on another row the arrows reached, gives the page in
    sight its first setting, as a shortcut gives it; on a page whose every setting a run has locked, the page's first
    control that is not a button, never the rail again, and from the run's foot the same. Escape anywhere else moves
    nothing: no page changes, the keyboard stays where it is and nothing is said; and a run goes on, its stop
    asked by no Escape, in the rail or on its own page's stop button."""
    from PySide6.QtWidgets import QPushButton, QToolButton
    from core.gui.panels.reduction_panel import ReductionPanel
    qt_app()
    given = _given(monkeypatch)
    window = build_window()
    rail = window.shell.rail
    said = []
    window.place_changed.connect(lambda part, page: said.append((part, page)))
    try:
        _shown_off_screen(window)
        with _in_front(window):
            for part, page in (("live", None), ("inference", "prior"), ("records", None), ("models", "builder")):
                window.go_to(part, page)
                first = window.focusWidget()
                rail.take_keyboard()
                QTest.keyClick(window.focusWidget(), Qt.Key_Down)           # another row the arrows reached
                QTest.keyClick(window.focusWidget(), Qt.Key_Escape)
                assert (window.focusWidget(), given[-1]) == (first, (first, Qt.ShortcutFocusReason)), (
                    f"Escape in the rail did not give the first setting of {part} the keyboard as a key")
                del said[:]
                QTest.keyClick(first, Qt.Key_Escape)
                assert (window.current_place()[0], window.focusWidget(), said) == (part, first, []), (
                    f"Escape on the first setting of {part} moved something")
            runner = window.panel(ReductionPanel)
            with running(runner):
                for target in ("reduction", "inference"):
                    window.go_to(target)                                     # every setting locked by the run
                    QTest.keyClick(window.focusWidget(), Qt.Key_Escape)
                    held = window.focusWidget()
                    assert not rail.isAncestorOf(held), f"Escape left the keyboard in the rail over {target}"
                    assert window.part(target).isAncestorOf(held), f"Escape gave the keyboard away from {target}"
                    assert not isinstance(held, (QPushButton, QToolButton)), f"Escape gave a button of {target} it"
                window.go_to("reduction")
                rail.take_keyboard()
                QTest.keyClick(window.focusWidget(), Qt.Key_Tab)            # the foot, a button while the run is live
                assert window.focusWidget() is rail.foot, "Tab from the rail did not reach the foot"
                QTest.keyClick(rail.foot, Qt.Key_Escape)
                assert window.part("reduction").isAncestorOf(window.focusWidget()), "Escape on the foot moved nothing"
                stop = runner.run_view.stop_button
                assert (controller().is_live(), controller().status.stop_requested,
                        controller()._run.token.requested.is_set(), stop.isEnabled()) == (
                    True, False, False, True), "Escape in the rail asked the run to stop"
                stop.setFocus(Qt.TabFocusReason)
                del said[:]
                QTest.keyClick(stop, Qt.Key_Escape)
                assert (controller().is_live(), controller().status.stop_requested,
                        controller()._run.token.requested.is_set(), stop.isEnabled()) == (
                    True, False, False, True), "Escape asked the run to stop"
                assert (window.current_place(), window.focusWidget(), said) == (
                    ("reduction", None), stop, []), "Escape on the stop button moved something"
    finally:
        window.hide()


def test_tab_reaches_the_rail_then_the_foot_when_it_is_a_button_then_the_page():
    """The Tab key reaches the rail's list as one stop, on the row of the place in sight, then the run's foot while
    it is a button, then the page from its first stop on, and from the page's last stop the rail again; Shift+Tab
    walks the same stops backwards. No other row of the rail is a stop, the steps under Parameter inference
    neither. On Effective temperature the page's first stop is its row of page names, which stands over the page."""
    from core.gui.panels.reduction_panel import ReductionPanel
    qt_app()
    window = build_window()
    rail = window.shell.rail

    def walk(start, key, count):
        """Where the keyboard goes, press after press of ``key`` from ``start``: the rail, the foot, or the page."""
        start.setFocus(Qt.TabFocusReason)
        been = []
        for _ in range(count):
            QTest.keyClick(window.focusWidget(), key, Qt.ShiftModifier if key == Qt.Key_Backtab else Qt.NoModifier)
            held = window.focusWidget()
            been.append("foot" if held is rail.foot else held if rail.isAncestorOf(held) else
                        "page" if window.shell.stack.currentWidget().isAncestorOf(held) else held)
        return been

    try:
        _shown_off_screen(window)
        for part, page, row in (("home", None, "home"), ("inference", "training", "training"),
                                ("temperature", None, "temperature"), ("help", None, "help")):
            window.go_to(part, page)
            stops = walk(rail.row(row), Qt.Key_Tab, 40)
            assert stops[0] == "page" and set(stops) == {"page", rail.row(row)}, (
                f"{part}: Tab reached another stop than the rail's row and the page")
            assert stops[stops.index(rail.row(row)) + 1] == "page", f"{part}: after the rail Tab did not reach the page"
            back = walk(rail.row(row), Qt.Key_Backtab, 40)
            assert back[0] == "page" and set(back) == {"page", rail.row(row)}, (
                f"{part}: Shift+Tab reached another stop than the rail's row and the page")
        window.go_to("temperature", "sweep")
        walk(rail.row("temperature"), Qt.Key_Tab, 1)
        assert window.part("temperature").names.isAncestorOf(window.focusWidget()), (
            "the first stop of Effective temperature is not its row of page names")
        with running(window.panel(ReductionPanel)):
            for part in ("home", "help"):
                window.go_to(part)
                assert walk(rail.row(part), Qt.Key_Tab, 2) == ["foot", "page"], (
                    f"{part}: Tab from the rail did not reach the foot, a button, and then the page")
                first = window.focusWidget()
                assert walk(first, Qt.Key_Backtab, 2) == ["foot", rail.row(part)], (
                    f"{part}: Shift+Tab from the page did not reach the foot and then the rail")
    finally:
        window.hide()


def test_after_a_move_the_keyboard_is_on_the_pages_first_setting_never_on_a_button_that_starts_a_run(monkeypatch):
    """After a move to any place -- by its key, each part and each step, and by ``go_to``, each page of a part that
    has several -- the keyboard is on the place's first setting, or on the button the place names to start at, which
    opens a place and starts nothing: Home's main button, and the button of Your own models' list that opens the
    builder. It is never on a button that starts or stops a run. Enter there starts no run and asks nothing, and a
    press of the button a place names starts none either. A part that cannot be built, opened by its key or by
    Enter on its row of the rail, gives the keyboard to its error page's "Try again", as a shortcut gives it."""
    from PySide6.QtWidgets import QAbstractButton, QPushButton, QToolButton
    from core.gui import main_window
    from core.gui.panels.base_panel import BasePanel
    from tests._fixtures import all_asked, forget_asked
    qt_app()
    given = _given(monkeypatch)
    started = []
    monkeypatch.setattr(BasePanel, "dispatch", lambda panel, *args, **kwargs: started.append(type(panel).__name__))
    window = build_window()
    moves = ([(sequence, *(("inference", does[5:]) if does.startswith("step ") else (does[5:], None)))
              for sequence, does in keys.BINDINGS if does.startswith(("part ", "step "))]
             + [(None, "temperature", "one_cell"), (None, "temperature", "sweep"), (None, "models", "list"),
                (None, "models", "builder")])
    try:
        _shown_off_screen(window)
        with _in_front(window):
            for sequence, part, page in moves:
                window.go_to("records")
                if sequence is None:
                    window.go_to(part, page)
                else:
                    _press(window, sequence)
                assert window.current_place()[0] == part and page in (None, window.current_place()[1]), (
                    f"{sequence or 'go_to'} did not open {part}, {page}")
                held, place = window.focusWidget(), window.current_place()
                named = getattr(window.part(part), "first_focus", lambda: None)()
                assert held is not None and window.part(part).isAncestorOf(held) and held.isEnabled(), (
                    f"{place}: the page holds no keyboard")
                if isinstance(held, (QPushButton, QToolButton)):
                    assert held is named, f"{place}: the keyboard is on a button the place does not name"
                forget_asked()
                QTest.keyClick(held, Qt.Key_Return)
                assert (started, all_asked(), window.current_place()) == ([], [], place), (
                    f"{place}: Enter started a run, asked something or moved: {started}")
                if isinstance(held, QAbstractButton) and held is named:
                    held.click()
                    assert (started, all_asked()) == ([], []), f"{place}: the button the place names started a run"
    finally:
        window.hide()

    def broken():
        raise _CannotBeBuilt("the cell file could not be read")

    monkeypatch.setattr(main_window, "ReductionPanel", broken)
    window = build_window()
    try:
        _shown_off_screen(window)
        with _in_front(window):
            _press(window, "Ctrl+5")
            again = _button(window.part("reduction"), "Try again")
            assert (window.focusWidget(), given[-1]) == (again, (again, Qt.ShortcutFocusReason)), (
                "the key of a part that cannot be built did not give its error page's Try again the keyboard")
            window.shell.rail.take_keyboard()
            QTest.keyClick(window.focusWidget(), Qt.Key_Return)
            assert (window.focusWidget(), given[-1]) == (again, (again, Qt.ShortcutFocusReason)), (
                "Enter on the row of a part that cannot be built did not give Try again the keyboard")
    finally:
        window.hide()


def test_no_key_fires_while_a_question_is_open():
    """While a question the window asks is open over the window, no key of the map does anything, pressed at the
    window or at the question, with the question in front as the screen puts it or with the window there: the place,
    the keyboard and the rail stay as they were, and nothing is said. Once the question is answered and the window
    is in front again, the same keys move as their rows say."""
    from core.gui.widgets.ask import Ask
    qt_app()
    window = build_window()
    said = []
    window.place_changed.connect(lambda part, page: said.append((part, page)))
    try:
        _shown_off_screen(window)
        window.go_to("live")
        held = window.focusWidget()
        question = Ask(window, "Start a new session?", "Applying a model replaces the whole session.", "Keep it",
                       "Apply and start a new session", loses_work=True)
        question.setAttribute(Qt.WA_DontShowOnScreen)
        question.show()                       # modal over the window, as the window shows it, with no loop of its own
        del said[:]
        try:
            for front, named in ((question, "the question"), (window, "the window")):
                with _in_front(front):
                    for sequence, _does in keys.BINDINGS:
                        for at in (window, question):
                            _press(at, sequence)
                    assert (window.current_place(), window.focusWidget(), _marked(window.shell.rail), said) == (
                        ("live", None), held, ["live"], []), f"a key moved the window, {named} in front"
        finally:
            question.reject()
        assert question.isHidden(), "the question is still open"
        with _in_front(window):
            for sequence, place in (("Ctrl+3", ("temperature", "one_cell")), ("Alt+2", ("inference", "prior")),
                                    ("F1", ("help", None))):
                _press(window, sequence)
                assert window.current_place() == place, f"{sequence} moved nothing once the question was answered"
    finally:
        window.hide()


def test_help_prints_the_key_map():
    """Help prints the key map under its help text: a table under the heading "Keys", with the columns "Key" and
    "What it does" and a row for each row of the map, in its order and its words, and no row to select, so that it
    takes no stop of the Tab key. In a window of the ordinary opening size the table stands whole under the text,
    which takes the room left, and nothing scrolls; in the smallest window the text keeps its least height, the
    table stands whole under it, and the two scroll together under the page's heading. Exactly while they scroll,
    the middle that holds them is a stop of the Tab key, reached by Shift+Tab from the text, and keeps the room of
    its focus ring round what it holds, with the bar's gap beside it, so that the ring crosses nothing it holds."""
    from PySide6.QtWidgets import QAbstractItemView
    qt_app()
    window = build_window()
    shown = window.part("help")
    table = shown.keys_table
    assert (_titles(table), _cells(table)) == (["Key", "What it does"], [list(row) for row in keys.KEYS]), (
        f"Help's table is not the key map: {_titles(table)}")
    assert (table.view.focusPolicy(), table.view.selectionMode()) == (Qt.NoFocus, QAbstractItemView.NoSelection), (
        "Help's table takes a stop of the Tab key or a selection")
    try:
        for size, scrolls in (((1300, 820), False), ((900, 600), True)):
            _shown_off_screen(window, *size)
            window.go_to("help")
            QApplication.processEvents()
            bar = shown.middle.verticalScrollBar()
            text_ends = shown.viewer.mapTo(shown, QPoint(0, shown.viewer.height())).y()
            assert table.mapTo(shown, QPoint(0, 0)).y() > text_ends, f"the table is not under the text at {size}"
            assert table.height() == table.sizeHint().height(), f"the table is cut at {size}"
            assert (bar.maximum() > 0, shown.viewer.height() == design.HELP_VIEWER_MIN_H) == (scrolls, scrolls), (
                f"at {size} the middle scrolls {bar.maximum()} px and the text is {shown.viewer.height()} px high")
            shown.viewer.setFocus(Qt.TabFocusReason)
            QTest.keyClick(shown.viewer, Qt.Key_Backtab, Qt.ShiftModifier)
            assert (window.focusWidget() is shown.middle) == scrolls, (
                f"at {size} Shift+Tab from the text did not reach the middle exactly while it scrolls")
            room = design.SIZE["focus_ring"] + design.SIZE["focus_gap"] if scrolls else 0
            kept = shown.middle.widget().layout().contentsMargins()
            got = (shown.middle.focusPolicy(), (kept.left(), kept.top(), kept.right(), kept.bottom()))
            assert got == (Qt.TabFocus if scrolls else Qt.NoFocus,
                           (room, room, room + (design.SIZE["scroll_gap"] if scrolls else 0), room)), (
                f"the middle at {size}: {got}")
    finally:
        window.hide()


# One PRISM at a time. A start claims the settings file it was given; a test claims one of its own, in its own
# temporary folder, and lets go of what it claimed before it ends.
def _answers(name):
    """Whether a local server listens under ``name``: a connection to it is made, and let go at once."""
    from PySide6.QtNetwork import QLocalSocket
    socket = QLocalSocket()
    socket.connectToServer(name)
    made = socket.waitForConnected(300)
    socket.abort()
    return made


def test_no_key_is_made_outside_the_key_map():
    """Every key of the window is made in one place: the window's loop over the key map, which Help prints and the
    keys' test reads. No other line under core/gui makes a shortcut or sets one on an action, so that a key added
    anywhere else cannot escape them."""
    from pathlib import Path
    gui = Path(__file__).resolve().parents[1] / "core" / "gui"
    made = []
    for path in sorted(gui.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines):
            if "QShortcut(" in line or "setShortcut(" in line:
                above = next((earlier.strip() for earlier in reversed(lines[:number]) if earlier.strip()), "")
                made.append((path.relative_to(gui).as_posix(), line.strip(), above))
    assert made == [("main_window.py", "shortcut = QShortcut(QKeySequence(sequence), self)",
                     "for sequence, does in keys.BINDINGS:")], made


def test_which_case_a_start_of_prism_falls_in():
    """A start that took the lock starts; one that found it held is handed over when the open PRISM took its
    connection; when nothing took it, it starts once the lock was removed, and gives up when the lock could not be
    removed."""
    from core.gui.one_instance import start_case
    cases = [(start_case(locked=True, connected=False, removed=None), "start"),
             (start_case(locked=False, connected=True, removed=None), "hand_over"),
             (start_case(locked=False, connected=False, removed=True), "start_after_removal"),
             (start_case(locked=False, connected=False, removed=False), "give_up")]
    assert [got for got, _wanted in cases] == [wanted for _got, wanted in cases]


def test_the_servers_name_is_one_per_settings_file(tmp_path):
    """The server a second start knocks on is named after the settings file: two files, two names; one file whose
    path is written another way -- its letters in another case where the system does not tell them apart, as on
    Windows, the other separator, a "." or a ".." among its folders -- one name; and the name is one a local server
    listens under."""
    import os
    from PySide6.QtNetwork import QLocalServer
    from core.gui.one_instance import server_name
    qt_app()
    path = str(tmp_path / "PRISM" / "PRISM.ini")
    folds = os.path.normcase("A") == "a"                 # the system's paths do not tell a letter's case apart
    assert folds == (os.name == "nt"), "the case of a path's letters is not set aside on Windows alone"
    ways = ((path.upper(),) if folds else ()) + (path.replace("\\", "/"), str(tmp_path / "PRISM" / "." / "PRISM.ini"),
                                                 str(tmp_path / "PRISM" / "other" / ".." / "PRISM.ini"))
    names = {server_name(way) for way in ways} | {server_name(path)}
    assert len(names) == 1, f"one settings file, written {len(ways) + 1} ways, names {len(names)} servers"
    assert server_name(str(tmp_path / "Another" / "PRISM.ini")) != server_name(path), "two settings files share a name"
    server = QLocalServer()
    try:
        assert server.listen(server_name(path)), "no local server listens under the name"
    finally:
        server.close()


def test_a_second_start_is_handed_over_even_while_the_first_is_busy(tmp_path, monkeypatch, brought_forward):
    """A second start on the settings file of a PRISM that is open -- one that holds the lock and listens -- connects
    at once and is handed over, though the open one's event loop has not run since: before it knocks, it lets the
    open one's process take the front. The open one hears the knock when its loop next runs, and not before: not
    while its main thread is held for a second, as while a figure is drawn. The second start claims as a start in
    another process does: in this one it would be handed back the claim the process holds."""
    import os
    from core.gui import one_instance
    from tests._fixtures import as_another_process, pump
    app = qt_app()
    path = str(tmp_path / "prism.ini")
    first = one_instance.claim(path)
    heard, knock = [], one_instance._knock
    first.knocked.connect(lambda: heard.append(True))
    monkeypatch.setattr(one_instance, "_knock", lambda *told: brought_forward.append(("knock",)) or knock(*told))
    try:
        began = time.monotonic()
        with as_another_process():
            second = one_instance.claim(path)
        took = time.monotonic() - began
        assert (first.case, second.case, one_instance.current() is first) == ("start", "hand_over", True), (
            first.case, second.case)
        assert took < 0.5, "the second start was not handed over at once"
        assert brought_forward == [("allow_foreground", os.getpid()), ("knock",)], (
            "the open PRISM was not let take the front before the knock")
        time.sleep(1.0)                               # the open PRISM's main thread is held
        assert heard == [], "the knock was heard before the event loop ran"
        pump(app, 0.2)
        assert heard == [True], "the open PRISM did not hear the knock once its event loop ran"
    finally:
        first.release()


def test_a_lock_left_by_a_prism_that_died_is_taken(tmp_path):
    """A PRISM that died -- its process ended at once, saying nothing -- leaves its lock file behind, and the next
    start takes the lock at once: the toolkit sees that the process the lock names is gone."""
    import os
    import subprocess
    import sys
    import threading
    from pathlib import Path
    from core.gui import one_instance
    qt_app()
    path = str(tmp_path / "prism.ini")
    code = ("import sys\n"
            "sys.path.insert(0, sys.argv[2])\n"
            "from PySide6.QtCore import QCoreApplication\n"
            "from core.gui import one_instance\n"
            "app = QCoreApplication([])\n"
            "print(one_instance.claim(sys.argv[1]).case, flush=True)\n"
            "sys.stdin.read()\n")
    repo = str(Path(__file__).resolve().parents[1])
    child = subprocess.Popen([sys.executable, "-c", code, path, repo], cwd=repo, env=dict(os.environ),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stop = threading.Timer(60, child.kill)            # a child that hangs is ended, and its line reads empty
    stop.start()
    try:
        said = child.stdout.readline().strip()
        assert said == "start", f"the PRISM that is to die did not start: {said!r}"
        child.kill()
        child.wait(timeout=30)
        assert (tmp_path / "PRISM.lock").is_file(), "the lock file went with its process"
        taken = one_instance.claim(path, tries=1, pause_ms=0)
        try:
            assert (taken.case, one_instance.current() is taken) == ("start", True), "the lock was not taken at once"
        finally:
            taken.release()
    finally:
        stop.cancel()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=30)
        child.stdout.close()
        child.stdin.close()


def test_a_lock_nothing_answers_for_is_removed_and_taken_or_else_the_start_gives_up(tmp_path, brought_forward):
    """When the lock is held and nothing takes the connection, the start knocks as it is told (the next test counts
    the knocks); then it removes the lock, takes it and listens, where the lock can be removed -- here a lock file
    that names no process. Where it cannot, the lock's holder lives and does not answer -- here one that never
    listens -- and the start gives up, holding nothing and listening for nothing, having let the holder's process
    take the front."""
    import os
    from PySide6.QtCore import QLockFile
    from core.gui import one_instance
    qt_app()
    path = str(tmp_path / "prism.ini")
    name = one_instance.server_name(path)
    (tmp_path / "PRISM.lock").write_bytes(b"")       # a lock file that names no process
    taken = one_instance.claim(path, tries=3, pause_ms=100)
    try:
        assert (taken.case, one_instance.current() is taken, _answers(name)) == ("start_after_removal", True, True), (
            "the lock nothing answers for was not removed and taken, or nothing listens")
        assert brought_forward == [], "a lock that names no process let a process take the front"
    finally:
        taken.release()
    holder = QLockFile(str(tmp_path / "PRISM.lock"))
    holder.setStaleLockTime(0)
    assert holder.tryLock(0), "the lock was not let go"
    try:
        refused = one_instance.claim(path, tries=2, pause_ms=100)
        assert (refused.case, one_instance.current(), _answers(name)) == ("give_up", None, False), (
            "the start did not give up beside a living holder")
        assert brought_forward == [("allow_foreground", os.getpid())], "the living holder was not let take the front"
    finally:
        holder.unlock()


def test_a_start_knocks_as_many_times_as_it_is_told_a_pause_apart(tmp_path, monkeypatch):
    """Beside a living holder of the lock that never answers, a start knocks as many times as it is told, with the
    pause it is told between two knocks and none before the first, and then gives up; told nothing, it knocks three
    times a second apart. The knocks are counted and the pauses noted, and nothing waits."""
    import inspect
    from types import SimpleNamespace
    from PySide6.QtCore import QLockFile
    from PySide6.QtNetwork import QLocalSocket
    from core.gui import one_instance
    qt_app()
    tried, slept = [], []

    class Counted(QLocalSocket):
        def connectToServer(self, name):
            tried.append(name)
            super().connectToServer(name)

    monkeypatch.setattr(one_instance, "QLocalSocket", Counted)
    monkeypatch.setattr(one_instance, "time", SimpleNamespace(sleep=slept.append))
    holder = QLockFile(str(tmp_path / "PRISM.lock"))
    holder.setStaleLockTime(0)
    assert holder.tryLock(0), "the lock could not be held for the test"
    got = []
    try:
        for told in ({"tries": 2, "pause_ms": 50}, {"tries": 4, "pause_ms": 70}, {}):
            del tried[:], slept[:]
            got.append((one_instance.claim(str(tmp_path / "prism.ini"), **told).case, len(tried), list(slept)))
    finally:
        holder.unlock()
    assert got == [("give_up", 2, [0.05]), ("give_up", 4, [0.07] * 3), ("give_up", 3, [1.0, 1.0])], got
    told = {name: parameter.default for name, parameter in inspect.signature(one_instance.claim).parameters.items()
            if name in ("tries", "pause_ms")}
    assert told == {"tries": 3, "pause_ms": 1000}, told


def test_the_lock_is_made_in_a_settings_folder_that_is_not_there_yet(tmp_path):
    """On a computer where PRISM has never run, the folder of the settings file is not there yet: the start makes it,
    takes the lock in it and listens; once let go, the lock file is gone."""
    from core.gui import one_instance
    qt_app()
    folder = tmp_path / "not" / "yet"
    taken = one_instance.claim(str(folder / "prism.ini"), tries=1, pause_ms=0)
    try:
        assert (taken.case, (folder / "PRISM.lock").is_file(), _answers(one_instance.server_name(
            str(folder / "prism.ini")))) == ("start", True, True), "the lock was not taken in the folder made for it"
    finally:
        taken.release()
    assert sorted(path.name for path in folder.iterdir()) == [], "the lock file stays after it was let go"


def test_a_lock_that_cannot_be_made_holds_no_start_back(tmp_path, caplog):
    """Where the lock cannot be made at all, PRISM starts all the same, holding no lock and listening for nothing,
    and a warning names the lock: here once where the settings file's folder is a file, and once where a folder
    stands in the lock's place, which names no process and cannot be removed."""
    import logging
    from core.gui import one_instance
    qt_app()
    a_file = tmp_path / "a-file"
    a_file.write_text("")
    (tmp_path / "beside" / "PRISM.lock").mkdir(parents=True)
    for settings_file in (a_file / "prism.ini", tmp_path / "beside" / "prism.ini"):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="core.gui.one_instance"):
            made = one_instance.claim(str(settings_file), tries=1, pause_ms=0)
        said = [record.getMessage() for record in caplog.records if record.name == "core.gui.one_instance"]
        assert (made.case, one_instance.current(), _answers(one_instance.server_name(str(settings_file)))) == (
            "start", None, False), "the start was turned away, or holds a lock or a listener"
        assert len(said) == 1 and str(settings_file.parent / "PRISM.lock") in said[0], "no warning names the lock"


def test_a_start_handed_over_builds_no_window_and_exits_zero(tmp_path, monkeypatch, brought_forward):
    """A start of PRISM while PRISM is open on its settings file hands over to the open one: it builds no window and
    no other widget, sets nothing of the look on the process -- the icon, the text size, the style, the fonts, the
    appearance, the figure settings -- and hands back the application with no window, having let the open one's
    process take the front; the open one hears the knock when its loop runs. With no error stream, as a start with
    no console has none, it hands over all the same. The entry point, beside the same open PRISM, ends with the
    status 0, showing nothing and running no event loop: it hands over before it builds anything."""
    import os
    import sys
    from core.gui import one_instance, settings
    from tests._fixtures import as_another_process, process_state, pump, whole_program
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    monkeypatch.setattr(sys, "stderr", None)
    widgets, found = len(QApplication.topLevelWidgets()), process_state(app)
    with whole_program(held_by_another=True) as built:
        changed = sorted(key for key, value in process_state(app).items() if value != found[key])
        assert set(changed) <= {"names", "root log handler", "quiet segment bar", "default store", "one-PRISM lock",
                                "local servers"}, f"the start that was handed over set {changed}"
        held = one_instance.current()
        heard = []
        held.knocked.connect(lambda: heard.append(True))
        assert (built, held.case) == ((app, None), "start"), "the start that was handed over built a window"
        assert len(QApplication.topLevelWidgets()) == widgets, "the start that was handed over built a widget"
        assert brought_forward == [("allow_foreground", os.getpid())], "the open PRISM was not let take the front"
        pump(app, 0.2)
        assert heard == [True], "the open PRISM did not hear the knock"
    from core.gui import __main__ as entry
    ran = []
    monkeypatch.setattr(QApplication, "exec", lambda application: ran.append("a loop") or 3)
    held = one_instance.claim(settings.settings().fileName())          # the open PRISM
    try:
        with _names_kept(app), as_another_process():
            assert (_main(entry), ran) == (0, []), "the start that was handed over ran an event loop"
    finally:
        held.release()


def test_a_start_beside_a_prism_that_never_answers_ends_with_the_status_one(tmp_path, monkeypatch, capsys):
    """The whole program started beside a living holder of the lock that never answers -- a PRISM busy past the
    start's three knocks, or one that does not listen -- builds no window and no other widget, holds no claim, says
    in one line that PRISM did not answer, and ends with the status 1. What the start's claim was is read as the claim
    came back, before the program lets go of anything: a give-up holds none."""
    import os
    from PySide6.QtCore import QLockFile
    from core.gui import one_instance, settings
    from tests._fixtures import whole_program
    qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    made, claim = [], one_instance.claim

    def recording(*args, **kwargs):
        """The start's claim as it comes back: its case, and the claim the process holds at that moment."""
        claimed = claim(*args, **kwargs)
        made.append((claimed.case, one_instance.current()))
        return claimed

    monkeypatch.setattr(one_instance, "claim", recording)
    holder = QLockFile(os.path.join(os.path.dirname(os.path.abspath(settings.settings().fileName())), "PRISM.lock"))
    holder.setStaleLockTime(0)
    assert holder.tryLock(0), "the lock could not be held for the test"
    widgets = len(QApplication.topLevelWidgets())
    try:
        with pytest.raises(SystemExit) as ended:
            with whole_program():
                pass
    finally:
        holder.unlock()
    assert made == [("give_up", None)], f"the start's case and the claim held when it came back: {made}"
    assert (ended.value.code, len(QApplication.topLevelWidgets()), one_instance.current()) == (1, widgets, None), (
        "the start beside a holder that never answers built something, holds a claim or ended otherwise")
    assert capsys.readouterr().err.endswith("PRISM is already running but did not answer; this start ends.\n")


def test_a_second_start_says_in_one_line_what_it_did(tmp_path, monkeypatch):
    """A start that builds no window says what it did in one line on the error stream, the only sign a start from a
    console gives -- that PRISM was open and was asked to come to the front, or that it did not answer and this start
    ends -- and ends with the status 0 or 1. With no error stream, as a start with no console has none, the line is
    written to the window's log, after the launch's dated line. Said where there is no stream at all, nothing is
    written and nothing is raised."""
    import io
    import sys
    from core.gui import __main__ as entry
    from core.gui import one_instance
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    for case, line, status in (("hand_over", "PRISM is already open: it was asked to come to the front.\n", 0),
                               ("give_up", "PRISM is already running but did not answer; this start ends.\n", 1)):
        monkeypatch.setattr(one_instance, "claim", lambda settings_path, case=case, **told: one_instance.Claim(case))
        stream = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        with _names_kept(app):
            assert _main(entry) == status, case
        assert stream.getvalue() == line, stream.getvalue()
        monkeypatch.setattr(sys, "stderr", None)
        with _names_kept(app):
            assert _main(entry) == status, case
        written = (tmp_path / "window-log.txt").read_text(encoding="utf-8").splitlines(keepends=True)
        launch = max(at for at, text in enumerate(written) if text.endswith(f" PRISM started ({sys.executable}).\n"))
        assert written[launch + 1:] == [line], written
        one_instance.say(case)


def test_a_prism_that_cannot_listen_turns_a_second_start_away_and_says_so(tmp_path, monkeypatch, caplog):
    """A PRISM that holds the lock but whose local server does not listen cannot hear a second start of itself, and
    a second start on its settings file is turned away while it is open, as its warning says, naming the server. The
    second start claims as a start in another process does."""
    import logging
    from PySide6.QtNetwork import QLocalServer
    from core.gui import one_instance
    from tests._fixtures import as_another_process
    qt_app()
    path = str(tmp_path / "prism.ini")
    monkeypatch.setattr(QLocalServer, "listen", lambda server, name: False)
    with caplog.at_level(logging.WARNING, logger="core.gui.one_instance"):
        first = one_instance.claim(path)
    try:
        said = [record.getMessage() for record in caplog.records if record.name == "core.gui.one_instance"]
        with as_another_process():
            refused = one_instance.claim(path, tries=1, pause_ms=0)
        assert (first.case, one_instance.current() is first, refused.case) == ("start", True, "give_up"), (
            f"the first start: {first.case}, the second: {refused.case}")
        assert len(said) == 1 and one_instance.server_name(path) in said[0] and (
            "so a second start of PRISM is turned away while this one is open") in said[0], f"the warnings: {said}"
    finally:
        first.release()


def test_the_open_prism_brings_its_window_to_the_front_when_a_second_start_knocks(tmp_path, monkeypatch,
                                                                                    brought_forward):
    """The whole program holds the lock beside its settings file and listens; a second start's knock on it brings
    the program's window to the front once its event loop runs, and not before. The second start claims as a start in
    another process does."""
    import os
    from core.gui import one_instance, settings
    from tests._fixtures import as_another_process, pump, whole_program
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (_app, window):
        with as_another_process():
            second = one_instance.claim(settings.settings().fileName(), tries=1, pause_ms=0)
        assert (second.case, brought_forward) == ("hand_over", [("allow_foreground", os.getpid())]), (
            "the second start was not handed over, the open PRISM let take the front first")
        pump(app, 0.2)
        assert brought_forward[1:] == [("come_to_front", window)], "the knock did not bring the window to the front"


def test_a_prism_whose_window_closed_while_a_run_goes_on_lets_the_next_start_open(tmp_path, monkeypatch):
    """A PRISM whose window is closed while a run still goes on, on the thread pool -- a run left to finish unseen --
    holds its lock and its listener while its event loop runs, until the window's close is accepted, and has let go
    of both by the time the loop ends, though the run goes on; letting go once more as the loop ends does nothing.
    The next start of PRISM on its settings file then takes the lock and opens a window of its own."""
    import threading
    from PySide6.QtCore import QRunnable, QThreadPool
    from core.gui import __main__ as entry
    from core.gui import app as gui_app
    from core.gui import one_instance, settings
    from tests._fixtures import whole_program
    qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    going, ended = threading.Event(), threading.Event()

    class Run(QRunnable):
        def run(self):
            going.wait(30)
            ended.set()

    with whole_program() as (_app, window):
        held, during = one_instance.current(), []

        class Loop:
            """The program's event loop, as far as its end goes: the window is closed while it runs, and it ends."""
            def exec(self):
                during.append(one_instance.current())
                window.close()
                return 0

        window.setAttribute(Qt.WA_DontShowOnScreen)
        monkeypatch.setattr(gui_app, "build_app", lambda **handed: (Loop(), window))
        QThreadPool.globalInstance().start(Run())
        try:
            assert _main(entry) == 0
            assert (during == [held], one_instance.current(), ended.is_set()) == (True, None, False), (
                "the claim was not held while the loop ran and let go as it ended, the run going on")
            again = one_instance.claim(settings.settings().fileName(), tries=1, pause_ms=0)
            try:
                assert again.case == "start", f"the next start was {again.case}"
            finally:
                again.release()
        finally:
            going.set()
            QThreadPool.globalInstance().waitForDone(30000)


def test_main_lets_go_of_the_claim_when_its_loop_ends_without_a_close(tmp_path, monkeypatch):
    """The program's event loop can end with the window never closed -- the application told to quit -- and the
    claim is then the entry point's to let go: it holds the lock and the listener while the loop runs and has let go
    of both as the loop returns, so that the next start of PRISM on its settings file opens a window of its own. The
    entry point claims before it builds the program, and hands the program its application and that claim: in a
    process that holds one, the claim it holds."""
    from core.gui import __main__ as entry
    from core.gui import app as gui_app
    from core.gui import one_instance, settings
    from tests._fixtures import whole_program
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (_app, window):
        held, during, handed = one_instance.current(), [], []
        assert held is not None, "the whole program holds no claim"

        class Loop:
            """The program's event loop, as far as its end goes: it ends at once, the window never closed."""
            def exec(self):
                during.append(one_instance.current())
                return 0

        window.setAttribute(Qt.WA_DontShowOnScreen)
        monkeypatch.setattr(gui_app, "build_app", lambda **told: handed.append(told) or (Loop(), window))
        try:
            assert _main(entry) == 0
            assert handed == [{"app": app, "claimed": held}], f"the entry point handed the program {handed}"
            assert (during == [held], window.isVisible(), one_instance.current()) == (True, True, None), (
                "the claim was not held while the loop ran, the window was closed, or the claim was not let go as "
                "the loop ended")
            again = one_instance.claim(settings.settings().fileName(), tries=1, pause_ms=0)
            try:
                assert again.case == "start", f"the next start was {again.case}"
            finally:
                again.release()
        finally:
            window.hide()


def test_closing_the_window_lets_go_of_the_claim_while_a_figure_window_keeps_prism_running(tmp_path, monkeypatch):
    """A window of its own that the window does not own -- here one with no parent, shown nowhere on the screen,
    which the close leaves open -- keeps the program's event loop running after the window has closed. The lock and
    the listener are let go as the window's close is accepted, not when the loop ends, so that the next start of
    PRISM on its settings file takes the lock and opens a window of its own beside the one left open. A close
    refused while a run is live -- the close's question asked, and "Keep running" its answer -- keeps both, and the
    window. The run that ended inside the whole program is put back as the program goes: no run is left to
    acknowledge, and no close waits for one."""
    from PySide6.QtWidgets import QWidget
    from core.gui import one_instance, settings
    from core.gui.panels.reduction_panel import ReductionPanel
    from tests._fixtures import asked, choose, whole_program
    qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    figure = QWidget()                                # a window of its own the window does not own, with no parent
    figure.setAttribute(Qt.WA_DontShowOnScreen)
    try:
        with whole_program() as (_app, window):
            held = one_instance.current()
            assert held is not None, "the whole program holds no claim"
            _shown_off_screen(window)
            figure.show()
            with running(window.panel(ReductionPanel)):
                choose(monkeypatch, words="Keep running")
                window.close()
                refused = (window.isVisible(), one_instance.current() is held, asked().title, controller().is_live())
            assert refused == (True, True, "Close PRISM?", True), (
                "a close refused while a run is live closed the window, let go of the claim, asked no question, or "
                "stopped the run")
            window.close()
            assert (window.isVisible(), figure.isVisible(), one_instance.current()) == (False, True, None), (
                "the claim was not let go as the window's close was accepted, a figure window still open")
            again = one_instance.claim(settings.settings().fileName(), tries=1, pause_ms=0)
            try:
                assert again.case == "start", f"the next start was {again.case}"
            finally:
                again.release()
        assert (controller().is_live(), controller().status.state, controller().status.closing) == (
            False, "idle", False), "the whole program left the run's end to acknowledge, or a close waiting for one"
    finally:
        figure.hide()
        figure.deleteLater()


def test_a_close_whose_settings_cannot_be_written_still_lets_go_of_the_claim(tmp_path, monkeypatch):
    """The window writes its settings as a close is accepted and lets go of the claim after them. A disk that cannot
    take the settings makes that write raise, and the claim is let go all the same: the window is closing, and a
    claim kept would turn the next start of PRISM away for as long as a figure window left open, or anything else,
    keeps the process alive. How the toolkit's binding reports the error is not asserted."""
    from core.gui import one_instance, settings
    from tests._fixtures import whole_program
    qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (_app, window):
        assert one_instance.current() is not None, "the whole program holds no claim"
        _shown_off_screen(window)

        def full_disk(_window):
            raise OSError("disk full")

        monkeypatch.setattr(MainWindow, "_save_state", full_disk)
        with contextlib.suppress(OSError):             # the binding reports the error itself, or hands it back here
            window.close()
        assert one_instance.current() is None, "a close whose settings could not be written kept the claim"
        again = one_instance.claim(settings.settings().fileName(), tries=1, pause_ms=0)
        try:
            assert again.case == "start", f"the next start was {again.case}"
        finally:
            again.release()


def test_bringing_the_window_forward_brings_the_question_open_over_it_first(monkeypatch):
    """The window brought to the front, as a second start's knock asks: out of the minimised state, a maximised one
    staying maximised; raised and activated; and its taskbar button flashed only when the system keeps the front for
    another window, through the one place that asks Windows for it. While a question is open over the window, it is
    the question that is raised and activated and whose hold of the front is read, so that the keyboard stays on what
    must be answered first. Every call that reaches the system is stood in for and noted: nothing is brought to the
    front here."""
    from PySide6.QtWidgets import QWidget
    from core.gui import main_window, this_computer
    from core.gui.widgets.ask import Ask
    qt_app()
    window = build_window()
    question, did = None, []

    def note(what):
        return lambda widget, *_rest: did.append((what, "question" if widget is question else
                                                  "window" if widget is window else type(widget).__name__))

    monkeypatch.setattr(QWidget, "raise_", note("raise"))
    monkeypatch.setattr(QWidget, "activateWindow", note("activate"))
    monkeypatch.setattr(this_computer, "flash", note("flash"))
    try:
        for asking in (False, True):
            if asking:
                question = Ask(window, "Error", "The cell file could not be read.", "OK", marker="err")
                question.setAttribute(Qt.WA_DontShowOnScreen)
                question.show()
            front = "question" if asking else "window"
            for holds in (False, True):
                monkeypatch.setattr(main_window, "_holds_the_front",
                                    lambda widget, holds=holds: note("front")(widget) or holds)
                window.setWindowState(Qt.WindowMaximized | Qt.WindowMinimized)
                del did[:]
                COME_TO_FRONT(window)
                wanted = [("raise", front), ("activate", front), ("front", front)] + (
                    [] if holds else [("flash", "window")])
                assert (did, window.windowState()) == (wanted, Qt.WindowMaximized), (
                    f"with {'a' if asking else 'no'} question open, the front {'held' if holds else 'refused'}: "
                    f"{did}, {window.windowState()}")
    finally:
        if question is not None:
            question.reject()
            question.deleteLater()


def test_off_windows_the_flash_is_the_toolkits_alert_on_a_real_screen_and_nothing_off_it(monkeypatch):
    """The taskbar's flash, as its module makes it. On Windows it is asked of Windows on the real platform alone, so
    here, off the screen, nothing is asked. Off Windows, on a real screen, it is the toolkit's alert, as it was before
    the flash went through that module: each system draws the eye to the window its own way (a dock's icon, a window
    marked urgent); off the screen there, and with no window, nothing. The other systems and their platforms are
    stood in for, and the alert is noted: nothing flashes here."""
    from PySide6.QtGui import QGuiApplication
    qt_app()
    window, alerted = object(), []                 # the window, as the alert is handed it
    monkeypatch.setattr(QApplication, "alert", lambda widget, *_rest: alerted.append(widget))
    assert (FLASH(window), alerted) == (False, []), "on Windows off the screen the flash asked something"
    for system, platform, flashes in (("linux", "xcb", True), ("linux", "wayland", True), ("darwin", "cocoa", True),
                                      ("linux", "offscreen", False), ("darwin", "minimal", False)):
        monkeypatch.setattr(this_computer, "sys", SimpleNamespace(platform=system))
        monkeypatch.setattr(QGuiApplication, "platformName", staticmethod(lambda named=platform: named))
        del alerted[:]
        assert (FLASH(window), FLASH(None), alerted) == (flashes, False, [window] if flashes else []), (
            system, platform)


def test_no_key_fires_while_a_question_with_no_window_is_open():
    """A question asked with no window under it -- the last-resort error's, before the window is built -- is modal to
    the whole program: while it is open, no key of the map moves the window, pressed at the window or at the
    question, whichever of the two is in front."""
    from core.gui.widgets.ask import Ask
    qt_app()
    window = build_window()
    said = []
    window.place_changed.connect(lambda part, page: said.append((part, page)))
    question = None
    try:
        _shown_off_screen(window)
        window.go_to("live")
        held = window.focusWidget()
        question = Ask(None, "Error", "Something went wrong that PRISM did not expect.", "OK", marker="err")
        question.setAttribute(Qt.WA_DontShowOnScreen)
        question.show()
        del said[:]
        try:
            for front, named in ((question, "the question"), (window, "the window")):
                with _in_front(front):
                    for sequence, _does in keys.BINDINGS:
                        for at in (window, question):
                            _press(at, sequence)
                    assert (window.current_place(), window.focusWidget(), _marked(window.shell.rail), said) == (
                        ("live", None), held, ["live"], []), f"a key moved the window, {named} in front"
        finally:
            question.reject()
    finally:
        window.hide()
        if question is not None:
            question.deleteLater()


def test_a_window_whose_home_cannot_be_built_opens_on_its_error_page(monkeypatch):
    """Home is built with the window, inside the guard every part is built in: when its construction raises, the
    window opens all the same, on Home's error page, with Home's row marked and no part built. The steps' states, a
    change of the store, a run's start and end and every move go on with no Home to read, and
    "Try again", the construction mended, builds Home in its place and shows it."""
    from core.gui import main_window
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.screens.shell import PartError
    qt_app()

    def refused(*_args, **_kwargs):
        raise _CannotBeBuilt("the records could not be read")

    monkeypatch.setattr(main_window, "HomeScreen", refused)
    window = build_window()
    rail = window.shell.rail
    try:
        _shown_off_screen(window, 1300, 820)
        page = window.part("home")
        assert isinstance(page, PartError) and window.shell.stack.currentWidget() is page, type(page).__name__
        assert (window.current_place(), _marked(rail), window.built_parts()) == (("home", None), ["home"], ())
        assert "Home could not be opened. The records could not be read." in _shown_words(page), _shown_words(page)
        window._store_changed()
        with running(window.panel(ReductionPanel)):
            window.go_to("help")
            window.go_to("home")
        window.go_to("home")
        assert window.shell.stack.currentWidget() is page, "Home was built again without being asked"
        monkeypatch.setattr(main_window, "HomeScreen", HomeScreen)
        _button(page, "Try again").click()
        home = window.part("home")
        assert isinstance(home, HomeScreen) and window.shell.stack.currentWidget() is home, "Try again built no Home"
        assert (window.built_parts(), _marked(rail)) == (("home", "reduction", "help"), ["home"]), (
            "Home built again is not counted among the parts built, or its row is not marked")
    finally:
        window.hide()


def test_the_last_resort_error_goes_through_the_shared_question(monkeypatch):
    """An error nothing else caught reaches the hook of last resort, which tells it through the shared question, as
    an error: its one sentence the error's message, written as a sentence as a part's error page writes it, and
    never the error's kind; one answer, "OK"; the kind and the trace behind "Show the full error". The question
    stands over the window once there is one, and with no window under it before, the window read when the error
    comes, not when the hook is installed. An error with no message of its
    own says that something went wrong. Each error is still written where it was before, and one raised while the
    question is open is written there and asks nothing more."""
    import sys
    from core.gui import app as gui_app
    from tests._fixtures import all_asked, choose
    qt_app()
    window = build_window()
    written, parents, real_ask = [], [], gui_app.ask
    monkeypatch.setattr(gui_app, "ask", lambda parent, *words, **more: parents.append(parent) or real_ask(
        parent, *words, **more))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)       # the hook installed here is put back at the end
    monkeypatch.setattr(sys, "__excepthook__", lambda kind, error, trace: written.append(f"{kind.__name__}: {error}"))

    def unhandled(error):
        try:
            raise error
        except Exception as raised:                                 # noqa: BLE001 -- handed to the hook, as uncaught
            sys.excepthook(type(raised), raised, raised.__traceback__)

    held = {}
    gui_app._install_excepthook(lambda: held.get("window"))     # installed before there is a window, as at start
    for parent in (None, window):
        held["window"] = parent
        unhandled(ValueError("sbi_posterior.sample() failed"))
        told = all_asked()[-1]
        assert (told.kind, told.marker, told.title, told.sentence, told.answers) == (
            "error", "err", "Error", "sbi_posterior.sample() failed.", ("OK",)), f"{told.title}: {told.sentence}"
        assert told.detail.startswith("Traceback") and told.detail.rstrip().endswith(
            "ValueError: sbi_posterior.sample() failed"), "the detail is not the kind and the trace"
        assert parents[-1] is parent, "the question does not stand over the window"
    unhandled(ValueError("the cell file could not be read"))
    unhandled(ValueError())
    unhandled(KeyError("model"))
    assert [told.sentence for told in all_asked()[-3:]] == [
        "The cell file could not be read.", "Something went wrong that PRISM did not expect.",
        "Something went wrong that PRISM did not expect: 'model'."]
    asked_before, raised_again = len(all_asked()), []

    def raise_again():                                              # once: while the first question is open
        if not raised_again:
            raised_again.append(True)
            unhandled(RuntimeError("a second error"))

    choose(monkeypatch, during=raise_again)
    unhandled(ValueError("the first error"))
    assert [told.sentence for told in all_asked()[asked_before:]] == ["The first error."], "a second question opened"
    assert written == ["ValueError: sbi_posterior.sample() failed", "ValueError: sbi_posterior.sample() failed",
                       "ValueError: the cell file could not be read", "ValueError: ", "KeyError: 'model'",
                       "RuntimeError: a second error", "ValueError: the first error"], (
        "the errors were not written to the console as before")


def test_the_programs_last_resort_question_stands_over_its_window(tmp_path, monkeypatch):
    """The whole program installs its hook of last resort before its window is built, and the question an error
    nothing else caught then asks stands over the window."""
    import sys
    from core.gui import app as gui_app
    from tests._fixtures import whole_program
    qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    monkeypatch.setattr(sys, "__excepthook__", lambda *_args: None)
    parents, real_ask = [], gui_app.ask
    monkeypatch.setattr(gui_app, "ask", lambda parent, *words, **more: parents.append(parent) or real_ask(
        parent, *words, **more))
    with whole_program() as (_app, window):
        sys.excepthook(ValueError, ValueError("the cell file could not be read"), None)
        assert parents == [window], "the last-resort question does not stand over the program's window"


def test_the_last_resort_question_cuts_a_message_its_screen_cannot_hold(monkeypatch):
    """A message too long for the last-resort question to stand whole on its screen -- here a network's saved
    settings read into a network of another shape, which names every setting it misses -- is cut after its last
    word that leaves the question whole, its frame and its answer in sight, and an ellipsis stands for the rest,
    which the full error holds. On a screen 768 high, less a taskbar, the question then stands whole at the ordinary
    text size; a message the question holds whole is not cut. A message with no space to cut at -- paths, one to a
    line -- is cut inside its one word, after its last letter that leaves the question whole."""
    import sys
    import torch
    from core.gui import app as gui_app
    from core.gui.widgets.ask import Ask
    from tests._fixtures import all_asked
    qt_app()
    try:
        torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(60)]).load_state_dict(
            torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(2)]).state_dict())
    except RuntimeError as raised:
        error = raised
    whole = gui_app.as_sentence(str(error))

    def height(sentence):
        question = Ask(None, "Error", sentence, "OK", marker="err", detail=whole)
        try:
            return question.sizeHint().height()
        finally:
            question.deleteLater()

    paths = "\n".join(f"cells/cell_{number:03}.txt" for number in range(300))    # one to a line, and no space
    screen = QApplication.primaryScreen().availableGeometry().height() - gui_app.FRAME
    for room in (768 - 48 - gui_app.FRAME, 600 - gui_app.FRAME):        # rooms of their own, not the run's screen
        shown = gui_app._fitted(whole, room)
        kept = shown[:-1]
        assert (shown[-1:], whole.startswith(kept + " ")) == ("\u2026", True), f"not cut at a word: {shown[-40:]}"
        more = whole[:whole.index(" ", len(kept) + 1)] + "\u2026"         # a word more
        assert height(shown) <= room < min(height(more), height(whole)), (
            f"in {room} px the question is {height(shown)} px, and {height(more)} px with a word more")
        cut = gui_app._fitted(paths, room)
        more = paths[:len(cut)] + "\u2026"                                # a letter more
        assert (cut[-1:], paths.startswith(cut[:-1]), len(cut) > 1) == ("\u2026", True, True), (
            f"a message with no space was not cut inside its word: {ascii(cut[-40:])}")
        assert height(cut) <= room < height(more), (
            f"in {room} px the question is {height(cut)} px, and {height(more)} px with a letter more")
    short = "The cell file could not be read."
    assert gui_app._fitted(short, screen) == short, "a message the question holds whole was cut"
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)       # the hook installed here is put back at the end
    monkeypatch.setattr(sys, "__excepthook__", lambda *_args: None)
    gui_app._install_excepthook(lambda: None)
    sys.excepthook(type(error), error, error.__traceback__)
    told = all_asked()[-1]
    assert (told.sentence == gui_app._fitted(whole, screen), told.detail.rstrip().endswith(str(error).rstrip())) == (
        True, True), "the question did not cut the message to its screen, or the full error does not hold it whole"


def test_the_class_fonts_are_set_once_at_start(tmp_path, monkeypatch):
    """The fonts of the classes the toolkit keeps a font of their own for are set once as PRISM starts: by the first
    apply of the appearance, after its style sheet, which empties them as the style set before it does. Then they
    are the lettering face at the text size, with the digits of equal width in lists and tables and nowhere else."""
    from PySide6.QtGui import QFont
    from core.gui import design, fonts
    from tests._fixtures import whole_program
    qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    monkeypatch.setattr(design, "system_text_scale", lambda: 1.0)
    sheet_was_set, real = [], fonts.apply_class_fonts

    def watched(app=None):
        sheet_was_set.append(bool(QApplication.instance().styleSheet()))
        return real(app)

    monkeypatch.setattr(fonts, "apply_class_fonts", watched)
    with whole_program():
        assert sheet_was_set == [True], f"the class fonts were set {len(sheet_was_set)} times: {sheet_was_set}"
        tabular = QFont.Tag.fromString("tnum")
        found = {name: (QApplication.font(name).family(), QApplication.font(name).pixelSize(),
                        QApplication.font(name).isFeatureSet(tabular)) for name in ("QAbstractItemView", "QMenu")}
        assert found == {"QAbstractItemView": ("Inter", 14, True), "QMenu": ("Inter", 14, False)}, found


# ── the launcher and the start with no console ─────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _names_kept(app):
    """The application's names as the block found them, put back as it ends: the entry point gives the application it
    starts PRISM's names."""
    names = (app.organizationName(), app.applicationName())
    try:
        yield
    finally:
        app.setOrganizationName(names[0])
        app.setApplicationName(names[1])


def _main(entry):
    """The entry point's ``main()``, run once its plain Windows message is seen to be the suites' stand-in -- a real one
    would wait for a person's OK -- and once the entry module is seen to hold nothing of the window's module, which
    ``main`` imports inside itself after its claim: a ``build_app`` of the module's own would build the whole program
    for real here, the test's stand-in passed by."""
    from tests._dialogs import native_box
    assert entry._native_box is native_box, "the entry point's Windows message is not stood in for: it would wait"
    assert "build_app" not in vars(entry), "the entry module imports the window's module before it claims"
    return entry.main()


def _launch_line(text: str) -> bool:
    """Whether ``text`` is the dated line a launch opens the window's log with, naming this interpreter."""
    import re
    match = re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d PRISM started \((.*)\)\.\n?", text)
    return match is not None and match.group(1) == sys.executable


def test_the_window_starts_and_logs_with_no_console_streams(tmp_path, monkeypatch):
    """Started with no console, as the launcher starts it, PRISM has no standard streams, and the window writes to
    them all the same: a library's warning between runs, a record another thread logs during a run, an error nothing
    else caught. Bound to the window's log, each of those is written there, after the launch's dated line naming the
    interpreter, and nothing is raised; a run's own lines, what it prints among them, go to the run's dated file and
    never to the log, and the other thread's record goes to the log and never to the run's file. Put back, the
    streams, the fault handler and the toolkit's message handler are as they were found."""
    import logging
    import threading
    from core.gui import __main__ as entry
    from core.gui.panels.reduction_panel import ReductionPanel
    from tests._fixtures import all_asked, process_state, whole_program
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    log = tmp_path / "window-log.txt"
    found = process_state(app)
    entry.bind_streams(log)
    try:
        bound = process_state(app)
        with whole_program() as (_app, window):
            logging.getLogger("sbi.inference").warning("Only 0.5% proposal samples are accepted.")
            text, stand_in = RunText(controller()), StandIn()
            with running(window.panel(ReductionPanel), stand_in):
                stand_in.say("the run prints this line", "print")
                turn_until(app, lambda: "the run prints this line" in text.texts())
                other = threading.Thread(target=lambda: logging.getLogger("core.gui.main_window").warning(
                    "A part of the window could not be opened."))
                other.start()
                other.join()
            run = controller().status.run
            try:
                raise ValueError("the cell file could not be read")
            except ValueError as raised:
                sys.excepthook(type(raised), raised, raised.__traceback__)
            told = all_asked()[-1]
    finally:
        entry.unbind_streams()
    after = process_state(app)
    keys = ("standard streams", "fault handler", "Qt message handler")
    assert {"standard streams", "Qt message handler"} <= {key for key in keys if bound[key] != found[key]}, (
        "the streams or the toolkit's messages were not bound")
    assert [key for key in keys if after[key] != found[key]] == [], "the streams were not put back as found"
    written = log.read_text(encoding="utf-8")
    ran = (tmp_path / "runs" / f"{run}.txt").read_text(encoding="utf-8")
    assert _launch_line(written.splitlines(keepends=True)[0]), written[:200]
    assert ("library: sbi.inference: Only 0.5% proposal samples are accepted.\n" in written,
            "warning: A part of the window could not be opened.\n" in written,
            "Traceback (most recent call last):" in written,
            "ValueError: the cell file could not be read\n" in written,
            "the run prints this line" in written) == (True, True, True, True, False), written
    assert ("the run prints this line" in ran, "A part of the window could not be opened." in ran) == (True, False), ran
    assert told.beneath == "The full error is in the window's log file.", told


def test_the_window_log_sits_beside_the_settings_file_or_in_the_temporary_folder(tmp_path, monkeypatch):
    """The window's log is ``window-log.txt`` in the folder of the settings file, here the test's own, made with that
    folder when neither is there yet, as on a computer where PRISM never ran. Where that folder cannot be made -- a
    file stands where it would be -- or the log cannot be opened there -- a folder stands in its place -- the log is
    in the system's temporary folder, which is not written until the log is opened."""
    import tempfile
    from core.gui import __main__ as entry
    from core.gui import settings
    beside = tmp_path / "window-log.txt"
    assert (entry.window_log_path(), beside.is_file()) == (beside, True)
    fresh = tmp_path / "never" / "ran"
    monkeypatch.setattr(settings, "_override_path", str(fresh / "prism.ini"))
    assert (entry.window_log_path(), (fresh / "window-log.txt").is_file()) == (fresh / "window-log.txt", True)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    (tmp_path / "a-file").write_text("", encoding="utf-8")
    (tmp_path / "taken" / "window-log.txt").mkdir(parents=True)
    for settings_file in (tmp_path / "a-file" / "prism.ini", tmp_path / "taken" / "prism.ini"):
        monkeypatch.setattr(settings, "_override_path", str(settings_file))
        assert entry.window_log_path() == temporary / "window-log.txt", settings_file
    assert list(temporary.iterdir()) == [], "the temporary folder was written before the log was opened"


def test_streams_are_bound_only_with_no_console_or_when_prism_runs_detached(tmp_path, monkeypatch):
    """The entry point binds the standard streams to the window's log when the process has no output stream or no
    error stream, as under ``pythonw``, and when ``PRISM_DETACHED`` says it was started detached from its terminal, as
    ``run.sh`` starts it; started from a console, it leaves them as they are. Bound or not, the start goes on: here it
    is handed over to an open PRISM and ends with the status 0."""
    from core.gui import __main__ as entry
    from core.gui import one_instance
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    bound = []
    monkeypatch.setattr(entry, "bind_streams", bound.append)
    monkeypatch.setattr(one_instance, "claim", lambda settings_path, **told: one_instance.Claim("hand_over"))
    out, err = sys.stdout, sys.stderr
    got = []
    for streams, detached in (((None, err), ""), ((out, None), ""), ((out, err), "1"), ((out, err), "")):
        monkeypatch.setattr(sys, "stdout", streams[0])
        monkeypatch.setattr(sys, "stderr", streams[1])
        monkeypatch.setenv("PRISM_DETACHED", detached)
        del bound[:]
        with _names_kept(app):
            got.append((_main(entry), list(bound)))
    log = tmp_path / "window-log.txt"
    assert got == [(0, [log]), (0, [log]), (0, [log]), (0, [])], got


def test_qts_own_messages_and_a_crash_of_the_interpreter_go_to_the_window_log(tmp_path):
    """With no console the toolkit's own messages and a crash of the interpreter itself would leave no trace, so the
    binding writes both to the window's log. The toolkit's messages, with their kind, through a message handler that
    is put back as the streams are, after which they go where they went before. A crash, as the fault handler dumps
    it: here in a process of its own, which binds, puts back, binds again and crashes, the fault handler taken by the
    binding only while nothing held it and let go with it. In this process the suites' runner holds the fault handler,
    and keeps it."""
    import os
    import subprocess
    from pathlib import Path
    from PySide6.QtCore import qInstallMessageHandler, qWarning
    from core.gui import __main__ as entry
    from tests._fixtures import process_state
    app = qt_app()
    log, heard = tmp_path / "window-log.txt", []
    previous = qInstallMessageHandler(lambda kind, context, text: heard.append(text))     # where they went before
    try:
        found = process_state(app)
        entry.bind_streams(log)
        try:
            during = process_state(app)
            qWarning("the toolkit could not load a font")
        finally:
            entry.unbind_streams()
        qWarning("the toolkit says this after the streams were put back")
        after = process_state(app)
    finally:
        qInstallMessageHandler(previous)
    written = log.read_text(encoding="utf-8")
    assert ("Qt warning: the toolkit could not load a font\n" in written, "put back" in written,
            "the toolkit could not load a font" in heard,
            "the toolkit says this after the streams were put back" in heard) == (True, False, False, True), (
        written, heard)
    keys = ("standard streams", "fault handler", "Qt message handler")
    assert (during["Qt message handler"] is not None, during["fault handler"],
            [key for key in keys if after[key] != found[key]]) == (True, True, []), (
        "the toolkit's handler was not installed and put back, or the fault handler was lost")
    crash = tmp_path / "crash" / "window-log.txt"
    crash.parent.mkdir()
    code = ("import faulthandler, sys\n"
            "sys.path.insert(0, sys.argv[2])\n"
            "from core.gui import __main__ as entry\n"
            "assert not faulthandler.is_enabled(), 'the fault handler was held before the binding'\n"
            "entry.bind_streams(sys.argv[1])\n"
            "assert faulthandler.is_enabled(), 'the binding did not take the fault handler'\n"
            "entry.unbind_streams()\n"
            "assert not faulthandler.is_enabled(), 'the fault handler was kept after the streams were put back'\n"
            "entry.bind_streams(sys.argv[1])\n"
            "print('bound, and about to crash')\n"
            "faulthandler._sigsegv()\n")
    repo = str(Path(__file__).resolve().parents[1])
    env = {name: value for name, value in os.environ.items() if name.upper() != "PYTHONFAULTHANDLER"}
    child = subprocess.run([sys.executable, "-c", code, str(crash), repo], cwd=repo, env=env, capture_output=True,
                           text=True, timeout=120)
    dumped = crash.read_text(encoding="utf-8")
    assert (child.returncode not in (0, 1), child.stdout, child.stderr) == (True, "", ""), (child.returncode,
                                                                                            child.stdout, child.stderr)
    assert (_launch_line(dumped.splitlines(keepends=True)[0]), "bound, and about to crash\n" in dumped,
            "Fatal Python error: Segmentation fault" in dumped, 'File "<string>", line 11' in dumped) == (
        True, True, True, True), dumped


def test_a_failure_before_the_window_is_shown_is_a_plain_windows_message_naming_the_log(tmp_path, monkeypatch, capsys):
    """A start of PRISM that fails before its window is shown -- as its window is built, or before its application
    exists -- writes the failure with its trace to the window's log and says so in a plain Windows message naming
    that file, not in a window of the toolkit's, which may be what failed; it lets go of the claim it made and ends
    with the status 1. Started with no console the trace is in the log the streams are bound to; from a console, in
    the log beside the settings file, and on the console too. The message is shown by one function alone, which no
    test calls: the suites stand in for it."""
    import ast
    import matplotlib
    from pathlib import Path
    from core.gui import __main__ as entry
    from core.gui import app as gui_app
    from core.gui import one_instance, settings
    from tests._dialogs import native_boxes
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    log = tmp_path / "window-log.txt"
    said = f"PRISM could not start. The details are in {log}."

    def broken(**handed):
        raise RuntimeError("the window could not be built")

    def refused(backend):
        raise ImportError("the figure library has no backend here")

    monkeypatch.setattr(gui_app, "build_app", broken)
    out, err = sys.stdout, sys.stderr
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    with _names_kept(app):
        status = _main(entry)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    written = log.read_text(encoding="utf-8")
    assert (status, native_boxes(), one_instance.current(), (tmp_path / "PRISM.lock").exists()) == (
        1, [said], None, False), "the start that failed said it otherwise, or kept its claim"
    assert ("PRISM could not start.\nTraceback (most recent call last):" in written,
            "RuntimeError: the window could not be built\n" in written) == (True, True), written
    again = one_instance.claim(settings.settings().fileName(), tries=1, pause_ms=0)
    try:
        assert again.case == "start", f"the next start was {again.case}"
    finally:
        again.release()
    capsys.readouterr()
    monkeypatch.setattr(matplotlib, "use", refused)
    with _names_kept(app):
        status = _main(entry)
    console = capsys.readouterr().err
    assert (status, native_boxes(), one_instance.current()) == (1, [said], None), "the failure before the application"
    assert ("ImportError: the figure library has no backend here" in log.read_text(encoding="utf-8")[len(written):],
            "ImportError: the figure library has no backend here" in console) == (True, True), console
    tree = ast.parse(Path(entry.__file__).read_text(encoding="utf-8"))
    shown = [function.name for function in ast.walk(tree) if isinstance(function, ast.FunctionDef)
             for node in ast.walk(function) if isinstance(node, ast.Attribute) and node.attr == "MessageBoxW"]
    every = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "MessageBoxW"]
    assert (shown, len(every)) == (["_native_box"], 1), "a Windows message is shown somewhere else than _native_box"


def test_the_claim_is_made_before_the_heavy_imports_and_once_per_process(tmp_path, monkeypatch, brought_forward):
    """A second start of PRISM only knocks on the open one, so it must not first pay for the window's imports, which
    load torch and the engine for seconds. The entry module sets the variable torch and MKL need as its first
    statement and imports nothing but ``os`` and ``sys`` (read off its syntax tree, and in a fresh interpreter, where
    importing it loads neither torch, the figure library, the toolkit nor the window); its ``main`` claims before it
    imports the window's module, and imports nothing heavy before the claim; run as a module it hands over to itself
    imported under its own name. A process claims once: a second claim is handed back the claim it holds, trying
    nothing, where it would otherwise knock on its own server and hand over to itself; let go, the next claim is a
    new one. The program handed a claim claims no more, and its window answers that claim's knock."""
    import ast
    import subprocess
    from pathlib import Path
    from core.gui import __main__ as entry
    from core.gui import app as gui_app
    from core.gui import one_instance, settings
    from tests._fixtures import as_another_process, pump, whole_program
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    light = {"faulthandler", "time", "tempfile", "traceback", "pathlib", "ctypes", "matplotlib", "PySide6.QtCore",
             "PySide6.QtWidgets", "core.gui.app_icon", "core.gui.one_instance", "core.gui.settings"}

    def imported(node):
        """The modules an import names, a module of ``core.gui`` named for each name it imports from there."""
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        if node.module == "core.gui":
            return [f"core.gui.{alias.name}" for alias in node.names]
        return [node.module]

    tree = ast.parse(Path(entry.__file__).read_text(encoding="utf-8"))
    body = tree.body[1:] if ast.get_docstring(tree) is not None else tree.body
    leading = [node for node in body[:2] if isinstance(node, (ast.Import, ast.ImportFrom))]
    guard = body[-1]
    opening = "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')"
    assert (sorted(name for node in leading for name in imported(node)), ast.unparse(body[len(leading)])) == (
        ["os", "sys"], opening), "the entry module does not import os and sys alone and then set the variable first"
    assert ast.unparse(guard) == ("if __name__ == '__main__':\n    from core.gui.__main__ import main as _main\n"
                                  "    raise SystemExit(_main())"), ast.unparse(guard)
    at_module = [node for node in body[:-1] if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert at_module == leading, "the entry module imports more than os and sys at its module level"
    functions = {node.name: node for node in body if isinstance(node, ast.FunctionDef)}
    heavy = sorted({(name, module) for name, function in functions.items() for node in ast.walk(function)
                    if isinstance(node, (ast.Import, ast.ImportFrom)) for module in imported(node)
                    if module not in light and (name, module) != ("main", "core.gui.app")})
    assert heavy == [], f"the entry module imports what a start before its claim must not pay for: {heavy}"
    main = functions["main"]
    claim_line = min(node.lineno for node in ast.walk(main)
                     if isinstance(node, ast.Call) and ast.unparse(node.func) == "one_instance.claim")
    window_line = min(node.lineno for node in ast.walk(main) if isinstance(node, (ast.Import, ast.ImportFrom))
                      and "core.gui.app" in imported(node))
    returns = [node.lineno for node in ast.walk(main) if isinstance(node, ast.Return) and node.lineno < window_line]
    assert (claim_line < window_line, bool(returns) and all(claim_line < line for line in returns)) == (True, True), (
        "the entry point imports the window before it claims, or a start handed over returns before its claim")
    code = ("import sys\nsys.path.insert(0, sys.argv[1])\nimport core.gui.__main__\n"
            "print(sorted(name for name in ('torch', 'matplotlib', 'PySide6', 'core.gui.app') if name in sys.modules))")
    repo = str(Path(__file__).resolve().parents[1])
    fresh = subprocess.run([sys.executable, "-c", code, repo], cwd=repo, capture_output=True, text=True, timeout=120)
    assert (fresh.returncode, fresh.stdout.strip()) == (0, "[]"), (fresh.stdout, fresh.stderr)

    path = settings.settings().fileName()
    first = one_instance.claim(path)
    try:
        second = one_instance.claim(path, tries=1, pause_ms=0)
        assert (first.case, second is first, one_instance.current() is first, brought_forward) == (
            "start", True, True, []), "a second claim in the process was not handed the claim it holds"
    finally:
        first.release()
    third = one_instance.claim(path, tries=1, pause_ms=0)
    try:
        assert (third.case, third is not first, one_instance.current() is third) == ("start", True, True)
    finally:
        third.release()

    held, claims, claim, real = one_instance.claim(path), [], one_instance.claim, gui_app.build_app
    monkeypatch.setattr(gui_app, "build_app", lambda argv=None: real(argv, app=qt_app(), claimed=held))
    monkeypatch.setattr(one_instance, "claim", lambda *args, **kwargs: claims.append(args) or claim(*args, **kwargs))
    with whole_program() as (_app, window):
        assert (window is not None, claims, one_instance.current() is held) == (True, [], True), (
            "the program handed a claim built no window, claimed again, or holds another claim")
        with as_another_process():
            second = claim(path, tries=1, pause_ms=0)
        pump(app, 0.2)
        assert (second.case, brought_forward[-1:]) == ("hand_over", [("come_to_front", window)]), (
            "the window does not answer the knock on the claim it was handed")


def test_main_hands_over_to_an_open_prism_and_builds_nothing(tmp_path, monkeypatch, brought_forward):
    """The entry point started beside an open PRISM -- one that holds the lock beside the same settings file and
    listens -- is handed over and ends with the status 0, having built nothing: no widget is made, the program is
    not built, no event loop runs, and nothing of the process is as it set it once it has ended; the open PRISM's
    process is let take the front first, and hears the knock when its loop runs."""
    import os
    from core.gui import __main__ as entry
    from core.gui import app as gui_app
    from core.gui import one_instance, settings
    from tests._fixtures import as_another_process, process_state, pump
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    built, ran, heard = [], [], []
    monkeypatch.setattr(gui_app, "build_app", lambda **handed: built.append(handed))
    monkeypatch.setattr(QApplication, "exec", lambda application: ran.append("a loop") or 0)
    held = one_instance.claim(settings.settings().fileName())            # the open PRISM
    held.knocked.connect(lambda: heard.append(True))
    try:
        widgets, found = len(QApplication.topLevelWidgets()), process_state(app)
        with _names_kept(app), as_another_process():
            status = _main(entry)
        left = sorted(key for key, value in process_state(app).items() if value != found[key])
        assert (status, built, ran, len(QApplication.topLevelWidgets()), left) == (0, [], [], widgets, []), (
            "the start that was handed over built something, ran a loop or left the process changed")
        assert brought_forward == [("allow_foreground", os.getpid())], "the open PRISM was not let take the front"
        pump(app, 0.2)
        assert heard == [True], "the open PRISM did not hear the knock"
    finally:
        held.release()


@pytest.mark.skipif(sys.platform != "win32", reason="a junction is Windows' link to a folder")
def test_a_settings_path_written_through_a_junction_names_the_same_server(tmp_path, brought_forward):
    """A settings folder reached through a junction -- a link to a folder, which Windows makes for anyone, with no
    administrator -- is the same folder: its settings file names one server whichever way its path is written, as it
    names one lock file, so a second start through the other spelling is handed over to the open PRISM, never turned
    away after knocking where nobody listens."""
    import os
    import subprocess
    from core.gui import one_instance
    from tests._fixtures import as_another_process
    qt_app()
    real, joined = tmp_path / "real", tmp_path / "joined"
    real.mkdir()
    made = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(joined), str(real)], capture_output=True,
                          text=True, timeout=60)
    assert made.returncode == 0, made.stdout + made.stderr
    try:
        names = {one_instance.server_name(str(folder / "prism.ini")) for folder in (real, joined)}
        assert len(names) == 1, "one settings file, written through a junction, names two servers"
        first = one_instance.claim(str(real / "prism.ini"))
        try:
            with as_another_process():
                second = one_instance.claim(str(joined / "prism.ini"), tries=1, pause_ms=0)
            assert (first.case, second.case) == ("start", "hand_over"), (first.case, second.case)
        finally:
            first.release()
    finally:
        os.rmdir(joined)                      # the junction alone: the folder it names stays


def test_the_last_resort_error_says_where_the_full_error_is(tmp_path, monkeypatch):
    """While the standard streams are bound to the window's log, the question of last resort says beneath its
    sentence that the full error is in the window's log file, and the error's trace is written there -- not where the
    console's hook writes it, which during a run is the run's own text -- and a message too long to stand whole is cut
    to the room its question has with that line beneath it. Not bound, nothing stands beneath the sentence, and the
    error is written where it was before."""
    from core.gui import __main__ as entry
    from core.gui import app as gui_app
    from core.gui.widgets.ask import Ask
    from tests._fixtures import all_asked
    qt_app()
    written = []
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)       # the hook installed here is put back at the end
    monkeypatch.setattr(sys, "__excepthook__", lambda kind, error, trace: written.append(f"{kind.__name__}: {error}"))
    gui_app._install_excepthook(lambda: None)

    def unhandled(error):
        try:
            raise error
        except Exception as raised:                                 # noqa: BLE001 -- handed to the hook, as uncaught
            sys.excepthook(type(raised), raised, raised.__traceback__)

    unhandled(ValueError("the cell file could not be read"))
    unbound = all_asked()[-1]
    log = tmp_path / "window-log.txt"
    room = 600 - gui_app.FRAME
    whole = " ".join(["the cell file could not be read"] * 400)
    entry.bind_streams(log)
    try:
        unhandled(ValueError("the model file could not be read"))
        bound = all_asked()[-1]
        shown = gui_app._fitted(whole, room, gui_app.FULL_ERROR_IN_LOG)
        question = Ask(None, "Error", shown, "OK", marker="err", beneath=gui_app.FULL_ERROR_IN_LOG, detail=whole)
        try:
            height = question.sizeHint().height()
        finally:
            question.deleteLater()
    finally:
        entry.unbind_streams()
    assert ((unbound.sentence, unbound.beneath), (bound.sentence, bound.beneath)) == (
        ("The cell file could not be read.", ""),
        ("The model file could not be read.", "The full error is in the window's log file.")), (unbound, bound)
    assert written == ["ValueError: the cell file could not be read"], f"written by the console's hook: {written}"
    text = log.read_text(encoding="utf-8")
    assert ("Traceback (most recent call last):" in text, "ValueError: the model file could not be read\n" in text) == (
        True, True), text
    assert (shown.endswith("\u2026"), height <= room) == (True, True), (
        f"the question with its line beneath is {height} px in a room of {room} px")


# The folders the launcher reads and writes, each pointed into a test's own folder: the user's, the local and the
# shared application data, the roaming one that holds PRISM's settings folder and its shortcut's mark, and the folder
# the shortcut goes in.
_LAUNCHER_FOLDERS = ("USERPROFILE", "LOCALAPPDATA", "PROGRAMDATA", "APPDATA", "PRISM_DESKTOP")


def _windows_desktop():
    """Windows' own desktop folder, as the shell names it: listed, never written."""
    import ctypes
    from pathlib import Path
    named = ctypes.create_unicode_buffer(1024)
    assert ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, named) == 0, "the desktop folder has no name"
    return Path(named.value)


def _launcher(tmp_path, option, *, conda_prefix=None, **folders):
    """``run.bat``, copied byte for byte into ``tmp_path``, run there with ``option``: ``--where`` or
    ``--make-shortcut``, and nothing else, since without one it starts PRISM; nor is a launcher run that does not
    know the option, for the same reason. Every folder it reads or writes
    (``_LAUNCHER_FOLDERS``, given here by their names in small letters) must be a folder of ``tmp_path``, and the
    run is refused before it starts when one is not, so that nothing can reach the person's own desktop or settings;
    the active environment is ``conda_prefix``, or none. Windows' own desktop is listed before and after, and must
    hold the same names. Handed back: the status the launcher ended with, the lines it printed, and the folder of
    the copy."""
    import os
    import shutil
    import subprocess
    from pathlib import Path
    assert option in ("--where", "--make-shortcut"), f"run.bat is never run here with {option!r}: it would start PRISM"
    copy = tmp_path / "launcher" / "run.bat"
    copy.parent.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "run.bat", copy)
    assert f'if /i "%~1"=="{option}"' in copy.read_text(encoding="utf-8"), (
        f"run.bat does not know {option}, so it would start PRISM: it is not run")
    env = {name.upper(): value for name, value in os.environ.items()
           if name.upper() not in _LAUNCHER_FOLDERS + ("CONDA_PREFIX",)}
    env.update({name.upper(): str(value) for name, value in folders.items()})
    if conda_prefix is not None:
        env["CONDA_PREFIX"] = str(conda_prefix)
    own = tmp_path.resolve()
    for name in _LAUNCHER_FOLDERS:
        assert env.get(name) and Path(env[name]).resolve().is_relative_to(own), (
            f"{name} is not a folder of the test's own, so the launcher would reach the person's own")
    assert " " not in str(copy), f"the copy's path holds a space, which the command line would split: {copy}"
    desktop = _windows_desktop()
    before = set(os.listdir(desktop))
    try:
        ran = subprocess.run(["cmd.exe", "/d", "/c", str(copy), option], cwd=str(tmp_path), env=env,
                             capture_output=True, text=True, timeout=120)
    finally:
        added = set(os.listdir(desktop)) - before
        assert not added, f"the launcher wrote on Windows' own desktop: {sorted(added)}"
    return ran.returncode, ran.stdout.splitlines(), copy.parent


def _environment(folder, *libraries):
    """A conda environment's folder as the launcher reads it: its ``pythonw.exe``, an empty file nothing runs, and a
    folder for each library named. Handed back: the interpreter's path."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pythonw.exe").write_bytes(b"")
    for library in libraries:
        (folder / "Lib" / "site-packages" / library).mkdir(parents=True)
    return folder / "pythonw.exe"


@pytest.mark.skipif(sys.platform != "win32", reason="run.bat is the launcher for Windows")
def test_the_launcher_finds_the_environment_or_says_what_to_do(tmp_path):
    """Double-clicked, the launcher must start PRISM with PRISM's own Python, which on this computer is not the first
    on the path, and with no console it must never fail in silence. Read through its ``--where``, on a copy of it whose
    every folder is the test's own: the active environment's interpreter first, when that environment's folder holds
    PySide6 and torch; an active environment that holds neither -- a prompt left in conda's base environment -- or
    only one, passed over for ``biophys-env`` under the user's ``anaconda3``; then the other usual places in their
    order; with none found, the words that say what to do and the status 1. Whether the desktop shortcut would be made
    is said from its mark, with where the shortcut and its mark would go. Nothing is started -- the one line that
    starts PRISM, and the Windows message, belong to a start with no option -- and nothing is written."""
    from pathlib import Path
    root = tmp_path / "computer"
    user, local, shared, roaming, desktop = (root / name for name in ("user", "local", "shared", "roaming", "desktop"))
    for folder in (local, shared, roaming, desktop):
        folder.mkdir(parents=True)
    folders = {"userprofile": user, "localappdata": local, "programdata": shared, "appdata": roaming,
               "prism_desktop": desktop}
    active = _environment(root / "active", "PySide6", "torch")
    base = user / "anaconda3"
    _environment(base, "numpy")
    half = _environment(root / "half", "PySide6")
    places = [_environment(folder / "envs" / "biophys-env") for folder in (
        user / "anaconda3", user / "miniconda3", local / "anaconda3", shared / "anaconda3", shared / "miniconda3")]

    def listing():
        return sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if "launcher" not in path.parts)

    def where(conda_prefix=None):
        status, lines, _ = _launcher(tmp_path, "--where", conda_prefix=conda_prefix, **folders)
        return status, lines

    mark = roaming / "PRISM" / "shortcut-made"

    def found(python, made=f"The desktop shortcut would be made now: \"{desktop / 'PRISM.lnk'}\", its mark \"{mark}\"."):
        return 0, [f"PRISM's Python: \"{python}\"", made]

    before = listing()
    got = [where(conda_prefix=active.parent), where(conda_prefix=base), where(conda_prefix=half.parent), where()]
    assert got == [found(active), found(places[0]), found(places[0]), found(places[0])], got
    assert listing() == before, "the launcher's --where wrote something"
    walked = []
    for python in places:
        walked.append(where())
        python.unlink()
    walked.append(where())
    assert walked == [found(python) for python in places] + [(1, [
        "PRISM's Python was not found. Open the Anaconda Prompt, type: conda activate biophys-env, then run run.bat "
        "from there."])], walked
    _environment(user / "miniconda3" / "envs" / "biophys-env")
    mark.parent.mkdir()
    mark.write_text("", encoding="utf-8")
    assert where() == found(places[1], f"The desktop shortcut was made once already, and is not made again: its mark "
                                       f"is \"{mark}\".")
    lines = [line.strip() for line in (Path(__file__).resolve().parents[1] / "run.bat").read_text(
        encoding="utf-8").splitlines()]
    starts = [line for line in lines if 'start "PRISM"' in line or ".Popup(" in line]
    popup = lines.index(starts[0])
    assert (starts[1:], lines[popup - 1], lines[popup - 2]) == (
        ['if "%LAUNCH_MODE%"=="start" start "PRISM" "%LAUNCH_PYTHONW%" -m core.gui'],
        'if "%LAUNCH_MODE%"=="start" (', "if not defined LAUNCH_PYTHONW ("), (
        "the launcher starts PRISM, or shows its Windows message, otherwise than in a start with no option", starts)


@pytest.mark.skipif(sys.platform != "win32", reason="run.bat is the launcher for Windows")
def test_the_launcher_makes_the_desktop_shortcut_once(tmp_path):
    """The launcher puts a PRISM shortcut on the desktop once: the environment's ``pythonw.exe`` with ``-m core.gui``,
    started in the repository's folder, with PRISM's icon -- each read back through Windows' own shell -- and a mark
    beside PRISM's settings file that says it was made. Run again after the shortcut was deleted, it makes none: made
    once is made. Every folder is the test's own, ``PRISM_DESKTOP`` naming the desktop; Windows' own desktop is
    listed before and after each run, and holds the same names."""
    import os
    import subprocess
    root = tmp_path / "computer"
    user, local, shared, roaming, desktop = (root / name for name in ("user", "local", "shared", "roaming", "desktop"))
    for folder in (local, shared, roaming, desktop):
        folder.mkdir(parents=True)
    folders = {"userprofile": user, "localappdata": local, "programdata": shared, "appdata": roaming,
               "prism_desktop": desktop}
    python = _environment(user / "anaconda3" / "envs" / "biophys-env")
    link, mark = desktop / "PRISM.lnk", roaming / "PRISM" / "shortcut-made"
    status, lines, repo = _launcher(tmp_path, "--make-shortcut", **folders)
    assert (status, lines, link.is_file(), mark.is_file()) == (
        0, [f"The desktop shortcut was made: {link}"], True, True), (status, lines)
    read = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                           "$link = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LAUNCH_LINK); "
                           "$link.TargetPath; $link.Arguments; $link.WorkingDirectory; $link.IconLocation"],
                          env={**os.environ, "LAUNCH_LINK": str(link)}, capture_output=True, text=True, timeout=120)
    assert read.stdout.splitlines() == [str(python), "-m core.gui", str(repo),
                                        f"{repo}\\core\\gui\\assets\\app\\prism.ico,0"], (read.stdout, read.stderr)
    assert sorted(path.name for path in desktop.iterdir()) == ["PRISM.lnk"], "the shortcut is not alone on the desktop"
    link.unlink()
    status, lines, _ = _launcher(tmp_path, "--make-shortcut", **folders)
    assert (status, lines, link.exists(), mark.is_file()) == (
        0, ["The desktop shortcut was made once already, and is not made again."], False, True), (status, lines)


def test_the_launcher_for_macos_and_linux_starts_prism_detached():
    """On macOS and Linux ``run.sh`` starts PRISM detached from the terminal that ran it, which may then be closed: it
    sets the variable torch and MKL need, says that PRISM runs detached (so that the window binds its streams to its
    log, since what it writes reaches no terminal), and starts the window under ``nohup``, its output thrown away, in
    the background, as the script's last line; it no longer replaces itself with the window. Read off the file,
    which no test runs."""
    from pathlib import Path
    lines = [line.strip() for line in (Path(__file__).resolve().parents[1] / "run.sh").read_text(
        encoding="utf-8").splitlines()]
    commands = [line for line in lines if line and not line.startswith("#")]
    wanted = ["export KMP_DUPLICATE_LIB_OK=TRUE", "export PRISM_DETACHED=1",
              "nohup python3 -m core.gui >/dev/null 2>&1 &"]
    assert commands[-3:] == wanted, commands
    assert [line for line in commands if "core.gui" in line or line.startswith("exec ")] == [wanted[-1]], commands
