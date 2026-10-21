"""The window's places -- the rail, ``go_to`` and the run's foot -- and Home.

Home is built as the window builds it, through the builders of tests/_fixtures.py, and handed what it states by a
function each test makes: a session is a plain namespace of the five fields the window's session holds, and the
steps' states are the rules' own. The window's reading of its session, its run and the record store in use, and
what each of Home's places opens in the window, are tested on a whole window.
"""
import contextlib
import time
import warnings
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeySequence, QTextLayout
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from core.gui import chain, design, keys, record_words
from core.gui.main_window import MainWindow
from core.gui.screens.home_screen import HomeFacts, HomeScreen, RecordRow
from core.gui.widgets.page import Columns, PageHead, Sentence
from core.gui.widgets.rule_table import StateCell, TextCell
from tests._fixtures import build_part, build_run_host, build_window, qt_app

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


def _session(**held):
    """A session holding what is named; every other field holds nothing."""
    fields = dict(draft=None, cfg=None, inf_prior=None, posterior=None, observation=None)
    fields.update(held)
    return SimpleNamespace(**fields)


def _under_way():
    """A model chosen and a prior in use: the training is the step in hand."""
    return _session(draft=SimpleNamespace(model="NADROWSKI"), inf_prior=SimpleNamespace(name="retrain_prior"))


def _facts(session=None, records=(), live_place=None, has_runs=False):
    """What Home is handed: the steps' states worked out from the session as the rules work them out."""
    session = _session() if session is None else session
    live = live_place[1] if live_place is not None and live_place[0] == "inference" else None
    return HomeFacts(session, chain.step_states(session, live=live), live_place, tuple(records), has_runs)


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
    home = _home(_facts(_under_way(), live_place=("temperature", "sweep")))
    assert home.next_heading.label.text() == "Next: nothing until the run ends"
    sentences = [label.plain_text() for label in home.findChildren(Sentence) if _shown(home, label)]
    assert ("A run is going in Effective temperature, Sweep study, and no other run can start until it ends. You can "
            "still move around and read results. Keep PRISM open; you can minimise it.") in sentences
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
                              (_facts(_under_way(), live_place=("live", None)), "&Go to the run", "Alt+G")):
        home = _home(facts)
        shown = [button for button in home.findChildren(QPushButton) if _shown(home, button)]
        main = [button for button in shown if button.property("cls") == "primary"]
        assert [button.text() for button in main] == [words] and home.first_focus() is main[0]
        assert QKeySequence.mnemonic(main[0].text()).toString() == key
        assert [button.text() for button in shown if "&" in button.text()] == [words]


def test_the_window_reads_for_home_its_session_its_run_and_every_record_of_the_store_in_use(tmp_path):
    """What the window hands Home: the inference screen's session and its six steps, once the screen is built, the
    part and page a live run is in (the step whose run it is then reads "Now"), and every record of the store in
    use that has a manifest,
    of every kind, each with its label, the word for its kind and the moment it was made, in local time. A
    simulation set is called by the prior it was simulated for, or said to have lost it. A folder with no manifest
    is no record. A store with no record and nothing chosen opens Home on its first opening."""
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
        assert facts.session is inference.session and facts.live_place is None
        assert facts.states == chain.step_states(inference.session) and not facts.has_runs
        rows = {(row.kind, row.id): row for row in facts.records}
        wanted = {(kind, ids[kind]) for kind in KIND_DIRS} | {("simulation", held.id)}
        assert set(rows) == wanted and len(facts.records) == len(wanted)
        for kind in KIND_DIRS:
            summary, = [one for one in store.list(kind) if one.id == ids[kind]]
            made = datetime.fromisoformat(summary.created).astimezone()
            row = rows[(kind, ids[kind])]
            label = "Simulation set for a prior that is no longer here" if kind == "simulation" else f"browse_{kind}"
            assert (row.label, row.kind_word, row.made, row.made.utcoffset()) == (
                label, record_words.kind_word(kind, summary.study), made, made.utcoffset())
        assert rows[("simulation", held.id)].label == "Simulation set for browse_prior"
        assert rows[("fdt", ids["fdt"])].kind_word == "sweep"
        assert not window.part("home").is_first_time()
        panel = window.panel(CrossValPanel)
        panel._set_busy(True)
        try:
            assert window._home_facts().live_place == ("temperature", "sweep")
        finally:
            panel._set_busy(False)
        posterior = window.part("inference").posterior_panel
        posterior._set_busy(True)
        try:
            live = window._home_facts()
            assert live.live_place == ("inference", "training")
            assert (live.states[2].marker, live.states[2].word, live.states[2].detail) == ("now", "Now", "Running")
        finally:
            posterior._set_busy(False)
        assert window._home_facts().live_place is None


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
    and while it is shown when the steps' states are worked out again and when a run starts or ends; never on its
    own."""
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
            panel = window.panel(CrossValPanel)
            panel._set_busy(True)
            try:
                assert home.next_heading.label.text() == "Next: nothing until the run ends"
            finally:
                panel._set_busy(False)
            assert home.next_heading.label.text() == "Next: start a training"
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
        panel._set_busy(True)
        try:
            started = len(reads)
        finally:
            panel._set_busy(False)
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


def _shown_off_screen(window, width=None, height=None):
    """Show ``window`` where no screen shows it, at a size when one is handed, and let its layouts settle."""
    window.setAttribute(Qt.WA_DontShowOnScreen)
    if width is not None:
        window.resize(width, height)
    window.show()
    QApplication.processEvents()


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
    follower = style.FocusFollower(app)
    window = build_window()
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
    follower = style.FocusFollower(app)
    kept = QWidget.setFocus

    def set_focus(widget, reason=Qt.OtherFocusReason):
        kept(widget, reason)
        follower.arrived(widget, reason)

    monkeypatch.setattr(QWidget, "setFocus", set_focus)
    window = build_window()
    rail = window.shell.rail

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
        panel = inference.posterior_panel
        panel._set_busy(True)
        try:
            assert inference.step_states() == rules(live="training")
            assert spoken()[2] == "Step 3, Training. Now: running"
        finally:
            panel._set_busy(False)
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


def test_the_foot_reads_nothing_is_running_at_start_and_names_the_running_place():
    """At the window's start the foot says nothing is running, is no control and no clock is ticking. While a
    panel's run is live it names the place by its part ("Running: Effective temperature") or, for a step, by the
    step, with the time it has gone on and that settings are locked; its tool tip and its accessible name name the
    place in full, its page too. It is a button then, and the clock ticks once a second. When the run ends the
    foot says nothing is running again and the clock stops; a foot that held the keyboard hands it to the rail."""
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    window = build_window()
    foot = window.shell.rail.foot
    idle = ("none", "Nothing is running", "Only one run at a time.", False)
    assert (foot._marker, foot._title, foot._words, foot.is_button()) == idle
    assert not window._run_timer.isActive(), "a clock is ticking before anything has run"
    for panel, title, whole in ((window.panel(CrossValPanel), "Running: Effective temperature",
                                 "Running: Effective temperature, Sweep study"),
                                (window.part("inference").posterior_panel, "Running: Training",
                                 "Running: step 3, Training")):
        panel._set_busy(True)
        try:
            assert (foot._marker, foot._title, foot.is_button()) == ("now", title, True)
            # tolerant by one tick: this is real elapsed time, not a stub
            assert foot._words in ("0:00 so far. Settings are locked.", "0:01 so far. Settings are locked.")
            assert foot.toolTip() == foot.accessibleName() == f"{whole}. {foot._words}"
            assert window._run_timer.isActive() and window._run_timer.interval() == 1000
            assert window._run_timer.parent() is window
        finally:
            panel._set_busy(False)
        assert (foot._marker, foot._title, foot._words, foot.is_button()) == idle
        assert not window._run_timer.isActive(), "the clock must stop when the run does"
    panel._set_busy(True)
    try:
        foot.setFocus(Qt.TabFocusReason)
    finally:
        panel._set_busy(False)
    assert window.focusWidget() is window.shell.rail.row("home"), "the run ended and the keyboard went nowhere"


def test_the_foots_time_keeps_its_width_as_it_ticks(monkeypatch):
    """The foot is written again every second while a run is live. Its words are laid out in the small lettering
    with digits of equal width, so the time keeps its width as it ticks: 4:07 and 4:11 are one width, and so on
    by the hour. Without the feature the same font gives two widths."""
    from PySide6.QtGui import QFont, QFontMetricsF
    from core.gui import fonts
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.screens import shell
    qt_app()
    window = build_window()
    foot, panel = window.shell.rail.foot, window.panel(CrossValPanel)
    laid_out = []
    kept = shell.clock_words
    panel._set_busy(True)
    try:
        for seconds in (247, 251, 600, 671, 36000, 40271):
            monkeypatch.setattr(window, "_run_started", time.monotonic() - seconds)
            window._tick_running()
            laid_out.append(foot._words)
    finally:
        panel._set_busy(False)
    assert laid_out[:2] == ["4:07 so far. Settings are locked.", "4:11 so far. Settings are locked."]
    assert kept(36000) == "10:00:00"
    font, equal, small = fonts.inter("small", tabular=True), QFont.Tag.fromString("tnum"), fonts.inter("small")
    assert font.isFeatureSet(equal)
    assert (font.family(), font.pixelSize(), font.weight()) == (small.family(), small.pixelSize(), small.weight())
    widths = [QFontMetricsF(font).horizontalAdvance(words) for words in laid_out]
    assert widths[0] == widths[1] and widths[2] == widths[3] and widths[4] == widths[5], widths
    without = QFont(font)
    without.unsetFeature(equal)
    one, other = (QFontMetricsF(without).horizontalAdvance(words) for words in laid_out[:2])
    assert one != other, "the digits have one width without the feature as well: the lines above prove nothing"


def test_clicking_the_foot_opens_the_running_place_and_a_bare_run_host_goes_nowhere(monkeypatch):
    """While a run is live the foot is a button that goes to the run's place, its page or step too, as
    ``go_to_run`` does, a step that cannot start among them, the place taking the keyboard as after a click. A panel
    the window has no place for -- the bare run host a test builds -- is named "a task", in the foot's title, tool
    tip and accessible name; the foot is no button and the window stays where it is."""
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    given = _given(monkeypatch)
    window = build_window()
    foot = window.shell.rail.foot
    for panel, place in ((window.panel(CrossValPanel), ("temperature", "sweep")),
                         (window.part("inference").validate_panel, ("inference", "checks"))):
        window.go_to("home")
        window.part("temperature").show_page("one_cell")
        panel._set_busy(True)
        try:
            del given[:]
            QTest.mouseClick(foot, Qt.LeftButton, Qt.NoModifier, foot.rect().center())
            assert window.current_place() == place
            assert given[-1:] == [(window.focusWidget(), Qt.OtherFocusReason)], (
                f"{place}: a click on the foot gave the keyboard otherwise than as a click gives it")
        finally:
            panel._set_busy(False)
    orphan = build_run_host()
    window.go_to("live")
    orphan._set_busy(True)
    try:
        assert (foot._title, foot.is_button()) == ("Running: a task", False)
        assert foot.toolTip() == foot.accessibleName() == f"Running: a task. {foot._words}"
        QTest.mouseClick(foot, Qt.LeftButton, Qt.NoModifier, foot.rect().center())
        window.go_to_run()
        assert window.current_place() == ("live", None), "a panel with no place moved the window"
    finally:
        orphan._set_busy(False)


def test_no_row_carries_a_mark_for_a_live_run_and_nothing_is_disabled():
    """A live run is said at the foot alone: no row of the rail is marked for it or carries a mark of it, no page
    name does, and no row of Home's three tables, which Home reads in sight as the run starts; the page names keep
    their words. Nothing is disabled by it: every part and every step still opens, and Effective temperature's
    other page can still be chosen by its name."""
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

    def during(panel):
        """A run of ``panel`` live, started while Home is in sight."""
        window.go_to("home")
        panel._set_busy(True)
        try:
            assert home.next_heading.label.text() == "Next: nothing until the run ends", "Home did not read"
            assert seen() == before, "a row carries a mark of the run"
            assert [name.text() for name in names] == page_words
            for place in keys.PARTS:
                window.go_to(place.key)
                assert window.current_place()[0] == place.key and rail.row(place.key).isEnabled()
            for step in keys.STEPS:
                window.go_to("inference", step.key)
                assert window.current_place() == ("inference", step.key) and rail.row(step.key).isEnabled()
            window.go_to("temperature", "one_cell")
            _name(window.part("temperature"), "sweep").click()
            assert window.current_place() == ("temperature", "sweep") and all(name.isEnabled() for name in names)
        finally:
            panel._set_busy(False)

    try:
        _shown_off_screen(window)
        before = seen()
        for panel in (window.panel(CrossValPanel), window.part("inference").posterior_panel):
            during(panel)
    finally:
        window.hide()


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
            panel._set_busy(True)                              # a title of two lines: the foot grows
            try:
                QApplication.processEvents()
                assert rail.foot.height() > usual
                assert _in_view(rail, rail.row("narrowing")), "the foot grew over the current row"
            finally:
                panel._set_busy(False)
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
    step open while Parameter inference is open, or else of the part open, down that whole row and nowhere
    else."""
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
            assert marked == list(range(top, top + row.height())), (part, step)
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
    the memory the card holds for the program and the memory it uses, and the device check. Each stand-in records
    its call and answers as a card with room to spare would, so that the code behind it goes on as it goes on a
    computer with one, whatever an earlier test woke on the card; nothing reaches a card."""
    import subprocess
    import torch
    from core import config
    calls = []
    card = config.DeviceConfig(device=torch.device("cuda"), dtype=torch.float32, batch_size=2 ** 11)

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


def test_opening_prism_touches_no_card_and_starts_no_program(monkeypatch):
    """From the window's build to its first idle turn with Home shown, nothing asks the graphics card anything and
    no other program is started: a program run, whether there is a card, its free memory and the device check are
    each recorded (not refused by raising: the callers swallow what is raised, so a stand-in that raised would prove
    nothing), and none is recorded. Nothing ticks then either: no timer of the window or of the application runs,
    and of every start of a timer from Python, a timer with no parent among them, none still runs after the first
    idle turn, nor is a single shot of a later moment waiting."""
    from PySide6.QtCore import QObject, QTimer
    app = qt_app()
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
    window = build_window()
    try:
        _shown_off_screen(window)                                 # Home shown, then one idle turn
        assert window.current_place() == ("home", None)
        assert calls == [], f"opening PRISM asked the card or started a program: {calls}"
        ticking = [timer for timer in app.findChildren(QTimer) + window.findChildren(QTimer) if timer.isActive()]
        ticking += [timer for timer in started if not isinstance(timer, QTimer) or timer.isActive()]
        assert (ticking, waiting) == ([], []), f"a timer runs at launch: {ticking}, {waiting}"
    finally:
        window.hide()


def test_only_the_known_pages_read_the_card_when_first_opened(monkeypatch):
    """Each part opened for the first time, in the rail's order, asks the graphics card nothing and starts no
    program, but Parameter inference, whose six steps are built together at its first opening: what it asks is
    what is listed, so many times each, and nothing else."""
    from collections import Counter
    qt_app()
    calls = _card_and_programs(monkeypatch)
    window = build_window()
    found = {}
    for place in keys.PARTS:
        calls.clear()
        window.go_to(place.key)
        found[place.key] = dict(Counter(calls))
    assert found == {place.key: _READ_AT_FIRST_OPENING.get(place.key, {}) for place in keys.PARTS}, found


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
        assert window._run_place(built.fdt_panel) == ("temperature", "one_cell"), "its run would have no place"
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
    qt_app()

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
        if during_a_run:
            runner._set_busy(True)
        try:
            with monkeypatch.context() as failing_now:
                failing_now.setattr(holder, attribute, failing)
                window.go_to(name)
        finally:
            for busy in ([False] if during_a_run else []) + [True, False]:     # the run's end, and a run after it
                try:
                    runner._set_busy(busy)
                except Exception as error:                        # noqa: BLE001 -- recorded, then asserted
                    raised.append(repr(error))
        page = window.part(name)
        assert isinstance(page, PartError) and window.shell.stack.currentWidget() is page, attribute
        greyed = sorted(type(panel).__name__ for panel in BasePanel._instances if not panel.controls.isEnabled())
        assert (raised, foot.accessibleName().startswith("Nothing is running"), greyed) == ([], True, []), (
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
    import logging
    import threading
    from core.gui import main_window
    from core.gui.panels.base_panel import BasePanel
    from core.gui.panels.simulate_panel import SimulatePanel
    from core.gui.screens.shell import PartError
    app = qt_app()
    window = build_window()
    runner, gate = window.panel(SimulatePanel), threading.Event()
    shown = runner.log_pane.toPlainText

    def run():
        logging.getLogger("core").info("the run's own record")
        gate.wait(10)

    def broken():
        raise _CannotBeBuilt("the cell file could not be read")

    def turn_until(done):
        began = time.monotonic()
        while not done() and time.monotonic() - began < 10:
            app.processEvents()
            time.sleep(0.01)

    capsys.readouterr()
    runner.dispatch(run)
    try:
        turn_until(lambda: "the run's own record" in shown())
        monkeypatch.setattr(main_window, "TemperatureScreen", broken)
        window.go_to("temperature")
        assert isinstance(window.part("temperature"), PartError)
    finally:
        gate.set()
    turn_until(lambda: not BasePanel._running)
    console = capsys.readouterr().err
    assert "the run's own record" in shown() and not BasePanel._running, f"the run's own record is lost: {shown()}"
    assert [words for words in ("could not be opened", "Traceback", "_CannotBeBuilt") if words in shown()] == [], (
        f"the run's log holds the failure: {shown()}")
    assert "error: Effective temperature could not be opened. the cell file could not be read" in console, (
        f"the console did not get the failure: {console!r}")
    assert "_CannotBeBuilt: the cell file could not be read" in console, f"the console got no trace: {console!r}"


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
    runner._set_busy(True)
    try:
        assert window.built_parts() == ("home", "reduction"), "a run's start built a part"
        for name, row in (("inference", "model"), ("temperature", "temperature"), ("live", "live")):
            window.go_to("home")                                  # the keyboard on Home's main button
            window.go_to(name)
            panels = window._panels_of(name, window.part(name))
            assert [panel.controls.isEnabled() for panel in panels] == [False] * len(panels), (
                f"{name}, first opened during a run, is not locked")
            assert (window.current_place()[0], window.focusWidget()) == (name, rail.row(row)), name
        assert len(window._all_panels()) == 10 and not runner.controls.isEnabled()
    finally:
        runner._set_busy(False)
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
    runner._set_busy(True)
    try:
        assert not window.part("live").controls.isEnabled(), "a part handed during a run is not locked"
        monkeypatch.setattr(main_window, "TemperatureScreen", TemperatureScreen)
        _button(window.part("temperature"), "Try again").click()
        rebuilt = window.part("temperature")
        assert isinstance(rebuilt, TemperatureScreen)
        assert [panel.controls.isEnabled() for panel in rebuilt.panels()] == [False, False], (
            "a part built again during a run is not locked")
    finally:
        runner._set_busy(False)
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
@contextlib.contextmanager
def _in_front(widget):
    """``widget`` as the window in front, as the screen makes a window when a person brings it there. Shown where
    no screen shows it, the window is brought there by telling the toolkit so, through a call the toolkit keeps
    though it marks it as old, whose warning is not let through; the focus the window gives is then delivered with
    its reason, as the screen delivers it. At the end no window is in front."""
    def tell(front):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            QApplication.setActiveWindow(front)
    tell(widget)
    try:
        yield widget
    finally:
        tell(None)


def _press(window, sequence):
    """Press the key ``sequence`` ("Ctrl+3", "Shift+F6") where a person's key goes: at the control that holds the
    keyboard in ``window``, or at the window itself when none does."""
    combination = QKeySequence(sequence)[0]
    QTest.keyClick(window.focusWidget() or window, combination.key(), combination.keyboardModifiers())


def _given(monkeypatch):
    """Every focus given from now on, each as the widget and the reason it was given for, in their order."""
    from PySide6.QtWidgets import QWidget
    given, kept = [], QWidget.setFocus

    def set_focus(widget, *reason):
        given.append((widget, reason[0] if reason else Qt.OtherFocusReason))
        return kept(widget, *reason)

    monkeypatch.setattr(QWidget, "setFocus", set_focus)
    return given


def test_every_key_of_the_map_does_what_its_row_says(monkeypatch):
    """The window makes one shortcut for each row of the key map and no other, each in the window's own context,
    so that only the window in front hears it, and none repeating while its key is held down; each row of the rail
    names in its tool tip the keys that open its place, and Help prints the rows of the map. With the window in
    front, each part's key and each step's key, pressed where a person's key goes, shows that place and gives its
    page the keyboard as a shortcut gives it, so that the focus ring shows there; a part first opened by its key,
    and Parameter inference first opened by a step's key, is built under the wait cursor with its row of the rail
    marked already. The ring shows also where the keyboard already stood, after a click in the help text or a move
    the mouse made, neither of which shows it. The key to the run goes to the running place only while the foot is
    a button, and moves nothing once the run has ended; the key between the rail and the page takes the keyboard
    from the page to the rail and back."""
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
    follower = style.FocusFollower(app)
    monkeypatch.setattr(style, "_FOLLOWER", follower)         # the application's follower, as the window finds it
    window, other = build_window(), build_window()
    rail = window.shell.rail
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
            orphan = build_run_host()
            orphan._set_busy(True)                     # a run of a panel with no place: the foot is no button
            try:
                _press(window, "Ctrl+R")
                assert (rail.foot.is_button(), window.current_place()) == (False, ("home", None)), (
                    "the key to the run moved while the foot is no button")
            finally:
                orphan._set_busy(False)
            runner = window.panel(ReductionPanel)
            runner._set_busy(True)
            try:
                del given[:]
                _press(window, "Ctrl+R")
                assert (rail.foot.is_button(), window.current_place()) == (True, ("reduction", None)), (
                    "the key to the run did not go to the run")
                assert given[-1:] == [(rail.row("reduction"), Qt.ShortcutFocusReason)], (     # its settings locked
                    "the running place took the keyboard otherwise than as a shortcut gives it")
            finally:
                runner._set_busy(False)
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
    follower = style.FocusFollower(app)
    window = build_window()
    rail = window.shell.rail
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
            runner = window.panel(ReductionPanel)
            runner._set_busy(True)
            try:
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
                runner._set_busy(False)
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
    asked by no Escape, in the rail or on its own Cancel button."""
    from PySide6.QtWidgets import QPushButton, QToolButton
    from core.gui.panels.base_panel import BasePanel
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.streams import CancelToken
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
            runner._set_busy(True)
            try:
                runner._cancel = BasePanel._active_cancel = CancelToken()   # what a run's start hands its panel
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
                assert (BasePanel._running, runner._cancel.requested.is_set(), runner.btn_cancel.text()) == (
                    True, False, "Cancel"), "Escape in the rail asked the run to stop"
                runner.btn_cancel.setFocus(Qt.TabFocusReason)
                del said[:]
                QTest.keyClick(runner.btn_cancel, Qt.Key_Escape)
                assert (BasePanel._running, runner._cancel.requested.is_set(), runner.btn_cancel.text()) == (
                    True, False, "Cancel"), "Escape asked the run to stop"
                assert (window.current_place(), window.focusWidget(), said) == (
                    ("reduction", None), runner.btn_cancel, []), "Escape on the Cancel button moved something"
            finally:
                runner._cancel = BasePanel._active_cancel = None
                runner._set_busy(False)
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
        runner = window.panel(ReductionPanel)
        runner._set_busy(True)
        try:
            for part in ("home", "help"):
                window.go_to(part)
                assert walk(rail.row(part), Qt.Key_Tab, 2) == ["foot", "page"], (
                    f"{part}: Tab from the rail did not reach the foot, a button, and then the page")
                first = window.focusWidget()
                assert walk(first, Qt.Key_Backtab, 2) == ["foot", rail.row(part)], (
                    f"{part}: Shift+Tab from the page did not reach the foot and then the rail")
        finally:
            runner._set_busy(False)
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
    while its main thread is held for a second, as while a figure is drawn."""
    import os
    from core.gui import one_instance
    from tests._fixtures import pump
    app = qt_app()
    path = str(tmp_path / "prism.ini")
    first = one_instance.claim(path)
    heard, knock = [], one_instance._knock
    first.knocked.connect(lambda: heard.append(True))
    monkeypatch.setattr(one_instance, "_knock", lambda *told: brought_forward.append(("knock",)) or knock(*told))
    try:
        began = time.monotonic()
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
    no console has none, it hands over all the same. Handed no window, the entry point ends with the status 0,
    showing nothing and running no event loop."""
    import os
    import sys
    from core.gui import one_instance
    from tests._fixtures import process_state, pump, whole_program
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

    class Application:
        def exec(self):
            ran.append("a loop")
            return 3

    monkeypatch.setattr(entry, "build_app", lambda: (Application(), None))
    assert (entry.main(), ran) == (0, []), "the start that was handed over ran an event loop"


def test_a_start_beside_a_prism_that_never_answers_ends_with_the_status_one(tmp_path, monkeypatch, capsys):
    """The whole program started beside a living holder of the lock that never answers -- a PRISM busy past the
    start's three knocks, or one that does not listen -- builds no window and no other widget, holds no claim, says
    in one line that PRISM did not answer, and ends with the status 1."""
    import os
    from PySide6.QtCore import QLockFile
    from core.gui import one_instance, settings
    from tests._fixtures import whole_program
    qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
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
    assert (ended.value.code, len(QApplication.topLevelWidgets()), one_instance.current()) == (1, widgets, None), (
        "the start beside a holder that never answers built something, holds a claim or ended otherwise")
    assert capsys.readouterr().err.endswith("PRISM is already running but did not answer; this start ends.\n")


def test_a_second_start_says_in_one_line_what_it_did(monkeypatch):
    """A start that builds no window says what it did in one line on the error stream, the only sign a start from a
    console gives: that PRISM was open and was asked to come to the front, or that it did not answer and this start
    ends. With no error stream, as a start with no console has none, nothing is written and nothing is raised."""
    import io
    import sys
    from core.gui import app as gui_app
    for case, line in (("hand_over", "PRISM is already open: it was asked to come to the front.\n"),
                       ("give_up", "PRISM is already running but did not answer; this start ends.\n")):
        stream = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        gui_app._say(case)
        assert stream.getvalue() == line, stream.getvalue()
        monkeypatch.setattr(sys, "stderr", None)
        gui_app._say(case)


def test_a_prism_that_cannot_listen_turns_a_second_start_away_and_says_so(tmp_path, monkeypatch, caplog):
    """A PRISM that holds the lock but whose local server does not listen cannot hear a second start of itself, and
    a second start on its settings file is turned away while it is open, as its warning says, naming the server."""
    import logging
    from PySide6.QtNetwork import QLocalServer
    from core.gui import one_instance
    qt_app()
    path = str(tmp_path / "prism.ini")
    monkeypatch.setattr(QLocalServer, "listen", lambda server, name: False)
    with caplog.at_level(logging.WARNING, logger="core.gui.one_instance"):
        first = one_instance.claim(path)
    try:
        said = [record.getMessage() for record in caplog.records if record.name == "core.gui.one_instance"]
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
    the program's window to the front once its event loop runs, and not before."""
    import os
    from core.gui import one_instance, settings
    from tests._fixtures import pump, whole_program
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (_app, window):
        second = one_instance.claim(settings.settings().fileName(), tries=1, pause_ms=0)
        assert (second.case, brought_forward) == ("hand_over", [("allow_foreground", os.getpid())]), (
            "the second start was not handed over, the open PRISM let take the front first")
        pump(app, 0.2)
        assert brought_forward[1:] == [("come_to_front", window)], "the knock did not bring the window to the front"


def test_a_prism_whose_window_closed_while_a_run_goes_on_lets_the_next_start_open(tmp_path, monkeypatch):
    """A PRISM whose window is closed while a run still goes on, on the thread pool -- a run left to finish unseen --
    holds its lock and its listener while its event loop runs, and lets go of both as the loop ends, though the run
    goes on: the next start of PRISM on its settings file then takes the lock and opens a window of its own."""
    import threading
    from PySide6.QtCore import QRunnable, QThreadPool
    from core.gui import __main__ as entry
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
        monkeypatch.setattr(entry, "build_app", lambda: (Loop(), window))
        QThreadPool.globalInstance().start(Run())
        try:
            assert entry.main() == 0
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


def test_bringing_the_window_forward_brings_the_question_open_over_it_first(monkeypatch):
    """The window brought to the front, as a second start's knock asks: out of the minimised state, a maximised one
    staying maximised; raised and activated; and its taskbar button flashed only when the system keeps the front for
    another window. While a question is open over the window, it is the question that is raised and activated and
    whose hold of the front is read, so that the keyboard stays on what must be answered first. Every call that
    reaches the system is stood in for and noted: nothing is brought to the front here."""
    from PySide6.QtWidgets import QWidget
    from core.gui import main_window
    from core.gui.widgets.ask import Ask
    qt_app()
    window = build_window()
    question, did = None, []

    def note(what):
        return lambda widget, *_rest: did.append((what, "question" if widget is question else
                                                  "window" if widget is window else type(widget).__name__))

    monkeypatch.setattr(QWidget, "raise_", note("raise"))
    monkeypatch.setattr(QWidget, "activateWindow", note("activate"))
    monkeypatch.setattr(QApplication, "alert", note("alert"))
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
                    [] if holds else [("alert", "window")])
                assert (did, window.windowState()) == (wanted, Qt.WindowMaximized), (
                    f"with {'a' if asking else 'no'} question open, the front {'held' if holds else 'refused'}: "
                    f"{did}, {window.windowState()}")
    finally:
        if question is not None:
            question.reject()
            question.deleteLater()


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
        runner = window.panel(ReductionPanel)
        runner._set_busy(True)
        try:
            window.go_to("help")
            window.go_to("home")
        finally:
            runner._set_busy(False)
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
    text size; a message the question holds whole is not cut."""
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
    whole = gui_app._as_sentence(str(error))

    def height(sentence):
        question = Ask(None, "Error", sentence, "OK", marker="err", detail=whole)
        try:
            return question.sizeHint().height()
        finally:
            question.deleteLater()

    screen = QApplication.primaryScreen().availableGeometry().height() - gui_app.FRAME
    for room in (768 - 48 - gui_app.FRAME, screen):
        shown = gui_app._fitted(whole, room)
        kept = shown[:-1]
        assert (shown[-1:], whole.startswith(kept + " ")) == ("\u2026", True), f"not cut at a word: {shown[-40:]}"
        more = whole[:whole.index(" ", len(kept) + 1)] + "\u2026"         # a word more
        assert height(shown) <= room < min(height(more), height(whole)), (
            f"in {room} px the question is {height(shown)} px, and {height(more)} px with a word more")
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
