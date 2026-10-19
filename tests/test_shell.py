"""The window's places -- the rail, ``go_to`` and the run's foot -- and Home.

Home is built as the window builds it, through the builders of tests/_fixtures.py, and handed what it states by a
function each test makes: a session is a plain namespace of the five fields the window's session holds, and the
steps' states are the rules' own. The window's reading of its session, its run and the record store in use, and
what each of Home's places opens in the window, are tested on a whole window.
"""
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeySequence, QTextLayout
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from core.gui import chain, design, keys, record_words
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
    """What the window hands Home: the inference screen's session and its six steps, the part and page a live run
    is in (the step whose run it is then reads "Now"), and every record of the store in use that has a manifest,
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
        facts = window._home_facts()
        assert facts.session is window.part("inference").session and facts.live_place is None
        assert facts.states == chain.step_states(window.part("inference").session) and not facts.has_runs
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
    """The rail lists the window's nine parts in their order, each by its name, and the stack holds each part's
    own page in that order; a click on a row shows that page, and ``part`` gives the same page by the same key.
    The six steps are rows of Parameter inference, in sight only while it is open."""
    from core.gui.widgets.rail import PartRow, StepRow
    qt_app()
    window = build_window()
    rail, stack = window.shell.rail, window.shell.stack
    pages = [stack.widget(n) for n in range(stack.count())]
    assert [row.place.key for row in rail.visible_rows() if isinstance(row, PartRow)] == [
        place.key for place in keys.PARTS]
    assert [rail.row(place.key).text() for place in keys.PARTS] == [place.name for place in keys.PARTS]
    assert [window.part(place.key) for place in keys.PARTS] == pages
    for place, page in zip(keys.PARTS, pages):
        rail.row(place.key).click()
        assert stack.currentWidget() is page and window.current_place()[0] == place.key, place.key
        steps = [row for row in rail.visible_rows() if isinstance(row, StepRow)]
        assert steps == ([rail.row(step.key) for step in keys.STEPS] if place.key == "inference" else []), place.key
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
    """A part with no setting of its own gives the keyboard to nothing outside it."""
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


def test_clicking_the_foot_opens_the_running_place_and_a_bare_run_host_goes_nowhere():
    """While a run is live the foot is a button that goes to the run's place, its page or step too, as
    ``go_to_run`` does, a step that cannot start among them. A panel the window has no place for -- the bare run
    host a test builds -- is named "a task", in the foot's title, tool tip and accessible name; the foot is no
    button and the window stays where it is."""
    from core.gui.panels.crossval_panel import CrossValPanel
    qt_app()
    window = build_window()
    foot = window.shell.rail.foot
    for panel, place in ((window.panel(CrossValPanel), ("temperature", "sweep")),
                         (window.part("inference").validate_panel, ("inference", "checks"))):
        window.go_to("home")
        window.part("temperature").show_page("one_cell")
        panel._set_busy(True)
        try:
            QTest.mouseClick(foot, Qt.LeftButton, Qt.NoModifier, foot.rect().center())
            assert window.current_place() == place
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


def test_help_shows_the_help_text_that_settings_held():
    """Help shows its heading, "Help", over the help text Settings used to hold, built here now: the titles of its
    five sections, and every word of the help text, in a viewer at least as tall as the look asks."""
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
    assert [label.text() for label in window.part("help").findChildren(QLabel)] == ["Help"]
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


def test_every_place_but_home_stands_the_same_margin_from_the_rail_and_the_windows_edges():
    """Every place but Home stands its first frame or its first line twelve pixels in from the rail, from the
    window's right edge, from its top and from its bottom, at the ordinary text size, and as much more as a larger
    text size scales it: a step and a panel by the frame their own layout holds, Effective temperature's row of
    page names too, lined up with the frame under it, and every other place by what its own layout holds, the list
    of Your own models by its group too. Home keeps the look's page margin, and the window gives it none."""
    from PySide6.QtCore import QMargins, QRect
    from PySide6.QtWidgets import QGroupBox
    from tests._look_helpers import text_size
    qt_app()

    def margins(window, part):
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

    for factor in (1.0, 1.5):
        with text_size(factor):
            window = build_window()
            try:
                _shown_off_screen(window, 1300, 820)
                for place in keys.PARTS[1:]:
                    window.go_to(place.key)
                    QApplication.processEvents()
                    assert margins(window, place.key) == (design.scaled(12),) * 4, (factor, place.key)
                listed = window.part("models")
                group, = [box for box in listed.findChildren(QGroupBox) if box.title() == "User-defined models"]
                at = group.mapTo(listed, QPoint(0, 0))
                assert (at.x(), at.y(), listed.width() - at.x() - group.width()) == (design.scaled(12),) * 3, factor
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
